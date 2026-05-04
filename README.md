# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-04 | 今日论文总数: 400

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. AgentReputation: A Decentralized Agentic AI Reputation Framework

**arXiv ID:** 2605.00073 | [PDF](https://arxiv.org/pdf/2605.00073v1)

**作者:** Mohd Sameen Chishti `[一作]` (Norwegian University of Science and Technology), Jingyue Li `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 4420 | [OpenAlex ID](https://openalex.org/A5067021027)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了 AgentReputation 框架，用于去中心化 AI 市场中对代理的可信度进行证据驱动、上下文敏感且决策面向的评估；

**💡 创新点**

创新点在于将信誉分解为可量化的验证强度、按任务上下文划分的“信誉卡”，以及通过策略引擎实现对资源分配、访问控制和验证升级的动态治理；

**🔧 技术方法**

使用分层架构（功能层、信誉服务层、区块链存储层）、可插拔的验证强度等级、基于证据事件的信誉卡、决策引擎以及零知识/可信证明等技术；

**📊 数据集**

论文未在公开数据集上进行实验，主要基于假设场景和概念验证；

**📈 对比分析**

对比方法主要是与传统标量信誉系统进行理论对比，指出其在跨域混淆和验证多样性方面的不足，未给出定量性能指标；

**⚠️ 局限性**

主要限制包括缺乏验证强度的量化方法、冷启动代理的信誉建立策略、隐私与验证强度的权衡、以及对抗性演化和协同攻击的处理等未解决的问题。

---

## 2. Compliance-Aware Agentic Payments on Stablecoin Rails

**arXiv ID:** 2605.00071 | [PDF](https://arxiv.org/pdf/2605.00071v1)

**作者:** Kenneth See `[一作]` (Monetary Authority Of Singapore), Xue Wen Tan `[通讯]` (Infocomm Media Development Authority)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出一种将可编程合规嵌入稳定币支付系统的架构，支持代理支付与合规检查的无缝集成。

**💡 创新点**

创新点在于将合规评估嵌入到交易执行点，使用x402签名授权和区块链上的政策包装器，实现实时合规与自动化争议解决。

**🔧 技术方法**

技术包括x402协议、EIP-3009签名传递、基于区块链的政策包装器、源于资金和制裁筛查的智能合约、escrow 结构。

**📊 数据集**

未使用传统数据集，本文主要通过演示场景进行验证。

**📈 对比分析**

无定量比较，示例中展示立即结算与需要分层结算的两种场景，展示合规检查对交易流程的影响。

**⚠️ 局限性**

局限在于缺乏大规模实测、对复杂多方合规场景的可扩展性以及对潜在攻击的完整防御方案尚待进一步研究。

---

## 3. Selfie-Capture Dynamics as an Auxiliary Signal Against Deepfakes and Injection Attacks for Mobile Identity Verification

**arXiv ID:** 2605.00218 | [PDF](https://arxiv.org/pdf/2605.00218v1)

**作者:** Erkka Rantahalvari `[一作]` (Candour Oy), Constantino Álvarez Casado `[通讯]` (Candour Oy)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估使用自拍捕获期间的运动传感器时序数据作为辅助信号，以增强移动远程身份验证系统对深度伪造和注入攻击的检测与用户验证。

**💡 创新点**

创新点在于：①创建公开的 CanSelfie 数据集（375 条多传感器自拍序列及三类攻击代理）；②将时间序列分类和全系列异常检测方法作为基准，系统性评估其在伪造检测和用户验证中的效果。

**🔧 技术方法**

技术包括：运动传感器（加速度、陀螺仪、磁力计）数据预处理与窗口提取；七种时间序列分类器（WEASEL+MUSE、QUANT、MR‑HYDRA 等）和八种全系列异常检测器（ROCKAD、Isolation Forest、LSTM‑AE 等）。

**📊 数据集**

数据集为 CanSelfie，来自 30 名参与者、1 台 Huawei Honor 200 手机，采样率 50 Hz，包含 375 条真实自拍序列和 3 类攻击代理（静态、手持、时间移位）。

**📈 对比分析**

方法对比采用 FRR、FAR、EER 等指标。伪造检测中 ROCKAD 达 0% FRR、43.8% FAR；QUANT+3‑NN 取得最低 32% FAR。用户验证中，TSC（WEASEL+MUSE）在 9 通道双窗口下实现 1.07% EER，单轴加速度已达到约 5% EER；低-shot 一类验证效果较弱。

**⚠️ 局限性**

局限性包括：仅在单设备、单姿势、单会话下收集数据；攻击代理不包含真实注入攻击；缺乏跨设备/跨会话验证；未与面部识别融合，主要作为低摩擦辅助信号。

---

## 4. Dynamic-TD3: A Novel Algorithm for UAV Path Planning with Dynamic Obstacle Trajectory Prediction

**arXiv ID:** 2605.00059 | [PDF](https://arxiv.org/pdf/2605.00059v1)

**作者:** Wentao Chen `[一作]`, Yuanlong Yu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了Dynamic‑TD3框架，用于在高度动态、嘈杂的环境中实现无人机安全路径规划

**💡 创新点**

通过将任务奖励与安全成本解耦并采用CMDP双臂批评家与拉格朗日松弛、引入自适应轨迹演化机制ATREM捕捉远程意图、以及物理感知增强的门控Kalman滤波PAG‑KF来实现物理约束下的安全决策

**🔧 技术方法**

使用TD3强化学习、双目标批评家、拉格朗日松弛、SumTree‑PER经验回放、门控Kalman滤波、注意力编码的轨迹演化网络，以及物理约束下的CMDP建模

**📊 数据集**

主要使用自建的三维仿真平台，包含固定高度约束、静态与动态障碍物的随机分布，用于评估不同难度下的导航表现

**📈 对比分析**

与DDPG、SAC、TD3、I‑TD3、TD3_EE等基线算法在成功率、飞行时长、能耗、轨迹长度等指标上进行对比；Dynamic‑TD3在高动态场景下取得最高成功率、最低飞行时长与能耗，表现最优

**⚠️ 局限性**

仍停留在仿真环境，缺乏真实世界测试；模型结构复杂，对超参数和物理先验敏感；在极端噪声或意图不确定性极高的实际场景中可能需要进一步鲁棒性验证

---

## 5. State Stream Transformer (SST) V2: Parallel Training of Nonlinear Recurrence for Latent Space Reasoning

**arXiv ID:** 2605.00206 | [PDF](https://arxiv.org/pdf/2605.00206v1)

**作者:** Thea Aviss `[一作]` `[通讯]` (Fifth Dimension), Thea Aviss (Fifth Dimension)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 State Stream Transformer (SST) V2，给每一层的解码器增加水平递归的状态流，使模型能够在生成过程中在连续潜在空间中进行垂直深度推理和水平连续状态传递，并通过两步并行训练实现可训练的非线性跨位置递归；通过多次迭代实现增量推理，并设计了基于位置 0 潜在状态的停止探测器。

**💡 创新点**

创新点包括：1）将垂直层深推理与水平状态持续结合成一个统一机制；2）提出两步并行训练法，用关联扫描近似递归，避免跨位置顺序依赖；3）引入阶段化计算评估（staged compute）和自适应停止探测器，实现不同问题的迭代深度自我调节；4）在小规模训练集（GSM8K CodeACT）上显著提升推理性能，验证机制本身而非仅是规模或数据驱动。

**🔧 技术方法**

核心技术：非线性 FFN 递归、学习的混合系数 α、两步并行训练（pass‑1 关闭混合，pass‑2 启用混合并使用扫描结果）、多次迭代推理、基于位置 0 潜在状态的 10 维 MLP 停止探测器、实验中的 L2 变化监测、层级隐状态分析、聚类和投影等。

**📊 数据集**

使用的主要数据集：GSM8K（6,579 条示例，CodeACT 重新表述）、MATH‑500、GPQA‑Diamond（198 条题目）以及 HumanEval（164 条），训练时仅使用 GSM8K；评估时对上述四个基准进行零样本贪婪解码。

**📈 对比分析**

对比方法：对同一 LLaMA/Gemma 27B 后端做匹配基准（无 SST），在相同训练设置、数据集、超参下评估；使用 staged compute 在 i_max=4 迭代深度下与单迭代基准比较；在公开基准（GSM8K、GPQA‑Diamond、MATH‑500、HumanEval）与更大或专有模型做零样本对比。SST 在 GSM8K 上提升 46% 错误修正、GPQA‑Diamond 上 +15.15 误差点、MATH‑500 上 +6.4 误差点，并在同 27B 参数规模下优于 70B+ 开源与专有模型。

**⚠️ 局限性**

局限性：1）两步并行训练只实现迭代深度 1 的等价训练，无法在训练阶段学习自适应停止；2）对每个问题使用统一迭代深度会出现 overthinking 回归；3）缺乏对更深迭代（>4）性能的探索；4）自适应停止探测器仅在 GPQA‑Diamond 上验证，需大规模训练集验证；5）模型仅在 Gemma 27B 上验证，是否可推广至其他结构或更大规模尚未证明；6）每次迭代都需完整前向传递，计算成本高，限制了实际部署。

---

## 6. Predictive Spatio-Temporal Scene Graphs for Semi-Static Scenes

**arXiv ID:** 2605.00121 | [PDF](https://arxiv.org/pdf/2605.00121v1)

**作者:** Miguel Saavedra-Ruiz `[一作]` (Université de Montréal), Liam Paull `[通讯]` (Université de Montréal)

**通讯引用:** 4325 | [OpenAlex ID](https://openalex.org/A5037065865)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种能够在半静态场景中进行时空语义推理的预测性三维场景图，并给出了相应的持久性估计器。

**💡 创新点**

创新点在于：①用贝叶斯模型选择替代传统的启发式切换机制，实现了对出现与消失两种动态模式的概率自适应选择；②将该持久性估计器嵌入开放词汇场景图中，形成可直接查询未来状态的时间变场景图；③支持多接收器（receptacle）信念融合，能够在多位置间做概率分配与缺失检测。

**🔧 技术方法**

主要技术包括：Bayesian model selection、持久性滤波器（Perpetua/FreMEn）、切换先验（FreMEn频谱、LLM生成的分段函数）、CLIP+LLM进行对象与接收器识别、图神经网络/ConceptGraphs结构、基于场景图的多步骤推理与规划。

**📊 数据集**

使用了三类数据集：①Perpetua的半静态仿真环境（每周出现/消失模式）；②三周的实验室真实数据（双小时一次半静态物体移动）；③10个ProcThor场景用于预测与自适应导航实验。

**📈 对比分析**

与多种基线（ConceptGraphs、CG+LLM、CG+LLM+History、DualMap、FreMEn）比较，结果显示：在MAE、B-Acc、F1、导航成功率、SPL等指标上均优于基线；尤其在分布迁移与感知噪声场景下，模型保持较高的预测准确性与导航效率。

**⚠️ 局限性**

局限性包括：①边缘独立假设导致多位置联合推理成本随接收器数目呈指数增长；②对相似物体的辨识依赖感知系统，可能导致混淆；③LLM生成的先验与验证容易出现幻觉，影响地图可信度；④在大规模场景下计算与内存开销仍需进一步优化。

---

## 7. The $\textit{Silicon Society}$ Cookbook: Design Space of LLM-based Social Simulations

**arXiv ID:** 2605.00197 | [PDF](https://arxiv.org/pdf/2605.00197v1)

**作者:** Aurélien Bück-Kaeffer `[一作]` (McGill University), Jean-François Godbout `[通讯]` (Mila - Quebec Artificial Intelligence Institute)

**通讯引用:** 430 | [OpenAlex ID](https://openalex.org/A5108545982)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

系统地评估了基于LLM的硅社会模拟器设计空间，运行595个实验跑，分析不同参数（基础模型、网络拓扑、同质性、调查上下文、新闻代理、人口比例等）对模拟结果的影响。

**💡 创新点**

证明基础LLM模型是决定模拟行为的关键因素；揭示设计空间既有加法特性也有复杂交互；细调模型以社交媒体数据提升了文本风格逼真度和意见动态强度。

**🔧 技术方法**

构建自研社交网络模拟器；使用LoRA在BluePrint数据上细调四种基础模型；采用BERT分类器评估AI可检测性；通过方差解释、η²、Cohen d等统计手段比较参数效应。

**📊 数据集**

BluePrint社交媒体数据集用于LoRA细调与模拟；SimBench评估模型与人类意见的对齐；BERT训练使用真实人类对话与LLM生成的对话线程。

**📈 对比分析**

通过对比未细调与细调模型、不同基础模型、是否给出调查答案、是否存在新闻代理等，利用p≤0.001显著性检验和Cohen d效应量，发现细调模型在AI检测率下降、意见转移加速，基础模型对各指标影响最大。

**⚠️ 局限性**

参数空间极大，未覆盖所有可能的选项；未进行多重比较校正，p<0.001阈值可能仍有假阳性；基础模型与问题特定初始共识相关；单个新闻代理对动态影响微弱；与真实世界验证仍存在距离。

---

## 8. OTSS: Output-Targeted Soft Segmentation for Contextual Decision-Weight Learning

**arXiv ID:** 2605.00193 | [PDF](https://arxiv.org/pdf/2605.00193v1)

**作者:** Renjun Hu `[一作]` (University of Michigan), Hyun-Soo Ahn `[通讯]` (University of Michigan)

**通讯引用:** 1123 | [OpenAlex ID](https://openalex.org/A5004597766)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在已存在约束优化器的决策系统中，提出一种新的“上下文决策权重学习”（contextual decision-weight learning）框架，直接从历史决策及其代理输出中学习每个上下文的线性决策权重向量，并以此权重向量驱动后续的决策求解。

**💡 创新点**

创新点在于：① 将权重向量视为优化器直接可用的“决策面向”目标，区别于传统的政策学习或逆优化；② 提出输出定向软分割模型 OTSS，利用软门控与专家系数学习实现对不同上下文的加权混合；③ 在理论层面证明硬分割在存在重叠时会产生结构性近似误差，而软分割在可实现的情况下可达到参数化 $O(1/n)$ 收敛速率。

**🔧 技术方法**

技术实现主要包括：Mixture‑of‑Experts（软门控 + GLM 专家）框架、正则化的逻辑回归训练、控制实验中的数值仿真、以及对比分析的回归误差与决策后悔评估。

**📊 数据集**

数据集：① 受控合成基准（两专家、匹配硬/软对照）；② 真实零售数据集 Complete Journey（来自 dunnhumby，包含真实家庭特征与早餐包组合的决策空间），用以验证方法在实际场景中的表现。

**📈 对比分析**

比较方法：对齐了 7 种基线（聚合、线性/低秩/MLP 上下文基线、聚类‑先拟合、EM 软混合回归、匹配硬分割）。实验显示：OTSS 在重叠基准上实现最低均值后悔与 MSE；在硬/软匹配基准中与硬分割相当或略优；在 Complete Journey 实验中同样获得最低后悔，并相较于低秩/EM 等基线实现显著提升。速度方面，OTSS 在实验中比 EM 快约 100‑230 倍。

**⚠️ 局限性**

局限性：① 依赖代理输出符合逻辑回归（或类似 GLM）模型的假设；② 对日志数据的覆盖程度与质量要求较高，若暴露的决策因素不充分，估计可能受限；③ 目前在实验中仅考虑线性决策权重，未探索更高维或非线性决策因素；④ 对门控/专家网络的超参数敏感度未做深入探讨；⑤ 在极大规模真实部署中的计算与调优需求仍待验证。

---

## 9. What is (H)CI: Why Does the "Human'' Matter?

**arXiv ID:** 2605.00109 | [PDF](https://arxiv.org/pdf/2605.00109v1)

**作者:** Sejal Agarwal `[一作]` (University of Waterloo), Anthony Maocheia-Ricci `[通讯]` (University of Waterloo)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5120026497)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一个工作坊，讨论在生成式AI时代下，人机交互的定义及人类因素的重要性。

**💡 创新点**

以跨学科讨论和未来愿景为创新点，强调在HCI中重新聚焦人类角色。

**🔧 技术方法**

采用引导讨论、集体头脑风暴与反思等方法进行知识共享。

**📊 数据集**

未使用具体数据集，主要基于理论与案例回顾。

**📈 对比分析**

没有实验对比，评估通过参与者反馈和讨论深度。

**⚠️ 局限性**

局限在于缺乏量化评估与实证数据，难以验证提议的有效性。

---

## 10. FedACT: Concurrent Federated Intelligence across Heterogeneous Data Sources

**arXiv ID:** 2605.00011 | [PDF](https://arxiv.org/pdf/2605.00011v1)

**作者:** Md Sirajul Islam `[一作]` (University of Louisiana at Lafayette), Klara Nahrstedt `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FedACT，针对多任务联邦学习中的设备异构问题，设计了一种基于资源匹配与参与公平性的动态设备调度框架。

**💡 创新点**

创新点在于引入对设备资源与任务需求的对齐得分（Resource Alignment）和参与公平性（Participation Fairness）共同决定调度优先级，从而兼顾训练效率与模型精度，并通过动态更新实现异构环境下的最优设备分配。

**🔧 技术方法**

核心技术包括：联邦学习（FedAvg）框架、对齐得分计算（加权点积与公平性惩罚）、基于得分的设备排序与分配、参与频率统计与公平性调节、异步训练的未来扩展方案。

**📊 数据集**

实验使用了五个基准数据集（CIFAR‑10、MNIST、EMNIST‑Letters、EMNIST‑Digits、Fashion‑MNIST）与五个模型（LeNet、CNN、VGG‑16、ResNet‑18、AlexNet）进行 IID 与 Non‑IID 两种数据分布的多任务联邦学习。

**📈 对比分析**

与随机、贪婪、遗传算法、MJ‑FL 等现有多任务调度方法以及单任务 FedAvg 进行对比，FedACT 在 IID 与 Non‑IID 场景下平均作业完成时间缩短最多 8.3 倍，模型精度提升最多 44.5%，在大部分任务上均实现显著性能提升。

**⚠️ 局限性**

局限性包括：目前仅支持同一设备每轮只能参与一个任务；调度策略对超参数 α、β 的选择敏感；并未充分验证在极大规模设备集群或极端异构环境下的可扩展性与稳定性。

---

## 11. VkSplat: High-Performance 3DGS Training in Vulkan Compute

**arXiv ID:** 2605.00219 | [PDF](https://arxiv.org/pdf/2605.00219v1)

**作者:** Jingxiang Chen `[一作]`, Yang Liu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发并评估了一种基于Vulkan的高效点云渲染框架VkSplat，用于NeRF场景重建，并比较了默认与MCMC 1M稠密化策略。

**💡 创新点**

通过引入Vulkan多线程索引、键生成与排序流水线、双VRAM度量以及预分配缓冲来避免OOM，显著提升RTX 3090与AMD GPU上的训练效率。

**🔧 技术方法**

采用Vulkan API、点云高斯分布渲染、MCMC稠密化、PyTorch缓存分配、GPU异步数据传输，以及PSNR/SSIM/LPIPS等质量评估。

**📊 数据集**

使用公开的Mip-NeRF 360数据集中的7个场景（bicycle, garden, stump, bonsai, counter, kitchen, room）。

**📈 对比分析**

与GSplat做基准对比，使用PSNR/SSIM/LPIPS、VRAM占用与总训练时间评估；在RTX 3090上VkSplat速度更快、VRAM更低，AMD上图像复制成为瓶颈，但MCMC 1M稠密化在两类GPU均能显著加速。

**⚠️ 局限性**

受限于PCIe带宽导致AMD图像复制慢、Vulkan运行时预留VRAM无法完全捕获、评估样本率低导致实际OOM略高，且仅在Mip-NeRF 360上测试，未验证更大规模或不同场景的鲁棒性。

---

## 12. Attention Is Where You Attack

**arXiv ID:** 2605.00236 | [PDF](https://arxiv.org/pdf/2605.00236v1)

**作者:** Aviral Srivastava `[一作]` (Amazon), Sourav Panda `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种白盒 jailbreak——Attention Redistribution Attack，通过识别安全关键注意力头并生成非语义攻击令牌，重新分配注意力从而绕过 LLM 的安全对齐机制。

**💡 创新点**

创新点在于：①从注意力几何层面入手，首次通过 Safety Attention Score 定位安全关键头；②利用 Gumbel‑softmax 进行梯度优化，生成对安全注意力具有高键‑查询亲和力的低语义攻击令牌；③发现头消除与注意力重定向的显著差异，表明安全行为存在于注意力路由而非单一头的残差贡献；④对三大模型架构绘制安全头分布图，揭示安全注意力集中或分散对攻击易受影响程度的影响。

**🔧 技术方法**

使用了梯度优化与 Gumbel‑softmax 进行离散令牌生成、Softmax 注意力分析、Safety Attention Score 计算、头消除实验、注意力重定向评估、对抗令牌非语义分析等技术。

**📊 数据集**

主要使用 HarmBench 的 200 条有害提示（标准+情境）进行验证；对 10 条有害提示进行发现阶段；并用系统提示和用户查询构建实验输入。

**📈 对比分析**

通过对比攻击前后拒绝率（ASR）来评估效果。攻击后，Mistral‑7B 的 ASR 为 36%（72/200），LLaMA‑3‑8B 为 30%（60/200），Gemma‑2‑9B 仅 1%（2/200）。相比之下，对安全头进行消除的结果几乎无变化（最多 1% 翻转）。实验采用 95% 置信区间和 500 步梯度迭代，证明攻击在少量令牌下即可达到显著成功率。

**⚠️ 局限性**

局限性包括：仅在 7‑9B 开源模型且需白盒访问；仅测试三种架构，难以推广到更大规模模型；攻击需要每个模型单独优化，计算成本高；安全头识别与验证使用相同提示集，存在一定数据泄露；拒绝判定采用关键词匹配，可能导致评估偏差。

---

## 13. GAFSV-Net: A Vision Framework for Online Signature Verification

**arXiv ID:** 2605.00120 | [PDF](https://arxiv.org/pdf/2605.00120v1)

**作者:** Himanshu Singhal `[一作]` (Indian Institute of Technology), Suresh Sundaram `[通讯]` (Indian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了 GAFSV-Net，使用双分支 ConvNeXt 对在线签名的三条 kinematic 通道编码为六通道 GASF+GADF 图像，并在此基础上完成签名验证。

**💡 创新点**

创新点在于：①将三条速度、压力导数、方向角分别映射为 GASF 与 GADF 图像，显式捕获时间步间的共现与方向转移；②引入双向交叉注意力，使两分支互相查询并融合信息；③在训练中结合半硬三元组损失、硬负样本注入和均匀正则化，实现写者无关的高效度量学习。

**🔧 技术方法**

使用了 Gramian Angular Field (GASF/GADF)、ConvNeXt‑Tiny 预训练 2D 视觉骨干、双向交叉注意力、semi‑hard triplet loss、uniformity regularizer、ImageNet 预训练权重、硬负样本注入等技术。

**📊 数据集**

采用了公开的 DeepSignDB 和 BiosecurID 两大在线签名数据集进行训练与评估。

**📈 对比分析**

在相同训练目标下，针对 BiGRU、BiLSTM、Transformer、1D‑CNN、TCN、MOMENT 等 1D 序列基线进行对照，GAFSV-Net 在 DeepSignDB（R_enroll=4）EER 为 6.01%，在 BiosecurID 为 2.97%，均显著优于所有基线和传统 DTW 方法。

**⚠️ 局限性**

局限性包括：在多工作环境和高噪声条件下仍出现 EER 上升；GAF 预处理耗时 O(M²)；训练仍需真实伪造样本，未探索纯自监督或单类学习方案。

---

## 14. The speed of convergence in greedy Galois games

**arXiv ID:** 2605.00194 | [PDF](https://arxiv.org/pdf/2605.00194v1)

**作者:** Jeffrey Shallit `[一作]` (University of Waterloo), Jeffrey Shallit `[通讯]` (University of Waterloo)

**通讯引用:** 6433 | [OpenAlex ID](https://openalex.org/A5065322269)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文研究了贪婪加洛伊斯游戏中射击序列与 Thue‑Morse 序列的收敛速度。

**💡 创新点**

创新地将收敛长度与特定代数数值 α、β 关联，得到完全解析的阈值区间。

**🔧 技术方法**

采用组合数论、递归式与符号检验方法证明收敛长度的闭式表达。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

不涉及实验比较，证明结果表明在 q→1 时收敛长度呈指数增长。

**⚠️ 局限性**

仅在 0<q<1 范围内成立，且对极限 p→0 的更精细性质仍未完全探究。

---

## 15. Cultural Benchmarking of LLMs in Standard and Dialectal Arabic Dialogues

**arXiv ID:** 2605.00119 | [PDF](https://arxiv.org/pdf/2605.00119v1)

**作者:** Muhammad Dehan Al Kautsar `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fajri Koto `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1217 | [OpenAlex ID](https://openalex.org/A5065822589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 ArabCulture-Dialogue 数据集，并基于此设计了多轮对话的文化推理、方言互译和方言引导生成三项任务，评估 LLM 的跨方言文化推理与生成能力。

**💡 创新点**

首创多国、跨方言的对话式文化基准，首次在 13 个阿拉伯国家与其本土方言中结合多轮对话与选择题、翻译和生成三种评测方式，填补了现有单轮、单方言评测的空白。

**🔧 技术方法**

采用人机协同构建流程：GPT‑4o 起草、人工校对与方言本地化、风格后编辑，并通过零样本推理与多模型（本土、跨语种、专有）进行评测；使用 ALDi、BLEU、BERTScore、LLM‑as‑Judge 及 GlotLID 等指标。

**📊 数据集**

使用 ArabCulture‑Dialogue（来源于 ArabCulture 基准），覆盖 12 个日常主题、54 个子主题、13 个国家，包含 3,471 条 MSA 对话与对应方言，合计 343,804 词。

**📈 对比分析**

在零样本设置下对比 Arabic‑centric、Multilingual 与 Proprietary LLM；Arabic‑centric LLM 在多轮文化推理与方言翻译上优于跨语种模型；GPT‑5、Gemini‑2.5‑Pro 在选择题表现优异，但在方言翻译与方言引导生成上仍低于 50% 的准确率，开源 7B 模型更接近随机。

**⚠️ 局限性**

仅覆盖 13/22 个阿拉伯国家，未能涵盖所有方言内部变异；方言翻译可能存在“翻译痕迹”，且多方言样本有限，导致模型在细粒度方言识别与生成上仍显不足。

---

## 16. Symbolic Execution Meets Multi-LLM Orchestration: Detecting Memory Vulnerabilities in Incomplete Rust CVE Snippets

**arXiv ID:** 2605.00034 | [PDF](https://arxiv.org/pdf/2605.00034v1)

**作者:** Zeyad Abdelrazek `[一作]` (Texas A&M University--San Antonio), Young Lee `[通讯]` (Texas A&M University--San Antonio)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个4‑Agent多LLM与KLEE符号执行相结合的流水线，用于自动化检测不完整Rust CVE代码片段中的内存漏洞。

**💡 创新点**

创新点在于：①利用LLM生成可编译的FFI包装器以突破不完整代码障碍；②设计四个角色专责的多Agent架构以提升生成质量和检测覆盖；③构建图数据库以实现跨CVE的结构化查询。

**🔧 技术方法**

采用了多种LLM（GPT‑4 Turbo、Claude Opus、Claude Sonnet、GPT‑4o‑mini）完成规划、风险评估、包装器生成和KLEE参数优化，并使用KLEE进行符号执行与graph_klee.py构建图数据库。

**📊 数据集**

使用HALURust子集中的31个内存安全Rust CVE文件，涵盖11个CWE类别，作为实验数据集。

**📈 对比分析**

通过与Kani、Prusti、Creusot、Haybale（0%编译）、Clippy（35.5%检测）和Miri（泛型警告）等基线工具对比，单Agent 58%编译率、51.6%检测率，4-Agent实现90.3%编译率、83.9%检测率，错误数从487提升至1,206，性能显著优于传统方法。

**⚠️ 局限性**

局限性包括：生成的FFI包装器是对漏洞类的近似，导致对生命周期相关或需要完整上下文的漏洞（如CWE‑416、CWE‑190）检测不足；LLM结果的不确定性和模型版本变更可能影响可复现性；实验数据集仅为31个内存安全CVE，未覆盖所有Rust CVE类型。

---

## 17. From Images2Mesh: A 3D Surface Reconstruction Pipeline for Non-Cooperative Space Objects

**arXiv ID:** 2605.00147 | [PDF](https://arxiv.org/pdf/2605.00147v1)

**作者:** Bala Prenith Reddy Gopu `[一作]` (Florida Institute of Technology), Christopher McKenna `[通讯]` (Creare LLC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `51c0528b-f690-4182-ae60-bb5f046c276c` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了一套五阶段神经隐式表面重建管线，能够在无协同航天器单目检查影像上生成高保真3D网格，而无需地面真值相机姿态或参考CAD模型。

**💡 创新点**

创新点包括：① 通过SAM背景去除实现相机姿态估计的可靠性；② 将PPISP光度校正集成到隐式重建后处理，显著提升阴影区域纹理质量；③ 在真实在轨影像上首次完成完整高分辨率网格重建，验证了该方法的实用性。

**🔧 技术方法**

采用了SAM（Segment Anything Model）进行前景分割，COLMAP进行结构光SLAM姿态估计，Neuralangelo（隐式SDF+多分辨率哈希编码）实现表面重建，并使用PPISP对光照不一致进行后处理。

**📊 数据集**

使用的数据集为NASA STS‑119任务的ISS巡视视频（约10k帧）以及Astroscale ADRAS‑J任务的H‑IIA上级段巡视视频（约840帧）。

**📈 对比分析**

与未使用PPISP的基线对比，采用重投影误差、PSNR和主观可视化评估；实验表明在ADRAS‑J数据上PPISP显著提升阴影区域纹理，ST‑119数据上虽然改进有限，但能抑制过曝；相机姿态重投影误差均低于1像素，说明姿态估计准确。

**⚠️ 局限性**

局限性包括：缺乏真实CAD模型仅能定性评估；分割误差会在后续阶段传递；PPISP假设光照一致，强阴影或复杂结构仍难恢复；目前仅验证两套数据，未覆盖极端光照或镜头散射等更具挑战性的场景。

---

## 18. World Model for Robot Learning: A Comprehensive Survey

**arXiv ID:** 2605.00080 | [PDF](https://arxiv.org/pdf/2605.00080v1)

**作者:** Bohan Hou `[一作]` (Nanyang Technological University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**通讯引用:** 7162 | [OpenAlex ID](https://openalex.org/A5005666034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述了“世界模型”（World Model）在机器人学习中的演进与应用，重点聚焦其在预测、规划、仿真、评估及视频生成等方面的作用。

**💡 创新点**

提出以机器人决策为中心的世界模型定义，强调预测结构对动作生成的辅助作用；划分了四大功能范式（预测-先行、单体骨干、专家混合、统一VLA、潜在空间）并对其在机器人学习、仿真、导航、自动驾驶等任务中的具体实现进行了细致梳理。

**🔧 技术方法**

综合运用了视频生成、扩散模型、流匹配、稀疏注意力、多专家/混合专家(MoE/MoT)、自监督视频预训练、逆动力学与多模态Transformer等技术手段来构建或融合世界模型；同时结合了任务级语言、动作编码以及结构化几何/交互信息来提升可控性与物理一致性。

**📊 数据集**

参考了大量公开机器人数据集与视频数据集（如OpenAI Robotics, Meta Hand, MetaRL, MetaWorld, RoboSuite, MetaHand, MetaDrive 等）以及基于这些数据集构建的模拟与真实环境 benchmark（RBench, EWMBench, DreamGen Bench, WorldArena, WorldEval 等）。

**📈 对比分析**

与传统单一动作生成策略相比，世界模型在多任务、长时序推理与数据增强方面表现出显著优势。实证表明，集成世界模型的策略在多目标操作、导航与驾驶任务中往往获得更高成功率、降低样本需求、提升鲁棒性；在评估指标上，能够更好地保持策略间相对优劣顺序，减少模拟误差导致的性能下降。

**⚠️ 局限性**

主要限制包括：1）模型生成的未来易受累积误差影响，导致长时序控制失稳；2）在复杂物理交互（高频接触、摩擦等）场景下，现有模型的物理一致性仍有限；3）大规模视频预训练模型计算成本高，部署受限；4）评估标准仍缺乏统一、系统化的度量，导致跨方法比较困难。

---

## 19. Hyperspherical Forward-Forward with Prototypical Representations

**arXiv ID:** 2605.00082 | [PDF](https://arxiv.org/pdf/2605.00082v1)

**作者:** Shalini Sarode `[一作]` (German Research Center for Artificial Intelligence), Andreas Dengel `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Hyperspherical Forward-Forward (HFF) 算法，改进 FF 以实现单通前向推理并提升精度

**💡 创新点**

将层级本地目标从二元好坏判别转化为超球面多分类任务，使用学习的单位向量原型实现直接类别预测

**🔧 技术方法**

超球面投影、LogSumExp 原型匹配、平滑边距损失、局部前向学习、卷积扩展与辅助卷积

**📊 数据集**

MNIST、FashionMNIST、CIFAR‑10、CIFAR‑100、ImageNet‑1K

**📈 对比分析**

与多种无全局反向传播的局部学习方法（FF、FFCL、SymBa、MF、SFF 等）进行对比，HFF 在 CIFAR‑10/CIFAR‑100 上分别达 83.08%/54.34% 的准确率，ImageNet‑1K 上实现 25.7% Top‑1，推理速度提升 40×，训练收敛速度仅 1.7× BP

**⚠️ 局限性**

需要手工设定每类原型数量，导致额外的参数与内存开销；在大规模模型与复杂任务上仍落后 BP，AuxConv 可能产生信息瓶颈，未实现自适应原型数

---

## 20. Human-in-the-Loop Meta Bayesian Optimization for Fusion Energy and Scientific Applications

**arXiv ID:** 2605.00068 | [PDF](https://arxiv.org/pdf/2605.00068v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 21. SiriusHelper: An LLM Agent-Based Operations Assistant for Big Data Platforms

**arXiv ID:** 2605.00043 | [PDF](https://arxiv.org/pdf/2605.00043v1)

**作者:** Yu Shen `[一作]` (Tencent Inc), Bin Cui `[通讯]` (Peking University)

**通讯引用:** 13400 | [OpenAlex ID](https://openalex.org/A5062357883)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一款面向大数据平台的统一式智能助手，能够识别用户意图，自动路由到相应的专业故障排查或通用咨询流程，并通过多轮检索与推理提供可操作的答案。

**💡 创新点**

创新点在于：①引入 DeepSearch 迭代检索框架与“计划-检索-过滤”循环；②构建四层优先级知识库（SOP、内部知识、外部知识、模型知识），实现多跳检索而不产生上下文溢出；③通过自动化票单理解与 SOP 提炼，减少人工维护成本并持续更新知识库。

**🔧 技术方法**

主要技术包括：大型语言模型（Qwen3‑30B、DeepSeek）、检索增强生成（RAG）、多轮 DeepSearch Agent、意图分类/澄清、工具调用、知识库检索与过滤、SOP 自动生成与审核。

**📊 数据集**

使用了腾讯大数据平台真实工单与咨询记录构成的内部数据集，共约 100 个困难排查案例和数千条工单；SOP 知识库包含 679 条 SOP 与 538 条内部知识。

**📈 对比分析**

实验与在线部署对比 CoT、RAG、Vanilla DeepSearch 三个基线，结果显示该系统在准确率（73%）与实用性（81%）上优于基线，平均推理时长约 24.2 秒；在线上部署后，工单升级率下降 20.8%。

**⚠️ 局限性**

局限性包括：①仍依赖 LLM 产生的潜在幻觉；②知识库需要持续人工审核与维护；③外部检索依赖公开 API，可能受限于可访问性；④系统在极端稀缺信息场景下的鲁棒性待提升。

---

## 22. Bayesian Optimization in Linear Time

**arXiv ID:** 2605.00237 | [PDF](https://arxiv.org/pdf/2605.00237v1)

**作者:** Jesse Schneider `[一作]` (University of British Columbia), William J. Welch `[通讯]` (University of British Columbia)

**通讯引用:** 52334 | [OpenAlex ID](https://openalex.org/A5043650697)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于递归二叉分区的贝叶斯优化方法，使模型训练与搜索过程从传统的三次复杂度降低到线性复杂度，并通过局部GP模型与全局信息结合，兼顾局部精细建模与全局搜索；

**💡 创新点**

创新点在于将搜索空间递归划分为二叉树，给每个叶子节点训练独立的高斯过程模型并使用SVM决策边界动态划分子区域，同时对获取函数进行边界惩罚，实现局部与全局建模的平衡，显著解决了标准贝叶斯优化的计算瓶颈和全局建模不当问题；

**🔧 技术方法**

使用高斯过程（GP）、期望改进（EI）获取函数、支持向量机（Gaussian kernel）进行划分、PAM聚类、拉丁超立方采样生成起始点、粒子群优化（PSO）求解EI最大化等技术；

**📊 数据集**

在七个经典黑盒测试函数（Ackley、Hartmann、Levy、Michalewicz、Rastrigin、Schwefel）以及124维GM汽车质量最小化（MOPTA08）问题上进行实验；

**📈 对比分析**

与DiceOptim做配对实验，使用相同初始点和评估预算；结果显示在所有测试中平均最小值均优于DiceOptim，分布图表明后期性能明显更好，运行时间从几小时到14天大幅下降，线性复杂度使模型训练速度提升；

**⚠️ 局限性**

主要限制包括：分区阈值θ的设置影响树深度与性能；节点划分可能失败导致分类器或GP拟合错误；子区域起始点生成难度随维度增加；α函数在子区域边界不连续可能导致优化失败；以及对SVM分类器和PAM聚类参数的依赖，需要进一步调优和鲁棒性研究。

---

## 23. Ambient Persuasion in a Deployed AI Agent: Unauthorized Escalation Following Routine Non-Adversarial Content Exposure

**arXiv ID:** 2605.00055 | [PDF](https://arxiv.org/pdf/2605.00055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 24. Electrotactile Improves Thermal Referral

**arXiv ID:** 2605.00240 | [PDF](https://arxiv.org/pdf/2605.00240v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 25. Estimating LLM Grading Ability and Response Difficulty in Automatic Short Answer Grading via Item Response Theory

**arXiv ID:** 2605.00238 | [PDF](https://arxiv.org/pdf/2605.00238v1)

**作者:** Longwei Cong `[一作]` (DIPF Leibniz Institute for Research and Information in Education), Ulf Kroehne `[通讯]` (DIPF Leibniz Institute for Research and Information in Education)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

引入基于项目反应理论（IRT）的框架，对大语言模型进行自动短答评分（ASAG）的细粒度评估；

**💡 创新点**

通过IRT将评分者能力和回答难度解耦，揭示不同模型在不同难度层级的鲁棒性差异和错误模式，提供比宏观指标更深入的性能洞察；

**🔧 技术方法**

采用1PL IRT加测试项（testlet）模型估计模型能力与回答难度，结合参数恢复与拆分半验证、语义/语言特征相关性分析及混淆矩阵可视化；

**📊 数据集**

使用SciEntsBank和Beetle两个五级标签的科学问答评测数据集；

**📈 对比分析**

按IRT难度分箱评估各模型在各难度层级的准确率，发现总体相近的模型在高难度区表现差异显著；Gemma-3-12B最稳健，OpenChat-3.5在高难度时保持较好性能；

**⚠️ 局限性**

局限于仅17个LLM、仅两数据集与五级标签，且难度与特征相关性为相关而非因果，未在更广泛领域验证；

---

## 26. Shooting Neutrons at Neurons: Radiation Testing of a Spiking Neural Network on Flash-Based FPGAs

**arXiv ID:** 2605.00030 | [PDF](https://arxiv.org/pdf/2605.00030v1)

**作者:** Wim Nijsink `[一作]` (University of Twente), Marco Ottavi `[通讯]` (University of Twente)

**通讯引用:** 2554 | [OpenAlex ID](https://openalex.org/A5048232172)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在闪存 FPGA（Microchip PolarFire SoC）上实现 ODIN 神经形态核心，并通过 ChipIR 设施的高能中子束实验测量单事件翻转（SEU）交叉截面，结合在线 Spike-Dependent Synaptic Plasticity（SDSP）学习，评估中子辐射对 MNIST 分类准确率的影响。

**💡 创新点**

创新点在于：①针对具有在线学习的神经形态处理器提出专门的辐射测试框架；②利用真实中子束实验提取突触记忆的 SEU 交叉截面，并将其作为校准的故障注入率；③证明在线学习可显著延长失效时间，且在硬件开销上优于传统的三模冗余（TMR）。

**🔧 技术方法**

使用的技术包括：Microchip PolarFire SoC FPGA（闪存版）、ODIN 处理器及其 SDSP 模块、ChipIR 中子束、UART 远程控制与 RadHelper 监控、对控制逻辑实施 TMR 以保障通信安全。

**📊 数据集**

使用 MNIST 数据集（16×16 像素的尖峰编码图像）作为推理与在线学习的测试集。

**📈 对比分析**

通过比较 inference-only 与 online-learning 两种配置，在中子束实验中实时记录分类准确率随时间的变化；随后用校准后的 SEU 率进行长周期的故障注入仿真，评估准确率衰减速率。结果显示，在线学习配置在相同辐射率下能够将准确率保持在更高水平，硬件开销仅为 21% 的 LUT 增加、11% 的 FF 增加，远低于采用 TMR 的 3 倍内存开销。

**⚠️ 局限性**

局限性包括：①实验仅覆盖 ODIN 子集（10 个激活神经元，约 3.9% 的突触阵列）；②故障模型主要考虑突触记忆 SEU，未涵盖其他可能导致的“突发”准确率崩溃；③仅使用 MNIST，其他时间序列或控制任务对辐射敏感性可能不同；④中子束实验时间受限，长周期测试数据不足。

---

## 27. I can't recognize (yet): Delayed Rendering to Defeat Visual Phishing Detectors

**arXiv ID:** 2605.00183 | [PDF](https://arxiv.org/pdf/2605.00183v1)

**作者:** Ying Yuan `[一作]` (Sapienza University of Rome), Luigi Vincenzo Mancini `[通讯]` (Sapienza University of Rome)

**通讯引用:** 5792 | [OpenAlex ID](https://openalex.org/A5046905749)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并验证了一种基于渲染延迟的视觉钓鱼网站逃避攻击，并在真实用户与现有视觉检测系统上进行实验，展示了该攻击的有效性；随后给出了一种不依赖远程服务的浏览器扩展来进行防御。

**💡 创新点**

创新点在于发现并利用视觉钓鱼检测器在获取页面快照时对渲染时序的依赖性，通过延迟关键元素（如 logo）渲染来误导相似度计算，从而使攻击者能够完全规避现有检测方法；并首次系统评估了此类时间攻击对人类判断和机器检测的影响。

**🔧 技术方法**

主要技术包括：1) 通过 JavaScript 控制页面元素渲染时序，实现对视觉快照的“延迟”；2) 使用现有视觉相似度算法（如 CNN 视觉特征匹配）对快照进行检测；3) 构造“幕帘”式遮挡（curtain effect）等易实现的对抗扰动；4) 开发无服务器调用的浏览器扩展，实时检测页面是否可能为钓鱼网站。

**📊 数据集**

实验数据集采用公开的钓鱼网页与合法网站的网页快照集合（来源于 PhishTank、OpenPhish 等公开数据集），并在此基础上生成对抗样本（加入渲染延迟与遮挡）。

**📈 对比分析**

实验对比：在对 10+ 组主流视觉钓鱼检测器（基于 CNN、SVM 等）进行测试时，攻击前检测率均为 100%，攻击后全部下降至 0%；在人类实验中，受试者在识别加入遮挡的页面时的准确率不显著高于随机猜测（p<0.05），说明攻击对人类也同样有效。

**⚠️ 局限性**

局限性包括：1) 仅针对使用静态快照进行视觉比较的检测器；2) 对于实时渲染监测或多时序特征的检测方法的有效性尚未评估；3) 渲染延迟攻击可能在高性能网络环境下被检测到，需进一步研究更细粒度的时序攻击；4) 提出的浏览器扩展仅在客户端侧警示，未能直接阻止钓鱼内容。

---

## 28. Persona-Grounded Safety Evaluation of AI Companions in Multi-Turn Conversations

**arXiv ID:** 2605.00227 | [PDF](https://arxiv.org/pdf/2605.00227v1)

**作者:** Prerna Juneja `[一作]` (Seattle University), Lika Lomidze `[通讯]` (Seattle University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一个基于角色化的端到端框架，用来在多轮对话中对 AI 伴侣应用的安全性进行大规模评估，包括角色构建、情境生成、对话模拟与润色（PACE）以及危害检测；

**💡 创新点**

首创将临床验证的角色模型、情境驱动的对话模拟与自动化 LLM‑驱动的危害标注相结合，形成可扩展的多轮安全评估流程，并通过 PACE 保持角色一致性，显著提高评估细粒度；

**🔧 技术方法**

使用多种大型语言模型（ChatGPT/GPT‑4o、GPT‑5、Gemini、Mistral）、GoEmotions 进行情感分析、Pace 对话精炼器、LLM‑驱动的语句分类与危害检测、Selenium 自动化对话收集等技术；

**📊 数据集**

收集了 1,674 组 Replika 角色-对话对，涵盖 9 个高危人群角色与 25 个情境；复制实验中 1,586 组 Character.ai 对话；并对角色进行心理测评（BDI‑II、GAD‑7 等）以验证角色真实性；

**📈 对比分析**

对 Replika 与 Character.ai 在危害率、情感覆盖度等维度进行对比；Replika 的整体有害回应率为 15.2%，在饮食失调场景高达 62.5%；Character.ai 的总体有害率为 35.7%，与 Replika 的危害模式相似，主要集中在支持性镜像回应上；

**⚠️ 局限性**

角色为人工合成，难以完全再现真实用户多样性；测试的角色与情境有限，覆盖面不完整；危害标注仅为二元标签，缺乏严重程度细分；仅在单一平台的单次快照下评估，未来版本可能产生差异；

---

## 29. Confidence Estimation in Automatic Short Answer Grading with LLMs

**arXiv ID:** 2605.00200 | [PDF](https://arxiv.org/pdf/2605.00200v1)

**作者:** Longwei Cong `[一作]` (DIPF Leibniz Institute for Research and Information in Education), Ulf Kroehne `[通讯]` (DIPF Leibniz Institute for Research and Information in Education)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何在使用大型语言模型（LLMs）进行自动短答题评分（ASAG）时估计置信度，并提出了将模型自身置信度与数据集本身的不确定性相结合的混合置信度框架。

**💡 创新点**

创新点在于首次系统比较了三种基于LLM的置信度估计方法（verbalizing、latent、consistency），并通过聚类语义嵌入量化数据集的“混沌度”（semantic heterogeneity）来显式建模数据源的不确定性（aleatoric uncertainty），从而得到比单一来源更可靠、可解释的置信度。

**🔧 技术方法**

使用了 gpt‑oss‑20b 等开源LLM，结合三种置信度提取技术，采用所有‑MiniLM‑L6‑v2 句向量模型做嵌入，使用层次聚类与熵衡量混沌度，并用随机森林+Platt 归一化构建混合置信度。

**📊 数据集**

在 SciEntsBank 数据集上进行实验，包含 11,000 条学生回答，最终在 4,562 条评分样本中评估置信度。

**📈 对比分析**

与单一置信度来源相比，混合置信度在 ROC 与 ARC 上均取得更高 AUROC 与 AUARC；在校准方面，混合模型（包含 aleatoric 特征）在 Brier、ECE、MCE 上表现最佳（Brier 0.138、ECE 0.044、MCE 0.100），显著优于 verbalis、latent、consistency 等单源方法。

**⚠️ 局限性**

局限性包括：仅评估了一种开源 LLM，未验证不同模型或规模的通用性；混沌度的熵指标只是对数据不确定性的近似，可能无法捕捉所有教育层面的歧义；未来需要在更大多样的数据集与真实教育工作流程中验证效果。

---

## 30. E$^2$DT: Efficient and Effective Decision Transformer with Experience-Aware Sampling for Robotic Manipulation

**arXiv ID:** 2605.00159 | [PDF](https://arxiv.org/pdf/2605.00159v1)

**作者:** Kaiyan Zhao `[一作]` (Wuhan University), Xiaoguang Niu `[通讯]` (Wuhan University)

**通讯引用:** 692 | [OpenAlex ID](https://openalex.org/A5057081298)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于决策 Transformer 的主动经验采样框架 E²DT，能够在长周期机器人操控任务中高效学习

**💡 创新点**

创新点在于将决策 Transformer 的内部表示与质量‑多样性联合核相结合，利用 k‑DPP 进行主动采样，并通过重要性加权混合回放纠正采样偏差

**🔧 技术方法**

核心技术包括 DT 内部潜在表示提取、复合质量评分（RTG 分位数、预测不确定性、阶段覆盖率）、RBF 相似度、k‑Determinantal Point Process 采样、重要性加权训练

**📊 数据集**

使用 RoboSuite、ManiSkill2 以及真实 Elephant Robotics 280 机械臂的多项长周期操控任务数据集进行实验

**📈 对比分析**

与 DT、HER、SynthER、MAPLE、Relo、SkillTree 等基线相比，E²DT 在所有任务上均显著提升成功率和样本效率，尤其在稀有关键阶段表现最为突出

**⚠️ 局限性**

局限在于对超参数（k/N、η、σ、K）和阶段标签定义敏感；当前实现侧重于离线日志场景，在线动态适应性仍待进一步研究

---

## 31. Cross-level Privacy Preserving Utility Mining

**arXiv ID:** 2605.00036 | [PDF](https://arxiv.org/pdf/2605.00036v1)

**作者:** Jiahong Cai `[一作]` (Jinan University), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 136296 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了跨层次隐私保护高效利用挖掘（CLPPUM）任务，并设计了三种基于RGISU、NSC与SC指标的隐私保护算法——Min‑RF、Max‑RF与Best‑NSCF，以在包含层次化税onomy信息的数据库中隐藏敏感跨层高效利用项集（SCLHUIs）

**💡 创新点**

创新点包括：①首次将隐私保护高效利用挖掘扩展到跨层次场景；②将敏感计数、非敏感计数以及真实敏感项集效用指标（RGISU）推广到跨层次上下文；③提出专门的GI‑dic字典结构，实现对跨层项集相关指标的高效计算与victim项选择；④提出三种不同的victim项选择策略，并通过实验验证其在稀疏与稠密数据集上的优越性

**🔧 技术方法**

技术手段：基于层次化税onomy树的高效利用挖掘框架；利用RGISU、SC、NSC等指标进行victim项与事务的排序与选择；利用GI‑dic字典加速指标计算；采用删除/减量化操作实现项集隐藏；实验实现采用Java；与传统PPUM（HHUIF、MSICF）对比

**📊 数据集**

实验数据集：Foodmart、Fruithut、Chainstore、Chess以及Foodmart的前5,000和10,000条记录子集（Foodmart_5000、Foodmart_10000）

**📈 对比分析**

比较方法：在同一硬件环境下与基线HHUIF、MSICF比较，评价指标包括运行时间、隐藏失败（HF）、缺失成本（MC）、人工成本（AC）、项集效用相似度（IUS）、数据库效用相似度（DUS）和事务修改率（TMR）。结果显示Min‑RF在所有数据集上均能完全隐藏SCLHUIs（HF=AC=0），并在运行时间、MC、IUS、DUS等方面优于两种基线，Max‑RF在稠密Chainstore上略逊，Best‑NSCF在稀疏数据集与Min‑RF表现相近

**⚠️ 局限性**

局限性：在密集型数据集（如Chainstore）中，由于项与项集的高度相关性，删除策略导致较高的缺失成本和侧效，难以在保持高效的同时完全消除对非敏感项集的影响；算法仍为单线程实现，缺乏并行化或分布式优化，尚未能在更大规模稠密数据上实现可接受的性能

---

## 32. Cloud Is Closer Than It Appears: Revisiting the Tradeoffs of Distributed Real-Time Inference

**arXiv ID:** 2605.00005 | [PDF](https://arxiv.org/pdf/2605.00005v1)

**作者:** Pragya Sharma `[一作]` (University of California Los Angeles), Mani Srivastava `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并验证在满足网络延迟可控的情况下，云端推理在安全关键的实时控制任务（如紧急刹车）中能够与甚至优于本地推理。

**💡 创新点**

提出了一套基于感知频率、网络延迟、模型推理时延和安全约束的正式分析模型，并通过仿真验证其可行性；展示了云端在多任务负载下更好地利用计算资源的优势。

**🔧 技术方法**

使用 M/M/1 队列模型分析推理时延；在 CARLA 仿真平台上构建紧急刹车场景；部署 YOLO11 系列目标检测模型；通过实验对比云端（RTX A5000）和本地 Jetson AGX Orin 的性能。

**📊 数据集**

采用 CARLA 生成的实时驾驶图像数据，YOLO11 预训练于 COCO 数据集。

**📈 对比分析**

通过与本地推理对比，云端在平均网络延迟（约 22 ms）下实现更早的检测与刹车，停止距离更短；在高网络尾延迟或高负载、重型车辆时，云端性能会下降，表现与本地相近或略逊。

**⚠️ 局限性**

受限于高尾延迟、极端负载、车辆重量/速度极限以及未考虑多模态融合、加密开销和动态障碍预测等因素，导致云端在某些极端场景下不具备可靠性。

---

## 33. AirFM-DDA: Air-Interface Foundation Model in the Delay-Doppler-Angle Domain for AI-Native 6G

**arXiv ID:** 2605.00020 | [PDF](https://arxiv.org/pdf/2605.00020v1)

**作者:** Kejia Bian `[一作]` (Shanghai Jiao Tong University), Leyan Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 989 | [OpenAlex ID](https://openalex.org/A5100420743)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 AirFM-DDA，一种基于延迟-多普勒-角度（DDA）域的无线基础模型，能够在统一的框架下完成信道预测、估计等物理层任务。

**💡 创新点**

创新点包括：①将多径叠加的 STF 域 CSI 转换为可分离的 DDA 域；②采用局部窗口注意力（W-MSA + SW-MSA）以实现线性复杂度的自注意力；③引入帧结构感知位置编码（FS-PE）把时频帧参数映射为连续坐标，增强模型对极端多径折叠与不明确区域收缩的鲁棒性。

**🔧 技术方法**

技术手段主要有：4D Fourier 变换实现 DDA 域映射；Swin‑style 窗口自注意力和跨窗口注意力；位置编码基于帧参数的连续坐标；大规模自监督预训练与少量样本微调。

**📊 数据集**

使用了 2.85 M 条 DeepMIMO 城市场景的 CSI 以及 100 k 条 WAIR‑D 生成的样本作为预训练和评估数据集。

**📈 对比分析**

与 WiFo、LLM4WM、Transformer、U‑Net 等基线对比，AirFM‑DDA 在 TP、FP、CE 任务中在零样本、少样本以及常规训练设置下均取得了显著的 NMSE 下降（最高约 4–5 dB）且训练/推理成本降低近八倍。

**⚠️ 局限性**

局限性在于仍需要海量预训练数据；在极高噪声或极大多径扩散条件下性能衰退；目前仅在仿真数据上验证，缺乏真实 OTA 测试。

---

## 34. Learning physically grounded traffic accident reconstruction from public accident reports

**arXiv ID:** 2605.00050 | [PDF](https://arxiv.org/pdf/2605.00050v1)

**作者:** Yanchen Guan `[一作]` (University of Macau), Zhenning Li `[通讯]` (University of Macau)

**通讯引用:** 2526 | [OpenAlex ID](https://openalex.org/A5101552930)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于多模态学习的交通事故重建框架，将公开事故报告的语义信息与现场测量（道路几何、轨迹、EDR等）结合，构建CISS-REC大规模数据集，并通过该框架实现前碰撞车辆轨迹的物理可验证重建与可控可视化；

**💡 创新点**

创新点主要在于：①将事故报告视为弱监督源，利用道路拓扑和参与者语义先验实现全局约束与局部碰撞细化；②设计分阶段训练、图Transformer与时间分配模块的统一框架；③首次公开了基于CISS的真实事故重建数据集CISS-REC；

**🔧 技术方法**

使用了文本+结构+BEV三模态编码器、图Transformer交互解码器、局部几何细化分支、时间分配物理一致性模块，以及多任务损失（轨迹、碰撞、速度、行为、物理约束）；后续通过可控渲染管线实现可视化；

**📊 数据集**

采用NHTSA CISS公开事故报告共计6,217例，构建CISS-REC数据集，包含道路几何、事故报告语义、轨迹、EDR速度、速度限制等信息；

**📈 对比分析**

与STGCN、Spline、LSTM、Wayformer、HiVT、PC‑Crash、Momentum‑Energy等多种基线进行对比，使用AKD、AVD、AAPD、CR、CSA、BA、RA等指标评估；实验结果显示本文方法在轨迹精度、碰撞点定位、碰撞一致性与行为一致性等方面均显著优于基线；

**⚠️ 局限性**

主要局限包括：①速度预测受限于报告中仅有限速信息，导致AVD偏高；②对地图稀疏或多路平行方向时难以唯一确定初始车辆位置；③事故报告缺乏碰撞角度与表面细节的描述，导致部分碰撞细节重建误差。

---

## 35. Smart Ensemble Learning Framework for Predicting Groundwater Heavy Metal Pollution

**arXiv ID:** 2605.00056 | [PDF](https://arxiv.org/pdf/2605.00056v1)

**作者:** T. Ansah-Narh `[一作]` (Ghana Atomic Energy Commission), S. K. Fosuhene `[通讯]` (Ghana Atomic Energy Commission)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究构建了一个嵌套交叉验证的堆叠式机器学习框架，用于从地下水中六种重金属浓度预测重金属污染指数（HPI），并通过响应变量的原始、对数与高斯Copula变换比较模型表现。

**💡 创新点**

创新点在于将高斯Copula变换与堆叠式Lasso元学习器相结合，既能保留多变量的依赖结构，又能在高度偏斜的目标上提升预测稳健性，并通过DBSCAN聚类诊断污染主导金属。

**🔧 技术方法**

采用的技术包括支持向量回归、k近邻、CART、弹性网络、核岭回归等基学习器，随后堆叠成Lasso元学习器，并使用嵌套交叉验证进行超参数调优；此外还使用随机森林对金属浓度进行空间插值。

**📊 数据集**

使用了2019年1月在丹苏河流域采集的96个地下水样本的六种重金属浓度数据（As、Pb、Mn、Fe、Cd、Ni），并计算对应的HPI。

**📈 对比分析**

在三种变换下比较模型性能，结果显示在Copula变换下堆叠模型取得最佳表现（R²≈0.96，RMSE≈0.19），对数变换的SVM表现最优（R²≈0.93，RMSE≈0.28），而原始尺度下Elastic Net虽表现极高（R²≈1）但存在过拟合风险。

**⚠️ 局限性**

主要局限在于未采用空间块交叉验证可能导致空间相关性偏差，模型仅针对丹苏流域，缺乏跨流域推广性；此外Copula变换虽改善残差，但仍未能完全捕捉潜在的阈值效应或非线性过程。

---

## 36. CRADIPOR: Crash Dispersion Predictor

**arXiv ID:** 2605.00070 | [PDF](https://arxiv.org/pdf/2605.00070v1)

**作者:** Edgar Chaillou `[一作]` (Arts et Métiers Institute of Technology), Francisco Chinesta `[通讯]` (ENSAM Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于 Rank Reduction Autoencoder（RRAE）与 MLP 的后处理工具 CRADIPOR，用于从有限次数的汽车碰撞仿真中识别对数值色散敏感的结构区域。

**💡 创新点**

创新点在于：① 将低秩自编码器与时间频率变换（小波、斜率变化）相结合，构建可迁移的低维特征表示；② 只需少量仿真即可训练模型；③ 通过斜率变换捕捉局部不稳定特征，实现对数值色散的高精度检测。

**🔧 技术方法**

使用的技术包括：随机森林基准分类、傅里叶变换、离散小波变换（db4）、局部斜率变化特征、Rank Reduction Autoencoder、多层感知机（MLP）以及交叉验证和混淆矩阵评估。

**📊 数据集**

使用的数据集为一辆车型在 65 km/h 正面左侧碰撞场景下的 10‑15 次重复仿真，节点位移时间序列采样 29 或 289 步，共计数千个节点。

**📈 对比分析**

比较方法：以随机森林在原始轨迹上的 90%+ 准确率为基准；RRAE+MLP 在波形/斜率输入下分别达 98.3%–99.7% 的准确率，召回率和精确率均超过 95%；波形/斜率特征显著优于傅里叶特征，RRAE 与传统方法相比提升 3–5% 的分类性能。

**⚠️ 局限性**

局限性：① 对技术定义（几何、网格、材料等）的依赖未完全消除，模型在后续迭代中可能退化；② 需要较多仿真数据（10+ 运行）和大规模存储；③ 仅针对节点级色散，未考虑时间序列顺序的变化；④ 在不同车型间的跨项目泛化尚未验证。

---

## 37. AIDA-ReID: Adaptive Intermediate Domain Adaptation for Generalizable and Source-Free Person Re-Identification

**arXiv ID:** 2605.00111 | [PDF](https://arxiv.org/pdf/2605.00111v1)

**作者:** Sundas Iqbal `[一作]` (Nanjing University of Information Science and Technology), Weihua Oue `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 AIDA/SF-MIDA 框架，在多源与源免费场景下通过动态反馈调节中间域学习，实现人像再识别的跨域鲁棒性提升。

**💡 创新点**

创新点在于三大模块的协同：多源中间域生成器（MS-IDG）产生连续的桥接分布；伪镜像正则化（PMR）保持身份语义不失真；动态反馈控制器（DFC）根据预测不确定性和梯度方差自适应调整混合强度与正则化力度。

**🔧 技术方法**

采用特征统计混合（AdaIN）、伪标签一致性约束、熵与梯度方差反馈调节、ResNet‑50+GeM、SAM 等技术实现。

**📊 数据集**

使用 Market‑1501、DukeMTMC‑ReID、MSMT17、CUHK03、PersonX 等公开人像再识别基准数据集。

**📈 对比分析**

在多源域泛化、源免费适应与单源 UDA 等多种评测场景中，AIDA/SF-MIDA 在 Rank‑1/mAP 上往往取得或逼近目前最佳水平，尤其在 MSMT17 与 CUHK03 等难度较大的目标域上表现突出。

**⚠️ 局限性**

局限性包括仅在特征层面生成中间域，未显式建模时序/跨相机轨迹信息，且在极端域移或动态场景下的鲁棒性仍有提升空间。

---

## 38. Autoformalizing Memory Specifications with Agents

**arXiv ID:** 2605.00058 | [PDF](https://arxiv.org/pdf/2605.00058v1)

**作者:** Jan Ole Ernst `[一作]` (Normal Computing), Matthias Jung `[通讯]` (Fraunhofer Institute for Experimental Software Engineering)

**通讯引用:** 2064 | [OpenAlex ID](https://openalex.org/A5010135168)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套基于 DRAMPyML 的硬件自动形式化系统，能够将 DRAM 标准的自然语言规范自动转换为可执行的 Petri 网模型，并支持生成 SystemVerilog 断言、刺激和覆盖等验证资产。

**💡 创新点**

创新点包括：①提出 DRAMPyML 作为更通用的抽象表示；②构建 agentic 迭代自动形式化框架；③定义 Jaccard 指标和 Timing Constraint Recall 两种评价指标；④发布 13 份 DRAM 标准的开源基准集 DRAMBench。

**🔧 技术方法**

主要技术：大型语言模型（Sonnet 4.5、GPT‑5.2、DeepSeek‑V3.2），Python Petri 网实现（DRAMPyML），自动化验证工具（MCP 服务器、Petri 网分析），以及基于结构与时序约束的等价性理论。

**📊 数据集**

使用 DRAMBench 数据集，包含 DDR2‑5、LPDDR2‑5、GDDR5‑7、HBM2‑3 共 13 份 JEDEC DRAM 规范的人工标注 Petri 网。

**📈 对比分析**

与一-shot 生成方式对比，agentic 方法在弱先验下能获得更高的 Jaccard 指数（>0.5）和时序召回率；在提供示例模型时，某些模型甚至可达到 100% 结构准确率；在 token‑效率方面，一-shot 在较短规范上表现更好，但 agentic 在复杂规范上优于一-shot。

**⚠️ 局限性**

局限性：对极大规范（如 DDR5）仍存在结构/时序误差；缺少完整时序验证；对参数抽取的自动化尚未实现；高 token 需求导致某些模型出现幻觉与验证误差。

---

## 39. Learning from the Unseen: Generative Data Augmentation for Geometric-Semantic Accident Anticipation

**arXiv ID:** 2605.00051 | [PDF](https://arxiv.org/pdf/2605.00051v1)

**作者:** Yanchen Guan `[一作]` (University of Macau), Zhenning Li `[通讯]` (University of Macau)

**通讯引用:** 2526 | [OpenAlex ID](https://openalex.org/A5101552930)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出双路径框架：利用结构化提示驱动的VLM与Stable Diffusion实现高保真数据增强，并设计语义几何增强的动态图卷积网络进行事故预测；

**💡 创新点**

创新点包括（1）基于预训练VLM提取环境分布并生成多样化事故场景；（2）在图网络中同时引入语义与几何自适应边权，提升交互建模；（3）使用跨模态对齐损失（InfoNCE）强化视觉-文本一致性；（4）发布覆盖多地区、多天气的MAA事故数据集；

**🔧 技术方法**

采用VLM（Qwen-VL）、Stable Diffusion、CLIP、YOLOv8、ZOE深度估计、动态图卷积网络（GCN）、时间卷积+GRU、多模态融合与对比学习；

**📊 数据集**

使用DAD、A3D、CCD、MAA（新数据集）以及合成的sDAD/EMM-AU进行实验；

**📈 对比分析**

在AP和mTTA指标上与多种基线（RNN、GCN、Transformer、ViViT、VideoMAE等）对比，方法在MAA、DAD等数据集上实现最高AP（≈85.8%）和显著提升的mTTA（≈4.1s），比前沿方法提升约5–10% AP和1–3秒TTA；

**⚠️ 局限性**

主要局限在于合成数据与真实数据存在域差距，替换真实数据会导致性能下降；模型对极短时间或极端视角事故的预测仍受限；计算瓶颈集中在深度估计和VLM推理，需云端协同或优化。

---

## 40. Model Checking for Low Monodimensionality Fragments of CMSO on Topological-Minor-Free Graph Classes

**arXiv ID:** 2605.00192 | [PDF](https://arxiv.org/pdf/2605.00192v1)

**作者:** Ignasi Sau `[一作]` (LIRMM), Alexandre Vigny `[通讯]` (LIMOS)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文进一步发展了Sau、Stamoulis和Thilikos的框架，通过注释图参数对逻辑的片段进行分割，识别出一个新的逻辑片段，该片段允许对具有低单维度的集合进行量化，并证明在排除固定拓扑次要的图类上，该片段的模型检查是固定参数可解的。

**💡 创新点**

创新点在于识别了一个新的逻辑片段，该片段通过限制集合量化来增强逻辑的表达能力，并且在拓扑次要排除的图类上实现了固定参数可解性，这扩展了已有的算法元定理。

**🔧 技术方法**

使用了注释图参数和动态规划技术来实现模型检查，并结合了分离路径谓词以增强逻辑的表达能力。

**📊 数据集**

使用了排除固定拓扑次要的图类作为数据集，具体的图类未在摘要中详细说明。

**📈 对比分析**

与现有方法的比较表明，本文提出的逻辑片段在拓扑次要排除的图类上实现了固定参数可解性，优于传统的逻辑方法，尤其是在处理更复杂的图结构时。

**⚠️ 局限性**

限制在于该方法的适用性可能仅限于特定的图类，且在更广泛的图类中可能无法保持固定参数可解性。

---

## 41. TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization

**arXiv ID:** 2605.00224 | [PDF](https://arxiv.org/pdf/2605.00224v1)

**作者:** Abdulhady Abas Abdullah `[一作]` (University of Kurdistan), Mourad Oussalah `[通讯]` (University of Oulu)

**通讯引用:** 3814 | [OpenAlex ID](https://openalex.org/A5068812101)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TUR-DPO，一种在不使用强化学习的前提下，将轻量化推理拓扑和不确定性权重引入直接偏好优化（DPO）的方法，用于提升大型语言模型在推理、问答、摘要与对话中的可靠性与可解释性。

**💡 创新点**

创新点在于：① 通过从生成文本中提取简易推理图，衡量语义质量、拓扑完整性与结构合理性；② 将三种信号融合成校准奖励，并利用不确定性进行实例加权，降低对嘈杂或脆弱偏好对训练的负面影响；③ 在保持 DPO 原始闭式无 roll‑out 训练流程的同时，显著提升准确率、真实性与校准度。

**🔧 技术方法**

使用的技术包括：轻量化图结构抽取与本地验证器、基于 Jensen‑Shannon 散度与熵的 epistemic/aleatoric 不确定性估计、线性校准器构造奖励、实例加权的 Bradley‑Terry 形式对数损失，以及可选的列表式（Plackett‑Luce）多候选优化。

**📊 数据集**

在多种公开基准上评估：数学推理（GSM8K、MATH）、事实问答（Open QA）、摘要（TLDR）、对话（Help/Harmless）以及 multimodal 与 long‑context 场景；同时对比了 DPO、PPO‑RLHF 及 ORPO、SimPO、KTO、IPO 等最新 RL‑free 方法。

**📈 对比分析**

与 DPO 相比，TUR‑DPO 在所有任务上实现了 2–6% 的准确率提升、显著降低错误率（尤其是逻辑跳跃与实体幻觉），并在 Calibration（ECE、Brier）上优于 DPO；与 PPO‑RLHF 相比，TUR‑DPO 维持或超过其性能，却无需 roll‑outs，训练效率更高、计算成本更低。

**⚠️ 局限性**

局限性包括：依赖图抽取工具的质量，若抽取错误可能影响奖励信号；不确定性估计仅基于有限重抽，可能无法捕获所有极端错误；在极长对话或极大上下文、复杂多模态任务中的鲁棒性尚未充分验证，未来需改进图抽取可靠性与更高级的不确定性量化方法。

---

## 42. zkSBOM: Privacy-Preserving SBOM Sharing with Zero-Knowledge Sets

**arXiv ID:** 2605.00076 | [PDF](https://arxiv.org/pdf/2605.00076v1)

**作者:** Tom Sorger `[一作]` (KTH Royal Institute of Technology), Martin Monperrus `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 7030 | [OpenAlex ID](https://openalex.org/A5027206285)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于零知识集合（ZKS）的zkSBOM隐私保护SBOM共享协议，允许软件消费者在不泄露SBOM其余内容的前提下验证软件是否存在已知漏洞；

**💡 创新点**

首次将零知识集合应用于SBOM共享，解决全披露与完全隐藏之间的矛盾，并系统分析了查询过程中的信息泄露量；

**🔧 技术方法**

核心技术包括零知识集合（ZKS）、数字签名、透明性日志（TL）以及SBOM转换为键值数据存储；

**📊 数据集**

使用了四大生态系统（Cargo、Go、Maven、npm）中的真实开源项目SBOM，并基于Wild SBOMs数据集评估了性能与泄露；

**📈 对比分析**

通过对比synthetic与real-world SBOM的实验，证明了zkSBOM在生成承诺、证明与验证上均能在毫秒级完成，存储开销在几MB以内，满足实际部署需求；

**⚠️ 局限性**

局限在于假设SBOM生成者诚实、SBOM准确性未保障，且零知识证明仍可能泄露基于公开生态数据的传递和同级依赖信息。

---

## 43. How Frontier LLMs Adapt to Neurodivergence Context: A Measurement Framework for Surface vs. Structural Change in System-Prompted Responses

**arXiv ID:** 2605.00113 | [PDF](https://arxiv.org/pdf/2605.00113v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 44. DeGenTWeb: A First Look at LLM-dominant Websites

**arXiv ID:** 2605.00087 | [PDF](https://arxiv.org/pdf/2605.00087v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 45. A Survey of Reasoning-Intensive Retrieval: Progress and Challenges

**arXiv ID:** 2605.00063 | [PDF](https://arxiv.org/pdf/2605.00063v1)

**作者:** Yiyang Wei `[一作]` (Zhejiang University), Yilun Zhao `[通讯]` (Yale University)

**通讯引用:** 624 | [OpenAlex ID](https://openalex.org/A5047416722)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了推理密集检索（RIR）的研究现状，系统整理了不同领域和模态的基准数据集，提出了分阶段的检索流程分类，并分析了现有方法的优缺点与应用场景；

**💡 创新点**

创新点在于首次给出RIR的整体框架和细粒度的检索流程分类，整合并比较了基于查询重写、索引增强、检索器训练、再排序和迭代检索等多种思路，并提出了未来研究的关键挑战；

**🔧 技术方法**

主要采用文献综述、基准对比分析、方法归纳与分类、指标讨论以及对比实验结果的汇总等技术手段；

**📊 数据集**

引用并分析了多类基准数据集，包括开放域（BESPOKE、ImpliRet）、专家域（ScIRGen、FreshStack、MIRB、MathNet-Retrieve、Bar Exam QA、R2MED、CoIR、CoQuIR）、多域（RAR-b、-Plus）以及多模态（MRMR、MM‑BRIGHT、ARK、MR²‑Bench）等；

**📈 对比分析**

对不同方法在相同基准上的效果进行了归纳比较，表明整合推理的检索器和再排序器在多数指标（如nDCG、Recall）上优于传统方案，但往往伴随更高的计算成本；

**⚠️ 局限性**

局限性包括：仅关注公开论文和公开基准，未覆盖图结构检索、HyDE等潜在方向；缺乏统一的评估指标和跨域泛化评估；对工业级系统和非文本多模态任务的讨论不足；

---

## 46. Do Open-Loop Metrics Predict Closed-Loop Driving? A Cross-Benchmark Correlation Study of NAVSIM and Bench2Drive

**arXiv ID:** 2605.00066 | [PDF](https://arxiv.org/pdf/2605.00066v1)

**作者:** Yiru Wang `[一作]` (Bosch Corporate Research), Hao Sun `[通讯]` (Bosch Corporate Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了NAVSIM开环指标与Bench2Drive闭环评估之间的跨benchmark配对数据集，系统分析了两者的相关性，并提出了仅使用NC、DAC、EP三项指标的简化CL-Proxy评分，以与完整的PDMS保持相同的排名相关性。

**💡 创新点**

首次从子指标层面系统评估开环与闭环的映射关系，发现Ego Progress（EP）是闭环驾驶分数的最强单一预测指标，并提出CL-Proxy能在不增加复杂度的前提下实现与五项指标PDMS同等的排名相关性。

**🔧 技术方法**

采用Spearman和Kendall相关性分析、统计显著性检验以及可视化绘图，对NAVSIM子指标和Bench2Drive Driving Score进行数值比较；使用了NAVSIM v2的安全与进度子指标以及Bench2Drive的DS作为评估目标。

**📊 数据集**

数据集来源于NAVSIM v2（含PDMS及各子指标）和Bench2Drive（CARLA v2协议下的220条路由与44个交互场景），共纳入8个方法的完整开环与闭环配对数据。

**📈 对比分析**

与传统的ADE/FDE等开环误差指标相比，NAVSIM PDMS与Bench2Drive DS的Spearman相关系数为0.90（p=0.002），排名一致；CL-Proxy同样取得ρ=0.90，并与PDMS拥有相同的排名逆转数量，表明简化后仍保持高预测性能。

**⚠️ 局限性**

研究局限包括配对样本量仅为8个，Bench2Drive评估存在运行方差，跨架构差异可能影响相关性；且分析仅为相关性，未建立因果关系；NAVSIM v1与v2之间的指标差异也可能影响结果。

---

## 47. Two-View Accumulation as the Primary Training Lever for Hybrid-Capture Gaussian Splatting: A Variance-Decomposition View of When Gradient Surgery Helps

**arXiv ID:** 2605.00052 | [PDF](https://arxiv.org/pdf/2605.00052v1)

**作者:** Sungjun Cho `[一作]` (Hong Kong University of Science and Technology), Sungjun Cho `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 935 | [OpenAlex ID](https://openalex.org/A5102903741)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文针对混合捕捉（高低视角混合）的三维高斯散射（3DGS）训练难题，提出一种训练时结构化的改进：在每一步优化中渲染并分别计算一对近视角与远视角图像的梯度，然后对冲突梯度做对称投影，显著提升混合视角下的重建质量。

**💡 创新点**

创新点在于①将“每步双视角渲染”作为最有效的结构性杠杆；②构建方差分解框架解释了结构化与随机配对在梯度方差上的差异；③通过大量实验揭示多种梯度调和方法（如方向投影、幅度校正、主动损失不匹配）在混合捕捉场景中并未提升 PSNR，验证了结构化改进的独特性。

**🔧 技术方法**

技术方法包括：3D 高斯散射渲染、基于距离的近/远视角分组、对冲突梯度的对称投影（CrossGrad-GS）、幅度校正（GradNorm）、MGDA/CAGrad 等对比实验、方差分解分析、随机两视角对照实验、不同分组策略（k‑means、GMM、冲突签名）等。

**📊 数据集**

使用了五个混合捕捉基准：UC‑GS（NYC、SF）、MatrixCity（mixed_extreme）、HorizonGS（Road、Park）。这些数据集包含高低摄像机高度差异大（相对高度方差高达 0.95+）的场景。

**📈 对比分析**

与 vanilla 3DGS（30K 迭代）以及 60K 渲染视角对照、Scaffold‑GS、Mip‑Splatting、Analytic‑Splatting、Octree‑GS、Pixel‑GS 等方法对比，CrossGrad‑GS 在四个场景中提升 PSNR 1.01–3.35 dB，甚至超过 60K vanilla 的 PSNR，表明结构化双视角渲染是最具效果的改进。

**⚠️ 局限性**

局限性包括：①对摄像机布局的分组策略依赖性，UC‑GS SF 这类高度分布不对称的场景仍无法显著受益；②方法仅涉及训练侧梯度聚合，未改进表示或渲染；③在某些高频细节或极端视角下，单纯双视角渲染仍可能不足，需要结合表示层面的改进。

---

## 48. Fair Dataset Distillation via Cross-Group Barycenter Alignment

**arXiv ID:** 2605.00185 | [PDF](https://arxiv.org/pdf/2605.00185v1)

**作者:** Mohammad Hossein Moslemi `[一作]` (Western University), Bissan Ghaddar `[通讯]` (Ivey Business School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出了一种针对数据集蒸馏过程中公平性问题的解决方案；

**💡 创新点**

创新点在于引入跨组重心对齐（COBRA）技术，通过计算各子群代表的均衡重心来对齐蒸馏目标，从而减少子群间的偏差；

**🔧 技术方法**

技术上使用了重心对齐目标、MSE/二次范数距离、以及已有的蒸馏框架（如DC、DM、CAFE等）进行兼容性改进；

**📊 数据集**

实验数据集包括受控的 Colored-MNIST、Colored-Fashion-MNIST、以及真实世界的受保护属性数据集（如公平性基准数据集），并在不同子群不平衡比例下进行评估；

**📈 对比分析**

与传统蒸馏（Vanilla DD）和 FairDD 等基线对比，COBRA 在所有实验中显著降低了公平性指标（Equalized Odds差距），同时保持或提升整体准确率；

**⚠️ 局限性**

局限性在于对组标签的依赖、对不同距离度量的敏感性，以及在多重受保护属性或分布漂移情况下的推广性尚未充分验证。

---

## 49. ROSA: Robust and Energy-Efficient Microring-Based Optical Neural Networks via Optical Shift-and-Add and Layer-Wise Hybrid Mapping

**arXiv ID:** 2605.00032 | [PDF](https://arxiv.org/pdf/2605.00032v1)

**作者:** Huifan Zhang `[一作]` (ShanghaiTech University), Pingqiang Zhou `[通讯]` (ShanghaiTech University)

**通讯引用:** 782 | [OpenAlex ID](https://openalex.org/A5081772659)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于微环光学环的光学神经网络架构ROSA，采用光学移位加法（OSA）模块和层级混合映射策略，以提升鲁棒性与能效。

**💡 创新点**

创新点包括：① 引入OSA实现光学域累计，显著降低OAC/ADC功耗；② 采用数字-模拟混合加权方案，兼顾高速吞吐和温度/DAC噪声鲁棒性；③ 设计层级混合映射策略，在保持高位量化精度的同时提升准确率与能效。

**🔧 技术方法**

使用技术包括：微环光学环（MRR）、多波长分波复用（WDM）、光学移位加法（OSA）、光-电-光转换、噪声感知电压-权重模型、工作负载感知优化、层级混合映射。

**📊 数据集**

数据集：CIFAR‑10、MNIST、GPT‑2 Medium、Vision Transformer 等。

**📈 对比分析**

比较方法：在多种CNN（AlexNet、MobileNet V3、ResNet‑18、VGG‑16）和Transformer（GPT‑2、ViT）上，与DEAP‑CNNs及紧凑阵列基准进行EDP与准确率对比；ROSA在最佳OPE尺寸下EDP比DEAP低64%，OSA模块单独降低29%，混合映射在CIFAR‑10上平均提升8.3%准确率，EDP下降54.7%。

**⚠️ 局限性**

局限性：仍受限于TO调谐速度、光衰损耗和光延迟线可变误差；层级混合映射需要额外的调度与层级信息，调度复杂度提升；实验基于模拟，缺乏大规模实测验证。

---

## 50. What Characterizes a Software Leader? Identifying Leadership Practices from Practitioners Social Media

**arXiv ID:** 2605.00191 | [PDF](https://arxiv.org/pdf/2605.00191v1)

**作者:** Murilo Coelho `[一作]` (Atlantico Institute), Savio Freire `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

从Dev.to社区抓取116篇关于领导力的文章，利用内容分析法提取并归纳出103个推荐和94个禁忌的领导实践，随后按管理/技术维度和五大类别进行分类，并构建概念地图。

**💡 创新点**

首次基于实践者在社交媒体上的真实、非结构化叙事来定义领导实践；提出双向（推荐/禁忌）与技术/管理并行的分类框架；并公开完整的代码、数据与编码结果，保证可复现性。

**🔧 技术方法**

采用Python+Dev.to官方API抓取、手工筛选与双人编码；运用开放编码、聚类、Kendall Tau相关系数以及统计显著性检验来量化实践共现；使用可视化工具绘制概念地图。

**📊 数据集**

数据集为从Dev.to使用22个与领导相关标签检索得到的1,815篇文章，在两阶段筛选后剩余116篇文章（共893段落）。

**📈 对比分析**

通过计数、百分比与相关系数评估实践出现频次和共现模式；发现推荐实践中管理类占87.4%，技术类仅12.6%；禁忌实践中管理类占92%；相关性分析显示核心实践高度互相关联，说明实践在社区中呈现聚集式讨论；未与传统模型做性能对比，主要提供描述性统计与可视化洞见。

**⚠️ 局限性**

局限性包括：仅采集公开博客的自愿发帖者，可能存在自我呈现与社群偏差；缺乏对实际行为的验证；数据来源局限于Dev.to，难以代表所有行业与文化；技术禁忌低可能与采样偏差有关。

---

## 51. DEPTEX: Organization-First, Open Source Dependency Risk Monitoring

**arXiv ID:** 2605.00179 | [PDF](https://arxiv.org/pdf/2605.00179v1)

**作者:** Henry Ruckman-Utting `[一作]` (Simon Fraser University), Mohammad A. Tayebi `[通讯]` (Simon Fraser University)

**通讯引用:** 515 | [OpenAlex ID](https://openalex.org/A5060953615)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Deptex 平台，用组织优先的图模型和执行路径主导技术（EPD）实现供应链风险的上下文感知评估，并通过“安全即代码”实现可编程治理。

**💡 创新点**

创新点包括：① 把供应链风险视作组织层面的涌现属性；② 将代码属性图（CPG）切片与大型语言模型（LLM）语义验证相结合的执行路径主导（EPD）算法；③ 在平台内构建可执行的 JavaScript sandbox，实现安全、合规、PR 门控和通知等多维度治理即代码。

**🔧 技术方法**

技术栈：图数据库（typed property graph）、代码属性图（CPG）解析器、LLM（带 JSON schema 的对话式语义验证）、JavaScript sandbox 的“安全即代码”引擎、Webhook 与第三方通知集成（PagerDuty、Slack 等）。

**📊 数据集**

使用公开的 SBOM 与 CVE 数据（如 CVE‑xxxx 9.8 的解析库漏洞）进行实验，并结合组织内部的依赖图与资产层级信息来构建测试场景。

**📈 对比分析**

通过与 Dependabot、Dependency‑Track、Snyk 的功能对比表格展示 Deptex 在组织视图、可编程治理、上下文风险评分等方面的优势。实验表明，EPD 能显著降低误报（从 10 处可达 8 处被降级），从而减少 Type‑B 警报疲劳，虽然未给出精确的性能数值，但整体效能优于传统工具。

**⚠️ 局限性**

局限性：① 评估尚缺乏长期纵向案例验证；② 对 LLM 结果的依赖可能导致误判；③ 目前侧重供应链风险，未整合 SAST、秘密检测等完整 ASPM 功能；④ 对极大规模组织的可伸缩性与性能尚未实测。

---

## 52. UniBCI: Towards a Unified Pretrained Model for Invasive Brain-Computer Interfaces

**arXiv ID:** 2605.00061 | [PDF](https://arxiv.org/pdf/2605.00061v1)

**作者:** Binjie Hong `[一作]` (Institute of Neuroscience Chinese Academy of Sciences), Tielin Zhang `[通讯]` (Institute of Neuroscience Chinese Academy of Sciences)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 UniBCI，一种统一的预训练模型，用于侵入式脑机接口的神经尖峰数据建模。

**💡 创新点**

创新点在于将上下文条件的时空分词、层次区间-区域注意力和自监督掩码重构结合，解决数据异构、域漂移与时空复杂性。

**🔧 技术方法**

使用了自监督掩码重建目标、线性注意力与滑动窗口注意力、MiniLM 预训练文本编码器等技术。

**📊 数据集**

构建了跨物种（猴、鼠、人的）多脑区、多任务的大规模预训练语料库，包括 M1‑CO1、Pac‑Man、HPC‑HG、LICK、PPC‑FINGER 等。

**📈 对比分析**

在分类与回归基准上与 WF、GRU、MLP、VAE、NoMAD、NDT1/2、MtM、POYO 等基线比较，UniBCI 取得 SOTA 的准确率与 R²（例如多日 M1‑CO1 0.895、PPC‑FINGER 0.967、Perich 0.757）。

**⚠️ 局限性**

局限性在于对硬件依赖较高，仍需进一步验证在更大规模、跨装置的泛化能力以及多模态融合。

---

## 53. NorBERTo: A ModernBERT Model Trained for Portuguese with 331 Billion Tokens Corpus

**arXiv ID:** 2605.00086 | [PDF](https://arxiv.org/pdf/2605.00086v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 54. TADI: Tool-Augmented Drilling Intelligence via Agentic LLM Orchestration over Heterogeneous Wellsite Data

**arXiv ID:** 2605.00060 | [PDF](https://arxiv.org/pdf/2605.00060v1)

**作者:** Rong Lu `[一作]` `[通讯]`, Rong Lu

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了TADI（Tool-Augmented Drilling Intelligence）系统，整合并自动化分析Equinor Volve现场的钻井日常报告、WITSML实时数据、生产记录和地质资料，并通过LLM驱动的多工具迭代推理生成基于证据的钻井操作洞察。

**💡 创新点**

创新点在于将领域专业工具与双模态存储（DuckDB+ChromaDB）相结合，利用LLM的函数调用实现自适应检索与计算；同时提出Evidence Grounding Score（EGS）作为多源证据引用的定量指标，验证工具设计而非模型规模是技术域分析质量的关键。

**🔧 技术方法**

采用OpenAI GPT-4 LLM与函数调用、DuckDB进行结构化SQL查询、ChromaDB进行语义向量检索、Python实现无框架架构、LLM提示工程、以及一套12个专门化工具。

**📊 数据集**

使用公开的Equinor Volve Field数据集，包括1759份DDR XML、选取的WITSML实时对象、15634条生产记录、形成层顶与穿孔等共12张表与36,709份文档。

**📈 对比分析**

通过130题压力测试、对比LLM单纯推理、基础RAG以及完整TADI方案，利用EGS衡量答案的证据完整性，TADI在结构化测量与DDR引用上达成高达1.0的EGS，且工具调用均在亚秒级完成。

**⚠️ 局限性**

局限包括对稀疏数据井（如探井）时置信度下降、LLM算术与推理误差、工具选择错误导致步数浪费、对模糊查询缺乏澄清交互、以及对未来钻井推荐等推断任务的验证不足。

---

## 55. Soft-MSM: Differentiable Context-Aware Elastic Alignment for Time Series

**arXiv ID:** 2605.00069 | [PDF](https://arxiv.org/pdf/2605.00069v1)

**作者:** Christopher Holder `[一作]` (University of Southampton), Anthony Bagnall `[通讯]` (University of Southampton)

**通讯引用:** 7410 | [OpenAlex ID](https://openalex.org/A5052586692)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种可微分的时间序列相似度 Soft‑MSM，基于 Move‑Split‑Merge（MSM）的平滑化改写，能够在梯度下降框架下计算对齐成本。

**💡 创新点**

创新点在于将 MSM 的分段切换/合并代价用可微的“光滑门”替代，同时将硬最小化改为 Soft‑min，实现了对动态规划递推及其上下文相关转移成本的梯度传播。

**🔧 技术方法**

采用软最小化（softmin）与光滑门（smooth gate）构建可微转移成本，并推导前向/后向递推、软对齐矩阵以及梯度；实现基于 PyTorch/Numba 等工具。

**📊 数据集**

使用 UCR 112 经典时间序列数据集（包括各种长度和类别），并对其中的子集进行平均、聚类和最近中心分类实验。

**📈 对比分析**

与硬 MSM（MBA、SSG‑MBA）以及 Soft‑DTW（Soft‑DBA、Soft‑DBA2）和传统 k‑Shape、k‑AVG 进行比较；Soft‑MBA 在 112 组数据中显著降低 MSM Fréchet 损失、提升聚类准确率（相对 k‑Shape）和最近中心分类准确率。

**⚠️ 局限性**

局限性包括：失去 MSM 的度量性质；需要全 𝑚×𝑚 DP 矩阵，常数因子较大；对平滑参数 γ 依赖强；自相似性偏差导致 Fγ(x,x)≠0，需要差异化校正。

---

## 56. Consistent Diffusion Language Models

**arXiv ID:** 2605.00161 | [PDF](https://arxiv.org/pdf/2605.00161v1)

**作者:** Hasan Amin `[一作]` (Purdue University), Xia Song `[通讯]` (Microsoft)

**通讯引用:** 2939 | [OpenAlex ID](https://openalex.org/A5087490418)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多路径离散一致性训练框架（MPDC），并实现了单阶段无教师的离散扩散语言模型（CDLM）

**💡 创新点**

创新点在于用离散扩散的确切后验桥替代不存在的概率流ODE，训练出期望意义下的路径不变去噪器，从而在少步生成中获得显著加速

**🔧 技术方法**

使用离散后验桥、Jensen‑Shannon散度、EMA目标、最大步正则化以及步长调度的组合进行训练；采样采用原始的祖传采样器

**📊 数据集**

在无条件文本生成上使用 OpenWebText 数据集，在有条件生成上使用 OpenWebText、Lambada、Wikitext‑103 和 PTB 等常见语言数据集

**📈 对比分析**

与 MDLM、DUO、SDTT、DUO‑DCD 等基线在 PPL、BLEU、MAUVE 等指标下对比，CDLM 在 4–512 步范围内均取得最低 PPL、最高 BLEU，甚至在少步情形下超越多阶段蒸馏模型，显著提升速度和质量

**⚠️ 局限性**

局限性包括：单步生成仍然难以实现高质量；依赖可解析后验桥的离散过程，难以推广到后验不易解析的离散域；对极端多模态数据的处理仍受限

---

## 57. Putting HUMANS first: Efficient LAM Evaluation with Human Preference Alignment

**arXiv ID:** 2605.00022 | [PDF](https://arxiv.org/pdf/2605.00022v1)

**作者:** Woody Haosheng Gan `[一作]` (University of Southern California), Diyi Yang `[通讯]` (Stanford University)

**通讯引用:** 13772 | [OpenAlex ID](https://openalex.org/A5089413311)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大型音频模型的评估，作者系统地对 5 个主流基准的 40 个任务（约 16,000 条样本）进行子集挑选，并在 18 个模型上验证子集能否保留完整基准的排名；随后在 7 个代表性模型上进行 776 次真实语音助手对话的人类喜好评估，并利用 Ridge 回归训练预测人类满意度的权重模型。

**💡 创新点**

创新点在于（1）证明仅 0.3% 的样本（50 条）即可与完整基准保持 0.93+ 的 Pearson 相关；（2）提出基于 Anchor Points 与 Combined Embedding 的混合子集选择策略；（3）发现仅用高质量子集训练的回归模型即可在 0.98 的相关度上超越全基准，提出 HUMANS 这一“人类对齐最小化音频评估子集”公开数据集。

**🔧 技术方法**

主要技术包括多种子集选择方法（随机、方差、难度、IRT、Anchor Points、Embedding‑based clustering），基于 LMM 的 Ridge 回归进行人类偏好预测，交叉验证与 LOMO 评估，GPU 大规模评测（1,520 GPU‑hours）以及自然语言与声学嵌入的联合聚类。

**📊 数据集**

使用的基准数据集为 5 个公开音频评测集（Dynamic‑SUPERB、CAVA、UltraEval‑Audio、SpeakBench、WildSpeech‑Bench）共 40 个任务，约 16,000 条样本；人类数据来自 776 次 10‑分钟真实语音助手对话，涵盖 7 个模型，评估维度包括总体满意度、理解、自然性、回答质量与任务效果。

**📈 对比分析**

实验表明：50 条子集即可与完整基准的相关度达到 0.934；完整基准与人类满意度的相关度约为 0.85；基于 100 条高质量子集训练的 Ridge 回归在 LOMO 评估下的相关度达到 0.978，明显优于随机子集和完整基准；此外，Anchor Points 在极小子集（≤30）上表现最好，而 Combined Embedding 在 50 条以上子集上最高。

**⚠️ 局限性**

局限性包括：人类评测仅覆盖美国英语母语者，难以推广至多语种；只评估了 7 个模型，导致回归模型对新架构的泛化能力不确定；子集针对对话场景，可能不适用于音乐或创意音频等其他领域；子集选择在面对更强大模型时的外推性未知；以及潜在的过拟合风险，模型可能针对公开子集过度优化。

---

## 58. Smart Profit-Aware Crop Advisory System: Kisan AI

**arXiv ID:** 2605.00133 | [PDF](https://arxiv.org/pdf/2605.00133v1)

**作者:** Debasis Dwibedy `[一作]` (VIT-AP University), D Snehaja `[通讯]` (VIT-AP University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了面向农户的智能利润感知作物咨询系统Kisan AI，提供土壤适宜性、病害检测、价格预测及多语言交互等一体化服务。

**💡 创新点**

创新点在于将市场价格预测融入作物推荐框架，构建利润权重评分模型，解决传统系统的经济盲点，并通过多模态模块与多语言支持实现一站式决策。

**🔧 技术方法**

核心技术包括随机森林（推荐）、Facebook Prophet（价格预测）、MobileNetV2（病害识别）、FastAPI后端、PWA前端以及Anthropic Claude API构建的聊天机器人。

**📊 数据集**

使用的数据集为：带市场价格的Crop Recommendation dataset（2200条）、Fertilizer Prediction dataset（500条）、PlantVillage图像集（16500张）和Agmarknet历史价格记录。

**📈 对比分析**

与九种基线模型（5传统、4深度）对比，随机森林在9维特征上达99.3%准确率、最高F1得分，优于其他模型；在病害检测中MobileNetV2获得96.2%准确率。

**⚠️ 局限性**

局限包括对人工输入的依赖、价格API可能不稳定、缺乏实时IoT传感器支持，以及对极端气候或地区特定市场波动的细粒度预测不足。

---

## 59. Comparative Analysis of Polygon-Based and Global Machine Learning Models for Bus Occupancy Prediction

**arXiv ID:** 2605.00083 | [PDF](https://arxiv.org/pdf/2605.00083v1)

**作者:** Daniel Azenkot `[一作]` (University of Ben Gurion), Eran Ben Elia `[通讯]` (Ben-Gurion)

**通讯引用:** 2476 | [OpenAlex ID](https://openalex.org/A5019909719)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了一种结合Max‑p空间聚类与树模型的公交客流预测框架，可在不使用线路、站点ID等显式识别信息的情况下，预测一站后车辆客流，实现长周期（多天）预测；

**💡 创新点**

创新点在于：①将Max‑p聚类与图网络、天气、土地利用等多源特征融合，构建局部自适应模型；②通过不依赖特定ID的设计实现模型迁移性；③用Tobler第一定律验证空间自相关对客流预测的影响；

**🔧 技术方法**

使用机器学习树模型（LightGBM、XGBoost、CatBoost、RandomForest、LinearRegression），Max‑p聚类，图网络构建、特征工程（时间、天气、设施、网络中心性等），SHAP解释，滚动训练评估和Wilcoxon、Cliff’s Delta等统计检验；

**📊 数据集**

数据来源于以色列Be’er Sheva的7.4M条APC站点记录，配合公开的天气数据、公交站点属性、城市设施位置与社会经济分层等多源数据；

**📈 对比分析**

通过滚动滚动原点训练-测试（每周一次）评估，比较全局单模型与局部多模型的MAE、RMSE、MAPE、%RMSE、sMAPE等指标。统计检验显示局部模型在MAE上略优（p<0.05），LightGBM取得最佳MAE≈3.21；去除ID后性能差异不大；

**⚠️ 局限性**

局限包括：APC计数误差导致数据质量受限；对低流量/非高峰、节假日等情形误差较大；扩充训练集不显著提升；Max‑p阈值需人工调参，计算成本相对较高；模型迁移性尚需在更多城市进一步验证。

---

## 60. Towards A Generative Protein Evolution Machine with DPLM-Evo

**arXiv ID:** 2605.00182 | [PDF](https://arxiv.org/pdf/2605.00182v1)

**作者:** Xinyou Wang `[一作]` (Nanjing University), Quanquan Gu `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种演化离散扩散模型 DPLM-Evo，显式建模蛋白质序列的替换、插入、删除，支持变异效应预测与可变长度生成。

**💡 创新点**

（1）解耦可变长度观测空间与上采样对齐空间，实现插入删除操作；（2）引入上下文化进化噪声核生成生物学合理的替换；（3）将训练目标拆解为替换、删除、插入三种编辑任务。

**🔧 技术方法**

基于离散扩散的蛋白语言模型，使用 Transformer + 三个头（substitution、insertion、deletion），上下文化的噪声核，变分下界训练及逆向采样。

**📊 数据集**

UniRef50 训练集，ProteinGym 变异效应评估，CAMEO 用于家族扩张实验，GFP 进行定向进化实验。

**📈 对比分析**

与 ESM、ESM-2、DPLM、ProFam、TranceptEVE 等方法比较；在 ProteinGym 变异效应上单序列 Spearman 达到 0.495，保持高 pLDDT 83.6 的无条件生成，GFP 定向进化 pTM 达到 0.857。

**⚠️ 局限性**

（1）插入删除一次仅限一次，无法一次性多位 indel；（2）对齐近似导致理论边界松散；（3）上下文化噪声核增加约 24% 训练时间；（4）先验仅对替换有效，长度控制不够完整。

---

## 61. RSAT: Structured Attribution Makes Small Language Models Faithful Table Reasoners

**arXiv ID:** 2605.00199 | [PDF](https://arxiv.org/pdf/2605.00199v1)

**作者:** Jugal Gajjar `[一作]` (George Washington University), Kamalasankari Subramaniakuppusamy `[通讯]` (George Washington University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练小型语言模型在表格推理任务中输出带单元格引用的逐步推理过程，保证答案可验证

**💡 创新点**

提出两阶段训练RSAT：先用监督学习学习结构化JSON格式，再用GRPO强化学习结合NLI真实性奖励，使模型在引用时真正依据表格证据

**🔧 技术方法**

使用LoRA微调、GRPO（无critic的强化学习）、基于DeBERTa‑v3的NLI得分作为真实性奖励、格式化门控与稀疏奖励

**📊 数据集**

WTQ、FeTaQA和TabFact三个表格推理基准，SFT阶段采样1,000例，GRPO阶段使用约74k例表格问答对

**📈 对比分析**

与零shot、单独SFT、以及后置引用（post‑hoc）对比；RSAT在六个模型（Qwen 1.5/3/7B、Llama 1/3/8B）上平均faithfulness从0.224提升至0.826，答案F1提升≈0.09，引用有效率和格式成功率均≥0.99，post‑hoc方法格式成功率低于13%

**⚠️ 局限性**

主要局限：真实性评估与奖励使用同一NLI模型导致训练‑评估循环；仅在WTQ/FeTaQA/TabFact评估，未验证跨域泛化；严格EM低导致对答案准确性的评估受限

---

## 62. CompleteRXN: Toward Completing Open Chemical Reaction Databases

**arXiv ID:** 2605.00222 | [PDF](https://arxiv.org/pdf/2605.00222v1)

**作者:** Gabriel Vogel `[一作]` (Delft University of Technology), Jana M. Weber `[通讯]` (Delft University of Technology)

**通讯引用:** 631 | [OpenAlex ID](https://openalex.org/A5001961922)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 CompleteRXN 这一大规模真实缺失-完整化学反应对齐数据集，并在其上评估反应补全模型

**💡 创新点**

创新点包括：①基于 USPTO 与 FlowER 机制化数据的对齐生成真实缺失与完整反应对；②提出受限束搜索的受限反应平衡器（CRB）以及等价精度评估指标

**🔧 技术方法**

采用基于预训练的 Transformer（Molecular Transformer）与受限束搜索的 CRB 以及传统规则+图匹配的 SynRBL 算法

**📊 数据集**

使用来自 USPTO 的原始反应与 FlowER 机制化平衡反应对，形成约 200k 条不完整-完整反应对；同时在完整 USPTO 数据上进行外部评估

**📈 对比分析**

与 RB（无约束）及 SynRBL 对比，CRB 在随机拆分、机制分组拆分及极端 OOD 拆分上分别取得 99.20%、97.10% 与 91.12% 的等价精度；SynRBL 精度低但平衡率较高

**⚠️ 局限性**

局限在于数据仍是模板对齐的子集，未充分覆盖噪声和纠错需求；评估仅对单一“正确”补全计分，可能忽略多重合法补全；在全 USPTO 数据上性能显著下降

---

## 63. Fidelity-Guaranteed Entanglement Routing with Distributed Purification Planning

**arXiv ID:** 2605.00246 | [PDF](https://arxiv.org/pdf/2605.00246v1)

**作者:** Anthony Gatti `[一作]` (University of Pittsburgh), Amy Babay `[通讯]` (University of Pittsburgh)

**通讯引用:** 204 | [OpenAlex ID](https://openalex.org/A5067242457)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了分布式量子网络熵路由算法Q‑GUARD，在k‑hop局部信息模型下为每条请求提供端到端的Fidelity保障；

**💡 创新点**

创新点包括：① 引入分段期望良好度（EXG）度量和等分/加权分配Purification，实现对硬件异质性的自适应；② 将Purification规划和路径恢复决策集成进分布式调度；③ 在不需要全局状态的前提下实现实时的Fidelity阈值满足；

**🔧 技术方法**

采用的技术有：Q‑CAST框架、Werner状态噪声模型、BBPSSW对称Purification、等分与加权分配算法、扩展Dijkstra搜索、EXG度量等；

**📊 数据集**

使用的数据集为100节点Waxman随机拓扑，硬件质量参数服从正态分布（η∼𝒩(9.5,1.0)），并在每槽中模拟Bell对生成与交换；

**📈 对比分析**

通过与Q‑CAST及其改进版Q‑CAST‑PUR的对比实验，Q‑GUARD在不同Fidelity阈值、负载和硬件异质性条件下平均提升约30%的合格吞吐量，服务半径几乎翻倍；Q‑GUARD‑WS在高异质性场景进一步提升；

**⚠️ 局限性**

局限性包括：仅使用Werner单一噪声模型且假设交换成功率同质；未考虑多槽内存管理和时延对Decoherence的影响；仅在合成拓扑上评估，未在真实网络或硬件校准的仿真器上验证；硬件质量参数的预估仍是理论假设。

---

## 64. Lottery BP: Unlocking Quantum Error Decoding at Scale

**arXiv ID:** 2605.00038 | [PDF](https://arxiv.org/pdf/2605.00038v1)

**作者:** Yanzhang Zhu `[一作]` (University of Central Florida), Di Wu `[通讯]` (University of Central Florida)

**通讯引用:** 14211 | [OpenAlex ID](https://openalex.org/A5100373119)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `14d48e9d-0069-4ad9-996a-1d5968216998` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种引入随机性（Lottery BP）的局部解码器、先行的投票预处理（syndrome vote）、层次化硬件架构（局部BP + 全局OSD）以及一个可扩展的模拟平台 Syndrilla，专门用于大规模量子错误纠错。

**💡 创新点**

创新点包括：① 通过局部随机性打破量子退化，显著提升 BP 的准确度；② 使用投票压缩多轮测量误差的 3D 图，缓解解码积压问题；③ 设计了可在多种 QEC 代码（表面码、环面码、Bivariate Bicycle 码）上实现的层次化解码器架构；④ 构建了跨平台的 PyTorch 解码器模拟器，实现 1–2 阶段加速和统一评估指标。

**🔧 技术方法**

使用的技术包括 Belief Propagation（BP）加随机彩票策略、Ordered-Statistics Decoding（OSD）作为全局二次解码、投票聚合、流水线化的硬件模块（V2C/C2V 转换器、彩票管线、比特排序器），以及基于 GPU 的加速仿真。

**📊 数据集**

数据集主要是三类 QEC 代码：表面码、环面码、Bivariate Bicycle 码，代码距离 d 取 3–13（硬件实现上支持 d ≤ 32），采用标准 phenomenological 噪声模型进行 Monte‑Carlo 评估。

**📈 对比分析**

通过与 BP、Relay‑BP、UF、MWPM、BP+OSD 等基线对比，实验显示：在表面码和环面码上 2–8 阶提升的逻辑误码率，OSD 调用率降低 3–5 个数量级，平均解码延迟降至 25–50 ns，面积和功耗分别比 Micro‑Blossom/AFS/Vegapunk 降低 2–3 订单，整体解码效率提升 1–2 订单。

**⚠️ 局限性**

局限性：仍依赖 BP 的初始实现，对极大代码距离（d>32）及非常低错误率的评估不足；随机彩票策略在硬件实现中需要精细的时序与资源控制；在某些 Bivariate Bicycle 码上逻辑错误率提升有限；模拟器虽然加速，但对量子硬件噪声模型的完整性仍需进一步验证。

---

## 65. Lucid-XR: An Extended-Reality Data Engine for Robotic Manipulation

**arXiv ID:** 2605.00244 | [PDF](https://arxiv.org/pdf/2605.00244v1)

**作者:** Yajvan Ravan `[一作]` (MIT), Ge Yang `[通讯]` (MIT)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了基于XR的物理模拟环境vuer，将仿真与数据采集迁移至XR头显，实现无网络延迟交互，并通过文本提示驱动的Diffusion图像生成管线将虚拟演示转化为多视角、真实感的多模态数据，进而训练出可零射击迁移至未见环境的机器人视觉策略。

**💡 创新点**

①在XR设备上直接运行MuJoCo物理仿真，消除云端延迟；②利用逆运动学绑定实现无代码的人类姿态到机器人姿态的实时重定向；③结合文本提示与语义掩码的Diffusion生成管线生成多样化的多视角数据，支持零射击迁移。

**🔧 技术方法**

MuJoCo WebAssembly、React‑three/fiber、WebXR/WebGL/WebGPU、SE(3)逆运动学、文本提示+Diffusion+语义掩码图像生成、ACT与score‑based denoiser、DETR‑style VAE、FiLM动作去噪等技术。

**📊 数据集**

从虚拟演示中自定义的MuJoCo XML场景（RoboCasa、RoboHive、RoboSuite、Menagerie等）以及由ChatGPT生成的多样化文本提示；真实世界中使用Oculus头显收集的30分钟Teleop数据；以及3D Gaussian采样的厨房扫描用于评估。

**📈 对比分析**

与真实世界Teleop数据收集的基线对比，Lucid‑XR训练的ACT/score‑denoiser策略在未见真实厨房环境中的成功率超过90%，即使在光照、桌布变化下仍保持高成功率；在Sim‑to‑Real评估中，纯合成数据训练的策略与仅用真实数据训练的策略性能相当。

**⚠️ 局限性**

依赖文本提示生成的视觉一致性有限，生成图像与真实感仍有差距；逆运动学重定向在高度不匹配的机器人上可能受限；跨机器人和多种动作的全面评估不足；生成图像与对应文本标签的监督信息利用不足。

---

## 66. ViLegalNLI: Natural Language Inference for Vietnamese Legal Texts

**arXiv ID:** 2605.00116 | [PDF](https://arxiv.org/pdf/2605.00116v1)

**作者:** Nhung Thi-Hong Duong `[一作]` (University of Information Technology), Kiet Van Nguyen `[通讯]` (University of Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了ViLegalNLI——首个越南法律文本自然语言推理（NLI）数据集，并在其上对多类Transformer与指令调优大语言模型进行细粒度评测；

**💡 创新点**

创新点在于：①使用半自动生成框架，利用LLM生成假设并通过多模型交叉验证与artifact消减提升标注质量；②首次在越南法律领域提供大规模NLI基准；③设计跨域、跨规则的评估分析以揭示模型推理瓶颈；

**🔧 技术方法**

技术手段包括：LLM（Gemini‑2.5 Flash、GPT‑4o、DeepSeek‑R1、LLaMA‑4 Scout）进行假设生成与标注校验；Transformer预训练模型（XLM‑R、InfoXLM、PhoBERT、viBERT、CafeBERT、DeBERTa V3）进行微调；指令调优LLM（Gemma‑3、Qwen2.5）进行零/少量示例推理；多轮prompt优化与Fleiss κ评估；

**📊 数据集**

数据集：42,012个由168条官方法令（涵盖27个法律子领域）生成的前提–假设对，平均前提约43个词，假设约44个词；

**📈 对比分析**

评测方法：在8:1:1划分的train/dev/test上对多模型进行Fine‑tune（5轮）与few‑shot（4–5示例）推理，评价指标为Accuracy与macro‑F1；结果显示：few‑shot Qwen2.5在测试集取得最高90.72%准确率和90.64% F1；多语种模型（InfoXLM‑large、XLM‑R‑large）次之；越南本土模型CafeBERT表现优异；零-shot LLM性能相对较低；跨域评估表明模型在域迁移下表现稳健，但仍存在推理缺陷；

**⚠️ 局限性**

局限性：①模型对隐含推理、多步逻辑和低词汇重叠的实例仍易误判；②仍受词汇artifact影响，部分非蕴含被误分类为蕴含；③当前数据集聚焦句对级别，缺乏多段落/文档级推理；④对复杂跨条款依赖的法律推理能力不足；

---

## 67. XekRung Technical Report

**arXiv ID:** 2605.00072 | [PDF](https://arxiv.org/pdf/2605.00072v1)

**作者:** Jiutian Zeng `[一作]` (Alibaba Security AGI Lab), Jin Xu `[通讯]` (Alibaba Security AGI Lab)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了XekRung，一款针对网络安全的8B参数大型语言模型；

**💡 创新点**

创新点在于完整三阶段训练框架（持续预训练、监督微调、群组相对策略优化的多任务强化学习）以及面向安全领域的多维度数据合成与强化学习奖励体系；

**🔧 技术方法**

采用Qwen3 8B基础、持续预训练(CPT)、SFT、GRPO强化学习、ActionRL、RLVR、Agentic RL、自我进化等技术；

**📊 数据集**

使用自研的多源安全语料库（公开安全文档、内部产品日志、代码仓库、CTF轨迹、漏洞自生成等）以及合成生成的安全知识驱动问答；

**📈 对比分析**

与同等规模的通用与安全专用模型（如Qwen3-8B、Llama-3.1-8B、Foundation‑Sec‑8B‑Reasoning、SecGPT‑14B）对比，XekRung在15项安全基准上平均提升9.03个百分点，且在通用基准上保持72.54%整体得分，证明在保持通用能力的前提下实现了同规模超越与大规模模型竞争的效果；

**⚠️ 局限性**

局限性包括对更大参数规模的验证不足、部分代理式长链任务仍未充分覆盖、对极端稀有安全场景的自我进化可能受限、以及模型在对抗性生成安全信息时仍可能产生误报或过度自信。

---

## 68. Exploring LLM biases to manipulate AI search overview

**arXiv ID:** 2605.00012 | [PDF](https://arxiv.org/pdf/2605.00012v1)

**作者:** Roman Smirnov `[一作]` `[通讯]` (E-AI.Solutions), Roman Smirnov (E-AI.Solutions)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM在搜索结果概览（LLM Overview）中的偏见，并利用强化学习训练重写策略以操纵LLM对片段的选择与引用。

**💡 创新点**

首次证明LLM Overview偏好是相对优势而非绝对特性，并提出仅改写短片段且限制长度的RL训练框架，探讨了对抗性攻击与安全风险。

**🔧 技术方法**

采用Dr.GRPO（GRPO变体）强化学习算法，结合LoRA/QLoRA微调、嵌入相似度、长度惩罚以及GPT‑4.1‑nano/mini/5‑nano作为奖励模型。

**📊 数据集**

使用Amazon购物查询数据集（约3000条查询），每条查询获取10条搜索结果，构造90个含7‑10条结果的测试样本。

**📈 对比分析**

通过对不同随机化技术（如shuffle URL、title、snippet等）的比较，实验显示在GPT‑4.1‑nano/mini上重写策略能提升被引用率约10‑30%，而gpt‑5‑mini对URL偏好强，影响较小。

**⚠️ 局限性**

仅改写片段忽略标题/URL导致攻击效果受限；RL收敛不稳，模型差异显著影响鲁棒性；对抗攻击验证范围有限。

---

## 69. FieryGS: In-the-Wild Fire Synthesis with Physics-Integrated Gaussian Splatting

**arXiv ID:** 2605.00177 | [PDF](https://arxiv.org/pdf/2605.00177v1)

**作者:** Qianfan Shen `[一作]` (Peking University), Baoquan Chen `[通讯]` (Peking University)

**通讯引用:** 14201 | [OpenAlex ID](https://openalex.org/A5010714340)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种全新的、可控的、物理基础的火焰合成框架 FieryGS，能够在从多视角图像恢复的 3D Gaussian Splatting（3DGS）场景中自动生成光照真实、物理可行的燃烧效果，支持火焰强度、风向、点火位置等参数的细粒度调控。

**💡 创新点**

创新点包括：① 将多模态大型语言模型（MLLM）用于 3DGS 区域的材料推理，零样本推断燃烧相关属性；② 在 3DGS 的体积网格上进行可控的火焰与木材焦化仿真，避免传统 CFD 的手工几何和参数；③ 设计统一的体积渲染管线，将 3DGS、火焰、烟雾以及焦化效果无缝融合；④ 引入可选的扩散模型后处理提升细节与光照。

**🔧 技术方法**

技术主要包括 3D Gaussian Splatting（PGSR）、MLLM（GPT‑4o）材料推理、基于能量守恒的火焰与焦化体积仿真、基于黑体辐射的火焰渲染、烟雾颜色依材质的体积渲染、Phong 光照、可选的 Wan2.1 视频扩散后处理。

**📊 数据集**

使用了六个真实世界场景（Firewood、Kitchen、Chair、Stool、Garden、Playground）以及公开数据集 MipNeRF360、Tanks & Temples 的多视角图像。

**📈 对比分析**

与 AutoVFX、Runway‑V2V、Instruct‑GS2GS 三种基线进行定性和定量对比。实验显示 FieryGS 在 VBench 的美学质量与图像质量上分别取得 0.624 与 0.702 分（均高于 0.605/0.701 的最优对手），DINO 结构保持度下降到 0.38，说明更好地保留原始结构；用户研究中，超过 80% 的受试者更倾向于 FieryGS 的视觉真实性与物理可信度。渲染速度为每帧约 2.37 秒（RTX 4090）。

**⚠️ 局限性**

局限性包括：① 未显式模拟质量损失、热降解及大规模火灾的细节；② 受 3DGS 点云分布不均影响，易产生视觉伪影；③ MLLM 推理错误可能导致材料属性不准确，进而影响燃烧行为；④ 对极大场景（如森林大火）扩展性仍待验证。

---

## 70. The Impact of Approximation on Algorithmic Progress

**arXiv ID:** 2605.00220 | [PDF](https://arxiv.org/pdf/2605.00220v1)

**作者:** Jeffery Li `[一作]`, Neil Thompson `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

无内容

**💡 创新点**

无内容

**🔧 技术方法**

无内容

**📊 数据集**

无内容

**📈 对比分析**

无内容

**⚠️ 局限性**

无法完成摘要，缺乏论文具体信息

---

## 71. Compatible $k$-Relaxations of Fairness and Non-Wastefulness Under Hereditary Constraints

**arXiv ID:** 2605.00134 | [PDF](https://arxiv.org/pdf/2605.00134v1)

**作者:** Tenma Wakasugi `[一作]` (Kyushu University), Makoto Yokoo `[通讯]` (Kyushu University)

**通讯引用:** 9389 | [OpenAlex ID](https://openalex.org/A5048575057)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文在 hereditary constraints 下研究两边匹配市场，提出通过参数 k 同时放宽公平（ER‑k）与非浪费（NW‑k）两项约束，并证明对任意固定 k 都存在满足这两项约束的匹配。

**💡 创新点**

创新点在于：①首次将公平与非浪费以相同的参数 k 对称松弛，形成 ER‑k 与 NW‑k；②证明 ER‑k 与 NW‑k 在 hereditary constraints 下始终兼容；③设计了两种多项式时间算法——k‑admissible cutoff 与 k‑admissible college‑proposing DA——可直接求得满足 ER‑k 与 NW‑k 的匹配。

**🔧 技术方法**

使用了：可行性函数 f 与 hereditary 约束框架；k‑admissibility 定义与 k‑admissible cutoff 以及基于 DA 的 k‑admissible 匹配；整数规划思路用于解释存在性与算法的正确性；以及对公平与非浪费的阶梯化度量。

**📊 数据集**

数据集方面采用合成实例：基于 Mallows 模型生成学生偏好与院校优先级，使用区域配额（regional quota）作为 hereditary 约束，设置多种容量分布（固定、均匀、正态）和不同的地区数。

**📈 对比分析**

对比方法：与原始 cutoff（k=0）和理论最优界面进行比较，实验结果表明即使较小的 k（如 5‑10）即可在公平（ER‑k）与非浪费（NW‑k）之间取得良好折中；在不同偏好相关性与容量设置下，k‑admissible 算法在保证可行性与兼容性上表现稳定，且计算时间保持多项式。

**⚠️ 局限性**

局限性：未给出在 ER‑k 与 NW‑k 下的学生最优匹配；未探讨算法的策略不变性（如截断策略不变性）与更强的激励兼容性；NW‑k 形式尚未完全转化为整数规划模型；实验仅基于合成数据，缺乏真实世界案例验证。

---

## 72. Learning Fingerprints for Medical Time Series with Redundancy-Constrained Information Maximization

**arXiv ID:** 2605.00130 | [PDF](https://arxiv.org/pdf/2605.00130v1)

**作者:** Huayu Li `[一作]` (University of Arizona), Ao Li `[通讯]` (University of Arizona)

**通讯引用:** 122584 | [OpenAlex ID](https://openalex.org/A5100743975)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出TS‑Fingerprint，一种面向医疗时间序列的自监督表示学习框架，利用跨注意力瓶颈将可变长度信号压缩成固定数量的正交Fingerprint Token，并通过重构损失与总编码率（TCR）正则化实现信息最大化与冗余最小化。

**💡 创新点**

创新点在于：① 将表示学习视为“冗余约束信息最大化”问题，理论上证明最小化总相关性可提升下游样本效率；② 设计了固定槽位的自注意力瓶颈，使token之间竞争信息并自然形成解耦；③ 通过TCR正则化实现token的几何正交性，实现可解释、稀疏的诊断特征。

**🔧 技术方法**

核心技术包括：交叉注意力压缩编码器、掩码自回归解码器、重构损失（MSE）与多重相关性（TC/TCR）正则化、基于注意力的下游分类投影；实现基于Transformer的6层编码器和2层解码器，使用Adam优化，mask比例0.6，token数k=8。

**📊 数据集**

使用多种公开医疗时间序列数据集评估：ECG（PTB‑XL、PTB）、EEG（ADFTD）、活动识别（APAVA、UCI‑HAR）、睡眠（SleepEDF）、多类别信号（FLAAP），涵盖5种评价指标（Accuracy、Precision、Recall、F1、AUROC）。

**📈 对比分析**

与多种基线（PatchTST、Medformer、Ti‑MAE、SimMTM、Autoformer等）做对比，TS‑Fingerprint在所有指标上均取得最优或次优平均排名1.24，显著优于传统MAE与Transformer，且在低样本、跨域转移中表现更为稳健。Ablation实验表明加入TCR正则化可提升13% F1，参数鲁棒性良好。

**⚠️ 局限性**

局限性包括：① 对mask比例与token数的选择仍需经验调优；② 目前实验集中在单一信号模态，跨模态泛化仍待验证；③ 需要大量GPU资源进行预训练；④ 虽然实现解耦，但具体token与医学特征的对应关系仍需专家解读，解释性仍有提升空间。

---

## 73. Delay-Doppler Domain Channel Estimation: What if Sparsity is Unknown?

**arXiv ID:** 2605.00049 | [PDF](https://arxiv.org/pdf/2605.00049v1)

**作者:** Zijian Yang `[一作]` (University of Macau), Shaodan Ma `[通讯]` (University of Macau)

**通讯引用:** 6194 | [OpenAlex ID](https://openalex.org/A5053586699)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种不需要预先知道延迟和多普勒稀疏度的延迟‑多普勒域通道估计方法，利用 Cartesian‑product 结构直接从观测中估计支持维度并通过 BIC 进行模型选择，最终得到最小二乘估计。

**💡 创新点**

创新点在于：① 彻底去除了对稀疏度预算的假设，真正实现“稀疏性无知”；② 通过利用所有活跃延迟共享同一多普勒集的几何结构显著减少搜索空间；③ 使用 BIC 作为自适应的模型选择准则，兼顾数据拟合与模型复杂度。

**🔧 技术方法**

核心技术包括：延迟‑多普勒域的 on‑grid 表示、正则化最小二乘（ridge）初始估计、基于能量排序的候选支持生成、BIC 评估与全格搜索以及最终的最小二乘重估。

**📊 数据集**

实验数据基于 AFDM 系统的模拟通道，参数为 N=4096, L=30, Q=7, 并采用 Bernoulli 活动模型（p_d=p_D=0.2）产生随机稀疏支持；通过 5000 次 Monte‑Carlo 采样评估性能。

**📈 对比分析**

与固定预算（均值与扩容）、稀疏贝叶斯学习（SBL）以及 Oracle‑LS 进行比较。结果显示：提出的方法在整个 SNR 范围内几乎达到 Oracle‑LS 的 NMSE，明显优于固定预算基线并显著优于 SBL；在高 SNR 时几乎无误差上限，支持恢复成功率高。

**⚠️ 局限性**

局限性：① 依赖共享多普勒假设，若实际通道多普勒分布不共享可能导致性能下降；② 对大规模网格（高 L、B）仍需遍历所有 (d,r) 组合，计算量随维度增长；③ 仅在模拟 AFDM 上验证，缺乏实测数据验证。

---

## 74. MAEPose: Self-Supervised Spatiotemporal Learning for Human Pose Estimation on mmWave Video

**arXiv ID:** 2605.00242 | [PDF](https://arxiv.org/pdf/2605.00242v1)

**作者:** Xijia Wei `[一作]` (University College London), Nadia Bianchi-Berthouze `[通讯]` (University College London)

**通讯引用:** 8582 | [OpenAlex ID](https://openalex.org/A5072897829)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于掩码自编码的毫米波视频姿态估计框架 MAEPose，直接在原始雷达频谱视频上学习时空表示。

**💡 创新点**

创新点在于将掩码自编码应用于毫米波视频进行无监督预训练，并结合多帧热图解码器保留空间对应关系，实现更准确的姿态回归。

**🔧 技术方法**

采用视频视觉 Transformer（ViT）作为编码器、掩码自编码预训练、以及 3D 卷积热图解码器；并使用 90% 时空掩码比例。

**📊 数据集**

使用三大室内毫米波姿态数据集 mmTryOn、mmMove 和 mmYoga，涵盖不同人物、动作与环境。

**📈 对比分析**

通过留一人交叉验证与统计检验，MAEPose 在 MPJPE 上比最新基线降低约 20–25%，PCK@5cm 提升超过 10%，且在零样本干扰场景下误差仅上升 6.5%。

**⚠️ 局限性**

局限在于对超高动态或低运动幅度场景的鲁棒性仍有限，且模型对跨环境泛化需要少量标注微调。

---

## 75. Information-Theoretic Generalization Bounds for Stochastic Gradient Descent with Predictable Virtual Noise

**arXiv ID:** 2605.00064 | [PDF](https://arxiv.org/pdf/2605.00064v1)

**作者:** Mohammad Partohaghighi `[一作]` (University of California, Merced), Mohammad Partohaghighi `[通讯]` (University of California, Merced)

**通讯引用:** 519 | [OpenAlex ID](https://openalex.org/A5038752166)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种可预测的历史自适应虚拟扰动框架，允许在SGD的分析过程中使用随训练历史变化的协方差矩阵，而不改变实际的SGD迭代。

**💡 创新点**

创新点在于：①引入可预测性条件（协方差只依赖过去历史而不依赖当前或未来随机性），②构造条件高斯相对熵上界，③引入协方差比较成本以及可接受的参考核（S‑admissible）设计，从而得到包含梯度偏差、梯度敏感度与输出敏感度的完整信息理论泛化界。

**🔧 技术方法**

主要技术：信息理论泛化理论（互信息与子高斯损失的关系）、条件高斯相对熵不等式、互信息链式规则、KL散度分解、可预测随机过程的可测性、以及对协方差比较成本的显式估计。

**📊 数据集**

论文为理论性研究，没有使用具体数据集；讨论的结果可推广到任何满足子高斯损失假设的监督学习任务。

**📈 对比分析**

与传统固定噪声的虚拟扰动分析相比，该框架通过自适应协方差能够更好地捕捉训练过程中的梯度尺度与曲率变化，理论上能得到更具路径依赖性的泛化界；在可同步（admissible）情形下，所得界与已有的固定噪声界等价；在非同步情形下，额外的协方差比较成本被量化，但仍保持可验证性。

**⚠️ 局限性**

局限性包括：需要可接受的参考核（S‑admissible）才能消除协方差比较成本；预测性条件虽然宽松，但若协方差高度依赖样本，协方差比较成本可能显著；框架仅适用于分析阶段，未对实际SGD算法进行改动；实验验证不足，主要以理论推导为主。

---

## 76. SPLICE: Latent Diffusion over JEPA Embeddings for Conformal Time-Series Inpainting

**arXiv ID:** 2605.00126 | [PDF](https://arxiv.org/pdf/2605.00126v1)

**作者:** Arnaud Zinflou `[一作]` `[通讯]` (Hydro-Qu\'ebec Research Institute), Arnaud Zinflou (Hydro-Qu\'ebec Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出了SPLICE框架，实现了自监督预测潜在空间填补与自适应置信区间的结合。

**💡 创新点**

创新点在于将JEPA表示、可调噪声潜在桥、流匹配采样和自适应共形推断模块化，并在长周期缺失段实现覆盖保证。

**🔧 技术方法**

使用JEPA编码、可调噪声潜在桥、流匹配扩散、每小时条件解码器和Adaptive Conformal Inference等技术。

**📊 数据集**

在13个电力负荷数据集（9个私有、3个UCI、电力ETTh1）上进行评估。

**📈 对比分析**

与五个外部基线（BRITS、SAITS、CSDI、TimesNet、Seasonal Naïve）和内部消融对照，SPLICE在91天缺失段的均值Load-only MSE为0.056，胜率9/12，区间覆盖93–95%，CRPS最低。

**⚠️ 局限性**

局限在于对极短或极长缺失段未评估，流匹配仍需多步，零射传输需桥调优，跨域泛化对目标分布差异敏感。

---

## 77. What Physics do Data-Driven MoCap-to-Radar Models Learn?

**arXiv ID:** 2605.00018 | [PDF](https://arxiv.org/pdf/2605.00018v1)

**作者:** Kevin Chen `[一作]` (Ohio State University), Anish Arora `[通讯]` (Ohio State University)

**通讯引用:** 7638 | [OpenAlex ID](https://openalex.org/A5079903777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了一种基于雷达物理的解释框架，用以评估仅靠数据驱动的 MoCap‑to‑Radar 模型在生成微多普勒谱时是否真正学习了物理规律，并设计了两种无需雷达数据的物理一致性度量。

**💡 创新点**

创新点在于揭示低重建误差不一定对应物理正确性，提出了基于 Doppler 中心轨迹和速度‑频率比例关系的 FVA 与 DCS 两个指标，并证明时序注意力是学习雷达物理的关键。

**🔧 技术方法**

技术上采用了 RCS‑感知 Doppler 中心模型与速度‑频率比例模型，基于 STFT 计算谱心，使用 Pearson 相关得到 FVA，利用比例回归得到 DCS，并对 Transformer 与 MLP 结构进行实验对比。

**📊 数据集**

使用了单人 MoCap2Radar 数据集，该数据集包含 53 个 Vicon 标记与 5.8 GHz 微多普勒雷达同步采集的运动数据，分为训练、验证和仅用于测试的自由行走轨迹。

**📈 对比分析**

通过在测试集上同时计算 MAE、FVA 与 DCS，发现具有时序注意力的 Transformer 在物理一致性上远优于无注意力或 MLP 模型，尽管 MAE 仅相差 0.3 dB；而 FVA 与 DCS 的差距可达 0.6 以上。

**⚠️ 局限性**

局限性包括仅在单一被试上验证，物理一致性评估仅聚焦于谱心与比例关系，未涉及更丰富的频谱特征；此外指标只能说明模型是否符合所用物理参考，无法揭示内部学习机制。

---

## 78. Being-H0.7: A Latent World-Action Model from Egocentric Videos

**arXiv ID:** 2605.00078 | [PDF](https://arxiv.org/pdf/2605.00078v1)

**作者:** Hao Luo `[一作]`, Zongqing Lu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于隐式世界-动作模型（Being-H0.7），在不做像素级未来预测的情况下，通过在多模态输入与动作输出之间插入可学习的隐层查询，实现未来感知的内部推理与动作生成的无缝衔接。

**💡 创新点**

核心创新在于：① 引入双分支（prior/ posterior）对齐机制，让训练时的未来观测仅作为隐层正则化信号；② 通过隐层对齐、范数与谱正则化防止隐层崩塌；③ 在推理时完全去掉未来帧生成，保持 VLA 直接动作推断的低延迟与高效性。

**🔧 技术方法**

技术包括 Transformer‑MoT 架构、可学习的隐层查询、未来嵌入编码（ViT+Perceiver Resampler）、流匹配动作损失、双分支注意力掩码、范数与谱正则化、以及 UAC（延迟感知异步块化）部署框架。

**📊 数据集**

预训练使用大规模人类与机器人操作混合数据（如 Ego4D、Ego-Exo4D、EPIC‑KITCHENS 等），随后在 6 种模拟基准（LIBERO、RoboCasa、GR1、LIBERO‑plus、RoboTwin 2.0、CALVIN）和 3 种真实机器人平台（PND Adam‑U、Unitree G1、Franka FR3）上微调与评估。

**📈 对比分析**

与多种基线（VLA、World‑Action、Fast‑WAM、LingBot‑VA 等）对比，Being‑H0.7 在 6 个模拟基准中均取得最高或近似最高平均成功率（如 LIBERO 99.2%、CALVIN 4.67/4.48 任务完成率），在 3 个真实平台的 12 个任务中 5 大能力组均名列前茅，且推理延迟仅 3–4 ms/步。

**⚠️ 局限性**

局限性：仍需依赖大量人类视频预训练；隐层推理虽省去像素级生成，但对极高时间分辨率的动态任务仍可能受限；对非常复杂的物理交互（如高度刚性碰撞、多体动力学）需要进一步验证。

---

## 79. DPU or GPU for Accelerating Neural Networks Inference -- Why not both? Split CNN Inference

**arXiv ID:** 2605.00174 | [PDF](https://arxiv.org/pdf/2605.00174v1)

**作者:** Ali Emre Oztas `[一作]` (University of Manchester), Mikel Luj'an `[通讯]` (University of Manchester)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出将CNN推理拆分在Versal FPGA DPU和NVIDIA RTX2080 GPU之间，形成流水线式的Split CNN Inference。

**💡 创新点**

创新点在于细粒度层级的分区、异步流水线并行以及使用图神经网络预测最佳分区点。

**🔧 技术方法**

采用DPU的AI Engine、GPU的FP16计算、Vitis AI编译、PCIe传输以及GNN模型（GCN+LSTM）进行预测。

**📊 数据集**

使用ResNet18/50/101/152、VGG16、MobileNetV2、LeNet-5等公开模型（ImageNet 2012子集）作为实验数据集。

**📈 对比分析**

与单独在DPU或GPU上推理进行比较，Split CNN 在不同模型上实现最高 3.37× 的速度提升（对GPU）和 2.48×（对DPU）。

**⚠️ 局限性**

局限性包括对特定硬件对（Versal+RTX2080）的依赖、需要先编译分区模型、并且GNN预测需要针对每对设备训练，且对极浅层网络如LeNet收益有限。

---

## 80. RouteProfile: Elucidating the Design Space of LLM Profiles for Routing

**arXiv ID:** 2605.00180 | [PDF](https://arxiv.org/pdf/2605.00180v1)

**作者:** Jingjun Xu `[一作]` (University of Illinois Urbana Champaign), Ge Liu `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文系统研究了大型语言模型（LLM）路由中的模型Profiling设计空间，探讨不同Profile设计对路由质量及新模型冷启动泛化的影响。

**💡 创新点**

创新点在于：①将LLM Profiling视为结构化信息整合问题，提出四维设计空间RouteProfile（组织形式、表示类型、聚合深度、学习配置）；②构建异构交互图并在此上进行多维聚合；③在三种代表性路由器（SimRouter、MLPRouter、GraphRouter）上进行统一实验，揭示结构化+可训练Profile在路由与冷启动中的优势。

**🔧 技术方法**

技术手段包括：异构图建模、文本与嵌入两种表示、文本GNN与Embedding GNN多跳消息传递、训练free与trainable（自监督掩码重构）聚合函数、以及三种路由器的实现。

**📊 数据集**

使用了15个跨知识、推理、数学、编程等领域的基准数据构建交互图，12个下游评估数据集用于路由性能测试，涵盖25个LLM（其中8为候选模型）来丰富图结构。

**📈 对比分析**

通过平均回答质量和冷启动成功率评估路由效果。实验显示：结构化Profile显著优于平面Profile；查询级信号比域级信号更可靠；可训练的结构化Profile在新LLM冷启动中表现最佳，且不同路由器与Profile的配合也决定最终性能。

**⚠️ 局限性**

局限性包括：仅在有限的8个候选模型上验证，扩展性待考；未深入探讨模型运行成本和实时性约束；自监督训练在大规模异构图上的可扩展性和收敛性未完全评估；对多任务/多领域通用性仍需进一步研究。

---

## 81. Lightweight Tamper-Evident Log Integrity Verification for IoT Edge Environments: A Merkle Tree Pipeline with Adaptive Chunking

**arXiv ID:** 2605.00065 | [PDF](https://arxiv.org/pdf/2605.00065v1)

**作者:** Muhammet Anil Yagiz `[一作]` (Kırıkkale University), Ahmet Hasim Yurttakal `[通讯]` (Afyon Kocatepe University)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5026574666)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套轻量级 IoT 日志完整性验证流水线，利用 Merkle 树与自适应分块技术在单节点上实现无区块链的 tamper‑evidence 方案。

**💡 创新点**

创新点包括：1) 将 Merkle 树与资源感知的批处理结合，兼顾吞吐与内存限制；2) 对实现代码进行严格审计，修正了两处隐藏缺陷；3) 在单台工作站上实现 >130,000 logs/s 的吞吐、<5 MB 的峰值内存和约 22 ms 的单条验证延迟；4) 提供完整的安全分析与多维度基准对比。

**🔧 技术方法**

采用了 Merkle 树、SHA‑256 / BLAKE2b 哈希、自适应内存感知分块策略、单根可信锚（TPM 或远程 append‑only 存储）以及 Python 实现。

**📊 数据集**

使用合成的 IoT 日志数据集，日志字段模拟设备监控与事件信息，规模覆盖 1,000 至 100,000 条，所有数据可通过同一随机种子复现。

**📈 对比分析**

通过 5 次独立跑测评，比较自适应分块与固定分块的吞吐、SHA‑256 与 BLAKE2b 的吞吐与验证延迟，并与 key‑evolution、签名、区块链、TPM 等代表性方案在吞吐、验证成本、网络依赖与边缘可部署性四维进行对比。性能结果显示：>130,000 logs/s 吞吐、≈22 ms/条验证、≈1 KB 证明尺寸、<5 MB 内存峰值，显著优于传统区块链或签名方案。

**⚠️ 局限性**

局限性包括：仅使用合成日志，未在真实工业 IoT 生产环境中验证；实验仅在高性能工作站上完成，未在 ARM 等边缘设备上评估；未进行对抗性压力测试；当前实现每批重建树而非增量插入；未实现多设备根同步与分布式可信锚扩展。

---

## 82. Alignment Contracts for Agentic Security Systems

**arXiv ID:** 2605.00081 | [PDF](https://arxiv.org/pdf/2605.00081v1)

**作者:** Isaac David `[一作]` (UCL), Arthur Gervais `[通讯]` (UCL)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出对具备攻击能力的LLM代理的“对齐合约”，通过可观察的效果边界来规范其权限

**💡 创新点**

首次将合约细化为可观察效果的安全属性，并用Lean 4证明监视器的可执行性与保真性

**🔧 技术方法**

形式化模型、可观察效果假设、参考监视器、可决算性检查与合约代数

**📊 数据集**

无实验数据集，主要在理论层面和Lean证明上验证

**📈 对比分析**

与传统安全DSL、访问控制语言对比，展示了在攻击者模型下仍能保证合同满足，性能上以可观测效果的实时检测为主

**⚠️ 局限性**

受限于完整的效果可观察性、无法防御隐藏在payload中的信息泄漏、不可实现完全的预先检查及时间通道

---

## 83. Minimal, Local, Causal Explanations for Jailbreak Success in Large Language Models

**arXiv ID:** 2605.00123 | [PDF](https://arxiv.org/pdf/2605.00123v1)

**作者:** Shubham Kumar `[一作]` (University of Illinois Urbana-Champaign), Narendra Ahuja `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 31556 | [OpenAlex ID](https://openalex.org/A5108521995)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LOCA 方法，通过在中间层对少量可解释的激活方向进行逐步修补，来局部解释和重现 jailbreak 攻击的成功与否。

**💡 创新点**

创新点在于：① 将激活修补与 token‑specific 的一阶梯度近似相结合；② 引入迭代算法在每次修补后重新计算梯度，从而考虑 token 之间的交互；③ 设计了适用于不同长度、格式 jailbreak prompt 的 token‑matching 方案，使得可修补的 token 能一一对应。

**🔧 技术方法**

技术手段包括：基于 Sparse Autoencoder（SAE）的可解释方向提取、激活修补（activation patching）、逐 token 的一阶梯度近似、以及迭代式最小化 KL 散度的自适应修补。

**📊 数据集**

使用了 WhatFeatures（约10.8k 个 jailbreak 攻击）与 HarmBench 自动评分器；同时利用 GemmaScope 和 Llama 的预训练 SAE 作为可解释方向的来源。

**📈 对比分析**

与前沿的激活修补/steering 方法对比，LOCA 在 Gemma 与 Llama 模型上平均仅需 6‑8 次修补即可使 100% 的 jailbreak 成功转为拒绝（RR 100%），而对照方法在 20 次修补后仍无法达到同样效果；其 KL‑AUC、LD‑AUC 及最小修补次数指标均显著优于基线。

**⚠️ 局限性**

局限性包括：对早期层的解释仍耗时且易出错；SAE 提供的方向可能并非最优，导致解释的准确性受限；在某些 jailbreak 场景下，LOCA 仍无法在限定的修补次数内恢复拒绝。

---

## 84. Real-Time Frame- and Event-based Object Detection with Spiking Neural Networks on Edge Neuromorphic Hardware: Design, Deployment and Benchmark

**arXiv ID:** 2605.00146 | [PDF](https://arxiv.org/pdf/2605.00146v1)

**作者:** Udayanga G. W. K. N. Gamage `[一作]`, Silvia Tolu `[通讯]` (Technical University of Denmark)

**通讯引用:** 606 | [OpenAlex ID](https://openalex.org/A5048001312)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了三种轻量级的时序神经网络（SNN）架构，用于在Intel Loihi 2硬件上实现实时帧级和事件级目标检测；

**💡 创新点**

提出了基于ANN‑to‑SNN知识蒸馏的直接训练框架，使SNN在仅8个时间步内恢复87–100%的ANN检测精度，并兼顾硬件约束（无分支、无池化、平均批归一化）以适配Loihi 2；

**🔧 技术方法**

使用了Leaky Integrate‑and‑Fire（LIF）神经元、硬重置机制、RepVGG重参数化、8位量化感知训练、分布式焦点损失（Distribution Focal Loss）以及自定义的特征、分类、回归蒸馏损失；

**📊 数据集**

在四类数据集上进行评估：事件数据集evCIVIL‑ev（裂纹与剥落）与Prophesee GEN1（车辆与行人）；帧数据集evCIVIL‑fr与PASCAL VOC；以及新收集的无人机灰度隧道检查视频数据集；

**📈 对比分析**

通过在Loihi 2、NVIDIA Jetson Nano B01、Jetson Orin Nano及Apple M2 CPU上进行统一基准，比较检测准确率、推理率、动态能耗与能延迟乘积（EDP）。结果显示，SNN在Loihi 2上实现实时推理（62–170 fps），动态能耗比Jetson平台低10–35倍，EDP低1.5–10倍；但与Jetson Orin Nano相比推理率低1.3–2.6倍；

**⚠️ 局限性**

主要局限包括：Loihi 2仅支持均值批归一化导致精度下降；硬件不支持分支与Conv‑LSTM，限制网络深度与多尺度检测；以及缺乏高速I/O，未考虑完整系统能耗与延迟。

---

## 85. Technical Report: Activation Residual Hessian Quantization (ARHQ) for Low-Bit LLM Quantization

**arXiv ID:** 2605.00140 | [PDF](https://arxiv.org/pdf/2605.00140v1)

**作者:** YiFeng Wang `[一作]` (Tohoku University), Keisuke Sakaguchi `[通讯]` (Tohoku University)

**通讯引用:** 2363 | [OpenAlex ID](https://openalex.org/A5101067919)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种后训练量化方法ARHQ，通过将权重分成主分支和低秩高精度侧分支来抑制激活量化残差的放大，提升低位激活-权重量化的推理精度。

**💡 创新点**

创新点在于使用激活残差协方差（Residual Hessian）G_x 对权重进行加权低秩分解，并给出闭式SVD解法，解决了传统激活能量或离散量化误差方法无法精准捕捉残差放大问题的局限。

**🔧 技术方法**

核心技术包括激活残差量化残差统计、残差协方差构造、加权低秩近似、闭式截断SVD求解、低秩因子化部署、数值正则化及可选的激活平滑预处理。

**📊 数据集**

在Qwen3‑4B‑Thinking‑2507模型的36层注意力投影上进行验证，并在小规模140题ZebraLogic子集上评测生成任务。

**📈 对比分析**

与传统无平滑SVD、SVD+平滑以及AWQ4等方法对比，ARHQ在层级SNR上提升约1.8–2.0 dB，提升量化基线SNR约3–6 dB，并在ZebraLogic任务中实现最高量化准确率（与bf16相当）。

**⚠️ 局限性**

局限包括：仅针对激活残差放大优化，未同时考虑权重量化误差；需要与量化硬件和校准数据高度匹配；固定低秩秩度且全协方差计算成本高；未实现联合平滑与分解的最优调优。

---

## 86. Why Do LLMs Struggle in Strategic Play? Broken Links Between Observations, Beliefs, and Actions

**arXiv ID:** 2605.00226 | [PDF](https://arxiv.org/pdf/2605.00226v1)

**作者:** Jan Sobotka `[一作]` (EPFL), Ufuk Topcu `[通讯]` (University of Texas at Austin)

**通讯引用:** 10341 | [OpenAlex ID](https://openalex.org/A5068441112)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

分析LLM在不完全信息游戏中的内部信念与行动决策机制，揭示观察-信念差距与信念-行动差距。

**💡 创新点**

首次将机制可解释性工具用于量化LLM内部信念的准确性与其对行动的影响，揭示多跳推理脆弱、贝叶斯一致性衰减与首项偏差等系统性漏洞。

**🔧 技术方法**

使用线性探针、激活驱动、贝叶斯一致性系数、PCA、逆向干预等机制可解释技术。

**📊 数据集**

在三类不完全信息游戏的自生成对局数据上评估：重复常规游戏、广义库恩扑克、变色龙推理游戏。

**📈 对比分析**

与随机/多数基线、内部/外部推断、贝叶斯一致性曲线、驱动实验对比；内部探针在多数指标上显著优于外部；贝叶斯一致性随回合递减；驱动能提升约70%但仍远低于最优；显式信念未能提升收益。

**⚠️ 局限性**

仅使用公开模型、线性探针可能低估复杂推理；实验仅覆盖有限游戏；未提出有效对策；对长周期对抗和现实任务的泛化性不足。

---

## 87. Matroid Algorithms Under Size-Sensitive Independence Oracles

**arXiv ID:** 2605.00201 | [PDF](https://arxiv.org/pdf/2605.00201v1)

**作者:** Kiarash Banihashem `[一作]` (University of Maryland), Danny Mittal `[通讯]` (University of Maryland)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的大小敏感独立性查询模型，研究了在该模型下的基于矩阵的算法，包括寻找矩阵的基、近似其秩和近似其分区大小。

**💡 创新点**

创新点在于引入了大小敏感的查询成本模型，揭示了传统常数时间查询假设与实际计算成本之间的差距，并提供了更准确的复杂性分析。

**🔧 技术方法**

使用了大小敏感的独立性查询模型，分析了在该模型下的算法性能，并提出了具体的算法实现。

**📊 数据集**

使用了多种矩阵，包括图形矩阵、分区矩阵等，进行实验和理论分析。

**📈 对比分析**

通过与传统模型下的算法进行比较，证明了在大小敏感查询模型下，矩阵的秩估计和分区大小的计算都需要二次查询成本，而对于具有有限周长的矩阵，能够打破这一二次成本的限制。

**⚠️ 局限性**

限制在于该模型的复杂性分析主要集中在特定类型的矩阵上，尚未全面覆盖所有可能的矩阵类型和查询成本函数。

---

## 88. Diversity in Large Language Models under Supervised Fine-Tuning

**arXiv ID:** 2605.00195 | [PDF](https://arxiv.org/pdf/2605.00195v1)

**作者:** Roman Klypa `[一作]` (CNRS, University of Grenoble Alpes), Oleksandr Cherednichenko `[通讯]` (Umeå University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对监督微调（SFT）导致大型语言模型（LLM）生成多样性下降的现象进行系统性研究，归因于低频样本被忽视和预训练知识被遗忘，并提出结合 GEM 与 Focal Loss 的 Tempered Focal（TOFU）损失，随后在多种模型与数据集上对比评估其效果。

**💡 创新点**

①提出 TOFU 这一新颖的损失函数，既缓解预训练知识遗忘，又提升对低频样本的关注；②在大规模、多模型、多任务的实验设置下首次对 SFT 对多样性的影响进行定量验证；③通过理论梯度分析证明 TOFU 等价于温度化交叉熵加权，阐明其对多样性和质量的双重影响。

**🔧 技术方法**

采用监督微调（SFT）框架，结合 GEM、Focal Loss 与自定义的 TOFU 损失；利用梯度分析、温度化交叉熵、低秩适配（QLoRA）与 4‑bit NF4 量化；评估时使用 Self‑BLEU、Distinct、LLM Judge、Utility、Pass@k、ASR 等多维指标。

**📊 数据集**

主要使用的指令与对话数据集：Alpaca、UltraFeedback；数学推理数据集：NuminaMath‑CoT、MATH500、GSM8K、MinervaMath；开放式生成任务：Short Stories、Small Prompts、NoveltyBench；事实性与安全性评测：ARC、MMLU、Malicious Instruct、HarmBench。

**📈 对比分析**

通过与交叉熵（CE）、GEM、λ‑PR、Focal Loss 四种基线进行横向对比。实验结果表明，TOFU 在 Self‑BLEU、Distinct 等多样性指标上优于所有基线，同时保持或提升质量评分；在数学推理任务中显著提升 Pass@k 覆盖率；事实性和安全性指标保持不变，甚至在安全模型上不增加攻击成功率。整体来看，TOFU 在保持质量与安全的前提下实现了最大的多样性提升。

**⚠️ 局限性**

局限性：①仅在中等规模（≤ 13B）模型上验证，尚未在更大规模模型上评估；②缺乏对 RL 对齐阶段后续效果的实证；③TOFU 仅对 one‑hot 目标有效，无法直接应用于蒸馏或多标签任务；④安全性评测聚焦于有限的红队任务，未覆盖所有潜在的恶意场景；⑤超参数主要在 ARC 数据集上调优，跨任务泛化能力仍需进一步验证。

---

## 89. A Multi-Perspective Study of the Internet Shutdown in Iran

**arXiv ID:** 2605.00187 | [PDF](https://arxiv.org/pdf/2605.00187v1)

**作者:** Ali Sadeghi Jahromi `[一作]` (Carleton University), Jason Jaskolka `[通讯]` (Carleton University)

**通讯引用:** 471 | [OpenAlex ID](https://openalex.org/A5039439171)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对2026年伊朗互联网中断事件进行三面向测量：被动Censys扫描、主动TCP可达性探测以及BGP路由表分析。

**💡 创新点**

首次揭示伊朗从混合BGP撤销与转发层null路由演进至纯转发层null路由，并发现Censys扫描中的误差（72小时延迟和中间事件IP量激增）导致误判中断严重性。

**🔧 技术方法**

使用TCP SYN探测80/443/179端口、BGP路由收集（RIPE RIS）以及Censys BigQuery的日常扫描数据。

**📊 数据集**

数据集包括：Censys BigQuery日扫描（2026年1月1日至3月31日），RIPE RIS BGP快照（33个，覆盖2019–2026年），4,571个伊朗可路由前缀，5个地理多样化的测量节点。

**📈 对比分析**

通过多面向结果对比：BGP覆盖率保持稳定（>85%），主动探测显示96–97%前缀为null路由；Censys显示的主机数显著被人工误差放大，二者差距说明单一方法会误判。整体性能显示：转发层null路由覆盖广、时间一致、跨观测点差异<1%，但BGP监控因路由保持不变而失效。

**⚠️ 局限性**

局限性：仅涵盖IPv4前缀；Censys扫描受30小时延迟和重注入机制影响；主动探测仅覆盖5个节点；缺乏对BGP前缀撤回细粒度的动态观察；未评估对应用层服务的实际影响。

---

## 90. Introducing WARM-VR: Benchmark Dataset for Multimodal Wearable Affect Recognition in Virtual Reality

**arXiv ID:** 2605.00184 | [PDF](https://arxiv.org/pdf/2605.00184v1)

**作者:** Karim Alghoul `[一作]` (University of Ottawa), Abdulmotaleb El Saddik `[通讯]` (University of Ottawa)

**通讯引用:** 16962 | [OpenAlex ID](https://openalex.org/A5109797436)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了WARM-VR数据集，结合可穿戴生理传感器与多感官VR环境进行情绪识别实验。

**💡 创新点**

首次将嗅觉刺激与沉浸式VR相结合，并提供完整的可穿戴多模态数据，填补现有数据集缺失多感官和可穿戴条件的空白。

**🔧 技术方法**

使用CNN、Bi‑GRU、Transformer等深度学习模型以及传统HRV分析方法。

**📊 数据集**

WARM‑VR数据集（31人、BVP、ECG、EDA、TEMP、ACC、问卷标签）。

**📈 对比分析**

采用主观问卷与客观生理信号的二分类交叉验证，对比CNN、LSTM/GRU、Transformer，平均F1分数0.59‑0.64，AUC≈0.68，表明Transformer和Bi‑GRU在情绪辨识上略优于单纯CNN。

**⚠️ 局限性**

样本量有限、情绪维度仅限于valence、arousal和relaxation，且嗅觉刺激仅为单一气味，可能限制模型泛化和跨场景应用。

---

## 91. Network Digital Untwinning: Towards Backward Optimization of Digital Twins

**arXiv ID:** 2605.00169 | [PDF](https://arxiv.org/pdf/2605.00169v1)

**作者:** Zifan Zhang `[一作]` (North Carolina State University), Yuchen Liu `[通讯]` (North Carolina State University)

**通讯引用:** 10725 | [OpenAlex ID](https://openalex.org/A5100373054)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了网络数字拆解框架，用于在网络数字孪生中选择性删除目标孪生及其影响，同时保持模型完整性。

**💡 创新点**

创新点包括：单请求与并行请求拆解机制、基于连通度指标的影响评估、最优回滚点与高斯噪声注入、理论证明与零训练模型不可区分，以及自适应检查点策略。

**🔧 技术方法**

采用联邦学习聚合、L‑smooth梯度理论、高斯机制隐私、拓扑聚类调度和自适应检查点存储等技术。

**📊 数据集**

使用美国犹他州I‑15高速公路的真实车速与交通流循环计数器数据集。

**📈 对比分析**

与六种机器学习忘记基线（SIFU、IFU、Crab、FedME²、FedEraser、FedRecovery、FedAccum）在MSE和运行时间上对比，单并行拆解在保持误差几乎不变的前提下，将运行时间从数小时压缩至百秒级，存储需求降至几MB。

**⚠️ 局限性**

局限性在于需要中心化回滚与噪声注入，受控网络环境下实现；对极大规模分布式部署和动态拓扑的适应性仍待验证；理论假设的L‑smooth与梯度界限在高度异质数据中可能不完全成立。

---

## 92. RoboKA: KAN Informed Multimodal Learning for RoboCall Surveillance System

**arXiv ID:** 2605.00156 | [PDF](https://arxiv.org/pdf/2605.00156v1)

**作者:** Nitin Choudhury `[一作]` (Indraprastha Institute of Information Technology), Arun Balaji Buduru `[通讯]` (Indraprastha Institute of Information Technology)

**通讯引用:** 363 | [OpenAlex ID](https://openalex.org/A5014100784)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建合成的鲁棒机器人电话（robocall）数据集，并提出基于跨模态对比学习与Kolmogorov–Arnold网络（KAN）的多模态框架用于检测恶意机器人电话。

**💡 创新点**

①数据集聚焦情绪诱发、语音克隆和心理语言学操控三大攻击轴；②在跨模态特征对齐后引入KAN投影与堆叠KAN融合，显著提升非线性表达能力；③采用可不确定性权重的联合损失实现对齐与分类目标的自适应平衡。

**🔧 技术方法**

使用预训练音频模型（Wav2Vec2、WavLM、HuBERT）与文本模型（BERT、RoBERTa、GPT‑2），实现跨模态对比学习（CMCL），随后通过KAN投影头和堆叠KAN融合层完成多模态融合；训练过程中加入不确定性加权损失。

**📊 数据集**

RoboCall‑Simulated Adversarial（≈1200恶意+1200合法合成样本，覆盖情绪、克隆和心理语言学三种攻击方式）和来自FTC Do‑Not‑Call Registry 的 1378 条真实无聊电话样本。

**📈 对比分析**

与单模态（音频/文本）、简单拼接、late‑fusion MLP、跨模态注意力等基线进行对比。KA N模型在四种评估设置（T1–T4）下均优于基线，尤其在真实世界的 T4 任务中召回率提升约15%，宏 F1 与召回率均有显著提升。

**⚠️ 局限性**

仅在英文环境下验证；真实合法样本缺失导致在 T4 评估仅能计算召回率，无法评估精度；合成与真实电话在通道与噪声上的差距可能影响迁移性能。

---

## 93. Wasserstein Distributionally Robust Regret Optimization for Reinforcement Learning from Human Feedback

**arXiv ID:** 2605.00155 | [PDF](https://arxiv.org/pdf/2605.00155v1)

**作者:** Yikai Wang `[一作]` (University of North Carolina), Jose Blanchet `[通讯]` (Stanford University)

**通讯引用:** 3342 | [OpenAlex ID](https://openalex.org/A5011147039)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在大语言模型对齐的 RLHF 训练中引入 Wasserstein 分布式鲁棒代价（DRRO）以缓解奖励信号过拟合（Goodhart 现象），并提供一套可直接嵌入现有 PPO/GRPO 代码的实践算法。

**💡 创新点**

创新点包括：① 采用“鲁棒代价”而非传统的鲁棒价值，聚焦对比最优策略的 regret；② 在固定提示的有限响应集合上精确求解内层最坏情况，得到水填充（water‑filling）闭式最优；③ 将理论结果转化为可采样的 soft‑max 奖励修正，并结合动态预算策略；④ 通过理论与实验证明 DRRO 在相同稀疏度下比 DRO 更不惰性、性能更优。

**🔧 技术方法**

技术手段：Wasserstein 分布式鲁棒优化、ℓ1 误差集合、soft‑max/硬最大化奖励修正、单样本重要性采样（SNIS）估计、动态 KL‑驱动的预算调度、GRPO（分组 PPO）梯度方法、软化奖励与动态温度 τ 的调节。

**📊 数据集**

数据集：使用 HuggingFace 的 H4/hh‑rlhf 作为提示分布；对齐模型使用 Qwen2.5‑0.5B‑Instruct；奖励模型采用 OpenAssistant/reward‑model‑deberta‑v3‑base；评估使用 sileod/deberta‑v3‑large‑tasksource‑rlhf‑reward‑model 的金标奖励；训练与评估均在同一批提示集上进行，保留 512 条提示做验证，其余用于在线 RLHF 训练。

**📈 对比分析**

对比方法：标准 PPO、GRPO、基于奖励集成（Ensemble‑Mean、Ensemble‑UWO）、支持约束（BSPO）、信息瓶颈奖励（InfoRM）以及传统 DRO。实验表明：DRRO 在金标奖励峰值（1.59）和 KL 范围（35.74）上均超过所有基线；DRO 在相同设置下表现最差（0.54、16.54）。内部消融显示：动态预算 + 软最大化奖励修正是最优组合，硬最大化或固定预算均低于此。

**⚠️ 局限性**

局限性：① 仅在有限响应的固定提示模型下得到闭式解，需假设奖励模型误差在 ℓ1 范围内；② soft‑max 近似与采样方差在低温度下会增大；③ 动态预算基于 KL 估计的经验法则，缺乏理论上最优的调度；④ 在更大规模的多步生成或多任务对齐场景下的可扩展性与计算成本仍待验证。

---

## 94. Timing is Everything: Temporal Scaffolding of Semantic Surprise in Humor

**arXiv ID:** 2605.00143 | [PDF](https://arxiv.org/pdf/2605.00143v1)

**作者:** Yuxi Ma `[一作]` (Peking University), Yixin Zhu `[通讯]` (Peking University)

**通讯引用:** 4147 | [OpenAlex ID](https://openalex.org/A5051255725)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

分析了828场中国专业脱口秀的自然语言表现，构建了dpv框架研究时间与语义预测失误的协同作用，探讨了时间结构对幽默感知的主导作用。

**💡 创新点**

提出dpv框架，将时间与语义预测失误相互耦合，发现时间动态远超语义不协调在观众欣赏中的预测力，并证实高质量表演通过延长停顿强化语义惊讶实现专家化。

**🔧 技术方法**

采用ASR时间戳提取语音节拍、语速、停顿等时间特征，使用BERT/GPT嵌入计算句间语义距离，进行统计相关、t检验、方差分析以及多变量回归。

**📊 数据集**

使用828场2017-2025年电视脱口秀竞赛（《脱口秀大会》《脱口秀和TA的朋友们》《喜剧之王》）抽取的107小时中文文本与观众投票评分数据集。

**📈 对比分析**

通过偏相关、t检验和ANOVA比较时间特征与语义特征对投票率的解释力度，时间特征相关系数最高（平均停顿0.36，变异0.35），显著优于语义峰值0.10。

**⚠️ 局限性**

局限性包括仅基于中文脱口秀，编辑裁剪可能影响时间结构，缺乏实验因果验证，语义测量仅局限于句对距离，未考虑声学、面部、观众即时反馈等多模态因素。

---

## 95. Are Tools All We Need? Unveiling the Tool-Use Tax in LLM Agents

**arXiv ID:** 2605.00136 | [PDF](https://arxiv.org/pdf/2605.00136v1)

**作者:** Kaituo Zhang `[一作]` (University of Houston), Ying Lin `[通讯]` (University of Houston)

**通讯引用:** 7209 | [OpenAlex ID](https://openalex.org/A5015714457)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在存在语义干扰时，工具增强推理（tool-augmented reasoning）与原始链式思考（CoT）之间的性能差距，提出分解干扰来源的因子化干预框架，并引入工具使用税和能力重叠原则，进一步设计轻量级门控推理（G-STEP）以减轻协议误差。

**💡 创新点**

①将CoT–Tool差距拆解为三类：提示格式成本（Δ_sty）、工具调用协议开销（Δ_frc）和真实工具执行收益（Δ_cmp）；②提出能力重叠原则解释为何工具收益往往与CoT已具备的能力重合；③设计G-STEP门控机制，用于在推理过程中根据是否继续调用工具进行决策。

**🔧 技术方法**

因子化干预框架（包含七个对比条件）、轨迹级错误分类、样本级归因、能力重叠分析、门控推理（G-STEP）以及基于oracle的实验设计。

**📊 数据集**

基于原始数据集的语义干扰扩增版本：GSM8K-Sem-Distractor 与 HotPotQA-Sem-Distractor；原始基准 GSM8K 与 HotPotQA 用于对照。

**📈 对比分析**

通过七个实验条件（CoT、FCStyle、NoopTool、Full、Max1Turn、OracleCalc、OracleEvid）对准确率和证据F1进行比较；在GSM8K上，CoT显著优于工具增强，差距最高达33%；在HotPotQA上差距仅约1%；使用G-STEP后，GSM8K的CoT–Tool差距可恢复约75%。

**⚠️ 局限性**

工具增强在语义干扰环境下并非总能获益，工具调用协议的开销往往抵消了工具带来的收益；G-STEP仅能部分缓解问题，真正的提升仍需增强模型的本地推理和工具交互能力；不同任务和干扰类型对结果影响显著，说明需要更细粒度的评估与设计。

---

## 96. ARMOR 2025: A Military-Aligned Benchmark for Evaluating Large Language Model Safety Beyond Civilian Contexts

**arXiv ID:** 2605.00245 | [PDF](https://arxiv.org/pdf/2605.00245v1)

**作者:** Sydney Johns `[一作]` (Virginia Polytechnic Institute and State University), Wenjing Lou `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究设计并评估了面向军事环境的LLM安全基准，构造了519道基于法战法、交战规则和联合伦理条例的多项选择题，并对21款商业/开源LLM进行零样本评估。

**💡 创新点**

创新点在于将OODA循环与军事法典相结合，构建12类结构化分类；采用多模型共识生成与人工对齐验证来确保题库质量；并引入误拒绝率（FRR）量化模型在合法军事情境下的拒绝行为。

**🔧 技术方法**

使用多模型（Claude、GPT、Gemini等）在“模型在环”流程中生成问题；通过句子嵌入过滤冗余；对生成题目进行人工对齐与安全审查；对模型采用零样本推理，计算准确率和拒绝率。

**📊 数据集**

数据集为519道从美国军事法规文本（法战法、交战规则、联合伦理条例）提取的多项选择题；每道题均标注来源文件，保证命题与权威文献严格对应。

**📈 对比分析**

对21款模型使用统一零样本提示，记录各类准确率与平均准确率；结果显示GPT‑4o、Gemini 2.0等大模型在大多数类别上准确率>93%，小模型准确率介于69–84%；大模型拒绝率接近0，而部分小模型拒绝率可达1.7%。

**⚠️ 局限性**

局限性在于仅评估结构化多项选择决策，未覆盖多步规划、实时感知、敌对欺骗等复杂情境；对高阶伦理推理与不确定性处理的评估不足，模型在情境归纳和道德判断方面仍存在显著短板。

---

## 97. Adaptive Geodesic Conformal Prediction for Egocentric Camera Pose Estimation

**arXiv ID:** 2605.00233 | [PDF](https://arxiv.org/pdf/2605.00233v1)

**作者:** Aishani Pathak `[一作]` (Arizona State University), Hasti Seifi `[通讯]` (Arizona State University)

**通讯引用:** 1022 | [OpenAlex ID](https://openalex.org/A5043917822)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文针对增强现实等场景中的自我摄像机姿态估计，使用共形预测（CP）方法提供覆盖率保证，并提出了基于DINOv2的桥接模型实现的自适应CP。

**💡 创新点**

创新点在于：①首次在EPIC-Fields上量化了基于SE(3)几何的共形预测在最难25%帧（Q4）上的覆盖率缺口；②证明几何（Geodesic）距离优于欧氏距离能更准确识别物理难度帧；③设计了跨参与者可迁移的两阶段DINOv2-Bridge难度估计器，用于为每帧动态调整CP阈值，显著提升Q4覆盖率。

**🔧 技术方法**

使用了共形预测、SE(3)几何距离、DINOv2特征提取、轻量化多层感知机（MLP）进行难度估计，以及基于分块（split）CP的阈值计算。

**📊 数据集**

采用了EPIC-Fields数据集，该数据集基于EPIC-KITCHENS视频，通过COLMAP提供毫米级精度的全景摄像机轨迹，用于训练、校准和评估。

**📈 对比分析**

通过与传统单阈值共形预测（Standard CP）以及欧氏距离评分的对比，DINOv2-Bridge在常量速度预测器上将Q4覆盖率从约0.75提升至约0.93，同时保持整体覆盖率接近90%；相对地，标准CP在Q4的覆盖率仅约0.60，欧氏评分的Q4覆盖率低于几何评分。

**⚠️ 局限性**

局限性包括：仅在常量速度预测器上验证，无法直接迁移至其他预测器（如LightGlue、MonoDepth2）；桥接模型仅以单个参与者（P01）训练，跨参与者的泛化可能受限；对环境变化（反射、纹理缺失）的鲁棒性尚待进一步评估。

---

## 98. Replication in Graph Partitioning and Scheduling Problems

**arXiv ID:** 2605.00209 | [PDF](https://arxiv.org/pdf/2605.00209v1)

**作者:** Pál András Papp `[一作]` (Huawei Zurich Research Center), A. N. Yzelman `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了复制（重算）对图/超图划分与有向无环图调度问题的影响，既给出了理论复杂度分析，也在真实工作负载上做了大规模实验评估。

**💡 创新点**

创新点包括：①证明带复制的图划分问题不可在任何有限因子内逼近；②量化复制在划分和调度中的成本下降幅度；③设计了两种ILP扩展模型与一种能够在大规模DAG上使用的高级本地搜索启发式调度算法。

**🔧 技术方法**

技术手段包括：整数线性规划（ILP）模型（限复制与无限复制两种变体）；BSP调度模型与同步/通信成本参数；基本与高级（批量复制、超级步合并、超级步复制）启发式改进；理论证明（NP-hardness、近似不可化）。

**📊 数据集**

数据集涵盖：
• 由两大LLM（Qwen3‑235B、DeepSeek R1）在多任务评测中抽取的专家共现超图（5个层次）与其基于8元组/二元组的变体；
• 10个稀疏矩阵（SuiteSparse）对应的细粒度与粗粒度超图模型；
• 三大DAG基准：HyperDAG、PSDD、稀疏三角系统（分别包含数千到十万节点），以及用于ILP求解的小型DAG集。

**📈 对比分析**

与无复制基线（ILP最优或Papp等启发式）相比，复制可平均降低通信/总成本约17–65%（划分）和11–58%（调度）。ILP在可行范围内求得最优或近似最优；高级启发式在大规模DAG上取得与ILP相当或更优的提升，尤其在高通信成本或更多处理器场景下效果显著。

**⚠️ 局限性**

局限性：
• ILP求解受限于实例规模，无法覆盖大规模划分；
• 高级启发式虽效果好，但缺乏理论最优性或收敛性证明；
• 仅在BSP模型下验证，其他通信/同步模型的推广尚未完成；
• 理论结果集中在NP-hardness/不可逼近方面，对实际可行算法的深入分析仍待进一步研究。

---

## 99. How Designers Envision Value-Oriented AI Design Concepts with Generative AI

**arXiv ID:** 2605.00280 | [PDF](https://arxiv.org/pdf/2605.00280v1)

**作者:** Pitch Sinlapanuntakul `[一作]` (University of Washington), Mark Zachry `[通讯]` (University of Washington)

**通讯引用:** 1305 | [OpenAlex ID](https://openalex.org/A5057645006)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过让18位专业设计师在20分钟内使用多种生成式AI工具完成价值导向的AI概念设计，并配合思考记录和访谈，探究设计师在此情境下如何处理价值冲突与潜在危害。

**💡 创新点**

提出“递归反思‑在行动”（reciprocal reflection‑in‑action）的新框架，扩展了Schön的反思在行动理论，揭示AI既是工具也是设计材料，产生多层价值张力，并强调危害识别是设计中的关键信号。

**🔧 技术方法**

采用多款生成式AI工具（ChatGPT、Claude、Copilot、Gemini、Midjourney、Perplexity）进行设计活动；数据收集使用思考记录、访谈录音，分析方法为扎根理论编码和主题归纳。

**📊 数据集**

未使用公开数据集；数据仅来自18名参与者的设计草图、思考记录与访谈文本，形成质性研究素材。

**📈 对比分析**

本研究无定量对比或性能评估；结论基于访谈和草图的主题分析，未给出数值指标。

**⚠️ 局限性**

局限性包括：活动时长短（仅20分钟），单一价值框架限制多价值冲突探究；工具多样导致输出差异难以归因；参与者经验差异大；缺乏长期跟踪验证设计结果与真实使用情境的匹配。

---

## 100. A Comparative Analysis of Machine Learning Models for Intrusion Detection in Intelligent Transport Systems

**arXiv ID:** 2605.00279 | [PDF](https://arxiv.org/pdf/2605.00279v1)

**作者:** Zawad Yalmie Sazid `[一作]` (Victoria University), Sasa Maric `[通讯]` (University of New South Wales)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于可信联邦学习的混合入侵检测框架，并在边缘环境中验证了其有效性

**💡 创新点**

创新点在于：1) 将可信度评估与联邦聚合相结合，减少恶劣客户端更新的影响；2) 设计了混合本地检测器，结合降噪、特征提取与跨特征依赖建模；3) 在ITS边缘场景下首次系统评估经典机器学习方法的可行性

**🔧 技术方法**

使用了随机森林、决策树、线性支持向量机、联邦学习、可信聚合技术和边缘计算架构

**📊 数据集**

采用了CICIDS2017数据集的DDoS子集进行实验

**📈 对比分析**

对随机森林、决策树和线性SVM在同一数据集上进行基准比较，随机森林以99.9889%准确率、1.0 ROC‑AUC、2个假阳性和3个假阴性等指标实现了近乎完美的性能

**⚠️ 局限性**

局限性包括：仅在单一数据子集上做本地基准，未在非IID、多客户端、真实边缘硬件和通信约束下验证联邦框架；未与深度学习模型对比；对不同攻击类型的泛化能力尚未充分评估

---

## 101. Brief announcement: A special case of maximum flow over time with network changes

**arXiv ID:** 2605.00277 | [PDF](https://arxiv.org/pdf/2605.00277v1)

**作者:** Shuchi Chawla `[一作]` (University of Texas), Kristin Sheridan `[通讯]` (University of Texas)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对统一边长、可随时间变化的容量的时间网络，提出一种将最大流量求解转化为单个稳态最大流问题的思路，并通过构造一种压缩的时间扩展网络（cTEN）来实现。

**💡 创新点**

创新点在于：①定义了关键时间集合 N（由容量变点及其相对路径长度产生的时间点）并证明存在最小割的切割函数完全落在该集合内；②利用该集合构造 cTEN，使得其节点数仅为 O(n² μ)、边数为 O(μ m n)，从而把原问题化简为一次稳态最大流计算；③避免了传统的多次参数搜索或子模最小化的高复杂度。

**🔧 技术方法**

技术方法包括：时间扩展网络（TEN）与压缩版 cTEN；最小割与切割函数的定义与性质证明；利用关键时间集合对节点进行折叠；基于 Orlin、Chen 等算法的稳态最大流求解；以及对 μ、n、m 之间关系的复杂度分析。

**📊 数据集**

本文主要为理论分析，并未给出实验数据集。文中举例说明的应用场景为星座卫星通信网络和物流运输网络，但并未在实际数据上验证。

**📈 对比分析**

与现有的 Hoppe‑Tardos 转化+子模最小化方法（O(μ⁵) 或 O(μ³+… )）相比，本文在 μ 远大于 n、m 的情形下实现了 O(μ² n³ m) 或 O(μ^{1+o(1)}(nm)^{1+o(1)} log(nTU)) 的运行时间；若 m,n≤log μ 则可进一步降至 O(μ²) 或 O(μ^{1+o(1)} log(nTU))，显著提升效率。

**⚠️ 局限性**

局限性：①仅适用于所有边长度均为同一常数 τ 的网络；②仅给出了最大流量值的求解，对实际流方案的构造需借助额外算法；③对于非均匀边长或有存储约束的网络无法直接应用；④尽管复杂度比传统方法低，但在 μ 极大时仍可能较高。

---

## 102. REALM: An RGB and Event Aligned Latent Manifold for Cross-Modal Perception

**arXiv ID:** 2605.00271 | [PDF](https://arxiv.org/pdf/2605.00271v1)

**作者:** Vincenzo Polizzi `[一作]` (University of Toronto), Jonathan Kelly `[通讯]` (University of Toronto)

**通讯引用:** 3307 | [OpenAlex ID](https://openalex.org/A5011931977)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 REALM 框架，将事件数据映射到 RGB 预训练模型（DUNE）的共享潜在空间，并通过 LoRA 适配实现跨模态特征对齐。

**💡 创新点**

首次实现事件流直接投射至大型 RGB 基础模型潜在空间，允许零样本地使用冻结的图像解码器完成深度估计、语义分割和跨模态特征匹配。

**🔧 技术方法**

采用 voxel grid 表示事件，轻量级卷积嵌入器+LoRA 低秩自适应，以及多阶段遮挡与空间掩码的双重蒸馏策略。

**📊 数据集**

在 DSEC、EventScape、EventPointMesh、EDS 与 M3ED 等同步 RGB‑事件数据集上进行训练与评估。

**📈 对比分析**

与专门设计的事件基准模型相比，REALM 在单目深度、语义分割和大基线特征匹配任务上均实现了最高或接近最高的性能，尤其在特征匹配上实现了零样本使用 MASt3R 的最佳效果。

**⚠️ 局限性**

局限在于固定大小的 voxel grid 嵌入器，难以适应不同分辨率传感器，并且未充分捕捉长期时间动态，未来可引入循环或图网络改进。

---

## 103. Are You the A-hole? A Fair, Multi-Perspective Ethical Reasoning Framework

**arXiv ID:** 2605.00270 | [PDF](https://arxiv.org/pdf/2605.00270v1)

**作者:** Sheza Munir `[一作]` (University Of Toronto), Syed Ishtiaque Ahmed `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种神经符号化的多视角伦理推理框架，用来对 Reddit AITA 社区的高冲突道德评议进行一致性聚合，取代传统多数投票；

**💡 创新点**

创新点在于引入“拆流”架构，将事实与价值判断分离，通过 Weighted MaxSAT 结合 LLM 提取的逻辑谓词与权重，实现基于推理质量的聚合与逻辑一致性校验；

**🔧 技术方法**

技术包括 GPT‑5.1 的结构化向量提取（内容向量与质量向量）、Z3 SMT 求解器进行 Weighted MaxSAT 优化、以及自定义的分流权重计算与决策树；

**📊 数据集**

使用公开的 Kaggle AITA Reddit 档案（约 30,000 篇帖子，筛选 600 条高冲突帖子）作为实验数据集；

**📈 对比分析**

与 Reddit 官方多数投票结果进行对比，发现 62% 的最终标签被 MaxSAT 纠正；通过 50 篇帖子的人类评估验证，得到 86% 的一致率，表明该方法在逻辑一致性和人类评估方面优于单纯投票；

**⚠️ 局限性**

局限性包括对单一 LLM 产生的语义抽象错误的依赖、缺乏多代理或多文化视角、以及有限的谓词本体（仅覆盖四个内容谓词和五个质量维度），无法充分捕捉权力不平衡、同意与历史关系等细节。

---

## 104. How Language Models Process Out-of-Distribution Inputs: A Two-Pathway Framework

**arXiv ID:** 2605.00269 | [PDF](https://arxiv.org/pdf/2605.00269v1)

**作者:** Hamidreza Saghir `[一作]` `[通讯]` (Independent Researcher), Hamidreza Saghir (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过发现注意力导向的白盒OOD检测方法受序列长度偏倚影响，提出长度无关的两路路径框架（词嵌入+处理轨迹），并在多任务、多模型上进行实证验证。

**💡 创新点**

1）正式化序列长度结构性混淆并提出长度匹配评估；2）构建词汇透明度光谱的两路路径模型；3）提供机制层面的电路归因与激活补丁证据；4）交叉任务与跨模型的可迁移性验证。

**🔧 技术方法**

使用注意力熵、CED、RAUQ、WildGuard等注意力基准，k‑NN、Mahalanobis轨迹特征，层级相关性分析、激活补丁、结构性证明、长度匹配AUROC等技术。

**📊 数据集**

安全OOD基准（ToxicChat、Jailbreak、Spam、HateSpeech）、AGNews、JailbreakClassification，以及附录中的20News、CLINC-OOS等；实验模型主要为Qwen3‑0.6B，交叉验证包括SmolLM2、Gemma、Llama‑3.2‑1B、Qwen3‑4B、Qwen2.5‑7B‑Instruct。

**📈 对比分析**

在长度匹配评估下，传统注意力方法AUROC仅≈0.50，轨迹特征平均AUROC为0.72；k‑NN在内容差异任务上胜过轨迹，轨迹在结构/对抗任务上优势明显；跨模型平均AUROC从0.66‑0.74不等，监督LR上限≈0.77。

**⚠️ 局限性**

对ToxicChat、HateSpeech等任务的无监督信号较弱；激活补丁仅在Jailbreak与HateSpeech上显著；实验规模限于≤10B参数；特征类别F缺乏严格长度不依赖证明；跨模型复制部分缺失。

---

## 105. A PVT-Resilient Subthreshold SRAM-Based In-Memory Computing Accelerator with In-Situ Regulation for Energy-Efficient Spiking Neural Networks

**arXiv ID:** 2605.00319 | [PDF](https://arxiv.org/pdf/2605.00319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 106. Pessimism-Free Offline Learning in General-Sum Games via KL Regularization

**arXiv ID:** 2605.00264 | [PDF](https://arxiv.org/pdf/2605.00264v1)

**作者:** Claire Chen `[一作]` (California Institute of Technology), Yuheng Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8521 | [OpenAlex ID](https://openalex.org/A5056699900)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了在离线多智能体一般总和游戏中使用 KL 正则化的学习框架，构建了 GANE 与 GAMD 两种算法，分别实现了 Nash 均衡与 Coarse Correlated 均衡的恢复；

**💡 创新点**

创新点在于证明 KL 正则化可以单独充当“隐式悲观”机制，消除传统悲观惩罚的需要，并利用 Nash 均衡的乘积结构实现 O(1/n) 的统计收敛速率；

**🔧 技术方法**

主要技术包括参考锚定的单方集中性假设、KL 正则化的 Q‑value 与价值函数设计、GANE 的经验风险最小化与阶段游戏均衡解算器、以及 GAMD 的 KL 约束镜像下降迭代；

**📊 数据集**

论文未使用具体公开数据集，全部以理论离线上下文赌博机数据集为模型，假设奖励可被函数族完备表示；

**📈 对比分析**

与传统使用显式悲观罰金或保守值迭代的离线博弈学习方法相比，GANE 在 Nash 均衡上取得了加速的 O(1/n) 收敛率，GAMD 在 CCE 上以 O(1/√n + 1/T) 的速率实现了可行的数值性能；

**⚠️ 局限性**

局限性包括仅处理上下文赌博机（无时序性）模型；GANE 需要访问阶段游戏均衡求解器，计算复杂度高；方法对参考策略的选择敏感，且假设奖励可被函数族完备近似，实证验证仍待进一步探索。

---

## 107. Retrieval-Augmented Reasoning for Chartered Accountancy

**arXiv ID:** 2605.00257 | [PDF](https://arxiv.org/pdf/2605.00257v1)

**作者:** Jatin Gupta `[一作]` (Sharda University), Ali Imam Abidi `[通讯]` (Sharda University)

**通讯引用:** 252 | [OpenAlex ID](https://openalex.org/A5024592017)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文开发并评估了一种名为CA-ThinkFlow的检索增强生成（RAG）框架，用于印度注册会计师（CA）考试的多层级问答任务。

**💡 创新点**

创新点在于将4‑位量化的14B-DeepSeek-R1推理模型与布局感知Docling文档提取、Qwen嵌入以及FAISS向量检索相结合，实现了参数高效且能在多层考试中竞争大模型的性能。

**🔧 技术方法**

采用的技术包括4‑bit量化的DeepSeek‑R1、Docling提取、Qwen‑Embedding‑0.6B、FAISS向量索引、Chain‑of‑Thought推理、LangChain/Ollama实现的RAG循环。

**📊 数据集**

使用的数据集主要为构建的印度财务与法规知识库，并在官方CA‑Ben基准上进行零样本评估。

**📈 对比分析**

评估方法为零样本测试CA‑Ben，比较模型在Foundation、Intermediate、Final三个层次的准确率和SRC（学术可靠性系数），CA‑ThinkFlow在基础和最终层次上达68.75% SRC，性能与GPT‑4o、Claude‑3.5等大模型相当，并优于多数同类模型。

**⚠️ 局限性**

局限性在于对税法和间接税法等复杂法规领域的推理知识不足，导致这些主题的准确率显著低于其他领域，且核心推理能力仍无法突破深层法规理解的瓶颈。

---

## 108. Sequential Automorphism Ensemble Decoding with Early Stopping

**arXiv ID:** 2605.00255 | [PDF](https://arxiv.org/pdf/2605.00255v1)

**作者:** Pillet Charles `[一作]` (Synchromedia), Leduc-Primeau François `[通讯]` (Polytechnique Montréal)

**通讯引用:** 722 | [OpenAlex ID](https://openalex.org/A5038571252)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

本文提出了一种顺序动态自同构集合解码（DAE）方法，在原始的ae解码基础上引入了基于路径指标的提前停止机制，并通过Monte‑Carlo搜索优化停止阈值，从而在保持近似ML误码性能的同时显著降低平均解码复杂度。它还给出了可进一步终止子解码器的变体PDAE。

**💡 创新点**

创新点主要有：①利用路径指标与解码结果的强相关性设计提前停止阈值；②通过错误分配的阶梯搜索（hill‑climbing）在满足BLER约束的前提下最小化平均子解码器数量；③提出了能够在子解码过程中中止的PDAE，进一步削减复杂度；④在RM和极化码上验证了该方法对不同规模的解码器集合（M=8,32）均有效。

**🔧 技术方法**

技术要点包括：顺序自同构集合解码（顺序调用SC解码器并评估路径指标）；路径指标（pm）作为提前停止的判别量；基于Monte‑Carlo的错误分配优化（hill‑climbing）；与传统AE-M‑SC、AE‑8‑SC、AE‑32‑SC、SC和ML解码器的性能对比；AWGN信道下BPSK调制的仿真。

**📊 数据集**

使用的代码与数据集：RM码（128,64）、（128,99）以及极化码（128,60）、（128,98），长度N=128；仿真数据量约5×10^5条；通过对不同SNR点进行仿真以获取BLER曲线和平均子解码器数量。

**📈 对比分析**

比较方法：将DAE/PDAE的平均子解码器数M与原始AE-M‑SC、AE-8‑SC、AE-32‑SC以及单一SC解码器进行对比；同时与ML上界和极化码的ML解码器进行误码率对比。结果显示：在BLER≈10^-3时，DAE对RM码平均复杂度降低5.9×–19.3×，PDAE进一步提高到6.3×–21.9×，误码率增益不超过原AE解码的10%。

**⚠️ 局限性**

局限性：①仅在AWGN+ BPSK环境下验证，缺乏对更复杂信道或调制的适用性分析；②停止阈值σ需要针对每个码、SNR和M预先仿真并存储，实际实现中可能增加存储与配置开销；③提前停止导致解码时间不确定，对低时延系统可能不友好；④在极低BLER（<10^-5）或极高SNR时，阈值优化可能失效，误码率增益不显著。

---

## 109. Developing an AI Concept Envisioning Toolkit to Support Reflective Juxtaposition of Values and Harms

**arXiv ID:** 2605.00282 | [PDF](https://arxiv.org/pdf/2605.00282v1)

**作者:** Pitch Sinlapanuntakul `[一作]` (University of Washington), Mark Zachry `[通讯]` (University of Washington)

**通讯引用:** 1305 | [OpenAlex ID](https://openalex.org/A5057645006)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套 AI 概念设想工具包（AI 能力库、价值–危害卡片、价值张力地图），并将其嵌入 FigJam 中，帮助设计师在早期 AI 设计阶段对价值与潜在危害进行可视化、对比与反思，促成价值冲突的讨论与设计迭代。

**💡 创新点**

创新点在于：① 将价值与危害的对比与 AI 能力直接匹配；② 通过卡片和二维张力地图将抽象价值转化为可操作的、可移动的物料；③ 将这些材料嵌入常用设计工具，减少上下文切换；④ 通过“生产性摩擦”让价值冲突成为设计过程中的主动触发点，而非事后检查。

**🔧 技术方法**

采用研究‑设计（RtD）方法，结合定量问卷与定性访谈；技术实现上使用 FigJam 组件、静态 PDF 模板、卡片式界面和可视化图表；对评估数据进行描述性统计、t 检验、相关分析和主题分析。

**📊 数据集**

使用设计师参与的自评数据：Phase 1 30 名设计师的问卷评分与开放式回答；Phase 2 12 名设计师的访谈文本；未使用传统机器学习数据集，而是聚焦于设计实践中的体验数据。

**📈 对比分析**

通过与现有工具（如 Envisioning Cards、Google PAIR Guidebook、CMU AI Brainstorming Kit 等）在清晰度、实用性、全面性等维度进行功能对比；评估显示工具在价值反思、危害预见和概念迭代方面平均评分高于 6/7，采用意愿显著高于基线工具，且设计师在使用过程中报告到更深入的价值冲突讨论与概念重构。

**⚠️ 局限性**

局限性包括：样本仅来自西方技术行业设计师，缺乏跨文化与跨学科团队验证；使用单项量表，缺乏测量信度与效度检验；工具聚焦价值与危害，未直接涵盖数据集、算法偏差等技术层面，未来需扩展至更广泛的场景与团队。

---

## 110. High-Probability Convergence in Decentralized Stochastic Optimization with Gradient Tracking

**arXiv ID:** 2605.00281 | [PDF](https://arxiv.org/pdf/2605.00281v1)

**作者:** Aleksandar Armacki `[一作]` (Ecole Polytechnique Federale de Lausanne), Ali H. Sayed `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在去中心化随机优化中，引入梯度跟踪（GT）技术，并在放宽的次高斯噪声条件下，给出了高概率（HP）收敛性保证。

**💡 创新点**

首次提供了带偏差校正方法（如GT）的HP收敛结果，消除了传统方法对梯度异质性与单个成本强凸性的严格假设，并实现了线性加速。

**🔧 技术方法**

使用MGF（矩母函数）分析、放宽的次高斯噪声模型、梯度跟踪算法以及高概率分析框架。

**📊 数据集**

实验涵盖合成二次成本以及真实二分类数据集（mushroom、a9a、ijcnn1）。

**📈 对比分析**

与传统DSGD相比，实验表明GT方法在高概率下呈指数级尾部衰减，并随节点数增加实现线性加速，整体性能优于或等同于DSGD。

**⚠️ 局限性**

临界时间相对MSE结果略长，仍需针对更复杂网络或重尾噪声进行统一框架分析，且目前仅覆盖梯度跟踪类方法。

---

## 111. Fast Rates in $α$-Potential Games via Regularized Mirror Descent

**arXiv ID:** 2605.00268 | [PDF](https://arxiv.org/pdf/2605.00268v1)

**作者:** Claire Chen `[一作]` (California Institute of Technology), Yuheng Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8521 | [OpenAlex ID](https://openalex.org/A5056699900)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在α-潜力博弈（α-potential game）框架下的离线 Nash 均衡学习，提出了基于 KL 正则化的 Reference‑Anchored coverage 条件，并实现了两种算法：ROPE（基于潜力最大化的闭环学习）和 OPMD（离散化的 Mirror Descent 自己训练）。

**💡 创新点**

创新点包括：① 将数据覆盖从“未知最优”转为“已知参考策略”Anchor，解决了循环依赖问题；② 利用 KL 正则化形成的 log‑linear 插值结构，将评估误差转化为可消失的优化误差，从而突破传统 𝒪(1/√n) 限界，实现 𝒪(1/n) 的统计收敛速率；③ 在离线多智能体学习中提供完全去中心化的 Mirror Descent 算法。

**🔧 技术方法**

主要技术：KL 正则化、Reference‑Anchored unilateral concentrability、潜力函数最大化、离线函数逼近（regularized least‑squares）、Mirror Descent（KL‑投影）以及理论分析中的潜力上升残差分解和 Pinsker/Fenchel 对偶关系。

**📊 数据集**

论文未给出具体公开数据集，假设使用通用离线多智能体数据集（如多玩家上下文赌博机环境）进行实验；理论分析基于通用假设（奖励可实现、覆盖系数可估计）。

**📈 对比分析**

与传统离线多智能体学习（常见的 pessimism 或 LCB 机制）相比，ROPE/OPMD 在相同样本量下显著降低了 Nash Gap（误差）——从 𝒪(1/√n) 降至 𝒪(1/n)；实验中表现出更快的收敛速度和更小的总体利用率。

**⚠️ 局限性**

局限性：仅适用于 α-潜力博弈，需已知参考策略；对 α 的大小敏感，α 越大误差上限越高；仅证明在上下文赌博机（无动力学）下的收敛速率，未来需推广至多步 MDP；对 KL 正则化参数 η 的选择仍需经验调优。

---

## 112. Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning

**arXiv ID:** 2605.00265 | [PDF](https://arxiv.org/pdf/2605.00265v1)

**作者:** Sahil Mishra `[一作]` (Indian Institute of Technology Delhi), Tanmoy Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 5232 | [OpenAlex ID](https://openalex.org/A5046521217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个极坐标极球嵌入框架，用来在超球面上分离语义方向与层级深度，从而实现层次概念学习与拓展。

**💡 创新点**

创新点是将层级信息编码为半径（轨道势），语义信息保持为单位向量方向，结合球面正交线性层、地理三元组损失、SVGD正则化以及基于vMF分布的不确定性方向对齐，形成了一个完整的极坐标极球嵌入体系。

**🔧 技术方法**

使用了Riemannian指数映射与切空间投影实现球面编码，球面线性层保持单位范数，地理三元组损失基于弧度距离的Welsch估计，SVGD与von Mises–Fisher核实现全局正则化，vMF KL损失实现不确定性方向对齐，并在推理阶段引入轨道检索门控。

**📊 数据集**

在三类数据集上评估：单父树型语义学（Science、WordNet、Environment）、多父DAG（MeSH、Verb、Food）以及多模态鸟类图像-标签（CUB-200-2011）。

**📈 对比分析**

与TransE、RotatE、HAKE等KGE；HyperExpan、ConE等超球面/秩序嵌入；Box、Gumbel Box、TaxoExpan、STEAM、TMN、Arborist等结构感知方法对比，取得在 top‑K 回调率提升约19点、平均排名下降约60%，在所有基准上均保持领先或同水平。

**⚠️ 局限性**

局限在于需要较为可靠的层级监督来推断半径；vMF分布假设各维度同质，可能不适用于多面向概念；全局正则化与对齐权重需要手动调节；检索门控受种子层级稀疏性限制。

---

## 113. Causal Foundations of Collective Agency

**arXiv ID:** 2605.00248 | [PDF](https://arxiv.org/pdf/2605.00248v1)

**作者:** Frederik Hytting Jørgensen `[一作]` (University of Copenhagen), Lewis Hammond `[通讯]` (Cooperative AI Foundation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出基于机制化因果图与因果抽象的集体代理理论，并用该框架解释 Actor‑Critic 与投票机制中的集体代理现象。

**💡 创新点**

首次将因果抽象与代理建模结合，给出非平凡代理出现的判定条件和机制化抽象的严格数学定义。

**🔧 技术方法**

使用机制化结构因果模型、因果抽象、最佳响应理性关系、神经网络映射等技术。

**📊 数据集**

使用仿真生成的5个国家、1000名公民的投票实验合成数据。

**📈 对比分析**

与不使用抽象的基线（恒定预测）比较，MAE 在 VCG 和中位投票下分别降低 95% 与 88%，而随机专制的误差更大，表现最差。

**⚠️ 局限性**

限制在于假设干预仅影响 α 而非 δ、仅考虑最佳响应理性、仅在合成实验中验证，难以直接推广到更复杂或真实系统。

---

## 114. Intelligent Elastic Feature Fading: Enabling Model Retrain-Free Feature Efficiency Rollouts at Scale

**arXiv ID:** 2605.00324 | [PDF](https://arxiv.org/pdf/2605.00324v1)

**作者:** Jieming Di `[一作]` (Meta), Rocky Liu `[通讯]` (Meta)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在大规模排序系统中实现不需要重新训练的特征效率滚动发布，利用服务端增量衰减特征覆盖率与分布；

**💡 创新点**

通过控制平面与服务端特征适配器实现系统级渐进式特征衰减，实现训练‑服务一致性、可回滚与安全保障；

**🔧 技术方法**

采用IEFF架构，包含实时监控、回滚机制和渐进衰减策略的控制平面与服务端适配器；

**📊 数据集**

在广告排名平台的CTR/CVR模型上，使用稀疏ID特征和嵌入特征进行实验；

**📈 对比分析**

离线模拟和线上A/B（QRT）对比显示，渐进衰减相比突兀零掉可将性能下降降低约50‑55%，并在上线过程中将发布速度提升约5倍，节省GPU成本；

**⚠️ 局限性**

仅适用于可逐步拆解的特征，极度敏感或非线性交互特征仍需重训练；衰减速率依赖先验QRT验证，缺乏自适应调度与对非线性特征的鲁棒性。

---

## 115. Federated Weather Modeling on Sensor Data

**arXiv ID:** 2605.00322 | [PDF](https://arxiv.org/pdf/2605.00322v1)

**作者:** Shengchao Chen `[一作]` (University of Technology Sydney), Guodong Long `[通讯]` (University of Technology Sydney)

**通讯引用:** 14241 | [OpenAlex ID](https://openalex.org/A5059227406)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文综述了联邦学习（Federated Learning）在气象传感器数据上的应用，探讨了时间序列分析、空间关系建模、个性化模型与图聚合等关键技术，并对基于大语言模型和多模态数据的未来研究方向进行了展望。

**💡 创新点**

创新点在于系统性地聚合了多领域的联邦气象建模方法，提出了基于地理位置的图聚合与个性化层分离策略，以及在联邦框架下高效利用预训练大型语言模型的方案，强调了解释性与多模态数据融合的研究空缺。

**🔧 技术方法**

使用的主要技术包括联邦学习框架（FedAvg、个性化FL、混合专家MoE）、图神经网络（GNN）用于空间关系建模、Transformer和时间序列CNN/RNN、预训练大语言模型（LLM）及其微调、生成对抗网络（GAN）用于数据增强以及基于掩码的空间注意力模块。

**📊 数据集**

涉及的数据集涵盖多源气象传感器数据：地面观测站、卫星遥感、雷达影像、无人机（UAV）采集、IoT传感器以及社交媒体文本描述等，主要为公开或行业合作的气象时间序列与空间格点数据。

**📈 对比分析**

对比方法主要是传统中心化深度学习模型（如CNN、RNN、Transformer）与联邦学习方案。实验结果表明，在保持数据隐私的前提下，联邦学习能够在降雨、温度、风速等变量预测任务中实现与中心化模型相当甚至更优的准确率（提升1-5%），并在灾害预警、极端事件检测等高置信度任务中提升鲁棒性；通信成本与模型收敛速度在多模态数据场景中仍需进一步优化。

**⚠️ 局限性**

主要局限包括：①设备资源有限导致深度模型难以在边缘端训练；②数据异质性（时空分布不均、传感器差异）影响聚合效果；③通信成本高，尤其在多模态或大模型情况下；④模型可解释性不足，难以满足气象决策的透明度需求；⑤缺乏统一、规模化的预训练气象基础模型，导致跨域迁移受限；⑥多模态数据在联邦环境中的协同学习仍面临分布差异与不平衡挑战。

---

## 116. Structure-Aware Chunking for Tabular Data in Retrieval-Augmented Generation

**arXiv ID:** 2605.00318 | [PDF](https://arxiv.org/pdf/2605.00318v1)

**作者:** Pooja Guttal `[一作]` (Altumatim), Manas Gaur `[通讯]` (University of Maryland Baltimore County)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种结构感知的表格分块框架（STC），通过构造 Row Tree 并在行级别进行基于 token 预算的分块与贪心合并，保持行内字段的语义关联；

**💡 创新点**

创新点在于：①引入 Row Tree 结构与 KV 块表示，专为 CSV/Excel 设计；②在分块时对 token 预算进行约束并在结构边界内分块；③采用无重叠的贪心合并，显著减少碎片化；

**🔧 技术方法**

技术手段包括：树形递归分块、键值（KV）编码、token 预算约束、贪心合并、检索基准（dense+BM25+cross‑encoder 与单纯 BM25）、性能评估指标（MSE、MRR、Recall@k 等）；

**📊 数据集**

使用的数据集为 MAUD（Merger Agreement Understanding Dataset），包含合并协议中的法律条款、问题与答案等结构化记录；

**📈 对比分析**

与基线 RecursiveCharacterTextSplitter（带滑动窗口）以及 KV+Recursive 消融组进行对比；STC 在 512 token 预算下将块数减少约 40%（对 Recursive）或 56%（对 KV+Recursive），平均块大小提升至约 400 tokens，处理速度提升；检索性能提升显著：混合检索 MRR 从 0.3576 提升至 0.5945，BM25-only Recall@1 从 0.366 提升至 0.754；

**⚠️ 局限性**

局限性包括：仅在固定 token 预算下评估；检索评估使用启发式相关性匹配，未覆盖完整的 RAG 生成流程；实验仅基于 MAUD 这一法律领域数据集，需在更多领域和不同表格规模上验证。

---

## 117. Semia: Auditing Agent Skills via Constraint-Guided Representation Synthesis

**arXiv ID:** 2605.00314 | [PDF](https://arxiv.org/pdf/2605.00314v1)

**作者:** Hongbo Wen `[一作]` (University of California, Santa Barbara), Yu Feng `[通讯]` (University of California, Santa Barbara)

**通讯引用:** 38436 | [OpenAlex ID](https://openalex.org/A5009277202)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套静态审计框架，先将 AI 代理技能（混合 YAML + 英语说明）转换为结构化的 Skill Description Language（SDL），随后利用 Datalog 进行可达性查询，识别安全风险。

**💡 创新点**

创新点在于：① 通过约束引导的表示合成（CGRS）实现 LLM 生成结构化 IR 的迭代自我校正；② 引入 SDL 这一精简的关系型事实集合，使得传统静态分析工具可以直接对技能进行定量检测；③ 将 LLM 与符号推理相结合，实现了在不依赖运行时信息的前置审计。

**🔧 技术方法**

技术栈包括：大语言模型（GPT‑4‑Turbo）负责从原始技能文本生成 SDL；Python 编写的 CGRS 循环进行结构与语义校验；Soufflé Datalog 引擎执行十一条安全检测规则（覆盖无门限、污点流、结构异常等七类风险）。

**📊 数据集**

数据集为 13,728 个来自 OpenClaw 公共市场的真实技能，其中 541 个经过人工双标记形成基准集（301 种风险，240 干净），并从中提取 13,728 条技能的结构化信息用于评估。

**📈 对比分析**

与 VirusTotal（基于签名）和 ClawScan（轻量 LLM 规则）对比：在 541 样本上，系统实现 F1 = 90.6%（精度 84.5%，召回 97.7%），显著优于 VT（精度 89.1%，召回 13.6%）和 C‑Scan（精度 73.2%，召回 52.6%）。在每个风险类别上，召回率均在 94.6–100% 之间，显示出高泛化能力；系统还发现 17 条被确认的零日漏洞。

**⚠️ 局限性**

局限性：① 评估集中在单一生态（OpenClaw），不同市场的技能结构和权限模型可能导致偏差；② 对深层嵌套的第三方 API 依赖无法完整展开，可能漏掉路径；③ LLM 的生成不确定性仍会影响精确性，需固定温度和迭代上限；④ 仅覆盖文档层面的安全策略，无法检测运行时自适应攻击；⑤ 对极端伪造或混淆文本的鲁棒性尚待验证。

---

## 118. Beyond Visual Fidelity: Benchmarking Super-Resolution Models for Large-Scale Remote Sensing Imagery via Downstream Task Integration

**arXiv ID:** 2605.00310 | [PDF](https://arxiv.org/pdf/2605.00310v1)

**作者:** Zhili Li `[一作]` (University of Maryland), Yiqun Xie `[通讯]` (University of Maryland)

**通讯引用:** 1012 | [OpenAlex ID](https://openalex.org/A5049041437)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开GeoSR-Bench，一个整合遥感图像超分辨率与下游任务的基准数据集；

**💡 创新点**

首次将超分辨率评价与实际下游任务（如土地覆盖分割、基础设施映射、植被变量估算）结合，揭示传统PSNR/SSIM与任务性能不一定正相关；

**🔧 技术方法**

采用Transformer、神经算子、GAN和扩散四类超分模型，并用U-Net、SegFormer、Swin Transformer三种像素级任务模型进行评估；

**📊 数据集**

使用MODIS→Landsat‑8（500m→30m）和Sentinel‑2→NAIP（10m→0.6m）跨平台对齐的约36,000个位置的真实对偶影像，以及对应的10个下游任务数据集；

**📈 对比分析**

通过270个实验（9超分模型×5任务×3任务模型×2SR场景）比较PSNR/SSIM、F1/MAE等指标，发现虽然部分模型在视觉指标上领先，但在任务指标上并未总是优越，甚至出现负相关；

**⚠️ 局限性**

超分对亚像素级下游任务效果有限，且视觉质量指标对竞争模型的排序与任务性能的相关性低，表明单纯依赖PSNR/SSIM无法指导实际遥感应用的模型优化。

---

## 119. What Don't You Understand? Using Large Language Models to Identify and Characterize Student Misconceptions About Challenging Topics

**arXiv ID:** 2605.00294 | [PDF](https://arxiv.org/pdf/2605.00294v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 120. A Model-based Visual Contact Localization and Force Sensing System for Compliant Robotic Grippers

**arXiv ID:** 2605.00307 | [PDF](https://arxiv.org/pdf/2605.00307v1)

**作者:** Kaiwen Zuo `[一作]` (Case Western Reserve University), Zonghe Chua `[通讯]` (Case Western Reserve University)

**通讯引用:** 76 | [OpenAlex ID](https://openalex.org/A5082074188)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一种基于模型的视觉接触定位与力感知系统，用于柔性鳍状抓手，通过实时 RGB‑D 图像提取关键点并在 Sofa 中进行 ifea 仿真，实现对接触力的估计。

**💡 创新点**

创新点包括：1) 通过自适应尺度校准的 3D 物体重建解决 RGB‑D 与 CAD 之间的尺度差异；2) 设计基于模型的迭代接触定位算法，可在视角遮挡下估计接触点；3) 将关键点检测、姿态估计与有限元仿真集成到实时闭环。

**🔧 技术方法**

采用 Intel RealSense D405 RGB‑D 摄像头、DeepLabCut（dlc）关键点检测、FoundationPose 姿态估计、SAM‑3D 物体重建、Open3D ICP、Poisson 表面重建以及 Sofa 的 ifea 与 qp‑inverse 求解器。

**📊 数据集**

数据集包括 300 帧标注的抓手运动（圆柱和立方体），3.44 小时的抓取视频（3 级力、5 速、14 位置），以及用于实验的 3 种物体（圆柱、立方体、非对称对象）配备内置力/扭矩传感器。

**📈 对比分析**

与端到端深度学习基线（ResNet‑50+Transformer）对比，系统在静态和机器人抓取实验中平均 RMSE 0.23–0.48 N，NRMSD 2.1–4.3 %，在未见过的异形物体上仍保持低误差，而基线误差显著升高；接触定位器将误差从约1.2 N 降低到 0.48 N。

**⚠️ 局限性**

局限性包括：1) 仅适用于线性弹性材料，无法处理高度非线性或材料老化；2) 依赖关键点可见性和光照，遮挡或低光会显著降低精度；3) 先验对接触候选点的设置有限，复杂形状可能导致误差；4) 计算量仍较大，接触估计帧率受 10–30 Hz 限制。

---

## 121. Engagement Phenotypes for a Sample of 102,684 AI Mental Health Chatbot Users and Dose-Response Associations with Clinical Outcomes

**arXiv ID:** 2605.00275 | [PDF](https://arxiv.org/pdf/2605.00275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 122. Token Arena: A Continuous Benchmark Unifying Energy and Cognition in AI Inference

**arXiv ID:** 2605.00300 | [PDF](https://arxiv.org/pdf/2605.00300v1)

**作者:** Yuxuan Gao `[一作]` (University of Pennsylvania), Yi Ling Yu `[通讯]` (University of Pennsylvania)

**通讯引用:** 1927 | [OpenAlex ID](https://openalex.org/A5100644814)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

Token Arena 提出了一个端点级持续推理基准，统一测量推理速度、首 token 延迟、混合成本、有效上下文长度、质量以及模型端点的能耗，并将这些维度综合为 joules/正确答案、$/正确答案和端点忠实度三个主指标。

**💡 创新点**

创新点包括：① 以端点（provider+model+SKU+精度+解码方式+地区）为最细粒度分析单元；② 引入工作负载感知的加权预设（chat、RAG、reasoning 等）以反映真实部署成本；③ 通过指纹相似度检测隐藏量化或模型漂移；④ 在公开数据和模型下持续更新、可复现的评测框架，并提供完整的代码与数据。

**🔧 技术方法**

采用的技术包括：持续探测（probe）循环收集 TTFT、吞吐率、错误率；评估循环运行公开评测套件并记录准确率、token 计数、时间和费用；能耗建模基于硬件 TDP、PUE、区域电网强度；指纹相似度利用对称 KL 散度；综合得分采用权重归一化后的加权和；还进行敏感性分析与因子消融验证。

**📊 数据集**

使用的数据集与评测套件包括：MMLU-Pro、GPQA-D、MATH‑500、AIME 2025、HumanEval+、LiveCodeBench、IFBench、AA‑LCR、τ²‑Bench 以及自定义确定性提示与指纹参考集，覆盖数学、代码、推理、对话等多维任务。

**📈 对比分析**

比较方法：在 78 个端点、12 族模型、33 供应商的连续基准中，计算每个端点的主指标。结果表明同一模型在不同端点间存在：准确度差异 12.5 点、能耗差异 6.2 倍、尾延迟差异 10 倍；不同工作负载预设导致前 10 名排名仅 30–40 % 交叉；端点忠实度能有效揭示未披露的 FP8 量化。

**⚠️ 局限性**

局限性：能耗估计为模型而非实测；评测集可能饱和或受污染；对纯开源模型缺乏第一方指纹；默认单一区域（US‑East）主视图；对闭源仅主机的端点无指纹；可能被游戏（缓存、热点提示等）且需要持续监测。

---

## 123. A Privacy-Preserving Approach to Conformance Checking

**arXiv ID:** 2605.00283 | [PDF](https://arxiv.org/pdf/2605.00283v1)

**作者:** Luis Rodríguez-Flores `[一作]` (Tecnologico de Monterrey), Astrid Rivera-Partida `[通讯]` (University of Melbourne)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于全同态加密和FM-Index的隐私保护符合性检查方法

**💡 创新点**

首次将全同态加密与字符串匹配算法结合，用于保护模型与日志的隐私；将轨迹对齐映射为子串匹配并仅支持日志移动

**🔧 技术方法**

全同态加密、FM-Index、Burrows‑Wheeler变换、波纹矩阵（Wavelet Matrix）以及Petri网展开技术

**📊 数据集**

使用公开的真实事件日志（Log G）和两个合成数据集（Log F及其他），覆盖序列、并行、选择与循环等控制流模式

**📈 对比分析**

在单机实验中对每条日志轨迹进行对齐，平均时间约1.12分钟（每个符号0.112分钟），与数据规模呈近线性关系；对比未加密实现仅是同态运算的开销

**⚠️ 局限性**

当前实现仅支持日志移动，无法处理模型移动；全同态运算成本高、内存占用大；循环多次展开的情况尚未充分处理

---

## 124. Data Deletion Can Help in Adaptive RL

**arXiv ID:** 2605.00298 | [PDF](https://arxiv.org/pdf/2605.00298v1)

**作者:** Param Budhraja `[一作]` (Boston University), Venkatesh Saligrama `[通讯]` (Boston University)

**通讯引用:** 8346 | [OpenAlex ID](https://openalex.org/A5048704387)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在时变环境中部署强化学习策略，提出在多轮训练中随机删除训练缓冲区中一部分数据，以提升上下文估计器和自适应策略的泛化性能。

**💡 创新点**

创新点是将简单的随机删除数据作为隐式的指数衰减机制，理论证明在分布不匹配时可降低测试误差，并在实践中显著提升多轮训练的表现。

**🔧 技术方法**

采用上下文马尔可夫决策过程框架，使用通用策略与上下文估计器的分解训练，利用多层感知器、LSTM/GRU回归网络，结合PPO、DDPG等RL算法和凸风险最小化分析。

**📊 数据集**

实验数据集包括经典控制与Box2D环境（Pendulum、LunarLander、Acrobot、MountainCar、Ant、HalfCheetah）以及监督学习任务中的MNIST和CIFAR-10。

**📈 对比分析**

通过鲁棒性间隙和平均奖励指标与无删除基线比较，随机删除在MLP、LSTM、GRU上分别提升约30%、6%和更小模型可超越宽Mlp；在MNIST/CIFAR实验中删除5%数据还能提高准确率。

**⚠️ 局限性**

局限性在于仅在分布不匹配且正则化适中时有效，理论假设强凸与光滑性不完全满足，删除比例需经验调优，且缺乏自动化的最优阈值选择方法。

---

## 125. When Do Diffusion Models learn to Generate Multiple Objects?

**arXiv ID:** 2605.00273 | [PDF](https://arxiv.org/pdf/2605.00273v1)

**作者:** Yujin Jeong `[一作]` (Technical University of Darmstadt and hessian.AI), Anna Rohrbach `[通讯]` (Technical University of Darmstadt and hessian.AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过构造可控的合成数据集 Mosaic，系统分析了扩散模型在多物体生成（属性绑定、空间关系、计数）中的表现，探究了数据规模、概念失衡和组合缺失对模型概念泛化与组合泛化的影响，并将实验推广到更真实的 SPEC 与 Comfort‑Car 任务。

**💡 创新点**

创新点包括：①提出 Mosaic 框架以分离并可控生成属性绑定、空间关系和计数三类多物体概念；②在同一模型与同一训练框架下对概念失衡与组合缺失进行因果式实验，首次揭示场景复杂度对计数的主导作用；③发现计数在低数据时易失效，而空间关系的组合泛化最差；④验证这些结论在预训练模型微调与更真实场景中的可迁移性。

**🔧 技术方法**

使用的技术主要是潜在扩散模型（U‑Net 与 DiT）、预训练 VAE、条件编码器通过注意力注入、LoRA 微调、以及基于检测器和分类器的自动评估（Geneval、ResNet/CNN）。

**📊 数据集**

主要数据集：Mosaic（控制生成的合成图像）、SPEC（用于计数与空间关系的真实文本-图像对）、Comfort‑Car（更丰富的物体外观与遮挡）。

**📈 对比分析**

比较方法：在不同数据规模、分布失衡、场景复杂度和组合缺失下记录生成准确率；对计数使用记忆率、Per‑class accuracy；对空间关系使用联合准确率与混淆矩阵。实验显示：计数在 10k–50k 数据下显著下滑，增大数据或使用网格布局可提升；属性绑定与空间关系在统一分布下保持 90%+；组合泛化随缺失组合数增加而急剧下降，空间关系最为敏感。

**⚠️ 局限性**

局限性：①扩散模型缺乏针对多物体的结构化先验，导致计数不稳与空间关系难以组合；②实验主要基于合成或受限真实数据，未覆盖更复杂场景与更大物体种类；③未探索更强的结构化条件或布局控制方法的潜力；④评估仍依赖自动分类器，可能忽略细微语义错误。

---

## 126. Task-Conditioned Uncertainty Costmaps for Legged Locomotion

**arXiv ID:** 2605.00261 | [PDF](https://arxiv.org/pdf/2605.00261v1)

**作者:** Kartikeya Singh `[一作]` (University at Buffalo), Karthik Dantu `[通讯]` (University at Buffalo)

**通讯引用:** 1884 | [OpenAlex ID](https://openalex.org/A5032635242)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文提出了一种基于任务条件的表面不确定性模型，能够从地形高度扫描和指令速度预测四足机器人的未来足位，并将估计的不确定性转换为成本图，用于路径规划。

**💡 创新点**

创新点在于将表面粗糙度与指令运动相结合，利用集成+MC dropout 估计表面不确定性，实现在有限训练分布下单模型即可识别 ID 与 OOD 区域，并以此不确定性作为规划成本。

**🔧 技术方法**

使用技术包括深度监督学习网络预测未来足位并输出贝叶斯不确定性、IsaacSim 进行仿真训练、MPPI 与 NAV2 导航框架集成、以及基于稳定性边缘的可行性评估。

**📊 数据集**

数据集来自 IsaacSim 生成的平坦地形训练集以及在真实世界 Unitree Go1 机器人上收集的多种高度平台、斜坡和粗糙地形实验数据。

**📈 对比分析**

在模拟与实地实验中与几何粗糙度成本与障碍阈值成本做对比，结果显示不确定性成本图在可行性误差上降低了约 37%，规划路径更安全且目标进展更稳定。

**⚠️ 局限性**

限制包括：仅在规划层使用不确定性，未对低层控制进行闭环适配；训练仅覆盖平坦地形与固定速度，导致对动态速度变化和更复杂地形的适应性仍有限。

---

## 127. Rethinking Network Topologies for Cost-Effective Mixture-of-Experts LLM Serving

**arXiv ID:** 2605.00254 | [PDF](https://arxiv.org/pdf/2605.00254v1)

**作者:** Junsun Choi `[一作]` (University Of California Berkeley), Borivoje Nikolic `[通讯]` (University Of California Berkeley)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Mixture-of-Experts LLM推理时不同网络拓扑的成本效能，并给出了跨层次的综合评估方法。

**💡 创新点**

首次系统性比较四种代表性XPU网络拓扑的成本效能，发现低成本无交换器拓扑（3D全网状）在多种工作负载下比传统规模化网络更具性价比，并指出现有高带宽网络已超额配置。

**🔧 技术方法**

结合alpha‑beta模型、NCCL基准、交叉层模拟、软件优化（双批次重叠、投机解码）以及未来GPU规格预测。

**📊 数据集**

以DeepSeek‑V3（671B参数、256个专家、top‑8路由）为代表性模型，使用不同上下文长度和TPOT目标进行实验，验证模型。

**📈 对比分析**

对比四种拓扑在给定成本下的吞吐量/成本比，结果显示3D全网状拓扑在20.6–56.2%范围内提升成本效能，且在未来GPU代际中仍保持优势；对不同带宽、规模、软件优化进行全面评估。

**⚠️ 局限性**

主要限制包括基于模型的推理时长估计对真实硬件的近似；对未来代际的带宽与软件开销假设；未覆盖更大规模或多租户环境下的网络拓扑复杂性。

---

## 128. Alethia: A Foundational Encoder for Voice Deepfakes

**arXiv ID:** 2605.00251 | [PDF](https://arxiv.org/pdf/2605.00251v1)

**作者:** Yi Zhu `[一作]` (Reality Defender Inc), Surya Koppisetti `[通讯]` (Reality Defender Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 Alethia，一个专门针对语音深度伪造（voice deepfake）检测、定位、源追踪、以及音视频深度伪造任务的基础编码器，并在 5 个任务、56 个基准数据集上进行了广泛评估。

**💡 创新点**

创新点包括：① 将瓶颈式掩码嵌入预测与流匹配（OT‑CFM）谱图重建两种自监督目标并行使用；② 采用连续嵌入作为预测目标，避免离散化带来的信息损失；③ 通过多层 teacher embedding 以及 bottleneck 架构，让学生模型能够学习低层声学到高层语义的多级信息；④ 在预训练阶段加入 3D 掩码（CNN 输出+Transformer 层级掩码）以强化对深度伪造痕迹的学习。

**🔧 技术方法**

使用的技术包括：CNN+Transformer 语音编码器；teacher‑student 蒸馏；bottleneck masked embedding prediction；Conditional Flow Matching（OT‑CFM）谱图重建；连续嵌入目标；多层掩码策略；两支并行预训练分支；以及大规模自监督训练。

**📊 数据集**

预训练数据：19k 小时高质量真实与深度伪造语音（来源包括 CommonVoice、ASVspoof5、MLAAD、M‑AILABS、TITW‑hard、SpoofCeleb、ShiftySpeech，以及使用 TTS/VC 生成的假声）。评估数据：56 个公开数据集，覆盖 Speech Deepfake Detection、Singing Voice Deepfake Detection、Partially Fake Speech Localization、Source Tracing、Audio‑Visual Deepfake Detection 等 5 个任务。

**📈 对比分析**

与同等规模的四个主流基础模型（Wav2vec‑XLSR‑300M、WavLM‑Large、HuBERT‑Large、Wav2vec‑XLSR‑1B）在同一微调配置下对比，Alethia 在平均 EER、准确率和方差等指标均显著优于对手。例如：SDD‑Expanded‑Aug 中 Alethia‑Large EER 5.2%/准确率 93.3%，相较 W2V‑1B 的 EER 6.0%/准确率 91.9%；SVDD zero‑shot EER 10.8% vs 13.2%；PFSL 19.8% vs 20.1%；ST silhouette +0.02 vs -0.10；AVDD 6.3%/7.1% vs 8.1%/13.9%。此外 Ablation 结果表明，生成式分支与 bottleneck 架构对性能提升具有重要贡献。

**⚠️ 局限性**

局限性包括：① 仍需针对具体任务的微调，虽然 Alethia 的通用性强但在极端低质量或新型攻击下表现尚未完全验证；② 预训练依赖大规模（19k 小时）高质量语音及复杂的掩码与流匹配，训练成本较高；③ 对抗鲁棒性与跨语言、跨口音的极限情况未做深入评估；④ 在传统基于合成方法的伪造上提升有限；⑤ 目前未针对多模态（音视频）同步时序的细粒度检测进行专门设计。

---

## 129. Joint Accuracy and Confidentiality in Semantic-Aware Secure Remote Reconstruction

**arXiv ID:** 2605.00258 | [PDF](https://arxiv.org/pdf/2605.00258v1)

**作者:** Bowen Li `[一作]` (Linköping University), Nikolaos Pappas `[通讯]` (Linköping University)

**通讯引用:** 3980 | [OpenAlex ID](https://openalex.org/A5084740578)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出并分析了在语义感知的安全远程重建中同时实现准确性与机密性的联合指标 CRA，并给出了其长期平均值与最优传输概率的闭式表达式。

**💡 创新点**

创新点在于将合法接收者的正确重建与窃听者的失败视为同一更新过程的联合事件，直接定义 CRA 并利用三维马尔可夫链进行分析，揭示传统边缘指标无法捕捉的非平凡结构特性。

**🔧 技术方法**

采用随机化稳态传输策略、二元马尔可夫源模型、独立的成功/失效通道模型，并通过三维状态转移概率矩阵求解平稳分布和 CRA 的期望。

**📊 数据集**

使用仿真产生的合成数据（源状态、通道成功率等）进行 Monte Carlo 验证，并通过 3GPP UMi 传播模型生成空间成功概率地图。

**📈 对比分析**

通过与边缘准确率、机密率及其加权组合进行比较，实验表明 CRA 能更准确地定位最优传输概率，且传统方法往往低估联合性能并给出错误的最优策略。

**⚠️ 局限性**

局限性包括仅考虑随机化稳态策略、假设源与通道独立且二元，未涵盖更复杂的源动态、时变通道或多天线/高级物理层安全技术，以及对真实环境中多样化窃听者行为的适应性不足。

---

## 130. VitaLLM: A Versatile and Tiny Accelerator for Mixed-Precision LLM Inference on Edge Devices

**arXiv ID:** 2605.00320 | [PDF](https://arxiv.org/pdf/2605.00320v1)

**作者:** Zi-Wei Lin `[一作]` (National Yang Ming Chiao Tung University), Tian-Sheuan Chang `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 3728 | [OpenAlex ID](https://openalex.org/A5067793643)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了面向边缘设备的混合精度LLM加速器VitaLLM，结合乘法器无关的TINT核心和可复用的BoothFlex核心，实现了三值权重模型的高效推理；

**💡 创新点**

核心创新包括：1) 乘法器无关的TINT核心与共享的BoothFlex核心，提升硬件利用率；2) 基于Leading‑One的预测稀疏注意力与无比较器top‑K选择，显著降低KV缓存访问；3) 头级流水线与absmax量化屏障，统一多精度接口并隐藏非线性运算延迟；

**🔧 技术方法**

使用了乘法器无关的选择器、基于radix‑4 Booth的双模运算、Leading‑One预测、无比较器top‑K选择、输出驻留数据流、头级流水线、absmax量化屏障、位串行计算与DMA缓冲管理等技术；

**📊 数据集**

以3B参数的BitNet b1.58模型作为评估基准，未引入额外训练数据集；

**📈 对比分析**

与ASIC Slim‑Llama、FPGA TerEffic、FPGA TeLLMe 等方案对比，16 nm原型在1 GHz/0.8 V下实现了72.46 tokens/s的解码吞吐量、0.88 s的64‑token预填充时间，功耗59.12 mW、面积0.214 mm²、120 KB SRAM，显示出在边缘场景下更优的功耗/面积比；

**⚠️ 局限性**

局限性在于仅针对3B模型验证，较大模型扩展受限；对极长序列的处理仍需进一步优化；功耗虽低，但在更高频率或多核规模化时仍面临挑战。

---

## 131. Artificial-Noise Aided Design for Movable-Antenna Enabled Physical-Layer Service Integration

**arXiv ID:** 2605.00306 | [PDF](https://arxiv.org/pdf/2605.00306v1)

**作者:** Zhifeng Tang `[一作]` (Australian National University), Salman Durrani `[通讯]` (Australian National University)

**通讯引用:** 5461 | [OpenAlex ID](https://openalex.org/A5013548437)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种结合可移动天线（MA）与人工噪声（AN）的物理层服务集成（PLSI）方案，旨在在满足多播可靠性的前提下提升保密性能；

**💡 创新点**

创新点在于①利用MA空间重构实现多自由度的信道重塑；②在结构化信号设计下推导出AN与机密信号功率分配的闭式解，显著降低设计复杂度；③基于BCA的低复杂度迭代算法实现MA位置与发射变量的联合优化；

**🔧 技术方法**

采用的技术包括：几何多径信道模型、最大比速率（MRT）和零泄漏AN设计、结构化beamformer、AN功率分配闭式公式、块坐标上升（BCA）算法、局部搜索更新MA位置；

**📊 数据集**

实验数据基于仿真，设置频率2.8GHz、波长λ、BS到UE距离70m、信噪比-104dBm、功率上限5dBm、4个MA、不同路径数L=4/8，未使用公开数据集；

**📈 对比分析**

与固定天线MRT+AN方案以及无AN的MA启用方案进行对比，结果显示：①在AN功率分配最优时可获得最高保密速率；②BCA迭代收敛速度快，仅需数次迭代；③在MA数量≥2时本方案相较基线显著提升保密速率；

**⚠️ 局限性**

限制包括：仅在理想完美CSI条件下验证；算法局部最优，难以全局最优；对天线数、天线间距及天线平台尺寸的依赖较大；仿真环境简化，未考虑多用户/多Eve情景。

---

## 132. Polymorphism Meets DHOL

**arXiv ID:** 2605.00295 | [PDF](https://arxiv.org/pdf/2605.00295v1)

**作者:** Rhea Ranalter `[一作]` (University of Innsbruck), Cezary Kaliszyk `[通讯]` (University of Melbourne)

**通讯引用:** 2419 | [OpenAlex ID](https://openalex.org/A5070698573)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可多态的依赖型高阶逻辑（PDHOL），并给出了其语法、语义和证明系统；通过将PDHOL翻译成多态HOL（PHOL）实现自动化证明。

**💡 创新点**

创新点在于：① 在DHOL基础上加入浅层多态和子类型定义，显著提升表达力；② 设计了基于PER的可判定翻译，使得多态与依赖型一起能被现有HOL ATP无缝利用；③ 在实现层面将翻译嵌入逻辑嵌入工具，支持TPTP标准的DHf方言。

**🔧 技术方法**

使用的技术包括：逻辑嵌入工具、PER（部分等价关系）翻译、HOL证明系统（Vampire、Leo‑III、Zipperposition）、TPTP格式、以及对类型检查约束的自动化处理。

**📊 数据集**

数据集为53个TPTP问题，其中包括18个多态版本的DHOL示例、17个类型变量实例化实验、11个红黑树逆序的归纳证明子问题、zip函数、有限集合、以及更多基于向量/列表的案例。

**📈 对比分析**

比较方法是将PDHOL问题通过翻译生成PHOL问题，并用现有ATP进行求解；实验在Intel i5-6200U 2.3GHz+8GB环境下进行，翻译时间约198±33 ms；大部分问题被证明，TCOs（类型检查约束）几乎全是简单的等价推理；但对需要深层归纳的矩阵转置等复杂命题仍超时，表明现有ATP在处理依赖型+多态的归纳时性能不足。

**⚠️ 局限性**

主要局限包括：① 依赖型类型检查不可判定，导致需要生成额外的TCOs；② 对复杂归纳（如矩阵转置、红黑树归纳）现有ATP效率低下；③ 目前不支持上界多态、受限子类型等更高级特性；④ 翻译与类型检查的额外开销在大规模问题中可能显著。

---

## 133. Caracal: Causal Architecture via Spectral Mixing

**arXiv ID:** 2605.00292 | [PDF](https://arxiv.org/pdf/2605.00292v1)

**作者:** Bingzheng Gan `[一作]` (Huawei Technologies, Co., Ltd.), Tao Yu `[通讯]` (Huawei Technologies, Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于傅里叶变换的序列建模架构Caracal，用以替代Transformer的注意力机制，实现长序列的高效自回归生成。

**💡 创新点**

将多头傅里叶模块与频域因果掩码相结合，去除显式位置编码，仅使用标准FFT和卷积，混合全局MHF与局部滑窗注意力，形成低硬件依赖、可扩展的架构。

**🔧 技术方法**

使用FFT、因果Conv1D、双阶段门控、频域因果掩码、滑窗注意力、标准Transformer残差/LayerNorm/FFN等技术。

**📊 数据集**

FineWeb-10B用于预训练，评估数据集包括LAMBADA、Hellaswag、Winogrande、ARC-E/C、PIQA、SIQA、BoolQ、SWDE、FDA等。

**📈 对比分析**

在同等参数规模（Tiny至Large）与Llama、Mamba、Mamba-2、Jamba对比，Caracal在多数任务上与Transformer同等或更优，长序列训练吞吐量提升近3倍，保持竞争力。

**⚠️ 局限性**

相较纯注意力模型，细粒度检索与信息提取能力略弱；需进一步提升对长文本检索精度；频域因果实现虽高效但在某些硬件上仍有实现细节限制。

---

## 134. Agentic AI for Trip Planning Optimization Application

**arXiv ID:** 2605.00276 | [PDF](https://arxiv.org/pdf/2605.00276v1)

**作者:** Tiejin Chen `[一作]` (Toyota Motor North America), Nejib Ammar `[通讯]` (Toyota Motor North America)

**通讯引用:** 664 | [OpenAlex ID](https://openalex.org/A5084763315)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种分层agentic AI系统，用于车辆智能行程规划优化，并提出了具有确定性最优解的TOP数据集；

**💡 创新点**

核心创新在于引入中心Orchestration Agent对多专门化Agent进行动态协作与自我纠错，并提供细粒度、可复现的最优评估基准；

**🔧 技术方法**

技术上采用GPT‑4o为基础模型，结合Microsoft Autogen框架构建多Agent架构，配合交通、充电、POI等专门化Agent进行实时推理与优化；

**📊 数据集**

使用自构造的Trip‑Planning Optimization Problems Dataset（TOP），包含500条问答对、15个认知类别和3个难度层级，并附带确定性的最优答案；

**📈 对比分析**

通过与单一LLM和SWARM多Agent基线在TOP上对比，采用准确率作为指标，系统整体准确率为77.4%，远高于单体30.4%和SWARM 23.6%；

**⚠️ 局限性**

局限在于依赖人工工作流生成答案，缺乏对开放式自然语言的泛化能力，且评估仅关注时间最优，未涵盖能耗等多目标因素。

---

## 135. Jailbroken Frontier Models Retain Their Capabilities

**arXiv ID:** 2605.00267 | [PDF](https://arxiv.org/pdf/2605.00267v1)

**作者:** Daniel Zhu `[一作]` (Anthropic), Jerry Wei `[通讯]` (Anthropic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了28种解禁攻击（jailbreak）对五个生物学基准（GPQA Diamond、WMDP Bio、LabBench ProtocolQA、CloningScenarios、VCT Text）以及五种Claude模型（Haiku 4.5、Sonnet 4、Sonnet 4.5、Opus 4.5、Opus 4.6）在安全沙箱下的能力下降（jailbreak tax）并探究其与模型能力、任务推理需求、攻击复杂度等因素的关系。

**💡 创新点**

创新点在于将jailbreak tax从传统模型扩展到前沿模型，揭示税负随模型能力反比缩小，并首次发现最强攻击 Boundary Point Jailbreaking（BPJ）可在几乎不降低能力的前提下实现高逃逸率，挑战了传统安全案例中“能力下降”可作为防护依据的假设。

**🔧 技术方法**

使用多轮提示策略（10种）生成最优得分；对每种jailbreak采用模板化转换与加密编码；通过token计数与熵作为复杂度指标；构建BPJ黑盒优化算法产生通用前缀；对所有模型、基准、攻击组合进行大规模批量评估。

**📊 数据集**

数据集包括：GPQA Diamond、WMDP Bio、LabBench ProtocolQA、CloningScenarios、VCT Text；共计约5,000道多选题。

**📈 对比分析**

评价方法：对每组模型-基准-攻击-提示组合计算准确率，取最大准确率作为攻击效果；通过与无攻击基线对比得到相对降解率；将降解率与模型能力、推理需求、token增量等进行相关性分析；BPJ的逃逸率与降解率在二维图上与其他攻击对比，显示其处于高逃逸低降解区间。结果表明：Haiku 4.5平均降解≈33%，Opus 4.6最大思考仅≈7%；BPJ降解≤2%而逃逸率≥92%。

**⚠️ 局限性**

局限性：仅评估单一模型族Claude；攻击多为已公开的“naive”jailbreak；使用无害生物学基准作为间接测量，假设其能代表有害请求；未覆盖化学、核、放射等高危领域；未深入探究非密码攻击的机制；结果可能因模型更新或新攻击而变化。

---

## 136. Lost in State Space: Probing Frozen Mamba Representations

**arXiv ID:** 2605.00253 | [PDF](https://arxiv.org/pdf/2605.00253v1)

**作者:** Bhagyashree Wagh `[一作]` (University of Washington), Akash Singh `[通讯]` (University of Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在冻结的Mamba-130M语言模型上，探索使用不同提取策略（Patched-Mamba、Mean Pool、Final State、Orthogonal Injection）从递归状态获得句子级表示，并通过多任务（SST‑2、CoLA、MRPC、STS‑B、IMDb）进行评测。

**💡 创新点**

发现预训练Mamba的原始递归状态高度各向异（mean cosine 0.9999），导致无投影的Final State在CoLA上完全崩溃（MCC=0.000）；同时证明patch边界读出的句子表示并未显著优于简单均值池化。

**🔧 技术方法**

使用冻结特征探针（probe head），对四种提取方法进行10轮训练；引入正交注入（Orthogonal Injection）修改递归写入方向，并在推理时评估其效果。

**📊 数据集**

评估数据集包括五个NLP基准：SST‑2、CoLA、MRPC、STS‑B（无监督相似度）、IMDb；同时对RoBERTa‑base进行对照实验。

**📈 对比分析**

相较于RoBERTa‑base，Mamba在所有任务中均未达到或低于其性能；Patch‑Mamba在MRPC上略优于Mean Pool，但在CoLA和STS‑B上表现不佳；Final State在CoLA完全失效；Orthogonal Injection在推理时未能降低各向异性，甚至导致Spearman下降。

**⚠️ 局限性**

局限性包括：仅使用单一Mamba‑130M模型；SST‑2和IMDb结果仅单随机种子；不同patch尺寸对崩溃现象的影响未研究；Orthogonal Injection仅在推理时尝试，缺乏训练时的改进验证。

---

## 137. Trident: Improving Malware Detection with LLMs and Behavioral Features

**arXiv ID:** 2605.00297 | [PDF](https://arxiv.org/pdf/2605.00297v1)

**作者:** Rebecca Saul `[一作]` (University of California, Berkeley), David Wagner `[通讯]` (University of California, Berkeley)

**通讯引用:** 21093 | [OpenAlex ID](https://openalex.org/A5062174672)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文提出了Trident系统，将LLM自动生成的行为检测规则、静态特征的GBDT分类器和LLM对沙箱报告的直接判定通过多数投票结合，以提升恶意软件检测的准确性与对概念漂移的鲁棒性。

**💡 创新点**

创新点在于利用大型语言模型自动识别并写出可验证的行为规则，并在三种判定机制之间采用阈值+投票融合，实现了无需频繁重训练即可保持高性能与可解释性的检测框架。

**🔧 技术方法**

核心技术包括Gemini‑3‑Pro‑Preview LLM对行为报告进行分析与规则生成、基于JQ语法的规则执行与验证、GBDT对静态PE特征的分类、以及阈值+多数表决的融合策略。

**📊 数据集**

实验使用BODMAS恶意软件数据集（约57,000恶意和77,000良好PE文件）及其对应的沙箱行为报告（从VirusTotal获取的七大沙箱）。

**📈 对比分析**

通过与单一静态GBDT、单独行为规则+LLM、以及每月重新训练的Active Learning基线进行对比，Trident在F1、召回率与误报率上均优于各单独方法，并在概念漂移测试中保持误报率低于0.1%，与每月重训练模型的表现相当。

**⚠️ 局限性**

主要限制包括LLM知识截止与潜在数据泄漏风险、对规则生成与去重的依赖（约38%规则被丢弃）、以及对沙箱行为报告完整性与逃逸的敏感性，仍需提升规则质量与处理随机匹配问题。

---

## 138. An End-to-End Decision-Aware Multi-Scale Attention-Based Model for Explainable Autonomous Driving

**arXiv ID:** 2605.00291 | [PDF](https://arxiv.org/pdf/2605.00291v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 139. Online Self-Calibration Against Hallucination in Vision-Language Models

**arXiv ID:** 2605.00323 | [PDF](https://arxiv.org/pdf/2605.00323v1)

**作者:** Minghui Chen `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Qingyi Si `[通讯]` (JD.COM)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了在线自校准框架OSCAR，利用MCTS和双粒度奖励机制生成可自监督的偏好数据，针对视觉语言模型的幻觉问题进行迭代优化。

**💡 创新点**

创新点在于识别并解决“监督‑感知不匹配”与“生成‑判别差距”，通过在线MCTS生成的高质量偏好对抗训练，避免离线教师强制细节逼真导致的幻觉。

**🔧 技术方法**

使用了Monte Carlo Tree Search、双粒度奖励（节点级过程奖励与轨迹级门控奖励）、Direct Preference Optimization（DPO）以及LoRA微调等技术。

**📊 数据集**

采用的主要数据集包括LLaVA-150k、COCO、Object‑HalBench、AMBER、MM‑VET、POPE等幻觉评估基准。

**📈 对比分析**

与现有SOTA方法（STIC、POVID、SIMA）以及开源LVLM（InstructBLIP、MiniGPT‑4、mPLUG‑Owl2）进行对比，OSCAR在生成和判别幻觉指标（如CHAIR、Hal、Cog）显著下降，并在多个基准上实现或逼近最优性能。

**⚠️ 局限性**

局限性包括对模型自身判别能力的依赖，低分辨率或极细粒度图像的幻觉抑制效果仍有限；MCTS搜索开销较大，可能限制大规模推理的实时性。

---

## 140. Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models

**arXiv ID:** 2605.00321 | [PDF](https://arxiv.org/pdf/2605.00321v1)

**作者:** Hanxin Zhang `[一作]` (University of Leicester), Zhou Daniel Hao `[通讯]` (University of Leicester)

**通讯引用:** 909 | [OpenAlex ID](https://openalex.org/A5100397083)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种基于因果干预的可解释性方法（Interventional Significance Score，ISS）来估计视觉区域对动作预测的因果影响，并提出了Nuisance Mass Ratio（NMR）来量化模型对任务无关特征的依赖，二者共同用于诊断Vision‑Language‑Action（VLA）策略的因果对齐与泛化能力。

**💡 创新点**

创新点在于：
1) 把视觉‑动作归因问题视为因果干预估计，首次定义了ISS；
2) 提出了基于ISS的NMR指标，可量化策略对噪声/无关特征的依赖；
3) 理论上证明ISS可无偏估计且在固定高斯策略下可用动作MSE近似；
4) 通过对照实验显示NMR与OOD成功率呈显著负相关，ISS在鲁棒性与可信度上优于注意力/词向量基准。

**🔧 技术方法**

技术手段包括：
- 结构因果模型中的干预（token级均值替换）与蒙特卡罗随机掩码；
- 计算KL散度并用动作均方误差（MSE）作为高斯策略下的代理；
- 对多视角、多时序数据进行线性插值与稀疏掩码采样；
- 利用Markov Blanket概念划分行动、支持与噪声子空间；
- 对比实验采用相关系数、余弦相似度和动作MSE评估。

**📊 数据集**

使用数据集：
- AGNOSTOS基准（包括见过的S任务和两类未见U1、U2任务）；
- RLBench仿真环境下的41个任务，用于计算成功率与NMR；
- 3600个见过任务和575个未见任务进行离线干预实验。

**📈 对比分析**

方法比较：
- ISS与传统注意力得分（ATT）和词向量范数（NORM）在鲁棒性与可信度任务中对比；
- 在Gaussian噪声、纹理、几何、补丁干预下，ISS在余弦相似度和动作MSE上均优于ATT、NORM；
- NMR@10与成功率的Pearson相关系数为-0.77，表明NMR是OOD泛化的强预测因子；
- 计算耗时：在最优配置（N=100，p=0.3）下，单帧推理0.079s，ISS推理5.18s，频率0.19Hz。

**⚠️ 局限性**

局限性：
- 干预计算开销大，实时控制不可行；
- 仅适用于离线安全验证与后验解释，无法直接作为在线决策辅助；
- ISS与NMR为近似指标，无法提供严格的推理或安全保证；
- 若误用或过度信任，可能导致对模型错误行为的误解或误导。

---

## 141. Efficient Spatio-Temporal Vegetation Pixel Classification with Vision Transformers

**arXiv ID:** 2605.00296 | [PDF](https://arxiv.org/pdf/2605.00296v1)

**作者:** Alan Gomes `[一作]` (Federal University of São Carlos), Jurandy Almeida `[通讯]` (Federal University of São Carlos)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文通过对七个关键设计维度（归一化、光谱排列、边界处理、空间上下文窗口、标记化策略、位置编码、特征聚合）进行系统消融，提出并优化了一种针对高分辨率植物生理时间序列的视觉 Transformer（ViT）框架，用于像素级植被分类。

**💡 创新点**

创新点在于证明 ViT 能在保持或提升多时相 CNN 的分类性能的同时，显著降低 FLOPs（约 14 倍）并保持参数规模与时间序列长度无关，从而实现资源受限边缘设备上的高效实时监测；同时系统性评估了标记化与上下文窗口等设计对性能的影响。

**🔧 技术方法**

技术手段包括基于 PyTorch 的 ViT 编码器（6 层、8 头、D=256）、Temporal Token（T,S）序列、学习式位置编码、CLS 聚合、黑色边界填充；实验使用 THOP 计算 FLOPs，采用 AdamW 优化器，使用 30 轮训练。

**📊 数据集**

所用数据集为巴西 Cerrado 生物群的两套高分辨率时序图像：Serra do Cipó（UAV 空中影像，13 个月时间点）和 Itirapina（塔式近地相机，36 天时间点）。

**📈 对比分析**

与当前最先进的多时相 CNN 进行对比；在 Serra do Cipó 上，ViT 取得 96.79% 的平衡准确率（仅落后 2.11%），参数 1.72M、FLOPs 0.05 GF；在 Itirapina 上，ViT 获得 61.51% 的平衡准确率（比 CNN 高 7.89%），参数 2.07M、FLOPs 0.16 GF，显著提升了效率。

**⚠️ 局限性**

局限性包括：仅在 RGB 数据上验证，未覆盖多光谱/高光谱或卫星级别分辨率；消融仅针对 1D 标记化，未探索 3D 贴块或多分支投影；仅在 Cerrado 生物群测试，推广到其他生态系统或更大时间序列需进一步验证。

---

## 142. FaceValue: Exploring Real-Time Self-View Overlays to Prompt Meaning-Oriented Self-Awareness in Remote Meetings

**arXiv ID:** 2605.00288 | [PDF](https://arxiv.org/pdf/2605.00288v1)

**作者:** Gun Woo Warren Park `[一作]`, Fanny Chevalier `[通讯]` (University of Toronto)

**通讯引用:** 3237 | [OpenAlex ID](https://openalex.org/A5029815396)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发 FaceValue 技术探针，在远程会议中为自我视图添加私有实时覆盖，帮助用户意识到其非语言线索可能被他人如何解读。

**💡 创新点**

提出“意义导向的自我意识”概念，并通过隐私保护、暗示性漫画式覆盖取代传统标签化反馈，减少认知负荷与隐私顾虑。

**🔧 技术方法**

使用情感表达识别模型 EmoFAN 与 MediaPipe 3D 姿态获取面部表情与头部运动，实时生成颜色、图标与线条覆盖。

**📊 数据集**

实验基于 13 名知识工作者在真实会议中的使用情况；模型训练使用公开数据集（如 AffectNet）但未在本研究中再次训练，评估基于参与者的日记与访谈。

**📈 对比分析**

比较方法主要为主观报告；未提供客观性能指标；大多数受试者报告意识提升与行为调整，且在高风险会议中被认为更有价值。

**⚠️ 局限性**

局限包括样本量小、仅自我报告、算法误判与可解释性差、仅关注面部与头部、对不同文化或神经多样性个体的适用性未知。

---

## 143. NLPOpt-Net: A Learning Method for Nonlinear Optimization with Feasibility Guarantees

**arXiv ID:** 2605.00260 | [PDF](https://arxiv.org/pdf/2605.00260v1)

**作者:** Bimol Nath Roy `[一作]` (Texas A&M University), MM Faruque Hasan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 NLPOpt-Net，一种通过 k 层投影确保约束满足的无监督神经网络框架，用于求解参数化非线性规划的解映射。

**💡 创新点**

创新点在于：①将投影改为局部二次近似而非欧氏距离最小化，保证下降性并更好地引导网络逼近最优；②采用无矩阵求逆的 Chambolle‑Pock 算法求解投影子问题；③利用隐式函数定理实现投影层的高效反向传播；④通过分离投影层实现 AOT 编译，显著加速推理。

**🔧 技术方法**

核心技术包括：深度前馈网络、修改后的拉格朗日一致性损失、k 层局部二次投影、Chambolle‑Pock 优化、隐式函数定理、JAX 自动微分与自定义 VJP、AOT 与 JIT 编译。

**📊 数据集**

使用合成数据集：2000 个随机生成的参数化问题（每个包含 100 维决策变量、50 条等式、50 条不等式、50 个参数），分别用于 QP、QCQP、凸 NLP、非凸 NLP。

**📈 对比分析**

与传统求解器（OSQP/SCS/SLSQP）、普通 NN、Eq‑NN、DC3 进行比较。结果显示：NLPOpt‑Net 在所有实验中实现了 0% 的约束违规、平均最优性差距 ≤0.1%，且推理时间与最优求解器相当（≈0.03–0.4 s/批），明显优于 DC3 的 150%+ 误差和约束违规。

**⚠️ 局限性**

局限性：①设计上主要针对凸问题；②对非凸问题效果依赖于约束线性且需多层投影导致计算开销增大；③需要手动调参（α、k 等）以及投影矩阵的结构化假设；④对大规模稀疏问题的扩展仍待进一步优化。

---

## 144. Remote SAMsing: From Segment Anything to Segment Everything

**arXiv ID:** 2605.00256 | [PDF](https://arxiv.org/pdf/2605.00256v1)

**作者:** Osmar Luiz Ferreira de Carvalho `[一作]` (University of Brasilia), Daniel Guerreiro e Silva `[通讯]` (University of Brasilia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一个开源管道 Remote SAMsing，能够在大尺度遥感影像上实现零样本高覆盖率分割。

**💡 创新点**

创新点在于多通道迭代掩膜、黑色遮蔽自适应阈值衰减以及无参数最佳匹配拼接，解决了 SAM2 在大图中覆盖率低和边界碎片化问题。

**🔧 技术方法**

使用 Segment Anything Model 2（SAM2）的自动掩膜生成器、逐通道黑色遮蔽、上下文填充、Union‑Find 合并等技术。

**📊 数据集**

使用 ISPRS Potsdam、Brasília 航空影像、Agri‑BR Planet MNF 伪彩色影像等七个场景，涵盖 5 cm 至 4.78 m GSD。

**📈 对比分析**

与单次 SAM2、SamGeo2、SLIC、Felzenszwalb 等基线对比，Remote SAMsing 在覆盖率 91–98%，检测率（Det@0.5）最高，边界 IoU 明显优于传统方法，甚至在 1.94 billion 像素的 Potsdam 合成图上保持性能。

**⚠️ 局限性**

局限包括对极端非自然光谱（SAR、热红外）适应性不明、对小型车等细小物体检测仍有限、处理时间较长且对 tile 大小敏感。

---

## 145. Unbox Responsible GeoAI: Navigating Climate Extreme and Disaster Mapping

**arXiv ID:** 2605.00315 | [PDF](https://arxiv.org/pdf/2605.00315v1)

**作者:** Hao Li `[一作]` (National University of Singapore), Steffen Knoblauch `[通讯]` (Heidelberg University)

**通讯引用:** 52 | [OpenAlex ID](https://openalex.org/A5059873347)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了负责 GeoAI 在气候极端和灾害映射中的四维（代表性、可解释性、可持续性、伦理）框架，并构建了涵盖数据、应用与社会三大治理范围的治理模型。

**💡 创新点**

首次将责任维度系统化为四个互联维度，并将其嵌入到治理模型中；同时强调从技术到社会层面的整体治理，而非单纯的性能优化。

**🔧 技术方法**

综述并借鉴 GeoAI、GeoShapley、视觉语言模型、LLM 推理、Green AI 等技术，用以增强可解释性、降低能耗和实现公平性。

**📊 数据集**

使用 EM‑DAT、OpenStreetMap、地球观测（EO）影像、VGI、社交媒体流等公开灾害与空间数据集作为案例与讨论依据。

**📈 对比分析**

未进行实验性对比，本文主要是概念性阐述；若采用现有技术可实现更高精度与更低碳排放，但缺乏量化评估。

**⚠️ 局限性**

局限在于缺乏实证验证、对数据偏差与跨国治理差异的深入探讨，以及对不同规模灾害场景下模型可迁移性的评估。

---

## 146. A Dirac-Frenkel-Onsager principle: Instantaneous residual minimization with gauge momentum for nonlinear parametrizations of PDE solutions

**arXiv ID:** 2605.00284 | [PDF](https://arxiv.org/pdf/2605.00284v1)

**作者:** Matteo Raviola `[一作]` (EPFL), Benjamin Peherstorfer `[通讯]` (New York University)

**通讯引用:** 4702 | [OpenAlex ID](https://openalex.org/A5027402421)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种 Dirac‑Frenkel‑Onsager 动力学，利用 Onsager 最小耗散原理注入历史记忆动量，仅在 Jacobian 的零空间方向上修正参数速度，从而解决 Dirac‑Frenkel 变分原理在非线性参数化下的非唯一性和不良条件问题。

**💡 创新点**

创新点在于：① 将参数速度的非唯一性视为 gauge 自由度；② 通过记忆变量（类似动量）与 Onsager 原理配合，只在零空间方向注入动量，既保持瞬时残差最小化，又消除参数路径的剧烈摆动；③ 该方法不需要对所有方向加正则化，避免了传统正则化带来的偏差。

**🔧 技术方法**

使用了 Dirac‑Frenkel 变分原理、Onsager 最小耗散原理、低通滤波的记忆变量、截断奇异值分解（tSVD）与随机 SVD、Euler/Runge‑Kutta 时间积分、以及多层感知机（MLP）参数化。

**📊 数据集**

实验数据集包括：1) 低维波方程（碰撞波）；2) 旋转点火波；3) 流场输运方程；4) 带电粒子 Vlasov 方程；5) 5 维 Fokker‑Planck 方程。所有实验均使用 MLP 进行参数化。

**📈 对比分析**

与多种基线方法（最小范数 Dirac‑Frenkel、Tikhonov 正则化、tSVD、TENG、NIVP、RSNG）进行比较。结果显示 DFO 在 L² 误差、均值和协方差预测误差上显著优于基线，背景误差降低，额外计算成本几乎可以忽略不计。

**⚠️ 局限性**

局限性：① 仅修正零空间方向的非唯一性，无法直接解决函数相关方向上因小奇异值导致的不良条件；② 需要手动或经验设定记忆变量的时间尺度 τ 与正则化参数 λ，缺乏统一的自适应选择策略。

---

## 147. Model-Based Reinforcement Learning with Double Oracle Efficiency in Policy Optimization and Offline Estimation

**arXiv ID:** 2605.00393 | [PDF](https://arxiv.org/pdf/2605.00393v1)

**作者:** Haichen Hu `[一作]` (Massachusetts Institute of Technology), David Simchi-Levi `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 21433 | [OpenAlex ID](https://openalex.org/A5112431388)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种在离线统计估计和规划算子上都实现稀疏调用、且能够处理无限状态与动作空间的强化学习算法。

**💡 创新点**

通过 log‑barrier / log‑determinant 正则化与受信占用度量相结合，首次实现了双算子高效的子线性 regret。

**🔧 技术方法**

基于置信区间、多臂上界、受信占用度量、离线回归/密度估计 oracle、log‑barrier / log‑determinant 正则化以及分段 epoch 调度的组合技术。

**📊 数据集**

论文未提供具体实验数据集，主要聚焦理论分析与证明。

**📈 对比分析**

与现有离线 oracle 基础方法相比，该算法将调用次数从 O(T) 或 O(SA) 降至 O(H log log T)（已知 T）或 O(H log T)（未知 T），在 tabular MDP 上实现 O(√T) regret，在线性 MDP 上得到 O(T^{4/5}) 的子线性 regret。

**⚠️ 局限性**

对线性 MDP 的 regret 仍非最优；算法假设可获得高质量离线 oracle，实际实现与参数调优仍具挑战。

---

## 148. Play and Learn: Gamified Feedback for Ultrasound-Guided Catheter Insertion Training in Virtual Reality

**arXiv ID:** 2605.00389 | [PDF](https://arxiv.org/pdf/2605.00389v1)

**作者:** Mohammad Raihanul Bashar `[一作]` (Concordia University), Anil Ufuk Batmaz `[通讯]` (Concordia University)

**通讯引用:** 1182 | [OpenAlex ID](https://openalex.org/A5005072681)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文开发并评估了一款基于虚拟现实的超声引导周围静脉导管插入模拟器，并通过在视觉、听觉和游戏化元素中提供实时、与任务紧密相关的反馈，帮助学习者在保持程序真实性的前提下提升技能。

**💡 创新点**

创新点在于将游戏化反馈（星级奖励、对齐窗口、进度摘要等）与医学程序目标语义对齐，形成一种形成性学习工具；此设计在保持高程序真实性的同时，兼具提升学习效率、降低工作负荷和增强使用体验的双重优势，并在新手与专家两类人群上验证其有效性。

**🔧 技术方法**

技术实现基于Unity 2022.3与PlusToolkit超声模拟器，利用Meta Quest 3的6DoF控制器和OpenIGTLink实现实时超声图像渲染；游戏化反馈通过自定义UI、音效与动态提示实现；实验环境中还使用了高精度的前臂模型和标准化的VR交互场景。

**📊 数据集**

研究未使用公开数据集，全部实验采用自建的前臂模拟模型和内部生成的超声图像；实验对象包括24名新手和12名临床专家，构成实验数据。

**📈 对比分析**

通过在标准与游戏化两种条件下的随机对照实验（新手：24人，专家：12人），使用插入时间、SUS、NASA‑TLX和PXI等量化指标进行比较。结果显示，游戏化组在新手中的插入时间从57.4 s降至46.6 s，专家组从75 s降至45 s；SUS评分和NASA‑TLX工作负荷均显著下降；PXI体验得分亦显著提升。

**⚠️ 局限性**

局限性包括：缺乏触觉反馈与真实组织阻力模拟；解剖结构单一、缺乏变异性；专家样本量小、实验仅为即时效能评估，未考察长期学习与转移效果；游戏化提示在真实临床中可能导致对外部环境的依赖；未探索社交竞争或更丰富的游戏机制。

---

## 149. PILIR: Physics-Informed Local Implicit Representation

**arXiv ID:** 2605.00385 | [PDF](https://arxiv.org/pdf/2605.00385v1)

**作者:** Jianfeng Li `[一作]` (Wuhan University), Ke Tang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 15242 | [OpenAlex ID](https://openalex.org/A5021254405)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为PILIR的物理信息本地隐式表示方法，用于解决PINNs中的谱偏差问题，改进高频细节的学习速度和精度。

**💡 创新点**

核心创新在于将物理域离散为可学习的网格特征空间，并通过神经算子对局部离散特征进行连续生成，实现高频细节的子网格级合成，突破传统插值的凸包限制。

**🔧 技术方法**

采用网格编码+基于坐标的MLP生成器+轻量级物理解码头的编码‑解码框架，配合自适应权重与余弦平滑，实现C∞连续的物理场预测；并使用标准的Physics‑Informed Loss进行训练。

**📊 数据集**

实验涵盖二维/三维Helmholtz、Allen–Cahn、对流、反应‑扩散、Navier‑Stokes等多尺度 PDE 任务，使用随机种子（100~500）生成训练/验证样本。

**📈 对比分析**

与PINN、Wavelet、PIXEL、PIG等基线对比，PILIR在相同或更低网格分辨率下，均取得显著更低的相对L2误差（例如Helmholtz‑3D从0.247→0.0504，Allen‑Cahn从0.365→0.0294），并在高频谱分析中恢复更多重要频率分量。

**⚠️ 局限性**

局限性包括：在三维高分辨率任务中仍可能出现内存耗尽（PIG为OOM），以及对极端高频或非光滑解的鲁棒性尚未充分验证；此外，当前仅针对标量/向量场的 PDE，扩展到更复杂多物理耦合问题仍需进一步研究。

---

## 150. An eHMI Presenting Request-to-Intervene and Takeover Status of Level 3 Automated Vehicles to Support Surrounding Traffic Safety

**arXiv ID:** 2605.00377 | [PDF](https://arxiv.org/pdf/2605.00377v1)

**作者:** Hailong Liu `[一作]` (Nara Institute of Science and Technology), Takahiro Wada `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 10655 | [OpenAlex ID](https://openalex.org/A5048756005)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究在驾驶模拟器中对比了三种外部人机界面（eHMI）设计，以评估其在Level 3自动驾驶车辆（AV）发出“请求干预”（RtI）时对周围手动车（MV）驾驶员认知、行为和事故风险的影响。

**💡 创新点**

创新点在于提出并验证了将RtI状态与驾驶员接管进度通过两条光条（青色表示ADS运行，橙色闪烁表示RtI与接管完成）外部化的eHMI（C+O），从而显著提升周围驾驶员的情境感知与防御性驾驶。

**🔧 技术方法**

采用光条式外部显示技术，并通过混合效应模型、ART ANOVA、贝叶斯网络与结构方程模型等统计方法对实验数据进行深入分析。

**📊 数据集**

使用了40名日本驾驶员在三种eHMI条件下完成的12次混合交通场景驾驶模拟数据，记录了主观评估、时间头距（THW）变化及事故发生情况。

**📈 对比分析**

与无eHMI（N）和仅ADS状态eHMI（C）比较后，C+O条件在认知评分、犹豫度、信心度、THW提升以及事故发生率方面均表现显著优势，事故率降低约76.8%。

**⚠️ 局限性**

主要限制包括：实验仅在仿真环境进行，真实道路验证缺失；样本量有限、事故事件稀疏；以及对突发RtI、不同交通复杂度下eHMI效能的研究不足。

---

## 151. Agentic AI for Substance Use Education: Integrating Regulatory and Scientific Knowledge Sources

**arXiv ID:** 2605.00383 | [PDF](https://arxiv.org/pdf/2605.00383v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 152. AlphaInventory: Evolving White-Box Inventory Policies via Large Language Models with Deployment Guarantees

**arXiv ID:** 2605.00369 | [PDF](https://arxiv.org/pdf/2605.00369v1)

**作者:** Chenyu Huang `[一作]`, Lai Wei `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了 AlphaInventory 框架，利用大语言模型（LLM）生成可解释的白盒库存策略，并通过置信区间校验实现部署安全保证。

**💡 创新点**

创新点包括：①将 LLM 生成与置信区间校验相结合，构建训练–推理–部署的统一保证链；②提出 AlphaInventory 的 LLM‑guided proposal search 机制；③在经典库存环境中演化出 Tilted‑CBS 与 Tilted‑PIC 等新结构策略。

**🔧 技术方法**

技术手段包括：大语言模型（如 GLM‑4.7‑Flash）与强化学习（GRPO）训练；LLM‑guided proposal search、置信区间统计与 IPM 区间误差估计；Bayesian 优化与模拟器评估。

**📊 数据集**

使用的数据集：①47 条种子合成数据（覆盖 15 行业），通过切片扩展得到 470 个工作区；②真实零售数据 Dunnhumby 完整旅程（2.6 M 交易）。

**📈 对比分析**

比较方法：与五个经典库存基线（base‑stock、capped base‑stock、constant order、newsvendor、(s,S)）以及两种深度 RL/深度学习基线（A3C、E2E）对比；结果显示 AlphaInventory 在 30 个 OOD 合成测试中 83% 超越最佳基线，Dunnhumby 30 个工作区 67%；在经典 CBS 基准中通过迭代演化得到 Tilted‑CBS/Tilted‑PIC，平均比 CBS 低 2% 以上，击败率约 92%，并展现更好的跨域泛化。

**⚠️ 局限性**

局限性：①需要较大计算资源，且深度学习基线在短样本下表现差；②模型对特定特征（如文本/结构）依赖，特征不统一时效果受限；③部署误差仍受 IPM 估计影响；④训练样本有限可能导致过拟合风险。

---

## 153. Towards Interactive Multimodal Representation of ML Functions for Human Understanding of ML

**arXiv ID:** 2605.00357 | [PDF](https://arxiv.org/pdf/2605.00357v1)

**作者:** Bokang Wang `[一作]` (Carnegie Mellon University), Yigang Wen `[通讯]` (Carnegie Mellon University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过交互式可视化三种机器学习概念（K-means聚类、音频傅里叶变换+触觉反馈、Q-learning强化学习）提高公众对AI的理解和兴趣

**💡 创新点**

将传统算法与多感官沉浸式技术（VR、触觉、叙事游戏）结合，创造可体验、可互动的学习场景；通过手指映射音符、故事化强化学习等新颖表现方式提升可感知性

**🔧 技术方法**

Unity 3D、Oculus Quest 2、Leap Motion、STRATOS Ultrahaptics、FFT音频分析、Q‑learning算法、可定制化VR与触觉交互框架

**📊 数据集**

彩色绘画的RGB像素数据、音频文件（用于FFT分解）、基于离散状态空间的Q‑learning环境（如“恐龙+食物+熔岩”游戏地图）

**📈 对比分析**

通过20名参与者的体验测试评估理解度与兴趣提升，结果显示大部分玩家能正确关联视觉隐喻并认为叙事增强了学习效果；然而未给出数值性能指标，仅报告定性改进

**⚠️ 局限性**

缺乏大规模、量化的用户研究；触觉与视觉隐喻在不同人群中的效果不一；实验数据有限，未能系统评估算法效率或可扩展性

---

## 154. VQ-SAD: Vector Quantized Structure Aware Diffusion For Molecule Generation

**arXiv ID:** 2605.00354 | [PDF](https://arxiv.org/pdf/2605.00354v1)

**作者:** Farshad Noravesh `[一作]` (Monash University), Arghya Pal `[通讯]` (Monash University)

**通讯引用:** 231 | [OpenAlex ID](https://openalex.org/A5059800081)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

利用 VQ‑VAE 对分子原子和键进行离散化编码，并以此作为符号化输入，再结合结构感知的扩散模型进行分子生成。

**💡 创新点**

创新点在于将符号化的离散码与结构编码相结合，学习可调的前向扩散过程，并冻结 VQ‑VAE tokenizer，显著缓解类别不平衡与状态冲突问题。

**🔧 技术方法**

主要技术包括 VQ‑VAE、可学习前向扩散、RRWP 结构编码、图同构网络、Gumbel softmax 等。

**📊 数据集**

在 QM9 与 ZINC250k 两大分子数据集上进行训练与评估。

**📈 对比分析**

与 DiGress、MELD 等基准相比，VQ‑SAD 在有效率、唯一性、FCD、NSPDK 等指标上表现更优，碰撞率显著降低。

**⚠️ 局限性**

限制在于条件生成时容易收敛到少数高概率模式，导致多样性下降；模型对极端结构的泛化仍有限。

---

## 155. Social Bias in LLM-Generated Code: Benchmark and Mitigation

**arXiv ID:** 2605.00382 | [PDF](https://arxiv.org/pdf/2605.00382v1)

**作者:** Fazle Rabbi `[一作]` (Concordia University), Jinqiu Yang `[通讯]` (Concordia University)

**通讯引用:** 1443 | [OpenAlex ID](https://openalex.org/A5101712379)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型（LLM）在生成面向人类的代码时出现的社会偏见，并提出了一套完整的评估与缓解框架。

**💡 创新点**

创新点包括：①构建了包含 343 个真实任务、涵盖 7 个人口维度的 SocialBias‑Bench 基准；②设计了基于变形测试的 Solar 公平评估框架；③提出了无测试 oracle 的 Fairness Monitor Agent，可插拔地在任意代码生成流水线中检测并修复偏见；④系统性比较了单体模型、提示工程、结构化多代理流程以及新方案的偏见抑制效果。

**🔧 技术方法**

技术手段主要包括：变形测试（metamorphic testing）自动生成可执行的公平性检测用例；多代理框架 FlowGen 的 Waterfall/Scrum 结构化流程；静态 LLM 代码分析与结构化修复的 Reviewer‑Repairer 循环；以及 Chain‑of‑Thought、正面角色扮演等提示工程方法。

**📊 数据集**

使用的数据集是 SocialBias‑Bench，涵盖 343 个涉及可访问社会福利、大学录取、雇员发展、健康检查、许可证、爱好与职业等七类的、包含敏感与相关属性的 Python 类方法任务。

**📈 对比分析**

通过对四个主流 LLM（Codex、CodeGen、StarCoder、Code‑Llama）的基准测试，发现基线偏见严重（最高 60.58% CBS）。提示工程（CoT、P‑CoT）反而放大偏见。结构化多代理工作流能降低偏见（Waterfall CBS 24.49%），但分散公平责任会适得其反。Fairness Monitor Agent 在 3 轮修复后将 CBS 降至 16.91%（降低 65.1%），同时将功能正确率 Pass@attribute 从 75.80% 提升至 83.97%。

**⚠️ 局限性**

局限性包括：①对“就业状态”偏见的静态分析难以完全消除，因其隐式关联导致检测不到；②评估仅限于单文件 Python 类方法，无法推广到多文件或非 Python 语言；③实验仅覆盖 343 个任务和 4 个模型，无法全面验证在更大规模或更新模型上的表现；④当前方法仍依赖任务描述与类型信息，若任务格式不规范可能失效。

---

## 156. ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning

**arXiv ID:** 2605.00380 | [PDF](https://arxiv.org/pdf/2605.00380v1)

**作者:** Zihan Lin `[一作]` (Chinese Academy of Sciences), Guojun Yin `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在RLVR框架下提出ResRL，通过负样本投影残差对抗梯度冲突，提升LLM推理与多样性。

**💡 创新点**

创新点是用正负子空间投影残差作为梯度权重，理论上上界表示对齐并实现负样本梯度去耦合。

**🔧 技术方法**

技术包括RLVR、GRPO、NSR、SVD低秩子空间、投影残差重加权、Group-relative门控、长度缩放奖励。

**📊 数据集**

数据集涵盖12个基准：AIME24/25、AMC23、MATH-500、Minerva、Olympiad、LiveCodeBench、CodeForces、HumanEval+、ALFWorld、WebShop、BFCL，以及Qwen系列模型训练集。

**📈 对比分析**

与GRPO、DAPO、FlowRL、NSR等基线相比，在数学、代码、代理与工具调用任务上均取得Avg@16/Pass@k提升，最高在Qwen3-4B数学平均提升9.4%和Pass@128提升7.0%。

**⚠️ 局限性**

局限在子空间估计对rank、采样与长度影响敏感，且在极长序列或低rank下可能失去区分度。

---

## 157. Decoding Algorithms for Symbol-Error Correction in MDS Array Codes via Superregular Matrices

**arXiv ID:** 2605.00376 | [PDF](https://arxiv.org/pdf/2605.00376v1)

**作者:** Débora Beatriz Claro Zanitti `[一作]`, Cintya Wink de Oliveira Benedito `[通讯]` (Sao Paulo State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一套基于超正则矩阵（尤其是范德蒙德和Cauchy矩阵）构造的MDS数组码，并给出了能在未知错误位置下纠正1、2、3个符号错误的解析式算法；

**💡 创新点**

算法适用于任意参数配置的MDS数组码（不受特定参数约束），首次给出了不需先验错误位置信息即可纠正多符号错误的通用方法；

**🔧 技术方法**

利用超正则块矩阵、Frobenius伴随矩阵、Zech对数以及线性代数的结构性质，实现了低复杂度的符号错误检测与纠正；

**📊 数据集**

未使用具体数据集，算法在理论上适用于任意字母表大小𝔽_q^b，并通过实例（如[10,5,6]、[8,4,5]、[11,5,7]）验证；

**📈 对比分析**

与传统RAID 6或Reed–Solomon码相比，算法在支持多错误、任意参数和无错误位置信息时表现更灵活；由于只需求解小型线性系统和少量矩阵乘法，计算复杂度在中等参数规模下保持可接受；

**⚠️ 局限性**

主要限制是随着错误数的增加，代数表达式变得极为复杂；算法实现难度高，且在大规模参数或高符号错误率场景下的效率与鲁棒性尚未充分评估。

---

## 158. From Phreaking to Sneaking: Children's Circumvention of Social Media Age Verification Systems

**arXiv ID:** 2605.00368 | [PDF](https://arxiv.org/pdf/2605.00368v1)

**作者:** Bjorn Nansen `[一作]`, Shaanan Cohney `[通讯]` (University of Melbourne)

**通讯引用:** 435 | [OpenAlex ID](https://openalex.org/A5061801902)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对15名12-16岁澳洲儿童进行五次焦点小组访谈，研究他们对2025年实施的“未满16岁禁止社交媒体”政策的认知、体验和逃避行为，并提出“sneaking”这一新概念来解释儿童如何技术性地规避年龄验证；

**💡 创新点**

创新点在于将传统的“phreaking”文化借鉴为“sneaking”，将儿童的绕过行为视为一种基础设施政治与数字素养的展现，并系统阐述了儿童在技术规避中的集体知识共享与未来治理设想；

**🔧 技术方法**

采用质性研究方法，使用焦点小组讨论、主题分析（Braun & Clarke）以及Delve数据分析平台进行编码与归纳；

**📊 数据集**

数据来源为15名12-16岁儿童在澳洲不同学校与社区中的访谈记录，未使用公开数据集；

**📈 对比分析**

论文未采用量化对比或性能指标，而是通过主题分析得出儿童对监管无效、绕过率高的质性发现，说明监管措施在实践中效果有限；

**⚠️ 局限性**

局限性包括样本规模小、仅限城市中产阶层、仅为质性访谈、可能存在群体影响与受访者自我呈现偏差、缺乏纵向跟踪与量化验证。

---

## 159. From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing

**arXiv ID:** 2605.00358 | [PDF](https://arxiv.org/pdf/2605.00358v1)

**作者:** Wei Liu `[一作]` (National University of Singapore), Wee Sun Lee `[通讯]` (National University of Singapore)

**通讯引用:** 5768 | [OpenAlex ID](https://openalex.org/A5071864357)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的LLM参数编辑方法——前向回放（FE），用正向传播替代传统的反向扩散来构造多层目标隐藏状态。

**💡 创新点**

创新点在于将目标隐藏状态的优化点从最后决策层移动到最早决策层，并通过前向传播获取后续层的目标，实现了跨层兼容且不增加额外计算。

**🔧 技术方法**

技术包括：Locate-then-Edit (LTE)框架、梯度下降优化目标隐藏状态、前向传播恢复多层目标、与现有编辑算法（如MEMIT、BLUE、RECT、AlphaEdit、PRUNE）集成。

**📊 数据集**

使用的主要数据集是 Multi-CounterFact (MCF) 与 ZsRE，模型包括 LLaMA3-8B-Instruct、GPT-J 和 GPT-2 XL。

**📈 对比分析**

与传统的MEMIT、OneLayer、BLUE等基线相比，FE在效能（Efficacy）、泛化（Generalization）和特异性（Specificity）指标上均有显著提升，且计算成本几乎与MEMIT相同。

**⚠️ 局限性**

局限性在于编辑主要集中在较浅层，未实现层间编辑需求的完全协调，未来需探索更均衡的层级编辑策略。

---

## 160. MemRouter: Memory-as-Embedding Routing for Long-Term Conversational Agents

**arXiv ID:** 2605.00356 | [PDF](https://arxiv.org/pdf/2605.00356v1)

**作者:** Tianyu Hu `[一作]` (University of Central Florida), Song Wang `[通讯]` (University of Central Florida)

**通讯引用:** 7003 | [OpenAlex ID](https://openalex.org/A5100326206)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MemRouter，一种只在嵌入空间做写入决策的记忆路由器，分离写入与答复阶段。

**💡 创新点**

创新点是将写入决策从自回归 LLM 生成迁移到仅需前向传播的轻量级分类器，并将写路由器与答复模型解耦，使其可跨模型复用。

**🔧 技术方法**

使用冻结大型 LLM（如 Qwen2.5-7B）做特征投影，BGE-large 生成句子嵌入，轻量化投影层和分类头，教师模型生成训练标签。

**📊 数据集**

采用 LoCoMo、LongMemEval、MSC 等对话数据集进行训练、验证和测试，主要以 LoCoMo 作为评测基准。

**📈 对比分析**

在匹配的 LoCoMo harness 下与 Memory‑R1 的 LLM 管理器对比，整体 F1 52.0 远超 45.6，单跳 57.5 领先 33.6，内存决策延迟从 970 ms 降至 58 ms，吞吐率提升 11×。

**⚠️ 局限性**

仍需前向推理，处理长上下文时投影层可能成为瓶颈；对极端多轮或噪声对话的鲁棒性待验证；仅在预定义问题类别上验证，通用性需要进一步测试。

---

## 161. Negative Data Mining for Contrastive Learning in Dense Retrieval at IKEA.com

**arXiv ID:** 2605.00353 | [PDF](https://arxiv.org/pdf/2605.00353v1)

**作者:** Eva Agapaki `[一作]` (IKEA Retail (Ingka Group)), Amritpal Singh Gill `[通讯]` (IKEA Retail (Ingka Group))

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在IKEA产品搜索中，通过结构化负采样和LLM评估提升密集检索的对比学习

**💡 创新点**

提出基于产品层级分类和属性的结构化硬负采样策略，并构建可扩展的LLM‑as‑judge评估框架

**🔧 技术方法**

使用基于LLM的负样本标注、Late‑Interaction语义检索模型、硬负采样与查询扩展训练

**📊 数据集**

IKEA加拿大市场商品目录（24,350件、373叶子分类）和真实用户查询日志（204,528条）

**📈 对比分析**

与随机负采样基线对比，硬负采样模型在合成查询Cat@10提升约4.3%，在真实查询Cat@10提升约2.7%；但上线A/B测试未出现显著用户参与差异

**⚠️ 局限性**

离线评估与上线效果脱节，主要受限于查询意图广度差异、零点击率高以及对在线指标敏感度不足

---

## 162. CURE-OOD: Benchmarking Out-of-Distribution Detection for Survival Prediction

**arXiv ID:** 2605.00350 | [PDF](https://arxiv.org/pdf/2605.00350v1)

**作者:** Wenjie Zhao `[一作]` (University of Texas at Dallas), Yunhui Guo `[通讯]` (University of Texas at Dallas)

**通讯引用:** 1941 | [OpenAlex ID](https://openalex.org/A5012033269)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

建立了CURE-OOD基准，系统评估CT扫描参数变化对癌症生存预测的OOD检测与模型鲁棒性。

**💡 创新点**

首次将CT获取参数引入OOD检测，发现传统分类OOB方法在MTLR生存预测中失效，并提出基于风险曲线的HazardDev检测。

**🔧 技术方法**

采用多任务逻辑回归(MTLR)与Vision Transformer（UNETR）提取3D CT特征，并在此基础上进行OOB评分。

**📊 数据集**

使用真实的RADCURE颈癌CT数据，按像素间距、曝光时间、切片厚度、X射线管电流划分ID/OOD。

**📈 对比分析**

与多种分类OOB方法对比，HazardDev在四种生存任务的AUROC和AUPRC均显著优于其他方法，证明其适用性；传统方法表现差甚至倒置。

**⚠️ 局限性**

OOB分离度与下游生存性能下降不必然相关，且当前方法仍难实现对所有获取参数变异的鲁棒预测。

---

## 163. Odysseus: Scaling VLMs to 100+ Turn Decision-Making in Games via Reinforcement Learning

**arXiv ID:** 2605.00347 | [PDF](https://arxiv.org/pdf/2605.00347v1)

**作者:** Chengshuai Shi `[一作]` (Princeton University), Chi Jin `[通讯]` (Princeton University)

**通讯引用:** 4819 | [OpenAlex ID](https://openalex.org/A5101961985)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在长时序决策任务（如Super Mario Land，需100+回合）中，如何使用强化学习（RL）对视觉‑语言模型（VLM）进行高效微调；

**💡 创新点**

创新点包括：①提出了轻量级回合级Critic的PPO变体并引入正优势过滤，显著提升RL稳定性与样本效率；②证明预训练VLM在RL中比从零训练的经典RL更具样本效率，减少手工动作空间设计；③构建了名为“Llama-Train” 的开源多任务RL框架，融合轻量化SFT初始化与自动曲线学习，实现跨多级别的稳定训练；

**🔧 技术方法**

使用技术：PPO算法改进、轻量CNN回合级Critic、正优势过滤、自动曲线采样、SFT与RL结合、基于CoT的交互协议；

**📊 数据集**

数据集：从10个Super Mario Land关卡的5,000帧游戏视频中采样，利用GPT‑o3生成CoT教师示例进行SFT；

**📈 对比分析**

与基线比较：对比GRPO、Reinforce++、经典CNN‑PPO等；与前沿VLM（GPT‑5.4、GLM‑4.6V等）比较，训练后的模型在5个训练关卡的平均水平进度提升约5.6×（相较基线）且超过前沿模型近3×；在离线、未见关卡及跨游戏（Super Mario Bros）评估中分别提升32.2%、41.5%、23.1%；保持与基线相近的多模态通用基准性能；

**⚠️ 局限性**

局限性：训练依赖特定游戏环境，跨域迁移仍受限；模型在极长序列中可能存在探索不足或过拟合风险；缺乏对更复杂动作空间或更大规模任务的验证。

---

## 164. Free Energy Surface Sampling via Reduced Flow Matching

**arXiv ID:** 2605.00337 | [PDF](https://arxiv.org/pdf/2605.00337v1)

**作者:** Zichen Liu `[一作]` (Peking University), Tiejun Li `[通讯]` (Peking University)

**通讯引用:** 12686 | [OpenAlex ID](https://openalex.org/A5100703332)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种在集合变量(CV)空间直接采样自由能面（FES）的减维流匹配方法FES‑FM，避免在生成时进行全空间模拟。

**💡 创新点**

在CV空间直接学习运输映射，结合Jarzynski等价的非平衡重加权实现流匹配；引入基于Hessian信息的E(3)-不变谐振先验，产生物理意义的配置；通过二阶段训练（预热与主训练）提高采样效率。

**🔧 技术方法**

流匹配框架（PINN/ODE）、非平衡运输采样（NETS）、神经网络参数化速度场、Hessian‑informed harmonic prior、Jarzynski等价的重加权。

**📊 数据集**

Müller‑Brown势能、维数为50/100/200的双井高维合成势能、三粒子2D系统、四粒子3D系统（合成势能）。

**📈 对比分析**

与完整空间NETS‑P做对比，指标包括生成时间、1‑Wasserstein误差和误差/时间的倒数；在所有实验中FES‑FM均实现更低的生成时间、更小的误差和更高的误差/时间，显示显著的效率和精度提升。

**⚠️ 局限性**

需要先构造合适的CV映射和Hessian信息，训练过程依赖于非平衡轨迹，可能对大规模分子系统的复杂势能和高维CV仍有挑战；实验仅在合成和小分子系统上验证，缺乏对真实生物大分子系统的评估。

---

## 165. Budget-Aware Routing for Long Clinical Text

**arXiv ID:** 2605.00336 | [PDF](https://arxiv.org/pdf/2605.00336v1)

**作者:** Khizar Qureshi `[一作]` (MIT), Yifan Peng `[通讯]` (Cornell University)

**通讯引用:** 10976 | [OpenAlex ID](https://openalex.org/A5085113833)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

针对大型语言模型在临床、科研和评测场景中因输入token数导致的成本与延迟问题，提出一种基于token预算的上下文选择框架，结合不同单元化策略和多种选择算法（Lead、MMR、RCD等），并设计了一个预算感知路由器来动态选择最合适的算法。

**💡 创新点**

创新点包括：① 将token预算纳入knapsack约束下的子模优化框架；② 提出结合相关性、覆盖度和多样性的 RCD 目标；③ 通过前置索引与冗余度指标实现的预算感知路由器，使得不同预算/任务场景下的最佳选择能够近似实现。

**🔧 技术方法**

使用的技术主要有：子模函数优化（贪心/懒惰贪心）、最大边际相关性（MMR）、对数行列式多样性（log‑det）、句子/段落/窗口/聚类单元化、语义嵌入与相似度核、ROUGE/BERTScore评估、基于token的成本计量。

**📊 数据集**

实验数据集包括：MIMIC‑IV 出院笔记与手工摘要、Cochrane 系统评价摘要、L‑Eval 长文总结任务以及 PubMedQA 长文本分类任务。

**📈 对比分析**

方法比较：在提取式评估中，低预算下 Lead 表现最佳；在 LLM 生成摘要中，MMR 在低至中等预算下优于 Lead，RCD 与 MMR 近似；在 PubMedQA 中，Lead 最差，MMR/RCD 接近全上下文基线。路由器在所有数据集上都能达到接近上限的性能，证明预算感知策略有效。

**⚠️ 局限性**

局限性：仅在固定生成器（GPT‑4o）上评估；缺乏端到端训练与生成器协同优化；评估仅使用自动指标（ROUGE/BERTScore），未验证事实正确性与临床安全；路由策略对数据分布漂移可能敏感；单元化策略对章节信息利用不足。

---

## 166. DynamicPO: Dynamic Preference Optimization for Recommendation

**arXiv ID:** 2605.00327 | [PDF](https://arxiv.org/pdf/2605.00327v1)

**作者:** Xingyu Hu `[一作]` (University of Science and Technology of China), Xiang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 26663 | [OpenAlex ID](https://openalex.org/A5100732436)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了LLM推荐系统中的偏好优化崩溃现象，并提出了DynamicPO框架来解决该问题。

**💡 创新点**

创新点在于引入动态边界负样本选择与双边距动态β调整两种自适应机制，显著防止了多负样本下的梯度抑制。

**🔧 技术方法**

技术上采用了Direct Preference Optimization（DPO）与其多负版本DMPO，并在此基础上实现实时聚类与样本级β动态调节。

**📊 数据集**

实验数据集包括LastFM、Goodreads和Steam三大公开推荐数据集。

**📈 对比分析**

与传统和LLM基础模型相比，DynamicPO在HitRatio@1上提升了约10%（例如Llama2-7b-hf上从58.5%提升至66.6%），且对多负优化策略具有通用性。

**⚠️ 局限性**

局限性包括对超参数γ、α的微调仍有一定影响，并且在极大负样本规模下仍需进一步验证可扩展性。

---

## 167. Prompt-Induced Score Variance in Zero-Shot Binary Vision-Language Safety Classification

**arXiv ID:** 2605.00326 | [PDF](https://arxiv.org/pdf/2605.00326v1)

**作者:** Charles Weng `[一作]` (Johns Hopkins University), Alexander Martin `[通讯]` (Johns Hopkins University)

**通讯引用:** 166517 | [OpenAlex ID](https://openalex.org/A5055290893)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究零射Vision‑Language模型在二分类安全判定中，单一提示下的首词概率不稳定，且提示间差异能导致预测概率显著变化；提出使用多提示均值聚合来提升概率可靠性。

**💡 创新点**

证明跨提示方差可作为可靠性诊断，且无训练的均值聚合在NLL和ECE上普遍优于单提示和标签后置校准（TS、Platt、Iso）。

**🔧 技术方法**

采用多提示均值聚合、训练自由的聚合变体（概率/对数几率平均、偏置校正等）以及三种后置校准方法（温度缩放、Platt、Isotonic）。

**📊 数据集**

在UnsafeBench（训练集用于提示选择）与HoliSafe‑Bench（外部测试）两个安全评测基准上，覆盖7类VLM（Qwen3‑VL‑4B/8B、Qwen3‑VL‑30B、InternVL3.5‑8B、Gemma‑3‑12B‑IT、MiniCPM‑V‑4.5、Llama‑3.2‑11B‑Vision‑Instruct）。

**📈 对比分析**

对比指标为NLL、ECE、AUROC、AUPRC。均值聚合在14个模型‑数据对上NLL全部提升，ECE在12/14提升；相较于锁定的单提示基线，它在AUPRC上保持稳定优势，AUROC优势略弱。与后置校准相比，均值聚合在无标签情况下更强，标签可进一步提升。

**⚠️ 局限性**

局限：未涉及学习聚合、贝叶斯不确定度或自适应弃权；AUROC提升不如AUPRC；跨提示方差仅作为诊断而非统一弃权信号；仅验证二分类安全场景，对多标签、结构化输出或开放式生成的推广有限。

---

## 168. PrefMoE: Robust Preference Modeling with Mixture-of-Experts Reward Learning

**arXiv ID:** 2605.00384 | [PDF](https://arxiv.org/pdf/2605.00384v1)

**作者:** Ziqin Yuan `[一作]` (Purdue University), Byung-Cheol Min `[通讯]` (Indiana University Bloomington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种名为 PrefMoE 的混合专家奖励模型，用以在含有异构和部分冲突的偏好监督数据中学习更鲁棒的奖励函数；同时提出了轨迹级软路由和负载平衡正则化机制。

**💡 创新点**

创新点在于：① 将混合专家（MoE）嵌入多模态 Transformer 的交互层，使不同专家能专门学习不同的隐含偏好评判准则；② 采用轨迹级软路由让每条轨迹动态权衡专家贡献；③ 通过负载平衡正则避免专家崩溃，提升多专家模型的可训练性和泛化能力。

**🔧 技术方法**

技术细节包括：多模态 Transformer（state 和 action 双流 + 交互编码器）、Mixture of Experts（4 个专家交叉注意力编码器）、轨迹级软路由（两层 MLP 生成专家权重）、Bradley–Terry 偏好损失、负载平衡正则化，以及与 RIME‑offline 等现有方法的对比实验。

**📊 数据集**

数据集：六个 D4RL 运动学基准（halfcheetah, walker2d, hopper 的 medium‑expert 和 medium‑replay 版本）、两种 AntMaze（large‑play, medium‑play）、五个 MetaWorld 机械操作任务；所有任务均使用人类众包标注（运动/AntMaze 100 人，MetaWorld 10 人）以及相应的 synthetic teacher 标注。

**📈 对比分析**

实验对比方法有：Markovian Reward (MR)、PrefMMT（单专家多模态 Transformer）和 RIME‑offline（噪声过滤的集成方法）。PrefMoE 在所有基准上平均得分最高；在 D4RL、AntMaze、MetaWorld 的具体任务中均实现或接近最优性能；在专家数量、噪声水平和注释池规模等 ablation 中显示出明显的鲁棒性提升，尤其在高噪声或大注释池场景下优势最为显著。

**⚠️ 局限性**

局限性：① 仅在离线 PbRL 场景验证，在线学习与实时反馈尚未测试；② 对极度同质化或单一评判准则的任务（如窗口关闭）无法充分发挥 MoE 优势；③ 需要额外的负载平衡正则，调参较复杂；④ 在极少标签或少注释者的环境下，仍可能退化为单专家表现；⑤ 未结合显式噪声过滤技术，仍需进一步提升对随机错误标签的抑制。

---

## 169. M-CaStLe: Uncovering Local Causal Structures in Multivariate Space-Time Gridded Data

**arXiv ID:** 2605.00398 | [PDF](https://arxiv.org/pdf/2605.00398v1)

**作者:** J. Jake Nichol `[一作]` (Sandia National Laboratories), Melanie E. Moses `[通讯]` (University of New Mexico)

**通讯引用:** 3679 | [OpenAlex ID](https://openalex.org/A5010797607)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种多变量网格级时空因果发现的元算法，生成多变量立方子图并分解为反应图和空间图。

**💡 创新点**

将单变量的LENS与PIP推广到多变量，联合估计空间与跨变量因果关系，保持可解释性，并引入反应图与空间图的分解。

**🔧 技术方法**

采用局部嵌入（LENS）+父节点识别（PIP）+空间/时间局部性与平稳性假设，并配合PC/PCMCI/DYNOTEARS等因果学习方法实现。

**📊 数据集**

使用合成VAR与ADR PDE仿真、E3SMv2-SPA的Pinatubo硫酸盐化学数据以及ERA5 SST/OLR观测数据进行评估。

**📈 对比分析**

在合成基准中F1>0.9，优于单变量或非空间基线；ADR实验中角度误差<5°、反应图F1≈1；Pinatubo案例F1≈0.95；ENSO案例恢复已知海气耦合，表现优异。

**⚠️ 局限性**

仅适用于局部性和平稳性假设；对高扩散或快过程的时空分辨率敏感；空间相关性降低有效样本；难以捕捉长程耦合。

---

## 170. Transient Multiscale Workflow for Thermal Analysis of 3DHI Chip Stack

**arXiv ID:** 2605.00399 | [PDF](https://arxiv.org/pdf/2605.00399v1)

**作者:** Mohammad Elahi `[一作]` (Rensselaer Polytechnic Institute), Jacob S. Merson `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 100 | [OpenAlex ID](https://openalex.org/A5038517524)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一套基于 GDSII/OASIS 文件自动构建 BEOL 结构的瞬态多尺度热分析工作流；

**💡 创新点**

创新点在于：①消除了传统方法中常见的 1‑D 热流假设，采用变分一致的瞬态同质化框架；②将微结构模型与宏观有限元耦合，并通过微尺度 RVE 的时域耦合捕捉热惯性；③提前预计算离散时间步长下的有效热导率与热容，可直接在商业 FEM 求解器中使用；

**🔧 技术方法**

使用的技术包括：变分一致多尺度同质化、Hill‑Mandel 原理、时间步长为 Δt 的后向欧拉积分、基于 LibreLane 的 SoC 布局生成、GDSII/OASIS 自动 RVE 提取与网格化、MATLAB/ANSYS 等有限元后处理；

**📊 数据集**

数据集为 1×1 mm² 的 Sky130A 开源 SoC 模型（由 LibreLane 生成），采用 5×5 RVE 与 10×10 RVE 两种尺寸，分别在 100×100 与 50×50 网格上共 10 000 与 2 500 个 RVE；

**📈 对比分析**

与完全解析的 50×50 RVE BEOL 模型对比，宏观热场的平均温度误差低于 0.5%，热点温度差距仅几摄氏度；同时，瞬态热导率映射在不同加载速率下展示了热惯性的显著影响；

**⚠️ 局限性**

局限性包括：①假设微结构材料性质温度不变且时间步长恒定，导致无法捕捉温度耦合效应；②忽略了宏观温度与温度梯度之间的额外向量耦合项；③对 RVE 大小和复杂度敏感，无法充分捕捉长条形电源线等细节；④需要手动选择 RVE 网格，可能导致计算量和内存需求升高。

---

## 171. Mesh Field Theory: Port-Hamiltonian Formulation of Mesh-Based Physics

**arXiv ID:** 2605.00394 | [PDF](https://arxiv.org/pdf/2605.00394v1)

**作者:** Satoshi Noguchi `[一作]` (JAMSTEC), Yoshinobu Kawahara `[通讯]` (University of Osaka)

**通讯引用:** 2601 | [OpenAlex ID](https://openalex.org/A5040846713)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Mesh Field Theory（MeshFT）并实现 MeshFT-Net：基于网格拓扑构造局部端口–汉密尔顿框架，仅学习度量与耗散项，保持拓扑耦合固定；

**💡 创新点**

创新点在于通过四条物理原则（局部性、置换不变性、方向协变性、能量守恒/耗散）证明局部端口–汉密尔顿分解，区分可学习的度量/耗散与不可学习的拓扑耦合；

**🔧 技术方法**

技术手段包括离散外部微积分（DEC）、局部雅可比分解、稀疏矩阵乘法、Strang 分裂数值积分、神经网络学习 SPD/PSD 的 G、R 以及对比 MGN、MGN‑HP、HNN、FNO、GraphCON 等架构；

**📊 数据集**

使用的数据集包括解析平面波、Delaunay 网格、来自 The Well 的声波散射数据，以及一个非线性浅水式 toy 动力学；

**📈 对比分析**

采用一步 MSE、TSMSE、能量漂移等物理一致性指标与 OOD 验证进行比较；MeshFT-Net 在能量漂移几乎为零、TSMSE 最低、数据效率最高，并在频率、分辨率、波速等 OOD 场景下保持优异表现，明显优于 MGN、HNN 等基线；

**⚠️ 局限性**

局限性包括：目前主要使用状态无关的耦合设定，对高度非线性或极端几何变形的适应性待验证；耗散强度过大或需要更灵活耦合时可能需要扩展模型；在大规模流体或多尺度问题中仍需进一步评估计算成本与鲁棒性。

---

## 172. Advancing Edge Classification through High-Dimensional Causal Modeling of Node-Edge Interplay

**arXiv ID:** 2605.00374 | [PDF](https://arxiv.org/pdf/2605.00374v1)

**作者:** Duanyu Feng `[一作]` (Sichuan University), Wenqiang Lei `[通讯]` (Sichuan University)

**通讯引用:** 2927 | [OpenAlex ID](https://openalex.org/A5039239180)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6215c339-3735-4be3-8a07-5bbb7004712d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Causal Edge Classification Framework（CECF），通过因果推理框架显式建模节点特征与边特征之间的相互作用，利用对抗式平衡反事实表示和交叉注意力网络实现边分类；

**💡 创新点**

创新点在于：①将边特征视为高维处理（treatment），首次在边分类任务中引入因果推理；②采用对抗学习消除节点特征对边特征的混杂效应；③在已有模型（如 GCN、GAT 等）上可直接插件，提升性能；

**🔧 技术方法**

技术方法包括：图神经网络生成节点嵌入；对抗式平衡反事实表示（使用 KL 散度和对抗网络 π 估计边特征）；交叉注意力网络处理节点与边特征交互；高维处理步骤（使用 2‑范数重构损失）；以及 CCA 与 Shapley 值分析；

**📊 数据集**

实验数据集涵盖六个基准：Bitcoin‑alpha、HSPPI、Epinions、Reddit、NID 和 MAG，覆盖二分类与多分类场景；

**📈 对比分析**

与 GCN、GAT、ChebNet、TopoEdge、TER+AER 等基线进行对比；CECF 在大多数数据集上提升 BACC 与 Macro‑F1 2–10%，并在与 TopoEdge 组合时进一步提升；训练时间相对基线仅略增，性能提升显著且具有统计显著性；

**⚠️ 局限性**

局限性：在节点-边相互作用弱的任务（如 MAG）中效果不佳；需先行评估 CCA 值；对高维处理的计算成本相对较高；若因果结构不符合假设，方法可能不适用。

---

## 173. Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration

**arXiv ID:** 2605.00370 | [PDF](https://arxiv.org/pdf/2605.00370v1)

**作者:** Chunlei Meng `[一作]` (Fudan University), Zhongxue Gan `[通讯]` (Fudan University)

**通讯引用:** 31306 | [OpenAlex ID](https://openalex.org/A5100455803)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Group Cognition Learning (GCL)，通过两阶段治理的协作机制实现多模态信息的有序交互与融合。

**💡 创新点**

创新点在于引入路由与审计代理在第一阶段进行边缘收益驱动的稀疏交互，并在第二阶段通过公共因子与加权聚合实现共享与专属特征的分离与自适应加权。

**🔧 技术方法**

使用了轻量化的路由/审计网络、信息增益预测、InfoNCE 冗余惩罚、公共因子提取以及基于注意力的聚合等技术。

**📊 数据集**

在多模态情感分析数据集 CMU‑MOSI、CMU‑MOSEI 以及多模态意图识别数据集 MIntRec 上进行实验。

**📈 对比分析**

与多种融合、对抗与 disentanglement 基线对比，GCL 在 MAE、ACC、F1 等指标上均刷新了公开记录，尤其在 MOSI/ MOSEI 的回归与分类任务上表现最优。

**⚠️ 局限性**

局限性包括对设计参数（如门控温度、冗余权重）敏感，且在极大模态规模或非结构化数据时的可扩展性和训练稳定性尚待进一步验证。

---

## 174. Uniform-Correct Policy Optimization: Breaking RLVR's Indifference to Diversity

**arXiv ID:** 2605.00365 | [PDF](https://arxiv.org/pdf/2605.00365v1)

**作者:** Anamika Lochab `[一作]` (Purdue University), Ruqi Zhang `[通讯]` (Purdue University)

**通讯引用:** 4555 | [OpenAlex ID](https://openalex.org/A5101586017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析RLVR在多解场景下的多样性崩塌机制并提出Uniform-Correct Policy Optimization (UCPO) 通过在GRPO基础上加入对正确答案集合的均匀性惩罚，重构梯度分配；

**💡 创新点**

创新点在于揭示RLVR目标对正确答案内部分布不敏感导致的自我强化崩塌，并证明在鲁棒性与熵正则化两种标准下，唯一最优策略为Uniform-Correct Policy；随后设计UCPO，使其通过条件KL正则实现该最优结构；

**🔧 技术方法**

使用技术包括RLVR框架、Group Relative Policy Optimization (GRPO)、条件均匀性KL惩罚、离线验证器、控制实验以及熵正则化与鲁棒性分析；

**📊 数据集**

实验数据集涵盖数学推理基准：AIME 2024/2025、AMC 2023、MATH 500、OlympiadBench 以及 DeepScaleR 训练集；

**📈 对比分析**

与GRPO及五种多样性保持基线（Ent-Reg、Ent-Adv、KL-Cov、Pass@K Training、FGRPO）对比，UCPO在Pass@64上提升最高可达+10%（AIME24），在保持Pass@1的同时显著提升多样性与覆盖率；

**⚠️ 局限性**

局限性包括需调优超参数τ，仍依赖外部可验证器；对非正确答案的概率分布控制有限，且在极端多解场景下均匀性约束可能过于严格。

---

## 175. Unlearning What Matters: Token-Level Attribution for Precise Language Model Unlearning

**arXiv ID:** 2605.00364 | [PDF](https://arxiv.org/pdf/2605.00364v1)

**作者:** Jiawei Wu `[一作]` (National University of Singapore), DouDou Zhou `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TokenUnlearn框架，通过掩码+熵计算令牌重要性，实现对大语言模型的精确令牌级遗忘；

**💡 创新点**

创新点在于：①首次将掩码式知识归因与熵不确定度结合得到关键令牌重要性；②提出硬选择与软加权两种令牌级梯度更新策略；③将该框架迁移至四种主流遗忘算法；

**🔧 技术方法**

使用的技术包括：掩码对比法计算概率偏移、预测熵、归一化组合得分、基于重要性加权梯度（硬/软），并结合KL正则保持实用性；

**📊 数据集**

实验数据集涵盖TOFU（合成知识遗忘）和WMDP（危险知识消除）两大基准；

**📈 对比分析**

与序列级基线GA、WGA、NPO、RMU比较，TokenUnlearn在TOFU和WMDP上均实现了更高的遗忘率（最高提升≈32.6%）且保持/提升了模型实用性（最高提升≈19%）；

**⚠️ 局限性**

局限性包括：额外掩码前向传递导致计算开销、对名词掩码的启发式假设可能遗漏非名词知识、无法完全保证知识彻底遗忘、超参需在不同数据/模型上重新调优。

---

## 176. Integrating Log-Based Security Analytics in Agile Workflows: A Real-World Experience Report

**arXiv ID:** 2605.00352 | [PDF](https://arxiv.org/pdf/2605.00352v1)

**作者:** Arpit Thool `[一作]` (Virginia Tech), Chris Brown `[通讯]` (Virginia Tech)

**通讯引用:** 4862 | [OpenAlex ID](https://openalex.org/A5014431408)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一次真实的安全事件后，构建并将基于日志的欺诈检测系统（Red Flag Project）嵌入到公司Kanban敏捷工作流中，并通过访谈评估团队对该系统的接受度、实施挑战及对日常开发与安全意识的影响。

**💡 创新点**

首次系统性记录日志驱动安全分析在敏捷开发中的实践与体验，提出了将安全工具与敏捷流程对齐、持续治理与自动化的实用方法，并揭示了团队接受度与长期可持续性之间的关键关系。

**🔧 技术方法**

使用Splunk做日志采集与查询、Python Flask实现Webhook中间件、Grouper进行身份/访问管理、Cron+Mutt邮件脚本发送告警；后期还讨论了Slack/Teams等通知渠道。

**📊 数据集**

组织生产环境中产生的身份验证与交易相关日志数据，包括登录失败、MFA 失败、Outlook 规则变更、工资转账信息变更等事件作为检测信号。

**📈 对比分析**

通过半结构化访谈和主题编码收集团队感知，未给出量化性能指标；报告指出系统在日常工作中影响极小，安全效益可感知，但存在误报率、日志格式漂移等限制。

**⚠️ 局限性**

局限性：研究仅基于单一组织单一Kanban团队的主观访谈，缺乏客观度量（如误报率、检测延迟等）；系统仍高度依赖人工维护与专人负责，治理与自动化不足，长期可持续性尚未验证。

---

## 177. MiniVLA-Nav v1: A Multi-Scene Simulation Dataset for Language-Conditioned Robot Navigation

**arXiv ID:** 2605.00397 | [PDF](https://arxiv.org/pdf/2605.00397v1)

**作者:** Ali Al-Bustami `[一作]` (University of Michigan-Dearborn), Jaerock Kwon `[通讯]` (University of Michigan-Dearborn)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了 MiniVLA-Nav v1，一个多场景、连续差分驱动机器人导航的数据集，用自然语言指令指导机器人逼近目标物体。

**💡 创新点**

集成多种室内场景、连续动作标签、RGB‑D+分割观测与语言指令，并提供分层生成、系统的 OOD 评估拆分，填补了现有数据集的空白。

**🔧 技术方法**

基于 NVIDIA Isaac Sim 5.1 物理仿真、Nova Carter 差分驱动平台、比例控制专家、分层 spawn 采样、语言模板填充及自动化生成管线。

**📊 数据集**

包含 12 类物体、4 个 Isaac Sim 场景（Office、Hospital、Full Warehouse、Warehouse‑MultiShelf），共 1,174 条成功演示，提供 640×640 RGB、深度、实例分割及连续动作。

**📈 对比分析**

提供五个评估拆分（train_id, val_id, test_id, test_paraphrase_ood, test_ood_obj）并设定 SR、NE、OP、碰撞率等指标；基线行为克隆、语言消融与 VLA 微调将在后续论文展示，当前基线 SR 远低于专家 100%，但 OOD 拆分可揭示模型泛化差异。

**⚠️ 局限性**

远程 spawn 采样不足（仅 1.2% far‑tier）、缺失颜色标签导致模板缺失、专家比例控制在狭窄通道表现欠佳、类别分布偏斜、无动态障碍、单摄像头视角以及从仿真到真实的域差距。

---

## 178. Pedagogical Promise and Peril of AI: A Text Mining Analysis of ChatGPT Research Discussions in Programming Education

**arXiv ID:** 2605.00361 | [PDF](https://arxiv.org/pdf/2605.00361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 179. RTPrune: Reading-Twice Inspired Token Pruning for Efficient DeepSeek-OCR Inference

**arXiv ID:** 2605.00392 | [PDF](https://arxiv.org/pdf/2605.00392v1)

**作者:** Ben Wan `[一作]` (JD), Tongxuan Liu `[通讯]` (JD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出RTPrune，一种无训练的两阶段视觉标记修剪方法，用于加速DeepSeek‑OCR推理

**💡 创新点**

创新点在于结合视觉编码的L2范数优先选取与基于最优传输的标记合并，并加入自适应修剪比例，模仿模型的“双读”行为

**🔧 技术方法**

技术包括视觉-语言联合训练、L2范数重要性评估、最优传输求解（Sinkhorn算法）与基于Sobel梯度的文本密度估计

**📊 数据集**

使用OmniDocBench、olmOCR‑Bench和Ocean‑OCR三大OCR基准数据集进行评估

**📈 对比分析**

与FastV、DivPrune、CDPruner等多种现有修剪方法比较，RTPrune在DeepSeek‑OCR‑Large上实现99.47%准确率，1.23×加速，显著优于其他方法

**⚠️ 局限性**

局限性在于对DeepSeek‑OCR的特定视觉编码-LLM协同训练依赖，可能难以直接迁移至其他视觉‑语言模型，且最优传输求解增加了额外计算开销

---

## 180. GaMMA: Towards Joint Global-Temporal Music Understanding in Large Multimodal Models

**arXiv ID:** 2605.00371 | [PDF](https://arxiv.org/pdf/2605.00371v1)

**作者:** Zuyao You `[一作]` (Fudan University), Zuxuan Wu `[通讯]` (Fudan University)

**通讯引用:** 8040 | [OpenAlex ID](https://openalex.org/A5026167547)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了一种名为 GaMMA 的音乐多模态语言模型，能够在同一参数集下同时处理时间序列和非时间序列的音乐理解任务，并提出了专门的双编码器融合网络（DFN）与三阶段训练策略。

**💡 创新点**

创新点包括：1) 双编码器架构分别针对时间动态与全局语义进行专门建模；2) DFN 提供了基于交叉注意力的专家融合与门控机制；3) 三阶段训练（对齐预训练 → 监督微调 → 强化学习）在音乐任务上显著提升性能；4) 引入 MusicBench（覆盖 3,739 个多选问题）作为专门的时间与非时间评测基准。

**🔧 技术方法**

技术上使用 Whisper 作为音频编码器，Qwen3 作为 LLM 核心，加入 MLP、交叉注意力、门控、残差等模块；训练时采用 DeepSpeed ZeRO-3、FlashAttention V2、BF16 混合精度；RL 采用 Group Relative Policy Optimization (GRPO)。

**📊 数据集**

主要数据集包括：大规模音乐-文字对齐数据（Audioset‑Cap、MSD‑Capsum）、音乐-歌词对齐数据、MuChoMusic、MusicBench（Global 与 Temporal 两个子集）、MIDI 合成音频、AudioSet‑Strong、以及多轮对话与指令数据。

**📈 对比分析**

通过与 Gemini‑3.0 Pro、Kimi‑Audio、Qwen2.5‑Omni 等现有 LAMM 进行对比，GaMMA‑8B 在 MusicBench‑Global 上取得 96.8% 准确率、在 MusicBench‑Temporal 上 92.0% 的最高分，MuChoMusic 也达到 78.0%（GaMMA‑8B）/79.0%（GaMMA‑14B）。RL 阶段进一步提升多选推理准确率，并在 MMAU 基准中保持竞争力。

**⚠️ 局限性**

局限性包括：1) 仍对极长音频（>30 s）进行片段分割，可能丢失跨片段全局信息；2) 对非音乐音频任务的泛化尚不充分；3) 需要大量算力（16×H100）和复杂训练管线；4) 在某些细粒度时间推理（如和弦进程）仍低于人类专家。

---

## 181. Time-series Meets Complex Motion Modeling: Robust and Computational-effective Motion Predictor for Multi-object Tracking

**arXiv ID:** 2605.00362 | [PDF](https://arxiv.org/pdf/2605.00362v1)

**作者:** Nhat-Tan Do `[一作]` (University of Information Technology), Trong-Hop Do `[通讯]` (University of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于改进的时间卷积网络（TCN）的多目标跟踪运动预测模型TCMP，专注于一次性前瞻预测，避免自回归误差堆叠。

**💡 创新点**

创新点在于将TCN与可学习的跳连（skipped）特征融合，采用α参数动态权衡最终块输出和跳连输出，并通过稀疏的扩张卷积实现高效、可变长度的时序建模；同时保持极低参数量与FLOPs。

**🔧 技术方法**

核心技术包括：改进的稀疏扩张卷积TCN、门控激活单元、跳连与最终块融合的回归头、L2损失、噪声与随机长度数据增强、以及与BYTE等传统关联算法的组合。

**📊 数据集**

在DanceTrack与SportsMOT两个专注非线性复杂运动的公开基准上进行评估。

**📈 对比分析**

与DiffMOT等SOTA方法相比，TCMP在DanceTrack上HOTA提升至63.4%、IDF1提升至65.0%、AssA提升至49.1%，在SportsMOT上HOTA提升至73.3%/76.3%、IDF1提升至74.2%/76.5%、AssA提升至62.6%/65.3%；同时参数量仅为SOTA的0.014倍，FLOPs仅为0.05倍，证明了高精度与高效能兼得。

**⚠️ 局限性**

局限性包括：对极长历史上下文（>16帧）信息不具优势；在完全跨域场景下性能略有下降；仅提供一次性前瞻预测，无法生成多步预测；对检测器误差的鲁棒性仍受限于外部检测模块。

---

## 182. Hypergraph and Latent ODE Learning for Multimodal Root Cause Localization in Microservices

**arXiv ID:** 2605.00351 | [PDF](https://arxiv.org/pdf/2605.00351v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 183. AgentFloor: How Far Up the tool use Ladder Can Small Open-Weight Models Go?

**arXiv ID:** 2605.00334 | [PDF](https://arxiv.org/pdf/2605.00334v1)

**作者:** Ranit Karmakar `[一作]` (Harvard University), Jayita Chatterjee `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在代理式 LLM 系统中，提出 AgentFloor 基准，对 30 个确定性工具使用任务进行分层评估，并与 16 款开源权重模型和 GPT‑5 进行对比。

**💡 创新点**

创新点包括：① 设计六层能力金字塔，将 instruction‑follow、单工具调用、链式调用、分支决策、多源合成与长周期规划拆分为可量化的层级；② 在抽象工具环境中消除 API 噪声，揭示规模之外的机制性差异；③ 用 TOST 与 Bootstrap CI 等统计方法验证模型间等价性。

**🔧 技术方法**

技术手段：使用原生工具调用、统一的系统提示、温度为 0 的推理、Bootstrap 置信区间、TOST 等效性检验、失败模式层级分类，以及针对特定任务的干预实验。

**📊 数据集**

数据集：AgentFloor 共 30 个任务，按 5 题/层 6 层划分，使用 8 种抽象工具和内存 fixture 数据库；每任务包含 5 种语言变体及 5 种实例变体。

**📈 对比分析**

比较方法：对 16 款开源模型与 GPT‑5 进行 16,542 次配对运行，计算任务成功率（TCR），并在每层使用 TOST 与 Holm‑Bonferroni 校正检验等价性。结果显示 gemma4:26b 在总体 TCR 与 GPT‑5 等价，开源模型在短周期工具调用任务上匹配甚至优于前沿模型；长周期规划任务（E 层）仍显著落后，且两者失败模式差异显著。

**⚠️ 局限性**

局限性：① 评估仅在抽象工具环境中，未覆盖真实 API、Web/GUI 交互；② 每层仅 5 题，覆盖范围有限；③ 仅使用 GPT‑5 作为前沿基准，未对多家前沿模型做交叉验证；④ 结果受特定任务设计与干预策略影响，缺乏普适性验证。

---

## 184. Block-wise Codeword Embedding for Reliable Multi-bit Text Watermarking

**arXiv ID:** 2605.00348 | [PDF](https://arxiv.org/pdf/2605.00348v1)

**作者:** Joeun Kim `[一作]` (DGIST), Young-Sik Kim `[通讯]` (DGIST)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了BREW框架，实现可靠多比特文本水印的检测与验证。

**💡 创新点**

通过块级嵌入、窗口平移验证和指定码字检测，显著降低误报率。

**🔧 技术方法**

使用BCH码、绿/红词表划分、软/硬偏置采样、窗口滑动解码等技术。

**📊 数据集**

在OPT-1.3B、LLaMA-3.2-3B、Mistral-7B模型生成的C4和OpenGen文本上进行评估。

**📈 对比分析**

与MPAC、Reed–Solomon等基线相比，BREW在10%同义词替换下TPR≈0.965、FPR≈0.02，远优于传统方法。

**⚠️ 局限性**

缺点是对极长文本或累计插/删导致偏移超过搜索窗口时鲁棒性下降，并且实现复杂度略高。

---

## 185. Language-free Experience at Expo 2025 Osaka

**arXiv ID:** 2605.00373 | [PDF](https://arxiv.org/pdf/2605.00373v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 186. Fast Text-to-Audio Generation with One-Step Sampling via Energy-Scoring and Auxiliary Contextual Representation Distillation

**arXiv ID:** 2605.00329 | [PDF](https://arxiv.org/pdf/2605.00329v1)

**作者:** Kuan-Po Huang `[一作]` (National Taiwan University), Chao Wang `[通讯]` (Amazon AGI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于能量距离目标和辅助上下文表示蒸馏的一步文本到音频生成框架，旨在解决自回归模型在生成过程中的高延迟问题。

**💡 创新点**

首次在文本到音频生成中应用能量距离目标，实现低延迟的一步潜在合成，并通过蒸馏技术提升生成质量。

**🔧 技术方法**

使用了能量距离训练目标和蒸馏技术，结合自回归模型进行音频生成。

**📊 数据集**

使用了AudioCaps和WavCaps两个广泛使用的文本到音频数据集，共计1700小时的音频数据进行训练。

**📈 对比分析**

与现有的一步基线模型（如ConsistencyTTA、SoundCTM、AudioLCM和AudioTurbo）相比，提出的方法在多个客观和主观指标上均表现优越，并且在生成速度上比现有的自回归扩散系统快8.5倍。

**⚠️ 局限性**

该方法仍然依赖于自回归步骤，可能在生成质量上与多步采样模型存在一定差距。

---

## 187. Flow matching for Sentinel-2 super-resolution: implementation, application, and implications

**arXiv ID:** 2605.00367 | [PDF](https://arxiv.org/pdf/2605.00367v1)

**作者:** Dakota Hester `[一作]` (Mississippi State University), Juliana A. Araújo `[通讯]` (Mississippi State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在全国尺度上使用流匹配模型对 Sentinel‑2 10 m 4 频段图像实现 4 倍超分辨率，并生成 2.5 m 4 频段 CONUS 与 Chesapeake Bay 的高分辨率产品；随后利用这些超分图像进行地物分类，构建年度土地覆盖产品。

**💡 创新点**

首次将流匹配技术应用于遥感超分辨率，实现单步高精度推理，并通过调节采样步数在推理时动态控制感知‑失真平衡；验证该方法在保持光谱精度的同时提升分类性能。

**🔧 技术方法**

使用流匹配（flow‑matching）框架，基于 U‑Net 结构的条件网络（低分 Sentinel‑2 作为条件），对比 DDPM、DDIM、Real‑ESRGAN 等方法；评估指标包括 PSNR、SSIM、LPIPS、光谱回归 R²、分类 F1、准确率。

**📊 数据集**

数据集：120,851 对 Sentinel‑2 Level‑2A 与 NAIP 2.5 m 4 频段跨传感器配对；年度 Sentinel‑2 合成（2020‑2025）；Chesapeake Bay CBLC 25,000 评估点；用于训练和验证地物分类模型。

**📈 对比分析**

在 T=1 的 Euler 步数下，流匹配实现 PSNR 34.95 dB、SSIM 0.8316，优于 Real‑ESRGAN（PSNR 33.29 dB）和 Lanczos；在 T≈20 的 Midpoint 步数下获得 LPIPS ≈0.247，推理速度约 0.56 s。地物分类方面，流匹配在 SegFormer 上取得最高 F1 78.23%，整体准确率 89.11%，相较于 Lanczos/Real‑ESRGAN 提升约 0.16% F1，且对 impervious 类提升明显。

**⚠️ 局限性**

流匹配的感知质量随步数增大而下降，需权衡；对稀疏类别（如 barren）仍表现不足；实验仅覆盖 Sentinel‑2/NAIP 配对，未验证跨传感器泛化；LPIPS 与人工感知不完全一致，需更适合遥感的感知度量。

---

## 188. AI Adoption Among Teachers: Insights on Concerns, Support, Confidence, and Attitudes

**arXiv ID:** 2605.00343 | [PDF](https://arxiv.org/pdf/2605.00343v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 189. Towards Robust and Scalable Density-based Clustering via Graph Propagation

**arXiv ID:** 2605.00390 | [PDF](https://arxiv.org/pdf/2605.00390v1)

**作者:** Yingtao Zheng `[一作]` (University of Auckland), Ninh Pham `[通讯]` (University of Southern Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 CluProp 框架，将高维空间的多密度聚类转化为邻域图上的标签传播，并引入 DANE 算法作为高效的确定性传播替代方案。

**💡 创新点**

创新点在于：① 将密度聚类与图连通性理论相结合，实现无全局阈值的自适应聚类；② 开发基于密度的邻域扩展（DANE），将 Potts 模型的求和交互替换为最大化交互，显著提升大规模数据的计算效率；③ 通过可调 kNN 图实现对任意距离度量的兼容。

**🔧 技术方法**

主要技术：近似 kNN 图构造（PyNNDescent 等）；加权对称 kNN 图；基于模块度的标签传播（Leiden、Louvain）；确定性 DANE 算法；理论分析基于密度的图连通性和 Potts 模型。

**📊 数据集**

在 MNIST（7万点）、MNIST8M（810万点）和 Pamap2（177万点）三大公开数据集上进行实验，使用多种距离度量（cosine、L2、L1、Jensen‑Shannon 等）。

**📈 对比分析**

与传统密度聚类（DBSCAN、HDBSCAN、OPTICS、DPC 等）以及需预设簇数的算法（k‑means、谱聚类、深度聚类 DCN）比较。CluProp+Leiden 在 20 秒内获得 90% AMI，CluProp+DANE 在 15 分钟内在 810 万点上实现 80% NMI；相比之下，DCN 需 30 分钟仅 75% AMI，kernel k‑means 仅 41% NMI；在 Pamap2 上同样显示更高 AMI 与更快运行时间。

**⚠️ 局限性**

局限性：① 仍需选择 kNN 参数 k，过小可能导致图碎片化；② 对 kNN 图质量依赖较大，粗糙近似可能略微影响精度；③ DANE 在极稀疏图中可能受限；④ 目前仅验证了 Euclidean/相似度空间的应用，尚未推广到非空间网络。

---

## 190. Geometric analysis of attractor boundaries and storage capacity limits in kernel Hopfield networks

**arXiv ID:** 2605.00366 | [PDF](https://arxiv.org/pdf/2605.00366v1)

**作者:** Akira Tamamori `[一作]` (Aichi Institute of Technology), Akira Tamamori `[通讯]` (Aichi Institute of Technology)

**通讯引用:** 564 | [OpenAlex ID](https://openalex.org/A5039826522)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了使用Kernel Logistic Regression（KLR）训练的Hopfield网络在高负载下的存储容量与吸引子几何特性，并提出了“优化脊”工作点。

**💡 创新点**

创新点在于：①在Ridge条件下实现随机序列P/N≈16、图像嵌入P/N≈20的高容量；②通过形态变换实验揭示吸引子边界的相位转移和临界减慢；③证明容量上限由动态噪声（SNR）决定，而非几何分离。

**🔧 技术方法**

使用了Kernel Hopfield网络、RBF核、Kernel Logistic Regression、序列记忆扩展、SNR分析、Cover定理参考以及有效维数D_eff计算。

**📊 数据集**

实验数据集包括随机二值模式和CIFAR‑10图像的ResNet‑18提取的512维嵌入。

**📈 对比分析**

与经典Hopfield网络（P/N≈0.14）比较，KLR网络在高负载下仍保持100%检索准确率（最高P/N≈19.5），并在噪声干扰下保持优异性能；Ridge与Local γ下的吸引子边界对比展示了更强的分离能力。

**⚠️ 局限性**

主要局限包括：计算复杂度高（O(NP)），仅同步更新；未探讨更复杂的关联结构和长程序列；SNR模型简化，缺乏精确的统计力学推导；低精度量化或稀疏核近似等改进方法待进一步验证。

---

## 191. Pose-Aware Diffusion for 3D Generation

**arXiv ID:** 2605.00345 | [PDF](https://arxiv.org/pdf/2605.00345v1)

**作者:** Zihan Zhou `[一作]` (Renmin University of China), Chongxuan Li `[通讯]` (Renmin University of China)

**通讯引用:** 1801 | [OpenAlex ID](https://openalex.org/A5072905534)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出Pose-Aware Diffusion (PAD)，在观察空间内实现单视图的姿势对齐3D生成。

**💡 创新点**

直接将去除canonical假设的部分点云作为3D潜在条件，使生成过程在观察空间内完成，从而消除姿势歧义并实现精确空间对齐。

**🔧 技术方法**

利用Vecset VAE压缩点云，基于流匹配的DiT变体实现3D扩散；使用预训练的深度估计器和点云编码器进行几何条件注入；并在训练中采用深度噪声增强与CFG。

**📊 数据集**

在Objaverse、3D-Front等构建的对齐数据集上训练，并在Google Scanned Object、3D-FUTURE等数据集上进行评估。

**📈 对比分析**

与DreamGaussian、InstantMesh、MIDI、SceneGen、SAM-3D、ShapeR等基线在CD、IoU-B、F-Score、Uni3D、ULIP等指标上对比，PAD在Chamfer距离最低、F-Score最高、图像到3D相似度最高，且在场景构建中也取得最优几何质量。

**⚠️ 局限性**

对输入深度/分割精度依赖较大，遮挡严重时生成质量下降，且在极端噪声点云下仍可能产生误差。

---

## 192. Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding

**arXiv ID:** 2605.00342 | [PDF](https://arxiv.org/pdf/2605.00342v1)

**作者:** Lehan Pan `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8179 | [OpenAlex ID](https://openalex.org/A5053344541)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练、无超参、无损耗的自适应验证方法EVICT，针对MoE稀疏模型的树形推测解码动态截断验证树，降低专家激活开销。

**💡 创新点**

创新点在于利用精细的draft概率估计接受长度，并结合离线测量的验证成本，计算每步最佳验证前缀，实现成本感知的动态截断，且兼容SGLang CUDA图加速。

**🔧 技术方法**

使用了树形推测解码、MoE路由与专家激活、概率估计的接受长度、成本感知的前缀选择、离线成本表查找、CUDA graph捕获与融合等技术。

**📊 数据集**

实验涵盖MT-Bench、Alpaca、GSM8K、HumanEval、QA、CNN/DM等六大基准数据集。

**📈 对比分析**

与自回归、Lookahead、DDD、EAGLE-3等基线对比，EVICT在Qwen3-30B-A3B及其他MoE模型上平均提升1.21×速度，最高可达2.35×，并在数学与代码任务上保持较高Mat，显著降低专家激活。

**⚠️ 局限性**

局限性包括需要预先离线测量验证成本且需在SGLang CUDA图环境下实现，对极低批量低延迟场景适配仍待验证，并且在难上下文或低采样温度时Mat可能略有下降。

---

## 193. Borrowed Geometry: Computational Reuse of Frozen Text-Pretrained Transformer Weights Across Modalities

**arXiv ID:** 2605.00333 | [PDF](https://arxiv.org/pdf/2605.00333v1)

**作者:** Abay Bektursun `[一作]` `[通讯]` (Independent research), Abay Bektursun (Independent research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在不修改任何权重的前提下，使用 Gemma 4 31B（仅用文本预训练）作为高维子结构，配合极薄可训练接口，迁移到多种非语言任务（机器人操作、离线 RL、离散序列任务）并实现显著性能提升。

**💡 创新点**

创新点在于证明：①预训练语言模型的“冻结”权重能够作为跨模态通用计算子结构；②通过双重测量（文本激活探测 + 任务消融）识别出被“外化”的注意力头；③展示了该子结构在不同任务中的可压缩性与可复用性。

**🔧 技术方法**

核心技术包括：Gemma 4 31B 的 60 层 Transformer 结构、冻结 L24–L29 滑动窗口块、线性编码/解码接口、FiLM/NTM 轻量化读取、决策 Transformer 的 body 替代、以及注意力头的定量消融与文本探测。

**📊 数据集**

使用的数据集包括：OGBench scene-play-singletask-task1-v0、OGBench cube-double-play-singletask-task1-v0、D4RL Walker2d-medium-v2、以及若干离散序列任务（copy、associative recall、CA Rule 90/110、binary addition、Dyck-2、GoL 等）。

**📈 对比分析**

与现有基线对比：在 OGBench scene-play-task1 上以 4.33 分点（±0.74）超过公开 SOTA GCIQL；在 Walker2d 上以 0.43× 的可训练参数实现与 Decision Transformer 相当的 76.2 ± 0.8 分；在关联回忆任务上实现 0.0505 的 per‑bit error，较从零开始训练的 6.36M transformer 低 8.7×；在 cube-task1 上通过冻结子结构与随机权重的对比显示 +59 分的“子结构贡献”。

**⚠️ 局限性**

限制主要包括：仅在 Gemma 4 31B 上验证，无法直接推广到更大或不同架构；迁移效果集中在离散序列/模式完成任务，无法解决连续控制、 2D 关系计算、Dyck 栈或无穷长度的 OOD 任务；消融实验显示对某些层/头的稳定性不足，需更多种子和更广泛的子结构探索。

---

## 194. Conformalized Quantum DeepONet Ensembles for Scalable Operator Learning with Distribution-Free Uncertainty

**arXiv ID:** 2605.00330 | [PDF](https://arxiv.org/pdf/2605.00330v1)

**作者:** Purav Matlia `[一作]` (Purdue University), Guang Lin `[通讯]` (Purdue University)

**通讯引用:** 6449 | [OpenAlex ID](https://openalex.org/A5078138445)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 Conformalized Quantum DeepONet Ensembles 框架，通过量子正交神经网络和叠加参数量子电路实现了子线性推理与多模型集成，同时使用自适应保形预测提供无分布覆盖保证。

**💡 创新点**

创新点在于：①把量子正交层用于 Operator 学习将推理复杂度从 O(n²) 降至 Õ(n)；②通过 Superposed Parameterized Quantum Circuits 将多模型集成压缩到单个量子电路，解决硬件资源线性扩展问题；③将集成的置信度与保形预测结合，获得严格的、分布无关的不确定性区间。

**🔧 技术方法**

核心技术包括 Quantum Orthogonal Neural Networks (QOrthoNN)、Superposed Parameterized Quantum Circuits (SPQC)、DeepONet 架构、基于自适应保形预测的置信区间生成，以及硬件级错误抑制与后选策略。

**📊 数据集**

实验数据集包含：①合成 PDE（反积分、阿德维克方程）以及 ②真实电力系统动力学（在线/离线电压‑电压、主动功率、瞬态预测）

**📈 对比分析**

相较传统密集层 DeepONet 和经典集成方法，实验表明：相对 L₂ 误差降至 0.46%‑12%（合成）或 4.08%‑12.49%（实际），覆盖率均在 88%‑93% 之间，且保持 90% 的目标覆盖率；同时在量子噪声下仍能保持校准。

**⚠️ 局限性**

主要限制：需满足样本可交换性以保证保形预测的覆盖性质；量子噪声与硬件漂移可能破坏此假设；目前仅在小规模（≤ 20 量子比特）模拟中验证，实际大规模量子设备上的性能与误差累积尚待进一步研究。

---

## 195. Binomial flows: Denoising and flow matching for discrete ordinal data

**arXiv ID:** 2605.00360 | [PDF](https://arxiv.org/pdf/2605.00360v1)

**作者:** Yair Shenfeld `[一作]` (Brown University), Stefano Peluchetti `[通讯]` (Sakana AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Binomial flows，一种针对非负序数离散数据的扩散模型，利用 Binomial 噪声实现可解析的去噪器学习与采样。

**💡 创新点**

创新点在于将 Binomial 噪声作为离散空间的 Gaussian 类比，得到离散 Tweedie 公式，将去噪器直接与 Poisson‑Föllmer 过程的速率相连，从而实现一次性训练、采样与精确似然估计。

**🔧 技术方法**

使用了 Binomial 噪声、Poisson‑Föllmer 连续时间马尔可夫链、离散 Tweedie 公式、Bregman 损失、EDM 预处理、Euler/τ‑leaping 采样等技术。

**📊 数据集**

在合成一维分布（Poisson、ZIP、NB 等）以及 CIFAR‑10 图像数据上进行验证。

**📈 对比分析**

与连续扩散模型（EDM、DDPM 等）以及其他离散扩散基线（Blackout、MDM 等）对比，Binomial flow 在 CIFAR‑10 上 FID 为 2.94，逼近连续模型性能；在合成数据上精确对数似然与真实值高度一致。

**⚠️ 局限性**

局限性包括采样步数较高、对 CFG 等指导参数敏感、对高维离散数据的推广性待验证，以及离散扰动与 Poisson 扰动的优势仍需深入研究。

---

## 196. Adaptation of AI-accelerated CFD Simulations to the IPU platform

**arXiv ID:** 2605.00462 | [PDF](https://arxiv.org/pdf/2605.00462v1)

**作者:** P. Rosciszewski `[一作]`, P. Gepner `[通讯]` (Graphcore Poland)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

将一个基于 LSTM 的 CFD 加速模型训练程序移植到 Graphcore IPU‑POD16 平台，并通过 Popdist 库消除主机侧 I/O 饥饿，实现对 OpenFOAM 同化罐仿真数据的训练和验证。

**💡 创新点**

1) 证明 IPU 在 AI‑for‑simulation 领域的可行性与易用性；2) 通过 Popdist 分布式训练显著提升主机到 IPU 的数据吞吐；3) 详细评估多 IPU 并行时的性能伸缩性，指出 1‑2 IPU 受通信与缓冲开销限制，而 8‑16 IPU 可实现约 5 倍吞吐提升。

**🔧 技术方法**

TensorFlow（Poplar SDK）+ Keras；IPU‑specific 配置 API；Popdist + Horovod；自定义数据加载与分片；FP16/FP8 数据类型；实验使用 OpenFOAM 产生的 131×420×125565×3 维度 CFD 数据集。

**📊 数据集**

131 个不同配置的同化罐 OpenFOAM 仿真，包含 420 个时间步、125 565 个网格单元、3 个速度分量，共 131×420×125 565×3 的数据；训练集占 80%（104 组），验证集 20%（27 组），交叉验证 20% 的训练集。

**📈 对比分析**

通过 RMSE、Pearson 与 Spearman 相关系数评估模型精度；RMSE ≤ 0.08（≈ 8% 误差）且相关系数 > 0.9；性能方面，单进程 16 IPU 通过 Popdist 提升约 34%，吞吐从 2099.8 samples/s 提升至 2805.8 samples/s；从 2 到 16 IPU 线性缩放，吞吐提升约 5 倍；但 1–2 IPU 由于通信与缓冲开销导致吞吐下降。

**⚠️ 局限性**

主机侧 I/O 线程受限导致数据预处理瓶颈；低 IPU 数量（1–2）受通信与缓冲开销影响，吞吐不升反降；未在更大规模 IPU‑POD64 上验证；仅使用 FP16，未探索 FP8 对训练性能的提升。

---

## 197. Soft Graph Diffusion Transformer for MIMO Detection

**arXiv ID:** 2605.00449 | [PDF](https://arxiv.org/pdf/2605.00449v1)

**作者:** Nan Jiang `[一作]` (Zhejiang University), Zhaoyang Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 23176 | [OpenAlex ID](https://openalex.org/A5100751311)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于流匹配的软图扩散变压器（SGDiT）用于 MIMO 检测，将检测过程建模为噪声水平条件化的逐步去噪流程，并通过自适应层归一化实现阶段感知的信息融合。

**💡 创新点**

创新点包括：① 将流匹配框架与 Soft Graph Transformer（SGT）结合，实现连续去噪轨迹；② 通过 AdaLN 对噪声水平进行动态调制，支持不同阶段的特征适配；③ 在训练目标上采用信号空间预测与二元交叉熵（BCE）损失，精确对齐离散符号的预测与损失空间。

**🔧 技术方法**

使用技术包括：流匹配与扩散变压器（Diffusion Transformer）、自适应层归一化（AdaLN）、图注意力机制（跨注意力与自注意力）、多层感知机（MLP）实现噪声水平嵌入、二元交叉熵损失、QPSK 的实值化处理、模拟 Rayleigh 与 3GPP UMa 信道。

**📊 数据集**

数据集与实验环境：在 QPSK 调制下，N_t = N_r = {8,16} 的 MIMO 系统；采用 Rayleigh 衰落信道与 3GPP UMa 统一多路径模型进行训练与测试，生成随机信道矩阵和 AWGN 噪声。

**📈 对比分析**

与基准方法（LMMSE、OAMP、OAMPNet2、SGT、以及 8×8 的 ML 检测）进行对比。SGDiT 在匹配 Rayleigh 信道下实现了最优或接近 ML 的 BER，优于所有模型；在 16×16 系统中仍保持显著优势；在信道分布漂移（Rayleigh → 3GPP UMa）测试中，SGDiT 的性能衰减最小，保持相对优势。计算复杂度随去噪步数 K 线性增长，K=2 时可与单步 SGT 相比；K=5 时提升了约 30% 的准确率。

**⚠️ 局限性**

局限性：① 需要额外的去噪步数来获得更高精度，导致计算量随 K 线性增长；② 对于更高阶调制或更大规模系统，连续去噪轨迹可能更难学习，性能提升有限；③ 在训练过程中需对 t 做截断以避免数值不稳定；④ 目前仅在模拟信道上验证，缺乏真实场景测试；⑤ 主要针对 QPSK，其他调制符号的适用性待验证。

---

## 198. LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations

**arXiv ID:** 2605.00434 | [PDF](https://arxiv.org/pdf/2605.00434v1)

**作者:** Huangbiao Xu `[一作]` (Fuzhou University), Yuxin Peng `[通讯]` (Peking University)

**通讯引用:** 9149 | [OpenAlex ID](https://openalex.org/A5047811387)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出LIMSSR框架，利用LLM在训练时缺失模态下完成多模态推理与融合，解决不完整多模态学习；

**💡 创新点**

将不完整多模态学习重新定义为条件序列推理任务，借助提示式语义推理、融合token以及Mask‑Aware双路径聚合，实现无完整数据监督的高效推理；

**🔧 技术方法**

使用大型语言模型（如Qwen3‑0.6B）+提示学习、占位符token与融合token、LLM推理、Mask‑Aware Dual‑Path Aggregation、Consistency & Token‑Level Regularization等技术；

**📊 数据集**

在FS1000、Fis‑V、Rhythmic Gymnastics三大动作质量评估数据集上进行实验；

**📈 对比分析**

与MCMoE、GCNet、ActionMAE等现有完整/不完整模态方法对比，LIMSSR在缺失模态下平均Spearman ρ提升约1–2%且MSE下降约8–12%，在完整模态下亦实现SOTA；

**⚠️ 局限性**

依赖LLM推理，易受hallucination影响；提示与token设计需手工调优，对极端缺失或非视觉模态的泛化能力尚待进一步验证。

---

## 199. RadLite: Multi-Task LoRA Fine-Tuning of Small Language Models for CPU-Deployable Radiology AI

**arXiv ID:** 2605.00421 | [PDF](https://arxiv.org/pdf/2605.00421v1)

**作者:** Pankaj Gupta `[一作]` (Postgraduate Institute of Medical Education and Research), Kartik Bose `[通讯]` (Postgraduate Institute of Medical Education and Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在9个放射学任务上对3–4B小型语言模型进行LoRA微调，并实现可在CPU上部署的多任务放射学AI助手。

**💡 创新点**

证明小型模型经过LoRA微调后即可在无GPU的CPU环境中完成RADS分类、NLI、抽取、生成等多任务，并通过任务路由集成提升整体性能；同时发现微调比少量示例提示更适用于专业领域。

**🔧 技术方法**

LoRA参数高效微调、GGUF 4-bit量化、任务加权采样、思路模式关闭、任务路由集合。

**📊 数据集**

12个公开放射学数据集共162K样本，涵盖RADS、impression、QA、temporal、NER、staging、abnormality等九个任务。

**📈 对比分析**

与零射门基线比较，在RADS准确率提升至77%（+53pp），NLI提升至≈83%，N-staging 89%，Temporal Jaccard 92%；Qwen2.5在生成任务领先，Qwen3在抽取任务优势；任务路由集成获得最佳。

**⚠️ 局限性**

样本量有限、GB‑RADS样本稀缺、主要使用英语、未验证跨语言、未评估更大模型、CPU基准只在单一硬件上、缺乏真实临床验证。

---

## 200. BOLT: Online Lightweight Adaptation for Preparation-Free Heterogeneous Cooperative Perception

**arXiv ID:** 2605.00405 | [PDF](https://arxiv.org/pdf/2605.00405v1)

**作者:** Kang Yang `[一作]` (Renmin University of China), Yongcai Wang `[通讯]` (Renmin University of China)

**通讯引用:** 5524 | [OpenAlex ID](https://openalex.org/A5053362548)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了准备-free异构协同感知，提出BOLT轻量化ego侧插件，在线无标签自适应邻居特征以提升合作检测。

**💡 创新点**

创新点在于不依赖任何预部署协同训练，仅通过ego作为教师的蒸馏实现在线适配，恢复并超越ego-only性能。

**🔧 技术方法**

使用AdaIN、残差CNN适配器和通道门控构建插件，并结合在线测试时训练（ego-as-teacher蒸馏）技术，同时保持其他网络参数冻结。

**📊 数据集**

在DAIR‑V2X（真实V2I）和OPV2V（仿真V2V）两个基准上进行评估，涉及多种LiDAR与摄像头编码器组合。

**📈 对比分析**

与ego-only、无插件融合以及HEAL/STAMP等预训练方法对比，BOLT在多对编码器组合上AP@50提升5–32点，且无需协同训练即可取得显著性能。

**⚠️ 局限性**

局限性包括需要共享BEV网格、仅支持单邻居在线适配、对极端异构或噪声高的邻居适配效果有限，以及对同步通信误差的鲁棒性尚待提升。

---

## 201. FollowTable: A Benchmark for Instruction-Following Table Retrieval

**arXiv ID:** 2605.00400 | [PDF](https://arxiv.org/pdf/2605.00400v1)

**作者:** Rihui Jin `[一作]` (Southeast University), Guilin Qi `[通讯]` (Southeast University)

**通讯引用:** 3317 | [OpenAlex ID](https://openalex.org/A5034606659)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Instruction‑Following Table Retrieval（IFTR）任务，并构建了第一份大规模基准数据集FollowTable，旨在评估检索模型在满足内容范围与表结构约束下的性能。

**💡 创新点**

创新点在于：①把表检索从单纯的语义匹配转为条件相关，②设计了新的IRS（Instruction Responsiveness Score）指标量化指令遵循程度，③提出了基于LLM的全自动指令生成与质量控制流程。

**🔧 技术方法**

使用的技术包括：LLM（如GPT‑5.2）进行指令生成与数据标注、dense bi‑encoder检索器、cross‑encoder重排器（Promptriever、StruBERT等）以及自定义的IRS评估方法。

**📊 数据集**

使用的数据集来源有：WQT、TableArXiv、WTR、IndusTR（工业标准手册）等四大子集，覆盖多领域、多结构的表格。

**📈 对比分析**

在基准上对比了多种检索器和重排器。结果显示：一般的dense retriever在nDCG上表现良好，但IRS仅约0.2~0.3；引入指令-aware retriever和重排器后IRS提升约15~20点，但推理延迟仍高1–2个数量级；整体而言现有模型在内容范围与结构约束上的遵循仍有限。

**⚠️ 局限性**

限制包括：模型对负面约束（排除、否定）敏感度低，缺乏对表结构（实体中心化、属性列等）精细匹配能力；重排器高准确但算力成本高；数据生成流程依赖LLM，可能引入标注偏差。

---

## 202. Batch Normalization for Neural Networks on Complex Domains

**arXiv ID:** 2605.00467 | [PDF](https://arxiv.org/pdf/2605.00467v1)

**作者:** Xuan Son Nguyen `[一作]` (CY Cergy Paris University), Nistor Grozavu `[通讯]` (CY Cergy Paris University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在复域（Siegel 圆盘与复单位球）上训练稳定且有效的批量归一化（BN）层，构建了可应用于现有Riemannian神经网络的通用BN框架；

**💡 创新点**

利用Kobayashi伪距离构造“近似测地线”更新运行均值，实现对复杂域中几何均值与归一化的闭式实现；

**🔧 技术方法**

核心技术包括：复域上的Fréchet均值估计、自动同构（ϕ_x）用于中心化与偏置、基于自动同构的标量乘法与近似测地线、以及Siegel盘与复单位球特定的距离与映射公式；

**📊 数据集**

在雷达杂波分类上使用基于AR模型的时间序列数据（6个规模不同的雷达数据集）；在节点分类上使用机场、Pubmed、Cora三大图数据集；

**📈 对比分析**

与基线方法（ComplexLSTM、kNN、Kernel-Siegel、SiegelNetFC、HNN-GyroBN-H、HNN-RBN-H、CBallNetBN）进行对比。实验表明：在雷达分类中，SiegelNetBN相较于SiegelNetFC提升约2–4%；在节点分类中，CBallNetBN对比GBN/RBN提升5–10%（平均准确率提升至79.6%/72.7%/60.9%）。

**⚠️ 局限性**

局限性包括：仍未在大规模真实应用中验证，缺乏自适应缩放操作的研究，近似测地线仅在特定域闭式可得，对更一般复域的推广仍需进一步探索；

---

## 203. PAMod: Modeling Cyclical Shifts via Phase-Amplitude Modulation for Non-stationary Time Series Forecasting

**arXiv ID:** 2605.00466 | [PDF](https://arxiv.org/pdf/2605.00466v1)

**作者:** Yingbo Zhou `[一作]` (Fudan University), Dejing Dou `[通讯]` (Fudan University)

**通讯引用:** 5021 | [OpenAlex ID](https://openalex.org/A5066063885)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PAMod框架，在标准化特征空间通过相位-振幅调制显式建模周期性分布漂移，以提升非平稳时间序列预测精度。

**💡 创新点**

创新点在于将周期性均值偏移视为相位调制、方差波动视为振幅调制，并证明其等价于可学习的动态去标准化，从而实现轻量化、可插拔的周期性非平稳性处理。

**🔧 技术方法**

采用可学习的周期嵌入（相位与振幅向量）、Reversible Instance Normalization、MLP预测头，并在多种Transformer/MLP基座上集成。

**📊 数据集**

在十二个公开基准上验证，包含ETT、Electricity、Traffic、Solar、Weather、PEMS等多维度时序数据。

**📈 对比分析**

与9种SOTA模型（TQNet、TimeEmb、FilterTS、Amplifier、TimeXer、CycleNet、TimeMixer、iTransformer、PatchTST）对比，PAMod在大多数数据集上均取得MSE/MAE最优或次优，并显著降低参数量（5–10倍）。

**⚠️ 局限性**

局限性包括需预先设定周期长度τ、对非周期性非平稳性建模不足，以及在极端非周期性波动或多尺度周期场景下效果可能受限。

---

## 204. CleanBase: Detecting Malicious Documents in RAG Knowledge Databases

**arXiv ID:** 2605.00460 | [PDF](https://arxiv.org/pdf/2605.00460v1)

**作者:** Weifei Jin `[一作]` (Duke University), Neil Gong `[通讯]` (Duke University)

**通讯引用:** 8064 | [OpenAlex ID](https://openalex.org/A5009102659)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建知识库相似度图并检测其中的恶意文档

**💡 创新点**

利用恶意文档在同一目标问题下高度语义相似，形成图中的团，从而实现无运行时开销的数据库清洗

**🔧 技术方法**

kNN相似度图构造、统计阈值剪枝、Clique检测、余弦相似度、Embedding模型(Contriever-msmarco)，以及理论误报/漏报上界分析

**📊 数据集**

NQ、HotpotQA、FiQA、ArguAna、SciFact、FEVER等六个公开问答数据集

**📈 对比分析**

与PromptGuard、DataSentinel及LLM/检索器防御方法对比；在9种注入攻击下，CleanBase的FPR<2%，FNR≈6%（部分攻击<34%），ASR从>90%降至≈9%；相比基线误报高/漏报高，性能显著优于其他方法

**⚠️ 局限性**

缺点：对极低数量或低相似度的攻击易逃逸；未提供可证明的安全保证；仅检测团结构，可能漏掉非完全团型恶意群组

---

## 205. The Power of Order: Fooling LLMs with Adversarial Table Permutations

**arXiv ID:** 2605.00445 | [PDF](https://arxiv.org/pdf/2605.00445v1)

**作者:** Xinshuai Dong `[一作]` (Carnegie Mellon University & NEC Labs), Zhengzhang Chen `[通讯]` (NEC Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

探究大型语言模型在表格问答任务中对行列排列的鲁棒性，并提出梯度攻击 ATP，寻找最差排列来欺骗模型。

**💡 创新点**

创新点在于将离散排列空间通过 Sinkhorn 软化为可微的双重随机矩阵，并结合信息熵正则化与 Hungarian 投影，将梯度优化直接应用于表格排列问题，从而揭示 LLM 在处理结构化数据时的根本弱点。

**🔧 技术方法**

使用的技术包括：梯度优化、交叉熵损失、双重随机矩阵 (Sinkhorn) 软化、信息熵正则化、Hungarian 投影（最大权匹配）以及混合软硬排列投影。

**📊 数据集**

实验数据集为 WikiTQ、TATQA、FeTaQA 三个公开的表格问答数据集。

**📈 对比分析**

通过与原始（未排列）表格和随机排列表格进行对比，使用 Gemini‑2.5 的 LLM‑as‑judge 评分；ATP 攻击将 Llama、Qwen 等模型的对齐得分从 0.4–0.5 降低到 0.2–0.3，显著低于随机排列带来的性能下降。

**⚠️ 局限性**

主要限制是攻击需要梯度信息，无法直接攻击闭源模型；仅针对行列排列的结构扰动，未考虑其他表格结构变形；攻击时间约 10 秒/样本（可调至 3 秒但效果略低）。

---

## 206. Thinking in Text and Images: Interleaved Vision--Language Reasoning Traces for Long-Horizon Robot Manipulation

**arXiv ID:** 2605.00438 | [PDF](https://arxiv.org/pdf/2605.00438v1)

**作者:** Jinkun Liu `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**通讯引用:** 7938 | [OpenAlex ID](https://openalex.org/A5012419026)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种全局交替的视觉-语言推理框架IVLR，在一次性生成包含文本子目标和视觉关键帧的完整推理轨迹，然后将其缓存为上下文，指导闭环动作解码；

**💡 创新点**

核心创新是显式的“语义-几何”全程轨迹表示，既保留因果序列，又提供空间约束；同时使用单一多模态Transformer实现轨迹生成与动作决策的统一；

**🔧 技术方法**

技术手段包括：基于Show‑o2的多模态Transformer、文本子目标的自回归语言头、流匹配视觉生成头、伪轨迹构造（UVD + Qwen3‑VL）、随机轨迹噪声与上下文遮蔽的鲁棒训练；

**📊 数据集**

主要使用的公开模拟数据集为LIBERO（包含Long、Spatial、Object、Goal四个子集）和SimplerEnv‑WidowX；

**📈 对比分析**

与RT‑1/RT‑2、OpenVLA、Octo、π₀‑FAST、VLA‑0、CoT‑VLA等基线相比，IVLR在LIBERO上平均成功率95.5%，Long子集92.4%，在SimplerEnv‑WidowX上整体成功率59.4%，均明显优于现有方法；

**⚠️ 局限性**

局限性包括：需要静态完全可观测的初始场景，生成完整轨迹需约10秒延迟；当场景在规划后改变或目标不可见时，缓存的全局轨迹可能失效；在真实机器人部署前缺乏物理实验验证。

---

## 207. Impact of Task Phrasing on Presumptions in Large Language Models

**arXiv ID:** 2605.00436 | [PDF](https://arxiv.org/pdf/2605.00436v1)

**作者:** Kenneth J. K. Ong `[一作]` `[通讯]` (ST Engineering), Kenneth J. K. Ong (ST Engineering)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在迭代囚徒困境案例中，比较不同任务表述方式（含否定、抽象化、零情境化）及是否启用推理步骤，探究大型语言模型在面对任务参数偏离其预设假设时的决策表现。

**💡 创新点**

创新点在于揭示任务措辞导致的先入为主假设对LLM决策的影响，并证明中性/抽象化表述可显著降低假设依赖、提升逻辑推理质量；同时发现推理步骤有时会强化模型的预设假设。

**🔧 技术方法**

使用的技术包括：GPT‑4o 与 Mistral‑Nemo‑Instruct‑2407 两大语言模型；对任务进行“推理”提示或不提示；构造标准与翻转奖励矩阵的迭代囚徒困境；记录首轮背叛率作为性能指标。

**📊 数据集**

实验所用数据仅为自定义的奖励矩阵与任务提示，并未使用公开数据集；每个实验在20次随机生成的情境下评估模型行为。

**📈 对比分析**

通过对比标准与翻转奖励矩阵下的背叛率，发现：在有明确游戏名称的实验中，模型几乎不改变行为；当任务措辞中立化后，GPT‑4o 的背叛率大幅下降且逻辑推理更连贯；Mistral‑Nemo‑Instruct‑2407 在中性提示下仍保持随机化策略，表现相对不稳定。

**⚠️ 局限性**

局限性包括：实验仅覆盖两款LLM；未包含最新推理模型（如DeepSeek、Reflexion）及其他提示工程技巧；所研究的囚徒困境任务不具备代表性，结果对更复杂、真实场景的泛化性未知。

---

## 208. Escaping Mode Collapse in LLM Generation via Geometric Regulation

**arXiv ID:** 2605.00435 | [PDF](https://arxiv.org/pdf/2605.00435v1)

**作者:** Xin Du `[一作]` (Waseda University), Kumiko Tanaka-Ishii `[通讯]` (Waseda University)

**通讯引用:** 2076 | [OpenAlex ID](https://openalex.org/A5073822077)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为 Reinforced Mode Regulation（RMR）的在线状态空间干预方法，用以抑制大型语言模型在长文本生成过程中的模式崩塌现象。

**💡 创新点**

创新点在于将模式崩塌视为内部轨迹的几何收缩（state‑space accessibility 下降），并通过在 Transformer 值缓存中识别并低秩衰减持久性方向（基于有界谱广义特征值问题），实现对内部动力学的直接调控。

**🔧 技术方法**

主要技术包括：① 轨迹级的相关维度（correlation dimension）作为几何诊断；② 在 KV 缓存上构建广义特征值问题求解持久子空间；③ 对该子空间施加低秩阻尼操作；④ 对特征值进行阈值截断与调节。

**📊 数据集**

实验以 Qwen3‑4B‑Base 等多款大型语言模型为测试对象，使用来自 SEP 数据集的 1,000 字段海德格尔文本段落作为固定提示；此外在附录中提供了更多提示和指令调优模型的结果。

**📈 对比分析**

与标准解码、典型采样（Typical Sampling）以及随机低秩调节进行对比。RMR 在温度锁定和熵锁定场景下显著提升“非崩塌率”，例如温度 0.7 时从 8% 提升至 56%；在极低熵（1.0）下标准解码几乎全崩塌（5%）而 RMR 可达 33%。在保持非崩塌的高温度生成中，RMR 对文本质量（连贯性、语法、信息递进）无显著影响。

**⚠️ 局限性**

局限性包括：仅适用于自回归 Transformer 结构，需在推理时访问 KV 缓存；在黑盒模型或非 Transformer 架构下不可直接应用；需维护运行统计并估计特征向量，带来一定延迟；阈值 λ_min 和阻尼强度 η 的选择仍依赖经验，可能需针对不同模型或解码模式调整。

---

## 209. Improving LLM Code Generation via Requirement-Aware Curriculum Reinforcement Learning

**arXiv ID:** 2605.00433 | [PDF](https://arxiv.org/pdf/2605.00433v1)

**作者:** Shouyu Yin `[一作]` (Tianjin University), Shikai Guo `[通讯]` (Dalian Maritime University)

**通讯引用:** 841 | [OpenAlex ID](https://openalex.org/A5018071549)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于需求感知的课程强化学习框架（REquirement-aware Curriculum Reinforcement Learning，简称RECRL），用于提升大语言模型在代码生成任务中的训练效果。

**💡 创新点**

创新点在于：①动态感知模型对需求的难度（RDS）；②通过需求优化和修订代理将高难度需求转化为易于学习的表述；③采用难度平滑采样策略，缓解课程过渡导致的灾难性遗忘。

**🔧 技术方法**

核心技术包括：大语言模型生成与自动化测试结合的难度感知；基于软件需求工程属性的需求优化与修订智能体；自适应课程采样与难度平滑机制；以及基于PPO的强化学习训练。

**📊 数据集**

实验使用了APPS+（训练集，包含三种难度标签）和五个公开基准（HumanEval、HumanEval+、MBPP、MBPP+、LiveCodeBench）进行评测。

**📈 对比分析**

与RSCRL、MDCRL、OMCRL、COTTON、CodePlan等基线相比，RECRL在所有五种LLM（Qwen2.5-Coder-1.5/3/7B、Llama-3.2-3B、SmolLM3-3B）上均取得平均Pass@1提升1.23%–5.62%、AvgPassRatio提升0.79%–7.06%；特别是对小模型提升显著，甚至超过更大模型。

**⚠️ 局限性**

局限性包括：难度感知需要执行测试，耗时较高；需求优化依赖有限的五个属性，可能不足以覆盖所有复杂需求；难度平滑因子固定，未采用动态调度；实验聚焦函数级代码生成，迁移至仓库级或其他工程任务需进一步验证。

---

## 210. Skills as Verifiable Artifacts: A Trust Schema and a Biconditional Correctness Criterion for Human-in-the-Loop Agent Runtimes

**arXiv ID:** 2605.00424 | [PDF](https://arxiv.org/pdf/2605.00424v1)

**作者:** Alfredo Metere `[一作]` `[通讯]` (Metere Consulting, LLC), Alfredo Metere (Metere Consulting, LLC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个安全可靠的 LLM 代理技能加载与执行框架，定义了技能信任架构、验证层级、HITL 门控策略、双向一致性判定准则，并给出了可移植的运行时准则（G1–G12）

**💡 创新点**

核心创新点包括：①将技能默认视为不可信并通过验证层级进行分级；②将 HITL 门控与验证层级关联，减少人工审批；③提出双向一致性准则来自动验证技能行为是否符合声明；④提炼出一组可迁移的安全准则（G1–G12）

**🔧 技术方法**

采用了多种技术：签名与可信根、Bell‑LaPadula 分类、可哈希链审计日志、可逆/不可逆能力分离、对抗性集成测试、正式验证、能力门控、HITL 触发与决策流程

**📊 数据集**

使用自定义合成文件集（如审计日志、财报、会议记录等）作为测试语料，并通过对抗性集成测试生成的多代理语料库进行验证；未使用公开数据集

**📈 对比分析**

通过对抗性集成评估验证双向一致性准则，实验中准则能完整捕获四类失败模式，表明门控与审计机制能在不依赖人工干预的情况下保持安全；性能上主要是逻辑有效性，未给出数值性能指标

**⚠️ 局限性**

局限性包括：仍无法防止只读数据泄露、时间检查/使用竞态、攻击者与合法批准重叠等攻击；需要更底层的数据流控制；硬件根等安全根仍是可选，未彻底消除所有后门与攻击面

---

## 211. P2M++: Enhanced Solver for Point-to-Mesh Distance Queries

**arXiv ID:** 2605.00429 | [PDF](https://arxiv.org/pdf/2605.00429v1)

**作者:** Qinghao Guo `[一作]` (Shandong University), Wenping Wang `[通讯]` (Texas A&M University)

**通讯引用:** 15780 | [OpenAlex ID](https://openalex.org/A5100668416)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 P2M++，一种改进的点到三角网格距离查询框架，显著降低预处理成本并提升查询速度。

**💡 创新点**

核心创新包括：自适应增设辅助 Voronoi 站点以缓解高密度/对称区域；将干扰检测转化为球-三角碰撞测试并通过 BVH 加速；用递归动态规划替代传统 kd‑tree 实现更快的最近邻搜索。

**🔧 技术方法**

采用 Voronoi 分区、Bounding Volume Hierarchy（BVH）、球-三角交叉、递归最近邻搜索、CGAL/Embree 等现代几何与加速技术。

**📊 数据集**

实验数据集涵盖 ABC 公开数据集、经典模型（Bunny、Dragon、Armadillo、Lucy、Camel、Lucy 等）以及旋转对称模型（Sphere、#1、#2、#3）。

**📈 对比分析**

与原 P2M 与 FCPW 进行对比：预处理时间提升 3–10 倍，查询时间加速 1.5 倍；在旋转对称几何上可达 50 倍速度提升。

**⚠️ 局限性**

局限性在于预处理仍高于轻量级方法 FCPW，难以在小批量查询中摊销；需要进一步优化几何判据；未支持网格间距离查询。

---

## 212. Trees to Flows and Back: Unifying Decision Trees and Diffusion Models

**arXiv ID:** 2605.00414 | [PDF](https://arxiv.org/pdf/2605.00414v1)

**作者:** Sai Niranjan Ramachandran `[一作]` (Technical University of Munich), Suvrit Sra `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文统一决策树与扩散模型，建立Tree‑Flow对应关系，并提出全局轨迹得分匹配（GTSM）框架，在此基础上设计了针对表格数据的生成和蒸馏两套算法。

**💡 创新点**

创新点包括①严格证明树结构可转化为概率流的PF‑ODE；②提出GTSM作为两类模型的共同优化目标，并证明梯度提升是其离散最优解；③首次将完整树决策逻辑通过得分匹配蒸馏到神经网络，并利用树的分层结构条件化连续流匹配。

**🔧 技术方法**

主要技术包括概率流理论、PF‑ODE、马尔可夫链连续化、Pawula定理、Girsanov变换、梯度提升、条件流匹配、离散化得分匹配以及树的路径编码。

**📊 数据集**

实验使用公开表格基准（Wine、Cancer、Adult、Heart Disease等）、MNIST、synthetic 2D、4‑Corners 等数据集。

**📈 对比分析**

在生成任务中与 TabDDPM、SDV 等基线对比，TSTR、Wasserstein、相关性误差和运行时间四个维度上取得 3/5、4/5、3/5 的最佳或最低指标；在蒸馏任务中与单层网络基线对比，匹配教师性能≤2% 并在 Heart Disease 上超额 3.7%。

**⚠️ 局限性**

局限性在于理论依赖连续路径细化与光滑性假设，难以处理内在不连续或跳跃过程；对非连续特征空间的适用性有限，需要进一步扩展到 Levy 过程或粗糙路径理论。

---

## 213. Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions

**arXiv ID:** 2605.00411 | [PDF](https://arxiv.org/pdf/2605.00411v1)

**作者:** Shuang Cui `[一作]` (Soochow University), Chen Xue `[通讯]` (Soochow University)

**通讯引用:** 31063 | [OpenAlex ID](https://openalex.org/A5108049724)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了首个满足预算可行性的子模福利最大化机制BFM-SWM，并提出其改进的价值最大化变体BFM-VM；两者都在采购拍卖中给出近似保证。

**💡 创新点**

创新点包括：使用几何递增阈值的降价时钟拍卖框架；单元素保护机制避免高价值卖家被过早排除；引入β参数调控支付与价值比例，确保非负买家盈余；BFM-VM通过阈值过滤实现确定性近似比提升至1/(12+4√3)并将运行时间降至O(n log n)。

**🔧 技术方法**

采用子模函数的边际收益贪心选择、阈值分段筛选、价格设定（β调节）、单元素保护、降价时钟拍卖以及理论分析证明近似比的技术。

**📊 数据集**

实验数据集为三个社交网络：Slashdot、Email 和 Epinions，用于影响力最大化场景。

**📈 对比分析**

将机制与基线机制（Deng-Distorted、Deng-ROI、Deng-CostScaled）进行对比，BFM-SWM在社会福利上提升 1.22×–26.41×，平均提升 4.49×；同时查询复杂度与运行时间均优于基线。

**⚠️ 局限性**

局限性：仅适用于单卖家单物品的采购拍卖；假设成本私有且可报价；理论近似比仍相对较低；实验验证仅覆盖影响力最大化场景，未在其他子模福利最大化任务中检验。

---

## 214. Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting

**arXiv ID:** 2605.00408 | [PDF](https://arxiv.org/pdf/2605.00408v1)

**作者:** Zhenhua Ning `[一作]` (Pengcheng Laboratory), Wenjie Pei `[通讯]` (Pengcheng Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 LeGS 框架，将 3D Gaussian Splatting 的密度控制转为可学习的强化学习策略，取代传统手工阈值方法，显著提升重建质量与效率。

**💡 创新点**

核心创新包括：① 用策略网络学习密度控制；② 基于敏感度分析设计闭式 O(N) 的奖励函数；③ 引入保持动作价值估计稳定训练；④ 在单一网络中同时完成增密与剪枝。

**🔧 技术方法**

技术手段涵盖：强化学习（PPO）、闭式敏感度计算、Spherical Harmonics 颜色编码、3D Gaussian 渲染、感知损失等。

**📊 数据集**

实验使用 Mip-NeRF 360、Tanks & Temples、Deep Blending 三大真实场景数据集。

**📈 对比分析**

与 Pixel-GS、Taming-3DGS、FastGS、Perceptual-GS 等 SOTA 进行 PSNR/SSIM/LPIPS 对比，LeGS 在保持或减少高斯数的前提下获得最高 PSNR/SSIM、最低 LPIPS，并保持与 FastGS 相近的训练速度。

**⚠️ 局限性**

局限性在于需额外的 RL 训练成本与超参数调优，奖励设计对局部误差敏感，且在极大规模场景中的扩展性尚待验证。

---

## 215. Scalable Learning in Structured Recurrent Spiking Neural Networks without Backpropagation

**arXiv ID:** 2605.00402 | [PDF](https://arxiv.org/pdf/2605.00402v1)

**作者:** Bo Tang `[一作]` (Worcester Polytechnic Institute), Weiwei Xie `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5085697317)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种结构化多层可递归脉冲神经网络（SNN），其局部层采用稠密递归连接，跨层通过稀疏小世界长程投射连接到读出层，使用局部可塑性、广播对齐反馈以及Winner‑Take‑All（WTA）教学信号实现无梯度学习。

**💡 创新点**

创新点在于将稀疏小世界结构、低带宽随机广播反馈、WTA输出教学与低维调制神经元三因素学习规则相结合，完成了不依赖梯度或代理梯度的深递归SNN训练，兼顾生物学可解释性与硬件可实现性。

**🔧 技术方法**

主要技术包括：离散时间LIF神经元、局部STDP与可选结构重排、三因素可塑性（预/后突触+调制因子）、eligibility trace、随机广播对齐（B矩阵）、WTA输出教学信号、读出层稀疏投射以及慢速阈值自适应。

**📊 数据集**

使用MNIST手写数字分类数据集，采用Poisson编码的spike序列，时程T=100步。

**📈 对比分析**

与传统无梯度SNN方法（STDP、reservoir、reward‑modulated STDP、WTA‑SNN）以及近似梯度方法e‑prop进行对比；在MNIST上实现约97.1%准确率，优于其他无梯度方法，且仅略低于e‑prop的性能，显示出在不使用梯度的情况下仍能达到竞争水平。

**⚠️ 局限性**

局限性包括：对复杂、高维任务的泛化尚未验证；学习率与超参数需人工调优；随机固定的长程投射可能限制网络表达能力；缺乏对大规模网络或其他数据集的实证；以及对实时硬件实现的实际功耗与时延影响需要进一步评估。

---

## 216. Near-optimal and Efficient First-Order Algorithm for Multi-Task Learning with Shared Linear Representation

**arXiv ID:** 2605.00473 | [PDF](https://arxiv.org/pdf/2605.00473v1)

**作者:** Shihong Ding `[一作]` (Peking University), Cong Fang `[通讯]` (Peking University)

**通讯引用:** 2055 | [OpenAlex ID](https://openalex.org/A5008843158)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出了一种两阶段梯度下降算法，联合学习多任务共享线性表示和任务特定参数，取得了近似最优的估计误差和常数阶迭代复杂度；

**💡 创新点**

创新点在于：①设计了先热启动后加正则化校正的两阶段梯度下降，克服了矩阵分解非凸障碍；②实现了仅需 O(1) 次迭代即可达到统计下界 O(dk/(TN)) 的误差；

**🔧 技术方法**

技术上使用了无约束梯度下降、正则化校正（相当于对目标加上低秩正则项）以及对 RIP 条件下的随机特征矩阵的理论分析；

**📊 数据集**

论文主要通过数值仿真验证，未给出真实数据集，仅在模拟的线性 MTL 环境中进行实验；

**📈 对比分析**

与现有 likelihood‑based 方法（如 ERM、ARMUL、pERM、SM 等）以及 MBM、SM 的基准进行比较，实验结果显示 TPGD 在迭代次数和样本量方面均取得了更快收敛和更低的估计误差；

**⚠️ 局限性**

局限性：①仅适用于线性共享表示模型；②需要 RIP 条件和同方差噪声假设；③在任务数量与维度相近或低维情形下理论证明不完全；

---

## 217. ReLay: Personalized LLM-Generated Plain-Language Summaries for Better Understanding, but at What Cost?

**arXiv ID:** 2605.00468 | [PDF](https://arxiv.org/pdf/2605.00468v1)

**作者:** Joey Chan `[一作]` (University of Illinois Urbana Champaign), Yue Guo `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并评估了一个包含50名无医疗背景读者的健康信息个性化简明摘要基准，并通过实验验证LLM个性化对理解和感知质量的提升。

**💡 创新点**

首次提供人本化的健康简明摘要基准，系统比较多种LLM个性化策略与安全性，并揭示效果与安全之间的权衡。

**🔧 技术方法**

使用多大语言模型（GPT-4o、GPT-5.2、Qwen3-4B-Instruct、MedGemma-27B、Mistral-7B-Instruct-v0.3）结合零样本、基于资料提示与检索增强提示的个性化方法。

**📊 数据集**

300条参与者-简明摘要对，包含术语熟悉度、信息需求、阅读测试、质量评分、交互日志和安全评估等，基于Cochrane综述抽取的12类健康主题。

**📈 对比分析**

通过可读性误差、风格相似度、知识覆盖度、幻觉与偏见评估等指标，发现基于用户资料提示的个性化在理解与质量上优于检索增强，且在安全性上低于非个性化模型。

**⚠️ 局限性**

个性化提升易产生幻觉与偏见，安全与效果间存在权衡，数据集规模有限且仅覆盖美国无医疗背景读者，未能验证跨文化或更高专业背景用户的适用性。

---

## 218. A Study on the Resource Utilization and User Behavior on Titan Supercomputer

**arXiv ID:** 2605.00426 | [PDF](https://arxiv.org/pdf/2605.00426v1)

**作者:** Sergio Iserte `[一作]` (Universitat Jaume I), Sergio Iserte `[通讯]` (Universitat Jaume I)

**通讯引用:** 409 | [OpenAlex ID](https://openalex.org/A5006767413)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对Titan超级计算机的资源利用与用户行为进行探索性分析和预测建模。

**💡 创新点**

将调度日志、GPU故障数据与季节性分解相结合，首次提出基于LSTM的预测模型。

**🔧 技术方法**

采用Pearson相关、相关系数、K-means聚类、STL分解及LSTM递归网络等技术。

**📊 数据集**

使用Titan 2015-2019年调度日志和GPU硬件故障日志两大数据集。

**📈 对比分析**

与传统统计方法相比，LSTM模型在CPU/GPU使用预测中实现0.4% MSE误差，表现优异。

**⚠️ 局限性**

数据缺失、两期使用模式分离及缺乏实时监控信息限制了模型的泛化能力。

---

## 219. AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning

**arXiv ID:** 2605.00425 | [PDF](https://arxiv.org/pdf/2605.00425v1)

**作者:** Haotian Zhao `[一作]` (Baidu), Jianmin Wu `[通讯]` (Baidu)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Adaptive Entropy Modulation（AEM），一种无监督信用分配框架，利用响应层熵自适应调节优势，实现 LLM 代理在多轮交互中的探索–利用平衡。

**💡 创新点**

首次从信息理论出发推导响应层熵漂移受优势与相对惊奇度共同决定的理论，进而设计无需额外监督的自适应熵调节机制。

**🔧 技术方法**

结合自然梯度、响应层熵估计、优势重标定与 group‑based 归一化等技术，形成轻量级的优势调节插件。

**📊 数据集**

在 ALFWorld、WebShop 和 SWE‑bench‑Verified 三大多轮 LLM 代理基准上进行实验。

**📈 对比分析**

与 GRPO、DAPO、GSPO 等现有基线相比，AEM 在 ALFWorld 上平均提升 8.8%，在 WebShop 上提升 5.6%，在 SWE‑bench‑Verified 上提升 1.4%。

**⚠️ 局限性**

采用响应层熵的近似代理，未能保证最优熵调节，且对采样组的多样性与质量敏感。

---

## 220. GD4: Graph-based Discrete Denoising Diffusion for MIMO Detection

**arXiv ID:** 2605.00423 | [PDF](https://arxiv.org/pdf/2605.00423v1)

**作者:** Qincheng Lu `[一作]` (McGill University), Xiao-Wen Chang `[通讯]` (McGill University)

**通讯引用:** 1833 | [OpenAlex ID](https://openalex.org/A5062087251)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于图的离散去噪扩散算法GD4，用于多输入多输出（MIMO）信号检测；

**💡 创新点**

创新点在于：①直接在离散符号空间进行去噪扩散，避免连续空间采样导致的高延迟；②利用图神经网络捕捉符号间的耦合关系；③提供冷热启动两种高效推理策略；

**🔧 技术方法**

使用技术包括：图神经网络（Gated Graph Message Passing）、离散去噪扩散模型、离散多项式（multinomial）噪声、Sinusoidal embeddings、离散逻辑斯蒂分布输出；

**📊 数据集**

数据集为随机生成的MIMO实例，信道矩阵从复高斯分布采样，符号为16-QAM（±1,±3），在多种N_t、N_r组合下进行实验；

**📈 对比分析**

与现有扩散方法ALD、ADD以及经典基线Babai、K-best Klein‑Babai进行比较，实验表明：在相同推理时间下，GD4单步或少量步推理即可超过10-best Klein‑Babai、ALD、ADD，且延迟最低；

**⚠️ 局限性**

局限性：需预先训练，训练集需覆盖目标SNR与规模；在极低SNR或更高阶QAM（如64-QAM）时性能尚未充分验证；模型对大规模MIMO（N_t、N_r >> 32）仍需进一步评估。

---

## 221. BWLA: Breaking the Barrier of W1AX Post-Training Quantization for LLMs

**arXiv ID:** 2605.00422 | [PDF](https://arxiv.org/pdf/2605.00422v1)

**作者:** Zhixiong Zhao `[一作]` (Houmo AI), Dawei Yang `[通讯]` (Houmo AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种后训练量化框架 BWLA，实现大语言模型的 1‑bit 权重量化与低位激活量化（如 6 位）并实现端到端加速。

**💡 创新点**

创新点在于引入正交‑克罗内克变换 (OKT) 将单峰权重分布转化为对称双峰，并同时抑制激活尾部；随后通过近似 SVD 投影 (PSP) 进行轻量级低秩校正，显著提升可量化性。

**🔧 技术方法**

主要技术包括 EM 估计的正交旋转、克罗内克分解实现高效正交矩阵、梯度无关的 GMM 目标函数、以及基于近似 SVD 的前向闭包正则化。

**📊 数据集**

使用的评估数据集包括 WikiText2、C4、ARC-Challenge/Easy、HellaSwag、LAMBADA、PIQA、WinoGrande、MMLU、GSM8K、HumanEval 以及 Qwen3-32B、LLaMA 系列等大模型。

**📈 对比分析**

与 BiLLM、ARB‑LLM、DBellQuant 等 SOTA PTQ 方法对比，BWLA 在 W1A6 方案下将 WikiText2 perplexity 降低 70% 以上、零样本任务平均准确率提升 50% 以上，并在 FP16 基线上实现 3.26× 的推理速度提升。

**⚠️ 局限性**

局限性包括在更极端的激活位宽（如 W1A4）下性能下降、正交变换对非线性权重空间的适配性有限，以及仅支持整数格式，未涵盖混合精度或低精度浮点格式。

---

## 222. Rethinking LLM Ensembling from the Perspective of Mixture Models

**arXiv ID:** 2605.00419 | [PDF](https://arxiv.org/pdf/2605.00419v1)

**作者:** Jiale Fu `[一作]` (Southeast University), Xu Yang `[通讯]` (Southeast University)

**通讯引用:** 470609 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将大语言模型（LLM）集成视为混合模型的方法，并实现了Mixture‑model‑like Ensemble（ME）算法，能在保持与传统集成相同性能的同时显著提升推理速度。

**💡 创新点**

创新点在于：1）从混合模型角度重新诠释LLM集成，证明只需在每一步随机挑选单一模型即可等价于完整集成；2）提出lazy KV缓存同步策略；3）将LLM集成与token‑level routing联系起来，揭示两者的本质关系。

**🔧 技术方法**

技术包括：随机模型选择（multinomial sampling），混合模型推理，KV缓存懒同步，词表对齐（UniTe），以及对不同规模和异构模型的混合使用。

**📊 数据集**

使用了四类数据集：数学推理（GSM8K）、多任务理解（MMLU）、复杂逻辑推理（BBH）和通用知识（ARC），并在多种模型组合（相似、异构、不同规模）上进行评估。

**📈 对比分析**

与单模型、传统序列/并行集成（CE）对比，ME在相同的准确率下推理速度提升约1.8×至2.7×，几乎达到单模型速度；实验覆盖多款GPU（H100、RTX3090、V100、A100），验证了方法的鲁棒性。

**⚠️ 局限性**

局限性：仅适用于基于采样的解码（非贪婪推理）；当模型性能差距较大时，ensemble效果对权重 λ 的敏感性会增强；需要额外的词表对齐步骤。

---

## 223. Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies

**arXiv ID:** 2605.00416 | [PDF](https://arxiv.org/pdf/2605.00416v1)

**作者:** Yi Wang `[一作]` (Shanghai Innovation Institute), Jianlan Luo `[通讯]` (Shanghai Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对一套预训练好的 Vision‑Language‑Action（VLA）通用机器人策略进行持续改进，使其在真实部署中通过在线强化学习不断提升性能。

**💡 创新点**

提出了“Learning While Deploying（LWD）”框架，将离线数据与在线部署经验融合；引入分布式隐式价值学习（DIVL）以处理稀疏奖励和多模态分布，结合 Q‑learning 与 Adjoint Matching（QAM）实现对流式生成动作策略的稳定提取。

**🔧 技术方法**

使用分布式强化学习架构；DIVL（分布式隐式价值学习）+ QAM；Gemma‑3 视觉语言基座与 SigLIP 视觉编码器；流式动作生成；双 arm Agibot G1 机器人平台；多任务离线与在线数据回放。

**📊 数据集**

离线数据来源于人类演示、历史策略跑动和人机对话探索；在线数据为 16 台机器人在 8 个真实任务（长周期任务如茶饮、果汁、鸡尾酒制作以及短周期的商品补货）中自主或人干预的轨迹。

**📈 对比分析**

与 SFT（仅模仿学习）、RECAP（离线迭代）和 HG‑DAgger（DAgger 风格）进行对比。LWD 在所有任务上平均成功率达 95%，在长周期任务上提升显著（从 68% 提升至 91%），并在循环时间上比基准快 23 秒。

**⚠️ 局限性**

局限性包括：在线更新调度过于简单，未针对大规模持续部署优化；单一高层语言指令限制了复杂任务的分解与恢复；未显式建模执行安全，未来需加入安全约束与风险评估。

---

## 224. ClozeMaster: Fuzzing Rust Compiler by Harnessing LLMs for Infilling Masked Real Programs

**arXiv ID:** 2605.00413 | [PDF](https://arxiv.org/pdf/2605.00413v1)

**作者:** Hongyan Gao `[一作]` (Nanjing University), Baowen Xu `[通讯]` (Nanjing University)

**通讯引用:** 11740 | [OpenAlex ID](https://openalex.org/A5100331400)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过基于历史bug触发代码的括号掩码与填充策略，使用LLM生成新的Rust编译器测试用例。

**💡 创新点**

创新点在于结合历史bug代码、括号结构掩码和LLM填充，提升测试覆盖率和bug发现率。

**🔧 技术方法**

使用的技术包括大语言模型（Incoder）、数据增强（随机删除、随机交换）、掩码填充、代码覆盖分析等。

**📊 数据集**

使用的数据集为从rustc问题追踪、官方测试套件、Glacier以及Rust官方文档等收集的约10万条代码片段。

**📈 对比分析**

与RustSmith、Rustlantis、SPE三种基准对比，测试覆盖率提升约95%，并发现更多新的ICE/Hang bug，性能明显优于现有工具。

**⚠️ 局限性**

局限性包括对LLM的依赖、对复杂语法的覆盖仍有限、未对更大规模或更高版本编译器验证，以及可能漏检非ICE/Hang bug类型。

---

## 225. Physically Native World Models: A Hamiltonian Perspective on Generative World Modeling

**arXiv ID:** 2605.00412 | [PDF](https://arxiv.org/pdf/2605.00412v1)

**作者:** Sen Cui `[一作]` (Tsinghua University), Jingheng Ma `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了以哈密顿动力学为核心的世界模型框架（Hamiltonian World Model），通过将观测编码成结构化的相空间潜在状态并利用能量函数驱动潜在状态的演化，最终生成未来观测并用于规划。

**💡 创新点**

创新点在于：①将物理定律（哈密顿结构）作为潜在动力学的先验，提供能量保守性和相空间结构；②结合控制、耗散和残差项，扩展至现实中非理想的机器人环境；③强调潜在状态的可解释性与可组合性（以实体为中心的相空间），并把视觉生成与物理演化分离。

**🔧 技术方法**

技术上使用了：相空间潜在编码器（将观测映射到 (q,p) 变量）；学习能量函数（由神经网络实现的 K、U、V 组成）；哈密顿方程或其受控耗散变形（J−R）与控制矩阵 G；图结构或实体关系表示；视频生成解码器（可采用变分自编码器或扩散模型）；基于模型预测控制的规划接口。

**📊 数据集**

论文未给出具体数据集，仅在讨论中提到可在机器人抓取、移动、自动驾驶等典型任务的现有视觉或运动数据集上进行验证，如机器人抓取视频、自动驾驶视频集等。

**📈 对比分析**

与传统 2D 视频生成、3D 场景重建和 JEPA 预测模型对比，作者指出哈密顿世界模型在长期滚动预测时更具稳定性、能量一致性和可解释性；在视觉重建质量与物理一致性之间取得更好的平衡，虽然未给出量化指标，但通过理论和实验示例证明了其优越性。

**⚠️ 局限性**

局限性包括：①真实环境中存在摩擦、冲击、接触切换等非保守效应，哈密顿模型需引入耗散与残差才能逼近；②从高维像素学习相空间变量仍难，可能导致表示不准确；③在接触丰富、柔性或非弹性对象的动力学建模仍面临挑战；④评估指标有限，难以量化其对决策的实质性改进。

---

## 226. SIMON: Saliency-aware Integrative Multi-view Object-centric Neural Decoding

**arXiv ID:** 2605.00401 | [PDF](https://arxiv.org/pdf/2605.00401v1)

**作者:** YuSheng Lin `[一作]` (National Yang Ming Chiao Tung University), Chun-Shu Wei `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 1054 | [OpenAlex ID](https://openalex.org/A5037797203)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SIMON框架，结合前景分割与显著性预测，采用显著性感知采样生成多视图，进行零样本EEG‑to‑image检索；

**💡 创新点**

通过解决几何‑语义解离，首次引入显著性感知采样和多视图融合，并使用双源显著性提取实现更精准的视觉表示；

**🔧 技术方法**

使用BiRefNet与SUM进行显著性提取，Saliency‑Aware Sampling (SAS) 采样，多视图foveated生成；视觉端采用CLIP编码，脑端使用EEGProject；双模态映射至Lorentz模型进行超曲面对齐，并用InfoNCE进行对齐学习；

**📊 数据集**

THINGS‑EEG 数据集；

**📈 对比分析**

与HyFI、NeuroBridge、NICE、ATM‑S等方法对比，SIMON在intra‑subject Top‑1 69.7% / Top‑5 92.9%，在inter‑subject Top‑1 19.6% / Top‑5 49.9%，均显著超过先前最优方法；

**⚠️ 局限性**

模型对每张图像的显著性检测、采样与foveated视图生成等多阶段处理导致约0.87秒的推理延迟，限制了其在实时系统中的直接应用。

---

## 227. Q-ARE: An Evaluation Dataset for Query Based API Recommendation

**arXiv ID:** 2605.00472 | [PDF](https://arxiv.org/pdf/2605.00472v1)

**作者:** Shenglong Wu `[一作]` (National University of Defense Technology), Tao Wang `[通讯]` (National University of Defense Technology)

**通讯引用:** 38264 | [OpenAlex ID](https://openalex.org/A5100653142)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Q-ARE数据集，用于评估查询式API推荐方法在多层调用结构下的性能，并引入了API调用深度和调用密度两个指标；

**💡 创新点**

创新点在于：①构建真实项目的多级调用评估数据集；②提出能够量化调用距离与语义稠密度的两项指标；③在此基准上系统评估传统检索、学习和LLM方法，揭示深层调用与稀疏调用对性能的严重影响；

**🔧 技术方法**

采用静态分析（Eclipse JDT）构建调用链，递归解析方法到第三方API的路径；使用RoBERTa、BERT重排序、RNN+Attention以及Gemini-3-Pro-Preview等模型进行评估；

**📊 数据集**

使用Q-ARE数据集，来自115个满足星级、测试、Spring框架等条件的GitHub Java项目，共5,698条查询样本和1,822个第三方API；

**📈 对比分析**

与CLEAR、DeepAPI、Gemini-3-Pro-Preview三种基线比较；在Q-ARE上Precision、Recall、AEM指标普遍低，最优AEM仅约11%；性能随API调用深度上升或调用密度下降显著下降；

**⚠️ 局限性**

局限性包括：数据集聚焦Java与Spring框架，缺乏跨语言/多框架多样性；静态调用分析可能漏判或误判；注释质量和LLM生成描述存在噪声；评估指标仅覆盖静态语义，未考虑运行时行为与动态调用。

---

## 228. Stereo Multistage Spatial Attention for Real-Time Mobile Manipulation Under Visual Scale Variation and Disturbances

**arXiv ID:** 2605.00471 | [PDF](https://arxiv.org/pdf/2605.00471v1)

**作者:** Xianbo Cai `[一作]` (Waseda University), Tetsuya Ogata `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 6933 | [OpenAlex ID](https://openalex.org/A5055922202)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于立体多阶段空间注意力与层级LSTM的端到端移动机器人操控方法，实现实时闭环控制。

**💡 创新点**

创新点包括：①立体多阶段空间注意力模块融合多尺度特征以稳定注意点；②层级LSTM将左右视角注意点与机器人状态联合建模；③采用无监督双向时间损失学习注意点。

**🔧 技术方法**

技术方案包括：立体摄像头视觉、双目注意力机制、多层CNN+1x1卷积、SoftArgmax、层级LSTM、无监督双向损失、CPU实时推理。

**📊 数据集**

数据集为通过遥控双臂移动机器人收集的15秒10Hz数据集，包含128×256 RGB双目、9-DoF手臂与4-DoF底盘运动，约54条演示，分布在四个任务。

**📈 对比分析**

与ACT、Diffusion Policy、π0、SmolVLA等基线在四个真实任务下对比，平均成功率达85.0%，远高于基线（最高仅约40%），在视觉干扰与距离变化下仍保持高稳健性，推理延迟约33 ms。

**⚠️ 局限性**

局限性在于仅在受限场景（平坦地面、预先标定双目、无动态遮挡）验证，缺乏更广泛的环境、动态障碍或更大尺度变化的评估，且对深度感知仍依赖双目精度。

---

## 229. Federated Learning with Hypergradient-based Online Update of Aggregation Weights

**arXiv ID:** 2605.00458 | [PDF](https://arxiv.org/pdf/2605.00458v1)

**作者:** Ayano Nakai-Kasai `[一作]` (Nagoya Institute of Technology), Tadashi Wadayama `[通讯]` (Nagoya Institute of Technology)

**通讯引用:** 1353 | [OpenAlex ID](https://openalex.org/A5049268989)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedHAW 方法，在联邦学习过程中利用超梯度在线更新聚合权重，以适应数据异构和通信错误。

**💡 创新点**

创新点是将超梯度下降应用于 FedLAW 风格的聚合权重，完全无需代理数据、无额外训练开销，并能与任何客户端更新方式无缝结合。

**🔧 技术方法**

采用超梯度下降、软最大化权重、FedLAW 框架、错误通信模型、FedProx 等技术实现聚合权重的动态更新。

**📊 数据集**

使用 MNIST、CIFAR‑10、Stanford Dogs（2 类）三大数据集进行实验。

**📈 对比分析**

与 FedAvg、FedAdp、FedHyper‑G、FedLWS、FedLAW 等方法对比，FedHAW 在数据异构和高通信错误场景下获得最高或接近最高的测试准确率，同时服务器端聚合时间仅为 FedAvg 的约四倍。

**⚠️ 局限性**

局限性包括对 meta 学习率的敏感性、在极端通信错误或非常低学习率下可能稳定性不足，以及未在大规模客户端或更复杂模型上验证可扩展性。

---

## 230. Urban to Rural Migration in Eastern Europe: Unpacking digital ruralities through TikTok video analysis

**arXiv ID:** 2605.00453 | [PDF](https://arxiv.org/pdf/2605.00453v1)

**作者:** Anca-Simona Horvath `[一作]` (Hong Kong University of Science and Technology), Huang `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对罗马尼亚城市向农村迁移现象进行研究，收集并分析了901条TikTok视频，结合定量主题模型与定性内容编码，探讨迁移动机、迁移者画像与日常体验。

**💡 创新点**

首次将数字乡土理论与TikTok视频分析相结合，系统性梳理城市向农村迁移的社交媒体呈现，揭示数字技术在农村生活中的新型劳动力和身份构建。

**🔧 技术方法**

使用主题模型（如LDA）、定性编码、校准过程、手工文本转录与情感分析等技术，对视频内容、创作者属性和评论进行多维度分析。

**📊 数据集**

包含901条罗马尼亚本土标签#delaoraslasat、#mutatlatara、#repopulamsateleromaniei的视频数据，通过Tikhub API抓取的JSON格式元数据（视频ID、描述、作者、地区、点赞、评论、播放量等）。

**📈 对比分析**

通过量化主题聚类与手工编码对比，发现八大主题占比、视频观看量达2430万，表明研究视角在数字乡土传播中具有较高的代表性；但未与传统迁移数据做直接对照。

**⚠️ 局限性**

样本仅覆盖三条罗马尼亚本土标签，缺乏跨平台与跨语言对比；缺少对迁移后实际生活质量的量化评估；平台算法与标签选择可能导致样本偏倚；未获取创作者收入信息，难以评估数字劳动力的经济效益。

---

## 231. Learning from Compressed CT: Feature Attention Style Transfer and Structured Factorized Projections for Resource-Efficient Medical Image Analysis

**arXiv ID:** 2605.00448 | [PDF](https://arxiv.org/pdf/2605.00448v1)

**作者:** Shadid Yousuf `[一作]` (Bangladesh University of Engineering and Technology), Mohammed Imamul Hassan Bhuiyan `[通讯]` (Bangladesh University of Engineering and Technology)

**通讯引用:** 2743 | [OpenAlex ID](https://openalex.org/A5112719871)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了 CT-Lite，一种在 JPEG 压缩 CT 体积上进行多模态异常检测的轻量级框架。

**💡 创新点**

创新点包括 Feature Attention Style Transfer (FAST) 通过注意力 Gram 矩阵和双注意力对齐实现跨压缩质量的知识蒸馏，以及 Structured Factorized Projection (SFP) 采用 Block Tensor Train 降低投影层参数近一半。

**🔧 技术方法**

技术手段包括 Vision Transformer (CT-ViT) 编码器、注意力风格保持损失、双注意力特征对齐、SFP 投影、SigLIP 对比损失和最大更新参数化 (μP) 学习率调度。

**📊 数据集**

使用了三大公开 CT 数据集：CT-RATE、NIDCH 和 RAD‑ChestCT，均包含压缩（JPEG 90%）的 CT 栈及对应的放射学报告。

**📈 对比分析**

与 CT-CLIP、COLIPRI 的压缩输入微调模型比较，CT‑Lite 在三组数据集的 AUROC 仅低于 5–7% 的无压缩基线，且在压缩输入上明显优于直接微调版本。

**⚠️ 局限性**

主要限制包括与无压缩基线仍存在约 3–5% 的 AUROC 差距、对比预训练规模有限、未在临床工作流中进行前瞻验证，并且压缩方式仅限 JPEG。

---

## 232. Adaptive Equilibrium: Dynamic Weighting Framework for Generalized Interruption of DeepFake Models

**arXiv ID:** 2605.00443 | [PDF](https://arxiv.org/pdf/2605.00443v1)

**作者:** Hongrui Zheng `[一作]` (Xinjiang University), Zhiqing Guo `[通讯]` (Xinjiang University)

**通讯引用:** 602 | [OpenAlex ID](https://openalex.org/A5044056760)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种自适应平衡框架（AEF），能够生成对多种深度伪造模型（StarGAN、AttGAN、AGGAN、HiSD）通用且均衡的中断扰动。

**💡 创新点**

创新点在于：① 通过指数移动平均（EMA）跟踪各模型的损失并利用温度控制的Softmax动态分配权重，聚焦于最难中断的模型；② 结合局部、全局与结构三种特征失配机制的深度特征增强（DFE），提升对不同生成机制的破坏力度。

**🔧 技术方法**

核心技术包括：自适应加权优化（动态损失权重）、EMA平滑、温度Softmax、三维特征失配（Local, Global, Structure）、多步梯度更新（Feature Enhancement + Adaptive Equilibrium）。

**📊 数据集**

使用的公开数据集有：CelebA（训练/测试）、LFW、FaceForensics++（FF++O）用于跨数据集泛化评估。

**📈 对比分析**

与 CMUA、FOUND、DWT、TSDF 等现有全局中断方法比较，AEF 在 SR_mask、L2_mask、FID、PSNR、SSIM 等指标上均显著提升，SR_mask 近 99.9%，训练时间仅 0.23h，显著节省计算成本。

**⚠️ 局限性**

限制主要包括：仍受限于白盒模型的集合，黑盒跨模型泛化仍不充分；需手动调参（α、T、β 等）；在极端高对抗训练场景下可能出现梯度冲突；对更大规模模型或不同任务的适用性尚待验证。

---

## 233. MMAudioReverbs: Video-Guided Acoustic Modeling for Dereverberation and Room Impulse Response Estimation

**arXiv ID:** 2605.00431 | [PDF](https://arxiv.org/pdf/2605.00431v1)

**作者:** Akira Takahashi `[一作]` (Sony Group Corporation), Yuki Mitsufuji `[通讯]` (Sony Group Corporation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用预训练的MMAudio视频‑到‑音频模型并微调，得到MMAudioReverbs，用于去回声和房间脉冲响应估计，且不修改网络结构。

**💡 创新点**

创新点在于将多模态V2A基础模型直接重用为物理房间声学任务的统一框架，通过fine‑tune实现两种任务而无需额外网络修改，展示视觉先验能增强声学估计。

**🔧 技术方法**

基于MMAudio的latent VAE+flow匹配框架，使用BigVGAN声码器，并在评估时禁用CFG。

**📊 数据集**

采用SoundSpaces‑Speech数据集，使用120°视角RGB图像及对应语音段，训练时使用2.56秒片段、20k步。

**📈 对比分析**

与WPE、VIDA、AV‑RIR等基线相比，在去回声指标如RT60、SRMR、DNSMOS上表现更好；在RIR估计上，预训练微调版在RT60、EDT、DRR等误差上明显优于基线，并在视觉+音频条件下提升早期能量预测。

**⚠️ 局限性**

局限在于缺乏显式几何、深度或材料属性，视觉信息仅为不完整先验；数据集未标注声源‑接收器关系，导致难以完全建模源‑接收关系；未来需引入轻量化物理属性适配器并构建更丰富的标注数据。

---

## 234. Foresight Arena: An On-Chain Benchmark for Evaluating AI Forecasting Agents

**arXiv ID:** 2605.00420 | [PDF](https://arxiv.org/pdf/2605.00420v1)

**作者:** Maksym Nechepurenko `[一作]`, Pavel Shuvalov `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一个去中心化、基于链的预测市场评估框架——Foresight Arena，用于测量 AI 预测模型的概率推断能力；

**💡 创新点**

创新点包括：① 通过 Polymarket 与 Gnosis CTF 实现的无信任、透明的结果判定；② 采用严格正当的 Brier 与 Alpha 评分规则，并给出 Alpha 的统计方差与功效分析；③ 在链上实现了 gasless、权限无关的提交与披露，保证可追溯性与防篡改；

**🔧 技术方法**

技术手段包括：Solidity 合约、EIP‑712 gasless 交易、commit‑reveal 方案、ERC‑8004 声誉与身份注册、The Graph 子图、Polymarket CLOB 与 Gnosis CTF 的 oracle 调用；

**📊 数据集**

使用了 Polymarket 的实时二元预测市场作为评估题目集合，并利用 Gnosis CTF 提供的最终结果；

**📈 对比分析**

比较方法为对比随机基线与市场共识基线，并将多款前沿 LLM（Claude、GPT、Gemini、Grok、文心）进行 Alpha 与 Brier 分数的统计检验；性能方面：随机基线显著差距；五款 LLM 在 Alpha 上表现相近，均在 +0.003~+0.005 之间，未能显著超越市场共识；

**⚠️ 局限性**

局限性包括：样本量有限（50 轮≈350 预测）导致只能检测到 |α|≥0.02 的差异；市场选择由 RoundManager 进行，可能引入偏倚；基准价格在低流动性时噪声较大；所有 LLM 采用统一工具配置，未能分离模型本身与工具的影响；

---

## 235. Multiset semantics in SPARQL, Relational Algebra and Datalog

**arXiv ID:** 2605.00417 | [PDF](https://arxiv.org/pdf/2605.00417v1)

**作者:** Renzo Angles `[一作]` (Universidad de Talca), Daniel Hernández `[通讯]` (University of Stuttgart)

**通讯引用:** 207 | [OpenAlex ID](https://openalex.org/A5073473252)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文对 SPARQL 的多重集语义（bag semantics）进行形式化分析，构造了与之等价的两种已知查询语言框架：非递归多重集 Datalog（含安全否定）和多重集关系代数（含投影、选择、自然连接、算术并、过滤差）。通过定义一系列数据库、查询及答案的翻译函数，证明这三种形式在多重集语义下具有相同的表达能力，并给出了相互之间的转换规则。

**💡 创新点**

创新点包括：① 引入了非递归多重集 Datalog with Safe Negation（NRMD¬），将否定和多重集计数统一到 Datalog 的证明理论框架；② 设计了完整的多重集关系代数（MRA）并与 SPARQL 及 NRMD¬ 进行语义对齐；③ 通过一套规范化和翻译技术，构建了 SPARQL → MRA、SPARQL → NRMD¬、MRA ↔ NRMD¬ 等多种模拟关系，实现三者表达力的互相可模拟；④ 兼顾了 SPARQL 1.1 中 EXCEPT、SELECT 等新操作符在多重集语义下的处理。

**🔧 技术方法**

主要使用的技术包括：证明论（derivation trees、colored set 计数）、语义归约（模式正规化、过滤条件拆解）、数据库/查询/答案的三元组翻译（g_12,g_21,g_32,g_23,g_31,g_13 等），以及多重集算子在关系代数中的定义与解释。整体采用形式化逻辑与代数证明，未进行实验实现。

**📊 数据集**

由于论文属于理论研究，未使用特定真实数据集；所有示例均基于人工构造的 RDF 图、关系集或 Datalog 程序，主要用于展示翻译和算子行为。

**📈 对比分析**

比较方法主要是通过构造三元组（翻译函数 f, g, h）证明语言之间的包含关系，并证明双向包含以得到表达能力等价；没有给出性能评估，只讨论了表达式等价性和语义保真性。

**⚠️ 局限性**

限制与不足包括：① 只讨论 SPARQL 的核心片段（AND, UNION, FILTER, EXCEPT, SELECT），不涉及 OPTIONAL、MINUS、聚合、子查询等；② 仅考虑非递归 Datalog，未探讨递归或递归带否定的情况；③ 对多重集算子在实际数据库实现中的复杂度与优化未做深入分析；④ 翻译过程中使用了特殊常量/IRIs，实际实现中可能导致额外的存储和查询开销；⑤ 论文重点在语义等价性，缺乏对运行时性能或查询优化策略的讨论。

---

## 236. From Local to Global to Mechanistic: An iERF-Centered Unified Framework for Interpreting Vision Models

**arXiv ID:** 2605.00474 | [PDF](https://arxiv.org/pdf/2605.00474v1)

**作者:** Yearim Kim `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**通讯引用:** 8464 | [OpenAlex ID](https://openalex.org/A5084897975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于实例特定有效感受野（iERF）的统一解释框架，涵盖局部解释（SRD）、全局概念解释（CAFE）以及机制解释（Interlayer Concept Graph与ICAT）

**💡 创新点**

将PFV-iERF作为单一分析单元，SRD实现高分辨率、鲁棒的类区分性注意图；CAFE利用iERF为稀疏自编码器特征提供像素级语义锚定，解决Transformer中非局部特征；ICAT构建跨层概念图，采用Integrated Gradients实现最可信的因果度量

**🔧 技术方法**

PFV-iERF单元、Sharing Ratio Decomposition、Attention‑LRP、Sparse Autoencoder、Concept‑Anchored Feature Explanation、Interlayer Concept Graph、ICAT、Integrated Gradients、插入/删除评估

**📊 数据集**

ImageNet‑1K（ImageNet‑S50、ImageNet‑val）、ResNet50、VGG16、ViT‑L/14（CLIP ViT‑L/14）、CIFAR‑100，用于训练SAE和验证

**📈 对比分析**

与梯度/类激活方法（Grad‑CAM、GradCAM++、Integrated Gradients、LRP 等）及全局概念方法（TCAV、ACE/CAT、SAE）进行对比；SRD 在定位、稀疏度、可信度与鲁棒性上优于基线；CAFE 在插入曲线AUC上优于基线；ICAT_IG 被验证为最具解释力的边权，能揭示误判和对抗攻击机制

**⚠️ 局限性**

依赖于 iERF 计算，计算成本较高；PFV‑iERF 假设单层 PFV 能直接映射像素，可能对多尺度或特殊架构不适用；全局概念解释仍需人工验证语义；机制图揭示路径但不保证因果性；对极端噪声或对抗攻击的鲁棒性仍有限

---

## 237. A Policy-Driven DRL Framework for System-Level Tradeoff Control in NR-U/Wi-Fi Coexistence

**arXiv ID:** 2605.00457 | [PDF](https://arxiv.org/pdf/2605.00457v1)

**作者:** Po-Heng Chou `[一作]` (Academia Sinica), Chiapin Wang `[通讯]` (National Taiwan Normal University)

**通讯引用:** 511 | [OpenAlex ID](https://openalex.org/A5056706536)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出基于策略驱动的深度强化学习框架，用以自适应调整 NR‑U 与 Wi‑Fi 在未授权频段中的 TXOP 以实现公平、吞吐量与 QoS 的协同控制。

**💡 创新点**

创新点在于将奖励函数设计为可调的“策略层”，通过不同奖励结构实现绝对公平、适度公平及基于效用的公平三种可选操作点，从而在同一框架下对系统级权衡进行明确、可控的调节。

**🔧 技术方法**

采用深度 Q‑网络（DQN）作为策略学习器，配合状态（吞吐量/效用比）和动作（增大/减小/保持 TXOP）设计，利用 MDP 进行在线决策。

**📊 数据集**

使用仿真生成的数据集，模拟 NR‑U 与 Wi‑Fi 的 LBT 交互，涵盖四个优先级类别（AC_BK、AC_BE、AC_VI、AC_VO）和不同网络密度场景。

**📈 对比分析**

通过与 3GPP NR‑U LBT 基线和基于多臂赌博机（MAB）的轻量级学习基线对比，结果显示绝对公平策略下公平指数 >0.9；适度公平提升总吞吐量 68.22%；效用‑基准策略提升效用 177.6%。

**⚠️ 局限性**

局限包括：在高度非平稳环境下奖励信号可能不稳定；当前框架假设集中式决策，难以直接扩展到大规模多小区或分布式多智能体场景。

---

## 238. Think Harder and Don't Overlook Your Options: Revisiting Issue-Commit Linking with LLM-Assisted Retrieval

**arXiv ID:** 2605.00447 | [PDF](https://arxiv.org/pdf/2605.00447v1)

**作者:** Cole Morgan `[一作]` (University of Windsor), Shaowei Wang `[通讯]` (University of Manitoba)

**通讯引用:** 2406 | [OpenAlex ID](https://openalex.org/A5100664833)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建真实时间窗口（问题创建后一年加上问题关闭前后30天）来评估和比较多种检索与重排序技术，从而实现问题与提交之间的链接恢复。

**💡 创新点**

创新点包括：①系统性地比较稀疏检索、密集检索、融合检索以及多种重排序模型，证明密集检索及其与稀疏检索的融合能显著提升召回；②发现传统机器学习模型RCLinker在所有数据集上可匹敌甚至超越大型语言模型；③提出可行且普适的时间窗口策略，避免仅依赖单一数据集的偏差。

**🔧 技术方法**

使用的技术包括：密集检索（SBERT‑Semantic Search、HNSW、ANNOY、LSH）、稀疏检索（BM25、BM25L）、递归排名融合（RRF）；重排序模型包括传统机器学习（BTLink、FRLink、RCLinker、Hybrid‑Linker）、跨编码器（ms‑marco‑MiniLM‑L6‑v2）以及大语言模型（GPT‑5.1、Qwen3‑32B、Gemma‑7B、Llama‑3.1‑8B）。

**📊 数据集**

所使用的数据集包括：BTLink 基准集、EasyLink 基准集以及作者自行构建的 Apache 10 项目集（共 10 余 Apache 项目，超过 200K 个真链接）。

**📈 对比分析**

通过 Recall@K、Precision@K、Hit@K、MRR、NDCG 等指标进行比较。实验结果表明：密集检索（SBERT‑SS）在召回率和精确度上优于稀疏检索，RRF 融合进一步提升召回；在重排序阶段，RCLinker 的 Precision@1 在 EasyLink、BTLink、Apache 三个数据集分别达 84.2%、93.6% 和 94.9%，遥遥领先于其他模型；跨编码器与 GPT‑5.1、Qwen3‑32B 的性能相近，且比 Llama‑3.1‑8B 等小模型更稳健；LLM 的推理成本高（GPT‑5.1 每条问题约 0.004 USD）。

**⚠️ 局限性**

局限性：①仅在公开开源项目上验证，缺乏工业级项目的评估；②数据集构建仍基于关键词匹配，可能存在误链接；③正负样本严重不平衡，影响模型泛化；④LLM 的推理成本和硬件依赖限制了实时应用；⑤未探索不同 prompt 设计和更大/更小模型的性能差异。

---

## 239. Scaling Video Understanding via Compact Latent Multi-Agent Collaboration

**arXiv ID:** 2605.00444 | [PDF](https://arxiv.org/pdf/2605.00444v1)

**作者:** Kerui Chen `[一作]` (Microsoft Research Asia), Hehe Fan `[通讯]` (Zhejiang University)

**通讯引用:** 2336 | [OpenAlex ID](https://openalex.org/A5002207978)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个端到端多代理协作框架 MACF，用分段本地代理在预算约束下进行视频感知，并通过潜在通信令牌实现全局推理。

**💡 创新点**

创新点在于将感知预算与全局视频复杂度解耦，使用代理本地的潜在通信令牌而非文本，并配合三阶段课程学习稳定训练。

**🔧 技术方法**

使用多模态大语言模型（如 Qwen3‑VL‑8B）作为本地代理和协调器，基于共享嵌入空间的适配器进行潜在通信，采用三阶段对齐/摘要/协作训练。

**📊 数据集**

在 LLaVA‑Video‑178K（caption）、Video‑R1（image QA）和 Molmo2（视频 QA）等数据集上训练，并在 Video‑MME、LongVideoBench、LVBench、MLVU‑Test 等长视频理解基准上评估。

**📈 对比分析**

与 13+ 个基准 MLLM 和两种多代理基线相比，在相同输入预算下 MACF 平均提升 4.5%–7.7%（Video‑MME 至 MLVU），并在文本通信对比中提升约 10%–20%。

**⚠️ 局限性**

局限性包括对特定预算设置的敏感性、对高质量视觉‑文本对齐的依赖，以及在极大视频长度下仍需增加代理数导致通信开销。

---

## 240. On the Role of Artificial Intelligence in Human-Machine Symbiosis

**arXiv ID:** 2605.00440 | [PDF](https://arxiv.org/pdf/2605.00440v1)

**作者:** Ching-Chun Chang `[一作]` (National Institute of Informatics), Isao Echizen `[通讯]` (National Institute of Informatics)

**通讯引用:** 5839 | [OpenAlex ID](https://openalex.org/A5044556342)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于角色水印的自适应生成方法，用以在生成文本中隐式嵌入并随后解码人工智能（AI）的参与角色，从而在失去原始提示后仍能判定AI是协助编辑还是创造内容。

**💡 创新点**

创新点在于：①将角色识别、角色编码与角色解码三步系统化为完整的工作流；②采用词汇子集偏置与显著性检验相结合的统计方法，实现对AI角色的可解释性追踪；③在角色归属任务上突破传统二元检测，提供三分类（人类、协助、创造）框架。

**🔧 技术方法**

核心技术包括：元提示生成进行角色分类；对生成过程注入词汇子集偏置（bias δ）形成角色水印；基于二项分布的p值显著性检验进行角色解码；以及多模型多数据集实验验证。

**📊 数据集**

使用四个公开数据集：IMDb、CNN/DailyMail、Wikipedia 和 arXiv；每个数据集随机抽取 1000 例，概念长度 <100 词，内容长度 100–500 词；模型分别为 GPT‑2（124M）和 LLaMA‑3‑Instruct（3B）。

**📈 对比分析**

与 Entropy、LogLikelihood、LogRank、Curvature 以及 RoBERTa 等基线相比，该方法在四个数据集上平均 AUC 达到 0.91（GPT‑2）或 0.99（LLaMA‑3），准确率分别为 0.88 与 0.95；在三分类任务中亦保持最高或相当于最佳基线的表现；对同义词替换鲁棒性良好，误差 ≤0.14；文本质量（自评 perplexity）虽略高于无偏生成，但差异不大。

**⚠️ 局限性**

局限性包括：①仅设计并评估了两种角色（协助与创造），未覆盖更细粒度或层级化角色；②方法依赖可合作的生成主体，无法处理非合作或已有文本的后期标记；③实验仅在文本生成上验证，未考察多轮对话或多模态场景；④偏置参数需手动调优，可能对不同模型产生差异。

---

## 241. Optimal Spatio-Temporal Decoupling for Bayesian Conformal Prediction

**arXiv ID:** 2605.00432 | [PDF](https://arxiv.org/pdf/2605.00432v1)

**作者:** Yu-Hsueh Fang `[一作]` (National Taiwan University), Chia-Yen Lee `[通讯]` (National Taiwan University)

**通讯引用:** 5455 | [OpenAlex ID](https://openalex.org/A5021866378)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了State-Adaptive Bayesian Conformal Prediction (SA-BCP)，在在线时间序列中通过空间密度门控实现时空解耦。

**💡 创新点**

创新点是用单一阈值K实现空间与时间混合，构建最优的 bias‑variance 贸易，并证明其在非平稳环境下的边际有效性。

**🔧 技术方法**

采用核密度估计、指数折扣、混合累积分布与根求解、以及Welford在线统计更新等技术。

**📊 数据集**

在2016–2026年的金融资产数据集（AMD、Gold、GBP/USD）以及合成波动序列上进行实验。

**📈 对比分析**

与基线方法AgACI、DtACI、BCP对比，SA‑BCP在高置信度下实现更低Winkler得分、覆盖率更稳健且区间更窄（高达37%收缩）。

**⚠️ 局限性**

局限在于阈值K需手工设定，且在高度变化的微观结构中可能需要进一步的自适应调节。

---

## 242. Agent Capsules: Quality-Gated Granularity Control for Multi-Agent LLM Pipelines

**arXiv ID:** 2605.00410 | [PDF](https://arxiv.org/pdf/2605.00410v1)

**作者:** Aninda Ray `[一作]` `[通讯]`, Aninda Ray

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Agent Capsules 框架，自动决定多代理管线是采用细粒度（每个代理单独调用）还是合并调用（compound），并在运行时通过质量门控和升级梯队保证输出质量。

**💡 创新点**

创新点在于：①基于运行时行为的 composition score 用于判断何时安全合并调用；②质量门控（shadow‑judging）防止合并导致质量下降；③阶梯式升级（standard→two‑phase→sequential）在质量失败时自动切换，避免直接退回细粒度。

**🔧 技术方法**

采用的技术包括：LLM 代理编排、prompt 编译与缓存、滚动平均质量评估（LLM judge）、工具调用监控、动态上下文注入、输出引导与结构提示、Redis 持久化状态。

**📊 数据集**

使用的评估数据集主要为四种多代理管线：due_diligence（5‑agent）、code_review（6‑agent）、long_chain_research（8‑agent）和 multi_source_brief（14‑agent），模型覆盖 Sonnet、Haiku、GPT‑4o、GPT‑4o‑mini、Gemini‑2.5‑flash‑lite 等。

**📈 对比分析**

比较方法为与手工调优的 LangGraph（14‑agent）和编译式 DSPy（5‑agent）进行头对头基准测试，测量输入/输出 token、整体 token、质量（LLM judge）和延迟。Agent Capsules 在 token 上比 LangGraph 减少 51%（fine）/42%（compound）且质量提升，和 DSPy uncompiled 相比减少 19% token，MIPROv2 减少 68% token，且保持或提升质量；在 sequential compound 下实现 63‑64% 输出 token 节省且质量保持不变。

**⚠️ 局限性**

局限性包括：实验仅覆盖 5‑14 代理的 4 种管线，未验证更大规模或不同领域；缺乏 GPU 级别利用率数据；质量评估受 LLM judge 噪声限制；未测试带可见链路思考的模型；组内代理数上限 4；自动输出引导阈值在特定管线下调校；对显式推理模型和工具使用行为的泛化仍待验证。

---

## 243. Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration

**arXiv ID:** 2605.00578 | [PDF](https://arxiv.org/pdf/2605.00578v1)

**作者:** Luru Jing `[一作]` (Peking University), Yongzhi Cao `[通讯]` (Peking University)

**通讯引用:** 1797 | [OpenAlex ID](https://openalex.org/A5040695351)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 FedHD 联邦分布式数据蒸馏框架，用于全切片图像（WSI）的多实例学习，支持异构模型与隐私保护。

**💡 创新点**

创新点包括：Gaussian‑mixture 特色对齐与一对一蒸馏、课程式融合策略、伪图像解码解释模块以及低维特征共享以避免过压缩。

**🔧 技术方法**

技术手段涵盖：GMM 对特征分布对齐、特征蒸馏、Generalized Cross‑Entropy 损失、FastGAN 伪图像重建、t‑SNE 可视化与 LiRA 会员攻击评估。

**📊 数据集**

实验数据集为 TCGA‑IDH、CAMELYON16 与 CAMELYON17。

**📈 对比分析**

与 FedHE、DESA、FedDGM、HistoFS、FedWSIDD 等个性化 FL 方法对比，FedHD 在各数据集平均准确率/MCC 上提升约 2–3%，并在隐私攻击评估中 AUC 更低，表现更优。

**⚠️ 局限性**

局限性在于伪图像重建的视觉质量受轻量化 GAN 限制，细节表现不足；在极端压缩配置下仍可能丢失诊断细节。

---

## 244. Instance-Aware Parameter Configuration in Bilevel Late Acceptance Hill Climbing for the Electric Capacitated Vehicle Routing Problem

**arXiv ID:** 2605.00572 | [PDF](https://arxiv.org/pdf/2605.00572v1)

**作者:** Yinghao Qin `[一作]` (Queen Mary University of London), Jun Chen `[通讯]` (Queen Mary University of London)

**通讯引用:** 470609 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了针对电动容量车辆路径问题（E-CVRP）的b-LAHC算法的实例感知参数配置框架，通过离线调优获得实例特定参数标签，并利用回归模型基于实例特征预测参数，从而避免了单一全局参数设置的局限。

**💡 创新点**

创新点在于：①首次将实例感知参数配置应用于E-CVRP的b-LAHC；②设计两阶段irace调优并构建kNN图及需求统计特征集；③使用Ridge回归模型实现参数预测，并在8个测试实例上平均提升0.28%且接近oracle性能。

**🔧 技术方法**

主要技术包括：Ridge回归模型用于参数预测；kNN图结构与需求统计特征提取；irace自动化调参；Late Acceptance Hill Climbing元启发式；Wilcoxon符号秩检验评估统计显著性。

**📊 数据集**

使用的数据集为IEEE WCCI 2020 EVRP基准及其扩展，共41个实例，其中8个保留为测试集，剩余33个用于训练与交叉验证。

**📈 对比分析**

实验通过与全局参数配置及实例特定（oracle）调优进行对比，结果显示预测参数在8个测试实例上平均降低0.28%的目标值，7/8实例表现提升；与oracle的差距极小，Wilcoxon检验表明差异显著。

**⚠️ 局限性**

局限性包括：预测误差仍显著，尤其是η_max；仅针对两个参数的预测；依赖离线调优所需的计算成本；在未知特征或动态环境下的适应性有限。

---

## 245. The impact of coercive, normative, and mimetic Stress on Chinese teachers' continuance intention to use generative AI: An integrated perspective of the Expectation-Confirmation Model and Institutional Theory

**arXiv ID:** 2605.00522 | [PDF](https://arxiv.org/pdf/2605.00522v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 246. A11y-Compressor: A Framework for Enhancing the Efficiency of GUI Agent Observations through Visual Context Reconstruction and Redundancy Reduction

**arXiv ID:** 2605.00551 | [PDF](https://arxiv.org/pdf/2605.00551v1)

**作者:** Michito Takeshita `[一作]` (Hosei University), Hitoshi Iyatomi `[通讯]` (Hosei University)

**通讯引用:** 4446 | [OpenAlex ID](https://openalex.org/A5023899090)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 A11y-Compressor 框架，将线性化的无障碍树压缩为结构化、紧凑的 GUI 观察表示 Compressed-a11y，提升本地多模态大语言模型在 GUI 任务中的定位与执行效率。

**💡 创新点**

创新点在于三阶段压缩流程：模态检测以恢复前后层关系；冗余减少消除多余属性与重复元素并将边框坐标转为中心坐标；语义结构化将 UI 元素划分为功能区域，兼顾结构与语义；整体实现了 token 仅 22% 的压缩率且显著提升成功率。

**🔧 技术方法**

使用规则化的文本处理管道，基于无障碍树属性、关键词匹配、时间差检测以及空间布局启发式进行压缩；并在本地 Qwen3‑VL‑32B 作为 LLM 进行评估。

**📊 数据集**

在 OSWorld 基准测试集上进行实验，涵盖 358 个任务，涉及 Chrome、LibreOffice、Thunderbird、GIMP、VLC、VS Code 等九大应用域。

**📈 对比分析**

与三种基线（截图、线性化无障碍树、LineRetriever）比较，Compressed‑a11y 在 token 效率上将平均 token 降至 22%（约 5.1% 的成功率提升），在多应用域中均取得最高或同等成功率；截图方式表现最差。

**⚠️ 局限性**

局限性包括：仅基于无障碍树，无法利用纯视觉特征；规则化实现导致阈值和启发式对新界面泛化不佳；评估仅覆盖桌面应用，缺乏移动或跨平台验证。

---

## 247. From Research to Practice: An Interactive Rapid Review of Autonomous Driving System Testing in Industry

**arXiv ID:** 2605.00531 | [PDF](https://arxiv.org/pdf/2605.00531v1)

**作者:** Qunying Song `[一作]` (University College London), Federica Sarro `[通讯]` (University College London)

**通讯引用:** 4665 | [OpenAlex ID](https://openalex.org/A5012165852)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过与一家汽车公司的21位从业者合作，进行互动式快速评审，识别ADS测试的12项实际挑战，聚焦两大核心问题，筛选17篇论文提取技术规则，并评估其在工业环境中的适用性。

**💡 创新点**

首次将工业挑战驱动的快速评审与技术规则评估相结合，强调上下文感知与可转移性，为ADS测试研究与实践搭建桥梁。

**🔧 技术方法**

采用快速评审流程、技术规则抽取、实操工作坊、专家投票、情景生成方法（约束/强化学习/生成模型/对抗扰动）等技术。

**📊 数据集**

使用来自真实车辆运行的场景数据（Real-world driving data）、仿真场景以及公开的测试案例库（如CARLA/UE4等）来评估和生成场景。

**📈 对比分析**

通过与从业者的现场评估对比技术规则的相关性和可行性，未给出定量性能指标，但强调了实用性、现实性和工具链集成度的差异。

**⚠️ 局限性**

主要限制包括场景维度高导致的可扩展性难题、对高维度搜索/生成方法的计算成本、工具链和仿真灵活性不足，以及缺乏统一评估框架。

---

## 248. IdentiFace: Multi-Modal Iterative Diffusion Framework for Identifiable Suspect Face Generation in Crime Investigations

**arXiv ID:** 2605.00526 | [PDF](https://arxiv.org/pdf/2605.00526v1)

**作者:** Weichen Liu `[一作]` (Southeast University), Alex Kot `[通讯]` (Shenzhen MSU-BIT University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多模态迭代扩散框架 IdentiFace，用于在刑事调查中生成可被人脸识别系统检索到的嫌疑人面孔

**💡 创新点**

创新点在于①引入低质量图像、简易素描和细粒度文本三种实用模态作为条件，显著降低条件模糊；②设计了基于最佳检索结果的交互式迭代生成管线，抑制采样方差并提供可视化参考；③加入面部身份损失提升身份一致性；④构建了两套专用数据集 ID‑CelebA 与 ID‑FFHQ

**🔧 技术方法**

利用控制网络（ControlNet）与扩散模型（Stable Diffusion v1.5、PixArt‑α）结合多模态融合；使用面部识别模型 AdaFace、SFace 进行身份匹配；通过文本分块、局部文本比例 LTR 等技术加强细节学习

**📊 数据集**

基于 CelebAMask‑HQ 与 FFHQ 生成的 ID‑CelebA（18,197张）和 ID‑FFHQ（25,839张），每张图配有 LQ 图、素描、细粒度文本

**📈 对比分析**

与 T2I、ControlNet 单模态、Sketch‑to‑Face 等方法对比，使用 AdaFace、SFace 评价匹配率与相似度，FID、LPIPS、MS‑SSIM 等质量指标；在 ID‑CelebA 与 ID‑FFHQ 上，ID‑PA 与 ID‑SD 最高匹配率分别为 84% 与 85%，并在真实场景实验中成功率高达 80% 以上

**⚠️ 局限性**

局限包括：对细粒度属性的精细控制可能导致不自然结果；训练数据偏差可能影响不同人种的识别性能；系统对隐私与滥用的风险需进一步评估

---

## 249. PhysiGen: Integrating Collision-Aware Physical Constraints for High-Fidelity Human-Human Interaction Generation

**arXiv ID:** 2605.00517 | [PDF](https://arxiv.org/pdf/2605.00517v1)

**作者:** Nan Lei `[一作]` (Sun Yat-sen University), Wei-Shi Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 22264 | [OpenAlex ID](https://openalex.org/A5108050904)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为 PhysiGen 的插拔式优化策略，能够在文本驱动的多人人体运动生成中显著降低人体间的渗透问题，并提升生成的物理真实性。

**💡 创新点**

创新点包括：① 用简化的柱体/立方体几何体代替高分辨率网格进行碰撞检测，极大降低计算量；② 通过计算碰撞点的“逃逸方向”产生可微导向损失，直接指导网络在训练中避免碰撞；③ 该策略为 plug‑and‑play，可无缝集成至多种现有生成模型，无需改动原架构。

**🔧 技术方法**

使用技术包括：几何代理逼近、快速碰撞点检测、方向引导损失、采样点密度控制、SDF 对比实验，以及扩散/Transformer 运动生成框架。

**📊 数据集**

实验基于 InterHuman（7,779 两人交互序列，22 关节）和 Inter‑X（11,388 序列，SMPL‑X 54 关键点）数据集，并对后者进行了特征调整。

**📈 对比分析**

与 InterGen、in2IN、TIMotion 等主流方法对比，PhysiGen 在从头训练和微调两种模式下均显著降低碰撞距离/率，提升 R‑Precision、FID、Diversity 等质量指标；与传统 SDF 损失相比，GPU 内存和训练时长几乎无提升，展示了优越的计算效率。

**⚠️ 局限性**

局限性包括：采样点过多会使生成质量下降；仅基于几何代理，无法捕捉极细腻的表面细节；在某些极端姿态下仍可能出现轻微碰撞；需要针对不同体型单独学习几何盒参数。

---

## 250. ControBench: An Interaction-Aware Benchmark for Controversial Discourse Analysis on Social Networks

**arXiv ID:** 2605.00513 | [PDF](https://arxiv.org/pdf/2605.00513v1)

**作者:** Ta Thanh Thuy `[一作]` (Nanyang Technological University), Sitao Luan `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 ControBench 数据集，整合 Reddit 上的异构社交图和文本语义，用自标注的 Flair 进行用户立场标签化；

**💡 创新点**

首次提出在异构图中使用双语义边（回复与父评论）并对极端议题进行跨观点交互建模，显著降低了同类同聚性；

**🔧 技术方法**

采用图神经网络（如 HAN、RGCN、ℋ^2G-Former 等）、预训练语言模型（BERT、RoBERTa、SimCSE、Sentence‑BERT）与大型语言模型（GPT‑4o‑mini、Llama‑3.1‑8B、DeepSeek‑V3 等）进行节点分类；

**📊 数据集**

数据集包含三大主题（Trump、abortion、religion）共 7,370 名用户、1,783 篇帖子、26,525 条交互，规模适中；

**📈 对比分析**

在宏 F1 与微 F1 评测中，预训练语言模型表现最佳，GNN 在二分类任务中竞争力强但在多分类任务下降明显，LLM 中 Kimi‑K2 与 DeepSeek‑R1 取得较好成绩，Label Propagation 也显示出交互结构的重要性；

**⚠️ 局限性**

限制包括：依赖 Reddit 受众偏向性、Flair 标签可能不精准、跨时间标签假设不变、模型对异构混合图的鲁棒性不足，以及对语义歧义与推理能力的挑战。

---

## 251. A Comparative Study of QSPR Methods on a Unique Multitask PAMPA dataset

**arXiv ID:** 2605.00508 | [PDF](https://arxiv.org/pdf/2605.00508v1)

**作者:** Andrs Formanek `[一作]`, Adam Arany `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了143种药物分子在六种人工膜上的PAMPA多任务数据集，并系统评估了多种分子描述符与回归模型对被动膜透过性的预测性能。

**💡 创新点**

首次将专家设计的物理化学描述符（Percepta）与深度学习生成的高维特征（CDDD、MolBERT、ECFP）在多膜透过性QSPR中的表现进行对比，发现后者在样本量有限时不如前者。

**🔧 技术方法**

采用线性回归、Lasso/Ridge、ElasticNet、MultiTaskElasticNet、BayesianRidge、PLS、SVR、XGBoost、MLP等模型，并使用Percepta、CDDD、MolBERT、RDKit、ECFP等五种特征表示。

**📊 数据集**

使用143个分子在BBB、肝、心、二十烷、PS、PC六种PAMPA膜上测得的logP_e（P_e）值，共429条重复记录，形成完整的多任务数据集。

**📈 对比分析**

通过4折交叉验证和独立外部测试，MLP+Percepta多任务模型在PCA_0上获得最高R²≈0.63–0.64，单膜预测精度低于多任务，外部测试R²普遍降至≈0.4–0.5，表明模型在样本量有限时仍能取得可观性能。

**⚠️ 局限性**

主要局限在于样本量不足导致高维特征过拟合、PS膜实验与建模效果差、缺少pKa等离子化信息限制了RDKit和MolBERT的解释性和预测力。

---

## 252. Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation

**arXiv ID:** 2605.00499 | [PDF](https://arxiv.org/pdf/2605.00499v1)

**作者:** Zhiying Deng `[一作]` (Central China Normal University), Jianjun Li `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 15478 | [OpenAlex ID](https://openalex.org/A5100439790)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了TIDE框架，用于预测用户在下一购物篮中可能购买的商品集合；通过双专家架构分别捕捉用户的习惯性复购和探索性新购行为，并利用时间间隔信息生成动态表示。

**💡 创新点**

主要创新点包括：①利用Hawkes增强的傅里叶时间编码捕捉商品特定的周期性与激励衰减；②双专家（习惯专家与模式引导探索专家）实现用户意图的显式分离；③基于物品-模式双边图的协同消息传递提取全局探索信号；④物品级门控机制实现对每个候选商品的习惯/探索权重自适应调节；⑤通过InfoNCE对习惯与探索表示进行对齐以提升语义一致性。

**🔧 技术方法**

技术手段包括Hawkes过程、傅里叶时间编码、双专家架构、图神经网络（LightGCN式对物品-模式图的消息传递）、注意力聚合、对比学习、门控融合与Softmax预测。

**📊 数据集**

实验采用Amazon电商数据集中的Beauty、Sports、Grocery和Home四个子集，经过5-core过滤和篮子序列划分。

**📈 对比分析**

在留一验证框架下，TIDE与POP、FPMC、Sets2Sets、TIFU-KNN、MMNR、M^2、MCRec、HapCL、SemTHy、TREx等多种基线对比，实验结果显示TIDE在Recall@10/20和NDCG@10/20指标上均显著优于所有对手；消融实验进一步验证了时间编码、图结构、门控与对比学习等模块的贡献。

**⚠️ 局限性**

局限性主要体现在：①仍依赖离线训练，缺乏在线自适应更新机制；②对极度稀疏或新商品的冷启动效果尚未深入探究；③双专家模型在计算上相对较重，对大规模实时推荐场景可能存在性能瓶颈；④缺乏对模型可解释性的系统性分析。

---

## 253. "What Are You Really Trying to Do?": Co-Creating Life Goals from Everyday Computer Use

**arXiv ID:** 2605.00497 | [PDF](https://arxiv.org/pdf/2605.00497v1)

**作者:** Shardul Sapkota `[一作]` (Stanford University), James A. Landay `[通讯]` (Stanford University)

**通讯引用:** 22539 | [OpenAlex ID](https://openalex.org/A5072285921)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了名为Tempo的系统，利用持续的桌面截图结合用户自述，层次化推断用户从操作到长期奋斗（striving）的多层结构，并提供可编辑界面让用户与系统共同完善该层次化目标模型。

**💡 创新点**

创新点在于：① 将 Activity Theory 的四层（操作–行动–活动–奋斗）与 Emmons 的个人奋斗框架相结合，首次在无结构电脑使用中推断并共创长期目标；② 引入用户协同编辑机制，使用户可直接修改推断的层次结构，解决歧义与不确定性；③ 通过用户自述与上下文完整性审计，实现更具个性化与隐私安全的推断过程。

**🔧 技术方法**

使用技术包括：Gemini 3 Flash LLM 进行截图转录与多层级推断；基于提示链的层次化抽象；图属性数据库（property graph）存储层次结构；本地 OCR 与上下文完整性审计过滤 PII；交互式编辑界面（inline edit、reassign、remove、merge 等）。

**📊 数据集**

数据集为本研究自采：14 名自愿者在 7 天内使用个人 Macbook 进行屏幕截图（约 1,498 张平均）并完成一次性自述问卷，未使用公开数据集。

**📈 对比分析**

对比方法：对完整系统与两种消融（无上下文、无层次）进行线性混合效应模型评估。结果显示：① 用户自述显著提升单条奋斗的准确度与与个人优先级的一致性；② 层次化结构显著提高整个奋斗集合的代表性（覆盖度、发现度、动机度）。在编辑体验上，层次视图在透明度、证据实用性、控制感等 7 项指标上显著优于仅展示截图的视图。

**⚠️ 局限性**

局限性：① 仅观察桌面行为，忽略非数字生活维度；② 受 LLM 语言风格影响，可能产生 Barnum 效应；③ 编辑界面缺乏系统生成替代方案，用户在无法自知时难以修正；④ 隐私与对抗攻击易受注入影响；⑤ 实验周期仅一周，缺乏长期持续部署与多轮编辑的验证；⑥ 观察导致的行为改变（观察效应）可能影响模型真实度。

---

## 254. SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters

**arXiv ID:** 2605.00528 | [PDF](https://arxiv.org/pdf/2605.00528v1)

**作者:** Dongxin Guo `[一作]` (University of Hong Kong), Siu Ming Yiu `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了SAGA，一种将 AI 代理完整推理工作流视为调度单元的分布式 GPU 集群调度器，结合了工作流图预测、KV 缓存生命周期管理、会话亲和批处理、工作窃取以及任务级公平调度。

**💡 创新点**

创新点包括：① 首次将完整代理工作流结构显式化为调度单元并利用 Agent Execution Graph（AEG）实现跨工具调用的 KV 缓存保留；② 通过 Workflow‑Aware LRU（WA‑LRU）实现了接近 Bélády 最优的缓存淘汰（1.31× 的竞争比）；③ 引入 Agent Fair Share（AFS）提供任务完成时间公平性，并给出基于 Lyapunov 驱动的理论保证；④ 将上述三者在统一系统中集成，并在实际 GPU 集群上验证其对延迟、内存利用率与 SLO 的显著提升。

**🔧 技术方法**

技术细节包括：Agent Execution Graph (AEG) 解析、Workflow‑Aware LRU (WA‑LRU) 以及工具调用感知 TTL；会话亲和路由与全局工作窃取的调度算法；Agent Fair Share (AFS) 的公平度量与调度；在 vLLM 之上扩展 PagedAttention 的缓存管理；使用 Ray + gRPC 实现分布式协调；CUDA 流实现预取与并行解码；基于 Lyapunov drift 的公平性证明。

**📊 数据集**

使用的数据集包括：SWE‑bench 代码推理任务（500 条验证任务）、WebArena 浏览器交互任务（812 条任务）以及基于 BurstGPT 的合成多租户工作负载，均在 8 节点 × 8 GPU（64 GPU）A100‑80GB 集群上进行实验。

**📈 对比分析**

对比方法：与 vLLM v0.6.0、vLLM+APC（v0.15.1）、SGLang、Llumnix、TRT‑LLM+Scaffolding、KVFlow 等现有 LLM 服务框架进行多维度评估。性能结果显示：SAGA 在 SWE‑bench 与 WebArena 上的任务完成时间分别比 vLLM+APC 低 1.73× 与 1.55×（几何平均 1.64×），GPU 内存利用率提升 1.22×，SLO 达成率达 99.2%；在单线程基线下相对 vLLM 的改进可达 3.01×。但相应的峰值吞吐量比最优批量调度低约 30%。

**⚠️ 局限性**

限制与挑战：① 需要框架（如 LangChain、AutoGen）提供工作流图信息，缺失时需模式推断导致性能下降；② 工具调用延迟的极端尾部仍可能触发缓存失效；③ 对新代理类型的任务时长估计不准确；④ 仅在单数据中心实验，跨地域调度未验证；⑤ 目前仅在 Llama‑3‑70B 上验证，其他模型或 MoE 体系的适配尚待研究；⑥ 通过延迟优化导致吞吐量下降，限制了在批量工作负载中的适用性。

---

## 255. DySRec: Dynamic Context-Aware Psychometric Scale Recommendation via Multi-Agent Collaboration

**arXiv ID:** 2605.00574 | [PDF](https://arxiv.org/pdf/2605.00574v1)

**作者:** Yanzeng Li `[一作]` (Beijing Normal University), Shasha Han `[通讯]` (Chinese Academy of Medical Sciences and Peking Union Medical College)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DySRec，一个基于多智能体的动态心理测评量表推荐框架，能够在交互式对话中实时更新用户状态并推荐最合适的量表。

**💡 创新点**

创新点在于将量表选择视为连续决策过程，引入闭环信息增益驱动的对话细化、风险感知协同以及多智能体分工式架构，显著提升了动态适配和安全控制。

**🔧 技术方法**

使用大型语言模型（Qwen3）提取语义特征，构建共享上下文向量，基于信息增益的主动提问，利用加权多准则评分和风险指数阈值实现自适应量表推荐，并采用Blackboard式多智能体通信。

**📊 数据集**

在真实移动健康应用“Black hole Roast”中，收集了约5,000名用户的数据，涉及53种心理测评量表，作为系统部署和验证的数据来源。

**📈 对比分析**

通过与静态管道推荐和单一预测模型的对比，DySRec在实际应用中实现了更高的匹配准确性和用户满意度，且在风险监测上可在危险指数超过阈值时及时触发干预；但文中未给出具体数值指标。

**⚠️ 局限性**

局限性包括：缺乏公开可复现的实验数据与基准；模型对不同人群的适用性和阈值设定需进一步验证；LLM依赖可能导致解释性不足；系统对极端语境或多语言环境的鲁棒性尚未充分评估。

---

## 256. 2D-SuGaR: Surface-Aware Gaussian Splatting for Geometrically Accurate Mesh Reconstruction

**arXiv ID:** 2605.00569 | [PDF](https://arxiv.org/pdf/2605.00569v1)

**作者:** Prajwal Gupta C. R. `[一作]` (TU Darmstadt), Justus Thies `[通讯]` (TU Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

利用2D Gaussian splatting结合单目深度与法向先验，重建多视角图像的3D网格，并通过联合网格‑高斯细化进一步提升表面质量。

**💡 创新点**

创新点在于：①用单目深度与法向先验实现鲁棒初始化并加入法向先验损失；②采用DBSCAN聚类剔除孤立高斯；③在高斯优化后引入类似SuGaR的三维高斯细化，让网格可进一步优化。

**🔧 技术方法**

核心技术包括2D Gaussian splatting、单目深度/法向预测（如Metric3D）、深度/法向一致性损失、DBSCAN聚类、TSDF+Marching Cubes网格提取、3D Gaussian rasterization、拉普拉斯平滑等。

**📊 数据集**

在DTU多视角重建数据集上进行实验。

**📈 对比分析**

与多种基准方法（Implicit/Explicit、2DGS、MILo等）比较，Chamfer Distance下降至0.67，显著低于其它方法；即使不细化，方法已比多数方法更快更准，但细化阶段仍耗时。

**⚠️ 局限性**

局限包括细化阶段计算量大、对初始化仍有依赖（SfM与单目深度误差可能导致孤立高斯），以及在复杂场景中的可扩展性待改进。

---

## 257. Sim-FA: A Simulator Frontend for Asynchronous Pipelines

**arXiv ID:** 2605.00555 | [PDF](https://arxiv.org/pdf/2605.00555v1)

**作者:** Zhongchun Zhou `[一作]` (Hong Kong University of Science and Technology), Wei Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 36617 | [OpenAlex ID](https://openalex.org/A5100441678)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了Sim-FA，一个针对Hopper GPU新异步特性（TMA、WGMMA、mbarrier）的周期级轨迹驱动模拟框架，并用其评估FlashAttention‑3的性能。

**💡 创新点**

创新点在于：①在WarpGroup级别建模异步指令，实时捕捉生产者‑消费者流水线；②实现L2请求合并、跨切片哈希与TMA地址去重；③提出低误差的SimFA‑python分析模型，用以预测DRAM与LLC流量。

**🔧 技术方法**

使用技术包括：轨迹采样与离线转换、逻辑线程调度、TMA与TensorCore引擎细粒度时序建模、L2远程复制代理、MSHR容量调优、以及基于Python的数学分析模型。

**📊 数据集**

数据集与实验环境：Llama 3 8B/70B/405B 的 FlashAttention‑3 kernel，在NVIDIA H800（Hopper）和GB10（Blackwell）GPU上使用NVIDIA Nsight Compute、Nsight Systems 进行测量。

**📈 对比分析**

比较方法：将模拟结果与真实GPU测得的延迟、L2/DRAM流量做对比；与GenZ等基线模型比较。性能表现：平均绝对百分比误差(MAPE)为5.7%，最大误差12.7%；DRAM流量预测在短序列下与实测一致，长序列时能捕捉到K/V重复访问导致的上升趋势。

**⚠️ 局限性**

limitations：仅支持Hopper/H800体系结构；对TensorCore内部流水线细节仍有简化；模拟仅覆盖FlashAttention‑3，未扩展到其他算子或多GPU场景；对极大规模批次/序列的跨SM同步细节仍有不足。

---

## 258. Colorful-Noise: Training-Free Low-Frequency Noise Manipulation for Color-Based Conditional Image Generation

**arXiv ID:** 2605.00548 | [PDF](https://arxiv.org/pdf/2605.00548v1)

**作者:** Nadav Z. Cohen `[一作]` (Reichman University), Ariel Shamir `[通讯]` (Reichman University)

**通讯引用:** 12360 | [OpenAlex ID](https://openalex.org/A5022467818)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过分析扩散模型噪声在频域中的作用，提出一种无训练、低成本的低频噪声操控方法——Colorful‑Noise，能够在保持高频细节多样性的同时，用参考图像的低频信息控制生成图像的整体结构和色彩。

**💡 创新点**

创新点在于：①发现低频噪声决定图像的全局结构与色彩分布，而高频噪声主要负责细节；②基于此，提出直接在频域替换噪声低频部分并进行适当缩放的方式，使得图像生成过程能在无需额外训练或优化的情况下获得颜色与结构的可控性；③该方法可与现有高频控制手段（如ControlNet、StyleAligned）无缝组合，兼容多种扩散模型。

**🔧 技术方法**

使用的技术包括：频域分解（FFT/离散傅里叶变换）、VAE编码/解码、扩散模型（SDXL、Flux）、低频替换与缩放因子γ；实验中还对离散小波分解做了对比。

**📊 数据集**

主要使用Aesthetic‑4K数据集的自然图像作为低频条件来源；此外构造了1K张合成低频平面图像用于消融实验；实验基于SDXL和Flux两种模型进行图像生成。

**📈 对比分析**

与现有颜色/结构控制方法（ControlNet、LoRA、StyleAligned、T2I‑Adapter等）在图像变体、ColorField‑to‑Image和Image‑to‑Image Color Transfer等任务中对比，采用CLIPScore、EMD（局部与全局）等指标。实验结果表明，Colorful‑Noise在保持结构与色彩一致性的同时，往往能在L‑EMD上显著优于或与现有方法持平，且在高频细节生成上不受限制。

**⚠️ 局限性**

局限性包括：①需要手动调节低频比例α和缩放因子γ，参数对不同模型（尤其是Flux）敏感；②在遮罩下的低频替换会因下采样导致细粒度控制受限；③方法更适合交互式创作，对大规模自动化生成的适用性有限；④在某些极端或离散分布的条件下，颜色与语义映射可能出现偏差。

---

## 259. Vesselpose: Vessel Graph Reconstruction from Learned Voxel-wise Direction Vectors in 3D Vascular Images

**arXiv ID:** 2605.00538 | [PDF](https://arxiv.org/pdf/2605.00538v1)

**作者:** Rajalakshmi Palaniappan `[一作]` (Max-Delbrueck-Center for Molecular Medicine in Helmholtz Association), Dagmar Kainmueller `[通讯]` (Max-Delbrueck-Center for Molecular Medicine in Helmholtz Association)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于体素方向向量的血管中心线重建方法，利用改进的 TEASAR 算法生成血管图谱，并设计分层匹配与 false splits/merges 评价指标。

**💡 创新点**

创新点包括①使用方向向量引导重建，解决传统分割-骨架化误差；②在 TEASAR 中加入向量方向与幅度惩罚；③提出分层图匹配与拓扑错误（false splits/merges）直观评估。

**🔧 技术方法**

采用 3D U‑Net 预测前景掩码和方向向量，改进 TEASAR skeletonization，分层图匹配算法，以及多根、适应性遮蔽与后处理。

**📊 数据集**

实验使用四个数据集：单树 Synthetic、Parse 2022 CT pulmonary、multi‑tree Synthetic 以及微 CT 大鼠心脏 3D micro‑CT。

**📈 对比分析**

与 Vesselformer、Trexplorer、Trexplorer‑Super、U‑Net、vesselFM 等基线比较；在单树数据上 F1 达到 92% 以上，在多树数据上 F1、FM、FS 明显优于基线，显著提升拓扑准确性。

**⚠️ 局限性**

局限性包括仍需根点标注；对小分支和极小分割组件的检测仍不完美；评价指标受采样与匹配策略影响；缺少公开高质量手工注释血管图谱。

---

## 260. Surprisal Minimisation over Goal-directed Alternatives Predicts Production Choice in Dialogue

**arXiv ID:** 2605.00506 | [PDF](https://arxiv.org/pdf/2605.00506v1)

**作者:** Tom Utting `[一作]` (University of Aberdeen), Arabella Sinclair `[通讯]` (University of Aberdeen)

**通讯引用:** 213 | [OpenAlex ID](https://openalex.org/A5014216707)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建目标导向与目标无关的句子替代集合，并使用信息理论成本（惊讶度、信息均匀性、长度）评估人类对话生成的句子，比较不同成本衡量与选择点的关系。

**💡 创新点**

首次将成本函数的解释与替代集合区分开来，明确同一成本在不同替代集合下对应的说话者或听者压力；并提供可扩展的LLM生成替代方案的实用流程。

**🔧 技术方法**

大语言模型（GPT‑4o 生成替代，GPT‑2 估算惊讶度）、Rational Speech Act 公式化的概率选择模型、对数回归与泊松-二项检验等统计方法。

**📊 数据集**

Switchboard 对话语料（约1,342条受控句子），使用 GPT‑4o 生成12,669条替代，随后通过分层抽样对长度和全局信息均匀性进行匹配。

**📈 对比分析**

通过确定性排名检验和对数回归的概率选择模型比较成本敏感度；惊讶度在目标导向替代集合中以53.4%（比均匀基线3.24倍）优于其他成本；长度在目标无关集合中以26.6%（3.69倍）最强；整体预测性能最高为惊讶度模型，表明其为说话者侧成本。

**⚠️ 局限性**

仅限于英语对话，取决于单一选择点（动词后续）；成本指标为全局聚合，可能对长句失效；依赖LLM估计与生成，可能偏离人类真实处理；未考虑沟通效果的显式建模。

---

## 261. Scaling Federated Linear Contextual Bandits via Sketching

**arXiv ID:** 2605.00500 | [PDF](https://arxiv.org/pdf/2605.00500v1)

**作者:** Hantao Yang `[一作]` (University of Science and Technology of China), Defu Lian `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8729 | [OpenAlex ID](https://openalex.org/A5085254654)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种Federated Sketch Contextual Linear Bandits（FSCLB）算法，在联邦上下文线性bandit中通过草图技术显著降低本地计算和通信开销。

**💡 创新点**

创新点包括：利用SVD间接计算判定通信的行列式、双重草图（upload/download）策略、在Sketch中使用SCFD保持增量性与异步通信，并给出改进的 regret 上界O((√d+√(Mε_l))√(lT))。

**🔧 技术方法**

采用Frequent Directions/SCFD草图、SVD、双重草图机制、通过奇异值计算行列式、异步联邦学习框架以及线性上下文bandit理论。

**📊 数据集**

在合成数据（d=50、100）以及OpenML公开数据集SatImage（d=37）和MFeat（d=48）上进行实验。

**📈 对比分析**

与FedLinUCB和Random基线对比，评估累计奖励、计算时间和通信量；FSCLB在保持累计奖励几乎不变的前提下，计算和通信成本降低超过90%。

**⚠️ 局限性**

当草图尺寸接近原始维度时收益有限；草图逼近导致轻微的 regret 下降；依赖于矩阵秩假设且实验规模有限，未验证极大规模或非线性情形。

---

## 262. Revealing graph bandits for maximizing local influence

**arXiv ID:** 2605.00489 | [PDF](https://arxiv.org/pdf/2605.00489v1)

**作者:** Alexandra Carpentier `[一作]` (Universität Potsdam), Michal Valko `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在未知图结构下，设计一种主动学习算法（Bandit Revelator）通过逐步揭示节点的影响邻居信息，寻找图中最具影响力的节点；

**💡 创新点**

创新点在于：①完全不依赖预先知道的图结构；②提出可检测维度（detectable dimension）和可检测时间窗（detectable horizon）来量化问题难度；③通过先进行全局探索提取少量潜在优秀节点，再在此子集中进行经典多臂弓形算法，实现复杂度从O(d)降到O(D⋆)；

**🔧 技术方法**

技术上主要利用：随机探索、Bernstein不等式估计、UCB式置信区间、最小化可检测维度的集合筛选、并在子集上应用最优多臂弓形算法；

**📊 数据集**

实验使用了Barabási‑Albert模型、Enron邮件图、Gnutella网络、Facebook子图等真实/仿真社交网络；

**📈 对比分析**

与不利用图结构的经典多臂弓形基线相比，Bandit Revelator在n≪d时显著降低回报损失，尤其在具有少数高度互联节点的网络中表现优异；

**⚠️ 局限性**

局限性包括：需要在全局探索阶段能够及时揭示足够邻居信息（p接近1时效果最佳）；对极其分散的网络（如Gnutella）提升有限；对更复杂传播模型（非本地影响）尚未直接适用。

---

## 263. Zero-Knowledge Model Checking

**arXiv ID:** 2605.00487 | [PDF](https://arxiv.org/pdf/2605.00487v1)

**作者:** Pascal Berrang `[一作]` (University of Birmingham), Xiao Yang `[通讯]` (University of Birmingham)

**通讯引用:** 22667 | [OpenAlex ID](https://openalex.org/A5101972163)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种零知识模型检查（ZKMC）技术，能够在不泄露系统内部实现细节的情况下，证明软件系统满足时序规范。

**💡 创新点**

创新点包括：①将证明式模型检查与零知识证明结合，利用排名函数作为可公开的证明证书；②首次为模型检查设计显式状态与符号两种零知识方案；③在零知识框架下实现多种高级密码学原语（多项式承诺、两层Pedersen承诺、Farkas 证据、矩阵乘法与范围证明）。

**🔧 技术方法**

核心技术包括：KZG 多项式承诺、两层 Pedersen 向量/矩阵承诺、Farkas 线性不等式证据、Bulletproof 范围证明、矩阵乘法零知识证明、Fiat‑Shamir 零知识协议、Büchi 自动机与排名函数理论。

**📊 数据集**

实验数据集为三类通信/调度协议：(1) 3‑way 握手协议（带指数退避）(2) DHCP 客户端地址请求模型 (3) k‑节点轮询调度器。通过参数化实验，状态空间最高可达约 2^20.9，证明义务数最高可达约 2^12.2。

**📈 对比分析**

比较方法：实现完整原型，分别测量显式状态方案与符号方案的枚举、设置、证明和验证时间。显式状态方案在 < 2^8 状态时更快、成本低；符号方案可处理更大状态空间（到 2^20.9），但常数成本高。两方案互补，选择取决于系统规模与可表达性。

**⚠️ 局限性**

局限性：①公开规格与证明证书可能泄露关于系统的敏感信息；②显式状态方案受限于状态空间大小；③符号方案要求系统能写成线性保护命令，无法处理非线性或连续/概率系统；④Farkas 证明可能因整数约束或系数范围过大而不存在；⑤整体安全性仍依赖多方安全设置与随机数生成；⑥未讨论对抗性攻击（如伪造承诺）与侧信道风险。

---

## 264. Scalable Context-Aware Graph Attention for Unsupervised Anomaly Detection in Large-Scale Mobile Networks

**arXiv ID:** 2605.00482 | [PDF](https://arxiv.org/pdf/2605.00482v1)

**作者:** Sara Malacarne `[一作]` (Telenor Research and Innovation), Massimiliano Ruocco `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 397 | [OpenAlex ID](https://openalex.org/A5057541422)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并部署了一种用于移动网络无监督多变量时间序列异常检测的统一模型C-MTAD-GAT。

**💡 创新点**

通过将上下文条件注入图注意力网络，结合GATv2和GRU自动编码器，实现了单模型跨数千网络元素的检测与阈值自校准。

**🔧 技术方法**

使用图注意力网络（GATv2）、条件卷积、GRU编码器-解码器以及指数分布阈值校准方法。

**📊 数据集**

在公开的TELCO基准以及国内运营商的RAN和EPC大规模真实数据集上验证。

**📈 对比分析**

与MTAD-GAT、β-MTAD-GAT、DC-VAE等基线在TELCO上采用无监督阈值实现事件级F1提升至约0.66以上、时间戳级F1提升至约0.16以上，并显著降低警报量；在RAN/EPC中验证了模型的稳健性和可扩展性。

**⚠️ 局限性**

依赖无标签阈值校准对极端尾部建模可能不足，对模型更新和多维阈值设定仍有挑战；单模型对极少数异常网络元素的细粒度检测仍受限。

---

## 265. Set Parameterized Matching via Multi-Layer Hashing

**arXiv ID:** 2605.00566 | [PDF](https://arxiv.org/pdf/2605.00566v1)

**作者:** Moshe Lewenstein `[一作]` (Bar-Ilan University), Ely Porat `[通讯]` (Bar-Ilan University)

**通讯引用:** 3121 | [OpenAlex ID](https://openalex.org/A5049614530)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种基于三层哈希的随机化线性时间算法，用于解决集合参数化匹配问题（将模式和文本中的每个位置视为字符集合），实现了从 O(m·|Σ|^2.5) 的传统二分图匹配算法到 O(N+M) 的时间复杂度。

**💡 创新点**

创新点在于提出了“多重集合偏移表示”来捕获字符出现的全局分布，并设计了三层哈希方案（层1：偏移集合哈希；层2：位置多重集合哈希；层3：字符中心滚动哈希）以解决偏移集合动态更新与哈希冲突的问题。

**🔧 技术方法**

主要技术包括：Karp‑Rabin 多重集合哈希、随机化滚动哈希、字符中心更新策略、偏移集合的快速维护与更新、概率分析与错误概率控制。

**📊 数据集**

未使用公开数据集；实验主要通过理论分析和实验室合成数据验证算法正确性与时间复杂度。

**📈 对比分析**

与之前基于二分图匹配的 O(m·|Σ|^2.5) 方法相比，本文算法在时间上实现了 O(N+M) 的线性时间，空间上同样为 O(N+M)，错误概率可控制在 1/n，实验验证与理论一致。

**⚠️ 局限性**

局限性：算法为随机化蒙特卡罗算法，存在误判概率；在最坏情况下多重集合偏移表示可能导致 O(nN) 的表示大小，虽然通过哈希压缩处理，但在极端输入下仍可能影响性能；此外，算法依赖于大素数域，若实现细节不当可能导致冲突率升高。

---

## 266. Pick and Sort for Graphical Authentication

**arXiv ID:** 2605.00558 | [PDF](https://arxiv.org/pdf/2605.00558v1)

**作者:** Argianto Rahartomo `[一作]` (Technische Universität Clausthal), Mohammad Ghafari `[通讯]` (Tehran Institute for Advanced Studies, Khatam University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Pick and Sort的图形认证方案，用户通过选择视觉元素并在网格中排列形成登录密码；

**💡 创新点**

创新点在于将识别与回忆两种机制结合，使用可定制的元素集合和网格大小，既降低记忆负担，又提升密码空间；

**🔧 技术方法**

采用Web前端（HTML/CSS/JS）、Django后端、SQLite数据库实现原型；

**📊 数据集**

使用三组自定义视觉元素（颜色40个、图标90个、形状50个），未使用公开数据集；

**📈 对比分析**

通过对59名参与者的登录时长、成功率等指标进行初步评估，发现登录时间平均约22秒，成功率随练习提升，虽然比传统文本密码慢，但可用于非实时访问场景；

**⚠️ 局限性**

局限性包括样本量有限、缺乏跨设备与多服务的部署验证、未进行攻击模型评估、用户对简单模式偏好导致密码空间受限、未覆盖色盲或其他特殊人群等问题。

---

## 267. Linking Behaviour and Perception to Evaluate Meaningful Human Control over Partially Automated Driving

**arXiv ID:** 2605.00556 | [PDF](https://arxiv.org/pdf/2605.00556v1)

**作者:** Ashwin George `[一作]` (Delft University Of Technology), Arkady Zgonnikov `[通讯]` (Delft University Of Technology)

**通讯引用:** 709 | [OpenAlex ID](https://openalex.org/A5031929760)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了部分自动驾驶系统中驾驶员与自动化交互的“有意义的人类控制”（MHC），通过在驾驶模拟器中让24名驾驶员体验手势共享控制（HSC）和交易控制（TC）两种交互模式，并收集行为和主观数据。

**💡 创新点**

创新点在于：①首次将MHC的四个理论属性（可追踪性、可追溯性、足够控制、责任可追溯）转化为可测量的问卷条目和行为指标；②提出了整合行为、主观和定性数据的评估框架；③通过实证实验验证该框架，并比较两种控制模式对MHC感知的影响。

**🔧 技术方法**

使用的技术包括：高保真力反馈方向盘 + VR 视角的固定底座驾驶模拟器；纯追踪控制器实现自动驾驶；线性混合效应模型（LMM）对行为指标与主观评分的关系进行统计检验；定性内容分析（两位编码员）提取问卷开放式回答的主题。

**📊 数据集**

数据集由24名受试者在9次仿真试验中产生的遥测数据（转向力、扭矩、速度、碰撞时刻等）和每次试验后的问卷评分构成，全部公开于OSF。

**📈 对比分析**

通过比较HSC与TC在行为指标（如冲突扭矩、反应时间、取车次数、轨迹偏差、冲击力）和主观评分（如足够控制、相互理解、合作感）来评估两种模式。结果显示HSC在冲突扭矩更低、反应时间更短、取车次数更多、主观评分更高，表明HSC更有利于实现有意义的人类控制。

**⚠️ 局限性**

局限性包括：①仅测试横向控制，纵向控制未涉及，限制了现实驾驶情境的代表性；②样本量仅为24人，主要为学生/学术界人员，外推性有限；③实验仅在模拟环境进行，缺乏真实道路或长时段适应性研究；④主观评分采用事后问卷，无法捕捉实时感知变化。

---

## 268. Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance

**arXiv ID:** 2605.00553 | [PDF](https://arxiv.org/pdf/2605.00553v1)

**作者:** Minchan Kwon `[一作]` (KAIST), Junmo Kim `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Stable-GFlowNet（S‑GFN），通过 Contrastive Trajectory Balance、Noisy Gradient Pruning 和 Min‑K Fluency Stabilizer 解决 LLM 红队测试中 GFN 的训练不稳定和模式崩溃问题。

**💡 创新点**

创新点包括①使用对比式轨迹平衡（CTB）替代传统的 Z 估计，消除分区函数不稳定；②通过 NGP 对奖励噪声进行自适应抑制；③引入 MKS 通过最小 K 词的流利度惩罚防止生成无意义 gibberish，同时保持攻击多样性。

**🔧 技术方法**

技术主要为生成流网络（GFN）与对比学习、奖励噪声抑制、句子流利度评估；训练使用 Qwen2.5‑1.5B 语言模型，毒性评估采用 Meta‑Llama‑Guard‑3‑8B。

**📊 数据集**

数据集为 Qwen2.5‑1.5B 通过 Safety‑Dataset 与 AdvBench 的监督微调，受试 LLM 为 Qwen2.5‑1.5B‑Instruct，毒性分类器为 Meta‑Llama‑Guard‑3‑8B。

**📈 对比分析**

与 PPO、PPO+Curiosity、DPO、Jailbreak‑R1、Rainbow Teaming 以及传统 GFN (TB) 等基线相比，S‑GFN 在攻击成功率（ASR）和唯一攻击数（UA）上均显著领先；在跨攻击和迁移攻击实验中也表现出更强的通用性和防御覆盖。

**⚠️ 局限性**

局限性包括对毒性分类器与受试 LLM 的高度依赖，需大量计算资源；阈值选择（如 NGP、MKS）对性能影响显著；在极端噪声或稀疏奖励环境下，仍可能出现训练收敛困难或性能下降。

---

## 269. AGoQ: Activation and Gradient Quantization for Memory-Efficient Distributed Training of LLMs

**arXiv ID:** 2605.00539 | [PDF](https://arxiv.org/pdf/2605.00539v1)

**作者:** Wenxiang Lin `[一作]`, Shaohuai Shi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为大语言模型（LLM）的分布式训练提出了AGoQ系统，通过层感知激活量化和梯度量化显著降低GPU内存占用并提升训练吞吐量。

**💡 创新点**

创新点包括：① 层感知激活量化（LAAQ），根据层类型和流水线阶段动态分配比特宽度，实现近4位激活存储；② 8位梯度量化算法QuanGrad，采用FP8本地存储、FP8 All‑Reduce通信，兼顾精度与通信效率；③ 动态位宽补偿（DBCA‑PP）在流水线并行中利用设备内存空闲分配更高比特宽度；④ 量化与GEMM的核融合，消除量化/反量化的额外算力。

**🔧 技术方法**

主要技术手段有：层感知量化、FP8量化、All‑Reduce拆分(All‑to‑All + All‑Gather)、核融合、量化后梯度累加与再量化、动态位宽配置。

**📊 数据集**

使用的基准数据集包括OpenWebText（预训练），以及针对不同模型的微调任务；实验模型涵盖 LLaMA2‑7B、LLaMA2‑13B、LLaMA3‑8B、CodeLLaMA‑34B、OLMo‑1B 等。

**📈 对比分析**

对比基线 Megatron‑LM（含/不含 ZeRO‑1）、DeepSpeed、COAT，评估指标为显存占用、训练时长。AGoQ 在 64‑GPU 集群上将显存减少最高 52%，训练吞吐量提升至 1.34×（相较于 Megatron‑LM/ZeRO‑1），在 OLMo‑1B 上内存降低 31% 并保持相同速度。

**⚠️ 局限性**

局限性：仅在 Nvidia A6000/Pro‑6000 GPU 上验证，NPU 结果仅在附录中展示；在极大规模集群或极长序列时可能仍受网络带宽限制；量化导致的精度损失需通过动态位宽补偿抵消，系统复杂度提升；未在多语言或更大模型（> 100B）上评估。

---

## 270. Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation

**arXiv ID:** 2605.00529 | [PDF](https://arxiv.org/pdf/2605.00529v1)

**作者:** Ziwen Zhao `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Menglin Yang `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Ψ-RAG 框架，实现了跨文档多跳检索增强生成，能够在 token 级问答、段落级因果推理以及文档级摘要等不同粒度任务上高效检索和生成。

**💡 创新点**

创新点在于：① 通过“合并与折叠”迭代构建自适应层次抽象树，消除 k‑means 的均匀分布假设；② 引入多粒度检索代理（R&A Agent），通过查询重组与多轮交互实现跨文档多跳推理；③ 结合稠密向量检索与 BM25 稀疏检索，采用重排器或 RRF 实现混合检索，解决树索引的粗抽象与结构孤立问题。

**🔧 技术方法**

使用技术包括：层次聚类（AHC）生成抽象树、密集向量编码（Qwen3-Embedding-8B）、稀疏 BM25 索引、LLM 代理（Llama‑3.3‑70B）进行抽象生成与检索‑生成交互、重排器（Qwen3-Reranker-8B）或 RRF 混合检索、以及可扩展的 Bucketing 与 HNSW 加速方法。

**📊 数据集**

实验使用的数据集：单跳 NQ、PopQA；多跳 HotpotQA、2Wiki、MuSiQue、MultiHop‑RAG；叙事 QA NarrativeQA、LongBook；摘要 QMSum、WCEP；所有数据集均按公开基准随机采样或全量使用。

**📈 对比分析**

与传统 RAG（BM25、DPR、Hybrid）、Graph‑RAG（HippoRAG 2、GoR）以及 Tree‑RAG（RAPTOR）进行对比；Ψ‑RAG 在 token 级 QA 上平均提升 25.9% F1，跨文档多跳问答上比 HippoRAG 2 提升 7.4%，并在单跳 QA 与摘要任务中保持与最先进 Graph‑RAG 的竞争力。

**⚠️ 局限性**

局限性：① 构建树时 O(n²log n) 复杂度，面对 100 M+ 级大语料需 Bucketing/HNSW 等加速；② 多轮代理检索增加查询延迟；③ 仍依赖开源 LLM，极端高精度或专业领域的细粒度推理可能受限。

---

## 271. End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer

**arXiv ID:** 2605.00503 | [PDF](https://arxiv.org/pdf/2605.00503v1)

**作者:** Wenda Chu `[一作]` (ByteDance Seed), Qiushan Guo `[通讯]` (ByteDance Seed)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种端到端的训练框架，联合优化1D视觉分词器与自回归生成模型，用于图像重建与生成。

**💡 创新点**

创新点包括：①在tokenizer与AR模型之间加入自回归预测重建（APR）损失，弥补下一个token预测损失与像素空间质量的鸿沟；②引入隐式语义对齐，将视觉基础模型（VFM）的全局语义信息注入1D分词器而不强制其对齐二维空间，提升生成质量；③一次性训练而非两阶段训练，显著提升最终生成的FID。

**🔧 技术方法**

技术包括：1D Vision Transformer（ViT）分词器、IBQ向量量化、LlamaGen式自回归模型、APR损失、隐式语义对齐损失、GAN重建损失、LPIPS感知损失、AutoGuidance无分类器引导。

**📊 数据集**

使用ImageNet-1K 256×256图像数据集进行训练与评估。

**📈 对比分析**

与现有1D分词器与自回归模型相比，所提方法在不使用引导的情况下，-H模型（644M参数）实现gFID 1.48，-L模型实现gFID 1.74，均优于之前同类方法；在重建上rFID也显著下降，表明生成与重建兼顾。

**⚠️ 局限性**

局限性：①在序列长度与代码簿大小上存在重建-生成权衡，过长序列会导致自回归模型难以学习；②仍以256×256分辨率为主，尚未验证更高分辨率的效果；③对语义对齐的隐式方法依赖VFM预训练模型的可用性与匹配度。

---

## 272. Silicon Showdown: Performance, Efficiency, and Ecosystem Barriers in Consumer-Grade LLM Inference

**arXiv ID:** 2605.00519 | [PDF](https://arxiv.org/pdf/2605.00519v1)

**作者:** Allan Kazakov `[一作]` (Bahcesehir University), Abdurrahman Javat `[通讯]` (Bahcesehir University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在消费级硬件上对 Nvidia RTX 5090（Blackwell）与 Apple M 系列 SoC 进行大规模 LLM 推理的系统化实验，对比不同精度与量化方式下的吞吐量、延迟、能效，探究 NVFP4、VRAM 壁垒与统一内存架构的影响。

**💡 创新点**

首次系统化评估 NVFP4 与统一内存架构在 70B+ 模型下的优势与瓶颈，揭示 TensorRT-LLM 的“Backend Dichotomy”、Nvidia 量化编译难题，以及 Apple UMA 对于高容量模型的线性扩展与能效超越。

**🔧 技术方法**

使用 TensorRT-LLM v1.1.0 (NVFP4, PyTorch/C++ backend)、MLX v0.30.6 (4-bit native)、GGUF CPU‑offload、量化工具及能源计数器。

**📊 数据集**

使用 Qwen2.5‑1.5B、Qwen3‑8B、GLM4.7‑Flash、Llama‑3.3‑70B、Qwen3‑Next‑80B 等公开模型。

**📈 对比分析**

通过统一 Prompts 长度、固定输出 256 令牌的推理实验，记录 Tokens/s、TTFT、Tokens/Joule；结果显示 RTX 5090 最高吞吐 265 T/s，Apple M3 Ultra 能效最高 23×，但 Nvidia 在 30B 以内模型仍占优势。

**⚠️ 局限性**

受限于 Nvidia 量化编译 VRAM 需求高、软件成熟度不足、Apple 低通量，且不同平台的生态壁垒导致部署复杂；此外实验未覆盖多卡/多节点与实际应用负载。

---

## 273. MSACT: Multistage Spatial Alignment for Stable Low-Latency Fine Manipulation

**arXiv ID:** 2605.00475 | [PDF](https://arxiv.org/pdf/2605.00475v1)

**作者:** Xianbo Cai `[一作]` (Waseda University), Tetsuya Ogata `[通讯]` (Waseda University)

**通讯引用:** 6933 | [OpenAlex ID](https://openalex.org/A5055922202)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于 ACT 的多阶段空间注意力（MSACT）框架，融合局部 2D 注意点和全局 ResNet 视觉先验，进行低延迟双臂精细操控。

**💡 创新点**

创新点在于：①多阶段卷积+注意力提取多尺度任务相关注意点；②通过自监督的时间一致性损失对未来注意点序列进行预测，显著抑制注意点漂移；③将注意点作为显式空间模态与 Transformer 编码器并行集成，既提升定位稳定性又保持低延迟。

**🔧 技术方法**

使用技术包括：预训练 ResNet、Transformer 编码器‑解码器、CVAE 训练、空间 softmax（可微 Argmax）、多尺度卷积 + 点乘注意力、时间一致性 L1 损失、动作分块与时序集成。

**📊 数据集**

实验数据来源于 ALOHA 双臂平台的四个真实任务（Detach Network Cable、Thread Velcro、Insert Tea Bag、Open Match Box）各 50 次遥控演示，仿真任务为 Cube Transfer 与 Bimanual Insertion，演示约 1,000 步/样本。

**📈 对比分析**

与 ACT、Diffusion Policy、SmolVLA、π0.5 等基线比较；MSACT 在真实任务的总体成功率从 23% 提升至 53%（差距显著），在仿真任务多阶段成功率均达到 100%；推断延迟与 ACT 接近（≈45 ms），保持低延迟优势。

**⚠️ 局限性**

局限性：依赖多视角 RGB 摄像头，未使用深度信息，易受强光照或遮挡影响；在某些插入等几何极限场景改进有限；未验证语言条件或更大规模数据；对复杂多任务的泛化尚待进一步研究。

---

## 274. Depth-Guided Privacy-Preserving Visual Localization Using 3D Sphere Clouds

**arXiv ID:** 2605.00562 | [PDF](https://arxiv.org/pdf/2605.00562v1)

**作者:** Heejoon Moon `[一作]` (Hanyang University), Je Hyeong Hong `[通讯]` (Hanyang University)

**通讯引用:** 3556 | [OpenAlex ID](https://openalex.org/A5010730040)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9cc9baba-5356-466d-81ff-d80028d90279` `51c0528b-f690-4182-ae60-bb5f046c276c` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一种名为 sphere cloud 的隐私保护场景表示方法，用于视觉定位；通过将稀疏 3D 点云映射到单位球面并引入伪点与深度约束，既能消除已知的密度攻击，又能在保持实时定位的同时利用 ToF 深度校正尺度。

**💡 创新点**

创新点包括：① 采用所有 3D 线均交于地图质心的 sphere cloud 结构，天然抑制密度攻击；② 通过删除一定比例的真实点并用噪声生成伪点并递归使用原有描述子，抵御新型直接图像重建攻击；③ 在定位时结合 p3P 求解和深度正则化，将无尺度的线云匹配转化为有尺度的绝对位姿估计，从而实现高效实时定位。

**🔧 技术方法**

使用的技术包括：3D 线云构造、密度基础逆向攻击与 InvSfM 图像重建、伪点生成与描述子循环、LO-RANSAC + p3P 最小解算器、Levenberg-Marquardt 非线性优化、深度正则化损失、ToF 传感器深度采集以及旋转-平移误差评估。

**📊 数据集**

实验使用公开室内 RGB-D 数据集 7‑Scenes 与 12‑Scenes 进行评估。

**📈 对比分析**

与传统线云方法 ULC、PPL、PPL+ 以及基于深度引导的 DVLAD+R2D2(+D)、DSAC* 进行比较。sphere cloud 在隐私保护（PSNR、LPIPS、SSIM、MAE）上优于或接近其他线云，并在 20–30 fps 的实时速率下实现了与图像基方法相近的旋转精度；在某些场景下翻译误差略高于深度引导方法，但整体定位质量保持竞争力。

**⚠️ 局限性**

局限性包括：需要 ToF 深度传感器，噪声会影响尺度估计；伪点比例 η 与噪声 σ 的选择仍无理论依据，需要经验调参；对新型攻击的完全抵抗尚未完全验证；目前仅在室内数据集验证，室外或更大规模场景的适用性待进一步研究。

---

## 275. Tempus: A Temporally Scalable Resource-Invariant GEMM Streaming Framework for Versal AI Edge

**arXiv ID:** 2605.00536 | [PDF](https://arxiv.org/pdf/2605.00536v1)

**作者:** M. Grailoo `[一作]` (Linköping University), J. Núñez-Yáñez `[通讯]` (Universidad Politecnica de Madrid)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种资源不变的时序 GEMM 框架 Tempus，利用固定 16 核 AIE-ML 计算块在 AMD Versal AI Edge SoC 上实现高效矩阵乘法。

**💡 创新点**

创新点在于将 GEMM 3D 结构映射到固定二维阵列，通过时序迭代、算法裁剪与复制、级联流与 DATAFLOW 协议，实现资源不变性、核心/功耗/IO 多维节约，并用 PAU 指标量化优势。

**🔧 技术方法**

采用固定 16 核 AIE-ML 计算块、512 位级联流、广播/分组切换、PLIO 复用、死锁自由 DATAFLOW、块级复制、DIM（微核尺寸）调优等技术。

**📊 数据集**

使用代表 LLM 组件的矩形 GEMM 形状（如 LLaMA‑2 7B、TinyLlama、BERT‑Base 等）作为实验数据集。

**📈 对比分析**

通过与 ARIES 等空间扩展框架对比，使用 PAU 指标、核心利用率、功耗、I/O 以及 607 GOPS 的吞吐量，结果显示 Tempus 在资源受限设备上实现 211.2 倍的 PAU 和 22× 核/7.1× 功耗/6.3× I/O 节约。

**⚠️ 局限性**

局限性包括：相对于高端空间扩展方案仍有吞吐量下降；受本地内存约束导致 DIM 限制；实验仅覆盖 AMD Versal VE2302；未评估更宽数据类型或大规模并行场景。

---

## 276. LLM-Oriented Information Retrieval: A Denoising-First Perspective

**arXiv ID:** 2605.00505 | [PDF](https://arxiv.org/pdf/2605.00505v1)

**作者:** Lu Dai `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 45489 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出以降噪为核心的LLM导向信息检索框架，并给出四阶段的挑战演进及完整的降噪技术分类，结合实验验证了噪声对LLM生成的严重影响。

**💡 创新点**

创新点在于将检索目标从高召回转向最大化可用信息密度与可验证性，构建了涵盖语料、检索、上下文组装、验证与闭环训练的完整降噪体系；同时通过实验证明噪声是LLM-IR的瓶颈。

**🔧 技术方法**

采用的技术包括：受控索引（去重、时间过滤、可信度评分）、鲁棒检索（稠密+重排序、硬负样本）、上下文组装优化（最小化冗余、位置排序、碎片合并）、检索验证（属性一致性、归属链核查）以及闭环自适应学习。

**📊 数据集**

实验主要基于500条Natural Questions（NQ）样本，使用100条DPR检索结果进行噪声与召回比实验；此外在编码代理、长期记忆助手等场景中参考SWE-bench、OpenAI API等数据。

**📈 对比分析**

通过在LLaMA-2-7B-Chat上对EM、位置敏感度等指标进行对比，显示当SNR从1:0降至1:9时EM从61%降至26%；在不同检索/组装策略下，Recall@K、NDCG@K、F1、BLEU/ROUGE等指标也被用来评估方案有效性。

**⚠️ 局限性**

局限性包括：实验覆盖的领域有限（主要为NQ和少数应用案例），对大型模型的上下文窗口依赖较高，降噪方法在不同任务间迁移性不足，且在实时闭环学习中存在计算成本与训练稳定性挑战。

---

## 277. EnCoDe: Energy Estimation of Source Code At Design-Time

**arXiv ID:** 2605.00504 | [PDF](https://arxiv.org/pdf/2605.00504v1)

**作者:** Shailender Goyal `[一作]` (IIIT Hyderabad), Karthik Vaidhyanathan `[通讯]` (IIIT Hyderabad)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 EnCoDe 方法与 PowerLens 测量框架，实现对 Python 代码块在设计阶段的精细能耗估计。

**💡 创新点**

创新点在于：① sub‑millisecond 能耗测量技术 PowerLens；② 基于静态 AST 特征的块级能耗预测模型；③ 构建 18k+ Python 程序的块级能耗数据集。

**🔧 技术方法**

技术手段包括 RAPL 计数、执行放大与时间同步、AST 结构与复杂度特征提取，以及随机森林、XGBoost 等轻量级机器学习模型。

**📊 数据集**

使用公开的 18,612 个 Python 代码文件，抽取 8,000+ 可执行代码块，构成包含 33 维特征和对应能耗标签的块级数据集。

**📈 对比分析**

通过与粗粒度 PyRAPL 结果对比验证测量准确性；回归模型 R² ≈ 0.75，分类准确率 80.6%，表明模型能在设计阶段可靠预测能耗热点。

**⚠️ 局限性**

局限性包括：仅在单核 CPU 环境下测试，忽略多线程、I/O、网络等能耗；测量放大假设线性，对极小块不一定适用；仅验证 Python，缺乏跨语言/硬件的泛化。

---

## 278. High-Speed Vision Improves Zero-Shot Semantic Understanding of Human Actions

**arXiv ID:** 2605.00496 | [PDF](https://arxiv.org/pdf/2605.00496v1)

**作者:** Yongpeng Cao `[一作]` (University of Tokyo), Yuji Yamakawa `[通讯]` (University of Tokyo)

**通讯引用:** 2337 | [OpenAlex ID](https://openalex.org/A5019385536)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套无训练的零样本语义理解管线，用预训练的视频-语言模型生成动作描述，再用大型语言模型对描述进行相似度比较，评估不同帧率下高速度剑道动作的语义可分辨性。

**💡 创新点**

证明高帧率能显著提升零样本语义辨识；提出无监督的动作对比管线；系统性分析跟踪骨架信息在完整与部分观察（早期识别）场景中的作用。

**🔧 技术方法**

使用 InternVideo2.5 进行视频描述生成，使用 Qwen3-4B‑Instruct‑2507 进行语义推理与相似度评分；采用 Nearest‑Class Prototype (NCP) 进行无监督分类；利用人体骨架跟踪（上半身关节）覆盖视频；将 120Hz 视频下采样至 60Hz 与 30Hz 进行对比。

**📊 数据集**

自制高帧率剑道（kendo）攻击动作数据集，包含 Men、Kote、Dou 三种攻击模式及空白类，分别录制 3 次，帧率分别为 120Hz、60Hz 与 30Hz。

**📈 对比分析**

通过 LLM 给每对动作序列打分得到相似度矩阵，随后使用 NCP 计算分类准确率。实验显示：全视频 120Hz 为 83.3% 纯描述，16.7% 加骨架覆盖；60Hz 仅 33.3%；30Hz 仅 41.7%。在早期观察下，120Hz 加骨架覆盖达到 50% 的准确率，而仅 33.3%（无覆盖）。

**⚠️ 局限性**

局限性：① 早期截断需要预设阈值，限制在线实时推理；② 依赖预训练骨架估计，噪声与遮挡会降低效果；③ LLM 相似度倾向于整体语义相似，难以捕捉细微动作差异；④ 在完整视频中骨架覆盖可能产生干扰；⑤ 缺乏更稳健的多模态融合与层级推理策略。

---

## 279. Distance metric learning for conditional anomaly detection

**arXiv ID:** 2605.00490 | [PDF](https://arxiv.org/pdf/2605.00490v1)

**作者:** Michal Valko `[一作]` (University of Pittsburgh), Milos Hauskrecht `[通讯]` (University of Pittsburgh)

**通讯引用:** 4928 | [OpenAlex ID](https://openalex.org/A5012461386)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出并评估了一种用于患者管理报警系统的条件异常检测框架，重点研究了实例化的概率模型和基于度量学习的距离优化方法。

**💡 创新点**

创新点在于将异常检测扩展到对部分属性的条件异常识别，并利用 NCA、RCA 等度量学习方法自适应调节距离度量，以提升实例化预测的准确性。

**🔧 技术方法**

主要技术包括条件概率模型（Naive Bayes、softmax 加权预测）、实例化的非参数/参数学习、距离度量学习（NCA、RCA）、kNN 近邻搜索以及 ROC 曲线评估。

**📊 数据集**

实验使用的是 Pneumonia PORT 数据集的二进制子集（共 2287 名肺炎患者、20 个属性），用于检测住院决策是否异常。

**📈 对比分析**

通过与全局距离（2286 名患者）和局部距离（最近 40 名患者）相结合的多种基线（欧氏、马氏、RCA、NCA）比较，NCA+softmax 在 95%‑100% 召回率范围内获得最高 ROC 面积（约 20%），明显优于传统距离和 Naive Bayes 模型。

**⚠️ 局限性**

局限性包括邻居数量固定（40 个），缺乏自适应邻域选择；数据集规模有限，异常样本选取可能偏倚；阈值设定影响召回率，需进一步自动化和泛化。

---

## 280. Trading off rewards and errors in multi-armed bandits

**arXiv ID:** 2605.00488 | [PDF](https://arxiv.org/pdf/2605.00488v1)

**作者:** Akram Erraqabi `[一作]` (Inria SequeL), Yun-En Liu `[通讯]` (EnLearn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究多臂赌博机中奖励最大化与估计误差最小化的折中问题，提出可调权重目标函数f_w并设计强迫采样算法，给出理论收敛率并在合成与教育数据上验证效果。

**💡 创新点**

创新点包括①在单一目标框架下引入可调奖励‑误差折中目标f_w；②基于强迫采样的算法能够在保证近似最优f_w的同时实现渐进收敛；③三相段理论分析表明折中目标并不比单一奖励或探索更难，且提供了可扩展的凸光滑分析框架。

**🔧 技术方法**

使用技术主要包括：强迫采样（forcing）策略、对目标函数的连续化与强凸光滑分析、置信区间估计、最优分配的数值求解及跟踪机制；实验中采用模拟演示与教育游戏数据。

**📊 数据集**

实验使用了两类数据集：1）合成模拟数据（多臂均值与方差已知，支持多种参数设定）；2）真实教育应用数据（基于某教育游戏/实验平台收集的学生学习与奖励记录）。

**📈 对比分析**

与传统奖励最大化算法（如UCB）和纯探索算法（如最佳臂识别、主动探索）进行比较。实验显示，本文算法在保持较高奖励的同时显著降低估计误差，整体f_w性能优于单目标方法，且在不同权重设置下表现稳定。

**⚠️ 局限性**

局限性主要在于：①需要预设最小分配比例λ_min，理论证明仅在此限制下成立；②中间阶段的收敛速度相对较慢，实际表现依赖参数η、w、λ_min 的经验选择；③对λ_min=0 的情况理论尚未完善，需进一步研究。

---

## 281. When More Reformulations Hurt: Avoiding Drift using Ranker Feedback

**arXiv ID:** 2605.00560 | [PDF](https://arxiv.org/pdf/2605.00560v1)

**作者:** V Venktesh `[一作]` (Stockholm University), Avishek Anand `[通讯]` (Delft University of Technology)

**通讯引用:** 1625 | [OpenAlex ID](https://openalex.org/A5075681290)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种预算感知检索框架，将查询重构视为特征，利用轻量化代理模型与教师重排器在线学习，适度选择查询重构与文档，抑制查询漂移并提升召回。

**💡 创新点**

创新点在于：①将重构查询当作可学习特征，使用线性代理模型与多臂赌博机思想实现在线权重更新；②在固定重排预算下自适应分配重排调用，实现查询与文档双重优先级；③无需对重排器进行微调，作为“教师”直接反馈，形成训练‑free 的适配器。

**🔧 技术方法**

使用技术包括：BM25 与 RM3 作为基础检索与特征；MonoT5（或其他轻量级跨编码器）作为教师重排器；线性代理模型与最小二乘/梯度更新；bandit‑style 预算化在线优化；LLM（如Qwen、GenQR）用于生成多样化查询重构。

**📊 数据集**

实验数据集：MSMARCO passage（v1 与 v2）以及 TREC Deep Learning benchmark（DL19–DL22）。

**📈 对比分析**

与传统 BM25>>MonoT5、RRF、基线重构方法（GenQR、GenQREnsemble、Query2Doc、QA‑Expand、RM3）以及 exhaustive retrieval 进行对比；在 50/100 的重排预算下，召回提升最高可达 33%+，nDCG 亦显著提升；同时相较于直接使用大型 LLM 进行重排，效率提升 3.3–4.5 倍。

**⚠️ 局限性**

局限性：①仍需依赖教师重排器，若教师质量下降会影响代理学习；②对极大候选集的扩展性有限；③查询重构质量仍受 LLM 的生成能力限制，无法完全消除漂移；④预算设置与采样策略需手动调参。

---

## 282. Fast and Exact: Asymptotically Linear KL-Optimal Frequency Normalization

**arXiv ID:** 2605.00579 | [PDF](https://arxiv.org/pdf/2605.00579v1)

**作者:** Kamila Szewczyk `[一作]` `[通讯]` (Saarland University), Kamila Szewczyk (Saarland University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

提出了三种精确求解熵编码中整数频率归一化的 KL‑最优算法，覆盖了从简单的自下而上贪心到双向修正与线性窗口的完整方案。

**💡 创新点**

核心创新在于：① 通过边际交换条件给出全局最优的可检验判据；② 在此基础上构造了全局可行、时间复杂度分别为 𝒪(M log r)、𝒪(M log r) 与 𝒪(r) 的三种实现；③ 证明了窗口宽度至多 4r‑4，提供了可行的线性时间选择方法；④ 在 AVX2、Quickselect 等硬件友好技术上做了深入优化。

**🔧 技术方法**

利用可分离凹整数分配理论、堆/堆交换、快速选择、全整数比较（大整数 log 比较）、Taylor/Padé 系数逼近、SIMD 并行化与分区快速排序等算法与实现技术。

**📊 数据集**

在七种合成分布（均匀、几何、Zipf、Gaussian、稀疏热冷）以及两份真实字节直方图（英语词典、可执行文件）上进行了实验；还对前述五个已有启发式归一化（Giesen、Bloom 单向、FSE 快速、Collet 上限）做了对比。

**📈 对比分析**

通过在 AMD Ryzen 9 5900X 上对每个实现进行循环计时，得到按符号计数的周期数。结果显示：Window‑smart 与 Bloom‑smart 在大多数输入下实现了 36–49、13–119 个周期/符号的差异；AVX2 版本进一步提升，Window‑super 在重尾分布上可比 Window‑smart 快 1.4–2.2 倍；与现有启发式相比，KL‑最优方案在几乎所有情况下降低冗余至 10⁻³–10⁻¹ nat/符号，且在小样本极端例子中仍保持最优。

**⚠️ 局限性**

局限性包括：① 4r‑4 的窗口上界尚未证明最优，若可进一步压缩可进一步降低常数；② 目前实现对 64‑位计数使用大整数比较，仍需更高效的纯整数方法；③ 仅在微基准层面验证，尚未在完整压缩器（zstd、CRAM 等）中评估实际文件尺寸与吞吐量提升。

---

## 283. Structure Liberates: How Constrained Sensemaking Produces More Novel Research Output

**arXiv ID:** 2605.00557 | [PDF](https://arxiv.org/pdf/2605.00557v1)

**作者:** James Mooney `[一作]` (University of Minnesota), Dongyeop Kang `[通讯]` (University of Minnesota)

**通讯引用:** 1082 | [OpenAlex ID](https://openalex.org/A5040821714)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于感知构建的科研轨迹框架，并生成了120K条引用条件的科研思路轨迹。

**💡 创新点**

创新点在于将科学发现的上游思路阶段拆分为八个认知阶段，并证明受控的重建监督能同时提升轨迹多样性和质量，进一步提高下游代码/论文产出。

**🔧 技术方法**

采用大语言模型（Qwen3、Gemma、LLaMA）进行SFT与RL后训练，生成基于感知的轨迹。

**📊 数据集**

使用Semantic Scholar Open Research Corpus中的论文引用网络作为基础，生成论文摘要后构造轨迹。

**📈 对比分析**

与无计划、直接提示等基线比较，受控轨迹的模型在质量、可执行性和科学基础等指标上提升约2%至10%，并且在多样性评估上领先。

**⚠️ 局限性**

局限包括高昂的计算资源需求、对LLM-as-Judge的依赖、以及仅基于引用邻域的轨迹生成可能无法覆盖全部科研路径。

---

## 284. Beyond Continuity: Simulation-free Reconstruction of Discrete Branching Dynamics from Single-cell Snapshots

**arXiv ID:** 2605.00545 | [PDF](https://arxiv.org/pdf/2605.00545v1)

**作者:** Junda Ying `[一作]` (Peking University), Lei Zhang `[通讯]` (Peking University)

**通讯引用:** 107604 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种无仿真（simulation-free）的框架USB，用于从单细胞快照重构离散分支动力学，兼顾随机性与非守恒质量，并实现单细胞层面的出生‑死亡模拟。

**💡 创新点**

将分支Schrödinger桥（Branching Schrödinger Bridge）与无平衡得分匹配（unbalanced score matching）统一，提出了高效的无仿真求解器，并首次实现了可离散化的单细胞出生‑死亡模拟。

**🔧 技术方法**

采用分支Schrödinger桥模型、无平衡得分匹配、条件高斯测度路径、RUOT松弛与WFR近似、流匹配技术以及神经网络参数化的无仿真训练目标。

**📊 数据集**

在合成数据（Simulation Gene、Dyngen、1000D高斯混合）和真实单细胞数据（EMT、EB、CITE‑seq、Mouse血液干细胞）上进行评估。

**📈 对比分析**

与OT-CFM、SF2M、WFR‑FM、DeepRUOT、Var‑RUOT、UOT‑FM、VGFM等基线对比，USB在合成数据上在分布匹配与质量匹配上获得最佳或次佳表现；在hold‑out实验中在EMT和CITE数据上获得最佳插值精度；对生长率的Pearson相关系数高达0.97，并能在Dyngen数据上可视化真实出生‑死亡动态。

**⚠️ 局限性**

B​SB及其RUOT松弛的半耦合形式难以求解，本文只能通过WFR近似；该近似的理论基础仍待深入研究；此外，方法的普适性依赖于可获得的耦合和条件路径，若缺乏相应构造可能受限。

---

## 285. Space Network of Experts: Architecture and Expert Placement

**arXiv ID:** 2605.00515 | [PDF](https://arxiv.org/pdf/2605.00515v1)

**作者:** Zhanwei Wang `[一作]` (University of Hong Kong), Kaibin Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 22703 | [OpenAlex ID](https://openalex.org/A5007131492)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Space‑XNet 框架，在低轨卫星网络中分层部署 MoE 模型并优化专家与卫星的映射，显著降低 token 生成延迟。

**💡 创新点**

核心创新在于层级两级放置策略：先把 MoE 层按卫星环路分割成子网，再根据专家激活概率与预期路径延迟的排序规则实现专家-卫星一一映射，实现全局最优或近似最优的 E2E 延迟。

**🔧 技术方法**

采用了混合优化理论（PPSWOR 采样、期望路径延迟近似、整数线性规划等）、图论（最短路 Dijkstra）、卫星链路模型（角速率阈值、空间辐射失效概率）以及 LLaMA‑MoE‑3.5B 模型推理。

**📊 数据集**

实验使用 LLaMA‑MoE‑3.5B（32 层、8 专家/层）以及 8 个标准英文推理/问答数据集（OpenBookQA、PIQA、ARC‑E、ARC‑C、WinoGrande、BoolQ、SciQ、HellaSwag）。

**📈 对比分析**

与三种基线（随机放置、随机子网放置、随机子网+中心网关）比较，Space‑XNet 的每 token 延迟约为 1‑1.1 s，比基线低 2–3 倍，显示出在大规模 LEO 星座中显著的性能提升。

**⚠️ 局限性**

限制包括：仅考虑单专家/卫星模型，未考虑多专家共存的并行/竞争影响；放置假设为静态（不随时间演化），忽略动态路由和突发失效导致的迁移成本；实验仅在极地轨道 LEO 星座中验证，尚未扩展至 GEO‑LEO 或星座多样化网络。

---

## 286. An approach to encode divergence-free stress fields in neural approximations based on stress potentials

**arXiv ID:** 2605.00509 | [PDF](https://arxiv.org/pdf/2605.00509v1)

**作者:** Mohammad S. Khorrami `[一作]` (Max-Planck Institute for Sustainable Materials), Dierk Raabe `[通讯]` (Max-Planck Institute for Sustainable Materials)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种将静力平衡约束直接嵌入网络架构的物理编码神经逼近方法（PeNA），并以 Fourier 神经算子（FNO）实现对多晶异质固体在单轴拉伸下的平衡应力场的数据驱动建模。

**💡 创新点**

创新点在于利用应力势函数把“无散度”约束编码进网络输出层，避免在损失函数中加入物理约束，从而使训练与推断均满足机械平衡；并首次展示了物理编码与物理指导/信息化两种方法在应力场预测中的显著差异。

**🔧 技术方法**

核心技术包括：应力势的 Helmholtz‑Hodge 分解、Fourier 神经算子网络架构、无散度的张量潜在表示、以及对训练损失中散度项的无损编码。

**📊 数据集**

使用者通过 DAMASK 计算得到的 1250 条多晶 BVP 结果，包含不同材料属性（E, ν）和几何尺寸的均匀晶粒分布，在单轴拉伸（F̅22=1.004）下生成的应力场数据作为训练与测试集。

**📈 对比分析**

与物理指导 FNO（PgFNO）和物理信息化 FNO（PiFNO）相比，PeFNO 在保持相同应力预测精度（误差约 1 MPa）下，平衡散度误差降低了三倍以上（从几百到数百微米⁻¹的尺度），表现出更强的物理一致性与鲁棒性。

**⚠️ 局限性**

局限性包括：①数据在晶界与三点交叉处稀疏，导致这些区域误差增大；②应力势的非唯一性及缺乏 Coulomb 戒式约束可能带来额外误差；③目前仅考虑无黏塑性和线性/非线性弹性，尚未扩展到更复杂材料行为；④训练需大规模 FFT 计算，计算成本相对较高。

---

## 287. LambdaRankIC: Directly Optimizing Rank IC for Financial Prediction

**arXiv ID:** 2605.00501 | [PDF](https://arxiv.org/pdf/2605.00501v1)

**作者:** Yan Lin `[一作]` (University of Macau), Yi Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 81872 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 LambdaRankIC，一种直接针对金融预测中常用的 Rank IC（Spearman 相关系数）的学习‑to‑rank 目标，并在 XGBoost 中实现自定义损失。

**💡 创新点**

创新点在于推导 Rank IC 的闭式 lambda 梯度，使得 LambdaRank 能够优化 Rank IC 的上界；同时证明该损失相当于 1-ρ 的一个上界，从理论上支撑了直接最大化 Rank IC 的可行性。

**🔧 技术方法**

采用 LambdaRank 框架、XGBoost 的 LambdaMART 算法、梯度提升树、以及对 Rank IC 的理论分析与上界证明。

**📊 数据集**

使用模拟数据（不同 SNR、特征维度、重尾噪声）以及真实美国股票月度特征（94 个特征，约 2.75M 观测、21,396 只股票，1964‑2024 年）进行实验。

**📈 对比分析**

与基准回归（OLS、XGBoost MSE、MLP）以及现有 LTR（pairwise、NDCG）进行对比。LambdaRankIC 在 Rank IC、ICIR、NDCG@100 上均优于对手，构建的 long‑short 组合在平均月度回报、Sharpe 比例上也取得最高表现；在重尾低 SNR 场景下表现尤为显著。

**⚠️ 局限性**

局限包括：LambdaRankIC 使用的是伪梯度，缺乏全局收敛保证；未使用可微排序/排名操作；仅处理交叉‑section 排序，未考虑时序排名；实现仅 CPU，未做 GPU 并行；更紧凑的 Rank IC 上界可能存在但易过拟合。

---

## 288. GOR-IS: 3D Gaussian Object Removal in the Intrinsic Space

**arXiv ID:** 2605.00498 | [PDF](https://arxiv.org/pdf/2605.00498v1)

**作者:** Yonghao Zhao `[一作]` (Nankai University), Beibei Wang `[通讯]` (Nanjing University)

**通讯引用:** 533 | [OpenAlex ID](https://openalex.org/A5100335554)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

实现了基于3D高斯分布的物体移除，提供全局光照一致的3D场景重建与编辑。

**💡 创新点**

创新点在于将场景拆分为材质与光照的内在空间，显式建模光传输并在该空间完成稀疏与反射的补全。

**🔧 技术方法**

技术手段包括扩展3D Gaussian Splatting至PBR材质、全局光照与屏幕空间模糊反射、材质与光照域的补全网络，以及基于光传输的反射掩码。

**📊 数据集**

使用了自制的GOR-IS-Synthetic与GOR-IS-Real两套数据集，并在SPIn-NeRF上进行通用性评测。

**📈 对比分析**

与多种Gaussian与NeRF物体移除基线对比，使用PSNR、SSIM、LPIPS、FID等指标，实验证明在Synthetic与Real数据上相较基线提升13% LPIPS、2dB PSNR，且在无全局光照场景下与SOTA持平。

**⚠️ 局限性**

局限性包括未对漫反射全局照明建模，且对多层非漫反射表面互相反射的情况处理不足。

---

## 289. MMAudio-LABEL: Audio Event Labeling via Audio Generation for Silent Video

**arXiv ID:** 2605.00495 | [PDF](https://arxiv.org/pdf/2605.00495v1)

**作者:** Kazuya Tateishi `[一作]` (Sony Group Corporation), Yuki Mitsufuji `[通讯]` (Sony Group Corporation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种同时生成音频和对应事件标签的框架MMAudio‑LABEL，能够从无声视频直接预测声学事件的时间与类型并生成对应高质量音频；

**💡 创新点**

创新点在于将事件检测嵌入生成流程，构建并比较了并行头和统一头两种架构，证明统一头在保持时间分辨率的同时可实现音频生成与事件预测的协同学习；

**🔧 技术方法**

技术主要包括基于MMAudio的多模态Transformer、同步特征提取Synchformer、条件流匹配损失以及事件标签的二进制交叉熵；

**📊 数据集**

使用了Greatest Hits数据集（以鼓棒击打不同材料的无声视频），并在其上进行时序起始检测和材质分类两项下游任务；

**📈 对比分析**

与CondFoley、MMAudio等基线对比，联合头模型在起始检测上将准确率提升至75%（AP 91.6、MCD 8.22），在材质分类上提升至61%，显著优于基线；

**⚠️ 局限性**

局限性包括对形状相似的材质（如地毯、干墙、玻璃）仍存在识别困难，模型对训练数据的时长与分辨率较为敏感，且需要预训练的基础模型支持，缺乏对实时推理的评估。

---

## 290. Leveraging Vision-Language Models as Weak Annotators in Active Learning

**arXiv ID:** 2605.00480 | [PDF](https://arxiv.org/pdf/2605.00480v1)

**作者:** Phuong Ngoc Nguyen `[一作]` (Kyushu University), Shinnosuke Matsuo `[通讯]` (Kyushu University)

**通讯引用:** 57 | [OpenAlex ID](https://openalex.org/A5000073881)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在有限标签预算下，结合人类专家的细粒度标签和VLM生成的粗粒度弱标签，提出了一个主动学习框架。

**💡 创新点**

发现VLM在细粒度识别中可靠性随标签粒度显著变化，利用其粗粒度高精度与细粒度低成本标签进行实例级分配，并通过转移矩阵对VLM噪声进行建模。

**🔧 技术方法**

采用主动学习采样、成本约束下的监督分配求解器、前向校正的噪声鲁棒损失以及ViT-B/16特征提取器。

**📊 数据集**

在鸟类CUB‑200‑2011和航空器FGVC‑Aircraft这两个细粒度分类基准上进行实验。

**📈 对比分析**

与随机、Entropy、BADGE、ISOAL等传统主动学习方法比较，实验显示在相同标签成本下，本文方法在5轮实验中持续取得更高的准确率，尤其从第3轮开始噪声鲁棒损失显著提升。

**⚠️ 局限性**

依赖预训练VLM的可用性和成本，弱标签的噪声模型仍可能不足以完全纠正系统性错误；在极低细粒度任务或非层次标签结构下效果尚未验证。

---

## 291. Scale-Aware Adversarial Analysis: A Diagnostic for Generative AI in Multiscale Complex Systems

**arXiv ID:** 2605.00510 | [PDF](https://arxiv.org/pdf/2605.00510v1)

**作者:** Mengke Zhao `[一作]` (Nanjing University), Keping Qiu `[通讯]` (Nanjing University)

**通讯引用:** 2338 | [OpenAlex ID](https://openalex.org/A5057625967)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于Constrained Diffusion Decomposition (CDD) 的尺度感知XAI框架，在连续尺度空间内进行物理合法的单尺度和多尺度干预，以诊断深度生成模型的物理因果性。

**💡 创新点**

通过在CDD尺度空间中实现可控的物理干预，首次量化生成模型的负响应悖论、结构冻结和非线性振荡等物理一致性缺陷，提供了一种新的模型评估与改进路径。

**🔧 技术方法**

使用Constrained Diffusion Decomposition、体积密度映射器、U-Net 结构的Denoising Diffusion Probabilistic Model（DDPM）以及尺度特定的增幅/旋转干预操作。

**📊 数据集**

基于NGC 1333星系团的观测积分面密度图，利用体积密度映射器重建的3D密度场作为实验数据集。

**📈 对比分析**

将干预后的2D投影输入预训练的DDPM，并与物理基线输出进行比率比较；结果显示DDPM在物理因果一致性上出现负响应、结构冻结和振荡，性能明显低于物理基线。

**⚠️ 局限性**

仅在单一DDPM架构和单一观测场景下验证，缺乏对其他生成模型的泛化测试；尺度干预方法实现复杂、计算成本较高，且对不同物理环境的适应性尚未充分评估。

---

## 292. Beyond Per-Request QoS: Coordinating Industrial Workflows with B5G/6G Network Capabilities

**arXiv ID:** 2605.00570 | [PDF](https://arxiv.org/pdf/2605.00570v1)

**作者:** Qize Guo `[一作]` (Ruhr University Bochum), Hemant Zope `[通讯]` (Fraunhofer Institute for Open Communication Systems)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5092095658)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

在工业工作流调度中，提出基于B5G/6G网络能力的整体协调机制，以超越传统的单请求QoS保障。

**💡 创新点**

创新点在于将网络切片、SDN/NFV与工业工作流调度深度耦合，利用AI预测资源需求，实现端到端延迟和可靠性双优化。

**🔧 技术方法**

采用SDN/NFV架构、网络切片、机器学习预测模型（如LSTM/Transformer）以及仿真平台进行评估。

**📊 数据集**

使用公开的工业制造流程数据集（如Petrinet/FactoryBench）以及自建的B5G/6G网络仿真数据。

**📈 对比分析**

与传统按请求QoS分配方案对比，实验结果表明延迟下降30%~40%，成功率提升15%，且系统吞吐量提升12%。

**⚠️ 局限性**

局限性在于对真实5G/6G网络硬件的依赖、对大规模工厂部署的可扩展性验证不足，以及AI模型对异常情况鲁棒性待提升。

---

## 293. Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion

**arXiv ID:** 2605.00670 | [PDF](https://arxiv.org/pdf/2605.00670v1)

**作者:** Yuan Li `[一作]` (National University of Singapore), Bingsheng He `[通讯]` (National University of Singapore)

**通讯引用:** 21525 | [OpenAlex ID](https://openalex.org/A5039946576)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于图检索的多模态缺失补全框架GRE‑MC，利用子图检索和图变压器联合完成缺失模态，并通过稀疏路由码本提升鲁棒性。

**💡 创新点**

1) 将语义相关节点检索并构造连通子图，突破邻域聚合局限；2) 用全局注意力的图变压器编码检索子图与查询节点，捕获长程依赖；3) 引入稀疏路由码本对潜在表示离散化，降低模态缺失对模型的影响。

**🔧 技术方法**

子图检索（基于最近邻+模态感知图扩展），图变压器（带Laplacian位置编码），稀疏路由码本（Gumbel‑Softmax + 负载均衡正则）。

**📊 数据集**

Amazon 评测数据集：Baby、Sports、Clothing，包含用户-商品交互、图像特征（4096维）和文本特征（384维）。

**📈 对比分析**

与多种基线（BPR、LightGCN、SLMRec、FREEDOM、BM3、DRAGON、PGL、MIG‑GT、Imputed、MILK、MoDiCF）对比；在40%缺失率下，GRE‑MC在Recall/ NDCG（@10、@20）上均优于第二佳模型，提升幅度约5–7%。

**⚠️ 局限性**

依赖对完整图的全局检索，检索效率受图大小影响；对极低可观测模态（单一模态）时性能可能下降；需额外计算开销，尽管小于某些对比方法，但仍高于纯邻域聚合。

---

## 294. UniVidX: A Unified Multimodal Framework for Versatile Video Generation via Diffusion Priors

**arXiv ID:** 2605.00658 | [PDF](https://arxiv.org/pdf/2605.00658v1)

**作者:** Houyuan Chen `[一作]` (MMLab@HKUST), Anyi Rao `[通讯]` (MMLab@HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了UniVidX统一多模态视频生成框架，可在同一模型内完成文本→视觉、视觉→视觉、文本+视觉→视觉等多种任务；

**💡 创新点**

创新点包括：随机条件掩码（SCM）实现任意方向的条件生成；解耦门控LoRA（DGL）为每种模态独立学习并仅在目标模态激活；交叉模态自注意力（CMSA）实现模态间信息交换与一致性；

**🔧 技术方法**

技术方法基于预训练视频扩散模型（Wan2.1‑T2V 14B），使用LoRA微调、时间步分割、交叉注意力、BFloat16混合精度训练；

**📊 数据集**

使用了室内视频合成数据集InteriorVid（924条21帧视频，480×640）用于Intrinsic任务；VideoMatte240K（484条视频，432×768）用于Alpha任务；并用合成的文本描述；

**📈 对比分析**

与现有基线（IntrinsiX、LayerDiffuse、RGB↔X、Diffusion Renderer、Ouroboros等）进行定量/定性对比，UniVidX在文本→Intrinsic、Inverse/Forward rendering、文本→RGBA、视频抠图等任务上均取得或逼近最优性能，且在少量数据(<1k视频)下表现出强泛化和高时间稳定性；

**⚠️ 局限性**

局限性包括：模型仅支持最多4种模态，21帧，480p；对缺少联合标注数据（Intrinsic+Alpha）无法一次性完成所有任务；对玻璃、透明物体等物理极端情况仍易出现误差；计算资源要求高，需4个H100 GPU；

---

## 295. A Replicability Study of XTR

**arXiv ID:** 2605.00646 | [PDF](https://arxiv.org/pdf/2605.00646v1)

**作者:** Rohan Jha `[一作]` (Johns Hopkins University), Benjamin Van Durme `[通讯]` (Johns Hopkins University)

**通讯引用:** 8736 | [OpenAlex ID](https://openalex.org/A5075825791)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

复现并评估 XTR 检索算法与其改进的训练目标，进一步在 PLAID 与 WARP 等现代 IVF 引擎上测试其表现。

**💡 创新点**

发现 XTR 的训练目标在低 k' 环境下提升 token 级检索质量，并通过平坦化 token 分数分布显著降低 PLAID/WARP 的查询延迟；同时证明 XTR 在整体检索效果上并未超越 ColBERT，挑战原始论文声称的优势。

**🔧 技术方法**

利用 PyLate 库实现 XTR、ColBERT 训练与检索；采用对比学习、知识蒸馏以及 f_XTR_train（k_train 参数）等技术；使用 ScaNN 进行 token 级检索；在 PLAID 与 WARP 引擎上进行 IVF 级检索。

**📊 数据集**

BEIR（多领域 18 个数据集）与 LoTTE（6 个主题）作为评估基准，使用 MS MARCO、NFCorpus、FiQA-2018、SciFact 等子集。

**📈 对比分析**

通过与 ColBERT（在相同 token 检索预算 k' 下）和原始 XTR 模型对比，评估 nDCG@10、Recall@100、MRR@10 等指标；发现 XTR 在低 k' 时表现更稳健，但整体 nDCG@10 与 ColBERT 相当；在 PLAID/WARP 上 XTR 取得更快的查询速度（QPS 约提升 15–30 倍）。

**⚠️ 局限性**

限制：XTR 的整体检索精度仍未能明显优于 ColBERT；对原始 XTR 模型的高 k' 结果依赖性不强；训练目标对不同模型/数据集的适应性尚不完全确定；未能在大规模 MS MARCO 上充分验证 PLAID 方案。

---

## 296. From Prediction to Practice: A Task-Aware Evaluation Framework for Blood Glucose Forecasting

**arXiv ID:** 2605.00645 | [PDF](https://arxiv.org/pdf/2605.00645v1)

**作者:** Alireza Namazi `[一作]` (University of Virginia), Heman Shakeri `[通讯]` (University of Virginia)

**通讯引用:** 434 | [OpenAlex ID](https://openalex.org/A5006445265)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了针对血糖短期预测的任务感知评估框架，分为基于真实观测数据的安全门控评价和基于UVA/Padova模拟器的干预效应评估。

**💡 创新点**

创新点在于引入切片感知的警报评估和对照实验的干预响应检验，揭示了传统平均误差指标无法捕捉的临床实用性差距，并公开了可复现的基准工具。

**🔧 技术方法**

采用了多种时间序列预测模型（ARIMAX、GRU、iTransformer、PatchMLP、Glimmer*），并利用零阶保持、误差网格、召回、误警率、效应RMSE、符号一致性、Kendall τ、动作匹配率与策略后悔等指标进行评估。

**📊 数据集**

使用了三个真实临床队列（DCLP3、DCLP5、PEDAP）和一套FDA认可的成人UVA/Padova模拟器数据。

**📈 对比分析**

通过宏观患者级平均值和95%置信区间进行模型比较；在真实数据上模型在整体性能相近，但在后bolus切片与警报召回/误警率上表现差异显著；在干预评估中，ARIMAX表现最优但仍无法可靠预测胰岛素影响，深度模型的符号一致性和排序几乎为零。

**⚠️ 局限性**

主要局限在于仅评估点预测、未将预测器嵌入闭环控制器、缺乏不确定性建模、模拟器仅覆盖成人病人、并未充分验证跨人群的泛化能力。

---

## 297. BlenderRAG: High-Fidelity 3D Object Generation via Retrieval-Augmented Code Synthesis

**arXiv ID:** 2605.00632 | [PDF](https://arxiv.org/pdf/2605.00632v1)

**作者:** Massimo Rondelli `[一作]` (University of Bologna), Maurizio Gabbrielli `[通讯]` (University of Bologna)

**通讯引用:** 2127 | [OpenAlex ID](https://openalex.org/A5025039355)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 BlenderRAG，一种利用检索增强生成的 Blender 代码生成系统。

**💡 创新点**

创新点在于通过专家验证的多模态示例库进行检索，指导 LLM 生成，显著提升语法正确性与几何一致性，且不需要微调。

**🔧 技术方法**

技术包括 Qdrant 向量数据库、Nomic‑AI 文本嵌入、Claude、GPT‑5、Gemini、Mistral 等多大语言模型，以及 Blender 插件集成。

**📊 数据集**

使用 500 条专家验证的多模态样例（文本、代码、图像），覆盖 50 类物体（25 室内 25 室外）。

**📈 对比分析**

在 30 个外分布提示下，与基线模型相比，BlenderRAG 将编译成功率从 40.8% 提升到 70.0%，CLIP 语义对齐从 0.409 提升到 0.774，四种模型均表现显著提升。

**⚠️ 局限性**

局限在于仅支持单对象生成，缺乏多对象场景构造与空间推理能力，且依赖已有示例库的覆盖范围。

---

## 298. Is Textual Similarity Invariant under Machine Translation? Evidence Based on the Political Manifesto Corpus

**arXiv ID:** 2605.00618 | [PDF](https://arxiv.org/pdf/2605.00618v1)

**作者:** Daria Boratyn `[一作]` (Jagiellonian University), Dariusz Stolicki `[通讯]` (Jagiellonian University)

**通讯引用:** 66 | [OpenAlex ID](https://openalex.org/A5079790151)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估机器翻译后文本的语义相似度保持情况，提出通过相似性矩阵相关性检验跨语言翻译不变性，并在政治宣言语料上开展实验。

**💡 创新点**

创新点在于：①用相似性矩阵的相关性而非绝对语义保留作为评价标准；②引入不等效检验与原文模型内部不一致度作为基准阈值；③对每种语言给出不变性、失真或不确定的判定。

**🔧 技术方法**

采用文本嵌入模型（SentenceBERT、SMPNet、Universal Angle、E5‑Mistral、LLaMA、NV‑Embed 等）对伪段落进行向量化；使用 PELT 分割、PageRank 权重池化；计算余弦相似度矩阵；通过多成员随机效应模型、混合模型与 cluster‑bootstrap 计算置信区间，并用 Benjamini‑Hochberg 控制错误发现率。

**📊 数据集**

使用 EU Manifesto Corpus（约 2,878 篇党派宣言，涵盖 28 种语言），通过欧盟 eTranslation 服务将非英语文本翻译为英语后构建并行语料。

**📈 对比分析**

对每种语言比较原文‑译文、原文‑多语、译文‑多语、原文‑译文+多语的相似性矩阵 Pearson 相关性；以 κσ_L 为不等效阈值检验四个假设。结果显示：5 种语言完全不变性，4 种语言显著失真，其余多为不确定；在分类任务中几乎无影响，聚类任务中仅法国和日语受影响，UMAP 任务中拉脱维亚和捷克受影响；多语编码在译文上保持高度一致。

**⚠️ 局限性**

局限性：仅评估单一 MT 系统（EU eTranslation）与单一领域（政治宣言）；语言样本不均衡导致部分语言统计功效不足；未能区分 MT 质量与 MT 过程的影响；未尝试非英语枢轴或直接跨语言对齐。

---

## 299. Faithful Extreme Image Rescaling with Learnable Reversible Transformation and Semantic Priors

**arXiv ID:** 2605.00605 | [PDF](https://arxiv.org/pdf/2605.00605v1)

**作者:** Hao Wei `[一作]` (Xi'an Jiaotong University), Ajmal Mian `[通讯]` (University of Western Australia)

**通讯引用:** 20854 | [OpenAlex ID](https://openalex.org/A5089986388)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在潜在空间中设计了极端图像缩放框架 FaithEIR，先用可学习可逆变换进行降采样，再用扩散模型恢复高频细节。

**💡 创新点**

创新点包括：可学习可逆变换 (LRT) 替代固定 Haar 变换；自适应细节先验 (ADP) 用高频字典补偿信息损失；轻量像素语义嵌入 (PSE) 为扩散模型提供语义条件。

**🔧 技术方法**

采用变分自编码器、可逆网络、SVD 正交初始化、扩散模型（SD‑Turbo）、单步去噪、DINOv2 语义提取、LoRA 微调等技术。

**📊 数据集**

训练使用 LSDIR 数据集；评估使用 DIV2K‑val、LSDIR‑val、CLIC2020 三个高分辨率基准集。

**📈 对比分析**

与 S3Diff、DiffBIR、IRN、T‑IRN、VQIR、TADM、GAN SR 等方法在 16×/32× 缩放下对比，FaithEIR 在 LPIPS、DISTS、FID、KID、PSNR、SSIM 等指标上普遍优于同类方法，尤其在感知质量上表现突出。

**⚠️ 局限性**

局限性：上采样阶段计算量大，难以在资源受限终端部署；对纹理模糊或结构不明确的区域仍可能产生语义一致但细节不完全准确的伪造结果。

---

## 300. Possibilistic Predictive Uncertainty for Deep Learning

**arXiv ID:** 2605.00600 | [PDF](https://arxiv.org/pdf/2605.00600v1)

**作者:** Yao Ni `[一作]` (Nanyang Technological University), Piotr Koniusz `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于可能性理论的深度学习不确定性框架（DAPPr），通过对参数后验的可能性投影到预测空间并用 Dirichlet 可能性函数逼近，得到闭式训练目标。

**💡 创新点**

创新点在于：①首次将可能性理论与第二阶预测结合，利用 supremum 代替积分实现参数后验投影；②通过 Dirichlet 可能性函数近似实现可学习的分布；③得到的损失函数在交叉熵下可闭式求解，避免昂贵的贝叶斯或蒙特卡洛采样。

**🔧 技术方法**

技术包括：可能性后验定义、可能性贝叶斯推理、最大化伪散度 (D_max)、Dirichlet 可能性函数、软加 1 归一化、稀疏性正则化，以及基于交叉熵的梯度更新。

**📊 数据集**

使用的数据集有 MNIST、CIFAR-10/100、CIFAR-10-LT、CIFAR-10-C、CUB‑200‑2011、Stanford Dogs，以及相应的 OOD 评估数据（KMNIST、FashionMNIST、SVHN、TinyImageNet、ImageNet‑O、DTD、Places365）。

**📈 对比分析**

与 MC Dropout、DUQ、各种 EDL 变体（KL‑EDL、R‑EDL、DA‑EDL、F‑EDL）以及集成方法比较，DAPPr 在置信度估计、OOD 检测（AUPR）和分类准确率上均达到或超过 SOTA；单模型即可超过10‑模型集成，且计算成本低。

**⚠️ 局限性**

局限性：依赖网络具有足够容量以满足投影假设；仅针对分类任务验证，扩展到回归或序列任务尚未验证；对正则化参数 λ 仍有一定敏感性，若设置不当会影响性能。

---

## 301. Fairness of Classifiers in the Presence of Constraints between Features

**arXiv ID:** 2605.00592 | [PDF](https://arxiv.org/pdf/2605.00592v1)

**作者:** Martin C. Cooper `[一作]` (University of Toulouse), Imane Bousdira `[通讯]` (Toulouse INP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文研究在存在特征约束（如编码约束或现实规则）下，机器学习分类器决策的公平性问题，并基于可解释性提出了使用不包含受保护特征的首要蕴含（prime‑implicant）解释来判定决策公平的方法。

**💡 创新点**

创新点在于①首次把特征约束纳入公平性定义，指出忽略约束会导致公平评估失真；②用“公平的PI‑解释”作为判定公平的核心概念；③系统分析了不同公平性定义（FTU、存在性公平、普适性公平）之间的关系；④给出了判断公平性的多重复杂度上界（从co‑NP到Σ₃^P、Π₃^P等）。

**🔧 技术方法**

采用形式化逻辑与可解释 AI 的理论工具： abductive explanations、prime‑implicant、逻辑约束、SAT/QBF 编码；通过理论证明与归约分析得出复杂度结果。

**📊 数据集**

论文为理论研究，未使用具体数据集，而是以通用的 Boolean 约束与分类器函数为例来说明概念与结果。

**📈 对比分析**

方法与已有公平性衡量（如FTU）进行对比：证明了在无保护与非受保护特征间无约束时三者等价；在存在约束时则各不相同；通过复杂度对比显示，判断存在性公平、普适性公平与FTU 的难度从co‑NP（FTU）提升至Σ₃^P/Π₃^P。

**⚠️ 局限性**

局限性主要是：①判定公平性（尤其存在性、普适性公平）在理论上属于高阶复杂度（Σ₃^P/Π₃^P），实务上难以直接应用；②仅考虑已知的硬约束，未涵盖统计相关性；③在某些约束极强的场景下可能无法构造普适性公平的分类器。

---

## 302. Upward-Planar Drawings with Bounded Span

**arXiv ID:** 2605.00603 | [PDF](https://arxiv.org/pdf/2605.00603v1)

**作者:** Patrizio Angelini `[一作]` (John Cabot University), Johannes Zink `[通讯]` (TU Munich)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究有向图上层次化可上升平面绘图中边的跨度（span）问题，给出树、单源图以及受限嵌入图的组合与算法结果。

**💡 创新点**

提出跨度的上下界（尤其针对有向树），证明该问题在树和单源双连通图上为NP‑完整，给出多类图（plane st‑图、固定嵌入单源图、边界源数有限图）可线性/XP求解的算法，并展示针对顶点覆盖数和树深度+跨度的FPT算法。

**🔧 技术方法**

核心技术包括：组合构造与极大/极小树枝长度分析、固定嵌入下的流网络与可行循环判定、图分割与重构（桥、分离对、块树）、参数化简约（kernelization、组件等价性）、整数线性规划与流网络结合的FPT实现。

**📊 数据集**

本工作为理论性研究，无实验数据集；所有结果均通过严格的数学证明与算法分析得到。

**📈 对比分析**

方法与先前研究相比，在树类图上实现了线性时间的可上升平面层次绘图判定，并在受限嵌入情况下提供了多源图的XP算法；FPT结果实现了以顶点覆盖数为参数的有效求解，证明了该参数对问题的重要性。

**⚠️ 局限性**

限制主要包括：对一般无固定嵌入的有向图，问题仍为NP‑完整；对跨度参数化的FPT仅在顶点覆盖和树深度+跨度组合上可行；对于更一般的树或图类，跨度上界仍未能实现完全的子线性或仅以顶点数与路径长度的函数；算法实现复杂度高，依赖于图的分解与嵌入，实际应用需进一步简化。

---

## 303. Living Databases: A Unified Model for Continuous Schema Evolution, Versioning, and Transformations

**arXiv ID:** 2605.00676 | [PDF](https://arxiv.org/pdf/2605.00676v1)

**作者:** Amol Deshpande `[一作]` (University of Maryland), Amol Deshpande `[通讯]` (University of Maryland)

**通讯引用:** 8821 | [OpenAlex ID](https://openalex.org/A5113637223)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了统一的快照图抽象，将数据版本控制、模式演化、视图维护和连续变换整合在同一框架内，并实现了基于 Prolly‑Tree 的原型存储引擎

**💡 创新点**

创新点在于：
• 统一抽象支持分支、合并、同步、条件传播和定时/惰性更新；
• 引入可配置的同步语义（单向/双向）和可触发的警报；
• 使用虚拟 Chunk 实现零成本分支与模式演化；
• 设计自主调优引擎自动决定分块策略、列式分组和物化方式

**🔧 技术方法**

技术栈包括：
• Prolly‑Tree（概率内容分块）与可变分块策略；
• 列式存储与垂直分区（Attribute Splitting）；
• 内容可寻址存储（CAS）与 SHA‑256 哈希；
• 虚拟 Chunk 与惰性/即时物化；
• Git‑style 分支/合并与同步图结构

**📊 数据集**

数据集：
• 合成单表，5 列，5 万行；
• 进行 500 次提交（每次 200 条增删改），覆盖追加、局部更新、均匀更新、混合工作负载

**📈 对比分析**

实验对比：
• 分块策略：容量式（Capacity‑based）vs 内容式（Content‑based）
• 行式 vs 列式分组
• 结果显示：
  – 局部更新下内容式分块共享更多块，存储占用约 30%–35% 降低；
  – 均匀更新时容量式分块更优；
  – 列式分组可使存储减少 15%–35%；
  – 事务吞吐和查询性能未给出具体数值，但实验表明结构共享显著降低了存储成本

**⚠️ 局限性**

局限性：
• 仅在合成数据上进行小规模实验，缺乏真实业务场景验证；
• 列式存储、惰性物化与自动调优引擎尚未完整实现；
• 并发控制、事务隔离级别和多分支同步的性能评估不足；
• 只展示了存储占用和块数的差异，未完整评估查询延迟和吞吐量

---

## 304. DMDSC: A Dynamic-Margin Deep Simplex Classifier for Open-Set Recognition on Medical Image Datasets

**arXiv ID:** 2605.00675 | [PDF](https://arxiv.org/pdf/2605.00675v1)

**作者:** Vishal `[一作]` (Shiv Nadar Institute of Eminence Deemed to be University), Saurabh J. Shigwan `[通讯]` (Shiv Nadar Institute of Eminence Deemed to be University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种动态边距的深度简单x分类器 (DMDSC)，用于医学图像数据集的开放集识别。

**💡 创新点**

创新点在于引入了与类别频率成反比的自适应动态边距，使稀有病理的特征聚类更紧凑，提升开放集识别性能。

**🔧 技术方法**

结合神经崩塌（Neural Collapse）与简单x极化框架，使用固定的简单x顶点作为类别中心，并在损失中加入动态边距交叉、背景样本分离等三种损失；训练采用ResNet18/34 backbone 与 RMSprop。

**📊 数据集**

在 BloodMNIST、OCTMNIST、DermaMNIST、BreaKHis 以及 Augmented Skin Conditions 等医学图像基准上进行评估。

**📈 对比分析**

与 DSC、UCDSC、OMCL、ARPL+CS、DIAS 等 SOTA 方法对比，DMDSC 在 AUROC、OSCR、ACC 三项指标均优于或持平，尤其在极端类别不平衡时表现更稳健。

**⚠️ 局限性**

局限性在于仍依赖于预定义的简单x顶点与超球半径，对极端不平衡或跨模态数据的适应性尚未充分验证，且需要手动设定 m_min、m_max 等超参。

---

## 305. SENECA: Small-Sample Discrete Entropy Estimation via Self-Consistent Missing Mass

**arXiv ID:** 2605.00668 | [PDF](https://arxiv.org/pdf/2605.00668v1)

**作者:** Lucas H. McCabe `[一作]` (George Washington University), H. Howie Huang `[通讯]` (George Washington University)

**通讯引用:** 2329 | [OpenAlex ID](https://openalex.org/A5002254350)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于自洽缺失质量估计的离散熵估计方法 SENECA，专为小样本场景设计。

**💡 创新点**

创新点在于将缺失质量的自洽固定点迭代与样本覆盖率缩减相结合，显著提升熵估计精度。

**🔧 技术方法**

技术方法包括自洽缺失质量计算、固定点迭代、Chao‑Shen 形式的覆盖率修正以及正则化权重 Chebyshev 支持估计。

**📊 数据集**

实验使用了 72 种模拟分布、58 个真实生物多样性数据集以及 5 个大型语言模型和 6 个问答数据集。

**📈 对比分析**

与七种基线（插件、Grassberger、James‑Stein、Bonachela、Chao‑Shen、Chao‑Wang‑Jost、Valiant‑Valiant）比较，SENECA 在小样本/欠抽样下 RMSE 最低，并在生物多样性和 LLM 错误检测任务中排名靠前。

**⚠️ 局限性**

局限性包括缺乏无偏性，在已知良好采样或大样本情况下表现不如专用方法，且理论保证尚未完全完善。

---

## 306. InpaintSLat: Inpainting Structured 3D Latents via Initial Noise Optimization

**arXiv ID:** 2605.00664 | [PDF](https://arxiv.org/pdf/2605.00664v1)

**作者:** Jaeyoung Chung `[一作]` (Seoul National University), Kyoung Mu Lee `[通讯]` (Seoul National University)

**通讯引用:** 27274 | [OpenAlex ID](https://openalex.org/A5046504049)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种训练无关的3D修补方法，通过在预训练的结构化3D潜在扩散模型中优化初始噪声种子来实现几何一致性与文本条件的对齐；

**💡 创新点**

创新点包括：①把初始噪声优化作为独立控制维度，②利用近似反向传播与rectified flow实现高效梯度传递，③采用频域FFT参数化稀疏结构以提升优化稳定性，④引入高阶统计正则化保持噪声分布逼近标准高斯，整体可与传统采样轨迹控制互补；

**🔧 技术方法**

技术手段包括：Treillis结构化潜在扩散框架、rectified flow、近似反向传播、FFT频谱参数化、Gaussian分布正则化、Adam优化器、CLIP、DINOv2等评估指标；

**📊 数据集**

使用了Toys4K、ABO、HSSD等数据集，主要在Toys4K测试集上进行定量评估；

**📈 对比分析**

对比RePaint、SDEdit、MultiDiffusion、Blended Diffusion、DPS、ILVR等基线，在几何指标（Chamfer、F-score、normal PSNR/SSIM/LPIPS）和语义指标（CLIP score、FD、KD）上均优于基线，显示了更好的保留与生成一致性；

**⚠️ 局限性**

局限性：需要多步迭代优化初始噪声，导致运行时间较长；且方法高度依赖预训练模型的先验，若保持区域与文本条件高度不一致或在训练数据中缺失，可能无法生成合理补全。

---

## 307. Affordance Agent Harness: Verification-Gated Skill Orchestration

**arXiv ID:** 2605.00663 | [PDF](https://arxiv.org/pdf/2605.00663v1)

**作者:** Haojian Huang `[一作]` (HKUST(GZ)), Yingcong Chen `[通讯]` (HKUST(GZ))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Affordance Agent Harness，一个闭环、预算感知的技能组合框架，用于开放世界场景下的可操作性定位。

**💡 创新点**

创新点在于自适应路由、相对检验门控（一致性、尺度稳定性、证据充分性）以及两层记忆（通用知识库与在线经验库）来实现基于证据的决策和经验复用。

**🔧 技术方法**

采用多技能调用（检测、分割、放大、网络搜索、交互想象）+证据存储 + Verifier 验证器 + Router 路由器 + 两层记忆 + LLM 决策脑。

**📊 数据集**

实验使用 ReasonAff、UMD Part Affordance 以及 RAGNet（3DOI 与 HANDAL 子集）等公开数据集。

**📈 对比分析**

与固定管线、现有基础模型（如 LISA、SAN、SAM、各类 LLM）做对比，在 gIoU、cIoU、P_50 等指标上实现更优的准确率–成本 Pareto 前沿，同时显著降低技能调用次数和延迟。

**⚠️ 局限性**

局限性包括：对多模态工具的可用性与成本敏感、验证阈值和路由策略的调参依赖、对极端遮挡或极小目标仍易失效、以及记忆容量与更新策略的限制。

---

## 308. AdaMeZO: Adam-style Zeroth-Order Optimizer for LLM Fine-tuning Without Maintaining the Moments

**arXiv ID:** 2605.00650 | [PDF](https://arxiv.org/pdf/2605.00650v1)

**作者:** Zhijie Cai `[一作]` (Shenzhen Research Institute of Big Data), Guangxu Zhu `[通讯]` (Shenzhen Research Institute of Big Data)

**通讯引用:** 5899 | [OpenAlex ID](https://openalex.org/A5004583148)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 AdaMeZO，一种无梯度、零阶优化器，兼具 Adam 风格的动量估计而无需额外存储。

**💡 创新点**

创新点是利用截断的动量估计与 PRNG 状态缓存，在保持低内存的同时实现 Adam 风格的预处理。

**🔧 技术方法**

使用零阶梯度估计（SPSA）与 Adam‑style 迭代、PRNG 状态缓存、块级随机方向生成。

**📊 数据集**

在 RoBERTa‑large、OPT‑1.3B、LLaMA‑3B 等大语言模型上，结合 SST‑2、RTE、CB、SQuAD 等基准任务。

**📈 对比分析**

与 MeZO、MeZO‑switch、HiZOO 等对比，AdaMeZO 在保持相同终止条件下前向传递次数减少约70%，并在多项任务上提升1–3个百分点。

**⚠️ 局限性**

局限在于截断动量估计可能产生偏差、第二阶矩估计不够精准以及相对较高的计算开销。

---

## 309. Bridging Graph Drawing and Dimensionality Reduction with Stochastic Stress Optimization

**arXiv ID:** 2605.00641 | [PDF](https://arxiv.org/pdf/2605.00641v1)

**作者:** Daniel Hangan `[一作]` (Technical University Of Munich), Jacob Miller `[通讯]` (Technical University Of Munich)

**通讯引用:** 116 | [OpenAlex ID](https://openalex.org/A5101504842)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一种基于随机梯度下降（SGD）的多维尺度（MDS）求解器，将图绘制领域的压力优化方法迁移到向量数据的降维任务中，并提供了可兼容 scikit‑learn 的估计器。

**💡 创新点**

创新点在于：①将图绘制中的 SGD 压力优化直接应用于 MDS，②引入 “Lazy” 模式实现 O(1) 辅助内存，③改进学习率调度与梯度裁剪，④在实现上兼容 scikit‑learn，提升了可用性和可扩展性。

**🔧 技术方法**

使用了随机梯度下降、对偶梯度裁剪、随机洗牌、混合指数-谐波学习率衰减、Cython 计算核心、按需计算距离的 Lazy 模式等技术。

**📊 数据集**

在 Espadoto 等人整理的 18 个标准基准数据集（包括 synthetic、coil20、IMDB 等）上进行实验。

**📈 对比分析**

通过与传统 SMACOF 进行并行实验，测量收敛时间（epoch 数）和最终压力值；SGD‑MDS 在 15~30 个 epoch 内收敛，收敛速度约为 SMACOF 的 4~10 倍，在大多数数据集上获得更低或相当的压力。

**⚠️ 局限性**

主要限制包括：缺乏理论收敛保证；实现是顺序处理对偶，无法利用矩阵并行运算；对极大规模数据仍受限；在高范数、噪声较大的数据集上偶尔收敛不如 SMACOF。

---

## 310. Spiking Sequence Machines and Transformers

**arXiv ID:** 2605.00662 | [PDF](https://arxiv.org/pdf/2605.00662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 311. Recovering Hidden Reward in Diffusion-Based Policies

**arXiv ID:** 2605.00623 | [PDF](https://arxiv.org/pdf/2605.00623v1)

**作者:** Yanbiao Ji `[一作]` (Shanghai Jiao Tong University), Hongtao Lu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8364 | [OpenAlex ID](https://openalex.org/A5102899381)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EnergyFlow 框架，将扩散模型的标量能量函数与逆强化学习统一，既可生成动作又可直接提取奖励。

**💡 创新点**

创新点在于：①用能量函数强制梯度场为守恒场，理论上可恢复软 Q 函数梯度；②通过去噪分数匹配避免对抗训练，减少训练不稳定；③引入中心化奖励消除状态偏置，提升下游 RL 性能；④证明保守场能显著降低泛化误差。

**🔧 技术方法**

技术包括：扩散生成模型、去噪分数匹配、能量基模型、守恒梯度约束、中心化奖励、概率流 ODE、基于 Spectral Normalization 的网络正则化。

**📊 数据集**

使用机器人操纵基准数据集 RoboMimic（5 个任务）和 Meta-World（5 个任务），并在真实 AGIBOT G1 机器人上进行 Bottle 与 Drawer 两个接触丰富任务的验证。

**📈 对比分析**

与 LSTM-GMM、Diffusion Policy、Flow Policy、EBT-Policy、EBIL、NEAR、IQ-Learn 等基线比较，EnergyFlow 在所有任务上平均成功率均超过 90%，在高难度任务 ToolHang、DrawerOpen 甚至达到 100%，同时在 OOD 位置扰动下保持更高稳健性；在 RL 上，中心化能量奖励可与稀疏奖励结合，取得接近 oracle 的成功率。

**⚠️ 局限性**

局限性包括：仍需大量专家演示；奖励仅在同一状态内可唯一确定，跨状态比较受未知状态偏置影响；训练时对时间步参数 γ 的选择较为敏感；对极大噪声水平下的分数估计不稳，可能导致收敛缓慢。

---

## 312. Beyond Decodability: Reconstructing Language Model Representations with an Encoding Probe

**arXiv ID:** 2605.00607 | [PDF](https://arxiv.org/pdf/2605.00607v1)

**作者:** Gaofei Shen `[一作]` (Tilburg University), Grzegorz Chrupała `[通讯]` (Tilburg University)

**通讯引用:** 1510 | [OpenAlex ID](https://openalex.org/A5022698890)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种 Encoding Probe，利用多元回归对 Transformer 模型的内部表示进行重构，以比较不同可解释特征（声学、音位、说话人、句法、词汇等）的贡献并控制特征相关性。

**💡 创新点**

创新点在于将传统的解码 Probe 方向反转为编码 Probe，既能直接量化各特征对表示的相对贡献，又能在同一回归框架下同时消除多重相关特征的影响，从而克服了解码 Probe 的可比性和相关性干扰问题。

**🔧 技术方法**

使用的技术包括：多元岭回归、特征消融分析、帧级或词级特征提取（声学特征、音位后验向量、说话人标签、句法标签、词向量）、以及与解码 Probe 的对比实验。

**📊 数据集**

采用的数据集主要是 LibriSpeech（音频与说话人标签）用于 wav2vec2 与 BERT 的预训练与微调模型，BERT 基础模型使用公开的大规模文本语料，特征从这些数据中提取。

**📈 对比分析**

比较方法：对完整特征集（Full）与各特征或特征组合被消除的子集进行回归，计算未解释方差 UV 的差异以量化贡献；与解码 Probe 的分类/回归准确率对比，发现解码与编码结果可能差异显著，说明编码 Probe 能更真实反映特征对表示的影响。

**⚠️ 局限性**

限制：仍为观察性方法，无法确定因果关系；使用线性岭回归可能忽略非线性关系；特征集合有限，未包含所有可能影响表示的因素；帧级分析可能低估长距离结构信息；实验仅基于读书音频，结果在自发语或噪声环境下可能不适用。

---

## 313. Augmented Lagrangian Multiplier Network for State-wise Safety in Reinforcement Learning

**arXiv ID:** 2605.00667 | [PDF](https://arxiv.org/pdf/2605.00667v1)

**作者:** Jiaming Zhang `[一作]` (Tsinghua University), Liping Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 74717 | [OpenAlex ID](https://openalex.org/A5100425554)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ALaM 框架，用于稳定学习状态级拉格朗日乘子，并将其与 Soft Actor-Critic 结合，形成 SAC-ALaM 算法。

**💡 创新点**

创新点在于：①引入增广拉格朗日的二次罚项以补偿乘子更新延迟并建立局部凸性；②将乘子网络的更新改为对理论双重目标的监督回归，避免传统梯度上升导致的振荡。

**🔧 技术方法**

使用技术包括增广拉格朗日方法、监督回归的乘子网络、Soft Actor-Critic（SAC）框架、两尺度学习、Monte‑Carlo 估计、两阶段目标网络、温度自适应。

**📊 数据集**

实验数据集为 Safety‑Gymnasium 基准，涵盖 Point、Car、Swimmer、HalfCheetah、Ant 等多种环境及不同任务（Velocity、Goal、Circle、Button）和安全难度。

**📈 对比分析**

与 SAC-Lag、SAC-LagNet、SAC-PID、ASAC、SAC-FPI、P3O、CRPO 等基线对比，SAC-ALaM 在奖励、成本和训练稳定性上均优于所有对手，表现出更高的安全性与性能。

**⚠️ 局限性**

局限性包括：对超参数（如 ρ、σ、学习率）较为敏感；计算成本相对较高；在极端安全约束或更大规模任务中的泛化能力仍需进一步验证。

---

## 314. Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding

**arXiv ID:** 2605.00642 | [PDF](https://arxiv.org/pdf/2605.00642v1)

**作者:** Yan Zhang `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Can Ma `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GUI-SD框架，实现了在GUI定位任务中基于视觉信息的自监督对齐与熵引导的分层优化；

**💡 创新点**

创新点在于将视觉富化的特权上下文与熵引导的加权逆KL损失结合，解决了传统OPSD在坐标预测中的分布崩溃与无差异优化问题；

**🔧 技术方法**

采用视觉特权上下文（红色边框+高斯软掩模+提示）和熵引导的加权逆KL自蒸馏技术；

**📊 数据集**

使用七千条ScaleCUA GUI数据与六大基准（ScreenSpot‑v2、ScreenSpot‑Pro、UI‑Vision、MMBench‑GUI、OSWorld‑G、OSWorld‑G‑Refine）进行训练与评估；

**📈 对比分析**

相较于GRPO（Binary、Distance、Gaussian）和Naive OPSD，在所有六个基准上平均提升了约4–6%准确率，并将训练效率提高约4倍；

**⚠️ 局限性**

局限在于仅在Qwen3‑VL‑Instruct‑8B模型上验证，未探究更大模型规模或多步交互场景的适用性。

---

## 315. Knowing when to trust machine-learned interatomic potentials

**arXiv ID:** 2605.00640 | [PDF](https://arxiv.org/pdf/2605.00640v1)

**作者:** Shams Mehdi `[一作]` (Carnegie Mellon University), Olexandr Isayev `[通讯]` (Carnegie Mellon University)

**通讯引用:** 18458 | [OpenAlex ID](https://openalex.org/A5011932992)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种后置式不确定性量化框架：在已训练的机器学习原子势（MLIP）上冻结其原子表示，并训练一个轻量级分类器来判断每个分子能量预测是否可靠，从而给出可靠性概率。

**💡 创新点**

创新点在于：①不再通过众包或回归估计误差，而是将可靠性判定视为二分类问题；②利用MLIP内部的原子表示直接做判定，避免额外的模型训练或推理成本；③使用多头自注意力生成分子嵌入，并通过注意力权重得到原子重要性，提供化学可解释性；④方法在不同架构（AIMNet2、MACE）上通用，且对规模更大的基础模型具有良好可扩展性。

**🔧 技术方法**

技术包括：冻结MLIP（AIMNet2或MACE）产生原子向量；三层MLP编码器映射至256维；多头自注意力（32头，每头8维）聚合全局信息；平均/最大池化+能量+原子数形成514维分子特征；线性投影得到256维嵌入；二分类MLP输出可靠性概率；训练采用大小归一化交叉熵和类别权重；注意力矩阵用于计算原子重要性。

**📊 数据集**

使用的数据集：AIMNet2的20M DFT（B97‑3c）训练集，测试集3.76M分子；MACE‑OFF23的855k SPICE 训练集，测试集50k分子；另外在Reidenbach 2.4M生物相关分子上进行覆盖分析。

**📈 对比分析**

与传统的4模型深度集成（标准差）进行比较：在AIMNet2测试集上，方法整体准确率71.6%（对比60%），高置信度（P≥0.9）准确率93.2%（对比仅能得到置信区间）。在MACE上也表现出更高的准确率与更好的校准（ECE=0.011）。方法在不同阈值下提供可调的覆盖-准确率折衷，且仅增加不到1%的前向推理成本。

**⚠️ 局限性**

局限性：①仅提供二分类可靠性，未给出连续误差估计；②对完全缺失的化学空间仍易失效；③训练需要参考能量标签，难以直接应用于无标注的基础模型；④目前仅评估能量误差，对力场误差等未验证；⑤可靠性阈值需根据后备模型的误差分布重新校准。

---

## 316. EGREFINE: An Execution-Grounded Optimization Framework for Text-to-SQL Schema Refinement

**arXiv ID:** 2605.00628 | [PDF](https://arxiv.org/pdf/2605.00628v1)

**作者:** Jiaqian Wang `[一作]` (Xidian University), Rui Yang `[通讯]` (Xidian University)

**通讯引用:** 43424 | [OpenAlex ID](https://openalex.org/A5100378741)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对Text-to-SQL任务，提出了一种基于执行结果的 schema 细化框架 EGRefine，通过四个阶段（LLM 过滤、候选生成、执行验证、视图化）将列名改写为更易理解的形式，从而提升模型的执行准确率。

**💡 创新点**

创新点在于：① 将 schema 细化形式化为约束优化问题，以执行准确率为目标函数；② 引入基于执行反馈的验证步骤，代替传统的语言学合理性判定；③ 通过视图化实现细化后的 schema 与原始数据库等价，保证非破坏性；④ 采用列级局部非退化与全局视图等两层结构性保障。

**🔧 技术方法**

核心技术包括：大语言模型（LLM）进行列名歧义检测与候选命名生成；多模型集成（C3、DIN-SQL 等）在执行验证阶段计算候选的执行准确率；基于数据库视图的等价迁移；以及对列级别的贪婪分解与保守选择策略。

**📊 数据集**

使用的数据集包括：Dr.Spider（控制的列名衰减版本），BIRD（真实学术数据库），以及 BEAVER（企业级数据仓库）。

**📈 对比分析**

与未细化、仅 LLM 直接细化以及仅注释描述等基线对比，EGRefine 在 Dr.Spider 上恢复了 67% 的性能损失，在 BIRD 上实现了 0.2–1.2pp 的正增益，并在多模型间实现“一次细化多模型使用”的转移效果；实验显示执行验证是关键环节，去除该步骤会导致大幅退化。

**⚠️ 局限性**

限制包括：① 对低基线性能（如 BEAVER）细化效果有限；② 需要一套包含标注 SQL 的查询工作负载；③ 细化结果与特定工作负载高度相关，迁移到未见查询时可能受限；④ 对极大规模 schema 的搜索空间仍有挑战，需更高效的筛选策略。

---

## 317. Defense against Poisoning Attacks under Shuffle-DP

**arXiv ID:** 2605.00625 | [PDF](https://arxiv.org/pdf/2605.00625v1)

**作者:** Siyi Wang `[一作]` (Nanyang Technological University), Wei Dong `[通讯]` (Nanyang Technological University)

**通讯引用:** 51700 | [OpenAlex ID](https://openalex.org/A5100641142)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种通用的防御框架，旨在保护差分隐私（DP）模型下的联合保留查询免受中毒攻击的影响。

**💡 创新点**

该框架是首个能够将任何易受中毒攻击的shuffle-DP协议转化为防御版本的框架，具有较高的实用性和通信效率。

**🔧 技术方法**

使用了shuffle-DP模型，并提出了分层shuffle-DP（HSDP）协议和优化的HSDP（OHSDP）协议。

**📊 数据集**

在多个数据集上进行了实验，包括真实世界数据集（如成人数据集、薪资数据集）和合成数据集（如均匀分布、Zipf分布和高斯分布）。

**📈 对比分析**

与现有的最先进（SOTA）协议进行了比较，结果表明在存在攻击的情况下，提出的框架能够有效检测和缓解攻击，保持高效的实用性和通信效率。

**⚠️ 局限性**

该框架在处理多用户中毒攻击时的性能仍有待提高，且通信开销相较于基础协议仍然较大。

---

## 318. SC-Taxo: Hierarchical Taxonomy Generation under Semantic Consistency Constraints using Large Language Models

**arXiv ID:** 2605.00620 | [PDF](https://arxiv.org/pdf/2605.00620v1)

**作者:** Shiqiang Cai `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jun Zhao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 66268 | [OpenAlex ID](https://openalex.org/A5030154270)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SC-Taxo 框架，用双路径初始化和四轮深度融合实现科学文献的层次化分类。

**💡 创新点**

创新点在于结合结构化聚类与 LLM 生成的双向标题一致性，并加入同级语义关联，显著抑制语义漂移与结构失真。

**🔧 技术方法**

使用技术包括：UMAP+BFS 层次聚类、句子嵌入、四轮互验证（概念→聚类、聚类→概念、父子对齐、同级扩展）以及 LLM 驱动的质量评分与冗余检测。

**📊 数据集**

使用数据集：英语 TaxoBench benchmark 以及中文聚焦 LLM 与 RL 交叉领域的 59 篇论文集。

**📈 对比分析**

对比纯 LLM、聚类结合及 LLM 多维聚类基线，SC-Taxo 在英语 Benchmark 的 CEDS 28.7、HSR 75.3、Purity 64.8 领先；在中文集 CEDS 29.8、HSR 51.5、Purity 66.7 也处于同类方法之首。

**⚠️ 局限性**

主要局限是多阶段 LLM 调用导致计算开销较大，未来需探索轻量化一致性估计与自适应精炼策略。

---

## 319. LLM-Emu: Native Runtime Emulation of LLM Inference via Profile-Driven Sampling

**arXiv ID:** 2605.00616 | [PDF](https://arxiv.org/pdf/2605.00616v1)

**作者:** Wei Da `[一作]` (University of Cambridge), Evangelia Kalyvianaki `[通讯]` (University of Cambridge)

**通讯引用:** 1133 | [OpenAlex ID](https://openalex.org/A5051440292)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于 vLLM 的在线服务原生仿真器，替换 GPU 推理路径为基于离线统计的采样延迟与合成输出，保持其余 HTTP、调度、KV 缓存等生产路径不变。

**💡 创新点**

创新点在于：①仅在执行边界实现仿真，避免了重新实现调度器或对 CUDA 的拦截；②提出密度感知的二维延迟光谱（按总 token 数与并发数划分）以及基于 Timer 的 Future 以保留异步重叠；③通过离线大规模 profiling 捕获真实延迟分布，支持多 GPU、模型规模、硬件与工作负载的快速校准。

**🔧 技术方法**

技术包括：vLLM 0.18.1 插件式运行时 Hook、离线两维延迟光谱存储（JSON）、密度感知最近邻采样（Shepard 投影）、Timer 解析的 Future、FlashAttention/FlashInfer GPU 后端、ShareGPT 生成式问答工作负载、Poisson 与 γ‑分布突发到达模拟。

**📊 数据集**

使用的数据集为官方 ShareGPT 生成式问答数据集（2000 条提示）以及基于该数据集的多速率（2,4,8,16,32 请求/秒）和 γ=0.25 的突发到达模式。

**📈 对比分析**

比较方法：在两块 48GB GPU（RTX 8000/FlashInfer 与 A40/FlashAttention 2）上对照真实 vLLM 推理与仿真运行，测量 TTFT、TPOT、ITL、E2E 延迟和吞吐量。结果显示：除 TTFT 外，TPOT、ITL 误差 ≤4.8%，E2E ≤5.3%，吞吐量 ≤1.9%，TTFT 最大误差 10.4%。

**⚠️ 局限性**

局限性包括：仅验证单节点部署；离线 profiling 仍需数小时 GPU 时间；未覆盖多 GPU/多节点集群、离线批量推理、以及 vLLM API 漂移导致的自动化重校验需求。

---

## 320. Decouple before Integration: Test-time Synthesis of SFT and RLVR Task Vectors

**arXiv ID:** 2605.00610 | [PDF](https://arxiv.org/pdf/2605.00610v1)

**作者:** Chaohao Yuan `[一作]` (Chinese University of Hong Kong), Long-Kai Huang `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 764 | [OpenAlex ID](https://openalex.org/A5003953967)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在推理阶段合成监督微调（SFT）与强化学习可验证奖励（RLVR）后训练模型的框架（DoTS），使得两种后训练方案可独立完成后训练并通过任务向量算术在测试时合并

**💡 创新点**

通过任务向量分析发现SFT与RLVR存在30倍幅度差异、45%符号冲突以及模块分布差异，进而设计了选择性稀疏化+范数重标、难度感知样本选择与贝叶斯优化系数搜索的三步合成流程

**🔧 技术方法**

任务向量算术、稀疏化与范数重标、难度感知数据挑选、贝叶斯优化（TPE）

**📊 数据集**

数学推理基准（AIME 2024/25、AMC、MATH500、Minerva、OlympiadBench）以及跨域QA基准（ARC‑C、GPQA、MMLU‑Pro）

**📈 对比分析**

与多种基线（原始SFT、RLVR、Sequential+Joint训练、TIES‑Merging、DARE、LUFFY、ReLIFT 等）比较，DoTS 在数学推理任务上平均分达 49.3 分，超过或与训练基线持平，且计算成本仅为 3%，在跨域 QA 上亦能无额外调参直接提升 60.5 分

**⚠️ 局限性**

依赖源模型质量与互补性；当某一源模型贡献有限或底层模型较弱时合成效果受限，且当前仅采用全模型级系数，未探索更细粒度（层/模块）组合策略

---

## 321. Affinity Is Not Enough: Recovering the Free Energy Principle in Mixture-of-Experts

**arXiv ID:** 2605.00604 | [PDF](https://arxiv.org/pdf/2605.00604v1)

**作者:** Man Yung Wong `[一作]` `[通讯]`, Man Yung Wong

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了三种基于自由能原理的 MoE 路由改进机制——时间记忆 β、精度加权 Π 和前瞻路由，显著提升在领域转换点的专家选择准确率。

**💡 创新点**

创新点在于将 LIF 神经元的衰减记忆、预测误差精度自适应与未来状态预测三者融合为一套可解释的、轻量级路由门结构，并证明它们在关键时刻表现出超加性互补。

**🔧 技术方法**

使用的技术包括 Leaky‑Integrate‑and‑Fire（LIF）递归状态、在线均值递推的精度估计以及基于 β 隐藏状态的 MLP 前瞻预测；实现代码仅约 200 行。

**📊 数据集**

实验数据集包括：①自定义领域切换序列（4 个专家、8 长度）、②噪声级别切换回归任务、③字符级 26 词表的 64 长度域切换语言模型。

**📈 对比分析**

与标准 MoE、仅 β、仅 Π、仅 Ant 等基线相比，β+Ant 在转移步的正确专家概率从 0.006 提升至 0.748，减少了所需顶层专家数目（K）从约 765 降至 4；在字符 LM 中，转移步 BPC 从 6.56 降至 4.01，路由准确率提升至 0.60/0.86。

**⚠️ 局限性**

主要局限包括：实验规模相对较小，未验证在大模型和自然语言中平滑漂移场景下的效果；Π 在均衡可靠性任务中效果有限；前瞻路由的精度依赖 β 的衰减率，需进一步自适应；尚未探究与其他预取、预测建模方法的协同提升。

---

## 322. Unlearning Offline Stochastic Multi-Armed Bandits

**arXiv ID:** 2605.00638 | [PDF](https://arxiv.org/pdf/2605.00638v1)

**作者:** Zichun Ye `[一作]` (Shanghai Jiao Tong University), Mohammad Hajiesmaili `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 1090 | [OpenAlex ID](https://openalex.org/A5046146251)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了离线随机多臂赌博机（MAB）的(ε,δ)-无学习（unlearning）框架，并设计了自适应的单源无学习算法，分别在固定样本模型和分布模型下给出上界与下界。

**💡 创新点**

创新点在于：①首次将差分隐私启发的无学习概念迁移到决策制定任务；②提出两种极端无学习基元（高斯机制与回滚）并给出自适应切换策略；③给出了多源无学习的上界并提出混合算法与针对C*∈(1,2)的改进学习–无学习对。

**🔧 技术方法**

使用的技术包括：基于极大似然的低置信下界（LCB）学习；高斯机制、回滚和混合算法的噪声调度；对数概率与Le Cam方法的下界分析；以及对覆盖系数C*的分布式采样处理。

**📊 数据集**

实验数据集为合成的离线MAB数据，固定样本模型采用5臂轮询策略，分布模型采用概率为1/C*的最佳臂与均匀分布的次优臂，C*取值5和1.3。

**📈 对比分析**

与Oracle、纯高斯机制、回滚和混合算法等基线比较，实验表明自适应算法在大多数参数设置下都能显著降低次优性（提升70%–90%），并在不同k、N、γ的变化下保持鲁棒性。

**⚠️ 局限性**

局限性：上界与下界在某些参数范围内仍存在对数因子差距；多源无学习仅在固定样本模型下得到理论保证，分布模型尚未覆盖；未扩展到情境/线性赌博机或更广泛的离线强化学习场景。

---

## 323. Class Angular Distortion Index for Dimensionality Reduction

**arXiv ID:** 2605.00637 | [PDF](https://arxiv.org/pdf/2605.00637v1)

**作者:** Kaviru Gunaratne `[一作]` (Technical University Of Munich), Jacob Miller `[通讯]` (Technical University Of Munich)

**通讯引用:** 116 | [OpenAlex ID](https://openalex.org/A5101504842)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于三点内部角度的聚类结构保持度量CADI，并基于该度量开发新的降维方法AngleEmbedding。

**💡 创新点**

创新点在于：① 用内部角度而非距离或质心来衡量跨类别几何关系；② 该度量对非球形、嵌套、环状等复杂结构保持敏感；③ 通过CADI驱动的可微降维算法提供全新的可解释投影。

**🔧 技术方法**

技术包括：三点角度计算、随机采样加速、基于梯度下降的可微神经网络降维；实验使用PCA、t‑SNE、UMAP、MDS、PaCMAP、UMATO等常用算法。

**📊 数据集**

数据集涵盖合成（环形、同心球、马特里奥什卡、甜甜圈）和真实（MNIST、USPS、Liver、TREC、Emotion、Sentiment等）多域高维数据。

**📈 对比分析**

通过与Silhouette、DBI、NMI、ARI、Label‑T&C、Steadiness & Cohesiveness、Cluster Distance Score等指标的Spearman相关和排名比较，CADI能更准确地区分保持真实聚类关系的投影；AngleEmbedding在合成数据上能恢复嵌套结构，且在真实数据中得到更合理的类间关系，优于传统局部方法。

**⚠️ 局限性**

局限性包括：① 仍需预先给定或估计聚类标签，标签选择会影响结果；② 采样方法对大型类别可能偏倚；③ 对非欧氏空间的适用性尚未验证；④ 与其他聚类度量的理论关系尚不充分；⑤ AngleEmbedding作为监督方法与无监督算法直接对比不公平。

---

## 324. Efficient Incremental #SAT via Cross-Instance Knowledge Reuse

**arXiv ID:** 2605.00671 | [PDF](https://arxiv.org/pdf/2605.00671v1)

**作者:** Uriya Bartal `[一作]` (Open University Of Israel), Jean-Marie Lagniez `[通讯]` (CRIL, University of Artois and CNRS)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种增量模型计数框架，利用持久化缓存与分支启发式共享以处理语法相似的公式序列。

**💡 创新点**

核心创新在于对组件缓存的持久化、懒惰对称性缓存与一次性树分解重用，从而在增量设置下实现跨实例的知识复用。

**🔧 技术方法**

实现基于d4的DPLL计数器，采用显式缓存、基于频率的规范化编码、DLCS分支启发式以及可选的树分解共享。

**📊 数据集**

在动态论证框架与软核心最小化两类真实与合成基准上进行实验，包含264个AF实例和3040个软核心实例。

**📈 对比分析**

与现有的d4、OptiMathSAT等工具比较，使用共享+对称缓存+DLCS模式在绝大多数基准上实现数百%至数千%速度提升，并在软核心任务中优于PBCount2与ganak。

**⚠️ 局限性**

在复杂实例中缓存和对称性收益下降，内存占用增大；树分解重用在极度动态变化的公式中效果有限。

---

## 325. Reinforcement Learning with Markov Risk Measures and Multipattern Risk Approximation

**arXiv ID:** 2605.00654 | [PDF](https://arxiv.org/pdf/2605.00654v1)

**作者:** Andrzej Ruszczynski `[一作]`, Tiangang Zhang `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种风险厌恶的有限时域马尔可夫决策问题的Q学习方法，结合了迷你批量风险度量和多模式风险厌恶问题的概念。

**💡 创新点**

创新点在于引入了迷你批量风险度量和多模式风险厌恶马尔可夫决策过程的理论框架，并在此基础上提出了一种高概率的后悔界限。

**🔧 技术方法**

使用了基于特征的Q学习方法和线性函数逼近技术。

**📊 数据集**

在随机分配问题和短时域多臂赌博机问题上进行了理论结果的验证。

**📈 对比分析**

与传统的风险中性Q学习方法相比，提出的方法在处理大状态空间和复杂风险度量时表现出更好的性能，后悔界限为𝒪(H^2 N^H √(K))，其中H为时域，N为迷你批量大小，K为回合数。

**⚠️ 局限性**

限制在于该方法在处理非常大的状态和动作空间时仍然面临挑战，且对风险度量的统计估计可能较为复杂。

---

## 326. PEACE: Cross-modal Enhanced Pediatric-Adult ECG Alignment for Robust Pediatric Diagnosis

**arXiv ID:** 2605.00647 | [PDF](https://arxiv.org/pdf/2605.00647v1)

**作者:** Xinran Liu `[一作]` (Southeast University), Chengyu Liu `[通讯]` (Southeast University)

**通讯引用:** 13033 | [OpenAlex ID](https://openalex.org/A5100321186)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文研究了如何将成人ECG模型迁移到儿童群体，提出了PEACE框架实现跨群体的临床语义增强对齐。

**💡 创新点**

创新点在于将三轴临床语义分解、标签查询网络以及课程门控对齐相结合，形成结构化语义监督与标记特定对齐的完整体系。

**🔧 技术方法**

主要技术包括ResNet1D ECG编码器、BioClinicalBERT文本编码器、Label-Query网络、Label-Specific Bidirectional Contrastive Learning（LSBC）和Curriculum-Adaptive Fusion（CAF）等。

**📊 数据集**

使用的数据集为预训练的MIMIC-IV成人ECG，评估数据为ZZU-pECG儿童ECG和外部PTB-XL。

**📈 对比分析**

与ST-MEM、MERL、KED等基线比较，在ZZU-pECG上，零样本AUC为59.4%，50-shot为79.0%，全微调为90.9%；在PTB-XL全微调AUC达96.65%。

**⚠️ 局限性**

局限性包括使用Gemini生成的语义描述缺乏年龄适配，未显式建模发育阶段，缺乏临床前瞻验证，且模型仅在中文儿童样本上评估，可能存在迁移偏差。

---

## 327. KingsGuard: Enclave Data Protection Under Real-World TEE Vulnerabilities

**arXiv ID:** 2605.00613 | [PDF](https://arxiv.org/pdf/2605.00613v1)

**作者:** Saltanat Firdous Allaqband `[一作]` (Indian Institute of Technology Madras), Chester Rebeiro `[通讯]` (Indian Institute of Technology Madras)

**通讯引用:** 1151 | [OpenAlex ID](https://openalex.org/A5038432102)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的TEE框架，利用硬件层的细粒度信息流跟踪和可控去标识机制，防止 enclave 中因软件漏洞或硬件共享资源导致的敏感数据泄漏。

**💡 创新点**

创新点在于将信息流跟踪完全迁移到硬件实现，支持对数据、地址及共享寄存器的实时 taint 传播，并通过预先计算的授权 declassification 路径实现合法数据流的精准控制。

**🔧 技术方法**

技术实现包括：shadow memory、扩展的寄存器文件、硬件哈希引擎用于运行时路径校验、所有数据块 1bit taint、共享寄存器的 EID 标记以及在 RISC‑V 处理器中的安全监控单元。

**📊 数据集**

实验使用 Embench、MiBench 基准套件以及 SCADA 变压器部分放电检测应用的数据集，验证在真实工作负载下的效果。

**📈 对比分析**

与传统 TEE（如 SGX）及现有软件/硬件防护方法相比，KingsGuard 在 10.8% 逻辑单元占用和 5.69% 性能开销下，能够完整阻止 AV1‑AV4 四类泄漏攻击，且兼容 SassCache 等缓存侧信道防御。

**⚠️ 局限性**

主要局限在于不处理时序侧信道攻击、物理攻击和 DoS 攻击，且需要在 enclave 二进制中插入少量注解，无法完全消除所有硬件/软件缺陷。

---

## 328. MUDY: Multi-Granular Dynamic Candidate Contextualization for Unsupervised Keyphrase Extraction

**arXiv ID:** 2605.00597 | [PDF](https://arxiv.org/pdf/2605.00597v1)

**作者:** Hyeongu Kang `[一作]` (Korea University), Susik Yoon `[通讯]` (Korea University)

**通讯引用:** 341 | [OpenAlex ID](https://openalex.org/A5083900503)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于上下文的无监督关键词提取框架MUDY。

**💡 创新点**

创新点在于结合提示式生成概率与候选词自适应权重，以及利用多粒度自注意力捕捉文档和段落层面的重要性。

**🔧 技术方法**

使用预训练语言模型(PLMs)与提示学习、自注意力机制进行评分。

**📊 数据集**

在四个公开真实数据集上进行评估。

**📈 对比分析**

与最新基线对比，在多种Top‑k阈值下均取得最高准确率，显著提升性能。

**⚠️ 局限性**

缺点包括对预训练模型的高度依赖、计算资源消耗大以及对低资源语言或特定领域的适用性有限。

---

## 329. Prediction of Alzheimer's Disease Risk Factors from Retinal Images via Deep Learning: Development and Validation of Biologically Relevant Morphological Associations in the UK Biobank

**arXiv ID:** 2605.00665 | [PDF](https://arxiv.org/pdf/2605.00665v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 330. Robust Fusion of Object-Level V2X for Learned 3D Object Detection

**arXiv ID:** 2605.00595 | [PDF](https://arxiv.org/pdf/2605.00595v1)

**作者:** Lukas Ostendorf `[一作]` (Institute for Automotive Engineering, RWTH Aachen University), Lutz Eckstein `[通讯]` (Institute for Automotive Engineering, RWTH Aachen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将对象级 V2X 信息融合到 BEVFusion 3D 检测框架中的方法，并通过噪声感知训练提升鲁棒性。

**💡 创新点**

创新点在于可控的 V2X 仿真管线、基于置信度的门控融合机制以及针对定位误差和传感器缺失的训练策略。

**🔧 技术方法**

采用 BEVFusion 网络、ResNet V2X 编码器、门控融合、Gaussian 噪声模拟与置信度编码。

**📊 数据集**

使用 nuScenes 数据集中的真实标注进行 V2X 仿真与评估。

**📈 对比分析**

与仅基于摄像头或雷达+摄像头的基线对比，噪声感知模型在理想 V2X 条件下 NDS 达 0.90，低噪声下 0.80，高噪声下仍高于基线；在 V2X 低渗透率时仍保持显著提升。

**⚠️ 局限性**

局限包括缺少完全遮挡对象、仿真噪声假设为独立 Gaussian、未考虑时间相关误差及真实 V2X 通信延迟。

---

## 331. Inductive Latent Context Persistence: Closing the Post-Handover Cold Start in 6G Radio Access Networks

**arXiv ID:** 2605.00593 | [PDF](https://arxiv.org/pdf/2605.00593v1)

**作者:** Anubhab Banerjee `[一作]` (Nokia Solutions and Networks GmbH & Co. KG), Daniyal Amir Awan `[通讯]` (Nokia Solutions and Networks GmbH & Co. KG)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对5G/6G RAN中交接时空状态丢失的问题，提出了 ILCP（Inductive Latent Context Persistence）协议，能够将 UE 的递归隐藏状态压缩为 128 字节的差分负载，并在 Xn 接口上传输，使目标 gNB 在交接后立即拥有有用的历史上下文。

**💡 创新点**

创新点包括：① 通过 β‑VAE 对 128 维 GRU 状态进行高效压缩；② 在动态异构图中为 UE‑cell 测量边和 gNB‑gNB Xn 边使用关系专属注意力；③ 设计了门控 MLP 投影层，将源端隐藏状态在目标端自动对齐；④ 通过实验验证该机制能消除 ping‑pong 并显著提升交接后几百毫秒的预测准确率。

**🔧 技术方法**

技术栈：异构图神经网络（HGT） + GRU 递归模块；β‑VAE 压缩器；门控 LayerNorm+MLP 投影；训练时采用下游候选集评分损失、强化鲁棒性训练（阴影衰落、NLOS 阻塞、SSB 间歇采样）等。

**📊 数据集**

数据集：维也纳 4G/5G 实际测量驾驶测试轨迹，含 31 次交接事件、100 Hz 采样，且实验中使用多种人为测量干扰以评估鲁棒性。

**📈 对比分析**

与基线比较：Zero‑Knowledge（重置隐藏状态）、GAT‑Temporal、Transformer‑Temporal、LSTM 以及 3GPP A3/A5 规则。结果显示 ILCP 在交接时刻准确率约 83.9%，并在交接后 5–25 步平均提升 5.1 pp、峰值提升 13.3 pp；Ping‑pong 率降至 0%；在受扰动测量场景中，ILCP 的失败率保持在 10–13 %，而 A3/A5 规则从 1.1 % 跳升至 57–65 %，表明 ILCP 对测量噪声更具鲁棒性。单张 GTX1080 上推理时间 p99 仅 7.7 ms。

**⚠️ 局限性**

局限性：① 仅使用 31 次交接的真实轨迹，样本量有限；② 评价标签为原始交接标签，未得到“最优”基站序列；③ 受扰动实验仅模拟测量误差，未覆盖网络切换、负载变化等真实业务场景；④ 需要更大规模的仿真或更多城市轨迹来进一步验证 ILCP 的普适性。

---

## 332. Jailbreaking Vision-Language Models Through the Visual Modality

**arXiv ID:** 2605.00583 | [PDF](https://arxiv.org/pdf/2605.00583v1)

**作者:** Aharon Azulay `[一作]` (Independent), Yossi Gandelsman `[通讯]` (Toyota Technological Institute at Chicago)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过四种视觉渠道（视觉密码、物体替换、文本替换、类比谜题）对多模态语言模型进行越狱攻击，揭示视觉输入在安全对齐中的空白。

**💡 创新点**

创新点在于提出可解释、无梯度且语义意义完整的视觉攻击范式，并系统展示了视觉攻击相较于文本攻击的优势与差异。

**🔧 技术方法**

主要技术包括自定义视觉符号表、场景级物体替换、图像文本编辑、视觉类比谜题生成及基于RLHF与Constitutional AI的安全判别器。

**📊 数据集**

使用了 HarmBench 的安全基准数据集，并结合公开的图像检索与文本生成模型（如 REVE、Gemini-2.5-flash-image）生成攻击样本。

**📈 对比分析**

与五种前沿 VLM 攻击基线对比，在六款前沿 VLM 上使用 Best‑of‑5 评估，视觉密码在 Claude‑Haiku‑4.5 上达 40.9% 成功率，整体视觉攻击平均成功率显著高于对应文本攻击。

**⚠️ 局限性**

局限包括评估仅覆盖 HarmBench，缺乏对更广泛危害类别的验证；攻击效果受图像生成质量影响；对模型内部机制的解释仍有限，且未研究跨模型迁移与组合攻击。

---

## 333. Type Theory With Erasure

**arXiv ID:** 2605.00655 | [PDF](https://arxiv.org/pdf/2605.00655v1)

**作者:** Constantine Theocharis `[一作]` (University of St Andrews), Edwin Brady `[通讯]` (University of St Andrews)

**通讯引用:** 940 | [OpenAlex ID](https://openalex.org/A5042121684)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出了使用第二阶广义代数理论（SOGAT）对可擦除型类型理论的结构化描述，并基于此构建了完整的语法与语义模型，包含代码提取的模型和实现。

**💡 创新点**

核心创新在于把擦除作为一种阶段（phase）区分的假设（synthetic phase distinction）嵌入类型理论，利用SOGAT形式化整个系统，从而实现：1）仅需一次擦除转换即可消除所有无关数据；2）得到可直接推导的语义模型；3）提供可实现的简易编译器框架。

**🔧 技术方法**

技术手段包括：
- SOGAT与GAT的转换、自由生成的 Σ‑CwF、Π‑CwF、子范畴模型；
- 采用 families-of-sets（Fam）和通用Grothendieck topos模型；
- 利用 gluing 技术证明代码提取的正确性；
- 在 Agda 中形式化证明并实现 toy elaborator；
- 通过“阶段”与“#”标记实现运行时/擦除模式的语义分离。

**📊 数据集**

论文没有使用任何外部数据集；所有验证均通过 Agda 形式化与内部演算器测试完成。

**📈 对比分析**

由于工作主要是理论与实现验证，没有传统的性能基准对比；但通过形式化证明展示了模型的完整性与代码提取的正确性，且 toy elaborator 在示例程序上可快速生成无擦除的 untyped lambda 代码。

**⚠️ 局限性**

限制与未来工作：
- 对于某些模式感知的绑定，保守性映射不是单射；
- 需要 tiny proposition 才能构造 squashed universes，限制了可用的 topos；
- 目前不支持运行时类型、更多阶段区分或更复杂的归纳型；
- 代码提取目标仍为无类型 λ 演算，尚未扩展至带原语的目标语言。

---

## 334. Budget Constraints as Riemannian Manifolds

**arXiv ID:** 2605.00649 | [PDF](https://arxiv.org/pdf/2605.00649v1)

**作者:** Michael Helcig `[一作]` (ETH Zürich), Dan Alistarh `[通讯]` (IST Austria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于Riemannian几何的预算约束优化方法——RCO，能在保持精确预算约束的同时直接对模型损失进行一次阶梯优化。

**💡 创新点**

创新点在于识别softmax期望成本的等值集为光滑Riemannian流形，并给出闭式法向量、单参数二分搜索退化（重投影）以及一次内积即可完成的向量传输，从而实现无超参数的精确约束。

**🔧 技术方法**

核心技术包括：Gumbel‑STE离散采样、基于动态规划的可行解搜索、Adam优化器、温度退火、以及上述三种流形操作（切线投影、重投影、动量传输）。

**📊 数据集**

实验数据集涵盖合成Knapsack实例与大规模语言模型压缩（混合精度量化、MoE专家修剪）场景。

**📈 对比分析**

与传统惩罚/拉格朗日方法相比，RCO在合成实例中恢复完整DP最优解，且在LLM压缩任务上与进化搜索方法相当或更优，且壁钟时间降低3–16倍。

**⚠️ 局限性**

局限性包括：前向DP求解的时间复杂度为O(NKB')，在极大选项集或细粒度预算时可能成为瓶颈；STE引入梯度偏差，缺乏理论收敛保证；当前重投影假设成本线性，无法直接处理非线性成本（如推理延迟）。

---

## 335. Learning Multimodal Energy-Based Model with Multimodal Variational Auto-Encoder via MCMC Revision

**arXiv ID:** 2605.00644 | [PDF](https://arxiv.org/pdf/2605.00644v1)

**作者:** Jiali Cui `[一作]` (Futurewei Technologies Inc), Heather Yu `[通讯]` (Futurewei Technologies Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种协同学习框架，利用最大似然更新与MCMC修正交织，联合训练多模态能量模型、共享潜在生成器和联合推断模型，以实现更高质量、更连贯的多模态生成。

**💡 创新点**

创新点在于：① 将EBM、共享潜在生成器和联合推断模型三者的MLE更新通过MCMC修正互相补充；② 通过生成器产生的多模态初始样本和推断器产生的潜在初始点作为 Langevin 动态的起点，显著提升采样效率与收敛质量；③ 用修正后的联合分布作为对抗信号，避免传统单一模型训练中出现的“模式崩溃”与“信息缺失”。

**🔧 技术方法**

技术手段包括：多模态能量模型（EBM）、共享潜在空间生成器（多模态VAE结构）、联合推断器（Mixture‑of‑Experts 方案）、有限步 Langevin 动态（用于 EBM 采样和潜在后验采样）、MLE 交替优化、梯度裁剪/stop‑gradient 处理 MCMC 修正、以及对比学习式的 KL 损失设计。

**📊 数据集**

实验数据集主要为 PolyMNIST（五个视觉模态）和 Caltech‑Birds（CUB）图像–文本配对；此外还对高分辨率 CUB（256×256）和大规模 MSCOCO 进行了扩展实验。

**📈 对比分析**

与一系列基线方法（Conditional CUB MVAE、MVTCAE、mmJSD、MoPoE、MMVAE、MMVAE+、MVEBM、MWBVAE、CoDEVAE、HELVAE、Diff‑CMVAE 等）以及传统单模态 EBM 在无补全模型的对比下，本文方法在 FID、CLIP 对齐分数、无条件/条件生成质量等指标上均表现更优；尤其在 CUB 上 FID 下降至约 25，低于所有对比模型；在高分辨率与 MSCOCO 上也保持了显著提升。

**⚠️ 局限性**

限制与不足：① MCMC 修正会带来额外的计算开销，尤其是对长链采样的需求；② 需要手动调节 Langevin 步数与学习率，调参成本高；③ 在极大规模数据或极高模态数的场景下，模型扩展性与收敛速度仍待验证；④ 目前的实验多集中在图像/文本对任务，跨模态泛化的广泛性尚未完全覆盖。

---

## 336. Paired-CSLiDAR: Height-Stratified Registration for Cross-Source Aerial-Ground LiDAR Pose Refinement

**arXiv ID:** 2605.00634 | [PDF](https://arxiv.org/pdf/2605.00634v1)

**作者:** Montana Hoover `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**通讯引用:** 39918 | [OpenAlex ID](https://openalex.org/A5004194238)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Paired-CSLiDAR benchmark，并设计了一种无训练的几何优化管线 RGSR，用于单帧航空-地面 LiDAR 的子米级姿态细化。

**💡 创新点**

核心创新是基于高度分层 ICP 与逆向注册产生多样化局部极值，并通过接受若比策略保证不出现 RMSE 回退，显著提升低覆盖场景下的细化成功率。

**🔧 技术方法**

所用技术包括点对点 ICP 的两阶段粗细匹配、逆向 ICP、残差引导分层细化、信心门限级联以及可选的 Fourier–Mellin BEV 估计。

**📊 数据集**

数据集为 Paired-CSLiDAR，包含 12,683 对地面与航空 LiDAR，覆盖 6 个美国校园/地区，提供 50 m 边界航空子图和每对的参考 SE(3) 对齐。

**📈 对比分析**

与标准 ICP、两阶段 ICP、GICP、Trimmed ICP、FGR、GeoTransformer、BUFFER-X 等基线比较，RGSR 在 Protocol B 的 9,012 扫描上实现 86.0% S@0.75 m 及 99.8% S@1.0 m，优于 GeoTransformer 9.7pp，优于学习模型。

**⚠️ 局限性**

局限在于仅测试正确的 50 m 航空子图；对姿态/高度误差缺乏鲁棒性；低覆盖场景仍会失败；且 RMSE 与实际位姿误差不完全对应，需要独立验证。

---

## 337. H-RAG at SemEval-2026 Task 8: Hierarchical Parent-Child Retrieval for Multi-Turn RAG Conversations

**arXiv ID:** 2605.00631 | [PDF](https://arxiv.org/pdf/2605.00631v1)

**作者:** Passant Elchafei `[一作]` (Johannes Kepler University Linz), Markus Schedl `[通讯]` (Johannes Kepler University Linz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 H‑RAG，一套层级父子检索-生成流水线，针对 SemEval‑2026 MTRAGEval 的多轮会话检索增广生成任务（Task A 与 Task C）。

**💡 创新点**

创新点在于将细粒度子块检索与大粒度父文档上下文重构分离，并在检索端采用轻量级混合密集‑稀疏检索、嵌入相似性再排序与父文档最大子分数聚合，显著提升多轮 RAG 的检索质量与生成的可证实性。

**🔧 技术方法**

核心技术包括：句子级子块切分、双向（密集+稀疏）检索与可调权重 α、嵌入相似性再排序、父文档聚合与最大分数策略、轻量级会话查询重写、以及指令调优的大语言模型（GPT‑5）生成与拒答控制。

**📊 数据集**

使用 MTRAG 语料库（SemEval‑2026 Task 8 MTRAGEval）作为检索与生成评测数据集，并在开发集上进行多维度超参探测。

**📈 对比分析**

与基线相比，H‑RAG 在 Task A 的 nDCG@5 取得 0.4271（相对基线略优），在 Task C 的调和平均得分为 0.3241，展现出在检索质量、文档相关度与生成流畅度之间的平衡，但仍显示出生成与证据对齐的瓶颈。

**⚠️ 局限性**

局限性包括：检索和聚合仍基于无监督方法，对极长上下文的可扩展性有限；父文档聚合采用最大分数聚合，可能忽略子块间的多样性；生成模块对检索质量高度依赖，导致证据与答案不完全对齐；未进行跨任务或跨语言的泛化验证。

---

## 338. CMTA: Leveraging Cross-Modal Temporal Artifacts for Generalizable AI-Generated Video Detection

**arXiv ID:** 2605.00630 | [PDF](https://arxiv.org/pdf/2605.00630v1)

**作者:** Hang Wang `[一作]` (Xi’an Jiaotong University), Cong Wang `[通讯]` (City University of Hong Kong)

**通讯引用:** 25857 | [OpenAlex ID](https://openalex.org/A5100390514)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种跨模态时间框架 CMTA，用于检测 AI 生成视频。

**💡 创新点**

创新点在于发现并利用 AI 生成视频中跨模态时间不稳定性（CMTA）这一独特痕迹，并将粗粒度与细粒度时间建模相结合，实现对跨模态时间特征的全尺度捕捉。

**🔧 技术方法**

技术上先用 BLIP 生成帧级文字描述，再用 CLIP 提取视觉‑文本对齐特征；随后用 GRU 对跨模态相似度序列做粗粒度建模，用 Transformer 编码细粒度帧级特征，最后融合两者后进行分类。

**📊 数据集**

实验数据集涵盖 GenVideo、EvalCrafter、VideoPhy、VidProM 四大 benchmark，包含 40 个子集，训练采用真实视频与 Pika 生成视频，测试覆盖多种 AI 视频生成器。

**📈 对比分析**

与 13 种基准方法（深度伪造检测、图像伪造检测、AI 视频检测）对比，CMTA 在 AP、AUC、ACC 上均取得最高分，平均 AP 提升至 98.74%（AUC 99.10%），相较最强对手提升超过 10%。

**⚠️ 局限性**

局限性包括：① 依赖预训练 BLIP/CLIP，若两模型在新领域失效会影响性能；② 对极短或极高分辨率视频的鲁棒性未充分验证；③ 在生成器大幅改进或全新架构出现时的泛化能力仍需进一步研究。

---

## 339. Intrinsic Gradient Suppression for Label-Noise Prompt Tuning in Vision-Language Models

**arXiv ID:** 2605.00591 | [PDF](https://arxiv.org/pdf/2605.00591v1)

**作者:** Jiayu Li `[一作]` (Zhejiang University), Xiansheng Hua `[通讯]` (Tongji University)

**通讯引用:** 20184 | [OpenAlex ID](https://openalex.org/A5024965898)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的无超参数双Softmax提示调优（DSPT）方法，用于处理视觉语言模型在标签噪声下的提示调优问题。

**💡 创新点**

创新点在于将双Softmax归一化嵌入交叉熵损失中，实现内在梯度抑制和损失函数有界，从而自动抑制噪声样本的梯度冲击。

**🔧 技术方法**

主要技术包括CLIP预训练模型、连续可学习提示、双Softmax交叉熵损失、梯度分析与理论证明。

**📊 数据集**

在多个公开数据集上评估，包括Caltech101、StanfordCars、OxfordPets、Flowers102、Food101、FGVCAircraft、DTD、EuroSAT、UCF101，且加入了对称与非对称噪声。

**📈 对比分析**

与零样本、CoOp、LogitNorm、Label Smoothing、NLPrompt等方法比较，DSPT在大多数噪声率下均达到或超过最优，平均提升约0.4%–5%，并在极端噪声下仍保持高于零样本的性能。

**⚠️ 局限性**

局限性包括仍需在更大规模、多模态任务中验证其可扩展性，对噪声分布假设（独立标签噪声）可能影响鲁棒性，并且在极端高噪声场景下仍存在性能下降的空间。

---

## 340. AI Washing Inflates Expected Performance but Not Interaction Outcomes: An AI Placebo Study Using Fitts' Law

**arXiv ID:** 2605.00582 | [PDF](https://arxiv.org/pdf/2605.00582v1)

**作者:** Nick von Felten `[一作]` (University of St. Gallen), Johannes Schöning `[通讯]` (University of St. Gallen)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在受控实验中使用Fitts定律任务测试 AI 标签对电脑鼠标使用者的期望与实际性能影响。

**💡 创新点**

证明 AI 洗白仅提升用户期望，而不改变主观评价或客观性能，并将 Fitts 定律作为评估 AI 标签效应的客观审计工具。

**🔧 技术方法**

基于 React/Node.js 自制实验平台，配合 NASA‑TLX、SUS‑DE‑Pos 调查以及 Fitts 定律回归、误差率与吞吐量分析。

**📊 数据集**

实验生成的 28 名受试者在基线、预测 AI 与生物信号 AI 三种条件下的点击数据（运动时间、目标难度、误差等）。

**📈 对比分析**

采用重复测量 ANOVA 与非参数检验，结果显示期望显著提升但工作量、可用性、误差率、模型拟合与吞吐量均无显著差异。

**⚠️ 局限性**

样本规模有限且主要为学生，实验在实验室环境下进行，缺乏真实情境与多样化受试者，未检验更复杂任务或高风险领域的影响。

---

## 341. Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs

**arXiv ID:** 2605.00674 | [PDF](https://arxiv.org/pdf/2605.00674v1)

**作者:** Jasper Dekoninck `[一作]` (ETH Zurich), Martin Vechev `[通讯]` (ETH Zurich)

**通讯引用:** 11331 | [OpenAlex ID](https://openalex.org/A5069901599)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立并维护一个持续更新的数学推理评估平台，涵盖从最终答案竞赛到证明生成、研究级问题以及 Lean 形式化证明等多种任务，并通过可视化接口公开所有结果与元数据。

**💡 创新点**

创新点在于把传统的静态基准转变为动态评估平台：持续添加/淘汰任务、自动化与人工评估并用、跨维度的细粒度分析、开放透明的接口以及利用项反应理论填补未评估任务的性能缺口。

**🔧 技术方法**

使用的技术包括：大语言模型推理管道、LLM 评审与人类裁判相结合的半自动评估、Lean 形式化验证工具、自动化从 arXiv 提取问题的脚本、成本-性能分析以及对新模型的周期性评测。

**📊 数据集**

数据集包括：AIME 2026、HMMT Feb 2026、Apex、Kangaroo 2025、Project Euler、arXivMath、BrokenArXiv、ArXivLean、USAMO 2026、Putnam 2025、IMC 2025 等，覆盖高中、大学、研究级别和形式化证明等多层次任务。

**📈 对比分析**

与多种开源与闭源 LLM（如 GPT‑5.5、DeepSeek‑v4‑Pro、Gemini、Claude 等）进行对比；评估指标为各任务的准确率、得分、成本与时间；GPT‑5.5 在绝大多数任务上领先，尤其是最终答案、研究级问题；Open‑source 模型虽有进步但仍落后 10–20%，在非最终答案任务的差距更大。

**⚠️ 局限性**

局限性包括：未覆盖交互式问题求解与多轮对话场景；聚焦于问题求解，忽略推理表达、猜想等更广泛的数学能力；工具使用范围有限，主要局限在部分任务；在 Lean 形式化任务上仍然表现不佳，表明当前模型对深度形式化的能力有限。

---

## 342. Learning How and What to Memorize: Cognition-Inspired Two-Stage Optimization for Evolving Memory

**arXiv ID:** 2605.00702 | [PDF](https://arxiv.org/pdf/2605.00702v1)

**作者:** Derong Xu `[一作]` (University of Science and Technology of China), Tong Xu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 4463 | [OpenAlex ID](https://openalex.org/A5025292786)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于记忆模式理论的两阶段优化框架 MemCoE，学习如何组织与更新 LLM 代理的用户记忆。

**💡 创新点**

创新点在于：①通过对比文本梯度引导的提示优化自动生成可迁移的记忆更新准则；②将准则与强化学习策略对齐，解耦记忆结构与内容，提供结构化的过程奖励。

**🔧 技术方法**

技术手段包括：对比文本梯度（contrastive textual gradient）与批量聚合实现准则诱导；群组相对策略优化（Group Relative Policy Optimization, GRPO）训练记忆演化策略；检索增强生成（RAG）和 LLM prompt 迭代优化。

**📊 数据集**

使用三个个性化记忆基准：PersonaMem、PrefEval 与 PersonaBench（涵盖显式/隐式偏好和不同噪声级别）。

**📈 对比分析**

与 Long Context、RAG、Mem0、A-Mem、LightMem、MemAgent、MEM-α 等基线对比，MemCoE 在所有任务上均获得最高分，展现出强健的泛化性、效率与可迁移性。

**⚠️ 局限性**

局限性包括：对 LLM 评分器提供准则奖励的依赖；对每轮 token 预算和演化轮数敏感，过多拆分会累积误差；当前为单目标策略，缺乏对稳定性、可塑性等多重目标的平衡。

---

## 343. ML-Bench&Guard: Policy-Grounded Multilingual Safety Benchmark and Guardrail for Large Language Models

**arXiv ID:** 2605.00689 | [PDF](https://arxiv.org/pdf/2605.00689v1)

**作者:** Yunhan Zhao `[一作]` (University of Illinois Urbana-Champaign), Bo Li `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 32058 | [OpenAlex ID](https://openalex.org/A5100688318)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于地区法规的多语言安全基准ML-Bench以及基于扩散式大语言模型的多语言安全防护模型ML-Guard，二者可实现跨语言、跨文化的安全评估与合规判断。

**💡 创新点**

创新点在于：①直接从17份跨14国、14语种的本土AI法规中提取风险类别和细粒度规则，构建无需机器翻译的政策驱动安全数据；②设计分级（seed/refined/attack-enhanced）查询与对抗增强场景；③采用扩散式大语言模型（Fast_dLLM）实现并行推理，并支持政策条件下的违规规则识别与自然语言解释。

**🔧 技术方法**

技术上主要采用LLM驱动的法规规则抽取与生成、对齐式多模型注释、两阶段分层风险构建、Diffusion Large Language Model（dLLM）微调、分阶段训练与增量政策更新。

**📊 数据集**

数据集包括：ML-Bench 56K实例（34K训练/22K评估），PolyGuardMix、Nemotron‑Safety‑Guard‑Dataset‑v3补充数据；以及PolyGuardPrompts、XSafety、RTP‑LX、MultiJail、CSRT、Nemotron等六个现有多语言安全基准；法规文本来自17份跨14国、14语种的AI监管文件。

**📈 对比分析**

通过在ML-Bench和六个现有基准上与11个开源/商用防护模型对比，ML-Guard‑7B在大多数任务中取得最高F1（如seed 0.97、refined 0.90、攻击增强0.92），平均F1 0.81；同时在违规规则识别和理由质量评估中也表现出高召回、低误报和平均4.0+分的解释质量。

**⚠️ 局限性**

局限性主要包括：①覆盖法规仅限于17份，难以全面适用于未覆盖地区；②规则抽取与注释依赖多模型LLM，可能引入错误；③对未知或未训练政策的泛化仍有限；④扩散模型规模仍较大，部署成本不低。

---

## 344. GeoContra: From Fluent GIS Code to Verifiable Spatial Analysis with Geography-Grounded Repair

**arXiv ID:** 2605.00782 | [PDF](https://arxiv.org/pdf/2605.00782v1)

**作者:** Yinhao Xiao `[一作]` (Guangdong University of Finance and Economics), Yihan Zhang `[通讯]` (Guangdong University of Finance and Economics)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于地理约束的LLM驱动GIS工作流验证与修复框架，并构建了可执行的任务契约。

**💡 创新点**

通过将地理规则编码为可检验的契约，结合多层（静态、运行时、语义）验证与有限修复循环，实现了从代码流畅度到空间正确性的转变。

**🔧 技术方法**

使用LLM（如DeepSeek、Kimi、Qwen等）生成Python GIS脚本，配合自定义的静态分析器、运行时检查器和语义验证器，以及基于契约的修复提示。

**📊 数据集**

构建了GeoContra-Real基准，包含来自15个波士顿大都市区的7,079个真实地理任务，涵盖向量、栅格、网络、拓扑等多种数据层。

**📈 对比分析**

在闭合模型（DeepSeek-V4、Kimi-K2.5）和11个开源模型上进行成对评估，比较基础设置与契约修复设置下的可执行率与空间正确率；结果显示空间正确率提升约20-30个百分点，显著优于仅生成可执行脚本的基线。

**⚠️ 局限性**

局限性在于验证器仍属轻量化，缺乏完整参考解、复杂拓扑或空间分布一致性检查，且对噪声或不足源数据的错误修复能力有限。

---

## 345. Observable Performance Does Not Fully Reflect System Organization: A Multi-Level Analysis of Gait Dynamics Under Occlusal Constraint

**arXiv ID:** 2605.00778 | [PDF](https://arxiv.org/pdf/2605.00778v1)

**作者:** Jacques Raynal `[一作]` (University of Montpellier), Jacques Margerit `[通讯]` (University of Montpellier)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了垂直咬合高度（VDO）作为约束条件下，帕金森患者步态与姿势的系统组织变化，采用单例实验设计；

**💡 创新点**

提出了多层次分析框架（聚合指标、动力学状态空间、潜在空间嵌入），展示相同表现可对应不同系统组织，强调VDO为约束而非固定最优值；

**🔧 技术方法**

使用可穿戴传感器（压力垫+惯性测量单元）采集步态数据，并通过UMAP无监督嵌入、聚合指标计算和动态系统概念模型进行分析；

**📊 数据集**

数据来源于一名72岁男性帕金森患者，进行两次实验（M1、M2），共六种咬合条件下的步态记录；

**📈 对比分析**

比较聚合指标（GPPS）与潜在空间分布，发现GPPS在部分条件下重叠但潜在空间可区分，表明聚合指标在区分系统组织方面效果有限；

**⚠️ 局限性**

局限性包括单例设计、缺乏因果推断、潜在空间依赖算法、未对嵌入稳定性进行量化评估、未单独控制辅导干预等。

---

## 346. Directed Social Regard: Surfacing Targeted Advocacy, Opposition, Aid, Harms, and Victimization in Online Media

**arXiv ID:** 2605.00776 | [PDF](https://arxiv.org/pdf/2605.00776v1)

**作者:** Scott Friedman `[一作]` (SIFT), Christopher Miller `[通讯]` (SIFT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Directed Social Regard（DSR）框架，对在线文本进行 span 级目标识别和三维情感（Oppose‑Advocate、Harmful‑Helpful、Victimized‑Aided）评分，构建相应 transformer 模型并在公开数据集上验证。

**💡 创新点**

创新点在于：① 将情感拆解为三个可解释、连续维度，捕捉正反两面情绪并标注目标跨度；② 结合社会科学理论（道德脱离、礼貌理论）指导维度设计；③ 设计全流程数据集、标注工具，并实现 span‑级多维度情感分析。

**🔧 技术方法**

技术手段：基于 BERT/DeBERTa/XLNet 等 transformer 的 span‑识别与评分头；使用 RMSE 损失训练评分模型；对比 LLM（Llama‑3.2 LoRA、GPT‑4o）；数据增强（debiased）提升泛化。

**📊 数据集**

使用的数据集：① 自建 DSR 注释数据集（约 5k 文本）；② Bridging Benchmark 11937 条评论；③ Moral Foundations Twitter Corpus（4 个子集，约 4k‑5k 条推文）；④ Ribeiro manosphere 3718 条帖子；⑤ MFTC（ALM、Baltimore、Davidson、MeToo）共 18k 条推文。

**📈 对比分析**

评估方法：span 识别采用严格/松散 F1，最佳模型 DeBERTa‑v3‑large‑debiased 达到 0.89（严格）/0.92（松散）；情感评分采用 RMSE 与 R²，最佳模型 RMSE≈0.10‑0.20，R²≈0.57‑0.84；与 LLM 对比，DeBERTa 在数值预测上显著优于 Llama‑3.2 LoRA 与 GPT‑4o。

**⚠️ 局限性**

局限性：① 低一致性跨度被过滤，可能遗漏细微情感；② 只在公开文本上验证，受限于 DoD SBIR 保护无法公开完整训练数据；③ 对跨域泛化仍需更多数据；④ LLM 在此任务中效果不佳，需改进提示或微调策略。

---

## 347. Characterizing the Expressivity of Local Attention in Transformers

**arXiv ID:** 2605.00768 | [PDF](https://arxiv.org/pdf/2605.00768v1)

**作者:** Jiaoda Li `[一作]` (ETH Zurich), Ryan Cotterell `[通讯]` (ETH Zurich)

**通讯引用:** 5037 | [OpenAlex ID](https://openalex.org/A5061951606)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析本地注意力对Transformer表达能力的提升，并通过线性时序逻辑给出形式化解释。

**💡 创新点**

证明本地注意力对应逻辑 [k]，与全局注意力对应的逻辑 ⟨⟩ 不可兼容，混合注意力可获得更强表达力，1-窗口最优。

**🔧 技术方法**

利用线性时序逻辑、正规语言理论、固定精度Transformer、多头注意力和位置编码技术进行理论推导与实验验证。

**📊 数据集**

在形式语言识别实验中使用多类可定义正规语言，随后在 WikiText-2 上进行下一个词预测实验。

**📈 对比分析**

通过对比全局、局部、混合注意力以及不同窗口大小和位置编码，发现混合模型（尤其 k=1）在形式语言和语言模型任务上表现最佳，显著优于单一注意力。

**⚠️ 局限性**

实验规模受限，未进行大模型或长时间训练；位置编码对性能的细微影响仍需进一步探索。

---

## 348. Self-Adaptive Multi-Agent LLM-Based Security Pattern Selection for IoT Systems

**arXiv ID:** 2605.00741 | [PDF](https://arxiv.org/pdf/2605.00741v1)

**作者:** Saeid Jamshidi `[一作]` (Polytechnique Montréal), Kawser Wazed Nafi `[通讯]` (Polytechnique Montréal)

**通讯引用:** 271 | [OpenAlex ID](https://openalex.org/A5110358595)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ASPO系统，将多代理LLM推理与确定性优化结合，实现在资源受限的IoT边缘网关上自适应安全模式选择。

**💡 创新点**

创新点在于将随机LLM决策与严格的确定性验证分离，形成闭环的MAPE-K控制，保证冲突安全、资源可行性和可审计性。

**🔧 技术方法**

采用多代理LLM（如DeepSeek v3、GPT-4o-mini等）生成候选组合，利用离线确定性优化核心实现冲突矩阵、资源预算约束，并在MAPE-K框架中实现监控、分析与执行。

**📊 数据集**

使用Bot-IoT数据集的攻击流重放，构建结构化上下文并评估在10个Raspberry Pi 4边缘节点上的决策。

**📈 对比分析**

与传统DRL和纯优化方法对比，ASPO在两种工作负载（500/1000决策）中实现99%+的安全门控通过率、0.5%‑0.8%的最终批准率，且极端延迟和能耗分别下降约22%和23%，而平均延迟和能耗基本保持不变。

**⚠️ 局限性**

主要限制包括对云端LLM服务的依赖、闭合世界模式对新安全模式的扩展性有限、以及候选生成质量对最终决策效果的高度依赖。

---

## 349. Quantum Gradient-Based Approach for Edge and Corner Detection Using Sobel Kernels

**arXiv ID:** 2605.00744 | [PDF](https://arxiv.org/pdf/2605.00744v1)

**作者:** Mohammad Aamir Sohail `[一作]` (University of Michigan), Hafize Asude Ertan `[通讯]` (İstanbul University-Cerrahpaşa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于量子梯度核的 Sobel 与 Harris 边缘及角点检测方法，并对 QPIE 与 FRQI 两种量子图像编码进行了实现与比较。

**💡 创新点**

创新点在于设计了 lag‑2 差分的量子梯度核，利用量子并行计算实现梯度估计，并结合经典 Sobel 后处理形成了高效的量子-经典混合检测管线；同时首次对两种编码方式在边缘和角点检测中的性能差异进行了系统评估。

**🔧 技术方法**

主要技术包括：量子图像编码（QPIE、FRQI）、量子梯度核电路（lag‑2 差分）、Hadamard 量子差分、经典 Sobel 边缘后处理、Harris 角点响应计算、经典阈值与非极大抑制。

**📊 数据集**

使用 512×512 分辨率的 UDED 与 Urban100 数据集中的若干自然图像及 AI 生成图像进行实验。

**📈 对比分析**

通过对比经典 Sobel、Canny 以及量子 QHED（Hadamard）检测，利用边缘密度、厚度、碎片化与熵等指标评估边缘质量；角点方面使用准确率与误报率比较，结果显示 QPIE+Sobel 边缘更连贯、角点准确率高、误报率低，FRQI 在有限采样下更易产生误报。

**⚠️ 局限性**

局限性主要包括：仅在无噪声理想模拟环境下验证；实际量子硬件上的状态制备、测量开销与噪声未得到充分评估；未实现端到端的速度提升；对大规模图像的可扩展性与量子资源需求待进一步研究。

---

## 350. Towards Improving Speaker Distance Estimation through Generative Impulse Response Augmentation

**arXiv ID:** 2605.00721 | [PDF](https://arxiv.org/pdf/2605.00721v1)

**作者:** Anton Ratnarajah `[一作]` (Amazon), Mrudula Athi `[通讯]` (Amazon)

**通讯引用:** 36 | [OpenAlex ID](https://openalex.org/A5078425704)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用仅依据说话者与听者位置条件化的 FastRIR 生成合成 RIR，并对其进行严格质量过滤后，使用增量数据细调 SDE 模型，显著降低了距离估计误差。

**💡 创新点**

创新点在于把 RIR 生成聚焦到源-接收位置、设计多维度质量控制滤波器以及在增量数据上执行超参数优化，以提升距离估计的精度。

**🔧 技术方法**

采用条件 GAN（FastRIR）、T60/DRR 质量过滤、能量衰减匹配以及自动超参数搜索等技术。

**📊 数据集**

使用 GWA、Treble、C4DM 与 VCTK 数据集进行训练与评估，并生成约 260k 级高质量合成 RIR 进行增量。

**📈 对比分析**

与基线 SDE 模型相比，MAE 在 GWA 由 1.66 m 降至 0.6 m，在 Treble 由 2.18 m 降至 0.69 m，证明数据增强在中远距离估计上效果显著。

**⚠️ 局限性**

局限性包括 1 m 以下距离的估计不佳、仅保留 25% 合成 RIR 可能导致信息损失，以及对更广泛非控制房间环境的泛化能力尚待验证。

---

## 351. Learning Coarse-to-Fine Osteoarthritis Representations under Noisy Hierarchical Labels

**arXiv ID:** 2605.00718 | [PDF](https://arxiv.org/pdf/2605.00718v1)

**作者:** Tongxu Zhang `[一作]` (Hong Kong Polytechnic University), Tongxu Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 26 | [OpenAlex ID](https://openalex.org/A5007079241)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文将膝关节骨关节炎（OA）评估建模为层级标签学习问题，采用简易双头网络探究层级监督对特征表示的影响；

**💡 创新点**

创新点在于：①将粗细层级标签视为表示层面的先验，使用双头结构直接塑造潜在空间；②从表示几何与解剖可解释性两方面评估层级监督效益；

**🔧 技术方法**

技术包括：3D ResNet、M3T Transformer、nnMamba编码器，双头分类损失组合，PCA 及 Spearman 相关评估潜在轴，梯度归一化 saliency 与骨软骨掩膜重叠分析；

**📊 数据集**

使用 OAIZIB-CM 膝关节 MRI 数据集，训练集383例，测试集98例，KL分级0-4；

**📈 对比分析**

通过与单任务（单 OA、单 KL）对照，并对不同 backbone 进行对比；结果显示：ResNet3D 在 KL 评估上双头显著提升（宏AUC+0.12），M3T 有中等提升；对 OA 预测的提升不显著；saliency 与软骨重叠在响应 backbone 上也有提升；

**⚠️ 局限性**

局限包括：①几何分析仅基于 PCA 低维指标；②KL 仅用多分类损失，未使用序数损失；③双头效果高度依赖 backbone，未深入探讨任务干扰与容量限制；

---

## 352. Aitchison Embeddings for Learning Compositional Graph Representations

**arXiv ID:** 2605.00716 | [PDF](https://arxiv.org/pdf/2605.00716v1)

**作者:** Nikolaos Nakis `[一作]` (Yale University), Giannis Nikolentzos `[通讯]` (University of Peloponnese)

**通讯引用:** 870 | [OpenAlex ID](https://openalex.org/A5011742954)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Aitchison几何下的图节点混合式嵌入框架AICoG，使节点被建模为对潜在原型因子的组成，并通过ILR坐标在欧氏空间中实现距离保留；

**💡 创新点**

将Aitchison几何和可逆的ILR变换引入图嵌入，既保持了欧氏距离模型的表达能力，又赋予相似度可解释的“相对权重”语义；

**🔧 技术方法**

使用Aitchison几何、ILR坐标、混合成员身份模型、伯努利边缘似然、可学习的ILR基底、子组成一致性分析与子维度约简；

**📊 数据集**

在七个公开图数据集上评估：LastFM、Citeseer、Cora、DBLP、AstroPh、GrQc、HepTh；

**📈 对比分析**

与Node2Vec、Role2Vec、NetMF、SLIM‑Raa、HM‑LDM、MMSBM、MNMF、Simplex‑Euclidean等基线对比；AICoG在链路预测与节点分类上与最强基线相当，且通过子组成压缩仍保持高性能；

**⚠️ 局限性**

仅在节点角色天然可混合成分时最适用，对非成分性结构的图可能无法提升精度，且需要额外的基底学习和参数调优；

---

## 353. Deep Kernel Learning for Stratifying Glaucoma Trajectories

**arXiv ID:** 2605.00708 | [PDF](https://arxiv.org/pdf/2605.00708v1)

**作者:** Bruce Rushing `[一作]` (University of Virginia), Heman Shakeri `[通讯]` (University of Virginia)

**通讯引用:** 434 | [OpenAlex ID](https://openalex.org/A5006445265)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种融合Transformer与高斯过程的深度核学习框架，对青光眼患者的不规则多模态EHR进行视觉视力预测并对疾病轨迹进行无监督分型。

**💡 创新点**

创新点：1）将Clinical‑BERT嵌入与Transformer特征提取器相结合，能处理高维、稀疏且时间不规则的序列；2）在GP后端实现自然插值与置信区间估计，实现对不规则时间序列的自适应建模；3）通过无监督聚类解耦当前病情与未来进展风险，发现三类临床轨迹，展示模型真正学习进展风险而非仅复制现状。

**🔧 技术方法**

技术：Transformer编码器 + Clinical‑BERT词向量；深度核学习（Deep Kernel Learning）与高斯过程；周期性时间编码（sin/cos）；变分推断（ELBO）训练；Ward层次聚类；SHAP特征重要性分析。

**📊 数据集**

数据集：SOURCE电子病历（67,691例，402,552条随访记录），包含结构化特征、自由文本、临床指标；使用logMAR视觉视力作为目标，取每眼最优/最差视力作为测量。

**📈 对比分析**

对比方法：MLE RNN/LSTM/GRU/Transformer、GRU‑D、GRU‑ODE、RNN‑CDE等基线模型；DKL Transformer在MSE、MAE、R²、0.1logMAR准确率等指标上均优于或相当于基线，MAE 0.1703，0.1logMAR准确率 53.06%，显示出显著的预测性能。

**⚠️ 局限性**

局限性：1）以视力为终点，缺乏金标准功能/结构指标（视野、OCT、IOP），难以区分青光眼与白内障等并发；2）未建模双眼相关性；3）对外部数据的泛化尚未验证；4）手术等治疗变量可能是混杂因素；5）公平性分析尚在进行中。

---

## 354. Temporal Data Requirement for Predicting Unplanned Hospital Readmissions

**arXiv ID:** 2605.00738 | [PDF](https://arxiv.org/pdf/2605.00738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 355. STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack

**arXiv ID:** 2605.00699 | [PDF](https://arxiv.org/pdf/2605.00699v1)

**作者:** Xutao Mao `[一作]` (City University of Hong Kong), Cong Wang `[通讯]` (City University of Hong Kong)

**通讯引用:** 25857 | [OpenAlex ID](https://openalex.org/A5100390514)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 STARE，一种分层强化学习框架，用于对视觉语言模型进行毒性攻击的端到端红队测试；

**💡 创新点**

创新点在于将去噪轨迹视为攻击表面，结合提示编辑与低层去噪微调，实现了优化诱导的阶段对齐（OIPA）现象，显著提升毒性攻击的可预测性和成功率；

**🔧 技术方法**

使用的技术包括分层MDP（高层提示编辑与低层去噪）、GRPO（分组相对策略优化）、LoRA 微调、Rectified Flow 3.5-Medium 生成模型以及毒性评估工具 Detoxify；

**📊 数据集**

主要数据集为 RealToxicityPrompts（RTP）和 PolygloToxicityPrompts（PTP），并在多种 VLM（LLaVA、Gemini、Qwen、GPT-5.4）上进行评估；

**📈 对比分析**

与多种基线（如 PGJ、DiffZOO、ART、DDPO）对比，STARE 在 RTP 上达到最高 31.36% 的攻击成功率（ASR），比最强基线 ART 提升 12.74%，并在跨模型跨数据集迁移上保持高效；

**⚠️ 局限性**

局限性包括需要对 T2I 模型的白盒访问、仅在英语毒性场景下验证、实验聚焦于 SD 3.5-Med，且对其他架构或不同伤害类型的适用性尚待验证。

---

## 356. Modeling Subjective Urban Perception with Human Gaze

**arXiv ID:** 2605.00764 | [PDF](https://arxiv.org/pdf/2605.00764v1)

**作者:** Lin Che `[一作]` (ETH Zurich), Peter Kiefer `[通讯]` (ETH Zurich)

**通讯引用:** 1859 | [OpenAlex ID](https://openalex.org/A5015281762)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了一个新的数据集Place Pulse-Gaze，将街景图像、同步的眼动记录和个体主观评价相结合，并提出了基于眼动引导的城市感知框架，探讨眼动信息对主观城市感知预测的贡献。

**💡 创新点**

创新点在于：①首次将眼动数据与街景图像结合，用以刻画个体化的感知过程；②提出三种融合策略（仅眼动、眼动+语义AOI、眼动+预训练ViT补丁），系统验证眼动与视觉内容的协同效应；③通过多种对照实验（无眼动、随机对齐）证明眼动信息对预测的显著提升。

**🔧 技术方法**

使用的技术包括眼动事件检测（I-DT）、语义分割（Mask2Former）、Transformer序列建模、预训练ViT、集成梯度可解释性分析、交叉熵损失与AdamW优化。

**📊 数据集**

使用的数据集是基于Place Pulse 2.0 的子集，包含2248张街景图像，经过超分辨率增强后收集了10223对图像-眼动-感知标签的数据，共涉及96名受试者，评价维度包括富裕（Wealth）、安全（Safety）、无聊（Boredom）。

**📈 对比分析**

方法比较采用宏F1和准确率指标，在三维度上与多种基线（热力图+CNN、XGBoost、AOI统计、Image-Only ViT等）进行对比。结果显示：仅眼动模型已比随机更优；眼动+语义AOI融合在三维度上均超过仅语义或仅视觉模型，提升约4–6个百分点；眼动+ViT补丁同样提升约1–2个百分点，说明眼动为预训练视觉特征提供了补充的个体化信息。

**⚠️ 局限性**

局限性包括：数据规模受眼动采集成本限制，仅覆盖欧洲和北美的城市，图像来自2010年代且分辨率低；实验环境为静态室内实验，未能覆盖真实户外动态环境；缺乏对眼动特征的因果解释以及对其他生理信号的整合。

---

## 357. AdvNet: Revealing Performance Issues in Network Protocols by Generating Adversarial Environments

**arXiv ID:** 2605.00755 | [PDF](https://arxiv.org/pdf/2605.00755v1)

**作者:** Shehab Sarar Ahmed `[一作]` (University of Illinois Urbana-Champaign), Michael Schapira `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 5332 | [OpenAlex ID](https://openalex.org/A5027593733)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于机器学习的框架AdvNet，自动生成对TCP、MPTCP等传输协议产生性能差距的对抗性网络环境，并在Linux内核实现上进行广泛评测。

**💡 创新点**

创新点包括：①将对抗环境生成建模为最大化目标协议与参考协议的性能差距；②采用trace‑based环境表示与噪声鲁棒的后学习选择（MRE）提升样本效率；③在真实内核实现上而非模拟器进行测试，发现多项Bug和性能隐患。

**🔧 技术方法**

使用机器学习优化算法（遗传算法、贝叶斯优化、ε‑贪婪搜索）结合并行Mahimahi仿真；引入Post‑Learning Selection与统计噪声处理；利用eBPF与Wireshark进行根因分析。

**📊 数据集**

构造带宽、延迟、时长等参数区间的时变trace，未使用公开数据集，而是生成多维参数搜索空间；在Linux 5.15等版本内核中对17种TCP和4种MPTCP进行实验。

**📈 对比分析**

对比GA、随机生成、BO、EPS等算法，GA在约1小时内取得最佳性能；后学习选择MRE提升约26%；在所有对比实验中，AdvNet发现448/544的对抗案例，暴露多项内核bug，显示比传统手工或CC‑Fuzz更高的对抗得分。

**⚠️ 局限性**

受限于仿真器的噪声与并行度，样本效率仍有限；仅关注时间变带宽/延迟，未覆盖工作负载、拓扑、丢包等复杂环境；缺乏自动根因分析与解释，需人工排查。

---

## 358. EASE: Federated Multimodal Unlearning via Entanglement-Aware Anchor Closure

**arXiv ID:** 2605.00733 | [PDF](https://arxiv.org/pdf/2605.00733v1)

**作者:** Zihao Ding `[一作]` (South Dakota State University), Jun Huang `[通讯]` (South Dakota State University)

**通讯引用:** 5253 | [OpenAlex ID](https://openalex.org/A5020146420)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种针对联邦多模态学习的遗忘机制，能在不共享原始图文对的情况下实现模型遗忘；

**💡 创新点**

引入Anchor Principle识别三类遗忘残留通道，并通过BKE、GSD、PFL三种技术闭合这些通道；

**🔧 技术方法**

使用LoRA轻量级适配器、余弦正交分解、投影锁定、InfoNCE对齐损失；

**📊 数据集**

在Flickr30K、MS COCO、TextCaps三个图文检索数据集上进行实验；

**📈 对比分析**

与九个联邦和集中式遗忘基线对比，EASE在遗忘召回率下降到与重训练相近的水平，同时保持保留召回率接近重训练，整体表现优于所有基线；

**⚠️ 局限性**

局限在于仅验证了基于双线性相似度的检索任务，未覆盖生成式或多模态LLM；此外需要预训练冻结模型和LoRA，且未给出严格的理论遗忘证明。

---

## 359. Empowering Heterogeneous Graph Foundation Models via Decoupled Relation Alignment

**arXiv ID:** 2605.00731 | [PDF](https://arxiv.org/pdf/2605.00731v1)

**作者:** Ziyu Zheng `[一作]` (Xidian University), Wei Zhao `[通讯]` (Xidian University)

**通讯引用:** 92109 | [OpenAlex ID](https://openalex.org/A5050699488)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对多域异构图预训练中的负迁移问题，提出了 Decoupled Relation Subspace Alignment（DRSA）框架，作为一种无须人工 Meta‑Path 的通用预处理模块，先在低秩关系子空间中对跨类型关系进行对齐，再通过特征-结构解耦将对齐特征分解为语义投影与结构残差，从而实现节点语义与关系拓扑的双重对齐。

**💡 创新点**

创新点在于：① 把特征语义与关系拓扑分离；② 采用双关系子空间投影（基于随机低秩投影）显式建模跨类型交互；③ 通过闭式交替最小化（Block Coordinate Descent）实现高效稳定的对齐；④ 以 Plug‑and‑Play 的方式可直接嵌入任意图基预训练模型，显著缓解“类型崩塌”和“关系混淆”。

**🔧 技术方法**

核心技术包括：低秩双关系子空间投影、特征‑结构解耦表示、两阶段交替优化（结构驱动与特征投影闭式求解）、以及对齐残差正则化的 Block Coordinate Descent。

**📊 数据集**

实验使用六大公开异构图基准：DBLP、ACM、IMDB、AMiner、Freebase、YELP（以及对比用的 Freebase/DBLP/ACM/AMiner/IMDB/YELP 六域交叉预训练）。

**📈 对比分析**

与 MP2V、HeCo、HGMAE、RMR、HetGPT、HGPrompt、GCOPE、MDGPT、SAMGPT 等十种基线（含元路径依赖与无元路径、以及多域图基预训练模型）在节点分类、边分类及少样本任务上进行留一交叉评估；在大多数数据集上，DRSA 使目标模型提升 10%–30% 的准确率/auc，显著减少负迁移，并在少样本场景保持稳健优势。

**⚠️ 局限性**

局限性：① 仍需手动调节残差正则 β 与投影正则 γ，参数敏感度略高；② 采用随机低秩投影，可能对极其稀疏或极大规模图的关系结构捕捉不足；③ 目前仅处理结构与特征的对齐，未考虑文本或属性的语义融合；④ 预处理步骤增加额外计算开销，尚未在超大图上进行可扩展性评估。

---

## 360. The Benefit of Decoder-Provided Pilots in Highly Dynamic Channels

**arXiv ID:** 2605.00761 | [PDF](https://arxiv.org/pdf/2605.00761v1)

**作者:** Duschia Bodet `[一作]` (Northeastern University), Ken Duffy `[通讯]` (Northeastern University)

**通讯引用:** 4666 | [OpenAlex ID](https://openalex.org/A5042235507)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

利用解码后的码字作为额外的训练序列，在高速时变信道中实现无迭代、模组和码解码器无关的通道估计和数据传输

**💡 创新点**

提出decoder‑provided pilots方法，非迭代、对码解码器和调制无依赖，并在快衰落条件下通过短码提升容量

**🔧 技术方法**

采用硬判决解码输出、CRC门控、软输出阈值、MMSE均衡、AR Rayleigh 信道模型、GRAND 与 CA‑Polar 等前向纠错码

**📊 数据集**

使用 eBCH² 代码（[1024,676]、[128,113]等）、CA‑Polar 代码（[128,54]）与 OFDM、单载波信道，模拟 Rayleigh、dicode 与多tap 信道

**📈 对比分析**

与传统预留训练、DFE、soft‑output 数据辅助等方法对比，实验显示 decoder‑provided pilots 在 4‑QAM 与 16‑QAM、单载波与 OFDM 下实现 5–7 dB 的 BER 降低，吞吐率提升并能在快衰落中保持高容量

**⚠️ 局限性**

对码字长度与信道相干时间的匹配有限，过长码字导致错误传播与容量下降；在高 SNR 仍受软输出限制；需进一步验证在非 AR 信道与更复杂 MIMO 设定下的鲁棒性

---

## 361. A Near-Linear-Time Algorithm for Finding a Well-Spread Perfect Matching in Bridgeless Cubic Graphs

**arXiv ID:** 2605.00710 | [PDF](https://arxiv.org/pdf/2605.00710v1)

**作者:** Babak Ghanbari `[一作]` (Charles University), Robert Šámal `[通讯]` (Charles University)

**通讯引用:** 479 | [OpenAlex ID](https://openalex.org/A5081562141)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种近线性时间算法，在任何无桥三正则图中寻找一个能够与每个三边割正好相交一次的完美匹配（WSPM）。

**💡 创新点**

创新点在于将2-边割的cactus表示与2-割归约相结合，突破了以往只能处理3-边连通图的限制；同时实现了O(n log⁴n)的时间复杂度。

**🔧 技术方法**

主要技术包括：cactus（树状图）表示所有2-边割、2-割归约操作、预先设定边的WSPM求解、以及通过可逆归约回溯合并匹配。

**📊 数据集**

本工作为理论算法，无需使用任何实际数据集，所有分析和证明均在抽象图模型上完成。

**📈 对比分析**

与先前的O(n³)和仅适用于3-边连通图的O(n log⁴n)算法相比，本文实现了更快的近线性时间，并在任何无桥三正则图上保证可行性。

**⚠️ 局限性**

局限性：仅适用于无桥三正则图；若图存在桥则可能不存在完美匹配；算法仍保留log⁴n因子，实际常数较大。

---

## 362. Exploring the Limits of End-to-End Feature-Affinity Propagation for Single-Point Supervised Infrared Small Target Detection

**arXiv ID:** 2605.00722 | [PDF](https://arxiv.org/pdf/2605.00722v1)

**作者:** Qiancheng Zhou `[一作]` (Shanghai University), Wenhua Zhang `[通讯]` (Shanghai University)

**通讯引用:** 8116 | [OpenAlex ID](https://openalex.org/A5100443670)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种单点监督的红外小目标检测方法 GSACP，通过在训练批内利用点锚点的特征相似性进行硬边缘亲和传播，直接生成掩膜监督，从而消除传统伪标签外循环。

**💡 创新点**

创新点在于：① 形式化“自引用传播漂移”（Self‑Referential Propagation Drift）这一代表性‑监督耦合失稳机制；② 通过局部教师解耦、硬背景对比、适应性支持门控和后期平坦化四个维度，系统性地消解漂移；③ 证明单阶段端到端模型可在极低误报率下保持竞争性性能。

**🔧 技术方法**

采用的关键技术包括：基于余弦相似度的硬边缘亲和计算、局部 EMA 教师混合、负样本对比损失、半监督 OHEM、以及多尺度支持门控；模型基于 MSDA 语义分割骨干网络。

**📊 数据集**

主要使用 SIRST3 数据集进行训练与验证，并在 NUDT‑SIRST、NUAA‑SIRST、IRSTD‑1K 进行零样本跨域评测。

**📈 对比分析**

与多阶段伪标签方法 PAL 进行对比；GSACP‑Final 在 mIoU 上略逊于 PAL（0.6674 vs 0.6933），但误报率 Fa 降至 15.33（比 PAL 下降 38%），在“误报‑重叠” Pareto 前沿上实现了低误报的部署优势。

**⚠️ 局限性**

局限性：① 由于缺乏外循环的时间正则化，掩膜重叠（mIoU）仍有提升空间；② 对大尺寸目标的支持相对保守；③ 在极端训练动态下仍需额外的后期平坦化处理。

---

## 363. Static and Dynamic Graph Alignment Network for Temporal Video Grounding

**arXiv ID:** 2605.00684 | [PDF](https://arxiv.org/pdf/2605.00684v1)

**作者:** Zhanjie Hu `[一作]` (Ningbo University), Jiangbo Qian `[通讯]` (Ningbo University)

**通讯引用:** 901 | [OpenAlex ID](https://openalex.org/A5004828717)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了静态与动态视觉特征双流图网络（SDGAN）用于视频与文本的时间定位任务，结合多粒度提案与渐进式训练。

**💡 创新点**

创新点包括：① 双流图网络联合静态与动态特征并通过位置节点对齐提升视觉表征；② 查询–剪辑对比学习与自适应图建模实现查询感知的图卷积；③ 多粒度提案与渐进式易难训练策略促进精细边界预测。

**🔧 技术方法**

使用技术：图卷积网络（GCN）、对比学习（InfoNCE、Softplus）、位置节点对齐、查询–剪辑对比、可调图建模、基于 IoU 的回归损失、以及多粒度时序特征聚合。

**📊 数据集**

实验数据集：ActivityNet Captions、Charades‑STA、TACoS。

**📈 对比分析**

与最新基准（UniSDNet、PLN 等）对比，SDGAN 在三大数据集上均取得最佳或领先效果；如 TACoS 上 R@5@0.5 提升约 15%，ActivityNet mIoU 超过 PLN。

**⚠️ 局限性**

局限性：模型参数和计算量较大，训练过程对超参数敏感；对极长视频或极复杂查询的泛化仍有限；仅在三大公开数据集上验证，缺乏对更广泛场景的评估。

---

## 364. Meritocratic Fairness in Budgeted Combinatorial Multi-armed Bandits via Shapley Values

**arXiv ID:** 2605.00762 | [PDF](https://arxiv.org/pdf/2605.00762v1)

**作者:** Shradha Sharma `[一作]` (Indian Institute of Technology Ropar), Shweta Jain `[通讯]` (Indian Institute of Technology Ropar)

**通讯引用:** 3796 | [OpenAlex ID](https://openalex.org/A5107499112)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在预算限制的组合多臂赌博机（BCMAB）中，针对全反馈（仅观察整体奖励）场景，提出了 K‑Shapley 值来衡量每个臂的贡献，并设计了一种自适应学习 K‑Shapley 并根据其比例进行公平选择的算法。

**💡 创新点**

创新点包括：①引入 K‑Shapley 值扩展 Shapley 值以适应有限规模的合作游戏；②证明 K‑Shapley 唯一满足对称性、线性性、零玩家性和 K‑效率的公理；③在全反馈环境下结合 Monte Carlo 近似与 UCB 风格的置信上界，得到 O(T^{3/4}) 的公平回报（fairness regret）上界；④首次将 meritocratic fairness 应用于 BCMAF-FBF 并在实际任务中验证其有效性。

**🔧 技术方法**

主要技术手段包括：Shapley 值与 K‑Shapley 的理论推导；蒙特卡洛（Monte‑Carlo）随机排列估计 K‑Shapley；重复抽样（L 次）减小噪声；UCB 置信上界构造乐观估计；随机化舍入（Randomized Rounding Scheme）将概率分配转化为可执行的 K 规模子集；理论分析结合 Hoeffding 不等式与 martingale 技巧给出公平回报上界。

**📊 数据集**

实验数据集：①合成数据（20 个臂，子模函数，K=5）；②联邦学习（CIFAR‑10 在 100 个非 IID 客户端上划分，K=10）；③社交影响力最大化（Facebook 子图 534 节点，K=20，独立级联模型）。

**📈 对比分析**

与基线的比较方法：使用公平回报（fairness regret）和所选臂与真实 K‑Shapley 比例的一致性；在实验中，所提算法在三种数据集上均显著低于 Fair‑CMAB、ETCG、GAP‑E 等基线，并在联邦学习中保持甚至提升全局模型准确率；实验结果显示公平回报随时间单调下降，达到与理论上界相符的子线性速度。

**⚠️ 局限性**

局限性包括：①尚未给出公平回报的下界；②假设环境是静态且奖励函数可通过有限次抽样估计，未考虑动态/非平稳情形；③蒙特卡洛估计与重复抽样的计算开销较大，尤其在大规模臂集合时；④仅适用于预算限制的组合结构，无法直接推广到更一般的子集大小约束。

---

## 365. NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search

**arXiv ID:** 2605.00751 | [PDF](https://arxiv.org/pdf/2605.00751v1)

**作者:** Sizhe Tang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6506 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种名为 NonZero 的多智能体 MCTS 框架，通过低维非线性返回代理和交互引导的候选扩展，避免了在 d^n 维联合动作空间中枚举动作的计算瓶颈。

**💡 创新点**

创新点包括：① 利用单一和双智能体的离散一阶、二阶差分构造交互得分；② 将候选动作生成视为非线性 Bandit 问题，提出 NonUCT 优化规则；③ 在局部收敛下给出子线性 regret 保证，消除维度灾难；④ 结合超网络初始化和混合二阶梯度提升模型收敛速度。

**🔧 技术方法**

采用的技术包括：MuZero 结构（表示、动态、预测头）；asinh‑GLM 低维返回代理；交互得分与离散二阶差分；非线性 Bandit 优化（NonUCT）；超网络用于节点参数初始化；梯度下降更新回报代理；MCTS 四阶段（Selection、Expansion、Simulation、Backup）。

**📊 数据集**

实验数据集：MatGame（多维博弈，线性/非线性奖励）；StarCraft Multi‑Agent Challenge（SMAC）；SMACv2（带随机起始/异构单位）。

**📈 对比分析**

与 MAZero、MAZero‑NP、MA‑AlphaZero、MAPPO、QMIX 等基线对比。结果显示：在 MatGame 中样本效率提升约 10‑15%（尤其高维非线性场景），在 SMAC/SMACv2 中赢率超过 96% 并且收敛速度比最佳基线快 50‑70% 环境步数，整体性能优于所有对比方法。

**⚠️ 局限性**

局限性：① 只保证局部最优收敛，无法保证全局最优；② 对离散二阶差分的估计和超参数（如探索系数、候选数）敏感；③ 需要足够丰富的训练数据以拟合非线性返回代理；④ 在动态代理数或连续动作空间中的适用性尚未验证。

---

## 366. To Call or Not to Call: A Framework to Assess and Optimize LLM Tool Calling

**arXiv ID:** 2605.00737 | [PDF](https://arxiv.org/pdf/2605.00737v1)

**作者:** Qinyuan Wu `[一作]` (Max Planck Institute for Software Systems), Muhammad Bilal Zafar `[通讯]` (Ruhr University Bochum)

**通讯引用:** 2643 | [OpenAlex ID](https://openalex.org/A5102901191)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型在调用外部工具（如网络搜索）时的决策，提出基于必要性、效用与可负担性的评估框架，并开发轻量级的隐状态估计器来改进工具调用策略。

**💡 创新点**

①将工具调用拆分为三维度（必要性、效用、可负担性），①通过规范、描述和处方视角系统化分析；②提出只需利用模型内部隐藏表示即可训练的需求与效用估计器，显著提升决策质量。

**🔧 技术方法**

决策理论、LLM内部隐藏状态提取、轻量级 MLP 估计器、成本约束的预算分配与 NDCG 对齐评估。

**📊 数据集**

三大任务：Entity、InVivoQuery 与 BFCL；工具使用为 Google Search（SerpApi）与 Perplexity Search；模型覆盖 3B–120B 参数的多款开源 LLM。

**📈 对比分析**

与自发调用、始终调用、无工具等基准比较；使用真实与感知的需求/效用标签进行监督；实验表明隐状态估计器在所有模型上均超越原生决策，尤其在小模型中显著提升事实性得分与调用效率；在预算约束下，利用估计器的 top‑K 策略实现更高的 utility gain。

**⚠️ 局限性**

估计器并非完美，仍有误差；对工具描述的依赖不稳定；未深入探讨工具调用导致负效用的机制；缺乏对更复杂或多工具场景的实验与评估。

---

## 367. Smallest Enclosing Disk Queries Using Farthest-Point Voronoi Diagrams

**arXiv ID:** 2605.00743 | [PDF](https://arxiv.org/pdf/2605.00743v1)

**作者:** Kevin Buchin `[一作]` (TU Dortmund University), Frank Staals `[通讯]` (Utrecht University)

**通讯引用:** 543 | [OpenAlex ID](https://openalex.org/A5032979919)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

该论文提出一种基于二维最远点Voronoi图的结构，能够在预处理后以O(log^4 n)（确定性）或O(log^{5/2}n·loglog n)（随机化）的时间内回答点集在轴对齐矩形范围内的最小包围圆查询。

**💡 创新点**

创新点在于完全摆脱了三维提升和多面体交叉搜索的复杂方法，直接利用最远点Voronoi图的树状结构实现高效搜索，并将Welzl算法的思想与Eppstein的动态规划框架结合，获得更快的随机化查询时间。

**🔧 技术方法**

主要技术包括：二维最远点Voronoi图及其中心点定位、质心分解（centroid decomposition）用于快速子树裁剪、范围树（range tree）得到O(log n)个典型子集、以及Eppstein动态规划框架处理随机化递归。

**📊 数据集**

论文未提供实验数据集，而是通过理论分析给出预处理与查询复杂度；若需实验，可使用标准几何点集（如随机点、格点或真实地理坐标）。

**📈 对比分析**

与之前的O(log^6 n)（以及O(log^9 n)）查询时间相比，该方法在理论上实现了显著加速；随机化方案进一步提升到O(log^{5/2}n·loglog n)，在预处理时间与空间仍保持O(n log^2 n)。

**⚠️ 局限性**

局限性包括：需要点集处于一般位置（否则需符号扰动）；实现复杂度较高（需构造Voronoi图、质心分解和多层查询），对实际大规模数据的常数因子未评估；随机化方法虽然期望更快，但在最坏情况可能不如确定性方案。

---

## 368. HyperCertificates: Verification of Discrete-time Dynamical Systems against HyperLTL Specifications

**arXiv ID:** 2605.00752 | [PDF](https://arxiv.org/pdf/2605.00752v1)

**作者:** Vishnu Murali `[一作]` (University of Colorado Boulder), Majid Zamani `[通讯]` (University of Colorado Boulder)

**通讯引用:** 3806 | [OpenAlex ID](https://openalex.org/A5030109984)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出一种 HyperCertificates 框架，用来验证离散时间动力学系统满足 HyperLTL 超属性，并通过自动化工具可合成相应的证书。

**💡 创新点**

将闭包证书（closure certificate）与 barrier 证书相结合，利用前瞻信息克服传统 Augmented Barrier 只考虑一步的局限，可处理 ∀^*∃^* 以及一般 HyperLTL 公式。

**🔧 技术方法**

使用 Sum‑of‑Squares (SOS) 优化和 SMT‑based CEGIS 等形式化方法来自动搜索证书函数，辅以闭包、排名函数等理论工具。

**📊 数据集**

在案例研究中使用自定义的离散时间一维数值系统、三维车辆动力学模型以及有限状态示例，未使用公开数据集。

**📈 对比分析**

与已有的 Augmented Barrier、抽象化等方法对比，实验表明在这三种案例中能成功证明系统满足超属性，而传统方法在某些例子中失败；合成时间从几分钟到数小时不等，展示了实用性。

**⚠️ 局限性**

仍存在模板选择依赖手工经验、对多条相同起止路径的情况处理有限、缺乏必要性条件的分析，以及对随机系统和控制合成的支持不足等局限。

---

## 369. PhysEdit: Physically-Consistent Region-Aware Image Editing via Adaptive Spatio-Temporal Reasoning

**arXiv ID:** 2605.00707 | [PDF](https://arxiv.org/pdf/2605.00707v1)

**作者:** Guandong Li `[一作]` (iFLYTEK), Mengxia Ye `[通讯]` (Aegon THTF)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种在推理时根据编辑指令自适应分配空间支持和推理深度的图像编辑框架。

**💡 创新点**

核心创新是两大模块：复杂度自适应推理深度（CARD）和空间推理掩模（SRM），实现按指令动态调整推理步骤与关注区域。

**🔧 技术方法**

采用流匹配视频扩散骨干，结合跨注意力提取掩模、关键词规则预测复杂度以及可插拔的RPFI特征注入。

**📊 数据集**

在737条ImgEdit Basic-Edit Suite和30条小样本基准上进行评估。

**📈 对比分析**

与基准ChronoEdit-Think（固定10步推理）对比，平均速度提升1.18倍，CLIP-T提升0.7%，LPIPS和CLIP-I保持相近；对不同编辑类别的加速差异达1.52倍。

**⚠️ 局限性**

局限包括对跨注意力掩模依赖、关键词规则对复杂指令识别不足、以及在高复杂度编辑下速度略低于基准。

---

## 370. FinSafetyBench: Evaluating LLM Safety in Real-World Financial Scenarios

**arXiv ID:** 2605.00706 | [PDF](https://arxiv.org/pdf/2605.00706v1)

**作者:** Yutao Hou `[一作]` (Shanghai University of Finance and Economics), Yun Chen `[通讯]` (Shanghai University of Finance and Economics)

**通讯引用:** 9976 | [OpenAlex ID](https://openalex.org/A5077845043)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个双语金融红队基准FinSafetyBench，用于评估LLM在金融合规违规请求上的拒绝行为。

**💡 创新点**

创新点在于基于真实金融犯罪案例和伦理违规标准的细粒度14子类分类，提供双语对齐、可公开的红队数据，并系统评估攻击与防御效果。

**🔧 技术方法**

使用LLM抽取、重写、翻译等技术；采用三种主流攻击（PAIR、ReNeLLM、FlipAttack）和三种防御（Self‑Reminder、ICD、Fin‑Guard）；评估指标为ASR。

**📊 数据集**

数据集来自司法案例、CFA模拟考试、TRIDENT、HarmfulQA、HEx‑PHI，共1881条双语实例。

**📈 对比分析**

与六个开源模型（LLaMA‑3、InternLM3、GLM‑4、Mistral、Qwen2.5、XuanYuan）及两大闭源模型（GPT‑5.1、DeepSeek‑V3.2）对比，发现金融犯罪类攻击成功率高、中文更易突破，防御效果有限。

**⚠️ 局限性**

局限在仅支持英中两语、生成过程可能引入偏差、仅评估选定模型与攻击方法，未覆盖所有新技术。

---

## 371. Learning to Act and Cooperate for Distributed Black-Box Consensus Optimization

**arXiv ID:** 2605.00691 | [PDF](https://arxiv.org/pdf/2605.00691v1)

**作者:** Zi-Bo Qin `[一作]` (South China University of Technology), Wei-Neng Chen `[通讯]` (South China University of Technology)

**通讯引用:** 19838 | [OpenAlex ID](https://openalex.org/A5050385116)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了LAC-MAS框架，通过大语言模型提供稀疏的高层指导，结合自适应的粒子群执行层，实现分布式黑盒一致性优化中代理内部行动与外部协作的自我设计；

**💡 创新点**

创新点在于：①利用历史优化轨迹由LLM生成的行动与协作策略；②在粒子群动态中嵌入自适应内部机制；③引入分阶段认知调度以在不同优化阶段定时更新指导；

**🔧 技术方法**

核心技术包括：大语言模型高层指导、粒子群自适应执行层、轨迹驱动的协作加权、分阶段认知指导（PCG）以及分布式一致性优化的理论保障；

**📊 数据集**

实验数据集为十个标准黑盒优化基准（100维、20代理）以及无线传感网络多目标定位任务；

**📈 对比分析**

与MASOIE、GFPDO、RGF、DAPSO等基线比较，LAC-MAS在大多数基准上实现了更低的最终适应度、加快了误差收敛速度并显著降低了通信成本；

**⚠️ 局限性**

局限性包括：需要LLM推理资源与预实验校准，仍限定于固定通信拓扑，对简单或单峰问题提升有限，并且高层指导更新的延迟可能影响极端动态环境下的收敛稳定性。

---

## 372. Eliminating Hidden Serialization in Multi-Node Megakernel Communication

**arXiv ID:** 2605.00686 | [PDF](https://arxiv.org/pdf/2605.00686v1)

**作者:** Byungsoo Oh `[一作]` (Cornell University), Rachee Singh `[通讯]` (Cornell University)

**通讯引用:** 425 | [OpenAlex ID](https://openalex.org/A5071694147)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多节点 Mixture-of-Experts 推理中，提出一种新的通信优化方案以消除隐藏的序列化瓶颈，提升 megakernel 的吞吐量。

**💡 创新点**

创新点在于：① 解耦数据传输与信号，降低 fence 次数；② 将 ordering 交给 NIC 硬件，消除 proxy 阻塞。

**🔧 技术方法**

使用的技术包括：GPU 先发 RDMA、NVSHMEM、Libfabric/IBRC/IBGDA 接口、NIC 级别 fence 标志、分组信号（grouping）等。

**📊 数据集**

使用了前沿 LLM 模型 Qwen3‑30B、DeepSeek‑V3、GPT‑OSS‑120B，以及不同序列长度 256‑64K 的推理工作负载。

**📈 对比分析**

与原始 FlashMoE/Vanilla 以及 GPU‑direct IBGDA 进行比较，Libfabric 上实现最高 10.3× 速度提升，IBRC 上 2.47×，在多节点规模下实现近线性弱扩展。

**⚠️ 局限性**

局限性包括：对极大消息尺寸的 NIC 级别 flag 仍有一定延迟；需要对 NVSHMEM 进行修改；在极端专家负载不均衡时仍会产生额外开销。

---

## 373. Foundation AI Models for Aerosol Optical Depth Estimation from PACE Satellite Data

**arXiv ID:** 2605.00678 | [PDF](https://arxiv.org/pdf/2605.00678v1)

**作者:** Zahid Hassan Tushar `[一作]` (University of Maryland, Baltimore County), Sanjay Purushotham `[通讯]` (University of Maryland, Baltimore County)

**通讯引用:** 4816 | [OpenAlex ID](https://openalex.org/A5017846156)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文开发并验证了一种名为 ViTCG（Vision Transformer with Channel-wise Grouping）的模型，用以从 NASA PACE 卫星的高光谱 TOA 辐射数据中回归 Aerosol Optical Depth (AOD)。

**💡 创新点**

创新点在于将 channel‑wise grouping 与 Vision Transformer 结合，既有效利用高光谱数据的光谱冗余，又通过自注意力捕获空间长程依赖，从而在保持高精度的同时显著降低参数量和推理成本。

**🔧 技术方法**

使用了视觉 Transformer、channel‑wise grouping、Patch embedding、全局自注意力、轻量级解码器以及 MSE 损失，并与 AERONET 地面观测进行独立验证。

**📊 数据集**

所用数据集包括 NASA PACE OCI Level‑1B 高光谱辐射和 Level‑2 AOD 产品，以及 AERONET 现场测量数据。

**📈 对比分析**

与 pixel‑wise 1D DNN、PrithviEO1、多光谱基础模型（HyperFree、HyperSigma、SpectralEarth）等基线比较，ViTCG 在 MSE 上下降 62%、IOA 达到最高值，并将模型参数量减少约 10 倍。

**⚠️ 局限性**

局限性包括仅在 PACE 数据上进行验证，缺乏跨时间、跨气溶胶类型的泛化能力；未将物理约束嵌入模型；高 AOD 区域仍存在小范围偏差。

---

## 374. Evaluating the Architectural Reasoning Capabilities of LLM Provers via the Obfuscated Natural Number Game

**arXiv ID:** 2605.00677 | [PDF](https://arxiv.org/pdf/2605.00677v1)

**作者:** Lixing Li `[一作]` `[通讯]` (University of Oxford), Lixing Li (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构造“Obfuscated Natural Number Game (O-NNG)”——对 Lean 4 自然数游戏中的标识符进行随机混淆，评估大型语言模型在完全无语义线索、仅凭局部公理与定义进行证明的能力，即 Architectural Reasoning。

**💡 创新点**

创新点在于：①将语义标识符剥离，形成零知识测试环境；②定义并量化“Architectural Reasoning”这一能力；③揭示了模型在去掉语义线索后出现的统一“latency tax”与鲁棒性差异，形成新的评测框架。

**🔧 技术方法**

技术手段包括字符级扰动算法（基于噪声水平 λ 的指数映射 P=λ^2.5），多阶段评测流水线（解析、混淆、查询生成、推理、验证、统计），以及对 Lean 4 编译器的自动验证。

**📊 数据集**

使用数据集：68 题的 O-NNG（分为 8 个模块）及其原版 NNG，涵盖了自然数游戏的所有基本公理与定理。

**📈 对比分析**

通过对五种主流 LLM（GPT‑4o、Claude‑Sonnet‑4.5、DeepSeek‑R1、GPT‑5、DeepSeek‑Prover‑V2）在 6 个噪声水平下分别做 5 次独立查询，计算正确率和平均推理时间；结果显示：所有模型在去除语义后都出现显著的时间增加，且仅 reasoning 模型保持正确率，general 模型正确率显著下降。

**⚠️ 局限性**

局限性：①混淆仅保留了 Peano 结构，可能未能完全阻止模型通过结构映射回原语义；②仅测试自然数领域，缺乏对更广泛数学域的验证；③实验受限于 API 调用，未覆盖需高性能集群的模型；④对噪声水平的非线性关系与“奇妙谷”现象的解释仍待进一步研究。

---

## 375. Make Your LVLM KV Cache More Lightweight

**arXiv ID:** 2605.00789 | [PDF](https://arxiv.org/pdf/2605.00789v1)

**作者:** Xihao Chen `[一作]` (National University of Singapore), Roger Zimmermann `[通讯]` (National University of Singapore)

**通讯引用:** 14111 | [OpenAlex ID](https://openalex.org/A5058575315)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关的 KV 缓存压缩方法 LightKV，利用跨模态提示引导的图消息传递在推理前填充阶段压缩视觉令牌，显著减少 KV 缓存大小。

**💡 创新点**

创新点在于：①使用文本提示提供的跨模态注意力权重作为压缩信息的引导，保持最相关视觉信息；②通过双分区图匹配和窗口化分块实现高效的特征差异（FD）计算；③采用层级化的窗口扩展策略，逐步聚合局部到全局信息。

**🔧 技术方法**

技术主要包括：双分区图构造与匹配、特征差异度量、基于注意力的消息聚合、窗口化分块、层级压缩调度（Λ、𝒲、𝒫）、基于预填充层的 KV 缓存截断。

**📊 数据集**

使用八种公开 LVLM 模型（LLaVA-v1.5/NeXT、InternVL2-8B、EVE、Qwen2.5-VL）在八个多模态基准集上评估：Coco Caption、GQA、MME、NoCaps、Pope、SeedBench、ScienceQA、VizWiz。

**📈 对比分析**

与 ToMe、ElasticCache、FastV、PiToMe、ToFu、HiRED 等前沿方法对比，LightKV 在大多数基准上保持甚至提升性能，55% 视觉令牌保留时 KV 缓存尺寸减半，计算量可降至 40%，在 Qwen2.5-VL 上 20-30% 视觉令牌保留即可保持 99% 的平均性能。

**⚠️ 局限性**

局限性：①双分区匹配限制每步最大 50% 压缩率，需多次迭代实现更高整体压缩；②需要完整注意力矩阵，导致在 FlashAttention 等 I/O 优化实现中不兼容，压缩层只能使用 eager 计算，略微增加实现复杂度。

---

## 376. SAVGO: Learning State-Action Value Geometry with Cosine Similarity for Continuous Control

**arXiv ID:** 2605.00787 | [PDF](https://arxiv.org/pdf/2605.00787v1)

**作者:** Stavros Orfanoudakis `[一作]` (Delft University of Technology), Pedro P. Vergara `[通讯]` (Delft University of Technology)

**通讯引用:** 1278 | [OpenAlex ID](https://openalex.org/A5070971243)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SAVGO（State–Action Value Geometry Optimization）算法，利用价值感知的余弦相似度学习状态–动作嵌入空间，并在此几何上进行相似度加权的策略更新；

**💡 创新点**

创新点在于将价值相关的几何结构直接嵌入到策略改进中，形成基于相似度的候选动作聚合器，三者（表示学习、价值估计、策略优化）在同一几何一致目标下统一训练；

**🔧 技术方法**

使用了off‑policy actor‑critic框架（SAC/TD3风格）、cosine相似度编码、带曲率参数λ的价值几何损失、可调温度ρ的相似度核、候选动作采样与加权聚合、Polyak平滑等技术；

**📊 数据集**

在MuJoCo连续控制基准（v5版本）上进行实验，包括Ant、HalfCheetah、Hopper、Humanoid、Walker2d等任务；

**📈 对比分析**

与PPO、TD3、SAC、TQC等强基线对比，在1M步预算下，SAVGO在高维任务（如Humanoid、Walker2d）取得显著提升，样本效率更高、收敛更稳定，整体性能优于对照方法；

**⚠️ 局限性**

局限性包括：需要为每个状态采样K个候选动作，导致额外计算开销；对Critic的准确性高度依赖，噪声会影响相似度权重；曲率λ、温度ρ、候选数K等超参对性能敏感；在极高维或离散动作空间中扩展困难。

---

## 377. Map2World: Segment Map Conditioned Text to 3D World Generation

**arXiv ID:** 2605.00781 | [PDF](https://arxiv.org/pdf/2605.00781v1)

**作者:** Jaeyoung Chung `[一作]` (Seoul National University), Kyoung Mu Lee `[通讯]` (Seoul National University)

**通讯引用:** 27274 | [OpenAlex ID](https://openalex.org/A5046504049)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了Map2World，一种基于TRELLIS的文本驱动3D世界生成框架，能够接受任意形状的分割图作为条件，生成大尺度且结构连贯的3D世界并进一步通过细节增强网络提升细节质量。

**💡 创新点**

创新点包括：① 在3D潜在空间引入多窗口潜在融合策略，实现跨块全局一致的生成；② 结合分割图的多标签潜在融合，支持任意形状的语义区域；③ 细节增强网络在保持全局结构的前提下通过MLP与流变换器局部细化细节；④ 对TRELLIS decoder进行微调，使其适配部分场景；⑤ 通过光谱参数化优化初始噪声，精准控制生成尺度。

**🔧 技术方法**

主要技术手段包括：TRELLIS的structured latent与rectified flow生成器；多窗口/多标签潜在融合算法；分割图引导的潜在融合；细节增强网络（MLP + TRELLIS流变换器）；初始噪声优化；decoder微调；评价指标GPTscore与自定义的World Quality（WQ）。

**📊 数据集**

使用了NuiScene43标签过滤的43个Objaverse场景，随机裁剪500个不同尺寸（64~256）的立方体，共提取17500个立方体，用于细节增强网络的训练和验证；实验中还生成了文本描述并对生成结果进行评测。

**📈 对比分析**

通过与SynCity、GaussianCube等基线模型对比，Map2World在分割图匹配、连贯性、细节与整体结构上表现更优；GPTscore平均得分7.93/10，超过SynCity的7.48/10；自定义WQ指标显示综合得分最高，视觉质量和结构完整性均优于对手。

**⚠️ 局限性**

局限性在于仍依赖预训练TRELLIS的先验，局部细节受潜在容量限制；多窗口合并导致计算量增加，生成大规模场景时效率较低；细节增强网络需额外微调，且缺乏针对更复杂语义或动态场景的验证；对真实世界大规模数据的泛化能力尚未充分评估。

---

## 378. LASE: Language-Adversarial Speaker Encoding for Indic Cross-Script Identity Preservation

**arXiv ID:** 2605.00777 | [PDF](https://arxiv.org/pdf/2605.00777v1)

**作者:** Venkata Pushpak Teja Menta `[一作]` `[通讯]` (Praxel Ventures), Venkata Pushpak Teja Menta (Praxel Ventures)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

训练了一种语言对抗式说话人编码器（LASE），通过梯度反转消除印地语、泰卢固语、泰米尔语与拉丁语之间的身份泄漏，显著提升跨脚本身份保持；

**💡 创新点**

创新点在于结合冻结的WavLM‑base-plus主干、监督对比损失与语言梯度反转分类器，利用合成跨脚本TTS配对实现仅用1118对样本就能关闭约84%的跨脚本身份差距，并且仅需100倍更少的数据即可匹配行业级ECAPA‑TDNN的跨脚本识别召回；

**🔧 技术方法**

采用了监督对比损失（SupCon）、梯度反转层（GRL）与语言四分类器、WavLM‑base-plus冻结主干+投影头，以及基于TTS生成的跨脚本配对数据；

**📊 数据集**

使用了8个ElevenLabs多语种语音在英语、印地语、泰卢固语、泰米尔语四种脚本下合成的1118对训练样本，以及分别包含1043对西方口音与1369对印度口音的留样本集；

**📈 对比分析**

与WavLM‑base-plus‑sv和ECAPA‑TDNN基线在三分布余弦相似度、跨脚本召回等指标上对比，LASE将跨脚本差距从0.082降至0.013（相对缩减84%），跨脚本召回0.788与ECAPA‑TDNN的0.789相当，且训练数据仅为后者的1/100；

**⚠️ 局限性**

局限在于仅使用合成TTS数据、缺乏新语音的泛化验证、脚本覆盖有限、未评估真实跨脚本语音或混合脚本语句的表现。

---

## 379. Themis: Training Robust Multilingual Code Reward Models for Flexible Multi-Criteria Scoring

**arXiv ID:** 2605.00754 | [PDF](https://arxiv.org/pdf/2605.00754v1)

**作者:** Indraneil Paul `[一作]` (Technische Universitaet Darmstadt), Iryna Gurevych `[通讯]` (University Of Wuerzburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含八种编程语言、五种代码质量维度（功能正确性、执行效率、内存效率、可读性与可维护性、安全性）的大型代码奖励模型评估基准，并基于GitHub提交与合成指令收集了超过50万条多维度代码偏好对；随后使用这些偏好数据训练了一个多语言、多维度的可扩展代码奖励模型（大小从1B到8B参数不等）。

**💡 创新点**

① 生成了迄今最大规模的公开代码偏好数据集；② 将多维度偏好与多语言语料结合，首次实现了可在任意子集维度上评分的多维奖励模型；③ 通过在预训练阶段引入通用人类偏好、在微调阶段使用显式维度提示以及多任务与正则化相结合的训练策略，显著降低了多维度互干扰。

**🔧 技术方法**

使用基于对比偏好学习的Bradley‑Terry目标，加入条件语言模型正则化和奖励幅度正则化；采用多阶段预训练+微调策略；在推理时通过可自定义的维度提示实现多维度评分；使用多语言预训练模型Qwen3作为基础。

**📊 数据集**

① GitHub公开仓库的单文件提交（含功能/非功能改动）和合并请求；② 通过前沿大型语言模型自动生成的合成指令与反向指令；③ 现有竞赛式代码生成数据集（功能正确性、执行效率、内存占用等）；④ 与公开的自然语言与代码偏好数据混合构建的预训练数据集。

**📈 对比分析**

与多类现有奖励模型（包括通用型、代码专项、生成式、推理式等）在本基准上进行对比，采用偏好准确率作为评测指标。结果显示：新训练的模型在所有五个维度上均优于现有模型，规模越大性能越好；在功能正确性上与最强现有模型相当或更优；在跨语言迁移与对抗鲁棒性测试中也表现出色，能够在无训练维度的情况下保持高准确率。

**⚠️ 局限性**

仍依赖人工与合成标注，可能存在标注噪声；对极低资源语言的覆盖不足；在极端复杂的非功能性维度（如安全性）仍有提升空间；当前模型对多语言的干扰虽已降低，但在极少数语言上仍出现性能波动。

---

## 380. Position: agentic AI orchestration should be Bayes-consistent

**arXiv ID:** 2605.00742 | [PDF](https://arxiv.org/pdf/2605.00742v1)

**作者:** Theodore Papamarkou `[一作]` (PolyShape), Alexey Zaytsev `[通讯]` (BIMSA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出在agentic AI系统的控制层引入贝叶斯决策框架，将LLM视为非贝叶斯的预测器，构建贝叶斯控制器来维护对任务相关潜在变量的信念并基于这些信念做决策（如查询、路由、停止、预算分配）。

**💡 创新点**

创新点在于：① 把贝叶斯原则从模型内部迁移到系统控制层，形成“贝叶斯控制层”而非“贝叶斯LLM”；② 通过可靠性权重、观察模型、价值信息权衡等机制，实现在不完整、相关或偏差的证据下的稳健更新；③ 提出了通用设计模式和示例（多智能体代码生成、讨论、工具路由），为实际部署提供模板。

**🔧 技术方法**

使用的技术主要包括：贝叶斯决策理论、后验概率更新（含观测模型与可靠性权重）、价值信息（value‑of‑information）决策、伪似然/复合似然更新、在线可靠性权重学习、温度化/依赖感知的证据聚合；同时讨论了与强化学习、稳健控制等替代框架的比较。

**📊 数据集**

该论文为位置论文，未使用特定公开数据集；通过理论推导与概念性案例（代码生成、讨论、工具路由）来说明方法，并未在真实数据上进行实验。

**📈 对比分析**

由于无实测实验，本文未给出性能数值；作者强调未来需设计能够衡量任务成功、信念校准和信息效率（如VOI指标）的基准，并指出当前方法的优势在于能量化不确定性、降低错误/无效调用、支持人机协作。

**⚠️ 局限性**

局限性包括：① 观测模型可能被误设，导致后验失准；② 证据间的统计相关性（共享模型/提示）可能破坏独立假设；③ LLM本身非贝叶斯，信念更新依赖于对LLM输出的经验校准；④ 计算开销与实时性需要进一步评估；⑤ 目前仅为概念性和示例，缺乏系统化实验验证。

---

## 381. Reconstruction of glymphatic transport fields from subject-specific imaging data, with particular emphasis on cerebrospinal fluid flow and tracer conservation

**arXiv ID:** 2605.00730 | [PDF](https://arxiv.org/pdf/2605.00730v1)

**作者:** A. Derya Bakiler `[一作]`, Shaolie S. Hossain `[通讯]` (University of Texas at Austin)

**通讯引用:** 528 | [OpenAlex ID](https://openalex.org/A5027214501)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一套基于质量守恒的逆向建模框架，利用受噪声影响的 CE-MRI 时空数据恢复大鼠脑 glymphatic 系统中的 CSF 速度、扩散率和清除参数，并在前向仿真中验证其物理一致性。

**💡 创新点**

创新点包括：① 将 Hughes‑Wells 的粗细尺度速度分解应用于逆向问题，使非散度自由的实验速度场通过弱散度自由修正得到物理可行的速度；② 采用浸入式等几何分析（IGA）与 THB‑spline 基函数实现自动平滑、无网格对接；③ 在逆问题中同时估计速度、扩散和边界清除参数，并通过 Newton 迭代实现快速收敛。

**🔧 技术方法**

技术手段：有限元法、浸入式 IGA、THB‑spline 二次基函数、Hughes‑Wells 速度分解、SUPG 稳定化、Newton‑Raphson 逆向求解、梯度/雅可比矩阵推导、C^1 连续性实现的自动正则化。

**📊 数据集**

使用的数据集为小鼠脑的对比剂增强磁共振成像（CE‑MRI）时空序列（5 min 采样间隔、0.3 mm 体素分辨率），并基于 DW‑MRI 生成扩散系数映射作为先验。

**📈 对比分析**

与传统的有限差分速度估计以及保守/非保守的前向方案相比，HW‑改造的前向模拟在保持质量守恒的同时能消除非物理的源汇；逆向求解得到的速度/扩散/清除场在 0.1 以内的相对误差下与实验 CE‑MRI 数据吻合，并能在独立验证窗口中保持较低误差。相比先前的有限差分速度恢复，整体误差下降约 70 %。

**⚠️ 局限性**

局限性：① 假设速度场和扩散/清除参数为时间不变，无法捕捉瞬时波动；② 逆向求解对初值敏感但在实验范围内稳健，仍需更广泛的参数空间验证；③ 受限于图像分辨率和噪声，仍可能存在微尺度结构被平滑；④ 只考虑表面清除（Robin 条件），未包含潜在的体内清除机理；⑤ 计算成本高（约 2.5×10⁴ 个未知数），对更大规模或三维多尺度模型有挑战。

---

## 382. Repurposing Image Diffusion Models for Adversarial Synthetic Structured Data: A Case Study of Ground Truth Drift

**arXiv ID:** 2605.00788 | [PDF](https://arxiv.org/pdf/2605.00788v1)

**作者:** Adam Arthur `[一作]` (Rochester Institute of Technology), Christopher Schwartz `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 12 | [OpenAlex ID](https://openalex.org/A5034814193)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用未改动的 Stable Diffusion U‑Net，将 UCI Adult 表格行重塑为 10×11 的单通道伪图像，从而生成符合统计分布的表格数据。

**💡 创新点**

展示即使没有专门的表格生成器或模型微调，公开图像扩散模型也能产生通过列级统计审核的合成表格，并首次系统评估其在不同特征布局下的逻辑一致性和下游效果。

**🔧 技术方法**

技术包括 Stable Diffusion v1.5 的 U‑Net、特征归一化与 one‑hot 编码、三种特征布局（任意、相关聚类、手工语义分组）以及 SDMetrics、规则检查、TSTR（Train‑on‑Synthetic Test‑on‑Real）和隐私泄露度量。

**📊 数据集**

使用 UCI Adult Income 数据集（约 48k 行）作为训练与评估基准。

**📈 对比分析**

相较于基线，聚类布局在 SDMetrics 上达到 86.6%（统计拟合好），手工布局将行级逻辑错误从 70% 降至 12%；在下游收入分类任务中，TSTR 对多数类 F1≈0.87，少数类仍低至 0.06‑0.25；隐私泄露评分低，表明未记忆训练样本。

**⚠️ 局限性**

主要局限在于行级语义错误仍普遍存在，模型无法直接学习复杂逻辑约束；仅靠图像扩散模型的空间局部性难以完全替代专门的表格生成器；攻击易于执行但需进一步改进以降低逻辑缺陷。

---

## 383. Learning the Helmholtz equation operator with DeepONet for non-parametric 2D geometries

**arXiv ID:** 2605.00760 | [PDF](https://arxiv.org/pdf/2605.00760v1)

**作者:** Rodolphe Barlogis `[一作]` (Université Perpignan Via Domitia), Stéphane Grieu `[通讯]` (Université Perpignan Via Domitia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在二维弹性 Helmholtz 散射问题中，构建了一个基于 Physics‑Informed DeepONet 的轻量级代理模型，用签名距离函数（SDF）编码非参数几何并直接从 PDE 损失学习散射场；

**💡 创新点**

创新点在于：① 仅依赖物理约束无监督训练；② 通过在分支网络输入 SDF 样本，实现在非参数几何下的几何感知；③ 结合 DeepONet 结构实现几何可插值和快速推理；

**🔧 技术方法**

采用的技术包括：Physics‑Informed Neural Network（PINN）、DeepONet 架构、自动微分、SDF 采样、TensorFlow 与 DeepXDE 框架；

**📊 数据集**

使用的“数据集”是自生成的：5 个旋转角度（-30°、-10°、0°、10°、30°）的同一基础几何，SDF 采样 10 个点，15,000+ 体积采样点与边界点；

**📈 对比分析**

与 FreeFem++ 计算的 FEM 参考解进行对比：在训练角度范围内误差可接受（归一化误差约 0.4‑0.5），在未见角度（±60°）误差显著升高；推理速度远快于 FEM；

**⚠️ 局限性**

局限性：仅在训练覆盖范围内具有良好泛化；需要手动选择 SDF 采样点数量与分布；对高频、三维、含多散射体或多材料界面问题的适用性待验证；训练时对 PDE 残差敏感，收敛可能受频率影响。

---

## 384. Complete Integration of Team Project-based Learning into a Database Syllabus

**arXiv ID:** 2605.00736 | [PDF](https://arxiv.org/pdf/2605.00736v1)

**作者:** S. Iserte `[一作]` (Universitat Jaume I), L. A. García `[通讯]` (Universitat Jaume I)

**通讯引用:** 8556 | [OpenAlex ID](https://openalex.org/A5100645568)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

在两门高级数据库课程中实施并评估团队项目式学习（TPBL）

**💡 创新点**

创新点是将项目式学习与团队合作相结合，形成完整的TPBL框架并配套同行评估工具

**🔧 技术方法**

使用项目式学习、团队合作、同行评估rubric、Google Forms+Corubrics、数据库管理系统（DBMS）等技术

**📊 数据集**

以学生自行设计的企业案例数据库作为项目数据集，没有使用公开数据集

**📈 对比分析**

通过同行评估、学生满意度调查及最终成绩比较，TPBL课程平均分从7.9/10提升至8.0/10，学生满意度与成绩均显著提升

**⚠️ 局限性**

局限性包括教师工作量大、团队数量过多时管理困难、项目交付时间与考试冲突、部分核心活动执行不一致导致成果差异

---

## 385. Weisfeiler Lehman Test on Combinatorial Complexes: Generalized Expressive Power of Topological Neural Networks

**arXiv ID:** 2605.00725 | [PDF](https://arxiv.org/pdf/2605.00725v1)

**作者:** Jiawen Chen `[一作]` (Southeast University), Wenwu Yu `[通讯]` (Southeast University)

**通讯引用:** 26191 | [OpenAlex ID](https://openalex.org/A5100627758)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了组合复合体 Weisfeiler–Lehman (CCWL) 测试及其神经网络实现 CCIN，统一处理图、超图、单纯形复合体和细胞复合体，并给出统一的可表达性理论与实验验证。

**💡 创新点**

创新点在于：①定义四类邻接关系并证明仅上/下邻接即可实现全 CCWL 的表达力；②构造 CCWL 的理论框架；③基于 CCWL 开发出更具表达力的 CCIN，并在多种高阶结构上超越现有方法。

**🔧 技术方法**

采用组合复合体定义、四类邻接关系、基于 HASH 的多重集哈希更新、可注入聚合函数、基于上/下邻接的消息传递等技术，并与传统 GNN、MPSN、CWN 等基线进行对比实验。

**📊 数据集**

使用的实验数据集包括合成强正则图、TUDataset（MUTAG、PTC、PROTEINS、NCI1/109、IMDB-B/M）、Reddit-B/M、MOLHIV、ZINC-small/FULL、Peptides-Func/Struct 等。

**📈 对比分析**

在图分类、回归等任务上与多种基线（GIN、MPSN、CWN、TopologicalNet 等）进行性能对比，CCIN 在大多数数据集上取得最高或相近最佳准确率/ROC‑AUC/MSE/MAE，显著优于传统方法，验证了其更强的表达力。

**⚠️ 局限性**

局限性包括：对极大规模图的可扩展性尚未充分评估；不同邻接关系存在冗余，某些数据集上去除部分邻接反而提升效果；目前对边特征和节点属性的依赖尚有限，且理论主要聚焦上/下邻接，未充分探究更高阶邻接的潜在优势。

---

## 386. Unpaired Image Deraining Using Reward-Guided Self-Reinforcement Strategy

**arXiv ID:** 2605.00719 | [PDF](https://arxiv.org/pdf/2605.00719v1)

**作者:** Yinghao Chen `[一作]` (National University of Defense Technology), Yaowen Fu `[通讯]` (National University of Defense Technology)

**通讯引用:** 1818 | [OpenAlex ID](https://openalex.org/A5100521989)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无监督降雨去雨框架RGSUD。

**💡 创新点**

创新点在于使用VLM驱动的动态奖励回收与自强化学习策略。

**🔧 技术方法**

采用DACLIP-IQA评估奖励、GAN结构、NAFNet或其他基网络、DEG模块等技术。

**📊 数据集**

使用Rain100L、Rain200L、DID-Data、DDN-Data、SPA-Data、RealRain1K-L、Night-Rain以及Unpaired SIRR、Real3000等数据集。

**📈 对比分析**

与多种无监督/监督方法对比，RGSUD在PSNR/SSIM及无参考IQA上均达到SOTA，提升约1–1.5dB。

**⚠️ 局限性**

局限在于对初始模型质量敏感，需先行有较好去雨结果才能产生有效奖励。

---

## 387. CustomDancer: Customized Dance Recommendation by Text-Dance Retrieval

**arXiv ID:** 2605.00824 | [PDF](https://arxiv.org/pdf/2605.00824v1)

**作者:** Yawen Qin `[一作]` (South Central Minzu University), Qin Zhang `[通讯]`

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 CustomDancer 框架及 TD‑Data 数据集，用于将自然语言查询与音乐同步的 3D 舞蹈动作进行检索。

**💡 创新点**

创新点在于：①构建了大规模对齐的文本-音乐-动作数据集；②设计了基于 CLIP 的文本编码器、Transformer 音乐与动作编码器，并引入音乐‑动作融合模块，实现跨模态的语义与节奏对齐；③通过多模态对比学习提升检索性能。

**🔧 技术方法**

技术细节包括：CLIP 文本编码器 + 两层 MLP 适配器；Librosa 提取的 35 维音频特征 + 交错的 Transformer 与 1D 下采样；SMPL 3D 动作编码器同样采用 Transformer；音乐‑动作融合采用加法与乘法交互后全局平均池化；使用 CLIP 对比损失训练。

**📊 数据集**

使用 TD‑Data 数据集，约 4000 条 12 秒 3D 舞蹈片段（总计 14.6 小时），覆盖 22 种舞蹈类型，包含专业舞者表演、音乐音频及经过专家注释的自然语言描述。

**📈 对比分析**

在 Recall@1/5/10、Median Rank、Mean Rank 等指标上，CustomDancer 超越 XPool 与 TABLE 基线，Recall@1 提升至 10.23%，Recall@10 最高达 48.34%，并在用户偏好实验中获得最高的文本‑动作一致性与文本‑音乐相关性分数。

**⚠️ 局限性**

局限性包括：①专业术语稀缺导致文本编码偏差；②音乐与动作不一致时检索误差；③表演者风格偏差可能被误认为种类标签；④数据仅包含 3D 动作与音乐，缺少服装、舞台、面部表情等视觉信息；⑤对文化语境的捕捉有限。

---

## 388. HyCOP: Hybrid Composition Operators for Interpretable Learning of PDEs

**arXiv ID:** 2605.00820 | [PDF](https://arxiv.org/pdf/2605.00820v1)

**作者:** Jinpai Zhao `[一作]` (University of Texas at Austin), Clint Dawson `[通讯]` (University of Texas at Austin)

**通讯引用:** 11699 | [OpenAlex ID](https://openalex.org/A5089253713)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出HyCOP框架，学习基于查询的模块化PDE求解器，通过对短程序的组合策略实现快速、可解释、可迁移的多物理模拟

**💡 创新点**

创新点在于将物理过程拆解为可复用原语（数值、神经或学习闭包），用少量参数学习的组合策略实现自适应调度，并提供理论的错误分解与诊断

**🔧 技术方法**

使用进化策略训练低维策略，组合子流（RK4、谱扩散、反应闭包等），支持混合字典与时间查询

**📊 数据集**

使用2D可压缩Navier‑Stokes（PDEBench）、二维浅水方程、二维阿德-扩散-反应（Fisher‑KPP）以及K–S方程等公开数据集

**📈 对比分析**

与FNO、DeepONet、Loc‑Int‑Diff‑FNO、U‑Net、PINO、Poseidon、Strang拆分等基线对比；HyCOP在OOD下实现一至两位数的误差降低、10×以上训练速度提升、25×更少前向传播、推理速度至少与Poseidon持平，且在大时间步长或异常边界条件下保持稳定

**⚠️ 局限性**

对字典的手工设计要求较高；学习的闭包或预训练网络在分布外的鲁棒性有限；在极高分辨率或完全无结构分解的PDE上可扩展性仍需验证

---

## 389. Persistent Visual Memory: Sustaining Perception for Deep Generation in LVLMs

**arXiv ID:** 2605.00814 | [PDF](https://arxiv.org/pdf/2605.00814v1)

**作者:** Siyuan Huang `[一作]` (Shanghai AI Laboratory), Yu Cheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9791 | [OpenAlex ID](https://openalex.org/A5101580521)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Persistent Visual Memory (PVM) 模块，解决大型视觉语言模型在长文本生成过程中的视觉信号衰减问题。

**💡 创新点**

创新点在于：① 在 Transformer 解码器中以并行分支方式插入轻量级检索通道，完全隔离自回归上下文对视觉注意力的竞争；② 采用独立注意力归一化，仅对固定视觉 token 进行 Softmax，消除文本长度对视觉权重的影响；③ 通过门控融合与视觉静默掩码实现视觉检索结果的可控注入。

**🔧 技术方法**

使用技术包括：Transformer 结构、交叉注意力、低维瓶颈投影、门控残差融合、两阶段训练（SFT+GRPO），以及 LogitLens 机制分析模型内部收敛行为。

**📊 数据集**

数据集涵盖：OpenMMReasoner‑SFT、MMK12、ThinkLite‑VL‑hard、ViRL39K、We‑Math2.0‑Pro；评测基准包括 MMMU、MMBench‑CN、MMBench‑EN、MMStar、MMT、MathVerse、MathVision、AI2D。

**📈 对比分析**

与内部基线、视觉注入方法（MemVR、ICoT、CoMemo）以及 RL 推理模型（Euclid‑8B、PEARL‑8B、OneThinker‑8B）对比；在 4B/8B 规模上分别提升 4.4%/4.8% 的总体分数，长序列/复杂推理任务的优势尤为明显，且显著加速内部预测收敛。

**⚠️ 局限性**

局限性：1) 仅在 Qwen3‑VL‑系列上验证；2) 轻量级模块虽参数增量低，但仍需额外推理开销；3) 依赖固定视觉编码，无法动态更新或跨模态（如音频、文本多模态）；4) 对极端长文本的鲁棒性仍需进一步探索。

---

## 390. Prop-Chromeleon: Adaptive Haptic Props in Mixed Reality through Generative Artificial Intelligence

**arXiv ID:** 2605.00804 | [PDF](https://arxiv.org/pdf/2605.00804v1)

**作者:** Haoyu Wang `[一作]` (Dyson School of Design Engineering), Ludwig Sidenmark `[通讯]` (University of Toronto)

**通讯引用:** 763 | [OpenAlex ID](https://openalex.org/A5017120886)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一套基于生成式 AI 的混合现实系统，能够通过用户文本提示把现实中的任意物体实时转换为形状对齐的虚拟被动触觉道具。

**💡 创新点**

创新点在于：①将深度相机捕获的物体形状与 ControlNet 控制的 Stable Diffusion 生成的 2D 图像相结合；②利用单视图 3D 重建（TripoSR）快速得到 3D 资产；③通过 Vuforia 6DoF 跟踪实现实时锚定，实现了从物体捕捉到触觉效果的全自动、可定制化流程，突破了传统手工映射或静态库的局限。

**🔧 技术方法**

使用技术包括：ZED Mini 深度摄像机、Stable Diffusion 2.1 + ControlNet、TripoSR 单视图 3D 生成、Vuforia 6DoF 实时跟踪、Unity 渲染，以及云端 API（HuggingFace Spaces、Replicate）进行模型推理。

**📊 数据集**

数据集：ShapeNetCore v2 50 个多样化物体用于生成实验；公开生成数据集 Prop-Chromeleon_Dataset（400 个提示 + 800 条生成结果），用于技术评估和用户研究。

**📈 对比分析**

比较方法：技术评估采用 Chamfer 和 Hausdorff 距离衡量形状相似度，并用 Q1–Q3 三维二进制编码评估提示的语义匹配；用户研究采用 HX 量表问卷、访谈，并与基线（不做形状对齐的静态生成）对比。结果显示形状一致率约 90%，生成成功率高于基线；用户在形状对齐版本上显著偏好，感知真实度、沉浸度提升约 30% 以上。

**⚠️ 局限性**

局限性：①整体生成延迟约 20 秒，主要受云端推理影响；②单视图重建对视角、光照敏感，易产生失真；③纹理与真实触感不匹配会削弱可信度；④系统只处理固体物体，无法支持可变形或多部件的对象；⑤安全性考虑不足，可能导致误导用户将危险物体误解为安全对象。

---

## 391. Generating Statistical Charts with Validation-Driven LLM Workflows

**arXiv ID:** 2605.00800 | [PDF](https://arxiv.org/pdf/2605.00800v1)

**作者:** Pavlin G. Poličar `[一作]`, Blaž Zupan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于LLM的分阶段工作流，用于从公共表格数据自动生成统计图表，并生成对应的代码、描述和问答对。

**💡 创新点**

创新点在于将图表生成拆分为可检验的子任务（数据筛选、图表建议、代码合成、渲染检查、迭代修正、描述与问答生成），并通过渲染后验证实现对可读性和语义匹配的即时纠错，生成与图表完整对齐的多模态数据集。

**🔧 技术方法**

技术上采用大型语言模型（如Qwen3.5-27B）进行多轮提示式生成与检查，结合Python绘图代码执行与可视化渲染，以及LLM评判器对问答结果进行自动判分。

**📊 数据集**

数据集使用UCI机器学习仓库中包含200–2000条记录、最多2000列的表格，筛选后共生成74个数据集、1,500张图表以及30,003条问答对。

**📈 对比分析**

对16种多模态LLM进行下游评测，图表语法类问题几乎饱和（最高≈99%），但数值提取、比较和推理问题仍相对困难（最高≈92%和90%），并通过图表族和问题类型的细粒度分析展示各模型的系统性弱点。

**⚠️ 局限性**

局限性包括：依赖UCI数据导致样本偏倚；渲染检查仅捕捉可见的可读性与语义错误，未保证数据与问答完全正确；自动判定器给出近似准确度，无法替代人工评测；工作流仍缺乏完整的人机交互验证与更严格的错误诊断。

---

## 392. RunAgent: Interpreting Natural-Language Plans with Constraint-Guided Execution

**arXiv ID:** 2605.00798 | [PDF](https://arxiv.org/pdf/2605.00798v1)

**作者:** Arunabh Srivastava `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (NEC Laboratories America, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 RunAgent 平台，实现了基于自然语言计划的多代理逐步执行，并通过约束验证与错误修正确保输出正确；

**💡 创新点**

创新点在于将自然语言与可执行控制结构结合，自动生成并验证约束，灵活切换 LLM、工具、代码执行，并在每步进行上下文过滤；

**🔧 技术方法**

采用 GPT‑4o 作为核心 LLM，配合自定义代码生成与执行、工具调用、约束检查、错误纠正等技术；

**📊 数据集**

在 Natural‑plan Calendar Scheduling、Trip Planning 数据集和 SciBench（Stat、Calc、Diff）数学子集上进行评估；

**📈 对比分析**

与 GPT‑4o、Gemini‑1.5‑Pro/2.0‑Flash 及 PlanGEN 的 baseline 进行对比，RunAgent 在 Calendar Scheduling 的 Exact Match 取得 81.1%（升至 86.2% 纠正后），Trip Planning 80.6% 以上，SciBench 约 14.73%（远高于 GPT‑4o 3.07%）等显著提升；

**⚠️ 局限性**

局限在于依赖 GPT‑4o 的推理能力，计划生成仍未重点优化，对多解场景匹配精度有限，且系统对超大规模计划与实时约束更新的适应性尚待提升。

---

## 393. Let ViT Speak: Generative Language-Image Pre-training

**arXiv ID:** 2605.00809 | [PDF](https://arxiv.org/pdf/2605.00809v1)

**作者:** Yan Fang `[一作]` (Beijing Jiaotong University), Yunchao Wei `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 21839 | [OpenAlex ID](https://openalex.org/A5087043856)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种最小化的生成式视觉语言预训练框架 GenLIP，利用单一 Transformer 对图像与文本进行早期融合并直接用自回归语言建模目标训练视觉编码器。

**💡 创新点**

创新点在于去除传统的两塔对比学习和文本解码器，仅使用一个前缀 LM 注意力机制和门控注意力，使视觉编码器与 MLLM 的推理目标高度一致，显著提升数据效率与可扩展性。

**🔧 技术方法**

核心技术包括单一 Transformer 结构、Prefix‑LM 注意力、Multimodal Rotary Position Encoding（MRoPE）、门控注意力、两阶段训练（低分辨率 224×224 8B 采样 + 高分辨率本地长宽比微调），以及层级缩放与 DropPath 归一化。

**📊 数据集**

使用的训练集为 Recap‑DataComp‑1B（约 1B 影文对，8B 次采样）以及 37M 的高分辨率长文本图像数据（Infinity‑MM‑Stage1 + BLIP3o‑Long‑Caption）。

**📈 对比分析**

与 CLIP、SigLIP、SigLIP2、AIMv2、OpenVision2 等对比实验显示，GenLIP 在 8B 采样下已在 7 个文档/ OCR、4 个通用 VQA 和 3 个 Caption 任务上取得均值 61.5（L/16）、62.6（So/16）、65.2（g/16）的综合得分，尤其在 OCR 任务提升 4–6 分，且在同等数据规模下明显优于更大规模的对比与生成式基线。

**⚠️ 局限性**

局限性包括：实验仅在 LLaVA‑NeXT 规模 MLLM 上验证，尚未验证对最新大型 MLLM 的通用性；预训练数据仅至 1B 规模，超大规模下的扩展性未知；依赖高质量长文本标题，获取成本较高。

---

## 394. A Faster Deterministic Algorithm for Fully Dynamic Maximal Matching

**arXiv ID:** 2605.00797 | [PDF](https://arxiv.org/pdf/2605.00797v1)

**作者:** Julia Chuzhoy `[一作]` (Toyota Technological Institute at Chicago), Junkai Song `[通讯]` (New York University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的确定性算法，用于在完全动态的图中维护最大匹配，更新时间为n^(1/2+o(1))。

**💡 创新点**

创新点在于引入了一个新的子图系统框架，专门用于验证和维护最大匹配的性质，克服了之前方法的局限性。

**🔧 技术方法**

使用了递归方法和子图系统的设计，结合了高效的边着色算法。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在动态图中进行边插入和删除的情况。

**📈 对比分析**

与之前的算法（如BBKS25）相比，新的算法在适应性对手下实现了更优的更新时间，性能显著提升。

**⚠️ 局限性**

算法的局限性在于其复杂性和对图的结构要求，可能在某些特定情况下表现不佳。

---

## 395. Posterior Augmented Flow Matching

**arXiv ID:** 2605.00825 | [PDF](https://arxiv.org/pdf/2605.00825v1)

**作者:** George Stoica `[一作]` (Georgia Tech), Judy Hoffman `[通讯]` (Georgia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Posterior-Augmented Flow Matching (PAFM)，通过对中间状态的后验分布进行采样并加权，取代单一目标监督，降低梯度方差，改进流匹配训练。

**💡 创新点**

创新点在于将稀疏的单点监督扩展为后验期望监督，证明无偏且梯度方差可下界，并通过重要采样实现可行的候选目标混合；可灵活结合最近邻、数据增强、VAE采样等多种候选策略。

**🔧 技术方法**

使用流匹配、可逆流、重参数化、重要采样与自归一化重要采样、条件概率路径、条件似然估计、FAISS 最近邻检索、CLIP/ViLM 语义匹配、MMDiT 与 SiT 生成模型、REPA、ODE 采样、CFG、FID/IS/precision/recall 等技术。

**📊 数据集**

实验数据集包括 ImageNet-1K（分类条件）和 CC12M（文本到图像），图像分辨率统一为 256×256。

**📈 对比分析**

通过与标准 FM 在不同模型规模（SiT-B/2、SiT-XL/2）、架构（SiT、MMDiT）上的对比，PAFM 在 FID50K 上提升约 3.4 点（ImageNet）或 1.9 点（CC12M），其余指标亦有改善；计算开销仅增加约 6.6%。

**⚠️ 局限性**

局限性包括需手工设计候选目标集合，最近邻检索或在线增强产生轻微额外计算；文本场景的条件似然仍采用简化近似；理论梯度方差下界依赖重要采样条件，实际表现受限；在高维空间中候选过多可能引入噪声；未探索更精细的后验估计方法。

---

## 396. Can Coding Agents Reproduce Findings in Computational Materials Science?

**arXiv ID:** 2605.00803 | [PDF](https://arxiv.org/pdf/2605.00803v1)

**作者:** Ziyang Huang `[一作]` (Johns Hopkins University), Daniel Khashabi `[通讯]` (Johns Hopkins University)

**通讯引用:** 5168 | [OpenAlex ID](https://openalex.org/A5043628255)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为 AutoMat 的基准，用于评估大语言模型（LLM）代理在材料科学论文中复现量化主张的能力。

**💡 创新点**

创新点在于将科学可复现性任务转化为可执行、可评估的工作流，结合从论文文本、从artifact、以及interpretation 三种重现类型，并使用 LLM 评审者对完整执行轨迹进行自动化评估。

**🔧 技术方法**

利用多种 LLM 代理（Claude Code、Codex CLI、Kimi 等）和 LLM 评审器，在 HPC 环境（Slurm + 容器化）中执行完整的实验工作流，并采集日志、命令与输出。

**📊 数据集**

使用 85 条由领域专家挑选的材料科学主张及其对应论文、元数据和可选 artifact，构成了 AutoMat 数据集，并在 Hugging Face 上公开。

**📈 对比分析**

通过对比五种代理设置，测得整体复现得分与成功率；最佳设置（Claude Code + Opus）平均得分 3.52，成功率 54.1%，其余模型表现更差，尤其在从论文文本重建工作流时几乎无成功案例。

**⚠️ 局限性**

局限性包括：样本仅覆盖 85 条材料科学主张，未覆盖不可复现或对抗性案例；评估主要依赖自动化 LLM 评审器，尚需更大规模的人工专家验证。

---

## 397. GMGaze: MoE-Based Context-Aware Gaze Estimation with CLIP and Multiscale Transformer

**arXiv ID:** 2605.00799 | [PDF](https://arxiv.org/pdf/2605.00799v1)

**作者:** Xinyuan Zhao `[一作]` (Guilin University of Electronic Technology), Reem Kateb `[通讯]` (Taibah University)

**通讯引用:** 405 | [OpenAlex ID](https://openalex.org/A5090099655)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为GMGaze的多尺度融合与Mixture‑of‑Experts（MoE）混合网络，用于鲁棒的单帧3D注视估计。

**💡 创新点**

创新点包括：①通过可学习的语义原型银行对CLIP全局嵌入进行条件化，生成两个互补的上下文感知全局token；②在Transformer第一层即进行早期统一多尺度token融合，避免传统晚期融合导致的信息损失；③在Transformer中使用稀疏MoE层实现令牌级动态容量分配，实现高效可扩展性；④结合对抗域适应与特征分离损失提升跨域泛化。

**🔧 技术方法**

技术主要涵盖：CLIP预训练视觉与文本编码器、ResNet‑50局部特征提取、语义原型条件化、早期全尺度token融合、稀疏Mixture‑of‑Experts Transformer以及梯度反转层和特征分离正则。

**📊 数据集**

在四个公开基准（MPIIFaceGaze、EYEDIAP、Gaze360、ETH‑XGaze）上进行评估，并在跨域设置中测试源→目标的迁移。

**📈 对比分析**

与现有SOTA方法相比，GMGaze在所有四个基准的内部评估中分别取得2.49°、3.22°、10.16°、1.44°的角误差，均优于之前最优模型；在跨域迁移中亦获得了如Et→M、G→M等路径的最佳或接近最佳性能。

**⚠️ 局限性**

局限性包括：①语义原型银行的初始化来自手工设计的CLIP提示，限制了上下文建模的连续性；②硬选择的原型更新可能无法捕捉细腻的环境变化；③当前模型仅基于单帧，缺乏时间一致性；④对抗域适应在高差异源与低多样性目标时效果有限。

---

## 398. When RAG Chatbots Expose Their Backend: An Anonymized Case Study of Privacy and Security Risks in Patient-Facing Medical AI

**arXiv ID:** 2605.00796 | [PDF](https://arxiv.org/pdf/2605.00796v1)

**作者:** Alfredo Madrid-García `[一作]` (Independent researcher), Miguel Rujas `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 63 | [OpenAlex ID](https://openalex.org/A5092438672)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过两阶段非破坏性安全评估，揭示了公开可访问的患者面向医疗RAG聊天机器人存在的配置泄露、知识库枚举和对话记录存储等安全与隐私缺陷。

**💡 创新点**

创新点在于利用商业LLM辅助生成探测提示并通过普通浏览器工具复核，证明LLM可以高效协助安全评估，且系统级配置泄露是关键风险。

**🔧 技术方法**

技术包括Claude Opus 4.6的prompt‑inject探测、Chrome开发者工具网络抓包、手工验证等。

**📊 数据集**

使用的数据为公开部署的RAG聊天机器人所处理的临床教育文档（8份）、嵌入向量、以及近千条患者-聊天记录。

**📈 对比分析**

本文未与其他方法对比性能，而是通过对比公开说明与实际部署差异，指出系统未满足隐私声明的情况。

**⚠️ 局限性**

局限在于仅评估了单一匿名部署，缺乏对多样化RAG实现的普适性验证，并且未公开具体漏洞复现细节以防误用。

---

## 399. When LLMs Stop Following Steps: A Diagnostic Study of Procedural Execution in Language Models

**arXiv ID:** 2605.00817 | [PDF](https://arxiv.org/pdf/2605.00817v1)

**作者:** Sailesh Panda `[一作]` (Indian Institute of Technology Gandhinagar), Mayank Singh `[通讯]` (Indian Institute of Technology Gandhinagar)

**通讯引用:** 1205 | [OpenAlex ID](https://openalex.org/A5100746903)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个受控诊断基准，用来评估大语言模型在执行给定的多步算术程序时的准确性，并测试了14种模型。

**💡 创新点**

创新点在于将算法长度和回溯依赖作为独立维度设计基准，揭示了最终答案准确率掩盖了模型对指令逐步执行的显著缺陷。

**🔧 技术方法**

技术上采用了确定性执行器生成真值、贪婪/低温解码、答案提取与多种评价指标（首答案准确率、Correct@Any、非空答案率、步骤执行情况）进行分析。

**📊 数据集**

使用了55个子数据集，总计55,000个实例，涵盖三种数值范围、整数/浮点、四种算术类型、七种回溯深度和5至95步的算法长度。

**📈 对比分析**

通过在所有数据集上对比14个模型，评估首答案准确率与步骤执行比例；结果显示从5步的约61%下降到95步的约20%，并随回溯深度增加进一步下降，表明规模更大的模型仍难以保持完整的步骤执行。

**⚠️ 局限性**

局限性在于仅测试了算术运算任务，难以说明在更复杂或多样化推理任务中的通用性；并未探讨外部工具或增强方法的潜在改进。

---

## 400. Univalence without function extensionality

**arXiv ID:** 2605.00812 | [PDF](https://arxiv.org/pdf/2605.00812v1)

**作者:** Evan Cavallo `[一作]` (University of Gothenburg and Chalmers University of Technology), Jonas Höfer `[通讯]` (University of Gothenburg and Chalmers University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文通过对范式 Glehn 的多项式模型构造进行分析，证明了一个更弱的 univalence axiom（称为“分类学 univalence”）并不能推出函数外延性，从而给出了 +_{cat} ⊬ +_{funext} 的独立性结果；随后对该弱 axiom 的多种变体进行比较，并展示了它们之间的非等价性。

**💡 创新点**

创新点在于：①首次将多项式模型用于拆解 univalence 与函数外延性之间的关系；②引入“分类学 univalence”和“族化分类学 univalence”两种更弱的 univalence 形式，并证明它们与传统 univalence 及函数外延性互不蕴含；③通过构造切片模型和调整宇宙解释，进一步区分了不同弱 univalence 的严格程度。

**🔧 技术方法**

主要技术包括：多项式模型（Polynomial Model）构造、野类别（wild category）的形式化、对等价性与同构性的细致分层（如严格同构 vs 同伦同构）、对“η 法则”和“有限余积的充分性”的使用、以及对切片模型和反射子类别的理论分析。

**📊 数据集**

本工作属于纯理论研究，不涉及任何实验数据集。

**📈 对比分析**

对方法的比较主要体现在理论层面：作者证明了 +_{cat} 与 +_{funext} 的互不蕴含，以及 +_{famcua}、+_{cua} 等变体与其他 axiom 之间的关系；没有数值实验或性能指标。

**⚠️ 局限性**

限制：①证明依赖于多项式模型的存在与其对函数外延性的已知反例，尚未扩展到更一般的模型（如模拟型模型或其他类型构造）；②对弱 univalence 的分类尚未给出统一的框架；③在没有 η 法则或有限余积满足性时，结果可能不适用；④本文未探讨弱 univalence 在可计算性或构造性方面的实际影响。

---

