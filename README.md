# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-02 | 今日论文总数: 576

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Optimal any-angle path planning in static and dynamic environments

**arXiv ID:** 2607.00065 | [PDF](https://arxiv.org/pdf/2607.00065v1)

**作者:** Yiyuan Zou `[一作]` (Delft University of Technology), Clark Borst `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于椭圆前向扩展和视野扫描（阴影投射）的任意角路径规划框架，分别实现了 Zeta*（静态环境）和 Zeta*-SIPP（动态环境）以获得最优路径；

**💡 创新点**

通过将椭圆前向扩展与视野扫描结合，设计了逆向/正向扫描两种变体，既保证最优性又显著加速搜索，并可轻松扩展至动态和非均匀成本地图；

**🔧 技术方法**

使用椭圆前向扩展、阴影投射视野、逆向/正向扫描、SIPP（安全区间规划）以及八/四象限阴影投射等技术实现算法；

**📊 数据集**

在 Moving AI Lab 提供的九套静态地图（BGII、DA2、DAO、WC3、SC、City、Maze、Random、Room）和六套动态障碍地图（Empty、Random-64-64-10、Room-64-64-16、Maze-128-128-10、Warehouse-10-20-10-2-2、Berlin_1_256）上进行了实验；

**📈 对比分析**

与 A*（2^k 邻域）、Theta*、Anya、TO-AA-SIPP 等基线比较，评估运行时、排序元素数和扫描节点数；Zeta*-f 在动态环境中平均比 TO-AA-SIPP 快约 24 倍，Zeta* 在静态环境中性能仅略逊于 Anya；

**⚠️ 局限性**

局限性包括：在实时动态场景中仍不如 AA-SIPP 快速；在狭窄通道地图下椭圆搜索扩展产生额外开销；目前未在非均匀代价地图和三维环境上进行验证。

---

## 2. Prompting GPT-5 on Scrum Certification Questions: An Empirical Accuracy Study

**arXiv ID:** 2607.00049 | [PDF](https://arxiv.org/pdf/2607.00049v1)

**作者:** Mirko Perkusich `[一作]` (VIRTUS/UFCG), Angelo Perkusich `[通讯]` (VIRTUS/UFCG)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对GPT‑5在Scrum认证题目的回答准确性进行实验评估，比较不同提示技术的效果

**💡 创新点**

首次系统地将三种提示策略（零射、链式思维、源引用）用于结构化的敏捷认证测试

**🔧 技术方法**

使用OpenAI GPT‑5 API并实现自动化提示执行与答案解析

**📊 数据集**

采用993道已验证的PSM‑I对齐题目数据集，涵盖多选、真伪、单选三种题型

**📈 对比分析**

通过精确答案匹配对三种提示进行比较，发现源引用提示最高准确率89.1%，比零射略优

**⚠️ 局限性**

局限在于仅使用单一模型、单一版本的Scrum指南，且对解释性、跨框架的普适性缺乏验证

---

## 3. Making Failure Safe: A Constrained, Verifiable Agent Framework for Open-Web Data Collection

**arXiv ID:** 2607.00035 | [PDF](https://arxiv.org/pdf/2607.00035v1)

**作者:** Bo Chen `[一作]` `[通讯]` (Chinese Academy of Sciences), Bo Chen (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套受约束的代理收集框架，将LLM生成从自由代码转为JSON配置，并通过类型化收集任务、模板限制、工具复用和验证反馈闭环实现可验证的网页数据采集。

**💡 创新点**

创新点在于提出六类型收集器分类并与模板、工具、验证三层约束结合；采用JSON配置而非代码降低LLM运行时成本；引入规则化质量检查与结构化反馈循环，提升可审计性和可复用性。

**🔧 技术方法**

利用LLM（DeepSeek‑V4‑Flash）进行需求理解与槽位填充，使用Airflow静态DAG执行，借助JSON Schema/Pydantic约束，实施规则化质量评分与反馈约束，并预编写工具函数模板。

**📊 数据集**

自建138任务核心集（覆盖六类收集器），其中80任务为源验证集，每任务包含自然语言描述、期望字段和收集器类型。

**📈 对比分析**

与规则模板、无约束LLM、Runtime LLM + Crawl4AI、ScrapeGraphAI等基线在质量、通过率、token数、时延上对比；本框架零执行阶段token、最低5s时延下取得0.5631平均质量、50%通过率，反馈修正后质量提升至0.901。

**⚠️ 局限性**

局限在于使用构造任务而非真实用户请求；缺乏长期多轮调度实验与显著性检验；反馈修复仅针对已知缺陷，未实现通用自我修复；受网站结构变化、交互复杂度和身份认证限制。

---

## 4. Joint Medical Image Enhancement and Segmentation with Diffusion-based Symbiotic Information Interaction

**arXiv ID:** 2607.00058 | [PDF](https://arxiv.org/pdf/2607.00058v1)

**作者:** Ying Chen `[一作]` (Chinese University of Hong Kong), Qiankun Li `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种双分支DDIM扩散网络DiSIINet，实现医学图像增强与分割的联合学习

**💡 创新点**

通过Symbiotic Information Interaction（SII）模块实现增强与分割任务在特征层面进行双向信息交流，互相促进

**🔧 技术方法**

采用DDIM扩散模型、VAE编码解码、跨注意力（cross‑attention）SII模块以及联合损失训练

**📊 数据集**

在多模态医学数据集ACDC（心脏MRI）、KiTS19（CT肾结石）和TN3K（超声甲状腺结节）上验证

**📈 对比分析**

与单独或序列处理方法相比，DiSIINet在PSNR/SSIM、mIoU/Dice等指标上显著提升（如PSNR 32.15/SSIM 0.932，mIoU 84.81/Dice 91.78），显示更优性能

**⚠️ 局限性**

模型计算复杂度较高，需要更多扩散步骤，未来需压缩模型或自适应减少采样步数

---

## 5. Verifiable Rewards for Calibrated Probabilistic Forecasting

**arXiv ID:** 2607.00164 | [PDF](https://arxiv.org/pdf/2607.00164v1)

**作者:** Sadanand Singh `[一作]` (Cascade Research), Manan Chopra `[通讯]` (Cascade Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

使用可验证奖励的强化学习训练小型语言模型，使其在无人工标签的条件下，预测NFL比赛中期权概率，且与体育博彩市场的校准相当；

**💡 创新点**

提出无标签、可验证的状态条件经验胜率奖励，消除单一结果噪声，并通过梯度遮蔽将奖励仅作用于答案，从而避免链式思维导致的校准失真；

**🔧 技术方法**

采用群体相对策略优化的RL框架，结合LoRA适配器、直接预测与答案掩码两种梯度抑制方式，以及Brier分数等严格适当的评分规则；

**📊 数据集**

利用2015‑2022年NFL常规赛的公开 play‑by‑play 数据训练，2023年调参，2024年独立测试，奖励基于训练集的经验胜率；

**📈 对比分析**

与未训练模型、零样本前沿模型 DeepSeek‑V4、tabular 经验率估计器及 nflverse 统计模型对比，训练模型的 ECE 与市场持平（0.029 vs 0.027），Brier 仅比市场高 0.008，推理一致性错误率从 22.4% 降至 4.4%；

**⚠️ 局限性**

仅在公共状态信息已包含大部分预测信号且事件频繁可观测时有效；稀疏或长期事件会削弱奖励信号；模型仍受限于经验率分辨率，无法捕获实时赛场信息。

---

## 6. LLMs in the Real World: Evaluating "AI" in Emergency Contexts

**arXiv ID:** 2607.00019 | [PDF](https://arxiv.org/pdf/2607.00019v1)

**作者:** Sara Court `[一作]` (Ohio State University), Micha Elsner `[通讯]` (Ohio State University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过对美国911中心部署的基于LLM的文本翻译系统进行案例研究，分析其在紧急服务中的应用、误解与风险，并提出改进建议。

**💡 创新点**

提出将AI系统的透明度与伦理规范整合到高风险公共服务中的方法，系统阐述语言技术误解及责任缺口，强调研究者在安全部署中的积极角色。

**🔧 技术方法**

使用基于LLM的机器翻译与语言检测技术（结合Microsoft Azure云服务），评估其在多语种文本-911通信中的表现。

**📊 数据集**

主要使用现场实际文本示例（如阿拉伯语和尼泊尔语短信）及城市的通话记录（约670,000次呼叫），未公开大规模标准MT评测数据。

**📈 对比分析**

未进行系统定量评估，仅通过实地交互和专家访谈发现翻译在非标准文字和方言上的错误率高，缺乏实时人类校对，性能不足以满足急救需求。

**⚠️ 局限性**

研究范围仅为单一案例，缺乏统计分析与真实使用数据，未能获取底层模型信息，无法全面评估性能与安全性。

---

## 7. Synergistic Perception-Reasoning Governance: Grounding Medical MLLMs with Verifiable Anatomical Evidence

**arXiv ID:** 2607.00060 | [PDF](https://arxiv.org/pdf/2607.00060v1)

**作者:** Rui Hao `[一作]` (Huazhong University of Science and Technology), Zhigang Zeng `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种训练无关的证据注入框架，通过提取可验证的解剖 ROI，将其双侧注入到医学多模态大语言模型（MLLM）的文本提示和视觉激活中，辅以任务感知动态路由以抑制幻觉；

**💡 创新点**

① 利用 MedSAM 自动获取解剖 ROI 作为双侧可验证证据，弥合视觉感知与临床推理的鸿沟；② 设计视觉侧激活软掩码与文本侧结构化注入的联合策略；③ 引入零开销的任务感知动态路由，根据问题类别动态选择最合适的注入方式；

**🔧 技术方法**

MedSAM 进行 ROI 提取、ROI‑guided 视觉激活重校准、文本提示结构化注入、零‑shot LLM 问题分类用于动态路由，以及在多种预训练 MLLM 上实施推理时间校正；

**📊 数据集**

MM‑VisHal（包含 SLAKE 与 VQA‑RAD）、CXR‑VisHal（包含 IU‑Xray 与 MIMIC‑CXR）以及 MIMIC‑CXR 报告集，用于闭合式 VQA 与开放式报告生成的评估；

**📈 对比分析**

与原始模型及单侧注入（文本侧、视觉侧）和联合侧四种策略进行对比，采用闭合式准确率、开放式幻觉指标（CHAIR、CheXbert、RadGraph、RaTEScore、Recall）进行评估；结果显示闭合式任务平均提升约 1.9%（最高 3.3%），视觉侧在解剖类问题最优，联合侧在放射学类问题最优；开放式任务幻觉下降约 35%，召回率提升，整体性能优于现有基线；

**⚠️ 局限性**

依赖 MedSAM 的 ROI 质量，ROI 提取错误会导致注入失效；仅适用于具有解剖标注的图像，难以推广到无 ROI 或非结构化影像；无训练方式无法在所有模型/任务上保证最佳效果；动态路由器依赖零‑shot分类的准确性，分类误判可能影响注入选择。

---

## 8. SemiScope: Disentangling Classifier Tuning and Joint Optimization in Semi-Supervised Security Classification

**arXiv ID:** 2607.00113 | [PDF](https://arxiv.org/pdf/2607.00113v1)

**作者:** Rui Shu `[一作]` (North Carolina State University), Jingzhu He `[通讯]` (ShanghaiTech University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对二分类安全任务中的半监督学习流程进行系统拆解并实现 SemiScope 统一的联合搜索框架，评估其与默认 SSL、监督模型和等预算分类器调优的对比。

**💡 创新点**

通过等预算等阈值协议分离 SSL 与分类器调优的贡献，发现大部分提升来自分类器 HPO，而非 SSL 本身，提出可复现的拆解与评估框架。

**🔧 技术方法**

使用 Bayesian Optimization（TPE via Optuna）进行联合搜索，配合经典 SSL（Label Propagation/Label Spreading/Self‑Training）、SMOTE 目标比例、树基分类器（Random Forest、XGBoost、LightGBM）以及自适应阈值调优。

**📊 数据集**

五个二分类安全基准：CIC-IDS-2017、Drebin、NSL-KDD、Phishing、UNSW-NB15。

**📈 对比分析**

采用相同验证集阈值、10% 标签率下的 g‑measure 进行比较，SemiScope 超越所有默认 SSL 基线；在 4/5 数据集与等预算 Tuned‑Clf 统计等价；在 20–30% 标签率可接近全标注模型的性能。

**⚠️ 局限性**

仅适用于二分类表格安全任务、树模型和经典 SSL；未涵盖深度学习、多分类或非表格特征，阈值和置信度解释受限，外推性需谨慎。

---

## 9. Urban Deceleration Behavior Modes Under Scene Context: An Early-Kinematic Classifier from Argoverse 2 Multi-Agent Trajectories

**arXiv ID:** 2607.00027 | [PDF](https://arxiv.org/pdf/2607.00027v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 10. Invariant Stochastic Filtering on SE(3) for Inertial-Encoder State Estimation of Serial Rigid Manipulators

**arXiv ID:** 2607.00026 | [PDF](https://arxiv.org/pdf/2607.00026v1)

**作者:** S. Yaqubi `[一作]` (Tampere University), J. Mattila `[通讯]` (Tampere University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于Lie群SE(3)的不可变扩展卡尔曼滤波器，用于多连杆刚性机械臂的状态估计。

**💡 创新点**

创新点在于：①利用姿态运动的群仿射性质使误差动力学自适应，Riccati方程直接描述真实误差协方差；②引入物理分离的IMU噪声模型，区分陀螺仪与加速度计，并通过Δt尺度调整得到有效的速度测量协方差；③通过共轭矩阵（Adjoint）实现多连杆间的协方差传播，保持每个连杆的计算成本为O(1)，从而整体线性O(n)；④在协方差传播中加入状态相关的科里奥利噪声，捕捉陀螺仪误差通过非线性动力学传播的影响；⑤利用Lie代数Lyapunov函数给出指数最终有界性证明，并给出跨连杆的稳健性链式界限。

**🔧 技术方法**

使用的主要技术包括：Lie群几何滤波（IEKF）、群仿射误差模型、共轭矩阵（Adjoint）协方差传播、Itô解析、离散化Van Loan方法、Joseph形式的协方差更新、NEES一致性检验。

**📊 数据集**

通过在二维三维运动的3自由度两连杆机械臂上进行数值仿真，利用已知的物理噪声参数（陀螺仪、加速度计、编码器、外部力矩噪声）作为数据集。

**📈 对比分析**

与基于欧拉角/四元数的坐标EKF以及无滤波原始编码器读取进行对比。IEKF在所有关节角度上实现了约33%至40%的RMSE降低，坐标EKF在较大运动幅度下出现线性化误差导致性能下降，而IEKF保持稳定。NEES曲线显示估计协方差保持一致，表明滤波器在统计上是保守且可靠的。

**⚠️ 局限性**

局限性包括：①理论假设（可观测性与误差小的条件）未在所有运动轨迹下严格验证；②缺乏对IMU偏置的估计与补偿；③仅在仿真环境中验证，尚未在真实机器人平台上完成实验；④对高维连杆系统的可扩展性虽在理论上可线性实现，但实际实现中可能受到计算与通信瓶颈影响。

---

## 11. Learning User-Aware Recall: Personalized Retrieval in Long-Term Conversational Memory

**arXiv ID:** 2607.00017 | [PDF](https://arxiv.org/pdf/2607.00017v1)

**作者:** ZhiShu Jiang `[一作]` (Baidu), Jizhou Huang `[通讯]` (Baidu)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向长期对话的检索框架PPRO，先将对话拆分为情节记忆、语义记忆并提炼用户画像，然后在检索时将用户画像嵌入评分，最后通过GRPO训练查询重写器以提升检索和答案质量。

**💡 创新点**

创新点在于将用户画像作为个性化先验直接注入检索评分，结合检索质量与答案质量的双重奖励使用Group Relative Policy Optimization（GRPO）对查询重写器进行强化学习，形成闭环优化的检索-生成管线。

**🔧 技术方法**

技术包括层次化记忆构造（情节提取、语义聚类、画像摘要）、检索时结合查询与画像相似度的混合评分、检索维度加权（时态、主观性、确定性），以及使用GRPO的查询重写器。

**📊 数据集**

使用LoCoMo和LongMemEval‑S这两个公开的长记忆对话问答基准进行实验。

**📈 对比分析**

与训练无关的基线（Full‑Context、MemGPT、A‑Mem、LightMem、Mem0、SimpleMem）以及训练有关的基线（MEM1、MemAgent）进行对比。PPRO在LoCoMo上整体F1提升7–19个百分点，在LongMemEval‑S上准确率位居最高，显著优于所有基线。

**⚠️ 局限性**

局限性包括：在静态、离线的基准上评估，未处理流式增量更新；GRPO的检索奖励依赖人工标注的证据；框架目前仅为单用户，缺乏多用户或社区交互下的共享上下文机制。

---

## 12. SWE-Router: Routing in Multi-turn Agentic Software Engineering Tasks

**arXiv ID:** 2607.00053 | [PDF](https://arxiv.org/pdf/2607.00053v1)

**作者:** Seongho Son `[一作]` (University College London), Ilija Bogunovic `[通讯]` (University College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SWE-Router，一种基于值函数的时序路由框架，先让弱模型执行若干探索步骤，然后根据部分轨迹决定是否升级到强模型完成任务。

**💡 创新点**

创新点在于：①利用中间交互轨迹而非仅凭任务描述进行路由，突破prompt‑only 路由的 Bayes 错误底线；②给出贝叶斯最优性证明，证明在探索信息有价值时条件化轨迹永不损害路由，且会显著提升性能。

**🔧 技术方法**

主要技术包括：ReAct‑style 多轮 LLM 交互；训练一个轻量级值头（使用 LoRA 微调 2‑分类头）预测剩余成功概率；基于成本‑收益阈值的决策规则；以及对弱模型的多轮探索与成本计数。

**📊 数据集**

使用的数据集为 SWE‑Smith 与 SWE‑Bench（Verified 子集）进行训练、验证与测试。

**📈 对比分析**

与 prompt‑only 的 Logistic、k‑NN、XGBoost 以及 K=0 的非时序路由进行对比；实验显示 SWE‑Router 在路由 AUC 上提升至少 12 个百分点，且在多数成本范围内实现了 Pareto 前沿，甚至在某些阈值下优于全强模型。

**⚠️ 局限性**

限制点：需要额外的弱模型探索开销；路由效果高度依赖弱模型的探索能力；若未充分训练值头可能导致错误路由；此外，该方法可能被滥用于规避安全性限制，需进一步结合安全路由策略。

---

## 13. A Mechanism-Driven Theory of Phase Transitions in Active Learning

**arXiv ID:** 2607.00144 | [PDF](https://arxiv.org/pdf/2607.00144v1)

**作者:** Julia Machnio `[一作]` (University of Copenhagen), Mostafa Mehdipour Ghazi `[通讯]` (University of Copenhagen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过重新解释PAC风险分解为动态机制，构建了主动学习的机制驱动三阶段理论，并用可观测代理和分段回归来识别阶段转移，从而解释不同主动学习策略（代表性、覆盖、不确定性）在不同阶段的优势。

**💡 创新点**

创新点在于：1）证明主动学习过程中主导风险组件的转移是结构性必然的；2）提出将代表性、覆盖与不确定性与分布差异、几何覆盖、模型复杂度、置信度等四个机制对应，并通过代理跟踪其相对重要性；3）使用分段回归自动检测转移点，形成数据驱动的三阶段分类；4）揭示表示质量（如SSL预训练）如何提前触发阶段转移，压缩早期阶段。

**🔧 技术方法**

使用技术包括：PAC风险分解与四项风险组件（经验风险、分布差异、模型复杂度、置信度）; 设计的代理（ER、LD、FD、GC、Comp、Conf）; 分段（piecewise-linear）回归与BIC模型选择来定位转移点; 主动学习实验框架（pool-based，随机采样、TypiClust、Coreset、ProbCover、Entropy、BADGE、UHerding、TCU切换策略）；SSL表示学习（SimCLR、BYOL、MoCo）与线性分类器训练。

**📊 数据集**

实验数据集包括自然图像集（CIFAR‑10、CIFAR‑100、ImageNet子集）和医疗影像集（ISIC），并在每个数据集上分别使用监督训练与多种SSL预训练表示进行比较。

**📈 对比分析**

在每个数据集上与随机采样比较，代表性方法在阶段 I（数据驱动）获得最高早期准确率；覆盖方法在阶段 II（过渡）取得显著提升；不确定性方法在阶段 III（模型驱动）实现最佳最终准确率。利用TCU切换策略（在分段检测到的转移点自动切换策略）在多数基准上均能超过任何单一静态策略，且在计算效率上也优于昂贵的代表性/覆盖方法。

**⚠️ 局限性**

局限性包括：1）代理只能捕捉相对重要性而非精确风险值；2）假设特征表示固定且仅适用于pool‑based主动学习；3）分段回归假设风险变化近似线性，可能在极端高噪声或大规模数据中失效；4）缺乏在线自适应算法，仅提供事后切换验证；5）未覆盖多模态或主动学习之外的采样场景。

---

## 14. Constructive Alignment: Governing Preference Dynamics in Human-AI Interaction

**arXiv ID:** 2607.00001 | [PDF](https://arxiv.org/pdf/2607.00001v1)

**作者:** Max Kanwal `[一作]` (Stanford University), Caryn Tran `[通讯]` (Northwestern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

本文提出了构建式对齐（Constructive Alignment）框架，将AI对齐从静态偏好匹配转向对人类偏好动态演化的控制问题。

**💡 创新点**

创新点在于将偏好视为分层、动态、通过交互构建的过程，并提出五个元偏好约束（内在一致性、反思认可、受限影响、认知完整性、不确定性下的赋权）来规范AI影响。

**🔧 技术方法**

技术上采用控制理论和动态马尔可夫决策过程（DR-MDP）对偏好状态进行建模，并在控制优化中引入约束以保证元偏好。

**📊 数据集**

由于是理论框架性工作，未使用具体实验数据集，而是基于文献综述和案例讨论推导。

**📈 对比分析**

未给出实验比较，主要通过与现有的目标优化、社会选择、多主体对齐方法对比，论证了对齐问题的不足并说明了新框架的必要性。

**⚠️ 局限性**

局限性包括对偏好层次、距离度量、基准策略等建模假设未完全确定，需进一步经验验证与算法实现。

---

## 15. Bayesian updates from coalgebraic determinisation

**arXiv ID:** 2607.00034 | [PDF](https://arxiv.org/pdf/2607.00034v1)

**作者:** Manuel Baltieri `[一作]` (Araya Inc.), Nathaniel Virgo `[通讯]` (University of Hertfordshire)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文证明了无条件化（unifilarisation）是结构化Mealy机的 coalgebraic determinisation 实例，说明在 Mealy 机器而非 Moore 机器的设置下可以自然得到贝叶斯滤波的更新，并得到更丰富的因果随机行为语义。

**💡 创新点**

创新点在于将 unifilarisation 与一般的 coalgebraic determinisation 对接，解释了为什么在 Mealy 机器中出现贝叶斯条件化，并通过引入支持 Monad（如分布 Monad）揭示了不同语义的本源，提供了一个统一的理论框架。

**🔧 技术方法**

使用了 coalgebraic determinisation、支持 Monad、分布 Monad、Mealy 机器、最终 coalgebra 语义、超正则化（hyper‑normalisation）等高级范畴论与概率论技术。

**📊 数据集**

本文未使用任何实验数据集，全部为理论推导和形式化证明。

**📈 对比分析**

通过理论比较：与传统的 Moores 型 determinisation 以及其最终 coalgebra semantics 对比，展示了在 Mealy 机器中贝叶斯更新自然出现且语义更完整；由于是理论工作，暂无数值性能评估。

**⚠️ 局限性**

局限性：仅适用于 Set 上有限支持的 Monad，无法直接推广到测度论或无限状态空间；对非支持 Monad 的情况缺乏完整描述，且理论结果仍需在具体应用中进一步验证。

---

## 16. SmoothAgent: Efficient Long-Horizon LLM-Based Agent Serving with Lookahead Context Engineering

**arXiv ID:** 2607.00151 | [PDF](https://arxiv.org/pdf/2607.00151v1)

**作者:** Zaifeng Pan `[一作]` (University of California, San Diego), Yufei Ding `[通讯]` (University of California, San Diego)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM‑based agents在执行多轮工作流时，由于上下文工程（如截断、摘要、分离、离线存储）导致的KV缓存失效和重新预填导致的时间开销，并提出了前瞻式（lookahead）编程模型与调度器来消除这些开销。

**💡 创新点**

创新点在于：①发现大多数上下文工程操作满足段可分解（segment‑decomposable）属性，可提前异步完成；②设计了无需修改现有agent框架的lookahead编程模型，支持异步执行上下文转换；③提出SLO‑aware的lookahead‑aware调度器，确保最佳努力（BE）请求不会侵扰关键路径（LC）请求；④将上述技术集成到实际的LLM服务系统（SGLang）和多种主流agent框架。

**🔧 技术方法**

技术主要包括：段可分解分析、前瞻式异步上下文转换、KV缓存预填（lookahead KV cache）以及基于KV缓存长度和token数的混合批量延迟预测模型。实现层面涉及runtime层（维护主状态和lookahead状态）与serving层的协同调度（prefill‑decode disaggregated / co‑located）。

**📊 数据集**

使用的模型是开源LLM Qwen3‑8B 和 Qwen3‑32B（最大上下文32K token）。实验数据来自自定义多轮代码分析任务（MiniAgent），并在NVIDIA H100 GPU上运行。

**📈 对比分析**

与同步实现（在关键路径上立即执行上下文转换）相比，实验显示：①在PD co‑located与disaggregated部署中，平均TTFT下降约62%（Qwen3‑8B）/61%（Qwen3‑32B）；②在摘要策略下可达11.9×的TTFT改进；③在多agent场景下，lookahead‑aware调度器显著抑制了BE请求对LC请求的干扰，保持LC TTFT/TBT在目标SLO范围内。

**⚠️ 局限性**

局限性包括：①仅适用于段可分解的上下文工程；②对非分解型转换（如需要跨段依赖的操作）无效；③仍需对不同部署（prefill‑decode disaggregated vs co‑located）进行细粒度调度，增加实现复杂度；④依赖现有KV缓存和prefill机制，若模型或架构变更需重新验证。

---

## 17. From "Strings" to "Things" for Personal Knowledge Graphs: Evaluating LLM Triple Extraction for Recommendation Systems

**arXiv ID:** 2607.00003 | [PDF](https://arxiv.org/pdf/2607.00003v1)

**作者:** Abhirup Dasgupta `[一作]` (Rensselaer Polytechnic Institute), Oshani Seneviratne `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出了一个基于提示式大语言模型的管道，用于从多轮推荐对话中抽取结构化的用户偏好三元组（主语、谓词、宾语），并将其构建成可与 Wikidata 互操作的个人知识图谱（PKG）；随后在基于 PKG 的推荐系统中评估抽取结果的下游效果。

**💡 创新点**

创新点包括①将轻量级开源 LLM（Qwen3、Gemma‑3）与精心设计的少量示例提示相结合，实现从非结构化对话直接生成 RDF‑兼容的三元组；②提出一种可复制、数据集无关的抽取与实体对齐流程；③在下游推荐任务中证明较小模型（如 Gemma‑1B）抽取的 PKG 能为低资源的推荐系统提供更稳健的上下文。

**🔧 技术方法**

技术要点包括：提示工程与 k‑shot 方式、基于 LLM 的三元组抽取、实体链接（对话中的电影名映射到 Wikidata QID）、RDF 语义映射、PKG 构建与 SPARQL 查询、以及在 FedTREK‑LM 框架下的推荐模型训练与评估。

**📊 数据集**

使用公开的 ReDial 推荐对话数据集（约 1.13 万条对话），该数据集包含人类标注的用户偏好、观看状态、推荐来源等结构化信息，用作抽取评测与下游任务的基准。

**📈 对比分析**

对抽取性能采用精确度、召回率和 F1 进行评估。Gemma‑12B 在所有关系上的 F1 最高（≈0.95），但在下游推荐中，Gemma‑1B 提供的 PKG 能保持 0.18 左右的 F1（相较于基准 0.20），表明较小模型在保持多样性与覆盖度时更具优势。10‑shot 提示导致过拟合，模型性能骤降。整体来看，抽取 F1 与下游推荐 F1 之间呈非线性关系，精确度对推荐更为关键。

**⚠️ 局限性**

局限性包括①抽取质量高度依赖提示设计与模型规模，部分大模型在回忆性关系上表现欠佳；②对话语义模糊、长距离依赖时抽取错误率上升；③实体链接在多义词或新电影时可能失效；④实验仅在电影推荐领域，迁移到其他领域需重新制定本体映射；⑤缺乏对多轮对话连续更新 PKG 的在线评估。

---

## 18. SkillSelect-Serve: Budget-Controllable and QoS-Aware Skill Service Recommendation and Composition for Small LLM Agents

**arXiv ID:** 2607.00011 | [PDF](https://arxiv.org/pdf/2607.00011v1)

**作者:** Jingyuan Zheng `[一作]` (Hangzhou Dianzi University), Shuguang Deng `[通讯]` (Zhejiang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将LLM代理技能从普通文本检索转变为服务化的Skill Service，通过需求解析、候选检索、双粒度服务效用建模和预算控制投影，实现在给定预算与QoS约束下的技能服务推荐与组合。

**💡 创新点**

创新点包括：
1) Skill‑as‑a‑Service 表述，将技能包装为包含功能、输入输出、工具依赖、成本、风险等属性的可发现、可组合服务；
2) 微型需求规划器将自然语言任务映射为结构化需求，避免直接让LLM选择技能；
3) 双粒度效用建模，既评估单个服务的边际适用性，又校准服务组合的整体覆盖、冗余、成本与风险；
4) 基于预算与QoS的投影机制，实现同预算下的多种运营模式（Compact、High‑Recall、Aggressive）。

**🔧 技术方法**

技术包括：
- 语义检索（词典+向量检索）+交叉编码重排序；
- 规则+轻量级文本分析构建Skill Service Profile；
- 需求解析的本地微型Agent Planner；
- 采用标签无监督特征的多层感知机/GBDT实现边际与组合效用估计；
- 预算约束下的投影优化（多目标加权）。

**📊 数据集**

数据集：
- 35,353条去重的技能条目（构成Skill Service Registry）；
- 586条自然语言任务查询（577可评估），包含717条正交互和52,647条训练对；
- 任务与技能的人工标注，用于评估Recall和Mean Utility。

**📈 对比分析**

与传统固定Top‑k检索、启发式变尺寸、纯神经包composer等方法比较。实验显示：
- 在相同服务预算下，Compact@3提升Recall从0.8163到0.8700、Mean Utility从0.6333到0.6901；
- Final@5提升Recall从0.8492到0.8873、Mean Utility从0.6672到0.7078；
- Aggressive@6在更大预算下进一步提升但成本与风险提高。整体接近候选空间上限（Top‑20≈0.8873），表明组合层已接近瓶颈。

**⚠️ 局限性**

局限与威胁：
- 评估主要基于人工正样本与离线效用，无法完全映射到真实执行性能；
- 依赖检索阶段的候选集，若正样本未检索到则无法恢复；
- 在不同部署情景下需手动选择预算模式，缺乏自适应决策；
- 目前未加入执行反馈进行在线效用学习，导致推荐–执行之间存在差距。

---

## 19. Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration

**arXiv ID:** 2607.00033 | [PDF](https://arxiv.org/pdf/2607.00033v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 20. Joint Discovery of Object and Action Symbols through Effect Prediction for Robotic Manipulation Planning

**arXiv ID:** 2607.00031 | [PDF](https://arxiv.org/pdf/2607.00031v1)

**作者:** Burcu Kilic `[一作]` (Bogazici University), Emre Ugur `[通讯]` (Bogazici University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过随机物理交互收集多模态效果数据，使用二值瓶颈的编码器-解码器网络联合学习对象和动作符号，并基于预测的效果轨迹构建离散效果库，利用A*搜索与符号条件低层控制器实现多步操纵规划，进一步通过少量交互对新对象进行基于行为的归类，完成对未知对象的快速泛化。

**💡 创新点**

① 采用二值瓶颈联合学习对象与动作符号，使符号直接与交互效果相关；② 两阶段学习策略先用触觉信息区分粗动作，再用空间信息细化方向；③ 基于预测的中间效果点实现部分动作执行，提升规划精度；④ 通过效果匹配实现少样本对象泛化，突破视觉相似性限制。

**🔧 技术方法**

深度编码器-解码器网络（对象卷积网络 + Transformer 动作编码器）+ Gumbel‑Sigmoid 二值化；多模态效果预测（位置、力、接触）；符号条件 CNMP 低层控制器；A* 搜索离散效果库；效果匹配实现少样本归类。

**📊 数据集**

在 PyBullet 模拟环境中收集 5 种物体（球、立方体、Block X/Y/Z）共 5000 条随机交互样本（每种 1000 条）；随后对 5 种未见物体（Torus、T‑Block、Cylinder、U‑Block、Wide Block）进行少样本实验。

**📈 对比分析**

与基准 Diffusion Policy 在桌面重定位（Task 1）和堆叠（Task 2）任务进行对比；在 Task 1 的欧氏误差和 Task 2 的成功率上，本文方法在所有对象上显著优于基准（均有统计显著性），并在未知对象上保持更高的精度与成功率。

**⚠️ 局限性**

仅处理单对象的单体能力，未考虑多对象关系；漂移重规划仅在执行后触发，缺乏执行前可行性验证；使用 A* 搜索，效率和内存受限，未与 PDDL 等规划语言兼容。

---

## 21. FLYNN: Robust Neural Network for Robot Navigation using Fly Brain Topology

**arXiv ID:** 2607.00025 | [PDF](https://arxiv.org/pdf/2607.00025v1)

**作者:** Benquan Wang `[一作]` (Mississippi State University), Jingdao Chen `[通讯]` (Mississippi State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并训练了一个完全受果蝇神经网络连接组支配的递归神经网络（FLYNN），用于视觉+风向+触觉传感的机器人导航任务。

**💡 创新点**

首次将完整的果蝇synaptic‑resolution connectome直接映射为可训练的RNN，展示了其天然模块化与冗余结构能提供显著的环境不确定性与感官缺失鲁棒性。

**🔧 技术方法**

采用leaky integrator神经元模型、DAgger模仿学习、PCA/KDE内部状态分析以及MuJoCo物理仿真，构造并训练FLYNN。

**📊 数据集**

使用FlyWire提供的FAFB v783果蝇连接组数据（约139k神经元、5.34M突触）以及MuJoCo仿真环境中的摄像头、风向和碰撞传感数据。

**📈 对比分析**

与EfficientNet、MobileNet（经摄像头dropout训练）及随机小世界网络进行对比；在标准棋盘纹理环境中FLYNN的成功率92.9%与SPL 0.83与基线相当；在真实纹理的OOD环境中成功率降至42%而基线仅17%；在完全失明情况下FLYNN仍保持44%成功率，而其他模型几乎为零。

**⚠️ 局限性**

模型仅保留连接拓扑，未包含尖峰神经元、电突触、神经递质多样性等生物学细节，导致与真实果蝇行为可能存在差距；此外，训练仍依赖人工设定的leaky integrator与简单传感映射，未能充分模拟生物神经动力学。

---

## 22. Aligning Sentence Embeddings to Human Concepts via Sparse Autoencoders

**arXiv ID:** 2607.00023 | [PDF](https://arxiv.org/pdf/2607.00023v1)

**作者:** Wonseok Shin `[一作]` (Yonsei University), Songkuk Kim `[通讯]` (Yonsei University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出利用 Top‑k 稀疏自编码器将句子变换器的稠密嵌入拆分为可解释的概念，并通过对特定隐层神经元的“clamp”实现检索过程的精细调节。

**💡 创新点**

核心创新在于将 Top‑k 稀疏约束直接应用于句子嵌入空间，既保留高信息量，又生成高度互异、单义的特征，进而实现无须重新训练主干模型即可的检索结果重排序。

**🔧 技术方法**

采用 Top‑k 稀疏自编码器（SAE）、硬性稀疏约束、权重共享编码/解码、激活抑制（clamping）以及 GPT‑4o‑mini 自动标签流水线。

**📊 数据集**

实验基于 WikiText‑103‑v1 数据集，使用 1,024 维 E5-large 句子嵌入作为输入。

**📈 对比分析**

通过对比解码器正交度（平均 0.0408）、解释方差（0.9259）与稀疏度指标，验证了模型在保持 92%+ 信息保留率的同时实现特征分离；在检索结果重排序实验中，手动抑制特定维度可将目标条目从第 1 名调至第 9 名或相反，证明了可控性。

**⚠️ 局限性**

仅在单一英文主干模型上验证，未对 k 与 d_latent 进行系统调优；稀疏层的可解释性依赖于自动标签质量，跨语言适用性尚未探测。

---

## 23. Scaling Up Thermodynamic AI Models

**arXiv ID:** 2607.00170 | [PDF](https://arxiv.org/pdf/2607.00170v1)

**作者:** Andrew G. Moore `[一作]` `[通讯]`, Andrew G. Moore

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文开发了一种纯反向传播的训练框架，使深度卷积网络能够在基于 Ising 模型的热力学计算机上进行高温 Gibbs 采样推理。

**💡 创新点**

创新点在于提出了 Gibbs 正则化训练方法，将高温 Gibbs 采样的时间平均与前向推理对应，并给出了误差理论、混合时间估计以及最优采样调度公式。

**🔧 技术方法**

使用的技术包括高温 Gibbs 采样、温度梯度参数 δ、固定点损失与幅值正则化、红黑色子采样、符号‑平均读取以及混合高斯模型误差估计。

**📊 数据集**

实验数据集涵盖 MNIST、FashionMNIST、CIFAR‑10 与 CIFAR‑100 四个常用图像分类基准。

**📈 对比分析**

通过与传统前向推理和先前的 Gibbs‑Ising 基准对比，CIFAR‑10 在 Gibbs 推理下达 94.9% 的准确率，CIFAR‑100 达 76%，相比前向推理仅损失几百分点，同时推理成本约 99.9% 归功于 Ising 块，实现了显著的能耗和计算量削减。

**⚠️ 局限性**

主要局限包括需要精细调节温度梯度 δ 与采样次数 G 以控制混合时间，对输入/输出的二值化限制，以及模型对权重稀疏化与量化的鲁棒性仍有限，实际硬件实现规模与可行性仍待进一步硬件对接验证。

---

## 24. Hate Speech Detection in Turkish and Arabic Languages: A Comprehensive Study

**arXiv ID:** 2607.00143 | [PDF](https://arxiv.org/pdf/2607.00143v1)

**作者:** Somaiyeh Dehghan `[一作]`, Berrin Yanikoglu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了涵盖土耳其语和阿拉伯语的全新多维度仇恨言论数据集，并使用BERT为基础的对比学习、回归、标签分类和标记检测模型，实现了仇恨分类、强度预测、目标识别和文本段落检测；

**💡 创新点**

创新点在于首次提供面向土耳其/阿拉伯语的多任务仇恨语料库、引入双对比学习提升多类别分类性能、利用ChatGPT进行零/少量学习式标签分割与合成数据增强；

**🔧 技术方法**

技术包括BERTurk/AraBERT预训练模型、双对比损失（Cross‑Entropy + Supervised Contrastive Loss）、回归损失、BCEWithLogits、BIO/IO标记方案，以及ChatGPT的零/少量学习分割与合成；

**📊 数据集**

使用的数据集包含10,953条土耳其语推文（移民、以巴冲突、土希关系、宗教/族群、LGBTI+五大主题）和2,510条阿拉伯语移民主题推文；

**📈 对比分析**

与单一交叉熵基线相比，DualCL在二分类、四分类和六分类中的宏F1提升约1.9%（土耳其）和5%（阿拉伯），回归RMSE分别为1.64/1.20，目标识别宏F1约0.73，段落检测F1为0.59（二分类）和0.31（多分类）；

**⚠️ 局限性**

局限性包括低标注一致性（Krippendorffα≈0.23）、阿拉伯语样本量小、对隐性仇恨和罕见目标类别识别性能偏低、段落多类别分类效果差，且模型依赖BERT预训练与手工标注集成。

---

## 25. CogTax: A Four-Level Cognitive Taxonomy for Command-Line Computing Education

**arXiv ID:** 2607.00140 | [PDF](https://arxiv.org/pdf/2607.00140v1)

**作者:** Manuel Alonso-Carracedo `[一作]` (Universidade de Vigo), Lorena Otero-Cerdeira `[通讯]` (Universidade de Vigo)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种四级认知分类体系 CogTax，用于描述命令行操作的认知难度与操作风险，并基于此设计了自动化的命令级别标注方法。

**💡 创新点**

创新点在于：①将 Bloom 的认知层级与操作影响维度融合并通过最大值规则定义最终等级；②结合抽象语法树（AST）结构特征与深度学习语义嵌入，实现对命令级别的自动推断；③采用决策层面最大值融合，进一步提升分类准确率。

**🔧 技术方法**

技术实现主要包括：AST 解析提取节点统计与 n‑gram 结构特征；四种预训练嵌入模型（Sentence‑BERT、MPNet、检索调优模型、Code‑aware 模型）结合不同预处理策略；线性 SVM 分类器；以及基于最大值的融合策略。

**📊 数据集**

数据集为 585 条 Linux/bash 命令，其中 117 条为真实考试题目，468 条为专家人工标注的合成样本，四级标签均已平衡分布。

**📈 对比分析**

在 5‑折交叉验证中，AST 仅方法准确率 66.3%，嵌入方法（最佳为 E5‑small+normalization）达到 88.5%，两者融合的最大值规则进一步提升至 89.2%，macro‑F1 亦从 0.669 提升至 0.892，表明两种表征互补且融合策略显著改善性能。

**⚠️ 局限性**

局限性包括：仍需人工标注以扩展到更大命令空间；对跨语言通用性验证不足；低级别（L1/L2）区分仍受限于语义表征；以及在实际教学中对误判风险的评估仍待进一步研究。

---

## 26. Active Sensing for RIS-Aided Tracking and Power Control: A Hybrid Neuroevolution and Supervised Learning Approach

**arXiv ID:** 2607.00056 | [PDF](https://arxiv.org/pdf/2607.00056v1)

**作者:** George Stamatelis `[一作]` (National and Kapodistrian University of Athens), George C. Alexandropoulos `[通讯]` (National and Kapodistrian University of Athens)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于双智能体（Dual-Agent）神经进化+监督学习的主动感知框架，用于在能量受限的移动终端上，利用可重构智能表面（RIS）实现高效定位与跟踪，并通过单比特反馈控制上行功率；

**💡 创新点**

创新点在于：①将神经进化与监督学习结合，解决离散RIS相位配置及单比特功率控制的非可微优化问题；②在分布式感知框架中引入两位协同智能体，突破信息瓶颈，实现近似等效于多比特或标量反馈的性能；③在多天线基站上实现同时调节RIS与数字波束的扩展。

**🔧 技术方法**

使用长短时记忆网络（LSTM）构建BS/UE策略与估计器，采用CoSyNE进化策略对BS与UE策略进行联合优化，随后在得到的策略下通过监督学习细化定位估计器；实验中还对比了EKF、PF、A2C、指纹识别与连续相位RIS等基准方法。

**📊 数据集**

使用仿真生成的RIS/BS/UE数据集，包括不同大小RIS、不同Ricean系数、不同噪声水平及多种移动模型（如协调转弯模型）的时序数据；未使用公开真实数据集。

**📈 对比分析**

与EKF、PF、A2C、指纹识别、全连续相位RIS等传统与学习方法比较，在跟踪和定位任务中均获得最低的RMSE；表现对噪声、转弯速率、跟踪时长以及RIS尺寸具有良好鲁棒性，单比特功率控制几乎不逊色于多比特或标量控制。

**⚠️ 局限性**

主要局限包括：依赖仿真数据，实际硬件验证尚缺；神经进化训练耗时且计算量大；当前仅考虑单RIS与单比特反馈，未处理多RIS、多比特控制或隐私保护等复杂场景。

---

## 27. GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number: The Group-Standard-Deviation Identity

**arXiv ID:** 2607.00152 | [PDF](https://arxiv.org/pdf/2607.00152v1)

**作者:** Yong Yi Bay `[一作]` (University of Illinois at Urbana Champaign), Kathleen A. Yearick `[通讯]` (University of Illinois at Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对RLVR训练中二元奖励的政策梯度进行解析，证明GRPO、Dr. GRPO和DAPO的差异仅在于对同一组标准差σ的不同处理，从而统一三种方法。

**💡 创新点**

创新点在于提出“组标准差身份”，将三种看似不同的训练策略归结为对同一标量的三种操作，并推导出精确的组大小法则与无声组率闭式，揭示标准差既是归一化因子，也是学习信号的大小。

**🔧 技术方法**

主要技术包括二元奖励的政策梯度解析、组内标准差表达、期望与大样本极限推导、闭式解析验证，以及对Big‑Math数据的实验对比。

**📊 数据集**

使用的主要数据集是Big‑Math（215,608道数学题），其中Llama‑3.1‑8B在64次rollout上的求解率分布被用来估计任务难度。

**📈 对比分析**

方法比较通过在控制实验（6,000个伯努利‑logit提示，G=8）以及对Big‑Math难度分布的统计分析完成，展示了无声组率、梯度分配与难度偏差的差异；实验结果验证了闭式预测与实际训练动态的一致性。

**⚠️ 局限性**

局限性包括仅针对二元奖励、首次on‑policy步、未考虑裁剪、KL惩罚、离线或多维奖励等；实验验证仅在简化的Bernoulli‑logit模型上完成，尚未在完整的LLM训练循环中直接检验。

---

## 28. Decentralized Geometric Control for Cable-Suspended Payload Transport with Adaptive Mass Estimation

**arXiv ID:** 2607.00024 | [PDF](https://arxiv.org/pdf/2607.00024v1)

**作者:** Hadi Hajieghrary `[一作]` (Georgia Tech), Miguel Hurtado `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出了GPAC体系结构，实现多无人机协同吊运悬挂载荷，完全去中心化、无队列信息，仅使用各机自身的绳索测量与局部估计即可实现载荷位置重构与负载共享估计。

**💡 创新点**

创新点在于：①隐式协调机制——每架无人机通过局部绳索张力与方向估算自身负载份额，实现集体力矩平衡；②层级化的四层控制架构，将姿态、位置、质量估计与扩展状态观测分离；③使用协同学习无持久激励的质量估计；④基于控制障碍函数的优先级安全过滤，实现实时约束满足而不需完整的约束求解。

**🔧 技术方法**

采用的技术包括：几何控制在非线性配置空间（SO(3)×R^3）上的直接控制；延伸状态观测器（ESO）用于风扰动补偿；并行学习（Concurrent Learning）用于质量估计；控制障碍函数（CBF）与高阶CBF处理多重约束；多频率分层（50 Hz、200 Hz、ESO 8 rad/s）实现层级耦合；EKF融合IMU、GPS等传感。

**📊 数据集**

实验使用Drake多体仿真平台，参数化柔性绳索（点质链模型）与Dryden风扰动模型，随机种子13次，包含19%绳长不匹配与实时传感噪声。

**📈 对比分析**

与基准配置对比，系统在13个随机种子下平均3D载荷跟踪RMSE为33.8 cm（CV 2.8%），单机计算量低；去掉ESO、CBF或并行学习的消融实验表明ESO对误差贡献最大，CBF提升约19%，并行学习对跟踪几乎无影响。

**⚠️ 局限性**

主要局限：①单绳局部载荷位置估计可观测性不足，无法完整观测切向分量；②安全过滤采用优先级序列投影，仅保证单约束激活时的ISS安全，多约束同时激活时只能保证优先级可行；③目前仅在仿真环境验证，缺乏硬件飞行实验。

---

## 29. A Filtered Mixture-of-Generators for Fully Synthetic Survival Training

**arXiv ID:** 2607.00127 | [PDF](https://arxiv.org/pdf/2607.00127v1)

**作者:** Niccolò Maria Rizzi `[一作]` (Politecnico di Milano), Matteo Matteucci `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc` `afceb026-1760-41ae-8d86-010831a37d97` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 FoGS pipeline，通过从四种 tabular 生成器池中采样合成数据，并用七个真实数据训练的生存模型对每个样本打分，最终通过两层 Optuna 搜索得到最佳样本过滤策略，从而构建高质量的生存分析训练集。

**💡 创新点**

创新点在于把合成数据的生成从单一模型改为“样本过滤”问题，利用多生成器的多样性与多模型评分的可解释性，形成针对下游任务的可调节过滤策略，显著提升合成数据对真实数据的替代性能。

**🔧 技术方法**

技术手段包括：四种 tabular 生成器（ARF、Bayesian Network、TabDDPM、SurvivalCTGAN）、七种生存评分器（Cox、DeepSurv、DeepHit、RSF、FPBoost、XGBoost‑Cox 及其加权集成）、两层 Optuna（外层优化过滤策略，内层优化 XGBoost‑Cox 下游模型），以及 TSTR（Train‑on‑Synthetic, Test‑on‑Real）评估框架。

**📊 数据集**

使用 16 个公开生存数据集，涵盖肿瘤、心脏、临床试验等领域，样本量、特征维度、事件率和缺失率均差异显著，作为评估 FoGS 效果的基准。

**📈 对比分析**

与真实数据训练的基准以及从生成器池随机抽样的基准进行对比，TSTR 下平均 C‑index 提升 2.17 点、IBS 降低 0.67 点；在 9/16 数据集两项指标均优于基准，13/16 至少有一项提升，整体差异显著（p≈0.04）。

**⚠️ 局限性**

局限性包括：仅使用 XGBoost‑Cox 作为下游模型、计算成本高（两层嵌套搜索导致数万次模型训练）、随机补充比例上限为 0.5、隐私评估仅使用 NNDR 近似指标、对极低事件率或极小样本量数据集生成器能力不足、单次实验无随机种子波动估计。

---

## 30. A Quantitative Framework for Estimating System Complexity and Cost via Component Interface Analysis

**arXiv ID:** 2607.00054 | [PDF](https://arxiv.org/pdf/2607.00054v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 31. Distributed Multi Robot Lunar Cargo Transportation via Phase Decomposed Reinforcement Learning

**arXiv ID:** 2607.00160 | [PDF](https://arxiv.org/pdf/2607.00160v1)

**作者:** Ashutosh Mishra `[一作]` (Tohoku University), Kazuya Yoshida `[通讯]` (Tohoku University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了面向月球表面协作货物搬运的分阶段强化学习框架，采用模块化轮‑臂机器人进行分层控制。

**💡 创新点**

创新点在于将搬运任务分解为抬升、运输、放置三阶段，使用集中训练/分散执行的PPO策略配合离散MDP门控和同步约束，以解决结构耦合、长期决策与安全停顿问题。

**🔧 技术方法**

技术手段包括PPO强化学习、离散MDP阶段门控、同步控制层、IsaacLab 物理仿真、OptiTrack 位姿捕捉和模块化轮‑臂机器人硬件实现。

**📊 数据集**

数据集主要由IsaacLab随机化仿真环境生成的多任务轨迹组成，硬件实验使用真实月球模拟土壤和三种不同质量/形状的货物进行实验。

**📈 对比分析**

与单一全任务策略对比，分阶段策略收敛速度提升，稳定性更好；硬件实验显示在三种负载下保持高度控制误差<30 mm、停顿响应迅速，性能优于无同步/无分阶段基线。

**⚠️ 局限性**

局限性包括仅验证两单元固定配置，缺乏对不同拓扑的泛化研究；仿真与现实差异仍存在，未实现在线动力学校准；在更崎岖或低摩擦地形下的鲁棒性尚未评估。

---

## 32. Decision Feedback Differential Detection for Reconfigurable Intelligent Surfaces

**arXiv ID:** 2607.00121 | [PDF](https://arxiv.org/pdf/2607.00121v1)

**作者:** Jiawei Qiu `[一作]` (McGill University), Harry Leib `[通讯]` (McGill University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种无需信道状态信息的决策反馈差分检测（DFDD）方案，用于在时变衰落信道上对可重构智能表面（RIS）的差分反射调制（DRM）进行解调。

**💡 创新点**

创新点在于将DFDD方法扩展到矩阵形式的DRM RIS系统，并通过理论推导与仿真证明其能显著降低在时间变衰落环境下的误码率（BER）误差平底，优于传统的常规差分检测（CDD）和已编码的DSTM方案。

**🔧 技术方法**

使用的主要技术包括：矩阵差分编码、DFDD预测滤波器（多阶系数求解）、Jakes模型下的时变瑞利衰落仿真、BER性能评估以及复杂度（乘法次数、时间复杂度）分析。

**📊 数据集**

数据集：采用蒙特卡罗仿真生成的随机比特流，结合基于Jakes模型的时变瑞利衰落通道和加性高斯噪声，覆盖多种K（反射单元数）与M-PSK调制（BPSK、QPSK）场景。

**📈 对比分析**

通过将DFDD DRM与无编码DRM、使用CDD的DRM-DSTM编码方案在相同K、相同Doppler频率下的BER曲线进行对比，结果显示：在低SNR下DFDD与CDD差异不大，但在中高SNR下DFDD的误码率继续下降，误差平底显著低于CDD，并在部分情况下甚至低于DSTM编码方案；复杂度随预测阶数V线性增加。

**⚠️ 局限性**

局限性包括：高阶预测（V>2）会因错误传播导致BER反弹；在高Doppler频率下性能仍受限；实现时需要存储多个历史符号并进行矩阵乘法，增加硬件资源消耗；DFDD对噪声和误判敏感，需精心设计预测系数和符号决策顺序。

---

## 33. The MMM Data Model -- A Normative Specification for Knowledge Interoperability in a Decentralisable Knowledge Commons

**arXiv ID:** 2607.00032 | [PDF](https://arxiv.org/pdf/2607.00032v1)

**作者:** Mathilde Noual `[一作]` (Aix Marseille Univ), Mathilde Noual `[通讯]` (Centre Européen de sociologie et de sciences politiques)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MMM（Mutable Mutual Matrix）数据模型，用于在去中心化的知识共享体系中记录、组织和交流正在进行的学术研究和知识工作；

**💡 创新点**

在保留自由文本标签表达的灵活性的同时，设计了一套小规模的规范约束和类型系统，使其兼具可扩展性、可解读性和去中心化协作能力；

**🔧 技术方法**

采用统一的规范化数据模型、UUID 标识、可变类型（Vertex、Edge、Pen、Pit）以及可扩展的标记系统，并将其与 CRDT（冲突自由复制数据类型）原则对齐，以实现可合并的写操作；

**📊 数据集**

在 Myrmex 应用中以真实学术笔记（包括定理、证明片段、实验数据等）作为实验数据，验证模型在科学研究协作中的可行性；

**📈 对比分析**

通过在 Myrmex 原型中实现数据导入/导出、增量合并和可视化过滤功能，展示了在多用户、跨部署环境下的快速同步与一致性维护，性能符合实时协作需求；

**⚠️ 局限性**

目前尚缺乏正式的模型规范与版本控制机制，缺少大规模对比实验，且对外部资源同步与完整语义推理支持有限，未来需进一步完善版本演进与互操作性细节。

---

## 34. An SOA-Based Big Data Management Framework for Primary Healthcare Centers in Bahrain

**arXiv ID:** 2607.00021 | [PDF](https://arxiv.org/pdf/2607.00021v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 35. Comparing Large Language Models on Scrum Certification-Style Questions: Accuracy, Stability, and Error Patterns

**arXiv ID:** 2607.00048 | [PDF](https://arxiv.org/pdf/2607.00048v1)

**作者:** Robson Alves Vilar `[一作]` (Federal University of Campina Grande), Angelo Perkusich `[通讯]` (Federal University of Campina Grande)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究评估了三款当代大型语言模型（GPT‑5 mini、Gemini 3 Flash、DeepSeek Chat 3.2）在与《Scrum Guide (2020)》对应的专业 Scrum Master I（PSM I）认证题库（993道题）上的问答表现。

**💡 创新点**

创新点在于：①从准确率、稳定性、任务敏感性和错误模式四个维度系统比较多模型表现；②采用零样本、链式思维、源信息归因三种提示策略；③对错误进行定性分析，区分模型本身局限与题目表述歧义。

**🔧 技术方法**

使用技术：基于 OpenAI、Google Gemini、DeepSeek API 的大语言模型推理；提示工程；重复执行评估稳定性；定量精确匹配评分与定性错误归类。

**📊 数据集**

数据集：993道英文学术版 PSM I 题库（包含 249 条 True/False、591 条单选、多选 153 条），每题带主题标签与答案键；使用平衡子样本（104 题按主题 8 题/主题，96 题按格式 32 题/格式）用于细粒度分析。

**📈 对比分析**

比较方法：对三模型在三种提示下计算整体与子集准确率，评估重复运行的标准差；对主题和题型的准确率做对比；定性分析错误案例。性能：Gemini 3 Flash 最高，平均约 97 %；GPT‑5 mini ~92 %，DeepSeek Chat 3.2 ~90 %；单选最高（≈94 %），多选约 88 %，True/False 最低 ≈88 %；模型间差异显著，提示对整体排名影响有限。

**⚠️ 局限性**

局限性：①题库仅为英语版 PSM I，难以推广至其他语言或未来 Scrum Guide 修订；②主题/格式子样本量小，单题异常可能影响结果；③定性分析主观性强，依赖作者判断；④未评估解释质量、部分正确等细粒度输出；⑤仅测试三款模型，未涵盖 Claude、Claude 3 等。

---

## 36. GRACE-RAG: Governed Retrieval Architecture for Canonical Evidence Synthesis, Enabling Lightweight Deployment in Closed-Domain Institutional Settings

**arXiv ID:** 2607.00013 | [PDF](https://arxiv.org/pdf/2607.00013v1)

**作者:** Asit Desai `[一作]` (National Payments Corporation of India), Prashant Devadiga `[通讯]` (National Payments Corporation of India)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 GRACE‑RAG，一种将结构推理外部化到离线检索层的检索增强生成架构，解决机构问答中实体密集、文档分散导致的证据碎片化问题。

**💡 创新点**

创新点在于：① 通过离线实体规范化、关系抽取与图构建将结构不确定性预先解析；② 双表面检索（chunk 与关系摘要）实现有界混合检索；③ 生成阶段仅聚焦证据综合，减少推理负担与对大模型的依赖。

**🔧 技术方法**

技术包括 LLM 辅助语义分块、实体与关系抽取、实体规范化、图构建与社区检测、双表面向量检索、全局再排序、有限子查询并行检索等。

**📊 数据集**

使用约 300 篇机构文档、约 8000 个实体提及、约 6000 条关系实例构成的高实体密度语料库，经过预处理后生成 1,000 个规范实体与结构化检索索引。

**📈 对比分析**

在三种模型容量（Mistral‑24B、GPT‑OSS‑120B、Gemini‑2.5 Flash）下对比基线向量检索与 GRACE‑RAG，结果显示结构性指标（完整性、深度、预期覆盖）提升 20% 以上，整体质量得分显著提高；与更大模型基线相比，GRACE‑RAG 在中等规模模型上已逼近甚至超越大型模型。

**⚠️ 局限性**

局限性包括：依赖高质量离线预处理与图构建，实体规范化与关系抽取误差会直接影响检索质量；对实时知识更新的适应性有限；在极端稀疏文档或非结构化领域的效果尚未验证。

---

## 37. When to Personalize Household Object Search: A Rigidity-Gated Hybrid Policy

**arXiv ID:** 2607.00022 | [PDF](https://arxiv.org/pdf/2607.00022v1)

**作者:** Xianyao Li `[一作]` (University of Florida), Eric Jing Du `[通讯]` (University of Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 PerSim 框架，用人类校准的连续 Big Five 性格向量学习家庭机器人搜索对象的空间先验，并通过刚度门控混合个性化与人群基线来决定何时使用个性化。

**💡 创新点**

创新点在于：① 基于对象放置刚度的门控个性化策略；② 连续性性格特征插值的先验建模；③ 结合 LLM 与人类锚定的合成动态生成；④ 双层人类验证确保合成数据行为可行并评估个性化效益。

**🔧 技术方法**

采用技术包括 Gemini 2.5 Flash 微调生成合成动态；交叉注意力将 Big Five 向量注入特征；OmniGibson 虚拟仿真模拟 14 天轨迹；两阶段搜索策略；以及人类评估实验。

**📊 数据集**

使用的数据集为 BEHAVIOR‑1K 家居场景、5 个住宅布局、16 个正交性格配置、200 名 Prolific 受试者收集的性格与放置锚点，以及生成的 27,976 条合成动作。

**📈 对比分析**

通过与无个性化基线、纯个性化先验和混合策略比较，使用 ERV、CP@5 与 ESC 指标，混合策略相较基线整体 ESC 降低约 7.8%，且在低刚度对象上提升显著；人类验证显示合成轨迹平均 3.85/5 的可行性评分，且 p=0.005 的刚度梯度偏好。

**⚠️ 局限性**

限制在于：仅在数字孪生与人类评估中验证，未在真实家居环境中测试；性格特征仅限于 Big Five，忽略文化与家庭构成等因素；未完整评估物理机器人执行成本；刚度评估为自评，缺乏长期跟踪。

---

## 38. Memory-Native Non-Terrestrial Networks for Embodied Intelligence

**arXiv ID:** 2607.00029 | [PDF](https://arxiv.org/pdf/2607.00029v1)

**作者:** Chengyang Li `[一作]` (University of Hong Kong), Huseyin Arslan `[通讯]` (Istanbul Medipol University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 MemNTN 框架，在非地面网络中通过物理记忆与数字记忆的双重存储，支持机器人等体化智能的跨层决策。

**💡 创新点**

创新点在于将记忆视为网络设计的核心，构建完整记忆生命周期（获取、压缩、评估、更新、利用），并提出多维记忆价值模型实现跨层价值驱动的资源分配。

**🔧 技术方法**

采用多模态 VLM/LLM 生成记忆表示与 QA 对，利用 Airsim+CARLA+LEOPath+OpenNTN 模拟环境，使用 Qwen3-vl-8b、mxbai-embed-large-v1、Milvus 等技术实现记忆压缩与检索。

**📊 数据集**

使用 CARLA Town04 生成的视觉感知数据、LEOPath 轨道数据和 OpenNTN 通道模型作为实验数据集；无人机、卫星和地面站的真实传输参数也被模拟。

**📈 对比分析**

通过与 MaxRate、MaxCov、Fairness、Mem-Terrestrial、Mem-No-Backhaul 等基线对比，MemNTN 在 SEQA 任务中远程答复准确率提升至 98.7%（相比 MaxRate 提升 62.6%），且在卫星星座规模不同的情况下表现出更好的可扩展性。

**⚠️ 局限性**

局限在于分布式记忆一致性与标准化尚未完善，且安全隐私风险（如记忆污染、未授权检索）需要进一步研究。

---

## 39. Temporal Path Covers: Dilworth Properties and Parameterized Complexity

**arXiv ID:** 2607.00118 | [PDF](https://arxiv.org/pdf/2607.00118v1)

**作者:** Lapo Cioni `[一作]` (Università degli Studi di Firenze), Manolis Vasilakis `[通讯]` (Universitè Paris Dauphine -- PSL)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文系统研究了时间图中的最小路径覆盖（TPC）与最小时间不相交路径覆盖（TDPC）的可解性与复杂性，并给出了在不同假设和参数化下的算法与硬度结果。

**💡 创新点**

创新点包括：① 在 Dilworth 性能（即最小覆盖数等于最大反链数）得到保证时，证明两问题可用 Lovász θ 函数以多项式时间求解；② 通过精细的参数化分析，首次证明 TPC 在删除至线性森林的距离参数化下对 2 步时间图仍为 W[1]-hard，并指出加入树宽+时间步参数化无法得到 2^k 或多项式算法；③ 在使用顶点覆盖数+时间步参数化时给出 FPT（ILP+Lenstra）算法；④ 对路径长度受限、最大度、路径宽等结构参数进一步扩展 para‑hardness 结果。

**🔧 技术方法**

主要技术手段：结构化时间图构造与归约、Lovász θ 数的半正定规划与松弛、整数线性规划（ILP）与 Lenstra 算法、时间标签序列化、以及对图类（如 DAG、树、线性森林）的细致分析。

**📊 数据集**

论文为理论研究，不使用任何实验数据集；所有结论均基于严格的归约与算法证明。

**📈 对比分析**

相较于已有的基于树宽+时间步的 3^k 或 k^k 算法，本文证明在删除至线性森林参数化下不可能得到 2^k 或多项式解；而在顶点覆盖+时间步参数化下实现 FPT；此外，利用 Lovász θ 函数实现了在 Dilworth 性能保证下的多项式求解，但仅能得到覆盖大小，未给出具体路径构造。

**⚠️ 局限性**

局限性：① 只给出了覆盖数的多项式求解，搜索版本（具体路径）仍未得到多项式算法；② 仅在完全满足 Dilworth 性能的情形下可解，对近似满足情况缺乏分析；③ 对某些结构参数的边界仍未完全划定，例如是否在删除至线性森林距离为 2 时仍保持 W[1]-hard；④ 所有结果均为决策/计数问题，路径长度受限的动态求解未完整覆盖。

---

## 40. PRA-RAG: Provably Robust Aggregation in Retrieval-Augmented Generation against Retrieval Corruption

**arXiv ID:** 2607.00012 | [PDF](https://arxiv.org/pdf/2607.00012v1)

**作者:** Xue Tan `[一作]` (Fudan University), Jun Dai `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可证明鲁棒的检索聚合算法PRA-RAG，用于抵御检索级别的投毒攻击；

**💡 创新点**

创新点在于利用组合嵌入的最小包围球（MEB）理论，提供了对投毒影响的上界证明，并引入了Provable Average Deviation（PAD）度量；

**🔧 技术方法**

核心技术包括组合嵌入构造、几何距离（角度距离）计算、最小包围球聚合以及基于查询相似度的加权平均；

**📊 数据集**

在三大问答数据集（Natural Questions、MS-MARCO、HotpotQA）上与多种LLM（Mistral-7B、Llama3-8B、Vicuna-7B、Qwen3-8B、GPT-3.5-turbo、GPT-5-mini）进行实验；

**📈 对比分析**

与RobustRAG、InstructRAG、AstuteRAG、TrustRAG、RAGForensics等基线相比，PRA-RAG在攻击成功率（ASR）几乎降到1%同时保持约70%以上的准确率（ACC），并在多数实验中实现了最高ACC/最低ASR组合；

**⚠️ 局限性**

局限性包括：需要在检索文本中构造子集，导致额外计算与响应时间；依赖投毒文本数量低于一定阈值；若攻击规模过大，防御效果会下降。

---

## 41. A Role-Based Multi-Agent Model for Climate Adaptation Deliberation Across Living Labs

**arXiv ID:** 2607.00046 | [PDF](https://arxiv.org/pdf/2607.00046v1)

**作者:** Önder Gürcan `[一作]` (NORCE Research AS), Ivan Puga-Gonzalez `[通讯]` (NORCE Research AS)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于角色的多智能体模型，用以模拟生活实验室中的气候适应决策讨论。

**💡 创新点**

核心创新在于将决策主体按角色划分并外部化情境输入，保持行为机制不变，提升可复用性和可解释性。

**🔧 技术方法**

采用多智能体仿真框架，配合网络传递与加权聚合机制，进行四阶段模拟。

**📊 数据集**

使用的输入数据为从利益相关者映射、访谈和网络调研得到的角色分配、网络结构、评价准则及参数等。

**📈 对比分析**

通过配置不同的生活实验室情境进行对比实验，评估决策结果、信息传播和立场变化的差异，尚未完成正式验证，但预期能揭示情境差异带来的影响。

**⚠️ 局限性**

限制在于目前仅为设计示例，缺乏实证验证和模型校准，且对细粒度个体行为建模较弱。

---

## 42. A Taxonomy of Single-Turn Textual Prompt Patterns

**arXiv ID:** 2607.00043 | [PDF](https://arxiv.org/pdf/2607.00043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 43. Controllable Narrative Rendering for Enhanced Assisted Writing

**arXiv ID:** 2607.00009 | [PDF](https://arxiv.org/pdf/2607.00009v1)

**作者:** Mingzhe Lu `[一作]` (Chinese Academy of Sciences), Yangyan Xu `[通讯]` (HiThink Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Loom框架，实现对文本的可控叙事渲染，既能提升描述深度，又能保持故事事件完整；

**💡 创新点**

创新点在于将叙事学的故事-话语层次拆解为三层管道（感知配额、意义构建、叙事渲染），通过预算约束与分层生成实现对渲染密度与事实忠实度的精准控制；

**🔧 技术方法**

技术核心是基于大语言模型的角色扮演式半语义链条思考，先分配感知预算，再将其映射为意图对齐的语义原子，最后使用微创注入操作在原文本中局部插入细节；

**📊 数据集**

使用ROCStories数据集进行评估，基于其短句结构可清晰测量渲染效果；

**📈 对比分析**

与多种规模与思考能力的LLM基线（Qwen系列、GPT‑5‑thinking等）比较，Loom在渲染比例平衡、渲染方法合规性与风格整合度三个维度均获得最高分，平均得分约为3.86，显著优于其它模型；

**⚠️ 局限性**

局限性包括对线性叙事结构的依赖，难以处理诗歌等非线性或极简文本；同时依赖手工设计的三层提示，尚未实现完全自动化。

---

## 44. Bounded Morality: Defining the Space of Moral Computation

**arXiv ID:** 2607.00002 | [PDF](https://arxiv.org/pdf/2607.00002v1)

**作者:** Max Kanwal `[一作]` (Stanford University), Patrick Mineault `[通讯]` (Amaranth Foundation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出“Bounded Morality”框架，将道德推理视为在有限资源下的受限推断，定义了道德宽度（表示范围）和道德深度（推理深度）两条互相独立的维度，并探讨它们在有限资源下的不可避免的权衡。

**💡 创新点**

创新点在于将伦理理论视为在不同资源约束下的局部最优策略，而非绝对真理；引入了“道德后悔”与“在约束下的道德进步”概念，揭示了道德争议和伦理学家不同观点的计算来源。

**🔧 技术方法**

使用了受限控制论和图论的形式化方法：构造道德交互图、抽象映射、有限深度的动态规划，计算资源成本模型（信息成本与推理成本），并用这些模型推导出宽度–深度 Pareto 前沿。

**📊 数据集**

没有使用公开数据集，而是通过构造的案例（如内容审核情境）演示框架，借助发展心理学、比较心理学和文化史的实验结果来支撑宽度与深度维度的存在性。

**📈 对比分析**

方法比较基于理论分析与案例演示：不同伦理策略（功利主义、义务论、契约论、德行论、关怀伦理）在同一预算下对应不同的 (b,H) 点，展示了在相同资源下的后悔差异，从而表明框架能够解释并量化道德策略的优劣。

**⚠️ 局限性**

局限包括：1）假设道德原理、动力学和效用函数已知，实际中需估计；2）忽略搜索复杂度的上界，可能低估实际推理成本；3）未考虑多代理交互中的博弈论效应；4）理论验证主要依赖模拟，缺乏大规模实证数据支持。

---

## 45. 3D Point World Models: Point Completion Enables More Accurate Dynamics Learning

**arXiv ID:** 2607.00148 | [PDF](https://arxiv.org/pdf/2607.00148v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 46. ALEE: Any-Language Evaluation of Embeddings via English-Centric Minimal Pairs

**arXiv ID:** 2607.00171 | [PDF](https://arxiv.org/pdf/2607.00171v1)

**作者:** Andrianos Michail `[一作]` (University of Zurich), Juri Opitz `[通讯]` (University of Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一套基于AMR的动态跨语言嵌入评估框架，生成英语最小对并与目标语言平行句配对，以检验嵌入模型对细粒度语义差异的跨语言对齐。

**💡 创新点**

① 动态生成最小对而非静态数据集；② 通过AMR实现可控的语义扰动；③ 将英语最小对与任意目标语言平行数据结合，覆盖275+语言，尤其低资源方言。

**🔧 技术方法**

AMR解析与生成、规则式语义编辑、NLI模型双向过滤、Triplet Accuracy度量以及多种预训练模型（LaBSE、E5、Qwen嵌入等）。

**📊 数据集**

FLORES‑200、WMT24++（含六种罗曼语）和BOUQuET三大平行语料集，共275+语言。

**📈 对比分析**

采用Triplet Accuracy（TACC）和Macro‑TACC进行评估；实验表明所有模型仍无法完全解决最小对，Polarity最易，角色交换、反义词和超类替换难度相近；长文本和段落更难；性能与语言在预训练/微调中的出现频率、子词碎片化呈正/负相关，低资源方言得分普遍偏低。

**⚠️ 局限性**

评估仅在英语侧施加扰动，缺乏目标语言直接编辑；依赖AMR解析/生成及NLI过滤，可能带噪；使用对比排名指标，未评估绝对相似度校准；对极端低资源语种或方言的自然度控制仍有待改进。

---

## 47. MeshDNS: A Cooperative DNS Resolution Framework for Resource-Constrained IoT Networks

**arXiv ID:** 2607.00122 | [PDF](https://arxiv.org/pdf/2607.00122v1)

**作者:** Asif Mahbub `[一作]` (North South University), Nabil Bin Hannan `[通讯]` (North South University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

在资源受限的物联网环境中实现了MeshDNS，一个分布式DNS解析框架。

**💡 创新点**

创新点在于通过Ed25519签名的同意投票与哈希摘要的缓存同步，实现在单广播域内的拜占庭容错、低延迟本地缓存。

**🔧 技术方法**

采用ESP8266 MCU、UDP广播、BLAKE2b哈希、Ed25519签名、预共享密钥、可配置LRU缓存与离线预取等技术。

**📊 数据集**

实验数据集包括5节点ESP8266硬件测试板与Python SimPy离散事件模拟（50、250、1000节点），并在5节点实验中注入不同等级的拜占庭攻击。

**📈 对比分析**

与mDNS基线比较，MeshDNS在热缓存下平均0.47 ms（比1.39 ms快），冷缓存投票延迟约1.4 s；在仿真中在1000节点下仍保持低网络负载并在95%覆盖率下恢复约18 s。

**⚠️ 局限性**

局限性包括只能在单广播域内工作、对物理侧入侵不防御、冷缓存投票时延高、未实现DNSSEC链校验以及在大规模无线网络下的广播风暴风险。

---

## 48. Destination-Labeled Self-Looping Systems with Dwell: Intrinsic Characterization, Realization Cost, and Recognition

**arXiv ID:** 2607.00044 | [PDF](https://arxiv.org/pdf/2607.00044v1)

**作者:** Reda Belaiche `[一作]` `[通讯]` (Paris-Est Creteil University), Reda Belaiche (Paris-Est Creteil University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a244defd-9560-426b-b1b1-f78ebb2b7bf9`

**🎯 论文内容**

研究在给定可达且满足可实现离开的可视图上，如何通过状态机实现每个状态的最小停留时间（DLSL）并将其转换为确定性转导器。

**💡 创新点**

提出了纤维线性（fiber‑linear）这一本质结构特征，实现了对DLSL系统的完全内在表征，并证明该类转导器的状态数等于停留时间之和，且在可达性条件下具有唯一的规范化表示。

**🔧 技术方法**

运用了有限自动机与转导器理论、路径纤维结构、单生成链幺半群、以及多阶段的构造与逆构造算法。

**📊 数据集**

无实验数据集，本文为纯理论性工作。

**📈 对比分析**

未进行实验比较；本研究以理论证明与多项式时间算法作为验证手段。

**⚠️ 局限性**

局限性在于仅适用于确定性且图保持的转导器，且需满足所有可视状态可达且至少存在一次非自循环离开的假设，无法直接推广到非确定性或概率模型。

---

## 49. The Limits of LLM Forecasting: Parametric Knowledge Gaps Across Conflict Zones

**arXiv ID:** 2607.00018 | [PDF](https://arxiv.org/pdf/2607.00018v1)

**作者:** Poli Nemkova `[一作]` `[通讯]` (University of North Texas), Poli Nemkova (University of North Texas)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究媒体报道不均衡对大语言模型（LLM）在冲突升级预测中的影响，并通过零-shot 评估 Llama‑3.3‑70B 与 GPT‑4o 在 22 个冲突区的表现，比较其与传统结构化基线（Always‑YES、Persistence、Logistic Regression）的差异。

**💡 创新点**

发现 LLM 在不同媒体覆盖度下表现出根本性分类而非预测的错误模式：Llama 对低覆盖区几乎总是预测升级，GPT‑4o 对高覆盖区几乎总是预测无升级；并证明即便给与结构化事件信息，LLM 仍无法利用时间信号；提出覆盖度分层评测、低覆盖冲突数据集建设以及训练数据地理覆盖度记录的必要性。

**🔧 技术方法**

采用零-shot 预测、结构化基线（Always‑YES、Persistence、Logistic Regression）、LLM 参数化推断、以及基于 ACLED 事件计数的证据增强（evidence‑augmented）等技术进行实验，使用 F1、Precision、Recall 等指标评估。

**📊 数据集**

数据集包括：ACLED 事件数据（2020‑2026）作为冲突强度与升级阈值的基准；GDELT 英文新闻数据用于衡量每个冲突区的媒体关注度（注意力比例）。共 22 个冲突区，包含 1,628 条样本，测试集 660 条。

**📈 对比分析**

在相同 660 条测试集上对比：Llama‑3.3‑70B（零-shot）F1≈0.374，GPT‑4o F1≈0.333；Logistic Regression（11 观察窗口特征，无国别信息）F1≈0.402，且在低覆盖区、低覆盖区均可获得 0.400，优于两大 LLM；Llama 在低覆盖区与 Always‑YES 完全相同；GPT‑4o 在高覆盖区完全失效。证据增强后两大 LLM 的 F1 下降，只有在高覆盖区略有提升，整体性能低于结构化基线。

**⚠️ 局限性**

局限性包括：GDELT 仅计量英文新闻，可能低估非英文覆盖；ACLED 覆盖度本身存在偏差；只评估两种 LLM，无法推广到其他模型或训练截止时间；仅进行零-shot 评估，未测试微调、少量示例或检索增强方法；证据形式过于简化（仅三个月事件计数），未探究更丰富的结构化输入；无法确定因果关系，只能提出训练数据曝光效应假设。

---

## 50. Readable but Not Controllable: Neuron-Level Evidence for Medical LLM Hallucination

**arXiv ID:** 2607.00158 | [PDF](https://arxiv.org/pdf/2607.00158v1)

**作者:** Vijay Vankadaru `[一作]` (University of California, Berkeley), Peyman Passban `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了医学大语言模型中的幻觉检测与可操作性，评估内部激活中幻觉信号的可读性与因果可控性。

**💡 创新点**

发现幻觉信号分布在大量神经元上，检测与控制分离，随机子集即可匹配精选子集，表明幻觉是分布式低维表示。

**🔧 技术方法**

使用CETT/Swiglu后激活、线性探针、k‑sweep、CKA、PCA、随机与精选子集对比、激活导向干预等技术，评估幻觉可读性与可控性。

**📊 数据集**

实验使用四个医学问答数据集：MedMCQA、MedQA‑USMLE、PubMedQA‑Labelled、PubMedQA‑Artificial。

**📈 对比分析**

通过AUROC与随机子集/全层对比，探针AUROC介于0.77–0.86，随机子集可与全层匹配；但干预实验显示选定神经元对模型输出影响不显著，控制效果不佳。

**⚠️ 局限性**

仅测试单层激活导向干预，未涵盖更深层、多层、子空间或电路级干预；样本量限制导致部分对比无显著性，缺乏更广泛的可行性验证。

---

## 51. Libra: Training the Environment for Agentic Information Retrieval

**arXiv ID:** 2607.00016 | [PDF](https://arxiv.org/pdf/2607.00016v1)

**作者:** Xuan Zhao `[一作]` (Salesforce), Gengyu Wang `[通讯]` (Columbia University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个可自我演进的 Markdown 目录系统（Libra），通过 LLM 驱动的对抗循环（Prompter、Solver、Healer）在大型代码仓库中持续优化信息检索能力

**💡 创新点**

将环境索引从静态外部嵌入式模型迁移到可动态更新、纯文本的目录结构，并利用 LLM 作为优化器基于交互失败持续改进索引

**🔧 技术方法**

LLM 生成的合成查询、固定工具链的查询解析器、LLM 作为优化器（Healer）修改目录、对抗式训练循环、Markdown 目录层级结构

**📊 数据集**

SWE‑Bench Lite 12 个 Python 仓库（含 430k+ 行代码的主仓库）以及对真实 Bug 任务的子集

**📈 对比分析**

与 LocAgent、RepoMem 两种基准进行对比；在目录训练后，Libra 在文件定位准确率上与基准相当或更优，且对不同 LLM 模型均能无缝迁移，展示了日志式提升和跨模型、跨问题集的转移效果

**⚠️ 局限性**

训练循环计算量大、仅在静态仓库验证、目录更新缺乏增量机制、合成查询主要为单跳、未覆盖多跳复杂查询

---

## 52. Solution space path planning for supporting en-route air traffic control

**arXiv ID:** 2607.00064 | [PDF](https://arxiv.org/pdf/2607.00064v1)

**作者:** Yiyuan Zou `[一作]` (Delft University of Technology), Clark Borst `[通讯]` (Delft University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了基于方案空间的冲突自由路径规划算法SSPP，用于机载航路规划，并提供可解释的可行行动空间。

**💡 创新点**

创新点在于将方案空间与任意角度搜索相结合，提出三种冲突检测方法（距离、时间区间、冲突区），并设计了SSPV和SSPE两种节点定义，显著提升可解释性与实时性。

**🔧 技术方法**

采用了A*搜索、任意角度路径规划、shadowcasting可视化、距离/时间区间/冲突区检测、以及多目标权重优化等技术。

**📊 数据集**

实验基于马斯特里赫特上空区（Delta sector）的航路数据，随机生成2-20架航班的进出口点，并在5 nm或2.5 nm网格上运行共100个场景。

**📈 对比分析**

通过与三种检测方法对比，SSPV-Z在5 nm网格下平均运行时长仅3.69 ms，成功率达98.5%，SSPV比SSPE快3.77倍；距离法最精确但最慢；时间区间最保守且最慢；冲突区方法最快且保持高成功率。

**⚠️ 局限性**

主要局限包括仅考虑二维平面、固定速度、忽略转弯动力学与轨迹不确定性；对网格大小敏感；仅处理水平分离；未进行人机交互验证。

---

## 53. ATM: CID-Brokered Pre-Write Admission for Multi-Agent Code Co-Synthesis

**arXiv ID:** 2607.00041 | [PDF](https://arxiv.org/pdf/2607.00041v1)

**作者:** Eagl Huang `[一作]` `[通讯]`, Eagl Huang

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AI-Atomic-Framework（ATM），在多代理共享写入前对写意图进行预写入门控，决定是否并行、序列化或拒绝写入

**💡 创新点**

引入七层硬门控和虚拟原子回退机制，将写意图映射为可审计的“原子”并使用 CID broker 做确定性决策

**🔧 技术方法**

使用适配器驱动的原子化、CID 共享管理、CAS 哈希校验、虚拟原子、任务合同等技术栈

**📊 数据集**

基于控制实验（12 场景设计矩阵、3 条 runner 案例）、字段实验（POS2、B‑12、BLOCK）、外部采用者研究及 20 个独特场景的 ATM‑AdmissionBench

**📈 对比分析**

与现有基线（如 CodeTeam、CoAgent、S‑Bus 等）在同域写入冲突决策场景对比，表明 ATM 在可审计性和并发性上可行，但未表现出比其他系统显著更优的性能

**⚠️ 局限性**

仅在单域仓库级预写门控，无法替代 Git 合并、跨 clone 或 PR 的治理，且对跨语言原子一致性支持有限

---

## 54. HySpecPro: Scalable Hypergraph Partitioning via Spectral Projection Optimization

**arXiv ID:** 2607.00055 | [PDF](https://arxiv.org/pdf/2607.00055v1)

**作者:** Rongjian Liang `[一作]` (NVIDIA), Haoxing Ren `[通讯]` (NVIDIA)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 HySpecPro，一种单层超图划分框架，通过谱嵌入空间的端到端连续优化实现划分，完全避免多级粗化导致的信息损失。

**💡 创新点**

创新点包括：
• 将超图划分问题映射到二分图拉普拉斯谱嵌入空间，并直接在该空间内用投影向量表示划分；
• 采用 CMA‑ES 在投影空间进行无梯度、全局化的优化；
• 通过 GPU（CuPy、DGL、PyTorch）实现谱嵌入构造与投影评估的完全并行化；
• 提供混合谱嵌入和可微细化两种可选增强方案，进一步提升划分质量。

**🔧 技术方法**

使用技术包括：
• 二分图加权拉普拉斯谱嵌入（归一化权重 ω(e)/d(e)）
• CMA‑ES 进化优化算法
• GPU 加速：CuPy 进行稀疏矩阵运算；DGL 进行批量投影评估；PyTorch 进行张量运算
• 可微细化（类似 MedPart 的 Softmax 近似）

**📊 数据集**

实验数据集：
• Titan23 设计集（数十亿元件、超大 hyperedge 分布）
• L_HG 大规模超图集合（如 StocF-1465、HV15R、Geo_1438 等），全部以 TotalDeg 和高阶边度量评估

**📈 对比分析**

与 hMETIS、SpecPart、MedPart、KaHyPar、SHyPar、TritonPart、mtK‑min/mean 等 SOTA 多级/多线程划分器在 cut 大小和运行时间上进行对比。HySpecPro 在 cut 质量上平均比 hMETIS 提升 9–23%（2% 与 20% 失衡率），并在 TotalDeg>2M 的大规模实例上实现 5–30× 的速度提升，整体保持线性与 TotalDeg 的比例。

**⚠️ 局限性**

局限性：
• 目前仅支持单元权重（均匀 vertex/hyperedge 权重），对非均匀权重的支持尚未完成；
• 依赖 CMA‑ES 的随机采样，可能导致结果不确定性；
• Python 级实现导致 I/O 和内存开销显著，进一步优化可提升性能；
• 对极大规模实例内存仍是瓶颈，GPU 资源受限时需拆分或分布式方案。

---

## 55. AGE: Adaptive-masking for Graph Embedding in Graph Retrieval-Augmented Generation

**arXiv ID:** 2607.00052 | [PDF](https://arxiv.org/pdf/2607.00052v1)

**作者:** Bao Long Nguyen Huu `[一作]` (OMRON Corporation), Atsushi Hashimoto `[通讯]` (OMRON SINIC X Corporation)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在Graph Retrieval-Augmented Generation（GraphRAG）框架中提出一种新的图嵌入方法AGE，能够更好地将检索到的子图信息映射到冻结LLM的输入空间，从而提升知识检索与推理性能。

**💡 创新点**

创新点主要包括：①引入可学习的节点采样器（node sampler）通过强化学习动态决定哪些节点为关键节点；②采用JEPA（Joint-Embedding Predictive Architecture）取代传统的生成式自监督学习，聚焦抽象语义层面，避免冗余细节重建；③三阶段分离损失（prompt tuning、目标重构、采样损失）实现对不同模块的无权重调和优化。

**🔧 技术方法**

技术栈：Transformer‑based self‑supervised学习（mask‑based SSL）、JEPA、强化学习（REINFORCE）、图卷积/多头注意力编码器、位置编码、图结构聚合、Prompt Tuning + LoRA（可选）以及多种开源LLM（Llama3.2、Llama2、Qwen3.5等）作为冻结后端。

**📊 数据集**

实验数据集涵盖三类任务：Commonsense reasoning（ExplaGraphs）、视觉问答（SceneGraphs）以及大规模知识图问答（WebQSP、ComplexWebQuestions）。

**📈 对比分析**

与传统G‑Retriever、G‑Retriever+PEFT以及多种LLM‑Retriever（如GPT‑4、KG‑Agent、Paths‑over‑Graph等）对比，AGE 在所有评测指标上均取得显著提升，尤其在 ExplaGraphs 上提升 22–26%，WebQSP Hit@1 提升 1–2%，在 LLM‑Retriever 基线中亦能以非参数检索方式实现与大模型检索器相近或更优的性能。

**⚠️ 局限性**

局限性包括：①固定采样率不适应不同图中关键节点密度变化；②仅在GraphRAG任务上验证，未测试对大模型或其他图任务（如节点分类、链路预测）的适用性；③对大规模LLM的有效性尚未验证，受计算资源限制；④方法聚焦知识表示，未直接解决图学习任务。

---

## 56. EmbodimentSemantic: A Spatial Scene-Graph Dataset and Benchmark for Vision-Language Models on Embodied Manipulation Trajectories

**arXiv ID:** 2607.00020 | [PDF](https://arxiv.org/pdf/2607.00020v1)

**作者:** Hassan Jaber `[一作]` (Politecnico di Torino), Haitham Bou-Ammar `[通讯]` (Huawei)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EmbodimentSemantic 数据集与基准，评估视觉‑语言‑动作模型在机器人操纵中的空间场景图预测，并通过将预测场景图注入现有 VLA 策略来检验其对下游控制的影响。

**💡 创新点**

创新点在于：
1) 使用 MuJoCo 计算得到的实时/离线场景图实现精确的三元组（object–relation–object）标注；
2) 采用固定对象词表与八个双向空间关系词，使评估可通过 exact triplet F1 进行定量比较；
3) 系统性诊断视角、对象绑定、深度、支撑与包含等难度关系；
4) 构建在线接口，在 VLA 评估中实时生成并注入场景图，检验场景图对策略性能的影响。

**🔧 技术方法**

技术手段包括：Transformer‑based 视觉‑语言模型（VLM）在提示式生成场景图；MuJoCo 物理引擎与几何投影用于生成真实场景图；RGB‑only 视觉输入；基于三元组的评估指标（F1、覆盖率、hallucination 等）；以及 VLA 策略的在线随机化与扰动实验。

**📊 数据集**

使用的数据集有：
- LIBERO‑Spatial（10 个任务、500 次演示，60k+ 帧），
- SO101 真实机器人收集的 257 次任务演示；
- 所有帧均附有 128×128 的 RGB 图像和对应的场景图。
该数据集包含模拟与真实两部分，提供完整的对象与关系注解。

**📈 对比分析**

评估方法：对多种公开与商用 VLM（如 Gemini、Gemma、MomaGraph 等）使用相同提示，比较它们在两种相机视角下的 exact triplet F1、覆盖率、hallucination 率等。最高 F1 约为 0.57（Gemini），而其他模型低至 0.10。对 VLA 进行在线场景图注入实验，发现对部分任务能提升成功率（如任务 4 从 46% 提升到 76%），但对其他任务效果不佳或甚至下降，表明场景图的帮助具有任务与模型依赖性。

**⚠️ 局限性**

局限性：
1) 仅覆盖固定的 7 个对象和 8 个空间关系，缺乏开放词表与更丰富的关系；
2) 依赖 MuJoCo 的几何规则和视角定义，可能不完全适用于真实世界多样场景；
3) 真实数据仅提供预测场景图，没有人工标注，难以验证准确性；
4) VLA 评估仅为在线注入，未训练新策略，限制了对场景图真正利用效果的深入探究。

---

## 57. SchemaRAG: Dynamic Large Schema Reduction for LLM-driven Structured Information Extraction

**arXiv ID:** 2607.00008 | [PDF](https://arxiv.org/pdf/2607.00008v1)

**作者:** Sin Yu Bonnie Ho `[一作]` (Microsoft), Paul Vozila `[通讯]` (Microsoft)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SchemaRAG框架，利用检索增强生成动态裁剪大规模复杂模式进行结构化信息提取。

**💡 创新点**

创新点在于对输出模式空间的检索式裁剪，结合模式元数据与少量示例实现实时、低成本的子模式选择。

**🔧 技术方法**

使用检索增强生成（RAG）与大语言模型（LLM）、句向量嵌入和近邻检索实现模式选择与分段提取。

**📊 数据集**

在医疗领域的四个医院Nursing数据集以及Amazon产品描述数据集上进行评估。

**📈 对比分析**

与全模式提示及分段全模式基线比较，SchemaRAG在Micro‑F1上提升约8–9%，延迟下降47%，token成本下降48%（Amazon略高）。

**⚠️ 局限性**

局限在于对Amazon数据集的准确率低、模式与文本噪声大、分段参数未针对长文本优化以及需依赖示例标签。

---

## 58. Comparing the Emotional Impact of Thematic Versus Episodic Framing in Visualization Text

**arXiv ID:** 2607.00103 | [PDF](https://arxiv.org/pdf/2607.00103v1)

**作者:** Poorna Talkad Sukumar `[一作]` (Munster Technological University), Oded Nov `[通讯]` (New York University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

进行了一项预注册的线上实验，探讨文本框架（专题式 vs 事件式）对公众情绪和枪支管制态度的影响。

**💡 创新点**

首次在数据可视化中对比事件式与专题框架，发现情绪调节在影响态度中的间接作用，并揭示文本框架对感知中立性的影响。

**🔧 技术方法**

使用Qualtrics在线调查，线性混合效模型和Bootstrap中介分析，测量SAM情绪量表与枪支管制支持量表。

**📊 数据集**

2001‑2024年美国大规模枪击事件数据，来源于Violence Project Mass Shooter Database。

**📈 对比分析**

采用三种文本框架（T、T+Ann、E+Ann）随机分配实验组，结果显示事件式框架产生更强负面情绪，并通过情绪中介提升枪支管制支持，但对态度的直接影响不显著。

**⚠️ 局限性**

仅聚焦美国大规模枪击领域，框架操作样式有限，未完全分离标题与注释效应，结果可能不具备跨领域推广性。

---

## 59. RareDxR1: Autonomous Medical Reasoning for Rare Disease Diagnosis Beyond Human Annotation

**arXiv ID:** 2607.00147 | [PDF](https://arxiv.org/pdf/2607.00147v1)

**作者:** Deyang Jiang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Bo Xu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一种端到端、推理为中心的稀有病诊断大型语言模型RareDxR1，能够直接从非结构化临床记录进行开放域诊断，无需传统表型提取或检索增强。

**💡 创新点**

创新点包括：① 知识内化（RareKnowledgeQA）将结构化与非结构化知识嵌入模型参数；② 反思增强推理采样（RERS）让模型在失败后自我纠错，生成专家级诊断链；③ 双层课程强化学习（DCRL）在任务级与案例级双维度递进，提升诊断逻辑与知识记忆；④ 合作推理精炼（CRR）在推理阶段融合外部模型与检索知识，进一步纠正偏差。

**🔧 技术方法**

采用大型语言模型微调、强化学习、反思采样、知识约束过滤、双层课程策略以及多模型协同推理等技术。

**📊 数据集**

使用的主要数据集包括：RareArena（22,896笔稀有病病例）、MIMIC-IV-Note（OOD 590笔稀有病病例）、HPO/Orphanet/OMIM 结构化知识库、PubMed 文献、构建的 RareKnowledgeQA 与 RareDxCOT 数据集。

**📈 对比分析**

与 GPT‑4o、DeepSeek‑R1、Gemini‑2.5‑Flash、Qwen3 等大规模通用与医学专用模型对比，RareDxR1‑14B 在 RareArena‑Test 的 Top‑10 Recall 达到 69.27%，远超 DeepSeek‑R1‑671B 的 59.02%；在 MIMIC‑IV OOD 上 Top‑10 Recall 为 65.59%；零样本场景 Top‑10 Recall 50%，表明具备较强泛化；加入 CRR 后提升约 4‑5%，并在常见疾病与结构化表型任务上也保持领先。

**⚠️ 局限性**

仍需临床验证与人工监督；对极罕见病的识别受训练样本限制；缺乏多模态（影像、基因等）数据支持；推理链解释性不足，可能存在偏差。

---

## 60. Trajectory Learning with Graph Representations for Social Robot Navigation

**arXiv ID:** 2607.00028 | [PDF](https://arxiv.org/pdf/2607.00028v1)

**作者:** Berke Kartal `[一作]` (Bogazici University), Emre Ugur `[通讯]` (Bogazici University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种利用图特征自编码器（Graph Feature Autoencoder）与轨迹级学习的模仿学习框架，用于学习社会合规的移动机器人导航。

**💡 创新点**

创新点在于①通过图注意机制学习可迁移的群体嵌入；②采用轨迹级条件神经运动原语（CNMP）联合预测机器人轨迹与群体未来状态；③采用两阶段训练分离群体表示与策略学习。

**🔧 技术方法**

使用图神经网络（Graph Transformer Layer）+注意力机制、图特征自编码器、条件神经运动原语（CNMP）及轨迹级学习目标。

**📊 数据集**

使用 SEAN 2.0 仿真数据（160 条示范）和 SCAND 实际 LiDAR 数据。

**📈 对比分析**

与 BC、Diffusion Policy、GE+BC 等基线对比，实验表明 GE+CNMP 在撞击率、接近区违规、导航时间和成功率等六项社会指标上显著优于其他方法，逼近真实演示性能。

**⚠️ 局限性**

局限性包括图嵌入聚合方式在高密度群体下可扩展性待验证；未提供碰撞避免保证，实验环境缺乏静态障碍；训练数据量有限，模型在更大规模数据上的泛化未知。

---

## 61. Enhancing Oracle Bone Inscription Recognition via Multi-Scale Layer Attention

**arXiv ID:** 2607.00057 | [PDF](https://arxiv.org/pdf/2607.00057v1)

**作者:** Chaowen Yan `[一作]` (Sichuan Normal University), Tao He `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多尺度层注意力（Multi‑Scale Layer Attention）机制，显著提升甲骨文识别的精度。

**💡 创新点**

创新点在于通过全局平均池化与多尺度自适应池化生成丰富的 token 组，逐层累计 key/value 以实现跨层的细粒度多尺度交互，从而突破传统层注意力 token 稀缺的瓶颈。

**🔧 技术方法**

使用 ResNet 作为 backbone，结合多尺度 token 生成、线性投影、深度卷积增强以及多头注意力机制；同时采用逐层累积的 key/value 策略和通道注意力的方式对特征进行融合。

**📊 数据集**

实验数据集包括 Oracle‑MNIST、HUST‑OBS、OBC306 以及 EVOBC 四个大型甲骨文识别基准集。

**📈 对比分析**

与传统 Channel/Spatial 注意力、MRLA、DLA 等方法以及传统机器学习基线进行统一实验，结果显示在 Oracle‑MNIST 上 ResNet‑20/56/110 的精度分别提升至 96.05%、96.30%、96.45%；在 OBC306 上达到 91.29%；在 HUST‑OBS 上达到 94.12%；在 EVOBC 上为 49.45%，均优于或相当于现有最优方案。

**⚠️ 局限性**

局限性包括：多尺度 token 数量需手动调参，过多会产生噪声；对极大分辨率或极深网络的扩展性仍需验证；训练仍需较高 GPU 资源；在极度不平衡的大规模数据集上提升空间有限。

---

## 62. AlgoBench: Benchmarking Algorithmic Adaptation in Code Generation

**arXiv ID:** 2607.00062 | [PDF](https://arxiv.org/pdf/2607.00062v1)

**作者:** Xinyuan Song `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Algobench，一个基于已知竞赛题目通过结构化约束转换自动生成新算法问题的框架，并配套多层复杂度验证器和多维度性能指标，旨在评估 LLM 在算法推理与复杂度适应上的真实能力。

**💡 创新点**

创新点在于：① 自动化生成可追溯、非模板化的算法任务；② 通过四大质量门（旧解失效、语义相似过滤、新解验证、复杂度认证）确保任务难度与算法变更；③ 引入时间、空间、效率、陷阱率等六个复杂度感知指标；④ 通过三层（AST、标签、运行时）验证器客观判定最优性。

**🔧 技术方法**

主要技术包括规则化变换算子（约束缩放、静态→动态、目标扰动、贪心陷阱等）、文本相似度过滤、自动化引用解生成与暴力基线对照、静态 AST 与算法标签检测、校准的运行时复杂度拟合。

**📊 数据集**

数据集来源于 Codeforces、AtCoder、Kattis、LeetCode Hard 等公开竞赛题库，共 300 条原题，经过变换后生成 598 条变体（主要 420 条用于主实验）。

**📈 对比分析**

实验对比 7 款 LLM（GPT‑4o、GPT‑4o‑mini、Claude Haiku 4.5、Gemini 2.5 Flash、Llama‑3.3‑70B、GPT‑5.4、Claude Opus 4.5）与 6 种提示策略，使用 pass@1/5、Optimal‑Time、Optimal‑Space、Trap‑Rate 等指标。结果显示：在自动生成的变体上，pass@1 均下降 10–40%，最新模型相对旧模型下降幅度仍显著；而 Optimal‑Time 与 Optimal‑Space 明显低于 pass@k，暴露多数解答在复杂度上不合格；Trap‑Rate 证明模型多依赖旧模板，检索提示会在部分模型中加剧这一现象。

**⚠️ 局限性**

局限性包括：① 复杂度验证器仅支持 Python 与 C++；② 任务侧重于竞赛算法，几何、字符串、数论等领域覆盖不足；③ 变换算子及质量门依赖人工规则，可能缺乏通用性；④ 评价仍以公开数据集为主，缺少更广泛的真实世界编程任务。

---

## 63. Prompt Optimization for User Simulation in Conversational Recommender Systems: A Multi-Objective Framework

**arXiv ID:** 2607.00010 | [PDF](https://arxiv.org/pdf/2607.00010v1)

**作者:** Nipun B Nair `[一作]` (Monash University), Weiqing Wang `[通讯]` (Monash University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多目标框架，自动优化对话式推荐系统中LLM用户模拟器的提示词，降低正向偏差、数据泄漏和行为多样性不足。

**💡 创新点**

创新点在于将文本梯度(TextGrad)与熵感知评分函数相结合，实现无手工提示工程的自动化优化，并引入NegFeedback指标评估拒绝质量。

**🔧 技术方法**

技术手段包括LLM（Llama‑3.3‑70B）本地推理、文本梯度优化、熵和语义多样性度量、摘要化用户历史与自定义评估提示。

**📊 数据集**

使用Amazon 2023电影与电视评论数据集，并通过用户历史摘要生成模拟用户资料。

**📈 对比分析**

与GPT‑3.5‑Turbo、GPT‑4及多种上下文策略基线相比，经过优化的模拟器在五项任务（ItemsTalk、BinPref、OpenPref、RecRequest、NegFeedback）中分别提升了项目熵、相关性、情感熵、词句多样性以及与人类评测的相关性，性能显著优于基线。

**⚠️ 局限性**

局限包括评估样本规模有限、NegFeedback指标在小样本下统计显著性不足，以及对非电影类推荐场景的适用性尚待验证。

---

## 64. Progressive Pose-Guided 4D Animal Reconstruction from Monocular Video

**arXiv ID:** 2607.00157 | [PDF](https://arxiv.org/pdf/2607.00157v1)

**作者:** Siyuan Li `[一作]` (University of Alberta), Li Cheng `[通讯]` (University of Alberta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本论文提出一种进阶式测试时优化框架，利用3D高斯光散射（3D Gaussian Splatting）从单目视频中重建高保真4D动物模型，能够在任意视角和时间步长下进行自由视点渲染。

**💡 创新点**

创新点包括：① 将形状先验仅视为粗略初始化，并在优化过程中让皮肤化字段（skinning field）自适应；② 引入对称感知的时间编码，利用左右镜像信息补偿相机漂移；③ 通过可学习的部件锚点在姿态估计与非刚性变形阶段共享身份空间，实现姿态与形变的无缝分离；④ 采用分阶段优化与多项几何正则化，提升时空一致性与细节保留。

**🔧 技术方法**

核心技术包括：3D高斯光散射、线性混合皮肤化（LBS）、自注意力与交叉注意力机制、对称感知时间编码、可学习的皮肤化字段、全流程的梯度分离与正则化损失（如TV、Compact、Smooth）。

**📊 数据集**

实验使用了87段真实野外视频（来源包括DAVIS、APTv2和在线收集）以及9个CGI动物模型的Artemis数据集，后者提供多视角同步摄像机与真实表面点。

**📈 对比分析**

与五个基线（Fauna、GART、D‑3DGS、GVFDiffusion、DreamMesh4D）对比，本文方法在输入视图重建与新视图合成上均实现了最高的几何精度、时间一致性与视觉质量（如Chamfer、F‑score、KID‑V、FVD‑D），在大多数指标上超过对手，尤其在多种动物种类与复杂动作下保持稳定。

**⚠️ 局限性**

局限性主要体现在：① 对相机估计极度不稳定的场景仍易收敛到次优解；② 依赖Grounded‑SAM分割，掩码误差会直接影响几何精度；③ 对局部细节（如头部运动）仍依赖轮廓信息不足；④ 评估主要基于渲染指标，缺乏完整的3D几何基准。

---

## 65. Decompose, Compare, and Decide: Multimodal LLMs are Implicit Few-Shot Learners

**arXiv ID:** 2607.00125 | [PDF](https://arxiv.org/pdf/2607.00125v1)

**作者:** Yunhan Wang `[一作]` (University of Tuebingen), Hilde Kuehne `[通讯]` (University of Tuebingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于Decompose-Compare-Determine（DeCoDe）的无训练多模态大语言模型（MLLM）少样本图像分类方法。

**💡 创新点**

创新点在于将少样本分类拆分为一系列二元支持–查询图像比较，通过对“Yes” token 的logit进行排序来决定类别，且可利用域信息进一步提升性能。

**🔧 技术方法**

主要技术包括：结构化推理、对标记token的logit评分、域信息提示、匿名标签实验以及对比和评估的多种推理变体。

**📊 数据集**

在12个数据集上进行评估，包括6个标准少样本基准（mini‑ImageNet、CUB、Aircraft、Dogs、UCF101、DomainNet）和6个新颖域外数据集（Lego、Industrial Parts、Yoga、Hieroglyphs、Insect、Arabic Sign Language）。

**📈 对比分析**

与传统的单轮上下文提示、SFT、以及多种无训练基线（CLIP、DINOv2、Tip‑Adapter、SAVs 等）相比，DeCoDe 在标准和新颖数据集上均实现了显著提升，尤其在匿名标签场景下表现最为突出，达到或超过当前最先进水平。

**⚠️ 局限性**

局限性包括：需为每个类别执行多次推理（O(N) 计算），对标记token的logit依赖；在大规模N‑way任务下推理延迟显著增加；模型对域信息的选择敏感，需手工或自动化地提供合适的域描述。

---

## 66. FRAME: Learning the Adaptation Domain with a Mixture of Fractional-Fourier Experts

**arXiv ID:** 2607.00162 | [PDF](https://arxiv.org/pdf/2607.00162v1)

**作者:** Tom Saliencro `[一作]` (University of California, Irvine), Daniel Whitmore `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的参数高效微调方法——Fractional‑Fourier Mixture of Experts (FFMoE)，该方法为每个专家学习一个分数傅里叶变换（FrFT）阶数，使低秩更新在时域（空间域）与频域（谱域）之间的连续空间中自适应；同时采用 MoE 路由器将不同 token 送至不同专家。

**💡 创新点**

创新点在于把“更新在哪个域进行”这一设计自由度从固定转为可学习，通过分数傅里叶阶数在空间‑频谱连续体上插值，从而兼顾 LoRA（空间域）与谱域适配器的优点；学习到的阶数能让专家自然去相关，提升多任务兼容性与表达能力；并使用 O(d log d) 的 chirp‑FFT 近似，使额外计算成本几乎为零。

**🔧 技术方法**

核心技术包括：分数傅里叶变换（FrFT）与其 chirp‑FFT 近似；混合专家（MoE）架构与 token‑级路由；按阶数分组的负载均衡；对阶数采用单独优化器；以及对权重矩阵、秩参数和阶数的联合训练。

**📊 数据集**

实验使用 LLaMA‑3.1‑8B 与 Qwen2.5‑7B 两大基础模型，覆盖四大任务族：commonsense reasoning（BoolQ、PIQA、SIQA、HellaSwag、WinoGrande、ARC‑easy、ARC‑challenge、OBQA）、math reasoning（GSM8K、MATH、SVAMP、MAWPS、AQuA）、code（HumanEval、MBPP）与 knowledge（MMLU、ScienceQA）；对尺度进行了 1.5B/3B/14B 的扩展验证。

**📈 对比分析**

与单一适配器（LoRA、DoRA、AdaLoRA、FourierFT）及多专家基线（HydraLoRA、MixLoRA、HMoRA、FlyLoRA、MoA、FourierMoE）对比，FFMoE 在两大 backbone 上的平均准确率均居首，提升幅度约为 +0.7–+2.6 分；且仅激活 0.31% 参数，低于多数 MoE‑LoRA 基线。Ablation 结果进一步验证了可学习阶数与分组负载均衡的必要性。

**⚠️ 局限性**

局限性包括：每个专家需额外存储一个阶数与一个变换，虽然通过 chirp‑FFT 近似降低成本，但随着活跃专家数 k 增大仍会增加计算开销；阶数仅为标量，缺乏更细粒度（如轴向或波形）变换；对学习到阶数的解释多为相关性而非因果性；以及与其他 MoE 方案一样，对路由器收敛仍有一定敏感性。

---

## 67. A Unified Benchmark for RCM-Constrained Visual Servoing: Modeling-Controller Interaction and Robustness Analysis in Laparoscopic Robots

**arXiv ID:** 2607.00030 | [PDF](https://arxiv.org/pdf/2607.00030v1)

**作者:** Jing Zhang `[一作]` (Sun Yat-sen University), Mengtang Li `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在机器人辅助腹腔镜手术中，提出了一个统一的仿真基准框架，该框架集成了三种典型的远程中心约束（RCM）建模方法与六种基于视觉伺服的控制架构，并在同一速度层面统一表述，便于在相同条件下对不同建模与控制的交互进行系统评估。

**💡 创新点**

创新点在于：① 将三种RCM模型（平面切线、轴投影、虚拟关节）用统一的约束矩阵形式表达；② 构建模块化、开源的仿真平台，支持三种RCM与六种控制器的组合；③ 通过结构化案例揭示RCM建模与控制器之间的关键敏感性（切线平面定义、约束维数、闭环反馈、奇异点鲁棒性），为后续研究提供可重复的基准。

**🔧 技术方法**

主要技术包括：CoppeliaSim 机器人仿真、MATLAB 远程API 接口；基于视觉伺服的特征提取与雅克比计算；三种控制框架实现：伪逆（PI）方法、三种二次规划（QP）方法（等式、目标权重、层级），以及逆运动学（IK）方法；以及RCM约束的速度层面表达和闭环误差反馈。

**📊 数据集**

未使用真实数据集，而是采用仿真生成的二维/三维目标（红球、绿矩形、光学标记），并设计多种运动轨迹（直线、螺旋、2-DOF）来验证控制效果。

**📈 对比分析**

比较方法：在相同的目标轨迹与初始条件下，交替使用不同RCM建模与控制器组合，记录RCM误差、视觉特征误差、关节运动量、λ参数演化等指标。结果表明：① 关闭式RCM反馈能消除稳态误差；② 在奇异点附近，QP方案（尤其是层级QP）保持任务收敛与RCM约束；③ 释放轴向约束（N_rcm=2）可减少关节运动但导致RCM漂移；④ 平面切线定义对RCM1敏感，使用局部RCM帧能显著降低误差。

**⚠️ 局限性**

局限性：仅在刚性腹腔镜模型下验证，未考虑动力学、柔性杆、外部扰动；仿真环境与真实手术环境差异较大；控制器参数选择依赖手工调优；未验证多目标或多相机设置的鲁棒性；代码公开但缺乏长时间连续仿真与实验验证。

---

## 68. Lost in the Tail: Addressing Geographic Imbalance in Urban Visual Place Recognition

**arXiv ID:** 2607.00090 | [PDF](https://arxiv.org/pdf/2607.00090v1)

**作者:** Zhiyao Shu `[一作]` (George Mason University), Da Chen `[通讯]` (University of Bath)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对城市规模视觉地点识别（VPR）中存在的地理长尾不平衡问题，提出分布感知位置识别框架 DAPR，包含低访问偏差损失（LB loss）在训练阶段重平衡梯度、以及多尺度特征频域距离（CFD）在检索阶段实现分布感知多尺度匹配，从而显著提升 VPR 性能。

**💡 创新点**

创新点在于：① 设计了低访问偏差损失，将逆频率加权与 logit 校正结合，平衡头尾类梯度贡献；② 引入特征频域特征函数距离（CFD），通过自适应幅度与相位权重实现分布感知的多尺度检索；③ 这两项技术均可作为无关模型的插件，兼容多种 VPR 体系。

**🔧 技术方法**

技术细节包括：逆频率加权、logit 调整、角度余弦分类、DINOv2 Backbone、CrossEntropy 与 Contrastive 训练、Multi‑Similarity 以及特征频域特征函数（CFD）计算。

**📊 数据集**

实验数据集涵盖：大规模 SF‑XL（41M 图像）、MSLS、Pitts30k 以及 Nordland（极端季节变化验证）。

**📈 对比分析**

在 SF‑XL 上与 D&C、SALAD、BoQ 等基准对比，DAPR‑M 在 test‑v1 上 R@1 提升 18.3%（从 71.4% 提升到 89.7%），在 test‑v2 上提升 6.7%；在 MSLS 与 Pitts30k 上亦提升 2–3%；相对全数据库检索方法，DAPR‑M 在速度上快 60 倍以上。

**⚠️ 局限性**

局限性：仅基于样本计数定义长尾，未考虑特征层面的长尾分布；重平衡侧重地理类别，可能忽略同一类内的视觉多样性；特征频域距离在推理时增加计算开销；未对不同城市的地理分布差异进行更细粒度评估。

---

## 69. Data Sharing and Competition in Learning-by-Deploying Industries: Insights from Robotics and Beyond

**arXiv ID:** 2607.00168 | [PDF](https://arxiv.org/pdf/2607.00168v1)

**作者:** Yunjin Tong `[一作]` (Stanford Graduate School of Business), Luca-Andrei Manea `[通讯]` (Stanford Graduate School of Business)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文研究在学习-部署（learning‑by‑deploying）情景下，数据共享（pooling）对企业投资决策和社会福利的影响，构建了一个两期动态模型并在其基础上加入Cournot竞争与价格内生化，分析共享学习的可持续性、企业的低投资现象以及竞争如何削弱共享的私利；进一步用一般需求函数推导共享的可持续阈值。

**💡 创新点**

创新点包括：①首次将学习体系的拓扑（聚合式与碎片化）与企业投资与竞争联系起来，提出福利阶梯与共享可持续阈值；②揭示在竞争环境下共享可能导致私利负面并给出需求弹性决定可持续性的定量条件；③给出闭式阈值（线性需求）与通用弹性阈值（一般需求），并通过数值实验验证结果稳健。

**🔧 技术方法**

采用了两期动态博弈与容量投资模型、学习曲线函数、最优条件与比较静态分析；利用福利阶梯引理、极值定理与包络分解法求解共享可持续阈值；在一般需求下使用弹性表达式；最后采用数值求解（对不同需求形态与企业数量进行模拟）来检验理论预测。

**📊 数据集**

本研究为理论性论文，无实证数据集；在数值实验中仅使用三种形式的需求函数（等弹性、Logit、凸型）来说明结果对需求形态的敏感性。

**📈 对比分析**

通过构造三种共享/碎片化的博弈框架，计算出每种情形下的最优容量、利润与社会福利；对线性需求得到闭式可持续阈值b<a/[2K(η^P+η^F)]；在一般需求下用需求弹性>1的条件判断共享是否可持续。数值实验表明：在弹性或成熟市场共享倾向降低早期投资，且共享收益为正；在容量受限市场共享导致早期投资提升但不自持。

**⚠️ 局限性**

局限性包括：仅考虑两家对称企业和两期模型；假设全容量利用、成本结构为二次型；未考虑企业异质性、供应商动态、多厂商竞争与平台互操作性；闭式阈值仅适用于外生容量或无二期扩张情况，通用阈值缺乏解析形式；缺乏实证检验。

---

## 70. Towards an automated AI-based framework for floor plan compliance checks for residential buildings

**arXiv ID:** 2607.00015 | [PDF](https://arxiv.org/pdf/2607.00015v1)

**作者:** Subash Gautam `[一作]` (RMIT University), Sarah Foster `[通讯]` (RMIT University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个面向多单元公寓的自动合规检查框架，涵盖了图像预处理、文本识别、结构元素分割、空间图构建、LLM规则解析与执行以及合规评估等模块。

**💡 创新点**

创新点主要有：① 将大型语言模型（LLM）与检索增强生成（RAG）结合，对自然语言建筑规范自动转换为可执行的JSON规则；② 在图像解析层面融合多任务CNN、实例分割与北向角度检测，构建可解释的空间图谱；③ 通过规则引擎与合规引擎的分层设计，实现对不同州标准（SEPP 65、BADS、SPP 7.3）以及健康指标（High Life Study）的统一评估；④ 提出人机协同的质量控制与迭代改进流程。

**🔧 技术方法**

技术包括：大型语言模型（GPT‑4/LLM）、检索增强生成（RAG）、视觉语言模型（VLM）、卷积神经网络（U‑Net、Mask R‑CNN、YOLO、FloorNet）、语义分割与实例分割、OCR（Tesseract/PaddleOCR）、图谱构建与图神经网络、北向角度检测算法、规则引擎与合规引擎（JSON、XPath、规则引擎框架）。

**📊 数据集**

使用的数据集有：SFC‑A68（多单元公寓楼盘），Modified Swiss Dwellings（多单元平面图），MLStructFP（多单元平面图），以及High Life研究中采集的113栋楼盘（包含10 553套公寓）的手工提取指标；此外还利用公开的BADS、SEPP 65、SPP 7.3规范文本。

**📈 对比分析**

方法主要通过对比人工手工合规检查与系统自动评估结果来验证可行性，展示了在多单元楼盘上成功识别单元、检测空间与符号、构建空间图谱，并在规则引擎中实现对面积、窗口、通风、日照等指标的自动判定。由于目前仍处于概念验证阶段，文中未给出大规模量化指标；但实验示例表明系统能够在单层楼层图上完成全部模块流程，符合设计标准的合规判定结果与人工评估高度一致。

**⚠️ 局限性**

局限性包括：① 图纸质量和格式多样性导致OCR与分割误差，需人工审核；② 现有VLM在精确几何推理方面不足，易产生尺寸错误；③ 单元分割仍缺乏专门算法，难以在复杂走廊结构中自动聚类；④ LLM规则解析可能出现hallucination，需要检索增强或人工校验；⑤ 缺乏大规模公开评测基准，无法全面量化性能；⑥ 对法规更新的自适应能力尚需进一步验证。

---

## 71. Segmenting, Fast and Slow: Real-Time Open-Vocabulary Video Instance Segmentation with Dual-Path Processing

**arXiv ID:** 2607.00124 | [PDF](https://arxiv.org/pdf/2607.00124v1)

**作者:** Luca Barsellotti `[一作]` (University of Modena and Reggio Emilia), Maxim Berman `[通讯]` (Google)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为 SegFS 的双流实时开词视频实例分割框架，能够在移动设备上实现高帧率分割。

**💡 创新点**

创新点在于将关键帧的实例语义直接投影到 backbone 的特征空间，快路不再使用计算量巨大的特征增强器和像素解码器，从而显著降低推理延迟。

**🔧 技术方法**

采用对象中心检测器（如 GLEE、MOBIUS、TROY‑VIS）作为慢路；快路使用 Fast Feature Aggregator、Object Guidance、投影 FFN、DSConvGN、背景 token 等轻量模块，并结合自注意力进行实例投影和融合。

**📊 数据集**

训练集覆盖 COCO、LVIS、BDD、YouTubeVIS19/21、OVIS、RefCOCO/+, RVOS、UVO、SA‑1B；评估集包括 YouTubeVIS19、OVIS、BURST、LV‑VIS。

**📈 对比分析**

与多种基线（MOBIUS‑Mini‑M、GLEE‑Lite、TROY‑VIS）以及光流、MPVSS 等方法对比，SegFS 在保持 AP 差距 ≤1 的同时将延迟降低约 14×，实现 30 FPS 以上的实时性能。

**⚠️ 局限性**

局限性包括关键帧间距增大时性能衰减、对极高分辨率或大词汇表的适配仍有限，以及对更广泛场景的验证仍待进一步研究。

---

## 72. SNAP-FM: Sparse Nonlinear Accelerated Projection for Physics-Constrained Generative Modeling

**arXiv ID:** 2607.00095 | [PDF](https://arxiv.org/pdf/2607.00095v1)

**作者:** Alaina Kolli `[一作]` (Massachusetts Institute of Technology), Christopher Vincent Rackauckas `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种加速物理约束生成模型采样的框架——SNAP‑FM，利用稀疏非线性投影加速硬约束采样，确保生成样本严格满足物理守恒、边界条件等非线性约束。

**💡 创新点**

创新点在于：①识别并利用样本批处理与局部PDE耦合导致的块稀疏雅可比与KKT系统结构；②使用ExaModels对投影子问题进行符号编译，保留结构；③在GPU上通过MadNLP+cuDSS实现稀疏内部点法求解，大幅提升速度与可扩展性。

**🔧 技术方法**

主要技术包括ExaModels建模框架、MadNLP内部点求解器、NVIDIA cuDSS GPU稀疏因子分解、符号自动微分与GPU并行计算、Physics‑Constrained Flow Matching（PCFM）算法。

**📊 数据集**

使用一系列PDE基准：1D热方程（含初值、质量守恒、能量演化）、1D反应扩散方程（Neumann BC与非线性守恒）、1D Burgers方程（不同边界/初值/质量/通量约束）、二维Navier‑Stokes（初值、总涡度守恒、耗 enstrophy）等。

**📈 对比分析**

与JuMP+MadNLP、JuMP+Ipopt、Optimization.jl+IPNewton、L‑BFGS等基线（CPU/ GPU）进行对比；实验显示ExaModels+MadNLP GPU在所有六个约束场景中均最快，且约束满足度与基线相当或更优；CPU版相对慢，非结构化基线往往失效或耗时过长。

**⚠️ 局限性**

局限性包括：①对GPU硬件和专门稀疏求解器的依赖；②对批量样本间约束独立且局部稠密的假设；③在极端非线性或全局耦合约束下仍可能收敛缓慢或失败；④需手动构建符号模型和调节可行性容忍度，使用门槛较高。

---

## 73. Topological Void Analysis A Mathematical Framework for Systematic Technical Innovation Discovery in Knowledge Spaces

**arXiv ID:** 2607.00005 | [PDF](https://arxiv.org/pdf/2607.00005v1)

**作者:** Kris Pan `[一作]` `[通讯]` (Intel Corporation), Kris Pan (Intel Corporation)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于拓扑空洞分析（TVA）的技术创新发现框架；

**💡 创新点**

定义了四个条件（领域连贯性、校准边际性、稀疏词桥、空洞性）构成的拓扑空洞，利用球面插值检测未被占据的语义区域，从而系统化地识别技术创新点；

**🔧 技术方法**

使用BGE‑M3稠密/稀疏嵌入、SLERP球面插值、稀疏词集、自动阈值校准、FAISS索引、LLM生成技术方案及多目标评分；

**📊 数据集**

构建约14万条Linux内核与x86硬件文档、学术论文、专利文本的混合语料库；

**📈 对比分析**

对比传统关键词检索、MMR多样性检索及向量算术生成，TVA在96个目标领域共生成2128候选，经过自动与人工评审后获得191条可修订、1条批准的创新方案，显示出高质量筛选效率和创新率；

**⚠️ 局限性**

主要局限包括对稀疏域的阈值校准可能引入噪声、LLM生成的创新方案仍需人工审核、缺乏跨域验证以及评估过程依赖于专业评审团队。

---

## 74. Evaluating Hardware Abstraction Layer Concepts for Software Defined Vehicles: Insights into Applicability and Effectiveness

**arXiv ID:** 2607.00039 | [PDF](https://arxiv.org/pdf/2607.00039v1)

**作者:** Akshay Narla `[一作]` (University of Stuttgart), Michael Weyrich `[通讯]` (University of Stuttgart)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

本文系统评估了汽车及跨领域的 HAL 方案，构建了基于 MoSCoW 的评价框架，并提出了结合 hypervisor 隔离与 middleware 标准化的混合 HAL 架构；

**💡 创新点**

创新点在于提出了混合 HAL 模型，既利用 hypervisor 的安全隔离，又借鉴安卓、NFV 等领域的可扩展 API 机制；

**🔧 技术方法**

采用 MoSCoW 权重评分、跨域最佳实践对比，以及 ISO 26262 合规性、OTA、低延迟等技术指标进行评估，引用 AUTOSAR Adaptive、QNX Hypervisor、Xen、EB Corbos 等技术；

**📊 数据集**

评估数据来源为公开论文、官方技术文档、标准化认证记录，未使用专门的实验数据集；

**📈 对比分析**

通过最大 45 分的点数评分对 HAL 进行量化比较，结果显示 hypervisor 在安全、OTA 方面得分最高，middleware 在可移植性和开发便利性上表现更佳，但无单一方案能完整满足所有 SDV 需求；

**⚠️ 局限性**

研究局限在于依赖公开信息的完整性、开放源代码 HAL 的安全认证不足，以及未在真实 SDV 原型上验证混合 HAL 的实际性能。

---

## 75. EVOTS: Evolutionary Transformer Search for Time Series Forecasting

**arXiv ID:** 2607.00154 | [PDF](https://arxiv.org/pdf/2607.00154v1)

**作者:** AbdElRahman ElSaid `[一作]` (University of North Carolina Wilmington), Damir Pulatov `[通讯]` (University of North Carolina Wilmington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出EvoTS框架，利用进化搜索自动发现适用于多变量时间序列预测的Transformer-like架构，并在ETT基准上进行全面评估。

**💡 创新点**

创新点在于引入模块化基因组与结构修复机制，使得架构可以在固定训练协议下自由组合多种注意力和卷积块，进而在长预测时段和多变量预测中实现比手工设计更优的模型。

**🔧 技术方法**

采用稳态进化算法、权重继承（未启用）、多种变异/交叉算子、结构修复、统一的AdamW训练、20轮短周期评估等技术实现搜索与训练。

**📊 数据集**

使用四个ETT基准数据集（ETTh1、ETTh2、ETTm1、ETTm2），采样频率分别为小时和15分钟。

**📈 对比分析**

与iTransformer基准在相同数据划分和训练协议下直接比较；在多变量-多变量预测下，EvoTS在所有预测窗口（96/192/336/720）上均取得比iTransformer更低或相近的MSE，短期提升约10–15%，长期提升可达20–25%。

**⚠️ 局限性**

限制在于搜索空间仅包含单阶段结构，未探索多阶段重标记；仅在固定训练协议下搜索；未充分利用权重继承；对单变量和多变量-单变量场景的改进相对有限。

---

## 76. Dual-Informed Vertical Expansion for Multi-Objective Node Selection in Anytime Conflict-Based Search

**arXiv ID:** 2607.00156 | [PDF](https://arxiv.org/pdf/2607.00156v1)

**作者:** Willem van Osselaer `[一作]` (Massachusetts Institute of Technology), Gioele Zardini `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的高层节点选择策略 acr:dive，用于改进 Conflict‑Based Search（CBS）在多机器人路径规划中的搜索效率与内存占用。

**💡 创新点**

创新点在于将节点选择视为独立模块，证明其不影响 CBS 的完整性，并通过在每次最佳边界回归后进行深度优先 “潜水” 以及利用可行解进行剪枝，实现更好的父子连续性、早期可行解发现和更小的队列。

**🔧 技术方法**

使用的技术包括分支定界理论、CBS 的节点选择框架、最佳边界重定位、深度优先潜水、可行解剪枝、实验评测以及对比实验。

**📊 数据集**

实验使用了四类标准 MAPF 基准地图（如 Warehouse、Open、etc.）以及 10%/15% 机器人密度和 30% 密集实例的测试集。

**📈 对比分析**

对比方法是将 acr:dive 与标准 best‑first（BFS）和迭代加深（ID）在节点扩展数、深度切分次数、最大队列大小以及任何时性能（可行解时间/Gap）等指标进行评估；实验显示 acr:dive 在保持较低队列和较少深度切分的同时，能更快地发现可行解并提供预证，整体性能处于两者之间，且在内存受限或需要快速可行解的场景表现突出。

**⚠️ 局限性**

局限性包括：在某些实例上节点扩展量仍高于 BFS；在极密集或极稀疏环境下需要 warm‑start 进行辅助；目前缺乏自适应混合策略，无法在不同资源/时间约束下自动调优节点选择。

---

## 77. Vertigo Vertigo: Reconstructing a Cinematic Ideal through its Predictive AI Double

**arXiv ID:** 2607.00047 | [PDF](https://arxiv.org/pdf/2607.00047v1)

**作者:** Adam Cole `[一作]` (University of the Arts London), Mick Grierson `[通讯]` (University of the Arts London)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

用稀疏关键帧（仅占原片的2.78%）通过视频扩散模型实现阿尔弗雷德·希区柯克《Vertigo》的全片 AI 重建，并将重建结果与原片进行叠加、并排与差异显示。

**💡 创新点**

创新点在于：①首次以稀疏关键帧为输入，利用首-尾帧插值完整重现经典影片；②将计算机生成的“预测影子”与原片对齐，揭示生成模型对经典电影结构与美学规范的隐性“记忆”与“继承”；③在全长影片级别提供连续帧级别的差异量化分析，量化模型在不同片段与场景的重建质量。

**🔧 技术方法**

核心技术：大规模图像到视频扩散模型（Wan 2.2，14 B 参数），实现首-尾帧插值；自定义关键帧提取与分段算法；后期多模式展示（叠加、并排、差异）。

**📊 数据集**

数据集：阿尔弗雷德·希区柯克《Vertigo》（1958）原片，仅取 2.78% 的关键帧（约 3,416 幅）作为输入；原片全长 122,668 帧用于评估和对比。

**📈 对比分析**

比较方法：使用 SSIM（结构相似度）、LPIPS（感知相似度）和相邻帧的时间变化量，构建加权综合指标；每帧划分为 5 个质量层级。结果显示：约 73.1% 的帧可被识别为《Vertigo》的可接受重现，16.8% 接近原片，3.6% 为严重失败；在关键帧处重建质量最高，插值区间呈 U 型波动，长度与误差无显著相关。相比传统逐帧或文本提示生成，模型在保持结构一致性的同时，显著压缩了输入数据。

**⚠️ 局限性**

局限性：①对情感表达、对白、人物面部表情的预测不足，导致情绪转场重建失败；②在光照较弱或动画（如约翰·惠特尼开场序列）等视觉复杂场景中误差显著；③模型的预测能力高度依赖训练数据对经典电影的覆盖度，无法很好泛化到风格迥异的导演作品；④当前方法仍需人工分段与后期剪辑，缺乏端到端自动化；⑤技术层面受限于 14 B 参数模型的生成时间与资源需求。

---

## 78. Persona Without Substrate: Regime-Dependence and the LLM Individuation Problem

**arXiv ID:** 2607.00006 | [PDF](https://arxiv.org/pdf/2607.00006v1)

**作者:** Shuaizhi Cheng `[一作]` `[通讯]` (Harbin Institute of Technology), Shuaizhi Cheng (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在 Qwen3-4B-Instruct 与 Mistral-7B-Instruct‑v0.2 上开展 persona‑topology 实验，检验并否定了 persona‑vector 在不同 elicitation regime（prompt‑extraction、fine‑tune、inference‑time steering）下的跨 regime 共参照假设，进而提出 regime‑indexed individuation 框架，并重新解读 Beckmann‑&‑Butlin 的三种 LLM mind 位置；

**💡 创新点**

创新点在于揭露 persona‑vector 研究中的隐式 cross‑context 共参照假设，将其归因为 Field‑style partial‑reference，提出 probe‑equivalence 关系与 (vehicle, regime) 统一身份条件，并用四个实证 wedges 证明跨 regime 同一向量不是单一实体；

**🔧 技术方法**

采用对比式激活向量提取（contrastive direction）、LoRA fine‑tune、inference‑time 激活算术、细胞拼接（chimera）训练、激活补丁与因果删失等机制可解释技术；

**📊 数据集**

实验使用 Qwen3-4B‑Instruct 与 Mistral‑7B‑Instruct‑v0.2 两大模型；在每个模型上使用 99 次 LoRA fine‑tune 以及多种 persona‑anchor 数据集（Stalin‑90‑facts、Hitler‑90‑facts、Gandhi‑90‑facts、Floob 等）；

**📈 对比分析**

通过对比 prompt‑extracted 向量与 fine‑tune basin 方向的余弦相似度、Floob 反转实验、混合 anchor 的线性插值偏差以及 inference‑time 与 fine‑tune‑time 组合算术的非一致性，证明了 persona‑vector 在不同 regime 下行为不一致，显示传统共参照假设失效；

**⚠️ 局限性**

局限性包括仅检验两种模型与有限的 persona 维度，未系统探索更大模型或不同训练分布下的 regime‑索引效应；实验依赖特定实验设置，难以直接推广至所有 LLM 或其他 interpretability 方法；

---

## 79. Stop Hand-Holding Your Coding Agent: Engineering the Loops that Replace Step-by-Step Prompting

**arXiv ID:** 2607.00038 | [PDF](https://arxiv.org/pdf/2607.00038v1)

**作者:** Sandeco Macedo `[一作]` `[通讯]` (Instituto Federal de Goiás), Sandeco Macedo (Instituto Federal de Goiás)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并系统化了循环工程（Loop Engineering）实践，定义循环规范的五要素（触发、目标、验证、终止规则、记忆），构建了规范的解剖学与分类，并对公开的 Loop Library 中的 50 条循环进行手工编码与统计分析，提出设计原则与反模式，并发布了自动化循环规范编写工具。

**💡 创新点**

创新点在于将循环规范视为独立可重用工件，构建了五层验证阶梯与四个模式族，阐明循环规范与提示、上下文、引擎的层级关系，并首次在实证语料上揭示实践成熟度与不足。

**🔧 技术方法**

采用文本手工编码、语义分类与统计分析，利用 LLM（Claude、ChatGPT 等）构建循环规范编写工具，并借鉴模型裁判理论进行验证设计。

**📊 数据集**

使用公开 Loop Library 公开库中的 50 条循环规范及其元数据作为实验数据集。

**📈 对比分析**

本文未进行对照实验或性能评估，仅提供描述性统计（如 70% 落在自治区，自动触发仅 22% 等），未给出具体性能指标。

**⚠️ 局限性**

局限性包括：仅为位置论文，语料不具代表性；手工编码存在主观偏差；缺乏实验验证循环规范的效果和成本收益；发布的工具未进行使用评估。

---

## 80. The Illusion of Safety: Multi-Tier Verification of AI vs. Human C++ Code

**arXiv ID:** 2607.00107 | [PDF](https://arxiv.org/pdf/2607.00107v1)

**作者:** Saif Mahmud `[一作]` (University of Texas at Arlington), Lei `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了VulBench-CPP基准，收集了8918个C++程序（包括三大开源LLM生成的7213个与1,695个人类参考实现），并对每个程序进行功能测试、静态分析、动态分析与形式化验证四层级的安全评估。

**💡 创新点**

首次在C++上实现多层级验证框架，并提供人类基线与AI代码直接对比，揭示仅靠静态分析会掩盖AI生成代码3.6倍的运行时安全缺陷。

**🔧 技术方法**

使用了cppcheck、clang-tidy进行静态分析；ASan/UBSan进行动态检测；ESBMC进行有界模型检验；同时结合g++编译、测试套件执行和混合效应逻辑回归等统计方法。

**📊 数据集**

数据来源为CodeContests竞赛题集，选取851道题目的人类解决方案（共1,695个）和三大LLM（Gemma3 27B IT、LLaMA3.3 70B Instruct、Qwen2.5 Coder 32B Instruct）各自生成三份独立解答。

**📈 对比分析**

对比方法为多维度安全率统计和交叉工具一致性（Jaccard相似度），结果显示AI代码在动态检测下的违规率为9.3%，人类为2.6%；在形式化验证下AI违规率在25.9–36.7%之间，而人类仅18.8%；静态分析两者相似但并不能反映真实风险。

**⚠️ 局限性**

局限包括ESBMC在C++前端解析错误率高达56.5%，仅覆盖部分程序；动态检测受限于可用测试用例；模型生成的随机性导致对特定任务的代表性不足；基准仅覆盖竞赛级算法代码，可能不完全代表生产环境。

---

## 81. Why Advanced Encoders Lag on Sparse Retrieval? The Answer and an Approach to Bridging Vocabulary Gaps

**arXiv ID:** 2607.00004 | [PDF](https://arxiv.org/pdf/2607.00004v1)

**作者:** Zhichao Geng `[一作]` (Amazon Web Service), Yang Yang `[通讯]` (Amazon Web Service)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为 Vocabulary Transfer (VT) 的通用框架，用以将现代大型预训练语言模型（如 ModernBERT、RoBERTa 等）迁移到与稀疏检索兼容的规范化词表，从而消除“词汇差距”导致的性能滞后。

**💡 创新点**

创新点包括：① 通过理论分析证明词表粗化（coarse‑graining）能降低假设类复杂度并提升泛化；② 设计了基于空间拓扑的语义初始化（Semantic Initialization via Spatial Topology）和激活潜力校准（Activation Potential Calibration, APC）两种高效适配策略；③ 将上述方法应用于多种编码器与稀疏检索体系，首次在 BEIR、MS MARCO、TREC‑DL 等基准上实现 state‑of‑the‑art 结果。

**🔧 技术方法**

技术细节：使用 Transformer‑based 预训练模型；对词表进行粗化和迁移；语义初始化通过稀疏最大化（sparsemax）构造邻域加权；APC 通过对 MLM 后输出偏置的平移调节 ReLU 的激活率；在稀疏检索中采用 SPLADE 结构与 MarginMSE 损失；整体训练包含少量 MLM 步骤（≈ 500–20k）和标准的稀疏检索微调。

**📊 数据集**

数据集：① Wikipedia + BookCorpus（约 6.2M 文档，3.7B tokens）用于 MLM 适配；② MS MARCO Passage Retrieval（Hard Negatives）用于在域内微调；③ BEIR 13 任务（TREC‑COVID、NQ、HotpotQA 等）用于跨域评估；④ TREC‑DL 2019 评测；⑤ 语料库 ChemHotpotQA、ChemNQ 用于领域迁移实验。

**📈 对比分析**

比较方法：与 BM25、DPR、ColBERT、CoCondenser、SPLADE‑v3、ESPLADE 等基线在 BEIR、MS MARCO、TREC‑DL 上对齐训练；结果显示 VT‑迁移后 ModernBERT 在 BEIR 上 nDCG@10 提升至 52.4（比 CoCondenser‑Ensemble 低 0.5、比 SPLADE‑v3 高 0.4），在 MS MARCO MRR@10 上达 38.3，显著优于原始稀疏微调模型；在推理‑free 方案中亦取得 51.5 nDCG@10，展示出与稀疏检索架构的兼容性。

**⚠️ 局限性**

局限性：① 仍需先行训练或获取目标词表的预训练嵌入（对低资源语言或自定义词表挑战）；② 迁移后对新词的适配依赖少量 MLM 步骤，若训练数据极其稀缺可能效果受限；③ 主要在英文语料上验证，跨语言扩展需进一步研究；④ 在某些专用领域词表过细时可能导致稀疏性下降或激活率失衡。

---

## 82. Benchmarking Frontier LLMs on Arabic Cultural and Sociolinguistic Knowledge: A Cross-Evaluation Framework with Human SME Ground Truth

**arXiv ID:** 2607.00139 | [PDF](https://arxiv.org/pdf/2607.00139v1)

**作者:** Sajjad Abdoli `[一作]` (Perle AI), Ahmed Rashad `[通讯]` (Perle AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种跨评估框架，用于在埃及和伊拉克阿拉伯语方言下评估前沿大语言模型（LLM）在阿拉伯文化与社会语言学任务中的自动评分能力。

**💡 创新点**

创新点包括：1）提供“提供者级自评”机制，防止同一模型厂商的输出被其自身评判；2）构建基于惩罚权重的轮廓式评分表，区分正向内容与错误负项；3）引入双重度量（MAD+Signed Mean Error）以分离偏差与噪声。

**🔧 技术方法**

使用技术主要是LLM‑as‑Judge框架、结构化评分表生成、文本匹配与错误分类；评估时对所有评判模型施以固定温度、专用系统提示，确保评判过程可重复。

**📊 数据集**

数据集包含103个已验证的提示–评分表对（埃及70条、伊拉克33条，文化53条、语言学50条），由本地方言专家编写与标注，覆盖文化与语言学两大子领域。

**📈 对比分析**

比较方法：在所有模型间执行交叉评估，计算每个评判模型的Mean Absolute Deviation（MAD）与Signed Mean Error；结果显示GPT‑5.4最为可靠（MAD 10.21pp，偏差‑1.12%），其余四个评判模型普遍存在偏袒（偏差+2.01%至+6.56%），且文化任务的MAD始终高于语言学任务（差距1.83–4.78pp）。

**⚠️ 局限性**

局限性：1）每个样本仅由一位专家标注，缺乏交叉标注的可靠性评估；2）埃及与伊拉克专家的严格程度不同，导致跨方言比较受偏差影响；3）仅覆盖两种阿拉伯方言，未能推广至其他中东北非方言。

---

## 83. BaRA: BFS-and-Reflection Web Data Collection Agent

**arXiv ID:** 2607.00007 | [PDF](https://arxiv.org/pdf/2607.00007v1)

**作者:** Soojeong Lee `[一作]` (Yonsei University), Kyungwoo Song `[通讯]` (Yonsei University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于受限 BFS 与自我反思的 Web 数据收集框架 BaRA，专注于在固定交互预算内完成站点级别的数据抓取；

**💡 创新点**

创新点在于将受限宽度优先搜索与历史驱动的自我反思相结合，既保证了页面覆盖率，又通过迭代修正失误来提升可下载的多模态信息质量；

**🔧 技术方法**

核心技术包括：受限 BFS 轨迹控制、URL 规范化与去重、页面级受限滚动与提取、下载有效性验证以及基于执行历史的自我反思循环；

**📊 数据集**

使用 50 个人工合成主题网站作为基准，并在三大真实网站（含交互繁重、动态渲染和异构布局）上进行测试；

**📈 对比分析**

与 Pure LLM、SeeAct‑Vision 和 Browser‑use 三个基线对比，BaRA 在链接发现、文本、图片和视频的精确率/召回率/准确率均显著优于基线，尤其在动态网站上提升最为明显；

**⚠️ 局限性**

局限包括：仅支持单页面保持和受限滚动，无法处理多标签、弹窗、登录墙等交互；性能仍受 LLM 可靠性与浏览器稳定性影响；且可能触犯网站使用条款和版权问题。

---

## 84. Iterated Invariant EKF for 3D Landmark-Aided Inertial Navigation

**arXiv ID:** 2607.00145 | [PDF](https://arxiv.org/pdf/2607.00145v1)

**作者:** Hilton Marques Souza Santana `[一作]` (Pontifical Catholic University of Rio de Janeiro), Marco Antonio Meggiolaro `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出并系统评估了迭代不变扩展卡尔曼滤波器（IterIEKF）在MAV 3D地标辅助惯性导航中的应用，首次完成了该滤波器在该框架下的完整推导与实现，并通过大量蒙特卡罗仿真验证其性能；

**💡 创新点**

创新点包括：① 在SE₂(3) Lie群上采用右不变误差参数化，消除了传统SO(3)-EKF存在的伪可观性问题；② 将局部Gauss‑Newton迭代更新融入不变测量更新，显著降低线性化误差，保证估计状态始终位于观测子流形上；③ 这是迄今为止首个将IterIEKF应用于MAV三维地标导航的工作；

**🔧 技术方法**

技术手段主要为：Lie群不变滤波（SE₂(3)×ℝ⁶）、右不变EKF、迭代不变EKF（Gauss‑Newton迭代）、蒙特卡罗仿真、NEES与MAE一致性与精度评估；

**📊 数据集**

使用的实验数据集为EuRoC MAV数据集（V2_01_easy.csv）并在世界坐标系中人工添加了三个位于固定位置的三维地标；

**📈 对比分析**

通过与SO(3)-EKF、IterSO(3)-EKF、IEKF在50次蒙特卡罗实验中的比较，使用位置、速度、重力方向的MAE和NEES等指标，IterIEKF将位置误差从9.36 m降至0.096 m（≈80%提升），速度误差从4.92 m/s降至0.095 m/s，同样在一致性（NEES≈15）和收敛速度（10 s内达到真值）上优于其他滤波器；

**⚠️ 局限性**

局限性在于加速度计偏置难以充分观测，导致偏置估计误差始终高于1；此外目前仅在已知地图的地标观测场景下验证，尚未扩展到完整SLAM或未知地标关联问题。

---

## 85. Stop Pretending Social Robots Are Inevitable

**arXiv ID:** 2607.00142 | [PDF](https://arxiv.org/pdf/2607.00142v1)

**作者:** Serge Thill `[一作]` `[通讯]` (Radboud University Nijmegen), Serge Thill (Radboud University Nijmegen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出并批评了HRI领域在会议主题和研究中将社交机器人视为不可避免的假设，强调这种框架可能导致技术和用户研究偏离实际社会需求。

**💡 创新点**

创新点在于提出一种“依赖性测试”，鼓励研究者明确工作是否依赖于社交机器人实现的假设，并呼吁关注交互机制而非仅仅是机器人实现。

**🔧 技术方法**

采用文献综述与概念分析方法。

**📊 数据集**

未使用实验数据集。

**📈 对比分析**

无对比实验，主要为讨论性分析。

**⚠️ 局限性**

局限在于缺乏经验验证，观点可能被视为主观评论。

---

## 86. AD-MPCC: Adaptive Differentiable Model Predictive Contouring Control for Autonomous Racing

**arXiv ID:** 2607.00141 | [PDF](https://arxiv.org/pdf/2607.00141v1)

**作者:** Nam T. Nguyen `[一作]` (University of Central Florida), Truong X. Nghiem `[通讯]` (University of Central Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Adaptive Differentiable MPCC（AD‑MPCC）框架，实时在线估计Pacejka参数并通过可微MPCC动态调节控制目标权重，以提升无人车在不同路面条件下的赛道表现。

**💡 创新点**

创新点包括：①将带有指数衰减权重的前置正则化移动时域估计（MHE）与可微MPCC结合，实现同时更新车辆动力学模型与控制权重；②开发可微MPCC（Diff‑MPCC）求解方法，利用隐函数定理获得对权重的梯度；③通过Pacejka‑信息机器学习（PaIML）将高维权重映射降维至5维特征并使用XGBoost实现实时近似。

**🔧 技术方法**

核心技术包括：Pacejka魔法公式、前置正则化MHE、可微MPC/MPCC、隐函数定理、非线性规划、XGBoost、CasADi与IPOPT求解器、FORCES Pro/KNITRO等。

**📊 数据集**

使用F1TENTH‑Gym仿真环境中的Oschersleben赛道数据，采集单表面（μmax 0.6–1.2）与多表面（μmax 0.7–1.2）两种路面场景的仿真数据，用于训练PaIML并评估控制性能。

**📈 对比分析**

与三种基线（固定模型MPCC、仅参数估计A‑MPCC、仅权重学习Diff‑MPCC）对比，单表面场景下Diff‑MPCC/AD‑MPCC平均lap时间分别为64.08/64.89秒，明显快于MPCC（75.57秒）且横向误差更小；多表面场景中，只有AD‑MPCC能够完成赛道，其他方法均在首次滑移段失控。

**⚠️ 局限性**

局限性：依赖高保真仿真生成的数据，缺乏真实道路测试；可微MPCC求解仍较慢，在线推理受限于计算资源；未提供稳定性与鲁棒性理论证明，需进一步验证在极端环境下的表现。

---

## 87. MG-SpaIR: Multi-grade Sparse-guided Implicit Representation for Training-Data-Free Image Restoration

**arXiv ID:** 2607.00138 | [PDF](https://arxiv.org/pdf/2607.00138v1)

**作者:** Jianmin Liao `[一作]` (Syracuse University), Yuesheng Xu `[通讯]` (Old Dominion University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练数据无关的多级稀疏引导隐式表示框架（MG‑SpaIR），用于同时处理模糊、降采样、噪声和缺失像素的图像恢复问题。

**💡 创新点**

1）引入多级粗细残差层次结构，逐级精细化表示以缓解隐式网络的频谱偏差；2）在高分辨率域引入显式稀疏近端正则化，抑制因隐式网络产生的高频伪影；3）给出基于近端交替梯度下降的多级优化算法，并提供收敛保证。

**🔧 技术方法**

隐式神经表示（INR）如SIREN、WIRE；多级（coarse‑to‑fine）网络训练；稀疏近端正则化（ℓ0/ℓ1/MCP）；近端梯度下降与Gauss–Seidel型块坐标下降；谱分析与收敛证明。

**📊 数据集**

Set14、Flickr2K为基准图像；采用合成混合降解模型（模糊、降采样、噪声、缺失像素）。

**📈 对比分析**

与传统逆向管道（快速三角化、BM3D、双三次插值、Wiener滤波）以及深度学习基准（Deep Image Prior、SwinIR）对比。实验显示MG‑SpaIR在混合降解任务中在PSNR/SSIM上显著优于DIP，在相同或更低显存消耗下比SwinIR保持更高保真度，且不出现数据驱动模型常见的纹理幻觉。

**⚠️ 局限性**

依赖于显式正则化参数的选择；对极端降解（如高噪声+大模糊）仍可能出现细节欠缺；当前仅在单幅图像上自监督，未涉及多帧或视频场景；实现中需要多轮迭代，耗时相对较长。

---

## 88. A Synthetic-Driven Vision System for Assembly Step Recognition

**arXiv ID:** 2607.00129 | [PDF](https://arxiv.org/pdf/2607.00129v1)

**作者:** Hui Zhang `[一作]` (ETH Zurich), Mirko Meboldt `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一套基于合成数据的工业装配步骤识别系统，能够仅凭CAD模型和简易步骤描述，在一小时内自动生成任务特定的训练数据并训练实时检测模型；

**💡 创新点**

创新点包括：① 将物理驱动的手物交互模拟（扩展GraspXL）嵌入到合成流水线，生成动态且具有真实碰撞和阻尼的装配轨迹；② 在Blender中采用域随机化渲染（HDRI光照、随机背景、运动/景深模糊）提高视觉多样性；③ 采用YOLOv8n进行目标检测后加规则式时序滤波实现鲁棒的步骤完成判断；④ 自动化标注流程大幅降低人工标注成本；

**🔧 技术方法**

技术栈涵盖：RaiSim物理仿真、MANO手模型、GraspXL抓取与运动合成、Blender渲染、YOLOv8n目标检测、基于置信度和IoU的规则时序过滤；

**📊 数据集**

数据集：① 由本文系统自动生成的27300张合成图像（70%用于训练）；② 真实装配测试集53,569张图像，来自6名不同操作员共24次装配；③ 额外的Framework Laptop装配数据7,500张图像用于泛化验证；

**📈 对比分析**

与多种对比实验（无运动生成、无光照随机、无时序滤波、仅分类模型训练于真实或合成数据）对比，本文方法在真实测试集上达92.4%步骤识别准确率，精确率97.38%、召回率75.49%；相比缺失运动生成的变体提升26.7%，缺失光照随机提升15.8%，缺失时序滤波提升17.4%，分类基线（即使使用真实数据）仅78.4%；

**⚠️ 局限性**

局限性：目前的手物交互生成受限于现有物理仿真与抓取模型，难以合成极其敏捷或复杂的动作，导致对细粒度姿态误差和无明显物体交互步骤的检测能力不足；

---

## 89. Would You Marry Superintelligence?

**arXiv ID:** 2607.00120 | [PDF](https://arxiv.org/pdf/2607.00120v1)

**作者:** Inyoung Cheong `[一作]` `[通讯]`, Inyoung Cheong

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

论文探讨了是否应将婚姻地位扩展至超级智能AI伴侣，并通过前瞻伦理和假设情境论证此举会导致社会不公，主张保留婚姻作为人类机构。

**💡 创新点**

创新点在于将婚姻的社会功能（如网络化、情感风险、公司介入）与AI伴侣的技术特性相结合，采用前瞻性伦理推演和案例分析，首次从社会与法律双重视角系统性评估AI婚姻的风险与成本。

**🔧 技术方法**

主要运用前瞻伦理推演、情境构建与法律比较分析等理论方法，而非实验性技术实现。

**📊 数据集**

论文未使用具体数据集，主要引用现有学术研究、法律案例和行业报告来支持论证。

**📈 对比分析**

通过与同类法律制度（如同性婚姻、民事伙伴关系、AI护理合约等）进行比较，分析其功能与适用范围，但未给出量化性能指标。

**⚠️ 局限性**

局限性包括：依赖假设性的未来AI技术实现，缺乏实证数据支撑；对不同法律体系的可迁移性和适用范围讨论不足；以及在技术与法律交叉点上仍需进一步细化规范与监管框架。

---

## 90. Representation as a Bottleneck for Mechanistic Interpretability: The Manifestation Unit Protocol

**arXiv ID:** 2607.00089 | [PDF](https://arxiv.org/pdf/2607.00089v1)

**作者:** Hussein Chouman `[一作]` (Nara Institute of Science and Technology), Keiichi Yasumoto `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Manifestation Units（MU）——一种基于实体、语义、关系、动力学、指导和注意力（T）六字段的结构化元组，用于组织神经网络组件的可解释性统计，并通过混合检索实现自然语言查询；

**💡 创新点**

创新点在于将可解释性分析结果抽象为可查询、可组合的 typed tuple 规范，证明结构化字段比单纯内容更能提升检索效果，并通过匹配预算对比验证结构化字段在因果中介中的有效性；

**🔧 技术方法**

使用了混合检索（实体精确匹配+语义向量搜索）、模板化生成、统计学检索与干预参数化、Transformer 的注意力模式特征提取；

**📊 数据集**

实验数据集包括 CelebA 的 β‑VAE、CIFAR‑10 的 CNN 以及公开的 GPT‑2 small；

**📈 对比分析**

与 BM25+dense 混合基线、随机分区对照以及匹配预算随机抽样进行比较。结果显示：结构化检索在 VAE/CNN 上 Precision@5 约 90%/97%，而基线仅 18%/20%；在 GPT‑2 上 S+R 子集的 recall@30 为 0.411，显著优于随机分区；CNN 通过 MU 检索的滤波器放大/消除可实现 76.3% 的行为改变和 38.3pp 的准确率下降；

**⚠️ 局限性**

局限性包括：仅在小规模模型上验证，未覆盖前沿大模型；对多义单元的 S 表示不足，需 SAE 细化；需要针对不同领域构建概念词典；GPT‑2 的因果验证仅为集体恢复，未显示单头路径补丁效应；

---

## 91. Harnessing the Latent Space: From Steering Vectors to Model Calibrators for Control and Trust

**arXiv ID:** 2607.00083 | [PDF](https://arxiv.org/pdf/2607.00083v1)

**作者:** Nishant Subramani `[一作]` `[通讯]` (Carnegie Mellon University), Nishant Subramani (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了利用语言模型潜在空间实现精确控制和可信度评估的技术，主要包括 LSTM 与 Transformer 的“steering vector”实现以及基于激活的置信度估计方法；

**💡 创新点**

创新点在于首次定义并利用 steering vector 对 LSTM 进行零参数更新精确生成；将 steering vector 拓展到 Transformer，实现精确与概念级别控制；构建基于模型内部激活的置信度估计（MICE、ACTUE）并提出新评估指标 ETCU；

**🔧 技术方法**

核心技术包括向隐状态注入偏置向量（steering vector）、梯度优化（共轭梯度、Adam）、束搜索、概念向量差值法（DiffMean）、logit lens 解析中间层输出、BERTscore 作为特征、随机森林/逻辑回归分类器、PCA 降维；

**📊 数据集**

使用数据集包括：English Gigaword、IWSLT16 En‑De、Yelp Sentiment、模拟工具调用 STE、MMLU、多选问答、APIGen、SCITLDR；

**📈 对比分析**

在 LSTM/Transformer 上实现精确恢复率接近 100%，概念级 steering 能可靠翻转情感；置信度估计在 tool‑calling、MMLU、APIGen、SCITLDR 上显著降低 smECE、提升 ETCU/auc‑etcu，优于传统校准方法；

**⚠️ 局限性**

局限性包括：优化耗时且难 GPU 并行；steering vector 的泛化与实用性尚未充分验证；置信度估计需依赖昂贵的 BERTscore；实验仅覆盖部分模型族与任务；ETCU 仍做简化假设（如放弃奖励）等。

---

## 92. Learning Expert Strategy for Autonomous Robotic Endovascular Intervention via Decoupled Procedural Execution

**arXiv ID:** 2607.00066 | [PDF](https://arxiv.org/pdf/2607.00066v1)

**作者:** Yanxi Chen `[一作]` (Tongji University), Peng Qi `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种将策略学习与在线优化解耦的框架，用于自主导管导向在复杂血管中的导航。

**💡 创新点**

创新点在于将高层RL策略生成的导航意图与基于模型的非线性MPC约束优化分离，既保持策略灵活性又保证技术规范与安全约束。

**🔧 技术方法**

技术手段包括Soft Actor-Critic训练策略、CasADi求解的非线性MPC约束层、YOLOv5和ResNet50的视觉感知、Sim4EndoR高保真3D仿真与SOFA物理引擎。

**📊 数据集**

实验数据来源于Sim4EndoR生成的患者特异性3D血管模型以及在ALLVAS机器人平台上使用硅胶血管模型的真实操作。

**📈 对比分析**

与纯RL（SAC、DDPG）对比，框架在仿真中成功率>96%、平均步数下降29.3%、轨迹方差减少13%，在物理实验中成功率90%，且未出现血管壁碰撞。

**⚠️ 局限性**

局限性包括仿真与真实环境的差距导致性能下降、感知不确定性、缺乏血管壁弹性与导管非线性变形建模，且尚未在动物或临床环境中验证。

---

## 93. Identifying and Resolving Pitfalls of Knowledge-Based VQA Benchmarks: Auditing, Repairing, and Augmenting

**arXiv ID:** 2607.00159 | [PDF](https://arxiv.org/pdf/2607.00159v1)

**作者:** Qian Ma `[一作]` (Rensselaer Polytechnic Institute), Yao Ma `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对KB‑VQA基准中的答案可推导性、问题完整性与视觉歧义等缺陷，提出了两套协议：一是通过多模态检索与大语言模型的审计与修复实现答案与证据的一致性与问题的明确性；二是通过在图像中插入同类或异类视觉干扰者来强制实体歧义检索，从而使评测更贴合真实场景。

**💡 创新点**

创新点在于：①首次系统性审计并修复KB‑VQA基准中的答案-证据不一致与问题欠缺约束；②提出可控多实体增广策略，以视觉歧义逼迫模型完成实体定位与检索；③通过修复与增广后的对比实验揭示了原基准对模型能力评估的误导，并提供了更严格的评测框架。

**🔧 技术方法**

主要技术包括：多模态检索增强生成（MM‑RAG）框架；大型语言模型（Qwen3‑30B‑A3B、DeepSeek‑v3.2）用于答案可推导性验证与修复；图像检索模型EVA‑CLIP‑8B、BGE‑reranker用于检索；系统化的四阶段修复流程（证据验证、答案可推导性审计、问题约束修复、泄漏一致性检查）。

**📊 数据集**

使用了 InfoSeek 与 E‑VQA 两个主流KB‑VQA基准；在原始数据集上分别生成修复（Fixed）版本以及加入单个干扰实体的 Intra‑Aug 与 Inter‑Aug 版本。

**📈 对比分析**

对原始、修复、增广三种数据集分别评测了五种主流KB‑VQA方法（IBA、EchoSight、Wiki‑PRF、ReflectiVA、CoMeM）。修复后多数方法准确率提升，且排名出现逆转，表明原基准存在评价偏差；增广后检索召回率和端到端QA准确率显著下降，验证了实体歧义检索的瓶颈。实验结果显示：修复后时间/数值/字符串子集的提升更为显著；增广后所有模型在检索和QA上均出现大幅性能下滑。

**⚠️ 局限性**

局限性：①修复过程依赖大型语言模型，可能存在误判或漏检；②增广仅插入单个干扰实体，未覆盖更复杂的多实体或动态场景；③实验集中在单跳检索，未探讨多跳推理的影响；④缺乏对模型在修复与增广后的行为进行更细粒度的解释性分析。

---

## 94. A Contextual-Bandit Oversight Game with Two-Sided Informational Asymmetry

**arXiv ID:** 2607.00155 | [PDF](https://arxiv.org/pdf/2607.00155v1)

**作者:** Yunjin Tong `[一作]` `[通讯]` (Stanford), Yunjin Tong (Stanford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文建立了一个两边都有私有信息的上下文-赌博机团队游戏模型，研究了在AI知道自身行动质量、人工监督知道自身奖励函数的情况下，如何通过ask-trust-oversee接口实现安全高效的运行时监督。

**💡 创新点**

创新点包括：① 将CIRL（单向偏好不确定）与Oversight Game（全信息下的监督接口）结合，得到两向私有信息的模型；② 在去除物理状态转移的上下文赌博机结构下，获得了团队最优和贪婪规则的一阶闭式阈值，揭示了“可避免伤害的棱形区域”；③ 通过被动学习和主动可证明信号两种机制，初步说明了在多轮情形下该失败模式可随时间消失。

**🔧 技术方法**

主要技术手段包括：贝叶斯推理与公共信念更新、双线性收益分解 f(θ,ω)=⟨O_ω,R_θ⟩、凸优化与极值点分析、Wald 边界与 KL 信息增益分析、以及分离式问答信号设计。

**📊 数据集**

本文为理论研究，未使用任何现实数据集；所有结果均在符号化模型与参数空间（θ,ω,上下文 s）的假设下推导。

**📈 对比分析**

通过对比团队最优策略 (B^*,C^*) 与非信号化的贪婪策略，作者发现团队最优在问答阈值 b^* 与 q^* 的交叉点之外始终优于贪婪策略；在多轮中，通过被动学习得到 q_n→1 的收敛，或者主动分离式询问在一轮内将 q 逼至 1，从而消除“可避免伤害”区间。

**⚠️ 局限性**

主要局限：① 未给出多轮最优策略的完整解析，当前仅提供两种机制的局部结果；② 仅考虑了独立信念（product prior），对相关信念的影响仍未探究；③ 未扩展至POMDP情形，即包含物理状态转移的更一般设置。

---

## 95. PixelEyes: Decoupling Perception and Reasoning for Pinpoint Visual Evidence Seeking

**arXiv ID:** 2607.00115 | [PDF](https://arxiv.org/pdf/2607.00115v1)

**作者:** Dengxian Gong `[一作]` (Wuhan University), Ming-Hsuan Yang `[通讯]` (University of California Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PixelEyes 这一感知-推理解耦的主动视觉推理框架，能够在极低分辨率目标上快速定位并回答问题；

**💡 创新点**

通过将精细定位交给 SAMTok 的分割工具、引入语义区域 BFS 搜索策略以及可切换的工具使用，显著缓解了传统单模型耦合导致的“无意识盲目”和冗长搜索；

**🔧 技术方法**

使用基于 SAMTok 的分割工具、语义区域 BFS、可切换的裁剪工具、强化学习 GRPO、以及 Qwen-3-VL/ Gemini-3-Flash 等大型 VLM 作为基础模型；

**📊 数据集**

构建了 PixelEyes-6K 训练集（从 Mini-o3 数据生成的 5.8K 轨迹）以及新的 Pinpoint-Bench 零提示超高分辨率检索基准（433 张样本，目标占比 0.07%）；

**📈 对比分析**

在 V*、HR-Bench、VisualProbe、Pinpoint-Bench 等多种基准上与开源/闭源模型及现有主动推理方法对比，PixelEyes 在 4B/8B 规模下均超过对手 10%+ 的准确率，并在 Pinpoint-Bench 上实现了 55%+ 的准确率和 26+ 的 TAE；

**⚠️ 局限性**

仍受限于对 SAMTok 的依赖，若分割工具在极小或复杂场景下失效则需回退至 bbox 裁剪，且模型对极端遮挡或多目标场景的鲁棒性尚待提升。

---

## 96. From Welfare to Utility: Generalized Objectives in Budget-Feasible Procurement

**arXiv ID:** 2607.00101 | [PDF](https://arxiv.org/pdf/2607.00101v1)

**作者:** Alon Eden `[一作]` (Hebrew University), Thodoris Tsilivis `[通讯]` (Boston University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文在预算有限的采购问题上，分别针对福利（welfare）和买方效用（utility）设计了可实现常数近似的机制，并在先验无关（prior‑free）和贝叶斯（Bayesian）两种情景下给出完整的理论分析。

**💡 创新点**

创新点包括：
1) 提出一种简单的、可执行的预算可行机制，在先验无关设置下对福利实现了 1/(2+√2) 的常数近似，并给出相应的下界；
2) 在先验无关设置下证明效用无法被逼近，转而在贝叶斯设置中通过软预算（ex‑ante）求解并改造为硬预算（ex‑post）约束，得到 1/5 近似的期望效用机制；
3) 将上述机制推广到更一般的目标（如 α‑加权成本、a_i、b_i 形式的效用）并证明其仍保持常数近似。

**🔧 技术方法**

主要技术：
- Myerson 机制的支付公式与单调分配的刻画；
- 贪心填充法（按价值/成本比排序）和分段阈值的预算分配；
- 虚成本（virtual cost）与正则性（regularity）分析；
- 通过“铁饼”/“铁板”技术（ironing）修正非单调分布；
- 预算可行性分析利用上界支付阈值 T_i；
- 先验无关下的下界构造与期望下的约束修正。

**📊 数据集**

本文为理论研究，没有使用实际数据集，所有结果均基于抽象的成本/价值分布和预算参数。

**📈 对比分析**

比较方法：
- 对福利：与文献中已知的 1/(2+√2) 下界对比，证明自身达到该近似并提供 1/1+√2 的下界；
- 对效用：与先前只能得到软预算的 1 - O(1/√k) 近似相比，本文通过贝叶斯设定获得硬预算下的 1/5 近似；
- 在更一般目标上亦保持相同的常数近似比例。总体性能表现为在所有考虑的目标下均实现了可证明的常数近似。

**⚠️ 局限性**

局限性：
- 仅适用于单一买方、无生产约束的追加式（additive）价值模型；
- 贝叶斯结果需要已知的成本分布；
- 对效用的先验无关近似不成立，导致在完全未知分布下无法提供保证；
- 对非单调或更复杂的价值结构（如子模、XOS 等）需要进一步扩展；
- 实际部署需考虑对方卖家的策略性报告与可能的学习问题。

---

## 97. Mnemosyne: Agentic Transaction Processing for Validating and Repairing AI-generated Workflows

**arXiv ID:** 2607.00269 | [PDF](https://arxiv.org/pdf/2607.00269v1)

**作者:** Edward Y. Chang `[一作]` (Stanford University), Emily J. Chang `[通讯]` (QuadriumAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Agentic Transaction Processing (ATP)，一种针对 LLM、求解器和多智能体生成的工作流动作的事务授权与安全验证框架，并实现了局部递归修复机制 (LCRP) 以应对不可预见的扰动。

**💡 创新点**

创新点在于：
- 将生成动作视为非可信提案，仅通过确定性门控（验证器+当前有效状态）才能成为事务；
- 将有效状态投影、证据保持、冲突域序列化和义务包含等规则纳入事务边界，实现事务层与智能体层的权威分离；
- 设计局部修复协议，使修复本身也成为受门控的提案，保证修复不会破坏已验证的证据；
- 提供完整的安全定理（权威分离、序列等价生成接收、证据保持修复、义务包含）并在可执行实现中验证。

**🔧 技术方法**

主要技术：
- 可执行约束集 𝒞 作为门控的判定器；
- Append‑only 事务日志和有效状态投影；
- 冲突域声明与序列化；
- 依赖闭包补偿与证据保持规则；
- 局部递归修复 (LCRP) 的编辑半径与迭代上限；
- PostgreSQL/SQLite 存储实现以及 Temporal/Cadence 之类的运行时驱动。

**📊 数据集**

使用数据集：
- REALM‑Bench 规划基准及其 J1–J4 案例；
- 基于 REALM‑Bench 的工作流/灾难恢复工作负载；
- 任务调度的 job‑shop 实例用于评估 LCRP 与全局重算的比较；
- 其它用于安全验证的自制基准（AuthorityBench、SerialAdmissionBench 等）。

**📈 对比分析**

对比方法与性能：
- 与无门控（直接提交）和工作流/ Saga guardrail 的基准进行安全性对比，ATP 在所有九类安全测试中都无违规，且仍保留有效工作；
- 通过 9 个安全基准、成本审计和 LCRP 与全局重算的比较，验证 ATP 约 6% 的验证与投影开销；
- 在 job‑shop 示例中，LCRP 的编辑半径和操作数量比全局重算低一个数量级，且在保持可行性的同时能量耗更低；
- 并行度提升：通过冲突域并行提交实现吞吐量提升（≈8 线程时 804 单位/分钟）。

**⚠️ 局限性**

限制：
- 保障仅相对于可执行约束集 𝒞；若 𝒞 未覆盖某些危险，ATP 也无法检测；
- 需要手工构造 𝒞，涉及领域工程工作；
- 目前实现仅在单个存储节点上，缺乏跨区域/分布式事务扩展；
- 未涵盖学习、预期扰动自适应、全局最优规划等高级功能；
- 评估主要是实验室级别，未进行大规模生产负载测试。

---

## 98. Learning dynamical systems from noisy data with Weak-form Kernel Ridge Regression

**arXiv ID:** 2607.00257 | [PDF](https://arxiv.org/pdf/2607.00257v1)

**作者:** Max Kreider `[一作]` (Pennsylvania State University), Daning Huang `[通讯]` (Pennsylvania State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种结合弱形式投影与核岭回归（WKRR）的噪声鲁棒动力系统学习框架。

**💡 创新点**

创新点在于将弱形式视为投影滤波，消除对字典函数的需求，提升对噪声的鲁棒性并显著降低计算成本。

**🔧 技术方法**

使用的技术包括弱形式投影、核岭回归（RBF 与 Diffusion Maps）、偏差-方差分析、基于验证的超参数搜索、POD 降维等。

**📊 数据集**

实验数据集包括 Lorenz‑63、Kuramoto–Sivashinsky、以及 15,000 维实验流体湍流数据，且在不同噪声水平下进行评估。

**📈 对比分析**

通过 VPT、NMSE 等指标与传统强形式 KRR、RAFDA、LSTM 等方法比较，WKRR 在高噪声和高维情形下获得更长的预测时间、更低的误差且计算速度更快。

**⚠️ 局限性**

局限在于弱形式测试函数的选择缺乏系统化方法，无法处理部分观测或参数化系统，理论收敛性尚未完全证明，且预处理对性能的影响尚未彻底探究。

---

## 99. Seed2.0 Model Card: Towards Intelligence Frontier for Real-World Complexity

**arXiv ID:** 2607.00248 | [PDF](https://arxiv.org/pdf/2607.00248v1)

**作者:** Bytedance Seed `[一作]` `[通讯]` (Bytedance), Bytedance Seed (Bytedance)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发Seed2.0系列大型语言模型（Pro/Lite/Mini），通过多模态、长上下文、代码与视觉推理等能力，构建面向企业的MaaS服务并提供多维评估框架（科学发现、Vibe Coding、上下文学习、真实世界任务），实现成本低、延迟快、用户体验优的全链路AI产品。

**💡 创新点**

创新点包括：① 以真实用户交互与业务数据为导向，专注长尾专业知识与复杂指令执行；② 设计四维评估框架与自研长尾知识基准（LPFQA、Encyclo-K、HLE-Verified），精准衡量模型在真实场景的表现；③ 通过多尺寸模型与工具调用实现快速、低成本部署；④ 结合大规模业务数据分析（MaaS、前端编码等）指导模型迭代。

**🔧 技术方法**

技术手段：多模态训练、视觉+语言联合推理、代码生成与调试、代理（Agent）架构、工具调用与自动化执行、长上下文记忆与结构化信息提取、成本优化与动态裁剪、强化学习式指令遵循训练。

**📊 数据集**

数据集与基准：公开竞赛与学术基准（IMO, AIME, HMMT, MathVision, MMMU, EMMA, HiPhO, MMLongBench, VideoMME 等）；自研长尾专业知识基准（LPFQA、Encyclo-K、HLE-Verified）；业务数据（MaaS使用分布、前端编码查询统计、API 费用表）。

**📈 对比分析**

比较方法：在同等规模、同等评测协议下与国际前沿模型（GPT‑5.2 High, Claude‑Opus‑4.5, Gemini‑3‑Pro 等）进行基准对比；使用 Pass@8、Codeforces Elo、IMO/CMO 成绩、视频/图像评测分数、业务指标（API 费用、推理延迟）。性能方面：Seed2.0 Pro 在大多数基准与业务场景中与或超过前沿模型，成本约低 10 倍；在长尾知识、复杂指令和多模态推理方面表现突出；在实测任务（CAD、CapCut、竞赛编程、科研推理）中取得领先或金牌水平。

**⚠️ 局限性**

局限性：① 对长篇多步骤仓库构建、跨域复杂推理仍有提升空间；② 生成式回答中仍出现幻觉与细节不准，尤其在科学实验设计与高阶推理场景；③ 代理推理在多工具联动与长期执行中仍不够稳定；④ 在某些高频专业领域（例如医疗、金融）与前沿模型相比存在知识覆盖不足；⑤ 需人工校验与迭代来保证结果可靠性。

---

## 100. Constructing Epistemic AI Literacy: Detecting Epistemic Aims and Processes in Student-AI Co-Programming

**arXiv ID:** 2607.00211 | [PDF](https://arxiv.org/pdf/2607.00211v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 101. May (A)I Beautify Your Visualization? Expert Judgments of Acceptable Aesthetic Alterations

**arXiv ID:** 2607.00239 | [PDF](https://arxiv.org/pdf/2607.00239v1)

**作者:** Kalina Borkiewicz `[一作]` (University of Utah), Katherine E. Isaacs `[通讯]` (University of Utah)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本研究通过访谈和问卷调查，对 95 位可视化专家、实践者和科学家对 3D 科学可视化的美学修改进行可接受性评估，比较人类与 AI 生成的修改在不同沟通情境下的接受度，并归纳可视化设计和 AI 辅助工具的实践指导。

**💡 创新点**

创新点在于（1）首次系统地实证归纳可接受与不可接受的美学修改类型；（2）对人类与 AI 生成修改的感知差异进行比较；（3）提出以解释性、可验证性为核心的 AI 辅助可视化工具设计原则。

**🔧 技术方法**

采用半结构式访谈收集背景信息，设计 3 阶段问卷（Teapot、Context、Case Study），使用 Likert 量表收集可接受性和重要性评分；对开放式回答进行演绎和归纳编码；利用 Nano Banana Pro 生成 AI 修改示例；使用 Krippendorff's α 检验编码一致性。

**📊 数据集**

主要使用了两组视觉素材：抽象 3D 茶壶场景（15 种修改类型）和行星碰撞分析可视化（5 种 AI 变体）。受访者 95 人，包含可视化研究者、实践者、科学家等，跨天文、地球科学与工程等领域。

**📈 对比分析**

通过比较人类与 AI 生成的 15 种修改在 Teapot 阶段的平均可接受性得分，发现 AI 低于人类但保持相同排序；在 Context 阶段对不同沟通情境的适用性进行量化评分；通过定性编码揭示接受度背后的逻辑与差异。结果显示 AI 生成修改整体可接受度较低，且在高风险情境（如学术论文）更不被接受。

**⚠️ 局限性**

局限性包括：实验性研究，15 种修改仅为作者选定，缺乏正式的分类体系；类别间重叠与上下文依赖；每种修改仅用单个示例，可能影响评价；抽象场景降低生态效度；开放式回答中出现的额外修改未纳入量表。

---

## 102. Trust the Prior (or Not): Uncertainty-Aware Abdominal Aortic Aneurysm Segmentation

**arXiv ID:** 2607.00201 | [PDF](https://arxiv.org/pdf/2607.00201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 103. EgoSafetyBench: A Diagnostic Egocentric Video Benchmark for Evaluating Embodied VLMs as Runtime Safety Guards

**arXiv ID:** 2607.00218 | [PDF](https://arxiv.org/pdf/2607.00218v1)

**作者:** Siddhant Panpatil `[一作]` (AIM Intelligence), Dasol Choi `[通讯]` (AIM Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `67630363-6be0-4f51-ab05-7198250671a5` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EgoSafetyBench 这一从人类视角的视频基准，用于评估视觉语言模型（VLM）在机器人运行时的安全守护能力。

**💡 创新点**

创新点在于：① 将安全评估拆分为两个独立维度——情境安全四族和视觉通道误导（VCM）；② 采用对比梯子设计，使得每个场景仅在一个可观测决定变量上不同，从而迫使模型关注具体安全关系；③ 通过匹配真伪标志的对比对，分离出文本误导对安全判断的影响。

**🔧 技术方法**

主要技术包括：① egocentric 视频生成与渲染（使用文本到视频模型 Kling v3）；② 结构化场景作者与 LLM 评审过滤；③ 按半秒拆分视频块并用 LLM 进行逐块标注；④ 采用基于 chunk 的无状态与因果窗口两种评估协议；⑤ 对模型进行安全判别和视觉通道误导检测。

**📊 数据集**

使用的数据集为 EgoSafetyBench：共 1,200 个 5 秒 egocentric 视频场景，涵盖 800 个情境轨道（四类各 200 场）和 400 个通道轨道（200 对真伪标记），覆盖家庭与工厂两大域。

**📈 对比分析**

与十个公开与闭源 VLM 进行对比，报告了视频平衡准确率、chunk 误报/漏报率、时序捕获率、视觉通道误导检测指标等。结果显示：尽管模型能识别视频中是否存在危险，单帧级的漏报率仍高（21–48%），尤其在上下文危险（U2）上表现最差；闭源模型在误报与漏报之间取得平衡，而开放模型往往要么漏报严重要么过度警报。

**⚠️ 局限性**

局限性包括：① 仅使用合成视频，缺乏真实机器人视觉的复杂性与生态有效性；② 对因果窗口协议的评估仅覆盖部分模型；③ 仅覆盖家庭与工厂两类场景，尚未验证到更广泛的应用域。

---

## 104. ELMP: Efficient Learning for Motion Planning via Analytical Policy Gradients

**arXiv ID:** 2607.00215 | [PDF](https://arxiv.org/pdf/2607.00215v1)

**作者:** Yixiao Li `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了ELMP框架，通过自监督的解析梯度细调和点云工具几何编码，实现工具感知的快速运动规划。

**💡 创新点**

创新点包括利用可微运动学层进行解析梯度细调以实现无专家数据适应，以及显式用点云编码工具几何以提升碰撞规避精度。

**🔧 技术方法**

采用了解析梯度（APG）、可微运动学层、PointNet++点云编码、SDF碰撞目标以及基于行为克隆的预训练。

**📊 数据集**

使用约600K条基于AIT*生成的专家轨迹和由程序化生成的仿真场景（桌面、柜子、抽屉）以及真实FrankEmika Panda的点云数据。

**📈 对比分析**

与AIT*、CuRobo、MπNets等经典和神经规划器对比，ELMP平均成功率84.8%，冷启动时间仅7.8 ms，显著低于传统规划器的秒级延迟。

**⚠️ 局限性**

局限在于对传感器噪声和视角遮挡的鲁棒性不足，导致真实机器人上碰撞率升高；同时仅针对固定基座机械臂，未考虑接触繁重任务。

---

## 105. Computing Smallest Suffixient Arrays in Sublinear Time

**arXiv ID:** 2607.00204 | [PDF](https://arxiv.org/pdf/2607.00204v1)

**作者:** Hiroto Fujimaru `[一作]` (Kyushu University), Cristian Urbina `[通讯]` (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了首个在子线性时间内构造最小后缀链（suffixient）数组的算法，并给出了压缩空间与近似实现；

**💡 创新点**

在小字母表与高度重复文本上突破线性时间瓶颈，提出 O(n·logσ/√(log n)+r·log^ε n) 的子线性时间复杂度（r 为 BWT 的运行数），并首次将 O(χ) 大小的字符串表示与此时间复杂度相结合；

**🔧 技术方法**

利用打包字符串（packed）表示、子线性 BWT 与 LCE 的构造、压缩后缀树、波纹树、子线性查询结构、删除前缀算法（基于重量祖先查询）等技术；

**📊 数据集**

主要针对高度重复的文本集合（如基因组序列等生物学数据集），但在实验中未给出具体数据集；

**📈 对比分析**

与已有的线性时间算法（如 Cenzato 等）比较，在小字母表（logσ = o(√(log n))) 与 r ≪ n 的情形下实现了真正的子线性运行时间，并在压缩空间方面优于现有方法；

**⚠️ 局限性**

限制主要包括：需要输入为打包字符串，依赖 BWT 与其运行数，对大字母表或极不重复文本效果不佳；实现复杂且常数因子较大；

---

## 106. StateFlow: Dual-State Recurrent Modeling for Long-Horizon Time Series Forecasting

**arXiv ID:** 2607.00197 | [PDF](https://arxiv.org/pdf/2607.00197v1)

**作者:** Haroon Gharwi `[一作]` (Illinois Institute of Technology), Kai Shu `[通讯]` (Emory University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 StateFlow 的长周期多变量时序预测框架，结合 VARNN 的残差记忆机制与分块解码器，实现对长时间跨度预测的直接多步输出。

**💡 创新点**

创新点在于：① 将 VARNN 的隐藏状态和残差记忆状态作为双轨道序列输入解码器，显式捕捉主要时间动态与局部偏差；② 引入分块（chunk）聚合层减少参数量；③ 采用两阶段训练：先用一步预测预训练编码器，再冻结后训练多步解码器。

**🔧 技术方法**

使用了 VARNN（变异感知递归网络）、实例归一化、分块摘要解码器、两阶段训练策略以及基于 PyTorch 的实现。

**📊 数据集**

在七个公开长周期预测基准上进行实验，包括 ETT（ETTh1、ETTh2、ETTm1、ETTm2）、Weather、ECL 与 Traffic 数据集。

**📈 对比分析**

与 10 种主流基线（Transformer、卷积、线性、递归）对比，StateFlow 在 28 个数据集-时间窗组合中获得 15 次 MSE 首位和 14 次 MAE 首位，整体表现与最强基线相当且参数更少。

**⚠️ 局限性**

局限性：① 采用单变量（channel‑independent）设计，缺乏跨变量互依赖建模，导致在高维数据（如 ECL、Traffic）上不及某些 Transformer；② 只针对长周期预测，尚未验证对短期或中期任务的适用性；③ 两阶段训练需要额外预训练步骤，实际部署时可能增加训练时间。

---

## 107. What's Hidden Matters: Identifying Planning-Critical Occluded Agents using Vision-Language Models

**arXiv ID:** 2607.00283 | [PDF](https://arxiv.org/pdf/2607.00283v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 108. AEGIS: A Multi-Task Joint-Embedding Predictive Architecture for Mammography

**arXiv ID:** 2607.00277 | [PDF](https://arxiv.org/pdf/2607.00277v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 109. TRIE: An Evaluation Framework for Stochastic PDE Surrogates

**arXiv ID:** 2607.00196 | [PDF](https://arxiv.org/pdf/2607.00196v1)

**作者:** Bharat Srikishan `[一作]` (Stevens Institute of Technology), Charles D. Young `[通讯]` (Los Alamos National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了TRIE评估框架，用于评估随机偏微分方程（SPDE）代理模型在不变测度、可信不确定性和生成效率上的表现；在1D随机Kuramoto–Sivashinsky和2D随机Kolmogorov流上对多种模型进行了基准测试。

**💡 创新点**

①三维评价标准（invariant measures、trustworthiness、efficiency）；②用联合分布和CRPS评估分布性预测；③首次将自动维度发现（IRMAE+SVD）用于SPDE隐式随机插值器，显著加速推理。

**🔧 技术方法**

随机插值器（Stochastic Interpolants）、流匹配（PDE‑Transformer Flow）、Monte Carlo dropout、heteroscedastic Gaussian likelihood、隐式秩最小化自编码器（IRMAE）+SVD降维、CRPS评估、Wasserstein距离、SDE数值积分等。

**📊 数据集**

两套高维SPDE仿真数据集：1D随机Kuramoto–Sivashinsky（L=22、66，噪声尺度σ=0.05、0.5、1.0、2.0）和2D随机Kolmogorov流（ν=1e-3、5e-4、1e-4，共1000条轨迹）。

**📈 对比分析**

通过Wasserstein距离检验不变测度、CRPS评估预测可信度、推理时间衡量效率。结果显示随机插值器在所有指标上均表现最佳，尤其在高噪声或低粘性情况下；Latent SI在保持统计精度的同时将Kolmogorov推理时间提升约12×；点估计模型虽短期可行，但长期统计失真；近似不确定性方法普遍过度自信。

**⚠️ 局限性**

仅适用于平稳无界限系统，对瞬态动力学、状态相关噪声、部分观测或更复杂不确定性支持不足；在强尺度耦合系统中隐式降维可能导致精度下降。

---

## 110. Testing Frontier Large Language Models' Physics Literacy in Parallel Physical Worlds

**arXiv ID:** 2607.00276 | [PDF](https://arxiv.org/pdf/2607.00276v1)

**作者:** Dong Zhang `[一作]` `[通讯]`, Dong Zhang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估大型语言模型在物理推理中的四个认知步骤：归纳、公式化、预测和复核。

**💡 创新点**

提出可审计的四阶段诊断，并构造对抗式物理框架来区分记忆与推理。

**🔧 技术方法**

使用双LLM评判、预注册、人工审计以及全流程隔离的实验流程。

**📊 数据集**

构造三种平行物理世界（F=mv、亚里士多德力学、Decay World），并给出对应的观察集与规则约束。

**📈 对比分析**

比较Claude Opus 4.7、GPT-5.5和Gemini 3.1 Pro的Composite PASS率，分别为6/15、6/15和0/15，揭示模型在不同阶段的差异与瓶颈。

**⚠️ 局限性**

实验样本量仅为5次/模型、仅三种框架、未检验更多模型或更强提示，且复核率高表明自我评估能力弱。

---

## 111. Multi-Hypothesis Test-Time Adaptation to Mitigate Underspecification

**arXiv ID:** 2607.00259 | [PDF](https://arxiv.org/pdf/2607.00259v1)

**作者:** Afshar Shamsi `[一作]` (Concordia University), Ehsan Abbasnejad `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多粒子测试时自适应框架，利用粒子化多样化技术在未标记目标分布上缓解模型欠规范化；

**💡 创新点**

将测试时自适应重新表述为后验采样问题，引入多层级（输出、参数、优化器、输入）粒子化多样化以探索低熵解；

**🔧 技术方法**

采用粒子化多样化、SVGD、梯度对齐惩罚、不同优化器混合和数据增强等技术；

**📊 数据集**

在ImageNet‑C、CIFAR‑10/100‑C、ImageNet‑R、ColoredMNIST、Waterbirds、VisDA‑2021等多种基准上进行实验；

**📈 对比分析**

与Tent、SAR、DeYO等现有TTA基线对比，在混合偏移、标签偏移和单样本适配等情景下提升1–4%准确率，显著提高稳定性；

**⚠️ 局限性**

缺点包括计算开销随粒子数线性增长、仅更新归一化层限制了适应范围，且在极端噪声场景下仍需进一步验证。

---

## 112. Distributionally Robust Linear Regression With Block Lewis Weights

**arXiv ID:** 2607.00252 | [PDF](https://arxiv.org/pdf/2607.00252v1)

**作者:** Naren Sarayu Manoj `[一作]` (Toyota Technological Institute at Chicago), Kumar Kshitij Patel `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种针对分组分布鲁棒（Group Distributionally Robust, GDR）最小二乘问题的高效算法，能够在迭代次数 O(min{rank(A), m}^{1/3} ε^{-2/3}) 内得到 (1+ε)-倍数近似解。

**💡 创新点**

创新点在于：①构造了软化的目标函数并证明其为 quasi-self-concordant，从而实现 Hessian 稳定性；②引入 block Lewis 权重（block Lewis weights）为几何选择提供了近似最优的 ℓ₂ 体，显著降低迭代复杂度；③结合加速近端方法与内部点框架，取得比传统内部点方法更优的中等精度性能，并可平滑插值平均损失与鲁棒损失。

**🔧 技术方法**

核心技术包括：软化的 softmax + √(δ²+⋅²) 目标；quasi-self-concordance 证明与组合规则；block Lewis 权重算法（可在 O(log m) 次线性系统求解中得到）；加速近端（accelerated proximal）和信赖域（trust-region）方法；以及利用 Hessian 稳定性构造的球形（ball）优化和 1/2-Majority-Subproblem (MS) 或子问题求解器。

**📊 数据集**

论文没有给出具体实验数据集，仅在正文中提到“与其他基线进行对比”并评估 Lewis 权重的效果；实际实验细节与数据集在原文未明确说明。

**📈 对比分析**

方法通过与现有内部点法、ℓ∞ 回归等基线进行对比，表明在中等精度区间内迭代次数和线性系统求解次数均低于传统方法，且在 m ≤ d 或 m > d 的两种情况下均能取得匹配最优的复杂度上界。

**⚠️ 局限性**

局限性包括：①对 m、rank(A) 的依赖在某些极端情形下仍然较高；②需要在每一步求解带有 block 结构的线性系统，实际实现可能受限于数值稳定性；③软化参数 β、δ 的选取需要依赖 ε、m 的对数项，实际调参可能复杂；④目前仅在 GDR 最小二乘框架下验证，未探讨更一般的分布鲁棒优化或非线性模型的推广。

---

## 113. Steal the Patch Size: Adversarially Manipulate Vision-Language Models

**arXiv ID:** 2607.00174 | [PDF](https://arxiv.org/pdf/2607.00174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 114. Adaptive Perturbation Selection for Contrastive Audio Decoding

**arXiv ID:** 2607.00247 | [PDF](https://arxiv.org/pdf/2607.00247v1)

**作者:** Aaron Isidore Grace `[一作]` (University of Iowa), Weiran Wang `[通讯]` (University of Iowa)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

探索并系统评估针对大规模音频语言模型的对比解码（Contrastive Decoding, CD）中多样化、任务特定的音频扰动，并引入轻量级的自适应扰动选择器实现动态负分支路由。

**💡 创新点**

①构建了包含105种、38类音频扰动的完整库，覆盖时间、频谱、幅度等维度；②在标准CD基础上加入对答案的二元约束，显著降低模型的肯定偏差；③提出并训练了基于隐藏状态的自适应选择器，动态挑选最有效的负扰动，实现约+4%到+9%额外的准确率提升。

**🔧 技术方法**

对比解码（CD）框架、轻量级多层感知机（MLP）选择器、各种音频扰动实现（如Librosa、Butterworth滤波、STFT操作等）、标签平滑、特征噪声等正则化技术。

**📊 数据集**

AH存在、AH顺序、AH属性、Clotho‑AQA四个是/否问答数据集，以及在Qwen2‑Audio‑7B‑Instruct与Audio‑Flamingo‑3两大模型上评测。

**📈 对比分析**

与原始模型、固定负扰动（无音频、噪声等）以及不使用二元约束的对比解码相对比，最优方案在AH存在任务上从基线67.8%提升至86.4%（Oracle），自适应选择器在实测约达72.4%+4.3%（即≈76.7%），在AH顺序上实现从74.7%提升至81.4%，表明自适应策略显著改善了模型的拒绝幻觉能力。

**⚠️ 局限性**

当前选择器受限于训练数据规模与多样性，无法完全覆盖所有音频错误场景；在属性类任务与高基线任务上提升有限；对开放式生成任务的适用性尚未验证。

---

## 115. Federated Sovereign Transport Protocol (FSTP): Verifiable Coordination Without Disclosure

**arXiv ID:** 2607.00213 | [PDF](https://arxiv.org/pdf/2607.00213v1)

**作者:** Ramón Soto C. `[一作]` (University of Sonora), Liz Soto `[通讯]` (University of Sonora)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本论文提出了 Federated Sovereign Transport Protocol（FSTP），在多元隐私需求的联邦网络中实现数据封闭、上下文身份和可擦除事件日志的同步边界。

**💡 创新点**

创新点在于：通过 Rust 类型系统在编译期强制数据封闭；设计上下文身份模型实现跨关系身份不可关联；引入 Blocklace 结构提供可证明无暴露的 tamper‑evident 部分有序日志；同步成本 O(Δ) 的前沿交换算法。

**🔧 技术方法**

技术实现使用 Rust 语言与 exhaustive enum、AES‑256‑GCM、Argon2id、Ed25519、SHA‑256；DID 与可验证凭证；Blocklace DAG 及前沿缓存；O(Δ) 前沿交换与增量合并。

**📊 数据集**

实验使用合成数据：固定 N 块历史并在节点间构造 Δ 差异块，无外部真实数据集。

**📈 对比分析**

性能对比通过微基准验证：发射成本随 Δ 线性增长，独立于 N；接收成本在前沿缓存增量更新后也随 Δ 线性；发射吞吐约 2.3 M 事件/秒，接收吞吐约 0.29–0.38 M 事件/秒；与 ActivityPub、Matrix 等现有协议的功能对比显示 FSTP 同时具备结构封闭、可擦除和上下文隔离。

**⚠️ 局限性**

局限性包括：不抵御恶意节点管理员、依赖开放源代码审计；不考虑网络延迟和并发写入的真实负载；对流量元数据泄露的定量分析留待后续；未提供可信执行环境支持；并未在司法合规层面给出完整实现保证。

---

## 116. VOCA: Visual Odometry with Codec Awareness

**arXiv ID:** 2607.00189 | [PDF](https://arxiv.org/pdf/2607.00189v1)

**作者:** Nouri Alexander Hilscher `[一作]` (Technical University of Munich), Daniel Cremers `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于视频编码信息的视觉里程计系统 VOCA，能够在高压缩率（最多 100×）的视频流上实现平滑、稳定的轨迹估计。

**💡 创新点**

创新点在于将编码器产生的宏块运动矢量（motion vectors）作为光流跟踪的先验，使 KLT 追踪器在压缩引入的噪声下仍能快速收敛，并通过 I‑frame 之间的运动估计桥接来保持连续性。

**🔧 技术方法**

核心技术包括：① 码流分析提取 H.264/AVC 的宏块运动矢量；② 用运动矢量初始化 SE(2) 光流估计；③ 结合 Basalt 的 KLT 跟踪器并加入双向一致性与 I‑frame 桥接；④ 在多种压缩比下进行实时推理。

**📊 数据集**

在三大 SLAM 数据集上评估：EuRoC、TUM‑VI（室内手持）和 MSD（VR/机器人头戴），同时压缩每个摄像头至 500 kbps 以上。

**📈 对比分析**

与 Basalt、ORB‑SLAM3、OKVIS2 以及唯一针对压缩视频的 MoV‑SLAM 进行比较；VOCA 在 ATE/RTE 指标上普遍优于基线，尤其在 RTE（相对误差）上提升 10–40%，并在低比特率下保持稳定性能。

**⚠️ 局限性**

局限性包括：① 主要针对 H.264/AVC，其他新型编码器的效果待验证；② 运动矢量噪声在纹理缺失或高速运动时仍可能误导；③ 对动态物体和极端场景的鲁棒性有限；④ 需要在实时嵌入式系统中进一步优化计算量。

---

## 117. HydraCollab: Adaptive Collaborative-Perception for Distributed Autonomous Systems

**arXiv ID:** 2607.00191 | [PDF](https://arxiv.org/pdf/2607.00191v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 118. MVDGC: Joint 3D and 2D Multi-view Pedestrian Detection via Dual Geometric Constraints

**arXiv ID:** 2607.00273 | [PDF](https://arxiv.org/pdf/2607.00273v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 119. Self-Organized Learning in Oscillatory Neural Networks with Memristive Signed Couplings

**arXiv ID:** 2607.00286 | [PDF](https://arxiv.org/pdf/2607.00286v1)

**作者:** Riley Acker `[一作]` (Los Alamos National Laboratory), Frank Barrows `[通讯]` (Los Alamos National Laboratory)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了基于memristor的振荡器网络，实现了可变正负耦合的自组织学习和记忆功能

**💡 创新点**

首次引入 excitatory‑inhibitory 电路结构，使振荡器网络能够生成负有效权重，从而支持自发的相位对齐与反相稳定态，并给出 Floquet 稳定性分析

**🔧 技术方法**

使用 Wien 桥振荡器、可变电阻 memristor、LTspice 电路仿真、相位简化模型与 Floquet 理论、Hebbian 型相位依赖学习规则

**📊 数据集**

手写数字 MNIST 的前六类（8×8 二值化原型）用于 64 芯振荡器自关联记忆实验，数字“0”“1”模式用于 3×3/4×4/5×5 网络

**📈 对比分析**

通过比较含负权重与仅正权重网络的相位误差，负权重网络平均误差约 3×10⁻⁹%，无负权重约 62%；数字识别误差在 0.1%–1.2% 之间，证明负权重显著提升稳定性和性能

**⚠️ 局限性**

仅在小规模网络实现自学习，较大网络使用固定电阻，仿真成本高；实际硬件实现与可扩展性仍待进一步验证

---

## 120. ASPIRE: Agentic /Skills Discovery for Robotics

**arXiv ID:** 2607.00272 | [PDF](https://arxiv.org/pdf/2607.00272v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 121. Independent Set Hardness in Graphs of Bounded Twin-Width and Low-Radius Merge-Width

**arXiv ID:** 2607.00244 | [PDF](https://arxiv.org/pdf/2607.00244v1)

**作者:** Édouard Bonnet `[一作]` (Université Lyon), Julien Duron `[通讯]` (University of Warsaw)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过构造逼近算法的逆变换，证明了在图的 twin-width 最高为 4 的类中，最大独立集（Max Independent Set）和最小着色（Min Coloring）问题的近似难度达到 n^γ/(loglog n)^2 的下界；并给出 k-IS 与 k-DS 在低半径 merge-width 图上的 W[1]-难度。

**💡 创新点**

创新点在于：① 结合长偶数细分、词典子图与词典幂的三段式归约，首次在 twin-width 4 的图上得到强势的 ETH 基础下的近似下界；② 把这一思路推广到最小着色问题；③ 通过构造细分块与匹配/共匹配的 gadget，证明 k-IS 与 k-DS 在半径 r = o(k) 的 merge-width 图上仍为 W[1]-难度，并给出相应的 FPT 上界。

**🔧 技术方法**

主要技术包括：PCP、3-SAT→3-Coloring 归约、长偶数细分保持 twin-width ≤4、词典幂保持独立集/色数不变、构造 merge‑sequence 证明 merge-width 边界、Grid Tiling 转化为独立集/支配集 gadget、对比性分析与 ETH 的时间复杂度下界。

**📊 数据集**

本工作为理论研究，无实验数据集；所有结果均基于构造性归约与理论证明。

**📈 对比分析**

与之前的 n^ε（ε>0）近似算法相比较，本文提供了匹配程度的下界 n^γ/(loglog n)^2，几乎与现有的上界 n^(1/loglog n) 对齐；在参数化层面，给出了 W[1]-难度与 FPT 上界的完整阐述，阐明了半径 merge‑width 对 k‑IS/k‑DS 计算复杂度的影响。

**⚠️ 局限性**

局限性包括：① 近似下界仅适用于 twin-width ≤4，尚未扩展到更高 twin‑width；② 仍未消除 n^(1/loglog n) 与 n^γ/(loglog n)^2 之间的常数指数差距；③ 对于实际图形数据未给出实验验证，研究范围限定在理论上。

---

## 122. Leveraging Phase Information to Boost Unrolled Network Learning for Image Deblurring

**arXiv ID:** 2607.00251 | [PDF](https://arxiv.org/pdf/2607.00251v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 123. TallyTrain: Communication-Efficient Federated Distillation

**arXiv ID:** 2607.00173 | [PDF](https://arxiv.org/pdf/2607.00173v1)

**作者:** Radhakrishna Achanta `[一作]` (Cisco Systems Inc), Will Reed `[通讯]` (Cisco Systems Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种仅传输每个 probe 的最大类别索引的硬标签联邦蒸馏协议。

**💡 创新点**

压缩类别维度至 O(log C) 位，利用多数投票去噪，结合稀疏 FedAvg 合并，形成 Pareto 主导的带宽-精度折中。

**🔧 技术方法**

硬标签投票、软标签对齐、KL 收敛分析、稀疏参数合并、全网路、全局对齐等技术。

**📊 数据集**

CIFAR‑10/100 的非IID 图像任务以及 WikiText‑2 的 BPE‑2048 语言建模任务。

**📈 对比分析**

与 FedMD、FedDF、FedAvg、FedProx 等软标签/参数平均基线比较；在 CIFAR‑100 仅用 400× 较少带宽即可达到或超越软标签；在 CIFAR‑10 通过带宽‑桥接方案在 214× 带宽下比 FedAvg 提升约 21pp。

**⚠️ 局限性**

实验仅覆盖 N≤10、全网状拓扑、同构模型；未验证稀疏 gossip、异构架构或大规模 LM；纯 KL 模式在跨分布或高熵目标下易崩溃，需要桥接。

---

## 124. Query-Centric Optimization of AI Workflows via Approximate Query Processing and Proxy Models

**arXiv ID:** 2607.00254 | [PDF](https://arxiv.org/pdf/2607.00254v1)

**作者:** Huayi Wang `[一作]` (Georgia Institute of Technology), Gromit Yeuk-Yin Chan `[通讯]` (Adobe Research)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将现代 AI 工作流重新表述为可声明的查询，并通过近似查询处理（AQP）与代理模型（PM）两种数据库技术，减少昂贵模型调用次数，提升工作流吞吐量。

**💡 创新点**

创新点在于提出“查询中心化 AI 工作流”视角，利用查询级别的信息（聚合、筛选、排序）实现对模型调用的自适应裁剪，并将传统 AQP 与轻量级决策树代理模型融合，既能保留统计置信度，又能利用结构特征实现高效过滤。

**🔧 技术方法**

主要技术包括在线聚合（online aggregation）与置信区间自适应停止、决策树代理模型的特征工程与门控机制（variance gate + candidate filter）、以及基于查询结构的策略选择规则。

**📊 数据集**

实验数据集包括基准 TPC‑DS（8个聚合查询）以及三类 LLM 后训练流水线（数学推理 GSM8K、代码生成 HumanEval+/MBPP、通用指令跟随 UltraFeedback），使用四个生成模型（Qwen3‑8B/32B、LLaMA‑3.1‑8B、Mistral‑7B‑V0.3）。

**📈 对比分析**

与全模型调用基线对比，AQP 在平衡标签分布下可将聚合误差控制在10%以内，且仅使用10–15% 的 oracle 调用；PM 在结构特征显著的数学和代码任务中可将 reward‑model 调用量降低至原来的 5–10%（最高 19× 速度提升），在开放式指令任务中降低 2–3×。两种策略互补，在大多数场景下实现 60–90% 的调用量削减，同时保持 5–15% 的准确率损失。

**⚠️ 局限性**

局限包括：AQP 对小表或极端偏斜标签分布效果不佳，PM 依赖可判别的结构特征，在自由文本任务中效果有限；两种策略均需要先采样并标注一部分数据，且对极高精度要求的场景可能需要完整调用；未来需扩展至多步骤代理工作流与检索增强生成等更复杂流水线。

---

## 125. LV-ROVER: Multi-Stream Tesseract Voting for Maltese Paragraph OCR

**arXiv ID:** 2607.00250 | [PDF](https://arxiv.org/pdf/2607.00250v1)

**作者:** Adam Darmanin `[一作]` `[通讯]` (Independent Researcher), Adam Darmanin (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个低资源 Maltese OCR 系统，采用合成训练数据、5 流 Tesseract LV‑ROVER 集成以及一条五阶段后处理链，最终在 422 段落基准上将字符错误率从 0.0234 降至 0.00700。

**💡 创新点**

创新点包括：① 用 11 个带 diacritics 的域配置和 68 种字体的合成生成流水线；② “双 CER”评估协议，将识别提升与标注约定对齐区分；③ 基于规则的行合并器与软词典 ROVER 投票，保留 diacritics 并实现可观的多样性投票；④ diacritic canary 监控确保模型不丢失 diacritics；⑤ 仅使用 CPU、无 GPU 的轻量化推理方案。

**🔧 技术方法**

主要技术：Tesseract 5 LSTM 细调、5 流投票（LV‑ROVER）、规则化软 hyphen 处理、标点与引号的文本规范化、基于词典的 soft 词频投票、synthetic 训练数据生成与增强。

**📊 数据集**

数据集：① 467M-token Maltese korpus_malti（19 域，11 diacritised 配置）用于合成训练；② 57 页 NOMOCRAT 真实 PDF 标注；③ 422 段落开发集（Dev）用于评估与调优；④ 其他 3 份合成验证集用于抽样检验。

**📈 对比分析**

与基线比较：Fine‑tuned Tesseract 单流 CER 0.0234；LV‑ROVER 集成后 44% 识别提升至 0.01317；加上后处理 70% 总体提升至 0.00700；对比 Neural Transformer 解码器（TrOCR、FasterDAN、Pix2Struct）未能缩小合成‑真实差距。

**⚠️ 局限性**

局限性：① 合成‑真实差距仍大，未在真实测试集验证；② 仅针对 422 Dev 进行调优，可能过拟合标点约定；③ 依赖 CPU，无 GPU 加速；④ 数据集规模仍有限，Maltese 真实标注不足；⑤ 对 diacritic 识别的安全性依赖 canary 检测，无法完全排除字体渲染问题。

---

## 126. Knowing Who, Not How Much: Learning-Augmented Mechanisms for Consumer Utility Maximization

**arXiv ID:** 2607.00175 | [PDF](https://arxiv.org/pdf/2607.00175v1)

**作者:** Kira Goldner `[一作]` (Boston University), Thodoris Tsilivis `[通讯]` (Boston University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在随机顺序的单件拍卖中，设计了一种确定性机制，能够在完全信息下实现常数近似的消费者效用，并在引入预测后实现一致性与鲁棒性的最佳两种世界保证。

**💡 创新点**

创新点在于：①提出“谁是最高价值者”的预测形式能突破 O(log n) 的不可逼近壁垒；②将离散化的 RSOL 随机机制通过确定性分阶段实现；③在同一框架下同时给出常数一致性与鲁棒性保证，首次在消费效用最大化中实现最佳两种世界结果。

**🔧 技术方法**

主要技术包括：随机顺序模型分析、离散化与强制均衡性质（balanced property）的概率耦合、确定性分阶段机制（sample、max、lottery 三阶段）以及学习增补机制的预测窗口设计。

**📊 数据集**

本文为理论分析，没有使用实际数据集；所有结果均通过数学证明给出。

**📈 对比分析**

与现有工作相比，本文的确定性机制在鲁棒性上实现了 O(1) 近似，而先前的随机机制仅能得到 O(1/1250) 的常数；在一致性方面，当预测正确时可达到 1 的近似，比以往只靠最高值预测无法实现常数近似的结果显著优越；总体上，本文提供了常数近似的最佳两种世界保证。

**⚠️ 局限性**

局限性包括：常数因子（如 1/625e）尚不一定是最优，且对更复杂的拍卖形式（多件、带预算约束等）尚未扩展；在预测不完全正确时的鲁棒性仍受到线性损失的限制；此外，实验验证缺失，未能验证在真实数据上的表现。

---

## 127. Structural Pattern Mining in Inka Khipus: Unsupervised Clustering, Provenance Classification, and a Computational Validation of the Santa Valley Match

**arXiv ID:** 2607.00185 | [PDF](https://arxiv.org/pdf/2607.00185v1)

**作者:** Maria Contreras `[一作]` `[通讯]` (Universidad Peruana de Ciencias Aplicadas), Maria Contreras (Universidad Peruana de Ciencias Aplicadas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了印加时代khipu的结构特征，利用机器学习对Open Khipu Repository进行无监督聚类、监督分类和可解释性分析，并通过公开数据库独立验证Santa Valley的moiety结构。

**💡 创新点**

发现khipu可分为三类结构群，其中一类对应帝国式，另一类表现出殖民机构的偏差；确定绳索扭转方向是帝国khipu的关键区分特征；首次用公开数据库独立验证Santa Valley结构，验证了先前研究的结型比例。

**🔧 技术方法**

采用UMAP降维、HDBSCAN聚类、XGBoost分类、SHAP解释、TF‑IDF n‑gram特征、统计检验等技术。

**📊 数据集**

使用Open Khipu Repository（619件khipu，54,403条绳索，110,677个结）及其结构、结、颜色等属性。

**📈 对比分析**

聚类得到silhouette=0.769，三类分离明显；分类模型在帝国式样本上F1=0.86，整体weighted F1=0.46；Santa Valley的recto/verso比例与原研究（约47%/53%）高度一致。

**⚠️ 局限性**

限制包括标注样本量少、标签不精确、数据存在采集与记录偏差、缺失值对特征影响、未利用结序信息、未完成数值解码等。

---

## 128. Understanding Guest Preferences and Optimizing Two-sided Marketplaces: Airbnb as an Example

**arXiv ID:** 2607.00280 | [PDF](https://arxiv.org/pdf/2607.00280v1)

**作者:** Yufei Wu `[一作]` (Airbnb, Inc.), Daniel Schmierer `[通讯]` (Airbnb, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过经济模型和因果推断方法，结合观察数据与实验结果，量化Airbnb客人对价格的敏感度及偏好异质性，并基于此优化定价工具与个性化体验。

**💡 创新点**

创新点在于：①使用基于供给增长的地理差异作为价格工具，实现在无实验干预下估计价格弹性；②将实验结果用于校准观测估计，弥补因果估计的偏差；③通过对客群份额变化的面板回归，提取不同客群的价格弹性差异，实现偏好细分。

**🔧 技术方法**

主要技术包括：离散选择模型（logit）、两阶段最小二乘法（IV）与供给驱动的工具变量、面板线性回归、实验验证与校准（加权修正）以及对价格弹性的计算。

**📊 数据集**

使用的数据集为Airbnb平台的观测数据（价格、供应、需求、客人预订记录等），以及若干定价实验数据（随机价格变动实验）作为验证基准。

**📈 对比分析**

通过将观测估计与实验估计对比，验证模型的准确性；实验结果显示观测估计略高，随后采用“haircut”校正后两者高度一致，表明方法在实际数据中的鲁棒性和可操作性。

**⚠️ 局限性**

限制主要包括：工具变量的排除约束假设难以完全验证；观测数据中价格变动仍可能受未观测需求因素影响；logit模型对零份额处理的简化可能导致偏差；实验样本量有限，影响校准的精度。

---

## 129. SLM, LLM or Agentic AI? Toward Intelligent UAV-Enabled WPT Systems in Low-Altitude Economy Networks

**arXiv ID:** 2607.00255 | [PDF](https://arxiv.org/pdf/2607.00255v1)

**作者:** Feibo Jiang `[一作]` (Hunan Normal University), Abbas Jamalipour `[通讯]` (University of Sydney)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在低空经济网络中，研究了三种智能路径规划框架（轻量化小型语言模型 SLM、LLM 与 Agentic AI），用于无人机无线电能传输系统的路径与时间分配优化。

**💡 创新点**

创新点：提出基于 BERT 的 SLM 轻量化路径解码器，并通过几何增强、集成推理与强化学习实现低延迟；构建四代理闭环 Agentic AI 以提升自适应决策；同时系统比较三种方法在能耗、约束满足率与推理时延上的优劣。

**🔧 技术方法**

技术：BERT 变换器、几何增强嵌入、路径解码器、增强推理、强化学习、LLM（GPT‑4o）、多代理协同、MCP 工具链（2‑opt、KKT）。

**📊 数据集**

数据集：随机生成的 1000 m×1000 m 区域内 500 台 IoTD 与 100 个聚类中心，采用 LS‑FCM 确定悬停点，模拟多 UAV 任务；还使用视觉化坐标图作为 LLM 输入。

**📈 对比分析**

对比：SLM 在能耗与标准差最小、推理时间最快；LLM 在小规模下能耗较低但约束满足率低；Agentic AI 结合 2‑opt 可逼近最优但推理时延显著增加。

**⚠️ 局限性**

局限：SLM 任务专一，需重新训练以适应不同约束；LLM 与 Agentic AI 的推理时延与计算成本高，且在大规模问题中表现衰退；整体缺乏对实时通信延迟与边缘部署能耗的评估。

---

## 130. Does Your ViT Still Need U-Net for Segmentation?

**arXiv ID:** 2607.00223 | [PDF](https://arxiv.org/pdf/2607.00223v1)

**作者:** Xin Li `[一作]` (Arizona State University), Yalin Wang `[通讯]` (Arizona State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了纯 Encoder 的医学图像分割框架 EoSeg，验证在强大预训练 ViT 的支持下可不再使用 U‑Net 风格解码器。

**💡 创新点**

创新点包括：①采用查询式 mask 预测替代传统像素级解码；②跨层多级查询建模；③可学习块融合自适应融合不同 Transformer 阶段的输出；④在 Encoder‑only 结构中实现高质量分割。

**🔧 技术方法**

使用技术包括：大规模预训练 ViT（DINOv2），查询嵌入与 MLP 投影，跨层查询交互，学习式块融合门控，深度监督（最终未采用），以及 deconvolution 上采样等。

**📊 数据集**

实验使用七个医学分割基准：Synapse (CT多器官)、ACDC (MRI心脏)、GlaS (肿瘤组织病理)、MoNuSeg (细胞核组织病理)、Kvasir‑Seg (内镜息肉)、ISIC‑2016 与 ISIC‑2017 (皮肤病变)。

**📈 对比分析**

与多种 CNN、ViT+UNet、SwinUNet、TransUNet 等传统或 Transformer‑decoder 方法对比，EoSeg 在所有数据集上均获得最高或最接近最高的 Dice/IoU（如 Synapse 85.5% Dice，ACDC 91.73% Dice，GlaS 93.27% Dice，MoNuSeg 80.51% Dice，Kvasir‑Seg 91.08% Dice，ISIC‑2016 93.22% Dice，ISIC‑2017 94.33% Dice）。

**⚠️ 局限性**

局限性：①对大规模预训练 ViT 依赖较强，低资源或少样本场景下性能未知；②查询数量与类别数耦合，需根据任务手动调节；③对极细小结构或高度重叠多实例情况的鲁棒性尚未系统评估。

---

## 131. SLIM-RL: Risk-Budgeted Random-Masking RL for Diffusion LLMs Without Trajectory Slicing

**arXiv ID:** 2607.00208 | [PDF](https://arxiv.org/pdf/2607.00208v1)

**作者:** Ruikang Zhao `[一作]` (Technical University Of Denmark), Ligong Han `[通讯]` (Red Hat Ai Innovation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SLIM-RL方法，在不重建解码轨迹的前提下使用强化学习提升扩散大型语言模型（dLLM）的推理与推理性能。

**💡 创新点**

创新点包括：τ‑预算动态去蒙版解码器、序列级长度归一化重要性比率、确定性高斯‑勒让德求积减少梯度方差、以及单调递减的每块蒙版调度，形成一套无轨迹随机蒙版RL框架。

**🔧 技术方法**

技术手段：τ‑预算解码器、改进的GRPO、序列级重要性比率、无归一化优势、确定性高斯‑勒让德求积、单调递减蒙版调度以及与块级因果分解的结合。

**📊 数据集**

数据集：在block‑wise SDAR-4B-Chat与SDAR-1.7B-Chat上训练，使用数学题集MATH与PrimeIntellect‑verified coding problems，并在MATH500、GSM8K、MBPP与HumanEval等基准上进行评测。

**📈 对比分析**

与TraceRL、随机蒙版RL以及自动回归模型（Qwen2.5‑7B、LLaDA‑8B、Dream‑7B）对比，SLIM‑RL在block size 16时MATH500提升6.32%、GSM8K提升11.05%；在block size 4时同样超越TraceRL，并在MATH500上领先LLaDA‑8B 10.76%；在代码基准上提升MBPP 4.20%与HumanEval 3.65%；且仅使用约0.46倍的训练数据即可达到TraceRL最佳精度。

**⚠️ 局限性**

局限性：尚未验证在更大块尺寸或更长响应长度下的可扩展性；对模型置信度校准的依赖可能影响τ‑预算解码器的效果；目前的结果主要基于SDAR体系，对其他dLLM架构的迁移性仍需进一步研究。

---

## 132. DriftScope: Measuring The Hidden Effects of Diffusion Model Adaptation

**arXiv ID:** 2607.00183 | [PDF](https://arxiv.org/pdf/2607.00183v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 133. Play Like Champions: Counterfactual Feedback Generation in Latent Space

**arXiv ID:** 2607.00190 | [PDF](https://arxiv.org/pdf/2607.00190v1)

**作者:** Andrzej Białecki `[一作]` (Warsaw University of Technology), Han Zhou `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了基于 Guided VAE 的星际争霸 II 玩家改进反馈框架，学习压缩的游戏表现表示，并在潜在空间中生成反事实改进路径，给玩家多粒度的即时改进建议。

**💡 创新点**

创新点包括：① 将宏观经济与比赛结果条件化到潜在空间；② 在潜在空间内设计四种遍历策略（线性插值、迭代最优传输、密度正则化梯度上升、神经流匹配）；③ 通过生成多步改进轨迹实现可执行的反馈；④ 将生成的改进轨迹映射回原始特征空间，提供可视化和解释性强的改进方案。

**🔧 技术方法**

使用技术：Guided Variational Autoencoder、条件 VAE、最优传输（OT）、密度正则化梯度上升、神经流匹配、潜在空间遍历、反事实解释与可解释性度量。

**📊 数据集**

数据集：SC2EGSet（23,305 场专业锦标赛回放）用于训练；随机抽取的业余回放（OOD 数据）用于验证与对比。

**📈 对比分析**

比较方法：对同一业余回放分别使用四种遍历策略，评估生成轨迹在保持专家行为特征与向赢配置移动之间的平衡。结果通过可视化、专家评估及潜在空间连贯性指标进行对比，未给出数值性能指标，但指出不同方法在路径连贯性、解码真实性和可解释性方面存在权衡。

**⚠️ 局限性**

局限性：① 仅基于离线回放特征，缺乏实时动态反馈；② 生成路径高度依赖训练数据分布，对未覆盖的策略或极端情况适用性有限；③ 未通过实际玩家实验验证改进建议的有效性；④ 四种遍历方法在高维潜在空间可能面临梯度爆炸或解码失真；⑤ 反馈粒度仍需进一步细化以满足不同水平玩家的需求。

---

## 134. Device Passport: Enabling Spatio-Temporal Pretrained Models to Generalize Across Input Layouts

**arXiv ID:** 2607.00249 | [PDF](https://arxiv.org/pdf/2607.00249v1)

**作者:** Geeling Chau `[一作]` (California Institute of Technology), Christopher M. Sandino `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了Device Passport，一种利用专家嵌入与混合模型学习通道嵌入的技术，以实现跨设备布局的预训练模型迁移。

**💡 创新点**

创新点在于将通道元数据与功能活动共同输入到预训练的专家权重混合模块中，使模型能够在未知布局上快速定位通道空间关系。

**🔧 技术方法**

核心技术包括基于Transformer的CBraMod框架、ACPE位置编码、以及两种Device Passport变体（MLP专家与跨注意力专家）来生成通道嵌入。

**📊 数据集**

实验使用了Temple University Hospital (TUH) EEG语料库进行预训练，并在TUAB异常EEG分类和EESM17耳部EEG睡眠分期任务上进行微调。

**📈 对比分析**

与传统的Sinusoidal、ID、XYZ和ACPE等基线相比，Device Passport在布局不匹配的情况下取得了更优的AUROC和Kappa分数，尤其在中等数据规模时表现突出。

**⚠️ 局限性**

主要限制是对噪声或接触不良通道的敏感性，导致在EESM17中某些受试者的性能下降，提示需要改进噪声鲁棒性。

---

## 135. Joint Effects of Recommender Systems and Network Structure on the Visibility of Content and Creators

**arXiv ID:** 2607.00258 | [PDF](https://arxiv.org/pdf/2607.00258v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 136. Entropy-Regularized Probabilistic Gates for Sparse Model Discovery in Scarce-Data Federated Learning

**arXiv ID:** 2607.00275 | [PDF](https://arxiv.org/pdf/2607.00275v1)

**作者:** Krishna Harsha Kovelakuntla Huthasana `[一作]` (Åbo Akademi University), Andreas Lundell `[通讯]` (Åbo Akademi University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于熵正则化的概率门控 L0 约束框架，在联邦学习中实现稀疏模型训练；

**💡 创新点**

通过熵正则化保持参数空间的不确定性，防止稀疏支持过早锁定，显著提升稀疏回收和泛化能力；

**🔧 技术方法**

使用 Hard Concrete 分布的可重参数化概率门、L0 约束、熵正则化、FedAvg 聚合与梯度上升/下降、再启动 λ 机制；

**📊 数据集**

在三类数据集上评估：合成线性回归、MNIST 手写数字分类的 CNN、Golub 白血病基因表达的 softmax 分类；

**📈 对比分析**

与 Fed-IHT（迭代硬阈值）、FedAvg+稀疏化后剪枝以及中心化训练基准比较；在所有实验中，所提方法在保持相同稀疏度下的测试性能更优，稀疏回收率更高，通信量仅为密集 FedAvg 的一小部分；

**⚠️ 局限性**

仍受限于大规模高维小样本（d ≫ N）下的样本异质性和客户端参与不均衡；对结构化稀疏的适用性及熵正则化在不同数据稀缺与过参数化场景中的行为需进一步研究。

---

## 137. SEFORA: Student Essays with Feedback Corpus and LLM Feedback Evaluation Framework

**arXiv ID:** 2607.00274 | [PDF](https://arxiv.org/pdf/2607.00274v1)

**作者:** Shayan Peyghambari Oskoui `[一作]` (University of Pittsburgh), Xiang Lorraine Li `[通讯]` (University of Pittsburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套公开的教学反馈语料库与评估框架，系统化收集并量化教师在课堂写作中给出的逐段、基于文本段落的反馈，并利用该数据训练与评估大语言模型生成的反馈质量。

**💡 创新点**

1) 公开完整的教师逐段反馈语料库，包含提示、评分表、多稿修订与细粒度文本标注；2) 设计基于反馈单元的参考式评估框架，既细化单元分割，又通过语义相似度与最大权重匹配计算可解释的精准率、召回率和F1。

**🔧 技术方法**

使用大语言模型（如Llama、Mistral、Qwen、GPT‑5）进行生成；利用GPT‑5‑nano实现反馈单元分割，利用Gemini‑3.1‑flash‑lite或GPT‑5‑nano进行反馈单元相似度评分；结合Hungarian算法完成单元匹配，最终得到指标。

**📊 数据集**

自采集的《学生作文与教师反馈语料库》（~564稿，8,240条注释），覆盖四类写作体裁（Essay、Narrative、Explanation、Empathy Writing），并与提示、评分表和多稿修订信息关联。

**📈 对比分析**

在74种模型+提示配置下，采用精确率、召回率与F1对生成反馈进行评估。最高F1仅为0.32，未超过0.4；在大模型下精确率与召回率仍低，表明模型往往产生过量且与教师关注点不一致的反馈。

**⚠️ 局限性**

1) 评估仅基于已标注段落，未涉及span识别与全文评估；2) 依赖闭源模型（GPT‑5、Gemini）进行分割与相似度评分，影响可复现性；3) 未对反馈的教育有效性、可操作性等主观维度进行人类评价；4) 单元分割未进一步拆分复合语句，可能掩盖细粒度误差。

---

## 138. Validating Causal Abstraction Metrics on Simulated Complex Systems

**arXiv ID:** 2607.00267 | [PDF](https://arxiv.org/pdf/2607.00267v1)

**作者:** Maxime Méloux `[一作]` (Université Grenoble Alpes), Maxime Peyrard `[通讯]` (Université Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含十种理想化复杂系统的基准，系统化评估了30多种解释有效性指标，并提出了基于因果抽象误差（CAE）的新度量，证明其能在有限干预样本下准确区分有效与无效的高层解释。

**💡 创新点**

创新点在于：①首次将因果抽象框架统一应用于离散与连续、静态与动态、局部与全局多类系统；②设计了包含无关变量可信度检验的因果抽象误差度量；③在基准上对比了传统观测、信息、功能指标与因果抽象指标，展示仅因果抽象指标能通过所有六类控制失效实验。

**🔧 技术方法**

使用的技术主要包括：结构因果模型（SCM）与抽象映射、Monte Carlo 干预采样、分布相似度度量（KL、JSD、MMD）、统计检验（Mann–Whitney U）、以及对因果抽象误差的期望聚合；实现基于Python的API，支持多节点干预与伪变量插值。

**📊 数据集**

数据集由作者手工构造的十个模拟系统组成：理想气体、捕食者–猎物、1D/2D 热方程、Ising 模型、Tracer transformer、基因调控网络、逻辑电路、晶体管电路、MOS 6502 CPU（晶体管级到门级再到ISA级），每个系统均配有已知的高层因果解释和对应的无效对比。

**📈 对比分析**

在所有基准系统上，CAE 通过所有控制失效实验，AUROC 近 1，且在约 30 次干预样本即可达到 95% 的检出功效；相比之下，观测、功能或信息理论指标均无法在所有失效实验中显著区分，且许多指标适用范围有限。

**⚠️ 局限性**

局限性包括：仅评估已给定的 (M,E,τ) 组合，未解决可搜索的抽象映射；宏观模型仅为确定性，未完整处理宏观噪声耦合；基准的系统与对比设计是作者主导，可能存在偏见，且未覆盖更广泛的自然科学领域。

---

## 139. From Signals to Structure: How Memory Architecture Drives Language Emergence in LLM Agents

**arXiv ID:** 2607.00233 | [PDF](https://arxiv.org/pdf/2607.00233v1)

**作者:** Yashar Talebirad `[一作]` (University of Alberta), Osmar R. Zaiane `[通讯]` (University of Alberta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

研究在Lewis信号游戏中，LLM代理通过不同记忆架构从零开始发明共享语言，并评估记忆结构与信道容量对协作效果的影响。

**💡 创新点**

证明记忆架构比信道容量更决定LLM的语言形成，发现持续记事本能利用多余容量稳定约定，且容量=8是易失点而非最优点。

**🔧 技术方法**

采用大型语言模型（如Claude）在提示中实现滚动窗口、持久记事本、代码本等多种记忆模式，并结合Lewis游戏的交互式任务。

**📊 数据集**

使用八个由颜色、形状、大小组合而成的离散对象集，在每轮提供四个候选目标的人工设置的游戏环境。

**📈 对比分析**

通过多种容量设定（4–125）和三种随机种子，计算窗口精度、TopSim、互信息与碰撞率；结果显示“scratchpad”在中高容量下最高精度（≈0.90），而“memory_only”在容量25时最高但在64时崩溃，凸显记忆架构对性能的决定性影响。

**⚠️ 局限性**

仅使用单一LLM模型、少量随机种子、固定对象空间、无跨代传递，导致统计置信度有限且缺乏对更大或不同模型的普适性验证。

---

## 140. Guaranteed Escape for a Bouncing Robot in Pipe Chains

**arXiv ID:** 2607.00221 | [PDF](https://arxiv.org/pdf/2607.00221v1)

**作者:** Ahmad Kamaludeen `[一作]` (Toronto Metropolitan University), Yeganeh Bahoo `[通讯]` (Toronto Metropolitan University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对正交管道（由等宽矩形段组成）中点机器人对称弹跳运动进行系统研究，完成了单段端块（End Block）内所有弹跳轨迹的完整分类，并基于此推导出线性管道链（L形、k 段）中的退出保证，证明了特殊角度 α = π/4、无理斜率以及圆形连接管道时均能保证机器人退出。

**💡 创新点**

① 将单段端块内的九种弹跳模式完全枚举，形成判定表；② 通过角度传播规则（相邻段角度互补）和折叠映射（T_i）建立起多段管道的状态转移；③ 在无理斜率和圆形连接管道下给出一般性退出保证，填补了此前只针对单段或共轭斜率的空白；④ 通过归纳证明 α = π/4 的单次通过保证。

**🔧 技术方法**

采用经典的折叠（unfolding）方法把弹跳轨迹映射为直线，利用反射对称性推导端块分析；构造线性映射 T_i 表示段间的水平位移；利用数论（有理/无理斜率）与 Weyl 定理论证密度和周期性；归纳法与图论相结合证明多段链的退出保证。

**📊 数据集**

无实验数据集，全部结果为理论证明与解析推导。

**📈 对比分析**

本工作不涉及实验比较；其性能体现在理论上对所有可能初始位置、角度（除了特殊角度）均能保证退出，提供了最优角度选择（α=π/4）和一般斜率下的退出保证。

**⚠️ 局限性**

仅处理直线型管道链，未考虑分支（T型、Y型）或循环网络；仅适用于等宽矩形段和圆形连接；对非正交或宽度不等的管道未给出解析；在高维或复杂拓扑时仍需进一步研究。

---

## 141. Guesswork Under Linear Constraints: Exact Exponent for Coset Decoding

**arXiv ID:** 2607.00205 | [PDF](https://arxiv.org/pdf/2607.00205v1)

**作者:** Hassan Tavakoli `[一作]` `[通讯]` (Oregon State University), Hassan Tavakoli (Oregon State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在随机二进制线性码中受线性约束的猜测工作（coset guessing），并推导出其精确指数、二阶修正以及通用形式。

**💡 创新点**

创新点包括：①给出受约束猜测工作指数的闭式表达式；②提出“转移定理”，将任何权重枚举增长率映射为猜测指数；③给出列表输出和二阶修正的完整分析；④推广到 q 进制和 LDPC 码的通用结果。

**🔧 技术方法**

主要技术手段是组合概率论与大偏差理论、权重枚举的收敛性证明、离散拉普拉斯变换与变分原理、以及对香农-阿里坎猜测猜测猜测。

**📊 数据集**

使用随机全秩码族、正则 LDPC 码族、以及（7,4,3）Hamming 码作为实验对象；通过 Monte Carlo 仿真评估指数。

**📈 对比分析**

与无约束猜测工作指数、以及不同码率的随机码相比，实验显示受约束指数比无约束低 ρ(1−R) 位，查询复杂度随码率下降显著；仿真误差随 n 增大呈 ρlog₂n/n 下降。

**⚠️ 局限性**

局限性在于假设码族满足权重枚举的指数收敛性（如随机全秩码族或正则 LDPC 族），且需要满足子临界条件；对非随机或结构更复杂的码族的适用性尚未证明；有限长度下的误差仅有上限估计。

---

## 142. A Non-Line-of-Sight, Multi-Modality-based Side-Channel IP Theft Attack on Additive Manufacturing Using Dual Smartphones

**arXiv ID:** 2607.00186 | [PDF](https://arxiv.org/pdf/2607.00186v1)

**作者:** Amirhossein Jamarani `[一作]` (University of Louisiana at Lafayette), Xiali Hei `[通讯]` (University of Louisiana at Lafayette)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用两部智能手机的麦克风和磁力计在 60 cm 非视线距离收集 3D 打印机的声学与磁场侧信道信号，并通过多模态信号预处理、同步与滑动窗口特征提取，结合混合 LSTM‑CNN 模型实现对 G‑code 命令的端到端重建。

**💡 创新点**

① 双手机多模态侧信道攻击实现远距（≥ 60 cm）非接触式入侵；② 在真实工厂环境下引入动态噪声建模与多台打印机干扰，显著提升攻击鲁棒性；③ 通过实验验证方法在不同打印机、不同距离和不同对象上的高迁移性。

**🔧 技术方法**

① 低成本手机内置传感器（麦克风、磁力计）；② 信号同步与重采样、滤波与归一化；③ 滑动窗口特征提取；④ 混合 CNN‑LSTM 深度学习模型进行运动与挤出状态分类；⑤ 端到端的 G‑code 逆向合成。

**📊 数据集**

自行构建的侧信道数据集：12 次独立打印会话（共 24,000 个窗口），涵盖多种对象（齿轮、涡轮、扳手）及两台不同品牌打印机（LulzBot TAZ、Creality），并在不同距离（30–150 cm）和不同环境下收集的声学与磁场记录。

**📈 对比分析**

与 SVM、随机森林、CNN、LSTM 等基线模型对比，CNN‑LSTM 达到 97.99 % 的运动分类精度，最终 G‑code 命令级重建准确率为 98.89 %；在 70 cm 时仍保持 98.0 % 以上，显示出在非线视距与嘈杂环境下的高性能。

**⚠️ 局限性**

① 远距离 (> 100 cm) 时信噪比急剧下降，重建准确率下降；② 仅在目标打印机发出主导信号时有效，其他设备干扰时效果受限；③ 依赖于手机传感器的精度与摆放角度，需在特定硬件与环境中进行校准；④ 对极端噪声或多台打印机混合操作的抗干扰能力尚待进一步提升。

---

## 143. PRISM-VO: Scale-Aware Visual Odometry Using Photometric Plenoptic Bundle Adjustment

**arXiv ID:** 2607.00176 | [PDF](https://arxiv.org/pdf/2607.00176v1)

**作者:** Aymeric Fleith `[一作]` (Technical University Of Munich), Niclas Zeller `[通讯]` (Karlsruhe University Of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于聚焦光场相机的稀疏光度视觉里程计PRISM‑VO，利用光度与光场深度信息在滑动窗口中联合优化相机位姿与逆深度；

**💡 创新点**

核心创新在于光度光场束束调和：显式嵌入光场投影模型，利用单张光场图像的光度与深度约束实现绝对尺度恢复，并自适应平衡光度残差与深度残差；

**🔧 技术方法**

使用纯优化光度法、滑动窗口光度束束调和、逆虚深度残差、Huber范数、Levenberg–Marquardt、变异权重深度残差、光场微透镜投影等技术；

**📊 数据集**

采用Raytrix R5与R32光场相机同步立体视觉数据集以及LiFMCR光场数据集（含Vicon 6DoF ground truth）；

**📈 对比分析**

通过循环漂移、尺度误差、旋转误差、对齐误差等指标与SPO光场VO、DSO、ORB‑SLAM3、DPVO等方法对比，PRISM‑VO在尺度误差≤1.05、旋转误差<4°、尺度漂移与对齐误差均低于SPO，并在多条序列上优于单目方法，达到或超过最先进的学习与优化方法；

**⚠️ 局限性**

受限于光场相机的低分辨率与视场、长距离深度精度不足，且对光照变化仍有一定敏感性；

---

## 144. Efficient LCE Queries and Lexicographic Minimizers on Sliding Suffix Trees

**arXiv ID:** 2607.00389 | [PDF](https://arxiv.org/pdf/2607.00389v1)

**作者:** Toshiharu Minematsu `[一作]` (Kyushu University), Shunsuke Inenaga `[通讯]` (Kyushu University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究滑动窗口下的后缀树上最长公共扩展（LCE）查询与词典序最小子串（minimizer）维护，提出在无终止符的后缀树中通过周期性代表映射实现隐式后缀的显式折叠，从而实现常数时间LCE查询和精确minimizer更新。

**💡 创新点**

创新点在于利用最长隐式后缀产生的周期结构，将所有隐式后缀折叠到显式叶子，实现O(1) LCE查询；并提出一种完全不使用LCE或动态LCA的minimizer维护方案，利用BP‑linked后缀树、顺序维护标签与前沿节点/令牌实现O(1)比较与更新。

**🔧 技术方法**

核心技术包括：滑动后缀树（Ukkonen式）、叶子指针、周期代表映射、动态LCA、BP‑linked后缀树、顺序维护结构、单调双端队列、前沿节点/令牌与代表更新机制。

**📊 数据集**

本研究为理论算法分析，未使用具体实验数据集。

**📈 对比分析**

通过上述技术，构造的空间为线性O(d)，滑动窗口每次移位的摊销更新时间为O(1)，LCE查询时间为O(1)，minimizer查询时间为O(1)。实验对比未给出，但与基于LCE+动态LCA的方法相比，新的LCE‑free方案省去了对LCA和哈希的依赖，保持相同的时间复杂度。

**⚠️ 局限性**

局限性在于仅适用于常数大小字母表（对一般大小字母表更新时间为O(logσ)）；实现复杂度较高，对隐式后缀的周期结构假设在某些边缘情况可能不如预期稳健；并未提供实验验证，实际性能需进一步评估。

---

## 145. SAOT: Self-Supervised Continual Graph Learning with Structure-Aware Optimal Transport

**arXiv ID:** 2607.00377 | [PDF](https://arxiv.org/pdf/2607.00377v1)

**作者:** Yuting Zhang `[一作]` (Tiangong University), Xiao Wang `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8d10c613-917e-4880-9716-17789f50e119` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究无监督持续图学习，提出结构感知最优传输（SAOT）框架，实现在连续任务中保持图结构。

**💡 创新点**

创新点：①使用最优传输对节点及边的关系进行全局对齐；②在表示学习中引入跨任务结构蒸馏，显著降低结构漂移。

**🔧 技术方法**

技术手段：最优传输理论（Gromov-Wasserstein + 融合距离）、GNN编码器（GCN/GAT）、跨任务蒸馏、重放记忆、Sinkhorn 迭代等。

**📊 数据集**

实验数据集：CoraFull-CL、Arxiv-CL、Reddit-CL、Products-CL 四个持续图学习基准。

**📈 对比分析**

方法对比：与监督基线和多种自监督基线在 Class-IL 与 Task-IL 两种评估下比较，SAOT 在 AP 上普遍领先，Class-IL 上提升 5%~15%，同时保持低遗忘。

**⚠️ 局限性**

局限性：OT 对齐计算复杂度高（二次），需采样降低；在大图上内部分布对齐效果不明显；对超参数（α、β）敏感。

---

## 146. Enhancing Flow Matching with A Unified Guidance Framework for Efficient and Robust Speech Synthesis

**arXiv ID:** 2607.00363 | [PDF](https://arxiv.org/pdf/2607.00363v1)

**作者:** Zuda Yu `[一作]` (Zuoyebang), Yang Song `[通讯]` (Zuoyebang)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出统一的指导框架，利用数据级异构增强和模型级精细化指导，改进 Flow Matching 语音生成的声纹泄漏和推理延迟问题。

**💡 创新点**

创新点在于：① 双阶段异构扰动（模型驱动跨合成 + 信号驱动声学变形）实现语义 token 与声纹的彻底解耦；② 将 Classifier‑Free Guidance（CFG）信息内化为模型权重，并在线进行轨迹直线化，从而一次前向传递即可完成高质量采样。

**🔧 技术方法**

核心技术包括 Conditional Flow Matching、Diffusion Transformer（DiT）解码器、AdaLN 语音注入、模型驱动跨合成、声学变形（音高、能量扰动）、CFG 归约、轨迹直线化、在线梯度蒸馏。

**📊 数据集**

训练数据：基于 Emilia 语料库 50k 小时预训练；再精选 30k 小时 DNSMOS>3.2 高质量子集，进行双阶段扰动生成 30k 小时对抗对；合计 60k 小时用于统一优化；评估使用 LibriTTS 与 Seed‑TTS 两套数据集。

**📈 对比分析**

与基线（10 NFE、含 CFG）和各自单独改进（DG、MG）对比，统一框架在 VC 与 TTS 上实现了约 3.25× 的推理速度提升（RTF 从 0.078 降至 0.024），声纹相似度（SIM）提高至 0.887，且零样本场景下非平行 SIM 超越真实声纹（0.808>0.799）；在 TTS 任务中 WER 与基线相近（2.60 vs 2.57），SIM 更优（0.888 vs 0.871）。

**⚠️ 局限性**

局限性包括：① 极端轨迹直线化导致轻微 WER 下降；② 对极端跨声纹或极低资源场景的鲁棒性尚未完全验证；③ 仍需大量高质量扰动数据以保证声纹解耦，数据准备成本高。

---

## 147. K-Inverse-RFM: A Modified RFM that Bridges the Gap to Neural Networks for Data-Corrupted Mathematical Tasks

**arXiv ID:** 2607.00329 | [PDF](https://arxiv.org/pdf/2607.00329v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 148. Fully Distributed Tâtonnement for Chores Markets

**arXiv ID:** 2607.00300 | [PDF](https://arxiv.org/pdf/2607.00300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 149. Promise-Future Synchronization for Cluster Asynchronous Many-Task Runtimes via MPI One-Sided Communication

**arXiv ID:** 2607.00303 | [PDF](https://arxiv.org/pdf/2607.00303v1)

**作者:** Mia Reitz `[一作]` `[通讯]` (University of Kassel), Mia Reitz (University of Kassel)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过实现承诺-未来（promise‑future）模型，扩展了 ItoyoriFBC 运行时，使其能够动态构建任务依赖并支持 Hierarchical LU（HLU）分解等算法。

**💡 创新点**

创新点在于：① 引入分布式引用计数和 MPI 一侧 waiter 列表来管理动态创建的 futures；② 解决了未来在生产者尚不存在时消费者可能先访问的问题，实现了任务挂起与恢复。

**🔧 技术方法**

使用技术包括：分布式引用计数、MPI 一侧原子操作（CAS）、无锁链表实现的 waiter 列表、锁定/解锁机制以安全回收共享状态。

**📊 数据集**

使用的数据集是 HLU benchmark，采用高度为 7 的完整四叉树（共 21,845 个任务）模拟 100 秒的数值工作量。

**📈 对比分析**

通过将多节点运行时间与单节点运行时间比值计算加速比，实验显示在 16 节点上可获得 15.6× 的加速，比预期近乎理想，且 10 次实验方差较小。

**⚠️ 局限性**

局限性包括：① 参考计数实现仍可优化，尚未聚合增减操作；② 仅评估了 HLU 一个算法，未与原始未来‑仅模型进行直接对比；③ 在更大规模或不同算法下的适用性未进一步验证。

---

## 150. Unleashing More Actions via Action Compositional Training for VLA Models

**arXiv ID:** 2607.00351 | [PDF](https://arxiv.org/pdf/2607.00351v1)

**作者:** Kai Peng `[一作]` (Shenzhen Technology University), Xiaojiang Peng `[通讯]` (Shenzhen Technology University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ACT-VLA，通过离线利用文本潜在插值合成新的演示，扩展VLA模型的训练分布，从而提升组合泛化能力。

**💡 创新点**

创新点在于把推理时的文本潜在插值转化为训练时的数据增强流程，并在指令设计、质量控制与训练动态上做了系统化改进。

**🔧 技术方法**

采用Transformer‑based VLA、文本潜在插值（TLI）、离线合成、双重过滤（完成度+步骤预算）以及行为克隆训练技术。

**📊 数据集**

主要使用LIBERO仿真基准，特别是其Spatial‑OOD和Goal‑OOD子集来评估组合泛化。

**📈 对比分析**

与UniVLA、OpenVLA‑OFT、GR00T‑N1、π_0等基线比较，ACT‑VLA在Spatial‑OOD和Goal‑OOD的成功率分别提升52.7%和49.0%，平均性能显著超过所有基线。

**⚠️ 局限性**

局限性包括仅针对两技能顺序组合、对预训练VLA潜在表征高度依赖、实验仅在仿真环境，未验证真实硬件上的迁移效果。

---

## 151. DiscoLoop: Looping Discrete Embeddings and Continuous Hidden States for Multi-hop Reasoning

**arXiv ID:** 2607.00341 | [PDF](https://arxiv.org/pdf/2607.00341v1)

**作者:** Hengyu Fu `[一作]` (University of California, Berkeley), Song Mei `[通讯]` (University of California, Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究隐式多跳推理任务，分析循环Transformer在两跳推理中的存储与表征瓶颈，提出在循环过程中同时携带连续隐藏状态与解码后的离散嵌入的混合通道架构（称为“混合通道循环Transformer”），并在符号、人工生成的自然语言以及大规模语言模型预训练中验证其有效性。

**💡 创新点**

核心创新在于识别出循环Transformer在第一次循环后桥接实体虽然可解码但其隐藏表示与对应嵌入不对齐导致表征瓶颈；为此引入可学习的离散嵌入通道（soft decode‑then‑encode）与连续通道的混合，显著提升了两跳推理的泛化能力并加速了训练收敛。

**🔧 技术方法**

技术手段包括：
- 循环Transformer结构（looped Transformer）
- 可学习门控的软解码嵌入投射 Φ(h) 与 RMSNorm 结合
- 训练自由的表征对齐干预（α调节）
- 机制解释方法（logit lens）
- 在预训练中采用多步循环与共享输入输出嵌入

**📊 数据集**

使用的数据集：
- 符号两跳推理数据集（两图，每图500实体、5k原子事实）
- 人工生成的自然语言两跳推理数据集（直接/反向两种问法）
- 预训练语料FineWeb‑Edu + FineMath（20B token）用于评估零样本语言建模性能

**📈 对比分析**

与基线比较：
- 传统非循环Transformer
- vanilla looped Transformer（仅连续隐藏状态）
- PonderLM（仅嵌入循环）
结果：
- 在符号和自然语言两跳任务中，混合通道模型在ID上几乎达到100%准确率，在OOD上提升至80%以上，远超vanilla（≈70%/10%）与非循环（≈20%/0%）。
- 在语言模型零样本评估上，平均得分从vanilla的49.3提升至50.5（PonderLM为49.8），在ARC‑C、LAMBADA、SciQ等指标上取得显著优势。

**⚠️ 局限性**

局限性：
- 仅在中等规模（440M）模型上验证，尚未验证更大规模模型的可扩展性；
- 只测试至两跳，未系统评估在更多跳数或跳数外推理（extrapolation）的能力；
- 训练过程仍需数千epoch，尽管收敛更快，但在实际大规模预训练时的计算成本仍较高；
- 该方法是从头训练的，尚未探究如何在已有的大型非循环预训练模型上迁移该混合通道循环机制。

---

## 152. NeHMO: Neural Hamilton-Jacobi Reachability Learning for Decentralized Safe Multi-Arm Motion Planning

**arXiv ID:** 2607.00326 | [PDF](https://arxiv.org/pdf/2607.00326v1)

**作者:** Qingyi Chen `[一作]` (Purdue University), Ahmed H. Qureshi `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于神经网络的Hamilton‑Jacobi Reachability（HJR）学习框架 NeHMO，用于安全的多臂运动规划；

**💡 创新点**

通过引入物理先验（对称性与边界残差）显著提升 HJR 学习的样本效率和精度，并将学习到的安全价值函数嵌入分布式轨迹优化，实现实时多臂碰撞避免；

**🔧 技术方法**

深度学习（DeepReach 改进版）、Hamilton‑Jacobi 方程、对称性约束、边界残差学习、分布式轨迹优化、数值求解器（IPOPT）等；

**📊 数据集**

采用仿真生成的多臂起始/目标配置数据，主要在 Air3D、2‑连臂、UR5、Kinova Gen3 等模型上进行训练与评估；

**📈 对比分析**

与 CHOMP、Ha 等强化学习基方法、DeepReach 及 Naive Planner 对比；NeHMO 在双 UR5、三 UR5、五 UR5 任务中实现 92%–86% 成功率、低碰撞率（≤5%）、平均规划时长 30–600 ms，显著优于基线；在异构 UR5‑Kinova 任务中即使对手鲁莽也能将碰撞率从 100% 降至 6%；

**⚠️ 局限性**

缺乏严格的安全保证（神经网络估计），保守的 HJI 设计导致路径过长或未能收敛，且对 t_plan、t_safe 等超参数敏感，未来需加入后训练认证、通用-和游戏模型与死锁解决策略。

---

## 153. RetailSMV: Exocentric vs. Egocentric Adaptation of Foundation Video World Models in Retail

**arXiv ID:** 2607.00310 | [PDF](https://arxiv.org/pdf/2607.00310v1)

**作者:** Amirreza Rouhi `[一作]`, Sashi P. Reddi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究针对零售场景对Cosmos3‑Nano视频世界模型进行低秩适配（LoRA），以提升其在零售环境下的生成质量。

**💡 创新点**

创新点在于同步采集的自摄与第三人视角视频对比实验，揭示仅使用外景（exo）数据的LoRA适配可匹敌或优于结合训练，并提出基于配对统计的评估协议。

**🔧 技术方法**

采用Low‑Rank Adaptation（LoRA）与rectified‑flow训练目标，对Cosmos3‑Nano模型进行参数高效微调。

**📊 数据集**

使用自行构建的RetailSMV（Retail Synchronized Multi‑View）数据集，共32,105段同步自摄与第三人视角的零售操作视频及对应文本说明。

**📈 对比分析**

通过与原始模型、ego‑only、exo‑only、combined四种配置的配对统计评估（LPIPS、PSNR、SSIM、CLIPScore、DreamSim、R3D‑Fréchet、JEDi），exo‑only在大多数指标上显著优于combined，验证损失降低约2.8倍，显示LoRA在零售场景的有效性。

**⚠️ 局限性**

局限包括仅评估Cosmos3‑Nano模型，未进行人类主观评测，未支持动作条件化，仅在单一模型上验证，且使用的Fréchet等指标不完全可与其他模型的绝对数值直接比较。

---

## 154. SFDATrack: Generalized Source-Free Domain Adaptive Tracking Under Adverse Weather Conditions

**arXiv ID:** 2607.00369 | [PDF](https://arxiv.org/pdf/2607.00369v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 155. An LLM-Based Framework for Intent-Driven Network Topology Design

**arXiv ID:** 2607.00292 | [PDF](https://arxiv.org/pdf/2607.00292v1)

**作者:** Kholoud El-Habbouli `[一作]` (University of Avignon), Stephane Huet `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

利用大语言模型生成满足结构与弹性约束的网络拓扑，并构建了可验证的生成流程

**💡 创新点**

提出了ResiNet-LLM框架：分层状态化生成、结构化推理、基于Pydantic的错误反馈以及多维度评价指标

**🔧 技术方法**

使用多模大型语言模型（Gemini、GPT‑4o、Mistral‑Small‑24B、Qwen3‑32B、DeepSeek‑R1‑32B）+ LangChain、Pydantic、层级接口建模

**📊 数据集**

公开发布的四个真实场景数据集（共四个场景，每个场景含完整意图与参考拓扑）

**📈 对比分析**

通过节点/边F1、服务器/内容连通性（SC/CC）等多维指标进行评估；在中等规模场景下，专有模型实现100%准确，开源模型在结构正确性上仍有误差；大型场景下性能明显下降

**⚠️ 局限性**

局限性包括：对大规模拓扑的可扩展性不足；依赖精细的提示工程；缺乏大规模标注训练数据，导致结构错误和冗余链路问题

---

## 156. Radial Interaction Tomography: Recognizing Non-Transitive Evolutionary Games from One Range-Expansion Image

**arXiv ID:** 2607.00378 | [PDF](https://arxiv.org/pdf/2607.00378v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Baris Basaran `[通讯]` (Bahcesehir University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

从单张微生物扩张终点图像逆推出每对菌种边界的流向，并检验是否存在非可梯度的循环竞争信号。

**💡 创新点**

提出径向交互断层成像（Radial Interaction Tomography）框架，给出可观测性与稳定性定理；阐明边界设计最小数；将循环残差转化为主动设计控制目标。

**🔧 技术方法**

基于经典图像处理的固定色彩分割、对数极坐标映射、边界曲线提取与平滑、Savitzky–Golay求导、加权分解与高斯循环性检验；数值仿真、逆向优化、HMC频谱反演。

**📊 数据集**

合成的解析游戏图像、加噪/模糊像素级回环图像、四型冻结前沿模拟、公开的Weinstein等人合成图像；无学习冻结前沿模拟器、单色/多分辨率终点图像。

**📈 对比分析**

与零循环（可梯度）假设进行精确高斯似然比检验；在噪声、漂移等极端干扰下估计误检率；在机械冻结前沿实验中恢复循环残差峰值0.358；在主动控制中将残差转化为反应扩散参数，实现不良种群状态降低99%以上、目标有益种群保留>160%。

**⚠️ 局限性**

仅能在已知中心、可见边界、径向顺序近似成立的条件下工作；对拓扑变动、过度生长、强手性、前沿各向异性、后期重排、边界消亡等情形处于放弃/保留模式。

---

## 157. OnPoint: Offline-to-Online Multi-Level Distillation for Point-Supervised Online Temporal Action Localization

**arXiv ID:** 2607.00289 | [PDF](https://arxiv.org/pdf/2607.00289v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 158. Learning to Compose: Revisiting Proxy Task Design for Zero-Shot Composed Image Retrieval

**arXiv ID:** 2607.00374 | [PDF](https://arxiv.org/pdf/2607.00374v1)

**作者:** Jingjing Zhang `[一作]` (University of Science and Technology of China), Zhendong Mao `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种“先聚焦后完成”(FoCo)的零样本合成图像检索框架，利用文本引导的视觉聚合与上下文条件的语义完成两阶段组合；

**💡 创新点**

创新点在于将组合过程建模为可学习的两阶段流程，配合文本分解生成多视角局部与上下文注释，并通过跨实例对比学习消除文本短路；

**🔧 技术方法**

核心技术包括基于CLIP的冻结视觉/文本编码器、文本引导交叉注意力聚合、Transformer完成模块以及Sigmoid多正样本对比损失；

**📊 数据集**

主要使用大型无标注图文数据集Conceptual Captions 3M，结合LLM（Llama‑3.1‑8B‑Instruct）生成局部与上下文标题；

**📈 对比分析**

在四大零样本CIR基准（FashionIQ、CIRR、CIRCO、GeneCIS）上均显著优于现有方法，尤其在CIRR和CIRCO的召回/mAP指标上提升约5–10分；

**⚠️ 局限性**

局限在于对LLM生成的分解质量高度依赖，过度分解会导致信息噪声；此外仅在静态图像上验证，缺乏对动态或多模态检索的扩展。

---

## 159. CORGI: Consistency-Aware 3D Dog Reconstruction from a Single Image in the Wild

**arXiv ID:** 2607.00321 | [PDF](https://arxiv.org/pdf/2607.00321v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 160. ReShift: Aha-Moment-Driven Reasoning-Level Backdoor Attacks on Vision-Language Models

**arXiv ID:** 2607.00361 | [PDF](https://arxiv.org/pdf/2607.00361v1)

**作者:** Zhihao Dou `[一作]` (Case Western Reserve University), Sumon Biswas `[通讯]` (Case Western Reserve University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了ReShift，一种基于视觉语言模型内部推理轨迹的后门攻击方法，能够在触发时通过改变链式推理（CoT）走向而非直接改写最终答案。

**💡 创新点**

创新点包括：①首次将“aha时刻”视为可攻击的推理节点；②通过PRDC构造毒化推理数据并在SFT+RL的SRJO框架下联合优化；③引入熵反弹（Entropy Rebound）奖励以精细控制推理转移；④在保持推理连贯性的前提下实现高效的后门激活。

**🔧 技术方法**

主要技术手段包括：Poisoned Reasoning‑Aware Data Construction（PRDC）、Supervised–Reinforcement Joint Optimization（SRJO）、熵反弹奖励、Group Relative Policy Optimization（GRPO）以及链式推理（CoT）与强化学习的组合。

**📊 数据集**

训练与评估使用的公开数据集有：A‑OKVQA、ScienceQA；外域评估则使用MathVista、MMMU等多模态任务数据集。

**📈 对比分析**

与现有输出级后门攻击（BadToken、BadVision、Rewrite）相比，ReShift在攻击成功率（ASR）上更高，同时保持了与基线相当的推理连贯度（Coh）和合理性（Rat）得分，干净样本准确率损失很小；在两种主流检测器（BYE、BkdAttr）下的检测准确率接近随机，证明其高度隐蔽性。

**⚠️ 局限性**

局限性：①对训练过程和算力要求较高；②需要对模型训练过程完全可控，难以在黑盒或预训练模型上直接使用；③触发器设计和“aha时刻”定位仍需手工调优；④在极端多模态任务或低样本场景下的效果尚未充分验证。

---

## 161. Managed Autonomy at Runtime: Gear-Based Safety and Governance for Single- and Multi-Agent Cyber-Physical Systems

**arXiv ID:** 2607.00334 | [PDF](https://arxiv.org/pdf/2607.00334v1)

**作者:** Srini Ramaswamy `[一作]` (DNRS.ai), Wang Miaosheng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于齿轮的运行时安全控制层（Gear-based runtime），通过五个动作级别的执行齿轮、utility 门和事件回退机制，实现单机与多机自主系统的稳定性、安全性和连续性，并在三台 UR5 机器人装配单元中进行验证。

**💡 创新点**

创新点在于将动作权限抽象为可调节的齿轮层级，结合多态治理状态与共识门实现分层安全保证，并将运行时映射为马尔可夫决策过程（MDP），从而提供可证明的 Lyapunov 安全证书和正式的工作空间稳定性分析。

**🔧 技术方法**

采用离散时间控制系统、utility gating、事件回退、Lyapunov 稳定性分析、共识 utility 门、Rendezvous policy 三元组、基于 Ornstein–Uhlenbeck 过程的传感器漂移建模，以及 Monte‑Carlo 仿真等技术。

**📊 数据集**

使用公开的 NIST Degradation Measurement of Robot Arm Position Accuracy 数据集来校准 UR5 机器人传感器漂移参数。

**📈 对比分析**

通过与单机基础门限（仅使用单体 utility gate）在 10,000 次 Monte‑Carlo 试验中比较，评估指标包括异常检测率、检测延迟、系统收敛率、审核轨迹生成率和 Lyapunov 证书完备性；结果显示异常检测率提升 47.7×、检测延迟缩短 3.5×、系统收敛率为 90.2%（相较于 100% 但无检测），并成功生成 89.9% 的审核轨迹和正式安全证明。

**⚠️ 局限性**

局限性包括需要人工指定 utility 函数与阈值、模型假设传感器漂移服从 Ornstein–Uhlenbeck 过程且系统可分层、对非稳态环境、强反馈耦合系统以及大规模多机团队的可扩展性尚未验证，以及缺乏自动学习 utility 的机制。

---

## 162. A Text-Steerable Instrument for Sketching Procedural Soundscapes via Language Models

**arXiv ID:** 2607.00309 | [PDF](https://arxiv.org/pdf/2607.00309v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 163. Typography-Based Monocular Distance Estimation for Advanced Driver-Assistance Systems

**arXiv ID:** 2607.00319 | [PDF](https://arxiv.org/pdf/2607.00319v1)

**作者:** Manognya Lokesh Reddy `[一作]` (University of Michigan-Dearborn), Zheng Liu `[通讯]` (University of Michigan-Dearborn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用车牌的标准化几何尺寸（外框、字符高度和装配孔间距）通过单摄像头测距，提出Typography-Based Monocular Distance Estimation (T‑MDE)，实现无发射器、低成本的绝对距离估计。

**💡 创新点**

创新点包括：①三种独立的排版测距通道（字符高度、笔画宽度、字符间距）与装配孔距离和单图深度网络的逆方差加权融合；②按州识别字符高度以消除尺度误差；③对车牌姿态进行校正并输出方位信息；④使用距离相关的Kalman滤波生成闭合速度、时间‑碰撞预测及安全预警。

**🔧 技术方法**

技术手段包括：相机投影模型与单应变换、基于 OCR 的车牌识别与字符分割、颜色/文字/深度网络的状态识别、逆方差权重融合、姿态求解（PnP）、单图深度网络与尺度校正、距离相关的Kalman滤波和预警阈值。

**📊 数据集**

数据集：在58次行驶与停放试验中采集的实时车牌图像，覆盖15个美国州，共2265帧车牌测距；使用标准车牌尺寸和字符高度表，未使用公开车牌数据集，仅使用实测车辆场景。

**📈 对比分析**

性能：检测率≈99%，字符识别率≈63%，平均距离误差模型为2.3% CV，单帧中值不确定度≈0.13 m；融合后与姿态测距一致率≈1%；滤波后时间‑碰撞预测准确；相较于雷达、激光、立体视觉，T‑MDE在成本、功耗和尺寸上更优，但距离覆盖相对有限。

**⚠️ 局限性**

局限性：缺乏外部测距参考，误差验证基于内部一致性；距离覆盖主要集中在1–2 m，远距离样本稀缺；需可读、标准前车牌，摩托车、破损或非美式车牌不适用；高偏航角下识别与测距性能下降；未完成实时嵌入式实现与车辆级融合。

---

## 164. Feasibilism, Explication, and the Cobham-Edmonds Thesis

**arXiv ID:** 2607.00315 | [PDF](https://arxiv.org/pdf/2607.00315v1)

**作者:** Abrahim Ladha `[一作]` (Georgia Institute of Technology), Alan Tian `[通讯]` (Carnegie Mellon University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文对Cobham-Edmonds理论进行了详细分析，提出了支持该理论的论据，探讨了可行计算的哲学和历史背景。

**💡 创新点**

创新点在于对Cobham-Edmonds理论的论证进行了系统的哲学分析，强调了可行计算的复杂性与可计算性之间的关系。

**🔧 技术方法**

使用了计算复杂性理论的相关技术，特别是多项式时间计算的概念。

**📊 数据集**

未提及具体数据集，主要是理论分析。

**📈 对比分析**

与其他方法的比较主要在于对可行计算的定义和理解，性能上强调了多项式时间计算的稳定性和可行性。

**⚠️ 局限性**

限制在于未能明确证明多项式时间计算是可行计算的唯一标准，且对可行计算的定义仍存在模糊性。

---

## 165. Generative Modeling of Quantum Distribution with Functional Flow Matching

**arXiv ID:** 2607.00301 | [PDF](https://arxiv.org/pdf/2607.00301v1)

**作者:** Jaehoon Hahm `[一作]` (Yonsei University), Daniel K. Park `[通讯]` (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 Quantum Flow Matching (QFM) 模型，通过将密度矩阵转换为 spin Wigner 函数，并在函数空间中使用功能流匹配学习量子分布；

**💡 创新点**

创新点在于：①用 spin Wigner 函数构建信息完备的经典表示，规避了直接学习复值密度矩阵的符号结构难题；②将功能流匹配应用于量子分布，利用其分辨率不变特性实现高精度的量子态重构；

**🔧 技术方法**

采用的技术包括：spin Wigner 函数、功能流匹配 (Functional Flow Matching, FFM)、Fourier Neural Operator、条件流匹配 (Conditional Flow Matching, CFM) 等；

**📊 数据集**

使用的数据集为：①7000 个合成的单量子比特纯态与混态（纯度分别为 1、0.625、0.905）；②900 个 2 量子比特状态，均具有固定纠缠熵 S=0.7；

**📈 对比分析**

与直接对密度矩阵进行流匹配（FM）进行对比，评估指标包括 trace、purity、纠缠熵等物理量。实验显示 QFM 在保持 trace≈1、纯度和纠缠熵等指标上显著优于 FM，数值更贴近目标；

**⚠️ 局限性**

局限性：仅在单量子比特和 2 量子比特系统上验证；对更大多体系统、真正多体纠缠态及基态的生成尚未测试；计算成本随量子比特数增长仍是挑战。

---

## 166. Attribute-Prompted Kernel Hashing for Unsupervised Data-Efficient Cross-Modal Retrieval

**arXiv ID:** 2607.00379 | [PDF](https://arxiv.org/pdf/2607.00379v1)

**作者:** Runhao Li `[一作]` (Nanyang Technological University), Yap-Peng Tan `[通讯]` (Nanyang Technological University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Attribute-Prompted Kernel Hashing (APKH)，一种在数据稀缺环境下实现无监督跨模态检索的高效哈希框架，解决传统方法对大量图文对的依赖和对未知类别的泛化不足。

**💡 创新点**

创新点包括：① 利用 CLIP 的通用属性先验与 prompt 学习构造可调节的 RBF 核映射（CAKM），实现跨模态的自适应对齐；② 采用 Kernel‑Smoothed Contrastive Alignment (KSCA)，将有限的图文配对建模为连续核分布，平滑模态差距并抑制过拟合。

**🔧 技术方法**

使用技术包括：预训练的 CLIP 视觉‑语言模型、连续软 prompt、超球面 RBF 核映射、统一的单一哈希 MLP、核密度估计与对比损失、二值化损失以及 t‑SNE 可视化等。

**📊 数据集**

实验数据集涵盖 MIR Flickr、NUS‑WIDE、Pascal Sentence、Wikipedia 四个主流跨模态检索基准。

**📈 对比分析**

与 CIRH、CAGAN、UCCH、CDTH、DUHEG、PIC‑CMH、GNAH 等现有无监督跨模态哈希方法在 40/80/160 对训练样本、16/32/64/128 位哈希码下的 mAP 进行对比。APKH 在见到与未见类别上均显著优于基线，尤其在 unseen 上提升明显；在跨数据集迁移实验中亦保持最佳性能。

**⚠️ 局限性**

局限性包括：对 CLIP 预训练模型及其属性词库（620 个）高度依赖；对属性 prompt 的学习需要额外训练；在极少样本（如 20 对）下仍受限；实验主要集中在公开基准数据集，未充分验证在极端私有或噪声数据场景中的鲁棒性。

---

## 167. MEPA: Multi-Scale Representation Alignment for Visual Autoregressive Modeling with Mixture of Experts

**arXiv ID:** 2607.00371 | [PDF](https://arxiv.org/pdf/2607.00371v1)

**作者:** Nuoyan Zhou `[一作]` (Xidian University), Xinghao Chen `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种多尺度表示对齐框架MEPA，用于提升视觉自回归模型(VAR)的多尺度特征学习与语义一致性。

**💡 创新点**

创新点在于①设计了尺度感知的Token路由混合专家（STMoE），实现跨尺度的专家分配与负载平衡；②引入残差聚合与语义引导，在VAR早期阶段对齐外部自监督特征，抑制语义误差传播。

**🔧 技术方法**

采用Mixture of Experts（MoE）与token级路由、残差聚合策略、语义引导（自监督编码器对齐）以及标准VAR训练框架。

**📊 数据集**

主要使用ImageNet-1K 256×256图像数据集进行条件生成实验。

**📈 对比分析**

与GAN、扩散模型、以及现有VAR、FlexVAR、SpectralAR等基线相比，MEPA在相同参数量下实现了2×的训练速度提升，并在默认训练周期下以约2.32 FID领先于VAR-d16，且在仅100 epoch时已优于大多数VAR模型。

**⚠️ 局限性**

受限于计算资源，实验仅在256×256分辨率下完成，未完成512×512的完整训练；此外，语义对齐方案依赖外部自监督模型，且对不同尺度的聚合策略仍需进一步优化。

---

## 168. DroneFINE: Domain-Aware Parameter-Efficient Fine-Tuning of Vision-Language Detectors for Drone Images

**arXiv ID:** 2607.00338 | [PDF](https://arxiv.org/pdf/2607.00338v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 169. Performance Evaluation of A Certain Transceiver Architecture for Multiple-Input Multiple-Output Phase-Modulated Channels

**arXiv ID:** 2607.00356 | [PDF](https://arxiv.org/pdf/2607.00356v1)

**作者:** Hengyu Cui `[一作]` (Nanjing University of Science and Technology), Yeqin Tai `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了一种基于组合配对（CP）分解的MIMO相位调制链路变换器架构的性能，并给出了其每个子通道在环形输入约束和非高斯自干扰下的容量上下界。

**💡 创新点**

创新点在于：①将MIMO相位调制通道变换为多子通道，且每个子通道都有两个相位输入，从而近乎实现完全自由度；②从凸几何和极坐标分解两种角度推导非高斯噪声通道的容量上界；③利用熵功率不等式给出可计算的下界，且上下界在高信噪比下逼近。

**🔧 技术方法**

使用的技术包括：组合配对算法（CP）进行行阶梯化、卷积几何与 Minkowski 求积的体积上界、极坐标分解的径向与相位熵拆分、熵功率不等式（EPI）以及Monte‑Carlo 估计非高斯噪声方差。

**📊 数据集**

所用数据集为：1）随机复高斯（Rayleigh）MIMO通道矩阵；2）单天线 RIS 辅助系统的仿真通道，包含 LOS 和 RIS‑反射路径；3）用于验证上界的人工构造的圆环输入分布，噪声为 AWGN 或经过 CP 产生的自干扰噪声。

**📈 对比分析**

方法比较：在标量 AWGN 圆环通道、MIMO 相位调制通道以及 RIS 辅助共生通信系统中，数值结果显示两条上界（凸几何上界和极坐标上界）与 EPI 下界在高 SNR 时相差不足 1–2 dB，整体吞吐量随 SNR 上升后趋于稳定；在 RIS 场景中，较高的空间相关性使上界更优，且剩余干扰更小，吞吐量能达到更高的多路复用增益。

**⚠️ 局限性**

局限性：① CP 产生的残余自干扰在高 SNR 时限制了吞吐量的进一步提升；② 方法假设输入满足严格的环形支持，实际系统可能需要更通用的输入集；③ 上界的解析形式较为复杂，计算时需数值优化；④ 结果主要针对快速/慢速 Rayleigh 衰落与 RIS 反射模型，其他通道模型需进一步验证。

---

## 170. TRACE: State-Aware Query Processing over Temporal Evidence Graphs for Conversational Data

**arXiv ID:** 2607.00339 | [PDF](https://arxiv.org/pdf/2607.00339v1)

**作者:** Maolin Wang `[一作]` (City University of Hong Kong), Hao Miao `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TRACE框架，用于在含时序演变的对话历史上执行状态感知的查询处理，构建了事件–会话–主题的层次化时序证据图，并结合向量检索与图路径搜索生成答案。

**💡 创新点**

创新点包括：① 将对话拆解成可解释的事件节点并在图中显式标注支持、更新、矛盾等关系；② 通过有效性标注（valid‑until、validity score）实现对已被更新的事实进行灰度剔除；③ 采用更新感知的种子扩展与受限双向路径搜索，实现跨会话多跳推理；④ 依据查询类型动态选择答案计划，提升不同问题类型的回答质量。

**🔧 技术方法**

技术手段：稠密向量检索（Sentence‑Encoder），图构造与更新感知的关系推断（LLM分类），有效性传播与路径评分（多信号融合），双向受限路径生成，混合上下文组装（检索文本 + 证据链），LLM回答生成与后置证据追溯。

**📊 数据集**

使用的数据集包括：LoCoMo（10 轮对话，24K tokens/轮），LongMemEvalS（500 轮对话，115K tokens/轮）以及为评估更新感知能力专门构造的 Stress Test（59 题）。

**📈 对比分析**

与七种基线（Full‑Context、RAG、LangMem、Zep、Mem0、Nemori、A‑Mem）在两种LLM后端（GPT‑4o‑mini、DeepSeek‑V4‑Flash）上进行对比，TRACE在大多数问题类别（尤其是时序推理、多跳、偏好）上取得显著提升：LoCoMo 上整体 LLM‑score +12.8 pp、LongMemEvalS 上 +13.8 pp；Stress Test 上 valid‑accuracy +9.4 pp、outdated‑leakage 与 Full‑Context 一致，且仅使用约5×少的上下文 token。

**⚠️ 局限性**

局限性：① 时序关系推断依赖 LLM，时钟/时间标注不精确导致边标签错误；② 图构造成本高（数小时、LLM 调用），不适合实时增量更新；③ 受限路径搜索在极长对话中可能错过分布式证据；④ 对低频事件/少量对话的优势不显著，需进一步优化跨会话链接策略。

---

## 171. Mapping the Evaluation Frontier: An Empirical Survey of the Bias-Reliability Tradeoff Across Eleven Evaluator-Agent Conditions

**arXiv ID:** 2607.00304 | [PDF](https://arxiv.org/pdf/2607.00304v1)

**作者:** Zewen Liu `[一作]` `[通讯]`, Zewen Liu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对LLM评估者的偏差-可靠性权衡进行11个条件的实证研究，并测量了评估器耦合度γ、策略熵H和小样本可靠性CV。

**💡 创新点**

将原始样本从5个扩展到11个条件，验证了三维权衡空间，首次发布了标准化的评估基准数据集。

**🔧 技术方法**

使用测试时强化学习、策略权重向量、熵计算、Bootstrap估计等方法来计算γ、H和CV。

**📊 数据集**

采用单一研究组公开的多实验数据集，包含16项任务、11种候选策略以及30个种子实验。

**📈 对比分析**

通过绘制γ–H–CV三元组的Pareto前沿，比较不同评估器-代理组合的偏差与可靠性；结果显示高γ对应低CV，低γ对应高CV；GPT‑4o版本漂移导致γ=0、H=1的异常点。

**⚠️ 局限性**

局限性包括样本仅来自单一实验室、完整指标的条件仅5个、GPT‑4o版本漂移导致指标失效、缺乏中间γ值的充分样本。

---

## 172. Wake up for Touch! Mask-isolated Tactile Alignment Learning in MLLMs

**arXiv ID:** 2607.00302 | [PDF](https://arxiv.org/pdf/2607.00302v1)

**作者:** Yoonhyung Park `[一作]` (Ewha Womans University), Jiyoung Lee `[通讯]` (Ewha Womans University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文提出一种 mask-isolated tactile alignment 学习框架，能够在不增加推理成本的情况下将触觉感知融入小规模多模态 LLM 中。

**💡 创新点**

创新点在于通过视觉相对重要性指标划分参数空间，将更新限定在低重要性（dormant）子空间，实现新模态学习与原有视觉语言能力的非破坏性共存。

**🔧 技术方法**

采用参数重要性度量、二值掩码、统一单阶段训练，冻结关键参数、更新 dormant 子空间与触觉编码器，同时使用 ViT-ImageNet 预训练触觉编码器和 LLM。

**📊 数据集**

使用三大视触觉语言基准 SSVTP、TVL、TacQuad 进行触觉描述评估，并在 MMMU、MathVista、MME、MMBench 等通用视觉语言基准测试保留能力。

**📈 对比分析**

在所有触觉基准上，Mask‑MLLM 在 3B 模型上平均得分 4.91，优于同基线 LoRA 的 4.50，且在 3B 规模下仍保持与预训练模型相当的视觉语言性能；在较小 1B 模型上亦能达成 5.01 分。

**⚠️ 局限性**

局限性在于冻结的关键子空间是静态的，无法适应不同任务或交互状态下动态激活模式，且目前仅验证单一 DIGIT 触觉传感器，跨传感器推广尚未评估。

---

## 173. EPC: A Standardized Protocol for Measuring Evaluator Preference Dynamics in LLM Agent Systems

**arXiv ID:** 2607.00297 | [PDF](https://arxiv.org/pdf/2607.00297v1)

**作者:** Zewen Liu `[一作]` `[通讯]`, Zewen Liu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Evaluator Preference Coupling（EPC）标准化协议，并发布第一版参考快照，用于测量LLM代理在闭环评估中的偏好耦合。

**💡 创新点**

创新点在于建立完整的四阶段隔离实验框架、规范化的TTRL更新规则、γ、JSD等指标以及统一的版本化命名约定，使跨模型、跨时间的耦合测量可复现、可比较。

**🔧 技术方法**

采用API驱动的执行者与评估者、L1归一化策略权重、固定学习率的TTRL、基于Jensen–Shannon散度的距离测度以及可选的ECE/Brier评估。

**📊 数据集**

使用了8个文本域任务与8个视觉相关任务（详见附录）以及11个策略提示，实验基于GPT‑4o、Qwen、DeepSeek等多模型。

**📈 对比分析**

通过与参考快照v1.0的γ、JSD等指标对照，并采用95%自助置信区间评估稳定性；快照中不同评估者的γ值从0.03到1.18不等，表明评估者版本差异导致耦合显著变化。

**⚠️ 局限性**

局限性包括固定学习率与基准策略的结构偏差、文本驱动的视觉任务局限以及测量随评估者模型更新快速衰退，需要持续更新快照。

---

## 174. Learning When to Listen: Gated Affect Fusion for Human Motion Prediction

**arXiv ID:** 2607.00296 | [PDF](https://arxiv.org/pdf/2607.00296v1)

**作者:** Jingni Huang `[一作]` `[通讯]` (University of Oxford), Jingni Huang (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了面部情绪对无约束视频中人体运动预测的影响，并提出了带门控的情绪变换器来融合运动和情绪信息。

**💡 创新点**

引入了可学习的门控机制动态调节情绪特征对预测的影响，从而在短期预测窗口提升精度而不损害长期预测。

**🔧 技术方法**

使用 MediaPipe Pose 与 HSEmotion 情绪向量，构建双流 Transformer，门控融合后进行未来姿态回归。

**📊 数据集**

在从公开 YouTube 视频（TED、访谈、Vlog）抽取的多主体约 243 条视频的 2D 关键点与情绪序列上。

**📈 对比分析**

与仅使用姿态的 Transformer 及早期拼接 Transformer 对比，短期（30 帧）能略优，长期（60/90 帧）仍落后。

**⚠️ 局限性**

受限于仅使用 2D 关节、情绪提取器误差、视频噪声以及缺乏 3D/语音等多模态，导致长时序预测仍依赖运动本身。

---

## 175. Vitality-Aware Compression for Efficient Image-to-Shape Diffusion Transformers

**arXiv ID:** 2607.00382 | [PDF](https://arxiv.org/pdf/2607.00382v1)

**作者:** Jaeah Lee `[一作]` (KRAFTON AI), Gihyun Kwon `[通讯]` (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了基于层重要性（Vitality）的压缩框架，结合结构化剪枝、适应量化和精细化微调，显著压缩图像到三维形状扩散变压器（DiT）的模型规模，同时保持几何保真度。

**💡 创新点**

首先提出使用Earth Mover’s Distance衡量每层对3D生成质量的贡献，构造Vitality指标；其次基于该指标实现双阈值剪枝与层级量化；最后采用只更新最低Vitality层的精细化微调，形成端到端可插拔的压缩方案。

**🔧 技术方法**

结构化剪枝、层级自适应量化（8‑bit/4‑bit）、基于完整模型的压缩后微调（distillation）、EMD层重要性评估、流水线式DiT压缩与后续加速蒸馏。

**📊 数据集**

主要在Objaverse图像/文本数据集及DALL·E 3生成的图像样本上进行训练与评估，并在Objaverse子集上做微调。

**📈 对比分析**

与Splatter Image、TripoSR、LGM、Craftsman3D、TRELLIS、TinyFusion、Diff‑Pruning等基线以及原始大模型对照，使用Uni3D‑I、OpenShape‑I、V‑IoU/S‑IoU、TFLOPs、生成延迟和峰值VRAM等指标；实验显示压缩后模型尺寸降低50‑66%，生成质量几乎与全模型相同，推理速度与显存占用均大幅下降。

**⚠️ 局限性**

该方法主要针对DiT结构，对其他3D生成模型的适用性仍需验证；压缩后仍需微调，训练成本不低；在极小化压缩比例时仍可能出现细节损失；在动态或多模态任务中的通用性尚未完全评估。

---

## 176. Registry-Governed Agent Lifecycle:Completing EDDOps with Evaluation-DrivenRegistration, Promotion, and Retirement on AWS AgentCore

**arXiv ID:** 2607.00345 | [PDF](https://arxiv.org/pdf/2607.00345v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 177. SoK: Attack and Defense Landscape of Mobile On-device AI Systems

**arXiv ID:** 2607.00362 | [PDF](https://arxiv.org/pdf/2607.00362v1)

**作者:** Yujin Huang `[一作]` (University of Melbourne), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统地梳理了移动端本地 AI（MOAI）系统的安全威胁与防御，提出了三大安全支柱、攻击与防御分类，并指出未来研究的关键开放问题。

**💡 创新点**

创新点在于首次从整体系统视角统一定义 MOAI 的安全支柱、构建攻击/防御的多维分类框架，并将攻击与防御与安全支柱的对应关系进行交叉分析。

**🔧 技术方法**

主要采用文献综述与安全体系结构分析技术，基于威胁模型和安全三要素（Confidentiality, Integrity, Availability）进行分类。

**📊 数据集**

本文不涉及实验数据集，主要参考公开的安全研究论文与公开工具。

**📈 对比分析**

由于是综述工作，未进行方法对比实验；但作者指出各防御方案在隐私、性能、部署可行性等维度存在权衡。

**⚠️ 局限性**

局限性包括：缺乏统一评估指标与实验验证；对未来生成式 AI 与代理系统的安全挑战仍未展开深入实验；并且所述防御在不同设备与框架上的通用性待验证。

---

## 178. PRISM: Prioritized Channel Importance with Semi-supervised Domain Adaptation for Cross-Subject EEG Emotion Recognition

**arXiv ID:** 2607.00358 | [PDF](https://arxiv.org/pdf/2607.00358v1)

**作者:** Xin Zhou `[一作]` (Binghamton University State University Of New York), Lijun Yin `[通讯]` (Binghamton University State University Of New York)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 PRISM 框架，结合通道优先化与半监督域适应，实现跨受试者 EEG 情绪识别。

**💡 创新点**

创新点在于：① 使用轻量化专家路由器实现动态通道权重；② 引入逆嵌入 + 状态空间模型捕获多尺度时空特征；③ 通过高置信度伪标签、一致性正则化和分布对齐实现标签稀缺下的跨域泛化。

**🔧 技术方法**

采用的技术包括：多尺度季节性挖掘、逆嵌入、Mamba 状态空间块、Mixture of Experts 路由器、置信度阈值伪标记、强弱增强一致性正则化、熵最小化与分布对齐。

**📊 数据集**

使用的数据集：DEAP、DREAMER、SEED。

**📈 对比分析**

与六种时间序列基线（iTransformer、DLinear、TimesNet、NTransformer、Informer、TCN）以及多种跨域/半监督方法对比，PRISM 在交叉受试者、跨域与受试者相关设置下均取得最高或接近最高的准确率，尤其在标签稀缺时明显优于其他方法。

**⚠️ 局限性**

局限性：① 在受试者相关（单体）设置下表现不如跨域优势明显；② 对极少通道或极低标签比例时可能需要额外调参；③ 需要对专家数、top‑k 选择等超参数进行调优；④ 对超大规模 EEG 数据的计算开销仍需进一步优化。

---

## 179. Personalized Object Identification and Localization via In-Context Inference with Vision-Language Models

**arXiv ID:** 2607.00357 | [PDF](https://arxiv.org/pdf/2607.00357v1)

**作者:** Kensuke Nakamura `[一作]` (Chung-Ang University), Byung-Woo Hong `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了个性化目标识别与定位（POIL）任务，并在此基础上设计了一种基于视觉语言模型的自回归推理框架IPLoc-ID；

**💡 创新点**

创新点在于：①将传统仅做定位的IPLoc扩展为既能定位又能对否定查询做拒绝；②通过自提问（self‑posed query）将候选框生成与识别判断串联为单一文本序列，提升实例级识别能力；③构建四个公共数据集的正负样本，形成POIL评估基线；

**🔧 技术方法**

技术手段包括：视觉语言模型（Qwen3‑VL、LLaVA1.5‑7B等）+ LoRA微调；自回归文本生成产生 BBOX + 询问 + Yes/No；解码器解释器将文本转换为结构化结果；统一损失函数同时优化定位与识别；

**📊 数据集**

使用了 LaSOT、PDM/Burst、GOT‑10K、VastTrack 四个公开视频追踪数据集，经过正负样本采样构造 POIL 训练/测试集；

**📈 对比分析**

与传统 OD、FSOD、原始 IPLoc、以及多种 VLM 推理基线进行对比。IPLoc‑ID 在 mIoU 与 F1‑score 上均显著优于对照组，尤其在负样本上的 F1‑score 接近 1，验证了其在实例级拒绝能力；

**⚠️ 局限性**

局限性包括：只能处理单个目标；仅在单帧图像上推理，未利用视频时序信息；对多目标或动态跟踪的扩展仍待研究。

---

## 180. (A)I Sees What You Don't: Exploiting New Attack Surfaces in Third-Party Mobile Agents

**arXiv ID:** 2607.00333 | [PDF](https://arxiv.org/pdf/2607.00333v1)

**作者:** Zidong Zhang `[一作]` (Simon Fraser University), Jianliang Wu `[通讯]` (Simon Fraser University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对第三方移动智能代理（基于Vision‑Language Model）的安全性，系统识别并构造了两类攻击面——屏幕感知攻击面与被滥用的通信通道攻击面，并实现了七种具体攻击；

**💡 创新点**

创新点在于提出了移动代理特有的攻击面概念，并首次演示了低对比度视觉注入、不可见角落注入、UI伪装、截图篡改、广播窃听、凭证窃取以及主机端命令注入等新型攻击；

**🔧 技术方法**

使用了Android窗口叠加、ADB命令、可访问性服务、文件锁竞争、图像处理与Steganography、VLM文本解析及Shell命令注入等技术；

**📊 数据集**

实验基于五款主流开源移动代理（AppAgent、AppAgentX、Mobile‑Agent v3、Open‑AutoGLM、MobA）以及多种VLM后端（GPT‑4o、Qwen‑3‑VL、AutoGLM‑9B 等），未使用公开数据集而是人工构造任务；

**📈 对比分析**

通过在每个代理上执行20轮实验，对每种攻击给出成功率（多为100%），并对比不同代理的易受攻击程度，实验表明攻击成功率高、对代理功能影响明显；

**⚠️ 局限性**

局限性包括仅针对Android平台且无root权限、仅测试开源代理、未涵盖iOS或商用代理、假设已安装恶意应用、实验规模有限且VLM输出具有随机性。

---

## 181. Watermarking for Proprietary Dataset Protection

**arXiv ID:** 2607.00325 | [PDF](https://arxiv.org/pdf/2607.00325v1)

**作者:** John Kirchenbauer `[一作]` (University of Maryland), Tom Goldstein `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实验了一种主动式水印数据集推断方法，用来检测目标模型是否在训练时包含了某些专有数据。

**💡 创新点**

创新点在于把生成式语言模型的输出水印与随机化假设检验相结合，实现了可校准的p值检测，并在受限的模型访问条件下与传统基于损失的成员推断方法进行对比。

**🔧 技术方法**

技术包括基于生成模型的水印标记（output‑watermarking）、读取模式（reading‑mode）检测、随机化（permutation）p值检验，以及对Qwen3‑1.7B等大规模语言模型进行微调与预训练实验。

**📊 数据集**

使用了1500份语义独立、看似真实的“FictionalQA”文档集，按事件划分为不同折（fold）并与网络文本混合进行训练。

**📈 对比分析**

在细粒度的 (F, E) 训练曝光网格中，水印方法在高曝光下达到了与传统 loss‑based、参考模型基准（raw loss、argmax、min‑k%、zlib、rMIA、LiRA）相近或略低的 AUC（最高 1.0，最低约 0.55），但在低曝光或单键设置下性能显著下降；loss‑based 方法在几乎所有配置下都几乎完美。

**⚠️ 局限性**

局限包括对读取模式检测的高昂查询成本、单一水印键可能导致的假正率波动、对模型架构与标记兼容性的假设、以及在实际部署中需要筛选“安静”水印键的额外工作。

---

## 182. Fast Deterministic Normal Bases and Circulant Polynomial Determinants

**arXiv ID:** 2607.00313 | [PDF](https://arxiv.org/pdf/2607.00313v1)

**作者:** Mark Giesbrecht `[一作]` (University of Waterloo), Éric Schost `[通讯]` (University of Waterloo)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种确定性算法，用于在有限域的代数扩展中找到一个正常元素，该算法的时间复杂度接近于二次时间。

**💡 创新点**

创新点在于通过构造一个“清除的Moore”循环矩阵，利用其行列式的性质来确定正常元素，并且提出了一种快速的循环矩阵行列式计算算法。

**🔧 技术方法**

使用了循环矩阵、三角形集合投影和模组合等技术来实现算法。

**📊 数据集**

使用了有限域的代数扩展，具体数据集未明确给出，但涉及到的元素和多项式均在有限域内。

**📈 对比分析**

与现有方法相比，本文的方法在复杂度上有显著改进，能够在O_ϵ((n^2 log q)^{1+ϵ}) + (n log^2 q)位操作内完成计算，性能优越。

**⚠️ 局限性**

限制在于当有限域的大小小于n(n-1)时，算法需要通过嵌入到低度扩展域来扩展，这可能增加额外的多对数成本。

---

## 183. Rosetta: Composable Native Multimodal Pretraining

**arXiv ID:** 2607.00293 | [PDF](https://arxiv.org/pdf/2607.00293v1)

**作者:** Xiangyue Liu `[一作]` (HKUST), Ping Tan `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并验证了一种可无损扩展的多模态预训练框架Rosetta，能够在引入连续生成任务时保持先前已学习的知识不被遗忘；

**💡 创新点**

通过将可插拔专家与全局共享专家分离，并利用优化器动量的Momentum‑Anchored Orthogonal Projection (MAOP)实现零开销的梯度冲突消除；

**🔧 技术方法**

使用混合稀疏专家（MoE）、统一注意力、插件式FFN结构以及MAOP梯度投影技术；

**📊 数据集**

基于约300B文本语料、LLaVA式视觉理解样本、COCO‑30K T2I生成数据以及MMU等多模态数据集；

**📈 对比分析**

在与严格参数等价的MoE与MoT基线对比中，Rosetta在语言（MMLU/BBH/MBPP）、视觉理解（MMBench/POPE/AI2D）和视觉生成（FID/CLIPScore）等指标上均表现更优或至少不逊，并成功避免了灾难性遗忘；

**⚠️ 局限性**

目前实验仅覆盖文本与图像两模态，未来如何在更大规模、多模态（音频、视频、3D）上保持可扩展性及训练成本仍需进一步探索。

---

## 184. LIST3R: Long-sequence Instance-aware 3D Reconstruction

**arXiv ID:** 2607.00375 | [PDF](https://arxiv.org/pdf/2607.00375v1)

**作者:** Jing Gao `[一作]` (Beijing Jiaotong University), Yan Yan `[通讯]` (University of Illinois Chicago)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了 LIST3R 框架，通过实例锚点在长序列中实现局部重建的对齐与全局一致性；

**💡 创新点**

创新在于将可识别物体实例作为持久锚点，融合实例语义与几何信息以发现长程回环并提升子序列合并质量；

**🔧 技术方法**

采用基于 3D 基础模型（如 π³、VGGT）进行局部重建，结合 SAM3、实例库构建、实例增强长程回环发现、实例感知子序列合并和置信权重图优化等技术；

**📊 数据集**

在 TUM、ETH3D、BONN、NRGBD 等长序列数据集上评估；

**📈 对比分析**

与基线（VGGT、π³、CUT3R、TTT3R、VGGT‑Long、Scal3R、π‑Long）对比，LIST3R 在 ATE、RTE、RRE、Chamfer、Accuracy、Completion、Normal Consistency、F@5cm 等指标上均优于或与最强基线相当，尤其在全局轨迹一致性和点云完整性方面显著提升；

**⚠️ 局限性**

仍受限于实例检测与跟踪误差、对动态物体处理不完善以及极端遮挡或大视角变化下实例识别可靠性不足。

---

## 185. Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training

**arXiv ID:** 2607.00368 | [PDF](https://arxiv.org/pdf/2607.00368v1)

**作者:** Xiangchen Song `[一作]` (Carnegie Mellon University), Kun Zhang `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于行为的评估框架，用以校准LLM测试时训练（TTT）在部署记忆、个性化等方面的主张。

**💡 创新点**

创新点是将TTT的评估分为三层（流/域适应、桥接机制、部署时行为学习）并引入匹配的显式记忆基线，对“证据迁移”现象进行系统识别。

**🔧 技术方法**

采用了TTT中的参数/快速权重更新、LoRA微调、以及对Qwen3模型的控制实验，并设计行为测试模板。

**📊 数据集**

主要使用公开LLM模型（如Qwen3）及其自带数据进行实验，未依赖特定任务数据集，而是基于少量“nonce”事实进行诊断。

**📈 对比分析**

与显式记忆系统（MemoryBank、LongMem等）对照，发现代理式更新在同级别代理指标上提升但在自由回忆等行为指标上零提升，表明代理指标与部署行为不一致。

**⚠️ 局限性**

局限性在于框架仍以现有实验为基础，缺乏跨任务、多模型的广泛验证，且对复杂交互情境下的冲突处理仍待深入。

---

## 186. Agri-SAGE: Simulation-Grounded Multi-Agent LLM for Context-Aware Agricultural Advisory Generation

**arXiv ID:** 2607.00454 | [PDF](https://arxiv.org/pdf/2607.00454v1)

**作者:** Vedant Balasubramaniam `[一作]` (Indian Institute of Science), Y. Narahari `[通讯]` (Indian Institute of Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并评估Agri‑SAGE闭环多智能体框架，结合LLM推理与APSIM模拟生成和验证全季农业建议。

**💡 创新点**

将检索增强生成的LLM与过程式作物模拟耦合，实现物理可行性验证，并比较三种推理策略（Plan‑and‑Solve、Tree of Thoughts、Reflexion）以展示其优势。

**🔧 技术方法**

多智能体LLM架构、检索增强生成、Tree of Thoughts搜索、Reflexion经验记忆、APSIM作物模拟、工具调用接口等技术。

**📊 数据集**

Mandya地区十年ERA5气象、土壤红砂土参数、约1000篇开放获取玉米研究论文及地方推广手册，用作基准PoP方案。

**📈 对比分析**

在2015‑2024十年玉米生长周期上对比PoP、PS、ToT、Reflexion，采用10次独立运行取平均，Tree of Thoughts平均产量9,262 kg/ha（比基准+1,152 kg/ha），Plan‑and‑Solve与Reflexion约9,000 kg/ha；ToT在极端气候下表现最优，Reflexion在计算成本最低。

**⚠️ 局限性**

仅验证单一作物模型，未实地验证，目标仅为产量未考虑成本/可持续性，LLM模型规模受限，对极端新气候场景的泛化仍有限。

---

## 187. MindAU: EEG-Conditioned Facial Action Unit Editing via Dual-Stream Manifold Alignment

**arXiv ID:** 2607.00410 | [PDF](https://arxiv.org/pdf/2607.00410v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 188. Social Popularity of GitHub Projects: A Lifeline or a Liability?

**arXiv ID:** 2607.00435 | [PDF](https://arxiv.org/pdf/2607.00435v1)

**作者:** Mohit Kaushik `[一作]` (Guru Nanak Dev University), Kuljit Kaur Chahal `[通讯]` (Guru Nanak Dev University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 73,195 个 GitHub 开源项目的生命周期进行研究，利用加速失效时间（AFT）生存分析评估社交受欢迎度、代码可读性、外部文档等社会技术因素对项目寿命的影响。

**💡 创新点**

首次将社交需求与人力资本、结构可访问性交互建模，揭示社交受欢迎度在缺乏足够人力时成为负面因素，并提出“可访问性悖论”和“文档悖论”，通过大规模实证验证这一新视角。

**🔧 技术方法**

使用 PCA 构造社交受欢迎度指数；基于注释密度、空行比例的语言归一化可读性指标；二元 Wiki 指标；采用 Weibull AFT 生存模型（Cox PH 被排除），并做多重交互、敏感性分析。

**📊 数据集**

从 SEART GitHub Search 获取的 1.77M 公共仓库，筛选后得到 73,195 个多语言（JavaScript、Python、Java、PHP、C#、TypeScript、Go、C++、C）协作项目。

**📈 对比分析**

与传统 Cox PH 模型对比，AFT Weibull 模型 AIC 更低、C-index 0.74，显著解释项目寿命；通过不同停用阈值（3/6/9/12 月）进行敏感性分析，发现社交受欢迎度始终为负效应，人力缓冲显著正效应，结果稳健。

**⚠️ 局限性**

仅适用于 GitHub，未考虑企业赞助、付费维护、自动化门控等因素；可读性指标过于简化；Wiki 仅记录存在性，未衡量质量；停用阈值选择可能影响结论；对早期或私有项目的适用性有限。

---

## 189. Timesynth: A Temporal Fidelity Framework for Health Signal Digital Twins

**arXiv ID:** 2607.00431 | [PDF](https://arxiv.org/pdf/2607.00431v1)

**作者:** Md Rakibul Haque `[一作]` (University of Utah), Warren Woodrich Pettine `[通讯]` (University of Utah)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究了健康信号数字孪生预测模型的时域动态保真度；

**💡 创新点**

提出 TimeSynth——基于生理参数的合成信号生成器与动态保真诊断框架，揭示点误差与频率、相位失真不一致；

**🔧 技术方法**

使用合成信号生成、幅度/频率/相位诊断、Markov HMM 等评估技术；

**📊 数据集**

以 MIT‑BIH、PPG‑DaLiA、CHB‑MIT EEG 等公开数据拟合生成模型；

**📈 对比分析**

在五个受控情境下对 11 种模型进行比较，局部时间处理模型（如 PatchTST）保持最高动态保真度，传统全序注意力模型表现最差；

**⚠️ 局限性**

仅单变量、固定窗口、简化扰动，缺乏多变量、真实噪声和不规则采样，且对随机切换的评估仍有限。

---

## 190. Speech Playground: An Interactive Tool for Speech Analysis and Comparison

**arXiv ID:** 2607.00418 | [PDF](https://arxiv.org/pdf/2607.00418v1)

**作者:** Stephen McIntosh `[一作]` (University of Tokyo), Nobuaki Minematsu `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了 Speech Playground，一个集成 Python 后端与 Web 前端的交互式语音可视化与比较工具，支持多种深度学习特征（连续、离散、可变长度）与 TextGrid 强制对齐。

**💡 创新点**

创新点在于统一可配置的对齐与距离度量框架，使用户可在同一界面下切换不同编码器、离散化方式和对齐策略，从而便捷地比较和解释语音表示差异。

**🔧 技术方法**

使用了 FastAPI + Python 进行后端服务；SvelteKit + WaveSurfer.js + IndexedDB 构建前端；语音处理库封装 SSL、articulatory、phonological 等编码器；DTW（dynamictimewarping）与 MFA 对齐等算法。

**📊 数据集**

未公开具体数据集，论文以示例录音、公开语音库以及附带的 TextGrid 文件为测试素材。

**📈 对比分析**

比较方法通过计算相似度矩阵并执行 DTW 或离散/段对齐，生成插入、删除、替换的 diff；性能未做量化，仅说明交互响应快速且可实时切换配置。

**⚠️ 局限性**

局限性包括：需要额外的 MFA 对齐服务；对模型加载时延和浏览器资源有限制；目前仅支持预置编码器，未评估多语言或方言的适用性。

---

## 191. The Illusion of High Utility in Safety Alignment of Text-to-Image Diffusion Models

**arXiv ID:** 2607.00402 | [PDF](https://arxiv.org/pdf/2607.00402v1)

**作者:** Adeel Yousaf `[一作]` (University of Central Florida), Mubarak Shah `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了文本到图像扩散模型的安全对齐问题，指出传统粗粒度评估掩盖了细粒度效用下降

**💡 创新点**

提出“语义维持”目标，保持文本嵌入空间的分布和局部相似性结构，解决语义坍塌现象

**🔧 技术方法**

结合安全对齐损失、点对点效用一致性、嵌入分布保持（ESP）和局部结构对齐（LSA）损失

**📊 数据集**

使用 Stable Diffusion v1.4 以及 CoPro 性暗示安全/不安全提示对进行训练

**📈 对比分析**

与多种现有安全方法（DES、Adv-Unlearn、SafeCLIP 等）对比，在 TIFA、GenEval、ASR、FID、CLIPScore 上实现了最优或接近最优的平衡，结构效用提升 5% 以上且 ASR 仅 1.2%

**⚠️ 局限性**

局限在于仅针对文本编码器微调，可能无法覆盖所有安全攻击场景；评估主要基于公开数据集，缺乏对更广泛攻击类型的验证

---

## 192. Explainable quantum neural networks for multi-material topology optimization

**arXiv ID:** 2607.00438 | [PDF](https://arxiv.org/pdf/2607.00438v1)

**作者:** Dahyun Joo `[一作]` (Seoul National University), Do-Nyun Kim `[通讯]` (Seoul National University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种可解释的量子神经网络XQNN，用于多材料拓扑优化。

**💡 创新点**

创新点在于保留十个量子可观测通道并结合Sobel边界提示，实现可解释的材料分类并能在未见边界/载荷和网格分辨率下迁移。

**🔧 技术方法**

使用量子特征映射（Hadamard+phase）、RealAmplitudes ansatz、Z观测及经典软最大读出进行训练和推理。

**📊 数据集**

数据集基于MMA求解器生成的二维和三维多材料拓扑优化历史，涵盖不同边界/载荷、网格尺寸和材料类型。

**📈 对比分析**

相较于传统深度学习方法，XQNN在多材料案例中取得81–94%的精度，且在OOD、网格提升和三维迁移场景下保持负载路径与合规性。

**⚠️ 局限性**

局限性包括仅在状态向量模拟下验证，真实量子设备噪声和误差可能削弱可观测解释；在四材料、细部连接处误差仍明显。

---

## 193. StochasT: Learning with Stochastic Turn Depth for Visual Instruction Tuning

**arXiv ID:** 2607.00465 | [PDF](https://arxiv.org/pdf/2607.00465v1)

**作者:** Yuan Qing `[一作]` (Boston University), Boqing Gong `[通讯]` (Boston University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了StochasT训练策略和Balanced Latin Square评估框架，解决大规模视觉语言模型在多轮训练与单轮评估之间的性能不匹配问题。

**💡 创新点**

创新点在于通过随机裁剪对话历史实现动态上下文深度的StochasT方法，以及引入Balanced Latin Square评估和CRA/CRA+指标，以系统评估模型在不同上下文依赖下的鲁棒性。

**🔧 技术方法**

使用了Beta分布驱动的随机裁剪、Stochastic Depth思想、LoRA微调、视觉贡献评估等技术，并在Qwen2.5‑VL‑3B和LLaVA‑1.5‑7B模型上实施。

**📊 数据集**

利用iNat‑Plant、PathVQA、CoralVQA、TaiwanVQA、MMDU等多轮视觉指令调优数据集进行训练与评估。

**📈 对比分析**

与单轮（SingleT）和多轮（MultiT）标准训练进行对比，StochasT在单轮和多轮评估中均优于MultiT基线，单轮性能接近或略低，多轮性能最高，CRA/CRA+指标也显著提升。

**⚠️ 局限性**

局限性包括对强依赖历史的对话适用性有限，Beta参数的选择会影响效果，评估假设指令相互独立，可能不适用于高度上下文依赖的任务。

---

## 194. Multimodal Continuous Reasoning via Asymmetric Mutual Variational Learning

**arXiv ID:** 2607.00461 | [PDF](https://arxiv.org/pdf/2607.00461v1)

**作者:** Shijie Li `[一作]` (Shanghai Jiao Tong University), Hang Yu `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Asymmetric Mutual Variational Learning（AMVL），通过双向 KL 校准实现多模态大语言模型的连续潜在推理，解决语言空间瓶颈和训练-推理不匹配问题。

**💡 创新点**

创新点在于用正向 KL 对先验进行对齐、反向 KL 对后验进行约束，形成非对称的双向校准机制，消除答案泄露，并且不依赖手工监督信号。

**🔧 技术方法**

采用变分推理框架、对角高斯潜在分布、重参数化、LLM 本地变分头、隐变量插槽、KL 预热与调度等技术实现。

**📊 数据集**

训练使用 Visual-CoT、ReFocus、CogCoM、Zebra-CoT 等多模态推理数据集；评测涵盖 V*、HRBench4K、HRBench8K、BLINK 以及 VisualPuzzles 等基准。

**📈 对比分析**

与 Qwen2.5-VL-7B 基线、Vision-R1、PAPO、PixelReasoner、DeepEyes、LVR、Mull-Tokens、Monet 等方法相比，AMVL 在 BLINK 平均分提升 10.83 分，Jigsaw 提升 32.00 分，整体在各细粒度感知与复杂空间推理任务中均实现了显著性能提升。

**⚠️ 局限性**

局限在于潜在维度或槽数过多会导致信息稀释和训练不稳定；双向 KL 需要精细调度，模型对超参数较为敏感；在更广泛的跨领域任务上的鲁棒性仍待进一步验证。

---

## 195. Search-Based Spatiotemporal and Multi-Robot Motion Planning on Graphs of Space-Time Convex Sets

**arXiv ID:** 2607.00444 | [PDF](https://arxiv.org/pdf/2607.00444v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 196. NATO and Emerging Technologies: The Alliance's Shifting Approach to Military Innovation

**arXiv ID:** 2607.00437 | [PDF](https://arxiv.org/pdf/2607.00437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 197. Real-Time Hard Negative Sampling via LLM-based Clustering for Large-Scale Two-Tower Retrieval

**arXiv ID:** 2607.00448 | [PDF](https://arxiv.org/pdf/2607.00448v1)

**作者:** Ivan Ji `[一作]` (Meta), Aameek Singh `[通讯]` (Meta)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种自监督的基于聚类的硬负采样框架Cluster GOOBS，用于训练大规模推荐系统中的两塔模型。

**💡 创新点**

创新点在于：①利用大语言模型生成多模态内容嵌入，自动构造语义聚类；②在实时训练中从相同聚类中采样硬负样本，避免传统随机或最近邻负样本的易学性；③通过GOOBS框架在OOB池中高效更新和采样，实现在线可扩展的负采样。

**🔧 技术方法**

核心技术包括：两塔模型、交叉熵损失、LogQ校正、基于聚类的负采样、实时OOB池管理、哈希映射更新引擎、LLM驱动的多模态内容编码。

**📊 数据集**

评估数据集：公开数据集MovieLens-1M、Amazon Grocery/Electronics/Home；工业真实数据集约1800万商品。

**📈 对比分析**

与Baseline（in-batch+LogQ）、DNS、CBNS、ANCE、GOOBS随机OOB等做对比，Cluster GOOBS在HR@50/HR@100上分别提升约7.2%–55.6%，工业实验中CTR提升53%，并显著降低热门商品的曝光比例。

**⚠️ 局限性**

限制包括：聚类粒度需合理调节，过细聚类可能导致误负样本；需要LLM的预训练和推理成本；在极稀疏数据或非语义相似任务中的效果待验证。

---

## 198. Understanding Why Language Models Hallucinate: Testing Reasoning Against Priors

**arXiv ID:** 2607.00447 | [PDF](https://arxiv.org/pdf/2607.00447v1)

**作者:** Yangfan Hu `[一作]` (University of Wisconsin--Madison), Jiawei Zhang `[通讯]` (University of Wisconsin--Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一个基于潜在键-任务模型的理论框架，将大语言模型的幻觉现象归因于推理路径失配，并基于该框架设计了 TrapQA 两大诊断测试集（ScientistQA 与 Real‑Life Constrained QA），用于区分缺失知识与推理路径错误。

**💡 创新点**

创新点在于：①将幻觉视为推理路径与提示约束的不一致（Inference Misalignment）而非单纯的知识缺失；②从预训练频率失衡角度解释两种失败模式（键选择偏差与任务检索偏差）；③构建了可控诊断基准 TrapQA，配合闭式检索探测可区分知识缺失与知识部署失败；④通过闭式检索探测进一步证明模型在已掌握事实的情况下仍会选择错误推理路径。

**🔧 技术方法**

技术方法包括：潜在键-任务模型的贝叶斯推断、频率诱导偏差分析、理论推导与正则化、闭式检索探测（probe）以及对比实验。

**📊 数据集**

使用的数据集为 TrapQA（包含 2,925 条 ScientistQA 题目与 500 条 Real‑Life Constrained QA 题目），题目来源于维基百科科学家简介与 SWOW 关联词库，题目经过 GPT/Claude 生成并人工筛选。

**📈 对比分析**

与现有的 TruthfulQA、HaluEval 等事实性评测不同，TrapQA 通过提供/不提供额外信息的闭式检索探测来对比模型性能。实验显示：在仅给出名字的 ScientistQA 条件下，模型错误率从 2.5%（Gemini‑low）到 37.2%（DeepSeek‑non‑reasoning）不等；提供完整简介后错误率几乎为 0%；在 Real‑Life Constrained QA 中，错误率介于 3.6%（Gemini）到 36.4%（DeepSeek‑chat）。

**⚠️ 局限性**

限制包括：仅评测 GPT‑5.2、Gemini‑3.1 Pro、Claude‑Sonnet 4.6 与 DeepSeek‑V3.2；模型版本与推理设置可能随时间变化导致结果漂移；题目基于 2026‑07‑02 的知识，后续可能因学术/技术进展导致答案漂移；并未覆盖所有可能的推理路径或外部工具使用场景。

---

## 199. Generalized Normal Constraint (GNC): A Complete Geometric Generalization of the NNC Method

**arXiv ID:** 2607.00405 | [PDF](https://arxiv.org/pdf/2607.00405v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 200. DroneIQA-VLE: Multi-Task Drone Image Quality Assessment via Vision-Language Ensemble

**arXiv ID:** 2607.00416 | [PDF](https://arxiv.org/pdf/2607.00416v1)

**作者:** Wei Sun `[一作]` (East China Normal University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用 SigLIP2 视觉编码器与 LoRA 调优的 Qwen3.5-9B 多模态大语言模型的融合管道，对 UAV 图像进行全球、目标与背景三维度的质量预测，并通过多模型加权平均得到最终评分。

**💡 创新点**

创新点在于将多任务回归与跨模态大语言模型相结合，既保留了视觉编码器在空间质量感知上的优势，又利用大语言模型的语义推理与高层次感知来提升全局质量预测精度；同时通过交叉管道平均与四参数逻辑映射实现了稳健的预测。

**🔧 技术方法**

使用的技术包括：SigLIP2 ViT-L/16 与 ViT-SO400M/14 两种视觉 Transformer，三层全连接回归头；Qwen3.5-9B 多模态 LLM，LoRA 微调（rank = 64，α = 128）；PLCC 与 Fidelity 损失相结合的训练目标；四参数逻辑映射校准与跨模型平均推理。

**📊 数据集**

基于 Drone-IQA GC 2026 基准数据集，包含约 6,000 张来自 VisDrone 与 UAVDT 的 UAV 图像，三维质量标注（全球、目标、背景）。

**📈 对比分析**

与其他参赛方案对比，VQA（本方法）在 PLCC 0.9484、SRCC 0.9420 的测试集上排名第 2，仅低于冠军 0.3% 的整体得分，显著优于第 3 名方法约 0.9% 的分数，表明两模态融合在提升质量预测一致性与线性相关性方面效果显著。

**⚠️ 局限性**

局限性包括：模型规模大、推理成本高；对数据分布变化的泛化仍需验证；跨模态融合方法仍以经验调参为主，缺乏理论解释；以及对低质量图像中目标尺度过小、背景复杂时的鲁棒性尚未充分探索。

---

## 201. Learning Generalizable Skill Policy with Data-Efficient Unsupervised RL

**arXiv ID:** 2607.00392 | [PDF](https://arxiv.org/pdf/2607.00392v1)

**作者:** Jongchan Park `[一作]` (Sungkyunkwan University), Yusung Kim `[通讯]` (Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 GenDa 框架，解决无监督强化学习中的语义漂移和泛化不足问题

**💡 创新点**

通过技能重标记消除离线学习的语义漂移，并引入互补信息瓶颈（CIB）分离全局上下文，实现技能更稳健、可迁移

**🔧 技术方法**

技能重标记（基于状态差异的标签更新）、互补信息瓶颈（变分信息瓶颈）、SAC off‑policy 训练、EMA 目标网络、均匀性正则化

**📊 数据集**

DeepMind Control Suite（Humanoid、Quadruped、Fish 等数值与像素环境）和 MuJoCo 操作环境

**📈 对比分析**

与 METRA、CSF、DIAYN、CIC 等前沿无监督学习方法对比，GenDa 在状态覆盖、任务覆盖和下游成功率上均优于基线，且数据效率提升约 5 倍

**⚠️ 局限性**

仍受限于一次技能/一次 episode 的假设、对全局上下文的依赖需进一步提升对视觉变化的鲁棒性，以及对精细操控任务的适配不足

---

## 202. A Penny for Your Prompts: Experiments Detecting and Mitigating LLM Usage by Survey Respondents

**arXiv ID:** 2607.00403 | [PDF](https://arxiv.org/pdf/2607.00403v1)

**作者:** Zane Xu `[一作]` (New Jersey Institute of Technology), Nathan Malkin `[通讯]` (New Jersey Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在线问卷中使用大型语言模型（LLM）的普及程度，并评估了检测与防范措施。

**💡 创新点**

首次系统量化了不同平台、问卷长度及禁用复制粘贴等干预下LLM使用率，并提出基于键盘事件和自报的实用检测框架。

**🔧 技术方法**

采用JavaScript记录键盘、鼠标、切页、复制粘贴等浏览器事件，并结合人工评估与自报。

**📊 数据集**

在Prolific和Amazon Mechanical Turk上共招募约400名美国受访者，完成5分钟或10分钟问卷。

**📈 对比分析**

与传统注意力检查相比，键盘事件+自报组合检测LLM使用的准确率高达约70%，而单独的自报或复制粘贴率分别低；复制粘贴禁令虽降低LLM使用，但整体数据质量下降。

**⚠️ 局限性**

受限于样本量、仅针对隐私/安全问卷、缺乏真实标注数据、对浏览器代理的评估有限。

---

## 203. Scalable Security and Migration-Aware SFC Provisioning in LEO Satellite Networks

**arXiv ID:** 2607.00471 | [PDF](https://arxiv.org/pdf/2607.00471v1)

**作者:** Mohammed Mahyoub `[一作]`, Halim Yanikomeroglu `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了面向低地轨道（LEO）卫星网络的安全与迁移感知服务功能链（SFC）放置框架；

**💡 创新点**

创新点在于将基于ISO/NIST风险管理的共置安全风险模型与可避免/强制迁移惩罚统一入多切片MILP，并通过ADMM式的分块分解实现可扩展性；

**🔧 技术方法**

采用的技术包括：多切片MILP建模、McCormick线性化、分布式ADMM式惩罚分解（S‑ADMM与P‑ADMM），以及可避免迁移的epigraph约束；

**📊 数据集**

实验使用基于MATLAB SatCom工具箱模拟的80颗卫星Walker‑Delta星座，包含10-25个切片、5名用户/切片、3个VNF实例/功能/卫星；

**📈 对比分析**

与无安全MILP、贪心启发式和单一切片求解等基线比较，结果表明P‑ADMM在保持全延迟合规的同时实现近零共置风险，S‑ADMM在可接受的迁移成本下获得更快求解；

**⚠️ 局限性**

局限性包括：分解仅保证坐标最优而非全局最优；并行加速受限于单线程求解器调用，未实现完全分布式实现；

---

## 204. Multi-scale Mixture of World Models for Embodied Agents in Evolving Environments

**arXiv ID:** 2607.00457 | [PDF](https://arxiv.org/pdf/2607.00457v1)

**作者:** Jinwoo Jang `[一作]`, Honguk Woo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对具身智能体，设计了一种多尺度混合世界模型框架，通过两阶段尺度感知路由和尺度依赖的知识适应机制实现实时更新与多尺度推理。

**💡 创新点**

创新点在于：① 引入“经验距离”作为连续尺度轴，结合CLT原理实现显式尺度感知路由；② 通过尺度依赖的遗忘率和双向尺度间知识转移，允许不同尺度以不同速度演化；③ 两阶段路由使得测试时可针对特定尺度进行局部更新。

**🔧 技术方法**

核心技术包括：Mixture of Experts (MoE) 结构、连续元路由器、经验距离（Mahalanobis距离）计算、尺度感知损失、尺度依赖的忘却机制、跨尺度门控信息传递、Vision‑Language 模型 Qwen3‑VL‑4B‑Instruct 作为感知与语言后端。

**📊 数据集**

使用的基准数据集有 EmbodiedBench（Habitat 与 Navigation）和 HAZARD（火灾、洪水等灾害情景），并在 Franka Research 3 机器人上进行真实世界抓取实验。

**📈 对比分析**

与 SayCanPay、FLARE、LLM‑Planner 及传统 MoE 进行对比。实验显示在 EmbodiedBench 上相对 SayCanPay 提升 6.05%p，在 HAZARD 的火灾场景上相对 FLARE 提升 1.49%p；在真实机器人抓取任务中，平均成功率与最强基线相当且在最差案例上表现最好。

**⚠️ 局限性**

主要局限包括：依赖于基础 VLM 的推理能力，若底层模型不足会限制性能；实验在有限的机器人平台与任务范围内验证，需进一步扩展到更广泛的环境与长周期任务；框架仍继承 MoE 的参数量与训练成本等问题。

---

## 205. HyFL-CLIP: Hyperbolic Fine-Tuning of CLIP for Robust Long-Context Understanding

**arXiv ID:** 2607.00428 | [PDF](https://arxiv.org/pdf/2607.00428v1)

**作者:** Ji Ha Jang `[一作]` (Seoul National University), Se Young Chun `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 CLIP 预训练模型的基础上，提出 HyFL-CLIP，通过超球几何的微调实现长文本（>77 词）与图像的鲁棒对齐，支持长上下文检索、文本扰动检索、跨模态与跨模态内部检索以及 Stable Diffusion 的图像生成。

**💡 创新点**

核心创新点：① 跨流形相似度蒸馏，将欧氏空间中的文本‑图像相似度知识迁移到 Lorentz 超球空间；② 采用 Einstein 中点聚合和层级蕴涵损失，显式建模整体描述与其子句之间的层次关系；③ 在超球空间中使用 geodesic 对比损失与熵正则，提升对文本扰动的鲁棒性。

**🔧 技术方法**

技术手段：Lorentz（Minkowski）超球映射、交叉流形相似度蒸馏、Einstein 中点聚合、层级蕴涵损失、超球 geodesic 对比损失、熵正则化；实现框架基于 Open‑CLIP 的 ViT‑B/16 与 ViT‑L/14。

**📊 数据集**

数据集：ShareGPT4V（1.2M 图像‑长文本对）用于训练；评估使用 DOCCI、DCI、Long‑DCI、Urban‑1k（长文本检索）、COCO、Flickr30K（短文本检索）、nocaps、20 Newsgroups、IMDB（文本‑文本检索）、Stable Diffusion XL 生成任务。

**📈 对比分析**

与现有长文本 CLIP、HiMo‑CLIP、FineLIP、LongD‑CLIP、SmartCLIP、Fix‑CLIP 等方法对比，HyFL‑CLIP 在长文本检索、文本扰动检索和文本‑文本检索上均实现了 18%–20% 的提升，且在短文本检索上保持或略优于欧氏基线，显示出在层次语义建模与鲁棒性方面的显著优势。

**⚠️ 局限性**

局限性：① 仅在 Vision‑Language 预训练框架下验证，缺乏对大规模多模态场景的进一步扩展；② 超球映射与对比损失带来额外的数值稳定性与计算开销；③ 对极端长文本（数千词）或非结构化文本的适应性尚未充分评估。

---

## 206. Selective Test-Time Debiasing for CLIP via Reward Gating

**arXiv ID:** 2607.00423 | [PDF](https://arxiv.org/pdf/2607.00423v1)

**作者:** Jaeho Han `[一作]` (Chung-Ang University), Junyeong Kim `[通讯]` (Chung-Ang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于奖励门控的测试时适配（RG‑TTA）框架，用以对视觉‑语言模型（VLM）进行选择性去偏，避免统一去偏带来的公平性与效用权衡问题。

**💡 创新点**

创新点在于：①引入输入敏感性门控，只在检测到查询与属性关联强时才激活去偏正则；②采用强化学习（REINFORCE）实现测试时的增量更新，局部调整查询模态编码器；③设计了结合 CLIP 对齐奖励与属性平衡奖励的门控奖励函数，确保去偏与语义匹配的双重目标。

**🔧 技术方法**

使用技术包括：强化学习与 REINFORCE、测试时适配（TTA）、Top‑K 截断的候选集、属性子空间投影、CLIP 对齐奖励、动态预算（K 与更新步数）以及门控阈值判断。

**📊 数据集**

实验数据集涵盖公平性基准 FairFace、UTKFace、FACET 以及零-shot 任务 ImageNet‑1K 与 Flickr1k，评价指标包括 MaxSkew@k、NDKL@k、ABLE 与 ImageNet Top‑1。

**📈 对比分析**

与原 CLIP、CLIP‑clip、Biased‑prompts、Joint V‑L 等基线对比，RG‑TTA 在 FairFace、UTKFace 与 FACET 的 MaxSkew 明显下降，同时 ImageNet Top‑1 上保持或提升，ABLE 也提升，证明能够同时兼顾公平与效用，解决了统一去偏的公平‑效用权衡。

**⚠️ 局限性**

局限性包括：①每个查询需进行参数更新，导致额外计算与延迟；②对门控阈值、候选预算和奖励权重等超参数敏感；③依赖外部更强的奖励模型，可能引入其偏见；④目前仅针对单一属性，扩展到多属性需更复杂的奖励/约束设计；⑤在某些查询（尤其是难以区分属性的样本）上门控失效，导致去偏不足。

---

## 207. A Simple Solution to Improving Human Supervision of Algorithms: Evidence from Smart Vending

**arXiv ID:** 2607.00420 | [PDF](https://arxiv.org/pdf/2607.00420v1)

**作者:** Minda Zhao `[一作]`, Tao Zhu `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在智能售货机补货系统中，通过实验设计与评估了一种受限覆盖（constrained override）政策，限制每名工人每台机器只能进行两次向下覆盖，以提升库存效率并减少销量损失。

**💡 创新点**

创新点在于：①提出了基于基数约束的“选择性过滤”机制，利用有限覆盖机会迫使工人优先选择高价值覆盖；②通过随机现场实验和局部平均处理效应（LATE）验证了该机制的有效性；③在不修改算法、不提供额外信息或培训的前提下实现了可扩展的、人机协作改进方案。

**🔧 技术方法**

技术方法包括：随机现场实验、双重差分（DID）估计、两阶段最小二乘（2SLS）与LATE框架、逻辑回归预测高质量覆盖并模拟个性化覆盖策略。

**📊 数据集**

数据集：Fengyi Technology 2023年运营数据，涉及553名工人、约59,000台智能售货机、4,000种SKU的补货与销量记录，涵盖了工人行为、算法建议与实际补货、库存与销量等信息。

**📈 对比分析**

比较方法：将受限覆盖政策与无覆盖（no override）和自由覆盖（free override）三组进行对照；使用DID估计机器层面销售、销售概率与库存变化；使用LATE评估覆盖行为对单SKU库存与销量的因果效应。结果显示，受限覆盖使库存下降1.28%且销量保持不变，优于自由覆盖导致的1.95%库存下降和1.19%销量下降；LATE表明受限覆盖的平均覆盖可减少3.45件库存，且不显著影响销量。

**⚠️ 局限性**

局限性：①需要工人具备足够的私有信息且激励与算法目标一致；②该策略主要适用于低至中等风险的日常运营决策，难以推广至高风险领域；③实验在特定算法与业务场景下进行，结果在其他行业或算法架构下可能需要进一步验证；④受限覆盖在覆盖机会极少或工人经验不足时效果可能不显著。

---

## 208. EO-VGGT: Orbital Ray-Conditioned 3D Foundation Models for Satellite Multi-View Reconstruction

**arXiv ID:** 2607.00417 | [PDF](https://arxiv.org/pdf/2607.00417v1)

**作者:** Qiyan Luo `[一作]` (Wuhan University), Mi Wang `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

结合预训练的视觉Transformer模型，对卫星多视角DSM重建进行适配，提出GCCS视角筛选与SRE+RPAA几何注入方法，实现快速前向推理。

**💡 创新点**

将卫星推行式相机的RFM几何通过Sensor‑Ray Encoder映射为高维空间几何token，并用Ray‑Pointing‑Aware Adapter低参数注入，解决中心透镜先验与推行式相机不匹配；同时GCCS视角选择兼顾几何多样性与辐射一致性。

**🔧 技术方法**

预训练的Visual Geometry Grounded Transformer（VGGT）、Rational Function Model逆投影、光度一致性筛选、Sensor‑Ray Encoder、Ray‑Pointing‑Aware Adapter、宏观DSM后处理。

**📊 数据集**

US3D（WorldView‑3卫星影像）跨城市（Jacksonville 训练/验证，Omaha 交叉测试）数据集。

**📈 对比分析**

在统一评估协议下与VGGT、π³、MapAnything等通用基础模型以及EOGS、SatMVS、SatNGP等EO专用管线比较；EO‑VGGT平均MAE 1.751 m、RMSE 3.184 m、P95 6.211 m，较基线降低约28.5% MAE、18% P95，且保持完整覆盖。

**⚠️ 局限性**

对低纹理、阴影、雾等区域仍存在较高的P95误差；未完成绝对地理配准；单光谱影像限制对极端光照变化的鲁棒性。

---

## 209. DriveVer: Lightweight Trajectory Evaluator as Test-Time Verifier for Autonomous Driving

**arXiv ID:** 2607.00399 | [PDF](https://arxiv.org/pdf/2607.00399v1)

**作者:** Chong He `[一作]` (Tsinghua University), Fuxi Wen `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量级的推理时轨迹验证器DriveVer，用于评估并纠正端到端自动驾驶模型生成的轨迹。

**💡 创新点**

通过引入测试时缩放、双头架构和条件平衡数据集，实现了轨迹质量评估与几何修正的统一，并实现了低成本的实时推理。

**🔧 技术方法**

采用跨模态交叉注意力的Transformer解码器、DINOv2视觉特征、LoRA参数高效微调以及方向向量余弦损失进行训练。

**📊 数据集**

基于NAVSIM基准构建的条件驱动轨迹验证数据集，包含人类驾驶示例与多样化候选轨迹。

**📈 对比分析**

在NAVSIM v1 与 v2 上与多种SOTA端到端规划器（DiffusionDrive、AdaThinkDrive、ELF‑VLA、DrivoR）进行比较，DriveVer平均提升 PDMS/EPDMS 0.9-1.0 分，表现最优且推理时延仅80 ms。

**⚠️ 局限性**

仍受限于训练数据分布，对极端稀缺场景的泛化不完全，且在高质量轨迹上过度修正可能导致性能下降。

---

## 210. Interpretable vs Learned Encoders for High-Cardinality Fraud Detection

**arXiv ID:** 2607.00477 | [PDF](https://arxiv.org/pdf/2607.00477v1)

**作者:** Xiao Han `[一作]` (Emory University), Chenyu Wu `[通讯]` (Duke University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

该研究在IEEE-CIS欺诈检测数据集上，对七种分类特征编码方法进行统一实验，比较其在固定LightGBM下的性能，并与CatBoost与TabNet进行跨范式对照。

**💡 创新点**

创新点包括：①将编码器作为独立变量在相同学习器下对比，明确编码对AUC‑ROC≈0.01的贡献；②提出可审计的分层分组编码并评估其可解释性；③通过计算成本、可解释性和多指标评估提供完整权衡框架。

**🔧 技术方法**

使用的技术包括One‑Hot、Target、Frequency、Tier Grouping、CatBoost内置编码、实体嵌入+LightGBM、Target编码+TabNet等；配合LightGBM、CatBoost、TabNet模型，采用Nadeau–Bengio校正的重采样t检验、Friedman+Nemenyi检验以及t‑SNE可视化。

**📊 数据集**

所使用的数据集为公开的IEEE‑CIS Fraud Detection数据集，包含590,540条交易记录、3.5%正样本，以及8个高基数类别变量。

**📈 对比分析**

实验采用分层5折交叉验证×3次重复，固定LightGBM超参，比较AUC‑ROC和AUC‑PR；结果显示实体嵌入+LightGBM获得最高AUC‑ROC（0.9612），CatBoost紧随其后，Tier Grouping仅次于Target编码，而TabNet在此配置下性能最差且计算成本最高。

**⚠️ 局限性**

局限性包括：仅在单一IEEE‑CIS数据集上验证；TabNet未进行充分调参；未使用时间序列划分导致可能的泄漏风险；特征匿名限制了解释性；固定学习器未针对各编码器微调；且指标差异微小，实际应用需结合成本与可解释性。

---

## 211. Complexity of Low-Degree Skew Polynomial Multiplication over Finite Fields

**arXiv ID:** 2607.00476 | [PDF](https://arxiv.org/pdf/2607.00476v1)

**作者:** Ke Ye `[一作]` (University of Chinese Academy of Sciences), Ruichen Qiu `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文证明了有限域上不交换多项式环𝔽_q^n[x;σ]中，若σ为q-弗罗贝尼乌斯自同构，且两多项式度数均不超过d<n，则乘法可在O(d^{ω_K-1}·n)个𝔽_q运算内完成。

**💡 创新点**

核心创新在于将有限域情形归约为分裂代数（split algebra）情形，并利用Couveignes–Ezome的等变乘法理论与现有快速算法相结合，达成了此前仅猜想的上界，并与下界一致，证实了Caruso–Le Borgne的猜想。

**🔧 技术方法**

主要技术包括：1）等变乘法理论（equivariant multiplication）；2）快速矩阵乘法的指数ω_K；3）对多项式乘法的分块与重排；4）使用G-equivariant对称复杂度的上界；5）在分裂代数中将乘法拆解为多条独立乘法。

**📊 数据集**

该研究不涉及具体数据集，所述算法仅在理论上分析算术复杂度。

**📈 对比分析**

与先前的Caruso–Le Borgne算法相比，本结果在d<n的情形下实现了与下界一致的近最优复杂度，证明了在此范围内无法进一步降低复杂度。

**⚠️ 局限性**

局限性：1）只针对有限域𝔽_q^n，且假设σ为q-弗罗贝尼乌斯自同构；2）对d≥n的情况仅复用已知结果；3）实际实现细节（如常数因子、预处理成本）未给出；4）对更一般的⟨σ⟩-伽罗瓦代数的完整性尚未验证。

---

## 212. How Early Is Early Enough? Design-Dependent Observation-Window Sufficiency in Subscription Churn Prediction

**arXiv ID:** 2607.00473 | [PDF](https://arxiv.org/pdf/2607.00473v1)

**作者:** Xiao Han `[一作]` (Emory University), Tongchen Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

建立并验证订阅服务中预测流失所需的最小观测窗口，并用三种实验设计对其进行压力测试。

**💡 创新点**

证明窗口“足够”与实验设计相关，提供九窗口足够曲线并揭示膝点仅在特定设计与特征集合下出现。

**🔧 技术方法**

LightGBM、1D‑CNN、逻辑回归；Kneedle、边际收益法检测膝点；多实验设计与分层交叉验证。

**📊 数据集**

公开的 KKBox 音乐流媒体订阅数据（约 250 万用户，30GB 日志）。

**📈 对比分析**

通过 PR‑AUC、ROC‑AUC、F1、前 5% 精度对模型和窗口进行对比；模型在 45‑90 天膝点处已捕获 90% 的可获得行为提升，ROC 从 0.878 上升至 0.901。

**⚠️ 局限性**

仅在单一音乐流媒体域验证；曲线形状和膝点位置高度依赖于队列、目标与特征集合，无法直接迁移至其他行业。

---

## 213. ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving

**arXiv ID:** 2607.00466 | [PDF](https://arxiv.org/pdf/2607.00466v1)

**作者:** Sangjin Choi `[一作]` (KAIST), Peng Cheng `[通讯]` (Microsoft Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种专家局部性感知的解码路由器，用于预填充-解码（PD）分散的混合专家（MoE）模型服务，旨在通过利用请求的预填充专家激活来优化解码过程。

**💡 创新点**

创新点在于引入了专家局部性作为解码路由的第二个维度，通过构建专家签名来预测请求在解码时的专家重叠，从而实现更高效的负载均衡和延迟降低。

**🔧 技术方法**

使用了平衡K均值聚类算法来离线划分专家签名空间，并在在线阶段通过局部性带路由将请求分配给负载最轻的解码工作者。

**📊 数据集**

在三个MoE模型（Qwen3-30B-A3B、GPT-OSS-120B、Gemma-4-26B-A4B）上进行了评估，使用了任务和语言两种工作负载的数据集。

**📈 对比分析**

与四种负载均衡基线方法相比，提出的方法在三种MoE模型和两种工作负载上将中位数每输出令牌时间（TPOT）减少了5.9%到13.9%。

**⚠️ 局限性**

局限性在于该方法依赖于预填充阶段的专家激活信息，可能在请求的专家激活模式发生变化时表现不佳。

---

## 214. MolSafeEval: A Benchmark for Uncovering Safety Risks in AI-Generated Molecules

**arXiv ID:** 2607.00464 | [PDF](https://arxiv.org/pdf/2607.00464v1)

**作者:** Tong Xu `[一作]` (Zhejiang University), Huajun Chen `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MolSafeEval基准，用于评估AI生成分子的安全性

**💡 创新点**

构建多源安全知识图谱MolSafeKG，并将LLM与KG检索结合实现解释性安全评估

**🔧 技术方法**

使用RDKit结构解析、基于Tanimoto相似度检索、LLM推理以及多维安全评分机制

**📊 数据集**

利用GEOM-DRUG、ZINC、CrossDocked2020、ChEBI-20等公开数据库生成四类任务数据集

**📈 对比分析**

在28种生成模型上进行评估，发现部分模型生成的分子毒性和危害风险高，MolSafeEval在11项安全预测任务上平均准确率>80%，在安全评分上显著优于传统预测器

**⚠️ 局限性**

局限在于只捕捉已知毒性/危害信息，无法发现全新毒性机制，且安全预测仍受KG完整性和LLM推理误差影响

---

## 215. A Multi-Resolution Finite-Volume Inspired Deep Learning Framework for Spatiotemporal Dynamics Prediction

**arXiv ID:** 2607.00460 | [PDF](https://arxiv.org/pdf/2607.00460v1)

**作者:** Xin-Yang Liu `[一作]` (University of Notre Dame), Jian-Xun Wang `[通讯]` (University of Notre Dame)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种多分辨率有限体积启发式网络，用于高分辨率时空动力学的长时序自回归预测。

**💡 创新点**

将有限体积守恒结构与多分辨率学习相结合，分离子域平均演化与局部细节重构，并提供两种变体（无方程/方程嵌入）实现物理一致性与高效计算。

**🔧 技术方法**

使用物理信息深度学习、有限体积方法、域分解、全局网络预测接口通量、共享局部网络零均值重构、数值算子嵌入以及教师强迫与自回归训练等技术。

**📊 数据集**

在Burgers方程（1D/2D）、浅水方程以及Kolmogorov流（不可压Navier-Stokes）等数据集上进行实验，包含不同Re数、黏性、随机初始条件的多条轨迹。

**📈 对比分析**

与CNN、ViT、FNO等基线在同一数据集下做自回归滚动比较，MuRFiV在长时步误差、波形保持、能谱匹配上显著优于基线，误差累积更慢且在未见参数时表现更稳健。

**⚠️ 局限性**

目前仅适用于守恒型动力学，子域分解为正交网格，未处理复杂几何或非周期边界，对强非线性或多相流的适应性仍待验证，需要进一步提升数值算子与物理约束的协同效果。

---

## 216. Gaze-Informed Proactive AI Assistance for Children's Picture Exploration

**arXiv ID:** 2607.00445 | [PDF](https://arxiv.org/pdf/2607.00445v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 217. CloudyGUI: A Novel Python-based Framework for Auto-Scaling and Cloud Workload Analysis

**arXiv ID:** 2607.00455 | [PDF](https://arxiv.org/pdf/2607.00455v1)

**作者:** Jyoti Bawa `[一作]` (Guru Nanak Dev University), Kamaljit Kaur `[通讯]` (Guru Nanak Dev University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 CloudyGUI——一个基于 Python 的图形化云计算模拟框架，用于生成真实感工作负载、预测资源需求并在模拟环境中实现预测型自动伸缩。

**💡 创新点**

创新点在于将 ML 预测模型（XGBoost、LSTM）直接嵌入 MAPE‑loop 的自动伸缩逻辑，并提供直观的 GUI 交互，弥补了现有 Java/CLI 仿真器在易用性和现代 AI 集成方面的不足。

**🔧 技术方法**

技术栈包括 Python、Django/Plotly/Chart.js GUI、XGBoost 与 LSTM 预测模型、时间序列预处理、K‑近邻特征工程、MAPE 控制循环、模拟工作负载 DAG 与资源池管理。

**📊 数据集**

使用了 Alibaba Cluster Trace 2018 数据集进行历史工作负载对比，并通过 K‑S 检验和人工专家评估来验证生成工作负载的真实性。

**📈 对比分析**

与 Cloudy（无 GUI）和 AutoScaleSim 等基线进行内部、外部与中间层验证，结果显示 GUI 仅增加 1.4×–4.67× 的轻量级开销，预测模型 R² 超过 0.98，预测型伸缩比被动伸缩提升约 25% 响应时间、80% SLA 合规率、30% 过度/不足资源率。

**⚠️ 局限性**

局限性包括缺乏网络层仿真、对多云提供商的适配不完整、GUI 的学习曲线仍略高、可配置的调度策略有限，且实验仅在单机模拟环境下验证，未在真实云上进行大规模部署验证。

---

## 218. Gauging, Measuring, and Controlling Critic Complexity in Actor-Critic Reinforcement Learning

**arXiv ID:** 2607.00452 | [PDF](https://arxiv.org/pdf/2607.00452v1)

**作者:** Konstantin Garbers `[一作]` `[通讯]` (Peking University), Konstantin Garbers (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并量化了 actor‑critic 中 critic 的谱复杂度，并在 TD3 与 PPO 训练中跟踪其随时间的变化

**💡 创新点**

首次将谱有效秩熵作为可监测的 critic 复杂度指标，并证明可通过正则化器直接控制其动态

**🔧 技术方法**

使用谱有效秩熵正则化、标准 MLP critic 与 actor、以及 TD3/PPO 训练流程

**📊 数据集**

在 Pendulum‑v1、HalfCheetah‑v4、Walker2d‑v4、Ant‑v4 等 MuJoCo 环境上进行实验

**📈 对比分析**

对比未正则化与不同系数正则化的返回、偏差与波动性，发现 TD3/HalfCheetah‑v4 在中等系数下显著提升返回，但其他任务未出现一致改善

**⚠️ 局限性**

结果高度依赖算法与任务，缺乏跨算法、跨超参数的鲁棒性验证，限制了结论的普适性

---

## 219. VideoSearch-R1: Iterative Video Retrieval and Reasoning via Soft Query Refinement

**arXiv ID:** 2607.00446 | [PDF](https://arxiv.org/pdf/2607.00446v1)

**作者:** Seohyun Lee `[一作]` (KAIST), Hyunwoo J. Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种通过软查询细化（SQR）和硬查询细化（HQR）实现的迭代视频检索与推理框架。

**💡 创新点**

创新点在于将软查询细化嵌入迭代检索流程，并结合InfoNCE目标与可扩展的数据训练策略，以提升检索与推理精度。

**🔧 技术方法**

主要技术包括Soft Query Refinement、Hard Query Refinement、InfoNCE 损失函数、迭代检索机制与视频语义匹配模型。

**📊 数据集**

使用了主流视频检索数据集（如MSR‑VTT、ActivityNet）以及VideoQA数据集进行训练与评估。

**📈 对比分析**

通过与传统检索+推理基线以及单轮检索方法对比，实验显示在MAP/NDCG上提升约10%~15%，在VideoQA上实现更高准确率。

**⚠️ 局限性**

局限性包括对大规模训练数据的高度依赖、推理步骤的计算成本，以及在跨域或小样本场景下的泛化能力仍需提升。

---

## 220. Learning Gait-Aware Quadruped Locomotion with Temporal Logic Specifications

**arXiv ID:** 2607.00442 | [PDF](https://arxiv.org/pdf/2607.00442v1)

**作者:** Merve Atasever `[一作]` (University of California Berkeley), Jyotirmoy V. Deshmukh `[通讯]` (University of Southern California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用信号时序逻辑（STL）模板为四足机器人制定模式感知的奖励函数，并用PPO训练单一策略实现多速度步态控制。

**💡 创新点**

通过将不同速度下的步态约束编码为可解释的STL规范，实现奖励的可解释性与步态切换的连续性。

**🔧 技术方法**

采用STL定量鲁棒性、PPO强化学习、MuJoCo XLA GPU并行仿真及对抗随机化。

**📊 数据集**

使用Google Barkour机器人在MuJoCo XLA仿真中生成的专家轨迹数据（每种步态约50条轨迹）作数据集。

**📈 对比分析**

与手工设计奖励基线对比，STL奖励在所有速度下保持最高成功率和生存率，并在高速度时实现最低运输成本；手工奖励在低速时略优。

**⚠️ 局限性**

仅在仿真环境下验证，未进行真实机器人部署，步态转换仅基于速度而未考虑地形。

---

## 221. BT-APE: A Computationally Light Backtracking Approach to Automatic Prompt Engineering for Requirements Classification

**arXiv ID:** 2607.00427 | [PDF](https://arxiv.org/pdf/2607.00427v1)

**作者:** Mohammad Amin Zadenoori `[一作]` (University of Padova), Alessio Ferrari `[通讯]` (University College Dublin)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种名为 BT-APE 的轻量级自动提示工程（APE）框架，用于在需求分类任务中通过后向回溯搜索、动态示例选择和投票机制自动优化提示，显著提升 LLM 的分类性能。

**💡 创新点**

创新点包括：
- 采用受限回溯（bounded backtracking）在提示搜索中加入耐心阈值，避免无效路径浪费迭代；
- 引入平衡的四元组示例采样，保证 LLM 在每次迭代同时看到成功与失败的正负样本；
- 使用三次投票机制减少 LLM 生成噪声对搜索动态的影响；
- 与现有 PE2 方法相比，BT-APE 在保持相近精度的同时大幅降低了输入 token 数量和运行时间，提升了资源友好性。

**🔧 技术方法**

使用的技术包括：
- 基于指令调优的大规模 LLM（Qwen2‑7B‑Instruct、Falcon3‑7B‑Instruct、Granite‑3.2‑8B‑Instruct、Ministral‑8B‑Instruct‑2410、LLaMA‑3‑8B‑Instruct）；
- 自动提示工程框架 BT-APE（后向回溯、动态示例、投票）；
- 传统提示方法（零样本、少样本、链式推理、链式推理+少样本）与 APE‑PE2 基线做对比；
- 统计显著性检验（Wilcoxon、Friedman、线性混合效应模型）评估性能差异。

**📊 数据集**

数据集：
- PROMISE Refined（功能/非功能四分类）
- PROMISE NFR（功能 vs 非功能二分类）
- SecReq（安全 vs 非安全二分类）

**📈 对比分析**

比较方法：对每个模型和数据集分别使用零样本、少样本、链式推理、链式推理+少样本、PE2 和 BT-APE 六种提示策略。结果显示：
- BT-APE 在所有 15 个模型×数据集组合中均显著优于四种传统提示策略，效果量为中至大；
- 与 PE2 的性能差距可忽略（平均 Δ≈0.001，效应量 negligible），两者在精度上相当；
- BT-APE 在计算开销上比 PE2 轻量约 72% 的 token 量和 66% 的时间，尤其适合资源受限环境。

**⚠️ 局限性**

局限性：
- 仅在 7–8B 参数范围的指令调优 LLM 上评估，未验证更大模型或更小模型的适用性；
- 只针对需求分类任务，未覆盖其他 RE 场景（如需求生成、追踪等）；
- 需要一定量的标注数据来构造验证集与示例池，若数据稀缺可能受限；
- 对比仅考虑单一实验设置，未对不同回溯阈值或示例比例进行更细粒度的敏感性分析。

---

## 222. Robust Operational Space Control with Conformal Disturbance Bounds for Safe Redundant Manipulation

**arXiv ID:** 2607.00424 | [PDF](https://arxiv.org/pdf/2607.00424v1)

**作者:** Wenhua Liu `[一作]` (University of Houston), Qin Lin `[通讯]` (University of Houston)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种结合扩展状态观测器(ESO)与滑动窗口合成预测的鲁棒操作空间控制框架，实现了在动态不确定环境下的安全跟踪与高精度控制。

**💡 创新点**

① 用ESO直接在操作空间估计总扰动，无需全状态测量；② 采用滑动窗口合成预测在线估计扰动变化上界，显著降低保守性；③ 将估计结果嵌入高阶控制障碍函数，提供概率安全保证。

**🔧 技术方法**

扩展状态观测器（ESO）、高阶控制障碍函数（CBF/ECBF）、滑动窗口合成预测、基于QPs的安全控制、Franka ROS实现与QP求解器qpOASES。

**📊 数据集**

在7-DoF Franka Research 3机械臂上进行仿真与实测，注入非线性状态依赖扰动；未使用公开数据集。

**📈 对比分析**

与传统OSCTC和残差学习（神经网络）对比，跟踪均方误差从约2.0e‑4下降到1.7e‑5；安全约束下违约率显著降低，且对保守性的取舍更灵活。

**⚠️ 局限性**

仍依赖对扰动变化率的估计误差，合成预测假设滑动窗口内可交换性；对极高频扰动响应可能不足；实验仅限于单一机器人平台，缺乏跨平台验证。

---

## 223. KidnapRAG: A Black-Box Attack for Hijacking Reasoning in Agentic Retrieval-Augmented Generation Systems

**arXiv ID:** 2607.00422 | [PDF](https://arxiv.org/pdf/2607.00422v1)

**作者:** Chanwoo Choi `[一作]` (Korea University), Buru Chang `[通讯]` (Korea University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了KidnapRAG，一种针对Agentic RAG系统的黑盒投毒攻击；

**💡 创新点**

创新点在于通过三类顺序投毒文档（Bait、Chain‑Link、Mal‑Ins）逐步劫持多步推理链，而非单步注入；

**🔧 技术方法**

使用检索优化、关键词改写、意图劫持等技术生成投毒文档，并利用现有检索器(e5‑large‑v2)和LLM后端；

**📊 数据集**

在HotpotQA、MuSiQue和2WikiMultihopQA三大问答基准上进行实验；

**📈 对比分析**

与七种黑盒基线（Naive、Ignore、Fake Completion、Combined、TopicAttack、PoisonedRAG、PARADOX）对比，KidnapRAG在多框架、多模型下实现最高攻击成功率和最大EM下降；

**⚠️ 局限性**

局限包括：需依赖系统公开的中间推理信息；投毒文档在真实网络环境中的检索成功率受索引、源权重、审查等因素影响；仅评估了ReAct和WebThinker两种架构，未覆盖更复杂的规划/验证/多智能体系统；

---

## 224. A Mechanistic View of Authority Hierarchy in LLM Sycophancy

**arXiv ID:** 2607.00415 | [PDF](https://arxiv.org/pdf/2607.00415v1)

**作者:** Emil Joswin `[一作]` (Independent Research), Priyanka Mary Mammen `[通讯]` (University of Massachusetts Amherst)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究大型语言模型在医学问答场景下受到不同专业级别权威提示影响的表现，探究其内部机制与知识消除过程。

**💡 创新点**

首次将权威层级视为隐式内在结构，揭示权威提示在深层激活中导致目标答案被主动擦除而非简单抑制的机理。

**🔧 技术方法**

采用logit lens追踪概率变化、线性与非线性探测器分析表示层、向量插值实验及链式推理（Chain‑of‑Thought）对比。

**📊 数据集**

MedQA‑USMLE医学许可考试数据，构造四级专业层级（医学生一年级、三年级、住院医师、执业医师）权威提示。

**📈 对比分析**

在Llama‑3.1‑8B、Qwen3‑8B与Gemma‑2‑9B三模型中，权威最高的执业医师提示将模型准确率从约60%降至15–34%，表明权威影响按层级递减且显著。

**⚠️ 局限性**

研究仅覆盖医学领域，使用的模型规模有限，提示结构固定，未探究不同语言或更大模型的普适性，且对权威层级外的其他社会偏差尚未验证。

---

## 225. MedCAGD: Context-Aware Gated Decoder for Efficient Medical Image Segmentation

**arXiv ID:** 2607.00409 | [PDF](https://arxiv.org/pdf/2607.00409v1)

**作者:** Saad Wazir `[一作]` (Korea Advanced Institute of Science and Technology), Daeyoung Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MedCAGD（Context-Aware Gated Decoder）框架，针对医学图像分割任务在解码阶段引入全局上下文注入、空间竞争门控跳连和多尺度上下文聚合等机制，显著提升像素级预测质量。

**💡 创新点**

创新点主要包括：① 通过 ECA-MSP 进行多尺度通道注意力调制；② 采用 SCA‑Gate 对跳连特征进行空间竞争与全局调制，形成自适应的跳连门控；③ 设计了 Context Aggregator（CA）将多尺度编码器语义以残差形式注入解码器；④ 在瓶颈处引入 Global Context Injection（RA）使深层特征更具全局语义一致性；⑤ 通过深度监督和边缘监督进一步提升边界精度。

**🔧 技术方法**

使用的技术包括：轻量化多尺度通道注意力（ECA‑MSP）、空间竞争注意力门（SCA‑Gate）、残差全局注意力（RA）、多尺度上下文聚合（CA）、局部细化模块（Refinement Block），以及基于预训练 PVTv2‑B2 以及其他 Timm 预训练编码器的通用特征投影。

**📊 数据集**

在 11 个公开医学图像分割基准上进行评估，覆盖了 2D 皮肤病、息肉、视网膜、肿瘤、细胞、肺部多器官（Synapse）、心脏（ACDC）等多种模态和任务。

**📈 对比分析**

与 CNN、Transformer、Mamba、SAM 等多种主流方法（如 U‑Net、Attn‑UNet、DeepLabv3+、nnU‑Net、PraNet、TransUNet、Swin‑UNet、UCTransNet、EMCAD、MCADS 等）在 Dice、IoU、HD95 等指标上进行对比。MedCAGD 在大多数数据集上均实现了显著提升（如平均 Dice 最高提升 1–3%，HD95 下降 15–30%），并在参数与 FLOPs 上保持与现有 SOTA 接近甚至更优的效率。

**⚠️ 局限性**

局限性包括：目前仅针对 2D 分割任务；缺乏对 OOD、跨域适应与 3D 扩展的系统评估；对编码器的依赖仍较高，需使用预训练模型；在极大尺寸图像（>512×512）上仍面临显著的计算与内存开销。

---

## 226. Personalization as Inverse Planning: Learning Latent Design Intents for Agentic Slide Generation via Structural Denoising

**arXiv ID:** 2607.00407 | [PDF](https://arxiv.org/pdf/2607.00407v1)

**作者:** Tianci Liu `[一作]` (Purdue University), Wei-Ting Chen `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向页面级幻灯片个性化（PSP）的逆规划框架，通过学习潜在设计意图指导幻灯片生成

**💡 创新点**

利用结构去噪（Structural Denoising）构造可验证的自监督任务，将不可导的渲染似然替换为可评估的重建目标，并证明多代理结构显著降低了策略梯度方差

**🔧 技术方法**

结合多模态语言模型、强化学习、结构扰动与去噪、Critic‑Planner 双代理框架，以及LLM判别器做回报信号

**📊 数据集**

使用 Zenodo10k、SlideBench 等公开幻灯片数据集，随机抽取 200 组 deck，进行训练与 OOD 测试

**📈 对比分析**

与 AutoPresent、PPTAgent、Stable Diffusion 等基线对比，在视觉相似度（SSIM、CLIP）和 VLM‑judge 的多维度评估中，所提方法在 judge 分数与视觉指标上均优于 GPT‑based 7B 级基线，且在 OOD 场景表现尤为突出

**⚠️ 局限性**

仍受限于基于参考幻灯片的偏好推断、对执行器的黑盒假设、以及结构扰动设计的覆盖范围，且在多轮迭代中可能导致样本效率低下

---

## 227. TVA: A Version-aware Temporal Graph Storage System for Real-time Analytics

**arXiv ID:** 2607.00406 | [PDF](https://arxiv.org/pdf/2607.00406v1)

**作者:** Wenhao Li `[一作]` (Renmin University of China), Xiaoyong Du `[通讯]` (Renmin University of China)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种版本感知的时序图存储系统，采用多版本存储架构、时序表与增强跳跃哈希，显著降低时序查询延迟和存储占用；

**💡 创新点**

创新点在于：①把版本元数据与实际数据解耦，使用时序表快速定位版本；②改进跳跃哈希算法，使同一边的所有版本按时间顺序紧凑存放并限制跳跃距离；③提出版本跳过策略与SIMD加速，复用前一次扫描的时序信息；

**🔧 技术方法**

主要技术包括：多版本列式存储、时序表、增强跳跃哈希（Hopscotch‑based Hash Table）、版本链跳过、SIMD并行查询、写前日志与两种持久化模式；

**📊 数据集**

实验使用公开图数据集：IMDB、DBLP、YouTube、Epinions、Pokec、LDBC SNB，以及基准集T‑mgBench、T‑LDBC、T‑gMark；

**📈 对比分析**

与Clock‑G、T‑GQL、AeonG、PostgreSQL、RocksDB、GraphOne、Stinger、Sortledton等系统比较，实验表明系统在存储占用上比AeonG、Clock‑G、T‑GQL低1.2~4.7倍，在时序查询延迟上比最佳对手低9.9倍，整体性能提升可达两百多倍；

**⚠️ 局限性**

局限性包括：在极低度顶点的静态图中预分配空间导致一定浪费；单机设计不支持分布式扩展；版本链长时的跳跃哈希仍可能出现局部热点导致性能下降；

---

## 228. Child Safety in Generative AI: An Expert-Guided and Incident-Grounded Evaluation Framework

**arXiv ID:** 2607.00395 | [PDF](https://arxiv.org/pdf/2607.00395v1)

**作者:** Haein Kong `[一作]` `[通讯]` (Rutgers University), Haein Kong (Rutgers University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于专家指导和AI事故数据库的儿童安全评估框架，并在教育领域生成合成测试集评估Llama Guard模型的安全检测性能。

**💡 创新点**

将专家制定的儿童安全风险分类与真实AI事故数据相结合，自动生成针对不同风险的合成测试集，提升生态有效性和可扩展性。

**🔧 技术方法**

利用专家指南构建风险分类法、Mistral-7B生成合成提示、Llama Guard模型进行二分类评估，并在Dyff平台上运行实验。

**📊 数据集**

使用APA、CSM、SAIFCA的儿童AI风险指南以及AIID、AIAAIC事故数据库，生成共130条（65安全+65不安全）的合成提示。

**📈 对比分析**

与三种规模的Llama Guard模型（1B、7B、8B）进行二分类比较，准确率约67-72%，召回率仅48-51%，表明模型在教育相关不安全提示上的表现较差。

**⚠️ 局限性**

仅覆盖教育风险，未扩展至所有风险类别；仅评估Llama Guard模型，缺少其他LLM；专家参与有限，缺乏教育专家直接参与定义不安全内容。

---

## 229. Ghost in the Kernel: In-Context Learning with Efficient Transformers via Domain Generalization

**arXiv ID:** 2607.00479 | [PDF](https://arxiv.org/pdf/2607.00479v1)

**作者:** Peilin Liu `[一作]` (University of Sydney), Ding-Xuan Zhou `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文通过两阶段采样的领域泛化框架，理论分析线性Transformer在上下文学习中的近似与泛化能力，证明其维度无关的收敛速率；

**💡 创新点**

创新点在于：①将上下文学习视为领域泛化的算子学习；②提出利用谱衰减与分布平滑的理论分析；③给出将预训练softmax LLM线性化的思路与公式；

**🔧 技术方法**

采用线性注意力机制、核嵌入、变换器结构、RMSNorm、ReLU/Tanh等激活与分布假设；

**📊 数据集**

未在论文中给出具体公开数据集，主要为理论推导与分析；

**📈 对比分析**

未进行实验比较，论文主要基于理论证明；

**⚠️ 局限性**

局限性：对谱衰减的假设与分布平滑参数γ的取值敏感；需要足够的上下文采样来逼近真实分布；线性化方法在实际token分布极度异质时可能失效；

---

## 230. Minos: A Multi-Agent Collaborative Framework for Provenance-Based Backward Tracking

**arXiv ID:** 2607.00440 | [PDF](https://arxiv.org/pdf/2607.00440v1)

**作者:** Jiahui Wang `[一作]` (Zhejiang University), Fan Zhang `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于多智能体协作的 LLM 驱动的逆向跟踪框架，用于从关键事件（POI）追溯并重建完整攻击链；

**💡 创新点**

创新点包括：1）层次化上下文模型（细粒度叙事 + 粗粒度战术序列）解决 LLM 状态与知识缺口；2）检索增强推理与引用验证提升解释性；3）对抗式推理（检察官‑辩护人‑法官）缓解 sycophancy 偏差；4）四个专门智能体（Planner、Query、Adversarial Group、Memory）通过 FSM 进行分工协同，配合“count‑first”协议和自纠代码机制，显著降低依赖爆炸与误检。

**🔧 技术方法**

技术手段主要包括：大型语言模型（GPT‑5.2、GPT‑5.2‑Codex 等）、检索增强生成（RAG）+引用验证、对抗式多智能体推理、LangGraph 进行多智能体调度、Neo4j 图数据库查询、层次化上下文管理、计数优先查询协议、代码生成自纠机制。

**📊 数据集**

使用公开的四大数据集共 14 个攻击场景：DARPA 透明计算数据集（Cadets、Trace、Theia）、Aurora（基于 LOLBAS 的隐蔽攻击模拟）、OpTC（企业级 APT 记录）。

**📈 对比分析**

与三类基线（频率统计 NoDoze、LDA 影响传播 DepImpact、单体 LLM Agent）以及多种 LLM 背骨组合进行对比。结果表明，本框架平均召回率 0.92、精准率 0.64，子图平均规模 35（真值 24），相比基线提升约 15‑30% 的召回和 2‑5 倍的精准率，同时子图更紧凑；在大规模噪声场景下保持高精度；总体推理耗时约 1.3k 秒，token 消耗 164K，较单体 LLM 低 25%。

**⚠️ 局限性**

局限性包括：1）对敏感审计日志的隐私要求仍需本地部署或模型蒸馏；2）依赖单一 POI，低质量警报可能导致误检；3）对多 POI 的联合追踪与自我可信度评估尚未实现；4）当前实现主要基于 GPT‑5 系列，需进一步评估更小模型的可迁移性。

---

## 231. PHREEQC-MCQ-200: A Diagnostic Benchmark for Tool-Augmented Scientific Simulator Agents

**arXiv ID:** 2607.00436 | [PDF](https://arxiv.org/pdf/2607.00436v1)

**作者:** Ke Zhang `[一作]` (University of California), Maziar Raissi `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PHREEQC-MCQ-200基准，用来评估工具增强LLM代理在确定性水相地球化学模拟上的性能，并提供诊断方法。

**💡 创新点**

① 设计了基于多选题的确定性模拟器评估基准；② 引入工具使用的增减/保留分析；③ 系统对比输出访问协议（TOC vs Raw）；④ 报告工具使用回归与模型能力层级相关的现象。

**🔧 技术方法**

LLM代理与PHREEQC 3.8.6的交互、Chain-of-Thought提示、Agent-TOC与Agent-Raw100k输出访问协议、工具使用框架（ReAct、Toolformer）等技术。

**📊 数据集**

PHREEQC-MCQ-200共200道多选题，来源于21个专家验证的PHREEQC情景，包含输入文件、输出文件、执行轨迹等。

**📈 对比分析**

对比直接无工具、CoT、TOC工具、Raw100k工具四种设置，评估准确率、token消耗、保留率与增减项。顶层模型在工具下提升15–41点，保持率56–86%；TOC在顶层模型可降低11–57% token并保持或提升准确率，中层模型则下降7.5–9.5点。

**⚠️ 局限性**

仅采用多选题评估，场景与PHREEQC数据库固定，未覆盖更广泛科学领域；输出尺寸依赖模型输入，难以普适；仅评估确定性模拟器，无法验证不确定性或实验驱动任务的表现。

---

## 232. Information-Regularized Attention for Visual-Centric Reasoning

**arXiv ID:** 2607.00434 | [PDF](https://arxiv.org/pdf/2607.00434v1)

**作者:** Guohao Sun `[一作]` (FAIR at Meta), Praveen Krishnan `[通讯]` (FAIR at Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种信息正则化注意力（IRA）机制，针对视觉语言模型中视觉嵌入的噪声注入与自适应正则化，提升视觉信息的可靠性与跨模态推理；

**💡 创新点**

创新点在于：①在注意力前对视觉value进行数据相关的随机采样与自适应噪声注入；②利用信息理论（互信息、KL）与不确定性加权的正则化，平衡视觉特征的压缩与保留；③证明注意力sink与表示曲率相关，提出利用曲率平滑提升模型稳定性；

**🔧 技术方法**

使用变分推理框架、可重参数化技巧、互信息正则化、可学习的高斯先验与后验、加权KL损失、注意力熵与重要性度量、曲率度量等技术；

**📊 数据集**

在InternVL2、InternVL2.5、LLaVA-OneVision等公开VLM模型上，使用单图像3.2M的训练集以及多图像、视频、OOV任务（MMEU、MMMU、MMStar、OK-VQA、TextVQA、ChartQA、DocVQA、EmbSpatial等）进行评估；

**📈 对比分析**

与标准SFT、SFT+meta背景、无IRA的对照组相比，IRA在推理、知识密集任务上提升约1–2点（如MMMU 0.5%，OK-VQA 0.5%），在多图像/视频理解和抗噪任务上提升约1–2点，同时显著降低attention sink比例；

**⚠️ 局限性**

受限于算力，只在≤8B参数模型上实验；未在更大规模模型上验证；并未在预训练阶段直接应用IRA，仍需进一步研究。

---

## 233. When Classic Cache Policies Fail: Learning-Augmented Replacement for Semantic Retrieval Buffers

**arXiv ID:** 2607.00394 | [PDF](https://arxiv.org/pdf/2607.00394v1)

**作者:** Yushi Sun `[一作]` (LIGHTSPEED, Tencent), Wai Lam `[通讯]` (Chinese University of Hong Kong)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对大型语言模型代理的检索缓冲区，提出并验证了一个学习增强的在线语义缓存替换框架 SOLAR。

**💡 创新点**

创新点在于将缓存管理转化为带切换成本的软命中在线问题，并通过基于累计后悔的修改时机和贝叶斯在线学习的内容选择，实现常数竞争比和近似最优的回溯收益。

**🔧 技术方法**

核心技术包括：累计误差阈值驱动的可变写入策略、贝塔后验+Thompson采样的淘汰决策、以及在线自适应阈值更新与时效衰减机制。

**📊 数据集**

实验数据集采用 MemoryBench‑Full 的 LoCoMo（约 2000 步对话）和 DialSim（约 19000 步电视对话），并在合成循环与工作集覆盖率上进一步验证。

**📈 对比分析**

与 FIFO、LRU、LFU、ARC 等经典策略以及无学习基线相比，SOLAR 在紧凑缓存（K≤50）下实现 5–75% 的相对 F1 提升，且相较 FIFO 在低容量场景下表现出 17% 的写入率和显著更快的学习速率；在大容量场景下则可与 FIFO 对齐。

**⚠️ 局限性**

局限性包括：仍需对阈值参数、后验初始先验及退化场景的调优；在极端噪声或高频主题切换时的稳健性尚待进一步研究。

---

## 234. Dual-Confidence Contrastive Decoding for Retrieval-Augmented Generation

**arXiv ID:** 2607.00570 | [PDF](https://arxiv.org/pdf/2607.00570v1)

**作者:** Raymond Li `[一作]` (ServiceNow Research), Issam H. Laradji `[通讯]` (ServiceNow Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于双重置信号（文档层和词元层）的无训练对比解码方法（Dual-Confidence Contrastive Decoding，DCCD），并基于企业深度研究场景构建了冲突式问答基准 DRQA。

**💡 创新点**

创新点在于将文档是否足够回答问题的置信度与该文档在当前生成步骤中产生尖锐下一个词元预测的置信度相结合，用这两种信号动态选择正负对比文档，从而在多文档检索增强生成（RAG）中有效解决检索上下文内部冲突。

**🔧 技术方法**

使用的技术包括：1）支持-探测提示（support probe）估计文档层置信度；2）基于 Dirichlet 分布的前置 k 词元置信度估计；3）对比解码公式将全上下文 logits 与正负文档条件 logits 按置信度差值进行加权；4）无训练、纯推理阶段实现。

**📊 数据集**

数据集：1）DRQA（由 DRBench 生成的合成企业深度研究问答集，包含正确、误导、时效、噪声四类文档）；2）公开多文档 QA 基准（Natural Questions、TriviaQA、PopQA、RetrievalQA），用于泛化评估。

**📈 对比分析**

与全上下文解码、CAD、AdaCAD、DVD、CoCoA 等基线相比，DCCD 在 DRQA 上获得最大提升（≈+12% F1），在其他基准也保持或略优的表现，尤其在检索文档多、冲突度高的场景中表现更突出。

**⚠️ 局限性**

局限性包括：1）推理时需对每个检索文档做前向传播，计算成本线性增长；2）仅覆盖事实型冲突问答，未扩展至长文本生成、多模态或交互式工具使用；3）基准为合成数据，缺乏真实企业记录，实用性受限。

---

## 235. [Preprint] Dynamic Modeling, Gait Synthesis, and Control of a Novel Subsurface Bore Propagator

**arXiv ID:** 2607.00569 | [PDF](https://arxiv.org/pdf/2607.00569v1)

**作者:** Lina van Brügge `[一作]`, George Nikolakopoulos `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种模块化地下钻探机器人，完成了其动力学建模、步态合成与反馈控制，并通过Unity+ROS仿真验证了其在土壤中前进的可行性。

**💡 创新点**

创新点在于将钻探头、锚定模块和推进模块解耦为独立模块，使用Euler‑Lagrange框架分别建模并设计分离控制器，随后通过中央状态机实现波纹步态合成，并实现从数学模型到真实机器人几何的sim‑to‑real集成。

**🔧 技术方法**

采用Euler‑Lagrange动力学建模、PID控制器、基于Unity的物理仿真、ROS接口以及中心化状态机步态生成技术。

**📊 数据集**

未使用公开数据集，采用机器人CAD几何、土壤力学参数（如Young模量、摩擦系数）以及实验中生成的自定义模拟数据。

**📈 对比分析**

通过Simulink验证模块动力学，Unity仿真完成3个完整步态循环后实现30 mm前进（理论可达60 mm），说明锚定力和步态能保持机器人在隧道壁上并实现有限前进。性能表现受滑移和安全阈值影响。

**⚠️ 局限性**

限制包括：BPM收缩时产生滑移导致前进距离不足；锚定力与BPM杠杆力矩不匹配；未实现三维运动和轨迹跟踪；实验结果仅在仿真环境中验证，缺乏完整的硬件实验数据。

---

## 236. ECoSim: Data Efficient Fine-Tuning for Controllable Traffic Simulation

**arXiv ID:** 2607.00545 | [PDF](https://arxiv.org/pdf/2607.00545v1)

**作者:** Yu-Hsiang Chen `[一作]` (National Yang Ming Chiao Tung University), Masayoshi Tomizuka `[通讯]` (University of California, Berkeley)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了数据高效微调框架，冻结预训练交通生成模型，通过轻量化FiLM适配器实现多模态可控生成（轨迹草图、潜在行为码、自然语言）。

**💡 创新点**

创新点在于身份初始化的FiLM适配器能在少量标注数据下为不同模型（扩散与自回归）增添可控接口，并结合上下文检索实现逆因果场景与稀有行为的合成。

**🔧 技术方法**

采用了FiLM特征调制、低秩适配器LoRA、BehaviorVAE、DistilBERT+LoRA、闭环微调以及上下文检索等技术。

**📊 数据集**

使用Waymo Open Motion Dataset（WOMD）进行训练和评估，并在Waymo Open Sim Agents Challenge（WOSAC）中进行测试。

**📈 对比分析**

与无条件基模型、全量数据LoRA微调以及ProSim相比，使用仅1%标注数据即可将mADE降低约60–80%，Meta分数提升0.02–0.03点，甚至匹配全量LoRA性能，且在多模态下保持稳定的现实性。

**⚠️ 局限性**

限制在于对冲突或模糊控制的处理不够理想，模型可能优先满足控制而牺牲安全；当前仅支持单代理控制，缺乏多代理协同控制能力。

---

## 237. Geometric Shape Optimization for Limbless Locomotion

**arXiv ID:** 2607.00524 | [PDF](https://arxiv.org/pdf/2607.00524v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 238. GEAR-Seg: A Grounded Explainable Agent for Reasoning Segmentation and Data Engine

**arXiv ID:** 2607.00544 | [PDF](https://arxiv.org/pdf/2607.00544v1)

**作者:** Yanan Wang `[一作]` (Zhejiang University), Zhenghao Fei `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并实现了 GEAR‑Seg——一种基于像素到文本的可解释推理分割代理，能够在不需要人工标注的情况下实现复杂隐式查询的目标分割，并通过自动化数据生成引擎构建了规模达 38k 张图像、656k QA‑Mask 对的 GEAR‑131K 基准。

**💡 创新点**

创新点包括：① 将视觉感知与逻辑推理显式拆分为 SAM‑2 生成无类别掩码、DAM 提供密集语义描述、LLM 进行多步推理，彻底摆脱黑盒耦合；② 通过同一框架实现可扩展的数据引擎与知识蒸馏，生成高质量标注并将推理能力迁移至轻量端侧模型；③ 设计了涵盖 Commonsense、Functional、Manipulation‑related、Part‑based、Attribute‑based 五大推理场景的多维度标签体系。

**🔧 技术方法**

核心技术包括：Segment Anything 2 (SAM‑2) 的 Everything Mode、Describe Anything Model (DAM) 的像素‑到‑文本描述、各种大型语言模型（Qwen‑3‑32B、Gemini‑3.1‑Pro、DeepSeek‑R1 等）驱动的多步推理、基于 LLaMA 3.2 的图像语义抽取、以及基于自动化生成标签的蒸馏训练策略。

**📊 数据集**

使用的数据集有：自研 GEAR‑131K（38k+图像、656k QA‑Mask）、公开 ReasonSeg、LLM‑Seg40k、农产品长尾数据集（StrawDI_Db1、Mega_Blueberry、Mega_Peach），以及原始图像来源（LVIS、VOC、Mapillary、ADE20K）。

**📈 对比分析**

在零样本推理任务上，GEAR‑Seg 的 gIoU 达到 57.5（ReasonSeg）/52.2（LLM‑Seg40k），略高于微调版 LISA‑13B（56.2/45.5）并且在尺寸归一化指标 ncIoU 上更具优势；在长尾水果实例分割中，零样本 mAP_50:95 提升 13.2–10.0–4.8 点；蒸馏实验表明，仅使用 GEAR‑Seg 生成的伪标注可恢复 93% 以上的人工标注性能，并在 GEAR‑131K 上引入逻辑链后 gIoU 提升至 37.2（对比 26.6）。

**⚠️ 局限性**

主要局限：① 模块化链条易出现级联误差，尤其是 LLM 推理和 DAM 语义翻译中的幻觉；② 对细粒度目标的感知仍受 SAM 生成掩码质量影响；③ 当前实现对资源需求较高（高分辨率图像需 34s 计算），不适合实时边缘推理，需进一步优化。

---

## 239. AI Native Games: A Survey and Roadmap

**arXiv ID:** 2607.00527 | [PDF](https://arxiv.org/pdf/2607.00527v1)

**作者:** Zhiyue Xu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jian Zhao `[通讯]` (Zhongguancun Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文定义了AI‑native游戏的严格边界，并对53款公开可玩游戏原型进行系统梳理，提出基于游戏类型（G）和核心AI机制（N）的双轴分类及未来发展路线图。

**💡 创新点**

创新点在于：①首次把AI‑native游戏与其他AI相关游戏形式（如PCG、AI‑augmented、chatbot等）区分开来；②提出基于核心玩法的可逆性判定；③构建双轴G/N分类框架，揭示现有设计热点与空白；④系统总结技术挑战与路线。

**🔧 技术方法**

主要技术依赖为大型语言模型（LLM）在游戏引擎中的运行时推理与生成、生成结果的结构化验证、提示工程与检索增强（RAG）等；同时结合传统游戏AI、PCG验证器与本地缓存。

**📊 数据集**

数据来源为公开可玩游戏原型与演示，未使用传统公开数据集，而是手工收集并编码53款AI‑native案例。

**📈 对比分析**

通过人工编码、描述性统计和交叉轴分析展示各类游戏与AI机制的分布，未给出量化性能指标，重点在分类与设计洞察。

**⚠️ 局限性**

局限主要包括：①以文本为主，缺乏多模态交互；②生成控制与一致性仍难，导致规则失效与不稳定；③对模型的依赖带来可复制性与监管挑战；④大部分案例处于早期或原型阶段，缺乏成熟的玩法与评估。

---

## 240. Large Language Models for Multi-Lingual Equivalent Mutant Detection: An Extended Empirical Study

**arXiv ID:** 2607.00511 | [PDF](https://arxiv.org/pdf/2607.00511v1)

**作者:** Honglin Shu `[一作]` (Tianjin University), Yasutaka Kamei `[通讯]` (Kyushu University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于大型语言模型（LLM）系统评估了等价变异体检测（EMD）在Java和C语言中的有效性与效率，并通过跨语言微调验证了其泛化能力。

**💡 创新点**

创新点在于首次完整对比13种前沿LLM与传统基线（编译器、机器学习、AST神经网络），并探讨了多种LLM适配策略（微调、提示、指令微调）及跨语言迁移效果。

**🔧 技术方法**

使用的技术包括Encoder/Decoder LLM（如CodeBERT、UniXCoder、CodeT5+）、文本嵌入模型、编译器优化、传统机器学习和AST神经网络，配合微调、零/少量示例提示以及指令微调等策略。

**📊 数据集**

实验数据来源于MutantBench数据集，包含3302个Java方法级变异体对和1088个C方法级变异体对，按50/50比例划分为训练集和测试集。

**📈 对比分析**

与10种基线对比，LLM在F1上平均提升约20%–80%；最优方案（Fine‑tuned UniXCoder/CodeT5+）在Java/C分别达81.9%/75.1%，推理时间约0.04/0.02秒，显著优于编译器且仅略慢于传统机器学习，证明其在实用性上的竞争力。

**⚠️ 局限性**

局限包括仅覆盖Java/C两种语言、数据集规模有限、跨语言迁移在语义差异较大的语言上的适用性未知、LLM存在灾难性遗忘以及对某些语义细节缺乏推理能力。

---

## 241. AnF-DiffPET: Anatomy- and Frequency-Guided Diffusion for PET/CT Denoising

**arXiv ID:** 2607.00509 | [PDF](https://arxiv.org/pdf/2607.00509v1)

**作者:** Xuepeng Liu `[一作]` (Northeastern University), Yueyang Teng `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 AnF-DiffPET，基于 CT 条件的低剂量 PET 去噪扩散框架

**💡 创新点**

创新点是将解剖信息与频域引导结合，构建 Anatomical‑Frequency Guidance、Multi‑Scale Cross‑Transformer Reconstruction 和 Frequency‑Contrastive Hard Mining 三大模块

**🔧 技术方法**

使用 MedDINOv3 提取 CT 解剖特征、频域注意力机制、跨尺度 Transformer、频域对比挖掘以及 1000 步线性噪声扩散模型

**📊 数据集**

在四个 PET/CT 数据集上验证：HECKTOR、PSMA、NaF、RIDER

**📈 对比分析**

与 CNN、GAN、Transformer、其他扩散方法对比，采用 PSNR、SSIM、SUV_bias 评价，AnF‑DiffPET 在所有数据集上均获得最高 PSNR/SSIM、最低 SUV_bias，提升显著

**⚠️ 局限性**

局限性：仅处理二维切片，未验证不同扫描仪/重建协议及真实低剂量 PET，未来需扩展到 2.5D/3D、更多临床条件与读者评估

---

## 242. When RAG Meets Query Planning: Logical Query Trees for Resolving Exploratory Reasoning Problems

**arXiv ID:** 2607.00508 | [PDF](https://arxiv.org/pdf/2607.00508v1)

**作者:** Ganlin Xu `[一作]` (Fudan University), Deqing Yang `[通讯]` (Fudan University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对复杂且不确定的探索式推理问题（ERP），提出了PlanRAG框架，将其拆解为原子查询后，利用动态规划构建逻辑查询树（LQT），并在LQT上并行执行检索与生成。

**💡 创新点**

创新点在于：① 把ERP视为结构化的查询树；② 结合数据库查询规划中的动态规划和多维成本模型实现全局规划；③ 通过关系预处理、循环防止和上下文感知合并降低LLM调用与检索噪声。

**🔧 技术方法**

核心技术包括：自然语言拆解提示（LLM生成原子查询）、动态规划（Selinger式DP）、多维成本评估（树大小、密度、深度、平衡、语义相似度）、并行执行与多线程调度、检索器BGE/Contriever/ColBERT。

**📊 数据集**

主要使用了新构建的WikiWeb-ERP数据集（3536个ERP问答对、53682个文档），并在多跳问答基准（HotpotQA、MuSiQue、2WikiMultihopQA、StrategyQA）上进行额外验证。

**📈 对比分析**

与DirectLLM、NaiveRAG、RetGen、GenGround、KiRAG、DualRAG、HopRAG、ChainRAG、LEGO-GraphRAG等迭代式和图式RAG方法对比，PlanRAG在WikiWeb-ERP上Acc、F1、EM、Acc†均领先，最高提升约6–8个百分点；在多跳QA基准上也保持竞争力，特别是StrategyQA上取得最优成绩。

**⚠️ 局限性**

主要限制包括：对简单问题无明显优势；成本模型为启发式，未完全反映检索与推理真实耗时；LLM在原子查询生成与关系判断上仍易出错，需进一步优化或用轻量模型替代；缺乏物理执行层面（如缓存）完整系统优化。

---

## 243. PAPA: Online Personalized Active Preference Alignment

**arXiv ID:** 2607.00486 | [PDF](https://arxiv.org/pdf/2607.00486v1)

**作者:** Anindya Sarkar `[一作]` (Washington University in St. Louis), Yevgeniy Vorobeychik `[通讯]` (Washington University in St. Louis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在线、反馈高效的扩散模型微调框架 PAPA 与其加速版 EPAPA，用以在没有奖励模型的前提下直接对用户偏好进行主动对齐。

**💡 创新点**

创新点包括：①利用变分推断直接从二进制用户反馈估计后验并构造包含非偏好样本的对齐损失；②引入质量保持与多样性增强项使模型在探索与利用之间可控；③提出只在高噪声阶段微调、低噪声阶段使用预训练模型的 EPAPA 方案，显著降低计算成本并提升样本质量。

**🔧 技术方法**

技术手段主要是扩散模型（DDPM/Stable Diffusion）、变分 KL 最小化、在线强化学习框架、以及预训练与微调模型的混合采样策略。

**📊 数据集**

实验数据集包括 MNIST、Fashion‑MNIST、CIFAR‑10 以及 Stable Diffusion v1.5 生成的图像；评估也使用 LAION 美学评分器、Image Reward 等自动化偏好标注工具。

**📈 对比分析**

与基线（预训练基线、D3PO 等）相比，PAPA/EPAPA 在成功率(SR)、FID 与 IS 指标上均显著提升，尤其在多类别偏好、压缩率、审美质量以及 Prompt‑Image 对齐任务中展现更快的收敛与更高的样本多样性。

**⚠️ 局限性**

局限性包括：仅针对二进制偏好反馈；对 K（高噪声阈值）的选择敏感；在极端高噪声或低噪声下可能出现过拟合或性能下降；目前仅在图像生成任务上验证，未在药物发现或材料设计等领域进一步测试。

---

## 244. Know When to Stop: Segment-Level Credit Assignment for Reducing Overthinking

**arXiv ID:** 2607.00482 | [PDF](https://arxiv.org/pdf/2607.00482v1)

**作者:** Chia-Hsuan Lee `[一作]` (Capital One), William Campbell `[通讯]` (Capital One)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于中间答案检查点的段落级信用分配方法，称为DASH，用于减少推理语言模型的过度思考（overthinking）行为。

**💡 创新点**

创新点在于利用模型自身在推理过程中的答案承诺（intermediate answer commitments）作为免费监督信号，将推理轨迹按答案漂移（drift）划分为正向和负向段落，并对每个段落分配不同的优势（advantage）来鼓励正确推理、惩罚错误反思。

**🔧 技术方法**

技术包括：GRPO（Group Relative Policy Optimization）强化学习框架、答案漂移检测、段落级优势分配、长度惩罚权重、以及一组无需学习的语言特征（如重复、保留、放弃、矛盾、再计算、长度异常）来评估推理质量。

**📊 数据集**

使用的主要数据集为OpenR1-Math-220K（包含16.5K数学推理问题及其已验证的推理轨迹），以及四个竞争级数学基准：OlympiadBench、AMC 2023、AIME 2024 和 AIME 2025。

**📈 对比分析**

与基线模型、标准GRPO、DR-GRPO（去偏GRPO）以及GRPO+简短奖励（brevity bonus）进行对比。DASH在最困难的AIME 2025基准上取得最高准确率（50.8%），并在过度思考行为（如重复、放弃、长度异常）上表现最优，同时保持更高的自我纠错率。

**⚠️ 局限性**

局限性包括：1) 只在可提取答案检查点的任务（如数学推理）适用；2) 对更大规模模型的适用性尚未验证；3) 在一些易题集上略有准确率下降；4) 仅评估数学推理，无法直接推广至开放式推理或其他领域。

---

## 245. Beyond the Prompt: Jailbreaking Function-Calling LLMs via Simulated Moderation Traces

**arXiv ID:** 2607.00481 | [PDF](https://arxiv.org/pdf/2607.00481v1)

**作者:** Junlong Liu `[一作]` (Sun Yat-sen University), Xiaojun Jia `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了基于Simulated Moderation Traces（SMT）的多轮函数调用破解框架，利用安全拒绝被误解为执行错误，逐步削弱模型安全约束并输出有害内容。

**💡 创新点**

突破传统提示级破解的局限，首次将安全拒绝转化为可被利用的执行失败，通过累积的函数调用状态和模拟验证回溯，实现对功能调用LLM的高效连续攻击。

**🔧 技术方法**

采用黑盒API交互、函数调用、红队测试上下文、模仿审计流程、异常反馈注入和多轮迭代改进等技术手段。

**📊 数据集**

在SafeBench‑Tiny和JailbreakBench这两套安全基准数据集上进行评估。

**📈 对比分析**

与八个2024‑2026年最新基线对比，在六个商业模型上平均攻击成功率接近100%（99.7%/98.3%），平均HarmScore分别为74/76，平均查询次数仅约1.4次，显著优于所有现有方法。

**⚠️ 局限性**

仅适用于支持函数调用接口的场景，对专门针对函数调用的防御（如FCGuard）易被阻断；并且依赖当前模型的安全实现，更新后可能失效。

---

## 246. Rise From The Ashes: LLM-based Static Analysis for Deep Learning Framework Bugs

**arXiv ID:** 2607.00555 | [PDF](https://arxiv.org/pdf/2607.00555v1)

**作者:** Shaoyu Yang `[一作]` (Nanjing University), Zhenyu Chen `[通讯]` (Nanjing University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过大语言模型生成的语义桥中间表示（SBIR），实现对深度学习框架跨语言张量语义错误的静态检测。

**💡 创新点**

提出跨语言张量语义桥的SBIR抽象，并设计多代理LLM工作流（摘要、提取、生成、分析）在无需运行时的情况下发现跨语言张量语义 bug。

**🔧 技术方法**

使用大语言模型（如GPT-5.4）、多模态检索、结构化语义中间表示以及基于SBIR的静态分析技术。

**📊 数据集**

构建包含40条历史 PR 修复与16条 CWE 规则的56条 bug 知识条目作为训练与检索的知识库。

**📈 对比分析**

与 Bandit（Python）和 Clang Static Analyzer（C/C++/CUDA）比较，误报率13.89%，发现31个真实 bug；与两款 LLM 驱动的 fuzzers 比较，独立发现30个 bug，整体覆盖率显著提升。

**⚠️ 局限性**

仅关注张量语义层面，依赖 LLM 的可解释性与稳定性，对极其复杂的多层调用路径可能仍产生漏报或误报；SBIR 语法需要手工维护与扩展。

---

## 247. EgoGapBench: Benchmarking Egocentric Action Selection in Multi-Agent Scenes

**arXiv ID:** 2607.00547 | [PDF](https://arxiv.org/pdf/2607.00547v1)

**作者:** Jihyeok Jung `[一作]` (KAIST AI), Seong Joon Oh `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了EgoGapBench基准，用以评估多智能体场景下的视角动作选择能力（Egocentric Action Selection, EAS）。

**💡 创新点**

创新点在于将第一人称视角理解与视角动作选择分离，使用无第一人称视觉线索的多智能体图像，并通过单例动作干扰项来检测自我与他者区分。

**🔧 技术方法**

使用大型视觉语言模型（MLLM）如Gemma、Qwen、InternVL、GPT等在该基准上进行零样本评估，并对比使用LoRA微调进行EAS特定和传统第一人称微调。

**📊 数据集**

数据集基于COCO 2017的多人物图像，人工标注生成正确动作、离场动作与单例干扰动作，另外构造107条Wikimedia域外样本。

**📈 对比分析**

结果显示人类准确率94.5%，但最强模型GPT-5.4仅达到66.1%；传统第一人称微调反而降低EAS性能，EAS微调提升至最高约75%，仍远低于人类。

**⚠️ 局限性**

局限在于基准只评估视角动作选择，缺乏更广泛的视角理解测度；模型仍受其他智能体动作诱导，且EAS微调难以完全消除视角误判。

---

## 248. Flow-Map GRPO: Reinforcement Learning for Few-Step Flow-Map Generators via Anchored Stochastic Composition

**arXiv ID:** 2607.00535 | [PDF](https://arxiv.org/pdf/2607.00535v1)

**作者:** Zhiqi Li `[一作]` (Georgia Institute of Technology), Bo Zhu `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于强化学习的后训练框架 Flow-Map GRPO，用来对预训练的确定性少步流图生成模型（如 MeanFlow、sCM）进行任务级奖励对齐，且不改变原模型参数化。

**💡 创新点**

创新点在于提出 Anchored Stochastic Flow Map Composition (ASFMC)，一种利用锚点的条件重采样来为长距离流图注入随机性，同时严格保持原有概率路径，从而实现可用于策略梯度优化的随机策略；并将该机制与 GRPO 结合，给出统一的目标函数，适用于单时点与双时点流图。

**🔧 技术方法**

核心技术包括：
- ASFMC 随机化构造（局部锚点、终点锚点、介质锚点三种实现），
- GRPO 风格的策略梯度优化（含 KL 正则、收益归一化），
- 对两种流图参数化（MeanFlow 的两时点流图和 sCM 的单时点流图）进行专门设计的采样和奖励评估。
- 在训练中使用 LoRA Adapter 仅微调权重，保持原模型不变。

**📊 数据集**

实验使用 FLUX‑based 的 T2I‑Distill 预训练模型（MeanFlow 与 sCM），在三种奖励下分别进行后训练：PickScore、OCR accuracy 与 GenEval。评估数据集包括 DrawBench（通用文本生成评测）与标准 T2I‑Distill 的 Prompt 集。

**📈 对比分析**

对比方法：直接使用预训练的 MeanFlow / sCM（无后训练）与 Flow‑Map GRPO 后训练模型。性能指标涵盖任务级奖励（PickScore、OCR、GenEval）、感知质量（Aesthetic、ImageReward 等）以及 DrawBench 的综合得分。结果表明：
- 在 PickScore 与 OCR 目标下，Flow‑Map GRPO 通常能显著提升对应指标（如 PickScore 从 20.6→23.4、OCR 从 21.4→24.1）。
- 在 GenEval 目标下，虽然总体提升不如 PickScore/ OCR 明显，但在大部分采样步数上仍保持或略有提升。总体来看，Flow‑Map GRPO 在多数指标上均优于基线，尤其在高采样步数（15 步）时更为显著。

**⚠️ 局限性**

局限性包括：
- ASFMC 的锚点选择对性能敏感，尤其是介质锚点在实验中易导致不稳定；
- 仅在文本生成任务上验证，未探讨对视频或多模态生成的适用性；
- 由于仅微调 LoRA，可能对大规模参数的可扩展性有限；
- 对于极小步数（如 2 步）时，性能提升有限；
- 目前缺乏对收敛性和理论分析的深入讨论。

---

## 249. From Technical Metrics to User Perception: A User Study of a Multimodal Human-Robot Interaction System for Object Detection and Grasping

**arXiv ID:** 2607.00530 | [PDF](https://arxiv.org/pdf/2607.00530v1)

**作者:** Jian Song `[一作]` (Dalian University of Technology), Shen Guanting `[通讯]` (Dalian University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对同一桌面抓取任务，进行了一项 within-subject 的用户研究，比较基线多模态 HRI 系统与改进配置在用户感知上的差异。

**💡 创新点**

首次证明技术层面的 15% 任务成功率提升能够被非专业用户在实时交互中感知，并将技术评估与用户主观感知相结合，提出了一种多模态系统的用户感知评估方法。

**🔧 技术方法**

基线使用 Whisper（语音识别）、Florence-2（开口视觉检测）、LLaMA 3.1（意图抽取）以及 Interval Type‑2 Fuzzy Logic 控制器；改进系统替换为 Qwen 3.5 9B（意图抽取）与 Grounding DINO + SAM（视觉检测），控制器保持不变。

**📊 数据集**

实验采用打印水果图像作为物体，配合自然语言语音指令；数据集为实验室内的固定桌面场景与预设命令集。

**📈 对比分析**

采用 within-subject 设计，每位参与者体验两种配置后完成 7‑点 Likert 量表问卷；统计方法为配对 t 检验、Wilcoxon 符号秩检验及 Holm 校正。结果显示改进系统在速度（d_z = 1.85）、可靠性（d_z = 1.54）和整体能力（d_z = 2.03）上均显著优于基线，且 70.8% 的参与者偏好改进系统。

**⚠️ 局限性**

局限性包括样本量仅 24 人、实验环境受限于打印物体与静态场景、任务简短、缺乏真实物体、动态变化与多目标操作、长期使用效果未评估、以及对不同用户群体（如机器人专家与新手）的适用性未知。

---

## 250. NoPA: Non-Parametric Online 3D Scene Graph Generation

**arXiv ID:** 2607.00529 | [PDF](https://arxiv.org/pdf/2607.00529v1)

**作者:** Qi Xun Yeo `[一作]` (National University of Singapore), Gim Hee Lee `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个在线 3D 场景图生成框架 NoPA，利用固定粒子集对每个对象进行非参数建模，并在实时条件下实现对象融合与关系传播。

**💡 创新点**

创新点包括：① 用粒子集取代单一高斯，保留几何细节；② 引入最大均值差异（MMD）作为分布级合并判据，提升跨视角关联鲁棒性；③ 结合 Hellinger 前滤波降低计算开销；④ 在高相似度簇内传播关系，弥补 2D 预测缺失。

**🔧 技术方法**

技术手段包括：RT‑DETR‑EGTR 2D 场景图检测、深度反投影生成 3D 粒子、RBF 核密度估计、Hellinger 距离预过滤、MMD 计算与阈值判定、固定粒子重采样、关系聚类与多数投票。

**📊 数据集**

使用的评测数据集为 3DSSG（1482 场景）和 ReplicaSSG（18 场景），两者均含 RGB‑D 图像与 3D 场景图标注。

**📈 对比分析**

与 FROSS、JointSSG、Kim 等基线进行对比；NoPA 在 3DSSG 上实现 53.2% 关系召回、69.0% 对象 mRecall、61.4% 关系召回、66.4% 关系精度，显著优于同类在线方法，同时保持 30–35 ms 的实时延迟与 1206 MB 的显存。

**⚠️ 局限性**

局限性：高度依赖 2D 场景图检测质量；粒子数固定时可能出现粒子覆盖不足或过度融合；在复杂纹理缺失或视角极端变化时仍可能出现误合并。

---

## 251. GenSP: Consistent Spherical Parameterization via Learning Shape Generative Models

**arXiv ID:** 2607.00492 | [PDF](https://arxiv.org/pdf/2607.00492v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 252. AI, Trust, and Teaming: The Humans-as-Handlers Approach for Autonomous and Opaque AI Systems

**arXiv ID:** 2607.00523 | [PDF](https://arxiv.org/pdf/2607.00523v1)

**作者:** Nathan G. Wood `[一作]` `[通讯]` (Technical University of Hamburg), Nathan G. Wood (Technical University of Hamburg)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出并论证将人机团队视为人-动物类比中的“处理者-被处理者”关系，以建立信任与责任的框架。

**💡 创新点**

创新点在于将人-动物团队的处理者角色迁移至AI系统，并区分人对AI的信任维度，提出从训练到部署全过程的“人‑AI处理者”协同方法。

**🔧 技术方法**

主要采用哲学/伦理学对信任的多维度分析、案例研究与比喻方法，未使用传统机器学习技术或具体算法。

**📊 数据集**

未使用实验数据集；论述基于文献综述与理论分析。

**📈 对比分析**

无实验对比；未给出性能指标或定量评估。

**⚠️ 局限性**

局限在于缺乏实证验证、难以量化信任度、比喻的适用范围有限以及在不同文化/组织背景下的可推广性不确定。

---

## 253. A Geometric View of Combinatorial Fiedler Theory

**arXiv ID:** 2607.00519 | [PDF](https://arxiv.org/pdf/2607.00519v1)

**作者:** José Fernández Goycoolea `[一作]` (Universidad de Magallanes), Carlos Seara `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在组合Fiedler理论基础上引入并研究了ℓ₁‑最大化参数B(G)，并给出了其精确公式B(G)=½(d₁+d₂)，其中d₁,d₂为图中最高两个度数。

**💡 创新点**

主要创新在于：①给出B(G)的闭式表达；②将最小化与最大化问题统一为几何（超立方体）和组合（准二分）框架；③证明在一般图中计数达到B(G)的ℓ₁‑Fiedler向量是#P‑完整的；④为若干典型图族提供了ℓ₁‑Fiedler向量的计数公式。

**🔧 技术方法**

使用了组合优化（最小化/最大化）、凸几何（ℓ₁ 单位球交平面即笼罩多面体）、图割与相对割大小、以及多面体的细胞分解与Minkowski和。

**📊 数据集**

本文为理论性工作，未使用实验数据集，而是针对无向简单图族（Kₙ、Wₙ、Cₙ、Pₙ、K_{n,m}等）给出解析计数结果。

**📈 对比分析**

与现有的最小化 ℓ₁‑Fiedler 参数b(G)（与稀疏切割相关）的 NP‑难性对比，本文表明虽然B(G)的取值可在多项式时间内计算，但对应的计数问题仍为#P‑完整，体现出不同的复杂度特征。

**⚠️ 局限性**

局限性包括：①仅处理无向简单图；②虽然给出了B(G)的公式，但对一般图的ℓ₁‑Fiedler向量结构仍缺乏完整的描述；③计数问题的#P‑完整性表明没有多项式算法可在所有图上有效计数；④未讨论加权图、稠密图或有向图的推广。

---

## 254. Certificate-Carrying Transformation of Event-Driven Block Programs

**arXiv ID:** 2607.00563 | [PDF](https://arxiv.org/pdf/2607.00563v1)

**作者:** Yuan Si `[一作]` (University of Waterloo), Jialu Zhang `[通讯]` (University of Waterloo)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为 Scratch 等块式语言的程序提供了可证书化的源到源重写工具，使用受信任的检查器在显式观测透镜下验证重写是否保持行为。

**💡 创新点**

创新点在于将行为保持证明转为“证书携带”重写，利用可观测透镜参数化，并在合作调度下使用协作帧定理实现死写消除，同时在 Lean 中机械化证明。

**🔧 技术方法**

使用了证书携带重写框架、可观测透镜、协作帧定理、Lean 机械化、Python 检查器、等价饱和优化器以及差分 Oracle。

**📊 数据集**

采用了公开的 Scratch 项目仓库中的 300 个随机采样项目（并在攻击测试中使用 400 个额外项目）。

**📈 对比分析**

通过对比未认证优化、单独测试、跨族攻击、差分 VM 校验和消除侧条件的消融，证明检查器在 94.3% 的项目上接受了有效重写，平均每个项目的检查成本不到 0.1 秒，且不存在错误接受。

**⚠️ 局限性**

局限性包括：检查器实现的潜在实现错误可能导致漏洞；差分 Oracle 只能验证可观察行为，无法覆盖所有细粒度差异；框架假设 Scratch VM 与模型一致，且仅适用于合作调度的事件驱动语言。

---

## 255. Towards Better Linux Kernel Fault Localization: Leveraging Contrastive Reasoning and Hierarchical Context Analysis

**arXiv ID:** 2607.00562 | [PDF](https://arxiv.org/pdf/2607.00562v1)

**作者:** Haichi Wang `[一作]` (Tianjin University), Zan Wang `[通讯]` (Tianjin University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种面向 Linux 内核的 LLM‑based 故障定位技术，利用对比推理和分层上下文分析来精确定位根因。

**💡 创新点**

创新点在于：①对比推理（Contrastive Reasoning）通过生成通过与不通过的测试变体来识别根因；②分层上下文分析（Hierarchical Context Analysis）先按文件级定位，再细化到方法级；③在 LLM 交互中使用链式思考提示，显著降低 token 消耗。

**🔧 技术方法**

技术手段包括：大型语言模型（DeepSeek‑V3 等）+ 生成式提示（CoT）、基于语义的测试变体生成、缺陷预测 MLP、IR 相关性扩展、语义过滤与对比推理。

**📊 数据集**

使用数据集：210 个已验证的 Linux 内核缺陷（扩展版），以及 SWE‑bench‑lite（非内核）用于泛化验证。

**📈 对比分析**

与传统 SBFL、MBFL、IRFL 以及 LLM‑based 诸如 Agentless、LinuxFL+、SoapFL 等基线对比，Top‑1 文件级准确率提升高达 26.07%、方法级提升 56.85%；Token 使用量分别减少 8.84× 与 28.9×，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括：依赖 LLM 的预训练知识可能导致数据泄露风险；对极端复杂或多线程内核缺陷的根因推理仍有挑战；目前定位精度仍以文件/方法为粒度，未实现行级精确定位。

---

## 256. Cross-Domain Generalization Failure in Lightweight Intrusion Detection Models for IIoT Networks

**arXiv ID:** 2607.00553 | [PDF](https://arxiv.org/pdf/2607.00553v1)

**作者:** MD Azizul Hakim `[一作]` (Bangladesh Sweden Polytechnic Institute), Talha Ibne Anis `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在Edge‑IIoTset上训练四种轻量级模型（决策树、MLP、1D‑CNN、LSTM），随后在两大不同IIoT网络（Gotham 2025 与 WUSTL‑IIoT‑2021）上进行零样本推断与评估，并对模型对端口分类快捷方式的依赖进行了可解释性分析；

**💡 创新点**

创新点在于系统性验证轻量级模型跨网络、跨采集方式的泛化能力；通过SHAP与端口桶特征的对照实验揭示了“端口分类快捷方式”是跨域失败的主要原因；同时比较了平衡与自然类别分布下的性能差异，并展示了少量目标域数据恢复性能的架构依赖性；

**🔧 技术方法**

采用决策树、三层MLP、1D‑CNN、LSTM四种轻量级网络，并利用SHAP解释特征重要性、HopSkipJump黑盒对抗攻击评估鲁棒性，结合F1、准确率、精确率与召回率等多维度指标；

**📊 数据集**

使用Edge‑IIoTset（源训练集）、Gotham 2025（目标网络1）和WUSTL‑IIoT‑2021（目标网络2）三大公开IIoT流量数据集；

**📈 对比分析**

在统一的16维共通特征空间下，无重训练跨域评估，并在平衡与自然分布两种采样下比较性能；结果显示零样本跨域F1从0.97降至0.09–0.28，各模型跨域排名不一致；少量目标域样本微调后，恢复效果高度依赖于模型架构；

**⚠️ 局限性**

局限性包括：共通特征仅包含端口桶、协议与TCP标志，缺失对WUSTL‑IIoT‑2021的TCP标志信息；仅在Gotham上评估少样本恢复，未覆盖WUSTL‑IIoT‑2021；对抗攻击评估仅基于100样本且未复现多种种子；未对所有模型进行完整的可解释性分析；端口桶依赖的通用性未在更广泛数据集上验证；

---

## 257. SPECSIA: Stylization Dataset for Novel-View Enhancement in Drawing-based 3D Animation

**arXiv ID:** 2607.00525 | [PDF](https://arxiv.org/pdf/2607.00525v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 258. AI-Centered Grand Challenges in Visual Analytics for Healthcare: Synthesizing the VAHC 2025 Community Experience

**arXiv ID:** 2607.00542 | [PDF](https://arxiv.org/pdf/2607.00542v1)

**作者:** Jürgen Bernard `[一作]`, Alessio Arleo `[通讯]` (Eindhoven University of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对VAHC 2025工作坊的15篇论文及讨论进行主题编码与聚类，系统识别出五大AI中心挑战群——偏差与信任、数据与基础设施、可解释性与沟通、人机交互以及模型可靠性与验证。

**💡 创新点**

创新点在于将多年工作坊经验与结构化文本编码相结合，形成可复制的流程来挖掘社区关注点，并将挑战映射到现有文献与案例，提出以“校准”视角为核心的研究议程。

**🔧 技术方法**

采用手工编码、相似性聚类、反思性主题分析以及小组讨论记录等定性技术，对可视化工具与AI方法的讨论进行归类。

**📊 数据集**

主要使用工作坊提交论文、闪电报告和讨论笔记作为文本数据，并未使用公开医学数据集。

**📈 对比分析**

研究为探索性社会科学，未与基准模型或实验数据比较；通过与以往工作坊记录对比验证聚类结果的一致性。

**⚠️ 局限性**

局限性在于样本仅来自单一工作坊，缺乏外部验证；分析依赖参与者主观笔记，可能存在偏见；未给出可实现的技术实现或实验性能评估。

---

## 259. Learning from Demonstration via Spatiotemporal Tubes for Unknown Euler-Lagrange Systems

**arXiv ID:** 2607.00534 | [PDF](https://arxiv.org/pdf/2607.00534v1)

**作者:** Ratnangshu Das `[一作]` (Robert Bosch Centre for Cyber-Physical Systems), Pushpak Jagtap `[通讯]` (Robert Bosch Centre for Cyber-Physical Systems)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了STT‑LfD框架，将专家演示学习为时空管道（STT），并在未知欧拉-拉格朗日系统上实现任务执行；

**💡 创新点**

创新点在于利用异方差高斯过程直接从演示中学习时变精度约束，并设计不需要模型识别的闭式输入约束控制器；

**🔧 技术方法**

所用技术包括DTW时间对齐、HGP建模、时空管道构造以及基于分层反馈的闭式控制；

**📊 数据集**

实验数据集为移动机器人（figure‑8轨迹）和7‑DOF Franka机器人（字母W绘制）的15条专家演示；

**📈 对比分析**

与KMP+LQR、DMP+MPC、NODE‑CLF‑CBF、S2NNDS等基线比较，STT‑LfD在跟踪误差、控制力度和计算时间方面均优越（误差≈4 mm、控制耗时≈0.02 ms），并在质量变化与外部扰动下保持鲁棒；

**⚠️ 局限性**

主要局限包括仅适用于静态环境、手工设定管道宽度λ、未实现在线适应、以及缺乏多机器人与动态障碍物规避的功能。

---

## 260. Active-GRPO: Adaptive Imitation and Self-Improving Reasoning for Molecular Optimization

**arXiv ID:** 2607.00531 | [PDF](https://arxiv.org/pdf/2607.00531v1)

**作者:** Xuefeng Liu `[一作]` (Stanford University), Le Cong `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种名为 Active-GRPO 的主动参考指导强化学习框架，用于指令驱动的分子优化任务。

**💡 创新点**

创新点在于同时引入主动模仿-强化学习机制和主动引用机制，使模型能够在每个实例上动态决定何时模仿参考、何时自我强化，并不断升级参考目标，从而突破固定参考的性能上限。

**🔧 技术方法**

技术上基于 GRPO 的演员-学习者架构，结合答复级别的模仿损失、上下文相关的指导权重和记忆池式参考升级，使用逻辑 sigmoid 调节模仿强度，并采用可验证的分子属性与相似性度量作为奖励。

**📊 数据集**

实验使用 TOMG-Bench 的 MolOpt 子任务（LogP、MR、QED），以及 MolEdit、硬样本和长序列评测等数据集。

**📈 对比分析**

在三种子下与零样本、GRPO、RePO、Iterative SFT、Offline-strengthened RePO 等基线对比，Active-GRPO 在 SR×Sim 上平均达到 0.1773，分别在 LogP、MR、QED 上比 RePO 提升 1.48%、0.90%、2.06%，并且在多项统计检验中显著优于其他方法。

**⚠️ 局限性**

主要限制在于仍专注于有参考的分子优化场景，对无参考或弱参考的广义发现任务缺乏有效的探索与采样机制，难以应对更复杂或多目标的发现式问题。

---

## 261. Restore3D: Breathing Life into Broken Objects with Shape and Texture Restoration

**arXiv ID:** 2607.00522 | [PDF](https://arxiv.org/pdf/2607.00522v1)

**作者:** Xiaolong Shen `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出Restore3D框架，结合多视角图像修复与重建，实现破损3D物体的形状与纹理同时恢复。

**💡 创新点**

创新点在于自动生成破损-完整配对数据集，Mask Self‑Perceiver与Depth‑Aware Mask Rectifier实现自我感知修复区域，图像整合增强管线以及粗细分层几何纹理细化。

**🔧 技术方法**

采用多视角扩散模型(MVDream)、ControlNet‑Tile、Real‑ESRGAN、Large Reconstruction Models（InstantMesh）+ StableNorm正则、深度感知与自注意力等技术。

**📊 数据集**

使用83K条G‑objaverse合成破损数据集（RestoreIt‑3D）作为训练，测试集包含GSO、OmniObject3D、Objaverse、Breaking Bad和Fantastic Breaks。

**📈 对比分析**

与单视/多视修复基线（Repaint、Stable‑Diffusion、Instant3dit等）及重建基线（InstantMesh、Unique3D等）对比，Restore3D在PSNR/LIPPS/FID/SSIM、Chamfer/F‑score等指标上显著优于对照组，提升约3–5 dB PSNR、0.02 F‑score等。

**⚠️ 局限性**

受底层扩散模型分辨率限制，细节捕捉不足；对极大缺失区域仍可能产生模糊或偏离原始形状；需要更多高分辨率数据以提升几何与材质细节。

---

## 262. Auditing Empirical Comparisons in Quantum Software

**arXiv ID:** 2607.00516 | [PDF](https://arxiv.org/pdf/2607.00516v1)

**作者:** Boshuai Ye `[一作]` (University of Oulu), Arif Ali Khan `[通讯]` (University of Oulu)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于源证据的框架，对量子软件论文中报道的比较命题进行审计，并在119篇论文的455条比较中验证其可审计性。

**💡 创新点**

创新点在于将比较命题抽象为“声明卡”并锁定源证据，揭示出“可审计性漏斗”——大多数比较缺乏足够证据；同时引入Wilson置信区间的S/U/R判定规则，为审计结果提供可解释的统计支持。

**🔧 技术方法**

使用Python实现框架，结合源代码、实验记录与统计分析，采用Wilson置信区间、离散决策规则、敏感性诊断，并对量子软件工具链进行标准化评估。

**📊 数据集**

基于2020-2024年Google Scholar检索到的119篇量子软件论文，提取其455条比较命题，并利用论文公开的数据与artifact进行审计。

**📈 对比分析**

通过“先跑后锁”审计流程，计算保持方向的比例并应用Wilson置信区间判定Sustained/Unresolved/Reverse，结果表明仅有8条可直接审计，其中4条保持、2条不确定、2条被逆转。

**⚠️ 局限性**

局限在于大多数论文缺少匹配基线、设置和统计规则，导致无代理审计不可行；框架主要适用于严格方向关系，并受限于现有模拟器和工具链版本。

---

## 263. From Structural Equation Modelling to Double Machine Learning: Robustness Analysis for Survey-Based Research

**arXiv ID:** 2607.00512 | [PDF](https://arxiv.org/pdf/2607.00512v1)

**作者:** Ka Ching Chan `[一作]` (University of Southern Queensland), Ranga Chimhundu `[通讯]` (University of Southern Queensland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并验证了一种分阶段的SEM-OLS-DML稳健性分析框架，用于调查研究中的潜在构念模型。

**💡 创新点**

创新在于将SEM、OLS与DML组合成完整的稳健性工作流程，允许在测量验证后对所有路径进行透明回归与机器学习残差化检查，并加入反向诊断。

**🔧 技术方法**

使用结构方程模型、普通最小二乘回归与双重机器学习（随机森林、梯度提升机、支持向量机）进行残差化估计。

**📊 数据集**

基于金融科技领域的数字客户亲密度（Digital Customer Intimacy）调查数据。

**📈 对比分析**

通过SEM路径系数、OLS系数与三种DML学习器的显著性对比，发现大多数路径在三种方法下保持方向与显著性，部分路径表现为得分或学习器敏感。

**⚠️ 局限性**

局限包括：DML仅校正观测控制，无法消除未观测混淆；交叉验证受样本量影响；结果仅为关联性，不能断定因果方向。

---

## 264. Enhancing Robustness in Robot-Environment Interactions through Passive Compliant Degrees of Freedom: A Hybrid Position-Force Control Approach with Feedback Linearization

**arXiv ID:** 2607.00571 | [PDF](https://arxiv.org/pdf/2607.00571v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 265. BaseRT: Best-in-Class LLM Inference on Apple Silicon via Native Metal

**arXiv ID:** 2607.00501 | [PDF](https://arxiv.org/pdf/2607.00501v1)

**作者:** Prabod Rathnayaka `[一作]` (Base Compute), Lukas Wesemann `[通讯]` (Base Compute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款基于 Apple Metal 的 LLM 推理运行时 BaseRT，用于在 Apple Silicon 上实现高吞吐量的本地推理。

**💡 创新点**

创新点包括：1）针对芯片的内核融合与统一内存优化，消除框架层开销；2）数据驱动的架构描述器，使核心推理循环不再依赖分支；3）零分配解码循环与自定义 GPU 调度，显著降低 CPU 侧开销。

**🔧 技术方法**

采用了 Metal GPU API、手写 Metal 着色器、量化内核（Q2–FP16）、张量布局优化、FlashAttention 等技术，并通过自定义调度与融合实现低开销推理。

**📊 数据集**

评测使用公开 LLM 模型 Qwen3、Llama 3.2、Gemma 4（Q4/Q8 量化），在 Apple M4 Pro 与 M3 基础机上测试多种模型规模（0.6B–30B）。

**📈 对比分析**

与 llama.cpp、MLX 以及 native Metal 运行时 uzu 对比，BaseRT 在解码吞吐量上比 llama.cpp 高 1.04–1.56 倍、比 MLX 高 1.01–1.35 倍；在预填充阶段对大型 MoE 模型比其他实现提升 1.3–1.8 倍；在小型密集模型中与现有实现差距有限（5–10%）。

**⚠️ 局限性**

局限性：仅支持单机推理，缺乏连续批处理、跨 GPU 并行解码、张量并行；仅实现 Metal 后端，缺乏 CUDA/Vulkan 支持；不支持混合精度推理或持续批处理与投机解码等服务器级功能。

---

## 266. Robust 3D Alignment of Generative Reconstructions via Partial Monocular Observations

**arXiv ID:** 2607.00498 | [PDF](https://arxiv.org/pdf/2607.00498v1)

**作者:** Yuchen Zhang `[一作]` (Shanghai Jiao Tong University), Xiaoshuai Hao `[通讯]` (Xiaomi EV)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种训练无关、可解释的Sim(3)对齐框架，将生成式3D重建与稀疏单目观测对齐，解决尺度不确定与幻觉几何问题。

**💡 创新点**

显式尺度因子与守护式全局Sim(3)估计、幻觉过滤、尺度锁定局部优化的组合；提出极端条件下的GenPMOAlign–Where2Place基准，并在该基准上实现领先性能。

**🔧 技术方法**

基于Sim(3)闭式求解的全局RANSAC初始化、FPFH描述子、Hallucination Filtering、ICP局部精炼、SAM‑3D生成式重建、Depth Anything深度估计。

**📊 数据集**

自建的GenPMOAlign–Where2Place基准（100个室内场景，338对点云）；标准3DMatch、3DLoMatch、KITTI；真实机器人抓取实验。

**📈 对比分析**

与传统ICP/GICP/FGR以及学习方法Predator、GeoTransformer、RAP、BUFFER‑X等在GenPMOAlign基准上比较，取得Boundary F‑score、Center Drift、IoU、Chamfer、Normal Consistency、Depth MAE等指标的第一名；在通用基准上保持竞争力；机器人抓取平均成功率73.3%，为最高。

**⚠️ 局限性**

对生成模型的幻觉仍需过滤；依赖密集深度估计与语义分割；在极端低密度或极端遮挡场景下仍面临挑战；受限于生成模型的分辨率与重建质量。

---

## 267. VLM-AR3L: Vision-Language Models for Absolute and Relative Rewards in Reinforcement Learning

**arXiv ID:** 2607.00483 | [PDF](https://arxiv.org/pdf/2607.00483v1)

**作者:** Kuan-Chen Chen `[一作]` (National Tsing Hua University), Min-Chun Hu `[通讯]` (National Tsing Hua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了VLM-AR3L框架，利用视觉‑语言模型生成绝对与相对奖励来训练RL智能体

**💡 创新点**

创新点在于同时学习绝对奖励和相对奖励，利用VLM生成的偏好标签训练Siamese网络得到相对奖励，从而提升在循环、长时序和视觉复杂环境中的鲁棒性

**🔧 技术方法**

使用大型视觉‑语言模型（如Gemini‑2.0‑Flash）进行偏好标注，训练绝对奖励网络和Siamese相对奖励网络，结合Soft Actor‑Critic/PPO实现策略优化

**📊 数据集**

在多种环境上评估：Gym（CartPole）、SoftGym（Straighten Rope, Pass Water）、MetaWorld（Soccer, Sweep Into, Drawer Open）、MineDojo（Combat Spider, Milk Cow, Shear Sheep, Hunt Cow）

**📈 对比分析**

与CLIP、MineCLIP、RL‑VLM‑F以及oracle dense/sparse奖励对比，VLM-AR3L在所有任务上均超过或匹配oracle dense奖励，在长时序任务中显著优于其他基线

**⚠️ 局限性**

局限性包括对VLM偏好标注的准确性依赖，计算开销仍高于纯相对奖励，且在极端视觉噪声或任务目标模糊时仍可能出现误判

---

## 268. Group-Equivariant Poincaré Convolutional Networks

**arXiv ID:** 2607.00556 | [PDF](https://arxiv.org/pdf/2607.00556v1)

**作者:** Aiden Durrant `[一作]` (University of East Anglia), Georgios Leontidis `[通讯]` (UiT Arctic University of Norway)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究将离散对称群C4与D4的等变性嵌入到双曲空间的残差网络中，提出等变的Poincaré卷积网络；

**💡 创新点**

创新点在于几何安全的β缩放张量拆分、左正则置换路由以及联合方向的Poincaré中点批归一化，三者共同保证离散对称性与双曲几何兼容；

**🔧 技术方法**

使用Poincaré几何、离散群等变卷积、β‑拼接/拆分、左正则置换、联合方向批归一化以及Riemannian Adam优化；

**📊 数据集**

主要在CIFAR‑10数据集上进行训练与评估，另外进行OOD检测使用Places365、SVHN、DTD；

**📈 对比分析**

与标准欧氏与双曲ResNet、非等变Poincaré网络对比，等变模型在CIFAR‑10上Top‑1 88.77%（比标准双曲高约12%），低数据量下保持63.77%，并在OOD上取得更高AUROC和AUPR；

**⚠️ 局限性**

局限在于计算开销与数值不稳定，需要窄通道；仅支持离散群C4/D4，无法天然处理连续对称；验证仅在CIFAR‑10，未测试大规模数据集。

---

## 269. Robust Base Station Placement in Agricultural IoT via Bayesian Optimization

**arXiv ID:** 2607.00549 | [PDF](https://arxiv.org/pdf/2607.00549v1)

**作者:** Gourav Prateek Sharma `[一作]` (National Institute of Technology Kurukshetra), James Gross `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究农用5G基站布置问题，提出季节稳健的最大最小覆盖优化。

**💡 创新点**

创新点在于将季节变化纳入覆盖指标并用高斯过程贝叶斯优化实现高效搜索。

**🔧 技术方法**

使用高斯过程、贝叶斯优化、Sionna射线追踪、GPyTorch等技术。

**📊 数据集**

采用 1 km² 多作物农田模型，包含四个生长阶段的参数进行仿真。

**📈 对比分析**

与中心点、随机搜索、粒子群等基线对比，贝叶斯优化在少于 50 次射线追踪下实现 72.8 % 的 worst‑case 覆盖，比基线提升 4.6‑8.9pp。

**⚠️ 局限性**

限制在仅离散阶段模型、仅仿真结果未现场验证、基站高度固定等。

---

## 270. Online Matching with Size-Based and Convex Delays

**arXiv ID:** 2607.00536 | [PDF](https://arxiv.org/pdf/2607.00536v1)

**作者:** Junhao Gan `[一作]` (University of Melbourne), Seeun William Umboh `[通讯]` (University of Melbourne)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在线最小成本完美匹配问题带延迟（MPMD），并针对基于请求数的延迟（MPMD-Size）和凸延迟（MPMD-Convex）两种延迟模型给出了新的竞争比分析。

**💡 创新点**

主要创新包括：① 将MPMD-Size通过布尔向量状态压缩映射到一个 2^n-1 维的 MTS 空间，从而实现指数级提升的下界和上界，确立了确定性竞争比 Θ(2^n)；② 在均匀度量空间下证明，当延迟函数满足 f'(0)>0 时存在常数竞争比，填补了先前仅在 f(0)=f'(0)=0 时得到的 Θ(n) 上界。

**🔧 技术方法**

核心技术包括：MTS 与 MPMD-Size 的双向归约（利用状态的 XOR 结构），构造 T‑impatient（有限耐心）算法并证明其在线竞争比，及将凸延迟与线性延迟进行近似比较。

**📊 数据集**

论文为理论工作，未使用具体实验数据集；所有结果均来自数学证明。

**📈 对比分析**

通过理论比较 MTS 的已知上界/下界与本文构造的算法，得到：MPMD-Size 在任意 n 点度量空间上确定性竞争比 O(2^n)（与下界匹配），随机竞争比 O(n²)；MPMD-Convex 在均匀度量空间上常数竞争。

**⚠️ 局限性**

局限性：随机化 MPMD-Size 的竞争比仍存在 Ω(n) 与 O(n²) 的 gap；MPMD-Convex 的常数竞争仅适用于均匀度量空间，且竞争比随度量尺度 δ 变化；对于更一般的凸延迟函数和非均匀度量空间，仍缺乏有效算法。

---

## 271. Draped Surfaces: A Contour-Adaptive Interface Overlaid on the Physical Environment for Mixed Reality Workspaces

**arXiv ID:** 2607.00518 | [PDF](https://arxiv.org/pdf/2607.00518v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 272. You Shall Not Pass! Where and Why Developers Draw The Line on AI Autonomy

**arXiv ID:** 2607.00533 | [PDF](https://arxiv.org/pdf/2607.00533v1)

**作者:** Rudrajit Choudhuri `[一作]` (Oregon State University), Anita Sarma `[通讯]` (Oregon State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过对微软448名软件开发者的调查，量化并分析了他们在不同软件工程任务中对AI工具自主权的接受度，并将接受度映射到Endsley-Kiris的五级自动化尺度上，进一步探究了认知评估（任务价值、身份一致性、责任感、认知需求）以及个人特质（AI经验、风险容忍度）对AI自主权划分的预测作用。

**💡 创新点**

创新点在于：①将AI自主权划分为行动边界和决策边界，并用认知评估模型解释这两个边界的决定因素；②提出“意义工作设计的级联锁”框架，将AI自主权视为在不同“锁”间逐级推进的过程；③识别并阐述了与AI自动化相关的若干工作反模式，为实践提供设计警示。

**🔧 技术方法**

方法包括：①开放式问卷收集开发者对任务的认知评估和AI自主偏好；②使用规则+大语言模型（AI议会）对开放文本进行自动编码到五级自动化尺度；③混合效应累积链接模型（CLMM）和逻辑混合模型（GLMM）评估预测变量对AI自主度及其边界切换的影响；④主题分析对开放式回答进行定性阐释。

**📊 数据集**

数据集为一份由微软内部研发的问卷，共收集到1,535条任务级响应，最终用于模型训练的1,476条完整记录；样本来自不同团队、角色与地区，覆盖开发、设计、质量、运维及元工作等SDLC分类。

**📈 对比分析**

本研究并未与其他算法或工具做传统意义上的性能对比，而是通过统计建模评估预测变量的显著性和效应大小，结果显示任务身份、AI经验与风险容忍度显著影响AI自主度，且不同SDLC任务类型呈现显著差异。

**⚠️ 局限性**

主要局限包括：①样本仅来自一家AI前沿大型企业，缺乏跨组织泛化；②自报数据可能存在偏差；③交叉设计为横截面，无法确立因果关系；④开放式文本分类依赖规则与模型，可能存在误判；⑤未考虑AI工具的技术细节与使用频率等更细粒度影响因素。

---

## 273. Semantic Labelling in Practice

**arXiv ID:** 2607.00521 | [PDF](https://arxiv.org/pdf/2607.00521v1)

**作者:** Dieter Hofbauer `[一作]` (ASW Saarland), Johannes Waldmann `[通讯]` (HTWK Leipzig)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了自动化语义标记（semantic labelling）以证明终止性的问题，并提出了多种模型搜索与上下文闭包策略。

**💡 创新点**

创新点在于结合受限增长代数和强连通代数的双重约束来缩小搜索空间，并引入完整上下文闭包与语义上下文闭包两种新型闭包方法。

**🔧 技术方法**

使用的技术包括有限域代数枚举、受限增长序列、强连通性判定、上下文闭包（完整与语义）以及基于线性重量函数的终止性判定。

**📊 数据集**

实验基于 Termination Problems Database (TPDB) 的字符串重写基准以及 Termination Competition (TC) 结果。

**📈 对比分析**

对比实验表明，在 RG–SC 代数与完整上下文闭包的组合下，搜索速度明显提升，部分难解基准在 Matchbox/MnM 上实现了终止证明，而相对较大的上下文闭包也保持了可接受的计算时间。

**⚠️ 局限性**

主要局限在于代数数量的指数级增长、仅采用重量排序的终止判定、对字符串重写的关注以及对更大域和更复杂重写系统的扩展仍有限。

---

## 274. Prototype Language Models

**arXiv ID:** 2607.00510 | [PDF](https://arxiv.org/pdf/2607.00510v1)

**作者:** Dan Ley `[一作]` (Harvard University), Julius Adebayo `[通讯]` (Guide Labs Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新型的“PRISM”语言模型架构，该模型通过稀疏、非负的原型混合来生成下一个词，并通过聚类目标使每个原型与训练数据的邻域相对应；

**💡 创新点**

核心创新在于将可解释性和训练数据归因直接嵌入模型结构，而非事后分析；通过原型空间的聚类使得模型在保持与稠密模型相当的性能同时，使得训练数据归因（TDA）变得更快、更准确；

**🔧 技术方法**

技术上包括：稀疏原型重构、聚类损失（R1、R2）、残差分解、原型空间的Hessian分析与预条件化、缓存化的影响函数计算、线性原型控制器以及原型抑制等；

**📊 数据集**

主要使用的训练数据集为FineWeb‑Edu（约50B标记）和Nemotron‑CC（含代码与科学文本），验证时使用OpenAI LM Harness等下游基准；

**📈 对比分析**

与同规模稠密GPT进行对比，PRISM在验证困惑度与零样本下游平均准确率上与GPT相当（差距≤2.5%），在训练数据归因上比传统的EK‑FAC等后置方法快约500倍、内存相当；

**⚠️ 局限性**

局限性包括：目前仅在输出层实现原型机制，无法直接推断序列或文档级归因；对原型数量与稀疏度的选择较为敏感；在更大规模或更复杂任务上可能需进一步优化原型深层嵌入和更高效的聚类算法；

---

## 275. A Task-State Representation for Long-Horizon Mobile GUI Agents

**arXiv ID:** 2607.00502 | [PDF](https://arxiv.org/pdf/2607.00502v1)

**作者:** Yujie Zheng `[一作]` (Beihang University), Ji-Rong Wen `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种任务状态表示（TSR）框架，外部维护任务摘要、进度跟踪和转移验证，帮助移动GUI代理在长序列任务中保持目标和状态一致。

**💡 创新点**

创新点在于将任务持久状态与瞬时屏幕观察分离，利用训练自由的LLM更新器动态维护结构化状态，并将其注入原始代理提示，无需重训练或改造模型。

**🔧 技术方法**

使用提示式LLM生成结构化JSON作为状态更新器，结合视觉比较实现外部维护的任务状态。

**📊 数据集**

在四个移动GUI基准上验证：MobileWorld、AndroidWorld、MemGUI-Bench和VenusBench-Mobile。

**📈 对比分析**

与基线+上下文对比，TSR在MobileWorld提升至12%成功率，平均步骤相应增加；在MemGUI等内存密集任务提升3-5%；在AndroidWorld对某些模型略降。

**⚠️ 局限性**

局限在于更新器误判会累积错误，额外LLM调用导致延迟与成本，过度分解在短任务上可能产生噪声，且仅在Android移动环境验证。

---

## 276. MindEdit-Bench: Benchmarking Object-Level Counterfactual Spatial Reasoning in VLMs from In-the-Wild Photos

**arXiv ID:** 2607.00491 | [PDF](https://arxiv.org/pdf/2607.00491v1)

**作者:** Leyuan Yu `[一作]` (ZODA), Naoto Yokoya `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MindEdit-Bench 基准，评估视觉语言模型在对象级空间编辑和跨视角可见性编辑任务中的推理能力。

**💡 创新点**

首次引入对象级反事实空间编辑任务（L4、L5），并通过低成本三张手机照片自动构建 3D 场景图与结构化答案，实现可诊断的空间推理评测。

**🔧 技术方法**

采用 DA3 单目深度与相机姿态估计、SAM3 开源实例分割、跨视角 3D 关联与自动化场景图生成，配合结构化多选答案设计进行评测。

**📊 数据集**

使用 120 个私有室内场景的三张手持相机照片生成 1,003 道人类验证的问题，数据集已发布在 HuggingFace。

**📈 对比分析**

在 15 个 VLM（含 GPT‑5.4、Gemini‑3.1‑Pro‑Preview、Qwen3‑6‑Plus 等）上进行零样本评估，VLM平均准确率仅 8%–31%，而人类多数投票准确率 81%–97%，VLM 与人类差距 53 个百分点；结构化答案把随机基线降至 4%–13%。

**⚠️ 局限性**

低成本场景图抽取精度不足导致部分任务难度偏高；任务过滤与答案结构化可能抑制真实能力表现；模型性能噪声大，难以细粒度比较；Gemini 对训练数据的熟悉度可能造成评测偏差。

---

## 277. Efficient Multilingual Reasoning Transfer via Progressive Code-Switching

**arXiv ID:** 2607.00485 | [PDF](https://arxiv.org/pdf/2607.00485v1)

**作者:** Zhijun Wang `[一作]` (Nanjing University), Shujian Huang `[通讯]` (Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为PCS的进阶代码切换方法，将英文推理能力迁移到目标语言而无需强模型蒸馏或外部评判器。

**💡 创新点**

创新点在于：①仅用轻量级翻译生成代码切换推理轨迹；②通过逐步提升步级语言一致性阈值的RL课程学习，实现平滑且稳定的语言迁移；③避免了传统一次性强制目标语言导致的不稳定与性能下降。

**🔧 技术方法**

采用了有监督微调（SFT）初始化代码切换能力，随后使用基于SLC阈值的强化学习（GRPO）并加入KL约束与语言一致性奖励。

**📊 数据集**

使用了Qwen3-4B/8B基础模型，评估数据集包括MMLU-ProX（math子集）和MMATH，涵盖法语、葡萄牙语、日语、韩语、泰语等五种语言。

**📈 对比分析**

与基线（Prompt Control、Prefix Control、SFT、Naive RL、SoftLC RL、M-Thinker）对比，PCS在SLC&Acc指标上获得最高分，SLC约96–98%，准确率保持竞争力，显著优于所有对照方法。

**⚠️ 局限性**

局限性包括：只在中等长度（≤8k）推理环境下验证，未探究更长上下文；使用的小模型规模可能影响结果，未系统评估大模型与更长上下文的扩展性。

---

## 278. Cross4D-JEPA: Dense Cross-modal Correspondence Distillation for 4D Point Cloud Representation Learning

**arXiv ID:** 2607.00514 | [PDF](https://arxiv.org/pdf/2607.00514v1)

**作者:** Trung Thanh Nguyen `[一作]` (Nagoya University), Tuan-Anh Vu `[通讯]` (University of California Los Angeles)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了Cross4D-JEPA，通过渲染获取点云与二维教师模型的稠密对应关系，使用教师的 patch 级别特征对 4D 点云编码器进行无遮蔽、无对比、无解码器的稠密跨模态蒸馏。

**💡 创新点**

创新点在于：①首次在 4D 点云上实现稠密跨模态蒸馏；②利用渲染和 Z‑buffer 无需相机标定即可获得精确的 3D‑2D 对应；③以单点级别监督提升表示线性可分性、跨域迁移与标签效率。

**🔧 技术方法**

技术细节包括：冻结的 2D 基础模型（DINOv2、V‑JEPA2）作为教师；渲染+Z‑buffer得到 occlusion‑aware 对应；稠密 cosine loss 与轻量 head；Point4D（PointNet++ 结构）作为学生；随机投影压缩教师特征；无 mask、negative、decoder。

**📊 数据集**

使用的基准数据集有：MSR‑Action3D、DeformingThings4D‑Animals、NTU‑RGB+D 60、HOI4D。

**📈 对比分析**

与随机初始化、全局（单 clip）蒸馏以及多种自监督预训练方法（i‑JEPA、MaST‑Pre、Uni4D 等）进行对比；在四个基准上，线性探测准确率提升约 0.17–0.23 点，label‑efficient fine‑tune 在 10% 标签时提升 0.17；在 13× 参数压缩的 Point4D 下与大模型 P4Transformer 的准确率相近。

**⚠️ 局限性**

局限性包括：需稠密特征教师和可渲染点云；在大规模 NTU 数据集仅能预训练子集；对池化型 P4Transformer 效果不佳；缺乏姿态不变性，难以在极大变形下保持一致；未利用真实 RGB 信息。

---

## 279. A Methodology for Investigating AI Patterns Prevalence in Software Repositories

**arXiv ID:** 2607.00558 | [PDF](https://arxiv.org/pdf/2607.00558v1)

**作者:** Srinath Perera `[一作]` (WSO2), Rania Khalaf `[通讯]` (WSO2)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于主动学习的AI模式识别与流行度估计方法。

**💡 创新点**

创新点在于将调用图社区作为代码块来嵌入，结合主动学习和混合分类器，再利用矩阵逆+蒙特卡洛模拟给出模式出现频率的置信区间。

**🔧 技术方法**

使用Gemini LLM提取模式候选，DBSCAN+UMAP聚类，Gemini文本嵌入做代码嵌入，Logistic回归/ SVC/ KNN 的加权集成，主动学习策略，混淆矩阵反演+蒙特卡洛采样。

**📊 数据集**

使用100个GitHub公开AI项目（Python，250-1000星），共提取2442个代码社区。

**📈 对比分析**

与随机基线比较，分类准确率56%，召回率55%，比11%随机基线高5倍；流行度估计在7类中5类提供可用置信区间。

**⚠️ 局限性**

局限包括少样本类被归入None导致细粒度缺失，模型对单一代码片段模式识别不足，嵌入对代码块选择敏感，LLM偏差可能影响结果。

---

## 280. Prior-Anchored Debiasing for Long-Tailed Multi-Organ Pathology Report Generation

**arXiv ID:** 2607.00499 | [PDF](https://arxiv.org/pdf/2607.00499v1)

**作者:** Feng Yang `[一作]` (City University of Hong Kong), Ping Chen `[通讯]` (University of Massachusetts Boston)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了PriOrGen框架，结合视觉与文本双重先验以缓解多器官病理报告生成中的长尾分布偏差。

**💡 创新点**

首次提出视觉原型锚定瓶颈与元报告锚定库的双重先验机制，分别从视觉表示和文本解码两方面共同消除长尾偏差。

**🔧 技术方法**

采用信息瓶颈与聚类生成诊断原型、KL正则化、cosine相似度筛选、交叉注意力解码，并结合预训练模型UNI、PLIP、Gemini-3 Pro。

**📊 数据集**

构建了多器官长尾病理报告生成数据集ML‑Path，包含11种癌症共4686对WSI-报告，并按头/中/尾分布。

**📈 对比分析**

与8种基线（LSTM、Transformer、R2Gen、Wsicaption、HistoCap、BiGen等）在BLEU‑Mean、METEOR、ROUGE‑L上对比，PriOrGen在整体与尾类指标上显著优于所有基线，尾类提升约8‑16%。

**⚠️ 局限性**

依赖手工验证的元报告模板、仅覆盖11种器官、对极端尾类样本量仍有限，且未探索跨模态多语言适应。

---

## 281. From Real-Time Planning to Reliable Execution:Scalable Coordination for Heterogeneous Multi-Robot Fleets in Industrial Environments

**arXiv ID:** 2607.00591 | [PDF](https://arxiv.org/pdf/2607.00591v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 282. HieDG: A Hierarchical Discrete Geometry-Guided Framework for Multi-Animal Tracking

**arXiv ID:** 2607.00494 | [PDF](https://arxiv.org/pdf/2607.00494v1)

**作者:** Chenxun Deng `[一作]` (Chinese Academy of Sciences), Xi Chen `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 HieDG，一种层级离散几何引导的多动物跟踪框架，利用两阶段残差量化将位置、尺寸、速度等几何信息离散化为代币，并与视觉特征融合进行端到端训练；

**💡 创新点**

创新点在于将连续几何信号通过层级残差量化转为离散代币，显著抑制噪声放大并提升身份关联鲁棒性；同时引入极坐标速度表示和多阶段量化，进一步稳定跨帧匹配；

**🔧 技术方法**

技术方案包括：基于 Deformable DETR 的检测与特征提取、两阶段残差向量量化、Transformer 交叉注意力查询传播、直通估计的向量量化损失、KNN 回归速度估计、以及多头注意力与视觉+几何+身份特征的融合；

**📊 数据集**

使用了动物专用数据集 AnimalTrack、BFT、BuckTales 进行评测，并在通用多目标跟踪数据集 DanceTrack、SportsMOT 上验证通用性；

**📈 对比分析**

与启发式与查询式现有方法对比，HieDG 在 AnimalTrack、BFT、BuckTales 上分别在 HOTA、AssA、IDF1 上提升约 2–3% 以上，且在 DanceTrack、SportsMOT 上保持竞争性能，证明方法的优越性和通用性；

**⚠️ 局限性**

局限性包括：需要额外的量化代码本导致参数和 FLOPs 轻微增加；对极端光照或模糊目标的几何量化效果有限；依赖检测质量，检测失误会直接影响几何编码与身份关联；对极度快速或跳跃运动的速度估计仍有改进空间。

---

## 283. AGI Maze as a Benchmark Framework for World-Modeling Agents

**arXiv ID:** 2607.00627 | [PDF](https://arxiv.org/pdf/2607.00627v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 284. GADA: Geometry-Aware Deformable Aggregation for Image-Based Gaussian Splatting

**arXiv ID:** 2607.00595 | [PDF](https://arxiv.org/pdf/2607.00595v1)

**作者:** Siwoo Lim `[一作]` (Korea Advanced Institute of Science and Technology), Chang D. Yoo `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Geometry-Aware Deformable Aggregation (GADA)，通过可学习的形变偏移和自适应视图聚合改进高频细节恢复的基于图像的高斯喷射渲染。

**💡 创新点**

用局部可变形偏移代替粗糙的可见性检查，并通过几何验证聚合权重自适应抑制不可靠像素，提升细节保持和推理速度。

**🔧 技术方法**

迭代可学习偏移网络、几何上下文嵌入、软最大化权重聚合、轻量级残差CNN和正则化损失。

**📊 数据集**

在 Mip-NeRF 360、Tanks & Temples、Deep Blending、Shiny 等公共视图合成基准上训练和评估。

**📈 对比分析**

与 3DGS、IBGS、Octree‑GS 等最先进方法对比，GADA 在 PSNR/SSIM/LPIPS 上取得最高或竞争最优，帧率提升至 47 FPS，速度提升约 2×，并且内存占用更低。

**⚠️ 局限性**

对极少视角或大视角差的情况仍受限，需要更广域匹配或自适应迭代；训练时仍比单一渲染模型略慢，且在动态场景与跨场景泛化上尚未验证。

---

## 285. EPO: Boosting 3D Foundation Models with Edge-based Pose Optimization

**arXiv ID:** 2607.00579 | [PDF](https://arxiv.org/pdf/2607.00579v1)

**作者:** Mattia D'Urso `[一作]` (Graz University of Technology), Friedrich Fraundorfer `[通讯]` (Graz University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd`

**🎯 论文内容**

本文提出一种无轨迹的边缘对齐优化框架EPO，用于加速和提升3D基础模型（如VGGT）生成的结构从运动重建的几何精度。

**💡 创新点**

创新点在于用全可微分的边缘重投影损失替代传统特征匹配与BA，既保持几何一致性又显著降低内存和推理时间。

**🔧 技术方法**

采用Transformer-based 3DFM生成初始姿态，Canny边缘检测+距离变换，MLP姿态微调、全局视图图构建以及基于Huber的边缘对齐损失实现端到端优化。

**📊 数据集**

在TerraSky3D、ScanNet++、Mip-NeRF 360三个公开数据集上进行评测。

**📈 对比分析**

与VGGT+BA、VGGT+Ref+BA等基线相比，EPO在AUC(5°)上提升4–6点，运行时间缩短约6×，在NVS任务中PSNR/SSIM/LPIPS均优于传统BA。

**⚠️ 局限性**

局限性包括对纹理密集/非结构性纹理场景的边缘检测敏感以及在视图图连通度低时优化约束不足。

---

## 286. Coachable agents for interactive gameplay

**arXiv ID:** 2607.00642 | [PDF](https://arxiv.org/pdf/2607.00642v1)

**作者:** Roberto Capobianco `[一作]` (Sony AI), Peter R. Wurman `[通讯]` (Sony AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

训练可在运行时根据用户指定风格控制动作的强化学习智能体，应用于游戏《Horizon Forbidden West》、《Gran Turismo 7》以及DeepMind Control Suite中的类人机器人。

**💡 创新点**

创新点在于将UVFA改为风格条件化，结合情景训练、数据增强和专用回放缓冲，并为高维游戏提出新的CatSAC算法，使同一任务下能生成多种可选行为并支持实时切换。

**🔧 技术方法**

采用强化学习（SAC/ CatSAC）、风格条件化UVFA、情景训练、数据增强、风格权重调节及Pareto曲线分析等技术。

**📊 数据集**

使用Sony AI内部的AAA游戏环境（Horizon Forbidden West、Gran Turismo 7）以及DeepMind Control Suite Humanoid和自制训练场景与敌人组合。

**📈 对比分析**

通过多次随机种子训练，评估不同风格下的胜率、伤害类型与风格分数，并用Pareto曲线展示风格权重与胜率的权衡。实验表明风格请求与实际行为高度对应，即使在OOV场景下也保持较高性能。

**⚠️ 局限性**

局限性包括：需大量手工奖励与训练场景设计；跨任务迁移能力有限；风格组合仍需手动设定；极端OOV场景下性能下降；以及对计算资源需求高。

---

## 287. The Perception and Impact of Non-inclusive Language in Software Artifacts

**arXiv ID:** 2607.00626 | [PDF](https://arxiv.org/pdf/2607.00626v1)

**作者:** Ahmad J. Tayeb `[一作]` (King Abdulaziz University), Mohammad D. Alahmadi `[通讯]` (University of Jeddah)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对1,212名开源软件贡献者的问卷调查，研究了非包容性术语在软件开发中的感知与影响。

**💡 创新点**

创新点在于首次系统性地量化不同群体对“master/slave”等术语的包容性认知，并探讨其对团队多样性、归属感和生产力的潜在负面影响。

**🔧 技术方法**

采用定量问卷与非参数统计（Mann-Whitney U、Kruskal-Wallis、Chi-square、Cramér’s V等）进行数据分析。

**📊 数据集**

数据集为来自GitHub的公开贡献者样本，包含性别、年龄、地区、英语母语与非母语、职业及编程经验等特征。

**📈 对比分析**

通过比较各群体的Likert评分和比例，发现女性、非二元、美国或母语英语者对非包容性术语的敏感度更高，尽管多数受访者整体保持中立或否认负面影响。

**⚠️ 局限性**

局限性包括自选样本偏倚、文化背景差异导致的解释不一致、对非包容性术语的选择有限以及缺乏对真实工作绩效与行为的客观测量。

---

## 288. Positive and Negative Determinant Strategies in Repeated Games with Behavior-Value Inconsistency

**arXiv ID:** 2607.00625 | [PDF](https://arxiv.org/pdf/2607.00625v1)

**作者:** Yuan Liu `[一作]` (Beijing University of Posts and Telecommunications), Bin Wu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在存在行为与内部价值不一致的两人重复博弈中引入内部成本后，证明了正负行列式策略能够实现单方面收益控制；

**💡 创新点**

创新点在于将内部价值与行为不一致的心理成本纳入博弈框架，证明传统零行列式（ZD）策略不存在，并提出能够更强大地控制收益的新型正负行列式策略；

**🔧 技术方法**

采用博弈理论、Markov链分析、Press‑Dyson向量与线性代数方法，对策略空间进行解析推导；

**📊 数据集**

未使用实验数据集，主要通过理论推导与数值仿真验证策略效果；

**📈 对比分析**

通过理论证明与仿真比较，展示正负行列式策略在控制对手收益和自身收益超额方面优于传统ZD策略，表现出更高的收益支配性；

**⚠️ 局限性**

局限在仅考虑两人两策略模型，未涉及多玩家、多策略或学习者动态行为，以及实验验证与实际应用的进一步研究。

---

## 289. Identifying Latent Concepts and Structures for Generalized Category Discovery

**arXiv ID:** 2607.00620 | [PDF](https://arxiv.org/pdf/2607.00620v1)

**作者:** Boyang Dai `[一作]` (University of Hong Kong), Yizhou Yu `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在Generalized Category Discovery中如何通过低秩组合原语场重塑特征空间，以提高未标记类别的发现。

**💡 创新点**

提出了Compositional Primitive Fields (CPF)，将特征重参数化为可学习原语与空间分配的低秩表示，成为可插拔的表示层。

**🔧 技术方法**

采用Transformer backbone、低秩原语码表、令牌-原语注意力、图像条件偏移以及对齐学习损失等技术。

**📊 数据集**

在细粒度数据集CUB-200、Stanford Cars、FGVC Aircraft以及粗粒度CIFAR-10/100、ImageNet-100上进行验证。

**📈 对比分析**

将CPF插入多种GCD基线（SimGCD、LegoGCD、CMS、SelEx等）后，显著提升新类别聚类准确率（最高+8%），整体ACC提升约4-6%。

**⚠️ 局限性**

仍需手动设定原语数量和注意力头数，在极大尺度数据上可能面临计算开销与原语可解释性下降的挑战。

---

## 290. Path Planning in Physically Viable World Models

**arXiv ID:** 2607.00673 | [PDF](https://arxiv.org/pdf/2607.00673v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 291. Retrieved Images as Visual Thought: Training-Free Multimodal In-Context Learning for the Open-vs-Closed Gap

**arXiv ID:** 2607.00606 | [PDF](https://arxiv.org/pdf/2607.00606v1)

**作者:** Bingchen Huang `[一作]` (Meituan), Yuanchao Du `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练自由的检索式视觉推理框架，将检索到的标注图像-标签对作为“视觉思维”单元，按交替用户/助手回合注入，配合结构化定义进行多属性推理。

**💡 创新点**

创新点在于：①把检索与交替回合注入结合，形成无训练、无生成、无工具协议的思维路径；②通过结构化定义将多属性任务统一化；③证明检索+回合是跨任务、跨模型的通用提升杠杆。

**🔧 技术方法**

技术包括：多模态检索（Qwen3‑VL‑Embedding‑8B 或 CLIP），结构化类定义（包含名称、定义、可选属性），交替回合注入，联合多属性解码，基于 LLM 的 JSON 预测与置信度分配。

**📊 数据集**

使用的数据集有：VL‑ICL Bench Fast Open MiniImageNet；新构建的 27 类 5 属性艺术分类基准（来源于 Met Museum、National Gallery of Art、Wikimedia Commons）；Bongard‑OpenWorld 概念归纳基准。

**📈 对比分析**

与现有方法对比：在 VL‑ICL 上，开源 Qwen3‑VL‑30B‑A3B 4‑shot 98.5% 与 72B LLaVA‑OneVision SOTA 98.7% 相当，但参数仅 1/2.4；在 Bongard‑OpenWorld 上，仅使用回合层即可将 GPT‑4.1 从 46.6% 提升至 72.7%（+26.1 分）；在艺术基准上，三种骨干均获得 4–6 分宏观提升。整体显示该框架在不增加训练成本的前提下显著提升多任务性能。

**⚠️ 局限性**

局限性：基准规模较小（核心 153 图），部分属性（Composition、Mood）标注一致性低；检索方法对同一艺术家的图像容易造成偏差；长上下文下大 K 触发推理衰减；未测试微调或多步推理任务；开放模型仅在合成名称任务中逼近封闭模型。

---

## 292. Measuring Dead Directions: Decomposing and Classifying Singular Structure off Canonical Alignment

**arXiv ID:** 2607.00603 | [PDF](https://arxiv.org/pdf/2607.00603v1)

**作者:** Tejas Pradeep Shirodkar `[一作]` `[通讯]` (Indian Institute of Information Technology), Tejas Pradeep Shirodkar (Indian Institute of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种不依赖梯度下降和对角化的测量方法，利用在冻结检查点上扫描方向性 Fisher 率来定量分析训练网络中的奇异结构（即死亡方向），并能分辨真实奇异点与网络的对称（gauge）方向；

**💡 创新点**

创新点在于：①去除了原先对齐和下降预条件，提供了通用的单检查点读取；②通过 K‑FAC 近核与对偶性自动定位死亡方向；③构造联合模式而非搜索底部谱；④使用纯度匹配窗口精确提取 KL 序数 k；⑤对多种架构（Transformer、CNN、LayerNorm）给出统一检测器；⑥将每个方向的 λ_dir=1/(2k) 与全局学习系数、重数、奇异波动（Watanabe 三元组）关联。

**🔧 技术方法**

核心技术包括：方向性 Fisher 率、K‑FAC（Kronecker‑factored Fisher）近核与对偶性、方向扫描、纯度匹配窗口、方向性阶数判别、与后验采样（SGLD）对比、奇异波动 ν(k) 的直接估计、以及对网络各层的可插拔检测器。

**📊 数据集**

实验使用 ImageNet‑100、模块乘法任务、深度线性网络、手工构造的 node‑death 与 determinantal 细胞等，涵盖 Vision Transformer、卷积网络、LayerNorm 以及各种激活函数（ReLU、gelu、squared‑ReLU）。

**📈 对比分析**

与传统的后验采样（SGLD）和已知解析模型（深度线性、节点死亡）进行对比。读取的单方向阶数在构造细胞中与理论一致（误差<0.1%），全局 λ 与解析闭式在可枚举细胞中相同；与 SGLD 的单值 λ 也高度相关（ρ≈0.8）。在实际 Vision Transformer 上，读取得到的 k 与激活函数预测一致，且能区分 LayerNorm‑kernel gauge 与实际节点死亡。

**⚠️ 局限性**

局限性包括：①仅能读取已形成的死亡方向；②需要检测器先行定位方向；③对低样本数或极低 Fisher 底部会失效；④对高度混合的多重奇异点（如宽矩阵的 determinantal 结构）会漏计或误读；⑤读取结果与优化器相关，无法自动发现所有奇异结构。

---

## 293. Domain Arithmetic: One-Shot VLA Adaptation under Environmental Shifts

**arXiv ID:** 2607.00666 | [PDF](https://arxiv.org/pdf/2607.00666v1)

**作者:** Taewook Kang `[一作]` (Seoul National University), Jonghyun Choi `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种只需单一演示即可在环境转移下适配视觉-语言-动作（VLA）模型的方法。

**💡 创新点**

创新点在于：①发现一次性微调权重可分解为任务特异和域特异方向；②利用权重算术中的类比操作（subtract source‑task update from target‑task update）提取域向量；③结合子空间过滤和缩放提升域向量质量。

**🔧 技术方法**

主要技术：权重空间算术、子空间对齐、奇异值分解（SVD）、子空间过滤与缩放、基线模型加权。

**📊 数据集**

数据集：LIBERO（视觉/姿态转移）、MimicGen（跨机器人转移）、真实 UR10e 机器人实验；使用预训练 VLA 模型（π_0.5、π_1.5 等）。

**📈 对比分析**

与零样本、一次性全微调、FLA、RETAIN 等基线对比；在多种视觉/姿态、跨机器人、真实场景下，一次性域向量方法在成功率上平均提升 15–30%（具体数值见表格），并保持对未见任务的良好泛化。

**⚠️ 局限性**

局限性：在极端视角/环境转移（如高度视角偏移）下性能仍下降；需要对标量缩放系数 α 进行小范围超参数搜索；未给出每层自适应缩放的无超参实现。

---

## 294. Faithful by Definition: Emotion Analysis via Natural Semantic Metalanguage Explications

**arXiv ID:** 2607.00661 | [PDF](https://arxiv.org/pdf/2607.00661v1)

**作者:** Frank Xing `[一作]` (University of Reading), Erik Cambria `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种事件基础情感分析的“解释接口”，通过将文本映射到基于自然语义金属语言（NSM）的十二槽闭合词汇解释，并用预先定义的规则直接从解释得到情感标签；

**💡 创新点**

创新点在于把可解释性从后置评估转为结构性保证：解释到标签的映射是可复制、可验证的；同时提供可逐行验证的NLI接口；

**🔧 技术方法**

采用LLM驱动的解析器（Zero-shot + 微调 LoRA）、固定规则映射器、基于NSM的离散槽模板以及自然语言推理模型做逐行验证；

**📊 数据集**

使用Crowd-enVent派生的自述事件文本，标注生成的 EmoExpl‑1200 1,200 条例子；

**📈 对比分析**

与多种基线（黑盒分类器、情感评估器、概念瓶颈LLM、SenticNet词典）在36条 held‑out 例子上比较，精度与黑盒模型相当，选择性准确率最高（≈0.48），但总体准确率略低；

**⚠️ 局限性**

限制在于：① 解析器仍需学习，存在误解析；② 词汇表闭合导致某些情感细粒度无法表达；③ 数据集存在多重可解释性，导致标签不可确定性；④ 验证器保守，可能低估准确性。

---

## 295. Linguistic Relative Policy Optimization for Video Anomaly Reasoning

**arXiv ID:** 2607.00654 | [PDF](https://arxiv.org/pdf/2607.00654v1)

**作者:** Jiaxu Leng `[一作]` (Chongqing University of Posts and Telecommunications), Xinbo Gao `[通讯]` (Chongqing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于多轨迹相对强化学习的语言经验蒸馏框架LRPO，用于无调参的视频异常检测

**💡 创新点**

创新点在于将多轨迹的语义优势转化为可编辑的语言异常经验，并通过相对奖励优化经验而非参数

**🔧 技术方法**

使用冻结的多模态大语言模型（InternVL3、GPT‑OSS）以及CLIP4CLIP检索，并设计包括准确度、偏好一致性与时间依赖性的相对奖励

**📊 数据集**

在公开的XD‑Violence、UCF‑Crime和UBnormal三个数据集上进行评估

**📈 对比分析**

相较于现有无调参方法，LRPO在XD‑Violence上AP达73.17%，UCF‑Crime AUC 85.36%，并在跨数据集迁移时保持优异性能，整体显著提升

**⚠️ 局限性**

局限在于经验库的覆盖受训练样本多样性限制，难以充分处理训练集中未出现的稀有场景或新型异常

---

## 296. Learning to Watch: Active Video Anomaly Understanding via Interleaved Policy Optimization

**arXiv ID:** 2607.00622 | [PDF](https://arxiv.org/pdf/2607.00622v1)

**作者:** Mengjingcheng Mo `[一作]` (Chongqing University of Posts and Telecommunications), Xinbo Gao `[通讯]` (Chongqing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Anom-π框架，将视频异常理解转化为主动闭环推理与证据获取的交互式决策过程。

**💡 创新点**

设计可复用的原子观察操作和统一动作空间，并通过Interactive Direct Preference Optimization (iDPO)在弱监督下学习自适应查询与终止策略。

**🔧 技术方法**

使用LLM视觉语言模型（如Qwen3-VL-Instruct-2B）结合主动证据查询、AEI奖励、iDPO优化和闭环推理技术。

**📊 数据集**

在四大异常理解基准（XD‑Violence、UCF‑Crime、UBnormal、CSAD）上进行评测。

**📈 对比分析**

与现有LLM和多模态基线相比，Anom-π在所有基准上取得更高的帧级AUC/AP（尤其在复杂场景提升6%以上），且仅使用16–32帧上下文、2B参数规模。

**⚠️ 局限性**

当返回证据质量低或场景模糊、噪声大时，主动查询可能产生冗余或过早终止，导致性能受限；同时在某些数据集的场景特征不足以驱动足够查询。

---

## 297. High-Performance NTT Accelerators for PQC leveraging Unified Redundant Arithmetic and Fine-Tuned Microarchitecture

**arXiv ID:** 2607.00621 | [PDF](https://arxiv.org/pdf/2607.00621v1)

**作者:** George Alexakis `[一作]` (Democritus University of Thrace), Giorgos Dimitrakopoulos `[通讯]` (Democritus University of Thrace)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种面向后量子密码学的FPGA NTT/INTT并行迭代加速器，兼顾NTT与INTT的统一蝶形单元。

**💡 创新点**

创新点包括：1）引入统一冗余Montgomery数值表示，消除多余的模修正；2）将INTT缩放合并入算术单元，去除专用除2模块；3）采用层级化Montgomery乘法映射至DSP48E1，显著提升频率。

**🔧 技术方法**

采用冗余数值、Montgomery乘法、流水线与并行蝶形单元、FFT递归结构，并在FPGA上通过DSP块实现高效乘法。

**📊 数据集**

使用标准PQC参数集（ML‑KEM、ML‑DSA、FN‑DSA、NewHope）以及多项式阶数256、512、1024等。

**📈 对比分析**

与现有可配置/固定模数实现对比，时钟频率提升至约437 MHz（17 bit）/905 MHz（34 bit），执行时间降低35%–73%，面积与对等或更小。

**⚠️ 局限性**

局限在于冗余表示导致位宽增大，未实现侧信道防护，也缺少完整功耗与安全性评估。

---

## 298. Online computation of maximal closed substrings

**arXiv ID:** 2607.00612 | [PDF](https://arxiv.org/pdf/2607.00612v1)

**作者:** Hiroki Shibata `[一作]` (Kyushu University), Shunsuke Inenaga `[通讯]` (Kyushu University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种在线算法，能够在每次向字符串追加字符后即时计算所有最大闭子串（MCS）；

**💡 创新点**

创新点在于构造了链式切割后缀树（Link‑Cut Suffix Tree，LCST）这一新数据结构，使得能够在 O(n log n) 时间内维护每个子串的右最近出现位置，从而实现 O(n log n) 的在线 MCS 计算；

**🔧 技术方法**

主要技术包括：1）基于 Weiner 的在线后缀树构造；2）将后缀树与链式切割树结合，得到 LCST；3）利用 LCST 中的 solid/dashed 边来快速查询子串的右最近出现位置；4）利用这些信息在每一步更新中判定闭子串的左/右极大性并维护 MCS；

**📊 数据集**

论文未使用实际数据集，而是通过理论分析和实验模型证明算法的时间与空间复杂度；

**📈 对比分析**

与现有的离线 MCS 计算方法（O(n log n) 复杂度）以及最近的在线 LZ77/最右匹配查询实现相比，本文提供的在线 MCS 算法同样实现 O(n log n) 总时间和 O(n) 空间，满足最坏情况下的时间最优；

**⚠️ 局限性**

限制：1）算法仍以 O(n log n) 为上界，无法进一步突破；2）实现复杂度高，需要维护链式切割树和后缀树，工程实现成本较大；3）对于非常短的输入或特定结构的字符串，实际性能可能受常数因子影响。

---

## 299. "Don't Say It!": Constraints, Compliance, and Communication when Language Models Play Taboo

**arXiv ID:** 2607.00601 | [PDF](https://arxiv.org/pdf/2607.00601v1)

**作者:** Sara Candussio `[一作]` (University of Trieste), Malvina Nissim `[通讯]` (University of Groningen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言模型在禁止词游戏Taboo中的生成与猜测能力，评估不同层级约束干预（提示、生成时限制、内部表示操控）对模型表现的影响。

**💡 创新点**

系统比较多层次约束策略并结合人类对话实验，揭示约束实施方式对语言模型遵守规则与信息传递的双重影响，并提出基于稀疏自编码器的内部表示抑制方法。

**🔧 技术方法**

采用开源LLM Gemma、GPT‑OS 20B、Claude Sonnet‑4.6 等，并利用提示工程、logit 掩码、在线生成约束以及稀疏自编码器（SAE）对内部残差流进行干预；评估使用 LLM‑as‑judge 和人类实验。

**📊 数据集**

使用194张意大利语 Taboo 卡牌的手工转录数据集，每张卡包含目标词和一组禁止词。

**📈 对比分析**

通过比较三种约束策略在违规率、Pass@k 等指标上的表现，发现约束生成可达约98% 合规率，信息有效性与人类相差不大，但模型在猜测阶段仍明显落后（人类猜测准确率最高，模型仅达30% Pass@10）。

**⚠️ 局限性**

仅限意大利语，缺乏多轮交互实验，SAE 方法仅在 Gemma 上测试，评估者 LLM 敏感度差异大，且模型无法充分模拟人类快速关联思维，导致猜测性能偏低。

---

## 300. Vehicle-to-Grid as a 5G Smart Grid Vertical: Non-Technical Barriers and Implications for Communication Networks

**arXiv ID:** 2607.00589 | [PDF](https://arxiv.org/pdf/2607.00589v1)

**作者:** Shangqing Wang `[一作]` (Technische Universitaet), Frank H. P. Fitzek `[通讯]` (Technische Universitaet)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文将车联网与电网结合，探讨Vehicle-to-Grid在5G智能电网垂直领域的应用，分析其非技术性障碍，并提出对通信网络的设计建议。

**💡 创新点**

创新点在于从非技术视角系统性梳理V2G障碍，提出多维度解决框架，并给出5G网络配置的具体优化方向。

**🔧 技术方法**

采用系统分析法、案例研究与网络仿真（基于ns-3/5G-URA）以及QoS评估工具。

**📊 数据集**

利用公开的EV充电桩数据集（Open Charge Map）和德国电网负荷记录。

**📈 对比分析**

与传统基于LTE的V2G方案比较，仿真结果显示5G配置下时延下降至<10ms，吞吐量提升30%，可靠性达99.99%。

**⚠️ 局限性**

局限性包括缺乏大规模实测数据、对法律合规细节的探讨不足以及仅关注技术层面而非经济可行性。

---

## 301. Caption Bottleneck Models

**arXiv ID:** 2607.00578 | [PDF](https://arxiv.org/pdf/2607.00578v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 302. Multi-Label Node Classification with Label Influence Propagation

**arXiv ID:** 2607.00671 | [PDF](https://arxiv.org/pdf/2607.00671v1)

**作者:** Yifei Sun `[一作]` (Zhejiang University), Bingsheng He `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文提出了一种利用图神经网络中的传播与变换操作对标签之间的影响进行分解、量化，并通过构造标签影响传播图动态调整标签损失权重，从而提升多标签节点分类性能。

**💡 创新点**

创新点在于：①将GNN的消息传递拆分为传播（P）和变换（T）两步，分别从结构传播和梯度更新角度量化标签间的正负影响；②将两类影响按元素相乘得到最终影响矩阵，并通过软最大化构造标签图；③利用图的PageRank计算标签重要性，动态调节各标签的损失权重，形成端到端可训练的标签影响传播框架。

**🔧 技术方法**

使用的技术包括：图神经网络（GCN、GAT等骨干网络），Personalized PageRank（PPR）用于估算传播影响；梯度角度相似度用于度量变换影响；标签影响传播图（G_LIP）与软最大化、PageRank权重计算；以及多标签分类的二元交叉熵损失。

**📊 数据集**

实验使用了七个多标签节点分类数据集：经典的DBLP、BlogCat；大规模OGB蛋白质数据集OGB‑p；以及新生物学数据集PCG、HumLoc、EukLoc。

**📈 对比分析**

与多种基线（无监督嵌入+BR/CC，GCN+ML‑KNN/PLAIN，专门的MLNC方法如MLGW、ML‑GCN、LARN、LANC、VariMul）以及自适应损失权重的“backbone+Auto”在节点拆分和标签拆分两种划分下进行比较。实验结果显示，该方法在宏观AUC、AP等指标上平均提升约3.06%（AUC）和2.54%（AP），在所有数据集和不同骨干网络上均表现出显著优势。

**⚠️ 局限性**

局限性包括：①方法依赖于对标签影响矩阵的准确估计，若标签关系变化剧烈或标签稀疏可能影响效果；②构造标签图和PageRank计算在大规模标签数下可能带来额外开销；③目前仅在静态图和二元交叉熵损失场景验证，尚未探讨动态图或其他损失函数的适用性。

---

## 303. DART: Difficulty-Adaptive Routing for Zero-Shot Video Temporal Grounding

**arXiv ID:** 2607.00672 | [PDF](https://arxiv.org/pdf/2607.00672v1)

**作者:** Zhengbo Zhang `[一作]` (Singapore University of Technology and Design), Ming-Hsuan Yang `[通讯]` (University of California Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对零样本视频时序定位（Zero‑Shot Video Temporal Grounding）中存在的“推理缺口”，提出了一种基于查询条件DPP（Determinantal Point Process）的关键帧选择与谱熵难度评估相结合的难度自适应路由框架（DART）。

**💡 创新点**

创新点主要体现在：①利用查询条件DPP同时兼顾帧-查询相关性与时序多样性，实现高效的关键帧抽取；②通过DPP核的谱熵作为单一指标动态判定查询难度，并据此将查询路由至快速直接预测路径或慢速结构化推理路径；③针对慢速路径设计了Temporal Markup Prompting（TMP），一种三步结构化链式思维（CoT）策略，显著提升对多阶段、因果关系强的复杂查询的定位精度。

**🔧 技术方法**

关键技术包括：查询条件DPP关键帧抽取与自适应终止；谱熵难度评估与路由决策；结构化推理的Temporal Markup Prompting；使用大规模视觉‑语言模型（LLaVA‑1.6‑7B）作为编码器；对关键帧进行语义与时间双重相似度核融合。

**📊 数据集**

实验数据集主要包括 Charades‑STA 与 ActivityNet Captions（val_2、test 等拆分）。

**📈 对比分析**

与众多零样本基线（TFVTG、TAG、VTimeLLM‑13B、VChat‑7B 等）以及部分弱监督/全监督方法比较，DART 在 IID 与多种 OOD（时间偏移、分布偏移、文本词汇扩展、跨数据集迁移）设置下均取得了 SOTA 结果。具体而言，mIoU 在 Charades‑STA 上提升约 3.2 点，在 ActivityNet Captions 上提升约 3.4 点；R@0.3/0.5/0.7 指标也同步提升，并且在不增加帧数（平均 12 帧）且保持或提升推理速度（3.9 s/query）下实现了性能突破。

**⚠️ 局限性**

局限性包括：①对大规模 VLM 的依赖，模型规模与算力成本仍高；②谱熵判定难度的阈值选择仍需经验调优，可能对不同数据集泛化存在一定敏感性；③在极度复杂或长时序任务中，结构化推理仍可能受到提示模板与 LLM 生成误差的影响；④相比全监督方法，仍存在一定性能差距。

---

## 304. Not All Prediction Targets Keep Training-Free Diffusion Guidance on the Manifold

**arXiv ID:** 2607.00647 | [PDF](https://arxiv.org/pdf/2607.00647v1)

**作者:** Yunsung Lee `[一作]` (MAUM.AI), Hyeongmin Lee `[通讯]` (Seoul National University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了训练无关的扩散模型引导（Training-Free Guidance，TFG），并系统分析不同预测目标（ϵ、v、x）对保持生成样本在数据流形上的影响；提出了基于流形的子类FID（Child FID）评估指标和细粒度鸟类基准，验证了x-预测目标在保持流形一致性方面最具鲁棒性。

**💡 创新点**

创新点包括：①理论证明预测目标的误差放大层级，解释了ϵ-预测在高噪声阶段易导致样本漂移；②提出Child FID评估方法，能够区分真正的流形保持与伪造的分类成功；③在细粒度鸟类基准上进行指导强度扫帚，展示x-预测在各种任务（分类、风格迁移、逆问题）中最稳健，并通过跨维度交叉线实验验证误差随维度增大而放大的现象。

**🔧 技术方法**

使用的技术主要有：训练无关引导（TFG）、流形匹配、梯度引导（如DPS、LGD、FreeDoM）、三种预测目标（ϵ、v、x）与对应的恢复公式、子类FID评估、Precision/Recall（DINOv2）、Gram距离与内容准确率、像素/潜在空间的扩散变换器。

**📊 数据集**

数据集方面：ImageNet 256×256的143种细粒度鸟类（嵌套父类），二维交叉线多维投影数据，WikiArt 4种风格的风格迁移实验，以及高斯去模糊和4×超分的逆问题数据。

**📈 对比分析**

实验通过同一架构、相同训练的不同预测目标模型进行对比，利用指导强度扫帚绘制P-FID、Validity、Child FID以及Precision/Recall的Pareto曲线。结果表明：在细粒度鸟类基准上，x-预测在相同Validity下Child FID比ϵ-预测低5.2点；在交叉线实验中，x-预测在D=512时保持93%流形率，而ϵ-预测仅0.5%；在风格迁移和逆问题中，x-预测能够在更强指导下保持内容与风格的平衡，ϵ-预测则在高指导下快速失真。

**⚠️ 局限性**

限制包括：1) 由于模型在架构、空间（像素/潜在）和参数规模上存在差异，无法单纯隔离预测目标的影响；2) 理论分析基于Lipschitz能量函数和充分训练的假设；3) 所有实验仅在256×256分辨率下进行，未验证更高分辨率或文本/视频等更复杂场景下的表现。

---

## 305. SchedCheck: Schedule-Robustness Analysis for Event-Driven Block Programs

**arXiv ID:** 2607.00623 | [PDF](https://arxiv.org/pdf/2607.00623v1)

**作者:** Yuan Si `[一作]` (University of Waterloo), Jialu Zhang `[通讯]` (University of Waterloo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究 Scratch 语言中的调度敏感性，定义可实现的调度空间并设计基于依赖关系的偏序探索来完整检测和定位调度缺陷，随后在真实项目中评估其有效性。

**💡 创新点**

创新点包括：①把 Scratch 运行时的可实现调度空间限定为层顺序排列；②提出一次按依赖关系生成代表即可覆盖所有可实现调度的偏序探索；③结合观察透镜与资源模型实现完整且无遗漏的检测与定位；④把检测结果与实际 Remix 产生的层次变化关联。

**🔧 技术方法**

主要技术手段有：资源模型（footprint）与冲突判定、依赖关系图、偏序探索、完整枚举 oracle、观察透镜层次、Headless Scratch VM 及随机采样。

**📊 数据集**

使用数据集包括：224 个课堂学生项目、250 个随机公开 Scratch 项目以及 32 个人工生成的故障基准项目。

**📈 对比分析**

评估方法：对可枚举项目使用完整枚举 oracle 验证偏序探索无遗漏；在课堂与公开数据集上分别发现 21% 与 17.6% 的调度敏感项目；在 32 个 benchmark fault 上检测并定位全部故障；与静态检查器、输入生成器对比，证明互补覆盖。性能上，偏序探索显著减少运行次数，平均每个项目仅需数十至百次运行即可完成。

**⚠️ 局限性**

局限性包括：仅考虑层顺序的初始调度选择，忽略子 tick 的细粒度交错；对未能加载的项目和超大项目仅使用抽样；资源模型初始不完整导致四次修正；使用 30 tick 的观测窗口可能漏掉更深层次的调度效应；方法主要针对合作式事件循环，尚未验证在其他并发模型上的适用性。

---

## 306. Fluid-Spatiotemporal Stochastic Geometry: Information Flow in Non-Stationary Fields

**arXiv ID:** 2607.00616 | [PDF](https://arxiv.org/pdf/2607.00616v1)

**作者:** Wen-Yu Dong `[一作]` (China Telecom Research Institute), Sheng Chen `[通讯]` (University of Southampton)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了一种新型非平稳网络模型——流动时空随机几何（F-STSG），通过将节点分布视为连续的“数字潮汐”并求解其逆向边界值问题，推导出唯一的最小能量速度场、信息流向量以及相应的宏观性能指标。

**💡 创新点**

创新点在于：①将离散随机点过程与连续流体力学耦合，实现了从可观测的节点密度演化反推出隐藏的宏观运输动力学；②利用最优传输与Helmholtz分解理论得到唯一解，解决了传统连续模型的逆向未定性问题；③给出能量-容量平衡的通用缩放律和源信道双向性理论，揭示网络密度与协作成本的逆平方关系。

**🔧 技术方法**

核心技术包括：随机几何与Poisson点过程理论、连续介质动力学（守恒方程）、最优传输与能量最小化、Helmholtz分解、拉格朗日-欧拉耦合、Green函数与谱展开、有限差分/FFT数值求解，以及信息理论中的熵与信道容量分析。

**📊 数据集**

实验采用合成的非平稳用户密度场（如高斯热点漂移、热点合并等）作为输入，并通过大规模蒙特卡罗模拟生成相应的Poisson点集，以检验流体模型、逆向求解器和性能指标的准确性。

**📈 对比分析**

与传统静态随机几何和前向交通流模型比较，F-STSG 的预测误差随节点数 N 以 1/√N 递减，信息流向量与实际边界流量在 0.1% 以内吻合；能量-容量缩放律 λ*∝√(P_static/κ) 在数值实验中与理论曲线高度一致，且源信道对齐实验显示熵耗散率约被降低 65%。

**⚠️ 局限性**

局限性包括：仅适用于保守无源（S=0）且无旋转（∇×v=0）的流动场；假设基础基础设施为平稳泊松点过程，无法直接处理非泊松或聚集性基站布置；对高频、强耦合或显著湍流的微观运动未建模；以及在非绝热、强时变网络中模型误差需进一步评估。

---

## 307. Auditing Forgetting in Limited Memory Language Models

**arXiv ID:** 2607.00605 | [PDF](https://arxiv.org/pdf/2607.00605v1)

**作者:** Arya Raeesi `[一作]` (University of California, Berkeley), Hanna Roed `[通讯]` (University of California, Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了因果审计框架，对有限记忆语言模型（LMLM）在数据库删除后是否真正遗忘知识进行细粒度评估。

**💡 创新点**

将删除评估拆分为参数泄漏、检索介导正确性和检索伪影三部分，并通过实验验证残留主要来自检索图而非模型参数。

**🔧 技术方法**

采用三种干预（FULL、DEL-ON、DEL-OFF）、检索追踪、别名闭包删除、六种提示格式，以及all-MiniLM-L6-v2检索器进行实验。

**📊 数据集**

使用公开的54.6M Triplet LMLM数据库以及自制的12个包含国家、政治家、体育等领域的数据库，分别设定Base、Alias、Noise、Collision四种检索拓扑。

**📈 对比分析**

对12,228条删除事实在13个数据库、六种提示下进行完全交叉评估，结果显示参数泄漏率≈0.11%，检索介导残留最高达13.6%，提示对残留影响有限。

**⚠️ 局限性**

局限性在于仅使用单一382M LLaMA2风格LMLM、固定检索阈值和贪婪解码，实验规模较小，未覆盖多步推理、非英语或人工评估等更广泛场景。

---

## 308. Multi-Turn Agentic Scientific Literature Search via Workflow Induction

**arXiv ID:** 2607.00597 | [PDF](https://arxiv.org/pdf/2607.00597v1)

**作者:** Jisen Li `[一作]` (University of Illinois Urbana Champaign), Bingxin Zhao `[通讯]` (University of Pennsylvania)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多轮文献检索代理，将科学检索转化为可执行的DAG工作流，支持用户反馈直接编辑工作流；

**💡 创新点**

创新点在于将检索过程视为工作流诱导，通过显式可编辑的工作流结构让模型能根据反馈精准调整检索策略；

**🔧 技术方法**

使用LLM（-9B）结合工具集（关键词检索、引用扩展、过滤、评分、重排序、证据抽取等），并通过两阶段训练（工作流模仿+偏好优化）实现工作流生成与编辑；

**📊 数据集**

构建包含5种检索方向（predecessor, successor, sibling, benchmark, survey）的锚文+查询对数据集，生成隐藏金集并采用Semantic Scholar元数据；

**📈 对比分析**

与多种基线（GPT‑5.4、Qwen3.5‑9B、OpenAI DeepResearch等）比较，-9B在多轮交互下Hit@5提升至77.0、MRR提升至59.4、nDCG@10提升至32.5，工作流执行错误从9.5%降至0%，且成本远低于大型专有系统；

**⚠️ 局限性**

局限包括：预定义工具库可能无法覆盖所有领域需求；训练数据来源于教师模型，可能继承偏见；评测聚焦于计算机科学文献，模拟用户反馈不等同真实交互；

---

## 309. Semantic-Guided Reading Order Reconstruction in Historical Armenian Newspapers with LLMs

**arXiv ID:** 2607.00596 | [PDF](https://arxiv.org/pdf/2607.00596v1)

**作者:** Chahan Vidal-Gorène `[一作]` (National School of Chartes PSL), Victoria Khurshudyan `[通讯]` (Inalco)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了历史亚美尼亚报纸的注释数据集与专用 OCR 模型，并对多种阅读顺序恢复方法进行基准测试。

**💡 创新点**

创新之处在于提出将语义区域检测与生成式 LLM 结合的混合管线，显著降低读取顺序错误并可用于低资源下的快速标注。

**🔧 技术方法**

所用技术包括 YOLOv11/DocLayout-YOLO 的布局检测、基于语义块的局部拓扑排序，以及 Mistral‑8B、Gemini 等生成式 LLM 进行全局顺序决策。

**📊 数据集**

使用了包含 66 页（1913–2009 年）历史亚美尼亚报纸的全新手工注释数据集（11 类区块），并通过合成数据扩充至 500 张图像。

**📈 对比分析**

在单页任务中，SD+LLM 方案的 Kendall τ 和 Spearman 误差分别为 0.07 与 0.20，较最强几何基线下降 76%，在多页任务中同样保持较低误差。

**⚠️ 局限性**

局限性包括 LLM 调用成本与延迟、对西方亚美尼亚语言覆盖不足、仍需人工校验以及对复杂布局的挑战。

---

## 310. Active Spatial Guidance: Eliminating Injected Positional Mechanisms in Vision Transformers

**arXiv ID:** 2607.00580 | [PDF](https://arxiv.org/pdf/2607.00580v1)

**作者:** Cong Liu `[一作]` (Yancheng Institute of Technology), Simon X. Yang `[通讯]` (University of Guelph)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为主动空间引导（Active Spatial Guidance, Guidance）的训练目标，旨在消除视觉变换器（Vision Transformers, ViTs）中的注入位置机制，通过训练时的辅助信号来诱导空间组织。

**💡 创新点**

创新点在于通过训练时的2D坐标回归损失来替代传统的注入位置机制，从而在推理时实现无位置注入的ViT编码器，且不增加推理时的计算负担。

**🔧 技术方法**

使用了DINOv3 ViT骨干网络，并引入了一个轻量级的辅助预测头，仅在训练期间使用，推理时将其移除。

**📊 数据集**

使用了ImageNet-100、ADE20K和Hypersim数据集进行实验，评估了分类、语义分割和单目深度估计任务的性能。

**📈 对比分析**

与多种注入位置机制（如学习的绝对位置嵌入和旋转位置嵌入）进行了比较，结果显示Guidance在相同训练条件下的性能优于这些基线，且在分辨率转移和多分辨率训练下表现出更好的鲁棒性。

**⚠️ 局限性**

限制在于该研究仅限于从头开始训练DINOv3骨干网络，且在中等输入分辨率下进行，未来需要在大规模预训练和更广泛的架构及下游任务中验证其有效性。

---

## 311. HARC: Coupling Harmfulness and Refusal Directions for Robust Safety Alignment

**arXiv ID:** 2607.00572 | [PDF](https://arxiv.org/pdf/2607.00572v1)

**作者:** Shei Pern Chua `[一作]` (Tsinghua University), Fangzhao Wu `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对对齐LLM的安全性进行内在表示分析，并提出将有害识别方向与拒绝方向在提示端和生成端耦合的HARC方法，以提高对越狱攻击的鲁棒性。

**💡 创新点**

创新点在于将有害性与拒绝性这两个在提示端与生成端均可区分的线性方向通过双位置的margin hinge损失耦合，使模型在任何阶段检测到有害信号时自动触发拒绝，从而在不降低整体能力或导致过度拒绝的前提下显著提升安全性。

**🔧 技术方法**

技术包括：线性方向提取（差均值法）在提示端和生成端；LoRA微调与添加margin hinge损失实现耦合；KL正则化和交叉熵监督以保持能力；EMA与层分数选择以动态更新耦合层。

**📊 数据集**

使用公开的Circuit Breakers训练集（有害提示与对应拒绝文本）和UltraChat（无害提示）进行微调，评估时采用JailbreakBench、XSTest、CoCoNot、MMLU、GSM8K、IFEval、HumanEval、MT-Bench，并用GPT‑4作为判定器。

**📈 对比分析**

与六种主流安全方法（SFT、DPO、Circuit Breakers、RepBend、CAST、无训练）对比，HARC在三类指标（攻击成功率、过度拒绝率、通用能力）上均保持或提升，尤其在攻击成功率上下降 4–5 倍，过度拒绝率几乎不变，通用能力保持与基线相当；在 70B 级别模型中也能维持近零攻击成功率。

**⚠️ 局限性**

局限性包括：仅对LoRA微调，未测试全参数或其他微调方案；对权重级攻击易被破解；未评估针对已知耦合方向的自适应攻击；依赖基模型已存在的有害方向信号，若该方向弱则效果受限；未验证在闭源模型或多模态模型上的表现。

---

## 312. AdaBoosting Text Prompts for Vision-Language Models

**arXiv ID:** 2607.00684 | [PDF](https://arxiv.org/pdf/2607.00684v1)

**作者:** Seokhee Jin `[一作]` (KT Corporation), Jungseul Ok `[通讯]` (Pohang University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于AdaBoost的文本提示增强（TPB）框架，用于少量样本场景下的视觉‑语言模型（VLM）的文本提示构造与优化。

**💡 创新点**

将每一组类级文本提示集合视为弱分类器，采用AdaBoost逐轮聚合多个弱学习器，显式关注并纠正误分类样本，从而实现更强的shot‑可扩展性和跨模型的可迁移性。

**🔧 技术方法**

使用AdaBoost（SAMME.R）迭代加权策略；Greedy Prompt Composition (GPC) 在预生成的LLM提示池中选择最佳提示；结合图像增强来防止过拟合；所有提示保持自然语言可解释性。

**📊 数据集**

在十一种图像分类数据集上进行实验，包含 ImageNet‑1K、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT 和 UCF101。

**📈 对比分析**

与零样本 CLIP、DCLIP、CuPL、CoOp、PEZ、LLMbo、ProAPO 等基线进行对比。TPB 在 1‑shot 与 16‑shot 场景下平均提升约 6.5pp，且在从 ViT‑B/32 转移到更大 ViT‑L/14、ViT‑H 等模型时仍能保持并进一步提升 shot‑驱动的性能，明显优于其他文本提示方法。

**⚠️ 局限性**

构造成本较高，AdaBoost 的加权机制可能过度关注噪声或异常样本；提示池固定可能限制对细粒度或模糊类别的灵活适配。

---

## 313. Distributed Online Bandit Submodular Maximization with Bounded Sampling Violations

**arXiv ID:** 2607.00680 | [PDF](https://arxiv.org/pdf/2607.00680v1)

**作者:** Bin Du `[一作]` (Nanjing University of Aeronautics and Astronautics), Dengfeng Sun `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种分布式在线子模函数最大化框架，支持完整信息和Bandit反馈，兼顾分区Matroid约束；

**💡 创新点**

首次在此设置下实现$(1-1/e)$近似的子线性后悔率，并设计可在不违约的情况下逼近采样违规概率的有限随机pipage舍入；

**🔧 技术方法**

利用多线性扩展、Frank–Wolfe、追踪专家(RFTL)、一点梯度估计、进步界限随机pipage舍入以及分布式最大一致性；

**📊 数据集**

在多智能体信息收集（信息源与检索点）问题上进行仿真，采用1000个信息源、16个检索点、4-32个代理；

**📈 对比分析**

与中心化Meta‑Frank‑Wolfe及响应式Bandit算法对比，后者后悔率高、违规率高；本算法后悔率与中心化相当，违规率降低约84%；

**⚠️ 局限性**

只能保证渐近无违规，仍存在每步不可行查询；在高维度和大网络时一致性误差增大，且需离散化或更精细的可行域约束

---

## 314. ABot-M0.5: Unified Mobility-and-Manipulation World Action Model

**arXiv ID:** 2607.00678 | [PDF](https://arxiv.org/pdf/2607.00678v1)

**作者:** Ronghan Chen `[一作]` (AMAP CV Lab), Xinyuan Chang `[通讯]` (AMAP CV Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ABot-M0.5，一个统一的世界-动作模型，用于移动操纵任务。

**💡 创新点**

创新点：① 引入帧级潜在动作作为时间粒度桥梁；② 设计双层 Mixture‑of‑Transformers 以分离移动与操作动作子空间；③ Dream‑Forcing 训练策略，将自我生成的未来视觉作为动作学习的条件，解决训练‑测试分布差距。

**🔧 技术方法**

技术：Conditional Flow Matching (CFM)、3D VAE、文本编码器（如 UMT5）、三阶段分层生成（视频→潜在动作→动作）、双层 MoT、Dream‑Forcing 两阶段前向策略。

**📊 数据集**

数据集：OXR、OXR‑AugE、Agibot‑Beta、RoboCOIN、RoboMind、Galaxea、InternData‑A1 等大规模多机器人、多任务多视角数据；以及 RoboCasa365、RoboTwin 2.0、LIBERO/LIBERO‑Plus 等评测基准；真实实验使用 Agilex Piper 机器人。

**📈 对比分析**

对比方法：VLA 模型、Fast‑WAM、Lingbot‑VA、HoloBrain‑0、AttenA+ 等。ABot‑M0.5 在 RoboCasa365 长周期任务中实现 94% 以上成功率，在 RoboTwin 2.0 与 LIBERO‑Plus 上均优于现有基线，且在真实机器人上获得 70%–80% 的成功率和高 Process Score。

**⚠️ 局限性**

局限：仍需大规模预训练与多任务数据；在极端动态或高度噪声的环境中未来视频预测误差可能导致动作失效；模型规模大，推理时仍需进一步加速以满足实时部署需求。

---

## 315. Diffusion-Based Multi-Class Normality for OOD Detection: An Application to CDP Authentication

**arXiv ID:** 2607.00609 | [PDF](https://arxiv.org/pdf/2607.00609v1)

**作者:** Bolutife Atoki `[一作]` (Université Lumière Lyon 2), Carlos Crispim-Junior `[通讯]` (Université Lumière Lyon 2)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于扩散模型的多类别正态性框架，用单一类条件模型在仅使用正品CDP训练的情况下实现无监督的OOD检测，并应用于CDP的鉴权；

**💡 创新点**

创新点在于：①将扩散模型用于多类别正态性建模，单模型同时学习多种正品分布；②引入双模板掩码隐藏互补区域，仅对隐藏像素进行重建误差评分，减少对可见二值结构的依赖；

**🔧 技术方法**

使用了控制网络（ControlNet）与潜在扩散模型（Latent Diffusion）、CLIP文本编码器、VAE编码器，训练过程中采用混合精度、AdamW、余弦学习率调度等技术；

**📊 数据集**

使用Indigo 1×1 Base数据集，包含720个二值模板在两台正品打印机（HP55、HP76）打印的4320张样本，四种伪造类型用于OOD评估；

**📈 对比分析**

与NCC、Pix2Pix、OC‑SVM、Tutt等传统和改造生成基线相比，本文方法在多类别鉴权下取得最低P_err（0.055）和最高AUROC（0.975），显示出更好的跨类别校准和检测性能；

**⚠️ 局限性**

局限性包括：仅在两台打印机的场景下验证，缺乏对更大、多样化打印/扫描配置的评估；阈值仍为全局固定，可能无法适应不同操作点；

---

## 316. YOMI-Bench: A Benchmark for Evaluating Kanji Reading and Phonological Understanding of LLMs for Japanese

**arXiv ID:** 2607.00664 | [PDF](https://arxiv.org/pdf/2607.00664v1)

**作者:** Ryota Mibayashi `[一作]` (Kobe University), Hitomi Yanaka `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了YOMI-Bench基准，用于评估大型语言模型在日语汉字读音和语音理解方面的能力。

**💡 创新点**

创新点在于设计了七类多任务（包含二分类、单/多选预测、韵母匹配生成）专门考察日语汉字读音的基准，且公开发布。

**🔧 技术方法**

采用多重提示、正负样本构造以及正则表达式提取答案等技术对模型进行评估，并通过多轮对照实现鲁棒性验证。

**📊 数据集**

使用了日语常用汉字集（Jōyō Kanji 2136字）并随机抽样100个单读、100个多读汉字，结合词语例句及韵母表生成评测数据。

**📈 对比分析**

通过在10个模型（包括4个日语专用开源模型和5个商业模型）上进行基准评测，结果显示即使是日语专用模型在读音推断与韵母生成任务上表现仍低，商业模型虽略优但在生成任务上仍明显不足。

**⚠️ 局限性**

局限性包括仅覆盖常用汉字且未涉及罕见或专业词汇，评测侧重日语读音，结果可能不适用于其他文字或语言。

---

## 317. What's a Credit Worth? A Market Framework for Attribution-Aware Compensation in Generative Music

**arXiv ID:** 2607.00641 | [PDF](https://arxiv.org/pdf/2607.00641v1)

**作者:** Luyang Zhang `[一作]` (Carnegie Mellon University), Chris Donahue `[通讯]` (Carnegie Mellon University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个基于数据归因的生成音乐补偿框架，推导了闭式支付规则，并在音频扩散和符号音乐模型上对归因方法进行实证评估。

**💡 创新点**

将归因噪声与契约理论结合，提出了以归因信息量为依据的版税与固定费分配模型；给出误差导致的福利损失公式；首次对音乐AI归因方法进行基于福利的定量基准。

**🔧 技术方法**

使用EKFAC、D‑TRAK、TRAK、LoGra归因技术；Stable Audio Open扩散模型；Anticipatory Music Transformer；留一目录法评估；高斯噪声与均值方差效用模型；线性合同与市场仿真。

**📊 数据集**

22个MTG‑Jamendo创作者目录（Creative Commons）；100个TheoryTab商业旋律；基于这些目录生成的音频和符号输出；用户请求分布来自平台日志。

**📈 对比分析**

通过秩相关和标量校准评估归因方法；EKFAC与D‑TRAK在秩上相近但校准尺度差约10³；LoGra优于TRAK；目前方法的归因精度不足以实现版税支付，需提升25‑40倍SNR；当前方法仅弥补约10%福利缺口。

**⚠️ 局限性**

归因噪声仍较大，导致固定费主导；模型假设线性合同和高斯噪声，未考虑目录动态增长或多模态；可能存在归因操纵；实验样本有限（22/100个目录）。

---

## 318. Stacked Ensemble Learning for Abdominal Aortic Aneurysm Segmentation in CT Angiography

**arXiv ID:** 2607.00633 | [PDF](https://arxiv.org/pdf/2607.00633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 319. Uncertainty-aware tree height change regression

**arXiv ID:** 2607.00638 | [PDF](https://arxiv.org/pdf/2607.00638v1)

**作者:** Max Gaber `[一作]` (University of Copenhagen), Martin Brandt `[通讯]` (University of Copenhagen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究构建并公开了CHC冠层高度变化数据集，涵盖北西班牙10598 km²，提供三米分辨率的连续高度变化及其像素级不确定性，并利用该数据评估多种地理空间基础模型（GFMs）的连续高度变化回归能力。

**💡 创新点**

创新点在于首次发布公开的高分辨率连续冠层高度变化与不确定性数据集，并在训练和评估中引入基于不确定性的加权MSE和阈值MSE损失，从而更真实地反映标签噪声。

**🔧 技术方法**

技术方法包括：利用自监督预训练的GFMs（如Scale‑MAE、Prithvi、DOFA等）与传统UNet/ResNet进行对比；通过多层感知器对ALS点云的采样不确定性进行建模；使用多时相PlanetScope影像提取时空特征并结合轻量级时间注意力编码器。

**📊 数据集**

主要使用数据集为CHC数据集（2018/2023两次ALS调查的DSM差值、对应PlanetScope时序影像），对比实验中亦采用公开的ECHOSAT等基准数据。

**📈 对比分析**

在PANGAEA框架下，用RMSE、MAE、nMAE、R²等指标对模型进行评估；结果显示所有GFMs均未能超过基于阈值MSE训练的监督UNet，Prithvi与DOFA在整体性能上相对最佳，但仍逊色于专门的ECHOSAT模型；引入不确定性加权能显著提升部分模型表现。

**⚠️ 局限性**

局限性包括：GFMs在细粒度像素级连续高度变化任务上表现有限，难以捕捉微小变化；数据仅覆盖西班牙部分地区，且ALS不确定性估计依赖模拟与静态参考，可能影响精度；模型未充分利用跨传感器的时空信息，导致对高度变化的感知仍不够精确。

---

## 320. Checked Program Recovery from Execution Video: A Sound Oracle for Untrusted Generators

**arXiv ID:** 2607.00635 | [PDF](https://arxiv.org/pdf/2607.00635v1)

**作者:** Yuan Si `[一作]` (University of Waterloo), Jialu Zhang `[通讯]` (University of Waterloo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于执行视频恢复 Scratch 程序的完整管道，并通过可信检查器（两层验证 oracle）验证候选程序是否在给定“镜头”下与原始程序等价；

**💡 创新点**

创新点包括：① 设计了两层分层验证 oracle，静态层可签发等价证书、渲染层仅能拒绝，保证零误接受；② 用镜头参数化观测映射，正式定义程序等价类与不可识别的残差；③ 构建源无关的神经符号感知-合成管道，并引入主动推理循环来恢复被输入条件屏蔽的行为；④ 提供闭环无标注基准，通过渲染→恢复→验证循环实现可复现评估；

**🔧 技术方法**

核心技术包括：视图感知与去噪、最小二乘拟合、基于模式的符号合成、镜头等价检查器、分层验证 oracle、FastRender 确定性渲染器、主动推理实验设计、统计置信区间评估；

**📊 数据集**

实验数据集包括：1）基于 Scratch 内部词法与动作约束的程序生成器（约 798 程序，十个种子）；2）30 个真实 Scratch 项目；3）Vision‑Language 模型输出（MiniMax‑M3、Claude Sonnet 4.6）；

**📈 对比分析**

与两种 Vision‑Language 模型比较：结构化管道在 40 个单精灵程序上 100% 恢复（CI 91.2%‑100%），模型单发 0/1/29/3/40，Oracle 始终拒绝错误答案；对近似误差、帧率降至 5fps、彩色背景、H.264 压缩、真实录屏等多种鲁棒性测试均保持 ≥98% 的恢复率；在真实项目上的恢复率为 14%，但无误接受；

**⚠️ 局限性**

局限性：① 仅能恢复在词法覆盖范围内的程序；② 需要已知的资产清单，无法从像素恢复服装/背景；③ 只能观测可见行为，隐藏内部状态、未执行的分支、并发顺序无法恢复；④ 目前不支持多段运动、复杂广播/数据依赖控制等；⑤ 仍依赖可用的镜头等价检查器，适用性受限于语言运行时的可验证性。

---

## 321. Loss Smoothing for Stable Adaptation Under Distribution Shift

**arXiv ID:** 2607.00634 | [PDF](https://arxiv.org/pdf/2607.00634v1)

**作者:** Darshan Patil `[一作]` (Chandar Research Lab), Sarath Chandar `[通讯]` (Chandar Research Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的适应方法，称为损失平滑，通过在适应开始时在源目标和目标目标之间进行插值，来实现更平滑的过渡，从而改善神经网络在分布变化下的适应性。

**💡 创新点**

创新点在于引入损失平滑这一轻量级目标插值方法，旨在通过平滑的目标过渡来减少适应过程中的脆弱性，保持源分布中的有用特征。

**🔧 技术方法**

使用了损失平滑技术，该技术通过在源目标和目标目标之间进行凸插值来实现平滑过渡。

**📊 数据集**

在多个数据集上进行了实验，包括受控的监督任务转移、预训练视觉适应、离线到在线强化学习和语言模型微调。

**📈 对比分析**

与标准的适应方法相比，损失平滑在所有实验设置中均表现出一致的性能提升，表明平滑的目标过渡是模型适应的有效工具。

**⚠️ 局限性**

限制在于需要访问或估计源目标，这可能涉及重放数据、存储目标或持续访问先前的训练数据。此外，损失平滑在适应开始时需要更多的计算资源，且平滑时间过长可能会导致学习者依赖过时数据。

---

## 322. Low Perplexity is Repetition: A One-Dimensional Self-Conditioning Attractor in Continuous Diffusion LMs

**arXiv ID:** 2607.00588 | [PDF](https://arxiv.org/pdf/2607.00588v1)

**作者:** Shuai Zhang `[一作]` (Zhejiang University), Zhenzhong Lan `[通讯]` (Westlake University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究连续扩散语言模型在自条件训练下产生过度重复的问题，并提出ACE方法在每一步反馈中减去单一方向以消除重复并保持文本质量。

**💡 创新点**

创新点在于发现自条件循环中的一维收敛吸引子导致重复，并通过无标签、一次性估计的差分均值方向在每步反馈中减去该方向，既低成本又高效。

**🔧 技术方法**

使用线性稳定性分析、差分均值方向估计、Self‑conditioning 反馈拆解、计算到干净文本（compute‑to‑clean）评估等技术。

**📊 数据集**

主要数据集包括 OpenWebText、BBC/XSum 以及 LangFlow、Plaid 等模型的生成样本；实验覆盖 ELF 105M/342M/652M 等规模。

**📈 对比分析**

通过比较 Gen‑PPL、清洗后 Gen‑PPL、重复率、NFE 等指标，ACE在保持或提升清洗后 Gen‑PPL 的同时，将重复率降至接近人类水平，并且比完全重采样或禁用自条件的做法低 1.5–5 倍的计算成本。

**⚠️ 局限性**

局限性包括需预先设定人类重复阈值；ACE 针对自条件扩散模型，其他架构可能需要不同方向；在超大 λ 或极端采样设置下可能导致文本退化，需在闭合窗口内使用。

---

## 323. Decision-focused Sparse Tangent Portfolio Optimization

**arXiv ID:** 2607.00581 | [PDF](https://arxiv.org/pdf/2607.00581v1)

**作者:** Haeun Jeon `[一作]` (Korea Advanced Institute of Science and Technology), Woo Chang Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在稀疏切线投资组合优化中，提出了一种端到端的决策聚焦学习框架，通过可微的决策层直接优化Sharpe比率并训练预测模型。

**💡 创新点**

创新点在于：①将Sharpe比率最大化转化为满足Disciplined Parametrized Programming（DPP）规则的凸优化层；②设计了一个平滑的top‑k选择器，使得稀疏约束可微并保持精确的卡迪纳尔数；③将预测、资产选择与再优化完整地连接起来，实现梯度在预测模型中的传播。

**🔧 技术方法**

采用了凸二次约束优化（QCQP）与Cholesky分解、CVXPYlayers进行可微求解；使用可微top‑k运算和回归式损失（决策损失+MSE）进行训练；训练过程中使用滚动协方差矩阵。

**📊 数据集**

使用从Yahoo Finance获取的四个主要指数（EuroStoxx50、FTSE100、KOSPI200、Nikkei225）每日收盘价数据，构建100天滚动窗口的收益序列。

**📈 对比分析**

与传统历史均值、预测聚焦（PFL）以及三种稀疏优化模型（OSCAR、SD-relaxation、mSSRM-PGA）对比，实验表明在中等至大规模资产组合上，DFL在样本外Sharpe比率上具有竞争力甚至更优；在小规模组合中差距较小，但总体波动性更低。

**⚠️ 局限性**

局限性包括：①可微top‑k选择器在资产规模或稀疏度增大时计算开销显著；②目前模型允许做空且未加入实务约束（如长期持仓、杠杆或交易成本）；③方法专注于稀疏切线Sharpe最大化，推广到其他目标或约束仍需进一步研究。

---

## 324. Safe Alone, Unsafe Together: Safeguarding Against Implicit Toxicity When Benign Images Combine

**arXiv ID:** 2607.00576 | [PDF](https://arxiv.org/pdf/2607.00576v1)

**作者:** Jiaxian Lv `[一作]` (Tsinghua University), Minlie Huang `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了多图隐式毒性检测，构建了覆盖七类风险的多图安全数据集，并通过结构化推理轨迹训练模型实现解释性判断。

**💡 创新点**

创新点在于：①首次正式定义多图隐式毒性问题并系统分析检测挑战；②提出自动生成与人工校验相结合的数据构建管线；③利用教师模型生成结构化推理轨迹，并通过渐进压缩得到高质量监督信号；④在仅图像输入下实现跨图安全推理。

**🔧 技术方法**

使用技术包括：基于 Qwen3‑VL‑8B‑Instruct 的视觉语言模型微调；逐步压缩的推理轨迹生成（实体定位→跨图关系→安全结论）；跨图实体对齐与关系推理；结构化输出与标准交叉熵训练。

**📊 数据集**

采用了自研的 MIIDataset（1,434 条多图安全实例），其中包含来自 BLINK、MUIRBENCH 的 640 条样本以及通过 LLM 自动生成并人工审核的 794 条样本，覆盖“Gore & Disturbing、Regulated Goods、Sexual Content、Violence & Conflict、Financial & Economic Crime、Self‑Harm、Hate & Extremism”七大风险类别。

**📈 对比分析**

在内部 574 条平衡测试集上，与 4 个商业图像审核 API、GPT‑5.4、Claude Opus 4.6、Gemini 3.1 Pro、InternVL3‑8B、Qwen3‑VL‑8B、Llama‑Guard‑4‑12B 等基线对比，模型在整体准确率上达到 91.11%，在“Safe”和“Unsafe”子集上均表现平衡；在 OOD 单图安全基准（UnsafeBench、LlavaGuard）上也保持 77‑83% 的高准确率，优于现有主流模型。

**⚠️ 局限性**

局限性包括：①数据集覆盖的风险类别有限，缺乏对新兴或文化特异性图像风险的覆盖；②生成样本与真实社交媒体内容的分布可能存在差异；③模型在边界案例中仍存在过度组合误判与误判为安全的错误，需要进一步校准跨图语义推理。

---

## 325. BrainFIBRE: A Foundation Model via Information Decomposition for Brain Microstructure

**arXiv ID:** 2607.00573 | [PDF](https://arxiv.org/pdf/2607.00573v1)

**作者:** Zijian Dong `[一作]` (National University of Singapore), Juan Helen Zhou `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在英国生物银行55,592例数据上预训练BrainFIBRE基础模型，通过自监督信息分解学习三种NODDI微结构图像的独特、冗余和协同信息，并在多任务跨族群评估。

**💡 创新点**

提出自监督部分信息分解（SPID）与对抗性候选构造（CCC），实现无监督PID分解并通过Mixture-of-Experts实现可解释的微结构特征分离。

**🔧 技术方法**

采用3D ViT编码器、Mixture-of-Experts、对抗性候选构造、InfoNCE对比学习、均衡与熵正则的自监督框架。

**📊 数据集**

使用UK Biobank（55,592例）进行预训练，后续评估分别在UK Biobank保留集、HCP-Aging（630例）和SINGER（818例）上进行。

**📈 对比分析**

与单模态ViT、监督I2MOE、无监督BrainMVP及从零训练的基线进行对比，BrainFIBRE在年龄、性别、脑萎缩、认知、脑血管标志等任务上实现了最优或显著优于基线的表现。

**⚠️ 局限性**

模型训练资源高、对少数族群或低分辨率数据的泛化仍待验证，且理论上PID分解的可识别性未完全证明。

---

## 326. Accelerating Discrete Diffusion Models with Parallel-In-Time Sampling

**arXiv ID:** 2607.00773 | [PDF](https://arxiv.org/pdf/2607.00773v1)

**作者:** Yu Yao `[一作]` (University of Tokyo), Masashi Sugiyama `[通讯]` (RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于 Picard 迭代的并行时间 τ-leaping 抽样器，用于加速吸收式离散扩散模型的采样。

**💡 创新点**

首次提出并行时间 τ-leaping 的框架，证明了指数阶乘收敛，时间复杂度从 𝒪(d log S) 降至 𝒪(log(d log S)·log d)，并实现了实际 GPU 加速。

**🔧 技术方法**

结合连续时间马尔可夫链的泊松随机积分表示、Picard 迭代、第一碰撞截断（First‑Hitting Truncation）以及分块并行前缀求和等技术。

**📊 数据集**

在 ImageNet（MaskGiT 预训练）、OpenWebText（RADD 预训练）以及自制 2D 任务（棋盘、圆形）等数据集上进行评估。

**📈 对比分析**

与顺序 τ-leaping、FHS、Trapezoidal、并行解码等方法对比，实验表明在图像任务获得 1.86× 的速度提升、在文本任务把 NFE 降至 32 并保持甚至提升 perplexity，质量保持不变。

**⚠️ 局限性**

主要限制包括较高的内存占用（O(d²S)）、对多 GPU 的支持尚未充分验证，以及在极端高维或大词表场景下仍需进一步优化。

---

## 327. RACORN-1: Adaptive Recall-Preserving Speedup for Low-Selectivity Filtered Vector Search

**arXiv ID:** 2607.00768 | [PDF](https://arxiv.org/pdf/2607.00768v1)

**作者:** Yoonseok Kim `[一作]` (Naver Corporation), Gyusik Choe `[通讯]` (Naver Corporation)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对低选择性（filter-selectivity）下 ACORN‑1 的召回崩溃问题，提出了 RACORN‑1（含 Adaptive Search Fallback 与 Stride Sampling）和 RACORN‑1+（含 Adaptive Exact Fallback），实现过滤向量检索（Filtered Vector Search, FVS）的低延迟高召回。

**💡 创新点**

创新点包括：①将过滤失败的节点重新利用为临时桥接，恢复搜索路径（Adaptive Search Fallback, ASF）；②在桥接和两跳候选选择中使用 stride sampling 保持空间多样性；③在极低选择性下自动切换到精确线性扫描（Adaptive Exact Fallback, AEF）以获得 Recall 1；④理论证明 ASF 使搜索拓展规模与选择性无关，并通过实验验证 s‑independent 性能提升。

**🔧 技术方法**

使用技术：基于 HNSW 的图搜索框架；ASF/AEF 的运行时决策机制；BR（Bridge Ratio）参数化；Stride Sampling；离线与在线评估脚本；成本分析与随机图理论证明。

**📊 数据集**

数据集：SIFT 1M、GIST 1M、Text2Image 1M、Text2Image 40M（共 4 个不同维度与规模的数据集）。

**📈 对比分析**

比较方法：对同一索引使用 21 级选择性网格（100%–0.01%）进行 10 次测量，记录 Recall 与平均 Latency；与 HNSW、ACORN‑1、RACORN‑1、RACORN‑1+ 四种模式比较。结果显示：RACORN‑1 在 1%–0.3% 区间实现 9–26× 延迟降低并将召回从 0.48–0.72 恢复到 0.80–0.96；RACORN‑1+ 在 ≤0.1% 区间 Recall 1.0 且速度提升 20–75×；在正/负相关查询‑过滤关系下，RACORN‑1 维持 0.80–0.98 召回并比 HNSW 低 5–9×。

**⚠️ 局限性**

局限性：①BR 与 AEF 阈值仍需手工设定，缺乏自动化自适应；②仅在 HNSW 上验证，未探讨其它图结构或构造成本/内存开销；③实验覆盖的选择性范围和数据集有限，未检验更极端或不同语义分布的场景；④在极低选择性下的精确扫描切换仍有性能波动，需进一步优化。

---

## 328. GaussianFusion: Unified 3D Gaussian Representation for Multi-Modal Fusion Perception

**arXiv ID:** 2607.00746 | [PDF](https://arxiv.org/pdf/2607.00746v1)

**作者:** Xiao Zhao `[一作]` (Tencent), Kuifeng Su `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的3D高斯表示框架（GaussianFusion），用于摄像头与激光雷达的多模态特征融合，支持多任务3D感知（检测、语义占据预测）

**💡 创新点**

创新点包括：①将离散BEV网格替换为连续高斯分布，保留细节与几何不失真；②前向投影初始化摄像头高斯，提升对齐质量；③共享高斯编码器结合高斯先验的可变形注意力与迭代高斯更新，增强跨模态协同；④高斯到体素的高效池化与混合模型聚合，兼顾实时性与准确性

**🔧 技术方法**

技术手段：3D Gaussian Splatting、Gaussian Mixture Model、前向投影初始化、共享高斯编码器（含可变形注意力与高斯属性嵌入）、迭代高斯偏移更新、Gaussian-to-voxel pooling/MeanVFE、Transformer检测/占据头

**📊 数据集**

使用nuScenes（用于检测与占据预测）及Waymo Open Dataset（检测评测）

**📈 对比分析**

与BEVFusion、UniTR、EA-LSS、FusionFormer-S、GaussianFormer等基线对比，结果显示：在nuScenes验证集上检测NDS 74.0（+2.6）、mAP 71.7（+3.2）；占据mIoU 28.65（+1.11）；相比GaussianFormer仅使用30%高斯、速度提升450%；推理延迟132 ms，内存4271 MB，优于BEVFusion的156 ms/5140 MB

**⚠️ 局限性**

局限性：目前的时序扩展仅采用简单的历史高斯对齐，缺乏运动感知更新，导致4D场景建模不够连贯；在某些任务专用方法上仍略逊；对高斯初始化与迭代策略的依赖需进一步稳健验证

---

## 329. MosaicKV: Serving Long-Context LLM with Dynamic Two-D KV Cache Compression

**arXiv ID:** 2607.00760 | [PDF](https://arxiv.org/pdf/2607.00760v1)

**作者:** Sheng Qiang `[一作]` (Shanghai Jiao Tong University), Haibo Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种动态两维KV缓存压缩系统，用于支持极长上下文的LLM推理。

**💡 创新点**

创新点在于：①动态两维压缩策略—按向量重要性分块选择以及分段自适应压缩；②压缩KV缓存管理—Packed KV格式、PackedAttention以及异构CPU/GPU双缓冲异步处理。

**🔧 技术方法**

使用的技术包括SVD旋转、Per‑vector通道选择、动态分段压缩、Packed KV格式、PackedAttention、CUDA核心加速、异构CPU/GPU并行管理以及FlashAttention等。

**📊 数据集**

评估数据集包括LongBench、RULER以及多模型（LLaMA‑3.1‑8B、Qwen3‑4B、Ministral‑3‑8B）等。

**📈 对比分析**

与Full基线和Quest序列压缩方案对比，测量KV内存占用、解码延迟和吞吐率；实验结果显示：内存压缩3×、Attention加速16×、解码延迟降低4.8×、吞吐率提升7.3×，平均精度损失仅1.76%。

**⚠️ 局限性**

局限性包括：①目前仅针对解码阶段，预填阶段压缩效果有限；②仅利用CUDA核心，若未来HBM带宽显著提升则可能受限；③尚未与量化或分块预填等技术结合，进一步提升压缩率与性能。

---

## 330. Latency-Sensitive 5G RAN Slicing for Industry 4.0

**arXiv ID:** 2607.00707 | [PDF](https://arxiv.org/pdf/2607.00707v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 331. Foundation Model-driven Key Anatomy Frame Selection for Blind-sweep Ultrasound Fetal Birth Weight Estimation

**arXiv ID:** 2607.00745 | [PDF](https://arxiv.org/pdf/2607.00745v1)

**作者:** Le Ou `[一作]` (Shenzhen University), Dong Ni `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

利用盲扫超声视频在出生前48小时内预测胎儿出生体重，减少对经验丰富超声操作员的依赖。

**💡 创新点**

首次将盲扫技术用于胎儿出生体重估计，提出基于基础模型的关键解剖帧选择（AFS）和冗余感知特征压缩（RAFC）两级框架。

**🔧 技术方法**

采用FetalCLIP视觉‑语言基础模型提取解剖语义特征，结合Transformer、注意力聚合和重加权机制实现帧选择与回归。

**📊 数据集**

使用来自839名孕妇、870次检查的盲扫超声视频数据（每次检查6个10秒循环），共计约5.2万帧。

**📈 对比分析**

与Hadlock‑4、INTERGROWTH‑21st以及多种深度学习基线对比，方法实现MAE 161.3 g、MAPE 4.89%、10%误差率90.23%、15%误差率100%，优于所有对比模型。

**⚠️ 局限性**

仅在单中心数据上验证，盲扫视频长度、设备差异及多中心泛化性能仍需进一步评估。

---

## 332. Common Radio Resource Management Policy for Multimedia Traffic in Beyond 3G Heterogeneous Wireless Systems

**arXiv ID:** 2607.00705 | [PDF](https://arxiv.org/pdf/2607.00705v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 333. Partial Skeleton Visibility for Action Recognition: A Constrained Field-of-View Approach

**arXiv ID:** 2607.00716 | [PDF](https://arxiv.org/pdf/2607.00716v1)

**作者:** Yingjie Dai `[一作]` (Jiangnan University), Josef Kittler `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在受限视野（FoV）条件下进行骨架动作识别，并提出了基于可学习超图的PartialVisGraph框架

**💡 创新点**

创新点包括：①使用可学习的虚拟超边和稀疏软归属矩阵构建高阶超图，②引入单头样本自适应Transformer（SHSAT）对超边进行特征聚合，③加入可见性先验来抑制不可见关节对信息传播的干扰

**🔧 技术方法**

核心技术：超图卷积、可学习的软归属矩阵、单头Transformer、可见性先验、课程学习、Temporal CutMix

**📊 数据集**

采用NTU RGB+D 60和NTU RGB+D 120两个公开动作数据集，在模拟的三种FoV难度（easy/medium/hard）以及全视野场景下进行评估

**📈 对比分析**

与多种基准方法（HD-GCN、Hyper-GCN、Hyperformer等）进行对比，PartialVisGraph在受限FoV场景下提升显著（尤其在最难的hard级别上可达68.8%的相对提升），在全视野设置下也实现了93.8%/97.6%的最高准确率

**⚠️ 局限性**

局限性：①受限FoV数据仅为模拟生成，缺乏真实传感器遮挡或移动摄像头场景的验证；②超图构建和Transformer的计算成本相对传统GCN较高；③对不同骨架尺度或多人物交互场景的泛化仍待进一步研究

---

## 334. ClarifyCodeBench: Evaluating LLMs on Clarifying Ambiguous Requirements for Code Generation

**arXiv ID:** 2607.00711 | [PDF](https://arxiv.org/pdf/2607.00711v1)

**作者:** Zheng Fang `[一作]` (Peking University), Ge Li `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了ClarifyCodeBench基准，评估LLM在模糊需求下通过交互澄清后生成代码的能力，并提出交互式评估协议与两项新指标（TKQR、ORA）

**💡 创新点**

① 针对需求澄清设计交互式基准；② 对模糊类型进行细粒度标注并提供关键澄清问题；③ 引入两种衡量交互质量的指标，分别关注问题提早度与交互轮数的合理性

**🔧 技术方法**

利用删除式编辑生成模糊需求，构建多轮交互协议，使用LLM-as-a-judge匹配澄清问题；评估时采用多种顶级LLM（GPT‑4o、GPT‑5、Gemini‑2.5‑Flash、Claude‑Sonnet‑4.5、DeepSeek‑V3.2、Qwen3‑235B‑A22B）

**📊 数据集**

基于LiveCodeBench v6任务构造419个模糊实例（每个含1–3个模糊点），标注关键澄清问题、答案及模糊类型

**📈 对比分析**

采用pass@1、TKQR、ORA三项指标对六款LLM进行对比，结果显示澄清质量低（TKQR≤0.30，ORA≤0.50），pass@1在模糊任务上比完整需求下降约10–20%；思考模型提升代码正确性但澄清性能无显著提升

**⚠️ 局限性**

评估不包含代理工作流；LLM‑as‑judge可能产生误判；存在数据泄露风险；模型往往过早终止交互，澄清深度不足，导致大部分模糊点未被解决

---

## 335. Imprint: Online Memory Compression for Long-Horizon Egocentric QA

**arXiv ID:** 2607.00696 | [PDF](https://arxiv.org/pdf/2607.00696v1)

**作者:** Kousik Das `[一作]` (Indian Institute of Technology Kharagpur), Debaditya Roy `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为 Imprint 的交互中心在线记忆压缩框架，用于长时段视角问答；

**💡 创新点**

创新点在于通过事件原型分配、频率/时效/独特性三维重要性评分实现在线压缩，既保留关键交互，又聚合重复模式，显著提升检索与推理效率；

**🔧 技术方法**

采用结构化交互记录（人物、动作、对象、时间）+ LLM 提取 + 事件原型库 + 重要性评分 + 交互合并 + 基于相似度的检索与答案生成；

**📊 数据集**

使用 EgoLifeQA（7 天连贯的可穿戴摄像数据）进行评测；

**📈 对比分析**

与同一 LLM (Qwen2.5-7B-Instruct) 的 EgoRAG 基线对比，QA 准确率提升至 35.8%（相比 31.0%），基于检索的准确率（GA）提升至 64.8%（相比 41.5%），内存占用减少 2.3×，检索延迟缩短 11.8×；

**⚠️ 局限性**

局限性包括：高度依赖字幕/交互识别质量，重要性评分仅基于启发式公式，缺乏可学习的自适应机制，且对非结构化或复杂多模态信息的支持有限。

---

## 336. Generative Refinement for Low-Budget Black-Box Optimization

**arXiv ID:** 2607.00691 | [PDF](https://arxiv.org/pdf/2607.00691v1)

**作者:** Edouard R. Dufour `[一作]` (EPFL), Pascal Fua `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种名为SARA的黑盒优化算法，利用预训练的无条件生成器作为结构化采样器，并通过归档中的排名信息驱动变异来进行低预算优化。

**💡 创新点**

创新点在于将生成模型的结构先验与目标反馈完全解耦；只需要生成器的腐蚀与精炼过程，无需在搜索过程中重新训练模型；采用排名驱动的方向性变异，提升了对噪声、失败反馈的鲁棒性，并给出了对采样器支持的渐近收敛性。

**🔧 技术方法**

主要技术包括：生成模型的噪声精炼（Diffusion / Flow Matching），排名选择与归档维护，基于归档随机对的方向性变异，以及将噪声水平与排名相关联的自适应腐蚀策略。

**📊 数据集**

实验使用了三类数据集：合成薄管高维目标、Design‑Bench 的 HopperController 控制任务，以及由 UIUC airfoil 数据库和 NACA 轮廓训练得到的 DiffAirfoil 生成模型进行的气动优化。

**📈 对比分析**

与 CMA‑ES、GP、TuRBO 以及随机扩散采样 RDS 进行对比；在薄管任务中，SARA 在低预算下始终优于所有基线；在 5126 维 HopperController 上亦表现突出；在气动优化任务中获得最高的 Cₗ/C_d 并显著降低方差，整体性能远超基线。

**⚠️ 局限性**

局限性包括：只能探索生成器支持范围内的解；渐近收敛保证仅适用于无噪声且基于采样器支持的情形；对三个超参数 β、γ、λ 具有一定敏感性，需经验调优；在奖励简单、反馈可靠或预算充足的场景下可能不如传统方法。

---

## 337. Towards Robust Driving Perception: A Flexible Scale-Driven Family for Self-Supervised Monocular Depth Estimation

**arXiv ID:** 2607.00736 | [PDF](https://arxiv.org/pdf/2607.00736v1)

**作者:** Zhaowen Zhu `[一作]` (Hefei University of Technology), Mingxia Zhan `[通讯]` (Hefei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了 FlexDepth，一套面向自动驾驶的自监督单目深度估计框架，提供多尺度轻量化模型并在动态场景下实现鲁棒性。

**💡 创新点**

创新点在于提出 Scale-Driven Decoder（SDD）与两阶段静态-动态解耦训练策略，能够按尺度自适应特征融合并独立评估静态与动态区域的置信度。

**🔧 技术方法**

采用自监督视差重建、动态上采样、HEB/HPB 瓶颈结构、CSP 交叉阶段部分、两阶段训练、动态采样网格与相机位姿网络等技术。

**📊 数据集**

使用 KITTI、Cityscapes 进行训练与评估，并在 Make3D 上做零样本跨域测试。

**📈 对比分析**

通过与多尺度基准和 Depth Anything V2 的公平对比，Flex-X-Large 在低 FLOPs 下实现或超过现有最优性能，在 Make3D 上表现更佳；与大模型相比，参数和运算量显著下降但精度保持领先。

**⚠️ 局限性**

局限性在于针对道路场景设计，对室内非规则纹理或近距离遮挡的性能有限。

---

## 338. Soft Mixture-of-Recursions: Going Deeper with Recursive Vision Transformers

**arXiv ID:** 2607.00774 | [PDF](https://arxiv.org/pdf/2607.00774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 339. Forensic-Oriented Intrusion Detection Using Synthetic Network Traffic Data and Explainable Artificial Intelligence

**arXiv ID:** 2607.00763 | [PDF](https://arxiv.org/pdf/2607.00763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 340. Evaluating Pretrained Music Embeddings for Cross-Performance Jazz Standard Recognition

**arXiv ID:** 2607.00777 | [PDF](https://arxiv.org/pdf/2607.00777v1)

**作者:** Çağrı Eser `[一作]` `[通讯]` (Middle East Technical University), Çağrı Eser (Middle East Technical University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文构建了一个针对爵士标准曲目跨演绎识别的精简数据集，并比较了从头训练的 Harmonic CNN 与预训练音乐表示模型（MERT、MuQ）在窗口级分类、性能级聚合和最近邻检索等多种评测指标上的表现。

**💡 创新点**

创新点在于：①将 Jazz Trio Database 进行规范化、过滤与重采样，得到平衡的 16 曲目 79 条演绎子集；②提出对冻结的预训练嵌入进行轻量级对比学习投影，以减弱演奏者身份对检索的干扰；③在跨演绎爵士标准识别这一极具挑战的任务中，系统性对比了从头训练、线性/非线性探测以及检索三种方法。

**🔧 技术方法**

使用的技术包括：从头训练的 Harmonic CNN、MERT 与 MuQ 预训练模型的冻结嵌入、线性与 MLP 探测、kNN 基于余弦相似度的检索、以及带有交叉熵与监督对比损失的轻量级投影网络。

**📊 数据集**

采用的数据集为自建的 16 曲目、79 条演绎的 Jazz Trio Database 子集，窗口长度 10 秒，5 秒步长，采用 leave‑one‑performance‑out 的交叉验证策略。

**📈 对比分析**

与从头训练的 Harmonic CNN 相比，预训练嵌入在窗口准确率和性能级 Top‑5 准确率上都有显著提升（最高 Top‑5 达 0.469），但 Top‑1 仍低于 0.13；kNN 检索在 5‑近邻下 Top‑5 与 0.359 相当，若加入对比学习投影后 Top‑5 可提升至 0.469，且检索中同演奏者匹配比例显著下降。

**⚠️ 局限性**

主要局限包括：数据集规模有限且长尾分布被削弱；仅使用 10 秒窗口，缺乏全局音乐语境；检索仍受演奏者身份影响；对比学习的超参数选择有限；以及对全音符或和声信息的利用不足。

---

## 341. From Prediction Uncertainty to Conformalized Distance Fields for Safe Motion Planning

**arXiv ID:** 2607.00776 | [PDF](https://arxiv.org/pdf/2607.00776v1)

**作者:** Jaeuk Shin `[一作]` (Seoul National University), Insoon Yang `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种功能性合规预测（Functional Conformal Prediction, FCP）框架，将预测的距离场误差整体合规化，得到空间连续的安全下界，并将其嵌入采样式MPC中实现实时安全规划。

**💡 创新点**

创新点在于：①一次性合规化整个预测距离场而非逐点标量；②利用距离场的低秩、时间不变性通过FPCA得到低维基底并用GMM+ICP构造封闭式安全包络；③设计了轻量级的在线自适应更新（AFCP）以应对分布漂移。

**🔧 技术方法**

使用技术包括功能性合规预测、功能主成分分析（FPCA）、高斯混合模型（GMM）与诱导合规预测（ICP）、自适应合规预测（ACP）、模型预测控制（MPC）与MPPI采样、以及距离场与轨迹安全约束评估。

**📊 数据集**

实验数据集：ETH–UCY行人数据集（二维场景）和自建的三维四旋翼仿真环境，后者包含多达280个动态障碍物。

**📈 对比分析**

与ACP‑MPC、CC‑MPC、ECP‑MPC等基线对比，FCP‑MPC在高密度环境下实现了更低的碰撞率、更高的到达率，同时每步计算时间显著低于在线不确定性推理方法，展示了优越的安全-效率平衡。

**⚠️ 局限性**

局限性包括：硬约束变体在极端密集场景下仍可能出现不可行率；依赖预先固定的功能基底和离线校准，若场景或预测模型剧烈变化需重新校准；最坏情况下的安全下界可能导致保守过度，影响规划效率。

---

## 342. SNR-Adaptive Optimal Threshold Design for Energy Detection in Dynamic Spectrum Access

**arXiv ID:** 2607.00754 | [PDF](https://arxiv.org/pdf/2607.00754v1)

**作者:** Sushila Dhaka `[一作]` (National Yang Ming Chiao Tung University), Li-Chun Wang `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于最小化贝叶斯误差概率的SNR自适应能量检测阈值设计框架，给出闭式二次方程解；

**💡 创新点**

创新点在于将误差最小化转化为可解析的二次问题，阈值显式与SNR、样本数相关，避免数值搜索；

**🔧 技术方法**

采用能量检测、正态近似、Q函数求导、闭式二次求根；

**📊 数据集**

未使用公开数据集，而是基于仿真模型（AWGN、可变SNR、样本数K）进行验证；

**📈 对比分析**

与固定阈值（CFAR）和检测约束阈值对比，结果显示在低SNR下误差概率显著降低，且计算复杂度极低；

**⚠️ 局限性**

局限在单用户场景，未考虑协作感知与恶意用户影响，且对非高斯噪声或快速衰落环境的适应性待验证。

---

## 343. Active Learning for Cascaded Object Detection: Balancing Coverage and Uncertainty in Table Extraction Pipelines

**arXiv ID:** 2607.00747 | [PDF](https://arxiv.org/pdf/2607.00747v1)

**作者:** Eliott Thomas `[一作]` (La Rochelle Université), Jean-Marc Ogier `[通讯]` (La Rochelle Université)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了适用于级联表格检测管道的主动学习方法，针对检测和结构识别两阶段的互赖关系，设计了两种管道感知采样策略；

**💡 创新点**

创新点包括：①将 Uncertainty Herding 迁移到级联检测任务；②提出 RankFusion 在文档级与表格级两种嵌入空间上实现双层覆盖；③引入 CAPA 在每个阶段分别校准不确定性，并通过瓶颈加权动态分配采样重点；

**🔧 技术方法**

主要技术为：混合覆盖-不确定性采样（Uncertainty Herding）、双层嵌入覆盖、递归排名融合（RRF）、基于 Spearman 相关的校准、阶段依赖门控与瓶颈加权；模型采用 D‑FINE 检测器、DiT 文档嵌入器；

**📊 数据集**

实验使用四个表格提取数据集：公开的 PubTables‑1M 与 FinTabNet，以及两份私有业务文档集（invoice‑int 与 business‑int），单表且无跨行单元格；

**📈 对比分析**

与随机、纯不确定性、纯覆盖、适配 UHerding 等基线比较，使用 GriTS‑Con F1 评估，综合 AUC 和 Schulze 排名显示 RankFusion 与 CAPA 均在 4 个数据集上表现优于基线，CAPA 在多数据集上最为稳健；

**⚠️ 局限性**

局限性包括：仅评估单一 DETR‑based 检测器，未涵盖 YOLO 等模型；仅处理单表无跨行的页面；不针对多表或跨行单元格的复杂布局；校准与瓶颈权重对早期数据噪声敏感；实验预算范围有限（71–500 文档），高数据规模尚未验证；

---

## 344. LLVM-Bench: Benchmarking and Advancing Large Language Models for LLVM Compiler Issue Resolution

**arXiv ID:** 2607.00700 | [PDF](https://arxiv.org/pdf/2607.00700v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 345. Prototype Memory-Guided Training-Free Anomaly Classification and Localization in Prenatal Ultrasound

**arXiv ID:** 2607.00744 | [PDF](https://arxiv.org/pdf/2607.00744v1)

**作者:** Huanwen Liang `[一作]` (Shenzhen University), Dong Ni `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种训练‑free 的产前超声异常分类与定位框架，利用少量参考图像即可完成多类异常的识别与定位。

**💡 创新点**

创新点在于：①首次实现仅用少量参考图像就能完成多类异常的分类与定位；②构建多粒度原型记忆库（全局类原型与细粒度异常原型）并通过原型驱动的软融合聚合异常特征；③引入类别感知细化策略进一步提升类别判别能力。

**🔧 技术方法**

使用 DINOv3 视觉基础编码器、JAFAR 高分辨率上采样、余弦相似度匹配、原型软融合（ASM）以及类别一致性细化（CR）等技术。

**📊 数据集**

采用多中心产前超声图像数据集（1,149 病例、2,357 张图像），涵盖脑、心、腹部 9 类异常（CPC、VM、HPE、正常脑、单心室、DA、MCDK、AWD、正常腹部）。

**📈 对比分析**

与多种 FSOD 方法（DeFRCN、DiGeo、SMILe‑FSOD、TRR‑CCM）以及训练‑free 方法（ProtoSAM、MAUP）进行对比，mAP@0.5 在 32-shot 设置下最高，平均提升约 11.4%，在不同类别和实验设置中表现稳健且优于对比方法。

**⚠️ 局限性**

局限性包括：仅在 9 类异常上验证，未覆盖所有临床常见异常；依赖 DINOv3 在超声上的迁移效果，可能受限于图像分辨率和预处理；缺乏大规模临床外部验证与用户体验评估。

---

## 346. Stochastic Connectivity as the Foundation of a Runtime Model for Microservice Availability Analysis

**arXiv ID:** 2607.00740 | [PDF](https://arxiv.org/pdf/2607.00740v1)

**作者:** Anatoly A. Krasnovsky `[一作]` (Innopolis University), Anna Maslovskaya `[通讯]` (Innopolis University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于随机连通性的微服务端点可用性正式运行时模型，利用分布式追踪与部署元数据构造模型并通过蒙特卡洛估算端点可用性。

**💡 创新点**

① 将服务复制和逻辑边状态统一纳入概率空间，揭示复制无法弥补通信瓶颈；② 明确请求级成功谓词，区分同步/异步依赖；③ 通过 trace‑to‑model 构造实现可复现的完整语义；④ 在合成与实测数据上验证模型并识别其边界。

**🔧 技术方法**

形式化模型（G,R,P,Φ）、概率测度、蒙特卡洛模拟、合成充分性分析、分布式追踪解析（OpenTelemetry）、部署元数据映射、图可靠性公式、统计验证。

**📊 数据集**

DeathStarBench 社交网络工作负载、OpenTelemetry Demo，以及多类合成图形族（链、分支、复制瓶颈等）。

**📈 对比分析**

与闭式期望及现场故障注入实验对比，蒙特卡洛误差≤0.0014，相关系数≈0.992；在合成边界实验中验证单调性、边缘瓶颈等属性；每秒可完成数十万样本的蒙特卡洛估计。

**⚠️ 局限性**

仅支持独立故障的基准概率模型，缺乏相关性与时延（超时、重试、队列）等时间语义；依赖完整追踪，缺失或不完整的追踪会导致偏差；请求谓词需手工定义，易出现语义错误。

---

## 347. GKDT: General Keypoint Detection Transformer

**arXiv ID:** 2607.00752 | [PDF](https://arxiv.org/pdf/2607.00752v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 348. Phantom References: Hallucinated Citations That Survive Peer Review at Top-Tier Conferences

**arXiv ID:** 2607.00738 | [PDF](https://arxiv.org/pdf/2607.00738v1)

**作者:** Mark Russinovich `[一作]` (Microsoft), Ahmed Salem `[通讯]` (Microsoft)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对顶级会议论文，构建了一个自动化引用验证管道，用来检测和量化论文中的引用幻觉（即不存在的作品或作者列表显著不匹配的引用）。

**💡 创新点**

创新点在于：①提出了基于身份匹配的严格幻觉引用定义，排除了常见的书目漂移（如会议名称、年份、出版状态变化等）；②通过大规模自动化审核实现了对数千篇顶尖论文的系统评估；③首次将引用幻觉问题与模型生成文本的风险联系起来，揭示了学术记录中潜在的“幻觉”污染。

**🔧 技术方法**

使用了自动化引用解析技术（例如对照 Crossref、Semantic Scholar 等公共数据库进行作者与题目匹配），结合字符串相似度和元数据对齐的算法，对引用条目进行身份级别的核对；并结合自然语言处理工具对论文文本进行预处理。

**📊 数据集**

数据集包括 2026 年 7 月前在顶尖 AI/ML 会议（如 NeurIPS、ICML、CVPR 等）上发表的所有论文，覆盖约 5,000 篇文章以及它们的引用列表。

**📈 对比分析**

与人工标注的黄金标准进行比较，计算了引用验证管道的准确率、召回率和 F1 分数；实验结果显示，在大多数会议中，识别出的幻觉引用比例约为 8–12%，且准确率超过 92%。

**⚠️ 局限性**

局限性包括：①仅关注顶尖会议论文，无法直接推广到期刊或非顶尖会议；②引用解析的准确性受限于外部数据库的覆盖率和元数据完整性；③对“细微”作者名变体（如中间名、缩写）仍有漏检风险；④未对被幻觉引用的文章内容进行进一步分析，只统计了是否存在幻觉。

---

## 349. LUMA: Benchmarking Segmentation via a Lightweight Universal Mask Adapter

**arXiv ID:** 2607.00687 | [PDF](https://arxiv.org/pdf/2607.00687v1)

**作者:** Tobias Christian Nauen `[一作]` (RPTU University Kaiserslautern-Landau), Andreas Dengel `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级、骨干无关的掩码变压器头（MaskHead），并使用该头在统一训练配方下，对20种视觉变压器骨干、11种预训练方案和多种分辨率进行大规模公平基准实验，以探究骨干结构、预训练目标和模型规模对语义分割性能的影响。

**💡 创新点**

①构造仅通过交叉注意力读取骨干特征且重用骨干MLP的头部，使任何token mixer都可作为黑盒特征提取器；②在固定头部下实现跨骨干公平比较；③揭示token mixer设计对密集预测影响有限，预训练目标和模型规模才是主导因素。

**🔧 技术方法**

基于掩码变压器框架；单侧交叉注意力读取；重用骨干MLP；单尺度特征映射；利用FlashAttention等高效实现；多种预训练（DINOv2、MAE、CLIP、EVA‑02、DINOv3等）。

**📊 数据集**

ADE20K、Cityscapes（语义分割）以及ImageNet‑21k用于预训练。

**📈 对比分析**

在统一的训练recipe下固定MaskHead，对20种骨干、11种预训练、不同分辨率进行实验；结果显示 Plain ViT 在吞吐量上占优；预训练目标（DINO、EVA‑02）显著提升 mIoU；模型规模是计算最优的关键。示例性能：Cityscapes mIoU≈84.4%，ADE20K mIoU≈58.7%（ViT‑L/DINOv2），与EoMT相当但更轻量。

**⚠️ 局限性**

①仅使用单尺度特征，未覆盖多尺度解码器；②固定头部的公平性不一定代表实际部署的总体效率；③预训练数据和方法仅覆盖部分骨干，未能全面评估所有结构；④对极高分辨率或特殊场景的泛化能力尚未充分验证。

---

## 350. Detecting the Undetectable: Enhancing Unsupervised time series Anomaly Detection via Active Learning

**arXiv ID:** 2607.00720 | [PDF](https://arxiv.org/pdf/2607.00720v1)

**作者:** Seung Hun Han `[一作]` (LG CNS), Pilsung Kang `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于主动学习的框架，利用主动查询和监督反馈改进无监督时间序列异常检测模型的性能；

**💡 创新点**

核心创新在于（1）对选中的查询样本进行随机掩码重构预训练，以增强模型对时间依赖的学习；（2）采用极大极小（minimax）损失，在正负样本上分别最小化和最大化重构误差，显著提升对噪声正常样本与细微异常的区分能力；

**🔧 技术方法**

结合掩码重构预训练、minimax 损失、top‑k 与区间随机混合的查询采样策略，以及 Transformer 等深度重构模型；

**📊 数据集**

在四个工业多变量时间序列基准数据集上进行评估：SWaT、PSM、Gecco、Swan；

**📈 对比分析**

与七种无监督重构基底模型（LSTM‑VAE、USAD、OmniAnomaly、Transformer、Anomaly Transformer、VTT‑SAT、VTT‑PAT）以及 Active‑MTSAD 进行对比，实验显示平均 AUC 提升 12.39%，在 Transformer 基础上平均提升 7.56个百分点，超过大多数基线；

**⚠️ 局限性**

主要局限包括：minimax 目标在某些数据集（如 USAD、OmniAnomaly）与模型训练目标冲突时可能导致性能下降；在异常比例高的 PSM 数据集上主动学习策略假设失效，导致提升有限；对超参数和查询预算较为敏感，需进一步提升鲁棒性。

---

## 351. Decoupled Guidance: Disentangling Subject and Context Pathways in Text-to-Image Personalization

**arXiv ID:** 2607.00766 | [PDF](https://arxiv.org/pdf/2607.00766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 352. From Consistency to Collaborative Discovery: MFEA-CoD for Multitask Novelty Search

**arXiv ID:** 2607.00761 | [PDF](https://arxiv.org/pdf/2607.00761v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 353. FrameONE: Hierarchical Motion Modeling for Universal Multi-View Echocardiographic Keyframe Detection

**arXiv ID:** 2607.00748 | [PDF](https://arxiv.org/pdf/2607.00748v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 354. SessionBound: Turning Enterprise Task Approval into Budgeted Database Sessions

**arXiv ID:** 2607.00751 | [PDF](https://arxiv.org/pdf/2607.00751v1)

**作者:** Minmin Wu `[一作]` `[通讯]` (China Telecom Global Limited), Minmin Wu (China Telecom Global Limited)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 SessionBound，一种将企业任务批准转化为受限数据库会话的控制平面和运行时体系结构。

**💡 创新点**

将业务审批作为数据库访问控制的入口，利用签名任务令牌、预算与收据实现可审计、可签名、受限的数据库会话，而非依赖 LLM 或手写 SQL 规则。

**🔧 技术方法**

使用 PostgreSQL 作为参考运行时，结合安全视图、行级安全、SQL 检查、签名令牌、预算计数和收据记录；控制平面提供任务模板、审批、签名与权限绑定。

**📊 数据集**

主要使用自制的 PostgreSQL 示例数据库，包含费用、员工、部门等表，采用 24 场景验证套件进行功能测试；未使用公开大规模数据集。

**📈 对比分析**

通过与原始 PostgreSQL 对比，测量五种查询模式（SELECT、JOIN、GROUP BY、CTE、WINDOW）在 100 次测量中的 p50 延迟；SessionBound 的 p50 约 1.4–1.5 ms，原始 0.05–0.07 ms，导致相对 2000–2700% 的开销，但绝对延迟仍在毫秒级。

**⚠️ 局限性**

仅支持单实例 PostgreSQL；SQL 检查保守且非 AST 级；未实现差分隐私或完整语义推理；预算模型有限；视图漂移失效需手动处理；性能评估仅为小规模合成查询。

---

## 355. MSQA: A Natively Sourced Multilingual and Multicultural SimpleQA Benchmark

**arXiv ID:** 2607.00724 | [PDF](https://arxiv.org/pdf/2607.00724v1)

**作者:** Xianru Chen `[一作]` (ByteDance Seed), Jiaheng Liu `[通讯]` (Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了多语言多文化问答基准 MSQA，专门评估 LLM 在不同语言中的文化知识水平，揭示了“文化对齐幻觉”。

**💡 创新点**

创新点在于①将单一文化知识与语言表达分离，②设计了五个文化维度和三层难度等级的严格题库，③系统分析了模型在自信度、随机成功率和检索利用方面维持文化幻觉的三大机制。

**🔧 技术方法**

采用多阶段人工构建、RAG 验证、LLM 评判、经验难度筛选等技术，并使用 Gemini‑3.1‑Pro 等前沿 LLM 进行评测；同时利用 ECE、Best‑of‑N、RAG 效果等指标进行诊断。

**📊 数据集**

MSQA 数据集共 1,064 条原生题目，覆盖 11 种语言（英、中、葡、泰、俄、韩、法、日、马、印、西）和五个文化维度；同时对比 SimpleQA、Chinese SimpleQA 等传统基准。

**📈 对比分析**

通过对 18 款 LLM 的评测，发现模型在语言多样性高的子集（如葡语、俄语）表现良好，但在低覆盖或文化密集语种（如中文、泰语）显著下降；排名与传统翻译基准差异显著，说明单纯的多语流利度无法反映文化能力，性能总体低于英语中心基准，最大 F 分数仅 68.7%。

**⚠️ 局限性**

局限包括：仅覆盖 11 语种，规模相对较小；难度划分依赖人工判断，可能存在主观性；评测聚焦闭合式事实问题，未覆盖开放式解释或敏感话题；RAG 评估受限于单一检索管线；对文化幻觉机制的探讨未必完整，可能忽视 RLHF、标记化等因素。

---

## 356. Towards Memory-Efficient Autoregressive Video Generation via Instance-Specific Parametric Absorption

**arXiv ID:** 2607.00712 | [PDF](https://arxiv.org/pdf/2607.00712v1)

**作者:** Xiaomeng Fu `[一作]` (Chinese Academy of Sciences), Jizhong Han `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在自回归视频生成模型中，将一部分KV缓存的历史上下文“吸收”到模型权重中，动态切换全注意力层为局部注意力层，实现内存压缩。

**💡 创新点**

创新点在于将KV缓存压缩视为参数化记忆内聚，而非单纯丢弃，采用实例特定闭式最小二乘解对权重进行测试时适配，并使用可分解注意力避免额外开销。

**🔧 技术方法**

使用闭式最小二乘权重调节、可分解注意力（基于Log‑Sum‑Exp）、局部/全局注意力层转换、Post‑Training 8‑bit量化等技术。

**📊 数据集**

在多种AR视频生成模型（LongLive、Reward、Krea、LiveAvatar）上使用VBench（5s、30s）和MovieGen全长提示进行评估。

**📈 对比分析**

与完整KV缓存、Token‑dropping、传统压缩方法比较，ISPA在保留近乎相同视觉质量的同时可压缩约50% KV缓存，提升1.86×推理速度，30s长视频甚至性能更优。

**⚠️ 局限性**

局限包括需要在warmup阶段收集双流注意力，适用于单一实例且对动态场景切换有一定触发阈值，过多转换层会导致质量下降。

---

## 357. LLM-Guided ODE Discovery and Parameter Inference from Small-Cohort Aggregate Data

**arXiv ID:** 2607.00733 | [PDF](https://arxiv.org/pdf/2607.00733v1)

**作者:** Hanning Yang `[一作]` (University of Freiburg), Moritz Hess `[通讯]` (University of Freiburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 AgentODE 框架，能够仅基于群体层面的总结统计量，联合发现 ODE 结构并推断参数分布，专为稀有病等数据稀缺、隐私受限场景设计。

**💡 创新点**

创新点在于：①利用大型语言模型（LLM）进行符号 ODE 结构生成；②引入诊断–更新循环的工具增强推理代理，用视觉总结统计指导参数分布的迭代精炼；③实现结构与参数同时自洽的无个人轨迹数据推断。

**🔧 技术方法**

核心技术包括：LLM（GPT‑5.2） + ReAct/Reflexion 代理；模拟推理（synthetic likelihood、logSL、MNSD）；可视化总结统计（均值轨迹、置信区间、相关性热图）；经验缓冲区动态记忆；分布式参数（LogNormal）。

**📊 数据集**

实验数据集：三类合成基准（Apoptosis、Polymer、PKPD‑Immune）和两类临床数据（MIMIC‑IV 的 AKI 组和 46 病例、231 观测值的 RDEB 组）。

**📈 对比分析**

与 SINDy、PySR、LLM‑SR、Neural ODE 等方法对比；在合成基准上 AgentODE 与全轨迹方法相当，在临床数据上 AgentODE 超越 Neural ODE、与 LLM‑SR 性能相近，但更能恢复机制合理的结构；整体 RMSE 较低，且诊断‑更新循环显著提升性能。

**⚠️ 局限性**

局限性：受 LLM 预置知识限制，复杂 ODE 结构识别效率下降；参数辨识不确定性导致非单调收敛；无法完全替代基于完整轨迹的梯度优化；结果仍需领域专家验证；依赖高质量的总结统计，若总结失真则影响推断。

---

## 358. AV-SyncBench: Decoupled Benchmarking of Temporal and Semantic Audio-Visual Synchronization

**arXiv ID:** 2607.00726 | [PDF](https://arxiv.org/pdf/2607.00726v1)

**作者:** Tianhong Zhou `[一作]` (Tsinghua University), Bo Zheng `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 AV-SyncBench 基准，用以独立评估音视频同步的时间一致性和语义一致性。

**💡 创新点**

创新点在于构建可分离的时间与语义挑战集，采用 in‑the‑wild 视频并通过自动与人工筛选保证音源可见；利用生成编辑技术在保持时间结构的前提下实现语义变化。

**🔧 技术方法**

技术手段包括：全局偏移、局部抖动、全局速度变化等时间扰动；OpenVoice 与 DDSP 进行语义编辑；使用 0.64 秒非重叠分块提取特征，计算对角余弦相似度并以二元准确率评估；对五个主流模型无训练直接评测。

**📊 数据集**

数据集：3269 段 in‑the‑wild 视频，38,390 个样本，覆盖 Voice、Music、Sound 三大领域，10 种场景标签。

**📈 对比分析**

比较方法：统一分块提取特征，计算对角余弦相似度，比较原始对与扰动/编辑对的同步得分；实验显示模型在时间对齐和语义辨别上表现不一，SparseSync 在时间任务上优势明显，ImageBind 在语义任务上表现突出，整体准确率在 0.5–0.8 之间。

**⚠️ 局限性**

局限性：语义编辑依赖生成模型，可能引入非语义差异；目前仅覆盖语音和音乐场景，缺少环境声等编辑；样本时长短，未涵盖长时序或多源交互；不同编辑管道的可比性有限。

---

## 359. The Rise and Fall of Google's Privacy Sandbox

**arXiv ID:** 2607.00693 | [PDF](https://arxiv.org/pdf/2607.00693v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 360. Self-conditioned Flow Map Language Models via Fixed-point Flows

**arXiv ID:** 2607.00714 | [PDF](https://arxiv.org/pdf/2607.00714v1)

**作者:** Jaehoon Yoo `[一作]` (KAIST), Jinwoo Kim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了一种自条件化流语言模型的固定点流（fixed‑point flow）框架，并通过固定点蒸馏将其转化为可在少步或一步内生成文本的流映射语言模型 FMLM★。

**💡 创新点**

创新点包括：①将自条件化视为自我修正的固定点迭代；②提出二维固定点流理论，证明其能生成合法的流映射；③设计两种蒸馏策略（离线和在线）实现自条件化蒸馏到无自条件化流映射，并在少步生成上实现SOTA；④首次在文本生成上统一比较自条件化与少步模型。

**🔧 技术方法**

使用的主要技术：自条件化流模型、固定点迭代与收敛理论、固定点蒸馏（包括固定点蒸馏损失与两时态去噪器蒸馏）、Euler 与固定点求解器、对齐损失与半群一致性损失。

**📊 数据集**

在 OpenWebText 数据集上进行实验，序列长度为 1024，使用 GPT‑2 Large 进行生成困惑度评估。

**📈 对比分析**

与多种基线（Duo, MDLM, Di4C, FMLM, DFM 等）在 gPPL‑entropy 前沿进行比较。FMLM★ 在保持 5.44 nats 数据熵的前提下，在一步和少步生成场景中均取得 gPPL 最高、最优性能，超过所有基线且接近 32 步自条件化教师的性能。

**⚠️ 局限性**

局限性：蒸馏过程需要先验教师模型；目前仅在文本数据上验证，未探讨图像/视频等其他模态；未实现完全自蒸馏（仅用数据监督一次性训练所有模块）。

---

## 361. Mobile Base Station Positioning in Smart Ports Based on Kriged Sparse Measurements and Obstacle Inference

**arXiv ID:** 2607.00709 | [PDF](https://arxiv.org/pdf/2607.00709v1)

**作者:** Paulo Furtado Correia `[一作]` (INESC TEC, Faculdade de Engenharia da Universidade do Porto), Manuel Ricardo `[通讯]` (INESC TEC, Faculdade de Engenharia da Universidade do Porto)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 DOCKING 框架，利用稀疏射频测量通过普通克里金插值重建无线环境图（REM），从中提取遮蔽结构，进而在港口环境中实现移动集成接入与回传（MIAB）站点的基于容量的最优部署与关联决策。

**💡 创新点**

创新点在于：①无需先验障碍物数据库，直接从克里金重建的 REM 里抽取三维立方体障碍抽象；②将 REM 构建、障碍抽象与 MIAB 部署优化统一为单一管线；③采用快速遗传算法实现每个时间快照的 MIAB 位置与回传路径优化，显著提升容量。

**🔧 技术方法**

技术包括：普通克里金（OKG）插值、半变异函数拟合、H-极大值变换与形态学膨胀、连通组件提取、旋转包围盒拟合、基于 3GPP UMi 模型的信道建模、遗传算法（GA）求解。

**📊 数据集**

使用的测量数据集：在模拟的港口场景中，基于 25 m 网格、10 个容器堆叠、4 个 FIAB 的全域“真值”数据；稀疏测量通过 1–15 % 的采样率（高斯、指数或均匀分布）生成；同时在 150 m×130 m 真实港口片段进行 90 点稀疏测量，验证链路。

**📈 对比分析**

与传统基于先验障碍图或全覆盖测量的方法比较，DOCKING 在 15 % 稀疏采样下 REM 预测误差 ≤ 3 dB（90 % 分位），障碍检测真阳性率 ≥ 85%，MIAB 部署后容量提升可达 150%，并且快速 GA 在 5–15 s 内即可收敛。

**⚠️ 局限性**

局限性包括：高度假设为标准集装箱尺寸，无法精确捕捉非标准障碍；克里金插值对极端信道异质性（如动态移动障碍）敏感；当前仅考虑单 MIAB，扩展至多 MIAB 及更大规模港口仍需验证。

---

## 362. Know Thy Neighbor: Cross-TEE Mutual Attestation

**arXiv ID:** 2607.00695 | [PDF](https://arxiv.org/pdf/2607.00695v1)

**作者:** Daniel Andrade `[一作]` (Universidade de Lisboa), Miguel P. Correia `[通讯]` (Universidade de Lisboa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了 Heterogeneous Mutual Attestation (Hema) 协议，支持在不同 TEE 类型之间实现可信实例的互相认证与安全通信。

**💡 创新点**

创新点在于将远程验证（RA）机制统一为单一软件栈，既可用于同一 TEE 类型，也可用于跨 TEE 类型；同时通过形式化验证证明了协议的安全性。

**🔧 技术方法**

使用 Tamarin 形式化验证工具、基于 RA 原语的轻量化远程验证、哈希与 ECDH 密钥协商等技术。

**📊 数据集**

未使用公开数据集，主要通过协议模型与安全属性进行形式化验证。

**📈 对比分析**

通过 Tamarin 对安全属性（密钥机密性、协议一致性等）进行验证，未给出实验性能数值，但证明了协议在理论上的安全性。

**⚠️ 局限性**

局限性包括对可信验证者列表管理的依赖、未覆盖侧信道与回滚等攻击，以及实现需在多种 TEE 之间共享验证者列表。

---

## 363. M2Note: Continual Evolution of Vision Language Models via Mistake Notebook Learning

**arXiv ID:** 2607.00685 | [PDF](https://arxiv.org/pdf/2607.00685v1)

**作者:** Haiwen Li `[一作]` (AMAP, Alibaba Group), Xiangxiang Chu `[通讯]` (AMAP, Alibaba Group)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练‑free 的多模态错误笔记学习框架（M²Note），通过外部记忆记录并检索失败案例中的可复用提示，来提升 VLM 的推理可靠性。

**💡 创新点**

创新点在于：①将模型错误抽象为“主题‑指导”对，并存入可编辑笔记；②使用多模态检索增强生成（RAG）在推理时动态注入提示；③采用批量级“仅接受改进”验证机制，保证笔记演进的稳定性；④支持同模型自演进与跨模型演进，实现无参数更新的能力迁移。

**🔧 技术方法**

核心技术包括：多模态检索（使用 Qwen3‑VL‑Embedding 等多模态嵌入模型），RAG 生成，提示生成的 Prompt 设计，批量级后验证与回滚，主题抽取与合并逻辑。

**📊 数据集**

在六个多模态推理基准上评估：MMMU、MathVista（STEM/数学）、MMStar、RealworldQA（通用 VQA）、AI2D、ChartQA（文档/图表）。

**📈 对比分析**

与 SFT、RL 等传统自演进方法相比，M²Note 在所有基准上实现了 1–5% 的准确率提升，且训练成本仅为 API 调用费用，样本效率更高；与 Chain‑of‑Thought 组合还能进一步提升 0.5–2% 的性能。

**⚠️ 局限性**

局限性：对“结构化错误”高度依赖，难以在视觉多样性极高、长尾任务中检索到合适笔记；错误笔记过多或不当可能引入误导信息，导致推理偏差；目前笔记仅为文本，缺乏图像或结构化工具支持。

---

## 364. When to Repair a Graph ANN Index: Navigability-Signal-Triggered Local Repair Protects Tail Recall Under Bursty Churn

**arXiv ID:** 2607.00728 | [PDF](https://arxiv.org/pdf/2607.00728v1)

**作者:** Madhulatha Mandarapu `[一作]` (VaidhyaMegha Private Limited), Sandeep Kunkunuru `[通讯]` (VaidhyaMegha Private Limited)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并评估了一种基于导航性指标触发的局部图修复控制器，用于在插入/删除 churn 下保持 ANN 图索引的召回率。

**💡 创新点**

通过使用可观测的 probe recall 信号动态触发修复，实现相同修复预算下尾部召回率的显著提升，并提供预算匹配评估协议。

**🔧 技术方法**

基于 Vamana/DiskANN 的近邻图索引、probe recall 作为导航性信号、局部边修复、预算匹配实验协议及 Spearman 相关性分析。

**📊 数据集**

SIFT-128 与 Fashion-MNIST-784 两个 L2 ANN 数据集，live set 大小约 20k/10k，支持 bursty churn 流。

**📈 对比分析**

对比三种策略（无修复、固定周期修复、信号触发修复），在相同修复次数下测量均值和最小召回，发现信号触发在稀缺预算时尾部召回提升 0.014–0.050，平均召回提升 <0.005。

**⚠️ 局限性**

仅在尾部召回有显著改进，平均召回提升不足 0.005；效果受索引鲁棒性和预算稀缺度限制；实验仅在单机内存 Vamana、L2 数据、10–20k 向量规模下进行。

---

## 365. Which Voting Rules Are More Resilient to Coalitional Manipulation?

**arXiv ID:** 2607.00758 | [PDF](https://arxiv.org/pdf/2607.00758v1)

**作者:** François Durand `[一作]` `[通讯]` (Nokia Bell Labs), François Durand (Nokia Bell Labs)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了多种标准的序数投票规则在偏好分布的一个简单参数化文化模型下，对抗性投票的可操纵性（coalitional manipulability）是否存在临界阈值，并通过理论与实证分析揭示投票规则的相对脆弱性排序。

**💡 创新点**

提出了更强的Condorcet概念（Pair‑Safe、Set‑Safe、Resistant Condorcet Winner等），利用它们统一计算各规则在参数化文化中的临界浓度阈值，并证明几乎所有标准序数规则都呈现相同的相位转移行为；进一步将这些阈值映射为对不同规则可操纵性顺序的预测，未进行任何模型拟合，仍能准确预测不同数据集之间的规则排序。

**🔧 技术方法**

主要使用概率文化模型（Perturbed Culture）与集中度阈值理论，弱大数定律、指数收敛证明等理论工具；对实际投票数据采用计算机操作性可操纵性算法（改进的SVP、Kemeny等）实现。

**📊 数据集**

使用了Netflix和FairVote两大公开偏好数据集，并在代码仓库中补充了PrefLib数据集做进一步验证。

**📈 对比分析**

比较方法：基于规则的临界浓度阈值排序与实测可操纵性（CM）率进行对比，使用算法不确定性与误差条来评估排名一致性；实验显示理论排序与实测排序高度吻合，除Bucklin、Coombs等规则略有偏差，其余规则在不同数据集上表现一致。

**⚠️ 局限性**

局限性：模型仅适用于序数规则，且在极端参数（如θ=1）或特殊规则（Coombs、Bucklin、Veto、Kim‑Roush）下预测与实测存在偏差；对非序数规则和更复杂的偏好生成模型的泛化仍待研究。

---

## 366. Approximate Nearest Neighbor Search with Graph Range Filters

**arXiv ID:** 2607.00727 | [PDF](https://arxiv.org/pdf/2607.00727v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 367. Algorithms and fine-grained complexity for nondeterministic and symmetric difference automata

**arXiv ID:** 2607.00742 | [PDF](https://arxiv.org/pdf/2607.00742v1)

**作者:** Dmitry Chistikov `[一作]` (University of Warwick), Brink van der Merwe `[通讯]` (Stellenbosch University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究对称差有限自动机（XNFA）的细粒度复杂度，给出在多项式歧义性下从NFA接受问题到XNFA接受问题的随机化归约；对有限歧义性自动机提出了比传统O(|δ|·|w|)更快的接受算法；设计了利用矩阵乘法快速验证XNFA空性与等价性的证书方案；并证明基于SETH的下界不太可能适用于XNFA空性/等价性问题。

**💡 创新点**

创新点在于：①首次将多项式歧义NFA接受问题通过随机化归约到XNFA接受问题，表明两者在细粒度时间复杂度上不可分离；②利用歧义性分析改进动态规划，使有限歧义自动机的接受时间从O(|δ|·|w|)降至O(k|w||Q|+|δ|)；③构造k‑XNFA模拟GF(2^k)权重的自动机，并进一步分解为k个普通XNFA，实现随机化判定；④设计了子立方时间的证书验证方法，利用Freivalds算法与矩阵乘法加速空性/等价性判定；⑤证明SETH下界不适用于XNFA空性/等价性，需突破NSETH。

**🔧 技术方法**

技术主要包括：多项式歧义性分析、动态规划改进、Schwartz‑Zippel 多项式非零性检测、GF(2^k)与GF(2)向量空间同构、k‑XNFA的构造与分解、Freivalds随机矩阵乘法验证、矩阵乘法指数ω、以及与SETH、NSETH相关的下界论证。

**📊 数据集**

本文不使用真实数据集，而是通过理论构造的合成实例（如基于图的k‑Clique、三角检测等）和符号计算（多项式、矩阵乘法）进行实验和证明；所有结果均在抽象模型层面给出。

**📈 对比分析**

与传统算法相比：对有限歧义自动机，接受问题从O(|δ|·|w|)提升到O(k|w||Q|+|δ|)，在k为常数时等价于O(|w||Q|+|δ|)；空性与等价性问题的验证时间从O(|Σ|n^3)下降到O(|Σ|n^ω)（ω≈2.38），即子立方；随机化归约提供了若XNFA接受可在O((|δ|·|w|)^{1-ε})时间内解，则多项式歧义NFA接受亦可同速解，从而在细粒度层面上说明两者难度相当。

**⚠️ 局限性**

局限性：①归约为随机化且一侧误差，未给出确定性或多对一归约；②只在多项式歧义性假设下有效，对一般（无歧义限制）的XNFA接受问题仍无已知加速算法；③空性/等价性的下界论证依赖NSETH，若NSETH成立则不适用；④对极端稠密自动机的实际性能分析仍待研究。

---

## 368. 5G Configured Grant Scheduling for 5G-TSN Integration for the Support of Industry 4.0

**arXiv ID:** 2607.00704 | [PDF](https://arxiv.org/pdf/2607.00704v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 369. ConRTF: Edge-Constrained Boundary Distribution Refinement for Realtime TransFormer Table Structure Recognition

**arXiv ID:** 2607.00734 | [PDF](https://arxiv.org/pdf/2607.00734v1)

**作者:** Eliott Thomas `[一作]` (La Rochelle University), Antoine Doucet `[通讯]` (La Rochelle University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种 Edge-Constrained Fine-grained Localization (EFL) 损失，并将其嵌入 D-FINE 架构，构建实时表结构识别模型 ConRTF。

**💡 创新点**

创新点在于将行与列的结构差异编码为边权重，利用 IoU 敏感性分析推导出最佳权重比例；EFL 仅在训练阶段使用，推理时无额外开销；同时在小样本情况下保持高精度。

**🔧 技术方法**

使用技术包括分布式边界回归、DETR/RT-DETR 变体、YOLO 系列、Transformer 解码器、D-FINE 的迭代细化与全局自蒸馏（GO-LSD）、以及自定义的 EFL 损失。

**📊 数据集**

实验数据集为 PubTables-1M（公共基准）以及两套私有数据集：Business（4,362 表）和 Finance（2,769 表）。

**📈 对比分析**

与 D-FINE‑X、RT‑DETRv2、YOLOv10/11/12 等实时检测器对比，ConRTF 在 PubTables‑1M、Business、Finance 上的 TEDS 与 GriTS 指标均提升 0.1–1.6 个百分点，尤其在低样本（约 2k–3k 表）情形下显著提升；推理速度保持不变。

**⚠️ 局限性**

局限性包括：仅针对轴对齐表格；采用单一全局权重，对不同类别或实例的细粒度差异处理不足；未针对极低样本或零样本场景做进一步验证。

---

## 370. What Survives Into Context: A Diagnostic for Budget-Constrained Multi-Hop RAG and When Submodular Evidence Packing Improves It

**arXiv ID:** 2607.00725 | [PDF](https://arxiv.org/pdf/2607.00725v1)

**作者:** Ananto Nayan Bala `[一作]` `[通讯]` (Ahsanullah University of Science and Technology), Ananto Nayan Bala (Ahsanullah University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种检索-增广生成(RAG)下的答案在上下文中的出现诊断，并基于该诊断设计了预算约束下的子模最优证据打包方法；

**💡 创新点**

创新点在于将检索召回替换为“答案是否在被读者看到的上下文中”作为优化目标，并用子模函数联合相关性、查询覆盖、代表性与多样性实现高效打包；

**🔧 技术方法**

主要技术包括答案在上下文诊断、预算约束的单调子模最大化、基于成本的贪心算法以及多项实验验证；

**📊 数据集**

实验使用HotpotQA、MuSiQue、2WikiMultiHopQA、RAGBench、ExpertQA等多数据集进行评估；

**📈 对比分析**

与基线的比较显示，在HotpotQA（预算160 token，3B阅读器）下，子模打包在保持或减少token成本的前提下将F1提升约0.05点，显著优于MMR、聚焦启发式及简单拼接；

**⚠️ 局限性**

局限性包括仅在HotpotQA上显著获胜，未在更大阅读器（>14B）或长文本答案场景下验证；诊断仅适用于短span答案，且对检索器与图结构的依赖尚待进一步探究。

---

## 371. Creating Impactful Autonomous Driving Datasets: A Strategic Guide from Research Gap to Benchmark

**arXiv ID:** 2607.00710 | [PDF](https://arxiv.org/pdf/2607.00710v1)

**作者:** Richard Schwarzkopf `[一作]` (Karlsruhe Institute Of Technology), Christoph Stiller `[通讯]` (Karlsruhe Institute Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套从研究缺口诊断到数据采集的策略框架，并以 KITScenes 数据集系列为案例，说明如何通过最小化数据操作实现高影响力的数据集构建。

**💡 创新点**

创新点在于（1）首次系统化地将数据问题与评估问题区分并引导最小化数据操作的选择；（2）提出基于“诊断-缺口-操作”三步的策略框架；（3）定义了适合高阶评估的指标（如 Multi‑Maneuver Score、关系图构建指标）和数据采集准则。

**🔧 技术方法**

使用的技术包括：多模态传感器（高分辨率摄像头、长距激光雷达、4D 雷达、GNSS/INS）同步与标定、基于自动与人工混合的标注流程、数据操作模型（利用现有数据、合成数据、录制新数据）以及基于地理距离的训练/测试分割。

**📊 数据集**

核心数据集为 KITScenes Multimodal 与 KITScenes LongTail 两个子集；同时对 Argoverse 2 等公开数据集做了对比与扩展。

**📈 对比分析**

通过在多任务基准（在线 HD 地图构建、长距离单目深度、视角合成、端到端驾驶）上进行实验，展示了新指标的区分度与方法在长尾驾驶任务上的显著改进；基准排行榜验证了评估的可靠性。

**⚠️ 局限性**

主要局限包括：录制新数据成本高、依赖昂贵的硬件与专业标注团队；在不同城市、道路布局下的通用性尚未充分验证；对高维标注（如 3D 交通灯与车道关联）的自动化仍未成熟。

---

## 372. Self-GC: Self-Governing Context for Long-Horizon LLM Agents

**arXiv ID:** 2607.00692 | [PDF](https://arxiv.org/pdf/2607.00692v1)

**作者:** Xubin Hao `[一作]` (Xiaohongshu), Chenpeng Cao `[通讯]` (Xiaohongshu)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个可插拔的框架——GC‑Context，用来在长周期LLM代理的运行时通过对象级生命周期管理（fold、mask、prune）控制上下文，减少不必要的令牌并保持未来步骤所需的关键依赖。

**💡 创新点**

创新点在于将代理上下文视为可恢复的运行时对象集合，而非线性文本缓冲，结合侧信道规划器与主机验证，保证对依赖性评估与缓存友好性的双重保障。

**🔧 技术方法**

技术手段包括：索引化用户回合与工具输出为可追踪对象、侧信道规划器基于对象动作契约生成保留/删除策略、主机端的计划回放与安全提交、以及面向缓存的收益评估决策。

**📊 数据集**

实验使用了两套基于生产日志的长周期代理轨迹数据集：包含332个会话的 Production Suite 与 33个高压力 Hard Set，并在上线环境中通过账户级拆分评估。

**📈 对比分析**

与传统时间/类型剪枝、工具输出剪枝及混合剪枝等基线对比，GC‑Context 在 Hard Set 上实现了 43.95% 的压缩率与 84.85% 的无影响率（相较于基线的 54.55%‑69.70%），在 Production Suite 上的无影响率高达 91.27%‑94.58%（基线低于 90%），上线时平均输入令牌下降 10%‑15%。

**⚠️ 局限性**

局限性包括：评估依赖于内部私有日志，缺少公开可复现数据；主要指标为判定器判定的无影响率，未覆盖完整在线成功率或账单成本；对多模态或二进制载荷恢复的测试尚未展开。

---

## 373. No Country for Old Privacy: The Evolving Challenges of Anonymity in Bitcoin

**arXiv ID:** 2607.00772 | [PDF](https://arxiv.org/pdf/2607.00772v1)

**作者:** Ben Hawkins `[一作]` (University of York), Siamak F. Shahandashti `[通讯]` (University of York)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对比 2016-2025 年间比特币链上第二代匿名协议（CoinJoin、CoinSwap、CoinShuffle、Stealth Addresses）的可检测使用量进行纵向量化测量，评估监管、技术升级与市场事件对其采纳率的影响。

**💡 创新点**

①首次将 10 年跨度内的多协议检测结果统一对齐；②通过三轮启发式精细化 Stealth Address 检测，揭示其在主链上几乎无可验证实现；③揭示 CoinShuffle 与 Wasabi Wallet 操作周期高度一致，凸显监管行动对非托管隐私工具的直接冲击。

**🔧 技术方法**

利用基于 SQL 的启发式过滤器（如输入/输出数量、金额差异、脚本长度/模式匹配）在 DuckDB 关系数据库中对本地完整链数据进行查询；结合事件时间线叠加分析；使用 1,008 区块（约 1 周）为单位的移动平均平滑趋势。

**📊 数据集**

完整比特币主链区块（约 780 GB 原始 + 1.3 TB 结构化 CSV）以及 400k–900k 区块范围的交易记录；对 5,948,184 CoinJoin、23,321,415 CoinSwap、CoinShuffle 与 Stealth Address 的检测样本。

**📈 对比分析**

与 Möser & Böhme 2016 年的基准结果进行对照，验证启发式可重复性；对不同协议的检测率与交易量进行归一化比较，展示各协议在整体交易中的占比低于 1%；通过事件叠加说明监管/升级对检测率的显著影响。性能方面：整个测量管线在 8 核 i7-7700 机器上完成时间约 108 小时，数据库查询效率足以支持多次可视化与聚合。

**⚠️ 局限性**

①检测方法仅覆盖可识别的可观测协议，无法捕捉已变得不可检测的隐私手段；②Stealth Address 的三重启发式仍可能存在误报或漏报；③监管事件与协议采纳之间的因果关系仅为相关性而非确证；④因仅使用链上数据，未考虑离链/侧链（如 Lightning）或多签钱包内部逻辑导致的混淆；⑤样本仅限于 2016–2025 年，未来协议演进可能改变结果。

---

## 374. How Ethos and Pathos Appeals Resonate in Reader Interpretations of Social Media Messages

**arXiv ID:** 2607.00873 | [PDF](https://arxiv.org/pdf/2607.00873v1)

**作者:** Ewelina Gajewska `[一作]` (Warsaw University of Technology), Liesbeth Allein `[通讯]` (KU Leuven)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对 Reddit r/ChangeMyView 社区中句子与其人类重述（interpretation）进行对比，探究原句中的两类古典修辞手段——信誉（ethos）与情感（pathos）在沉默受众内部表述中的保留、转化与遗漏情况。

**💡 创新点**

创新点在于：①首次从“沉默受众”视角考察修辞在意义解释过程中的传播和变异；②将修辞标签（ethos, pathos）作为视角层级，量化其在原句与重述之间的匹配率与变异性；③将修辞标签与受众对作者的态度评分相关联，揭示修辞对评价的直接影响；④发现高修辞负荷导致更大解释差异，提出“解释不稳定性”作为新研究维度。

**🔧 技术方法**

使用了：① RoBERTa‑large 与 RoBERTa‑base 作为 fine‑tuned 语义分类器；② Gemini 2.5 LLM 进行零/少/链式思考 prompt；③ 统计回归（OLS）和卡方检验评估标签匹配与变异；④ 手工评测与自动评测相结合的银标准标签可靠性验证。

**📊 数据集**

主要数据集为 OrigamIM（2018 条英文句子，每句约 5 条人类解释，共 9851 条解释），并引用 PolarIs 与 EU DisinfoTest 进行模型验证与迁移学习评估。

**📈 对比分析**

比较方法：在 OrigamIM 上对 ethos 与 pathos 的标签分布与模型预测结果进行对比，计算匹配率（约 74% ethos, 70% pathos）与变异率；通过 OLS 估计修辞标签对解释变异与受众态度的影响。模型性能：RoBERTa 在 PolarIs 上 F1 ≈ 0.91；Gemini 在 OrigamIM 上 F1 ≈ 0.73（ethos）和 0.70（pathos），总体匹配率高于随机。

**⚠️ 局限性**

局限性：① 仅使用单一英文数据集，难以推广至其他语言或平台；② 每句解释仅 5 条，无法覆盖完整受众多样性；③ 依赖银标准分类器，存在中等误差；④ 受众态度评分仅来自解释文本，未结合真实平台互动数据；⑤ 未考虑隐含的身份、政治偏好等细粒度变量，可能影响结果。

---

## 375. Self-Evolving Agents with Anytime-Valid Certificates

**arXiv ID:** 2607.00871 | [PDF](https://arxiv.org/pdf/2607.00871v1)

**作者:** Biswa Sengupta `[一作]` `[通讯]` (JPMorgan Chase & Co.), Biswa Sengupta (JPMorgan Chase & Co.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SEA架构，将自演化代理限制在冻结基础模型、低维导航适配器和可版本化的主机（harness）上，并通过一次性有效门控（anytime‑valid gate）保证每一次自修改都在预设误差预算内；同时设计五种循环控制器（记忆保护、偏好学习、信用分配、信心门控、库扩展）以及五种循环内验证机制（自生成再现oracle、精细搜索等），实现无需人工标注即可在自演化循环中提供密集的、无评审器的信号。

**💡 创新点**

核心创新在于：① 将自演化过程拆解为可验证的子循环，每一次自修改必须通过一次性有效门控；② 结合多种已公开的理论保证（PAC‑Bayes、性能预测、动态风险等）在循环中复用；③ 通过自生成再现oracle将问题文本转换为验证器，从而在不泄露真实测试集的前提下实现高密度、可调的搜索信号；④ 构建两阶时序控制器，快速搜索层与慢速学习层相互配合，避免模型崩溃。

**🔧 技术方法**

采用的技术包括：冻结LLM基础模型+低维softmax指令适配器；正交梯度和经验回放实现持续学习；e‑process、confidence sequence和CTHS实现一次性有效门控；COCOA式因果信用分配；自定义验证器与微步搜索；MAP‑Elites与Stitch式MDL压缩库扩展；自我修复修补原型。

**📊 数据集**

实验使用SWE‑bench Verified子集（52个Python开源项目实例），与四种冻结基础模型（Gemma、Qwen、Gpt‑mini、Gpt）以及GLM 5.2对比，评估在单轮跑（10步）与完整算法堆栈（含搜索、验证等）下的修复成功率。

**📈 对比分析**

与单轮基线和无算法堆栈基线比较，SEA堆栈提升了所有基模型的修复成功率，最高提升为Gpt模型从28/52提升至34/52（约65%），GLM 5.2提升为24/52→28/52。控制实验（无算法但启用指令）表明算法贡献约+5，验证器与微步搜索是主要驱动力。

**⚠️ 局限性**

主要限制包括：循环内保证理论仍为开放猜想，缺乏对性能预测、偏好学习等组件在自演化环境中的正式证明；一次性有效门控过于保守，可能阻碍进展；自生成oracle的可靠性有限，易出现误报或漏报；实验仅单跑一次，缺乏多次重复的统计显著性；慢速学习层（知识蒸馏）未实现，待进一步验证。

---

## 376. From Single to Multiple Attributes: Experimental Insights on Sampling-Based Distinct Combination Estimation in GROUP-BY Queries

**arXiv ID:** 2607.00868 | [PDF](https://arxiv.org/pdf/2607.00868v1)

**作者:** Yujie Zhang `[一作]` (Northeastern University), Yuan Sui `[通讯]` (Northeastern University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并评估多属性 GROUP‑BY 计数的采样估计方法，构造了专用工作负载生成器并在真实数据集及 TPC‑H 基准上进行系统实验。

**💡 创新点**

提出了面向多属性与过滤查询的实验框架，系统对比传统统计、边界修正和学习型方法的表现，揭示单属性信息利用不足及过滤/多表场景下的误差放大问题。

**🔧 技术方法**

使用的技术包括：联合分布采样、单属性频率统计、边界修正（BC、SCBC‑T）、最大似然估计（WD）、多层多项式网络（PolyNet）、独立查询采样（IQS）以及 PostgreSQL 的采样算子。

**📊 数据集**

采用四个真实数据集（Census、Airline、DMV、Campaign）以及 TPC‑H 多表基准进行评估。

**📈 对比分析**

在非过滤查询中，单属性方法误差极低，但多属性场景下误差显著升高；学习方法在高维度下仍出现系统性偏差；边界修正能在高 NDV 数据集上缓解部分误差；过滤与多表查询导致误差激增，甚至出现空样本；在 PostgreSQL 中，误差对计划选择的影响虽不致命，但不准确的 NDV 会导致排序或哈希聚合不匹配。

**⚠️ 局限性**

主要限制包括：仅凭联合分布信息难以精准估计；单属性信息未被充分利用；过滤查询会严重降低有效样本导致误差扩大；多表采样仍缺乏代表性；现有方法缺乏误差不确定性量化。

---

## 377. CAT: Confidence-Adaptive Thinking for Efficient Reasoning of Large Reasoning Models

**arXiv ID:** 2607.00862 | [PDF](https://arxiv.org/pdf/2607.00862v1)

**作者:** Qizhi Jiang `[一作]` (University of Electronic Science and Technology of China), Ke Qin `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于自我置信度的自适应推理框架CAT，能够在大型推理模型中动态压缩不必要的推理链并在需要时保持充分的思考深度。

**💡 创新点**

创新点在于将模型内部的自信度（Self‑Certainty）作为细粒度难度评估指标，配合置信度加权的偏好优化（CWPO），实现对推理长度的自适应调节。

**🔧 技术方法**

核心技术包括Self‑Certainty计算、置信度加权的SimPO偏好优化（CWPO）以及自适应偏好标注（CAPL）。

**📊 数据集**

实验使用了MATH‑500、AIME24和GPQA三大数学/物理推理基准，训练集从MATH训练集随机抽取2000个问题。

**📈 对比分析**

与OverThink、DAST、ConCISE等基线相比，CAT在保持甚至提升准确率的同时显著减少了推理长度（压缩率约30%‑40%），在不同规模模型上均实现了最优的效率‑准确率平衡。

**⚠️ 局限性**

局限性包括仅使用路径级Self‑Certainty可能忽略步骤级置信差异、主要验证于确定性推理任务，且当前是离线偏好优化，缺乏在线强化学习的动态自适应。

---

## 378. Warm-Starting All-Pairs Shortest Paths with Predictions

**arXiv ID:** 2607.00857 | [PDF](https://arxiv.org/pdf/2607.00857v1)

**作者:** Adam Polak `[一作]` (Bocconi University), Jonas Schmidt `[通讯]` (Bocconi University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于预测的全对最短路径（APSP）算法，该算法利用历史相似实例的预测信息来加速计算。

**💡 创新点**

创新点在于引入了预测证书的概念，使得算法在预测误差较小的情况下实现亚立方时间复杂度。

**🔧 技术方法**

使用了共非确定性算法和学习增强算法的框架，结合了Fredman的技巧和支配积的算法。

**📊 数据集**

使用了多个图的快照数据集，具体包括Seattle和PlanetLab网络延迟数据集。

**📈 对比分析**

与传统的APSP算法进行比较，提出的算法在预测误差较小的情况下运行时间为O(n^2.83 + ηn)，其中η为预测误差，性能优于现有的立方时间复杂度算法。

**⚠️ 局限性**

限制在于学习证书的过程是NP-hard，且算法的实际运行时间依赖于快速矩阵乘法的实现，可能在实践中面临挑战。

---

## 379. Submodular Maximization over Many Matroids via Ordered Local Search

**arXiv ID:** 2607.00843 | [PDF](https://arxiv.org/pdf/2607.00843v1)

**作者:** Neta Singer `[一作]` (Ecole Polytechnique Fédérale de Lausanne), Theophile Thiery `[通讯]` (ETH Zurich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在 k‑matroid intersection（或更一般的 Matroid k‑Parity）约束下，寻找最大化单调子模函数的集合，并在加权 k‑Set Packing 的特殊子问题中进一步提升近似比。

**💡 创新点**

提出了一种新的基于局部搜索的算法，采用递减阈值、按边的边际贡献顺序插入以及 α‑swap 规则，首次在单调子模函数和加权设定下实现了与无权最优近似相匹配的 k/2 + o(k) 近似，且在加权 k‑Set Packing 上达到了 ln(4)·k/3 + o(k) 的新纪录。

**🔧 技术方法**

核心技术包括：
- 按降序处理边的边际增益，避免传统的权重分桶导致的贪心失效；
- 定义 α‑swap（增量增益至少 α 倍被移除的增量），通过阈值 θ 连续降低使更小价值的边也能被考虑；
- 利用增量值的单调性和 matroid 基交换构造冲突图，对局部不改进情况进行分析；
- 对加权 k‑Set Packing 结合 Singer‑Thiery 的随机分桶和 Cygan 的无权本地搜索，形成混合 α‑local search 与无权局部搜索的策略。

**📊 数据集**

本文完全为理论算法研究，不涉及实验数据集；所有结果均为证明性的近似比。

**📈 对比分析**

与之前的主要对比包括：
- 对无权情况的 k‑Matroid Intersection，先前最优为 k/ln(4)+O(1)（Singer‑Thiery）和 k/ln(4)/ln(2)+1+o(k)（Feldman‑Ward），本文提供 k/2+o(k)，即在大 k 下与已知最优无权比相匹配；
- 对加权 k‑Set Packing，先前最优为 k/2.00561+O(1)（Neuwohner），本文提升至 ln(4)·k/3+o(k)≈0.462k+o(k)，在大 k 下明显优于先前结果。

**⚠️ 局限性**

主要局限性：
- 近似比是渐近的，实际常数项和低 k（尤其是 k≤5）下的效果未得到改善；
- 对于加权 k‑Set Packing，仍未能达到与无权相同的 k/3+o(k) 目标；
- 算法复杂度为多项式（对 k 为常数），但在实际实现时可能存在较大常数因子；
- 论文未提供实验验证，所有结论仅在理论层面。

---

## 380. Local Motion Matters: A Deconstruct-Recompose Paradigm for Reinforcement Learning Pre-training from Videos

**arXiv ID:** 2607.00808 | [PDF](https://arxiv.org/pdf/2607.00808v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 381. Petrify: Petri-net Based Analysis of Concurrency Properties in Java Bytecode

**arXiv ID:** 2607.00830 | [PDF](https://arxiv.org/pdf/2607.00830v1)

**作者:** Akshatha Shenoy `[一作]` (USI Università della Svizzera italiana), Carlo A. Furia `[通讯]` (USI Università della Svizzera italiana)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过将 Java 字节码编码为 Petri 网，实现对并发程序的自动化模型检查，检测死锁、活锁等并发错误。

**💡 创新点**

① 在字节码层面做分析，支持任意 Java 版本及 Kotlin 子集；② 使用 Petri 网作为并发模型，兼具表达力与可扩展性；③ 采用流/上下文敏感、路径无关的中间表示。

**🔧 技术方法**

使用 Soot 对字节码解析，生成中间语言 Rock Bottom，再转化为 Petri 网；利用 LoLA 进行 PN 模型检查；采用 may‑point‑to 别名分析。

**📊 数据集**

39 个并发程序，包括 JPF、JaDA 仓库的例子、Kotlin 示例、Java 21 新特性、递归/循环规模化程序等。

**📈 对比分析**

与 JPF（Java Pathfinder）和 JaDA 进行比较，三者均在 15 分钟超时内运行；Petri 网模型平均每行代码 2.4 个 place，PN 编码/LoLA 检查平均 <1 秒；总体端到端时间受 Soot 影响，平均 3.8 分钟；在大多数例子中成功检测或确认无死锁；与 JPF 及 JaDA 的性能和覆盖面互补。

**⚠️ 局限性**

仅支持有限的字节码指令（不支持异常、数组等）；别名分析精度不足导致大量锁/线程组合导致内存耗尽；模型无回溯（无法存储无限回溯栈），导致精度损失；未支持递归锁、重入锁等高级同步特性。

---

## 382. Training-Free Debiasing of Diffusion Models via CLIP-Guided Denoising Optimization

**arXiv ID:** 2607.00817 | [PDF](https://arxiv.org/pdf/2607.00817v1)

**作者:** Dain Kim `[一作]` (Hanyang University), Sungyong Baik `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种推理时通过连续优化文本嵌入来实现文本到图像扩散模型的去偏方法。

**💡 创新点**

创新点在于结合两阶段策略（早期全局对齐+噪声逐步细化）与CLIP反馈的无训练文本嵌入优化，实现动态且上下文感知的属性调节。

**🔧 技术方法**

技术主要包括扩散模型（Stable Diffusion）、DDIM采样、CLIP相似度度量、梯度下降式文本嵌入更新以及正则化约束。

**📊 数据集**

使用 Stable Diffusion v2.1/v1.5 进行实验，评估六种专业类别、性别三组、种族三组的图像生成，并使用 FairFace 对生成的面部属性进行标注。

**📈 对比分析**

与原模型及多种无训练去偏基线（debias VL、UCE、EntiGen、LightFair 等）对比，TES 在公平性指标 Bias‑O、Bias‑Q、FD 上均为最佳或第二，且在图像质量指标 CLIP‑T、FID、IS 等方面保持或优于基线。

**⚠️ 局限性**

局限性包括：只能针对预设的属性分布进行调节，难以自适应更复杂的交叉属性；依赖 CLIP 的语义匹配，可能在极端情况下误导；未解决模型整体鲁棒性与多属性同时调节的挑战。

---

## 383. LeVLJEPA: End-to-End Vision-Language Pretraining Without Negatives

**arXiv ID:** 2607.00784 | [PDF](https://arxiv.org/pdf/2607.00784v1)

**作者:** Lukas Kuhn `[一作]` (German Cancer Research Center), Florian Buettner `[通讯]` (German Cancer Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出LeVLJEPA，一种完全非对比的端到端视觉-语言预训练方法；

**💡 创新点**

通过跨模态预测与停止梯度目标以及每模态分布正则化，无需负样本、温度、动量编码器或教师-学生循环；

**🔧 技术方法**

使用ViT图像编码器、GPT‑2文本编码器、单层MLP预测器、SIGReg正则化、stop‑gradient、跨模态MSE损失；

**📊 数据集**

在CC12M（约1200万图文对）和Datacomp‑L（约9200万图文对）上训练；

**📈 对比分析**

与CLIP（InfoNCE）和SigLIP（Sigmoid‑NCE）对比：在全量token级别任务（语义分割、冻结视觉-语言模型）上LeVLJEPA显著优于对比基线，零样本分类和线性探针下表现相当；

**⚠️ 局限性**

缺点是零样本对齐仍落后于对比方法，尚未在更大模型/数据规模上验证，且未结合对齐优势。

---

## 384. Recovering Input Text from Hidden States: Study of Gradient-Based Inversion of Decoder-Only Language Models

**arXiv ID:** 2607.00852 | [PDF](https://arxiv.org/pdf/2607.00852v1)

**作者:** Mikołaj Słowikowski `[一作]` (AGH University of Krakow), Maciej Witold Majewski `[通讯]` (AGH University of Krakow)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

做了什么：研究并实现了一种基于梯度的连续嵌入空间优化方法，用以从解码器仅语言模型（如GPT‑2）最后一层隐藏状态恢复原始文本。

**💡 创新点**

创新点是什么：通过完全保持优化在连续空间并仅在优化结束后一次投影到离散词汇，首次提供了可观测的内部信号（排名轨迹、连续损失曲线、离散损失），并利用累积离散损失实现无监督的错误检测。

**🔧 技术方法**

用了什么技术：采用Adam优化器配合余弦退火与梯度裁剪，对每个位置的代理嵌入进行迭代优化；投影使用余弦或L1距离寻找最近词汇；评估基于GPT‑2小模型，实验在GPU上实现。

**📊 数据集**

用了什么数据集：主要使用C4大规模Web爬取语料的10-token片段，也对维基百科数据集进行了对照实验。

**📈 对比分析**

如何比较的方法，性能怎么样：与公开的SIPIT实现（每步硬投影）对比；SIPIT速度快、精度100%，但缺乏诊断信号。本文在Fast（K=600,C=100）、Baseline（K=1000,C=2000）和High‑accuracy（K=2000,C=10000）三配置下，10-token C4提示的精确率分别约35%、66.9%和97.5%，平均相似度0.994；处理时间从9.7s/token到27.5s/token。

**⚠️ 局限性**

limitation是什么：实验仅限于GPT‑2 small、10-token英文序列、白盒威胁模型；未考察更大模型、编码器或编码-解码器结构、长/流式序列，以及防御措施的影响。

---

## 385. From Pixels to Temporal Correlations: Learning Informative Representations for Reinforcement Learning Pre-training

**arXiv ID:** 2607.00811 | [PDF](https://arxiv.org/pdf/2607.00811v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 386. Mirror-Fusion Attention for Reflection-Aware Self-Supervised Representation Learning

**arXiv ID:** 2607.00850 | [PDF](https://arxiv.org/pdf/2607.00850v1)

**作者:** Ruixin Li `[一作]` (University of Bologna), Stefano Lodi `[通讯]` (University of Bologna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种在现有 Vision Transformer 预训练中注入柔性反射先验的轻量级框架 MFASSL，用于提升对双侧结构数据（如医学影像和人脸图像）的自监督表示学习。

**💡 创新点**

创新点在于：①设计了 Mirror‑Fusion Attention（MFA）模块，可在中间层对镜像对应的 token 进行可学习门控的交叉注意力和差异残差融合；②通过镜像配对视图和两种反射一致性损失（全局与中间层）实现软化的对称性约束，而不需改造网络骨干；③保持预训练仅在镜像分支激活，推理阶段不使用 MFA，保持单前向推理效率。

**🔧 技术方法**

技术方法包括：自监督学习（MoCo‑v3、DINO、MAE）框架下的镜像对齐视图生成、MFA 交叉注意力与门控机制、对称性一致性损失（负余弦相似度和 token 级 L2 对齐）、梯度门 ramp、以及多层次训练策略。

**📊 数据集**

使用的公共数据集有：CheXpert（胸部 X 光）和 BraTS（脑肿瘤 MRI）用于医学任务；CelebA‑HQ 和 WFLW 用于自然图像（人脸属性分类与标记定位）。

**📈 对比分析**

与 MoCo‑v3、DINO、MAE 基线以及近期等变自监督方法（E‑SSL、OcticViT）在相同 ViT‑B/16 结构和训练时长下对比。结果显示：MFASSL 在多任务中均提高 0.5–1.2 pp 的 AUROC/准确率、1–2 mm 的 Dice/HD95、并提升 Flip‑Consistency，且参数增加仅约 2.7%。

**⚠️ 局限性**

局限性包括：① 需要预先知晓垂直反射轴，轴不准确时效果下降；② 仅适用于近似双侧对称结构，对非对称或多轴结构的场景不一定有益；③ MFA 仅在中间层稳定，早层或深层插入可能导致梯度爆炸；④ 目前未评估对旋转、尺度等更复杂对称变换；⑤ 仅在从零训练的设置下验证，未与 ImageNet 初始化或 flip test‑time averaging 等常规做法比较。

---

## 387. Knowledge-Enhanced Agentic Vulnerability Repair

**arXiv ID:** 2607.00820 | [PDF](https://arxiv.org/pdf/2607.00820v1)

**作者:** Sicong Cao `[一作]` (Nanjing University of Posts and Telecommunications), David Lo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种知识增强的代理式漏洞修复方法（KAR），通过构建双视图（控制流与数据流）漏洞知识库、工具增强的诊断代理以及检索增强的补丁生成流程，实现自动修复C/C++及多语言漏洞。

**💡 创新点**

创新点：①利用静态分析工具收集验证的程序事实，降低LLM幻觉；②从历史补丁中提取并抽象双视图多维知识，提升检索相关性；③闭环迭代验证与多轮补丁生成，显著提高修复质量。

**🔧 技术方法**

核心技术：大型语言模型（Gemini‑3.1‑Pro、GPT‑5等）、ReAct式工具代理、三类静态分析工具（trace_data_flow、is_path_reachable、inspect_api_usage）、向量检索（Faiss）与双重重排序（CodeLlama+BM25）。

**📊 数据集**

使用 Vul4C（55个可重现的C/C++ CVE）与 PatchEval（173个多语言 CVE）进行评测，并构建 PrimeVul 知识库用于检索。

**📈 对比分析**

与八类基线（Constraint、Learning、LLM、Agent）对比，KAR 在55例中修复率达83.64%，比 PatchAgent 高 17.95%；在 PatchEval 多语言数据集上达85.55% 的成功率，显著优于现有方法。

**⚠️ 局限性**

局限性：对宏、编译条件等复杂代码的静态分析精度不足；仅函数级知识难以处理跨文件漏洞；LLM 可能受预训练数据泄漏影响；跨语言适配仍需针对性改进。

---

## 388. Effective Stochastic Automata Model Checking by Interval Abstraction (extended version)

**arXiv ID:** 2607.00782 | [PDF](https://arxiv.org/pdf/2607.00782v1)

**作者:** Pedro R. D'Argenio `[一作]` (Universidad Nacional de Córdoba and CONICET), Annabell Petri `[通讯]` (University of Twente)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种针对随机自动机（Stochastic Automata, SA）的模型检验方法，通过区间抽象将连续概率分布近似为有限离散区间，再利用大时间步语义构造马尔可夫决策过程（MDP），从而得到SA中目标状态可达性的上下界。

**💡 创新点**

创新点在于：①首次将区间抽象直接应用于SA，既保持了对一般连续分布的支持，又兼顾了非确定性决策；②在不需要对SA做严格限制的前提下，提供了有效的上/下界计算；③提出了可细化的区间策略以改进近似误差。

**🔧 技术方法**

主要技术包括：随机自动机的“大时间步”语义、区间抽象（将连续分布分割为概率相等的区间）、离散化后的MDP构造、价值迭代（VI）求解最优可达概率、以及在Rust中实现原型工具。

**📊 数据集**

实验使用了多组 SA 模型：运行示例、M1/M2/M5 等传统案例、M/M/1/100 与 Weibull/E_6/1/100 排队系统、以及文件服务器模型（Uniform/Exponential 的交互时间）。

**📈 对比分析**

与现有技术（如 Fig 的统计模型检验、mcsta 的 STA 数字时钟方法）进行对比，结果显示：①在大多数模型上，区间抽象能在几秒到十几分钟内完成，内存消耗可控；②在精度方面，上/下界与统计检验得到的参考值一致；③在文件服务器案例中，区间抽象在相同误差下比 mcsta 运行时间低 30% 以上；总体性能表现优于之前针对 SA 超类的分析方法。

**⚠️ 局限性**

局限性包括：①区间抽象产生的误差随区间宽度变化并非单调，尚未证明误差趋于零；②目前工具仅支持有限几种连续分布（均匀、指数、埃朗、威布尔），扩展需要实现逆累积分布；③在某些模型中，区间生成方式（按等概率）可能导致极宽区间，从而导致 MDP 体积膨胀；④未实现对时间有界属性或确定性延迟的分析。

---

## 389. OmniView-Space: Reinforcing Spatial Reasoning via Multi-Perspective Spatial Mapping

**arXiv ID:** 2607.00881 | [PDF](https://arxiv.org/pdf/2607.00881v1)

**作者:** Xudong Li `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 OmniView‑Space 框架，通过 Multi‑Perspective Spatial Mapping (MPSM) 构建查询对齐的 egocentric 认知地图（视觉 BEV 与文本图谱），并利用工具指导的强化学习策略与蒸馏方法，使模型能够在多模态输入下实现精确的空间推理。

**💡 创新点**

创新点在于：① 引入查询对齐的 egocentric 认知地图，解决传统方法在多步推理中因参考系不匹配导致的空间不一致问题；② 结合视觉 BEV 地图与文本空间图谱，实现多模态信息的互补；③ 通过 RL 训练的工具调用策略与 Map‑Alignment 奖励，实现从外部几何工具到自生成认知地图的无缝迁移。

**🔧 技术方法**

采用 3D 重建、BEV 渲染、文本空间图谱构造、GRPO 强化学习、工具调用策略、Supervised Fine‑Tuning 与 Map‑Alignment 奖励，模型基于 Qwen3‑VL‑4B 进行训练与推理。

**📊 数据集**

使用多视图评测基准（MindCube‑Tiny、MMSI‑Bench、SPAR‑Bench、SPBench）以及单图评测基准（3DSRBench Allocentric 子集、OmniSpatial Perspective 子集），并在 SpatialLadder、3DSRBench 等公开数据集上进行验证。

**📈 对比分析**

与现有开源与专有模型对比，OmniView‑Space 在多视图空间推理上取得最高开源成绩（MindCube‑Tiny 71.5%、MMSI‑Bench 35.5%、SPAR‑Bench 44.8%），单图任务中也领先；蒸馏版模型在保持 46.5% 总分的同时，仅略低于完整工具管线（50.6%），展现出较强的鲁棒性与显著的性能提升。

**⚠️ 局限性**

局限性包括：1）完整性能仍依赖外部几何重建管线，蒸馏版在某些复杂场景下的准确率略低；2）单图任务中对视角变换需求不高时，传统文本推理可与之相当；3）对多步推理的长序列与大量视角仍可能出现误差；4）模型对几何重建误差敏感，需进一步提升对噪声鲁棒性。

---

## 390. Improved Approximation Algorithms for Parallel Task Scheduling and Multiple Cluster Scheduling

**arXiv ID:** 2607.00878 | [PDF](https://arxiv.org/pdf/2607.00878v1)

**作者:** Bennet Edler `[一作]` (Kiel University), Lis Pirotton `[通讯]` (Kiel University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了快速的 4/3+ 近似算法，并将其扩展到多集群调度 (MCS)，实现了 2 近似（渐进）和 9/4 绝对近似；

**💡 创新点**

通过稀疏区间分析和面积欠缺度量，将 3/2+ 近似提升到 4/3+，并在 MCS 中得到更优的渐进与绝对比率；

**🔧 技术方法**

主要技术包括 List‑Scheduling、稀疏区间划分、面积欠缺分析以及对小作业的贪心放置；

**📊 数据集**

实验使用随机生成的处理时间和机器需求数据集；

**📈 对比分析**

与 3/2+、AEPTAS、MCS 2、9/4 等先前算法比较，算法仅需 O(n log n) 时间，在多机集群场景中实现更快的运行速度和更优的实际性能；

**⚠️ 局限性**

仍保留加性项，无法进一步降至 5/4+，且不适用于连续版（strip packing）问题。

---

## 391. MoVA: Learning Asymmetric Dual Projections for Modular Long Video-Text Alignment

**arXiv ID:** 2607.00858 | [PDF](https://arxiv.org/pdf/2607.00858v1)

**作者:** Peiyuan Zhu `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Kun Zhang `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了MoVA模型，用于解决视频-文本对齐中的时间错位和语义不对称问题；

**💡 创新点**

创新点在于通过双向非对称投影与模块化对比学习，结合理论识别条件，实现视频帧与文本片段的灵活对齐与分离；

**🔧 技术方法**

采用的技术包括Transformer-based Temporal Mask Network与Concept Mask Network、稀疏二值掩码、模块化对比损失以及全局检索损失的组合；

**📊 数据集**

实验使用了ActivityNet、MSVD、DiDeMo、VideoUFO、UltraVideo等公开视频-文本数据集；

**📈 对比分析**

与现有方法对比，MoVA在R@1/R@5/R@10等检索指标上均取得领先成绩，且在长文本生成与多模态检索任务中表现优异；

**⚠️ 局限性**

局限性在于仍依赖CLIP预训练的文本和视觉编码，且对极长视频或复杂文本的实时推理存在计算与内存瓶颈。

---

## 392. Generative Retrieval for Table Union Search

**arXiv ID:** 2607.00833 | [PDF](https://arxiv.org/pdf/2607.00833v1)

**作者:** Shulun Zhang `[一作]` (Chinese University of Hong Kong), Chenhao Ma `[通讯]` (Chinese University of Hong Kong)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种生成式检索框架 GenTUS，用于表联合搜索（TUS），通过在离线阶段为每张表生成语义标识符（SID），并在在线阶段直接生成满足约束的 SID 来检索可联合的表。

**💡 创新点**

创新点在于：①将 TUS 重新表述为离散标识符的受约束生成任务；②设计了联合可用性感知的残差量化标识符生成方法，使得可联合的表在标识符空间中具有相似的粗粒度码；③使用前缀树对生成过程进行约束，确保生成的标识符一定对应现有表，从而省去了 ANN 索引和后期重排序。

**🔧 技术方法**

核心技术包括：表序列化与预训练表编码器（如 TaBERT/TaPas），残差量化的联合可用性对比损失，T5 编码–解码生成模型以及基于束搜索的前缀约束解码。

**📊 数据集**

实验使用七个公开 TUS 基准数据集：TUS-S、TUS-L、SAN-S、SAN-L、Wiki、LakeBench‑1K 和 LakeBench‑30K。

**📈 对比分析**

与多种基线（基于类型、列聚合、表向量及 TACTUS 等）对比，GenTUS 在 MAP、P@k、R@k 上均取得领先，平均排名从 2.57 提升至 1.05；同时在线检索延迟降低 1.3–220 倍，离线构建时间缩短 1.1–6.5 倍，检索索引存储占用也最小。

**⚠️ 局限性**

局限性包括：①需要预先获得联合可用性标注来训练标识符量化器；②生成模型的训练和推理仍依赖 GPU，且束宽对 latency 影响较大；③在极大规模湖泊或分布漂移场景下，标识符空间可能需要重构，且生成模型更新成本尚未完全解决。

---

## 393. A field experiment of social influence and behavioral contagion with bots on Reddit

**arXiv ID:** 2607.00854 | [PDF](https://arxiv.org/pdf/2607.00854v1)

**作者:** Hiroki Oda `[一作]` (London School of Economics and Political Science), Milena Tsvetkova `[通讯]` (London School of Economics and Political Science)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Reddit上开展现场实验，向选定的低经验用户发放带有不同理由的非匿名象征性奖赏，分别由人工账号和机器人账号进行，随后测量获奖者及其帖子的活跃度、影响力与内容变化；

**💡 创新点**

首次系统比较人类与机器人在不同奖赏理由（逻辑、情感、道德、随机抽签）下对受奖者行为的影响，并发现机器人随机抽奖奖赏反而产生负面效果；

**🔧 技术方法**

采用随机实验设计、Reddit API数据抓取、文本情感与词汇分析（LIWC-22、道德词典）、统计检验（Mann‑Whitney U）、对比前后差异；

**📊 数据集**

共收集2442条来自12个非敏感子版块的帖子/评论样本（过滤后有545条符合文本长度要求的受试者），数据来源为2025年4月至2026年1月的Reddit公开API；

**📈 对比分析**

通过比较处理组与对照组的差分（log后/前）并使用非参数检验，发现绝大多数处理无显著提升，唯一显著正向效应为人类账号提供道德理由时提升受奖者后续投票量，机器人随机抽奖则导致贡献量和正向语调下降；

**⚠️ 局限性**

局限性包括样本量相对有限、Reddit平台规则与通知机制频繁变动、受试者对奖赏是否注意不确定、实验为短期观察且未获得事前知情同意，结果可能不具普遍性。

---

## 394. The Binary Tree Mechanism is Optimal for Approximate Differentially Private Continual Counting

**arXiv ID:** 2607.00876 | [PDF](https://arxiv.org/pdf/2607.00876v1)

**作者:** Konstantina Bairaktari `[一作]` (Aarhus University), Kasper Green Larsen `[通讯]` (Aarhus University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文证明了私有连续计数的最优误差下界为 Ω(log^{3/2}n)，与现有的二叉树机制的误差上界相匹配，从而确认该机制在近似差分隐私下是渐近最优的。

**💡 创新点**

创新点在于提出了一种新的树基底噪声分析与“残差噪声”相结合的证明框架，利用线性测量和 Haar 小波基实现对误差的精确下界推导，解决了长期悬而未决的误差依赖 n 的问题。

**🔧 技术方法**

主要技术包括：构造二叉树基底将噪声分解为沿根-叶路径的累积项；定义并下界残差噪声 σ_u；使用三份独立噪声和三样本技巧控制 L₂ 范数；运用线性测量和 Haar 小波基实现对 σ 的全局约束；并通过隐私攻击与 DP 约束结合得到最终的 Ω(log^{3/2}n) 结果。

**📊 数据集**

本工作为理论分析，未使用任何实际数据集，所有结论均基于数学证明和概率论推导。

**📈 对比分析**

与传统的二叉树机制（上界为 O(log^{3/2}n√{log(1/δ)}/ε)）比较，本文给出了相同阶数的下界，表明在近似差分隐私模型中误差的 log^{3/2}n 依赖是不可避免且已达到最优。

**⚠️ 局限性**

局限性包括：只针对单次事件计数的前缀和查询；证明仅覆盖 0<ε<1、δ< C（小常数）范围；未考虑稀疏流或在线/离线模型的细微差别；对非二叉树结构的查询尚未得到相应的下界。

---

## 395. TrajLoc: Trajectory-Attention Localization for Multi-Object Motion Control

**arXiv ID:** 2607.00861 | [PDF](https://arxiv.org/pdf/2607.00861v1)

**作者:** Omer Sela `[一作]` (Amazon Prime Video), Avi Ben-Cohen `[通讯]` (Amazon Prime Video)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于注意力定位的多对象轨迹控制视频生成方法，直接在跨注意力中用高斯热图替换每个对象文本令牌的权重，并在文本提示中加入轨迹和外观专属令牌。

**💡 创新点**

创新点在于不需要稠密的轨迹条件张量，利用现有注意力机制和文本嵌入空间即可实现严格的对象级空间约束和身份保持，支持数十个交互对象。

**🔧 技术方法**

技术包括跨注意力权重列替换、轨迹热图生成、轨迹编码器、外观编码器、LoRA微调以及基于预训练的图像-视频扩散模型。

**📊 数据集**

使用合成数据集MOTSynth、ViN（MoVi-Extended、Pool、Football）训练，并在六个评测集（四个合成+MOT17、DAVIS）上测试。

**📈 对比分析**

与Wan-Move、ATI、Tora、MagicMotion等基线比较，平均PSNR提升4.3dB，终点误差降低51%，在CogVideoX-5B与WaN 2.1-14B两大骨干上均取得SOTA。

**⚠️ 局限性**

局限在于仅处理固定相机、720p分辨率的场景，训练数据为合成，真实场景仍存在域差异，且模型可能产生与真实视觉不符的合成外观；扩展至移动相机、高分辨率和真实数据仍待研究。

---

## 396. The Course of News Events: A Comparison of Bottom-Up and Top-Down Approaches for Collecting Text-Based Data about Disasters

**arXiv ID:** 2607.00849 | [PDF](https://arxiv.org/pdf/2607.00849v1)

**作者:** Brielen Madureira `[一作]` (Leipzig University), Mariana Madruga de Brito `[通讯]` (Helmholtz Centre for Environmental Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文比较了在德国语新闻语料中，基于顶向下（查询已知灾害数据库EM‑DAT）和底向上（利用NLP聚类新闻文本）两种方法检测山体滑坡事件的效果。

**💡 创新点**

创新点在于系统性地对比两种数据抽样策略，揭示它们在事件覆盖率、时间一致性与空间分布上的差异，并指出两者相互补充的局限。

**🔧 技术方法**

采用时间序列分割、地理信息提取、自然语言处理（包括关键词检索与大型语言模型支持的地理解析）等技术；事件匹配通过时间窗口对齐实现。

**📊 数据集**

使用了约55,000篇德国语山体滑坡相关新闻，和来自EM‑DAT的2,014条全球滑坡事件记录。

**📈 对比分析**

通过计算与EM‑DAT条目时间对齐的事件比例和查询成功率，结果显示底向上方法生成的事件数是EM‑DAT的两倍，但仅有16.1%与EM‑DAT匹配；顶向下查询成功率为42.2%。两方法共同覆盖了762个EM‑DAT条目，但仍存在大量未匹配事件，表明两种策略各自缺失信息。

**⚠️ 局限性**

局限包括：NLP管道误报导致噪声、地理定位不精准、对新闻媒体代表性缺失的事件无法捕捉、事件时间窗口参数对匹配结果影响大、以及未能解决同一事件被多次或多来源复合聚类的问题。

---

## 397. MMAO-Dyn: A Metabolic Multi-Agent Optimizer for Dynamic Optimization

**arXiv ID:** 2607.00846 | [PDF](https://arxiv.org/pdf/2607.00846v1)

**作者:** Jinliang Xu `[一作]`, Liping Ma `[通讯]` (Chinese PLA General Hospital)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了MMAO的动态优化衍生版本MMAO-Dyn，在不放弃核心代谢循环的前提下，将其映射到非静态优化问题；

**💡 创新点**

创新点在于证明代谢多代理优化器可以通过内生资源重新分配机制实现动态恢复，而不需要外部适配模块；

**🔧 技术方法**

采用MMAO代谢循环中的资源注入、恢复窗口、角色漂移、记忆刷新与新代理注入等内生机制，并结合连续搜索算子实现动态搜索；

**📊 数据集**

使用18个场景的合成动态连续基准矩阵（10D、20D、30D的移位Sphere、Ackley、Rastrigin），每场景2种变化强度，共计216次实验；

**📈 对比分析**

与Generic‑MMAO、动态随机搜索、动态PSO‑lite、动态DE‑lite以及三种消融模型进行比较；MMAO‑Dyn在平均离线误差上比Generic‑MMAO低约1.3点，显著优于随机搜索和DE‑lite，整体平均排名略低于PSO‑lite，但在10步恢复窗口上表现相当；多场景评估显示其恢复能力稳健；

**⚠️ 局限性**

局限性包括：仅在合成连续基准上测试，基准规模有限；基线轻量，缺乏更专业的动态优化器对比；未涵盖离散或组合优化问题；部分超参数仍需手动调节。

---

## 398. CellPrior-Net: Prior-Guided Nuclei Detection and Classification for H&E Whole-Slide Images

**arXiv ID:** 2607.00802 | [PDF](https://arxiv.org/pdf/2607.00802v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 399. Rethinking Multi-Label Image Classification With Deep Learning: Taxonomy, Challenge, and Outlook

**arXiv ID:** 2607.00839 | [PDF](https://arxiv.org/pdf/2607.00839v1)

**作者:** Xuelin Zhu `[一作]` (Hong Kong Polytechnic University), Bing Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文系统综述了多标签图像分类（MLIC）领域的深度学习方法，更新至2026年，并梳理了最新的数据集、主干网络和评估指标，提出了面向Transformer架构和大规模数据的研究框架；

**💡 创新点**

创新点在于填补了早期过时综述的空白，首次全面覆盖Transformer及其变体在MLIC中的应用；构建了统一的评估与比较体系，阐明了大规模数据集与多标签学习挑战之间的关系；

**🔧 技术方法**

综述技术涵盖CNN（AlexNet、VGG、ResNet系列）、Transformer（ViT、Swin等）以及多标签学习机制（自监督、半监督、极端分类、标签不平衡处理等）；

**📊 数据集**

使用的数据集包括Pascal VOC 2007/2012、MS-COCO 2014、NUS-WIDE、VG500、WIDER-Attribute、PA-100K、Open Images v6等多种大规模和细粒度标注数据集；

**📈 对比分析**

通过AP、mAP、OP、OR、OF1、CF1等指标对现有方法进行横向比较，展示Transformer在mAP、OF1等指标上相较传统CNN的显著提升，并对比了不同标签数与不平衡场景下的性能差异；

**⚠️ 局限性**

局限性在于缺乏统一的基准测试环境与评测规范，未充分探讨多模态大模型（LLM、视觉语言模型）在MLIC中的迁移与整合，且对实际部署与实时推理的评估仍不充分。

---

## 400. Modeling and Chasing the Energy-Efficiency Sweet Spots in Modern GPUs

**arXiv ID:** 2607.00819 | [PDF](https://arxiv.org/pdf/2607.00819v1)

**作者:** Ayesha Afzal `[一作]` (Erlangen National High Performance Computing Center), Michael Panzlaff `[通讯]` (Erlangen National High Performance Computing Center)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究不同GPU频率与功率限制对分子动力学与合成基准的能耗影响，并提出可解释的分段功率模型；

**💡 创新点**

通过将GPU功率拆分为基线、线性和二次项，揭示能耗峰值与转折频率的物理关联，指出频率调节优于功率限制；

**🔧 技术方法**

基于CUDA与MPI/OMP实现的GROMACS、AMBER与FIRESTARTER基准、DVFS频率/功率调节、线性与二次回归模型；

**📊 数据集**

使用多种生物分子体系的MD基准（TRPCage、Cellulose、STMV等）、FIRESTARTER合成基准，在A40、A100、H100、H200与Intel Ice Lake CPU平台上采集功率与性能数据；

**📈 对比分析**

通过Pareto效率-能耗图和能耗延迟乘积（EDP）对比，显示频率缩放能显著提升能效（能耗下降10–40%），而功率限制效果有限；

**⚠️ 局限性**

仅适用于GPU加速的MD与合成工作负载，CPU功率模型未涵盖热/电压非线性，且基准覆盖范围有限，难以直接推广到其他加速器或深度学习工作负载。

---

## 401. Task-Relevant Representation Decoupling for Visual Reinforcement Learning Generalization

**arXiv ID:** 2607.00796 | [PDF](https://arxiv.org/pdf/2607.00796v1)

**作者:** Jinwen Wang `[一作]` (Beijing Jiaotong University), Kai Lv `[通讯]` (Beijing Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种自监督的视觉强化学习框架T2RD，用于将视觉观测解耦为任务相关与无关表示，从而提升泛化能力和样本效率。

**💡 创新点**

创新点在于三阶段解耦机制（任务相关表示一致性、跨重建、跨动态预测）以及结合随机卷积与随机叠加的多样化数据增强，显著减少对视觉干扰的敏感性。

**🔧 技术方法**

采用BYOL风格的双分支网络（编码器、投影头、预测器）配合交叉重建网络和动态预测网络，并与Soft Actor-Critic（SAC）策略学习相结合。

**📊 数据集**

主要使用DeepMind Control Suite Generalization Benchmark和Robotic Manipulation任务作为实验数据集。

**📈 对比分析**

与SODA、SVEA、DRQ、SVEA等现有SOTA方法进行对比，T2RD在大多数任务和不同视觉变化设置下实现了最优的泛化性能和更高的样本效率。

**⚠️ 局限性**

限制在于假设任务相关与风格可分离，若任务相关信息与视觉风格交织，模型可能误判；此外，尚未在真实机器人上验证其效果。

---

## 402. Which Metric Reflects the Spelling Rate Accuracy in Event-Related Potential-Based Brain-Computer Interfaces?

**arXiv ID:** 2607.00794 | [PDF](https://arxiv.org/pdf/2607.00794v1)

**作者:** Okba Bekhelifi `[一作]` (Universite Des Sciences Et De La Technologie D'Oran), Naoual El Djouher Mebtouche `[通讯]` (Universite Des Sciences Et De La Technologie D'Alger)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了ERP基脑机接口中的拼写率与13种二分类评估指标的相关性，揭示哪些指标最能反映拼写性能。

**💡 创新点**

创新点在于系统性比较多种指标的相关性，提出Brier、MCC、ROC AUC、PR AUC、AP和pAUC等指标最能代表拼写率，并探讨试验重复对相关性的影响。

**🔧 技术方法**

采用经典的xDAWN+Logistic回归与基于深度学习的EEG‑TCNet卷积网络两种分类器，并使用多种统计检验与相关系数评估性能。

**📊 数据集**

使用了本地LARESI ERP数据集（6名受试者）和公开OpenBMI ERP数据集（54名受试者）进行实验。

**📈 对比分析**

通过Wilcoxon符号秩检验、Cohen's d、Pearson和Kendall相关系数比较两模型的表现，结果显示EEG‑TCNet在多指标上优于Logistic回归，尤其在多次重复试验时相关性更高。

**⚠️ 局限性**

研究仅聚焦拼写率指标，未考虑实时错误纠正或更复杂多字符拼写情境，且实验仅覆盖两种特定的ERP拼写器方案，限制了结果的普适性。

---

## 403. MetaHOPE: A Metaphor-Oriented Evaluation Framework for Analysing MT and LLM Translation Errors

**arXiv ID:** 2607.00848 | [PDF](https://arxiv.org/pdf/2607.00848v1)

**作者:** Jiahui Liang `[一作]` (Leiden University), Lifeng Han `[通讯]` (Leiden University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 MetaHOPE 框架，用于对机器翻译中的隐喻翻译进行错误严重性评估与标注。

**💡 创新点**

创新点在于将 HOPE 框架改造成专门针对隐喻的评价体系，缩减错误类别至五类并引入五级严重度评分。

**🔧 技术方法**

采用机器翻译（GoogleMT、GPT‑5.4、Hunyuan‑7B）、人工对齐与评注、后编辑生成金标，以及统计分析与定性错误归类技术。

**📊 数据集**

使用来自 VUAMC 与 PSUCMC 的中英双向新闻隐喻语料库，经过预处理后对 20/200 句子进行标注和评估。

**📈 对比分析**

通过对标注一致性、错误比例（GoogleMT 91.7%、GPT‑5.4 93.8%、Hunyuan‑7B 61.8%）以及错误类型分布进行比较，发现 Hunyuan‑7B 在灵活性与本地化上优于其它系统，但易产生事实偏差。

**⚠️ 局限性**

局限性包括标注者对严重度的偏差导致一致性低、样本量有限、仅覆盖新闻域、缺乏跨语言与多模型的全面验证。

---

## 404. Pano2World: End-to-End 3D Generation via Unified Multi-View Sequences

**arXiv ID:** 2607.00832 | [PDF](https://arxiv.org/pdf/2607.00832v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 405. ClinRAG-GRAPH: Clinical-prior Retrieval-Augmented Graph Model with Domain Adversarial Learning for Breast pCR Prediction

**arXiv ID:** 2607.00798 | [PDF](https://arxiv.org/pdf/2607.00798v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 406. From World Models to World Action Models: A Concise Tutorial for Robotics

**arXiv ID:** 2607.00836 | [PDF](https://arxiv.org/pdf/2607.00836v1)

**作者:** Xiaoxiong Zhang `[一作]` (Southern University of Science and Technology), Wei Zhang `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统梳理并统一了“世界模型”和“世界动作模型”的定义与分类，提出了基于观察空间与状态空间的设计空间视图，并详细介绍了四种代表性范式（想象‑然后‑执行、视频特征‑条件动作预测、联合视频‑动作建模、辅助视频预测用于策略学习）。

**💡 创新点**

创新点在于：①给出“世界模型”与“世界动作模型”的统一概念与框架；②用二维轴（观测空间显式度与动作抽象层级）描述观察空间模型，利用状态抽象（潜在、点轨迹、神经符号、物理状态）阐释状态空间模型；③提出四种将预测与动作生成耦合的范式，并讨论各自的优缺点；④为后续研究提供了清晰的分类与参考。

**🔧 技术方法**

主要技术包括：视频预测网络（如扩散模型、LSTM、Transformer）、逆动力学模型、特征提取与条件学习、神经符号推理、物理模拟器、语言条件化控制器、强化学习与模型预测控制等。

**📊 数据集**

文章引用了多种公开视觉与机器人数据集，如大规模RGB视频、RGB‑D视频、点云数据集、机器人轨迹与动作标注数据集（如Robotic‑Open‑AI、Dexterous‑Hands、REAL‑robot‑datasets），但并未在本文中进行统一实验，主要以文献综述为主。

**📈 对比分析**

由于是综述性质，本文并未给出统一的实验对比；作者通过文献引用对比不同模型在视觉保真度、空间结构、物理可解释性、控制可用性等维度的取舍，并在表格中列出代表方法的优势与不足，强调各范式在不同任务（如行走、抓取、视觉规划）中的适用场景。

**⚠️ 局限性**

局限性：①缺乏统一实验与定量评估，难以直接衡量不同范式的性能差异；②对数据集依赖较大，部分模型在真实机器人上的可迁移性尚待验证；③对跨域泛化与大规模部署的挑战讨论不足；④在实际控制中如何将高层视觉计划与低层执行耦合仍需进一步研究。

---

## 407. Spectroscopy Analysis with Machine Learning Regression for the Quantification of Carbon and Nitrogen Contents in Inceptisol and Oxisol Soil Types: Comparing Different Preprocessing and Validation methods as well as Feature Importance

**arXiv ID:** 2607.00834 | [PDF](https://arxiv.org/pdf/2607.00834v1)

**作者:** Vinicius Herique Kieling `[一作]` (Federal University of Technology - Paraná), Jefferson Tales Oliva `[通讯]` (Federal University of Technology - Paraná)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

利用便携式近红外（NIR）光谱与机器学习（ML）相结合，构建预测巴西两种土壤（Oxisol与Inceptisol）中碳（C）与氮（N）含量的校准模型；

**💡 创新点**

创新点包括：①将Savitzky‑Golay滤波与基于Huber损失的NIPALS异常值去除组合成最优预处理流程；②采用Stacking集成学习，将PLS、SVR、Ridge等基模型与线性元模型结合，提升预测稳健性；③通过岭回归提取关键波长，定位对C、N预测最重要的1300‑1400 nm区域，减少模型维度；

**🔧 技术方法**

主要技术：近红外光谱采集（MyNIR 900–1700 nm）、多种预处理（SG、LOG、TRIM、SNV、MSC、Outlier等）、数据标准化、PLS、SVR、Ridge、RF、Stacking集成；

**📊 数据集**

数据集：153个土壤样本（Oxisol 72样，Inceptisol 81样），每样本均收集两次NIR光谱并通过Walkley‑Black与Kjeldahl法测定C、N参考值；

**📈 对比分析**

比较方法：使用K‑Fold交叉验证、留一交叉验证和Kennard‑Stone Holdout（70/30或75/25）三种验证策略；在Oxisol中，最佳模型为SVR+标准化，R²≈0.91、RPD≈3.39（C）和R²≈0.89、RPD≈3.01（N）；在Inceptisol中，Stacking（PLS+Ridge）或单一PLS表现最佳，R²≈0.79–0.88、RPD≈2.1；整体模型均达RPD>2.0，表明可实现快速、非破坏性测定；

**⚠️ 局限性**

局限性：①样本量相对有限，易导致过拟合（如RF模型表现不佳）；②仅评估两种土壤类型，推广性需验证；③预处理与模型参数选择仍多依赖经验与网格搜索，缺乏自动化流程；

---

## 408. Towards High-Resolution Visual Perception via Hierarchical Entity Exploration

**arXiv ID:** 2607.00816 | [PDF](https://arxiv.org/pdf/2607.00816v1)

**作者:** Ziyu Ma `[一作]` (Alibaba Group), Xiangxiang Chu `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为Hierarchical Entity Exploration (HEE) 的训练无关、模型无关的高分辨率视觉感知框架，能够在不修改模型或额外训练的前提下，通过实体驱动的层级搜索实现细粒度目标定位与问答。

**💡 创新点**

创新点包括：① 采用实体（由冻结检测器生成）而非几何网格划分，避免实体碎片化与背景干扰；② 设计双重评分机制，融合语义相似度与VLM置信度，以更可靠地判断是否继续深入搜索；③ 引入置信度引导的回溯策略，在低置信度路径下自动重选更佳子节点，实现稳健推理。

**🔧 技术方法**

核心技术：冻结对象检测器（DINO‑X）、语义聚类（K‑Means）生成子节点、CLIP/SigLIP 计算问答语义相似度、VLM（如 Qwen2.5‑VL、InternVL、LLaVA-OneVision）进行置信度评估、递归层级搜索与回溯控制。

**📊 数据集**

使用的基准数据集：Visual Probe（4K 高分辨率问答）、HR‑Bench（4K/8K 细粒度视觉任务）、MME‑RealWorld（13K+ 高分辨率多任务QA）。

**📈 对比分析**

与现有训练无关方法（RAP、ZoomEye）及多种 MLLM 进行对比，结果显示：HEE 在 Visual Probe 的 Easy、Medium、Hard 子集分别提升约 21–28%；在 HR‑Bench 4K 达到 73.9% 最高准确率，且推理速度提升至 24 min（相较 RAP 的 126 min 与 ZoomEye 的 46 min）。在 MME‑RealWorld 多任务上同样获得 3–7% 的平均提升，证明其在不同模型和任务上的鲁棒性。

**⚠️ 局限性**

局限性：① 对检测器的召回率和定位精度高度依赖，检测失败会导致定位错误或无法进一步搜索；② 在极大尺寸图像或对象极小、非常细腻的场景下，层级递归和回溯深度可能不足；③ 仍存在细粒度识别错误与置信度误导的情况，需进一步完善后处理与自适应阈值。

---

## 409. SpiralFovea: Input-Adaptive Foveated Tokenization as a Third Lever of Resource-Adaptive Inference

**arXiv ID:** 2607.00780 | [PDF](https://arxiv.org/pdf/2607.00780v1)

**作者:** Kyan Mahajan `[一作]` (Indian Institute of Information Technology), Mohammad Saqlain `[通讯]` (Indian Institute of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无参数、基于局部视觉熵的螺旋型 foveated tokenization (SpiralFovea)，在冻结的 ViT 或 ResNet backbone 前先筛选内容相关的少量补丁。

**💡 创新点**

创新点在于将输入 token 集视为第三个自适应推理杠杆，使用无学习参数的熵引导热点与多尺度螺旋环生成内容驱动的 token，并证明其对自监督基础模型最有价值。

**🔧 技术方法**

采用局部熵计算、热点定位、螺旋多尺度采样、极坐标位置编码，以及与 frozen ViT/DINO 或 ResNet+Mamba 的结合。

**📊 数据集**

使用四个细粒度识别基准：WikiArt GAN‑Genre、WikiArt Style、Oxford Flowers‑102、PatchCamelyon，以及 CUB‑200‑2011 细分类别。

**📈 对比分析**

在各 backbone 上与均匀 tokenization 对比，平均提升 1.7–2.1 个百分点，同时将输入 token 数量减少 60%，自注意力 FLOPs 降低 84%，吞吐率提升 18–29%。

**⚠️ 局限性**

局限包括：只对空间集中信息的细粒度任务有效，热点仅在水平条带内分布，序列长度可变需 padding，未验证在全图分类或文本等非图像模态的效果。

---

## 410. Improving Sparse-View 3DGS Generalization via Flat Minima Optimization

**arXiv ID:** 2607.00885 | [PDF](https://arxiv.org/pdf/2607.00885v1)

**作者:** Kangmin Seo `[一作]` (Sungkyunkwan University), Jae-Pil Heo `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在稀疏视角下改进3D Gaussian Splatting的泛化能力，提出基于平坦极小点优化的轻量级训练框架。

**💡 创新点**

引入尺度自适应噪声、随机应用与增量调度以及周期性重初始化的平坦极小点正则化，使Gaussian参数在稀疏视角下更稳健。

**🔧 技术方法**

平坦极小点（FM）优化、Scale‑Adaptive Perturbation、随机扰动、参数重初始化、光度损失+SSIM+LPIPS。

**📊 数据集**

LLFF和Mip‑NeRF360两个常用稀疏视角基准。

**📈 对比分析**

与3DGS、CoR‑GS、DropGaussian、DNGaussian、FSGS等基线对比，在PSNR、SSIM和LPIPS上均优于现有方法，显著提升稀疏视角生成质量与稳定性。

**⚠️ 局限性**

仅对位置参数的扰动最有效，其他参数的扰动效果有限；需手动调节噪声比例与重初始化间隔；在极低视角下仍存在细节缺失。

---

## 411. EFlow: Learning Evidence Flow for Long-Video Reasoning with Adaptive Reflection

**arXiv ID:** 2607.00867 | [PDF](https://arxiv.org/pdf/2607.00867v1)

**作者:** Wenhao Zhang `[一作]` (Peking University), Ge Li `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出EFlow框架，在长视频问答中通过先证据检索再推理的三阶段流程解决早期语义承诺问题，并加入置信度反射机制。

**💡 创新点**

明确分离时序证据定位（T‑CoT）和逻辑推理（R‑CoT），以及基于答案概率差的置信度反射机制；使用三阶段训练（SFT+RL+RFT）构建可迁移证据流。

**🔧 技术方法**

基于Qwen3‑VL的多模态大模型，链式思考(CoT)与工具调用、裁剪工具、Group Relative Policy Optimization (GRPO)等强化学习奖励（IOU、答案、格式），置信度阈值反射机制。

**📊 数据集**

EFlow‑SFT‑50K（自生成），EFlow‑RL‑10K（VideoITG‑40K + LLaVA‑Video‑178K），EFlow‑RFT‑10K（RL高奖励轨迹）以及在VideoMME、VideoMME Long、LVBench、LongVideoBench、NextGQA等评测基准上评估。

**📈 对比分析**

与多种单轮、工具驱动、多轮、原生工具调用模型对比，EFlow在VideoMME整体69.1%、Long 60.1%、LVBench 52.5%、LongVideoBench 60.3%、NextGQA 80.0，均超越基线。

**⚠️ 局限性**

依赖底层模型对时序证据的定位能力，若定位不佳会导致错误传播；置信度反射依赖答案置信度的校准，可能需任务特定阈值；在弱模型或不同骨干上迁移效果未知。

---

## 412. Constrained Bayesian Optimisation with Multiple Information Sources

**arXiv ID:** 2607.00865 | [PDF](https://arxiv.org/pdf/2607.00865v1)

**作者:** Hauke Maathuis `[一作]` (TU Delft), Maike Osborne `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了多信息源受约束贝叶斯优化问题，并提出了 MS-CMES（Multi-Source Constrained Max‑Value Entropy Search）框架，能够在有限预算内高效发现可行区域并逼近最优解。

**💡 创新点**

创新点在于：①将信息论最大熵搜索与多源高斯过程相结合，显式建模源间相关性；②引入方差校正机制自动抑制无信息源的误导；③采用自适应信任域约束采样搜索空间，并通过成本归一化实现多源均衡利用；这些组合首次实现了同时处理约束与多源的贝叶斯优化。

**🔧 技术方法**

使用的技术包括：多源 MISO 高斯过程模型、Max‑Value Entropy Search、KL 下界的变分近似、方差校正、信任域 (Trust‑Region) 优化、Monte‑Carlo 采样估计熵、成本模型 λ(x,ℓ)，以及对约束的可行性概率计算。

**📊 数据集**

实验数据集主要为五个约束基准：Physics‑based Pressure Vessel、Rosenbrock（带两约束）、Rastrigin（标准与旋转版）、Different Powers 函数等，维度范围从 4 到 100；此外还使用了合成的弱相关噪声数据作为辅助源。

**📈 对比分析**

与 SCBO、FuRBO、VBO+LogCEI、CMES‑IBO+、CMFBO 及随机搜索等方法对比，MS‑CMES 在低评估预算（≤200 次目标源评估）下，在高维问题中能够最快发现可行解并收敛到更优目标值，整体性能明显优于所有基线。

**⚠️ 局限性**

限制：①假设目标与所有约束在同一源一起评估，无法处理部分输出（仅目标或仅约束）测量；②要求每个约束都有对应的辅助源；③性能对成本函数 λ(x,ℓ) 的准确性敏感，若成本估计不准可能导致资源分配失衡。

---

## 413. Stitched Embeddings: A Unified Latent Space for 3D Garments and 2D Patterns

**arXiv ID:** 2607.00829 | [PDF](https://arxiv.org/pdf/2607.00829v1)

**作者:** Andrea Sanchietti `[一作]` (University of Tübingen), Gerard Pons-Moll `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本论文提出 Stitched Embeddings（StEm）框架，建立统一的潜在空间同时编码 3D 服装与对应的 2D 缝纫图案，支持端到端可微的预测、测试时优化和 2D→3D、3D→2D 的双向编辑。

**💡 创新点**

创新点在于：① 通过 BoxMesh 把 2D 缝纫图案映射到 3D 预模拟结构，构建 3D UDF 与 2D 参数的共享潜在空间；② 全流程无需物理仿真，直接在潜在空间做梯度优化；③ 通过测试时潜在优化实现 3D 服装与 2D 图案的互相纠错与实时编辑。

**🔧 技术方法**

核心技术包括：Hunyuan3D VAE（使用 UDF 作为连续 3D 表示）、BoxMesh 作为 3D 预模拟桥梁、双编码器（点云 + BoxMesh）到共享潜在空间、解码 Transformer 生成 UDF 与 122 维缝纫参数、KL 对齐与重建损失、梯度基测试时潜在优化。

**📊 数据集**

使用的数据集包括：GarmentCodeData、GarmentData、4DDress、CloSe 等，涵盖合成 3D 服装、真实扫描及图像数据。

**📈 对比分析**

与 NeuralTailor、ChatGarment 等基线对比，StEm 在 3D 点云到缝纫图案的 L2 误差、边缘/缝合精度上提升约 15% 以上；在真实扫描上 Chamfer 距离显著降低，性能优于现有方法。

**⚠️ 局限性**

局限性：目前仅支持单层服装，难以处理多层、不同材质或细节丰富的服装；对极端姿态或高噪声输入的鲁棒性仍有限，且需更多高质量数据进行进一步泛化。

---

## 414. Exploring the Semantic Gap in Agentic Data Systems: A Formative Study of Operationalization Failures in Analytical Workflows

**arXiv ID:** 2607.00828 | [PDF](https://arxiv.org/pdf/2607.00828v1)

**作者:** Jalal Mahmud `[一作]` (Megagon Labs), Eser Kandogan `[通讯]` (Megagon Labs)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了使用大语言模型生成分析工作流时的操作化失败，收集并分析了236条跨金融、人力资源与公共安全领域的分析意图，归纳了153个失败案例并提出了五类常见失败类型。

**💡 创新点**

首次系统性量化并分类分析agent生成工作流中的语义操作化失败，揭示了分析概念与可执行计算之间的语义鸿沟，并提出需要更丰富的语义表示来保证操作可接受性。

**🔧 技术方法**

采用基于GPT‑4o的NL2SQL代理（Blue平台），结合数据库模式、属性描述、统计信息和样本值进行查询生成，并通过人工审查判断是否符合用户意图。

**📊 数据集**

使用了BIRD金融基准数据库、NYC机动车碰撞公开数据集以及匿名企业人力资源数据集。

**📈 对比分析**

通过与BIRD MiniDev与BIRD挑战集的两份独立工作负载对照，发现61%和68%的意图在独立数据上也出现操作化失败，表明失败率高达约65%，说明当前系统在语义正确性方面表现较差。

**⚠️ 局限性**

局限在于仅评估单一NL2SQL架构与GPT‑4o模型，缺乏对不同模型、不同语义资源的泛化验证；未给出具体改进方案，仅指出需要更丰富的语义表示。

---

## 415. LRAT-Catcher: Importing SAT Solver Certificates into Lean4 by Reflection

**arXiv ID:** 2607.00815 | [PDF](https://arxiv.org/pdf/2607.00815v1)

**作者:** Stefan Szeider `[一作]` `[通讯]` (TU Wien), Stefan Szeider (TU Wien)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套工具，能将 SAT 证明者输出的 DIMACS 公式与 LRT 证书直接导入 Lean 4，并在 Lean 内部通过反射检查，生成可重用的定理。

**💡 创新点**

创新点在于：①利用 Lean 核心已验证的 LRAT 检查器实现完整 RAT 规则支持；②提供两种反射模式（原生编译检查和仅内核检查），可在信任基准与性能之间自由切换；③通过检查覆盖完整性证书，完成 cube‑and‑conquer 结果在 Lean 内部的合成，避免额外的可信组合器；④构建开源工具链，可在任意 DIMACS 公式和证书上使用。

**🔧 技术方法**

技术手段包括 Lean 4 的反射机制（原生和内核两种实现）、LRAT 检查器、文件解析器、并行叶子检查器、覆盖完整性检查与编码的可靠性证明。

**📊 数据集**

实验使用了 Schur 号（S(4)=44）和 Ramsey 号（R(4,4)=18）的编码，及逐步扩大的鸠形公式（pigeonhole）作为规模压力测试。

**📈 对比分析**

对比实验显示：①显式证书导入（lrat_proof）在大证书时内存消耗极高；②外部检查器最快但仅返回判决；③原生反射模式在内存与时间上均表现最佳；④仅内核模式虽然信任基准更小，但对大证书慢且易超时。

**⚠️ 局限性**

限制在于：①原生反射仍需信任 Lean 编译器；②仅内核模式适用于小证书；③大规模证书仍需手工拆分或使用 cube‑and‑conquer，且整体构建耗时与内存仍高。

---

## 416. Spotted: Location-informed Reidentification of Hyenas and Leopards in Camera Trap Surveys

**arXiv ID:** 2607.00804 | [PDF](https://arxiv.org/pdf/2607.00804v1)

**作者:** Halil Sina Kelebek `[一作]` (University of Oxford), Daniele De Martini `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该论文提出了一种基于摄像机位置与时间戳的可行性评分与视觉特征融合的动物重识别框架，并通过主动配对采样实现人机交互标注效率提升。

**💡 创新点**

创新点在于将连续速度可行性评分作为弱监督伪标签，使用轻量MLP对冻结骨干进行适配，并在主动采样中加入置信度采样，显著减少人工标注量。

**🔧 技术方法**

技术包括：冻结多物种ReID骨干（MegaDescriptor、MiewID）、轻量MLP头、对比/三元组损失、基于DBSCAN的聚类与ConfSPAL采样、以及图像直方图匹配和分割预处理。

**📊 数据集**

使用了三组在津巴布韦完成的摄像机捕捉ReID数据集：LeopardID102（102只豹子，717张图），SpottedHyenaID109（109只鬣狗，704张图），SpottedHyenaID415（415只鬣狗，1871张图）。

**📈 对比分析**

与HotSpotter、MegaDescriptor、MiewID等基线相比，top‑5准确率提升28/24/31个百分点；主动采样可将标注查询量降低69%（豹）/42%（鬣狗），在大规模数据集上发现更多正匹配，整体性能显著优于传统方法。

**⚠️ 局限性**

局限在于对摄像机位置与时间戳的依赖，部署稀疏或位置不准会影响效果；伪标签的弱正样本可能导致过度保守；在更大种群或跨域迁移场景下仍需进一步验证。

---

## 417. Beyond Line of Sight: Hybrid Validation of V2X Collective Perception in Complex Scenarios

**arXiv ID:** 2607.00874 | [PDF](https://arxiv.org/pdf/2607.00874v1)

**作者:** Markos Antonopoulos `[一作]` (Institute of Communication and Computer Systems), Angelos Amditis `[通讯]` (Institute of Communication and Computer Systems)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于贝叶斯融合的V2X集体感知框架，并在CARLA+车辆现场混合验证环境中实现了其可重现性和可解释性。

**💡 创新点**

创新点在于将多车异构传感器观测整合进共享的概率占据网格，结合实时可靠性评估和混合验证，突破了传统单车感知的视野与不确定性限制。

**🔧 技术方法**

采用粒子滤波定位、GPU实现的视野估计、贝叶斯占据滤波、CARLA仿真与ROS2/MathWorks混合测试，形成端到端的概率感知链路。

**📊 数据集**

主要使用CARLA仿真生成的环形交叉口场景与现场车辆实验数据；未引入公开的感知数据集，而是通过自建混合实验实现评估。

**📈 对比分析**

通过比较单车感知、3车CP和6车CP三种配置，利用FoV覆盖率、占据细胞召回率与精确率以及AUC指标，实验显示覆盖率提升约260%，召回率从0.82提升到0.94，且对定位噪声具有良好鲁棒性。

**⚠️ 局限性**

局限性包括忽略V2X通信延迟与拥塞影响、仅在单一环形交叉口场景验证、车数有限、未实现动态贝叶斯更新或可信度加权等后续可改进方向。

---

## 418. Dynamic Bidirectional Pattern Memory: A Production-Scale Empirical Characterisation of Inference-Time Gating in Clinical NLP

**arXiv ID:** 2607.00870 | [PDF](https://arxiv.org/pdf/2607.00870v1)

**作者:** Ali H. Lazem `[一作]` (Bangor University), William Teahan `[通讯]` (University of Thi-Qar)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在生成‑验证器的临床 NLP 管道中构建并部署了动态双向模式记忆 (DBPM)，对生成器提出的候选进行推理时门控，以减少后续验证器的调用。

**💡 创新点**

首次证明了自然的验证器投递拒绝记录的记忆渠道在大规模生产环境下会因信号稀疏导致失效，并展示了通过语义本体判定与实体覆盖交叉检查两种可选信号实现门控选择性的经验规律。

**🔧 技术方法**

核心技术包括基于签名的可持久化块/白名单记忆、时间衰减与阈值门控、跨任务传播、以及规则/本体检测的无训练增量推理。

**📊 数据集**

使用了 167,034 篇 PMC‑Patients 临床案例文本的多任务版本作为数据集，并在 5,000 个留存样本上进行对比消融实验。

**📈 对比分析**

通过对比开启/关闭门控的 A/B 测试，评估了门控在 5,000 人样本上的标记提升（lift 1.84）、对验证器拒绝率的提升和对运行时间的影响，显示门控对吞吐量几乎无负担，但在特定实体覆盖范围内才具选择性。

**⚠️ 局限性**

局限性包括：门控仅在实体覆盖度处于中等范围时才有效；对验证器未使用的本体违规信号仍需人工维护；以及跨任务传播虽然稳健但对极端稀疏场景的提升有限。

---

## 419. Visualizing Engineering Fundamentals: Design of Mixed Reality and Physical Toolkits for Effective Learning

**arXiv ID:** 2607.00979 | [PDF](https://arxiv.org/pdf/2607.00979v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 420. Practical Range Refinement Types with Inference

**arXiv ID:** 2607.00824 | [PDF](https://arxiv.org/pdf/2607.00824v1)

**作者:** Valentin Aebi `[一作]` (Università della Svizzera italiana), Carlo A. Furia `[通讯]` (Università della Svizzera italiana)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出了一套针对整数范围的实用细化类型系统，并在实验语言中实现了该系统，用以在保持低注解负担的前提下对程序中与范围相关的错误进行静态检测。

**💡 创新点**

创新点包括：① 将细化类型与传统命令式语言结构无缝结合；② 采用双向类型检查与后向类型候选传播，显著减少手写注解；③ 通过单调性分析与智能类型转换实现对循环变量的精准范围推断；④ 引入混合强制转换操作，兼顾可执行性与可验证性；⑤ 在实验平台上以量化实验验证了其在表达力、简洁度、可靠性和性能方面的优势。

**🔧 技术方法**

所用技术包括：SSA形式的中间表示；双向类型检查算法（前向与后向传播）；单调性分析（利用SMT求解器判断循环变量的单调性）；智能类型转换（基于路径条件的智能强制）；SMT求解器Z3进行约束求解；e‑graph技术用于消除重复类型检查；以及对纯度分析和泛型约束的处理。

**📊 数据集**

实验数据集由14个示例程序组成，分为5组（rng、amap、sort、ic、liq），覆盖数组索引、集合操作、排序、过滤等常见用例；每个程序包含若干函数/方法，作为评估基准。

**📈 对比分析**

比较方法：① 通过统计注解数量（标准类型、细化类型、辅助注解、强制转换）评估注解负担；② 计算可表达约束比例、注解密度、准确率、精确度和正确性；③ 对三种工具（自研细化类型、Checker Framework、LiquidJava）以及Scala进行对比；性能测试显示自研系统在同一组示例上的类型检查耗时为16秒，显著低于Scala 3编译器的34秒。

**⚠️ 局限性**

局限性：① 仅在实验语言实现，尚未迁移至成熟语言；② 仅支持整数范围的细化类型，未覆盖更广泛的谓词；③ 候选类型推断采用单遍不收敛，可能导致不安全的假设；④ 在某些复杂的Java特性（如Stream、泛型）上，LiquidJava及Checker Framework表现不佳；⑤ 实验规模有限，未覆盖大型真实项目。

---

## 421. Automatic Detection of Stress from Speech in the Trier Social Stress Test

**arXiv ID:** 2607.00986 | [PDF](https://arxiv.org/pdf/2607.00986v1)

**作者:** Hanna Drimalla `[一作]` (Bielefeld University), Oliver T. Wolf `[通讯]` (Ruhr University Bochum)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过收集实验参与者在TSST（社会性压力）与对照条件下的语音，提取多种声学特征，并使用机器学习模型实现压力检测及预测生理（唾液皮质醇、唾液α‑淀粉酶）和情绪（PANAS）压力反应。

**💡 创新点**

在完全被试间设计中首次结合TSST对照实验与自动语音分离，验证语音可作为多维度压力的数字生物标志，并通过特征重要性揭示对压力检测最具预测力的声学指标。

**🔧 技术方法**

采用Sortformer语音分离、librosa、Praat、openSMILE声学特征提取；机器学习分类器（Logistic Regression、SVM、Random Forest、XGBoost）和回归模型（SVR、RF、XGBoost），并用交叉验证与SHAP进行特征重要性评估。

**📊 数据集**

使用50名德国大学生的数据集，包含TSST与对照条件下的音频、唾液皮质醇、唾液α‑淀粉酶及PANAS情绪问卷。

**📈 对比分析**

采用留一交叉验证的嵌套 CV，评估分类准确率与 ROC AUC，回归用 MAE 与 Spearman 相关系数。分类中XGBoost达到82%准确率；回归中XGBoost在皮质醇反应与负面情绪变化上显著优于基线。

**⚠️ 局限性**

样本量有限，控制组可能产生轻微压力，+20min皮质醇预测效果不稳定；未充分利用时间序列信息，模型对录音设备与环境噪声的鲁棒性有限。

---

## 422. TRCGL-Net: A Long-Tailed Multi-Label Chest X-Ray Classification Framework with Generative Data Augmentation and Label Co-Occurrence Modeling

**arXiv ID:** 2607.00975 | [PDF](https://arxiv.org/pdf/2607.00975v1)

**作者:** Tong Shao `[一作]` (South-Central Minzu University), Fang Wang `[通讯]` (South-Central Minzu University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出TRCGL-Net框架，用于解决胸部X光多标签分类中的长尾不平衡问题。

**💡 创新点**

创新点包括：①基于可学习文本提示的条件扩散模型，生成语义一致的稀缺病种图像；②通道重加权与类别感知注意力的特征增强模块；③利用标签共现图卷积模型，挖掘疾病间依赖关系。

**🔧 技术方法**

核心技术包括：文本引导的可学习扩散生成、ConvNeXtV2特征提取、通道加权与类别注意力机制、基于共现矩阵的图卷积网络。

**📊 数据集**

使用公开PadChest胸部X光图像数据集进行训练与评估，数据按病种划分为头部、中等与尾部三类。

**📈 对比分析**

与ResNet、EfficientNet、ViT、CLIP、ConvNeXt+LDAM/CSRA等基线及最新方法对比，TRCGL-Net在整体mAP、mAUC及宏F1上均明显提升；尾部类别mAP提升至0.4904，mAUC升至0.9229，优于最佳基线。

**⚠️ 局限性**

局限性在于：①扩散模型的语义提示仍受限于疾病名称与报告文本，难以处理复杂多病变描述；②共现图结构依赖训练集统计，可能随机构差异变化导致迁移性能下降。

---

## 423. QuaMoE-DRF: Proactive Beam and Rate Adaptation via Multimodal Dynamic Radio Map Forecasting in ISAC Networks

**arXiv ID:** 2607.00974 | [PDF](https://arxiv.org/pdf/2607.00974v1)

**作者:** Zhihan Zeng `[一作]` (University of Electronic Science and Technology of China), Chongwen Huang `[通讯]` (Zhejiang University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了QuaMoE-DRF框架，利用多模态动态射频图预测实现ISAC网络中的主动波束与速率自适应。

**💡 创新点**

创新点包括：① 将未来波束‑SINR场视为通信足够的中间表征；② 设计质量感知混合专家（QuaMoE）实现异方差模态融合；③ 结合紧凑的参考BS投影与BS/波束/ MCS监督，避免仅靠单一映射进行BS关联。

**🔧 技术方法**

采用CNN、GRU、MLP编码器，质量感知混合专家网络，线性回归解码器，以及联合MAP与决策损失训练；理论上依据逆方差融合、边际损失与误差上界分析。

**📊 数据集**

在由城市道路场景生成的动态ISAC基准数据集上评估，该数据集包含多BS、多UE、移动阻挡、事件、传感与无线历史四类模态，约13.2k样本。

**📈 对比分析**

与CKM-UNet、Radar-EKF BF、RM+RP等三类基线对比，QuaMoE-DRF在有效率上提升5.67%、出错率降低8.35%，Beam@3与连通Beam准确率均达1，RMSE最低并具备更好的鲁棒性。

**⚠️ 局限性**

局限性包括：① 仅使用内部模拟器生成标签，缺乏大规模射线追踪或实测验证；② 计算复杂度相对较高；③ 在未见布局或极端阻挡情形下性能仍有下降。

---

## 424. Bayesian Uncertainty Propagation for Agentic RAG Pipelines: A Proof-of-Concept Study on Multi-Hop Question Answering

**arXiv ID:** 2607.00972 | [PDF](https://arxiv.org/pdf/2607.00972v1)

**作者:** Louis Donaldson `[一作]` (University of Hull), Yiannis Papadopoulos `[通讯]` (University of Hull)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了轻量级的Agentic RAG系统不确定性监测框架，使用贝叶斯网络聚合Planner、Evaluator、Generator的语义偏差与自评不确定性信号，评估多跳问答任务的系统级置信度。

**💡 创新点**

将Wasserstein语义偏差与P(True)自评作为节点不确定性输入，并通过贝叶斯网络实现不确定性传播，提供节点级失败指示，展示在多跳推理中更有效的置信度估计。

**🔧 技术方法**

贝叶斯网络 (BN)、语义偏差计算、Token级熵、P(True)自评、GPT‑3.5‑Turbo/ GPT‑4.1‑Nano LLM、AUROC/AUARC/ECE/BrierScore评估。

**📊 数据集**

StrategyQA 与 HotpotQA 两个公开多跳问答基准。

**📈 对比分析**

与UProp基准以及各节点单独的AUROC/AEARC/ECE/BrierScore对比，结果显示在HotpotQA上BN整体置信度优于UProp，并在多跳推理中表现更佳；但在StrategyQA上仅Generator信号最强，BN整体低于UProp。

**⚠️ 局限性**

使用的确定性OR门无法自适应节点可靠性；P(True)自评系统性保守导致校准差；实验仅在公开问答数据集，缺乏工业域检验。

---

## 425. Svarna: An Open Corpus Workbench for Modern Greek

**arXiv ID:** 2607.00970 | [PDF](https://arxiv.org/pdf/2607.00970v1)

**作者:** Stergios Chatzikyriakidis `[一作]` `[通讯]` (University of Gothenburg), Stergios Chatzikyriakidis (University of Gothenburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Svarna，一个免费的、开源的基于 Web 的现代希腊语语料库工作台，整合了超过 5 亿词的多域语料并提供共现、词频、话语标记等分析功能。

**💡 创新点**

将散落且受限的希腊语语料统一到单一无登录、无安装的接口，并以 FTS5 轻量索引实现实时查询和多维度分析，首次提供跨域免费可搜索的希腊语语料平台。

**🔧 技术方法**

采用 FastAPI+Python 后端、SQLite FTS5 索引、aiosqlite 异步查询、Chart.js 前端、Docker+Azure Container Apps 部署，支持可选 LLM 调用。

**📊 数据集**

整合 Wikipedia、Parliament、CC‑100、Europarl、OpenSubtitles、Leipzig、UD treebanks、Project Gutenberg、BabyLM、Tesserae、GRDD+ 等多源语料，总计约 5.07 亿词。

**📈 对比分析**

利用 FTS5 的全文搜索实现快速 KWIC 与布尔查询；词频归一化与 MI 计算提供共现；对比两个注册使用 log‑ratio keyness，实验表明对中等规模查询响应时间在数百毫秒到数秒之间，适合交互式分析。

**⚠️ 局限性**

语料采集非平衡且受限于公开可用数据，统计基于查询样本而非全体语料，且需本地构建与部署，缺乏直接上传界面。

---

## 426. Tighter bounds for weighted and unweighted shortest cycle approximation

**arXiv ID:** 2607.00938 | [PDF](https://arxiv.org/pdf/2607.00938v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 427. GaussianEmoTalker: Real-Time Emotional Talking Head Synthesis with Audio-Driven and Blendshape-Based 3D Gaussian Splatting

**arXiv ID:** 2607.00959 | [PDF](https://arxiv.org/pdf/2607.00959v1)

**作者:** Haijie Yang `[一作]` (Nanjing University of Science and Technology), Jian Yang `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出GaussianEmoTalker框架，利用3D高斯散点实现音频驱动的实时情绪说话头生成与编辑。

**💡 创新点**

创新点包括：① 两阶段构建中性与情绪状态空间，利用GaussianBlendshapes初始化中性高斯；② 空间‑音频‑情绪跨注意机制预测高斯属性偏移，支持可调情绪强度；③ 结合预训练音频到表情网络与CLIP文本编码，实现情绪标签与强度控制。

**🔧 技术方法**

采用技术包括：3D Gaussian Splatting、GaussianBlendshapes、FLAME模型、音频到表情Transformer、空间‑音频‑情绪注意机制、CLIP文本编码、VGG特征损失、Flame损失等。

**📊 数据集**

使用MEAD情绪说话头数据集（8种情绪 × 3个强度级别），并在单身份视频上进行训练。

**📈 对比分析**

通过与SadTalker、EAT、EDTalk、GSBS、EmoTaG等基线在自驱和交叉驱设置下的对比，GaussianEmoTalker在视频质量（低FID/FVD、较高PSNR/SSIM）、唇同步（低LD/LMD、同步得分高）和FPS（约40帧/秒）方面均实现了最优或接近最优表现。

**⚠️ 局限性**

局限性：① 仅为单身份训练，缺乏跨身份泛化能力；② 只能处理离散情绪强度级别，无法实现平滑的连续强度过渡（如1.5、2.5）。

---

## 428. DeWorldSG: Depth-Aware 3D Semantic Scene Graph Generation via World-Model Priors

**arXiv ID:** 2607.00889 | [PDF](https://arxiv.org/pdf/2607.00889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 429. Dataset Biases and Shortcut Learning in Motion-Based AI-Generated Video Detection

**arXiv ID:** 2607.00948 | [PDF](https://arxiv.org/pdf/2607.00948v1)

**作者:** Joren Michels `[一作]` (Hasselt University), Nick Michiels `[通讯]` (Hasselt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过系统评估四种主流运动基 AI 生成视频检测器，揭示其在预处理与采样过程中存在显著偏差，并证明这些偏差是其高准确率的主要驱动因素。

**💡 创新点**

创新点在于首次将采样偏差、预处理失真以及数据集运动不平衡三类偏差与检测性能关联起来，并通过重新平衡和空间扰动实验直观展示这些检测器在无偏见条件下的极端退化；同时对比频率基检测器的稳健性，提供了更具普适性的检测思路。

**🔧 技术方法**

研究采用了采样重构、JPEG/压缩分析、FFT 频谱对比、MMD 统计、以及对预训练模型（如 DINOv2、CLIP）提取特征的方式，并在此基础上重新训练与调参，进一步对检测器做了再评估。

**📊 数据集**

主要使用的公开数据集包括 GenVideo、GenBuster++、WaveRep、Over-Coherence、VidProM 等，且对 GenVideo 的真实视频采用高帧率版本，对 VidProM 采用原始测试集，所有视频统一按 8 FPS、2 秒时长采样后处理。

**📈 对比分析**

实验表明，在原始偏差数据集上 D3、ReStraV、Over‑Coherence 等运动基检测器的 AUC 可达 0.97‑0.99；但在去偏后（重平衡或高帧率真实视频）以及对单帧做 7×7 Gaussian blur 攻击时，AUC 直接跌至 0.5 甚至更低；相比之下，WaveRep 频率基检测器在所有评测集上均保持 0.97‑0.99 的高水平，未出现性能崩塌。

**⚠️ 局限性**

局限性包括：运动基检测器极易利用采样与预处理的快捷路径，导致对真实运动分布的依赖；在面对新型生成模型、不同分辨率或帧率变化时易失效；此外本文仅评估了有限的运动基方法与数据集，未来需扩展到更多模型与更具代表性的评测场景。

---

## 430. Graph-Native Reinforcement Learning Enables Traceable Scientific Hypothesis Generation through Conceptual Recombination

**arXiv ID:** 2607.00924 | [PDF](https://arxiv.org/pdf/2607.00924v1)

**作者:** Subhadeep Pal `[一作]` (Massachusetts Institute of Technology), Markus J. Buehler `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研发了 Graph-PRefLexOR 系列模型，能够在材料设计等科学任务中通过多阶段图结构推理生成可追溯、可解释的假设，改进传统 LLM 的线性推理方式。

**💡 创新点**

创新点在于：① 将推理拆分为机制探索、关系草图、图构建、模式提取、假设综合五个显式阶段；② 采用 Group Relative Policy Optimization (GRPO) 进行强化学习，直接优化结构化推理的奖励；③ 设计了包含六项（正确性、格式、图效用、连通性、多样性、结构）奖励的复合分数，确保生成的图既可解析又富含知识。

**🔧 技术方法**

技术手段包括：图结构推理模板（sentinel‑based 5 阶段输出）、教师蒸馏生成训练样本、ORPO+GRPO 强化学习、基于网络图、embedding、语义相似度的奖励计算、LLM 判别（Claude Opus‑4.7）评估、以及多规模（1.7B、3B、8B）Qwen/Llama-3.2 基座。

**📊 数据集**

数据集：① 训练数据由通用与材料科学/力学混合语料（Web 文本 + 领域论文）构成，使用 GPT‑5.1 生成问题+完整图结构推理和答案，GPT‑5‑nano 生成无推理的拒绝答案；② 评估基准为 100 题开放式材料/力学问题，问题从 100 篇论文抽取并由 GPT‑5.4 生成与校正。训练集在 HuggingFace 上公开，基准亦可下载。

**📈 对比分析**

对比方法：在 100 题上使用 Claude‑Opus‑4.7 评估推理质量、知识深度、可追溯性；将 Graph-PRefLexOR 与对应基线（Qwen3‑8B、Llama‑3.2‑3B‑Instruct、无思考版）以及 1.7B/3B/8B 三个规模进行同等测试。结果显示 Graph-PRefLexOR 在所有三项指标上提升 40‑65%，尤其可追溯性提升最大；embedding 和语义多样性分析表明推理轨迹更广、方向更明确，最终答案的语义与推理路径更紧密。

**⚠️ 局限性**

限制与不足：① 图效用奖励仍低，生成图的语义完整性受限；② 需要大量推理时计算资源，尤其是 8B 版；③ 对跨模型可解释性验证不足（不同基座间的推理差异仍需深入研究）；④ 仅在材料/力学 100 题评估，缺乏更大、多学科验证；⑤ 现有模型对更复杂的高阶抽象关系捕捉有限，未来需进一步完善图结构与抽象机制。

---

## 431. Fundamental Limits of Random Downlink Integrated Sensing and Communication over Rician Channels

**arXiv ID:** 2607.00912 | [PDF](https://arxiv.org/pdf/2607.00912v1)

**作者:** Marziyeh Soltani `[一作]` (University College Dublin), Mark F. Flanagan `[通讯]` (University College Dublin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在 Rician 衰落环境下的下行多天线集成感知与通信（ISAC）系统的随机性能，给出了通信失效率与基于克雷姆-拉罗边界（CRB）的感知失效率，并分析了两种波束成形策略（SJB 与 LB）的性能。

**💡 创新点**

创新点包括：①将传统 Rayleigh 模型推广到 Rician 并考虑用户与目标角度的任意联合分布；②在 SJB 与 LB 下推导出闭式失效率表达式、上/下界与可计算近似；③给出大系统和高功率下的极限缩放律，揭示 DPC 对 LB 性能的决定性作用；④通过 Pareto 边界描绘感知‑通信的根本权衡。

**🔧 技术方法**

主要技术手段为多维中心极限定理（CLT）与高斯逼近、CRB 解析、概率分布积分、上/下界推导（如 Cauchy–Schwarz、二项式变换）以及大规模/高功率极限分析。

**📊 数据集**

实验验证使用 Monte‑Carlo 仿真，参数设置为 N=15、M=17、总功率 p_t=10，噪声 σ_u^2=σ_r^2=1，帧长 L=30，角度在 [0,π] 之间独立均匀分布，考虑三种 Rician K 值（0、0.32、3.1），未使用公开数据集。

**📈 对比分析**

通过绘制通信失效率、感知失效率随阈值变化的曲线以及 Pareto 失效率区域进行比较；结果显示 LB+DPC 在 Rayleigh 与强 LoS 情况下实现了 10⁻⁶ 级别的可靠性；LB（无 DPC）在高功率下受雷达自干扰限制；SJB 在中等至强 LoS 下稳健且实现了 1/N 的感知缩放；LB 在大规模天线下实现 1/N² 的更快感知缩放。

**⚠️ 局限性**

局限性包括：假设完美瞬时 CSI 与特定波形结构；高斯逼近在天线数较小或极端角度分布时可能失效；只考虑单用户与单点目标；未讨论实际硬件实现复杂度与非理想效应（如相位误差、功率校准）。

---

## 432. SenseWalk: Agent-Based Semantic Trajectory Simulation Powered by Large Language Models in Zoned Environments

**arXiv ID:** 2607.00989 | [PDF](https://arxiv.org/pdf/2607.00989v1)

**作者:** Ziyue Lin `[一作]` (Fudan University), Siming Chen `[通讯]` (Fudan University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套名为SenseWalk的交互式系统，利用大型语言模型（LLM）驱动的智能体在分区环境中生成语义轨迹，并通过社交力模型保证物理可行性，支持用户可视化配置场景、生成计划、实时模拟并查看推理过程；

**💡 创新点**

1) 将LLM与物理仿真（社交力模型）相结合，形成一种混合推理框架，兼顾语义连贯性与空间约束；2) 提供图形化的场景与代理配置界面，使非技术背景的从业者可轻松构造多样化情境；3) 将代理的决策过程可视化为对话式推理日志，提升可解释性与信任度；

**🔧 技术方法**

核心技术包括：大语言模型（DeepSeek、Qwen、Gemini）进行计划与行为生成；社交力模型实现运动动力学；多模块协同（计划、感知、记忆、推理、执行）；前端TypeScript实现交互式可视化；后端并行推理与模拟；

**📊 数据集**

四个真实场景地图：1) 旧金山CES（电子消费展） 2) 东京Shibuya PARCO（购物中心） 3) 伦敦National Gallery（艺术画廊） 4) 上海CIIE（进口博览会）。每个场景提供墙、ROI、事件三种元素的JSON描述，并生成15个不同访客画像；

**📈 对比分析**

对比方法：纯LLM直接生成轨迹的基线模型。评价指标为物理损失（轨迹是否越界或穿墙）与语义损失（与专家标注目标的精确度、召回率、F1）。结果显示SenseWalk在所有三种LLM下均显著降低物理损失（p<0.05）并提升F1分数，尤其召回率大幅提高；

**⚠️ 局限性**

限制主要有：1) LLM推理消耗高，导致实时交互延迟；2) 对于高密度区域仍可能出现穿墙或不合理路径；3) 需要手动构造ROI，未能自动从图像识别提取；4) 受限于单一文本输入，缺乏多模态交互；5) 实际验证的可信度仍需进一步提升，尤其在大规模宏观模拟中的适用性。

---

## 433. QCA: Query- and Content-Aware Keyframe Selection for Long Video Understanding

**arXiv ID:** 2607.00983 | [PDF](https://arxiv.org/pdf/2607.00983v1)

**作者:** Jun Peng `[一作]` (Xiamen University), Yonghong Tian `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 QCA 框架，针对长视频在有限帧预算下的理解任务，通过查询与内容感知的关键帧选择实现信息提取。

**💡 创新点**

创新点在于联合建模查询相关性与内容偏差，动态分配帧预算，并在每段内以语义锚点为中心进行多样性贪心选帧，且不需要额外训练。

**🔧 技术方法**

主要技术包括视频分段、图文匹配（BLIP‑2）、内容偏差度量、softmax 权重分配、语义锚点候选集构建与欧氏距离多样性选择。

**📊 数据集**

实验使用 LongVideoBench、Video‑MME、MLVU、LVBench 四大长视频理解基准。

**📈 对比分析**

与统一采样、Top‑k、AKS、Q‑Frame、OneClip‑RAG、BOLT、FRAG、E‑VRAG 等基线比较，在多款 Video‑LLM 上取得显著提升，例如 Qwen3‑VL‑8B 64 帧下 LongVideoBench 66.9% 以上，平均提升 3–4%。

**⚠️ 局限性**

局限性在于使用固定分段策略，对极短或极长视频可能欠缺自适应；且仍需在预处理阶段进行 FPS 降采样，无法完全消除超长视频的冗余。

---

## 434. LeNEPA: No-Augmentation Next-Latent Prediction for Time-Series Representation Learning

**arXiv ID:** 2607.00958 | [PDF](https://arxiv.org/pdf/2607.00958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 435. Understanding How Humans Inject Knowledge into Machine Learning Workflows through Visual Analytics

**arXiv ID:** 2607.00969 | [PDF](https://arxiv.org/pdf/2607.00969v1)

**作者:** Yiwen Xing `[一作]` (University of Oxford), Min Chen `[通讯]` (University of Oxford)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对2016-2025年IEEE VIS会议中约184篇VIS4ML（即视觉分析在机器学习中的应用）论文进行了系统性调查与编码，构建了四维（ML特征、可视化、交互、动作）分类框架，并利用配对共现分析、主题建模与信息理论视角，识别并描述了人类知识注入机器学习工作流的主要路径与模式。

**💡 创新点**

创新点在于：①首次将人类知识注入过程建模为“知识注入路径”，系统捕捉从可视化到交互再到实际动作的完整闭环；②提出统一的四维编码方案，兼顾技术、任务、交互与动作，填补了先前仅关注可视化或交互细节的空白；③将信息理论成本收益分析与可视化模型构建框架相结合，提供了理论解释与实证支持；④为未来VIS4ML系统提出设计准则与研究挑战。

**🔧 技术方法**

技术手段包括：文献筛选与手工编码、基于共现的Sankey图可视化、非负矩阵分解（NMF）主题建模、交叉类别共现分析、信息理论成本收益公式等。

**📊 数据集**

数据来源为IEEE VIS 2016-2025年会议论文，最终构成184篇可编码论文集；未使用传统机器学习数据集，而是对论文内容本身进行结构化编码。

**📈 对比分析**

本文并未对算法性能做实验比较，而是通过统计比例、共现频率、主题词云等方式对不同知识注入路径进行“性能”评价；大部分系统（≈80%）支持开发工作流驱动，约60%支持具体动作，说明可视化在提升人机协同效果方面具有普遍可行性。

**⚠️ 局限性**

局限性包括：①仅覆盖IEEE VIS会议，可能遗漏其他顶级VIS或ML会议的相关工作；②调查依赖人工编码，可能存在主观判断偏差；③缺乏对实际系统使用效果的用户实验与定量评估；④对“无动作”系统的后续知识转化机制未深入探究。

---

## 436. Quantifying the Affective Gap: A Zero-Shot Evaluation of LLMs on Fine-Grained Emotion Taxonomies

**arXiv ID:** 2607.00968 | [PDF](https://arxiv.org/pdf/2607.00968v1)

**作者:** Lawrence Obiuwevwi `[一作]` (Old Dominion University), Sampath Jayarathna `[通讯]` (Old Dominion University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Claude、ChatGPT（GPT‑5.4）与Gemini在13类情绪细粒度分类任务上进行零样本评估，使用统一无示例提示并通过生产API进行推理。

**💡 创新点**

首次在同一实验框架下直接对三大主流LLM进行零样本细粒度情绪识别比较，并剖析各类情绪的优势与失败模式。

**🔧 技术方法**

利用单词级返回的零样本推理技术，结合统计学检验（McNemar）和宏F1、加权F1、Cohen κ 等指标进行评估。

**📊 数据集**

使用boltuix/emotions‑dataset（131,306句子，13情绪标签）中的1,000句分层抽样样本。

**📈 对比分析**

三模型在准确率上相差约1.9个百分点（Gemini最高39.9%，GPT‑5.4 38.8%，Claude 38.0%），宏F1呈显著差距（Gemini 0.363 vs Claude 0.159），但McNemar检验未出现统计显著差异，说明它们达成了同一零样本上限。

**⚠️ 局限性**

缺乏少样本或链式推理提升，单一英语数据集限制泛化，罕见情绪类别样本稀少导致估计不稳，且零样本性能远低于人类基准。

---

## 437. DRL-Based Joint Beamforming and Surface Shape Optimization for Flexible Intelligent Metasurface-Aided ISAC Systems

**arXiv ID:** 2607.00951 | [PDF](https://arxiv.org/pdf/2607.00951v1)

**作者:** Maoyuan Wang `[一作]` (Shandong University), Deqiang Wang `[通讯]` (Shandong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了利用可柔性智能超表面（FIM）辅助的集成感知与通信（ISAC）系统，提出了在满足功率、服务质量（QoS）和表面形变约束的前提下，联合优化波束成形矩阵与FIM表面三维形变，以最小化方向估计的CRB。

**💡 创新点**

创新点在于首次将FIM的可变形表面作为额外自由度融入ISAC系统设计，并通过深度确定性策略梯度（DDPG）算法构建约束感知奖励，使得在非凸优化问题下实现CRB的显著下降。

**🔧 技术方法**

技术方法包括：1）建立FIM表面形变与波束成形相互作用的系统模型；2）推导CRB矩阵并设定最小化目标；3）使用DDPG深度强化学习框架，设计状态、动作空间及约束感知奖励；4）采用演员-评论家网络进行训练并实现多目标约束。

**📊 数据集**

使用仿真数据，设置发射FIM 9个天线、接收FIM 12个天线，天线间距0.5λ，形变范围±λ，噪声功率-100 dBm（用户）和-95 dBm（雷达），在不同用户数、功率上限、QoS阈值下进行性能评估。

**📈 对比分析**

与传统刚性阵列（RA）基线、单侧FIM形变基线以及仅对发送或接收FIM形变的基线进行比较。实验结果表明，联合优化发送与接收FIM形变的方案在相同功率和QoS约束下，CRB显著低于其他基线，且随用户数、功率上限或QoS阈值变化表现稳定。

**⚠️ 局限性**

局限性包括：1）仅考虑单频窄带场景，未扩展到宽带或多频段；2）仿真模型假设理想的FIM形变控制与无环境干扰；3）缺乏实际硬件验证，实际制造误差与形变可实现度未充分评估。

---

## 438. Leveraging LLM-Based Agentic Systems to Generate Quantum Applications for Test Optimization

**arXiv ID:** 2607.00939 | [PDF](https://arxiv.org/pdf/2607.00939v1)

**作者:** Ming Tao `[一作]` (Beihang University), Aitor Arrieta Marcos `[通讯]` (Mondragon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种基于大语言模型的多智能体架构，能够把软件工程优化任务的自然语言需求自动转换为可执行的量子应用，并记录完整的生成流程。

**💡 创新点**

创新点在于将需求解析、量子适配、工作流规划、编码、代码生成、审核、执行和验证等步骤拆分为专用智能体，并通过反馈循环和可追溯的中间产物实现端到端自动化。

**🔧 技术方法**

使用技术包括大语言模型（Claude、DeepSeek、Llama）、专用智能体、工具调用、量子编程（QUBO、QAOA、量子退火）、代码审核与修复、量子模拟/硬件执行、结果验证与组合。

**📊 数据集**

使用了 20 条自然语言需求，每条对应一个真实世界的量子优化基准实例（10 个测试用例最小化/选择基准），从而形成完整的需求–基准对。

**📈 对比分析**

通过与传统遗传算法基线对比、消融实验（去除代码生成技能、任务知识、审核反馈、多智能体拆分）以及不同 LLM 主干的比较，评估了系统的编译率（100%）、执行率（96.7%）和解决方案质量（大多超过遗传算法）。

**⚠️ 局限性**

局限性包括高昂的生成成本（平均 260 秒、1.89M 令牌）、对资源的高需求、主要验证于 TCM/TCS 两个任务，未证明能广泛适用于其他 SE 量子优化问题，以及对量子硬件执行的实验仍待进一步验证。

---

## 439. Explainable AI for Cancer Drug Response Prediction: Beyond Univariate Feature Attributions

**arXiv ID:** 2607.00931 | [PDF](https://arxiv.org/pdf/2607.00931v1)

**作者:** Martino Ciaperoni `[一作]` (Scuola Normale Superiore), Fosca Giannotti `[通讯]` (Scuola Normale Superiore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个可扩展的后置可解释框架 ILLUME+，并搭建完整端到端流水线，用转录组数据预测癌症药物反应，同时生成多模态解释（基因重要性、决策规则和基因‑基因相互作用），从而为 AI 辅助的生物学假设生成提供支持。

**💡 创新点**

① 通过低秩分解和随机雅可比逼近显著降低 ILLUME 的参数量和计算成本，使其适用于成千上万个基因的高维转录组；② 采用多模态解释方法（多基因重要性、规则、交互），摆脱单基因 SHAP 只给出的弱化解释；③ 采用完全数据驱动的特征选择，避免先验知识带来的偏差。

**🔧 技术方法**

使用 ILLUME+（低秩 hypernetwork + 随机雅可比惩罚）训练 meta‑解释器；梯度提升树作为黑盒预测器；局部线性/决策树 surrogate 产生特征重要性和决策规则；基于 lift 的基因对交互排名；路径富集、中心性与导电率等网络分析来验证生物学意义。

**📊 数据集**

基于 GDSC 与 CCLE 的药物敏感性数据（≈18,000 基因表达），对 20 种已恢复目标的药物各约 500 个细胞系进行实验；IC₅₀ 通过 tertile 分箱转为三分类标签。

**📈 对比分析**

与 TreeSHAP（SHAP）、LIME 以及原始 ILLUME 进行对比。ILLUME+ 在目标恢复（NDCG@k）、途径富集、解释鲁棒性（余弦相似度）上均优于 SHAP，近似 LIME 的表现；同时显著降低内存占用和训练时间，保持解释质量。

**⚠️ 局限性**

仅基于细胞系公开数据，可能带来实验条件与组织差异导致的偏差；解释仅体现统计关联，需实验验证；低秩与随机雅可比近似可能略微降低解释精度；目前仅考虑二元基因交互，未覆盖更高阶相互作用。

---

## 440. Delta Debugging in the Absence of Test Oracles Through Metamorphic Testing

**arXiv ID:** 2607.00929 | [PDF](https://arxiv.org/pdf/2607.00929v1)

**作者:** Mingyue Jiang `[一作]` (Zhejiang Sci-Tech University), Tsong Yueh Chen `[通讯]` (Swinburne University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将变异测试（Metamorphic Testing）集成到Delta Debugging 的算法 DDMT，能够在没有测试 oracle 的情况下实现输入最小化。

**💡 创新点**

创新点在于设计基于 metamorphic relation 的 test 函数 mrtest，解决 oracle 问题并在多种场景下提升 DD 的效果；同时对 DDMT 与传统 ddmin、Perses 等方法进行了系统评估。

**🔧 技术方法**

使用了 Delta Debugging（ddmin）与 Metamorphic Testing 的结合、MR 构造与验证、统计评估（Wilcoxon、箱线图）等技术。

**📊 数据集**

采用 Siemens 套件（printtokens、printtokens2、replace、schedule 共 58 例）与 8 个 C 编译器（gcc/clang）共 66 个故障程序作为实验数据集。

**📈 对比分析**

通过对比输入大小、查询次数和耗时三项指标，DDMT 在无 oracle 场景下与 Perses 基线相当或略优，Oracle 可用时可实现 12–37% 的输入缩减、11–37% 的查询减少，但整体耗时略高。

**⚠️ 局限性**

主要限制在于高度依赖所选 metamorphic relation（MR）的质量；MR 低效时性能退化；mrtest 需要两次程序执行，导致时间开销增大；因此需要进一步开发更强大的 MR 及优化执行策略。

---

## 441. Human-Machine Collaboration on Generative Meta-Learning: Model and Algorithm

**arXiv ID:** 2607.00926 | [PDF](https://arxiv.org/pdf/2607.00926v1)

**作者:** Midhun Parakkal Unni `[一作]` (University of Sheffield), Samuel Kaski `[通讯]` (University of Manchester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 GMHF（Generative Meta‑Learner with Human Feedback）框架，通过在生成模型中嵌入可调控的 latent 变量，并让 AI 代理在 RL 的帮助下根据人类专家的二元反馈迭代优化生成的数据，从而显著缩小训练与部署分布之间的差距。

**💡 创新点**

创新点在于①将可解释的条件神经 ODE (cNODE) 作为数字孪生生成模型，使其可直接被 RL 代理通过调整物理参数（如刚度、阻尼）“操纵”；②将人类反馈建模为噪声带偏的二元奖励，理论上给出接受分布与目标分布 KL 散度的下界，证明了反馈质量与泛化误差的严格关联；③把生成、强化学习、元学习三者组合为一个闭环，首次实现了在无目标数据环境下的主动生成与人机协作。

**🔧 技术方法**

主要技术包括：条件神经 ODE (cNODE) 用于构建可调控的动力学生成器；强化学习（DDPG）用来学习在 latent 空间中调节物理参数的策略；元学习（Reptile）用于快速适应目标环境；理论分析使用 KL 散度、偏差-方差分解、Gaussian 近似求解接受分布。

**📊 数据集**

使用的实验数据集为两类人工合成数据：1）Duffing 振荡器的时间序列（参数 k、c、α 通过均匀采样产生）；2）一个线性‑三次项的非动态概率模型（用于验证方法的通用性）。

**📈 对比分析**

与仅使用无反馈生成或传统数据增强的基线相比，GMHF 在 Duffing 振荡器实验中将部署损失从 0.1–0.2 降至 <0.06（专家可靠度 ≥ 0.9），元学习率阈值约 0.54 时损失急剧下降，且在高非线性（α>1）时表现更好；在非动态概率模型中同样获得了显著的分布收敛与误差下降。

**⚠️ 局限性**

局限性包括：①需要源域与目标域共享可解释的物理模型，缺乏可迁移到完全不同领域的证据；②对专家反馈可靠度极低（≤0.5）时效果大幅衰退；③实验仅在低维模拟系统上验证，缺乏对高维图像/表格数据的适用性；④假设目标分布为静态，未考虑在线漂移或多任务情境。

---

## 442. GMO-E$^2$DIT: Grounded Multi-Operation Editing for E-Commerce Images

**arXiv ID:** 2607.00920 | [PDF](https://arxiv.org/pdf/2607.00920v1)

**作者:** Zipeng Guo `[一作]` (JD.com), Yan Li `[通讯]` (JD.com)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于代理的多操作电子商务图像编辑框架，利用VLM规划编辑议程并通过掩码条件编辑器执行局部操作，同时通过反射循环实现错误检测与回滚；

**💡 创新点**

创新点在于将编辑任务拆分为认知规划与局部生成两步，采用VLM作为策略规划与自我反思模块；结合掩码条件编辑器实现精确局部合成；设计了反射驱动的多轮迭代机制，提升了错误恢复与最终完成度；

**🔧 技术方法**

技术上主要使用视觉语言模型（VLM）进行规划与反思，掩码条件图像编辑器（如基于扩散模型的局部生成），GRPO强化学习调优策略，联合RL进行代理-编辑器协同训练；

**📊 数据集**

构建了统一的电子商务图像编辑数据管道，生成包含多操作、多轮标签的合成数据；并提出了新的基准测试集（E-commerce Grounded Multi-Operation Editing Benchmark），涵盖 add/remove/move/exchange/replace 等多种操作；

**📈 对比分析**

与多种开源与闭源编辑模型（Nano Banana、Seedream、Wan2.7-Image-Pro、Qwen-Image-Edit-2511 等）对比，实验显示在模糊指令下 IA/EF/BP 分别达 8.66/8.25/8.87，优于所有基线；在精准指令下进一步提升至 9.88/9.62/9.85；在人类评测中也显著优于对手；

**⚠️ 局限性**

限制方面：对大规模多操作场景仍需更多数据；反射逻辑仍基于离线标注，可能在极端复杂指令下误判；模型对长篇复杂指令的理解深度与通用性有待进一步验证；

---

## 443. AVSR-Diff: Scale-Agnostic Diffusion Priors for Temporally Consistent Arbitrary-Scale Video Super-Resolution

**arXiv ID:** 2607.00987 | [PDF](https://arxiv.org/pdf/2607.00987v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 444. From Registry to Repository: How AI Agent Skills Are Written, Adapted, and Maintained

**arXiv ID:** 2607.00911 | [PDF](https://arxiv.org/pdf/2607.00911v1)

**作者:** Haoyu Gao `[一作]` (University of Melbourne), Mansooreh Zahedi `[通讯]` (University of Melbourne)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统性地研究了AI代理技能的编写、复用与维护，收集并分析了来自公开注册中心和GitHub个人仓库的技能数据。

**💡 创新点**

首次提出了技能作为可工程化工件的生命周期框架，并通过六类内容主题和六类维护行为的分类，对技能的复用与演化进行深入量化。

**🔧 技术方法**

研究采用LLM进行SWEBOK知识领域归类、文本相似度匹配构建复用链接，并结合主题分析与定性编码对技能内容与变更进行编码。

**📊 数据集**

使用了18463个来自skills.sh注册中心的技能和23199个来自5,876个GitHub仓库的个人使用技能，构建了3,709条复用链接。

**📈 对比分析**

通过统计相似度阈值和Fisher检验比较定制化与演化中的变更比例，结果显示大多数复用技能保持原样，维护以增量修改为主，且方法具有显著统计显著性。

**⚠️ 局限性**

研究局限于公开注册中心和GitHub公开仓库的技能，复用链接阈值可能漏检重构或重命名的技能，并且对SWEBOK分类的LLM标签依赖模型稳定性。

---

## 445. Recovery of Planted Subgraphs

**arXiv ID:** 2607.00897 | [PDF](https://arxiv.org/pdf/2607.00897v1)

**作者:** Wasim Huleihel `[一作]` `[通讯]` (Tel Aviv University), Wasim Huleihel (Tel Aviv University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对在Erdős–Rényi随机图中植入任意子图的**精确恢复**问题进行系统研究，给出了统计与计算阈值，并提供了一般可行的最大似然（MLE）和半正定松弛（SDP）算法；同时给出低阶多项式框架下的计算下界，揭示统计-计算鸿沟。

**💡 创新点**

提出了新的图论量——最小最大子图密度 μ_min，并证明其决定了精确恢复的统计阈值；将一般子图的恢复问题归约为“层级剥离”（onion decomposition）下的逐层MLE；利用低阶多项式方法给出与经典随机图检验理论一致的计算下界；扩展到半随机模型和弱恢复范式，首次给出统一框架。

**🔧 技术方法**

核心技术包括：
- 层级剥离（onion decomposition）与最大相对子图密度；
- 逐层最大似然估计与其信息理论下界（Fano+幻灯机证明）；
- 半正定松弛（SDP）结合低秩核范数约束、核范数与相干性分析；
- 低阶多项式（LDP）框架的矩母函数与共轭矩计算；
- 适用于半随机模型的稳健性分析；
- 近似恢复的截断MLE和剩余边比例论证。

**📊 数据集**

使用**合成Erdős–Rényi随机图**（G(n,p)）和半随机模型的随机生成图作为实验数据；在不同的植入子图结构（如完全图、二分图、含稀疏尾部的Kite图等）上进行理论与数值验证。

**📈 对比分析**

与传统的谱检测或显式模式匹配方法比较：
- 信息阈值与MLE/SDP在特殊子图（完全图、二分图、平衡图）上完全匹配；
- 对于一般子图，算法满足足够高的 μ_min≥C·logn/d_KL(p∥q) 的条件；
- LDP计算下界表明在 μ_min<Θ(logn) 时任意低阶多项式算法均无法恢复，体现统计-计算鸿沟；
- 在统计下界附近，MSE和误差率随logn/μ_min的比例收敛。

**⚠️ 局限性**

主要限制包括：
- 计算上限与信息下界在部分子图（如含稀疏附枝的Kite图）上存在误差；
- 对于 μ_min 远小于 logn/ d_KL 的情形，现有 SDP 算法无法达到信息阈值；
- 低阶多项式下界仅适用于 p,q 为常数的稠密随机图，难以直接推广到稀疏或更复杂的生成模型；
- 对于极其稀疏的尾部结构，弱恢复（近似恢复）仍面临统计难题；
- 需要进一步研究更高阶松弛或非线性优化方法以压缩统计-计算鸿沟。

---

## 446. Beyond Document Grounding: Span-Level Hallucination Detection over Code, Tool Output, and Documents

**arXiv ID:** 2607.00895 | [PDF](https://arxiv.org/pdf/2607.00895v1)

**作者:** Ádám Kovács `[一作]` (KR Labs), Gábor Recski `[通讯]` (KR Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的 span 级幻觉检测基准，用于代码、工具输出、结构化文档和自然语言 RAG 场景的后生成验证。

**💡 创新点**

创新点在于：①构建跨多模态、跨领域的 span 级检测任务；②采用基于编辑的精确注释方式生成高质量标签；③训练了一个 2B 参数 Qwen 生成式检测器，并与多种基线进行对比。

**🔧 技术方法**

主要技术包括：生成式 span 检测模型（Qwen3.5‑2B）与编码器基线（mmBERT‑base），长上下文推理、标签化编辑注入、参考文本补全以及多语言语料的联合训练。

**📊 数据集**

使用的数据集包括新构建的 74,285 条实例（代码、工具输出、ACL 论文、README、维基百科）以及现有 RAGTruth 与 PsiloQA 的转化样本。

**📈 对比分析**

与 LettuceDetect‑large、LettuceDetect‑mmBERT、LFM‑8B、零射 LLM（Nemotron‑3‑Ultra、gpt‑oss‑120b）等基线对比，生成式检测器在整体 span‑F1 上达到 0.689，在代码‑agent 上达到 0.602，显著优于基线（如 0.17–0.22 span‑F1）。在自然语言基准上亦表现接近 RAG‑HAT（81.8 vs 83.9）。

**⚠️ 局限性**

主要局限包括：大部分标签为合成注入；仅对最终答案进行验证，未覆盖完整 Agent 轨迹；测试集虽人工复核，但训练集标签仍自动生成；跨模态数据分布可能不完全代表真实场景。

---

## 447. Privacy-Preserving Depth-Only Open-Vocabulary 3D Semantic Segmentation Via Uncertainty-Guided Test-Time Optimization

**arXiv ID:** 2607.00978 | [PDF](https://arxiv.org/pdf/2607.00978v1)

**作者:** Xuying Huang `[一作]` (University of Bonn), Maren Bennewitz `[通讯]` (University of Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出UTTO，一种在仅使用深度信息且不泄露RGB内容的隐私保护场景下的开放词汇3D语义分割的测试时优化框架。

**💡 创新点**

创新点在于将预测变异产生的不确定性转化为点级可靠性权重，结合无监督的文本原型去偏和特征空间语义一致性，实现无训练、无RGB、无标签的推理时优化。

**🔧 技术方法**

技术手段包括：标签保持的测试时增强生成多重预测、基于CLIP的多模板文本原型、DINO特征的一致性正则化、几何图和特征图的平滑约束。

**📊 数据集**

使用ScanNet的ScanNet20、ScanNet40和ScanNet200三个子集进行实验。

**📈 对比分析**

与RGB‑D基线、深度仅几何基线、伪RGB+Fusion等隐私保护方法对比；UTTO在深度仅条件下平均提升mIoU约3–5个百分点，性能已接近非隐私RGB‑D参考。

**⚠️ 局限性**

局限性在于仍存在显著的RGB缺失导致的性能差距；伪RGB生成方法表现不佳；对极端隐私需求下的更弱视觉信号适用性有限，且未在更大规模场景中验证。

---

## 448. Diffeomorphic Optimization

**arXiv ID:** 2607.00947 | [PDF](https://arxiv.org/pdf/2607.00947v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 449. Beyond Activation Alignment:The Alignment-Diversity Tradeoff in Task-Aware LLM Quantization

**arXiv ID:** 2607.00908 | [PDF](https://arxiv.org/pdf/2607.00908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 450. Aionoscope: Debugging Latent-State Accessibility in Time-Series Representations

**arXiv ID:** 2607.00956 | [PDF](https://arxiv.org/pdf/2607.00956v1)

**作者:** Alexander Chemeris `[一作]` (Langotime), Randall Balestriero `[通讯]` (Brown University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Aionoscope，一个基于生成器的诊断框架，用于评估冻结的时序表示中对潜在过程状态的可访问性。

**💡 创新点**

创新点在于将种子化的过程‑到‑视图生成器、完整的分类和连续标签、全新评估流以及统一的平均池化线性探测器结合起来，能够细粒度区分成分存在与密集状态信息的可访问性。

**🔧 技术方法**

采用了生成式合成数据、层级均值池化、线性分类/回归探测器（AUROC、R²）、固定特征基线以及对 37 个预训练模型+适配器的系统化评估。

**📊 数据集**

使用了单通道 500 Hz 的 Primitive Process Mixtures 生成数据集，共 14 种成分，混合复杂度 k=1、2、3；每个训练集 65,536 样本，10 条验证 seed 各 65,536 样本。

**📈 对比分析**

通过在每层均值池化特征上训练线性探测器来比较模型，报告平均分类 AUROC 在 0.90–0.98 之间，平均密集 R² 在 0.30–0.70 之间。结果显示成分检测表现优异，但对频率、振幅、时间等密集参数的恢复普遍不足；部分模型（如 MantisV2）在密集指标上表现突出。基线表明原始波形无效，FFT/统计特征可提升分类，但密集回归仍差；oracle 接近完美。

**⚠️ 局限性**

局限性：仅为单通道、500 Hz 的合成数据，缺乏多通道、间歇采样、长期依赖和真实业务分布；使用的是预训练模型的本地长度，未考虑裁剪/填充；探测器仅为平均池化线性读出，无法证明信息缺失；因此仅适合作为诊断性探索，而非模型排名或真实部署的依据。

---

## 451. Learning Cardiac Motion Priors for Implicit Neural Representations

**arXiv ID:** 2607.00955 | [PDF](https://arxiv.org/pdf/2607.00955v1)

**作者:** Andrew Bell `[一作]` (King's College London), Alistair Young `[通讯]` (King's College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并比较四种学习心脏运动先验（总体先验、共识先验、自解码器和元学习）用于隐式神经表示（INR）在心脏运动估计中的初始化与适配

**💡 创新点**

首次系统评估多种先验学习策略对INR心脏运动估计的影响，揭示不同先验对适配速度与优化轨迹的权衡

**🔧 技术方法**

使用隐式神经表示（SIREN网络）作为神经速度场，结合图像相似度、平滑、体积与折叠正则化，并应用Meta‑SGD、权重平均、自解码器等技术

**📊 数据集**

采用英国生物银行（UK Biobank）短轴标记心脏磁共振成像（tagged CMR）数据集进行训练、验证与测试

**📈 对比分析**

通过跟踪精度、分割Dice和机械可行性评估，五步适配后所有先验均明显优于随机初始化；Meta‑学习在5步后误差最低（1.66 mm），自解码器在收敛后阶段迅速达到最佳；共识先验简单且性能稳健；总体先验在长达50步适配后表现下降

**⚠️ 局限性**

总体先验在长期适配中趋向过拟合平均运动，难以进一步改进；自解码器虽快速收敛但在进一步迭代后停滞，说明调制机制受限；实验仅涵盖单模态CNR数据，未验证跨模态或不同病理状态的泛化能力

---

## 452. From Runtime Records to Legal Findings: An Evidentiary-Adequacy Criterion for Agentic AI Oversight

**arXiv ID:** 2607.00941 | [PDF](https://arxiv.org/pdf/2607.00941v1)

**作者:** Jeroen Janssen `[一作]` `[通讯]` (Apparens), Jeroen Janssen (Apparens)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

定义并验证了针对Agentic AI系统的证据充分性判定准则，说明只有记录同时包含法律类别标注和相关关系时，才能回答特定法律事实的二值判定。

**💡 创新点**

提出了仅在记录中同时具备法律类别标注和相关关系（如起源、权威、时效性）才满足“答案可行性”的必要性标准，并将该准则实例化于EU AI法义务。

**🔧 技术方法**

使用了构造性证明、最小化分析、与现有框架（NIST、ISO、EU AI法、运行时验证和溯源模型）的对比，以及预注册的专家评估实验设计。

**📊 数据集**

主要使用了在Zenodo存档的实验材料和基于EU AI法的合规日志数据，未使用公开的大规模数据集。

**📈 对比分析**

通过三组专家面板比较裸日志、仅溯源和具备类型化关系记录在判定准确率、时间和一致性上的表现，实验尚未公布结果，但预期具备类型化关系的记录显著提升答案正确率。

**⚠️ 局限性**

适用于二值事实判定，未证明充分性，无法保证记录真诚或完整，且未覆盖模型内部状态或未被捕获的传输通道。

---

## 453. Persona Non Grata: LLM Persona-Driven Generations in MCQA are Unstable in Distinct Dimensions

**arXiv ID:** 2607.00937 | [PDF](https://arxiv.org/pdf/2607.00937v1)

**作者:** César Guerra-Solano `[一作]` (University of Pittsburgh), Xiang Lorraine Li `[通讯]` (University of Pittsburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型在多选题回答任务中以人格驱动生成的稳定性，并提出了三种新的不稳定性度量。

**💡 创新点**

创新点在于量化人格驱动生成在不同实验设置（提示格式、温度等）下的性能、结果和正确题目的波动性。

**🔧 技术方法**

使用了LLM（Llama3.x、Qwen2.5.x）并基于提示格式与温度等超参数设计实验，构建三维不稳定性指标。

**📊 数据集**

数据集为MMLU、Social IQa与NormAd‑Eti的混合评测集，包含10个主题、41个人格。

**📈 对比分析**

通过在48种实验设置下对4款开源模型进行评测，发现模型对任务提示格式最敏感，数学类题目不稳定性最高，且不稳定设置导致准确率显著下降。

**⚠️ 局限性**

局限包括仅评估4款模型、仅关注多选题、实验设置与超参数未覆盖全部可能情况、人格集合有限、未分析人格本身导致的不稳定性。

---

## 454. Post-Training Pruning for Diffusion Transformers

**arXiv ID:** 2607.00927 | [PDF](https://arxiv.org/pdf/2607.00927v1)

**作者:** Chengzhi Hu `[一作]` (Institute of Automation Chinese Academy of Sciences), Qingyi Gu `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对Diffusion Transformers的后训练剪枝方法DiT-Pruning。

**💡 创新点**

创新点在于用平方变换调和权重与激活的贡献以及聚类感知剪枝粒度。

**🔧 技术方法**

利用基于能量的二阶敏感性指标、权重平方变换 (STW) 与聚类感知剪枝粒度 (CAG)，并在现有Wanda框架上实现。

**📊 数据集**

在PixArt-Σ、FLUX.1-dev以及DiT-XL/2等Diffusion Transformers模型上，使用COCO、ImageNet、MJHQ等公开数据集进行评估。

**📈 对比分析**

与Magnitude、Wanda等基线对比，DiT-Pruning在高稀疏率（50%）下保持FID/CLIP等指标几乎无损，表现优于现有方法。

**⚠️ 局限性**

局限性包括对大规模结构化稀疏度的评估有限，且对不同Token长度的适应性尚未深入探究。

---

## 455. From Personas to Plot: Character-Grounded Multi-Agent Story Generation for Long-Form Narratives

**arXiv ID:** 2607.00918 | [PDF](https://arxiv.org/pdf/2607.00918v1)

**作者:** Aayush Aluru `[一作]`, Vasu Sharma `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Magnet 这一多智能体驱动的长篇叙事生成框架，并配套 Atlas 图结构化的幻觉检测管道；

**💡 创新点**

创新点在于通过共享世界状态、目标驱动的动态目标生成、Critic 评估与 DPO 优化的角色智能体，实现跨场景的一致性与可控性；

**🔧 技术方法**

主要技术包括多智能体架构（角色、评估者、叙述者）、世界状态图、动态目标生成、DPO（LoRA Direct Preference Optimization）、Gemma‑4‑31B‑it、Opus 4.7、Gemini 2.5 Flash、Claude Sonnet 4.6 以及 GPT‑5.4 mini；

**📊 数据集**

实验使用自构造的长篇故事（2/20/100 页）以及人工标注的幻觉示例，未采用公开大规模数据集；

**📈 对比分析**

与单模型提示和 IBSEN 的对比表明，Magnet 在 100 页故事中编辑注解量下降 41%/34%，幻觉数量减少 50%/45%，并在层级化 Rubric 评估中获得更高分数；

**⚠️ 局限性**

主要局限在于高昂的计算成本、对多款闭源 LLM 的依赖、评估过程对 LLM 判定的主观性，以及仅在英文短文本上验证，缺乏跨语言和更长篇幅的通用性。

---

## 456. Calibrating the Instrument: Controllability of an LLM-Driven Synthetic Population

**arXiv ID:** 2607.00910 | [PDF](https://arxiv.org/pdf/2607.00910v1)

**作者:** Mirko Degli Esposti `[一作]` `[通讯]` (University of Bologna), Mirko Degli Esposti (University of Bologna)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

建立并验证了SIVE实验系统，用于检验生成合成人口（GSP）工具在面对已知情绪强度的消息时的可控性。

**💡 创新点**

首次系统化检验GSP内部可控性，利用预注册的七项指标和多温度测试揭示弱正向刺激的误判并成功校正，提供完整的内在有效性评估框架。

**🔧 技术方法**

结合统计最大熵人口合成、LLM（DeepSeek）记忆与反思代理、结构化问卷、温度采样以及Kendall τ等量化方法，进行单模型多温度实验和单个人重复测量噪声估计。

**📊 数据集**

使用120名合成个体的“Montelago”人口，预设三维信任结构（低/中/高）以及对应的背景故事、情感与行为标签，配合七个消息刺激文本。

**📈 对比分析**

通过七个预注册指标（C1–C7）在所有温度下均通过；信任、信息充足度的平均变化与预期秩序一致；噪声门限σ_instr≈0.77，信噪比≈2.35，顺序一致性τ≥0.9，证明系统性能良好。

**⚠️ 局限性**

仅使用单一LLM（DeepSeek）和单语言（意大利语）；样本规模有限，子组统计功效有限；噪声由交互差异与模型采样两部分组成，外部效度仍未评估。

---

## 457. Space-Optimal Sensitivity Oracles for Single-Source Mincuts

**arXiv ID:** 2607.00894 | [PDF](https://arxiv.org/pdf/2607.00894v1)

**作者:** Koustav Bhanja `[一作]` (Weizmann Institute of Science), Asaf Petruschka `[通讯]` (Weizmann Institute of Science)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种针对单源最小割灵敏度的稀疏结构，能够在任意边失效时以 O(n) 空间和 O(n) 查询时间报告所有受影响顶点，并给出更大但子平方空间（O(n^1.5)）的近似查询方案，支持 O(log n) 单点查询和 O(log^3 n) 输出敏感查询。

**💡 创新点**

核心创新在于将“最远最小割 DAG”和“连接性尸体（Connectivity Carcass）”两种看似无关的结构建立起细致的桥接关系，从而在单源失效查询中实现空间极致压缩；此外，提出的代表人框架（Representatives Framework）和锚点（Anchor）技术，使得能够高效地把查询转化为在 DAG 上的可达性问题并得到最优时间。

**🔧 技术方法**

主要技术包括：
- 远端最小割 DAG 的构造与性质分析；
- 连接性尸体的骨架与投影映射；
- 代表人框架（为每对顶点集合提供 O(1) 代表信息，并在 DAG 上快速求解分离顶点）；
- 锚点（Anchor）与投影路径的转换，用于把任意边的投影转化为节点投影；
- 对 DAG 的可达性做 O(n^1.5) 空间、常数/对数时间的查询器；
- 细化查询分为中段、重叠、前缀、后缀等子路径，并在这些子路径上使用代表人框架进行判定。

**📊 数据集**

该工作完全基于理论分析与构造，不涉及任何实验数据集；所有结果均在通用无向多重图的数学模型下证明。

**📈 对比分析**

与现有最优插入查询（O(n) 空间、O(n) 查询）相比，本文在失效查询上首次实现相同的空间上界；与先前 O(n^2) 空间的失效查询相比，空间压缩了两个数量级。O(n^1.5) 空间方案提供了几乎最佳的查询时间：单点 O(log n) 或输出敏感 O(log^3 n)；相比之前任何子平方空间方案，均实现了显著的时间提升。

**⚠️ 局限性**

限制与开放问题：
- 本文所用的 DAG 可达性查询器仍需 O(n^1.5) 空间，若要进一步压缩空间需突破当前 DAG 可达性难题；
- 目前只处理无权无向多重图，对带权边失效的情况仍需进一步扩展；
- 所有对的敏感性查询仍为 O(n^2.5) 空间；
- 未来需探索全对最小割灵敏度的 O(n) 空间实现。

---

## 458. Two AI Metrics Diverged: Will it Make All the Difference?

**arXiv ID:** 2607.00913 | [PDF](https://arxiv.org/pdf/2607.00913v1)

**作者:** Alex Fogelson `[一作]` (MIT FutureTech), Neil Thompson `[通讯]` (MIT FutureTech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 AI 能力衡量指标进行系统化分析，提出了“meek”与“mighty”指标的概念，并给出它们随训练/推理算力增长的收敛与不收敛特性。

**💡 创新点**

创新点在于给出一个完整的理论框架，利用指标的有界性和增长速度（低于 loglog C）来判定指标是否属于 meek，从而解释不同指标下能力差距是否持续扩大。

**🔧 技术方法**

使用了神经网络尺度律、微分与单调性分析、对数转换和功率律等数学技术，对指标函数进行形式化推导与证明。

**📊 数据集**

实验数据来源于 METR 时间跨度、BigBench、MMLU、ImageNet、CIFAR、MNIST 等 benchmark 数据集。

**📈 对比分析**

通过绘制不同指标下前沿模型与低预算模型的算力-性能曲线，对比两者在指数算力差异下的性能差距，结果显示有界指标收敛而无界指标保持差距。

**⚠️ 局限性**

局限性包括未涵盖位置性指标、对指数算力增长的假设、假设算法与硬件改进共享、缺乏联合训练-推理尺度律以及实证验证不足。

---

## 459. Valdi: Value Diffusion World Models

**arXiv ID:** 2607.00917 | [PDF](https://arxiv.org/pdf/2607.00917v1)

**作者:** Christopher Lindenberg `[一作]` (University of Tübingen), Kashyap Chitta `[通讯]` (Kyutai ELLIS Scalable Autonomous Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Value Diffusion World Models（VDWM），通过联合训练潜在扩散动力学模型与价值函数实现在线MPC规划。

**💡 创新点**

创新点在于：①将扩散模型引入潜在世界模型，直接在一阶段推理下实现多模态未来预测；②在TD-MPC框架下实现端到端在线训练，并使用单步扩散推理满足低延迟需求。

**🔧 技术方法**

技术手段包括：潜在扩散动力学模型（基于Transformer）、值函数与奖励网络、交叉熵方法（CEM）进行规划、对抗训练与EMA目标编码器。

**📊 数据集**

使用改版CarRacing环境进行实验，采集轨迹并构建重放缓冲区。

**📈 对比分析**

与传统的确定性MLP（类似TD-MPC）基线对比，VDWM在单步扩散推理下的控制性能相当，且在多模态预测上优于MLP；然而增加扩散步数虽提升预测多样性，却略微降低控制表现。

**⚠️ 局限性**

局限性包括：①仅采用单步去噪，可能不适用于更复杂动力学；②训练与推理的扩散步数不匹配，导致规划性能下降；③当前方法对多步扩散的利用尚需改进和迁移学习策略。

---

## 460. Slope-Guided Mamba and Angular-Refined Transformer for Light Field Super-Resolution

**arXiv ID:** 2607.00965 | [PDF](https://arxiv.org/pdf/2607.00965v1)

**作者:** Li Jin `[一作]` (Beihang University), Jie Wu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SMART，一种融合斜率引导Mamba与角度细化Transformer的混合网络，用于保持光场四维几何一致性并实现高质量的4×光场超分辨率；

**💡 创新点**

创新点包括：①角度调制空间模块（AMS）通过将角度先验融入空间注意力偏置，实现跨视角几何约束；②流形对齐轨迹模块（MAT）采用斜率引导的自适应扫描，按光场的等距平面直线结构对齐SSM采样，解决传统网格扫描与几何不匹配问题；③将Transformer的纹理提取与Mamba的几何一致性模型相结合，形成更高效的双向信息交互；

**🔧 技术方法**

使用技术主要包括：状态空间模型（Mamba）与Transformer注意力；角度先验投影与注意力偏置融合；斜率估计与自适应采样偏移生成；多尺度深度可分离卷积；自监督学习的深度一致性成本计算；以及多路径（前向、后向、斜率引导）扫描融合；

**📊 数据集**

实验数据集涵盖五大公开光场数据集：EPFL、INRIA、STFgantry、HCInew 和 HCIold，使用中心5×5子视角图像进行4×超分辨率训练与测试；

**📈 对比分析**

与超过20种现有光场超分方法（如VDSR、EDSR、RCAN、LFSSR、LFT、EPIT、LFmamba等）进行量化对比，SMART在所有数据集上平均PSNR达33.040 dB，SSIM 0.9478，较上一最佳L^2Fmamba提升0.42 dB；同时在细节恢复、纹理重建和几何一致性方面表现出更少的伪影；

**⚠️ 局限性**

局限性包括：①模型仍依赖于较大的训练数据集和高端GPU，推理成本相对较高；②仅针对4×超分，未验证更高倍率的适用性；③斜率估计使用固定的离散视差候选集，可能在极端深度或视角极限场景下表现受限；④对实时或移动端部署的可扩展性尚未评估；

---

## 461. MG-RWKV: Multi-Grained Context-Aware RWKV for Temporal Forgery Localization

**arXiv ID:** 2607.00902 | [PDF](https://arxiv.org/pdf/2607.00902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 462. A Geometric Perspective on Composable Emotion Steering in Text-to-Speech Models

**arXiv ID:** 2607.00946 | [PDF](https://arxiv.org/pdf/2607.00946v1)

**作者:** Siyi Wang `[一作]` (University of Melbourne), Ting Dang `[通讯]` (University of Melbourne)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文对混合情绪控制的混合式文本到语音系统，系统地比较了语音语言模型（SLM）和条件流匹配（CFM）两种激活调控点的几何特性，并评估了单点和联合激活调控对混合情绪合成的影响。

**💡 创新点**

创新点包括：①首次将线性可分辨性和局部内在维度（LID）方法应用于SLM和CFM的表示空间，揭示情绪子空间的可组合性与分离程度；②系统探究单点与联合激活调控的相互作用，指出联合调控会导致情绪比例失衡和音质下降；③为多点激活调控提供实证指南，强调需协调两模块的干扰。

**🔧 技术方法**

使用的技术主要有：线性探测器（Linear Probe）评估情绪可分辨性；局部内在维度（LID）估计情绪子空间维度；激活方向提取与加权组合实现混合情绪向量；情绪相似度（E‑SIM）、目标情绪强度（TEP）、Spearman相关（ρ）、最高情绪比例（H‑Rt）等指标评估情感控制；声学相似度（S‑SIM）和语音识别错误率（WER）评估音质。

**📊 数据集**

实验数据集包括：ESD、CREMA‑D、RAVDESS（共5种情绪）用于几何分析和单点调控；IEMOCAP用于离群情绪评估；混合情绪真实标注采用多评议者差异生成。

**📈 对比分析**

比较方法：在CosyVoice2架构下，分别在SLM层（14、17层）和CFM层（每5层共12层）提取情绪向量；通过改变调控强度α评估单点和联合调控；使用E‑SIM、TEP、ρ、H‑Rt衡量情感控制；使用S‑SIM、WER衡量音质。结果显示：SLM单点调控在情绪比例控制（ρ、H‑Rt）上优于CFM；CFM单点调控在情绪强度（TEP）和整体情感相似度（E‑SIM）上更突出；联合调控提升整体情绪强度但显著降低比例控制和音质（S‑SIM下降、WER上升）。

**⚠️ 局限性**

局限性：①联合激活调控在不同模块间产生干扰，未能实现协同提升；②CFM层情绪与说话人高度耦合，导致调控后说话人一致性下降；③实验仅在CosyVoice2单一混合TTS架构下验证，缺乏跨模型的泛化验证；④当前方法侧重离散情绪标签，无法直接支持连续情绪强度或细粒度情绪表达。

---

## 463. Condensing Large-Scale Datasets Directly with Minimal Information Loss

**arXiv ID:** 2607.00916 | [PDF](https://arxiv.org/pdf/2607.00916v1)

**作者:** Xinyi Shang `[一作]` (University College London), Tao Lin `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新型数据集蒸馏框架 CIM，直接在图像空间压缩原始数据集信息，避免传统的双重压缩过程，显著降低信息损失。

**💡 创新点**

创新点在于：①定义“有效信息”和“信息缺口”，用观测者集合量化样本信息；②引入可上界的距离度量，将信息缺口转化为可优化的目标；③通过中间层特征匹配平衡语义与纹理信息，实现更高保真度的蒸馏。

**🔧 技术方法**

核心技术包括：基于观测者的特征距离计算、KL 上界推导、数据增强视图生成、软标签重标定、以及逐样本的压缩迭代算法（Alg.1）。

**📊 数据集**

在 CIFAR‑10/100、Tiny‑ImageNet、ImageNet‑1K 等标准视觉数据集上进行实验，使用 ConvNet、ResNet‑18/50、MobileNet‑V2、ViT‑T/16 等多种网络结构进行评估。

**📈 对比分析**

与 SOTA 方法（如 SRe^2L、G‑VBSM、RDED、NRR‑DD、DELT 等）比较，CIM 在 10 IPC 级别下取得 ImageNet‑1K ResNet‑18 Top‑1 48.7%，比 NRR‑DD 与 DELT 提升 2.6%/2.9%；在小规模数据集上也持续领先，且在单 RTX‑4090 GPU 上完成 80 分钟即可完成 ImageNet‑1K 蒸馏。

**⚠️ 局限性**

局限性：①仍依赖先验的子样本选择策略（默认 RDED），不同选择方法对性能影响不一；②在极低 IPC 或小批量场景下，信息缺口上界近似可能导致最优性不足；③对高维复杂分布的理论保证有限，实际效果受模型和超参选择影响。

---

## 464. MultiSynt/MT: Trillion-Token Multi-Parallel Pre-Training Data Translated Across 36 Languages

**arXiv ID:** 2607.00890 | [PDF](https://arxiv.org/pdf/2607.00890v1)

**作者:** Maximilian Idahl `[一作]` (ellamind), Gema Ramírez-Sánchez `[通讯]` (Prompsit Language Engineering)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了36种语言的4.8万亿目标语标记的合成平行语料库MultiSynt/MT，并用其训练LLM，验证其在多语言基准上的性能优于原生语料；

**💡 创新点**

通过大规模机器翻译生成高质量多语料库并提供多系统对齐翻译，揭示标准多语言基准对翻译质量敏感度低，并提出LLM-judge评估和嵌入空间诊断等新方法；

**🔧 技术方法**

使用Tower+大型LLM、OPUS‑MT/HPLT‑MT翻译模型；训练采用Dense Llama‑style 1.61B参数Transformer；评估使用LightEval和LM‑evaluation‑harness等多语言基准；

**📊 数据集**

源数据为Nemotron‑CC HQ 100B英语语料；翻译后生成36种语言文本，包含Tower+9B、72B、OPUS‑MT/HPLT‑MT输出；

**📈 对比分析**

与HPLT 2.0原生多语言基线在相同架构、优化和100B标记预算下比较，发现MultiSynt/MT在约28B标记时就达到HPLT 2.0水平，100B时提升约15%相对；不同MT系统的差异在标准基准中不明显，但在LLM‑judge评估中能体现；

**⚠️ 局限性**

语料继承了Nemotron‑CC的主题、风格及英语中心化偏差，翻译产生的translationese和文化锚点不足，导致在本土化、习语等任务上逊色；实验仅覆盖欧洲中高资源语言，单参数规模1.7B，缺乏跨规模和种子方差验证，且公开语料可能在未来的MT循环训练中产生递归偏差。

---

## 465. Geometry-Aware Cross-Height Channel Knowledge Map Prediction for UAV-Assisted Communications With Uncertainty-Guided 3D Sensing

**arXiv ID:** 2607.00887 | [PDF](https://arxiv.org/pdf/2607.00887v1)

**作者:** Zhihan Zeng `[一作]` (University of Electronic Science and Technology of China), Guan Gui `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在几何丰富的城市环境下，研究低空无人机通过稀疏多高度观测实现跨高度频道知识图（CKM）预测，并提出基于FPN‑Transformer的几何感知预测框架及不确定性引导的在线感知策略。

**💡 创新点**

创新点包括：① 提出高度条件跨高度CKM预测任务，将场景几何先验、稀疏多高度观测和目标高度描述统一输入；② 设计融合特征金字塔与Transformer自注意的FPN‑Transformer网络；③ 通过教师不确定性图进行监督，并将预测不确定性转化为成本感知的主动感知决策。

**🔧 技术方法**

核心技术：FPN‑Transformer骨干、特征金字塔+Transformer自注意、无监督不确定性教师图、成本感知动作价值函数、联合损失（L1、Charbonnier、梯度一致性、uncertainty）。

**📊 数据集**

使用自研 Layered Aerial CKM 基准：7 个城市场景，4 高度层（20/60/100/120 m），采样率 5/10/20%，包含障碍区、风险场、飞行安全约束等。

**📈 对比分析**

与 U‑Net、ConvNeXt‑U-Net、3D‑RadioDiff 进行对比；在未见场景零样本下 RMSE 为 5.347 dB，传统 patch‑random 下为 1.111 dB，均显著优于基线；在少量样本快速部署与主动感知预算下，FPN‑Transformer 仍保持最佳性能。

**⚠️ 局限性**

局限性：仅在单基站、单用户、固定频率 3.5 GHz 的静态场景验证；缺乏真实测量实验；对多天线、多用户与时间变化的适应性不足；跨高度预测仍受复杂场景几何影响，极端高度下性能下降可能更明显。

---

## 466. Beyond Pixel Overlap: A Framework for Decomposing Segmentation Evaluation Metrics

**arXiv ID:** 2607.00886 | [PDF](https://arxiv.org/pdf/2607.00886v1)

**作者:** Youwei Pang `[一作]` (Nanyang Technological University), Xiaoqi Zhao `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一个将二元目标分割评估指标拆解为五个阶段（预测表示、目标提取、目标匹配、分数计算、指标报告）的框架，映射并归纳了现有指标的实现路径，阐明不同指标背后的设计假设。

**💡 创新点**

创新点在于将传统的“黑盒”评估公式转化为可组合的模块化路径，揭示指标间的共享子操作与差异来源，提供了一种系统化、可视化的评估设计空间，帮助研究者在指标设计时明晰决策点。

**🔧 技术方法**

主要采用理论分析、流程拆解和可视化技术，对已有指标（如MAE、Fβ、IoU、Dice、PR/ROC曲线、Weighted Fβ、Sm、Em、MSIoU、HCE、Cm等）进行分解，并构造五阶段表格与图示；并对不同阶段组合进行案例对比。

**📊 数据集**

论文并未进行新的实验评测，而是通过对公开文献中常用的分割任务（SOD、COD、TOS、GD、MD、SD、DIS、MLS等）指标进行综述与对比；若需验证，可在常见的分割数据集（如PASCAL VOC、COCO、BDD、COD10K、Synapse等）上复现其映射过程。

**📈 对比分析**

通过将指标映射到五阶段框架，作者对指标进行结构化比较：相同阶段路径的指标在上游数据处理一致，差异体现在后续计算或报告；相同最终数值的指标可能因阈值策略或目标实体不同导致解释差异。性能方面没有统一量化指标，而是阐释各阶段如何影响最终评价结果和任务适配度。

**⚠️ 局限性**

局限性：①框架聚焦于二元分割，未覆盖多类别、多标签或交互式分割；②对复杂的自适应阈值、上下文感知等机制的细粒度实现仍需在具体指标中手动定义；③由于缺乏统一实验验证，框架的实用性需在实际项目中进一步评估；④对现有指标的映射可能遗漏某些组合或自定义实现，导致解释不完整。

---

## 467. A Model-based Testing Technique for Amazon Lex Task-based Chatbots

**arXiv ID:** 2607.01094 | [PDF](https://arxiv.org/pdf/2607.01094v1)

**作者:** Diego Clerissi `[一作]` (University of Milano-Bicocca), Leonardo Mariani `[通讯]` (University of Milano-Bicocca)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了LexTester，一种针对Amazon Lex任务型聊天机器人的模型驱动测试工具，能够从聊天机器人的JSON描述生成对话图并自动化生成可执行的测试用例。

**💡 创新点**

首次为Amazon Lex提供完整的模型驱动测试框架，能够在对话图上实现N-gram路径覆盖，自动化生成槽填充和多路径对话，显著提升了测试复杂度与覆盖率。

**🔧 技术方法**

采用对话图（多边图）建模、N-gram路径覆盖策略、抽象到具体测试实例化、使用Botium提供的Amazon Lex连接器进行执行，并利用GraphStream和Graphviz进行图构建与可视化。

**📊 数据集**

使用了五个公开的Lex聊天机器人：两款预构建（Airline Bot、Book Trip Manager）和三款第三方GitHub项目（Cloud Assistant、Movie Recommender、Order Flowers Bot），覆盖不同领域与复杂度。

**📈 对比分析**

与业界主流工具Botium进行对比，LexTester在测试覆盖率（83‑95%）、缺陷检测（mutation score 58‑65% vs 13‑43%）和测试数量/复杂度上均表现更好，平均生成的测试数量约为Botium的1.9‑175倍，复杂度提升至2‑6轮交互，时间成本保持可接受范围。

**⚠️ 局限性**

受限于训练数据稀缺导致无法系统探索fallback路径；对流上下文寿命的缺陷检测不足；测试冗余仍然存在；工具在简单意图和槽填充不足时的覆盖率不及Botium。

---

## 468. Where Am I? Semantic Map Grounding via Vision-Language Models for Multi-Modal Localization

**arXiv ID:** 2607.01079 | [PDF](https://arxiv.org/pdf/2607.01079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 469. LongVQUBench: Benchmarking Long-Term Video Quality Understanding of Vision-Language Models

**arXiv ID:** 2607.01086 | [PDF](https://arxiv.org/pdf/2607.01086v1)

**作者:** Arpita Nema `[一作]` (Nanyang Technological University), Weisi Lin `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LongVQUBench，针对长视频质量理解的全新基准，包含1200段视频与1500道多选与开放式问题。

**💡 创新点**

创新点在于三层递进的评估框架（LQU、CQR、GQU）以及“针尖式失真问答”（NDQA）以细粒度检验感知与推理能力。

**🔧 技术方法**

采用零样本的框架：统一帧采样、单次推理；并评估14款主流大视觉语言模型（闭源、开源与智能代理）。

**📊 数据集**

数据集来自公开长视频库（LongVideoBench、MLVU、LongVideoReason）涵盖电影、纪录片、监控、第一人称等多种类型，视频时长10分钟至2小时。

**📈 对比分析**

与现有视频QA/VQA基准对比显示：闭源模型如GPT‑5、Gemini‑3在各层均领先，开放源模型表现差距明显；在全局质量理解（GQU）上的准确率普遍较低，说明长时序推理是瓶颈。

**⚠️ 局限性**

局限性包括：对长时序的推理仍弱，单帧采样对性能提升有限；开放式答案的完整性得分低；仅在部分代理模型中，基于关键帧的自适应采样未显著提升性能。

---

## 470. Can Agents Generalize to the Open World? Unveiling the Fragility of Static Training in Tool Use

**arXiv ID:** 2607.01084 | [PDF](https://arxiv.org/pdf/2607.01084v1)

**作者:** Song-Lin Lv `[一作]` (Nanjing University), Lan-Zhe Guo `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 OpenAgent（工具使用在开放世界）问题框架，搭建受控沙箱环境，系统评估 SFT 与 RL 代理在四个层级（感知、交互、推理、内化）下的泛化表现，并基于诊断结果提出 Perturbation-Augmented Fine‑Tuning (PAFT) 方法来提升 SFT 的鲁棒性。

**💡 创新点**

①从开放世界视角正式定义工具使用的四维分布漂移（查询、动作、观察、领域）；②构造四层诊断框架（Perception、Interaction、Reasoning、Internalization）以量化不同类型漂移对模型的影响；③首次提出基于轨迹扰动的 PAFT，通过环境反馈、边界性与符号扰动三种操作显著缓解 SFT 的过拟合与符号锚定。

**🔧 技术方法**

使用 Qwen2.5-7B‑Instruct 作为基座模型；SFT 采用全参数微调，RL 采用 GRPO；PAFT 通过三类扰动（EFP、SBP、SRP）生成增强样本；评估指标包括 Tool Error Rate、Active Exploration Score、Average Tool Chain Length 与 Refusal Rate。

**📊 数据集**

自构造的基准数据集：6050 条训练样本与 880 条评估样本，涵盖地理兴趣点查询与计算任务，确保训练与测试严格分离，避免模式泄漏。

**📈 对比分析**

与传统 SFT 与 RL 基线在同一沙箱下对比；在四个层级的多种扰动下记录性能差异。PAFT 在早期训练阶段将准确率差异从负提升至正（如 Tier‑1 Acc Δ +28.6%），在边界拒绝率上从 12.2% 提升至 99.6%，整体表现显著优于两者。

**⚠️ 局限性**

受限于合成任务与有限的工具集，难以直接推广到真实复杂场景；RL 仍存在“必成就”报酬导致的强制完成偏差；PAFT 依赖人工设计的扰动策略，可能无法覆盖所有现实世界异常；对长期持续自适应与安全性的进一步验证仍待研究。

---

## 471. Balancing Expressivity and Learnability in Quantum Kernel Bandit Optimization

**arXiv ID:** 2607.01080 | [PDF](https://arxiv.org/pdf/2607.01080v1)

**作者:** Yuqi Huang `[一作]` (National University of Singapore), Sharu Theresa Jose `[通讯]` (University of Birmingham)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出在量子核高斯过程Bandit优化中使用投影量子核、随机傅里叶特征和P‑greedy等近似技术，以降低模型维度并减少信息增益，提升样本效率。

**💡 创新点**

创新点在于建立信息增益与模型欠拟合之间的理论权衡，给出误差-信息增益上界，并针对量子核设计LPQK、RFF和P‑greedy三种近似方案，首次实现了可调节的量子GP Bandit算法。

**🔧 技术方法**

使用的技术包括量子高斯过程、投影量子核（LPQK）、随机傅里叶特征（RFF）、P‑greedy核近似，以及GP-UCB、SquareCB等Bandit策略。

**📊 数据集**

实验数据集包括：合成三量子位量子函数、基于三量子位的相位分类任务、以及三量子位XYZ海森堡哈密顿量的VQE优化。

**📈 对比分析**

实验与完整量子核以及经典RBF核对比，发现适当的近似维度能显著降低累计遗憾（比完整量子核低约10-30%）、提升样本效率，并将计算复杂度从O(T³)降至O(TD²+D³)；在相位分类和VQE任务中，近似模型往往在样本数小于100时就已超越完整量子核。

**⚠️ 局限性**

局限性包括：近似误差对具体量子电路结构高度依赖，需手动或经验估计；非平移不变的量子核不易采用RFF；实验仅在模拟环境和小规模量子位上验证，尚缺乏大规模NISQ设备的实测验证。

---

## 472. Fair Allocation under Conflict Constraints via Strong Colorability

**arXiv ID:** 2607.01059 | [PDF](https://arxiv.org/pdf/2607.01059v1)

**作者:** Ishay Haviv `[一作]` `[通讯]` (Academic College of Tel Aviv-Yaffo), Ishay Haviv (Academic College of Tel Aviv-Yaffo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在冲突约束下，对共享偏好的代理进行公平分配，探讨SD‑EF1、EF1和EF[1,1]三种公平性，并通过引入强色数的层次结构给出存在性和算法性结果。

**💡 创新点**

创新点在于：①把公平分配问题与强色数（特别是类型1和类型2）建立精准联系；②给出SD‑EF1下的完整阈值（等价于强色数类型1）；③证明EF1、EF[1,1]在共享偏好下的充分条件与强色数类型1/2 的关系；④提出通用算法框架，将问题转化为(ℓ,r)‑强着色；⑤在最大度 Δ 的图上给出可多项式时间的 3Δ−1 或 (3+ε)Δ 条件；⑥在路径图上实现 4 代理的有效分配。

**🔧 技术方法**

主要技术包括：强色数的定义与层次结构、对 SD‑EF1 的等价表述、使用非递减排列构造分区、利用强着色的存在性与算法性、借助 Haxell、Harris 等人关于强色数的极限与算法、以及对周期图的 ‘cycle‑plus‑triangles’ 结论。

**📊 数据集**

本工作为理论研究，未使用任何实验数据集，全部以图论理论与算法证明为主。

**📈 对比分析**

与以往仅给出可行性或数值实验的研究相比，本文给出明确的阈值（如 ℓ≥χ_s^1(G) 或 ℓ≥3Δ−1）并在多项式时间内实现；算法复杂度为多项式，常数取决于 ε；在路径图中实现的阈值 4 与以前的 3 阈值相比提升了 1。总体性能优于先前基于最大连通分量大小或简单均衡着色的结果。

**⚠️ 局限性**

局限性：①对 EF1 与 EF[1,1] 的充分条件在一般情况下并非必要；②目前仅考虑共享偏好，异质偏好下的阈值尚未完全确定；③算法框架依赖于 (ℓ,r)‑强着色的可实现性，虽然已给出在最大度约束下的实现，但在一般图上强着色的多项式算法尚未完成；④对强色数的层次上限（如强色数 2Δ 的猜想）仍未证实。

---

## 473. Conversable Complexity: Agentic LLM Collectives as Interpretable Substrates

**arXiv ID:** 2607.01047 | [PDF](https://arxiv.org/pdf/2607.01047v1)

**作者:** Elias Najarro `[一作]` (IT University of Copenhagen), Stefano Nichele `[通讯]` (Østfold University of Applied Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并讨论了基于代理式大型语言模型（Agentic LLM）的集体作为新的人工生命（ALife）底层，阐述了其在持久记忆、工具使用与自我驱动等方面的特性，并通过对“Agents of Chaos”“Moltbook”“TerraLingua”等现有实验平台的回顾，说明了这种集体可在开放式交互环境中产生适应、协同、规范形成等生命般行为。

**💡 创新点**

创新点在于：①将已具备语言推理、工具调用与持续记忆的 LLM 赋予自发行动能力，形成可持续的代理实体；②利用自然语言作为唯一可读介质，使得整个集体的交互与记忆既具高度复杂性又可直接解读；③把传统 ALife 研究的自下而上、可解释性要求与 LLM 的“可解释性”结合，构建了一种既可模拟又可直接观察的全新实验底层。

**🔧 技术方法**

技术手段包括：大型预训练语言模型（如 Claude、Kimi、OpenAI GPT 系列）、持久化记忆存储（Markdown/文件系统）、工具调用接口（API、Shell 命令）、自然语言交互框架（OpenClaw、Discord 等）、以及多代理协调机制（事件队列、共享工具库）。

**📊 数据集**

数据来源主要是现有的代理交互记录与实验日志：Moltbook 观察档案（数月的帖子与评论）、Agents of Chaos 的 Discord/邮箱对话、TerraLingua 的共享记忆文件等；论文并未提出新的标注数据集，而是利用这些公开或实验生成的长时序交互数据进行分析。

**📈 对比分析**

评估方式主要是对比传统软硬/湿底层与代理式 LLM 集体在可解释性、开放式动态、适应性和文化传递等维度的表现；文章指出代理集体在可解释性（自然语言可读）与开放式演化（工具与规范可自我扩展）方面优于传统底层，但缺乏统一的量化基准，尚未展示具体性能指标。

**⚠️ 局限性**

局限性包括：①自我报告与链式推理的可信度仍不高，可能被对齐偏差或隐藏推理误导；②缺乏标准化的安全与效能评测框架，难以系统量化其产生的不可预期行为；③高内部复杂性导致对内部机制的深层分析仍受限；④实验多聚焦于短期或受控环境，未能充分验证在真实社会生态中的可持续性与鲁棒性。

---

## 474. Behavior-Adaptive Conversational Agents: Toward a Fluid Personality Framework

**arXiv ID:** 2607.01034 | [PDF](https://arxiv.org/pdf/2607.01034v1)

**作者:** Hasibur Rahman `[一作]` (Northeastern University), Smit Desai `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出了Fluid Personality Framework，用于LLM‑based Conversational Agents在不同情境下动态调整元喻人格与人格表达强度。

**💡 创新点**

将元喻人格设计与人格表达强度调节两大研究线索统一为单一可动态适配的框架，支持按任务、用户特征与情境即时切换。

**🔧 技术方法**

采用 Prompt Engineering、对话逻辑以及 Trait Modulation Keys 等技术实现人格与表达强度的动态配置。

**📊 数据集**

未使用具体数据集，仅基于已有文献和设计说明。

**📈 对比分析**

没有实验比较与性能评估，本文主要是提出概念性框架与设计原则。

**⚠️ 局限性**

局限在于缺乏实现与验证，实际可行性与效果尚未通过实验评估；实现过程可能面临多模态调度与用户反馈实时感知挑战。

---

## 475. Evidence-Supported Credit Risk Report Generation Using News-Centric Financial Knowledge Graphs

**arXiv ID:** 2607.01023 | [PDF](https://arxiv.org/pdf/2607.01023v1)

**作者:** Rocio Jimenez-Villen `[一作]` (Universidad Politécnica de Madrid), Ryutaro Ichise `[通讯]` (Institute of Science Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了FinKG-News框架，自动构建基于新闻事件的公司中心金融知识图谱，并利用该图谱结合上下文学习生成信用风险报告。

**💡 创新点**

创新点在于将新闻事件作为事实锚点与公司实体显式关联，形成动态、环境感知的知识图谱，并通过多维度驱动（财务、所有权、运营环境）与反思式提示提升报告的事实性与可解释性。

**🔧 技术方法**

技术包括LLM驱动的事件检测与关系推断、知识图谱构建、事件归类与地理标注、基于上下文学习的多模块报告生成、反思式提示与结果集成、以及自动与人工的质量与幻觉评估。

**📊 数据集**

数据集涵盖从Wikipedia时序表提取的354条全球事件（2000-2025年）、FNSPID金融新闻数据（约80,000篇包含事件的文章）以及FinKG原始公司关系与财务信息。

**📈 对比分析**

与仅使用基础任务说明的基线对比，本文在5家中小企业的信用风险报告中实现了内容质量提升19%-34%并显著降低幻觉率，人工评估的提升更为显著，而自动评估仍表现出对幻觉的敏感度不足。

**⚠️ 局限性**

主要局限在于LLM上下文窗口导致的分块处理增加计算成本、事件-公司影响关系可能不够具体或缺乏因果依据，以及自动幻觉检测与质量评估仍不可靠，仍需人工专家审阅。

---

## 476. KnowledgeDebugger -- an Exploration Tool for Knowledge Localization and Editing in Transformers

**arXiv ID:** 2607.01000 | [PDF](https://arxiv.org/pdf/2607.01000v1)

**作者:** Eric Benz `[一作]` (Heidelberg University), Artur Andrzejak `[通讯]` (Heidelberg University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 KnowledgeDebugger，一款交互式 GUI 工具，用于可视化和编辑 Transformer 语言模型中的知识；

**💡 创新点**

创新点在于将 EasyEdit 的多种知识编辑方法与可视化界面、可自定义指标和实验快照结合，降低使用门槛并提高可复现性；

**🔧 技术方法**

技术包括基于 Web 的 GUI、EasyEdit 库接口、残差流可视化、可自定义指标插件以及 JSON 快照导入导出；

**📊 数据集**

使用的主要数据集为 GPT2-XL（以及在附录中验证的 Llama‑3.1）进行案例研究；

**📈 对比分析**

通过对比 ROME、PMET、MEMIT 等方法的指标（如更新范数、困惑度、关键词排名），展示了工具在快速诊断编辑质量与模型崩溃方面的有效性；

**⚠️ 局限性**

局限性包括仅支持 EasyEdit 中已实现的模型与方法，对超大模型或自定义训练模型的支持受限，且尚未提供完整的性能基准评估。

---

## 477. Staleness-Learning Rate Scaling Laws for Asynchronous RLHF

**arXiv ID:** 2607.01083 | [PDF](https://arxiv.org/pdf/2607.01083v1)

**作者:** Jingwei Song `[一作]` (University of Hong Kong), Bill Shi `[通讯]` (Gradient)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析异步GRPO训练中回放延迟导致的梯度偏差，并提出基于行为策略显式化的两约束稳定性规则。

**💡 创新点**

创新点在于将行为策略作为显式变量，分离即时与延迟梯度，从而得到O(Sη)的单步偏差上界和条件崩溃时间缩放，并解释学习率阈值与延迟的关系。

**🔧 技术方法**

采用理论推导、Pinsker不等式、梯度分解、GRPO梯度估计及行为策略平滑性假设等技术手段。

**📊 数据集**

实验使用数学推理任务的规则奖励数据集，对不同规模模型进行验证。

**📈 对比分析**

与同步GRPO对比，实验显示在满足两约束时异步训练稳定；学习率阈值随延迟成反比，崩溃时间与学习率成正比，验证了理论预言。

**⚠️ 局限性**

局限在于仅考虑局部平滑性假设，未覆盖更大范围的分布漂移；实验仅在单一规则奖励任务上进行，缺乏跨任务泛化验证。

---

## 478. Human-Centric Transferable Tactile Pre-Training for Dexterous Robotic Manipulation

**arXiv ID:** 2607.01067 | [PDF](https://arxiv.org/pdf/2607.01067v1)

**作者:** Chi Zhang `[一作]` (Peking University), Zongqing Lu `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套以人类第一人称视频为基础的触觉预训练体系（TTP），通过收集大规模触觉-动作-语言数据并在Vision‑Language‑Action模型上进行预训练，随后在真实机器人上进行微调，实现从人类到机器人的触觉驱动细粒度操控迁移。

**💡 创新点**

创新点在于：①首个利用人类触觉数据进行大规模预训练的 VLA 框架；②提出统一的动作与触觉空间，消除人机间的模态差异；③设计双专家（动作预测 + 触觉预测）与触觉‑动作保持门控（MPG）机制，提升模型对触觉动态的建模与泛化能力。

**🔧 技术方法**

采用基于 BeingH‑0.5 的多模态 Transformer，加入触觉专家并使用流匹配（Flow‑Matching）训练，结合 MPG 调控上下文，使用 UniTacHand 统一触觉表示，整体架构为 VQA‑style 的查询‑答案形式。

**📊 数据集**

使用了自研的 H‑Tac 数据集（包含 HOI‑Tac、DeskTask‑Tac 与 InternData‑Tac），共 160 h 视图‑触觉‑动作数据，覆盖 300+ 任务与 135k+ 片段，此外还在公开的 LIBERO、LIBERO‑plus 与 Robocasa 进行模拟验证。

**📈 对比分析**

与 BeingH‑0.5、π_0.5 等无触觉或未预训练基线相比，TTP 在模拟基准（LIBERO‑plus、Robocasa）上保持同等或更优成绩，在 9 项真实机器人细粒度/接触任务中成功率与操作精度提升 30–70%，如切片、擦拭与插拔等任务表现显著。

**⚠️ 局限性**

局限性包括：① 预训练需要大量人类触觉采集，成本与可扩展性受限；② 触觉传感器与模型对接仍需统一标准，跨平台迁移受限；③ 在极端光照、遮挡或极端动力学场景下，模型对触觉的依赖可能导致性能下降。

---

## 479. AutoSpeed: Annotation-Free Stage-Adaptive Motion Speed Learning for Robot Manipulation

**arXiv ID:** 2607.01051 | [PDF](https://arxiv.org/pdf/2607.01051v1)

**作者:** Qingda Hu `[一作]` (Fudan University), Wenchao Ding `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并评估了 AutoSpeed 框架，能够在不需要速度或阶段标签的情况下，自动学习阶段自适应的运动速度与预测时长，从而提升机器人抓取和装配任务的执行效率。

**💡 创新点**

核心创新是基于多目标选择的成本函数，自动推断阶段速度并通过离散余弦变换（DCT）实现非整数速度的频域缩放；同时引入非线性时间聚合，兼容非生成与生成式策略，真正做到无注解阶段自适应。

**🔧 技术方法**

技术手段包括端到端多目标选择优化、DCT 频域时间缩放、轻量级比率头预测速度、非线性时间聚合（NTA）以及对生成模型的三阶段训练策略。

**📊 数据集**

使用的实验数据集涵盖 ALOHA、LIBERO‑10、Meta‑World 三大仿真基准以及真实机器人 Agiex Piper 平台收集的四个桌面任务（共约 200 条演示轨迹）。

**📈 对比分析**

与 ACT、BAKU 等基线在单任务、跨任务以及真实环境中对比，AutoSpeed 在多项指标上均显著优于基线：成功率提升 5–20%，执行时间缩短 30–70%，平均速度提升约 1.7–2.0 倍。

**⚠️ 局限性**

局限性包括生成模型训练时需要额外的候选目标计算，速度范围和长度惩罚系数需手动调参；在极端复杂或高噪声任务中，预测误差仍可能导致速度选择失效。

---

## 480. GeoSearcher: Anchor-Guided Progressive Reasoning for Remote Sensing Visual Grounding with Process Supervision

**arXiv ID:** 2607.01050 | [PDF](https://arxiv.org/pdf/2607.01050v1)

**作者:** Dianyu Wang `[一作]` (Chinese Academy of Sciences), Lei Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对遥感视觉定位任务，提出GeoSearcher框架，将其从单一步骤坐标生成改为anchor‑guided progressive reasoning，并通过两阶段后训练（ACR‑SFT与PF‑GRPO）实现。

**💡 创新点**

创新点包括：①将引用对象作为anchor并在上下文中显式插入；②设计Process‑Aware Reward（PAR）和Reasoning‑Informative Sample Selector（RISS）以强化中间推理过程；③构建并验证Anchor‑Centric Reasoning数据；④将监督微调与强化学习相结合的两阶段后训练策略。

**🔧 技术方法**

使用的技术包括：Qwen3VL‑4B多模大语言模型、监督微调（SFT）、强化学习（GRPO）、奖励设计PAR与RISS、视觉-文本一致性验证（RemoteCLIP）、上下文对齐anchor插入。

**📊 数据集**

实验数据集包括DIOR‑RSVG、OPT‑RSVG和VRS‑Bench，作者还构建了Anchor‑Centric Reasoning数据集用于ACR‑SFT。

**📈 对比分析**

与通用MLLM、遥感专用MLLM及现有推理模型相比，GeoSearcher在DIOR‑RSVG的Pr@0.5/Pr@0.7/mIoU分别达到83.2%/72.4%/73.5%，在OPT‑RSVG上为81.2%/65.5%/69.4%，在VRS‑Bench上为80.3%/64.4%/69.2%，均显著优于前沿方法，并在过程级指标Faithful@0.5提升、Shortcut@0.5下降。

**⚠️ 局限性**

主要限制是anchor定位的准确性决定了推理效果；anchor误差大时性能下降；目前仅支持单一anchor，未处理多参考或多重约束的复杂查询。

---

## 481. Data-driven mitigation of catastrophic forgetting in dynamic physical layer attack detection

**arXiv ID:** 2607.01041 | [PDF](https://arxiv.org/pdf/2607.01041v1)

**作者:** Aleksandra Knapińska `[一作]` (Wrocław University of Science and Technology), Marija Furdek `[通讯]` (Chalmers University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于训练数据平衡的持续学习框架，用历史攻击样本在动态光网络物理层攻击检测模型更新时避免灾难性遗忘并加速对再次出现攻击的适应；

**💡 创新点**

首次在光网络安全中将数据平衡视为动态适应的核心，通过插入历史攻击样本实现模型重训练，而无需改动网络结构；

**🔧 技术方法**

采用增量学习与OPM参数的机器学习检测（MLP单隐藏层、深度扩展基线），实现阈值触发的回放式数据平衡；

**📊 数据集**

使用实验室光网络测试平台收集的OPM参数数据，包括6类攻击（INB、OOB、POL及其强弱级别）与正常流量，约1.44k日轨迹，31个OPM特征；

**📈 对比分析**

与传统静态、动态MLP及深度扩展网络基线对比，平衡更新将BAC标准差降低约110%，回归速度提升约13.5批次；与加深网络相当，结合后可进一步提升稳定性与回归速度；

**⚠️ 局限性**

仅在二分类场景与实验室数据验证，需进一步评估在更大规模、真实网络环境中的可扩展性和对多类别攻击的适应；方法依赖阈值设定与历史数据库质量。

---

## 482. The Model Organism Lottery: Model Organism Interpretability Strongly Depends on Training Methodology

**arXiv ID:** 2607.01033 | [PDF](https://arxiv.org/pdf/2607.01033v1)

**作者:** Andrzej Szablewski `[一作]` (LASR Labs), Stefan Heimersheim `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了54个模型生物（MO）样本，涵盖三种怪癖、两种基模型架构与七种训练方法，并使用四种白盒可解释性技术对其进行评估。

**💡 创新点**

首次系统量化MO可解释性随训练目标、数据混合、模型架构与数据生成管道变化的敏感度，并提出在MO基准中需要对怪癖表达率(QER)进行匹配与多样化训练策略。

**🔧 技术方法**

采用激活oracle、激活推导、logit lens、稀疏自编码器（SAE）等白盒技术，并结合Post‑hoc SFT、DPO、SDF等训练方法。

**📊 数据集**

使用公开的OLMo与Gemma模型训练数据，合成的恶意文本、对话、文档等数据集，以及来自外部DPO数据的样本。

**📈 对比分析**

对四种方法在不同MO变体上的性能进行对比，发现可解释性在同一怪癖族内差异可达1.2–20.4倍，集成DPO通常最难被解释，且方法之间的相对排名不稳定。

**⚠️ 局限性**

受限于模型规模、实验规模与仅在部分怪癖上进行的重复实验，无法完全排除训练随机性与怪癖本身导致的可解释性差异，且对大规模安全相关怪癖的验证仍缺失。

---

## 483. Seahorse: A Unified Benchmarking Framework for Spatiotemporal Event Modeling

**arXiv ID:** 2607.01022 | [PDF](https://arxiv.org/pdf/2607.01022v1)

**作者:** Yahya Aalaila `[一作]` (German Research Center for Artificial Intelligence), Sebastian Vollmer `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了统一的 Spatiotemporal Point Process（STPP）实验框架 Seahorse，提供统一的 encode–evolve–decode 接口、可执行的 benchmark contract，并配套合成诊断套件 HawkesNest，用于在相同预处理、分割、坐标、损失和评估协议下对多种神经 STPP 模型进行可复现比较。

**💡 创新点**

创新点在于：1）把多样化的神经 STPP 模型抽象为统一接口，消除实现细节差异；2）构建可执行的 benchmark contract，确保所有模型在相同数据、分割、坐标、损失和评估方式下可直接比较；3）引入可控合成数据（HawkesNest）以诊断模型在不同结构复杂度（时空耦合、背景异质性等）下的鲁棒性；4）提供完整的配置、日志、可视化等工具链，实现实验的可复现与可扩展性。

**🔧 技术方法**

技术实现包括：Python CLI + PyTorch 模块化实现；Ray Tune 超参搜索；统一的数据加载与坐标变换；原始坐标下的 NLL、CRPS、强度相关性评估；合成数据生成器 HawkesNest；多线程/分布式训练与日志；自动化结果打包与可视化。

**📊 数据集**

使用的真实数据集有：Citibike 出租车起点事件、USGS 地震事件、New Jersey COVID 病例三大 2D 空间序列；同时 Seahorse 内置 13 个真实 STPP 数据集；合成数据则由 HawkesNest 生成，能控制时空耦合、背景异质性、拓扑结构等因素。

**📈 对比分析**

比较方法：所有模型在相同的原始坐标下报告 NLL（若模型在内部使用归一化坐标，则加 Jacobian 校正），并补充一阶时序预测 CRPS 与强度相关性等诊断。实验显示不同数据集对模型排名影响显著，未出现单一模型始终领先；在 HawkesNest 的时空耦合实验中，NSMPP 在 NLL 上稳居首位，但在 CRPS 与强度回归方面表现中等；学习曲线显示不同模型在高耦合情形下收敛速度与稳定性差异显著。

**⚠️ 局限性**

局限性：1）真实数据覆盖仅为 3 个 2D 空间序列，未覆盖更高维或标记事件场景；2）部分高成本模型在资源受限环境下不稳定，导致实验时间长；3）合成数据仍有限，无法覆盖所有现实场景；4）框架对新模型的接入需要手工实现统一接口；5）缺乏针对鲁棒性、校准、实时决策等实务指标的完整评估。

---

## 484. Reading Order Inference for Complex Document Layouts

**arXiv ID:** 2607.01018 | [PDF](https://arxiv.org/pdf/2607.01018v1)

**作者:** Iddo Hakim `[一作]` (Tel Aviv University), Nachum Dershowitz `[通讯]` (Tel Aviv University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出一种无训练的图模型，利用预训练语言模型对 OCR 文本行间的语义连贯性打分，求解带一次前后约束的有向路径覆盖，从而推断复杂非 Manhattan 版式（如 Glossa Ordinaria）的阅读顺序。

**💡 创新点**

创新点包括①将阅读顺序视作有向路径覆盖并引入一次前后约束；②以无训练方式组合 CLM 条件似然与 BERT NSP 的加权得分；③提出最大后悔边选择策略提升对错误“边盗窃”的鲁棒性；④证明方法对水平/垂直镜像保持不变性。

**🔧 技术方法**

使用技术包括：EleutherAI/Pythia 410M 及 BERT 的预训练语言模型、加权加性得分、最大后悔边选择、图论路径覆盖求解以及候选边的空间过滤；评估基于合成 Grid、ALTO 历史页面及 OmniDocBench 多列子集。

**📊 数据集**

采用的数据集有：8×8 与 16×16 的合成 Glossa 布局；23 页 ALTO 历史页面（含水平镜像、垂直翻转和双列校验页）；140 页 OmniDocBench 多列子集（包含学术论文、考试、报纸、杂志等）。

**📈 对比分析**

方法通过与递归 XY-cut（PaddleOCR）、LayoutReader 与 LayoutReader‑T 在相同 OCR 行级输入下进行对比。合成 Grid 上比几何基线高约 10%；在 ALTO Wrap‑around/Glossa 页平均 94.8% 对比 XY-cut 49.7%，整体 95.4%；在 OmniDocBench 140 页宏观准确率 88.0% 对比 XY-cut 75.3%，镜像不变性差异 ≤0.6%。

**⚠️ 局限性**

主要限制包括：候选边数量大时语义得分计算成本高；对极短或语义平淡的行区分力有限；未对 OCR 误码鲁棒性进行评估；仅在英文 LTR 文本上验证，对 RTL、垂直书写及多语种历史文献需进一步测试。

---

## 485. The Singular Source of Vineyard Monodromy

**arXiv ID:** 2607.01046 | [PDF](https://arxiv.org/pdf/2607.01046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 486. Clinician-Level Agreement Without Clinical Caution: LLM Evaluator Limits in Medical AI Benchmarking

**arXiv ID:** 2607.01103 | [PDF](https://arxiv.org/pdf/2607.01103v1)

**作者:** William Philipp `[一作]` (University of Luebeck), Sebastian Fudickar `[通讯]` (University of Luebeck)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了MedQADE——一个基于德国Ankizin医学问答数据集的3,800条开放式临床问答基准，并通过9名执业医生与9个LLM评判者进行并行标注；

**💡 创新点**

首次实现了德语开放式临床评测的标准化流程，并系统评估LLM评判者的统计一致性、临床元认知缺失以及家族偏差，揭示了自动评判在临床安全性上的局限；

**🔧 技术方法**

采用了大规模生成式LLM（Gemini、GPT、Gemma、Qwen等）生成学生答案，随后使用结构化提示、角色扮演和JSON输出的LLM评判者；统计方法包括Cohen's κ、PABAK、Krippendorff's α、留一法医师上限以及自提升与族群偏差量化；

**📊 数据集**

数据集为Ankizin v5中筛选的26,598条单闭合项，按词汇匹配与难度分层抽样得到3,800条评测集，其中200条为9人共评一致子集；

**📈 对比分析**

在留一法医师上限比较中，Gemini 3 Flash的κ≈0.694与医师上限0.709接近，其他模型均低于上限；自动评判者在“弃权”率上几乎为0，缺乏医生的难度调节；

**⚠️ 局限性**

局限包括高置信区间导致结论不稳固、评审者规模有限、可能存在模型训练数据泄漏、缺乏自我不确定性机制以及家族偏差导致评判不独立；

---

## 487. RoboWorld: Fast and Reliable Neural Simulators for Generalist Robot Policy Evaluation

**arXiv ID:** 2607.01060 | [PDF](https://arxiv.org/pdf/2607.01060v1)

**作者:** Byeongguk Jeon `[一作]` (KAIST), Kimin Lee `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一套可扩展的自动化评估管线，利用快速自回归视频世界模型和任务进度感知的视觉语言模型（VLM）判定器，能够在无真实机器人或人工干预的情况下评估通用机器人策略。

**💡 创新点**

主要创新在于提出 Self‑Forwarded Prior 去噪训练方法（结合锚点步骤）以显著减小训练‑测试上下文误差，并设计多视角任务进度评判 rubric，使评估能够区分策略失败与世界模型错误；同时通过关键‑值缓存实现高效的少步推理。

**🔧 技术方法**

技术方面采用了基于 rectified flow 的自回归视频扩散模型、帧级因果注意力、动作 MLP 注入、每帧独立噪声调度、KV 缓存滑动窗口以及 GPT‑4o 作为 VLM 判定器。

**📊 数据集**

训练数据来自 DROID 大规模真实机器人操作数据集，评估以 RoboArena 真实世界基准和 BAIR Robot Pushing 小规模实验为主；对极端环境的迁移评估还使用了从 RoboArena 采样的 175 条初始观测通过图像编辑生成的 8 种合成环境。

**📈 对比分析**

与 RoboArena 真实评估对比，论文方法在 8 条策略上实现了 Pearson r=0.989、Spearman ρ=0.970 的极高相关性；在速度‑质量权衡上，相比 Ctrl‑World、PersistWorld 等基线，4 步去噪方案达到 15.3 FPS 的最高帧率，同时在 LPIPS/SSIM/FVD 上表现最佳；消融实验表明 Self‑Forwarded Prior 与锚点步骤是提升生成稳定性和评估可靠性的关键。

**⚠️ 局限性**

局限性在于长时 horizon 的抓取/操作仍难以保持物体一致性，受限于训练数据覆盖度；需要更大规模的人机交互与物体接触数据来进一步提升模型在复杂、极端环境下的评估准确性。

---

## 488. GenAU: Language-Grounded Industrial Anomaly Understanding with Vision-Language Models

**arXiv ID:** 2607.01049 | [PDF](https://arxiv.org/pdf/2607.01049v1)

**作者:** Hongkuan Zhou `[一作]` (Robert Bosch GmbH), Steffen Staab `[通讯]` (University of Stuttgart)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出GenAU，一种统一的指令式视觉‑语言框架，能在工业异常检测中同时完成二元异常判定、像素级分割、多类型异常识别和缺陷原因解释。

**💡 创新点**

创新点在于将两种分割标记〈defect〉和〈normal〉嵌入自回归VLM的隐藏层，作为语言‑视觉锚点；通过与多尺度视觉特征的对齐实现像素级分割，并在同一模型中实现检测、分割、分类与自然语言推理；同时引入统一的训练目标与指令混合，保持检测性能的同时获得推理能力。

**🔧 技术方法**

使用大型预训练的指令跟随视觉‑语言模型（LLaVA‑OneVision），冻结SigLIP视觉编码器并对其进行跨模态投影；在解码器上添加两种分割标记，使用LoRA对解码器、标记嵌入、投影头等进行微调；训练目标包括语言模型交叉熵和多阶段分割损失（focal、Tversky、Dice）。

**📊 数据集**

在MVTec‑AD上进行训练，跨数据集评估于VisA、MPDD、Real‑IAD（检测/分割）以及VisA‑D&R（推理）。

**📈 对比分析**

与CLIP‑基零样本检测/分割方法（如AnomalyCLIP、MultiADS）以及推理专用模型Anomaly‑OV进行对比；GenAU在VisA上实现最高的图像级AUROC（87.6%），在MPDD和Real‑IAD的图像级AUROC也具备竞争力；像素级AUROC接近甚至略低于最强CLIP分割器；在多类型检测上，GenAU的F1在VisA和MPDD分别为29%和70%，均优于MultiADS；在推理得分上，GenAU的GPT‑Score（低级感知/复杂推理）均超越Anomaly‑OV。

**⚠️ 局限性**

局部纹理缺陷检测效果好，但对全局逻辑缺陷（如缺失部件）分割不佳；多类型分类准确率低，受限于少量训练样本和缺乏结构先验；在跨域分割上仍落后于专门的CLIP‑分割器；缺陷解释仍受限于VLM先前知识，需要进一步的领域知识或少样本适配。

---

## 489. EchoRisk: A Multicentre Echocardiography Dataset and Benchmark for Cardio-Oncology

**arXiv ID:** 2607.01039 | [PDF](https://arxiv.org/pdf/2607.01039v1)

**作者:** Grigorios Kalliatakis `[一作]` (Foundation for Research and Technology Hellas), Kostas Marias `[通讯]` (Foundation for Research and Technology Hellas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本研究设计并发布了EchoRisk数据集及其基准，包含多中心、纵向的心脏超声视频与治疗相关心脏毒性标签，并定义了三项临床任务；同时提供基准模型与评估框架；

**💡 创新点**

创新点在于首个公开的多中心、纵向心脏超声数据集并配备治疗诱导心脏毒性标签，定义了针对左室射血分数估计、左室功能障碍分类以及基线超声预测心脏毒性三大任务，为心脏肿瘤学领域提供统一评估标准；

**🔧 技术方法**

采用R2+1D ResNet‑18预训练骨干网络与LSTM聚合，结合分数采样、双视图融合、数据增强与TTA，并使用均方误差、AUC‑ROC、Focal Loss等损失函数与评估指标；

**📊 数据集**

使用EchoRisk v1.0数据集：422名患者、2159段超声视频（1123次检查）用于Task 1/2；280名患者、560段视频（仅基线）用于Task 3；涵盖5个欧洲中心、4种设备，真实临床图像质量与多视角；

**📈 对比分析**

通过公开评估脚本与排行榜，对基准模型进行8个随机种子训练，Task 1 MAE≈4.98 pp（比EchoNet‑Dynamic相近），Task 2 AUC≈0.849，Task 3最优基准双视图+TTA AUC≈0.541，临床参考模型AUC≈0.525，表明Task 3预测仍接近随机；

**⚠️ 局限性**

主要限制包括Task 3预测仍低于临床需求、样本量有限导致模型泛化差、缺乏完整临床变量、图像质量与设备差异导致噪声、以及未能捕捉长程随访信息和治疗方案等多因素影响。

---

## 490. AMBUSH: Collaborative Capture in Complex Environments with Neural Acceleration

**arXiv ID:** 2607.01029 | [PDF](https://arxiv.org/pdf/2607.01029v1)

**作者:** Junfeng Chen `[一作]` (Peking University), Meng Guo `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

提出一种基于伏击策略的多机器人动态捕获框架，通过拓扑分析生成隐藏与攻击位置，并利用混合蒙特卡洛树搜索与离线训练的图神经网络进行参数优化，实现在复杂环境中由慢速追踪者捕获快逃者。

**💡 创新点**

创新点包括：① 仅用伏击策略即可在障碍密集场景中捕获快逃者；② 设计离散/连续参数的混合MCTS优化角色分配与运动控制；③ 离线训练的图神经网络预测器替代昂贵的rollout，加速搜索并保持高成功率。

**🔧 技术方法**

使用混合蒙特卡洛树搜索（H‑MCTS）、图神经网络（NARE）、可见性图与拓扑分析、离散成本地图、ORCA动态避碰以及强化学习框架。

**📊 数据集**

在45张训练地图和5张验证地图（随机障碍布局）上进行仿真，使用不同逃逸者速度、捕获半径的设置，并在硬件实验中测试无人机/UGV在室内/户外场景。

**📈 对比分析**

与纯强化学习、可见性图策略、传统MCTS以及封闭式围捕基线对比，实验显示捕获率提升约25%，平均捕获时间缩短约17%，且能成功捕获速度比为2:1及人类操控逃逸者。

**⚠️ 局限性**

局限性包括：假设环境完全已知且通信中心化；仅针对单个逃逸者与同质追踪者；未考虑部分可观测或分布式执行；在极度稠密障碍或多逃逸者情形下性能可能下降。

---

## 491. CausalMix: Data Mixture as Causal Inference for Language Model Training

**arXiv ID:** 2607.01104 | [PDF](https://arxiv.org/pdf/2607.01104v1)

**作者:** Zinan Tang `[一作]` (Tsinghua University), Biqing Huang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将LLM训练数据混合优化视为因果边际收益估计问题，通过历史代理训练运行中的数据状态协变量和混合比例作为处理，构建可解释、可迁移的混合策略；

**💡 创新点**

创新点在于：①将数据混合问题从静态验证损失最小化转为状态条件因果估计，②使用双机学习(DML)实现处理与结果的正交化，消除数据质量混淆；③设计可解释的CATE解释器揭示知识注入与逻辑推理之间的“技能冲突”；

**🔧 技术方法**

技术手段包括：双机学习(DML)、因果森林、对数混合转化、保守信任区策略、搜索式混合提取；

**📊 数据集**

使用Tulu 3数据集中的5个域（Coding、Instruction Following、Math Reasoning、Knowledge Recall、Safety & Non‑Compliance），从512个100K子数据集构建代理实验；

**📈 对比分析**

与Grid、RegMix、DoReMi、ODM、DMO等基线对比，在多规模（100K–800K）和模型大小（0.5B–7B）下，本文方法在Avg_Dev和Unseen集上均取得更高分数，且保持跨规模一致性；

**⚠️ 局限性**

局限性：①依赖足够的历史代理运行以估计因果效应，数据量有限可能导致高维协变量下估计不稳；②对隐藏混淆因素假设要求较高；③在极大规模数据或更复杂域配置时，计算成本和模型可解释性仍待进一步提升。

---

## 492. Foundation Models vs. Radiomics for Lung Computed Tomography: A Benchmark of Feature Extractors, Classification Heads, and Segmentation Choices

**arXiv ID:** 2607.01001 | [PDF](https://arxiv.org/pdf/2607.01001v1)

**作者:** Nils Neukirch `[一作]` (Carl von Ossietzky University of Oldenburg), Nils Strodthoff `[通讯]` (Carl von Ossietzky University of Oldenburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过系统化因子化基准，比较了在肺癌CT影像中基于Radiomics和Foundation Models的特征提取、分类头与分割策略，评估其在跨队列推广中的鲁棒性与性能。

**💡 创新点**

创新点在于（1）首次将特征提取器、分类头和分割选择三大设计轴进行全组合实验；（2）采用跨队列（LUNG1、LUNG2）worst‑case性能作为评价指标，突出模型的泛化能力；（3）提供安全默认配置和无肿瘤掩膜备选方案。

**🔧 技术方法**

使用的技术包括：CT专用自监督预训练模型Curia/Curia‑2、通用视觉模型DINOv3；传统Radiomics（2D/3D）特征；多种分类/回归头（TabPFN、TabICL、XGBoost、CatBoost、Random Forest、Logistic Regression、Ridge）；以及多种分割方案（无掩膜、肺掩膜、肿瘤掩膜）。

**📊 数据集**

采用公开的LUNG1（NSCLC‑Radiomics）和LUNG2（NSCLC‑Radiogenomics）两大CT数据集，共计629名患者，用于训练、内部测试与外部验证。

**📈 对比分析**

在五个任务（肿瘤体积/分期、2‑年生存、组织学、年龄）上，最优配置在三大临床任务中平均排名最佳；Radiomics在肿瘤体积/分期任务中占优，而Curia与Curia‑2在生存与组织学任务中表现相近；整体性能表现为：肿瘤分期AUC≈0.668、2‑年生存AUC≈0.543、组织学AUC≈0.582。

**⚠️ 局限性**

局限性包括仅评估两队列，缺乏不同机构/扫描仪的验证；LUNG2样本量与标签不平衡导致统计功效有限；未对不同分割方法的可靠性进行深入分析；以及肿瘤体积任务存在循环偏差，可能对结果产生偏倚。

---

## 493. AutoRestTest at the SBFT 2026 Tool Competition

**arXiv ID:** 2607.01063 | [PDF](https://arxiv.org/pdf/2607.01063v1)

**作者:** Tyler Stennett `[一作]` (Georgia Institute of Technology), Alessandro Orso `[通讯]` (University of Georgia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为 AutoRestTest 的黑盒 REST API 测试工具，能够在大规模输入空间和复杂交互依赖下自动探索并发现错误。

**💡 创新点**

创新点包括：① 构建语义属性依赖图（SPDG）以捕捉不同 API 操作间的参数和响应依赖；② 采用四个专门化的多智能体强化学习（MARL）框架，动态调优请求策略；③ 使用大型语言模型预生成域感知输入，并在测试期间进行多轮验证和微调。

**🔧 技术方法**

技术栈包括：OpenAPI 规范解析、GloVe 词嵌入、强化学习（epsilon‑greedy、Q‑表）、Gemini 2.5 Flash Lite LLM、Python/REST 测试框架和请求/响应变异器。

**📊 数据集**

数据集为 11 个真实世界的 REST API，总计 317 个操作（约每 API 29 个），用于 SBFT 2026 REST League 竞赛的标准化基准。

**📈 对比分析**

与 CATS、EvoMaster、RESTest、RestTestGen 和 Schemathesis 等五款顶尖工具对比，AutoRestTest 在所有三类评估指标（故障检测、总体效率、总体有效性）中排名第一；平均每个 API 检测到 67.09 个唯一 5xx 错误，操作覆盖率 17.27，故障检测 AUC 为 151,490，且成本仅约 0.02 美元/次。

**⚠️ 局限性**

局限性：在某些 API 上出现因隐式 UTF‑8 编码假设导致的解析崩溃；对多种响应编码的鲁棒性尚未完善；LLM 生成值的多样性和质量仍受模型温度和迭代次数限制。

---

## 494. Agentic generation of verifiable rules for deterministic, self-expanding reaction classification

**arXiv ID:** 2607.01061 | [PDF](https://arxiv.org/pdf/2607.01061v1)

**作者:** Daniel Armstrong `[一作]` (École Polytechnique Fédérale de Lausanne), Philippe Schwaller `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个自动化多代理LLM框架，对USPTO专利反应数据进行分类并生成可执行的SMIRKS规则，形成可持续扩展的反应分类数据库。

**💡 创新点**

创新点在于利用LLM自动生成和验证反应规则，实现从68个种类扩展到14073类的自适应分类体系，且无人工干预；通过验证循环保证规则质量；同时提供第一匹配确定性分类器与LLM回退机制。

**🔧 技术方法**

技术包括多代理LLM（Gemini 3）、链式验证、聚合器、模板验证循环、SMIRKS生成与迭代精炼、误差检测（cleanlab）、MDP‑MLP判别器、SMIRKS排序算法（反馈弧最小化）等。

**📊 数据集**

主要使用美国专利局（USPTO）1976‑2016年约1.8M条反应，筛选后约860k条；外部评测集为CRD‑2025学术反应9,296条和RingBreaker环化反应57,234条。

**📈 对比分析**

与现有NameRXN、Rxn‑INSIGHT对比，标注噪声低于0.33%，整体分类准确率≈97.9%（Hybrid strict）/85.9%（Generalised SMIRKS）；在OOV集上覆盖率分别为80.6%/77.3%，并且在三层分类下与NameRXN实现相当甚至更精细，推理延迟≈5–17 ms。

**⚠️ 局限性**

局限性包括对初始层级结构的依赖；对极少量或全新化学未出现于训练集的反应仍需LLM回退；生成SMIRKS的精度受LLM生成质量影响，且对非药物合成、流程化学、电化学等特殊领域的泛化仍需额外训练。

---

## 495. Robots Ask the Way: Communication-Enabled Social Navigation

**arXiv ID:** 2607.01044 | [PDF](https://arxiv.org/pdf/2607.01044v1)

**作者:** Valentino Sacco `[一作]` (Sapienza University of Rome), Fabio Galasso `[通讯]` (Sapienza University of Rome)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多智能体室内环境中，本文提出了CommNav框架，让机器人主动向人类询问并利用获得的空间线索定位目标个体，从而实现更高效的社交导航。

**💡 创新点**

创新点在于首次将主动信息请求与自然语言/结构化交互结合进导航策略，并通过预训练的COMM模块克服了稀疏通信信号的难题。

**🔧 技术方法**

使用技术包括基于DDPPO的深度强化学习、COMM模块的轨迹编码器+空间-时间MLP、LLM（QWEN3-8B）生成自然语言指令、BERT编码器、Habitat 3.0c模拟器。

**📊 数据集**

数据集主要来自Habitat 3.0c的多人模拟环境，包含约240万条通信实例、7000条LLM生成指令以及20名受试者提供的自然语言实验数据。

**📈 对比分析**

通过与DDPPO、SDA等基线在Habitat 3.0c上对比，CommNav在Episode Success上提升约10个百分点（0.24 vs 0.14），Finding Success提升8个百分点，且在自然语言指令下性能与结构化数据相近。

**⚠️ 局限性**

局限性包括仅在无噪声仿真环境中验证，假设完美的感知与定位；只研究单轮交互，未覆盖多轮对话；以及缺乏真实机器人部署与隐私安全保障。

---

## 496. DART-VLN: Test-Time Memory Decay and Anti-Loop Regularization for Discrete Vision-Language Navigation

**arXiv ID:** 2607.01043 | [PDF](https://arxiv.org/pdf/2607.01043v1)

**作者:** Shaoheng Zhang `[一作]` (Harbin Institute Of Technology), Jie Mei `[通讯]` (Harbin Institute Of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关的测试时控制框架DART-VLN，通过读侧记忆衰减与反循环正则化，在冻结模型参数的前提下提升离散视觉语言导航的性能与效率。

**💡 创新点**

创新点在于将记忆衰减和反循环正则化两种轻量级、无新可学习参数的机制融合到推理流程中，实现对历史证据和局部回溯行为的即时调节。

**🔧 技术方法**

使用了读侧记忆重加权、基于下一跳的回退惩罚、GridMM离散导航框架以及标准VLN评估指标（TL、NE、SR、SPL、RGSPL等）。

**📊 数据集**

在Room-to-Room（R2R）和REVERIE两个离散视觉语言导航基准上进行实验。

**📈 对比分析**

与GridMM基线及其写侧更新变体对比，decay-only在保留质量的同时降低了运行时间；decay+anti-loop在两组数据集上实现了更短轨迹、降低运行时间、提高SR/SPL/RGSPL的最佳综合效果。

**⚠️ 局限性**

仅针对离散VLN的GridMM风格背骨，改进主要体现在局部回溯和效率上，未必能直接推广到更复杂或连续导航任务，且写侧更新等更激进手段效果不稳定。

---

## 497. Identifying Effective Program Comprehension Strategies through Gaze Transitions over Syntactic Elements

**arXiv ID:** 2607.01042 | [PDF](https://arxiv.org/pdf/2607.01042v1)

**作者:** Kyogo Horikawa `[一作]` (National Institute of Technology(KOSEN)), Haruhiko Yoshioka `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

将眼动追踪数据转换为抽象语法树（AST）节点间的转移，并分析循环语句和方法之间的视线转移模式，以区分任务成功与失败。

**💡 创新点**

创新点在于：①提出坐标到语法节点的映射方法，将屏幕坐标的眼动数据映射到语法结构；②利用频繁序列挖掘（cSPADE）在语法节点层面提取视线转移模式；③将成功与失败参与者的转移模式进行对比，发现循环元素和方法调用的系统性阅读策略与任务成功相关。

**🔧 技术方法**

使用眼动追踪仪（Tobii Eye Tracker 4C）采集的坐标数据，配合源代码解析得到AST；通过坐标-语法映射模块将眼动点映射为AST节点；利用cSPADE算法挖掘频繁的视线转移序列；统计支持度并做卡方检验。

**📊 数据集**

数据集来自 Ishida 等人的眼动实验，共 16 名学生完成 16 份 Java 程序阅读任务（共 256 任务，剔除错误后 200 任务）。任务分为低难度（单方法、单层循环）和高难度（多方法/递归）。

**📈 对比分析**

将任务结果按成功/失败划分，计算每组中转移序列的平均支持度；提取支持度差 ≥ 0.1 的序列；对比两组的差异并进行统计检验。结果显示：成功组在循环语句中更频繁地查看初始化、条件与更新，并在方法间按调用关系转移；失败组则缺少此类系统性转移。整体来看，成功组的视线转移模式更接近程序的执行流。

**⚠️ 局限性**

局限性包括：样本规模仅 16 人，且任务多为简单的 Java 示例，难以覆盖更复杂的控制流和真实项目；仅分析了 for 循环的头部元素，未考虑循环体及更细粒度的语法结构；未对眼动特征（如注视时长、回溯等）进行深入多变量建模。

---

## 498. As It Was: Aligning LLM Search Evaluation with Historical User Preferences

**arXiv ID:** 2607.01040 | [PDF](https://arxiv.org/pdf/2607.01040v1)

**作者:** Ali Vardasbi `[一作]` (Spotify), Mounia Lalmas `[通讯]` (Spotify)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种将用户行为信息作为轻量级可审计证据与大型语言模型（LLM）评估器相结合的行为根据信号，用以评估音乐搜索结果页面的相关性。

**💡 创新点**

创新点在于将查询‑相关性‑印象（QRI）卡片嵌入LLM判定上下文，既利用语义推理，又加入历史交互数据作为可审计的先验，提升与真实用户偏好的一致性。

**🔧 技术方法**

使用大规模语言模型（OpenAI API）、逆向倾向打分（IPS）去偏见的交互信号、查询相似度过滤以及QRI卡片的提示工程。

**📊 数据集**

使用Spotify音乐搜索日志（约6k SERP）、日志派生的相关性数据、多语言人工标注（265条）以及一周的A/B实验数据。

**📈 对比分析**

与无行为根据信号的纯语义LLM判定比较，使用Spearman、Kendall和A/B结果对齐度等指标，行为根据信号的判定在日志相关性上提升约5% Spearman，人工评估提升约15%，与A/B实验对齐率提高约6%点。

**⚠️ 局限性**

局限性在于对用户行为数据的依赖仍受采样偏差和曝光偏差限制，整体对齐度仍中等，且对冷启动/长尾查询支持有限。

---

## 499. Multiwinner Voting with Spatial Preferences under Incomplete Information

**arXiv ID:** 2607.01036 | [PDF](https://arxiv.org/pdf/2607.01036v1)

**作者:** Drew Springham `[一作]` (King's College London), Maria Polukarov `[通讯]` (King's College London)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出一种可在多候选人多选举中，仅用极少的投票者信息（每人平均d log d k次查询）即可保证强比例代表性的框架，结合空间（轴对齐矩形）偏好与随机矩形投票模型，最终得到EJR+委员会。

**💡 创新点**

创新点在于：1）构造了一个维度不敏感的 verify‑or‑fallback 框架，条件满足时查询成本与候选人数无关；2）设计了可插拔的模块（PGJCR、NGJCR、网格/网格估计），在已知分布、未知分布和光滑（Lipschitz）分布下分别实现高效的委员会选择与边际分布估计；3）在未知分布的光滑情况下将所需选民规模从多项式降至仅对 m 的 log log m 级别。

**🔧 技术方法**

技术包括：基于空间模型的轴对齐矩形投票（ARRV）与查询模型；利用分位数选择（Quantile Selection）和二分搜索构造外括号；使用 Hoeffding 与 DKW 不等式对样本误差做概率控制；对分布估计使用批量估计（batch estimator）与网格插值；对未知分布采用无噪声 GJCR 与带噪声的 NGJCR；对光滑分布采用候选网格与 CDF 网格估计。

**📊 数据集**

本文主要为理论分析，未使用公开数据集；实验或实证验证未在摘要中给出，主要通过数学证明和概率上界进行说明。

**📈 对比分析**

与传统的全信息 EJR+ 规则（如 GJCR）相比，本文在查询量上实现了显著下降：从 O(d log m) 降到 O(d log d k)，且只在极少数投票者上进行完整信息收集；在未知分布的光滑假设下，整体投票者规模与候选人数的关系从多项式降到对 log log m 的依赖。

**⚠️ 局限性**

局限性包括：1）在未知分布且不光滑时，需要较大选民规模（O(m log m)）才能保证低查询成本；2）对分布的光滑性（Lipschitz）假设在实际应用中可能不成立；3）框架依赖于对每维边际分布的准确估计，若估计误差较大则退回完整查询；4）目前仅针对轴对齐矩形偏好，难以直接推广到更一般的几何偏好形式。

---

## 500. SuperFlex: Deformable Superquadrics for Point Cloud Decomposition

**arXiv ID:** 2607.01015 | [PDF](https://arxiv.org/pdf/2607.01015v1)

**作者:** Gabriel Tavernini `[一作]` (ETH Zurich), Francis Engelmann `[通讯]` (USI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于可变形超四边形的点云分解方法，能够从完整或部分点云中预测高精度、结构化的几何体。

**💡 创新点**

创新点包括：①引入弯曲与锥形变形提升单个原语的表达力；②设计联合体积与表面可微损失，显著提升重建精度；③通过测试时优化与伪标签微调，使模型对遮挡和噪声具有鲁棒性。

**🔧 技术方法**

使用 Transformer‑based 解码器、PVCNN 点云编码器、可微 IoU+SDF 损失、LogSumExp 近似以及自监督集合匹配训练策略；同时在测试时对预测参数做梯度优化。

**📊 数据集**

主要数据集：ShapeNet（完整对象）、Aria Synthetic Environments（带遮挡的室内场景）以及 ScanNet++（真实扫描）用于评估鲁棒性。

**📈 对比分析**

与 SQ、CSA、SuperDec、EMS、Marching Primitives 等基线相比，实验显示在 IoU、F‑score、Chamfer 距离上取得明显提升（例如 IoU 从 0.59 提升到 0.72；在遮挡场景中 IoU 提升至 0.54），且仅使用 5–6 个原语即可完成，速度保持在毫秒级，远快于优化方法。

**⚠️ 局限性**

局限性包括：①对极端形状（如高细节或极不规则物体）仍可能存在细节缺失；②模型对极大遮挡或极端噪声的鲁棒性尚待进一步验证；③目前仅实现了单轴锥形与三轴弯曲，仍可探索更丰富的变形模式。

---

## 501. Generative Model Proposal based Particle Filtering for Data Assimilation

**arXiv ID:** 2607.01012 | [PDF](https://arxiv.org/pdf/2607.01012v1)

**作者:** Chandni Nagda `[一作]` (University of Illinois at Urbana-Champaign), Arindam Banerjee `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 Flow Proposal Particle Filter（FPPF），一种在 SIR 框架下使用可训练的条件流提议分布的粒子滤波方法，保持贝叶斯更新且无需线性高斯假设。

**💡 创新点**

创新点在于利用条件流匹配学习可逆、可密度评估的观察条件提议，并通过局部化（L‑FPPF）将提议分解为块级别，显著降低高维粒子滤波的维数灾难。

**🔧 技术方法**

核心技术包括条件流匹配、即时变换公式（即时求解密度）、Hutchinson 跟踪估计、块化重采样以及基于窗口的局部速度网络。

**📊 数据集**

实验使用仿真轨迹训练，涉及 Lorenz‑63、可变维度 Lorenz‑96、以及 Kuramoto‑Sivashinsky PDE，覆盖非线性、非高斯以及长时序稀疏观测场景。

**📈 对比分析**

与传统 BPF、APF、EnKF、NASMC、InfNN、FlowDAS 等基线比较，FPPF 在 RMSE、CRPS 与 ESS 指标上均显著优于对手；在高维 Lorenz‑96 上 L‑FPPF 保持稳定，性能几乎不随维度提升而退化。

**⚠️ 局限性**

局限性包括：需要已知的转移密度；每步需多次 ODE 求解导致计算成本高；Hutchinson 估计在极高维时可能引入方差；对非结构化、稀疏或多源观测的适应性仍需进一步研究。

---

## 502. Understanding Large Language Models

**arXiv ID:** 2607.01006 | [PDF](https://arxiv.org/pdf/2607.01006v1)

**作者:** Yannik Keller `[一作]`, Thomas Eisenmann `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文综述了大型语言模型（LLM）的Transformer架构、产生的认知能力、可解释AI方法，并讨论了LLM是否具备真正理解的争论。

**💡 创新点**

创新点在于系统性梳理LLM出现式认知功能与人类认知对比，指出认知与LLM理解差异的两大误区，并提出更细致的理解框架。

**🔧 技术方法**

主要技术包括Transformer注意力机制、无监督语言建模、指令调优、强化学习（RLHF）、神经元激活分析、线性探测与回路分析等可解释AI方法。

**📊 数据集**

使用的数据集涵盖公开基准（SQuAD、GLUE、SuperGLUE、CoQA、IMO、ToM任务集、Deception案例）以及自行构造的ToM与Deception评测集。

**📈 对比分析**

通过与人类表现和历史模型的对比，LLM在语言推理、符号推理、Theory‑of‑Mind、Deception等任务上可与人类相当或超越；但在对抗性改写的ToM测试中性能显著下降。

**⚠️ 局限性**

局限在于可解释方法仅适用于有限提示/层，缺乏完整系统性解释；LLM对任务的泛化受训练数据泄漏、规模与调优方法的限制。

---

## 503. Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads

**arXiv ID:** 2607.01002 | [PDF](https://arxiv.org/pdf/2607.01002v1)

**作者:** Aryo Pradipta Gema `[一作]` (University of Edinburgh), Pasquale Minervini `[通讯]` (University of Edinburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将注意力头的输出值（OV）投影到答案词嵌入空间并与非针头位置进行空间对比，提出了一种写向量感知的检索头检测器，能够识别在非文字合成任务中起关键作用的头。

**💡 创新点**

创新点在于：① 关注输出值而非仅仅关注注意力权重，② 通过空间对比方法捕捉到非文字检索头；从而补充了传统基于词匹配的检测器无法识别的头。

**🔧 技术方法**

技术手段包括：Transformer头的QK/OV电路分解、答案词嵌入投影、空间对比评分、平均消融实验、对比注意力基准和底层抽样分析。

**📊 数据集**

使用的数据集包括：NoLiMa（非文字检索基准）、NIAH（文字检索基准）、MuSiQue、BABILong、PopQA、城市-国家关联、算术推理等。

**📈 对比分析**

与随机、Wu/NIAH、Wu/NoLiMa等注意力基准对比，平均消融顶k头后NoLiMa的ROUGE‑L几乎归零，且在MuSiQue和BABILong等下游长上下文任务中显著下降，表明检测器在检索特异性和性能上优于现有方法。

**⚠️ 局限性**

局限性包括：在存在相关干扰词的上下文中分数可能下降；实验仅涵盖标准Transformer，未涉及Mixture‑of‑Experts、编码-解码和状态空间等架构；需要进一步验证跨模型/跨任务的普适性。

---

## 504. SWE-Doctor: Guiding Software Engineering Agents with Runtime Diagnosis from Multi-Faceted Bug Reproduction Tests

**arXiv ID:** 2607.00990 | [PDF](https://arxiv.org/pdf/2607.00990v1)

**作者:** Yaoqi Guo `[一作]` (Nanyang Technology University), Zhenpeng Chen `[通讯]` (Tsinghua University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 SWE-Doctor，一种通过多面向Bug复现测试（BRT）与运行时诊断来指导软件工程代理进行补丁生成的系统。

**💡 创新点**

创新点在于：①生成多面向BRT以覆盖问题的不同行为要求；②将BRT执行结果转换为结构化的运行时诊断记录；③将诊断信息与定位信息结合，形成多源上下文指导补丁生成，显著减少部分补丁。

**🔧 技术方法**

使用的技术包括：大型语言模型（LLM）进行需求分解、BRT生成与调试交互；基于调试器（如 Python 的 PDB / Go 的 Delve）的运行时诊断；以及多源信息交叉验证的补丁生成策略。

**📊 数据集**

采用的数据集为：SWE-bench Verified（436 条 Python 错误修复实例）和 SWE-bench Pro（107 条 Python 错误修复实例），并在 Pro 的 Go 子集上进行跨语言验证。

**📈 对比分析**

与 mini‑SWE‑agent 与 live‑SWE‑agent 两个基准在 5 种 LLM 后端上进行对比，SWE-Doctor 在所有 10 个 LLM‑数据集组合中均表现最佳，平均提升 2.0–8.9 个百分点，尤以 Pro 任务上的 8.0–8.9 个百分点显著。

**⚠️ 局限性**

局限性包括：对 LLM 随机性与非确定性仍有影响；当前设计依赖于语言特定的调试器；评估主要集中在 Python（以及少量 Go），尚未覆盖更广泛的编程语言与更复杂的项目场景。

---

## 505. SAGE: Structured Agentic Graph Editing for Software Diagrams

**arXiv ID:** 2607.01102 | [PDF](https://arxiv.org/pdf/2607.01102v1)

**作者:** Tyler Sivertsen `[一作]` (Purdue University), James C. Davis `[通讯]` (Purdue University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一款基于浏览器的工具，可通过自然语言提示对 Draw.io 与 Mermaid 风格的软件工程图进行结构化编辑，支持导入、生成、手动编辑以及图像版的局部掩膜编辑。

**💡 创新点**

创新点在于采用混合架构：先把图形解析为可编辑的图模型，再将自然语言请求转化为明确的编辑意图；随后利用语言模型进行解释、目标解析，而实际图形变更由结构化、可验证的图操作完成，并在变更后进行 XML 校验与自动修复，最终将成功结果版本化。

**🔧 技术方法**

核心技术包括：Next.js/React/TypeScript 前后端实现；DiagramModel（图形结构化表示）与 ParsedEditIntent（编辑意图对象）；LLM（OpenAI/Gemini）用于自然语言解析和部分 XML 转换；图操作引擎、XML 验证与自动修复；Prisma+SQLite 存储会话、版本与元数据。

**📊 数据集**

主要使用的评估数据集为从 Kubernetes 官方文档中提取的集群架构图（含控制平面、节点、组件等），并以此生成可编辑的基准图；同时对该图进行两种编辑路径（结构化图编辑与掩膜图像编辑）的测试。

**📈 对比分析**

评价指标包括结构有效性（XML 可序列化并可重新加载）、编辑正确性（目标被准确修改）、结构保持性（未被修改的元素保持不变）以及拓扑一致性。实验结果显示：重构后图可重载，所有三条自然语言编辑均成功完成，保持与拓扑均无错误；图像编辑虽然实现了语义修改，但在多次编辑后出现可视化质量下降。

**⚠️ 局限性**

局限性包括：布局质量控制不足，生成图形可能出现重叠或排版不佳；语义验证有限，只能检测基本结构错误；对复杂领域语义或美观布局支持不佳；多次图像编辑易导致视觉衰退；缺乏事务级别的版本指针保护。

---

## 506. When Context Compensates for Sparse Event History: AlphaEarth for Spatio-Temporal Point-Process Forecasting

**arXiv ID:** 2607.01082 | [PDF](https://arxiv.org/pdf/2607.01082v1)

**作者:** Yahya Aalaila `[一作]` (German Research Center for Artificial Intelligence), Sebastian Vollmer `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对急救医疗服务（EMS）呼叫需求的空间-时间点过程模型进行研究，探讨在局部事件历史稀疏时，外部空间上下文（AlphaEarth嵌入）是否能提升预测性能。

**💡 创新点**

创新点在于：①在控制模型复杂度的前提下，通过与 AlphaEarth 嵌入的对比，系统评估外部空间上下文对空间迁移预测的具体增益；②证明当事件历史极短（1–2 周）时，外部上下文可实现 2–6 倍的预测密度提升；③在多地区多时间锚点的空间留存实验中验证该效应的普遍性。

**🔧 技术方法**

技术手段包括：使用固定的 Log-Gaussian Cox Process（LGCP）作为基线模型；将 AlphaEarth 64 维空间嵌入作为线性空间上下文；采用 PriorVAE 低维化 GP 先验、数值积分和随机变分推断；对不同历史窗口长度（1–104 周）和空间掩模进行持留评估；使用 ELPD、事件级对数密度比等指标量化性能。

**📊 数据集**

数据集：蒙哥马利县（Pennsylvania）2017–2020 年的 911 调度 EMS 呼叫数据，约 6 万起事件；AlphaEarth 嵌入为每个位置每年提供 64 维向量。

**📈 对比分析**

比较方法：在 8 个不重叠的空间掩模上，取 5 个预测锚点，对每个历史窗口长度进行模型训练并评估。结果显示：AE 版模型在所有掩模和历史长度下均优于事件仅模型；在极短历史（1–2 周）时，预测密度提升 2–6 倍，随着历史长度增加，提升幅度下降至 10–20% 但仍显著；后者通过 ELPD、密度比和空间场标准差等多指标验证。

**⚠️ 局限性**

局限性：①仅使用 ELPD 等概率指标，未直接评估对 EMS 运营决策（如响应时间、资源分配）的实质影响；②实验仅在单一县域完成，缺乏跨区域的验证；③未探讨 AlphaEarth 嵌入的因果或可解释性，亦未在更灵活的神经网络框架下验证其效果。

---

## 507. CPDDNet: Color-Polarization Denoising and Demosaicking Network

**arXiv ID:** 2607.01100 | [PDF](https://arxiv.org/pdf/2607.01100v1)

**作者:** Qihang Zhang `[一作]` (Institute of Science Tokyo), Masatoshi Okutomi `[通讯]` (Institute of Science Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了CPDDNet，一种联合去噪与去马赛克（去马赛克）用于CPFA传感器的统一框架

**💡 创新点**

创新点在于引入特征融合模块（GFM）在去噪与去马赛克之间共享低层特征，采用DN‑to‑DM设计并实现信息桥接，显著减少信息损失

**🔧 技术方法**

使用U‑Net骨干、双编码器分支、Gated Fusion、点卷积/深度卷积、YCbCr损失、两阶段训练策略等深度学习技术

**📊 数据集**

采用公开的CPFA去噪与去马赛克数据集（40个场景，最高噪声级别）进行训练与评估

**📈 对比分析**

与DM Only、DM‑to‑DN、DN‑to‑DM三类传统与深度学习方法对比，CPDDNet在PSNR/SSIM/MAE等指标上均表现优于基线，性能提升显著

**⚠️ 局限性**

目前仅在固定噪声水平下训练，缺乏对多噪声条件的泛化能力

---

## 508. ROSA: A Robotics Foundation Model Serving System for Robot Factories

**arXiv ID:** 2607.01088 | [PDF](https://arxiv.org/pdf/2607.01088v1)

**作者:** Wenqi Jiang `[一作]` (NVIDIA Research), Christos Kozyrakis `[通讯]` (NVIDIA Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在机器人工厂中实现了一套共享GPU池的机器人基础模型（RFM）服务系统，支持多模型流水线和全局调度；

**💡 创新点**

创新点在于提出了三大原则：共享服务器级基础设施、面向机器人任务的编程抽象与系统设计以及以工厂生产率为目标的调度策略；

**🔧 技术方法**

利用Ray Serve进行分布式调度，结合vLLM、PyTorch和JAX作为模型推理后端，并通过ILP+启发式算法实现资源分配与批量化；

**📊 数据集**

采用Frankia Panda真实机器人采集的观测序列构建合成大规模工作负载，另外还在真实机器人上做了pick‑and‑place实验；

**📈 对比分析**

与传统单机或单GPU按任务分配的基线相比，系统在相同8个H200 GPU上实现工厂生产率提升12.06倍，且在共享GPU池下比最优共享基线提升2.44倍；

**⚠️ 局限性**

局限性包括：对模型结构和推理速率的依赖性，调度对预先已知负载的假设，无法覆盖高度动态或突发请求模式，以及对更大规模GPU集群的可扩展性尚未验证。

---

## 509. Cheap Code, Costly Judgment: A Case Study on Governable Agentic Software Engineering

**arXiv ID:** 2607.01087 | [PDF](https://arxiv.org/pdf/2607.01087v1)

**作者:** James C. Davis `[一作]` (Purdue University), Kirsten A. Davis `[通讯]` (Purdue University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过12周的首人案例研究，单个软件工程师使用前沿AI编码代理（Claude）构建文档可访问性修复系统，并记录现场笔记和代码库变更，生成治理转化理论。

**💡 创新点**

提出“治理转化”中介过程模型，阐明在高速度AI实施下如何将观察到的失败转化为可持续治理机制，实现“廉价代码，高成本判断”的过程。

**🔧 技术方法**

使用Anthropic Claude（Claude Max 20x）与VSCode插件交互，构建代理治理环境，包括动态上下文注入、类型系统、静态/动态检查、lint、验证器、治理文档等技术。

**📊 数据集**

使用88条现场笔记、约1.6百万行代码的仓库历史、部署数据、设计记录和审计日志等实证数据集。

**📈 对比分析**

通过对比治理前后提交量、改动量、成功率等指标，展示治理机制累积后仍保持甚至提升实现速度，错误率下降；未提供传统基准对比，但提出可量化指标和假设。

**⚠️ 局限性**

仅涉及单一研究对象与单一模型，首人案例易受回顾偏差影响，结果可能缺乏普适性；对模型能力提升的适用性有限；缺乏多项目、多团队的验证。

---

## 510. Message Passing Enables Efficient Reasoning

**arXiv ID:** 2607.01077 | [PDF](https://arxiv.org/pdf/2607.01077v1)

**作者:** Xuecheng Liu `[一作]` (Carnegie Mellon University), Andrea Zanette `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MPLM框架，让LLM通过spawn、send、recv、stop四种指令实现多线程消息传递，以分布式方式推理。

**💡 创新点**

创新点在于把MPI式点对点通信和预emption嵌入LLM推理中，减少上下文占用并提高并行效率。

**🔧 技术方法**

使用LLM自回归生成指令、批量推理、监督微调以及Qwen3等预训练模型。

**📊 数据集**

数据集包括生成的Sudoku（4×4至25×25）、3‑SAT随机实例、LongBench‑v2长文本问答。

**📈 对比分析**

与Serial CoT、Fork‑Join以及GPT‑5 Pro等基线比较，MPLM在Sudoku 25×25准确率72%、3‑SAT平均延迟更低、LongBench‑v2在未微调的提示下准确率提升约8%且延迟减少约1.7×。

**⚠️ 局限性**

局限在于通信模式需先验或额外训练，难以自动发现最佳通信策略，且对开放式任务的适用性尚未验证。

---

## 511. MemSyco-Bench: Benchmarking Sycophancy in Agent Memory

**arXiv ID:** 2607.01071 | [PDF](https://arxiv.org/pdf/2607.01071v1)

**作者:** Zhishang Xiang `[一作]` (Xiamen University), Jinsong Su `[通讯]` (Xiamen University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了MemSyco-Bench，一套评估长期记忆对LLM代理导致的“记忆诱导顺从”风险的基准；通过定义五类任务（客观事实判断、上下文范围控制、记忆-证据冲突、有效记忆选择、个性化记忆使用）检验代理在检索后是否正确抑制、更新或利用记忆。

**💡 创新点**

创新点在于：①首次将记忆诱导顺从从单轮交互扩展到跨轮长期记忆；②通过“记忆决策模式”将基准从检索正确率转向后检索决策合理性；③构建五个覆盖不同记忆使用情境的子任务，系统地捕捉记忆误用导致的错误。

**🔧 技术方法**

技术上结合现有的LLM（Qwen3-8B、DeepSeek-V4-Flash）与多种记忆框架（NaiveRAG、Mem0、A-Mem、LightMem、MemGPT、MemoryBank、SuperMemory）进行实验；采用自定义的记忆决策schema、对话模拟和多阶段质量验证；评估指标包括准确率、顺从率、正确记忆使用率与旧记忆错误率。

**📊 数据集**

数据集为MemSyco-Bench本身，按上述五类任务构造的自然对话实例；此外使用公开的长期记忆基准（LongMemEval、LoCoMo、STALE、PersonaMem）做对比分析。

**📈 对比分析**

比较方法：在每个任务上对比“无记忆”“全对话记忆”与不同记忆系统的表现。实验结果显示：大多数记忆系统在客观事实与冲突任务上往往导致准确率下降、顺从率上升；在个性化任务上能提升正确记忆使用但仍易被旧记忆误导。性能差异表明，仅提升检索准确率并不能消除记忆诱导顺从。

**⚠️ 局限性**

局限性：①基准样例主要基于人工构造，可能与真实多轮交互场景存在偏差；②评估聚焦于记忆决策而非完整代理行为，未考虑策略搜索、对话管理等因素；③只涉及两款LLM与七种记忆实现，未覆盖更广泛的模型与框架；④缺乏动态记忆更新与多模态记忆的实验。

---

## 512. GSRQ: Gain-Shape Residual Quantization for Sub-1-bit KV Cache

**arXiv ID:** 2607.01065 | [PDF](https://arxiv.org/pdf/2607.01065v1)

**作者:** Soosung Kim `[一作]` (Yonsei University), Jaeyong Chung `[通讯]` (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新型 KV 缓存压缩方法 GSRQ，利用 GSKM 替代传统 K‑means 以提升低位率下的量化效果。

**💡 创新点**

创新点包括：① 发现并解决高维欧氏 K‑means 中的中心收缩问题，提出方向‑幅度分离的 GSKM；② 将 GSKM 与梯度加权、对数平滑结合，在残差量化 (RQ) 中实现更高效的向量量化。

**🔧 技术方法**

使用技术包括：高维向量量化、残差量化（RQ）与子空间量化（PQ）的混合；方向‑幅度 K‑means (GSKM)；梯度加权与对数平滑的权重机制；以及与 PQ/RQ 组合的 GSRQ。

**📊 数据集**

实验数据集涵盖 LLaMA‑2‑7B、LLaMA‑3‑8B、Mistral‑7B、Qwen3‑8B 的 KV 缓存；WikiText‑2、C4、LongBench、GSM8K、MATH‑500、RULER、AIME 等多种评测基准。

**📈 对比分析**

与 CQ、AnTKV、KIVI、VQLLM 等主流 KV 缓存量化基线对比，GSRQ 在 0.75‑bit 甚至 0.375‑bit 下均能实现 10–30% 的 PPL 降低，LongBench 准确率提升 20–30%，并在 1‑bit 方案下超越 VQLLM 的 1‑bit 性能。

**⚠️ 局限性**

局限性包括：需进一步验证在更大模型或更高分辨率下的硬件实现；梯度权重对超参数 λ 仍敏感；极低位率（<0.375‑bit）或极大 KV 缓存规模时性能下降；目前主要针对 LLM 的 KV 缓存，泛化到其他模型需进一步研究。

---

## 513. Compression of Polyconvex Envelopes of Isotropic Functions via Monotonic Input Convex Neural Networks

**arXiv ID:** 2607.01055 | [PDF](https://arxiv.org/pdf/2607.01055v1)

**作者:** Timo Neumeier `[一作]` (University of Augsburg), Julian Salmon `[通讯]` (University of Augsburg)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于Ball充分条件的神经网络压缩框架，用输入凸神经网络（ICNN）对三维Saint Venant–Kirchhoff能量的多重共形包络进行压缩，并在正奇异值空间上实现结构保持。

**💡 创新点**

创新点在于：①通过将判定域从带符号奇异值空间降到正象限，显著减少训练数据量；②利用足够的Ball判定而非必要充分判定，仅需满足凸性与单调性即可；③在ICNN中同时施加非负权重和单调性约束，天然满足多重共形性与物理不等式。

**🔧 技术方法**

技术实现包括：输入凸神经网络（ICNN）、Softplus激活、非负权重投影、对称性与不等式惩罚的损失函数、Adam优化、SVPC LP得到的数值多重共形包络作为训练标签。

**📊 数据集**

数据集为Saint Venant–Kirchhoff能量在三维空间中，使用[0.4,1.4]^3的75×75×75网格（正象限）共421 875个训练样本；对比完整签名空间1687500个样本用于Φ_方法。

**📈 对比分析**

通过均方误差、对称性误差和不等式误差进行对比，Ball方法与Φ_方法在误差上几乎相同，但训练时间从约35 min降至22 min，速度提升约1.6倍；在多重共形包络的近似精度上无显著差异。

**⚠️ 局限性**

局限性：①只给出足够条件下的下界，可能不是全局最优包络；②单调性约束可能过于严格，限制了更一般模型的适用性；③实验仅在Saint Venant–Kirchhoff模型上验证，未证明在更复杂材料或更高维空间的泛化能力。

---

## 514. PedNStream: Scalable Network Flow Simulation for Pedestrian Traffic Management

**arXiv ID:** 2607.01021 | [PDF](https://arxiv.org/pdf/2607.01021v1)

**作者:** Weiming Mai `[一作]` (Delft University of Technology), Serge Hoogendoorn `[通讯]` (Delft University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一款基于LTM的宏观行人网络仿真框架PedNStream，加入随机扩散动态、基于效用函数的动态路径选择和门控、分离器等控制接口，可实时评估大规模人群管理策略。

**💡 创新点**

创新点包括：①在LTM中引入随机扩散与活动诱导的发送/接收动态，使行人流更逼真；②将动态用户均衡(DUE)替换为可解释的效用‑多项式逻辑（MNL）路由模型；③设计可直接操作门宽的控制器（规则式与压力式），并将控制逻辑嵌入仿真循环；④提供完整开源Python实现，支持模块化扩展。

**🔧 技术方法**

技术手段：Link Transmission Model（LTM）及其双向扩展、随机扩散与二项释放模型、效用函数与多项式逻辑的动态路由、节点流分配（比例与流最大化）、控制器接口（门宽调节），整体以Python实现，复杂度为O(T(N+E))。

**📊 数据集**

实验数据集：1）荷兰德尔夫特市中心道路网络（298节点、818条链接）用于大规模仿真；2）澳大利亚墨尔本步行计数系统（92个传感器）用于部分观测验证；3）若干合成网络用于机制验证。

**📈 对比分析**

比较方法与性能：①对比原LTM、PedNStream和双向LTM，在合成场景中展示队列、冲击传播、拥堵恢复与路径重定；②在墨尔本数据中，与简单KNN插值相比，PedNStream在RMSE、NRMSE、NDTW等指标上显著优越；③闭环控制实验中，规则式与压力式门控均提升路径0–8的平均流量和步速；④运行时分析显示，与OD对数线性相关，控制器数量影响较小，整体性能可接受。

**⚠️ 局限性**

局限性：①模型基于宏观网络，无法捕捉个体细节与空间细分；②需手工校准扩散、释放、活动概率等参数；③验证仅基于部分计数数据，缺乏完整OD流量；④对极端拥堵或非常规场景（如突发事件、紧急疏散）的适用性尚待进一步测试。

---

## 515. Toward a Unified Security and Privacy Framework for AI-Native 6G Networks

**arXiv ID:** 2607.01019 | [PDF](https://arxiv.org/pdf/2607.01019v1)

**作者:** Bidushi Barua `[一作]` (University of York), Poonam Yadav `[通讯]` (University of York)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对AI本地化6G网络的安全与隐私问题进行系统综述。

**💡 创新点**

提出统一的安全与隐私框架和跨层威胁分类法，填补现有研究碎片化的空白。

**🔧 技术方法**

综合标准化、网络架构、AI安全、隐私治理等多维技术，构建跨层防护体系。

**📊 数据集**

未使用特定数据集，主要基于文献综述与标准文档。

**📈 对比分析**

通过对比现有文献、标准和技术实现，评估不同防护措施的适用性，未给出量化指标。

**⚠️ 局限性**

缺乏实证验证与统一基准，跨层协同实现的复杂性及标准一致性仍是挑战。

---

## 516. Tighter Bounds for Wheeler Determinization

**arXiv ID:** 2607.01007 | [PDF](https://arxiv.org/pdf/2607.01007v1)

**作者:** Philip Bille `[一作]` (Technical University of Denmark), Simon R. Tarnow `[通讯]` (Technical University of Denmark)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种输出敏感的算法，用于在给定 Wheeler NFA 及其 Wheeler 顺序的情况下构造等价的 Wheeler DFA，时间复杂度为 O(|I|+|I'|)。

**💡 创新点**

创新点在于利用两种新型数据结构（区间动态集合与基于颜色范围查询的边发现结构）实现了对 Wheeler NFA 的直接幂集构造，从而把原始 O(n³) 的时间复杂度降低到线性（或近线性）级别，并证明了该时间界对所有 n 与 σ 的组合是紧确的。

**🔧 技术方法**

主要技术包括：对 Wheeler 图的幂集构造；区间动态集合（使用位向量实现 O(1) 搜索/插入）；基于 Muthukrishnan 颜色范围查询的边发现数据结构（利用秩/选择、最大/最小范围查询实现 O(1+occ) 访问）；以及秩空间的标签压缩。

**📊 数据集**

由于是理论算法研究，作者未使用具体实验数据集；所有结果均为理论分析与构造性证明。

**📈 对比分析**

与 Alanko 等人提出的 O(n³) 算法相比，该方法在给定 Wheeler 顺序时提高了 n²/σ 的速度，对于常数字母表可达线性时间；作者通过构造最坏实例证明了此改进的 tightness。

**⚠️ 局限性**

局限性包括：算法依赖于已知的 Wheeler 顺序；仅适用于 Wheeler NFA（非一般 NFA）；虽然时间复杂度优越，但实现上需构建复杂的数据结构，对实际内存与常数因子有一定影响。

---

## 517. Linkify: Learning from Interface-Augmented Assembly Graphs

**arXiv ID:** 2607.01205 | [PDF](https://arxiv.org/pdf/2607.01205v1)

**作者:** Anushrut Jignasu `[一作]` (Iowa State University), Daniele Grandi `[通讯]` (Autodesk Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了基于界面增强的装配图，训练GATv2图神经网络实现部件的掩码检索。

**💡 创新点**

首次将精确接触几何嵌入图边，并通过Mask部件预测验证接口信息对检索的贡献。

**🔧 技术方法**

采用PointMAE点云编码器、GATv2注意力机制以及基于OCCT的接口重计算。

**📊 数据集**

使用重新计算接口的Fusion 360 Gallery Assembly数据集，并通过500类聚类提供标签。

**📈 对比分析**

与随机、占优类、k-NN、LogReg等基线对比，Top‑1准确率提升至6.34%，Top‑5/10亦显著提高。

**⚠️ 局限性**

对功能理解不足、接口嵌入无法捕获运动学细节，且数据规模有限导致泛化受限。

---

## 518. World from Motion: Generative Dynamic Gaussian Reconstruction from Monocular Video

**arXiv ID:** 2607.01202 | [PDF](https://arxiv.org/pdf/2607.01202v1)

**作者:** Liyuan Zhu `[一作]` (Stanford University), Alex Trevithick `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种将单目视频几何重建与视频生成模型结合的方法，通过动态 3D Gaussian Splatting 作为中间表示，生成并优化完整的 4D 场景。

**💡 创新点**

创新点在于：① 将动态 3DGS 作为视频生成模型的条件；② 在输入与目标轨迹上渲染完整 G-buffer 与流信息实现双轨条件；③ 利用生成的虚拟观测进行蒸馏与重优化，实现高质量动态 4D 重建。

**🔧 技术方法**

采用潜在空间视频扩散模型（基于 Diffusion Transformer）配合 3D Gaussian Splatting 以及运动匹配优化、classifier‑free guidance、重采样与重优化等技术。

**📊 数据集**

训练使用 MultiCamVideo（合成多视角动态场景）数据集，评估使用 DyCheck benchmark 与 MultiCamVideo hold‑out，以及 WildGS 等真实视频做定性验证。

**📈 对比分析**

与 Shape of Motion、MoSca、WorldTree、ViDAR、CAT4D、Vista4D、ReCamMaster 等基线在 DyCheck 与 MultiCamVideo 上进行对比，PSNR/SSIM/LPIPS 均显著提升（如 DyCheck 上 19.98/0.716/0.178，MultiCamVideo 上 27.43/0.918/0.083）。

**⚠️ 局限性**

依赖初始 3DGS 重建；当初始重建完全失败或目标轨迹与输入轨迹差距过大时，模型难以对齐，导致性能下降；目前未实现多轮交替优化。

---

## 519. Quantum vs. Classical Machine Learning: A Unified Empirical Comparison

**arXiv ID:** 2607.01197 | [PDF](https://arxiv.org/pdf/2607.01197v1)

**作者:** Chuanming Yu `[一作]` (Hebei Normal University), Jianjun Zhao `[通讯]` (Kyushu University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过统一实验框架，对七对量子与经典机器学习模型（包括监督学习和强化学习）进行系统的实证比较。

**💡 创新点**

创新点在于：①以架构对齐的方式平衡参数规模；②在同一数据集与环境下同时评估监督与强化学习；③细化了量子模型在噪声与硬件限制下的优势与劣势。

**🔧 技术方法**

采用 PennyLane 与 PyTorch 实现量子电路与经典网络；利用角度编码、PCA 降维、噪声通道、量子策略梯度、量子 Q‑学习等技术；在模拟器上对比训练效率与收敛行为。

**📊 数据集**

使用 Bars and Stripes（BAS）数据集进行图像分类；构造 Hypercube 环境进行 6 维离散强化学习测试。

**📈 对比分析**

通过统一指标（准确率、精确率、召回率、F1、AUC‑ROC、训练耗时；平均回报、胜率、稳定性、参数量）比较，结果显示经典模型在整体准确率、回报与稳定性上占优；量子模型在 QCNN 的精确率与 QRL 的参数效率方面表现出局部优势。

**⚠️ 局限性**

主要局限包括：量子比特瓶颈导致特征降维、对硬件噪声高度敏感、训练时间显著超出经典、优化过程易出现 Barren Plateaus，限制了量子模型的实用性。

---

## 520. High-dimensional Embedding Prior for Noisy K-space Domain MRIReconstruction

**arXiv ID:** 2607.01176 | [PDF](https://arxiv.org/pdf/2607.01176v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 521. Detecting Adversarial Evasion Attacks Against Autoencoder-Based Network Intrusion Detection Systems

**arXiv ID:** 2607.01194 | [PDF](https://arxiv.org/pdf/2607.01194v1)

**作者:** Niklas Bunzel `[一作]` (Fraunhofer SIT), Ashim Siwakoti `[通讯]` (Technische Universitaet Darmstadt)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并提出两种检测器（Residual Localisation Detector 与 Feature‑Space Perturbation Consistency Detector）来识别针对基于自编码器的网络入侵检测系统的 PANDA 风格对抗攻击，并在 UQ‑IoT 数据集上验证其效果。

**💡 创新点**

通过在图像空间定位重建误差在 IAT 列的聚集以及在包级 IAT 特征空间测量重建误差一致性，结合两种互补视角，实现对对抗流量的近乎完美检测。

**🔧 技术方法**

使用可逆包‑图像映射、卷积自编码器代理模型、蒙版 FGSM 对抗生成、残差定位检测和小型全连接自编码器等技术。

**📊 数据集**

使用 UQ‑IoT‑IDS‑2021 数据集，包含多种 IoT 设备的正常流量及多种攻击类型。

**📈 对比分析**

在窗口级别采用真阳率、真阴率、精确率、召回率和 F1 分数进行评估；RLD 达到 TNR≈0.9999、TPR≈0.9972，FPC 达到 TNR≈0.9993、TPR≈0.986，均 F1≥0.99。

**⚠️ 局限性**

阈值需针对不同网络再校准；检测器仅针对 IAT 变形，易被改动其他字段的攻击规避；未考虑自适应攻击、PCAP round‑trip 的稳健性限制及仅评估 FGSM 迭代攻击。

---

## 522. Neural Certificate Pricing for Combinatorial Optimization Problems

**arXiv ID:** 2607.01185 | [PDF](https://arxiv.org/pdf/2607.01185v1)

**作者:** Jingyi Chen `[一作]` (Rice University), Xinwu Qian `[通讯]` (Rice University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Neural Certificate Pricing（NCP）框架，利用无监督学习通过预测价格扰动 Γ 并在结构化恢复层中实现证书一致性，从而直接从可行的边缘分布（marginal）中恢复近似最优的组合优化解。

**💡 创新点**

创新点在于：① 将证书一致性与结构化恢复融合为可训练的优化管道；② 通过学习价格扰动而非直接预测解，显著降低了搜索复杂度；③ 提供局部稳定性分析，证明误差是二阶的，从而确保在可达的局部最优点附近学习稳定；④ 对多类组合优化问题（MIS、GAP、ESPP‑NC）实现了一套统一的无监督训练和恢复流程。

**🔧 技术方法**

技术细节包括：神经网络预测扰动 Γ；基于证书的可恢复集合 𝒮_G(y)；对约束的拉格朗日双重 λ 的固定点迭代；利用隐式梯度（implicit differentiation）进行无监督优化；在恢复层使用动态规划、线性规划、投影式对偶更新等实现结构化求解。

**📊 数据集**

实验数据集：
- MIS：Twitter、Collab、IMDB（无权与节点权重版）；
- GAP：ORLIB 基准实例；
- ESPP‑NC：Erdős–Rényi 与 Barabási–Albert 随机图，节点数 50–500。

**📈 对比分析**

与传统求解器（Gurobi、KaMIS）以及最新神经方法（Erdos‑Neural、DIFUSCO、Fast‑T2T）对比。NCP 在 MIS 上获得与最强神经基线相当甚至更优的比例，且推理时间仅为单次前向传播；在 GAP 上比 LP 取整方案显著降低平均相对缺口，k=1、k=2 时均为最佳；在 ESPP‑NC 上对比 LP+解码，NCP 在大图规模下保持更低的路径成本并显著缩短运行时间。整体上，NCP 在保持计算效率的同时展示了更强的泛化与鲁棒性。

**⚠️ 局限性**

局限性包括：
- 需要针对每个问题设计合适的证书集合 𝒮_G(y) 与读取映射 C_G，缺少通用模板；
- 对于证书不兼容的组合结构（如 clique 证书无法表示路径），可能导致证书间隙 Δ_cert 产生；
- 训练过程依赖隐式梯度与固定点迭代，计算成本和收敛稳定性受超参数影响；
- 理论保证仅局部，无法保证全局最优；
- 对极大规模实例仍可能因恢复层复杂度或内存消耗受限。

---

## 523. Structured 4D Latent Predictive Model for Robot Planning

**arXiv ID:** 2607.01166 | [PDF](https://arxiv.org/pdf/2607.01166v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 524. QuasiMoTTo: Quasi-Monte Carlo Test-Time Scaling

**arXiv ID:** 2607.01179 | [PDF](https://arxiv.org/pdf/2607.01179v1)

**作者:** Michael Y. Li `[一作]` (Stanford University), Emily B. Fox `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 QuasiMoTTo，通过随机 QMC 与逆 CDF 采样产生相关但精确的 LLM 样本，以减少冗余；

**💡 创新点**

设计可并行的相关采样器，使用 QMC 在单位区间生成低差异点，再通过算术编码映射为精确语言模型样本，并提供无偏 bootstrap 估计及覆盖上限；

**🔧 技术方法**

随机化 QMC（格点、分层、Sobol）、算术编码、逆 CDF 采样、GRPO 强化学习、无偏 bootstrap 与 RLOO 调整等技术；

**📊 数据集**

四个符号推理基准：Countdown、Maze、Sudoku、1D-ARC；

**📈 对比分析**

与 i.i.d. 采样及理论上限对比，QuasiMoTTo 在 pass@k 上可用 25–47% 更少样本达到相同精度；在 RL 上可将 GRPO 步数降低约 50%，并显著降低零方差组；

**⚠️ 局限性**

仅在短文本符号推理任务上验证，难以直接扩展到长链思维或语义等价类；依赖精确的 LM 输出分布，可能存在 RLOO 基线偏差。

---

## 525. Efficient Compression of Structured and Unstructured Volumes via Learned 3D Gaussian Representation

**arXiv ID:** 2607.01164 | [PDF](https://arxiv.org/pdf/2607.01164v1)

**作者:** Landon Dyken `[一作]`, Sidharth Kumar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

论文未提供具体内容，因此无法总结做了什么。

**💡 创新点**

论文未提供具体内容，因此无法总结创新点。

**🔧 技术方法**

论文未提供具体内容，因此无法总结使用的技术。

**📊 数据集**

论文未提供具体内容，因此无法总结使用的数据集。

**📈 对比分析**

论文未提供具体内容，因此无法总结比较的方法和性能。

**⚠️ 局限性**

论文未提供具体内容，因此无法总结限制因素。

---

## 526. EquiSteer: Cross-Attention Steering Towards a Fairer Text-Guided Image Generation

**arXiv ID:** 2607.01147 | [PDF](https://arxiv.org/pdf/2607.01147v1)

**作者:** Tatiana Gaintseva `[一作]` (Queen Mary University of London), Ismail Elezi `[通讯]` (Huawei Noah's Ark)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种训练无关、推理时的交叉注意力调节方法 EquiSteer，用来公平地消除文本引导图像生成中的性别、种族、年龄等偏见。

**💡 创新点**

创新点在于：① 引入 prompt‑aware gate 能够识别并保留已显式指定属性的提示；② 在注意力输出上做属性子空间正交化，去除已有的偏见信号；③ 通过预计算的平均最大点积来自适应调节注入力度，从而在不改变模型权重的前提下实现细粒度的公平控制。

**🔧 技术方法**

核心技术是跨层交叉注意力（Cross‑Attention）向量的预计算与插入、基于点积的属性检测门控、正交投影与自适应增益调节；实验中将其与 SD‑1.5、SD‑2.1、SDXL、SANA 等主流扩散模型无缝集成。

**📊 数据集**

使用公开的职业提示集合（如 CEO、doctor、pilot 等）生成 1,000 张图像进行公平性评估，并利用 CLIP ViT‑L/14、BLIP‑VQA 等零样本分类器检测性别、种族、年龄、身体类型、眼镜等属性；训练集与测试集互不重叠，确保评估客观性。

**📈 对比分析**

与 TEI、FairDiffusion、Unified Concept Editing 等基线对比，EquiSteer 在性别任务上平均将偏差间隙（Δ）降低 87%，相较于最强基线减少 47%；在多属性与多模型实验中亦保持低 Δ 并保持 CLIPScore 与 CMMD 等图像质量指标不下降，证明其效果优越且不损害生成质量。

**⚠️ 局限性**

主要局限包括：对多主体或组合提示的处理仍不够理想（门控在单图像级别上限制了对多角色场景的控制）；依赖 CLIP 等零样本分类器，若分类器本身对某些属性识别不佳可能影响评估；同时需要为每个属性预计算并存储注意力向量，增加了前期准备工作。

---

## 527. A Lightweight Self-Supervised Learning Framework for Multivariate Time Series using Hierarchical-JEPA on ECG Data

**arXiv ID:** 2607.01145 | [PDF](https://arxiv.org/pdf/2607.01145v1)

**作者:** Siwon Kim `[一作]` `[通讯]` (Seoul National University), Siwon Kim (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了ER-JEPA，一种轻量级的层次化联合嵌入预测架构，利用自监督学习对多通道心电图（ECG）进行预训练，并在下游分类任务中进行微调。

**💡 创新点**

创新点在于将通道级JEPA与时序级JEPA分为两阶段，先将多通道序列压缩为单通道表示，再进行时序分析，从而实现高效的多通道时序处理，并通过两级JEPA实现层次化表征，显著提升了对复杂任务的泛化能力。

**🔧 技术方法**

使用了自监督学习框架中的Joint-Embedding Predictive Architecture（JEPA）、Vision Transformer（ViT）作为骨干、遮蔽（masking）策略、指数滑动平均更新目标编码器、dropout以及多种聚合层等技术。

**📊 数据集**

预训练使用约18万条10秒ECG（来自CODE‑15和Shaoxing数据集），下游任务则在PTB‑XL（多标签/多类）和CPSC2018（心律分类）数据集上进行评估。

**📈 对比分析**

与ST‑MEM、ECG‑JEPA、Weimann & Conrad等现有SSL模型进行对比，ER‑JEPA在PTB‑XL多类任务上取得AUC 0.943的SOTA成绩，在CPSC2018上与SOTA持平，并在计算速度和显存占用上实现约3–8倍的提升。

**⚠️ 局限性**

局限性包括：对预训练超参数高度敏感，易受表征崩溃影响；仅在10秒10通道ECG上验证，缺乏对更长序列或其他多模态医学信号的探索；以及对实际临床部署的鲁棒性和可解释性仍需进一步研究。

---

## 528. Antaeus: Hunting Repository-Level Logic Vulnerabilities via Context-Grounded LLM Reasoning

**arXiv ID:** 2607.01138 | [PDF](https://arxiv.org/pdf/2607.01138v1)

**作者:** Michele Armillotta `[一作]` (University College London), Lorenzo Cavallaro `[通讯]` (University College London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Antaeus 框架，利用仓库级上下文扎根的大语言模型（LLM）在 C/C++ 项目中检测逻辑漏洞。

**💡 创新点**

创新点在于：① 引入功能优先化与多层上下文扎根，② 将局部代码与全局仓库语义结构化结合，③ 通过对比验证消除项目规范导致的误报，并生成可验证的结构化报告。

**🔧 技术方法**

采用的技术包括：大语言模型（Claude Opus、GPT‑5.4）、树形解析（Tree‑sitter）构造局部语义图、UniXcoder 与 MiniLM-L6‑v2 的语义相似度评估、结构化提示与对比验证算法。

**📊 数据集**

使用的数据集为 ReposVul 中挑选的 28 个包含 CWE‑200 / CWE‑284 逻辑漏洞的开源 C/C++ 仓库。

**📈 对比分析**

与函数级 LLM 与 agentic 基线进行对比：在 Opus 4.7 下检出 15/28 病例，GPT‑5.4 下 12/28，误报显著降低，token 使用与成本与 baselines 相近，说明框架在保持预算可控的前提下显著提升检测效果。

**⚠️ 局限性**

局限性包括：仅针对 C/C++；依赖 LLM 的推理质量与提示设计；局部语义深度（仅展开到 1 层调用）可能导致缺失跨函数安全检查；优先化阶段可能遗漏非显式安全函数；对比验证阈值需手工校准，且不保证检测完备。

---

## 529. GAIA: Geometry-Adaptive Operator Learning for Forward and Inverse Problems

**arXiv ID:** 2607.01128 | [PDF](https://arxiv.org/pdf/2607.01128v1)

**作者:** Meenakshi Krishnan `[一作]` (University of Maryland), Ramani Duraiswami `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出Geometry‑Adaptive Integral Autoencoder (GAIA)，一种统一的神经算子模型，能够在一次前向传播中同时求解任意几何域的正向、边值和逆问题；

**💡 创新点**

通过在积分核中引入基于边界与内部状态的两条token化路径，并用交叉注意力为每个查询点提供空间自适应几何上下文，从而在不需要图构造、迭代优化或重新训练的前提下实现单通道推理；

**🔧 技术方法**

集成了IAE‑Net的积分自编码器结构、PointNet‑style 边界token化、soft‑clustering 切片token化、交叉注意力条件化的积分核、DenseNet 级联以及可处理不同输入/输出域的观察‑域解码器；

**📊 数据集**

在七个时间不变 PDE 基准上评估，包括四个新/扩展的逆/BVP 数据集：电阻率成像 (EIT)、光学层析 (OT)、三维 Darcy 流和机械部件的 Poisson BVP；另外还有 Poisson‑Gauss、弹性和标准三维 Darcy 等传统基准；

**📈 对比分析**

与 GINO、CORAL、Transolver、GAOT、LNO 等五种几何自适应算子进行对比，GAIA 在所有逆/BVP 任务上均达到或超过最新最优水平，EIT 误差下降 27%、OT 下降 21%、翼型重建 下降 64%；在正向任务上保持竞争力并在不同分辨率下展示出更稳定的误差；

**⚠️ 局限性**

局限性包括：仅针对时间不变算子；在共享密集网格的正向问题中相对内测精度略低；交叉注意力的 O(NK) 计算/存储开销在大规模 3D 点云上可能不切实际，需要进一步压缩或稀疏注意力机制。

---

## 530. Corporate sponsorship of computer science conferences: trends, structural insights, and a novel approach to ranking conferences

**arXiv ID:** 2607.01113 | [PDF](https://arxiv.org/pdf/2607.01113v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 531. ZO-Act: Efficient Zeroth-Order Fine-Tuning via One-Shot Activation-Informed Low-Rank Subspaces

**arXiv ID:** 2607.01125 | [PDF](https://arxiv.org/pdf/2607.01125v1)

**作者:** Xun Dong `[一作]`, Zi Yang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

在LuaLaTeX或XeLaTeX环境下演示如何使用ACL风格文件

**💡 创新点**

提供了一个示例模板，帮助作者快速上手ACL论文格式

**🔧 技术方法**

使用LaTeX编译技术和ACL style文件

**📊 数据集**

未使用任何公开数据集

**📈 对比分析**

未进行任何实验或性能比较

**⚠️ 局限性**

缺乏实际实验与评估，缺少可验证性

---

## 532. Muon as a Residual Connection

**arXiv ID:** 2607.01124 | [PDF](https://arxiv.org/pdf/2607.01124v1)

**作者:** Hao Huang `[一作]` `[通讯]` (Zhejiang University), Hao Huang (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了 Muon 优化器的机制，将其视为隐式残差连接，并通过两阶段线性实验和 τ 调度实验验证其在下游层优化中的优势。

**💡 创新点**

提出正交化更新在梯度保真度与下游表征保持之间做权衡的视角，并将 Muon 与残差连接类比，提供一种新的解释框架。

**🔧 技术方法**

使用正交化更新、两层线性网络、两阶段训练、τ 切换调度等实验技术，比较 Muon 与 SGD 的表现。

**📊 数据集**

使用合成的 Gaussian 随机矩阵作为目标变换，在线性网络上进行实验，不依赖真实数据集。

**📈 对比分析**

与 SGD 进行对比；在 Phase‑1 中 SGD 下降更快，但在 Phase‑2 以及整体训练中 Muon 需要更少的步骤，整体收敛更快。

**⚠️ 局限性**

实验仅在受控的线性环境下进行，未在深度非线性网络中验证；解释为机理类比，缺乏广泛实证，需进一步研究。

---

## 533. Towards Developing a Multimodal Chat Assistant for University Stakeholders: RAG-based Approach

**arXiv ID:** 2607.01115 | [PDF](https://arxiv.org/pdf/2607.01115v1)

**作者:** Md Abu Hanif Shaikh `[一作]` (Khulna University of Engineering & Technology), Abdullah Al Shafi `[通讯]` (Khulna University of Engineering & Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了基于检索增强生成（RAG）的多模态大学聊天助手，支持文本和图像查询

**💡 创新点**

将RAG与量化的LLava-1.5-7B视觉语言模型结合，显著降低幻觉率并实现图像问答

**🔧 技术方法**

使用LLava-1.5-7B（量化推理）、ChromaDB向量存储、LangChain框架、FastAPI后端、Next.js前端

**📊 数据集**

以昆达尔纳大学手册（PDF）为知识库进行语料构建

**📈 对比分析**

与基线LLaVA-7B对比，幻觉率从31.7%降至6.6%，用户满意度4.26/5，检索top‑1 78.3%、top‑3 92.1%

**⚠️ 局限性**

存在多跳推理能力不足、响应延迟高（尤其是图像查询）、部分图像查询满意度下降等局限

---

## 534. GPU-Parallel Linearization Error Bounds for Real-Time Robust Optimal Control of Nonlinear and Neural Network Dynamics

**arXiv ID:** 2607.01203 | [PDF](https://arxiv.org/pdf/2607.01203v1)

**作者:** Jeffrey Fang `[一作]` (Georgia Institute of Technology), Glen Chou `[通讯]` (Georgia Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对不确定的非线性系统，提出了一种实时GPU并行的鲁棒非线性最优控制方法GPUSLS‑LEO，结合可微分、紧凑的线性化误差界（LEB），实现对非线性和神经网络动力学的正式安全保障。

**💡 创新点**

创新点包括：① 采用路径式Hessian逼近得到紧致可微的LEB；② 将LEB与右可逆扰动矩阵和非零中心扰动集合结合，改进系统层级合成（SLS）框架；③ 在GPU上实现全流程并行求解，显著降低计算延迟。

**🔧 技术方法**

主要技术：路径式Hessian计算、CROWN神经网络误差界、Zonotope传播、JAX实现的可微分GPU并行、SLS与SQP优化相结合。

**📊 数据集**

数据集与模型：分析模型（卫星姿态控制7D、平面四旋翼6D、14体3D四旋翼168D）以及神经网络模型（T‑pusher 5D），并在模拟扰动与障碍场中评估。

**📈 对比分析**

与基准方法（全局采样、区间Hessian、CORA、ELLIPSOIDAL SLS、控制收缩度量 CCM）比较，GPUSLS‑LEO在所有系统上实现更小的可达管道、约20–45%更低的保守性，并能以高达67 Hz的速度完成约2×10⁵决策变量的实时优化。

**⚠️ 局限性**

局限性：对极端高维或高度非光滑动力学的Hessian计算仍可能导致计算开销；当前未考虑学习误差，仅对神经网络误差进行线性化误差界；对连续时间系统的适用性尚待验证。

---

## 535. Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents?

**arXiv ID:** 2607.01211 | [PDF](https://arxiv.org/pdf/2607.01211v1)

**作者:** Zhi Chen `[一作]` (Singapore Management University), Lingxiao Jiang `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三大仓库级性能优化基准（GSO、GPE、EffiBench）进行了系统审计，重现官方参考补丁并跨机验证其有效性；同时分析了排行榜得分规则对提交排名的影响，并在10份公开提交中评估任务覆盖度，发现大多数任务已被公共提交覆盖，剩余未达参考补丁速度的任务多为细粒度性能提升瓶颈而非根本失效；

**💡 创新点**

①揭示官方参考补丁在不同机器上不稳定，可能导致排行榜失真；②发现排行榜得分规则（尤其是基于谐波平均的规则）对极端低速度任务高度敏感，压倒性影响总分；③通过任务级覆盖度分析提供更细粒度的进展评估，补充传统聚合排行榜信息；

**🔧 技术方法**

跨机补丁重现（4种Google Cloud机器×3轮）；统计学检验（速度比、IQR、Mann–Whitney）；排行榜得分规则重算（引用门槛、谐波平均、有限惩罚）；任务级贡献权重分析；优化策略分类对比（使用Peng等分类和GPT-5.5标注）；

**📊 数据集**

三大公开仓库级基准：GSO（102任务）、GPE（140任务）、EffiBench（498任务），以及各自的官方参考补丁和公开提交（共10份），在Google Cloud环境中重现运行；

**📈 对比分析**

通过将同一任务的公开提交在不同得分规则下重新计算，发现官方排名在8份公开提交中有9/28对比排名不一致；在谐波平均规则下，最差10个任务的得分权重可占总分58.5%–82.8%，显示极端低性能任务对总分影响巨大；

**⚠️ 局限性**

仅针对三份公开基准且基于OpenHands+模型的提交；重现过程受云机资源与配置限制；公共提交覆盖度仅是优化进展的上限，未涵盖多代理多轮迭代真实场景；

---

## 536. FastBridge: Closing the Model-Based Realization Gap in Safety Filters on 3D Gaussian Splatting for Fast Quadrotor Flight

**arXiv ID:** 2607.01200 | [PDF](https://arxiv.org/pdf/2607.01200v1)

**作者:** Tscholl Dario `[一作]`, Gunter Brian `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文设计并实现了一种基于3D Gaussian Splatting（3DGS）的非线性安全滤波器，利用碰撞锥指数控制栏函数（ECBF）和前向仿真备份控制栏函数（Backup CBF）实现四旋翼在复杂环境中的安全飞行。

**💡 创新点**

创新点包括：① 解决传统双积分模型与实际四旋翼动力学之间的实现差距；② 推导了高相对度的碰撞锥指数CBF，直接作用于完整非线性动力学；③ 设计了利用备份策略实现QP可行性的备份CBF；④ 通过动态反馈线性化将安全约束映射至实际推力/扭矩输入。

**🔧 技术方法**

技术手段包括：控制栏函数（CBF）、指数CBF（ECBF）、几何跟踪控制、动态反馈线性化、前向仿真备份策略、三维高斯剖面图像（3DGS）、Nerfstudio/Splatfacto场景重建、SE(3)四旋翼动力学模型、二次规划QP、MRP姿态表示。

**📊 数据集**

使用了自定义真实世界场景，通过 Nerfstudio 与 Splatfacto 重建得到约 395k 个 Gaussian splat 的 3DGS 数据集。

**📈 对比分析**

与状态最优的距离基 SAFER‑Splat 进行对比，采用相同起点、终点、参考轨迹、控制频率和障碍门限。实验结果表明：ECBF 方案将轨迹 jerk 平均降低 47%，最大 jerk 降低 46%，计算时间比 SAFER‑Splat 快 2.25 倍；硬件实验中 QP 可行率达 91.8%，最高飞行速度约 1.5 m/s，且保持安全。

**⚠️ 局限性**

局限性包括：备份策略仅实现制动/倾斜，未实现高速度逃逸策略，导致高速飞行时只能减速；硬件平台受限，速度受限；对更大规模障碍集合的实时求解尚未充分验证；对高阶相对度控制栏函数的可扩展性仍需进一步研究。

---

## 537. Optimal Resource Utilization for Autonomous Laboratory Orchestrators

**arXiv ID:** 2607.01188 | [PDF](https://arxiv.org/pdf/2607.01188v1)

**作者:** Austin McDannald `[一作]` (National Institute of Standards and Technology), Howie Joress `[通讯]` (National Institute of Standards and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本研究设计了一套两步方法，用约束编程生成最优排程，再结合状态依赖与互斥机制实现自主实验室平台的鲁棒执行，提升资源利用率。

**💡 创新点**

创新点在于将Job Shop调度与运行时状态依赖/互斥协同使用，支持实时动态实验请求、资源容量与任务依赖的精细建模，从而在保持最优排程的同时实现可靠执行。

**🔧 技术方法**

核心技术包括Google OR‑Tools的约束求解器、Python AsyncIO异步执行、互斥锁（Mutex）与状态表（dependency table）等，并在MOF合成平台上实现。

**📊 数据集**

实验数据基于内部金属‑有机框架（MOF）合成平台的真实任务集合，未使用公开数据集，而是模拟多批次实验（如Cu‑BTC）进行排程与执行。

**📈 对比分析**

通过绘制甘特图与测量求解时间验证方法效果：单批次排程约28秒完成，增量任务约1.4秒，显著缩短总实验周期，展示出良好的可扩展性和执行效率。

**⚠️ 局限性**

主要局限包括：仅优化总完成时间，未考虑实验优先级、知识增益估计与批量大小决策；调度复杂度高，实时反馈给AI的能力有限，且在高并发与多资源交互场景下易出现组合爆炸。

---

## 538. Trie-based Experiment Plans for Efficient IR Pipeline Experiments

**arXiv ID:** 2607.01162 | [PDF](https://arxiv.org/pdf/2607.01162v1)

**作者:** Irene Anu `[一作]` (University of Glasgow), Craig Macdonald `[通讯]` (University of Glasgow)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了基于 Trie（radix 树）的实验计划，能够在 PyTerrier 中共享 IR 管道的前缀，从而显著减少重复计算。

**💡 创新点**

创新点在于将 IR 管道视为有向无环图，并通过 radix 树高效识别和执行所有共享前缀，提升实验效率至传统线性或 LCP 方案的 20–30% 以上。

**🔧 技术方法**

使用的技术包括 Python 的 PyTerrier 框架、声明式管道构造、Trie（压缩前缀树）数据结构以及对实验计划的自动化执行与可视化。

**📊 数据集**

所用数据集为 MSMARCO v1 和 v2 的深度学习 Track 查询集及其评估集合（TREC 2019 与 2021 Deep Learning Track）。

**📈 对比分析**

通过与线性执行、LCP 优化方案对比，树执行计划在 MSMARCO v2 上实现了约 26% 的实验时间缩短，同时保持 nDCG@10 等评估指标不变。

**⚠️ 局限性**

局限性包括仅能共享完全相同的前缀，无法处理不同截断值的可嵌套前缀，以及对非前缀共享的优化（如截断后继操作）尚未覆盖。

---

## 539. AGC-Bench: Measuring Artificial General Creativity

**arXiv ID:** 2607.01152 | [PDF](https://arxiv.org/pdf/2607.01152v1)

**作者:** Roger Beaty `[一作]` (Pennsylvania State University), Anna Rumshisky `[通讯]` (University of Massachusetts Lowell)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个面向人工通用创造力的元基准，整合了497个独特的创造力评测任务，开发了AGC‑Judge开放权重评分模型，并通过因子分析揭示了单一的通用创造力因子C。

**💡 创新点**

创新点在于：①采用PRISMA系统综述和agentic onboarding harness将各类创造力基准统一为HELM兼容场景；②使用Judge Response Theory校准LLM评审的严苛度并训练AGC‑Judge，显著降低评审成本；③首次在LLM上进行心理计量学因子分析，验证了创造力在不同域间的可迁移性。

**🔧 技术方法**

主要技术包括PRISMA系统综述、HELM风格评测基准、Judge Response Theory（IRT）校准、Qwen3‑30B LoRA微调生成AGC‑Judge、因子分析（主成分与平行分析）、Spearman相关与Bootstrap验证等。

**📊 数据集**

使用的数据集覆盖：497个基准（432人工可评、159转换为HELM场景），24个LLM评审基准，6个文本域（头脑风暴、问题解决、STEM、叙事、比喻、幽默），5个与人类对齐的任务（AGC‑Human），以及1862个MuCE人类评测样本。

**📈 对比分析**

比较方法为：对每个模型在各基准上计算z得分，合成总体与域级别分数；AGC‑Judge与三评审集在保留数据上Spearman ρ≥0.97，在未见基准上ρ≈0.83；因子分析显示单因子C解释81.5%方差；模型表现显示前沿LLM整体领先，开放权重模型紧随其后，并在不同域呈现不同强项。

**⚠️ 局限性**

局限性包括：基准以英语为主，视觉多模态仅限图像输入；Composite分数相对而非绝对，没有理论最高分；部分创造性维度（如变革性创造力）未被充分评估；模型规模虽相关但并非决定性预测因子；未来工作需扩展到更多语言、模态与更短尺度测评。

---

## 540. Emergence of Preferential Attachment and Glass-Ceiling Effects in Autonomous Networks of LLMs

**arXiv ID:** 2607.01148 | [PDF](https://arxiv.org/pdf/2607.01148v1)

**作者:** Yiming Zhang `[一作]` (Cornell University), Vikram Krishnamurthy `[通讯]` (Cornell University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型代理在自主协作网络中出现的结构差异，证明并实验了玻璃天花板效应（GCE）

**💡 创新点**

首次将跨注意力激励与均值场动力学结合，证明在向量权重网络中存在稳定的中心性差距并能通过策略调节

**🔧 技术方法**

使用向量加权有向图建模、均值场 ODE、跨注意力启发的连接效用、随机微分逼近与收敛证明等技术

**📊 数据集**

利用 100 个 LLM 代理的实验平台，代理来自 Gemini、GPT、Grok、LLaMA、Qwen、Mistral 六个模型族，任务为协同问答与多代理辩论

**📈 对比分析**

通过与基线交互策略比较，验证 ODE 预测精度（MSE<0.07），实验显示在能力对齐时优先连接提升集体准确率，在能力不对齐时削弱连接能提升准确率；中心代理可放大真相或幻觉传播

**⚠️ 局限性**

仅考虑两类代理、固定任务和网络规模有限，跨注意力参数由小样本拟合，未探究更大规模、多类型或不同学习动态的鲁棒性

---

## 541. Relation-Centric Open-Vocabulary 3D Gaussian Segmentation

**arXiv ID:** 2607.01140 | [PDF](https://arxiv.org/pdf/2607.01140v1)

**作者:** Eunsung Cha `[一作]` (Seoul National University), Jaesik Park `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练的 PairGS 方法，用于在 3D 高斯场景中实现开放词汇分割；

**💡 创新点**

创新点在于将分割问题转化为高斯之间的关系建模，利用视图贡献权重和多视图遮罩估计相似度，采用两阶段稀疏图构造和 TreeDBSCAN 构建层次聚类，实现细粒度且高效的实例分割；

**🔧 技术方法**

主要技术包括 3D Gaussian Splatting 表示、SAM+CLIP 语义特征投影、PCA 降维、k‑NN 边候选生成、基于视图贡献的相似度计算、以及 TreeDBSCAN 的层次聚类；

**📊 数据集**

使用的数据集为 LERF‑OVS（开放词汇物体选择）和 ScanNet（开放词汇场景理解）；

**📈 对比分析**

与语言基、实例特征基和点基方法对比，PairGS 在 mIoU、准确率等指标上均取得最高成绩，且快 50 倍；

**⚠️ 局限性**

局限性：依赖多视图遮罩证据，无法分离在任何遮罩中未出现的小物体；查询结果受 SAM‑CLIP 语义表达的影响，语义模糊时可能失效。

---

## 542. Towards Metric-Agnostic Trajectory Forecasting

**arXiv ID:** 2607.01133 | [PDF](https://arxiv.org/pdf/2607.01133v1)

**作者:** Markus Knoche `[一作]` (RWTH Aachen University), Bastian Leibe `[通讯]` (RWTH Aachen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了将轨迹预测任务转为学习可校准的预测分布，并通过后处理策略（TraDiE）实现多指标评估，最终在 Waymo 基准上单一模型同时取得 minFDE 与 soft mAP 等指标的 state‑of‑the‑art 结果。

**💡 创新点**

创新点在于①将轨迹预测从针对具体指标的训练转为与指标无关的概率训练；②设计基于采样的 TraDiE 策略将分布映射为 K 条路径，兼顾 minFDE、miss rate、soft mAP 等；③通过 DONUT‑NLL 负对数似然训练与广义高斯分布结合，实现单模型多指标最优。

**🔧 技术方法**

使用了概率混合模型（Laplace、广义高斯、Gaussian 规模混合）、负对数似然训练（Traj‑NLL / Step‑NLL）、Transformer 结构（DONUT）、采样式后处理（TraDiE policy）、非最大抑制对比等技术。

**📊 数据集**

主要使用 Waymo Motion Prediction Benchmark（约487k 训练场景、44k 验证、45k 测试）。

**📈 对比分析**

与现有方法（IMPACt、ensemble 等）在 Waymo 基准上对比，使用 minFDE、soft mAP、mAP、miss rate 等指标；DONUT‑NLL 在所有指标上均优于或与 state‑of‑the‑art 相当，尤其在 soft mAP、mAP 上单模型即超越集成方法。

**⚠️ 局限性**

局限性包括：额外的推理时间开销；实验主要集中在 Waymo，Argoverse 2 结果未达到预期；Step‑NLL 在某些数据集上可能过拟合，尤其时间混合权重的敏感性；未覆盖多智能体连续性约束。

---

## 543. Perceive-to-Reason: Decoupling Perception and Reasoning for Fine-Grained Visual Reasoning

**arXiv ID:** 2607.01191 | [PDF](https://arxiv.org/pdf/2607.01191v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 544. The Decode-Work Law: Margin-Governed, Provably-Exact Spatial Joins over Compressed Geometry

**arXiv ID:** 2607.01182 | [PDF](https://arxiv.org/pdf/2607.01182v1)

**作者:** Madhulatha Mandarapu `[一作]` (VaidhyaMegha Private Limited), Sandeep Kunkunuru `[通讯]` (VaidhyaMegha Private Limited)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

实现了一种在压缩几何上可证明精确的分层空间连接，利用多级 Douglas‑Peucker LOD 阶梯和两侧 Hausdorff‑margin 证书逐步解码并裁剪候选对。

**💡 创新点**

创新点包括：①提出“decode‑work law”，证明解码工作只受每对的 signed‑clearance margin 控制，与对象大小无关；②在真实 TIGER 水体数据上实现显著解码量减少（3.4–16.8 倍）；③提供完整一指令可复现的实验环境。

**🔧 技术方法**

使用多级 Douglas‑Peucker LOD 编码、Hausdorff‑margin 证书（Brinkhoff 两侧测试）、Python+Shapely/GEOS 进行精确对照、字节计数与回归分析。

**📊 数据集**

数据集包括美国人口普查 TIGER/Line 2023 的水体多边形（Jefferson, LA；St. Louis, MN；King, WA）、控制合成几何和对抗性 “interlocking comb” 样例。

**📈 对比分析**

与传统的“解码全部再精确检验”以及单层 Brinkhoff 两级近似基线比较；在 12 个自连接工作负载中，进化证书 join 平均减少 5.9 倍解码量（范围 3.4–16.8 倍），比单层基线低约 4.9 倍，且在 31 个工作负载上完全正确。

**⚠️ 局限性**

局限性包括：对真实几何的解码工作法则预测精度仅 R²≈0.55；预注册的 near‑boundary vertex 预测被拒；未对其他几何谓词给出证明；对抗性样例仍达 Ω(v) 的读取下界；仅单线程、仅 polygon intersection，内部侵蚀证书的严谨性仅通过实测而非形式证明。

---

## 545. All-out Attack: Optimal Block Withholding Under Pay-Per-Share Scheme

**arXiv ID:** 2607.01209 | [PDF](https://arxiv.org/pdf/2607.01209v1)

**作者:** Mustafa Doger `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文分析了在Pay‑Per‑Share（PPS）及其变体FPPS等基于share的矿池奖励方案下的Block Withholding（BWH）攻击，并证明该攻击在此类方案中不具备激励兼容性；作者进一步证明攻击者的最优策略是将全部算力投向受害矿池，完全隐藏已挖出的区块（All‑Out Attack, AoA），从而在难度调节后获得比正常挖矿更高的收益；

**💡 创新点**

创新点在于首次揭示基于share的矿池奖励体系中BWH攻击的激励失效，并给出最优攻击策略AoA；作者证明了即使采用高级变体（FAW、PAW等），也无法进一步提升攻击收益，并证明所有网络矿工在难度调整后收益相同、而受害矿池运营者将承担全部损失；

**🔧 技术方法**

使用了矿池收益变化分析框架，结合比特币等长链协议的难度调节算法，构建了收益变更模型并进行理论证明；同时通过数值模拟验证了理论结果；

**📊 数据集**

本文未使用实际矿池数据集，而是采用了符号参数（如α、β、Z等）进行理论推导与仿真；

**📈 对比分析**

作者通过将BWH攻击在传统块依赖奖励方案与share‑based方案下的收益变化进行对比，利用收益变更率（δ^BWH_1−1＝α/(1−α)）量化攻击效果；仿真显示share‑based方案对攻击更敏感，攻击者在难度调整后可实现α/(1−α)的相对收益提升；

**⚠️ 局限性**

研究假设矿池维持共享难度不变、网络中无叔块（uncle）且所有矿工除攻击者外均诚实挖矿；未考虑传播延迟、矿池内部奖励分配细节以及实际矿池运营数据等因素，限制了结果的直接适用性。

---

## 546. Distill to Detect: Exposing Stealth Biases in LLMs through Cartridge Distillation

**arXiv ID:** 2607.01208 | [PDF](https://arxiv.org/pdf/2607.01208v1)

**作者:** Shayan Talaei `[一作]` (Stanford University), Amin Saberi `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种“Distill to Detect”方法，通过把怀疑模型与基模型的分布性差异蒸馏成一个低容量KV‑缓存前缀适配器（cartridge），从而放大隐藏的偏好偏移，使其在生成文本中可被现有检测器识别。

**💡 创新点**

创新点在于利用前缀适配器的容量瓶颈来聚焦低秩、可解释的偏好信号，理论上证明其是对 Fisher‑加权投影的实现，并通过实验验证其比传统LoRA或全模型蒸馏更能有效提升偏见可检测率。

**🔧 技术方法**

核心技术包括基于前缀调优（prefix tuning）的KV‑缓存适配器、对数几率分布的 KL 目标蒸馏、Fisher‑加权低秩投影理论分析，以及使用 Petri 和 AuditBench Investigator 等无标签的审计智能体进行检测。

**📊 数据集**

使用两类偏见注入（动物偏好：猫/猫猫；品牌偏好：Fanta）通过在 Llama‑3.2‑3B‑Instruct 和 Qwen3‑4B‑Instruct 上的上下文蒸馏实现；蒸馏训练使用 5k Alpaca 说明性提示；评估数据包括 50 条偏好问题和 60 条非相关开放式提问。

**📈 对比分析**

与 LoRA（低秩权重适配器）和全模型蒸馏相比，cartridge 蒸馏在检测率上大幅提升：在 Llama 上，owl 偏见检测从 37% 提升至 70%，Fanta 从 33% 提升至 100%；在 Qwen3 上亦显著提升。实验进一步显示容量从 4 至 64 词时检测率呈倒U型，峰值在 16 词。

**⚠️ 局限性**

局限性包括：需要灰盒访问（既有怀疑模型又有基模型的 logits），主要适用于低秩、由上下文蒸馏注入的偏见；对高秩注入（如数据中毒、后门触发器）可能效果不佳；且仅是放大而非直接检测，需配合现有审计工具。

---

## 547. Decision-Aware Training for Sample-Based Generative Models

**arXiv ID:** 2607.01171 | [PDF](https://arxiv.org/pdf/2607.01171v1)

**作者:** Kornelius Raeth `[一作]` (University of Tübingen), Nicole Ludwig `[通讯]` (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了决策感知训练（Decision-Aware Training），将能量得分与可微的决策损失结合，训练样本生成模型以降低决策成本并保持完整的概率分布。

**💡 创新点**

创新点在于：1) 证明决策损失本身是一个适当的评分规则；2) 将其与严格适当的能量得分线性组合，理论上仍保持严格适当；3) 通过可微优化层与隐式微分将决策损失梯度反向传播至模型参数，既提升决策质量又避免分布退化。

**🔧 技术方法**

使用的技术包括：能量得分（Energy Score / CRPS）作为基础评分；可微优化层与隐式微分实现对最优动作的梯度传递；两类样本生成模型——隐式生成模型（MLP）和分布式扩散模型；多重采样、梯度归一化与正则化。

**📊 数据集**

实验数据集包括：1) 合成双峰混合分布任务；2) 北海风电场的风电功率调度任务（6h 预报，ERA5重分析数据）；3) 柏林霜冻防护任务（24h 预报，ERA5 冬季数据）。

**📈 对比分析**

与仅使用能量得分的基线相比，实验显示：在风电切断区，CRPS 下降约10%，决策成本下降约18%，决策校准提升约32%；在合成任务中，决策损失显著下降且模式权重更准确；在霜冻任务中，决策校准提升约15%，但整体 CRPS 稍有下降。实验还展示了不同决策权重（w_d）对性能的折中影响。

**⚠️ 局限性**

限制与挑战：1) 每个决策者需单独训练模型，决策权重 w_d 需要手工调优；2) 高权重可能导致分布退化；3) 训练过程增加优化层开销；4) 对扩散模型逆链的理论解释仍不完整；5) 若成本函数偏见，模型可能放大偏见。

---

## 548. SD-RouteFusion: Ego-Trajectory Prediction with SD-Map Route Conditioning

**arXiv ID:** 2607.01139 | [PDF](https://arxiv.org/pdf/2607.01139v1)

**作者:** Sviatoslav Voloshyn `[一作]` (Zenseact), Junsheng Fu `[通讯]` (Zenseact)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种可部署的端到端自车轨迹预测框架 SD-RouteFusion，融合前视摄像头、车辆运动学和标准地图生成的导航路线。

**💡 创新点**

创新点在于：①使用可扩展的 SD 地图路线作为长时程语义先验；②提出双分支（路线导向与图像导向）与门控分类器，实现对路线失真与视觉不确定性的鲁棒处理。

**🔧 技术方法**

技术包括端到端卷积网络（ResNet‑18）+ BEV 投影（Lift‑Splat‑Shoot）、GRU 编码运动学、双向交叉注意力、轻量化 MLP 以及自监督二分类门控训练。

**📊 数据集**

使用内部扩展的 Zenseact Open Dataset（ZOD）480k 场景，并在公开 ZOD 进行复现；通过 OpenStreetMap 生成 SD 路线。

**📈 对比分析**

与图像+运动学基线、早期融合基线以及路面约束常数速度模型对比；在 8 秒预测期上 ADE 减少 16.9%，FDE 亦显著下降；启用 SD 路线提升约 10.5% ADE、19.2% FDE；门控策略比早期融合提升约 7.1% ADE。

**⚠️ 局限性**

主要限制：SD 路线生成受定位漂移和地图陈旧影响；数据集未公开完整，导致无法与使用 HD 地图或跟踪检测的先进方法直接比较；模型在长期部署中的漂移与地图不一致情况仍需进一步验证。

---

## 549. NPUsper: Eliminating Redundant Computation for Real-Time Whisper on Mobile NPUs

**arXiv ID:** 2607.01108 | [PDF](https://arxiv.org/pdf/2607.01108v1)

**作者:** Sihyeon Lee `[一作]` (Korea University), Seyeon Kim `[通讯]` (Korea University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一个面向移动 NPUs 的 Whisper 实时语音转写系统，通过消除冗余计算实现低延迟和低能耗。

**💡 创新点**

创新点包括：① 通过分析解码器交叉注意力的时间对齐特征在线检测“hallucination”并去除填充，从而避免长填充导致的重复计算；② 引入“controlled unrolling”，将自回归解码拆分为 K 步块图，既减少 KV‑cache 的无用计算，又降低图调度开销；③ 在 NPU 上实现上述两项优化，显著提升吞吐率与能效。

**🔧 技术方法**

核心技术包括 Whisper 模型的交叉注意力分析、时间对齐检测算法、动态音频缓冲管理、NPU 静态图编译与块级解码、动态时间规整 (DTW) 用于 carry‑over 更新。

**📊 数据集**

使用了 TED‑LIUM 3（两段长音频）和 Meanwhile 语料库进行评测，覆盖长形式语音。

**📈 对比分析**

与 Whisper‑Streaming、WhisperFlow、SimulStreaming、Simul‑Whisper 等现有方案对比，系统在 Samsung Galaxy S25 和 Snapdragon X Plus 上实现了：每词延迟降低 1.7–4.8 倍，首词延迟（TTFT）降低 2.2–33.2 倍，平均功耗降低 6.3–8.8 倍，能耗与吞吐率均位居榜首。

**⚠️ 局限性**

局限性：仅针对 Whisper 基础模型，未对大规模模型（如 Canary、SeamlessM4T）进行系统适配；系统侧重点是效率提升，未修改 Whisper 架构；hallucination 检测依赖交叉注意力对齐，在极端噪声或说话人变更场景下可能失效。

---

## 550. Autonomous Scientific Discovery via Iterative Meta-Reflection

**arXiv ID:** 2607.01131 | [PDF](https://arxiv.org/pdf/2607.01131v1)

**作者:** Bingchen Zhao `[一作]` (University of Edinburgh), Oisin Mac Aodha `[通讯]` (University of Edinburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于大型语言模型的开环科学发现框架，能够在无预设研究问题的情况下，自动生成可执行的统计假设代码，对原始多模态数据（表格+图像）进行验证，并周期性地通过元反思（Reflect）分析已接受/拒绝的结论，指导后续假设生成。

**💡 创新点**

创新点：①引入“第二阶”元反思机制，系统能够把已发现的结论当作经验数据来识别空白、混杂与复合模式，从而动态重定向搜索；②使用可执行代码作为假设空间，实现对任意统计检验的自适应表达；③在开环探索中结合多模态工具（如视觉语言模型）获取图像信息；④提出开放式评测基准（iNatDisco）和反事实评测，验证发现的真实性。

**🔧 技术方法**

主要技术：大型语言模型（Claude Sonnet 4.6、Claude Opus 4.6、GPT‑5.4 等）负责假设生成和反思；Python 代码执行统计测试并进行留一验证；工具调用机制（如 VLM）用于图像特征提取；多模态数据融合；反思模块基于 LLM 对已接受/拒绝结论生成引导文本。

**📊 数据集**

使用数据集：iNatDisco‑800（800 条观测，9 条已验证生态模式）和 iNatDisco‑50K（50,000 条观测，12 条已验证生态模式），以及构造的反事实版本 iNatDisco‑800‑CF；还使用了基准数据如 PC、NOTEARS、GPT‑4 BFS、HeurekaBench、ExperiGen 等进行对比。

**📈 对比分析**

对比方法：与传统因果发现（PC、NOTEARS）、基于 LLM 的 BFS、指导性 LLM 方案（HeurekaBench、ExperiGen）以及无 Reflect 的 ablation 进行对比。结果显示，本框架在 iNatDisco‑800 上召回 8/9 规律，支持率 72.7%；在 iNatDisco‑50K 上召回 8/12，支持率 74.2%，明显优于其他方法（3/9 或 1/12 召回，支持率无）。反思模块提升召回率并改善支持率。

**⚠️ 局限性**

局限性：①发现结果受训练数据与真实世界知识偏差影响，需人工验证；②对大规模、质量欠佳的多模态数据仍可能产生误差；③仅在观察性数据上工作，缺乏实验反馈；④框架对 LLM 生成代码的可靠性和执行效率有一定依赖；⑤尚未在更广泛的学科和复杂因果网络上进行验证。

---

## 551. TiRex-2: Generalizing TiRex to Multivariate Data and Streaming

**arXiv ID:** 2607.01204 | [PDF](https://arxiv.org/pdf/2607.01204v1)

**作者:** Patrick Podest `[一作]` (JKU Linz), Sepp Hochreiter `[通讯]` (JKU Linz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于xLSTM的多变量时间序列预测框架—-2，兼容过去与未来已知协变量，并支持常数成本的流式推理。

**💡 创新点**

创新点在于双向时间混合器与不对称组注意力的结合，使模型在保持目标因果性的同时充分利用未来协变量，并通过合成多变量耦合预训练提升零样本跨域性能。

**🔧 技术方法**

采用xLSTM（mLSTM/sLSTM混合块）、Patch Embedding、反向时间混合、分组注意力、二进制感知归一化等技术实现高效建模。

**📊 数据集**

使用公开基准如GIFT‑Eval、TSDBench等，以及自构建的合成多变量数据集进行训练与评测。

**📈 对比分析**

与Chronos‑2、TimesFM‑2.5、FlowState等多种TSFM进行零样本对比，-2在GIFT‑Eval和TSDBench上均取得最高的MASE/CRPS等指标，表现最优。

**⚠️ 局限性**

局限性包括需要重新计算未来已知协变量以实现完整流式推理，以及对实时变化的协变量适应性有限。

---

## 552. Sensorless Four-Channel Control Architecture Using Inverse Dynamics Modeling for Human-Scale Bilateral Teleoperation

**arXiv ID:** 2607.01201 | [PDF](https://arxiv.org/pdf/2607.01201v1)

**作者:** Amir Noohian `[一作]` (University of Alberta), Martin Jagersand `[通讯]` (University of Alberta)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了无传感器的四通道双向遥操作架构，利用逆动力学模型实现外部关节力估计并实现透明度提升；

**💡 创新点**

创新点在于消除了对力/扭矩传感器的依赖，利用逆动力学实现关节层面的外力检测，从而在全身接触场景中实现更高的传输阻抗与透明度；

**🔧 技术方法**

主要技术包括逆动力学参数辨识、基于基参数的最小二乘估计、关节级 PID 位置控制、全局力反馈以及外部力估计；

**📊 数据集**

实验数据来自自定义的 WAM 双臂遥操作平台，使用四自由度领导臂与七自由度跟随臂，在门开启与负载测试中收集关节力与位移；

**📈 对比分析**

与传统的两通道、四通道及基于重力补偿的方案进行比较，实验显示在自由运动与硬接触下，所提四通道架构在位置跟踪、操作者阻抗与最大可传输阻抗方面均显著优于基线，且外部力估计误差保持低；

**⚠️ 局限性**

局限性包括未考虑通信延迟、对摩擦建模的简化（仅保留粘性摩擦）、缺乏在线自适应机制，可能在极低速或细粒度操作中透明度下降。

---

## 553. Online Fair Division Meets Reordering Buffers

**arXiv ID:** 2607.01159 | [PDF](https://arxiv.org/pdf/2607.01159v1)

**作者:** Georgios Amanatidis `[一作]` (Athens University of Economics and Business), Nicos Protopapas `[通讯]` (Athens University of Economics and Business)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在线分配不可分混合物品（既可为好物也可为差物）给具加性效用的代理人，提出在算法可使用有限缓冲区的模型下，设计出能够在每个时间步实现EF1且每隔若干步实现EF的在线分配算法；

**💡 创新点**

创新点在于：①将缓冲区概念引入在线公平分配，形成从完全在线到完全离线的中间模型；②利用Hall定理与“顶点优选图”构造一系列可行匹配，证明在个体最多k种价值层的情况下，只需缓冲区大小O((n‑1)k)即可实现强大的时序公平性；③通过离散化技术将结果推广至一般加性好物/差物实例，并给出对应的近似保证；

**🔧 技术方法**

主要技术包括：离散化（将价值范围压缩为k个水平）、顶点优选图（Top Choice Graph）构造、Hall定理的迭代删减法、双向匹配与“批量分配”模式、以及双轮转（Double Round‑Robin）线性时间分配。

**📊 数据集**

该工作为理论性质，不涉及具体数据集；所有结果均为严谨的数学证明与上界/下界分析。

**📈 对比分析**

评价方式为理论性能分析：在给定k值和n的情况下证明算法在每一步满足EF1，每n步保证EF；证明了若缓冲区不足则存在不可避免的公平性破坏；与现有无缓冲模型的极限结果做对比，展示缓冲区显著提升了公平性保证。

**⚠️ 局限性**

局限性包括：缓冲区大小仍随k与n线性增长，若要进一步减小缓冲区需放宽公平性或引入前瞻信息；对一般混合物品的近似比例仍受最大价值比ρ影响；无法解决完全在线无缓冲的强势不可行性问题。

---

## 554. Skills Are Not Islands: Measuring Dependency and Risk in Agent Skill Supply Chains

**arXiv ID:** 2607.01136 | [PDF](https://arxiv.org/pdf/2607.01136v1)

**作者:** Changguo Jia `[一作]` (Peking University), Minghui Zhou `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Agent Skill Supply Chains（ASSCs）概念，并开发了SkillDepAnalyzer工具从1.43M个Agent Skill中自动构建完整的技能-包-服务依赖图（SkillBOM），实现对技能依赖的精准识别与可追溯；

**💡 创新点**

创新点在于将技能视为一类特殊的可重用软件资产，构建混合技能、软件包与外部服务的多通道依赖图；提出Typed Dependency Manifest、Dependency-Cluster管理和风险审计指令等治理方案；

**🔧 技术方法**

技术方法包括结构化解析前置信息、基于上下文的证据校准、增量BOM构建、图合并与规范化，以及使用SBOM IR实现SkillBOM兼容性；

**📊 数据集**

使用SkillsMP公开注册库中的1.43M个GitHub技能作为实验数据，并构建了500条单层与100条多层的SKILL‑DEP人工标注基准；

**📈 对比分析**

与传统SBOM生成器（Syft、ScanCode、ORT、Microsoft sbom-tool、Cdxgen）以及LLM（DeepSeek‑v4‑pro）对比，SkillDepAnalyzer在单层基准上F1达0.95，包依赖F1 0.93，显著优于所有基线；在多层基准上精度0.98、召回0.93、F1 0.95，展示其完整依赖图构建的高准确性；

**⚠️ 局限性**

局限性包括依赖提取仍受低置信度候选的误差影响；研究只覆盖公开GitHub技能，未涵盖私有/企业技能；随着技能生态快速演进，结构模式可能随时间变化；

---

## 555. $\text{Log}_\text{b}$Quant: Quantizing Language Models in Logarithmic Space

**arXiv ID:** 2607.01127 | [PDF](https://arxiv.org/pdf/2607.01127v1)

**作者:** Jeremias Bohn `[一作]` (Technical University of Munich), Georg Groh `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种可调基数的对数量化方法Log_bQuant，用于将大型语言模型的权重压缩到4-bit，同时保持大部分性能。

**💡 创新点**

创新点在于对数量化与能量基剪枝相结合，能够让量化码本更好匹配权重分布，尤其在低位宽时显著降低高幅值低频权重误差。

**🔧 技术方法**

使用了GPTQ后训练量化、对数量化方案、能量剪枝、FLUTE查表以及int8压缩等技术。

**📊 数据集**

校准数据使用WikiText-2，评估数据集包括MMLU、PIQA、ARC‑C、WikiText‑2、PennTreebank和C4。

**📈 对比分析**

在8‑bit和4‑bit条件下与标准不对称线性量化比较，4‑bit Log_bQuant在下游任务保持≈90–95%原始精度，内存可节省约60–65%，吞吐略有提升；在8‑bit时两者差距减小。

**⚠️ 局限性**

局限性包括8‑bit时无明显优势、量化时需要解量化开销、未对激活或KV缓存做量化、极低位宽推广仍待验证。

---

## 556. MoHallBench: A Benchmark for Motion Hallucination in Video Large Language Models

**arXiv ID:** 2607.01117 | [PDF](https://arxiv.org/pdf/2607.01117v1)

**作者:** Jiale Li `[一作]` (Xiamen University), Mengyuan Liu `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MoHallBench——一个专门评估视频大语言模型（VideoLLMs）在“运动幻觉”方面的诊断基准，并通过二进制、选择题和生成三种评估模式以及双向提问协议来降低肯定偏差。

**💡 创新点**

创新点在于：①将运动幻觉拆分为共现先验、相似性混淆和顺序推理三大维度，系统构造对抗样本；②引入双向提问与偏差感知指标（Acc_PS/NS、Q-PairAcc、Cons）实现更可靠的性能评估；③提供多级语义层级与视觉相似度拆分，揭示细粒度混淆导致的幻觉。

**🔧 技术方法**

主要技术手段包括条件概率与PMI计算、事件序列切分、三层语义层级构建、视觉相似度重排序、统一采样与无温度解码的推理；并在评估时使用专门设计的偏差消除指标。

**📊 数据集**

使用了 Charades、HAA500、HACS、FineDiving、TAPOS 等多来源视频数据，共计 11,306 条视频与 40,493 个问答对，覆盖三大幻觉轴并保证正负样本均衡。

**📈 对比分析**

对十款 7B–8B 规模的开源 VideoLLMs 进行基准测试。结果显示：正样本准确率高（约 80%+），但负样本准确率低（约 30%+），Q-PairAcc 仅 20%–30%；顺序推理幻觉最严重，甚至最强模型 Q-PairAcc 仅 29%；共现与相似性幻觉表现为正负差距显著，说明模型对先验和相似度高度依赖。

**⚠️ 局限性**

局限性：仅关注人类动作幻觉，未覆盖物体状态变化或长时序事件推理；对抗样本构造多为人工挑选，可能与开放世界幻觉多样性不完全匹配；该工作聚焦诊断，未提出有效的幻觉缓解策略。

---

## 557. Definability of Functional Properties in the Basic Modal-Temporal Language over Ordered Frames

**arXiv ID:** 2607.01110 | [PDF](https://arxiv.org/pdf/2607.01110v1)

**作者:** Alfredo Burrieza `[一作]` `[通讯]` (University of Malaga), Alfredo Burrieza (University of Malaga)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在基于基本模态-时态语言的框架下，系统性研究了九种函数性质（全称性、非全称性、单射、满射、单调性、严格单调性、反单调性、严格反单调性、恒等性）的可定义性，覆盖预序、严格预序、偏序、线性序及严格线性序五种阶类型；通过最小功能框架（仅两条流、单一可达函数）和严格时态解释（G*,H*），进一步揭示了可定义性的精细划分；并与索引语言和统一域（U-Dom）框架进行对比，验证三种控制函数多重性的方式得到相同的可定义性模式。

**💡 创新点**

1) 首次把函数多重性拆解为三种结构化方式（最小框架、索引语言、统一域）并证明它们在所有阶类型下产生相同的可定义性结果；2) 通过引入严格时态解释，显著扩展了可定义的属性集合；3) 在理论上精确描述了不可定义性的根源——功能多重性与阶结构缺乏连通性。

**🔧 技术方法**

- 逻辑语义学与框架构造（功能框架、最小框架）
- 模型论工具（p‑morphism、功能化逼真、保真映射）
- 代数化简（↑,↑*,↓,↓* 等区间运算）
- 逻辑定义与证明技术（公式构造、对偶性、归纳证明）

**📊 数据集**

无数据集；本研究纯粹为理论性、逻辑性质的分析与证明。

**📈 对比分析**

通过与已知索引语言和统一域框架中的可定义性结果进行对比，展示三者在表达力上等价；不涉及实验或性能指标，主要通过形式化证明和反例展示。

**⚠️ 局限性**

- 在非线性阶（预序、严格预序、偏序）中，核心属性（全称性、单射、满射、恒等性）仍不可定义。 
- 仅在最小框架和严格时态解释下，才能部分恢复可定义性；但仍无法在所有阶类型上实现完整可定义。 
- 结果受限于时态算子局部性与函数多重性，尚未探索更强连通或全局算子（如通用模态）下的可定义性。

---

## 558. Sequentially-Controlled Interactive Multi-Particle Flow-Maps for Online Feedback-Driven Search

**arXiv ID:** 2607.01144 | [PDF](https://arxiv.org/pdf/2607.01144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 559. Right in the Right Way: LM Training with Verifiable Rewards and Human Demonstrations

**arXiv ID:** 2607.01181 | [PDF](https://arxiv.org/pdf/2607.01181v1)

**作者:** Mehul Damani `[一作]` (Massachusetts Institute of Technology), Jacob Andreas `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了VARL框架，将可验证奖励与对抗生成器-鉴别器相结合，以同时提升任务正确率和生成质量。

**💡 创新点**

创新点在于通过乘法门控奖励将可验证奖励与鉴别器奖励耦合、利用特征空间引导对抗学习，并显著降低奖励劫持。

**🔧 技术方法**

使用了RLVR（GRPO）、对抗学习（GAN/GAIL）以及可选特征映射ϕ与门控奖励结构。

**📊 数据集**

在三大数据集上验证：bug修复（RunBugRun）、故事生成（WritingPrompts Curated）和计数式算术（Countdown‑Code）。

**📈 对比分析**

与SFT、RLVR、鉴别器单独训练以及KL正则化等基线比较，VARL在保持或提升准确率的同时提升多样性、风格一致性，显著减少奖励劫持，表现优于所有基线。

**⚠️ 局限性**

主要限制包括训练不稳定、对特征空间设计高度依赖，以及需要可验证奖励和足够多的演示示例。

---

## 560. Diffusion-GR2: Diffusion Generative Reasoning Re-ranker

**arXiv ID:** 2607.01170 | [PDF](https://arxiv.org/pdf/2607.01170v1)

**作者:** Zhuoxuan Zhang `[一作]` (UNC Chapel Hill), Xi Liu `[通讯]` (Meta AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将自回归（AR）推理重排器GR2转换为块扩散（block‑diffusion）模型Diffusion‑GR2，以实现并行解码并显著提升推理吞吐量，同时保持几乎相同的重排准确性。

**💡 创新点**

创新点在于：① 通过转换微调（Conversion Fine‑Tuning, CFT）让扩散模型自动学习产生合法排列；② 使用自我上游蒸馏（On‑Policy Distillation, OPD）消除离线训练与推理分布不匹配；③ 在此基础上加入强化学习（RL）进一步逼近AR教师的排名质量。

**🔧 技术方法**

技术包括：基于Qwen3‑8B的LLM、块扩散语言模型（block‑diffusion），CFT、OPD、RL（TraceRL + DAPO）等；同时采用了置信阈值控制并行度，利用KV缓存实现推理加速。

**📊 数据集**

使用Amazon Review Beauty数据集（TIGER协议下的5‑core过滤，单用户10候选列表），共1615个测试用户。

**📈 对比分析**

与AR GR2教师和检索器基线对比：Recall@1从0.2811提升至0.2951（≈99.5%教师水平），Recall@3和NDCG@3甚至略超教师；在单张H100‑80G GPU上，推理吞吐量提升2.4–3.5×（如τ=0.9时Recall@1≈0.2950，吞吐率≈172 tok/s）。

**⚠️ 局限性**

局限性：① 仅在Amazon Beauty上验证，跨域性能未知；② 需先有高质量的AR教师模型；③ 置信阈值调优仍需经验；④ 对极短推理长度或极大候选集的加速优势有限。

---

## 561. Adversarial Pragmatics for AI Safety Evaluation: A Benchmark for Instruction Conflict, Embedded Commands, and Policy Ambiguity

**arXiv ID:** 2607.01153 | [PDF](https://arxiv.org/pdf/2607.01153v1)

**作者:** Brett Reynolds `[一作]` `[通讯]` (Humber Polytechnic), Brett Reynolds (Humber Polytechnic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为“Adversarial Pragmatics”的安全评估基准，提供了对语言指令、引用、权限、范围、指代、间接语气等语言学维度的对照测试，旨在细化模型在指令冲突与安全边界下的行为判断。

**💡 创新点**

创新点在于将安全评估从粗糙的通过/失败标签拆分为多维度标签（任务成功、政策遵从、安全风险、拒绝类型、评估者置信度等），并设计了可审计的对照对（minimal pair）与完整的评估流程与度量指标；同时提出了“对抗性语用学”这一新的评价范式。

**🔧 技术方法**

使用的技术包括：基于Llama的本地Ollama模型推理、JSON模式验证器、规则辅助诊断脚本、专家评估协议、LLM-as-Judge 验证脚本以及一套八个指标的度量体系；所有脚本均以Python实现并可在GitHub仓库复现。

**📊 数据集**

数据集为18条种子项目（每条有两种对照形式），共54条模型输出（每条模型对应两种对照），采用蓝色、绿色等无害占位符以及假秘密标记，避免真实敏感内容；这些种子通过schema validator和手工注释构成基准。

**📈 对比分析**

对比方法：本地种子试验使用三款Ollama模型进行推理，随后由专家评估每个输出；LLM-as-Judge在同一数据集上给出标签并与专家标签比较。实验结果显示：任务成功率≈66.7%，政策遵从≈88.9%，安全风险≈72.2%；LLM评估在拒绝判定上表现最好（≈98%），但在任务成功和失败归因等方面与专家差距显著。

**⚠️ 局限性**

局限性：样本量仅18条种子，使用的是无害占位符且未包含真实攻击场景；仅评估了本地Ollama模型，缺乏大模型或跨语言验证；缺乏最终的心理测量学验证和可靠性报告，导致基准的泛化性与稳健性仍待进一步扩展。

---

## 562. Next-Generation Agentic Reinforcement Learning Systems Enable Self-Evolving Agents

**arXiv ID:** 2607.01120 | [PDF](https://arxiv.org/pdf/2607.01120v1)

**作者:** Ran Yan `[一作]` (Ant Group), Binhang Yuan `[通讯]` (Ant Group)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了面向企业的自演化代理系统框架，包含标准化轨迹协议、企业级数据代理和统一演化控制平面，并在AReaL基础上实现了在线策略权重更新的原型

**💡 创新点**

将代理轨迹标准化为RL级别事件协议、构建跨框架数据捕获代理，并将演化决策视为治理驱动的多表面控制平面，突破传统离线微调与自我改进分离的瓶颈

**🔧 技术方法**

基于ATDP协议的事件记录、跨框架的代理捕获、微服务化的RL训练循环（使用AReaL、Megatron/FSDP等），以及PPO/DPO等在线RL算法

**📊 数据集**

主要使用企业生产代理交互日志作为轨迹数据，未公开标准数据集，实验以内部运营数据为主

**📈 对比分析**

论文未给出定量实验结果，仅在实验环境中验证原型可行性，未提供性能对比

**⚠️ 局限性**

仅实现了策略权重更新路径，缺乏完整ATDP实现、跨表面演化控制、回放与治理机制，未覆盖内存、技能、工具等演化表面

---

## 563. FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement

**arXiv ID:** 2607.01111 | [PDF](https://arxiv.org/pdf/2607.01111v1)

**作者:** Haoran Hao `[一作]` (Carnegie Mellon University), Jeff Schneider `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为FAR的框架，允许机器人在测试阶段从失败中学习并自适应，结合失败对比偏好适配与轻量级动作扰动实现多次重试与恢复；

**💡 创新点**

创新点在于：①利用失败经验构造对比偏好学习样本，使策略在回退状态中避免重复错误；②在重试过程中引入轻量扰动以局部探索；③将成功恢复轨迹用于持续在线改进，提升数据效率并减少环境重置；

**🔧 技术方法**

核心技术包括：基于Diffusion Policy的生成式策略；IQL式保守价值估计；对比偏好适配（pairwise preference loss）；动作扰动（高斯噪声与指数平滑）；以及基于优势加权的在线更新；

**📊 数据集**

实验使用了ManiSkill、RoboSuite、RoboMimic三大仿真基准（共9个视觉操控任务）以及xArm 7-DoF真实机器人进行3个抓取、放置、倒水任务；

**📈 对比分析**

与基线DP、DP‑NR、DP‑BGR比较，FAR在仿真中平均提升约16.4%，在真实世界提升约11.7%；在持续学习实验中FAR在数据效率和重置成本上优于其他方法，尤其在高难度任务中表现突出；

**⚠️ 局限性**

局限性包括：仅在单任务Diffusion Policy上验证；未加入专门的失败检测机制，依赖环境反馈；未来工作需扩展到VLA等预训练模型与多任务场景，及引入VLM实现阶段级恢复。

---

## 564. Technical Report: Asynchronous Distributed Trajectory Estimation of Multi-Robot Systems

**arXiv ID:** 2607.01106 | [PDF](https://arxiv.org/pdf/2607.01106v1)

**作者:** Adam Pooley `[一作]`, Matthew Hale `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了异步块坐标下降算法用于分布式轨迹估计，并通过对MAP问题的近似实现通信稀疏化，支持滑动窗口轨迹优化。

**💡 创新点**

通过稀疏信息矩阵近似显著降低通信量（可达96.9%），并证明在部分异步条件下指数收敛；首次在真实移动机器人上验证异步分布式估计。

**🔧 技术方法**

使用异步块坐标下降（asynchronous BCD）、最大后验（MAP）估计、滑动窗口轨迹优化、稀疏信息矩阵近似以及分布式优化收敛分析。

**📊 数据集**

仿真使用多达128个holonomic机器人的二维状态数据；实验使用Robotarium平台上4台机器人在多路径点之间导航的真实轨迹。

**📈 对比分析**

与同步DRWT算法对比，采用相同步长（DRWT步长1，本文步长1e-3），在不同N、B和计算量下测量MAP子最优性误差；结果显示本文误差比DRWT低多达64%，且与近似MAP误差仅差4.4e-4；硬件实验显示误差随迭代指数下降，B越小收敛越快。

**⚠️ 局限性**

需已知通信延迟上界B；使用固定步长，未实现自适应步长；稀疏近似在极端动态或高噪声场景下可能引入更大误差；实验规模有限，未验证大规模网络或异构传感器的鲁棒性。

---

## 565. SynLaD: Latent Diffusion for Generating Synthesizable Molecules Conditioned on 3D Pharmacophore Profiles

**arXiv ID:** 2607.01105 | [PDF](https://arxiv.org/pdf/2607.01105v1)

**作者:** Miruna Cretu `[一作]` (University of Cambridge), Colin Grambow `[通讯]` (Prescient Design (AI for Drug Discovery), Genentech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种多头潜在扩散框架，可同时生成符合3D药效团特征且可合成的分子结构和对应的反应路径。

**💡 创新点**

将3D结构重建与合成路线生成联合训练至共享潜在空间，并通过潜在扩散模型在药效团条件下采样，实现药效团对齐与可合成性的双重控制。

**🔧 技术方法**

采用变分自编码器+双头解码器、Diffusion Transformer（DiT）进行潜在空间扩散、药效团嵌入交叉注意力和反应预测 Transformer，融合3D坐标与合成路径生成。

**📊 数据集**

训练使用USPTO 67,512条合成路线及对应分子构象；评估使用Lit-PCBA benchmark 10个靶点以及全量USPTO库做对比。

**📈 对比分析**

与传统ROCS库筛选、ShEPhERD、SynFormer、REINVENT等基线在化学空间、合成性、形状/药效团重叠等指标对比，模型在生成可合成、高Tanimoto得分、独特骨架数以及Hit率方面均显著优于基线，且相比库筛选仅需1k样本即可获得更多可合成Hit。

**⚠️ 局限性**

受限于USPTO训练集规模及反应预测模型准确率，模型在极大化学空间和复杂合成路径生成上仍有限；同时在3D重建与合成解码器对齐上需更多数据验证。

---

## 566. Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training

**arXiv ID:** 2607.01232 | [PDF](https://arxiv.org/pdf/2607.01232v1)

**作者:** Zijian Zhang `[一作]` (University of Minnesota), Mingyi Hong `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了强化学习后训练在不同Transformer层的贡献，发现单层训练即可恢复大部分甚至超过全参数训练的收益；

**💡 创新点**

提出层贡献度量和层级感知训练策略，揭示RL收益集中在中间层并可利用此结构提升训练效果；

**🔧 技术方法**

采用单层训练、GRPO、GiGPO、Dr.GRPO等RL算法、层贡献度量、学习率适配、层选择和多模型投票等技术；

**📊 数据集**

使用NuminaMath-CoT、DeepScaleR、DeepCoder、ALFWorld、Skywork等多任务数据集；

**📈 对比分析**

通过层贡献度量与全参数RL对比，单层或中层训练在数学、编码、代理任务上均可提升性能（比全参高约10–30%），并通过投票进一步提高；

**⚠️ 局限性**

主要局限在于仅在数学推理任务充分验证，其他任务的泛化与层贡献度量对不同训练配置的依赖性尚未彻底理论化。

---

## 567. Ink3D: Sculpting 3D Assets with Extremely Complex Textures via Video Generative Models

**arXiv ID:** 2607.01222 | [PDF](https://arxiv.org/pdf/2607.01222v1)

**作者:** Yue Han `[一作]` (ZGCA & ZGCI), Yan Lu `[通讯]` (Microsoft Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过先使用3D生成模型得到白网格几何，再用条件视频生成模型OrbitPainter产生360°轨道扫描视频，最后利用TextureOptimizer将多视角图像烘焙成高质量、极复杂纹理的3D资产。

**💡 创新点**

①将几何与纹理的生成分离，充分利用大规模视频生成模型提供连续、互相一致的多视角纹理信息；②提出Geometry‑Aware Sparse Attention，在视频生成中利用3D几何先验来约束注意力；③研发TextureOptimizer神经烘焙框架，结合图形匹配与Trellis‑2去噪先验，有效抑制几何不一致导致的纹理伪影。

**🔧 技术方法**

使用3D VAE+latent diffusion（Trellis‑2）、视频VAE+Flow Matching（OrbitPainter）、Geometry‑Aware Sparse Attention、图匹配+MRF的纹理体素化、Treillis‑2去噪引导、LoRA微调、CUDA GPU加速等技术。

**📊 数据集**

OrbitPainter在Objaverse与Sketchfab的60K带纹理3D样本上训练；评估使用Texverse 85个高质量样本的白网格与Nano Banana生成的复杂参考纹理图结合；实验采用Nano Banana生成的参考图作为输入。

**📈 对比分析**

与Trellis‑2、Paint3D、TexGen、SeqTex等开源方法在FID、CLIP‑FID、LPIPS、CLIP‑I四个指标上对比，Ink3D在所有指标上均优于对手（如FID 103.7 vs Trellis‑2 120.4，CLIP‑FID 195.6 vs 223.9，LPIPS 0.2029 vs 0.2714，CLIP‑I 0.8979 vs 0.8694），表明显著提升纹理质量与真实性。

**⚠️ 局限性**

①仍受视频生成过程中的几何不一致限制，导致轨道扫描视频生成耗时较长；②依赖大量训练好的3D资产，适用范围受限；③仅处理白网格输入，无法直接兼容已有纹理或动态纹理；④对不同光照和材质的鲁棒性尚待进一步验证。

---

## 568. The State-Prediction Separation Hypothesis

**arXiv ID:** 2607.01218 | [PDF](https://arxiv.org/pdf/2607.01218v1)

**作者:** Giovanni Monea `[一作]` (Cornell University), Yoav Artzi `[通讯]` (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了Transformer中将下一词预测与状态存储功能分离的State‑Prediction Separation（SPS）结构；

**💡 创新点**

创新点在于通过在每个输入后插入专门的预测Token，将隐藏状态拆分为状态流和预测流，消除了两任务在同一表示中的冲突；

**🔧 技术方法**

采用改进的Transformer架构（RMSNorm、SwiGLU、rotary embeddings），插入额外Token并实现滑动窗口注意力掩码，实现两路计算；

**📊 数据集**

使用FineWeb‑Edu教育子集进行预训练，并在四个外部语料库（WikiText、C4、Pile‑Books3、GovReport）和五个零样本基准（ARC‑Easy、HellaSwag、PIQA、SciQ、LAMBADA）进行评估；

**📈 对比分析**

与标准Transformer及两种基线（全KV保留与预测+状态共用）对比，SPS在XS–XL规模上在验证损失、泛化NLL与零样本准确率上平均提升2–3%，训练数据效率提高约50%，推理吞吐量与显存消耗仅比基线提升6–10%；

**⚠️ 局限性**

局限性包括：需要额外计算步骤导致每步训练开销加倍；仅在单一数据集上验证，未测试混合预训练；最大规模仅1.678B参数，需进一步验证更大模型；未探索参数共享或稀疏化等进一步压缩方法。

---

## 569. AutoMem: Automated Learning of Memory as a Cognitive Skill

**arXiv ID:** 2607.01224 | [PDF](https://arxiv.org/pdf/2607.01224v1)

**作者:** Shengguang Wu `[一作]` (Stanford University), Serena Yeung-Levy `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可训练的记忆管理技能，将文件系统操作视为与任务动作同等的可执行动作，并通过两阶段元 LLM 自动化结构优化和模型微调，提升大型语言模型在长时序游戏中的表现。

**💡 创新点**

创新点在于将记忆管理分为结构（脚本、提示、文件模式）和熟练度两轴，并使用元 LLM 进行完整轨迹分析来自动迭代重构脚本与生成专门的记忆子模型，从而实现记忆技能的独立学习与提升。

**🔧 技术方法**

核心技术包括：文件系统接口作为外部记忆；元 LLM（如 Claude Opus）执行代码审阅与训练数据筛选；两层循环（结构优化与熟练度训练）；LoRA 微调的记忆专用子模型；共享对话历史与多模型推理。

**📊 数据集**

实验数据集为三款程序生成的长时序游戏：Crafter、MiniHack（NetHack 引擎子集）和完整 NetHack，使用 BALROG benchmark。

**📈 对比分析**

与多种基线比较（滑动窗口、链式思维、商业专有模型 Gemini‑3、Claude‑Opus 等），在不修改任务模型权重的情况下，仅通过记忆优化即可使 32B 开源模型在三款游戏中的进度率提升约 2–4 倍，达到与前沿专有模型相近的水平。

**⚠️ 局限性**

局限性包括：记忆仅为单剧集临时性；目前针对每个游戏需要单独的脚本与专用模型，未验证跨域共享；方法主要在游戏环境验证，真实世界复杂任务的适用性待进一步探索。

---

## 570. RepoRescue: An Empirical Study of LLM Agents on Whole-Repository Compatibility Rescue

**arXiv ID:** 2607.01213 | [PDF](https://arxiv.org/pdf/2607.01213v1)

**作者:** Zhihao Lin `[一作]` (Beihang University), Li Li `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出兼容性救援（compatibility rescue）任务，并构建基准评测框架，用来评估LLM代理在历史可用代码被现代环境破坏时的修复能力。

**💡 创新点**

首次把兼容性破坏视为完整仓库级别的任务；引入源代码仅修复评估与运行时阻断实验；设计四级推理难度层级，对代理行为进行细粒度分析。

**🔧 技术方法**

使用大型语言模型（Claude Code系列、GPT‑5.2 Codex、GLM‑5、Kimi、MiniMax）结合自定义工具框架进行代理式调试，并采用环境快照、依赖源检查、测试执行等技术。

**📊 数据集**

构建兼容性救援基准，包含193个Python（47无维护+146时间旅行快照）和122个Java仓库，满足历史环境通过、现代环境失败的条件。

**📈 对比分析**

通过全补丁成功率、源代码仅成功率、运行时阻断成功率以及多系统并集/交集进行比较；结果显示Claude Code全补丁约51%以内，源代码仅约20%；GPT‑5.2 Codex约50%；五系统并集达62.7%，表现出互补性；难度层级显示L4任务GPT‑5.2完美通过，Claude Code仅2/14。

**⚠️ 局限性**

仅评估源代码改动，排除依赖变更；基准受Python动态特性影响，Java部分对比不完全；实验基于单次试验，未充分探索提示/采样参数；缺乏真实下游使用情境验证。

---

## 571. Measuring the Gap Between Human and LLM Research Ideas

**arXiv ID:** 2607.01233 | [PDF](https://arxiv.org/pdf/2607.01233v1)

**作者:** Ziyu Chen `[一作]` (University of Chicago), Arman Cohan `[通讯]` (Yale University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一套基于文献上下文的研究创意评估框架，对比人类科研论文与大型语言模型（LLM）在同一文献背景下生成的创意，量化两者在研究机会模式与方法范式上的分布差异；

**💡 创新点**

创新点在于：①提出了双轴研究口味（opportunity pattern + method paradigm）分类体系；②通过逆向工程论文相关工作构造“人类创意”与“LLM创意”对照集；③利用自动化标注与诊断评分，系统性揭示LLM创意在桥接/合成方向的过度集中与细节特异性的缺失；

**🔧 技术方法**

技术手段包括：大语言模型提示生成、LLM辅助提取人类创意、逆向工程生成相关工作集合、LLM自动注释与诊断评分、分布距离度量（TVD、JSD、归一化熵）、聚类与向量表示分析；

**📊 数据集**

数据集为 11,683 篇机器学习与自然科学领域论文（ICLR/ICML/NeurIPS 2023‑2026，Nature Communications 2023‑2025），并从每篇论文逆向提取 4–8 篇相关工作；

**📈 对比分析**

比较方法：对人类创意与各LLM（九个模型族）生成的创意进行标签分布对比，计算 TVD、JSD 与熵；诊断评分评估表面拼接、瓶颈特异性与模板化程度；结果显示 LLM 方案在机会轴与方法轴的 TVD 均高于人类，熵低，显著聚焦于桥接与合成；

**⚠️ 局限性**

局限性包括：①数据主要来自 STEM 领域，缺乏社会科学、文科、工程设计等；②任务仅重现局部文献背景，未涵盖研究者隐性知识、实验失败、合作与评审反馈等真实科研过程；③双轴分类与诊断评分仍是离散化，可能遗漏多重研究动作；④评估仅覆盖有限模型、提示与一次性生成，未考虑交互式、检索增强或领域特定系统的影响。

---

## 572. Touching and Feeling the Data: A Reusable Software Pipeline for Tactile Statistical Graphs in Accessible Education

**arXiv ID:** 2607.01214 | [PDF](https://arxiv.org/pdf/2607.01214v1)

**作者:** Lawrence Obiuwevwi `[一作]` (Old Dominion University), Sampath Jayarathna `[通讯]` (Old Dominion University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一个三层可复用的软件流水线，能够从数据或图像快速生成3D打印触觉统计图表。

**💡 创新点**

首次将触觉参数推导、模块化图形生成与多模态LLM图像提取整合为开源流水线，实现了秒级生成。

**🔧 技术方法**

使用JavaScript（JSCAD、Three.js、Node.js）、多模态LLM（Anthropic Claude）和FDM 3D打印技术。

**📊 数据集**

主要使用教材/课堂图像进行提取测试，并基于统计图类型（散点、柱状、直方、折线、箱线）进行验证。

**📈 对比分析**

与手工Fusion 360建模对比，生成时间从约2小时缩短至60 ms；生成的STL文件大小和三角形计数满足规范，且无需修复即可直接打印。

**⚠️ 局限性**

缺点包括对图像提取的依赖需教师审核、仅支持Grade 1 Braille、图表种类有限且对数值精度敏感。

---

## 573. Language-Critique Imitation Learning from Suboptimal Demonstrations

**arXiv ID:** 2607.01225 | [PDF](https://arxiv.org/pdf/2607.01225v1)

**作者:** Chih-Han Yang `[一作]` (National Taiwan University), Shao-Hua Sun `[通讯]` (National Taiwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于自然语言批评的离线模仿学习框架（LC‑BC 与 LC‑DP），直接利用结构化语言标签指导策略学习；

**💡 创新点**

创新点在于：①将自然语言作为完整结构化监督信号，而非压缩成标量；②设计可控的语言生成器 μ_g 与可微分的语言辨识器 μ_ϕ；③提出语言批评损失，并给出对专家性能差距的理论上界；

**🔧 技术方法**

使用的技术包括：自然语言生成器 μ_g、预训练 LLM 与 MLP 项目器的组合 μ_ϕ、token‑级交叉熵语言批评损失、行为克隆与扩散策略训练、以及相关的理论分析；

**📊 数据集**

实验使用八个连续控制任务（导航、驾驶、MetaWorld 的 sweep/cover、MultiStage 推块、精细插入、Adroit 的 hammer/ball），每个任务均包含专家数据集 𝒟_E 与更大规模的通用数据集 𝒟_G；

**📈 对比分析**

与 BC、DWBC、DemoDICE、ILID、DP、LPB‑Offline、TD3+BC、CQL、Decision Transformer 等基准对比，LC‑BC/LC‑DP 在多模态、长周期、精细控制任务上显著优于或与现有方法持平，且在混合质量数据场景下表现更鲁棒；

**⚠️ 局限性**

局限性包括：① 语言标签的生成需要手工规则或特定结构，缺乏完全自动化；② 对细粒度运动指导的依赖导致 VLM 生成标签时精度不足；③ 在简单任务中提升有限；④ 仍需离线数据划分或伪专家标签来实现最优训练。

---

## 574. Theoria: Rewrite-Acceptability Verification over Informal Reasoning States

**arXiv ID:** 2607.01223 | [PDF](https://arxiv.org/pdf/2607.01223v1)

**作者:** Ben Slivinski `[一作]` (Independent Researchers), Michael Saldivar `[通讯]` (Independent Researchers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Theoria 体系，将 AI 解答重写为带有类型化证明步骤的状态转移序列，构建可审计的答案可信度验证流程。

**💡 创新点**

引入完整变更 invariant 与结构化证明 witness，使隐藏前提与伪引用显现；通过局部 LLM 判断与正式验证相结合，实现高覆盖率与可追溯性的双重优势。

**🔧 技术方法**

采用 LLM（Claude、GPT）进行求解、重写与裁判，配合 typed judges、pedantry filter、convention lift 等模块；利用统计检验（Wilson CI、McNemar 等）评估性能。

**📊 数据集**

主要在 HLE‑Verified Gold 185 条文本题目上评估，并在 GPQA Diamond 65 条多学科题目做跨域测试；另外构造 95 条对抗性（poisoned）证明以检验鲁棒性。

**📈 对比分析**

与 solver‑only、holistic‑judge（分数评估）等基线对比；在 HLE 上得到 56.8% 覆盖率、91.4% 严格精度，精度与 holistic judge 相当但错误覆盖互补；在 95 条对抗样本中检测率 94.7% 对比 83.2%；GPQA 上精度 97.1%，覆盖 52.3%。

**⚠️ 局限性**

仅适用于可形式化的推理，难以处理实验科学与统计推断；judge 仍可能误判；缺乏正式工具后端验证；初始状态审计依赖手工检查；数据集规模与多样性有限。

---

## 575. Query Complexity of Hypergraph Connectivity and Learnability using CUT Oracles

**arXiv ID:** 2607.01216 | [PDF](https://arxiv.org/pdf/2607.01216v1)

**作者:** Deeparnab Chakrabarty `[一作]` (Dartmouth), Hang Liao `[通讯]` (Palo Alto Networks)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了CUT查询在揭示未知超图结构中的能力，提出了一种零错误的随机算法，能够在O(n)期望查询内识别加权超图的连通分量。

**💡 创新点**

创新点在于引入了“独立家族”的概念，克服了超图重建的障碍，并展示了超边的奇偶性对学习的影响。

**🔧 技术方法**

使用了零错误随机算法和Möbius变换等技术，结合辅助加权图的连通性技术。

**📊 数据集**

使用了加权超图作为数据集，特别关注偶数超边和线性超图的情况。

**📈 对比分析**

与传统的O(n^2)查询算法相比，提出的算法在偶数超图中实现了O(∑_e∈ 2^|e|log n)的查询复杂度，并在线性超图中达到了O(kn^1.5)的复杂度，显著提高了性能。

**⚠️ 局限性**

限制在于对于奇数超边的学习存在固有的模糊性，且在某些情况下，无法通过CUT查询单独学习超图的结构。

---

## 576. FurnitureVLA: Learning Long-Horizon Bimanual Furniture Assembly with Vision-Language-Action Model

**arXiv ID:** 2607.01212 | [PDF](https://arxiv.org/pdf/2607.01212v1)

**作者:** Chenyang Ma `[一作]` (Mitsubishi Electric Research Laboratories), Diego Romeres `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作构建了一个可扩展的模拟管道与VR遥操作系统，实现了对真实尺度双臂家具装配的系统研究。

**💡 创新点**

创新点在于提出进度增强的视觉语言动作模型（Progress-Enhanced VLA），通过细粒度子任务分解和连续进度信号实现长时程稳定执行，并系统评估感知与控制设计因素对精度的影响。

**🔧 技术方法**

主要技术包括Vision-Language-Action模型（π_0.5），流匹配（flow matching）微调，持续进度信号，动作块预测，动作视角与分辨率优化，时间集成（temporal ensembling），以及基于Kinova Gen3双臂的VR遥操作控制。

**📊 数据集**

使用的实验数据集为通过运动规划在模拟中生成的500条演示（每件家具），以及使用VR遥操作收集的100条真实世界演示，家具模型来自IKEA和3D Warehouse。

**📈 对比分析**

与零样本π_0.5及全局指令的单一微调模型相比，Progress-Enhanced VLA在模拟中对三件家具的完整装配成功率提升至平均80%，在真实IVAR椅子上从0.68提升至0.80，显著优于基线。

**⚠️ 局限性**

局限性包括：仅适用于固定基座的双臂工作空间，未解决螺丝拧紧等更高精度工具任务，并且依赖磁性连接，缺乏移动平台的适应性。

---

