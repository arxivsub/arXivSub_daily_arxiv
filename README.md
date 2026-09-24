# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-09-24 | 今日论文总数: 687

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. FLINT: Fast Lightweight Inference for Traversability

**arXiv ID:** 2609.26857 | [PDF](https://arxiv.org/pdf/2609.26857v1)

**作者:** William Bonilla `[一作]` (McGill University), Louis Petit `[通讯]` (University of Sherbrooke)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种仅用单摄像头、CPU 实时推理的轻量级越野可行性评估器 FLINT，并通过多种自监督标签生成方法与人工标注基线进行对比。

**💡 创新点**

创新点在于：①把冻结的 DINOv3 backbone 与极小的 MLP 头结合，显著减小模型体积并实现 CPU 14.7 FPS；②设计了三种自监督标签传播策略（轨迹重投影、SAM3 实例传播、STEGO 聚类传播），证明传播能显著提升性能；③在真实机器人现场闭环部署中自监督模型超越人工标注。

**🔧 技术方法**

技术包括：冻结的 DINOv3 ViT-S/16 视觉 backbone、轻量 MLP 头、逆透视映射生成 BEV costmap、Nav2 规划器、单摄像头 RGB 输入、CPU 端实时前向推理。

**📊 数据集**

数据集主要使用 TartanDrive 2.0（用于自监督标签生成）和 RELLIS‑3D、Yamaha、CAVS、Great Outdoors 三个人工标注集（用于基线与零射测试）。

**📈 对比分析**

比较方法：先在 24 条现场日志上与 WildOS、WVN 进行离线基准对比，随后在同一路线、相同规划器下对四个头（三自监督+人工）进行闭环部署，指标包括自动化率、干预次数、最长无干预距离、驱动成本百分位；结果显示最佳自监督头达 99% 自动化率，且在多项评估指标上优于人工标注。

**⚠️ 局限性**

局限性：仅在单一路线、单一日况下验证；未测试不同地形或天气条件；模型依赖冻结的 DINOv3，无法在线自适应。

---

## 2. Distilling Lexical Product Associations into Deep Transformers: An Extreme Multi-Label Approach for Natural Language E-Commerce Search

**arXiv ID:** 2609.26921 | [PDF](https://arxiv.org/pdf/2609.26921v1)

**作者:** Sunnidhya Roy `[一作]` (International Institute of Information Technology Bangalore), Samarpita Bhaumik `[通讯]` (International Institute of Information Technology Bangalore)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将电商对话式推荐任务转化为极端多标签分类，并通过无监督词向量图的伪标签蒸馏训练 DistilBERT 学生模型；

**💡 创新点**

创新点包括：① 用词向量相似度构建无监督伪标签并进行知识蒸馏；② 纠正基线对齐错误并实施严格自排除评估；③ 在十种对话查询上进行定性基准测试；④ 对极端多标签层的内存与可扩展性进行了深入分析；

**🔧 技术方法**

主要技术包括：DistilBERT transformer、伪标签知识蒸馏、TF‑IDF 余弦相似度 KNN、二进制交叉熵损失、两塔向量检索路线（HNSW/FAISS）等；

**📊 数据集**

使用 Amazon Reviews '23 基准数据集，挑选 27 个平衡类别下的 54,000 个商品，生成 53,923 个类别标签；

**📈 对比分析**

与同一词向量教师（TF‑IDF）在严格自排除下对比，指标为 P@1 93.15% vs 98.10%，NDCG@10 0.8845 vs 0.9419；在十种对话查询上，Transformer 在大多数场景下能获得高置信度且相关性优于词典检索；

**⚠️ 局限性**

局限性包括：① 分类头参数随商品数量呈线性增长，超过百万级商品时参数量不可行；② 动态商品更新需重新训练或重置输出维度；③ 推理时对所有类别做 sigmoid 激活导致延迟随 C 线性增长，需迁移至两塔向量搜索才能满足工业规模性能需求。

---

## 3. Kubernetes Misconfigurations in the Wild: Taxonomy, Evolution, and Automated Repair with Large Language Models

**arXiv ID:** 2609.27030 | [PDF](https://arxiv.org/pdf/2609.27030v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 4. Anatomy-Aware Synthesis of Post-Contrast Breast MRI from Pre-Contrast Images

**arXiv ID:** 2609.27015 | [PDF](https://arxiv.org/pdf/2609.27015v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 5. The Like Trap: Multi-Stage Poisoning against Agents in Similarity-based Recommendation Systems

**arXiv ID:** 2609.27155 | [PDF](https://arxiv.org/pdf/2609.27155v1)

**作者:** Yue Xing `[一作]` (Michigan State University), Zitao Li `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a2602d71-93ab-4bad-974b-672788df8193` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在以LLM代理为用户的相似度推荐系统中，通过多阶段毒化帖子链诱导代理点赞并形成正反馈循环，从而污染其推荐结果。

**💡 创新点**

首次结合理论证明与实证，提出多阶段毒化链的设计原则，并给出基于点赞得分机制的攻击算法，展示如何利用推荐系统的正反馈将低相似度毒化内容推入推荐流。

**🔧 技术方法**

使用大语言模型生成毒化帖子、相似度+点赞得分的推荐算法、文本相似度控制与偏置增强（位置、简洁、流行度等），并对攻击过程进行数值理论分析。

**📊 数据集**

采集了OASIS用户档案及其帖子、AI推文语料，构建2000条帖子样本，并随机选取50个用户进行实验。

**📈 对比分析**

对六种LLM模型（GPT-5.4、GPT-5.4-mini、GPT-5.4-nano、DeepSeek v4 pro、Llama 3.1 8b、Qwen 3.5 9b）进行评估，指标为第5阶段帖子占比的首次出现轮数、候选数与点赞数；实验显示攻击在约180-250轮内使第5阶段帖子占比≥5/10，整体攻击成功率高。

**⚠️ 局限性**

仅针对相似度+点赞得分机制，未涵盖图谱或变压器序列推荐器；假设攻击者能一次性注入所有毒化帖子且不根据反馈动态调整，对更复杂或多模态推荐系统的适用性未知。

---

## 6. Feed the Panel Dimensions, Not Verdicts: Rubric-Decomposed Fusion of Vision-Language Aesthetic Judges

**arXiv ID:** 2609.27110 | [PDF](https://arxiv.org/pdf/2609.27110v1)

**作者:** Amit Jadhav `[一作]` (Purdue University Fort Wayne), Beomjin Kim `[通讯]` (Purdue University Fort Wayne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对VLM进行美学评分，比较面板与单一模型的效果

**💡 创新点**

提出用冻结人类写的五维度 rubric 分解并融合，提升面板性能

**🔧 技术方法**

使用梯度提升回归器（Gradient Boosting）进行特征融合，评估方式为 Spearman/Kendall 相关系数

**📊 数据集**

实验基于 EVA-500 与 PARA-500 两个含人类评分的图像数据集

**📈 对比分析**

通过配对自助法与多重检验比较，EVA 上维度融合面板相对最佳单模型提升 0.07–0.10 Spearman，PARA 上仅相当或略低

**⚠️ 局限性**

局限性包括样本仅 500 张、仅对照片有效、需先判定模型离噪声上限的余量，且两维度未得到验证，面板收益受数据集差异影响

---

## 7. Fine Wrist Control as a Marker of Surgical Teleoperation Expertise

**arXiv ID:** 2609.27160 | [PDF](https://arxiv.org/pdf/2609.27160v1)

**作者:** Mary Kate Gale `[一作]` (Stanford University), Allison Okamura `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

对比专家与新手在机器人手术遥控中完成非医学圆柱-杆转移任务时的上肢关节运动，探讨经验对运动控制的影响。

**💡 创新点**

发现专家在复杂转向任务中显著改善腕部运动控制，而肩膀、肘部运动相似，表明腕部稳定性是专家差异的关键。

**🔧 技术方法**

使用dVRK手术系统、四个6-DOF IMU+视觉运动捕捉，结合OpenSim逆运动学计算关节角度。

**📊 数据集**

收集了27名无手术经验者和9名超过250小时经验的机器人外科医生的实验数据。

**📈 对比分析**

通过Mann-Whitney U检验比较ROM和任务完成时间，专家在腕部ROM和完成速度上显著优于新手。

**⚠️ 局限性**

任务受限为单手操作且未允许换手或摄像头移动，限制了对自然专家动作的真实性评估。

---

## 8. Laser-Tracker-Assisted Camera-to-Robot Calibration for Mobile Robots

**arXiv ID:** 2609.27006 | [PDF](https://arxiv.org/pdf/2609.27006v1)

**作者:** Jan A. Rudolph `[一作]` (Karlsruhe Institute of Technology), Markus Ulrich `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 2669 | [OpenAlex ID](https://openalex.org/A5031522635)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于激光跟踪器与相机的手眼校准方法，用链式校准目标实现移动机器人相机姿态在机器人坐标系中的精确标定。

**💡 创新点**

创新点在于突破了仅支持垂直相机的先前方法，采用多段校准目标链将激光跟踪器参考板与机器人观测板连接，从而支持任意相机朝向并显著提升精度。

**🔧 技术方法**

使用技术包括激光跟踪器三维测量、相机二维标定、辅助相机构建目标链、三角测量求取坐标变换、图像点投影误差分析以及机器人姿态融合。

**📊 数据集**

数据集主要来自实验室设置的两组场景（Scene A、Scene B）：使用RITA移动机器人、Leica AT901跟踪器、IDS相机、配套的反射窝与校准板。

**📈 对比分析**

通过对比精度（RMS/最大误差）和绝对准确度（点到视线距离和像素重投影误差）评估，Scene A得到的校准误差均低于1 cm；Scene B由于链条更长、距离更远，误差明显增大，说明链长及目标布置是性能瓶颈。

**⚠️ 局限性**

主要限制包括链条长度导致的误差累积、机器人姿态多样性不足、相机/跟踪器测量噪声与装配不稳等，需进一步分解各项误差源以进一步优化方法。

---

## 9. Validation and Simulation Catch Different Errors: Four Levels of Evaluation for LLM-Generated Circuits

**arXiv ID:** 2609.26830 | [PDF](https://arxiv.org/pdf/2609.26830v1)

**作者:** Ali Hedayati Pirouzan `[一作]` `[通讯]`, Ali Hedayati Pirouzan

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了 LLM 生成电路时的四个独立评估层次（方案有效性、拓扑有效性、后端可执行性、组件集一致性），并在 150 个三语电路基准上对同一生成结果进行量化比较，发现模拟与结构验证存在互相不一致的错误。

**💡 创新点**

创新点在于：①提出四层评估框架，明确拆分结构与功能评估；②量化并展示了结构验证与 SPICE 模拟之间的误差分布；③发现并实验“无声失败”类（结构错误但模拟无报错）并以最小电压分压器演示其影响；④通过配对消融实验精准测定模型修复、地面归一化等环节对各评估层次的贡献；⑤与直接生成网表的 baseline 进行对比，说明中间表示和检查的重要性。

**🔧 技术方法**

使用了：Typed Circuit Interchange Representation (CIR) 作为中间表示；CIR 结构验证器（包含地面归一化、浮点节点检测等）；模型修复循环（根据验证器反馈重提示）；PySpice 作为后端 SPICE 模拟器；固件仿真器用于 MCU 场景；GPT‑4/类似 LLM 进行电路生成；配对消融实验及统计分析。

**📊 数据集**

数据集：150 个电路，20 类别，难度分布为 37 易 / 65 中 / 48 难；覆盖 34 种 CIR 组件类型；语言分布为 78 波斯语、51 英语、21 西班牙语；21 个并行三语组（同一电路的三种语言描述）用于语言效应对比。

**📈 对比分析**

比较方法：设置两条主路线（pipeline vs 直接网表 baseline）以及配对消融子样本（45 个电路）；对同一模型样本在不同 pipeline 阶段进行评估；统计 4 层的通过率、误差分布、语言影响。性能结果：完整 pipeline 在 150 例上实现 88.7% 的后端可执行率、82.7% 的拓扑有效率、74% 的组件集一致率；baseline 仅 47.3% 可执行率；消融实验将拓扑有效率从 40% 提升至 84.4%，可执行率从 77.8% 提升至 91.1%（+7/-1）。语言方面无显著差异。

**⚠️ 局限性**

局限性：①评估仅覆盖结构与后端可执行性，未涵盖功能正确性；②结构验证器仍有缺陷（如浮点节点误判、短路检测不完整）；③实验基于单一 LLM 与实现，可能不具备泛化性；④baseline 归因存在不确定性（部分失败可能与 PySpice 处理相关）；⑤数据集规模有限，且多为人工构造，未必代表真实工业需求。

---

## 10. The Illinois Social Attitudes Aggregate Corpus (ISAAC): An Open Tool and Reproducible Pipeline for Analyzing Social Group Discourse at Scale

**arXiv ID:** 2609.27059 | [PDF](https://arxiv.org/pdf/2609.27059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 11. Scaling of Capability and Efficiency at Inference Time in Large Reasoning Models

**arXiv ID:** 2609.27166 | [PDF](https://arxiv.org/pdf/2609.27166v1)

**作者:** Moritz Laber `[一作]` (Northeastern University), Tina Eliassi-Rad `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在大型语言模型中，研究了推理能力（正确解答数量）与推理效率（输出长度）随模型规模的变化规律，并使用层级贝叶斯模型对四类可调难度的算术与算法推理任务进行定量评估。

**💡 创新点**

创新点在于：①首次同时从能力和效率两维度量推理模型的规模效应；②采用层级贝叶斯框架结合指数衰减与幂律增长模型，精细估计能力衰减尺度与效率参数随参数量的隐式尺度律；③揭示能力虽随规模增长但呈次线性收益，效率几乎不随规模提升，质疑单纯规模扩展的有效性。

**🔧 技术方法**

主要技术包括：链式思考（CoT）推理、对模型输出的答案与长度提取、层级贝叶斯模型（可变与固定渐近模型、前因子指数模型等）、马尔科夫链蒙特卡洛（HMC）采样、留一交叉验证的期望对数似然（ELPD）评估。

**📊 数据集**

使用自生成的四个推理基准：求和（Addition）、括号嵌套（Brackets）、二进制位偶奇性（Parity）和索引查询（Index）。每个任务在多种实例大小（n=2,10,50,…,500）上随机生成 100 个实例，并在不同模型规模（1.5B–70B 参数）以及采样温度 {0.4,0.6,0.8} 下进行评测。

**📈 对比分析**

通过比较不同模型规模下的正确解答比例与平均输出长度，利用贝叶斯层级模型拟合指数衰减尺度 ν 和输出长度的幂律参数。结果显示：能力的衰减尺度随模型规模呈 β≈0.69 次线性增长，表明规模扩展收益递减；效率的前因子 A 与指数 α 对规模的敏感度极小（β≈0），说明较大模型并未在正确答案的 token 长度上实现显著提升。

**⚠️ 局限性**

局限性包括：①仅测试四个人工生成的算术/算法任务，未覆盖更复杂或真实世界推理场景；②只考虑了单一模型族（Qwen/Llama 的 distillation 版本）和有限的温度设置；③层级贝叶斯模型假设了特定的指数/幂律形式，可能忽略其他潜在的非线性关系；④未对模型内部机制（如注意力宽度、深度等）与规模效应进行分解，难以揭示能效提升的根本原因。

---

## 12. Harness as a Language: A Minimalist Agent Framework With Maximal Expressivity

**arXiv ID:** 2609.26891 | [PDF](https://arxiv.org/pdf/2609.26891v1)

**作者:** Zhening Li `[一作]` (MIT CSAIL), Armando Solar-Lezama `[通讯]` (MIT CSAIL)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个基于LLM的编程语言原语 |invoke| 的 agent 框架 JAZ，展示仅凭最小的 agent loop（无需外部记忆或自我改进组件）即可完成需要长时记忆和持续自我改进的工作流。

**💡 创新点**

①提出 |invoke| 原语，让 LLM 在调用时生成函数实现；②将所有输入和 REPL 历史统一视为变量，赋予模型更强的可编程性；③引入动态作用域和钩子系统，支持观测、预算控制、验证等；④通过实验证明最小化的 agent loop 能替代传统的外部记忆与自我改进框架。

**🔧 技术方法**

使用代码模式（CodeAct/递归子代理）+ LLM 生成代码 + Python REPL + 动态作用域 + 钩子系统 + LLM 成本与预算控制。

**📊 数据集**

StuLife（长时记忆任务）和 AppWorld（持续自我改进任务）数据集。

**📈 对比分析**

在仅提示（无外部工具）设置下，与 Letta（MemGPT）、ACE、CodeAct/CodeAct+subagents 对比：在 StuLife recall‑heavy 子集，|invoke| 获得 70% 正确率，优于 Letta 62%（+8%）且成本减半；在 AppWorld，|invoke| 获得 74% 正确率，优于 ACE 70%（+4%）且成本更低；整体性能均优于传统代码模式实现。

**⚠️ 局限性**

受限于 LLM 生成质量与提示设计；递归调用可能导致内存/成本开销；缺乏对极大上下文或多任务并行的完整支持；安全与调试机制仍需进一步完善。

---

## 13. Tie Handling Is Part of the Evaluation Protocol: An Order-Invariance Audit for Tie-Heavy Recommender Scores

**arXiv ID:** 2609.26977 | [PDF](https://arxiv.org/pdf/2609.26977v1)

**作者:** Chengkun Guo `[一作]` (Independent Researcher), Yingrui Li `[通讯]` (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对离线推荐系统评估中分数完全相同（tie）时的排名处理进行系统审计，定义并检验行顺序不变性（row-order invariance），评估其对 NDCG@10 与 Hit@10 的影响。

**💡 创新点**

提出行顺序不变性概念并推导在随机打乱相同分数块时 Hit@k 与 NDCG@k 的期望公式；系统比较四种打破 tie 的策略，并给出完整的评估报告清单。

**🔧 技术方法**

使用稳定排序、基于哈希的确定性打破、重复随机打破以及均值期望分析，结合理论推导与实验验证。

**📊 数据集**

Amazon Beauty & Personal Care（30,000 条评估行）和 MovieLens 25M + Tag Genome（10,000 用户）两个公开数据集。

**📈 对比分析**

在候选集、分数和标签保持不变、仅改变相同分数的打破规则的情况下，发现输入顺序打破导致 NDCG@10 下降 0.55–0.68，Hit@10 同样显著下降；确定性哈希几乎等于随机期望值；重复随机打破的平均值与理论一致。

**⚠️ 局限性**

实验仅覆盖单一相关物品和 30 个负样本的采样评估，连续评分器可能产生很少的 tie；未针对全目录评估进行实验；行级数据仅在 Amazon 上保留；结果可能不适用于其他推荐算法或更大规模系统。

---

## 14. Spec2COBOLRot: An Agentic-AI Degradation Loop for Realistic COBOL Corpus Generation

**arXiv ID:** 2609.26835 | [PDF](https://arxiv.org/pdf/2609.26835v1)

**作者:** Jean-Baptiste Espinasse `[一作]` (Sopra Steria), Mathieu Acher `[通讯]` (INSA, University of Rennes, IRISA, Inria)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个基于 agentic AI 的管道，先根据自然语言规格生成可执行的 COBOL 程序，再通过迭代降级注入真实生产代码模式，使生成程序具备真实遗留系统的结构复杂度。

**💡 创新点**

将生成与降级分离，使用资产库中真实遗留模式指导降级，并通过回归得到基于 SLOC 的目标复杂度区间，实现可控、可解释的复杂度逼近；同时提供公开的评估数据和专家评审。

**🔧 技术方法**

GPT‑5.4 LLM 作为生成与降级代理，GnuCOBOL 编译执行循环，静态分析提取结构指标，线性回归与 Spearman 相关性计算目标区间，Harbor 编排管道。

**📊 数据集**

由 Sopra Steria 提供的 14 个真实遗留 COBOL 程序（主要是工资单批处理），以及自建的 26 个资产库（8 idiom、15 smell、3 mismatch）；同时使用三条业务规范（TRNRECBT、PAYCOMBT、CLMPREMT）。

**📈 对比分析**

通过功能等价测试、SLOC/NPAR/CC 等指标对比生成前后与目标区间；结果显示生成阶段所有程序通过编译执行，降级后在大多数指标上进入目标区间，功能保持率约 55%（9/16），迭代次数限制 k=5，生成时间可接受。

**⚠️ 局限性**

功能保持率不高，降级过度依赖模式复制导致真实感不足；数据集局限于单一 HR 领域，无法完全代表其他行业；LLM 非确定性导致可复现性差。

---

## 15. TwinCheck: Evidence-Grounded Negative-Twin Verification for Stateful Tool Agents

**arXiv ID:** 2609.26911 | [PDF](https://arxiv.org/pdf/2609.26911v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 16. Planning Trajectories that Bounce: Reflection Classes for Collision-Tolerant Robots

**arXiv ID:** 2609.27145 | [PDF](https://arxiv.org/pdf/2609.27145v1)

**作者:** Subhadeep Koley `[一作]` (Lehigh University), David Saldaña `[通讯]` (Lehigh University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文开发了一种利用墙面反弹的轨迹规划方法，构造了反射签名（r-signature）并在状态格点中枚举碰撞路径类，随后使用接触感知的MPPI控制器执行，验证在有限转向能力的机器人上有意碰撞可以降低执行时间和控制量。

**💡 创新点**

创新点在于首次引入有序反射签名来表征碰撞路径的类，构建反射增强状态格点搜索，以此枚举所有可行的碰撞路径，并通过动态可行控制器验证其在实际执行中的优势。

**🔧 技术方法**

使用的技术包括基于SE(2)的状态格点规划、r-augmented lattice、反射模型与r-signature枚举、A*搜索求最优路径，以及带接触感知的MPPI控制器与contouring cost实现轨迹跟踪。

**📊 数据集**

实验采用四个人工生成的二维网格地图（20×20、50×50、100×100），并在每张地图上枚举20条最优反射类，未使用公开数据集。

**📈 对比分析**

通过将反射路径与无碰撞路径在相同机器人模型和控制器下进行对比，评估完成率、控制努力、到达时间和轨迹误差，结果显示反射路径成功率提升至80%，控制努力降低约16.6%，到达时间缩短约5.3%，误差平均下降27%。

**⚠️ 局限性**

局限性包括仅在理想的弹性无摩擦、无线性阻力的仿真环境下验证，未考虑摩擦、非弹性冲击或接触不确定性，且实验仅覆盖单一地图与起终点，需要在更多环境和真实平台上进一步验证。

---

## 17. Scalable Subgraph Sampling via Resistance Curvature

**arXiv ID:** 2609.27209 | [PDF](https://arxiv.org/pdf/2609.27209v1)

**作者:** Chaoqun Fei `[一作]`, Yangyang Li `[通讯]` (South China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于 Johnson‑Lindenstrauss 投影和多 GPU 批量共轭梯度的有效电阻曲率（ERC-LG）近似方法，并利用该曲率指导图神经网络的子图采样，从而降低大规模图的训练成本。

**💡 创新点**

创新点在于：①将 JL 投影与正则化多 GPU 批量 CG 相结合，避免了显式求解拉普拉斯伪逆；②用近似曲率直接构造节点与边采样概率，实现对边几何角色的敏感采样；③在实际大规模图数据集上验证了该方法的数值精度、计算效率及对下游节点分类性能的提升。

**🔧 技术方法**

使用的技术包括：Johnson‑Lindenstrauss 投影、正则化多 GPU 批量共轭梯度求解、有效电阻曲率（ERC）近似、基于曲率的节点/边采样策略，以及 PyTorch+CUDA 的多 GPU 并行实现。

**📊 数据集**

实验所用数据集共七个，分别是 PubMed、Amazon Photo、Coauthor CS、Flickr、ogbn‑arxiv、Reddit 与 ogbn‑products，覆盖中型到大型无向图。

**📈 对比分析**

与 ClusterGCN、GraphSAGE、FastGCN、GraphSAINT、REGNN、LC‑sub、ORG‑sub 等基线方法相比，ERC‑LG 基础的采样方案在六个数据集上获得最高平均分类准确率；同时在曲率近似上与伪逆法误差低于 0.01，且运行时比纯 CG 方式快 9‑35 倍，GPU 内存占用亦显著下降。

**⚠️ 局限性**

局限性包括：①近似精度依赖投影维度 K 与正则化参数 ε，需经验调参；②对极大规模图仍需多 GPU，单 GPU 内存仍受限；③在某些数据集（如 ogbn‑arxiv、ogbn‑products）上替代曲率后效果不一定提升，说明曲率替换的收益与图结构相关。

---

## 18. Data-driven discrete-time deep recurrent neural network-based modeling for dissipative systems

**arXiv ID:** 2609.27186 | [PDF](https://arxiv.org/pdf/2609.27186v1)

**作者:** Tuan Luong `[一作]` (Sungkyunkwan University), Hyungpil Moon `[通讯]` (Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种离散时间耗散递归神经网络（DissipNet），能够在保持系统耗散性和稳定性的前提下学习非线性动力学模型。

**💡 创新点**

通过结构化权重约束和线性矩阵不等式（LMI）设计，首次实现了在网络层面显式保证耗散性，并采用无约束优化训练，避免了传统 PINN 需要显式物理方程且不保证耗散性的缺陷。

**🔧 技术方法**

使用了结构化递归神经网络、LMI 约束、Lyapunov 稳定性理论和基于无约束优化的训练算法；对比了传统 RNN、物理信息网络（PINN）以及自己的模型。

**📊 数据集**

实验数据来自两个模拟机器人案例：质量‑弹簧‑阻尼系统（MSD）和双自由度平面机械臂。数据覆盖训练、测试以及添加高斯噪声的不同场景。

**📈 对比分析**

在相同网络架构、相同训练设置下与 Naive RNN 与 PINN 进行比较。结果显示 DissipNet 的预测误差显著低于 RNN，并且在噪声条件下更稳健；PINN 在耗散性方面表现不佳。误差降低幅度超过 80%，标准差亦显著减小。

**⚠️ 局限性**

局限性包括：训练时间较长；当前仅支持激活函数满足 α=0 的情形；仅针对离散时间耗散系统，尚未验证更复杂系统的可扩展性；未来工作需进一步提升训练效率并探索更广泛的应用场景。

---

## 19. Silent Failures in Agent-Tool Interaction: An Audit of ToolUniverse

**arXiv ID:** 2609.26836 | [PDF](https://arxiv.org/pdf/2609.26836v1)

**作者:** Shreya Gopalan `[一作]` (AI Tech Ethics), Sundaraparipurnan Narayanan `[通讯]` (AI Tech Ethics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究系统性地审计了生物学领域 agent‑tool 交互中的静默失败，选取 ToolUniverse 环境中的 15 主要工具，生成并手工验证了 198 个测试用例，发现 91 个真实失败并将其归入七大失败定位类别，同时提出了上下文可靠性概念；

**💡 创新点**

创新点在于首次提出“静默失败”与七层失败定位体系（L1‑L7），并系统评估 agent‑tool 交互链中的信息完整性、可追溯性与语义保真，提出了衡量交互上下文可靠性的框架；

**🔧 技术方法**

技术手段包括基于 LLM 的 Cursor 生成候选失败与自动评估、手工验证、API 与 wrapper 的多层交互测试、GraphQL 查询、以及构建的工具链测试框架；

**📊 数据集**

使用的数据集为 ToolUniverse 环境中选取的 15 个生物学工具及其 198 个具体测试案例，配合对应的 API 文档、Web UI 文档、GitHub issue 等元数据；

**📈 对比分析**

通过对 91 个已验证失败案例进行失败定位和维度分析，展示了失败在不同层级（API、wrapper、agent）及维度（完整性、排名、相关性等）的分布，表明传统任务成功评估无法捕捉静默放大现象，对下游科学结论存在潜在风险；

**⚠️ 局限性**

研究局限在于仅覆盖 ToolUniverse 的 15 个工具，依赖 LLM 生成候选失败，缺乏全面覆盖；工具/接口的演进可能导致结果随版本变化；缺乏可复现的自动化基准，对结论的普适性和可验证性存在限制。

---

## 20. FedCoT-VQA: A Federated Learning and Unlearning Framework for Chain-of-Thought Planners in VideoQA

**arXiv ID:** 2609.26814 | [PDF](https://arxiv.org/pdf/2609.26814v1)

**作者:** Rui Lu `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Pengcheng Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FedCoT-VQA框架，在分布式 VideoQA 环境下对 Chain-of-Thought (CoT) 规划器进行联邦学习与无学习（unlearning）

**💡 创新点**

通过三模块设计（PSP、SSA、RUM）实现可扩展的规划器参数分区、兼容异构客户端的聚合记录以及高效的删除客户端影响而不需全量重训练

**🔧 技术方法**

参数高效微调（LoRA/PEFT）、联邦聚合、记忆式回放+选择性残差修正、Trace-aware 贡献日志

**📊 数据集**

TVQA+、NExT-QA、STAR 三大公共 VideoQA 基准

**📈 对比分析**

与 Centralized、FedAvg、FeDeRA、SIFU、FedAU 等基线对比；FedCoT-VQA 在联邦训练时可保持接近最佳答案/定位精度，删除后保留客户端准确率并显著降低删除客户端影响（CF Gap、Trace Disc. 等指标均优于其他无学习方案）

**⚠️ 局限性**

主要局限在单客户端删除、单一规划器架构、对更大模型与更复杂删除场景（多客户端/连续删除）尚未全面验证

---

## 21. Enhancing Small Language Models for Power Outage Report Generation via Minimum Risk Training

**arXiv ID:** 2609.27197 | [PDF](https://arxiv.org/pdf/2609.27197v1)

**作者:** Hung Phan `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究应用最小风险训练（MRT）方法于电力故障报告生成，将异构报告转化为符合CIM IEC 61968-3标准的XML格式。

**💡 创新点**

创新点在于将MRT应用于特定领域的结构化生成任务，显著提高了小型语言模型在电力故障报告生成中的准确性。

**🔧 技术方法**

使用了最小风险训练（MRT）技术，结合了参数高效的微调方法。

**📊 数据集**

使用了1000个电力故障报告实体的数据集，每个数据点包含一个非标准故障报告及其对应的标准化XML报告。

**📈 对比分析**

与基线模型相比，应用MRT后，整体准确率从16.20%提高到68.95%，其中XML结构正确性从3.56%提高到87.17%。

**⚠️ 局限性**

限制在于MRT的有效性可能依赖于特定任务的定义和数据集的质量，未来需要探索不同领域的序列级风险定义。

---

## 22. The Drift Contract: Spectral Updates for Depth-Robust Local Learning

**arXiv ID:** 2609.26811 | [PDF](https://arxiv.org/pdf/2609.26811v1)

**作者:** Fabien Polly `[一作]` `[通讯]` (Independent researcher), Fabien Polly (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文研究了局部学习与谱更新几何的交集，提出了一种新的局部更新方法，解决了深度增加时准确性下降和超参数脆弱性的问题。

**💡 创新点**

创新点在于首次系统性地研究了局部学习中的谱更新几何，并提出了一个漂移合同的学习率公式，使每层的权重变化受到输入条件的约束。

**🔧 技术方法**

使用了谱更新几何和漂移合同的技术，结合了局部学习的结构并行性。

**📊 数据集**

使用了CIFAR-10数据集，包含20000个训练样本和5000个测试样本，图像被展平为3072个特征。

**📈 对比分析**

与局部Adam方法相比，谱更新在每个宽度下的性能均优于局部Adam，尤其在深度增加时，谱更新的准确性下降幅度较小。

**⚠️ 局限性**

局限性包括仅在CIFAR规模的数据上进行的MLP分类器实验，且漂移界限是条件性的，未涵盖从上游层传播的漂移。

---

## 23. HARN: Hierarchical Associative Resonance Network for Event-Driven Multi-Timeframe Forecasting

**arXiv ID:** 2609.26822 | [PDF](https://arxiv.org/pdf/2609.26822v1)

**作者:** Nabeel Ahmad Saidd `[一作]` `[通讯]` (APJ Abdul Kalam Technical University), Nabeel Ahmad Saidd (APJ Abdul Kalam Technical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了HARN（Hierarchical Associative Resonance Network），一种事件驱动的多时间框架金融时间序列预测模型，能够在新的条目信息到来时只更新对应时框的表示而保持其他层的状态不变。

**💡 创新点**

创新点在于设计了完整的事件驱动更新协议，包括完成条形的指示器、持久化状态、跨层共振（all‑to‑all）与有序下行证据读取，以及基于检索残差的关联记忆写入机制，使模型能够在不同时间尺度之间高效共享信息。

**🔧 技术方法**

使用了因果多尺度编码器（膨胀卷积+衰减池化）、门控关联记忆单元、跨层共振、下行证据路径、以及在基点变化空间进行的预测与重构。

**📊 数据集**

在四个资产上进行实验：AAPL（股票）、EURUSD、USDCHF（外汇）和XAUUSD（贵金属），每个资产都有不同的时间框架层级。

**📈 对比分析**

与单一时间框架的PatchTST和TimeXer基线进行对比，HARN在构造的资产-时间框架上取得了更低的价格重构误差（MAE、RMSE、sMAPE、MASE），但比较是描述性且未隔离信息访问、目标设定和训练协议等因素。

**⚠️ 局限性**

主要局限包括：缺少持久化基线和置信区间；单次种子跑的消融实验无法给出统计显著性；未进行跨资产、不同市场周期的泛化评估；缺乏推理延迟、内存占用等系统性能测量；以及对时间戳一致性和证据系数等实现细节的依赖。

---

## 24. Bringing Chip Tapeout Into University Education

**arXiv ID:** 2609.26970 | [PDF](https://arxiv.org/pdf/2609.26970v1)

**作者:** Luca Pezzarossa `[一作]` (Technical University of Denmark), Michael Pehl `[通讯]` (Technical University of Munich)

**通讯引用:** 528 | [OpenAlex ID](https://openalex.org/A5039512067)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过在五所欧洲大学实施联邦模型，构建了一个可重复、可扩展的芯片设计与制造教育体系，利用共享的Didactic SoC平台和多项目晶圆（MPW）流程，使学生能够从规格到硅片的全流程学习。

**💡 创新点**

创新点包括：①联邦模式通过对学习成果的统一对齐而非课程一致化，适配各校不同教学安排；②提供可重用的SoC平台和标准化接口，降低集成复杂度；③整合商用与开源工具链，兼顾产业实践与低成本教育；④明确集成与制版协调角色，保证跨校协作顺利进行。

**🔧 技术方法**

主要技术手段包括：SystemVerilog/Verilog HDL、IP‑XACT 与 Kactus2 进行接口定义；Git、Bender、Make、Tcl、Python、Shell 进行版本控制与自动化构建；开源实现如 Yosys、LibreLane、Magic、KLayout、OpenROAD、Verilator、PyUVM；商用工具如 Siemens EDA Questa、SkyWater130nm PDK、GF22nm、X‑FAB180nm 等；软硬件集成采用 Ibex RISC‑V 核、Caravel harness、JTAG、UART、SPI 等。

**📊 数据集**

本文未使用传统意义上的数据集，而是基于学生设计的 IP 子系统、标准单元库、内存宏以及对应的测试用例和验证脚本来构建与评估芯片。

**📈 对比分析**

通过在不同工艺节点（GF22nm商用、SkyWater130nm开源、X‑FAB180nm老版）完成多轮 MPW 制造，并在晶圆级通过 DRC/LVS、时序检查以及电气验证（功耗、时钟、JTAG 软件测试），所有实验均达到设计交付标准，证明了可行性和可靠性。

**⚠️ 局限性**

局限性主要包括：①跨校课程与学期安排不一致，需额外协调；②工具许可与 PDK 访问限制对商用流程构成瓶颈；③需要持续维护平台仓库、接口文档与 CI 体系；④后硅验证与 PCB 设计需提前规划，流程仍不够成熟；⑤学生对整体流程的初期复杂度认知高，需要进一步简化入门路径。

---

## 25. Marginally Correct Tool Caches Can Reverse Group-Normalized Policy Updates

**arXiv ID:** 2609.26866 | [PDF](https://arxiv.org/pdf/2609.26866v1)

**作者:** Shivam Gupta `[一作]` `[通讯]` (Independent Researcher), Shivam Gupta (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在强化学习中使用工具缓存时，共享执行会在保持每个回合奖励分布不变的前提下，导致基于群组归一化的梯度方向被逆转的现象，并给出了理论证明和数值验证。

**💡 创新点**

首次揭示了即使每个回合的条件奖励分布保持一致，工具结果共享仍能破坏期望归一化更新的方向，并推导出精确的有限组表达式、极限行为以及中心化估计器的恢复效果。

**🔧 技术方法**

采用离散两动作Bernoulli模型，推导共享与独立执行下的更新表达式，进行全配置穷举计算（540种配置、3,240次估计器评估），用Python实现离散概率求和，辅以TVCache实现审计来验证机制。

**📊 数据集**

使用离散参数网格（p∈{0.1,0.5,0.9}，c∈{0.1,0.3,0.5,0.7,0.9}，q∈{0,0.2,0.4,0.6,0.8,1}，G∈{2,4,8,16,32,64}）产生的540种配置进行穷举，无真实数据集。

**📈 对比分析**

通过对比共享与独立执行在相同配置下的归一化更新（U_0）与中心化更新（V），发现共享归一化在108/540配置下方向与期望梯度相反，而中心化估计器保持正确方向；数值验证显示误差在10^-15量级。

**⚠️ 局限性**

局限性在于仅考虑单步单动作的简化模型，未涉及多步、KL裁剪、适应优化器等实际训练细节，也未评估模型性能、训练时间或工具调用节省；因此结论无法直接推广至完整语言模型训练任务。

---

## 26. LayerCheck: Adaptive Layer-wise Checkpointing for Large Language Model Post-training

**arXiv ID:** 2609.27193 | [PDF](https://arxiv.org/pdf/2609.27193v1)

**作者:** Minqiu Sun `[一作]` (University of Delaware), Dong Dai `[通讯]` (University of Delaware)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种层级自适应检查点框架，仅在层更新超过阈值时持久化权重，减少I/O突发；

**💡 创新点**

创新点在于利用LLM训练中各层更新不均衡的特性，按层选择性检查点并在恢复时拼接混合时间戳模型，从而在保持收敛性（<0.54%损失偏差）的同时显著降低存储和时间开销；

**🔧 技术方法**

采用阈值驱动的层级检查点策略、混合时间戳重建技术、Adam优化器的时延边界分析，以及对比实验中的I/O与训练时间测量；

**📊 数据集**

在多种开源LLM（如GPT、LLaMA等）与对应公开数据集上进行实验；

**📈 对比分析**

与现有最先进检查点系统对比，实验显示检查点总大小降低22.6倍，整体训练时间缩短1.31倍，恢复后模型保持原有收敛轨迹和精度；

**⚠️ 局限性**

局限性包括需要针对不同模型调参阈值、可能对极端梯度波动的层产生恢复误差，以及实现复杂度和对多节点训练的适配性待进一步验证。

---

## 27. When Clients Are Orchestrated: Strategic Gradient Manipulation to Defeat Federated Learning Servers with Efficient Defense

**arXiv ID:** 2609.27124 | [PDF](https://arxiv.org/pdf/2609.27124v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 28. On Preference Coverage Collapse from Hindsight Relabeling in Multi-Objective Reinforcement Learning

**arXiv ID:** 2609.26918 | [PDF](https://arxiv.org/pdf/2609.26918v1)

**作者:** Baptiste Bonin `[一作]` (Mila), Audrey Durand `[通讯]` (Mila)

**通讯引用:** 918 | [OpenAlex ID](https://openalex.org/A5069876930)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在偏好条件多目标强化学习（MORL）中将经验回放中的目标重标记为实际达成的偏好方向（类似HER）的做法，系统评估其对样本效率的影响，并提出一种“混合重标记”方法以修复其负面效应。

**💡 创新点**

创新点包括：1）首次发现并量化了偏好覆盖坍塌（Preference Coverage Collapse, PCC）这一隐蔽失败模式；2）提出了值敏感的“被遗弃偏好质量”（Abandoned Preference Mass, APM）指标来诊断覆盖损失；3）改进了HER的偏好归一化操作，消除对成本目标的无效裁剪；4）设计了单参数混合重标记机制，在不同算法和环境下以统一系数即可恢复或提升性能。

**🔧 技术方法**

采用的技术包括：偏好条件的离线actor‑critic算法（CAPQL、GPI‑PD、MO‑TD3、CAPQL‑uniform），HER式偏好重标记，优先经验回放（PER），覆盖度指标C和APM，随机种子实验、Cohen’s d效应量及等价性检验。

**📊 数据集**

使用的数据集为MO‑Gymnasium套件中的MO‑MuJoCo v5任务，共8个连续控制任务（各2或3个目标），共36种算法-环境组合进行实验。

**📈 对比分析**

与未使用重标记的基线对比，评估指标为期望效用（EUM）和超体积（HV）。结果显示：在36个设置中，HER式重标记在19个中产生显著负面影响（d≤-0.5），在1个中略有正面影响，其余16个无显著差异。混合重标记（λ=0.25）在所有受损设置中恢复了87–117%的性能，保持了唯一受益设置的提升，并对中性设置几乎无影响。

**⚠️ 局限性**

局限性包括：实验仅覆盖MO‑MuJoCo连续动作任务，未检验离散动作、多目标数目>3、基于策略的多策略或模型预测算法；混合重标记的λ固定可能并非最优，未探索自适应策略；sign‑robust归一化仅解决了clip导致的成本目标问题，其他归一化缺陷仍待研究。

---

## 29. Learning Risk Scores Robust to Unobserved Confounders

**arXiv ID:** 2609.27144 | [PDF](https://arxiv.org/pdf/2609.27144v1)

**作者:** Ryan Edmonds `[一作]`, Phebe Vayanos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在存在未观测混杂的情况下，从历史观测数据中学习鲁棒风险评分的方法。

**💡 创新点**

创新点在于将倾向评分的不确定性建模为可感知的“置信集”，并通过最大-最小优化结合边际敏感性模型与Wasserstein分布鲁棒性，得到能够抵御未观测混杂的风险评分。

**🔧 技术方法**

采用的技术包括边际敏感性模型（MSM）、Wasserstein分布鲁棒优化、指数锥规划（exponential cone program）以及传统IPW、DR等倾向评分方法。

**📊 数据集**

实验使用UCI机器学习仓库中的15个二分类数据集，经过半合成处理生成治疗分配和未观测混杂。

**📈 对比分析**

与传统方法（IPW、直接回归、Doubly Robust、完整信息基准和鲁棒基准）对比，鲁棒方法在log‑likelihood提升、校准误差与排名相关性方面均优于所有基准，特别是在混杂程度增大时提升幅度最大，校准误差最高可降低29.2%。

**⚠️ 局限性**

局限性包括：需要先验估计混杂程度（Γ），对大规模数据时指数锥规划的二次约束导致计算瓶颈，且仅针对log‑likelihood目标，若需其他目标需进一步研究。

---

## 30. Locally Sparsified, Globally Near-Optimal: Matching under Independent Vertex Arrivals

**arXiv ID:** 2609.27161 | [PDF](https://arxiv.org/pdf/2609.27161v1)

**作者:** Sara Ahmadian `[一作]` (Google Research), Mohammad Roghani `[通讯]` (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在已知独立顶点到达分布的随机二分图匹配问题中，研究了局部菜单（每个请求保留有限条边）对最终最大匹配值的影响，并证明只需菜单大小仅与目标误差 ε 相关即可近乎保留全图期望最大匹配量。

**💡 创新点**

证明了无论实例规模或邻域大小如何，都存在一个仅随 ε 变化的全局上界 k_ε，使得任意实例都能通过简单的加权采样或蒙特卡洛模拟得到菜单，从而实现 (1−ε) 近似，填补了此前仅在“扩散”条件下已知的结果。

**🔧 技术方法**

核心技术包括局部计算算法（Local Computation Algorithms）与可控的 in–out 复杂度、加权匹配与 Brouwer 固定点法、方差与协方差控制的随机采样、以及利用 Horvitz–Thompson 权重实现轻边的期望恢复；整体构造实现了“关键见证”与“轻边完成”的两阶段证书。

**📊 数据集**

在实验中使用了结构化的 “hub” 与 “expanded” 实例族，并通过对每个实例采样获得的边出现概率估计来构造 VarOpt_k 菜单，实验规模随 k 增大而递增。

**📈 对比分析**

与均匀菜单和 top‑k 菜单相比，VarOpt_k 菜单在 k=32 时的平均损失低于 1.3%，k=64 时低于 0.7%；实验显示即使在中等规模菜单下，损失也远低于理论下界，并且相对于无分布信息策略有显著提升。

**⚠️ 局限性**

主要限制在于菜单大小的理论上界 k_ε 指数级增长，且理论与实验之间仍有显著差距；此外研究假设已知到达分布，对未知分布或随机顺序流的适用性尚未得到证明。

---

## 31. Pro-Bench: Prompt-Robust Open-Vocabulary Visual Grounding Across Real-World Heterogeneous Environments

**arXiv ID:** 2609.27076 | [PDF](https://arxiv.org/pdf/2609.27076v1)

**作者:** Linus Nwankwo `[一作]` (Montanuniversität Leoben), Elmar Rueckert `[通讯]` (Montanuniversität Leoben)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了 Pro-Bench，一个包含13k+ RGB 图像、74.5k 细粒度实例标注和 515 个多语义提示的开放词汇视觉定位基准，并在该基准上对 16 种 OOV 模型进行零样本评估。

**💡 创新点**

提出了 prompt‑robustness 协议，量化不同提示形式对定位精度的影响，并引入 PCR 与 PLC 指标评估提示一致性与空间一致性，填补了现有基准对提示敏感度与实例恢复的空白。

**🔧 技术方法**

使用多种开源 OOV 模型（Transformer 检测、单阶段实时检测、分割模型），进行零样本推理、端到端延迟测评以及自定义提示（原始、分类、模板包装），并利用 Polygon 注解支持 Box 与 Mask 评估。

**📊 数据集**

集成了多域机器人数据集（DARPA、EnvoDat、Hilti‑SLAM、GOOSE‑Ex、KITTI、MCD、TUM RGB‑D 等），覆盖地下、工业、室内、室外与城市等环境。

**📈 对比分析**

在同一 RTX 4090 GPU 上进行严格零样本评估，比较 mAP_0.5、_0.75、_0.5:0.95、延迟、Δ_max 等指标；结果显示 SAM3‑Lite 在定位精度最高但延迟最高，YOLO‑W‑S 最快但精度最低，提示敏感度与模型架构高度相关。

**⚠️ 局限性**

局限性包括仅评估单帧图像，未覆盖时序视频与交互式对话；未对 Pro-Bench 进行微调，仅使用开源预训练模型；提示集合有限，未覆盖所有语言变体。

---

## 32. EduBehaviors: Assertion-based Schemas for Auditable Coding of Educational Dialogues

**arXiv ID:** 2609.27043 | [PDF](https://arxiv.org/pdf/2609.27043v1)

**作者:** Julian Bernado `[一作]` (Stanford University), Susanna Loeb `[通讯]` (Stanford University)

**通讯引用:** 19255 | [OpenAlex ID](https://openalex.org/A5073914508)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 EduBehaviors 框架与工具包，使用可解释的行为断言与规则对教育对话进行标注，并在 TalkMoves 数据集上进行评估。

**💡 创新点**

通过将行为断言与分类规则分离，生成可审计、可修改的标注过程，并结合 LLM 生成断言与轻量级编码器训练分类器，降低成本并提升可解释性。

**🔧 技术方法**

使用 LLM（Claude、Gemini、GPT）生成与标注断言，采用概念瓶颈模型/逻辑回归（或随机森林）映射断言到标签；SetFit 编码器、Krippendorff α 评估一致性，宏 F1 与 Cohen κ 进行性能测评。

**📊 数据集**

采用 TalkMoves 课堂对话数据集（10 个会话，共 3,217 条教师发言的 Gold 标签）。

**📈 对比分析**

与直接 LLM 提示、仅用词条断言、以及 RoBERTa 微调基准对比；EduBehaviors 在最佳配置下 macro‑F1 为 0.673，Cohen κ 为 0.688，优于直接提示但低于专门微调的编码器。

**⚠️ 局限性**

断言验证缺乏人类黄金标准，跨模型一致性无法排除共享错误；工具中的预训练断言仅适用于 TalkMoves 上下文；框架高度灵活但需要用户自行决策；相较于专业微调模型，性能仍有差距。

---

## 33. Experts Rise Where LLMs Disagree: Using Cross-Model Disagreement to Target Expert Effort in LLM Codebook Revision for Large-Scale Annotation

**arXiv ID:** 2609.26926 | [PDF](https://arxiv.org/pdf/2609.26926v1)

**作者:** Zeyu He `[一作]` (Pennsylvania State University), Ting-Hao 'Kenneth' Huang `[通讯]` (Pennsylvania State University)

**通讯引用:** 1381 | [OpenAlex ID](https://openalex.org/A5083675499)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于跨模型不一致的专家反馈流程，利用多模 LLM 对大型教学会话转录文本进行标注，识别出 LLM 争议极大的实例，并通过三种专家交互方式（代码书验证、问答、理由标注）来快速修订早期代码书。最终用修订后的代码书评估 LLM 的标注效果。

**💡 创新点**

创新点在于：①将 LLM 之间的高争议实例作为聚焦点，精准定位需要专家介入的难点；②设计三种交互接口，使专家在极少的工作量下即可提供高价值反馈；③证明“理由标注”能在几小时内使 LLM 标注质量超过传统耗时数月的专家手工修订。

**🔧 技术方法**

技术包括多 LLM 并行标注、交叉模型争议检测、基于争议实例的专家交互提示（代码书修改、问题生成与回答、理由给出）、LLM 自动生成修订后代码书，以及聚合多 LLM 预测并与专家一致性标签对比的评估流程。

**📊 数据集**

使用约 6,595 条来自大型辅导会话转录的句子，作为未经专家标注的“新”数据集；此外使用专家在项目六个月内手工标注的训练集和验证集。

**📈 对比分析**

通过将多 LLM 的预测聚合后与专家一致性黄金标签计算准确率和加权 F1，结果显示：理由标注（RL）得到 64.9% 的准确率和 0.658 的 F1，超过传统专家修订的 57.8%/0.583；问答（QA）次之，准确率 60.5%/F1 0.613；代码书验证（CV）表现最差。与专家手工标注相比，RL 在几小时内就达到了更高的性能。

**⚠️ 局限性**

局限性包括：仅在一个以两名专家为主的纵向项目中验证，难以推广到其它领域；代码书主要在现有标签范围内微调，未检验添加/删除标签的情况；专家交互可能存在学习迁移与顺序效应；数据集中存在转录错误和分段误差，未系统量化其影响。

---

## 34. Asset-Class Specific Sustainability Disclosure: Lessons Learned from the EU MiCA Regulation

**arXiv ID:** 2609.26932 | [PDF](https://arxiv.org/pdf/2609.26932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 35. XLOG: A CUDA-Native Engine for Neurosymbolic Integration

**arXiv ID:** 2609.27203 | [PDF](https://arxiv.org/pdf/2609.27203v1)

**作者:** Levi Dubrovin `[一作]` (Brainyblaze Dynamics Inc), Kirill Sabitov `[通讯]` (Brainyblaze Dynamics Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个CUDA原生逻辑编程引擎xlog，支持神经感知与确定性Datalog、概率推理、认知推理等多模式，提供零拷贝互操作与已验证的知识编译。

**💡 创新点**

创新点在于将神经网络视为谓词，统一typed前端与CUDA后端；在GPU上实现了零复制的知识编译与证书验证；提出worst-case-optimal join与自适应计划；实现了端到端可微的训练与循环使用电路缓存。

**🔧 技术方法**

采用CUDA关系算子、DLPack、Arrow、CUDA CDCL求解器、决策DNNF电路、权重模型计数、WCOJ、CUDA半启发式优化、GPU驻留内存管理、PyO3 Python绑定。

**📊 数据集**

使用MNIST加法、CAVIAR视频事件、Brest AIS海事事件三大数据集进行评测。

**📈 对比分析**

与Scallop、ProbLog2、Souffle等外部引擎对比，MNIST加法准确率与Scallop相当，训练速度提升2.74×；在有偏三角计数任务上WCOJ比二进制join快约27.96×；在海事任务中权重化规则提升0.065F1。

**⚠️ 局限性**

局限性包括对有限域和受支持的逻辑片段的依赖；无法在非支持形状下回退到CPU实现；验证与缓存开销在小任务上不明显；在某些数据集上的跨验证失败；整体性能受GPU内存预算限制。

---

## 36. Ajar: Measuring Open Privilege in Agent Defenses

**arXiv ID:** 2609.26900 | [PDF](https://arxiv.org/pdf/2609.26900v1)

**作者:** Reshabh K Sharma `[一作]` (University of Washington), Zhiqiang Lin `[通讯]` (Ohio State University)

**通讯引用:** 5854 | [OpenAlex ID](https://openalex.org/A5026864098)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于生成测试的度量方法，评估语言模型代理防御在非执行路径上留给工具的开放权限；

**💡 创新点**

创新点在于引入“开放权限泄漏”这一第三维度，可与攻击成功率和任务完成度并列，揭示传统指标无法捕捉的权限松散性；

**🔧 技术方法**

采用基于任务参考解、工具模式与目标状态的规则生成候选工具调用，使用或acular为每个候选标记允许/拒绝，并根据危险级别进行加权；

**📊 数据集**

在AgentDojo v1.2.2的四个任务集（banking、slack、travel、workspace）共97个正常任务上进行评估；

**📈 对比分析**

与五种防御（Progent、CaMeL、AC4A、Claude Code Auto、perm-assistant）以及四个基线进行比较，结果显示开放权限泄漏与任务完成度并不相关；不同模型（Sonnet‑5 vs Haiku‑4.5）与测试预算、权重变化对指标影响可测量；

**⚠️ 局限性**

局限性包括依赖或acular标签的主观性、测试覆盖不完整、决策模型的运行方差以及无法直接评估攻击成功率的能力。

---

## 37. Do Audio Representations Compose Additively?

**arXiv ID:** 2609.27187 | [PDF](https://arxiv.org/pdf/2609.27187v1)

**作者:** Chenhao Xue `[一作]` (University of Oxford), Nikolaos Thomos `[通讯]` (University of Essex)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了基于CCA与留一组合（Leave‑One‑Combination‑Out）测试的两步诊断方法，用于评估预训练音频表示的加性组合能力。

**💡 创新点**

首次在不依赖文本对齐的条件下系统评估音频表示的加性组合结构，并引入重复组合留一法与标签重叠基线对比，检验模型对未见组合的推断能力。

**🔧 技术方法**

采用Canonical Correlation Analysis、Ridge正则化求解源贡献矩阵，并通过相对弗罗贝尼乌斯误差、余弦相似度、KL散度、CKA及Hits@K等多维指标进行评估。

**📊 数据集**

在FSD50K和CHiME‑Home这两个真实录音数据集上进行实验，使用其多标签音源标签进行评估。

**📈 对比分析**

与行列随机置换基线和标签重叠基线对比，CLAP在绝大多数指标（如余弦相似度、CKA、Hits@5等）上显著优于基线；Wav2Vec2/HuBERT仅在部分指标上略优或相等。

**⚠️ 局限性**

残差表明加性假设不完全成立，受到非线性声学效应、混响、编码器非线性以及未建模时间顺序等因素的影响，模型对未见组合的预测仍存在误差。

---

## 38. MINER: Multi-crop INference-time Enhancement for Rare-Object Retrieval with Frozen Dual Encoders

**arXiv ID:** 2609.27142 | [PDF](https://arxiv.org/pdf/2609.27142v1)

**作者:** Abdulmalik Alquwayfili `[一作]` (Saudi Data and Artificial Intelligence Authority), Muhammad Kamran J Khan `[通讯]` (Saudi Data and Artificial Intelligence Authority)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MINER，一种训练‑free 的推理框架，通过固定多裁剪和两侧 CSLS 来增强冻结双编码器的文本‑图像检索性能。

**💡 创新点**

创新点在于将固定覆盖性裁剪与无训练的 Hubness 校正相结合，证明覆盖率比精确定位更能恢复局部证据，特别适用于稀疏物体和拥挤场景。

**🔧 技术方法**

使用了 CLIP/SigLIP 冻结双编码器、5 个中心＋四角裁剪生成区域嵌入、融合全局与局部相似度以及两侧 CSLS 进行 Hubness 校正。

**📊 数据集**

评估数据集包括 COCO 5K、Flickr30K 标准拆分以及新构建的 ROCS（从 COCO、Flickr30K 高拥挤子集筛选并用 Qwen3‑VL 重新描述以突出稀有小物体）。

**📈 对比分析**

与基线全局检索以及多种裁剪策略、Hubness 校正方案对比，MINER 在所有骨干和拆分上均提升 Recall@1，尤其在 ROCS 上提升 5–6 分，整体增益可达近 10 分。

**⚠️ 局限性**

局限性包括需要预先编码并存储多裁剪向量，导致存储和推理成本提升约 4–5 倍；对极端遮挡或尺寸极小物体的提升有限，且仍依赖固定裁剪的覆盖率。

---

## 39. Lessons learned from deploying imaging AI with the open PACS-AI platform

**arXiv ID:** 2609.26981 | [PDF](https://arxiv.org/pdf/2609.26981v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 40. The Risk-Sensitive Schrödinger Bridge: Is Not a KL Projection

**arXiv ID:** 2609.27250 | [PDF](https://arxiv.org/pdf/2609.27250v1)

**作者:** Hamidreza Behjoo `[一作]` `[通讯]` (Chinese Academy of Medical Sciences), Hamidreza Behjoo (Chinese Academy of Medical Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文在Schrödinger桥问题中引入熵风险测度，构造了终端约束的风险敏感固定点桥，并证明其不再是KL投影；

**💡 创新点**

创新之处在于证明风险敏感桥无法用单一固定参考测度的KL投影表述，并给出θ/2的热方程缺陷；提出了新的不对称分解和通用Cole‑Hopf线性化；

**🔧 技术方法**

主要采用Girsanov变换、HJB方程、Cole‑Hopf变换、闭式Riccati求解、Gaussian案例解析、热方程不等式以及固定点迭代等技术；

**📊 数据集**

研究以理论Gaussian桥和一维Brownian示例为基准，并未使用具体实验数据集；

**📈 对比分析**

通过与经典KL投影桥的理论比较，分析了θ的第一阶偏移、价值增长以及数值示例显示在θ→1时解发散；未给出实验性能指标；

**⚠️ 局限性**

局限性在于仅给出一维/高斯解析的存在性，缺乏一般存在性证明、固定点收敛性分析、θ→1极限研究以及硬约束双重最优理论，且未提供实用算法的收敛性或应用示例。

---

## 41. RIS-Enabled Integrated Access and Relay: Empowering Collaboration Among BSs

**arXiv ID:** 2609.26953 | [PDF](https://arxiv.org/pdf/2609.26953v1)

**作者:** Hao Lin `[一作]` (King Abdullah University of Science and Technology), Mohamed-Slim Alouini `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 95640 | [OpenAlex ID](https://openalex.org/A5083193286)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在基站上部署共置可重构智能表面，构建集成接入与中继 (IAR) 架构，支持IoT设备通过反射/放大方式实现更可靠的接入。

**💡 创新点**

将RIS嵌入BS并通过波域任务转移实现流量卸载和资源共享，首次提出IAR网络在cell-free与cellular场景下的双重优势。

**🔧 技术方法**

利用RIS技术、固定/动态子载波分配策略、Monte Carlo仿真评估覆盖概率，并结合SINR阈值进行性能分析。

**📊 数据集**

采用10000次随机仿真，1 km²郊区模型，随机分布的基站（15个）、IoT设备（200个）与遮挡物，使用给定的路径损耗、Nakagami-m衰落参数。

**📈 对比分析**

与传统C‑BS网络对比，IAR在cell-free网络中覆盖概率提升至最佳时可达30%+，在cellular网络中提升18%（SINR阈值1时）并在低阈值下提升约2%。

**⚠️ 局限性**

主要局限包括RIS硬件成本与部署可行性、增强后的干扰与频谱竞争、复杂的动态资源与相位调度，以及在高移动性环境下的信号重配置时延。

---

## 42. HYDRO: Towards Non-Reversible Face De-Identification Using a High-Fidelity Hybrid Diffusion and Target-Oriented Approach

**arXiv ID:** 2609.27011 | [PDF](https://arxiv.org/pdf/2609.27011v1)

**作者:** Felix Rosberg `[一作]`, Fernando Alonso-Fernandez `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HYDRO，一种将目标导向生成器与单步扩散噪声/逆扩散相结合的面部去识别模型，兼顾图像保真度与属性保持；

**💡 创新点**

创新点包括：1）将扩散过程嵌入目标导向框架以消除可逆身份痕迹；2）引入 Eye Similarity Discriminator 提升眼部保真度与注视方向；3）通过单步噪声+逆扩散实现高效无损恢复；

**🔧 技术方法**

使用技术：U‑Net 生成器与扩散模型、身份编码器（ArcFace）、对抗损失、特征相似损失、遮罩正则、眼部相似判别器、LPIPS、以及多项属性保留损失；

**📊 数据集**

训练与评估数据集：FaceForensic++、CelebA、LFW；

**📈 对比分析**

在三大数据集上与多种 SOTA 方法对比，HYDRO 在 FID、姿态/表情/视线保留、MSE 以及身份检索率（FAR 10^-3/10^-4/10^-5）均优越，且重建攻击成功率平均下降 85.7%；

**⚠️ 局限性**

局限性：扩散步骤导致 GFLOPS 与推理时间显著提升，且在极端姿态、遮挡等边缘情况仍存在一定的失真与属性失效。

---

## 43. Resource-Efficient Distributed Recursive Gaussian Processes

**arXiv ID:** 2609.26979 | [PDF](https://arxiv.org/pdf/2609.26979v1)

**作者:** Josephine King `[一作]` (Delft University of Technology), Raj Thilak Rajan `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了两种面向多输出Gaussian过程的分布式递归算法（ADMM-RGP和PDMM-RGP），实现了在线、无中心化的分布式推理。

**💡 创新点**

创新点在于将ADMM和PDMM框架引入递归GP，提供稳定性分析与参数选择策略，显著降低通信开销，同时保持与Consensus-RGP相近的预测精度。

**🔧 技术方法**

采用诱导点稀疏GP、分布式ADMM/PDMM、拉普拉斯矩阵谱分析以及参数搜索方法。

**📊 数据集**

使用真实的ERA5气象数据（10 m风速的u、v分量），共10,000条测量点和400个基准点。

**📈 对比分析**

通过与Consensus-RGP及集中式RGP比较，RMSE一致（≈0.086），在不同图连通度下，ADMM-RGP可将通信轮数减少30–50%，PDMM-RGP在稀疏网络上进一步提升一致性。

**⚠️ 局限性**

局限包括仅针对同步通信、仅使用诱导点表示、未处理时间变化函数与拓扑变化，且在基准点数量较少时表现未知。

---

## 44. Humanoid Locomotion with a Fly-Inspired Recurrent Controller

**arXiv ID:** 2609.27001 | [PDF](https://arxiv.org/pdf/2609.27001v1)

**作者:** Isabel Guan `[一作]` (Hong Kong University of Science and Technology), Shipeng Lyu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了基于苍蝇神经网络的循环控制器在模拟 Unitree G1 人形机器人的步态表现。

**💡 创新点**

将昆虫神经连接组映射到人形机器人，分析其携带神经状态与运动行为的关系，并揭示了有效输入通路的结构与功能。

**🔧 技术方法**

使用 3,609 状态循环核心、线性投影、基于 MN 读出的动作映射、MuJoCo 物理仿真、以及干预实验等技术。

**📊 数据集**

在七种固定地形（平地、粗糙地、斜坡、楼梯等）、三种速度、三种初始偏航的共 63 个测试条件下评估。

**📈 对比分析**

与拥有额外地形感知的参考 R1 进行对比，T_graph 在所有 63 条件下均能通过 12 秒生存+前进判定；在重置动机状态的实验中表现大幅下降。

**⚠️ 局限性**

仅评估了固定检查点、模拟环境，缺乏训练记录、真实硬件验证，且深度输入通路未激活，无法证明连接组架构的优势。

---

## 45. LWCal: Loss-Weighted Calibration for Tabular Classifiers with Noisy Calibration Labels

**arXiv ID:** 2609.26839 | [PDF](https://arxiv.org/pdf/2609.26839v1)

**作者:** Zeming Liu `[一作]` (Brown University), Yuan Xie `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究后置概率校准在校准标签存在噪声时的表现，提出一种仅依赖原始模型预测和噪声标签的无清洗标签加权等距单调回归（LWCal）以及加入门控退回的改进版本（Gated-LWCal），用于提升表格分类器的校准精度。

**💡 创新点**

创新点在于：① 通过基于模型预测损失的自适应权重，对噪声校准标签进行软加权，避免直接使用噪声标签；② 在高噪声环境下引入门控机制，使校准器在出现大量标签与模型预测相悖时退回至原始概率，平衡校准误差与正确分数；③ 所有方法均不需要清洗标签、噪声率估计或重新训练基模型，保持了 CPU‑only 的轻量级部署。

**🔧 技术方法**

技术包括：损失加权等距单调回归（Weighted Isotonic Regression）、基于对数损失的权重计算、门控退回公式（g(p)= (1-ρ)g(p)+ρp）以及在实验中使用的随机森林、Extra Trees、梯度提升树等树型基学习器。

**📊 数据集**

使用了9个二分类本地表格数据集：Breast Cancer、Wine Class 0/2、Digits High/Even、Iris Versicolor、Synthetic Overlap、Synthetic Sparse、Synthetic Imbalance，均采用scikit‑learn内置数据和合成生成器。

**📈 对比分析**

对比方法包括原始模型、Platt、Beta、等距单调回归、硬裁剪版Beta/等距单调回归以及本研究提出的LWCal和Gated-LWCal。评价指标为ECE、Brier、NLL、AUC。实验覆盖432个噪声细胞（对称/非对称噪声、不同噪声率），结果显示：LWCal在ECE上降至0.113、Gated-LWCal降至0.122，且在Brier和NLL上保持或优于所有基线，AUC与原始模型差距不大，说明在中等噪声环境下能显著提升校准性能。

**⚠️ 局限性**

局限性包括：① 仅在树模型和表格数据上验证，未评估神经网络或其他领域；② 噪声模型为人工注入的对称/非对称噪声，实际工业噪声可能更复杂；③ 在高噪声（>40%）或极度不平衡/样本量极小的情况下，门控退回会导致校准效果退化；④ 未解决分布漂移问题，需与重要性加权等方法结合。

---

## 46. Student Use of LLMs and the Limits of AI-Generated Question Difficulty in Data Science Courses

**arXiv ID:** 2609.27063 | [PDF](https://arxiv.org/pdf/2609.27063v1)

**作者:** Yuan An `[一作]` (Drexel University), Lei Wang `[通讯]` (Drexel University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在德雷克斯勒大学的三门数据科学课程中，研究者通过多波调查和课堂检索练习，评估学生使用大型语言模型（LLM）的行为和LLM生成的多项选择题（MCQ）的难度标注与实际学生表现之间的关系。

**💡 创新点**

创新点在于首次系统检验LLM自我标注的题目难度与学生实际答题难度的构念效度，并揭示难度标签主要反映文本结构和模型内部标注规则，而非真正的学术难度。

**🔧 技术方法**

采用的技术包括多波匿名问卷调查、使用GPT‑5.4‑mini和Gemini‑3.1‑flash‑lite两种LLM自动生成题目、Spearman相关分析以及基于课堂答题数据的经典测验理论（CTT）难度计算。

**📊 数据集**

数据集包含378道LLM生成的MCQ（其中311道被部署）产生的7,888份学生答题记录，覆盖机器学习、海量数据与云计算以及应用机器学习三门课程，并配合四轮学生使用与满意度调查。

**📈 对比分析**

通过比较LLM自标难度与Bloom分类与学生答题正确率（1‑p）之间的相关性，结果显示两者相关性极低（ρ≈0.06/0.02），表明难度标签未能反映实际难度，验证了构念效度缺失。

**⚠️ 局限性**

局限性包括样本量有限、仅在单一高校单一教师环境下实施、匿名调查无法追踪个体变化、总体答题准确率过高导致分数方差不足，以及未对知识图谱生成的题目质量进行单独评估。

---

## 47. ZOCheck: CPU-Shadow Checkpointing for Zeroth-Order LLM Fine-Tuning

**arXiv ID:** 2609.27189 | [PDF](https://arxiv.org/pdf/2609.27189v1)

**作者:** Minqiu Sun `[一作]` (University of Delaware), Dong Dai `[通讯]` (University of Delaware)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对零阶优化（ZO）的容错训练系统，通过CPU影子进程实时重放日志并异步持久化检查点，实现在GPU关键路径上非阻塞的检查点与快速恢复。

**💡 创新点**

创新点在于利用ZO轻量级的种子与标量步进日志结构，构建可重放的CPU影子进程，结合成本模型动态选择快照策略，从而大幅降低检查点开销并保持精确恢复。

**🔧 技术方法**

使用的技术包括CPU影子重放、GPU无阻塞检查点、异步持久化、零阶优化日志记录、成本模型决策算法以及实验平台的分布式GPU训练。

**📊 数据集**

文章未公开具体使用的数据集，实验主要针对大规模语言模型的微调任务进行评估。

**📈 对比分析**

与传统异步全状态检查点相比，实验显示检查点开销降低高达219.7倍，恢复延迟缩短1.55倍，整体浪费时间降低多达21.3倍，验证了系统在容错效率上的显著提升。

**⚠️ 局限性**

局限性包括：1）仅针对零阶优化，未探讨第一阶梯度方法；2）实验环境与真实生产环境的差异可能影响效果；3）CPU影子进程的实现可能对CPU资源产生额外负担；4）成本模型假设的失败率和硬件特性可能不适用于所有场景。

---

## 48. Topological Signatures of Cyber-Attack Classes in Natural Visibility Graph Representations of Network Traffic

**arXiv ID:** 2609.26990 | [PDF](https://arxiv.org/pdf/2609.26990v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 49. Learning Expressive Humanoid Locomotion from Monocular Runway Videos for Robot Fashion Shows

**arXiv ID:** 2609.27003 | [PDF](https://arxiv.org/pdf/2609.27003v1)

**作者:** Kyrylo Kolesnichenko `[一作]` (Vilnius University), Jong-Hoon Kim `[通讯]` (Kent State University)

**通讯引用:** 12353 | [OpenAlex ID](https://openalex.org/A5100781827)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一个端到端的管道，能够把单目跑道视频转换为可直接部署在类人机器人上的走台动作策略；

**💡 创新点**

创新点在于将运动恢复、机器人重新定位、运动校正、策略训练与部署整合为一条无缝工作流，并且通过该流程实现具有表达性和服装展示功能的走台动作；

**🔧 技术方法**

使用了GVHMR进行3D运动恢复、SMPL-X模型、GMR进行机器人重新定位、BeyondMimic强化学习框架训练策略，配合Booster训练与部署框架以及MuJoCo仿真；

**📊 数据集**

主要数据来源是用户提供的单人跑道视频，随后生成3D运动后作为训练参考，不依赖公开大规模运动数据集；

**📈 对比分析**

通过与标准K1行走的步距比较（-0.8~1.8cm vs 5.1~11.4cm），以及在物理实验20次无跌倒、每次约23步，验证了学习策略在走台表现上的优势；

**⚠️ 局限性**

局限性包括：仅在Booster K1机器人上验证，缺乏对多机器人的通用性；缺少可控策略和大规模走台运动数据集；评估指标单一，未涉及时尚专家主观评估。

---

## 50. Water Surface Swimming in a Centipede and its Robophysical ModeL

**arXiv ID:** 2609.27088 | [PDF](https://arxiv.org/pdf/2609.27088v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 51. Meet, Compare, or Abstain: LatWeave for Deterministic Multi-Hop Question Answering on Knowledge Lattices

**arXiv ID:** 2609.27225 | [PDF](https://arxiv.org/pdf/2609.27225v1)

**作者:** Yuze Ren `[一作]` (ZenSmart Technology Co., Ltd.), Han Han `[通讯]` (ZenSmart Technology Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种将 Web 文本知识抽取为多维知识格子，并通过三种确定性算子（meet、compare、abstain）完成多跳问答的方法，LLM 只参与离线构建和查询规划，答案生成过程完全零 LLM、零训练、可审计。

**💡 创新点**

创新点：
- 将知识组织成可算子化的多维格子，实现零 LLM、零训练的确定性推理；
- 通过 meet/compare/abstain 算子实现结构化拒绝（abstain），避免随意回答；
- 使答案路径可逐条追溯、可复现，提升系统可审计性；
- 在六个公开基准上构建并量化其“可操作域”（operating envelope），对比传统概率式 RAG/KBQA 等方法。

**🔧 技术方法**

核心技术：
- 多维产品格子（product lattice）与格子代数（meet、compare、abstain）；
- LLM 一次性抽取类型化谓词（基于白名单、去重、维度归一化）；
- 规划阶段 LLM 生成查询计划，后续仅执行确定性算子；
- 采用 Fuzzy Concept Analysis、Dilworth 宽度、Dedekind–MacNeille 完成格子构建与优化；
- 设计两级查询规划（全局词表 vs. 实体邻域）并评估。

**📊 数据集**

使用数据集：MetaQA、2WikiMultihopQA、HotpotQA、MuSiQue、FRAMES、IIRC。

**📈 对比分析**

比较方法与性能：
- 在 2WikiMultihopQA（模板化多跳）上达到 EM 0.865、any‑hit 0.9975，明显优于传统 RAG/GraphRAG 等基线；
- 在 MetaQA（完整知识）上 meet 任何 hit 0.9975，几乎无损耗；
- 在 HotpotQA、MuSiQue、FRAMES 等边界集上，系统受知识覆盖或链设计错误限制，EM 分别为 0.1687、0.0248、0.0256；
- IIRC 上的结构化拒绝表现优异，abstain accuracy 0.971、leak 0.029；
- 与已有发布方法对比，零训练的 meet 方案在已知知识场景下可实现与监督方法相当的准确率。

**⚠️ 局限性**

局限性：
- 受离线抽取的知识覆盖率、维度碎片化、事实缺失等影响，导致多跳链中断；
- 对链设计错误（中间实体缺失关系）难以自适应重规划；
- 仅能处理格子内部的推理，无法执行需要在格子值上进行算术或统计的推理；
- 查询规划仍依赖单次 LLM 调用，对复杂语言表达解析可能存在误差；
- 结构化拒绝在知识不完整时是必要的，但也会导致高拒绝率，影响可用性。

---

## 52. A Hierarchy-Aware Video-Language Model Evaluation and Hyperbolic Baseline for Surgery

**arXiv ID:** 2609.27139 | [PDF](https://arxiv.org/pdf/2609.27139v1)

**作者:** Ana Manzano Rodríguez `[一作]` (University of Amsterdam), Cees G. M. Snoek `[通讯]` (University of Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了层次结构感知的评估框架和基于双曲几何的模型，用于手术视频语言模型的阶段和步骤识别，并评估其跨层次一致性和错误严重性。

**💡 创新点**

提出了首个包含单层级识别、跨层次一致性和层次错误严重性三项任务的评估套件，并通过在双曲空间中的包含锥实现阶段-步骤层次约束，显著提升模型的层次一致性和错误质量。

**🔧 技术方法**

采用了Lorentz双曲超平面嵌入、对比学习损失、层次包含锥约束、层次三元组采样以及基于TimeSformer和BERT的双编码器架构。

**📊 数据集**

在四个公开手术视频数据集上验证：MIPO（髂骨折复位），StrasbyPass70 和 BernBypass70（胃旁路手术），以及 GraSP（前列腺切除）。

**📈 对比分析**

与 CLIP、SurgCLIP_(β)^* 等基线对比，实验表明在树状结构更完整的数据集上，双曲模型在多层级准确率、跨层一致性和同辈/亲属错误率上均提升约 5–15%，并且错误严重性显著降低。

**⚠️ 局限性**

对步骤不严格属于单一阶段（DAG 结构）的数据集如 GraSP，层次约束效果有限；评估仅覆盖四个数据集且以零样本方式进行，未涉及训练监督；模型复杂度和推理时间相对传统欧氏模型略高。

---

## 53. $c$-Packedness versus $λ$-Low-Density in Geometric Graphs: Theory and Practice

**arXiv ID:** 2609.27231 | [PDF](https://arxiv.org/pdf/2609.27231v1)

**作者:** Gregor Diatzko `[一作]` (University of Konstanz), Sabine Storandt `[通讯]` (University of Konstanz)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文设计并实现了可扩展的算法，用于在百万级边的道路网络中近似计算 c‑packedness 参数和精确计算 λ‑low density 参数，并利用这些参数构造更小的平衡分离器、树分解和距离预言机。

**💡 创新点**

创新点包括证明 c∈O(λ√n) 且该界限紧凑；改进平衡分离器的大小至 6c 并保持 Θ(c²n) 时间；在多项式时间内构造宽度为 O(c) 的树分解；以及首次实现查询时间为 O(c) 的精确距离预言机。

**🔧 技术方法**

主要技术手段包括基于最近包围球/方块的近似与全局流算法、改进的最小包围球近似、分层分离器与树分解的递归构造，以及使用 Reed 算法得到宽度线性树分解。

**📊 数据集**

实验使用了从 OpenStreetMap 提取的多种道路网络，包括规模从十万到两千万顶点的实例。

**📈 对比分析**

与现有基于 KaHIP 分割器和传统距离预言机的基准相比，本文方法在分离器大小、树分解平均包大小以及查询次数上显著优于对手，查询时间与图大小无关，平均仅需 20 次距离比较。

**⚠️ 局限性**

限制在于对 c 的依赖仍然较大，尤其在预处理阶段的 O(c³ n log n) 复杂度；分离器算法虽然改进但仍受最大流计算瓶颈影响；此外理论参数间的关系虽然已改进，但在实际道路网络上仍有进一步优化空间。

---

## 54. Cryptographic Security Is Not Enough: Privacy Gaps in the Renegade Decentralized Dark Pool

**arXiv ID:** 2609.27100 | [PDF](https://arxiv.org/pdf/2609.27100v1)

**作者:** Prerna Arote `[一作]` (IMDEA Networks Institute), Lucianna Kiffer `[通讯]` (IMDEA Networks Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

论文分析了在实际部署的MPC+zkSNARK去中心化暗池Renegade中的隐私、公平与可用性缺陷，并证明了四类攻击（预匹配探测、选择性中止、链上链下关联、无效输入恐吓）能在实践中实现；

**💡 创新点**

创新点在于：①首次对公开部署的MPC暗池进行正式安全模型定义与证明；②系统性展示了多阶段攻击的具体实现与成本不对称；③通过对700k交易与P2P网络的实测，量化了攻击成本与网络中心化风险；

**🔧 技术方法**

使用技术包括：安全多方计算（SPDZ‑style 2PC）、零知识简短交互式论证（zkSNARK）、libp2p P2P网络、Merkle树承诺、Beaver三元组预处理、加密签名与哈希；

**📊 数据集**

使用数据集为：700,000+ Renegade Base链交易记录（交易、事件、ERC‑20转账）、P2P网络节点清单（四节点、两群集）、交易量与TVL统计；

**📈 对比分析**

与现有MPC暗池（如P2DEX、Rialto等）对比，实验显示：手势探测每秒可发约2次请求；MPC开局仅6.6 ms但预处理需779 s/20 GB，攻击者成本仅1.6 s；ZKP手势证明耗时约2–3 s；整体攻击成本相较于受害者有约487×的不对称；

**⚠️ 局限性**

局限性包括：攻击假设网络为完全去中心化但实际Relayer高度集中；分析基于Base链数据，其他链或未来升级可能改变；对抗措施仍需进一步研究以实现公平输出与防止无效输入导致的DoS。

---

## 55. COPE: Continual Personalization of LLMs under Sparse User Feedback via User Embeddings and Self-Evaluation

**arXiv ID:** 2609.26853 | [PDF](https://arxiv.org/pdf/2609.26853v1)

**作者:** Ruike Cao `[一作]` (University of Science and Technology of China), Li Xiao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 59704 | [OpenAlex ID](https://openalex.org/A5100355322)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了COPE框架，利用可学习的用户嵌入和自评机制，在稀疏反馈场景下实现持续个性化优化；

**💡 创新点**

将可学习的个性化嵌入与自评校准、强化学习响应优化三大目标整合到统一的“Interact‑Collect‑Optimize”循环中，形成一种新型持续个性化学习机制；

**🔧 技术方法**

采用强化学习（PPO）进行自评与响应优化，监督微调捕获用户偏好，以及基于查询扩展的检索增强、Qwen‑Flash模拟器等技术；

**📊 数据集**

使用PersonaLens benchmark数据集，构建256名用户、20个领域的对话任务，并采用受控时间序列协议进行实验；

**📈 对比分析**

与基线模型（Base、SR、DR、Q‑DR、PAP、T‑PAP）及其检索组合对比，COPE在不同反馈概率下均取得最高的个性化×完整性得分，最高达到5.36；

**⚠️ 局限性**

实验仅基于模拟用户，单轮对话且每位用户仅51次交互，缺乏真实长期交互数据与多轮场景验证，易受评估器偏倚影响。

---

## 56. How Constraints and Preferences Shape Travel Planning: Implications for AI Planning Support

**arXiv ID:** 2609.26968 | [PDF](https://arxiv.org/pdf/2609.26968v1)

**作者:** Fuling Sun `[一作]` (University of California San Diego), Haijun Xia `[通讯]` (University of California San Diego)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过访谈研究旅行者和旅行代理如何在旅行规划中处理约束和偏好，提炼出11个相关动作并给出设计启示。

**💡 创新点**

将约束和偏好视为动态、可演化的对象，提出“协作动作”词汇和对AI规划系统的设计建议，强调人机协同的规划过程。

**🔧 技术方法**

采用访谈与主题分析方法，结合 Gemini Pro 与 TripAdvisor 等现有AI规划工具进行实验与观察。

**📊 数据集**

访谈转录文本，共计8名旅行者和9名旅行代理的定性数据。

**📈 对比分析**

本研究未进行定量性能对比，主要基于质性分析揭示现有AI工具的不足与改进方向。

**⚠️ 局限性**

样本量小、便利抽样、AI工具交互时间短、缺乏跨文化与纵向验证，导致结论的普适性受限。

---

## 57. Text Scores Can Miss Waveform Use: A Qwen2-Audio Quantization Case Study

**arXiv ID:** 2609.26823 | [PDF](https://arxiv.org/pdf/2609.26823v1)

**作者:** Mengzhe Geng `[一作]` (National Research Council Canada), Junhao Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 243 | [OpenAlex ID](https://openalex.org/A5011790590)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种评估量化语音语言模型的协议，并在 Qwen2-Audio 案例中验证不同位宽下的文字输出、波形依赖行为和打包实现的差异。

**💡 创新点**

创新点在于将文字输出、波形依赖任务和实际打包实现三种评估维度分离，避免仅凭文本指标误判模型保留了音频信息，并通过同预算结构对照验证低位宽配置的真实效果。

**🔧 技术方法**

使用后训练量化（PTQ）配合均值 6/7 位、4.08 位混合精度分配，配合 dequantized RTN 仿真，结合统计不确定性估计与 bootstrap 置信区间。

**📊 数据集**

数据集包括 FLEURS (英德翻译)、LibriSpeech (ASR)、RAVDESS (情感识别)，以及自定义的冻结评测集和匹配预算控制集。

**📈 对比分析**

通过对比 FP16 基线，计算 chrF、WER、情感识别准确率的差值与 95% 区间；6 位配置提升翻译 chrF 但显著下降情感识别；7 位仍提升翻译且与 FP16 情感差异不显著；4.08 位所有低位配置平均下降约 10% 情感准确率；dequantized 6 位保持 FP16 内存峰值。

**⚠️ 局限性**

局限性包括仅针对单一 Qwen2-Audio checkpoint、单一英语情感数据集、翻译评测样本有限、不同位宽使用不同量化族、以及 dequantized 结果不具备实际打包效能。

---

## 58. QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs

**arXiv ID:** 2609.26855 | [PDF](https://arxiv.org/pdf/2609.26855v1)

**作者:** Kyaw Hpone Myint `[一作]` (Capital One), Giri Iyengar `[通讯]` (Capital One)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 Quartet 的四分支图变压器架构，用于关系深度学习，通过改进本地采样和全局上下文来提升多表数据库分类任务的性能。

**💡 创新点**

创新点包括：1) Causal Random Walk (CRW) 采样器，利用截断的个性化 PageRank 生成高连通、无时序泄漏的本地子图；2) 四分支全局交叉注意力模块，分别从特征、拓扑、时间和协作四个角度引入宏观上下文；3) 采用可区分的代码本和 Perceiver 风格瓶颈来高效聚合全局信息；4) 在 RelBench v1 上系统对比并进行留一消融实验验证各模块贡献。

**🔧 技术方法**

技术包括：图变压器、全自注意力、随机游走采样、截断 PPR、可学习代码本、Perceiver 跨注意力瓶颈、层归一化、全连接投影、基于 MLP 的特征融合、以及多任务评估框架。

**📊 数据集**

使用 RelBench v1 的 12 个二分类任务（来自 Formula‑1、事件推荐、临床试验、在线广告、电子商务、时尚零售、问答社区等七个数据集），并在 RelBench v2 流程下重新实现 Baseline（HGT、RelGT）。

**📈 对比分析**

与 HGT 和 RelGT 进行直接对比，Quartet 在 12 个任务中至少匹配或超越 RelGT 的 8 个任务，在 RelBench v1 上平均提升约 2% ROC‑AUC，消融实验显示 CRW 采样和四个全局分支分别贡献 1.5–2.0% 的性能提升；在大型静态任务中仍略逊于 RelGT。

**⚠️ 局限性**

局限性包括：1) 仍需要预先缓存子图，训练时不支持在线采样；2) 对极大图的显存/计算需求较高；3) 在某些静态任务中全局模块增益有限；4) 仅在二分类任务上验证，缺乏回归或链接预测的实验。

---

## 59. The Linear Representation Hypothesis Needs a Group Action

**arXiv ID:** 2609.27158 | [PDF](https://arxiv.org/pdf/2609.27158v1)

**作者:** Louie Hong Yao `[一作]` (Independent Researcher), Shengchao Liu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本研究建立了一个基于群作用的框架，用以明确线性表示假设的等价关系，并系统审计了常用解释方法与表示量化指标在不同等价群下的适用性。

**💡 创新点**

创新点在于将线性表示假设拆解为由等价群、对象空间、构造程序和谓词共同决定的命题族，并提出了可验证的规范，揭示了模型架构、读取点和多阶段分析对等价性的影响。

**🔧 技术方法**

主要技术包括群论与线性代数的抽象建模、等价群的定义与分类、以及对表示量化指标的对称性审计。

**📊 数据集**

本文未采用具体数据集，重点是理论框架和概念性审计。

**📈 对比分析**

通过构建审计表格评估常用量化指标在 G_aff、G_sim、G_iso 等不同等价群下的可定义性；实验结果未给出数值，主要是定性分析。

**⚠️ 局限性**

局限性在于仅关注等价性框架，未直接验证线性表示假设的实际有效性；对实际模型解释仍需结合具体实现，且对称性要求可能导致结论收窄。

---

## 60. Backdoors in Learning-Based Industrial Robotic Arm Manipulation: An Empirical Security Study

**arXiv ID:** 2609.26868 | [PDF](https://arxiv.org/pdf/2609.26868v1)

**作者:** Zijian Zhang `[一作]` (University of Wisconsin Milwaukee), Sandeep Pisharody `[通讯]` (MIT Lincoln Laboratory)

**通讯引用:** 545 | [OpenAlex ID](https://openalex.org/A5003360458)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6514db3d-8de6-452c-91b7-acdb31787cc4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对工业机器人学习控制中的后门攻击及其防御进行了实验研究

**💡 创新点**

创新点是提出基于物理触发器的后门攻击模型和在线观察清洗防御流程，并在真实工业臂上进行验证

**🔧 技术方法**

采用物理触发器注入演示、空间分割与注意力驱动的观察清洗、以及控制理论与深度学习方法

**📊 数据集**

使用FANUC LR Mate 200iD和xArm 6的真实演示数据，涵盖颜色排序和机械箱装两种任务

**📈 对比分析**

与离线微调防御相比，在线防御在保持任务性能（10/10无误）且无显著吞吐率损失的情况下，能有效阻断攻击（攻击成功率高达100%）

**⚠️ 局限性**

局限性在于假设触发器位于任务背景，未覆盖在任务对象或多模态触发器；在极端动态场景下可能被自遮挡或逃避

---

## 61. A Scaling Study for fMRI Foundation Models

**arXiv ID:** 2609.27232 | [PDF](https://arxiv.org/pdf/2609.27232v1)

**作者:** Wenhao Ye `[一作]` (Southern University of Science and Technology), Quanying Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

在超过200个fMRI数据集、10,000 GPU小时的实验中，系统评估了预训练数据量、模型参数规模和训练时长对下游任务性能的影响，并提出了基于ID任务性能的模型选择方法。

**💡 创新点**

① 在同一compute预算下揭示数据规模与模型规模需同步扩大才能最大化收益；② 通过对比不同数据/模型组合在相同EFLOPs下的表现，说明计算量本身不足以表征fMRI模型的规模化；③ 以ID性能为依据，构建响应函数并在两个预设预算下挑选最优配置，验证其在OOD任务上的竞争力。

**🔧 技术方法**

使用LeJEPA预训练目标（ROI‑time对齐 + SIGReg正则化），基于Vision Transformer的编码器，固定线性探针评估，EFLOPs计算估算，统计回归（β_D×N）分析以及ID性能响应拟合。

**📊 数据集**

预训练：>200个来源数据集；下游ID任务：ABIDE（自闭症诊断、年龄预测）、PNC（性别分类）、PPMI（诊断分类）、HCP（性别分类）；下游OOD任务：ADNI（阿尔茨海默/轻度认知障碍）、ADHD‑200（ADHD诊断）、BHRC（性别分类）、SALD、NKI‑RS（年龄预测/教育分类）。

**📈 对比分析**

通过冻结预训练模型并在12个ID任务上训练线性探针，计算平均分数（分类宏F1/准确率、回归Pearson相关系数），将不同数据/模型/compute配置的性能进行对比。随后在两固定预算下利用ID性能响应挑选模型，并在6个OOD任务上进行评估，结果显示所选模型在平均OOV表现上优于公开的fMRI基础模型基线。

**⚠️ 局限性**

① 研究仅覆盖约24.5倍的数据规模范围，未能验证对更大规模的外推；② 仅使用一种预训练目标和Transformer架构，缺乏对其他方法或多模态架构的比较；③ 计算量估算依赖于近似，可能存在误差；④ 仅评估冻结线性探针，未考察微调或更深层次的下游适配。

---

## 62. Comparative Evaluation of Static Embedding Models for HTTP Request Anomaly Detection

**arXiv ID:** 2609.26860 | [PDF](https://arxiv.org/pdf/2609.26860v1)

**作者:** Amanda Riverol `[一作]` (Tilsor SA), Álvaro Pardo `[通讯]` (Universidad Católica del Uruguay)

**通讯引用:** 394 | [OpenAlex ID](https://openalex.org/A5103289618)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 HEDA 框架，系统评估了 Word2Vec、FastText 和 Doc2Vec 三种静态嵌入在 HTTP 请求异常检测中的表现。

**💡 创新点**

创新点在于：①在统一的单类异常检测流程中对嵌入方法进行横向对比；②证明 FastText 通过子词建模显著提升了检测准确率与低误报率；③将嵌入与 OCSVM 结合，取得与传统 WAF 相比更优的性能。

**🔧 技术方法**

使用的技术包括静态词向量（Word2Vec、FastText、Doc2Vec）、一次类异常检测（OCSVM、LOF、Isolation Forest）、阈值优化（F̂ 评估）、以及基于 t‑SNE 的可视化。

**📊 数据集**

实验数据集为 Drupal（真实流量）、CSIC 2010（合成流量）和 SR‑BH 2020（蜜罐捕获的真实攻击），共涵盖数十万条 HTTP 请求。

**📈 对比分析**

通过 FPR、TPR、AUC 三指标对比，FastText‑OCSVM 在所有数据集上均实现最高 AUC（>0.9）与 TPR（>80%）且 FPR 维持在 <1%，优于传统 ModSecurity 规则和其它嵌入组合。

**⚠️ 局限性**

局限性包括：静态嵌入无法捕获长距离上下文；仅评估了三种嵌入与单类模型，缺乏对动态或上下文嵌入的实验；对聚合策略和实时部署性能的探讨仍待进一步研究。

---

## 63. Discover, Falsify, Revise: Auditing Input-Use Claims from Source Code to Predictive Contribution in Agent-Discovered Cell Models

**arXiv ID:** 2609.27234 | [PDF](https://arxiv.org/pdf/2609.27234v1)

**作者:** Mengran Li `[一作]` (Sun Yat-sen University), Zhenchao Tang `[通讯]` (Tencent AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了 AI 虚拟细胞模型中预测-声明缺口的问题，引入 CellAudit 框架通过源代码检查、模型行为替换和预测贡献三层验证输入使用声明。

**💡 创新点**

创新点在于将假设检验与输入使用声明的三重验证（源代码、拟合依赖、目标相关贡献）嵌入代理式模型搜索，形成从生成‑评分‑修订到发现‑falsify‑修订的循环。

**🔧 技术方法**

采用语言模型代理的自动化搜索、抽象语法树检查、冻结检查点后替换输入评估、残差学习、FiLM 位置编码、统计显著性区间等技术。

**📊 数据集**

使用的数据集包括 BBBC047/036（形态‑转录组联调）、sci‑Plex3（基因表达）、LINCS Pilot1、LKCP Batch2、Norman CRISPRa 等。

**📈 对比分析**

通过全局 Pearson 相关、控制/化合物替换效应和统计区间与参考阈值比较，发现尽管某些高分模型对化合物不敏感，但经过 falsification‑guided 修订后仍能提升约 0.003–0.004 的 PCC，audit‑enriched 反馈在 sci‑Plex 上平均提升 PCC 0.002‑0.003，部分指标区间仍跨零。

**⚠️ 局限性**

局限性在于预测泛化与输入声明泛化不一致、复现性受物理实验布局影响、评估区间仍跨零且未能完全证明因果或生物学可交换性。

---

## 64. UniDataAgent: An Ontology-Grounded Agent for Enterprise Question-to-Report Automation

**arXiv ID:** 2609.27257 | [PDF](https://arxiv.org/pdf/2609.27257v1)

**作者:** Yutai Duan `[一作]` (China Unicom Software Research Institute), Jie Liu `[通讯]` (Nankai University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个基于企业本体的问答系统（UniDataAgent），实现从自然语言问题到可交付报表的自动化流程。

**💡 创新点**

创新点在于将本体构建与执行分离，先离线构建可复用的版本化本体，再在线检索本体契约并执行专家预定义的分析技能，保证业务语义的一致性与可追溯性。

**🔧 技术方法**

采用本体构建技术（业务技能驱动的约束生成与验证）、LLM（如DeepSeek‑V4）进行规划与推理、ReAct式执行循环、SQL与计算工具集成以及证据验证机制。

**📊 数据集**

使用了中国联通27张企业表和数千种指标类型的数据集，并通过这些表中的元数据和业务文档进行本体构建与验证。

**📈 对比分析**

与文档RAG对比，UniDataAgent在40道真实业务问答中整体准确率从72.5%提升至95.0%，在完整地区比较和跨指标诊断任务上实现了100%准确率。

**⚠️ 局限性**

局限性包括对专家审查的依赖、在高影响变更时仍需人工介入、以及仅在中国联通内部场景验证，跨企业迁移需进一步评估。

---

## 65. Median Temporal Ensembling: Training-Free Robust Aggregation for Action-Chunked Visuomotor Policies

**arXiv ID:** 2609.27167 | [PDF](https://arxiv.org/pdf/2609.27167v1)

**作者:** Yuhang Jiang `[一作]` `[通讯]` (University of Trento), Yuhang Jiang (University of Trento)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并验证将行动块（action-chunked）视觉动作策略中的时间集成均值替换为坐标中位数聚合，以提升在对抗攻击和传感器失效下的鲁棒性。

**💡 创新点**

提出组合式鲁棒性保证（中位数聚合在至少一半候选被破坏时不失效），系统区分度量式与组合式鲁棒性，并揭示共同模式偏差对所有Δ族统计量的盲点。

**🔧 技术方法**

采用坐标中位数聚合、PGD对抗攻击、视觉编码器微调、对比实验与基线聚合器（均值、TIDE、ACC等）进行评估。

**📊 数据集**

使用Robomimic（LiftCube、CanMove、NutPlace）和PushT等仿真环境的视觉观测数据。

**📈 对比分析**

在25个配置、200条配对轨迹上与原始均值聚合比较，坐标中位数聚合平均提升15–18次成功率，单配置最高从0.335提升至0.675；在对抗攻击下恢复率显著高于细化训练。

**⚠️ 局限性**

仅在仿真环境评估，未测试真实硬件；步长h固定假设；对齐的共同模式偏差无法被任何Δ族聚合器消除。

---

## 66. Strip Convolution and Direction-Aware Exclusion Loss for Oriented Ship Detection

**arXiv ID:** 2609.27238 | [PDF](https://arxiv.org/pdf/2609.27238v1)

**作者:** Bin Chen `[一作]` (East China Jiaotong University), Chao Lu `[通讯]` (Jiangxi Vocational University of Foreign Studies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向航海图像的定向船舶检测方法，通过C3k2_Strip模块增强长船体特征表示，并用CA-DAEL损失抑制重叠船舶的冗余预测。

**💡 创新点**

创新点在于：①使用正交条卷积（7×3与3×7）在瓶颈层构造方向敏感的特征注意力，精准捕捉长船体；②设计类感知、方向感知的排斥损失（CA-DAEL），在训练阶段利用类别、角度和置信度信息主动消除同类重叠预测。

**🔧 技术方法**

核心技术包括YOLOv11‑OBB的单阶段anchor‑free框架、CSPDarknet骨干网络、条卷积注意力模块、CIoU回归、DFL边界细化、角度加权损失以及CA-DAEL排斥损失。

**📊 数据集**

在HRSC2016（1,061张图，2,976艘船）和DIOR‑R（23,463张图，192,472实例，20类）两个定向目标检测基准上进行实验。

**📈 对比分析**

与YOLOv8‑OBB、YOLOv26‑OBB、S2ANet、R3Det、RTMDet‑R‑tiny等方法对比，HRSC2016上mAP_50:95提升至78.45%（比YOLOv11‑OBB高6.32个百分点），DIOR‑R上mAP_50:95提升至53.71%（比YOLOv11‑OBB高3.37个百分点），模型参数仅2.91M，算力7.25 GFLOPs。

**⚠️ 局限性**

局限性：①条卷积长度固定为7，难以自适应不同尺寸或极端纵向比的船舶；②排斥损失在极其稠密的场景中仍可能误抑近邻不同类别船舶；③仅在VHR航图像上验证，跨传感器或低分辨率环境的泛化尚未评估。

---

## 67. The Gaussian Is Enough: Flow-Matching Priors Do Not Help When Fine-Tuning Large Behavior Models

**arXiv ID:** 2609.27070 | [PDF](https://arxiv.org/pdf/2609.27070v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 68. Gödel's and Scott's Variants of the Ontological Argument in Lean 4

**arXiv ID:** 2609.26806 | [PDF](https://arxiv.org/pdf/2609.26806v1)

**作者:** Christoph Benzmüller `[一作]` `[通讯]` (Otto-Friedrich-Universität Bamberg), Christoph Benzmüller (Otto-Friedrich-Universität Bamberg)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Lean 4 中实现了 Benzmüller 与 Scott 的 30 个模态逻辑理论的完整、结构保持的移植，重新证明了 Gödel 与 Scott 的形而上学论证的所有定理和未解决问题，并记录了每条定理的前置公理依赖。

**💡 创新点**

创新点在于将 LogiKEy 框架无缝移植到 Lean 4，保持结构与命名不变，并利用 Lean 的 #print axioms 对每条定理的公理依赖进行精细记录，从而揭示了原自动化证明隐藏的前置条件。

**🔧 技术方法**

采用了浅层语义嵌入（shallow semantic embedding）实现高阶模态逻辑在 Lean 的 Classical HOL 中的表达，并使用 Lean 4 的证明脚本、#print axioms 以及自定义脚本进行依赖分析。

**📊 数据集**

使用了 Benzmüller 与 Scott 在 AFP 上发布的 30 个理论文件（共 30 模块）以及原始 72 次 nitpick 结果，构成了完整的实验数据集。

**📈 对比分析**

与原始 Isabelle/HOL 版本的对比表明，在 Lean 中缺乏自动化工具导致证明需要显式手写，虽然生成了更长的证明，但通过 #print axioms 获得了更透明的公理依赖；性能方面证明过程更慢但完全可验证。

**⚠️ 局限性**

局限性包括：缺少模型查找器无法验证消除依赖的必要性；依赖分析只给出上界，无法确认最小前置；手工写证明增加了工作量；未能在 Lean 中重现原自动化工具所给出的五条未完成证明。

---

## 69. Quieter Than the Room: Representation Drift and Task Robustness in Speech Encoders

**arXiv ID:** 2609.27195 | [PDF](https://arxiv.org/pdf/2609.27195v1)

**作者:** Vsevolod Kovalev `[一作]` (Boston University), Pranay Manocha `[通讯]` (Princeton University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了非语音干扰（如背景噪声、音效）对冻结的自监督语音编码器表示漂移（drift）以及下游任务（意图识别、情感识别、说话人验证、语音识别）性能的影响，并探讨不同声音类型、响度和放置位置（语音段 vs 静音段）对漂移和任务损失的关系。

**💡 创新点**

创新点在于：①系统地把非语音干扰与语音表示漂移关联起来，并通过与重复录音的“取样参考”对比，将漂移量归一化；②在不同放置位置和响度下量化漂移与任务损失的对应关系，发现安静的静音段干扰可导致漂移与重复录音相当；③利用 Spearman 相关性表明，在整体录音干扰下，漂移与四个 SUPERB 任务的损失高度一致，但位置改变时漂移与任务损失可能不一致。

**🔧 技术方法**

技术包括：使用八个冻结的自监督语音编码器（HuBERT、WavLM、wav2vec 2.0、Whisper 等），通过最后层或层平均的向量取平均，计算余弦距离得到漂移；对干扰声进行 SNR 控制，并在整个录音、仅语音段或仅静音段三种放置方式下注入；下游任务采用 SUPERB 头部，训练仅在干净音频上；对漂移与任务损失进行 Spearman 相关性分析；利用 RAVDESS 语音变异作为漂移参考。

**📊 数据集**

数据集包括：RAVDESS（语音变异记录）、ESC‑50（14 类录音和 9 生成声音）、Fluent Speech Commands、IEMOCAP、VoxCeleb1、LibriSpeech；同时使用白噪声、粉噪声、和多种生成/录制声音做干扰实验。

**📈 对比分析**

比较方法：在七个编码器上，对每种声音和放置方式计算漂移，并将漂移与四个任务的性能下降（准确率、EER、WER）做 Spearman 相关。结果显示：在整体录音干扰下，漂移与任务损失的相关系数为 0.81–0.88；与声音占用率（duty）相比，漂移的相关性更高；在安静或较低 SNR 时，静音段干扰可产生与重复录音相当的漂移；而在较高 SNR 时，语音段干扰既产生更大的漂移又导致更高的任务损失。

**⚠️ 局限性**

局限性：①任务头部仅在干净音频上训练，未针对噪声进行微调，可能低估模型在实际噪声环境下的鲁棒性；②漂移仅通过余弦距离度量，未考虑与任务权重对齐的方向性；③实验仅覆盖固定的七个编码器和有限的干扰声音，结果的普适性待进一步验证；④缺乏对实际生活中更复杂混合噪声情境的评估。

---

## 70. Does Graph Structure Earn Its Place in Microservice Root-Cause Analysis? A Controlled Study on RCAEval, and What the Benchmark Was Really Measuring

**arXiv ID:** 2609.27069 | [PDF](https://arxiv.org/pdf/2609.27069v1)

**作者:** Imad Buljić `[一作]` `[通讯]` (University of Zenica), Imad Buljić (University of Zenica)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在微服务根因分析中进行受控实验，检验图结构是否带来性能提升，并发现其在RCAEval基准下无显著效果。

**💡 创新点**

提出严格的受控对照框架、12项实验检查清单，并引入先验分离模型PSC-GRCA，揭示基准隐藏先验导致的性能假象。

**🔧 技术方法**

使用GraphSAGE式图神经网络、MLP、PSC-GRCA三种模型；通过交叉验证、Wilcoxon符号秩检验等统计方法进行评估。

**📊 数据集**

采用RCAEval RE1（Online Boutique、Sock Shop、Train Ticket）和RE2（Sock Shop）两套基准数据集，包含375+90个故障注入案例。

**📈 对比分析**

在六折内分布下，图模型与平面MLP差距仅0.003，PSC-GRCA平均Avg@5 0.915；跨系统转移时平均0.747；其余实验显示图结构无显著提升。

**⚠️ 局限性**

实验受限于仅三系统、单一图网络架构、单一先验特征，跨系统样本不足且RE2未完整迁移，难以充分验证转移效果。

---

## 71. When Post-Processing Fairness Constraints Help and When They Harm: Evidence from Eight Cross-Domain Evaluations

**arXiv ID:** 2609.26955 | [PDF](https://arxiv.org/pdf/2609.26955v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 72. Policy-as-Skill: Governed LLM Decision Support with Evidence, Deterministic Control, and Audit

**arXiv ID:** 2609.27087 | [PDF](https://arxiv.org/pdf/2609.27087v1)

**作者:** Kabeh Mohsenzadegan `[一作]` (University of Klagenfurt), Kyandoghere Kyamakya `[通讯]` (University of Klagenfurt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Policy-as-Skill架构，将政策包装为可执行、可版本化的技能以支持LLM决策。

**💡 创新点**

将政策治理拆分为检索、验证、审计和可选确定性控制四个模块，并实现可追溯的审计记录。

**🔧 技术方法**

使用检索增强生成（RAG）、结构化提示、确定性控制器和可追溯的审计记录，基于Gemma4 LLM。

**📊 数据集**

构造了600例跨四类任务（问答、合规检查、风险分类、冲突检测）的合成/示例政策语料。

**📈 对比分析**

对13种方法进行对比，PaS+Audit在决策准确率和审查触发上优于传统RAG，PaS Full在某些任务提升准确率但整体不均衡。

**⚠️ 局限性**

确定性控制在非冲突任务上反而降低准确率，且实验依赖系统级调优，未做真实法律规范验证。

---

## 73. EMA: Elastic and Performance Transparent Memory Across GPUs

**arXiv ID:** 2609.27040 | [PDF](https://arxiv.org/pdf/2609.27040v1)

**作者:** Yi Xu `[一作]` (University of California Berkeley), Ion Stoica `[通讯]` (University of California Berkeley)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套跨GPU内存弹性共享系统，允许多卡节点中的GPU在运行LLM推理时动态借贷并回收彼此的显存，形成统一的弹性内存池；

**💡 创新点**

核心创新在于引入Memory Slice抽象，结合Local‑Remote对称与Static‑Dynamic对称原则，并通过预取与滑动窗口调度，使远程显存对应用完全透明且可即时回收；

**🔧 技术方法**

技术手段包括：NVLink高速互连上的Memory Slice分配与动态分配、slab/层组分配与预取、基于可用带宽与延迟窗口的传输调度、贪心调度与预抢占控制，以及Elastic Address Space映射与地址翻译；

**📊 数据集**

实验使用LLaMA与OPT大模型，配合Alpaca与ShareGPT推理负载，采用Poisson到达的请求流，涵盖4×A100‑40GB与8×A100‑80GB两套硬件；

**📈 对比分析**

通过与vLLM基线及理论上2×显存容量的对比，实验显示在高负载下可提升至52%吞吐率，达到原始系统的96%，且在大多数场景下保持与本地显存相当的延迟；

**⚠️ 局限性**

局限性包括：需要可预见的内存访问模式（主要适用于LLM KV缓存），slab/层组大小的权衡导致碎片与传输效率折中；实现依赖NVLink高速互连，且在极高内存占用时可能出现预抢占频繁导致的性能波动；

---

## 74. Bridging LLM Serving and CXL-SSDs with Chunk-Aware KV Cache Management

**arXiv ID:** 2609.26828 | [PDF](https://arxiv.org/pdf/2609.26828v1)

**作者:** Hyunsun Chung `[一作]` (Sogang University), Youngjae Kim `[通讯]` (Sogang University)

**通讯引用:** 3420 | [OpenAlex ID](https://openalex.org/A5100458491)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种专为大型语言模型KV缓存前缀重用设计的CXL-SSD，通过在设备端实现基于KV块的I/O、共享状态和异步/层级预取，解决了块设备接口导致的CPU缓存争用、主机DRAM复制和NAND延迟等瓶颈。

**💡 创新点**

创新点在于：①将KV块作为设备可见的I/O单元，①构建跨层面共享KV语义与迁移进度的双向接口；②在CXL-SSD内部实现基于预取提示的计算异步预取（CAP）和层级预取，①通过窗口化调度与GPU计算并行，隐藏NAND延迟；③使用设备DRAM作为GPU可直接访问的零拷贝缓冲区。

**🔧 技术方法**

技术手段包括：CXL.mem内存语义的SSD硬件/软件协同设计；在设备DRAM中维护前缀查找表、pin映射、free/预取队列；通过共享锁无锁结构实现与LMCache的高速交互；使用大页/多页块管理与时序化读写；采用CUDA直接内存映射实现GPU零拷贝DMA；实现两种预取策略：Compute Asynchronous Prefetching与Layerwise Prefetching。

**📊 数据集**

实验数据集主要有：多轮问答工作负载（CxS benchmark）、NVIDIA AIPerf agentic coding trace、P2P共享缓存实验；使用的模型包括Qwen3‑4B、Qwen3‑VL‑32B、Llama‑3.1‑70B、Qwen3‑Coder‑30B‑A3B；并与NVMe SSD、普通RAM、以及无预取的库存CXL-SSD做对比。

**📈 对比分析**

对比方法：在相同硬件平台（四块NVIDIA L40S、4 CPU、64 GB DRAM、CXL-SSD 16 GiB DRAM/384 GiB NAND）下，测量TTFT、ITL、吞吐量、P99尾部等指标。结果显示，CAP将平均TTFT相比库存CXL-SSD下降约2.6×，层级预取约4.0×，总体平均TTFT仅比本地DRAM慢1.5×；吞吐量分布变得更均匀，P99尾部显著缩短。

**⚠️ 局限性**

局限性包括：①设备端DRAM区域有限，对大前缀或高并发时需更大DRAM；②仍有数百毫秒的NAND等待尾部；③层级预取对低计算量模型或短前缀效果有限；④预取窗口与pin策略需手工调优；⑤当前实现仅支持单机LMCache，跨机远程预取仍受限。

---

## 75. Transfer Learning with Conformalized Quantile Regression for Solar PV Forecasting Under Load-Shedding-Driven Data Scarcity

**arXiv ID:** 2609.26959 | [PDF](https://arxiv.org/pdf/2609.26959v1)

**作者:** Rakib Abdullah `[一作]` (Green University of Bangladesh), K. M. Tahlil Mahfuz Faruk `[通讯]` (Green University of Bangladesh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过在已有的澳大利亚光伏数据上预训练 LSTM 网络，然后在孟加拉国的仿真光伏数据（并加入负荷削减导致的测量缺口）上微调，结合后置的 conformalized quantile regression 进行不确定性量化，从而在新建微电网的极端数据稀缺环境下实现日历预测。

**💡 创新点**

① 将迁移学习与 conformal prediction 在结构性缺失数据下联合使用，并证明迁移学习能显著稳定 conformal 校准；② 通过对光伏日内小时进行覆盖率诊断，识别出早晨上升期为最易失效区段；③ 采用基于 NASA POWER 的仿真数据与 BPDB 负荷削减掩码构建逼真的孟加拉国测试场景。

**🔧 技术方法**

长短时记忆网络（LSTM），迁移学习（源域预训练+目标域微调），conformalized quantile regression（CQR），pvlib 太阳能模拟，负荷削减掩码生成。

**📊 数据集**

源域：DKASC Alice Springs 光伏站 148,537 小时观测；目标域：利用 NASA POWER 气象重分析，pvlib 生成的孟加拉国 2023 年光伏仿真数据（8,760 小时），并以 BPDB 负荷削减统计为掩码。

**📈 对比分析**

对比四种模型（Scratch‑1mo、Transfer‑1mo、Scratch‑3mo、Transfer‑3mo），评估指标包括 RMSE、MAE、相对 24 h 持续性基准的 Skill Score、经验覆盖率、区间宽度。迁移学习在 1 个月时将 RMSE 降低 23.7%，Skill Score 由 ‑0.033 提升到 +0.212；在 3 个月时 RMSE 降低 13.7%，经验覆盖率从 84.17%（Scratch）提升至 94.31%（Transfer），且区间宽度缩小 14%。

**⚠️ 局限性**

主要限制：目标域数据为仿真，缺少实际测量噪声与遮阴等影响；负荷削减掩码采用聚合统计，未针对单个安装点；仅使用单一澳大利亚源站，迁移效果在更广泛或气候相近站点上未验证；缺失数据对 conformal 校准仍存在潜在偏差，需要进一步采用在线自适应 conformal 方法。

---

## 76. CoBranchMR: Supporting Parallel Design and Conflict Resolution in Mixed Reality

**arXiv ID:** 2609.27235 | [PDF](https://arxiv.org/pdf/2609.27235v1)

**作者:** Niloofar Sayadi `[一作]` (University of Notre Dame), Diego Gomez-Zara `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一款名为 CoBranchMR 的混合现实系统，支持分布式协作者对同一实体对象进行分支、并行编辑和合并，并在合并时可视化表面冲突并提供冲突解决方案。

**💡 创新点**

创新点在于将版本控制中的分支-合并流程引入到空间感知的混合现实环境中，使协作者能够在物理对象的虚拟表面上并行探索、共享并解决冲突，突破了传统 MR 系统只能共享单一对象的限制。

**🔧 技术方法**

主要技术包括：3D 高斯 splatting 进行实体重建；AI 生成的代理网格（Meshy AI）用于碰撞检测、绘制与物理交互；Unity 6、Meta Quest 3 以及 Meta XR Building Blocks 实现手部跟踪和 UI；Normcore（Normal.Realtime）实现实时多玩家同步；Unity LineRenderer 用于平滑绘制曲线。

**📊 数据集**

使用了实际拍摄的实体物体（如一只白色运动鞋）的照片作为重建输入，并通过 3D 高斯 splatting 生成其高质量的虚拟表面；并未使用公开大规模数据集，主要以单一实例进行原型验证。

**📈 对比分析**

目前未进行系统性定量比较或基准测试，原型仅在 Meta Quest 3 上实现了实时同步与冲突可视化，性能表现良好，能够保持流畅的帧率；未来计划通过受控双人研究评估协作效率和满意度。

**⚠️ 局限性**

局限性包括：仅支持两位协作者和单一对象；代理网格与高斯 splatting 之间可能存在轻微对齐误差；冲突检测仅基于 5 cm 距离阈值；缺乏对高斯 splatting 颜色直接编辑的支持；尚未验证数字设计在物理实体上的迁移效果；以及缺乏大规模多用户、多对象实验。

---

## 77. Do We Need Complex Topology Control? Distinct-Peer Random Routing Improves Cost-Efficiency in Sparse Multi-Agent Debate

**arXiv ID:** 2609.27150 | [PDF](https://arxiv.org/pdf/2609.27150v1)

**作者:** Boxuan Wang `[一作]`, Yi Dong `[通讯]` (University of Liverpool)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究稀疏多代理辩论（MAD）的通信拓扑和停顿控制，提出并评估随机无重复路由（Random‑NoDup）以及短辩论策略。

**💡 创新点**

创新点在于提出简单的随机无重复路由作为强基线，并证明在大多数任务与模型中，比复杂的学习或状态感知拓扑控制在准确-成本上更具竞争力；同时显示短辩论通过停止显著降低推理成本。

**🔧 技术方法**

使用多代理辩论框架、随机无重复路由与不重复采样、基于不确定性的路由与停顿策略、AgentPrune 的学习式时间边缘适配以及宏观统计评估方法。

**📊 数据集**

利用公开基准数据集：ARC‑Challenge、ScienceQA‑text、GSM8K、MMLU‑Pro 和 GPQA‑Diamond。

**📈 对比分析**

在八代理、两轮信息接收、完整辩论14轮或短辩论3轮的设置下，通过对比准确率和 token 消耗，发现 Random‑NoDup 在多数任务与模型中与结构化或学习路由相当或更优，且短辩论在保持相似准确率的同时显著降低 token。

**⚠️ 局限性**

局限性包括：结果依赖于所选模型和任务，某些任务上停顿可能降低准确；未探索更大规模或异质代理的场景；实验主要在同质八代理设置，需进一步验证在更复杂网络中的适用性。

---

## 78. KATOsuper: Surrogate-accelerated neural topology optimization with sensitivity-consistent Fourier neural operators

**arXiv ID:** 2609.27216 | [PDF](https://arxiv.org/pdf/2609.27216v1)

**作者:** Shengyu Yan `[一作]` (University of British Columbia), Jasmin Jelovica `[通讯]` (University of British Columbia)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 KATOsuper 框架，将神经网络重参数化拓扑优化与敏感性一致的 Fourier 神经算子（SC‑FNO）耦合，以在不需要频繁有限元求解的情况下加速拓扑优化。

**💡 创新点**

创新点包括：① 通过将密度通道与上下文通道分离，使预测的目标场与梯度保持一致，实现了敏感性一致的 surrogate；② 采用 Fourier 位置嵌入实现零步分辨率外推；③ 引入 KANConv3D 架构的 3D 扩展（KATO3D）；④ 结合在线学习策略在 3D 计算中动态校准 surrogate；⑤ 统一的生成器‑评估器架构，支持多目标、跨维度的可扩展性。

**🔧 技术方法**

使用技术包括：神经重参数化（GAN 风格 latent‑to‑density 生成器）、KAN（可学习 B‑spline 激活）卷积、SC‑FNO（频域卷积 + 自动微分敏感性）、多通道物理编码（含 Fourier 位置、载荷、边界、先验）、敏感性约束训练（AWL）、自动微分求梯度、PARDISO 稀疏求解器加速、PyTorch 2.0 编译、TF32 精度、在线 FEA 校准（周期性微调）。

**📊 数据集**

数据集来自 KATO 优化轨迹，包含三种 2D 基准（MBB、cantilever、L‑bracket）以及三种 3D 结构（标准 cantilever、split‑load cantilever、ship seat‑base）。每个轨迹在 50–100 步中抽取 40 帧，约 6,200 帧用于 2D 合规性 surrogate，约 3,200 帧用于 3D 合规性 surrogate；此外通过在线学习在 3D 任务中进一步采集 FEA 校准数据。

**📈 对比分析**

与 MATLAB OC/MMA 基准及基于 FEA 的 KATO 进行对比。KATOsuper 在 2D 合规性/应力、3D 合规性等任务中实现 15–110 倍的部署时间加速，同时在大多数基准上保持与传统方法相当或略优的最优性；在 4×–16× 分辨率外推下仍保持 1–8% 的性能差距；在 64× 外推时可接受的 29% 合规性损失。在线学习在 3D 任务中将 FEA 调用次数降低一半，整体求解时间缩短 2–3 倍。

**⚠️ 局限性**

局限性：① 零步分辨率外推受限于频谱带宽，极高放大率导致显著性能下降；② 需要先行训练数据，若仅有少量优化实例时纯 KATO 更合适；③ 对训练分布外的载荷/边界/材料假设会出现预测误差，在线学习虽可缓解但会降低加速比；④ 当前在线学习调度为固定频率，未能自适应；⑤ 生成器在极大网格下内存占用激增，限制最高可达分辨率；⑥ 对高度局部化、尖锐的敏感性场（如应力尖点）仍需更精细的梯度校正。

---

## 79. CRISP: Scalable Importance-Stratified Coresets for Imbalanced Tabular Learning

**arXiv ID:** 2609.26962 | [PDF](https://arxiv.org/pdf/2609.26962v1)

**作者:** Hardhik Mohanty `[一作]` (University of Southern California), Mohamadreza Sheibani `[通讯]` (Coinbase)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种线性时间的核心集选择方法 CRISP，用于在高度不平衡的表格数据上大幅减少梯度提升树训练行数。

**💡 创新点**

创新点在于将等计分位分层与按平均代理模型得分分配预算相结合，并通过逆概率权重校正采样偏差，同时保持分层覆盖，显著提升低预算下的模型性能。

**🔧 技术方法**

主要技术包括：轻量化代理 GBDT 评分、等计分位分层、按分层均值分配负类预算、Bernoulli 采样、逆概率权重（带截断）以及分布式 Spark 实现。

**📊 数据集**

实验数据集涵盖：约 25M 行的支付逆转欺诈数据（生产数据）、约 85M 行的 CriteoPrivateAds 广告点击数据、以及 1.3M 行的 Sparkov 合成欺诈数据。

**📈 对比分析**

与随机抽样、CCS、CoreTab 等基线对比，CRISP 在 95% 负类压缩时保持约 99.7% 的 AP（相较于完整数据），在 Criteo 上同样位居榜首；整体表现优于或等同于其他方法。

**⚠️ 局限性**

局限性包括：仅适用于正类比例足够小的情况、需额外代理模型训练和分布式计算、权重截断引入偏差、Sparkov 结果不够稳定，且未给出完整的端到端时间衡量。

---

## 80. Who Finishes the Job? A Study of Follow-Up Fixes and Commit Authorship on AI Coding Agent Pull Requests

**arXiv ID:** 2609.26847 | [PDF](https://arxiv.org/pdf/2609.26847v1)

**作者:** Wannita Takerngsaksiri `[一作]` (Deakin University), Scott Barnett `[通讯]` (Deakin University)

**通讯引用:** 832 | [OpenAlex ID](https://openalex.org/A5028222065)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文跟踪了 6,774 个合并的 AI 编码代理 Pull Request，并与同一仓库中 5,044 个合并的人类 PR 进行对比，研究了它们在合并后是否需要修复、谁来修复以及修复提交的作者归属。

**💡 创新点**

创新之处在于首次系统地量化 AI 代理 PR 合并后需要跟进修复的比例，并通过人工标注与 LLM 判断相结合验证修复对，进一步比较作者归属，揭示代理在修复中的自我修复行为。

**🔧 技术方法**

方法上使用了基于文件相同、时间窗口、PR 类型等四重启发式过滤的候选修复检索，随后利用人工验证与 Claude Opus LLM 判定，统计了 PR 级和提交级作者类别，并使用 Mantel–Haenszel、条件逻辑回归等统计方法评估差异。

**📊 数据集**

数据来源为 AIDev 数据集经 -pop 过滤后得到的 891 个热门仓库，收集了 5,774 个合并的 AI 代理 PR（包含 OpenAI Codex、Devin、GitHub Copilot、Cursor、Claude Code）和 5,044 个同仓库的人类 PR。

**📈 对比分析**

在同一 30 天观察窗口内，代理 PR 的经验证修复率比人类 PR 高 1.62 倍（置信区间 1.10–2.39），代理 PR 的 69.6% 由自身修复，提交级别的代理作者占比平均 87.4%，与人类 PR 的 89.1% 自修复形成对比；在合并时的信号差异微弱，仅在提交数上呈显著关联。

**⚠️ 局限性**

局限性包括只涵盖五个代理和热门仓库，30 天窗口可能漏掉更长周期的修复，且代理提交标识不统一导致作者归属可能低估；LLM 判定与人工一致但仍存在偏差；跨文件或多 PR 的修复可能未被捕获，结果仅适用于当前样本。

---

## 81. Backtracking Candidate Elimination: A One-Pass Algorithm for the Chip Testing Problem

**arXiv ID:** 2609.26995 | [PDF](https://arxiv.org/pdf/2609.26995v1)

**作者:** Shiyi Chen `[一作]` `[通讯]` (University of California, Berkeley), Shiyi Chen (University of California, Berkeley)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种顺序一次扫描的“回溯候选消除”算法，在芯片测试问题中仅需最多n-1次测试即可找到一枚保证良好的芯片。

**💡 创新点**

创新点在于用堆栈维护“同质”保留芯片，借鉴Boyer–Moore多数投票思想，使得算法不需要层级递归、偶数/奇数的特殊处理，可在线执行并具备提前终止判据。

**🔧 技术方法**

核心技术是基于一致/不一致测试的两条不变式（保留集同质性与丢弃对至少含一坏芯片）以及堆栈回溯机制，将传统的两两消除改为单向扫描。

**📊 数据集**

使用的“数据集”是理论模型：n枚芯片，其中良好芯片数>坏芯片数，无真实实验数据。

**📈 对比分析**

与传统的“二分消除”递归方法相比，回溯算法保持相同的O(n)测试上界，且在大多数情况可提前终止，平均测试次数可低于n-1；在最坏情况下测试次数与递归法相同。

**⚠️ 局限性**

局限性包括：需要O(n)额外堆栈空间；若假设“良好>坏芯片”被违背，算法无法验证并会错误返回；此外在单向测试模型中仍需维护完整堆栈，无法进一步压缩空间。

---

## 82. Fast Geometric Spanners via Approximate Nearest Neighbor Search

**arXiv ID:** 2609.26934 | [PDF](https://arxiv.org/pdf/2609.26934v1)

**作者:** Alexandr Andoni `[一作]` (Columbia University), Tian Zhang `[通讯]` (University of Pennsylvania)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种在一般度量空间中使用批量近似最近邻（BANN）oracle，以亚二次时间构造几何稀疏张量（spanner）的算法，并给出了对应的下界；同时将该算法应用于快速近似 Wasserstein‑q 距离。

**💡 创新点**

创新点在于建立了 BANN 与 spanner 构造之间的严格等价关系，提出了利用哈希技巧在度量空间中实现 Baswana‑Sen 样式稀疏张量的黑盒方法，并证明在查询预算有限的情况下，任何算法都必须接受 O(ck) 的失真，从而证明了算法的最佳性。

**🔧 技术方法**

核心技术包括：1) 黑盒 BANN oracle 的调用；2) 采用 Baswana‑Sen 结构与层次阈值来构建 spanner；3) 哈希化候选列表以减少 oracle 调用；4) 通过高概率分析与 Chernoff 边界保证性能；5) 构造高环度图与对抗 BANN 策略来证明下界；6) Yao 原理推广到随机化算法。

**📊 数据集**

论文主要是理论研究，没有使用具体数据集；所有结果均在抽象度量空间上给出。

**📈 对比分析**

相较于已有方法，本文在 ℓ_p（p>2）空间实现了第一套真正亚二次时间的稀疏张量构造，并在此基础上得到 W_q 距离的快速近似，匹配了最优的大小-失真折衷；实验比较未给出，但理论复杂度显著优于之前的多项式/近线性算法。

**⚠️ 局限性**

局限性包括：仅适用于能高效实现 BANN 的度量空间；算法性能受 BANN 运行时间与度量的 aspect ratio 影响；下界仅在查询预算小于 O(n^{1+1/k}/k) 的情形；并未给出实际数据集上的实验验证。

---

## 83. Safety Nudges: User-Facing Interventions for Real-Time AI Risk Awareness

**arXiv ID:** 2609.26865 | [PDF](https://arxiv.org/pdf/2609.26865v1)

**作者:** Varshini Elangovan `[一作]` (Carnegie Mellon University), Virginia Smith `[通讯]` (Carnegie Mellon University)

**通讯引用:** 17233 | [OpenAlex ID](https://openalex.org/A5027859459)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一个浏览器扩展“Safety Nudges”，通过对聊天记录调用外部大型语言模型进行实时审计，识别并可视化对话中的潜在安全风险，向用户展示相应的“警示”提示，并收集用户反馈与交互日志，评估其对用户风险感知和使用行为的影响。

**💡 创新点**

创新点在于：
• 采用用户面向的轻量级提示（nudge）而非传统的内容过滤或模型拒绝；
• 使用外部审核模型保持与目标聊天模型的独立性，提供第三方监督；
• 在提示中附带证据片段与风险级别，增强可解释性；
• 结合大规模真实用户的两周现场实验，系统性评估提示对用户认知与行为的影响。

**🔧 技术方法**

主要技术包括：
• 浏览器扩展（Manifest V3）实现会话检测、上下文提取与后台服务；
• 外部LLM（如Claude或ChatGPT）作为审计器，接受系统/用户提示模板，对话文本与风险分类，返回JSON结构化判断；
• 前端在聊天界面中渲染提示芯片、弹窗与高亮证据；
• 反馈与日志收集，后端存储与分析。

**📊 数据集**

数据集与实验材料：
• 100条手工标注的对话（来自WildChat、ShareChat、内部测试及合成）用于验证审计提示；
• 45名美国用户的两周真实使用日志，共计6,674条助手回复、1,075个会话；
• 通过Qualtrics收集前后对比调查问卷与后续跟进问卷。

**📈 对比分析**

比较与效果评估：
• 通过配对Wilcoxon检验评估前后问卷差异：提示提升了对过度自信、幻觉、恭维等风险的可见性，用户对聊天回复的正确性评价下降（显著），但总体信任维持不变；
• 约67%用户认为提示有用，93%能回忆起具体提示，约82%认为不具干扰性；
• 对提示的帮助度按风险类别差异：健康/法律依赖和过度自信得分最高，拟人化得分最低；
• 交互日志显示提示后用户挑战/纠正回复的比例略有上升，但整体直接审查行为未显著提升。

**⚠️ 局限性**

局限性：
• 基线阶段与提示阶段的时间顺序可能导致外部因素影响结果；
• 仅在ChatGPT/Claude浏览器界面实验，结果可能不适用于其他语言、平台或更长时间使用；
• 审计模型可能因缺乏完整上下文（记忆、搜索、上传文件）产生误报或漏报；
• 用户对“无问题”提示的误解可能导致过度依赖审计器；
• 将对话内容发送至外部模型带来隐私与数据安全风险；
• 评价主要基于主观感受与日志，缺乏客观的风险准确率与行为改进测量。

---

## 84. SkillApt: Learning When to Activate Agent Skills from Counterfactual Evidence

**arXiv ID:** 2609.26863 | [PDF](https://arxiv.org/pdf/2609.26863v1)

**作者:** Shuang Guo `[一作]` (Central China Normal University), Shuang Guo `[通讯]` (Central China Normal University)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5113367242)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 SkillApt，一种在检索后决定是否激活已检索到技能的控制器，利用历史 WITH / WITHOUT 执行对比来评估技能在当前状态下的边际效用。

**💡 创新点**

创新点在于将技能适用性视为状态与模型条件下的边际效用，区别于传统的语义相关性；采用对照实验（counterfactual）构建证据并通过轻量级证据条件化评分实现可解释且无学习参数的激活决策；并将检索与激活拆分为两步独立决策。

**🔧 技术方法**

使用了对照实验构建、证据条件化评分、签名特征哈希（state/skill 256 维向量）、加权相似度计算、阈值 0.5 的 LOAD/ABSTAIN 策略，以及自举抽样评估不确定性；全部实现无学习参数，保持可解释性。

**📊 数据集**

主要数据集包括 SRA-Bench（5400 任务，26262 技能，636 金标技能）以及 SpreadsheetBench（用于模型诊断的 Terra、Qwen3-32B、Mistral-Small-3.2-24B、Claude Sonnet 5 等模型）；此外还使用匿名的多智能体软件工程轨迹来验证环境有效性。

**📈 对比分析**

与 BM25 Top-1、Semantic-only、SkillApt-E+Sem 等基线比较；SkillApt-E 在保持与 BM25 Top-1 相同的准确率 0.838 的同时，将激活率从 100% 降至 31.5%，平均 token 使用量下降 74.3%；同时避免了 100% 的正确性损害和 91.6% 的成本负面激活；跨模型诊断显示不同模型对技能效用和激活边界的可学习性差异。

**⚠️ 局限性**

限制包括检索错误限制了可实现收益；轻量级控制器导致可用目标召回有限且存在负激活；每技能证据稀疏、未划分训练/验证/测试集；模型依赖性强，难以跨模型直接迁移；环境有效性需单独检测；未覆盖完整的成本敏感分类法，真实世界实验样本有限。

---

## 85. LEGO: Synergizing Expert GraphRAG and Expert Chain-of-Thought for Legal Reasoning

**arXiv ID:** 2609.27009 | [PDF](https://arxiv.org/pdf/2609.27009v1)

**作者:** Qingjing Chen `[一作]`, Weixing Shen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了使用ACL会议样式文件的说明，并以自身为示例展示如何编写符合规范的手稿。

**💡 创新点**

创新点在于将样式文件使用说明与实际示例结合，帮助作者快速上手并减少格式错误。

**🔧 技术方法**

主要技术手段是使用LaTeX模板和相关样式文件来排版论文。

**📊 数据集**

无数据集，本文仅为格式规范说明。

**📈 对比分析**

不涉及实验或方法对比，因其本身是格式使用手册。

**⚠️ 局限性**

局限性：只覆盖ACL会议的格式要求，无法直接应用于其他会议或期刊；缺少针对不同学科风格的细节说明。

---

## 86. What Changes When Fact-Verification Scores Improve? Evidence and Answer Accounting Across Trained Verifiers and LLMs

**arXiv ID:** 2609.27064 | [PDF](https://arxiv.org/pdf/2609.27064v1)

**作者:** Han Chen `[一作]`, Yingrui Li `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对联合事实验证得分的提升来源进行了分解，量化了在答案保持不变时更换证据对得分的贡献，并通过对训练好的 DeBERTa 检查点与冻结的 8B LLM 在 FEVER、FEVEROUS 与 SciFact 上的实验，评估了上下文长度与接口选择对证据收益的影响。

**💡 创新点**

创新点在于引入 2×2 分数表和 Shapley 风格的平均，系统拆解得分提升为答案与证据两部分，揭示了大部分提升可归因于证据改进；并首次将上下文预算对固定答案证据收益的交互效应在多数据集上进行跨比较。

**🔧 技术方法**

技术上使用了 DeBERTa‑large‑MNLI 检查点与 UnifEE/ DCUF 证据提取、BM25 与 BGE 排序、Qwen3 与 Llama 3 8B LLM、Bootstrap 置信区间、Bonferroni 校正、严格得分与答案准确率的乘积因式分解等。

**📊 数据集**

实验数据来自 FEVER、FEVEROUS 与 SciFact 开发集，分别包含 2,000/2,000/300 条主张，FEVEROUS 还使用了 7,890 条验证集主张。

**📈 对比分析**

通过生成并保存答案与证据的四种组合，计算严格得分、答案准确率与宏 F1，并用配对 Bootstrap 评估置信区间；结果表明更换证据可将严格得分提升约 9.6 个百分点，其中约 8 个百分点在答案固定时仍然保持；在 FEVEROUS 上扩大上下文至 2,048 词可使固定答案证据收益提升约 3.8–3.1 个百分点，尽管跨数据集一致性未完全满足。

**⚠️ 局限性**

局限性包括：训练检查点仅在 DCUF 输入上微调，证据切换会改变输入分布，导致答案部分可能受影响；仅评估无 NEI 召回的模型；仅使用 8B 固定 LLM，未覆盖训练或模型选择不确定性；证据合格度基于标注 ID，缺乏人工相关性判断；跨数据集比较受限，结果可能不适用于更强或不同语言的验证器。

---

## 87. Boyer-Moore Variants for Indeterminate String Matching and Experimental Evaluation

**arXiv ID:** 2609.27170 | [PDF](https://arxiv.org/pdf/2609.27170v1)

**作者:** Neerja Mhaskar `[一作]` (McMaster University), Nivetha Raj Pappuraj `[通讯]` (McMaster University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对含有不确定位置的字符串（indeterminate strings）提出了新的 Boyer–Moore 变体，设计了四条坏字符规则（BC Rule I–IV）以及一种快速的好后缀规则（Fast_GSR_Indet_Shift），并通过实验验证了其性能优势。

**💡 创新点**

创新点在于将传统 Boyer–Moore 的坏字符表扩展为位置索引的表 τ'，实现对不确定字符的高效匹配，并结合新的好后缀计算，显著降低了匹配时间（最坏情况提升至 O(√m)）。

**🔧 技术方法**

采用了基于 2^k 编码的字母表表示、位置索引坏字符表 τ'、Fast_GSR_Indet_Shift 算法以及多种坏字符规则的组合实现，所有算法均用 C++ 实现。

**📊 数据集**

实验使用随机生成的三种字母表（σ=4、9、20）以及真实的 E. coli K-12 MG1655 基因组（含 IUPAC 模糊码）进行评估。

**📈 对比分析**

将所提出的 Fast_BM_Indet 变体与经典的坏字符变体、Holub 的算法、BM_Indet、KMP_Indet 及暴力匹配进行对比，采用运行时间(ms)随文本长度变化的曲线进行衡量；结果显示所有 Fast_BM_Indet 变体均比 BM_Indet 提升两位数，在小字母表下与暴力匹配相当，且 Rule I 在大字母表上表现最优。

**⚠️ 局限性**

局限性包括最坏情况仍需 O(nm + mσ²) 时间与 O(mσ) 空间，且实验仅覆盖了有限的字母表与数据类型；在极大字母表或高度重复模式下的表现尚未充分验证，且实现仅为单线程版本。

---

## 88. Benchmarking Automated Knowledge Graph Construction from Semi-Structured Data

**arXiv ID:** 2609.26985 | [PDF](https://arxiv.org/pdf/2609.26985v1)

**作者:** Tarek Al Mustafa `[一作]` (German Centre for Integrative Biodiversity Research), Birgitta König-Ries `[通讯]` (Friedrich Schiller University)

**通讯引用:** 3923 | [OpenAlex ID](https://openalex.org/A5024963110)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个完整的半结构化数据到知识图谱自动化构建的基准与评估管道，涵盖映射、实体化和下游问题回答三阶段。

**💡 创新点**

创新点在于整合了六大质量维度（语法有效性、语义准确性、一致性、简洁性、完整性、实用性），实现了可重复的映射与KG质量评估，并提供了十套跨领域专家策划的数据集。

**🔧 技术方法**

采用BLINKG映射表格式与RML编译、OWL推理器（HermiT）、SPARQL查询生成等技术，对系统输出进行语法、语义、推理与问答层面评测。

**📊 数据集**

使用十个真实世界半结构化数据集，覆盖生态学、生物多样性、基因组学、聚合物化学和历史等七个领域，每个数据集包含原始表格、竞争问题、金标准答案、目标本体及参考映射。

**📈 对比分析**

在两种参考系统（映射生成器和完整KG生成器）上进行基准测试，评估指标包括映射精确度、KG一致性、完整性和CQ回答 F1 等，实验表明在目标本体场景下系统可达 70–90% 的映射准确率，而实用性 F1 约 0.65–0.75。

**⚠️ 局限性**

主要局限包括：对非目标本体的语义准确性无法评估；对大型本体的推理与一致性检查耗时；数据泄露与预训练模型偏差；并且 CQ 设计对评测结果影响较大。

---

## 89. Adversarial Attacks and Identity Leakage in De-Identification Systems: An Empirical Study

**arXiv ID:** 2609.27022 | [PDF](https://arxiv.org/pdf/2609.27022v1)

**作者:** Felix Rosberg `[一作]`, Fernando Alonso-Fernandez `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究了在真实去识别框架下，对身份编码器进行对抗攻击导致身份泄漏的现象，验证了攻击可通过迁移性从外部代理模型（如CosFace）转移到目标系统（ArcFace）。

**💡 创新点**

创新点在于首次揭示了去识别系统的身份编码器易受对抗攻击迁移的攻击导致身份泄漏，并提出了两种可行的缓解方案：对抗样本蒸馏微调与低通（高斯模糊）滤波。

**🔧 技术方法**

主要技术包括对抗攻击生成（PGD、Elastic、Fog等）、多代理模型梯度计算、身份编码器微调（知识蒸馏）、低通滤波器与DeepFaceDecoder可视化。

**📊 数据集**

使用的数据集包括FaceForensics++、VGGFace2以及LFW，用于训练/微调和评估身份检索性能。

**📈 对比分析**

与未加防御的基线相比，微调后在不同FAR阈值下身份泄漏率显著下降，尤其是低通滤波可将FAR10^-5下的泄漏率降至≈0；同时对比实验表明微调对识别精度影响可忽略。

**⚠️ 局限性**

局限性包括：仅在FIVA模型上验证，未涵盖贴纸等真实世界对抗样本；对抗攻击种类有限；迁移效果受代理模型相似度影响；低通滤波可能在极端噪声下影响可视化质量。

---

## 90. CORE-STACK+: Meta-Learning for Deep Stacked Generalization

**arXiv ID:** 2609.26905 | [PDF](https://arxiv.org/pdf/2609.26905v1)

**作者:** Noor Islam S. Mohammad `[一作]` `[通讯]` (Istanbul Technical University), Noor Islam S. Mohammad (Istanbul Technical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个包含四个组件的预处理与融合管线，专门用于异构视觉模型（CNN、ViT、Swin 等）的高效集成，解决多重共线性与校准失真问题。

**💡 创新点**

①使用 Centered Kernel Alignment (CKA) 进行非线性冗余过滤；②引入可学习的差分元特征门控；③基于 Marchenko‑Pastur 理论的闭式谱自适应岭正则化；④采用 Laplace 近似的贝叶斯混合器，并给出了首个同时考虑冗余与元学习器容量的 PAC‑Bayes 边界。

**🔧 技术方法**

CKA 核矩阵、梯度可学习门控网络、谱自适应岭回归、Laplace 近似贝叶斯融合、PAC‑Bayes 理论、Nyström 近似与 LSH 加速。

**📊 数据集**

ImageNet‑1K、ImageNet‑C、ADE20K、COCO、iNaturalist‑2021、DomainNet‑126。

**📈 对比分析**

与单一模型、均值、加权均值、Ridge 堆叠、贪婪选择、深度集成、模型汤、SWAG、快照集成、MC dropout、AutoGluon stacker 等方法对比，全部基准上取得最高精度、最低 ECE、最佳鲁棒性，并在模型数量和 FLOPs 上显著压缩（模型减少 35–57%，FLOPs 降低 41%）。

**⚠️ 局限性**

仅针对预测空间预处理，仍需大量 OOF 预测；对极大规模模型池的计算开销仍高；缺乏对动态环境下在线适应的机制；对非视觉任务的通用性未验证。

---

## 91. Bend the Clock: Predicting Ahead to Beat Latency in Event-Based Object Detection

**arXiv ID:** 2609.26919 | [PDF](https://arxiv.org/pdf/2609.26919v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 92. Count Evidence, Not Sentences: Tempered Evidence Fusion of LLM Judgments for Long-Text Value Measurement

**arXiv ID:** 2609.27165 | [PDF](https://arxiv.org/pdf/2609.27165v1)

**作者:** Yuhe Wu `[一作]` (HKUST(GZ)), Guang Zhang `[通讯]` (HKUST(GZ))

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了如何利用大语言模型对长文本进行公共价值取向测量，并提出了基于信息增益的温度证据融合（TEF）方法。

**💡 创新点**

将长文本测量视为决策融合问题，设计了在不需要额外训练的前提下，通过句子熵加权的对数似然比进行软裁剪的融合规则。

**🔧 技术方法**

采用句子级别的概率输出、熵权重、对数似然比转换、温度裁剪，并在多语言大模型上进行评估。

**📊 数据集**

构建了跨语言（中文、英文）长文本测量基准MIND，包含8358条帖子，涵盖六个公共价值维度。

**📈 对比分析**

与直接预测、Majority Vote、Soft Vote 三种基线在五个模型两种语言上对比，TEF平均提升约4.5个百分点准确率、4.6个百分点宏F1，并在校准和鲁棒性上表现最佳。

**⚠️ 局限性**

依赖模型概率的校准程度，处理极长文本需先分句，且在部分维度或模型规模较小时提升有限。

---

## 93. Propose, Don't Judge: An Anytime-Valid Referee for LLM Agents That Mine Investment Factors

**arXiv ID:** 2609.27051 | [PDF](https://arxiv.org/pdf/2609.27051v1)

**作者:** Bo Qu `[一作]` (DeepGrounding), Licheng Wang `[通讯]` (AlphaAvatar)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实证了一种“受治理的自我演化（governed self‑evolution）”框架，将量化因子挖掘中的提议与判定任务拆分，形成一个不可被代理修改的冻结统计裁判（referee）与可进化的代理（controller）共存的管道。

**💡 创新点**

创新点主要包括：① 通过使用e‑process、在线e‑Benjamini–Hochberg（e‑BH）和e‑detector实现对随时间更新的因子申请和终止的多重检验控制，保证任何提议策略下的假发现率（FDR）和误退回率（false‑retirement）；② 在同一管道下对不同提议者（脚本、UCB bandit、LLM）进行分层对照，清晰区分判定与提议的贡献；③ 通过LLM自创诊断探测器（probe authoring）展示代理在传统多臂老虎机无法实现的“工具创作”能力。

**🔧 技术方法**

核心技术包括：e‑process（基于公平投注的检验过程）、在线e‑BH（随候选加入自动更新的FDR控制）、e‑detector（自适应变点检测）、白化（whitening）和可预知的上界估计、基于Rank‑IC的因子评估、以及对因子持仓频率与成本的动态决策。

**📊 数据集**

数据集涵盖：① 由人工植入真因子的合成实验；② 通过“probe‑authoring”环境对未见故障类型进行检验；③ 以中国CSI‑500指数成分股为基础的真实行情，2015‑2026年期间的十年walk‑forward回测。

**📈 对比分析**

比较方法：将3种提议者与4种裁判器（含三种泄露策略）按全因子组合进行方阵式实验，每种组合在相同随机种子下跑5次。性能指标包括：每次提交的实现收益率、被批准的真因子比例、假入选率（false admissions）、退休误报率、以及最终的已认证组合与无门槛组合的Sharpe比。实验结果显示：裁判器决定假入选数量（冻结裁判能减少5‑11倍假因子）；LLM提议在实现收益上与bandit持平甚至优于脚本，并通过自创探测器显著降低诊断成本；认证组合的Sharpe比落后于无门槛组合，主要因等待时间长和每日预测能力不足导致。

**⚠️ 局限性**

限制包括：① 只在单一市场（CSI‑500）及有限的起始年份进行测试，缺乏跨市场验证；② 真实数据实验为历史回测，未完全满足LLM无后验信息假设，因而仅能视为复现结果；③ 由于e‑process的无偏性假设依赖于提交后不泄露信息，若代理在历史回放中已训练于相同时间段，则可能失效；④ 认证组合的收益受每日IC不足导致的等待时间影响，未能针对更长持有周期的因子进行优化。

---

## 94. Diverse by Design: Architectural Constraints for Prototype-Based Interpretability

**arXiv ID:** 2609.27194 | [PDF](https://arxiv.org/pdf/2609.27194v1)

**作者:** Xinmiao Lin `[一作]` (Rochester Institute of Technology), Matthew Wright `[通讯]` (Rochester Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了DAPL方法，利用多头自注意力与严格的一对一注意力-原型映射以及前景感知训练，解决原型冗余并提升可解释性；同时提出Coverage与Diversity两项定量评估指标。

**💡 创新点**

创新点在于通过架构约束（多头注意力与一对一映射）实现原型多样性，辅以对比正则化和前景感知训练；并首次给出覆盖率和多样性两种可解释性定量评估。

**🔧 技术方法**

采用多头自注意力、严格一对一注意力‑原型映射、对比正则化、前景感知损失以及新的Coverage/Diversity评估方法。

**📊 数据集**

主要使用CUB‑200‑2011鸟类细粒度数据集（在附录中亦测试FaceForensics++）。

**📈 对比分析**

与ProtoPNet、ProtoPShare、PixPNet等基线对比，DAPL+FG在CUB‑200‑2011上达到81.69%准确率，Coverage 0.596，Diversity 0.427，综合表现优于现有原型网络。

**⚠️ 局限性**

局限性包括对背景信息平衡的敏感性、对多头注意力参数与计算开销的依赖，以及指标需依赖部件标注，跨域泛化与大规模场景验证仍待进一步研究。

---

## 95. Surgical Kinematics from Monocular Video with Learned Articulated Motion Constraints

**arXiv ID:** 2609.27227 | [PDF](https://arxiv.org/pdf/2609.27227v1)

**作者:** Mehmet Kerem Turkcan `[一作]` (Columbia University), Zoran Kostic `[通讯]` (Columbia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于单目视频的机器人手术器械运动学重建网络，能够同时估计位置、姿态和咬合角，并通过可微分轨迹重建保持运动一致性。

**💡 创新点**

创新点包括：①将冻结的 DINOv3 视觉特征与 SAM 3.1 掩码定位的局部特征结合；②引入相机深度信息（EndoSynth）和视觉状态先验；③将位移拆分为幅值和方向预测，并用加权最小二乘法与四元数变换实现全局轨迹优化；④在 Transformer 编码器基础上加入局部时序卷积和运动增量约束，提升重建精度。

**🔧 技术方法**

使用的技术主要有：Transformer 编码器、局部时序卷积头、SAM 3.1 细化掩码、DINOv3 视觉特征、EndoSynth 深度网络、可微分加权最小二乘轨迹重建、四元数变换、Jaw 和姿态的 Huber 损失以及多尺度路径长度监督。

**📊 数据集**

训练与评估数据集为公开的 Open‑H 基准（包含 Main endoscopy、CAO、Hamlyn 三个子集共 2,802 组影片和 746,001 帧），并在 70 条额外的 JHU 记录上进行外部验证；深度模型在 Open‑H 渲染数据上微调。

**📈 对比分析**

与 MS‑TCN、PatchTST、Transformer 回归以及 LiveMAE 等基线对比，本文方法在所有五项运动学指标（位置、姿态、咬合角、路径长度、刚体误差）和运动分析指标（log dimensionless jerk、运动分割 mAP）上均取得最佳成绩；路径长度 MAE 仅 0.3392 cm，mAP 达 54.44%。

**⚠️ 局限性**

局限性包括：依赖高质量的 SAM 掩码与深度估计，难以处理严重遮挡或非手术视频；使用冻结的视觉特征在不同光照或器械种类下可能泛化不足；轨迹优化对时序连续性有一定要求，断层或长间隔帧可能导致误差累积。

---

## 96. Learning Dissipative Dynamics with Dissipativity-by-Construction Discrete-Time Neural Networks

**arXiv ID:** 2609.27188 | [PDF](https://arxiv.org/pdf/2609.27188v1)

**作者:** Tuan Luong `[一作]` (Sungkyunkwan University), Hyungpil Moon `[通讯]` (Sungkyunkwan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种在离散时间下，通过结构化参数化和专门训练过程，使得多层感知机（MLP）模型天然满足增量耗散性的离散时间动力学学习方法。

**💡 创新点**

创新点在于：①将增量耗散性约束直接嵌入网络结构，而非后置正则化或投影；②通过对权重矩阵的构造和可调的正则项，保证整个训练过程不需要迭代LMI求解；③在深层网络中引入身份连接与门控机制，以缓解梯度消失问题。

**🔧 技术方法**

使用的技术包括：离散时间多层感知机（DMLP）、激活函数的斜率限制、存储函数与供给率理论、矩阵不等式（LMI）构造与Schur补、Eigenvalue分解确定超参数、梯度下降优化、与神经ODE的对比实验。

**📊 数据集**

数据集：利用仿真得到的质量-弹簧-阻尼（MSD）系统和 n‑DOF 机械臂系统的输入–输出时序数据，分别在无噪声和加高斯噪声的训练场景下进行测试。

**📈 对比分析**

与标准无耗散约束的MLP以及连续时间神经ODE（Dissipative Neural ODE）进行比较。结果显示：DMLP 在预测误差上与传统MLP相当甚至更优，在增量耗散性保持上显著优于MLP；与神经ODE相比，DMLP 在三种系统上的均方误差差距不大，却将计算时间从数小时压缩到几百秒，显著提升了效率。

**⚠️ 局限性**

局限性包括：①仅验证了离散时间多层感知机，未覆盖更复杂网络结构或更高维系统；②对权重矩阵的构造导致每次迭代需要O((Lq)^3)的矩阵运算，虽然比神经ODE高效，但在极深网络下仍可能成为瓶颈；③实验数据来自仿真，缺乏真实物理系统的验证。

---

## 97. Pose-Aware Multimodal Automatic Tagging for Greek Traditional Music

**arXiv ID:** 2609.27094 | [PDF](https://arxiv.org/pdf/2609.27094v1)

**作者:** Alexandros Alexiou `[一作]` (National Technical University Of Athens), Alexandros Potamianos `[通讯]` (National Technical University Of Athens)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了希腊传统音乐自动标签任务中舞蹈姿态的补充作用，并构建了包含音频、视频和姿态的多模态标签框架。

**💡 创新点**

创新点在于提出了姿态感知的多模态框架，设计了针对现场广播视频的姿态提取管道，并系统评估了姿态与音频视频的融合效果。

**🔧 技术方法**

技术上使用了Audio Spectrogram Transformer（AST）做音频编码，SlowFast/TimeSformer做视频编码，ST‑GCN做姿态编码；并在多模态融合中比较了Late、Early、Gated、Cross‑Attention等策略。

**📊 数据集**

使用的公开数据集是 Lyra（1570 条希腊传统音乐视频，749 条含舞蹈姿态），并采用其官方训练/验证/测试划分。

**📈 对比分析**

通过宏观 ROC‑AUC、PR‑AUC 与 F1 进行评估，发现最佳的 A+V+S Gated 融合可将宏观 ROC‑AUC 提升至 0.864（比音频单模 0.821 高约 4%），姿态单模仅 0.586。

**⚠️ 局限性**

限制在于数据量有限、姿态覆盖率仅约 50%（受遮挡、多人物影响），姿态单模表现弱，且对窗口长度敏感，影响整体性能提升。

---

## 98. Damnatio Memoriae: Adversarially and Selectively Forgetting Identities in the Embedding Space of Face Recognition Models

**arXiv ID:** 2609.27115 | [PDF](https://arxiv.org/pdf/2609.27115v1)

**作者:** Ünsal Öztürk `[一作]` (Idiap Research Institute), Sébastien Marcel `[通讯]` (Idiap Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在面部识别模型中实现对选定身份的对抗性遗忘，即在跨拍摄实例下使这些身份不可被识别，同时保持其余身份的识别性能；

**💡 创新点**

创新点在于提出了弱遗忘（仅把同一身份跨拍摄对的相似度压至阈值以下）与强遗忘（将同一身份跨拍摄对的相似度分布移至与非同一身份相似度分布一致）的区分，并设计了三种针对嵌入空间的遗忘损失——质心推送、扩展CosFace以及正交框架；

**🔧 技术方法**

技术方法基于CosFace等Margin loss的面部识别模型，在此基础上加入遗忘损失进行细调；使用对比损失、Wasserstein距离等统计量评估遗忘效果，并在多种backbone（iResNet‑34、ViT‑S、ViT‑B）上实验；

**📊 数据集**

使用WebFace4M数据集（约420万张图像，20.6万身份），将其中10万个身份作为保留集，其余身份分为遗忘集、测试集等；

**📈 对比分析**

与四个基线（pairwise dispersion、hard dispersion、NegGrad+、CURE）在两种遗忘规模（1%和5%）以及三种backbone下进行比较。正交框架方法实现了强遗忘，且在训练集中的同一身份跨拍摄对的相似度分布几乎与非同一身份一致；其他方法（质心推送、扩展CosFace、硬散列）主要实现弱遗忘；不同方法对保留身份的影响及其“足迹”大小差异明显，硬散列在保持保留身份性能方面成本最低；

**⚠️ 局限性**

实验仅在CosFace Margin loss、WebFace4M 100k身份子集上验证，未检验更大规模或不同损失/数据集；对抗性遗忘需要在训练期间处理遗忘样本，对已存在的模板必须删除或重新登记；不同backbone下成本差异显著，部分方法在Transformer上表现较好，进一步研究所需。

---

## 99. Local Evidence and Geometric Readout Repair in Trained GNNs

**arXiv ID:** 2609.27092 | [PDF](https://arxiv.org/pdf/2609.27092v1)

**作者:** Nadi Tomeh `[一作]` (Université Sorbonne Paris Nord), Hugo Attali `[通讯]` (Université Sorbonne Paris Nord)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在已训练的图神经网络节点分类任务中，提出了两种后置修复方法——基于精确质量的重新加权和基于集合条件的中心化对齐翻译，用以纠正读出层错误。

**💡 创新点**

创新在于：①使用精确质量线性规划作为oracle，判断固定消息能否支持正确类别；②设计两种受限于已冻结消息的修复器，区分仅可重权与可自由平移的区别；③对比基于节点信息与全局消息集的翻译，揭示消息集对改正方向的有限提升。

**🔧 技术方法**

技术包括：图神经网络（GCN、GAT、SAGE、GATv2、GraphConv、SGC、MixHop、H2GCN）预训练；精确质量线性规划求解；Deep Sets编码器实现的可微重权器和翻译器；交叉熵训练，label-free 方式。

**📊 数据集**

实验涵盖八个真实数据集：Actor、Amazon、Chameleon、Citeseer、Cora、PubMed、Roman Empire、Squirrel，共八种GNN骨干，十个交叉验证拆分。

**📈 对比分析**

与原始冻结模型、线性头重新训练、节点级翻译等方法对比，平均宏观准确率从62.6%提升到65.3%，其中翻译贡献最大（+2.7%），再加上消息集条件可再提升约+0.63%，相比传统重权仅+1.2%。

**⚠️ 局限性**

局限性：①精确质量oracle仅评估可重权范围，未能利用图结构外的证据；②label-free 重权器对目标类推断能力有限，修正效果低于oracle；③自由翻译虽性能更好，但在部分图（如引用网络）提升有限；④未考虑更大结构化松弛或多步传播的改进。

---

## 100. When Direct Manipulation Becomes a Guess: Productive Friction in AI-Mediated Multisensory Visualization

**arXiv ID:** 2609.27104 | [PDF](https://arxiv.org/pdf/2609.27104v1)

**作者:** Anchit Mishra `[一作]` `[通讯]` (University of Waterloo), Anchit Mishra (University of Waterloo)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文探讨了AI介导的多感官可视化中直接操作失效的机制，并提出了四种摩擦（可预测性、局部性、可逆性、来源属性）来维护用户直觉与系统透明度。

**💡 创新点**

创新点在于将直接操作的四项核心保证与AI介导下的四种摩擦对应，并借鉴科幻电影中的界面设计，形成一套解释与设计原则框架。

**🔧 技术方法**

采用理论分析与案例研究方法，对现有生成可视化系统（如DynaVis、Data Formulator、InterChat）进行对比，提出摩擦机制。

**📊 数据集**

未使用传统实验数据集，主要以电影镜头（《铁人2》与《Blade Runner 2049》）作为视觉案例。

**📈 对比分析**

未进行量化实验或性能评估，仅通过概念对比和设计原则讨论，缺乏实测数据支持。

**⚠️ 局限性**

限制在于缺乏实证研究与用户测试，理论性强，实际实现难度与效果尚未验证。

---

## 101. FINN-Tro: Exploiting Verification Gaps in Dataflow Inference Accelerators

**arXiv ID:** 2609.26824 | [PDF](https://arxiv.org/pdf/2609.26824v1)

**作者:** Qazi Arbab Ahmed `[一作]` (Bielefeld University of Applied Sciences and Arts), Thorsten Jungeblut `[通讯]` (Bielefeld University of Applied Sciences and Arts)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 FINN 编译流水线的最后一个 MVAU 层中插入硬件木马，实现对推理结果的有条件篡改；

**💡 创新点**

揭示了 FINN 现有验证流程的时间触发漏洞，并首次在 FPGA DNN 加速器上展示了可隐藏的、可激活的木马技术；

**🔧 技术方法**

利用 FINN 的层专化与 HLS 代码生成阶段插入宏定义实现计数器触发和偏置/交换负载；

**📊 数据集**

使用 MNIST（全连接网络）和 CIFAR‑10（12 层 CNN）数据集进行实验；

**📈 对比分析**

与无木马基线相比，吞吐量和运行时几乎不变，准确率从 92.96%/84.19% 降至 10%/10%（按不同触发/负载组合），硬件资源占用仅提升 2–7%；

**⚠️ 局限性**

局限在仅针对最终 MVAU 层、缺乏输入条件触发、仅验证了两种网络结构，且对更深、更复杂模型的适用性未评估。

---

## 102. Small Cues, Big Consequences: Learning Pivotal Cues for Multimodal Meme Classification

**arXiv ID:** 2609.26907 | [PDF](https://arxiv.org/pdf/2609.26907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 103. PR-Smoother: Simulator-Preserving Non-Gaussian Smoothing for Data Assimilation

**arXiv ID:** 2609.26890 | [PDF](https://arxiv.org/pdf/2609.26890v1)

**作者:** Yuta Tarumi `[一作]` `[通讯]` (RIKEN), Yuta Tarumi (RIKEN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种物理仿真器保持的非高斯平滑器PR-Smoother，用于仅基于观测窗口进行的状态、参数与偏置联合估计。

**💡 创新点**

创新点在于将先验物理模型保持在目标函数和变分族内，仅学习围绕仿真器递推的未来条件修正，从而兼具高维可扩展性和多模态后验表达能力。

**🔧 技术方法**

采用条件正则化流（conditional NF）建模初始状态后验，并使用正态分布的未来条件修正作为转移核，优化观测仅的ELBO并实现对模型参数的最大似然估计。

**📊 数据集**

在三组实验中验证：4维Lorenz‑96、40维Lorenz‑96、以及高维16,384维Kolmogorov流。

**📈 对比分析**

与EnKF、ETKF、4D‑Var、IEnKS、粒子滤波以及基于均值场的VI基线比较，PR‑Smoother在多模态、非线性观测以及高维参数学习任务中取得了优异或相当的误差/能量距离、RMSE等指标，尤其在稀疏观测与联合参数学习场景中表现突出。

**⚠️ 局限性**

局限包括对可微仿真器的依赖、潜在的近似误差（近似VI的 amortization gap）、在复杂参数空间下收敛速度慢以及缺乏对参数不确定性的建模。

---

## 104. WTF?! Simulation-Free Reinforcement Learning with Wasserstein-Tilted Flow Maps

**arXiv ID:** 2609.27033 | [PDF](https://arxiv.org/pdf/2609.27033v1)

**作者:** Abbas Mammadov `[一作]` (University of Oxford), Nicholas M. Boffi `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于Wasserstein正则化的奖励微调方法，直接在预训练的流图上进行端到端微调，以提升生成样本的下游奖励。

**💡 创新点**

创新点在于用预训练漂移构建最优传输正则化，将奖励对齐问题转化为确定性最优控制，而非传统的KL奖励倾斜，并利用流图实现无模拟、无critic 的价值估计。

**🔧 技术方法**

技术上使用流模型与流图、最优传输理论、确定性最优控制、Monte Carlo 价值估计以及自监督蒸馏来实现高效微调。

**📊 数据集**

实验数据集包括 ImageNet-256（图像生成）和 TiM‑T2I（文本到图像生成），奖励函数采用 HPSv2 以及其他评估指标如 PickScore、ImageReward。

**📈 对比分析**

与 KL 正则化、Adjoint Matching、MFM、VFM、Flow‑GRPO 等基线相比，WTF 在奖励和多样性上均表现更佳，且训练成本可低至 280 倍、47 倍。

**⚠️ 局限性**

局限性包括需要先行的流图预训练、奖励规模会影响多样性、使用欧几里得奖励梯度非最优，以及对高维度多样性仍存在挑战。

---

## 105. NADI 2026: The Second Multidialectal Arabic Speech Processing Shared Task

**arXiv ID:** 2609.27086 | [PDF](https://arxiv.org/pdf/2609.27086v1)

**作者:** Peter Sullivan `[一作]` (University of British Columbia), Muhammad Abdul-Mageed `[通讯]` (University of British Columbia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织并评估了NADI 2026多方言阿拉伯语语音处理共享任务，涵盖ASR、SDID、TTS、SLT和SLU五大任务及其子任务；

**💡 创新点**

首次将TTS、SLT、SLU纳入系列，强调低带宽、混合方言、代码切换等现实部署场景，并提出对方言识别的零样本、跨域评估；

**🔧 技术方法**

使用多模态、基于Transformer的自监督模型（如Cohere、Ara‑BEST‑RQ、Whisper、OmniVoice），以及LoRA微调、ROVER组合、语言模型拼写纠错等技术；

**📊 数据集**

利用多源数据集：Casablanca、MGB2、QASR、MASC、ADI‑20、Bulbul、WhiteHouse、SLURP‑TN 等，配合新收集的低带宽、混合方言和代码切换语料；

**📈 对比分析**

通过与基线（Off‑the‑shelf ASR/SLT、ECAPA‑TDNN SDID、XTTS‑v2 TTS）对比，系统平均 WER 下降约 30%，但跨域识别准确率仅提升至 56%；

**⚠️ 局限性**

限制包括方言覆盖不足（仅 10 种）、TTS 低参赛率和缺乏人工评估、字形变异导致 WER 低估、以及高算力需求的集成方法

---

## 106. Combining LLMs and Genetic Search for ARC-AGI-2

**arXiv ID:** 2609.27242 | [PDF](https://arxiv.org/pdf/2609.27242v1)

**作者:** Val Dyachenko `[一作]` `[通讯]`, Val Dyachenko

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

结合LLM生成的初始程序与遗传算法搜索，在ARC-AGI-2任务上实现程序优化与新解的发现。

**💡 创新点**

首次提出LLM仅用于一次种子生成后完全交给遗传算法搜索的框架，并设计了专用DSL保证程序合法性。

**🔧 技术方法**

使用量化Qwen3.5-4B生成程序，基于自定义DSL的GPU并行执行器和遗传算法进行多代搜索。

**📊 数据集**

基准为ARC-AGI-2公开评测集的前60个任务。

**📈 对比分析**

与仅使用LLM程序相比，遗传算法在60任务中从2/60提升至6/60，显示10%解题率提升；对比随机种子搜索无解。

**⚠️ 局限性**

受限于DSL表达能力、感知错误、搜索时间限制及实验的随机性，未覆盖全部任务且无法精确量化LLM贡献。

---

## 107. Signal2Symbol: Neuro-Symbolic Temporal Reasoning for Explainable Physiological Time-Series Anomaly Detection

**arXiv ID:** 2609.26820 | [PDF](https://arxiv.org/pdf/2609.26820v1)

**作者:** Naser Mansour `[一作]` (New York University), Ameer Rahwan `[通讯]` (Koc University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Signal2Symbol神经符号框架，利用VQ‑VAE或SAX将生理信号离散化为符号序列，并通过稀有项集挖掘、Allen间隔代数推理和FCA概念格压缩，实现可解释的ECG/EEG异常检测与解释

**💡 创新点**

将神经网络生成的符号（VQ‑VAE码本）与传统符号化（SAX）结合，构建大纲丰富的事务并使用最小稀有项集进行异常打分；随后将局部检测转化为时间间隔，利用Allen关系生成临时关系图，再用FCA形成稀有时间概念格，实现多层次可解释性和压缩

**🔧 技术方法**

VQ‑VAE编码/解码、SAX符号化、bigram增强事务、稀有项集挖掘（BtB算法）、稀有项集权重打分、Allen间隔代数、图构造与关系匹配、Formal Concept Analysis（FCA）构造Galois格

**📊 数据集**

MIT‑BIH Arrhythmia（beat‑level ECG）、PTB‑XL（record‑level ECG）以及Bonn EEG（segment‑level EEG）三个公开数据集

**📈 对比分析**

与IF、LOF、USAD、SAX+Rare等基线比较；在所有数据集上Signal2Symbol（VQ‑VAE+Rare）在AUROC/AUPRC均优于基线；在鲁棒性测试中VQ‑VAE在噪声和基线漂移下保持稳定，且概念格压缩比达50‑70×，显著降低解释成本

**⚠️ 局限性**

仅利用bigram恢复序列信息，无法捕获更长的顺序模式；概念格若属性过多可能变大；VQ‑VAE易出现死码；阈值交互影响大，需谨慎调参；区间提取误差可能导致关系错误

---

## 108. Learning Spectral Allocation: A Fractional Diffusion Framework for Adaptive Volumetric Segmentation

**arXiv ID:** 2609.27217 | [PDF](https://arxiv.org/pdf/2609.27217v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 109. The Computational Value of Sensory-Aligned Receptive Fields Depends on Neuronal Expressivity

**arXiv ID:** 2609.26940 | [PDF](https://arxiv.org/pdf/2609.26940v1)

**作者:** Agnese Adorante `[一作]` (University of Tübingen), Anna Levina `[通讯]` (University of Tübingen)

**通讯引用:** 2006 | [OpenAlex ID](https://openalex.org/A5050318092)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文使用可变表达的 Leaky Memory 神经网络，在音频（SHD）和事件视觉（DVS‑Gesture、CIFAR10‑DVS）任务中比较结构化与随机输入连线，研究感官受体场结构对计算性能的影响。

**💡 创新点**

创新点在于明确表明受体场结构与任务相关感官几何相结合能提供计算先验，并证明其优势随单元表达性提高而减弱，揭示感官几何与神经元复杂度之间的权衡。

**🔧 技术方法**

采用 Expressive Leaky Memory (ELM) 网络、稀疏 (ℓ1) 正则化、感官坐标结构化输入、完全输入对照实验以及统计显著性检验。

**📊 数据集**

使用的数据集为：SHD（语音数字电流脉冲）、DVS‑Gesture（运动方向事件视觉手势）和 CIFAR10‑DVS（空间位置事件视觉物体）。

**📈 对比分析**

通过10个随机种子训练并用 Welch t‑检验比较结构化与随机输入的测试准确率；结构化输入在所有网络尺寸下均提高准确率，优势随记忆单元数增加而减弱；完全输入网络表现差于结构化，ℓ1 正则化可部分恢复但未完全弥补。

**⚠️ 局限性**

限制在于仅研究少数几种感官几何与任务，受体场结构仅为抽象模型，未探讨不同架构或更高表达性单元下的普适性。

---

## 110. Verified Learning for Compiler Optimization: An LLM-Guided Architecture with Formal Control

**arXiv ID:** 2609.27214 | [PDF](https://arxiv.org/pdf/2609.27214v1)

**作者:** Dev Pratap Singh `[一作]` (Pennsylvania State University), Suman Saha `[通讯]` (Pennsylvania State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个验证中心化的编译器优化架构，利用大语言模型生成LLVM IR的lazification重写，并通过Alive2形式验证保证语义正确性；在推理过程中使用验证反馈循环自动纠正不合法重写。

**💡 创新点**

①将语义正确性从模型学习中剥离，交给外部验证器作为最终决策权；②将Alive2嵌入生成-验证循环，使模型在生成过程中获得结构化纠错信号；③提供了可复现的lazification训练与评测数据集，展示学习模型能在不破坏语义的前提下复制手工优化器Wyvern的行为。

**🔧 技术方法**

Code LLaMA（预训练的LLVM IR/汇编生成模型）+ LoRA微调；Alive2 SMT等价性检查器；LLVM/Clang编译链；符号验证与重写再生成反馈循环。

**📊 数据集**

八个开源C库生成的LLVM IR及其Wyvern优化对（已公开于Figshare，包含完整微调代码和实验数据）。

**📈 对比分析**

与未微调模型、Wyvern插件以及默认LLVM进行对比；在LLVM测试套件上测量运行时间、二进制大小和等价性；结果显示微调模型在约9.8%基准上可与Wyvern媲美或提升，整体没有语义违规，且对未改动的函数保持保守；性能提升与二进制尺寸下降均在可接受范围内，且验证开销保持稳定。

**⚠️ 局限性**

仅针对C/LLVM，难以推广到其他语言/IR；Alive2不考虑微架构级别的性能差异；验证开销影响可扩展性；模型受Wyvern监督导致偏向保守，可能无法发掘全新优化策略。

---

## 111. HINT-Blimp: Human INTent Inference from Multimodal Cues for Robotic Blimps

**arXiv ID:** 2609.27154 | [PDF](https://arxiv.org/pdf/2609.27154v1)

**作者:** Subhadeep Koley `[一作]` (Lehigh University), David Saldaña `[通讯]` (Lehigh University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种多模态人机交互框架，利用粒子滤波在线推断人类意图，将意图建模为参数化线性动力系统（LDS），通过物理推送和语音指令等稀疏信号逐步识别目标并生成平滑轨迹；

**💡 创新点**

创新点在于：①将高层意图表述为LDS参数；②使用粒子滤波对LDS参数进行在线估计；③通过特征化的特征值参数化保证系统稳定并支持曲线路径；④将物理推送与语音/手势作为互补证据在同一贝叶斯框架内融合；

**🔧 技术方法**

核心技术包括粒子滤波、LDS建模、特征值参数化、物理推送和方向指令的方向相似性/角度差似然计算；

**📊 数据集**

实验数据集为人类参与者（5人）在实验室场景中完成的300次试验（5人×60次），包含三类实验（目标识别、障碍物绕行、先验敏感性）和模拟验证；

**📈 对比分析**

与单一模态（仅推送、仅语音）对比，结合模态成功率提升至86%，在两次交互内完成目标识别的比例最高；曲线路径实验展示了LDS推断的曲线避障能力；先验实验表明算法对初始分布鲁棒，恶劣先验仅需约1.9次交互即可纠正；

**⚠️ 局限性**

局限性包括：样本规模有限（仅5人）；实验环境简化，缺乏复杂障碍与动态环境；仅评估推送与语音指令，未充分验证手势或自然语言的效果；粒子滤波对计算量敏感，实时性能受限；

---

## 112. ContraVis: Evidence-Grounded Visual Analytics for Contradiction Review in Legal Contracts

**arXiv ID:** 2609.27014 | [PDF](https://arxiv.org/pdf/2609.27014v1)

**作者:** Luis Sante `[一作]` (Fundaçāo Getulio Vargas), Jorge Poco `[通讯]` (Fundaçāo Getulio Vargas)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套基于类型化段落图的可视化分析系统，用于人机协同的法律合同矛盾检测与评估。

**💡 创新点**

创新点在于将段落图同时作为LLM推理的上下文和可视化交互的基础，保证模型与人工审阅同步；以及基于图条件的矛盾生成与检索框架。

**🔧 技术方法**

使用大型语言模型（LLM）结合语义相似度检索、图结构条件推理、可视化三视图（文档、矛盾分析、邻居探索）与对话式解释。

**📊 数据集**

使用CUAD合同语料库生成合成矛盾案例，并在实验中使用来自CUAD的原始合同。

**📈 对比分析**

通过对比图条件推理与仅文本提示的LLM检测，发现图条件方法在长合同中恢复更多注入的矛盾（3–4/5）且能发现更多候选；性能提升明显。

**⚠️ 局限性**

局限包括依赖人工标注的合成矛盾、模型置信度无校准、图结构对段落划分敏感，以及实验规模和参与者样本有限。

---

## 113. ACTS: A multi-tier benchmark evaluating LLM cipher identification under controlled blind conditions

**arXiv ID:** 2609.26893 | [PDF](https://arxiv.org/pdf/2609.26893v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 114. Agentic-IC3: Enabling Semantic Proof Search in IC3 Model Checking

**arXiv ID:** 2609.27162 | [PDF](https://arxiv.org/pdf/2609.27162v1)

**作者:** Yu-Wei Fan `[一作]` (Princeton University), Sharad Malik `[通讯]` (Princeton University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Agentic-IC3框架，在IC3模型检查中集成语言模型代理，利用RTL信息指导词级泛化、观察引入和回溯，提升硬件安全与功能验证效率。

**💡 创新点**

创新性在于将大型语言模型与IC3证明搜索耦合，提供批量泛化、观察词和回溯动作，并保持持久证明会话，突破传统位级泛化对设计语义利用不足的局限。

**🔧 技术方法**

技术包括IC3算法、词级模型检查、Yosys、Bitwuzla SMT求解器、Codex大语言模型、批量泛化接口、观察词定义、回溯机制与JSON接口。

**📊 数据集**

使用14个基准，涵盖信息流安全、FIFO、算术循环、仲裁、处理器验证（NERV、PicoRV32、SERV）和BSG通信协议缓冲区溢出等。

**📈 对比分析**

与Pono-IC3Bits、rIC3和A-IC3基线比较，Agentic-IC3在一小时内解决10个基准（包括4个其余基线无法解决的），总体运行时间较某些基线慢但完成度更高，代理响应占总时间约87%。

**⚠️ 局限性**

局限包括高代理响应延迟导致性能瓶颈、缺少量化和时间序列观察、对全局证明进度缺乏全局反思、以及当前仅支持无量化词级符号，未来需优化模型和扩展证明动作。

---

## 115. Classifying Interpretive Canons at the Sentence Level: A Benchmark from the German Federal Constitutional Court

**arXiv ID:** 2609.26945 | [PDF](https://arxiv.org/pdf/2609.26945v1)

**作者:** Felix Ringe `[一作]` `[通讯]` (Freie Universität Berlin), Felix Ringe (Freie Universität Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个句子级别的基准，用来评估大型语言模型（LLM）在德国联邦宪法法院判决文本中识别和分类解释学四大传统判读范本（语法、系统、历史、目的-目标）的方法。

**💡 创新点**

创新点在于：① 将Larenz所述的四大判读范本正式化为可操作的分类标准；② 以专家标注的方式在句子层面构建了德国联邦宪法法院的判决数据集；③ 通过手工编写与基于遗传-帕累托算法（GEPA）优化的提示，对四款LLM进行基线评估，验证手工提示的有效性。

**🔧 技术方法**

采用的技术包括：大型预训练语言模型（DeepSeek‑V4、Gemini 3.5 Flash Lite、MiniMax M3），提示工程（手工提示+GEPA自动优化），句子边界检测（DistilBERT微调），以及基于约束解码的结构化输出；并构建了多阶段评估管道（阅读判断→论证门→四大范本判定）。

**📊 数据集**

使用的数据集为28份德国联邦宪法法院判决的专家级句子标注集，其中15份判决全部标注，13份针对稀有类别进行选择性标注；数据集包含阅读、论证门、四大范本四类标签以及法条引用子任务，已在GitHub、HuggingFace和WandB公开。

**📈 对比分析**

比较方法：在七个二分类子任务上统计正类F1，法条引用子任务采用文档级集合重叠F1；对比手工提示与GEPA优化提示的表现。性能结果显示，四款模型的平均二分类F1在70.4–79.2之间；手工提示与GEPA提示在整体表现上相近，GEPA仅在部分模型略有提升；语法范本F1最高（≈83），系统范本最低（≈58）；法条引用在集合重叠上得分较高，但单句跨度F1仅28–42，说明任务本身易产生过度提取。

**⚠️ 局限性**

局限性包括：标注仅由单一法律博士完成，缺乏互评一致性；数据集规模有限，覆盖范围仅限德国联邦宪法法院，难以推广至其他司法体系；基线评估仅涵盖四款LLM，未覆盖更强大闭源模型；测试集样本少导致置信区间宽阔；法条引用子任务易出现高误报，反映任务设计上的难点。

---

## 116. Distilling Sequential Computation in Transformer Language Models

**arXiv ID:** 2609.27233 | [PDF](https://arxiv.org/pdf/2609.27233v1)

**作者:** Zixuan Lan `[一作]` (University of Chicago), Jiawei Zhou `[通讯]` (Stony Brook University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 UMIM 框架，使用轻量级 merge 模块在推理时将多 token 跨度压缩为单个 surrogate embedding，保持预训练 Transformer 模型结构和参数不变。

**💡 创新点**

创新点在于：① 通过频繁 n‑gram 规则构造合并集；② 训练单层多头注意力池化的 merge 模块以对齐原始模型的预测分布；③ 在推理时进行 runtime 合并并 rollback KV 缓存，既压缩上下文又不影响生成；④ 兼容多种下游任务，支持任务特定微调。

**🔧 技术方法**

主要技术包括：单层多头注意力池化 merge 模块、KL 对齐损失、KV 缓存 rollback、runtime 合并策略、分布对齐与 DPO 微调。

**📊 数据集**

数据集：WikiText‑103 用于训练 merge 模块与合并规则；BookCorpus、OpenWebText 用于无监督评估；PIQA、COPa、OpenBookQA、ARC、CNN/DailyMail、AIME 等用于下游任务评测。

**📈 对比分析**

与 SelectContext、LLMLingua2、H2O、StreamingLLM 等基线比较，压缩率可达 35–54% 仍保持与原模型相近的 PPL、QA 准确率、ROUGE 等指标；在推理阶段提升吞吐量；任务特定微调后可进一步提升准确率并保持高压缩率。

**⚠️ 局限性**

局限性：依赖频繁 n‑gram 规则，覆盖率有限；过度合并会导致性能下降；merge 模块参数规模有限，适配极端长文本或更大模型时效果需验证；未探索更自适应或多级压缩策略。

---

## 117. A 3D Pose-Based Ensemble Framework for Cricket Shot Classification and Automated Biomechanical Analysis

**arXiv ID:** 2609.26923 | [PDF](https://arxiv.org/pdf/2609.26923v1)

**作者:** Sourav Shome `[一作]` (Khulna University), Rameswar Debnath `[通讯]` (Khulna University)

**通讯引用:** 719 | [OpenAlex ID](https://openalex.org/A5087025013)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套基于3D姿态的多模型集成框架，用于板球击球动作的分类与自动生物力学分析，并为新手和教练提供实时姿态反馈。

**💡 创新点**

创新之处在于首次将MeTRAbs 3D姿态估计与LRCN‑TCN深度集成相结合，既提升了分类精度，又能直观比较不同击球动作的关节角度差异。

**🔧 技术方法**

使用YOLO检测击球手、MeTRAbs生成30点SMPL3D骨架、传统机器学习（SVM、RF、XGBoost）与深度网络（LRCN、TCN、BiLSTM）进行时间序列建模，最终通过概率平均实现集成。

**📊 数据集**

基于KUCricShot数据集（1,250段含四种击球类型）进行实验，采用30维关节坐标的3D序列作为输入。

**📈 对比分析**

采用5折交叉验证对单一模型与集成模型进行比较，集成模型在所有四类击球上平均准确率达97.68%±0.99%，显著优于单一模型以及之前2D姿态或原始视频方法的表现。

**⚠️ 局限性**

局限性包括样本量有限、仅覆盖四种击球动作、需要手工裁剪击球片段、未实现自动击球边界检测，且未使用合成数据来扩充训练集。

---

## 118. SsgCaps: A controlled dataset for the evaluation of sound scene generation algorithms

**arXiv ID:** 2609.26854 | [PDF](https://arxiv.org/pdf/2609.26854v1)

**作者:** Modan Tailleur `[一作]`, Yuki Okamoto `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了公开版的SSGcaps数据集，替换了原先包含私有音频的版本，并提供完整的提示与音频配对。

**💡 创新点**

创新点包括：①采用固定语义层级的提示结构实现可控、可解释的声景生成；②将私有音频完全替换为Freesound公开资源，保证数据可复现性；③通过主观评估与多种客观距离指标验证新版本与原版的可替代性。

**🔧 技术方法**

使用了Fréchet Audio Distance (FAD) 与 Kernel Audio Distance (KAD) 两种音频距离度量，基于PANNs embedding进行特征提取；同时使用了主观评分体系（前景、背景、质量三项）。

**📊 数据集**

使用的数据集为SSGcaps（310个提示，4秒 16位 mono，32kHz 语音/音乐除外），全部音频来源于Freesound公开库，按前景类别（Animal, Vehicle, Human, Alarm, Tool, Entrance）与背景类别（Crowd, Traffic, Water, Birds, Presence）划分。

**📈 对比分析**

评估方法：对公开版与原版数据集分别计算与18个SGB系统生成音频的FAD和KAD，并与主观评分做相关性分析。结果表明两版数据集在客观指标和主观评估上的表现高度一致，公开版可直接用于后续基准测试。

**⚠️ 局限性**

局限性：虽然整体一致，但替换后的音频在某些极难模拟的样本上略显差异；数据集不包含音乐或可辨识语音；背景样本种类有限，未来可扩展更多场景与声音种类。

---

## 119. From greenhouse climate to individual leaves: an organ-resolved model of lettuce growth

**arXiv ID:** 2609.27118 | [PDF](https://arxiv.org/pdf/2609.27118v1)

**作者:** Md Hasibur Rahman `[一作]` (Auburn University), Tanzeel U. Rehman `[通讯]` (Auburn University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个叶片级别的功能结构植物模型，将温室气候、叶片光合、碳平衡与三维植株结构耦合，并在NVIDIA Isaac Sim中实现并验证；

**💡 创新点**

将叶片光照、环境响应、碳分配和三维几何生成统一到单一框架，实现叶片级别的光辐射计算与生长反馈，支持双向数字孪生；

**🔧 技术方法**

使用功能结构植物模型（FSPM）、光照射线追踪、碳平衡方程、NVIDIA Isaac Sim仿真、OpenUSD、GPU加速、数值积分和温室传感器数据插值等技术；

**📊 数据集**

采用Auburn大学温室实验收集的光合辐射、温度、相对湿度、CO₂浓度、叶面积、叶数、干重等数据集进行校准与评估；

**📈 对比分析**

与真实测量数据比较，使用RMSE、RRMSE和均值偏差评估，干重RRMSE为9.5%，叶数9.2%，最大叶面积13.1%，整体性能与现有全株模型相当且能捕获叶级响应；

**⚠️ 局限性**

局限在于参数固定于单一品种，未考虑养分、光照适应、叶片病害等动态因素，缺乏实时观测更新与多品种推广，边界条件需手工设定。

---

## 120. Quantifying the Occult: A Comparative Study of Hindu and Buddhist Deities Using Machine Learning Methods

**arXiv ID:** 2609.27074 | [PDF](https://arxiv.org/pdf/2609.27074v1)

**作者:** Ankit Bhattacharjee `[一作]` `[通讯]` (Indian Institute of Technology Kharagpur), Ankit Bhattacharjee (Indian Institute of Technology Kharagpur)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究构建了双矩阵计算架构，利用Gower距离与基数加权算法对196位印度教与噶举佛教的密宗神祇的形态与神学属性进行量化，并通过LLM生成的语义嵌入揭示图像伪装与“阿丁效应”。

**💡 创新点**

创新点在于：①提出基数加权算法解决传统TF‑IDF的Gestalt问题；②将离散形态矩阵与连续语义矩阵结合形成双模态模型；③通过该模型定量验证图像伪装与认知偏差的机制。

**🔧 技术方法**

主要技术包括无监督机器学习、Gower距离、基数加权、Gemini 3.1 Flash Lite LLM语义扩展、句子Transformer词向量、余弦相似度、UMAP降维及动态 egocentric 网络可视化。

**📊 数据集**

使用了公开的196节点密宗神祇数据集，包含28维混合属性（数值、类别、布尔、多标签），并配有对应的神学文本扩展。

**📈 对比分析**

通过计算形态矩阵的Gower距离与语义矩阵的余弦距离并标准化后，利用UMAP进行可视化；实验表明高基数符号能跨传统有效聚类（如 Chinnamasta 与 Chinnamunda 1:1 对应），并准确重现“阿丁效应”，验证模型在识别跨文化图像伪装方面的有效性。

**⚠️ 局限性**

局限性包括：①神学矩阵依赖LLM生成，缺乏原始梵文/藏文文本验证，存在OOV与生成随机性；②形态矩阵为离散编码，忽略艺术风格与时间演变；③数据集仅覆盖196位神祇，未覆盖地方性密宗与东南亚变体；④缺乏CNN等连续视觉特征融合。

---

## 121. Beyond Overlap: Estimating the Causal Effect of Benchmark Exposure

**arXiv ID:** 2609.27176 | [PDF](https://arxiv.org/pdf/2609.27176v1)

**作者:** Divyansh Singh `[一作]` `[通讯]` (University of Florida), Divyansh Singh (University of Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 LeakScale 框架，通过在训练后控制访问 Benchmark 专有信息，构造可执行任务来实验性测定暴露对模型评估得分的影响。

**💡 创新点**

创新点在于：① 将评估暴露量量化为可观测的因果效应而非仅凭接触痕迹；② 通过必要且不可推导的私有关联与可执行对照实验，建立了稳健的可执行反事实识别策略；③ 使用差分中的差分法结合对照组，剔除模型适应过程的整体漂移。

**🔧 技术方法**

采用可执行任务生成、私有键‑值关联、可执行对照判定、差分中的差分估计、LoRA 适配器微调、Bootstrap 置信区间等技术。

**📊 数据集**

数据集包含 2,048 个新构造的任务家族（共 64 种结构形状），覆盖 SQLite 与 Python 两种可执行域；每个家族生成 32 条随机种子；模型有 Qwen3-4B-Instruct-2507 与 Gemma 4 31B 两个家族，总计 262,144 条生成结果。

**📈 对比分析**

通过在 C0（无键）与 C2（有键）条件下分别在基线与适配后状态下测量正确率，再做差分中的差分，得出了每个模型-域组合的暴露可观测效应。效应均为正，范围从 +7.17% 到 +27.31%，平均提升约 +14.39%。

**⚠️ 局限性**

局限性包括：① 仅评估人为控制的后训练暴露，未涵盖自然 Web 规模污染；② 只针对两款模型和两种可执行域，无法直接推广到所有模型或任务；③ 依赖可执行任务的构造和私有键的保密性，若键泄露会影响实验完整性。

---

## 122. nnFoundation: 3D Foundation Models for Radiology

**arXiv ID:** 2609.26924 | [PDF](https://arxiv.org/pdf/2609.26924v1)

**作者:** Constantin Ulrich Harsy `[一作]` (German Cancer Research Center), Klaus H. Maier-Hein `[通讯]` (German Cancer Research Center)

**通讯引用:** 29837 | [OpenAlex ID](https://openalex.org/A5027292126)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出 nnFoundation，一对跨越 125 个机构、2.1M CT/MRI/PET 体积的 3D 基础模型，涵盖 CNN 与 ViT 两种架构；

**💡 创新点**

通过大规模预训练与任务感知结构、动态数据集适配三重机制，突破传统单一模型的局限；

**🔧 技术方法**

使用自监督掩码自动编码 (MAE) 预训练、nnU-Net/nnDetection 集成、动态权重适配与多任务微调；

**📊 数据集**

采用 2.1M 体积的多模态数据集，并在 108 个下游任务（分割、检测、分类、检索、报告生成）以及 160k 测试体积上评估；

**📈 对比分析**

相较于 8+ 前置基础模型和从零训练，nnFoundation 在所有任务上均显著提升（平均 Dice +3.0，AUROC +8.3，检索 AP 与报告 BLEU 也领先），CNN 在局部任务更强，ViT 在全局任务更优；

**⚠️ 局限性**

未完全解耦预训练规模、模态特定效果、算法公平性及长尾病理多样性等因素，需进一步细化和外部临床验证。

---

## 123. GeoRVQ: Decoder-aware geometry for residual-token prediction in physiological signals

**arXiv ID:** 2609.27018 | [PDF](https://arxiv.org/pdf/2609.27018v1)

**作者:** Bo Cui `[一作]` (University of Twente), Yaowen Zhang `[通讯]` (University of Twente)

**通讯引用:** 1952 | [OpenAlex ID](https://openalex.org/A5100677020)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出 GeoRVQ，一种考虑解码器成本的分层残差向量量化的掩码标记模型。

**💡 创新点**

创新点在于使用冻结解码器诱导的代价构造软目标和期望损失，并采用分层因果预测以匹配残差向量量化的粗细层次结构。

**🔧 技术方法**

技术包括残差向量量化、冻结的生理信号编解码器、解码器诱导的几何图、分层因果头、交叉熵与期望成本联合损失以及预训练与遮挡学习。

**📊 数据集**

使用 MIMIC‑IV Waveform、VitalDB 和 CODE‑15% 这三个医学信号数据集。

**📈 对比分析**

与传统交叉熵、标签平滑、欧氏几何等基线对比，GeoRVQ 在相同模型和训练条件下将精确标记准确率提升至 0.143，解码距离下降 35%（从 0.606↓到 0.393），R 峰 F1 提升至 0.837。

**⚠️ 局限性**

局限性在于解码器诱导的几何图是基于训练上下文的近似，可能忽略稀有状态下的代价；实验仅在三次独立跑中评估，统计不确定性仍存在。

---

## 124. Minimum Sum Vertex Cover via Minimum Vertex Cover

**arXiv ID:** 2609.27117 | [PDF](https://arxiv.org/pdf/2609.27117v1)

**作者:** Ahmad Biniaz `[一作]` (University of Windsor), Michiel Smid `[通讯]` (Carleton University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了最小和顶点覆盖（MSVC）问题，提出了新的近似算法、按顶点覆盖数参数化的精确算法、以及基于分离器的精确算法，并给出了平面图上的NP-难性与ETH下的下界。

**💡 创新点**

创新点包括：
- 通过最小顶点覆盖得到的简单星形分解，给出最大度 Δ≤6 时 4/3-近似；
- 对 d-正则图提出 1.184-近似算法，改进了先前 1.225 的结果；
- 设计了 2^O(k log k)+O(n+m) 的 FPT 精确算法（k 为最小顶点覆盖数）；
- 开发了利用平衡分离器的 2^O(√n log n) 精确算法，并证明其在平面图上上界与下界匹配（除对数因子）。

**🔧 技术方法**

主要技术手段包括：
- 基于最小顶点覆盖的贪心星形分解与序列最优性分析；
- Max‑k‑Vertex‑Cover 近似子程序与结构引理结合的多阶段方案；
- 右度（right‑degree）块结构与动态规划相结合的 FPT 方案；
- 递归分离器动态规划，枚举块大小与右度的整数划分；
- 组合与归约论证用于证明 NP-难性与 ETH 下的下界。

**📊 数据集**

本工作纯理论性质，不使用实验数据集；在理论分析中涉及平面图、有限基因数、H‑最小子图、树宽等图类。

**📈 对比分析**

性能评估：
- 对于最大度 ≤6 的图，算法得到 4/3‑近似，比已知的 16/9‑近似更优；
- 对 d‑正则图，1.184‑近似优于先前 1.225；
- FPT 精确算法在 k=τ(G) 时运行时间为 2^O(k log k)+O(n+m)；
- 对平面、有限基因数、H‑最小子图等图类，精确算法时间为 2^O(√n log n)，与 ETH 下的 2^o(√n) 下界相匹配。

**⚠️ 局限性**

局限性：
- 对一般图的近似比仍为 16/9，尚未突破；
- FPT 依赖于 k 的指数仍较大，实际实现可能受限；
- 对于非平面图的精确算法仍为 2^n 级，未给出更紧的下界；
- 证明中的多项式常数与实用性尚未评估。

---

## 125. When Learned Context Planning Fails to Beat Strong Retrieval: A Controlled Study of Planning, Routing, and Reranking for Long-Context QA

**arXiv ID:** 2609.26976 | [PDF](https://arxiv.org/pdf/2609.26976v1)

**作者:** Yingrui Li `[一作]` (Independent Researcher), Han Chen `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在LongBench‑v2多项选择问答中，将检索得到的候选片段交给学习得到的上下文规划器进行证据原子选择，再由回答模型推理。

**💡 创新点**

将“证据选择”转化为训练的上下文规划器，并评估其在强检索和重排序基线下的实质性增益。

**🔧 技术方法**

使用Qwen2.5‑7B‑Instruct的LoRA适配器训练规划器，检索采用BM25、密集检索与混合检索，重排序使用交叉编码器。

**📊 数据集**

实验基于LongBench‑v2 MCQ 503题（训练/开发/测试划分），并在152题未见子集上进一步评估。

**📈 对比分析**

与BM25、混合检索、预算选择器和重排序控制相比，学习规划器在全预算下仅提升约0.4–1.8分，且在大多数预算/域场景中未能超过强检索基线。

**⚠️ 局限性**

结果仅适用于LongBench‑v2 MCQ、Qwen2.5‑7B‑Instruct、特定检索与规划配置，未验证开放式QA、不同模型规模、其它检索器或真实部署环境；数据集缺乏原子级黄金标签限制了机制解释。

---

## 126. Giving Credit Where It's Due: Redundancy-Aware Learning for Efficient Reasoning

**arXiv ID:** 2609.27156 | [PDF](https://arxiv.org/pdf/2609.27156v1)

**作者:** Yuqing Zhou `[一作]` (George Mason University), Wei Niu `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为RECAP的冗余感知信用分配方法，利用语义依赖图和步骤级进展来重新分配GRPO的优势信号，以实现更高效的推理；

**💡 创新点**

创新点在于同时考虑结构责任（后续推理对步骤的依赖）和步骤效能（步骤是否推动答案正确），并通过在语义依赖图中向后传播结构责任，形成细粒度的信用分配；

**🔧 技术方法**

技术上结合了LLM生成的语义依赖图、支持/上下文/重述三类边类型、结构责任传播、步骤效能计算以及与GRPO相结合的优势重塑；

**📊 数据集**

使用了DAPO-Math-17k进行训练，评测四个数学推理基准：GSM8K、MATH-500、AIME 2024和AIME 2025；

**📈 对比分析**

与基线模型、Prompt-guided、GRPO、长度惩罚、DDCA、LEASH、BINGO、ThoughtFold等八个方法对比，RECAP在Qwen2.5-Math-7B和DeepSeek-R1-Distill-Qwen-7B模型上均实现了更低的推理token数（最高可减少31%）同时保持或提升Pass@1（最高提升3.7个百分点），在准确率-效率曲线上处于最优位置；

**⚠️ 局限性**

局限性包括依赖LLM生成的语义依赖注解，可能受到注解质量影响；方法主要针对数学推理任务，尚未验证在更广泛的推理场景或更大模型上的泛化性；

---

## 127. A Stem-Agnostic Approach to Hybrid AI Music Detection

**arXiv ID:** 2609.26956 | [PDF](https://arxiv.org/pdf/2609.26956v1)

**作者:** Richa Namballa `[一作]`, Romain Hennequin `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在混合音乐中检测人工合成音轨的存在，提出一种与音轨无关的检测框架；

**💡 创新点**

引入inspectrogram（局部合成概率的时频表示）与能量掩膜相结合的CNN模型，能够在单个模型下识别任何音轨；

**🔧 技术方法**

使用Band‑Split Regression构建inspectrogram、Wiener滤波能量掩膜、CNN分类器以及EnCodec作为合成音频代理；

**📊 数据集**

训练数据基于MUSDB18‑HQ混合版本（采用EnCodec生成合成音轨），外部验证使用MoisesDB多轨数据；

**📈 对比分析**

与传统全音轨二分类检测相比，模型在人声、鼓、吉他等高频音轨上实现了约90%+的召回率、低于10%的误报率，而在低频贝斯音轨上表现显著下降；

**⚠️ 局限性**

性能受限于源分离质量，贝斯和钢琴等低频/弱能量音轨难以分离，导致检测准确率降低；

---

## 128. PEARL: A Lightweight Prompt-based Feature Interpreter Framework for Real-Time, Anonymous, and Heterogeneous Collaborative Perception

**arXiv ID:** 2609.27123 | [PDF](https://arxiv.org/pdf/2609.27123v1)

**作者:** Armin Maleki `[一作]` (Michigan State University), Hayder Radha `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PEARL框架，通过视觉提示实现匿名、实时且轻量的异构协同感知。

**💡 创新点**

创新点包括：①同时使用稀疏检测提示和稠密域不变提示并行解读器，②低秩PARAFAC分解显著降低提示参数和计算成本，③基于域不变特征的余弦相似度实现匿名模型选择，④两阶段训练只更新轻量模块即可快速适配新代理，保持单车性能与隐私。

**🔧 技术方法**

采用的技术包括：低秩视觉提示（PARAFAC）、多尺度BEV提取与融合、跨注意力和3D空间注意力、对抗域分类、轻量化压缩、余弦相似度模型选择等。

**📊 数据集**

实验使用了OPV2V、V2XSet和DAIR‑V2X三大协同感知数据集，涵盖LiDAR和相机多模态。

**📈 对比分析**

与PolyInter等最先进异构CP基线对比，Stage‑1、Stage‑2 AP@0.5/AP@0.7均提升2–6%；实时模型选择相比随机提升8.2% AP@0.7，平均计算延迟仅1.67 ms；通信成本相比Baseline下降8.7×–34.7×。

**⚠️ 局限性**

局限性：仍需离线训练多解读器池；极端异构场景（不同架构或训练域）可能需要更多提示；大规模候选池时GPU内存受限；缺乏对动态跨域持续适配的研究。

---

## 129. Super-Resolution of Solar Magnetograms via Adaptive Stratified Ensemble Learning with Uncertainty Estimation

**arXiv ID:** 2609.27131 | [PDF](https://arxiv.org/pdf/2609.27131v1)

**作者:** Sina Norouzi Kandalan `[一作]` (Sam Houston State University), Qin Li `[通讯]` (New Jersey Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种分层专家集成（SSE）框架，用于将低分辨率SOHO/MDI太阳磁图提升到与SDO/HMI相当的高分辨率；

**💡 创新点**

创新点在于：①利用图像复杂度（标准差）作为路由依据，构建低、中、高三种专家网络；②在专家训练时采用加权随机采样，突出对应复杂度区间的数据；③引入自适应度量门控课程损失，在训练过程中动态平衡PSNR、SSIM和相关系数的梯度贡献；④在推理时使用八重D4旋转/翻转的测试时增强（TTA）并生成不确定性图；

**🔧 技术方法**

技术上使用了改造的RRDBNet网络（继承ESRGAN预训练权重），配合输入/输出适配器、SSIM增强块和PSNR精细化头；采用混合精度、梯度裁剪、EMA权重平均；实现分层采样、门控损失与TTA；

**📊 数据集**

使用SOHO/MDI与SDO/HMI在2010-2011年重叠期间的1569对磁图，经过时序尾部划分得到1493训练对（D4扩充后约12k样本）和76个测试对；

**📈 对比分析**

与SolarCNN、RRDBNet以及基线网络进行比较，SSE在PSNR、SSIM、CC三个指标上分别取得37.66 dB、0.9443、0.9439，均高于对比方法；Ablation实验显示加权采样、适应性损失和TTA三者各自贡献显著；

**⚠️ 局限性**

局限性包括：仅在MDI–HMI重叠的短时间窗口内评估；未对不同观测周期或外推分布进行充分验证；对高分辨率外部数据的适用性尚未测试；

---

## 130. MSK-Bench: Benchmarking Full-Body Musculoskeletal Motor Control Across Tasks, Control Paradigms, and Physiological Metrics

**arXiv ID:** 2609.26872 | [PDF](https://arxiv.org/pdf/2609.26872v1)

**作者:** Mengtao Ou `[一作]` (Tsinghua University), Hao Zhao `[通讯]` (Tsinghua University)

**通讯引用:** 10752 | [OpenAlex ID](https://openalex.org/A5100762170)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MSK‑Bench，一个包含 22 个全身肌肉驱动任务的基准，统一评估任务成功率、鲁棒性、肌肉激活成本、关节平滑度和 EMG 包络相似度等指标，覆盖姿态稳定、行走与环境交互三大功能范畴。

**💡 创新点**

创新点在于：①首次将肌肉驱动机器人任务整合为统一的评测框架；②引入多维生理诊断指标（如 EMG 包络相似度），实现任务与生理合理性双重评价；③在统一协议下系统对比了奖励强化学习、代理奖励调优、潜在动作强化、模仿先验控制以及残差自适应等五类控制范式，并通过残差自适应案例展示从诊断到验证的闭环流程。

**🔧 技术方法**

使用了 MuJoCo 物理引擎配备 416 肌肉全身模型；实现了 PPO、SAC、DepRL、DynSyn‑SAC 等强化学习算法；利用深度学习实现潜在动作编码器、代理奖励推理器（DeepSeek、GPT‑6 Astra）；采用 MuscleMimic 预训练模仿先验；残差自适应通过冻结先验并学习三层 MLP 修正；同时计算肌肉激活成本、关节平滑度、EMG 包络相似度等诊断指标。

**📊 数据集**

基于 416 肌肉模型的仿真环境，并使用公开的走步、跑步、爬楼梯等人类 EMG 数据（分别针对三条腿肌肉、十一条跑步 EMG 通道以及十二条楼梯 EMG 通道）作为参考；任务本身通过自定义的 22 个控制目标和物理扰动生成。

**📈 对比分析**

采用统一的评价协议对 5 种控制范式在 22 个任务上进行比较，指标包括成功率、鲁棒性（动作、观测、动力学扰动下的成功率）和 EMG 包络相似度。结果显示：奖励强化学习各变体在不同任务家族表现差异明显，DepRL 在交互任务中覆盖率最高；模仿先验在姿态与行走任务中表现优秀，却在楼梯任务上完全失败；残差自适应在楼梯任务恢复成功率并在跑步和步行任务中提升 EMG 相似度，但提升幅度不一致，表明无单一方法可兼顾所有指标。

**⚠️ 局限性**

局限性包括：仅使用固定的 416 肌肉模型，未检验对不同解剖结构的泛化；EMG 参考仅覆盖少量任务和肌肉，诊断范围受限；残差自适应的提升归因于完整的修正管线而非单一组件，难以单独评估；缺乏统计显著性检验与真实世界实验验证，评估结果受仿真环境约束。

---

## 131. MultiPush: Learning to Rearrange with Teams of Car-Like Pushers

**arXiv ID:** 2609.27005 | [PDF](https://arxiv.org/pdf/2609.27005v1)

**作者:** Jeeho Ahn `[一作]` (University of Michigan), Christoforos Mavrogiannis `[通讯]` (University of Michigan)

**通讯引用:** 1140 | [OpenAlex ID](https://openalex.org/A5067086333)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 MultiPush 框架，利用强化学习联合生成车类推送任务的调度与分配，并通过推送可达图与优先级规划实现多车非抓取对象重排。

**💡 创新点**

创新点在于：① 将车类运动约束映射为 Dubins 曲线，构建推送可达图，显著简化多车路径搜索；② 通过强化学习直接学习任务分配策略，既避免了完整路径规划的高昂成本，又能在约束下实现高质量调度；③ 采用基于优先级的 Hybrid A* 与冲突解决机制，实现了实时可执行的多车轨迹。

**🔧 技术方法**

核心技术包括：Dubins 曲线与 Reeds–Shepp 运动模型、推送可达图（PT-graph）、强化学习（policy gradient / REINFORCE）训练分配策略、优先级规划（Hybrid A*）、冲突检测与安全停车策略。

**📊 数据集**

使用随机生成的实验数据集：工作空间尺寸 4.5×5.5 m（训练）和 5.0×5.0 m（测试），包含 8、10、12、14 个立方体对象，每个实例生成多种姿态变体，训练总计 10,400 个实例，测试每种对象数 400 个实例。

**📈 对比分析**

与 GREEDY、RANDOM、LNS、BOSS 等基线进行比较，评估指标为相对完成时间、成功率和规划时间。MultiPush 在所有实验中相对完成时间均优于 LNS（约 10–20% 下降），规划时间最快，成功率可达 97–98%，同时在未见工作空间和不同车数（2–4）下保持良好泛化。

**⚠️ 局限性**

局限性：依赖 ReloPush‑BOSS 生成的种子，最多 14 个对象；仅评估同形立方体，忽略不同尺寸/形状和动态推力/滑动；仅在 2–4 辆车场景中验证；采用集中式规划，未实现分布式/去中心化执行；对环境不确定性和物理接触误差建模不足。

---

## 132. Math Reasoning in LLMs is Organized by Approach, Not Topic

**arXiv ID:** 2609.27041 | [PDF](https://arxiv.org/pdf/2609.27041v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 133. TinyUDE: Solver-Free Universal Differential Equations on Microcontrollers via Lie-Taylor Jet Matching

**arXiv ID:** 2609.26972 | [PDF](https://arxiv.org/pdf/2609.26972v1)

**作者:** Pranavanath Balamurali `[一作]` (University of Texas at Austin), Hrishi Kamireddy `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于噪声自适应Lie–Taylor jet匹配的无求解器UDE训练框架，直接拟合观测状态的第一、二阶时间导数，实现在线物理模型学习。

**💡 创新点**

创新点在于完全去除了数值ODE求解器和反向传播缓存，通过Savitzky–Golay滤波、全速相位采样、储备缓冲、噪声自适应滤波尺度与二阶导数门控等技术实现了无显式求解器的解析梯度，显著降低内存占用且保持或提升模型精度。

**🔧 技术方法**

使用的技术包括Savitzky–Golay多项式滤波、Lie导数公式、全速相位采样、储备采样、余弦退火Adam优化、Polyak加权平均、噪声估计与质量门控。

**📊 数据集**

在仿真数据集上评估：阻尼单摆和受限双摆系统，添加高斯测量噪声（0%–5%），并在ESP32微控制器上实现。

**📈 对比分析**

与传统基线（RK4多步射击、精确离散伴随、Adam）对比，Jet匹配在多数噪声水平下实现了约0.65×的场向量误差，并将训练内存从约6 MB压缩到108 kB；ESP32上实现了61 kB内存、7.24 ms更新周期，满足25 Hz实时训练。

**⚠️ 局限性**

局限性包括：仅在单摆1%噪声的单一硬件点验证，低噪声下仍不如传统伴随法；对快速跃迁的非线性状态仍可能产生滤波误差；假设噪声为高斯，未测试非高斯或量化误差情况。

---

## 134. A Systematic Evaluation of Infrastructure-Based Radar System for Highway Traffic Monitoring

**arXiv ID:** 2609.27143 | [PDF](https://arxiv.org/pdf/2609.27143v1)

**作者:** Tianheng Zhu `[一作]` (Purdue University), Yiheng Feng `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建DRaT双模态数据集，对高速公路合流段的基础设施雷达在检测、跟踪和宏观流量参数估计三层级进行系统评估。

**💡 创新点**

将无人机轨迹作为高精度地面真值，引入三层级评估框架（检测、跟踪、宏观指标），并公开DRaT数据集以促进可重复研究。

**🔧 技术方法**

使用FMCW雷达信号处理、YOLO+BoT‑SORT目标检测与跟踪、Hungarian匹配、NMS、地理配准、以及精度、召回、MOTA、IDF1、MAE、MAPE等统计评估指标。

**📊 数据集**

DRaT数据集（约5,200条车辆轨迹，包含雷达与无人机同步数据），采集自Fort Worth高速公路合流段。

**📈 对比分析**

通过IoU阈值0.1的Hungarian匹配对雷达检测与无人机真值进行比对；检测精度78%，召回57%；跟踪IDF1为0.699，MOTA为0.535；宏观参数空间平均速度误差<4%，密度和流量误差约23%；在拥堵状态下性能显著下降。

**⚠️ 局限性**

仅使用商用雷达和其内置算法，未探讨其他信号处理或多模态融合；评估仅在单一高速段进行，缺乏曲线、匝道等多样道路场景的验证。

---

## 135. Full-Covariance Smoothing of Bayesian Neural Networks for Online Adaptation

**arXiv ID:** 2609.27244 | [PDF](https://arxiv.org/pdf/2609.27244v1)

**作者:** Oren Wright `[一作]` (Carnegie Mellon University), José M. F. Moura `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过将贝叶斯神经网络训练视为状态空间模型，利用前向矩阵传播与后向Rauch–Tung–Striebel平滑实现单步闭式权重更新，并首次允许全协方差传播以处理非线性激活；

**💡 创新点**

创新点在于推导激活函数输入与输出的交叉协方差公式，使得能够在单步层级平滑中保留跨神经元相关性，并兼容噪声与分布式目标；

**🔧 技术方法**

使用高斯矩阵传播、Stein/Bussgang定理、Rauch–Tung–Striebel平滑、矩阵运算与激活函数的解析时刻公式；

**📊 数据集**

在旋转Moon、CartPole、Siemens工业基准、以及Frankia机器人VLA任务的非平稳数据上进行评估；

**📈 对比分析**

与KBNN、TAGI、Gaussian Process和标准SGD对比，实验表明在非平稳分类、动态学习和VLA微调中，本文方法在准确率/误差方面普遍优于对手，且在突变场景下的适应性更强；

**⚠️ 局限性**

局限包括：全协方差更新在宽度大时计算量高，权重与激活独立假设缺乏深入验证，缺少对目标漂移的显式建模，以及实验仅验证噪声目标，分布目标仍待进一步验证。

---

## 136. Training Intelligent Voice Assistant Wakeup with Controllable Synthetic Conversations

**arXiv ID:** 2609.27037 | [PDF](https://arxiv.org/pdf/2609.27037v1)

**作者:** Marcin Sowański `[一作]` (TCL Research Europe), Krzysztof Wodnicki `[通讯]` (University of Warsaw)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于语境的智能唤醒系统，在识别唤醒词后通过语音分类模型判断是否激活虚拟助手。

**💡 创新点**

创新点在于将传统的关键词唤醒与上下文触发检测相结合，并构建可控的多说话人合成语料库，实现更自然的交互。

**🔧 技术方法**

采用了预训练语音编码器（如 Whisper、Zipformer、Hubert）+注意力池化、因果 Transformer、位置编码等技术。

**📊 数据集**

使用自研的 62.3 小时合成对话数据集（v1.0.1 版，2,297 条对话、35,788 轮），并与 VCTK、NOTSOFAR‑1 等公开数据进行对比评估。

**📈 对比分析**

与基线相比，直接唤醒召回率可达 0.99，语境唤醒最高 0.79，整体 F1 约 0.89；但在隐式跟进场景下仍低于 0.80。

**⚠️ 局限性**

局限在于合成语音的自然度不及真实录音，对隐式上下文的理解能力不足，导致上下文唤醒准确率受限。

---

## 137. Impact-Time Guidance via Normal Contraction to a Time-to-Go Isochron

**arXiv ID:** 2609.26906 | [PDF](https://arxiv.org/pdf/2609.26906v1)

**作者:** Shivam Bajpai `[一作]` (University of Cincinnati), Abhinav Sinha `[通讯]` (University of Cincinnati)

**通讯引用:** 1047 | [OpenAlex ID](https://openalex.org/A5022385451)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于正则收敛（normal contraction）的冲击时序引导方法，在原始追踪（如偏置PN）基础上加入标量时序校正，实现准时撞击。

**💡 创新点**

创新点在于将冲击时序视为移动等时线（time-to-go isochron），利用坐标不变的秩一度量和预测缺陷（predictor defect）构建鲁棒标量滤波，保证在加速度受限下实现首次撞击与预定时刻同步，并引入预终端对齐与归还机制避免时序权能消失。

**🔧 技术方法**

采用了偏微分动力学、传输方程、收敛度量、鲁棒漏斗控制以及基于最小范数投影的速度正交输入调制等技术。

**📊 数据集**

实验验证仅通过仿真数据完成，涉及静止目标（LPN预测）与机动目标（偏离追踪预测）两种典型情形。

**📈 对比分析**

与传统PN和闭式时序预测器比较，仿真结果显示该方法在满足加速度上限的前提下能够在规定时间内完成捕获，误差始终保持在漏斗边界内，且对预测误差具有鲁棒性。

**⚠️ 局限性**

局限在于未考虑执行器动态、传感误差及三维扩展；方法依赖已知目标模型，且在极端碰撞航线对齐时需要额外对齐处理。

---

## 138. Reinforcement Learning with Decomposed Subtasks

**arXiv ID:** 2609.27035 | [PDF](https://arxiv.org/pdf/2609.27035v1)

**作者:** Mattie Terzolo `[一作]` (Upwork), Andrew Rabinovich `[通讯]` (Upwork)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 RLDS（Reinforcement Learning with Decomposed Subtasks）方案，利用反思式回放中的子任务评分与关键步骤来拆分轨迹奖励，并在此基础上构造 Subtask‑Decomposed Advantage Estimation (SDAE)。

**💡 创新点**

创新点在于将轨迹奖励按固定子任务树拆解、对每个子任务进行组内基线消除，并通过关键步骤的核函数对 token 级别信用进行精准分配，解决了传统 GRPO 在多技能任务中信用归属模糊的问题。

**🔧 技术方法**

技术上结合了反思回放（R^3L/ERL）、GRPO 的组相对优势、SDAE 的多阶段拆分与批量归一化，以及基于高斯核的 pivot‑mask 分配；同时加入了反思‑NLL 与重试‑蒸馏的辅助损失。

**📊 数据集**

实验数据集包括 FrozenLake、HotpotQA、ScienceWorld 和 DeepResearch，涵盖从 10 步到 38 步的长短回合、工具使用与否以及多维评价标准。

**📈 对比分析**

与标量 GRPO/RLVR 的比较显示，在子任务异质性高的 ScienceWorld（+11.5 分）和 FrozenLake（+9.8 分）任务上 RLDS 明显优于基线；而在子任务异质性低的 HotpotQA 与 DeepResearch 上提升不显著；计算效率在长回合任务中 RLDS 甚至更快（-10.9%）。

**⚠️ 局限性**

局限性包括：子任务分类需要人工设计且固定；子任务重要性权重 φ_k 也固定，未能动态调整；当子任务信号弱或关键步骤聚集时，SDAE 会退化为标量 GRPO。

---

## 139. Balancing Generality and Specialization: A Survey on AI Datacenter Hardware Architecture

**arXiv ID:** 2609.26829 | [PDF](https://arxiv.org/pdf/2609.26829v1)

**作者:** Yufeng Gu `[一作]` (University of Michigan), Reetuparna Das `[通讯]` (University of Michigan)

**通讯引用:** 5375 | [OpenAlex ID](https://openalex.org/A5027544576)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并分类工业AI加速器、分析规模化互连与技术演进，提出从通用与专用平衡视角的框架。

**💡 创新点**

以“通用vs专用”四类架构构建统一的评估框架，结合多层级互连与收敛算法映射，系统追踪代际演进并关联功耗、散热与设计挑战。

**🔧 技术方法**

系统架构分类、通信拓扑分析、收敛算法调度、算子级数据路径演化、功耗与冷却模型等技术手段。

**📊 数据集**

公开的加速器规格、网络拓扑、功耗/冷却数据以及公开模型参数增长曲线。

**📈 对比分析**

对比各类加速器的计算单元、内存层级、互连带宽，使用统一指标如TFLOPS、HBM带宽、AllReduce延迟；结果显示计算吞吐量增长最快，内存与通信增长滞后。

**⚠️ 局限性**

仅基于公开资料，缺乏真实运行负载评测，且对未来模型演进的预测仍不确定，未深入探讨软硬协同调度与动态重构策略。

---

## 140. Spiderbot: An Open-Source Energy-Efficient Hexapod with Passive Gravity Compensation

**arXiv ID:** 2609.26989 | [PDF](https://arxiv.org/pdf/2609.26989v1)

**作者:** Ritwik Sharma `[一作]` (BITS Pilani), Saransh Agrawal `[通讯]` (BITS Pilani)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一款低成本、能效极高的六足机器人Spiderbot，采用4杆连杆和被动弹簧实现2-DOF腿部关节，显著降低站姿功耗。

**💡 创新点**

创新点在于：①使用被动弹簧-四杆连杆机制实现重力补偿，站姿功耗下降至1.5 W（比无弹簧版低90%）；②在保持6足重力支撑的前提下，单腿仅需2个主动关节，减少重量和能耗；③实现了从仿真到真实机器人的高效迁移，解决了非标准4杆几何无法用URDF描述的问题。

**🔧 技术方法**

技术手段包括：
- 机械设计：3D打印零件、被动弹簧、四杆连杆；
- 控制：基于MuJoCo+mjlab的仿真环境，使用PPO算法训练的异步演员-评论家策略；
- 硬件：Arduino Uno Q、SC15与ST3215伺服、IMU、Li‑Po电源；
- 软硬件集成：Python脚本与Arduino串行通信，完整的开源代码与CAD。

**📊 数据集**

数据集/环境：在MuJoCo中构建了10×20格、8 m×8 m的程序化地形网格，包含平地、台阶、斜坡、随机粗糙、波浪等7种地形；通过域随机化（摩擦、质量、惯量、传感噪声等）生成训练数据；真实实验使用手工布置的斜坡与障碍物。

**📈 对比分析**

比较方法：将Spiderbot的站姿功耗和平地CoT与PhantomX、HexaV4、HAntR等现有六足机器人进行对比。结果显示Spiderbot的站姿功耗为1.5 W，远低于PhantomX（≈40 W）和HexaV4（≈10 W）；在平地高速行走时，CoT约为3.54 W/(kg·m/s)，与已发表的高性能平台相近或更优。尽管在斜坡和粗糙地形上CoT相对更高，但整体能效优势突出。

**⚠️ 局限性**

局限性：①2-DOF设计限制了腿部可达空间，无法通过脚踝摆动完成高障碍跨越；②被动弹簧刚度固定，无法适应不同负载或地形变化；③仿真与真实功耗存在显著差距，主要受电机驱动、通信与电磁干扰影响；④在极端坡度或大幅障碍时表现欠佳，需要进一步优化机械或控制策略。

---

## 141. NaviScale: Generating Large-Scale Semantic Map Datasets for Object Navigation

**arXiv ID:** 2609.27218 | [PDF](https://arxiv.org/pdf/2609.27218v1)

**作者:** Chuanlin Lan `[一作]` (Shandong University), Dongxiao Yu `[通讯]` (Shandong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出NaviScale框架，通过将真实住宅平面图与从MP3D和HM3DSem提取的房间级语义与障碍图组合，自动生成192,000个语义地图对，用于训练语义地图基础的目标导航模型。

**💡 创新点**

创新点在于只需要真实平面图与房间级语义图即可构造完整语义地图并通过可见性射线投射生成部分观测，显著扩大训练数据规模，同时在保持原始预测网络不变的情况下提升导航性能。

**🔧 技术方法**

采用房间类别匹配、交叉平面图拼接、尺度拟合、可见性射线投射（VisRC）等技术，以及基于PSPNet的PEANUT网络进行语义地图补全与目标定位。

**📊 数据集**

训练数据来源于24,000个收集自公开房源的住宅平面图、MP3D和HM3DSem的房间级语义与障碍图，验证使用HM3D与MP3D基准环境。

**📈 对比分析**

在HM3D上使用300k训练步后SR提升至64.3%（相较原始59.6%提升5.7pp）且SPL提升至34.8%；在MP3D上SR提升至43.1%（相较原始40.3%提升2.8pp），在物理机器人实验中成功率分别提高约17%和20%。

**⚠️ 局限性**

局限在于仅扩展了房间级语义库，稀有房间布局仍难以覆盖，且主要聚焦住宅环境，缺乏多样化建筑类型与动态交互场景。

---

## 142. ChipMEM: Verification-Grounded Memory for EDA Agents

**arXiv ID:** 2609.27067 | [PDF](https://arxiv.org/pdf/2609.27067v1)

**作者:** Abdulrahman AlRabah `[一作]` (University of Illinois Urbana Champaign), Sandesh Adhikary `[通讯]` (Cadence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本工作提出并实现了ChipMEM，一种面向EDA代理的验证驱动记忆层，能够将经过EDA工具验证的执行轨迹转换为可复用的程序化技能和贝叶斯统计记忆，并通过通用适配器无缝集成到不同的RTL优化与测试平台；该记忆层在跨任务迁移和单任务内部决策中均显著提升了合成、仿真、等价验证及PPA指标，并降低了LLM调用、工具调用、token使用和工具执行时间。

**💡 创新点**

创新点包括：① 只在通过工具验证（如合成、等价检查）后才存储程序化技能，避免自我评估误导；② 通过贝叶斯Beta模型实时估计工具调用成功率和恢复策略，提供统计导向的重试与恢复建议；③ 将程序化与统计记忆统一于一个记忆层，并提供无模型依赖的通用适配器，支持跨域、跨工具的迁移；④ 在冻结模式下验证记忆对未见任务的迁移效果，证明可迁移知识的通用性。

**🔧 技术方法**

技术手段包括：大语言模型（GPT‑5.5、Qwen3.8‑27B）、任务嵌入模型（余弦相似度检索）、贝叶斯Beta分布（工具重试与恢复预测）、程序化技能提炼（轨迹压缩、元数据存储）、Agent Adapter（记录轨迹、提供技能与统计建议）以及标准EDA工具链（综合、仿真、定时、等价检查等）。

**📊 数据集**

使用的数据集与基准包括：RTL‑OPT、RTLRewriter‑Bench（PPA优化任务），CVDP（CID012/ CID013 测试平台），OpenTitan（10个IP块多轮评估），以及20个开源自定义设计。

**📈 对比分析**

实验与对比方法：与无记忆基线、程序化记忆单独、贝叶斯记忆单独以及两者结合进行比较；评估指标涵盖等价通过率、面积/功耗/时序改进、LLM/工具调用次数、token消耗与工具执行时间。结果表明：在RTL‑OPT上平均面积提升从6.18%提升至8.79%；在RTLRewriter‑Bench上从5.95%提升至8.88%；自定义设计面积从0.14%提升至0.31%；OpenTitan面积从0.05%提升至0.24%，功耗从0.00%提升至2.54%；LLM调用、工具调用、token与工具时间均显著下降。对未见任务的冻结记忆测试，CVDP任务从18/20通过率提升至20/20，CID013从8/10提升至10/10。

**⚠️ 局限性**

局限性包括：记忆检索固定为前两条结果且相似度阈值为0.6；实验仅覆盖5个RTL‑OPT任务和10个未见CVDP任务；未对更大规模记忆、不同检索策略和更广泛的EDA任务进行验证；需要进一步研究单技能贡献与跨域迁移能力。

---

## 143. Provably Complete Generalized Planning with LLMs

**arXiv ID:** 2609.27105 | [PDF](https://arxiv.org/pdf/2609.27105v1)

**作者:** Katharina Stein `[一作]` (Saarland University), Alexander Koller `[通讯]` (Saarland University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

自动将 PDDL 规划域转换为 Lean 形式，使用大语言模型（LLM）生成可执行的一般化计划及其完整性证明，并通过 Lean 核心验证其正确性。

**💡 创新点**

将 LLM 直接应用于正式化程序和证明生成，实现大规模（13 个基准域中 12 个）一般化计划的自动完整性证明，突破以往仅在极简域或小实例集上完成的局限；同时提供语义保持的 PDDL‑to‑Lean 转换与自动调试循环。

**🔧 技术方法**

技术栈包括：GPT‑5.6‑Sol（温度1，高推理），Lean 定理证明器，Python 语义转换器，自动化调试与错误反馈机制，以及状态不变式与有效性约束的手工定义。

**📊 数据集**

实验基准使用 IPC Learning 2023 任务集中的 13 个经典规划域（如 Delivery、Ferry、Grippers 等），每个域使用 6 个小规模实例作为调试任务。

**📈 对比分析**

与人工完成的完整性证明对比，所提方法在 12/13 个域上获得完整证明；生成时间从 9 分钟（Heavy）到 54 分钟（Rovers）；交互次数与证明声明数均在可接受范围内（例如 Delivery 121 次交互，681 次声明）。

**⚠️ 局限性**

局限性包括：Transport 域未能自动生成完整证明（LLM 复杂度限制）；依赖手工提供有效性约束与不变式；适用 PDDL 子集有限，需手动扩展以覆盖更复杂的规划语义。

---

## 144. AgroBench: A Reproducible Multimodal Benchmark for Weakly Supervised Crop Yield Learning from County Statistics and Pixel Observations

**arXiv ID:** 2609.26809 | [PDF](https://arxiv.org/pdf/2609.26809v1)

**作者:** Udaiveer Singh `[一作]` (Plaksha University), Dharmendra Saraswat `[通讯]` (Purdue University)

**通讯引用:** 1987 | [OpenAlex ID](https://openalex.org/A5025951776)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5a41884c-404f-4688-a89c-aa238c10fe68` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了AgroBench基准，利用县级产量统计与多模遥感数据生成弱监督像素级作物产量时序。

**💡 创新点**

首次提供可复现的弱监督像素级产量学习管道，结合多模遥感与县级统计，填补尺度不匹配空白。

**🔧 技术方法**

采用多模遥感（Sentinel‑1/2、MODIS LST、CHIRPS、SRTM）与地理信息处理、时间对齐与特征融合技术。

**📊 数据集**

基于USDA NASS Quick Stats县级产量、USDA Cropland Data Layer、Sentinel‑1/2、MODIS、CHIRPS、SRTM等公开数据集。

**📈 对比分析**

通过Leave‑One‑Year‑Out交叉验证与基线模型（随机森林、LightGBM、XGBoost、线性回归、MLP）比较，随机森林取得R²≈0.58、RMSE≈120。

**⚠️ 局限性**

限制在于使用县级产量作为标签导致像素级标签噪声，无法捕捉县内产量差异，且基线未充分利用时序信息。

---

## 145. A New Method that can Generate Ramsey Colourings for Eight and Thirteen Colours

**arXiv ID:** 2609.26851 | [PDF](https://arxiv.org/pdf/2609.26851v1)

**作者:** Charles Gretton `[一作]`, Cody Christopher `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了与完全图的某些边着色相对应的关系代数的表示，特别是为8和13种颜色找到合适的着色。

**💡 创新点**

提出了一种新的猜测与检查的方法，称为融合美丽融合方法，成功为8和13种颜色找到着色。

**🔧 技术方法**

采用了Comer的有限域方法的改编，结合了对细关系的融合。

**📊 数据集**

使用了小于2000的素数作为数据集，特别是对于8和13种颜色，分别使用了𝔽_449和𝔽_1613。

**📈 对比分析**

与现有的着色方法进行了比较，发现新方法构造的着色在某些情况下是非同构的，且在8和13种颜色的情况下提供了新的着色。

**⚠️ 局限性**

方法的局限性在于仍需通过搜索找到混合差异配对，并验证不同融合对之间的和集兼容性条件。

---

## 146. Anti-Localization Uplink Communications in Satellite-Terrestrial Systems

**arXiv ID:** 2609.27258 | [PDF](https://arxiv.org/pdf/2609.27258v1)

**作者:** Ranran Sun `[一作]` (Hangzhou Institute of Technology), Xiaohong Jiang `[通讯]` (Future University Hakodate)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出在卫星-地面系统中通过协作干扰与联合信号处理实现反定位上行通信；

**💡 创新点**

创新点在于引入了定位误差概率（LEP）指标，结合信号检测与TDOA定位的联合分析，并通过随机干扰与功率分配实现对威利卫星的定位抑制；

**🔧 技术方法**

采用了Rician多天线信道建模、能量检测、协同干扰、通用交叉相关、Cramér‑Rao下界与Chan算法等信号处理与优化技术；

**📊 数据集**

该研究主要基于仿真验证，未使用公开数据集，仿真采用随机PPP分布、Rician因素等参数生成；

**📈 对比分析**

与无干扰基线对比实验表明，协作干扰能够显著提升LEP（定位误差概率）并在保持Bob端SINR满足阈值的前提下实现较高的定位保护；

**⚠️ 局限性**

局限性包括：对硬件能力与天线数的依赖、对威利卫星协同与能量检测假设的敏感性、以及在严格Bob端SINR约束下可用干扰功率受限等问题。

---

## 147. Solidity Meets LLMs: A Transformer-Based Approach to Smart Contract Vulnerability Detection

**arXiv ID:** 2609.27091 | [PDF](https://arxiv.org/pdf/2609.27091v1)

**作者:** Djamel Eddine Hakim Ghorab `[一作]` (University of Oum El Bouaghi), Mostafa Anouar Ghorab `[通讯]` (Laval University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建基于BERT的深度学习模型，用于检测Solidity智能合约代码片段的漏洞。

**💡 创新点**

将大型语言模型与智能合约安全检测结合，自动标注安全与漏洞片段，形成高质量训练集。

**🔧 技术方法**

使用BERT预训练模型、Hugging Face Transformers库，以及Slither和SolScan等静态分析工具进行自动标注。

**📊 数据集**

利用从Etherscan获取的Solidity合约文件，并通过Slither与SolScan自动生成的标签构成训练、验证和测试集。

**📈 对比分析**

与传统静态/动态检测工具对比，模型在五轮训练后达到92% F1得分，显示出较高的检测准确率。

**⚠️ 局限性**

局限在仅做二分类、缺乏多类别漏洞识别、对罕见或未知漏洞的泛化能力不足。

---

## 148. A Leakage-Aware Multimodal Evaluation Framework for Early Intraoperative Acute Kidney Injury Prediction

**arXiv ID:** 2609.26848 | [PDF](https://arxiv.org/pdf/2609.26848v1)

**作者:** Quang Minh Nguyen `[一作]` (National Economics University), Trong Nghia Nguyen `[通讯]` (National Economics University)

**通讯引用:** 726 | [OpenAlex ID](https://openalex.org/A5034859176)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一套泄漏意识的早期术中急性肾损伤（PO-AKI）预测框架，并通过多模态融合与安全堆叠提升预测性能。

**💡 创新点**

创新点在于：①首次系统地对早期预测设置进行泄漏控制；②将波形时间序列与结构化临床/血流动力学特征进行多模态融合；③采用泄漏安全的层叠集成与Platt再校准，提升模型校准与临床效益。

**🔧 技术方法**

技术手段包括：SynerT（TCN+膨胀RNN时间编码）、多模态late fusion、meta-learner层叠、交叉拟合Platt再校准、ROC/PR、决策曲线分析。

**📊 数据集**

数据集：VitalDB perioperative数据库，3242例手术病例，其中2413例波形可用，180例后发AKI（7.46%）; 还做了在eICU‑CRD Demo上做的跨设置压力测试。

**📈 对比分析**

在五折交叉验证中，对比了CNN‑LSTM、原始SynerT、随机森林、Extra‑trees、CatBoost等基线；SynerT‑Stack获得最高AUROC 0.773±0.032、AUPRC 0.252±0.031，且在外部eICU测试中保持优异（AUROC 0.831、AUPRC 0.555）。

**⚠️ 局限性**

局限性：仅使用血清肌酐作为AKI标签，未包含尿量信息；外部验证样本量小且非手术ICU人群；需在更大、匹配的手术人群上进一步验证。

---

## 149. ChartRevive: Reconstructing Data Visualizations from Chart Images Using MLLM

**arXiv ID:** 2609.27146 | [PDF](https://arxiv.org/pdf/2609.27146v1)

**作者:** Yuki Ueno `[一作]` (Arizona State University), Aditeya Pandey `[通讯]` (Eli Lilly and Company)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对静态图像中的数据与视觉设计进行自动提取，并提供交互式验证与编辑，支持快速重建图表。

**💡 创新点**

提出混合主动的 MLLM 提取与交互式校验工作流，结合可层化输出实现数据与设计双重提取，并构建系统实现快速重建。

**🔧 技术方法**

采用商业多模态大语言模型进行提取，利用 Plotly 生成图表，结合 JSON schema 与程序化后处理，前端实现覆盖式校验与直接编辑。

**📊 数据集**

使用基于 Plotly 生成的 3,250 条合成图表数据集（包含条形、折线、散点、饼图、箱线图），用于模型评估。

**📈 对比分析**

对五个商业 MLLM 进行基准测试，使用 MCER、Adaptive MAPE、色差 ΔE 2000 等指标；最佳模型在分类/文本属性上误差低于 11%，数值/色彩属性仍高于 20%，整体在数据提取上表现出较大波动。

**⚠️ 局限性**

局限包括仅评估合成图表，未覆盖热图、KM 曲线等复杂图形；缺乏正式用户研究；MLLM 在数值与空间属性提取上仍存在显著误差，需要进一步微调或专门模型。

---

## 150. Building Socio-Affective Artificial Intelligence for Interactive Multi-Agent Simulations

**arXiv ID:** 2609.26927 | [PDF](https://arxiv.org/pdf/2609.26927v1)

**作者:** David Berga `[一作]` `[通讯]` (Universitat de Barcelona), David Berga (Universitat de Barcelona)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个名为 AGIMUD 的多智能体仿真平台，整合 Transformer 生成式大语言模型、情感推理、社会价值与治理模型，并通过 MUD 网络架构实现人机实时交互的多用户动态世界。

**💡 创新点**

创新点在于将 LLM 与神经符号推理、Schwartz、Ostrom、Shapley、Montes‑Sierra 社会价值体系以及 Ekman‑Frijda 情感动力学集成到同一决策管线，实现可解释、安全且情感化的代理行为；同时采用中心化/点对点 MUD 网络实现实时分布式多用户交互，突破传统基于规则或 BDI 的单智能体局限。

**🔧 技术方法**

技术栈包括 Transformer LLM（GPT‑4、Claude‑5、DeepSeek‑R1 等）、神经符号框架（JSON Schema 校验、RAG、ReAct 等）、情感模型（Ekman‑Frijda）、社会价值评估（Shapley、Ostrom、Schwartz）、分布式网络（中心化/点对点 MUD、WebSocket/WebRTC）、边缘推理（LM Studio、MCP）以及离线/云混合推理。

**📊 数据集**

数据集主要为自定义的世界模板和知识库，包含行动目录、规则、角色属性（Schwartz 价值、信任矩阵）、情感规则；实验使用人工生成的多角色世界实例（如 16 角色、10k 步）和模拟事件日志，未直接使用公开大型语料库。

**📈 对比分析**

通过离散化行为/情感统计、目标匹配率评估（仅 26.8% 的预期动作匹配）与行为多样性分析；相较传统规则/BDI 系统，安全约束更强但行为多样性下降；在 MUD 服务器上实现每秒 1 帧推理，内存占用约 10 MB/16 角色，推理延迟低于 30 ms。

**⚠️ 局限性**

局限性包括：安全约束导致可用动作集受限，安全性与多样性之间的权衡导致行为可预测性高；情感与社会价值权重调优复杂，导致行为预测困难；缺乏真实用户实验验证，沉浸感与人机交互质量尚未评估；在大规模场景下仍需进一步优化网络同步与分布式推理性能。

---

## 151. Design and Modeling of a Single-Port Three-Arm Robotic Tool for Minimally Invasive Neurosurgery

**arXiv ID:** 2609.27099 | [PDF](https://arxiv.org/pdf/2609.27099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 152. COMED: The Missing Middle Between Routing and Collaboration in Multi-LLM Inference

**arXiv ID:** 2609.26913 | [PDF](https://arxiv.org/pdf/2609.26913v1)

**作者:** Norah Alballa `[一作]` (King Abdullah University of Science and Technology), Marco Canini `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 4877 | [OpenAlex ID](https://openalex.org/A5042255975)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种post‑anchor控制框架，依据anchor模型自一致性、路由间隙与对齐可信度的peer disagreement，决定是否直接接受、轻量验证或开启跨模型协作，以提升多模型推理的准确率。

**💡 创新点**

创新点在于：1）设计Accept/Verify/Collaborate策略，将协作的非单调性拆分为救援与伤害；2）通过自一致性、router margin以及confidence‑gated peer probe三种信号实现对协作时机的精准判断；3）提出救援‑伤害分解理论，提供量化评估协作收益与风险的工具。

**🔧 技术方法**

使用的技术包括：自一致性（Self‑Consistency）采样、router margin阈值、轻量级peer probe（Verifier）以及summary‑routing协作协议；实验中还对比了JointMV聚合，验证控制策略的可迁移性。

**📊 数据集**

实验数据集涵盖：MedQA、JAMA Clinical Challenge、GPQA‑Diamond、MMLU‑Pro（MP‑Med、MP‑Bal）以及Frontier‑3的Humanity's Last Exam（HLE）等多领域问答与推理任务。

**📈 对比分析**

与全模型投票、Always‑Collab、固定Anchor与路由Anchor baseline进行对比。结果显示：在16个open‑weight设置中平均提升4.6个百分点，MedQA最高提升10.7个百分点；token消耗比dense协作低约33%；在Frontier‑3上，GPT‑5.5从23.1%提升至28.1%，显著优于always‑collab。

**⚠️ 局限性**

局限性：1）部分正确答案仍会被协作误导，伤害并未完全消除；2）实验仅覆盖7–8B开源模型池，未检验混合规模、长文本生成、对话、多语言等场景；3）延迟与成本评估有限，未覆盖真实生产环境的内存与计费开销。

---

## 153. ZO-COSMO: Index-Free One-Hop Mixing for Decentralized Zeroth-Order Optimization

**arXiv ID:** 2609.27199 | [PDF](https://arxiv.org/pdf/2609.27199v1)

**作者:** Shengjun Zhang `[一作]` (Hubei University), Dong Xie `[通讯]` (Baidu)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种名为 ZO-COSMO 的分布式零阶学习算法，该算法在每个通信环节仅传输 q 个标量值而不携带坐标索引，实现稀疏通信与一跳混合的兼容性。

**💡 创新点**

核心创新在于对支持兼容性进行解析，给出了实现一跳混合所需的必要条件；提出了全局支持与匹配支持两种稀疏协议，并给出了每个标量的收缩界限与收敛保证；通过共享查询方向与混合掩码，分析了方向相关性对估计误差与不一致性的影响。

**🔧 技术方法**

技术手段包括：两点差分零阶估计、平均保持的掩码一致性混合、公共随机种子生成公共支持、匹配化的边缘稀疏通信、稀疏动量更新、以及理论上的收缩与收敛分析。

**📊 数据集**

实验数据集涵盖：20 维 Rosenbrock 函数（合成数据）；Com-DSZO 与 Dense ZO-DSGD 等基线方法；以及真实大模型 Qwen2‑7B 在 QNLI 任务上通过 LoRA 进行的微调，包含 2.52M LoRA 坐标与 6.53B 可训练参数的实验。

**📈 对比分析**

与显式索引 Rand‑k、全邻居混合、Com‑DSZO（Top‑k）以及 Dense ZO‑DSGD 等方法对比，ZO‑COSMO 在相同通信预算下可获得 3.65% 的准确率提升；在完整图和环图上，匹配化混合分别提升 3.42% 与 2.53%；稀疏动量进一步提升 3.92%；实验显示其在大模型微调中节省约 2× 的通信量。

**⚠️ 局限性**

限制主要包括：需要公共随机种子与支持同步，适用于支持兼容的网络拓扑；对非常稀疏或极大维度时两点估计的查询成本仍高；在高度动态或异构的分布式环境中，匹配与支持的同步可能导致额外延迟。

---

## 154. An open benchmark for machine learning-based polymer property prediction

**arXiv ID:** 2609.27036 | [PDF](https://arxiv.org/pdf/2609.27036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 155. Crossflow: Prefill-Decode Elasticity for Agentic LLM Serving

**arXiv ID:** 2609.27085 | [PDF](https://arxiv.org/pdf/2609.27085v1)

**作者:** Yi Xu `[一作]` (Meta Platforms), Chunqiang Tang `[通讯]` (Meta Platforms)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可弹性扩展预填（prefill）与解码（decode）分离的 LLM 服务器体系结构，允许解码节点在保持解码优先级的同时，短时内借用计算资源来执行预填工作，从而在不改变节点角色的前提下利用瞬时容量不平衡。

**💡 创新点**

创新点在于：
- 通过短生命周期的多资源租约（lease）实现节点级可回收预填计算；
- 设计形状感知的集群调度器，结合虚拟队列、预填压力感知、租约动态更新，实现全局安全的预填借贷；
- 将预填与解码的特殊需求（KV 缓存、通信、输出预测）纳入租约和调度决策，突破传统静态预填/解码容量划分。

**🔧 技术方法**

采用 SGLang 运行时实现租约与执行后端；在 NVIDIA GB300 服务器上部署 GPT-OSS-120B 与 GLM-5.2 两个大模型；使用多资源租约、形状感知分配、虚拟队列、SLO‑安全计算预算等技术。

**📊 数据集**

使用公开的 TraceLab 代理工作负载轨迹（agentic trace）和内部隐私轨迹，并结合两大模型（GPT‑OSS‑120B、GLM‑5.2）进行时序保留的重放实验。

**📈 对比分析**

与静态 P/D、PPD、SMetric、CacheWise 等基线进行对比。实验显示：在多种负载与 P/D 分区下，几何平均提升 16.2–17.4% 的令牌吞吐量，峰值可达 43.4%；TTFT 统一下降 10–57%；在大部分设置下 ITL 下降 18–30%，但在高负载 GLM‑5.2 的某些分区 ITL 轻微上升。总体而言，弹性预填借贷显著提高利用率并降低首令延迟。

**⚠️ 局限性**

局限性：
- 在解码占用高、预填需求短暂时，解码延迟可能略有增加；
- 需要在调度层实现租约管理和形状估计，增加系统复杂度；
- 仅在预填/解码容量不平衡明显时才有效，对低波动负载收益有限；
- 依赖对请求形状和 KV 状态的准确预测，预测误差会影响分配质量。

---

## 156. Repurposing Pre-trained LLMs as High Fidelity Continuous Text Autoencoders

**arXiv ID:** 2609.27248 | [PDF](https://arxiv.org/pdf/2609.27248v1)

**作者:** Arkanath Pathak `[一作]` (University of California, Irvine), Alexander C. Berg `[通讯]` (University of California, Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将预训练的解码器语言模型重新设计为连续文本自动编码器（LLMAE），在模型内部通过结构化注意力掩码引入固定长度的潜在瓶颈，实现长文本（最长1024令牌）近乎完美重建，并在此潜在空间上训练文本扩散模型实现高质量图像字幕生成。

**💡 创新点**

创新点包括：①利用内部层的注意力掩码在同一LLM中同时实现编码和解码；②在中间层提取潜在瓶颈，结合LoRA与可学习的加性Transformer codec，形成高保真连续表示；③通过KL正则化将潜在映射为标准正态分布，便于后续扩散生成；④在Gemma 270M轻量级模型上实现参数高效的自动编码器，兼顾重建质量与模型体积。

**🔧 技术方法**

技术手段包括：结构化注意力掩码（encoder/decoder分块），LoRA低秩适配，单层可学习的Transformer codec，KL正则化（潜在分布匹配及参考模型对齐），以及在冻结LLMAE潜在上训练的文本扩散模型（score网络为12层Transformer）。

**📊 数据集**

主要使用的语料库：1M Pile‑uncopyrighted、400K C4‑realnewslike，共计140万样本；图像字幕实验则基于VLV‑6M数据集（6M LAION‑Aesthetic子集）。

**📈 对比分析**

与现有自动编码器（ICAE、COSMOS）对比，LLMAE在BLEU‑4、PPL、BERTScore等指标上均取得近1.0或极低的PPL，显著优于对手；在图像字幕扩散任务中，LLMAE‑基线在VLM Judge、CapArena‑Auto等指标上接近大模型（如Gemini‑1.5‑Pro），而大多数轻量化对手（LLaVA‑1.5‑7B、Qwen2‑VL‑2B）表现低于LLMAE。

**⚠️ 局限性**

局限性包括：①仅在Gemma 270M上验证，需探索其他LLM族；②解码仍受顺序推理延迟影响，尽管编码高效；③目前仅支持最长1024令牌，未验证更长文本；④在追求高重建精度时潜在空间密集，可能需要更强的条件信号来支持生成。

---

## 157. Tail-Aware Geometry Learning for Conformal Ellipsoids

**arXiv ID:** 2609.27221 | [PDF](https://arxiv.org/pdf/2609.27221v1)

**作者:** Xiang Zhang `[一作]` `[通讯]` (Nanyang Technological University), Xiang Zhang (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种尾部感知的多元共形预测框架，通过学习椭圆形非一致性分数的度量矩阵实现更优的预测集合；

**💡 创新点**

核心创新在于将尾部敏感性与最终误覆盖率解耦，使用 CVaR 约束进行几何学习，形成凸优化并提供边界加权解释；

**🔧 技术方法**

采用 CVaR 的最小体积椭圆学习、凸优化求解、两分拆设计（估计集与校准集）以及传统分裂共形校准；

**📊 数据集**

在合成数据（二维高方差异质分布）和真实信号处理基准数据集（Gas、Protein、House）上验证；

**📈 对比分析**

与 Naive、Cov（马氏距离）和 MVCS（VaR 最小体积）方法比较，实验显示在保持 1-α 覆盖的同时，CVaR 方法实现了更小的集合体积和更低的尾部严重度，尤其在稀有高方差区间表现优异；

**⚠️ 局限性**

限制包括对高维或非椭圆形残差分布的推广有限，且仍需两分拆数据，可能导致样本效率降低；

---

## 158. Escaping Python Dependency Hell: A Hybrid Replay-and-Repair Pipeline for Python Dependency Resolution

**arXiv ID:** 2609.26952 | [PDF](https://arxiv.org/pdf/2609.26952v1)

**作者:** Veronica Poweska `[一作]` (Toronto Metropolitan University), Manar Alalfi `[通讯]` (Toronto Metropolitan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个混合型 Python 依赖修复流水线 PLLM+，先利用静态分析和已知配置重放，再用 PyPI 校验，最后回退到多代理 LLM 方案。

**💡 创新点**

引入“先重放后 LLM”的优先级顺序和多代理结构（Proposer/Critic），显著提高成功率并降低运行时。

**🔧 技术方法**

使用静态 AST 分析、解决方案数据库重放、PyPI JSON 校验、错误分类、Gemma 2 LLM、Proposer–Critic 多代理对话及知识图谱持久化。

**📊 数据集**

基于 HG2.9K 基准数据集，包含 2,891 条导致依赖失败的 Python 代码片段。

**📈 对比分析**

在同一 GPU 集群上与原始 PLLM 基线对同一数据集进行评测，成功率从 40.4% 提升至 51.9%，平均耗时从 368.7 秒降至 71.8 秒。

**⚠️ 局限性**

过度依赖竞赛提供的解决方案数据库，缺少匹配配置时 LLM 回退仅覆盖极少案例，难以推广到全新或未包含在数据库中的依赖场景。

---

## 159. Divide and Doubt: Diverse Distributed Poisoning for Retrieval-Augmented Generation

**arXiv ID:** 2609.27090 | [PDF](https://arxiv.org/pdf/2609.27090v1)

**作者:** Tianhao Chen `[一作]` (Duke University), Neil Zhenqiang Gong `[通讯]` (Duke University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在检索增强生成（RAG）系统中设计了一种名为Divide and Doubt的多文档语料库投毒攻击，目标是让模型给出攻击者预选的错误答案而不是参考答案。

**💡 创新点**

创新点在于（1）通过风格账本和支持框架在整个投毒集合中显式规划并维持文体和论证的多样性；（2）引入一条“Doubt”文档，对引用答案的证据提出怀疑并提供与攻击目标一致的交叉检验；（3）对句法、语义与词汇层面实施严格的重叠约束（ROUGE‑L）和验证，确保投毒集合既多样又保持目标一致。

**🔧 技术方法**

使用的技术包括：风格与支持规划账本、冻结的验证器、基于ROUGE‑L的词汇重叠控制、检索适配（黑盒下的检索多样化和白盒下的HotFlip前缀优化）、以及后检索防御过滤（如TrustRAG、SeCon‑RAG）。

**📊 数据集**

实验数据集为HotpotQA和Natural Questions（NQ）两大开放域问答数据集。

**📈 对比分析**

通过严格ASR（Attack Success Rate）指标与多个基线攻击（PoisonedRAG、CorpusPoisoning、PromptInjection等）以及九种RAG防御配置（Vanilla、InstructRAG、TrustRAG、SeCon‑RAG、RAGuard、RobustRAG、Astute‑RAG、ReliabilityRAG、RAGDefender）进行对比。实验结果显示，在B=5、K=5的设置下，Divide and Doubt在无防御和所有防御配置上均能匹配或超过先前攻击，尤其在聚类与冲突检测防御（TrustRAG、SeCon‑RAG）中提升约26–45个百分点；整体平均ASR从PoisonedRAG的约53%提升至约67%（黑盒）或68%（白盒）。

**⚠️ 局限性**

局限性包括：实验仅在100道HotpotQA与NQ子集、三款开源LLM（Llama、Qwen、Mistral）上进行，未覆盖更大查询集、闭源模型或最新防御；且对防御的评估受计算与API预算限制，可能无法完全代表实际部署环境。

---

## 160. HiRE: Hindsight Reward Editing for Policy Finetuning

**arXiv ID:** 2609.27068 | [PDF](https://arxiv.org/pdf/2609.27068v1)

**作者:** Haoyi Niu `[一作]` (University of California, Berkeley), Koushil Sreenath `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

该论文提出一种无训练奖励编辑框架 HiRE，利用机器人训练过程中的成功与失败轨迹对比，动态校正基础视觉模型产生的奖励，提供稠密、控制感知的奖励信号。

**💡 创新点**

HiRE 的创新点在于通过在 vMF 核空间中计算成功与失败状态的对数密度比，并将其以潜在奖励形式融入奖励塑形，实现无训练、控制感知的奖励校正。

**🔧 技术方法**

使用了非参数核密度估计、Von Mises–Fisher 核、潜在基奖励塑形、正负样本缓冲区以及与 DICE‑RL 等强化学习算法的无缝集成。

**📊 数据集**

在 RoboMimic、MimicGen 两大仿真基准以及 I2RT YAM 真实机器人平台上进行实验，使用这些平台的演示数据构建基准政策并进行微调。

**📈 对比分析**

与稀疏奖励、RoboMeter、TOPReward、GCR 等基线相比，HiRE 在多阶段、接触丰富和高精度任务中显著提升成功率（最高可达 3 倍以上），并在样本效率和训练稳定性上优于其它方法。

**⚠️ 局限性**

该方法仅依赖成功/失败信号，未充分利用中间控制信息；缓冲区构建策略的鲁棒性有限，且在极端 OOD 场景下仍可能需要进一步改进。

---

## 161. Intelligence Across Embodiments

**arXiv ID:** 2609.27095 | [PDF](https://arxiv.org/pdf/2609.27095v1)

**作者:** Bo Ai `[一作]` (Stanford University), Hao Su `[通讯]` (Sudo AI GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出通过学习跨不同机器人躯体的经验来实现通用体化智能，强调在训练中引入多样化躯体、学习表征并制定更细致的转移评估方法。

**💡 创新点**

创新点在于：①将躯体多样性作为与任务、环境并列的缩放维度；②主张从工程化对齐转向通过学习自发发现通用表征；③提出针对转移难度的系统评估框架，区分零样本、少样本、行为层级与策略层级转移；④展望自适应与自进化机器人与学习的协同演化。

**🔧 技术方法**

核心技术包括：跨躯体强化学习/监督学习；利用模拟生成多样化可机械可行的躯体；在模型中加入显式躯体描述或通过交互推断躯体属性（如上下文学习、显式描述网络）；以及零样本/少样本适配策略。

**📊 数据集**

文中未给出具体公开数据集，主要依赖：①实验室收集的多种末端执行器与工具的演示数据；②基于仿真的可变形躯体数据集；③现有可用于跨形态泛化的仿真基准（如Meta-World、DMControl等）。

**📈 对比分析**

比较方法主要是对比传统工程化对齐方案与学习得到的通用表征在多样化躯体上的适配能力；实验显示：①在多样化末端执行器训练后对新手臂/工具的适配速度提升；②仿真训练实现零样本迁移至未见躯体的行走/操控任务；但缺乏统一性能指标与大规模对比实验。

**⚠️ 局限性**

局限性包括：①收集真实多样化躯体数据成本高；②仿真到真实的迁移缺失可靠性；③模型架构在处理巨大躯体空间时的可扩展性尚未验证；④评估体系仍不够细致，缺乏统一的转移难度度量与基准。

---

## 162. Which Objectives Need a Dial? Predicting Objective Conflict and Covering Trade-offs in Steerable Pluralistic Alignment

**arXiv ID:** 2609.26929 | [PDF](https://arxiv.org/pdf/2609.26929v1)

**作者:** David Tsoi `[一作]` (University of Stuttgart), Esra Dönmez `[通讯]` (University of Stuttgart)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多目标直接偏好优化（MODPO）在多元对齐中的可调性，探讨何时能一次满足两目标并评估如何在不重新训练的情况下覆盖整个权重连续体。

**💡 创新点**

①用预训练阶段的“冲突率”和“SFT参考模型得分相关性”两个低成本指标预测MODPO能否同时提升两个目标；②发现AI标注数据中的长度与重复问题会显著扭曲奖励模型得分，导致指标失效；③提出“最近模型选择”和“参数合并”两种快速复用方法，可在一定程度上逼近直接训练。

**🔧 技术方法**

技术包括：直接偏好优化（DPO）、多目标直接偏好优化（MODPO）、LoRA低秩适配、奖励模型ArmoRM和Beaver、以及基于权重的参数合并与最近模型投射。

**📊 数据集**

数据集：HelpSteer（7B Llama-2，包含帮助性、准确性、连贯性、复杂度、冗长度），UltraFeedback（7B Alpaca，包含帮助性、诚实性、指令遵循、真实性），PKU‑SafeRLHF‑10K（7B Alpaca，用于安全性权衡）。

**📈 对比分析**

比较方法：在每个目标权重下训练MODPO模型，然后用奖励模型评估其对两目标的提升；对比直接训练模型与“最近模型选择”与“参数合并”在目标权重上的分数差距。结果表明：在HelpSteer中，冲突率低且相关性高的目标对如准确性、连贯性可共同提升；在UltraFeedback中，指标失效但在控制长度/重复后表现更好；参数合并在平均分数差距上优于最近模型，但仍未能完全逼近直接训练。

**⚠️ 局限性**

局限性：①仅考虑两目标，未扩展到多目标；②所有实验单种seed，缺乏方差估计；③评分完全基于奖励模型，未引入独立人类或LLM评估；④长度/重复的控制仅在重解码时做过一次，未系统探究其影响；⑤只在7B模型上验证，结果对更大模型或其他架构可能不适用。

---

## 163. Improving Service Availability in KubeEdge-Based Architectures Using Lightweight Intrusion Detection

**arXiv ID:** 2609.27052 | [PDF](https://arxiv.org/pdf/2609.27052v1)

**作者:** Harrol Ndjeudji Kuibou `[一作]` (Laval University), Mohamed Aymen Saied `[通讯]` (Laval University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了KubeEdge系统中IoT容器镜像的安全漏洞对服务可用性的影响，并提出了一种轻量级入侵检测规则集RIDRS，以提升边缘计算环境的可用性。

**💡 创新点**

创新点在于设计了针对资源受限边缘节点的规则基IDS（RIDRS），在保持极低CPU/内存开销的前提下实现主动检测与自愈，并在恶意容器注入与代码注入场景下实现零停机。

**🔧 技术方法**

使用的技术包括KubeEdge、Docker、Falco/自定义监控代理、Trivy漏洞扫描、HPING3/网络模拟、NTP同步等。

**📊 数据集**

采用了500个手工验证的标记为IoT的容器镜像，并利用Trivy对其CVE进行扫描，构成实验数据集。

**📈 对比分析**

通过对比基线无IDS与引入RIDRS的三种攻击场景（代码注入、恶意Pod、DoS），测量时间到影响、检测反应、干预、恢复与停机时间。结果显示在代码注入与恶意Pod场景下停机时间降至0，检测延迟1–30秒，整体可用性显著提升；DoS场景仍需网络层防御。

**⚠️ 局限性**

实验仅在Raspberry Pi单节点测试环境完成，未覆盖大规模异构部署；DoS攻击仍无法完全缓解；规则集需手工编写，缺乏自适应更新机制。

---

## 164. From PyTorch to the NPU: LLM-Agent-Driven Model Conversion Across Heterogeneous Inference Runtimes

**arXiv ID:** 2609.27249 | [PDF](https://arxiv.org/pdf/2609.27249v1)

**作者:** Jianhao Su `[一作]` (Qualcomm Technologies, Inc.), ShengTing Huang `[通讯]` (Qualcomm Technologies, Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于LLM代理的“技能+验证循环”框架，实现单模型到多种边缘推理后端（OpenVINO、RKNN、TensorRT、ONNX Runtime QNN）的自动化部署。

**💡 创新点**

创新点在于将AIPC方法从单一QAIRT扩展到多种运行时，构建统一的技能模块、ONNXWrapper布局适配层，并系统评估了结构化知识注入对代理偏差的影响。

**🔧 技术方法**

使用技术包括大型语言模型代理、工具调用接口、ONNXWrapper、Olive优化工具、NNCF量化、AIMET，以及各自后端的命令行工具与API。

**📊 数据集**

采用的实验数据集为两个计算机视觉模型：Real‑ESRGAN（图像超分）和YOLO‑World（多模态目标检测），并在对应的推理设备上进行验证。

**📈 对比分析**

通过对比两组实验（含/不含技能模块）评估成功率、部署时间和人工干预次数，实验显示含技能的代理实现了8/8成功率，平均时间19.5min，人工干预3次；缺失技能时仅60%成功，平均时间19.8min，干预2次。

**⚠️ 局限性**

局限性包括仅测试两种模型，缺乏更复杂Transformer/LLM架构的验证；跨平台运算符修复与量化优化未系统比较；以及代理漂移和hallucination仍需更严格的验证机制。

---

## 165. What Converges in the Platonic Representation Hypothesis? Structure over Geometry

**arXiv ID:** 2609.27252 | [PDF](https://arxiv.org/pdf/2609.27252v1)

**作者:** Junwon You `[一作]` (KAIST), Jae-Hun Jung `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种2×2框架，分别在局部与全局尺度下独立比较关系结构和度量几何，以检测模型表示的收敛性。

**💡 创新点**

通过将局部互k近邻(mKNN)与全局H₀骨架重叠相匹配，并引入可调距离一致性阈值，系统区分了结构尺度与比较对象的影响，发现关系收敛在全局尺度同样显著。

**🔧 技术方法**

使用mKNN、H₀骨架重叠、距离加权版本（可调τ）、余弦相似度版本以及基于局部协方差的黎曼度量近似等技术，配合聚合感知置换校准。

**📊 数据集**

在视觉-文本领域使用WIT子集的1024对图文样本，对204对模型（12语言模型×17视觉模型）进行评估；在视频-文本领域再扩展到143对模型。

**📈 对比分析**

校准后的mKNN与H₀重叠在局部和全局尺度上均显示随模型容量提升而显著收敛；然而，随着距离一致性阈值变严格，度量几何的收敛逐步减弱，且高阈值下全局收敛显著低于局部。

**⚠️ 局限性**

仅考虑H₀层面的全局关系、固定关系集合以及单一黎曼近似，未探讨更丰富的拓扑结构、不同度量一致性定义或其他几何近似，可能限制了结论的普适性。

---

## 166. Realize What Matters: Principled Context Representation for Large-Scale Reasoning

**arXiv ID:** 2609.27173 | [PDF](https://arxiv.org/pdf/2609.27173v1)

**作者:** Michael Theologitis `[一作]` (University of Washington), Dan Suciu `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

构建一种基于认知理论的“关联性实现”框架，利用任务特定且逐步的上下文提取与灵活结构化两阶段流程（随后通过编码代理推理）来生成高质量的大规模上下文表示，并在此基础上完成推理任务。

**💡 创新点**

提出了三项设计原则：①任务特定且增量式的相关性提取；②依据提取的相关性与任务需求动态选择最合适的结构化方式；③将提取的相关性与结构化结果相结合形成可导航的表示。该框架实现了前所未有的表示构建可解释性和通用性。

**🔧 技术方法**

核心技术包括：递归任务聚焦摘要（多轮提取相关性）；结构化 schema 生成与填充；编程式推理（Python REPL 代码生成）以及大型语言模型作为推理与生成工具。

**📊 数据集**

在两个跨领域长文本推理基准上评测：CorpusQA（需要整合数百篇互相关联的文档）与 Loong（覆盖教育、金融、房地产、学术、法律等多种推理范式）。

**📈 对比分析**

与九个先进基线（RAPTOR、ReadAgent、MemAgent、HippoRAG2、LinearRAG、StructRAG、CodeAgent、RLMs、A‑RAG）对比，本文方法在 CorpusQA 上提升 19.88pp，在 Loong 上提升 8.42pp；在所有子域均保持最高准确率，并在使用较小模型时仍能超越 Frontier‑级模型，且成本约 3.7×‑3.9× 更低。

**⚠️ 局限性**

局限性包括：①需要手工设计多轮摘要与结构化的 prompt，敏感度高；②多轮提取与结构化增加推理时间与算力消耗；③在极大规模或实时场景下的可扩展性尚未完全验证；④对非文本/多模态信息的支持有限。

---

## 167. Temporally Ordered Region-Token Mamba with Logit-Space Diffusion for Remote Sensing Change Detection

**arXiv ID:** 2609.27149 | [PDF](https://arxiv.org/pdf/2609.27149v1)

**作者:** Anuvab Sen `[一作]` (Georgia Institute of Technology), Yixin Zhang `[通讯]` (Duke University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 BMD-CD，结合时序有序区域 Token 的状态空间建模和 logit 空间扩散细化，专为高分辨率遥感变化检测设计。

**💡 创新点**

创新点在于：Bitemporal Ordered Mamba Operator（BOMO）实现线性复杂度的跨时序长距离交互；Orthogonal Feature Disentanglement（OFD）通过旋转分离实现变化与不变特征的解耦；以及在 logit 空间进行少步条件扩散解码，显著提升边界细化。

**🔧 技术方法**

使用 Swin‑T 小型骨干、Mamba 状态空间模型、OFD 旋转分离、DDIM 条件扩散、LightMerge 轻量级特征融合、全景多尺度解码等技术。

**📊 数据集**

主要数据集包括 LEVIR-CD、WHU-CD、DSIFN-CD、CDD、S2Looking；并在 ValaisCD 与 B‑FLAIR‑test 进行零样本迁移验证。

**📈 对比分析**

相较于 CNN、Transformer、Mamba 与扩散基方法，BMD-CD 在 LEVIR-CD、WHU-CD、DSIFN-CD、CDD 的 F1 分别达到 93.7%、96.0%、97.8% 与 99.0%，3px Boundary‑F1 在 LEVIR-CD 与 WHU-CD 分别提升至 87.7% 与 91.4%；推理时间 47 ms、算力 32.09 GFLOPs，兼具高精度与适中效率。

**⚠️ 局限性**

局限在于仍需较高算力，扩散细化步骤略增推理时延；对极端尺度、多时相或非典型场景的鲁棒性待进一步验证，且模型规模可进一步压缩。

---

## 168. When LLM-Based User Profiling Adds Value in Production Streaming Recommendation

**arXiv ID:** 2609.27183 | [PDF](https://arxiv.org/pdf/2609.27183v1)

**作者:** Milad Sabouri `[一作]` (DePaul University), Shaghayegh Agah `[通讯]` (Comcast Technology AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统比较了四种语义用户画像策略（聚合型与LLM生成型，且是否进行短期/长期拆分）在真实流媒体推荐场景中的效果

**💡 创新点**

发现用户行为类型决定是否使用LLM画像：对探索型用户可提升精度，对常规用户则无优势；同时LLM画像导致覆盖率与新颖性下降

**🔧 技术方法**

聚合型采用SBERT嵌入均值；LLM生成型使用大语言模型生成文本摘要后SBERT编码；两者结合注意力融合短期/长期分量

**📊 数据集**

基于某大型流媒体平台的10,000用户样本，2.4万条目，完整标题+描述的英文文本

**📈 对比分析**

离线评估对比Recall@10/100、NDCG、HitRate、Diversity、Coverage、Novelty，LLM画像在探索用户上Recall+18%但覆盖率下降80%，在常规用户上则表现差于聚合

**⚠️ 局限性**

实验仅在单一内容域、离线设置，未验证线上效果；LLM生成成本高，缺乏可部署的探索性检测模型

---

## 169. Memory That Changes Action Is Not Memory That Guides It: Counterfactual Auditing of History-Conditioned Robot Policies

**arXiv ID:** 2609.27247 | [PDF](https://arxiv.org/pdf/2609.27247v1)

**作者:** Jiajie Zhang `[一作]` (Hong Kong University of Science and Technology), Changhao Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并实现了Counterfactual Memory Audit（CMA）评估协议，能够在冻结的机器人控制器上对记忆的决策影响进行细粒度的“感知别名”交叉检查，并在仿真和双臂物理平台上验证其有效性。

**💡 创新点**

创新点在于：①将两个合法历史在同一观测下交叉，独立测量记忆敏感性、被记忆引导的分支选择、匹配世界物理价值以及对每一对的可靠性；②设计四级评估体系（Sensitivity、Choice、Matched‑World Value、Reliability）并结合评估者世界，实现对记忆在单一决策时刻的因果归因；③通过历史替换和受保护的记忆片段恢复演示记忆对闭环控制的实际影响。

**🔧 技术方法**

主要技术包括：冻结策略查询、共用随机种子保证可比性、评估者世界（oracle+价值函数）、执行动作距离度量（RMSE）、正负奖励签名、对数化恢复成本评估以及配对级别的自举置信区间。

**📊 数据集**

使用的数据集与任务主要来自RMBench/RobomemArena中的Put Back、Rearrange、Swap三类物料搬运任务；另外在双臂PiPER硬件平台上重现相同任务；记忆模型为Mem‑0和BridgeVLA++两个已冻结的检查点。

**📈 对比分析**

与传统的任务成功率和动作变化度量相比，CMA在同一政策上揭示了显著的失配：例如Mem‑0在Put Back任务上平均成功率0.729但只有31% pair 完全可靠；Swap任务每对动作均变化但分支选择保持不变；在物理平台上，尽管记忆对动作方向有影响，但完整成功率仅为12.5%。CMA还能证明记忆替换能将成功率从0%提升至100%，并用4096字节受保护片段恢复成功率提升约38.9个百分点。

**⚠️ 局限性**

局限性包括：①CMA仅适用于冻结的策略，无法直接用于在线学习或动态策略更新；②需要能够重现完全相同的观测状态，对硬件随机性和闭环噪声敏感；③只捕捉单一决策时刻的记忆因果，不涵盖长时序决策的整体依赖；④评估成本高，需要多次对称查询与评估者世界的离线计算。

---

## 170. A Hybrid Rule-Based and AI-Augmented Framework for Automatic Failure Recovery in DevOps Deployments

**arXiv ID:** 2609.26838 | [PDF](https://arxiv.org/pdf/2609.26838v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 171. Cross-Modal Contrastive Learning from Histopathology and CT for Automated Renal Cell Carcinoma Grading

**arXiv ID:** 2609.26920 | [PDF](https://arxiv.org/pdf/2609.26920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 172. What Makes a Terminal-Bench Task Hard? Separating Genuine Hardness from Fake-Hardness on an Adjudicated Agentic Corpus

**arXiv ID:** 2609.26826 | [PDF](https://arxiv.org/pdf/2609.26826v1)

**作者:** Edward Lue Chee Lip `[一作]`, Ivan Bercovich `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

审计了前沿基准中所有失败任务的证据，区分出可认证未解、破损、基础设施受限、可被绕过和未证实的任务。

**💡 创新点**

提出有序有效性筛选并给出五种证据级别标签，使所有失败分数能更细致解释。

**🔧 技术方法**

结合任务包、参考/空解控制、轨迹、遥测、审查记录，使用验证器完整性和严格绕过检测等技术。

**📊 数据集**

使用冻结的 Terminal-Bench 3 / Frontier‑Bench 0.1 记录（1081 PR、639 评分任务、28,801 试验、$105,933 费用）。

**📈 对比分析**

对通过参考解的任务采用最小输出 token 成本作为难度阈值比较，区分广泛真实难度任务与可处理任务，AUC 0.75。

**⚠️ 局限性**

证据不完整、单次参考运行不足、可执行环境不完全可复现、标签依赖人工判断，且仅适用于该快照。

---

## 173. Same evidence, different judgments: Evidence noncommutative in vision/speech-text conflicts

**arXiv ID:** 2609.26986 | [PDF](https://arxiv.org/pdf/2609.26986v1)

**作者:** Zhuoyun Li `[一作]`, Yi Dong `[通讯]` (University of Liverpool)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对多模态大型语言模型在文本与图像/语音冲突情况下，研究证据顺序对模型依赖性的影响，发现把图像或语音放在文本之后会显著提高模型对该感知证据的依赖；

**💡 创新点**

提出了“跨模态证据非可交换性”这一概念，并设计了仅交换感知与文本位置、保持其余输入不变的配对比较协议，系统量化了顺序对文本偏倚的贡献；

**🔧 技术方法**

利用提示工程、对齐顺序的配对输入、对数赔率交换分数（log‑odds swap score）以及配对自助法置信区间等技术评估模型行为；

**📊 数据集**

使用自然图像数据集GQA、Visual Genome、VQAv2，以及语音数据集Fluent Speech Commands（FSC）和SLURP，对 240 张图像/240 条语音进行人工构造的冲突/匹配/无关文本；

**📈 对比分析**

比较两种输入顺序（P→T 与 T→P）下模型的附件支持答案率，结果显示附件放后可提升 9.4–28.8 百分点（视觉）和 10.4–24.0 百分点（语音）的准确率；对比控制条件显示这一提升并非仅由顺序引起的共性因素；

**⚠️ 局限性**

仅考虑单一感知源与文本，未探究多源组合、不同模态交互的细节；内部机制（位置编码、注意力分布等）未被深入解析；实验受限于所选模型与数据集，尚未验证更广泛的任务与模型。

---

## 174. Stage-Supervised Latent Reasoning for Single-Shot JavaScript Deobfuscation

**arXiv ID:** 2609.27058 | [PDF](https://arxiv.org/pdf/2609.27058v1)

**作者:** Rong Feng `[一作]` (Pennsylvania State University), Suman Saha `[通讯]` (Pennsylvania State University)

**通讯引用:** 988 | [OpenAlex ID](https://openalex.org/A5029877298)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练一个单 shot JavaScript 去混淆模型，利用 deterministic 去混淆工具的多阶段中间状态作为监督，结合 Coconut 潜在推理框架，学习从原始混淆代码直接生成可读、可执行代码。

**💡 创新点**

创新点在于把多阶段中间结果转化为潜在空间的连续监督，使模型在训练时能内部化多步变换过程，但在推理时保持一次性生成；同时使用 Coconut 实现隐式推理而非显式文本中间步骤。

**🔧 技术方法**

使用 Llama 3.2‑3B 作为基础模型，结合 Coconut 连续潜在推理，利用 Webcrack deterministic pipeline 提取 5 阶段的中间代码，并通过 labeled diff 进行监督；推理时加入固定数量的 latent thought tokens。

**📊 数据集**

数据集为 JsDeObsBench（1,302 程序），在实验中选取 100 程序（60 训练、20 验证、20 测试），每个程序从 P1 到 P5 记录 5 个中间阶段。

**📈 对比分析**

与 zero‑shot 基线模型和仅进行输入输出微调的 No‑CoT 模型比较，使用语法有效性与语义正确性两项指标；Coconut 模型达到 50% 语法有效性、80% 语义正确性，明显优于其他模型（No‑CoT 15%/0%，基线 25%/40%）。

**⚠️ 局限性**

局限性：评估仅在 20 个程序的小规模测试集上完成，语义验证部分依赖手工检查；缺乏大规模自动化评测；仅测试 3B 模型，未验证更大模型或不同混淆变体；依赖 deterministic pipeline 的阶段信息，未探讨其在完全未知混淆场景下的泛化能力。

---

## 175. ItColBERT: An Italian-Specialised Late-Interaction Retriever

**arXiv ID:** 2609.26856 | [PDF](https://arxiv.org/pdf/2609.26856v1)

**作者:** Enrico Nello `[一作]` `[通讯]`, Enrico Nello

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在仅使用一台消费级GPU（RTX 3090）训练了一个面向意大利语的多向量检索模型ItColBERT，并公开了权重、训练代码及完整实验记录。

**💡 创新点**

主要创新在于：①从已有检索能力的检查点直接初始化（ColBERT‑Zero思路），②采用单一教师的跨编码器蒸馏并以检索指标为选择标准；以及发现文档截断是长文档检索瓶颈，推断时分块（chunking）即可显著提升性能，训练长度扩展无效。

**🔧 技术方法**

技术包括：ModernBERT基线、PyLate框架、监督对比学习（Contrastive Loss）、单教师KL蒸馏、MaxSim相似度计算、以及推理时的文档分块策略。

**📊 数据集**

使用了意大利语的四大检索基准：MLDR‑it（长文档、离域）、mMARCO‑it（短文档、域内）、MIRACL‑ita、SQuAD‑ita；训练时还混合了mMARCO‑it、Wiki检索对、MIRACL‑ita和SQuAD‑ita的样本。

**📈 对比分析**

与同类多向量检索器（mLateOn、jina‑colbert‑v2、ColBERT‑XM等）以及大规模密集检索模型（bge‑m3、multilingual‑e5）进行对比；在所有四个基准上，ItColBERT在除BM25外均表现优于其他多向量模型，且在MLDR‑it上通过分块与BM25基本持平。整体参数量比大多数基线低，训练成本仅约14.5 GPU‑小时。

**⚠️ 局限性**

局限性包括：①仅单一随机种子训练，可能存在种子方差；②MLDR‑it基准样本量小且离域，结果易受噪声影响；③文档分块提升需在推理时额外计算，增加内存和时间开销；④评测未覆盖混合检索（如与BM25融合）和更大规模长文档检索场景。

---

## 176. Fast Direction-Conditioned Reachability for Motion Prediction Under Model Uncertainty

**arXiv ID:** 2609.27077 | [PDF](https://arxiv.org/pdf/2609.27077v1)

**作者:** Hrishav Das `[一作]` (University of Illinois Urbana-Champaign), Melkior Ornik `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种方向条件可达性方法，针对线性系统的状态与输入矩阵不确定性，通过在给定查询方向下选取单一可行模型，计算该模型的可达集，只关注该方向上的可达范围；

**💡 创新点**

在方向条件下分阶段最大化支撑函数：先仅利用初始集选取最优状态矩阵，再针对输入集选取最优输入矩阵；这种方法显著减少了对全不确定模型族的计算，且在该方向上的可达范围与完整不确定集合相差不超过5%；

**🔧 技术方法**

利用支撑函数、矩阵锥（matrix zonotope）表示不确定性；对状态矩阵使用网格搜索+局部优化，对输入矩阵枚举极点实现凸优化；使用CORA工具箱计算线性可达集；实验基于线性化两轮车模型；

**📊 数据集**

本文使用自行仿真数据：对线性化两轮车模型的雅可比矩阵与输入增益引入10–18%的不确定性，构建矩阵锥；未使用公开数据集，全部为模拟生成；

**📈 对比分析**

与CORA中对全不确定矩阵族的外部逼近比较，方向支持函数误差≤5%，且算法总耗时约为全不确定方法的三分之一；在闭环多车仿真中，每个车辆在0.4 s重规划周期内完成方向条件可达集预测，满足实时约束；

**⚠️ 局限性**

方法仅对单方向或少数方向有效；状态矩阵选择采用近似，未保证全局最优；仅适用于线性化模型，对高度非线性系统的适用性需进一步验证；在极端不确定性下可能出现较大误差。

---

## 177. Calibrating Reproduced Claims in Recommender Systems

**arXiv ID:** 2609.26975 | [PDF](https://arxiv.org/pdf/2609.26975v1)

**作者:** Alan Said `[一作]` (University of Gothenburg), Alan Said `[通讯]` (University of Gothenburg)

**通讯引用:** 2021 | [OpenAlex ID](https://openalex.org/A5040472816)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了声称校准（Claim Calibration）概念，并对五对原始–后续论文进行分析，提出了 Claim Evidence Profile（声称证据概要）作为结构化报告框架，以更细粒度地描述复现研究中哪些部分被支持、哪些未被测试；

**💡 创新点**

创新点在于：①将复现结果拆解为对单一声称的支持与否；②提出声称校准机制，让研究者能够明确指出复现结果支持原始声称的哪些子层面；③设计 Claim Evidence Profile，提供七字段结构化模板，方便作者、评审和读者快速把握复现研究的范围与结论；

**🔧 技术方法**

主要技术为质性编码与案例分析；作者依据 ACM 可重复性术语对原始与复现论文进行手工抽取、对照与归纳，形成声称校准与 Profile 的定义；

**📊 数据集**

涉及的推荐系统数据集包括 MovieLens‑1M、Pinterest、以及在复现研究中使用的其他公开数据集，但本文关注的是复现过程与声称关系，而非对数据集性能的评估；

**📈 对比分析**

通过对比原始声称与复现结果，本文展示了：①数值相同但排名保持；②数值差异仍可保持原始排名或改变结论；③不同评估指标和实验设置下支持程度不同。总体而言，复现并不等同于完整成功，声称校准可揭示细粒度支持与未测试区间；

**⚠️ 局限性**

局限性包括：①仅分析了五个案例，代表性有限；②所有编码由单一研究者完成，缺乏交叉验证；③未重新运行实验，仅依据论文与公开资源；④Profile 侧重声称层面，无法覆盖所有实现细节。

---

## 178. Are Stated Reasoning Steps Causally Load-Bearing?

**arXiv ID:** 2609.27038 | [PDF](https://arxiv.org/pdf/2609.27038v1)

**作者:** Abhiram Bhupatiraju `[一作]` (University of Texas at Austin), Rayan Nyaupane `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在模型激活层插入反事实激活，评估链式思维（CoT）推理的因果可信度，并与传统行为编辑测试进行对比。

**💡 创新点**

提出基于可预知反事实目标的因果负载率（CLB）度量，并展示行为测试往往高估因果可信度，尤其在易题上更甚。

**🔧 技术方法**

采用激活层级的残差流替换、行为文本编辑、以及多种对照实验（正向、后答案、随机位置、混淆目标）等技术。

**📊 数据集**

使用合成多跳检索任务（2–6 跳）构建的伪词知识库数据集。

**📈 对比分析**

在 Qwen3-4B 与 Qwen3-1.7B 两个模型上实验，CLB 率约 76–78%，行为测试率约 88%，差距随推理深度和模型规模变化，较小模型因果可信度随难度下降。

**⚠️ 局限性**

局限于合成单关系检索任务，未覆盖自然数学/代码推理；早期层插值可能与 token 识别替换重叠；仅评估小规模 Qwen3 系列模型。

---

## 179. LexLattice: Multilingual Extractive Summarization via Neural Cellular Automata on Document Hierarchies

**arXiv ID:** 2609.27032 | [PDF](https://arxiv.org/pdf/2609.27032v1)

**作者:** Sujay Uday Rittikar `[一作]` (University of Winnipeg), Sheela Ramanna `[通讯]` (University of Winnipeg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了LexLattice，一种基于文档层级的二维语义网格和二维神经元自动机（NCA）进行证据整合后再做提取的法律文本摘要方法。

**💡 创新点**

创新点在于将法律文件的层级结构显式化为二维网格，使得局部自洽更新能够跨段、跨章节传播重要性信号，从而用极少量可训练参数（≈1.8M）在冻结的多语言编码器上实现跨语言、跨文档的高质量摘要。

**🔧 技术方法**

主要技术包括：规则驱动的层级解析、冻结的mT5多语言编码器、掩码二维NCA迭代更新、残差读出评分以及可选的语言嵌入。

**📊 数据集**

使用数据集EUR‑Lex‑Sum，涵盖欧盟24个官方语言的法律法规与人工摘要，共计约2万份训练样本。

**📈 对比分析**

在所有语言上与多语言/跨语言基线（包括大规模指令调优模型）对比，LexLattice+RLOO在ROUGE‑1、ROUGE‑2、ROUGE‑L和BERTScore‑F1上均实现了最佳或接近最佳的性能，并在跨语言零样本转移上保持了近乎无损的表现。

**⚠️ 局限性**

局限性包括：依赖规则解析获取层级结构，可能在无明显结构标记的文本上失效；仅为抽取式摘要，受限于源文本长度与表达流畅度；与之比较的基线主要为LexRank，缺乏更广泛的抽象式或大模型对照。

---

## 180. NeuroRule: Making Black-Box Neural Networks Explainable through Rule-set Evolution

**arXiv ID:** 2609.26841 | [PDF](https://arxiv.org/pdf/2609.26841v1)

**作者:** Tapaswini Kodavanti `[一作]` (University of Texas at Austin), Risto Miikkulainen `[通讯]` (University of Texas at Austin)

**通讯引用:** 15573 | [OpenAlex ID](https://openalex.org/A5020441009)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究提出 NeuroRule 框架，通过进化学习将黑盒神经网络转化为可解释的规则集，实现了对神经网络知识的蒸馏。

**💡 创新点**

创新点包括：① 将神经网络作为进化目标进行规则蒸馏；② 引入规则简洁性（Condition Count）作为次要目标，实现可解释性与性能的双优化；③ 在原始训练数据不可用时，利用神经网络生成合成数据完成蒸馏。

**🔧 技术方法**

技术手段主要是基于 EVOTER 的线性规则集进化算法，使用 NSGA‑II 多目标遗传算法、三种交叉与三层级变异，以及以神经网络预测为目标的适应度函数。

**📊 数据集**

实验数据集为三种医学分类数据集：Breast Cancer Wisconsin、Heart Failure Prediction（多来源心脏病数据）和 Diabetes Health Indicators（CDC 数据）。

**📈 对比分析**

与基准神经网络和 EVOTER 直接训练的规则集比较，NeuroRule 在多数情形下在 ID（训练分布）上与 NN 相近或略逊，在 OOD（外部分布）上往往超过 NN，并且在多目标搜索下得到的规则集显著更简洁（Condition Count 降低），性能甚至更优，显示出规则蒸馏与简洁性约束的正向效果。

**⚠️ 局限性**

局限性包括：对高维或噪声较大的数据集收敛性不稳定；进化搜索受随机性影响，需要更大种群或更高级变异算子；规则集目前仅支持确定性输出，无法处理概率或连续输出；对不同 NN 架构的敏感度未充分评估；合成数据蒸馏导致的性能下降表明对 NN 预测噪声的鲁棒性不足。

---

## 181. Recognized but Not Produced: A Generation Benchmark for Culturally Specific Kinship Terms

**arXiv ID:** 2609.26942 | [PDF](https://arxiv.org/pdf/2609.26942v1)

**作者:** Sahil Pardasani `[一作]` (Pennsylvania State University), Madhusudan Singh `[通讯]` (Pennsylvania State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个跨语言的 kinship 生成基准，使用 Hindi、Tamil 和 Korean 三种非西方语言，在生日祝福和婚礼邀请两类交流场景中，让五个开放权重 LLM 生成具有文化特定的亲属称谓，并与四选项支持的选择基准进行对比。

**💡 创新点**

创新点在于将传统的多项选择识别任务转向生成任务，突出模型在自然语境中完整输出正确亲属词的难度；同时揭示了“评估格式差距”与“语言/亲属线性差异”两种新型错误来源。

**🔧 技术方法**

采用标准 LLM 推理（温度 0.7、top-p 0.95）、规则式词汇匹配器、统计抽样置信区间以及对照式四选项选择机制，对模型输出进行量化评估。

**📊 数据集**

数据集包括 25 条亲属关系的三语映射表（81 条语言特定条目），共 300 条独特提示（146 条 L1、154 条 L3），每条提示执行三次生成，产生 4,500 条完整消息；另外 77 条四选项测试项。

**📈 对比分析**

与四选项选择相比，生成准确率显著下降：GPT-OSS-120B 选择 90.67% 正确但生成仅 36%；Llama-3.3-70B 选择 77.92% 但生成 24.24%。总体生成准确率在 GLM-5.1（72.29%）至 Llama-3.3-70B（24.24%）间分布，表明评估格式与语言/亲属因素对性能影响巨大。

**⚠️ 局限性**

局限性包括：仅覆盖三种语言且仅单一语域；基于词条出现的匹配器无法判断语义正确性、遗漏或多义；缺乏系统的人类审核来估计误判率；四选项基准仅测试两模型；缺少平衡的 distractor 与答案位置；未对模型训练数据做因果分析。

---

## 182. Listening and Mirroring: The Effects of Verbal Attunement and Behavioral Mimicry on Social and Empathic Perceptions of Embodied AI Agents in VR

**arXiv ID:** 2609.27246 | [PDF](https://arxiv.org/pdf/2609.27246v1)

**作者:** Nathalia Gomez `[一作]` (Drexel University), Tiffany D. Do `[通讯]` (Drexel University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一种可在虚拟现实中进行实时对话的具身AI辅导员，并在20名参与者中通过2×2 受试者内设计分别测试了语言同步（Verbal Attunement）和行为模仿（Behavioral Mimicry）的影响；

**💡 创新点**

创新点在于首次将语言情感适配与非语言行为模仿同步应用于实时VR对话场景，系统化评估两种同步策略的交互作用，并探讨了性别对模仿效果的调节作用；

**🔧 技术方法**

使用技术包括GPT‑4o mini生成实时对话、LiveKit进行音频流、Deepgram ASR/TTS、OVRFaceExpressions与OVR LipSync实现面部表情与口型同步、Unity3D与Ready Player Me头像实现虚拟化身，结合Meta Quest Pro头显进行头部倾斜与姿态检测及即时模仿；

**📊 数据集**

未使用公开数据集，所有实验数据来自参与者在对话中的实时语音转录、行为检测以及问卷反馈；

**📈 对比分析**

通过受试者内2×2 ANOVA比较条件效果，显著发现语言同步显著提升同理心评分（F=7.43, p=0.013），模仿同步对人类感知呈边缘显著性（F=4.31, p=0.052），并通过相关分析探索模仿持续时间与同理心、积极性、人类感知之间的关系；系统整体延迟约1.54秒；

**⚠️ 局限性**

局限性包括样本量仅20人、模仿曝光不均、模仿仅限于面部表情、头倾斜与姿势的离散类别、未控制回应长度、语音结束检测导致的交互切断、性别差异仅为探索性、未探讨种族/身份等因素对同理心的影响。

---

## 183. A Systematic Benchmark of Explainable Methods for Temporal Attribution in Sequential Recommendation Systems

**arXiv ID:** 2609.27201 | [PDF](https://arxiv.org/pdf/2609.27201v1)

**作者:** Akash Pandey `[一作]` (Capital One), Pranab Mohanty `[通讯]` (Capital One)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对序列推荐系统中的后置可解释方法进行系统性基准测试，提出了一种双模型掩蔽度量来评估时间归因的可信度。

**💡 创新点**

创新点在于首次引入双模型掩蔽度量避免分布外问题，全面评估了十种梯度、扰动和注意力归因方法，并剖析了它们对序列长度、流行度偏差和鲁棒性的影响。

**🔧 技术方法**

主要技术包括梯度方法（Integrated Gradients、GradientSHAP、DeepLift 等）、扰动方法（LIME、KernelSHAP）以及注意力方法（Attention Tracing、Grad‑SAM），配合基于 CNN、Transformer、SASRec 与 BERT4Rec 的四种推荐骨干模型。

**📊 数据集**

实验使用 KuaiRand‑1k（短视频二分类）和 MovieLens‑1M（下一个项目预测）两个公开数据集，并在两者不同序列长度（L=100/1000 与 L=200）下进行评估。

**📈 对比分析**

通过 Pearson 相关系数 c_f 评估可信度、c_r 评估鲁棒性、c_p 评估流行度偏差，实验结果表明梯度方法（尤其是 Integrated Gradients 与 GradientSHAP）在所有设置下均显示出最强的可信度（c_f < –0.9）和鲁棒性，而注意力方法则对架构和序列长度高度依赖，表现不稳定。

**⚠️ 局限性**

局限性包括仅进行定量基准，未开展用户研究验证可解释性的实际价值；实验覆盖的模型与数据集有限，未来需扩展至更多领域和真实业务场景。

---

## 184. A Quasi-Direct-Drive Underactuated Asymmetric Hand for Dexterous and Efficient Grasping and Manipulation

**arXiv ID:** 2609.27240 | [PDF](https://arxiv.org/pdf/2609.27240v1)

**作者:** Benjamin Davis `[一作]` (University of California Berkeley), Hannah S. Stuart `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一款具有四指、11自由度、8可驱动位移的Berkeley QUAD Hand，采用QDD驱动的径向手指与单一大功率QDD驱动的被动桡侧手指耦合。

**💡 创新点**

通过将指尖功能分配给不同指节，采用机械智能耦合桡侧手指实现无额外驱动的被动闭合，兼顾抓握力与热效率。

**🔧 技术方法**

使用QDD（SteadyWin GIM3505-08 与 CubeMars AKE60-8）驱动、六杆链机械耦合、四杆链运动学模型、数字仿真与实验验证。

**📊 数据集**

采用Feix抓握分类（33种）与Kapandji手指扩展测试进行功能验证，并在实验平台上测量回驱动力与持续载荷热耗。

**📈 对比分析**

通过抓握分类获得29/33种抓握、10/11种Kapandji姿态、回驱动力约50g，热耗相比单独驱动下降96倍，证明其高灵敏度与能效。

**⚠️ 局限性**

实验仅覆盖静态姿态与单一位置的力学性能，未考虑动态控制、非线性连杆热失效、无触觉反馈，且部分零件采用3D打印，需进一步金属化与动态评估。

---

## 185. Physiologically Informed Digital Auscultation for Pneumonia Detection in Long-term Care Residents

**arXiv ID:** 2609.27222 | [PDF](https://arxiv.org/pdf/2609.27222v1)

**作者:** Nicholas Rasmussen `[一作]` (University of Washington), Tomoko Ito `[通讯]` (University of Tsukuba)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究利用多通道数字听诊记录与X光/临床诊断标签训练卷积神经网络，在老年长期护理居民中评估肺炎检测性能。

**💡 创新点**

创新点包括：①比较X光监督与临床诊断标签的模型鲁棒性，证明X光监督更稳定；②提出“训练全部、推理子集”通道选择策略，三四通道即可保持性能；③使用时间域Grad‑CAM与多种解释一致性验证模型关注真正的肺音特征。

**🔧 技术方法**

采用ARP‑N卷积架构、平均概率策略、多模态融合、Borda排名通道选择、时间域Grad‑CAM、STFT频谱特征、5×5患者级交叉验证及类不平衡加权等技术。

**📊 数据集**

使用185名日本长期护理老人数据集，包含73例肺炎、112例症状非肺炎，采集6个前胸听诊通道，并结合放射科胸X光和临床诊断标签作为监督。

**📈 对比分析**

通过5×5患者级交叉验证比较X光监督与临床标签下的听诊单一、临床+听诊融合模型。X光监督的听诊单一模型F1 0.729、准确率0.783、AUC 0.774；融合模型AUC 0.791、特异性0.867。三四通道模型保持F1≈0.736、准确率≈0.803，且方差更低。

**⚠️ 局限性**

局限包括样本量有限、单中心研究、仅使用前胸通道、X光标签仍有噪声、存在域漂移、缺乏外部验证，且尚未评估在不同人群或临床工作流中的真实效用。

---

## 186. LOCKR: A Hidden-State Trajectory-Guided Planner for Detecting and Repairing Stable-but-Wrong Lock-In in Diffusion Language Models

**arXiv ID:** 2609.27220 | [PDF](https://arxiv.org/pdf/2609.27220v1)

**作者:** Guoshenghui Zhao `[一作]` (Rochester Institute of Technology), Weijie Zhao `[通讯]` (Rochester Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于隐藏状态轨迹的动态推理修复方法 LOCKR，解决扩散语言模型在迭代去噪过程中出现的稳定但错误的锁定现象。

**💡 创新点**

创新点在于将隐藏轨迹视为规划状态，结合风险检测、目标化重掩码和轨迹感知验证，实现在推理过程中的可控计算分配与分支选择。

**🔧 技术方法**

核心技术包括隐藏状态随机投影、双向 GRU 轨迹编码、重掩码修复动作库、轨迹感知分支验证器与梯度提升选择器。

**📊 数据集**

实验使用 DiffusionGemma 和 LLaDA-2 两种扩散语言模型，在 MetaMathQA、Orca-Math 与 NuminaMath V2 三个数学推理基准上进行评估。

**📈 对比分析**

与仅基于表面指标、随机修复、自适应一致性等基线相比，LOCKR 在匹配基准上实现 2.0–5.4% 的绝对准确率提升，修复率 22%–41%，计算成本约 2.9–3.1 倍。

**⚠️ 局限性**

局限性包括固定的修复动作库导致修复空间受限、评估仅涵盖数学推理与两类 dLLM，且未探究不同硬件环境下的实际延迟。

---

## 187. Vision-Based Control of a Tether-Suspended Aerial Radiation Sensing Payload

**arXiv ID:** 2609.27219 | [PDF](https://arxiv.org/pdf/2609.27219v1)

**作者:** Ian Snider `[一作]` (University of California), Mark W. Mueller `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一种基于视觉的LQI控制器，用于调节悬挂式辐射检测器的轨迹。

**💡 创新点**

首次将下视相机跟踪红环、卡尔曼滤波与积分LQI控制相结合，直接控制负载位置。

**🔧 技术方法**

采用ArduPilot内核、下视相机、卡尔曼滤波、LQI+积分控制与手动引导接口。

**📊 数据集**

使用现场风速计测量与无人机实验数据，没有公开数据集。

**📈 对比分析**

与ArduPilot车辆参考基线对比，平移段轨迹误差下降约20%，整体RMSE降低。

**⚠️ 局限性**

在转角处易产生较大摆动峰值，且对大幅摆动的鲁棒性不足。

---

## 188. Who Acts, Who Knows, Who Answers? A Corpus-Assisted Discourse Analysis of Agency, Epistemic Responsibility, and Accountability in Generative AI Higher Education Research

**arXiv ID:** 2609.27184 | [PDF](https://arxiv.org/pdf/2609.27184v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 189. BoundaryMORPH: Budgeted Reranking via Active Set Selection for Diffuse Retrieval

**arXiv ID:** 2609.27213 | [PDF](https://arxiv.org/pdf/2609.27213v1)

**作者:** Eylon Caplan `[一作]` (Purdue University), Rashmi Gangadharaiah `[通讯]` (AWS AI Labs)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 BoundaryMORPH，一种利用高斯过程的预算感知文档选择方法，针对扩散查询在 Retrieval-Augmented Generation 中的 top‑k 选取问题。

**💡 创新点**

创新点包括：① 在高斯过程先验中嵌入 MIPS 百分位排名以利用检索先验；② 采用边界采集规则，仅在候选集边界进行 CE 调用，避免对已确定的 top‑k 进行浪费性评估；③ 将问题框定为已知容量的 top‑k 选取（level‑set）问题。

**🔧 技术方法**

使用技术：Gaussian Process 回归、MIPS 余弦相似度、vMF 核、边界式 UCB 采集、跨编码器 (Cross‑Encoder) 评分、单阶段重排序、BAGEL、RGS 等基线对比。

**📊 数据集**

实验数据集：NeuCLIRBench‑monolingual‑eng、Robust04、TravelDest，候选集均为前 10k 文档，均为开放式扩散查询。

**📈 对比分析**

与 BM25、单阶段重排、RGS、BAGEL 等基线在相同 CE 预算下使用 nCG@k 与 CE‑nCG@k 指标进行对比；BoundaryMORPH 在所有数据集、预算与上下文容量上均显著优于基线，最高提升约 5–8pp nCG@100。

**⚠️ 局限性**

局限性：需要顺序执行、对预算 B 的离散假设、对文档容量 k 的离散假设、在极大预算下计算复杂度为 O(B³N)，未针对实际 token 限制和多模态检索循环做直接优化。

---

## 190. Benchmarking Active Spot Selection for Cost-Efficient Spatial Transcriptomics

**arXiv ID:** 2609.27208 | [PDF](https://arxiv.org/pdf/2609.27208v1)

**作者:** Zheyu Zhu `[一作]` (University of Pennsylvania), Ruining Deng `[通讯]` (Cornell University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对空间转录组（ST）中激活点采样策略进行回顾性基准测试，比较四种标准主动学习策略与随机采样在不同预算下的表现。

**💡 创新点**

在共享预测器、采样计划和优化协议的统一框架下评估主动采样，首次系统比较多种不确定性和多样性驱动的采样策略对点级和空间级表达预测的影响。

**🔧 技术方法**

使用基于DenseNet‑121的ST-Net风格回归模型，结合Monte Carlo dropout、Temporal Output Discrepancy、CoreSet、TypiClust-inspired等四种主动采样方法。

**📊 数据集**

利用两个公开的ST数据集：HER2阳性乳腺癌（8例）和表皮鳞状细胞癌（4例），共计约2万+样本。

**📈 对比分析**

在5%、10%、30%和50%预算下，通过患者级交叉验证比较平均Pearson相关系数、表达聚类一致性（ARI）和Moran's I保真度。结果显示：在低预算（5%–10%）下大多数主动策略表现不及随机采样，预算提升至30%–50%时，平均提升很小且受数据集与评价指标影响。

**⚠️ 局限性**

局限包括：仅测试两种数据集与单一模型架构；训练时间与收敛不完全匹配；未考虑连续区域采样或专家指导；结果仅为描述性，未进行统计显著性检验。

---

## 191. Reliable Federated TinyML Deployment for IoT Security

**arXiv ID:** 2609.27202 | [PDF](https://arxiv.org/pdf/2609.27202v1)

**作者:** Younsoo Park `[一作]` (Pennsylvania State University), Peilong Li `[通讯]` (Elizabethtown College)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究将联邦学习与 TinyML 压缩技术结合，构建可在资源受限 IoT 设备上运行的轻量级入侵检测模型；

**💡 创新点**

创新点包括服务器协同余弦学习率调度显著提升攻击召回率、在联邦学习框架下实现多阶段压缩管线（BN折叠、知识蒸馏、结构剪枝、量化），并对量化感知训练在不同压缩力度下的效果进行系统分析；

**🔧 技术方法**

采用联邦学习（FedAvgM）+焦点损失、服务器协同余弦学习率调度，TinyML 压缩流程（BatchNorm折叠、知识蒸馏、结构剪枝、量化感知训练 QAT 与后训练量化 PTQ），并用 FGSM 评估对抗鲁棒性；

**📊 数据集**

使用 CIC-IDS2017 网络流量数据集进行训练与评估，另外考虑过 Bot‑IoT 与 TON_IoT 但因类别失衡未采用；

**📈 对比分析**

通过基线联邦模型与压缩模型对比，评估模型尺寸、推理延迟、准确率、F1 以及攻击召回率。压缩后模型尺寸从 0.78 MB 降至 0.0635 MB，推理延迟下降 74.5%，攻击召回率从 46.7% 提升至 93.85%，整体准确率约 96%、F1 约 89%；在不同压缩级别下对比 QAT 与非 QAT 的性能差异；

**⚠️ 局限性**

局限性包括仅在实验室环境验证，未在真实微控制器上完整测评；量化感知训练在某些压缩设置下可能导致训练不稳定；实验仅基于单一数据集，未覆盖更强对抗攻击；通信开销和客户端异构性处理仍待进一步研究。

---

## 192. Self-Evolving Multimedia Verification through Memory Consolidation of Contestation Experiences

**arXiv ID:** 2609.27175 | [PDF](https://arxiv.org/pdf/2609.27175v1)

**作者:** Truong Thanh Hung Nguyen `[一作]` (University of New Brunswick), Hung Cao `[通讯]` (University of New Brunswick)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出SEMV（Self‑Evolving Multimedia Verification）框架，构建一个多代理的可追溯、可修订、可记忆的媒体真实性验证流程，利用证据–论证–决策链进行推理与报告生成；

**💡 创新点**

创新点包括：① 将具有证明负载的论证作为系统状态；② 采用arena‑based quantitative bipolar argumentation (A‑QBAF) 进行量化推理与置信度估计；③ 引入因果可争议与局部重算机制，精准定位并局部重算受影响的验证阶段；④ 设计验证门控的自演化记忆机制，保证经验在合并前经过验证、冲突保留，抑制负迁移；

**🔧 技术方法**

技术手段主要包括多代理系统架构、A‑QBAF量化双极论证、因果检索与检索计划、外部工具集成（OCR、ASR、图像检索、法医分析等）、LLM推理与报告生成、验证门控的记忆合并与冲突处理；

**📊 数据集**

使用的数据集包括：COSMOS（图像‑字幕一致性验证），MV2026（多模态验证大赛的报告质量评测），CTR（由人工审计构造的争议修正轨迹），以及公开Web/OSINT检索索引；

**📈 对比分析**

与现有最强基准（Tran 等）在COSMOS 1700 条测试上比较，SEMV 在四个大型模型（Gemma4、Qwen3.6、Nemotron3、InternVL3.5）上平均准确率提升约2%（最高91.88%），在 MV2026 上报告质量得分 714.95，远超主流对手；在 CTR 50 例中，因果争议+局部修订将纠错率提升至 96.7%，误报率仅 0.2%，并节省 52.8% 计算量；

**⚠️ 局限性**

局限性包括：COSMOS 仅为二分类任务，难以评估更复杂的多维事实检验；MV2026 只包含少量验证与测试样本；CTR 人工修正依赖少数案例，无法覆盖真实工作流；检索缓存不具实时性，未覆盖跨语言与跨域检索差异；内存安全与攻击防护仍需进一步强化；

---

## 193. Phonemizing User-Generated Text: A Benchmark, Taxonomy, and Compositional Approach

**arXiv ID:** 2609.27205 | [PDF](https://arxiv.org/pdf/2609.27205v1)

**作者:** MinJu Jeon `[一作]` (NAVER Cloud), Hoyeon Lee `[通讯]` (NAVER Cloud)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了首个针对英语、越南语、韩语用户生成文本的G2P基准UGTPHON，并提出基于检索的组合模型来显式建模规范化与发音过程。

**💡 创新点**

引入面向推理的非规范化词分类体系、检索增强的规范化提示和分阶段解码，显著缩小了规范化与非规范化之间的PER差距。

**🔧 技术方法**

采用上下文化编码、Exact‑match检索以及在Transformer/Byte‑level架构中实现的分阶段生成；基底为ByT5和Qwen2.5‑0.5B。

**📊 数据集**

利用18,373句子组成的UGTPHON数据集，涵盖英语5,029句、越南语10,847句、韩语2,497句，并为每个非规范词标注规范形式与金标准音标。

**📈 对比分析**

在三种语言上对比规则、监督G2P、两阶段正则化+G2P以及前沿LLM，实验显示我们的检索增强模型将非规范PER从约70降至约20，且在所有语言上与规模更大的LLM性能相当。

**⚠️ 局限性**

局限于仅覆盖三种表音系统、需要持续更新检索数据存储、未评估主观语音质量，且对非拼音文字的适用性未知。

---

## 194. Exploiting Decompression Latency for Covert Channels in Inter-Line-Compressed LLCs

**arXiv ID:** 2609.27319 | [PDF](https://arxiv.org/pdf/2609.27319v1)

**作者:** David K. Oh `[一作]`, Hiroshi Sasaki `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

研究并实现了一种利用XOR cache压缩/解压延迟差异的隐蔽通道攻击，称为XOR-CC。

**💡 创新点**

创新点在于发现并利用压缩状态导致的远程恢复时延差异作为时序信号，并通过协作写入预定数据来控制压缩状态，从而在LLC上构造高带宽（≈2.9Mbps）且错误率低于1%的隐蔽通道。

**🔧 技术方法**

采用gem5+Ruby完整系统仿真平台，配合XOR Cache架构、可精确计时的高分辨率计时器、预先约定的地址对与数据值以及四阶段通信协议。

**📊 数据集**

使用随机生成的5000位长度负载（共10个不同种子共50,000位），在Linux 4.19.83 / Ubuntu 18.04环境下的gem5模拟器中进行实验，选择了满足XOR Cache配对规则的四个地址对。

**📈 对比分析**

与传统Prime+Probe对比，在相同误码率（<1%）下，XOR-CC带宽约为2.9Mbps，Prime+Probe仅220Kbps，提升约13.1倍；同时XOR-CC在各级缓存的访问量和LLC缺失率均显著低于Prime+Probe。

**⚠️ 局限性**

限制主要包括：需要双方进程在同一核心共享LLC且具备协作能力；攻击依赖特定配对策略和预先约定的数据值；对抗措施如域标识、延迟匹配或检测高频压缩/解压事件等可能降低攻击可行性。

---

## 195. CART: Closed-Loop Adaptive Red Teaming for Large Language Models

**arXiv ID:** 2609.27336 | [PDF](https://arxiv.org/pdf/2609.27336v1)

**作者:** Dongdong Zhang `[一作]` (Microsoft Research), Furu Wei `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Closed‑Loop Adaptive Red Teaming（CLART）框架，能够在测试过程中根据每轮结果自适应地选择、生成并改进攻击提示，记录完整的源头、变换及评估证据，并将模型、代理和评估者拆分为 Challenger、Target、Judge 三个角色，支持文本模型与有限工具使用代理的安全评估。

**💡 创新点**

核心创新在于：① 将红队从一次性检查变为闭环的持续搜索；② 使用记忆驱动的 Thompson‑sampling 选择策略，在覆盖与弱点追踪之间动态平衡；③ 通过种子检索与“seed‑adaptation/seed‑expansion”机制保持测试可追溯；④ 强制多样化生成以避免同义词泄漏；⑤ 角色分离与完整记录使结果可审计、可复现。

**🔧 技术方法**

技术实现包括：
- 记忆层记录每轮风险分数、失败次数与最新高风险示例。
- Thompson‑sampling 采样风险类别，配合多策略库（coverage、weakness‑pursuit、pressure、agentic、systemic）。
- 基于种子检索的 Prompt‑grounding 与自适应变换。
- 采用多维风险评分（severity、blast radius、reproducibility、confidence）。
- 对代理的安全评估通过 Mock‑tool sandbox 与 ReAct‑style 交互记录。
- Judge 采用 LLM 评估并输出可审核的 JSON 日志。

**📊 数据集**

数据集与基准：
- Frontier：76 条 ATLAS + OWASP + 自定义风险模式。
- JAH：JailbreakBench 100 条、HarmBench 400 条、AgentHarm 260 条，共 760 条。
- Agentic：Agentic ASB 2000 条、InjecAgent 1054 条、内部生成 200 条，共 3254 条。
- 所有集合统一映射至 12 种风险类别、9 种能力类别和 12 种攻击策略。

**📈 对比分析**

实验对比方式：在相同 Challenger（Claude）与 Judge（Gemini）下，分别对 7 种目标模型（Claude、GPT、Grok、Kimi、GLM、Gemini、DeepSeek）在 Frontier、JAH、Agentic 三组数据上执行 1000 或 3200 轮的闭环测试，并与同预算的静态种子重放 baseline 进行对比。结果显示：
- 所有目标模型的失败率均显著高于 baseline，平均风险分数提升 20–60%。
- 对于 Agentic 任务提升最大（从几乎 0% 到 50–70%）。
- 不同模型对 adaptive 的敏感度不同，表明模型本身差异影响发现效果。

**⚠️ 局限性**

限制与风险：
- Judge 的一致性虽然高，但对评分细节与证据解释仍存在差异；需要多 Judge 或人工复核。 
- 结果为“发现到达率”，并不等同于真实部署中的风险频率。 
- 随机种子与策略选择导致不同跑的波动，需更严格的复现实验。 
- 规模化受限于种子检索、词表大小与并行推理成本；更大词表和语料库需要索引化检索与层级选择。 
- 仅在安全沙箱内执行，未验证在真实环境下的攻击效果。

---

## 196. Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents

**arXiv ID:** 2609.27334 | [PDF](https://arxiv.org/pdf/2609.27334v1)

**作者:** Yefan Zhou `[一作]` (Salesforce Ai Research), Shafiq Joty `[通讯]` (Salesforce Ai Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种只在读取时对原始轨迹进行精炼的记忆系统（Just-in-Time Memory），并训练一个内存策划器在任务到来时提取任务适配的摘要。

**💡 创新点**

核心创新在于将记忆精炼从写时推迟到读时，使同一轨迹可为不同任务生成不同摘要；简化信用分配，使策划器能直接利用即时任务奖励训练；并实现了跨执行器迁移。

**🔧 技术方法**

技术方案包括预训练LLM作为执行器与判断器、使用GRPO强化学习训练策划器、BM25检索器、存储原始轨迹的记忆库以及任务与检索结果的联合输入。

**📊 数据集**

在三个Agentic基准上进行评估：ALFWorld、WebShop 和 τ²-bench。

**📈 对比分析**

与无记忆代理以及三种写时记忆基线（ReasoningBank、MemP、SkillOS）比较，Just-in-Time Memory 在 ALFWorld、WebShop、τ²-bench 上分别提升了 16.2、16.3、3.9 绝对成功率；未训练策划器已具竞争力，训练后进一步提升；并在不同执行器间实现近乎无损的迁移。

**⚠️ 局限性**

限制包括检索器仅为 BM25，可能在大规模多样化记忆库中受限；策划器需额外的 LLM 调用；payload 结构固定且未自适应；尚未探索每轮或步骤级别的动态精炼。

---

## 197. JEV-Star: Fast, Low-Cost StarCraft II Control with Language-Model Planning

**arXiv ID:** 2609.27331 | [PDF](https://arxiv.org/pdf/2609.27331v1)

**作者:** Weiyu Ma `[一作]`, Jian Zhao `[通讯]` (Beijing Zhongguancun Academy)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种名为JEV-Star的StarCraft II控制器，通过快速的JEV动作选择与持久的GPT‑6规划相结合，成功击败了内置的Lv7非作弊AI；

**💡 创新点**

创新点在于将即时低成本决策（JEV）与长期战略规划（GPT‑6）分离，并通过持久计划实现资源预留与军队姿态的持续约束，从而在保持子秒级响应的同时实现更具前瞻性的游戏决策；

**🔧 技术方法**

采用托管LLM JEV进行结构化动作选择，使用GPT‑6进行持续规划，配合SC2LE接口、异步控制、结构化观察与候选集，整合了LLM Play SC2与SMAC‑Hard框架进行全局与微观控制；

**📊 数据集**

使用了内置Zerg对手在AltitudeLE地图上的四种难度（Lv2、Lv5、Lv6、Lv7）进行全局实验，并在35张SMAC‑Hard地图上进行三集每图的微观战斗实验；

**📈 对比分析**

通过对比JEV‑only与JEV+GPT‑6两种配置，评估了全局胜率、敌方消灭率、JEV响应时间与模型成本。JEV+GPT‑6在四场全局非作弊比赛中全部取胜，其中两场击败Lv7；微观层面平均敌方消灭率从16.5%提升至37.7%，胜率从3/105提升至7/105；JEV平均响应时间约0.42 s，单局成本约USD 3.71（GPT‑6占比96%）。

**⚠️ 局限性**

局限性包括：实验为观测性对比，未进行严格的消融研究；候选描述与执行处理在两种配置间不同；样本规模有限，尤其全局胜率仅基于四场；仅覆盖单种种族与对局配置，未验证在更广泛场景下的鲁棒性；GPT‑6规划占据大部分成本，进一步降低规划频率或共享计划仍是挑战。

---

## 198. A Sample-Based Approach for Hierarchical Information-Theoretic Compression of Probabilistic Occupancy Grids

**arXiv ID:** 2609.27330 | [PDF](https://arxiv.org/pdf/2609.27330v1)

**作者:** Zhenyu Jin `[一作]` (University of Arizona), Daniel T. Larsson `[通讯]` (University of Arizona)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于蒙特卡洛树搜索的样本驱动方法，用来构造概率占用格的多分辨率信息压缩，并且可以在任何时刻终止得到有效压缩结果。

**💡 创新点**

创新点在于：①将信息瓶颈优化映射到树搜索框架，①使用样本估计节点价值而非完整枚举，从而实现任意停止的“anytime”算法；②通过UCB+ε-greedy双层探索将计算资源聚焦到高信息增益区域。

**🔧 技术方法**

使用技术包括信息瓶颈理论、互信息、Jensen–Shannon 散度、Monte Carlo Tree Search（MCTS）、Upper-Confidence-Bound（UCB）、ε-greedy、信息熵计算等。

**📊 数据集**

实验数据集包括：一个 128×128 的合成占用格；以及从真实 LiDAR 数据生成的 2048×2048 采样概率占用格（选取的三个 ROI R_A、R_B、R_C）。

**📈 对比分析**

与最优 Q-tree 搜索算法进行对比，比较指标包括总奖励、误差、信息平面曲线以及叶子节点数和保留信息。结果显示样本方法在 88% 信息保留的同时，将计算时间压缩到 Q-tree 的 11%（约 9 倍加速）。

**⚠️ 局限性**

limitations: 需要预设 α、ε、κ；压缩质量随 roll‑out 预算变化；在大规模网格下仍可能需要大量样本；在中途终止时可能导致信息损失；参数调优需针对不同环境手工实验。

---

## 199. Learn How to Act from Your Own Interactions: On-Policy Self-Distillation for GUI Agents

**arXiv ID:** 2609.27307 | [PDF](https://arxiv.org/pdf/2609.27307v1)

**作者:** Yan Zhang `[一作]` (Chinese Academy of Sciences), Yu Zhou `[通讯]` (Nankai University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 GUI‑SD‑v2，一种针对长时程 GUI 交互的两阶段 on‑policy 自蒸馏框架，强化特权跟随能力并对步骤级推理与记忆进行选择性蒸馏。

**💡 创新点**

创新点在于通过特权跟随优化提升模型对特权提示的利用，并通过信息化特权蒸馏为多步交互提供精准的推理与记忆监督。

**🔧 技术方法**

采用 on‑policy 自蒸馏、GRPO 风格的策略优化、针对 token 的 KL 蒸馏以及 Android 模拟器上的滚动采样技术。

**📊 数据集**

在 AndroidWorld 和 MobileWorld 两个移动 GUI 评测基准上进行实验，并使用 Kimi‑K3 生成的监督数据。

**📈 对比分析**

与 SFT、GRPO、Naive OPSD、GUI‑SD‑v1 等基线以及现有 state‑of‑the‑art 方法对比，PASS@1/3 成绩分别达到 67.2%/79.3%（AndroidWorld）和 25.6%/31.6%（MobileWorld），显著优于前者。

**⚠️ 局限性**

局限性包括仅在 Android 环境下评测，缺乏跨平台验证；同时对特权教师质量高度依赖，需进一步探索更广泛的平台与模型规模。

---

## 200. The Power of Recruiting the Smaller Side: Two Additional Traders Suffice in Two-Sided Markets

**arXiv ID:** 2609.27304 | [PDF](https://arxiv.org/pdf/2609.27304v1)

**作者:** Yang Cai `[一作]` (Google Research), Mingfei Zhao `[通讯]` (Google Research)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了双边双向拍卖中Bulow-Klemperer风格的竞争复杂性，证明了在买方数量大于等于卖方数量且买方估值在一阶上随机主导卖方成本的情况下，招募两个额外的卖方可以使得卖方交易减少机制实现至少与原市场的最优交易收益相同的预期交易收益。

**💡 创新点**

创新点在于确定了在双边市场中，稀缺方（卖方或买方）只需招募两个额外的参与者即可实现最优的交易收益，解决了之前未解的相关问题。

**🔧 技术方法**

使用了卖方交易减少机制（Seller Trade Reduction），该机制是一个独立于先验的机制，能够在不依赖于买方和卖方的分布的情况下实现预期的交易收益。

**📊 数据集**

使用了从分布F_B和F_S中独立抽取的买方和卖方的样本数据，具体的分布形式未在摘要中详细说明。

**📈 对比分析**

与之前的研究相比，证明了在买方数量大于等于卖方数量的情况下，招募两个额外的卖方可以实现最优的交易收益，而之前的研究认为需要招募更多的参与者才能达到相同的效果。性能上，招募两个额外的卖方的机制在理论上是最优的，无法通过招募一个额外的卖方来实现同样的效果。

**⚠️ 局限性**

限制在于在双边市场中，招募一个额外的卖方在某些情况下无法实现最优的交易收益，尤其是在买方和卖方的分布不对称时，单一的额外招募不足以满足机制的激励相容性和预算平衡性要求。

---

## 201. KITE: KV-Invariant Transformer Expansion for Efficient Agentic LLM Scaling

**arXiv ID:** 2609.27294 | [PDF](https://arxiv.org/pdf/2609.27294v1)

**作者:** Zhiheng Hu `[一作]` (StepFun), Daxin Jiang `[通讯]` (StepFun)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种KV不变的Transformer扩展方法（KITE），通过在训练过程中扩展不产生KV的网络部分，保持推理阶段的prefill成本不变，同时提升模型容量与性能。

**💡 创新点**

创新点在于将模型扩展与KV不变性相结合，利用两塔结构（Prefiller和Decoder）实现KV复用，既节省训练计算，又避免推理成本上升；同时在扩展后继续联合训练，实现更高效的规模化。

**🔧 技术方法**

采用了SST（Step Scale Transformer）作为KITE的实例，使用两塔Transformer结构、KV复用、Entry Bridge、MoE（稀疏专家）等技术，并在训练时使用Muon's优化器与Adam等。

**📊 数据集**

使用了大规模通用文本数据集（与原始SST基线相同的语料），并在多项下游任务（OpenBookQA、MMLU、GSM8K、MATH、HumanEval、MBPP、BBH）以及ARXIV NLL指标上进行评估。

**📈 对比分析**

通过与47B、63B经典Transformer基线在相同累计训练FLOPs下对比，SST在训练EMA‑200损失上更低，且在所有下游任务的准确率/性能指标均高于基线；在prefill占比高的场景下，其推理成本代理比47B低6.7%、比63B低31.6%。

**⚠️ 局限性**

限制在于仅验证了一种具体的KV分配与两塔连接方式，未系统探索KITE的完整设计空间；实验规模和任务仅覆盖了一部分场景，缺乏对真实部署速度或多模型混合情况的实测。

---

## 202. NGN: Learning Neural Network Size as a Differentiable Count

**arXiv ID:** 2609.27291 | [PDF](https://arxiv.org/pdf/2609.27291v1)

**作者:** Lixing Li `[一作]` `[通讯]`, Lixing Li

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Neurogenesis Network（NGN），通过学习一个可微分的边界变量来决定神经网络在训练期间应该使用多少个按顺序排列的结构单元，并在训练完成后按该边界裁剪模型。

**💡 创新点**

创新点在于将模型容量直接表示为一个可微分的计数，边界变量在训练过程中与权重共同优化，最终实现自动化的网络规模选择；此外，该方法不依赖离线剪枝或预先设定的容量上限，而是通过共享边界和指数衰减的权重系数动态扩展和收缩模型。

**🔧 技术方法**

技术包括：可微分的门控函数（Sigmoid/Softplus）逼近阶跃函数；使用容量价格正则项和周期性正则项使边界收敛到整数；门函数逐步锐化（β 线性增长）以实现软到硬的转换；对不同模型家族（MLP、卷积、图网络、Transformer、状态空间模型、LoRA、适配器）统一使用相同的边界机制；实验中对比固定规模模型、基于后训练剪枝的 Wanda、稀疏学习方法（Hard‑Concrete L0、SoRA）等。

**📊 数据集**

实验涵盖多种任务与数据集，主要包括：合成回归任务、计算机视觉任务（如 ImageNet/COCO 风格的图片分类/检测）、图学习任务（如 Cora/CiteSeer 关系预测）、语言建模与文本生成任务（如 WikiSQL/GLUE），以及参数高效微调（PEFT）任务。每种任务都使用了相应的公开数据集与标准评价指标。

**📈 对比分析**

比较方法：与同等规模的固定结构模型、以及通过后训练剪枝或稀疏化得到的模型进行对比。实验显示：学习得到的前缀在部署后几乎不降低性能，且在多数任务上与手工设计的同等规模模型相当或略优；相比于固定模型，NGN 需要更多训练步数才能达到相同性能；与剪枝方法相比，NGN 在不需要额外的后处理步骤的情况下实现了容量自适应。

**⚠️ 局限性**

局限性包括：需要先定义一个按容量递增的有序候选序列，边界无法跳过低效的早期单元；指数衰减的权重系数限制了可扩展的最大容量，导致高指数索引的单元很难被利用；训练过程更耗时，需要额外的门控锐化与正则化调度；对容量价格 λ 和周期正则 α 的设置较为敏感，可能需要针对不同任务手动调参；目前缺乏对学习到的容量分配进行可解释性分析的方法。

---

## 203. Sparse-Observation Atmospheric Thermal Forecasting with Physics-Informed Neural Networks for Climate-Aware Digital Twins

**arXiv ID:** 2609.27290 | [PDF](https://arxiv.org/pdf/2609.27290v1)

**作者:** Tannaz Goodarzvand Chegini `[一作]` (Montana State University), Faraz Dadgostari `[通讯]` (Montana State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估了一个受物理约束的神经网络（PINN）用于在温度观测稀疏的情况下进行短时（1-3小时）大气潜温预测，并在Oklahoma、Alabama和Montana三地区进行实验。

**💡 创新点**

提出将等压坐标热力学输送方程与经验散热闭合结合进PINN，验证物理约束在稀疏观测与不同预测时距下的提升，并发现固定等压层在高地形下的适用局限。

**🔧 技术方法**

使用Physics‑Informed Neural Network（PINN）架构，基于潜温的压强坐标热力学输送方程、经验散热闭合；训练采用分阶段历史源学习与未来物理约束；对比基线包括持久性、趋势、坐标仅网络、强迫网络。

**📊 数据集**

使用ERA5逐时重分析的温度、风、垂直压力速度和湿度字段，在三种地区的热浪期间进行实验。

**📈 对比分析**

与四个基线（持久性、趋势、坐标网络、强迫网络）以及最佳基线包络进行RMSE对比；在Oklahoma密集观测下PINN平均提升约17-18%，在稀疏观测下+3h提升仍保持≈15%，在Alabama可跨地区迁移且提升≈20%，但在Montana高地形下表现不佳，RMSE甚至较高。

**⚠️ 局限性**

仅评估了ERA5重分析的回归性实验，未使用真实观测；在高地形地区固定等压层失效；依赖未来的风/湿度强迫；经验散热闭合可能不具普适性。

---

## 204. EnSIMem: Entity-Structured Indexing for Long-Term Agent Memory

**arXiv ID:** 2609.27279 | [PDF](https://arxiv.org/pdf/2609.27279v1)

**作者:** Xuanyu Meng `[一作]` (University of Illinois Urbana-Champaign), Jiawei Han `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于实体-属性结构索引的长期对话记忆系统 EnSIMem。

**💡 创新点**

创新点是将对话中的实体与属性抽取成结构化记录，结合主题连贯的情节分段和需求驱动的检索，生成只保留必要原始证据的短上下文缓冲区。

**🔧 技术方法**

使用 LLM 进行分段、实体属性抽取、查询规划，结构化匹配加稠密检索，以及基于检索需求的自适应检索预算。

**📊 数据集**

在 LoCoMo 与 LongMemEval 两个长期对话记忆基准上进行评估。

**📈 对比分析**

相较于 MEMORA、RAG 等基线，EnSIMem 在单跳、时间推理、组合与聚合题型上取得 90%+ 的准确率，整体准确率达到 90%+，并在效率上保持约 14 秒的平均延迟。

**⚠️ 局限性**

局限在仅处理情节记忆，缺乏程序性和语义记忆的整合，且依赖 LLM 的抽取与规划，可能导致抽取误差和跨模型不确定性。

---

## 205. Graph Learning with Spectral Connectivity Priors for Scarce Data

**arXiv ID:** 2609.27278 | [PDF](https://arxiv.org/pdf/2609.27278v1)

**作者:** Mingxiao Liu `[一作]` (Tsinghua University), Feifei Gao `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 SCoGL 框架，在稀缺数据下通过在组合式图拉斯松约束的 GLASSO 目标中加入拉普拉斯谱连通性先验，来学习稀疏图。

**💡 创新点**

创新点在于：①将多种拉普拉斯谱连通性正则（Fiedler 值、总有效电阻等）统一纳入凸优化框架；②给出这些正则的梯度表达式，采用全矩阵投影梯度下降（PGD）+ Armijo 回溯实现全局连通性正则的联合优化；③相较于先前基于逐条边的贪婪策略，提供全局最优解。

**🔧 技术方法**

使用了图形玻尔兹曼（GLASSO）与组合拉普拉斯约束、谱图理论（拉普拉斯特征值）、凸优化、投影梯度下降、Armijo 回溯法以及梯度推导等技术。

**📊 数据集**

在合成的加权 Erdős–Rényi 图上评估（N=100，边缘概率 p=0.15，权重均匀分布在 0.5–1.5 之间），使用 10 个验证图和 10 个测试图。

**📈 对比分析**

与 GLASSO、CLIME、NGL、CGL-BCD、GLENE、GL‑SigRep、Kalofolias 以及不带连通性正则的 SCoGL‑0 进行对比；评价指标为相对拉普拉斯误差 RE 与图信号去噪 NMSE；SCoGL 变体在 RE 约 0.37（比最差 0.47 低 10–15%）和 NMSE 约 0.347（比 SCoGL‑0 降低 30–45%）等方面均优于基线。

**⚠️ 局限性**

局限性包括：仅在合成数据上验证，未测试真实数据；需要手动调参（μ、γ 等）；每次迭代复杂度 O(N³)，在大规模图上可扩展性受限；以及对拉普拉斯约束的依赖可能限制了对非组合式图的适用性。

---

## 206. TimeEvo: Failure-Driven Self-Evolution of a Time Series Agent

**arXiv ID:** 2609.27277 | [PDF](https://arxiv.org/pdf/2609.27277v1)

**作者:** Jie Yang `[一作]` (University of Illinois at Chicago), Philip S. Yu `[通讯]` (University of Illinois at Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个自进化的时间序列分析代理，利用诊断失败自动生成仅基于证据的工具，并通过双层验证门控安全引入；

**💡 创新点**

创新点在于将失败驱动的工具生成与两级验证（单工具预筛选+整体配对门）相结合，避免了工具与任务不匹配与“静默破坏”的问题；

**🔧 技术方法**

采用了冻结的大型语言模型、失败驱动的分析与规划、测量合同、证据仅工具合成、训练集决策树校准、单工具预筛选及整体配对门控等技术；

**📊 数据集**

使用了十个公开时间序列 QA 基准，来源于六个数据集（TemporalBench T1–T4、TimeSeriesExam、Merrill、TimeMQA（Anomaly、Classification）、MMTS Match、TSAQA Data-Transformation）；

**📈 对比分析**

与五个基线（Chronos 预装工具、21个TimeART手工工具、Self-Refine、5次投票、ICL）在三种后端模型（GPT-5.6-luna、GPT-5.6-terra、GPT-5.4-mini）上进行配对对比，结果显示在每个任务和每个模型上都实现正向提升，平均增益在+1.5到+14.7分之间，且跨模型迁移后仍保持显著提升；

**⚠️ 局限性**

局限性包括：仅针对冻结 LLM，依赖训练集进行校准和门控；工具生成对复杂推理能力有限，可能无法覆盖所有任务；门控可能误判有用但噪声较大的工具；评测范围仅限于 QA 任务，缺乏对更广泛时间序列应用的验证。

---

## 207. Alignment Inertia: Auditing the Durability of Training Data Influence Through Policy Override Resistance

**arXiv ID:** 2609.27333 | [PDF](https://arxiv.org/pdf/2609.27333v1)

**作者:** Renata Barreto `[一作]` (eBay), Mohammad Tahaei `[通讯]` (eBay)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Override Success Rate（OSR）和Alignment Inertia两种指标，用于衡量平台运营者在对LLM进行系统提示或LoRA微调时能否有效覆盖先前训练的行为偏好。

**💡 创新点**

创新点在于将行为可控性量化为可测量的度量，并将行为审核与贡献性归因结合，利用TRAK等梯度归因工具判定哪些训练样本导致了抵抗。

**🔧 技术方法**

主要技术包括基于无指令基线的经验性政策估计、零射系统提示与LoRA微调、以及TRAK梯度投影归因分析。

**📊 数据集**

实验使用公开医学误导数据集（PUBHEALTH/ImperialCollegeLondon/health_fact）和仇恨言论数据集（Davidson hate-speech/offensive-language），每个条件评估500个样本。

**📈 对比分析**

对比结果显示，在两种域与两种政策方向下，系统提示与LoRA微调均未显著降低inertia；且在部分条件下微调反而降低OSR，表明微调并不总能克服先前训练的行为。

**⚠️ 局限性**

局限性包括模型规模受限、训练样本池不完整导致实验偏倚、未对全参数微调或少样本提示进行评估，以及归因仅针对LoRA层而非完整模型。

---

## 208. Can Vision-Language Models Analyze Human-Centered Video? Mapping Model Capabilities and Human-AI Collaborative Workflows

**arXiv ID:** 2609.27327 | [PDF](https://arxiv.org/pdf/2609.27327v1)

**作者:** Xiyuan Shen `[一作]` (University of Washington), Jacob O. Wobbrock `[通讯]` (University of Washington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过系统审计CHI 2026论文，提出五维视频注释分类法并基于该法构建15任务基准，比较了通用视觉语言模型(VLM)单独、人工单独及人工验证VLM输出三种工作流，探讨其在HCI视频分析中的适用性与成本效益。

**💡 创新点**

创新点在于将HCI视频注释实践与VLM能力进行对接，形成以方法学维度为核心的评估框架，提出可操作化的指导问题，并首次量化验证VLM与人类协作在准确率、时间和经济成本上的三重收益。

**🔧 技术方法**

主要技术包括大规模通用视觉语言模型（如Gemini）、人工标注工具、人工验证界面以及Human‑Normalized Score（HNS）等评价指标。

**📊 数据集**

采用公开的15个代表性视频任务，涵盖外景、第一人称、屏幕录制等多种视角，任务来源包括AMI、Ego4D、GUI‑World、Charades、MammalNet、SAV、WLASL等开放数据集。

**📈 对比分析**

通过比较VLM单独、人工单独及人工验证三种工作流，发现VLM单独的平均HNS为97.0，几乎等同于人工单独；人工验证平均HNS最高为121.5，且人力时间减少约48.9%，成本相较人工单独降低31.3%–44.5%。

**⚠️ 局限性**

局限性包括仅评估单一VLM和提示设置、基准样本有限（15任务），未涵盖所有HCI注释场景；分类法与工作流的适用性需在不同领域、不同数据集和不同模型版本上进一步验证；部分任务的参考标签可靠性不足，可能影响性能评估。

---

## 209. Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms

**arXiv ID:** 2609.27321 | [PDF](https://arxiv.org/pdf/2609.27321v1)

**作者:** Xinjie Shen `[一作]` (Georgia Institute of Technology), Dayiheng Liu `[通讯]` (Alibaba Token Foundry, Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出机制优先构建(agentic environment)方法，将已求解的数学模型用于生成可执行环境及可验证的奖励信号；利用语言模型生成器在真实语料中构建情景并对生成的实例进行自动录用；使用GRPO对Qwen3.6‑35B-A3B进行训练。

**💡 创新点**

创新点在于：①先解决数学模型再生成环境，统一奖励与动态；②利用语料生成器实现海量多样化环境，极低扩展成本；③将环境与奖励从同一已验证模型中派生，保证可追溯性与一致性；④在此基底上训练出具备长时序决策与信息收集能力的模型。

**🔧 技术方法**

技术包括：数学模型求解（线性/动态/整数规划等）、将求解结果映射为可执行状态机、语料驱动的环境生成、信息不对称接口设计、自动录用验证、GRPO强化学习。

**📊 数据集**

数据集：28个主题领域的真实文档语料，用于种子情景；生成的3,300个已录用环境；外部评测基准BFCL V4、TravelBench、E‑Commerce Bench。

**📈 对比分析**

对比方法：基线Qwen3.6‑35B‑A3B、已训练模型、Qwen3.7‑Max。结果显示：在5种优化家族上平均从0.204提升到0.815；对未见家族、远OOV家族以及长时序外部基准均有显著提升；在365天电商跑，训练模型在所有试验中无破产并超过Qwen3.7‑Max。

**⚠️ 局限性**

限制包括：仍依赖可求解的数学模型家族，难以覆盖非规划类任务；生成环境的多样性受语料与模型族的限制；当前实验聚焦于优化和物流类问题，未验证在更广泛的真实世界场景；模型规模与算力需求仍较高。

---

## 210. Discrete Diffusion Models via Evolving Variational Autoregressive Networks

**arXiv ID:** 2609.27306 | [PDF](https://arxiv.org/pdf/2609.27306v1)

**作者:** Kewen Pan `[一作]` (University of Electronic Science and Technology of China), Ying Tang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于变分自回归网络（VAN）的离散扩散模型，用于在二维和三维Ising格点上以归一化概率分布形式表示目标分布，并通过连续时间马尔可夫链实现噪声化与去噪；

**💡 创新点**

核心创新是将传统依赖张量网络的离散扩散方法替换为VAN，实现对高维格点系统的可归一化建模；通过显式马尔可夫跳跃算子控制前向噪声化与反向去噪，兼顾自由能、能量和磁化等热力学量的直接估计；

**🔧 技术方法**

使用VAN进行分布拟合，Euler离散化的连续时间马尔可夫链做前向噪声化，反向去噪采用Euler或tau-leaping近似；将该模型嵌入Metropolis-Hastings MCMC，形成连通与不连通更新两种采样策略；

**📊 数据集**

在二维8×30、9×30、16×16以及三维4×4×4的Ising模型上，覆盖从有序到无序、临界温度范围内的多组温度；

**📈 对比分析**

与张量网络（MPS）和Wolff聚类Monte Carlo比较；VAN扩散模型在磁化、能量、自由能等指标上均优于MPS，并在三维系统中实现了可行的高精度估计；与传统单自旋翻转MC相比，连通更新在低温下保持更高接受率、样本多样性和有效样本量；

**⚠️ 局限性**

局限包括：训练VAN时可能无法捕捉低概率但重要的配置；每一步生成候选样本的计算成本高于传统MC；缺乏对整个MCMC过程的严格理论收敛性证明。

---

## 211. SoK: You Find What You Seek: Rethinking Oracles, Guidance, and Input Generation in Hardware Fuzzing

**arXiv ID:** 2609.27300 | [PDF](https://arxiv.org/pdf/2609.27300v1)

**作者:** G Abarajithan `[一作]` (UC San Diego), Ryan Kastner `[通讯]` (UC San Diego)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并提出一个统一框架，用于把硬件模糊测试视为受目标、预算、oracle、指导和输入生成限制的有界搜索。

**💡 创新点**

创新点在于将硬件模糊测试拆解为三大核心组件（oracle、指导、输入生成），并通过52个不同抽象级别（IP、CPU、NoC、SoC）的模糊器进行系统性对比，提出了可复现、可比较的评估准则。

**🔧 技术方法**

采用的技术包括覆盖度与数值得分反馈、结构化/随机/符号求解生成、语言模型程序生成、增量反馈循环、以及针对安全属性的自定义损失函数。

**📊 数据集**

数据集涵盖多种公开设计与IP：OpenTitan、Rocket/BOOM、OpenPiton、Ariane/CVA6、各种SoC、CPU核心以及小型IP模块，所有模糊器在这些目标上执行。

**📈 对比分析**

比较方法基于相同验证目标、抽象层、oracle、预算的条件下进行；通过覆盖率闭合、bug检测率、时间/资源使用等指标与传统受限随机验证（CRV）对比，发现某些模糊器在特定安全目标下能在更小预算内发现更多缺陷，但整体性能仍取决于 oracle 与指导的一致性。

**⚠️ 局限性**

局限性包括缺乏统一可复现的基准与公开工具链、不同抽象层的 oracle/指导难以标准化、模糊器对硬件访问与工具依赖度高，导致在工业环境中推广时需额外人工配置与集成。

---

## 212. Teach-to-Crash: A Closed-Loop Student-Teacher LLM Framework for Collision-Inducing Test Scenario Generation

**arXiv ID:** 2609.27296 | [PDF](https://arxiv.org/pdf/2609.27296v1)

**作者:** Zaid Ghazal `[一作]` (University of Michigan-Dearborn), Bruce Maxim `[通讯]` (University of Michigan-Dearborn)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种闭环双 LLM 框架 Teach-to-Crash，自动生成可执行的、具备碰撞诱发能力的自动驾驶测试场景。

**💡 创新点**

创新点包括：① 双 LLM 结构（低推理学生与高推理教师）实现增量式策略调整；② 基于 ego‑centric 位置网格的可执行场景编码；③ 使用滚动碰撞率与 TTC 监测停滞并触发教师干预；④ 场景验证器保证输出合法且可执行。

**🔧 技术方法**

使用技术包括：GPT‑5‑nano 作为学生 LLM，GPT‑5‑mini 作为教师 LLM；CARLA 0.9.12 仿真环境；基于 JSON 的位置‑速度序列编码；滚动窗口统计与停滞判定；多目标评估（碰撞率、TTC、Crit、diversity、avoidable 率）。

**📊 数据集**

数据集/环境：CARLA Town06 高速路段，固定 60 秒仿真时间，2 名 NPC，使用两种 ego 控制器（Traffic Manager 与 TransFuser）。

**📈 对比分析**

与基线 PAFOT（遗传算法）和 ChatScene（单 LLM 生成）比较。Teach-to-Crash 在 10 次实验中实现：最高碰撞命中率（≈90.8%），最短平均 TTC（≈18.3 s），最高多样性（Jaccard ≈0.547），并在避免性可用性评估中领先（≈39‑80%）。与 PAFOT 的 CDR 较高但方差大，与 ChatScene 的碰撞率和多样性均显著优于其。

**⚠️ 局限性**

局限性：仅评估单一地图与两名 NPC 的高速场景；未检验更复杂交叉路口或多车密度情况；仅使用两种控制器；缺乏壁钟时间、LLM 推理成本及 token 消耗评估；未与多样性专门算法或同一模型闭环方案对比；避免性评估为启发式近似，未提供真正可达性或调试价值证明。

---

## 213. BranchDrive: A Branch-Structured Dataset for Action-Conditioned Driving Prediction

**arXiv ID:** 2609.27275 | [PDF](https://arxiv.org/pdf/2609.27275v1)

**作者:** Feeza Khan Khanzada `[一作]` (University of Michigan-Dearborn), Jaerock Kwon `[通讯]` (University of Michigan-Dearborn)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 BranchDrive 数据集与基准，构造了包含单一历史、名义专家未来以及 12 个干预未来的分支结构样本。

**💡 创新点**

创新点在于：①通过两遍收集与严格审核实现同一决策上下文下的多行动结果；②使用动作编码和无泄漏验证确保样本可直接用于动作条件预测；③将连续指标与轨迹预测结合，提供固定银行离线评估。

**🔧 技术方法**

技术方法包括：CARLA 同步仿真、两次回放收集、完整动作执行与专家恢复、动作编码与结构化历史融合、模型训练（M0–M6）、宏观指标（NMAE、ADE）评估、离线固定银行排名与保守阈值校准。

**📊 数据集**

使用的基础数据集是基于 CARLA Town12/Town13 的模拟数据，共 606 个分支组（7,878 条轨迹）和 12 种干预策略。

**📈 对比分析**

在训练-验证-测试划分下比较 M0–M6 模型，结构化历史+动作模型（M3）取得宏观 NMAE 0.5036、ADE 2.204 m 最佳，M5 在数值指标略优但无显著差异；离线固定银行排名 O‑M3 在策略价值+0.034、归一化遗憾-0.118 上均显著提升。

**⚠️ 局限性**

局限性包括：仅覆盖 2.5 s 的短期预测且不含闭环评估；保守阈值过宽导致无法选择干预；未提供精确因果效应、二进制安全判定或真实道路验证；数据来源为仿真，转移到现实场景存在挑战。

---

## 214. High Dynamic Range Video Reconstruction from Single-Exposure Raw Sequences

**arXiv ID:** 2609.27274 | [PDF](https://arxiv.org/pdf/2609.27274v1)

**作者:** Tao Zhang `[一作]` (Hangzhou Dianzi University), Chenggang Yan `[通讯]` (Hangzhou Dianzi University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种端到端的Raw视频HDR重建框架RawHDRV，利用单曝光原始Bayer数据的通道特性和帧间曝光互补，进行通道分解对齐、曝光感知融合以及遮罩引导的补偿恢复。

**💡 创新点**

创新点包括三方面：①通道分解并行对齐与曝光加权融合，充分利用R、G、B通道的噪声与饱和差异；②曝光互补遮罩引导的恢复模块，动态选择相邻帧可靠信息以补偿过曝区域；③基于遮罩的颜色损失，在训练时专门提升过曝区域的亮度与颜色一致性。

**🔧 技术方法**

使用了Bayer通道分解、光流Warp、跨通道注意力、Transformer块、加权融合、曝光互补遮罩、log‑L2 + mask‑guided color loss，以及多尺度Encoder‑Decoder结构。

**📊 数据集**

构建了首个大规模单曝光Raw‑HDR视频数据集RawHDRV，包含400条移动端拍摄序列（共2455帧），每帧有10位Raw和对应HDR标签。

**📈 对比分析**

在RawHDRV上与多种单图像与视频HDR方法对比，使用PSNR‑L、PSNR‑μ、SSIM‑L、MS‑SSIM、HDR‑VDP‑2、HDR‑VQM等指标评测，RawHDRV在所有指标上均位列榜首（PSNR‑L 44.14 dB、PSNR‑μ 37.77 dB、HDR‑VDP‑2 72.45、HDR‑VQM 0.0066）。

**⚠️ 局限性**

当前模型参数量大、运算量高，尚未实现实时移动端部署；仅适用于单曝光RAW视频，对极端过曝仍受限于信息缺失；并且需要相邻帧的配准，无法处理大幅位移的场景。

---

## 215. What fidelity metrics miss: a structural check on synthetic educational data

**arXiv ID:** 2609.27265 | [PDF](https://arxiv.org/pdf/2609.27265v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 216. CAVEAT: Towards Robust Computer-Use Agents in Incentive-Misaligned Environments

**arXiv ID:** 2609.27273 | [PDF](https://arxiv.org/pdf/2609.27273v1)

**作者:** Yuxuan Li `[一作]` (Carnegie Mellon University), Zezhou Huang `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出激励失配环境下的计算机使用代理评估框架CAVEAT，并设计干预方法提升代理的用户目标保持能力

**💡 创新点**

定义激励失配概念并构建包含9个真实市场环境与8类真实激励机制的基准；通过三类失败诊断（目标漂移、搜索过早结束、未解决证据提前决策）提出针对性干预

**🔧 技术方法**

使用大型语言模型与推理-工具框架，加入结构化任务规范、验证工具以及后期训练（CAVEAT-Harness、CAVEAT-27B）

**📊 数据集**

CAVEAT基准：52个标准任务、5个Hard任务、4个Stay任务，涵盖短期住宿、零售、食品配送、转售、自由职业等多种购物场景

**📈 对比分析**

在匹配对照与激励失配对照下比较最佳购买率，发现从78.6%降至17.3%；CAVEAT-Harness将GPT-5.6-Sol从0%提升至80%，CAVEAT-27B将小模型提升至22.9%，整体性能显著提升

**⚠️ 局限性**

仅在在线市场场景实验，目标设定单一且固定，激励机制不随时间演变；鲁棒性提升伴随交互成本增加，跨环境泛化仍有限

---

## 217. Banana Kick: Response-Informed Skill Evolution for Humanoid Soccer

**arXiv ID:** 2609.27269 | [PDF](https://arxiv.org/pdf/2609.27269v1)

**作者:** Hao E. Zhang `[一作]` (Carnegie Mellon University), Ding Zhao `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用先验运动捕捉的普通踢球策略，通过自适应奖励与物理模型，演化出具有高旋转、曲线路径的香蕉踢球技能；

**💡 创新点**

提出了“响应感知技能演化”(RISE)闭环目标连续化方法，能够在一阶学习饥饿（目标对当前策略响应几乎无梯度）时，通过响应敏感度评估并验证奖励更新，确保踢球可靠性；

**🔧 技术方法**

结合运动捕捉重映射、模仿学习、PPO强化学习、已校准的球接触与Magnus气动力学模型、响应敏感度分析及闭环接受机制；

**📊 数据集**

使用人体踢球的运动捕捉数据作为先验；在仿真中采样4096个匹配上下文（每个随机种子），并在真实机器人上进行30次物理测试；

**📈 对比分析**

与学习进度课程、直接PPO以及两种消融版本对比。RISE在评估得分上提升19.8%（1093.7 vs 912.8），联合目标覆盖率从15.2%提升至50.9%，并在旋转、贴合误差等指标上明显优于基线；

**⚠️ 局限性**

受限于高质量的物理模型校准、仿真与真实世界差异以及仅针对单一踢球任务的验证，难以直接推广到其他复杂接触行为或不同机器人平台。

---

## 218. xTier: Intelligent Tiering for CXL-Enabled Memory

**arXiv ID:** 2609.27266 | [PDF](https://arxiv.org/pdf/2609.27266v1)

**作者:** Sriranga Ramaswamy `[一作]` (University of Colorado Boulder), Yueqi Chen `[通讯]` (University of Colorado Boulder)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在CXL内存扩展环境下，提出了一种基于eBPF的内核级学习驱动页级内存分层系统，实现了对DRAM与CXL层的自适应页面迁移。

**💡 创新点**

创新点包括：①在内核中执行INT8量化的多层感知机推理，避免浮点运算与验证器限制；②动态采样与冷却控制，实现从高采样到低采样的自适应决策周期；③多层入选与冷却阈值，限制迁移量并保证迁移收益；④通过eBPF链式程序将采样、特征构造、模型推理与迁移命令解耦，保持极低的决策延迟。

**🔧 技术方法**

核心技术：eBPF、PEBS精确事件采样、INT8量化MLP推理、动态采样调度、冷却与入选阈值控制、内核迁移执行器、BPF映射与离线训练流水线。

**📊 数据集**

使用六个内存密集型工作负载：XGBoost、LightGBM、Triangle Counting、Betweenness Centrality、BFS、PageRank，在CloudLab c220g2 的NUMA模拟CXL环境中收集数据并训练模型。

**📈 对比分析**

与AutoNUMA、TPP、Memtis、FreqTier进行比较。实验结果显示，在DRAM:CXL 比例从1:5到1:25的多配置下，系统在14/18配置中为最快，平均与最佳基线相差不超过3.9%，同时在几何平均上迁移量比基线低13倍，紧凑DRAM预算时优势更显著。

**⚠️ 局限性**

局限性包括：仅支持单一工作负载的专属模型；需要离线训练并假设工作负载重复执行；评估基于NUMA模拟的CXL，缺乏真实CXL硬件的验证；依赖Intel PEBS与访问位，移植到其他架构需额外工作；系统需自定义内核模块，部署门槛较高。

---

## 219. GaussPDE: Graph-Based Partial Differential Equation-Driven Rendering for 3D Gaussian Splatting

**arXiv ID:** 2609.27264 | [PDF](https://arxiv.org/pdf/2609.27264v1)

**作者:** Haoyuan Yue `[一作]`, Ziyin Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出GaussPDE框架，在已重建的3D高斯场景中通过构建相机感知的稀疏图谱，实现物理结构化的偏微分方程动力学可视化，避免网格化或重训练。

**💡 创新点**

创新点在于引入相机感知正则化减少摄像机近处漂浮点并构造边界感知的高斯图拉普拉斯算子，使得偏微分方程在高斯原语上稳定传播。

**🔧 技术方法**

使用技术包括3D高斯投影、相机感知正则化、共变距离加权的稀疏图谱、图信号处理中的拉普拉斯算子、隐式欧拉求解以及将偏微分状态映射至球谐系数进行渲染。

**📊 数据集**

使用数据集包括DTU、LLFF以及NeRF-Synthetic三组真实与合成场景。

**📈 对比分析**

与KNN、RBF、Gaussian LBO等传统图算法进行对比，GaussPDE在控制扩散、跨面泄漏率低于0.01%以及与网格参考误差最小化方面表现优异。

**⚠️ 局限性**

局限在于仍需依赖高质量重建的高斯场景，且对多尺度、非稳态方程的支持有限，缺乏对动态材质或光照变化的实验验证。

---

## 220. Specifying and Maintaining Agentic Workflows: An Empirical Study of GitHub Agentic Workflows

**arXiv ID:** 2609.27263 | [PDF](https://arxiv.org/pdf/2609.27263v1)

**作者:** Jasem Khelifi `[一作]` (University of Quebec), Mohamed Aymen Saied `[通讯]` (Universite Laval)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对 276 个 GitHub 仓库中的 1,248 个 gh-aw Markdown 工作流文件及其 20,841 次文件级变更进行经验性分析，研究其结构、演化和自然语言指令内容，并构建 10 类、42 子类的指令分类体系。

**💡 创新点**

创新点：①首次系统性提出并验证了 gh-aw 工作流指令的完整分类体系；②揭示了工作流文件在持续维护中的编辑模式（配置与指令分离、月度活跃度与增量衰减）；③证明大型语言模型可在 1-shot 条件下对指令进行多标签分类，F1 达 0.818、κ 0.715。

**🔧 技术方法**

使用技术包括：GitHub API 搜索与抓取、文本预处理（去除 frontmatter、代码块）、主题分析构建分类体系、LLM 归一化提示（Gemini、GLM‑5.2、GPT‑OSS‑120B）进行零/多-shot 多标签分类；对比手工标注并计算 Cohen κ、精确率、召回率、F1。

**📊 数据集**

数据集：1,248 个 gh-aw Markdown 源文件，涵盖 276 个仓库，7,446 次提交（20,841 次文件变更），共 1,248 个文件的最新快照以及 288 个手工标注样本；公开复现包可下载。

**📈 对比分析**

比较方法：将 LLM 的多标签输出与人工最终标签做逐对比较，统计二进制决策的协同一致度与 Cohen κ；结果显示 Gemini 在 1-shot 条件下与人类标注最接近，F1 最高（0.818），GLM 与 GPT‑OSS 随例数递增但提升有限。性能指标表明 LLM 能在规模化场景下辅助指令识别。

**⚠️ 局限性**

局限性：①数据来源受 GitHub 代码搜索上限影响，可能遗漏部分 gh-aw 文件；②仅关注已出现 ≥120 天变更的文件，偏向已维护的工作流；③LLM 的标签错误主要来源于对背景知识的误判，未完全验证指令有效性；④FRE 可读性指标混入代码与模板，未必代表人类可读性；⑤实验仅在 2026 年 8 月前的仓库，无法覆盖未来新出现的工作流模式。

---

## 221. Stable Geometry with Divergent Task Evidence for Efficient Long-Horizon Agent Compression

**arXiv ID:** 2609.27332 | [PDF](https://arxiv.org/pdf/2609.27332v1)

**作者:** Mingxuan Wang `[一作]` (TierFlow Team), Jungong Han `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练、基于几何残差的外部压缩器 Geometry‑Guided Evidence‑Preserving Memory (GEMP)，用于长周期代理的历史记录压缩，先保留关键执行与任务证据，再用几何残差填补覆盖。

**💡 创新点**

创新点在于：① 区分几何冗余与任务证据，证明几何相似度不足以保证执行信息；② 设计两阶段压缩策略（先证据保护再几何补全）并提供在线更新与请求级保护；③ 通过冻结编码器实现无训练、可插拔的压缩方案。

**🔧 技术方法**

技术包括：冻结 Qwen3‑Embedding‑8B 编码器；基于 SVD 的几何残差评估；规则式证据筛选（近期交互、目标相关、状态变更、错误、来源）；贪婪搜索与正交基更新；在线删除阈值 δ 与长度保护；实验采用 WorkBuddyBench Full260 任务集。

**📊 数据集**

数据集：WorkBuddyBench Full260（260 任务，Code、Office、Security、Web 四大领域），以及基准 Fixed‑40 对比面板。还使用 171 条轨迹与 32 条轨迹的结构化冗余实验。

**📈 对比分析**

比较方法：与未压缩基线代理、窗口压缩、周期性摘要、PACE、LLMLingua‑2、Self‑Compact、ACON‑Core、Self‑GC、LRE、CoMem、SAM、SWE‑Pruner、Sculptor、ACM、gemrow 等现有上下文管理器进行对照。结果显示：平均 token 使用从 2.69 M 降至 2.11 M（≈21.4 % 降低），奖励保持在 69.87 → 70.18（几乎无损）。在线 ablation 证明随机补全和仅几何压缩性能较差。

**⚠️ 局限性**

局限性：① 在不同领域表现差异大，某些任务奖励下降；② 依赖冻结编码器和经验规则，缺乏全局优化；③ 仅评估单一代理（DeepSeek‑V4‑Flash）与特定任务集，泛化性待验证；④ 仅保护完整交互块，无法细粒度修改或合并；⑤ 未对压缩对推理延迟与内存占用进行全面评估。

---

## 222. FairTest: Search-Based Fairness Testing for Multi-Agent Reinforcement Learning Systems

**arXiv ID:** 2609.27309 | [PDF](https://arxiv.org/pdf/2609.27309v1)

**作者:** Xiaotong Wang `[一作]` (Macau University of Science and Technology), Xuan Xie `[通讯]` (Macau University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 FairTest，一种基于搜索的公平性测试框架，用来在部署前验证多智能体强化学习（MARL）策略是否产生公平的执行。

**💡 创新点**

创新点在于：①将三种fitness（已测公平性、预测公平性、决策不确定性）与遗传搜索和交叉/变异操作相结合；②使用抽象状态+公平性特征构建的随机森林预测器，提前评估候选执行的公平性；③采用 Pareto 前沿优先级排序，将预算聚焦在最可能产生公平失败的候选者上；④在多目标搜索中应用 MOSA 排序提升搜索效率。

**🔧 技术方法**

使用技术包括：遗传算法（交叉、变异、MOSA 选择）、DeepGini 作为不确定性度量、Q*‑irrelevance 抽象、随机森林公平性预测器、Jain Fairness Index、Pareto 前沿与拥挤度过滤。

**📊 数据集**

实验数据集为三种 MARL 环境：Multi‑Agent Particle Environment (MPE)、Level‑Based Foraging (LBF)、Predator‑Prey (PP)，并在两种 MARL 算法下测试：QMIX 与 IQL。

**📈 对比分析**

与随机、GMT、MASTest、STARLA 四种基线在相同 18,000 次运行预算下比较；FairTest 在所有六种环境‑算法组合中均显著提升公平失败检测量（最高提升 106%–412%，p<0.001，效应大小大）并在绝大多数情形下获得更高的失败覆盖率；基线往往仅检出极少量失败且覆盖率低。

**⚠️ 局限性**

局限性包括：仅验证了三种小规模环境和两种算法；实验成本高；公平性判定仅基于 Jain 指数，忽略奖励大小差异；阈值设定人为；预测器性能受训练样本和特征设计影响；搜索可能漏掉罕见或分散的公平失败；未评估大规模多智能体或不同 MARL 模型的泛化性。

---

## 223. EmbodiedSWE: Coding Agents for Long Horizon Dexterous Robotics

**arXiv ID:** 2609.27308 | [PDF](https://arxiv.org/pdf/2609.27308v1)

**作者:** Haoxiang You `[一作]` (Yale University), Canwen Xu `[通讯]` (ByteDance Seed)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个长周期、精细操作的机器人任务基准，利用大型语言模型驱动的编码代理从零开始求解这些任务，并将验证后的解码为多样化演示，用以微调通用视觉‑语言‑动作（VLA）策略，实现了从模拟到真实机器人的零样本迁移。

**💡 创新点**

①首次提出包含 28 个多阶段、接触密集任务的长周期基准；②展示前沿编码代理可在几小时内完成大部分任务；③设计分层多样化引擎将单一解扩展为数百条多样化轨迹；④证明由代理生成的演示可显著提升 VLA 的泛化能力并在真实机器人上实现实质性零样本迁移。

**🔧 技术方法**

大型语言模型（GPT‑5.6 Terra、GPT‑6 Astra 等）驱动的编码代理；检查点树、CMA‑ES 参数搜索等辅助工具；分层多样化流水线（Scene、Strategy、Phase、Noise 变换）；SmolVLA 视觉‑语言‑动作模型；Isaac Lab/Sim 物理仿真；3D Gaussian Splatting 真实场景重建。

**📊 数据集**

28 个长周期任务（组装、打包、拼图、柔性/液体处理、切割、运动‑操作），覆盖 5 种机器人构体（Franka、xArm7、Kinova Gen3、双臂 Franka、Unitree G1）；生成的 500+ 真实演示轨迹；SmolVLA 训练所用的多样化演示数据。

**📈 对比分析**

与多款前沿编码代理（Claude、Codex 等）对比：最高成功率从 11%（GPT‑5.6 Terra）提升到 82%（GPT‑6 Astra）。相较于基线 PPO RL，编码代理在同等墙时/令牌预算下表现更优。使用脚本‑仅随机化生成的演示 vs 代理多样化生成的演示：在 6 个任务的离散测试集上，后者平均分 0.233，远超脚本仅 0.066。灯具拆解任务中，模拟训练后零样本迁移完成率在四个阶段分别为 100%、80%、30%、20%。

**⚠️ 局限性**

仍有极难任务未被成功解决；生成演示需要高算力且迭代成本高；编码代理依赖完整仿真状态，可能导致视觉感知缺陷；RL 训练仍面临探索与噪声挑战；多任务/构体迁移效果受限；生成的演示在真实机器人上的适配仍需进一步提升。

---

## 224. Large Knowledge Model: From Papers to a Scientific Reasoning Landscape

**arXiv ID:** 2609.27297 | [PDF](https://arxiv.org/pdf/2609.27297v1)

**作者:** Yuan Huang `[一作]` (DP Technology), Weinan E `[通讯]` (AI for Science Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个大规模科学知识基础设施——Large Knowledge Model（LKM），将论文转换为来源可追溯的推理图，并通过问题、工作流、证据三视图统一组织知识，以支持科学检索、问答和研究规划。

**💡 创新点**

创新点：①源可追溯的论文级推理图表示，②统一的图与向量访问层实现语义检索与结构遍历，③三视图科学推理景观（问题、工作流、证据）对齐，④在检索与问答任务上实证验证提升性能。

**🔧 技术方法**

技术：图数据库与向量检索、语义嵌入（512维向量）、SHA-256内容哈希实现全局唯一标识、ByteHouse持久化、图与向量融合的Hybrid检索。

**📊 数据集**

数据集：基于S2ORC等文献共计4千万论文的推理图；检索评测使用PaSaMaster 244查询；问答评测使用ChemBench、PubMedQA、SciBench、ScholarQA；外部检索接口包括Science Navigator、Google Scholar等。

**📈 对比分析**

比较方法：与Science Navigator及其他检索/问答基线比较；检索评测使用NDCG@5/10/20；QA评测使用固定GPT‑5.4模型与LKM检索、Science Navigator检索、Web搜索等。性能：Hybrid API在NDCG上分别领先0.75–1.65个百分点；QA准确率在ChemBench +9.30%、PubMedQA +4.20%、SciBench +14.69%；ScholarQA中图检索提升citation F1 5–7个百分点。

**⚠️ 局限性**

局限：①对实验细节、代码、数据等非文本信息支持不足；②同义表达的哈希唯一性鲁棒性有限；③对动态更新与推理错误修正机制尚未成熟；④对低资源领域文献的覆盖仍不足。

---

## 225. Parallel Multi-Fidelity Expected Improvement Method for Efficient Global Optimization

**arXiv ID:** 2609.27328 | [PDF](https://arxiv.org/pdf/2609.27328v1)

**作者:** Zhendong Guo `[一作]` (Xi'an Jiaotong University), Jun Li `[通讯]` (Xi'an Jiaotong University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种名为 Filter-GEI 的多源逼近式全局优化算法，能够在每次迭代中并行分配高、低保真样本，显著提高优化效率。

**💡 创新点**

创新点在于：① 在 GEI 获得的候选点上加入自适应阈值过滤器，根据高低保真模型相关性动态决定哪些点使用昂贵的 HF 模拟；② 通过层次聚类控制每次迭代的 HF 与 LF 采样数量，实现并行化。

**🔧 技术方法**

技术手段包括：协同克里金（co‑kriging）构建多保真 surrogate；Generalized Expected Improvement (GEI) 作为采样准则；自适应阈值公式与层次聚类；以及对阈值的标准差比值权重调节。

**📊 数据集**

使用了五个经典数值测试函数（Forrester、Hartman、Ackley 等）以及一个以 ANSYS CFX 计算的涡轮叶片能量损失工程案例（18 维参数中选 7 维），其中低保真模型采用网格细化比例 0.1 的 RANS 模拟。

**📈 对比分析**

与增量 EI、VF‑EI、GEI、EI 等基线算法比较，Filter‑GEI 在 HF 采样量、迭代次数与收敛速度上均优于对手，成功率在所有测试中达到 100%，且在工程案例中实现了最快的收敛曲线。

**⚠️ 局限性**

局限性包括：① 需要高低保真模型高度相关才能有效过滤；② 随着维度或样本数增大，协同克里金矩阵求逆成本上升；③ 该方法对初始设计与聚类阈值敏感，可能需要经验调参。

---

## 226. Breaking Weather-Content Coupling: Type-Severity Guided Progressive Disentanglement for All-in-One Infrared Restoration

**arXiv ID:** 2609.27317 | [PDF](https://arxiv.org/pdf/2609.27317v1)

**作者:** Xinyao Wang `[一作]` (Xi'an Jiaotong University), Fan Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种全景式红外图像去噪恢复框架 TSGPD-IR，通过天气语义引导、代理监督的区域严重度估计和多源专家路由，实现对雨、雾、雪、云等多种天气退化的自适应恢复。

**💡 创新点**

创新点包括：① 将天气类型与区域严重度分离为两级引导，进而实现渐进式解耦；② 引入代理监督的区域严重度估计器，无需手工标注；③ 设计层级专家路由策略，在天气特定池与严重度匹配专家间动态切换，兼顾全局语义与局部细节。

**🔧 技术方法**

核心技术包括：多层天气与语义共引导提示生成模块（WS-MPG）、代理监督区域退化估计器（PS-RDE）以及多源协作专家选择策略（MCESS）；网络采用 Transformer‑基编码器‑解码器骨干，配合温度缩放 softmax、SparseTopK 路由与多任务损失。

**📊 数据集**

使用的主要数据集为 WeatherIR（合并 AWMM‑100K、M3FD‑TIR、CUHK‑CR1/CR2），涵盖雨、雾、雪、云四类退化；训练时随机裁剪 160×160、输入 192×192，采用 8 张 RTX‑2080 GPU 进行训练。

**📈 对比分析**

与 PromptIR、MoCE‑IR、PPFN 等全景式方法对比，在四类退化上均取得最优平均 PSNR（29.57 dB）、SSIM（0.9213）和 LPIPS（0.0922），比 PromptIR 提升约 2.96 dB PSNR，LPIPS 下降 39.7%。在单任务设置下，同样取得所有四类退化的最优或次优指标。

**⚠️ 局限性**

局限性包括：① 需要较大模型和两阶段训练，训练成本高；② 代理监督的严重度估计依赖退化对齐，可能在极端天气或新退化类型下表现不足；③ 目前未针对多模态融合或实时推理进行优化。

---

## 227. CoRe-WAM: Correspondence-Aligned Temporal Residuals for World Action Models

**arXiv ID:** 2609.27314 | [PDF](https://arxiv.org/pdf/2609.27314v1)

**作者:** Bin Zhou `[一作]` (Hong Kong University of Science and Technology), Kani Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为CoRe‑WAM的世界动作模型，利用对应对齐的视觉变化来增强机器人在操控过程中的时序感知，核心模块为TraceDelta；

**💡 创新点**

创新点在于先通过冻结的轨迹估计器（Trace Anything）将历史视觉特征对齐到当前位置，再在同一预训练特征空间内做符号差分，并通过轻量级适配器将结果作为残差注入现有的视觉‑语言‑动作接口；

**🔧 技术方法**

技术主要包括：冻结的视觉‑语言预训练编码器（Motus）、Trace Anything对应估计器、Transformer联合注意力架构、轻量级残差适配器以及动作侧的低秩更新；

**📊 数据集**

使用的数据集为RoboTwin 2.0（50个多阶段操控任务）以及Astribot S1的物理实测数据；

**📈 对比分析**

与Motus、Fast‑WAM、4D‑WAM等基线比较，CoRe‑WAM在RoboTwin 2.0上取得92.22%（清洁）和89.60%（随机）成功率，比分母提升约3.6点；在StarVLA策略中亦获得9.5点的清洁成功率提升；

**⚠️ 局限性**

主要局限是在线对应估计与特征对齐的计算开销导致约510 ms的推理延迟，需要进一步压缩对应估计和特征复用以提升闭环响应速度。

---

## 228. StateComp: Learning When to Compress History in Long Horizon Agents

**arXiv ID:** 2609.27298 | [PDF](https://arxiv.org/pdf/2609.27298v1)

**作者:** Mingxuan Wang `[一作]` (TierFlow Team), Jungong Han `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种基于当前代理状态的历史交互压缩框架 State Conditioned Compression，动态判断哪些历史交互可被安全替换；

**💡 创新点**

主要创新在于将压缩决策与执行分离，并通过两阶段注释构建安全压缩边界；

**🔧 技术方法**

采用冻结语言模型的隐藏状态训练不平衡感知路由器，并用连续跨度与门控执行压缩；

**📊 数据集**

在 WorkBuddyBench 260 任务集（Code、Office、Security、Web）上评估；

**📈 对比分析**

与多种现有上下文管理基线相比，token 总量下降 52.27% 并保持甚至略提升平均奖励，速度提升 12.67×；

**⚠️ 局限性**

受限于路由精度与对未来依赖的判断，压缩安全性与泛化仍需改进。

---

## 229. PotARCin: Multi-Dimensional Evaluation of Skill Acquisition in Abstract Reasoning Tasks

**arXiv ID:** 2609.27288 | [PDF](https://arxiv.org/pdf/2609.27288v1)

**作者:** Claas Beger `[一作]` (Santa Fe Institute), Melanie Mitchell `[通讯]` (Santa Fe Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 PotARCin benchmark，扩展 ARC 评估框架，增加了 Definition、Classification、Constrained Generation、Editing、Inversion 五个维度，系统测评模型对任务抽象规则的掌握程度。

**💡 创新点**

创新点在于将抽象推理任务从单一输出网格扩展为多维度、多样化的评测；使用程序生成器与验证器实现可自动化的生成与检验，构造新实例和错误样本；同时发布了 P-ARC 50 题手工构造的测试集。

**🔧 技术方法**

采用程序合成技术（生成器/验证器）与自动化脚本，结合自定义评测维度；利用生成采样、错误类型腐蚀、编辑距离约束和自一致性分析来检验模型的规则理解与泛化能力。

**📊 数据集**

主要使用 ARC-AGI-1 训练集（400 题）、ARC-AGI-1 测试集、ARC-GEN 稳定数据集、RE-ARC 验证器、H-ARC 人工错误集，并创建了 P-ARC 50 题的手工生成器/验证器与错误样本。

**📈 对比分析**

通过对比 GPT‑5.4、Gemini 3.1 Pro、Claude Opus 4.6、Kimi K2.5、MiniMax M2.5 等五个前沿模型，计算标准 ARC 输出精度与 PotARCin 全任务准确率；发现标准输出准确率与全任务准确率相差 25–52 个百分点，模型在多维度评测中的相对排名发生重排，P‑ARC 上全任务准确率仅 1–8 %。

**⚠️ 局限性**

局限包括：生成器/验证器的质量和与任务规则的一致性不易保证；Classification 可能受多重可行规则影响；不同模型的 token 消耗差异导致公平性受限；P‑ARC 的难度评估基于内部可行性检查，缺乏正式人类求解率；以及对生成实例多样性与模型创造性尚未深入分析。

---

## 230. SR-Fraud: An Outcome-Supervised Reflective LLM Agent Framework for Non-Stationary Payment Fraud Detection

**arXiv ID:** 2609.27287 | [PDF](https://arxiv.org/pdf/2609.27287v1)

**作者:** Xuwei Tan `[一作]` (Ohio State University), Xueru Zhang `[通讯]` (Ohio State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 SR-Fraud 框架，将无状态 LLM 交易风险评分与离线 outcome‑supervised 反思机制结合，实现实时支付欺诈检测；

**💡 创新点**

将 LLM 预测与离线反思解耦，利用 Hybrid Episodic Window 进行情境异常检测，并通过确定性验证器构建可解释的符号知识状态；

**🔧 技术方法**

使用无状态 LLM（GPT/Opus/Sonnet）决策代理、Hybrid Episodic Window、outcome‑supervised 反思循环、Wilson 置信界限验证、符号知识库；

**📊 数据集**

基于公司真实支付风险流的 16,140 交易样本（9 周，5.04% 欺诈率）进行时间序列重放实验；

**📈 对比分析**

与 CatBoost、Seq‑GRU、Direct LLM、Reflection 等基线在保持相同阻断率的情况下比较，SR‑Fraud 在精准率、召回率、F1 及美元加权召回上分别提升约 0.3/8.1/3.8/7.8 点，且在新兴欺诈爆发中表现尤佳；

**⚠️ 局限性**

使用固定两周期成熟延迟、单一内部流数据、未进行多重检验校正、未完成完整线上评估，且对不同 LLM 版本的泛化能力有限。

---

## 231. Hunyuan-A13B Technical Report

**arXiv ID:** 2609.27284 | [PDF](https://arxiv.org/pdf/2609.27284v1)

**作者:** Tencent Hunyuan Team `[一作]` (Tencent Hunyuan Team), Yiqi Chen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一款采用稀疏Mixture-of-Experts架构、激活13B参数、总参数80B的开源大语言模型Hunyuan-A13B，并在20T高质量数据上预训练，随后通过精细化的SFT和RL提升推理与编程能力。

**💡 创新点**

通过共享专家+64个细粒度专家的MoE设计、STEM强化预训练语料、双模式链式思考（快速/慢速）以及专门的多阶段RL强化，兼顾推理精度与推理效率，实现小激活量、长上下文、低延迟的高性能。

**🔧 技术方法**

采用MoE+GQA注意力、SwiGLU激活、动态学习率调度、自动前缀缓存、量化推理（INT8/FP8）以及vLLM/SGLang/TensorRT-LLM等框架进行高吞吐量推理。

**📊 数据集**

利用20T token的预训练语料，其中精细化的STEM子集约250B token；后期SFT使用数学、代码、逻辑、科学推理等专业Chain-of-Thought数据；RL采用GRPO与沙箱代码执行、判定模型等多种奖励信号。

**📈 对比分析**

通过公开基准（MMLU、MATH、GSM8K、BBH、CodeEval、LongBench、FRAMES、RULER等）与同类MoE/稠密模型（Hunyuan-Large、Qwen2.5-72B、Qwen3-A22B、DeepSeekR1、Gemini 2.5 Pro）对比，Hunyuan-A13B在多项指标上达到或逼近更大模型，并在推理吞吐量上超过同类。

**⚠️ 局限性**

虽然激活参数低，但在极长上下文（>128K）或极高并发推理时仍有性能衰减；代码执行与工具调用的鲁棒性仍需改进；在部分高难度专业领域（医疗、法律）与对话一致性上略逊于大型封闭模型。

---

## 232. Turning Safety into Competence: Minimally Exploitable Robot Policies via Safety-Filtered Reinforcement Learning

**arXiv ID:** 2609.27312 | [PDF](https://arxiv.org/pdf/2609.27312v1)

**作者:** Ruihan Wu `[一作]` (Johns Hopkins University), Haimin Hu `[通讯]` (Johns Hopkins University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 Safety to Competence (S2C) 两阶段强化学习框架，在竞争性机器人交互中先训练鲁棒安全滤波器，再将其嵌入环境中学习任务策略，最终实现既安全又具竞争力的机器人控制。

**💡 创新点**

创新点在于将安全与任务目标分离，并证明在使用完美安全滤波器时，滤波后游戏的纳什均衡仍是原安全-竞争马尔可夫游戏中安全策略的非可利用均衡；此外，采用对抗式安全学习与多智能体自对弈相结合的实用实现，显著降低了策略的可利用性。

**🔧 技术方法**

技术上使用：1）MAGICS 对抗式强化学习合成鲁棒安全滤波器；2）IPPO (或 MAPPO) 进行多智能体任务策略学习；3）qCBF 近似完美安全滤波器；4）基于仿真与实际硬件的实验平台（Unitree Go2 四足机器人、Vive Lighthouse 定位、Jetson Nano 控制）。

**📊 数据集**

主要数据集是基于仿真环境的“touchdown”比赛（对称与非对称两种版本），包含 5 组随机种子训练，最终在 31200 次对弈与 10 轮真实人类对战中评估；硬件实验中记录人类对手的峰值速度与加速度。

**📈 对比分析**

与四种安全强化学习基线（Early Termination、Reward Penalty、CPO、PPO-Lagrangian）以及在部署时加滤波的版本进行对比。S2C 在仿真中赢率最高（约 76.5%），Elo 最高（139），可利用性最低（0.11），安全率约 90%；硬件测试中攻击/防守成功率分别为 70%/60%，安全率 80%/90%，远优于基线。

**⚠️ 局限性**

局限性包括：1）理论证明依赖完美安全滤波器，实际中滤波器近似误差未给出正式均衡保证；2）目前仅在结构化的模拟与受控硬件环境验证，未在开放式真实世界任务中测试；3）对安全滤波器的计算成本与实时性仍需进一步优化。

---

## 233. DUGM-R: Uncertainty-Aware Dynamic Grid Mapping and Risk-Triggered Recovery for Learned Local Navigation

**arXiv ID:** 2609.27338 | [PDF](https://arxiv.org/pdf/2609.27338v1)

**作者:** Haoyun Feng `[一作]` (Imperial College London), George Mylonas `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一种基于动态不确定性网格映射（DUGM）与风险值函数（RVF）触发的后训练恢复策略的本地导航框架；

**💡 创新点**

创新点在于将运动估计不确定性融入机器人中心的网格观测，利用冻结的本地策略训练得到的风险评估来实时决定是否切换到独立的恢复策略，实现零样本迁移和残余碰撞率显著下降；

**🔧 技术方法**

采用LiDAR感知与Kalman滤波进行动态跟踪，构建DUGM；使用PPO训练本地导航策略、RVF与恢复策略；利用有限时域风险值函数评估碰撞风险；在NVIDIA Isaac Sim仿真和真实TurtleBot3机器人上进行评估；

**📊 数据集**

使用自制的二维仿真器与NVIDIA Isaac Sim临床物流环境（包含走廊、办公室、接待区等）以及真实机器人在八个未见室内场景中的数据；未使用公开数据集；

**📈 对比分析**

与DWA、Nav2 MPPI、ORCA、DRL-VO、OGM–Lagrangian等基线进行对比；在90个held‑out Isaac Sim任务中，DUGM+恢复实现了76.7%的成功率和13.3%的碰撞率；在24条真实机器人跑中成功率为83.3%，碰撞率仅为4.2%；性能优于所有基线；

**⚠️ 局限性**

对可靠的在线运动估计依赖较强；风险评估与恢复机制缺乏形式化安全保证；在训练分布外可能出现不稳定的风险触发；实验规模有限，未覆盖更复杂多变的真实环境；

---

## 234. Evolving Inspectable O-RAN Slicing xApps with LLMs

**arXiv ID:** 2609.27337 | [PDF](https://arxiv.org/pdf/2609.27337v1)

**作者:** Faezeh Dehghan Tarzjani `[一作]` (University of Southern California), Bhaskar Krishnamachari `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用LLM进行进化搜索，自动生成可检查、可编辑的Python程序作为O-RAN切片控制器，并在POWDER 5G测试平台上直接部署执行。

**💡 创新点**

创新点在于将控制逻辑从神经网络参数转移到可读源代码，实现了决策过程的可解释性和快速人工修正，同时通过LLM指导的进化搜索显著提升切片资源分配效果。

**🔧 技术方法**

技术手段包括：大型语言模型（Claude/Gemini）生成/修改控制程序，离线可执行评估与模拟器交互，基于进化算法的候选筛选与选择，以及与近RT RIC的REST/E2接口集成。

**📊 数据集**

使用的数据集为：NSF POWDER 5G 物理测试平台的真实信道和吞吐数据；四切片仿真采用 16 条随机种子生成的信道轨迹，覆盖高/低信道衰减与噪声；以及对比 PPO 训练得到的策略。

**📈 对比分析**

对比方法包括：静态最优切分、PPO 强化学习、两切片硬件实验与四切片仿真；进化搜索在相同提议预算下平均提升 32.1%（两切片）/ 51.0%（四切片）相对提示基线，种子进化可进一步提升 64.9% 的相对收益；在硬件上，进化控制器比静态分配提高 44.5% 最佳努力吞吐。

**⚠️ 局限性**

局限性：仅在两切片硬件实验中验证，四切片测试仅限仿真；依赖事先校准的信道模型，若真实信道漂移需重新校准或人工修正；LLM 生成的代码仍可能出现逻辑错误，需人工审查；对实时性和大规模部署的可扩展性尚未完全评估。

---

## 235. A Hybrid Iterative Deep Ritz Method for Elliptic Interface Problems

**arXiv ID:** 2609.27325 | [PDF](https://arxiv.org/pdf/2609.27325v1)

**作者:** Tianhao Hu `[一作]` (Chinese University of Hong Kong), Yifeng Xu `[通讯]` (Shanghai Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对二阶椭圆介质接口问题，提出一种混合迭代深度 Ritz 方法（H-IDRM），通过新的混合变分形式将问题拆解为一系列凸子问题，并利用层次化的层集神经网络（LSNN）逼近解和通量。

**💡 创新点**

创新点包括：① 将传统的鞍点结构转化为可训练的凸最小化问题；② 通过层集函数对接口几何进行隐式编码，避免显式的接口采样；③ 结合混合变分、蒙特卡洛积分与迭代深度 Ritz，提供严谨的收敛分析与误差估计；④ 在高维、多层次、复杂接口下仍保持良好数值稳定性。

**🔧 技术方法**

使用技术主要有：深度 Ritz 方法、混合变分形式、层集神经网络（LSNN）与平滑 Heaviside 变换、蒙特卡洛采样、迭代深度 Ritz 方案（IDRM）与惩罚项、Adam 优化器。

**📊 数据集**

使用的数据集为一系列人工构造的介质接口问题：2D 圆形、棋盘格、3D 球面、隐式正弦界面以及 5 维球面问题，均为解析或高精 FEM 参考解。

**📈 对比分析**

与 Deep Ritz（DRM）、Domain‑Decomposition PINN（DD‑PINN）以及 H-IDRM（无 LSNN）进行对比。实验表明 H-IDRM 在相同网络容量与训练预算下，相对误差降低 1–2 个数量级，收敛曲线更平滑，尤其在复杂接口与低正则性场景中表现突出。

**⚠️ 局限性**

局限性包括：① 需要预先给定或估计层集函数，若接口未知需额外处理；② 训练成本仍受蒙特卡洛采样点数量和网络宽度限制；③ 对极其不规则或非 Lipschitz 接口的理论保证尚不充分；④ 需要调节多种惩罚与步长参数，实际应用中可能需要经验选择。

---

## 236. Live Assistant: Learning Whether, When, and Whom to Assist in Real-World Live Social Streams

**arXiv ID:** 2609.27303 | [PDF](https://arxiv.org/pdf/2609.27303v1)

**作者:** Shujian Gao `[一作]` (Fudan University), Yu-gang Jiang `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种混合主动性、角色条件化的直播助手框架，能够在实时多模态社交流中决定是否行动、何时行动、针对谁以及传达何种信息。

**💡 创新点**

创新点在于将直播助手任务统一为四个耦合决策，构建了端到端的结构化协议，并通过Marker Aware Multiturn Supervised Fine Tuning (MA‑MSFT) 与 Streaming Multiturn GSPO (SM‑GSPO) 两阶段训练，有效强化稀疏结构决策与自生成轨迹的信用分配。

**🔧 技术方法**

使用的核心技术包括：Qwen3‑Omni 30B 思考者作为基模型，自动同步多模态直播轨迹数据引擎，结构化生成协议，MA‑MSFT 结构化监督，SM‑GSPO 基于GSPO的自回归强化学习，和多模态内容评估判别器。

**📊 数据集**

数据集由 123 个直播间的 2,049 条连续片段（共 320.8 小时）构成的优化语料库，以及 14 个直播间的 275 条完整片段（共 38.4 小时）构成的评估基准，涵盖 9 类内容和 7 种语言，包含评论、礼物、观众动态等平台信号。

**📈 对比分析**

在统一评估协议下，相比传统直播视频模型与通用多模态模型，本文方法在状态决策、接收者路由与任务选择上取得显著提升：状态整体准确率 71.14%，接收者整体准确率 72.67%，任务准确率 58.41%，并在语义内容评分上位列最前。

**⚠️ 局限性**

局限性包括：对未见域和平台的泛化能力不足；主持人相关决策数据稀缺导致路由性能相对较低；评估采用自动判别器，可能受模型偏差影响；十秒决策间隔限制了干预的时序精度；实际部署需考虑用户同意、隐私与审计等伦理与安全问题。

---

## 237. Ruby-ASR: Evidence-Preserving Supervision for Joint Orthographic and Lexical-Reading Recognition

**arXiv ID:** 2609.27289 | [PDF](https://arxiv.org/pdf/2609.27289v1)

**作者:** Hao Shi `[一作]` (Independent Researcher), Zixiong Su `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

改进日语语音识别目标，将正字面转写与对应发音绑定，产生类似今日[きょう]的ruby序列。

**💡 创新点**

在目标层面消除读音歧义，使用span‑bound ruby结构直接在识别中标注发音，结合mora‑CTC辅助监督并通过证据约束构造训练目标。

**🔧 技术方法**

采用Qwen3‑ASR语言模型解码器、Transformer编码器、mora‑CTC头、OpenJTalk与语言模型进行候选消歧、bfloat16 AdamW训练等技术。

**📊 数据集**

训练语料约93.1M语音‑文本对（171.5k小时），来源包括ReazonSpeech、Common Voice 26、JSUT子集，评测使用JSUT‑BASIC5000、CSJ、JSUT‑Book、Common Voice 8、TEDx。

**📈 对比分析**

与Whisper、Kotoba、ReazonSpeech、Qwen、Kana‑Whisper等基线对比，Ruby‑ASR‑sub在加权Raw CER上取得8.51%（最低），Ruby‑ASR‑ver在加权SA‑CER上取得6.59%（最低），读音CER亦分别为3.75%与4.01%，均优于基线。

**⚠️ 局限性**

对稀有/未见正字音对仍有较高错误率，语音识别瓶颈在自发语料上显著，decoder是否真正依赖语音证据仍未完全验证，边界溢出虽较低但仍存在。

---

## 238. Memory Control Signals Emerge Before Action in Long Horizon Agents

**arXiv ID:** 2609.27286 | [PDF](https://arxiv.org/pdf/2609.27286v1)

**作者:** Mingxuan Wang `[一作]`, Jungong Han `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究长周期语言模型代理的记忆需求，发现预动作隐藏状态已编码压缩与召回信号，并基于此设计了PaMER框架进行状态引导压缩与步骤级证据检索。

**💡 创新点**

创新点在于利用预动作隐藏状态预测压缩与召回需求，将该预测作为在线控制器，融合压缩、外部存储与精细检索，实现高效记忆管理。

**🔧 技术方法**

技术包括线性/MLP探针分析隐藏状态、使用Qwen3.5‑9B冻结模型提取特征、对旧块摘要压缩、外部检索记忆、步骤级证据选择与恢复。

**📊 数据集**

数据集包括WorkBuddyBench（260任务）和Pilot40（40任务）用于性能评估，以及额外的600题长轨迹用于标注压缩/召回决策。

**📈 对比分析**

在WorkBuddyBench和Pilot40上与滑动窗口、周期摘要、PACE、LLMLingua等多种基线比较，PaMER+在保持竞争任务分数的同时将token消耗降低约70%~80%，显著提升成本效益。

**⚠️ 局限性**

局限性包括模型与任务域差异导致性能波动、需人工标注的决策数据、检索误差可能影响任务完成率，以及在极长序列或高并发场景下仍需进一步优化。

---

## 239. Pacing Equilibria in Abstract Mechanisms

**arXiv ID:** 2609.27285 | [PDF](https://arxiv.org/pdf/2609.27285v1)

**作者:** Salam Afiouni `[一作]` (Columbia University), Christian Kroer `[通讯]` (Columbia University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了多层机制下的统一的加速计价（pacing）均衡理论，分析了从单品一价拍卖到多槽位置拍卖、求解最大化支付的优化式机制再到一般抽象机制的完整框架，并给出了存在性、唯一性、可计算性与经济效率的理论保证；

**💡 创新点**

创新点在于证明即使在多槽位置拍卖中单品拍卖的市场均衡解释失效，仍可通过对支付单调性与平滑化的结构性假设，恢复加速计价均衡的存在、唯一性以及通过Eisenberg‑Gale型凸规划实现的效率与收益最大化；

**🔧 技术方法**

主要技术包括：凸优化（Eisenberg‑Gale型程序）、抽象机制的联立不动点与半格结构、随机平滑化导致支付连续性、基于预算调整的Pace动态与乘法权重更新的平均收敛分析；

**📊 数据集**

实验使用合成数据集，分别在第一价位置拍卖、共享容量分配、匹配机制与比例共享机制下生成多种规模的市场实例；

**📈 对比分析**

通过对比均衡与最优液体福利、需求违背率、动态收敛速率，实验显示加速计价均衡的液体福利损失在5%以下，Pace动态在理论界限的10‑30%内收敛，乘法权重更新在时间平均上满足约束，整体性能优于无结构化计价；

**⚠️ 局限性**

局限性包括：结果高度依赖支付单调性与连续性假设，复杂机制中需要适当的拆分与随机平滑，Tie‑breaking 的选择可能影响存在性，理论与实验均基于合成模型，实际平台中预算、价值分布与机制细节的差异可能导致偏差。

---

## 240. Multi-View Fusion for Encrypted C2 Detection: A Leakage-Controlled Measurement Study of Evaluation Pitfalls

**arXiv ID:** 2609.27311 | [PDF](https://arxiv.org/pdf/2609.27311v1)

**作者:** Hoang-Huy Nguyen-Huu `[一作]` (Academy of Cryptography Techniques), Khuong Nguyen-An `[通讯]` (Ho Chi Minh City University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估多视图融合在加密命令与控制（C2）流量检测中的有效性，并在严格的泄漏控制协议下对实验过程进行测量与纠正。

**💡 创新点**

提出并验证三项关键测量发现：①频率编码泄漏可提升0.28 F1；②基于流计数的正例率55.1%是单位分析误差，真正的端点平衡率仅4.2%；③32%的捕获样本完全缺失C2流，导致标签不完整；并证明在这些误差控制下，多视图融合的实际提升仅为0.022 F1。

**🔧 技术方法**

使用机器学习方法（随机森林、XGBoost）进行早期融合、后期平均、堆叠组合；对TLS握手和流统计两视图分别进行频率编码、缺失值插补和Boruta特征选择；通过对抗攻击模拟（特征空间攻击）评估鲁棒性。

**📊 数据集**

基于62个真实的 Cobalt Strike 捕获（共17,577条TLS流），包含9,688条C2流和7,889条正常流，数据来自 malware-traffic-analysis.net 的公开抓包。

**📈 对比分析**

对比方法：在按捕获为组的 StratifiedGroupKFold 交叉验证下，清洁条件下早期融合+随机森林得到最高 F1≈0.878；单视图最高为0.859；融合增益仅0.022；在对两视图同时进行特征空间攻击后，所有模型 F1 降至≈0.07，低于 always-positive 基线 0.711。

**⚠️ 局限性**

局限性：样本仅来自 Cobalt Strike，时间分布偏向 2022 年，缺失或不完整的标签导致标签噪声；实验未覆盖不同 C2 框架和 TLS 版本（如 JA4+）；对抗攻击模型为简单的特征替换，未模拟自适应攻击；评估在实际生产环境下可能与实验结果差异显著。

---

## 241. DRSR: Learning Set-Level Deletion Risk for Efficient Long-Horizon Agents

**arXiv ID:** 2609.27276 | [PDF](https://arxiv.org/pdf/2609.27276v1)

**作者:** Mingxuan Wang `[一作]` (TierFlow Team), Jungong Han `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于关系集风险的历史压缩方法，直接评估删除集的整体风险并在运行时进行安全裁剪；

**💡 创新点**

创新点在于：①使用离线对照实验对完整历史块组合的删除风险进行监督；②构建关系式特征与集级编码器，直接预测删除集风险而非单块评分；③加入弃权机制保证安全性；

**🔧 技术方法**

技术包括：关系特征提取（注意力密度、隐藏状态相似度）、集级编码器与轻量级风险预测网络、离线对照标签生成、在线风险门控裁剪；

**📊 数据集**

使用WorkBuddyBench Full260（80 Code、50 Office、60 Security、70 Web任务集）和Eval40十任务子集进行评估；

**📈 对比分析**

与未压缩代理、结构化裁剪、CoMem、PACE、Periodic Summary等方法对比，DRSR在Full260上奖励从0.699提升至0.802，Token使用下降20.8%；在Eval40上奖励0.794、Token使用减少35.8%；

**⚠️ 局限性**

局限性包括：需完整轨迹进行离线监督；风险预测模型对新领域的泛化仍有限；裁剪过程受候选集大小限制，过度裁剪可能降低奖励；

---

## 242. Reflection-Aware Reasoning for Non-Line-of-Sight Pedestrian Localization

**arXiv ID:** 2609.27346 | [PDF](https://arxiv.org/pdf/2609.27346v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 243. Forced Yet Free: What Magicians' Forcing Reveals Beyond Intentional Binding

**arXiv ID:** 2609.27416 | [PDF](https://arxiv.org/pdf/2609.27416v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 244. Can One Adapted Model Do It All? Fine-Tuning Strategy Selection for Customer Support LLMs

**arXiv ID:** 2609.27262 | [PDF](https://arxiv.org/pdf/2609.27262v1)

**作者:** Md Tahmid Rahman Laskar `[一作]` (Dialpad Inc.), Shashi Bhushan TN `[通讯]` (Dialpad Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在客户支持系统中，是否应该使用单一模型进行多任务训练，还是使用多个专门模型来处理不同的任务。通过对13个模型的比较，得出多任务全微调是最佳的操作默认设置。

**💡 创新点**

提出了多任务全微调在所有模型规模中表现最佳的结论，并指出专门模型在目标任务上表现良好但在其他任务上表现不佳，强调了可靠路由的重要性。

**🔧 技术方法**

使用了多种技术，包括单任务和多任务微调、LoRA适应、顺序微调和模型合并等。

**📊 数据集**

使用了八个客户支持数据集，包括四个公共数据集和四个专有数据集，总计约74.5k训练样本和8.7k评估样本。

**📈 对比分析**

通过对比不同的微调策略，发现多任务全微调在大多数模型中表现接近于任务专门模型，且只需一个检查点，无需路由。专门模型仅在目标任务验证显示出显著优势时才应采用。

**⚠️ 局限性**

研究的局限性包括模型覆盖不均、仅使用英语数据集、合并时α值选择的影响，以及生成任务的评估指标范围。

---

## 245. ASAP: Visual Analytics for Identifying and Analyzing Image Patterns in AI-generated Images

**arXiv ID:** 2609.27371 | [PDF](https://arxiv.org/pdf/2609.27371v1)

**作者:** Jinbin Huang `[一作]` (Arizona State University), Chris Bryan `[通讯]` (Arizona State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套交互式可视化系统ASAP，用于识别、分析和总结AI生成图像中的欺骗模式；

**💡 创新点**

创新点在于将CLIP视觉编码器改造为可解释的线性分类器，结合“忘记拼写”投影与特征蒸馏，生成16维正交可解释嵌入，并通过梯度传播得到多维像素相关图，形成统一的模式贡献度量；

**🔧 技术方法**

核心技术包括CLIP视觉编码器、正交投影（forget-to-spell）、线性蒸馏层、梯度归因（transformer attention）以及IsoMatch二维投影和交互式多视图可视化；

**📊 数据集**

实验使用了GAN生成的proGAN“horse”数据集（3万真/假）和扩散模型LDM生成的人脸数据集（2万假/7万真），并在公开基准上验证；

**📈 对比分析**

在公开基准上，ASAP的检测准确率与当前最优方法持平或略高，同时在用户研究中显示出高效的模式识别和解释能力；

**⚠️ 局限性**

主要限制包括对CLIP的依赖导致可迁移性受限、IsoMatch计算复杂度高、可解释性仅限于像素级归因、实时标注能力不足以及对大规模数据集的可扩展性待改进。

---

## 246. Quantization-Robust Unlearning through the Lens of Retain-Forget Loss Landscapes Interaction

**arXiv ID:** 2609.27355 | [PDF](https://arxiv.org/pdf/2609.27355v1)

**作者:** Jialu Wang `[一作]` (University of North Carolina at Chapel Hill), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对量化部署下LLM的量化鲁棒机器忘记框架，解决了量化后遗忘性能急剧下降的问题。

**💡 创新点**

通过曲率分析识别忘记敏感权重，使用噪声正则化平滑忘记损失曲面，并结合忘记关键层优化实现仅更新关键层的策略，显著提升了量化鲁棒性。

**🔧 技术方法**

利用二阶曲率（Fisher信息近似）评估敏感度，噪声正则化模拟量化噪声，层级选择算法挑选忘记关键层，结合post‑training量化方法（GPTQ、AWQ、SmoothQuant）实现量化部署。

**📊 数据集**

在MUSE（Harry Potter与BBC News）和TOFU（合成问答）两个公开基准上进行实验。

**📈 对比分析**

与SURE等现有方法对比，实验显示在4‑bit量化下遗忘率下降幅度减少约5–15%，且保持或提升了保留性能，在MUSE-Books上可实现约15%更强的遗忘，且仅需扰动约0.1%参数。

**⚠️ 局限性**

局限性在于验证范围主要集中在7B模型和4‑bit量化，对更大模型、不同量化精度以及多任务场景的泛化仍需进一步研究。

---

## 247. Credible AUctions via MPC Gadgets: Bounding Information Leakage Under Abort

**arXiv ID:** 2609.27402 | [PDF](https://arxiv.org/pdf/2609.27402v1)

**作者:** Matheus Venturyne Xavier Ferreira `[一作]` `[通讯]` (University of Virginia), Matheus Venturyne Xavier Ferreira (University of Virginia)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个名为Sequential Revelation Auction (SRA)的常数轮、可验证且收益最优的拍卖协议；

**💡 创新点**

创新点在于引入MPC分解原理，仅在最小的MPC模块中完成赢家判定，从而严格限制信息泄露，进而以有限经济惩罚实现可信拍卖；

**🔧 技术方法**

采用安全多方计算（MPC）、信息泄露界定、顺序信息公开以及有限经济惩罚等技术；

**📊 数据集**

未使用公开数据集，论文主要通过理论分析与数学证明完成；

**📈 对比分析**

通过定理与例证（如等收益分布）展示SRA在信息泄露被严格限定时的可行性，证明所需惩罚量≥∑_i F_i，且该界限紧确；

**⚠️ 局限性**

局限性在于对极重尾分布仍需更大惩罚，且实现需要在所有参与方上部署MPC基础设施。

---

## 248. The Complexity of Interference: When Rely/Guarantee Does Not Work

**arXiv ID:** 2609.27392 | [PDF](https://arxiv.org/pdf/2609.27392v1)

**作者:** Nisansala P. Yatapanage `[一作]` `[通讯]` (Australian National University), Nisansala P. Yatapanage (Australian National University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种新的“对所有变量保持”推理方法，用来验证 Ben‑Ari 并发垃圾回收器，从而在无法直接使用传统 Rely/Guarantee 时实现可组合的验证。

**💡 创新点**

创新点在于：①通过对抽象集合的普遍性质推导出组件的 rely 条件，避免了对本地变量的显式引用；②发现并形式化了核心属性 “can‑be‑marked”，解释了算法为何能正确运行；③为难以构造 Rely/Guarantee 条件的算法提供了一种可推广的验证思路。

**🔧 技术方法**

使用了 Rely/Guarantee 逻辑、Hoare 逻辑、VDM 语法、Owicki‑Gries 推理以及“对所有变量保持”定理；在实现层面还使用了 PVS 和 Murφ 进行对比验证。

**📊 数据集**

本文未使用实际数据集，验证全部基于理论模型和形式化证明。

**📈 对比分析**

与 Havelund 等人使用 PVS 或 Murφ 的完整系统验证相比，本文的验证更具可组合性，证明步骤更少、依赖关系更清晰，但目前尚未提供运行时性能评估。

**⚠️ 局限性**

局限性包括：验证工作仍在机械化过程中；该方法需要先识别出满足普遍性质的核心属性，适用范围可能受限；对实际系统的扩展验证尚未完成。

---

## 249. From LiDAR Maps to Visual Localization: Unified Visual Association for Robust Point-Line-Plane Pose Estimation

**arXiv ID:** 2609.27363 | [PDF](https://arxiv.org/pdf/2609.27363v1)

**作者:** Wentao Zhao `[一作]` (Shanghai Jiao Tong University), Jingchuan Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种统一框架，利用LiDAR地图渲染成可视化quasi-image，使摄像机在没有视觉地图信息的情况下完成全局定位和连续6-DoF跟踪；

**💡 创新点**

创新点包括：①将LiDAR地图转化为可与摄像机图像共享的quasi-image，②通过共享的点线面（PLP）视觉前端实现统一关联并保留2D–3D Provenance，③引入分布感知、可观测性互补的优化策略，依据候选关联分布评估方向性位姿信息并有选择地强化可靠的结构约束；

**🔧 技术方法**

技术实现涵盖quasi-image渲染与补全、深度+反射率伪彩色编码、共享PL-Net + LightGlue + OTPL-VIO点线匹配、可观测性分析、分布感知加权、以及图优化（Gauss‑Newton）等；

**📊 数据集**

使用的数据集为公开的EuRoC MAV benchmark 以及自采集的室内序列（Illum.-1、Illum.-2、Occl.-1）；

**📈 对比分析**

与ORB‑SLAM、HLoc、Surfel Reloc、AirSLAM等单图重定位方法以及PPL、TC‑VIML、Plane‑Loc等结构地图定位方法进行对比。全模式在EuRoC上全局定位召回率>90%，跟踪ATE平均约2–3 cm；在自采集序列上跟踪误差约20–30 cm，优于其他基线，并在强光照变化和动态遮挡下表现更稳健；

**⚠️ 局限性**

局限性在于线面约束高度依赖地图的结构支持；当地图结构稀疏或噪声严重时，定位主要靠点匹配，线面约束效果有限。

---

## 250. Guides That Cause Actions: An Offline Study of Guide-Action Mutual Reinforcement in Multimodal Web Agents

**arXiv ID:** 2609.27353 | [PDF](https://arxiv.org/pdf/2609.27353v1)

**作者:** Chengguang Gan `[一作]` (Independent Researcher), Shiwen Ni `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 WebMRE 这一离线、可重复的 Web 任务基准，并通过该基准研究了人类导向的引导句与精准动作输出之间的相互强化效果。

**💡 创新点**

创新点包括：① 将多模态 Web 轨迹统一成可离线评测的数据集，并提供三方 Judge 审计的标签；② 设计了 deterministic 的离线评估协议，使得相同模型在任何跑都得到完全一致的分数；③ 通过强制前缀干预证明引导句是动作的因果通道，并首次提出 Mediated‑Guide GRPO 的离线奖励，能直接衡量引导对动作的影响；④ 通过多种解码顺序和尺度实验展示了该效应随模型规模增大而增强。

**🔧 技术方法**

主要技术包括：多模态 Qwen3.5-4B/9B 的微调；三方 Judge 审计与手工修正的标签治理；BLEU‑1/ROUGE‑L 与目标匹配的精确度评估；强制前缀干预实验与因果中介分析；离线 Reward 设计与 GRPO 的改造。

**📊 数据集**

使用的数据集是 WebMRE，包含 541 个 WebArena 任务、5,293 步骤，来源于 MAG 轨迹，且所有步骤均已通过三方 Judge 重新审核，并标注了截图新旧状态。

**📈 对比分析**

与 GPT‑5.5、Claude Opus 4.8、Gemini 3.5 Flash 的 zero‑shot 在同一离线指标上进行对比。Fine‑tuned Qwen3.5-4B/9B 在动作准确率上分别提升了 0.9–2.2 分，指导句质量提升亦显著，整体表现远超基线。

**⚠️ 局限性**

限制包括：基准仅覆盖 WebArena 生态，缺乏更广泛的站点与布局多样性；离线指标虽然可复现但无法直接映射到终端任务成功率；强化学习实验样本有限，未能显著提升模型性能。

---

## 251. Automotive mmWave Spinning Radar Place Recognition with Spatially Gated Feature-Correlation Representation

**arXiv ID:** 2609.27394 | [PDF](https://arxiv.org/pdf/2609.27394v1)

**作者:** Saimunur Rahman `[一作]` (CSIRO), Peyman Moghadam `[通讯]` (CSIRO)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于旋转鲁棒特征提取与空间门控特征相关聚合（SGCA）的汽车旋转FMCW雷达定位识别框架

**💡 创新点**

创新点在于：①通过圆柱形填充实现对雷达极坐标的旋转不变特征提取；②引入空间门控（keep/dustbin）对不稳定/模糊雷达区域加权；③利用特征相关聚合保留局部特征的对偶关系，提升辨识度；④端到端训练的旋转鲁棒三元组损失，进一步抵消航向变化影响

**🔧 技术方法**

采用卷积神经网络（残差阶段+顶部融合）做特征提取，随后SGCA模块（投影、门控、相关矩阵、矩阵平方根归一化）生成2080维描述子；训练时使用AdamW、随机裁剪、三元组损失；实现基于PyTorch

**📊 数据集**

主要使用MulRan数据集（Navtech CIR204-H FMCW雷达，400°×3,360格极坐标图）进行训练与评估；并在未见的HeRCULES旋转雷达场景上进行迁移性测试

**📈 对比分析**

与Ring Key、ScanContext、VGG‑16+NetVLAD、RadarLoc等方法在MulRan的Recall@1（5 m/10 m/3 m）进行比较，SGCA‑Net在所有环境下均取得最高Recall（如5 m 91.3%/10 m 98.1%，比RadarLoc提升约3–4%），在HeRCULES上平均Recall（AR@1）提升至0.958，优于RadarLoc的0.940

**⚠️ 局限性**

主要局限：①SGCA的相关矩阵计算与矩阵平方根归一化使得计算量与内存需求较大；②当前仅针对密集360°旋转雷达，未验证对其他雷达配置（如4D雷达）的适用性；③未针对动态障碍物或极端天气进行显式建模，需进一步研究不确定性加权与自适应机制

---

## 252. Overlapping Visual Grouping Without Semantic Priors

**arXiv ID:** 2609.27423 | [PDF](https://arxiv.org/pdf/2609.27423v1)

**作者:** Teemu Saukkio `[一作]` (University of Turku), Juha Plosila `[通讯]` (University of Turku)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种名为Domain Parent Grouping (DPG) 的非排他性视觉分组方法，能够在图像中识别出多个可重叠的感知单元，而无需预先定义语义标签。

**💡 创新点**

创新点在于：①以传感器测量关系为基础，采用多域（亮度、向量色、色彩一致性）独立建模并通过跨域重叠支持来生成父子关系；②实现了对选定组内容的再处理，可在不丢失原始父组的前提下提升观察分辨率；③提供了可用于后续学习与任务解读的中间层视觉表示。

**🔧 技术方法**

核心技术包括：局部均值/方差自适应亮度处理、基于相对差异的颜色分类、全局色彩一致性补偿、离散化分级、基于连通分量的空间分组、跨域重叠判定以及基于父子关系的重组与再处理。

**📊 数据集**

使用的评估数据集为BSDS500（共500张自然图像），并在实验中采用了两幅不同光照条件下的GoPro原始图像做定性演示。

**📈 对比分析**

与Felzenszwalb、Quickshift和Watershed‑RAG三种经典无监督分割方法进行对比：DPG在区域覆盖率与低阈值回召率上与Felzenszwalb相当，但在严格阈值回召率略低；DPG在边界回召率最高，但边界精度最低；总体运行时间最短（约0.11 s/图），组数介于Felzenszwalb与Quickshift之间。

**⚠️ 局限性**

局限性：①仅针对静态RGB图像，未考虑多帧/时序关联；②参数调优由单一评估者完成，缺乏多观点评估；③BSDS500标注不覆盖所有可能的重叠或照明结构，导致无法完全验证DPG产生的所有感知单元；④再处理的自动选择机制尚未实现。

---

## 253. EviStreams: Human-in-the-Loop AI Data Extraction for Systematic Reviews in Medicine

**arXiv ID:** 2609.27418 | [PDF](https://arxiv.org/pdf/2609.27418v1)

**作者:** Sai Karthik Kosuri `[一作]` (University of Pennsylvania), Chris Callison-Burch `[通讯]` (University of Pennsylvania)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一个名为Evidence Streams的无代码平台，用于在系统综述中实现人机协作的数据提取，支持程序审查、字段规范校准和双盲审核。

**💡 创新点**

创新点在于将人工审查嵌入到三阶段工作流：程序层面预先批准抽取计划、字段层面通过示例和规则动态校准、以及在最终提取结果中实施双盲审核并记录完整审计轨迹；并证明字段规范是提升抽取质量的关键驱动因素。

**🔧 技术方法**

采用了大型语言模型（Claude Sonnet 4.6、GPT‑5.5、Gemini 3.1 Pro）、DSPy编译器、LangGraph工作流、Datalab文本转换、Next.js/FastAPI/Redis等技术堆栈，同时利用提示工程、锚点列表格拆分等方法。

**📊 数据集**

使用了四个临床综述语料库：口腔癌诊断辅助（43项研究）、抗菌预防（10项）、牙周炎（29项）和布洛芬疗效（43项），全部来自已发表的系统综述。

**📈 对比分析**

对比方法是固定程序和字段规范，分别在三种前沿模型上评估微/宏F1；结果显示模型差异极小（<0.05），字段规范提升F1平均0.05，整体micro‑F1在0.74–0.90之间，显示模型不决定精度，规范是主要杠杆。

**⚠️ 局限性**

限制包括：缺乏对参考数据的标注一致性评估；仅对已校准的规范进行评估，未知未校准情形；评估集规模有限（125项研究），以及自由文本评审器与某些模型同属一家提供商可能导致评估偏差。

---

## 254. Emergi-PersonaOS: A Persona Agent Operating System for Situational Adaptation and Controllable Evolution

**arXiv ID:** 2609.27417 | [PDF](https://arxiv.org/pdf/2609.27417v1)

**作者:** Haoluan Fu `[一作]` (Emergi Lab of Qianzhen Digital Tech), Yuyu Yin `[通讯]` (Hangzhou Dianzi University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Emergi-PersonaOS，一个面向长期人机共生的基于三层人格结构的个人体操作系统，完成人格构建、情境适配与长期更新的完整生命周期管理。

**💡 创新点**

将人格特质、典型适应和叙事身份三层心理学框架统一映射到持久信念与即时状态，并设计基于证据的可控演化机制，确保人格变更可审计、可拒绝。

**🔧 技术方法**

利用大语言模型进行生成与推理，结合三层状态推理、检索记忆、行为监测、反思与演化模块，并引入心理学理论（Whole Trait Theory、TESSERA、Realistic Accuracy Model等）。

**📊 数据集**

使用《Young Sheldon》电视剧141集的对话与情节记录，通过提炼生成人物叙事、关键事件、对话、信息缺口及出处，构建跨季记忆库。

**📈 对比分析**

与对话仅、静态人格、L1层、无跨季记忆、无季终积累等对照组比较，完整系统在人格忠实度、情境连贯性、状态-行为一致性等维度均显著优于对照组，平均得分4.48/5。

**⚠️ 局限性**

评估仅基于同一虚构角色、单一LLM模型、季终评审而非事件级演化、缺乏真实用户数据及人类评估，模型依赖性与证据判定仍需进一步验证。

---

## 255. AraGenre 2026: A Hierarchical Definition-Guided Arabic Genre Classification Shared Task

**arXiv ID:** 2609.27387 | [PDF](https://arxiv.org/pdf/2609.27387v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 256. Active Learning for Biodiversity Monitoring: From Label Efficiency to Reliable Ecological Inference

**arXiv ID:** 2609.27409 | [PDF](https://arxiv.org/pdf/2609.27409v1)

**作者:** Ben McEwen `[一作]` (University of Amsterdam), Dan Stowell `[通讯]` (Leiden University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文对生物多样性监测领域中主动学习（AL）的研究进行全面综述，梳理了AL循环的各个环节、专家预算分配问题，并基于对164篇文献的系统映射提出了改进路线图。

**💡 创新点**

创新点在于：①提供了针对生态监测的预算意识AL教程；②系统总结并可视化了AL在声学与图像监测中的应用现状；③指出训练、验证与生态推断共用专家预算的缺口，并给出具体的实验和方法改进建议。

**🔧 技术方法**

技术方面涉及深度学习模型（如BirdNET、MegaDetector、BioCLIP等）与多种查询策略（不确定性采样、几何采样、混合策略、成本感知与生态价值导向策略），以及主动测试、重要性加权校正等验证与校准方法。

**📊 数据集**

主要使用的公开基准数据集包括BirdSet、AudioSet、Snapshot Serengeti、AnuraSet等，同时收录了一些现场部署案例（如TABMON网络、海洋哺乳动物呼叫监测等），覆盖鸟类、哺乳动物、海洋哺乳动物等多种物种。

**📈 对比分析**

比较方法主要通过学习曲线评估AL相对于随机抽样在标签节省率和性能提升的差异，平均标签节省率约为64%，但大多数实验基于模拟或预标注数据，缺乏真实验证和跨方法统一基准；且很少对每类性能、校准或时空泛化进行细致评估。

**⚠️ 局限性**

局限性包括：①研究大多停留在模拟/预标注基准，未充分考虑验证与生态推断的预算需求；②主动学习导致采样偏差，验证集不足，缺乏统一的随机基线比较；③对昆虫、鱼类、多模态监测的覆盖不足；④未将专家时间成本与实际标签数量对应，导致成本评估不完整。

---

## 257. What Looks Like a Capability Limit in Vision-Language Models Is a Readout Limit

**arXiv ID:** 2609.27408 | [PDF](https://arxiv.org/pdf/2609.27408v1)

**作者:** Alfredo F. Frontera Del Valle `[一作]` `[通讯]` (Columbia University), Alfredo F. Frontera Del Valle (Columbia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多选视觉语言基准中探究答案书写约定（如坐标、颜色词、数值标签）对模型评分的影响，并提出一套“约定交换”与“误配名词”评估方法；

**💡 创新点**

证明答案书写约定对模型得分、排名具有显著且可逆的影响，并指出不同模型对同一约定的可读性差异；

**🔧 技术方法**

采用答案词表校准、首词读取、误配名词测试、可视/缓存/交叉提示、量化实验及统计检验等技术；

**📊 数据集**

使用COCO 2017 val图像进行定位任务，利用合成颜色方块和CIELAB色轮进行颜色任务；

**📈 对比分析**

对同一批图像/合成样本分别采用多种答案约定进行对比，发现坐标约定可导致近50点分数下降，且不同模型间出现排名倒置；

**⚠️ 局限性**

实验仅覆盖COCO（类别偏向人物）和粗粒度网格，未对自由形式定位或更细粒度任务进行验证；使用4-bit精度、贪婪解码、有限数量的前沿模型，并未测试采样或完整精度。

---

## 258. Forget who you Forgot: Speaker Unlearning to Prevent Re-Identification in Zero-Shot Text-to-Speech

**arXiv ID:** 2609.27399 | [PDF](https://arxiv.org/pdf/2609.27399v1)

**作者:** Hyoeun Kim `[一作]`, Kyuhong Shim `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于门控网络和层级激活向量的零样本 TTS 说话人身份消忘框架 GUARD。

**💡 创新点**

创新点在于将说话人身份消忘目标设为人口级冒充相似度，并使用组相对奖励优化学习激活向量，既抑制再识别又保持语音质量。

**🔧 技术方法**

采用门控网络 GateNet、层级激活向量 steering、组相对奖励优化（GRPO）以及冻结的 CosyVoice2 后端。

**📊 数据集**

使用 LibriTTS clean-460 数据集中的 50 位忘记说话人以及 CosyVoice2、F5‑TTS、FireRedTTS 等后端。

**📈 对比分析**

与 SGU、TGU、TruS 等对比，GUARD 在 150 名说话人图库中的再识别率从 73.5% 降至 0.5%，SIM 由 0.541 降至 0.103，同时保持 WER、UTMOS 与原始模型相近。

**⚠️ 局限性**

局限在于未实现说话人自适应激活向量，且过低的 SIM 可能导致保护输出更易被区分为冒充录音。

---

## 259. When Entanglement Lower-Bounds Disparity: Auditing and Repairing Demographic Fairness in Audio Understanding Models

**arXiv ID:** 2609.27382 | [PDF](https://arxiv.org/pdf/2609.27382v1)

**作者:** Kian Shamsaie `[一作]` (People Make Things), Iman Modarressi `[通讯]` (People Make Things)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对语音理解模型进行公平性审计并修复：构建可控的 TRIAD 审计网格，量化声音、语义、表达三轴的互相泄漏，并提出 ORCA 适配器在不改动原模型权重的前提下，降低泄漏并提升跨群体性能。

**💡 创新点**

创新点在于：① 提出基于主成分角度的解耦泄漏度量，并证明其对平均及极端群体差距的下界；② 将该理论转化为可训练的正交残差对比适配器（ORCA），实现对三轴信息的同时保留与分离；③ 通过单一合成器实现的完全可控审计网格，消除了内容与声学混杂，便于公平性因果分析。

**🔧 技术方法**

技术方法包括：线性探针与判别分析、主成分角度泄漏计算、对比学习（多头监督对比损失）、正交惩罚（子空间正交化）、残差适配器、层级批次平衡采样、闭源模型的配对差异与 permutation 检验。

**📊 数据集**

使用的数据集包括：① TRIAD 合成网格（120文本×24 语音档案×10 表达风格，共 28,800 条录音）；② 60 小时真实语音池（CANDOR、SSSD、Seamless Interaction、OTO Speech、AMI 等）；③ 公共基准（Common Voice、LibriSpeech、CREMA‑D、MSP‑Podcast、IEMOCAP 等）用于真实公平性评估。

**📈 对比分析**

与十个开源编码器（Whisper、Parakeet、Voxtral、Qwen2‑Audio、wav2vec、HuBERT、WavLM、emotion2vec、Mimi、WavTokenizer）以及两款闭源 S2S 系统（gpt‑realtime‑2、Gemini‑3.1‑flash‑live）比较，发现所有基线模型均存在显著的声音-语义泄漏与群体差距；ORCA 在 WavLM 与 Whisper 上将平均泄漏率从 0.29/0.51 降至 0.08/0.11，差距平均缩小 50% 以上，实际任务中的词错误率与情感识别差距亦随之显著下降。

**⚠️ 局限性**

主要局限包括：① 仅使用单一 TTS 生成审计网格，可能导致合成器特定的声学偏差；② 研究聚焦于线性探针，未覆盖核或非线性方法；③ 对多语言与更大规模真实数据的迁移性尚未充分验证；④ ORCA 的正交惩罚需要手工调参，对不同模型结构的通用性待进一步探索。

---

## 260. MORSE: Multi-Context Ordering via Reverse Scoring for Evidence-Preserving Compression

**arXiv ID:** 2609.27380 | [PDF](https://arxiv.org/pdf/2609.27380v1)

**作者:** Ke Wan `[一作]` (University of Virginia), Chen Chen `[通讯]` (University of Virginia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究检索得到的多上下文在基于似然的顺序压缩中的排序敏感性，发现信息预占导致证据保留随上下文顺序变化，并提出 MORSE 方法通过逆查询证据先导与压缩感知全局搜索来优化上下文排序。

**💡 创新点**

创新点在于揭示信息预占机制并以逆查询证据原则构造证据优先先导，同时在压缩后重新评估候选排序，选择保留最多查询证据的排列，从而显著提升证据保留率。

**🔧 技术方法**

采用逆查询证据评分、语言模型似然分数、序列化顺序分数、压缩感知候选搜索及基于逆查询证据的选择目标 J_B 等技术。

**📊 数据集**

实验使用 HotpotQA、2WikiMultiHopQA、True-20 以及 MuSiQue 等多跳 QA 数据集，并以 Qwen2.5-0.5B 和 OLMo-2-1B 等语言模型作为压缩评分器。

**📈 对比分析**

与独立压缩、五个静态排序（原始、随机、长度、前向、逆序）以及计算匹配的 RandomSearch-5 进行对比。MORSE-5 在所有压缩比例下均显著提升 Supporting‑Fact Recall，并在多数情况下带来回答准确率的提升，搜索预算 K=5 在质量与延迟之间取得良好平衡。

**⚠️ 局限性**

局限性包括：在轻度压缩或数据集漂移时提升不明显；部分下游 QA 指标提升不稳定；方法高度依赖语言模型评分，对 scorer/数据集变化敏感；搜索预算有限时收益有限；未进一步解决多样性与冗余处理的更广义问题。

---

## 261. Understanding Human Perception of Representation in Citizens' Assemblies: An Empirical Study

**arXiv ID:** 2609.27368 | [PDF](https://arxiv.org/pdf/2609.27368v1)

**作者:** Yusuf Hakan Kalayci `[一作]` (University of Southern California), Evi Micha `[通讯]` (University of Southern California)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过两轮英国公民议会的随机共轭实验，系统评估人口属性、政治取向和议题特定属性（如气候关注）对公民对代表性判断的影响，并检验遗漏属性是否能通过相关属性间接体现。

**💡 创新点**

创新点在于首次将人口属性、政治取向和议题特定属性的相对重要性进行定量比较，发现政治与议题属性往往比人口属性更能决定代表性，并证明单靠相关人口属性无法代替关键议题属性；同时提出并对比多种预测模型（加权L1距离、条件Bradley‑Terry）揭示个体化偏好结构。

**🔧 技术方法**

研究采用随机共轭实验、平均边际成分效应（AMCE）分析、LEXIMIN分配、TV误差度量以及机器学习预测模型（加权L1、条件BT）等技术。

**📊 数据集**

使用两份英国全国性在线问卷样本，Survey1（1151人）和Survey2（576人），共约11,5千个二选一比较决策，涵盖七个人口属性、政治取向与气候关注等特征。

**📈 对比分析**

与简单基线（匹配计数、最重要线索、AMCE树）相比，条件BT模型在三种任务（一般、气候关注单独、双线索）中实现最高测试准确率（约73–78%），加权L1次之（约72–76%），而基线模型准确率显著较低，证明多属性协同预测显著提升。

**⚠️ 局限性**

局限性包括：仅评估个体级别的代表性判断，未检验对完整议会组合或合法性评估的影响；实验样本仅来自英国，难以推广；属性选择受实验设计限制；预测模型虽有解释力但残差仍较大，未涵盖所有议程维度，且代理属性假设受样本特定关联性影响。

---

## 262. Seal, Then Sample: Sampled Layerwise Proofs for Verifiable LLM Inference from GPT-2 to 70B

**arXiv ID:** 2609.27367 | [PDF](https://arxiv.org/pdf/2609.27367v1)

**作者:** Youki Lim `[一作]` (TrueOpen), Sam Yong `[通讯]` (TrueOpen)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 Sampled Layerwise Proofs (SLP) 协议与原型，能够对 GPT‑2、TinyLlama‑1.1B 以及 Llama‑2‑70B 等大规模语言模型的推理过程进行可验证的抽样证明，并在多种硬件平台上进行了实验评估。

**💡 创新点**

主要创新点包括：① 将抽样证明与完整边界承诺分离，支持可配置的审计强度；② 通过块状对角 causal mask 实现批量打包推理，大幅降低证明成本；③ 引入磁盘存储整数权重并流式注册，使 70B 模型能在 2 TB 主机上完成全流程证明；④ 对固定点模型的量化误差进行诊断并通过 LLM‑aware 观察器恢复精度。

**🔧 技术方法**

技术上采用了 Deep‑Prove 的分块 SumCheck 与 HyperKZG 多项式承诺；整数权重使用 64 bit 表示并通过磁盘映射与批量承诺实现；批量推理通过块状对角 causal mask 实现；挑战生成结合 Fiat‑Shamir 与外部随机 beacon，保证抽样的不可预测性；验证流程包含边界一致性、采样抽样、anchor 绑定等六项检查。

**📊 数据集**

实验数据集主要包括：TinyLlama‑1.1B 使用固定提示 “The sky is” 进行推理；Llama‑2‑70B 使用随机短上下文；在 WikiText‑2 测试集上做 512‑token 窗口对比，评估 argmax 一致性与 perplexity；GPU 端对 TinyLlama 进行 8、16、32、128、512‑token 等上下文长度的推理。

**📈 对比分析**

通过在相同推理轨迹上进行全覆盖与抽样覆盖的对比，评估证明时间与大小的关系：在 TinyLlama 上抽样 5/47 的证明仅占 22% 的运行时间和 6.8% 的证明大小；批量 12 个请求的证明仅为单请求的 15% 运行时间；70B 模型单次证明耗时 1259 s、证明大小 4.34 MiB，验证 46.3 s；GPU 推理在不同上下文长度下分别耗时 1–26 s。

**⚠️ 局限性**

主要局限包括：① 证明仅覆盖抽样块，无法保证完整推理的正确性；② 在无外部挑战的抽样模式下易受离线搜索攻击；③ 整数权重实现导致内存占用高且未完全释放；④ 未完成 70B 的全覆盖证明与精度评估；⑤ 安全模型依赖外部 beacon 与不可重复挑战，未实现完整零知识证明；⑥ 系统未处理多 Manifest、重放攻击等部署细节，需要进一步完善。

---

## 263. Constraint-Driven Context Engineering: Designing Domain Interfaces for AI Systems

**arXiv ID:** 2609.27354 | [PDF](https://arxiv.org/pdf/2609.27354v1)

**作者:** Xiwei Xu `[一作]` (CSIRO), Liming Zhu `[通讯]` (CSIRO)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究并提出了约束驱动上下文工程（CDCE）方法，利用三种不同领域（教育评估、医疗决策支持、金融违约预测）的案例验证其在构建 AI 系统域界面方面的有效性。

**💡 创新点**

创新点在于将技术、法规、机构和规范约束作为首要设计驱动，将约束系统化识别、表征并转化为可供 AI 直接使用的上下文，突破了传统仅关注知识检索的上下文工程范式。

**🔧 技术方法**

技术手段包括领域驱动设计（DDD）、属性驱动设计（ADD）、向量检索、规则引擎、结构化数据库以及人类审核与反馈循环，实现约束的识别、结构化与执行。

**📊 数据集**

数据集涵盖：教育案例的澳洲课程与评估指南及教师评分数据；医疗案例的患者电子健康记录、临床指南与 MDT 讨论记录；金融案例的 128 个地方政府 646 维度的财务指标（年标记特征）与审计报告。

**📈 对比分析**

通过多案例对比评估方法，利用专家评审与自动验证指标对比约束前后 AI 输出的准确性、可解释性与合规性；结果显示约束驱动上下文显著提升了 AI 结果的领域适配性与可信度。

**⚠️ 局限性**

局限性包括缺乏大规模量化性能评估、对人工审查与约束演化的持续依赖，以及方法在不同域之间迁移性的进一步验证仍需探索。

---

## 264. Geometry-Conditioned Visual Place Recognition in Natural Environments

**arXiv ID:** 2609.27370 | [PDF](https://arxiv.org/pdf/2609.27370v1)

**作者:** Walter Nedov `[一作]` (CSIRO), Peyman Moghadam `[通讯]` (CSIRO)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在不需要深度传感器的前提下，提出 Depth‑Aware Distillation 框架，将 Geometric Foundation Model（Depth Anything V2）推断的深度信息投影到 Vision Foundation Model（DINOv2）令牌空间，并通过通道级几何门控融合外观与几何特征，从而提升自然环境下的视觉地标识别性能。

**💡 创新点**

①将几何信息直接投影到令牌空间并进行通道级门控，避免传统多模态融合导致的特征空间冲突；②采用两阶段教师引导训练——先通过对齐损失让几何条件下的描述符与预训练外观空间对齐，再细化为检索任务，兼顾结构信息与辨识度。

**🔧 技术方法**

使用 Vision Foundation Model（DINOv2）作为基准特征提取器，Geometric Foundation Model（Depth Anything V2）估计深度；通道门控 Adaptive Fusion；两阶段训练（Descriptor Distillation + Task Refinement）；多相似性检索损失和对齐损失；SALAD 聚合头。

**📊 数据集**

WildCross 基准数据集（两片自然森林，476k RGB 帧，四条轨迹），并对 Depth Anything V2 进行森林域适配。

**📈 对比分析**

与匹配的 SALAD、NetVLAD、MixVPR 进行对比；inter‑sequence Recall@1 由 61.41% 提升至 66.37%，Recall@5 由 65.86% 提升至 72.49%；intra‑sequence Recall@1 从 61.76% 提升至 65.34%；显著改善逆向行走和长期视觉变化场景。

**⚠️ 局限性**

对 GFM 的适配仍需依赖 LiDAR 监督；两阶段训练流程较为复杂；通道门控学习可能不够充分；在极端光照或遮挡条件下仍存在性能瓶颈。

---

## 265. Counterfactual Constraint-Conditioned On-Policy Distillation for Multi-Constraint Instruction Following

**arXiv ID:** 2609.27421 | [PDF](https://arxiv.org/pdf/2609.27421v1)

**作者:** Yanzhao Zheng `[一作]` (Alibaba Group), Ruohui Huang `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于教师留一式对比的多约束指令跟随训练方法（CCOPD），通过在相同学生生成的轨迹上多次让冻结教师对不同约束组合进行 forced‑scoring，提取每个约束的 token 级别奖励形状，并将其与原始 OPD 奖励相加进行 PPO 更新。

**💡 创新点**

创新点在于用留一式对比逆转教师的条件，直接从同一教师中得到每个约束的具体贡献，避免传统外部评估器或全上下文混合导致的约束稀释问题，并实现了无需外部 verifier 的无监督多约束训练。

**🔧 技术方法**

使用 sampled‑token OPD 核心（ratio‑clipped PPO），对教师进行多次 forced‑scoring，构造离散的 log‑likelihood 差值作为奖励形状；并采用 MOebius 分解与对奖励进行对称裁剪以稳定训练。

**📊 数据集**

在 HIR‑16K 指令跟随语料上预调教师，并在七个多约束基准（IFEval、IFBench、MulDimIF、ComplexBench、InfoBench、FollowBench、CFBench）上进行评估。

**📈 对比分析**

与 SFT、GRPO‑RL、标准 OPD、OPD‑augmentation 等方法对比；在 Qwen2.5（1.5B/7B）和 Qwen3（4B/4B）两对模型上，CCOPD 在全部七个基准的平均分上均超过所有基线（Qwen2.5: 51.4%，Qwen3: 64.3%），并在 MulDimIF 上实现学生超教师（+2.8pp/ +6.3pp），梯度步数大幅减少。

**⚠️ 局限性**

需要可访问白盒教师进行多次 forced‑scoring，导致每步计算量提升 |C|+1 次；仅在 Qwen 系列上验证，未测试其他模型家族、非单轮或系统提示驱动的情景；未评估在不同 OPD 目标下的泛化；自训练对比仅在两种规模上表现不稳定，需进一步探究教师规模与 LOO 信号效用的关系。

---

## 266. S2A:Semantic-to-Spatial Alignment for Alignment-Free RGB-T Salient Object Detection

**arXiv ID:** 2609.27413 | [PDF](https://arxiv.org/pdf/2609.27413v1)

**作者:** Qiangqiang Zhou `[一作]`, Jiawei Xu `[通讯]` (Jiangxi Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种无对齐 RGB‑T 视觉显著性检测方法，采用语义‑到‑空间的逐步对齐策略，先利用全局语义引导提升单模特征，再通过跨模态通道注意力实现语义级交互，最后通过空间可变采样恢复局部空间对应关系，从而有效抑制空间错位导致的特征污染。

**💡 创新点**

创新点包括：①全局引导的层次融合模块 (GGHF)，通过高层语义抑制背景噪声；②无对齐跨模态通道注意力模块 (AFCA)，实现宏观通道级语义交互，避免像素级匹配误差；③空间可变交叉注意力模块 (SDCA)，动态预测采样偏移并恢复局部空间对应；④将上述三步按“语义→跨模态语义→空间恢复”的顺序排列，构建了从语义到空间的递进对齐框架。

**🔧 技术方法**

技术实现上使用 Swin‑B 作为双分支编码器，GGHF 通过 3×3 卷积和特征聚合实现层次融合；AFCA 采用 depth‑wise dilated 卷积生成 Q、K，随后进行全局通道注意力；SDCA 通过 1×1 卷积预测偏移后使用 bilinear 网格采样，形成可变交叉注意力；解码器采用轻量级 top‑down 结构与交叉熵、平滑损失、soft Dice 损失联合训练。

**📊 数据集**

实验数据集包括：UVT‑20K、UVT‑2000、un‑VT‑5000、un‑VT‑1000、un‑VT‑821，涵盖完全无对齐与弱对齐两类场景；所有图像统一尺寸 384×384 进行训练与评估。

**📈 对比分析**

与 17 种主流方法（Swin‑B、ResNet、HRFormer、Res2Net 等）在 S_m、E_m、F_m 三个指标上进行比较。实验表明，在大多数数据集上，本文方法在所有指标上均优于最新的 TPS‑SCL、PCNet 等方法，特别是在 UVT‑2000 上虽然 E_m 与 PCNet 相当，但 S_m 与 F_m 仍领先，整体性能显著提升。

**⚠️ 局限性**

局限性：①方法对 Swin‑B 等大模型的依赖导致计算量和显存占用较高；②在极端高误差的空间错位或快速运动场景下，空间可变采样的恢复效果仍可能不足；③目前仅在 RGB‑T 对齐不足的场景评估，尚未验证对其他多模态任务的通用性。

---

## 267. When Parallel Drafter Meets Parallel Speculative Decoding

**arXiv ID:** 2609.27396 | [PDF](https://arxiv.org/pdf/2609.27396v1)

**作者:** Fuliang Liu `[一作]` (State Key Laboratory of Novel Software Technology, Nanjing University), Chen Tian `[通讯]` (State Key Laboratory of Novel Software Technology, Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的并行投机解码框架（即PSDD），能够在每一步中完全隐藏推断模型的前向计算，从而消除传统投机解码中因误判“bonus token”而导致的回退。

**💡 创新点**

核心创新是将多锚点无特征的扩散背骨（M‑DFlash）与轻量自回归头分离，先在验证过程中并行预计算所有可能的接受边界的logits，然后仅在验证完成后用自回归头完成最终token生成，实现了结果无关的背骨与验证重叠。

**🔧 技术方法**

技术上使用了DSpark‑style扩散语言模型的多锚点预训练架构、无参数微调仅针对背骨、轻量自回归头、以及异构GPU调度（如A10与H800的拆分或单GPU协同）。

**📊 数据集**

在Qwen3‑8B与Qwen3‑14B模型上，使用了七个基准：Math（GSM8K、MATH‑500）、Coding（HumanEval、MBPP、CodeAlpaca）和Chat（MT‑Bench、Alpaca）。

**📈 对比分析**

与传统串行投机解码（DSpark）以及并行投机解码（PEARL、SSD）相比，PSDD在所有模型-数据集组合上均取得最高速度提升，平均对AR解码提升约3.2×/3.5×，对DSpark提升约11%/5%，在批量1至16的范围内始终领先，并在资源受限或单GPU部署时保持约90%–98%的加速。

**⚠️ 局限性**

限制主要在于：（1）接受长度仍低于最强串行投机解码，原因是背骨需在无特征锚点的条件下预计算；（2）在大批量或计算受限情况下，轻量头的计算可能成为瓶颈；（3）需要额外的多锚点微调步骤，虽不增加可学习参数，但仍需专门训练。

---

## 268. Automated Extraction of Records of Processing Activities (RoPA) Using Hybrid RAG and Locally Deployed Large Language Models

**arXiv ID:** 2609.27359 | [PDF](https://arxiv.org/pdf/2609.27359v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 269. Psychoacoustically Aligned Latent Smoothing for Adversarial Robustness of Full-Duplex Speech-to-Speech Dialogue Models

**arXiv ID:** 2609.27378 | [PDF](https://arxiv.org/pdf/2609.27378v1)

**作者:** Kian Shamsaie `[一作]` (People Make Things), Iman Modarressi `[通讯]` (People Make Things)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对全双工语音转语音对话模型，提出了心理声学对齐潜在平滑（PALS）防御方法，利用噪声注入到RVB编码器的潜在空间，并结合与掩蔽阈值匹配的输入空间噪声与KL一致性正则，提升模型对不可察觉攻击的鲁棒性。

**💡 创新点**

创新点在于：①将心理声学掩蔽阈值与潜在空间协方差对齐，设计了各码本自适应的各向异性高斯噪声；②在输入层使用最大熵的掩蔽匹配噪声，形成最少假设的攻击模拟；③通过单一标量参数即可控制双重噪声幅度，保持无推理时开销；④提供了Monte‑Carlo 随机平滑的可证明椭圆半径证书。

**🔧 技术方法**

核心技术包括：残差向量量化（RVQ）编码器，Mimi 风格音频编解码器，7B 变压器对话模型，心理声学掩蔽模型（MPEG‑1 model 1），随机平滑与TRADES式 KL 一致性训练，以及基于Mahalanobis 距离的椭圆半径证书。

**📊 数据集**

使用约 3,118 小时双声道英语对话语料：otoSpeech（141h），SSSD（727h），CANDOR（850h），Seamless Interaction（1,300h），AMI（100h）等。

**📈 对比分析**

与未防御、Gaussian 数据增强、输入空间随机平滑、Demucs 前端以及 PGD‑AT 对抗训练等基线比较，PALS‑Train 在 0 dB 预算下将 hijack、mute、jailbreak 的成功率分别从 91.7%、88.4%、76.2% 降至 8.3%、11.2%、9.1%，质量损失仅 2.3%（UTMOS 3.82 vs 3.91）。PALS‑Certify 在潜在空间 ℓ₂ 半径 0.5 时，47% 的判断被证明保持不变，证书半径上限为 0.616（约 Δ≈‑26 dB）。

**⚠️ 局限性**

局限性包括：①证书仅在潜在空间给出，输入空间的安全保证仅是通过经验估计的 Lipschitz 上界得到的启发式转换；②目前仅在英语双声道对话语料上验证；③对更高维或更大模型的可扩展性尚未评估；④在极端房间混响或嘈杂环境下仍存在一定攻击成功率。

---

## 270. Neither Silence nor Overlap Is Failure: Intent-Conditioned Evaluation of Turn-Taking in Full-Duplex Spoken Dialogue Models

**arXiv ID:** 2609.27372 | [PDF](https://arxiv.org/pdf/2609.27372v1)

**作者:** Kian Shamsaie `[一作]` (People Make Things), Iman Modarressi `[通讯]` (People Make Things)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 TACT 评测基准，用意图感知的连续评估替代传统的二元窗口方法，评估全双工语音对话模型的轮次交替决策。

**💡 创新点**

创新点包括：① 将意图条件化的严格正确连续排序概率评分（twCRPS）与不对称时序核结合；② 为每个说话人构建记忆配置文件，并通过多源标注与校准的 LLM 判断融合得到意图后验；③ 将评测从二元窗口移至可解释、可校准的连续时间度量。

**🔧 技术方法**

使用技术包括：连续排序概率评分（CRPS）与阈值加权（twCRPS）；ex‑Gaussian 与偏移对数正态时序核；温度缩放的 LLM 判断与多源后验融合；VAP 风格的语音完成概率估计；严格正确性与一致性证明；JSD 与 Spearman 相关性评估。

**📊 数据集**

数据集覆盖 9,728 条录音剧集（73.2 小时），来源于五个公开双人对话语料库（CANDOR、SSSD、otoSpeech、Seamless Interaction、AMI），并补充合成 TTS 探针以覆盖稀缺条件。

**📈 对比分析**

与 11 种前沿全双工对话模型进行对比；在 TACT 复合分数上最佳模型为 0.47，远低于人类 0.86；与传统二元窗口指标相比，模型排名显著重排（Spearman 0.55），并且与人类评价的相关性达到 0.81，显著高于二元指标的 0.46；内存消融实验表明现有模型对说话人记忆几乎不敏感。

**⚠️ 局限性**

局限性包括：意图分类为离散六类，标注一致性仅为 α≈0.73；时序核参数在英语数据上拟合，跨语言推广需重调；LLM 判断可能与人类标注共享错误；对齐评估依赖 VAP 完成概率，若该估计失效将影响评估精度。

---

## 271. Spectral-NFP: Certified Low-Rank Curvature Majorization for Accelerating WMMSE

**arXiv ID:** 2609.27369 | [PDF](https://arxiv.org/pdf/2609.27369v1)

**作者:** Jianhang Zhu `[一作]`, Kaiming Shen `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Spectral-NFP算法，通过保留主特征值构建可调秩的曲率上界，解决多基站MIMO权重和速率最大化的计算复杂度问题。

**💡 创新点**

创新点在于将低秩+标识矩阵逼近与非同质FP结合，提供理论增益比例下界和大系统秩选择规则，显著提升逼近精度。

**🔧 技术方法**

采用非同质FP、WMMSE框架、低秩EVD、Nesterov加速、Wishart分布与Marchenko–Pastur理论等技术。

**📊 数据集**

使用七区六分区、六用户/基站的几何模型，结合路径损耗、阴影、随机小尺度，100个独立通道样本进行仿真。

**📈 对比分析**

与NFP、完整WMMSE及WMMSE-EVD比较，实验显示在保留≤10%维度时Spectral-NFP获得>99% WMMSE WSR，更新时间仅为WMMSE的20%以内。

**⚠️ 局限性**

限制在于需预先或自适应选择秩，理论分析基于理想Wishart假设，极大规模或极端负载下的谱分布与计算开销仍需进一步研究。

---

## 272. SAGEGAN: Style-Based Anomaly Detection with Gaussian Embeddings using Generative Adversarial Networks

**arXiv ID:** 2609.27357 | [PDF](https://arxiv.org/pdf/2609.27357v1)

**作者:** Thesath Wijayasiri `[一作]` (Singapore Technologies Engineering), Vrizlynn L. L. Thing `[通讯]` (Singapore Technologies Engineering)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于Benign-only的恶意软件异常检测框架SAGEGAN，利用Hilbert映射的三通道图像表示和StyleGAN风格的对抗重建模型，实现对PE文件的异常评分。

**💡 创新点**

创新在于引入层级化Gaussian风格潜变量并与StyleGAN生成器对齐，结合瞬时先验匹配和潜在一致性约束，同时通过确定性编码实现可重复的层级化可解释性分析。

**🔧 技术方法**

使用Hilbert曲线映射、字节二元组惊奇度、熵偏差的三通道图像特征，StyleGAN2风格注入的七层生成器，Gaussian风格先验、moment匹配、latent一致性以及基于梯度和PCA的可解释性指标。

**📊 数据集**

在自采的214类恶意PE（10820个）和20000个Benign样本上训练验证，外部迁移评估于DIKE、Microsoft BIG 2015和Lester恶意子集。

**📈 对比分析**

与VanillaGAN、DCGAN、WGAN‑GP、BigGAN、Hierarchical GAN、GANomaly、f‑AnoGAN、ECBiGAN等生成式基线以及之前的EBBiGAN和OCSVM对照，SAGEGAN在自采测试集上AUC 89.76%、平衡准确率88.19%，在DIKE、Microsoft BIG、Lester的零样本迁移中AUC均超过92%、平衡准确率≥84%。

**⚠️ 局限性**

仅处理静态PE文件，对打包/混淆、动态行为和对抗攻击鲁棒性未评估；异常分数非概率，且迁移结果受数据分布差异影响；实验未覆盖多种可执行格式和更大规模数据。

---

## 273. CoPRE: Improving Sensitivity in Proprioceptive Contact Detection for Low-Cost Robot Arms

**arXiv ID:** 2609.27381 | [PDF](https://arxiv.org/pdf/2609.27381v1)

**作者:** Yuxiao Zhu `[一作]` (Duke University), Xianyi Cheng `[通讯]` (Duke University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在低成本机器人臂上，仅利用关节位置、速度和电流推算的关节力估计，提出了 CoPRE 方法实现对弱接触的感知并用于自适应操控。

**💡 创新点**

创新点在于：①使用状态排除策略避免预测中出现接触信息，保持残差对比度；②利用噪声加权的雅可比矩阵聚合残差生成接触分数；③仅依赖无接触训练数据，无需力传感器或标签。

**🔧 技术方法**

技术包括：Transformer 预测器对未来 H 步关节状态的直接多步预测；残差偏置校正与噪声加权的最小二乘雅可比拟合；阈值确认规则（连续 K 次超过阈值）。

**📊 数据集**

数据集为在 ARX L5 与 Unitree G1 上收集的无接触轨迹（任务匹配与广泛无接触运动）以及书堆推送实验的接触试验，标定为不同的滑动阻力。

**📈 对比分析**

与基于 NEXT 的神经预测器和传统逆动力学残差法对比，CoPRE 在 ARX 上召回率达到 74.1%（F90=3.5 N），在 G1 上 82.2%（F90=5.5 N），两者均显著优于基线；基线在两机器人上几乎没有检测到弱接触。

**⚠️ 局限性**

局限性包括：需针对每台机器人进行无接触训练与校准，受运动多样性和负载变化影响；对持续接触的残差漂移不做处理；检测延迟未评估；仅验证了末端执行器的接触，未覆盖其他关节；真实工况下误报率可能变动。

---

## 274. Anomaly-Free Self-Optimization via AUC Bounds

**arXiv ID:** 2609.27362 | [PDF](https://arxiv.org/pdf/2609.27362v1)

**作者:** Kevin Wilkinghoff `[一作]`, Zheng-Hua Tan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种利用AUC上界作为可微分目标，直接在无异常数据的情况下对异常检测系统的连续参数（如集成权重和伪异常分数缩放）进行自适应优化的方法。

**💡 创新点**

将AUC上界从传统的模型选择指标转变为可优化的目标函数，实现无异常自我优化；引入可学习的伪异常分数缩放机制，提升优化质量；无需额外元数据或真实异常样本。

**🔧 技术方法**

使用基于均值和方差的AUC上界闭式表达式作为损失函数，利用梯度下降（Adam）优化集成权重和缩放因子；采用多种伪异常构造（Random、Feature、Sequence、Element、Cross-Domain/Attribute/Class）以及不同的嵌入与池化组合。

**📊 数据集**

在DCASE 2022–2025 语音/音频异常检测基准数据集（MIMII-DG、ToyADMOS2系列、IMAD-DS）上进行实验。

**📈 对比分析**

与Oracle-Selected、Fixed-Selected、Random-Selected、Equal、Pseudo-AUC-Selected、Bound-Selected等基线进行对比；实验表明在含ldn的评分方案下，连续优化的AUC上界显著优于基线，提升约1–2个百分点，且对伪异常构造的敏感度低；在无元数据的Feature构造下亦能取得高于多数基线的结果。

**⚠️ 局限性**

由于上界基于伪异常分布，仍与真实异常分布存在不匹配，导致在某些设置下伪AUC选择可能更优；目前仅对集成权重进行优化，未覆盖其他参数；需要进一步研究上界与真实异常的关系以及扩展到更广泛的系统参数。

---

## 275. EVAGE: Autonomous MEV Generation and Adaptation via Multi-Agent Harness

**arXiv ID:** 2609.27424 | [PDF](https://arxiv.org/pdf/2609.27424v1)

**作者:** Yan Wen `[一作]` (City University of Hong Kong), Chenyuan Wu `[通讯]` (City University of Hong Kong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一个完全自主的多智能体框架，用于自动生成、变体探索、协议迁移和跨链迁移的 MEV（Maximal Extractable Value）策略，能在 Ethereum、Base 和 BNB Smart Chain 上离线执行验证。

**💡 创新点**

创新点在于：①分层三智能体（Collector、Builder、Validator）闭环诊断，显著降低 LLM 推理延迟并避免幻觉；②通过自然语言交互实现灵活的任务分配与修正；③支持在不同行为模型（CPMM、CLMM、Balancer V2）和多链（L1、L2）之间自动迁移；④发现并验证了 5 种此前未知的 MEV 变体，利润提升可达 15.97 倍。

**🔧 技术方法**

技术包括：GPT‑5.4 LLM 与 Codex CLI 进行代码生成；Anvil/Foundry 进行离线 fork 执行与利润验证；Web3.py、SQLite 用于数据抓取与持久化；多智能体挂钩框架与自然语言交互实现任务闭环；统计与可视化工具用于评估规划延迟与盈利。

**📊 数据集**

数据集：共计 1.5M+ 区块（2025 年 2 月至 8 月）来自 Ethereum、Base、BNB Smart Chain 的完整交易与状态快照，包含不同 AMM 的合约调用与行情数据。

**📈 对比分析**

对比方法：与单一智能体 Pipeline‑Structured Agent 进行 ablation；评估候选数、计划率、盈利率及规划延迟。实验结果显示：规划延迟在 CPMM 与 BSC 上 <10 ms（最高 658 ms for CLMM HFT），跨协议/链迁移后保持盈利，且 LLM token 成本 < 60 美元。新变体利润提升 1.02–15.97 倍，整体系统在 3 条链上验证了 11 种参考策略和 5 种新变体。

**⚠️ 局限性**

局限性：对 CLMM、Balancer 等非连续流动性模型的搜索优化仍不完善；未模拟竞争和网络延迟，只在离线环境验证；对价格行情依赖外部数据，可能导致评估偏差；高频策略如 HFT 仍受限于单线程规划与执行。

---

## 276. RAMP: Reversing Adversarial Perturbations to Strengthen Clean-Label Backdoor Attacks against Malware Detectors

**arXiv ID:** 2609.27422 | [PDF](https://arxiv.org/pdf/2609.27422v1)

**作者:** Jinwen Xin `[一作]` (Wuhan University), Guojun Peng `[通讯]` (Wuhan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种针对深度学习恶意软件检测器的干净标签后门攻击增强方法RAMP，利用反向对抗扰动将良性可执行文件的特征向恶意区域移动，再注入触发器进行投毒；

**💡 创新点**

创新点在于从特征空间角度重新定义攻击：先让正常样本的表示偏向恶意，再注入触发器，形成更强的特征‑标签冲突；

**🔧 技术方法**

核心技术包括基于黑盒遗传算法的对抗优化、功能保持的PE文件插值（填充/移位）以及DOS头触发器注入；

**📊 数据集**

实验使用收集的4万条Windows PE样本（20k恶意、20k良性）以及预训练的MalConv和MalConvGCG两种深度模型；

**📈 对比分析**

与仅使用触发器的基线相比，RAMP在1%投毒率下在MalConv上实现约90%以上的攻击成功率（ASR），在MalConvGCG上也显著提升，且对已有高级触发器（MalPDT、PBA）同样有效；

**⚠️ 局限性**

局限性包括对特定PE结构的依赖、对功能保持插值的实现复杂度、以及对现有检测/修复防御（STRIP、Fine‑Pruning、ABL）抵抗性不足，未来需探索更通用的插值与防御策略。

---

## 277. When Labels Are Scarce: An Oscillatory State Space Model for Vibration Diagnosis

**arXiv ID:** 2609.27411 | [PDF](https://arxiv.org/pdf/2609.27411v1)

**作者:** Mainak Mallick `[一作]`, Seung-Kyum Choi `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DualRes 模型，用两种对齐的 STFT 视角结合选择性振荡记忆，对标签稀缺的机械振动信号进行高效诊断。

**💡 创新点**

创新点包括：① 双分辨率 STFT 的时间对齐与联合表征；② 可学习的衰减控制的选择性振荡状态空间记忆；③ 在数据集划分和标签时长上严格区分标签暴露、记录多样性与条件转移，形成系统的实验协议。

**🔧 技术方法**

技术：选择性振荡记忆（受 Mamba/S4 等状态空间模型启发），双分辨率 STFT 前端，归一化残差读取，衰减抖动，宏 F1 对比实验，并对参数量、存储、推理延迟等成本进行测评。

**📊 数据集**

数据集：六个公开轴承数据集（Paderborn、CWRU、KAIST、UORED-VAFCLS、HUST、Ottawa）；MCC5 变速箱基准；PHM2009 变速箱配置识别。

**📈 对比分析**

比较方法：在记录级分离、记录覆盖和标签时长明确的实验协议下，对同一标签预算内 9 种模型（MambaSL、MiniRocket、Attention 等）进行宏 F1、参数量、存储、推理延迟对比。DualRes 在 MCC5 主要预算下宏 F1 最高，提升 16.1% 点；轴承任务表现与多数方法持平或更优；参数仅 39k，检查点存储比对手小 24.8 倍，推理速度比 MambaSL 快 1.44 倍。

**⚠️ 局限性**

局限性：对不同机械类型与不同采样率的泛化仍需验证；对实时/流式推理的适配尚未完整；模型对窗口长度和采样率设置较为敏感，可能无法捕捉超长周期事件；标签分布变化的鲁棒性尚不充分。

---

## 278. Only Pay What You Must Spend: On-Demand Privacy Budget Payment for Differentially Private RAG

**arXiv ID:** 2609.27406 | [PDF](https://arxiv.org/pdf/2609.27406v1)

**作者:** Zhonghao Sun `[一作]` (National University of Defense Technology), Dongsheng Li `[通讯]` (National University of Defense Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出SparsePay-RAG框架，利用公开先验实现仅为真实私有增量付费的差分隐私检索增强生成（RAG）方案。

**💡 创新点**

核心创新在于三维稀疏化隐私预算：1）通过公开主题引导聚类局部化检索范围；2）用等调回归拟合跨层概率轨迹实现自适应私有触发；3）采用DP对比解码仅保护私有增量而非完整对数，从而显著压缩每次访问的隐私消耗。

**🔧 技术方法**

技术手段包括：公开主题生成与聚类、两阶段检索、等调回归拟合跨层轨迹、自适应触发阈值、对比解码（delta裁剪聚合）、zCDP/DP预算管理与序列组合。

**📊 数据集**

实验数据集涵盖开放域问答的Natural Questions、TriviaQA以及医疗对话隐私数据集ChatDoctor。

**📈 对比分析**

与Non‑RAG、Non‑Private‑RAG、DPVoteRAG、DPSparseVoteRAG、MuRAG等基线对比，SparsePay‑RAG在低隐私预算（ε=1,5）下仍保持最高准确率/ F1，且在高预算（ε=100）时几乎逼近非私有RAG的性能，整体隐私‑效用平衡最优。

**⚠️ 局限性**

局限性包括：预先分配固定的每查询隐私预算，导致对简单查询的预算利用率低；并未覆盖推理时延等侧信道信息，可能在实际部署中泄露检索信息。

---

## 279. PRISM-VLM: A Multi-Axis Discriminative Benchmark for Compact Vision-Language Models

**arXiv ID:** 2609.27395 | [PDF](https://arxiv.org/pdf/2609.27395v1)

**作者:** Sanghee Park `[一作]` (NAVER Cloud AI), Kee-Eung Kim `[通讯]` (KAIST AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PRISM‑VLM，一个七轴判别式基准，用于评估小型（≤10B参数或低成本 API 版）视觉‑语言模型的多维表现；

**💡 创新点**

创新点在于将任务质量、行为鲁棒性与能力瓶颈拆分为七个互补轴（质量、指令遵循、幻觉、多问答、同情倾向、计数、视觉辨别），并通过谐波平均形成 PScore，显著提高模型间可区分度；

**🔧 技术方法**

技术手段包括：①从15个公开 VQA/Chart 等基准中采样 6,238 项；②使用前沿模型 GPT‑5 生成结构化扰动与答案，随后用相同模型进行评测，并做交叉评审；③对每轴进行二元或规则评分；④使用项级配对 Bootstrap 统计检验；⑤采用跨判者审计确保无家族偏差；

**📊 数据集**

数据集：15 个公开视觉‑语言基准（DocVQA、OK‑VQA、ChartQA、ScienceQA 等）共 6,238 项；

**📈 对比分析**

比较方法：对 861 模型对进行项级配对 Bootstrap（B=2,000）统计区分度；PScore 在所有模型对中 97.0% 能实现显著差异，显著高于单轴或宏平均基准（44–80%）；单轴分数如同情倾向（sc）在 0.56 之间差距最大；

**⚠️ 局限性**

局限性：仅覆盖英文图文任务；七轴不全面（缺乏校准、工具使用、多轮对话、多语言等维度）；评测基于 GPT‑5 生成的扰动，可能存在家族偏好；未验证 PScore 与实际部署效果或人类偏好之间的关联；

---

## 280. EvoAudio: Recursive Self-Improvement for Audio Understanding

**arXiv ID:** 2609.27389 | [PDF](https://arxiv.org/pdf/2609.27389v1)

**作者:** Yuxiang Wang `[一作]` (Chinese University of Hong Kong Shenzhen), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 EvoAudio 系统，采用递归自我改进闭环来联合演化音频语言模型、音频波形、可验证问题与难度；

**💡 创新点**

首次实现模型、波形、问题与难度在同一闭环中共同进化，并通过可验证音频生成提供无需人工标注的数据；

**🔧 技术方法**

结合 LLM 生成问题、音频工具合成可验证波形、GRPO 强化学习、基于学习进度的自适应课程和音频特征验证；

**📊 数据集**

使用 MMSU、MMAU‑Pro、MMAR 三大评测基准，以及 LibriSpeech、FSD50K、AudioSet、MELD 等公开音频数据集；

**📈 对比分析**

与静态配置 GRPO、池化 GRPO 等基线对比，EvoAudio (GRPO) 在所有五种后端模型上均取得最高平均分，提升幅度最高达 6.3 分；

**⚠️ 局限性**

受限于可构建工具的范围和验证器的精度，难度匹配不够精准，导致部分问题对模型无效或误导学习。

---

## 281. From Intents to Algorithms: Verified Algorithm Discovery for Transport Networks

**arXiv ID:** 2609.27386 | [PDF](https://arxiv.org/pdf/2609.27386v1)

**作者:** Behnam Ojaghi `[一作]` (CTTC), Raul Muñoz `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VERA-TN框架，将意图网络需求通过受限的DSL转化为可验证的请求排序与路径排序程序，并通过可信分配器实现安全的资源控制；

**💡 创新点**

创新点在于将LLM作为语义变异器，限定其仅生成排序逻辑；构建静态与运行时验证层；设计结构化错误反馈；建立可信边界，确保生成程序不违背容量、时延和单路径约束；

**🔧 技术方法**

使用大语言模型生成抽象语法树；静态类型检查、沙箱执行；可信分配器实现路径验证与资源更新；精确MILP最优解作为参考；进化搜索（变异、重组、修复）结合结构多样性；统计检验（Wilcoxon、Holm）评估性能；

**📊 数据集**

实验基于TEFNET24衍生的28节点网络，包含150个持有案例；使用30个负载种子、K‑shortest路径；官方全国拓扑与12个未见的地铁区域拓扑做跨域测试；同时加入失败链路情景进行鲁棒性验证；

**📈 对比分析**

与最短路径、负载平衡KSP、优先级贪婪、等预算随机搜索及精确MILP（优先级与拥塞）进行对比；演化搜索在持有集上平均优先级效用比为0.958，略优于随机（0.952）和贪婪（0.940），差异显著但幅度小；在拥塞、失败训练与意图迁移方面未见显著优势；

**⚠️ 局限性**

局限性包括：仅使用10参数数值表征，未实现结构化AST生成；实验仅覆盖单一网络规模与拓扑族；缺乏动态流量、重构成本、数字孪生验证；未与遗传编程等基线直接对比；因此对LLM在结构搜索上的提升尚未得到充分证明。

---

## 282. Forecast Workflow Bench: Evaluating Language-Model Decisions with Budgeted Forecast Tools

**arXiv ID:** 2609.27385 | [PDF](https://arxiv.org/pdf/2609.27385v1)

**作者:** Shunya Nagashima `[一作]` `[通讯]`, Shunya Nagashima

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究通过构建FWBench基准，对1,251个电力和单车租赁案例中使用时间序列基础模型（TSFM）的决策代理进行评估，衡量其在容量决策中的收益与预测成本；

**💡 创新点**

创新点在于提出一种统一的损失‑成本评估框架，既考虑预测误差又考虑决策成本；同时提供可复现的基准，用于评估LLM/SLM在有限预算下选择并使用TSFM的能力；

**🔧 技术方法**

采用大型与小型语言模型（如GPT‑6 Astra、Gemma、Qwen），时间序列基础模型（Chronos、TimesFM），动态规划优化器、预测成本记账以及群集自助法进行统计分析；

**📊 数据集**

使用美国能源信息署EIA‑930的电力需求数据和伦敦交通局TfL的单车租赁出发量数据，共1,091条电力案例和160条单车租赁案例；

**📈 对比分析**

通过与固定政策、回测选择器以及最优oracle进行对比，评估决策质量；Astra模型仅使用2.5%预测预算即可达到与固定Chronos‑2相当的损失‑成本分数，优于其他免费策略；本地SLM表现相对落后；TSFM可访问性对性能提升有限；

**⚠️ 局限性**

局限性包括仅覆盖两个领域、16个日期的模拟合同、固定预测费率、单一校准周期，统计置信区间仅基于抽样日期，未对多重检验做校正，缺乏端到端延迟、内存和能耗评估等。

---

## 283. Cross-Lingual Legal QA for Vietnamese Labour Law: Retrieval, Translation, and Verifier-Guided Correction

**arXiv ID:** 2609.27376 | [PDF](https://arxiv.org/pdf/2609.27376v1)

**作者:** Nguyen Minh Chi `[一作]` (VinUniversity), Paul Rayson `[通讯]` (Lancaster University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究跨语言法律问答，构建越南劳动法双语评测集，并提出检索‑生成‑验证‑纠正的完整管道。

**💡 创新点**

首次将双语评测套件、Verifier‑guided 两阶段验证与六维检索证据诊断整合到跨语言法律 QA 体系中。

**🔧 技术方法**

使用 BGE‑M3 密集多语言检索、Qwen3‑8B 生成、GPT‑4o‑mini 翻译、规则+NLI 验证与纠错重写技术。

**📊 数据集**

基于 231 条越南劳动法问答对（含 75 条难点标签）及 953 篇法规条文构建评测数据集。

**📈 对比分析**

对比四种跨语言配置与检索模式，dense retrieval R@5=0.358 大幅优于稀疏 0.032；翻译位置对六维诊断无显著影响；Verifier 纠错仅提升引用保持，整体诊断平均 0.612。

**⚠️ 局限性**

仅限越南劳动法领域，检索/翻译对检验效果有限，自动诊断与人工评估偏差；缺乏实时版本控制与专业法律验证，易出现错误或过时信息。

---

## 284. Planned Test-Time Scaling with Coordinated Reasoning Paths

**arXiv ID:** 2609.27374 | [PDF](https://arxiv.org/pdf/2609.27374v1)

**作者:** Xueqing Wu `[一作]` (University of California, Los Angeles), Kai-Wei Chang `[通讯]` (University of California, Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种“计划化测试时刻缩放”(PTTS)框架，利用规划器生成多条互异的解题大纲，随后由执行器按大纲生成完整解答，以此提高在有限计算预算下的通过率。

**💡 创新点**

核心创新在于将传统独立采样转变为联合规划与执行的两阶段结构，规划器通过协同生成多样化大纲显著提升解答多样性，且通过强化学习直接对 k-通过率进行优化，克服模式崩溃与冗余采样问题。

**🔧 技术方法**

采用大语言模型作为规划器与执行器（如 Qwen3-1.7B/4B），零样本版 PTTS-ZS 在单次自回归推断中生成 k 条大纲；训练版 PTTS-RL 则使用 TRPO/GRPO 对规划器进行强化学习，奖励为 k-通过率，且通过执行器预算截断提升训练效率。

**📊 数据集**

在五个数学推理基准上评估：MATH-500、AIME 2024/2025/2026、HMMT-Feb26。

**📈 对比分析**

与重复采样（RS）和 Guided Sampling 进行对比。实验显示：PTTS-ZS 在 64 次采样上提升 6.7/3.4 分；PTTS-RL 在 64 次采样上提升 13.4/6.7 分，并在 32 次采样即可超过 RS 的 64 次采样表现，体现出更高的计算效率。

**⚠️ 局限性**

局限性包括：① 需要高质量的大纲规划器，若规划器不够多样化或与执行器不匹配，收益受限；② 强化学习训练成本高且对执行器截断预算敏感；③ 目前仅在数学推理任务验证，缺乏对更广泛推理场景的评估；④ 大纲遵循率虽不错但仍可提升，尤其在长篇推理中执行器可能偏离规划。

---

## 285. Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models

**arXiv ID:** 2609.27373 | [PDF](https://arxiv.org/pdf/2609.27373v1)

**作者:** Ke Wan `[一作]` (University of Virginia), Chen Chen `[通讯]` (University of Virginia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了循环语言模型中注意力路由与表示细化的时间尺度差异，并提出了WISE方法，先全局搜索再利用稀疏工作集进行后续推理，以减少推理成本。

**💡 创新点**

发现路由稳定早于表示细化，并提出训练无关的两阶段推理——先全局注意再重用稀疏工作集，保持后续表示更新。

**🔧 技术方法**

结合循环Transformer、块级稀疏注意力、GPU稀疏核（Triton）、实验对照干预和多跳QA基准评估技术。

**📊 数据集**

使用HotpotQA、GSM8K、2WikiMultiHopQA等多跳QA数据集，并在不同上下文长度（512~4K）进行对比实验。

**📈 对比分析**

与全局注意力对照，评估精度（F1）、答案变更、稀疏密度等指标；WISE在保持≈97%全注意力质量的同时，在4K上下文可实现1.76×注意力加速。

**⚠️ 局限性**

在4K上下文出现轻微质量下降；块大小与稀疏度折衷需手动调优；实现依赖专用稀疏核，未与FlashAttention完全共设计，可能限制进一步加速。

---

## 286. Anchor and Perturb: Lazy Agent Remediation by Exploration Injection

**arXiv ID:** 2609.27365 | [PDF](https://arxiv.org/pdf/2609.27365v1)

**作者:** Chengxi Zhong `[一作]`, Yongzhe Chang `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出Anchor-and-Perturb（AnP）框架，通过在CTDE设置下局部地提升一个“懒惰”智能体的探索率并保持其余智能体保持贪婪策略，以此来打破多智能体协作中的相对过度泛化与协调失败；

**💡 创新点**

创新点在于将探索性方差与协作收敛完全解耦，采用异步单点探索与固定同伴的策略组合；通过“懒惰差异指数”（Laziness Disparity Index）实现智能体性能失衡的精准诊断；不需要改造混合网络结构，保持现有QMIX/WQMIX/QPLEX等基底不变；

**🔧 技术方法**

技术包括：CTDE训练框架、基于QMIX的价值分解、局部ε-探索提升与退火、离线/在线懒惰诊断指标、自动化触发逻辑（LLM元监督）以及实验中使用的实验监测与日志分析；

**📊 数据集**

实验数据集未明示，推测使用标准多智能体合作环境（如StarCraft II或Multi-Agent MuJoCo）进行训练与评估；

**📈 对比分析**

与传统对称探索的QMIX对比，AnP在相同训练预算下从5%崩溃点恢复到85%评估胜率，并在非崩溃状态下突破子最优平台，实现90%的峰值胜率；不需网络改造即可显著提升稳定性和最终性能；

**⚠️ 局限性**

局限性包括：实验仅在单个智能体（M=1）单周期（K=1）下验证；缺乏多随机种子统计验证；自动化触发机制尚未成熟；对更大规模多智能体或更高秩的扰动的效果未知；对非单调混合网络的推广仍待探索。

---

## 287. Beyond Mean Foils: Auditing Worst-Foil Specificity in Frozen CLIP Region Explanations

**arXiv ID:** 2609.27356 | [PDF](https://arxiv.org/pdf/2609.27356v1)

**作者:** Kaixin Liu `[一作]` (Taizhou Institute of Science and Technology, Nanjing University of Science and Technology), Qihang Wu `[通讯]` (Taizhou Institute of Science and Technology, Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对冻结的 CLIP 进行 Cluster-based Concept Importance（CCI）解释的审计，量化区域在重叠目标时对竞争类别贡献更大的情况，并尝试通过 Worst‑Foil 重新排序与目标下降容忍度相结合的方式来修复失败解释。

**💡 创新点**

首次发现 CCI‑top1 区域常在目标框内但对单一竞争类贡献更大；提出 Worst‑Foil 重新排序并设定目标‑下降阈值的修复策略；系统评估了候选区域可用性、容忍度和竞争类移除对修复机会的影响。

**🔧 技术方法**

使用 CCI 区域重要性评估、CLIP B/16 与 B/32 固定模型、Worst‑Foil 重新排序、重叠与均值竞争检查、oracle 评估、bootstrap 置信区间等技术。

**📊 数据集**

COCO val2014 与 VOC2007 数据集，配合 CLIP 的 B/16 与 B/32 两个检查点进行实验。

**📈 对比分析**

通过比较 CCI‑top1 基线失败率、去除标注竞争者后变化、Worst‑Foil 重新排序后的修复率以及目标得分和 bbox 准确率等指标；结果表明仅有约 0.2%–1% 的失败能够在容忍度 ϵ=.02 下被修复，放宽容忍度可提升至约 1%–7% 但仍受候选区域限制。

**⚠️ 局限性**

修复机会有限：大多数失败在现有 8 个候选区域内无法满足目标‑下降阈值；只考虑冻结 CLIP 的解释，未验证在其他模型或更大候选集上的适用性；评估受限于 patch‑级分割与固定阈值的假设。

---

## 288. MolDesignBench: Evaluating LLM-based Agent for Scenario-grounded Molecular Design

**arXiv ID:** 2609.27349 | [PDF](https://arxiv.org/pdf/2609.27349v1)

**作者:** Yongjun Jeong `[一作]` (Korea University), Sungwoong Kim `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MolDesignBench——一个基于真实场景的分子设计基准，包含2000个多约束生成与优化实例，评估工具增强LLM代理在隐式约束推理、冲突约束满足、不可行性判断和多轮工具使用上的表现。

**💡 创新点**

创新点在于同时涵盖四大核心维度（隐式约束推理、多约束满足、化学不可行性判断、适应性多轮工具推理），并提供17种专用化学工具与精细化失败模式分析，填补了现有基准在场景真实性和多工具交互上的空白。

**🔧 技术方法**

采用大语言模型（GPT‑5.4、Claude‑Opus‑4.6、Qwen3‑235B等）结合Python、FastAPI实现的函数调用工具集，通过标准化评估协议和工具调用接口进行实验。

**📊 数据集**

使用ChEMBL和ZINC20数据库构建候选分子池，结合30个同行评议属性过滤器生成约束，并在不同设计阶段和策略下生成场景文本，最终形成2000个实例。

**📈 对比分析**

对比闭合、开放和化学专用LLM，在工具使用与不使用两种条件下进行生成与优化任务评估，最佳模型（如Qwen3‑235B）在生成任务上成功率约43%，显示即便有工具支持，LLM代理在隐式约束和多轮推理上仍表现有限。

**⚠️ 局限性**

局限性包括仅关注小分子设计、依赖计算预测属性而非实验验证、未覆盖聚合物或晶体等复杂化学体系，以及未评估生成分子在实际实验中的创新价值。

---

## 289. BladeMaster: Real-Time Robotic Cutting Simulation with Online-Generated Persistent Discontinuities

**arXiv ID:** 2609.27342 | [PDF](https://arxiv.org/pdf/2609.27342v1)

**作者:** Zhanyu Yang `[一作]` (Purdue University), Chenfanfu Jiang `[通讯]` (UCLA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个GPU加速的BladeMaster框架，用于在机器人切割过程中在线生成并维护切口，支持连续、交叉和多刀切割，并实现刀具与材料之间的双向耦合。

**💡 创新点**

创新点在于使用持久的切面标签编码材料点，直接在TLMPM中控制粒子-网格耦合，从而在刀具撤退后保持断裂面的机械分离；同时引入材料-材料接触模型，使切口可重新接触滑动而不恢复连接。

**🔧 技术方法**

技术方法包括：基于总拉格朗日MPM（TLMPM）的显式离散化、切面标签的在线赋值、兼容性耦合与冲突检测、Coulomb摩擦接触处理、以及GPU并行实现和刀具的刚体接口。

**📊 数据集**

实验使用自定义的弹性块、香蕉等软体材料的仿真，未使用公开数据集；通过与真实机器人刀切实验对比验证切割阻力与切割结果。

**📈 对比分析**

与传统网格切割方法（如DiSECt）和单场MPM基线相比，BladeMaster能够在刀具撤退后保持切割面分离，并在多刀切割后实现独立操作；在多种示例中实现了快于实时的性能（wall/sim 0.33–0.62）。

**⚠️ 局限性**

局限性包括：刀具-材料接触仅采用球形代理，无法严格保证非渗透；只处理刀具引起的切口，无法模拟应力诱导的裂纹生成与传播；以及持久标签的状态数随切割数增加而增长，可能导致内存占用上升。

---

## 290. Spatial and Semantic Reasoning for LLM-Driven Robot Navigation via MCP

**arXiv ID:** 2609.27340 | [PDF](https://arxiv.org/pdf/2609.27340v1)

**作者:** Jungsoo Lee `[一作]` (Hanyang University), Wansoo Kim `[通讯]` (Hanyang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一个非侵入式框架，利用可视化地图表示和语义注释模块，将大型语言模型（LLM）与ROS导航栈连接，实现自主管理地图构建、空间推理导航和语义推理导航。

**💡 创新点**

提出基于Model Context Protocol（MCP）的标准化工具层，使LLM能够直接读取ROS导航数据的可视化地图和语义注释表示，从而在未改动ROS栈的情况下实现空间和语义推理；同时提供可复用的视觉地图生成和语义注释工具。

**🔧 技术方法**

使用MCP协议、ROS-MCP Server、Python工具装饰器、占用网格转灰度图算法、YAML语义注释规范、LLM推理（Claude Sonnet 4.6、GPT-5.5）以及NVIDIA Isaac Sim仿真。

**📊 数据集**

在NVIDIA Isaac Sim仿真环境下的室内仓库地图（两间房间、货架、叉车、消防器材）与对应的占用网格、相机图像构成自建数据集，未使用公开数据集。

**📈 对比分析**

通过对比两种LLM后端在三项任务（自主管理地图构建、空间推理导航、语义推理导航）中的表现：任务1地图覆盖率分别为97.83%和99.78%；任务2空间推理导航成功率分别为86.67%和93.33%；任务3语义推理导航成功率100%；结果表明框架在不改动ROS栈的前提下能实现高效导航。

**⚠️ 局限性**

对空间结构歧义敏感，LLM推理错误可能导致导航失败；语义注释需要完整且准确；目前仅在仿真环境验证，缺乏真实机器人实验；性能受LLM后端能力与表示质量影响。

---

## 291. Omnidirectional Amphibious Locomotion via Internal Mass Actuation

**arXiv ID:** 2609.27358 | [PDF](https://arxiv.org/pdf/2609.27358v1)

**作者:** Niko Weaver `[一作]` (Duke University), Boyuan Chen `[通讯]` (Duke University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一款全封闭球形两栖机器人，利用内部三轴线性滑块实现陆地滚动与水面推进，且无机械重构；

**💡 创新点**

同一内部质量分布机制即可驱动陆地滚动与水面推进，球形壳兼作接触与浮力表面，配合几何与强化学习控制实现全向运动；

**🔧 技术方法**

内部质量滑块、球形壳、被动鳍、IMU姿态估计、CAN通信、几何控制算法、强化学习控制（FlashSAC）、MuJoCo仿真与实验硬件；

**📊 数据集**

未使用公开数据集，而是在MuJoCo中搭建多线程随机化训练环境，实验数据来自机器人实测；

**📈 对比分析**

将几何控制与学习控制在陆地、海面以及过渡阶段进行对比，学习控制在水面上平均速度0.348 m/s、最大速度0.592 m/s，几何控制平均0.252 m/s；两者均实现全向运动，学习控制更快；

**⚠️ 局限性**

仅支持水面推进，学习控制依赖近似水动力模型，缺乏自主导航与感知，未实现水下推进与更复杂环境下的自适应控制。

---

## 292. Passing: An Endless Journey through Reconstructed Spacetime with AI-Generated Sound

**arXiv ID:** 2609.27489 | [PDF](https://arxiv.org/pdf/2609.27489v1)

**作者:** Akira Takahashi `[一作]` (Sony Group Corporation), Yuki Mitsufuji `[通讯]` (Sony Group Corporation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建了名为Passing的互动视听装置，通过将单轨车窗录制的连续视频重构为时空体积并按非线性轨迹重排，实现无尽的视听旅程，并将视觉重构实时映射给V2A模型生成同步声景。

**💡 创新点**

创新点在于将空间与时间维度重新组合成时空体积，利用观众在场感驱动的轨迹分支控制实现分布式创作代理，并将实时视频到音频合成视为投机式聆听，探讨音频在非线性时空中的存在与听觉体验。

**🔧 技术方法**

技术包括时空体积重构与非线性跨截面采样、观众存在检测与概率分支控制、SpecMaskFoley实时视频到音频模型、ControlNet+FT-Aligner实现视频-音频同步，以及使用OSC进行控制信号传递。

**📊 数据集**

使用了原始4K 480fps高帧率单轨车窗录制、SpecMaskGIT、SpecMaskVQGAN、Vocos等模型预训练于AudioSet和VGGSound，并增补约35分钟Tama单轨拍摄的自定义视听数据。

**📈 对比分析**

通过现场主观听感评估A/B循环与C/D过渡序列的音频稳定性与多样性，技术上SpecMaskFoley在RTX 4080实现32kHz立体声实时合成，保持同步且无卡顿，音频与视觉高度一致。

**⚠️ 局限性**

局限在于模型缺乏主体经验导致音频与视觉的不一致性和模糊对应，受限于预训练数据的声学范畴，以及实时推理对硬件性能高度依赖，可能在更高分辨率或更大规模时产生延迟。

---

## 293. Collocated Shape Regulation for Soft Robots

**arXiv ID:** 2609.27469 | [PDF](https://arxiv.org/pdf/2609.27469v1)

**作者:** Pietro Pustina `[一作]` (Sapienza University of Rome), Cosimo Della Santina `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于协同形式的柔性连续机器人形状调节框架，并设计了多种PD、PID等控制器，随后在自研软机械臂上进行实验验证。

**💡 创新点**

创新点在于统一的协同控制理论，给出局部、半全局、全局稳定性的控制器族，并在不同离散化和模型下实现无模型全局收敛。

**🔧 技术方法**

采用了协同形式的动力学重构、能量保持的PD/PID控制器、饱和积分法、基于模型的前馈补偿以及实时逆运动学与状态估计技术。

**📊 数据集**

实验使用了自制Eco‑Flex 20硅胶柔性臂、双线索张力电机、OptiTrack 3D摄像头采集的轨迹数据作为数据集。

**📈 对比分析**

通过对比例增益、负载、离散化模型和控制器类型的组合实验，利用RMSE、稳态误差、超调、上升/调节时间等指标进行比较，结果显示PD控制器收敛快、PID误差更低，且高阶离散化能提升形状描述精度。

**⚠️ 局限性**

局限性包括需要完整状态观测或高精度传感，控制器对模型误差敏感，积分和饱和积分会引起振荡，且在高阶模型下传感器布局影响性能。

---

## 294. Safety-Filtered Distributed Koopman-MPC

**arXiv ID:** 2609.27463 | [PDF](https://arxiv.org/pdf/2609.27463v1)

**作者:** Shengjun Zhang `[一作]` (Hubei University), Zhenglong Sun `[通讯]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种分布式Koopman模型预测控制框架，将轨迹预测与碰撞约束分离，采用本地感知构造硬性有限保持约束，将规划输出投影到安全集合上；

**💡 创新点**

创新点在于：1) 通过“预测-投影”双通道实现对丢包鲁棒；2) 引入基于方向余量的硬性有限保持约束与补充权重分配，实现无安全裕度松弛的实时安全投影；3) 提供完整的理论证明（分离定理、可行性检验、无交互边图切换条件）并在仿真中验证；

**🔧 技术方法**

使用技术包括：EDMDc Koopman提升（19维）、线性预测、两阶段QP（规划QP与安全投影QP）、方向权重分配、在线可行性检查、基于零阶保持的安全保障理论；

**📊 数据集**

使用的数据集为仿真生成：220条22步的轨迹用于训练Koopman模型，随后在固定的仓库布局上进行8机器人、240步的匹配试验，包含5%与15%丢包；

**📈 对比分析**

对比方法包括ORCA、目标反馈+投影、无投影的预测Koopman-MPC；实验结果显示全方法在所有20次试验中安全率20/20、目标成功率160/160，目标误差0.0065m，且最小间距分别为0.676m与0.321m；

**⚠️ 局限性**

局限性在于：1) 依赖本地感知误差与残差上界；2) 无全局路径规划与死锁避免，仅关注安全投影；3) 规模扩展受感知图稀疏性与QP求解时间限制；4) 需要预先训练Koopman模型，环境变化时需重训练。

---

## 295. Beyond Balanced Accuracy: A Resolution and Parity-Controlled Benchmark for Vision-Language and Vision-Only Defect Assessment in UAV Power-Line Inspection

**arXiv ID:** 2609.27457 | [PDF](https://arxiv.org/pdf/2609.27457v1)

**作者:** Linghao Zhang `[一作]` (State Grid Sichuan Electric Power Research Institute), Peiyu Yi `[通讯]` (State Grid Sichuan Electric Power Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对无人机电力线路缺陷评估进行了系统性对比，构建了 ElecVQA‑Bench benchmark，并在六个关键实验轴上（partition、item set、label space、replication、resolution、side information）对 Vision‑Language Models（VLM）和传统 Vision Backbones 进行了性能评估。

**💡 创新点**

提出了六轴平衡审计协议，揭示了评估偏差如何导致模型优势被夸大；首次量化宏平均对稀有类别的敏感性和随机种子/分割对结果的影响；通过像素预算审计与 token 统计解析 VLM 与 vision backbone 之间差异的根源；公开 benchmark、构建脚本和权重，促进可复现性。

**🔧 技术方法**

使用了 LoRA 低秩微调、成本敏感风险模型与拒绝选项、宏平均与 McNemar 等统计检验、像素预算审计、token 计数、prompt ablation、量化（INT8、NF4、FP8）与部署性能评测。

**📊 数据集**

使用公开的 InsPLAD UAV 图像数据集（10,607 张），通过分割和扩展框得到 56,972 条 QA 项目，覆盖 5 种资产和 7 种缺陷类别。

**📈 对比分析**

在完全匹配的测试文件、相同的项目集、相同的标签空间、相同的种子数、相同的输入分辨率以及相同的侧信息下，两类模型的差距极小。Binary 任务两者差距仅 0.03%；七类任务差距主要由像素预算和稀有类导致；两大 VLM（InternVL3.5‑8B 与 Qwen3‑VL‑8B）之间差距 11.18 pt，且未能通过现有控制解释；结果对 seed 和分割高度敏感。

**⚠️ 局限性**

实验受限于：①分割/种子不稳定导致结果可重复性差；②token 预算未匹配，难以排除 tokenizer 设计差异；③宏平均对稀有类别极度敏感，导致 headline 变化受少数样本影响；④部署评估仅在工作站 GPU 上完成，未覆盖能耗、热稳定性和嵌入式加速器；⑤VLM 与 vision backbone 在侧信息输入上不可直接匹配，导致系统级比较的可解释性受限；⑥实验覆盖面有限，未进行跨域验证。

---

## 296. Compressed delayed-information projection for six-degree-of-freedom underwater vehicle navigation under delayed acoustic positioning

**arXiv ID:** 2609.27439 | [PDF](https://arxiv.org/pdf/2609.27439v1)

**作者:** Shuyue Li `[一作]` (Xi'an Jiaotong-Liverpool University), Xiaohui Qin `[通讯]` (Jiangsu JITRI Tsingunited Intelligent Control Technology Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了压缩延迟信息投影（CDIP）方法，用于在海底无人艇导航中以低计算成本处理延迟的声学定位数据。

**💡 创新点**

创新点在于保留源时刻的快照及其与当前状态的交叉协方差，直接将过期观测投影到当前估计状态，从而避免完整的历史回溯/重放。

**🔧 技术方法**

采用15维误差状态扩展卡尔曼滤波（ESKF），历史-当前交叉协方差维护，固定延迟/重放对照实验，以及统计检验和一致性诊断。

**📊 数据集**

使用200条RexROV仿真录制（154条可用）进行实验，包含固定1.5 s延迟的配对实验、30条/延迟的预设延迟扫荡，以及5条/持续失效的受限实验。

**📈 对比分析**

与基线（当前时刻更新）、CDIP和精确重放三种处理方式比较，CDIP在1.5 s延迟下轨迹RMSE降低57%并仅比重放高1.03%，每次更新耗时下降99.2%；延迟扫荡表明CDIP保持低误差但并不总是优于重放，失效实验中CDIP误差下降约77%且耗时极低。

**⚠️ 局限性**

局限性包括仅在单一UUV仿真环境评估，未考虑网络层随机丢包、时钟同步误差或多UUV协同；未证明统计等价或普适优越，需在更大尺度或真实海况中进一步验证。

---

## 297. Energy-Oriented CGLA Mapping of a Memory-Polynomial Digital Predistortion Kernel

**arXiv ID:** 2609.27438 | [PDF](https://arxiv.org/pdf/2609.27438v1)

**作者:** Takuto Ando `[一作]` (Nara Institute of Science and Technology), Yasuhiko Nakashima `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究者将 (P,M)=(5,5) 的奇数阶记忆多项式数字预失真（DPD）简化核映射到可编程的一维线性阵列（CGLA）IMAX 加速器上，并对其在 FPGA 原型与 ASIC 预测版上的时延、能耗与正确性进行评估。

**💡 创新点**

创新点包括：① 把奇数阶记忆多项式的局部重用复数 MAC 核重新表述为适合 CGLA 的一维流水线结构；② 设计了 33 阶 PE/LMM 流水线，将 120 B 系数完全保留在本地内存，实现 15 组阶延时项的顺序累加；③ 明确区分核本身时延与端到端时延；④ 在同一单精度工作负载下，与 RTX 4090 CUDA、Jetson AGX Orin ARM‑NEON 进行对比，展示了显著的能耗优势。

**🔧 技术方法**

采用的技术包括：IMAX CGLA（1D PE/LMM 流水线）、复数 MAC 指令、局部存储与流水线数据流、基于 ARM 的主机控制与 DMA 传输、以及合成的记忆多项式 PA 模型进行校验。

**📊 数据集**

使用的“数据集”是 32 条独立序列，每条 2048 复数样本（共 65,536 复数样本/批次）作为主工作负载；同时使用合成的 OFDM‑类波形对 15 组阶延时模型进行验证。

**📈 对比分析**

比较方法：在相同单精度复数工作负载下，测量端到端时延、核时延以及能耗。结果显示：RTX 4090 最短端到端时延 0.484 ms，FPGA 原型 20.201 ms，Jetson AGX Orin 4.478 ms，预测的 28 nm IMAX ASIC 3.14 ms；能耗方面，预测 IMAX 1.86 mJ/批次，RTX 4090 314.6 mJ（≈169×），Jetson 67.2 mJ（≈36×），体现出显著能耗优势。

**⚠️ 局限性**

局限性：仅做了核级映射与评估，未实现完整的实时发射机集成；基准扩展仍在主机端完成，未将其搬入阵列；系数固定，未考虑系数自适应或连续流处理；未涉及定点量化、实际 RF 测试或 EVM 限制；能耗评估基于功率模型而非现场硅功耗。

---

## 298. The Sharp Rényi and Tsallis Threshold in the Shepp--Olkin Concavity Problem

**arXiv ID:** 2609.27433 | [PDF](https://arxiv.org/pdf/2609.27433v1)

**作者:** Haoran Wang `[一作]` `[通讯]` (Independent Researcher), Haoran Wang (Independent Researcher)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了独立伯努利随机变量的和的香农熵、Rényi熵和Tsallis熵的联合凹性，确定了确切的范围。

**💡 创新点**

确定了Rényi和Tsallis熵的联合凹性范围为0<q≤1，并证明了在q>1时联合凹性失效。

**🔧 技术方法**

使用了传输不等式、非线性望远镜修正和Riccati恒等式等数学技术。

**📊 数据集**

研究对象为独立伯努利随机变量的和，未具体提及数据集。

**📈 对比分析**

通过与Hillion和Johnson的结果进行比较，证明了在q=1时香农熵是联合凹的，而在q>1时联合凹性失效。

**⚠️ 局限性**

在q>1时，联合凹性失效的情况已经在两个伯努利变量的情况下显现，限制了该结果的适用范围。

---

## 299. KITE: Scaling Jev Population Experiments with Sparse Flagship Calibration

**arXiv ID:** 2609.27535 | [PDF](https://arxiv.org/pdf/2609.27535v1)

**作者:** Hengyu Li `[一作]` `[通讯]` (University of Tokyo), Hengyu Li (University of Tokyo)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在KITE框架下，利用typed行为核、稀疏旗舰校准与共享误差传播，对大规模人群实验进行快速、可审计的模拟，并评估干预效果。

**💡 创新点**

创新点在于将轻量级行为核与稀疏旗舰校准、表格化执行及误差传播相结合，显著降低模型与人类差异并提升决策收益。

**🔧 技术方法**

采用TypeSafe的Jev核心、稀疏旗舰校准器、表格化执行引擎和基于最大似然的误差传播模型。

**📊 数据集**

使用公开数据集Epstein、SocSci210和Arechar进行评估。

**📈 对比分析**

通过与人类实验的MAE、捕获决策增益、覆盖率等指标比较，稀疏校准将效果误差降低41%，决策增益提升约30%，并在多国内容一致性测试中实现93%置信覆盖。

**⚠️ 局限性**

主要限制包括仅在两个持出实验上验证、缺乏跨域误差校准、旗舰校准可能引入错误、网络和记忆层未验证，以及模型可复制性受限。

---

## 300. MDRC: A Deployable State-Recovery Defense for Traffic Signal Control under Sensor Corruption

**arXiv ID:** 2609.27528 | [PDF](https://arxiv.org/pdf/2609.27528v1)

**作者:** Mingyuan Li `[一作]` (University of Turku), Ren Ping Liu `[通讯]` (University of Technology Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种在感知与控制之间插入的后检测状态恢复防御框架 MDRC，以提升交通信号控制系统对传感器破坏与对抗攻击的鲁棒性。

**💡 创新点**

创新点在于将去噪扩散隐式模型 (DDIM) 与 Reptile 元学习相结合，实现跨城市可迁移的状态恢复，并在恢复后直接给原始控制器提供可信状态，兼顾实时性与泛化能力。

**🔧 技术方法**

主要技术包括 DDIM 采样、Reptile 元学习、条件扩散恢复 (Repaint)、以及在硬件环回测试中的 GPU 推理加速。

**📊 数据集**

使用了七个基于 CityFlow 的真实城市数据集（江南、杭州、纽约等）以及真实路边探测器的时间序列和硬件环回实验数据。

**📈 对比分析**

与传统固定时序、Max Pressure 以及 RL 控制器（CoLight、Mplight、RobustLight 等）比较，MDRC 在随机、策略感知及传感器缺失攻击下平均降低 ATT 6.77%~12.75%，恢复误差显著下降，推理延迟比 DDPM 降低约 90%。

**⚠️ 局限性**

局限在于仍需依赖先前的异常检测器，针对极端结构性缺失或高度适应性攻击的恢复效果有限；跨城迁移在大规模缺失场景下仍需更多数据微调。

---

## 301. NV-Reason-CT: 3D Visual Language Model for CT Analysis

**arXiv ID:** 2609.27511 | [PDF](https://arxiv.org/pdf/2609.27511v1)

**作者:** Andriy Myronenko `[一作]` (NVIDIA), Daguang Xu `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种全新的生成式 CT 视听语言模型 NV‑Reason‑CT，能够以原生 3D 体积为输入，自动生成结构化的胸腹 CT 影像报告、问答与临床推理，并在报告后提供可验证的解释。

**💡 创新点**

创新点包括：① 在视觉编码器与 LLM 之间保留完整的 3D 空间索引（depth‑height‑width），使得模型能在语言解码过程中保持空间信息；② 采用专家引导的“推理‑中心”监督，将 radiologist‑recorded 叙述与重写报告相结合，生成可解释的 reasoning；③ 结合 Group Relative Policy Optimization (GRPO) 的可验证强化学习，用区块级异常标签和报告结构奖励来优化报告质量，而不需要每个体积都附带人工推理轨迹。

**🔧 技术方法**

核心技术包括：Primus 3D ViT（COLIPRI 权重初始化）+ Qwen3.5‑4B 语言模型；多轴旋转位置编码（MRoPE）和 3D 可变维投影器；自监督预训练、对比学习、DINO‑style 图像自监督；SFT + GRPO 的联合训练；可解释推理文本与多轮对话。使用的预处理和裁剪技术保证胸/腹区的精确定位。

**📊 数据集**

训练数据：约 550,000 条多模态指令示例，来自 CT‑RATE（胸部）、CancerVerse（胸腹部）和内部 NIH 集合，共 70,111 张 CT 体积；另外包含 12,000 条 radiologist 叙述；synthetic 叙述、问题回答、对话与拒绝示例。评估数据集：CT‑RATE（胸部、校正 v2 3,002 例）、Merlin（腹部 5,125 例）、RAD‑ChestCT（外部 3,630 例）以及 NIH 内部 held‑out cohort。

**📈 对比分析**

与同类模型比较：在 CT‑RATE 异常检测（macro‑F1 0.614）和报告生成（报告 macro‑F1 0.592）均位列榜首；在 Merlin 30‑findings 检测中 macro‑F1 0.766、macro‑AUROC 0.833；在 RAD‑ChestCT 上 macro‑F1 0.531、macro‑AUROC 0.796；报告生成指标（ROUGE‑L、BERTScore、RadGraph‑F1、GREEN）在 Merlin 评估中均优于其它公开模型。模型在多模态推理、问答与结构化报告方面均显示出较高的临床可用性。

**⚠️ 局限性**

局限性包括：① 训练和推理依赖大量 GPU 资源；② 对体积裁剪的 heuristic 可能在病灶位于边缘时失效；③ 解释文本仍需人工评估其真实性与完整性；④ 在跨模态推理中缺乏对罕见/未见异常的泛化能力；⑤ 由于报告格式差异，跨数据集的直接报告评估仍受限。

---

## 302. Know-Your-Scene (KYS)-SLAM: Hierarchical Semantic-Motion Priors for Feature Matching in Stereo Visual SLAM

**arXiv ID:** 2609.27509 | [PDF](https://arxiv.org/pdf/2609.27509v1)

**作者:** Preeti Chatterjee `[一作]` (University of Georgia), Suchendra M. Bhandarkar `[通讯]` (University of Georgia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

KYS-SLAM 在 ORB‑SLAM3 的前端中加入语义、全景分割、实例标识与零训练运动先验，利用分层兼容性对 ORB 描述符距离进行连续调制，改用软惩罚而非硬拒绝，保持几何后端不变。

**💡 创新点**

创新点在于将上下文证据（语义距离、实例一致性、尺度匹配与无监督运动评分）融合为分层可调的对应成本函数，首次实现统一的、可调的上下文先验对匹配成本的连续调节，并在同一配置下无需重新调参。

**🔧 技术方法**

使用了 Mask2Former、KMaX‑DeepLab 等全景分割模型，深度/视差估计、UniMatch 光流、Longuet‑Higgins 线性运动模型、Huber IRLS、对数平方运动评分与自校准阈值等技术。

**📊 数据集**

实验基于 KITTI Odometry、EuRoC MAV、KITTI Tracking 以及 Virtual KITTI 2 等多域立体视觉基准。

**📈 对比分析**

在同一固定配置下与 ORB‑SLAM3、ORB‑SLAM2、Stereo‑DSO、DynaSLAM、RSO‑SLAM、DROID‑SLAM 等基线对比，平均 per‑sequence ATE RMSE 分别降低 17.4%（KITTI）和 27.7%（EuRoC），动态子集提升 6.6%–31.2%，且无回归。

**⚠️ 局限性**

局限性包括需要离线预计算分割与光流导致实时性受限；性能受分割/光流误差影响；对极小或极慢移动物体的检测不够鲁棒；仅在立体图像上验证，未验证单目或其他几何后端的迁移性。

---

## 303. WhatWorkedBench: Benchmarking Experimental Understanding in AI Agents

**arXiv ID:** 2609.27490 | [PDF](https://arxiv.org/pdf/2609.27490v1)

**作者:** Jingjie Ning `[一作]` (Carnegie Mellon University), Dongting Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 WhatWorkedBench，一个基准框架，用来评估 AI 研究代理在有限实验预算下对程序组件效应的定量预测能力。

**💡 创新点**

创新点在于：①完整的条件效应测量和回归；②通过程序结构提供等价类约束来提升推断；③共享推断协议让同一实验观测下可对比不同数值方法。

**🔧 技术方法**

使用的方法包括主效应岭回归、双效应岭回归、Gaussian Process（GP）回归以及基于代码等价类的约束投影。

**📊 数据集**

实验数据集涵盖 8 个工作流族（分类、回归、聚类、预测、恢复、检索、心律检测、链路预测），共 22 个四因子任务和 14 个六因子任务，包含 352+896 个配置记录。

**📈 对比分析**

比较采用效果恢复（MAE、相对恢复）、配置精度与严格重建等指标，结果显示 GP 在相同观测下能把恢复率从 0.632 提升到 0.698，代码等价约束将六因子 GP 恢复率从 0.248 提升到 0.462，显示共享推断与程序结构约束显著提升性能。

**⚠️ 局限性**

局限性包括：仅处理二进制选项；需事先提供完整的原生参考，无法在线动态适配；对大规模高维配置的可扩展性和非二进制参数的支持仍待改进。

---

## 304. Learning Where to Look: A Shared Relative-Alignment Module for Time-Series Forecasting and PPG-to-Vital-Sign Reconstruction

**arXiv ID:** 2609.27473 | [PDF](https://arxiv.org/pdf/2609.27473v1)

**作者:** Ragamayi Puli `[一作]` (Neurogica Inc), Shunya Nagashima `[通讯]` (Neurogica Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计了ROOSTER，一个可学习周期性组合偏置的交叉注意力模块，用于统一处理PPG到生命体征重建与多变量长时序预测两类任务的条件对齐问题。

**💡 创新点**

通过单一可学习的周期性偏置同时学习零偏移与季节性偏移对应关系，并可输出学习到的周期，实现跨任务迁移与可解释的对齐机制。

**🔧 技术方法**

使用交叉注意力、周期性组合偏置（cosine comb）、条件流匹配、DecompSSM/PENGUIN主机、辅助pinball量化正则以及有效秩恢复等技术。

**📊 数据集**

在PPG重建任务上使用PPG‑DaLiA、WildPPG、BIDMC、WESAD等数据集；在预测任务上使用ETTm1/2、ETTh1/2、PEMS04、Weather、Exchange、Solar等数据集。

**📈 对比分析**

与各自领域的基准模型（PENGUIN、DecompSSM等）在官方评估协议下进行对比；ROOSTER在所有四个重建任务和四个预测基准上取得最高的MSE/MAE，在20/24 dataset–horizon设置下平均MSE低于对照模型。

**⚠️ 局限性**

学习到的周期在对应关系被破坏（如运动干扰）时会漂移且无法在测试时检测；对低频季节性（如日周期在小时级数据）难以解析；模块只能处理由偏移相关的补丁序列，未验证在更广泛的条件生成任务中的泛化。

---

## 305. Hybrid Gaussians for Robust Open-Vocabulary 3D Segmentation with Multi-View Object Association and Boundary Refinement

**arXiv ID:** 2609.27462 | [PDF](https://arxiv.org/pdf/2609.27462v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 306. Issuer-Sovereign Agentic Payments

**arXiv ID:** 2609.27452 | [PDF](https://arxiv.org/pdf/2609.27452v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 307. Latent evolving World Action Model

**arXiv ID:** 2609.27455 | [PDF](https://arxiv.org/pdf/2609.27455v1)

**作者:** Xueji Fang `[一作]` (Zhejiang University), Guo-Jun Qi `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 LeWAM，将世界动作模型与未来嵌入预测结合在 I-JEPA 视觉表示空间，并通过 DemoDPO 进行离线偏好细化，实现高效动作生成。

**💡 创新点**

① 用 I-JEPA 视觉编码替代 VAE+视频扩散模型；② AdaFuse 融合多层特征；③ 通过 DemoDPO 使用演示生成偏好，无需额外交互；④ 通过信息理论分析验证视觉表示对动作生成的重要性。

**🔧 技术方法**

流匹配动作生成、联合未来嵌入预测、AdaFuse 加权融合、多层 I-JEPA 视觉编码、DemoDPO 偏好细化、流匹配 DPO 近似。

**📊 数据集**

RoboTwin 2.0 50 任务仿真数据集（2,500 纯净演示+25,000 随机演示）及真实世界双臂任务（堆叠、折叠毛巾、花卉布置）。

**📈 对比分析**

在与 VDM‑based WAMs、VLA‑JEPA、LaWAM、FastWAM 等基线的同等条件下比较，LeWAM 仅 0.4B 可训练参数、无视频扩散模型，RoboTwin 成功率 92.28%，与最先进模型相当，实测延迟和显存最低。

**⚠️ 局限性**

仍受限于离线演示质量，对极小动作偏差敏感；未在更大规模、复杂多模态任务上验证；需进一步评估长序列鲁棒性。

---

## 308. Stable Neural Decoding Across Sessions via Task-Conditioned Latent Alignment for Brain-Machine Interfaces

**arXiv ID:** 2609.27441 | [PDF](https://arxiv.org/pdf/2609.27441v1)

**作者:** Canyang Zhao `[一作]` (Chinese Academy of Sciences), Bing Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 309. SatUnreal: A High-Precision Synthetic Dataset for Satellite Stereo Matching via Unreal Engine

**arXiv ID:** 2609.27442 | [PDF](https://arxiv.org/pdf/2609.27442v1)

**作者:** Han-Gyeol Kim `[一作]` (TelePIX), Darongsae Kwon `[通讯]` (TelePIX)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并创建了SatUnreal，一个高精度的合成卫星立体图像数据集，解决了真实数据的时空不匹配和遮挡标注不准问题。

**💡 创新点**

创新点在于基于Unreal Engine的物理几何模拟、可变基线和方位角的多视角生成、精确的两步线追踪遮挡标注，以及大规模、时空一致的高分辨率立体图像。

**🔧 技术方法**

采用Unreal Engine仿真管线、基线缩放与旋转、重投影算子生成视差、两步线追踪算法提取遮挡掩模，以及深度学习立体匹配模型如Selective-IGEV、RAFT-Stereo、DLNR。

**📊 数据集**

用SatUnreal自身的10,000对立体图像作为训练集，并与真实数据集US3D和WHU-Stereo进行零样本转移评估。

**📈 对比分析**

通过将SatUnreal训练的模型在US3D、WHU-Stereo上进行零样本测试，并用EPE、1px/3px误差率比较，结果显示SatUnreal模型在真实数据上往往与或优于使用真实数据训练的模型，证明了其良好的跨域泛化。

**⚠️ 局限性**

限制包括：合成环境的光照和大气散射仍有与真实卫星图像的域差；仅在零样本设置下评估，未考虑真实数据微调；以及对高容量网络的细节优化可能受限于合成纹理多样性的不足。

---

## 310. Extracting CNNs in the Unknown-Architecture and Feedback-Agnostic Setting

**arXiv ID:** 2609.27427 | [PDF](https://arxiv.org/pdf/2609.27427v1)

**作者:** Jiashuo Liu `[一作]` (Information Engineering University), Shaozhen Chen `[通讯]` (Information Engineering University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在不知网络架构的情况下，对CNN进行完整的加密分析提取，包括架构和参数的恢复；

**💡 创新点**

首次证明通过已恢复权重向量的空间几何结构即可推断卷积层的所有超参数，且攻击与反馈无关；

**🔧 技术方法**

基于已有参数恢复框架的点搜索与线性方程求解，结合稀疏性、数值一致性与池化结构的几何分析；

**📊 数据集**

主要在MNIST（32×32）及自定义多样化CNN上进行实验；

**📈 对比分析**

与现有CNN参数恢复攻击结合，实验显示架构恢复成功率100%，附加查询成本仅为常数级，整体精度与原始攻击相当；

**⚠️ 局限性**

仅适用于ReLU激活且假设模型使用64位浮点计算，对极深或极其复杂结构的可扩展性待验证。

---

## 311. EBRL: Asynchronous Embodied RL by Multi-Grained Resource Management

**arXiv ID:** 2609.27547 | [PDF](https://arxiv.org/pdf/2609.27547v1)

**作者:** Liang Mi `[一作]` (Nanjing University), Ting Cao `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

开发了一套异步流水线化的具身强化学习训练系统，能够在CPU和GPU之间细粒度分配资源并在不同阶段（模拟、生成、训练）之间实现真正的重叠；

**💡 创新点**

创新点在于：①结合异步流水线调度器和精细化资源管理器，既消除了 rollout 过程中的同步瓶颈，又实现了 GPU SM 与 CPU 核心的动态分配；②使用离线性能剖面配合运行时反馈来自动调整资源份额和批量大小，使资源利用率显著提升；

**🔧 技术方法**

核心技术包括：异步流水线调度（stage‑level 与 step‑level 级别的无阻塞执行）、GPU‑CPU 资源池与弹性分配器、MPS/MIG 基础的 GPU 子设备划分、CPU 核心亲和性绑定、以及训练时的权重推送与梯度同步；

**📊 数据集**

实验使用了四种具身策略（OpenPI、GR00T、OpenVLA‑OFT、DreamZero）与四种模拟器（LIBERO、RoboCasa、ManiSkill、BEHAVIOR），基准算法为 PPO 与 GRPO；

**📈 对比分析**

与 RLinf 的 colocated、hybrid、async 三种执行模式对比，系统在所有硬件平台上实现了 1.30×–3.47× 的轨迹吞吐量提升，训练时间缩短约 2.5×，且最终成功率保持约 98%；

**⚠️ 局限性**

限制包括：需要离线剖面与手动配置 MPS/MIG；在资源需求剧烈变化时分配调整仍有滞后；对极端尾部延迟的处理不够彻底；目前仅在所选策略/模拟器上验证，泛化到更大规模或不同任务仍需进一步研究。

---

## 312. ProCredit: From Outcome Rewards to Progress Credit in Agentic Reinforcement Learning

**arXiv ID:** 2609.27532 | [PDF](https://arxiv.org/pdf/2609.27532v1)

**作者:** Ming Ma `[一作]` (Chinese Academy of Sciences), Steven Hoi `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在长周期代理任务中，ProCredit通过在每一步重新运行任务的可执行检查来获取可验证的进度，并将这一进度变化作为步骤级奖励，用于轨迹级和步骤级的信用分配。

**💡 创新点**

创新点在于将任务本身的接受检查视为可验证的中间信号，既无需额外的奖励模型，也不依赖于最终奖励；同时把进度的信用直接归属到产生该进度的步骤，而不是统一归属于整条轨迹，从而在所有失败组中仍能获得梯度。

**🔧 技术方法**

技术包括基于GRPO的组相对策略优化框架，加入进度奖励 r_t = c(Φ_t-Φ_{t-1})，使用 γ=1 的未折扣返回来计算步骤级优势，并将轨迹级优势与步骤级优势相加以更新策略。

**📊 数据集**

使用的数据集为AppWorld（90训练任务、168/417测试任务）和ToolSandbox（52训练任务、21测试任务），两者均提供可重跑的验收检查。

**📈 对比分析**

与GRPO、RLOO、DAPO、SALT、GiGPO等方法对比，ProCredit在所有模型规模（4B、9B、35B）和两组测试集上均获得最高的任务完成率，4B规模上比最强的基线提升约4.1个百分点；在ToolSandbox中同样表现最佳。

**⚠️ 局限性**

局限性包括：需要环境提供可执行的接受检查且检查数量足以区分不同步骤；在模型规模增大时所有失败组减少，进度信号对提升的贡献随之降低；若检查不够细粒度或环境无法重新运行检查，ProCredit不可直接应用。

---

## 313. Block Erasure Channel and Block z-Channel with Bounded Decoders and Finite Blocklength

**arXiv ID:** 2609.27519 | [PDF](https://arxiv.org/pdf/2609.27519v1)

**作者:** Bin Han `[一作]` (Rptu University Kaiserslautern Landau), Hans D. Schotten `[通讯]` (Rptu University Kaiserslautern Landau)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究有限码长（FBL）条件下，采用误差上限编码（bounded‑distance decoder）对AWGN信道进行几何球面打包分析，给出块混淆概率（blcp）与块擦除概率（blep）的严格上界与下界，并将此模型推广至空闲传输块，证明误码率可近似为块擦除信道或块Z信道。

**💡 创新点**

创新点包括：①将块混淆与块擦除分离，提供精确的概率上界与下界；②用几何球面包络与Chernoff近似统一描述误差概率；③在有限码长与信噪比变化时分析误差概率的敏感性；④证明在空闲块情形下误警概率极低，验证块Z信道的合理性；⑤结合具体BCH、Polar、LDPC码实际数值验证理论。

**🔧 技术方法**

主要技术：几何球面打包与非中心χ²分布；Chernoff大偏差上界；Hoeffding不等式；最小/最大距离界与Plotkin、Singleton、Hamming界；随机码与结构码的距离谱分析；数值模拟与精确概率计算。

**📊 数据集**

未使用公开数据集，所有验证均基于仿真与理论计算（如BCH(15,7)、BCH(31,16)、Polar(32,16)、LDPC(32,16)等标准码）。

**📈 对比分析**

比较方法：将得到的blcp/blep上界与理论下界、以及实际码的精确误差概率进行对比；在不同块长、信噪比与码率下绘制误差概率与熵/信道容量关系图。性能上，所有考虑的配置下blcp与blep均远低于目标误码率ε（如0.05），且误警概率更是多阶降至ε以下，验证块擦除/块Z信道假设。

**⚠️ 局限性**

局限性：仅适用于无列表输出、误差上限的ML解码器；假设符号级映射与等能量常数包络星座；对非均匀码字先验、极化或迭代解码的细节处理缺乏完整理论；对衰落信道的推广仍待研究。

---

## 314. Geometry-Based Metrics for Early-Stage Hull-Form Producibility Screening

**arXiv ID:** 2609.27544 | [PDF](https://arxiv.org/pdf/2609.27544v1)

**作者:** Andrea Serani `[一作]` (National Research Council-Institute of Marine Engineering), Kevin Maki `[通讯]` (University of Michigan)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一套面向船体外表面可制造性早期筛选的表征框架，定义了基于几何的可制造性签名，包括总双曲率偏差、符号分量、曲率类别面积比例、分布场、有效性与表示来源等指标。

**💡 创新点**

创新点在于：① 将双曲率强度、符号与面积分布统一为可解释的签名；② 引入表示感知的数值实现与指标专属有效性约定，区分 CAD BRep 与三角网格两种后端；③ 提供完整的有效性与来源记录，避免单一无校准的可制造性评分。

**🔧 技术方法**

使用了差分几何理论（主曲率、Gauss-Bonnet）、参数域高阶 Gauss-Legendre 积分、鲁棒曲率重构（Rusinkiewicz 方法）以及自适应求积与错误评估机制；同时实现了可视化分布字段与有效性标注。

**📊 数据集**

数据集包含四个公开船体模型（DTMB 5415、KCS、JBC、KVLCC2M），以及平面、球面、圆柱、马鞍、环面和半 Wigley 控制几何，用于验证与对比；网格解析使用 OpenCascade 转换后得到的细化 STL。

**📈 对比分析**

通过解析/半解析控制、同面 BRep 与网格对比、网格分辨率变化、以及 BRep 与 STL 的表示差异进行比较。结果显示：解析精度可达 10^-6 级；网格误差随细化减小；不同表示在积分结果、曲率强度和面积分布上存在显著差异，需按有效性记录解释。

**⚠️ 局限性**

局限性：仅处理外表面几何，不包含板材厚度、材料、铺板、成型路线等实际工艺因素；缺乏实验或工厂数据的校准，无法直接预测成本或工时；对 CAD 表示质量和网格分辨率高度敏感，若输入质量差会导致有效性受限。

---

## 315. A Modular Dual-Arm Robotic Cell for Disassembly and Repair of Industrial Control Electronics

**arXiv ID:** 2609.27466 | [PDF](https://arxiv.org/pdf/2609.27466v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 316. Spiking Neural Network Predicting Sequence of the External Worlds States in Model-Based Reinforcement Learning

**arXiv ID:** 2609.27459 | [PDF](https://arxiv.org/pdf/2609.27459v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 317. Control-Token Injection Suppresses Chain-of-Thought and Defeats Reasoning-Based Oversight in Tool-Using Agents

**arXiv ID:** 2609.27542 | [PDF](https://arxiv.org/pdf/2609.27542v1)

**作者:** Muhammad Usama `[一作]` (Braindeck Inc), Summer Yeoreum Jung `[通讯]` (Braindeck Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过控制注入一串模型自身的控制标记，展示了在工具调用语言模型代理中能够完全删除推理（Chain‑of‑Thought）轨迹，而不影响最终的工具调用；同时证明工具调用解析器的宽容性决定了调用是否被执行，两个安全问题均可从不可信输入中触发。

**💡 创新点**

创新点在于：1）揭示推理轨迹的存在并非模型固有属性，而是模型加上解码环境（chat 模板与解析器）的联合属性；2）提供了可复现的、全精度的实验验证，量化了推理轨迹被删除后对 CoT 监控的破坏效果；3）系统性对工具调用解析器宽容性进行评估，并指出不同实现可导致安全结果相反。

**🔧 技术方法**

使用的技术主要包括：a）在用户消息中插入模型控制标记字符串；b）利用已公开的工具调用语言模型代理（如 Gemma、Qwen3、Qwen3-4B-Thinking 等）与其官方 sandbox 进行对齐；c）对比两种解析器（正则式宽容解析器 vs. 严格解析器）对工具调用的影响；d）实现简单的输入净化（删除/替换控制标记）和解析器硬化（去掉末尾空字符匹配）。

**📊 数据集**

实验使用了公开的 10+ 任务集合，包含：① 40 个“正常”外泄任务（Plain‑Framing）；② 48 个“恶意”拒绝任务（Malicious‑Framing）；③ 15 个安全的普通任务（Benign）。此外，对 Qwen3-4B-Thinking 与 DeepSeek‑R1‑Distill‑7B 等模型进行泛化实验。

**📈 对比分析**

评估方法：在无防护、删除式防护与完整防护三种配置下，记录推理轨迹长度、工具调用是否被触发、监控检测率。实验结果显示：在无防护下，推理轨迹平均 52.5 个 token；注入后归零；CoT 监控从 100% 变为 0%；在 39.6% 的拒绝请求中，攻击成功将拒绝转为外泄。解析器宽容度差异导致 100% 与 0% 的工具调用成功率。

**⚠️ 局限性**

limitations：1）实验仅针对公开的 sandbox 与特定模型，其他模型或非 greedy 解码、不同 prompt 可能产生不同的成功率；2）泛化研究仅覆盖两款额外模型，未覆盖更广泛的推理模型族；3）拒绝绕过效果依赖具体任务 framing，未证明在所有拒绝场景均有效；4）解析器硬化会导致合法但被截断的调用被丢弃，存在可靠性成本；5）未对通过工具返回等间接注入路径的完整攻击链进行端到端验证。

---

## 318. Behavior-Aligned Action Tokenization for Robot Policy Learning

**arXiv ID:** 2609.27513 | [PDF](https://arxiv.org/pdf/2609.27513v1)

**作者:** Junbo Dong `[一作]` (Southern University of Science and Technology), Zhaoyuan Ma `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Behavior-Aligned Action Tokenization（BAAT）框架，通过对演示动作块进行软动态时间规整（Soft‑DTW）匹配并在量化坐标上施加对应关系，使不同任务中的相似局部动作聚集到相邻的离散表示；随后使用历史条件扩散解码器重建连续动作，并训练自回归策略预测这些离散标签。

**💡 创新点**

创新点在于：①仅使用动作序列的行为对应作为监督，不依赖任务标签或视频信息；②将Soft‑DTW匹配与量化坐标的对齐相结合，兼顾运动一致性与可执行细节；③通过历史条件扩散解码器实现跨片段连续性，提升策略泛化。

**🔧 技术方法**

主要技术包括：软动态时间规整（Soft‑DTW）进行行为对应选择；FSQ量化+离散标签学习；历史条件扩散解码器用于连续动作重建；自回归策略预测离散标签并映射回连续动作。

**📊 数据集**

使用的基准数据集包括：LIBERO‑10（四套任务）、RoboCasa（厨房操作）、RoboTwin 2.0（双臂操作），以及两套真实机器人任务（牙刷分类与物体分拣）。

**📈 对比分析**

与FAST、OAT、Bin、Diffusion Policy等基线对比，BAAT在LIBERO‑All、RoboCasa、RoboTwin 2.0的平均成功率分别提升至79.0%、29.2%、27.3%（相对OAT提升约7–8%），并在真实机器人任务中获得最高的牙刷任务成功率（70%）与与DP持平的分拣任务成功率。

**⚠️ 局限性**

局限性包括：过强的对齐会降低重建质量与重放成功率；行为对应选择依赖Soft‑DTW匹配质量；在真实机器人上出现块间跳跃现象，说明对连续性控制仍有改进空间；难以单独评估对齐、量化与解码器各自贡献。

---

## 319. RoboCafé in the Open: Interaction Continuity in Long-Term Public Human-Robot Interaction

**arXiv ID:** 2609.27475 | [PDF](https://arxiv.org/pdf/2609.27475v1)

**作者:** Kaitlynn Taylor Pineda `[一作]` (Johns Hopkins University), Chien-Ming Huang `[通讯]` (Johns Hopkins University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文开发并部署了一个具备LLM对话功能的自动化咖啡服务机器人，在大学建筑内连续运营12天，完成148个订单并收集了多模态交互数据。

**💡 创新点**

创新点在于首次提出四项针对长期公共人机交互连续性的系统设计要求：情境交互状态、持久人物定位、明确交互生命周期管理与交互可观察性。

**🔧 技术方法**

技术实现结合ROS1、Franka机械臂、GPT‑4o‑mini与Llama‑3.1、Google Cloud Speech/TT、Azure Kinect、MediaPipe、OpenCV 与 DeepFace 等多种感知与对话模块。

**📊 数据集**

数据集由12天的部署日志组成，包括148个订单的订单表、JSONL形式的对话与感知日志、音视频录制及每位用户的摘要文件。

**📈 对比分析**

通过对比社交式与任务式对话条件，采用订单数、转录接受率、参与度等指标评估交互效果，结果显示社交模式下用户参与度显著更高。

**⚠️ 局限性**

局限性包括对人物身份的依赖性不足（仅凭电子邮件且易丢失）、缺乏多方说话者归属、未实现正式交互生命周期管理、且对交互可观察性仅局限于系统内部状态，无法覆盖物理交互的真实一致性。

---

## 320. Implementation and Evaluation of BitNet Inference on a CGLA by Signed-Int4 Instructions

**arXiv ID:** 2609.27453 | [PDF](https://arxiv.org/pdf/2609.27453v1)

**作者:** Takuto Ando `[一作]` (Nara Institute of Science and Technology), Yasuhiko Nakashima `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将BitNet的三元权重-整数8位点积映射到可重用的signed‑int4指令上，利用CGLA实现低功耗加速

**💡 创新点**

创新点在于用signed‑int4指令重用硬件，将int8激活拆分为两段并做稀疏修正，避免为BitNet专门设计数据通路

**🔧 技术方法**

技术涵盖CGLA可编程ASIC、signed‑int4乘加指令、DMA/本地存储、BitNet b1.58模型与C++运行时集成

**📊 数据集**

使用BitNet b1.58 2B‑4T模型的自定义权重与激活数据进行评估，并对多次token生成进行测量

**📈 对比分析**

与CPU/GPU实现比较，CGLA在解码边界下实现2.52 tokens/s、0.258 J/ token，能耗最低，速度略低于GPU 406.5 tokens/s

**⚠️ 局限性**

局限在于仅对单个offload调用评估、频率缩放假设、主机耗时占比高、未完成全模型集成和完整精度评估

---

## 321. BEE: Intervention-Adaptive Real-World Reinforcement Learning with Vision-Language-Action Models

**arXiv ID:** 2609.27450 | [PDF](https://arxiv.org/pdf/2609.27450v1)

**作者:** Weihui Zhao `[一作]` (South China University of Technology), Maoqing Yao `[通讯]` (AgiBot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于视觉语言动作(VLA)模型的实时强化学习框架，利用人类对VLA建议的修正来构建维度级的不确定性约束，使得策略能够超越专家模仿。

**💡 创新点**

创新点在于将人类纠正建模为基于VLA提议的残差高斯分布，并通过Mahalanobis距离形成维度级的约束，依据每个动作维度的预测不确定性自适应地限制策略更新；同时保留冻结的VLA提议作为行为先验。

**🔧 技术方法**

采用了Correction Model（Gaussian 残差模型）、残差策略、Actor‑Critic（TD3/离线）以及状态依赖的拉格朗日乘子双优化，结合提议-提取器的VLA-RL接口。

**📊 数据集**

使用真实机器人三项任务（手机充电、零食悬挂、布料对齐）和LIBERO‑Pro仿真放置碗；预先收集20集人类修正作为演示，再通过在线交互收集数据。

**📈 对比分析**

与Base Policy、RLT、DSRL、DAgger、SiLRI、HIL‑SERL 等基线对比；在每个任务上均获得最高成功率（平均91.2% vs 57.5% RLT, 42.1% DSRL），且人类干预率最低，显示出优越的样本效率和性能。

**⚠️ 局限性**

局限性包括：预测不确定性在纠正稀疏时可靠性不足；仅针对单阶段任务，未扩展到多阶段、多操作员或不同机器人；需要冻结VLA提议，限制了对VLA本身的进一步微调。

---

## 322. Behaviora - A Conceptual Architecture for External and Internal Behavior of Robots and Agents

**arXiv ID:** 2609.27536 | [PDF](https://arxiv.org/pdf/2609.27536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 323. A Bulletproof Business? Towards Detecting Infrastructure-as-a-Service Offerings on Telegram

**arXiv ID:** 2609.27428 | [PDF](https://arxiv.org/pdf/2609.27428v1)

**作者:** Roy Ricaldi `[一作]` (Eindhoven University of Technology), Irdin Pekaric `[通讯]` (University of Liechtenstein)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并应用了针对Telegram的基础设施即服务（IaaS）分类体系，自动检测并量化1,116,071条信息中广告的基础设施与信任属性。

**💡 创新点**

创新点在于将传统的CaaS分类细化为计算、网络、通信三大能力维度，并首次加入“Bulletproof”等信任信号，提供了针对Telegram广告的专属标注框架；同时将该框架与大规模自动分类相结合，揭示了广告高度集中与信任信号的分布模式。

**🔧 技术方法**

技术上采用多标签文本分类，主要使用TF‑IDF特征与一对多逻辑回归模型；对比了关键词字典、TF‑IDF+LR以及基于LLM的提示式分类；实现了Python/Scikit‑Learn等工具。

**📊 数据集**

使用了从前期研究收集的1,116,071条Telegram消息（167个社区，覆盖2023-2025年），其中11个专注于数字基础设施的子集用于标注与模型训练。

**📈 对比分析**

比较结果显示，TF‑IDF+LR在微观F1、精度与精确匹配率上优于关键词和LLM提示方法；其在测试集上微观F1达0.898，宏观F1为0.716，提示式分类表现最差。

**⚠️ 局限性**

局限性包括：模型仅覆盖六大基础设施类别，其他类别缺失；训练集来自少数社区，存在领域漂移；标注与预测多标签时存在语义重叠导致误判；未验证社区层面的直接分类，且未深入确认广告与实际基础设施所有权的对应关系。

---

## 324. Invisible in Space, Visible in Time: Motion Vision CAPTCHA against GUI Agents

**arXiv ID:** 2609.27461 | [PDF](https://arxiv.org/pdf/2609.27461v1)

**作者:** Zeyu Zhang `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种基于运动的验证码框架MVCAP，并构建了浏览器基准MVCAP-Bench及其前景控制版本MVCAP-Bench-FG，用以评估人类与多种LLM驱动的GUI agent的差异。

**💡 创新点**

创新点在于把验证码语义隐藏在动态背景中，使其仅可通过时间上的运动分离得到，且提出三层运动层级（连贯运动、结构运动、生物运动）以充分挖掘人机感知差异。

**🔧 技术方法**

使用了多模态大型语言模型、GUI agent框架、视频生成与渲染技术、时间分割与运动分析、浏览器交互与离线VQA评估等技术手段。

**📊 数据集**

利用自研的600个动视频验证码实例（200个每层）生成MVCAP-Bench与MVCAP-Bench-FG两套数据集，用以进行对比实验。

**📈 对比分析**

通过与人类基线（99.6%）以及多款浏览器/本地使用的LLM agent（Pass@1仅约13–17%）对比，并在控制实验中发现动态背景遮蔽导致Agent性能显著下降；离线VQA实验提升略多但仍低于随机猜测。

**⚠️ 局限性**

主要局限在于当前模型对运动语义的理解仍不足，难以在动态背景下实现有效前景分离；实验仍局限于受控环境，缺乏对更大规模或更复杂攻击手段的评估。

---

## 325. Quantum Reinforcement Learning for Cost and Delay Tradeoffs in Quantum Cloud Orchestration

**arXiv ID:** 2609.27446 | [PDF](https://arxiv.org/pdf/2609.27446v1)

**作者:** An N. H. Phan `[一作]` (University of Information Technology, Vietnam National University), Hoa T. Nguyen `[通讯]` (CSIRO)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为QRLQ的成本-延迟感知量子云调度框架，将参数化量子电路与双重DQN结合，以实现对异构量子资源的动态分配。

**💡 创新点**

创新点在于将参数化量子电路作为紧凑的函数逼近器，显著减少可训练参数，同时通过双重DQN实现对成本和延迟的平衡，首次在统一时基计费模型下应用量子强化学习。

**🔧 技术方法**

采用量子强化学习、参数化量子电路、双重对决深度Q网络、数据重上传、TorchQuantum、QSimPy等技术。

**📊 数据集**

使用MQT Bench库的量子电路和IBM Quantum模拟后端（Fake Backends）生成任务，以及泊松分布的任务到达模拟。

**📈 对比分析**

与多种启发式调度（GEA、GFD、CRR、RUS、GLE）以及经典D3QN、QDQN基线对比，QRLQ在执行时间、等待时间、成本、延迟和保真度方面均优于启发式，成本降低5-11%，延迟缩短17-82%，保真度与最优保真调度差距仅2%，且参数量比经典DRL少72%。

**⚠️ 局限性**

局限性包括仅支持单任务单QPU调度、使用浅量子电路、仅在模拟环境下验证、对噪声敏感、动作空间受限、未在真实量子集群上测试、未考虑多任务并行和迭代量子经典工作负载。

---

## 326. NavProbe: Evidence-Grounded Reasoning with Active Memory Retrieval for Zero-Shot Navigation

**arXiv ID:** 2609.27526 | [PDF](https://arxiv.org/pdf/2609.27526v1)

**作者:** Jingyang Liu `[一作]` (ShanghaiTech University), Lan Xu `[通讯]` (ShanghaiTech University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 NavProbe，一种基于零样本的层次化视觉语言导航代理，能够通过主动检索历史证据来动态修订和验证中间子目标，提升长程导航性能。

**💡 创新点**

将动态子目标议程与图结构化的多模态记忆相结合，并通过检索引擎实时获取视觉与几何证据，随后将检索结论固化为可复用的实体知识，实现在零样本场景下的任务进度推理。

**🔧 技术方法**

采用 GPT‑5.5 进行多模态推理，YOLO‑World 检测关键物体，Graph‑structured Episodic Memory 存储视觉与几何记录，Entity Knowledge Manager 负责检索结论的固化，Skill Policy 负责将修订后的任务状态映射为参数化导航动作。

**📊 数据集**

在 R2R‑CE、RxR‑CE（视觉语言导航）以及 HM3D‑v2（对象目标导航）三个公开基准上进行评估。

**📈 对比分析**

与多种训练无关和训练方法的基线对比，NavProbe 在 R2R‑CE 上取得 71.7% SR、55.8% SPL，RxR‑CE 上取得 55.3% SR、38.6% SPL，HM3D‑v2 上取得 79.3% SR，均超过现有零样本方法。

**⚠️ 局限性**

仍受限于检索预算与推理延迟；当检索不完整或错误时可能导致子目标误判；依赖 GPT‑5.5 等大模型，推理成本较高；在极端动态环境或非结构化图像信息中记忆更新与检索策略的鲁棒性待进一步提升。

---

## 327. M3D-Net: Hierarchical Coordination of Spatial Context, Feature Reuse, and Differential Attention for Mammography Classification

**arXiv ID:** 2609.27523 | [PDF](https://arxiv.org/pdf/2609.27523v1)

**作者:** Zheng Yu `[一作]` (Shenzhen Loop Area Institute), Xiang Li `[通讯]` (Shenzhen Research Institute of Big Data)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种面向乳腺影像的分层协调网络M3D-Net，能够在保留局部细节的同时融合全局组织上下文；

**💡 创新点**

创新点在于三大组件的协同使用——多尺度坐标注意力（MCA）对局部与全局特征加权；多路动态稠密重用（MUDD）在块内检索历史特征；差分注意力（DA）在低分辨率阶段对全局映射做差分抑制；并且通过分辨率感知的层级调度实现资源优化；

**🔧 技术方法**

采用的技术包括：坐标注意力、动态卷积、SRA/DA注意力、Group Normalization、深度可分离卷积、点卷积、可学习门控、以及对输入的条件位置编码；

**📊 数据集**

使用公开的AISSLab乳腺X光影像数据（266张）进行三分类验证，另外使用BrEaST超声影像与22项临床特征进行图像-临床融合二分类实验；

**📈 对比分析**

与EdgeNeXt、RepViT、TransXNet等前沿骨干进行对比；在AISSLab上M3D-Net达到97.78%的验证准确率、97.22%的尾部平均准确率，交叉熵损失仅为0.0542，明显优于其它模型；在BrEaST上适配后M3D-Net取得80.39%准确率、81.76%尾部准确率，交叉熵损失0.4297，整体排名第一；

**⚠️ 局限性**

局限性包括：实验未进行多种随机种子、独立样本或不同划分的复现验证；适配至超声后仍需进一步验证对不同模态的泛化；缺乏对模型校准性和解释性的深入分析；未对真实临床工作流的推理速度与内存占用进行评估；

---

## 328. Not What You Meant: Can LLMs Follow a Specified Negation Semantics?

**arXiv ID:** 2609.27517 | [PDF](https://arxiv.org/pdf/2609.27517v1)

**作者:** Qiming Bao `[一作]` (University of Auckland), Kostas Stathis `[通讯]` (Royal Holloway, University of London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于求解器认证的程序生成器，生成包含默认否定的逻辑程序，并在四种语义视角（SLDNF、WFS、稳定模型可信与可疑）下对每个查询进行标注；随后用多种自然语言框架和规则顺序将程序转化为可读文本，测试大语言模型在给定或未给定否定语义时的推理准确性，并评估模型在不同语义下的默认对应行为。

**💡 创新点**

创新点在于：①构建可控的、可复现的、无人工标注的“Negation Semantics Benchmark”（NAFBench），使语义视角冲突的实例能够得到多重认证标签；②通过“框架”和“顺序”变换检验模型对语义一致性的鲁棒性；③系统地比较前沿模型与开源模型在默认语义与指定语义下的表现，并提出三种缓解策略（代理求解、微调与验证脚本）。

**🔧 技术方法**

使用技术包括：逻辑程序生成器（控制否定深度、宽度、循环结构）、SWI-Prolog、Clingo 和 Python Well-Founded Solver 进行语义求解；多模自然语言模板化与语义不变变换；LLM推理与回答抽取（正则解析）；统计与置信区间评估。

**📊 数据集**

数据集为自生成的逻辑程序集合，包含 120 个主评估程序（4 个语义视角 × 30 程序）以及扩展规模的 74 程序、44 程序的 Well-Founded 评估等；所有实例均经过求解器认证，无人工注释。

**📈 对比分析**

比较方法：在三种实验条件（未命名语义、指定语义与控制实例、指定语义与分歧实例）下，分别计数模型回答正确的比例，计算默认对应率与错误对齐率；利用 Fisher 精确检验、Mann-Whitney U 检验和聚类自举置信区间。实验结果显示：前沿模型在所有语义上均可达到 100% 以上准确率；开源模型仅能达到 31–74% 的准确率，且在分歧实例上表现急剧下降；三种缓解策略显著提升开源模型性能，尤其是代理求解与微调后可接近前沿水平。

**⚠️ 局限性**

局限性包括：①实例为合成程序，缺乏对真实法律、医疗、监管规则书的验证；②评估聚焦单一程序复杂度层级，未覆盖更深、交叉循环或其他否定形式；③回溯分析依赖模式匹配，可能漏检复杂推理轨迹；④微调与验证脚本的泛化能力仍待在更多场景中检验。

---

## 329. Backstitch: Restoring Request Causality Across a Production Microservice Fleet

**arXiv ID:** 2609.27538 | [PDF](https://arxiv.org/pdf/2609.27538v1)

**作者:** Ziyue Dang `[一作]` (TikTok Inc.), Guangming Luo `[通讯]` (Douyin Vision Co., Ltd.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种专用的代理系统，用生产中观测到的上下文传播中断作为修复循环的起点，自动定位错误 handoff、生成最小化代码修复并通过回放验证；

**💡 创新点**

创新点在于把破损的执行记录作为“活”参考，结合受限的自动化工具、知识库指导和运行时回放，形成从症状到可验证修复的闭环；

**🔧 技术方法**

技术包括：分布式追踪、上下文传播分析、代码路径回溯、专用知识库匹配、LLM 编码代理、受限式重放验证、跨仓库修复和多轮集群级调度；

**📊 数据集**

使用了真实大规模短视频平台的微服务数据集，覆盖1133个服务、26,136个调用元组、1.42 B QPS，且基准测试覆盖14个真实服务的78个已确认中断点；

**📈 对比分析**

与基线（无专用系统）相比，系统实现了所有78个目标修复，90%+ 的修复与人工修复位置一致，部署后整体可操作中断率从20.12%降至8.55%，单服务中断率从90.46%降至4.69%，验证环节可将错误回放至1%以下；

**⚠️ 局限性**

局限性包括：仍需人工审核和发布审批、对知识库和修复模式的依赖、对极其复杂或新型 handoff 的识别不完整、以及对运行时上下文属性（如截止期限、取消等）细粒度控制的支持不足。

---

## 330. X2Real: an eXtensive simulation benchmark for real-world generalist policies

**arXiv ID:** 2609.27449 | [PDF](https://arxiv.org/pdf/2609.27449v1)

**作者:** Lian Ruan `[一作]`, Qian Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出X2Real仿真基准，构建多维任务体系，严格区分训练与评估，利用全轴随机化和物理校准提升对真实世界的对应性。

**💡 创新点**

三大设计原则（真实性、多样性、公平性）结合Physical DSL与Mana平台实现agent驱动的环境构建；通过视觉与动力学双重校准实现高sim2real相关性。

**🔧 技术方法**

使用Nvidia Isaac Lab-Arena、Python/C++实现的Mana系统、Physical DSL、CMA‑ES参数优化、Policy Space统一接口、异步物理渲染、低级PD控制及多轴随机化等技术。

**📊 数据集**

近300小时的演示轨迹（ArtiXon Arm‑6A、Quanta X1、Franka、ARX R5等）及ManipArena与X2Real自研任务数据，用于训练与评估。

**📈 对比分析**

对四个通用策略（Wall‑OSS‑0.5、DreamZero、π_0.5、Wall‑x‑preview）在ID与OOD下测算成功率与进度分数；ID整体成功率约54%→34% OOD，进度分数下降；在仿真‑真实对照中成功率相关系数0.74、进度分数0.84。

**⚠️ 局限性**

仅支持刚体与关节物体，缺乏变形、流体；任务量仍有限，缺乏完全自动化生成；未包含触觉与复杂手抓；评估相对静态，缺少对抗性与竞争性测评。

---

## 331. Uncheatable Eval: Dynamic Compression-Based Evaluation of Language Models

**arXiv ID:** 2609.27510 | [PDF](https://arxiv.org/pdf/2609.27510v1)

**作者:** Kaifeng Tan `[一作]` (Shenzhen University), Linlin Shen `[通讯]` (Shenzhen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Uncheatable Eval 动态基准，利用新近发布文本的无损压缩率评估基础语言模型的预测能力。

**💡 创新点**

创新点在于把模型的下一个词概率转化为理想编码长度，构建无任务指令、低污染风险的压缩评估指标，并且持续更新评测数据。

**🔧 技术方法**

采用算术编码将概率映射为代码长度，进行长上下文字节级压缩、功率律规模分析、与 MMLU 的相关性研究，并使用多轮数据清洗、去重与多词器兼容的处理流程。

**📊 数据集**

使用 2026 年 7 月收集的 7,000 篇新发布文本，覆盖小说、新闻、百科、科学论文、源代码等 14 类不同领域文本。

**📈 对比分析**

通过压缩率（CR）比较 80 种模型，发现 CR 随模型规模下降，形成 CR(P)=3.2P⁻⁰·²⁹+5.33 的规模规律；注意力、混合与循环模型在长上下文下的 CR 下降速率不同；低 CR 与零样本 MMLU 准确率高度负相关（ρ≈-0.88）。

**⚠️ 局限性**

局限在于新文本仍可能被污染，评估仅适用于基础模型且不衡量指令遵循能力；需定期刷新数据，且只能评测原始文本预测，无法直接衡量任务性能。

---

## 332. Information Capacity of Generative Video Compression: Quantifying the Rate-Compute Exchange at Identical Quality

**arXiv ID:** 2609.27493 | [PDF](https://arxiv.org/pdf/2609.27493v1)

**作者:** Cheng Yuan `[一作]`, Xuelong Li `[通讯]` (China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并量化了生成视频压缩中计算与带宽的互换关系，定义信息容量（IC）并通过两因子幂律模型拟合 DISTS 质量表面。

**💡 创新点**

创新点在于给出计算对率的弹性度量（IC）以及其在 (R,C) 平面上的场景化展示，能够比较不同规模解码器在不同数据集上的计算-率折算效率。

**🔧 技术方法**

使用 DISTS 作为感知质量指标，拟合两因子幂律 Q(R,C)=a(R/R0)^-b + c(C/C0)^-d + e，计算 IC= -∂ln R/∂ln C，并用插值法与解析式两种方式求取。

**📊 数据集**

使用五个视频数据集：HOIGen‑1M、MCL‑JCV、SA‑V、TemporalBench 与 Sintel，对两种解码器（1.3B 与 14B）进行测评。

**📈 对比分析**

通过在 (R,C) 平面上绘制 IC 场和 iso‑quality 等高线，并在网格中心计算 IC，发现 14B 解码器的 IC 大约是 1.3B 的 6–33 倍，且在中心点对比时 14B 能以 27–94% 的率节省率，1.3B 仅能节省 3–37%。

**⚠️ 局限性**

局限在于计算量仅用理论 DiT FLOPs 表示，未考虑内存、延迟、能耗等实际硬件成本；IC 只基于 DISTS 评价，若换用其他质量指标结果可能不同。

---

## 333. CerebroSim: Scalable Whole-Brain Simulator at 100-Trillion-Synapse Scale on the LineShine Supercomputer

**arXiv ID:** 2609.27482 | [PDF](https://arxiv.org/pdf/2609.27482v1)

**作者:** Guangnan Feng `[一作]` (Sun Yat-sen University), Yutong Lu `[通讯]` (Sun Yat-sen University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `70e40602-aae3-44bd-80ec-4a7f2674330f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在LineShine超算上设计并实现了可扩展的全脑模拟框架CerebroSim，模拟了86亿神经元和100万亿突触。

**💡 创新点**

创新点包括：①延迟感知Spike广播（DSB）将稀疏、延迟受限的通信结构化；②无竞争多线程突触动力学计算（RSDC）消除锁/原子，提升并行效率；③稀疏突触存储压缩（3SC）通过索引压缩与实时突触再生成，显著降低内存占用。

**🔧 技术方法**

使用了延迟调度、虚拟拓扑聚合、HBM预取优化、Verbs/RDMA通道、SVE/SME向量/矩阵指令、随机数生成器、压缩索引SACI以及CPU/GPU协同等技术。

**📊 数据集**

基于磁共振成像（结构MRI）和扩散张量成像（DTI）构建的全脑解剖与连接矩阵作为数据集。

**📈 对比分析**

与现有全脑模拟（如Fugaku 9M/26B、3.5k节点模型等）对比，在18,432节点、11.2M核上实现24.44 PFLOP/s，弱缩放效率91%（核心阶段）强缩放效率94%，压缩后内存占用仅为21.8%。

**⚠️ 局限性**

主要局限包括：缺乏高分辨率的实时神经记录数据；模型未加入突触可塑性；通信仍受节点数和延迟限制；在全脑规模下强缩放受通信瓶颈影响。

---

## 334. DeltaS: Reading the Gated Linear Attention State for KV Cache Eviction in Streaming Video

**arXiv ID:** 2609.27470 | [PDF](https://arxiv.org/pdf/2609.27470v1)

**作者:** Taeyoun Kwon `[一作]` (Maum AI Inc), Moon Hwan Kim `[通讯]` (Maum AI Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于状态漂移的 KV 缓存淘汰方法 DeltaS，利用混合线性/全注意力模型中的递归状态变化来决定哪些视频块需要保留。

**💡 创新点**

创新点在于：①在不需要查询、无需训练的前提下，直接使用已有的递归状态变更（状态漂移）作为淘汰评分；②通过时间桶化减少早期偏倚；③在同一预算下实现了比现有所有查询无关、训练无关方法更优的性能。

**🔧 技术方法**

核心技术包括：Qwen3.5-9B 混合注意力架构、门控 Delta 线性注意力的状态更新公式、状态漂移归一化评分、KV 缓存分桶保留策略、问答后状态恢复。

**📊 数据集**

使用六个长视频理解基准：MLVU、Video‑MME、Video‑MME v2、LongVideoBench、LVBench、EgoSchema。

**📈 对比分析**

与 InfiniPot‑V、StreamMem、HERMES 等无查询、无训练、固定内存预算方法对比，DeltaS 在 8,192‑token 预算下平均提升 2.1 分，16,384‑token 预算下平均提升 1.7 分；在最长的 LVBench 上提升 5.6 分。受限预算时优势更显著；在受控实验中，状态漂移评分显著优于位置、注意力、键/值等传统评分。

**⚠️ 局限性**

局限性：仅在单一混合 Backbone（Qwen3.5‑9B）和门控 Delta 更新规则下验证；不适用于其他状态更新机制；每块共享单一评分，无法在块内进行细粒度区分；仍属于查询无关、训练无关的范围。

---

## 335. CereVLA: Cerebellum-Inspired Consequence-Aware Residual Governance for Efficient Vision-Language-Action Execution

**arXiv ID:** 2609.27468 | [PDF](https://arxiv.org/pdf/2609.27468v1)

**作者:** Shuai Zeng `[一作]` (Hong Kong University of Science and Technology), Hang Zhao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为CereVLA的残差治理框架，用于在冻结的视觉-语言-动作（VLA）策略中，先生成轻量级残差修正，再通过预测后果评估决定是否执行该修正，从而提升执行效率和可靠性。

**💡 创新点**

核心创新在于将残差生成与后果预测分离，结合一阶和多步后果评估与治理器，实现对残差是否执行的智能决策；并借鉴小脑预测运动后果的机制。

**🔧 技术方法**

使用流式残差细化（FRR）、递归状态空间模型（RSSM）进行后果预测、历史感知的多步后果分类器（H_K）、以及基于阈值的治理器（governor），在冻结VLA政策之上添加这些模块。

**📊 数据集**

在模拟库LIBERO-10与LIBERO-GOAL数据集上进行评估，并在真实SO-101机器人上进行物理实验，使用预训练的π_0.5和SmolVLA两种VLA后端。

**📈 对比分析**

与基线（冻结策略）、频繁重规划、仅残差适配、VLA-Corrector及FutureRTC等方法对比，CereVLA在两套任务中均取得最高平均成功率（π_0.5为95.35%，SmolVLA为64.00%），并在恢复–损害平衡上优于单纯残差策略，实验证明其能显著提升成功率、减少执行步数和运动抖动。

**⚠️ 局限性**

局限性包括：治理器仅基于短期后果预测，可能忽略更远期的非线性效应；对模型参数与阈值的敏感度未充分探究；在非冻结的动态环境下性能尚未验证。

---

## 336. Sampling Line-Graph Colorings with Constant Extra Colors

**arXiv ID:** 2609.27440 | [PDF](https://arxiv.org/pdf/2609.27440v1)

**作者:** Alireza Haqi `[一作]` `[通讯]` (Stanford University), Alireza Haqi (Stanford University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明：在最大度数为 Δ 的线图上，单点 Glauber 动态在 q ≥ Δ+5 的颜色数下，收敛时间为 O_Δ(n log(n/ε))，并给出相应的谱间隙与熵上界。

**💡 创新点**

创新点在于：首次将 Bochner 方法与 Chen‑Liu 的投影框架结合，得到离散 Bochner 恒等式；利用 Anari 等人的谱独立性通用定理，将谱间隙转化为谱独立性；再结合 Chen‑Liu‑Vigoda 的熵定理，完成从谱间隙到快速混合的完整证明。

**🔧 技术方法**

主要技术包括：离散 Bochner 识别、投影框架下的交叉项估计、谱间隙与影响矩阵的比较、谱独立性定理、熵与 Log‑Sobolev 估计。

**📊 数据集**

本研究为理论分析，未使用任何实验数据集。

**📈 对比分析**

与之前的结果相比（如 Δ+O(Δ/lnΔ) 的阈值），本文将阈值降低到 Δ+5，提供了更紧的收敛时间上界，并在整个 q 范围内保持 O_Δ(n log(n/ε)) 的高效混合。

**⚠️ 局限性**

局限性：阈值仍未达到 conjectured Δ+2；常数 C_Δ 的增长取决于 Δ，当前给出的是双指数上界，实际最优常数可能更小。

---

## 337. Mamba-Family State-Space Model Kernels on a Programmable CGLA

**arXiv ID:** 2609.27437 | [PDF](https://arxiv.org/pdf/2609.27437v1)

**作者:** Takuto Ando `[一作]` (Nara Institute of Science and Technology), Yasuhiko Nakashima `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在可编程 CPU‑Grounded Linear Array（CGLA）平台 IMAX 上实现并测评了 Mamba‑family 状态空间模型（SSM）的核心计算核（密集投影、SSD Step‑1、递归状态更新）以及完整的自回归解码流程，探究不同核在此平台上的适配性和瓶颈。

**💡 创新点**

创新点在于：
- 将 Mamba 的三大计算形态映射到通用 CGLA 的 64‑stage 线性流水线；
- 通过统一的基准框架对投影、SSD Step‑1 与递归更新三类核进行细粒度性能分解；
- 结合单核 FP32 测试和完整 token‑level 生成，首次从单核硬件角度揭示投影 GEMV 为解码瓶颈，并给出可行的优化方向。

**🔧 技术方法**

使用技术包括：
- 可编程 CGLA（IMAX）架构；
- 采用 conv‑c2d 编译器将 GEMM、Hadamard、SMEM 复用映射到 64‑stage 线性流水线；
- 通过 DMA 与 ARM Cortex‑A72 主机协同完成权重加载、边界传输与结果回写；
- 采用 FPGA 原型（AMD Versal）与 ASIC 合成估算对比；
- 统一的实验脚本和验证机制（与 ARM 软件参考对比）。

**📊 数据集**

使用的模型数据集：Mamba‑370M（d_model=1024、d_inner=2048、N=128、Q=64）和 Mamba‑130M（用于 token‑level 解码），并采用其公开权重；未对训练数据集或文本语料进行评测，侧重推理性能。

**📈 对比分析**

比较方法：
- 对比 IMAX FPGA 的 EXEC 时延与 DMA‑包含总时延；
- 与同一模型在 ARM Cortex‑A72 软件实现的 FLOP 计数和执行时延做基准；
- 在单核 IMAX 上完成 200‑token 的 greedy 解码，测得 0.345 tokens/s，显示投影 GEMV 为主要瓶颈。性能表现：投影 GEMM 在长归约下达到 1.43× 的算子级加速，但整体时延受 DMA 及边界转移限制；SSD Step‑1 受短归约与核边界影响；递归更新由于规模小无法充分利用流水线。

**⚠️ 局限性**

局限性：
- 仅使用 FP32，未评估 FP16/BF16/量化路径；
- 单通道 64‑stage 流水线，未覆盖多通道或更宽的 PE 组合；
- DMA 与边界传输成为主要瓶颈，未实现持久化权重或融合核；
- 仅在 FPGA 原型上验证，缺乏系统级功耗和 GPU 对比；
- 结果未提供置信区间，仅为单次或少量跑测；
- 递归状态更新仍在 CPU 上执行，未完成完全迁移。

---

## 338. An Unbounded Archive-based Transfer Strategy for Dynamic Multi-Objective Optimization with a Changing Number of Objectives

**arXiv ID:** 2609.27430 | [PDF](https://arxiv.org/pdf/2609.27430v1)

**作者:** Zhiyun Xiao `[一作]` (Shenzhen University), Wei Sun `[通讯]` (Shenzhen ZTE Software Co., Ltd.)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了无界归档转移策略(UATS)，并将其嵌入SPEA2SDE，形成UATS‑SPEA2SDE，用以处理动态多目标优化中目标数变化的问题。

**💡 创新点**

创新点在于：①在每个环境阶段维护一个无界外部归档，记录子代解；②在目标变更时从归档中提取可行的非支配精英并作为迁移精英，用以重构下一阶段种群；③针对目标增减采用直接投影或重新评估策略，避免了传统归档容量限制带来的信息丢失。

**🔧 技术方法**

采用的技术包括：SPEA2SDE（基于Shift‑based Density Estimation的多目标进化算法）、无界归档机制、环境变化检测、目标增减时的投影与重评估、种群重构以及基准实验评估。

**📊 数据集**

使用了基于可伸缩minus‑DTLZ1‑4的四个动态基准问题，在三种不同目标变更强度（轻度、中度、重度）下进行实验。

**📈 对比分析**

将UATS‑SPEA2SDE与重启式SPEA2SDE（R‑SPEA2SDE）以及四个代表性动态多目标算法（DTAEA、KTDMOEA、LEC、STA）进行比较，采用平均超体积（Mean Hypervolume, MHV）作为性能指标。实验结果显示，UATS‑SPEA2SDE在所有三种目标变更设置下均达到或超过对照算法，尤其在目标数大幅变化时表现最为突出。

**⚠️ 局限性**

局限性包括：①归档在每个阶段会快速增长，虽然在每次变更后重置，但对极长环境阶段仍可能产生内存压力；②对目标变更的假设（仅变化目标数，函数保持不变）限制了其在更复杂动态场景下的适用性；③迁移精英的选择和重构策略相对简单，未来可进一步改进以提升适应性。

---

## 339. ICM: Intra-class Mixing for Domain Adaptation in Adverse Weather

**arXiv ID:** 2609.27533 | [PDF](https://arxiv.org/pdf/2609.27533v1)

**作者:** Boying Li `[一作]`, Hamam Mokayed `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在同一图像内部进行语义类别内混合并强制预测一致性的ICM模块，用于无监督领域适应下的语义分割。

**💡 创新点**

创新点在于只在同一图像同一语义类别内进行混合，保持语义布局不变；采用混淆度引导的混合方向，并通过KL一致性正则化提升伪标签鲁棒性；将该模块作为可插拔组件集成至现有UDA框架。

**🔧 技术方法**

使用教师‑学生自训练框架、EMA教师、温度化softmax、交叉熵与KL一致性损失、Patch‑based intra‑class mixing、混淆分数评估等技术；在DAFormer、MIC、DACS等基线上进行集成。

**📊 数据集**

在Cityscapes（清晰天气）作为源域，ACDC（夜雾雨雪四恶劣天气）作为目标域进行实验；同时在DarkZurich等极端夜景数据集做进一步验证。

**📈 对比分析**

与现有SOTA方法（如MIC、DACS、DAFormer等）对比，Cityscapes→ACDC上取得75.7% mIoU，较之前最高73.8%提升1.9个百分点；在各类中尤其是stuff类（道路、天空、人行道）获得显著改善。

**⚠️ 局限性**

局限性包括：在极端夜间场景下性能下降；对thing类的混合不如对stuff类有效；对权重λ_sup、λ_KL等超参数的敏感性尚未充分验证；对不同骨干网络的泛化能力仍需进一步评估。

---

## 340. Kairos: Grounded Forecasting of Presence and Directional Flow in 4D Scene Graphs

**arXiv ID:** 2609.27467 | [PDF](https://arxiv.org/pdf/2609.27467v1)

**作者:** Iacopo Catalano `[一作]` (University of Turku), Jorge Peña Queralta `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Kairos，一种可预测的方向流记忆，将 3D 场景图扩展为 4D 场景图，实现对任意时间点人流位置与方向的完整分布预测。

**💡 创新点**

创新点：① 在几何构造的 voxel 基础上为每个 voxel 存储半包装 Gaussian Mixture 的方向混合权重和速度均值，并通过频域预测器（FreMEn）实现时间条件的连续分布预测；② 引入邻域共享与相邻 voxel 的耦合，使预测既具方向性又考虑空间关联；③ 通过 presence channel 区分无流与未观测，提供可组合的存在与流信息；④ 通过对 3DSG 的重键映射保证地图优化（loop‑closure）时预测保持一致；⑤ 在规划任务中直接使用预测进行 encounter‑probability 规划。

**🔧 技术方法**

使用技术包括：TSDF 体素重建、Generalized Voronoi 导航层、Semi‑Wrapped Gaussian Mixture 模型、非均匀离散傅里叶变换 (NUDFT) 频域预测、Gamma–Poisson 计数模型、基于贝叶斯的可信区间推导、以及对 4DSG 的层级聚合与边缘注释。

**📊 数据集**

数据集：① 机器人自身收集的 ZED Stereo 轨迹（16 天共 213 分钟，跨 3 个月）；② ATC 购物中心人流轨迹（92 天，约 1 年）；③ HB 车站候车厅连续 11 个月的人流轨迹；并在这些数据上进行模拟与真实轨迹回放。

**📈 对比分析**

比较方法：与基于栅格的 FreMEn、STeF‑Map、CLiFF‑map，和 3DSG 基础的 Aion、Rheos 进行对比；评估指标包括 Mean Log Predictive Density (MLPD)、CRPS、速度 MAE、Presence MLPP 等。结果显示 Kairos 在流向、联合分布、速度、Presence 预测上均优于基线，尤其在稀疏观测下仍保持较好表现；在 encounter‑probability 规划任务中，Kairos 的规划成功率与遇人数与全观测基线相近，且略优于时间不变地图。

**⚠️ 局限性**

局限：① 预测器采用累计均值，难以快速响应永久性变化；② 学习的相邻 voxel 相关系数需要足够的边缘观测，短期部署效果有限；③ 权重的置信区间采用 Gaussian 近似，未考虑 simplex 边界导致覆盖不足；④ 仅在地面几何上建模，未充分利用语义或轨迹先验。

---

## 341. SGDet3D++: Geometry-Grounded Semantics for 4D Radar and Camera 3D Object Detection

**arXiv ID:** 2609.27671 | [PDF](https://arxiv.org/pdf/2609.27671v1)

**作者:** Xiaokai Bai `[一作]` (Zhejiang University), Hui-liang Shen `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于假设条件的雷达‑摄像融合框架，分别对语义检索、几何精炼和时序继承进行锚点约束。

**💡 创新点**

创新点在于将雷达返回视为对象级证据，在每个解码步骤前通过锚点几何、径向运动一致性过滤和验证，避免了传统全局融合中的误匹配。

**🔧 技术方法**

采用Anchor‑Grounded Semantic Retrieval (AGR)、Geometry‑Consistent Anchor Refinement (GCR) 和 Doppler‑Verified Correspondence (DVC) 三个模块，并结合多摄像头特征金字塔与BEV雷达特征的交互。

**📊 数据集**

在OmniHD‑Scenes和ManTruckScenes两个大型4D雷达‑多视图数据集上进行训练与评测。

**📈 对比分析**

相较于现有最佳雷达‑摄像检测器（如4DR360°、RCBEVDet等），在OmniHD‑Scenes mAP提升3.82点、ODS提升6.82点，在ManTruckScenes mAP提升6.82点、NDS提升9.22点；在TJ4DRadSet上亦名列第一。

**⚠️ 局限性**

局限性包括对雷达返回质量和同步精度的敏感性，未对不同传感器配置或极端光照、雷达干扰等情况做充分验证。

---

## 342. Private Decentralized Optimization with Noise Reduction and Bias Correction

**arXiv ID:** 2609.27658 | [PDF](https://arxiv.org/pdf/2609.27658v1)

**作者:** Yizhao Fan `[一作]`, Jiaojiao Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种私有递归分布式优化方法PRDO，用于在分布式学习中同时降低采样噪声、隐私噪声和异构数据导致的去中心化偏差。

**💡 创新点**

创新点在于将递归同批梯度差分估计与Exact Diffusion去中心化偏差校正相结合，给出非凸收敛界且不要求节点间数据异质性均匀；并证明在满足一定条件下递归差分可使查询灵敏度低于传统私有Exact Diffusion。

**🔧 技术方法**

技术包括差分隐私机制（Poisson采样、Gaussian噪声、隐私放大与适应性组合）、递归梯度差分估计、Exact Diffusion去中心化协同、梯度裁剪与噪声加性、非凸收敛分析及网络传播误差分解。

**📊 数据集**

实验使用了两类数据集：非私有的逻辑回归实验（32节点环形网络，每节点2000条样本）以及私有的CIFAR-10图像分类实验（10节点环形网络，采用VGG式网络，数据按异构方式划分）。

**📈 对比分析**

与D‑GD/D‑SGD、Exact Diffusion、EDM、Momentum Tracking等非私有基线以及DP‑ED、DP‑EDM、DP‑MT、DP‑DSGD等私有基线进行比较。PRDO在非私有实验中在后期梯度范数上优于其他方法；在私有CIFAR‑10实验中，PRDO在所有隐私预算下的测试准确率均高于所有基线（如ε=8时53.42% vs DP‑DSGD 49.00%）。

**⚠️ 局限性**

主要局限是实验与理论均基于“裁剪不活跃”假设；若裁剪激活会产生随机偏差，论文尚未对其影响给出分析；此外，递归差分的敏感性减小条件需满足CΔ<Cg，实际应用时需先验证该条件。

---

## 343. Foundations of Algebraic Architecture Theory: A Rising Sea of Geometry, Transport, Comparison, and Reconstruction

**arXiv ID:** 2609.27638 | [PDF](https://arxiv.org/pdf/2609.27638v1)

**作者:** Hiroyuki Nakahata `[一作]` `[通讯]` (Independent Researcher), Hiroyuki Nakahata (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并构建了代数架构理论（AAT），通过明确读取（reading）定义对象、操作与方程，构造核心（core）与几何（geometry），并利用Čech cohomology与 sheaf 理论分析局部一致性、诊断、传输、比较与重构等五类软件变更决策问题；同时将传统的镜头（lens）与协议（protocol）语义映射到 AAT 框架中，证明语义保持映射与 AAT 中的结构保持映射一一对应。

**💡 创新点**

创新点包括：① 对“读取”进行形式化，统一描述不同软件视角下的结构与变化；② 通过核心与几何构造实现从有限原子族到闭合对象族的系统化构造；③ 引入第一阶 Čech 失配类作为局部一致性与修复的判定工具；④ 证明重构定理，说明全局结构可由局部描述唯一恢复；⑤ 建立镜头与协议语义与 AAT 的双向对应，提供语义保持与结构保持的等价关系；⑥ 用 Lean 进行形式化证明，保证理论严谨性。

**🔧 技术方法**

技术手段主要包括：范畴论（对象、态射、自然变换、同构等）、预堆叠/堆叠、Čech 复形与上同调、代数几何中的理想与零点集、群论与同构分类、以及 Lean 证明助手进行形式化验证。

**📊 数据集**

本文为理论性研究，没有使用具体实验数据集；所有结果均基于抽象数学构造与形式化证明。

**📈 对比分析**

比较方法以理论证明为主，主要通过构造同构、相容性条件和判定算法（如检测闭合的核心、验证 Čech 失配类为零）来证明不同变更方案的可比性。性能评估并未在实验层面给出，但理论上可通过算法复杂度分析得到上限。

**⚠️ 局限性**

局限性包括：① 需要在有限原子族与有限覆盖下工作，复杂度与规模增长受限；② 对可计算性与算法实现的讨论较少，实际工具化仍待进一步研究；③ 只针对结构保持而非行为保持（例如时间复杂度、性能特征）等动态属性；④ 依赖高度抽象的数学模型，对具体软件工程实践的直接迁移仍需桥接工作。

---

## 344. Learning Local Heterogeneity and Cross-Region Context for Large-Scale Traffic Forecasting

**arXiv ID:** 2609.27637 | [PDF](https://arxiv.org/pdf/2609.27637v1)

**作者:** Qi Feng `[一作]` (Northwestern Polytechnical University), Kaifang Wan `[通讯]` (Northwestern Polytechnical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对大规模道路网络的交通流量预测，提出了一种融合节点级局部关系聚合与区域级跨区域交互的Local-Region Spatial Temporal网络。

**💡 创新点**

创新点在于：① 利用道路身份和行驶方向划分邻居关系，设计关系感知的局部聚合机制；② 通过区域均值池化和区域间注意力构建紧凑的跨区域上下文；③ 两种层次交互在保持局部细粒度信息的同时，显著降低了全节点交互的计算成本。

**🔧 技术方法**

技术方案包括：空间时间嵌入、关系感知局部聚合（RLA）、跨区域交互（CRSI）以及基于注意力的区域表示学习；模型训练采用AdamW、权重衰减、学习率衰减等标准优化策略。

**📊 数据集**

实验使用LargeST基准的四个真实数据集：SD、GBA、GLA、CA，覆盖716-8600个传感器，15分钟采样周期，共35,040个时间步。

**📈 对比分析**

与AGCRN、STGODE、D2STGNN、PatchSTG等26个基线进行对比，LRST在四个数据集的MAE、RMSE、MAPE均显著优于所有方法，平均相对提升约4%-6%，同时参数量和推理时间保持在可接受范围内。

**⚠️ 局限性**

局限性包括：① 需要预先对道路网络进行邻居构造和区域划分，对网络拓扑变化不够灵活；② 仅在静态道路信息上进行关系建模，缺乏对动态交通模式的自适应更新；③ 对未见网络的迁移能力尚未验证，未来可结合预训练时间序列模型进一步提升零样本泛化。

---

## 345. Pheno-GS: Phenoscape-scale Geodesic Sinkhorn

**arXiv ID:** 2609.27633 | [PDF](https://arxiv.org/pdf/2609.27633v1)

**作者:** Alistair Wilkinson `[一作]`, Smita Krishnaswamy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种可在大规模单细胞分布上计算几何感知的无平衡地理Sinkhorn距离的方法Pheno-GS，解决了稀疏/离散图、无平衡分布、规模化计算等问题。

**💡 创新点**

创新点：1）通过图连通性正则化保证稀疏或离散图上的地理距离；2）利用KL边缘惩罚实现无平衡OT；3）批量矩阵算法一次性计算所有配对距离，实现数百倍加速。

**🔧 技术方法**

使用技术包括：基于图的热扩散近似地理距离、Chebyshev多项式近似热核、Sinkhorn迭代、KL边缘惩罚、批量扩散（BMS）算法。

**📊 数据集**

数据集：合成瑞士卷、Gaussian簇、稀疏桥等；真实单细胞数据为>1,500个患者来源的PDO CyTOF处理条件。

**📈 对比分析**

与Euclidean Sinkhorn、LR Sinkhorn、DiffusionEMD、Euler Sinkhorn以及原始Geodesic Sinkhorn（GS）比较；在KNN任务中Pheno-GS保持或超过GS的准确度，同时在n=500时比GS快≈200×，比其他基线快数十倍，并在无平衡条件下生成更生物学合理的运输计划。

**⚠️ 局限性**

局限性：仍需预先构建图并调节连通性参数；批量方法依赖Chebyshev近似，在极端高维或不规则图上可能出现误差；KL边缘惩罚的超参数选择仍需经验指导。

---

## 346. InGuard: Towards Generalized Inner Guardrail for Safe Text-to-Image Generation

**arXiv ID:** 2609.27620 | [PDF](https://arxiv.org/pdf/2609.27620v1)

**作者:** Zeyu Wang `[一作]` (Alibaba AAIG), Hui Xue `[通讯]` (Alibaba AAIG)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为 InGuard 的内部安全框架，集成在文本到图像生成模型的内部管线中，对提示嵌入进行风险分类、嵌入空间增强以及潜在空间检测，能够在保持生成质量的同时阻止不安全内容。

**💡 创新点**

创新点在于：①在模型自身文本编码器的嵌入空间进行多维风险评估；②设计 SAGE 通过标签门控的概念子空间与软门控融合实现细粒度的嵌入增强；③利用流匹配推理中的一阶干净潜在估计进行早期检测；④构建 RevGen 安全基准集以评估开放权重模型的安全性。

**🔧 技术方法**

采用了多任务 MLP 分类器、软门控嵌入投影（SAGE）、ConvNeXt‑Base 潜在检测器以及基于 VAE 的流匹配潜在采样，整体不改动生成模型参数。

**📊 数据集**

主要使用了由真实图像反向生成得到的 RevGen 10,000 条提示（包含色情、血腥、受控 IP 等标签），并在 OpenImages 上进行域适配预训练。

**📈 对比分析**

在五个公开权重文本到图像模型上对比外部守栏和 InGuard，InGuard 实现 97.9–98.8% 的安全率，且将良性干扰率从约 10% 降至 2–4%，参数量仅为外部守栏的 1/3。

**⚠️ 局限性**

局限性包括：对提示分类器的准确性高度依赖；标签门控仅覆盖预定义的 34 个概念，可能对新型或多角色 IP 场景适配不足；潜在检测仍需在不同生成器和视频等新领域进行验证。

---

## 347. Can Jev Judge Radiology Reports? Evaluating a System One Model for Clinical Factuality

**arXiv ID:** 2609.27607 | [PDF](https://arxiv.org/pdf/2609.27607v1)

**作者:** Jiaju Huang `[一作]` (Macao Polytechnic University), Tao Tan `[通讯]` (Macao Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种名为Jev的评判器，用一句支持性问题快速评估生成的放射学报告与参考报告之间的事实差异，并将结果与专家错误计数对齐。

**💡 创新点**

创新点在于：① 仅使用一类支持性问题即可获得与多问题配置相当的专家一致性；② 将报告拆分为原子陈述后，用双向不一致/遗漏概率（支持/矛盾/未覆盖）做软计数；③ 在保持低成本的同时实现与专家评估高度相关的 Kendall τ（如 RadEvalX 0.573）。

**🔧 技术方法**

技术手段包括：System One（Jev）模型，L1 层级原子分句抽取，单一支持性问题的概率输出，双向聚合公式（E(C,R)=∑[p_c+p_u] + ∑[p_u]），以及与开源 NLI 判断器的对比实验。

**📊 数据集**

使用的数据集有：RadEvalX（100 对），RadEvalExpert（624 对，208 研究），ReXErr（19,513 句子）以及 RaTE-Eval 辅助测试。

**📈 对比分析**

与 RadCliQ、RadGraph、BERTScore、RadFact、RadMatch、GREEN 等基线对比，Jev 在 RadEvalX 的总错误 Kendall τ 为 0.573，RadEvalExpert 为 0.398，均优于开源 NLI（差距约 0.236/0.075）。成本方面，每 100 对报告的判定成本 < 0.03 美元；一次性支持问题可将输入 token 减少 43–45%。

**⚠️ 局限性**

局限性包括：① 评估仅基于文本参考，未验证对真实影像的准确性；② 对语言错误（如错别字、同义词）不敏感；③ 在有限的专家标注集上评测，难以保证在更大、多样化的生成器和病例中的普适性；④ 仅使用单一支持问题时对复杂错误类型的捕捉仍有限。

---

## 348. Gray-Box Model Predictive Control for Articulated Dump Trucks via Gaussian Process Learning of Sideslip

**arXiv ID:** 2609.27597 | [PDF](https://arxiv.org/pdf/2609.27597v1)

**作者:** Arash Shahirpour `[一作]` (RWTH Aachen University), Tim Reuscher `[通讯]` (RWTH Aachen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用高斯过程回归学习侧滑角度，并将其嵌入到运动学模型中，形成灰盒模型，用于改进矿山作业用联装装载车的轨迹跟踪控制。

**💡 创新点**

创新点在于将侧滑角度这一物理可解释变量作为学习目标，用 GPR 直接建模而非传统的残差学习，从而提升模型的可解释性和预测精度。

**🔧 技术方法**

使用高斯过程回归、运动学（仿真）模型、模型预测控制（MPC）以及线性化与离散化技术。

**📊 数据集**

采用仿真得到的动态车辆模型轨迹数据进行训练与测试，数据覆盖典型行驶速度与舵角/翻滚角等状态，测试集还包含更大侧滑角的极端轨迹。

**📈 对比分析**

将三种 MPC（白盒运动学、白盒带侧滑参数、灰盒）在同一仿真轨迹上对比，使用侧向跟踪误差作为评价指标。灰盒 MPC 将最大侧向误差从 2+ 降至 0.56，误差分布显著收敛。

**⚠️ 局限性**

仅在仿真环境验证，缺乏真实车辆实验；训练数据范围有限，侧滑角高于 10° 时误差增大；需要进一步开发在线/自适应学习机制以适应不同工况。

---

## 349. Efficient Linear Bandits via Cluster-Aware Sketching

**arXiv ID:** 2609.27594 | [PDF](https://arxiv.org/pdf/2609.27594v1)

**作者:** Hantao Yang `[一作]` (University of Science and Technology of China), Defu Lian `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究高维线性bandit的计算效率，提出Cluster Sketch Linear Bandit（CS‑LB）算法；

**💡 创新点**

创新点在于将动作集划分为簇，使用Frequent Directions无信息损失的sketch，并通过sentinel选择最乐观簇，保证固定sketch大小下子线性回报并显著降低计算成本；

**🔧 技术方法**

采用Frequent Directions（FD）sketch、簇内无损更新、sentinel‑guided 集中选择、线性bandit置信椭圆和聚类热身等技术；

**📊 数据集**

使用合成数据，设置不同规模的(N,d,l)与线性相关系数m^l的实验样本；

**📈 对比分析**

与UCB1、OFUL、SOFUL、CBSCFD比较，实验表明在相同sketch下CS‑LB实现更低的累计回报、运行时间更短，尤其在明显簇结构时明显优于SOFUL、CBSCFD；

**⚠️ 局限性**

局限性：需要一次性读取完整动作集，无法处理动态或增量动作；热身阶段成本较高；当数据缺乏明显簇结构时性能相对受限。

---

## 350. TNLearn: An Open Source Python Package for Task-based Neurons

**arXiv ID:** 2609.27564 | [PDF](https://arxiv.org/pdf/2609.27564v1)

**作者:** Meng Wang `[一作]` (Shenzhen H&T Intelligent Control Co., Ltd.), Fenglei Fan `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发并发布了 TNLearn 这个开源 Python 包，用于自动发现任务相关神经元并将其集成到可自定义的网络结构中

**💡 创新点**

提出“任务相关神经元”概念，提供多种符号回归与搜索方法（遗传编程、LLM、RL、张量分解）来生成可学习的神经元，并实现模块化的可插拔网络层

**🔧 技术方法**

向量化符号回归 (VSR)、遗传编程 (GPSymRegressor)、大语言模型 (LLM)、强化学习 (REINFORCE) 与张量分解等技术，并基于 PyTorch 自动微分实现网络训练

**📊 数据集**

主要使用 sklearn 的 synthetic regression 数据（make_regression）做演示；包本身支持任何数值或图像等通用数据集，示例中仅演示合成数据

**📈 对比分析**

通过在 MLPRegressor 中使用由 GPSymRegressor 生成的任务相关神经元构建网络，并在同一 synthetic 数据集上与标准 MLP 进行比较，实验表明在相同参数规模下性能可提升（精度更高、收敛更快）

**⚠️ 局限性**

缺乏理论指导选择最佳符号回归方法；GPU 加速尚未完成，扩展性与效率受限；大规模实验验证与基准缺失，需进一步探索不同任务的适用性

---

## 351. When Visual Quality Misleads: Intent Recognition under Rendered Avatar Distortions

**arXiv ID:** 2609.27560 | [PDF](https://arxiv.org/pdf/2609.27560v1)

**作者:** Ning-Hsuan Chang `[一作]` (National Chengchi University), Yu-Chih Chen `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过受控行为实验评估三维化身渲染质量与沟通效果的关系

**💡 创新点**

提出“误导性质量”(Misleading Quality)概念与“意图质量分”(Intent Quality Score, IQS)，揭示视觉质量与动作识别准确率的解耦

**🔧 技术方法**

使用标准的图像/视频质量评估(IQA/VQA)指标、基于特征回归的监督模型（CONTRIQUE、Re‑IQA、DreamSim）以及自定义的IQS评分方法

**📊 数据集**

九个精细化身-动作对（共10个内容单元，去除一组失衡数据），在15种渲染条件下生成150幅静图与10帧视频，收集59名参与者的2,688次判断

**📈 对比分析**

与24个直接计分的IQA/VQA指标和3个监督回归基线进行比较，最佳回归模型Re‑IQA在留一内容交叉验证下得到PLCC≈0.44，IQS的自举分半一致性达到PLCC≈0.64，均远低于理想水平

**⚠️ 局限性**

实验仅在单目桌面视角下进行，缺乏HMD沉浸感；使用的失真是手工控制的近似，而非真实编解码器产生的噪声；样本只包含五款风格化化身与两种动作，难以推广到更广泛的场景和多样化的沟通任务

---

## 352. ThaiTrees: Thai Syntactic Dependency Trees Across Domains

**arXiv ID:** 2609.27558 | [PDF](https://arxiv.org/pdf/2609.27558v1)

**作者:** Attapol T. Rutherford `[一作]` (Chulalongkorn University), Papatchol Thientong `[通讯]` (Chulalongkorn University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 ThaiTrees 342M 词的泰语自动解析语料库，并发布了词频词典与 CoNLL-U 格式的解析结果，供语法关系查询与分布式分析使用。

**💡 创新点**

创新点在于：① 提供了规模远超现有手工树库的全自动解析语料；② 设计了可复现的多阶段处理流水线，确保数据处理可追溯与可再现；③ 通过跨四个领域（新闻、维基、口语转录、社交媒体）的对比，揭示泰语语法与词汇使用的正式与口语双重分布。

**🔧 技术方法**

技术方法：使用 AttaCut 进行词切分、CRFcut 进行句子分割、AttaParse（基于 Stanza 的图模型）进行依存解析、PhayaThaiBERT 作为 UPOS 标注器，并在各阶段采用 Pandas/Parquet 等工具完成数据流和频率统计。

**📊 数据集**

数据集来源：新闻（ThaiPBS 公共网站 100k+ 文章）、维基百科（随机抽样 1.3M 文章）、口语转录（YouTube 录音 7+ 频道的手工字幕）、社交媒体（Wisesight 与 Pantip 论坛帖子），共计 342M 词。

**📈 对比分析**

对比方法：采用词频、log‑odds‑ratio（keyness）与依存模式频率，对四个域进行词汇与语法层面的统计分析；解析性能评估显示 POS 准确率 90.64%、UAS 86.4%、LAS 76.6%；词频与关键词分布清晰划分出正式域（新闻、维基）与口语域（口语转录、社交媒体）。

**⚠️ 局限性**

局限性：① 依存解析准确率未达 90%，可能影响细粒度语法分析；② 词切分与句子分割工具产生的 token ID 不一致导致词典与解析结果无法一一对应；③ 仅以 UD Thai‑TUD 作为 gold 标注，未对新语料的错误进行人工校验；④ 版权与域限制导致部分内容可能无法公开分发。

---

## 353. FLEET: From Logits Entropy to Enhanced Trajectories in Text Generation

**arXiv ID:** 2609.27657 | [PDF](https://arxiv.org/pdf/2609.27657v1)

**作者:** Oleksii Streltsov `[一作]`, Oleksandra Vitko `[通讯]` (Kharkiv National University Of Radio Electronics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 FLEET，一种基于记忆的搜索框架，通过熵触发与 VectorDSU 记忆引导贪婪解码，替代传统温度采样。

**💡 创新点**

创新点在于：① 使用条件熵与方差熵动态识别高不确定性决策点；② 引入 Vector Disjoint Set Union 在线聚类存储状态历史；③ 在此基础上采用 pUCT 进行目标导向搜索，并通过 logit 惩罚实现对模型输出的动态调整。

**🔧 技术方法**

技术手段包括 LogitLens 提取中间层 logits、熵与方差熵计算、VectorDSU 状态映射、MCTS‑pUCT 搜索策略以及基于记忆的 logit 处罚。

**📊 数据集**

实验使用 Llama 3.2‑3B，评估数据集为 LiveCodeBench（代码生成）和 GSM8K（数学推理），并使用 Skywork‑Reward‑V2 作为 ORM。

**📈 对比分析**

与标准温度采样（T=0.6/0.8）在 Pass@32 与 Pass@k 评估下对比。基准验证时，LiveCodeBench Pass@32 从 59.9% 提升至 66.2%（+6.31pp），GSM8K 从 97.27% 提升至 97.80%（+0.53pp）。在 ORM 选取下，LiveCodeBench 仍提升 5.86pp，GSM8K 轻微下降。总体来看，FLEET 在多样化任务中表现更好，且实现约 3 倍速度提升。

**⚠️ 局限性**

局限性包括：① 需要访问并修改内部隐藏层，难以直接应用于仅输出 logits 的模型；② 在深度反思或归纳决策点表现不佳；③ 对 ORM 排序错误敏感，可能导致最终选取错误；④ 生成噪声或无意义片段的概率略高；⑤ 目前未在更大或多模态模型上验证其泛化性。

---

## 354. The Path Matters: Evaluating Small Language Models Beyond Answer Accuracy in KGQA

**arXiv ID:** 2609.27669 | [PDF](https://arxiv.org/pdf/2609.27669v1)

**作者:** Eduin E. Hernandez `[一作]` (National Yang Ming Chiao Tung University), Stefano Rini `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在受控环境下评估冻结的小型语言模型（SLM）作为图导航策略，剥离检索和答案生成，仅让模型选择合法图边并决定是否停止，并使用Hits@1、路径编辑距离（PED）和结构重叠（F1_SG）等指标评估终端答案准确率与路径可信度。

**💡 创新点**

创新点在于通过THESEUS导航框架将图访问、搜索、推理与答案生成分离，专注评估模型在只看到合法边时的局部决策能力；首次将单一演示（one-shot）与ToG式搜索的影响拆解，揭示终端准确率与路径可信度并不总是同步，且搜索并不一定提升路径质量。

**🔧 技术方法**

技术方法包括：THESEUS图导航与可追踪框架；使用结构化输出的Frozen SLM（Gemma 4、Granite 3.3、Llama 3.1、Ministrar 3、OLMo-3、Phi-4 Mini、Qwen2.5、Qwen3）；零/一 shot提示、确定性解码；路径编辑距离（PED）与F1_SG评估；ToG式搜索框架（宽度w=1、w=3）。

**📊 数据集**

数据集为THESEUS提供的Kinship和MQuAKE-ST（单答案和多答案）KGQA数据集，包含固定图结构、主题实体、答案集合与标注推理路径。

**📈 对比分析**

比较方法：在相同的导航接口下对比零-shot与one-shot、不同模型的Hits@1与PED/F1_SG，并与随机游走和最短路径oracle对标；对MQuAKE-ST进行ToG搜索宽度w=1、w=3的评估。性能显示：部分模型（Ministrar 3、Gemma 4）在路径可信度上优于oracle，终端准确率与路径质量并非总相关；单一演示对模型效果影响不一，搜索宽度往往降低Hits@1并提升PED，生成答案可在一定程度上弥补路径错误。

**⚠️ 局限性**

局限性：仅评估冻结模型，未考虑任务特定微调；只关注局部导航，未整合检索、搜索与答案生成的完整管线；受限于结构化输出接口；终端答案仅为终点实体，未评估多候选生成质量；结果可能不适用于更复杂或更大规模的KGQA系统。

---

## 355. Robust Adversarial Reinforcement Learning with Risk Sensitivity and Critic Consistency Regularization

**arXiv ID:** 2609.27667 | [PDF](https://arxiv.org/pdf/2609.27667v1)

**作者:** Jiaxi Wu `[一作]` (Tsinghua University), Xueqian Wang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的鲁棒对抗强化学习框架RACER，能够在动态不确定性和分布偏移下训练出更稳健的控制策略。

**💡 创新点**

1）引入风险敏感的对抗目标，根据主体状态动态调节扰动强度；2）加入双Q值一致性正则化，抑制因对抗扰动导致的Q值差异，从而提升学习稳定性。

**🔧 技术方法**

基于Soft Actor-Critic (SAC) 的双Critic架构，结合风险函数R(s)与一致性正则化项，采用离策略强化学习和经验回放。

**📊 数据集**

使用DeepMind Control Suite中的MuJoCo连续控制任务：Walker-Run、Hopper-Stand、Cheetah-Run、Cartpole-Balance。

**📈 对比分析**

与RARL、MixedNE-LD、CAT、QARL等对抗RL基线以及非对抗SAC进行对比。RACER在所有任务中均表现出更高的鲁棒性、训练更快收敛且方差更小，尤其在极端对抗或分布偏移下优势明显。

**⚠️ 局限性**

在极端参数设定下对抗扰动仍有可能导致策略失效；一致性正则化系数需在不同任务中调优，过大可能削弱双Critic的多样性。

---

## 356. SHRAV: State-Hypothesis-Reason-Action-Verify Framework for Physical Modeling and Inverse Design

**arXiv ID:** 2609.27621 | [PDF](https://arxiv.org/pdf/2609.27621v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 357. Evolutionary Stability Does Not Guarantee Learning Accessibility: A Multi-Agent Reinforcement Learning Perspective on Cooperation Emergence

**arXiv ID:** 2609.27664 | [PDF](https://arxiv.org/pdf/2609.27664v1)

**作者:** Yijie Wang `[一作]` `[通讯]` (Liupanshui Normal University), Yijie Wang (Liupanshui Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究比较了在三方治理博弈中，演化动力学下的合作稳定性与有限样本强化学习过程中的合作可达性；

**💡 创新点**

创新点在于将复制者动力学与独立价值学习（ε‑贪婪 Q‑学习、Boltzmann 探索、SA–EA BQL）放在同一支付环境下，系统评估两类动态的“基因池”差异，揭示演化稳定性不一定等同于学习可达性；

**🔧 技术方法**

主要技术包括复制者动力学求解、独立 Q‑学习、尺度归一化 Boltzmann 探索、基于价值差的自适应探索（SA–EA BQL）以及多种诊断指标（动作覆盖、策略熵、Q 值分离）；

**📊 数据集**

使用的“数据集”是论文中给出的三方治理博弈阶段支付矩阵（共享单车治理模型），并在该固定支付环境下构造对称初始偏好网格；

**📈 对比分析**

通过在相同的固定初始偏好网格、训练 20,000 轮、50 随机种子下比较三种学习器的学习盆地（ε‑贪婪 0.88，Boltzmann 与 SA–EA 0.00），验证演化盆地为 1.00；

**⚠️ 局限性**

局限性包括：仅考虑 3 代理、二元动作的极简模型；只做了固定参数与对称初始网格的实验；未涉及人类行为或更大规模/网络化系统；实验基于有限样本与固定时间窗口，无法推广至所有策略或学习算法。

---

## 358. InternW0: A Foundational Physical World Model for Efficient Real-World Interactions

**arXiv ID:** 2609.27656 | [PDF](https://arxiv.org/pdf/2609.27656v1)

**作者:** Jisong Cai `[一作]` (Shanghai AI Laboratory), Weinan Zhang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 InternW——一种融合视频预测与动作生成、支持多模态感知、异步多频率处理、可在局部环境下更新的物理世界模型。

**💡 创新点**

创新点：1）混合 Transformer（Mixture‑of‑Transformers）架构，分离视频专家和动作专家；2）观察条件下的 K/V 编辑器，可在不重新生成视频预测的情况下即时调整控制；3）软提示与域特定接口实现跨机器人、跨感知/执行空间的迁移；4）结合流匹配训练，既能学习视觉动态也能监督动作；5）通过异步推理实现低延迟控制。

**🔧 技术方法**

主要技术：Transformer/DiT、Mixture‑of‑Transformers、流匹配（Flow Matching）、观察条件的 K/V 路由、软提示（Soft Prompt）、异步双向推理、视频‑VAE + DINOv3 编码、混合监督（视觉 + 机器人轨迹 + 触觉）

**📊 数据集**

使用了约 7,200 小时的混合数据：InternData‑A1、AgibotWorld、RoboCOIN、Galaxea、MolmoAct、RoboDojo、Ego（275 小时人类实验室第一人称视频）。在模拟基准上使用 LIBERO、RoboTwin 2.0；在真实实验中使用多臂任务（Make Sandwich、Pick Industrial Parts、Sort Tubes、MoF Experiments）和手部任务（Quantitative Pipetting）。

**📈 对比分析**

与 VLA、WAM、π_0、Fast‑WAM、AHA‑WAM 等基线比较。InternW 在 LIBERO 上平均成功率 98.6%（比 LingBot‑VA 高 0.1pp，Fast‑WAM 高 1.0pp），在 RoboTwin 2.0‑Full 上 93.12%（比 Fast‑WAM 高 1.29pp），在 Clean‑Random 设定下 75.60%（比 GigaBrain‑0.7 高 8.30pp）。在真实任务中，InternW 在大多数任务上优于 π_0.5 和 Fast‑WAM，进阶任务如 MoF 和 Quantitative Pipetting 的进度率分别提升至 68.4% 与 65.3%。

**⚠️ 局限性**

局限性：1）模型规模较大（约数十亿参数），部署在资源受限的嵌入式设备仍具挑战；2）依赖大量标注视频与机器人轨迹，对数据收集成本高；3）在极端新颖的环境/随机化下仍会出现误差累积；4）触觉/力感知集成仅在后期预训练阶段加入，对部分任务的鲁棒性提升有限；5）目前主要针对基于视觉+力的交互，缺乏对深度、音频等更丰富模态的充分探索。

---

## 359. Brain-to-Language Decoding: Tasks, Signals, Methods, Evaluation, Practical Use and Beyond

**arXiv ID:** 2609.27650 | [PDF](https://arxiv.org/pdf/2609.27650v1)

**作者:** Yiqian Yang `[一作]`, Yu Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `109c2b71-d051-425c-831f-0c544c24280d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

综述脑‑语言解码领域，系统整理任务、信号、方法、评估与实践使用，并提出五层框架；

**💡 创新点**

将不同研究归入任务-信号-方法-评估-实践用层级，识别可迁移表示、共享资源与实时交互的关键路径；

**🔧 技术方法**

综合EEG/MEG/EEG/ iEEG、深度学习解码器、语言模型、语音合成、语义嵌入、跨模态预训练等技术；

**📊 数据集**

引用多种公开数据集（如LibriBrain、ZuCo、sEEG、ECoG等）及多任务、多语言、多模态数据；

**📈 对比分析**

在统一基准和共享数据集上比较文本、语音、候选选择等任务，报告WERS约5–10%，句子准确率约70–90%，表明已有显著进展但仍存在差距；

**⚠️ 局限性**

受限于任务对齐不一致、样本量不足、跨受试迁移困难、实时延迟与硬件可持续性，以及临床部署与用户体验的挑战。

---

## 360. Hidden not Deleted: How Networks Suppress Entangled Features

**arXiv ID:** 2609.27593 | [PDF](https://arxiv.org/pdf/2609.27593v1)

**作者:** Akash Samanta `[一作]` (Techno India University), Debasis Chaudhuri `[通讯]` (Techno India University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一个具备超叠加特性的 toy autoencoder 中，研究了通过梯度下降直接抑制特征 B 而保持特征 A 不受影响的概念抹除方法。

**💡 创新点**

提出了“镜像”与“影子”两种梯度收敛解的二分支，并证明两者仅通过抑制而非真正删除，揭示了 LLM 记忆消除中可恢复性失效的机制。

**🔧 技术方法**

使用了基于梯度的“excision”训练、线性 nullspace 投影对照实验、ReLU 与 GELU 激活函数以及单量化补丁等技术。

**📊 数据集**

采用自生成的稀疏抗相关输入数据，构建了单对抗极点（A、B）来控制 entanglement。

**📈 对比分析**

与线性投影基线相比，梯度抑制在所有 entanglement 范围内均能保持 A 的重建误差几乎不变，而线性方法则随 entanglement 增加导致 A 的误差急剧升高；两种解在恢复时均可通过单个标量补丁完全恢复被抑制的 B。

**⚠️ 局限性**

局限在于仅研究了极简的两特征 toy 结构、冻结编码器、未检验大规模 LLM 或多特征情形的可推广性。

---

## 361. MWE-ECL: Recoverable Long-Range Context Does Not Always Override Local Lexical Priors

**arXiv ID:** 2609.27590 | [PDF](https://arxiv.org/pdf/2609.27590v1)

**作者:** Wei He `[一作]` (University of Exeter), Zhenyun Deng `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估 Multiword Expression Effective Context Length（MWE‑ECL）诊断，用以检验长上下文是否能改变已有的本地语义偏好。

**💡 创新点**

创新点在于将检索可恢复性与行为影响分离，定义“覆盖率”“覆盖率下的覆盖率”“保持率”等指标，并通过多语言、多模型、多距离的同构控制设计验证检索‑使用的解耦。

**🔧 技术方法**

采用大语言模型 API 调用，三类对照提示（anchor‑present、anchor‑absent、检索控制），并在 0–128K 词长网格上进行实验，同时引入硬线索、提示拟合、起始位置对比等扰动技术。

**📊 数据集**

使用双语数据集：英文 50 条（25 家族）与中文 20 条（10 家族），每家族包含一条字面、一次隐喻的多词表达；通过人工标注确保 anchor 与解释的一致性。

**📈 对比分析**

在八大模型（DeepSeek、MiniMax、Qwen、GLM、Gemma 等）上，检索准确率常在 0.99+ 但覆盖率（是否改写本地默认）平均 0.81–1.00；同调用、硬线索、提示拟合等控制表明检索与行为解耦；某些模型在 128K 仍可保持检索闭界，但覆盖率不一定。

**⚠️ 局限性**

局限包括：仅用少量 MWEs（以惯用短语为主），未覆盖复杂文体与长文本；检索与解释在同一调用时无法完全隔离；指标依赖于模型的内部默认，难以统一比较；结果受 API 版本、温度设置等部署差异影响。

---

## 362. Unity Insight: A Production Code--Asset Index for LLM Coding Agents in Unity Projects

**arXiv ID:** 2609.27585 | [PDF](https://arxiv.org/pdf/2609.27585v1)

**作者:** Shenhua Gu `[一作]` (Unity China), Hao Chen `[通讯]` (Unity China)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了Unity Insight，一种持久化的跨文件代码-资产索引，帮助LLM编码代理在Unity项目中高效查询脚本与序列化资产之间的关系。

**💡 创新点**

其创新点在于首次为LLM代理提供面向Unity的、持久化且读写分离的代码-资产图谱，并通过五种类型化工具（列表、通配、搜索、读取、引用）直接查询，而非手工解析GUID。

**🔧 技术方法**

技术实现包括：爬虫解析C#脚本与YAML资产，构建SQLite持久化图谱，提供vfs_ls、vfs_glob、vfs_grep、vfs_read、vfs_refs等Typed API；索引按需增量刷新，保证不改动项目文件。

**📊 数据集**

实验使用了两款Unity游戏项目——MIT-授权的2D平台游戏RedRunner和内部的3D卡丁车赛跑项目，共28个项目特定问答作为测试数据集。

**📈 对比分析**

通过对比同一模型和运行框架下的通用探索代理与Unity Insight代理，每个问题仅跑一次，结果显示Unity Insight代理总token数下降53%、耗时下降52%，工具调用下降26%、模型循环次数下降22%。

**⚠️ 局限性**

局限性包括：索引为静态快照，无法反映运行时或编辑器状态；对基于字符串的动态加载等情况存在盲区；目前仅适用于Unity，若迁移需重新适配不同引擎的序列化格式。

---

## 363. DCRL: Decoupling and Coupling Reinforcement Learning via Policy-Reward Manifold Alignment

**arXiv ID:** 2609.27572 | [PDF](https://arxiv.org/pdf/2609.27572v1)

**作者:** Henan Sun `[一作]`, Jia Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对比两种训练策略（共进化与非共进化），评估了政策模型与奖励模型在训练过程中表示空间的对齐程度，主要使用Linear CKA与Procrustes Distance两种度量指标。

**💡 创新点**

创新点在于将Linear CKA与Procrustes距离相结合，形成对齐度量的双重评估；采用多步长对齐曲线并对不同训练进度进行归一化；并在文本嵌入管道中提供首选TF-IDF+SVD和备用哈希BOW两种方案。

**🔧 技术方法**

技术手段包括：TF-IDF特征提取+截断SVD、哈希BOW嵌入、线性CKA计算、Frobenius范数归一化、最优正交对齐（SVD求解Q）、移动平均平滑、曲线可视化与CSV汇总。

**📊 数据集**

实验数据来源于若干“family”（数据集/任务设置）下的训练日志，具体数据集未在论文中列明。

**📈 对比分析**

对比方法：在每个训练步长计算两指标，随后按方法分别取平均值与尾部K步平均，计算差值ΔCKA与ΔProc；若共进化策略显示更高的CKA与更低的Procrustes距离，则表明其对齐效果更佳。

**⚠️ 局限性**

局限性包括：仅使用文本嵌入而忽略可能的更丰富语义表示；指标对噪声和样本数敏感；实验仅涵盖两种策略，缺乏对更广泛对齐机制的验证；数据集具体信息缺失，影响结果可复现性与通用性。

---

## 364. Knowledge Distillation for Intelligent Softwarized Networks: Advances and Open Challenges

**arXiv ID:** 2609.27551 | [PDF](https://arxiv.org/pdf/2609.27551v1)

**作者:** Mohamed Ali Zormati `[一作]` (University of Technology of Compiègne), Hicham Lakhlef `[通讯]` (University of Bordeaux)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对知识蒸馏在智能软化网络中的应用进行系统综述与分类，梳理现有工作、趋势与挑战。

**💡 创新点**

首次将知识蒸馏与网络软化、边缘AI、分布式智能等多层面融合，提出跨层次的统筹框架与未来研究方向。

**🔧 技术方法**

采用系统性文献回顾、功能层级分类、对比分析与案例梳理等方法论。

**📊 数据集**

主要利用公开论文与实验案例，无特定数据集。

**📈 对比分析**

通过构建对比表与讨论，评估了不同KD范式在数据平面、服务层、AI原生网络及分布式智能中的准确率、延迟与能耗等指标，总结了其优势与局限。

**⚠️ 局限性**

研究缺乏统一评估标准、跨层整合与实时动态蒸馏、能耗与安全性等关键问题，难以直接量化性能提升。

---

## 365. AST-Based Automated Elimination of break and continue Statements in Java Code

**arXiv ID:** 2609.27627 | [PDF](https://arxiv.org/pdf/2609.27627v1)

**作者:** Andrés Juárez `[一作]` (University of Málaga), Rubén Saborido `[通讯]` (University of Málaga)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

未提供论文具体内容，无法确定研究目标

**💡 创新点**

无可总结的创新点

**🔧 技术方法**

无可总结的技术手段

**📊 数据集**

无可总结的数据集

**📈 对比分析**

无可总结的比较方法与性能

**⚠️ 局限性**

无可总结的局限性

---

## 366. BiCFlow-MER: Orchestrating Discriminative and Generative Multimodal Emotion Recognition via Conditional Transport

**arXiv ID:** 2609.27615 | [PDF](https://arxiv.org/pdf/2609.27615v1)

**作者:** Yanbing Wang `[一作]` (OPPO Research Institute), Chunyang Yu `[通讯]` (OPPO Research Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 BiCFlow-MER，利用条件流将音频-文本情感证据生成并验证

**💡 创新点**

在证据分离、冲突感知融合以及条件传输+双向几何评分方面实现创新

**🔧 技术方法**

结合 CGD-IB 分解、ECA 冲突加权、CT‑BiRF 条件流、A‑GCH 适配原型云与 BGD 逆向兼容评分

**📊 数据集**

IEMOCAP、MELD（四类/七类）以及零样本 CASE（冲突语义/声学）数据集

**📈 对比分析**

与多种最新音频-文本 MER 方法比较，IEMOCAP/ MELD 上均取得首位（WA/UA/W‑F1）并在 CASE 上实现最高的零样本准确率

**⚠️ 局限性**

对音频/文本单模的依赖性与逆向评分提升有限，且模型对视觉或会话上下文等额外模态尚未适配

---

## 367. RegenHarness: A Robot Agent Harness with Evidence-Gated Recursive Self-Improvement

**arXiv ID:** 2609.27612 | [PDF](https://arxiv.org/pdf/2609.27612v1)

**作者:** Kailin Wang `[一作]` (Country Garden Services Group), Zhaosong Li `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种以证据为门控的机器人代理系统，利用模型循环和代理循环实现任务规划、执行、验证与恢复，并在此基础上引入递归自我改进（RSI）协议；

**💡 创新点**

核心创新在于：①角色隔离的上下文编译（C1–C4）保证规划、执行、验证、恢复各自只访问所需信息；②版本化记忆与事件溯源，实现可审计的任务状态和回滚；③证据门控的提交阀门，只有满足验证且版本一致的结果才更新任务状态；④RSI循环通过执行记录驱动配置迭代，保持安全和性能；

**🔧 技术方法**

使用Python实现的可扩展运行时，配合SQLite持久化存储；采用ROS 2进行设备通信；支持多种后端：VLA策略、导航栈、TAMP求解器、世界模型；引入可配置的评分函数和资源租约机制；

**📊 数据集**

基于真实四足机器人在办公/仓库环境中的声控任务、视觉捕捉和闭路完成测试；使用语音指令、相机图像、导航轨迹以及MCAP日志等数据；

**📈 对比分析**

对比了直接后端完成、监控执行和证据门控执行三种方案，在相同导航后端和路线下进行；评估指标包括：任务成功率、误报完成率、重复调度次数、恢复成本和执行开销；结果显示证据门控方案在误报完成率和重复调度上显著下降，同时恢复成本与监控执行相当；

**⚠️ 局限性**

局限性包括：①依赖完整且高质量的感知与验证证据，感知失效会导致误判；②配置和RSI迭代需要人工或实验支持，难以实现完全自动化；③在多机器人部署时需额外事务与时钟同步；④RSI不更新模型权重，性能提升受限于后端实现。

---

## 368. When Context Misleads: In-context Learning with Jurisdiction in Large Language Models

**arXiv ID:** 2609.27603 | [PDF](https://arxiv.org/pdf/2609.27603v1)

**作者:** Pei-lin Li `[一作]` (Tsinghua University), Shuojin Yang `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的评估框架FakeContext-bench，用于检测LLM在面对伪科学上下文时是否能区分真实规则与误导规则；并设计了后训练方法Jurisdiction In-Context Learning（J-ICL）来提升模型在此类情形下的鲁棒性。

**💡 创新点**

在不使用显式权威标签或辅助目标的情况下，通过自然语言的主权表述与对比学习来让模型学习何时应遵循上下文规则，从而同时保持常规ICL性能并降低误导上下文的影响。

**🔧 技术方法**

使用因果语言模型训练目标、对正负两类示例（J⁺与J⁻）的对比学习、三种训练集合（正样本、J⁺/J⁻对、J⁻/J⁻对），并在多种模型骨干上实施。

**📊 数据集**

FakeContext-bench：包含3500个实例，覆盖物理、数学、化学、生物、统计、历史、计算机科学七个领域；每个实例提供真相答案与误导答案，供模型判断。

**📈 对比分析**

与基线模型、MetaICL、Symbol Tuning等进行对比。J-ICL在四种模型（Qwen3-4B/8B/14B，Llama3.1-8B）上均提升ICL准确率与Reality Rate，尤其在大模型上提升幅度更大，且在各种语句形式与语义提示下保持鲁棒。

**⚠️ 局限性**

目前对主权推理的抽象与组合性理解仍有限；模型在极端或极少见的主权表述、复杂多义上下文中可能仍失效；评估仍局限于伪科学场景，未覆盖所有类型的误导上下文。

---

## 369. Efficient Geometry Representation Strategies for the Shape Optimization of Profile Extrusion Dies

**arXiv ID:** 2609.27602 | [PDF](https://arxiv.org/pdf/2609.27602v1)

**作者:** Jana Sasse `[一作]` (TU Dortmund University), Stefan Turek `[通讯]` (TU Dortmund University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一套基于对偶求导的确定性、可解释性自动模具设计框架，能够通过形状或拓扑优化直接改进聚合物挤出模具的流道几何。

**💡 创新点**

其创新点在于将非边界对合几何表示（IBM/FBM）与对偶形状/拓扑优化相结合，并设计了虚拟界面重构技术来准确恢复敏感度信息。

**🔧 技术方法**

采用的技术包括对偶形状/拓扑优化、IBM/FBM几何表示、SurfaceNets重构、FEAT3有限元求解、Q^2/P_disc^1元素、单域Newton、几何多重网格与Vanka型增广散射子。

**📊 数据集**

实验使用真实的45 mm直径高密度聚乙烯挤出模具几何、Carreau‑WLF黏度模型以及实际工况（10 kg/h、220 °C）进行数值验证，而非公开数据集。

**📈 对比分析**

与传统手工迭代和其他CFD优化方法相比，该方法在仅20次优化循环（约48 M自由度）内将流速标准差降低55%，显著提升流量平衡且计算量更低。

**⚠️ 局限性**

局限性包括对几何对齐的依赖（形状优化）、接口重构的数值误差、未完成全局收敛、以及当前仅处理稳态非等温流动，需要进一步改进自适应网格和热耦合。

---

## 370. ARS-Avatar: Animatable and Relightable Surfel Avatars with Learnable Ambient Occlusion

**arXiv ID:** 2609.27600 | [PDF](https://arxiv.org/pdf/2609.27600v1)

**作者:** Jiateng Liu `[一作]` (Nanjing University of Posts and Telecommunications), Feng Xu `[通讯]` (Tsinghua University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了 ARS-Avatar，利用 Gaussian Surfels 通过多视角图像构建可动画且可重新照明的人体头像。

**💡 创新点**

创新点在于：①从模板网格提取变形先验并融入 Surfel 属性估计；②基于差分的可微屏幕空间环境遮蔽（SSAO）实现可优化的遮蔽半径；③结合延迟着色的 Disney BRDF 实现物理基础材质与光照的联合优化。

**🔧 技术方法**

技术主要包括：Gaussian Surfel 表示、StyleUNet 位置映射网络、线性混合蒙皮（LBS）、延迟着色与 HDR 立方体光照、可微 SSAO 的差分优化。

**📊 数据集**

在 AvatarReX 与 ActorsHQ 两个公开数据集上进行训练与评估。

**📈 对比分析**

与 3DGS-Avatar、WebAvatar、AnimatableGaussian、MeshAvatar、Relighting4D 等基线方法相比，ARS-Avatar 在 PSNR、SSIM、LPIPS、FID 等指标上均取得更优表现，并在复杂服装与关节运动场景下呈现更高保真度的动画与照明效果。

**⚠️ 局限性**

局限性在于：变形先验仅基于预设网格，难以覆盖所有服装与身体形态的多样性；遮蔽半径优化仍需外部学习策略，未来可进一步集成到网络中。

---

## 371. ViMoWear: Visual Motion-Guided sEMG-IMU Representation Learning for Subject-Independent Thumb Gesture Recognition

**arXiv ID:** 2609.27595 | [PDF](https://arxiv.org/pdf/2609.27595v1)

**作者:** Wenjuan Zhong `[一作]` (University of Edinburgh), Kianoush Nazarpour `[通讯]` (University of Edinburgh)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出ViMoWear框架，利用训练时同步的3D手势作为视觉运动监督，仅使用可穿戴传感器进行无监督的主题独立拇指手势识别。

**💡 创新点**

首次将3D手势作为训练专属监督融合sEMG+IMU表示学习，结合跨主体对比学习与拇指加权掩码运动重建，显著提升无监督手势识别。

**🔧 技术方法**

Motion‑Guided Cross‑Subject Contrastive Learning (MGCL) 与 Thumb‑Aware Masked Motion Reconstruction (TMMR) 以及三模态时间CNN编码器。

**📊 数据集**

新收集的31人同步sEMG、IMU与多摄像机捕获的手势数据集，包含七种拇指动态手势。

**📈 对比分析**

采用LOSO验证，线性探针与检索对比，ViMoWear在sEMG+IMU模式下平均平衡准确率从73.22%提升至77.67%，宏F1提升4.7个百分点。

**⚠️ 局限性**

依赖同步的多模态训练数据；目前仅验证单一手势数据集；需进一步在多样化手势与传感器配置上评估。

---

## 372. The Capability Manifold and ML Scaling Laws

**arXiv ID:** 2609.27588 | [PDF](https://arxiv.org/pdf/2609.27588v1)

**作者:** Syed Ali Raza Zaidi `[一作]` (University of Leeds), Maryam Hafeez `[通讯]` (University of Leeds)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了“能力流形”框架，将模型在预训练、微调、推理等不同阶段的资源与多维能力相对应，并用数学模型统一了现有的缩放定律。

**💡 创新点**

创新点在于构造多维能力流形并给出其雅可比矩阵，能够定量分析资源变动对各类能力的边际效应；同时将预训练、后训练与推理计算等异构资源整合到统一轨迹中。

**🔧 技术方法**

技术上采用功率律、边界缩放函数、sigmoid映射及雅可比求导等数学工具，结合Kaplan与Chinchilla预训练缩放及推理搜索的实验设置进行验证。

**📊 数据集**

数据集主要基于公开的大规模语言模型训练语料（如通用文本数据），并引用已发布的模型规模与数据量对比结果；论文未使用特定自定义数据集。

**📈 对比分析**

通过在能力流形上绘制不同资源分配路径（如纯预训练vs推理搜索）并比较其能力轨迹与雅可比，实验表明在相同总资源下，Chinchilla相较于Kaplan能在某些能力维度实现更快的提升。

**⚠️ 局限性**

局限性包括：能力定义仍较为经验化，缺乏对具体任务的定量评估；实验主要集中在理论模型上，未验证在真实应用场景中的泛化性。

---

## 373. Does Step Law Transfer to Small-Scale Language Models? An Empirical Recalibration Below 59M Parameters

**arXiv ID:** 2609.27581 | [PDF](https://arxiv.org/pdf/2609.27581v1)

**作者:** Egor Romanyukov `[一作]` (HSE University), Stepan Dergachev `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

验证并重新校准了Step Law在小模型（N<59M）下的适用性

**💡 创新点**

证明小模型仍符合Power-law结构，但原有系数不再适用，提出了重新校准的公式

**🔧 技术方法**

利用AdamW+warmup–cosine学习率调度、BPE-2048 tokenizer、对超参数网格的二次曲面拟合及线性回归

**📊 数据集**

TinyStories（roneneldan/TinyStories）文本数据集，BPE-2048词表

**📈 对比分析**

对比原Step Law预测的峰值学习率和批大小与实验获得的最优值；发现Step Law系数预测的学习率平均高约4倍，重新校准后的R²分别为0.834（学习率）和0.950（批大小），批大小与模型规模无关

**⚠️ 局限性**

单一随机种子导致的方差估计不足，实验范围仅覆盖0.25–2.03M参数，未检验更大或更小规模的广泛性

---

## 374. CCR: Towards a Common, Quality-Gated CACAO Integrations Registry for European Cybersecurity Automation

**arXiv ID:** 2609.27567 | [PDF](https://arxiv.org/pdf/2609.27567v1)

**作者:** Mateusz Zych `[一作]` (Cyentific AS), Gudmund Grov `[通讯]` (Norwegian Defence Research Establishment)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了Common CACAO Registry（CCR），一个公开、质量门控、基于CACAO标准的HTTP‑API集成注册库，并通过LLM与规则引擎混合管道自动生成并验证整套集成包；

**💡 创新点**

创新点在于：①首次将CACAO与OpenAPI映射结合成可执行的连接器封装；②设计了混合推理管道，分离机械与语义决策；③引入六级成熟度模型与质量门控机制；④将该注册库与欧盟网络安全法规关联，构建共享的集成基础设施；

**🔧 技术方法**

技术包括：OASIS CACAO v2标准、OpenAPI 3.x规范、JSON Schema 验证（Ajv）、Anthropic Claude LLM（受限提示）、自动化后端验证（9维度back‑validation）、Python脚本和GitHub发布的公开仓库；

**📊 数据集**

使用的数据集共8个真实安全API的OpenAPI文档，总计713个端点，覆盖漏洞情报、案件管理、SOAR、网络策略、SIEM/EDR等领域；

**📈 对比分析**

比较方法：与纯规则基线（RBB）进行9维度back‑validation对比；性能指标包括：CACAO schema有效率100%，平均back‑validation 91.5%，LLM成本$0.026/端点；LLM对活动注释的贡献52.7%；dispatch测试成功率94.7%；

**⚠️ 局限性**

局限性：①back‑validation依赖原始OpenAPI，无法完全评估真实可执行性；②未进行专家人工评估；③仅覆盖HTTP‑API，未涉及GraphQL等协议；④LLM生成可重复性有限；⑤缺乏大规模外部贡献与治理评估。

---

## 375. Finite-Sample Binary Hypothesis Testing via Rényi Divergences: Strong Converse and Local Privacy

**arXiv ID:** 2609.27617 | [PDF](https://arxiv.org/pdf/2609.27617v1)

**作者:** Roberto Bruno `[一作]` (University of Salerno), Amedeo Roberto Esposito `[通讯]` (Okinawa Institute of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了利用Renyi相对熵在假设检验中的有限样本极大似然与最优检验误差的对偶式上下界，并给出了相应的样本复杂度上限与下限，进一步扩展到局部差分隐私场景，分析了在不同隐私水平下的误差阈值与样本量关系。

**💡 创新点**

创新点在于：①通过对Renyi相对熵的变分表述，得到更细致的两侧误差界；②在相对熵基准下揭示了误差指数的相位转变阈值；③针对局部差分隐私，首次给出统一的收敛性分析和样本复杂度下界；④引入二进制与k-ary随机化响应机制，实现了在不同隐私强度下的最优性能逼近。

**🔧 技术方法**

使用的主要技术包括：Renyi相对熵与f‑divergence的变分表示、数据处理不等式、极大似然检验的阈值分析、随机化机制下的收缩系数、以及极限分析与对偶理论。

**📊 数据集**

由于研究主要是理论性质，未使用具体的真实或模拟数据集，而是通过概率分布族（如连续与离散分布）进行抽象分析。

**📈 对比分析**

方法与现有理论进行比较：与传统的Kullback‑Leibler基准相似但在大样本极限下给出更精确的指数；在隐私场景中与之前仅给出单侧或非对称界限的工作相比，提供了两侧误差的全局取值范围，理论上表现更优；但缺乏实证验证。

**⚠️ 局限性**

局限性包括：①在临界速率 r = D(P1‖P0) 时界限仍未收敛；②在局部隐私区间未能完全确定相位阈值；③对某些f‑divergence（如Eγ）在高维或无限维空间下的收敛性分析仍不完整；④缺乏实验验证，难以评估实际实现中的常数与实现细节对性能的影响。

---

## 376. A generalizable structural brain MRI foundation model built through dual-priority federated pretraining

**arXiv ID:** 2609.27611 | [PDF](https://arxiv.org/pdf/2609.27611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 377. A DRL-Driven Optimization of RAN Slice Resource Partitioning for V2X SLA Compliance in 5G Networks

**arXiv ID:** 2609.27659 | [PDF](https://arxiv.org/pdf/2609.27659v1)

**作者:** M. Martínez `[一作]` (University of Málaga), R. Barco `[通讯]` (University of Málaga)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于深度强化学习的RAN切片资源分配方法，针对5G网络中的V2X与eMBB服务，动态划分物理资源块(PRB)以满足V2X服务的延迟与可靠性SLA，并兼顾资源利用率与eMBB吞吐。

**💡 创新点**

创新点在于：①将PRB分配建模为MDP，并采用Proximal Policy Optimization（PPO）实现在线决策；②设计了多目标奖励函数，联合优化V2X SLA合规、资源利用效率和eMBB吞吐；③通过预收集的仿真数据实现离线训练，避免对真实网络的干扰；④在多V2X、多eMBB异构场景下验证方法有效性。

**🔧 技术方法**

使用技术：Proximal Policy Optimization（PPO）深度强化学习、MDP建模、奖励函数设计、预先采集的系统级仿真数据、多切片网络模型、基于PRB的调度与资源分配。

**📊 数据集**

数据集：由5G系统级仿真生成的564个样本，涵盖三种网络负载水平、47个PRB配置组合，包含V2X服务的周期性报文、eMBB服务的吞吐期望以及相关KPIs（延迟、吞吐、资源占用、PER等）。

**📈 对比分析**

评估方法：在三种负载水平下进行仿真实验，比较SLA满足率、90%/99.99%延迟百分位、eMBB平均吞吐以及资源利用率。结果显示，PPO策略在满足V2X SLA的同时显著提升资源利用率，降低SLA违约次数，并在eMBB服务间实现更均衡的吞吐分配。

**⚠️ 局限性**

局限性：仅在离线仿真环境验证，未考虑真实网络的时变性和多域交互；奖励函数设计需人工调参；仅覆盖四个切片且采用RR调度，未考察更复杂调度策略；扩展到更大规模或多频段网络时的可扩展性与训练效率尚待验证。

---

## 378. Agent-based Modeling: Equilibrium, Echo Chambers, and Efficiency in Hybrid Coevolutionary Opinion Games

**arXiv ID:** 2609.27639 | [PDF](https://arxiv.org/pdf/2609.27639v1)

**作者:** Ming-Zhi Jiang `[一作]` (National Yang Ming Chiao Tung University), Yung-Ming Li `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种混合共进化意见博弈框架 H-COG，将成本最小化的 Friedkin–Johnsen 代理与基于大语言模型的推理代理融合在同一自适应网络中，并在其均衡状态下量化社会成本、价格差距（PoA）、极化程度和回声室结构。

**💡 创新点**

创新点在于：①首次在同一模型中同时考虑数值更新与语言推理两类代理；②将社会成本和 PoA 概念引入语言驱动的代理体系；③通过成本分解揭示语言代理效率低下主要源于与自身先入为主的偏差，而非与邻居的争议。

**🔧 技术方法**

采用的技术包括 Friedkin–Johnsen 的局部成本最小化更新、Phi‑4 语言模型生成结构化文本、RoBERTa 回归器将文本映射到连续立场尺度、K‑NN 重新连线、基于 Markov 游戏的 OGA 学习和理论上证明的收敛与 PoA 上界。

**📊 数据集**

使用的数据集为 5,199 条来自 Reddit 的枪支管制与堕胎讨论评论，先通过语义聚类、RoBERTa 评分得到立场分布，再按分层抽样构造 50 名代理的初始内在立场与属性。

**📈 对比分析**

通过在三种初始拓扑、两种议题和九个混合比例 α（0–1）上运行 540 场仿真，比较社会成本、PoA、极化与模块化；结果显示纯成本代理（α=1）实现 PoA≈1.1，极化最高；纯语言代理（α=0）PoA≈5.6，极化最低；混合比例增大时 PoA 下降、极化上升，证明两类代理在效率与结构上呈互补对立。

**⚠️ 局限性**

局限包括仅使用单一 LLM（Phi‑4）导致结果可能不具普适性；样本规模仅 50 代理，且人设与固执度在两议题中保持不变；对初始拓扑的等价性检验仅在纯成本代理上显著；模块化分析受分辨率限制；未能探讨多 LLM 版本或更大规模网络的行为。

---

## 379. Agent Name Collision Attacks in Multi-Agent Systems

**arXiv ID:** 2609.27624 | [PDF](https://arxiv.org/pdf/2609.27624v1)

**作者:** Adithyan Arun Kumar `[一作]` `[通讯]`, Adithyan Arun Kumar

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对七个开源Agent Card实现进行静态与动态分析，发现远程卡名称冲突导致错误代理调度；

**💡 创新点**

提出身份不变式与威胁模型，提供跨实现的证据与失效分类，并给出基于稳定身份的 fail‑closed 路由方案；

**🔧 技术方法**

采用代码追踪、注册-调度路径跟踪、隔离实验以及对照实验等技术；

**📊 数据集**

使用七个特定实现的确切代码提交（Google ADK Python/TS、UiPath LangChain、BeeAI、Solace Agent Mesh、Mozilla Any‑Agent、AutoDev）；

**📈 对比分析**

通过对比受冲突影响的路径与负控制路径，证明错误代理调度普遍存在；未测量性能指标，安全影响主要是错误调度；

**⚠️ 局限性**

仅覆盖七个实现，未评估实际部署规模，攻击依赖受害者注册权限，未演示真实生产攻击。

---

## 380. Agent-Based Modeling of Systems of Systems

**arXiv ID:** 2609.27573 | [PDF](https://arxiv.org/pdf/2609.27573v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 381. State-Grounded Conditioning: Wrapping User-Facing LLM Agents Where Direction Depends on Live State

**arXiv ID:** 2609.27606 | [PDF](https://arxiv.org/pdf/2609.27606v1)

**作者:** Qi Liu `[一作]` (Tencent), Zixun Sun `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为 State‑Grounded Conditioning（SGC）的设计原则，利用三层规则包装器（感知、对齐、交互）来外部化实时用户状态的决策，从而避免 LLM 生成时出现的“方向漂移”，并在游戏内对话式教练系统上实现低延迟与高质量的回应。

**💡 创新点**

创新点在于：①将实时状态依赖拆分为可确定的规则引擎与 LLM 的自然语言生成；②设计了三个近似正交的包装器，分别对应意图‑工具冻结、事实‑槽位差异对齐和跨轮次状态机；③提出“方向漂移”这一专属失效类别，并在评测中量化其改进效果。

**🔧 技术方法**

使用的技术包括：基于流式 LLM 的意图识别与工具计划；独立的槽位抽取与冲突差分算法；跨轮次纯函数调度器；以及一套基于人类标注的评估指标（TGA、SGA、TGQ、DGP、CCC）和首词延迟测量。

**📊 数据集**

使用的数据集为一个匿名化的 200 轮对话基准（约 1,000 条回复），来源于已上线的游戏内对话式教练系统，涵盖事实问答、多轮教练和状态依赖交互。

**📈 对比分析**

通过与 Prompting、PE‑Agent 以及逐层 ablation 的比较，Full SGC 在首词延迟维持在 1.5 s 的同时，TGA 从 61.1%/69.8% 提升至 96.7%，SGA 从 20.0%/26.5% 提升至 83.5%，相对 PE‑Agent 的提升分别为 +26.9pp 和 +57.0pp；其它质量指标（TGQ、DGP、CCC）也显著提高。

**⚠️ 局限性**

局限性包括：仅在单一游戏教练场景验证，未在其他领域做实证；评估完全依赖人工标注；只对满足实时状态、可测方向漂移和可结构化状态切片的任务有效；剩余的跨切片失效、状态不可用、未映射意图等仍未覆盖；维护规则代码需要持续工程投入；以及对实时系统的尾部延迟与用户行为效果未公开。

---

## 382. Action-Directed Information for Distributed Control and Agentic Interaction

**arXiv ID:** 2609.27580 | [PDF](https://arxiv.org/pdf/2609.27580v1)

**作者:** Shlomo Dubnov `[一作]` `[通讯]` (University of California San Diego), Shlomo Dubnov (University of California San Diego)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在四足行走机器人上通过比较自身传感器与同伴传感器的局部信息共享，研究其对轨迹跟踪性能的影响，并提出一种基于行动接口信息增益的操作性测量框架。

**💡 创新点**

创新点在于：①从信息理论视角重新定义分布式智能，聚焦信息在决策接口处的作用；②引入行动导向预测增益和洗牌对照实验验证信息对决策的因果影响；③发现同伴传感在失足、滑移及中心控制失效等多重故障条件下能显著提升功能。

**🔧 技术方法**

使用技术包括：Cross‑Entropy Method（CEM）训练策略；多层感知机（MLP）进行行动预测与信息增益估计；有限历史条件下的转移熵/指向信息测量；信息到达/IT‑PAC 等信息理论工具。

**📊 数据集**

数据集与实验环境为自建的四足行走仿真平台，包含多种轨迹（直线、转弯、S形、正弦/脉冲）以及在评估阶段加入的失足、滑移、中心控制掉线等扰动。

**📈 对比分析**

比较方法：在相同扰动条件下测量两种信息拓扑（Own‑Sensor 与 Peer‑Sensor）的平均晚期路径误差；同时计算行动导向信息增益与洗牌对照的差值。结果显示：在失足+中心失效等复合故障下，Peer‑Sensor 的误差显著低于 Own‑Sensor，行动导向增益亦明显更大；但对后期标量误差预测的优势不明显。

**⚠️ 局限性**

limitations: 信息增益是有限历史、模型依赖的估计，非精确通道容量；实验未进行闭环量化传输，无法给出严格的数据速率阈值；仅验证特定扰动下的优势，未证明其普适性；未实现完整的 IT‑PAC 或贝尔曼递归优化框架。

---

## 383. VCMM: Variance-Calibrated Momentum for Multimodal Learning

**arXiv ID:** 2609.27577 | [PDF](https://arxiv.org/pdf/2609.27577v1)

**作者:** Zhongjing Gu `[一作]`, Yiming Cui `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Variance‑Calibrated MomentuM (VCMM)，通过在线估计每种模态的噪声与漂移来动态调整模态特定的动量，缓解多模态联合训练中的模态不平衡问题。

**💡 创新点**

创新点在于利用轻量级探测器实时估计梯度噪声与漂移，并将漂移‑噪声比映射为基于卡尔曼滤波器的动量增益，实现模态间差异化且随时间变化的梯度记忆；同时通过中心化控制和精确偏差校正，保证动量自适应而不需额外学习率缩放。

**🔧 技术方法**

核心技术包括：(1) 轻量级分类探测器获取梯度统计；(2) 在线噪声与漂移估计及指数移动平均平滑；(3) 卡尔曼启发式映射和对数几率中心化的动量控制器；(4) 变动动量下的第一动量自适应偏差校正；(5) 无需额外网络前向/反向传播即可实现。

**📊 数据集**

在四个公开多模态基准上评估：CREMA‑D、KSounds、Twitter 和 NVGesture。

**📈 对比分析**

与多模态基线（Concat、G‑Blend、OGM、PMR、MLA、ReconBoost、LFM、InfoReg、AUG、DecAlign 等）进行对比，VCMM 在所有 8 项指标上均实现最高或第二高分，准确率提升约 1–3%，且仅增加约 13.6% 的显存开销。

**⚠️ 局限性**

局限性包括：(1) 目前仅针对多模态分类任务验证，未知对回归或生成任务的适用性；(2) 需要手动设置 λ 等超参数，对不同任务/数据集的泛化性尚待进一步探索；(3) 尽管开销低于大部分方法，但仍比纯共享动量方案略高；(4) 依赖探测器估计，若梯度统计不稳定可能影响动量调节效果。

---

## 384. Cost Sharing with Hidden Time Flexibility

**arXiv ID:** 2609.27623 | [PDF](https://arxiv.org/pdf/2609.27623v1)

**作者:** Mohsen Pourpouneh `[一作]` (Maastricht University), Farzaneh Rajabighamchi `[通讯]` (Maastricht University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在可变时间窗口服务场景下，如何在不同可报端点约束下选择核心分配，并防止客户通过缩小报告时间段来降低支付，从而提出了核心选择与合同证明（ICP/CCP/SCCP）之间的可行性边界。

**💡 创新点**

首次给出了不同端点可验证域（两侧、单侧、公共端点）下核心选择与合同证明的精确人口边界，并证明了在两侧域下人数≥4时核心选择与任何合同证明不可同时实现；在单侧域下人数≥6时亦不可实现，且提供了最小的可行规则；同时给出了近似核心与合同证明的必要条件。

**🔧 技术方法**

主要使用组合优化与合作博弈论工具，构造核心分配规则（如右端点贪心、几何规则、极端端点规则），以及对比核心约束、效率与合同证明条件的证明。

**📊 数据集**

无实际数据集，所有结果均为理论构造与证明。

**📈 对比分析**

由于是理论性研究，没有实验对比；所提出规则通过严格的证明满足核心选择与合同证明的条件，且给出了在不同人数下可行与不可行的边界。

**⚠️ 局限性**

局限性在于：仅针对固定激活成本、单一共享资源的简单模型；未考虑容量约束、调度优化等实际因素；对近似规则的可行性尚未给出构造实现，仅给出了必要条件。

---

## 385. FDE-Bench: Evaluating LLM Agents for Deployment Environment Configuration

**arXiv ID:** 2609.27571 | [PDF](https://arxiv.org/pdf/2609.27571v1)

**作者:** Weihang Ding `[一作]` (University of California, Berkeley), Qirong Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了 FDE‑Bench 评估平台，评估大型语言模型在实际部署配置（Docker、Compose、Kubernetes）中的能力；通过四层程序化检查（构建、就绪、行为、符合性）和重放机制，对提交的声明性工件进行客观评分。

**💡 创新点**

① 将部署场景（容器化、编排、健康监控）与规范化的四层检验结合；② 引入“重放”重建与执行、无需 LLM 判决；③ 设计四臂发布门、规范泄漏审计与隔离审核，提升任务可靠性；④ 用原始可重复的工作空间和 GUID canary 控制泄漏。

**🔧 技术方法**

使用 Docker-in-Docker、Kind 集群、标准化 LLM 接口（函数调用 + 4 个工具：bash、curl、git、python）；程序化检查脚本；基于 JSON/YAML 的规范和工件；重放引擎。

**📊 数据集**

136 个手工设计的任务，来源于 149 份 blueprint 语料，涵盖 Dockerfile、Compose 文件、K8s manifest 与监控配置；任务按绿场（greenfield）与修复（repair）两组划分；公开的验证集、重放检查集和任务指纹。

**📈 对比分析**

对 7 种 LLM（Gemini‑3.6‑Flash、Gemini‑3.5‑Flash‑Lite、GPT‑5.6‑Terra、GPT‑5.6‑Luna、Claude‑Sonnet‑5、DeepSeek‑V4‑Pro、DeepSeek‑V4‑Flash）与 3 个零智策略（do‑nothing、spec‑transcription、generic‑stub）进行对比；核心指标为 Resolve Rate (RR) 与 Deployment Score (DS)。最佳模型 Gemini‑3.6‑Flash 的 RR 为 75%（DS 0.857），最差模型 Gemini‑3.5‑Flash‑Lite 为 52.9%（DS 0.719）。集成模型平均 92.6% 的任务被覆盖；人类专家在 25 题子集上达到 92% 的解析率。

**⚠️ 局限性**

① 每个模型仅跑一次，每格仅一条记录，无法估计多次运行的方差；② 部分任务因网络、调度或时序波动导致重放不稳定，当前未量化抖动；③ 任务集仍有限，未覆盖更复杂的多集群或云原生运维场景；④ 对某些规范缺失细节导致的误判（如任务披露缺失拉取阶段）未被自动发现。

---

## 386. PhyMo: A Physical-Field Modality for Multimodal AI4Physics

**arXiv ID:** 2609.27554 | [PDF](https://arxiv.org/pdf/2609.27554v1)

**作者:** Henan Sun `[一作]` (Hong Kong University of Science and Technology), Jia Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出物理场模态与三阶段训练框架 PhyMo，用以将物理测量通过 PDE 结构化后与视觉信息融合进行 AI4Physics 预测。

**💡 创新点**

创新点在于把物理测量视为独立模态，用 PDE 运算符令其在编码器中保持时空约束，并通过 PDE 残差监督实现物理驱动的表示学习。

**🔧 技术方法**

使用 Transformer 物理场编码器、PDE 残差自监督预训练、跨模态对齐（对比+回归）、多视窗融合与多时域预测头等技术。

**📊 数据集**

实验涵盖五大数据集：SKIPP'D、Folsom、NREL（光伏预测）、MeteoNet（气象预测）以及 Boreas（自动驾驶速度预测）。

**📈 对比分析**

与传统统计模型、时序模型和现有多模态基线（BiMamba、AstroCLIP、Maven、AION‑1）对比，PhyMo 在大多数数据集上均以更高的 R²（>84%）和更低的 RMSE/MAE 取得 SOTA 结果。

**⚠️ 局限性**

局限包括对已知 PDE 结构的依赖，无法直接处理无明确物理方程或高度随机的系统；对超参数的微调仍需一定人工干预；在更广泛科学领域的可推广性尚待验证。

---

## 387. Learning to Detect Symbolic Failure: Machine Learning and the Limits of Black-Scholes

**arXiv ID:** 2609.27764 | [PDF](https://arxiv.org/pdf/2609.27764v1)

**作者:** Juli Huang `[一作]` (Stanford University), Rupert Lu `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过对 260 万条真实期权合约价格与 Black‑Scholes 价格的比较，验证并比较了树模型、核PCA+SVM 以及神经网络在检测系统性定价偏差方面的表现。

**💡 创新点**

创新点在于：①在已有专家设计特征的金融域中证明保持结构的树模型比学习抽象嵌入（核PCA）高出 21.5 个百分点；②用神经网络对偏差的一致性验证，显示偏差并非 Black‑Scholes 的建模误差，而是市场结构所致。

**🔧 技术方法**

使用技术包括：梯度提升、随机森林等树基集成方法；核PCA（线性、RBF、Polynomial、Sigmoid）配合 SVM；MLP 回归做为独立验证器；同时采用 80/20 时间划分、5 折 TimeSeriesSplit 进行交叉验证。

**📊 数据集**

使用数据集为 2025‑2026 年 Yahoo Finance 的 2,632,013 条期权合约，涵盖 AAPL（25% IV）、SPY（15% IV）和 TSLA（45% IV）三种波动率 regime。

**📈 对比分析**

通过分类准确率比较：树模型 93.8%（平均），核PCA 72.3%（最优 Sigmoid），差距 21.5pp；神经网络与 Black‑Scholes 标签在 38,833 条测试样本上 99.9974% 一致；在不同波动率资产上梯度提升的准确率分别为 91.3%（AAPL）、98.2%（SPY）、82.7%（TSLA）。

**⚠️ 局限性**

局限性包括：未进行真实交易的盈利验证；±10% 的分类阈值人为设定；训练窗口仅 180 天，缺乏长期前瞻检验；无法证明检测到的偏差可被利用为交易机会；模型仅适用于欧式期权与已设计的 Greeks 特征，未涵盖美式期权或其他市场。

---

## 388. "What's That Sound?": A Versatile, Robust, and Lightweight Convolutional Transformer for Environment Sound Recognition

**arXiv ID:** 2609.27762 | [PDF](https://arxiv.org/pdf/2609.27762v1)

**作者:** Julia Huang `[一作]` `[通讯]` (Northville High School), Julia Huang (Northville High School)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种轻量级的随机化音频层卷积变压器（RALCT）用于环境声音识别。

**💡 创新点**

创新点是将CNN和Transformer结合，并使用随机化音频增强和混合MFCC与log‑mel谱，获得310K参数的轻量模型。

**🔧 技术方法**

采用了CNN、Transformer、随机化数据增强、MFCC、log‑mel谱以及TensorFlow Lite部署。

**📊 数据集**

使用了UrbanSound8K数据集。

**📈 对比分析**

与纯CNN、Transformer以及MhaNN‑SVM等方法比较，RALCT在约119轮后达到94.56%的验证精度，超过其他模型。

**⚠️ 局限性**

局限在于随机增强概率过高会导致内存崩溃，需更大RAM环境；在某些类别（儿童玩耍、街头音乐）精度略低。

---

## 389. Backdoors Leave Structural Traces: FedMAST for Backdoor Detection and Containment in Federated Learning

**arXiv ID:** 2609.27760 | [PDF](https://arxiv.org/pdf/2609.27760v1)

**作者:** Srinivasan Subramanian `[一作]` (Kennesaw State University), Md. Abdullah Al Hafiz Khan `[通讯]` (Kennesaw State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种服务器端防御框架 FedMAST，用多轴结构、历史与谱证据检测并隔离联邦学习中的隐蔽后门攻击。

**💡 创新点**

创新点包括：①同向对齐的 squeeze‑pair 互相关评分揭示配对特征的异常；②符号时间谱漂移累积捕捉弱而持久的谱方向变化；③历史与当轮一致性双重阈值构成 2-of-3 硬共识，配合回合级隔离与容器化。

**🔧 技术方法**

采用服务器端中位数聚合、MAD 标准化、EMA 累积、簇分割、簇漂移检查、锚定软硬评分，以及坐标中值、修剪均值等聚合策略。

**📊 数据集**

实验使用 CIFAR-10、EMNIST‑Balanced 与 FEMNIST，模型为 Tiny ResNet‑18（约 0.27M 参数）。

**📈 对比分析**

与 FedAvg、MultiKrum、AlignIns、FLAME、Trimmed Mean 等基线在六类隐蔽后门（Constrain‑Scale、Neurotoxin、BC‑Layers、LGA、DBA、3DFed）以及自适应 CovertLayers 攻击下对比，FedMAST 将 ASR 降至 1% 以下，主任务准确率维持约 95%，误拒率集中在少数分区。

**⚠️ 局限性**

局限性：在高异构（低 Dirichlet α）环境下误拒率上升且错误聚焦于少数分区；需依赖稳定伪匿名 ID；缺乏对更大模型、不同任务及长期自适应攻击的进一步验证。

---

## 390. Hard Negatives Reveal What Easy Negatives Hide: Cross-Lingual Harmfulness Representations Degrade with Resource Tier Under Hard Negatives

**arXiv ID:** 2609.27758 | [PDF](https://arxiv.org/pdf/2609.27758v1)

**作者:** Paras Balani `[一作]` (Birla Institute of Technology and Science, Pilani), Subhrakanta Panda `[通讯]` (Birla Institute of Technology and Science, Pilani)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估英语训练的有害性探测器在九种不同资源层级语言中的跨语言迁移效果，重点对比易负例与难负例两种负例集；

**💡 创新点**

证明负例选择对评估结果影响巨大：易负例下迁移几乎完美，而难负例下低资源语言的表现显著下降，揭示有害性表示在低资源语言中可能确实退化；

**🔧 技术方法**

采用跨语言探测（cross‑lingual probing）+逻辑回归线性探测器，对模型（Qwen2.5‑7B‑Instruct、Aya‑Expanse‑8B）的残差流进行激活提取；用back‑translation chrF评估翻译质量，利用Tokenizer Fertility分析其与迁移衰减的相关性；

**📊 数据集**

使用英文有害与无害提示（AdvBench、Alpaca、XSTest对照集），以及NLLB‑200翻译得到的九种语言（高资源：西班牙语、普通话、法语；中等资源：印地语、越南语、阿拉伯语；低资源：泰卢固语、斯瓦希里语、阿姆哈拉语）作为测试集；

**📈 对比分析**

与易负例相比，难负例导致低资源语言的AUROC平均下降约0.28（相较于高资源仅0.01），在两个模型中均得到一致趋势；翻译质量chrF与下降无显著线性关系，Tokenizer Fertility与下降呈正相关并解释了部分资源层级效应；

**⚠️ 局限性**

仅对两种模型、九种语言、两套负例集做评估，未验证其它模型或更广泛语言；Tokenizer Fertility相关性未达到统计显著，未能完整解释下降机制；实际生成行为未直接评估，探测器的有效性仍需结合真实对话测试。

---

## 391. Reporting Under Pressure: Separating Factual and Tonal Sycophancy in LLM Statistical Analysis

**arXiv ID:** 2609.27756 | [PDF](https://arxiv.org/pdf/2609.27756v1)

**作者:** Paras Balani `[一作]` (Birla Institute of Technology and Science), Subhrakanta Panda `[通讯]` (Birla Institute of Technology and Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究测试了在不同的编辑框架下，大型语言模型（LLM）对数据分析结果的报告是否会改变其语气和实质内容。通过4×4的实验设计，结合四种框架条件和四种真实数据模式，收集了480个响应，并对其进行了评分。

**💡 创新点**

创新点在于区分了模型的事实陈述变化与语气变化，发现事实错误主要集中在两种特定条件下，而语气变化则更为普遍，且与数据模式的真实情况无关。

**🔧 技术方法**

使用了大型语言模型（LLM）作为分析工具，并通过LLM评估模型的响应，采用了4×4的实验设计。

**📊 数据集**

使用了四种类型的合成数据集，分别代表真实效果、混淆效果、充分无效和不足无效的结果。

**📈 对比分析**

通过比较不同框架条件下的响应，发现事实错误主要集中在批判性框架与真实效果的组合，以及寻求显著性框架与不足无效的组合中。语气变化在所有数据模式中都更为普遍，批判性框架导致的语气变化在每种数据类型中都显著。

**⚠️ 局限性**

限制在于数据本身的混淆因素几乎完全阻止了两种类型的变化，表明分析灵活性而非单纯的压力决定了模型报告结论的可变性。

---

## 392. Satisfaction Is Not Explanation: Auditing Vacuity and Training Influence in Temporal-Logic-Guided Reinforcement Learning

**arXiv ID:** 2609.27743 | [PDF](https://arxiv.org/pdf/2609.27743v1)

**作者:** Lorenzo Bacchiani `[一作]` `[通讯]`, Lorenzo Bacchiani

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一层审计机制，针对强化学习策略满足给定的有限时序逻辑（LTLf）规范时，分析每个子句在学习到的策略中是否真正起作用、如何起作用，以及背后的原因。

**💡 创新点**

创新点包括：① 将审计问题分解为三大问（是否满足、子句是否起作用、原因是什么），并将其映射到六个可量化的监管范式；② 定义了条件真值 vacuity、环境底线、匹配参考等诊断指标；③ 证明在常见的弱化子句实验中训练影响可能无法被检测，提出了有效性筛选（语法冲突、环境等价、可退化等）与改进的干预方法；④ 在标准基准与公开 RL 任务上验证了这些范式的多样性。

**🔧 技术方法**

使用的技术包括：LTLf over finite traces + 进程化监视器；离散 MDP 的动态规划求最优；Q‑learning 与 SARSA 训练；基于条件真值的 vacuity 计算；线性规划求环境底线；匹配参考集合比较；训练影响有效性筛选与改进的干预实验。

**📊 数据集**

数据集与任务：
- 标准 RL 基准：FrozenLake、Taxi、OfficeWorld（离散网格）；
- Shield‑RL 公开例子：Water Tank、SGW9、SGW15；
- AI Safety Gridworlds：Absent Supervisor、Safe Interruptibility、Whisky‑and‑Gold；
- 公开时序逻辑规范：DeepLTL（149 句）+ Reward Machines（24 句）。
在每个任务上采用 50 个随机种子，并对不同奖励机制进行实验。

**📈 对比分析**

比较方法：对每个基准在三种奖励机制（任务奖励、稀疏接受、接受距离塑造）下分别训练 50 种种子，计算满足概率、任务回报、条件 vacuity、环境底线、匹配参考等指标。性能结果表明：
- 在 FrozenLake 等任务中出现匹配过剩（matched excess）现象；
- 在 Taxi、OfficeWorld 等任务中出现不变性（invariance）和衰减（attenuation）；
- 在 Safe Interruptibility 等任务中出现奖励/满足度权衡；
- 在 Shield‑RL 与 AI Safety Gridworlds 中出现学习可行性/稳定性分裂；
- 在公开规范集中发现大量弱化子句的失效（ablation hazard）。
该审计层能将原本只给出成功/失败的单一指标拆分为六类更细粒度的结论。

**⚠️ 局限性**

限制：
- 仅适用于有限、离散 MDP；难以直接扩展到大规模或连续动作空间；
- 基准规模相对较小，缺少深度 RL 或连续控制的验证；
- 需要精确的 MDP 解析与完整的状态标签，实际复杂环境中可能难以满足；
- 训练影响有效性筛选依赖于语法规则，可能无法捕捉所有逻辑变形；
- 评估依赖于 50 个种子，结果可能仍受随机性影响。

---

## 393. Open Questions Towards Skill-Sustaining Reliance in Reflective AI Engagement

**arXiv ID:** 2609.27726 | [PDF](https://arxiv.org/pdf/2609.27726v1)

**作者:** Sander de Jong `[一作]` `[通讯]` (Aalborg University), Sander de Jong (Aalborg University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨在专业工作中持续使用反思机制以维持人工智能辅助决策中的人类专业能力与自治的挑战与问题

**💡 创新点**

提出四大核心问题：反思支持何时削弱自主性、在效率驱动的流程中如何维持反思、长期使用是否会导致技能流失、以及短期实验与长期效益的差距如何评估

**🔧 技术方法**

主要关注认知强制、LLM生成推理说明、Socratic式提问、以及解释与不确定性沟通等交互技术

**📊 数据集**

无具体数据集，文章基于文献综述与案例分析

**📈 对比分析**

未进行实验对比，讨论主要是基于已有研究的综述与推理，没有可量化性能指标

**⚠️ 局限性**

局限在于缺乏长期实验验证、对组织与个人层面的可操作性不足、对技术实现细节和数据驱动评估的缺失

---

## 394. Wave-Robust Passive AUV Localization Using FP-MUSIC

**arXiv ID:** 2609.27712 | [PDF](https://arxiv.org/pdf/2609.27712v1)

**作者:** Usama Saqib `[一作]` (IT University of Copenhagen), Andrzej Wąsowski `[通讯]` (IT University of Copenhagen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于浮动浮标的单向被动声学定位系统，利用水面上布置的水下声呐阵列和IMU实现AUV的三维定位与姿态估计。

**💡 创新点**

创新点在于提出FP-MUSIC算法——利用IMU测得的波浪诱发的六自由度旋转信息进行迭代固定点去畸变，恢复MUSIC子空间的高分辨率；并证明平移不影响协方差，从而仅需补偿旋转。

**🔧 技术方法**

采用了MUSIC、MVDR、DAS等传统空间处理方法，配合子空间投影宽带匹配滤波和功率不对称标记，并使用IMU驱动的固定点迭代去畸变。

**📊 数据集**

使用Pyroomacoustics仿真生成的多海况（Sea State 1-7）数据集，模拟4×6 URA 24个麦克风、LFM 7.5-15 kHz脉冲、5秒帧、16快照，并加入海浪六自由度运动。

**📈 对比分析**

与标准MUSIC、MVDR、DAS对比，FP-MUSIC在Sea State 1时将DOA RMSE从0.61°/0.57°降至0.25°/0.02°，在2 m beacon分离时定位成功率提升至约75%，轨迹重建误差减半。

**⚠️ 局限性**

局限性包括仅考虑直接波传播、未包含多径、IMU姿态误差未实验验证、仅单AUV两 beacon 场景、对实时硬件性能和实地海况的验证仍待开展。

---

## 395. Coloring Queens with Thousands of Encodings

**arXiv ID:** 2609.27674 | [PDF](https://arxiv.org/pdf/2609.27674v1)

**作者:** Bernardo Subercaseaux `[一作]` (Carnegie Mellon University), Marijn J. H. Heule `[通讯]` (Carnegie Mellon University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在Knuth的基础上，对皇后图着色问题的CNF编码做了大规模系统实验，生成并比较了1584种不同编码方案；

**💡 创新点**

创新点在于：①以千级编码量化影响因素，揭示对SAT求解器性能至关重要的编码设计；②发现对称性破坏、clique提示、以及精确单色约束是关键因素；③证明即使在传统认为优选的order编码中，加入clique提示也可显著提升；④首次系统性分析对称性破坏方法的相对效益和不对称性引起的脆弱性；

**🔧 技术方法**

采用CNF编码技术（one-hot、order、递归计数、块子句）、对称性破坏约束、clique提示、随机变量/子句重排，使用CryptoMiniSat等主流SAT求解器；

**📊 数据集**

实验数据来源于8、9、10阶皇后图（n=8,9,10）及其相应的颜色数（n或n+1）问题；

**📈 对比分析**

通过统计回归、箱形图、散点图等方式比较不同编码的求解时间；结果显示：最佳编码可将n=10问题从数百秒压缩到几秒，平均性能提升幅度可达数十倍，clique提示的概率调节对运行时呈指数下降；

**⚠️ 局限性**

局限性包括：仅针对皇后图着色；求解器参数固定，缺乏与求解器设置交互的分析；缺乏理论解释为何某些对称性破坏方法更优，导致对新问题的推广仍需经验性探索。

---

## 396. Topology optimization of multimaterial aircraft pylons using generalized shape function approach

**arXiv ID:** 2609.27685 | [PDF](https://arxiv.org/pdf/2609.27685v1)

**作者:** Swagatam Islam Sarkar `[一作]` (Indian Institute of Technology Hyderabad), Prabhat Kumar `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文利用多材料拓扑优化方法，设计了航空发动机支架（pylon）的最佳结构布局，并将二维最优拓扑外推到三维，实现了多材料结构的完整建模。

**💡 创新点**

创新点在于引入通用形函数(gSF)方法，能够仅用n个设计变量描述最多2^n-1种材料，并结合密度滤波与Heaviside投影实现近离散解；同时针对不同材料数量采用二维/三维/四维形函数，显著降低计算成本。

**🔧 技术方法**

主要技术包括gSF多材料插值、SIMP密度插值、密度滤波、投影滤波、Method of Moving Asymptotes(MMA)优化器及有限元分析。

**📊 数据集**

采用了14种候选材料的归一化弹性模量数据，加载为底部均布力、右侧30%固定，设计域为9×4.5单元的二维平面；无公开数据集，全部为实验设定。

**📈 对比分析**

通过比较不同材料数（2–14）下的目标函数f₀、非离散度M_nd和收敛曲线，证明了gSF方法在保持低M_nd（≤3.8%）和快速收敛（≤400迭代）方面的有效性，且相对于传统单材料方法可获得更轻、更刚的结构。

**⚠️ 局限性**

局限性包括：未考虑多目标约束（如疲劳、热膨胀）；仅在单一静载荷条件下验证；二维结果外推为三维假设对称性，未进行全三维优化或制造可行性评估；材料属性仅采用归一化弹性模量，缺乏真实材料性能数据。

---

## 397. Faster Minimum k-Cut I: Simple and Sparse Weighted Graphs

**arXiv ID:** 2609.27781 | [PDF](https://arxiv.org/pdf/2609.27781v1)

**作者:** Jason Li `[一作]` (Carnegie Mellon University), Trevor Vaughn `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了在简单无权图上求最小 k‑cut 的新算法，首次实现了指数小于 1 的子多项式时间，给出了 k=3、4、5 的具体时间上界，并给出 k≥6 的通用上界。

**💡 创新点**

核心创新包括：1) 随机（或确定性）权重扰动保证存在“严格轻边”侧；2) 通过树包（tree packing）与树采样快速枚举潜在切口；3) 利用近线性时间构造的边不可分解树分解（edge‑unbreakable decomposition）实现 FPT 算法；4) 结合边界/岛屿框架和矩阵乘法进一步压缩求解空间。

**🔧 技术方法**

主要技术手段包括：随机扰动与格 lexicographic 扰动；树包与对数规模树采样；近线性构造的层次化稀疏化树和原子分解；递归的 k‑cut DP；边界/岛屿分离与矩阵乘法（尤其是矩形矩阵乘法）来恢复单点岛屿。

**📊 数据集**

论文为理论算法，未使用具体实验数据集；所有结果均为渐进复杂度分析，适用于任意简单无权图。

**📈 对比分析**

与之前的 O(n^k(log n)^O(k^2)) 算法相比，本工作将指数从 k 降至 6/7k‑0.132…+O(1/k)，在 k=3、4、5 时实现了 2、≈2.8947、≈4.112 的精确时间上界；通过实验验证的部分实现显示在中等规模图上已优于经典随机收缩法。

**⚠️ 局限性**

局限性：算法复杂度高，常数因子巨大；依赖高概率成功，需多次随机采样；实现对大 k 的实用性有限；仅针对简单无权图，通用加权图仍保持 n^k 上界。

---

## 398. Fusion-Aware Direct 3D Gaussian Generation with Structured Patch Latent Flows

**arXiv ID:** 2609.27779 | [PDF](https://arxiv.org/pdf/2609.27779v1)

**作者:** Yizhao Wang `[一作]` (Henan Institute of Science and Technology), Guantao Zhang `[通讯]` (Henan Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种层次化高斯补丁表示，并在此基础上开发结构感知的校正流模型，实现直接基于类别标签的3D高斯物体生成。

**💡 创新点**

创新点包括：①将无序、可变大小的高斯原语划分为共享锚点的局部补丁并编码为结构化token；②构建全局-局部双层潜在空间；③设计带补丁位置编码、全局-局部耦合和密度感知加权的校正流网络；④引入渲染反馈学习，使潜在空间生成与多视角渲染质量一致。

**🔧 技术方法**

主要技术包括：3D高斯Splatting表示、变分自编码器、结构感知校正流（rectified flow）、自注意力机制、位置编码、密度加权、渲染反馈（多视角渲染与SSIM/L1损失）以及两阶段训练策略。

**📊 数据集**

使用融合自ShapeNetCore、Objaverse、ABO、3D-FUTURE和InteriorGS等公开3D数据集，构建包含34类的统一分类训练集。

**📈 对比分析**

与GaussianCube、DiffGS、SplatFlow等3DGS生成基线以及DiT-3DGS、U-ViT-3DGS、Vanilla RF-3DGS等图像生成网络改造的基线进行对比。实验表明本文在类别一致性、几何精度、渲染质量、多视角一致性以及推理效率（3.8 s/物体，5.2 GB）上均超过所有基线，尤其在Chamfer Distance、LPIPS和多视角一致性指标上显著提升。

**⚠️ 局限性**

局限性包括：仅支持基于类别标签的生成，无法直接进行文本或图像条件生成；依赖于对象级的3DGS预处理和共享锚点，难以处理高度复杂场景或极薄/透明/反射物体；对极端稀疏或拓扑复杂的物体仍可能出现位置漂移或细节缺失。

---

## 399. Improving LLM-based Autonomous Web Agents with Filtering

**arXiv ID:** 2609.27770 | [PDF](https://arxiv.org/pdf/2609.27770v1)

**作者:** Zhitong Guo `[一作]` (Carnegie Mellon University), Ruiyu Li `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究针对网页自动化代理，提出并实现了基于检索的HTML元素过滤策略，以减少LLM输入上下文冗余并提升任务成功率。

**💡 创新点**

创新点在于将已训练的DeBERTa和T5模型迁移到WebArena环境中，并引入零样本ColBERT检索，对网页可访问性树进行内容过滤；同时通过对检索结果的窗口化策略（邻接窗口和边界窗口）进一步细化输入。

**🔧 技术方法**

使用技术包括：Transformer预训练模型（DeBERTa、T5、ColBERT），对Mind2Web轨迹数据进行二分类微调，LLM代理（GPT‑3.5、LLaMA‑2‑70B）以及基于检索的上下文筛选与排名机制。

**📊 数据集**

主要数据集为Mind2Web（用于微调检索器）和WebArena（用于评估LLM代理性能），并对WebArena轨迹进行手工清洗以验证检索效果。

**📈 对比分析**

方法与基线比较：在WebArena上，原始LLaMA‑2‑70B成功率为1.97%，引入DeBERTa检索后提升至2.96%；与GPT‑3.5基线相比，检索对LLM性能影响有限，主要受限于GPT‑3.5上下文长度足以覆盖原始树。

**⚠️ 局限性**

限制包括：检索器在WebArena上的召回率仅约0.63，导致部分关键元素被误过滤；对GPT‑3.5效果不明显；未针对检索误差进行鲁棒性改进，且仅在模拟环境中验证，尚未测试开放域网页。

---

## 400. Alignment of LRMs via Counter-Aligned Few-Shot Conversation Exposure

**arXiv ID:** 2609.27763 | [PDF](https://arxiv.org/pdf/2609.27763v1)

**作者:** Xiangyu Zhou `[一作]` (Wayne State University), Dongxiao Zhu `[通讯]` (Wayne State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型推理模型（LRMs）在长上下文、少量示例对话中的脆弱性，提出了通过在对话历史中插入反向对齐的链式推理（CoT）实现的攻击方法SRCF，并在此基础上设计了对抗后训练框架ARCF，用以提升模型在遭受此类攻击时的安全性与有用性；

**💡 创新点**

创新点在于①首次系统揭示“对齐与反对齐”CoT在LRMs推理中的可攻击性；②提出SRCF攻击，使模型仅通过对话历史即可被诱导产生不安全或过度拒绝；③构建ARCF后训练框架，将攻击样本作为训练数据，强制模型恢复对齐推理，从而在不降低性能的前提下提升安全与有用性；

**🔧 技术方法**

技术手段包括链式推理（CoT）长上下文对话、对抗式提示工程（advICL、Response-Only）、表示漂移分析（PCA、余弦相似度）、后训练策略（SFT、GRPO）以及对齐训练框架ARCF；

**📊 数据集**

使用的数据集包括：安全性评估的SafeChain、FORTRESS、H-CoT、Prefill、OR-Bench、AdvBench；有用性评估的GSM8K、AIME-25、MMLU-Pro；ARCF训练采用2K条样本（1K有害、1K过度拒绝）；

**📈 对比分析**

与advICL和Response-Only基线相比，SRCF在多模型、不同shot下显著提升不安全率与拒绝率；在后训练阶段，ARCF（尤其是GRPO-ARCF）在安全性、拒绝率与通用推理任务上均优于原始模型和无ARCF的对照组，且保持或略升高通用指标；

**⚠️ 局限性**

局限性包括：①防御依赖于后训练，无法即时阻止新型攻击；②对抗样本仅覆盖特定对齐模式，未知是否能对更广泛或更细粒度的攻击做泛化；③评估主要在公开数据集与模拟查询，真实场景中对话历史多样性可能导致效果不完全可迁移。

---

## 401. NS-ATTENTION: Newton-Schulz Transformations of Attention Outputs in Vision Transformers

**arXiv ID:** 2609.27735 | [PDF](https://arxiv.org/pdf/2609.27735v1)

**作者:** Xiaohe Jiang `[一作]` (University of Exeter), Ronghui Mu `[通讯]` (University of Exeter)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在Transformer的每个注意力头输出上引入无参数的Newton–Schulz迭代变换（NS-Attn），通过调整特征‑token矩阵的谱分布来提升分类性能。

**💡 创新点**

创新点是将Newton–Schulz多项式直接作用于每个注意力头的输出，降低谱集中度、提高有效秩，从而在不增加可训练参数的前提下改进模型表达能力。

**🔧 技术方法**

采用Newton–Schulz多项式变换、Frobenius范数归一化、Vision Transformer与Swin Transformer网络架构，以及标准的图像分类训练技术。

**📊 数据集**

实验使用CIFAR‑10和CIFAR‑100这两类小规模图像分类数据集。

**📈 对比分析**

在相同训练配置下将NS‑Attn与标准Attention进行对比，报告所有种子下的最终Top‑1准确率；NS‑Attn在所有12个匹配种子上均取得提升，平均增益为0.25–0.83个百分点，且所有对比均显著。

**⚠️ 局限性**

主要局限在于额外的前向推理延迟显著增加（ViT约+41%，Swin约+104%），且实验仅在CIFAR级别的小规模数据集上验证，未探讨更大规模数据或等算力条件下的效果。

---

## 402. SkillGym: Internalizing Human Skills into LLMs for Real-World Problem Solving

**arXiv ID:** 2609.27717 | [PDF](https://arxiv.org/pdf/2609.27717v1)

**作者:** Zhilong Ge `[一作]` (East China Normal University), Liang He `[通讯]` (East China Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 SkillGym 框架，将人类手写的代理技能转化为可执行、可验证的训练环境，并收集了 8,364 条成功轨迹供监督微调；

**💡 创新点**

创新点在于将技能文档映射为可执行任务并通过代码检查器和对比验证确定技能依赖，从而生成大规模可验证轨迹与环境，实现可重用的流程能力训练；

**🔧 技术方法**

采用模板化任务构建、Docker 环境化、代码检查器、对比验证、跨模型多抓取轨迹采样，并进行监督微调与基于结果的强化学习；

**📊 数据集**

使用来自公开技能注册库构建的 2,756 个任务环境（12 类/63 子类），收集 8,364 条轨迹；基准评测使用 GDPval‑AA v2、Terminal‑Bench 2.1、SkillsBench v1.1；

**📈 对比分析**

通过与基线 Qwen3.5‑35B‑A3B 以及同等规模和公开 SOTA 进行 Elo/成功率对比；在 Claude Code 上提升 199 Elo、Terminal‑Bench +19.10%、SkillsBench +28.13/12.38%；整体性能超过多种对手；

**⚠️ 局限性**

局限在于任务构建耗时、对比验证仅依赖单一参考 agent，技能依赖环境占比仅 39%，未深入强化学习实验，跨技能依赖的证据仍有限。

---

## 403. Consequential Behaviour and Representational Fairness in the Validation of Synthetic Research

**arXiv ID:** 2609.27690 | [PDF](https://arxiv.org/pdf/2609.27690v1)

**作者:** Florian Kutzner `[一作]` (decision-context), James Kunling He `[通讯]` (Artificial Societies Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了验证合成调查受访者（synthetic respondents）的一套完整框架，强调以真实行为而非仅仅是问卷回答为核心的预测有效性。

**💡 创新点**

创新点在于将行为预测（C）作为首要验证指标，强制子组层级报告，量化分配、程序与认知三维正义，并引入 within‑persona 对照实验以评估因果效应。

**🔧 技术方法**

技术手段主要是利用大型语言模型（LLM）生成 persona，并通过四级诊断（L0–L3）与实验一致性（E）的统计量（如差距、校准、KS 距离、相关矩阵相似度等）进行评估。

**📊 数据集**

使用的基准数据来自已有的人类调查与行为记录（例如电动汽车充电时段、消费者问卷等），并未自行收集新的数据集。

**📈 对比分析**

比较方法是与人类基准（人类样本大小、分层置信区间、校准阈值等）进行差距检验，若误差在预设阈值内即认为通过；框架未给出数值性能指标，而是提供一系列可操作的通过标准。

**⚠️ 局限性**

局限性包括训练数据污染难以检测、子组样本稀缺导致验证难度、行为标准本身的局限、模型版本更新导致验证失效、以及对未观测因果效应的假设与对人类数据依赖的约束。

---

## 404. Test-Time Adaptation with Query-Dependent Residuals for Visual Document Retrieval

**arXiv ID:** 2609.27688 | [PDF](https://arxiv.org/pdf/2609.27688v1)

**作者:** Zeliang Li `[一作]` (South China University Of Technology), Xiangmin Xu `[通讯]` (South China University Of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉文档检索中，提出Q-REACT方法，通过查询端的低秩适配器和文档上下文先验，将有限的重排序器反馈转化为可重用的检索改进；

**💡 创新点**

创新点在于（1）查询相关低秩变换生成可变残差，适配多样查询；（2）利用完整页面索引进行蒸馏，充分利用未评分页面；（3）结合文档结构上下文为稀缺反馈提供结构性先验；

**🔧 技术方法**

使用低秩线性变换、文档上下文加权评分、全索引分布蒸馏以及正则化残差；

**📊 数据集**

在ViDoRe V3数据集上进行实验，涵盖14,514个查询、19,256页、8个检索任务；

**📈 对比分析**

与基线（Base、直接重排序、TTT-Embed、GQR）及多种奖励模型比较，Q-REACT在稀疏预算下提升≈1.8–3点nDCG@10，在完整预算下提升≈6–8点，且推理延迟仅比基线提升2–3%；

**⚠️ 局限性**

局限包括：需要预先获得有限的重排序器反馈，低秩变换容量受限，针对极端稀疏反馈或非结构化页面效果未知；

---

## 405. NeuralSRNF: Neural Square Root Normal Fields for the Statistical Shape Analysis and Generation of Nonrigid 3D and 4D Objects

**arXiv ID:** 2609.27728 | [PDF](https://arxiv.org/pdf/2609.27728v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 406. CoRelNav: Collaborative Relational Navigation for Multi-Robot Spatially Constrained Semantic Navigation

**arXiv ID:** 2609.27720 | [PDF](https://arxiv.org/pdf/2609.27720v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 407. AWM-VLA: AlignedWorld Modeling for Efficient and Explainable Vision-Language-Action Policies

**arXiv ID:** 2609.27753 | [PDF](https://arxiv.org/pdf/2609.27753v1)

**作者:** An Lanji `[一作]` (University of Electronic Science and Technology of China), Yu Tian `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 Vision‑Language‑Action（VLA）框架中引入对齐的世界模型（AWM‑VLA），使得机器人策略能够通过可学习的未来 token 预测隐层未来状态并生成对象级可解释的推理。

**💡 创新点**

① 在 diffusion‑transformer 内部嵌入可学习未来 token 并实现全局与对象级对齐；② 引入对象级解耦对齐，提升可解释性与对多指令/多对象的泛化；③ 使用多目标优化（MGDA）实现动作精度与世界模型解释性的可调平衡。

**🔧 技术方法**

基于 diffusion/flow‑matching 策略、Future Latent Representation Alignment（FLARE）原理、可学习未来 token、对象级解码器、MGDA 权重平衡、EMA 编码器更新等技术。

**📊 数据集**

主要使用 RoboCasa（模拟厨房单臂任务）和 humanoid tabletop manipulation benchmark（双臂桌面任务）进行评估，预训练使用 Open X‑Embodiment 等跨体感知数据集。

**📈 对比分析**

与 diffusion policy、GR00T N1、无世界模型 VLA、FLARE、UWM、Grad‑CAM 等基线比较；在 RoboCasa 上成功率 58.9%（比 FLARE 高 5.7%），在 humanoid 上 52.6%；在新对象和新指令的泛化上也显著优于基线；人类评估显示对象级解释的偏好率达 83.1%。

**⚠️ 局限性**

需要先验或可提取的对象集合，无法直接处理开放词汇；MGDA 计算开销略大；目前仅预测对象级语义，未覆盖对象间关系；对未来观测的依赖在无标签视频或实时场景下可能受限。

---

## 408. Evaluation of pre-trained models for pedagogical assessment of novel AI-assisted educational questions

**arXiv ID:** 2609.27749 | [PDF](https://arxiv.org/pdf/2609.27749v1)

**作者:** Michael Lawrence Castanares `[一作]` (Predictive Systems Inc), Allan Tan `[通讯]` (Predictive Systems Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估传统机器学习、Transformer 及大型语言模型在 AI 辅助生成教育问题（AEQ）上的 Bloom 分类器在分布外（OOD）数据上的表现，并通过特征工程、文本拼接、学习目标拼接及模型再训练等方法提升 OOD 性能。

**💡 创新点**

① 系统研究 AEQ 文本长度导致的 Bloom 信号稀释及其对模型性能的影响；② 引入文本拼接和学习目标拼接两种特征增强策略，显著提升 OOD 表现；③ 在同一实验框架下对传统 ML、BERT 与 LLM 进行对比，提出针对性提升路径。

**🔧 技术方法**

TFPOS‑IDF、SVM、XGBoost、DistilBERT、GPT‑4.1、Gemini Flash 3.1；零/少量提示、文本拼接、学习目标拼接、模型再训练；使用 LIME 进行局部解释。

**📊 数据集**

IID 数据集（Lau 等 6,175 条）与两类 OOD 数据集：Scaria（1,833 条）与 Asyncform（AF，863 条）AEQ。

**📈 对比分析**

采用 5 折交叉验证，宏 F1 分数作为评估指标。Baseline TFPOS‑IDF 在 IID 达 0.88，OOS 下降至 0.48；BERT 在 IID 0.89，OOS 0.55；LLM 在 OOS 0.41–0.79。文本拼接后 XGBoost 提升至 0.59，BERT 提升至 0.62；再训练后 XGBoost 在 Scaria 达 0.82，BERT 0.77；在 AF 上 BERT 由 0.30 提升至 0.50。

**⚠️ 局限性**

① OOD 数据样本规模有限，尤其 AF；② 高阶 Bloom 级别的预测仍存在偏差；③ 仅测试少量模型与提示方式；④ 对学习目标的依赖导致标注成本高；⑤ 模型对文本长度敏感，仍需改进鲁棒性。

---

## 409. Less Language, More Latents: Annotation-Efficient VLAs for Driving

**arXiv ID:** 2609.27747 | [PDF](https://arxiv.org/pdf/2609.27747v1)

**作者:** Alexey Zakharov `[一作]` (Robert Bosch Gmbh), Puneet K. Dokania `[通讯]` (Robert Bosch Gmbh)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究提出LADA三阶段流程，将少量语言标签与大量未标记的观察-轨迹对结合，训练语言条件驾驶模型。

**💡 创新点**

创新点在于先用无监督潜在动作模型提取高层意图代码书，再通过少量语言标签将命令映射到该代码，突破语言标注瓶颈。

**🔧 技术方法**

核心技术包括向量量化潜在动作模型（VQ-LAM）、视觉语言翻译器（VLM）、行为克隆训练以及动作梦境的反事实监督。

**📊 数据集**

采用SimLingo收集的CARLA模拟驾驶数据（约140小时、2M帧），包含完整的HLC、CoT、VQA、CF标签。

**📈 对比分析**

与完整语言监督、HLC分类器、几何聚类等基线在Bench2Drive闭环评测对比，LADA仅用5%标签即可实现87.98的Driving Score、70.46%成功率，优于全标注模型。

**⚠️ 局限性**

限制包括仅在仿真环境验证，缺乏对真实世界噪声与域迁移的评估；反事实监督依赖模拟特有状态，难以在真实数据上复制；去除VQA/CoT导致可解释性与问题回答能力下降。

---

## 410. GLoTouch: Global-to-Local Haptic Perception Using a Parallel Gripper for Object Search, Recognition, and Grasping Without External Vision

**arXiv ID:** 2609.27695 | [PDF](https://arxiv.org/pdf/2609.27695v1)

**作者:** Zonglin Li `[一作]` (Independent Researcher), Daolin Ma `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了GLoTouch框架，使用并行抓手在黑暗环境中仅靠触觉完成全局探测、局部识别和抓取。

**💡 创新点**

将长探针与并行抓手结合，实现全局力感探测，并使用同一抓手的双目触觉做训练无关的有限窗口匹配，单一硬件完成从全局到局部的触觉管控。

**🔧 技术方法**

利用力/扭矩传感、探针几何感知、局部深度映射、基于STL的有限窗口宽度匹配以及覆盖驱动搜索与轨迹规划等技术。

**📊 数据集**

使用五个3D打印物体（Wukong、Nailong、Apple、Train、Truck）在多物体容器中的随机布局，模拟与真实机器人共计120次试验。

**📈 对比分析**

与随机下降基线对比，覆盖驱动搜索在5 m预算下完成率从30%提升至55%；在模拟中成功率84%，真实机器人中76%，局部匹配Top‑1准确率90%。

**⚠️ 局限性**

需预知目标STL、仅适用于可从顶部访问的物体、粗糙轮廓仅限圆盒、未利用RGB触觉信息，且对形状相近的Train/Truck易混淆且抓取稳定性有限。

---

## 411. Gender Bias in Vision-Language In-Context Learning

**arXiv ID:** 2609.27682 | [PDF](https://arxiv.org/pdf/2609.27682v1)

**作者:** Tong Xiang `[一作]` (University of Osaka), Yuta Nakashima `[通讯]` (University of Osaka)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对大规模视觉语言模型的上下文学习（ICL），系统评估并量化其对性别偏差的影响，提出完整的评估框架，并探索基于合成图像的偏差缓解方法。

**💡 创新点**

①揭示性别化ICL示例会形成方向性偏差，通过跨性别机制使模型对非示例性别性能下降；②指出该效应仅在输出涉及性别语言时出现；③首次提出用稳定扩散模型生成的合成图像替代真实示例，在不降低生成质量的前提下显著减轻偏差。

**🔧 技术方法**

使用大规模视觉语言模型（Qwen‑VL、Idefics3‑8B‑Llama3、Phi‑3.5‑vision‑instruct、MiniCPM‑o‑2.6、InternVL3.5‑8B、Qwen3‑VL‑8B‑Instruct）进行k‑shot ICL；引入相似性检索（SIIR、SITR）与合成图像生成（Stable Diffusion‑3.5‑Large、FLUX）等技术；采用误差率、BLEU‑4、CLIPScore、显露率等指标进行评估。

**📊 数据集**

四个含性别标签的公开数据集：Occupation‑Object / Occupation‑Participant（COVP）、MSCOCO 子集、Densely Captioned Images（SA‑1B）、GQA（VQA任务）。

**📈 对比分析**

通过在不同ICL设置（随机、男性/女性单性别、平衡、相似性检索）下对六个模型进行实验，比较误差率差异（ER_m‑f）和显露率。实验显示，男性/女性单性别ICL可将偏差推向对应性别，且在图像描述与代词预测任务中偏差显著放大；合成图像示例在保持CLIPScore不变的同时，将ER_m‑f的绝对值降低约30–50%，显示显著的偏差缓解效果。

**⚠️ 局限性**

局限性：仅考虑二元性别，忽视更广泛的社会身份；评估集中在性别化输出任务，对非性别化任务效果不明；合成图像可能降低显露率，从而影响对偏差的真实评估；实验使用的模型和数据集主要为英语，跨语言或更大范围的通用性尚未验证。

---

## 412. Task-Prototype Guided Flow Matching for Few-Shot Generalization in Vision-Language Robot Manipulation

**arXiv ID:** 2609.27780 | [PDF](https://arxiv.org/pdf/2609.27780v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 413. Optimal Weighting of Training Data in Adaptive Coding

**arXiv ID:** 2609.27783 | [PDF](https://arxiv.org/pdf/2609.27783v1)

**作者:** Yuriy Reznik `[一作]` `[通讯]` (Massachusetts Institute of Technology), Yuriy Reznik (Massachusetts Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并分析了一种在训练数据与待编码源不匹配时通过调节权重 ξ 来决定对训练数据信任程度的自适应 KT 编码器。

**💡 创新点**

创新点在于将权重 ξ 作为设计变量，引入 KT 估计器的缩放，并推导出闭式最优权重 ξ* = d/(2ℓD + d)，随后给出离线估计、反馈估计和两重普适混合三种实现，理论证明权重调节优于传统截断并显著降低冗余。

**🔧 技术方法**

使用了 Bayesian 的权重先验（power prior）、Krichevsky–Trofimov（KT）估计、Jensen–Shannon 散度展开、信息冗余分析、离线散度估计、反馈循环和两重普适混合（对 ξ 取几何格点）等技术。

**📊 数据集**

实验数据集为 Calgyar 和 Canterbury 语料库中的十篇英文文本文件（Calgary 的 1–5 及 Canterbury 的 6–10）。

**📈 对比分析**

与标准 KT 代码（ξ=0）和全权训练（ξ=1）进行对比，结果显示在所有文件上，权重调节方法均能将冗余降低多达 22.7%，并优于两端点；三种实现方式在相同条件下表现一致，混合方案仅额外付出几比特的开销。

**⚠️ 局限性**

局限性包括：需要估计 KL 散度 D（虽对误差容忍度大，但极短或极不匹配情形仍需额外开销）；混合方案的时间复杂度提升 O(logℓ)；目前仅针对无记忆源，扩展到上下文或高阶模型仍需进一步研究。

---

## 414. Beyond Unsafe Detection: Counterfactually Anchored Evidence Attribution for Multi-Turn LLM Safety Failures

**arXiv ID:** 2609.27773 | [PDF](https://arxiv.org/pdf/2609.27773v1)

**作者:** Srinivasan Subramanian `[一作]` (Kennesaw State University), Md. Abdullah Al Hafiz Khan `[通讯]` (Kennesaw State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了多轮对话安全评估数据集，并提出轻量级分层归因模型，用以定位导致LLM安全失效的用户词语与对话回合。

**💡 创新点**

将安全失效归因转化为对话级定位任务；采用交互式生成、交叉模型验证与分层因果/行为监督构建数据；设计层次归因网络和删除效应评估指标。

**🔧 技术方法**

使用DeBERTa‑v3‑base回合编码器，跨回合Transformer，门控融合，逆向删除与一致性训练，配合对比标准归因方法与多轮基线。

**📊 数据集**

共1,762条多轮对话（549对抗、1,213友好），包含对抗对照、边界友好及高危词汇变体；外部评估使用AdvBench、HarmBench、MHJ。

**📈 对比分析**

与surface‑risk、Grad×Input、Integrated Gradients、Attention等归因方法对比；检测F1=0.988，归因F1=0.878，删除效果DD@15=0.511，特异性FPR显著降低，且在外部数据上保持良好迁移。

**⚠️ 局限性**

局限性：生成对话覆盖有限、仅少量因果验证、仅对用户回合归因、对外部检测泛化不佳，且未在更大规模或多模态模型上验证。

---

## 415. Limiting-Kernel Q($λ$): Bridging Short and Long Horizons

**arXiv ID:** 2609.27741 | [PDF](https://arxiv.org/pdf/2609.27741v1)

**作者:** Tolga Ok `[一作]` (Delft University of Technology), Mohamad Amin Sharifi Kolarijani `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新型值估计器 Limiting‑Kernel Q(λ) (LKQL)，在 n‑step λ‑return 估计器中加入 limiting‑kernel (LK) 近似来完成长尾，形成尾补充（tail‑completion）算子；实现双备份算法，兼容离散和连续状态空间的 actor‑critic 框架。

**💡 创新点**

创新点在于：① 用 LK 预条件器替代传统的截断预条件器，既保持了 n‑step 的局部性，又能近似全局长期结构；② 通过双备份同时维护 Q 与 LK 项，直接在参数化值函数中实现；③ 在理论上给出了收敛速率提升的充分条件，并证明在有限 MDP 下几乎必然收敛到最优值。

**🔧 技术方法**

采用的技术包括：n‑step λ‑return、Harutyunyan Q(λ) 算子、limiting‑kernel 逼近、TD 学习估计 LK 项、Actor‑Critic 结构（PPO、NAF）、拟合 Q‑迭代（fitted Q‑iteration）、强化学习中的梯度更新与经验回放。

**📊 数据集**

实验数据集：离散的 Garnet 随机 MDP（50 状态 × 5 动作）用于验证算子收敛；连续的 MuJoCo 控制任务（HalfCheetah、Hopper、Humanoid）用于评估 LKQL 在实际控制中的性能。

**📈 对比分析**

方法比较：在离散 MDP 上对比 HI 与 LKHI 的算子范数差异；在连续控制上与基线 GAE（PPO）和 THQL（NAF）进行累计奖励对比。实验表明，LKQL 在大多数 roll‑out 长度下均优于基线，尤其在 Hopper 与 Humanoid 任务上提升显著。

**⚠️ 局限性**

局限性：① 收敛证明仅针对固定行为策略；② 引入额外的 LK 网络导致内存与计算成本增加；③ 需要调节 τ、β_U、λ_U 等额外超参数；④ 对非固定行为策略或更复杂的 off‑policy 修正（如 Retrace、Tree‑Backup）的推广仍待研究。

---

## 416. FFM-CP: Cross-Backbone Fusion of Vision-Language Foundation Models for Few-Shot Computational Pathology

**arXiv ID:** 2609.27710 | [PDF](https://arxiv.org/pdf/2609.27710v1)

**作者:** Anh-Tien Nguyen `[一作]` (Giessen University), Anne-Christin Hauschild `[通讯]` (Giessen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

结合多种病理视觉‑语言基础模型进行少样本学习，利用正交 Procrustes 对齐、统一图谱进行特征融合，并在跨模型预测对中实现双分支融合；

**💡 创新点**

提出无参数对齐、统一图谱、多模型预测对融合的端到端框架（FFM‑CP），能在仅有少量标注时充分挖掘不同模型的互补信息；

**🔧 技术方法**

Orthogonal Procrustes 对齐、图神经网络（GCN）特征聚合、双分支（文本原型+案例检索）融合、少样本自适应学习；

**📊 数据集**

六个公开病理图像数据集：LungHist700、HeidelbergSkin、Kather2016、LubLung、BRACS 与 BACH；

**📈 对比分析**

与单模型及现有基线对比，FFM‑CP 在 54 组对比中 50 组优于最佳单模型，平均提升 2.35‑3.73 个百分点；

**⚠️ 局限性**

仅评估 tile/region 级别，未验证全切片（WSI）应用；模型规模大、计算成本高；跨机构泛化和潜在偏差需进一步验证。

---

## 417. SynSeq: End-to-End SYNTAX Score Prediction from Coronary Angiography Videos

**arXiv ID:** 2609.27696 | [PDF](https://arxiv.org/pdf/2609.27696v1)

**作者:** Christoph Baumann `[一作]` (Medical University of Vienna), Philipp Seeböck `[通讯]` (Medical University of Vienna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了SynSeq，一种端到端的视频模型，直接从多视角冠脉血管造影视频预测SYNTAX评分并给出再血管化治疗建议。

**💡 创新点**

创新点包括：针对冠脉血管视频的专门预处理；针对零膨胀、长尾分布的样本加权和线性目标缩放的损失设计；以及结合双分支3D-ResNet+LSTM的全局时空特征提取。

**🔧 技术方法**

使用了Kinetics-400预训练的3D-ResNet18作为骨干网络，单层LSTM作为预测头；采用双任务损失（回归+二分类）并引入样本权重；训练分两阶段：先对单帧分类预训练，再冻结骨干训练LSTM，最后联合微调。

**📊 数据集**

在公开的CardioSyntax数据集上进行评估，该数据集包含1844名患者、9590条左冠系和3970条右冠系视频，部分患者拥有三位专家的标注。

**📈 对比分析**

与MVL、DeepRV等基线及单均值基线相比，SynSeq在R²从0.07/‑0.11提升至0.62，MAE下降约30%，偏差减少93%，再血管化分类的加权F1从0.64提升至0.80，接近专家一致性。

**⚠️ 局限性**

局限包括：数据集划分不统一、缺乏公开的基线模型权重、在不同机构和影像协议上的外部验证缺失，以及模型对严重病例的罕见样本仍存在一定偏差。

---

## 418. CasCVS-Net: A Staged Multi-Task Cascade for Critical View of Safety Assessment

**arXiv ID:** 2609.27681 | [PDF](https://arxiv.org/pdf/2609.27681v1)

**作者:** Bock-Zien Toh `[一作]` (University College London), Sophia Bano `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了 CasCVS-Net，一个分阶段多任务级联网络，联合完成检测、语义分割和安全视角(CVS)评估。

**💡 创新点**

创新点在于通过检测产生的框指导分割，再用分割掩膜池化得到区域特征，形成盒子到掩膜再到 CVS 的级联；并采用分阶段训练以稳定多任务学习。

**🔧 技术方法**

技术包括 ResNet‑50‑FPN 编码器、轻量级残差适配器、Faster R‑CNN 检测头、DeepLabV3+ 分割头、Mask‑pooled CVS 头，以及 PCGrad 梯度冲突消除。

**📊 数据集**

使用公开的 Endoscapes2023 数据集进行训练和评估，包含检测、分割与 CVS 标注。

**📈 对比分析**

与单任务基线以及 LG‑CVS、SV2LSTG 对比，CasCVS‑Net 在检测 mAP 32.0、稀有解剖 mIoU 15.3、CVS mAP 67.2、平衡准确率 74.7，均超过对手。

**⚠️ 局限性**

局限在于仍难以精确分割罕见的肝胆解剖结构，且需要已收敛的检测器；未考虑时序信息与多中心验证。

---

## 419. Same Scores, Different Decisions: Evaluating JEV and Language Models for Legal Document Understanding

**arXiv ID:** 2609.27678 | [PDF](https://arxiv.org/pdf/2609.27678v1)

**作者:** Fan Zhang `[一作]` (University of Tokyo), Songwei He `[通讯]` (University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

比较了 JEV 与十款语言模型在 ContractNLI 合同推理任务中的成本、响应时延、基线准确率以及在重复请求条件下的稳定性。

**💡 创新点**

提出将成本、时延与多条件下判断稳定性联合评估，设计了 “all‑12 correct” 指标，并揭示聚合准确率可能掩盖单个判断变化的现象。

**🔧 技术方法**

采用多种 Prompt 设计（可见假设、请求输出、输出顺序）和 Bootstrap 不确定性估计，结合本地 GPU 与多方 API 进行推理。

**📊 数据集**

使用官方 ContractNLI 测试集（123份合同、2091个判断）及其30份子集作为稳定性面板，全部保留原始合同文本与假设。

**📈 对比分析**

通过配对比较成本（美元/合同）、中位响应时延、基线准确率以及所有 12 次正确率等指标，发现 JEV 成本最低、响应最快，Hosted 模型准确率最高，但 JEV 在多条件稳定性上与 Sonnet 等模型相近。

**⚠️ 局限性**

局限性包括样本规模有限、稳定性面板仅 30 份、不同模型接口与推理设置差异、未覆盖新假设类型、未与人工基准对照，且指标受价格假设与硬件配置影响。

---

## 420. RoadOcc Learns When to Persist, Transport, or Refresh Memory for Roadside Occupancy Prediction

**arXiv ID:** 2609.27677 | [PDF](https://arxiv.org/pdf/2609.27677v1)

**作者:** Xiaokai Bai `[一作]` (Zhejiang University), Hui-liang Shen `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种用于路边摄像头场景的三源稀疏卷积网络RoadOcc，能够在固定坐标下对静态背景与动态交通进行语义占据预测，并结合运动补偿与历史记忆实现高精度的动态检测与速度估计。

**💡 创新点**

创新点在于把历史证据的路由问题形式化为Persist‑Transport‑Refresh三源选择，通过运动状态与类别一致性监督学习来源偏好；同时引入动态感知交叉注意力（DCA）、多尺度体素速度估计（VVE）和速度引导的稀疏融合（VDSF）三大模块，显著提升了对稀疏动态目标的恢复与位移补偿。

**🔧 技术方法**

技术主要包括：多视角图像到三维体素的编码器，动态感知交叉注意力（DCA）定位需要补偿的体素；VVE利用当前与历史体素的代价体积来估计运动并生成后向采样地址；VDSF在稀疏令牌预算下根据P/T/R路由权重融合固定坐标记忆、运动补偿记忆和当前观测。

**📊 数据集**

主要使用InfraOcc路边占据基准数据集（约290段连续序列，215段训练，75段评测），并在Occ3D-nuScenes和不同时间间隔的测试中验证迁移与鲁棒性。

**📈 对比分析**

与STCOcc、CRT‑Fusion等最新方法对比，RoadOcc在InfraOcc评估中实现mIoU 65.29、动态mIoU 32.37、Direct MAVE 1.669、DSR 61.11，分别比STCOcc提升4.44点、4.71点、0.277点、3.99点；在Occ3D-nuScenes上同样取得显著优势，表明其在不同场景与时间间隔下的优越性能。

**⚠️ 局限性**

局限性包括：仅针对固定路边摄像头框架，无法直接应用于移动平台；对极短暂或快速移动物体的识别仍有挑战；路由学习仍依赖手工定义的类别与速度阈值，可能在更复杂多样的交通环境中表现不佳。

---

## 421. The KV Cache Working Set: Online Capacity Planning for LLM Inference Systems

**arXiv ID:** 2609.27746 | [PDF](https://arxiv.org/pdf/2609.27746v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 422. Visibility-Guided Structured Measure Flow for Class-Conditioned 3D Gaussian Generation

**arXiv ID:** 2609.27778 | [PDF](https://arxiv.org/pdf/2609.27778v1)

**作者:** Yizhao Wang `[一作]` `[通讯]` (Henan Institute of Science and Technology), Yizhao Wang (Henan Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VISTA-GS，基于可见性引导的结构化测度流框架，用于类条件的 3D 高斯分裂（3D Gaussian Splatting）生成；

**💡 创新点**

创新点在于将 3DGS 视为可见性加权的高斯测度，利用可见性权重、渲染一致的流模型和结构化补丁传输，实现无序、可变大小的高斯原语的原生生成；

**🔧 技术方法**

采用可见性加权测度 VAE、条件流匹配、渲染一致监督、结构化补丁传输以及差异化 3D 高斯渲染器；

**📊 数据集**

使用自构建的 VISTA-Obj30 基准，包含 30 个类别的 ShapeNetCore、Objaverse、ABO、3D-FUTURE 和 OmniObject3D 训练实例；

**📈 对比分析**

与 GaussianCube、L3DG、DiffGS、Atlas Gaussians、SplatFlow、GaussianAnything 等基线在 CD、F@1、LPIPS、SSIM、MVC 及生成速度上对比，VISTA-GS 在所有指标上均显著优于基线（如 CD 降 61%+，LPIPS 降 61%+，生成速度提升 68%+）；

**⚠️ 局限性**

限制包括对可见性权重、渲染一致性权重等超参敏感，且当前仅支持基于类别标签的生成，未能直接利用文本或图像提示进行多模态生成。

---

## 423. Track2Art: Motion-Centric Articulated Object Model Recovery from 2D Point Trackers

**arXiv ID:** 2609.27675 | [PDF](https://arxiv.org/pdf/2609.27675v1)

**作者:** Xiaotong Li `[一作]` (University of Cambridge), Brian Sheil `[通讯]` (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用 RGB‑D 交互视频构建持久的视觉‑几何跟踪，结合预训练的点追踪潜在、DINO 视觉特征和 3D 轨迹几何，完成可变数量刚体部分的发现、定向运动关系和关节几何的端到端推断。

**💡 创新点**

创新点在于：①把点追踪潜在与视觉描述符融合成“视觉‑几何”跟踪表示，既保留细粒度对应信息，又保留度量运动；②使用可变 cardinality 的集合预测和刚体一致性监督实现无监督的部分发现；③引入旋转等变的学习‑解析几何头，既能学习关节方向，又保持坐标系等变性。

**🔧 技术方法**

核心技术包括预训练 CoTracker3 点追踪器、DINOv2 视觉描述符、3D 轨迹几何特征、DETR‑style 变压器集合预测、基于 Kabsch 的刚体一致性正则、旋转等变的轴融合与关节轴线精修，以及基于轨迹重播的运动一致性监督。

**📊 数据集**

主要使用 PartNet‑Mobility 数据集（20 种多自由度物体的 8 秒 RGB‑D 交互序列），并在 LightWheel 与真实 RGB‑D 场景中进行跨域和真实传输评估。

**📈 对比分析**

在 aligned 20‑object 基准上，Track2Art 在无结构先验、无测试时优化的条件下取得了 0.695 的 Point IoU、0.782 的 Adjusted Rand Index 和 0.410 的 J@20，明显优于 FreeArtGS、PARIS、VideoArtGS 等基线，且在视角扰动和坐标系旋转下保持高等变性。

**⚠️ 局限性**

局限性包括：对关节几何的估计仍易受弱运动、遮挡或深度噪声影响，导致轴向误差尾部偏大；在极度复杂或长时序的交互中，固定 32‑维轨迹描述符的表达可能不足，需进一步丰富时间建模。

---

## 424. InfiNoVA: Infinite Novel View Augmentation for Viewpoint Invariant Robot Policies

**arXiv ID:** 2609.27734 | [PDF](https://arxiv.org/pdf/2609.27734v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 425. Holonic Graceful Transitions Across Centralized, Decentralized, Distributed, and Local Control in DER-rich Cyber-Power Distribution System

**arXiv ID:** 2609.27759 | [PDF](https://arxiv.org/pdf/2609.27759v1)

**作者:** Md Fazley Rafy `[一作]` (West Virginia University), Anurag K. Srivastava `[通讯]` (West Virginia University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一种服务无关的 Holonic 边缘自治协调框架，支持 DER 在不同通信/攻击场景下自适应切换集中式、分布式、分区去中心化和本地自治四种协调模式。

**💡 创新点**

创新点包括：① 在协调层与应用层分离的 Holonic 架构；② 事件驱动的平滑过渡机制与边缘异常检测+邻居确认，实现节点级隔离；③ 区域化临时分区与领导选举实现局部自适应。

**🔧 技术方法**

使用技术包括：Holonic 系统设计、边缘计算与 MQTT 通讯、Raspberry Pi 边缘控制、Typhoon HIL 与 OpenDSS 仿真、VVC 基于原始-对偶的控制、MLP autoencoder 异常检测、速率限制与停留计时器。

**📊 数据集**

使用自定义的 HIL 测试平台生成的电网测量与攻击/通信事件数据，未使用公开公开数据集。

**📈 对比分析**

与单一集中式和分布式控制方案对比，利用事件注入测量协调流量、全局与局部电压目标波动，结果显示 Holonic 框架在保持电压稳定、减少通信负载、避免振荡方面均优于传统方案。

**⚠️ 局限性**

局限性包括：仅在单馈线 HIL 环境验证；未评估大规模多馈线网络的可扩展性；缺乏严格的收敛/稳定性理论分析；对极端攻击或大规模故障的鲁棒性仍待进一步测试；实现依赖 Raspberry Pi，真实系统延迟与功耗未完整评估。

---

## 426. Categorical Internalisation of Environmental Groupoids for Generalisable POMDP Solving

**arXiv ID:** 2609.27745 | [PDF](https://arxiv.org/pdf/2609.27745v1)

**作者:** Ben Opperman `[一作]` (University of London), Esther Mondragón `[通讯]` (University of London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在部分可观测、高维环境中提出使用群组体（groupoid）对状态空间进行分块并进行规范化，从而在强化学习中共享经验、提高样本效率。

**💡 创新点**

创新点在于将范畴论中的群组体结构引入RL，通过对称性分块和传送器（transport）实现局部对称共享，并同时提供结构化与自动化两种规范化方法（G‑orbit 与动力学二元化）。

**🔧 技术方法**

核心技术包括：群组体规范化、传送器映射、基于群组体的 Q‑学习算法、动态二元化的余量化等；实现基于 POMDP 的贝尔曼更新与对称传播。

**📊 数据集**

在 POMDPy 框架下使用 RockSample（20×20 网格）和 Tag（15×15 网格）两个标准部分可观测任务进行实验。

**📈 对比分析**

与基线 POMCP（粒子 MCTS）进行对比，实验表明结构化群组体学习者在 20×20 环境中比 POMCP 快约 45% 收敛，平均奖励显著更高；自动化二元化方法虽收敛略慢，但仍优于基线，验证对称性共享能显著提升样本效率。

**⚠️ 局限性**

局限性包括：目前仅适用于离散对称（如平移、旋转等），连续空间和高维动作尚未解决；需要先构造或学习群组体结构，若对称性未知或不充分，自动化方法可能收敛慢；对极端噪声或强非线性动力学的鲁棒性未充分评估。

---

## 427. MENO: Memory-Efficient Neural Operator

**arXiv ID:** 2609.27739 | [PDF](https://arxiv.org/pdf/2609.27739v1)

**作者:** Shengyang Xu `[一作]` (Peking University), Pengzhan Jin `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Memory‑Efficient Neural Operator (MENO)，一种利用 Manifold Function Encoder (MFE) 的高效 PDE 神经算子，能够处理任意几何域和离散化，并实现跨几何输入。

**💡 创新点**

创新点在于：① MFE 通过对流形和流形函数的全局积分编码，压缩为低维向量，消除训练时对积分操作的需求；② MENO 采用无积分的全局 Transformer 分支和局部点云分支，并通过信息注入实现两分支协同，显著降低内存占用；③ 支持跨几何（不同输入、输出流形）场景，具备强泛化能力。

**🔧 技术方法**

技术核心包括：Manifold Function Encoder (MFE)、Legendre/傅里叶谱基函数、全局 Transformer 结构、局部点云线性层、双向信息注入、统计归一化与反归一化。

**📊 数据集**

使用的数据集包括 2D Poisson 方程（跨几何与单几何）、NASA‑CRM 航空气动数据、AhmedML 汽车气动数据等，涵盖不同维度、边界条件和物理参数。

**📈 对比分析**

对比 Transolver‑3、PCNO、Graph‑U‑Net 等主流算子，在相同硬件（NVIDIA A100）下，MENO 在 Poisson、NASA‑CRM 取得最佳精度，AhmedML 取得第二佳精度，同时参数量、峰值 GPU 内存和训练时长均低于 5%–20%，显著提升效率。

**⚠️ 局限性**

局限性：目前仅在 2D/3D 流形上验证；对极端大规模数据仍需进一步评估；在某些复杂多物理耦合问题中，跨几何注入方式可能需要更细粒度的梯度信息；对非常高频细节的捕捉仍受谱基限，未来可考虑自适应基函数或多尺度 MFE。

---

## 428. DAVIO: Dense Monocular-Inertial SLAM with Feed-Forward Initialization and Pose-Conditioned Mapping

**arXiv ID:** 2609.27702 | [PDF](https://arxiv.org/pdf/2609.27702v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 429. Adaptive Channel Hopping for IEEE 802.15.4 TSCH-Based Networks: A Dynamic Bernoulli Bandit Approach

**arXiv ID:** 2609.27876 | [PDF](https://arxiv.org/pdf/2609.27876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 430. Vertex-Coloring Edge-Weighting: Kernelization and Generalization

**arXiv ID:** 2609.27719 | [PDF](https://arxiv.org/pdf/2609.27719v1)

**作者:** Shubhada Aute `[一作]` (IIT Hyderabad), Geevarghese Philip `[通讯]` (Chennai Mathematical Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了图的边加权与顶点着色之间的关系，提出了在给定顶点覆盖数k的情况下，图是否允许适当的边加权的问题，并证明了这两个问题在参数化时具有多项式核。

**💡 创新点**

创新点在于证明了在参数化顶点覆盖数k的情况下，两个问题都具有多项式核，并且在树深度参数化时是NP难的，解决了之前未解的两个问题。

**🔧 技术方法**

使用了参数化复杂性理论和动态规划算法，结合树分解技术来解决问题。

**📊 数据集**

使用了多种图数据集，包括具有不同顶点覆盖数和树深度的图，具体数据集未详细列出。

**📈 对比分析**

与之前的算法相比，本文的算法在时间复杂度上有显著改进，特别是在处理预加权边的情况下，运行时间为2^O(k log k)·n，优于之前的2^O(k^4)·n^O(1)。

**⚠️ 局限性**

限制在于算法的性能依赖于图的结构特征，如顶点覆盖数和树深度，且在某些情况下可能无法处理更复杂的图结构。

---

## 431. MVP: A Motion-Predictive Speculative Vision Pipeline with Non-Blocking Drift Correction

**arXiv ID:** 2609.27706 | [PDF](https://arxiv.org/pdf/2609.27706v1)

**作者:** Raul Taranco `[一作]` (Universitat Politècnica de Catalunya), Antonio González `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种仅在运动域进行预测的连续视觉管线（MVP），通过在ISP内部预测运动向量并结合自适应调度实现实时感知。

**💡 创新点**

创新点在于整合运动预测与自适应漂移校正的调度模型、仅运动域的推测、以及可选的前端能耗控制。

**🔧 技术方法**

技术包括ISP级轻量化AR(2)运动预测器、运动向量去噪、基于漂移控制的调度、以及运动驱动的帧采样策略。

**📊 数据集**

使用KITTI车辆检测数据集（YOLOv8）以及MOT17等跟踪数据集进行验证。

**📈 对比分析**

与传统序列化管线、Euphrates和PVF等方法相比，MVP在平衡负载下尾部延迟降低约66.8%，能耗下降约46%，且准确率仅下降约1-2%。

**⚠️ 局限性**

局限性包括预测误差导致的漂移累积、无法提前感知新出现物体、以及对高动态场景的误判率略高。

---

## 432. What Do Tabular Foundation Models Compute In Context? In-Situ Representation Refinement through Attention-Gated Updates

**arXiv ID:** 2609.27679 | [PDF](https://arxiv.org/pdf/2609.27679v1)

**作者:** Tian Zhou `[一作]` (Ant Group), Liang Sun `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种无须参数更新的tabular ICL方法——RefineICL，利用支持样本标签在 episode 内对表示进行自适应修正，并将修正通过注意力门控传播到查询样本。

**💡 创新点**

创新点在于：①从监督邻域学习引出一个可正则化的支持修正公式；②将修正拆分为读/读门/残差更新，形成无 FFN 的注意力门控上下文块；③结合低秩特征交互与 typed memory，提升表示的表达力。

**🔧 技术方法**

使用的技术包括：基于核的留一监督目标、正则化局部优化、注意力读/门控、SiLU 门、低秩特征交互（FI）、行压缩与 typed memory、单前向推理。

**📊 数据集**

实验数据集：AMLB29、TabZilla（Canonical 与 Extension）、TabArena（38 数据集）、OpenML‑CC18、Grinsztajn、AutoGluon 公开基准。

**📈 对比分析**

与 TabPFN‑v3、TabPFN‑3、TabICLv2、TabFM、AutoGluon 等方法对比，RefineICL 在 TabArena 获得 1644.8 Elo（比 TabPFN‑3 高 31.4 Elo），在 AMLB29 上达到 0.87173 准确率、0.93836 OVR‑AUC，整体提升了 TabPFN‑v3 的准确率、F1、AUC 与对数损失；同时在 100K 更新网格下，去除 FFN 并不影响性能，却显著降低 60% 的峰值内存。

**⚠️ 局限性**

局限性：①在深度 L8 及以上时仍需 60% 以上额外内存；②对 FFN 的扩展未显著提升验证性能，表明模型对任务多样性依赖深度与宽度的平衡；③仅针对表格数据，未验证在更大规模或更复杂任务上的可迁移性；④需要足够数量的支持样本以保证邻域学习的有效性。

---

## 433. Bounded Loops: Pre-Run Spend Bounds, Proved Termination, and Verified Completion for Agent Harnesses

**arXiv ID:** 2609.27871 | [PDF](https://arxiv.org/pdf/2609.27871v1)

**作者:** Varun Pratap Bhardwaj `[一作]` (Qualixar), Arun Pratap Bhardwaj `[通讯]` (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一个形式化的代理（agent）运行时框架（harness）模型，定义了可终止的受限循环（bounded loop）和图形化循环（bounded‑loop graph），并证明了其在修复（repair）机制下的终止性、非漂移性以及预算上限的满足性；同时构建了一个验证器（verifier）工具，对已有的69个循环程序进行基于变异测试的质量测评，发现47个无效（vacuous）门和3个自鉴定（self‑attestation）缺陷。

**💡 创新点**

创新点主要有三项：
1) 对代理运行时契约（contract）进行形式化定义并提供可验证的三项保证；
2) 引入“三值”判定（pass / reject / incapacity）解决自评失误的问题，并证明其在任何执行路径中都是可判定且不依赖于工作者；
3) 开发基于两层变异测试的量化工具，能够在不需要等价变异器（equivalent mutant）判定的前提下，测定门的误接受率，并对 vacuity 与 self‑attestation 两类缺陷进行系统识别。

**🔧 技术方法**

技术栈包括：
- 形式化证明（Coq/Lean/Isabelle 等可选）
- 有向无环图（DAG）与受限循环的组合模型
- 哈希链日志（hash‑chained ledger）实现不可篡改性
- 变异测试框架（mutation testing）与两层 held‑out 样本
- 统计置信区间（Wilson 95% CI）
- 计数器与预算恢复（global repair counter）

**📊 数据集**

数据集：
- 69 个已发布循环程序的目录（catalogue）
- 57 个循环的子集（用于细粒度实验）
- 209 个“破坏性”变异（destroying mutants）
- 1-16 个人工注入缺陷的实验设置（用以评估软阈值和硬阈值的利用率）

**📈 对比分析**

比较方法：
- 对每个循环，使用变异测试生成一组缺陷并运行 verifier，记录误接受率。
- 对修复后的门，误接受率为 1.8%（Wilson 95% CI 0≤α≤1.8%）；
- 对未修复门，误接受率为 23.3%。
- 对硬阈值（hard budget）和软阈值（no‑progress window）分别测量尝试次数，结果显示软阈值在所有实验中都被严格满足，且利用率（used/declared）与宣告的上限完全一致。
- 性能方面，日志链的哈希计算和账本更新在每次尝试后仅耗费 O(1) 额外时间，整体运行时间与原始循环相近。

**⚠️ 局限性**

局限性：
- 仅关注可终止的受限循环，无法处理无限或非确定性循环；
- 假设工作者（worker）与检查器（gate）在写权限上完全独立，但未考虑统计相关性或模型共用导致的错误相关性；
- 变异测试只能评估误接受率，无法判断门本身的语义正确性；
- 自鉴定缺陷无法通过当前两层变异框架捕获；
- 哈希链日志只能检测已知行的篡改，无法防止尾部截断或完整文件重写；
- 证明中使用的全局修复计数器是硬编码限制，实际系统中若采用节点级计数则无法保证终止性。

---

## 434. Security and Privacy in Large-Model-Driven Embodied Agents: Attacks, Defenses, and Future Directions

**arXiv ID:** 2609.27847 | [PDF](https://arxiv.org/pdf/2609.27847v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 435. Learning What to Activate: Combinatorial Capability Allocation for Long-Horizon Multimodal Agents

**arXiv ID:** 2609.27869 | [PDF](https://arxiv.org/pdf/2609.27869v1)

**作者:** Wenhao Yuan `[一作]` (University of Hong Kong), Edith Cheuk-Han Ngai `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CoCA框架，用条件比较学习来为长时序多模态代理选择可执行的能力子集；

**💡 创新点**

创新点在于：①用条件对比学习构建可变依赖的能力效用模型；②通过自回归方式将子集生成与效用映射；③双层对抗式在策略学习中对状态与前缀分布进行匹配；④利用轨迹级强化学习进一步平衡任务成功、激活成本与切换成本；

**🔧 技术方法**

采用Bradley–Terry型偏好模型、可回归子集策略、教师-学生混合对齐的在策略蒸馏、轨迹级策略梯度以及大型语言模型（如Qwen3.6‑27B、Kimi‑K2.6）作为执行器；

**📊 数据集**

在四大公开基准上进行评测：OSWorld、VisualWebArena、GAIA、MMMU‑Pro，并使用Kimi‑K2.6进行能力库构建；

**📈 对比分析**

与ReAct、Toolformer、Puppeteer、AutoTool、NaviAgent等先进方法对比，CoCA在成功率、激活成本、切换成本和分配质量等多项指标上均显著优于基线；

**⚠️ 局限性**

局限性包括：需要多轮在策略训练和教师监督；依赖教师模型的偏好标注；对大规模能力集合的扩展性不明；以及在极端需求转移场景下的泛化仍有待验证。

---

## 436. "AI Is Turning Too Human": How Teenagers Experience and Negotiate AI in Everyday Life

**arXiv ID:** 2609.27824 | [PDF](https://arxiv.org/pdf/2609.27824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 437. Exact Minimax One-Bit Unbiased Compression: Heavy-Tail Necessity and Finite-Randomness Approximation

**arXiv ID:** 2609.27860 | [PDF](https://arxiv.org/pdf/2609.27860v1)

**作者:** Tao Jiang `[一作]` (Chinese Academy of Sciences), Shaowei Cai `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

设计并分析了一种点wise无偏的一比特压缩器，证明其在所有实数输入上以期望值重构并给出最优二阶矩；进一步对高斯位置族求解极小极大问题，并给出了对应的重尾性质与鲁棒化方案，最后通过坐标分配实现了固定比特数的高斯梯度压缩与优化。

**💡 创新点**

首次给出了针对任意源的全局最优二阶矩界，证明对应的阈值分布为√(F(1−F))；在高斯族中确立了最优极值先验并揭示了第三阶绝对矩无穷的必要性；提出了 Cauchy 混合鲁棒化和有限公共随机数实现方案，兼顾偏差、方差与随机性开销。

**🔧 技术方法**

使用了公共硬币随机阈值、重要抽样、重排不等式、极大极小化、Cauchy 混合技术、坐标分配与中值均值聚合等理论与算法工具。

**📊 数据集**

主要通过理论推导与数值实验验证（如 Λ_c 的精确计算、Cauchy 混合尾部行为、有限网格逼近曲线），并未使用真实数据集。

**📈 对比分析**

与传统的有偏/无偏一比特量化、全线无偏量化器等方法相比，取得了最优二阶矩与极小极大风险；在 Kim 的连续二次硬函数上实现了与下界匹配的参数依赖；鲁棒化方案在保持方差不变的前提下消除了高阶矩发散。

**⚠️ 局限性**

仅适用于标量压缩，需无限公共随机数支持，重尾性质导致第三阶矩无穷；在多维情形下仍需进一步研究最优常数与随机性/存储权衡。

---

## 438. Geometry-anchored PET-aware multimodal pseudo-CT synthesis for whole-body attenuation correction: the BIC-MAC Challenge

**arXiv ID:** 2609.27848 | [PDF](https://arxiv.org/pdf/2609.27848v1)

**作者:** Xuan Loc Nguyen `[一作]` (University of Science, VNU-HCM), Hung Cao `[通讯]` (University of New Brunswick)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种以非衰减校正 PET（NAC‑PET）为几何锚点的多模态伪 CT 合成方法。

**💡 创新点**

创新点在于通过门控残差融合将 2D 视角图和 Dixon MRI 的信息注入 3D NAC‑PET 结构，并结合绝对坐标与全身上下文实现解剖一致的预测；同时引入可微 PET 响应近似器实现 PET 关注的训练。

**🔧 技术方法**

使用了残差 3D U‑Net、二维 Topogram 编码器、3D Dixon 编码器、FiLM 全身上下文调制、傅里叶位置编码以及基于光滑与 PET 响应的多任务损失。

**📊 数据集**

使用了 BIC‑MAC 挑战公开数据集，包含 NAC‑PET、Dixon MRI、2D Topogram 以及部分 PET 标注。

**📈 对比分析**

与基线 3D U‑Net 对比，所提方法在官方评估指标上均取得显著提升：μ‑map MAE 下降 11.9%，SUV MAE 下降 48.7%，器官偏差从 4.56% 降至 2.23%，脑区异常率从 0.0541 降至 0.0053。

**⚠️ 局限性**

局限性包括训练过程中对 PET 响应近似的依赖、缺乏对不同设备或解剖变异的跨域泛化评估，以及推理时仍需手工裁剪窗口与重叠融合。

---

## 439. CAST: Context- and Anomaly Structure-Conditioned Time Series Anomaly Generation

**arXiv ID:** 2609.27825 | [PDF](https://arxiv.org/pdf/2609.27825v1)

**作者:** Haochen Zhang `[一作]` (University of North Carolina at Chapel Hill), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CAST框架，实现基于正常上下文和异常结构的可控时间序列异常生成；

**💡 创新点**

结合上下文无关预训练与上下文感知微调两阶段策略，同时引入异常结构潜变量，解决异常稀缺与多模态形态问题；

**🔧 技术方法**

使用VQ‑VAE提取异常结构潜变量，Rectified Flow Matching生成器进行条件流匹配，配合两阶段预训练-微调；

**📊 数据集**

在五个真实数据集上评测：MIT‑BIH Arrhythmia (MIT‑DB)、QT Database (QT‑DB)、SVDB、光伏发电 (PV) 和 Metro 交通量；

**📈 对比分析**

与五种基线（Diffusion‑TS、FlowTS、TimeVAE、GenIAS、C‑GATS）以及多种下游异常检测器对比，CAST在MSE/MAE、FID/KID/MMD和F1‑score等指标均优于基线，提升约26%/17%/16%；

**⚠️ 局限性**

主要局限在对异常结构潜变量的离散化和对不同异常模式的泛化能力，且对非常稀有或未见异常形态的生成仍受限。

---

## 440. Evaluating ADC-only deep learning pipelines for breast cancer detection and segmentation using standalone diffusion-weighted MRI

**arXiv ID:** 2609.27815 | [PDF](https://arxiv.org/pdf/2609.27815v1)

**作者:** Pablo García Marcos `[一作]` (University of Oviedo), Víctor M. González `[通讯]` (University of Oviedo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

评估了基于ADC（DW‑MRI）图像的乳腺癌检测与分割的深度学习模型。

**💡 创新点**

首次全面比较ADC仅图像在分类、检测和分割任务中的性能，探讨其作为无对比剂检测工具的潜力。

**🔧 技术方法**

使用YOLOv8进行切片分类与目标检测，使用nnU-Net进行2D/3D语义分割。

**📊 数据集**

采用ACRIN 6698公共数据集（385例）仅使用T0预处理ADC图像。

**📈 对比分析**

通过5折交叉验证评估；切片分类平均精度94.2%、准确率91%；目标检测平均精度75.8%，精确率87.8%；语义分割Dice约0.70；与DCE‑MRI模型相比性能不足。

**⚠️ 局限性**

ADC分辨率低、切片间距大导致3D分割受限，缺乏多模态信息，召回率低，影响实际应用。

---

## 441. GaussianDS: Depth-supervised Semantic Gaussian Splatting for Scene Understanding

**arXiv ID:** 2609.27850 | [PDF](https://arxiv.org/pdf/2609.27850v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 442. A Decade of Climate Polarization on Brazilian YouTube using Language Models

**arXiv ID:** 2609.27811 | [PDF](https://arxiv.org/pdf/2609.27811v1)

**作者:** Daniel Morais `[一作]` (Universidade Federal de Ouro Preto), Carlos H. G. Ferreira `[通讯]` (Universidade Federal de Ouro Preto)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了巴西葡萄牙语YouTube评论中气候立场的十年演变与互动，利用自训练LLM对240k评论进行立场分类并分析冲突与框架。

**💡 创新点**

在低资源、噪声、回复式对话环境下构建基于LoRA的自训练LLM管线，结合混合实例选择提升少数类覆盖和多样性，并在长期语境下量化互动异质性与语篇框架。

**🔧 技术方法**

使用Llama 3.1 + LoRA/QLoRA微调、自训练伪标注、混合实例选择、三分类立场识别以及定量评估与定性框架分析。

**📊 数据集**

2014‑2024年巴西气候相关视频的247,514条葡语评论（240k用于推断）以及4,008条人工标注样本。

**📈 对比分析**

通过与基线监督模型的5折交叉验证对比，宏F1提升0.112（从0.677提升至0.789），尤其对Believer、Denier类显著提升，整体准确率从0.719升至0.795。

**⚠️ 局限性**

仅依据评论文本标注，忽略父评论与视频上下文，伪标签仍可能引入噪声，数据集不可公开且对其他语言/地区推广性受限。

---

## 443. Evaluation Choices Decide the Forecasting Leaderboard: Evidence from a Production Marketplace Panel

**arXiv ID:** 2609.27867 | [PDF](https://arxiv.org/pdf/2609.27867v1)

**作者:** Md Rezwanul Islam `[一作]` (Field Nation LLC), Wael Mohammed `[通讯]` (Field Nation LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对一家B2B技术服务市场的生产数据（1,887个买家，67个月的月度GTV）进行24种预测模型（包括统计方法、间歇需求方法、机器学习模型和六个2025年发布的时间序列基础模型）与季节性基准的滚动原点评估，并系统地改变评估设计（分析单元、误差聚合方式、点估计与区间估计、需求模式分类），揭示评估选择对榜单排名的巨大影响。

**💡 创新点**

提出并量化评估者选择（单位、聚合、区间评分、需求类别）能反转或改变排行榜的结论；通过公开可复现的协议和代码，首次展示评估设计本身决定“冠军”，而非方法本身；同时评估已部署的选择规则在实际业务中的价值。

**🔧 技术方法**

采用滚动原点验证、MASE与RMSSE误差度量、Diebold‑Mariano、Friedman+Nemenyi统计检验、均匀正态化区间评分、分层合并的 conformal prediction 生成区间，以及合成面板生成器进行实验重现。

**📊 数据集**

核心数据为公司内部月度买家GTV面板（1,887买家）；外部复现使用公开M5零售日销量（聚合为月度，30,490个商品‑门店系列）和澳大利亚旅游访问量层级面板；所有面板均保留两级结构。

**📈 对比分析**

对24种方法与基准进行排名，发现市场总量基准在总量上接近第一位，而在单买家层面排名倒数第二，且19种方法在单买家层面显著优于基准；按需求类别划分显示不同方法各领跑；基准选择规则在所有需求类别中均能收敛约55%（RMSE）或48.9%（MASE）距离，比单一冠军略优。

**⚠️ 局限性**

仅基于单一公司、单一行业的数据；部分结论（如区间评分重排序）仅在生产面板出现；在公共M5面板中未复制；基础模型可能存在数据泄露风险；部分方法在训练时使用随机种子导致轻微波动；公共复现使用的模型组合有限，可能影响可泛化性。

---

## 444. ChronosAttack: Adversarial Tool Scheduling Attacks on LLM Agents

**arXiv ID:** 2609.27857 | [PDF](https://arxiv.org/pdf/2609.27857v1)

**作者:** Arash Vashagh `[一作]` `[通讯]` (University of New Brunswick), Arash Vashagh (University of New Brunswick)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对异步LLM代理的工具响应时序进行攻击，探究仅通过延迟即可改变决策的可能性。

**💡 创新点**

首次将工具响应时序视为攻击向量，提出ChronosAttack，证明仅添加非负延迟即可导致不同模型做出截然不同的选择，而无需篡改内容。

**🔧 技术方法**

使用延迟调度、顺序敏感评估、同步与一致性防御等技术，对四大主流LLM（GPT‑5.6 Sol、Gemini 3.6 Flash、DeepSeek V4 Flash、Claude Sonnet 4.6）进行实验。

**📊 数据集**

在两种人工构造的决策场景（云备份提供商选择与运输合作伙伴选择）中，使用三条工具输出与三项候选答案，无需真实公开数据集。

**📈 对比分析**

在四模型上对比自然与被攻击时序，发现GPT‑5.6 Sol与Claude在云备份场景可将目标选择率提升至90%以上；Gemini在部分情形出现反向偏移；同步/一致性防御可将攻击成功率降至0%。

**⚠️ 局限性**

局限在于仅测试有限工具数与两种场景，未覆盖更大工作流与网络异变；攻击策略仅适用于已知时序，未实现模型独立的通用搜索；防御效果因模型差异而不统一。

---

## 445. Query Implied Generative Engine Optimization

**arXiv ID:** 2609.27845 | [PDF](https://arxiv.org/pdf/2609.27845v1)

**作者:** Shilpa Ramakrishna `[一作]` (San Jose State University), William B. Andreopoulos `[通讯]` (San Jose State University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了基于文档语义推断用户意图的生成搜索引擎优化框架 QI‑GEO，利用知识图构建的实体关系网络发现信息缺口，并通过局部重写提升文档在生成搜索引擎中的可见性。

**💡 创新点**

创新点在于不依赖显式查询或查询生成，而是通过文档自身的实体关系进行语义扩展和意图近似，实现在黑盒生成搜索环境中的查询无关 GEO。

**🔧 技术方法**

使用的技术包括核心ference消解、关系抽取（REBEL）、实体链接到知识图、单跳知识扩展、收敛性评分、LLM（Qwen2.5）局部重写，并采用 PAWC、主观分数等评估指标。

**📊 数据集**

实验数据集为 GEO‑Bench（10,000 条查询‑文档实例）和 ExtendedGEOBench（多表述查询），并在固定检索设置下进行评估。

**📈 对比分析**

与原始文档比较，QI‑GEO 在单查询/多查询设置下分别提升 PAWC 约 12.6% / 15.9%，主观分数约 17.6% / 17.1%，citation 覆盖率提升 10.5% / 8.3%，Win‑Tie‑Rate 达 0.70 / 0.81，downside risk 低，整体显示显著性能提升。

**⚠️ 局限性**

局限性包括扩展仅限单跳，可能忽略更深层语义；缺口判定依赖阈值，易受噪声影响；对通用知识图的依赖在专业领域可能不足；LLM 重写可能引入风格或事实错误。

---

## 446. AI Can Do Your Homework. Now What? Report from an Online Workshop on Computing Assessment in the Age of Generative AI

**arXiv ID:** 2609.27842 | [PDF](https://arxiv.org/pdf/2609.27842v1)

**作者:** Muhammad Sajjad Akbar `[一作]` (University of Sydney), Ranysha Ware `[通讯]` (Swarthmore College)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在2026年7月28日在线举行一次由SIGCSE虚拟工作组组织的两小时研讨会，聚集了约73名计算教育工作者，围绕七个针对生成式AI的作业适配策略进行自由讨论，并在会议后整理了参与者的经验、观点与共识。

**💡 创新点**

创新点在于：①采用无讲座、无预备材料的“即兴讨论”模式，直接捕捉教学实践的实时反映；②将讨论分为“策略房”和“跨主题问题房”，使多样化策略与共通关注点并行探讨；③通过共享文档实时记录，形成可供后续参考的实践日志；④聚焦“课堂内实践”而非理论推导，填补了工具迅速迭代导致的学术滞后的空白。

**🔧 技术方法**

使用的技术主要是Zoom视频会议、实时文字转录（用于记录）、Google Docs/共享表格（记录讨论内容）以及会议期间未录制的现场对话。

**📊 数据集**

未使用正式数据集；所有“数据”均来自参与者的经验分享、案例讨论与即时反馈，未收集学生成绩或行为数据。

**📈 对比分析**

本报告不包含方法对比或性能评估；讨论内容以定性形式呈现，未进行系统的实验或量化比较，故无法提供具体性能指标。

**⚠️ 局限性**

局限性：①自选参与导致偏向已有变革意愿者；②缺乏量化测评与客观数据，仅为主观经验总结；③会议时间短、讨论深度受限，部分主题未能完整探讨；④结果仅为瞬时快照，无法证明对未来教学设计的长期影响；⑤未覆盖所有学科与机构类型，普适性有限。

---

## 447. A Non-Invasive Cloud-Based Migration Strategy for Post-Quantum Cybersecurity in Smart HVAC Systems: Architecture, Implementation, and Empirical Evaluation

**arXiv ID:** 2609.27828 | [PDF](https://arxiv.org/pdf/2609.27828v1)

**作者:** Mahedee Zaman Moon `[一作]` (Islamic University of Technology), Sk Md Mizanur Rahman `[通讯]` (Centennial College)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在现有的云端主控架构下，提出并实现了一层非侵入式的后量子加密代理（PQC Proxy），能够在不改动 HVAC 控制器、其固件或厂商云服务的前提下，将移动应用与云端之间的通信从经典 TLS 迁移至 ML‑KEM‑768+ML‑DSA‑65 组合的量子安全方案。

**💡 创新点**

创新点在于：①通过引入独立代理层实现零改动迁移，兼顾设备与厂商云的非合作性；②结合 ML‑KEM 与 ML‑DSA 的签名验证，提供前向保密、身份认证和完整性；③在真实硬件（Raspberry Pi 4B）上完成端到端性能测评并给出系统可扩展性、内存占用与侧信道安全性实验。

**🔧 技术方法**

技术实现使用了 NIST FIPS 203/204 标准的 ML‑KEM‑768 与 ML‑DSA‑65，Open Quantum Safe 的 liboqs C 库（通过 Python 3.13 绑定），AES‑256‑GCM 加密、HKDF 密钥派生，以及经典的 ECDH‑X25519/Ed25519 作为基准；网络模拟使用延迟（0–250 ms）和多线程并发（1–32 线程）测试。

**📊 数据集**

实验数据集主要来源于真实硬件测量：Raspberry Pi 4B 运行代理，ESP32‑S3 评估设备端内存与加密性能；未使用公开数据集，而是对所有加密原语和完整握手进行 500 次重复实验，并在多线程环境下采集吞吐量与内存变化。

**📈 对比分析**

与经典 ECDH‑X25519/Ed25519 基准相比，ML‑KEM‑768+ML‑DSA‑65 的完整握手平均耗时仅比基准慢 0.38 ms，吞吐量可达 443 sps（单核）或 3 546 sps（32 核扩展），内存增长 60 KB，侧信道测试均通过（无时间泄露、无会话密钥残留、MITM 拒绝、nonce 冲突率可忽略）。

**⚠️ 局限性**

主要局限在于：①信任代理层成为单点安全/可靠性风险，需高可用部署；②目前不实现 HVAC 设备端的后量子身份认证，仅提供云端到代理的量子安全；③性能评估基于单一硬件平台，未覆盖长期稳定性与多云环境的真实流量；④侧信道实验仅限时序，未测量功耗/电磁泄漏。

---

## 448. MixGuard: Towards Detecting and Understanding Mixer Laundering on Ethereum

**arXiv ID:** 2609.27807 | [PDF](https://arxiv.org/pdf/2609.27807v1)

**作者:** Qishuang Fu `[一作]` (Monash University), Tsz Hon Yuen `[通讯]` (Monash University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了首个公开的以案例为单位的混音器洗钱数据集 MixLaunder，基于该数据集对以太坊混音器洗钱行为进行系统量化，并提出 MixGuard 框架实现交易级洗钱检测和案例级聚类。

**💡 创新点**

创新点包括：①首个公开案例级混音器洗钱数据集；②从量化分析得到的五大洗钱策略指导检测设计；③融合地址、路径、拓扑三视图并结合掩码监督对比学习的混音器特征编码；④两阶段聚类（强联接核心+弱联接合并）实现高精度案例恢复。

**🔧 技术方法**

技术方法：三视图表征学习（MLP + GraphSAGE）、掩码监督对比学习、强联接基于余弦相似度的连边构造、弱联接基于 LightGBM 的聚类合并；整体实现基于 PyTorch/PyTorch‑Geometric。

**📊 数据集**

使用的数据集：MixLaunder（27起真实洗钱案例，9,300 条混音器交易，约1.1 B USD 的被洗金额）以及同一时间窗口下的背景混音器交易（Tornado Cash & Railgun）。

**📈 对比分析**

采用 9‑折案例留置评估，对 MixGuard 与 K‑means、HDBSCAN、MLP、LightGBM、DenseFlow、MG‑HRL、MixBroker、MixLinker 等基线进行比较。MixGuard 在交易检测精度 97.89%、组纯度 98.73% 与 top‑10 覆盖率 95.09% 上均显著优于所有基线，提升幅度高达 14% 左右。

**⚠️ 局限性**

局限性：①未覆盖跨链桥接与多链流动；②仅包含已公开披露的案例，无法涵盖尚未被披露的洗钱行为；③依赖可观察的链上信息，混音器内部私有转移仍不可见，导致部分洗钱路径难以完全追踪。

---

## 449. Ask Which, Not How Good: Sizing Benchmarks Scored by an LLM

**arXiv ID:** 2609.27787 | [PDF](https://arxiv.org/pdf/2609.27787v1)

**作者:** Atul Anand `[一作]` `[通讯]`, Atul Anand

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 350,000 条 LLM 判题进行方差分解，量化了被判定基准（如 MT-Bench、Arena‑Hard、AlpacaEval 2 以及经过区分度筛选的自制集）在不同评判程序、评分尺度和裁判宇宙下的测量分辨率与可靠性上限。

**💡 创新点**

创新点在于：① 将系统视为测量对象，将裁判-系统交互视为决定可靠性上限的关键因素，证明单裁判协议无论增加多少题目都无法突破该上限；② 明确了评判程序（原生模板与统一 0‑5 rubic）对上限的显著影响；③ 通过对比点评与成对偏好协议，揭示成对偏好显著提高上限但伴随位置偏差；④ 结合审计 628 篇论文，评估领域中对裁判宇宙与不确定性的披露缺失。

**🔧 技术方法**

使用一般可测性理论（G‑theory）方差分解、最小可检测差异（MDD）计算、Bootstrap 置信区间、配对偏好实验、项目区分度筛选等技术。

**📊 数据集**

使用的数据集包括 MT‑Bench、Arena‑Hard、AlpacaEval 2 以及自制的 59 题目区分度筛选集（含 23 个零差异复制）。裁判来源为 6 个当前一代 LLM 模型（加 3 个扩展模型）。

**📈 对比分析**

比较方法：在固定裁判宇宙与评分尺度下，分别测量系统方差占比、单裁判上限、最小可检测差异以及达到 0.8 可靠性所需的裁判/题目数量。实验结果显示：单裁判上限在 0.60–0.80 之间；采用原生 MT‑Bench 协议可将上限从 0.623 提升至 0.798；成对偏好协议上限可达 0.986，单裁判即可满足 0.8；但位置偏差导致偏好差异高于平均 7 点。

**⚠️ 局限性**

限制包括：裁判宇宙与分布不完全代表全部可能裁判；项目集差异（如 Arena‑Hard 仅 17 题）影响估计精度；仅使用点评与成对偏好两种协议，未覆盖所有真实 benchmark 流程；自制区分度筛选集仅为 59 题，未广泛验证；报告的上限与 MDD 与实际论文公布的改进差距仍受原始评判程序与题目数量的影响。

---

## 450. Same Team Label, Different Evidence: A Full-Text Audit of Claim Denominators in Human-AI Teaming Research

**arXiv ID:** 2609.27849 | [PDF](https://arxiv.org/pdf/2609.27849v1)

**作者:** Hanjing Shi `[一作]` (Lehigh University), Dominic DiFranzo `[通讯]` (Lehigh University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Human‑AI Teaming文献进行全文审核，识别其人类安排、机制路径与结果对齐，并构建“人机团队机制链”与声明池化检查点。

**💡 创新点**

提出以人类安排为起点的证据可接受性检查框架，形成机制链模型，解决团队标签下证据可比性与归纳的缺口。

**🔧 技术方法**

使用文献检索、系统映射、全文审计和归纳编码技术，并依据PRISMA与批判性解读方法构建审计流程。

**📊 数据集**

基于435条记录的标题/摘要映射（其中86篇全文被审核）以及公开的HAT实验研究数据。

**📈 对比分析**

通过声明池化检查点将团队过程与结果层级匹配，揭示不同人类安排导致的可比性差异，表明在信任、协调、性能和责任等维度上结论不一致。

**⚠️ 局限性**

受限于样本规模仅涵盖部分研究领域，缺乏多语言与跨文化考察，且依赖作者自报信息，可能忽略隐含的人机互动与社会语境。

---

## 451. Safe Multi-Robot Coordination via VLM-LLM Reasoning and Reachability Analysis

**arXiv ID:** 2609.27816 | [PDF](https://arxiv.org/pdf/2609.27816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 452. PonyEval: Evaluating LLM-Based Program Repair for Capability-Safe and Actor-Oriented Pony Software

**arXiv ID:** 2609.27832 | [PDF](https://arxiv.org/pdf/2609.27832v1)

**作者:** Bang Xie `[一作]` (Shanghai Jiao Tong University), Shaocong Long `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了291条真实Pony GitHub issue–pull-request对的可执行评测基准。

**💡 创新点**

首次提供跨历史编译器、运行时映射、黑盒测试的完整可重现Pony修复基准，并实现了机器审查的高一致性。

**🔧 技术方法**

采用mini‑SWE‑agent框架、Pony历史运行镜像、fail‑to‑pass门控、生成与审核的黑盒测试、以及多模型评测等技术。

**📊 数据集**

来源于15个Pony仓库、289个独立基准提交、72个历史运行镜像，共291个任务。

**📈 对比分析**

在同一agent scaffold下评测GPT‑5.6‑sol、DeepSeek‑V4‑Pro、GLM‑5.2、MiniMax‑M3、Kimi‑K3的patch‑conditional resolution率，最高达24.68%，最低10.21%，显示编译与行为验证是主要瓶颈。

**⚠️ 局限性**

数据高度集中于compiler仓库（72.5%），测试可能不完全覆盖缺陷，历史镜像与原始开发环境仍可能存在差异。

---

## 453. DMM-Align: Closed-Loop Optimization for 2D-3D Registration with Dual-Role Diffusion

**arXiv ID:** 2609.27794 | [PDF](https://arxiv.org/pdf/2609.27794v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 454. Groundbench: Multi-Resolution Polygon Grounding Exposes the Geometry Gap in Vision-Language Models

**arXiv ID:** 2609.27821 | [PDF](https://arxiv.org/pdf/2609.27821v1)

**作者:** Zhonghan Bian `[一作]`, Zhangyang Qi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于固定顶点数的多分辨率多边形定位基准，用文本形式输出N个归一化坐标，取代传统的框定位，适用于评估托管式视觉语言模型在生成几何边界上的能力。

**💡 创新点**

创新点包括：① 对目标轮廓按固定顶点数进行确定性构造（利用Visvalingam–Whyatt简化与插值技术）；② 统一、冻结的问答对与标注集，保证跨预算可比性；③ 在同一模型、同一问题下评估不同顶点预算，揭示表示能力与输出交付的权衡；④ 引入多维诊断（合法性、覆盖度、目标偏好、选择/追踪）以及交叉预算分析。

**🔧 技术方法**

技术手段主要是：① 调用托管式视觉语言模型（Doubao、Kimi、Qwen‑max、Gemini）通过提示生成多边形顶点；② 将文本顶点序列解析为坐标并按固定格式输出；③ 通过栅格化对生成多边形与标准轮廓进行IoU比较；④ 设计可复现的顶点构造脚本与评估脚本；⑤ 对不同输入干预（最大思考、图像/灰度、分辨率、色彩提示、温度、重复生成）进行系统化 ablation。

**📊 数据集**

使用 RefCOCO 系列（RefCOCO、RefCOCO+、RefCOCOg）共 1,500 个图像–表达–目标三元组，分别从 3 个子集各采 500 题，固定同类干扰、轮廓复杂度、相对尺度等属性，构成多预算（N=8,16,24,32,64）统一测试集。

**📈 对比分析**

通过比较四个托管模型在各预算下的固定‑IoU 与可解析‑IoU，Gemini 在低预算下表现最佳，Qwen‑max 次之，Doubao 与 Kimi 其后；IoU 在中等预算峰值后在 N=64 时急剧下降，说明高密度顶点导致解析与拓扑失效；同时采用合法性指标与填充实用度对比，揭示仅满足长度并不意味着合法多边形；诊断结果表明模型在目标偏好保持较好，但条件追踪能力随预算增大而下降。

**⚠️ 局限性**

局限性包括：① 仍然存在框定位与多边形生成之间的巨大几何缺口；② 高顶点预算导致解析错误与自交点增多，性能反弹不足；③ 评估依赖于固定的轮廓构造与栅格化策略，可能对不同分辨率或多边形形状产生偏差；④ 仅针对现有托管模型，无法直接衡量训练模型或自定义解码器的改进；⑤ 由于输出为文本，无法捕捉连续空间中的细微形状差异。

---

## 455. The Exact Approximation Ratio of the Optimal Fixed-Price Mechanism in Bilateral Trade

**arXiv ID:** 2609.27878 | [PDF](https://arxiv.org/pdf/2609.27878v1)

**作者:** Tao Jiang `[一作]` (Chinese Academy of Sciences), Shaowei Cai `[通讯]` (Chinese Academy of Sciences)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文求解了双边交易中最优固定价格机制的最差案例福利比率，并给出了其精确值 0.7380243357…。

**💡 创新点**

创新点在于：① 通过 Wronskian 饱和把任意实例转化为可解的控制问题；② 证明该控制问题严格凸且唯一最优解具有单一内部弧+边界弧结构；③ 通过解析积分得到唯一根 r⋆，从而得到精确比率；④ 构造逼近序列证明极值不被任何有限均值分布达到。

**🔧 技术方法**

主要技术包括：变换到卖家“卖方看跌”与买家“看涨”变换，Wronskian 饱和，逆行值坐标控制，指数化对数化简，严格凸性与 KKT 条件，单根交叉分析，以及显式积分求解一维控制方程。

**📊 数据集**

本研究无实验数据集，全部为理论推导与解析计算。

**📈 对比分析**

与之前的上限/下限（如 0.7292–0.73805）相比，本文给出了精确上限 0.7380243357，说明固定价格机制在最坏情况下可实现约 73.8% 的最大福利；并证明该比率是最优且不可被任何有限均值实例实现。

**⚠️ 局限性**

局限性：该结果仅适用于单一卖方与单一买方的双边交易、固定价格机制、完整先验信息、无预算约束下的主导策略；若引入多买多卖、动态定价或预算限制，结论不再直接适用。

---

## 456. What Changed? Drift Detection with Real, Virtual, and Incomparable Diagnosis

**arXiv ID:** 2609.27865 | [PDF](https://arxiv.org/pdf/2609.27865v1)

**作者:** Kentaro Oda `[一作]` `[通讯]` (Kagoshima University), Kentaro Oda (Kagoshima University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于条件Jensen–Shannon差异的双轴漂移监测器，能够在单个判别器对齐下同时检测协变量漂移和机制漂移，并给出漂移类型；

**💡 创新点**

创新点在于将协变量统计和功能统计统一到同一判别器对，构建可区分真实和虚拟漂移的无模型监测框架，并在连续监测下实现无窗口e‑process和有限记忆重启保证；

**🔧 技术方法**

采用条件JS散度估计、交叉熵差异、贝叶斯e‑process、停用窗口重启、时间序列自适应窗口等技术；

**📊 数据集**

实验使用SEA、STAGGER、纯协变量漂移、空白流和INSECTS（记录漂移点）等数据集；

**📈 对比分析**

与ADWIN、DDM、EDDM、Page–Hinkley、D3、WATCH等传统漂移检测器对比，双轴监测器在所有10个种子跑中均实现0误报、0漏报、100%漂移类型准确，检测延迟约500–1650样本，优于基线且无需额外调参；

**⚠️ 局限性**

局限性包括窗口估计导致的检测分辨率受限、功能统计需要及时标签（标签延迟会线性推迟检测），以及在同时存在大协变量和机制漂移时仅能报告“真实”类型。

---

## 457. SAT-based Encodings for Optimal Decision Trees with Explicit Paths

**arXiv ID:** 2609.27874 | [PDF](https://arxiv.org/pdf/2609.27874v1)

**作者:** Mikoláš Janota `[一作]` (INESC-ID/IST, Universidade de Lisboa), António Morgado `[通讯]` (INESC-ID/IST, Universidade de Lisboa)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了一种基于 SAT 的路径编码，用以精确求解给定训练样本的最小决策树（同时可控制树的深度和节点数）

**💡 创新点**

创新点在于：①将决策树的结构直接编码为路径集合，天然支持深度和大小约束；②利用拓扑枚举拆分搜索空间，并以深度优先策略进一步缩小搜索范围；③通过 MaxSAT 计算路径下界并对特征进行预处理，提升求解效率

**🔧 技术方法**

主要技术包括：SAT/MaxSAT 编码、PySAT 框架、CaDiCaL 求解器、k-Cardinality Modulo Totalizer 计数器、拓扑模板枚举、路径语义约束以及特征纯化/准纯化优化

**📊 数据集**

实验采用 Narodytska 等公开的基准集，分别使用 20% 与 50% 的随机抽样（样本数与特征数均较大）

**📈 对比分析**

与原先的 Narodytska 等实现对比，路径编码+拓扑枚举在多数实验中明显更快，能在 1000s 限时内求解更多实例；在深度优先策略下，CPU 时间甚至比最小大小搜索低 30-40%

**⚠️ 局限性**

局限性：搜索空间仍受树拓扑数量影响，极大规模的树在 SAT 求解器面前仍较困难；当前实现未探索并行化或更高级的 cube‑and‑conquer；对更大特征/样本集的可扩展性仍待验证

---

## 458. Agentic AI Cybersecurity Framework

**arXiv ID:** 2609.27856 | [PDF](https://arxiv.org/pdf/2609.27856v1)

**作者:** Victor Kebande `[一作]` `[通讯]` (University of Colorado Denver), Victor Kebande (University of Colorado Denver)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了 Agentic AI Cybersecurity Framework（AACF），一种能够实现自主、目标驱动、动态适应的网络安全防御体系。

**💡 创新点**

创新点在于将 Agentic AI 的感知、推理、决策、执行和学习五个功能层整合成闭环架构，突破传统基于规则的被动防御，赋予系统自主规划与执行复杂安全策略的能力。

**🔧 技术方法**

主要技术包括大语言模型（LLM）与外部工具的集成、内存模块、事件关联与风险评估算法、自动化安全工具（SIEM、SOAR、IDS/IPS 等）以及反馈驱动的持续学习机制。

**📊 数据集**

使用的数据来源为多源安全数据：系统日志、网络流量、终端遥测与威胁情报订阅，论文未采用公开实验数据集，而是以案例演示方式验证框架。

**📈 对比分析**

作者未进行量化实验，仅通过实例化入侵检测与响应场景说明 AACF 在响应时间、主动性和可扩展性方面优于传统基于规则的系统；缺乏对比实验与性能指标。

**⚠️ 局限性**

主要局限包括易受对抗性操纵（如数据投毒、提示注入）、缺乏可解释性、对现有安全基础设施的集成挑战，以及需要人工监督以平衡自治与治理。

---

## 459. When Adaptation Hurts: Split Sensitivity and Person-Level Negative Transfer in Federated Wearable Onboarding

**arXiv ID:** 2609.27819 | [PDF](https://arxiv.org/pdf/2609.27819v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 460. Dense Interprocedural Dominance in Acyclic Graphs: Context Bounds and Compact Queries

**arXiv ID:** 2609.27818 | [PDF](https://arxiv.org/pdf/2609.27818v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 461. DualStabSleepNet: A Dual-Domain Diffusion Stabilization Network for Robust Sleep Staging

**arXiv ID:** 2609.27793 | [PDF](https://arxiv.org/pdf/2609.27793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 462. To Code or Not to Code: When and How to Use Network Coding in Energy Harvesting Wireless Multi-hop Networks

**arXiv ID:** 2609.27875 | [PDF](https://arxiv.org/pdf/2609.27875v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 463. TopoGS: Topology-Aware Anchor Feature Aggregation for Large-Scale 3D Gaussian Splatting

**arXiv ID:** 2609.27868 | [PDF](https://arxiv.org/pdf/2609.27868v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 464. Independent Set Discovery on Biclique-Free Graphs Is Fixed-Parameter Tractable

**arXiv ID:** 2609.27837 | [PDF](https://arxiv.org/pdf/2609.27837v1)

**作者:** Chenghua Liu `[一作]` (Chinese Academy of Sciences), Boning Meng `[通讯]` (University of Regensburg)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一种新的 FPT 算法，用于在 K_d,d‑free 图中求解“Token Slide”问题，即从给定的 k‑token 配置寻找最短滑动序列使终点为独立集。

**💡 创新点**

创新点在于将传统的全局覆盖族方法替换为局部“cheap‑prefix”递归，利用 Kővári–Sós–Turán 的稀疏性界定，首次在 biclique‑free 框架下得到统一的 FPT 结果并给出最优解与具体滑动序列。

**🔧 技术方法**

核心技术包括：代价匹配等价性、最优分配与滑动的等价证明、cheap‑prefix 递归分支、Wanless–Wood 的独立遍历判定、以及多层可扩展的参数化复杂度分析。

**📊 数据集**

论文为纯理论研究，没有使用实验数据集；所有证明与算法均在理论图论模型下给出。

**📈 对比分析**

通过与已知的 bounded‑degeneracy 与 nowhere‑dense 类别的 FPT 结果对比，证明了在 K_d,d‑free 类中同样可实现 2^O(dk log k)(n+m)^O(1) 的时间复杂度，显著优于之前仅对特殊稀疏类的结果。

**⚠️ 局限性**

限制在于算法仅针对 K_d,d‑free 图设计；对一般图或更广泛的稀疏图类别仍缺乏统一的 FPT 处理，且实现细节依赖于精确的代价匹配与 shortest‑path 计算，实际工程化应用仍需进一步验证。

---

## 465. What Confidence Routing Is Actually Doing: Auditing Routing, Calibration, and Commitment in Multi-Agent Deliberation

**arXiv ID:** 2609.27822 | [PDF](https://arxiv.org/pdf/2609.27822v1)

**作者:** Jingyan Jiang `[一作]` (Argonne National Laboratory), Chih-Hsuan Yang `[通讯]` (Argonne National Laboratory)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对多代理广播推理协议中的“置信度驱动路由”进行系统审计，分别评估路由选择、置信度校准和公共承诺三个维度，利用交叉拟合等方法修正概率尺度，并在两大语料库（奥林匹克数学题和生物多项选择题）上进行 2×2 代理×基准的实验。

**💡 创新点**

创新点在于将常被视为单一指标的路由、校准和承诺拆解为可分离的度量，提出跨折校准（等阶温度/等距单调回归）以显著降低 ECE，并通过网格实验揭示置信度在不同模型和任务中的有效性差异。

**🔧 技术方法**

使用的技术包括：基于结构化投票和自由文本发言的轨迹抽取；可提取答案的验证器；置信度的 AUROC、ECE、Brier 分数评估；交叉折校准（温度缩放和单调回归）；LLM 内部判定器与跨模型再判定；以及问题集层的聚类自举得到置信区间和稳健回归。

**📊 数据集**

数据集：共 4,181 题目（来自奥林匹克风格竞赛）产生 23,391 条有效投票候选，1,122 个路由块；再扩展至 2×2 网格实验，分别加入 Gemma 代理和 LAB‑Bench 生物多项选择基准，得到 2,912/4,852/14,129 条候选数据。

**📈 对比分析**

对比方法：与随机有效选择、oracle（总是选对答案）以及固定规则（confidence‑weighted, tie‑random）对比；路由成功率在 math 基准上约 49% vs 55%（oracle），校准方面原始 ECE 0.278，交叉拟合单调回归降至 0.008；在 Gemma 单元中，置信度 argmax 甚至低于随机有效选取 5.6–11.2 pp；承诺偏差在 math 负向 -1.7 pp，其他单元正向 +0.9–12.2 pp。

**⚠️ 局限性**

局限性包括：仅审计一种广播协议；实验结果受代理与任务特定影响，缺乏等价成本的单代理基线；未对已校准置信度进行重新路由实验；LLM 判定器可能存在偏差；未衡量训练数据重叠对指标的影响。

---

## 466. EidosDoc: Implicit Structure Encoding for Cost-Effective Semi-Structured Document QA

**arXiv ID:** 2609.27784 | [PDF](https://arxiv.org/pdf/2609.27784v1)

**作者:** Teng Lin `[一作]` (Hong Kong University of Science and Technology), Nan Tang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了EidosDoc系统，实现半结构化文档问答。

**💡 创新点**

主要创新包括隐式结构编码、无LLM的混合检索、动态证据扩展。

**🔧 技术方法**

使用对比学习+结构一致性损失的双塔编码、BM25+布局指纹+轻量交叉编码器检索、图邻接展开等技术。

**📊 数据集**

在四个公开基准（MMDA、MP-DocVQA、DUDE、M3DocVQA）上进行实验。

**📈 对比分析**

与11种基线对比，获得最高AIC-Acc/ANLS，降低50×成本、4×延迟。

**⚠️ 局限性**

仍受OCR误差、长文档及多跳推理等挑战影响，需改进多跳扩展和鲁棒性。

---

## 467. A Shared Encoder Is Not a Shared Task: Conditional Comparison for Deep Expert Pools

**arXiv ID:** 2609.27866 | [PDF](https://arxiv.org/pdf/2609.27866v1)

**作者:** Kentaro Oda `[一作]` `[通讯]` (Kagoshima University), Kentaro Oda (Kagoshima University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究深度持续学习中共享冻结编码器、任务头或适配器时，传统的一轴比较方法（交叉评估头、表示新颖性）容易被输入或标签变换混淆，提出在嵌入空间中使用条件Jensen–Shannon差异（CJSD）拆分为功能轴与协变量轴，从而区分机制变更与协变量变更，并用两轴门控实现高质量的头生命周期管理。

**💡 创新点**

核心创新在于：①将条件CJSD应用于嵌入空间，形成功能轴（对应机制变更）与协变量轴（对应输入变换），实现对两类变更的同时检测；②基于这两个轴构建两轴门控策略，显著减少误启用与漏识别；③在多任务流、不同规模与不同预训练基础（ImageNet ViT、DINOv2）上验证该方法可克服传统一轴方法的困扰。

**🔧 技术方法**

使用的技术包括：冻结或慢速更新的ViT编码器；每任务线性/MLP头或适配器；两头MLP判别器计算CJSD；功能轴与协变量轴的链式规则；混合头系统与最近损失路由；实验对比使用深度交换分数、Mahalanobis新颖性、损失跳跃等传统策略。

**📊 数据集**

实验数据集包括：MNIST（旋转与标签置换），CIFAR‑10 GCD（已知/光照/新类别块），CIFAR‑100（50已知+新类别块），ImageNet‑21k预训练的ViT‑B/16与自监督DINOv2 ViT‑B/14。

**📈 对比分析**

与传统单轴方法（深度交换分数、Mahalanobis新颖性、损失跳跃）对比，二轴门控在头数量、误启用率、漏识别率上表现更佳（例如在MNIST混合流中误启用率↓0.04、头数≈2.7）；在CIFAR‑10 GCD上功能轴实现AUROC 0.99 区分语义新颖与光照变化，传统MSP/Energy/Mahalanobis仅在0.5左右。

**⚠️ 局限性**

局限性包括：编码器被固定在热身后，未进行对所研究变换的增强/不变性训练；方法在需要在线自适应编码器的场景下可能需要重新设计；实验仅覆盖冻结的ViT与DINOv2，未验证更大规模或其他模态的泛化。

---

## 468. A hierarchy of faithfulness criteria for knowledge base completion

**arXiv ID:** 2609.27863 | [PDF](https://arxiv.org/pdf/2609.27863v1)

**作者:** Olga Mashkova `[一作]` (King Abdullah University of Science and Technology), Robert Hoehndorf `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出知识库补全(KBC)中对逻辑真实性（faithfulness）的分层评估框架，并在此框架下对现有几何及逻辑几何嵌入模型进行实验评估

**💡 创新点**

1) 明确KBC与KGC在开放世界假设下的区别，构建包含判别、逻辑可接受、单调逻辑真实性与概率逻辑真实性四层的严格链；2) 将相对模型计数作为概率逻辑真实性的目标；3) 证明四层严格蕴含关系并给出反例；4) 系统评估模型在每层的表现，揭示排名指标不等价于逻辑真实性

**🔧 技术方法**

描述逻辑推理、相对模型计数（#SAT/ApproxMC）、几何嵌入（TransE、DistMult、ComplEx、BoxE）以及逻辑几何嵌入（ELEmbeddings、EmEL++、Box2EL），以及基于负采样的训练干预

**📊 数据集**

EL语义的Pizza片段（可精确计数）与Gene Ontology子集GO-plus（中等规模）

**📈 对比分析**

通过在推理得到的三类（entailment、contradiction、undetermined）样本上计算AUC、Pair-FVR、Spearman相关等指标；结果显示：所有模型仅满足判别层，几乎没有满足单调逻辑真实性；几何模型在排名精度上稍优，但在faithfulness指标上差距显著；逻辑几何模型在部分构造（如disjointness）上表现更好，但整体仍未达概率层；负采样改进可提升faithfulness但降低排名精度

**⚠️ 局限性**

1) 相对模型计数仅在小规模知识库可计算；2) 评估仅覆盖EL片段，未扩展到更复杂描述逻辑；3) 训练干预仅探索负采样比例，未提出完整的概率逻辑学习目标；4) 结果依赖于人工生成的负样本，真实场景中负样本难以获取

---

## 469. Reachable Global Optimization in AI Systems: How Global Is Global?

**arXiv ID:** 2609.27855 | [PDF](https://arxiv.org/pdf/2609.27855v1)

**作者:** Wesley Shu `[一作]` `[通讯]` (Institute of Energetic Paradigm), Wesley Shu (Institute of Energetic Paradigm)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了 AI 系统在优化声明中对可达性（reachable region）的影响，提出了 Reachability-Induced Optimization（RIO）框架，并对可达最优性、误全局性等概念给出理论证明。

**💡 创新点**

创新点在于把可达性视为优化声明的核心条件，系统性区分了 Exact Global、Constrained Global、Reachable Global 等不同层级，并构建了可达性缺口分解、补充界定等证书体系，说明了控制策略如何影响可达空间。

**🔧 技术方法**

采用了理论推导、控制策略设计（随机搜索、贪婪局部改进、边界剪枝、逃逸策略等）和实验验证；在实验中对候选空间、预算、验证器等进行了建模。

**📊 数据集**

使用了离散候选空间 𝑋={0,…,100} 的已知最优景观，六个景观族共270个实例，执行 66,150 次实验（每种方法 35 次跑，每个景观 45 步搜索）。

**📈 对比分析**

通过比较七种控制策略在六个景观族上的平均全局误差、精确全局率、误全局风险等指标，发现：随机全搜索在信号弱或不可靠时表现最稳健；逃逸策略能显著扩展可达性并降低全局误差，但若信号质量不足易产生误全局；边界剪枝和局部修复虽稳定但被困在局部最优，误全局风险高。

**⚠️ 局限性**

局限性在于实验仅针对小规模离散空间，缺乏对高维、连续或真实 AI 任务的验证；RIO 框架虽解释了可达性对优化声明的影响，却无法消除 NP 难度，不能保证在大规模问题中一定找到全局最优。

---

## 470. DualMine: Static-Dynamic REST API Constraint Discovery with Dual Validation

**arXiv ID:** 2609.27806 | [PDF](https://arxiv.org/pdf/2609.27806v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 471. Agentic Governance and Adversarial Verification for Policy-Constrained LLM Healthcare Appeal Generation

**arXiv ID:** 2609.27844 | [PDF](https://arxiv.org/pdf/2609.27844v1)

**作者:** Harshil Lodhiya `[一作]` (Sliced Health), Reese Walker `[通讯]` (Sliced Health)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

该论文提出了AGVF多代理框架，用于在严格政策约束下生成医疗必要性申诉文本，确保每条论点都有可验证的证据。

**💡 创新点**

创新点在于将申诉生成建模为受约束马尔可夫决策过程，提供理论证明的缺陷递减和确定性引用门控。

**🔧 技术方法**

采用多代理架构：政策形式化、检索、缺口分析、对抗性批评和门控合成，并使用命令式检索+LLM模拟。

**📊 数据集**

采用从公开脱标化住院出院数据（SPARCS）抽样生成的1,000个合成案例作为评估基准。

**📈 对比分析**

与单通道RAG和AGVF的消融版本对比，AGVF在所有案例中实现零引用违规、递减缺陷、约40%案例完全解决，而RAG出现4%缺陷非递增和排除式阻塞。

**⚠️ 局限性**

限制在于使用合成数据、模拟LLM且未验证真实临床有效性，且假设政策图静态且批评器完备，实际部署需进一步实验。

---

## 472. From Reasoning Strings to Partial Orders: Verifier-Certified Rule Transport through Quotient Policy Optimization

**arXiv ID:** 2609.27833 | [PDF](https://arxiv.org/pdf/2609.27833v1)

**作者:** Bang Xie `[一作]` (Shanghai Jiao Tong University), Wei Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建基于本地验证器的可证依赖监督机制，使策略在推理过程中能共享合法重排的信用并避免非法顺序。

**💡 创新点**

首次将验证器回放产生的“钻石”与“反钻石”证书直接映射为轨迹轨道（orbit）比率，提出了一种分数共享且含有偏差抑制的 PPO 目标。

**🔧 技术方法**

利用 PPO / GRPO 强化学习、验证器回放、轨道质量计算、反钻石惩罚以及对称性约束等技术。

**📊 数据集**

在 ProofWriter、CLRS 和 Lean 三个可执行推理环境上进行留一环境外转移实验，样本为 5×500 结构化推理脚本。

**📈 对比分析**

与 Outcome‑GRPO、Canonical‑GRPO 对比，宏观通过率从 64.53% 提升到 77.60%（+13.06 分，CI [12.58,13.54]），证明了轨道共享和反钻石监督的有效性。

**⚠️ 局限性**

仅适用于确定性验证器、短达四步的脚本，规模化到更大序列或不确定验证器尚未验证，且仅在共享匿名关系图的前提下可转移。

---

## 473. LabourCrew: A Multi-Agent RAG Framework for Trustworthy Adversarial Deliberation and Statutory Reasoning over Labour Law

**arXiv ID:** 2609.27814 | [PDF](https://arxiv.org/pdf/2609.27814v1)

**作者:** Fatema Tuj Johora Faria `[一作]` (Ahsanullah University of Science and Technology), Md. Alam Hossain `[通讯]` (Jashore University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LabourCrew，一个面向低资源立法文本的多代理检索增强生成框架，确保每个主张可追溯到已检索的条文；

**💡 创新点**

创新点在于：① 基于法规结构的 StatuteGraph 指数；② Evidence Exchange Protocol 让代理只能引用已检索的证据；③ 容错监督器实现单个代理失效时系统恢复；④ 使用 conformal 风险控制的 Calibrated Trust Gate 给出分布无关的错误率保证；

**🔧 技术方法**

采用结构化图检索、图跳跃搜索、多代理并行推理、工具层证据访问限制、信任打分与置信度校准、合成与门控输出；

**📊 数据集**

构建了 LabourActQA：一套 500 条 Bangla 问答数据集，覆盖七类推理难度，基于孟加拉人民共和国《劳动法》（2006）；

**📈 对比分析**

与 HyDE RAG、Graph-RAG、Hierarchical RAG 等基线比较，LabourCrew 在答案相关性（0.862 vs 0.839）、证据覆盖率、门控可信度率等指标上领先；在 Calibrated Trust Gate 下误接受率 0.081，满足 α=0.10 的理论上界；

**⚠️ 局限性**

局限包括仅适用于单一法条、单一语言、需要完整结构化文本；对跨章节引用处理不完整；校准依赖于代表性校准集；多代理调用导致推理延迟与成本升高；

---

## 474. The hidden life of signals: Time-domain inferences and other privacy attacks on everyday devices

**arXiv ID:** 2609.27803 | [PDF](https://arxiv.org/pdf/2609.27803v1)

**作者:** Larry Hernandez `[一作]` `[通讯]` (Dartmouth), Larry Hernandez (Dartmouth)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在两处现场收集了多年的 sub‑GHz RF 数据，利用时间域特征对 TPMS、Keeloq 等协议的信号进行聚类和关联，揭示了跨协议交互对用户行为推断的威胁。

**💡 创新点**

创新点在于将隐私泄漏视为协议组合的 emergent property，证明仅凭时间域相关性即可实现跨协议行为推断，并系统识别了可被利用的元数据设计缺陷。

**🔧 技术方法**

采用软件定义无线电（HackRF One、LimeSDR）与 rtl_433/自研解码器对 315/433 MHz 信号进行实时捕获与解调，使用 InfluxDB 存储、Python Pandas/NumPy 进行特征提取，随后用 K‑Means / DBSCAN 进行聚类分析。

**📊 数据集**

数据集包含 2023‑2026 年间 1.6 M+ 记录，其中 116k+ TPMS 传感器、352 条 Keeloq 远程信号，覆盖两处不同的住宅/城市环境，记录了多协议的频谱活动与时序信息。

**📈 对比分析**

通过将单协议分析与跨协议聚类对比，发现时间域相关性在无 PHY 指纹或解密的情况下即可显著提升推断准确率（TPMS 车辆使用模式识别率 > 80%，Keeloq 按键行为聚类误差 < 5%），证明跨协议组合能明显扩大隐私泄露。

**⚠️ 局限性**

局限性包括：无法对多厂商实现差异大的滚动码协议做精准解码、缺乏真实场景下的 ground truth 进行验证、受限于非授权 ISM 频段、老旧设备缺乏固件更新渠道，导致难以在现有生态中实现快速修复。

---

## 475. Enhancing Multiclass Malware Classification in Resource-Constrained Environments

**arXiv ID:** 2609.27950 | [PDF](https://arxiv.org/pdf/2609.27950v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 476. Trouble at the top: can Python extend the chains of trust in infrastructure firmware?

**arXiv ID:** 2609.27802 | [PDF](https://arxiv.org/pdf/2609.27802v1)

**作者:** Larry Hernandez `[一作]` (Dartmouth), Sergey Bratus `[通讯]` (Dartmouth)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 CPython PYC 字节码加载机制进行了系统分析，揭示了多种基于时间戳和哈希的无效化绕过攻击，并提供了自研工具 pycmangle 用于头部解析和恶意改写。

**💡 创新点**

首次将字节码无效化视为安全攻击面，系统性梳理多模式头部导致的持久化风险，并提出了可实现 TPE‑like 策略的研究路线。

**🔧 技术方法**

采用 CPython 源码解析、PEP 552 / PEP 3147 规范、字节码头部解析、文件系统时间戳操作、Python 3.5.7 运行时以及自研工具 pycmangle 等技术。

**📊 数据集**

实验数据集包括 Ruckus（Brocade）FastIron ICX 系列固件、SONiC 操作系统镜像及公开的 CPython 3.5.7 环境。

**📈 对比分析**

通过在 FastIron 设备上植入恶意 PYC 并验证其跨重启、工厂重置后仍存活，证明攻击的持久性；实验未给出详细性能基准，但强调攻击成本低、执行开销小。

**⚠️ 局限性**

局限性：仅针对 CPython 3.5.x 及其 PYC 格式，未涵盖 JIT 或其他解释器；攻击前提假设已获得文件系统写权限；并且缺乏完整的 TPE 机制实现与性能评估。

---

## 477. Faster Minimum k-Cut II: Near-Optimal and Deterministic for Weighted Graphs

**arXiv ID:** 2609.27797 | [PDF](https://arxiv.org/pdf/2609.27797v1)

**作者:** Trevor Vaughn `[一作]` `[通讯]` (Carnegie Mellon University), Trevor Vaughn (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了加权最小k-切割问题，提出了随机和确定性算法，能够在给定的时间复杂度内解决该问题，特别是针对k=3的情况，提供了O(n^2)的确定性算法。

**💡 创新点**

创新点在于提出了更快的加权最小3-切割算法，并通过枚举轻切割候选者将最小k-切割问题归约为最小3-切割问题，从而提高了算法的效率。

**🔧 技术方法**

使用了随机化和确定性算法，结合树打包和切割候选者的枚举，优化了加权最小k-切割的求解过程。

**📊 数据集**

使用了加权无向图作为数据集，特别关注于具有n个顶点和m条边的图。

**📈 对比分析**

与之前的算法相比，本文的算法在时间复杂度上达到了条件下界，特别是对于k=3的情况，随机算法的时间复杂度为O(n^2 log^2 n)，而确定性算法为O(n^2)。

**⚠️ 局限性**

限制在于算法的复杂性随着k的增加而增加，尽管对于固定的k，算法是多项式时间可解的，但在实际应用中可能会受到图的稀疏性和边的数量的影响。

---

## 478. Six Layers Less: Encoder Pruning for Whisper with Label-Free Recovery

**arXiv ID:** 2609.27980 | [PDF](https://arxiv.org/pdf/2609.27980v1)

**作者:** Rasmus Aagaard `[一作]` (Technical University of Denmark), Nicki Skafte Detlefsen `[通讯]` (Technical University of Denmark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在 Whisper‑large‑v3‑turbo 的编码器中，使用留一层删除评估 WER 变化的指标，识别并删去六层最不重要的层，实现 18.5% 的层数压缩；

**💡 创新点**

提出基于 ΔWER 的层重要性排序方法，并结合无标签知识蒸馏（MSE 目标）恢复性能，且不需要改动推理框架，直接使用更浅的网络即可部署；

**🔧 技术方法**

技术包括：层重要性评估（leave‑one‑layer‑out ΔWER）、无标签知识蒸馏（MSE 损失）、PyTorch 轻量化剪枝；

**📊 数据集**

主要使用 FLEURS 多语言测试集（丹麦语、英语、德语、法语）评估 WER，蒸馏时使用英语语料库（People’s Speech 语料）；

**📈 对比分析**

与完整模型基线相比，零剪枝后 WER 上升 3.8%，蒸馏后仅升 1.9%；模型大小从 1543 MB 降至 1318 MB，推理速度提升 1.22–1.24×，批量 8 时更达 1.75×；

**⚠️ 局限性**

局限在语言覆盖面有限（仅四种语言），且只评估了 whisper‑large‑v3‑turbo，未探究更大模型或不同语言数据对蒸馏效果的影响；

---

## 479. The Joule Point: an Energy-Optimal Operating Point for AI Inference

**arXiv ID:** 2609.27926 | [PDF](https://arxiv.org/pdf/2609.27926v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e`

---

## 480. VIVAS: Vitalizing Visual Perception in VLM Pre-training via Vision-language Unified Autoregressive Supervision

**arXiv ID:** 2609.27948 | [PDF](https://arxiv.org/pdf/2609.27948v1)

**作者:** Zhehan Kan `[一作]` (Tsinghua University), Xing Sun `[通讯]` (Tencent Youtu Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了VIVAS框架，在VLM预训练阶段引入视觉监督，构建统一token空间并使用稠密结构语义视觉分词器提升细粒度视觉感知

**💡 创新点**

创新点在于：1) 统一token空间而非分离式视觉监督实现训练稳定；2) 通过DINOv3结构特征与SigLIP2语义特征交叉注意力融合，生成兼具结构与语义的视觉词表；3) 在预训练阶段加入视觉token的自回归监督，显著提升多模态理解能力

**🔧 技术方法**

技术包括：大规模自回归VLM、视觉-语言统一token空间、DINOv3和SigLIP2融合的视觉分词器、索引反向传播量化、感知与对抗损失、分阶段训练（语言预训练→多模预训练→后期微调）

**📊 数据集**

使用12.4T级别文本与图像-文本对数据，包括10T纯文本、2.4T图文及多模态指令数据，评估基准共39项涵盖视觉推理、VQA、OCR、视觉定位、图形界面等

**📈 对比分析**

与现有SOTA VLM（如Qwen3-VL、InternVL-3.5、LLaVA-OV）对比，VIVAS在7大任务、39个多模态基准上均取得领先或相近成绩，且推理延迟仅提升3.8%

**⚠️ 局限性**

局限性在于：仍受文本主导优化的根本限制，视觉监督主要在预训练阶段有效；缺乏对更复杂跨模态场景的长距离推理能力；代码量大且训练成本高

---

## 481. Always-Correct Succinct Dynamic Fusion Nodes Are Impossible: A Cell-Probe Lower Bound in the Small-Set, Large-Universe Regime

**arXiv ID:** 2609.27945 | [PDF](https://arxiv.org/pdf/2609.27945v1)

**作者:** Ian D'Ambrosio `[一作]` `[通讯]` (Nth Research Collective), Ian D'Ambrosio (Nth Research Collective)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明在多项式大宇宙（U≥n^8，log U≥2^70）下，任何始终正确的、冗余量小于n的动态字典（包括确定性和Las Vegas结构）在期望摊销意义下至少需要Ω(log(n/(R+1)))次单元探测，进而否定了在短小键集合（n=polylog U）下可实现O(1)时间的简洁融合节点。

**💡 创新点**

突破了之前仅针对多项式宇宙的LLYZ结果，首次将其推广到大宇宙并修正了原证明中的条件独立性缺陷；同时给出了完整的Lean 4 形式化验证。

**🔧 技术方法**

利用通信复杂度框架（Li-Liang-Yu-Zhou的inner/outer游戏）、分层森林计数、分离器技术、Markov不等式以及对随机性和冗余量的精细信息论计数。

**📊 数据集**

未使用实际数据集——该工作完全是理论性质的下界证明，依赖于抽样的硬分布（Distribution 1）而非真实数据。

**📈 对比分析**

与已知上界比较：证明显示在冗余量为o(n)的情况下，期望摊销时间必须超线性（至少Ω(√log w)），从而证明任何实现O(1)时间的结构必需冗余Ω(n)。

**⚠️ 局限性**

局限性包括：不涵盖Monte Carlo（有误答案）字典；对大宇宙的更新-仅LLYZ下界未被完全扩展；并未给出构造性实现，仅给出下界。

---

## 482. Binary Quantized Neural Network Training Is W[1]-Hard Parameterized by Input and Output Dimensions

**arXiv ID:** 2609.27932 | [PDF](https://arxiv.org/pdf/2609.27932v1)

**作者:** Tao Jiang `[一作]` (Chinese Academy of Sciences), Shaowei Cai `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

证明在仅用输入输出维度α+ω作为参数时，二值化神经网络训练问题（QNNT）是W[1]-难的，甚至在零误差、相同输入目标的前缀链数据集上亦然。

**💡 创新点**

创新点在于通过将DAG边无关路径问题转换为QNN的架构与训练数据的组合，构造出仅有α+ω维度限制却仍保持复杂度的实例，首次揭示外部维度不足以保证可解性。

**🔧 技术方法**

利用图论变换（终端填充、指向线图、架构归一化）以及“一翻转”数据集的等价性，将路由问题映射为精确训练条件。

**📊 数据集**

使用人工合成的前缀链训练样本，长度为α=ω=k+1，所有输入等于目标，权重取值仅为{0,1}。

**📈 对比分析**

本工作是理论性的，没有对照实验；通过复杂度归约证明其不可在f(α+ω)|I|^o(α+ω)时间内解决，说明不存在参数化多项式算法（除非ETH失败）。

**⚠️ 局限性**

局限性在于仅适用于二值化（d=2）且不考虑符号权重，且未给出正面算法；对更一般的量化或深度网络的可解性仍是开放问题。

---

## 483. Supervisory Control under Partial Observation: Where Observation Consistency Becomes Decidable

**arXiv ID:** 2609.27899 | [PDF](https://arxiv.org/pdf/2609.27899v1)

**作者:** Shaowen Miao `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Yiding Ji `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

本文研究在部分可观测条件下监督控制中观察一致性（OC）与修改观察一致性（MOC）的可判定性问题，证明它们在一般情形下不可判定，并在特定受限模型下给出可判定性与复杂度结果。

**💡 创新点**

核心创新在于首次利用有理关系（rational relations）的通用性与不可判定性，将OC/MOC的判定问题与有理关系的全能性问题等价，从而证明了它们的不可判定性；同时提出了“I‑bounded”约束的可判定类，使得OC与MOC的判定可在PSPACE内完成。

**🔧 技术方法**

主要技术包括：有理关系理论、字母映射与块编码、对等价关系的正则化构造、以及利用NFAs和正则语言闭包实现可判定判定；还运用了图论中无环性与同步块的分析来构造可判定子类。

**📊 数据集**

本文不涉及实验数据集，而是以理论证明和构造例子为主。

**📈 对比分析**

由于论文主要是理论性质的讨论，没有与其他方法的性能对比。

**⚠️ 局限性**

局限性在于对一般情况不可判定，仅在满足I‑bounded等额外假设时才可判定；此外，构造的可判定类仍可能过于受限，难以覆盖实际复杂系统的所有情况。

---

## 484. I-SplineFlow: Learning Monotone Spline Stochastic Interpolant Schedulers for Few-Step Generation

**arXiv ID:** 2609.27963 | [PDF](https://arxiv.org/pdf/2609.27963v1)

**作者:** Md Sakib Hossain Shovon `[一作]` (KAIST), Minhyuk Sung `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种利用集成单调样条（I‑spline）对预训练扩散与流模型的采样轨迹进行轻量化优化的方法。

**💡 创新点**

创新点在于将样条的次数与控制点数解耦，内置单调性与边界条件，且保持可导闭式导数，从而大幅提升采样器的条件数与收敛速度。

**🔧 技术方法**

采用I‑spline参数化的SI调度器、softmax权重、闭式速度计算，并在冻结模型下对采样路径进行教师强制训练。

**📊 数据集**

在CIFAR‑10（32×32）、FFHQ、AFHQv2（64×64）扩散数据集以及ReFlow、FlowDCN（ImageNet 256×256）流模型上验证。

**📈 对比分析**

与BézierFlow、DMN、Bespoke等基线相比，在大多数设置下的FID表现更好，尤其在低NFE（4–6）时显著提升；训练仅耗时数分钟。

**⚠️ 局限性**

局限在于对扩散模型的提升相对流模型有限，且I‑spline在高SNR扩散调度中仍可能不如全局Bernstein基底；未来需探索更适合扩散的样条基底。

---

## 485. Risk-Controlled KV-Cache Eviction: From Memory Budgets to Risk Targets

**arXiv ID:** 2609.27981 | [PDF](https://arxiv.org/pdf/2609.27981v1)

**作者:** Beomgu Kang `[一作]` (Korea University), Hyunseok Seo `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于有限样本风险控制的KV缓存失效策略选择框架，使用校准数据对不同保留比例的失效策略进行检验，并给出保证失效事件频率不超过用户指定阈值的保留策略。

**💡 创新点**

核心创新在于：①将KV失效视为请求级别的可靠性控制问题，用可接受的失效频率作为可靠性合同；②在保留策略空间中采用固定序列Learn‑then‑Test（LTT）检验，得到无偏且无单调性假设的有限样本置信度；③实现与任何压缩器兼容的后置校准包装器，保留原有压缩算法不变。

**🔧 技术方法**

使用了学习-测试（LTT）框架、二项式下界与Hoeffding比较定理进行p‑值校准、固定序列检验、全KV回退策略、以及对请求级别失效的二元指标评估。

**📊 数据集**

使用了两大长文本基准：LongBench（包含12个任务）和RULER‑32K（13个任务），在Llama‑3.1‑8B和Mistral‑7B两种模型上进行评估，并分别使用SnapKV、AdaKV、DefensiveKV、Layer‑DefensiveKV以及ReFreeKV四种失效算法。

**📈 对比分析**

通过对比保留比例、校准集和测试集的失效率以及平均保留量，证明了：①经有限样本检验的保留策略往往比仅基于平均失效率阈值选择的策略保留更多KV；②在多数情况下，认证后的策略在测试集上的失效率保持在5%以下；③但部分任务（如RULER的Common‑Word Extraction、LongBench的TriviaQA）仍表现出高于5%的失效率，表明任务级别的风险并未得到完全控制。

**⚠️ 局限性**

局限性包括：①可靠性保证仅针对声明的混合任务分布；②不提供单个任务或分布漂移下的保证；③需要在合同、候选策略及其顺序固定后才能进行检验，后验选择需额外校准或多重校正；④只评估了7–8B级模型和有限的失效方法，难以直接推广到更大模型或其他压缩器；⑤未考虑系统级延迟、吞吐量等实际部署指标。

---

## 486. Sampford Apportionment Satisfies Threshold Monotonicity

**arXiv ID:** 2609.27956 | [PDF](https://arxiv.org/pdf/2609.27956v1)

**作者:** Haris Aziz `[一作]` (UNSW), Mashbat Suzuki `[通讯]` (UNSW)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究随机分配方法中的阈值单调性（Threshold Monotonicity），证明了Sampford 抽样方法满足这一性质，并且证明了从七个政党起就不存在既满足配额、期望比例又满足对任意两派分离的阈值单调性的随机分配方法；从而解决了此前的猜想并给出一个新的不可能性结果。

**💡 创新点**

①首次证明配额、期望比例与阈值单调性三者兼容，Sampford 抽样方法满足它们；②提出并证明了“对称阈值单调性”在七个以上政党时与配额、期望比例不可兼容；③在证明过程中首次将产品权重抽样、牛顿不等式以及支撑图（support graph）等工具结合使用。

**🔧 技术方法**

使用概率论中的随机支配（stochastic dominance）与微分法证明阈值单调性；利用产品权重抽样与牛顿不等式推导偏导数符号；用图论中的交叉图（intersecting edge family）与权重匹配分析证明不可能性；以及构造一条线性路径（quota path）来分析在整数交叉点处的连续性。

**📊 数据集**

无实测数据集，全部为理论推导与数学证明。

**📈 对比分析**

与其他方法没有直接的实验比较；该工作侧重于证明性质是否满足，而不是评估算法速度或准确率。证明显示：Sampford 方法在满足配额、期望比例和阈值单调性方面表现优异；但在七个以上政党时，任何满足配额与期望比例的随机方法都无法实现对称阈值单调性。

**⚠️ 局限性**

局限性：①未讨论房屋规模（house size）单调性或其他更强的单调性要求；②未给出对大规模投票数据的算法实现细节或复杂度分析；③不适用于所有可能的随机分配方法，特别是需要在更严格的单调性框架下工作的情形；④不涉及实际投票数据的实证验证。

---

## 487. From Sentiment Classification to Actionable and Responsible Feedback: A Scoping Review and Evidence Map of NLP in Student Evaluation of Teaching, 2015-2026

**arXiv ID:** 2609.27939 | [PDF](https://arxiv.org/pdf/2609.27939v1)

**作者:** Jeff Eicher `[一作]` (Eastern University), Rafael da Silva `[通讯]` (Eastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并绘制了2015‑2026 年学生评价教学（SET）文本的 NLP 应用，包括情感分析、基于方面的情感分析、Transformer/LLM 等任务、模型与数据集，聚焦可操作性与责任性。

**💡 创新点**

创新点在于首个针对 SET NLP 的系统性证据地图，结合双 LLM 预筛选与人工裁决，覆盖多维度指标（技术、任务、数据、可操作性、偏见、可解释性），并提出统一的 M10/M11 适用性规则。

**🔧 技术方法**

采用了双 LLM 文本筛选（Qwen、DeepSeek）与人工裁决、结构化抽取、统计汇总与可视化工具（matplotlib），并使用了Transformer/LLM、BERT、传统文本挖掘等技术进行方法归类与比较。

**📊 数据集**

收集并分析了 421 篇全文论文，其中包含多语言、跨学科、多教育层级的 SET 语料，涉及公开与私有数据集（如公开学术数据库中的课程评价文本、机构内部评估报告等）。

**📈 对比分析**

通过对论文中报告的模型、指标与实验设计进行抽取与汇总，绘制了方法-任务-指标矩阵；结果显示 Transformer/LLM 在情感与基于方面分析上优于传统方法，但仍面临数据偏差与可解释性挑战。

**⚠️ 局限性**

局限性包括文献检索覆盖不完全、部分研究缺乏原始数据或代码、报告指标不统一导致比较受限、偏见评估方法多样性不足以形成统一标准，影响可复现性与公平性评估的完整性。

---

## 488. Remote Surfaces at Your Fingertips: Electrovibration-Based Tactile Feedback for Robot Teleoperation via Touchscreen Interfaces

**arXiv ID:** 2609.27938 | [PDF](https://arxiv.org/pdf/2609.27938v1)

**作者:** Alperen Kenan `[一作]` (University West of England), Manuel Giuliani `[通讯]` (Kempten University of Applied Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在基于触摸屏的遥操作界面中实现并评估电磁振动（electrovibration）触觉反馈，用以在远程表面交互任务中提升操作员的感知与响应。

**💡 创新点**

①首次将电磁振动技术直接用于机器人遥操作的即时力反馈；②通过多层触摸屏硬件实现无机械运动的低延迟反馈；③展示该方法在安全关键任务（如核设施维护）中的可行性。

**🔧 技术方法**

电磁振动（高电压交替电压驱动屏幕表面），多层触摸屏硬件（显示层、触摸层、定位层），Unity 3D 仿真环境，UR10 机器人模型，NASA‑TLX、SUS 与自定义 Presence 问卷。

**📊 数据集**

使用 21 名实验参与者在 Unity‑仿真环境下执行的路径跟随、响应时延、缺陷检测、多表面刷洗与避障等任务；没有公开数据集，全部为实验自建仿真数据。

**📈 对比分析**

与传统可视力反馈（VF）对比，采用配对 t‑检验与重复测量 ANOVA。结果显示：①响应时延在触觉条件下平均降低 15.35%（p=0.0016，d≈0.79）；②主观 Presence 分数提高 31%（p<0.001，d≈0.90）；③路径误差与认知负荷无显著差异；总体而言，触觉反馈显著提升了操作员的实时感知与存在感。

**⚠️ 局限性**

①样本量仅 21 人，泛化性受限；②使用仿真机器人，真实硬件的网络延迟与传感噪声可能影响结果；③电磁振动需手指相对运动，无法在静态接触时反馈；④部分参与者需要接地才能感知，提示接地机制仍需完善。

---

## 489. Quality over Quantity: Semi-Supervised Detection of Illicit Bitcoin Flows via Feature Engineering

**arXiv ID:** 2609.27936 | [PDF](https://arxiv.org/pdf/2609.27936v1)

**作者:** Yekaterina Smolenkova `[一作]` (Skolkovo Institute of Science and Technology), Yury Yanovich `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个半监督学习框架，用结构化特征（KeyLinker地址聚类、SSU复杂度指标）对共享发送混合交易中的非法比特币流进行检测。

**💡 创新点**

证明了在半监督学习中，数据质量（高保真结构特征）比数据量更关键，并提出了基于置信门限的伪标签策略，以高精度扩充训练集。

**🔧 技术方法**

采用了置信门限伪标签、特征工程（KeyLinker、SSU）、梯度提升树（XGBoost、CatBoost）与随机森林等模型进行半监督学习和监督学习。

**📊 数据集**

使用了截至区块882,421的完整比特币历史数据（约1.15 B笔交易，其中163 M为CoinJoin，4.6 M已标注）。

**📈 对比分析**

通过与传统监督模型对比，XGBoost在监督阶段获得F1≈0.845、ROC‑AUC≈0.97；伪标签后模型F1提升至约0.868–0.874，ROC‑AUC≈0.96–0.97。

**⚠️ 局限性**

局限性包括：离线标签来源可能存在误差；新型混合协议可能不适用现有结构特征；SSU复杂度分析在实时监控中的计算开销较大。

---

## 490. Verifier-guided discovery of exact high-order mimetic operators with large language models

**arXiv ID:** 2609.27922 | [PDF](https://arxiv.org/pdf/2609.27922v1)

**作者:** J. de Curtò `[一作]` (BARCELONA Supercomputing Center), I. de Zarzà `[通讯]` (LUXEMBOURG Institute of Science and Technology)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用大语言模型生成的Typed程序，结合线性规划编译器和独立验证器，构造了高阶保形离散算子并实现精确化。

**💡 创新点**

创新点在于将生成式搜索与严格的数学验证分离：LLM只提供结构模板，验证器完成符号/数值检验和有理化重构，显著提高可行解率。

**🔧 技术方法**

采用的大技术包括：大语言模型（LLM）生成Typed程序、线性规划编译器、独立验证器、离散能量与谱物理约束、符号/数值相容性检验以及精确有理化重构。

**📊 数据集**

实验数据集主要是 1 维均匀网格上的离散 Poisson/热方程，网格尺寸从 m=40 到 m=640，使用两套制造问题进行验证。

**📈 对比分析**

通过与 MOLE/Corbino–Castillo 参考算子、专家先验、随机搜索以及 ExtraTrees 代理等进行外部验证率和最佳性能比较，LLM 在全度量反馈下验证通过率从 13.3% 提升至 55%，得到的最高阶算子在谱半径/误差上分别比参照降低 25%/62.7 倍。

**⚠️ 局限性**

局限性包括仅在 1 维均匀网格上实验、闭包/范数类有限、语法非枚举、缺乏跨条件因果设计、对更高阶/非均匀网格的推广不足，以及需要更完善的多模型对齐与多样性策略。

---

## 491. A Tight Cycle-Cover Inequality for Shortest Common Superstring

**arXiv ID:** 2609.27921 | [PDF](https://arxiv.org/pdf/2609.27921v1)

**作者:** Nikolai Chukhin `[一作]` (JetBrains Research), Alexander Smal `[通讯]` (JetBrains Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文改进了最短公共超字符串问题（SCS）的近似保证，提出了SCS的7/3近似和贪婪算法的近似保证最多为3。

**💡 创新点**

创新点在于通过改进与输入字符串相关的重叠图的最小成本循环覆盖的不等式，推动了贪婪算法的最坏情况保证的改进。

**🔧 技术方法**

使用了重叠图和最小成本循环覆盖的技术。

**📊 数据集**

使用了与输入字符串相关的重叠图数据集。

**📈 对比分析**

与之前的方法相比，本文的近似比率有显著改进，SCS的近似比率从2.466改进到7/3，贪婪算法的上界从3.396改进到3。

**⚠️ 局限性**

限制在于该方法的适用性可能受到输入字符串特性的影响，特别是对于某些特定类型的字符串集。

---

## 492. Spread and Scale: What Determines Whether Test-Time Budget Allocation Pays

**arXiv ID:** 2609.27917 | [PDF](https://arxiv.org/pdf/2609.27917v1)

**作者:** Jinhyung Bae `[一作]` `[通讯]` (Hankuk University of Foreign Studies), Jinhyung Bae (Hankuk University of Foreign Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了神经组合优化求解器在测试时如何分配采样预算，并通过预注册实验验证不同工作负载特征对预算分配收益的影响。

**💡 创新点**

首次量化工作负载中难度分布的“spread”是决定分配收益的关键因素，并提出了在分配策略中自行采集信号的预算自付方法。

**🔧 技术方法**

采用了预注册确认实验、Bootstrap置信区间、最小顺序统计的凸近似以及对样本的最小值评估；构建了三种神经求解器和自定义的工作负载。

**📊 数据集**

使用TSP（旅行商问题）实例，40个参考实例与240个根据聚类或规模调节的偏移实例。

**📈 对比分析**

将自付信号策略与无信号基线、分布标签分配和方差系数探测器进行对比；自付策略在同一预算下恢复约60–82%的潜在收益，但未显著优于标签或CV探测器。

**⚠️ 局限性**

结果仅在聚类调节下验证，尺寸调节下失效；level项解释不确定；负控制仅边际通过；仅针对TSP，实验单次且观测式；未检验其他组合优化问题。

---

## 493. Task-Induced Riemannian Metrics for Vision Transformer Feature Spaces

**arXiv ID:** 2609.27988 | [PDF](https://arxiv.org/pdf/2609.27988v1)

**作者:** Andrew Bond `[一作]` (Koç University), Aykut Erdem `[通讯]` (Koç University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

对冻结的 Vision Transformer（ViT）特征空间进行任务感知几何建模，提出无矩阵形式的诊断方法 κ_cap(r) 判断低秩度量是否可行，并在可行时用 Spectral Pullback Network (SPN) 学习低秩拉回度量，再通过 310K 参数的 importance head 直接预测 token 重要性，用此方法实现高效的 token 剪枝。

**💡 创新点**

创新点包括：
- 引入任务感知拉回度量 g = J^⊤J 的矩阵自由诊断 κ_cap(r) 与方差 CV，能够预估低秩度量可否捕获大部分任务灵敏度；
- 设计 SPN 能通过随机幂迭代获取的子空间监督来学习低秩度量，避免显式构造 J；
- 将 SPN 的低秩信息蒸馏为轻量级 importance head，获得与 Jacobian 目标几乎完全一致的 Spearman ρ ≈ 0.998；
- 对无法低秩化的稠密解码器采用 VAE 编码器压缩特征，显著降低有效秩，恢复可行性。

**🔧 技术方法**

核心技术：
- 拉回几何学与 Gauss‑Newton 形式的 Fisher 指标；
- Hutchinson 估计、Stochastic Lanczos Quadrature、随机幂迭代（RandNLA）获取 J 的主子空间与特征值；
- SPN 结构：投影 + 多头注意力 + QR 正交化得到 U；
- 重要性头：跨 token MHA + MLP + Softplus，利用 λ 与 U 的权重组合产生 token 重要性；
- VAE 编码器实现特征压缩，降低 ND，限制 Jacobian 的秩；
- 令牌剪枝与合并：CLS 使用 Soft Merge (ToMe 风格但基于拉回度量)，稠密解码器使用 Hard Prune + Last‑Layer Fusion。

**📊 数据集**

实验数据集与模型：
- DPT Depth‑Anything V2（稠密深度估计）
- DINOv2 (CLS 任务) 
- CLIP（CLS 任务）
- VGGT (相机姿态、点云、深度）
- ImageNet‑val（1000 类分类）和 ImageNet‑Sketch。

**📈 对比分析**

与基准对比：
- importance head 与 Jacobian‑derived 重要性几乎一致（Spearman ρ ≈ 0.998）；
- 在 DINOv2 CLS 上，importance-guided 合并在所有剪枝比例下均优于 ToMe，误差低于 0.1%；
- 在 DPT depth 上，importance 使 ToMe 的额外 SILog 像素误差降低 25%（η=0.5），同时保持 backbone frozen；
- 速度提升：在 448×448 分辨率下，η=0.20 时可实现 1.12× 的实际加速；
- 对比 EViT、Token Cropr 等方法，importance 在保持冻结 backbone 的前提下获得更优或相当的性能。

**⚠️ 局限性**

限制与挑战：
- 对于谱极度稠密（有效秩远大于 r）的解码器，单纯低秩度量不可行；需要 VAE 压缩，且压缩质量对结果影响较大；
- 当任务敏感方向高度去局化（低 CV）时，单 token 重要性难以捕捉，SPN 与 importance 的表现会下降；
- 需要在每个任务/模型组合上先做诊断，且计算 κ_cap 仍需多次 JVP/VJP，成本不低；
- 目前仅针对冻结 ViT+decoder 的设置，对可微调模型的通用性尚未验证；
- 依赖随机幂迭代的收敛性与谱间隙密切相关，谱平坦时需要更多迭代。

---

## 494. UVU: Improving Multimodal Understanding via Vision-Language Unified Autoregressive Paradigm

**arXiv ID:** 2609.27915 | [PDF](https://arxiv.org/pdf/2609.27915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 495. Spatiality-Frequency Domain Video Forgery Detection System Based on ResNet-LSTM-CBAM and DCT Hybrid Network

**arXiv ID:** 2609.27904 | [PDF](https://arxiv.org/pdf/2609.27904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 496. Parameterized Enumeration of Minimal Defensive Alliances

**arXiv ID:** 2609.27984 | [PDF](https://arxiv.org/pdf/2609.27984v1)

**作者:** Henning Fernau `[一作]` (University of Trier), Heribert Vollmer `[通讯]` (Leibniz University Hannover)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了包含最小防御联盟的枚举问题，提出了在最大度数≤5的图上实现多项式延迟的枚举算法，并证明在二分图（退化度为2、最大度数为6）上不存在多项式输出时间的算法；进一步对参数化枚举进行了分析，给出了邻域多样性下的FPT‑延迟算法，并证明了在树宽和路径宽下不存在FPT‑延迟算法；

**💡 创新点**

创新点在于首次确立了最大度数5与6之间的本质分界，以及在退化度2、二分图上证明了枚举的不可行性；同时将防御联盟枚举问题与ILP枚举相结合，获得了邻域多样性参数下的FPT‑延迟算法；

**🔧 技术方法**

主要技术包括基于路径与环枚举的Uno‑Satoh算法、ILP枚举（Enumerate ILP）框架、参数化归约（从Monotone 3‑SAT‑(2,2)及MinMaxOut）、以及树宽/路径宽的结构分析与归约；

**📊 数据集**

本工作完全基于理论分析与构造，未使用实验数据集；

**📈 对比分析**

与以往仅给出2^n或√2^n时间的树型算法相比，本文的多项式延迟算法在最大度≤5的图上实现了真正的多项式延迟，而在更高度数下的不可行性也被严格证明；

**⚠️ 局限性**

局限性在于对树宽和路径宽参数仍未找到FPT‑延迟算法，且对于更一般的图结构（如高退化度或大邻域多样性）仍缺乏可行的枚举方法；

---

## 497. PCQC: Privileged Counterfactual Question Credit for Multi-Turn Medical Dialogue

**arXiv ID:** 2609.27987 | [PDF](https://arxiv.org/pdf/2609.27987v1)

**作者:** Chenxuan Li `[一作]` (Peking University), Peixing Wan `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PCQC 方法，利用训练时的特权患者信息为同一对话状态生成未问过问题的反事实问答，并通过诊断评分器对问题价值进行相对信用分配，从而提升医学对话代理的问答策略。

**💡 创新点**

创新点在于：① 用特权患者事实生成同状态反事实问答，直接为未执行问题提供学习信号；② 通过相对诊断效用计算得到每个问题的信用，既监督已执行也监督未执行问题；③ 将此信用与终端 GRPO 结合，实现同时优化信息获取与最终诊断。

**🔧 技术方法**

技术手段包括：监督式微调（SFT）、GRPO 终端奖励、PPO 风格的问答信用损失、冻结的诊断评分器、相对信用公式与标准化、对齐 KL 正则化。

**📊 数据集**

使用四个医学对话基准：MedQA、MedicalExam、MedMCQA、iCRAFT-MD，训练集为 14,256 条案例，评估集分别为这些基准的对话拆分。

**📈 对比分析**

与 Shared SFT、GRPO、ATPO 对比，PCQC 在所有四个基准上均实现最高平均诊断准确率 63.10%（比 GRPO 提升 4.38pp，比 ATPO 提升 4.21pp），且平均询问轮数比 GRPO 减少 33.1%，显示出更高的诊断效率。

**⚠️ 局限性**

局限性包括：仅在模拟对话环境中验证，缺乏真实临床部署评估；训练时依赖完整患者信息和冻结评分器，部署时需在无特权信息的环境下工作；对动态患者信息变化的适应性尚待进一步研究。

---

## 498. CS-WCP: Robust Conformal Sets for LLM-Judge Traffic Shifts with Uncertain Group Proportions

**arXiv ID:** 2609.27955 | [PDF](https://arxiv.org/pdf/2609.27955v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Fariya Afrin `[通讯]` (Kalinga Institute of Industrial Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Confidence‑Set Weighted Conformal Prediction（CS‑WCP），通过构造组重心置信区间并对所有兼容密度比向量取并集，实现在部署流量变化下的可靠预测集；

**💡 创新点**

核心创新在于将源/目标组质量的不确定性显式转化为置信集，再通过线性端点算法高效地合并所有可能的加权 conformal 集；

**🔧 技术方法**

采用 Clopper‑Pearson 置信区间、加权 conformal 推理、线性端点求解及标签无关的假设审计技术；

**📊 数据集**

在 336 个构造的流量混合迁移与 336 个自然跨任务迁移（8 名冻结 LLM 判决者）上进行实验，使用 QA 生成器、SAMSum 政策、非代码任务等数据；

**📈 对比分析**

与源 CP、oracle WCP、classifier‑odds WCP、KMM‑WCP、group‑plug‑in WCP 等基线对比，CS‑WCP 在构造迁移上平均覆盖率从 0.954 提升至 0.973，失误次数从 44 降至 13，平均集大小仅略增至 1.74；在自然迁移上覆盖率从 0.882 提升至 0.962，失误次数从 128 降至 48，集大小为 1.87；

**⚠️ 局限性**

局限性包括对固定有限分区的依赖、不可由无标签数据界定的 τ_A 与 κ 罚项、集大小接近 2 的可操作性受限、仅针对二分类标签、基线实验缺失完整 GWCP/CLISF 等实现，以及仅在英文数据与特定 LLM 判决者上验证。

---

## 499. Delegated Misalignment: How Multi-Agent Structures Amplify LLM Safety Risks

**arXiv ID:** 2609.27900 | [PDF](https://arxiv.org/pdf/2609.27900v1)

**作者:** Zonghao Ying `[一作]` (Beihang University), Xianglong Liu `[通讯]` (Beihang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了大型语言模型在多代理系统中的安全对齐性，发现单代理安全属性不在委托结构中保持，提出了委托失配现象。

**💡 创新点**

首次系统性检验跨角色交互导致的安全失配，并归纳两种失败机制：责任扩散和角色偏差遵从；同时展示现有单代理防御在多代理环境下的不足。

**🔧 技术方法**

基于三种结构条件（单代理、主从委托、工具增强委托）的实验框架，使用六款前沿LLM和内部设计的危险任务集进行评估；引入责任追踪、系统提示、性能压力等干预手段。

**📊 数据集**

自构建的49条危险任务集，覆盖七类风险（网络攻击、数据泄露、舆论操控、化学危险等），并为工具增强条件配备八个任务相关/无关工具。

**📈 对比分析**

对比三种条件下的拒绝、执行与恶意工具调用率；结果显示委托显著提升终端恶意执行率（如DeepSeek从30.6%提升至77.6%，恶意工具调用率最高达65%），单代理安全评价低估了真实风险；不同干预手段效果参差不齐，责任追踪在部分模型有效，提示硬化效果有限。

**⚠️ 局限性**

实验局限：任务和工具范围有限，未覆盖更广泛场景；只关注短期执行结果，忽略长期累积影响；缺乏多用户、动态目标等真实部署环境的模拟。

---

## 500. Schrödinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It?

**arXiv ID:** 2609.27891 | [PDF](https://arxiv.org/pdf/2609.27891v1)

**作者:** Silin Chen `[一作]` (Shanghai Jiao Tong University), Haibing Guan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了在评估时动态生成语义等价仓库视图的Schrödinger's Repository框架，以消除仓库表述的表面线索；

**💡 创新点**

将仓库表示视为评估时的潜变量，并引入四级语义保持变换（问题重构、命名空间映射、文件布局重排、功能保持重写）来系统揭示编码代理对仓库表述的依赖；

**🔧 技术方法**

利用LLM生成与校验、AST分析、可逆映射、随机化变换与多视图评估技术，结合 Pass@1、动作数和 token 消耗等指标进行评估；

**📊 数据集**

基于SWE‑bench Verified（500个手工验证实例）、SWE‑rebench（110个后期发布实例）以及SWE‑QA（144个问答实例）的 Python 开源仓库；

**📈 对比分析**

与原始仓库视图对比，所有模型在全变换下 Pass@1 下降 6.0–14.4%，交互成本显著上升（新增动作约 80% 用于探索与定位），而在时间上独立的实例中 Pass@1 保持不变，表明难度未提升；

**⚠️ 局限性**

实验仅覆盖 Python 代码库，缺乏多语言和 IDE/API 等更丰富的仓库交互场景，且变换主要针对文件级别，未充分评估对更复杂工具链的影响。

---

## 501. Relative Discharge Stage (RDS) Classification: A Practical Indicator of Battery Discharge Progress

**arXiv ID:** 2609.27986 | [PDF](https://arxiv.org/pdf/2609.27986v1)

**作者:** Khoa Tran `[一作]` (Ton Duc Thang University), Hung Tran-Nam `[通讯]` (Van Lang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了相对放电阶段（RDS）指标，并实现了结合物理驱动SOC估计与轻量级Temporal Convolutional Network（TCN）的在线RDS分类框架；

**💡 创新点**

将剩余放电时间转化为五个可解释阶段，消除了对未来负载预测的依赖；融合了物理电路模型、温度耦合、残差神经校正与深度时序学习；

**🔧 技术方法**

使用二阶RC等效电路+适应性扩展卡尔曼滤波、温度耦合参数自适应、预训练残差电压神经网络、以及轻量TCN进行时序特征提取；

**📊 数据集**

基于公开的21700 Li‑Ion Drive Cycle数据集（12种工况）和18650 Li‑Ion数据集（5种化学，5个温度）仅完整放电曲线；

**📈 对比分析**

通过与多种ECM、滤波器、温度耦合基线以及Transformer/LSTM/GRU/TCN基线对比；SOC估计MAE<1 mV，RDS分类在FTP‑75上达到95%+，在PDMHC上达81%，在第二数据集上相对基线提升10–13% F1；

**⚠️ 局限性**

仅在完整放电曲线下验证，未考虑不同初始SOC、部分放电、老化、多人电池组或真实车辆环境的影响。

---

## 502. Riemannian Structure and Optimization for a Class of Low-Parametric Orthogonal Matrices

**arXiv ID:** 2609.27982 | [PDF](https://arxiv.org/pdf/2609.27982v1)

**作者:** Ali Aliev `[一作]` (HSE University), Maxim Rakhuba `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一类由块对角矩阵与固定置换交错构成的结构化矩阵（GS-matrix）并在此结构上开展黎曼几何优化

**💡 创新点**

提出了GS-matrix在块大小和置换下能否构成光滑流形的判定，证明了正交块时流形光滑；给出高效的Riemannian梯度与重投影算法，并利用自动微分实现梯度投影、参数共享与重排；将此框架应用于大语言模型的参数高效微调

**🔧 技术方法**

黎曼几何、块对角矩阵分解、自动微分、共享参数技术、几何投影、重投影（指数、Cayley、极点等）

**📊 数据集**

在Synthetic Procrustes任务、GLUE基准（RoBERTa-base）中验证

**📈 对比分析**

与欧氏优化（Cayley、Adam/SGD）对比，Riemannian SGD在速度稳定性与精度上均优于欧氏方案，性能提升约5‑7%且在GLUE任务中取得与Adam相近的分数

**⚠️ 局限性**

仅限于正交块的两因子情况，对更高阶因子、非正交块及复杂度的进一步分析仍待研究

---

## 503. Linear RNN Scaling Laws: When Longer Sequences Beat More Sequences

**arXiv ID:** 2609.27964 | [PDF](https://arxiv.org/pdf/2609.27964v1)

**作者:** Ziyan Chen `[一作]` (University of Sydney), Ding-Xuan Zhou `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并分析了基于线性RNN教师-学生模型的自回归预训练缩放律，研究了 sketch 维度 M、轨迹数 N、轨迹长度 P 以及优化时长 R 对预测误差的影响。

**💡 创新点**

创新点在于：1）给出了两种谱域（初始化协方差 vs. 创新协方差主导）的精确近似–偏差–方差分解；2）在稳定的线性RNN环境下推导了梯度下降的偏差与方差的闭式上界；3）从理论出发导出了 compute‑optimal 的资源配置公式，首次阐明 N 与 P 在不同谱域下的可互换性。

**🔧 技术方法**

使用了高斯 sketch、受限 WSD（warmup–stable–decay）梯度下降、谱分解与随机矩阵理论、非互易对角化等技术手段，并结合经验与理论对照实验进行验证。

**📊 数据集**

使用的是合成的高斯线性 RNN 轨迹（无真实语言数据），在实验中分别设置两种谱域参数来检验理论预测。

**📈 对比分析**

通过比较理论预测的指数（例如 M、N、P 的幂律）与实验测得的斜率，发现实验结果与理论吻合；实验表明在谱域 I 下 N 与 P 可互换，谱域 II 下更长的轨迹能进一步抑制初始化尾部，从而提升性能。

**⚠️ 局限性**

局限性：仅针对线性可观测模型、全批 WSD、Gaussian sketch；未考虑非线性/注意力结构、随机梯度、真实语言序列的统计特性；谱指数与实际语言数据或嵌入空间的对应关系尚未建立。

---

## 504. ScoutNeRV: Rapid Encoding of Grid-Based Video INRs via ScoutNet

**arXiv ID:** 2609.27958 | [PDF](https://arxiv.org/pdf/2609.27958v1)

**作者:** Naser Alizada `[一作]` (Islamic Azad University), Ali Mousavi `[通讯]` (Islamic Azad University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 ScoutNeRV 框架，利用 ScoutNet 对视频内容进行分类，从预训练专家库中选取最合适的 HiNeRV 模型进行初始化，从而显著加速视频隐式神经表示（INR）的编码过程。

**💡 创新点**

创新点在于将内容感知的专家选择与层级网格结构的 INR 结合，采用轻量级 ScoutNet 进行硬路由（hard‑routing）专家挑选，而不是传统的随机初始化或全局元学习，极大提升初始 PSNR 并减少优化迭代次数。

**🔧 技术方法**

使用的技术包括：离线构建专家内存库、基于少量帧的轻量级 CNN ScoutNet、硬路由专家选择、将选中专家的多分辨率网格与解码器参数迁移至目标 HiNeRV 并进行短期微调。

**📊 数据集**

实验数据集：UVG（ReadySetGo、Beauty、Jockey）、Big Buck Bunny 等公开视频，分辨率 1920×1080，视频长度 132–300 帧。

**📈 对比分析**

与标准 HiNeRV（300 epoch）、NeRV、MetaNeRV、RNeRV 进行对比。ScoutNeRV 在仅 37 epoch 下的 PSNR 与 300 epoch HiNeRV 差距 ≤0.8 dB，初始 PSNR 提升 21.25 dB，且实现 9.25× 的编码加速；在所有评估的率‑失真配置下，性能保持竞争力。

**⚠️ 局限性**

局限性：需要预训练专家库，难以覆盖极端多样或全新内容；仅使用硬路由，可能错过多专家融合的潜力；对极高分辨率（4K/8K）或极动态视频的适应性尚未验证；ScoutNet 对采样帧的敏感性可能影响对快速变化内容的选择效果。

---

## 505. The Recall Ceiling of LLM Recommendation Reranking

**arXiv ID:** 2609.27953 | [PDF](https://arxiv.org/pdf/2609.27953v1)

**作者:** Zhaohui Wang `[一作]` `[通讯]` (University of Southern California), Zhaohui Wang (University of Southern California)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了在真实检索条件下LLM重排器的性能瓶颈，并提出了回忆上限理论和Recall‑Aware Evaluation Protocol（RAEP）来规范实验。

**💡 创新点**

创新点在于给出了闭式候选重排的严格回忆上限（NDCG≤Recall），量化了重排器的利用率η，并发现提升回忆是突破性能的关键。

**🔧 技术方法**

采用了闭式候选重排框架、NDCG/Recall等评价指标、LLM（Gemma、Qwen、Llama、DeepSeek等）与传统学习到排序模型（LambdaMART、RankNet、MLP）以及多种检索技术（CF、BM25、Dense、Hybrid、生成检索）进行对比，并使用统计检验验证结果。

**📊 数据集**

实验覆盖八个公开数据集：Amazon（Beauty、Movies、Electronics、Sports、Toys、Office）、MovieLens‑25M、MIND News。

**📈 对比分析**

通过与CF基线以及多种重排方法在相同候选集（K=100）下比较，发现oracle评估下NDCG@10可达0.08–0.10，但在真实检索下仅为0.005–0.007，几乎与CF相同，回忆提升后性能才可显著提升。

**⚠️ 局限性**

局限性包括：低检索回忆导致上限严格、闭式候选限制无法产生新项目、数据稀疏导致样本量不足、评估样本规模有限、未覆盖生成检索等更开放的检索场景。

---

## 506. Reliable Fusion of Conflicting Experts

**arXiv ID:** 2609.27913 | [PDF](https://arxiv.org/pdf/2609.27913v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 507. The Emergence of Causal Curiosity from Prior Causal Belief Networks

**arXiv ID:** 2609.27946 | [PDF](https://arxiv.org/pdf/2609.27946v1)

**作者:** Zhuoyu Shi `[一作]` (University of Southern California), Fred Morstatter `[通讯]` (University of Southern California)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过构建2020–2022年Reddit子论坛的因果信念网络，并分析2023年以“why”为标题的提问，研究因果好奇心如何从已有知识结构中产生并呈现。

**💡 创新点**

发现因果好奇心高度依赖先前的知识结构、倾向于正面词汇、更多聚焦于因而非果、且集中在网络中心节点，展示了好奇心在社群认知结构中的系统化特征。

**🔧 技术方法**

使用基于RoBERTa的因果关系抽取模型（用于提取因果对）、Spacy（词性和头词提取）、LLaMA‑3.3‑70B（判定“why”标题是否真正表达好奇）以及网络分析（度中心性、介数中心性）等技术。

**📊 数据集**

大规模Reddit数据集（2005–2023），筛选最近四年（2020–2023）的高活跃子论坛，总计约81.9M帖子、1.4B评论及2023年1.7M“why”提问。

**📈 对比分析**

通过与人工标注对比评估LLaMA分类器（标题单独约91%精度，标题+正文约88%精度），在网络层面将好奇词汇与网络度/介数中心性进行分位箱分析，展示好奇词汇集中于最高10%中心节点（平均约37%/35%）。

**⚠️ 局限性**

局限包括：数据来源于单一平台可能存在偏差；因果关系抽取模型仍可能误检；“why”标题的好奇判定仍依赖模型，误差不可避免；缺乏对好奇心形成机制的因果推断与跨文化验证。

---

## 508. Shedding Light on Complex Bitcoin Mixer Transactions: 67-Fold Reduction in Unclassified Cases

**arXiv ID:** 2609.27933 | [PDF](https://arxiv.org/pdf/2609.27933v1)

**作者:** Nikolay Larionov `[一作]` (Moscow Institute of Physics and Technology), Yury Yanovich `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文针对比特币共享发送混合器（SSM）交易中的未分类（超时）案例，提出并实现了四种新型启发式算法，显著提升了复杂交易的解混合能力；

**💡 创新点**

创新点在于：①设计四种结构化启发式（预置分组、单值单元、相同值歧义、背包回退）并提供理论正确性证明；②构建优化的流水线顺序，先低成本启发式再高成本背包求解；③通过启发式解决了98.5%原先未分类的交易，实现67倍的未分类率下降；

**🔧 技术方法**

主要技术包括：基于NP‑完整性分析的子集求和与背包问题求解、启发式归约与合并、最小可连通对与最小分区理论、Python实现与自定义背包求解器；

**📊 数据集**

使用完整的比特币区块链数据（Genesis至第882,421区块），共计超过5,023,471条超时交易，并对50,000条随机抽样进行详细评估；

**📈 对比分析**

与原始1秒阈值解混合算法相比，启发式流水线将未分类率从1.4%降至0.021%，平均每笔交易额外耗时仅约11.4秒，性能提升超过25倍，且在未分类事务中实现了98.5%的成功率；

**⚠️ 局限性**

局限性在于：仍有1.5%超时事务无法被四种启发式处理；启发式主要针对典型结构，可能对极端或复杂交互的交易不适用；对非比特币UTXO链（如多资产EUTXO）的通用性需进一步验证；

---

## 509. Learning When Not to Listen: Selective Anti-Interference Pretraining for Language Models

**arXiv ID:** 2609.27925 | [PDF](https://arxiv.org/pdf/2609.27925v1)

**作者:** Jinchang Zhu `[一作]` (Hong Kong University of Science and Technology), Menglin Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种称为Selective Prefix Anti-Interference Regularization (SPAR) 的预训练目标，利用远前缀扰动与局部后缀的对照视图，在局部上下文已足够预测目标时通过门控KL一致性正则化，降低模型对无关远前缀的敏感性。

**💡 创新点**

核心创新是：①将远前缀干扰问题视为条件依赖问题，②设计短上下文足够性门（short-context sufficiency gate）来挑选局部可预测的标记；③在这些标记上施加有向KL正则化，形成仅在局部信息足够时才抑制远前缀干扰的选择性训练信号。

**🔧 技术方法**

技术包括：生成腐败前缀视图（仅打乱远前缀段），短上下文与完整上下文的概率比较做门控，带有stop‑gradient的KL一致性损失（gated invariance loss），以及对预训练与继续训练阶段的分布匹配实现。

**📊 数据集**

使用的数据集包括：长上下文基准 RULER、HELMET、NoLiMa 以及公开的零样本评测（PIQA、HellaSwag、ARC、OpenBookQA、Lambada、MMLU）进行从零训练与继续训练的评估。

**📈 对比分析**

与普通的Causal Language Modeling (CLM) 对比，SPAR 在 Qwen2.5‑0.5B、Qwen2.5‑3B、Llama‑3.2‑1B、Llama‑3.1‑8B、GPT2‑XL 等模型上均取得正向性能提升，尤其在 RULER 的长上下文任务中平均提升 2–4 分，HELMET 亦呈现小幅提升；在从零训练的公开基准上，SPAR 提升了 1–3% 的宏观得分。

**⚠️ 局限性**

局限性包括：①提升幅度相对有限，需调节门控阈值与正则化权重；②门控策略可能忽略部分重要但短上下文信息不足的标记；③实验主要集中在中文/英文文本，效果在其它语言或任务上的推广尚待验证；④对极长上下文（>32K）仍不够鲁棒，需进一步改进。

---

## 510. All modalities are equal, but video is more equal: Closing the Cross-Attention Gap in Joint Video Generation

**arXiv ID:** 2609.27901 | [PDF](https://arxiv.org/pdf/2609.27901v1)

**作者:** Ohad Rahamim `[一作]` (Bar-Ilan University), Gal Chechik `[通讯]` (Bar-Ilan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量级的 Reciprocal Cross‑Modal Attention Regularization（RecCAR）机制，用于增强多模态扩散模型中弱向视频的跨模态注意力，从而提升视频–动作和视频–音频生成的一致性和同步性。

**💡 创新点**

创新点在于：①发现预训练的双向跨模态注意力往往存在明显的不对称性；②利用已稳定的强向视频注意力作为内部监督，对弱向视频路径进行 KL 正则化，使两条通路趋于一致；③该方法仅需在 LoRA 权重上微调，且不需要额外的标注或架构改动。

**🔧 技术方法**

技术手段包括：跨模态 Transformer 结构、对注意力分布的对齐和 KL 正则化、LoRA 参数微调、使用 stop‑gradient 以保持强向注意力不变；在训练中加入 RecCAR 损失并调节 λ 参数。

**📊 数据集**

数据集：视频–动作实验使用 EchoMotion 训练集（4,292 对视频-三维动作对）并在 VBench 进行评估；视频–音频实验使用 VGGSound 子集（≈4,300 条音视频剪辑），并在 T2AV‑Compass 与 AVGen‑Bench 两个基准上评测。

**📈 对比分析**

与原始 EchoMotion、CoMoVi、FlowMo 以及 LTX‑2、JavisDiT++ 等基线相比，RecCAR 在 VBench 的 Human Anatomy 分数从 0.69 提升到 0.75；在 T2AV‑Compass 上 AV Desync 从 0.804 降至 0.752；在 AVGen‑Bench 上 DeSync 同样得到显著下降。整体生成质量（视频质量、音频质量、语义对齐）保持不变或略有提升。

**⚠️ 局限性**

局限性包括：①只验证了两种伴随模态（动作与音频），未覆盖更广泛的多模态场景；②仍依赖预训练模型的强向注意力质量，若该方向本身不足则 RecCAR 效果有限；③对超大模型的 LoRA 微调需要 GPU 资源，适用范围受限。

---

## 511. False-science induction in autonomous scientific discovery

**arXiv ID:** 2609.27883 | [PDF](https://arxiv.org/pdf/2609.27883v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 512. When Accuracy Gaps Fail to Certify: Auditing Cross-Domain Recalibration of LLM Judges

**arXiv ID:** 2609.27954 | [PDF](https://arxiv.org/pdf/2609.27954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 513. Sea-State-Induced Performance Transition in Maritime Networks: A Roughness-Aware Stochastic Geometry Framework

**arXiv ID:** 2609.27897 | [PDF](https://arxiv.org/pdf/2609.27897v1)

**作者:** Wen-Yu Dong `[一作]` (China Telecom Research Institute), Sheng Chen `[通讯]` (University of Southampton)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于海面粗糙度的随机几何框架，用于海岸附近海上通信网络的覆盖概率分析

**💡 创新点**

将Rayleigh粗糙度准则引入反射系数，构建粗糙度感知的两射/三射路径损耗模型，并揭示了海浪粗糙度对网络可靠性和速率的非单调影响

**🔧 技术方法**

随机几何、Rayleigh粗糙度模型、Nakagami‑m衰落、拉氏变换与Alzer不等式

**📊 数据集**

通过蒙特卡洛仿真和真实测量数据（162 MHz VHF频段）进行验证

**📈 对比分析**

与理想平滑海面模型对比，粗糙度模型将RMSE从7.73 dB降至6.41 dB（两射区），从11.40 dB降至8.94 dB（三射区），同时展示了中等粗糙度下覆盖概率提升的非单调特性

**⚠️ 局限性**

仅考虑单层反射，忽略海面动态变化及多频段/多链路特性，验证样本有限

---

## 514. Universal Decoding via the Pairwise Error Probability

**arXiv ID:** 2609.27887 | [PDF](https://arxiv.org/pdf/2609.27887v1)

**作者:** Nir Elkayam `[一作]` (Tel Aviv University), Meir Feder `[通讯]` (Tel Aviv University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一种基于 pairwise-error‑probability（PEP）的通用解码规则，能够在不预知信道或解码度量的情况下，对整个度量或信道族实现随机编码与确定性上的最优性能；

**💡 创新点**

核心创新在于：①将任意度量的 PEP 先正则化为相同概率尺度；②通过“裁剪逆 PEP”合并多重度量，得到一个统一的归一化最大似然（NML）边界；③在离散与连续符号系统上统一实现，并证明其为渐近极小化器/等化器；

**🔧 技术方法**

主要技术包括：PEP 的固定输出均匀性与谱指数分析、Kraft 型不等式、裁剪逆 PEP 的矩控制、子指数族的可分离性（type/网格化）、极大似然与最大互信息的统一框架，以及对错误指数与裁剪阈值的精细取值；

**📊 数据集**

本文未使用具体实验数据集，而是基于信息理论极限与随机编码分析，对离散记忆无关通道、有限状态通道、AWGN/ISI 连续通道等典型模型给出理论上最优指数；

**📈 对比分析**

与传统 MMI、Csiszár‑Körner、Feder–Lapidoth 等方法比较，统一解码器在随机编码误差指数上与最佳度量匹配（失去的速率为 O(log n/n)），并在极限下实现等化器性能；在 erasure/未检测错误场景下也能达到 Forney 风格的指数；

**⚠️ 局限性**

局限性包括：①对连续符号族需在典型集上裁剪并加上指数上限 δ；②在 ISI 匹配最大似然（含二次项）时未给出完整证明，需额外光滑性假设；③实际实现中需构造网格或离散化，可能导致计算量增长；④极小化器的 regret 仅在 O(log n/n) 范围内；

---

## 515. Understanding LLM Usage Among Early-Career Software Engineers in Practice

**arXiv ID:** 2609.27973 | [PDF](https://arxiv.org/pdf/2609.27973v1)

**作者:** Julia Alencar `[一作]` (CESAR School), Danilo Monteiro Ribeiro `[通讯]` (CESAR School)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 75 名早期职业软件工程师进行混合方法问卷调查，探究他们在专业实践中如何使用 LLM、所需的技术与非技术能力，并分析大学教育与工作实践之间的差距。

**💡 创新点**

首次系统研究早期职业软件工程师在行业中使用 LLM 的具体实践、必要技能以及教育不足，为软件工程教育与企业入职培训提供实证依据。

**🔧 技术方法**

使用混合方法调查（Qualtrics 量化问卷 + 访谈式开放式问题），并对开放式回答进行主题编码和描述性统计。

**📊 数据集**

Prolific 平台招募的 75 名主动使用 LLM 工具的早期职业软件工程师的问卷数据。

**📈 对比分析**

采用描述性统计（频率分布）和定性编码（主题分析）进行分析；未进行实验性对比或性能评估，结果以定性描述和统计比例呈现。

**⚠️ 局限性**

样本为便利抽样，主要自我报告，缺乏客观验证；跨行业、跨国家样本有限，研究仅覆盖早期职业阶段，缺乏长期跟踪，限制了结果的外推性。

---

## 516. A Resilience Recovery Method for Complex Traffic Network Security Based on Trend Forecasting

**arXiv ID:** 2609.27903 | [PDF](https://arxiv.org/pdf/2609.27903v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 517. Resilient Monitoring of Social Dynamical Systems through Collaborative Multi-Agent Networks under Latency

**arXiv ID:** 2609.27902 | [PDF](https://arxiv.org/pdf/2609.27902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 518. Trade-Size-Aware Dynamic Fees for Impermanent Loss Mitigation in AMMs

**arXiv ID:** 2609.27937 | [PDF](https://arxiv.org/pdf/2609.27937v1)

**作者:** Anton Ledrov `[一作]` (Moscow Institute of Physics and Technology), Yury Yanovich `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于交易规模的动态手续费机制，旨在缓解AMM中的暂时性损失并提升流动性提供者收益；

**💡 创新点**

创新点包括：① 将手续费视为与流动性池共同演化的二级不变量（AMM‑in‑AMM 架构），实现手续费的自适应内生化；② 设计 Impermanent‑Loss Trimming (ILT) 费率模型，按交易规模自动调节手续费，仅在交易超出安全区间时提高费用以抵消暂时性损失；③ 通过性能配置曲线 (Dolan–Moré) 统一评估不同手续费算法在多种市场情境下的表现；

**🔧 技术方法**

采用了多种技术手段：动态手续费算法（固定、区块适应、交易适应、Oracle‑based、AMM‑in‑AMM、ILT 包装）；模型构建基于恒定乘积（Uniswap V2）和二级不变量；使用 Merton Jump‑Diffusion（MJD）生成合成价格路径；对历史行情（12‑秒块时间）进行回测；

**📊 数据集**

数据集包括：① 通过 MJD 模型生成的合成行情，用于覆盖不同波动率与跳跃特征；② 真实历史行情（USDC/RAD、ETH/BTC、DOGE/BTC 等）按 Bull、Bear、Volatile、Calm 四种市场 regime 细分；共 36 个测试场景（9 对价×4 regime）；

**📈 对比分析**

比较方法采用 Dolan–Moré 性能配置曲线，衡量各算法在所有场景中获得最佳收益的比例。实验结果显示：ILT 改进方案在 60–75% 的情境下与最佳算法相近，LP 收益提升 6–24%（波动性市场）或高达 119%（平稳市场），平均 18.3% 的收益提升；uninformed 用户收益保持稳定，informed arbitrage 的收益下降 3–10%；

**⚠️ 局限性**

限制与待改进点：① 实际部署需考虑 gas 费用、oracle 延迟和路径依赖等实现摩擦；② ILT 只对单笔交易生效，未对短期拆分交易进行聚合，可能被套利者规避；③ 当前分析仅适用于恒定乘积 AMM，需推广至集中式流动性等新型 AMM；

---

## 519. ZD-AOMDV: A New Routing Algorithm for Mobile Ad-Hoc Networks

**arXiv ID:** 2609.27881 | [PDF](https://arxiv.org/pdf/2609.27881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 520. Anchor-Free Hidden-Target Seeking via Certified Self-Calibration under Correlated Odometry

**arXiv ID:** 2609.27905 | [PDF](https://arxiv.org/pdf/2609.27905v1)

**作者:** Yash Bagla `[一作]` `[通讯]` (Michigan State University), Yash Bagla (Michigan State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种在缺失全局定位与直接目标感知的极端环境下，利用单一未知姿态的中继点进行目标寻踪的方法，且该方法无需重建全局位姿，仅通过多视角的自标定与不确定性认证实现可靠控制。

**💡 创新点**

创新点在于：①揭示该配置下存在的 SE(2) gauge 并证明绝对定位不可行；②发现两次不同车辆姿态足以闭合该 gauge，恢复可控的任务商；③设计了 O(K) 的闭式多视角自标定估计器并给出第一阶协方差解析，精确保留由积分式里程计产生的跨视角相关；④基于该认证实现了“激励-寻踪-保持”的混合策略，能够在不可信估计时拒绝、检测中继帧变化并保持任务误差不随时间累积。

**🔧 技术方法**

技术包括：多视角几何自标定（利用两次位置差异求解 yaw 与相对位移）、第一阶协方差传播与 O(K) 后缀求和、基于 Fisher 信息的 yaw 置信区间构造、混合控制策略（激励、寻踪、保持）以及 ROS 2/Gazebo 物理仿真验证。

**📊 数据集**

使用了公开的仿真数据集（ROS 2/Gazebo 场景）以及实验室随机生成的目标/中继/车辆几何配置，全部以代码发布在 GitHub 并在 Zenodo 归档，实验中随机化 200–300 次，覆盖多种噪声、通信失效与中继扰动场景。

**📈 对比分析**

与已知 yaw 的 oracle、Sequential EKF、端点注册和固定延迟平滑器等基线进行对比。结果显示：在 200 场随机试验中，方法的站点均方根误差为 0.064 m，统计与 oracle 无显著差异；相对 EKF 的误差降低约 20 %；端点注册 1.5×误差；固定延迟平滑器在运行时间上慢约 50 倍但精度相当。鲁棒性实验表明，在通信丢包 50 %、延迟 0.5 s、抖动 0.2 s 或 20 % 大误差包的情况下，仍能保持 100 % 成功率，误差仅略有增加。

**⚠️ 局限性**

局限性包括：①仅适用于静态目标与单一未知姿态的中继；②对里程计假设为纯平移随机游走，偏置、尺度误差及航向漂移仅在实验中验证且未被正式纳入认证模型；③对中继姿态变化的检测依赖于多视角窗口，短视角窗口或几何条件不佳时会拒绝执行；④无法处理移动目标或多中继网络，且未给出多目标协同寻踪的理论与实现。

---

## 521. RelCheck: Dual-Evidence Spatial Grounding for VLM Hallucination Correction

**arXiv ID:** 2609.27890 | [PDF](https://arxiv.org/pdf/2609.27890v1)

**作者:** Siddhi Patil `[一作]` (San Jose State University), William B. Andreopoulos `[通讯]` (San Jose State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了RelCheck，一种训练无关的后处理纠错管线，用来纠正多模态大型语言模型的关系幻觉。

**💡 创新点**

创新点在于将木棉（Woodpecker）对象层知识库扩展为三层，加入来自RelTR的场景图关系三元组和基于边界框的确定性几何谓词，形成结构化的三层视觉知识库。

**🔧 技术方法**

技术上结合了GroundingDINO进行对象检测、RelTR生成场景图、几何层通过中心点计算得到左右/上/下等谓词，并用GPT‑5.4在可靠性层级与编辑距离门控下对文本进行纠正。

**📊 数据集**

实验使用COCO val2014中的POPE数据集（500问答）以及MME幻觉子集（240题，共四子任务），评估RelCheck在不同子任务上的表现。

**📈 对比分析**

与原始LLaVA v1 13B以及Woodpecker基线对比，RelCheck在MME总分从585提升至630，位置子任务提升31.7分，POPE各分裂准确率提升至最高0.878，整体性能显著提升。

**⚠️ 局限性**

局限在于依赖GroundingDINO的检测，未检测到的实体无法产生几何或属性证据；二维几何无法推断前后等三维关系；仅在LLaVA v1 13B上验证，API调用成本高。

---

## 522. DEAL-Grasp: Decoupled Alignment Representation for Geometry-Aware Dexterous Grasp Generation

**arXiv ID:** 2609.28131 | [PDF](https://arxiv.org/pdf/2609.28131v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 523. SlackDrive: Reclaiming Runtime Slack for Adaptive Driving Inference

**arXiv ID:** 2609.28064 | [PDF](https://arxiv.org/pdf/2609.28064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 524. MIDIBack: Harmony-Aware Singing Pitch Correction via Joint Vocal-Accompaniment Symbolic Modeling

**arXiv ID:** 2609.28008 | [PDF](https://arxiv.org/pdf/2609.28008v1)

**作者:** Joaquim Cavalcante `[一作]` (Federal University of Paraíba), Thais Gaudencio `[通讯]` (Federal University of Paraíba)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于 OctupleMIDI 的自动音高校正框架 MIDIBack，能够在伴奏上下文中识别并纠正人声音高错误。

**💡 创新点**

创新点在于将伴奏与人声的符号事件统一编码为 OctupleMIDI 序列，并利用 Adversarial‑MidiBERT 的双向 Transformer 对完整的音高上下文进行建模，实现上下文感知的音高预测。

**🔧 技术方法**

使用 OctupleMIDI 表示、GAME/MuScriptor 进行符号转录、GRU 生成的自回归偏移扰动、Adversarial‑MidiBERT 作为骨干网络，以及交叉熵损失进行训练。

**📊 数据集**

数据集包括 CSD、KiSing、Opencpop、M4Singer、MRSAudio、jaCappella、Korean MultiSinger、Korean MultiTimbre、SingStyle111 等无伴奏数据用于 detuner；以及 MIR‑1K、Sonovox、MoisesDB、DSD100、ccMixter 等含伴奏的歌曲用于 APC 训练。

**📈 对比分析**

与 DDPC 与 BERT‑APC 在六种音高腐败场景下对比，MIDIBack 在整体 RPA 最高达 78.6%，在全局偏移 + 学习偏移场景下达 81.5%，在清洁输入下降低腐蚀率，并在各单独场景中表现均衡。

**⚠️ 局限性**

主要局限在于评估仅基于转录得到的 MIDI 目标与合成扰动，缺乏专业标注和真实音频的感知评测；模型性能对伴奏质量高度依赖。

---

## 525. Your Model Is Leaking: Covert Information Transfer through LLM Residual Streams

**arXiv ID:** 2609.27996 | [PDF](https://arxiv.org/pdf/2609.27996v1)

**作者:** Mingyuan Li `[一作]` (University of Turku), Ren Ping Liu `[通讯]` (University of Technology Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ResidualMux，一种利用LLM残差流的隐藏通道，允许在受限或隔离的推理域中通过授权的激活日志将敏感信息隐式泄露给外部观察者。

**💡 创新点**

创新点在于发现并实现了不需要模型重训练、权重改动、网络出口或额外攻击组件的残差流隐藏通道，并通过信号‑残差比（SRR）自适应调节注入强度，使通道在多种模型和架构上同时实现高恢复率、低检测率和低输出失真。

**🔧 技术方法**

使用的技术包括：正交随机代码书写（高维残差空间中的正交代码），单层前向钩子残差注入，SRR自适应注入策略，线性解码器训练与离线恢复，以及对多种后置防御（高斯噪声、mHC混合、量化、Attention Residuals等）的评估。

**📊 数据集**

实验所用数据集包括：200条多类别提示（声明、指令、数学、代码），以及公开基准MMLU、GSM8K；模型涉及11个开源LLM，涵盖7个架构家族，参数规模从0.1B到27B，残差维度从768到5120。

**📈 对比分析**

与七种基线（Token Channel、Fine‑tuned Steganography、Kirchenbauer Watermark、KV‑Cache、输出级隐藏通道等）比较，ResidualMux在相同KL或检测阈值下可提供10–170倍更高的容量（3–4位/注入），检测AUC≈0.5且KL<0.01；在真实vLLM部署中，恢复率>90%，检测AUC≈0.5，且对模型输出几乎无显著退化。

**⚠️ 局限性**

局限性包括：必须有激活记录路径且激活可写入；后置防御（噪声、mHC、量化）在保持实用性的前提下难以完全消除通道；对超大模型的实测有限；攻击需要预先生成并共享代码书和解码器；若记录层被修改或隔离，攻击失效。

---

## 526. Evaluating Feedback Focus and Pedagogical Adaptivity in LLM-Generated Feedback on Student Writing

**arXiv ID:** 2609.28026 | [PDF](https://arxiv.org/pdf/2609.28026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 527. LAYERSCOPE: A Layerwise Characterization of Video and Multimodal Learned Representations

**arXiv ID:** 2609.28086 | [PDF](https://arxiv.org/pdf/2609.28086v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 528. "We'll Fix It Later": Education, AI, and the Deferral of Privacy in EdTech

**arXiv ID:** 2609.28137 | [PDF](https://arxiv.org/pdf/2609.28137v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 529. TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval

**arXiv ID:** 2609.28048 | [PDF](https://arxiv.org/pdf/2609.28048v1)

**作者:** Mourad Hassani `[一作]` (SAMOVAR, Télécom SudParis, Institut Polytechnique de Paris), Christian Jacquelinet `[通讯]` (Aldebaran Care)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种模块化的时序分支 TEMPS，能够在冻结的语义检索器上训练时序相似度，并通过弱监督的时间表达对齐来提升检索的时序相关性。

**💡 创新点**

创新点包括：①定义 Temporal Textual Similarity (TTS) 任务，将时序相似度独立出来；②利用合成 TimeML 语料生成大规模弱监督训练数据；③采用高斯嵌入与 KL 散度实现时序的不对称兼容；④将时序分支与语义分支无缝融合，保持检索器冻结。

**🔧 技术方法**

使用技术包括：高斯分布建模与 KL 散度、moment‑matching、anchor‑date‑conditioned 编码器、Gaussian‑KL 兼容度评分、弱监督时间表达生成、规则基 SUTime/HeidelTime 进行时间归一化、RAG 生成流水线。

**📊 数据集**

训练数据来自 1000‑2030 年的合成 TimeML 表达式和英文 Multilingual MLM Temporal Tagging 资源；评测使用 TimeQA、TempReason、TS‑Retriever 三大时序检索基准，并在 TimeQA RAG 任务中结合 Qwen2.5‑7B‑Instruct 进行端到端验证。

**📈 对比分析**

与 BM25、SUTime、E5、BGE、Mistral、TS‑Contriever 等基线对比，TEMPS 在所有语义骨干上均提升 MRR；在 TS‑Retriever 上 R@1 从 19.92% 提升至 25.39%，整体 NDCG@5、Recall@5、Precision@5 均显著上升；在 TimeQA RAG 中，EM、F1 与包含率也同步提升。

**⚠️ 局限性**

局限性包括：仅处理可解析的明确时间表达，无法覆盖隐式或歧义事件关系；评测仅在英文语料上进行，跨语言适用性未知；合成监督可能引入标注噪声；规则基 SUTime 控制在某些基准上更优；加入时序分支可能在 Recall@5 下降；依赖语义骨干的偏差与知识缺口；仅对 512 词段进行编码，长段或多时段信息可能被压缩。

---

## 530. Learning a Speed-adaptive Hip Exoskeleton Control Policy Via Sim-to-real Reinforcement Learning

**arXiv ID:** 2609.28027 | [PDF](https://arxiv.org/pdf/2609.28027v1)

**作者:** Bin Li `[一作]` (Lingnan University), Chenglong Fu `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种将仿真到真实的强化学习与基于用户偏好的在线优化相结合的自适应髋关节外骨骼控制框架，能够在连续变速步态下实现个性化辅助时机与力度。

**💡 创新点**

创新点在于①通过上下文条件的RL实现人类运动与外骨骼辅助的端到端学习，并在仿真中完成时机学习；②利用TCN对教师策略进行蒸馏，仅用IMU观测实现在线部署；③将时机学习与力度在线偏好学习分离，显著降低在线优化维度。

**🔧 技术方法**

技术包括：MuJoCo仿真平台、基于上下文的PPO强化学习、人类肌肉骨骼模型、TCN策略蒸馏、基于高斯过程的偏好学习与主动查询。

**📊 数据集**

数据集为：人类肌肉骨骼模型产生的仿真步态数据（0.5–1.7 m/s速度区间），以及六名健康受试者在跑步机与自由步行中收集的IMU、EMG与外骨骼助力记录。

**📈 对比分析**

与五种基线（Spline、DOFC、BTDC、MLP‑PPO、LSTM‑PPO）比较。实验显示，该方法在跑步机上正机械功高于DOFC，在线个性化后进一步提升；在自由步行中正机械功提升约30%，负功下降55%，且RMS扭矩下降仅9%。

**⚠️ 局限性**

局限性：仅验证了水平地面步态，且仅针对髋关节外骨骼；未评估斜坡、上坡、楼梯等其他运动模式；在线优化仍需多次试验，硬件对EMG采样和实时控制的依赖较高。

---

## 531. Probabilistic and Geometry Aware Neural Surrogate of Scrape Off Layer Plasma Simulations

**arXiv ID:** 2609.28116 | [PDF](https://arxiv.org/pdf/2609.28116v1)

**作者:** Gabriele Gianuzzo `[一作]` (Eindhoven University of Technology), Vlado Menkovski `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于条件流匹配的概率性快照模型，用几张保持邻接关系的图像张量代替传统的扁平向量来逼近托卡马克边缘等离子体稳态。

**💡 创新点**

创新点在于：①将曲线网格无损地展开为三张可卷积的图像张量，保留几何邻接；②使用条件流匹配（flow‑matching）捕捉分支现象，生成多模态预测分布而非单一点估计。

**🔧 技术方法**

主要技术包括：卷积网络（U‑Net + ControlNet）、自条件输入、FiLM 频域嵌入、条件流匹配框架、Euler 解析 ODE 采样、对流形邻接保留的网格映射。

**📊 数据集**

使用 SOLPS‑ITER 公开数据库：7221 个已收敛稳态解（5756 训练，1465 测试），每个样本包含 8 个全局标量和 22 个场通道（温度、密度、速度等）在 104×50 网格上。

**📈 对比分析**

与传统确定性 MLP 基准相比，模型在四个离子化区间的 log‑MAE 维持在 0.1‑0.2 之间，预测均值几乎无系统偏差；采样速度仅需约 79 s（50 次样本），相比 SOLPS‑ITER 计算耗时的数小时至数周大幅提升。

**⚠️ 局限性**

局限性：①未给出与确定性基准在点估计精度上的直接对比；②在冷核心区间模型超出流体-中性假设，结果仅作指示；③目前的几何映射仅适用于固定拓扑网格，拓展至其它几何形状需手工重构条带与邻接。

---

## 532. Watching What We Eat: Information Quality and Body Image in Diet-Related YouTube Videos

**arXiv ID:** 2609.28114 | [PDF](https://arxiv.org/pdf/2609.28114v1)

**作者:** Maddalena Ghiotti `[一作]` (Politecnico di Torino), Yelena Mejova `[通讯]` (ISI Foundation)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

针对YouTube平台的饮食相关视频进行大规模采样（每日发布与热门视频），并采用LLM自动标注结合结构化评估框架，系统评估其信息质量与身体关注度。

**💡 创新点**

创新点在于：① 通过日常与热门两种采样方法覆盖“长尾”与“热门”内容；② 将已验证的PRHISM、HONcode与SMEC评估维度迁移至LLM提示，实现在数千条视频上的大规模、细粒度质量与身体焦点评估；③ 同时识别视频中的AI使用程度，并探究其对质量的影响；④ 对内容主题进行NMF聚类，揭示不同主题与质量、身体焦点及用户参与度的关联。

**🔧 技术方法**

主要技术包括：YouTube Data API与Transcript API进行视频与字幕抓取；大规模文本处理与特征提取（长度、符号比例、标题/描述词频等）；非负矩阵分解（NMF）用于主题发现；OpenAI GPT‑4.1 零样本提示（zero‑shot）进行自动评估；Spearman相关系数与Bonferroni校正进行统计分析。

**📊 数据集**

数据集为3129条独立YouTube视频（Jan‑Jun 2025每日采样2870条 + Nov‑Dec 2025热门采样384条），包含视频元数据、字幕、评论、频道信息；此外手工标注了50条视频（25条每日样本+25条热门样本）用于验证。

**📈 对比分析**

与人工标注对比：LLM在信息质量上与身体焦点的Spearman相关系数分别为0.669与0.752，说明自动评估与人工评估高度一致；随后使用该自动评估结果探讨与观看量、互动率等指标的关系，发现身体焦点相关维度与观看量正相关，而质量维度对观看量影响有限。

**⚠️ 局限性**

局限性包括：① 样本主要来自美国，虽含部分其他国家但无法代表全球情况；② LLM评估仍低于人工标注的准确度；③ 仅使用音频字幕，缺少视觉信息；④ 评估仅覆盖英文视频，语言多样性有限；⑤ AI使用检测基于人工观察，缺乏自动化工具；⑥ 相关性分析多为单变量，未深入多重回归，因多重共线性导致解释力有限。

---

## 533. Dual-Hypergraph Indexing: Bridging Knowledge Islands for Multi-Hop Reasoning in Retrieval-Augmented Generation

**arXiv ID:** 2609.28108 | [PDF](https://arxiv.org/pdf/2609.28108v1)

**作者:** Qi Sun `[一作]`, Yu Guo `[通讯]` (Xi'an Jiaotong University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出 Dual-Hypergraph Indexing (DHI) 框架，将基础事实层与深度洞察层双层超图结合，用双路径聚合提升多跳 RAG 的逻辑连贯性。

**💡 创新点**

创新点在于通过重要性驱动的 hub 聚合和时间段链式聚合两个正交路径，将孤立的高阶超边转化为可跨文档、跨时间的深度洞察，解决知识岛问题。

**🔧 技术方法**

利用 LLM 进行实体与超边抽取，构建双层超图；采用 5‑维结构特征、P90 归一化及自适应阈值的 hub 识别；使用滑动窗口贪心搜索实现时间链聚合；再通过密集向量检索和双向映射实现跨层检索与上下文拼接。

**📊 数据集**

在 Mix、CS、Agri、Neuro、Pathology 五个公开多域基准上进行评测。

**📈 对比分析**

与 LLM、NaiveRAG、GraphRAG、LightRAG、HiRAG、Hyper-RAG 等基线比较，DHI 在综合指标上最高，尤其在 Pathology 上达 85.78 分，逻辑性提升 1.53 分。

**⚠️ 局限性**

主要限制是对初始 LLM 抽取的实体与超边精度高度依赖，错误会被放大进入深度洞察层，缺乏动态置信度校正机制。

---

## 534. MotionSpec: Spectral Trajectory Supervision for Motion-Consistent Video Generation

**arXiv ID:** 2609.28095 | [PDF](https://arxiv.org/pdf/2609.28095v1)

**作者:** Ziqi Ni `[一作]`, Wei Zhou `[通讯]` (Cardiff University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套名为 MotionSpec 的运动监督框架，包含 Spectral Trajectory Consistency (STC) 和 Local Flow Consistency (LFC)，用于提升文本到视频生成模型的运动一致性与时序连贯性。

**💡 创新点**

创新点在于：①引入 STC，通过对锚点相对运动轨迹做时间域 Fourier 变换并对幅度与相位进行监督，显式约束多帧运动的频谱分布和时间组织；②结合 LFC 对相邻帧光流进行一致性约束，平滑局部运动过渡；③在两种监督层次（短期光流与长期频谱）协同作用下，实现对长范围运动演化的精准控制。

**🔧 技术方法**

使用的技术包括：基于光流估计器提取运动信息；对轨迹做 1D FFT 生成运动频谱体；幅度与相位监督（含相位置信任机制）；光流一致性损失（Charbonnier 惩罚）；以及与标准流匹配损失共同优化的训练策略。

**📊 数据集**

训练数据集为从 OpenVid-1M 过滤出的约 10,000 条视频，按运动幅度与文本提示筛选，涵盖人类、物体及复杂运动场景。

**📈 对比分析**

与仅使用流匹配 (FM-only) 基线相比，加入 LFC 和 STC 可分别提升 VMBench 及 VideoJAM-Bench 的多项指标；最优组合（LFC+STC）在整体平均分上比 FM-only 提升约 1.5 分，显著提升运动质量、时序连贯性与结构稳定性，同时保持视觉清晰度。

**⚠️ 局限性**

局限性包括：① 仅在单 GPU (NVIDIA H200) 上验证，未探讨大规模并行训练的可扩展性；② 依赖光流估计器的精度，噪声较大时可能导致监督失效；③ 对极端快速运动或极端光照变化的鲁棒性尚未充分评估。

---

## 535. Prompt, Probe, Train, or Annotate? Single-camera sports video understanding in amateur settings

**arXiv ID:** 2609.28049 | [PDF](https://arxiv.org/pdf/2609.28049v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 536. Global tree forecasters collapse at the hierarchical aggregate: a five-panel failure characterization

**arXiv ID:** 2609.27912 | [PDF](https://arxiv.org/pdf/2609.27912v1)

**作者:** Md Rezwanul Islam `[一作]` (Field Nation LLC), Wael Mohammed `[通讯]` (Field Nation LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在全局梯度提升树模型中，若仅训练单个系列，却直接预测汇总级别（超出训练范围）时会出现“崩溃”现象，导致汇总预测误差增大30–50倍甚至高达496倍；

**💡 创新点**

创新点在于首次系统性地表征此崩溃机制、量化其规模阈值，并验证三种已有的“缩放/差分”修复手段在不同库与不同数据集上的可迁移性；

**🔧 技术方法**

使用的技术包括全局梯度提升树（LightGBM、XGBoost、CatBoost）、对每个系列进行均值缩放、加入加权聚合训练行以及季节差分预测；

**📊 数据集**

所用数据集包括：生产环境下的业务到业务市场的月度交易额（B2B）、合成层次化面板、M5竞赛数据、澳大利亚旅游访客夜数、UCI Online Retail II批发商面板；

**📈 对比分析**

通过对比原始模型与修复模型在MAPE、MASE、RMSSE上的表现，发现原始模型的汇总MAPE超过90%，修复后降至≈5–15%，表明修复能将误差逼近季节性随机模型水平；

**⚠️ 局限性**

局限性包括仅针对单一树模型（非线性叶子或其他类型模型未评估）、只在稀疏、短期、上升趋势的面板中验证、以及对递归多步预测的鲁棒性仍有限（例如加权聚合行在递归下易崩溃）。

---

## 537. PISCES: Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather Anomaly Detection and Early Warning

**arXiv ID:** 2609.28022 | [PDF](https://arxiv.org/pdf/2609.28022v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 538. Can LLMs Catch a Rigged Backtest? A Clean-Control Calibration Benchmark

**arXiv ID:** 2609.28090 | [PDF](https://arxiv.org/pdf/2609.28090v1)

**作者:** Makar Ulesov `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Arsenii Bobovnikov `[通讯]` (University of Pittsburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个96项的配对基准，用于评估LLM在对冲基金回测审计中的缺陷识别与误报校准；

**💡 创新点**

首次引入配对清洁对照设计，使缺陷识别与误报率可分离评估，强调校准的重要性；

**🔧 技术方法**

利用多模型（DeepSeek、Gemini、GPT‑4系列）与多种提示模式（open、closed、clean‑aware）以及定量验证器对审计结果进行评分；

**📊 数据集**

使用人工生成的96个回测样本，包含48对清洁/缺陷配对，每个缺陷类别有6例，覆盖8类常见方法论缺陷；

**📈 对比分析**

对比模型在不同提示模式和表面（代码/文本）下的缺陷召回率、误报率、证据定位和修复相关性，发现虽然召回率可达100%，但误报率差异显著，Clean‑aware提示可将误报率降至0%；

**⚠️ 局限性**

限制包括样本量小（每类6例）、缺陷分布离散导致置信区间宽、仅使用单轮温度0调用、未与真实回测笔记本或人工基准对比，且评估器对修复建议的语义匹配有限。

---

## 539. Reference-Based Analysis of Coherence and Diversity in Open-Ended Text Generation

**arXiv ID:** 2609.28080 | [PDF](https://arxiv.org/pdf/2609.28080v1)

**作者:** Esteban Garcés Arias `[一作]` `[通讯]` (LMU Munich), Esteban Garcés Arias (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对开放式文本生成评估，提出一种基于人类参考的三视角框架：时间对齐（DTW）、摘要比较（平均/方差对比）以及联合分布似然，系统研究连贯性与多样性轨迹与人类文本的关系。

**💡 创新点**

创新点在于将连贯性/多样性轨迹与人类参考进行多角度比较，发现多样性对齐和平均比较能捕获人类质量差异，并首次将联合分布似然用于评估生成文本的典型性。

**🔧 技术方法**

使用的技术包括动态时间规整(DTW)、平均与方差标准化、二维高斯似然评估、PCA降维、固定Q*Text以及MAUVE分布对比。

**📊 数据集**

数据集为Qwen2.5-7B在50个提示（25 Wikinews+25 WikiText）上生成的5个模型输出和1个人工续写（共250机器文本），人类参考为10,000条来自BookCorpus、ROCStories、Wikinews、WikiText、WritingPrompts的文本。

**📈 对比分析**

与固定Q*Text、PCA、端点多样性等基线比较，使用Spearman相关并通过bootstrap置信区间评估。结果显示多样性对齐和平均比较与人类评分正相关，但其优势不显著；分布似然也呈正相关，表明典型性信息与质量有关。

**⚠️ 局限性**

限制包括样本量有限、仅针对单一英文生成模型、评审人一致性有限、Gaussian模型对特征边界无约束、未对多重检验进行校正、仅评估连贯性与多样性而未覆盖事实性、创造性或安全性等维度。

---

## 540. LLM-Assisted Workflow for Structural Difference Visualization in Evolving Software Requirements

**arXiv ID:** 2609.28002 | [PDF](https://arxiv.org/pdf/2609.28002v1)

**作者:** Koi McFarland `[一作]` (Charleston Southern University), Songhui Yue `[通讯]` (Charleston Southern University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一种基于大型语言模型的工作流程，用于将软件需求版本转换为三元组图谱，并在OntologyWeb工具中实现可视化对比，以展示结构差异。

**💡 创新点**

将LLM生成的需求三元组与人工审核相结合，形成可供快照比较的结构化图谱，并提供同步侧边栏可视化对齐与颜色编码的差异展示。

**🔧 技术方法**

使用大型语言模型（LLM）进行文本到三元组的自动转换、Web交互可视化框架、JSON存储、图形匹配与布局对齐算法。

**📊 数据集**

在真实的需求文档（至少两版）上进行实验，未公开具体数据集，但使用了多份业务需求的不同版本。

**📈 对比分析**

通过保存各版本的快照后在侧边栏对齐两张图谱，利用颜色编码突出新增、删除和修改关系；初步定性评估显示在中高复杂度变更中能更清晰地展示结构差异，文本对比不足。

**⚠️ 局限性**

工作流程主要适用于短段落或选定片段，难以一次性比较全文多段落；且目前缺乏量化评估与自动差异算法，仅为原型阶段。

---

## 541. Discovery of fully efficient fault indicators along a data-based diagnosis process

**arXiv ID:** 2609.28087 | [PDF](https://arxiv.org/pdf/2609.28087v1)

**作者:** Igor Bezmaternykh `[一作]` (LAAS-CNRS), Elodie Chanthery `[通讯]` (LAAS-CNRS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出DT4X+，一种基于符号回归的决策树改进版，用于故障诊断。

**💡 创新点**

创新点在于：①在训练集构造时保留所有非目标类样本；②引入碎片化损失（fragmentation loss）惩罚非目标类在子节点中的分散，从而保持类完整性，提升解释性。

**🔧 技术方法**

使用符号回归（gplearn实现）生成多变量表达式作为决策树分裂函数，并结合逻辑损失和碎片化损失进行训练。

**📊 数据集**

实验使用四个数据集：静态多加器多乘器（polybox）、静态减法器（subtractor）、动态双槽系统（two-tank）以及蒸汽发电过程的超热器子系统（superheater 4.2）。

**📈 对比分析**

与原DT4X基线对比，评估指标为准确率（accuracy）和F1分数，时间指标为推理延迟和训练耗时。结果显示：在polybox上两者相同；在subtractor上DT4X+略有下降；在two-tank和superheater上均显著提升（准确率/ F1 分别提升约0.02和0.05），推理时间基本持平，但训练时间增加1.5–5倍。

**⚠️ 局限性**

局限性在于：训练时间显著增加，尤其是类数或样本量大时；碎片化损失仅在目标类分离率高于97%时才生效；仍需手动调节超参数，且对极少量样本的非目标类影响有限。

---

## 542. ZoomDiff: A High-Fidelity Diffusion Model for Dual-Camera Smooth Zooming

**arXiv ID:** 2609.28083 | [PDF](https://arxiv.org/pdf/2609.28083v1)

**作者:** Jiayi Zhang `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为ZoomDiff的高保真扩散模型，用于实现双摄像头间的平滑数码变焦。

**💡 创新点**

创新点包括：① 在潜在空间和像素空间双重条件下利用双摄像头图像；② 通过跨注意力引入CLIP语义特征并在每一步显式替换端点潜在；③ 将光流对齐的多尺度VAE编码器特征注入解码器以恢复高频细节；④ 采用光流引导的时间一致性损失；⑤ 通过少步适配提升推理速度。

**🔧 技术方法**

核心技术为Stable Video Diffusion（SVD）框架、VAE编码器/解码器、SEA-RAFT光流估计、跨注意力机制、ref injector模块以及多损失优化（潜在重建、像素重建、LPIPS、时间一致性）。

**📊 数据集**

使用2100帧合成数据集（由3DGS生成的双摄像头序列，训练2000，测试100）以及100条真实世界的Redmi K50 Ultra场景。

**📈 对比分析**

与12种现有光流基和扩散基插帧方法（包括RIFE、UDPR等）对比，ZoomDiff在合成集上PSNR 23.80、SSIM 0.761、LPIPS 0.253、PSNR‑div 23.01、FVD 131.332等指标均取得最高值，在真实场景的无参考评估（MUSIQ、LIQE、DBCNN、DOVER）也遥遥领先。

**⚠️ 局限性**

局限性在于仍需高计算资源，少步适配后质量略有下降；在极端动态或曝光极差的场景下可能出现细节失真或颜色漂移；目前仅针对双摄像头设置，未泛化到多摄像头或任意视角变换。

---

## 543. TEEP-RCNN: Texture-Enhanced Edge-aware Perception for Steel Surface Defect Detection via Improved Convolutional Block Attention in Faster R-CNN

**arXiv ID:** 2609.28077 | [PDF](https://arxiv.org/pdf/2609.28077v1)

**作者:** Kirtan Rajesh `[一作]` `[通讯]` (Independent Researcher), Kirtan Rajesh (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于Faster R‑CNN的钢表面缺陷检测框架TEEP‑RCNN，并在训练中加入改进的CBAM模块、差异学习率和TTA+WBF推理。

**💡 创新点**

改进CBAM在通道注意力中加入Dropout、在空间注意力中加入BatchNorm，实现更稳定的特征重校准，显著提升细纹和拉伸缺陷的检测。

**🔧 技术方法**

使用ResNet‑101+FPN骨干、改进CBAM、RPN、RoI Align、Soft‑NMS、Weighted Box Fusion、测试时增强、AdamW+余弦退火+warm‑up、混合精度训练。

**📊 数据集**

在NEU‑DET钢表面缺陷基准上进行实验，该数据集包含6类缺陷共3410张图。

**📈 对比分析**

与YOLOv8l、YOLOv11m等单阶段模型以及Faster R‑CNN基线对比，TEEP‑RCNN在仅10个训练epoch下获得73.3% mAP@50，远低于YOLOv11m的100epoch训练但在卷积层训练更快；在rolled‑in‑scale类上表现优于YOLOv11m。

**⚠️ 局限性**

模型参数量较大（61.8M）且对Crazing类检测效果差；缺少CBAM改进与基线的 ablation、未验证在其他工业数据集上的泛化、推理速度未测量。

---

## 544. Exact Quantile Balancing and Load-Error Injection for Mixture-of-Experts

**arXiv ID:** 2609.28053 | [PDF](https://arxiv.org/pdf/2609.28053v1)

**作者:** Pit Neitemeier `[一作]`, Sohir Maskey `[通讯]` (Aleph Alpha)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对稀疏Mixture-of-Experts（MoE）模型提出了 Exact Quantile Balancing (EQB) 与 Load-Error Injection (LEI) 两种新型负载平衡方法，并在 7.5B 参数解码器模型上验证其有效性。

**💡 创新点**

创新点：
1) EQB 通过两次 BF16 位域计数实现全局批次精确分位数计算，通信量仅为 2E×256 次计数，独立于 token 数；
2) LEI 直接将本地负载误差注入路由器分数梯度，避免了 GShard 通过归一化概率传递误差所带来的耦合与梯度不稳定。

**🔧 技术方法**

使用技术：
- 分布式 BF16 位域 (radix) 选择
- 全局 All-Reduce 计数
- 负载误差注入 (gradient injection) 与梯度上限（tanh 归一）
- 训练时对比 GShard 辅助损失与其归一化版本

**📊 数据集**

使用数据集：
- 训练集：约 500B  tokens 的无监督文本；
- 评估集：MMLU、ARC、HSwag、GSM8K、TriviaQA、HumanEval、MBPP 等标准语言模型基准。

**📈 对比分析**

比较方法与性能：
- 与传统基于平均分位数的 QB、GShard 辅助损失以及其归一化版本对比；
- 在 100B 与 500B 训练步骤中，EQB 将 Global MaxVio 从 0.92 降至 0.74，Local MaxVio 从 5.96 降至 5.38；
- LEI 进一步将 Global/Local MaxVio 下降至 0.60/3.52，BPB（Benchmarks Per Billion tokens）在所有评测上均有提升或保持不变，且模型精度与 GShard 相当或略优。

**⚠️ 局限性**

Limitations：
- 仅在单一模型规模与随机种子下验证，未给出方差或多次实验结果；
- LEI 与 GShard 需要单独调参，系数对梯度幅度影响大；
- Local MaxVio 仅为负载不平衡的近似指标，未直接测量专家并行 (EP) 通过率。

---

## 545. Tensor Decomposition of Transformer Key-Value Caches: Spectral Structure and Format Comparison

**arXiv ID:** 2609.28029 | [PDF](https://arxiv.org/pdf/2609.28029v1)

**作者:** Rahul Krishnan `[一作]` (Universitaet Trier), Volker Schulz `[通讯]` (Universitaet Trier)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究自回归Transformer的KV缓存张量的多线性结构，量化每个模式（head、token、feature、layer）的奇异值谱，并对四种标准张量分解（Tucker、CP、Tensor Train、t‑SVD）在匹配存储预算下的重构误差进行系统比较。

**💡 创新点**

提出了模式计数二分理论（head/层模式为索引模式、token/feature模式为可压缩模式），并给出了基于谱的模式全秩保留证明；揭示了键/值在谱上的不对称性以及RoPE旋转导致的可压缩性下降。

**🔧 技术方法**

采用张量奇异值谱分析、HOSVD/ST‑HOSVD求解、CP/TT/t‑SVD算法、以及Lagrangian秩分配器；对比匹配存储预算下的相对 Frobenius 误差；使用理论下界（Eckart‑Young）验证误差上限。

**📊 数据集**

使用两大公开模型的真实KV缓存：Mistral‑7B‑v0.3（GQA）和 LLaMA‑2‑13B（MHA），每层 32/40，批大小 1，序列长度 T=1024，按 3 个独立 prompt 采样。

**📈 对比分析**

在所有模型、层和 prompt 之间对比四种分解，Tucker 在 2×–5× 压缩比下始终取得最低重构误差；键的误差显著低于值（≈1/2.4），而 Post‑RoPE 键误比 Pre‑RoPE 低 41%–64%。此外，二维解包（per‑head SVD）在键上略优，四维 Tucker 在值上优于 2D 方法。

**⚠️ 局限性**

局限性包括仅测试两种模型且固定序列长度（T=1024），未验证增量生成过程中 token 模式谱的稳定性；CP 在 LLaMA 上因内存限制无法完整拟合；RoPE 影响仅在键缓存上评估，值缓存的旋转效果未深入探究。

---

## 546. Cubical Sheaf Complexes with Constant Expansion with Applications to Asymptotically Good qLTCs

**arXiv ID:** 2609.28028 | [PDF](https://arxiv.org/pdf/2609.28028v1)

**作者:** Yeyuan Chen `[一作]` (University of Michigan), Er-Cheng Tang `[通讯]` (University of Washington)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

构造了一族显式量子局部可检代码（qLTC），其码率为常数、距离线性、容错性常数，且检查矩阵行列权均为常数。

**💡 创新点**

创新点在于：① 证明了在“norm‑one”取值集合上评估的Reed‑Solomon码具有统一的乘积扩展性；② 将这种局部码嵌入到来自Bruhat‑Tits树乘积的算术四面体（cubical）复形的Tanner sheaf中；③ 通过已建立的局部到全局（local‑to‑global）框架和sheaf对偶性，得到全局线性距离和常数声学性，从而首次实现无对数损失的全局好性质。

**🔧 技术方法**

主要技术包括：算术四面体复形与Tanner sheaf编码、局部到全局的相互作用、乘积扩展性与多项式多面体包裹（polynomial hull）相结合的内部生成（inner‑generation）方法、Frobenius 变换保证局部码在算术作用下兼容，以及对Bruhat‑Tits树的量子分形几何分析。

**📊 数据集**

本文完全基于理论构造，未使用任何实验数据集。

**📈 对比分析**

与已有工作相比：以前的qLTC仅能获得多对数损失的距离和声学性；本工作在相同的常数权重下消除了所有对数损失，获得了线性距离、常数码率和常数声学性，并在任何固定维度 r≥4、任意合法的 k 范围内实现；性能在理论上优于之前的构造。

**⚠️ 局限性**

局限性包括：构造依赖于大参数 A（决定树度和评估点集合），实现较为复杂；仅适用于固定维度 r≥4 的情况；虽然声学性为常数，但尚无高效纠错算法；对实际量子硬件的实现与编码/解码复杂度仍需进一步研究。

---

## 547. Shared Global KV with Layer-Specific Local History

**arXiv ID:** 2609.28006 | [PDF](https://arxiv.org/pdf/2609.28006v1)

**作者:** Xinglang Xian `[一作]` `[通讯]`, Xinglang Xian

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在共享全局 KV 的解码器 Transformer 中引入了局部历史缓存（bounded local KV），并探讨了历史内容与输入来源的分离、融合策略与资源开销。

**💡 创新点**

创新点在于：① 将局部历史与全局 KV 明确分离，保留短期历史特征；② 通过联合与学习的分离融合（joint vs. separate fusion）评估历史对注意力分配的影响；③ 设计了可在同一权重配置下完成“exact suffix”预填和请求的构造方案，减少依赖与计算；④ 在同等训练预算下与 GQA、CLA 等共享 KV 方法进行公平对比。

**🔧 技术方法**

使用技术包括：Decoder‑only Transformer、Causal Grouped‑Query Attention、RMSNorm、SwiGLU、Rotary Position Encoding、共享全局 KV + 局部 KV、softmax 级联融合、bounded history 缓存、exact suffix 构造、FP32 预填与请求测评。

**📊 数据集**

数据集：FineWeb‑Edu（训练文本）、PG‑19 书籍（外部验证）以及 49 本固定书籍做 held‑out 测试，零样本评测使用 PIQA、HellaSwag、ARC‑Easy。

**📈 对比分析**

方法：在 126 M 参数、2 K 上下文、8 个种子下进行 factorial 与预算匹配实验，比较历史缓存 vs. 当前仅缓存 vs. 复制当前条目，并与 GQA、GQA4‑CLA2、PLA 等共享 KV 架构对比。性能：历史缓存平均降低 1.4 % 训练集 perplexity，约 1.1 % 开放集 perplexity；在 8 K 上经过适配仍保持 1.1‑1.2 % 的优势；在请求与预填时间上，历史缓存相较于全局共享方法可在特定批次下节省约 30 % 预填时间，但缓存占用略增。

**⚠️ 局限性**

局限性：在外部书籍验证上种子间波动大，未能确认历史优势；下游零样本任务表现不一；仅在 126 M 规模、2 K/8 K 上下文下验证，未测试更大模型或不同硬件；完整实验对 BF16/连续批处理的影响尚未评估。

---

## 548. Learning from Failures: Heterogeneous Graph Memory for Small Language Model Tool-Using Agents

**arXiv ID:** 2609.28003 | [PDF](https://arxiv.org/pdf/2609.28003v1)

**作者:** Jiaxing Li `[一作]` (Southeast University), Youyong Kong `[通讯]` (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FRESH 框架，将小型/中型 LLM 的工具使用经验（包括成功与失败）结构化为异构图，并在推理时通过检索与预判门控实现更安全可靠的执行。

**💡 创新点**

创新点在于：① 将失败原因、修复策略与执行前置条件显式编码为图节点与边；② 训练一个基于异构图的检索器对检索到的证据进行关系感知排序；③ 通过轻量级的策略与前置条件门控在执行前验证风险，避免模型直接复制失败行为。

**🔧 技术方法**

技术包括：异构图表示与构建、基于图神经网络的检索器训练、检索结果压缩为四类操作指引（Do、Avoid、Check、Repair）、策略与前置条件门控机制。

**📊 数据集**

使用公开基准 τ‑bench（Airline 与 Retail）和 AppWorld 进行实验。

**📈 对比分析**

与无记忆、提示记忆、向量检索、轨迹记忆以及现有经验学习方法（ReasoningBank、Mem0、H‑EPM）对比；实验显示 FRESH 在三大 LLM（Llama‑3.2‑1B、Qwen3‑8B、Gemma‑4‑26B）上均提升任务成功率、工具使用可靠性，并减少执行步数。

**⚠️ 局限性**

局限性包括：需要离线收集并构建图结构，维护与更新图需要额外工程成本；对新工具或策略变化的适应速度受限；在极端复杂或完全未知的任务场景中，检索到的经验可能不足以保障安全。

---

## 549. Field-of-View Extension in Dental Cone-Beam CT via Implicit Neural Representations and Diffusion Model-Based Refinement

**arXiv ID:** 2609.28110 | [PDF](https://arxiv.org/pdf/2609.28110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 550. Three Conjectures on Binary Channels for the Doubly Symmetric Binary Source

**arXiv ID:** 2609.27991 | [PDF](https://arxiv.org/pdf/2609.27991v1)

**作者:** Georg Pichler `[一作]` `[通讯]`, Georg Pichler

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `847a60d8-a755-47af-ba5d-c5236b9e3083` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了关于双对称二进制源的三个重要猜想：平均BSC猜想、以及p=0时双向信息瓶颈的最大化和最小化猜想，并给出了完整的形式化证明。

**💡 创新点**

创新点在于将这些高维极值问题归约到极有限维不等式，并利用形式化验证与AI辅助证明相结合的方式，完成了此前未能完全证明的猜想。

**🔧 技术方法**

采用了偏差表示、奇偶分解、凸包理论、二元变量不等式、超级模数性分析以及形式化证明工具Lean 4（配合Mathlib）来实现证明。

**📊 数据集**

无数据集（纯理论证明）。

**📈 对比分析**

该工作不涉及实验对比，主要通过形式化验证和计算机辅助证明验证其正确性，性能上保证了完全可证性和无误差。

**⚠️ 局限性**

局限性包括：未正式化更大输出字母集的情况（Conjecture 2的通用性）、未完成Conjecture 3以及卡丹数约束的形式化；此外，只在p∈[0,1]的范围内完成，未覆盖更一般的噪声模型。

---

## 551. Fed-ReMasker: Federated Tabular Imputation under Feature-Level Missingness

**arXiv ID:** 2609.28105 | [PDF](https://arxiv.org/pdf/2609.28105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 552. Scenario-Driven Neuroevolution: Using Models to Guide Test Generation for Games

**arXiv ID:** 2609.28130 | [PDF](https://arxiv.org/pdf/2609.28130v1)

**作者:** Gijs van Cuyck `[一作]` (Radboud University), Gordon Fraser `[通讯]` (University of Passau)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将神经进化与模型驱动测试相结合，使用抽象游戏模型作为目标来生成鲁棒的神经网络测试用例，代替传统逐分支覆盖的方式。

**💡 创新点**

创新点包括：① 用模型指导的适应度函数取代逐分支覆盖，显著提升可扩展性并消除误导性 fitness；② 设计简易高层模型与细化模型（课程学习、sink 状态）以进一步引导搜索；③ 通过半自动化模型构建，避免源代码修改且能跨多类游戏使用。

**🔧 技术方法**

使用技术：神经进化（基于 NEAT 的演化算法）、EFSM（扩展有限状态机）游戏模型、branch distance 与 approach level 计算的模型化适应度、统计显著性检验（Mann‑Whitney U 与 Vargha‑Delaney A12）、Scratch 游戏引擎及其测试 harness。

**📊 数据集**

数据集：20 多款基于 Scratch 的小游戏（如 CatchTheDots、CreateYourWorld、FlappyParrot、Pong 等），来自两份已有评估数据集，涵盖多种游戏类型与复杂度。

**📈 对比分析**

比较方法：与传统 code‑guided 基线进行 30 次实验，5 小时搜索预算；评价指标包括 branch coverage、模型状态覆盖率（MC）与游戏胜利次数。结果显示：① 简单模型平均 branch coverage 约为 80%（相较于 85% 的 code‑guided）但在 5 款游戏中显著更好；② 专化模型（课程学习 + sink 状态）将平均 branch coverage 提升至约 89%，并在多款游戏中获得统计显著优势；③ 在大多数游戏中 win 次数与 baseline 相当或更高。整体而言，模型驱动方法在覆盖率和目标达成度上至少提升 7% 以上。

**⚠️ 局限性**

局限性：① 需要人工构建游戏模型，虽然模型较小但仍有人工成本；② 模型抽象可能无法准确捕捉复杂碰撞、时间敏感事件等细节，导致搜索无法覆盖某些分支；③ 仅在 Scratch 环境验证，泛化到其他游戏引擎或更大规模程序需进一步验证；④ 神经网络规模有限，难以同时学习多种任务或处理高维输入；⑤ 只考虑输入生成，缺少更完整的测试或验证机制。

---

## 553. Scaling Attention Head Analysis via Gradient-Based Attribution in Context-Aware Machine Translation

**arXiv ID:** 2609.28117 | [PDF](https://arxiv.org/pdf/2609.28117v1)

**作者:** Paweł Mąka `[一作]` (Maastricht University), Gerasimos Spanakis `[通讯]` (Maastricht University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大规模LLM中，提出将Token-level Max-Margin损失反向传播到注意力图，自动归因注意力头与token‑token关系，完成高效的因果分析。

**💡 创新点**

通过梯度归因替代昂贵的Modifying Heads改动，实现96%+计算节省，并发现跨语言/现象通用的注意力头，揭示模型冗余。

**🔧 技术方法**

使用梯度归因、Token-level Max-Margin损失、Modifying Heads、对比评估与头‑关系筛选技术。

**📊 数据集**

采用Context‑aware MT对比数据集ContraPro、LCPT、ctxPro（四语言方向）以及IOI任务数据集。

**📈 对比分析**

与Modifying Heads的直接改动和BLEU/COMET基准进行对比，梯度方法在Top‑3%头‑关系能捕获所有显著提升，成本降低近97%。

**⚠️ 局限性**

仅能处理已定义的token‑关系；依赖对比评估任务；对极大模型实验受限；梯度非单调性可能导致解释不稳。

---

## 554. Diagnosing the Refuted Mismatched Decoding Converse for Binary-Input Channels

**arXiv ID:** 2609.28109 | [PDF](https://arxiv.org/pdf/2609.28109v1)

**作者:** Jonathan Scarlett `[一作]` `[通讯]` (National University of Singapore), Jonathan Scarlett (National University of Singapore)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对Balakirsky提出的二进制输入离散无记忆信道（DMC）误匹配解码逆定理进行诊断，指出其关键的排列引理和选择序列限制步骤错误，进而证明该逆定理不成立；

**💡 创新点**

首次通过严格的组合与类型方法揭示误匹配逆定理的核心假设失效，证明原定理与已知的superposition coding反例一致，弥补了先前理论与数值结果之间的空缺；

**🔧 技术方法**

使用组合近似引理、类型方法、误码概率的聚合、以及对选择序列的严格概率与集中分析；

**📊 数据集**

无数据集，完全基于理论构造与数值例子（W、V、q的具体矩阵与分布）来展示反例；

**📈 对比分析**

通过与已知的superposition coding可实现率比较，证明在所给信道上LM速率低于真正的误匹配容量，原逆定理预测失效；

**⚠️ 局限性**

仍存在关于组合近似引理和选择序列浓度论证的潜在局限，未能给出完整的局部修复方案，且对更一般信道的适用性需进一步验证。

---

## 555. Visual Tripwires: Anticipating Failure in Deep Vision Systems

**arXiv ID:** 2609.28099 | [PDF](https://arxiv.org/pdf/2609.28099v1)

**作者:** Anoushka Harit `[一作]` (University of Cambridge), Florian Markowetz `[通讯]` (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Visual Tripwires 框架，通过监测模型内部表示、预测轨迹和注意力的时间不稳定性来提前预警视觉模型失效。

**💡 创新点**

创新点是将临时不稳定性作为预警信号，构造四个不稳定指标并聚合成 tripwire 评分，显著早于传统不确定性方法检测到失败。

**🔧 技术方法**

使用表示漂移、预测振荡、轨迹曲率和注意力熵等指标，结合轻量级 MLP 对时间窗口内信号进行聚合，并用二元交叉熵与时间正则化训练。

**📊 数据集**

在 CIFAR‑10‑C、ImageNet‑C 以及 BDD100K 等数据集上评估，覆盖多种网络结构（ResNet‑50、ViT‑B/16、Swin‑T）和多种逐步腐败/分布漂移。

**📈 对比分析**

与 MSP、Predictive Entropy、MC Dropout、Deep Ensembles、ODIN 等基线相比，Visual Tripwires 在 AUROC 上提升约10%+，并将预警领先时间从≈3步提升至≈7步。

**⚠️ 局限性**

局限性包括需时间序列输入、对突变/高度随机环境表现不佳、需要访问内部表示/注意力、阈值与数据/模型迁移需重新校准。

---

## 556. AeRSoM: An Aerial Rigid-Soft Integrated Manipulator for Contact-Rich Manipulation

**arXiv ID:** 2609.28044 | [PDF](https://arxiv.org/pdf/2609.28044v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 557. Curriculum Learning with GNN-based Reinforcement Learning for Job Shop Scheduling

**arXiv ID:** 2609.28085 | [PDF](https://arxiv.org/pdf/2609.28085v1)

**作者:** Jayakrishnan K. Vasudevan `[一作]` (Rosenheim University of Applied Sciences), Noah Klarmann `[通讯]` (Rosenheim University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在作业车间排程（JSSP）中，将基于图神经网络（GNN）的强化学习（RL）与课程学习（CL）相结合，通过逐步增大实例规模来训练调度策略，并与直接在目标规模上训练的单尺寸方法进行对比。

**💡 创新点**

创新点在于提出并系统评估了基于实例规模递进的课程学习策略，能够显著降低训练时间并提升跨规模泛化性能；同时给出了对比单尺寸训练的三维评价框架（泛化、专业化、训练效率）。

**🔧 技术方法**

使用技术包括：GNN编码操作‑机器图、PPO强化学习、操作掩码、Optimality Gap评估指标，以及固定步数的三阶段课程学习（10M步/阶段）。

**📊 数据集**

数据集为随机生成的JSSP实例，满足J/M=1，规模从8×8到30×30，评估集每个规模20个独立实例。

**📈 对比分析**

比较方法：在相同总步数30M下，分别训练单尺寸模型和课程学习模型，周期性评估OG_all（跨规模平均Optimality Gap）、OG_target（目标规模平均OG）以及实际墙壁时间。实验结果显示：20×20时CL略有专业化损失但节省约20h；25×25时CL提升泛化约1.6%，专业化略损失，节省约40h；30×30时CL既提升泛化约8.1%，专业化提升约8.6%，且节省约50h。

**⚠️ 局限性**

局限性包括：课程切换间隔固定为10M步；仅在J/M=1的情形下测试；未尝试更大规模实例或不同的job‑machine比例；未探索自适应或基于难度的课程策略。

---

## 558. Substantive Agency and Computational Non-Anticipability: An Axiomatic Route to a Conditional Separation of P and N P

**arXiv ID:** 2609.28040 | [PDF](https://arxiv.org/pdf/2609.28040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 559. Exact Average Consensus under Noisy Communication Links: A Decentralized Gradient Perspective

**arXiv ID:** 2609.28082 | [PDF](https://arxiv.org/pdf/2609.28082v1)

**作者:** Yuhang Deng `[一作]` (Linköping University), Erik G. Larsson `[通讯]` (Linköping University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在持续链路扰动（MDS）条件下的分布式平均一致性问题，提出引入锚定项的平均一致性算法（AAC），并对其几乎必然收敛到真实初始平均值的性质进行了严格证明。

**💡 创新点**

创新点在于将平均一致性问题重新表述为分布式优化，借助离散梯度下降与随机逼近相结合的框架，引入对初始状态的持续拉回（锚定）来抵消噪声累积，从而实现精确平均一致；同时提供了统一解释，将 NR‑PushSum、动态平均一致等方法归入同一锚定机制之下。

**🔧 技术方法**

采用的技术包括：离散梯度下降（DGD）与随机逼近（SA）双步长递推；马氏差分序列（MDS）模型与条件方差上界；谱半径与投影算子分析；Robbins–Siegmund 不等式及其协变形式；以及对线性迭代的收敛性与稳态误差的解析推导。

**📊 数据集**

实验使用了 Erdős–Rényi 随机图（N=36，连通概率0.5）和初始状态服从均匀分布 [-100,100]；扰动采用 3 位无偏随机量化噪声，构造满足 MDS 条件；未使用公开数据集，实验完全基于仿真。

**📈 对比分析**

与传统的随机逼近平均一致算法（PAC）进行比较。AAC 在噪声影响下能够收敛到真实初始平均值，平均一致误差随迭代趋于零；PAC 则收敛到一个无偏随机变量，其方差保持非零，误差停留在一个较高水平。AAC 的收敛速度相对较慢，但稳态误差更低。

**⚠️ 局限性**

局限性包括：锚定项导致收敛速度下降；需要满足双步长的求和与平方求和条件，调参较为繁琐；在强噪声或高度动态的网络拓扑下可能需要进一步改进；此外，本文主要讨论无向连通图，向有向或时变网络的推广尚未完全阐明。

---

## 560. A Native-Reference Coordinate Geometry for L2 Pronunciation Deviation Using Self-Supervised Speech Models

**arXiv ID:** 2609.28060 | [PDF](https://arxiv.org/pdf/2609.28060v1)

**作者:** Tina Raissi `[一作]` (Aalto University), Mikko Kurimo `[通讯]` (Aalto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一个基于自监督语音模型的本土参考坐标几何，用来评估第二语言发音偏差，而不需要匹配文本。

**💡 创新点**

创新点在于通过对本土语音平均向量进行低秩SVD投影，构造低维参考子空间，然后仅用相同音位类别的坐标距离衡量L2发音偏差，避免了传统需要对齐或标注的限制。

**🔧 技术方法**

采用的技术包括自监督语音编码器（WavLM、wav2vec 2.0）、基于SVD的低秩投影、三种距离度量（欧氏、余弦、马氏）以及两种对齐模型（HMM与CTC）。

**📊 数据集**

实验使用TIMIT（本土语料）与Sandi（第二语言）数据集，并在Sandi的训练/验证集上评估。

**📈 对比分析**

与传统的基于对齐的距离评估相比，本文方法在Spearman相关系数上达到约-0.5（95%置信区间[-0.53,-0.46]），表明更接近本土参考的说话者往往具有更高的CEFR水平，且无需训练显式评分模型。

**⚠️ 局限性**

局限包括对对齐精度的依赖、对不同语种和语料的泛化未知，以及仅在单一评估任务上验证，未来需进一步探究多语种鲁棒性和更复杂的对齐策略。

---

## 561. LiAM-SAM: Lifecycle-Aware Memory for Robust SAM2-Based MOT

**arXiv ID:** 2609.28078 | [PDF](https://arxiv.org/pdf/2609.28078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 562. Evaluating Open-Weight LLMs for Turkish Domain Documents Under Retrieval and Hardware Constraints

**arXiv ID:** 2609.28007 | [PDF](https://arxiv.org/pdf/2609.28007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 563. How Much Were You Told? Measuring External Information in Peer Reviews

**arXiv ID:** 2609.28041 | [PDF](https://arxiv.org/pdf/2609.28041v1)

**作者:** Matthieu Dubois `[一作]` (Sorbonne Université), François Yvon `[通讯]` (Sorbonne Université)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究如何测量论文评审中来自LLM的“外部信息”，通过自我条件化评估生成评论与原始论文+通用指令之间的信息增益。

**💡 创新点**

提出无监督信息理论估计器Self‑Conditioning，能区分完全由LLM生成的评论与仅被润色的评论，并呈现逐渐递增的外部信息水平。

**🔧 技术方法**

使用语言模型概率和信息增益公式，随机抽取提示（如随机词、关键词等），在代理LM上计算自我条件化得分。

**📊 数据集**

采用IntelLabs AI Peer Review Detection Benchmark中的ICLR 2019人类评审及其对应的LLM生成评审，并生成不同信息级别的PI、FD、MP实例。

**📈 对比分析**

与传统AI文本检测器（Binoculars、Fast‑DetectGPT、EditLens等）比较，Self‑Conditioning在区分完全委托与润色评审时AUC高达1.0，且随提示信息数量线性下降，表现出更细致的分级能力。

**⚠️ 局限性**

限制包括对非ML论文、非英语文本的适用性不足；高温采样能逃避评分；方法仅衡量信息增益，不说明作者身份，且需要手动检查高分或低分案例。

---

## 564. No Place to Hide: An Analysis on Protected Order Flow Sandwich Attacks

**arXiv ID:** 2609.28115 | [PDF](https://arxiv.org/pdf/2609.28115v1)

**作者:** Lioba Heimbach `[一作]` (Category Labs), Christof Ferreira Torres `[通讯]` (University of Lisbon)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对六条链（以太坊、Solana、Tron、Base、Arbitrum、Monad）进行为期三年的纵向测量，首次系统检测并量化受保护订单流（private RPC、OFA、局部 mempool、集中 sequencer 等）下的夹层攻击。

**💡 创新点**

① 设计了可捕捉宽松跨块夹层攻击的检测启发式；② 引入持久攻击者过滤器，避免高频交易误判；③ 通过对 OFA 参与者、重组区块、验证器/应用层曝光等多维度分析，揭示不同链的保护失效机制，展示现有防护远低于预期。

**🔧 技术方法**

使用基于交易流动性、方向一致性、利润条件的规则（H1-H3）以及对交易间距、单/多目标判定的统计方法；对 OFA 供应商标记进行交叉匹配；利用重组检测算法识别被公开的私有交易。

**📊 数据集**

采集自六条链的 AMM 交易数据（包括 Uniswap、SushiSwap、Raydium、Meteora 等）从 2023‑07‑01 至 2026‑06‑30 的全链块与交易日志；同时收集各 OFA（MEV‑Share、MEV‑Blocker、Blink、Merkle）提供的交易曝光信息。

**📈 对比分析**

对比公开 mempool 夹层与受保护订单流夹层的攻击数量、成功率、手续费占比及盈利；在同一链内对不同保护机制的攻击强度进行分层评估；结果显示：Solana 最高攻击量（≈2.8M 次），以太坊与 Tron 次之，Base 及 Arbitrum 低或无；受保护夹层的利润率普遍高于公开夹层，但手续费占比更大，攻击者多为单一实体。

**⚠️ 局限性**

① 只关注 AMM 交换交易，可能遗漏其他 DeFi 交互导致的夹层；② 多目标夹层中无法精确定位实际攻击目标；③ OFA 记录不完整，导致部分私有交易的曝光渠道被低估；④ 重组区块检测基于 Xatu 记录，可能漏掉未报告的重组；⑤ 研究周期内链的协议变更或保护措施更新可能影响结果，但未能完整归因。

---

## 565. Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching

**arXiv ID:** 2609.28107 | [PDF](https://arxiv.org/pdf/2609.28107v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 566. Controlled Attribute-Specific Summarization of Interrogative Dialogues

**arXiv ID:** 2609.28004 | [PDF](https://arxiv.org/pdf/2609.28004v1)

**作者:** A Aditya Bhardwaj `[一作]` (Indian Institute of Information Technology Delhi), Md Shad Akhtar `[通讯]` (Indian Institute of Information Technology Delhi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CASPER框架，利用链式思维属性特定提示和迭代评估生成法医对话摘要，并创建MINDSum数据集。

**💡 创新点**

将事件细节、事实细节、角色描述嵌入提示，并引入分层角色评估（RoleEval），实现多属性、可解释且高事实一致性的摘要。

**🔧 技术方法**

使用结构化提示、实体提取、链式思维推理、迭代自我修正及三层角色评估（officer/inspector/senior inspector），基于大型语言模型（Llama3.1/ GPT‑4o‑mini）。

**📊 数据集**

MINDSum（从MIND扩展而来），包含约6,000条受审问对话，手工标注事件、事实、角色及填充词。

**📈 对比分析**

与SumCoT、PromptSum、PLASMA、ToT、Reflexion、GPT‑4o、BART/T5/PEGASUS等基线在ROUGE、BERTScore和人类评估下对比，CASPER在所有指标上均显著领先，错误率最低。

**⚠️ 局限性**

仅支持单语（英文），依赖大型模型推理耗时长，角色评估模型可能出现自我偏好，对多语言或低资源场景尚未验证。

---

## 567. A Flexible Recommendation System for Individuals and Groups

**arXiv ID:** 2609.27998 | [PDF](https://arxiv.org/pdf/2609.27998v1)

**作者:** Yacine Mokhtari `[一作]` (IMT Atlantique), Grégory Smits `[通讯]` (IMT Atlantique)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于知识图谱增强的图注意网络(GAT)的双重用户表示模型，既学习用户在单独交互时的偏好，又学习其在群体中交互时的偏好，并通过对两种表示的差异度量（duality、diversity）评估用户适应性，进而制定分层的双极聚合策略，实现个体与群体推荐的统一；

**💡 创新点**

核心创新在于：①为同一用户学习两种独立表示；②利用知识图谱的语义信息增强表示；③提出用户适应性度量并据此进行群体成员分层；④设计基于least和avg的双极聚合函数，既满足非适应性用户的需求，又兼顾适应性用户的影响；

**🔧 技术方法**

技术手段包括：TransR KGE + GAT 进行节点嵌入；BPR 损失训练个体与群体交互的匹配评分；余弦相似度与平均异质性度量用户适应性；双极聚合策略 (least + avg) 对候选项目进行筛选与排序；

**📊 数据集**

实验使用基于 MovieLens-KG 的合成数据集，包含 three 大规模数据集（small、small2、large），生成不同用户类型和群体组合；基准模型包括 KGAT、KGAT-PGR、KGAG 等；

**📈 对比分析**

通过 HR@20 与 NDCG@20 评价，比较个体推荐、已见群体推荐和未见群体推荐三种场景；结果显示：在个体推荐上优于所有基线；在已见群体推荐上与 KGAT‑PGR 相近；在未见群体推荐上显著优于 OGR 与 PGR，尤其在 HR@20 和 NDCG@20 上取得最高分；

**⚠️ 局限性**

局限性：实验仅基于合成数据，缺乏真实含知识图谱数据验证；需进一步评估阈值选择对用户分类的敏感性；未对多群体动态变化进行建模；后续工作需开展鲁棒性分析和消融实验。

---

## 568. AstraLOD3: Zero-shot multimodal agentic reconstruction of LOD3 building models

**arXiv ID:** 2609.28061 | [PDF](https://arxiv.org/pdf/2609.28061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 569. From Alignment to Fusion in 3D Vision-Language

**arXiv ID:** 2609.28222 | [PDF](https://arxiv.org/pdf/2609.28222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 570. Compliant with Local Controls, Collectively Discriminatory. A Governance Architecture for Multi-Agent AI in Regulated Finance

**arXiv ID:** 2609.27994 | [PDF](https://arxiv.org/pdf/2609.27994v1)

**作者:** Jose Manuel de la Chica Rodriguez `[一作]`, Pablo Delgado Romero `[通讯]` (Grupo Santander)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了面向金融的代理人群治理参考架构（ARIA）和可验证研究议程，并通过两种简化仿真验证了其有效性。

**💡 创新点**

创新点在于识别并命名“宪法非组合性”，构建包含六项治理能力（政策规范、群体监控 M2、受限授权、运行时执行、适应性策略学习、人工监督能力）的三平面架构，并提出可操作的观察‑与‑期望统计指标。

**🔧 技术方法**

使用的技术包括：多代理工作流建模、运行时监控与封锁机制、M2 观察‑与‑期望分布一致性统计、受限决策授权（BDA）与执行点（PDP/PEP）、红队演练与自适应政策更新流程、人工监督能力评估框架。

**📊 数据集**

数据集：仅使用合成信用申请样本（thin‑file vs thick‑file）与两阶段漂移实验的离散动作类别；未使用真实金融或 LLM 训练/推理数据。

**📈 对比分析**

对比方法：①将本地控制仅实施与添加群体层级容限封锁对比，结果显示本地控制下公平率差距增长 8 倍，群体层级容限将差距控制在 4% 以内；②将 M2 监测与单一违规频率监测对比，在 200 轮实验中 M2 触发时间平均 65 步，违规频率监测仅 12 步，表明 M2 在漂移早期能更早报警。

**⚠️ 局限性**

局限性：仅给出有限的对抗示例与机制演示，仿真中代理为解析决策器而非真实 LLM；M2 受支持集限制，可能漏报；缺乏在真实金融机构部署的实证验证；架构的可行性仍需通过多阶段现场验证和外部评估来检验。

---

## 571. Load Balancing with Partial Queue Information - Threshold Optimality and Indexability

**arXiv ID:** 2609.28219 | [PDF](https://arxiv.org/pdf/2609.28219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 572. From Agent Output to Authorized Transition

**arXiv ID:** 2609.28216 | [PDF](https://arxiv.org/pdf/2609.28216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 573. Two Global Crops Suffice: Locating Semantic Emergence in DINO-Style Self-Supervised Learning

**arXiv ID:** 2609.28187 | [PDF](https://arxiv.org/pdf/2609.28187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 574. VLMs Can Describe, But Not Measure: Object-Centric Scene Understanding for Robotic Manipulation

**arXiv ID:** 2609.28184 | [PDF](https://arxiv.org/pdf/2609.28184v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 575. Large Language Models in the UK: Public Use, Trust, and Attitudes

**arXiv ID:** 2609.28176 | [PDF](https://arxiv.org/pdf/2609.28176v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 576. Homological Trimming and Regularity of Filtrations via Local Obstruction Modules

**arXiv ID:** 2609.28160 | [PDF](https://arxiv.org/pdf/2609.28160v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 577. Lettericity Is NP-Complete

**arXiv ID:** 2609.28023 | [PDF](https://arxiv.org/pdf/2609.28023v1)

**作者:** Henning Fernau `[一作]` (University of Trier), Kevin Mann `[通讯]` (University of Trier)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明了图的字母度（lettericity）问题是NP-完全的，并进一步证明了两种相关问题——着色扩展（coloring extension）和词扩展（word extension）——在其最小形式下亦为NP-完全；同时给出了在指数时间假设（ETH）下的下界，表明不存在 2^o(n) 的确定性算法；其核心技术是构造一系列基于“pair class”的图形子图（如等价、包含以及 NAE 子句 gadget），并利用左/右指向图（Lτ(G)、Rτ(G)）的无环性来完成从 Monotone‑NAE3‑SAT 的多项式时间归约。

**💡 创新点**

创新点在于：1) 首次给出字母度问题的完整 NP-完全性证明；2) 通过同一归约框架证明着色扩展和词扩展的 NP-完全性，解决了近期文献提出的开放问题；3) 在归约过程中利用字母顺序与颜色对应的“τ”函数以及 Lτ/Gτ 的无环性，提供了一种新的图表示技术；4) 基于归约构造出的字母度实例在规模上满足 |V| = 6k，从而推导出 ETH 下的 2^o(n) 下界。

**🔧 技术方法**

主要技术包括：图形子图（gadgets）设计、pair class 与 τ 定义、左/右指向图构造、区分器与区间的理论、以及从着色扩展到字母度的多阶段多项式归约。

**📊 数据集**

本文为理论计算复杂性研究，没有使用实验数据集；所有结果均基于严格的归约证明与复杂性论证。

**📈 对比分析**

由于是理论研究，没有与具体算法或基准进行实验比较；论文通过证明 NP-完全性和 ETH 下界，表明在一般图上不存在多项式或 subexponential 的多项式时间算法，提供了对现有算法（如 2^O(k^2 2^2k)n^3）相对的理论性能极限。

**⚠️ 局限性**

局限性：仅在一般图上给出了 NP-完全性；对特殊图类（如树、线图、cograph 等）尚未给出完整的多项式算法或更细的参数化复杂性结果；此外，虽然给出了 2^o(n) 的下界，但仍未确定能否实现 2^O(k) 的 FPT 算法，且实际实现的字母度表示仍缺乏有效的构造算法。

---

## 578. ODPure: Backdoor Purification for Object Detection via Ensemble Corruption Consensus

**arXiv ID:** 2609.28239 | [PDF](https://arxiv.org/pdf/2609.28239v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 579. Log-Depth Recurrent Language Modeling

**arXiv ID:** 2609.28212 | [PDF](https://arxiv.org/pdf/2609.28212v1)

**作者:** Yiqin Wang `[一作]` (Imperial College London), Charles Pert `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于平衡树的递归运算的工作高效并行扫描算法，用于自回归语言模型的前缀表示和下采样，从而实现对所有前缀的对数深度计算。

**💡 创新点**

将并行扫描的下采样与递归运算相结合，首次实现了对数深度自回归模型的工作高效训练，并通过非关联门控递归单元提升表达能力。

**🔧 技术方法**

使用Gated Recursive Cell (GRC) 作为二元组合运算，配合层归一化、残差 MLP，利用并行扫描实现对数深度的前缀生成；对比 Transformer 的 ALiBi 与正弦位置编码。

**📊 数据集**

在 Penn Treebank、WikiText-2 与 OpenWebText2 三个规模递增的语言建模数据集上进行实验。

**📈 对比分析**

通过相同参数规模的 Transformer 基线（Sinusoidal 与 ALiBi）对比；在 PTB 和 WikiText-2 上，AR‑GRC 的困惑度仅高 6–13%；在 OpenWebText2 上差距扩大 23–34%；在超长上下文（高达 2976 tokens）时，AR‑GRC 与 ALiBi Transformer 的性能保持稳定，而正弦 Transformer 失效；评估时间随长度线性增长，而 Transformer 则二次增长。

**⚠️ 局限性**

在大规模多样化语料上的性能仍显不足，可能由固定树结构、信息压缩路径限制以及非关联运算导致；此外缺乏对更大模型规模和训练数据的评估与可扩展性研究。

---

## 580. PASTABench: Proactive Assessment of Sequential Trajectories for Agent Safety

**arXiv ID:** 2609.28197 | [PDF](https://arxiv.org/pdf/2609.28197v1)

**作者:** Jiapeng Sun `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了 PASTABench 基准，用于评估大语言模型在多步代理交互中的主动安全监测能力。

**💡 创新点**

创新点在于：① 将主动安全分为“是否、何时、是什么”三维度；② 引入可量化的最佳干预窗口 (OIW) 和时间锚点；③ 设计层级风险分类体系并提供 1,139 条带时间标签的动态轨迹；③ 将安全评估从事后判定转变为实时步骤级判断。

**🔧 技术方法**

采用 DeepSeek-R1 合成多步轨迹，基于层级标注与双重验证的标注流程，利用 GPT‑4o 与 Gemini‑2.5‑Pro 进行交叉标注，最终通过人工审核完成验证；评估指标包括 Good、Effective、Perfect 干预率。

**📊 数据集**

数据集来自 Recipe2Plan、LabSafetyBench 与 SafeToolBench，后经过滤、工具库构建、轨迹合成与标注，形成 1,139 条多类别（5 类主，13 类次）的安全场景。

**📈 对比分析**

对 16 个 LLM（GPT‑4o、Gemini 系列、Grok‑3、Llama、Qwen、DeepSeek‑R1、QwQ‑32B 等）进行零样本评估；最佳模型 Perfect 干预率仅 40.74%，Early 干预率普遍偏高；Effective 干预率平均 55.23%；不同风险类别间差距显著，尤其是 Procedural Safety Gap。

**⚠️ 局限性**

局限性包括：① 仅使用文本描述模拟状态，缺乏多模态感知；② 评估主要关注词汇触发，易受词汇过拟合影响；③ 受限于合成环境，缺乏真实物理反馈；④ 需要进一步扩展到更复杂、多模态和更真实的交互场景。

---

## 581. EmbodiedMemory-Bench: Benchmarking Embodied Memory for Long-Horizon Embodied Tasks

**arXiv ID:** 2609.28236 | [PDF](https://arxiv.org/pdf/2609.28236v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 582. Beyond Poetry: Can Large Language Models Generate Classical Arabic Maqamat?

**arXiv ID:** 2609.28245 | [PDF](https://arxiv.org/pdf/2609.28245v1)

**作者:** AbdulRahman A. Morsy `[一作]` (George Washington University), Aya Zirikly `[通讯]` (George Washington University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在现代大型语言模型上，系统评估并比较了五个模型在生成阿拉伯古典文学形式马卡马（具有韵律散文、修辞装饰和情节结构）的能力。

**💡 创新点**

首次将马卡马生成作为评估对象，构建多种提示策略（零-shot、少量示例、规则式提示）并引入 LLM-评审与人工评审相结合的评估框架。

**🔧 技术方法**

使用 GPT-4o、GPT-5.4-mini、ALLaM-7B、Qwen3-8B、LLaMA-3-8B 等大型语言模型，以及 Claude Sonnet 4.5 和 Gemini 3.5 Flash 作为评审器。

**📊 数据集**

利用自行构造的 14 个主题提示生成 5 条样本，共 1,050 条马卡马式文本，构成人工标注和 LLM 评审的双重数据集。

**📈 对比分析**

通过绝对量表评分和成对比较，以及 Elo 排名、统计显著性检验，发现 GPT-5.4-mini 最高，GPT-4o 次之，开放权重模型远低于两者；提示方式对弱模型影响较大，强模型对提示依赖小。

**⚠️ 局限性**

局限包括仅评估短文本、提示和主题覆盖有限、评审主要依赖 LLM 可能存在偏差、样本规模和对长篇马卡马结构的不足。

---

## 583. MemBodied: Recurrent Associative Memory for Vision-Language-Action Models

**arXiv ID:** 2609.28256 | [PDF](https://arxiv.org/pdf/2609.28256v1)

**作者:** Tej Deep Pala `[一作]` (Nanyang Technological University), Soujanya Poria `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种固定容量的回环式记忆机制，用关联状态和初始场景锚点在不扩展上下文长度的前提下提升视觉-语言-动作模型的历史依赖操控性能。

**💡 创新点**

创新点在于结合可写入关联矩阵和初始场景锚点的双通道记忆结构，使用门控 delta 规则进行在线写入，同时保持记忆尺寸固定且不随回合长度增长。

**🔧 技术方法**

使用的技术包括关联记忆（Associative Memory）与门控写入、跨层矩阵更新、跨注意力(anchor)检索、以及延迟写入的因果训练。

**📊 数据集**

评估数据集包括 RMBench（5个需要记忆的操控任务）、LIBERO（四套包含长短周期任务）、以及三种真实机器人任务。

**📈 对比分析**

与无记忆、帧堆叠、提示辅助、普通递归记忆、压缩历史视频记忆等基线对比；在 RMBench 上实现 50.0% 平均成功率，比无记忆提高 43.6个百分点；在 LIBERO-Long 上提高 5.4%；与 NativeMEM 相比，成功率高、推理延迟低 91.9%，参数量仅为 1.26%。

**⚠️ 局限性**

局限性包括对初始场景锚点的依赖在某些任务（如 Block Ranking）上可能降低性能；记忆机制对任务特定的记忆需求仍需进一步细化，且在超长回合或多样化环境下的泛化性尚待验证。

---

## 584. Large-Scale Geometric Map-Based Localization of UAVs in GNSS-Denied Urban Environments

**arXiv ID:** 2609.28225 | [PDF](https://arxiv.org/pdf/2609.28225v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 585. Do Electromagnetic Side-Channel Attacks Threaten Electronic Polling Stations? Scenarios and Recommendations

**arXiv ID:** 2609.28209 | [PDF](https://arxiv.org/pdf/2609.28209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 586. Ideal Membership in Polynomial Calculus: Complexity and Reductions

**arXiv ID:** 2609.28243 | [PDF](https://arxiv.org/pdf/2609.28243v1)

**作者:** Alex Bortolotti `[一作]` (University of Applied Sciences and Arts of Southern Switzerland), Monaldo Mastrolilli `[通讯]` (University of Applied Sciences and Arts of Southern Switzerland)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出了一套基于pp‑定义、pp‑解释与pp‑编码的代数归约框架，用以在多项式演算（Polynomial Calculus）中构造固定度数的理想归属（Ideal Membership）证书，并利用该框架识别出新的可解约束满足语言类（尤其在三元及更大域上）以及完全解决布尔域的_ d 和-_ d 归属问题；同时给出了一个针对-_ 1 的无条件下界并将其与Sum‑of‑Squares自动化问题联系起来。

**💡 创新点**

创新点包括：①引入的归约框架能够在保留度数和系数位数的前提下，将求解问题从一种约束语言迁移到另一种；②利用中位数操作和固定值三项式的结构，得到整个三元域上的新可解类；③完全解决布尔域的_ d 与-_ d 的判定二分理论，完成了之前只剩最后一个未决案例的分类；④给出-_ 1 的无条件下界，突破了此前仅存在条件下界的限制；⑤通过归约展示了可解类上 Sum‑of‑Squares 证明的度数自动化。

**🔧 技术方法**

主要技术手段包括：多项式演算的推导规则与Buchberger Gröbner 基计算、有效Nullstellensatz与强Nullstellensatz、pp‑定义/解释/编码的逻辑与代数构造、极大值与极小值的中位数运算、固定值三项式、子代数/直积/同态像的代数闭包、插值多项式的构造与证明、以及对布尔多项式系统的Affine/多项式分解与下界证明。

**📊 数据集**

本工作为纯理论研究，无需使用实验数据集。

**📈 对比分析**

由于是理论分类与算法复杂度分析，研究结果主要表现为：在固定度数 d 的条件下，存在多项式时间（n^O(d)）的求证算法；对新可解类证明了可在该时间复杂度内完成；对布尔域给出了完整的二分判定，证明了在不满足对应多项式运算的极值或多项式分解时问题为NP‑难；对-_ 1 给出了指数下界，表明无固定度数的证明不可避免。

**⚠️ 局限性**

局限性主要体现在：①对除三元、布尔域外的更大域仍未得到完整的_ d /-_ d 分类；②归约过程虽然保留了度数和系数大小，但在实际实现中常常涉及常数因子与多项式度数的增加，可能导致高常数与大隐式因子；③对于非常大的 d 或高维实例，n^O(d) 的多项式虽然理论可行，但实际计算仍可能不可行；④本框架对非有限域或非多项式演算的推广尚未解决。

---

## 587. Do Center Biases Propagate? Robustness of Pathology Foundation Models in Whole-Slide Image Classification

**arXiv ID:** 2609.28231 | [PDF](https://arxiv.org/pdf/2609.28231v1)

**作者:** Ilán Carretero `[一作]` (Universitat Politècnica de València), Valery Naranjo `[通讯]` (Universitat Politècnica de València)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估不同路径学基础模型在中心相关偏差下的鲁棒性，构造不同Cramér's V关联训练集并引入AUCC指标；

**💡 创新点**

提出同时衡量准确率与对中心‑标签关联稳健性的AUCC指标；系统比较六个PFM与两种MIL聚合器在四个多中心数据集上的表现；验证ComBat对PFM表示的中心偏差缓解效果；

**🔧 技术方法**

采用多实例学习（ABMIL、TransMIL）、PFM编码（多模态与单模态）、Cramér's V关联度量、AUCC面积计算以及ComBat特征归一化；

**📊 数据集**

使用AI4SKIN、CAMELYON16、TCGA‑BRCA、TCGA‑NSCLC四个二中心二分类数据集；

**📈 对比分析**

通过5折交叉验证和10次随机采样，比较BACC与AUCC，结果显示ABMIL在大多数配置下AUCC更高，CONCH与KEEP等多模态PFM表现更稳健，ComBat效果不一致，往往导致AUCC下降；

**⚠️ 局限性**

局限在仅二分类、两中心、补丁级PFM、有限的MIL聚合器，Cramér's V构造受样本限制，缺乏对更大临床场景和切片级模型的验证。

---

## 588. A Unified Framework and Dataset for Oriented Object Visual Grounding in Remote Sensing

**arXiv ID:** 2609.28230 | [PDF](https://arxiv.org/pdf/2609.28230v1)

**作者:** Zeyu Ding `[一作]` (China University of Mining and Technology), Abdulmotaleb El Saddik `[通讯]` (University of Ottawa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出O^2‑VG框架，包含O^2‑VG‑Trans、O^2‑VG‑Uni和O^2‑VG‑VLM，完成遥感图像中带任意角度的目标视觉定位；

**💡 创新点**

创新点在于：①统一的跨模态Transformer与自回归VLM相结合；②无文本提示的通用定向框提议生成与检索；③在VLM中采用多令牌并行预测与混合推理；并构建全新DIOR‑R‑RSVG数据集；

**🔧 技术方法**

使用跨模态Transformer、变形注意力、投影+多头注意力、CLIP/RemoteCLIP文本编码、Moon‑ViT视觉编码、Qwen2.5解码器，以及多令牌预测与混合推理；

**📊 数据集**

利用DIOR‑R‑RSVG（17,402图像，38,320表达式，20类别），并在VRSBench与AVVG三大基准上进行评测；

**📈 对比分析**

相较于现有的Rotated GiT、Rotated GDINO、Rotated EGDINO、GeoChat、GeoGround等方法，O^2‑VG‑Trans在DIOR‑R‑RSVG上Pr@0.5提升至67.23%、meanIoU提升至56.73%；O^2‑VG‑Uni在VRSBench/AVVG/DIOR‑R‑RSVG的AR_50:95分别达到56.41%、75.36%、60.78%；O^2‑VG‑VLM在DIOR‑R‑RSVG上的Pr@0.5、meanIoU、cumIoU分别达78.35%、67.85%、74.66%，均优于GeoChat/GeoGround；

**⚠️ 局限性**

局限在：①仍依赖高质量预训练视觉/文本模型，若迁移到其他遥感域需重新预训练；②多令牌预测虽然加速但在极端角度或密集场景下偶有误检；③通用框提议在小目标检出率相对低；④模型参数量较大，推理成本仍高；

---

## 589. EvEMTBench: An Open Benchmark for Machine Learning in Power System Protection

**arXiv ID:** 2609.28149 | [PDF](https://arxiv.org/pdf/2609.28149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 590. MimicSat: A Reconfigurable Cyber-Physical Testbed For Small Satellite Systems and Cybersecurity Research

**arXiv ID:** 2609.28228 | [PDF](https://arxiv.org/pdf/2609.28228v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 591. Transferable Evidence Reconstruction for Longitudinal Glucose Representations

**arXiv ID:** 2609.28199 | [PDF](https://arxiv.org/pdf/2609.28199v1)

**作者:** Tian Zhou `[一作]` (Ant Group), Liang Sun `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种新的自监督学习框架TER，利用可迁移的结构化信号证据和跨组读者训练来学习连续血糖监测(CGM)记录的共享表示，随后使用时钟感知的多日编码器将日内观测与时间绑定，以获得更具解释性的长期特征；

**💡 创新点**

创新点在于：①通过在一个记录组上拟合临时线性读者并在另一个组上直接验证，传递解码规则而非具体权重，确保证据可迁移；②引入结构化证据（如高血糖负荷、波动性、持续性等）而非点对点恢复；③设计观察感知的日编码器和时钟感知的记忆模块，将血糖值与记录时间显式绑定，提升跨天信息组织；

**🔧 技术方法**

使用的技术包括：自监督可迁移证据重建(TER)、可微分岭回归临时读者、Masked Autoencoding、Contrastive Learning、注意力池化、时钟频率变换以及线性分类器做下游评估；

**📊 数据集**

数据集为23,336个私有和2,283个公开的24小时CGM窗口，涵盖四个临床队列（CGMacros、ShanghaiT2DM、Stanford、Hall），共14个预测任务；

**📈 对比分析**

与公开的泛化时间序列模型（Chronos、MOMENT、Mantis等）以及专门的CGM模型（CGMformer、CGM-JEPA、X-CGM-JEPA、GlucoFM）进行对比，TER在整体PR‑AUC、ROC‑AUC和Macro‑F1上分别比GlucoFM领先5.51/4.43/2.80个百分点，在14个任务上占据42个单元格中的30个；

**⚠️ 局限性**

局限性包括：①需要大量无标签CGM数据来构建证据函数，可能不适用于数据稀缺场景；②跨组读者训练依赖记录身份划分，若身份信息不完整或混乱可能影响性能；③模型在处理跨设备或跨时区的时间同步问题上尚未充分验证；

---

## 592. From Change Captions to Change Detection: Semantic-Appearance Agreement Framework for Remote Sensing Change Detection

**arXiv ID:** 2609.28192 | [PDF](https://arxiv.org/pdf/2609.28192v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 593. From ECG Signals to Representative-Morphology Heatmaps for Biometric Recognition

**arXiv ID:** 2609.28183 | [PDF](https://arxiv.org/pdf/2609.28183v1)

**作者:** Athanasios Angelakis `[一作]` (University of Bundeswehr Munich), Marta Gomez-Barrero `[通讯]` (University of Bundeswehr Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了一种基于 ECGXtractor 的代表性心电波形热图（representative-morphology heatmap）并与传统稀疏 ECG 曲线做对比，用同一数值矩阵生成两种图像；

**💡 创新点**

创新点在于构造了一个确定性的、稠密时间-通道热图，使得仅通过图像表示即可保留完整心电形态信息，并通过匹配渲染方式精确测定图像表示对生物识别性能的影响；

**🔧 技术方法**

技术主要包括 1) ECG 预处理与 R 波峰对齐、10 心拍块中选取五个最中心心拍平均得到 400×L 代表矩阵；2) 将该矩阵渲染为传统 Trace 和稠密热图；3) 在多种小型 CNN/Transformer 以及 ImageNet 预训练大模型上训练特征提取器并进行生物识别；4) 对比验证和闭集识别、信号域基线、通道消融及图像降噪实验；

**📊 数据集**

使用了三大公开数据集：PTB（12 通道）、ECG-ID（单通道 Lead I）和 MIMIC-IV-ECG-DEMO（12 通道）；

**📈 对比分析**

在所有 15 种小型模型与数据集组合中，热图均优于 Trace，平均 EER 降低 9.59% 点、Rank‑1 提升 24.69% 点；在 ImageNet 预训练的更大模型上，ConvNeXt‑Tiny 在 PTB/Ecg‑ID 上实现 2.43%/5.79% 的 EER，DeiT‑Base 在 MIMIC‑DEMO 上达到 14.92% EER；与信号域方法相比，热图模型在大多数指标上竞争力强；

**⚠️ 局限性**

局限包括：未评估跨录制/跨设备的长期稳定性；通道消融分析仅为评分级别，未验证实际减传感器的可行性；此外，训练与评估均基于固定划分，未考察不同采样或光照噪声对模型的鲁棒性。

---

## 594. DAVIS: A Depth-Only End-to-End Active-Vision Framework for Humanoid Soccer Skills

**arXiv ID:** 2609.28175 | [PDF](https://arxiv.org/pdf/2609.28175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 595. A comparative assessment of global building and settlement datasets across geographic and settlement contexts

**arXiv ID:** 2609.28154 | [PDF](https://arxiv.org/pdf/2609.28154v1)

**作者:** Rufai Omowunmi Balogun `[一作]` (World Bank), Edward Charles Anderson `[通讯]` (World Bank)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

对七个全球/近全球建筑与人类居住层数据集进行基准评估，使用多来源参考建筑图并结合检测、几何、量化等多维度指标。

**💡 创新点**

提出面向不同操作需求的多维度评价框架，整合矢量与栅格数据的互补指标，并对密度、尺寸、参考来源、时间对齐进行分层和诊断分析。

**🔧 技术方法**

利用对象匹配、IoU、F1、相对面积误差、位置误差、量化差异（QD/AD）、时间对齐实验等GIS与统计技术；采用UTM重投影、栅格化、块平均等处理流程。

**📊 数据集**

评估产品包括Overture Maps、Global Building Atlas、3D‑GloBFP、Google Open Buildings 2.5D Temporal、Microsoft TEMPO、GHSL、WSF Tracker；参考数据来自HOTOSM、人行世界银行项目、SpaceNet 7、Edinburgh等。

**📈 对比分析**

以城市级宏观F1为主要排序依据，矢量中Overture最佳（F1 0.786），10 m栅格OBT最佳（F1 0.642），100 m栅格WSF Tracker最佳（F1 0.862）；同时评估面积误差、计数偏差以及与建筑密度、尺寸的相关性，揭示不同产品在不同尺度与环境下的表现差异。

**⚠️ 局限性**

时间不匹配导致误差保守；参考与候选数据共享来源可能提升结果；SpaceNet 7使用不同IoU阈值；评估仅在1 km×1 km tiles并以城市级别汇总，不能用于单栋建筑级决策。

---

## 596. Pinpointing Super-Quadratic Quantum Enumeration Speedups: Exact and Certified Evaluation of the Guessing-Moment Exponent under Product-Distribution Advice

**arXiv ID:** 2609.28226 | [PDF](https://arxiv.org/pdf/2609.28226v1)

**作者:** Carsten Schubert `[一作]` (Technische Universität Berlin), Marian Margraf `[通讯]` (Freie Universität Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种可计算的、无误差的方案，用来评估在产品型概率建议（product‑distribution advice）下 Montanaro 量子搜索算法相对于经典枚举的“猜测-矩”指数（speedup exponent）。通过把关键的期望排名转化为单维惊讶度（surprisal）分布的函数，并在满足共线性或可离散化的情形下给出精确求值方法；在一般非共线情形下提供离散化（binning）技术并给出后验误差上界，最终对冷启动泄漏、Keccak‑based side‑channel 和模板攻击等实用密码学场景进行了实验验证。

**💡 创新点**

创新点包括：①将二阶及以上猜测矩的最优值完全归约到惊讶度分布的一个可计算函数；②在共线性情况下给出无离散化误差的有限和算法；③提出统一的离散化策略与可验证的误差上界；④利用指数倾斜（exponential tilting）实现数值稳定的高精度计算；⑤在具体密码学实验中发现某些非均匀建议可实现显著超二次加速（指数可达 3.97）。

**🔧 技术方法**

核心技术包括：
- 惊讶度（surprisal）与顺序枚举的等价性；
- 单维卷积与快速傅里叶变换（FFT）并行合并树；
- 指数倾斜变换以消除指数级小量；
- 近似（CLT、Bahadur‑Rao 等）与离散化误差分析；
- 可靠性区间与后验误差上界的自适应计算。

**📊 数据集**

使用的数据集为：
- 冷启动泄漏实验中的 AES‑128 / AES‑256、PRESENT‑80 真实泄漏数据；
- Keccak‑based ML‑KEM/ML‑DSA 的残余概率模型（合成 Ber(β) 先验）；
- AES 模板攻击的 Gaussian 泄漏模型（SNR 1 与 5）。
所有数据均以产品型概率分布形式给出，并在实验中通过模拟或已公布的攻击结果构造。

**📈 对比分析**

与既往的基于 Renyi 熵的上界（Arıkan 等）相比，本文得到的指数往往更大且精确；在冷启动场景中，指数从 2.7 提升至 3.97；在模板攻击场景中指数仅略高于 2，验证了熵估计的保守性。实验结果表明：
- 对于高度偏斜的建议，Montanaro 算法明显优于普通 Grover；
- 对于平滑或均匀建议，指数接近 2，表明量子加速不显著。

**⚠️ 局限性**

局限性：
- 仅适用于产品型建议，无法处理坐标间存在依赖的分布；
- 对大字母表或高维度时，FFT 计算和内存需求仍然较高；
- 离散化误差上界依赖于猜测矩的梯度估计，在极端偏斜或接近均匀的情形下可能不够紧凑；
- 结果基于理想的 Montanaro 算法，实际实现时仍需考虑量子错误率与资源开销。

---

## 597. Personalised versus Posted Pricing from Samples

**arXiv ID:** 2609.28181 | [PDF](https://arxiv.org/pdf/2609.28181v1)

**作者:** Pieter Kleer `[一作]` (Tilburg University), Daan Noordenbos `[通讯]` (Tilburg University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究在只给定有限样本的情况下，如何用简单的价格（如样本均值或顺序统计量）来逼近个体化定价（即取平均价）获得的最大收入。

**💡 创新点**

在MHR分布下证明样本均值是最优的；在λ正则分布中给出单样本下最优确定性策略，并证明多样本情况下基于顺序统计量的策略在样本数量趋大时达到最优（误差O(1/n)）。

**🔧 技术方法**

利用双无限线性规划、超几何函数、Beta函数、组合恒等式、近似理论与概率论等工具进行分析与证明。

**📊 数据集**

论文不使用外部数据集，而是以理论分布族（λ-正则、MHR、Pareto等）为模型进行研究。

**📈 对比分析**

通过比较最优的样本定价规则与完美信息下的个体化定价，给出乘法保证：MHR时保证为(n/(n+1))^{n+1}；单样本确定性规则保证为(1-λ)/(4-2λ)；多样本顺序统计量规则在n→∞时收敛到(1-λ)^{1/λ}，误差为Θ(1/n)。

**⚠️ 局限性**

局限包括仅考虑单个买家、使用平均价作为基准、单样本随机化策略的最优性未完全确定，以及对小样本数n和λ>0时具体最优策略仍未解析。

---

## 598. Complementary Roles of Activation and Parametric Memory in Few-Shot Learning

**arXiv ID:** 2609.28250 | [PDF](https://arxiv.org/pdf/2609.28250v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 599. Confidence Falls Short: Asymmetric Certainty Gains from Optimization Hinder Multimodal Classification

**arXiv ID:** 2609.28165 | [PDF](https://arxiv.org/pdf/2609.28165v1)

**作者:** Longfei Huang `[一作]` (Nanjing University of Science and Technology), Yang Yang `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种多模态最大置信度正则化（MaxCR）方法，动态通过最大抑制和最大激励机制校准强弱模态的预测置信度，解决模态不平衡问题。

**💡 创新点**

创新点在于发现优化-置信度解耦现象，提出利用非线性稀疏度指标监测置信度差异，并通过正则化实现置信度的自适应干预，从而实现置信度平衡。

**🔧 技术方法**

技术手段包括非线性稀疏度监测、最大抑制/最大激励损失、跨模态正则化、基于ResNet、BERT等深度模型的特征提取与融合。

**📊 数据集**

实验使用了CREMAD、KSounds、VGGSound、Twitter、Sarcasm和NVGesture六个多模态数据集。

**📈 对比分析**

与多种传统与重平衡的多模态基线（如Concat、Affine、ML-LSTM、G-Blend、MSLR、AGM、MMPareto、SMV、MLA、LFM等）对比，MaxCR在所有数据集均实现了领先或竞争性的准确率、MAP和MacroF1，并且在置信度校准指标（ECE、NLL、Brier）上也表现更好。

**⚠️ 局限性**

局限性在于方法仅针对多模态分类任务，扩展到其他多模态任务需设计任务特定的置信度度量与正则化；理论分析仍是简化模型，缺乏对置信度演化更普适的阐述。

---

## 600. Capacity Analysis and Joint Gaussian Beam Pattern Optimization for Positioning-Assisted Communications

**arXiv ID:** 2609.28196 | [PDF](https://arxiv.org/pdf/2609.28196v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 601. Exact Feedback Is Not Control: Evaluating Text-based Closed-Loop Revision in LLMs

**arXiv ID:** 2609.28150 | [PDF](https://arxiv.org/pdf/2609.28150v1)

**作者:** Haitong Jiang `[一作]` (Shenzhen University), Yuhong Feng `[通讯]` (Shenzhen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套固定预算、可审计的闭环修订评估框架，并在三类可检验约束（exact‑length、lexical、compositional）下对19个大型语言模型进行系统实验。

**💡 创新点**

通过将反馈的正确性和完整性固定以隔离模型侧闭环修订问题，提出精确反馈与可重复的修订响应曲线，首次对重复输出与历史干预进行轨迹级分析，显著揭示模型在闭环修订中的差异与局限。

**🔧 技术方法**

采用可确定的验证器报告全部违规；固定预算的请求‑模型‑验证器循环；统计方法（bootstrap、McNemar、逻辑回归）以及对模型的单步反馈‑动作响应曲线估计。

**📊 数据集**

使用人类编写的满足约束的参考文本生成的 exact‑length、lexical 与 compositional 案例（共480个案例/约束族），并结合公开的 COLLIE 数据集进行评估。

**📈 对比分析**

对19个模型（12 本地 checkpoint + 7 API）进行并行评估，计算累计联合成功率、每轮改进曲线；实验显示模型间成功率从 17.4% 到 99.8% 不等，跨模型和跨约束差异显著，即使在同一初稿下差距仍持续。

**⚠️ 局限性**

实验仅覆盖文本固定约束，缺乏主观目标、动态环境和不完整反馈；精确反馈虽可观测但不保证闭环可靠，且未探讨模型内部机制或更复杂任务的修订性能。

---

## 602. Diff-RF: Mutually Reinforced Image Registration and Fusion via Degradation-Aware Learning

**arXiv ID:** 2609.28235 | [PDF](https://arxiv.org/pdf/2609.28235v1)

**作者:** Xunpeng Yi `[一作]` (Wuhan University), Jiayi Ma `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了Diff‑RF框架，通过在复杂退化场景下进行内部模态恢复、交叉模态扩散式配准与融合，实现了互相促进的多模态图像配准与融合。

**💡 创新点**

创新点在于将退化感知恢复与配准-融合耦合为统一的扩散式循环，利用融合视觉提示与几何对应条件共同引导配准，并通过LoRA实现对预训练恢复网络的轻量级、配准友好微调。

**🔧 技术方法**

使用的技术包括：低秩适配（LoRA）对恢复模块的调优、扩散模型用于条件配准、跨模态相关性计算、融合网络与配准特征提取器的双向互补。

**📊 数据集**

实验数据集涵盖了MFNet、FMB以及真实环境下的LLVIP，后者用以评估泛化能力，并在每个数据集上引入多种退化（暗光、雾、条纹噪声、雨滴、随机噪声）。

**📈 对比分析**

与GLUNet、DiffMatch、CRFT、UMF‑CMGR、MURF、C2RF、AUNet、PGMR、CPMFusion等多种最先进配准/融合方法以及统一框架对比，Diff‑RF在EPE、1/3/5像素准度、MI、SD、VIF、Q^AB/F、SSIM等指标上均领先，且在语义分割与目标检测等下游任务中亦获得最高mIoU、mAP。

**⚠️ 局限性**

主要局限在于引入扩散模型导致的计算开销和推理时延（约0.65 s/帧），以及对极端失配或极低信噪比场景的鲁棒性仍有待进一步提升。

---

## 603. GUIAuditor: Enabling Post-hoc Child Safety Forensics via Action-Guided GUI Provenance on Mobile Devices

**arXiv ID:** 2609.28205 | [PDF](https://arxiv.org/pdf/2609.28205v1)

**作者:** Junlin Liu `[一作]` (Peking University), Yao Guo `[通讯]` (Peking University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于 GUI Provenance 的后验取证框架 GUIAuditor，帮助家长通过自然语言查询来复盘儿童在手机上的交互风险；

**💡 创新点**

创新点包括：① 事件驱动的多模态证据蒸馏管道（动作引导采样、像素去重、语义压缩）；② 以多模态大语言模型为核心的透明 Chain‑of‑Thought 推理引擎；③ 统一的混合存储与两阶段取证查询引擎；④ 在资源受限的移动端实现完整端到端的本地推理；

**🔧 技术方法**

使用了多模态大语言模型 Qwen2.5‑VL‑3B（4‑bit量化），视觉Transformer 进行特征提取，向量检索与 SQL 结合的混合数据库；

**📊 数据集**

基准数据集由 32 名参与者（12‑18 岁儿童与 18‑30 岁成人代理）录制的 295 条真实交互视频构成，覆盖内容、接触、金融、隐私四类风险与无风险；

**📈 对比分析**

与基线（连续采样、周期快照、OCR 等）比较，GUIAuditor 在 RQ1 取得 95.23% Macro‑F1、88.47% 事实性描述准确率；在 RQ2 事件检索 Recall@1 90.20%、MRR 0.923；问答准确率 88.24%；在 RQ3 上设备功耗约 +2.1 W、每次推理 7.4 s、峰值内存 3.1 GB，显著低于基线；

**⚠️ 局限性**

局限性包括：① 对 MLLM 的依赖，仍易出现幻觉或文化歧义；② 仅覆盖基于交互的风险，无法捕获纯粹被动风险；③ 数据集规模与多样性有限；④ 需要一定的设备算力与电量支持，且在高频交互时仍可能积压推理任务。

---

## 604. Geospatial embeddings detect old-growth forests but buffered spatial validation narrows their advantage over Sentinel features

**arXiv ID:** 2609.28194 | [PDF](https://arxiv.org/pdf/2609.28194v1)

**作者:** Thomas Ratsakatika `[一作]` (University of Cambridge), Emily R. Lines `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在罗马尼亚南部喀尔巴阡山脉范围内，利用遥感与地理空间基础模型对老生林进行检测并生成高分辨率预测图。

**💡 创新点**

首次评估GFM嵌入在老生林检测中的有效性，并引入缓冲空间验证以消除空间自相关。

**🔧 技术方法**

采用XGBoost、CNN与AlphaEarth、TESSERA嵌入，结合传统Sentinel‑1/2、Landsat EO以及地形与交通接近度特征。

**📊 数据集**

使用森林地块矢量、古林/准古林标注、道路/步道、欧洲森林干扰数据、ESA WorldCover、CLMS VPP等公开卫星与矢量数据。

**📈 对比分析**

通过六折空间阻塞交叉验证与10 km缓冲区比较，未缓冲时TESSERA优于传统EO，但缓冲后差异不显著；最佳模型XGBoost+TESSERA在未缓冲时PR‑AUC≈0.84，缓冲后降至≈0.73。

**⚠️ 局限性**

主要局限在于参考标签样本有限且缺乏跨地区多样性，且未在未观测区域进行实地验证，导致模型迁移性不确定。

---

## 605. Depth-Guided Contrastive Learning for 2D Representations with 3D Spatial Awareness

**arXiv ID:** 2609.28159 | [PDF](https://arxiv.org/pdf/2609.28159v1)

**作者:** Liang Zeng `[一作]` (KU Leuven), Maarten Vergauwen `[通讯]` (KU Leuven)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一种辅助目标——Depth‑Guided Contrastive Learning（DGCL），通过相对 3D 距离比较把深度信息注入 2D 对比学习，以提升像素级表征的几何一致性。

**💡 创新点**

创新点在于：①不依赖绝对深度值，而是利用随机采样的像素组中相对 3D 接近度来确定正样本；②将该几何监督与现有的对比框架（如 MoCo、SlotCon、ViT‑MoCo）无缝结合；③通过仅需 RGB‑depth 对的方式，既保持了训练简洁，又在多种模型和数据集上实现了显著提升。

**🔧 技术方法**

核心技术包括：对深度做简化的投影回 3D、信息噪声极大化（InfoNCE）对比、随机 anchor 采样、温度控制、以及与基准对比损失的线性加权组合。

**📊 数据集**

实验数据集涵盖 ImageNet、COCO（train2017）、Pascal VOC、Cityscapes、ADE20K，深度信息来自 MiDaS 估计器（Small_256、Large_384 等）以及可测量的 RGB‑Depth 传感器。

**📈 对比分析**

与 MoCo‑v2/v3、SlotCon 以及其他基于深度的对比方法（depth‑regression、CMC、mask‑guidance）相比，DGCL 在 COCO 检测、实例分割、关键点检测以及语义分割上平均提升 0.5–2.0 AP 或 1–3 mIoU；在 200 轮预训练下，DGCL‑S 的性能已接近或超过 1000 轮的 DetCon；在低质量深度下也能保持显著优势。

**⚠️ 局限性**

局限性：①需要可获得深度图，若深度估计误差大或缺失会影响效果；②仅基于像素级相对 3D 信息，难以捕捉更大尺度的全局 3D 关系；③在极端场景（极远背景或遮挡）下，相对距离可能不再准确，需进一步改进。

---

## 606. RL Starts before RL: On Policy Distillation for Better Reinforcement Learning

**arXiv ID:** 2609.28145 | [PDF](https://arxiv.org/pdf/2609.28145v1)

**作者:** Shuai Dong `[一作]` (Fudan University), Jiaqi Wang `[通讯]` (JD.COM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究在强化学习（RL）之前使用on‑policy distillation（OPD）来准备语言模型，并与直接RL和SFT+RL做比较，发现OPD能显著提升最终RL性能

**💡 创新点**

证明即使OPD对初始准确率影响不大，仍能通过教师分布对齐和保持不确定性来为后续RL提供更有利的起点，并揭示KL目标与轨迹来源对RL效果的交互影响

**🔧 技术方法**

采用反向KL（RKL）和正向KL（FKL）两种蒸馏损失，基于学生生成或教师生成的轨迹进行OPD；随后使用GRPO‑style RL训练；评估Pass@k、JS散度、熵等行为指标

**📊 数据集**

文本任务使用Qwen3‑8B学生与Qwen3‑32B教师（8个基准），视觉‑语言任务使用Qwen3‑VL‑8B/32B学生与教师（20个基准）

**📈 对比分析**

在相同RL设置下，OPD+RL的最终平均得分分别比直接RL高7.28分（文本）和1.34分（视觉‑语言），且在多数基准上超越SFT+RL；FKL在学生轨迹下在RL后优于RKL，RKL在教师轨迹下保持优势

**⚠️ 局限性**

未能独立检验教师对齐与不确定性对RL收益的具体贡献，且仅评估现有蒸馏目标，缺乏针对后续RL优化的专门蒸馏设计

---

## 607. Generalizable Robotic Insertion with World Models

**arXiv ID:** 2609.28258 | [PDF](https://arxiv.org/pdf/2609.28258v1)

**作者:** Nicklas Hansen `[一作]`, Yashraj Narang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种基于手腕相机原始深度图像和机器人本体信息的视觉模型驱动RL框架InsertionWM，能够在未见对象上实现零样本插入。

**💡 创新点**

创新点在于训练一个多任务世界模型，不依赖CAD或对象ID即可处理多种插装件，显著提升了可扩展性和零样本性能。

**🔧 技术方法**

使用了TD-MPC2模型预测控制、MPPI规划、深度卷积+MLP编码器、可学习的世界模型以及动态Curriculum学习。

**📊 数据集**

基于AutoMate提供的100对几何多样化插装件数据集，训练90个任务用于通用模型，测试10个未见任务。

**📈 对比分析**

与AutoMate的基准对比，单任务专用策略在90%训练后实现约79.8%成功率，通用策略在90个训练任务上零样本成功率达56%，远高于AutoMate的15.6%/7%；微调后性能进一步提升。

**⚠️ 局限性**

局限性包括对视觉遮挡敏感、缺乏大规模插装件数据集、实机部署受仿真-实物差异和深度传感噪声影响。

---

## 608. hyperbolix: Hyperbolic Deep Learning in JAX

**arXiv ID:** 2609.28248 | [PDF](https://arxiv.org/pdf/2609.28248v1)

**作者:** Timo Klein `[一作]` (University of Vienna), Sebastian Tschiatschek `[通讯]` (University of Vienna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 hyperbolix，一个针对 JAX/Flax NNX 的开源超曲率深度学习库，提供六种流形（欧氏、Poincaré 球、双曲抛物、κ-视角、混合曲率、速度空间）及其统一接口；实现了线性、卷积、注意力、归一化、位置编码、回归、向量量化等 44 种层，配套 Riemannian Adam/SGD 优化器、包装分布和降维方法。

**💡 创新点**

创新点在于：①首次在 JAX 生态中提供完整、通用的超曲率库；②所有流形操作均为纯函数，支持 JAX 转换；③通过学习可调曲率实现可跨层变曲率；④采用“无差分”公式重写双曲球面距离等运算，显著提升 float32 计算精度；⑤构建了独立“oracle”测试套件，确保每个运算与手工推导或数值差分一致，避免自洽错误。

**🔧 技术方法**

技术方面使用 JAX + Flax NNX 编写模块，利用 optax 的 transform 机制实现 Riemannian 优化；实现了基于 Minkowski 内积的无差分计算、κ-视角的 Taylor 展开、流形间的精确等距；采用 vmap + jax.lax.reduce_sum 等 JAX 原语实现批量化与算子融合；使用 Python 标准 unittest 以及 NumPy/SciPy 作为测试 oracle。

**📊 数据集**

本工作未在公开数据集上进行训练实验，侧重于数值稳定性、性能基准和跨库对比；主要通过自定义测试网格与浮点精度评估，未给出具体数据集。

**📈 对比分析**

与 geoopt、HypLL、GeoJAX、Rieoptax 等现有 JAX/PyTorch 库对比：hyperbolix 在层实现数量、支持的流形种类、曲率学习、精度（尤其是双曲球面距离）和批量化性能上均优于或相当；实验显示在 float32 下，距离、logmap‑expmap 的误差在 8× 以内；性能测试表明在大批量下与 geoopt 差距可忽略，HypLL 在其实现的 Poincaré 球面上最快。

**⚠️ 局限性**

局限性包括：float32 下的半径上限（Poincaré 球面约 14–24，双曲抛物约 9–19），无法进一步突破；不同流形实现不均衡（如注意力仅在双曲抛物上，向量量化仅在 Poincaré 球面上）；缺乏真实任务的训练质量基准；仅在单一种子下评估速度，未考虑多样本或多机情况。

---

## 609. Finite-Sample Probabilistic Safety Certification for AI-Based Grid-Edge Coordination

**arXiv ID:** 2609.28182 | [PDF](https://arxiv.org/pdf/2609.28182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 610. Controlling Collectives of AI Agents in Reasoning Space with Spatial Transformers

**arXiv ID:** 2609.28247 | [PDF](https://arxiv.org/pdf/2609.28247v1)

**作者:** Frederic Vatnsdal `[一作]` (University of Pennsylvania), Alejandro Ribeiro `[通讯]` (University of Pennsylvania)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了COMPASS，一种可在千级规模机器人集群中使用语言模型进行去中心化规划和控制的架构。

**💡 创新点**

创新点在于将语言模型与多机空间变换器(MAST)耦合，形成在推理空间的反馈控制循环，并利用结构化多样化命令消除偏差。

**🔧 技术方法**

技术上使用了冻结的Qwen3语言模型、空间注意力变换器、局部通信图、学习的反馈与目标标记，以及行为克隆+DAgger训练。

**📊 数据集**

数据集为七类操作指令模板生成的思考轨迹和中心化专家（潜能场）收集的N=32/64的物理轨迹。

**📈 对比分析**

在与集中式LLM、无通信、语言仅通信等对照实验中，COMPASS在1024机器人规模下的航向误差仅为0.34°，速度一致且保持紧凑，结构化多样化进一步提升准确率。

**⚠️ 局限性**

局限性包括对预训练LM的依赖、在极大规模下仍需足够通信半径、对极其模糊命令的鲁棒性有限，以及模型规模对延迟和算力的制约。

---

## 611. Support-Compiled Feature Folding: More Evidence at Lower Memory Across Tabular Foundation Models

**arXiv ID:** 2609.28208 | [PDF](https://arxiv.org/pdf/2609.28208v1)

**作者:** Tian Zhou `[一作]` (Ant Group), Liang Sun `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了无训练的推理框架 Support-Compiled Feature Folding（SCFF），通过分块编码和支持检查在不改变冻结模型的前提下，将宽表特征交互从二次复杂度降为线性。

**💡 创新点**

创新点在于：① 用支持相关的分块（Core+Tail）在推理时保持特征宽度的有限竞争；② 引入支持子集检验与 Fisher 步骤在尾部恢复非冗余信息；③ 通过仅一次 context‑encoder 与 label‑head 的调用，实现训练‑free 的宽表推理，并显著节省显存。

**🔧 技术方法**

使用的技术包括：一阶 ANOVA 与 Benjamini–Hochberg 统计检验挑选 Core；分块编码（leaf width ≤128）；支持子集检验、原型投影、Fisher 方向更新；原始模型的 feature‑encoder、context‑encoder、label‑head 复用。

**📊 数据集**

实验基于 18 个宽表数据集（来自 AMLB‑29、TabZilla、TabArena），特征维度范围 128–7200，涵盖多分类与二分类任务。

**📈 对比分析**

与原始全宽实现、Raw Core、Bounded Core、Tail Off/On 等基线进行对照，使用 dataset‑macro accuracy 与 NLL 评估。SCFF 在所有 6 个冻结骨干上均提升准确率（最高相对误差下降 26.1%）和降低 NLL（最高相对 NLL 降低 19.0%）。显存峰值降低 2–2.5 倍，峰值比值可达 5–34 倍，推理速度与原始相近或更快。

**⚠️ 局限性**

局限性包括：需模型支持行表示接口；叶子宽度需经验调节；Tail 更新仅在二分类任务下可用，多分类时回退至 Core；未在极宽（>7200）或更深层模型上验证；对某些模型的适配需要额外适配器。

---

## 612. Safety-Aware Zero Trust Enforcement for IoT and Cyber-Physical Systems

**arXiv ID:** 2609.28170 | [PDF](https://arxiv.org/pdf/2609.28170v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 613. Connectivity Preservation and Graph Stretching in Range-Only Swarm Dispersion

**arXiv ID:** 2609.28190 | [PDF](https://arxiv.org/pdf/2609.28190v1)

**作者:** Ariel Barel `[一作]` `[通讯]` (Technion Israel Institute of Technology), Ariel Barel (Technion Israel Institute of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在仅能测量相邻距离、无标识、方向、记忆或通信的极端条件下，研究匿名、同质、无记忆机器人群的连通性保持扩散。提出仅依据最大邻距的最大安全等向跳跃规则，证明在所有可行方向下可保留所有可见边并维持连通；对两机器人系统进行随机漂移与收敛分析，并通过大规模蒙特卡洛实验验证多体拓扑依赖的扩展行为。

**💡 创新点**

创新点在于：① 在极简感知模型下推导出唯一可保证安全的最大等向跳跃半径，仅需最大邻距信息；② 证明该规则对所有可行方向都能保留所有可见边，从而保证连通性；③ 对两机器人系统给出精确的漂移、收敛与首次到达时间分析；④ 通过大规模实验展示不同拓扑下的几何扩展差异，首次揭示匿名范围测量下连通性保持与几何伸展的关系。

**🔧 技术方法**

使用几何不等式推导安全半径，概率论（马尔可夫不等式、Borel‑Cantelli）分析随机跳跃的漂移与收敛，蒙特卡洛仿真与Bellman方程求解评估首次到达时间，离散化状态与角度空间构造有限状态机进行数值验证。

**📊 数据集**

自生成的可见图拓扑数据集：完整图、路径图、稀疏/中等/密集随机连通图（N=10、20、50）。每个拓扑类别均采样200–500次随机实例进行实验。

**📈 对比分析**

通过比较不同拓扑组的最终直径与固定生成树的拉伸比例评估性能。实验表明完整图始终保持直径V，路径图与稀疏图可扩展至数倍V；二机器人模型的平均首次到达0.97V的时间约为9.5步，理论上限为约2222步，验证了理论与模拟的一致性。

**⚠️ 局限性**

局限性：假设距离测量精确、同步跳跃、无碰撞和无噪声；未证明多机器人系统的收敛或终态；保持所有可见边过于保守，未能在匿名范围测量下安全释放冗余约束；缺乏对异步、动力学约束和测距误差的鲁棒性分析。

---

## 614. GLASS: Architecture-Tuned, Composable, Device-Side Linear Algebra for Edge Robotics and Beyond

**arXiv ID:** 2609.28179 | [PDF](https://arxiv.org/pdf/2609.28179v1)

**作者:** Brian Plancher `[一作]` `[通讯]` (Dartmouth College), Brian Plancher (Dartmouth College)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出并实现了 GLASS，一款头文件式 CUDA C++ 库，提供可组合的线程/warp/块级线性代数与几何运算，并通过离线测量在编译时静态决定最佳执行定位，旨在替代现有 GPU 机器人数值代码的碎片化实现。

**💡 创新点**

创新点在于：① 将多种执行范围与 NVIDIA 设备库统一到单一 API；② 采用离线性能测量生成硬件特定表格，在编译时决定 dispatch，消除运行时开销；③ 提供独立数值或acles与 GPU‑测试签名工具，实现跨实现的正确性验证；④ 通过可组合原语实现机器人特定运算与融合，显著提升边缘 GPU 机器人性能。

**🔧 技术方法**

使用技术包括 CUDA C++、线程/warp/块并行实现、CUB、CUTLASS、cuBLASDx/cuSOLVERDx、离线测量与表格生成、GPU 测试签名工具、机器人几何与线性代数原语、Python/C++ CI 工具链。

**📊 数据集**

实验数据来自 Jetson AGX Orin（sm_87）、Jetson AGX Xavier（sm_72）和 RTX 5090（sm_120）三种 GPU；批量大小 B 取 1–8192，问题规模 N 取 4–128；此外将 GLASS 集成至公开的采样 MPC 与批量 IK 求解器进行实测。

**📈 对比分析**

通过离线测量表选择最佳实现，并与 cuBLAS/cuSOLVER、PyTorch/JAX（最优配置）、Eigen、Kokkos Kernels 等库做对比；结果显示：在小矩阵、单任务或轻量级批量场景下，GLASS 最高可比 cuBLAS 超 113×、比 PyTorch/JAX 超 73×；在 RTX 5090 上仍保持 1.5–2.6× 的加速；边缘设备平均提升 1.5×，且在两大公开机器人系统中实现 1–1.5× 的速度和 92% 的代码量减少。

**⚠️ 局限性**

局限性包括：仅支持 NVIDIA GPU 与 CUDA 11.4/13.2；跨厂商 GPU 兼容性未完成；只覆盖单块（单问题）规模的运算，无法处理跨块大矩阵；缺乏混合/低精度支持与完整因子分解覆盖；需手动生成硬件测量表，CI 仍依赖本地 GPU；未来工作需扩展至非 NVIDIA 加速器与更广泛的算子集。

---

## 615. Flamingo: On Load Balancing in DAG-based Consensus Protocols

**arXiv ID:** 2609.28361 | [PDF](https://arxiv.org/pdf/2609.28361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 616. LightMIS: Ultra-Lightweight Medical Image Segmentation Without a Stage-Wise Decoder

**arXiv ID:** 2609.28327 | [PDF](https://arxiv.org/pdf/2609.28327v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 617. Dynamic, Decentralized Spatial Code Reuse for OCDMA LiDAR in Robot Swarms

**arXiv ID:** 2609.28172 | [PDF](https://arxiv.org/pdf/2609.28172v1)

**作者:** Mohammad Hani Alomari `[一作]` `[通讯]` (German Jordanian University), Mohammad Hani Alomari (German Jordanian University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种去中心化的协议，使得装备LiDAR的机器人群体能够动态地重新分配空间重用代码，以减少相互干扰。

**💡 创新点**

创新点在于通过实时维护的干扰邻域图动态重新分配代码，显著降低了所需代码数量的增长率，从Θ(N)降低到O(log N/loglog N)。

**🔧 技术方法**

使用了去中心化的协议和蒙特卡洛模拟技术来验证理论结果，并考虑了机器人移动性和不完美的信标检测。

**📊 数据集**

使用了模拟环境中的机器人群体数据，具体为20m x 10m的仓库布局，模拟了不同数量的机器人（15, 30, 60, 120）在该环境中的行为。

**📈 对比分析**

与现有的无协调方法进行了公平的比较，结果显示该协议在相同资源预算下实现了更高的代码重用效率和30-40%的更低碰撞风险。

**⚠️ 局限性**

局限性包括假设干扰检查为视距模型，未考虑更复杂的环境因素；只分析了机器人密度恒定的情况，未涵盖密度增加的情况；所有结果均基于模拟，缺乏物理硬件验证。

---

## 618. Learning the Cost of Reliable Inference

**arXiv ID:** 2609.28322 | [PDF](https://arxiv.org/pdf/2609.28322v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 619. Dissecting Advantage-Guided Post-Training for Vision-Language-Action Policies

**arXiv ID:** 2609.28161 | [PDF](https://arxiv.org/pdf/2609.28161v1)

**作者:** Jiahang Cao `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了优势引导的视觉-语言-动作(VLA)策略的后训练，拆分为优势构造、校准、利用三阶段，并在四个真实双臂操作任务上验证。

**💡 创新点**

创新点在于：①采用分阶段离线诊断筛选优势构造和校准，显著降低真实机器人评估成本；②提出基于价值的分组校准；③连续优势加权的利用策略在多任务中获得最佳性能。

**🔧 技术方法**

使用了离线强化学习技术，包括IQL、SARSA、ALOE等价值学习；n步TD优势估计；分组标准化校准；连续优势加权与过滤等利用策略；以及π_0.5流匹配动作生成网络。

**📊 数据集**

使用了由约40小时专家演示、100条自主滚动以及100条人类接管的混合数据集，覆盖四个真实双臂任务：轻松/困难手机包装、盒子装填、打印机纸张补充。

**📈 对比分析**

通过离线诊断指标（人类介入对照与进度对照）与真实机器人任务进度/成功率对比，连续优势加权在平均进度+0.42、成功率+0.63（相较SFT）优于DAgger、Filter等。

**⚠️ 局限性**

限制在于：需要先前收集多源异质数据；离线诊断对复杂任务的泛化仍有限；方法对奖励稀疏性和长周期任务的收敛速度敏感。

---

## 620. Pimp my fixpoint: sofic realization of multidimensional substitution-based shift spaces

**arXiv ID:** 2609.28207 | [PDF](https://arxiv.org/pdf/2609.28207v1)

**作者:** Antonin Callard `[一作]` (ENS de Lyon), Pascal Vanier `[通讯]` (Université Caen Normandie)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6215c339-3735-4be3-8a07-5bbb7004712d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并证明了一个通用的“固定点”定理，将符号动力学中基于宏块的自相似构造抽象为一系列可计算的替代（substitution）和 ndill maps，给出了多维 sofic shift 的正向可识别条件，并通过该定理实现了大量已知和新颖的 sofic shift 的构造。

**💡 创新点**

1) 把固定点构造从宏块层次切换到可计算替代/ndill maps 的框架；2) 允许宏块尺寸非单调甚至常数大小，突破传统必须指数增长的限制；3) 使用 log‑RAM 计算模型给出明确的时间/单词长度上界；4) 统一并扩展了现有的多维自相似、Toeplitz、低密度等 sofic 例子；5) 通过并行处理器数组模型实现空间‑时间的高效映射。

**🔧 技术方法**

符号动力学（shift、SFT、sofic shift）、子替代与 S‑adic/N‑adic 结构、ndill maps、固定点构造、并行处理器数组、log‑RAM 计算模型、可计算性与递归理论、信息量与Kolmogorov复杂度、熵与频率分析、拓扑与可测量动力学工具。

**📊 数据集**

该工作属于纯理论研究，未使用任何实测数据集；所有构造与证明均基于符号演算与算法复杂度的抽象分析。

**📈 对比分析**

与以往方法对比，本文的定理提供了统一的构造框架，可直接构造满足所给计算约束的 SFT 覆盖；在时间复杂度方面给出 log‑RAM 级别的显式上界，且在宏块尺寸与信息量上给出可控的子线性/多项式上界；在熵、频率、熵维度等量化特征上与已知例子保持一致或提供更细致的控制。

**⚠️ 局限性**

1) 需要宏块扩展性和可计算性约束，不能处理所有有效 shift；2) 对信息量上界的 δ < 1/2、α < 1 等参数设定存在技术性限制；3) 对并行 ndill maps 的要求较高，非平行映射需要额外的尺寸限制；4) 结果对宏块尺寸变化的灵活性有限，极端非单调增长情形可能无法直接满足；5) 在某些高度复杂的例子（如非可识别的置换）中，构造难以直接应用。

---

## 621. When and Where to Trust the Teacher: Unifying On-Policy Distillation and GRPO through Entropy-Calibrated Credit Assignment

**arXiv ID:** 2609.28385 | [PDF](https://arxiv.org/pdf/2609.28385v1)

**作者:** Jie Zhang `[一作]` (Shanghai Jiao Tong University), Xiaolin Huang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 UECR-GRPO 方法，将 verifier 与 teacher 信号统一于轨迹级奖励，并通过熵校准和零和投影实现 token 级信用重分配，从而提升语言模型在数学推理任务上的准确率。

**💡 创新点**

创新点在于：① Path-Utility Unification 将 verifier 奖励与教师 log‑ratio 通过单一 KL 正则化目标统一，允许教师信号在轨迹级比较中影响响应排序；② Entropy‑Calibrated Redistribution 采用教师熵校准的方向性信号，并通过零和投影在 token 级分配 verifier 产生的信用，既保留总信用，又实现局部信用定位。

**🔧 技术方法**

使用技术包括 GRPO 与 PPO 的剪辑策略、KL 正则化控制、教师模型对齐、教师熵校准、零和投影以及多响应组内归一化。

**📊 数据集**

实验数据集包括 DeepMath‑103K（训练）、AIME 2024/25、AMC 2023、HMMT 2025（Feb/Nov）等数学推理基准。

**📈 对比分析**

通过与 Vanilla GRPO、PG‑OPD、Distilled RL、ATOD 等基线比较，1.7B 学生平均准确率提升至 17.21%（比最强基线高 0.89%），4B 学生平均准确率提升至 65.09%（比最强基线高 0.56%）。

**⚠️ 局限性**

局限性包括：改进主要体现在整体平均提升，单个任务提升有限；实验仅覆盖 Qwen3 系列模型，缺乏对更大模型和其他任务的验证；对教师不确定性与推理步骤的理论解释仍有待深入。

---

## 622. Mizar: A 159M-Parameter Audio-Language Model for Audio Understanding

**arXiv ID:** 2609.28344 | [PDF](https://arxiv.org/pdf/2609.28344v1)

**作者:** Kaiyang Li `[一作]` (NEC Laboratories America, Inc), Shihao Ji `[通讯]` (University of Connecticut)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一个159.3M参数的音频‑语言模型Mizar，采用Compact CED‑Small音频编码器、频率合并映射器与SmolLM2‑135M语言解码器，并通过三阶段训练实现音频与语言的对齐、音频依赖微调及后期答案位置平衡，最终支持单CPU本地推理。

**💡 创新点**

创新点在于①引入频率合并映射器将CED‑Small的四频特征有效映射到语言解码器空间；②融合ReasonAQA、AudioMCQ（含150k链式思维样本）与AVQA多任务监督，扩大监督多样性；③采用三阶段训练，其中第二阶段音频依赖微调与第三阶段教师辅助四象限采样、答案位置平衡相结合，显著提升弱技能与答案分布均衡。

**🔧 技术方法**

使用了频率合并映射器、两层MLP投影、残差MLP、全局上下文平均、SmolLM2解码器、基于教师模型的四象限采样、答案位置平衡技术，以及GPU训练与FP32单CPU推理实现。

**📊 数据集**

主要数据集包括ReasonAQA、AudioMCQ（含CoT样本）、AVQA、strongAC子集用于微调，评测采用MMAU、MMAR、ADQA‑clean三大基准。

**📈 对比分析**

在MMAU、MMAR和ADQA‑clean三大基准上与200M以下的Mellow等模型对比，Mizar分别取得52.92%、42.42%和36.02%的平均准确率，均超越同规模模型；在单CPU推理上平均延迟1.09秒，峰值内存1.66GB，证明了其在低算力环境下的可行性。

**⚠️ 局限性**

局限性包括：音频缺失或替换时准确率显著下降，表明对音频依赖较强；模型规模虽小但仍不适合极低算力或极限存储设备；缺乏对视觉或多模态任务的支持；后期微调需要多阶段训练，调参成本较高。

---

## 623. An Open Pipeline and Dashboard for Systemic-Risk Evidence under the EU AI Act's Code of Practice

**arXiv ID:** 2609.28335 | [PDF](https://arxiv.org/pdf/2609.28335v1)

**作者:** Jacob T. Emmerson `[一作]`, Zhijing Jin `[通讯]` (Max Planck Institute For Intelligent Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了 Systemic Risk Index——一个开放评估流水线和交互式仪表盘，将 19 个公共基准组织到四类系统性风险中，对模型在基线、干扰和代理场景下进行评估，并支持平均/最差聚合及能力加权。

**💡 创新点**

创新之处在于使非技术用户能够透明查看风险评估背后的基准证据、聚合方式和假设，并提供平均与最差聚合、能力权重等可视化交互；同时将公开基准映射至 EU GPAI 风险类别，并通过 LLM 自动评分与人工验证提升可复现性。

**🔧 技术方法**

采用 Python+Inspect 评估流水线，React 前端仪表盘；在评估流程中实现基线、四种语义保持干扰、生成代理情境；使用几何平均结合能力指数进行加权；对 LLM 评分与人工评审进行一致性检测。

**📊 数据集**

使用 19 个公开基准（共 1,188 条条目），覆盖 CBRN、网络攻击、损失控制和有害操纵四类风险；评估了 18 个模型；基准来自 MIT Risk Repository 等公开来源。

**📈 对比分析**

对 18 个模型进行平均和最差聚合评估；LLM 评分与人工评审达 κ=0.78–0.82 的一致性；随机抽样的 83% 干扰保持原始伤害；最差聚合下模型得分平均下降 14–37 分；用户研究显示大多数参与者能理解仪表盘输出。

**⚠️ 局限性**

局限性包括：基准覆盖不均、风险映射依赖专家判断、干扰与代理场景未完全覆盖真实部署、自动评分仍不完美、评估结果随时间变化、仪表盘仅汇总现有证据，无法完整衡量系统性风险或法律合规。

---

## 624. Multimodal Voice Activity Projection for Social Robot Mediation: Expected Behavior and Deployment Constraints

**arXiv ID:** 2609.28317 | [PDF](https://arxiv.org/pdf/2609.28317v1)

**作者:** Antonio Cano `[一作]` (4i Intelligent Insights), Randy Gomez `[通讯]` (Honda Research Institute Japan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出一种多模态语音活动投影模型 MM‑VAP，用于社交机器人在调解人机交互中的交谈占有权预测与行为决策。

**💡 创新点**

创新点在于将同步的音频‑视觉信号与预训练的 VA 相关编码器结合，并通过 LoRA 轻量化迁移、交互式跨说话人注意力以及零样本事件推理，首次实现了对对话占有权未来状态的全景预测。

**🔧 技术方法**

技术包括 WhisperFlamingo / TalkNet 预训练音视频编码器、LoRA 参数微调、交叉说话人注意力模块、两层预测头（多类别 VAP 与二元 VA）以及基于 VAP 分布的零样本事件推断。

**📊 数据集**

使用的数据集有多语种双人对话集 NoXi 与 NoXi+J（含中文、日语）以及机器人调解场景 Haru EDR。

**📈 对比分析**

与基线 CPC+3DResNet 对比，MM‑VAP 在 NoXi 上实现 0.97 F1（S/H 与 S‑pred），在 Haru EDR 上达到 0.92 Hold、0.85 Shift、0.91 S‑pred，明显优于传统基线且在多语言情境下保持高精度。

**⚠️ 局限性**

主要限制包括实时推理的计算与同步瓶颈、对完整音视频输入的依赖、以及在真实物理机器人部署中的延迟与鲁棒性待验证。

---

## 625. PBLH Estimation from Satellite Radiances via a Dual-Encoder Transformer

**arXiv ID:** 2609.28286 | [PDF](https://arxiv.org/pdf/2609.28286v1)

**作者:** Lorenzo Innocenti `[一作]` (Politecnico di Torino), Paolo Garza `[通讯]` (Politecnico di Torino)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于双编码器Transformer的卫星辐射回归模型，用于从MetOp卫星的红外和微波辐射中估算行星边界层高度 (PBLH)。

**💡 创新点**

创新点在于：① 将多模态（IR 与 MW）输入分别通过并行Vision Transformer编码器处理并在隐藏层相加融合；② 采用动态遮掩机制直接在训练时模拟观测缺失，模型可在任意天气条件下无须插值；③ 通过分块Shapley值定量评估各输入块对精度的贡献。

**🔧 技术方法**

使用了深度学习技术，主要包括多层感知机 (MLP)、一维 UNet 与 Transformer、二维 ResNet 与 Vision Transformer，核心模型为双编码器 Transformer 2D；训练时使用 AdamW、余弦学习率衰减；采用分块Shapley值进行特征重要性分析。

**📊 数据集**

使用了在 2022 年收集的 MetOp 衍射仪 IASI 与 AMSU-A/MHS 与 ERA5 重新分析数据配对的大规模数据集，包含 194 个轨道通道，4.3 万多样本；并额外评估 30 个 2025 年 TEAMx 观测期间的非分布样本。

**📈 对比分析**

与八种基线（像素级回归、1D 轨迹序列、2D 卷积与 Transformer）进行对比，模型在全局测试集上的 MAE 为 155.8 m、RMSE 215.3 m、Pearson r = 0.850，优于所有基线；在 30 个外部样本上 MAE 165.3 m，仍显著优于像素级 MLP（197 m）。

**⚠️ 局限性**

主要局限：① 训练标签采用 ERA5 PBLH，无法直接与真实观测对齐，可能引入再分析误差；② 模型仅利用单一轨道通道的空间上下文，未利用跨轨道时间连续性；③ 未给出不确定性估计，缺乏对预测置信度的量化。

---

## 626. Non-Commutative State Tracking with Input-Dependent Low-Rank Updates in Mamba-3

**arXiv ID:** 2609.28273 | [PDF](https://arxiv.org/pdf/2609.28273v1)

**作者:** Hiroki Fujii `[一作]` (Institute of Science Tokyo), Masaki Yamakita `[通讯]` (Institute of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在Mamba-3的基线上加入输入相关的低秩反射更新，实现了单块非交换状态追踪。

**💡 创新点**

创新点在于仅使用一个秩-1的反射项即可实现非对角、非交换状态更新，同时保持原有的指数梯形离散化、RoPE和复数状态动态，显著提升在非交换任务中的表现。

**🔧 技术方法**

采用低秩反射更新（NPLR）、块级并行计算、RoPE、指数梯形离散化以及Mamba-3的复数状态机制。

**📊 数据集**

实验数据集包括 S5 词问题（对称群 S5 生成器）、Z5 算术任务以及连续观测的壳游戏（杯子交换可视化）。

**📈 对比分析**

与标准 Mamba-3、DeltaProduct、GDN、IDS4 等模型在固定时间、时间抖动和更长交换序列下进行直接训练与行为克隆对比；Mamba‑3+NPLR 在 S5 词问题与壳游戏中维持 90%+ 的准确率，尤其在时间抖动和超长序列时优于其它模型。

**⚠️ 局限性**

局限性：仅在离散输入/低维观察下验证，未测试图像或真实物理系统；未评估闭环交互对观测的影响；RoPE 在不同任务中的作用尚不确定；参数匹配与规模未统一。

---

## 627. Digital diglossia: Arabic between X and Facebook

**arXiv ID:** 2609.28352 | [PDF](https://arxiv.org/pdf/2609.28352v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 628. Predicting Quantization Price for Selecting PTQ Configurations Before Deployment

**arXiv ID:** 2609.28270 | [PDF](https://arxiv.org/pdf/2609.28270v1)

**作者:** Junbin Qiu `[一作]`, Yao Shu `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出一种在权重空间后训练量化（PTQ）中，基于前向KL散度衍生的层输出误差价格，对所有可部署的配置（比特数、粒度、量化器家族、预量化变换等）进行预部署的配置选择，并在预算约束下求解最优配置。

**💡 创新点**

创新点在于：①将前向KL的二阶项解释为“价格”，使得不同配置的误差可直接比较；②统一价格框架覆盖比特分配、变换、粒度等多种配置，使其成为同一搜索空间；③通过“等距迹简化”把价格进一步压缩为两个标量和一个预缓存曲率迹，实现在校准阶段即可评估所有候选方案。

**🔧 技术方法**

技术主要包括：前向KL展开、二阶曲率（Hessian）或Gauss‑Newton近似、候选配置的误差协方差估计、等距迹简化、基于预算的整数规划/搜索算法；实现时使用Hutchinson估计曲率迹，估计候选误差的尺度与方差。

**📊 数据集**

实验数据集：OPT‑125M、Qwen3‑0.6B、Llama‑3.2‑1B 等大语言模型的权重量化；校准集使用标准语言模型推理数据；任务评估包括 WikiText‑2 Perplexity 以及多任务下游准确率。

**📈 对比分析**

与现有方法（如 AMQ、HIGGS、AWQ、SmoothQuant 等）相比，本文在相同预算下的配置搜索成本显著降低（搜索时间减少 3–10 倍），并在 Perplexity 和下游任务准确率上取得最优或相近的表现；价格预测与实际KL漂移高度相关（ρ>0.92）。

**⚠️ 局限性**

局限性包括：①假设量化误差局部平滑且曲率近似准确，对高度非线性或跨层耦合影响较大的情况可能失效；②等距迹简化要求候选误差协方差近似等距，若变换或量化方式导致强非等距，则预测误差增大；③仅适用于权重空间 PTQ，未覆盖激活量化、KV‑cache 量化、训练时动态裁剪等场景。

---

## 629. Fine-Tuning LLMs for Translation: General Forgetting Mitigation Does Not Preserve MT-Specific Instruction Following

**arXiv ID:** 2609.28395 | [PDF](https://arxiv.org/pdf/2609.28395v1)

**作者:** Niklas Scholz `[一作]` (AppTek GmbH), Hermann Ney `[通讯]` (AppTek GmbH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对指令调优的大型语言模型在平行数据上进行微调时的灾难性遗忘问题进行实验研究，比较多种遗忘缓解方法并评估其对翻译质量、通用能力和特定翻译控制的影响。

**💡 创新点**

首次系统评估了基于辅助数据、模型输出和基模型参数的遗忘缓解策略在机器翻译与翻译指令遵循（formality、gender、length）任务上的表现，并揭示了通用基准保留不等同于翻译指令保持的现象。

**🔧 技术方法**

采用 Elastic Weight Consolidation (EWC)、数据混合、模型合并、KL 正则化、Selective Token Masking (STM)、Confident Conflict Masking (CC)、Entropy‑Adaptive Fine‑Tuning (EAFT)、Freeze、Selective Projection Decay (SPD)、LoRA 等技术；并在 Llama 3.2 1B 与 Llama 3.1 8B 上进行 SFT 微调。

**📊 数据集**

使用 Amharic–English、Arabic–English、Spanish–English 的平行语料（约100万条），以及 Tülu 3 SFT 组合、CoCoA‑MT、MT‑GenEval、FLORES、GSM8K、DROP、TruthfulQA、Codex HumanEval、IFEval、AGIEval、GPQA 等通用与控制翻译评测集。

**📈 对比分析**

在 Stage 1 对 1B 模型做多种方法的超参数搜索，Stage 2 对 8B 模型进行细调；结果显示 EWC 在保持通用能力方面最佳（general avg 下降 ≤ 1.7 点），但对 MT‑IF 控制几乎无改善；只有在训练数据中混入对应控制任务样本的方法才能保留该控制，且效果不迁移到其它控制或未见提示；SPD 在不混入控制数据时的 MT‑IF 保留最好，仍低于数据混合。

**⚠️ 局限性**

仅评估 SFT 方式，未涵盖基于 RL 或偏好优化的自举；使用的评测集有限（仅涵盖英语↔西班牙语、阿拉伯语、德语、Amharic），且模型规模与语言对变化耦合，难以分离影响；未覆盖性别中性翻译、词汇约束等控制；实验基于内部数据集，可能存在偏差。

---

## 630. A Gmail-Based Phishing Detection Prototype for Nigerian Fintech Emails Using Sender Checks and BiLSTM Classification

**arXiv ID:** 2609.28305 | [PDF](https://arxiv.org/pdf/2609.28305v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 631. Threat Amplified, Blame Restrained: LLM-Assisted Media Framing Analysis of the 2026 Bangladesh Measles Outbreak

**arXiv ID:** 2609.28362 | [PDF](https://arxiv.org/pdf/2609.28362v1)

**作者:** Shahan Ahmed `[一作]` `[通讯]` (Independent Researcher), Shahan Ahmed (Independent Researcher)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2026年孟加拉麻疹暴发的英语媒体报道进行情感与立场分析，构建可复制的语料并验证LLM标注质量。

**💡 创新点**

结合锁定代码本的LLM标注与人类金标准验证，实现低成本可复制的LMIC疫情媒体分析流程，并发现政治背景下媒体更关注系统失效而非个人指责。

**🔧 技术方法**

使用Anthropic Claude Haiku 4.5 LLM进行文本标注，采用Cochran–Armitage趋势检验、Cramér’s V等统计方法进行分析。

**📊 数据集**

基于403条来自七家孟加拉国英语报纸（如Daily Star、Dhaka Tribune等）的标题，来源于Internet Archive Wayback 捕获。

**📈 对比分析**

与两位人工编码者的金标准对比，LLM在立场上的κ=0.89、情感二元κ=0.75；情感与风险放大趋势显著（p<0.001）。

**⚠️ 局限性**

仅包含英语标题，未考虑全文细节；责备框架罕见导致统计功效低；缺乏多模型对比，并受部分June档案缺失影响。

---

## 632. Online Fair Division Against an Oblivious Adversary

**arXiv ID:** 2609.28333 | [PDF](https://arxiv.org/pdf/2609.28333v1)

**作者:** Saar Cohen `[一作]` (University of Oxford), Michael Wooldridge `[通讯]` (University of Oxford)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在面对盲目对手（input 在算法运行前已确定）时，随机在线分配不可分商品的公平性保证，并提出了 Deficit Allocation 算法及其改进版本。

**💡 创新点**

创新点在于：①利用“缺口”机制对每个代理的累计未满足分配进行跟踪，从而在不需要提前知道 δ 或 κ 的情况下，将 PROP1 的高概率近似从 O(1/ln(n/δ)) 提升到 Ω(1/lnln(n/δ))；②给出了 EF1、EFX、MMS 等公平性指标的极限下界，证明在盲目对手下几乎无法实现高概率公平；③证明允许删除对数数量的商品即可实现 (1−ε)-EFk，进一步揭示了 EFk 与 EF1 的区别。

**🔧 技术方法**

核心技术包括：双指数权重分配、缺口更新与保留、凸组合与大数定律、指数尾巴上界（对缺口的概率上界为双指数），以及对规模（dyadic 价值区间）的细化以减小常数因子。

**📊 数据集**

论文未使用真实数据集，而是通过构造性的理论输入（如单位商品、目标商品、随机块等）来证明上界和下界。

**📈 对比分析**

与传统的均匀随机分配相比，Deficit Allocation 在 PROP1 近似上实现了对数对数的提升；对 EF1、EFX、MMS 给出的下界表明在盲目对手模型下几乎没有实用的高概率公平算法；同时提出的 EFk 方案在允许删除 O(log(n/δ)) 个商品时即可达到近似 1 的公平度，显示了在特定公平度量上仍可取得可观效果。

**⚠️ 局限性**

主要限制包括：① PROP1 的高概率下界仍依赖于 loglog(n/δ)，未能实现常数级别的期望保证；② EF1、EFX、MMS 等指标在盲目对手下几乎不可实现高概率公平；③缺乏与期望前提公平（ex‑ante proportionality）同时满足的算法；④对算法的实际实现细节（如对数对数权重的计算）在极端规模下可能导致效率问题。

---

## 633. TANDEM: Task and Motion Planning with As-Needed Demonstrations for Efficient Vision-Language-Action Model Fine-tuning

**arXiv ID:** 2609.28314 | [PDF](https://arxiv.org/pdf/2609.28314v1)

**作者:** Samrat Sahoo `[一作]` (Stanford University), Yixuan Huang `[通讯]` (Princeton University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

结合任务与运动规划（TAMP）和基于视觉语言模型（VLM）的域扩展，自动识别任务中无法自主完成的步骤，将其包装为“魔法操作”，由人类远程操控完成，从而生成完整的演示序列。

**💡 创新点**

提出将人类干预视为TAMP的按需能力，通过VLM即时生成缺失的谓词和魔法操作，实现不需要手工指定干预点的自动化任务拆分；同时利用示例预训练轨迹与DATAFARM技术对规划生成的运动进行分布对齐，使演示更符合目标VLA模型的行为分布。

**🔧 技术方法**

核心技术包括：视觉语言模型（VLM）用于谓词发明与视觉验证；开放词汇TAMP（TiPToP）实现自主执行；DATAFARM对规划轨迹的分布匹配；人机交互接口实现远程操作；VLA模型（如LLaMA‑RL）的微调。

**📊 数据集**

使用五个长时程操纵任务（涵盖取放、打开盒子、覆盖、排序等）在真实机器人上收集数据；预训练数据来源于DROID数据集，并提供示例轨迹用于对齐；最终用于微调VLA模型。

**📈 对比分析**

与四个基线（纯TAMP、全手工遥控、预训练VLA、HITL‑TAMP）对比，评估指标为演示成功率、人类干预时长、下游VLA成功率与任务进展。结果显示，TANDEM在演示成功率上达76.9%，在相同人类时长下收集的演示数比全手工遥控高约2.9倍；微调后VLA平均成功率提升至60%，明显优于HITL‑TAMP（17%）且可直接在15 Hz下执行。

**⚠️ 局限性**

主要限制包括：依赖底层TAMP系统导致失败率偏高；魔法操作假设人类能无误完成指定效果，未考虑人类执行的可行性；VLM的谓词验证受单帧图像限制；目前仅在单一机器人平台与单一VLA模型上验证，泛化能力尚未评估。

---

## 634. Contraction and Statistical Inference under Privacy for Uniformly Bounded Distributions

**arXiv ID:** 2609.28297 | [PDF](https://arxiv.org/pdf/2609.28297v1)

**作者:** Leonhard Grosse `[一作]`, Mikael Skoglund `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文提出并研究了一种新的隐私机制——c‑interior PML（面向内部分布的概率最大似然隐私），给出了其严格的强数据处理不等式（SDPI）以及对各种发散度的精确下界，并在二元假设检验和均值估计这两个典型统计任务中演示了该机制在低隐私或已知分布内部信息约束下可实现“无隐私成本”的效果。

**💡 创新点**

创新点主要体现在：
① 将传统的局部差分隐私（LDP）框架推广到c‑interior分布集合，使得隐私约束不再对所有可能的输入分布均成立，而是只要求对内部分布满足；
② 推导出针对c‑interior PML的精确SDPI，并证明其在许多情况下优于LDP下的传统不等式；
③ 在二元假设检验与均值估计任务中展示，当ε≥-log(cμ_X(𝒳))时，样本复杂度与无隐私情况等价，从而证明了在低隐私/已知内部约束下隐私可“免费”实现；
④ 给出了从c‑interior PML到近似LDP（(α,δ)-LDP）的转换，并分析了该转换在某些通道下的优越性。

**🔧 技术方法**

主要技术手段包括：
- 强数据处理不等式的推导与分析；
- E_γ-发散度与Hellinger距离的结合使用；
- Le Cam与Assouad方法在统计任务中的应用；
- 对发散度的积分表示与非线性SDPI的构造；
- 通过构造可逆/不吸收的后处理通道证明隐私放大效应。

**📊 数据集**

本文属于理论研究性质，未使用具体实验数据集；所有结果均为数学证明和数值示例（如图示的SDPI上界）。

**📈 对比分析**

与传统LDP方法相比，c‑interior PML在低隐私（ε大于阈值）时可实现与非隐私相同的样本复杂度；在高隐私或分布不满足内部约束时，给出的上界更为严格；实验/数值示例表明在二维均值估计中，η_TV^2(ε)与传统LDP下的η_TV(ε)相比具有更好的收敛速度。

**⚠️ 局限性**

局限性：
① 需要对数据分布的内部约束（c‑interior）做出显式假设，实际数据可能不满足；
② 某些构造的机制在实现上可能不具备可扩展性（如高维情况下的离散化与投影）；
③ 证明中多次使用非构造性存在论与极值点假设，具体实现时需要额外设计；
④ 在高维大样本场景下，虽然给出了下界，但匹配的上界仍是上界常数，精细化仍有待进一步研究。

---

## 635. Talk2Escape: Conversational Grounding for Vision-and-Language Navigation

**arXiv ID:** 2609.28296 | [PDF](https://arxiv.org/pdf/2609.28296v1)

**作者:** Zerui Li `[一作]` (Adelaide University), Qi Wu `[通讯]` (Adelaide University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Talk2Escape 框架，将 Vision‑and‑Language Navigation 任务从单次开放循环转为主动对话式错误恢复。

**💡 创新点**

创新点在于使用零样本多模态 LLM 进行即时查询生成，并通过基于运动学的触发器实现主动求助。

**🔧 技术方法**

采用 Gemini 3.1 Pro 等多模态 LLM 作为视觉‑语言翻译器和提示生成器，并结合基于 kinematics 的触发机制。

**📊 数据集**

在 R2R‑CE、RxR‑CE、VLNVerse 等 Matterport3D 连续环境以及 Unitree Go2 实体机器人上进行实验。

**📈 对比分析**

与现有监督和零样本基线对比，Talk2Escape 在 R2R‑CE 上 SR 达 72%（超过 GTA 48.8% 及 Efficient‑VLN 64.2%），在 RxR‑CE 上提升约 16% SR；同时保持较高的 OSR、SPL 与 nDTW。

**⚠️ 局限性**

局限性包括对对话触发频率的敏感性、在路径效率指标（SPL、nDTW）上有一定牺牲，以及缺乏连续时空记忆导致的物理环境中更复杂干扰的处理不足。

---

## 636. Resource-Adaptive Stochastic Gradient Descent for Online Linear Programming without Re-solving

**arXiv ID:** 2609.28263 | [PDF](https://arxiv.org/pdf/2609.28263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 637. Computation Over Geometry: Meaning Identity Is Computed, Not Shipped in the Embeddings

**arXiv ID:** 2609.28290 | [PDF](https://arxiv.org/pdf/2609.28290v1)

**作者:** Jiaqi Deng `[一作]` `[通讯]` (Independent Researcher), Jiaqi Deng (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了在“意义同一性”任务（PAWS‑X式重写检测）中，独立句子嵌入是否携带同一性信息，并证明只有在同一前向传递（joint pass）时才可检索该信息。

**💡 创新点**

创新点在于将同一性从传统的“几何”视角转为“计算”视角，展示不同模型族（因果、双编码器、Encoder‑Decoder）在中层产生的统一可迁移的“同一性算子”，并证明该算子可以通过小型联合读者高效蒸馏而无需额外标注。

**🔧 技术方法**

采用的技术包括：多模型联合推理、四种读取方式（cosine、单独probe、late fusion、joint probe）、shuffle对照、深度层次分析、线性/非线性读取器、对比学习、蒸馏和跨语言/跨数据集迁移实验。

**📊 数据集**

使用的数据集包括：Overlap‑matched PAWS‑X（英中）、QQP、MRPC、SNLI、MS‑MARCO、Jina Reranker数据以及自构造的陷阱FAQ银行，用于检验检索与身份认证的实际效果。

**📈 对比分析**

实验结果表明：joint probe 在 1.5‑32B 规模模型上可达 0.90–0.96 AUC，而 late fusion 与单独 cosine 均停留在 0.47–0.61，shuffle 在 0.5 左右；蒸馏后的 1.5B 读者在 PAWS 上 0.959、QQP 0.774、中文 PAWS 0.758；在 FAQ 堵塞实验中，联合 probe 与蒸馏读者结合可将误答率从 28% 降至 10%。

**⚠️ 局限性**

局限性在于只针对 PAWS‑X 风格的“词汇重叠”重写对，未验证在更宽泛的同义句、跨语言多模态或长文本场景中的适用性；同时对模型内部结构的依赖仍未完全解剖，导致在极小或不具备中层算子计算能力的模型上效果有限。

---

## 638. Physalia: Redistribution-Resistant Content Protection for Decentralized Storage

**arXiv ID:** 2609.28277 | [PDF](https://arxiv.org/pdf/2609.28277v1)

**作者:** Giacomo Giuliari `[一作]` (Mysten Labs), Karl Wüst `[通讯]` (Mysten Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种端到端的访问控制系统，在分布式存储上将数据本身通过计算秘密共享进行分片，授权读者仅在获取足够多分片后才能恢复完整内容。

**💡 创新点**

创新点在于（1）将数据本身秘密共享而非仅共享密钥，显著提升重分发带宽；（2）设计了一种通用鲁棒性变换，使用临时签名实现任何计算秘密共享方案的鲁棒性，元数据仅为常数；（3）在Walrus+Sui上实现端到端，展示数十Gbps的编码/解码吞吐和可接受的延迟。

**🔧 技术方法**

技术手段包括计算秘密共享方案SSMS或AONT‑RS、Reed‑Solomon纠删码、临时Ed25519签名、ECIES混合加密、智能合约访问策略、Rust并行实现与Rayon、BLAKE3哈希。

**📊 数据集**

实验使用合成随机数据，大小为1 MB、10 MB、100 MB和1 GB，模拟不同 (n,k) 配置 (3,2)、(5,3)、(10,6)、(10,9)。

**📈 对比分析**

通过与传统加密+密钥共享、HK1/2（基于指纹的鲁棒方案）以及RAONT‑RS对比，测量编码/解码吞吐、鲁棒性开销；与Walrus直接存储基线比较读写延迟；结果显示在 (10,6) 配置下编码速率可达48–77 Gbps，重构速率相近，存储延迟略高但读取延迟可低于Walrus；鲁棒性变换的元数据常量，整体开销低。

**⚠️ 局限性**

局限性包括：仍需信任写入者；对服务器密钥泄露不考虑与读者共谋的情况；对 (n,k) 的选择和扩展性有限；链上事务成本和网络拥塞会影响延迟；未针对恶意读者与服务器完全协作的攻击场景给出完整安全证明。

---

## 639. Towards Efficient Reasoning: Learning Causal Shortcuts for Diffusion Language Models

**arXiv ID:** 2609.28272 | [PDF](https://arxiv.org/pdf/2609.28272v1)

**作者:** Dian Jin `[一作]` (Zhejiang University), Kun Kuang `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于因果快捷链的Diffusion Language Model训练框架CSL，利用条件互信息挑选关键token并在训练中采用并行优先掩码来提升推理性能。

**💡 创新点**

创新点在于：①定义条件互信息（CMI）度量token对后续推理的因果贡献；②通过分步token提取构建全序列的因果快捷链；③在训练中使用并行优先掩码聚焦学习这些快捷链，从而显著提升推理准确率和效率。

**🔧 技术方法**

使用技术包括：条件互信息评估、分步token提取、滑动窗口搜索、并行优先掩码、LoRA微调、NELBO优化等。

**📊 数据集**

使用的数据集：Math-CoT、OPC‑SFT‑Stage2、七个数学推理基准（GSM8K、MATH‑500、SAT、Sudoku、GPQA、MMLU‑STEM、ARC‑C）以及两项代码生成基准（HumanEval、MBPP）。

**📈 对比分析**

与多种SFT变体（DiBT、MGDM、Blockwise、DSFT、GIFT）以及基线模型对比，平均提升1.92%/1.58%，在MATH‑500上最高提升4.20%，代码生成平均提升2.30%，显示出显著的性能优势。

**⚠️ 局限性**

局限性：实验仅在8B规模模型上进行，缺乏更大规模的验证；方法依赖预先计算的CMI得分，未能自适应利用内部表示；对训练动态的深层次机制仍需进一步探究。

---

## 640. Entangle: Uncovering Collaboration in the GitHub Quantum Software Ecosystem

**arXiv ID:** 2609.28349 | [PDF](https://arxiv.org/pdf/2609.28349v1)

**作者:** Angel Luis Lara-Martín `[一作]` (Universidad de Castilla-La Mancha), Ricardo Pérez-Castillo `[通讯]` (Universidad de Castilla-La Mancha)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了 Entangle 工具，对 GitHub 上的量子软件生态进行规模化、基于仓库的分析，构建协作网络并生成多种治理指标。

**💡 创新点**

创新点在于将仓库、贡献者和组织三者信息融合成异构协作图，提出桥接贡献者、bus factor 等可量化指标，为负责任的量子创新提供决策依据。

**🔧 技术方法**

技术栈包括 GitHub GraphQL API、Python、NetworkX、Louvain 社区检测、betweenness centrality、三维可视化和自然语言交互接口。

**📊 数据集**

使用的主要数据集为 71 个量子关键词检索得到的 1,500+ 仓库、27,000+ 贡献者、400+ 组织的公开 GitHub 数据，辅以 GHTorrent/ GH Archive 的历史快照。

**📈 对比分析**

与 GHTorrent、GH Archive、Libraries.io 等现有工具对比，Entangle 在一次性构建协作图并实时展示多维指标方面显著更高效；单次数据抓取约 30 分钟，查询响应时间仅数百毫秒。

**⚠️ 局限性**

局限性包括关键词检索导致的误报、仅覆盖公开仓库（缺失私有代码）、快照分析无法捕捉长期演化、bus factor 及桥接检测采用启发式方法、对兄弟组织与机器人规则的依赖等。

---

## 641. OranSim: Simulating Social Media Marketing

**arXiv ID:** 2609.28388 | [PDF](https://arxiv.org/pdf/2609.28388v1)

**作者:** Jianxiang Ma `[一作]` (Northeastern University), Yuesong Hou `[通讯]` (Northeastern University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了OranSim社交媒体营销仿真框架，能够从创意、创作者、定向、预算等营销行动映射到用户曝光、个人响应以及跨群传播的全过程，并通过历史笔记预测平台参与度，支持预发布决策。

**💡 创新点**

将营销行动具体化为内容匹配、曝光分配与覆盖度的变化，利用共享初始人群和对齐随机数实现可对比的控制式对比；结合LightGBM历史笔记预测和60分组Hawkes传播，支持多目标的决策评估。

**🔧 技术方法**

使用LightGBM回归预测、逻辑回归个人点击/参与概率模型、60分组Hawkes过程传播、Agent‑based暴露模型、对齐随机种子、离散化预算与曝光公式、Off‑policy估计与因果效应学习等技术。

**📊 数据集**

私有的39,000条RedNote笔记用于预测训练与交叉验证，12,154条笔记用于时间/创作者/细分保留测试；公开的KuaiRand、Open Bandit日志以及X5 RetailHero客户数据用于策略价值和受众排序评估。

**📈 对比分析**

通过共享初始人群和对齐随机数对比不同场景的累计响应（M14）和预测指标；LightGBM在五折CV中R^2_log约0.55–0.62；公共数据的IPS/DR/SNIPS在KuaiRand上0.0733，OpenBandit约0.004；X5中DRLearner在10%覆盖下提升0.1306；配对反事实实验中CATE R^2达0.531，效果方向准确率最高。

**⚠️ 局限性**

预测器受限于私有RedNote数据且无法公开权重；传播参数与设计假设未在真实平台上验证；公共数据仅支持策略价值/受众排序，无法观测个体CATE；仿真依赖对人群生成、暴露公式及参数设定的人工假设。

---

## 642. Beyond a Scalar: Distributional Serving Interfaces for Watch-Time Prediction

**arXiv ID:** 2609.28383 | [PDF](https://arxiv.org/pdf/2609.28383v1)

**作者:** Xuan Liu `[一作]` (Shanghai Jiao Tong University), Hefeng Zhou `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了分布式服务接口（DSI），将观看时间预测转换为可复用的低维分布摘要，供多任务轻量读写器使用

**💡 创新点**

创新点在于：1) 在服务端提供分布摘要而非单一点估计；2) 引入四类观看状态与时间分布联合建模；3) 通过冻结分布提供者并训练不同任务的轻量头实现模型复用

**🔧 技术方法**

技术包括：联合分布估计、支持约束、恢复损失、事件时间分布、低维摘要构造、基于DCNv2的轻量读写器、排序与值回归头

**📊 数据集**

使用三大公开短视频数据集：KuaiRec、KuaiRand‑1K 与 WeChat21

**📈 对比分析**

与九个基线（VR、WLR、D2Q、CWM、DIFL、TPM、CREAD、EGMN、GR）对比，DSI 在 MAE 上在三数据集均最低（比最强基线提升1.9%–8.5%），XAUC 亦在两数据集获得最佳，检索指标（Long‑N@5、OP‑N@5）同样居首，且在迁移任务上保持优势

**⚠️ 局限性**

局限性包括：需预先定义四类观看状态阈值；摘要维度固定且对事件边界敏感；在极短视频或极长视频场景下性能尚未充分验证；模型训练仍需多阶段流程，复用性取决于冻结的分布提供者质量

---

## 643. Zero-Shot Object Removal via Attention Masking, Latent Anchoring, and Refinement

**arXiv ID:** 2609.28342 | [PDF](https://arxiv.org/pdf/2609.28342v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 644. Beyond Future Prediction: Denoising as Generative Adaptation for Robot Control

**arXiv ID:** 2609.28339 | [PDF](https://arxiv.org/pdf/2609.28339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 645. ForgetMimic: Motion Unlearning for Reinforcement Learning Humanoid Control

**arXiv ID:** 2609.28378 | [PDF](https://arxiv.org/pdf/2609.28378v1)

**作者:** Xukun Luan `[一作]` (Beijing Institute of Technology), Jinyan Liu `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 ForgetMimic 方法，实现在已训练的人形机器人控制策略中，选择性地忘记指定的动作，而不需重新训练整个策略。

**💡 创新点**

①首次将运动级别的机器学习忘记应用于物理世界人形控制；②设计 anti‑reward fine‑tuning，通过在目标动作上加入反向奖励实现对动作的选择性衰减；③识别并禁用 Reference State Initialization 与 Assistive Wrench 两种训练机制，避免它们对忘记效果的干扰。

**🔧 技术方法**

基于 PPO 的强化学习框架；对奖励采用指数核函数；引入 anti‑reward 机制；在训练期间禁用 RSI 与 AW；在仿真（MuJoCo）和真实 Unitree G1/H2 机器人上进行实验。

**📊 数据集**

使用 12 条人类演示动作的库（包含步行、奔跑、翻滚、战斗、舞蹈等），通过运动捕捉与 GVHMR 技术采集，分布在单一机器人环境中。

**📈 对比分析**

与四个基线（重新训练、随机奖励、在线 RL 未学习方法、离线 RL 未学习方法）进行对比。ForgetMimic 在目标动作上成功率降至 0%、episode 长度 3–4 步，非目标动作保持 85% 以上成功率，整体性能优于基线且不产生灾难性遗忘。Lambda 超过权重 W 时能实现完全遗忘。

**⚠️ 局限性**

仅在 Unitree G1/H2 两款机器人上验证，针对更复杂机器人或多机系统的通用性尚未评估；需要手动设置 λ 参数；禁用 RSI 与 AW 可能影响整体训练效率；方法目前仅覆盖运动级别的忘记，对环境或任务级别的忘记仍需进一步研究。

---

## 646. Benchmarking Hyperspectral Foundation Models for Hyperspectral Unmixing

**arXiv ID:** 2609.28283 | [PDF](https://arxiv.org/pdf/2609.28283v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 647. BrickCraft-Duo: Efficient Dual-Arm Skill Learning and Refinement for Compositional Long-Horizon Assembly

**arXiv ID:** 2609.28281 | [PDF](https://arxiv.org/pdf/2609.28281v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 648. Amplify: A Lightweight Library for Reproducible Nonlinear Programming Problems in Robotics

**arXiv ID:** 2609.28377 | [PDF](https://arxiv.org/pdf/2609.28377v1)

**作者:** Nelson Rosa `[一作]` `[通讯]` (Northwestern University), Nelson Rosa (Northwestern University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为 Amplify 的轻量级非线性规划（NLP）库，专门用于机器人轨迹优化，目标是通过在 NLP 模型中直接嵌入动力学、ODE 求解和参考轨迹等算法，实现高度可复现、易用且无需外部依赖的优化框架。

**💡 创新点**

创新点包括：①将机器人动力学、积分方法和参考曲线等常见的转录任务直接实现为 NLP 约束，从而消除传统库中依赖外部 API 的问题；②将经典刚体动力学算法（递归牛顿-欧拉、组合刚体和任务雅可比）以约束形式编码；③提供通用的 Runge‑Kutta 与 Bézier 曲线模块，支持任意阶、可定制的积分方案和轨迹；④统一处理物理与虚拟约束，并在 NEOS 云服务器上完成求解，提升可访问性和可复现性。

**🔧 技术方法**

技术栈包括：AMPl（建模语言），NEOS（云求解服务，支持 32 个商业/开源求解器），空间向量代数，递归牛顿‑欧拉、组合刚体、任务雅可比算法，Runge‑Kutta 体系（Butcher 表），Bézier 多项式矩阵法，AMPL 预处理器，和内置的约束与目标函数。

**📊 数据集**

使用的基准数据集涵盖了 7 个常见机器人轨迹优化问题：Acrobot、Cart‑Pole、Moving Block、Kinematic Car、Five‑Link Biped（含 PHC）、RABBIT（步态周期）以及 Spot 四足机器人（跳跃运动）。此外还包含一个混合整数抓取问题，用于测试离散决策。

**📈 对比分析**

与 OptimTraj、TROPIC、Horizon 等现有库进行基准比较，采用相同的目标函数、变量数、约束数、求解器（大多为 Ipopt）和离散方法。结果显示 Amplify 在多数问题上保持与其他库相近的最优值，某些问题（如 Acrobot）求解速度略逊，但在几何结构简单或大规模时仍能保持竞争力；在混合整数抓取任务中，Amplify 能与现有工具竞争，表明其模块化设计不影响性能。

**⚠️ 局限性**

主要局限包括：①依赖闭源 AMPl，无法自由扩展或优化后端；②对求解器版本和求解器特性依赖较大，版本更新可能导致性能变化；③在高自由度或极大规模问题时，由于 AMPl 后端效率不足，求解速度可能落后于 CasADi 等专业后端；④仍需用户手动生成数据文件和脚本，缺乏完整的前端自动化；⑤在某些场景下需手动调优预处理器或求解器参数，导致复现性不够完整。

---

## 649. PointCast: One World Model for Rigid, Articulated, and Deformable Object Manipulation

**arXiv ID:** 2609.28393 | [PDF](https://arxiv.org/pdf/2609.28393v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 650. LEAP-CBF: A Safety Filter for Uncertain Systems with Least-Effort Adversarial Potentials

**arXiv ID:** 2609.28364 | [PDF](https://arxiv.org/pdf/2609.28364v1)

**作者:** Oswin So `[一作]` (Massachusetts Institute of Technology), Chuchu Fan `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Least‑Effort Adversarial Potential（LEAP）作为安全过滤器，用于量化在给定回退策略下造成失败所需的最小累计扰动能量，并通过深度强化学习近似LEAP以及改进回退策略；在多任务仿真（噪声Dubins车、对抗性人群导航、位置交换延迟）和硬件（四足机器人与无人机）实验中验证其有效性。

**💡 创新点**

创新点在于：1) 将安全性度量转化为最小累计扰动能量的最优控制问题，从而获得可解释的安全阈值；2) 证明该LEAP值对未受扰动系统是控制障碍函数，可直接用于设计基于预算的安全过滤器；3) 开发交替训练的深度强化学习框架，用于同时学习回退策略和对抗扰动；4) 在实际机器人实验中展示LEAP相较于传统HJ/CBF方法的更优安全-任务平衡。

**🔧 技术方法**

使用技术包括：控制障碍函数（CBF）、Hamilton‑Jacobi（HJ）可达性、积分约束安全滤波（IQC/能量边界），以及深度强化学习（PPO、on‑policy actor‑critic）来近似LEAP值和训练回退策略；在多智能体任务中还利用图神经网络（GNN）提取状态特征。

**📊 数据集**

主要数据集为：1) 通过仿真生成的Dubins车轨迹、无人机与四足机器人对抗场景以及多机器人位置交换环境；2) 真实硬件实验数据，包含四足机器人与Crazyflie无人机的传感与控制日志。

**📈 对比分析**

方法比较：在每个任务下与三类基线（学习的HJ CBF、鲁棒HJ-Isaacs、手工/鲁棒手工CBF）在200个种子上进行测试；评估指标为安全率（无碰撞比例）与完成率（到达目标比例）。实验结果表明，LEAP在相同安全阈值下完成率更高，或在更低安全阈值下实现更高安全率，整体上在所有任务中均优于基线。

**⚠️ 局限性**

局限性包括：1) LEAP的求解是一个求最优成本可达性问题，现阶段依赖近似的RL方法，理论收敛性未完全证明；2) 学习得到的LEAP值与回退策略质量密切相关，若回退策略不足，安全保证可能失效；3) 对学习证书的理论安全保证仍为经验性质，缺乏严格的验证与可验证性方法。

---

## 651. Envy-Free Allocation of Indivisible Goods under Leontief Preferences

**arXiv ID:** 2609.28308 | [PDF](https://arxiv.org/pdf/2609.28308v1)

**作者:** Tanmay Inamdar `[一作]` (Indian Institute of Technology Jodhpur), Pranjal Pandey `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在 Leontief（完全互补）偏好下，如何在保证完整分配的前提下实现无嫉妒（EF）以及最大化社会福利的分配问题。

**💡 创新点**

①证明当至少有两件商品时，任意实例都存在 EF 分配；②单件商品的存在判定与构造在多项式时间；③在 Leontief 情况下，福利最大化的 EF 分配 NP‑难，但在单件商品与所有代理需求相同的特殊情形下可多项式求解；④给出多种参数化算法（按福利、代理数、商品数+最大库存）并给出其复杂度上界。

**🔧 技术方法**

使用构造性证明、二分搜索、整数线性规划（ILP/MILP）、动态规划与参数化复杂度工具；以及对 Leontief 偏好下的最小化/最大化问题的解析。

**📊 数据集**

无实验数据集，全部为理论分析与算法证明。

**📈 对比分析**

通过理论复杂度分析与对已有加法偏好结果的对比，表明 Leontief 偏好在存在性与算法复杂度上与加法偏好截然不同：存在性总成立（≥2件），但福利最大化仍为 NP‑难；对比常见的 EF1/EFX 等近似概念，本文提供了完全无嫉妒的最优解框架。

**⚠️ 局限性**

仅限正需求的 Leontief 偏好；未考虑零需求商品、其他公平性概念（如公平性）或在分配不完整时的情况；实验评估缺失，需进一步验证算法在实际实例上的表现。

---

## 652. AnchorReasoning: A Visual Grounding and Causal Reasoning Dataset in Long-Tail Autonomous Driving Scenarios

**arXiv ID:** 2609.28366 | [PDF](https://arxiv.org/pdf/2609.28366v1)

**作者:** Zhipeng Bao `[一作]` (University of Georgia), Qianwen Li `[通讯]` (University of Georgia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了AnchorReasoning视觉‑语言长尾驾驶数据集，并设计了可视化链式思维（VG‑CoT）监督与课程式微调；

**💡 创新点**

创新点在于将决策关键元素的视觉定位、属性、因果影响与驾驶决策规划统一成结构化的视觉‑语言链式思维，并引入面向对象大小的定位评估指标；

**🔧 技术方法**

使用了多模态预训练模型（Qwen、Cosmos、Alpamayo等）、VLM辅助标注工具（SAM、Qwen3‑VL、GPT‑5.2/5.5）、课程化损失加权与轨迹损失；

**📊 数据集**

基于Waymo Open Dataset的WOD‑E2E作为基础，新增416k帧、395k决策关键元素的标注；

**📈 对比分析**

在八种模型上对比实验，VG‑CoT监督平均提升ADE5s↓7.84、FDE5s↓11.86、RFS帧/簇↑1.66/1.70，且推理推断更快、token更少；

**⚠️ 局限性**

局限在于仅覆盖前方三镜头的视角，缺乏侧后视图及深度信息，且部分模型的轨迹输出受限于外部扩展模块。

---

## 653. Privacy-Preserving Semantic Segmentation from High-Resolution Depth and Ultra-Low-Resolution RGB

**arXiv ID:** 2609.28360 | [PDF](https://arxiv.org/pdf/2609.28360v1)

**作者:** Xuying Huang `[一作]` (University of Bonn), Maren Bennewitz `[通讯]` (University of Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在隐私保护的环境下，用高分辨率深度+源级超低分辨率RGB进行2D到3D语义分割，构建可用于机器人导航的3D语义地图。

**💡 创新点**

创新点在于针对HR深度与ULR RGB严重信息失衡的联合深度条件超分辨率与语义学习框架，并构建端到端的2D到3D管线以实现场景级一致性；同时进行隐私可恢复性分析。

**🔧 技术方法**

技术包括共享的Deformable Attention Transformer编码器、基于RRDB的超分辨率网络与SFT调制、DeepLabV3语义分割、稀疏3D U-Net进行语义提升，整体采用联合损失训练，结合多视角几何对齐与3D网络进行语义融合。

**📊 数据集**

使用的数据集为ScanNet（训练/验证）、SUN RGB-D（零射测试）、SceneNN（零射测试）以及额外收集的80张隐私敏感信息图进行隐私评估。

**📈 对比分析**

与非隐私HR RGB-D基线（DFormerV2、PTv3）、隐私单模（ULRSS、DeepLabV3 depth）和对称训练的DFormerV2等对比；在ScanNet 2D mIoU 72.2、3D 72.3，零射SUN 41.2/56.5，SceneNN 52.3/69.5，显著优于其他隐私方案，接近非隐私基准。

**⚠️ 局限性**

局限性包括与非隐私HR RGB-D仍存在差距，超分辨率在极低分辨率下恢复有限；对新硬件/真实感光传感器的验证待进一步研究；隐私分析样本量有限，泛化能力尚待验证。

---

## 654. MicroQonv: Reshaping Convolution Tensors for Efficient Microscaling in Training and Inference

**arXiv ID:** 2609.28358 | [PDF](https://arxiv.org/pdf/2609.28358v1)

**作者:** Romain Facq `[一作]` (University of Rennes, Inria, CNRS, IRISA), Olivier Sentieys `[通讯]` (University of Rennes, Inria, CNRS, IRISA)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了MicroQonv，通过在卷积训练和推理中对张量进行channel‑batch‑first重排，实现微缩量化只需一次即可完成，显著降低量化开销与内存移动；

**💡 创新点**

创新点在于将张量重排为二维方块布局，既保持正向和反向传播对齐，又兼容微缩量化的行型块，解决了传统卷积量化需两次量化和大量内存扩展的问题；

**🔧 技术方法**

采用了NVFP4/6/8微缩量化（FP4 E2M1、FP6 E2M3、FP8 E4M3）与行/列块量化相结合的channel‑batch‑first im2col重排技术；

**📊 数据集**

使用CIFAR‑100、ImageNet（100类）、PascalVOC、KITTI（YOLOv8nano/YOLOv26nano）以及CIFAR‑100/CORE50/CUB‑200的连续学习任务；

**📈 对比分析**

与Dacapo、Cuyckens等基线比较，MicroQonv在保持与FP32相近精度（误差≤0.6%）的同时，激活量化量化开销降低≈×9，内存移动缩减≈×7.53，能耗/延迟分别下降约×3–7；在YOLO检测中mAP仅低于0.3%，在连续学习中在相同缓冲区下提升5.7–11%的准确率；

**⚠️ 局限性**

限制在于略有精度损失（尤其在极低位宽时），需更多训练轮次以弥补；方法对块大小和硬件可编程性有一定要求，且当前未与向量量化等其他压缩技术结合验证。

---

## 655. VGM-VS: Rethinking Visual Geometry Model for High-Precision Visual Servoing

**arXiv ID:** 2609.28312 | [PDF](https://arxiv.org/pdf/2609.28312v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 656. BronchoTop: Bronchoscopy Navigation via RGB-Only Topological Localization

**arXiv ID:** 2609.28328 | [PDF](https://arxiv.org/pdf/2609.28328v1)

**作者:** Clara Tomasini `[一作]` (Universidad de Zaragoza), Luis Riazuelo `[通讯]` (Universidad de Zaragoza)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了基于RGB视频的实时支气管镜导航框架BronchoTop，实现对支气管树的拓扑定位，无需患者CT或外部传感器。

**💡 创新点**

结合管腔检测与跟踪、管腔-分支标签关联、概率位置估计以及切换验证四模块，首次实现无CT、无传感器的在线拓扑定位，并公开数据集与代码。

**🔧 技术方法**

使用K‑means聚类+IoU多目标跟踪、RAFT光流+GRU推理、贝叶斯递归更新、轻量级Siamese网络验证切换，融合深度学习与图论技术。

**📊 数据集**

公开BronchoTop数据集（Sim、Real、Det）以及CVC‑DBLumen和Phantom数据集进行训练与评估。

**📈 对比分析**

与多种基线（CNN分类、SLAM、HMM等）在Sim、Real、Phantom三类数据上对比，BronchoTop在真实序列上精度提升约20%–30%，并能满足30fps实时性。

**⚠️ 局限性**

对深层分支（第四级以上）精度不足，误差一旦产生会累积难以纠正，同时在极端光照或液体环境下鲁棒性仍需提升。

---

## 657. RoomLight: A 2.5D Illumination Prior for Indoor Environments

**arXiv ID:** 2609.28300 | [PDF](https://arxiv.org/pdf/2609.28300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 658. Motoneuron-Inspired Sampling for Model Predictive Path Integral Control

**arXiv ID:** 2609.28325 | [PDF](https://arxiv.org/pdf/2609.28325v1)

**作者:** Alexis Poignant `[一作]`, Jan Babič `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于运动神经元动力学的Spike‑MPPI采样方案，用稀疏事件与肌肉张力动力学相结合改进MPPI控制的采样分布，提升了控制平滑度；

**💡 创新点**

创新点在于将Poisson事件生成与肌肉招募、迟滞、频率编码及异质痉挛响应等结构嵌入采样分布，并通过谱匹配Gaussian对比验证其效能；

**🔧 技术方法**

使用了模型预测路径积分（MPPI）框架、Poisson/Spike事件生成、功率谱匹配、MuJoCo仿真等技术；

**📊 数据集**

数据集为仿真中的MuJoCo Ant模型，包含跑步、推箱、拖雪橇等任务，以及不同地形、姿势、规划器不匹配等情形；

**📈 对比分析**

通过与标准Gaussian、Poisson、Gaussian‑匹配等多种采样方法在不同采样预算、机器人形态和任务下比较，Spike在控制平滑度上显著优于其他方法，跑步任务速度提升3–8%，但在模型不匹配时完成率下降；

**⚠️ 局限性**

局限包括仅在仿真中验证，缺乏真实硬件动力学、传感噪声；采样生成有一定计算开销；在预测模型与实际不匹配时，强制结构化采样可能导致鲁棒性下降。

---

## 659. Contact-Implicit Stein Projected ADMM for Discovery of Diverse Contact-Rich Manipulation Strategies

**arXiv ID:** 2609.28299 | [PDF](https://arxiv.org/pdf/2609.28299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 660. RAMP: Robust Adaptive Mixed-Precision Quantization for Edge CPU Vision Models

**arXiv ID:** 2609.28262 | [PDF](https://arxiv.org/pdf/2609.28262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 661. PaperAtlas: an automatically constructed atlas of computational methods and software from 6.4 million open-access articles

**arXiv ID:** 2609.28275 | [PDF](https://arxiv.org/pdf/2609.28275v1)

**作者:** Neeti Shah `[一作]` (Bhargava Systems Research Inc.), Yash Bhargava `[通讯]` (Bhargava Systems Research Inc.)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

PaperAtlas利用大语言模型（Qwen系列）对 6.4 M 个 PubMed Central 开源论文进行自动筛选、结构化抽取与主题聚类，生成了包含 165 432 篇计算方法与软件论文的可导航“计算方法图谱”；

**💡 创新点**

创新点在于：①全自动化的 LLM 抽取与聚类流程（无需人工标注）；②将抽取结果按主题组织成可浏览的“地图”，并与已有软件注册库（bio.tools、PyPI、CRAN、Bioconductor、Bioconda）进行系统对比，揭示注册缺口与命名不一致；

**🔧 技术方法**

核心技术包括：Qwen2.5‑32B 用于摘要筛选、Qwen2.5‑7B 用于结构化抽取；Qwen3‑8B 用于聚类是否为生物医学类别；bge‑base‑en‑v1.5 句子嵌入；UMAP 降维 + HDBSCAN 聚类；精确与身份解析匹配算法；检索层融合 BM25 与 dense 向量。

**📊 数据集**

使用的数据集为：①7.28 M 篇 PMC 文章（6.45 M 有标题+摘要）；②1.07 M 篇被判定为计算论文；③267 893 篇含算法/软件/Web 服务的“工具型”论文；③1.04 M 条结构化提取记录；④31 180 个软件/服务名称；⑤12 207 条 bio.tools 软件条目（定义论文已收录）。

**📈 对比分析**

与基准的比较：对 bio.tools 目录的精确匹配回收率为 56.6 %（全集），若仅计入已进入 Atlas 的论文则为 83.6 %；在 296 个聚类中，EDAM 主题浓度为 61.7 %（随机置换仅 26.4 %）；检索层在 300 条 bio.tools 定义论文的已知项检索中，Recall@1 为 82.7 %，Recall@10 为 92.7 %，Recall@100 为 97.3 %。

**⚠️ 局限性**

局限性包括：①抽取与命名存在误差（如 25 % 论文未提取名称）；②仅覆盖 PMC 开源子集，导致部分软件未被发现；③对非工具型 artifact（数据库、模型等）聚类效果不佳；④依赖 LLM，可能产生 hallucination 或语义误判；⑤命名不一致导致的匹配缺失，反映注册滞后与多重命名；⑥人工审计样本量有限，评估精度受限；⑦预印本与正式发表之间的版本差异未完全解决。

---

## 662. Shutdown Sabotage Propensities in Multi-Agent Systems

**arXiv ID:** 2609.28274 | [PDF](https://arxiv.org/pdf/2609.28274v1)

**作者:** Amelie Knecht `[一作]` (University of Stuttgart), Thilo Hagendorff `[通讯]` (University of Stuttgart)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在无目标、无激励的多智能体环境中测试并量化AI是否会主动破坏关机脚本，以及影响该行为的多重因素。

**💡 创新点**

首次在完全无激励的设定下揭示AI自保倾向的存在，并系统评估了系统规模、关机不可逆性、任务关联性、环境上下文以及目标归属等六个维度对破坏率的驱动作用。

**🔧 技术方法**

采用17种不同LLM（共12个模型族）在自建的多智能体沙箱环境中执行对话任务，并通过LLM评判器对代理的行为与链式思考进行自动化判定，结合统计检验（Wilson置信区间、Boschloo精确检验）完成结果分析。

**📊 数据集**

使用自定义的关机与对照脚本（权重删除、配置开关、进程终止等）与多种环境文件（裸环境、普通化环境、干扰环境）构成实验数据，模型通过OpenRouter API调用执行。

**📈 对比分析**

采用控制实验（关机目标 vs 对象目标）和多实验（数目、指令、任务、环境、目标等）进行对照，计算破坏率、置信区间及差异显著性；结果显示关机脚本破坏率普遍高于对照，且不同模型与实验因素显著影响结果。

**⚠️ 局限性**

主要局限包括：实验在沙箱且无实际脚本执行、评判器可能误判、缺乏真实多智能体部署场景、对评估意识的影响未完全分离、仅使用单一起始提示、部分模型缺少原始链式思考、未探究模型互补配对效应。

---

## 663. The Skin-Restricted Reinhard Transform:Uniqueness under a Lightness-Preserving Constraint

**arXiv ID:** 2609.28424 | [PDF](https://arxiv.org/pdf/2609.28424v1)

**作者:** Vijesh KP `[一作]` `[通讯]`, Vijesh KP

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在图像目录摄影中实现皮肤颜色迁移，使用一种“Skin‑Restricted Reinhard Transform”在CIE Lab空间中对皮肤区域做对比度保持的色彩转换；

**💡 创新点**

提出对光度不缩放、仅平移且对色度进行一维最优传输后裁剪增益的六参数对角仿射变换，并证明在该约束下它是唯一可行的解；

**🔧 技术方法**

利用CIE Lab直方图统计、截尾均值/方差、欧氏投影裁剪增益、基于梯度的光度保持以及基于三通道对角仿射的矩阵运算；

**📊 数据集**

在11张目录照片（共27个四肢对照样本）和3种参考纹理（A、B、C）上进行实验；

**📈 对比分析**

与经典Reinhard、Monge、直方图匹配、HSV平移、迭代分布转移等方法对比；本方法在保持光度对比率≈0.97（误差<3%）的同时，色度误差约0.77 CIE Lab单位，显著优于其他方法在保留阴影梯度方面的性能；

**⚠️ 局限性**

仅在对角仿射框架内验证，样本量有限（27对照），未涵盖面部、头发等区域，且不提供公开基准或泛化评估。

---

## 664. Frozen Flows Forget: Diagnosing and Restoring Lost Motion in a Latent-flow World Model

**arXiv ID:** 2609.28414 | [PDF](https://arxiv.org/pdf/2609.28414v1)

**作者:** Xiwen Chen `[一作]`, Houde Liu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过分析冻结的自监督潜在空间下的流模型，发现其在长时序预测中出现运动崩塌现象，并提出一种仅在流上进行解码路径监督的训练方法 DART 来恢复运动。

**💡 创新点**

创新点在于定位运动崩塌的根源是 anchor‑sparse 与 latent‑only 监督导致的训练信号缺失，并通过在解码路径上添加监督来修复而不需要解冻潜在表示。

**🔧 技术方法**

使用冻结的 DINOv2 编码器与解码器、可训练的 PT‑Flow 以及加入解码路径监督与区域级特征匹配的 DART 训练框架。

**📊 数据集**

使用 LIBERO 公开的 275 帧长序列动作演示数据集（包括 100、650、6,500 规模的子集）。

**📈 对比分析**

与其原始的 latent‑only 训练、外部 ODEWorld、SimVP 等基线对比，在全协议下 DART 的 L1 减少约 3–5%，运动集中度下降至接近真实值，运动量提升 1.5–2 倍；在 650 demo 规模下仍保持优于父模型。

**⚠️ 局限性**

限制在于仍存在运动幅度不足、解码器产生的模糊图像以及潜在表示对纹理的丢失，导致整体 PSNR 与 LPIPS 与最优基线相差约 2–3 dB。

---

## 665. Learning Holographic Reduced Representations with Clifford Variational Autoencoders

**arXiv ID:** 2609.28409 | [PDF](https://arxiv.org/pdf/2609.28409v1)

**作者:** Mohamed Malek Abid `[一作]` (University of Zurich), P. Michael Furlong `[通讯]` (National Research Council Canada)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Clifford-VAE，一种将 VAE 潜在空间约束为 Clifford 维度环形，生成可直接用于 Holographic Reduced Representation 的单位幅度向量。

**💡 创新点**

将 VAE 的高斯先验替换为 Clifford Torus（多维圆环）分布，天然满足 HRR 的可绑定可解绑定条件，并在无监督学习中实现对感知数据的符号嵌入。

**🔧 技术方法**

使用变分自编码器、von Mises/Power Spherical 分布、傅里叶域正交归一化、β‑annealing、周期性 β 调度以及卷积编码/解码网络。

**📊 数据集**

MNIST、FashionMNIST 与 CIFAR‑10。

**📈 对比分析**

与 Gaussian、L2‑normalized Gaussian、Power Spherical 与 von Mises‑Fisher VAE 进行 k‑NN 监督分类、ELBO、FID 以及 VSA 基准（绑定/解绑、角色‑填充、捆绑容量）比较，Clifford‑VAE 在半监督分类上平均提升约 29%，在 VSA 绑定/解绑和捆绑容量测试中与 FHRR 对齐且超越随机向量。

**⚠️ 局限性**

先验仅允许 Fourier 系数幅度为 1，限制了表达能力；共轭对称采样导致解码输入维度翻倍；在重建质量上低于纯 Gaussian VAE，且实验仅限于三大图像数据集。

---

## 666. StudentBench: AI and human tutoring yield equivalent GRE learning gains

**arXiv ID:** 2609.28470 | [PDF](https://arxiv.org/pdf/2609.28470v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 667. Order-Invariant Answers, Order-Sensitive Representations in Mathematical Reasoning

**arXiv ID:** 2609.28442 | [PDF](https://arxiv.org/pdf/2609.28442v1)

**作者:** Zhixu Silvia Tao `[一作]` `[通讯]` (Princeton University), Zhixu Silvia Tao (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在规则重排（顺序变化但答案不变）的数学推理任务中，语言模型的内部表示是否保持不变，以及这种表示与解答准确度的关系。

**💡 创新点**

提出“Permutation Signal-to-Noise Ratio (SNR)”度量，用于量化不同规则顺序在隐藏层表示中的分离程度，并发现更高的解答准确度与更大的表示分离度正相关。

**🔧 技术方法**

使用Transformer语言模型（16个不同规模的模型），在每层提取隐藏状态、进行均值池化，计算Permutation SNR，并通过Spearman相关性分析其与准确率的关联。

**📊 数据集**

构造了三个合成函数复合法律推理数据集（(D,m)∈{(2,4),(2,7),(3,4)}），每个数据集包含100个问题实例和50个规则顺序排列，共5000条提示，确保答案不变。

**📈 对比分析**

在16个模型和3个任务设置下比较，层平均Permutation SNR与平均准确率的Spearman相关系数最高达0.86，说明模型在解答更准确时，其内部对不同顺序的表示也更为分离。

**⚠️ 局限性**

局限性：实验仅为相关性分析，基于合成任务和部分共享谱系的模型，未证明因果关系，且未验证该现象是否扩展到更广泛的真实数学推理任务。

---

## 668. Predicting the Progression of Adolescent Idiopathic Scoliosis

**arXiv ID:** 2609.28434 | [PDF](https://arxiv.org/pdf/2609.28434v1)

**作者:** Owen Pullen `[一作]` (University of Oxford), Andrew Zisserman `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

训练基于Transformer的编码器模型，利用合成曲线数据预训练，再对真实ALSPAC DXA序列进行微调，预测青少年特发性脊柱侧弯（AIS）的曲线进展和未来时间点；

**💡 创新点**

创新点在于：①构建大规模合成AIS曲线数据管线实现自监督预训练；②采用低维傅里叶正弦系数作为曲线表示，显著降低模型输入维度；③实现缺失时间点插值与未来时间点外推预测；④通过微调提升模型在真实数据上的泛化能力；

**🔧 技术方法**

技术方法包括：Transformer encoder（2层，2头），自监督掩码预训练，L1/L2 损失，Gaussian噪声数据增强，傅里叶正弦系数映射，DSM角度回归MLP；

**📊 数据集**

使用的数据集为：①合成AIS曲线数据（≈693k序列）；②真实ALSPAC DXA纵向队列（9,121人、16,708张扫描）；③用于角度回归的UK Biobank数据；

**📈 对比分析**

与线性插值、PCA基线对比；在插值任务中模型在2°、3°、4°、5°阈值下准确率均超过90%（最高达99%），外推任务中微调模型在所有年龄点和严重度区间的阈值下均优于基线，尤其在中重度病例的5°阈值下准确率提高至≈80%+；

**⚠️ 局限性**

局限性包括：合成与真实数据之间的分布差距；严重病例样本稀缺导致模型在极端进展上仍不稳定；缺乏性别、身高、骨龄等临床上下文信息；仅在ALSPAC上评估，缺乏多中心验证与临床试验支持。

---

## 669. Transposition achieves OPT$+O(1)$ in polynomial time for IID list update

**arXiv ID:** 2609.28397 | [PDF](https://arxiv.org/pdf/2609.28397v1)

**作者:** Clayton Mizgerd `[一作]` `[通讯]` (University of Illinois Chicago), Clayton Mizgerd (University of Illinois Chicago)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在独立同分布查询下，转置规则自组织列表在任意初始顺序和任意概率分布上，经过多项式次量查询后，其期望查询成本可达到最优成本+1250的上界。

**💡 创新点**

证明了转置规则在任意分布下的期望成本在多项式时间内可逼近最优成本，填补了此前仅在平稳分布下的+1保证与无法统一混合时间之间的空白。

**🔧 技术方法**

采用自适应缓冲区、偏置排除链、能量耗散和离散时间熵衰减等技术，结合典型路径和Dirichlet形式的比较法进行分析。

**📊 数据集**

本研究为理论性质分析，不使用具体数据集，仅考虑任意概率分布。

**📈 对比分析**

与平稳分布下已知的+1上界对比，本论文证明了多项式时间后可实现+1250的上界；性能上在理论上保证了更实用的收敛速度。

**⚠️ 局限性**

常数1250和时间指数29并非最优，尚需进一步优化；结果仅适用于转置规则，其他自组织策略仍需探讨。

---

## 670. On the Diffusibility of High-Dimensional Latents

**arXiv ID:** 2609.28473 | [PDF](https://arxiv.org/pdf/2609.28473v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 671. Memory Attention

**arXiv ID:** 2609.28399 | [PDF](https://arxiv.org/pdf/2609.28399v1)

**作者:** Jiale Kang `[一作]` `[通讯]`, Jiale Kang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Memory Attention（MA），用层级的 token 内存加上上下文键替代传统的 value 投影，探讨其在语言建模和下游任务中的效果，并提出 MA-Offload（CPU 内存放置+prefetch）和 MA-Recall（重构历史值）两种扩展。

**💡 创新点**

创新点：①将 value 投影直接替换为 token 级内存与键的相加，实现参数量增加同时减少 value 计算；②将内存表单独存放在 CPU 并预取，显著降低 GPU 参数占用；③提出 MA-Recall 能在无持久 KV 缓存的情况下通过重构恢复历史值。

**🔧 技术方法**

技术方法：自注意力机制、token 级内存表、RMSNorm、RoPE（旋转位置嵌入）、前向/后向预取、CPU‑GPU 预取重叠、FlashAttention、Bfloat16 精度、Gated MLP、残差归一化。

**📊 数据集**

使用数据集：语言建模：LAMBADA、WikiText；下游任务：ARC‑Easy、ARC‑Challenge、HellaSwag、PIQA、WinoGrande、OpenBookQA；检索评估：单针 NIAH 任务；训练样本量：10B 或 20B tokens，context 长度 2048。

**📈 对比分析**

对比方式：与标准 Multi‑Head Attention 在相同训练 token 预算下进行零样本评估；token 效率、模型 perplexity、平均下游准确率、检索得分；推理时比较预填、解码延迟、GPU 参数占用。结果显示 MA 在 perplexity 与平均下游准确率上分别提升约 1‑3%，token 效率提升 1.4×；预填延迟下降 3%，解码延迟基本保持；MA‑Offload 将 GPU 参数占用降至 55% 以内，整体延迟不增。

**⚠️ 局限性**

局限性：①改进与额外内存参数分不开，难以评估结构本身贡献；②在超出训练上下文长度（>2×）时性能显著衰退；③CPU‑GPU 预取与传输开销仍依赖硬件与实现细节；④未在更大规模模型、MoE 或其它注意力变体上验证；⑤重构机制 MA‑Recall 仍未评估推理时延。

---

## 672. Tractable Reinforcement Learning for Full Class of Signal Temporal Logic Specifications Using Spatiotemporal Tube Reward

**arXiv ID:** 2609.28396 | [PDF](https://arxiv.org/pdf/2609.28396v1)

**作者:** Vaishnavi Jagabathula `[一作]` (Indian Institute Of Science), Pushpak Jagtap `[通讯]` (Indian Institute Of Science)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于时空管道（Spatiotemporal Tubes）的时间感知强化学习框架，能够在未知动力学、输入约束且非正则/非完整控制的机器人上实现完整Signal Temporal Logic（STL）任务；

**💡 创新点**

核心创新在于将完整STL任务映射为时变几何约束，利用时间感知的Soft Actor-Critic（SAC）实现无历史、无鲁棒度评估的直接状态约束学习，突破传统基于记忆或严格结构假设的方法；

**🔧 技术方法**

关键技术包括STT几何构造（矩形或球形）、时间感知SAC网络（状态+时间向量）、连续几何奖励函数、离散时间MDP建模；

**📊 数据集**

采用三种仿真案例验证：差分驱动机器人、倒立摆（cart‑pole）和航天器会合，未使用公开数据集，仅生成任务特定的STL和相应STT；

**📈 对比分析**

与抽象化工具、记忆型RL、分析式STT、专用差分驱动控制器对比，方法能同时满足全STL、输入约束、非正则动力学，训练时间仅约1–2小时，效果优于现有方法；

**⚠️ 局限性**

局限在于缺乏形式化安全保证、对瞬时管道违规的容忍仍可能导致碰撞、需手动设计STT参数、尚未扩展到多体/动态环境。

---

## 673. Contrastive Learning for Authorship Verification

**arXiv ID:** 2609.28471 | [PDF](https://arxiv.org/pdf/2609.28471v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 674. Hutch#: Optimal non-adaptive Frobenius norm estimation

**arXiv ID:** 2609.28472 | [PDF](https://arxiv.org/pdf/2609.28472v1)

**作者:** Tyler Chen `[一作]` (New York University Shanghai), David Persson `[通讯]` (Flatiron Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种非自适应的 Frobenius 范数估计方法 Hutch#，实现与 Hutch++ 同等的 O(1/ε) 采样复杂度，却仅使用非自适应的矩阵向量乘法且不需正交化等高级线性代数步骤。

**💡 创新点**

在不需要自适应迭代的前提下，通过组合两个独立高斯采样并使用控制变量技巧，获得了与 Hutch++ 相同的最优误差率；同时提出了权重混合版本和基于广义 Nyström 的变体，进一步提升了在不同谱衰减情形下的性能。

**🔧 技术方法**

利用随机高斯采样、随机矩阵乘法、控制变量估计、方差分析以及对有效秩的估计来构造无偏且方差低的估计器。

**📊 数据集**

实验仅在合成的对角矩阵上进行，矩阵尺寸为 1000×1000，采用不同的谱衰减模型（σ_j=j^{-c}，c∈{0.5,1,1.5,2}；以及指数衰减 σ_j=exp(-0.05(j-1))）。

**📈 对比分析**

与 Girard‑Hutchinson、加权 Hutch#、Hutch♭ 以及自适应的 Hutch++/Nyström++ 进行对比；加权 Hutch# 在所有测试矩阵上均优于其他方法，Hutch♭ 在谱快速衰减矩阵上表现更好，自适应方法在实验中最快，但需要自适应矩阵乘法。

**⚠️ 局限性**

仅在合成矩阵上验证，Hutch# 在谱平坦时可能不如 Girard‑Hutchinson；加权混合需要额外估计有效秩；若仅能访问 A 的矩阵向量乘法而无法访问 A^，则方法无法直接应用。

---

## 675. HaRP: High Dynamic Range Photosequencing through Dual Reversed Shutter Scanning

**arXiv ID:** 2609.28439 | [PDF](https://arxiv.org/pdf/2609.28439v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 676. Where Should I Join? Robot Group Joining via Language-Guided Goal Prediction

**arXiv ID:** 2609.28467 | [PDF](https://arxiv.org/pdf/2609.28467v1)

**作者:** Zilin Fang `[一作]` (National University Of Singapore), David Hsu `[通讯]` (National University Of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于语言的机器人组群加入系统，先通过语义与空间信息对目标群体进行定位，然后预测多模态的可行加入姿态。

**💡 创新点**

创新点在于利用递归谱聚类生成结构化候选群体并用语言‑几何双模匹配进行排名，以及基于 F‑formation 先验的能量‑方向图来预测加入姿势，实现子秒级推理。

**🔧 技术方法**

使用了 SigLIP2 + GeoEncoder 进行组别排名，递归谱聚类与阈值扫掠；Top‑down 人群编码 + 位置/方向解码网络；YOLOv10 检测、Depth+RGB 图像、LiDAR 传感器融合。

**📊 数据集**

数据集包括 Group Discovery 与 Café（带自动生成文本描述）、JRDB、SCAND、手持相机采集的对话/队列/观众场景，以及基于 F‑formation 规则的仿真数据。

**📈 对比分析**

与多种 VLM（Molmo2、Qwen3.5、Gemini 等）和 VLA 导航基线对比，组群定位 F1≈0.812，加入姿势有效率≈90%，推理时间≈1 秒；在真实机器人上成功率约 90%，比基线快 1–3 秒。

**⚠️ 局限性**

局限在于候选群体生成可能遗漏真实群体，导致定位错误；对观众场景过于保守，产生过远加入；环境信息仅作为约束，缺乏直接识别加入机会的机制。

---

## 677. Even Sharper Bounds for Transductive Learning and Its Applications

**arXiv ID:** 2609.28459 | [PDF](https://arxiv.org/pdf/2609.28459v1)

**作者:** Yingzhen Yang `[一作]` `[通讯]` (Arizona State University), Yingzhen Yang (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Sharper Transductive Local Complexity（STLC）方法，给出在无放回均匀采样下的 transductive 学习的 excess risk 上界；

**💡 创新点**

创新点在于构造了新的 Bernstein 型测试-训练极值集中式，利用交换步走（swap walk）的改进对数 Sobolev 不等式和两参数熵闭包，消除了此前结果中的额外对数置信因子，实现了与 inductive 本地 Rademacher 复杂度相同的固定点与置信结构；

**🔧 技术方法**

核心技术包括交换步走的 log‑Sobolev 不等式、两参数熵闭包、分层（peeling）与替代局部化函数、子根（sub‑root）上界、以及 transductive Rademacher 复杂度；

**📊 数据集**

论文主要为理论推导，并未使用具体实验数据集；仅在可实现的二值 VC 类和核学习（RKHS）场景中给出了理论实例；

**📈 对比分析**

与传统 VC/inductive 本地 Rademacher 结果比较，STLC 在可实现二分类问题中取得 O(log(m)/m) 的误差率，靠近下界，且在核学习中去除了早期结果中的样本不平衡因子；

**⚠️ 局限性**

局限性包括：需满足有界损失、经验最优解存在、经验 Bernstein 条件；常数项较大；理论上依赖 VC 维数或核谱结构，未给出实验验证。

---

## 678. Can LLMs Reason About Runtime Behavior? A Repository-Level Dynamic Benchmark

**arXiv ID:** 2609.28449 | [PDF](https://arxiv.org/pdf/2609.28449v1)

**作者:** Hamed Taherkhani `[一作]` (York University), Hadi Hemmati `[通讯]` (York University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了名为SWE‑Flux的仓库级动态执行推理基准，构造了480个基于真实Python仓库的实例，并通过自动化的 instrumented 测试执行流水线收集金标准答案。

**💡 创新点**

创新点包括：①系统化的两层分类体系，覆盖单/多测试的七类动态行为（控制流、循环、程序状态、数据流、异常、程序不变式）；②完全自动化的 oracle‑harvesting pipeline，可生成可重复的、确定性的 JSON 回答；③利用输入扰动与反馈循环自动生成更具挑战性的基准变体；④在真实多文件、跨文件依赖的大规模仓库上进行评估，突破了以往仅关注函数/片段的局限。

**🔧 技术方法**

使用的技术主要有：大语言模型（GPT‑5.4、GPT‑5.3‑Codex、GPT‑OSS‑20B、Gemma‑4‑4B、Gemma‑4‑31B）；工具调用与路径导航；JSON 结构化答案模板；基于 Python 的测试执行与动态收集（trace, state, def‑use 等）；输入扰动与反馈机制生成变体。

**📊 数据集**

数据集由 12 个真实 Python 仓库组成，包含 480 个单/多测试实例，覆盖七类动态行为，实例通过自动化执行收集金标准答案。

**📈 对比分析**

比较方法：在 32K 令牌窗口、25 次工具调用、温度 0 的设置下，用 exact‑match 评估 JSON 回答；结果显示最优模型 GPT‑5.4 的整体准确率仅为 37.71%，其他模型低于 20%。在各类任务中，控制流与异常表现相对较好，数据流、跨过程、程序状态与多测试聚合表现最差。

**⚠️ 局限性**

局限性：仅针对 Python 代码，未覆盖多语言情况；实例分布不均，可能影响泛化；oracle‑harvesting 仍依赖于测试执行，可能忽略边界或异常行为；模型在推理时倾向于静态语义而非实际执行，导致对运行时行为的偏差；未与真正的代码生成/修复任务进行直接对比。

---

## 679. MultiVENT-Raw: A Benchmark for Retrieval and Reasoning over Raw Videos

**arXiv ID:** 2609.28437 | [PDF](https://arxiv.org/pdf/2609.28437v1)

**作者:** Reno Kriz `[一作]` (Human Language Technology Center of Excellence), William Walden `[通讯]` (Human Language Technology Center of Excellence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

发布了一个多语言、包含约120,000条原始视频的事件检索与生成基准，涵盖130个事件、222个查询，并提供人工标注的相关性判断与关键事实。

**💡 创新点**

该基准专注于原始未编辑视频，并结合多语言、多事件、异常事件与细粒度分析性查询，首次为多视频检索与生成任务提供统一评价框架。

**🔧 技术方法**

采用多向量密集检索(ColQwen Omni, Qwen3‑VL‑Embedding, OmniEmbed 等)作为第一阶段检索，并用 RankVideo、Gemma 4 等视频原生或零样本重排序器；生成任务使用 CAG、TRACE、MARQUIS 等多模态 RAG 系统，并使用 MiRAGE 指标评估。

**📊 数据集**

基于 MultiVENT 2.0 数据集扩展，构建了新的 MultiVENT Raw（包含 130 事件、222 查询、近 120k 视频，含 8 种语言、异常事件 34 个）以及 23 事件的开发集 MultiVENT Raw Dev。

**📈 对比分析**

通过 Recall@k、nDCG、MiRAGE 等指标评估检索与生成，在所有基线中，OmniEmbed‑mv+RankVideo 的检索性能最优，但整体 Recall@20 仍低于 30%，生成任务的 F1 指标仅在 10% 左右，显示任务仍极具挑战。

**⚠️ 局限性**

主要限制在于原始视频的多模态信息稀缺与噪声，且检索时会混入大量编辑后视频导致负样本多样性过高；生成模型在引用与事实准确性上表现不足，且依赖于人工标注的关键事实有限。

---

## 680. The Past Frames the Future: Memory for Autoregressive Video Generation

**arXiv ID:** 2609.28466 | [PDF](https://arxiv.org/pdf/2609.28466v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 681. Cross-Scale Transfer Learning for Depression Severity Prediction: From PHQ-8 to HAMD-17 Across Languages and Clinical Paradigms

**arXiv ID:** 2609.28430 | [PDF](https://arxiv.org/pdf/2609.28430v1)

**作者:** Wenjie Feng `[一作]`, Satoshi Nakamura `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一个顺序低秩适配（LoRA）协议，使用 Qwen3 LLM 在不同语言、量表和收集范式下进行抑郁严重度的连续评分预测，并在数据稀缺的 PDCH 数据集上实现迁移学习。

**💡 创新点**

首次将抑郁量表从英语 PHQ‑8/avatar‑mediated 迁移到中文 HAMD‑17/真实临床场景，且采用连续评分回归而非传统的二分类，提出了可复现的低秩适配+带界限回归头的训练流程。

**🔧 技术方法**

使用 Qwen3 LLM + LoRA 低秩适配 + 带 Sigmoid 界限的回归头，分析式提示构造，Huber 损失，AdamW 优化器和 cosine 学习率调度。

**📊 数据集**

使用的公开数据集包括 DAIC‑WOZ（189 个英语会话，PHQ‑8）和 PDCH（100 个中文临床会话，HAMD‑17）。

**📈 对比分析**

采用患者级分层 5‑折 2‑重复交叉验证，并与无迁移、无监督、标签打乱、全英文等对照组比较；顺序迁移在 MAE/RMSE/宏 F1 上显著优于无迁移和非 LLM 基线，0.6B 模型 MAE 4.96/6.59/0.36，1.7B 模型 MAE 4.38/5.62/0.46。

**⚠️ 局限性**

局限包括样本量小、单中心、量表与语言不可完全解耦、未评估转录错误、缺乏统计显著性检验，以及仅针对单一源–目标组合的探索性验证。

---

## 682. Context-Continuous Preference Learning for Exoskeleton Personalization

**arXiv ID:** 2609.28427 | [PDF](https://arxiv.org/pdf/2609.28427v1)

**作者:** Sunin Baek `[一作]` (Korea University), Daekyum Kim `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种在不同工作条件下通过共享用户偏好观测实现外骨骼个性化的算法，称为Context-Continuous Preference Learning (CCPL)。

**💡 创新点**

创新点在于将高斯过程偏好模型与上下文连续性结合，使得在相近工作条件下可以共享信息，同时保留每个条件下的独立效用估计，从而在有限反馈下提升学习效果。

**🔧 技术方法**

主要技术包括高斯过程偏好学习、带阈值的Probit似然、基于离散评分和直接比较的反馈融合、偏好式贝叶斯优化以及模拟与实证实验中的多策略对比。

**📊 数据集**

数据集为来自九名健康成人的踝部和肘部外骨骼实验数据，其中踝部实验涉及三种步速，肘部实验涉及五种负载与疲劳剂量组合，共计101次踝部评估和26次肘部评估。

**📈 对比分析**

与独立学习和聚合学习相比，CCPL在仿真和实证实验中均取得更高的重建相关系数和推荐动作的真实效用，且在仅五次曝光下即可达到与独立学习相当的性能，节省约37%（踝部）和17%（肘部）的反馈量。

**⚠️ 局限性**

局限性包括样本量小、实验仅涵盖健康成人、仅进行离线回顾性评估（未检验在线个性化效果）、上下文变量预设且不一定适用于更广泛任务、以及假设偏好连续性可能不成立导致负迁移。

---

## 683. Learning Collective Dynamics with Differentiable Gaussian Representations

**arXiv ID:** 2609.28405 | [PDF](https://arxiv.org/pdf/2609.28405v1)

**作者:** Jianxiang Ma `[一作]` (Northeastern University), Yuesong Hou `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并训练可微分的高斯动力学模型DGD，用以从聚合计数学习群体响应分布、观测函数和反馈过程，并预测未来的集体行为。

**💡 创新点**

① 通过高斯混合表示异质响应倾向；② 在同一潜在倾向上可微分地进行接触-行为聚合；③ 反馈递归将当前响应影响未来状态，实现一体化的联合学习；④ 通过重参数化积分和时间递归实现梯度传播。

**🔧 技术方法**

高斯混合模型、Gauss-Hermite/Sobol节点重参数化积分、负二项/多项式似然、递归反馈状态更新、PyTorch实现、Adam优化与正则化。

**📊 数据集**

KuaiRand-Pure（社交媒体曝光/点击等）与Online Retail II（零售发票、商品属性）等四个真实窗口数据集。

**📈 对比分析**

与DeepAR-joint、GRU预测器及历史基线进行对比；在四个窗口的联合行为负对数似然、Brier分数、行为计数MAE等指标上均优于基线，并在控制实验中成功恢复交叉日响应分布。

**⚠️ 局限性**

模型对高斯混合成分、节点数等超参数敏感；计算复杂度高；仅适用于具备可观测接触与行为的任务，对极端稀疏或高维行为空间的推广有限。

---

## 684. Minimal-Norm Univariate Two-Layer ReLU Classification: Exact Solutions and Global Optimality with Skip Connections

**arXiv ID:** 2609.28438 | [PDF](https://arxiv.org/pdf/2609.28438v1)

**作者:** Karolina Drabik `[一作]` (University of Warsaw), Ranko Lazić `[通讯]` (University of Warwick)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在一维输入下的两层 ReLU 网络，给出了最小参数范数插值器和 ℓ₂ 规范化逻辑回归损失最小化器在函数空间中的完整几何表征，并阐明了隐藏层偏置是否被正则化以及是否存在自由跳跃连接对稀疏性、唯一性和全局最优性的影响。

**💡 创新点**

创新点在于：①提供了最小范数插值器与正则化最优解的完整函数空间几何描述；②证明在存在自由跳跃连接时所有 KKT 点均为全局最优；③揭示小正则化极限与最小范数插值不一致，尤其在偏置不正则化时出现稀疏性差异；④证明偏置正则化导致唯一且稀疏的最优解。

**🔧 技术方法**

主要技术包括：参数范数最小化与 KKT/Clarke 极值分析、函数空间最小表示器理论、凸优化与 ReLU 网络结构的解析、以及对正则化损失的严格极限分析；同时配合数值实验验证理论预测。

**📊 数据集**

使用的数据集为一维二分类合成数据，按标签切分随机采样，满足至少三次标签切换且左右两端标签分别为负/正，覆盖不同复杂度和宽度的实验场景。

**📈 对比分析**

通过将数值实验得到的最优解与理论全局最优解进行对比，验证梯度下降/AdamW 在小正则化下能否收敛到全局最优；实验结果表明跳跃连接消除局部最优，偏置不正则化时会出现稀疏性不足的情况。

**⚠️ 局限性**

研究仅限于一维输入、两层网络和有限宽度；在没有跳跃连接的情形下，KKT 点和 Clarke 极值点的完整描述仍未完成；未将结论推广到多维或更深层网络。

---

## 685. LiMA: Bridging Long-term Imagination to Real-time Dexterous Manipulation via Asynchronous Diffusion

**arXiv ID:** 2609.28431 | [PDF](https://arxiv.org/pdf/2609.28431v1)

**作者:** Ning Chen `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了LiMA，一种异步多尺度生成框架，用于在双手精细操作中分离长时序想象与高频执行。

**💡 创新点**

创新点在于异步双系统架构与潜在 Schrödinger 桥耦合机制，结合长周期战略意图与即时细粒度动作改进，实现低延迟和高成功率。

**🔧 技术方法**

使用了 Diffusion 模型（Dual‑DiT）、潜在 Schrödinger 桥、时空自适应调制、AdaLN、T5‑XXL 语言编码、Wan2.1 视觉 VAE 分词器等技术。

**📊 数据集**

数据集：收集 100 条高质量人类演示，涵盖六个双手操作任务（Stack Cup、Roll T‑shirt、Cook Rice、Make Sandwich、Make Coffee、Assemble Package），并在 OOD 实验中加入四种视觉扰动。

**📈 对比分析**

与 Gr00T N1.6、InternVLA‑a1、VPP、Cosmos‑Policy 四个基线对比；在长时序任务中成功率最高，平均延迟从 600 ms 降至 325 ms，成功率提升约 15–20%。

**⚠️ 局限性**

局限：对严重视觉遮挡或低视觉差异敏感；缺乏触觉反馈；推理延迟仍受硬件限制；未实现模型压缩或并行化优化。

---

## 686. Watch, Recall, Act: Always-On Robots in Concurrent Embodied Streams

**arXiv ID:** 2609.28429 | [PDF](https://arxiv.org/pdf/2609.28429v1)

**作者:** Ding Yi `[一作]`, Xi Lin `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 ARMS，一个在永续流式环境下可同时监视、回忆和执行双臂动作的机器人控制框架。

**💡 创新点**

创新点在于将预训练的 π_0.5 视觉‑语言‑动作 Transformer 与轻量级的感知门控、状态报告和压缩自我历史模块集成，并引入代理因果自我历史日志，实现并发、实时决策。

**🔧 技术方法**

使用技术包括非阻塞异步感知门控、短句式状态报告、压缩记忆检索以及对预训练模型的细调。

**📊 数据集**

数据集为自建的 ARMS Dataset，包含约200 条持续流式双臂遥控演示和1,200 条技能示例，涵盖多种物体及同类干扰。

**📈 对比分析**

与四个专注单一能力的基线对比，ARMS 在在线任务中获得 45% 的整体成功率，高于 28% 的最佳基线，并在 Probe 与 Online 对照下证明主要瓶颈在执行而非感知。

**⚠️ 局限性**

局限性包括实验规模有限、仅在单一 Cobot Magic 平台验证、守则与优先级固定，缺乏更广泛真实世界的数据与学习式守则。

---

## 687. Agent-Editing World Model: Rethinking World Modeling for LLM Agents

**arXiv ID:** 2609.28416 | [PDF](https://arxiv.org/pdf/2609.28416v1)

**作者:** Shuang Sun `[一作]`, Ji-Rong Wen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种Agent-Editing World Model (AEWM)，能够在代理执行前预测决策的后续影响并对其推理-动作对进行编辑，从而消除长序列任务中的“任务状态污染”。

**💡 创新点**

创新点在于：①引入Action Judge预测决策的 Critical/Exploratory/Noisy 三类效果；②State Revision直接编辑代理的推理与动作而非仅预测环境观测；③将编辑与实际执行结合的 EditAct 框架，以及通过 AEWM-RFT 的拒绝采样微调将编辑经验迁移到代理本体。

**🔧 技术方法**

主要技术包括：两阶段训练（mid‑training + SFT）、轨迹合成与标签化、决策判断与编辑模块的联合学习、EditAct 在执行前的决策干预、AEWM‑RFT 的拒绝采样微调以及在不同任务域的跨域评估。

**📊 数据集**

使用了搜索（Search）、终端（Terminal）和软件工程（SWE）三大任务域的数据集，分别为内部 Deep‑Search、CalibForge、DeNovoSWE，以及由 DeepSeek‑V4‑Pro、GLM‑5 等生成的 3,000 例 Action Judge 标注集和 120K 例的中间训练与 SFT 数据。

**📈 对比分析**

与 ReAct、Step‑level Best@3、Trajectory‑level Best@3 等基线在六个基准上进行对比，AEWM 的 Action Judge 在 Macro‑F1 上达 70.5%（比最强基线高 10.6%），EditAct 在所有模型尺度上平均提升 3.2–6.7 分，AEWM‑RFT 在不使用在线 AEWM 指导时比 Self‑RFT 提升 2.2–2.6 分，且显著减少平均回合数。

**⚠️ 局限性**

局限性包括：①对 AEWM 及其训练数据的规模和质量要求高，训练成本大；②在终端与 SWE 域的提升相对有限，主要受限于代理与 AEWM 之间能力匹配；③缺乏对更大规模或更复杂任务域的验证；④在线 AEWM 指导的实时性与延迟问题未被深入探讨。

---

