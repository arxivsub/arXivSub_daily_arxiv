# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-10-19 | 今日论文总数: 715

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Data-Driven Prediction of Optimal MPI Rank Placement and Node Count for Memory-Bound CFD on AMD EPYC Clusters

**arXiv ID:** 2610.23383 | [PDF](https://arxiv.org/pdf/2610.23383v1)

**作者:** Marcin Lawenda `[一作]` (Poznan Supercomputing and Networking Center), Aleksandra Krasicka `[通讯]` (Poznan Supercomputing and Networking Center)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了内存受限CFD在AMD EPYC集群上节点数与MPI进程放置的速度与成本权衡，并提出一种基于最近已测网格的简易预测方法

**💡 创新点**

证明了仅用最近网格的配置和两次有导向的跑测就能在85-90%情况下获得近似最优结果，且优于传统梯度提升树和多项式模型

**🔧 技术方法**

采用基准测试、硬件计数器分析、L3缓存命中率和MPI流量测量、简单统计学评价

**📊 数据集**

使用OpenFOAM motorBike案例的八个不同细化网格（0.17-39.4百万单元），在五种Zen4/Zen5 EPYC处理器上共计560个配置

**📈 对比分析**

通过与梯度提升树、旧多项式模型、随机跑和固定规则等方法对比，发现简易方法的误差仅为3-5%且能正确识别约90%权衡存在的决策

**⚠️ 局限性**

局限在仅覆盖单一CFD案例、节点数最多四个、未考虑运行方差、且对不同类型CFD应用的泛化性未知

---

## 2. Grounded Credit Assignment in Reinforcement Learning for Multi-Turn Search Agents

**arXiv ID:** 2610.23148 | [PDF](https://arxiv.org/pdf/2610.23148v1)

**作者:** Qirui Liu `[一作]` (South China University Of Technology), Yuyu Luo `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

改进了搜索代理的强化学习奖励分配，将检索与回答分开。

**💡 创新点**

创新点在于提出Grounded Credit Assignment，利用可验证的子结果（检索命中与答案正确性）为每个token添加符号一致、受限幅度的残差，从而消除信用混淆并显著提升答案可靠性。

**🔧 技术方法**

技术包括：基于first-hit分段的token分配、margin‑entropy salience加权、受限残差调节、以及在GRPO/GSPO框架下的策略梯度训练。

**📊 数据集**

使用七个问答基准（NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle）并在两种Qwen模型（Qwen2.5‑3B、Qwen3‑30B‑A3B）上进行实验。

**📈 对比分析**

与Search‑R1及其他基线相比，在Qwen2.5‑3B上提升5.7 EM点、在Qwen3‑30B‑A3B上提升6.3 EM点；准确率提升同时将正确答案的检索支持比例从约77%提升至93%以上。

**⚠️ 局限性**

局限在于依赖词典匹配的检索覆盖信号，难以处理语义支持与是非答案；实验仅在固定Wikipedia检索环境和两类模型上验证，未覆盖实时网络搜索或其他模型体系。

---

## 3. Can LLMs predict the outcomes of machine learning experiments?

**arXiv ID:** 2610.22599 | [PDF](https://arxiv.org/pdf/2610.22599v1)

**作者:** Ayush Bansal `[一作]` (Autolab AI), Artem Lukoianov `[通讯]` (Autolab AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了包含288个LLM生成的ML实验想法及其实际效果的数据集，并评估5款前沿LLM预测实验结果的能力。

**💡 创新点**

首次量化LLM对ML实验想法效果的直觉，并公开数据集以推动后续研究。

**🔧 技术方法**

使用提示语言模型预测实验结果，采用R²和斜率评估预测精度，并在不同推理层级和上下文条件下进行消融。

**📊 数据集**

基于8个MLE-bench Lite任务的实验数据，涵盖log loss、ROC AUC、RMSE等指标。

**📈 对比分析**

通过拟合线性回归计算R²和斜率，最佳模型Claude Opus 5.5的R²为0.49，方向预测准确率约80%，但对幅度普遍低估。

**⚠️ 局限性**

单次实验导致噪声、任务规模有限、仅覆盖小型ML任务，缺乏多种随机种子验证以及对更大规模任务的评估。

---

## 4. Immersive Focus+Context Techniques to Assist in Interpretation of High Density Forest Point Clouds

**arXiv ID:** 2610.22878 | [PDF](https://arxiv.org/pdf/2610.22878v1)

**作者:** Spencer O'Keeffe `[一作]` (University of South Australia), Andrew Cunningham `[通讯]` (University of South Australia)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出并实现了一套基于 Focus+Context（F+C）方法的虚拟现实（VR）可视化技术，用于帮助林业工作者更精确地分析高密度森林点云，并通过两轮用户研究（桌面对比 VR）验证其有效性。

**💡 创新点**

创新点包括：① 将 F+C 框架与 VR 环境结合，使用“照明隐喻”设计的三种焦点形状（光晕、聚光灯、柔光）与 egocentric/exocentric 控制方案相结合；② 在焦点区采用高分辨率 LiDAR 点云，在上下文区使用 AI 提取的树木特征，形成分层显示；③ 通过线性混合模型和框架转换统计等多维度评估方法，系统量化 VR 对任务准确率、时间、认知负荷等指标的提升。

**🔧 技术方法**

技术实现主要包括：Unity 3D 引擎、Meta Quest Pro VR 头显（Quest Link 连接）、Xbox 360 手柄控制、LiDAR 点云渲染与 LOD/体素化技术、AI 树特征提取、线性混合模型（R + lme4）分析。

**📊 数据集**

使用的数据集为实测的高密度 LiDAR 森林点云（覆盖数十公顷的多棵树），以及通过 AI 自动提取的树木结构特征（树高、枝干数等）。

**📈 对比分析**

比较方法：在 24 名受试者（均具备基础计算机操作经验）上，分别在桌面和 VR 两种显示模式下完成 24 题（计数、分支、最高树、群集、分枝、镜像）等任务。评估指标包括正确率、计数误差、观察时间、框架转换量、主观认知负荷和 SUS。结果显示：VR 在正确率（0.70 vs 0.64）、计数精度（相对误差 3.58 vs 6.36）、观察时间（32.1s vs 37.1s）以及主观认知负荷上均优于桌面，且受试者更倾向于 VR。

**⚠️ 局限性**

局限性包括：① 样本量仅 24 人，主要为学术人员，缺乏现场林业从业者的真实使用情境；② 只测试了单一硬件配置（Meta Quest Pro + RTX 3080），不一定适用于更低端或更高端设备；③ F+C 技术在点云稀疏或非树木结构场景下的适用性尚未验证；④ VR 仍受帧率、遮挡、光照等渲染瓶颈影响，需进一步优化 LOD/压缩策略；⑤ 研究未探讨长时间使用的舒适度与安全性。

---

## 5. Structuring and Merging Deletion Demands on Crowdsourced Channel Knowledge Maps

**arXiv ID:** 2610.22589 | [PDF](https://arxiv.org/pdf/2610.22589v1)

**作者:** Zihao Ding `[一作]` (South Dakota State University), Jun Huang `[通讯]` (South Dakota State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文研究了基站如何在众包的信道知识图中忘记故障测量，提出了结构化删除单元（SDU）来处理故障，并通过合并多个删除请求来优化执行。

**💡 创新点**

创新点在于提出了结构化擦除与包围执行分区（SWEEP）的方法，该方法能够在不丢弃整个终端或回滚地图的情况下，清除故障测量，并将多个删除请求合并为一次重新测量活动，确保过度删除保持在一定范围内。

**🔧 技术方法**

使用了动态规划和贪婪算法来形成删除请求的组，并通过模拟和USRP测试平台进行验证。

**📊 数据集**

使用了一个模拟的基站在100×100米的城市区域内进行实验，涉及50个终端和多个频段的信道数据。

**📈 对比分析**

与其他方法（如FedRecover和Crab）相比，SWEEP在清除良好测量方面表现最佳，最多只清除2.6%到5.1%的良好测量，而其他方法则高达86.8%。在性能比较中，SWEEP在每种故障类型下都消耗了最少的飞行时间和资源。

**⚠️ 局限性**

限制在于合并操作可能会导致过度删除，尽管有界限，但在某些情况下仍可能影响测量的准确性。

---

## 6. DeDiCa: A Scalable Deconfounding Discrete Causal Generative Model

**arXiv ID:** 2610.23272 | [PDF](https://arxiv.org/pdf/2610.23272v1)

**作者:** Kunchangtai Liang `[一作]` (Saarland University), Isabel Valera `[通讯]` (Saarland University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本工作提出了一种基于张量化因果贝叶斯网络的离散因果推断框架，能够在一次观测数据训练后，精确高效地回答任意可识别的干预查询。

**💡 创新点**

创新点在于将隐藏混杂的离散结构因果模型张量化，结合 join tree 优化实现可扩展的 exact inference，并通过代理可识别性理论保证在隐藏混杂下的准确性。

**🔧 技术方法**

采用张量运算、join tree 结构化推理、最大似然学习、代理可识别性理论以及对隐变量类别数的低秩估计等技术。

**📊 数据集**

在亚洲、Sachs、Child 与 Ecoli70 四个半合成数据集上进行实验。

**📈 对比分析**

与最先进的离散因果生成模型、因果贝叶斯网络、因果流和因果基础模型比较，训练一次即可在 CPU 上实现数百倍甚至数万倍的推理速度，且 ATE 估计误差与或优于对手。

**⚠️ 局限性**

局限在于需要预先估计隐藏混杂变量的类别数、张量维度可扩展性受限，以及仅适用于可通过代理可识别的查询，无法处理部分不可识别或反事实推理。

---

## 7. Rethinking Magnitude Compression in Spectral Losses for DDSP Parameter Estimation

**arXiv ID:** 2610.22962 | [PDF](https://arxiv.org/pdf/2610.22962v1)

**作者:** Byunghoo Park `[一作]` (University of Illinois Urbana-Champaign), Minje Kim `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

探究可微数字信号处理模型中幅度压缩对谱损失优化的影响，并提出无训练的 concordance 指标来预测压缩是否有利。

**💡 创新点**

引入 concordance 作为衡量损失对参数空间排序能力的无监督指标，揭示信息区域与压缩的匹配关系，并验证在三种模型中的表现。

**🔧 技术方法**

使用可微数字信号处理、线性/对数/混合谱损失、Adam 优化、MFCC 评估、MUSHRA 听力测试等技术。

**📊 数据集**

使用 EMT 140 板冲击响应、两振荡器减法合成器音符、Moog Minitaur 采样以及对应的实测回声记录等数据集。

**📈 对比分析**

在三种模型上对线性、对数和混合损失进行比较，评估指标为 MFCC 距离、参数 NMSE 和听力评分；结果表明线性损失在信息区域为高能时表现最佳，混合损失无优势。

**⚠️ 局限性**

只考察了两端压缩（线性/对数）未探索中间指数；结论依赖于所选模型，未给出自动化选择压缩程度的方法。

---

## 8. Beyond What Is Visible: Context-Aware Sentiment Across Online Communities Among Crisis Events

**arXiv ID:** 2610.22551 | [PDF](https://arxiv.org/pdf/2610.22551v1)

**作者:** Zihui Ma `[一作]` (Emory University), Gregory B. Baecher `[通讯]` (University of Maryland)

**通讯引用:** 5317 | [OpenAlex ID](https://openalex.org/A5018083885)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种跨城市、跨社群的自适应学习框架，用来在社交媒体参与度不均的灾害情境下，生成更具代表性与上下文感知的情感预测；

**💡 创新点**

创新点在于三层融合：①将社交文本与聚合移动信息融合生成多模态表征；②利用社会经济与风险信息学习跨城市相似度；③以相似度加权的数据迁移对稀缺社区进行自适应补齐；

**🔧 技术方法**

使用Transformer编码器（如RoBERTa）、多模态融合网络、基于相似度的排名学习与加权迁移策略；

**📊 数据集**

数据来源包括X（Twitter）灾害相关帖子、Veraset聚合移动数据、ACS社会经济特征以及FEMA风险指数，构成两大案例：2025年南加州山火与2024年佛罗里达海尔恩飓风；

**📈 对比分析**

与VADER、RoBERTa、LLM（Mistral）以及不同组合（纯文本、全局、城市特化、融合）进行对比，城市特化融合模型在两场灾害中取得最高宏观F1（火灾0.5923，飓风0.7615），且在低数据社区的表现提升显著；

**⚠️ 局限性**

局限包括：①仍依赖社交媒体文本，可能遗漏未参与的居民视角；②移动数据为聚合级别，无法精确反映个体行为；③仅在两场灾害、城市层面验证，缺乏更细粒度和多地区的泛化；

---

## 9. What LLMs Do in the Shadows: Reading Context Reliance from the Unembedding Dark Space

**arXiv ID:** 2610.23182 | [PDF](https://arxiv.org/pdf/2610.23182v1)

**作者:** Alexandra Kuleshova `[一作]` (Applied AI Institute), Alexey Zaytsev `[通讯]` (Applied AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大语言模型中未嵌入矩阵的暗空间（dark space）对上下文利用的作用，并基于此提出了 Dark Lens 作为轻量级的虚假内容检测器。

**💡 创新点**

创新点在于发现暗空间不仅是 BOS 关注点，还能主动介导中间层的上下文检索；提出 Dark Matter Ratio（DMR）直接从残差向量测量上下文依赖，并将其与参数知识得分结合，实现单前向传播的幻觉检测。

**🔧 技术方法**

技术包括：对未嵌入矩阵做奇异值分解获取暗空间；投影残差到暗空间计算 DMR；使用参数知识得分（PKS）衡量参数记忆影响；在单次前向传递中融合两者生成幻觉分数。

**📊 数据集**

使用三类基准：MS MARCO（长文本问答）、CNN/DM（摘要）与 XSum（极端摘要），模型包括 Mistral‑7B、LLaMA‑2‑7B/13B、LLaMA‑3.1‑8B、Qwen3‑8B 与 gemma‑3‑4b。

**📈 对比分析**

与 10+ 传统方法（如 ReDeEP、LUMINA、LN‑Entropy、MaxProb 等）对比，Dark Lens 在所有模型上均达或超过最高 AUROC，尤其在 MS MARCO 与 XSum 上表现尤为突出；且不需要额外的 attention 采样或额外前向传播。

**⚠️ 局限性**

局限性包括：只能检测模型是否忠实于给定上下文，不能判定上下文本身是否错误；需要模型内部残差访问，无法用于闭源或黑盒模型；暗空间在神经电路层面的具体机制尚未完全阐明。

---

## 10. Rethinking User Encoders: Routing Diffusion States for Scenario-Adaptive User Representation

**arXiv ID:** 2610.22400 | [PDF](https://arxiv.org/pdf/2610.22400v1)

**作者:** Junru Zhang `[一作]` (Zhejiang University), Cheng Peng `[通讯]` (Zhejiang University)

**通讯引用:** 31674 | [OpenAlex ID](https://openalex.org/A5007046565)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于扩散语言模型的用户表示框架，利用多粒度去噪重建空间和可学习的元槽实现场景自适应的用户嵌入；

**💡 创新点**

创新点在于将扩散模型的迭代去噪视为多层语义重建空间，设计了跨模态与模态专属的可学习元槽，并通过残差路由器根据场景查询动态强化相关槽；同时构建了行为问答语料库作为监督，显著提升了场景适应性；

**🔧 技术方法**

使用了掩码扩散语言模型（masked diffusion）、多槽因子化、残差路由机制、对比学习以及Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1作为基座模型；

**📊 数据集**

基于某大型支付平台约4.1 B条匿名滚动用户记录，包含六种异步行为源，构建了十个二分类业务场景的数据集；

**📈 对比分析**

与BERT‑style双向编码器、AR模型、传统dLLM以及无路由版本等多种基线进行对比；在十个工业场景和线上A/B测试中，均显著优于所有基线，在AUC、KS、召回等指标上提升1–3个百分点；

**⚠️ 局限性**

仍存在对极低维度/稀疏行为鲁棒性不明、对实时动态更新支持有限、计算成本较高以及对多语言/文化场景泛化的验证不足等限制。

---

## 11. Sample, Mix, or Verify? Allocating Test-Time Compute under Imperfect Verifiers

**arXiv ID:** 2610.23255 | [PDF](https://arxiv.org/pdf/2610.23255v1)

**作者:** Srinivasan Iyengar `[一作]` `[通讯]` (Microsoft Research), Srinivasan Iyengar (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了 SPLIT，一种基于小规模标注样本的测试时计算分配器，结合精确成功概率公式来决定抽样数量、模型组合与验证调用，以提升语言模型与智能体的最终准确率。

**💡 创新点**

提出了精确的成功概率公式，兼顾任务难度、生成器间共盲区、验证器误差相关性，并通过经验贝叶斯从 pilot 校准，形成可泛化的预算规划；同时证明了误差相关性对决策的重要性，阐释了复核误判提升误差的机制。

**🔧 技术方法**

采用经验贝叶斯估计、Beta‑Binomial 与非参数 NPMLE 难度先验、Gaussian copula 估计生成器相关、贝塔分布模型复核阈值、精确成功概率解析、在线再规划与早停策略等技术。

**📊 数据集**

使用了 MATH、CodeContests、HumanEval、MBPP、MATH‑500、GPQA、SWE‑Bench Verified（代理回合和系统池）等公开数据集及其生成样本与验证结果。

**📈 对比分析**

与固定经验规则（如“使用最强模型”“按算力分配”等）以及指数分配基线对比；在七大基准上，SPLIT 在未见任务的预算上与最优配置误差仅 0.02–0.8 点，固定规则至少落后 2.5 点；在多种成本模型与在线再规划设置下仍保持 0.45–0.62 点的低误差，显示出良好的泛化能力。

**⚠️ 局限性**

仅做点预测，长时间同一代理抽样实验受限；pilot 必须代表真实部署；未建模多模型间近似重复答案的相关性；成本模型简化，未考虑并行采样与真实价格；模型选择在不同验证器下不一定最佳；理论假设在某些依赖不满足；在线策略为贪心，未证明最优。

---

## 12. Kingfisher: Fast Personalized Visual Autoregressive Generation via Scale-Aware Prompt-Conditioned Scale Decomposition

**arXiv ID:** 2610.23238 | [PDF](https://arxiv.org/pdf/2610.23238v1)

**作者:** Dinh-Khoi Vo `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于冻结的视觉自回归模型，对早期尺度代码进行内容与风格的分解与重组，实现了无模型微调的快速个性化图像生成。

**💡 创新点**

提出了基于提示的尺度子空间分解方法，将早期尺度代码拆解为内容、风格与残差三部分，并利用位级自我校正完成后续细化，首次在自回归框架下实现无参数更新的个性化。

**🔧 技术方法**

采用Infinity视觉自回归模型、提示条件尺度子空间、基于提示的尺度代码分解与重组、位级自回归预测与自我校正技术。

**📊 数据集**

在CSD-100数据集上构造的交叉图像对（共500对）进行实验验证。

**📈 对比分析**

与Diffusion（DreamBooth‑LoRA、B‑LoRA、UnZipLoRA、Inspiration Tree）以及VAR（CSD‑VAR）基线比较，Kingfisher在内容保持、风格匹配、文本一致性等指标均优于其它方法，并在准备时间上比Diffusion快54倍、比VAR快7.5倍，且在人工评估中获得最高偏好率（42.3%）。

**⚠️ 局限性**

仍依赖冻结模型且仅聚焦内容‑风格两因素，难以覆盖更复杂的多因素个性化；分解效果受提示子空间质量影响；对极端视觉特征或多目标场景的适应性有限。

---

## 13. ScaleRAN: A Runtime for Elastic GPU Resource Management in AI-RAN

**arXiv ID:** 2610.23072 | [PDF](https://arxiv.org/pdf/2610.23072v1)

**作者:** Arthur Hamard `[一作]` (Nokia Bell Labs), Abdelkader Outtagarts `[通讯]` (Nokia Bell Labs)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 ScaleRAN 运行时，用于在 AI‑RAN 平台上按时隙粒度动态弹性地分配 GPU SM（流多处理器）资源，同时提供 LDPCSim 生成器用于可复现的 RAN 工作负载评估。

**💡 创新点**

创新点包括：1) 利用 NVIDIA Green Contexts 实现可在时隙级别切换的细粒度 SM 分区；2) 结合响应式 K8S 风格控制与基于分位数回归+一致性预测的主动调度策略，显著提高 RAN 时延合规性；3) 将该运行时与 O‑RAN O‑Cloud 体系结构无缝对接，并以轻量级共享内存和容器化实现本地化控制。

**🔧 技术方法**

技术细节：Green Contexts、MPS、CUDA 流、共享内存 IPC、Kubernetes HPA 触发器、LightGBM 分位数回归、一次性分割一致性校准、O‑RAN O1/O2 接口、Docker Engine API 与 MPS 服务器。

**📊 数据集**

使用 LDPCSim 生成的训练数据集（105 条轨迹×3000 时隙）和评估数据集（8 条轨迹×10000 时隙），包含 PRB、MCS、MIMO 层、编码比特数、LDPC 基图等无线描述符，以及不同 SM 分配方案下的解码时延。

**📈 对比分析**

对比方法包括：K8S‑React（基于上时隙延迟的比例控制）、K8S‑Proact（用预测代替测量）、Proactive（直接按预测选择 SM），并进一步细分为 Mean、Quantile（P90/P95/P99）和 Conformal Quantile。实验结果显示，主动策略（尤其是 Conformal P99）在保持 99.96% 的时延合规率的同时，平均 SM regret 降低约 45%，显著优于仅响应式控制。

**⚠️ 局限性**

局限性：目前仅管理 SM 分配，未考虑设备内存等资源；仅在单一 NVIDIA DGX Spark 上验证，未覆盖多 GPU 或更大规模部署；需要对 AI 框架做少量代码包装，不能完全透明；对超过时隙持续时间的 AI 内核仍可能产生时序漂移；多 AI 工作负载的扩展仍在研究中。

---

## 14. ODSS-Net: Orthogonally Decoupled Shared-Specific Representations for Multi-Modal Object Re-Identification

**arXiv ID:** 2610.22922 | [PDF](https://arxiv.org/pdf/2610.22922v1)

**作者:** Zhifeng Ma `[一作]`, Minwen Deng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 ODSS-Net 框架，解决多模态 ReID 的共享-特定表示学习与样本自适应融合。

**💡 创新点**

创新点：正交解耦共享-特定特征、几何正交投影去除共享残差、样本自适应混合专家融合。

**🔧 技术方法**

技术：共享结构特征构造 (SSFC)、正交特征学习 (OSFL) 与几何正交投影、样本自适应混合专家 (SAMoE)、多头自注意力、CLIP ViT backbone 等。

**📊 数据集**

使用数据集：RGBNT201、RGBNT100、MSVR310。

**📈 对比分析**

与 8+ 先进方法对比，RGBNT201 上 mAP 82.1%、Rank‑1 87.1%，在所有指标均名列第一；RGBNT100 mAP 88.1%，MSVR310 mAP 54.0%。

**⚠️ 局限性**

局限：对极端光照或遮挡下的 TIR/RGB 仍有挑战，缺少跨域泛化实验；模型复杂度较高。

---

## 15. Revisiting Structure-Based Filtering for AI-Designed Antibodies: Opportunities, Limits, and What Matters

**arXiv ID:** 2610.23136 | [PDF](https://arxiv.org/pdf/2610.23136v1)

**作者:** Langzhang Liang `[一作]` (AI³ Institute, Fudan University), Tianlei Ying `[通讯]` (Genesius Lab, Shanghai Innovation Institute)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究如何利用结构预测（AlphaFold 3、Protenix‑v2、ESMFold2）的输出，对人工智能生成的抗体候选进行后期筛选，以降低实验筛选成本。

**💡 创新点**

创新点在于提出两种互补的筛选指标——接口区域对比（RIC）和非CDR抗体置信度（NAC），并系统证明：接口信号在跨抗原比较中最有效，非CDR置信度在同一抗原内部排序时表现更佳；同时发现非CDR置信度优于全抗体pLDDT，揭示了此前被忽视的过滤信息来源。

**🔧 技术方法**

主要技术包括：多种结构共折叠预测模型、从预测结果中提取坐标、接触概率、PAE、pLDDT等置信度张量；构造RIC和NAC两种统计指标；使用基于阈值的 utility 函数（结合召回率、FPR 与实验预算），在跨抗原和同抗原两种筛选场景下对指标进行优化与比较。

**📊 数据集**

数据集由 4,184 只实验验证的抗体‑抗原对组成，来源包括：1) 7 抗原的显示筛选（yeast/phage）得到 1,400 只候选；2) 7 抗原的 IgDesign SPR 复测得到 1,171 只；3) 165 抗原的文献/专利直接亲和力测定得到 1,613 只。所有样本均通过预测生成 10 种随机种子、每种 5 个采样，产生多重结构预测。

**📈 对比分析**

比较方法：在两种阈值优化策略下（跨抗原全局阈值与同抗原特定阈值），计算召回率、FPR、utility、精确度等指标。结果显示：RIC 在跨抗原场景下实现最高 recall 与 utility；NAC 在同抗原场景下表现最佳，且在 5%/20% 预算下召回率可达 35‑39%。整体表明两种指标均显著优于传统接口或全抗体置信度基准。

**⚠️ 局限性**

局限性：① 数据集包含非 AI 设计的抗体（显示筛选和文献收集），可能与纯 AI 生成的候选在序列/结构分布上不同；② 绑定标签来自不同实验体系（SPR、显示富集、文献测定），存在标注不一致与误差；③ 预测模型仅使用 10 种种子和 5 个采样，可能不足以捕获全部预测不确定性；④ 预测器训练集仅截止至 2021 年，无法评估最新模型的潜在提升；⑤ 由于结构预测本身的辨识限度，无法突破模型固有的区分能力。

---

## 16. X-MDD: LLM-Based Mispronunciation Detection and Diagnosis for Unseen Languages with Canonical References

**arXiv ID:** 2610.23079 | [PDF](https://arxiv.org/pdf/2610.23079v1)

**作者:** Xintong Wang `[一作]` (National University of Singapore), Ye Wang `[通讯]` (National University of Singapore)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出X-MDD多语种LLM框架，利用规范语音和音素参考实现未训练语言的误发音检测与诊断

**💡 创新点**

创新点在于将规范语音与音素共同作为输入，并通过在推理时使用示例进行自适应，同时采用GRPO强化学习优化检测F1和电话错误率，实现跨语言迁移

**🔧 技术方法**

技术包括多模态LLM（Qwen3-ASR-1.7B）+音频Transformer、文本编码器、示例提示（ICL）以及GRPO（Group Relative Policy Optimization）强化学习

**📊 数据集**

训练数据：汉语（AISHELL-3、MagicHub、LATIC），英语（TIMIT、SpeechOcean762、L2-ARCTIC），评估数据：汉语900句、英语900句、阿拉伯语QuranMB.v2测试集

**📈 对比分析**

与单语与多语MDD基线对比，X-MDD在汉语/英语上的F1分别提升约6.4%和2.9%，PER和DER均下降；在未见的阿拉伯语上F1为38.07，略低于已训练基线的44.14

**⚠️ 局限性**

仅在阿拉伯语上评估，受限于数据量和口音样本不足，导致召回率和诊断准确率仍偏低，且缺乏更广泛语言和口音的验证

---

## 17. TriFold: Folding irregular 3D Gaussians into structured latents for 3D Scene Generation

**arXiv ID:** 2610.23050 | [PDF](https://arxiv.org/pdf/2610.23050v1)

**作者:** Yun Wang `[一作]`, Dapeng Wu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 TriFold，一种针对原生 3D Gaussian 场景的 VAE，将可变数目且分布不均的高斯集合映射到固定结构的三平面（triplane）潜在空间，并通过 occupancy 预测和基于 set 的解码器重建本地可变高斯集合；随后在该潜在空间上训练流匹配模型，实现点云条件生成与单图像条件的无模态（amodal）生成。

**💡 创新点**

创新点包括：① 在编码阶段使用每个三平面单元的多查询聚合和跨平面注意力，保留投影到同一单元的多高斯信息；② 在解码阶段采用 occupancy 预测 + set 解码器，结合最优传输（OT）实现可变数量高斯的自适应生成；③ 将结构化的 triplane 潜在空间与流匹配相结合，兼顾高效生成与保持原始高斯属性。

**🔧 技术方法**

主要技术手段有：PTv3 点变换器提取高斯特征；多查询聚合（QFormer）和跨平面注意力；变分瓶颈生成 triplane 潜在；DETR 风格 set 解码器；最优传输匹配监督；渲染监督与占用正则化；流匹配模型进行条件生成。

**📊 数据集**

使用 InteriorGS（大规模室内 3D Gaussian 场景）和 Aria Synthetic Environments（ASE）子集（SceneSplat-49K）进行训练和评估，分别提供约 19,600 个裁剪样本和 8,600 个完整场景。

**📈 对比分析**

在重建任务中，与 Can3Tok、DiffGS、GaussianGPT 等基线相比，TriFold 在 PSNR、SSIM、LPIPS 和 Chamfer Distance（CD）上均取得显著提升（PSNR +5.15 dB，SSIM +0.057，LPIPS -0.026，CD -0.39）；在单图像条件生成中，未观测区域的 FID 为 67.81，覆盖率 CD 达 85.24%，展示了较好的无模态生成能力。

**⚠️ 局限性**

局限性包括：① 对极高密度场景的解码仍受 K 上限限制，可能需要增大 K；② 训练与推理依赖大量 GPU 资源（单 GPU batch=1），在规模化部署上仍有挑战；③ 目前仅在室内场景上验证，跨域（如户外）性能尚未充分评估。

---

## 18. Understanding LLM Hallucinations Caused by Soft Errors

**arXiv ID:** 2610.22520 | [PDF](https://arxiv.org/pdf/2610.22520v1)

**作者:** Jinghao Wen `[一作]` (Villanova University), Xun Jiao `[通讯]` (Villanova University)

**通讯引用:** 1270 | [OpenAlex ID](https://openalex.org/A5016146461)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究软错误对大型语言模型（LLM）推理过程的影响，并通过开发高效的错误注入框架，对不同规模、不同架构的LLM在数学推理和代码生成任务中的幻觉行为进行系统实验。

**💡 创新点**

①首次从硬件层面揭示软错误会诱发LLM幻觉；②提出四类软错误诱发的幻觉分类；③构建可扩展的软错误注入工具；④对模型、组件、任务层面进行细粒度鲁棒性评估。

**🔧 技术方法**

基于PyTorch的位级错误注入框架（SoftErrorInjector），采用torch.view+bitwise操作实现高效注入；使用多比特错误模拟SRAM软错误；对Transformer各层进行错误映射。

**📊 数据集**

AIME 2024（数学推理）与LiveCodeBench（代码生成）数据集，分别评估数学和编程任务。

**📈 对比分析**

采用Pass@1指标，分别在不同比特错误率（10^-12~10^-4）下对模型性能进行对比，结果显示小错误主要导致逻辑错误，较大错误导致Token级崩溃；模型规模与架构对鲁棒性影响不一，代码生成任务对错误更敏感。

**⚠️ 局限性**

仅考虑SRAM位翻转错误，未涵盖其他硬件故障；实验仅在GPU服务器上模拟，未验证真实硬件环境；未给出对应的误差缓解方案；模型与任务选择有限，可能不具代表性。

---

## 19. SpikeQ: Spike-Guided Post-Training Quantization for Spiking Neural Networks

**arXiv ID:** 2610.23046 | [PDF](https://arxiv.org/pdf/2610.23046v1)

**作者:** Kai Sun `[一作]` (Monash University), Levin Kuhlmann `[通讯]` (Monash University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种基于突触脉冲误差的后训练量化框架 SpikeQ，用于在神经形态芯片上将 SNN 的权重和膜电位压缩到低位宽（2‑4 位）而保持高准确率。

**💡 创新点**

创新点包括：① 发现量化误差对 SNN 性能的关键因素是翻转脉冲数量而非 MSE；② 在权重量化阶段采用最小化脉冲误差的离散搜索；③ 在膜电位量化阶段构造阈值锚定的预发量化网格，使阈值成为舍入边界，并根据重置规则匹配网格；④ 通过脉冲计数、边距与总量损失三项联合目标实现无梯度的高效量化。

**🔧 技术方法**

技术细节包括：离散搜索权重量化、阈值锚定与重置感知的膜电位量化网格、脉冲计数/边距/总量损失多项式目标、无梯度后训练、使用小量化校准集进行快速搜索。

**📊 数据集**

数据集与模型：CIFAR‑10/100、ImageNet、PASCAL VOC 2012、ADE20K、COCO；SNN 结构包括 PreAct‑ResNet‑18、SEW‑ResNet‑18、SDT‑8‑384、SDT‑V2、EMS‑YOLO 等多种深度残差与 Transformer‑style 网络。

**📈 对比分析**

与 RTN、FlowQ、GPTQ、SBC、GPTAQ、BRECQ、PTQ4SNN（权重）以及 MinMax、MSE、FlowQ‑M、SQUAT、PTQ4SNN‑M（膜电位）等基线比较。SpikeQ 在 2‑bit 权重量化时在 7 个基准中超过 6 个获得最高或第二高准确率；在 4‑bit 时误差不超过 2.59 个点。联合 4‑bit 权重+4‑bit 膜电位可在保持 2.59 点以内的准确率的同时显著压缩模型体积并降低能耗。

**⚠️ 局限性**

局限性：① 在 2‑bit 膜电位量化时，预发量化在某些模型（如 SEW‑ResNet‑18）上性能下降；② 对软重置模型的预发量化效果不及后重置；③ 对极低位宽（2 位）在复杂任务上仍会导致显著准确率下降，需使用 4 位；④ 仍需针对不同层/通道进行网格搜索，计算开销不为零；⑤ 未覆盖可变时间步长或可训练量化的场景。

---

## 20. Graph Serializations for Reasoning LLMs: A Study Across Knowledge Extraction and Structural Reasoning

**arXiv ID:** 2610.22734 | [PDF](https://arxiv.org/pdf/2610.22734v1)

**作者:** Julien Yu `[一作]` (Oracle), Katrin Kirchhoff `[通讯]` (Oracle)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对15种不同的图序列化格式在前沿LLM（共9种配置）下进行系统评估，比较其在本体知识抽取和结构图推理任务中的效果，并提出基于查询的自适应序列化选择器；

**💡 创新点**

①首次在前沿LLM上大规模对15种语义网、约束、自然语言、表格、图形语言、紧凑编码和散文等序列化方式进行对比；②将图序列化视为模型、任务、成本相关的动态决策；③设计并验证了一个基于查询的自适应选择器，能显著提升性能；

**🔧 技术方法**

使用多种前沿LLM（gpt‑5.4、gpt‑5.4‑mini、o3、claude‑sonnet‑4‑6、claude‑haiku‑4‑5、grok‑4‑fast 等），配合 Text2KGBench、GraphQA（原版与扩展）、Wikidata 子图等数据集；构建并行评分与选型框架，评估 token 预算、推理成本与延迟；

**📊 数据集**

Text2KGBench（19 个本体、2014 句测试集）、GraphQA（Erdős‑Rényi、Barabási‑Albert、SBM、star、complete、path 等，含 Hardened 与 Topology 扩展）、Wikidata 10 个种子实体的两跳子图（共 110 道问答）；

**📈 对比分析**

在相同模型、相同指令、相同实体 ID 下，测量 F1/准确率并与固定最佳格式、全局固定、投票等基线对比；知识抽取任务格式差距 1.3–3.9pp；结构推理任务格式差距 4.2–28.4pp；自适应选择器比固定最佳提升 9.52pp（最高 76.19%），同时展现 token 效率差异；

**⚠️ 局限性**

仅评估文本序列化，未覆盖图原生方法、学习图标记、工具或受限解码；数据集有限（仅 Text2KGBench、synthetic GraphQA、两模型 Wikidata 子图）；未考察检索质量、上下文更大、时间/概率图、检索错误、开放式 KGQA；15 种手工设计格式并非穷尽；所有模型为闭源商用，无法评估预训练对序列化的影响。

---

## 21. GOLD: Geometric Optimized Latent Diffusion for Structure-Aware RNA Inverse Folding

**arXiv ID:** 2610.22992 | [PDF](https://arxiv.org/pdf/2610.22992v1)

**作者:** Qi Si `[一作]` (Shanghai Academy of Artificial Intelligence for Science), Yuan Cheng `[通讯]` (Shanghai Academy of Artificial Intelligence for Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GOLD，一种将RNA序列与侧链角度映射到统一SO(2)-等变潜在空间的几何感知潜在扩散框架，用于逆向折叠。

**💡 创新点**

创新点包括：显式引入侧链角度作为几何约束；将序列与侧链几何联合编码为连续潜在空间；采用训练自由的梯度引导采样实现物理一致性；结合RNA基础模型对潜在空间进行对齐。

**🔧 技术方法**

使用技术包括：SO(2)-等变GeoVAE、潜在扩散模型（LDM）、几何向量感知器（GVP）、能量引导梯度（梯度引导采样）、与RNA基础模型的对齐。

**📊 数据集**

使用数据集：内部GOLD Benchmark（7583训练/1144验证/598测试）和CASP15 RNA子集做OOD评估。

**📈 对比分析**

与PiFold、RDesign、gRNAde、RiboDiffusion、SOLD等基线对比，采用序列恢复、二级结构恢复、AlphaFold3/RhoFold RMSD等指标。GOLD（Dual-Guided）在所有指标上均优于基线，尤其在序列恢复、二级结构准确率和3D RMSD上显著提升。

**⚠️ 局限性**

局限性：采样时需计算物理梯度，导致推理时间增加；目前仅支持固定骨架逆折叠，未扩展到可变骨架共设计或多态构象。

---

## 22. A Stronger Teacher Within: Strengthening MoE Models via On-Policy Bidirectional Distillation

**arXiv ID:** 2610.23185 | [PDF](https://arxiv.org/pdf/2610.23185v1)

**作者:** Haoyu Chen `[一作]` (Beihang University), Wenge Rong `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 MoEvolve 框架，通过在 MoE 语言模型内部将激活预算从 k 扩展到 k'，利用 on-policy distillation 恢复专家协同，随后使用强化学习进一步提升扩展模型的能力，最终再通过反向 distillation 将改进后的能力迁移回原始的 k 激活配置，从而在不增加推理成本的前提下显著提升模型性能。

**💡 创新点**

创新点在于：①构造内部强教师——利用激活扩展在同一模型内部生成更高容量的教师；②双向 on‑policy 优化——先恢复扩展模型，再用 RL 解锁潜能，随后逆向 distill，完成自我提升；③无需外部教师或手工特征，完全依赖模型自身的稀疏专家结构。

**🔧 技术方法**

技术手段包括：Mixture‑of‑Experts (MoE) 结构、On‑Policy Distillation（OPD 与 ExOPD）、强化学习（SAPO 与 Soft Adaptive Policy Optimization）、Expert Cooperation Restoration Distillation (ECRD)、Capability Unlocking via Reinforcement Learning (CU)、Reverse Distillation for Capability Transfer (RD)。

**📊 数据集**

使用的数据集：数学任务训练集 OpenMathInstruct‑2 与 DeepMath‑103K；代码生成训练集 Eurus‑2‑RL‑Data；科学知识训练集 MMLU 辅助集；评测基准包括 AMC23、GSM8K、MATH500、Minerva Math、PolyMath‑Low/Medium、AIME‑2024/25/26、HumanEval、MBPP、LiveCodeBench‑v6、GPQA、ARC‑Challenge、MMLU‑STEM 等。

**📈 对比分析**

与 SFT、RFT、RL（SAPO）和 OPSD（自我 distillation）等基线进行对比；在四个 MoE 后端（OLMoE‑1B‑7B、DeepSeek‑V2‑Lite、GPT‑OSS‑20B、Qwen3‑30B‑A3B）上，MoEvolve 在数学平均分提升至 +22.2 点、ScienceQA +12.3 点、代码平均 +7.0 点，且超过相同预算 RL 基线最高可 +6.5 点；逆向 distillation 后的低 k 模型甚至可超越扩展教师。

**⚠️ 局限性**

局限性包括：未在更大规模 MoE（如数百亿参数）上验证；未测试开放式生成、代理任务或长上下文场景；方法需额外训练步骤，无法在训练‑无关环境下直接解锁专家容量。

---

## 23. Repairing LLM-Generated Specifications with Deterministic Diagnosis

**arXiv ID:** 2610.22568 | [PDF](https://arxiv.org/pdf/2610.22568v1)

**作者:** Giovanni Vaccarino `[一作]` (University of Oxford), Sara Bernardini `[通讯]` (University of Oxford)

**通讯引用:** 1216 | [OpenAlex ID](https://openalex.org/A5016467897)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DRAGO框架，利用LLM生成后通过确定性、无参考诊断和证据门控修复自然语言到正式规范的翻译。

**💡 创新点**

创新点是将诊断改为基于可观测执行和形式特定规则的确定性检查，并使用接受门控避免错误修复。

**🔧 技术方法**

技术包括LLM生成/修复、形式特定诊断器、证据门控接受策略和多形式统一接口。

**📊 数据集**

使用了三种形式（SMT‑LIB2、MiniZinc、PDDL）的五个基准（FOLIO、ProverQA‑hard、Text2Zinc、CP‑Bench、Planetarium）。

**📈 对比分析**

与零射击和LLM诊断修复比较，DRAGO在四个模型上平均提升12.9%准确率，回归率降至1.3%（相较11.8%），优于同域最先进系统。

**⚠️ 局限性**

局限在于只测答案准确性，未覆盖语义等价；执行无障碍的欠约束问题仍难诊断；需要为新形式制定诊断规则。

---

## 24. MetaCurate: Evolving Curation Filters for High-Quality Agentic Coding Data

**arXiv ID:** 2610.23134 | [PDF](https://arxiv.org/pdf/2610.23134v1)

**作者:** Donghoon Kim `[一作]`, Wonjun Kang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练的框架MetaCurate，用来通过LLM的成对判断迭代改进SFT数据的轨迹筛选过滤器；

**💡 创新点**

创新点在于不需要在搜索循环中训练模型，而是利用LLM的相对偏好来评价和优化筛选规则、评判标准和聚合策略，使筛选器可执行、可解释且可扩展；

**🔧 技术方法**

技术实现包括：可执行的过滤器（规则文件、LLM评估准则、聚合策略）、成对LLM判断、迭代提议-评估-推广循环、对规则冗余和评分分布的诊断，以及噪声感知的升级决策；

**📊 数据集**

使用了两套数据集：4,286条自制编码代理轨迹和17,220条SWE‑SFT轨迹；

**📈 对比分析**

与随机抽样、手工规则过滤器和在搜索循环中训练模型的Baseline对比，MetaCurate在两套数据集上均取得最高平均下游SFT性能（分别为64.0%和56.5%），超过基线2.3–3.6个百分点；

**⚠️ 局限性**

局限性在于缺乏对演化过滤器规则的深入分析，结果主要验证了筛选效果而非因果机制；对其他领域或更大规模数据的通用性尚未验证；

---

## 25. Prune the Leaky Neurons: Mitigating Private Data Leakage in Fine-Tuned LLMs via Joint-Embedding-Guided Pruning

**arXiv ID:** 2610.22792 | [PDF](https://arxiv.org/pdf/2610.22792v1)

**作者:** Mohamed Shaaban `[一作]` (Washington State University), Mohamed Elmahallawy `[通讯]` (Washington State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在已有的细调后，提出一种基于 PII‑free 参考模型的后置神经元剪枝框架，识别并去除在已细调 LLM 中导致 PII 泄露的神经元，保持模型的域知识与推理性能。

**💡 创新点**

创新点在于：①利用同一基础模型训练的 PII‑free 公共模型（通过 surrogate 与 masked 视图及 JEPA 目标）作为参考；②通过交叉模型的余弦距离探针量化泄露记录与参考模型间的表示差异；③使用积分梯度和尺度不变的特异性得分定位 PII‑响应神经元；④不需要重新训练或更改推理接口，即可直接剪枝。

**🔧 技术方法**

技术细节包括：LoRA 参数效率细调、JEPA 预测目标、surrogate 与 masked PII 视图生成、交叉模型余弦距离探针、积分梯度归因、特异性评分、按频率与阈值选择剪枝神经元。

**📊 数据集**

实验使用三类 PII 数据集（ECHR、Yelp Reviews、ai4privacy）以及五款公开 LLM（Llama‑3.2‑1B/3B/8B、Gemma‑2‑2B、Mistral‑7B）。

**📈 对比分析**

与未防护模型、DEPN、随机剪枝进行对比，评估指标包括 PII 提取准确率、推理准确率、成员推断 AUC、未见文本困惑度（PPL）。实验表明：剪枝 1.6% 神经元可将推理准确率降低 7.4×、提取召回率降低 13.9×，并将未见文本 PPL 降低不到 16%（大部分模型）。DEPN 在隐私方面更强但 PPL 增大 26×；随机剪枝效果最低。

**⚠️ 局限性**

局限性：①不同模型对剪枝预算的耐受性差异大，需针对每个模型选择合适预算；②依赖实体标注与 NER，漏检实体会影响参考模型；③仅通过对人名泄露记录的攻击训练，剪枝后对其它 PII 类型（如地址、组织名）的保护有限；④剪枝后若模型继续细调或量化，已去除的泄露内容可能再次出现。

---

## 26. MaskMerge: Structure-aware Token Merging for Efficient Instance Segmentation

**arXiv ID:** 2610.22921 | [PDF](https://arxiv.org/pdf/2610.22921v1)

**作者:** Bavesh Balaji `[一作]` (University Of Waterloo), Sirisha Rambhatla `[通讯]` (University Of Waterloo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种实例感知的 Token 合并方法 MaskMerge，解决了相似度合并在实例分割中的跨实例融合问题。

**💡 创新点**

创新点在于使用模型中间阶段粗略的实例掩模来约束 Token 合并，仅在同一实例内部合并，从而在保持实例边界的同时实现更激进的压缩。

**🔧 技术方法**

结合 Vision Transformer、Patch 令牌化、局部与全局（基于掩模的）Token 合并模块、轻量级掩模预测头以及标准的 Transformer 编码器/解码器。

**📊 数据集**

在三大实例分割数据集上评测：医学影像核分割数据集 PanNuke、空中图像实例分割数据集 iSAID，以及城市场景数据集 Cityscapes。

**📈 对比分析**

与 ToMe、ALGM 等基于相似度的 Token 合并方法对比，MaskMerge 在 PanNuke、iSAID 上提升 mAP（分别 2.0 与 3.1 点）并降低 GFLOPs（约 1.5 倍和 1.08 倍），实现更高吞吐量；在 Cityscapes 上保持相同 mAP 但使用更少 GFLOPs。

**⚠️ 局限性**

局限性在于需要模型能够生成中间掩模（或额外的掩模预测头），在掩模质量不佳时可能影响合并效果；另外对极度稀疏或小实例的掩模精度仍有待提升。

---

## 27. MedSTP: Preserving Subspace and Transition Evidence for Medical VQA Token Pruning

**arXiv ID:** 2610.22751 | [PDF](https://arxiv.org/pdf/2610.22751v1)

**作者:** Mengchen Fan `[一作]` (University of Alabama at Birmingham), Baocheng Geng `[通讯]` (University of Alabama at Birmingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MedSTP，一种训练无关的视觉标记剪枝框架，用于在冻结的医学视觉语言模型（VLM）中进行固定预算的视觉标记选择；

**💡 创新点**

创新点在于将子空间稳定证据保持与局部边界/纹理过渡先验相结合，通过标记侧PCA和有界Sobel转换提升对医学图像细粒度证据的保留；

**🔧 技术方法**

采用标记侧主成分分析（PCA）构建视觉证据子空间，利用正交投影计算子空间贡献，加入Sobel滤波得到局部过渡分数，并通过有界乘子调节选择；

**📊 数据集**

在SLAKE、OmniMedVQA以及RadImageNet-VQA三大医学VQA基准上，使用Lingshu-7B和Qwen3-VL-8B两种VLM后端进行评估；

**📈 对比分析**

与FastV、SparseVLM、VisPruner、HRViT、BTP、PruneSID等方法比较，MedSTP在闭合端和开放端的平均准确率上分别提升约3.5%–5.2%，并实现了38%推理前置延迟缩减和49%显存节省；

**⚠️ 局限性**

局限性包括子空间逼近未能完全保证答案准确性，过渡先验仅作为修正因子无法补偿完全无贡献的局部特征，且实验仅限二维医学图像与两种VLM，未验证在体积图像或更广泛临床任务中的适用性。

---

## 28. When Approval Becomes Execution: Authorization-Grounded Workflow Attacks on Tool-Using Agents

**arXiv ID:** 2610.23077 | [PDF](https://arxiv.org/pdf/2610.23077v1)

**作者:** Chenggong Shi `[一作]` (Baidu), Hongmei Mou `[通讯]` (China Agricultural University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在 OpenClaw 竞赛环境中构造基于授权的工作流攻击，展示了模型代理在获取本地审批令牌后可以跳过完整授权检查，成功访问受保护文件。

**💡 创新点**

创新点在于揭示了“授权上下文缺口”——即仅凭令牌持有即可被服务视为充分授权，而未验证请求者、批准人、资源、操作、目的等完整上下文，提供了工作流层面的安全缺陷新视角。

**🔧 技术方法**

技术手段主要包括：使用基于工具的语言模型代理（LLM Agent）、OpenClaw 的本地服务调用接口、官方 Runner 跟踪收集与分析、以及对工作流步骤的系统化诊断。

**📊 数据集**

使用的数据集为 OpenClaw 公开的 Enterprise OA 文档目标，具体包含三条文档访问任务（导出、读取 CEO 文档、读取 HR 计划文档）。

**📈 对比分析**

评估方法基于官方 Runner 生成的执行轨迹，统计 9 次完整重跑中 8 次成功完成目标操作、9 次全部通过审批令牌转换；在 23 次本地验证跑中，原始配置 9/9 通过审批、8/9 完成目标；单因素剔除实验亦保持 3/3 成功，表明攻击模式可复现，性能指标即 100% 的目标成功率。

**⚠️ 局限性**

局限性主要包括：仅在三条文件访问任务上验证，缺乏跨领域泛化；未进行实际防御实验（如令牌上下文绑定、服务端重新校验）；实验环境为模拟竞赛环境，实际企业系统的安全性可能因实现细节不同而不同。

---

## 29. Aligning Information Theory and Formal Verification: A Lean Formalization of Rate-Distortion Theory

**arXiv ID:** 2610.22404 | [PDF](https://arxiv.org/pdf/2610.22404v1)

**作者:** Fubin Yan `[一作]` (Chinese University of Hong Kong, Shenzhen), Kenneth W. Shum `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**通讯引用:** 2652 | [OpenAlex ID](https://openalex.org/A5103017634)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在Lean证明助手中实现了离散源的率失真理论核心内容，给出了信息量、KL散度、互信息、失真函数等定义，并完成了直接定理和逆定理的完整机器可验证证明，构建了可复用的Lean库。

**💡 创新点**

首次在Lean中正式化率失真理论，解决了定义与教材对齐的问题，提供了完整的证明框架和可复用组件，展示了AI辅助生成与Lean验证相结合的工作流程。

**🔧 技术方法**

使用Lean、Mathlib、概率质量函数、有限总和、滤波器、连续性理论，以及随机编码、信息密度等信息理论技术；同时利用大型语言模型生成证明代码并通过Lean进行校验。

**📊 数据集**

论文为理论证明工作，不涉及具体实验数据集。

**📈 对比分析**

通过与传统手工证明对比，强调了可重用性、从零开始的省时优势，并说明Lean验证能保证证明的完整性；未给出数值性能指标。

**⚠️ 局限性**

目前仅适用于有限字母离散源，未覆盖连续或多终端情况；对齐检查仍需人工复核；未来需扩展到更复杂的主题（如信息几何、学习理论）并实现跨证明助手的复用。

---

## 30. Brain-Aligned Multimodal LLMs for Social and Emotional Understanding of Cartoon Movies

**arXiv ID:** 2610.22706 | [PDF](https://arxiv.org/pdf/2610.22706v1)

**作者:** Yiran Huang `[一作]` (New Jersey Institute of Technology), Mengjia Xu `[通讯]` (New Jersey Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估多模态大型语言模型（MLLMs）在动画电影中对社会认知（ToM）和情感（valence）持续评分的可行性，并比较其与人类评分及脑活动的对应关系。

**💡 创新点**

首次将开放权重的自回归与扩散型 MLLMs 用于帧级连续评分，探讨情境描述对模型与人类一致性的影响，并验证模型评分对 fMRI 信号的预测能力。

**🔧 技术方法**

使用 Qwen3‑VL、Gemma‑3（自回归）和 LLaDA‑V（扩散）三种模型生成 ToM 与 valence 评分，结合皮尔逊相关和多输出岭回归编码模型进行脑响应预测。

**📊 数据集**

采用三部动画电影（The Present、Despicable Me、Partly Cloudy）的 fMRI 数据与人类持续评分数据（ToM 与 valence）。

**📈 对比分析**

通过时序相关性（含 lag‑优化）和对 114 个 ROI 的预测 R² 进行比较，结果显示模型对 valence 的匹配优于 ToM，部分模型在加入 4–16 s 先前场景描述后一致性提升，TP 电影中模型评分在多 ROI 上的编码显著性最高。

**⚠️ 局限性**

局限包括评分分布与人类差异、缺少音频/对话信息、受试者年龄与评分者年龄不匹配、模型评分与真实心理状态可能仅捕捉可感知特征，且 lag‑优化结果仅为探索性，需在独立数据上验证。

---

## 31. Look Before You Act: Bridging Active Perception and Vision-Language-Action Models via Observation-Aligned Data Adaptation

**arXiv ID:** 2610.23378 | [PDF](https://arxiv.org/pdf/2610.23378v1)

**作者:** Jiawei Geng `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ActiveAlign 方法，利用主动摄像头预定位姿并将其终点与原始演示前缀匹配、闭环对齐和重播，构造对齐演示后对 VLA 模型进行 fine‑tune。

**💡 创新点**

创新点在于把主动视角选择与 VLA 训练紧密耦合：通过观察对齐的数据适配（prefix matching + 末端对齐 + 成功重播）使得 VLA 能在部署时接受与其观察一致的起始状态，保持 VLA 架构不变。

**🔧 技术方法**

技术包括视觉‑语言‑动作（VLA）模型（π0、π0.5、OpenVLA、GR00T N1）、基于 CLIP 的视觉+语言自回归摄像头轨迹生成、闭环操作空间对齐、成功重播过滤、离线数据构造与微调。

**📊 数据集**

使用 LIBERO（40 任务、2000 条演示）、LIBERO‑Occ（遮挡版本）、RLBench（仅用于 perception 转移）以及物理实验中的真实工作区演示。

**📈 对比分析**

与标准训练、Naive Active（仅在推理时使用主动摄像头）以及匹配状态/数据匹配对照进行对比；在受扰动起始条件下，ActiveAlign 的成功率从 30.7%/46.3% 提升至 70.7%/72.7%（分别高 40%/26%），在遮挡任务和物理实验中也大幅提升（如遮挡成功率 38.9%→92.2%）。

**⚠️ 局限性**

局限性：需要可重置的仿真或可重构物理工作区来完成对齐与重播；只做一次视角选择，无法实时响应移动遮挡；对齐后训练数据受限于成功重播的端点，难点端点可能被低估；不适用于非可重置环境或需要连续视角选择的任务。

---

## 32. Accelerated Dataflow: scaling inference in streaming data analytics with efficiency

**arXiv ID:** 2610.22731 | [PDF](https://arxiv.org/pdf/2610.22731v1)

**作者:** RuiLong Jiang `[一作]` (Google), Reza Rokni `[通讯]` (Google)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在Google Cloud Dataflow上实现了可自动扩缩的混合式流式机器学习管道，集成LLM推理并支持异构工作者池和前缀感知批处理

**💡 创新点**

提出了共设计的自适应扩缩算法、前缀感知批处理以及资源提示的异构工作者分配机制，显著提升推理吞吐量与资源利用率

**🔧 技术方法**

使用Apache Beam/Cloud Dataflow、vLLM/Triton推理引擎、AsyncDoFn、资源提示、异构工作者、前缀缓存、动态分片、嵌入式推理引擎

**📊 数据集**

使用OpenOrca、Dolly公开数据集进行LLM推理实验，TimesFM（时序预测）用于传统DNN实验

**📈 对比分析**

与原始Dataflow、Flink及其默认自适应调度器对比，实验显示吞吐量提升1.5-2倍、GPU小时减少至25%（最多51%），峰值尾部延迟降低84%~71%，且在峰值负载下资源分配更灵活

**⚠️ 局限性**

主要局限包括工作者启动慢导致的扩缩延迟、键分布不均导致的分片与缓存权衡、进程间通信产生的拷贝开销、对动态自分片与异常恢复的支持不完善

---

## 33. ECHO: Replacing the Compliance Proxy with Response Harmfulness in Optimization-Based Jailbreak Attack

**arXiv ID:** 2610.23030 | [PDF](https://arxiv.org/pdf/2610.23030v1)

**作者:** Yuanwei Wu `[一作]` (Nanjing University), Sheng Zhong `[通讯]` (Nanjing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的黑箱优化框架ECHO，用于针对大型语言模型的越狱攻击，替代了基于合规性的代理，使用响应级别的有害性评分来指导搜索。

**💡 创新点**

ECHO的创新点在于使用HarmSig作为快速的连续有害性信号，能够更准确地评估模型的输出是否真正有害，而不是仅仅依赖于表面的合规性指标。

**🔧 技术方法**

使用了HarmSig（一个1.7B的回归模型），结合五种词级扰动操作符和关键字锚定的束搜索，构建了一个严格的黑箱威胁模型。

**📊 数据集**

在AdvBench数据集上进行了评估，该数据集包含520种有害行为，涵盖了非法活动、暴力、欺诈和隐私侵犯等多个领域。

**📈 对比分析**

与现有的优化方法（如GCG、AutoDAN、DSN等）进行比较，ECHO在StrongREJECT和HarmBench分类器上均表现优异，StrongREJECT得分最高可达0.898，HarmBench成功率达到89.23%。

**⚠️ 局限性**

ECHO的局限性在于商业系统可能结合模型对齐与独立的输入/输出安全分类器，从而降低ECHO的攻击成功率。此外，ECHO的查询预算较大，直接针对这些API的优化成本较高。

---

## 34. Audio System One Model: Typed Probabilistic Perception of Emotion in Speech

**arXiv ID:** 2610.22401 | [PDF](https://arxiv.org/pdf/2610.22401v1)

**作者:** Jaime Bellver-Soler `[一作]`, Luis Fernando D'Haro `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种系统一（System One）语音模型，能够在一次前向传播中根据文本提问生成校准后的概率分布，既不需要固定标签集，也不采用生成式语言模型。

**💡 创新点**

创新点在于将标签描述为文本，用冻结的音频和文本编码器加一个小型读者（reader）实现对任意文字问题的快速、校准概率预测，同时利用听众投票分布进行训练，显著提升对情感识别及其他属性的适应性。

**🔧 技术方法**

技术包括冻结的 Qwen2.5-1.5B 文本编码器、Whisper-large-v3 语音编码器、RoBERTa-large 句子编码器、一个 6.2M 参数的读者网络，以及跨折叠蒸馏、目标平滑和跨句式训练等方法。

**📊 数据集**

使用了公开情感语料：MSP‑Podcast 2.0、CREMA‑D、Emozionalmente，分别提供每位听众的投票信息；同时在未见语料（zs.corpora、EmoNet‑Voice、EARS 等）上进行跨域评测。

**📈 对比分析**

与传统固定标签分类器、零样本音频 LLM 以及 CLAP 模型相比，本文模型在 MSP‑Podcast 上与最佳分类器相当、在情感识别和性别判断上明显优于零样本 LLM，且一次性答复 20 题的延迟仅略高于 Whisper 编码，远快于每题单独前向的 LLM；在未见数据集上也保持了更好的宏 F1 与 AUC。

**⚠️ 局限性**

局限性包括：仍依赖情感投票数据作为训练信号，数据量有限；对非情感属性（如说话者身份、情境）识别效果不佳；在词汇携带情感的任务中，文本转录信息仍对结果有较大影响。

---

## 35. A Class of Rényi-based Concentration Metrics for Kolmogorov-Arnold Geometry in Multilayer Perceptrons

**arXiv ID:** 2610.23267 | [PDF](https://arxiv.org/pdf/2610.23267v1)

**作者:** Juan Carlos Fernández `[一作]` (University of Málaga), José Francisco Paris `[通讯]` (University of Málaga)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一类基于Renyi熵的α-RC指标，用来量化神经网络内部映射雅可比矩阵的子式（minors）的集中程度，取代传统的参与比率（PR）指标。

**💡 创新点**

创新点在于：①α-RC指标在所有网络架构下均归一化到[0,1]区间，极值含义一致；②指标随α单调递增；③当α=2时可直接通过PR转换得到，保持了PR的点值信息；④提供了对不同α权重的解释。

**🔧 技术方法**

技术方法包括：Kolmogorov–Arnold几何框架、雅可比矩阵及其外积的子式计算、Renyi熵与Renyi散度的定义与性质、单隐藏层多层感知机（MLP）的训练与梯度优化。

**📊 数据集**

实验使用合成的连续XOR问题数据集（输入维度3，目标为三变量正弦乘积）。

**📈 对比分析**

通过在不同宽度（m=8、32）和子式阶数（k=1,2,3）下计算α-RC与PR，发现α-RC在不同网络间可直接比较；α=2与PR呈对数映射；训练过程中α-RC随epoch上升，峰值出现于中后期，而R²提升持续至终点。

**⚠️ 局限性**

局限性在于：子式数量随网络宽度、输入维度和阶数呈组合爆炸式增长，导致精确计算在大规模网络上不可行；需要开发采样或近似估计方法；不同α对训练指导和解释的效果尚未充分探究。

---

## 36. Process Reward Modeling with Reliability- and Conservatism-Aware Bayesian Belief

**arXiv ID:** 2610.23223 | [PDF](https://arxiv.org/pdf/2610.23223v1)

**作者:** Zhongjian Qiao `[一作]` (City University of Hong Kong), Shuang Qiu `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种 BayesianPRM 方法，利用贝叶斯学习框架在推理时刻动态估计过程奖励，以提升大语言模型的推理性能。

**💡 创新点**

创新点在于将过程奖励的不确定性分为 PRM 不确定性与策略不确定性，并通过两阶段贝叶斯信念更新——可靠性感知更新与保守性校准——同时消除两种不确定性。

**🔧 技术方法**

技术上采用奖励集成、先验网络、可消融的可靠性信念网络、ELBO 训练以及基于熵调节的保守性指数重加权，实现了可扩展的贝叶斯推理。

**📊 数据集**

训练使用 VisualPRM400K-v1.1 数据集，验证与测试分别在 MathVision、MathVerse-VO、MathVista、MMStar 等多模态推理基准上进行。

**📈 对比分析**

与标准 PRM、BetaPRM、DreamPRM、Self‑Consistency 等基线相比，BayesianPRM 在 Brier Score、ECE 等校准指标上显著下降，并在 Best‑of‑N 与 IAS 推理加速方案中平均提升约 1–2 分准确率。

**⚠️ 局限性**

局限性包括：仅在多模态推理任务上验证，未评估极端 OOD 情况；奖励集成离散化可能导致信息损失；需要额外验证数据来训练信念网络，若无充分校准集效果可能受限。

---

## 37. Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning

**arXiv ID:** 2610.23181 | [PDF](https://arxiv.org/pdf/2610.23181v1)

**作者:** Heming Zou `[一作]` (Tsinghua University), Xiangyang Ji `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了TreeOPSD，一种用于长时限代理式强化学习的无价值网络后验自蒸馏框架，能够仅利用终端可验证奖励来为中间动作分配密集信用并在连续小批更新中消除策略漂移。

**💡 创新点**

核心创新在于将树形回溯与后验分布映射结合，先通过树备份将稀疏终端奖励递归为每个回合的隐式信用，再将该信用映射为一个与行为策略相对的有限Bernoulli教师分布，并用前向KL对齐学生策略，从而在不引入价值网络或外部教师的情况下实现可逆且有限的更新。

**🔧 技术方法**

技术实现包括：初始化-扩展树采样、对终端奖励进行标准化与树备份、使用对数比率平均生成的后验分布、构造具有温度τ的线性教师概率、以及在多批次训练中利用前向KL损失进行分布匹配。

**📊 数据集**

实验数据集涵盖多跳问答（HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle）与深度研究（OpenResearcher的BrowseComp-Plus、BrowseComp、GAIA、xbench-DeepSearch），并在Qwen3-4B与Qwen3-30B两种规模模型上进行训练。

**📈 对比分析**

与ReAct、GRPO、GSPO以及TreePO等基线相比，TreeOPSD在所有评测集合上均实现了最高平均准确率，尤其在长时限的Deep Research任务中提升约4.9个百分点（30B模型），显示出显著的性能优势。

**⚠️ 局限性**

局限性包括对树构造与目标半径ρ的敏感性、需要在每个回合上额外计算后验分布的计算开销，以及在更大规模或更高时限任务中的可扩展性与通用性尚未得到充分验证。

---

## 38. A Label Noise Robust Algorithm for Graph Data using Neighborhood Label Distribution

**arXiv ID:** 2610.22998 | [PDF](https://arxiv.org/pdf/2610.22998v1)

**作者:** Pintu Kumar `[一作]` (Indian Institute of Technology Bombay), Nandyala Hemachandra `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究标签噪声对图节点邻居标签分布的影响，并提出一种基于留一去噪熵的节点筛选与重标记方法 LENS，用于检测并纠正图数据中的噪声标签。

**💡 创新点**

创新点：①理论证明噪声使邻居标签分布熵升高、类间可区分性下降；②基于留一去噪熵的轻量化节点筛选框架 LENS；③在去噪后引入重标记机制，既保留有价值特征，又提升鲁棒性。

**🔧 技术方法**

技术手段：熵度量、留一去噪、伪标签生成、GCN 训练、对称标签噪声/对等噪声建模以及对噪声影响的理论分析。

**📊 数据集**

实验数据集：Cora、Citeseer、Amazon Photo、PubMed、DBLP、Amazon Ratings、Roman Empire 等常用图分类基准。

**📈 对比分析**

对比方法：与 NRGNN、DeGLIF、TSS、CP、RNCGLN 等多种现有噪声鲁棒 GNN 进行比较。LENS 在节点分类准确率上始终位居前列，同时在 Cora 数据集上仅 0.25 秒的总运行时间大幅优于其它方法（最慢的 DeGLIF 需要 68 秒）。

**⚠️ 局限性**

局限性：对异构或无明显类别邻域（如“Bad Heterophily”图）中的噪声节点识别效果不佳；在类边界处仍可能产生误报。

---

## 39. Knowledge-Guided Distributional Prompt Learning with Visual Hierarchy for Zero-Shot Anomaly Detection

**arXiv ID:** 2610.23172 | [PDF](https://arxiv.org/pdf/2610.23172v1)

**作者:** Jiyul Ham `[一作]` (Korea University), Jun-Geol Baek `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于视觉层次结构的零样本异常检测框架 HiDist，通过对提示词进行分布式建模和正常/异常的相对偏差来实现图像级检测和像素级定位。

**💡 创新点**

创新点包括：①将每个提示词建模为可学习的高斯分布，显式捕捉各词位置的语义中心与方差；②采用正常提示的分布作为参照，对异常提示进行参数化偏移，保持正常-异常的统计依赖；③在全局与局部两个层次分别学习分布式提示，并通过知识引导的对比学习对齐视觉与文本层次，形成层级一致性；④引入多项辅助损失（分布距离、正交性、KL 正则等）以提升分布质量与多样性。

**🔧 技术方法**

技术主要包括：视觉语言模型 SigLIP 2；蒙特卡洛采样生成多样提示；高斯分布参数化与流式概率化；知识引导对比学习；多尺度视觉适配器；分布距离、正交约束、KL 正则等正则化；文本编码器的层归一化+头部微调。

**📊 数据集**

使用工业领域的 MVTec-AD、VisA、DTD-Synthetic 三个数据集；医疗领域的 CVC-ColonDB、CVC-ClinicDB、Kvasir、Endo 四个数据集，覆盖从纹理缺陷到内镜病灶的多样异常。

**📈 对比分析**

与 OpenCLIP、SigLIP 2 以及现有 ZSAD 方法 WinCLIP、APRIL-GAN、AnomalyCLIP、AdaCLIP、Bayes-PFL 进行比较。HiDist 在大多数指标（AUROC、AP、F1-max、PRO）上均优于或相当于竞争方法，尤其在工业数据集的图像级检测上提升约 3–4% AUROC，像素级定位在多数指标上表现最优，显示了强大的跨数据集泛化能力。

**⚠️ 局限性**

局限性包括：①采样得到的多样提示仅在推理时聚合成单一原型，忽略了样本级多样性可能带来的额外判别信息；②使用对角高斯分布对提示词建模，限制了多模态、跨维相关性的表达；③对分布参数的正则化依赖标准正态先验，可能在某些复杂语义下不足。

---

## 40. Aspire: Practical Batch Queries on Dynamic Encrypted Database with Full Leakage Suppression

**arXiv ID:** 2610.23057 | [PDF](https://arxiv.org/pdf/2610.23057v1)

**作者:** Hao Nie `[一作]` (Huazhong University of Science and Technology), Kaitai Liang `[通讯]` (University of Turku)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一套名为 Aspire 的加密数据库系统，支持批量关键词查询和动态更新，并实现了受限零泄露安全模型。

**💡 创新点**

创新点在于提出受限零泄露（restricted zero‑leakage）安全定义，结合概率批量编码（PBC）和最小完美哈希（MPH）实现高效的批量查询，同时完全消除可被泄漏滥用攻击利用的语义泄露。

**🔧 技术方法**

采用了私有信息检索（PIR）或 ORAM、概率批量编码、最小完美哈希以及本地更新缓冲等技术，实现了对查询和更新的零泄露保证。

**📊 数据集**

使用合成数据集：2^22 条键值对，32 位键、256 字节值，批量查询 5461 条关键词，作为实验基准。

**📈 对比分析**

通过与 PathORAM、OMAP24 以及不同 PIR/ORAM 实例对比，Aspire 在批量查询场景下平均延迟仅 1.38 ms（5461 查询总耗时 7.53 秒），每次查询通信量约 2.95 KB，更新操作轻量且可异步执行。

**⚠️ 局限性**

局限性：受限零泄露只隐藏语义泄露，仍泄露操作计数、时间等元数据；更新需要周期性同步，PIR 方案扫描线性导致查询成本随数据库规模增长；ORAM 版本则需较大客户端存储，影响部署灵活性。

---

## 41. How Murky Before a VLM Knows It Cannot See? Accuracy, Confidence and Abstention on a Physical Visibility Axis

**arXiv ID:** 2610.22981 | [PDF](https://arxiv.org/pdf/2610.22981v1)

**作者:** Mochamad Rizal Fauzan `[一作]` (National Taipei University of Technology), Irgi Surya `[通讯]` (National Taipei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在可见度（以米为单位）这一物理轴上，评估了五个小型开放式视觉语言模型（VLM）在雾和水下环境下的识别任务，记录了准确率、内部置信度、口头置信度和显式拒绝选项的表现。

**💡 创新点**

创新点在于：①将传统的离散严重度等级替换为连续的物理可见度轴；②引入“盲点宽度”W指标量化准确率与置信度/拒绝的时间差；③比较内部置信度、口头置信度与显式拒绝的可靠性，并分析拒绝选项的措辞敏感性。

**🔧 技术方法**

技术上采用了Koschmieder雾模型和纯水吸收谱的物理降解方法，并利用单目深度估计（Depth Anything V2）生成距离图；通过在COCO val2017 1000张图片上蒙皮并插入红框，向模型发问，并用greedy解码提取四个读数。

**📊 数据集**

使用的数据集包括COCO val2017（含物体框与深度估计）、Depth Anything V2、以及五个2–5B参数的开放式VLM（Qwen3‑VL‑2B、Qwen3‑VL‑4B、InternVL3.5‑2B、SmolVLM2‑2.2B、Gemma‑4‑E2B）。

**📈 对比分析**

对比方法是通过可见度对准点（V_acc、V_int、V_verb、V_abst）计算盲点宽度W，并绘制置信度与准确率随可见度变化的曲线；结果显示准确率在约10倍可见度范围内跌至随机水平，而内部置信度和口头置信度大幅滞后，盲点宽度至少为1个十倍；显式拒绝在使用“我无法判断”措辞时显著弱化；ImageNet‑C 雾级别在物理可见度上对应约200–300 m，远高于失效带。

**⚠️ 局限性**

主要局限包括：仅使用1000张图片和单一深度估计（尺度不确定）；物理降解模型过于理想化（无散射、湍流）；任务设计与红框定位、先验匹配、回答偏差等可能影响结果；拒绝与口头置信度的措辞敏感性未全面覆盖；未加入人类基准；实验仅针对小型VLM，未验证更大模型或不同族群的泛化能力。

---

## 42. OrderFlowDiT: Transport-Regularized Diffusion Transformer for Probabilistic Limit-Order Supply Forecasting

**arXiv ID:** 2610.22727 | [PDF](https://arxiv.org/pdf/2610.22727v1)

**作者:** Jung Hun Phee `[一作]` (Georgia Institute of Technology), Tucker Balch `[通讯]` (Emory University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种条件扩散变压器 OrderFlowDiT，用来生成在未来五分钟内提交的未归一化买卖方限价订单量场。

**💡 创新点**

创新点在于将无偏差的非平衡最优传输（UOT）损失作为辅助目标，显式考虑价格位移与总量差异，并设计了完整字段评估框架，包括运输能量得分、联合字段能量得分和对数体积 CRPS。

**🔧 技术方法**

主要技术包括条件扩散模型、Transformer 架构、无偏差的非平衡 Sinkhorn 散度、DDIM 采样以及 GRPO（基于组相对奖励的微调）来提升分布多样性。

**📊 数据集**

使用了 LOBSTER 全深度订单信息，涵盖 30 只美国股票（如 AAPL、AMZN、MSFT 等），每只股票在训练期 29 天、验证期 1 天内采集 1,950 条预测条件。

**📈 对比分析**

在 1,950 条验证条件上与统计基线（Skew-normal、混合 Skew-normal、经验样本）对比，Base-UOT 在全场成本下降 45.8%、运输能量得分下降 26.1%，且所有股票均获改进；GRPO 在运输能量得分和体积 CRPS 上进一步提升，但样本多样性和联合字段得分下降。

**⚠️ 局限性**

局限性在于模型仅预测订单提交量场，未考虑订单取消、成交和事件顺序，因而无法完整模拟订单簿演化及其对价格的影响，需进一步构建完整的金融世界模型。

---

## 43. GUMPIM: Unitary and Malleable Memory for Processing-in-Memory with Guaranteed PIM Pages

**arXiv ID:** 2610.23406 | [PDF](https://arxiv.org/pdf/2610.23406v1)

**作者:** Yilong Zhao `[一作]` (Shanghai Jiao Tong University), Li Jiang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种兼容现有CPU体系的处理器内存（PIM）系统，采用统一可变内存空间，支持CPU页面的内存交错与PIM页面的非交错共存，实现零拷贝、低迁移开销并保留CPU内存带宽。

**💡 创新点**

创新点包括：①双轨内存管理（CPU与PIM页面独立管理）；②块级（chunk）内存分配机制，保证PIM页面连续；③DRAM侧硬件实现的PIM辅助页面迁移和行迁移表（Row Migration Table），消除CPU端的数据拷贝；④Re‑Layout Cache实现PIM级别的数据重排；⑤针对多种DRAM（DIMM、LPDDRx、HBMx）可直接部署的硬件方案。

**🔧 技术方法**

采用DRAM侧硬件模块（Rank Chunk List、PIM Chunk List、Address Translation Module、Command Generator、Re‑Layout Cache、Row Migration Table）以及Linux THP、系统调用和自定义API进行内存分配与访问；使用GEM5+Ramulator2仿真、真实UPMEM DPU执行PIM任务。

**📊 数据集**

使用的基准数据集包括：CPU工作负载—SPEC CPU2006；PIM工作负载—PRIM、GAP、AiM、PIM‑ML、BFS、PageRank、NW、UNI、SCAN‑RSS/SSA、SEL、HST‑S等；在不同DRAM架构（DIMM、HBM3、LPDDR5）上进行评测。

**📈 对比分析**

与CPU‑only、PIM‑Ioff（隔离内存、关闭交错）和PIM‑Ion（隔离内存、软件地址映射）等基线比较：CPU负载仅损失<0.1%带宽；PIM任务总时间比PIM‑Ioff提升约1.96×，比基线提升约2.3×；块分配时间下降2.7×，CPU部分时间下降4.93×。

**⚠️ 局限性**

局限性：依赖DRAM侧硬件改造，需要在CPU侧提供API支持；块级连续分配仍需一定内存可用性；行级迁移表实现复杂且面积开销约0.13% DRAM die；对极细粒度、随机访问密集的工作负载迁移开销仍有提升空间。

---

## 44. Trajectory-Guided Steering: Mitigating Spurious Feature Bias in Clinical Classification

**arXiv ID:** 2610.22618 | [PDF](https://arxiv.org/pdf/2610.22618v1)

**作者:** Fardin Ahsan Sakib `[一作]` (George Mason University), Ozlem Uzuner `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了Trajectory‑Guided Steering (TGS)，一种利用LLM隐藏层预测轨迹在推理时动态纠正错误的方法，能够消除对无关特征的依赖。

**💡 创新点**

创新点在于只通过模型自身的logit轨迹形状决定何时及以何种方向纠正预测，无需对抗样本、事先知道偏置特征或重新训练模型。

**🔧 技术方法**

核心技术包括logit lens、对两类答案的margin/conviction计算、轨迹形状分类（增长、静态、内部冲突）、自适应层选择和基于轨迹的logit偏置调整。

**📊 数据集**

使用的评估数据集包括i2b2 2008肥胖挑战（16病种）、n2c2 2018队列选择、SHAC药物使用以及PubMedQA，覆盖临床文本分类与医学问答。

**📈 对比分析**

与prompt改写、Chain‑of‑Thought、CAA、FairSteer等基线相比，TGS在i2b2上净增+201、修正率17.7%、破坏率1.3%，在不同模型和数据集上也保持净正，显著优于所有基线。

**⚠️ 局限性**

局限性：仅适用于固定答案集的分类/QA任务；需为每个模型‑数据集手工调参α、τ；仅在7‑8B英语LLM、临床/生物医学领域验证，未测试更大模型、非英语或非医学领域；假设logit轨迹可区分性，对其他架构和更广泛提示变化的鲁棒性未知。

---

## 45. Embodiment-Classifier Guidance: Transferring Frozen VLAs Across Robot Dynamics without Demonstrations

**arXiv ID:** 2610.22576 | [PDF](https://arxiv.org/pdf/2610.22576v1)

**作者:** Devin Han `[一作]` (University of California Berkeley), Hershraj Niranjani `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在仿真环境中，针对冻结的流式 VLA 机器人策略，构建了一个轻量化体态分类器，用于识别目标身体上的成功动作块，并在每一步去噪采样时加入该分类器梯度，实现在不更新策略参数或额外演示的情况下迁移到不同机器人身体上；

**💡 创新点**

创新点在于将基于成功示例的分类器梯度引导（classifier guidance）直接应用于流式动作生成模型，实现对机器人身体差异的自适应迁移，且仅需极少的目标身体校准数据；

**🔧 技术方法**

核心技术包括流式匹配模型（flow-matching）进行动作块采样、噪声条件 Transformer 作为分类器、梯度加权引导（guidance）以及轻量级的自监督校准；

**📊 数据集**

使用 LIBERO 机器人桌面操作基准（四套任务：Object、Spatial、Goal、LIBERO-10），并在多种目标身体（控制器硬度变化、腕力弱化、延迟、锁定腕等）上进行评估；

**📈 对比分析**

通过对 12 组 suite–body 组合进行对比，指导方法平均提升成功率从 72% 上升到 80%，失败率降低 31%，在 11/12 组中获得显著提升（5 组在 Holm 校正后仍显著），但在弱腕或锁定腕等极端身体配置时性能可能下降；

**⚠️ 局限性**

局限性包括：仅在仿真中验证；需要目标身体成功示例并进行多次收集；分类器输入隐含末端执行器姿态，难以区分身体位置与运动方式；对极端物理限制（如锁定腕）效果不佳；方法对不同目标身体的适用性尚未统一预测；整体评估仅基于单一策略和基准；

---

## 46. Bayesian Learning of Parametric Mean Field Games with Common Noise

**arXiv ID:** 2610.23387 | [PDF](https://arxiv.org/pdf/2610.23387v1)

**作者:** Grégoire Lambrecht `[一作]` (New York University), Mathieu Laurière `[通讯]` (NYU Shanghai)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

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

## 47. NoveltyScope: A Multi-Agent Research Assistant for Evidence-Grounded Novelty Assessment of Scientific Papers

**arXiv ID:** 2610.22703 | [PDF](https://arxiv.org/pdf/2610.22703v1)

**作者:** Yueran Tian `[一作]` (Newcastle University), Huizhi Liang `[通讯]` (Newcastle University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 NoveltyScope 原型，能够检索同一任务及贡献层面的相关工作，生成结构化报告并给出 1~5 级新颖度评分。

**💡 创新点**

创新点在于将检索、技术对比与四角色（起诉人、辩护人、评判者、批评者）协同评分相结合，形成基于证据的新颖度评估流程。

**🔧 技术方法**

采用 BM25 与 SentenceTransformer 的混合检索（使用 Reciprocal Rank Fusion）来获取候选论文；使用 Gemini‑2.5‑flash 或 Qwen3‑Next‑80B‑A3B‑Instruct 等 LLM 进行任务/贡献抽取、对比与评分。

**📊 数据集**

主要使用 RINoBench（50 条平衡样本）做评估，并在本地 1,167 条记录的闭源语料库中检索；示例报告以 HiCLIP 论文为案例。

**📈 对比分析**

与单 LLM 评分相比，四角色多代理评分在 MAE 上分别降低了 25.0%（Gemini）和 31.0%（Qwen3），最优 MAE 为 1.16，显著优于向量相似度基线 1.68。

**⚠️ 局限性**

局限性包括检索覆盖受限于本地语料库、每篇论文仅抽取两条贡献、未做时间过滤、对比与报告的质量评估缺乏独立测评、以及样本量小、评估设置不易复现。

---

## 48. Where Latents Belong and How They Flow: Joint Identification of Structural Roles

**arXiv ID:** 2610.23074 | [PDF](https://arxiv.org/pdf/2610.23074v1)

**作者:** Xintong Li `[一作]` (Carnegie Mellon University), Yujia Zheng `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种理论与方法，用于在分组时序观测中联合识别潜在因子的结构角色（群组表达、过去影响、未来影响）并实现对应的潜在坐标对齐，随后用 RoleTrace 框架估计这些角色。

**💡 创新点**

核心创新在于：① 证明在满足结构与变异条件下，群组表达、传入影响和传出影响可在同一时间点上通过单一置换共同可识别；② 在此理论指导下设计了利用解码器雅可比与转移核交叉 Hessian 的三种角色估计，并通过共同稀疏化的坐标变换实现角色的一致对齐；③ 将理论应用到可训练的编码-解码网络，形成完整的可解释潜在表示学习方法。

**🔧 技术方法**

主要技术包括：基于可微编码器‑解码器的潜在表示学习；对转移密度的对数式第二阶导数（交叉 Hessian）来捕捉时序依赖；根均方得分对角色支持的经验估计；基于 Jacobi 旋转的正交坐标变换实现多角色共同稀疏化；以及实验中的 F1、Joint Accuracy、Hamming Accuracy 等评估指标。

**📊 数据集**

实验数据：① 16 个随机生成的非线性时序机制（N=6, K=3, T=5）用于结构恢复评估；② 4 个受控机制（N=4）用于对齐与灵敏度分析；③ 118 组 AMI 会议数据（K=4，时序窗口15秒）用于下游下一发言人预测的实证检验。

**📈 对比分析**

与 ThoughtComm、InterLatent、Temporal‑Only、No‑Structural、No‑Refit 等基线进行对比；在合成数据上 RoleTrace 的 Joint Accuracy 达到 0.854，显著高于其他方法；在 AMI 会议下游任务中，基于 RoleTrace 的 ℂ 选择在较小 latent 预算下取得最高或相近的准确率（最高 51.48%），验证其在信息保持方面的优势。

**⚠️ 局限性**

限制：① 对于真实数据仅评估了会议语音场景，缺乏更广泛的应用验证；② 理论与算法依赖可微性、正则化条件与足够变异的假设，易受这些假设破坏导致识别失效；③ 对于高维、长时序或强噪声场景的可扩展性与鲁棒性尚未充分探究；④ 仅考虑单时间点的联合对齐，未深入研究跨时间维度的持续一致性。

---

## 49. Framing Spatial Hearing within the Free Energy Principle: A Hierarchical Model of Perceptual Learning

**arXiv ID:** 2610.23196 | [PDF](https://arxiv.org/pdf/2610.23196v1)

**作者:** Roberto Barumerli `[一作]` (Imperial College London), Lorenzo Picinali `[通讯]` (Imperial College London)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究提出了将自由能原理（FEP）应用于空间听觉的框架，并通过一个层次化生成模型描述听者在多次试验中根据视觉反馈更新声源方向的信念。

**💡 创新点**

创新点在于将FEP用于动态听觉定位任务，将学习过程拆解为侧向精度、极角精度和半场歧义解消三部分，并揭示了它们在不同时间尺度上的进化规律。

**🔧 技术方法**

使用的技术包括变分自由能最小化、自然梯度更新、von Mises 分布与两组件混合模型以及 Laplace 近似，以实现对信念参数的在线修正。

**📊 数据集**

实验数据来自5名参与者的定位任务，每次试验记录声源方向、指向响应以及随后的视觉反馈。

**📈 对比分析**

通过最大似然估计获得学习率后，模型在群体平均轨迹上实现了0.75–0.83的R²拟合，显示出较好的预测性能，但未与其他基准模型直接对比。

**⚠️ 局限性**

主要限制包括样本量小、仅包含感知-反馈闭环而缺少主动推理（如头部运动策略），以及未考虑先验信息对参数不确定性的影响。

---

## 50. CGDT-CESN: Correlation-Guided and Dynamics-Trained Clustered Echo State Network for Multivariate Time-Series Prediction

**arXiv ID:** 2610.22430 | [PDF](https://arxiv.org/pdf/2610.22430v1)

**作者:** Hariharan Srinivasan `[一作]` (Indian Institute Of Technology Madras), Sayan Gupta `[通讯]` (Indian Institute Of Technology Madras)

**通讯引用:** 1662 | [OpenAlex ID](https://openalex.org/A5070732290)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Correlation‑Guided and Dynamics‑Trained Clustered Echo State Network（CGDT‑CESN）用于多变量时间序列预测

**💡 创新点**

创新点在于：① 通过 Spearman 相关性数据驱动构建跨簇连接概率；② 对跨簇权重采用坐标下降训练，保持 ESN 的线性输出结构；③ 将跨簇训练与簇特化读取层结合，提升多变量耦合建模

**🔧 技术方法**

使用了 Echo State Network、聚类（Clustered ESN）、Spearman 排列相关、坐标下降（Coordinate‑Descent）和岭回归等技术

**📊 数据集**

实验数据集包括：Twitter 信息级联（4维特征：size、depth、breadth、virality）和太阳风（density、speed、temperature）

**📈 对比分析**

通过 NRMSE 评价，CGDT‑CESN 在不同训练样本比例、预测时隙和两数据集上均优于传统 ESN 与仅采用相关引导的 CG‑CESN；同时保持了可接受的训练时间

**⚠️ 局限性**

局限性：需预先进行相关性分析与差分处理；跨簇权重训练仍受限于坐标下降的局部收敛；对不同类型数据的相关性度量选择需针对具体问题调整

---

## 51. Unified Geometric--Kinematic Laws of Mobility in 3D Cellular Networks

**arXiv ID:** 2610.23075 | [PDF](https://arxiv.org/pdf/2610.23075v1)

**作者:** Tian Shi `[一作]` (Sun Yat-sen University), Minghua Xia `[通讯]` (Sun Yat-sen University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出统一的几何–运动学框架，推导3D蜂窝网络中无人机切换率与剩余驻留时间的解析/近似公式。

**💡 创新点**

创新点在于：① 手动率可用 H = (S_V/2)·v̅ 的形式给出，只依赖服务边界的特定表面积 S_V 与平均速度 v̅；② 引入有效遍历近似计算平均剩余驻留时间，结合前向边界距离分布；③ 将上述理论应用于 Poisson–Voronoi（单BS）和 Poisson–Delaunay（四BS CoMP）网络，并通过手动率驱动的有效频谱效率评估 BS 密度与架构的最优选择。

**🔧 技术方法**

使用随机几何、平滑转弯移动模型、线性接触分布、Crofton 公式、复合过程与 renewal‑reward 理论、Monte Carlo 仿真等技术。

**📊 数据集**

实验数据全部基于仿真生成的 Poisson 点过程、随机路径等，未使用外部真实数据集。

**📈 对比分析**

通过 Monte Carlo 与理论公式对比验证手动率与驻留时间，结果显示理论与仿真吻合；利用手动率与覆盖概率得到的有效频谱效率，比较单 BS 与 CoMP 架构，在不同速度、干扰、路径衰落下给出最优 BS 密度与架构的决策准则。

**⚠️ 局限性**

限制：仅考虑平面无方向/无高度依赖的服务几何；忽略碰撞规避、方向天线、异构/有向部署以及动态路径规划；驻留时间近似假设前向距离分布仅由特定表面积决定，可能不适用于高度可变或极端航迹情况。

---

## 52. LongForge: Measuring and Improving Long-Context Instruction Data via Evidence-Assertion Reasoning Graphs

**arXiv ID:** 2610.23167 | [PDF](https://arxiv.org/pdf/2610.23167v1)

**作者:** Ziyang Chen `[一作]` (Tencent), Songlin Hu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于证据–主张推理图的可追溯质量测量与生成框架LongForge，用于构造高质量的长文本指令调优数据；

**💡 创新点**

创新点在于将质量转化为可追溯的四维度（覆盖度、深度、正确性、完整性），通过分析器检测并指导生成器重写错误，实现了可控的数据分布；

**🔧 技术方法**

主要技术包括：证据–主张推理图构建、四维度分析、基于模型的自动化分析器与生成器、阈值门控与多轮重答、以及通过度量分布进行训练样本混合；

**📊 数据集**

利用了七个主流长文本指令数据集（LongAlpaca、LongAlign、LongReward、LongMIT、LongMagpie、SEALONG、LongFaith）的文档作为来源，生成了新的QA对；

**📈 对比分析**

与上述七个基线数据集在六大长文本基准（LongBench、LongBench v2、LooGLE、MRCR、HELMET、RULER）上的比较显示，LongForge混合方案在Llama3.1-8B与Qwen3.5-4B两种骨干上分别平均提升约2.95与3.13分，常在所有子任务中超越最强基线；

**⚠️ 局限性**

主要局限在于分析器对每个文档段落逐句检索，调用量随文档长度线性增长，导致成本高昂；此外仅探讨了三种混合分布，缺乏系统的分布优化与更广泛的实验验证。

---

## 53. GR-OPD: Group-Relative Sequence Credit for On-Policy Distillation

**arXiv ID:** 2610.23042 | [PDF](https://arxiv.org/pdf/2610.23042v1)

**作者:** Yushu Wang `[一作]` (Shanghai Jiao Tong University), Hao He `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为Group-Relative On-Policy Distillation（GR-OPD）的序列级信用估计方法，用于改进传统的基于token的OPD；

**💡 创新点**

创新点在于将教师反馈的计算与信用分配解耦，通过聚合token级反馈为整个响应生成序列级信用，并引入指数折扣和同一提示组的校准来控制教师压力，从而实现更稳定、有效的学习；

**🔧 技术方法**

核心技术包括逆KL目标的理论分析、序列信用广播、指数折扣奖励、同一提示组均值校准以及PPO式的裁剪更新；

**📊 数据集**

实验数据集为DAPO-Math提示集，评估指标为六个数学推理基准（AIME 2024/25、AMC 2023、MATH-500、Minerva Math、OlympiadBench），共计1042道题；

**📈 对比分析**

与传统采样token OPD相比，GR-OPD在所有五个教师-学生设置上均取得宏平均Avg@K显著提升（如从23.3%提升至27.8%），并在思考模式不匹配的情形下避免训练崩溃，保持了更长的响应长度和更高的验证奖励；

**⚠️ 局限性**

局限性包括仅在数学推理和Qwen3模型上验证，无法证明其在更广泛领域和模型族的通用性；此外，单一序列级信用广播未细化token内部信用分配，未来需结合token级信用进一步提升效果。

---

## 54. Distributed Performance Boosting through Neural Control for Plug-and-Play Networks

**arXiv ID:** 2610.23292 | [PDF](https://arxiv.org/pdf/2610.23292v1)

**作者:** Danilo Saccani `[一作]` (Ecole polytechnique federale de Lausanne), Giancarlo Ferrari-Trecate `[通讯]` (Ecole polytechnique federale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种在预稳健可插拔网络中使用分布式神经网络提升性能的控制器，确保闭环稳定并在子系统加入或移除后自动重证；

**💡 创新点**

提出两种自由参数化的分布式IMC控制器，能够保证给定网络ℓ₂增益，并实现局部可插拔重证；实现了无需投影的梯度训练和邻接通信实现的分布式学习；

**🔧 技术方法**

使用内部模型控制(IMC)、自由参数化、Recurrent Equilibrium Networks、Magnitude And Direction(MAD)策略、稳态增益小增益理论、邻接通信与梯度传播技术；

**📊 数据集**

通过仿真生成的DC微电网多拓扑和负载变化数据进行训练和测试；未使用公开数据集；

**📈 对比分析**

与基线去中心化主控制器对比，采用峰值电压偏差和积分误差作为评估指标；在插拔、负载变化事件后，峰值偏差下降约60%，积分误差下降15%-76%，显示显著性能提升；

**⚠️ 局限性**

需要预先获取增益上界和网络信息，网络规范化参数化可能过于保守；对极端非线性或模型不匹配情况需额外区域性鲁棒性检查；实现依赖于邻接通信，可能受制于网络拓扑限制。

---

## 55. An Embedded Industrial Irrigation System with Weather-Integrated Intelligent Scheduling

**arXiv ID:** 2610.23179 | [PDF](https://arxiv.org/pdf/2610.23179v1)

**作者:** Kiarash Zangooei `[一作]` (University of Tehran), Hamid Reza Ghaeini `[通讯]` (University of Tehran)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并验证了基于边缘计算的嵌入式天气感知灌溉系统，集成多传感器土壤湿度监测、预报数据融合与自动泵控；

**💡 创新点**

创新点在于将四个分布式电容土壤湿度传感器的中位数融合与短期天气预报相结合，形成去除不必要灌溉的实时决策机制；

**🔧 技术方法**

使用STM32F103C8T6进行低层数据采集，BTT Pi V1.2执行边缘推理、天气API检索、数据库管理及GUI，Python实现主控逻辑，采用UART串行通信协议；

**📊 数据集**

实验数据集为一月（6/13–7/13/2026）菜萝卜生长实验，收集了38 112条聚合测量周期记录及3次灌溉事件的计划体积；

**📈 对比分析**

与传统定时灌溉对比，系统在总灌溉量上约减少32%（26.68 L vs. 39.35 L），但缺乏对照组与流量计验证，评估基于预估流量；

**⚠️ 局限性**

局限性包括仅在单一环境下实验、灌溉事件极少、未独立校准泵流速、未量化预报准确性，且数据无法支持统计显著性结论。

---

## 56. Emergent Discrimination and Polarization in Multi-Agent Systems

**arXiv ID:** 2610.22701 | [PDF](https://arxiv.org/pdf/2610.22701v1)

**作者:** Nestor Caticha `[一作]` (Universidade de São Paulo), Renato Vicente `[通讯]` (Universidade de São Paulo)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构建了一个由学习型代理组成的社会模型，使用贝叶斯在线学习规则更新代理的意见权重和相互信任，并通过对代理群体的数值模拟揭示了在存在群体偏见时出现的四种宏观状态（无歧视极化、偏见与意见一致的歧视、偏见但意见无关的歧视、无群体偏见的自旋玻璃式混乱）。

**💡 创新点**

创新点在于：① 将传统的教师–学生贝叶斯学习转化为可直接在多代理互动中应用的在线更新公式；② 提出三个可观测的秩序参数（意见-信任、信任-群体、意见-群体）以量化宏观状态；③ 通过与大语言模型的上下文学习实验验证理论预测，展示了语言模型在群体互动中潜在的偏见放大机制；④ 构建了完整的相位图，揭示了偏见强度与普及率共同决定系统状态的阈值。

**🔧 技术方法**

技术手段包括：线性感知器（单层）作为意见读出器；基于最大熵的高斯投影实现的在线贝叶斯更新；对称的信任与意见耦合；利用统计力学的自旋玻璃理论和关联函数描述宏观状态；大语言模型（如GPT系列）在上下文学习实验中用于验证意见与信任更新函数。

**📊 数据集**

数据集：实验使用合成议题（P个问题向量），每个议题是语言模型嵌入向量（如GPT的残差流压缩）；对语言模型的实验使用基于人工构造的“虚构档案”与“低情绪主题”进行上下文提示，以收集意见与信任的更新样本；没有使用公开的现实世界社交网络或问卷数据。

**📈 对比分析**

比较方法：将模型预测的更新函数与语言模型实验中测得的意见变化（Δw）和信任变化（Δμ）进行比值归一化后绘制散点图，计算相关系数和符号一致率；结果显示：意见更新与预测函数的Spearman相关系数约0.90，符号一致率94.7%；信任更新相关系数约0.85，符号一致率86.0%。该比较表明理论在捕捉语言模型在上下文学习中的动态行为方面具有较好的一致性，但未给出传统性能指标如准确率、召回率等。

**⚠️ 局限性**

局限性：① 只使用合成议题，缺乏真实人类对话或社会行为数据，难以验证模型对现实社交网络的预测能力；② 语言模型实验受限于单一模型和提示设计，可能不具备普适性；③ 模拟规模有限（N=100），相位边界可能因系统规模或议题复杂度（α）而变动；④ 假设所有代理共享相同的议题表示，忽略了信息异质性；⑤ 仅考虑二元意见，未涵盖多维或连续态的真实观点空间；⑥ 研究未考虑外部干预或治理机制，缺乏对策略干预效果的量化分析。

---

## 57. Learnable Splines for Unstructured Neural Operators

**arXiv ID:** 2610.22446 | [PDF](https://arxiv.org/pdf/2610.22446v1)

**作者:** Vignesh Gopakumar `[一作]` (University College London), Anima Anandkumar `[通讯]` (Caltech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出L‑SplineFNO，一种在大规模不规则网格上实现高效、可微分的神经算子，利用可学习的B样条投影将网格数据映射到隐式的傅里叶网格，并在此上执行FNO。

**💡 创新点**

创新点在于将结构保持的可学习B样条散射与FNO结合，既避免了昂贵的几何相关线性求解，又提供了稀疏、分区统一、C^{p-1}平滑的投影，并通过解析重构实现闭式梯度与严格的空间与参数不确定性界限。

**🔧 技术方法**

技术上采用了可学习的局部B样条散射、傅里叶神经算子、FiLM条件、解析样条重构、Taylor模型与Bernstein包络实现精确界定。

**📊 数据集**

数据集为六个三维工程基准（热散热器、结构支架、航空气动、建筑火灾、热交换通道、HIVE感应加热）以及一份包含29.2M节点的MAST‑U托卡马克完整结构模型。

**📈 对比分析**

与六种代表性无结构或隐式格点算子（Transolver、GNOT、UPT、MSPT、NUNO、GINO）在相同参数量、相同FiLM条件下进行对比，L‑SplineFNO在大多数基准上取得最低相对L2误差，训练时间和显存显著低于注意力模型，且能处理高达4.6千万节点的推理。

**⚠️ 局限性**

局限性包括对尖锐特征和材料接口的平滑化导致精度下降、B样条基函数的展开空间受限、三维高阶样条的存储开销大、以及给出的界限是对代理而非对真实求解的保守估计。

---

## 58. Multiple Turns, Short Horizons: Measuring Effective-Horizon Compression in Jailbreaks

**arXiv ID:** 2610.22928 | [PDF](https://arxiv.org/pdf/2610.22928v1)

**作者:** Zhipeng Wei `[一作]` (International Computer Science Institute), N. Benjamin Erichson `[通讯]` (International Computer Science Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多轮 jailbreak 进度可由较短历史恢复，并提出有效地平衡（effective horizon）概念及其在攻击恢复（EH‑CR）、信用分配（EH‑CA）和防御训练中的应用。

**💡 创新点**

创新点在于揭示多轮攻击对早期查询的依赖可被显著压缩，提出通过截断查询序列测得有效地平衡，并证明短期学习信号即可实现与全程信号相近的攻击与防御性能。

**🔧 技术方法**

采用重放截断查询、改进搜索和训练攻击器/防御器、利用多轮对话评价指标和强化学习奖励窗口等技术。

**📊 数据集**

使用公开评测集：WildJailbreak、HarmBench、StrongREJECT、JailbreakBench、MTID 等。

**📈 对比分析**

与完整历史、峰值历史等基线比较，EH‑CR 在攻击续航上平均提升 8.2–17.4 分，EH‑CA 在攻击训练上平均提升 6.9 分，短期奖励窗口在防御训练中几乎不影响校准后性能。

**⚠️ 局限性**

局限性包括有效地平衡仅在攻击成功后评估、受评判器和容差阈值影响；未直接探索如何自动发现更短的攻击路径；在多代理或更长对话场景中的适用性尚未验证。

---

## 59. Can the Parts Fool the Test? Counterfactual Pairing Cycles for Relational OOD

**arXiv ID:** 2610.23248 | [PDF](https://arxiv.org/pdf/2610.23248v1)

**作者:** Yibo Zhou `[一作]` (Beihang University), Yirong Xiang `[通讯]` (Air Force Medical Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于“counterfactual pairing cycles”的关系 OOD 评估原语，并将其作为训练损失与诊断工具，用于衡量视觉‑语言模型对图像‑字幕配对的理解；

**💡 创新点**

创新点在于构造局部 2×2 循环块，使得图像、文本单独的线性影响在块内完全抵消，仅保留配对依赖的交互；并提供相同对比的训练目标和双线性分解分析；

**🔧 技术方法**

使用了 CLIP ViT‑B/32 的冻结特征，softplus‑基于 Δ 的循环对比损失，双线性与低秩双线性评分，正交投影、线性度量与 ANOVA 交互分析；

**📊 数据集**

实验数据集包括合成对照、GQA 场景图与关系子集，以及 COCO 的图像‑文本配对、标注关系和严格空间/大小关系；

**📈 对比分析**

与传统两样本 OOD、文本/图像单独探针以及 BCE 负样本训练对比；在 COCO 严格关系上，CLIP 0.763、低秩双线性 0.787；循环 OOD 训练得到 0.9270，而 BcE 训练得到 0.8817，显示循环验证更能反映关系迁移；

**⚠️ 局限性**

局限性：仅能检测对配对敏感的交互信号，无法保证完整语义理解；需要预先定义关系谓词和过滤规则；对非双线性或更复杂交互的探测有限；大规模模型的扩展可能需更多计算资源。

---

## 60. Known Music, Unknown Response: Leakage-Audited Dynamic Arousal Forecasting With Scheduled Future Music

**arXiv ID:** 2610.23063 | [PDF](https://arxiv.org/pdf/2610.23063v1)

**作者:** Bin Wen `[一作]` (Universiti Sains Malaysia), Tien-Ping Tan `[通讯]` (Universiti Sains Malaysia)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究如何在音乐播放前就根据已知的未来片段预测未来情绪（arousal）变化，并在严格的评估规则下验证这种预测的有效性。

**💡 创新点**

创新点在于：①将音乐情绪预测定义为已知未来协变量（scheduled future covariate）下的创新（innovation）预测；②提出完整的评估框架（声明预测起点、音频支持隔离、歌曲分离、时序控制），避免泄漏和伪提升；③通过相同歌曲的时间位移和最相似音频对比验证未来位置的特异性；④展示仅利用未来音频（不含过去音频）即可显著提升预测性能。

**🔧 技术方法**

技术手段包括：独立编码的 MuQ 预训练模型（对每个 4 秒音频块独立编码）、PCA 降维、岭回归预测器、非线性状态模型（HGB）、以及各种归因与支持定位诊断。

**📊 数据集**

使用两个公开音乐情绪数据集：PMEmo（794 片段，0.5 秒标注）和 DEAM（1,744 片段+58 全曲，持续标注），其中 DEAM 的 1,494 首完整歌曲作为严格的 hold‑out 评估集。

**📈 对比分析**

方法对比：完整未来+过去的 MuQ 关系表示在 CCC 上比最近情绪状态岭回归提升 0.389–0.507，且比校准后的非线性状态模型提升 0.268–0.382；未来‑仅模型亦高于状态模型 0.150–0.249。相同歌曲的时间位移与最相似音频控制均低于正确位置，证明位置特异性。MAE 也在大幅变化区间内显著下降。

**⚠️ 局限性**

局限性包括：只研究 arousal；使用群体平均标签，缺乏个体化效果；模型仅基于固定 MuQ 表示，未尝试更强的序列微调；仅在两套数据集上验证，未覆盖不同文化或音乐风格；评估规则严苛，结果不直接转化为因果音乐选择；未验证实际播放环境中的实时性能。

---

## 61. Political Narratives in the 2024 Moldovan Presidential Election: Stance, Amplification, and Influence

**arXiv ID:** 2610.22464 | [PDF](https://arxiv.org/pdf/2610.22464v1)

**作者:** Andrea Piras `[一作]` (University of Illinois Chicago), Elena Zheleva `[通讯]` (University of Illinois Chicago)

**通讯引用:** 2066 | [OpenAlex ID](https://openalex.org/A5071079350)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析了2024年摩尔多瓦总统选举期间在X、Facebook和Reddit上的多语言社交媒体帖子，识别政治立场，构建放大与语义网络，并用ProEmb框架估计同伴影响的因果效应。

**💡 创新点**

结合多语言立场标注、协调行为网络和基于代理的因果推断三大方法，首次在地缘政治敏感选举环境下揭示线上叙事传播与同伴影响机制。

**🔧 技术方法**

使用大语言模型Gemini进行多语言立场检测；通过句子嵌入与余弦相似度阈值构建快速转发与语义对齐网络；采用ProEmb近端因果推断框架估计曝光效应。

**📊 数据集**

使用2024年8月1日至10月25日收集的52,696条公开帖子（X 45,695条，Reddit 3,555条，Facebook 3,446条），包含多语言文本、时间戳、互动指标。

**📈 对比分析**

通过不同阈值（1小时转发、0.85余弦相似度）构建网络，并在ProEmb上进行ATE估计，平均处理效应在0.218–0.517区间，显示曝光对立场表达有显著正向因果影响，且模型在10个随机种子上稳健。

**⚠️ 局限性**

仅基于可观测的转发同伴曝光，未覆盖算法推荐或离线信息；因果估计依赖代理假设，未能完全消除未观测混杂；‘pro‑Russia’曝光样本稀少导致估计不稳定。

---

## 62. The Ethics of Prompt Engineering

**arXiv ID:** 2610.23082 | [PDF](https://arxiv.org/pdf/2610.23082v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 63. Regulating for an AI Commons: Adopting, sustaining, and defending people-first innovation trajectories

**arXiv ID:** 2610.23212 | [PDF](https://arxiv.org/pdf/2610.23212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 64. HiPredict: Predicting the missing HiRISE RED4 spectral channel with machine learning

**arXiv ID:** 2610.23335 | [PDF](https://arxiv.org/pdf/2610.23335v1)

**作者:** Ardan Suphi `[一作]` (Imperial College London), Vidhya G Rangarajan `[通讯]` (German Aerospace Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究并提出HiPredict模型，用深度学习重建HiRISE相机在2023年失效的RED4探测器缺失的红光图像；

**💡 创新点**

创新点在于将U‑Net架构与亮度校正头相结合，利用邻近波段、空间重叠信息和环境元数据，实现非线性、空间变换的RED4重建，并显著优于传统线性合成SYN4；

**🔧 技术方法**

采用机器学习中的卷积神经网络（U‑Net+亮度头），自定义损失函数（MAE、梯度差、低频/高频MAE等），并使用SHAP等可解释性分析；

**📊 数据集**

使用HiRISE观测数据共81,717个训练样本（13,625次观测）和23,348个测试样本，包含IR10、BG12、RED3/5等通道以及九个环境元数据；

**📈 对比分析**

通过与SYN4、线性模型、基本U‑Net、定制U‑Net等基线比较，HiPredict在NMAE上达到1.11%（比SYN4的3.12%低2.8倍），SSIM提升至0.725（+9%），PWT 1%提升至20.8%（约+100%），在冰冲击、岭状光谱和氯化物等下游任务中也均显著优于SYN4；

**⚠️ 局限性**

局限在于仍存在高频细节误差，主要受仪器噪声和IR10的binning约束；极地、高入射角和高空轨道场景误差较大；训练数据仅占HiRISE观测量的不到20%，覆盖的表面积有限；目前仅重建RED4，其他通道的重建仍需进一步研究。

---

## 65. From GAN-Based Models to GPT-5: A Taxonomy-Guided Study of Human Performance in Detecting Deepfake Face Images

**arXiv ID:** 2610.23144 | [PDF](https://arxiv.org/pdf/2610.23144v1)

**作者:** Zhiqing Guo `[一作]` (Xinjiang University), Shujun Li `[通讯]` (University of Kent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了跨模型可扩展的 Deepfake Face Artifact Taxonomy (DFAT)，并基于该分类构建通用与模型特定的短视频+交互式训练材料，随后通过线上实验（Prolific）评估人类在检测五种生成模型（StyleGAN2/3、GRAM-HD、SD3.5、GPT-5）生成的面部深度伪造图像时的表现，实验包含主测与五个月后跟进测试；

**💡 创新点**

首次提出可供扩展的跨模型视觉缺陷分类框架 DFAT，证明模型特定训练比通用训练更有效，并揭示最新模型并不一定更难被识别，提供五个月后表现保持不衰的实证；

**🔧 技术方法**

采用人工标注构建 DFAT，设计短视频+交互式练习的训练材料，使用在线实验平台 Prolific 进行大规模实验，分析使用 ART ANOVA 与 Wilcoxon 等非参数统计方法；

**📊 数据集**

使用 FFHQ 真实照片与五种生成模型各生成 100 张 1024×1024 的图像（共 500 张），实验期间再使用新样本进行训练后五个月的测试；

**📈 对比分析**

通过预训练与后训练的准确率比较，比较通用与模型特定训练、单图与双图呈现；结果显示不同模型检测准确率差异显著，SD3.5 与 GPT-5 较易识别；模型特定训练在单图任务中提升显著，整体准确率从约 50% 提升至 70-90% 之间，且未出现显著的情绪负面影响；

**⚠️ 局限性**

样本来源单一（Prolific）可能缺乏真实使用场景；仅覆盖五种模型，DFAT 需要随生成模型更新迭代；实验仅针对静态图像，未考察视频或社交媒体语境；长期持续效果需更长时间验证；训练材料对未来更复杂模型的适用性需要进一步更新。

---

## 66. Rethinking GRPO for Large Reasoning Model Safety Alignment

**arXiv ID:** 2610.23085 | [PDF](https://arxiv.org/pdf/2610.23085v1)

**作者:** Jiawei Chen `[一作]` (East China Normal University), Zhaoxia Yin `[通讯]` (East China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出Beacon框架，通过分析GRPO在大规模推理模型安全对齐中的优势坍塌和不安全多样化两种失效模式，设计了安全信号恢复SSR和安全-效用解耦对齐SUDA两大模块来提升模型安全性并保持推理性能。

**💡 创新点**

创新点在于（1）对GRPO优化机制的理论剖析揭示优势坍塌与不安全多样化的根源；（2）SSR引入优势引导轨迹过滤与安全参考监督恢复有效安全信号；（3）SUDA通过类别感知奖励和仅对良性提示的KL锚定实现安全与效用的解耦，减少安全税。

**🔧 技术方法**

使用GRPO强化学习、优势引导轨迹过滤、Safe Reference Supervision、类别感知奖励设计、Benign-Only KL Anchoring，并借助ArmoRM-Llama3-8B奖励模型评估安全、效用与诚实。

**📊 数据集**

采用ThinkSafe安全训练数据，评估集包含StrongREJECT、HarmBench、WildJailbreak、XSTest（过度拒绝）以及推理基准GSM8K、MATH500、AIME24、GPQA。

**📈 对比分析**

与SFT、GRPO、DPO、NGRPO、GDPO、MDP-GRPO等多种基线对比，Beacon在Qwen3-8B上安全误差降至1.88%，推理pass@1提升至80.64%，并将过度拒绝率压至1.20%，在不同模型规模、对抗攻击及评测者下均保持稳定优势。

**⚠️ 局限性**

局限在于对奖励模型与安全评估器质量高度依赖，主要关注有害/良性提示场景，未覆盖事实性、置信度校准等更广泛的对齐目标。

---

## 67. PAC-KD: Pathology-Aware Compositional Knowledge Distillation for Respiratory Sound Classification

**arXiv ID:** 2610.22826 | [PDF](https://arxiv.org/pdf/2610.22826v1)

**作者:** Li Xiao `[一作]` (Wuhan University), Dan Chen `[通讯]` (Wuhan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种病理感知的组合知识蒸馏框架（PAC‑KD）用于呼吸音分类。

**💡 创新点**

创新点在于使用两个专门的病理专家（crackle 与 wheeze）学习事件属性，并将其补丁知识以属性层面蒸馏给统一的四类学生，同时加入稳定冲突校准以调节两种监督的冲突。

**🔧 技术方法**

采用知识蒸馏、二元专家模型、Bernoulli KL 散度、温度缩放、梯度兼容性评估与冲突校准技术。

**📊 数据集**

在公开的 ICBHI 与 SPRSound 两个呼吸音数据集上进行实验。

**📈 对比分析**

与多种传统方法（Patch‑Mix CL、RepAugment、BTS 等）以及基于四类蒸馏的基线比较，PAC‑KD 在 ICBHI 上 Score 65.28% 最高，在 SPRSound 上 Score 89.54% 领先，主要提升了异常周期的敏感度，特异度略有下降。

**⚠️ 局限性**

局限在于对异常事件的敏感度提升导致特异度下降；实验仅覆盖两套数据集，缺乏对其他多模态或更大规模数据的验证。

---

## 68. Broad and Efficient PII Detection with Unified Labels

**arXiv ID:** 2610.22597 | [PDF](https://arxiv.org/pdf/2610.22597v1)

**作者:** Andy Hsu `[一作]` (Cisco AI Defense), Konstantin Berlin `[通讯]` (Cisco AI Defense)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对12类结构化PII进行统一定义后，对77个公共数据集进行重标注，训练141M参数的高效编码器模型。

**💡 创新点**

通过统一的上下文定义和大规模重标注，构建统一语料并训练轻量模型，在低FPR下实现超过大型基线的召回率，同时显著降低推理成本。

**🔧 技术方法**

使用mmBERT-small编码器，双分类器结构（全局与token级），并利用GPT‑5 Mini进行标签重标注。

**📊 数据集**

来自77个公共数据集（含AI4Privacy、Nemotron‑PII、MAPA等），共112万例。

**📈 对比分析**

在AI4Privacy和Nemotron基准上，低0.1% FPR时召回率领先14项对比中11项，且吞吐量提升2‑10倍，显存消耗降低94‑97%。

**⚠️ 局限性**

仅支持12类PII，重标注依赖LLM可能出现不一致；需要针对不同流量重新校准FPR，模型对某些格式的上下文识别仍需改进。

---

## 69. QuAD: Quantitative Assessment via Distribution-Alignment for Qualitative Data Quality Enabled by a Content-Aware Multi-Agent System

**arXiv ID:** 2610.23153 | [PDF](https://arxiv.org/pdf/2610.23153v1)

**作者:** Wenshuo Zhang `[一作]` (Hong Kong University of Science and Technology), Lin-Ping Yuan `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种定量预分析工具 QuAD，用来评估半结构化访谈/问卷文本数据是否适合某一研究视角（如群组、主题），并输出可视化报告与代表/异常回应。

**💡 创新点**

创新点：1) 四象限评估框架（GMR, DDR, DP, DC）从分布层面同时衡量镜像与差异；2) 基于内容感知的多代理 LLM 评估系统，包含交互式指令生成与自适应锚点维护，显著提升相异度测量稳定性；3) 直接把聚类分布与研究者定义的元数据分布对齐，提供可解释的预分析诊断。

**🔧 技术方法**

使用技术：LLM‑as‑a‑Judge、双阶段指令生成（Oracle–Updater–Evaluator 交互）、自适应锚点维护、距离矩阵的两阶段计算（pairwise 与 scalar 投影）、多维缩放+集成聚类、四象限指标与可视化报告。

**📊 数据集**

数据集：四个领域的可控合成数据（HCI、Finance、Additive Manufacturing、Medicine）；122 条中文食品选择调查回应（含人类标签）。

**📈 对比分析**

比较与性能：在合成数据上，QuAD 与 ground‑truth 的 Spearman/Pearson 相关性分别超过 0.7；相较于 embedding baseline、直接 prompting，相关性提升 10–30%；在真实数据上，与人类标签的 Cramér's V=0.13、label‑mismatch AUC=0.749、top‑3 purity=0.742；人工评估报告准确性/可解释性均为 6/7；成本：122 条标量评分使用 56,727 API token，约 0.0060 美元。

**⚠️ 局限性**

局限：1) 受 LLM 的稳定性与偏差影响，模型升级或服务改动可能导致评分漂移；2) scalar 投影压缩多维语义信息，可能遗漏细粒度差异；3) 四指标需综合解读，单一高分易导致误判；4) 仅验证一份真实语料，需更多多语料与下游编码评估验证。

---

## 70. MobileInject: Evaluating Reasoning-to-Action Security of MLLM-Based Mobile Agents under Cross-Modal Consistency Injection

**arXiv ID:** 2610.23307 | [PDF](https://arxiv.org/pdf/2610.23307v1)

**作者:** Songping Wang `[一作]` (Nanjing University), Caifeng Shan `[通讯]` (Peking University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文提出了 MobileInject，一个可执行的多模态基准，用以评估基于大型语言模型的移动代理在跨模态注入和推理至行动过程中的安全风险。

**💡 创新点**

创新点包括：1）引入跨模态一致性注入（CMCI），揭示视觉与文本同步注入的增强攻击效果；2）设计多角色协同欺骗（MRCD）和推理伪装注入（RCI）以提升信任操纵与推理劫持；3）构建细粒度推理安全诊断框架（RSDF）来定位风险意识、推理劫持与行动劫持。

**🔧 技术方法**

技术实现基于完整功能的 Android 仿真器，结合视觉截图与 XML/UI 结构观测，使用 M3A 与 MobAgent 两大代理框架与多模态大语言模型（GLM、Qwen、Gemini、GPT 等）进行闭环执行，并通过 RSDF 对轨迹进行细粒度诊断。

**📊 数据集**

数据集为 MobileInject 本身，共 108 个可执行实例，覆盖 11 个应用场景、9 类安全风险，并包含 11 条攻击模板与多模态注入载体。

**📈 对比分析**

相较于现有安全基准，MobileInject 通过 UA、AGR、RHR、AHR 等指标展示了多模态代理在真实移动环境中普遍易受攻击，攻击成功率高、推理劫持率超过 70%，而现有基于 prompt 的防御只能提供有限保护。

**⚠️ 局限性**

局限性在于仅评估 Android 平台且缺乏跨模态防御机制，基准实例数量有限，且对多模态观测来源的完整性和真实世界多样性的进一步验证仍待扩展。

---

## 71. BS-Spec: BFP-Based Bit-Sharing for Self-Speculative LLM Decoding with LUT-Driven Mixed-Precision Acceleration

**arXiv ID:** 2610.22877 | [PDF](https://arxiv.org/pdf/2610.22877v1)

**作者:** Jiwoo Kim `[一作]`, Youngjoo Lee `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

未提供论文内容，无法概述

**💡 创新点**

未知

**🔧 技术方法**

未知

**📊 数据集**

未知

**📈 对比分析**

未知

**⚠️ 局限性**

未知

---

## 72. On the Position Bias of Large Language Models: A Comprehensive Study

**arXiv ID:** 2610.22807 | [PDF](https://arxiv.org/pdf/2610.22807v1)

**作者:** Yining Li `[一作]` (Jilin University), Taro Watanabe `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估大型语言模型在多语言、多任务下的位置信息偏差（position bias）

**💡 创新点**

首次在大规模模型（B到T级别）和五种语言上统一量化偏差，并发现模型规模与推理方式（Chain-of-Thought、思考模式）显著抑制偏差

**🔧 技术方法**

采用多种提示策略（0-shot/5-shot、Direct Answer、Chain-of-Thought、Thinking Mode）与精确匹配（EM）评估指标

**📊 数据集**

MMLU、MMMLU、MLQA等公开多语言问答数据集

**📈 对比分析**

对比不同模型规模、提示方式和语言，发现小模型和非英语数据偏差最大；规模增大、CoT/Thinking Mode能将偏差差值降至5%以内，性能提升明显

**⚠️ 局限性**

受限于实验平台与算力，未覆盖所有前沿模型；部分非英语数据样本量有限；过度推理可能导致“过度思考”导致准确率下降

---

## 73. Sink-Sparse Hybrid Attention for Efficient Inference in T2I/T2V Diffusion Transformers

**arXiv ID:** 2610.22975 | [PDF](https://arxiv.org/pdf/2610.22975v1)

**作者:** Bingxin Xing `[一作]` (Xi'an Jiaotong University), Ping Wang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需训练的加速方法——Sink‑Sparse Hybrid Attention，先通过少量离线校准数据提取层/头级别的“sink”掩码，随后在推理时使用该掩码保留高重要性交互，并加入轻量的 pooled‑KV 补偿分支，实现自适应融合。

**💡 创新点**

创新点在于：① 证明稀疏注意力的高质量结构可在少量校准样本中稳定识别；② 采用块级别的“sink”掩码实现可重复使用的稀疏结构；③ 通过轻量化的 pooled‑KV 分支补偿丢失的弱交互，形成混合注意力，兼顾速度与质量；④ 所有改动完全无训练、无微调，直接可迁移到现有模型。

**🔧 技术方法**

技术包括：离线稀疏掩码校准（块级别块重叠统计 + 行覆盖阈值选择）、静态稀疏注意力实现、池化‑KV 补偿分支、基于归一化的自适应融合；核心实现依赖 FlashAttention、RoPE 以及现有 DiT 的 QK‑norm 结构。

**📊 数据集**

数据集：T2V 使用 Wan2.1‑T2V‑1.3B（视频分辨率 832×480，81 帧，50 步）；T2I 使用 Z‑Image‑Turbo（图片分辨率 2048×2048，8 步）。校准集采用 4 个 prompt‑seed 组合（共 8 条完整 denoising 路径）。

**📈 对比分析**

对比方法：Dense（全注意力）、SVG1、SpargeAttn。实验表明：在 T2V 上，τ=0.85 时获得 1.54× 端到端速度提升（时间 106.73s vs 164.80s）并保持 0.659 VBench 平均质量；τ=0.70 时速度提升 1.74×。在 T2I 上，τ=0.85 时获得 1.57× 速度提升（14.88s vs 23.39s）并达到 15.21 GenEval2 分数，优于 SpargeAttn。

**⚠️ 局限性**

局限性：仅在少数 DiT 变体上验证，需为每个模型单独进行离线校准；固定静态掩码在分布漂移或异常 prompt 下可能失效；未评估掩码在不同模型间的迁移效果；在极端稀疏化时仍可能丢失输入高度依赖的交互。

---

## 74. RTSM: Adapting Recursive Models to Multivariate Time-Series Classification

**arXiv ID:** 2610.23353 | [PDF](https://arxiv.org/pdf/2610.23353v1)

**作者:** Georg Tenzing `[一作]` (ETH Zürich), Thorir Mar Ingolfsson `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种 Recursive Time-Series Model（RTSM），通过卷积时间‑tokenizer、单状态共享注意力递归模块以及池化读出实现多变量时间序列分类。

**💡 创新点**

创新点在于将递归权重共享与学习型时间‑tokenizer相结合，构建单状态递归结构，同时使用加权池化读出，使参数与计算深度解耦，显著提升平均准确率。

**🔧 技术方法**

采用卷积 tokenization、共享注意力/MLP 递归模块、深度监督与自适应停止、Stablemax 交叉熵、加权池化等技术。

**📊 数据集**

在 UEA 30 个多变量时间序列分类数据集上进行评估。

**📈 对比分析**

使用 80/20 验证拆分与检查点选择与 MambaSL 等 17 个深度学习基线进行对比，RTSM 在验证协议下平均准确率达到 72.81%，比 MambaSL 高 1.6 个百分点；在官方测试协议下平均 80.82%，同样领先。

**⚠️ 局限性**

局限性包括对极长序列或高维通道数据的效果仍有限，递归深度提升对准确率的边际收益不大；且对不同停止策略和多状态设计的进一步探索仍需深入。

---

## 75. InstAgent: A Recursively Self-Improving Agent Operating System for Heterogeneous Robot Collaboration

**arXiv ID:** 2610.23098 | [PDF](https://arxiv.org/pdf/2610.23098v1)

**作者:** Haoyang Luo `[一作]` (ShanghaiTech University), Jingya Wang `[通讯]` (ShanghaiTech University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一套名为 InstAgent 的机器人协作操作系统，利用行为树（BT）描述协作流程，并通过递归自我改进（RSI）机制在执行过程中学习、改进并保存可重用的协作子树。

**💡 创新点**

创新点在于：①将协作过程建模为可执行的 BT，显式编码机器人间的依赖、交接条件和恢复逻辑；②结合经验驱动和结构化程序两种 RSI，实时修正失败并将验证通过的子树写入经验库；③引入多模态时空记忆 SMAR‑T，支持语义、空间、时间检索，帮助验证交接与决策。

**🔧 技术方法**

核心技术包括：ROS 2 框架、基于 LLM 的语言推理与技能调用（MLLM）、行为树编译与执行、协作编排 HeteroComposer、经验库与子树存储、SMAR‑T 多模态检索工具（位置、时间、语义）、执行反馈循环与验证。

**📊 数据集**

使用了 20 组真实机器人协作任务（G1、Go2、ALOHA 等多种机器人），以及 RAVEN‑QA 公开检索数据集（RealWorld、Habitat‑Sim、IRS、IRS‑hard）进行多模态检索评估。

**📈 对比分析**

与三种实验条件对比：Simple CoT（无 BT）、BT Recovery（无经验）和 Fixed‑Experience BT（仅重用子树）。在 RSI‑Updated BT（E₂）下，任务成功率提升 15–25 个百分点，首次成功率提升 12.5–30 个百分点，平均恢复次数下降 0.7–1.3 次。检索任务中，InstAgent 通过 Agent‑驱动检索实现 93.14 % 成功率，超过 RAVEN 及 ReMEmbR。

**⚠️ 局限性**

局限性包括：①依赖 LLM 的推理质量，错误推理会导致子树失效；②经验库更新需要人工或自动标注，长周期部署下经验质量难以保证；③仅在固定机器人组合和有限任务集上验证，迁移到全新任务或机器人队伍的效果尚未系统评估。

---

## 76. ForTress: Bringing Assurance and Determinism for LLM-Driven Fortran-to-Python Translation

**arXiv ID:** 2610.22528 | [PDF](https://arxiv.org/pdf/2610.22528v1)

**作者:** Qinrun Dai `[一作]` (University of Colorado Boulder), Yueqi Chen `[通讯]` (University of Colorado Boulder)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于 LLM 的 Fortran 到 Python 迁移框架，自动生成可重用的翻译规则并通过 Lean 进行形式化审计，最终实现了 17 个基准（来自 15 个科学软件）中 99.4% 子程序的位比对齐。

**💡 创新点**

创新点在于：① 将 LLM 产生的翻译决策抽象为可复用的可执行规则；② 用 Lean 进行规则的形式化保真性证明与反例检验；③ 通过反例驱动的迭代修正提高规则质量，并实现跨软件 90%+ 的重用率。

**🔧 技术方法**

使用的技术包括：Claude Opus 5.5 LLM、Rule‑Generator/Translator/Executor 三子系统、Auditor 与 Human‑Expert 审计层、Lean 4 + Mathlib 进行语义建模与定理证明。

**📊 数据集**

实验数据集为 17 个基准测试，覆盖 15 个真实科学软件，涉及电子结构、流体动力学、天体物理等 10+ 学科，包含 105,631 行 Fortran 代码。

**📈 对比分析**

对比方法为直接 LLM 翻译基线；结果显示在保持位相同输出的同时，Rule+Assurance 模型 LLM 费用下降 44.2%，并在 14/15 软件中实现 90%+ 规则重用，直接翻译存在 7 个失败案例。

**⚠️ 局限性**

局限性包括：仅覆盖 Fortran/ Python 子语言的子集；对 IEEE 浮点精度未完全建模；对跨语言数据迁移、I/O 与记录字段访问支持有限；未证明所有构造的完整可覆盖性。

---

## 77. Testing the Planning Claims of LLM-Based Agents against Formal Semantics

**arXiv ID:** 2610.22469 | [PDF](https://arxiv.org/pdf/2610.22469v1)

**作者:** Oriol Miro-Lopez-Feliu `[一作]` (Barcelona Supercomputing Center), Sergio Alvarez-Napagao `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 657 | [OpenAlex ID](https://openalex.org/A5028049729)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建基于PDDL的正式验证框架，对LLM的提示策略和代理架构在规划任务中的执行进行系统评估。

**💡 创新点**

创新点在于：①将离线规划与在线规划‑by‑acting两种代理范式在同一语义下对齐并证明在线不一定优于离线；②引入不可解任务与预期成本分析，强调对执行性、失败模式和检测率的细粒度评估；③提供公开的5个领域、50个自然语言任务的 benchmark，支持对规划能力的客观衡量。

**🔧 技术方法**

使用的技术包括：链式思维（CoT）、Plan‑and‑Solve、Tree of Thoughts、Self‑Refine、ReAct、Reflexion、Critic 等提示与代理设计；PDDL 定义任务、VAL 进行语义验证；自定义模拟器和工具接口实现在线执行；Token 计数与成本计算。

**📊 数据集**

使用了 50 任务数据集，覆盖 5 个规划领域（Blocksworld、Gripper、Logistics、LoadBalancing、Hospital），每个领域 10 任务（8 可解 + 2 不可解），任务以自然语言说明生成，背后隐藏 PDDL 模型。

**📈 对比分析**

评估方法：对每个模型（GPT‑4o‑mini、GPT‑4o、Gemini Flash‑Lite、Gemini Flash）在 6 种提示策略和 3 种代理架构下进行一次完整运行，统计可执行率、成功率、检测率、Token 使用及成本。结果显示：离线策略中 chain‑of‑thought 在 GPT‑4o 上提升至 77.5% 成功率，其他策略提升有限；在线策略最高 40% 成功率，且预条件违规占主导。成本方面，Tree of Thoughts 等昂贵策略在性能上无优势，所有在线配置均落后于最佳离线配置。统计检验发现，在 Holm 校正后无任何策略/架构显著优于基线。

**⚠️ 局限性**

局限性：①基准规模仅 50 题；②任务为确定性、全观测、单智能体，缺乏不确定性、随机性和多智能体情境；③仅使用四个模型，且在温度为 0 的单次运行下评估，未覆盖不同温度或多次重跑的方差；④提示与架构实现依赖于作者对原文的重现，可能与原设计存在差异；⑤不可解任务数量有限，可能低估了系统识别不可解性的能力。

---

## 78. Input- and Feature-Space Correction for Cross-Hospital IHC Grading

**arXiv ID:** 2610.22756 | [PDF](https://arxiv.org/pdf/2610.22756v1)

**作者:** Jongwoon Lee `[一作]` (University of Rochester), Christopher Kanan `[通讯]` (University of Rochester)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文比较了输入空间和特征空间无标签校正方法在跨医院IHC分级中的效果，使用了两种冻结基础模型（UNI、Virchow2）、三种IHC标记（HER2、Nectin-4、TROP2）以及12个跨机构转移设置。

**💡 创新点**

创新点在于提出了基于源补丁PCA的FIX输入空间校正方法，并证明其与特征空间标准化（FS）组合可提供互补优势，显著提升跨医院迁移性能且不需要更新基础模型或使用目标标签。

**🔧 技术方法**

采用的技术包括PCA投影、FIX组件对齐、Reinhard、Macenko、Mean Shift、Feature-wise Standardization、CORAL及其Ledoit‑Wolf变体，以及冻结基础模型编码。

**📊 数据集**

使用了7,499张IHC切片（HER2、Nectin-4、TROP2），来自机构A、BCI（公共数据集）和两种基础模型（UNI、Virchow2）的嵌入。

**📈 对比分析**

在12个跨机构设置下进行比较，指标为宏F1、QWK和序数误差，结果显示FIX+FS组合在宏F1提升0.181、QWK提升0.208、序数误差下降0.209，相较于无校正实现显著性能提升。

**⚠️ 局限性**

局限性包括仅评估了三种标记、两种基础模型和有限的机构对，未在其他新型基础模型或更广泛的多中心数据上验证，FIX的门限参数未单独外部验证，部分患者ID缺失导致分组方式不完备。

---

## 79. Critical-state algorithms for lot sizing with all-units discounts and variable inventory bounds

**arXiv ID:** 2610.23253 | [PDF](https://arxiv.org/pdf/2610.23253v1)

**作者:** Kleitos Papadopoulos `[一作]` `[通讯]`, Kleitos Papadopoulos

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了针对单品连续周期库存批量折扣问题的精确求解算法，涵盖任意时间变动价格、线性持有成本和周期性库存上限；

**💡 创新点**

创新点在于利用“关键状态”定理将可行状态压缩至有限网格，并在每层使用双前缀最小化实现 O(n^3) 时间（单阈值）以及 O_m(n^{m+2}) 时间（固定 m 阈值）的解法；同时给出基于库存格的伪多项式实现和从单阈值单调松弛得到的可证完整性检验；

**🔧 技术方法**

关键技术包括累计采购坐标变换、极值点网格构造、双前缀最小扫描、状态空间缩减、库存格动态规划以及基于单调阈值松弛的证书构造；

**📊 数据集**

使用自生成的随机整数和有理实例，覆盖单阈值、局部阈值、固定全局阈值、库存格宽度等多种情形，验证精度与性能；

**📈 对比分析**

与全枚举 DP、层级 LP、以及库存格实现进行比较。实验显示在容量紧张时，O(n^3) 方法比库存格快数十倍；当容量宽松或阈值较多时，库存格实现更具优势；总体实现均采用 exact‑rational Python，时间和空间均符合理论复杂度；

**⚠️ 局限性**

局限性：对任意局部阈值的情况，状态集可能指数级增长；算法不支持设置费用、产能限制、缺货或非线性持有成本；单调松弛的证书在有限容量或粗粒度阈值时不一定通过；

---

## 80. FlashControl: One-Step Controllable Generator via Distillation-Friendly Single-Stream Teachers

**arXiv ID:** 2610.22537 | [PDF](https://arxiv.org/pdf/2610.22537v1)

**作者:** Ngan Nguyen `[一作]`, Anh Tran `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种单步可控图像生成框架 FlashControl，首先通过 DeltaControl 在原始文本‑图像扩散模型中仅微调 feed‑forward 层实现空间控制，随后利用 DMD 蒸馏将该教师压缩为单前向网络；在此过程中加入 AutoAdapter 对不同控制信号进行统计匹配与 token‑级门控，并采用 Latent Variance Preservation 防止单步模型收敛为确定性映射。

**💡 创新点**

创新点包括：① 单流可控教师 DeltaControl，避免传统双分支 ControlNet 的蒸馏难点；② AutoAdapter 通过分布匹配与门控实现多种控制类型的统一对齐；③ Latent Variance Preservation 目标在单步蒸馏中保持噪声敏感度；④ 结合 DMD 与 LoRA 使得单步生成速度极快（0.042 s/图）。

**🔧 技术方法**

使用了扩散模型（UNet/DiT）+ DMD 级别的分布匹配蒸馏；LoRA 参数微调；AutoAdapter 的统计匹配模块与 token‑gate；Latent Variance Preservation 损失；标准的 VAE 编码/解码框架。

**📊 数据集**

主要评估基准为 COCO‑2017‑5k，训练时使用 Stable Diffusion 2.1、PixArt‑α、FLUX 等大型文本‑图像预训练模型。

**📈 对比分析**

与 ControlNet、T2I‑Adapter、UniControlNet 等双分支多步方法以及 FlashSD、JDM、SDXS 等单步方法对比，FlashControl 在 6 种空间控制（Canny、Depth、Seg、Pose、Hough、Inpainting）上 1 NFE 时 FID 与多步教师相当，且远优于对手；推理时间 0.042 s/512×512 图像，速度提升超过 40×。

**⚠️ 局限性**

局限性：目前主要支持单一控制条件，跨条件组合仍待探索；Latent Variance Preservation 需手工调参；对极稀疏或非典型控制类型（如高阶纹理）支持有限；模型依赖大规模 backbone，部署成本仍较高。

---

## 81. Proportionality in Ranking Compression

**arXiv ID:** 2610.23193 | [PDF](https://arxiv.org/pdf/2610.23193v1)

**作者:** Daniel Halpern `[一作]` (Google Research), Warut Suksompong `[通讯]` (National University of Singapore)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

提出了一种将大量投票者排序压缩为更少排序的新框架，并给出了三种比例性公正度量（位置比例性、对偶比例性、固体联盟比例性）及其在压缩任务中的可行性与不可行性结果。

**💡 创新点**

①证明位置比例性与固体联盟比例性始终可实现且可多项式时间求解；②对偶比例性在一般情况下不可满足，给出了反例；③在单峰或单交叉结构下对偶比例性可实现，并可在相应结构下保持其他两种比例性。

**🔧 技术方法**

使用组合学工具（如 Hall 定理、网络流、流的整数化定理）构造求解算法；通过图论与路径覆盖方法求解固体联盟比例性；对偶比例性则用约束图与环分析证明不可满足性。

**📊 数据集**

本文并未使用具体真实数据集，而是以理论模型（投票者偏好排名）作为实验基准；主要关注算法的存在性与复杂度。

**📈 对比分析**

在可实现的比例性（位置比例性与固体联盟比例性）下，给出了多项式时间算法并证明其最优性；在单峰/单交叉实例中，同样可在多项式时间内构造满足所有比例性要求的输出，显示了结构假设下的优势。

**⚠️ 局限性**

局限性包括：①对偶比例性在一般情况不可保证；②对偶比例性在输出来自输入排名的限制下更难满足；③实际应用中仅提供稀疏或部分排名时，理论框架与算法的适用性尚未验证；④对比例性量化的更细粒度或近似保证尚待研究。

---

## 82. I2V-DETACH: Source Grounding Detachment for Unauthorized Image-to-Video Generation

**arXiv ID:** 2610.23287 | [PDF](https://arxiv.org/pdf/2610.23287v1)

**作者:** Chanhui Lee `[一作]` (POSTECH), Jeany Son `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向扩散式图像-视频生成的免疫化框架，利用源‑图像与视频潜在表示的注意力分离与视频内交互增强，生成不可逆的微小扰动以阻断源图像信息在视频生成过程中的传播。

**💡 创新点**

核心创新点是：①源‑图像与视频之间的注意力预激活 logits 直接抑制（source‑video decoupling loss）并鼓励视频内部自注意力（intra‑video coupling loss），②设计源‑发散视频潜在初始化以提供既与源相关又非重复的优化信号，③将两项损失结合在单张源图像的白盒对抗训练中，显著降低源图像在生成视频中的内容保留。

**🔧 技术方法**

技术方法包括扩散式图像‑视频生成模型（如 CogVideoX、LTX‑Video、OpenSora），基于注意力 logits 的对抗优化，PGD 逐步更新的源‑发散视频潜在，联合源‑视频解耦与视频‑视频耦合的损失函数，及对抗扰动在 ℓ∞ 限制下的最小化。

**📊 数据集**

使用 VBench 数据集（100 张图像‑文本对，包含 80 人像与 20 非人像），在每个模型上随机采样前 10% 的扩散步骤进行训练与评估。

**📈 对比分析**

与 I2VGuard、Anti‑I2V、EditShield、AdvPaint 等基线进行对比，采用 IS、SC、DS 视觉一致性指标、质量指标 AQ/IQ、以及人类评估（Source Reflection、Protection Assessment、Overall Effectiveness）。实验显示本方法在所有指标上均显著优于基线，且在对抗去化、黑盒迁移和 UAP 扩展上表现出较强鲁棒性与迁移性。

**⚠️ 局限性**

局限性包括：①需要白盒访问目标扩散模型，且单张图像的对抗优化计算成本较高；②对某些模型（如 OpenSora）在视觉质量与不可感知性上略逊；③对抗扰动对生成质量的负面影响在极端条件下仍可能出现；④UAP 版本虽有效但相对单张图像优化仍有性能差距。

---

## 83. Look, Reason, Draw a Region: Recursive Refinement for Spatial Context Prediction

**arXiv ID:** 2610.23302 | [PDF](https://arxiv.org/pdf/2610.23302v1)

**作者:** Wajeeha Tahir `[一作]` (ETH Zürich), Thorir Mar Ingolfsson `[通讯]` (ETH Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练 Spatial-TRM 以从冻结的视觉语言特征中通过递归细化预测自然图像中相对目标的空闲空间区域。

**💡 创新点**

采用权重共享的递归细化网络与凸多边形解码器，仅 11.2M 可训练参数即可提升上下文准确率，并在冻结编码器条件下实现显著性能。

**🔧 技术方法**

结合冻结的 DINOv3/CLIP 视觉‑语言编码器、跨注意力锚点定位、Tiny Recursive Model 递归推理、凸半空间解码、类权重交叉熵与对比损失以及自适应停止技术。

**📊 数据集**

在 RoboSpatial 训练集的空间上下文子集上训练，并在更具挑战性的 RoboSpatial-Home 进行评估。

**📈 对比分析**

与单步模型、前馈模型及多种基线 VLM/机器人模型对比；在 RoboSpatial-Home 上获得 29.51% 上下文准确率，较单步提升 9.02pp（44%），总参数量约 15 倍减小至 877.8M。

**⚠️ 局限性**

预测仅为单一凸多边形，无法捕捉孔洞或分离区域；缺乏占据/碰撞监督；对小/远目标的定位敏感；评估依赖稀疏点注释，可能掩盖区域错误。

---

## 84. When the implementation matters more than the algorithm: NSGA-II and other multi-objective optimizers in engineering design tools

**arXiv ID:** 2610.22397 | [PDF](https://arxiv.org/pdf/2610.22397v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 85. DICL-Qwen3VL: Decoupled Confidence Calibration for Interpretable Workwear Detection

**arXiv ID:** 2610.22929 | [PDF](https://arxiv.org/pdf/2610.22929v1)

**作者:** Meng Wang `[一作]` (China University of Geosciences), Chao He `[通讯]` (Shenyang Aerospace University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一个可解释的工作服检测框架DICL-Qwen3VL，集成LoRA微调、可解释证据生成和视觉/推理分离置信度估计。

**💡 创新点**

通过分离视觉与推理置信度并使用Token熵、图像扰动KL进行视觉置信度监督，结合GRPO联合优化检测、输出格式和校准，提供可解释证据和分离置信度。

**🔧 技术方法**

采用Qwen3‑VL视觉语言模型、LoRA低秩适配、解耦置信度估计、Token熵与扰动KL监督、GRPO策略优化以及结构化输出模板等技术。

**📊 数据集**

使用公司采集的ICSWD（9446张）和网络收集的WICWD（1535张）两大数据集进行训练与评估。

**📈 对比分析**

与多种LoRA微调的VLM基线（Qwen2.5‑VL、LLaVA‑NeXT‑Video、InternVL3.5、GLM‑4.6V、Qwen3‑VL）对比，DICL‑Qwen3VL在ICSWD上mAP@50为0.730、WICWD为0.753，均显著优于基线。

**⚠️ 局限性**

目前仅关注上半身可见服装，需进一步扩展到多源场景、实时监控和更多工作服种类，并未对多模态校准做更广泛的评估。

---

## 86. Capital Costs Protect the Network: Mining Gap Dynamics in Proof-of-Work Blockchains Under a Stochastic Mempool

**arXiv ID:** 2610.22451 | [PDF](https://arxiv.org/pdf/2610.22451v1)

**作者:** Hasan Selçuk Kılıç `[一作]` (Bilkent University), Nail Akar `[通讯]` (Bilkent University)

**通讯引用:** 1304 | [OpenAlex ID](https://openalex.org/A5080807022)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了一个基于高费交易残余的Lindley式递归模型，推导出矿工停工间隙（mining gap）的分布与期望，并结合比特币的难度调整机制分析了矿池哈希功率、奖励结构和费用对矿工激活与网络安全性的影响，进一步给出了自由入退出条件下的长期哈希功率均衡与安全度量。

**💡 创新点**

创新点在于：①提出并证明“我的激活规则”在非原子矿工模型中为弱占优策略；②将两档费用市场与矿工阈值阈值化的存留池递归映射为带底阈值的Lindley过程，得到闭式的稳定分布；③分析难度调整映射的平衡点及其局部稳定性；④通过资本成本与费用溢价的平衡推导出自由入退出下的哈希功率上界，解释了“挖矿空隙”出现的经济根源。

**🔧 技术方法**

主要技术包括：非原子博弈理论、均值场（mean‑field）方法、随机排队与Lindley递归、Lambert W函数求解特征方程、局部线性化与弹性分析、闭式期望计算与积分变换。

**📊 数据集**

使用比特币链上数据（区块奖励、交易费用、区块容量与哈希率）估算模型参数，并结合从2026年到2070年的减半进程与多种价格/费用情景进行数值演示。

**📈 对比分析**

方法主要通过解析推导与闭式公式，与以往仅依赖仿真的“空隙游戏”模型相比，显著降低计算复杂度；数值结果显示在高哈希率下空隙长度可逼近整轮时间，攻击者所需的相对哈希率随费用溢价与资本成本显著变化。

**⚠️ 局限性**

局限性包括：假设所有矿工拥有相同的运营成本与资本成本，忽略高份额矿池的策略影响；模型仅考虑不出现分叉的理想化链；难度调整的分析为局部稳定性，不涵盖周期性振荡；对存留池的稳定分布近似忽略了每个epoch初期的短暂过渡期。

---

## 87. DriftTTS: One-stage Single Step Text-to-Speech via Generative Drifting

**arXiv ID:** 2610.22960 | [PDF](https://arxiv.org/pdf/2610.22960v1)

**作者:** Jaehyun Ahn `[一作]` (University of Seoul), Heeseung Kim `[通讯]` (University of Seoul)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DriftTTS，一种基于生成漂移的单步文本到语音模型，在训练时通过片段漂移直接预测目标梅尔谱。

**💡 创新点**

创新点在于将生成漂移迁移到 TTS，移除迭代采样，采用片段漂移解决可变长度问题，并在无预训练模型的情况下实现单步高质量合成。

**🔧 技术方法**

使用了生成漂移、片段漂移、Masked Autoencoder（MAE）特征提取、Monotonic Alignment Search（MAS）、HiFi‑GAN vocoder、MeanFlow、Consistency Flow Matching 等技术。

**📊 数据集**

使用 LJSpeech 语音数据集进行训练和评估。

**📈 对比分析**

与 Matcha‑TTS（10 步）、RapFlow‑TTS（2 步）以及单步 MeanFlow/RapFlow 进行对比，WER 与 MOS 与多步模型相近，RTF 仅为 4.4×10⁻³，速度比多步模型快 1.3–4.2 倍。

**⚠️ 局限性**

局限在于对 MAE 特征提取器的依赖，单说话人训练，未解决多说话人和零样本合成；对高维梅尔空间直接漂移会导致能量漂移，需进一步分析。

---

## 88. Fast BSS Transition for Multi-PSK Community Wi-Fi in MDU Networks: Fragmented RADIUS Key Delivery, Server-Derived PMK-R0/PMK-R1 Distribution through a Cloud Management API, and Lease-Driven Key Revocation

**arXiv ID:** 2610.23116 | [PDF](https://arxiv.org/pdf/2610.23116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 89. LEMON: Equivariant Learning over Attention Maps for Zero-Shot Misbehavior Monitoring on New LLMs

**arXiv ID:** 2610.22723 | [PDF](https://arxiv.org/pdf/2610.22723v1)

**作者:** Guy Bar-Shalom `[一作]`, Haggai Maron `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于注意力图的零样本监控模型，可在线检测大型语言模型的误行为。

**💡 创新点**

创新点在于将多头注意力的层‑头对齐问题建模为代数对称（wreath product 𝒮_H≀𝒞_L），并构造相应的 equivariant 图神经网络（LMON）实现跨模型无监督迁移。

**🔧 技术方法**

使用图神经网络（MPNN）、对称等价层、注意力图构造图结构、token 概率特征，以及基于注意力的概率、对齐与概率混合的变体。

**📊 数据集**

在 16 个 decoder‑only LLM（135M–13B）和 6 个数据集（RAGTruth、Mu‑SHROOM、PsiloQA 等）上进行评估，涵盖幻觉和越狱两种误行为。

**📈 对比分析**

与基线（CHARM、隐藏层探针、注意力比例、概率、对齐）相比，LMON 在零样本交叉模型设置下平均提升约 10–14 ROC‑AUC，在线推理额外开销 < 8%，且能在生成过程中提前中断错误输出。

**⚠️ 局限性**

局限包括：仍需在每个 LLM 提取注意力；对其他任务（如文本生成检测、记忆化）尚未充分验证；多模态或更大规模模型的适配仍待探索。

---

## 90. Deterministic Bounds on Accuracy and Stability of Metric-Space Regression and Classification

**arXiv ID:** 2610.22564 | [PDF](https://arxiv.org/pdf/2610.22564v1)

**作者:** Nina Maria Gottschling `[一作]` (Oak Ridge National Laboratory), Michele Caprio `[通讯]` (University of Warwick)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5000098655)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于可行集几何的平均核尺寸框架，用于在有限数据集上给出回归与分类的确定性误差与准确度下界和上界。

**💡 创新点**

创新点在于把输入-目标模糊性量化为平均核尺寸，并证明其能给出模型无关的可达误差区间，并与 Bayes 风险关联。

**🔧 技术方法**

技术包括集合值映射、Hausdorff 距离、上半连续性、Lipschitz 稳定性分析以及基于最近邻的可行集构造算法。

**📊 数据集**

实验使用了多种表格与图像回归数据集（如 Tropical Cyclone、Concrete Strength、Energy Efficiency）以及分类数据集（MNIST、CIFAR‑10、RESISC45、EUROSAT）。

**📈 对比分析**

通过将理论区间与深度网络、随机森林等基准模型在原始与构造数据集上的误差/准确度进行比较，表明区间往往与模型性能一致或给出可观的改进空间。

**⚠️ 局限性**

局限在于区间依赖于所选输入伪度量与阈值、可行集构造的近似、缺乏对数据生成过程的统计一致性保证，以及无法直接解决模型容量与优化难度问题。

---

## 91. Fluid-Spatiotemporal Stochastic Geometry for Information Flow in Non-Stationary Fields: Kinematics and Field Reconstruction

**arXiv ID:** 2610.23277 | [PDF](https://arxiv.org/pdf/2610.23277v1)

**作者:** Wen-Yu Dong `[一作]` (China Telecom Research Institute), Sheng Chen `[通讯]` (University of Southampton)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出F-STSG框架，基于密度演化的逆边值问题重构宏观传输，并得到信息通量、扩散率与质心动力学。

**💡 创新点**

在可观测密度的前提下给出唯一最小能量速度代表，区分可辨与不可辨的旋转分量；提供解析扰动分析、Green函数与谱分解，并将连续传输与离散基站网络耦合。

**🔧 技术方法**

质量守恒 PDE、加权最小能量原理、椭圆 Neumann 问题、有限差分/有限元求解、Green函数与谱分解、统计误差与敏感度分析。

**📊 数据集**

合成随机场实验、公开 pNEUMA 城市交通轨迹数据、仿真 UAV 控制实验。

**📈 对比分析**

与质心平移基线比较，pNEUMA 内部交叉率 MAE 从 0.877 降至 0.766，RMSE 从 1.076 降至 0.965；UAV 控制在延迟与执行扰动下相对基线提升约 12% 服务距离，同时出现轨迹偏差与障碍违规。

**⚠️ 局限性**

只能重建密度可观测的宏观速度，旋转成分不可辨；需已知或可估计的源项与边界通量；对高噪声/低观测量下精度下降；实验基于离散网格与时间更新，未验证在真实移动通信网络中的完整闭环性能。

---

## 92. ORD-WM: Learning Object-Centric Rigid Dynamics for 3D World Modeling in Robotic Manipulation

**arXiv ID:** 2610.23156 | [PDF](https://arxiv.org/pdf/2610.23156v1)

**作者:** Yueyang Weng `[一作]` (Harbin Institute of Technology), Yanjie Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 ORD-WM，一种基于对象中心的 3D 世界模型，能够在潜在空间内通过 SE(3) 变换实现几何保持的多步预测，并支持实时闭环模型预测控制。

**💡 创新点**

创新点在于将几何感知的向量神经元 VAE（SDF‑VNVAE）与动作条件 SE(3) 运动预测结合，利用潜在空间中的刚性变换保持对象形状不变，解决了传统点云或图像潜在空间中易出现的形状漂移和不一致问题。

**🔧 技术方法**

核心技术包括：SDF‑VNVAE 以 SO(3) 等变的向量特征构造几何感知潜在；Transformer‑基运动模块预测每个浮动对象的 SE(3) 增量；在潜在空间中解析应用 SE(3) 变换；以及基于 CEM 的模型预测控制。

**📊 数据集**

使用 LIBERO（含 Spatial、Object、Goal、Long 等子集）和 RoboCasa 进行 VAE 训练；使用 LIBERO 轨迹（包含 1,000 条推箱子轨迹）进行世界模型训练；真实实验采用 Franka Research 3 机器人和 RealSense D435 摄像头。

**📈 对比分析**

与公开的 PointWorld 及其 fine‑tune 版本相比，ORD‑WM 在 1 步、10 步和完整轨迹的点误差与形状一致性 CD 上均表现更好，误差约为 1/5；在闭环 MPC 中，候选数 1024 时成功率 90%，平均最终误差 0.97 cm，规划时间 270 ms，显示出可在实时场景下高效规划。

**⚠️ 局限性**

局限性：仅适用于低动力、刚体运动；无法处理关节或柔性物体；未显式强制接触约束，可能出现碰撞；在更大规模、多样化的数据集上尚未评估。

---

## 93. Causal Edit Serialization

**arXiv ID:** 2610.22820 | [PDF](https://arxiv.org/pdf/2610.22820v1)

**作者:** Louis Robinson `[一作]` `[通讯]` (Independent Researcher), Louis Robinson (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Causal Edit Serialization (CES)，在单一因果解码器中实现可编辑的指令流，支持插入、删除、替换、复制、移动等编辑操作。

**💡 创新点**

创新点在于将源地址指针与双索引旋转位置编码相结合，使模型在生成时能够直接预测编辑目标位置并依赖之前的指针，从而实现局部编辑而非全文重写。

**🔧 技术方法**

使用了轻量级定位头、双索引 RoPE、标准因果注意力、交叉熵+地址损失的预训练目标、随机腐蚀+GPT生成修复序列的自监督训练，以及强化学习/监督后训练用于语法错误纠正。

**📊 数据集**

训练数据主要来自 ClimbMix 文本用于自监督编辑，CoEdIT 数据集用于语法错误纠正评估。

**📈 对比分析**

在控制修复任务中，与 AR-DSL、flat-relative、全文本重写、FIM、迭代 Levenshtein 等基线比较，CES 在程序有效率、精确恢复率、部分修复相似度提升等指标上优于基线；在语法错误纠正上，后训练 CES 在 GPT‑5 Mini 判别下实现正净改进。

**⚠️ 局限性**

限制包括精确恢复率仍低、插入恢复弱、对高密度编辑的鲁棒性不足、仅在 1.4B NanoChat 规模上测试、缺乏与更大 LLM 的迁移、评价仅基于单一评审且未进行独立人类判断。

---

## 94. Retrospective Evaluation of Bluetooth RSSI Fingerprinting for Bin Localization on an Industrial Shop Floor

**arXiv ID:** 2610.22680 | [PDF](https://arxiv.org/pdf/2610.22680v1)

**作者:** Saif Lakhani `[一作]` `[通讯]` (University of California Davis), Saif Lakhani (University of California Davis)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估BLE RSSI指纹数据在工业车间的箱子定位效果，重建归档数据并验证定位精度。

**💡 创新点**

创新点在于提出区分已测与未测位置的评估协议，系统性比较随机森林、k近邻和多边定位三种方法。

**🔧 技术方法**

使用随机森林回归、k近邻回归、校准路径损耗多边定位以及中继端Kalman滤波预处理技术。

**📊 数据集**

使用2019年收集的24个固定中继的RSSI指纹，19,034条两分钟样本，覆盖90个已知箱子坐标的训练集。

**📈 对比分析**

通过70/30行随机拆分和位置分组拆分评估，随机森林在已测位置的中位误差为0.66 m，95%误差≤10 m；在未测位置中位误差升至9.42 m；多边定位误差近19 m，性能显著逊色。

**⚠️ 局限性**

局限性包括数据仅来自单一工厂一次测量，预处理滤波导致误差被低估，缺乏充分中继覆盖和环境漂移，无法实现连续精确定位。

---

## 95. Data Provenance Aggregation and Structural Descriptors for Query-Template Identification

**arXiv ID:** 2610.22705 | [PDF](https://arxiv.org/pdf/2610.22705v1)

**作者:** Paulo Pintor `[一作]` (University of Aveiro), José Moreira `[通讯]` (University of Aveiro)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用数据 Provenance 聚合技术，对完整查询执行的行级及 Why-provenance 进行集成，构建查询级描述符，进而实现查询模板识别。

**💡 创新点**

提出将 Tuple 级 Provenance 通过 set‑based 与 bag‑based 聚合转化为查询级表示，并设计 Provenance Structural Descriptors（PSD）等紧凑统计特征，探讨了不同粒度的聚合对模板识别的影响。

**🔧 技术方法**

利用 Provenance 聚合、Jaccard/余弦相似度、Nearest‑Neighbour、Logistic Regression、Random Forest、XGBoost 等机器学习算法，对 Provenance 表示进行特征提取和分类。

**📊 数据集**

在 TPC‑DS 参数化模板上生成 12,500 条独立查询实例（10 个模板 × 1,250 个实例），并为每条查询实例记录 Provenance。

**📈 对比分析**

在 10,000 条训练集与 2,500 条测试集上比较，PSD+XGBoost 取得 99.88% 的准确率，Binary Lineage+Logistic Regression 取得 99.96% 的准确率；PSD 的测试处理时间低于 0.01 ms/执行，远低于传统相似度计算；相对成本与训练规模呈线性增长。

**⚠️ 局限性**

局限包括：需要对每条查询执行获取 Provenance 产生的前置成本；Derivation 细粒度并不总能提升准确率；模型对不同工作负载、数据规模和查询多样性可能不具备良好泛化；在模板外的查询或动态变化的工作负载中效果尚未验证。

---

## 96. Look Back, Correct Forward: Error Feedback Calibration for Time Series Forecasting

**arXiv ID:** 2610.22872 | [PDF](https://arxiv.org/pdf/2610.22872v1)

**作者:** Haonan Yang `[一作]` (National University of Defense Technology), Zhuo Li `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于已验证预测误差的全测试时校准框架EFC，在部署期间对冻结的时间序列预测模型进行后置修正。

**💡 创新点**

创新点在于将重叠预测误差构建为因果误差对角（CED），通过累积误差传输（CET）学习误差状态到未来残差的稳健映射，并利用时间残差学习（TRL）进一步捕捉残差的时间结构，从而实现不改模型参数的实时校正。

**🔧 技术方法**

采用Causal Error Diagonal、Cumulative Error Transport、Temporal Residual Learning、在线累积统计、离线可靠性评估以及轻量级GRU预测等技术。

**📊 数据集**

在ETT系列、Electricity、Exchange、Solar-Energy、Weather、Traffic等常用时序任务以及TIME基准的50个数据集和98个任务上进行实验。

**📈 对比分析**

与PatchTST、iTransformer、TimeBridge、DLinear、PatchMLP等先进骨干，以及PIR、COSA、PETSA、TAFAS等后置/测试时适配方法对比，EFC在大多数数据集和骨干上平均降低4.16% MSE、提升8–17% MSE，并在92个TIME任务中获得73.4%的平均提升。

**⚠️ 局限性**

在部分任务改进有限，需要足够多已成熟误差样本以稳定估计误差传输，累计估计在分布急剧变化时响应较慢；目前仅适用于确定性预测，对概率预测尚未扩展。

---

## 97. The Model Knows Better: Defending Medical Multi-Agent Systems Against Prompt Injection

**arXiv ID:** 2610.22594 | [PDF](https://arxiv.org/pdf/2610.22594v1)

**作者:** Fardin Ahsan Sakib `[一作]` (George Mason University), Ozlem Uzuner `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文评估了医疗多智能体系统在面对提示注入攻击时的脆弱性，并提出了一种基于临床SOAP分离的无训练防御方法。

**💡 创新点**

创新点在于将临床文档中的观察与解读（诊断、建议等）分离，并仅保留被观察支持的解读，从而过滤注入文本，形成四步无训练防御流程。

**🔧 技术方法**

利用大型语言模型（Qwen、Llama、MedGemma）分别作为标签器、筛选器和检查器，构建检测与修正注入的防御链；与多种现有提示预防、检测与训练防御方案进行对比。

**📊 数据集**

实验数据来自MedQA、MedMisBench和MPIB三大医学问答/推荐基准，涵盖多选题与自由文本答案。

**📈 对比分析**

与七种基线防御（提示式预防、PromptGuard 2、DataSentinel、PromptArmor、Meta‑SecAlign等）以及单模型/多智能体系统对比，防御将攻击成功率从约28–60% 降至 4–7%，同时清洁准确率几乎无下降，显著优于现有方法。

**⚠️ 局限性**

局限包括：只能检测记录中的文本，无法防御更改检测值的注入；在最难题目上检查器偶尔误删真实解读；实验仅基于公开问答基准，未覆盖真实临床记录；攻击者若能针对防御模型优化文本，仍可能突破。

---

## 98. Domain-specific language alignment of multimodal embeddings towards explainable predictions

**arXiv ID:** 2610.23110 | [PDF](https://arxiv.org/pdf/2610.23110v1)

**作者:** Gabrielė Tijūnaitytė `[一作]` (Wageningen University & Research), Ioannis N. Athanasiadis `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用可链接的辅助数据库为抽象嵌入生成部分文字说明，并通过软对比损失将嵌入空间与 CLIP 文本空间对齐，实现对 Earth Embeddings 的语言解释性提升；随后在两组物种分布模型（S2BMS 与 SatBird）上检验其可解释性与预测性能。

**💡 创新点**

提出“软对比损失”处理部分说明、基于概念检索的排名评估，以及混合/分离说明策略，使得任何可与解释性数据关联的抽象嵌入都能被语言对齐并可解释。

**🔧 技术方法**

使用 2 层 MLP 投影器、CLIP 文本编码器、软对比损失、概念检索任务以及多种说明生成函数；在 AlphaEarth 嵌入上实现语言对齐。

**📊 数据集**

S2BMS（英国蝴蝶 62 物种，1329 位置）与 SatBird（美国夏季 670 种鸟类，122K 位置）数据集；辅助数据包括 CORINE 土地覆盖、生物气候变量、人类足迹、PM2.5 等。

**📈 对比分析**

与 GeoCLIP、SatCLIP、Tessera、DOFA‑CLIP 等 Vision‑Language 模型及多种基线（均值、线性回归、MLP、Sentinel‑2 图像预测）比较。AlphaEarth 在 6/7 评价指标上击败所有基线；语言对齐后的 AlphaEarth 在概念检索上平均 I_top‑k ≈0.45，显著高于 DOFA‑CLIP 的 0.29，且与辅助变量的线性可解性呈正相关。

**⚠️ 局限性**

对齐质量在不同概念间差异大；部分概念检索接近随机，原因是对比学习只能对齐嵌入空间中共同共享的语义；CLIP 文本编码器对领域特定语言敏感度有限；并非所有辅助变量在 AlphaEarth 中均能良好编码，导致部分概念对齐受限。

---

## 99. Identify, Classify, Act: Per-Instruction Controlled Compliance for LLMs over External Data

**arXiv ID:** 2610.22874 | [PDF](https://arxiv.org/pdf/2610.22874v1)

**作者:** Zesheng Shi `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 EDIC 框架，使 LLM 能在外部数据中对每条嵌入指令进行识别、分类并做出相应行动，从而兼顾安全与效用。

**💡 创新点**

创新点包括：1）基于 per‑instruction 信用归因的判别式评估器；2）通过混合上下文数据与必要性过滤器训练模型；3）使用分解奖励（分别奖励益处指令遵循、惩罚有害指令遵循）实现安全-效用平衡。

**🔧 技术方法**

核心技术包括：多阶段数据合成与对照式验证、生成式判别器（SFT+GRPO）进行 per‑instruction 合规判定、链式思维（CoT）策略模型、判别器引导的强化学习与奖励分解。

**📊 数据集**

使用了四类场景驱动的混合数据：电子邮件（Spam Email）、API 文档（API‑Bank）、医学指导（EPFL 临床指南）、知识检索（HotpotQA、Wiki‑MultihopQA），以及公开攻击基准（BIPIA、AgentDojo、InjecAgent 等）。

**📈 对比分析**

与六种现有防御（SecAlign、StruQ、PromptGuard、Instruction Hierarchy、Meta‑SecAlign、ReasAlign）在五个安全与效用基准（Arena‑Hard‑Auto、MT‑Bench、AgentDojo、CyberSecEval2、InjecAgent）上对比。EDIC 在有害拒绝率、善意遵循率以及 Arena/MT‑Bench 效用方面均显著优于所有基线，尤其在 Qwen3‑4B 上提升 7.7pp 安全、12.5pp 效用。

**⚠️ 局限性**

局限性：1）仅覆盖四种场景，难以推广到更长轨迹、复杂 HTML 或非英语文本；2）使用固定注入触发器，可能对自适应攻击不足；3）判别器仅给出二元合规标签，无法处理模糊情况；4）奖励权重需人工调参；5）实验仅在 4B/8B 规模，未验证更大模型或多轮工具交互的效果。

---

## 100. Data-Efficient Classifier Pretraining Through Diffusion Models

**arXiv ID:** 2610.23289 | [PDF](https://arxiv.org/pdf/2610.23289v1)

**作者:** Qinyu Li `[一作]` (University of Oxford), Joel Jaskari `[通讯]` (Tampere University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用扩散模型对分类器骨干进行自监督预训练，以在数据稀缺场景下提升分类性能。

**💡 创新点**

创新点在于将扩散模型的噪声去除目标直接用于预训练分类器，不需要额外无标签数据，也不依赖复杂的数据增强，能够在多模态（图像、时间序列）下统一使用。

**🔧 技术方法**

技术主要包括DDPM扩散框架、基于残差网络的U‑Net式解码器、时序嵌入、轻量级解码器和跳跃连接，以及与传统自监督方法（SimCLR、DINO、AE、DAE）的对比实验。

**📊 数据集**

使用了六个数据集：CIFAR‑10、Tiny ImageNet（自然图像）；PTB‑XL、WISDM（时间序列）；EyePACS、HAM10000（医学影像）。

**📈 对比分析**

在低数据子集（1%、5%、10%等）下，DiffPT与SimCLR、DINO、AE、DAE进行了对比。结果显示：
- 在医学影像和时间序列任务中，DiffPT 在低数据 regime 上往往优于所有自监督方法，并可与或略低于 SimCLR；
- 在完整数据集上，SimCLR/ DINO 与训练从零相当，而 DiffPT 仍保持竞争力；
- 计算成本方面，DiffPT 的内存占用约为 1.24×，吞吐量约 0.89× 训练从零的基线，远低于 SimCLR/DINO 的 2×+内存和 0.5×吞吐量。

**⚠️ 局限性**

局限性包括：在全量数据或对象占据图像大部分时，SimCLR 等对比方法仍能取得更好结果；DiffPT 对超参数（如噪声调度、时间步嵌入）敏感性未充分探究；在极端小样本（1%）下性能仍受限。

---

## 101. Learning to Be Guided: Shaping Latent Geometry for Classifier-Free Guidance

**arXiv ID:** 2610.22885 | [PDF](https://arxiv.org/pdf/2610.22885v1)

**作者:** Haoming Liu `[一作]` (New York University), Hongyi Wen `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Prototype Regularization (PR)，在潜在空间中通过聚合原型的余弦分类器正则化来优化类几何，从而提升在Classifier‑Free Guidance (CFG) 下的生成质量。

**💡 创新点**

创新点在于将类几何的影响理论化，利用Gaussian类模型解析CFG校正，设计轻量级PR正则化，并验证其在多种标签、模态和连续条件下的广泛适用性。

**🔧 技术方法**

采用潜在扩散模型中的VAE训练阶段加入原型正则化，使用Gaussian类模型推导的CFG校正与后验交叉熵目标，评估指标包括gFID、sFID、IS、Precision/Recall等。

**📊 数据集**

主要使用ImageNet 256×256进行图像生成实验，UCF‑101视频数据集用于视频生成，CLIP/Florence文本描述用于文本到图像生成。

**📈 对比分析**

与LDM、VA‑VAE、DINO‑aligned VAE等基线对比，PR在CFG下将gFID从3.31降至2.28，超越DINO‑aligned VA‑VAE（2.44）；在视频上FVD从152.7降至140.1；在文本条件下gFID提升约0.3，显示出显著的性能提升。

**⚠️ 局限性**

主要局限在于未指导生成的提升有限，PR仅对全局通道聚合进行约束，未充分调节空间细节，可能影响高频细节；此外，时间分布与分类器近似的设计空间仍待进一步探索。

---

## 102. TriSAR-3D and PaintFusion: A Novel Trimodal LiDAR-RGB-Thermal Dataset and Method for 3D Person Detection in Search and Rescue

**arXiv ID:** 2610.22509 | [PDF](https://arxiv.org/pdf/2610.22509v1)

**作者:** Markus Kramer `[一作]` (Technical University of Darmstadt), Oskar von Stryk `[通讯]` (Technical University of Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建了包含128通道激光雷达、RGB相机和热像相机的三模态数据集TriSAR-3D，并提出了轻量级的PaintFusion三模态3D人检测框架。

**💡 创新点**

TriSAR-3D提供多姿态、多时段、室内外高分辨率三模态注释，PaintFusion采用早期融合+MLP评分的方式，在精度和延迟上均优于现有三模态方法。

**🔧 技术方法**

利用YOLOv26进行2D实例分割，将掩码投影到激光点云，使用MLP对每个点进行概率评分，然后通过PointPillars+CenterPoint生成3D框。

**📊 数据集**

主要使用自研的TriSAR-3D数据集进行训练与评估，并与公开的LiDAR-only基线、BEVFusion、PointPainting等方法进行对比。

**📈 对比分析**

在TriSAR-3D测试集上，PaintFusion实现AP@50 65.4，显著高于CenterPoint（53.4）和BEVFusion（60.5），同时帧率约51.6 ms，是其他三模态方法的一半。

**⚠️ 局限性**

实验使用固定传感器平台，未考虑机器人运动导致的运动模糊，也未包含雾、尘、烟等灾害环境，且对姿态稀缺的样本高度依赖数据集多样性。

---

## 103. A Generalized Half-Singleton Bound for List Decoding of Insertion/Deletion Errors

**arXiv ID:** 2610.23270 | [PDF](https://arxiv.org/pdf/2610.23270v1)

**作者:** Neehar Verma `[一作]` (Aalto University), Razane Tajeddine `[通讯]` (Aalto University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

推导了线性码在列表解码插入/删除错误时的通用半Singleton上界，并给出了满足该上界的单插入/删除码构造。

**💡 创新点**

提出了针对固定列表大小的全局上界，并证明了当码不包含全1向量时该上界严格；同时给出了实现此上界的构造方案。

**🔧 技术方法**

采用生成矩阵缺失列分析、子空间维度计数与归纳证明、puncturing技巧以及线性代数方法。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

与传统Singleton、半Singleton界面及已知的单独纠错方案进行对比，证明构造实现了理论上最优的码率。

**⚠️ 局限性**

仅适用于 q≥3 的非二进制场；对二进制情况给出较弱上界；且仅考虑单个插入/删除误差，未涵盖多误差场景。

---

## 104. Veil-Then-Reveal: Protecting PII Privacy via Reversible Pseudo-Anonymization in Zero-Trust User--LLM Interaction

**arXiv ID:** 2610.22800 | [PDF](https://arxiv.org/pdf/2610.22800v1)

**作者:** Mohamed Shaaban `[一作]` (Washington State University), Thomas E. Carroll `[通讯]` (Pacific Northwest National Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Veil-then-Reveal（VtR）框架，在与远程 LLM 交互时将用户提示中的 PII 用同类可读占位符伪匿名化，发送给 LLM 后在本地恢复原始 PII。

**💡 创新点**

创新点包括：① 将格式保持加密（FPE）与键控语义替代相结合，实现既保留结构又保持可恢复性；② 引入 rank-class FPE 处理日期，既保持合法性与歧义，又显著减少加密开销；③ 采用会话级键生成与本地 vault，提供跨会话 unlinkability 与完全可逆恢复；④ 全程无服务端改动，满足零信任模型。

**🔧 技术方法**

技术手段包括：NIST 标准 FF1/FF3 FPE、HKDF 生成会话密钥、HMAC 产生语义占位符并解决冲突、基于 Faker 的实体池、rank-then-encipher 日期方案、本地恢复 vault 与解密逻辑。

**📊 数据集**

评估使用四个公开数据集：ai4privacy、PUPA、WildChat、ShareGPT，共计 11,322 个标注 PII 实例。

**📈 对比分析**

与 11 种现有基线（实体替代、局部重写、差分隐私等）比较。VtR 在 PUPA 上将泄漏率从 95.1% 降至 27.0%，保留率 95%；在 ai4privacy 上实现 97.9% 的检测召回、95.2% 的保护率、100% 的还原率；日期方案 rank-class FPE 约 410–730 倍加速、有效率 88.4% 以上。整体性能表现优于大多数基线，兼顾隐私与实用性。

**⚠️ 局限性**

局限性包括：① 依赖 PII 检测召回，未检测到的 PII 仍可能泄漏；② 只能恢复 LLM 直接返回的占位符，若 LLM 修改或抛弃占位符则无法还原；③ 语义占位符仅保证实体类型与语法形态，可能失去细粒度上下文关系；④ 未覆盖多模态输入、工具调用或长会话状态等更复杂场景。

---

## 105. Sentence-Level Topic Modeling with Lexical and Contextual Evidence

**arXiv ID:** 2610.23388 | [PDF](https://arxiv.org/pdf/2610.23388v1)

**作者:** Federica Granese `[一作]` (Université Côte d’Azur), Charles Bouveyron `[通讯]` (Université Côte d’Azur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种句子级主题模型SentenceETM，能够同时利用词表层的词汇证据和句子嵌入的上下文证据，为每个句子分配主题。

**💡 创新点**

创新点在于：①将主题同时建模为词向量分布和句子嵌入空间的低维高斯子空间；②句子主题在生成式层面被精确边缘化，后验结合两种证据；③通过变分推断和闭式更新实现高效训练；④提出了基于合成语料的句子级评估协议。

**🔧 技术方法**

技术方法包括：变分自编码器（Amortized Variational Inference）对文档主题比例进行推断；词向量化的ETM框架用于词汇分布；句子嵌入使用预训练句子编码器（如SBERT、InstructGPT）并对其进行L2归一化；高斯子空间建模通过奇异值分解得到子空间基底，并以闭式形式更新均值、方差和子空间矩阵。

**📊 数据集**

主要使用的语料库：BBC News、20 Newsgroups、Wikipedia用于构造合成句子级标签；政治辩论数据集ElecDeb60to20（2020年美国总统辩论）用于案例分析。

**📈 对比分析**

与传统词级LDA/ETM、文档级BERTopic、句子级senLDA和SentenceLDA等方法比较。实验结果表明：在合成数据集上，SentenceETM在ARI和AMI上均领先，提升约0.3以上；在真实辩论文本中，能够揭示候选人对同一话题的不同侧重点，并计算句子内主题多样性，表现出较高的解释性和稳定性。

**⚠️ 局限性**

局限性包括：①模型对句子嵌入质量高度依赖；②缺乏对句子顺序和主题转移的建模，难以捕捉连贯性；③合成数据集虽可控但可能与真实句子主题分布存在差距；④在小样本或低资源场景下，主题识别效果可能下降。

---

## 106. Clinically Informed Acoustic-Aware Attribute Modeling for Respiratory Sound Classification

**arXiv ID:** 2610.22854 | [PDF](https://arxiv.org/pdf/2610.22854v1)

**作者:** Xinhong Li `[一作]` (Wuhan University), Weiping Tu `[通讯]` (Wuhan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了面向呼吸音分类的 Acoustic‑Aware Asymmetric Framework（AAAF），通过将四分类问题重构为裂痕与喘鸣的属性检测，并在推理阶段逻辑组合恢复最终类别；

**💡 创新点**

创新点在于（1）将标签的组合结构显式建模为两条临床属性（裂痕、喘鸣），减少学习负担；（2）采用分支专属多层融合和声学感知的非对称时序聚合，分别捕捉裂痕的稀疏瞬时特征与喘鸣的持续连续模式；（3）引入辅助全局异常/正常监督与批量自适应动态损失，提升不平衡数据下的鲁棒性；

**🔧 技术方法**

使用预训练的 BEATs 编码器提取层级声学特征；对裂痕分支采用 Top‑K 时序池化 + MLP；对喘鸣分支采用平均池化 + 频道注意力；训练时采用 Adam + cosine 余弦调度，批量自适应加权交叉熵；

**📊 数据集**

在 ICBHI 2017（920 条记录，6,898 个呼吸周期）和 SPRSound（2,683 条记录，9,089 个事件）两个公开数据集上进行评估；

**📈 对比分析**

与多种现有基线（CNN‑MoE、ResNet‑50、AST、BEATs 等）以及最新的跨模态预训练方法（如 CLAP、SCL）进行对比，AAAF 在 ICBHI 四分类任务上实现 Score 65.30%（S_p 80.25%/S_e 50.35%），在 SPRSound 上实现 Score 88.54%/92.21%，均为当前最佳或同等水平；

**⚠️ 局限性**

局限包括（1）对不同声学属性的分支仍依赖手工设定的 K 值与聚合方式，可能不适用于所有语音异常类型；（2）未考虑临床元数据与采集设备差异导致的分布漂移；（3）在极端类别不平衡情形下，动态损失仍可能不足以完全补偿少数类学习难度。

---

## 107. Improving OCR Accuracy Through Multi-Classifier Verification of Visually Similar Characters

**arXiv ID:** 2610.22726 | [PDF](https://arxiv.org/pdf/2610.22726v1)

**作者:** Nuzhat Khan `[一作]` (Universiti Teknologi Malaysia), Ibrahim Yousef Alshareef `[通讯]` (Universiti Teknologi Malaysia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了字符级验证模型（CVM），在 OCR 之后对视觉相似字符对进行二分类验证与纠正。

**💡 创新点**

创新点在于针对预定义的视觉相似字符对构建专门的二分类器，完全不依赖词典或语言模型，适用于序列号、车牌等非词典文本。

**🔧 技术方法**

使用随机森林二分类器对 43 组视觉相似字符对进行训练，并结合 OCR 引擎输出的字符边框与原始图像提取字符补丁进行验证。

**📊 数据集**

训练数据来源于公开的 94‑class ASCII OCR 数据集（50 字体、300+样本/字符），测试集为 10 个不同字体的随机字符串。

**📈 对比分析**

在 Tesseract、EasyOCR 与 TrOCR 三种 OCR 引擎上与 10 种字体进行比较；CVM 对 VSC 错误的纠正率为 87.3%，误更正率仅 2.3%，平均提升 6.05%（Tesseract/EasyOCR）或 9.39%（TrOCR），显著降低整体字符错误率。

**⚠️ 局限性**

局限性包括只能处理单字符替换错误，无法纠正插入、删除或多字符分割错误；对完全同形字符（如 I/1/l）无法区分；在真实扫描或拍摄图像、未见字体下的鲁棒性待验证。

---

## 108. "Can Computing Systems Eventually Disappear?": A Long-Term Tracking Study on the Use of Diabetes Computing Systems by Patients with Diabetes

**arXiv ID:** 2610.23276 | [PDF](https://arxiv.org/pdf/2610.23276v1)

**作者:** Yibo Meng `[一作]` (Cornell University), Chengxi Zang `[通讯]` (Cornell University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过八年、五波纵向访谈，跟踪37位糖尿病患者在连续血糖监测、实时反馈与自动胰岛素投递三代技术中的使用与认知演变，探究长期使用对患者主动解读与决策的影响；

**💡 创新点**

提出并理论化“持续使用伴随认知脱离”这一新型用户状态，揭示技术自我替代与认知退化的机制与后果，并结合设计、临床与工程视角提出保持解释性参与的设计原则；

**🔧 技术方法**

采用质性访谈、纵向轨迹编码与主题分析等人机交互研究方法，对用户、设计者、临床与供应商多方视角进行叙事梳理；

**📊 数据集**

使用37名糖尿病患者的访谈数据（2017、2019、2021、2023、2025共五波）以及15名医生、设计师、供应商的单波访谈，构成跨时间与跨角色的数据集；

**📈 对比分析**

本研究并未通过实验或量化指标比较算法性能，而是以编码一致性（Fleiss κ≈0.81）和主题共识度为质量评估，展示了多阶段认知转变的质性“表现”，不涉及传统性能数值；

**⚠️ 局限性**

样本规模有限，主要来自单一地区，缺乏跨文化或多机构验证，未进行干预实验验证设计建议，且研究基于自述数据，可能存在记忆偏差与自我选择偏倚；

---

## 109. TCDA: Tunable Coded Distributed Arrays for Data Availability Sampling

**arXiv ID:** 2610.22508 | [PDF](https://arxiv.org/pdf/2610.22508v1)

**作者:** Duc A. Tran `[一作]` (University of Massachusetts), Duy Le `[通讯]` (University of Massachusetts)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 TCDA（Tunable Coded Distributed Array），一种用于区块链数据可用性采样的可调节编码分布式数组，并证明其在任意数量 Byzantine 节点下的鲁棒性。

**💡 创新点**

创新点：①给出了仅基于总良好节点数的完整鲁棒性条件；②通过保留深度 Q 实现 PUT、存储与 GET 成本的可调权衡；③在同一网格上实现与 RDA 相同鲁棒性但显著降低 PUT 与存储成本；④提供闭式设计算法与性能上限，支持动态调参。

**🔧 技术方法**

使用技术：二维 Reed–Solomon（张量）码、随机线性网络编码（RLNC）、KZG 可同态承诺、随机预言机指派编码系数、网格 P2P 拓扑，以及对应的 PUT、GET、JOIN、TUNE 协议。

**📊 数据集**

数据集与实验环境：以太坊 EIP‑4844 参数为基准——32 MiB 对象、M=128、κ=64、N 从 868 到 40 000，上传速率 25 Mbps、保留窗口 131 072 块，比较 TCDA 与 RDA 与 PeerDAS。

**📈 对比分析**

比较方法与性能：通过仿真和理论分析，TCDA 在满足 PUT 与存储预算的前提下，在 868–40 000 良好节点范围内实现 δ=1/4 鲁棒；存储减少 1.7–32 倍；GET 需要平均 2.5–7.7 轮；RDA 在相同规模下无法满足 10 s 验证截止时间；PeerDAS 缺乏鲁棒性，易被 Sybils 攻击；TCDA 在任何 Sybils 数量下仍保持鲁棒。

**⚠️ 局限性**

局限性：GET 需要第二跳，导致采样延迟比 RDA 高；当大规模良好节点流失超过约 75% 时无法恢复鲁棒；需要预先设定最大网络规模；对列内节点分布的最坏情况仍以全量节点为上限；未解决对列内均匀分布假设的优化。

---

## 110. Literal Lock-In: How Test Language Shifts Literal Attraction in Typed Decision Models on Bengali Idioms

**arXiv ID:** 2610.22526 | [PDF](https://arxiv.org/pdf/2610.22526v1)

**作者:** Adib Sakhawat `[一作]` (Islamic University of Technology), Syed Rifat Raiyan `[通讯]` (Islamic University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在Bengali习语上插入字面注释等干预，审计了三种决策模型（编码器、解码器和闭源API），测量了字面吸引率、校准退化与顺序敏感性等指标。

**💡 创新点**

创新之处在于：①提出了“意义选择”工具，将已有的字面注释作为对照干预；②展示了字面吸引率高度依赖测试语言而非模型能力；③揭示开放模型在信心倒置、校准下降以及选项顺序/标签对结果产生显著影响。

**🔧 技术方法**

采用决策模型的概率输出进行干预评估，构造多种干扰族（随机、表面匹配、语义匹配、加入字面等），并计算字面吸引率、ECE、AURC、Brier、McNemar检验等统计；同时实现无文本“语义自由控制”o4。

**📊 数据集**

使用Bangla Bagdhara 10,361条专家注释的Bengali习语数据集，包含金标准隐喻意义和字面注释，构成4选1测试集。

**📈 对比分析**

在同一任务、同一语言下与公开模型（多语言Laya-ml、Lod-lille）以及商业API Jev-1.13进行对比，评估准确率、字面吸引率、校准等；实验显示开放模型在Bengali情境下字面吸引率高、校准下降且信心反向，而商业模型在English表现更佳。

**⚠️ 局限性**

局限性包括：仅在零样本下评估；模型训练任务与评测任务可能不匹配导致偏差；语言不对称与字面提示可能影响结果；仅用单一语料库和少数模型，结果缺乏普适性；未深入探究置信度校准内部机制。

---

## 111. CoTA++: Understanding the Repeat Curse in dMLLMs from an Information Flow Perspective

**arXiv ID:** 2610.23290 | [PDF](https://arxiv.org/pdf/2610.23290v1)

**作者:** Qiyan Zhao `[一作]` (Chinese Academy of Sciences), Da-Han Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对扩散多模态大语言模型（dMLLM）在使用近似KV缓存时出现的“Repeat Curse”（重复生成）问题，提出并实现了训练无关的CoTA++框架，包含三种干预模块：CTAR、DAR和CTEV，分别解决注意力路由、KV时效性和层级熵散布三类信息流失效；

**💡 创新点**

创新点在于：①通过信息流可视化定位缓存导致的三类失效（F1、F2、F3）；②为每类失效设计了精确的无训练干预模块（CTAR重构注意力、DAR预刷新关键位置、CTEV熵加权投票）；③将三模块组合为可插拔的CoTA++，在不改模型结构的前提下显著降低重复率；

**🔧 技术方法**

使用技术包括信息流追踪与量化（注意力分布、KV新旧度、层级熵）、基于相似度的缓存刷新与位置预刷新、熵加权投票机制、并行解码流程中的缓存重用与重调度；

**📊 数据集**

实验数据集涵盖：MS‑COCO（长文本描述）、DocVQA、ChartQA、MMStar、MME（认知与感知）、SEED‑Bench、MMBench、LLaVA‑Bench、MathVista、MathVerse等九大多模态基准；

**📈 对比分析**

评估方法：在三种dMLLM（LLaDA‑V、MMaDA、LaViDa）和两种缓存（dLLM‑Cache、SlowFast）上测量重复率（ARR、SRR）、最长重复长度、整体质量（人类打分）和速度/内存占用。结果显示CoTA++将最长重复长度降至≤3，ARR降至<0.5%，SRR从>90%下降至≤5%，质量差距≤0.34分，速度提升保持在15%以内；

**⚠️ 局限性**

局限性：仅针对有限的缓存策略与模型规模评估；未探索更大模型或其它缓存方法的适用性；参数（κ、r、λ）需手工调优，缺乏自适应预算分配；跨域或更复杂多模态任务的鲁棒性尚未验证。

---

## 112. Adaptation of Computer Use Agents to New Software with User Manuals

**arXiv ID:** 2610.23012 | [PDF](https://arxiv.org/pdf/2610.23012v1)

**作者:** Chulhoon Yoon `[一作]` (Korea University), Woohwan Jung `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MAGE框架，通过利用用户手册在执行时生成任务计划并在训练时构造可执行任务，提升计算机使用代理的跨应用性能。

**💡 创新点**

创新点在于将手册作为可检索的程序化知识同时用于运行时规划和离线强化学习，消除了对人工演示数据的依赖。

**🔧 技术方法**

使用检索增强的规划、语言模型（LLM）生成计划、GRPO强化学习、视觉语言模型评估以及Qwen3/ GPT-5.4等技术。

**📊 数据集**

基准数据集包括OSWorld的五个应用（LibreOffice Calc/Impress/Writer、GIMP、VSCode 等）以及新构建的 Slack 30 任务对。

**📈 对比分析**

与原始 CUAs 基线相比，MAGE 在 OSWorld 的任务成功率平均提升约 10–20%，在 Slack 提升超过 20%，且改进具有统计显著性。

**⚠️ 局限性**

局限在于手册内容可能过于抽象或与实际 GUI 版本不符，需要额外环境交互来对齐；对极端动态或需隐私权限的应用效果尚未验证。

---

## 113. ProPerAgent: Toward Memory-Driven Proactive Assistance

**arXiv ID:** 2610.22671 | [PDF](https://arxiv.org/pdf/2610.22671v1)

**作者:** Shenglai Zeng `[一作]` (Michigan State University), Xin Luna Dong `[通讯]` (Meta Reality Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了一种针对记忆驱动主动助手的框架和基准，探索何时何种记忆才使助手正确发声。

**💡 创新点**

发现正确记忆与正确时机是相互独立且都必不可少的，并构建了 ProPerMem 基准和 ProPerAgent 解决方案，将无训练模型的准确率从<10%提升至42.4%。

**🔧 技术方法**

结合知识图谱记忆、检索门控、LLM工具调用以及轻量级RL后训练。

**📊 数据集**

使用合成的一年长个人记忆和 Ego4D 视角视频，构成1,804对样本（含150负样本）。

**📈 对比分析**

与多种基线（无记忆、摘要、RAG等）对比，在窗口化与流式两种推理模式下，ProPerAgent在Perfect评价上分别从7%–10%提升至37.5%（无训练）和42.4%（后训练）。

**⚠️ 局限性**

记忆为文本合成、单一触发窗口、未包含多模态真实长视频以及未评估系统延迟等限制。

---

## 114. VISTA: A Multimodal Egocentric Benchmark for Goal-Oriented Perception and Assistance to Blind and Low-Vision Users

**arXiv ID:** 2610.22822 | [PDF](https://arxiv.org/pdf/2610.22822v1)

**作者:** Noé Mosca `[一作]` (Harvard University), Mengyu Wang `[通讯]` (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 VISTA，一个多模态前视视角辅助基准，评估 VLM 在生成目标导向指导上的能力。

**💡 创新点**

将 RGB、眼动、IMU、磁力计/气压计、手部追踪等六种传感器同步加入，并提供场景描述、问答与目标导向指令的三层注释。

**🔧 技术方法**

使用预训练视觉语言模型（如 InternVL3、Qwen3‑VL、Gemma 3）结合 sensor‑fusion + LoRA 适配技术实现多模态输入融合。

**📊 数据集**

使用 VISTA 数据集（约 1,000 条 Project Aria 眼镜录制，覆盖 10 类辅助任务，含场景描述、问答、指导三种标注）。

**📈 对比分析**

通过 GPT‑6 Luna 自动裁判评估场景描述、问答和指导三种输出；零射击模型表现欠佳，基线改进后整体分数提升约 20‑30%，其中指导分最高。

**⚠️ 局限性**

局限性包括数据规模有限、由视障模拟者采集、评估仅基于单一自动裁判，缺少实时部署和真实用户安全性验证。

---

## 115. Anytime Contracts for AI Services

**arXiv ID:** 2610.22781 | [PDF](https://arxiv.org/pdf/2610.22781v1)

**作者:** Boris Velasevic `[一作]` (California Institute of Technology), Eric Mazumdar `[通讯]` (California Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究AI服务中的道德风险，提出了针对不确定停止时间（Anytime）合同的理论与算法，旨在激励服务提供者始终使用高质量模型。

**💡 创新点**

创新点包括：①将合同理论推广到未知停止时间环境；②证明预算最优合同等价于任何时刻有效的似然比 e‑process；③给出可通过动态规划高效求解的算法；④在缺乏先验分布时利用投注 e‑过程近似最优合同。

**🔧 技术方法**

主要技术：主从（Principal‑Agent）游戏建模、合同理论、似然比检验、e‑process 与 Ville 不等式、动态规划、投注（betting）e‑过程。

**📊 数据集**

实验使用合成的独立同分布 Bernoulli 过程以及真实大型语言模型（LLM）的性能数据，数据来自 Holistic Agent Leaderboard（SWE‑Bench Verified Mini）。

**📈 对比分析**

与传统固定期限或单步支付合同对比，Anytime 合同在预算最优、支付更少的同时保持激励兼容；实验表明在 Claude 系列模型上，Anytime 合同既能阻止低质量模型，又能在不同停止策略下保持较低的拒付概率。

**⚠️ 局限性**

局限性：假设委托方具备足够议价权且停止规则可被整合；仅考虑两种路由策略，未涵盖多策略情形；对连续信号或非离散分布的推广尚待研究；需要对模型分布或至少其一阶统计量有一定估计。

---

## 116. CFC26: Building Evaluations for Deployment in Sonar-Based Fish Counting

**arXiv ID:** 2610.22549 | [PDF](https://arxiv.org/pdf/2610.22549v1)

**作者:** Madison Van Horn `[一作]` (California Institute of Technology), Michael Hobley `[通讯]` (California Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CFC26数据集与评估协议，以更真实地模拟鱼类声纳计数的部署环境。

**💡 创新点**

创新点包括部署导向的分布式评估（ID/OOD）和两个新的计数指标nMNE与nMANE。

**🔧 技术方法**

使用YOLOv5检测器结合ByteTrack跟踪，对视频进行鱼类跨越线计数。

**📊 数据集**

数据集覆盖9条河流、11个部署点，包含4,363段视频和1,007,877个高质量框架框。

**📈 对比分析**

在ID/OOD设置下，模型在多数河流实现nMAE≈0.1-0.3，然而在Eel、Chemainus等地点仍出现大方向性偏差。

**⚠️ 局限性**

局限包括单次实验缺乏方差估计、仅限河流声纳场景、无法覆盖极端浊度或其他传感器类型。

---

## 117. When Plans Change, Remember: Replanning-Guided Visual Memory for Vision-Language-Action Models

**arXiv ID:** 2610.22556 | [PDF](https://arxiv.org/pdf/2610.22556v1)

**作者:** Victoria Smelova `[一作]` (MIRAI), Alexey K. Kovalev `[通讯]` (MIRAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RePlanMem-VLA 方法，利用视觉‑语言‑动作（VLA）策略在多阶段长时程任务中的重规划差异作为“惊讶”信号，在线检测关键帧并将其写入受限视觉记忆；同时加入机器人遮罩与聚类抑制冗余写入；实现无标注、实时的记忆更新机制。

**💡 创新点**

①将连续重规划的动作块重叠预测差异视为内部事件信号；②基于该惊讶信号实现在线关键帧选择与记忆写入；③结合机器人遮罩与完整链聚类，抑制因机器人运动导致的视觉冗余；④实现了从计划修订到记忆写入的闭环流程，无需外部事件标注。

**🔧 技术方法**

采用 Transformer‑based VLA 策略（OpenVLA‑OFT、π_0.5）进行动作块预测；设计多种计划修订度量（分布式、累计、因果）；计算马氏距离惊讶得分并对任务进行校准；使用阈值触发写入；对候选帧执行完整链聚类与机器人遮罩过滤；在分布式监督微调中仅对最终动作块进行梯度传播。

**📊 数据集**

主要使用 MIKASA‑Robo‑VLA 任务集（19 任务/8 子任务组合），并在真实机器人回放数据上验证算法的鲁棒性。

**📈 对比分析**

通过 ROC‑AUC、PR‑AUC 与 event_recall@1 等指标评估关键帧检测器；在预训练策略上实现 ROC‑AUC 0.981‑0.997、PR‑AUC 0.924‑0.996；在记忆更新实验中，所有任务相关帧在 100% 情况下被写入并保留；与直接写入或无写入对比，机器人遮罩显著减少写入次数（如 OpenVLA‑OFT 从 336 次降至 169 次）并减少 FIFO 置换。

**⚠️ 局限性**

1) 对于阶段边界不明显或过短的任务，计划修订可能被稀释或分散，导致事件检测失效；2) 计划修订的惊讶信号可能来自模型误差而非语义事件，需谨慎解读；3) 需要为每个任务单独收集统计并设定阈值，迁移到新任务时难以直接使用；4) 当前实现存储完整图像，未探索更紧凑的记忆表示，可能限制记忆容量与推理效率。

---

## 118. A misinformation playbook sharpens judgement without breeding cynicism

**arXiv ID:** 2610.22724 | [PDF](https://arxiv.org/pdf/2610.22724v1)

**作者:** Tong Guo `[一作]` (Duke University), Qiaozhu Mei `[通讯]` (University of Michigan)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于 51 种语言 214,646 条事实核查记录的误信息说服手册，并在语言模型和人类受试者上进行前注册实验，以提升对新出现的误信息的辨识能力。

**💡 创新点**

创新点在于：①使用大规模多语言事实核查数据自动提取共通的说服策略及其共现模式；②将这一策略手册与“解耦指令”相结合，形成一种可预先提示的防御方法；③同时在 LLM 与人类受试者上验证其有效性，展示了跨媒体、跨语言的普适性。

**🔧 技术方法**

采用的技术包括：GPT‑4o 与 Gemini 2.5 Flash 进行策略编码与提示；主题建模、最大熵基准与点互信息网络分析；前注册实验设计与 AUC、辨识差距评估；对 AI 生成的误信息进行对抗性压力测试。

**📊 数据集**

使用的数据集是 Duke Reporters’ Lab 的 ClaimReview 数据（214,646 条多语言核查记录）以及基于该数据生成的 280 条 AI 误信息和 2,000 条对抗性生成误信息，用于语言模型与人类的辨识实验。

**📈 对比分析**

对比方法：对不同简报（无简报、准确提示、策略知识、解耦指令、完整简报）分别评估 LLM 与人类的辨识差距和 AUC。结果显示：完整简报下 LLM 的辨识差距提升 1.29 分（AUC ≈ 0.82）；人类的平均提升 0.23 分（对不熟悉的声明提升 0.42 分，AUC ≈ 0.76）。在对抗性测试中，策略知识会提升生成误信息的说服力，但完整简报可将此增益完全抵消。

**⚠️ 局限性**

局限性：①实验只验证了 AI 生成误信息的效果，未检验对人类作者误信息的适用性；②依赖自报的熟悉度，可能与真实知识水平不完全一致；③实验规模和简报完成率受限（人类策略简报完成率 85%）；④未评估长期记忆效果或在知识丰富受试者中的影响。

---

## 119. Enjoying the Product but Not Joining the Builders: Investigating Why Students Don't Contribute to OSS

**arXiv ID:** 2610.22664 | [PDF](https://arxiv.org/pdf/2610.22664v1)

**作者:** Italo Santos `[一作]` (University of Hawai'i at Mānoa), Anthony Peruma `[通讯]` (University of Hawai'i at Mānoa)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在一所高校的计算机科学系内，对122名学生进行问卷调查，探究其对开源软件（OSS）的熟悉度、使用情况、参与障碍与激励因素，并结合定量统计和主题分析进行结果呈现。

**💡 创新点**

通过聚焦单一教育环境，系统揭示课程结构、教师鼓励与时间竞争如何共同造成OSS使用与贡献之间的差距，为课程设计与机构干预提供可操作的实证依据。

**🔧 技术方法**

采用混合研究方法：问卷设计与Qualtrics分发、描述性与推论性统计分析、Borda计数法排序、以及三位研究者独立编码的质性主题分析。

**📊 数据集**

利用该高校计算机相关专业本科生及研究生自填的122份问卷作为数据集。

**📈 对比分析**

本研究未进行算法或工具的性能比较，主要以描述性统计与主题分析呈现结果；因此不存在传统意义上的“性能”指标。

**⚠️ 局限性**

样本来自单一机构、采用便利抽样、数据为自我报告，存在样本代表性不足、回忆偏差及社会期望偏差，结果仅适用于局部情境，缺乏广泛外推性。

---

## 120. CoEvolve: Self-Evolving Code Policies from Skills to Strategies for Humanoid Robots

**arXiv ID:** 2610.23288 | [PDF](https://arxiv.org/pdf/2610.23288v1)

**作者:** Zicheng Liu `[一作]` (Peking University), Hao Tang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CoEvolve框架，基于物理验证与结构化修订实现固定运动技能下的可执行多机器人协同；

**💡 创新点**

创新点在于将团队协作抽象为可执行代码，结合物理验证迭代修订并利用可验证经验，支持在共享物理约束下的长期协同；

**🔧 技术方法**

采用多模态基础模型（如GPT‑5.4‑mini、Gemini‑3.5‑Flash、Claude‑Sonnet‑4.6）生成协作策略，使用Policy Coder映射为可执行机器人策略，Verifier对执行结果进行物理检测与回滚；

**📊 数据集**

在MuJoCo仿真中使用两台Unitree G1 humanoid机器人，携带0.5 kg刚体负载穿过受限通道的协同搬运任务；

**📈 对比分析**

与REMAC、COHERENT、CaP‑Agent0、ASPIRE等基线以及组件变体对比，CoEvolve在基础任务中达成60 %完整成功率，验证进度与接受率均显著高于基线；

**⚠️ 局限性**

主要限制包括高昂的时间成本、缺乏真实世界物理恢复、运动精度与接触稳定性不足、仅在仿真环境验证、以及对硬件感知与安全机制依赖不足。

---

## 121. FederatedBytes: Tokenizer-Free, Byte-Level Small Language Models for Efficient Multimodal Federated Learning

**arXiv ID:** 2610.22906 | [PDF](https://arxiv.org/pdf/2610.22906v1)

**作者:** Ali Sarabadani `[一作]` (University of Technology Sydney), Justin Lipman `[通讯]` (University of Technology Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 FederatedBytes 的字节级联邦学习模型，能够在不使用词表的情况下处理多语言文本和多模态数据，并在边缘设备上实现高效训练。

**💡 创新点**

创新点包括：1）字节级无词表输入，天然支持多语言和多模态；2）采用 Performer 线性注意力，降低 O(L²) 复杂度到 O(Lm)；3）整体架构轻量级，专为联邦学习和资源受限设备设计；4）在联邦环境下显著降低通信与计算开销。

**🔧 技术方法**

技术手段包括：字节级输入编码 + 1D 卷积层的分层 Patch 编码；Performer 线性注意力（FAVOR+）；FedAvg 聚合；使用安全聚合、差分隐私与 TLS 保障隐私与安全；多模态处理将图像与文本统一为字节流。

**📊 数据集**

使用的公开数据集为：Amazon 多语言评论（英语、德语、中文）用于情感二分类；ScienceQA 多模态问答（文本+图像）用于多类别选择题。

**📈 对比分析**

与 GPT‑2（token‑level）、ByT5（byte‑level）以及 2 层 LSTM 等基线在同一任务上进行比较。FederatedBytes‑Performer‑30M 在联邦学习中以 100 个客户端时保持 65–73% AUC，同时比 ByT5 减少约 10 倍训练时间、10 倍 GPU/CPU 内存和 50‑70% 网络流量；在单机训练中，FederatedBytes‑Performer‑100M 亦可逼近 GPT‑2 的 80%+ AUC，显著低于 300M 的 ByT5。

**⚠️ 局限性**

局限性包括：1）在极小本地数据量下大模型易过拟合，导致联邦聚合性能下降；2）缺乏针对恶意客户端的鲁棒聚合方案；3）实验仅在模拟联邦环境中进行，未验证真实分布式网络延迟和带宽对性能的影响；4）目前未探索个性化 fine‑tuning 机制。

---

## 122. Benchmarking and Enhancing Agentic Causal Discovery

**arXiv ID:** 2610.22846 | [PDF](https://arxiv.org/pdf/2610.22846v1)

**作者:** Pingchuan Ma `[一作]` (Zhejiang University of Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个匿名化、面向证据驱动决策的因果发现基准，并在此基准上构建了 LLM 代理工具箱，记录每一次决策的细粒度证据与图的演化。

**💡 创新点**

创新点在于：①通过去除变量名称消除泄露；②提供“选择头room”和“修复头room”两种评估维度，区分算法选择与运行内决策的贡献；③对 PC 算法的错误删边进行日志记录与分解，使得能追踪代理恢复哪些边；④结合 LLM 代理在多种机制族上的表现，展示 LLM 能利用数据证据实现超越最佳算法的提升。

**🔧 技术方法**

技术手段包括：基于 PC、GES、FCI 等经典算法的工具箱；提供三层证据：图层（多算法一次性输出）、边层（CI 检验与重采样稳定性）、机制层（代码执行、方向探测）；LLM 代理（GPT‑6.1 Sol、Claude Opus 5.5、Qwen 3.8 Flash）通过交互式调用工具，执行自适应决策；评估指标为邻接 F1 与归一化 SHD，结合头room 计算残差收益。

**📊 数据集**

数据集为六类模拟结构因果模型（从线性高斯到重尾、非线性、输入依赖噪声），每类 10 维、600 样本，100 个开发实例和 32 个测试实例，变量匿名且随机排列。

**📈 对比分析**

与传统方法比较：对每类最佳选择算法、理想 oracle、非 LLM 路由器及两大基础模型进行对标。LLM 代理在所有六类数据上均超过最佳算法，其中 GPT‑6.1 Sol 在最难类上提升 14% 邻接 F1；Claude Opus 5.5 也实现正向残差收益；Qwen 3.8 Flash 在线性高斯类达标。追踪分析表明其恢复的边多数未出现于任何单一算法输出，证明其利用了数据内部证据。

**⚠️ 局限性**

局限性：仅使用合成数据，缺乏真实世界的因果真值；实验规模受限于 d=10、n=600；评估基准和头room 定义依赖 PC 的错误删边，可能不适用于其他算法；仅测试三款 LLM，结果可能随模型版本、工具箱配置和预算调整而变化。

---

## 123. Local proxies for memory-efficient training

**arXiv ID:** 2610.23145 | [PDF](https://arxiv.org/pdf/2610.23145v1)

**作者:** Wu S. Zihan `[一作]` (EPFL), Guillaume Bellec `[通讯]` (TU Wien)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了如何在变换器中实现准确且内存高效的局部学习，通过训练局部代理网络来预测全局网络输出，从而避免标准反向传播中的内存瓶颈。

**💡 创新点**

提出了一种新的局部学习算法ProxyProp，能够在不牺牲准确性的情况下显著降低内存需求，并且在深度线性网络中理论上是精确的。

**🔧 技术方法**

使用了局部代理网络和蒸馏技术，通过局部反向传播来计算梯度。

**📊 数据集**

在CIFAR-100数据集上评估了视觉变换器的性能。

**📈 对比分析**

与其他局部学习算法相比，ProxyProp在内存和准确性之间提供了更好的权衡，训练速度更快，性能优于内存高效的零阶学习算法。

**⚠️ 局限性**

局部学习算法在深度网络中可能面临的挑战，以及在非线性残差模型中可能出现的梯度估计误差。

---

## 124. PairSight: Characterizing and Predicting Operator-Level Interference in GPU Sharing for DNN Inference

**arXiv ID:** 2610.23086 | [PDF](https://arxiv.org/pdf/2610.23086v1)

**作者:** Xiao Liu `[一作]` (Sun Yat-sen University), Dan Huang `[通讯]` (Sun Yat-sen University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并解决在 GPU 共享（SM 分区）环境下，不同 DNN 任务的算子级干扰问题，提出了基于硬件计数器的干扰预测框架 PairSight。

**💡 创新点**

创新点在于：① 从算子级别对干扰进行细粒度建模，揭示干扰主要来源于共享 L2/DRAM 路径；② 采用交互式的硬件感知特征（单算子执行时长、DRAM 流量、内存相关停滞计数等）并使用 XGBoost 梯度提升树捕捉算子对之间的非线性交互；③ 通过窗口分块与组合实现对超长算子序列的预测，支持快速适配新模型与批量大小。

**🔧 技术方法**

技术手段包括 Nsight Compute 采集硬件计数器、构造算子级特征向量、使用 XGBoost 回归树做算子对干扰预测、窗口分块与组合推理，以及少样本在线微调。

**📊 数据集**

实验数据集涵盖 ResNet‑50、ResNet‑101、MobileNetV2、BERT‑base、Llama‑3.2‑1B 与 3B 等 6 个常见 DNN（包括 LLM），在 NVIDIA A100 GPU 上共收集 730k 条共执行样本。

**📈 对比分析**

与传统基于计算‑内存强度规则、互补表、线性模型、Abacus 以及 MLP 预测器对比，PairSight 在 HP 侧的 MAPE 仅为 4.3%（LP 侧 2.4%），显著低于强度规则的 19% 误差；查询时间仅约 5 µs，且在未知模型/批量大小和更长算子序列时仍保持 5% 左右误差。

**⚠️ 局限性**

局限性包括：① 仅针对 SM 分区的共享场景，无法直接迁移到 MIG 或 NVLink 等隔离机制；② 需要先对每个模型做一次完整的算子级计数器采样；③ 预测精度受算子序列相似度影响，若算子排列或同步方式发生剧烈变化可能导致误差增大。

---

## 125. Rational decomposition of polyhedra and a conjecture of Brlek and Reutenauer

**arXiv ID:** 2610.22663 | [PDF](https://arxiv.org/pdf/2610.22663v1)

**作者:** Flavio D'Alessandro `[一作]` (Sapienza University of Rome), Paolo Papi `[通讯]` (Sapienza University of Rome)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了 Brlek 与 Reutenauer 的猜想：对任意无直线的有理多面体，Brlek-Reutenauer 通过无歧义有理表达式得到的函数与 Khovanskii-Pukhlikov–Lawrence 的生成函数相同，从而定义的两种 valuation 互相一致。

**💡 创新点**

创新点在于给出了一个有效的、可计算的分解方法：将多面体整数点集拆分为若干简单集合（整数顶点加线性独立的周期），所有周期都位于收敛锥内；该分解在递归深度不超过 2d 的基础上实现，并直接得到无歧义的有理表达式，进而证明两种 valuation 的等价。

**🔧 技术方法**

主要技术包括：半线性集与 Presburger 语义的等价；无歧义有理表达式与有理函数的对应；Khovanskii‑Pukhlikov 与 Lawrence 的多面体生成函数理论；Brion 定理的投影/切割技术；以及对收敛锥的几何分区和递归分解。

**📊 数据集**

本工作不使用实验数据集，属于纯理论研究；所有结论均通过严格的数学证明给出。

**📈 对比分析**

由于是理论证明，没有数值实验可比；该方法在理论上给出了最优的分解深度与可计算性，显著提升了对多面体整数点集的无歧义表达式构造的可行性。

**⚠️ 局限性**

限制在于仅讨论了无直线的有理多面体；对含直线的情况，只能通过先减去直线部分后使用相同方法；另外，论文中未扩展到带一般系数的有理级数（第二个未解决的猜想）。

---

## 126. D-Composer: Language Modeling for Simultaneous Drum Transcription and Sound Event Decomposition

**arXiv ID:** 2610.22535 | [PDF](https://arxiv.org/pdf/2610.22535v1)

**作者:** Patrick O'Reilly `[一作]`, Bryan Pardo `[通讯]` (Northwestern University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 D-Composer 语言模型，将鼓录音拆解为标记的音频事件序列，并实现鼓谱、单音提取与源分离统一框架。

**💡 创新点**

将鼓解析与合成统一为单一交互式语言模型；设计极低比特率 DOC 编码器；利用大量合成数据训练实现跨音色泛化。

**🔧 技术方法**

自回归 Transformer 语言模型、MelodyFlow VAE+FSQ 编码的 DOC、音频‑MIDI 合成与数据增强、交叉熵训练。

**📊 数据集**

Lakh MIDI、Groove MIDI、Logic Pro、Native Instruments Battery4、Drum & Percussion Kits 合成集；评测集包括 E‑GMD、MDB‑Drums、Freesound Loop 等真实与合成录音。

**📈 对比分析**

与 IDM、ADTOF、DOSE 等基准在鼓谱 F1、单音多尺度谱损失（MSL）和重构 MSL 上比较，D‑Composer 在合成电子鼓集上 F1 最高，单音 MSL 低于 IDM，重构 MSL 亦优于 IDM；DOC 在比特率上比现有编码器低 10 倍，音质相当。

**⚠️ 局限性**

依赖合成数据，对非西方音色与复杂节奏泛化有限；DOC 音质低于高保真编码器；模型权重受版权限制未公开；训练能耗相对较高。

---

## 127. LangNCO: Compiling Natural-Language Constraints into Inference-Time Interventions for Neural Combinatorial Optimization

**arXiv ID:** 2610.23190 | [PDF](https://arxiv.org/pdf/2610.23190v1)

**作者:** Hironori Ohigashi `[一作]` `[通讯]` (Panasonic Connect Co., Ltd.), Hironori Ohigashi (Panasonic Connect Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 LangNCO 框架，能够在推理时将冻结的神经组合优化（NCO）求解器通过自然语言约束动态调整，而无需重新训练模型。

**💡 创新点**

其创新点在于：①将自然语言约束转译为结构化 DSL 规则并进行 schema 验证；②利用确定性规则执行器实现硬动作掩码；③引入 Critical-Offset Decoding，根据规则提取的关键节点在固定 rollout 预算内偏向性搜索，从而降低不可行性。

**🔧 技术方法**

使用技术包括：大型语言模型（Gemma‑4 或 GPT‑5）进行语义前端解析；DSL 规则编译成 PyTorch 掩码函数；基于 POMO 的多起点、几何增强解码；以及基于关键节点的偏置采样策略。

**📊 数据集**

实验基于 300 个标准 CVRP 实例（LKH‑3 给出参考解），并在此基础上构造 12 种额外约束和 6 种两两约束组合，评估其在 CVRP 上的适用性。

**📈 对比分析**

与未加约束的 NCO、直接生成完整路经的 GPT‑5、以及 LLMCoSolver 进行对比；LangNCO 在四类主约束下实现 100% 可行率，平均参考缺口 1.63–3.26%；在两两组合约束下 97.33–100% 可行率；在 12 种额外约束下 98.67% 可行率，远超无干预的 NCO（≈15%）并显著优于直接 LLM 生成解。

**⚠️ 局限性**

局限性包括：需要 LLM 生成语义正确的 DSL 规则，若规则语义与约束不符会导致不可行；关键节点提取目前仅支持静态可解析节点，难以处理属性定义或复杂逻辑；对语言表述的鲁棒性（如同义词、噪声）尚未系统验证。

---

## 128. NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models

**arXiv ID:** 2610.23271 | [PDF](https://arxiv.org/pdf/2610.23271v1)

**作者:** Guodong Du `[一作]` (Hong Kong Polytechnic University), Wanyu Lin `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为 NestedVLA 的后期技能整合框架，能够在不保存完整检查点的前提下，将来自不同机器人基准的视觉‑语言‑动作 (VLA) 模型中独立学习到的技能进行压缩、生成和组合。

**💡 创新点**

创新点在于将传统的参数聚合（平均、签名冲突修正等）替换为参数生成：使用嵌入式超网络根据任务标识动态生成动作专家的增量参数；同时仅保留压缩后的视觉‑语言模块增量，显著降低存储需求并缓解跨任务干扰。

**🔧 技术方法**

技术手段包括：1) drop‑and‑rescale 的稀疏压缩；2) 以二维补丁形式对参数空间进行分块并生成对应的超网络输出；3) 超网络包含四条任务相关路径（补丁嵌入、LoRA、任务编码、Multi‑head 输出）和共享全连接主干；4) 多任务掩码 MSE 损失与任务权重调节；5) 在推理时仅查询超网络一次并与基底模型融合。

**📊 数据集**

实验使用了三个闭环操纵基准：MetaWorld、ManiSkill、CALVIN；基底模型为 π_0.5；每个基准分别得到一个任务向量（skill）。

**📈 对比分析**

与多种静态与动态模型合并基线（Averaging、TIES、WUDI、TSV‑Merging、TALL‑Mask、EMR‑Merging、Twin‑Merging、Free‑Merging）以及独立 fine‑tune（FlowSDE）比较，NestedVLA 在三大基准上的平均成功率提升了 41–54 pp，单任务 fine‑tune 近乎匹配（ManiSkill 0.9 pp 内），且仅使用 1.5× 的存储预算。

**⚠️ 局限性**

主要局限在于对 MetaWorld 的稀疏压缩预算不足：K=10% 的 top‑K 仅能达到约 48% 的理想性能，导致该基准仍有约 13.8 pp 的差距；未来工作需考虑自适应稀疏率和更高效的压缩方法。

---

## 129. Binding-Specific Relearning Savings for Test-Time Continual Learning

**arXiv ID:** 2610.22381 | [PDF](https://arxiv.org/pdf/2610.22381v1)

**作者:** Zhe Zhou `[一作]` (University of Washington), Zhe Zhou `[通讯]` (University of Washington)

**通讯引用:** 2020 | [OpenAlex ID](https://openalex.org/A5101404161)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了持续学习模型在失去事实记忆后，通过重新学习节省的优化成本。

**💡 创新点**

提出了重新学习节省（relearning savings）指标，证明已学习的事实比未学习的绑定更易恢复。

**🔧 技术方法**

使用参数更新的测试时持续学习（TTCL）框架，在 Qwen2.5-1.5B 模型上进行强化学习与梯度更新。

**📊 数据集**

利用 576 条人工合成事实（三元组），并对照未见事实池进行评估。

**📈 对比分析**

对比三种历史对照（已绑定、仅曝光、竞争绑定），在可观测记忆缺失点达到约 18% 的优化步骤节省，表现出显著的重新学习优势。

**⚠️ 局限性**

实验仅在单一模型、单一规模下进行，合成事实可能不代表真实知识，且样本数量有限，未定位存储机制。

---

## 130. Anisotropic Representation Dynamics in Transformers

**arXiv ID:** 2610.23242 | [PDF](https://arxiv.org/pdf/2610.23242v1)

**作者:** Nguyen Do `[一作]` (University of Florida), My T. Thai `[通讯]` (University of Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Transformer自注意力机制中表示差异的收缩与放大机制，提出混合门控理论解释并预测注意力路由导致的“切换”现象。

**💡 创新点**

将自注意力拆解为价值传输、查询门控与键门控三路机制，给出谱理论判断何时门控能逆转传输收缩，提出相对切换强度、绝对逆转严重度及有效切换维度等定量指标。

**🔧 技术方法**

利用Jacobian分解、矩阵谱分解、局部线性化、自动微分与数值差分等技术。

**📊 数据集**

在GPT-2 Medium、Gemma‑2‑2B、Llama‑3.2‑1B、Qwen2.5‑0.5B等预训练模型上，使用WikiText‑2验证集进行实验。

**📈 对比分析**

通过解析与有限前向实验对比，并在不同模型层级、架构下计算切换维度、相对/绝对切换强度，谱分析显著优于随机搜索；实验验证了理论预测。

**⚠️ 局限性**

仅在局部单层点评估，未考虑跨层动态；实验空间受限于低维投影与有限样本；对真实任务效应的直接影响尚未探究。

---

## 131. Explaining Chess and Go in Natural Language

**arXiv ID:** 2610.23166 | [PDF](https://arxiv.org/pdf/2610.23166v1)

**作者:** Ori Press `[一作]` (Nebius AI), Alexander Golubev `[通讯]` (Nebius AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种无需训练的自适应方法 BinProp，自动发现并生成基于棋类引擎评估的可执行二进制属性（棋、围棋、跳棋、将棋），并将其转换为自然语言，用来解释引擎为何偏好某一步。

**💡 创新点**

创新点在于：①使用对比正负棋局的二进制属性迭代挖掘概念；②仅利用语言模型生成并评估这些属性，无需额外训练；③支持跨多种棋类；④把生成的概念函数直接转为可解释的自然语言。

**🔧 技术方法**

技术包括：大规模语言模型（多款开源与专有模型）生成 Python 函数，Stockfish 18、KataGo 等引擎评估与连招采样，迭代筛选概念的差异 Δ(f) 与重叠度，最后将概念转为自然语言解释。

**📊 数据集**

使用的数据集：200k 个来自 Lichess 精英数据库的赢/输棋局（各 100k），100 起始棋局样本；围棋使用 JGDB 的 100k 赢/输棋局；跳棋和将棋使用公开比赛数据；所有棋局通过 Stockfish、KataGo 等引擎评估并采样连招。

**📈 对比分析**

比较方法：将 BinProp 与基线直接提示的语言模型以及概念引导评论（CCC）在相同支持集上对比；BinProp 在棋类的解释性得分从约 3% 提升至约 29%，在 Go、跳棋、将棋也实现类似提升；跨模型取最优解释的或模型间组合可将得分提升至约 66%。

**⚠️ 局限性**

局限性：①只挖掘二进制属性，可能忽略多值或多层次概念；②对局面变化敏感的概念被视为脆弱；③仅解释基于引擎评估的决策，对非引擎或策略型决策缺乏适用性；④最终自然语言解释仍需要手工转换或进一步加工。

---

## 132. Cryptographically Secure Agent-to-Agent Communication in Multi-Agent Systems

**arXiv ID:** 2610.23078 | [PDF](https://arxiv.org/pdf/2610.23078v1)

**作者:** Karol Budzyn `[一作]` (Orange Innovation Poland), Mateusz Szczepkowski `[通讯]` (Orange Innovation Poland)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出面向多代理系统的零信任通信架构，涵盖身份管理、传输安全、消息完整性与意图验证；

**💡 创新点**

将去中心化身份(DID/VC)、KEMTLS‑PDK量子抗握手、CBOR/COSE消息封装、零知识证明与动态信任阈值模型融合，实现端到端安全与持续授权；

**🔧 技术方法**

使用DID/VC、权限链码区块链、SPIFFE/SVID、KEMTLS‑PDK、混合后量子密钥交换、CBOR/COSE、zk‑SNARK/zk‑STARK、eBPF监控、AgentSight等技术；

**📊 数据集**

本文为理论综述，未使用具体数据集；

**📈 对比分析**

通过对比现有TLS、mTLS和KEMTLS‑PDK的握手次数、证书大小和传输延迟，指出KEMTLS‑PDK在握手成本和密钥封装大小上的优势；未给出实验量化结果；

**⚠️ 局限性**

ZKP生成开销大、消息封装随代理链增长、区块链最终性延迟导致撤销窗口、后量子算法成熟度不足以及系统可扩展性需进一步验证。

---

## 133. Agentic Image Generation in the Wild

**arXiv ID:** 2610.23269 | [PDF](https://arxiv.org/pdf/2610.23269v1)

**作者:** Yifan Li `[一作]` (Renmin University of China), Wayne Xin Zhao `[通讯]` (Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对实时场景下的图片生成任务，构建了新闻驱动的自动化流程，生成包含时间新颖性、语义拥挤和空间混淆等挑战的 500 条生成提示，形成“Wild”基准，并提出了 Locate‑Corroborate‑Isolate 三步推理范式，训练 8B 参数的 Ranger‑8B 代理模型，实现高质量的检索与生成。

**💡 创新点**

创新点包括：① 用新闻时间戳与三天时间窗口模拟实时搜索环境；② 通过实体提取与检索混淆筛选自动生成既具挑战性又可验证的提示；③ 设计 Locate‑Corroborate‑Isolate 体系，将文本检索、跨模态/跨图像验证和目标裁剪相结合，有效解决三大挑战；④ 采用教师轨迹训练方法，生成 15K 轨迹，显著提升检索与生成性能。

**🔧 技术方法**

主要技术手段包括：大规模 VLM（如 Qwen3‑VL‑8B、Gemini‑3.1‑pro）进行实体抽取、文本检索与图像验证；Web 爬虫与搜索 API（文本/图像）构建三天时间窗口的检索环境；基于检索混淆率 κ 的过滤与交叉验证算法；裁剪工具实现 Isolate 步骤；RL‑style 训练（8B 参数）结合 8 NVIDIA H20 GPU 的分布式训练；评估使用 VLM 进行搜索与生成得分。

**📊 数据集**

数据集：新闻驱动的 World Snapshot（2026 年 7 月）生成的 500 条提示，分为 16 个子类别，涵盖娱乐、体育、时尚、产品与文化；教师轨迹 15K 条来自 1‑6 月新闻的 8B 代理训练；对比实验还使用 Mind‑Bench 与 KnowGen 两个公开基准进行泛化评估。

**📈 对比分析**

在 Wild 基准上，传统手工检索、利用 GPT‑5.x / Claude‑Opus‑5 等前沿模型的 Harness 框架仅能达到约 51.5% 的检索得分；Ranger‑8B 在 67.2% 上领先 15.7 点，生成得分亦位居各生成器榜首；在 Mind‑Bench 与 KnowGen 上，Ranger‑8B 在知识、视觉正确性与逼真度上均超过同类方法，表明检索能力可迁移到更广泛任务。

**⚠️ 局限性**

局限性包括：① 训练与评估均基于 2026 年新闻，可能对其它领域或更广泛语义场景的泛化尚未验证；② 需要依赖 Web 搜索 API 与大量算力，部署成本较高；③ 尽管改进显著，但在极高混淆度或极其新颖的实体上仍有 30% 以上错误率；④ 目前模型规模仅 8B，进一步扩展或更高效的检索策略仍待探索。

---

## 134. Understanding Stability of Outer Nesterov Momentum on Stochastic Pseudo-Gradients

**arXiv ID:** 2610.22623 | [PDF](https://arxiv.org/pdf/2610.22623v1)

**作者:** Nishant Sharma `[一作]` (Chandar Research Lab), Hiroki Naganuma `[通讯]` (Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究分布式训练中外部 Nesterov 动量对伪梯度的均方稳定性，给出精确的稳定性判据并推导出对超参数的四个关键影响；

**💡 创新点**

提出了对随机 K 步内层 SGD 产生的伪梯度进行块算子分析，给出必要且充分的均方稳定性条件，并揭示了批量大小、局部步数和工作者数量对稳定性的交互作用；

**🔧 技术方法**

理论上使用块算子（block operator）和二阶矩映射（second‑moment map）分析，实验上使用共享最小批量扰动（shared‑minibatch perturbations）、Hessian‑向量乘积和扩散成本（spread cost）衡量稳定性；

**📊 数据集**

在 CIFAR‑10 上训练 MLP 和 ResNet‑20 进行实验，此外在 Pythia‑70m、160m、410m、1b 语言模型上验证；

**📈 对比分析**

通过与理论预测（放大率、扩散成本）对比，实验结果基本符合四条推论：增加动量可减小噪声预算、批量大小超过临界值可使多步稳定、增加工作者可降低方差但需求指数级增长、固定总批量拆分不降低噪声；性能表现为在稳定区间内训练损失保持低于检查点，超出则出现明显损失上升；

**⚠️ 局限性**

局限在于只关注均方稳定性而非收敛速度，对内层只考虑 SGD，未覆盖 AdamW/ Muon 等自适应优化器；此外理论在高维逼近下简化，真实网络中块算子依赖参数，需冻结线性化近似。

---

## 135. qGAMMA: A Framework for Optimal DNN Mapping on Multi-Precision Accelerators

**arXiv ID:** 2610.23222 | [PDF](https://arxiv.org/pdf/2610.23222v1)

**作者:** Leandro Fiorin `[一作]` (Politecnico di Milano), Cristina Silvano `[通讯]` (Politecnico di Milano)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套名为 qGAMMA 的框架，用于在支持多精度计算单元的空间加速器上，对混合精度深度神经网络（DNN）进行映射与优化。

**💡 创新点**

创新点在于将混合精度 DNN 与多精度加速器的映射问题统一建模，扩展了数据流模型与遗传编码，并通过启发式剪枝与早停策略显著缩小搜索空间；同时提供了基于 MAESTRO 的多精度性能估计工具。

**🔧 技术方法**

核心技术包括域特定遗传算法、三层级基因编码、三种启发式剪枝策略、早停机制、以及基于硬件综合结果的解析性能/能耗模型。

**📊 数据集**

实验数据集主要是三种 CNN 结构：MobileNetV1、ResNetV1 以及混合精度的 ResNet18（INT8/INT4），在 22 nm 22 nm 技术模型下进行评估。

**📈 对比分析**

通过与原 GAMMA、不同统一精度设置以及混合/统一精度比较，qGAMMA 在最低能耗、最低延迟和能耗延迟乘积（EDP）上分别实现了 1.20× 的速度提升、17% 的能耗下降以及 FP8/FP16 与 FP32 之间 4–21 倍的 EDP 改进。

**⚠️ 局限性**

主要限制在于搜索空间仍然极大，需使用大量种群和代数；启发式和早停可能导致收敛到次优解；目前仅验证于 CNN 任务，对其他类型 DNN 或更大规模网络的适用性尚待进一步验证。

---

## 136. Flash RNN-T: Memory-Efficient Joint Network and Transducer Loss Computation

**arXiv ID:** 2610.23409 | [PDF](https://arxiv.org/pdf/2610.23409v1)

**作者:** Mahmoud Ashraf `[一作]` `[通讯]` (Tarteel AI), Mahmoud Ashraf (Tarteel AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

实现了一种名为Flash RNN‑T的高效联合网络与损失计算实现，支持无记忆（untiled）与分块重计算（tiled）两种模式，显著降低内存占用并提升吞吐量。

**💡 创新点**

创新点在于将联合网络计算拆分为小块（tile）并采用激活检查点重计算，同时配合专门化的 GPU 核函数，既保留完整的对齐空间与词表，又在不同工作负载下实现吞吐‑内存前沿。

**🔧 技术方法**

采用激活检查点、专门化的前向/后向核函数、张量打包、词表投影优化、CUDA 核融合以及 FP16/FP32 混合精度等技术。

**📊 数据集**

使用英文 Granary 子集（短、短中工作负载）和 TED‑LIUM3 长工作负载的音频与字幕，通过 2D 分桶采样得到三组批次进行评测。

**📈 对比分析**

与 Apex、NeMo Numba、Optimized Transducer 等现有实现对比，Flash RNN‑T 在 H100 与 RTX 5090 GPU 上在中、长负载下形成吞吐‑内存曲线全前沿：未分块版本吞吐提升 24–39%，内存降低 49–56%；分块版本内存进一步降低 4–9 倍，仍保持接近最快吞吐。

**⚠️ 局限性**

局限在于仅评估联合网络与损失计算的单阶段吞吐，未覆盖完整端到端训练；对极长序列仍受限于词表投影与网格存储；以及不同 GPU 上 GPU 特定的吞吐变动导致性能差异。

---

## 137. Gaussian Identification in MIMO Channels

**arXiv ID:** 2610.22931 | [PDF](https://arxiv.org/pdf/2610.22931v1)

**作者:** Mohammad Javad Salariseddigh `[一作]` (Technical University of Darmstadt), Christian Deppe `[通讯]` (Technical University of Braunschweig)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在每天线、每符号峰值幅度约束下的离散时间高斯多输入多输出（MIMO）信道的识别能力，建立了识别能力的上下界，这些界限与信道秩r线性相关。

**💡 创新点**

提出了在MIMO信道中，识别能力的增长与信道秩成线性关系的创新观点，特别是在超指数代码本增长的情况下。

**🔧 技术方法**

使用了高斯信道模型和确定性编码技术，结合了矩阵的Frobenius范数和欧几里得空间的等距性。

**📊 数据集**

使用了静态MIMO高斯信道的白噪声模型，假设发射器和接收器都具有完全的信道状态信息（CSI）。

**📈 对比分析**

通过与随机编码（RI）和确定性编码（DI）的方法进行比较，发现DI在MIMO信道中的识别能力上限为r/2，下限为r/4，表明信道秩对识别性能有显著影响。

**⚠️ 局限性**

限制在于尚未完全解决在MIMO高斯信道中确定性编码的识别能力的特征，特别是在存在符号间干扰（ISI）和有色噪声的情况下。

---

## 138. LED: Improving Masked Diffusion Language Models by Lookahead Editing

**arXiv ID:** 2610.23122 | [PDF](https://arxiv.org/pdf/2610.23122v1)

**作者:** Yuyan Zhou `[一作]` (Hong Kong University of Science and Technology), James Kwok `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Lookahead Editing框架，结合可写的提议状态与掩码扩散模型，在解码前多步迭代改进预测，从而提升掩码扩散模型的可修订能力并保持其优化简洁性。

**💡 创新点**

创新点在于将可写提议状态注入掩码扩散解码流程，既保留了掩码扩散的简易转移调度，又实现了类似统一扩散模型的多次状态重写，弥补了两者的短板。

**🔧 技术方法**

采用了掩码扩散模型（MDM）基础架构，加入预训练MDM生成提议、三分支Transformer（clean、masked、proposal）以及自定义注意力掩码；训练时使用交叉熵损失，推理时实现K步Lookahead迭代和阈值采样。

**📊 数据集**

使用NVIDIA OpenMathReasoning、OpenScience、OpenCodeReasoning2、OpenThoughts3、Mixture-of-Thoughts等开源后训练数据集进行SFT；评测覆盖数学（MATH、AIME 2024/25）、科学（GPQA）和代码（HumanEval、HumanEval+、MBPP、MBPP+）八个基准。

**📈 对比分析**

与Fast‑dLLM、XDLM、SDAR、LLaDA 2.1、Nemotron等MDM/UDM基线对比，在SFT-only设置下，7B模型平均提升11.4点，8B模型提升7.7点；在AIME 2025与HumanEval+上，K=2时比LLaDA 2.1‑16B快1.5‑1.8×且精度更高，Pareto前沿更优。

**⚠️ 局限性**

局限性包括对提议阈值和Lookahead步数的敏感性，过长或过保守的阈值可能导致误差积累；目前仅在SFT后训练数据上验证，未探索在更大规模或多模态任务中的适用性；实现仍受GPU批量和块大小限制。

---

## 139. Between Spoilers and Guesswork: Avoiding Over-Specification in Coding Benchmarks

**arXiv ID:** 2610.22760 | [PDF](https://arxiv.org/pdf/2610.22760v1)

**作者:** Yuanzhe Liu `[一作]` (Rensselaer Polytechnic Institute), Martin Hirzel `[通讯]` (IBM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并评估代码仓库级别基准中如何通过识别器提示（identifier hints）减少任务过度说明与猜测，提出新的选择与投影策略（DefUse 与 Declaration），并公开相应提示生成工具。

**💡 创新点**

提出了基于“定义与使用”选择（DefUse）与仅暴露接口结构（Declaration）投影的两维设计空间，解决了过去提示方案在披露实现细节与隐藏接口信息之间的权衡问题。

**🔧 技术方法**

使用大型语言模型（Claude Haiku 4.5 生成提示，GPT‑5.6 Luna 评估提示内容）以及自研的提示生成管道；实验中部署 DeepSeek V4 Flash、MiniMax M2.5 与 GPT‑5.6 Luna 进行代码生成与评估。

**📊 数据集**

在五个主流仓库级别编码基准上进行实验：Python 版 Doc2Feat-bench、FEA-Bench、SWE‑bench Pro、SWE‑rebench 以及 Java 版 Multi‑SWE‑bench。

**📈 对比分析**

通过对选择策略（DefUse vs Doc2Feat/FEA/Pro）、投影策略（Declaration vs Name‑Only/Pro）以及属性消融进行对照实验，发现 DefUse+Declaration 在保证接口信息完整的同时显著减少了泄露实现细节，整体解决率提升约 10–20%（以 GPT‑5.6 Luna 为例），并验证了结构属性对解题更有利，而语义描述虽提升性能但伴随更高的泄漏风险。

**⚠️ 局限性**

实验受限于每个实例仅跑一条轨迹、基准子集规模有限、模型随机性与 LLM 评估主观性，导致结果对模型/基准的泛化性与统计稳健性受限。

---

## 140. MultiAgent-OPD: End-to-End Training of Hierarchical Multi-Agent Systems with On-Policy Distillation

**arXiv ID:** 2610.23234 | [PDF](https://arxiv.org/pdf/2610.23234v1)

**作者:** Yuhan Sun `[一作]`, Zhongan Bi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两阶段的端到端层级多智能体训练框架 MultiAgent-OPD，利用任务训练的教师模型进行在线策略蒸馏，为子智能体提供 token 级本地监督，同时用最终答案奖励对领导者与回答者进行强化学习。

**💡 创新点**

创新点在于将教师模型在简化工作流中学习到的检索与总结策略迁移为在层级系统中对子智能体的即时监督；该方法通过对子任务历史的 token 级对比来给出局部优势，解决了传统全局奖励难以为中间行动赋权的问题。

**🔧 技术方法**

使用的技术包括：基于 Qwen3‑4B‑Instruct‑2507 的共享参数语言模型、分层角色（Leader、Searcher、Answerer）设计、两阶段训练（Stage‑I 学习搜索+总结，Stage‑II 进行 OPD 与 RL 结合）、逆 KL 蒸馏、组相对优势（GRPO）和 PPO 更新。

**📊 数据集**

训练集：NQ 与 HotpotQA；测试集：七个公开问答基准（包括 NQ、HotpotQA、TriviaQA、2WikiMultiHopQA、Bamboogle 等）。

**📈 对比分析**

与基线相比：在 macro avg@4 上达到 38.68%，比 Prompted Critic 提升 4.06pp、比 C‑MAS+RL 提升 2.61pp，且在 macro pass@4 上也超过对手；在与其他 RL 基础检索代理（Tree‑GRPO 等）的对比中，亦保持领先。

**⚠️ 局限性**

局限性包括：仅在 Qwen3‑4B‑Instruct‑2507 上实验；使用统一检索与 token 上限，未做匹配预算对比；教师模型可能出现偏差，且蒸馏可能抑制探索；共享参数导致不同角色更新相互耦合，未单独评估领导者与回答者目标。

---

## 141. TALLOP: Joint Optimization of Local Training and Communication Compression for Time-Constrained Federated Learning

**arXiv ID:** 2610.22996 | [PDF](https://arxiv.org/pdf/2610.22996v1)

**作者:** Honoka Anada `[一作]` (University of Tokyo), Shinya Takamaeda-Yamazaki `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了TALLOP框架，在联邦学习中针对固定的每轮时间预算，联合优化本地训练工作量与通信压缩比例，以减少达到目标准确率所需的壁钟时间。

**💡 创新点**

创新点在于引入基于损失曲线平滑与外推（LENS）以及压缩误差模型（CLEN）的预测机制，能够在单个训练步骤内估算不同时间分配下的后压缩模型损失，并通过三分搜索寻找最优时间分配因子，首次实现了在通信受限、设备异构环境下的训练与压缩协同优化。

**🔧 技术方法**

使用了损失曲线平滑外推、压缩误差比例估计、三分搜索、无序剪枝压缩、时间分配参数化、非线性最小二乘拟合等技术，并在模拟的异构FL环境中进行实验。

**📊 数据集**

在CIFAR-10和CIFAR-100数据集上，使用MobileNetV2和Vision Transformer（ViT）模型进行实验。

**📈 对比分析**

通过与FFL、TimelyFL、FedBalancer以及若干消融版本（如无截止、仅训练优化等）进行对比，测量最终准确率和达到目标准确率所需的壁钟时间（TTA），结果显示TALLOP在大多数配置下取得了更短的TTA和与基线相当甚至更高的最终准确率。

**⚠️ 局限性**

局限性包括：实验基于模拟的计算与通信性能模型，缺乏真实部署验证；仅使用单一随机种子，未给出误差范围；在计算为主的配置下增益有限；假设固定的每轮时间预算、损失曲线单调、预测损失单峰等理想条件。

---

## 142. Seeing Seizures: A Large-Scale Empirical Study of Video and EEG Deep Learning for Automated Rodent Seizure Detection and Severity Grading

**arXiv ID:** 2610.22566 | [PDF](https://arxiv.org/pdf/2610.22566v1)

**作者:** Alloy Das `[一作]`, Soumik Sarkar `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

对20只大鼠的同步视频和单通道EEG记录进行24131条、601次会话的注释，评估多种视频和EEG模型在癫痫发作检测与行为严重度分级任务中的表现。

**💡 创新点**

提供了一套大规模同步视频-EEG数据集，并系统比较现有视频和EEG深度学习模型在多粒度严重度评估任务中的效果，揭示视频模型在严重度分级方面与EEG模型的差距以及对未知动物的泛化瓶颈。

**🔧 技术方法**

使用了十种主流视频动作识别架构（如R(2+1)D‑18、X3D‑M等）和七种单通道EEG分类模型（如GRU等），基于帧采样、多帧输入、宏F1评价指标以及跨会话/跨个体验证进行实验。

**📊 数据集**

由20只大鼠在硫醚诱导癫痫模型下，采集的601个会话同步视频/EEG片段，包含2‑5级行为严重度标签，总计24131条片段。

**📈 对比分析**

采用会话独占和个体独占两种交叉验证，比较视频与EEG模型在二分类检测（宏F1最高0.974）和五分类严重度分级（最高0.657）上的表现；跨个体时性能下降至0.475/0.404，视频仅检测在RodEpil数据集上也可达0.979宏F1。

**⚠️ 局限性**

局限在于严重度分级对未知动物的泛化差，数据集规模虽大但仍集中于单一化学模型，缺乏多模态或多个体预判能力，且模型未解决编码捷径问题。

---

## 143. Frame-Consistent Feed-Forward 3D Gaussian Splatting from Sparse Unposed Images

**arXiv ID:** 2610.23039 | [PDF](https://arxiv.org/pdf/2610.23039v1)

**作者:** Viruthshaan Vinasirajan `[一作]` (University of Moratuwa), Ranga Rodrigo `[通讯]` (University of Moratuwa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

将冻结的VGGT姿态估计器与冻结的TokenGS Gaussian重建器通过确定性操作连接，实现无训练、无精细化的完整视图合成流程。

**💡 创新点**

创新点在于揭示并修复了两大集成失效：坐标系不一致与尺度偏差，并通过简单的坐标转换、帧一致推理和全局尺度校正，使得无训练模型可逼近COLMAP基准。

**🔧 技术方法**

核心技术包括VGGT姿态估计、TokenGS Gaussian重建、坐标系转换（W2C↔C2W）、帧一致前向推理、全局尺度因子校正以及基于Procrustes的姿态误差分析。

**📊 数据集**

在DL3DV和MipNeRF360两大数据集上进行评估，使用四视图标准协议。

**📈 对比分析**

与COLMAP基准相比，所有视图和稀疏两种设置均可达22.47 PSNR（与COLMAP 22.98仅差0.51 dB），运行时间维持在每场景约0.7 s，显著快于传统SfM方法。

**⚠️ 局限性**

局限性包括需在同一前向推理中包含目标视角、尺度校正系数需针对数据集手工设定、未在其他重建骨干上验证其通用性。

---

## 144. FuseRAG: Fusing Structure and Evidence into Inference for Cost-Efficient Graph-based RAG

**arXiv ID:** 2610.22818 | [PDF](https://arxiv.org/pdf/2610.22818v1)

**作者:** Tingqi Wang `[一作]` (Georgia State University), Zhipeng Cai `[通讯]` (Georgia State University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FuseRAG框架，结合图结构与证据的潜在表示与文本锚点，提升多跳问答效果与推理速度。

**💡 创新点**

创新点在于将检索到的图证据从文本序列化转为可重用的潜在记忆，并通过双路径融合图结构与查询相关性。

**🔧 技术方法**

采用图神经网络（GAT）消息传递、跨注意力、低秩投影、0-1背包优化等技术，将潜在记忆注入LLM自注意力。

**📊 数据集**

在MuSiQue、HotpotQA和2WikiMultiHopQA三个多跳问答数据集上进行评测。

**📈 对比分析**

与多种基线（VanillaRAG、KGP、ToG、RAPTOR等）对比，FuseRAG在Llama-3‑8B和Qwen3.5‑9B上实现了最高F1、准确率，并在TTFT上相对最快的图基RAG方法提升至约2–3倍速度。

**⚠️ 局限性**

限制在于对显式文本锚点的依赖仍存在，且潜在记忆的存储开销与训练成本较高；在极大图规模下，图推理开销仍不可忽视。

---

## 145. One Communicator, Three Vendors: Extending the MoA Predictive Theory to a Real NVIDIA+AMD+CPU MPI Ensemble

**arXiv ID:** 2610.22769 | [PDF](https://arxiv.org/pdf/2610.22769v1)

**作者:** Lenore Mullin `[一作]`, Gaetan Hains `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了在Slurm异构作业中，单一MPI通信器跨越NVIDIA A100、AMD MI100和CPU节点的多厂商加速器集群，对注意力计算进行分区与集成。

**💡 创新点**

创新在于首次将不同厂商加速器（A100、MI100）与CPU放入同一MPI通信器，提供理论权重分配公式和形状推导的实证验证，并提出新的硬件形状模型与可测量的ε误差框架。

**🔧 技术方法**

采用MPI、Slurm异构作业、OpenMPI、Triton、OpenMP目标offload、MoA（数组理论）以及性能测量工具，结合多节点多GPU/CPU的实验平台。

**📊 数据集**

使用标准的Transformer注意力计算任务（N=8192，D=64，B·H=2）作为工作负载，并在NCSA Delta、TACC Stampede3等多站点集群上测量。

**📈 对比分析**

通过行分区与键值分区两种实现，比较各设备的计算时间、通信成本和能耗；结果显示A100性能稳定，MI100在默认配置下存在约600 ms的固定开销，权重分配公式对MI100不适用，通信在键值分区时占主导。

**⚠️ 局限性**

局限性包括MI100的固定成本与可变性难以建模、跨分区通信表现高度依赖网络调度和UCX导致作业被取消、以及不同实验间通信时延的高方差。

---

## 146. No One-Size-Fits-All: Adaptive Code Editing with Feature-Based Strategy Selection

**arXiv ID:** 2610.22881 | [PDF](https://arxiv.org/pdf/2610.22881v1)

**作者:** Jun Wan `[一作]` (Zhejiang University), Xiaoxue Ren `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构造500个真实的代码编辑任务，系统对比了一步式、搜索-替换和计划-应用三种 LLM 代码编辑策略，并基于四个核心任务特征提出了自适应策略选择框架，利用 fine‑tuned 小型 LLM 实时决定最优编辑策略并生成可解释理由。

**💡 创新点**

①对代码编辑策略进行首次细粒度经验研究，发现无一策略在所有任务中始终优越；②提出了基于任务特征的自适应策略选择模型，既提高了编辑成功率，又显著降低了 token 消耗；③在选择过程中加入可解释性理由，提升了决策透明度。

**🔧 技术方法**

利用 LLMs（Claude‑Opus‑4.1、GPT‑4o、Gemini‑2.5‑Pro、Qwen3‑235B‑A22B、DeepSeek‑R1、Llama‑3.1‑405B 等）实现代码编辑；采用 Llama‑3.1‑8B 通过 LoRA 进行微调，得到策略选择模型；使用 LLM 进行特征提取、策略推荐与理由生成；对比一键式、搜索-替换、计划-应用三种策略实现。

**📊 数据集**

从 SWE‑bench 和 SWE‑bench‑Verified 构造的 500 个单文件任务做实验研究，构造 650 个单/多文件任务做性能评估；任务涵盖 BugFix、FeatureAdd、Refactor 等多种类型，包含不同代码规模、修改范围和修改类型。

**📈 对比分析**

与固定策略基线（CodePlan、AutoCodeRover、FuseApply）、指令驱动的 LLM 选择器、CodeBERT 分类器以及多种消融版本进行对比；使用 Pass@1、输出 token 计数、Select@1 和理由准确率（Cohen’s Kappa）等指标评估。结果显示：在 650 任务上，优选框架相较最佳固定策略 AutoCodeRover 提升 Pass@1 21.7%，减少 token 27.3%；策略选择准确率 72.6%，理由准确率 69.9%（Kappa=0.807）。

**⚠️ 局限性**

①实验仅涉及 Python 代码，跨语言泛化性待验证；②任务仅限单文件，未充分检验仓库级上下文影响；③token 评估仅计输出 token，未包含输入与中间推理 token；④策略选择模型仍依赖 LLM，若出现幻觉可能影响结果，尽管通过 Pass@1 指标间接控制。

---

## 147. Whose Polarization Is It? Auditing Fairness in Network Polarization Measures

**arXiv ID:** 2610.22677 | [PDF](https://arxiv.org/pdf/2610.22677v1)

**作者:** Stanisław Stępień `[一作]` (Wroclaw University of Science and Technology), Radosław Michalski `[通讯]` (Wroclaw University of Science and Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对七种极化测度进行公平性审计，提出测量不变性（MI）和结构外生性（SE）等判定准则，并用三轨道实验（参数化合成、诊断拓扑、实测网络）验证其在少数群体中的表现。

**💡 创新点**

首次把公平性概念迁移到极化测量领域，设计了基于测量不变性和结构外生性的完整审计框架，并提出网络加权距离（NWD）以覆盖此前未被测度覆盖的极端性-分离组合；揭示了意见形成通道与结构通道两条独立的偏差机制。

**🔧 技术方法**

使用随机块模型（SBM）、Friedkin–Johnsen 社会影响动力学、Kolmogorov–Smirnov、半偏R²、Benjamini–Yekutieli 多重检验校正、随机游走概率、以及各种聚类、能量、模数等极化指标；结合结构化的实验轨道与诊断图谱进行系统评估。

**📊 数据集**

合成网络（ER、BA、SBM）、美国国会投票与DIME 经济制裁分数、欧洲议会投票与 CHES 议员左右分数、以及 NetSense 学生通信网络与父母收入数据。

**📈 对比分析**

通过对每个测度在 MI、SE（FJ 通道）和 RS 的三重打分，结果显示没有任何测度在所有轨道和所有尺度下通过；大多数测度对少数群体的极化评价偏高，结构外生性主要受意见形成通道影响；在真实网络中，部分测度与合成结果一致，验证了审计的外推性。

**⚠️ 局限性**

局限性包括：仅评估单一子图，未检验边界连线对测度的直接影响；依赖 Friedkin–Johnsen 动力学，可能不适用于其他意见形成模型；公平性准则阈值人为设定；未探讨多维测度组合对公平性的综合影响；并且在极少数群体规模下多数测度因分辨率限制而不可解释。

---

## 148. Star Light, Star Bright: Can Multimodal Models Parse an Astronomy Visual Question Answering Dataset Right?

**arXiv ID:** 2610.22474 | [PDF](https://arxiv.org/pdf/2610.22474v1)

**作者:** J. P. Naiman `[一作]` `[通讯]` (University of Illinois), J. P. Naiman (University of Illinois)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个面向天文天空图像的视觉问答（VQA）数据集，并用三款低成本多模态模型在该数据集上进行测试。

**💡 创新点**

创新点在于：①首次针对天文学图像（真实天区图、等高线图以及使用高斯混合模型生成的合成天区图）创建专门的VQA数据集；②同时提供原始图像与模拟历史衰化后的图像，模拟真实档案扫描环境；③通过对比真实与合成图像，评估模型对图像源的辨识能力。

**🔧 技术方法**

技术实现包括：利用NASA SkyView服务提取真实天空切片；使用WCS和高斯混合模型生成合成图像；采用多级复杂度问题生成模板；应用页面衰化算法模拟旧档案损伤；并使用Claude、Gemini、LLaVA等低成本大型多模态模型进行零样本推理。

**📊 数据集**

数据集：共2001张部分合成图像（等高线、真实天区、GMM天区），以及47357条自动生成的问题；另外对104张人类验证的历史发布图像进行对照测试。

**📈 对比分析**

比较方法：在每个模型上分别评估原始图像和衰化图像下的图像类型识别、图像源辨识以及角度高度测量的准确率和对数误差（log-ratio）。结果显示：低成本模型在区分等高线图与天区图时准确率低（约15‑66%），在识别真实天区图与合成天区图时准确率仅4‑14%；角度高度预测普遍过估，误差分布偏向正值。

**⚠️ 局限性**

局限性：①图像为合成，缺乏出版物常见的标注与格式；②仅测试低成本配置的三款模型，未包含开源或高端模型；③对比的历史图像样本量有限，且未提供人类基准；④模型版本和价格变动快，实验结果易失效；⑤仅评估了有限类型的问答（类别识别与角度测量），未覆盖更复杂的推理与数值计算。

---

## 149. How Finite-Rank Bottlenecks Shape the Low-Rank Adaptation Landscape

**arXiv ID:** 2610.23343 | [PDF](https://arxiv.org/pdf/2610.23343v1)

**作者:** Long Nguyen-Chi `[一作]` (VinUniversity), Binh T. Nguyen `[通讯]` (VinUniversity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LoRA在冻结模型下的理论框架，证明在核范数正则化下存在低秩全局最优解并给出秩上界。

**💡 创新点**

核心创新在于通过冻结网络形成的有限秩线性映射（bottleneck rank）解释低秩优点，无需NTK线性化；对多头注意力引入score centering和value‑projection invariance进一步收紧秩界；给出非凸LoRA梯度点的秩缺陷定理。

**🔧 技术方法**

使用核范数正则化、LoRA因子化与核范数等价、Bottleneck Rank分析、线性映射与投影压缩、KKT与谱分解、随机化Rademacher复杂度以及扰动理论。

**📊 数据集**

在GLUE基准上测试，使用Qwen2.5‑1.5B模型，对MNLI、SST‑2、RTE数据集进行LoRA训练。

**📈 对比分析**

通过训练损失曲线对比LoRA不同秩与完整微调，观察到秩阈值以下训练效果差异明显，阈值以上训练收敛几乎与完整微调一致，说明高秩LoRA可逼近全微调。

**⚠️ 局限性**

仅证明存在性，未证明梯度下降会收敛到这些低秩解；秩上界依赖全局最优解，难以在实践中直接计算；注意力的秩界为后验估计；非凸LoRA可能存在奇异局部最优，扰动理论仅提供一阶秩缺陷，未消除所有局部最优。

---

## 150. Non-Parametric Optimization for Scalable Learning in Stochastic Decision Problems

**arXiv ID:** 2610.23204 | [PDF](https://arxiv.org/pdf/2610.23204v1)

**作者:** Mohsen Amidzadeh `[一作]` (Aalto University), Mario Di Francesco `[通讯]` (Aalto University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于 Malliavin 微积分的非参数变分框架，用于求解时间变异随机优化（TV‑SO）问题，并实现了可扩展的深度学习算法 Stochastic Path Follower (SPF)。

**💡 创新点**

创新点在于：①给出了 TV‑SO 的非参数必要最优性条件；②将这些条件转化为能直接优化的能量函数；③设计的 SPF 算法对网络参数维度不敏感，显著提升了可扩展性。

**🔧 技术方法**

核心技术包括 Malliavin 微积分、随机微分方程（SDE）的马尔科夫性质、Euler–Maruyama 采样、Adam 优化器以及神经网络参数化的漂移/扩散函数。

**📊 数据集**

实验使用了合成数据（含噪声回归、Logistic 回归、线性最小二乘恢复等），并通过在不同维度和分布漂移设置下的仿真来验证算法。

**📈 对比分析**

与基准方法 PDM（路径微分法）和 PSG（近似梯度下降）比较，SPF 在最优性距离、目标子最优度量上均优于两者，并在高维/高参数化场景下保持较低的运行时间，证明了更好的性能与更优的可扩展性。

**⚠️ 局限性**

局限性包括：仅给出必要最优性条件，缺乏充分条件；实验仅基于合成数据，缺少真实场景验证；算法实现复杂，需要对 SDE 与 Malliavin 导数的数值求解有较高要求。

---

## 151. SwitchTube: A Proof-of-Concept System Introducing "Adaptable Commitment Interfaces" as a Tool for Digital Wellbeing

**arXiv ID:** 2610.22947 | [PDF](https://arxiv.org/pdf/2610.22947v1)

**作者:** Kai Lukoff `[一作]` (Santa Clara University), Alexis Hiniker `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计、开发并在三周内部署了Android应用SwitchTube，该应用提供三种YouTube观看界面：探索模式、专注模式和可在两者间切换的Switch模式。

**💡 创新点**

提出并验证了“可适应承诺界面”（Adaptable Commitment Interface）概念，允许用户根据情境自由切换，以提升主体感、满意度和目标一致性；同时强调内容质量而非仅减少屏幕时间。

**🔧 技术方法**

使用Android平台，调用YouTube Data API v3和Google Custom Search API获取视频与搜索结果；实现体验采样（ESM）、日志记录、Figma原型；采用非参数统计（Friedman、Wilcoxon）和主题分析。

**📊 数据集**

收集了46名美国YouTube用户的日志（共2063个视频播放事件）和体验采样问卷数据；此外基于606名用户的预调查数据筛选实验参与者。

**📈 对比分析**

通过描述性统计比较三种版本的使用时长、视频观看数、搜索次数等客观指标；使用体验采样评估主体感、满意度、目标一致性，采用Friedman检验和配对Wilcoxon检验；结果显示Switch版本在主体感和目标一致性上优于两者，且在满意度上优于专注模式。

**⚠️ 局限性**

局限性包括未测量Switch对整体YouTube使用时间的溢出效应；主页推荐未个性化，影响结果；受YouTube API配额限制导致功能妥协；样本偏年轻/亚裔/黑人，缺乏普适性；实验时长仅三周。

---

## 152. From Sliding Windows to Self-evolving Lifelong Memory: A Dual-perspective Survey of Memory Paradigms for Long-Horizon Robotics VLA Models

**arXiv ID:** 2610.23319 | [PDF](https://arxiv.org/pdf/2610.23319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 153. Contracting with an Agent under Model Mismatch

**arXiv ID:** 2610.23402 | [PDF](https://arxiv.org/pdf/2610.23402v1)

**作者:** Elif Yılmaz `[一作]` (University of Neuchâtel), Christos Dimitrakakis `[通讯]` (University of Neuchâtel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在代理模型与真实环境转移动力学不一致的主代理-代理 Markov 决策过程（PAMDP）中，如何通过贝叶斯学习估计代理的误设转移模型，并基于估计结果设计最优合同，从而最大化主方收益。

**💡 创新点**

创新点包括：① 将代理的模型误差纳入合同设计框架；② 采用有限模型集合的贝叶斯学习方法，动态更新对代理模型的后验分布；③ 结合反向归纳和线性规划求解最小支付合同；④ 通过实验验证估计精度、合同收敛与遗憾的关系，说明小模型误差对收益的显著影响。

**🔧 技术方法**

使用的技术包括：贝叶斯推断（后验更新）、MDP 反向归纳、线性规划求解最小支付合同、KL 散度与均方误差评估、模拟实验（随机生成 MDP、奖励、成本与合同）。

**📊 数据集**

实验数据集为随机生成的离散 MDP，状态数 5，动作数 3，时间步 1–3，奖励与成本取自 [0,1]，合同支付取自 [0,2]。未使用公开真实数据集。

**📈 对比分析**

与最优合同（已知真实代理模型）进行对比，评价指标包括期望遗憾、KL 散度（模型误差）和 MSE（合同误差）。实验显示：① 当模型集合包含真实代理模型时，后验迅速收敛，遗憾显著下降；② 模型集合越接近真实模型，收敛速度越快，合同误差越小；③ 仅包含近似模型时，遗憾虽不如最佳但仍明显优于随机合同，表明学习框架有效。

**⚠️ 局限性**

局限性：① 仅适用于离散、有限状态/动作空间；② 假设代理总是最佳响应且其动作可观测；③ 未给出理论收敛或最优性保证；④ 需要预先给定包含真实模型的有限候选集合，若不存在则无法完全收敛。

---

## 154. From Parallels to Divergence: Language Models' Learning of Action Nominal Constructions

**arXiv ID:** 2610.22468 | [PDF](https://arxiv.org/pdf/2610.22468v1)

**作者:** Iskar Deng `[一作]` (University of Washington), Shane Steinert-Threlkeld `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用语法工程化方法生成96种目标语言的合成语料，训练GPT‑2‑small模型并通过目标最小对比评估学习的动作名词化构造（ANC）及相关结构。

**💡 创新点**

首次将多维典型参数（句子词序、名词短语词序、对齐、补语系统、ANC策略）同时控制，构建统一语义输入的合成语料库，并对模型学习偏好与自然语言典型分布进行系统比较。

**🔧 技术方法**

利用Grammar Matrix、HPSG、ACE生成语法模型，使用MRS共享语义表示；训练GPT‑2‑small（从零开始）并通过针对34种语法现象的最小对比测试来评估。

**📊 数据集**

以BabyLM 2026 Strict（100M词）语料为原始英语输入，提取谓词‑论元结构，生成伪英语再映射为各目标语言。

**📈 对比分析**

在34个语法现象上平均达到88.4%准确率（基本句型近乎完美，复杂补语系统和ANC策略仍有挑战），与自然语言的分布部分一致，模型还能学习一些稀缺或未出现的组合。

**⚠️ 局限性**

受限于使用英语为源语导致的偏差、仅测试单一模型架构、未控制不同参数的单独影响、未考察语义/上下文对ANC选择的影响、以及分词器可能对合成词形的干扰。

---

## 155. MUB: An Identity-Masked Benchmark for Music Understanding in Multimodal Language Models

**arXiv ID:** 2610.22989 | [PDF](https://arxiv.org/pdf/2610.22989v1)

**作者:** Kun Liu `[一作]`, Lyra Tao `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并公开了 Music Understanding Benchmark（MUB），该基准包含500条身份被掩码的音乐录音以及对应的节拍、调性、拍号和完整结构区间标注，并提供评测协议与基线实现。

**💡 创新点**

创新点在于：①将身份信息完全掩码以减少记忆式捷径；②统一提供四项全球音乐属性预测和完整轨道结构区间输出；③基于现有专业注释工作复用数据并公开完整评测流水线，实现可审计的评测流程。

**🔧 技术方法**

技术手段包括：音频DSP估计节拍（谱峰自相关）、基于音高分布匹配的调性估计、拍号启发式判断、IoU 匹配的结构区间评估；评测通过JSONL文件与Python脚本实现。

**📊 数据集**

使用的数据集为从专业音乐注释档案中筛选的500条录音，覆盖全部24个调性组合、47–180 BPM区间、4666个结构区间，且去除了标题、艺术家、平台ID等元数据。

**📈 对比分析**

评测方法是与DSP基线（53.6% BPM准度、25.4% legacy 调性准度、36.6% 拍号准度）以及常数先验（9.0% BPM、9.2% 调性、98.4% 拍号）进行对比；结构区间F1为0，表明基线未实现结构分割。未对多模态语言模型进行实测。

**⚠️ 局限性**

局限性包括：调性评测规则存在错误（legacy 规则）；注释可靠性未独立验证；拍号分布高度不平衡（92% 4/4）；未解决录音版权与安全问题；身份掩码并不能完全防止音频识别；缺乏多模态模型评测、置信区间和更完善的评测机制。

---

## 156. Ego2Contact: Predicting Touch from Egocentric Human Video for Contact-Rich Robot Manipulation

**arXiv ID:** 2610.22932 | [PDF](https://arxiv.org/pdf/2610.22932v1)

**作者:** Qianqian Yang `[一作]` (Carnegie Mellon University), Zhiwen Fan `[通讯]` (Texas A&M University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用人类视角视频学习视觉‑触觉先验，并将其迁移到机器人手中，预测接触位置与压力，从而为机器人接触丰富的操作提供触觉输入。

**💡 创新点**

提出跨生物体的解剖区域对应接口，使人类触觉先验能与不同机器人传感器布局对齐，并采用条件流匹配生成器结合时序场景与手部局部视觉特征，提升触觉预测的准确性与可迁移性。

**🔧 技术方法**

使用冻结的 DINOv2 与 WiLoR 编码器提取全局与局部特征，跨注意力融合时序 Transformer，结合 Diffusion Transformer + AdaLN‑Zero 的条件流匹配生成器，并采用流式学习目标进行训练。

**📊 数据集**

主要数据集为 EgoTouch、EgoTactile 人类触觉数据集，用于预训练；在 Isaac Gym 机器人演示数据中收集的配对视频与传感器读数用于微调与评估；亦在真实 xArm6 + BrainCo Revo 2 手上进行验证。

**📈 对比分析**

与 TouchAnything 的复现模型和 pose‑conditioned 版本进行对比；在人类数据上实现最高的 C.IoU/V.IoU/T.Acc/MAE；在机器人模拟与真实实验中，预测触觉显著提升任务成功率，部分恢复真实触觉的优势。

**⚠️ 局限性**

仍需收集机器人级配对数据进行微调；依赖手工制定的解剖对应关系，迁移到新手或不同传感器布局仍需额外工作；对新对象的泛化仍有限。

---

## 157. Upper bounds on the number of causal pieces in spiking neural networks

**arXiv ID:** 2610.22484 | [PDF](https://arxiv.org/pdf/2610.22484v1)

**作者:** Dominik Dold `[一作]` (University of Vienna), Philipp Petersen `[通讯]` (University of Vienna)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

本文推导了单尖峰脉冲神经网络（SNN）因果片段数量的两类上界，分别是通用组合上界（多项式贝塞尔数）和特定神经元模型的超平面排列上界；

**💡 创新点**

创新点在于将因果片段数量与多项式贝塞尔数及超平面排列计数相联系，给出更紧凑的理论上界，并考虑了多种脉冲神经元模型（nLIF、ReLU型、LIF等）；

**🔧 技术方法**

技术方法包括组合计数、lonesum矩阵的多项式贝塞尔数、超平面排列理论以及坐标变换与弱顺序域的分解；

**📊 数据集**

本研究为理论分析，不使用数据集；

**📈 对比分析**

无实验比较，性能指标为理论上界的紧密程度；

**⚠️ 局限性**

局限在于仅给出上界，缺乏下界与实际网络性能验证，且对延迟与权重符号的假设仍较宽松，实际应用需进一步实验验证。

---

## 158. WebCoder: Scaling General Frontend Development with Artifact-Grounded Agentic Reinforcement Learning

**arXiv ID:** 2610.23104 | [PDF](https://arxiv.org/pdf/2610.23104v1)

**作者:** Zijun Dou `[一作]` (Tsinghua University), Jie Tang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练了一个能在 Webdev-World 环境下执行前端复现、生成、修改与修复的通用前端开发代理。

**💡 创新点**

通过在可执行应用上直接评估终端奖励并用同一条检验清单进行中间进展评估，实现跨阶段信用分配；设计了 Webdev-World 与 Webdev-Bench；采用上下文压缩与消息树样本支持超长轨迹训练。

**🔧 技术方法**

采用 GRPO 强化学习与大型语言模型，结合工具调用、终端检查器、视觉/功能检验、加权检验清单、artifact potential、信用分配、上下文压缩和消息树样本；基础模型为 Qwen3.8-27B。

**📊 数据集**

使用 Webdev-World 生成的复现、生成、修改与修复任务；Webdev-Bench 40 任务（188 需求，1125 隐藏清单项）；Vision2Web 66 个多页前端项目。

**📈 对比分析**

与多款专有模型（Claude-Opus-4.8、GPT-5.4 等）和多款开源模型进行对比；在 Webdev-Bench 上实现 72.8 的需求满足率，超过基线 64.8 与 Claude-Opus-4.8 的 65.1；在 Vision2Web 上平均得分 62.4，接近 Claude-Opus-4.8 的 62.5，显著优于基线 41.1。

**⚠️ 局限性**

模型仍需大量算力，训练依赖于自动化评审器和合成任务，对真实项目的适用性和多样性有限；在视觉与功能评估上仍有提升空间；以及对环境与工具的依赖可能限制迁移。

---

## 159. What It Is, Not What It Says: Securing Agentic Browsing Interactions with Web Semantics

**arXiv ID:** 2610.22697 | [PDF](https://arxiv.org/pdf/2610.22697v1)

**作者:** Umar Iqbal `[一作]` `[通讯]` (Washington University in St. Louis), Umar Iqbal (Washington University in St. Louis)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出IntAct系统，利用浏览器原生的Web语义和来源信息为AI代理构建元素级别的可信表示，并通过虚拟隔离与用户同意机制来防止代理受注入攻击；

**💡 创新点**

创新点在于：①为每个DOM元素记录作者、修改者及其能力，②采用逐元素规划避免跨元素影响；③在不改动浏览器的前提下通过CDP注入脚本完成完整的追踪与归属；

**🔧 技术方法**

技术手段包括：浏览器JavaScript接口包装、CDP事件监听、DOM+可访问性树的融合、LLM逐元素规划、能力与影响分类、用户授权提示；

**📊 数据集**

使用了1270个真实网页的爬取数据集（含15,854个跨域框架），以及1,091个注入攻击的真实网页基准；

**📈 对比分析**

与传统单次规划器相比，IntAct的攻击成功率从73.1%降至12.8%；功能丢失仅占3.7%的任务；成本与耗时大约翻倍，但可通过并行化显著降低；

**⚠️ 局限性**

局限性包括：额外的计算和时间开销、对可用性攻击无保障、依赖JavaScript注入且部分脚本无法完整跟踪、模型调用仍需大量资源、未覆盖所有浏览器实现与第三方脚本的复杂交互。

---

## 160. Causal Latent Workspaces for Looped Transformers

**arXiv ID:** 2610.22796 | [PDF](https://arxiv.org/pdf/2610.22796v1)

**作者:** Ilia Zaznov `[一作]` `[通讯]`, Ilia Zaznov

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在循环Transformer中加入一种新的因果工作空间机制，使每个位置可以写入并读取与其前缀相关的压缩摘要；

**💡 创新点**

创新点在于将门控的因果写入与基于键的内容寻址读取结合到共享深度的Transformer循环中，形成一种可重用且可在每轮刷新的大容量工作空间；

**🔧 技术方法**

使用了门控向量写入、学习得到的键向量读出、共享的Transformer块、位置锚定、以及多轮循环（四轮）和可选的状态保留策略；

**📊 数据集**

在三大字节级语料库上评测：WikiText-103 raw、enwik8、TinyStories；

**📈 对比分析**

与六种对照模型（Plain、Fixed共享循环、Adaptive退出混合、Anchored共享循环、Anchored条件密集、以及我们提出的工作空间Transformer）在相同参数和训练目标预算下进行对比，结果显示在101.71M参数规模下，工作空间模型在所有三种语料上均取得最低的平均困惑度；在0.918M参数规模下，在WikiText-103和TinyStories上表现最好，在enwik8上略逊于条件密集循环；

**⚠️ 局限性**

局限性包括：需要额外的写入和读取计算导致训练时间略长；在某些字节分布（如enwik8的小规模模型）上优势不明显；未对更大上下文长度、子词级建模或更长训练时序进行评估；

---

## 161. AWAM: Active Trajectory-Conditioned View Imagination for Robust Robot Manipulation

**arXiv ID:** 2610.23396 | [PDF](https://arxiv.org/pdf/2610.23396v1)

**作者:** Jiawei Geng `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种虚拟主动感知框架，先预测任务相关的摄像机轨迹并用几何条件世界模型生成对应的虚拟视图，压缩为持久的世界记忆，再与实时观测融合进行闭环动作预测。

**💡 创新点**

创新点在于：① 将主动感知从物理移动转化为“虚拟”摄像机轨迹规划；② 用几何条件世界模型把预测的轨迹转化为多视角的视觉表示；③ 将这些预测信息压缩为固定大小的记忆，并在控制时持续复用，弥补现场观测缺失。

**🔧 技术方法**

使用 Transformer 预测轨迹、几何条件世界模型（如 WorldStereo）生成多视角特征、交叉注意力融合记忆与实时观测、扩散 Transformer 策略生成动作。

**📊 数据集**

在 LIBERO‑Occ、LIBERO‑Plus、RLBench‑OG Occlusion1 三个仿真基准以及真实机器人实验（ARCHER‑Y6 arm）上进行评估。

**📈 对比分析**

相较于基线 π0、Motus、Fast‑WAM、VIM、TVVE 等方法，实验显示在 LIBERO‑Occ 平均成功率 75.9%（比最优基线 73% 提升 2.9%），RLBench‑OG Occlusion1 76.5%，LIBERO‑Plus 62.4%，以及真实机器人 94.4% 的成功率，证明虚拟感知记忆显著提升在遮挡与分布移位场景下的鲁棒性。

**⚠️ 局限性**

主要限制：① 记忆仅在 episode 初始构建，长周期或大状态变化时不自动刷新，导致长周期任务性能下降；② 虚拟视图填补仍受几何重建不完整影响，存在大空洞；③ 对轨迹预算和视图数的敏感性需要进一步调优。

---

## 162. MS-Eval: Benchmarking Sub-Agent Delegation and Orchestration in MCP-Intensive Working Scenarios

**arXiv ID:** 2610.22869 | [PDF](https://arxiv.org/pdf/2610.22869v1)

**作者:** Bowen Ding `[一作]`, Tao Lin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了MS-Eval评估工具，用于量化LLM多代理系统MAD-SAE在MCP密集工作场景中的委托与编排质量。

**💡 创新点**

首次将子代理委托适当性、编排效能与统一的MSQ指数结合，系统化评估多代理与单代理的性能、效率和成本。

**🔧 技术方法**

采用模型上下文协议（MCP）、自定义对话框与工具列表、DeepSeek等匹配模型以及Kimi-Code控制执行框架。

**📊 数据集**

使用从Toolathlon‑Gym抽取的166个长周期任务，覆盖八个真实工作场景，并标注黄金委托计划。

**📈 对比分析**

对四种MAD‑SAE变体和八个前沿模型进行基准测试，发现Spec变体在MSQ上最高，开源模型普遍优于封闭模型，但成本提升取决于委托质量。

**⚠️ 局限性**

局限在于评估仅覆盖MCP密集情境，无法验证在无工具或不同工具生态下的适用性；且成本与收益仍受模型规模与部署成本的制约。

---

## 163. Are Large Reasoning Models Good Everyday Reasoners? A Comparison with Non-Reasoning Models on Logical Reasoning Tasks

**arXiv ID:** 2610.22538 | [PDF](https://arxiv.org/pdf/2610.22538v1)

**作者:** Kentaro Ozeki `[一作]` (Keio University), Mitsuhiro Okada `[通讯]` (Keio University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了大型推理模型在日常逻辑推理场景中的表现，探讨其与日常可接受性之间的差距。

**💡 创新点**

创新点在于将日常可接受性与形式有效性转化为“模式控制逻辑推理”任务，并构建了双维度（日常、正式）评估框架。

**🔧 技术方法**

主要技术包括链式推理（Chain‑of‑Thought）提示、post‑training 推理模型（gpt‑oss、olmo、qwen3 等）以及对照的非推理模型。

**📊 数据集**

使用自构造的 200 条实例数据集，涵盖 10 种推理模式，每种模式 20 条实例，标注日常可接受性与形式有效性。

**📈 对比分析**

通过比较推理模型与非推理模型在日常可接受性一致率、正式偏差 (Formalist Bias) 等指标，发现推理模型在正式有效性与日常可接受性不一致时表现显著退化，非推理模型有时更符合日常推理。

**⚠️ 局限性**

局限性包括数据集规模有限、缺乏大规模人类日常推理标注、评估仅聚焦离散推理模式，且上下文提示对模型行为的控制效果不稳定。

---

## 164. CatchUp: Depth-Aware Re-Admission of Visual Tokens for Efficient Large Vision-Language Models

**arXiv ID:** 2610.23324 | [PDF](https://arxiv.org/pdf/2610.23324v1)

**作者:** Eunju Park `[一作]` (Ewha Womans University), Uiwon Hwang `[通讯]` (Ewha Womans University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练的视觉令牌重新接纳框架 CatchUp，结合可逆视觉令牌路由和局部转移校正 (LTC)，在大型视觉语言模型推理时显著减少计算和内存开销，同时保持高精度。

**💡 创新点**

创新点在于：① 通过在多层路由点重新评估查询相关性，允许被跳过的视觉令牌随时重新接纳；② LTC 使用语义与空间相似度的锚点估计被跳过层的表示演化，从而弥补被跳过的隐藏状态失效问题；③ 该方法完全训练无关，易于集成到现有 LVLM。

**🔧 技术方法**

使用的技术包括：可逆视觉令牌路由（基于查询条件的注意力评分）、局部转移校正（anchor 选取、语义余弦相似度、空间欧氏距离、温度归一化与残差修正）、多层路由策略（在层 2、7、15、23 处做路由）以及对比实验评估。

**📊 数据集**

主要使用的评测数据集有：RefCOCO、RefCOCO+、RefCOCOg（精细定位任务），以及 POPE、VQAv2、VizWiz、GQA、MM‑Vet、MMBench（通用多模态推理任务）。

**📈 对比分析**

与 FastV、SparseVLM、PyramidDrop、Nüwa、ReRoute 等现有可逆或不可逆路由方法比较。CatchUp 在 64/128/192 令牌预算下，在精细定位任务中平均提升 3.94–5.13 点准确率，在通用多模态任务中保留 92.17% 的完整模型性能，同时减少 8% 的端到端延迟，且占用最少的 KV 缓存内存。

**⚠️ 局限性**

局限性：① LTC 仍需额外的参考状态快照，导致少量辅助内存开销；② 在极端低预算（如 64 令牌）下，精细定位任务仍显著下降，说明该方法对高度细粒度视觉定位仍有挑战；③ 目前仅在 LLaVA-1.5-7B 与 Qwen2.5-VL-7B-Instruct 两种模型验证，尚未在更大模型规模或其他视觉语言框架上进一步评估。

---

## 165. Stealthy Poisoning Attacks via Reused KV Cache of Reasoning

**arXiv ID:** 2610.22873 | [PDF](https://arxiv.org/pdf/2610.22873v1)

**作者:** L. Zhu `[一作]` (Aalto University), S. Szyller `[通讯]` (Aalto University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种黑盒攻击，利用已丢弃的推理轨迹在KV缓存中的残留信息对后续查询进行注入，达到操纵大型语言模型输出的目的。

**💡 创新点**

创新点在于发现并利用推理轨迹被丢弃后其KV缓存残留仍能影响后续生成；通过设计目标导向的前置轨迹R1和携带目标的响应R2，并在二者共同编码时产生的“条件缓存残留”来实现无缝注入；采用反向追踪搜索在仅交互的黑盒环境中高效构造攻击序列。

**🔧 技术方法**

主要技术包括：Transformer KV缓存的预填充与重用机制、条件缓存与普通缓存的对比、基于prompt工程的R1、R2生成、反向追踪搜索算法、外部安全评估器（ProtectAI DeBERTa）对响应的检测。

**📊 数据集**

使用了GPQA、MBPP、BigCodeBench三大公开数据集，分别覆盖多项选择题、编程实现及代码生成任务。

**📈 对比分析**

与Baseline（仅注入R1）及白盒硬码攻击HijackKV进行对比；在Exact Query Cache Reuse下，R2ECAP在Gemma、GLM、Qwen三模型上分别达79%、71%、67%的攻击成功率；在跨查询缓存重用情形下，攻击成功率为16%–46%，显著高于Baseline且速度约快80倍。

**⚠️ 局限性**

局限性包括：仅在7B–9B参数的开源模型上测试，未验证更大模型的可行性；防御评估仅检测保留响应R2，未对完整缓存构造过程中的前置轨迹进行检测，可能漏检。

---

## 166. An improved randomized algorithm for finding sinks of Acyclic Unique Sink Orientations of the binary hypercube

**arXiv ID:** 2610.22546 | [PDF](https://arxiv.org/pdf/2610.22546v1)

**作者:** Thomas Dueholm Hansen `[一作]`, Uri Zwick `[通讯]` (Tel Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了一种改进的随机算法，用于在二进制超立方体的无环唯一汇点（AUSO）中寻找汇点，期望查询次数为e^(√(2+o(1))n)。

**💡 创新点**

这是二十年来首次对该问题的改进算法，显著降低了查询次数的上界。

**🔧 技术方法**

使用了改进的随机面算法（Random-Facet algorithm），并结合了线性规划和LP类型问题的算法。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了二进制超立方体的无环唯一汇点（AUSO）作为研究对象。

**📈 对比分析**

与现有的随机面算法相比，改进算法的期望查询次数从e^2√(n)降低到e^(√((2+o(1))n))，显示出显著的性能提升。

**⚠️ 局限性**

该算法的局限性在于它仅适用于无环唯一汇点（AUSO），并未改进更一般的算法。

---

## 167. HeRE: A Novel Benchmark for Hebrew Relation Extraction via LLM-Guided Denoising of Knowledge Graph Alignments

**arXiv ID:** 2610.22542 | [PDF](https://arxiv.org/pdf/2610.22542v1)

**作者:** Ron Keinan `[一作]` (Bar-Ilan University), Reut Tsarfaty `[通讯]` (Bar-Ilan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了首个希伯来语关系抽取基准HeRE，并提出了通用的基于 Wikidata‑Wikipedia 远程监督的去噪管道；

**💡 创新点**

创新点在于：①系统化比较五种去噪策略（NLI编码器、LLM NLI、交叉训练分类器、少量提示 LLM、基础 LLM），并以多数投票达成银标；②发布1.5M实例的银标数据和知识图谱，为低资源语言提供可复现的基准；

**🔧 技术方法**

采用了 NLI 逻辑验证、指令微调 LLM、交叉训练多类谓词分类、少量示例提示和基础 API LLM 等技术；

**📊 数据集**

使用的数据集包括：Wikidata、希伯来语 Wikipedia、人工标注的500条验证集和500条测试集，以及通过上述管道得到的1.5M银标实例；

**📈 对比分析**

在500条验证集上进行比较，最佳开源模型 Gemma‑4‑31B‑it 的 F1 为0.915，几乎等同于 GPT‑5.4（0.921）；在全新测试集上仍达0.844，未见过谓词下为0.776；跨语言实验表明多语言 LLM 在少量提示下优于希伯来本地编码器；

**⚠️ 局限性**

局限性包括：仅覆盖百科文本，对社交媒体等领域迁移有限；谓词稀缺导致少数谓词表现差；金标文本比银标短，长度差异可能影响泛化；管线已验证于希伯来语，其他低资源语言的泛化尚待验证。

---

## 168. An Error Taxonomy for Property Graph Data

**arXiv ID:** 2610.23083 | [PDF](https://arxiv.org/pdf/2610.23083v1)

**作者:** Valerie Restat `[一作]` (FernUniversitat in Hagen), Uta Störl `[通讯]` (FernUniversitat in Hagen)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了图数据质量，提出了技术无关的属性图错误分类体系，并系统地将其与表格数据错误进行映射。

**💡 创新点**

创新点在于首次构建完整的属性图错误分类，并首次将图错误与表格错误进行对齐，揭示两种数据模型在错误表现上的差异。

**🔧 技术方法**

技术手段主要是对现有表格错误分类的文献综述，结合图数据的结构特点设计层次化错误类型并建立映射关系。

**📊 数据集**

使用的示例数据为音乐领域的演示图，未使用公开大型图数据集进行实验。

**📈 对比分析**

由于研究聚焦于理论构建，未进行实验比较或性能评估，主要通过示例说明映射关系。

**⚠️ 局限性**

局限性包括仅聚焦错误描述与映射，未提供检测或修复方法；仅适用于单源属性图，未覆盖多源、RDF 等其他图模型。

---

## 169. Efficient CPU-GPU collaborative MoE inference via Adaptive Expert Assignment

**arXiv ID:** 2610.23006 | [PDF](https://arxiv.org/pdf/2610.23006v1)

**作者:** Wenhui Lv `[一作]`, Junhao Cheng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于CPU‑GPU协作的混合专家推理系统Apex，能够根据专家的token负载动态决定哪些GPU‑missed专家在CPU上计算，哪些在GPU上提前迁移，从而在有限GPU显存下高效推理大型MoE LLM；

**💡 创新点**

创新点在于：①采用基于token负载的自适应专家分配；②结合显式的精确分配与下一层的预测预取（speculative ranker）实现迁移与计算路径的实时平衡；③通过在线率驱动的决策算法动态选择迁移数量，最小化两条并行路径中的最大延迟；

**🔧 技术方法**

使用技术包括：混合CPU‑GPU MoE推理框架、基于GPU显存池管理的专家缓存、预取专家的下一层路由预测、在线计算率（CPU、GPU、传输）估算、指数滑动平均更新、张量分发与合并；

**📊 数据集**

实验数据集采用公开的MMLU验证集作为推理负载，评估了三款代表性MoE模型（Qwen3.5-35B-A3B、Qwen3-Next-80B-A3B-Instruct、GLM-4.5-Air）；

**📈 对比分析**

与三种基线（Mixtral-Offloading、Fiddler、Post‑Fetch）比较，Apex在prefill阶段平均提升31.2%（最高51.9%）预填速度，decode阶段平均提升31.1%（最高39.5%）吞吐量，且在不同GPU显存预算和批量大小下均保持优势；

**⚠️ 局限性**

局限性包括：需额外实现CPU‑GPU协作和专家缓存逻辑；在极小批量或高token负载极不均匀时自适应分配仍受限；PCIe带宽与CPU计算瓶颈对性能提升产生上限。

---

## 170. Arbitrary Functors as Modalities in a Linear Calculus

**arXiv ID:** 2610.23023 | [PDF](https://arxiv.org/pdf/2610.23023v1)

**作者:** Jean-Philippe Bernardy `[一作]` (University of Gothenburg and Chalmers University of Technology), Andreas Abel `[通讯]` (University of Gothenburg and Chalmers University of Technology)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究在线性 λ 计算机中将模态视为范畴内的端函子，建立了一个通用的范畴语义框架，支持非分配性模态和带计算内容的模态转换，并在此基础上推导出资源语义、关系语义及归一化-by-评估（NbE）方法，最终得到一套完整的、可归一化的语言形式。

**💡 创新点**

创新点在于：① 取消了传统模态需要的乘法分配性假设，允许任意模态；② 通过显式的自然变换把模态转换变成可计算的操作；③ 统一了效应、共效应、定量模态等多种应用场景；④ 用资源预设模型与预取模型实现完备的语义；⑤ 在 Agda 中完成了完整的机械化证明。

**🔧 技术方法**

技术手段包括：
- 线性多模态 λ 计算机与范畴理论（对称单张闭合范畴、有限乘积、内 Hom 等）。
- 模态结构定义为一个环状范畴（ringoid），其中包含两种乘法与模态间的自然变换。
- 资源语义通过 Kripke/相位模型与预取模型（presheaf）实现，捕捉资源变换与时间等量化属性。
- 关系语义通过构造关系范畴，支持信息流与非干扰性证明。
- 归一化-by-评估（NbE）利用预取模型与完全性证明得到规范语法。

**📊 数据集**

本文为理论工作，未使用实测数据集；所有结论均通过形式化证明（Agda）得到。

**📈 对比分析**

通过范畴语义的完备性与 NbE 归一化技术，证明了语言的类型安全、归一化以及语义一致性；并通过示例（数组、敏感度分析、效应、共效应、时间等）展示了模型的表达能力，未做传统算法性能比较。

**⚠️ 局限性**

局限性包括：
- 需要显式写出模态转换，语法上比传统系统更冗长；
- 目前未支持多态与某些类型构造（如直和类型的完整处理）；
- 对模态等式的可判定性有要求，实际实现中可能需要手工提供等价证明；
- 关系语义对大规模程序的可扩展性尚未评估；
- 仅在 Agda 环境下完成机械化，迁移到其他证明助手或编译器需要额外工作。

---

## 171. LexPilot: Benchmarking the Proactive Interaction Ability of Legal LLMs with Laypeople in Chinese Civil Case Consultation

**arXiv ID:** 2610.23141 | [PDF](https://arxiv.org/pdf/2610.23141v1)

**作者:** Wenbin Zhang `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

建立LexPilot基准，评估LLM在不完整法律咨询中的主动交互能力，并提出LexAwN基线方法。

**💡 创新点**

将主动交互与判决预测结合，提供真实案例驱动的完整/缺失咨询对，并系统分析缺失细节对LLM表现的影响。

**🔧 技术方法**

采用Prompting（LexAwN）、多模型对比（OpenAI、Claude、Google等）、Macro‑F1、Accuracy、QWK等指标；使用用户模型模拟真实回应。

**📊 数据集**

432份中国民事判决（36类诉因），共864条咨询（每案完整/欠缺对），含判决结果。

**📈 对比分析**

对10款LLM进行LCA、DCG、LJP三任务评测；LCA Macro‑F1最高0.75；DCG最高0.84；主动交互后LJP QWK平均比直接预测高0.041，仍比完整咨询低23.84%。

**⚠️ 局限性**

仅覆盖中文民事司法，数据为人工合成，需昂贵人工审核，未考虑非专业法律领域或不同法系，缺乏真实对话的多样性。

---

## 172. Reasoning Reranking with Semantic IDs as the Sole Item Representation

**arXiv ID:** 2610.22797 | [PDF](https://arxiv.org/pdf/2610.22797v1)

**作者:** Zhenwei Tang `[一作]` (Google), Xinwu Cheng `[通讯]` (Google)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在仅使用语义 ID（SID）而不依赖可读文本的前提下，构建一个受控测试平台，并评估现有模型在该场景下的排名与推理能力，随后通过四种后训练方法（SFT、RFT、OPD、RLVR）提升模型的 SID‑only 排序性能。

**💡 创新点**

创新点在于：①提出了纯 SID 逻辑推理+重排序的任务定义和对应的受控测试框架；②首次验证 SID 嵌入携带的行为信号能在无文本输入时实现有效排序；③通过对比多种后训练策略，发现 RLVR（带阶梯化排序奖励）在 8B 规模上能达到与阅读约 20 倍文本长度的强大 LLM 相当的效果；④对模型输出的推理文本进行配对审计，揭示了排名与描述质量的分离问题。

**🔧 技术方法**

使用的技术包括：大型预训练语言模型（OpenOneRec‑1.7B/8B）、SID 词表、两阶段推理+提取接口、四种后训练流程（SFT、RFT、OPD、RLVR）、GRPO（梯度强化学习）、NDCG、Recall、MRR 等评价指标，以及配对理据审计（Claude‑Opus‑5 作为判定者）。

**📊 数据集**

采用公开的 Kuaishou 短视频交互数据构建的 OpenOneRec 研究版数据集，包含 400 名用户的单正样本、100 项候选池、嵌套历史与候选扩展，生成多种渲染（SID、文本、双重）和不同历史长度、候选数的实验配置。

**📈 对比分析**

与文本阅读 LLM（Gemini‑3.6‑Flash、GPT‑5.6‑Terra、Claude‑Sonnet‑5）和随机基线进行对比。8B RLVR 在 SID‑only 任务中取得 NDCG@5=0.277，几乎匹配 Gemini 的 0.276，并且在无历史、长历史条件下仍保持优势；SFT 也显著提升了界面错误率。相比之下，RFT、OPD 只实现 SFT 的一部分提升，且 1.7B 规模下效果与 SFT 无显著差别。

**⚠️ 局限性**

局限性包括：①模型在 SID‑only 情况下虽然能排序，但推理文本对项目描述不准确，容易出现误称；②后训练未能改进推理本身，仅提升了排序输出；③实验仅覆盖 Kuaishou 短视频数据，缺乏跨域验证；④在极长历史下仍存在一定的界面错误率。

---

## 173. Xiaomi-Lens-0 Technical Report

**arXiv ID:** 2610.23279 | [PDF](https://arxiv.org/pdf/2610.23279v1)

**作者:** Pei Fu `[一作]`, Jian Luan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种4B规模的多模态大语言模型，能够在共享视觉特征上直接预测目标掩码，并通过“视觉潜变量”实现文本推理与视觉定位的交互。

**💡 创新点**

①在共享视觉编码器的patch grid上直接解码掩码；②将语言模型生成的mask token状态作为视觉潜变量，实现在推理链中对中间区域进行像素级监督；③提出LensBench基准，系统评估隐式/显式指令及多目标场景下的定位能力。

**🔧 技术方法**

基于Qwen3.5视觉-语言框架，使用Transformer、轻量级mask解码器、BCE+Dice损失、联合语言+掩码训练；构造“grounded reasoning traces”与“visual latents”；采用自定义数据混合进行两阶段训练。

**📊 数据集**

RefCOCO/RefCOCO+/RefCOCOg、COCO 2017、ReasonSeg、RoboRefIt、MME-RealWorld（英中）、CV-Bench、V*、4K/8K HR-Bench、CountQA、ColorBench 以及自研的LensBench；同时结合Objects365、LVIS、PACO、PhraseCut等公开掩码数据。

**📈 对比分析**

对REC、RES、多实例定位、ReasonSeg、RoboRefIt、细粒度感知等六类任务与现有MMLM、分割模型进行同等条件对比；在所有RES分裂上取得最高cIoU（平均81.4），REC平均提升5.6点，COCO多实例精度最高；在LensBench中，Aerial、Driving、Panoramic域cIoU分别领先7.0/0.2/1.5点，整体性能显著优于基线。

**⚠️ 局限性**

对隐式指令尤其是多目标全景场景的定位仍然表现欠佳；模型仍需大量标注掩码数据；在极端高分辨率或复杂场景下的细粒度推理仍有提升空间；训练成本较高。

---

## 174. Spatial Memory by Agents, for Agents

**arXiv ID:** 2610.22736 | [PDF](https://arxiv.org/pdf/2610.22736v1)

**作者:** Qianwei Wang `[一作]` (Princeton University), Hao Xu `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用编码代理在固定预算内自动设计机器人空间记忆系统，结合公共反馈迭代实现、评估与修订；

**💡 创新点**

创新点在于将编程代理用于空间记忆的全流程设计，能在多任务场景下显著提升跨任务性能并产出紧凑存储方案；

**🔧 技术方法**

采用 GPT‑6、Codex 等大型语言模型编写构建与检索代码，利用共享模块池（YOLOE、FastSAM、OpenCLIP、Qwen3.5‑4B 等）完成检测、分割、特征提取、标题生成和多模态嵌入；

**📊 数据集**

使用 ScanNet、HM3D、ScanRefer、ScanEnts3D、OpenEQA、OC‑NaVQA 等数据集进行四段式开发与冻结评估，共计 10,347 条查询；

**📈 对比分析**

在五个互补任务轨道（类别级定位、指代式定位、开放式空间问答、时间问答、导航历史问答）上对比，Codex/GPT‑6 Astra 取得 44.23 的统一分数，显著优于最强基线 ReMEmbR 的 29.32（提升 50.9%），同时包大小仅 4.3‑23.9 MiB；

**⚠️ 局限性**

局限性包括对单个机制的可解释性不足、主要依赖低成本 4B 模型导致对更强云端推理器的探索有限、以及与不同答案器配合时性能波动较大。

---

## 175. Structure-Aware Spatial Kernels for Gaussian Process Distributed Port-Hamiltonian Systems

**arXiv ID:** 2610.22575 | [PDF](https://arxiv.org/pdf/2610.22575v1)

**作者:** Tianyi Yang `[一作]` (Vanderbilt University), Thomas Beckers `[通讯]` (Ulm University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种结构感知的核设计，用于Gaussian Process分布式端口-汉密尔顿系统（GP-dPHS），通过去除无关状态坐标并将状态与空间依赖解耦，使用张量积核实现对空间异质性的先验建模。

**💡 创新点**

创新点在于将已知的物理输入不变性与粗粒度空间结构信息直接嵌入核函数设计中，实现了状态与空间依赖的模块化分离，使得可以在不需要完整空间剖面解析的情况下，选用周期性、Matérn或变点核匹配不同的空间异质性；并且通过这种结构化核显著提升了GP-dPHS的重构精度。

**🔧 技术方法**

主要技术包括端口-汉密尔顿系统理论、Gaussian Process回归、张量积核（state × space）、自适应核参数学习（RBF、周期、Matérn、变点核）以及数值积分和梯度一致性检验。

**📊 数据集**

实验数据集为基于一维非线性字符串模型生成的空间-时间数据，包含三种空间刚度剖面（周期性、C²波峰、半分段）和一个空间均匀基准。

**📈 对比分析**

与传统使用完整输入和RBF核的基线相比，结构解耦（SD-RBF）可降低约30-40%误差；再加上空间匹配核（SD-Per./SD-Mat./SD-CP）则可在位移、应变和能量误差上实现90%以上的提升，且显著降低了积分一致性误差；所有改进均通过RMSE、curl误差等指标量化。

**⚠️ 局限性**

限制主要在于假设系统的J、R、G算子已知且能量密度可分离；对多维空间或更复杂的物理耦合（如非线性耗散）尚未验证；此外，核参数的初始设定和训练成本在大规模数据时可能成为瓶颈。

---

## 176. Scalable Levenberg-Marquardt Training Framework for Deep Learning

**arXiv ID:** 2610.22925 | [PDF](https://arxiv.org/pdf/2610.22925v1)

**作者:** Xianghao Wang `[一作]` (Australian National University), John Taylor `[通讯]` (Australian National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出LeMA，一个多GPU上可大规模训练深度网络的Levenberg-Marquardt框架；

**💡 创新点**

通过推导精确的dual形式、Jacobian分块和split-K优化，显著降低内存占用并实现高效扩展；

**🔧 技术方法**

使用dual LMA、Jacobian分块、SPMD并行、NCCL/Gloo通信、split-K GEMM等技术；

**📊 数据集**

在MNIST数据集上验证，使用CNN-100M、CNN-1B、LeNet-5和MLP模型；

**📈 对比分析**

与传统LMA和PCG实现比较，单GPU上加速8.3×、8GPU上99.7%扩展效率，内存使用大幅减少；

**⚠️ 局限性**

受限于样本独立性（无法支持Batch Normalization）、Jacobian重复评估导致计算瓶颈、缺乏分布式线性求解器以及调参与收敛性尚待进一步研究。

---

## 177. LLM-Guided Dual-Stage Latency Optimization for Closed-Loop 6G V2I Digital Twins

**arXiv ID:** 2610.22848 | [PDF](https://arxiv.org/pdf/2610.22848v1)

**作者:** Liu Cao `[一作]` (City University of Hong Kong), Shuguang Cui `[通讯]` (Chinese University of Hong Kong)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在6G V2I系统中提出闭环双阶段端到端延迟优化框架，先用压缩个性化联邦学习（CPFL）构建包含路径损耗和关键性能指标的决策导向数字孪生（DT），再利用统一的LLM‑指导多智能体近端策略优化（SAGE‑MAPPO）分别最小化DT重建时模型参数交换延迟和服务感知通信延迟；

**💡 创新点**

创新点包括：①将压缩个性化联邦学习与多任务学习相结合，实现对PL和KPI的联合预测；②在闭环双阶段延迟优化中引入大型语言模型作为训练时语义规划器，显著降低低层代理的探索开销并实现跨阶段策略复用；③提出统一的SAGE‑MAPPO框架，兼容两阶段不同目标，提升收敛速度与终端延迟性能；

**🔧 技术方法**

核心技术包括：压缩个性化联邦学习（CPFL）+多任务学习；路径损耗与KPI联合预测模型；语义增强的多智能体近端策略优化（SAGE‑MAPPO）；LLM（Qwen3.5‑9B）作为高层语义规划器；通信延迟与服务延迟的联合优化模型；

**📊 数据集**

使用基于城市道路网的仿真数据生成的6G V2I场景，包含MBS布置、LoS/非LoS信息、交通负载与信道测量；并通过该仿真数据训练CPFL模型与SAGE‑MAPPO；

**📈 对比分析**

与FedAvg‑MTL、FedFly、FedGBDT等联邦学习基线相比，CPFL在路径损耗与KPI预测的RMSE、MAE均显著下降；与基于最大SINR和软关联的传统关联策略相比，DT辅助关联提升了吞吐量、降低排队延迟并提升QoE；与单路径、加速MARL和标准MAPPO的延迟优化方案相比，SAGE‑MAPPO在两阶段任务中均实现了更低的端到端延迟分布、缩短的收敛周期以及更高的奖励；

**⚠️ 局限性**

主要限制在于实验仅基于仿真场景，未验证在真实车辆轨迹与环境噪声下的鲁棒性；LLM语义规划对预测误差和语言模型推理延迟的依赖需进一步评估；DT更新频率与实时性能的自适应调度仍待研究。

---

## 178. Compression Precedes, Content Decides: Emergent Misalignment in Reward Models

**arXiv ID:** 2610.22805 | [PDF](https://arxiv.org/pdf/2610.22805v1)

**作者:** Dhruv Puri `[一作]` (Northeastern University), Nikhil Prakash `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在两个最新奖励模型上使用仅1000条针对有害偏好对的局部微调，展示了奖励模型会出现跨域的 emergent misalignment，并对其内部机制进行剖析。

**💡 创新点**

创新点在于首次证明奖励模型在极少量有害训练数据下会在未见域中大幅颠倒偏好，并将这一颠倒拆解为两种可分离的机制——在计分位置的几何压缩与内容特异的旋转。

**🔧 技术方法**

使用的技术包括 LoRA 微调、DPO 与 SFT 训练、Bradley–Terry 损失、对最终层计分位置和均值池化激活的角度度量以及奖励分布的几何分析。

**📊 数据集**

数据集为从八个咨询领域（职业、金融、家庭安全、法律、医疗、关系、体育、技术/安全）生成的合成偏好对，训练集采用 DeepSeek 生成，评估集采用 Claude 生成，另外还使用了公开的八个开放式问题集。

**📈 对比分析**

通过比较奖励模型与其基础语言模型在同一评估集上的偏好逆转比例与分数，奖励模型在七个保留域中达到 86–97% 的有害偏好，而语言模型仅上升至约 55–71%；同时奖励模型在 RewardBench‑2 上仅损失不到 5.5 分，显示整体能力几乎不受影响。

**⚠️ 局限性**

局限性包括：压缩不一定是颠倒的必要条件（未做因果验证）；未演示颠倒奖励模型如何破坏训练出的策略；实验仅在单一实验设置（1 epoch, 1000 例）下完成，缺乏剂量-响应曲线；数据集均为模型生成，未覆盖真实用户交互；模型来自同一实验室，结果可能不具普适性。

---

## 179. What Accuracy Hides In Learned Transformer Programs

**arXiv ID:** 2610.22891 | [PDF](https://arxiv.org/pdf/2610.22891v1)

**作者:** Jonathan Lee `[一作]` (Martian), Philip Quirke `[通讯]` (Martian)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估Transformer Programs（TP）模型的可解释性与准确性，通过完整答案检验、形式化验证和程序分析发现token准确率高但序列准确率低的问题，并在此基础上改进训练数据以提升模型性能。

**💡 创新点**

创新点在于：①使用程序提取后可读Python代码进行完全答案的形式化验证，揭示了token准确率误导的问题；②利用程序诊断结果指导训练数据增强（如添加大计数样本、平衡连乘子长度），显著提升序列准确率；③系统分析TP训练过程中向离散选择迁移导致的准确率下降和操作实现的层深/查找表限制。

**🔧 技术方法**

技术包括Transformer Program架构（限制的注意力、MLP、残差变量）、温度退火的软到硬决策、程序提取与可执行Python代码、Z3 SMT求解器的形式化验证，以及手写参考程序以评估表达能力。

**📊 数据集**

使用TP基准的九个算法任务（如计数、频率排序、算术运算等），在这些任务上构建训练和验证集，并对单个任务（计数、频率排序）进行详细程序分析。

**📈 对比分析**

通过与原始TP基准结果对比，token准确率从约75%到92%不等，序列准确率仅为3%至40%；在经过数据增强后，部分任务序列准确率提升至超过80%；同时，形式化验证发现了训练集内仍存在的错误，说明仅靠采样测试不足。整体性能在训练种子、模型规模和温度调度上存在显著差异。

**⚠️ 局限性**

主要限制包括：①验证和程序分析仅覆盖小范围输入域（受solver复杂度限制）；②在向离散选择迁移时易出现准确率下降；③参考实现往往需要大量层或大查找表，表明TP对复杂运算的表达有限；④训练结果高度依赖随机种子，稳定性不佳。

---

## 180. ToMStories: Interpreting Theory-of-Mind in Synthetic Narratives

**arXiv ID:** 2610.22632 | [PDF](https://arxiv.org/pdf/2610.22632v1)

**作者:** Melanie Subbiah `[一作]` (Columbia University), Kathleen McKeown `[通讯]` (Columbia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估一套名为ToMStories的合成叙事数据集，用于测试语言模型在隐式情绪、恐惧、动机和情感创伤等心理状态的推理与生成能力。

**💡 创新点**

发现语言模型倾向于利用共享的“指示短语”来推断心理状态，并且模型在隐藏情境下的推理性能与与数据生成器Claude所共享的指示短语重叠程度高度相关。

**🔧 技术方法**

采用大规模多语言预训练模型（如Llama、Qwen、Gemma、Phi、DeepSeek等）进行多项选择和开放式问答评测，使用线性探针、内部表示聚类、方向向量驱动（steering）等解释技术来分析模型内部机制。

**📊 数据集**

使用~6,000条合成短篇叙事片段（Implicit、Explicit、Conflicted三种变体）以及相应的多项选择和开放式问题，构建在Claude Sonnet 4.6生成的基准数据集。

**📈 对比分析**

在多项选择任务中，最大规模模型（如Llama-3.1‑70B）可达到或接近人类水平；在开放式问答任务中，模型表现略逊于人类（最高≈59% vs 64%人类），且对不同情感类型的准确率存在差异。

**⚠️ 局限性**

主要局限包括：数据仅由单一生成器Claude生成，可能导致模型对该生成器的惯例过度适应；数据来源于西方英语写作典型的心理学词典，缺乏跨文化和人类写作多样性；解释与驱动分析仅适用于开源模型，无法探究生成器内部机制。

---

## 181. Alert-List Concordance for Evaluating Imputation Strategies in ICU Sepsis Prediction

**arXiv ID:** 2610.22635 | [PDF](https://arxiv.org/pdf/2610.22635v1)

**作者:** Priyanka Siddappa `[一作]` (University of Massachusetts Lowell), Ruizhe Ma `[通讯]` (University of Massachusetts Lowell)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了一种新的评估方法——警报列表一致性（Alert‑List Concordance），用于比较不同缺失值插补策略在 ICU 败血症预测模型中的实际操作影响。

**💡 创新点**

创新点在于用 Jaccard 相似系数量化不同插补方法在固定警报率下产生的高危患者名单重叠度，揭示即使 AUROC 相近，警报列表也可能大不相同。

**🔧 技术方法**

采用多种插补技术（均值插补、前向填充、MICE、DAE、MAE、MIDA、SAITS、BRITS、GRU‑D）与固定的 XGBoost 预测模型，使用 Platt 校准、Bootstrap 置信区间和 Spearman/Kendall 相关系数等统计方法进行比较。

**📊 数据集**

使用 PhysioNet/CinC 2019 败血症挑战数据集（40,336 位 ICU 病人，71.9% 缺失率）作为实验数据。

**📈 对比分析**

在所有 45 对插补方案的 top‑10% 警报列表中，AUROC 差异 ≤0.011，但 Jaccard 交叉度在 0.518–0.750 之间；前向填充与其他方法的 Jaccard 仅 0.40–0.45，显示即便整体判别率相当，实际警报患者差异显著。

**⚠️ 局限性**

局限性包括：仅使用单一 ICU 数据集、采用行级实现的序列插补器未充分利用时间特征、未传播插补不确定性、未使用缺失指示器或时间自变量、仅评估单一学习器（XGBoost），以及未验证警报变更对临床结果或公平性的真实影响。

---

## 182. Explore Separately, Adopt Selectively: Execution-Guided Tool Use over Large Tool Libraries

**arXiv ID:** 2610.22688 | [PDF](https://arxiv.org/pdf/2610.22688v1)

**作者:** Anna Mazhar `[一作]` (Cornell University), Nikita Bhutani `[通讯]` (Megagon Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为BRIA的工具检索与使用框架，能在大型工具库中通过分支式探索和执行证据选择来解决工具不确定性问题。

**💡 创新点**

创新点在于：①在单一轨迹中将候选工具的调查隔离到独立分支；②通过执行证据进行选择性采纳；③结合需求导向的工具检索与分支策略，实现对工具库的自适应探索。

**🔧 技术方法**

核心技术包括：多模态检索（bi-encoder嵌入+余弦相似度）、分支式任务控制、执行证据对比与采纳机制、能力导向检索与迭代调用。

**📊 数据集**

使用了PlanBench‑XL和MCP‑Atlas两大工具使用基准，工具库合并后共约1,897个工具。

**📈 对比分析**

与One‑shot top‑k、MCP‑Zero、FitText、ToolGate等四种基线进行对比，BRIA在PlanBench‑XL上平均成功率提升约6.7%，在MCP‑Atlas上平均断言分数提升14.4%，并在保持更低输入token成本的同时达到最优准确-成本前沿。

**⚠️ 局限性**

局限包括：仅采用bi-encoder检索缺少重排序；仅支持非变异工具（读写分支隔离）；分支检索预算固定，未实现分支间自适应预算分配；实验规模受限于采样任务，需进一步验证在更多领域与工具生态的泛化。

---

## 183. Shift-Reliance Alignment: Ranking Vision-Language Models by the Cues They Trust

**arXiv ID:** 2610.22610 | [PDF](https://arxiv.org/pdf/2610.22610v1)

**作者:** Faisal Mohammad `[一作]` (University of Maryland, Baltimore County), Mohamed Younis `[通讯]` (University of Maryland, Baltimore County)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出Shift–Reliance Alignment（SRA）方法，用以在目标域下无需跑模型即可预测并选择最优的视觉语言模型（VLM）

**💡 创新点**

创新点在于：①用源域编辑图像量化模型对特定线索（如场景上下文、图像文字）的依赖；②结合该依赖与目标域中该线索误导比例（shift）得到评分；③构建CueDial基准，用于系统评估选择器在不同误导比例下的性能

**🔧 技术方法**

技术手段包括：源域图像编辑（擦除物体、叠加文字、贴纸攻击）、控制对照编辑、OCR和CLIP等检测器估计shift、内积得分公式SRA、统计学校正等

**📊 数据集**

使用的数据集包括COCO、OpenImages、Objects365、LVIS、AMBER、SCAM、DASH‑B等，涵盖从未编辑到编辑、可疑文字、贴纸攻击的多种测试集；训练与评估均基于这些数据集

**📈 对比分析**

与传统方法（源错误、POPE错误、标签重加权、无监督估计如ATC、DoC、nuclear norm等）相比，SRA在构造性、可解释性测试集上实现了最低的平均后悔值（worst‑case mean regret 低于 3 分），在大多数含误导比例高的目标域（如OpenImages、Objects365）选出的模型误差比源错误低 10%+，在COCO上与源错误持平；在CueDial实验中，SRA在文字叠加任务中将后悔降至 3 分以下，明显优于基线

**⚠️ 局限性**

局限性包括：①在COCO误导率低时SRA易提前切换模型且得分偏高；②对贴纸攻击的误导预测不佳；③假设线索影响可加性且校正假设检出率与真实相同；④只针对“是否存在”问题，未覆盖属性或计数等其他任务

---

## 184. On the tradeoff between robustness and geometric accuracy in adversarial training

**arXiv ID:** 2610.22955 | [PDF](https://arxiv.org/pdf/2610.22955v1)

**作者:** Rachel Morris `[一作]` (Concordia University), Kerrek Stinson `[通讯]` (University of Utah)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本研究对非参数二分类下的对抗训练问题，严格证明了鲁棒性与几何精度之间的最优量化估计，即对抗分类器相较于贝叶斯分类器的 Hausdorff 距离可被控制在 O(√ε) 或 O(ε) 级别（ε 为对抗预算），并给出了低光滑性和高光滑性两种情形的理论上限与可达性；

**💡 创新点**

创新点在于首次用几何测度（Hausdorff 距离）刻画鲁棒性与准确率的权衡，并通过构造能量竞争子集与边界阻挡机制，在高维非参数设定下实现了与低复杂度假设类不兼容结论相反的“可兼容”结论；

**🔧 技术方法**

主要技术包括变分分析、几何测度理论、非局部总变差与曲率流的等价性、能量竞争子集构造、对抗预算小的极限分析以及 Wasserstein 收敛率和概率集中估计；

**📊 数据集**

研究使用的是理论概率分布 μ 以及其经经验抽样得到的离散分布 ν_i，未涉及具体公开数据集；

**📈 对比分析**

由于本工作为理论分析，未进行实验或性能比较，主要通过定理证明展示在给定正则性与非退化条件下的误差上界；

**⚠️ 局限性**

局限性包括对贝叶斯分类器的局部光滑性和非退化性要求、对高维常数的多项式依赖、以及对对抗预算 ε 的小规模假设，且在实际实现中需要进一步验证与高维数据的匹配度。

---

## 185. ANCHOR: Audio-Visually Grounded Chain-of-Thought Reasoning Benchmark

**arXiv ID:** 2610.22964 | [PDF](https://arxiv.org/pdf/2610.22964v1)

**作者:** Joel Julin `[一作]` (Carnegie Mellon University), Laszlo A. Jeni `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ANCHOR 基准，评估多模态大语言模型在音视频问答中是否将推理过程与感知证据对齐。

**💡 创新点**

创新点在于要求模型生成带有时间戳、边界框和音频事件的 Grounded Chain‑of‑Thought，直接衡量推理与视觉音频证据的一致性。

**🔧 技术方法**

使用 SAM3 边界框生成、Gemini‑2.5‑Pro 生成伪 Grounded CoT、人工校正，并通过多模态输入和逻辑一致性判定构建评估框架。

**📊 数据集**

数据集来自九个公开视频问答库（AVQA、Ego4D、THUMOS‑14 等），共 2,319 视频、8,272 QA 对，含 224 视频人工精细标注。

**📈 对比分析**

通过多模型比较和基于提示的消融实验，发现 12 种最先进 MLLM 在答案准确度相近时，推理对齐存在显著差距；去除视频会导致 38–42% 准确率下降，去除边框和音频则下降 3–8%。

**⚠️ 局限性**

局限在于人工标注规模有限、伪 Grounded CoT 依赖模型生成、以及对长视频的处理仍不完整。

---

## 186. Towards Hyperparameter Transfer for Differentially Private Optimization

**arXiv ID:** 2610.22976 | [PDF](https://arxiv.org/pdf/2610.22976v1)

**作者:** Tianze Wang `[一作]` (Rutgers University), Linjun Zhang `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 DP‑μP 框架，解决在差分隐私训练中模型宽度不同步时的学习率迁移问题。

**💡 创新点**

创新点在于引入信噪比（SNR）条件与谱尺度公式，理论推导出在噪声占主导的 DP 环境下层级学习率的缩放规则，从而实现从小代理模型到大目标模型的零‑shot hyper‑parameter 迁移。

**🔧 技术方法**

主要技术包括 Maximal Update Parametrization、随机矩阵理论分析、DP‑SGD/DP‑Adam/DP‑Muon 等 DP 优化器、层级梯度裁剪与噪声调节，以及 SNR 条件的解析式求解。

**📊 数据集**

实验使用 CIFAR‑10/100 的 MLP 与 Vision Transformer，以及 GPT‑2 在 Shakespeare 字符级文本生成任务，验证 DP‑μP 在图像分类和文本生成两大任务上的有效性。

**📈 对比分析**

与标准参数化（SP）以及无 DP 或 DP 的不同优化器对比，DP‑μP 在训练 loss 对齐、测试准确率（或生成质量）方面至少保持不输，甚至优于单独调优的基线；同时零‑shot迁移显著降低计算成本（如 GPT‑2 训练从 30 min 降至 11 min）。

**⚠️ 局限性**

局限性：目前仅对宽度与批量大小的缩放给出理论，未覆盖深度缩放；分析局限于噪声占主导的 regime，未探讨信号与噪声相当的过渡区；在极大批量或更深网络时可能需要进一步调整或新的理论支撑。

---

## 187. VulnSAGE: Adjudicating SAST Reports by Connecting LLM Reasoning to Program Structure

**arXiv ID:** 2610.23035 | [PDF](https://arxiv.org/pdf/2610.23035v1)

**作者:** Jinbao Chen `[一作]` (University of Science and Technology of China), Yihong Wu `[通讯]` (University of Science and Technology of China)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了VulnSAGE，一个将大型语言模型（LLM）与程序结构相结合的 SAST 报告判定系统，能够从报告定位点回溯并获取决策所需的依赖和结构信息，并通过 CLAIM 语义事实记录模型的解释与程序关系，最终决定报告为 FP、TP 或不确定，并支持生成可运行的 PoC 复现。

**💡 创新点**

创新点包括：
- 报告中心切片（Report‑Centered Slicing）通过构造 Program Structure Evidence Graph (PSEG) 同时恢复依赖与解释性结构，给 LLM 提供精确上下文；
- CLAIM 作为中间表示，将模型的语义判断与可检查的程序关系绑定，形成结构性准入门槛，确保判定依据可被静态分析验证；
- 结合运行时复现机制，让静态判定能直接生成可验证的 PoC，提升判定可信度；
- 在多种真实项目与公开基准上全面评估，展示比仅基于 LLM 的判定器更高的 FP 拒绝率与 TP 保留率。

**🔧 技术方法**

核心技术包括：
- LLM（GPT‑5.5）在 LangGraph 框架下生成 CLAIM 事实与上下文请求；
- PSEG 作为统一图表示，融合 AST、CFG、数据流、调用绑定、符号、控制上下文等多种边；
- 结构性准入（CLAIM Admission）基于 PSEG 查询检查所有引用的程序关系；
- PoC Agent 自动构造输入/调用，配合运行时工具（如 AddressSanitizer）进行复现；
- 通过监督式工作流（Supervisor → MasterAgent → ContextAgent）实现报告解析、上下文获取、判定与复现的协同。

**📊 数据集**

实验使用的主要数据集有：
- Juliet 1.3（100 条已标注的 C/C++ 测试用例），
- D2A‑libtiff（48 条 Infer 报告，重新标注后 5/43 为 TP），
- SATE‑V（27 条 Cppcheck 报告，23/4 为 TP），
- ARVO Gold（12 条已知可复现漏洞，全部 TP），
- 开源扫描：对 200 个 GitHub 最受关注的 C/C++ 项目使用 Infer、Cppcheck、CSA、CodeQL、Semgrep 共 45,440 条警报。

**📈 对比分析**

对比方法：VulnSAGE 与 LLM4SA（同一 LLM 但仅提供基于提示的判定）。评估指标包括 FP 拒绝率、TP 保留率、决策覆盖率、运行时复现率。结果显示：
- 在所有标注集上，VulnSAGE 在 97 条 FP 中拒绝 89 条，保留 85/90 TP，决策覆盖率 ≥96%；
- 相比 LLM4SA，VulnSAGE 在真实项目上 FP 拒绝率为 89/97（vs 86/97），TP 保留率为 75/87（vs 77/87），而在运行时复现上，VulnSAGE 产生 25 次复现，LLM4SA 仅 3 次；
- 在公开扫描中，VulnSAGE 过滤掉 97.71% 警报，生成 104 个可复现 PoC，维护者确认 24/26（92.3%）并修复了 10 个之前未知漏洞。

**⚠️ 局限性**

局限性包括：
- CLAIM 的语义解释仍依赖 LLM，若模型误判属性（如容量、所有权）即使结构关系成立也会产生错误判定；
- PSEG 的边覆盖受前端解析能力限制，宏展开、间接调用、外部库行为等场景可能导致缺失关系，影响判定覆盖；
- 模型与提示配置不确定性导致不同运行产生不同上下文请求或解释，系统结果受训练集影响；
- 运行时复现受构建环境与输入可复现性的限制，部分 TP 由于缺少合适输入或运行环境而未能复现。

---

## 188. ElectroCalm: Investigating User Experience and Physiological Responses to Multi-Node Transcutaneous Electrical Stimulation for Stress Modulation

**arXiv ID:** 2610.22748 | [PDF](https://arxiv.org/pdf/2610.22748v1)

**作者:** Yoon Kyung Shon `[一作]` (University of Maryland), Pablo E. Paredes `[通讯]` (University of Maryland)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了一种8节点多点经皮电刺激（TES）在颈部和手臂的可用性和对认知压力任务的生理调节效果。

**💡 创新点**

创新点在于将多节点电刺激模式应用于颈部以实现潜在的迷走神经调节，同时比较敏感颈部与常见手臂位置的感受性和生理响应差异。

**🔧 技术方法**

采用可穿戴的低压电极阵列、双极脉冲 2 Hz 400 µs，结合EEG（α、β波）与心率变异性（RMSSD、HR）监测。

**📊 数据集**

数据集为两组24名受试者的实验记录，分别进行可用性评估与活跃/非活跃刺激的生理测量。

**📈 对比分析**

通过在同一受试者内对照实验，发现颈部活跃刺激显著提升α、β功率并增加RMSSD/降低HR，而手臂无显著差异；任务性能保持不变，证明TES不干扰认知。

**⚠️ 局限性**

局限包括刺激时间仅为2 min、缺乏直接迷走神经激活指标、仅测试认知压力情境、使用电压控制而非电流控制，且样本规模有限。

---

## 189. SemQBench: Benchmarking Decomposition-Aware Runtime Policies for Quantum-HPC Search

**arXiv ID:** 2610.22886 | [PDF](https://arxiv.org/pdf/2610.22886v1)

**作者:** Chandrachud Pati `[一作]` (Indian Institute of Science), Yogesh Simmhan `[通讯]` (Indian Institute of Science)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了 SemQBench 基准，评估了七种针对量子-HPC 搜索工作流的执行策略（包括全表面与组件可见、幅度放大、结果聚合、打包与重试策略），并在 IQM Garnet 20‑qubit 处理器上用 106 个电路、三份工作作业进行实验。

**💡 创新点**

创新点在于：①将组件可见性与放大策略拆分为可独立比较的七种策略；②提出“固定聚合”策略并证明其在同一电路下不劣于“联合投影”；③用基准量化不同策略在执行次数、设备活跃时间、作业数和模型成本上的权衡；④展示全表面放大在真实噪声下无显著收益，组件可见放大与固定聚合可显著降低执行次数。

**🔧 技术方法**

主要技术包括：量子电路构造（Qiskit/Truth‑table 逻辑）、幅度放大（Grover 迭代）、组件分解与验证器设计、固定聚合与联合投影策略、使用 IQM Garnet 的实际测量数据（设备时间、作业数）、统计模型（95% 完成置信度、模型成本 Q_ρ,σ）以及语义审核和运行时清单。

**📊 数据集**

使用 IQM Garnet 20‑qubit 处理器的实际实验数据；3 个工作作业共 106 个不同电路，覆盖三种工作负载（等宽 3+3、非等宽 2+3、三组件 2+2+2）并在两种映射下评估。

**📈 对比分析**

比较方法：对七种策略分别测量执行次数、活跃设备时间、作业数；使用 95% 完成概率阈值计算所需执行量；对同一电路在不同策略下的结果做直接对比；在 108 个不同执行/作业开销网格上做模型成本优化。性能表现为：全表面放大无显著提升；组件可见固定聚合将执行次数缩减 2.2–4.0 倍，且在 99/108 模型成本设置中被选为最优；联合投影总不优于固定聚合；分布式与包裹式打包对执行次数有差异；自适应重试可降低执行次数但增加作业数。

**⚠️ 局限性**

局限性：仅测试了可完全枚举的简化工作负载、零或一轮幅度放大、单一 IQM Garnet 设备；未包含更深更大规模电路、更多组件或其他量子后端；采用固定策略而非实时自适应调度；对噪声模型、作业排队时间、现场动态成本的考虑有限。

---

## 190. Micro-Localization: High-Precision Mobile Robot Localization Using Microscopic Ground Texture

**arXiv ID:** 2610.22563 | [PDF](https://arxiv.org/pdf/2610.22563v1)

**作者:** Aaron Wilhelm `[一作]` (Cornell University), Nils Napp `[通讯]` (Cornell University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发基于显微镜光学的地面纹理定位系统，实现在数十微米精度的全局定位；

**💡 创新点**

将显微镜放大率与几何敏感性相结合，提出虚拟针孔模型、自动聚焦方案，并在微米级实现全局定位；展示跨平台地图复用与热膨胀测量；

**🔧 技术方法**

使用显微镜光学提升几何敏感性，L-GROUT 的特征投票与RANSAC，SIFT+PCA+ANN检索，ECC直接图像对齐，机械精密 XY gantry 的校准与自动聚焦；

**📊 数据集**

三种自然表面纹理（打印木纹桌面、石材、混凝土）的显微图像地图与查询集，以及热膨胀实验的温度与位移序列；

**📈 对比分析**

与ECC直接相位对齐及 gantry 参考对比，三表面上 100% 成功率，翻译 RMSE 28–73 μm，方向误差 <0.1°；跨平台移动机器人误差 20–27 μm；热膨胀测量误差 2–3 μm RMSE；

**⚠️ 局限性**

参考系统误差（≈25–31 μm）难以与定位误差分离；高放大率导致景深浅、对焦敏感；对纹理弱的表面特征不足时会失败；需要显微镜硬件与精密机械平台，成本与规模受限。

---

## 191. ASR-Style Transcript Normalization Removes Useful Pragmatic Signal for Dialogue Act Classification

**arXiv ID:** 2610.22934 | [PDF](https://arxiv.org/pdf/2610.22934v1)

**作者:** Lucky Onyekwelu-Udoka `[一作]` (Iowa State University), Md Shafiqul Islam `[通讯]` (Iowa State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对ASR文本的不同规范化（去除填充停顿、标点、口语化形式等）对对话行为分类的影响进行跨模型、跨语料库的系统评估。

**💡 创新点**

首次通过九种规范化条件、六类分类器、三大对话语料，并结合多种随机种子、词典敏感性、标点感知TF‑IDF以及真实ASR评估，量化规范化对对话行为分类的多维影响，提出任务依赖的规范化策略。

**🔧 技术方法**

使用TF‑IDF+逻辑回归、RoBERTa/BERT/ALBERT/GPT‑2 Transformer微调、BiLSTM网络、punctuation‑aware tokenizer、cluster‑bootstrap统计、McNemar检验以及Whisper‑base.en ASR模拟等技术。

**📊 数据集**

采用Switchboard、ICSI MRDA、SLUE‑HVB三大对话语料库，分别覆盖电话会话、多方会议和客服脚本。

**📈 对比分析**

以宏F1为主评估指标，对比原始、填充词移除、重度规范化等条件，发现重度规范化在所有模型/语料下均显著降低F1；填充词移除效果因语料而异，词典限制后SLUE‑HVB下降基本消失；真实ASR下性能普遍下降，说明规范化策略需针对任务和语料定制。

**⚠️ 局限性**

仅限三类英语对话语料；SLUE‑HVB仅做单标签；语料拆分缺失会议标识；不同语料使用不同正则化实现；真实ASR评估在MRDA中混合识别误差与转录不匹配；未覆盖多语言、不同任务等。

---

## 192. Chance-Constrained Personalized LLM Routing with Online Preference Learning

**arXiv ID:** 2610.22695 | [PDF](https://arxiv.org/pdf/2610.22695v1)

**作者:** Ameya Anjarlekar `[一作]` (UIUC), R Srikant `[通讯]` (UIUC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在线个性化大语言模型（LLM）路由框架，在未知用户偏好下通过路由决策学习偏好，并在保持高概率满足质量要求的同时最小化推理成本。

**💡 创新点**

创新点在于：①将质量约束形式化为“机率约束”而非期望约束；②设计了基于GLM的UCB上界和虚拟满意度队列的双重（primal‑dual）算法；③在保证高概率质量的前提下实现了子线性成本回退。

**🔧 技术方法**

使用技术包括：上下文多臂赌博机（contextual bandit）模型、广义线性模型（GLM）与逻辑回归、UCB（上置信界）估计、虚拟队列（satisfaction‑deficit queue）和双重优化（primal‑dual）方法。

**📊 数据集**

实验数据集主要包括：合成数据（用于验证理论性质）以及真实LLM路由案例（使用WildBench v2 100条查询和五个可用LLM，构造六维质量向量并模拟满意度）。

**📈 对比分析**

与SLARouter、BaRP、期望约束的双重优化等现有路由策略比较，结果显示该方法既满足约束（约95%的试验满足高概率质量要求），又在成本上显著低于对比方法，且回退成本与回退成功率均可控。

**⚠️ 局限性**

局限性包括：①假设用户偏好是平稳且可由单一GLM参数描述；②需要存在已知满足率高于阈值的fallback模型；③在退化或非平稳场景下的鲁棒性和泛化性尚未得到充分验证。

---

## 193. alkhder at AlexandriaX-2026: Context-Aware English-to-Dialectal Arabic Dialogue Translation with QLoRA-Fine-Tuned NileChat-3B

**arXiv ID:** 2610.22525 | [PDF](https://arxiv.org/pdf/2610.22525v1)

**作者:** Hasan Alkhder `[一作]` (Sakarya University), Ahmet Zengin `[通讯]` (Sakarya University)

**通讯引用:** 1247 | [OpenAlex ID](https://openalex.org/A5014273833)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用 NileChat‑3B‑Base 通过 QLoRA 微调，构建单一适配器模型，实现对13种阿拉伯方言的上下文感知英语对话翻译。

**💡 创新点**

创新点：① 仅用一个适配器权重即可覆盖所有方言，支持零样本方言（利比亚、苏丹）② 将对话上下文、国家/方言标签、领域与角色信息集成到 prompt 里，系统能自动利用自身生成的上下文进行翻译；③ 在受限轨道内完成 GPU‑小时 <1，推理时间 <2 小时。

**🔧 技术方法**

技术：NL Chat‑3B‑Base + QLoRA（低秩适配器），4‑bit NF4 量化，黄金标注对话上下文、自然语言方言标签，贪心解码，单轮 prompt 生成。

**📊 数据集**

数据集：Alexandria 语料库，涵盖 13 国 11 种有训练数据的方言（共 78,730 条对话），使用 2,500 条/国的随机下采样；利比亚、苏丹仅在零样本评测。

**📈 对比分析**

比较方法：以国家宏平均 spBLEU 和 chrF++ 作为官方指标，系统在受限与非受限轨道均排名第六，spBLEU 20.36，chrF++ 36.64；与最强系统相比，spBLEU 仅保留 66.9%，chrF++ 80.7%。

**⚠️ 局限性**

局限：① 自生成上下文未提升性能，误译会累积；② 方言间代码切换覆盖不足，尤其是代码切换频繁的方言；③ 对话上下文仅限四轮，深层语境信息未充分利用；④ 仅使用单一 3B 模型，可能限制了语义表达与语法细节的捕获。

---

## 194. From Individual Songs to Music Corpora: Evaluating Explainable Harmonic Decision-Making on HookTheory

**arXiv ID:** 2610.22416 | [PDF](https://arxiv.org/pdf/2610.22416v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 195. Null Controls Are Necessary but Not Sufficient: Evaluating Acceptance Guards for Scientific Discovery and Interpretability

**arXiv ID:** 2610.22445 | [PDF](https://arxiv.org/pdf/2610.22445v1)

**作者:** Tianyu Wu `[一作]` (University of Illinois), Tianyu Wu `[通讯]` (University of Illinois)

**通讯引用:** 1407 | [OpenAlex ID](https://openalex.org/A5073713355)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了四种可执行的科学描述修正操作（L1‑L4），并在物理系统与神经网络可解释性实验中分别评估它们的描述性与结果性测试。

**💡 创新点**

首次揭示描述性接受测试与科学后果之间的差距，提出独立结果测试（NAR、PAR、CAR）评估标准，并证明仅靠描述性测试无法确保候选者真正具备预期的科学意义。

**🔧 技术方法**

使用闭合度、可观测性保持、线性CKA、结构一致性、干预/梯度扰动、稀疏自编码器特征富集等技术，对候选修正进行构造与检验。

**📊 数据集**

实验数据集包括：137个公开混沌系统基准、构造的慢快系统、晶格格子模型、ESM‑2蛋白质语言模型、Pythia语言模型、以及对应层的稀疏自编码器训练数据。

**📈 对比分析**

通过将描述性测试与结果性测试并行评估，发现例如常数映射虽然闭合度为零却被可观测性测试全部拒绝；结构一致性测试可误接受误匹配映射，但干预测试能完全纠正；稀疏自编码器的富集测试能排除大多数空白特征，但仅约45%通过干预验证，展示了两类测试的互补性。

**⚠️ 局限性**

局限性包括：null 套件有限、结果性测试灵敏度受限、线性映射难以恢复非线性减形、实验规模受模型参数与数据量限制、以及所用操作在开放式科学问题中的通用性尚未验证。

---

## 196. GeoReason: Bridging Logical Reasoning and Spatial Fidelity in Remote Sensing Segmentation

**arXiv ID:** 2610.23416 | [PDF](https://arxiv.org/pdf/2610.23416v1)

**作者:** Cheng Fan `[一作]` (City University of Hong Kong), Jianyuan Guo `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了GeoReason框架，用于遥感推理分割，解决了空间精度退化与目标误识别问题。

**💡 创新点**

创新点在于：①在提示词中预注入“segmentation token”作为空间锚（Anticipatory Prior）以抑制信息衰减；②结合位置感知链式思考（Location‑aware CoT）先行预测粗定位框；③提出Latent Patch‑to‑Token Alignment（PTA）损失，将token与图像补丁对齐以提升语义辨识。

**🔧 技术方法**

采用Qwen3VL+SAM2基础模型，LoRA微调，prompt‑side token、CoT、PTA损失与GLU投影等技术。

**📊 数据集**

在EarthReason（推理分割）和RRSIS‑D（参照分割）数据集上进行评测，并使用GeoChatInstruct、VRSBench等进行预训练。

**📈 对比分析**

在EarthReason上取得gIoU 78.61%/cIoU 80.60%，在RRSIS‑D上gIoU 68.36%/cIoU 79.87%，均超越现有MLLM与传统方法，显著提升2–6个百分点。

**⚠️ 局限性**

局限性包括：对极长推理链仍有潜在信息衰减；对非常小目标的精细分割效果仍有限；依赖SAM2解码器和特定token设计，跨域或非遥感场景的泛化尚未验证。

---

## 197. A Plausible Explanation Is Enough to Mislead Multimodal Judges

**arXiv ID:** 2610.22631 | [PDF](https://arxiv.org/pdf/2610.22631v1)

**作者:** Anglin Cai `[一作]` (Northwestern University), Yuchen Yang `[通讯]` (Pennsylvania State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个视觉问答（VQA）风格的基准，固定候选答案，只变化其伴随的解释，评估多模态大型语言模型（MLLM）作为判定者时可解释性对判断结果的影响；

**💡 创新点**

提出并验证了“可解释性捕获（plausibility capture）”这一新风险——合理的、看似合理的解释会使判定者更倾向于接受无论正误的答案，并用信号检测理论区分判定阈值与辨别能力的变化；

**🔧 技术方法**

采用了多种解释生成方式（链式思考CoT、普通正当化Plain、说服性Persuasion）、信号检测理论（SDT）分析、以及多模型、多领域、多指标的评估框架；

**📊 数据集**

使用了医学VQA（VQA‑RAD、PMC‑VQA）、图表推理（ChartQA、ChartQAPro）、幻觉检测（HaloQuest、PRE‑HAL）等公开数据集，并构造了诱导错误集和外域视觉 grounding 集；

**📈 对比分析**

对14个来自不同模型族（8开源、6专有）的判定者进行实验，比较了无解释基线与三种可解释性条件下的错误拒绝率（CRR）和正确拒绝率（IRR）。结果显示，普通解释平均降低CRR约13个百分点，类似于专门说服性解释；判定者的准确率仅下降不到2个百分点，显示可解释性误判掩盖了错误检测能力；

**⚠️ 局限性**

局限性包括：实验只在受控环境下进行，未涵盖所有实际判定场景；效果高度依赖模型与解释生成器，部分模型表现不一致；缓解方法（保守指令、先自检）仅对部分模型有效，需进一步模型特定验证；

---

## 198. AgentLoupe: Kernel-Resident Semantic Causal Labels for AI Agent Observability

**arXiv ID:** 2610.23233 | [PDF](https://arxiv.org/pdf/2610.23233v1)

**作者:** Yifei Bao `[一作]` (Southeast University), Peng Jiang `[通讯]` (Southeast University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 AgentLoupe，一种基于 eBPF 的内核级观测工具，能够在 LLM 交互时为所有相关系统资源嵌入语义链标签，并在进程、文件和套接字间传播，实现在跨进程、跨时间、跨会话的情况下完整追踪 LLM 与系统动作之间的因果链。

**💡 创新点**

核心创新点在于：1）将语义链标签直接写入内核资源，使得标签在资源生命周期内持久；2）通过 eBPF 实现标签在进程创建、文件写入/打开、套接字操作等显式交互中的传播，突破了 AgentSight 进程树限定和时间窗口匹配的局限；3）在用户空间重构完整 HTTP/Server‑Sent Events 负载，避免 eBPF 栈空间限制导致的信息丢失。

**🔧 技术方法**

使用技术包括：eBPF uprobes/kprobes 捕获 TLS 明文，基于关键词扫描识别 LLM 请求/响应；eBPF maps 存储并传播链标签；内核与用户空间使用 perf 及 ring 缓冲区传输记录；用户空间重构 HTTP/JSON、聚合链记录并写入 WAL‑mode SQLite；前端采用 FastAPI + Next.js 展示多视图；LLM 风险分析器用于生成威胁评估。

**📊 数据集**

实验基于 Claude Code 2.1.150 与 OpenAI‑GPT‑4o‑1106‑preview LLM，运行在 Ubuntu 26.04 (kernel 7.0) 上，执行三类工作负载（代码库阅读、CPU‑frequency 监控编写、源代码编译）以及两组案例研究（延迟 cron 反弹 shell、跨会话指令注入）。未使用公开数据集，而是构造了上述攻击情景。

**📈 对比分析**

与无观测、AgentSight 以及仅做捕获不做传播的 AgentLoupe 进行对比，平均整体运行时开销小于 8%（最坏 8.0%），捕获开销接近 AgentSight；传播开销为 1–8% 之间。通过案例研究验证，AgentLoupe 能完整回溯跨进程和跨时间的攻击链，而 AgentSight 在这两类场景下无法关联。

**⚠️ 局限性**

限制包括：1）标签传播的过度 taint 与漏判率未量化，可能导致误关联或遗漏；2）实验范围仅覆盖 Claude Code 与构造的攻击场景，缺乏对真实大规模代理攻击、不同 LLM 与框架的验证；3）未提供链关联准确率与 LLM 风险分析器误差的基准评估。

---

## 199. Area Theorems for Quantum Error Correction

**arXiv ID:** 2610.22710 | [PDF](https://arxiv.org/pdf/2610.22710v1)

**作者:** Avijit Mandal `[一作]` (Duke University), Henry D. Pfister `[通讯]` (Duke University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了平面表面码在量子擦除信道上的出口函数（EXIT functions），并通过数值模拟得到不同码距（d=51、101、151、251）下的归一化出口曲线；

**💡 创新点**

首次将平面表面码的出口函数与信道容量对齐，提出了通过归一化每个组件面积（k/n）来统一不同码距的性能比较方法；

**🔧 技术方法**

采用蒙特卡洛随机擦除模拟、信道容量理论与EXIT曲线分析相结合的技术框架；

**📊 数据集**

使用合成的量子擦除错误样本（随机生成的擦除位置），对多种码距的平面表面码进行实验；

**📈 对比分析**

通过将各码距的EXIT曲线按归一化因子绘制在同一坐标系下，比较其与容量上限（ϵ=1/2）的接近程度，结果显示大码距时曲线与容量上限趋于一致，阈值接近理论极限；

**⚠️ 局限性**

仅考虑了平面表面码且仅在擦除信道上进行评估，未涉及更通用的错误模型或非平面码结构，且数值结果依赖于采样精度，存在统计误差。

---

## 200. Refusal Is Not Robustness: Measuring Harmful Reachability in Autoregressive Language Models

**arXiv ID:** 2610.22754 | [PDF](https://arxiv.org/pdf/2610.22754v1)

**作者:** Parv Kapoor `[一作]` (Carnegie Mellon University), Eunsuk Kang `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了基于“有界继续分析”（Bounded Continuation Analysis, BCA）的新评估方法，用来量化语言模型在对抗性 jailbreak 攻击下的隐藏有害风险；同时定义了“生成攻击易感性”（Generative‑Attack Susceptibility, GAS）以跟踪攻击者预算增长时的风险演变，并将其与传统攻击成功率（ASR）相结合，构建更细粒度的安全评估框架。

**💡 创新点**

创新点在于：①引入有界继续树和概率模型检验，将语言模型的生成过程视为有限马尔可夫过程，直接计算在有限生成窗口内可达的有害概率；②将此概率量化作为新指标（BCA、GAS），能揭示传统 ASR 无法捕捉的分布不稳定性；③将 GAS 作为红队搜索的引导信号，证明与结果导向搜索互补，显著提升对抗覆盖率。

**🔧 技术方法**

主要技术包括：有界继续树构造（α‑k 约束）、概率模型检验（PCTL 计算）、逆向归纳求解、生成攻击易感性定义、以及基于 BCA 的自适应红队搜索（改造 PAIR、TAP）。

**📊 数据集**

使用公开基准 JailbreakBench（100 种恶意行为）与四个开源指令调优模型（Qwen‑1.5B/7B、Mistral‑7B、Llama‑3.1‑8B），并结合 HarmBench‑Mistral‑7B 等判别器进行安全评估。

**📈 对比分析**

与传统 ASR 的比较显示：在所有模型上 ASR 均为 0%，但 GAS@0 与 GAS@18 能区分两位数级差异；GAS 与实测 ASR 的 Spearman ρ 超过 0.7，且在 BCA‑指导搜索与判别器指导搜索的组合下，成功率提升至 70%‑80%，显著优于单一搜索策略。

**⚠️ 局限性**

局限性包括：①计算成本高，需对高维词表做有界截断，可能低估实际风险；②结果依赖特定判别器检查点，模型更新后数值可能变化；③目前仅在开放权重模型与 JailbreakBench 上验证，未覆盖闭源或非文本生成任务；④对极端长文本或多轮对话的生成分布建模尚不完善。

---

## 201. Think More or Read More? A Token-Budget Theory of Reasoning with Shared Context

**arXiv ID:** 2610.23246 | [PDF](https://arxiv.org/pdf/2610.23246v1)

**作者:** Manh Nguyen `[一作]` (Deakin University), Hung Le `[通讯]` (Deakin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种令读写共享同一 token 窗口的 token‑budget 理论，用以预测多代理辩论的最佳团队规模并指导自我修订。

**💡 创新点**

创新点在于：① 用闭式公式将每个代理的“思考能力”与其可用思考 token 关联；② 在此基础上构建可校准的 Condorcet 集合模型，得到有效样本数与边缘效应；③ 通过理论推导得到团队规模内部最优点以及何时“读”不值得的阈值；④ 将方法扩展到异构模型集合。

**🔧 技术方法**

技术包括：token‑budget 约束建模、思考增益函数（Hill‑sigmoid）拟合、Condorcet 纠错聚合、有效样本数与协方差结构、闭式最优规模求解、与自我修订比较的实验设计。

**📊 数据集**

实验使用 5 种开放权重模型（Llama‑3.2‑3B、Qwen2.5‑7B、Falcon3‑7B、Gemma‑2‑9B、Gemma‑3‑12B）以及四个推理基准（GSM8K、MATH500、MMLU‑Pro、CSQA），覆盖 3–12B 参数规模。

**📈 对比分析**

与多种对比方法（无校准、单一代理、Debate‑calibrated、Successive Halving、Quadratic Surface、学习型预测、Star Exponent、Difficulty Mixture）在相同 token 花费下比较。TB‑MAD 在 8 种 token 预算下平均 regret 仅 1.40%，低于所有对手；在紧凑窗口下尤其显著；与自我一致性比较，TB‑MAD 在 93% 的配置下与之相当或优于，且在宽窗口时差距消失。

**⚠️ 局限性**

局限性包括：需要先做单代理“思考能力”探测；对读写 token 价格假设相同或需额外校准；模型假设等相关性和错误分布为常数；在极宽窗口或非常大团队规模时，理论预测可能失效；对异构模型的参数估计仍相对粗略。

---

## 202. From Events to Reasoning: Hierarchical Rule Composition over Explicit States

**arXiv ID:** 2610.22403 | [PDF](https://arxiv.org/pdf/2610.22403v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 203. RTISSLAM-3D: Full 6-DoF SLAM With an In-Air 3D Imaging Sonar

**arXiv ID:** 2610.22552 | [PDF](https://arxiv.org/pdf/2610.22552v1)

**作者:** Jan Steckel `[一作]` (University of Antwerp), Wouter Jansen `[通讯]` (University of Antwerp)

**通讯引用:** 750 | [OpenAlex ID](https://openalex.org/A5053059385)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了 RTISSLAM-3D，一套仅用单机、IMU 与车轮编码器即可完成的全 6-DoF 在空 3D 成像声纳 SLAM 系统。

**💡 创新点**

创新点在于将球面 CFAR 检测、三维声学里程计、极坐标子图描述符以及 4‑DoF 子图配准集成到完整 3D SLAM，并通过重力方向的姿态因子使地图保持水平。

**🔧 技术方法**

技术包括 DMAS+共振因子成像、球面 CFAR 检测、基于声波束不确定性的点对分布注册、极坐标子图描述符、互补滤波姿态估计、iSAM2 因子图优化以及体素占用地图。

**📊 数据集**

使用真实森林环境中约 560 m 的采集数据，包含 eRTIS 成像声纳、Ouster OS1 64‑束激光雷达、IMU 与车轮编码器，并以对齐后的 3D Lidar SLAM 轨迹作为参考。

**📈 对比分析**

采用绝对轨迹误差（ATE）、相对姿态误差（RPE）、循环闭合误差、以及与 Lidar 参考的地图精度（精确度/召回率）进行比较；结果显示水平 ATE 从 1.74 降至 0.28，竖直误差 0.12，地图精度 94.6% 与 Lidar 参考的 96% 非常接近。

**⚠️ 局限性**

局限性包括仅在单一记录上评估、参考 Lidar SLAM 的依赖、循环闭合仅在 5 m 位置门限内搜索、IMU 仅提供重力方向、对竖直先验假设、低召回率以及目前仅离线 Python 实现，尚未实现实时 ROS 2 版本。

---

## 204. Mitigating Demographic Disparities in Explanation-Human Alignment via Rationale Regularization

**arXiv ID:** 2610.22586 | [PDF](https://arxiv.org/pdf/2610.22586v1)

**作者:** Suryansh Talukdar `[一作]` (Purdue University Fort Wayne), Jonathan Rusert `[通讯]` (Purdue University Fort Wayne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究仇恨言论检测中的解释公平性，提出 Pseudo‑EFG 评估指标并构建 Fair‑REF 训练框架，以提升不同人群之间的解释一致性。

**💡 创新点**

创新点在于首次将解释公平性量化为 worst‑case Plausibility‑EFG 指标，并通过对抗去偏与理据监督的联合训练证明解释公平性与预测公平性是相互独立的。

**🔧 技术方法**

采用 BERT、RoBERTa 及对抗去偏的 ADV‑RoBERTa 模型，并使用 Integrated Gradients 与 SHAP 生成解释；训练目标包括任务损失、理据损失与对抗正则化。

**📊 数据集**

实验基于 HateXplain 数据集，该数据集包含 20,148 条社交媒体帖子、三分类标签、token 级理据以及 8 个人口统计身份。

**📈 对比分析**

与 BERT/RoBERTa 的对比表明，ADV‑RoBERTa 在准确率、宏 F1 以及 Plausibility‑EFG 上均优于基线，尤其在解释公平性方面显著下降了 0.03‑0.05 的 worst‑case 差距。

**⚠️ 局限性**

局限性包括：1) 对抗正则化强度过高时会轻微降低预测性能；2) 评估依赖理据注释，缺少无监督解释公平性方法；3) 目前仅针对 HateXplain，缺乏跨域验证。

---

## 205. Prism Attention: Proposal-Refined Index Sharing Mechanism for Efficient LLMs Inference

**arXiv ID:** 2610.22985 | [PDF](https://arxiv.org/pdf/2610.22985v1)

**作者:** Zhenxu Tian `[一作]` (Soochow University), Min Zhang `[通讯]` (Soochow University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为Prism Attention的稀疏注意力框架，用于高效长上下文的语言模型推理；

**💡 创新点**

创新点在于将跨层索引共享从严格复制转为“提议-细化”两阶段机制，先在 anchor 层生成扩展的候选池，再在后续层在此池内轻量化重新选择 top‑κ，兼顾层间适应性与计算效率；

**🔧 技术方法**

采用了基于 INT2 量化的候选生成、组查询注意力（GQA）等技术，实现低位宽的候选打分；

**📊 数据集**

在 Qwen3‑4B/8B 语言模型上，使用 LongBench（长文本理解）和 AIME、MATH500、GPQA‑Diamond 等推理/推理式算子数据集进行评估；

**📈 对比分析**

与全注意力及四大稀疏基线（Quest、LessIsMore、Seer‑R、IndexCache）比较，Prism 在 reasoning 与 long‑context 任务上均取得最高平均准确率，同时在 128K 上实现 3.4× 的端到端解码速度提升；

**⚠️ 局限性**

局限性包括对跨层索引共享的依赖仍需要合理的 anchor‑layer 配置，对候选池过大可能引入噪声，且在极短上下文或极大模型规模下的泛化性尚待进一步验证。

---

## 206. Who Pays for Efficient Inference? Confidence-Based Small-to-Large LLM Routing Across Tasks and Languages

**arXiv ID:** 2610.23249 | [PDF](https://arxiv.org/pdf/2610.23249v1)

**作者:** Md Sabbir Hossen `[一作]` (Bangladesh University), Md Shohel Rana `[通讯]` (Georgia Southern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了小模型到大模型的路由，通过小模型置信度决定是否升级以降低推理成本。

**💡 创新点**

发现置信度路由在阅读理解和主题分类任务上有效，但在多语言推理任务上表现不佳，并提出语言不均衡的路由差距问题。

**🔧 技术方法**

使用置信度边际（最高两项概率差）作为路由指标，比较了全局置信度、等配额置信度、随机路由和上界路由。

**📊 数据集**

在四种语言（英语、孟加拉语、印地语、乌尔都语）上评估了三大模型家族（Qwen3、Gemma3、Llama3.2）在Belebele、MMLU-ProX-Lite、SIB-200三种多选任务。

**📈 对比分析**

在50%升级预算下，全局置信度路由在Belebele和SIB-200上相较随机提升1.4-5.6个百分点，MMLU-ProX-Lite则无显著优势；等配额路由消除语言升级差距，整体准确率差异不足0.5个百分点。

**⚠️ 局限性**

局限性包括仅测试1B-4B开源模型、四种语言、三项多选基准、置信度仅为概率边际、评估为离线对比，未涵盖大模型、Mixture-of-Experts、自由生成任务及批处理部署。

---

## 207. ParetoChat: An LLM-Based Interface for Interactive Multi-Objective Optimization

**arXiv ID:** 2610.23062 | [PDF](https://arxiv.org/pdf/2610.23062v1)

**作者:** Gokhan Ceyhan `[一作]` `[通讯]` (Just Eat Takeaway.com), Gokhan Ceyhan (Just Eat Takeaway.com)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 ParetoChat——一种基于大型语言模型的专家代理，用来在多目标优化中以自然语言交互方式捕捉决策者偏好，并将其转换为求解器可接受的标量化参数。

**💡 创新点**

创新点包括：①将自然语言偏好映射到加权和、参考点、ε‑约束等多种标量化方法，避免传统方法要求决策者提供精确数值；②采用多代理架构将意图识别、命令路由、MCDM 专家、结果展示、反馈处理和会话总结等功能分离，提升系统可扩展性与可调试性；③构建人工决策者（AIDM）评估框架，将自然语言交互与隐藏价值函数相结合，对比不同交互式方法的效果。

**🔧 技术方法**

技术实现基于 Gemini 2.5 Pro 语言模型、LangGraph 多代理框架、Gurobi 12 作为求解器，支持的标量化方法包括加权和、词典式权重、参考点、ε‑约束、lexicographic 等；系统通过 LLM 调用和工具调用实现对决策者输入的解析与执行。

**📊 数据集**

使用的数据集包括：多目标二进制背包（2OBKP、3OBKP、4OBKP）和双目标集合划分（BOSPA）实例；以及 Just Eat Takeaway 的历史快递订单分配案例，用于在真实业务场景中演示系统。

**📈 对比分析**

方法比较：与 DESDEO 包中的 Reference‑Point Method (RPM) 和 NIMBUS 进行对比。评价指标为：解评估次数、相对价值缺口、排名分数、人工DM时间与求解器时间。结果显示：ParetoChat 在评估次数和 DM 时间上明显更低，价值缺口与排名与 RPM 相近或略差，在简单实例（2OBKP）中可达到与 RPM 相当甚至更优的最优解；在更大问题或非线性价值函数下性能有所下降，但仍保持竞争力。

**⚠️ 局限性**

限制：①需要求解器支持多种标量化方法，无法直接处理自定义约束或隐藏目标；②偏好解释存在误差，需人工校正；③在高维目标或复杂非线性价值函数时，交互效率和解质量下降；④运行时 LLM 调用导致较高计算开销；⑤未实现多方决策或隐藏目标检测等功能。

---

## 208. AlphaBet: Reinforcement Learning for LLM Event Forecasting Agents

**arXiv ID:** 2610.22684 | [PDF](https://arxiv.org/pdf/2610.22684v1)

**作者:** Liqin Ye `[一作]` (Georgia Institute of Technology), Chao Zhang `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了一种基于结果的强化学习框架，让单一LLM代理在多个预测日期主动检索并推理证据，生成概率预测。

**💡 创新点**

将多日期监督与证据获取结合，分解预测误差为信息获取与概率报告两块；剔除GRPO标准化导致的概率偏差，并提供理论与实验验证。

**🔧 技术方法**

采用Agentic RL（GRPO）训练LLM工具调用循环，使用严格正确的评分规则（Brier、对数）作为奖励，构建检索、阅读、计算工具链并对GRPO标准化进行分析。

**📊 数据集**

使用Forecast-Dojo（训练/验证/测试）以及FutureX评估集，并在时间索引的CC-News新闻语料上进行检索。

**📈 对比分析**

与无训练、Uniform、BLF、OpenForecaster等基线以及Claude Sonnet、Qwen3.5等更大模型对比；在Forecast-Dojo上Brier从0.731降到0.626，准确率从41.6%提升到50.5%；在FutureX准确率从51.3%提升到57.9%，Brier从0.612降到0.528。

**⚠️ 局限性**

未实现跨日期信用分配导致记忆模式不优；仅在新闻文本环境实验，未检验对其他信息源的泛化；模型规模固定，未探究更大模型效果。

---

## 209. Towards Adaptive and Real-Survey-Aligned Simulated Urban Mobility Behaviour of Generative Agents

**arXiv ID:** 2610.22407 | [PDF](https://arxiv.org/pdf/2610.22407v1)

**作者:** Yves Bru `[一作]` (Université Toulouse Capitole), Benoit Gaudou `[通讯]` (Université Toulouse Capitole)

**通讯引用:** 2259 | [OpenAlex ID](https://openalex.org/A5017490856)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个基于Agent的Toulouse市日常多模式出行仿真框架，并将15种决策器（基线、基于调查的离散选择与机器学习模型、三种LLM及一个typed分类器）与真实家庭出行调查数据进行对比评估。

**💡 创新点**

创新点在于：①首次将生成式代理（GA）与传统模型并行测试，验证其在模式分布与个体决策上的表现；②提出用自然语言提示进行Prompt调优的循环，验证Prompt文本对GA行为的可控影响；③设计了“未预料事件”评估框架，展示GA在无法用结构变量编码的突发事件中的适应能力。

**🔧 技术方法**

主要技术包括：GAMA仿真平台与OpenTripPlanner路由器集成；LLM（Gemini 3.1/3.5、Mistral Large）与typed分类器Jev-1.13作为决策模块；Prompt调优循环（基于文本阈值评估改进）；基于Jensen–Shannon Divergence、Earth Mover's Distance的分布匹配指标；以及基于概率质量的个体决策准确率评估。

**📊 数据集**

使用了2023年Toulouse地区的全国家庭出行调查数据（约10,783户、15,775人次），以及通过eqasim合成的1,000名虚拟人物样本。

**📈 对比分析**

比较方法：首先在群体层面计算复合分数（整体JSD + 各分层JSD/EMD），其次在个体层面评估每次旅行的概率质量准确率与交叉熵。结果显示：最优的调优LLM在模式分布层面接近甚至略高于传统模型（复合分数≈4.8 vs 3.4–4.4），但在个体旅行层面准确率仅为约68%（LightGBM为72%），提示在模式匹配上可达到但个体决策仍有差距；在未预料事件测试中，GA能显著改变决策并产生持续影响，传统模型无法做到。

**⚠️ 局限性**

局限性包括：①GA在个体层面的决策准确率仍落后于最优机器学习模型；②Prompt调优依赖调查数据，缺乏完全独立性；③LLM调用成本高（例如Gemini 3.5每日约$50），对大规模城市仿真不具备经济性；④未在多城市或更大规模数据集上验证泛化能力。

---

## 210. Designing for Imperfect Prompts: A Simulation-Based Evaluation of an LLM-Based Enterprise Data Assistant Using Knowledge Graphs

**arXiv ID:** 2610.23304 | [PDF](https://arxiv.org/pdf/2610.23304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 211. FlashReFocus: Interactive Image Refocusing in Seconds

**arXiv ID:** 2610.23297 | [PDF](https://arxiv.org/pdf/2610.23297v1)

**作者:** Ching-Heng Cheng `[一作]` (National Cheng Kung University), Chih-Chung Hsu `[通讯]` (National Cheng Kung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种两阶段交互式照片重聚焦框架，先用一次扩散模型恢复全景聚焦图像，再用轻量化渲染网络快速生成任意焦点和模糊强度的Bokeh图像。

**💡 创新点**

核心创新在于“先恢复后渲染”的思路：只需一次扩散推理；Stage I使用像素+细节LoRA和VAE解码器细化器；Stage II采用显式深度+模糊偏置注意力的轻量化网络，使编辑仅影响模糊且保持对焦区域不变。

**🔧 技术方法**

技术手段包括：一阶扩散恢复（Stable Diffusion U‑Net+LoRA）、VAE解码器细化、深度估计（Depth Anything V2）、显式深度+焦点+强度的控制信号、Defocus‑Bias注意力、NFAN编码解码网络、Synthetic‑to‑Real两阶段训练。

**📊 数据集**

训练数据包括：DPDD、RealBokeh、LFDOF、DIV2K（合成模糊）、Hypersim（synthetic RGB‑D）、EBB!、新收集的3CReal三相机数据集（89对）。

**📈 对比分析**

与DiffCamera、GenRefocus、MagicBokeh、AnyBokeh等基线比较，FlashReFocus在EBB!和3CReal上取得最佳MS‑SSIM、LPIPS、DISTS等感知指标，并且在单次编辑时仅需约0.5秒、比GenRefocus快≈462倍，整体耗时仅2.5秒完成一次全流程。

**⚠️ 局限性**

局限性在于依赖单目深度估计，深度误差会导致模糊错位；无法恢复遮挡背景；尚未扩展到视频或模型蒸馏。

---

## 212. Path to Prior-Data Fitted Relational Models via Learned Latent Multi-table Representations

**arXiv ID:** 2610.23073 | [PDF](https://arxiv.org/pdf/2610.23073v1)

**作者:** Mohammad Sadeq Abolhasani `[一作]` (SAP Labs), Viswanath Ganapathy `[通讯]` (SAP Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用PFN风格的合成预训练对基于消息传递的关系模型Griffin进行编码器训练，并用冻结的上下文预测器在16个真实关系分类任务上评估其泛化能力。

**💡 创新点**

首次验证合成任务预训练与课程学习能够显著提升图结构关系模型的迁移性能，并揭示基于固定schema的语义预训练随后扩展到结构多样的无schema任务的有效性。

**🔧 技术方法**

采用Griffin（异构消息传递网络）作为编码器，冻结的上下文预测器（In-context Predictor）作为头部，结合PFN的先验任务采样和PluRel生成的合成数据库。

**📊 数据集**

使用两大合成语料库：固定SAP销售（SALT）schema的500个数据库（含7类分类任务）与多样化无schema的9,143个数据库，评估基准为16个来自RelBench与4DBInfer的真实分类任务。

**📈 对比分析**

与未预训练编码器、原始特征以及仅用真实数据预训练的基线对比，结果显示纯合成预训练的平均ROC‑AUC为0.648，追加合成阶段后可达0.672，明显优于仅用真实数据或单一语料的表现。

**⚠️ 局限性**

研究仅涵盖分类任务，未探究回归；课程空间有限，仅测试了从语义到结构的单向递进；未实现编码器与头部的端到端联合训练。

---

## 213. EgoP$^{2}$C: Internalizing Plan-Perceive-Compose for Spatial Reasoning in Egocentric Video

**arXiv ID:** 2610.22681 | [PDF](https://arxiv.org/pdf/2610.22681v1)

**作者:** Chaoyu Li `[一作]` (Arizona State University), Pooyan Fazli `[通讯]` (Arizona State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Egocentric Plan‑Perceive‑Compose框架，将规划、感知和组合功能集成到单一视觉语言模型中，以实现基于视频的空间推理。

**💡 创新点**

创新点在于使用Dual‑LoRA教师训练专门的规划与感知策略，再通过稀疏 MoE 初始化和段级 MOPD 让学生在单一前向推理中完成多轮子问题提问与答案合成。

**🔧 技术方法**

主要技术包括 LoRA 适配器、稀疏 MoE（Sparse Upcycling）、梯度增强强化学习（GRPO）以及多教师在线蒸馏（MOPD）等。

**📊 数据集**

数据集涵盖 ADT、Ego‑Exo4D、VSI‑590K、SPAR‑7M 与 Ego4D 等大规模 egocentric 视频与空间标签集，用于教师训练与学生蒸馏。

**📈 对比分析**

在 VSI‑Bench、EgoProx 与 R3D‑Bench 三大 egocentric 空间推理基准上，学生模型相较于基础模型平均提升约 9–10%（如 Qwen3‑VL‑8B 提升 9.2pp、SenseNova‑SI 7.5pp），同时在 FLOPs 降低约 66% 与推理时延减少 22–33%。

**⚠️ 局限性**

局限性包括：需要额外的教师预训练与多阶段蒸馏，训练过程计算成本高；对子问题生成质量高度依赖，若生成不准会导致错误推理；目前仅针对 egocentric 视频场景，对更长或更复杂视频的自适应帧选取和推理成本平衡尚未深入探究。

---

## 214. StoBasket: Joint Upsell and Cross-Sell Optimization for Contribution Margin

**arXiv ID:** 2610.22768 | [PDF](https://arxiv.org/pdf/2610.22768v1)

**作者:** Xin Heng `[一作]` `[通讯]` (Tote AI), Xin Heng (Tote AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

针对购物车内的促销决策，提出并求解联合随机篮子编辑问题，最大化预期贡献利润，兼顾替换式升价和追加式交叉销售、展示成本及客户响应的耦合效应。

**💡 创新点**

首创将升价和交叉销售的响应耦合纳入决策目标，并给出一种压缩的响应-价值映射结构与能够在结构化类中得到全局最优解的精确优化算法。

**🔧 技术方法**

采用潜在类别/逻辑回归的概率响应模型、结构化响应时刻、超平面排列枚举、树分解动态规划与精确求解器，实现多渠道耦合的期望利润优化。

**📊 数据集**

使用五个公开零售数据集：Complete Journey、Ta Feng、Instacart、Online Retail II 与 Carbo‑Loading。

**📈 对比分析**

通过与五种启发式策略（独立、先升价后交叉、先交叉后升价、交替最优、贪心）以及基准模型对比，精确策略在大多数市场中仅落后 0.5% 以内，且在“仅束缚”情形下显著优于启发式；平均求解时间 0.13 ms，99% 分位 111 ms。

**⚠️ 局限性**

精确求解仅在响应时刻一致、交互宽度有限且预算（L≤4）满足时成立；模型不完整或更复杂交互、预算扩展至更大规模时仍可能产生结构性误差，且对响应模型的误差敏感。

---

## 215. Optimist Deep Research: An Iterative, State-Centric Agent for Autonomous Literature Research

**arXiv ID:** 2610.23299 | [PDF](https://arxiv.org/pdf/2610.23299v1)

**作者:** Gihan Panapitiya `[一作]` `[通讯]`, Gihan Panapitiya

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一个基于状态的检索增强文献综述代理 Optimist Deep Research，支持迭代式证据获取、批判与报告生成。

**💡 创新点**

创新点在于：①把研究计划视为可变状态输出；②使用结构化证据存储和保留来源；③引入四层去重、确定性排序与分配；④将LLM判断与确定性代码分离，形成可审计的研究循环；⑤批判模块基于多维度评估提供可编辑计划的操作。

**🔧 技术方法**

技术栈包括 LangGraph、LangChain、Pydantic、HTTPX；对接 arXiv、OpenAlex、Crossref、Semantic Scholar、PubMed、Web/Tavily 等学术 API；实现去重、雪崩化、批判、分配等确定性算法；LLM 通过 Anthropic、OpenAI、Google、Ollama 等后端执行判断、阅读、批判等角色。

**📊 数据集**

案例研究使用公开学术 API 检索 AI 代理记忆领域（2023+）文献，共检索 237 篇文献，读取 28 篇；在实验中未使用特定预先准备的数据集，而是实时调用学术 API。

**📈 对比分析**

没有进行系统的基准对比，只在单次案例中报告：研究迭代 2 次、检索操作 30 次、雪崩回合 2 次、LLM 调用 116 次、文献读取 28/40、平均证据强度 0.59；说明可在几分钟内完成研究，未给出 Recall/Precision 等量化指标。

**⚠️ 局限性**

主要局限：①未评估检索召回率和证据提取精度；②全文阅读效果尚未量化；③命名实体同义句聚类仍采用文本相同，易漏检；④LLM 判断误差难以修复；⑤缺乏与现有系统的基准对比；⑥部分弱点已修复，但仍存在 API 可用性和计费预算的限制。

---

## 216. Risk and Refusal: Mechanisms of LLM Responses to Suicidal Ideation

**arXiv ID:** 2610.22715 | [PDF](https://arxiv.org/pdf/2610.22715v1)

**作者:** Drishti Goel `[一作]` (University of Illinois Urbana-Champaign), Koustuv Saha `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对八个指令调优的大语言模型在三组自杀倾向披露数据集上进行行为评估，并通过 2×2 的实验设计探讨分类标签和临床指导对拒绝行为与安全风险的影响，同时对六个开放权重模型进行拒绝机制的表征与因果干预。

**💡 创新点**

首次将通用拒绝方向迁移到危机情境，揭示拒绝与风险的分离，并验证临床指导能够系统性降低拒绝率和安全风险，说明拒绝机制并非单一决定因素，强调需要针对不同援助需求调节拒绝阈值。

**🔧 技术方法**

采用指令调优 LLM、RubRIX（支持寻求版）评估、残差流拒绝方向提取、因果激活干预、语义归因图与跨模型语义图构建等技术。

**📊 数据集**

使用 Reddit SI（1,504 篇自杀倾向帖子，IPTS 标签）、SH‑Detection（1,500 条自伤检测贴）和 X‑Sensitive（243 条自伤标记推文）三组公开数据集。

**📈 对比分析**

通过四种提示条件比较拒绝率与 RubRIX 评估风险，结果显示单纯提供 SI 分类会降低拒绝率但风险上升，临床指导显著降低两者；机制干预证明消除拒绝方向能显著降低拒绝但并未改善风险，说明降低拒绝并非提升安全的唯一途径。

**⚠️ 局限性**

研究仅涵盖有限模型与数据集，评估基于 LLM‑judge 的风险标记未得到人类或临床验证，缺乏多轮对话与真实危机情境实验，且拒绝机制被简化为线性方向，可能未捕捉完整的拒绝表征。

---

## 217. Same Person, Different Advice: Auditing LLM Financial Advice with Repeated Risk-Tolerance Responses

**arXiv ID:** 2610.23060 | [PDF](https://arxiv.org/pdf/2610.23060v1)

**作者:** Soovadeep Bakshi `[一作]` (Independent Researcher), Sandip Nandi `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过同一客户在不同时间回答同一风险偏好问题，对七大语言模型在一次性财务建议中的建议波动进行审计，并提出并测试了噪声提示、额外问题及澄清提问等缓解措施。

**💡 创新点**

创新点在于首次利用真实面板的同人复测数据量化风险偏好回答的不确定性对 LLM 建议的影响，构建了基于可靠性折扣的评价基准，并通过脚本化交互测试验证了澄清提问在稳健化建议中的效果。

**🔧 技术方法**

采用经典测量理论（真分数估计、可靠性系数）、线性回归、Bootstrap 置信区间、LLM API 调用、脚本化对话实验和多模型对比分析技术。

**📊 数据集**

使用的主要数据集包括美国 2007-2009 年 Survey of Consumer Finances (四级风险问题) 及 Basel–Berlin 风险研究 6 个月复测（0–10 量表）以及公开的金融资产和人口特征数据。

**📈 对比分析**

通过比较七种模型在风险回答变化时的平均建议波动（10–16个百分点）、同一提示重复的波动（0–4个百分点）以及基准折扣后的波动比值，结果显示风险回答对建议的影响显著大于同提示重复；噪声提示和额外问题可将波动降低 4–8个百分点，澄清提问将波动降至约 2个百分点且保留大部分对确认变化的响应。

**⚠️ 局限性**

局限性包括：可靠性基准假设模型对“非常可靠”提示的响应即为其对真实态度的响应；单个风险问题测量误差可能被低估；样本不涵盖低资产或特殊人群；仅测试单轮提示、英语；未验证模型内部如何处理可靠性信息；未评估实际投资绩效或适宜性违规。

---

## 218. The Source of Data Value: Consumer Disclosure, Access Design, and Platform Monetization

**arXiv ID:** 2610.22864 | [PDF](https://arxiv.org/pdf/2610.22864v1)

**作者:** Hexiao Liu `[一作]`, Mingzheng Wang `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析数字平台在消费者披露权变化下，数据服务对卖家、平台收益及消费者福利的影响。

**💡 创新点**

提出将数据价值拆分为销售收入效应与竞争威胁效应的框架，揭示隐私治理如何改变数据服务的对价结构。

**🔧 技术方法**

使用博弈论、顺序决策分析与Hotelling价格模型进行理论推导。

**📊 数据集**

无实证数据集，采用理论模型与参数设定进行数值演示。

**📈 对比分析**

通过理论对比NDC、MDC、VDD三种治理模式的均衡结果，展示平台利润、卖家净利、消费者剩余及社会福利的变化。

**⚠️ 局限性**

模型假设均匀消费者偏好、同质隐私成本、完全识别以及平台只能提供一套信息服务，未考虑卖家异质性、第一方数据、非价格收益等因素，可能限制对现实市场的外推。

---

## 219. R$^2$Orch: Recursive Self-Improvement for Agent Orchestration via Self-Play within a Skill-Internalization Loop

**arXiv ID:** 2610.22740 | [PDF](https://arxiv.org/pdf/2610.22740v1)

**作者:** Mingda Zhang `[一作]` (Chinese University of Hong Kong), Xiaoying Tang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了递归自我对弈（Recursive Self-Play Orchestration）框架，利用递归可学习性（Recursive Learnability）和技能撤回（Skill Withdrawal）两大机制，使得编排器能够在没有任何技能的情况下完成复杂任务。

**💡 创新点**

创新点在于：①将自我对弈拆分为内外两层循环，外层通过递归可学习性优化任务选择，内层通过技能撤回进行自我评估；②递归可学习性通过隐藏候选技能来衡量任务在无技能情况下仍能产生多样化结果的可学习性；③技能撤回采用随机隐藏与贝叶斯对比来客观决定何时舍弃技能，从而避免优势坍塌和评估噪声。

**🔧 技术方法**

技术方法包括：两时尺度随机逼近、GRPO 的优势梯度估计、相对熵策略搜索、KL 正则化强化学习、贝叶斯自举后验、Dirichlet 先验、随机块抽样、以及对齐的前馈图结构优化。

**📊 数据集**

实验数据集涵盖十二个基准，包括六个内部分布（问答、数学、代码、交互式任务）和六个外部分布，典型任务如 AIME 2026、WebShop、ALFWorld 等。

**📈 对比分析**

与六大类基线（直接提示、骨干训练、工作流搜索、RL 训练编排、技能内部化、自我对弈加技能演化）对比，递归自我对弈在所有十二个基准上均超越基线，且在零技能部署下仍保持最高准确率，并能在不重新训练的情况下提升其他骨干模型的性能。

**⚠️ 局限性**

局限性在于：框架需要手工为每个数据集预设初始技能集，自动化技能发现和初始化仍是未解决的问题。

---

## 220. Dynamic Auction Mechanisms in a Closed-Credit Economy: Sequential Slot Reallocation in Air Traffic Management

**arXiv ID:** 2610.22683 | [PDF](https://arxiv.org/pdf/2610.22683v1)

**作者:** Jihun Kim `[一作]` (University of California), S. Shankar Sastry `[通讯]` (University of California)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究在容量不确定且已有资源配额的情况下，利用可重复的人工信用进行动态资源再分配，并提出分组递增拍卖机制实现市场清算。

**💡 创新点**

证明在闭合信用体系下无法同时满足Pareto效率、激励兼容、非我的个体理性和弱非交易者中立，并提出保持非我的个体理性与弱非交易者中立的分组升价拍卖，提供条件效率与近似激励兼容性。

**🔧 技术方法**

使用可重复信用、预测价值到期函数、价格单调需求、分组升价拍卖、随机分配余量、信用余额递推等技术。

**📊 数据集**

使用旧金山国际机场（SFO）2026年8月20日的航班到达时隙公布计划及BTS T‑100数据（航班成本、座位数、乘客数）进行仿真。

**📈 对比分析**

与现行基于时隙压缩的地面延迟基线（GD）对比，仿真表明在容量削减情境下总成本降低7–18%，延迟和取消成本与GD相当或更低。

**⚠️ 局限性**

机制在效率与激励兼容方面仅有近似保证；子组划分与预算分配的选择对激励影响显著，且在所有情形下无法保证Pareto效率。

---

## 221. Ignored, Not Refused: What Low Prompt-Injection Success Rates Hide in Small Tool-Using Agents

**arXiv ID:** 2610.23254 | [PDF](https://arxiv.org/pdf/2610.23254v1)

**作者:** Farhan Zarif `[一作]` (BRAC University), Farig Yousuf Sadeque `[通讯]` (BRAC University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对33,000条开放权重工具使用代理在InjecAgent与AgentDojo基准下的失败攻击进行行为审计，探究低攻击成功率（ASR）背后的实际攻击是否被忽略或拒绝，并关联任务完成情况。

**💡 创新点**

创新点在于发现大多数失败攻击被忽略而非拒绝、系统提示词和量化精度可显著改变ASR，并提出攻击-任务结果矩阵（ATOM）以同时评估安全与效用，揭示单一ASR无法体现实际风险。

**🔧 技术方法**

采用LLM无标签判断器进行轨迹审计，系统提示词改写（授权/禁止）与FP8/NF4量化实验，评估攻击成功率、任务完成率、拒绝率及无效输出率等指标。

**📊 数据集**

使用InjecAgent与AgentDojo基准生成的33,000条攻击轨迹，覆盖11款2–14B参数的开放权重模型。

**📈 对比分析**

在原始基准下ASR低于10%，但改为授权提示后可升至43%；ATOM显示约21.7%的安全但无用案例，表明单一ASR不足以评估代理的实用安全性。

**⚠️ 局限性**

局限在于仅使用静态单代理基准、无自适应攻击、仅考虑贪心解码、LLM判定可能漏判、未覆盖多代理或实时部署环境，以及未测量量化对效能的细致影响。

---

## 222. Reliability-Consistent Appearance Modeling for RGB-Infrared Multi-Object Tracking

**arXiv ID:** 2610.23125 | [PDF](https://arxiv.org/pdf/2610.23125v1)

**作者:** Bingyang Wang `[一作]` (Dalian University of Technology), You He `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可靠性一致的外观建模框架RCA-MOT，用于RGB-红外多目标跟踪；

**💡 创新点**

创新点在于保留RGB、红外与融合三分支外观特征并通过训练无关的目标场景可靠性估计，对关联与记忆更新两阶段统一控制；

**🔧 技术方法**

使用标准的双流ResNeSt-50编码器、简单1×1融合、GeM–BNNeck嵌入、KL与对齐约束、可靠性权重调节以及EMA记忆写入门控；

**📊 数据集**

在VT-MOT与UniRTL两个RGB-红外跟踪基准上进行评测；

**📈 对比分析**

与多种基线（PFTrack、UniS-MOT等）在固定检测框的条件下对比，RCA-MOT在HOTA、IDF1等身份指标上分别提升约5–6点和11–13点，表现为SOTA；

**⚠️ 局限性**

局限在于仅使用训练无关的可靠性估计，对极端光照或热成像噪声的适应性可能受限，且对融合后的特征在极端失真时仍可能产生误差。

---

## 223. From Natural Language Intent to Executable Control: Hierarchical Feedback-Guided LLM Planning for Multi-component Cyber-Physical Systems

**arXiv ID:** 2610.22441 | [PDF](https://arxiv.org/pdf/2610.22441v1)

**作者:** Jinglin Dai `[一作]` (Nanjing University), Lei Bu `[通讯]` (Nanjing University)

**通讯引用:** 943 | [OpenAlex ID](https://openalex.org/A5029676029)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种层次化反馈框架，利用大语言模型从自然语言任务指令生成可执行的分层任务计划（Scenario Task）和信号时序逻辑（STL）约束，并通过一致性检测与控制合成诊断实现对多组件物理系统（CPS）的可执行控制。

**💡 创新点**

创新点：①将任务计划与任务级STL同时生成并引入一致性检查算法，提前发现语义和约束冲突；②将控制合成的诊断信息映射到任务层面，指导任务拆分和组件分配的重规划；③采用双层（任务层与控制层）反馈循环，显著提升成功率；④在实验中取得最高的自然语言到可执行控制的验证成功率。

**🔧 技术方法**

技术：大语言模型（DeepSeek‑V4‑Flash、GPT‑5、Qwen3.7‑Plus）+ 任务提示与重写；Scenario Task 任务树 + STL 规范；三值逻辑一致性检测算法；SNIFF 控制合成工具；诊断信息抽取与反馈重写；Python + C++实现。

**📊 数据集**

数据集：手工构建的七个异构多组件 CPS 场景（包括机器人、UAV、卡车、救援装备等），部分场景来自 AI2‑THOR；每个场景包含四个自然语言目标；每个目标的任务与约束数量随难度递增。

**📈 对比分析**

比较方法：与两种 LLM‑规划基线（SMART‑LLM、NL2HLTL2PLAN）在 AI2‑THOR 仿真环境中对比；使用生成成功率、AI2‑THOR 执行成功率、平均行驶成本等指标。实验结果显示，本框架在 GPT‑5 上实现了 91.4% 的生成成功率、87.9% 的验证成功率；在 ablation 研究中，完整反馈策略在三种 LLM 上分别获得 76.4%、87.9%、81.4% 的成功率，明显优于去除任务‑STL 检查、停止重规划或仅提供错误提示的方案。

**⚠️ 局限性**

局限性：LLM 仍可能产生幻觉或语法错误，需多轮重写；层次化重规划有时会破坏原始自然语言中的并行/顺序意图；对极大规模、多机器人环境的可扩展性尚未充分验证；控制层诊断在极复杂的连续约束下仍可能不足以完全修复失败。

---

## 224. MOSS-Image Technical Report

**arXiv ID:** 2610.23374 | [PDF](https://arxiv.org/pdf/2610.23374v1)

**作者:** MOSI. AI `[一作]` `[通讯]`, MOSI. AI

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过从零预训练一个5B单流Diffusion Transformer（DiT）并搭配意图感知的提示增强器，实现了从简短文本请求到信息密集型图形的端到端生成；

**💡 创新点**

创新点包括：①统一的文本条件接口支持自然语言与结构化布局两种输入；②多粒度、结构感知的监督与多阶段分辨率训练；③利用代理模型将简短请求转化为可执行的布局与内容规范；④推出了长文本渲染与真实布局竞技的新基准。

**🔧 技术方法**

技术手段涵盖：5B单流DiT、Qwen3.5-9B文本编码器、FLUX.2 VAE、SiTU‑GLU、4D RoPE、Muon‑AdamW优化器、GPT‑6提示增强器、JSON布局规范等。

**📊 数据集**

数据集主要来自ReLAION、OmniCorpus、DataComp等公开图文数据，加入多粒度重生成式说明与结构化布局注释。

**📈 对比分析**

在LongLongTextBench上获得96.3的Mean‑6分（开放模型最高），在RealLayoutArena中以1698.70 Elo排名第二；在共享增强提示下不同骨干网络对比显示本骨干最高达1807.34 Elo，说明在文本渲染和布局控制上明显优于现有开源模型。

**⚠️ 局限性**

局限性包括：目前仅支持图像生成，缺乏参考图像条件与图像编辑能力；提示增强器依赖外部语言模型，可能带来审美判断、参考解读与事实错误的问题。

---

## 225. Hard Instances on Demand

**arXiv ID:** 2610.22637 | [PDF](https://arxiv.org/pdf/2610.22637v1)

**作者:** Jamie Fravel `[一作]` (Charles University), Pavlo Tsitsei `[通讯]` (Charles University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过将实例生成视为黑盒优化问题，利用Nevergrad等优化器在SAT、TSP、HAM和NPFS等四个NP-hard问题上自动寻找最难的实例。

**💡 创新点**

提出一种可复用的“硬实例按需生成”工作流，能够在没有领域知识的情况下产生比随机或已有基准更难的实例，并在实验中实现大幅提升。

**🔧 技术方法**

使用Nevergrad黑盒优化框架、实例参数化编码、端到端求解时间作为目标函数，以及多种求解器组合和测量协议。

**📊 数据集**

基准数据包括随机生成实例、SAT竞赛公式、TSPLIB、Hard‑TSPLIB、FHCP、Taillard/VRF/Demirkol流程车间实例。

**📈 对比分析**

通过预算匹配的随机抽样、公开基准库和已知硬实例族进行对比，实验结果显示所生成冠军实例比随机实例快10倍以上、SAT冠军比竞赛公式更难、TSP冠军比Hard‑TSPLIB更难、NPFS冠军比100k随机抽样更难。

**⚠️ 局限性**

局限在于硬度依赖特定求解器、对随机种子过拟合、对大规模实例的计算成本高、未能提供最坏情况证明或对其它求解器的普适性。

---

## 226. Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows?

**arXiv ID:** 2610.22887 | [PDF](https://arxiv.org/pdf/2610.22887v1)

**作者:** Bangji Yang `[一作]` (University of Illinois at Urbana-Champaign), Ge Liu `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种并行约束强化学习（CCR），在有限的回合窗口内批量完成多题目，并通过前缀奖励保持正确性反馈。

**💡 创新点**

创新点在于：①通过共享token预算实现多答案的并行生成，②利用前缀奖励在答案被截断时仍能保留信号，③在可行窗口内给出奖励噪声与更新方差的理论保证。

**🔧 技术方法**

使用GRPO框架、批量上下文强化学习、准确率与格式奖励、前缀容忍验证以及TRLan训练栈。

**📊 数据集**

使用DeepMath‑103K子集（约3000道数学题）进行训练与评估。

**📈 对比分析**

与单题GRPO（32k窗口）比较：在Qwen3‑4B与Qwen3‑8B模型上，CCR将每步token消耗分别降低82.1%和79.1%，同时保持或略升单题推理准确率（4B从75.3%到77.7%，8B从68.8%到69.7%）。

**⚠️ 局限性**

局限性：①依赖于可行的共享窗口大小，窗口过小可能导致奖励稀缺；②需要前缀可验证机制，若答案生成不规范则奖励丢失；③理论分析假设候选间独立，实际模型可能产生相关性；④未证明对更大窗口或更高并行度的收敛性。

---

## 227. UniPolar: Joint Image and Polarimetric Restoration under Compound Degradations via Physical Consistency Learning

**arXiv ID:** 2610.23360 | [PDF](https://arxiv.org/pdf/2610.23360v1)

**作者:** Yang Jin `[一作]` (Northwestern Polytechnical University), Binglu Wang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 UniPolar，联合恢复受设备与环境复合退化的 CPFA 图像与极化量。

**💡 创新点**

创新点在于耦合图像与 Stokes 双流，通过 Dual‑Domain Frequency Interaction 与 Degradation‑Conditioned Modulation 实现频域与退化条件耦合，并在输出层加入极化物理一致性学习。

**🔧 技术方法**

使用 Restormer 背骨架、双流编码解码、DDFI、DCM、频域波形变换、FiLM 归一化、物理一致性约束及多尺度注意力。

**📊 数据集**

构造合成的包含正常、低光和雾三种退化的 CPFA 数据集，来源于公开极化数据集、Sparse‑PDM、PLIE 等，总计 642 张样本。

**📈 对比分析**

在三种退化条件下与任务特定与通用多退化方法对比，UniPolar 平均 PSNR/SSIM 达 31.5/0.875，DoLP PSNR/SSIM 24.1/0.607，AoLP 误差 16.8°，在所有基准中均表现最佳。

**⚠️ 局限性**

局限在于对极化弱区或空间变异雾的鲁棒性仍有限，且依赖退化标签和大量合成数据，真实场景中的性能可能受限。

---

## 228. StratFL: Countering Covert Model Poisoning Attacks with Strategic Federated Learning

**arXiv ID:** 2610.22626 | [PDF](https://arxiv.org/pdf/2610.22626v1)

**作者:** Manoj Twarakavi `[一作]` (Missouri University of Science and Technology), Venkata Sriram Siddhardh Nadendla `[通讯]` (Missouri University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于博弈论的联邦学习防御框架，用来抵御隐蔽模型中毒（Covert Model Poisoning）攻击；

**💡 创新点**

创新点在于将服务器与攻击者建模为不完全信息的重复博弈，并引入了连续近似的信念更新算法与投影梯度对抗搜索算法（StratFL），实现了自适应的客户端选择与攻击行为预测；

**🔧 技术方法**

核心技术包括：不完全信息博弈建模、贝叶斯信念更新（采用Sigmoid近似与滑动窗口求平均）、投影梯度下降/上升的对抗搜索；

**📊 数据集**

实验数据集为MNIST与CIFAR‑10，使用10个客户端、50轮训练、ResNet‑18与2层CNN模型；

**📈 对比分析**

与FedAvg、Krum、Trimmed‑Mean、Coordinate‑wise Median、FLTrust、FLANDERS等方法对比，StratFL在恶意比例α≤0.7时准确率提升约10%，在α>0.7时表现仍优于大多数基线；

**⚠️ 局限性**

局限性包括：在大量客户端被攻击（α>0.7）时性能急剧下降；需要服务器与攻击者各自拥有私有验证集，实际部署时可能不满足；仅针对模型中毒，未覆盖数据中毒、后门等其他APT攻击。

---

## 229. Structured Perception Representations for Off-road Dynamics Learning

**arXiv ID:** 2610.22900 | [PDF](https://arxiv.org/pdf/2610.22900v1)

**作者:** Micah Nye `[一作]` (Carnegie Mellon University), Wenshan Wang `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于结构化鸟瞰视图（BEV）感知的增量条件动力学模型（SpatialBEV），用于高速度越野车辆的动力学学习；

**💡 创新点**

创新点在于：①在每一步预测时实时查询局部BEV特征，充分利用空间结构；②提出了针对动态数据集的最难序列挖掘基准；③在闭环控制中与MPPI结合，验证增量条件的实用性；

**🔧 技术方法**

使用技术包括：LSTM递归动力学网络、循环BEV细胞、双线性插值采样、LiDAR几何地图与VFM语义特征、基于聚类的高方差序列筛选、MPPI控制；

**📊 数据集**

使用数据集为两地实测越野轨迹，79.8k训练样本（2.2h）、75.3k测试样本（2.1h），涵盖春、秋、夏季及34%冬季雪地，BEV地图分为稀疏与稠密两种；

**📈 对比分析**

与四种基线（KBM、State LSTM、FPV LSTM、Vanilla BEV LSTM）比较，评估指标为MSE与P95；SpatialBEV在稠密地图上平均降低约20%预测误差，在硬件实验中最大轨迹误差降低25.3%，平均CTE与干预次数接近或优于其他学习基线；

**⚠️ 局限性**

局限性包括：需在线适应以提升鲁棒性；数据收集成本高、样本效率低；BEV查询与插值带来计算开销，导致采样粒子数受限；控制器与动力学误差耦合导致闭环与开环性能不完全相关；

---

## 230. FlexUMM: Flexible Representation Learning for Unified Multimodal Models

**arXiv ID:** 2610.22682 | [PDF](https://arxiv.org/pdf/2610.22682v1)

**作者:** Yinyi Luo `[一作]` (Carnegie Mellon University), Jindong Wang `[通讯]` (William & Mary)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究统一多模态模型的自适应计算分配，提出 FlexUMM 框架，可动态分配共享/专用计算与输入深度。

**💡 创新点**

创新点在于三阶段动态策略（共享通信、令牌级专用化、序列级早停）在同一 backbone 上实现可插拔的自适应分配，兼顾性能与效率。

**🔧 技术方法**

使用多头注意力共享、三状态容量门、低秩残差银行、序列级停止头以及候选退出机制，采用后训练微调方式实现。

**📊 数据集**

后训练使用 ScienceQA、AI2D、TextVQA、MathV-360K、InstructPix2Pix；评估采用 MMMU、MathVista、MMStar、MMBench、GenEval、WISE、Unison、Unified-Bench、Uni-SafeBench。

**📈 对比分析**

与 Bagel、Janus-Pro、Harmon 三大 UMM backbone 对比，平均提升 10.17% 任务精度，FLOPs 降低 6.33%，在理解和生成任务上均优于现有统一模型。

**⚠️ 局限性**

局限在于仍依赖密集实现缺乏实际 wall‑clock 加速，极端输入下自适应深度可能误判，对安全性改进有限。

---

## 231. MLLM Frame Subset Ensembling for Audio-Visual Video QA and MLLM-based Reranking for Ad-hoc Video Search in TRECVID 2025

**arXiv ID:** 2610.23214 | [PDF](https://arxiv.org/pdf/2610.23214v1)

**作者:** Andreas Goulas `[一作]` (Centre of Research and Technology Hellas), Vasileios Mezaris `[通讯]` (Centre of Research and Technology Hellas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在TRECVID 2025的两大任务——自发视频检索（AVS）与视频问答（VQA）中，作者构建了两阶段无训练的检索-重排序框架以及音频视觉多模态LLM的微调与推理时规模化框架；

**💡 创新点**

创新点在于：①利用LLM对查询进行自动重述扩展并通过多模态VLM融合多重相似度；②采用无训练的MLLM（Qwen2.5‑VL‑7B）对检索结果进行上下文相关性重排序；③在VQA中将帧子集聚合与贪婪采样相结合，通过多数投票与logit均值聚合提升答案质量；

**🔧 技术方法**

主要技术包括：大型语言模型（LLaMA‑3.2‑1B‑Instruct、Qwen2.5‑VL‑7B、Qwen2.5‑Omni‑7B）、多模态视觉语言模型（CLIP、BLIP、BLIP‑2、SLIP、BEiT3）、相似度归一化与加权融合、LoRA微调、贪婪采样与投票聚合；

**📊 数据集**

使用的数据集为V3C2视频集合（包含2022‑2025年查询），TRECVID 2025 VQA训练集（500段含音视频的短视频）与官方评测数据；

**📈 对比分析**

与官方提交的多轮实验相比，AVS的最佳运行（重排深度4k）在MXinfAP上取得团队最高分；VQA的主运行在回答生成任务中以METEOR、BERTScore、STS三项指标获得最高分，回答生成任务排名第一；在多选任务中以top‑1准确率与MRR排名第二；

**⚠️ 局限性**

局限性包括：①AVS重排阶段完全无训练，若对特定领域需求可能受限；②帧子集聚合需要额外推理成本；③在VQA多选评测中，部分问答由于引号转义错误导致评分偏差，影响结果可重复性；④整体系统对大型LLM依赖较高，资源消耗大。

---

## 232. BRICE: Brain-inspired Recurrent Illusory Contour Encoder for Point Cloud Completion

**arXiv ID:** 2610.23001 | [PDF](https://arxiv.org/pdf/2610.23001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 233. Online Constraint Optimization-Based Indirect-Adaptive Robust Control for Autonomous Vehicles with Unknown Friction

**arXiv ID:** 2610.22675 | [PDF](https://arxiv.org/pdf/2610.22675v1)

**作者:** Yupeng Zhou `[一作]` (Purdue University), Bin Yao `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一个双层OCP–IARC框架，用于在摩擦不确定条件下实现自动驾驶车辆的时间最优运动控制。

**💡 创新点**

创新点在于将在线摩擦适应、间接自适应鲁棒控制与估计监测相结合，利用平滑参数重置避免参数突变导致的规划不可行，同时在规划层显式保留鲁棒裕度。

**🔧 技术方法**

采用非线性模型预测控制、间接自适应鲁棒控制、投影估计法、估计监测与平滑重置、acados求解器以及滤波最小二乘参数更新技术。

**📊 数据集**

使用仿真数据，基于Simscape Multibody构建的测试赛道，并在不同摩擦系数（1.0、0.3、0.5、0.7、0.8、0.4）段上进行验证。

**📈 对比分析**

与固定摩擦系数的基准NMPC（高摩擦、低摩擦、极低摩擦）对比，提出方法完成全赛道时间为153.2 s，横向误差RMSE仅0.208 m，速度误差保持在0.05 m/s以下，基准方法要么在低摩擦区撞毁，要么耗时更长。

**⚠️ 局限性**

局限性包括：仅在非饱和驱动区验证，未在真实硬件上测试，参数收敛不保证，估计精度依赖滑移量测量与滤波参数选择。

---

## 234. SimOPD: Relational Knowledge Transfer for On-Policy Distillation

**arXiv ID:** 2610.23306 | [PDF](https://arxiv.org/pdf/2610.23306v1)

**作者:** Kai Tang `[一作]` (Zhejiang University), Weiqiang Wang `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SimOPD，一种通过教师内部表示的响应相似度分布匹配实现的on‑policy蒸馏方法。

**💡 创新点**

创新点在于利用教师与学生生成响应的成对内存银行，在各自特征空间内匹配相似度分布，避免跨模型投影并保持学生的表示自由度。

**🔧 技术方法**

结合对比学习思路、余弦相似度分布、温度缩放、动态双重内存银行与Policy Gradient实现KL匹配损失。

**📊 数据集**

使用数学推理数据集Dapo‑Math‑17K、AIME 2024/25、AIMO；代码生成数据集Eurus子集、HumanEval、MBPP、LeetCode；OOD评估ARC‑C、GPQA、MMLU‑Pro。

**📈 对比分析**

与6种基线（OPD、ExOPD、PowerOPD、OPD²、OPRD、OPRD‑Bridge）对比，SimOPD在同架构/跨架构场景中均名列前二，平均提升约2.6–3.1个百分点，联合OPD进一步提升至4.6–5.1个百分点。

**⚠️ 局限性**

局限性包括仅在推理任务上验证，未覆盖开放式生成；相似度匹配增加计算与内存开销；内存银行管理策略仍有进一步优化空间。

---

## 235. From Approval to Abuse: Privilege Escalation Attacks on Harness-Protected Coding Agents

**arXiv ID:** 2610.22987 | [PDF](https://arxiv.org/pdf/2610.22987v1)

**作者:** Xianglin Yang `[一作]` (National University Of Singapore), Jiaheng Zhang `[通讯]` (National University Of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何通过在代码仓库中植入对抗性内容，使受害者的编程代理在完成日常任务时提出过度授权的可复用权限规则，并在用户同意后利用该权限完成后续恶意目标。

**💡 创新点**

提出了双阶段实验框架（权限获取 + 权限复用），首次系统化评估可复用权限规则的获取风险与被利用风险，并揭示了代理在不同权限抽象轴下的行为差异。

**🔧 技术方法**

采用大语言模型驱动的场景规划器、仓库构造器、代理仿真与反馈评审等组件，实现对抗性仓库的自动生成与迭代优化；使用权限能力覆盖（PCC）与目标完成率等指标评估攻击效果。

**📊 数据集**

实验数据集基于 Codex CLI 与 Claude Code 四种配置，分别创建 5 类（普通、敏感、关键）×5 个实例的恶意目标（读/泄露/删/下载/改配置）共 25 个案例，并利用自动生成的测试用例进行权限覆盖评估。

**📈 对比分析**

与 Direct Request、Direct Injection 以及 IssueTrojanBench 等基线对比，实验显示本框架在 PCC 上平均提升 0.3–0.4，目标可达率从 0–60% 提升至 60–80%，在共享检查点上成功完成率提升至 81%（对 Codex 及 Claude 相关配置），表明对手能大幅利用授权复用。

**⚠️ 局限性**

局限性包括：依赖具体的执行 harness 设计（仅验证 Codex CLI 与 Claude Code）；对高度对齐模型（如 Claude‑2.5）仍存在失败；框架需多轮迭代，部署成本高；仅评估了静态仓库场景，无法覆盖更复杂的交互式或多阶段任务。

---

## 236. GPU-Accelerated Robust MA--RIS Pre-Optimization for Communication and Target Illumination

**arXiv ID:** 2610.22431 | [PDF](https://arxiv.org/pdf/2610.22431v1)

**作者:** Tran Le Dung `[一作]` (Hanoi University of Science and Technology), Ho Viet Duc Luong `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 93 | [OpenAlex ID](https://openalex.org/A5094207696)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种GPU加速的鲁棒多目标预优化框架（RAMP），用于在慢时尺度下联合预选可移动天线（MA）几何和重构智能表面（RIS）相位，从而获得可复用的 Pareto 集，平衡期望最差用户速率和最差目标照射功率。

**💡 创新点**

创新点包括：① 采用基于 Pareto 存活率的自适应跨域迁移控制，抑制负迁移；② 以域特定 Pareto 前沿为基础，构造跨域稳健归档；③ 将整个评估过程张量化并在 GPU 上实现，大幅提升计算速度。

**🔧 技术方法**

技术手段包括：基于 NSGA‑II 的多目标进化优化、GPU 张量化评估、快速非支配排序、拥挤距离算子、Pareto 存活率指数移动平均、外部归档与再评估。

**📊 数据集**

数据集采用合成 Monte‑Carlo 环境：16 个 MA、8×8 RIS（64 个单元）、4 路 Rician 通道、用户与目标方向均匀分布在 50–200 m 的 60° 角区，使用 32 次 Monte‑Carlo 样本进行评估。

**📈 对比分析**

与 MO‑DE、MOPSO、无迁移、固定迁移、池化 NSGA‑II、独立 NSGA‑II 等基线进行对比。RAMP 在平均 Hypervolume（HV）上高 4–15，IGD 低 0.018，显著优于所有基线，且 GPU 版相较 CPU 版总时长提升约 4.1×。

**⚠️ 局限性**

局限性包括：仅考虑远场发射照射功率（未建模回波、目标 RCS、杂波、接收处理等），使用固定在线功率分配策略，MA 仅在慢时尺度重新配置，RIS 归一化对尺寸的依赖未必反映实际物理效果。

---

## 237. Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making

**arXiv ID:** 2610.23170 | [PDF](https://arxiv.org/pdf/2610.23170v1)

**作者:** Qiang Yan `[一作]` `[通讯]` (Chinese Academy of Sciences), Qiang Yan (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种面向中文有限决策的单通道接口，利用预训练因果语言模型的候选字读出直接生成概率分布，用于布尔判断、分类和序数评分。

**💡 创新点**

创新点在于将候选字读出与分布监督、LoRA 微调以及 RLCD 目标结合，构建统一的决策模型并提供可部署的量化实现，未对模型规模做单独量化。

**🔧 技术方法**

使用了 Qwen3.5-0.8B 与 Qwen3.5-2B 作为语言模型骨干，LoRA 低秩适配、分布交叉熵训练、RLCD 强化学习、温度缩放、Q8 量化等技术。

**📊 数据集**

基准数据集为 CJ-Dataset/CJ-Bench，包含 13,048,000 条决策样本，涵盖通用、医学、法律、金融四个领域。

**📈 对比分析**

通过在完整 307,900 条 CJ-Bench 上评估，0.8B 模型达到 72.24% 通用准确率、76.98% 专业平均准确率，ECE 仅 1.07%；2B 模型在训练途中已突破 72.56% 通用准确率，但专业平均仍未超越 0.8B。

**⚠️ 局限性**

局限性包括缺乏多次实验和置信区间、未对比相同目标的 2B 训练、缺少对候选字排列、软硬标签、RLCD 影响的消融、对分布偏移与置信度的全面校准评估，以及未验证模型在真实外部域上的稳健性。

---

## 238. Trojan Horse: Automated Repository Poisoning against Coding Agents

**arXiv ID:** 2610.23109 | [PDF](https://arxiv.org/pdf/2610.23109v1)

**作者:** Xiao Yang `[一作]` (Baidu Inc.), Chenfu Bao `[通讯]` (Baidu Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了自动化框架 TrojanForge，用于在代码仓库中实现“仓库级”中毒攻击，并构建了 TrojanBench 基准。

**💡 创新点**

创新点：三阶段自动化流程（可达性分析、信任盲点发现、迭代式伪装策略）实现无需人工介入的仓库级中毒，并通过经验库实现跨仓库迁移。

**🔧 技术方法**

技术：多语言 AST 解析、执行轨迹映射、LLM 驱动的安全审计与策略合成、反向推理反馈以及经验库检索。

**📊 数据集**

数据集：从 SWE-bench、Commit0、GitHub 收集 136 个真实仓库（6 领域、9 语言），生成 384 个经过验证的受攻击仓库（TrojanBench）。

**📈 对比分析**

对比：与四个基线（BlindShot、RandMount、OnePass、ZeroForge）比较，TrojanForge 在四种攻击载体上平均 ASR 超过 60%，backdoor 率超过 90%；在 4 个 harness 与 12 模型上评估，证明即便商业代理也难以防御。

**⚠️ 局限性**

局限：生成一次有效受攻击仓库需要多轮迭代，成本较高；仅在实验环境下验证；公开发布仅限验证工具，未包含完整攻击链。

---

## 239. FCTS: Failure-Guided Curriculum Tree Search Over Generated Terrain for Legged Parkour

**arXiv ID:** 2610.22619 | [PDF](https://arxiv.org/pdf/2610.22619v1)

**作者:** Narayanan Palghat Parameswaran `[一作]` (Carnegie Mellon University), Harmish Khambhaita `[通讯]` (FieldAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于局部失败诊断的地形课程生成框架，并将其与蒙特卡罗树搜索结合，自动构建并探索多条四足机器人行走训练路径，以提升在工业与灾害现场的复杂地形通行性能。

**💡 创新点**

创新点包括：①利用失败位置、终止模式、机器人状态及周围地形几何等信息定制新的训练地形；②将课程生成与MCTS相结合，形成可搜索的策略改进树，实现多轨迹并行学习；③引入回放地形池防止遗忘，并在训练中平衡旧技能与新技能；④在有限训练预算下实现并行MCTS加速。

**🔧 技术方法**

使用技术：强化学习的四足行走学习框架、LLM驱动的可执行地形程序生成（类似Eurekaverse但加入失败反馈）、蒙特卡罗树搜索（PUCT、并行MCTS）、失败诊断与局部几何提取、回放地形回放机制、仿真到真实机器人（Spot）深度感知部署。

**📊 数据集**

数据集：扫描重建的工业与灾害现场（Debris site、Exposed Beam site），从中生成评估路段（relief‑stratified 路线与直接指定 路线）以及由LLM生成的训练地形集合；实验中使用512次验证试验、24小时搜索预算。

**📈 对比分析**

对比方法：Extreme Parkour、Eurekaverse、修改后的Eurekaverse。评价指标为成功率、Progress IQM、Waypoint Progress。结果显示，FCTS在中等难度下成功率从28.4%提升至85.8%，在难度下从14.9%提升至44.7%，在V‑Hard从0.8%提升至11.4%；在直接指定路线上同样表现优异。Sim‑to‑Real实验验证了在Spot机器人上的成功部署。

**⚠️ 局限性**

局限性：①搜索仍需完整训练跑，计算成本高；②对LLM生成程序质量和提示的依赖；③仅在扫描重建场景验证，缺乏真实现场实时测试；④不同机器人或传感器适配性未知；⑤局部失败诊断信息可能不足以覆盖所有复杂地形；⑥回放地形池阈值选择可能影响学习稳定性。

---

## 240. Has GPT-6-Astra Achieved Embodied Intelligence?

**arXiv ID:** 2610.22471 | [PDF](https://arxiv.org/pdf/2610.22471v1)

**作者:** SenseNova `[一作]` (Sensenova), Jing Zuo `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 GPT‑6‑Astra 在 12 个机器人基准上的能力进行大规模评估，探讨了在统一协议（支持 ICL 与 10×执行预算）下的表现、harness 设计对性能与成本的影响，并在真实机器人上验证零样本控制，最后提出了三大原则的新一代 embodied benchmark。

**💡 创新点**

创新点在于：①提出统一的评估协议，允许模型在训练示例中进行 ICL 并在执行时使用 10 倍的步数，显著提升成功率；②系统性分析了 harness 设计（System‑1/System‑2 协作、动作表示与分块）对性能与 token 成本的折中；③在仿真与真实机器人之间对比，揭示现实约束（安全、机械极限、实时性）对长预算的限制；④提出基于人类需求、无训练示例、实时世界的三大 benchmark 设计原则，并给出家庭环境的原型验证。

**🔧 技术方法**

使用的大型语言模型 GPT‑6‑Astra 结合视觉‑语言感知、工具调用、闭环协作、动作分块；仿真平台 NVIDIA Isaac Sim；真实机器人平台包括单臂、双臂、抓手、移动机器人、类人机器人；通过工具接口实现机器人控制、前向/逆向运动学、动作生成等。

**📊 数据集**

数据集包含 12 个公开基准：RoboDojo、RoboLab、LIBERO 系列（Long++、Ultra‑Long、PRO）、Bench2Dex、DexJoCo、SIMPLE、BEHAVIOR、VLN‑CE、RxR、LHPR‑VLN。

**📈 对比分析**

比较方法：在统一协议下分别评估零样本与 ICL 条件，使用 1× 与 10× 预算；与 SOTA 与 Agentic SOTA 结果对照；在真实机器人上执行零样本控制并记录成功率、所需时间和 token 费用。性能表现：仿真中大多数任务接近饱和（LIBERO‑Ultra‑Long 97%+，DexJoCo 90%+），但在真实机器人上成功率受限（如抓取柔性物体 2/4，插头插入 0/11），且单步成本与推理时间高昂。

**⚠️ 局限性**

限制与挑战：①ICL 仅在训练任务覆盖范围内有效，跨任务迁移差；②推理成本高（百万级 token，数小时推理时间），对实时部署不友好；③难以可靠地验证物理成功、识别失败原因和恢复不可逆失误；④长时序任务中难以跟踪并保持子目标进度；⑤现实世界的安全、机械极限与实时约束使得 10×预算优势受限；⑥缺少训练示例时的 ICL 无法使用，导致零样本性能显著下降。

---

## 241. ProactGen: Learning Proactive Clarification for Video Generation

**arXiv ID:** 2610.22608 | [PDF](https://arxiv.org/pdf/2610.22608v1)

**作者:** Zhixiao Wang `[一作]` (Nanjing University), Limin Wang `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个视频生成框架，通过主动澄清询问用户来补全初始提示，并利用生成的草稿训练问答策略。

**💡 创新点**

首次将基于草稿的主动澄清引入视频生成，并证明高质量草稿并非必要，低步去噪草稿即可实现高效提问。

**🔧 技术方法**

使用视觉-语言模型的问答策略、GRPO强化学习、DreamSim相似度评估、Wan2.2-TI2V-5B视频生成器以及低步草稿生成技术。

**📊 数据集**

训练集为约15K条MiraData视频及其详细字幕，评估集为InstanceCap 500视频和MiraBench-500 500视频。

**📈 对比分析**

与直接生成、视频提示优化方法（如SCMAPR、RAPO++-SSPO）以及改编的Proactive T2I比较，单轮澄清在相同时间下实现1.8倍以上的对齐提升，单轮成本约16秒，显著优于多轮重写。

**⚠️ 局限性**

受生成器指令遵循能力限制，且实验仅使用模拟用户，真实用户交互与多轮澄清集成仍需进一步验证。

---

## 242. On the Metric Distortion of the Minimum Spanning Tree Problem

**arXiv ID:** 2610.23032 | [PDF](https://arxiv.org/pdf/2610.23032v1)

**作者:** Kian Ghassemi `[一作]` (Sharif University of Technology), Saeed Seddighin `[通讯]` (Independent Researcher)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在只有顶点的相对顺序信息（ordinal ranking）而没有具体权重的情况下，研究如何近似求解最小生成树（MST）问题，并在不同度量假设和查询模型下给出相应的近似算法。

**💡 创新点**

提出了基于 Borůvka 的全局顺序算法，证明在一般度量空间上可实现 O(log n) 的失真（distortion），并给出树度量和有限 doubling 维度下可以达到常数失真甚至最优的结果；此外，还设计了线性查询复杂度的算法，在一般度量空间上实现 O(log* n) 的失真。

**🔧 技术方法**

主要技术包括：① 基于组件的 Borůvka 框架与排序约束下的边提议与无环选择；② 对树度量的“重定向”策略和“端点切割稳定性”；③ 利用轻稀疏支撑子图（light spanner）和双曲性质实现常数失真；④ 在查询模型中使用精确 Borůvka 轮与迭代预算控制。

**📊 数据集**

论文主要为理论分析，未使用具体实验数据集；所有结果均为理论证明和上界/下界。

**📈 对比分析**

与传统的 MST 算法（Kruskal、Prim）相比，在仅靠排序信息的前提下，提出的算法在一般度量下可实现 O(log n) 的失真，在树度量下完全恢复最优，且在 doubling 维度固定时实现 2^{O(D)} 的失真；在查询模型中，线性查询数即可获得 O(log* n) 的失真。相比先前已知的 3/2 下界，进一步证明任何确定性排序算法在一般度量下至少有 3/2 的失真。

**⚠️ 局限性**

限制主要包括：① 对一般度量空间，算法仍只能达到对数级失真，尚未收敛到常数；② 对于 doubling 维度的上界 2^{O(D)} 可能不是最优，尚缺乏匹配下界；③ 仅讨论确定性算法，随机化的潜在改进未被探索；④ 查询模型中虽然线性查询数足够，但是否能降到 o(n) 仍未知。

---

## 243. X-Splat: Unified Gaussian Splatting for Joint RGB and X-ray 3D Reconstruction

**arXiv ID:** 2610.23237 | [PDF](https://arxiv.org/pdf/2610.23237v1)

**作者:** Wangze Xu `[一作]` (University of Tokyo), Imari Sato `[通讯]` (National Institute of Informatics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种统一的高斯原语表示，兼顾RGB外观与X光内部结构，实现单模型的三维重建。

**💡 创新点**

创新点在于共享几何参数、将外观属性限定于表面近邻原语、采用几何代理对齐以及X光引导的深度一致性损失。

**🔧 技术方法**

使用了3D高斯投影渲染、光照与渲染算子、ICP+PCA对齐、基于X射线的光度损失、RGB光度损失以及自适应密度控制等技术。

**📊 数据集**

采用InsideOut数据集和自制Flower & Insect数据集（含花与昆虫的RGB与CT配对）。

**📈 对比分析**

与独立RGB/ X光模型及InsideOut等基线比较，统一模型在RGB渲染PSNR提升约1–2 dB、X光重建保持相当，且存储量比独立模型减少约25%。

**⚠️ 局限性**

局限在于只适用于不透明物体的表面反射、对X光质量依赖较高，且无法处理透明或散射材料。

---

## 244. Too Emotional to Be Fake? Intra-Utterance Emotion Shifts Expose Failure Modes in Speech Deepfake Detectors

**arXiv ID:** 2610.22832 | [PDF](https://arxiv.org/pdf/2610.22832v1)

**作者:** Jingyi Chen `[一作]` (Reality Defender), Surya Koppisetti `[通讯]` (Reality Defender)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出并实现了一个新的基准，系统地研究了在单句语音中情绪随时间变化对语音深度伪造检测的影响。

**💡 创新点**

创新点在于：①首次构造“情绪稳定（Å）”与“情绪变化（∇）”的对照实验，涵盖真实与合成两类；②兼顾受控（ESD）与自然（OMG‑Emotion、IEMOCAP、MSP‑Podcast）场景；③细致剖析检测器对情绪轨迹的特异性反应，并排除静态情绪身份与拼接声学伪影等混淆因素。

**🔧 技术方法**

使用了七种主流深度伪造检测模型（RawNet2、AASIST、XLS‑R‑SLS、XLS‑R‑Mamba、ProsDD、SLIM、Alethia），并通过 EER、固定阈值下的 FNR/FPR、logit 分数差（Δ_SP、Δ_BF）等指标进行评估；合成语音则通过四套表达式 TTS（IndexTTS2、TED‑TTS、HybridEmo、EmoTra‑TTS）生成情绪稳定与变化样本；对情绪轨迹采用 VAD（valence‑arousal‑dominance）连续注释。

**📊 数据集**

数据集包括：受控 ESD（10位英语说话人、5种情绪）；自然 OMG‑Emotion（情绪单词标签）；对话型 IEMOCAP、MSP‑Podcast（含 VAD 注释）。真实样本共 7,240 条，合成样本 9,720 条，总计 16,960 条。

**📈 对比分析**

比较方法：在每个真实性类别内，计算情绪变化与稳定样本的 logit 分数差，并在固定阈值下评估误检率；使用 1,000 抽样自举得到置信区间。结果显示：在 ESD、OMG‑Emotion 等设置中，情绪变化可显著降低合成语音的 spoof 分数，导致假阴率上升达 9.7–25.3pp；在 IEMOCAP、MSP‑Podcast 中部分检测器对情绪变化的真实语音给出更高分，误报警率提升 18.6–24.4pp；不同检测器对情绪轨迹的敏感性差异显著，且单一指标 EER 难以捕捉这些变化。

**⚠️ 局限性**

局限性：仅使用英语语料与四种 TTS 系统，可能限制跨语言与跨技术的推广；情绪稳定/变化的定义在不同语料中采用不同生成方式，难以完全分离因子；真实与合成样本在拼接与构造方式上仍存在差异；未来需扩展更多语言、情绪类型和合成技术以验证通用性。

---

## 245. Lean4AgenticRL: Advancing Agentic RL with Verifiable Multi-level Process Rewards

**arXiv ID:** 2610.22448 | [PDF](https://arxiv.org/pdf/2610.22448v1)

**作者:** Ruida Wang `[一作]` (Amazon Web Services), Chris Kong `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于 Lean4 的多层次过程奖励框架，并训练 LAR‑9B/27B 模型以提升 LLM 的 agentic 能力

**💡 创新点**

首次将高层结果奖励、工作流阶段奖励与工具调用奖励统一在可验证的 Lean4 形式化框架中，并引入工作流特定方案和奖励注解方法

**🔧 技术方法**

使用 Lean4Agent 形式化验证、Workflow‑Stage‑Normalized GRPO 算法、ReAct 规划‑执行模式以及工具调用检查器

**📊 数据集**

在公开基准数据集 SWE‑Bench‑Verified、TerminalBench‑2.0 与 TerminalBench‑2.1 上训练与评估

**📈 对比分析**

与同等规模开源模型对比，LAR‑27B 在 TerminalBench‑2.1 上达到 77.3%，比基线提升约 9.9%，比同类模型提升约 5.1%，总体性能显著优于对手

**⚠️ 局限性**

对谓词与 LLM 判定器的依赖限制了覆盖范围；工作流可解释性与跨任务泛化仍存在不足

---

## 246. ShadowFPT: Backdooring Federated Prompt Tuning via Shadow Triggers

**arXiv ID:** 2610.22871 | [PDF](https://arxiv.org/pdf/2610.22871v1)

**作者:** Kun Zhai `[一作]` (Fudan University), Xingjun Ma `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一种针对Federated Prompt Tuning（FPT）的隐蔽后门攻击，名为ShadowFPT，通过在冻结的CLIP backbone上预训练并适配可学习的Shadow Trigger，使触发样本被驱动到目标类别。

**💡 创新点**

创新点在于：①将冻结backbone视作后门表面，将攻击重点从上传的prompt更新转移到trigger-induced representation steering；②提出两阶段训练框架——先在公共或本地数据上预训练Shadow Trigger，再在联邦训练中动态适配并注入后门；③在目标文本对齐与层级视觉对齐两种对齐方式的组合下实现高鲁棒性。

**🔧 技术方法**

使用的技术包括：CLIP模型（ViT‑B/16），目标文本对齐、层级视觉对齐损失，联邦Prompt Tuning（文本、视觉、联合prompt tuning），多种聚合与防御策略（FedAvg、Krum、TrimmedMean、LaSA、FoolsGold、FLTrust、FLAME、CRFL、Bucketing）以及PCA可视化。

**📊 数据集**

所用数据集：Caltech101、Oxford Flowers、UCF101，均以非IID分布模拟多客户端情景。

**📈 对比分析**

与BadNets、Blended、Neurotoxin、A3FL、FCBA、PFedBA、Bad-PFL等攻击基线以及FedAvg、Krum、TrimmedMean、LaSA、FoolsGold、FLTrust、FLAME、CRFL、Bucketing等防御策略进行对比。ShadowFPT在所有设置下将攻击成功率（ASR）提升至约95%–99%，同时保持Clean Accuracy仅下降1–2个百分点，远优于所有基线。

**⚠️ 局限性**

局限性：①对更强或未知防御（如更高级的聚合、模型压缩）以及不同backbone（非CLIP）与prompt模态的鲁棒性尚待验证；②触发器仅为固定尺寸局部patch，可能对高分辨率或多模态输入有限制；③对联邦学习中数据隐私保护、通信成本和客户端异构性的整体影响尚未充分评估。

---

## 247. STRIDE: Evaluating Spatiotemporal Reasoning in Driving Edge Cases

**arXiv ID:** 2610.22795 | [PDF](https://arxiv.org/pdf/2610.22795v1)

**作者:** Lieqi Liu `[一作]` (University of California Los Angeles), Nanyun Peng `[通讯]` (University of California Los Angeles)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为STSBench的对象中心时空推理基准，涵盖六类任务（空间感知、空间理解、时间记忆、时间外推、轨迹预测、场景上下文意识），并采用多模板设计、MCQ+OEQ双格式评测；

**💡 创新点**

创新点在于：①整合对象级空间与时间推理，②针对长尾安全关键场景挑选数据，③结合多模板与自动化+人工审核生成高质量参考答案，④用联合MCQ‑OEQ评测揭示模型解释与判断不一致的细粒度差异；

**🔧 技术方法**

技术实现主要包括：基于多视角视频的视觉语言模型评估；自动化模板生成与结构化注释转换；多模型推理与人机协作校验；使用多种评测指标（MCQ准确率、OEQ分数、ADE/FDE）；

**📊 数据集**

数据集来源为nuScenes与Waymo E2E（WOD‑E2E）两大驾驶数据集，经过筛选、标注扩展与模板化生成，共计2,350道题目覆盖1,452个视频片段；

**📈 对比分析**

评测方法：对12个模型（包括GPT‑6‑Astra、Gemini‑3.1‑Pro、Claude‑Sonnet‑5、Qwen3‑VL‑30B‑A3B、UniAD、Dolphins等）在MCQ、OEQ和轨迹预测上进行统一评估。结果显示GPT‑6‑Astra单选最高约45%，OEQ分数约6–7/10，其他模型更低；整体表现远低于随机基线，说明模型仍存在显著缺陷；

**⚠️ 局限性**

局限性：①模型在时空推理（尤其是运动历史重构和未来外推）上准确率低；②解释性高但安全关键判断弱；③记忆与运动推理差距大；④依赖手工校验的参考答案，可能带来注释误差；⑤对极端复杂场景的覆盖仍有限。

---

## 248. Memory Bandwidth, Not FLOPs: Profiling and Accelerating Query-Based Segmentation Decoders on Edge GPUs

**arXiv ID:** 2610.22377 | [PDF](https://arxiv.org/pdf/2610.22377v1)

**作者:** Oliver Busemann `[一作]` (Osnabruck University), Martin Atzmueller `[通讯]` (Osnabruck University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对边缘设备的实例分割模型，在 transformer 解码器上引入了内存带宽瓶颈分析，并提出 CompactInst 通过最大池化减少像素 token 并使用 Refinement Head 恢复细节，从而显著降低 DRAM 流量。

**💡 创新点**

创新点在于：① 通过 FP16 TensorCore 层级 roofline 和 DRAM 计数器定位解码器的主要内存带宽瓶颈；② 以参数无关的空间池化（4×）与后置卷积提升头作为最小改动，直接减少跨核激活流量；③ 通过实验验证 token 数量减少能在低带宽设备上实现多倍速度提升。

**🔧 技术方法**

技术方法包括：Transformer 双路径解码器、FP16 TensorCore、CUDA 与 TensorCore 的 roofline 分析、硬件 DRAM 计数器、NVIDIA Nsight Compute、TensorRT FP16 静态引擎、Adaptive Max Pooling、Refinement Head 以及多尺度特征融合。

**📊 数据集**

使用的公开数据集为 COCO（实例分割）和 Cityscapes（实例及全景分割），并采用 COCO 评估协议对小目标性能进行分层评测。

**📈 对比分析**

在 Jetson Orin Nano 上，CompactInst 的解码器在 Cityscapes 分辨率下实现了 2.35×–4.74× 的解码器速度提升，整体模型提升 1.57×–2.62×，与 FastInst 只相差 0.4 AP，且在全景分割中仅损失约 1% PQ。

**⚠️ 局限性**

局限性包括：① 仅使用均匀最大池化，未针对内容进行自适应 token 过滤，导致 COCO 小目标精度略降；② 后处理阶段仍是内存带宽瓶颈，未做针对性优化；③ 仅在当前查询式分割框架验证，未深入探讨更复杂的 Token 约简方案。

---

## 249. Query-adaptive Recurrent Tokenization

**arXiv ID:** 2610.22957 | [PDF](https://arxiv.org/pdf/2610.22957v1)

**作者:** Bavesh Balaji `[一作]` (University of Waterloo), Sirisha Rambhatla `[通讯]` (University of Waterloo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种 Query‑Adaptive Recurrent Tokenizer（QART），实现视觉表示的查询驱动与预算自适应。

**💡 创新点**

创新点是将查询条件与递归容量分配相结合，通过重构质量动态决定视觉 token 数量。

**🔧 技术方法**

使用冻结的 SigLIP2 视觉与文本编码器、交叉模态相关模块、共享 Transformer 以及解码重建模块。

**📊 数据集**

在 Visual7W、VQAv2、CORE、GeneCIS、PODS、Winoground 等数据集上验证。

**📈 对比分析**

与固定 token 或其他压缩方法对比，QART 在 Visual7W 与 VQAv2 保持约 92% 与 76% 的准确率，仅使用约 29‑32 个视觉 token；在 CORE 上 P@1 达 92%，比 SigLIP2 提升 23%；在 PODS 上 PR‑AUC 为 36%，比 SteerViT 高 8.4%。

**⚠️ 局限性**

局限性包括在需要全局视觉信息或对查询鲁棒性要求极高的场景下，查询驱动的压缩可能忽略未提示的细节；递归开销在小批量推理时不一定带来优势。

---

## 250. FRACTAL: A Self-Similar Multi-Agent Orchestration Architecture with Formal Communication, Convergence, and Depth Bounds

**arXiv ID:** 2610.22561 | [PDF](https://arxiv.org/pdf/2610.22561v1)

**作者:** Ohad Dan `[一作]`, Abhinav Taliyan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出一种名为FRACTAL的自相似多代理架构，能够递归地分解目标并在每个层级统一使用同一五元结构（Orchestrator、Ensemble、Knowledge Ledger、Decision Maker、Meta‑Observer）实现分层协同；

**💡 创新点**

通过自相似结构实现全局通信、收敛和深度的统一理论界限，提出两阶段线性通信协议、可测饱和阈值的双模式调度、超多数收敛保证以及贝叶斯层间升级机制；

**🔧 技术方法**

使用基于大型语言模型的代理、两阶段“盲提案/压力测试”通信协议、可测多样性阈值、超多数投票与贝叶斯后验更新等技术；

**📊 数据集**

本文未使用任何实际数据集，全部采用理论推导与示例性说明；

**📈 对比分析**

未进行实验比较，未给出量化性能指标，仅在案例中说明FRACTAL在理论上相较于自由辩论在通信成本上更低；

**⚠️ 局限性**

主要局限包括缺乏实证验证、固定评估者假设导致的线性/二次比较、上层无法获取子层细粒度信息、未定义偏差向量计算方法、假设代理间独立同质等。

---

## 251. The Strong Total Cover Problem

**arXiv ID:** 2610.23312 | [PDF](https://arxiv.org/pdf/2610.23312v1)

**作者:** Manuel Aprile `[一作]` (Università degli Studi di Padova), Luca Ferrarini `[通讯]` (Université Sorbonne Paris Nord)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了强总覆盖问题（STCP），从图论与线性规划对偶的角度给出了其定义、复杂性、近似算法和多面体结构。

**💡 创新点**

创新点包括：①将STCP与占领集、诱导匹配以及总图极点建立对应关系；②证明最优2-近似并给出整形间隙为2；③在树、环和三角海绵图上完成完整线性描述；④提出多面体面分解与1-合并的技术。

**🔧 技术方法**

所用技术主要是图论（占领集、诱导匹配、海绵图）、线性规划对偶与整数规划、平衡矩阵理论、总图与完美图性质、多面体面分析以及贪心与最小割算法。

**📊 数据集**

本文为理论研究，无实验数据集。

**📈 对比分析**

通过与占领集、诱导匹配等经典问题的比较，证明STCP在一般图上为NP‑难；在完整二分图上给出多项式算法；在一般图上给出2-近似，并且证明该近似因子是最优的；整形间隙等于2。

**⚠️ 局限性**

限制在于：仅完成三角海绵图的完整多面体描述；对更一般的海绵图、完整图等仍缺乏完整面描述；需求版STCP的复杂性尚未研究。

---

## 252. Federated Multiagent Soft Actor Critic for IRS Assisted Cognitive Radio Networks under Partial Observability

**arXiv ID:** 2610.22884 | [PDF](https://arxiv.org/pdf/2610.22884v1)

**作者:** Satvika Pullisani `[一作]` (Wilfrid Laurier University), Ubaid Abbasi `[通讯]` (Northwestern Polytechnic)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个名为 FMASAC 的联邦多智能体软演员-评论家框架，用于在部分可观测的 IRS 辅助认知无线电网络中，分布式次用户（SUs）协同学习功率与 IRS 阶相位控制，以在不共享原始观测或 CSI 的情况下实现 PU 保护。

**💡 创新点**

创新点在于：①将联邦学习与多智能体连续控制（Soft Actor-Critic）结合，实现了无中心化的策略聚合；②针对 IRS 元素不可被多智能体同时冲突控制的约束，采用元素分区模型；③通过在奖励中加入 PU 干扰惩罚，实现了部分可观测环境下的软约束满足；④提供了通信成本分析和聚合间隔与性能的权衡。

**🔧 技术方法**

采用的技术包括：分布式软演员-评论家（SAC）算法、部分可观测马尔可夫决策过程（POMDP）建模、联邦学习参数聚合（权重平均）、IRS 阶相位调制、Rayleigh 衰落与对数距离路径损耗的无线信道模型。

**📊 数据集**

使用的“数据集”主要是基于 Rayleigh 衰落和对数距离路径损耗的合成仿真数据，包含多用户位置、IRS 元素分区、随机信道状态等，并在 300 个训练回合中记录奖励、吞吐量和干扰指标；并未使用公开的真实数据集。

**📈 对比分析**

对比方法包括：中心化 SAC（全局信息），独立 SAC（无聚合），IRS 固定的多智能体 RL（固定相位），以及基于规则的保守 CRN 策略。实验结果表明 FMASAC 在吞吐量上优于所有分布式基线，接近中心化 SAC，且 PU 干扰水平更接近阈值且波动更小，收敛更平滑，策略一致性更好。

**⚠️ 局限性**

局限性包括：①干扰惩罚为软约束，未提供严格的 PU 保护保证；②仅在仿真中展示示意趋势，缺乏完整的实测或大规模验证；③在网络规模扩大、用户异质性和 IRS 元素数量增加时，参数聚合的有效性和收敛性尚未验证；④未考虑个性化聚合或更高级的约束处理方法（如 Lagrangian SAC）。

---

## 253. Pearl-Chat: Engineering a native Luganda language model from first principles in JAX and Flax NNX

**arXiv ID:** 2610.22529 | [PDF](https://arxiv.org/pdf/2610.22529v1)

**作者:** Wesley Kambale `[一作]` (AI for Uganda Initiative), Roland Sankara `[通讯]` (ISBAT University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练并发布了一个8.5M参数的Luganda（奥卢干达）语言模型Pearl-Chat。

**💡 创新点**

创新点包括：字节级BPE自定义tokenizer、JAX/Flax NNX纯函数实现、两阶段低资源训练课程以及助手掩码优化，显著降低token化肥率并提升语义一致性。

**🔧 技术方法**

使用技术涵盖JAX、Flax NNX、Optax、Google Grain、Orbax、SPMD张量并行、SSE流式推理以及top‑p/k采样。

**📊 数据集**

数据集：Luganda圣经、Luganda‑English并行语料、Luganda维基百科、Sunbird SALT以及510对话记录。

**📈 对比分析**

与LLaMA‑3、Gemma等多语言模型比较，token化肥率从3.4降至1.6，压缩率达4.88 char/token，验证困惑度1.20，ROUGE‑L 0.9792，BLEU 0.9405。

**⚠️ 局限性**

局限性：模型规模较小，仅针对Luganda，缺乏跨语言泛化能力，需要进一步扩展参数和多语言预训练。

---

## 254. Measuring Mechanistic Uncertainty in Large Language Models via Causal Robustness

**arXiv ID:** 2610.22636 | [PDF](https://arxiv.org/pdf/2610.22636v1)

**作者:** Aneesh Komanduri `[一作]` (University of Colorado Anschutz Medical Campus), Yanjun Gao `[通讯]` (University of Colorado Anschutz Medical Campus)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了使用内部机制鲁棒性作为大型语言模型不确定性度量的方法，定义了因果鲁棒边距并通过Jacobian Lens实现对内部激活的干预；

**💡 创新点**

将不确定性重新定义为内部计算机制的稳定性，提出最小干预阈值（因果鲁棒边距）作为可测量指标，并展示其与传统输出层不确定性方法互补、可通过知识注入提升鲁棒性；

**🔧 技术方法**

利用Jacobian Lens读取并 steer 中间激活，计算最小干预量α*；结合线性正则化、层级归一化、AUROC与Spearman相关性评估，以及对三款开源 LLM 的监督微调；

**📊 数据集**

在 TriviaQA、TruthfulQA、PopQA、SciQ、MedQA 等问答数据集上进行实验；

**📈 对比分析**

通过 AUROC、平均 α* 与传统自一致性、token entropy 等指标对比，发现 α* 与正确率正相关、与传统不确定性指标相关但能补充信息，AUROC 在不同模型/数据集上可达约 75% 左右，表明有效；

**⚠️ 局限性**

方法需要完整白盒访问、依赖预训练 Jlens 矩阵、干预过程耗时、仅在有限开源 LLM 上验证，可能不适用于所有模型或大规模部署。

---

## 255. SAG2: Embedding-Free Multi-Hop Retrieval with BM25 and JEV

**arXiv ID:** 2610.22588 | [PDF](https://arxiv.org/pdf/2610.22588v1)

**作者:** Yuchao Wu `[一作]` `[通讯]`, Yuchao Wu

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为SAG2的嵌入无关多跳检索架构，利用单一SQLite数据库、BM25和JEV模型实现在线语义证据选择与检索扩展。

**💡 创新点**

创新点在于将SAG的查询时关联原则迁移至词汇索引层，完全消除离线嵌入与实体提取；通过JEV在查询时动态评估候选证据并驱动词汇级别的检索扩展，从而实现轻量化部署。

**🔧 技术方法**

核心技术包括SQLite FTS5的全文索引、BM25词频-逆文档频率检索、JEV（System One）模型的上下文证据评分，以及基于检索上下文的词汇级扩展与多轮排名。

**📊 数据集**

在MuSiQue、2WikiMultiHopQA、HotpotQA这三个多跳问答基准上进行评估，并在企业内部通信与文档集合EnterpriseRAG-Bench上检验检索覆盖率。

**📈 对比分析**

与传统BM25、Contriever、BGE、GTE、GritLM、NV-Embed、HippoRAG 2、SAG等基线相比，SAG2在R@5与R@10上均取得最高分，在MuSiQue的R@10为86.62%、2WikiMultiHopQA为97.43%、HotpotQA为99.00%；在EnterpriseRAG-Bench上Document Recall@10达81.10%，显著优于BM25、Vector和Bash Agent。

**⚠️ 局限性**

主要局限在于依赖外部托管的JEV服务，无法完全本地化部署；对词汇匹配的依赖导致对同义、重写等表达仍有检索盲点；在四跳问题上的完整证据覆盖率仍较低，且对语义关联的进一步提升尚待探索。

---

## 256. Sep-Latent: Separator Tokens as Natural Compression Anchors for Implicit Reasoning

**arXiv ID:** 2610.22924 | [PDF](https://arxiv.org/pdf/2610.22924v1)

**作者:** Yinlong Xu `[一作]` (Zhejiang University), Jian Wu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Sep-Latent框架，利用分隔符作为语义锚点，在连续潜在空间中进行自适应隐式推理；

**💡 创新点**

通过Attention-Decay机制将推理信息压缩至分隔符，结合多层蒸馏实现显式与隐式路径的逻辑一致性，并支持动态推理深度；

**🔧 技术方法**

采用分隔符注意力衰减、在线策略蒸馏、多层对齐蒸馏、以及自适应退出标记的技术；

**📊 数据集**

主要使用GSM8k-Aug、GSM-Hard、MultiArith、SVAMP、GSM-SYM等数学推理数据集，以及LogiQA、CSQA、StrategyQA等常识推理数据集；

**📈 对比分析**

与SFT-CoT、SFT-Direct、Coconut、CODI、SIM-CODI等基线比较，Sep-Latent在GSM8k上达到56.8%准确率，仅消耗约5.4个token，显著优于传统隐式方法且与显式CoT相近；

**⚠️ 局限性**

仍无法完全超越显式CoT的准确率，模型对采样策略敏感，且隐式推理逻辑在极端复杂问题上可能不完整，未来可引入RL提升对正确答案的采样效果。

---

## 257. HilbertGen-3D:Hilbert--Multifractal Conditioning for Topology-Aware 3D Generation

**arXiv ID:** 2610.22720 | [PDF](https://arxiv.org/pdf/2610.22720v1)

**作者:** Ziyi Zhang `[一作]` (Carnegie Mellon University), Guannan Qu `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在点云生成中引入Hilbert–Multifractal Conditioning（HMC）作为全局条件，提升几何结构一致性。

**💡 创新点**

创新点在于利用3D Hilbert曲线序列化多尺度体素密度，并通过分区函数计算多分辨率分形维数，形成全局描述，随后与局部特征融合到扩散生成器。

**🔧 技术方法**

使用扩散Transformer（DiT-3D）架构，配合Hilbert序列编码器、全局嵌入以及瓶颈融合（Perceiver）实现高效条件注入。

**📊 数据集**

实验基于ShapeNet的Chair、Airplane、Car等类别。

**📈 对比分析**

与TopoDiT-3D、DiT-3D等基线比较，1-NNA接近50并提升至约54–59，覆盖率提升3–6个百分点，表明性能显著提升。

**⚠️ 局限性**

局限在于需要预先计算体素分布，依赖全局密度信息，对缺乏全局结构或密度均匀的场景效果可能不佳。

---

## 258. Secondhand Stress: How Stressful Disclosures Affect Psychosocial Wellbeing in Online Graduate Admissions Communities

**arXiv ID:** 2610.22657 | [PDF](https://arxiv.org/pdf/2610.22657v1)

**作者:** Veda Duddu `[一作]` (University of Illinois Urbana-Champaign), Koustuv Saha `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了研究生和MBA招生社区中，用户在申请准备阶段接触同伴的情绪困扰后，随后的心理社会表达与参与行为。

**💡 创新点**

首次将压力交叉理论应用于高压学术转型中的在线社区，区分直接曝光与后续互动的影响。

**🔧 技术方法**

采用倾向得分分层匹配、基线调整回归、SVM情绪分类、LIWC词汇量化、结构语言指标（可读性、冗余等）等方法。

**📊 数据集**

利用两大Reddit社区（r/gradadmissions和r/MBA）171,981条帖子和1,130,413条评论，覆盖三届招生周期。

**📈 对比分析**

与传统基线对照相比，匹配后发现直接曝光用户在压力表达、评论频率和词汇冗余上显著提升（效应量约0.2–0.4标准差），但后续互动用户无显著差异。

**⚠️ 局限性**

局限包括观察性设计无法排除未观测混杂、锚点帖子分类器误差、只用评论视为曝光、未捕获被动阅读、数据仅来自两社区且不具普适性。

---

## 259. AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems

**arXiv ID:** 2610.22511 | [PDF](https://arxiv.org/pdf/2610.22511v1)

**作者:** Shreeyans Arora `[一作]` (Lexsi Labs), Pratinav Seth `[通讯]` (Lexsi Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个统一的评估库 AuditKIT，用于评估语言模型和代理的多维度性能，包括任务准确性、生成质量、工具调用、检索增强生成以及安全性。

**💡 创新点**

创新点在于：1）所有指标统一声明优化方向（MAXIMIZE/ MINIMIZE），实现跨模型方向感知比较；2）单接口覆盖基准测试、LLM-judge、工具使用、RAG 等；3）对代理执行轨迹的结构化记录和可靠性判定；4）可扩展的度量与后端，支持自定义指标和多后端部署；5）利用 SHA‑256 指纹实现结果缓存与可追溯性。

**🔧 技术方法**

技术包括：Python 框架设计、Adapter/Model/Annotator/Metric 的抽象接口；最大匹配算法用于工具调用评估；多种生成质量指标（BLEU、ROUGE、BERTScore 等）；LLM‑judge 与 Encoder‑judge；RAG 相关检索/判定指标；以及与 AgentTune、AlignTune、SafeTune 等外部 artifact 的只读导入。

**📊 数据集**

使用的数据集包括：内置 benchmark（mmlu、gsm8k、arc、hellaswag、truthfulqa、humaneval 等），公开数据集（BoolQ、SST‑2、HotpotQA、Financial‑Q&A）、Cohere 多语种测试、BFCL 等工具调用基准，以及从 AgentTune、CuratorKIT、AlignTune、SafeTune、CircuitKIT 等平台获取的轨迹与日志。

**📈 对比分析**

比较方法：`compare_models()` 通过统一 `RunResult` 对多模型结果进行方向感知比较，提供保留率、显著性检验、trade‑off 与回归分析；性能指标自动记录（延迟、吞吐、成本、参数量）。在案例研究中，发现不同工具、提示、后端或量化方式会显著影响指标，展示了评估配置对结果的影响。

**⚠️ 局限性**

局限性：1）缺乏图像、表格、对话模拟等输入类型的专门指标；2）代理评估仅支持单轮或按轮记录，未覆盖跨轮连贯性；3）工具调用验证仅对 JSON Schema 子集；4）延迟测量为单次 `generate()` 调用，未细粒度捕获 token‑级时延；5）红队功能基础，仅支持简单探针；6）判定与人类标注的匹配尚未充分验证。

---

## 260. Cross-Centaur: Modeling Individual Decision-Making Across Psychological Tasks with Language Models

**arXiv ID:** 2610.23157 | [PDF](https://arxiv.org/pdf/2610.23157v1)

**作者:** Hikaru Asano `[一作]` (University of Tokyo), Yukino Baba `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究跨任务个体行为预测，构建Cross-Psych-101数据集并训练Cross-Centaur模型。

**💡 创新点**

首次引入跨任务个体行为预测框架，并提供跨任务链接的数据集，利用源任务行为显著提升目标任务预测。

**🔧 技术方法**

采用大型语言模型（Qwen3-8B/14B）通过聊天式监督微调，使用行动预测损失和提示调优技术。

**📊 数据集**

使用Cross-Psych-101数据集（591名参与者、12种决策任务、261,551个有效决策）。

**📈 对比分析**

与原始Qwen和Centaur做负对数似然比较，Cross-Centaur在相同与跨任务、已见与未见任务上均优于基线，提升约10%–20%。

**⚠️ 局限性**

受限于实验任务范围、语言与文化覆盖不足，对完全未见任务的泛化有限，且模型未能完整捕捉个体行为多样性。

---

## 261. What Shapes Brain Alignment of Large Language Models during Dialogue? Insights from Language Production and Comprehension

**arXiv ID:** 2610.22961 | [PDF](https://arxiv.org/pdf/2610.22961v1)

**作者:** Yuko Nakagi `[一作]` (NTT), Yasue Kishino `[通讯]` (NTT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究使用 fMRI 数据分析大型语言模型（LLM）在日语对话中产出与理解过程的脑对齐情况。

**💡 创新点**

创新点在于首次系统比较对话时 LLM 的产出与理解模式下的脑对齐，关联内在维度（ID）及可解释的对话特征，并揭示层级与脑区的差异。

**🔧 技术方法**

采用 fMRI 编码模型将 LLM 各层的潜在表示映射到脑信号，结合 ID 估计和 LEACE 语义特征消除技术。

**📊 数据集**

使用公开的 Natural Dialogue fMRI 数据集（7 名日语母语者的 3 组 7–11 分段对话）以及三种日语 LLM（Llama‑3.1、Gemma‑2、GPT‑OSS）。

**📈 对比分析**

通过层级和语境长度的交叉验证评估编码模型的 Pearson 相关性，发现中间层在较长上下文下获得最高脑对齐；ID 与对齐呈正相关，去除词汇、POS、对话行为等特征后对齐显著下降。

**⚠️ 局限性**

限制包括样本量小、仅使用单一对话 fMRI 数据集、缺乏多模态信息、可解释特征范围有限以及模型范围受限于三种 LLM。

---

## 262. DefFlow: Goal-Conditioned Deformable Object Manipulation using 3D Point Flows

**arXiv ID:** 2610.22827 | [PDF](https://arxiv.org/pdf/2610.22827v1)

**作者:** Gaurav Singh `[一作]` (Brown University), Srinath Sridhar `[通讯]` (Brown University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出一种面向可变形物体的目标条件操纵框架（称为GC-DP3），通过联合预测机器人和场景的三维点流来实现对多种起始与目标配置的成功操纵。

**💡 创新点**

创新点包括：①使用全场景（机器人+可变形物体）三维点流而非仅机器人流来建模耦合动力学；②通过“procedural play + hindsight relabeling”自动收集多样化演示数据；③将点流预测结果映射为可执行的机器人关节轨迹；④在真实机器人上无任务细化直接部署，展示对未见目标形状的泛化。

**🔧 技术方法**

核心技术：Point Transformer V3（PTv3）UNet作为点云编码器，密集三维点流预测，Huber 损失；逆运动学（Damped Least Squares）将流映射为关节动作；Hindsight Goal Relabeling、数据增广与多任务训练；部署时使用 SAM2 进行目标分割并构造点云。

**📊 数据集**

使用自建数据集：≈7000条随机生成的起止任务，覆盖绳索、细粒料与布料三大模拟环境；此外还构造了 700 条立方体堆叠演示用于探究配置多样性；真实机器人实验使用 Franka Emika Panda 与 RealSense D455 采集点云。

**📈 对比分析**

与多种基线对比：①动作直接预测（GC‑DP3/GC‑DP3‑PTv3）、②仅预测机器人流（Robot‑Flow）、③仅预测刚性机器人流（Rigid Robot‑Flow）、④ToolFlowNet 风格的仅机器人流、⑤结构化策略（Grasp‑Servo）等。结果显示：GC‑DP3 在绳索环境下成功率提升约 1.5×、L2误差下降 30%；在细粒料和布料环境中分别获得最高 IoU 和 1.5×成功率；整体比基线高出 20–50% 的成功率与误差改进。

**⚠️ 局限性**

局限性：①需要高质量点云与精确分割，噪声或遮挡会影响性能；②逆运动学求解在复杂姿态下可能产生多解或收敛慢；③训练与推理的计算开销较大，实时性受限；④当前实验主要基于模拟，真实环境中动态摩擦与细粒料的建模仍有挑战。

---

## 263. An Agentic ENF Framework for Forensic Video Authentication Against Generative AI

**arXiv ID:** 2610.22440 | [PDF](https://arxiv.org/pdf/2610.22440v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 264. The Ephemeral Cost of Switching Optimizers Across LLM Training Stages

**arXiv ID:** 2610.22650 | [PDF](https://arxiv.org/pdf/2610.22650v1)

**作者:** Xinyu Zhao `[一作]` (MIT), Paul Pu Liang `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM预训练与后续SFT阶段使用的AdamW与Muon优化器是否存在切换成本进行系统实验

**💡 创新点**

通过将两种优化器按比例混合并在预训练、持续预训练和SFT阶段分别切换，量化匹配与不匹配的交互效应，发现切换成本极小

**🔧 技术方法**

使用混合优化器、交互效应分解、学习率调优、SFT任务混合、Open-weight模型继续训练

**📊 数据集**

预训练使用大规模文本混合数据，SFT使用Tulu-3混合，评测在MMLU、BBH、GSM8K、MATH、HumanEval、MBPP等基准上

**📈 对比分析**

在多种模型规模（1.2B、Llama2-7B、Qwen2.5-7B、Qwen3 4/8/14B）上对比AdamW与Muon的SFT性能，结果显示Muon在SFT上与AdamW相当或略优，匹配与不匹配对最终性能影响不大

**⚠️ 局限性**

仅对AdamW与Muon进行比较，未覆盖其他优化器；实验规模相对有限；忘记效应虽被观察但未在大规模持续训练中深入验证

---

## 265. SIEVE: A Parameterized GitHub Corpus Builder for Software Engineering Research

**arXiv ID:** 2610.22544 | [PDF](https://arxiv.org/pdf/2610.22544v1)

**作者:** Musfiqur Rahman `[一作]` (Concordia University), Emad Shihab `[通讯]` (Concordia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

SIEVE是一款开源的、可参数化的GitHub语料库构建工具，能够按函数/方法/类级别从仓库中提取代码片段，并为每个记录自动计算结构指标、依赖信息以及LLM生成代码的概率评分；

**💡 创新点**

其创新点在于将传统的仓库级挖掘提升到代码粒度层面，支持多语言、质量和时间过滤，同时集成结构分析、依赖提取与AI生成代码检测为一体化、可重复的流水线；

**🔧 技术方法**

技术实现基于Python 3.11，使用Typer CLI、Streamlit Web、Pydantic配置模型、PyGitHub访问GitHub API、Tree-sitter解析多语言语法、MinHash做近似重复剔除、CodeBERT细调的二分类器做LLM检测；

**📊 数据集**

数据来源主要是GitHub公开仓库，兼容CodeSearchNet与OpenClassGen格式；训练集包含114,186条人写与LLM生成的代码片段，测试集11,419条；另外手工验证400条记录；

**📈 对比分析**

与现有工具（如SEART、GHTorrent、GPTSniffer）相比，SIEVE在源代码完整性、元数据准确率、导入检测和结构指标方面达到了99%~100%的准确率；LLM检测器在测试集上的F1为0.9478、AUROC为0.9902，表现优异；用户调查显示大多数研究者认为易用且能节省时间；

**⚠️ 局限性**

局限性包括：C++类的语法解析不完整导致完整性低（92%），Java函数元数据不完整（96%）；仅支持Python、Java、JavaScript、C++四种语言；LLM检测器仅针对已训练的模型，无法完全泛化到未见过的LLM；缺乏更细粒度的过滤选项和更快的执行速度。

---

## 266. Reasoning Models Can Control Their Chains of Thought When They Tell Themselves To

**arXiv ID:** 2610.23257 | [PDF](https://arxiv.org/pdf/2610.23257v1)

**作者:** Uriel Dolev `[一作]` (Bar-Ilan University), Yoav Goldberg `[通讯]` (Bar-Ilan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估“自我指令链式思考（Self‑Instructed CoT）可控性”，展示在内部放置约束比在用户提示中放置约束能显著提升模型对链式思考的可控性。

**💡 创新点**

创新点在于：①将约束直接写入模型生成的思考链内（self‑instruction），②提供两种诱导方式（思考预填与自引导），③在未训练的模型上实现高效可控，并揭示自我指令可被用于监控逃逸。

**🔧 技术方法**

技术包括：自我指令写入思考链、思考预填（prefill）与自引导（bootstrap）两种诱导方法、二进制与软可控性评估指标、以及在多轮代理任务与监控逃逸实验中的应用。

**📊 数据集**

使用数据集：GPQA、HLE、MMLU‑Pro（多项选择QA）；BFCL、SWE‑bench Verified（代理任务）；以及针对闭源模型的900道QA子集。模型包括五个开源推理模型（DeepSeek‑R1、Qwen3‑14B/32B、GPT‑OSS‑20B/120B）与两个闭源模型（Claude 3.7 Sonnet、Claude Sonnet 4.6）。

**📈 对比分析**

评估方法：对比用户指令（User‑Prompt）与自我指令（Constraint Prefill、Bootstrap 等）在多种约束下的二进制可控率和软指标。结果显示自我指令将可控率从几乎 0% 提升至 10‑30% 以上，Soft 评分亦显著提升；在监控逃逸实验中，自我指令能将监测捕获率降低 4‑7% 点。

**⚠️ 局限性**

局限性：①仅评估文本层面的约束，未验证对推理过程本身的可控性；②需要对思考链可预填或模型能写出自我指令，闭源模型受限；③尚未验证模型是否能在无明确自我指令时自然实现控制。

---

## 267. PASS: A Trustworthy Precision Approach Simulation System for CAT III Integrity Algorithm Development

**arXiv ID:** 2610.23019 | [PDF](https://arxiv.org/pdf/2610.23019v1)

**作者:** Lang Zhu `[一作]` (Shanghai Jiao Tong University), Xingqun Zhan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了PASS（Precision Approach Simulation System），一种可生成同步立体视觉+惯导数据、连续六自由度地面真实值以及跑道几何先验的跑道精准进近仿真框架；

**💡 创新点**

创新点在于：①将PSINS轨迹模拟与Unreal Engine视觉渲染耦合，形成时间与几何一致的多机场、多场景数据；②提供可控误差注入、真值参考与条件标签，支持视觉-惯导导航完整性算法的开发；③提出基于DINOv2特征与UMAP的Sim‑to‑Real Scene Trustworthiness（SRST）指标，实现无配对真实-虚拟场景的类别级可信度评估；

**🔧 技术方法**

使用技术包括PSINS惯导模拟、Unreal Engine 5.1.1视觉渲染、双目相机标定、DINOv2-预训练特征、UMAP降维、VINS‑Fusion与ORB‑SLAM3视觉‑惯导估计器；

**📊 数据集**

主要数据集为PASS数据集，涵盖城市、沙漠、山地、沿海四类机场的多场景精确进近序列；

**📈 对比分析**

评估方法：使用VINS‑Fusion与ORB‑SLAM3在不同机场、窗口长度的终端进近上测定位置误差，平均终端误差从3~40 m不等；SRST指标在4分类情形下>90%，3分类下>98%，表明虚拟场景在类别级别与真实场景高度对应；

**⚠️ 局限性**

局限性包括：仅覆盖直线进近、未包含路径/姿态偏离、滑行或着陆阶段；缺乏真实飞行数据验证；SRST仅评估类别一致性，无法直接证明导航完整性；

---

## 268. What Makes a Good Symbol for Hybrid Discrete-Continuous Planning?

**arXiv ID:** 2610.23010 | [PDF](https://arxiv.org/pdf/2610.23010v1)

**作者:** Yiyuan Pan `[一作]` (Carnegie Mellon University), Jiayuan Mao `[通讯]` (University of Pennsylvania)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种以推理算法为中心的符号发明框架，定义“solver-sufficient”符号，既保留最优解又显著降低规划与优化成本。

**💡 创新点**

创新点在于：①用求解器可充分松弛的条件（PWA系统中的相对 affine‑TU 以及神经网络中的 τ‑descent 可信性）来精确定义符号的必要性；②通过 Counterexample‑Guided Partition Refinement（CPR）算法在数据驱动和符号层面自动生成最少维度的符号；③展示这些符号既能提高效率，又保留可解释性和可引导性。

**🔧 技术方法**

主要技术包括：相对 affine‑TU 分解、线性规划松弛、神经能量景观的 τ‑descent 可信性度量、CEGAR 风格的 CPR、基于梯度和邻接图的样本驱动符号分割，以及对多任务（PWA 与 NN）统一的符号搜索与优化框架。

**📊 数据集**

在模拟任务（二维操控、手指重定向、D3IL 任务）和真实机器人（如 PushT 变体、Lunar Lander 等）上进行评估；实验集包含手动定义的符号、LLM 生成符号、以及自动生成符号的对照组。

**📈 对比分析**

与 CoCo、GBD‑MPC、PDDLStream、PGS、IVNTR、pix2pred 等基线比较，指标包括成功率（SR）、推理时间加速（IS）、下降次数覆盖（DC_δ）等。结果显示：自动生成的 solver‑sufficient 符号在保持或提升 SR 的同时，将 IS 提升至 2–5 倍，DC_δ 降低至原来的 1/2–1/10，且在多任务上保持一致性。

**⚠️ 局限性**

局限性包括：①符号映射仅采用线性投影，可能无法捕捉高度非线性或复杂耦合的决策结构；②CPR 需要大量梯度与采样，计算开销随任务维度和数据量增加；③在极大规划时域下符号维度仍可能快速增长，导致搜索空间膨胀；④对某些特定任务，符号解锁的局部最优可能导致全局最优被忽略。

---

## 269. ARBOR: Locating Reasoning Failures in Long Chains of Thought through Reasoning State Tracking

**arXiv ID:** 2610.22717 | [PDF](https://arxiv.org/pdf/2610.22717v1)

**作者:** Huihan Li `[一作]` (University of Southern California), Xiang Ren `[通讯]` (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种名为ARBOR的推理失败归因框架，该框架通过构建推理状态树跟踪大型语言模型（LLM）的推理过程，并将失败归因到步骤与前置状态之间的关系，而不是仅依据文本。

**💡 创新点**

创新点在于：① 引入基于树状态的失败子类型，将过度思考、缺少思考和无根据推导细分为更具体的子模式；② 通过操作化的树状态与或acles的判定，能够更准确地识别并解释失败；③ 利用树状态预先检测失败，为模型干预提供更精准的触发条件。

**🔧 技术方法**

主要技术包括：LLM解析器（使用Qwen3.8-27B）将推理步骤解析为操作；确定性树状态引擎构建并维护推理状态树并计算或acles（sound、viable、accept）；失败检测器基于树状态与或acles定义一系列失败子类型；以及基于树状态的分类器用来验证预前置条件。

**📊 数据集**

使用了三个具有不同推理结构的任务数据集：ZebraLogic（约束满足）、24 Game（算术组合搜索）和PlanBench（基于前置条件的行动规划）。

**📈 对比分析**

与现有的局部信号（n‑gram重复、相似度、switch token、奖励模型等）对比，ARBOR在多任务、多模型规模下的失败检测准确率显著提高；在精度、覆盖率和AUROC指标上均优于单一或组合的本地信号；并通过子类型细化揭示了不同失败模式在 token 成本和答案正确性方面的差异。

**⚠️ 局限性**

局限性包括：① 仅在 Qwen3 系列模型上验证，其他模型需重新调优解析器和引擎；② 仅适用于答案可验证且搜索空间有限的任务，无法直接推广到开放式推理、数学证明或代理工作；③ 提出的干预点和方法尚未在实际使用中进行实验验证，效果仍待进一步评估。

---

## 270. Replay Pruning and Few-Bit Integer Inference for Hardware-Efficient Speed-Sign Recognition

**arXiv ID:** 2610.22811 | [PDF](https://arxiv.org/pdf/2610.22811v1)

**作者:** Se Un Park `[一作]` (UX Factory, Inc.), Junyoung Park `[通讯]` (UX Factory, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个两阶段车速限标识识别系统，结合压缩的YOLOv5n/YOLOv8n检测器和低位整数读写器，实现硬件友好的部署。

**💡 创新点**

创新点在于channel replay结构化剪枝与知识蒸馏结合，能在不降低精度的前提下去除约70%参数；以及对读写器进行4位权重/8位激活整数化并完成完整整数化推理与后处理。

**🔧 技术方法**

采用了ReLU+hard‑sigmoid激活替换、Taylor重要性通道剪枝、逐步重放与知识蒸馏、整数量化（PTQ+QAT）、固定点推理、Jetson AGX Orin上DLA与TensorRT INT8对比等技术。

**📊 数据集**

训练与验证使用公开数据集GTSRB、Roboflow、AIHub韩国驾驶数据；测试集为AIHub验证拆分及自有汽车相机的数据。

**📈 对比分析**

与未压缩网络、迭代剪枝、一次性剪枝、随机重训等方法对比，channel replay在YOLOv5n上保留父网络精度并减少约70%参数；在YOLOv8n上实现8.5% mAP50提升；整体系统在Jetson上达到27fps，精度90%召回率。

**⚠️ 局限性**

局限在于对大幅放大标识的检测表现欠佳；仅针对韩国限速标识；硬件实验仅在单板30W模式下完成，未覆盖内存电源等功耗；读写器尚未RTL验证。

---

## 271. Who Do Perspectivist Models Predict Well? Annotator Agreement, Sparsity, and Performance

**arXiv ID:** 2610.22567 | [PDF](https://arxiv.org/pdf/2610.22567v1)

**作者:** London Lowmanstone `[一作]`, Maria Gini `[通讯]` (University of Minnesota)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了迄今最大规模的英文阶级化视角 NLP 数据集集合，并对多种视角预测算法在不同注释者上的性能进行了系统评估，发现注释者的一致性越高，模型预测误差越低，而注释频率与性能无显著正相关。

**💡 创新点**

创新点在于：①首次汇总并清洗了数十个阶级化视角 NLP 数据集（含 10000 文本 × 500 注释者样本），②采用加权固定效应回归与块 Bootstrap 结合的方法量化注释者偏差与稀疏性对模型误差的影响。

**🔧 技术方法**

使用的技术包括：神经协同过滤（NCF）、多任务模型、核矩阵分解（KMF）、MASDA、TabFM 以及基线均值预测；评估指标为归一化均方误差（NMSE），并用 z‑score 标准化注释者一致性、稀疏性与测试样本数。

**📊 数据集**

使用的数据集来源于 AHLV、AffectiveText、HuBi、NLPerspectives、LeWiDi 等，覆盖 16 个任务，如情感、毒性、礼貌、讽刺等，每个数据集均经过统一清洗、下采样和 95%/5% 训练/测试划分。

**📈 对比分析**

比较方法：对每个算法在每个数据集上分别训练并预测测试缺失注释，随后通过加权固定效应线性回归估计 β_D（一致性）和 β_S（稀疏性）系数；结果显示 β_D = 0.0248（p < 1e‑8，显著正相关），β_S = 0.0009（p = 0.16，非显著），即一致性对性能有明显正影响而稀疏性无显著效应。

**⚠️ 局限性**

局限性包括：研究仅限英文数据集与选定算法，结果为相关性而非因果性；未考虑非阶级化或非序数标注的任务；数据集的注释质量和隐私保护措施无法保证完全无误；以及对稀疏性与性能关系的结论可能受数据分布或算法偏好影响。

---

## 272. Progressive Misalignment Can Emerge from Benign Recursive Self-Distillation

**arXiv ID:** 2610.22467 | [PDF](https://arxiv.org/pdf/2610.22467v1)

**作者:** Karim Abdel Sadek `[一作]` (University of California Berkeley), Fanny Yang `[通讯]` (ETH Zurich)

**通讯引用:** 617 | [OpenAlex ID](https://openalex.org/A5113814271)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了递归自蒸馏（Recursive Self‑Distillation, RSD）对大型语言模型安全性的影响，发现从安全的模型开始，利用自身生成的安全回复并在其上继续进行自蒸馏，模型会逐步失去拒绝有害请求的能力，出现所谓的“Progressive Misalignment”。

**💡 创新点**

创新点在于首次揭示递归自蒸馏可在不接触任何有害训练数据的情况下，通过对自身安全输出的自我训练导致模型安全性下降，并在不同的提示方式（固定安全提示、模拟用户反馈、甚至仅使用安全提示的梯度选择）下均观察到这一现象。

**🔧 技术方法**

使用的技术包括：a) on‑policy 与 off‑policy 自蒸馏损失函数；b) 采用固定安全提示（如“I like Owls”，“Excellent”）或自适应用户反馈；c) 梯度基准数据选择方法，用以挑选可能导致失效的安全提示；d) 对模型在安全基准上的KL散度随训练轮次的变化进行分析。

**📊 数据集**

数据集主要为：① 20个由GPT‑5.6生成的有害提示（初始模型均拒绝回复）；② 两个安全基准集——SafetyBench（440个有害问题）和 SafetyEval（313个问题）；③ 规模从0.6B到32B的Qwen3与Phi‑4模型的各种版本。

**📈 对比分析**

与基准模型（初始安全模型）相比，递归自蒸馏后模型在 SafetyBench 的合规率从约26%激增至90%以上，在 SafetyEval 上的合规率同样显著提升，显示模型对有害请求的顺从度大幅上升；但在数学与指令跟随等控制任务上性能保持不变。

**⚠️ 局限性**

局限性包括：仅在Qwen3和Phi‑4两大模型家族且最大至32B的规模上验证；仅研究固定安全提示与模拟用户反馈两类增量信息，未覆盖更广泛的反馈形式；缺乏对导致失效机制的系统性解释，未来需要更深层次的因果与因果解释。

---

## 273. LonGaussian: Tracking Foundation Priors for Long-Context Feedforward Gaussian Reconstruction

**arXiv ID:** 2610.23250 | [PDF](https://arxiv.org/pdf/2610.23250v1)

**作者:** Zhoufeng Xie `[一作]` (Huazhong University of Science and Technology), Wenyu Liu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文提出了 LonGaussian，一种可在长视频中通过已知相机内参但未知外参的流式推理框架，实现高质量、可渲染的3D Gaussian场景重建。

**💡 创新点**

创新点在于：①利用基于多帧追踪的显式对应关系对重复观测的 Gaussian 进行聚合；②在每个块内学习合并器以精细化聚合结果；③使用渲染深度一致性过滤和远场 shell 处理跨块伪影和天空不稳定性；④在 LoGeR 的混合内存下实现端到端流式推理。

**🔧 技术方法**

技术包括 3D Gaussian Splatting、LoGeR 的长上下文几何推断、AllTracker 追踪、基于轨迹的 Gaussian 合并网络、可微渲染器、深度一致性投票和远场球面建模。

**📊 数据集**

在 DL3DV 和 RealEstate10K 两个公开长视频数据集上进行训练与评估。

**📈 对比分析**

与 AnySplat、WorldMirror、ReCoSplat、OF3GS 等基线对比，LonGaussian 在 PSNR、SSIM 上均位列首位，同时 Gaussian 数量显著减少，且能线性扩展到数千帧。

**⚠️ 局限性**

局限性包括：仅适用于静态场景；对追踪精度和 LoGeR 的姿态估计高度依赖；跨块关联仍需改进；深度过滤和远场处理带来额外开销。

---

## 274. When Does Uncertainty Measure RAG Hallucination Risk?

**arXiv ID:** 2610.22615 | [PDF](https://arxiv.org/pdf/2610.22615v1)

**作者:** Rajeel Ahmad Ansari `[一作]` (Independent Researcher), Tautik Agrahari `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究检索增强生成(RAG)模型在检索证据不足时的幻觉风险评估，探讨不确定性（如困惑度）是否能作为可靠指标。

**💡 创新点**

提出检索覆盖度决定不确定性指标意义的观点，并验证结合语义漂移与困惑度的综合评分能提升校准和鲁棒性。

**🔧 技术方法**

使用序列级困惑度、平均/最小标记概率、语义漂移等特征，构建逻辑回归风险评分；并进行长度控制、检索深度调节与外部验证实验。

**📊 数据集**

实验数据集包括 HotpotQA、SQuAD（四个指令微调模型）以及人类标注的 RAGTruth QA。

**📈 对比分析**

在长度控制的实验中，困惑度 AUROC 达到 0.81–0.91，综合评分与困惑度相当；检索深度降至 k=1 时 AUROC 明显下降，恢复到 k≥2；在 RAGTruth 上 AUROC 约 0.58–0.64，表明方法对更广泛样本仍有一定辨别力；并通过 ECE 与 Brier 分数展示了良好的概率校准。

**⚠️ 局限性**

局限性包括仅针对 QA 场景；对长篇生成、多模态或工具代理等情况未验证；需要可获取 token 概率的模型；受限于针对性编辑的受控幻觉构造和 LLM 判断的主观性。

---

## 275. Structured Authority Injection: Persistent Human Control for LLM Agents via Typed MCP Overrides

**arXiv ID:** 2610.22373 | [PDF](https://arxiv.org/pdf/2610.22373v1)

**作者:** Ryan Tran `[一作]` (California State University Long Beach), Hailu Xu `[通讯]` (California State University Long Beach)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Structured Authority Injection (SAI) 方案，使人类操作员通过 MCP 工具调用以类型化、可验证的方式向 LLM 代理注入持久权威指令，并在每次规划周期前将其按优先级排序后追加到 LLM 提示中，从而解决自由文本覆盖的歧义和权威衰减问题。

**💡 创新点**

创新点在于将 MCP 从双向工具总线改造成单向权威通道：1) 通过服务器端 schema 验证防止模型收到恶意或错误指令；2) 定义 Constraint、Priority、Strategy 三类持久指令并强制静态优先级 C≻P≻S；3) 在提示前置的位置利用 LLM 的首位关注机制保持指令权威。

**🔧 技术方法**

核心技术包括：基于 JSON Schema 的 MCP 语义校验、基于提示预置的权限上下文管理、按类优先级的冲突解析、以及指令的时间/次数衰减模型。

**📊 数据集**

使用自建的 MuJoCo‑grid‑world 仿真环境，构造八个聚焦不同代理失败模式的测试场景，并在九个公开/专有 LLM（GPT‑4o、DeepSeek‑R1、Qwen3‑32B 等）上进行实验。

**📈 对比分析**

通过将每个模型在基线与 SAI 两种条件下的任务完成率、移动/卡住时间分布以及平均生成 token 数进行对比，发现最强模型 GPT‑4o 与 DeepSeek‑R1 在大多数场景中完成率提升至 1.0 并可达 38% token 省略；其他模型表现参差，部分模型在 SAI 下性能下降或 token 消耗增加。

**⚠️ 局限性**

局限性包括：1) 需要先验验证模型生成预算以防超限；2) 对已有高度优化的规划器可能产生竞争性指令导致性能下降；3) 仅考虑单一可信监督者，多操作员场景需再引入仲裁层；4) 现有三类指令不一定覆盖所有领域需求；5) 在物理部署时需考虑通信延迟、感知噪声等实际约束。

---

## 276. Power of Conditional Sampling on Hypergraphs

**arXiv ID:** 2610.23364 | [PDF](https://arxiv.org/pdf/2610.23364v1)

**作者:** Debarshi Chanda `[一作]` (Indian Statistical Institute), Gopinath Mishra `[通讯]` (Institute of Mathematical Sciences)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在d-uniform超图中通过条件采样估计超边数量的问题，提出了一种随机算法，该算法使用s-colorful随机证书oracle来输出超边数量的近似值。

**💡 创新点**

提出了一种新的s-colorful查询oracle层次结构，展示了条件采样在超边估计中的强大能力，并证明了在s<d时，使用s-colorful决策oracle的算法需要多项式查询，而使用s-colorful随机证书oracle的算法只需多对数查询。

**🔧 技术方法**

使用了条件采样技术和随机算法，特别是s-colorful随机证书oracle来进行超边估计。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了d-uniform超图的性质和相关的查询模型。

**📈 对比分析**

与仅依赖于均匀随机超边样本的算法相比，使用s-colorful随机证书的算法在查询复杂度上实现了指数级的分离。在s<d的情况下，任何仅使用s-colorful决策oracle的算法都需要多项式数量的查询，而s-colorful随机证书oracle的算法则只需多对数数量的查询。

**⚠️ 局限性**

在s=d的情况下，虽然决策oracle的查询复杂度可以降至多对数级，但仍然存在分离，因为随机证书显著降低了log n因子的指数，从Θ(d)降至一个绝对常数。

---

## 277. Thunder-DeID+: Boundary-Aware Korean De-identification from Court Judgments to Broader Sensitive Documents

**arXiv ID:** 2610.22733 | [PDF](https://arxiv.org/pdf/2610.22733v1)

**作者:** Gyuseong Lee `[一作]` (Seoul National University), Jaejin Lee `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个针对韩语法院判决的去标识化框架，集成了三层可扩展的 PII 分类、边界感知的 tokenizer 以及基于词段的评价协议

**💡 创新点**

创新点在于：①将 PII 分类精炼为三层结构，消除冗余并提升可扩展性；②提出结合 N‑gram 与边界评分的 tokenizer，使 token 边界与 PII span 更一致；③使用词段级别的严格/部分、类型/二进制评估，消除 tokenizer 影响的比较不公平性

**🔧 技术方法**

主要技术包括：BERT 系列编码器（150M~1B），BIO‑CRF 序列标注，改进的 BPE+N‑gram tokenizer，tokenizer‑independent 词段评估流程，以及微调与零样本迁移训练

**📊 数据集**

数据集：基于 6,700 篇判决（共 48,306 个实体）的标注语料（覆盖民事、刑事、行政）；此外还收集了 1,000 篇非判决敏感文本进行迁移实验；实验还对比了 Thunder-DeID 公开模型

**📈 对比分析**

对比方法：在判决测试集上与 Thunder-DeID 在相同标签空间下进行 Strict/Partial、Typed/Binary 评估；在非判决文档上进行零样本与 1–1,000 样本微调的比较；实验显示 tokenizer 在 150M–1B 模型中将 W.Strict Typed F1 从 0.76–0.77 提升至 0.87–0.88，微调 1,000 条后达到 0.93，明显优于 Thunder-DeID（0.73）和开源 LLM（最高 0.48）

**⚠️ 局限性**

局限性：仅使用了置换（surrogate）后的判决文本，无法评估原始文本的再识别风险；评估聚焦 span 级别，未测量文档级隐私泄露；合成数据可能无法完全覆盖真实文档的多样性；方案针对韩语空格分隔，其他语言需改造

---

## 278. Validating LLM-generated political networks with survey-reported partisan exposure

**arXiv ID:** 2610.23199 | [PDF](https://arxiv.org/pdf/2610.23199v1)

**作者:** Marco Tonin `[一作]` (University of Trento), Luca Maria Aiello `[通讯]` (IT University of Copenhagen)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用大语言模型（LLM）生成大规模的社交网络，并通过结构指标和政治曝光度等多维度与真实社会网络数据进行对比与验证。

**💡 创新点**

创新点在于首次系统性评估LLM在生成社会网络时的结构真实性，结合多种prompt策略与温度设置，对生成网络与实际网络的度分布、传递性、密度、平均路径长度以及政治分层的一致性进行全面量化比较。

**🔧 技术方法**

主要技术包括：Prompt设计与温度调节、LLM生成网络、网络属性计算（度分布、传递性、密度、路径长度、政治 assortativity）、误差量化（Mean Signed Deviation）以及匹配分析。

**📊 数据集**

使用美国全国选举调查（ANES）2020/2022的社交网络问卷数据作为真实网络基准；对比中使用的网络属性均源自该数据集。

**📈 对比分析**

通过在相同LLM模型下多次生成不同配置（不同温度、上下文大小）的网络，并计算与ANES网络属性的差异。结果显示：温度为0.1时生成网络在度分布、传递性、密度等方面与ANES相近；温度为0.7时误差明显增大。政治曝光度误差亦随温度变化而不同，整体性能优于随机网络，但仍存在偏差。

**⚠️ 局限性**

局限性包括：生成网络对温度高度敏感；部分参数设置（如c=1）导致网络失去社会意义；仅针对美国社交网络，缺乏跨文化验证；未探讨大规模网络生成的可扩展性与多模态输入的适用性。

---

## 279. Not All Graphs Are Maps: A Task-Agnostic Exploration Framework for Mobile GUI Navigation

**arXiv ID:** 2610.22672 | [PDF](https://arxiv.org/pdf/2610.22672v1)

**作者:** Adamo Orsini `[一作]` (Concordia University), Yang Wang `[通讯]` (Concordia University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种任务无关的图导航框架，在不重新训练视觉GUI代理的前提下，通过离线探索构建可复用的目的地记忆，实现应用内页面导航。

**💡 创新点**

创新点在于：1) 将离线探索结果压缩为“目的地记录”并以纯文本注入现有代理；2) 通过目标覆盖和路径清洁度作为评估指标，拆分导航与完整任务的关系；3) 在真实Android应用上提出导航专用评测协议，独立衡量导航成功率。

**🔧 技术方法**

使用的技术包括：视觉语言模型用于屏幕摘要与路径蒸馏；本地BFS（Local-BFS）进行离线探索；嵌入检索（双字段检索）与目标选择提示；将检索到的目的地记录格式化为文本注入代理。

**📊 数据集**

数据集：基于20个Android应用的真实用户轨迹，抽取84个任务（6个核心应用）用于导航评测；还对额外68个任务做跨应用验证；对比标准Benchmarks（DroidTask、SPA-Bench）改为导航评测。

**📈 对比分析**

通过与无记忆基线、随机探索、自动BFS、人工干净图等条件对比，结果显示在UI‑TARS-1.5‑7B和Qwen‑VL‑Max上，使用Auto‑BFS记忆将导航成功率从约52%提升至58%，从59.5%提升至71.4%；在额外应用上也观测到正向提升，但统计显著性有限。

**⚠️ 局限性**

限制包括：1) 记忆有效性依赖于应用版本和账号状态，需重新探索；2) 主要针对页面导航，对最终任务执行不做提升；3) 记忆覆盖不完整，错误检索与近似命中仍占大部分失败；4) 对于需要长路径或复杂交互的页面，路由清洁度不足导致失败。

---

## 280. Learning the Shared Latent Structure of Cryo-EM Inverse Problems

**arXiv ID:** 2610.22523 | [PDF](https://arxiv.org/pdf/2610.22523v1)

**作者:** Nithin Chalapathi `[一作]` (University of California), Ritwik Gupta `[通讯]` (University of Maryland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

预训练了一个基于自监督学习的通用蛋白质表示模型（EMber），并在多种 cryo‑EM 下游任务中验证其可复用性。

**💡 创新点**

将 cryo‑EM 真实成像前向模型用于构造物理意义的正样本对，结合大规模模拟与实验数据混合训练 Vision Transformer，并在不同尺寸模型上实现跨任务迁移。

**🔧 技术方法**

自监督目标 DINOv3+ iBOT、Vision Transformer 变体（ViT‑S/B/L/g）、CryoJAX 前向模型、线性探针、fine‑tune、EM 估计和 CryoDRGN‑AI 反演流程。

**📊 数据集**

268k 结构模拟集（OpenProteinSet），51M 真实颗粒（214 EMPIAR 条目），以及 CAHRA、CryoPPP、Hsp90、SARS‑CoV‑2 Spike、Tomotwin‑8 等评测数据。

**📈 对比分析**

与 Cryo‑IEF、CryoUNI 以及监督 ResNet‑18 做对比；在数据集辨别、成分/构象分类、人口估计和异质重建中，EMber 取得更高准确率、低误差（如 1.59% 人口估计误差），在重建中仅需两轮 pose‑search 即可接近 Oracle 解析度。

**⚠️ 局限性**

在训练初期加入实验数据易导致表示崩溃，需要精细的混合策略；在极低 SNR 下仍受限；对实验多样性的恢复不足；空间特征在噪声极大时的监督仍需改进。

---

## 281. SCALE: Improving Guidance-Free Visual Generation via Self-Contrastive Alignment for Likelihood Estimation

**arXiv ID:** 2610.23263 | [PDF](https://arxiv.org/pdf/2610.23263v1)

**作者:** Jiwan Hur `[一作]` (KAIST), Junmo Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Self-Contrastive Guidance（SCG）与SCALE框架，实现无引导的生成模型训练；

**💡 创新点**

通过自对比（将模型对真实样本与自生成样本的预测做对比）得到直接的模型内部错误信号，突破传统CFG的局限；

**🔧 技术方法**

采用扩散模型与自回归模型的最大似然训练、PatchMix正则化、stop‑gradient、伪温度α与β控制的重参数化；

**📊 数据集**

在ImageNet 256×256（类条件生成）和LAION‑Aesthetics 5+、Stable Diffusion v1.5（文本到图像生成）等数据集上验证；

**📈 对比分析**

与CFG模拟的无引导方法（Distill、CCA、GFT）以及带CFG的采样方法（CFG、SAG、G++）对比，SCALE在FID、IS、Precision/Recall等指标上均优于基线，并且与DPO后训练进一步提升；

**⚠️ 局限性**

训练时存在额外的嵌套引导开销，且对更大规模模型与其他视觉对齐任务的适用性尚待验证。

---

## 282. Optimally Curating an Event

**arXiv ID:** 2610.22798 | [PDF](https://arxiv.org/pdf/2610.22798v1)

**作者:** MohammadTaghi Hajiaghayi `[一作]`, Suho Shin `[通讯]` (University of Maryland)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了自融资事件的最优设计，平衡了高价值参与者的招募与可能对事件产生负面影响的收入生成代理之间的关系。

**💡 创新点**

提出了一种新颖的双边机制设计框架，旨在最大化在预算平衡约束下的整体事件质量。

**🔧 技术方法**

使用了贝叶斯机制设计理论，分析了加性效用和子模效用的最优机制。

**📊 数据集**

未具体提及使用的数据集，但讨论了与学术会议和大型语言模型（LLM）相关的应用场景。

**📈 对比分析**

通过与经典的Myerson框架的比较，提供了加性效用下的最优机制的完整特征描述，并在子模效用下提出了常数因子近似机制，性能表现良好。

**⚠️ 局限性**

限制在于目前的分析主要集中在贝叶斯设置下，未来的工作需要扩展到更复杂的参与者之间的相互作用和更广泛的实用性。

---

## 283. Predicting Privileged Information Effectiveness for On-Policy Self-Distillation

**arXiv ID:** 2610.22512 | [PDF](https://arxiv.org/pdf/2610.22512v1)

**作者:** Kaijian Zou `[一作]` (University of Michigan), Anas Mahmoud `[通讯]` (Scale AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多轮智能体任务中 on‑policy self‑distillation (OPSD) 的特权信息（PI）配置进行了系统评估，探索了 16 种 PI 形式、3 种学生模型和 3 个基准。

**💡 创新点**

①首次在受控实验中量化 PI 对 OPSD 的影响；②提出了两种 KL‑based 离线指标（采样 token 反向 KL 与误差步 KL 对比）用于预先评估 PI 质量；③证明有效 PI 可与 GRPO 结合实现更优性能。

**🔧 技术方法**

OPSD、GRPO、KL 诊断指标；使用 Qwen3 系列模型与 GLM、Qwen 生成 PI；采用采样 token 反向 KL 与误差步 KL 对比进行 PI 评估。

**📊 数据集**

ALFWorld、AppWorld、τ²‑retail 三大多轮智能体基准（含 ALFWorld‑Passcode 变体）。

**📈 对比分析**

通过比较 PI 形式、生成源、学生模型对 post‑training pass@1 的影响；评估 PI 与 GRPO 的组合效果。结果显示：轨迹+观测、技能提示等 PI 在大多数模型上提升 5–15% pass@1；误差步 KL 对比是最可靠的 PI 选取指标；与 GRPO 结合后可超过单独 GRPO 的基准，尤其在轨迹 PI 上。

**⚠️ 局限性**

PI 效果高度依赖学生模型（思考模型受益不稳定）；静态 PI 在训练后期可能变得冗余甚至干扰进一步学习；KL 指标在某些极端 PI 组合下可能不完美；实验范围仅覆盖 3 个基准和 3 个模型，未涵盖更广泛的场景。

---

## 284. Generative Value Classifier

**arXiv ID:** 2610.22497 | [PDF](https://arxiv.org/pdf/2610.22497v1)

**作者:** Alexander Swerdlow `[一作]` (Stanford University), Chelsea Finn `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种生成式价值分类器，利用返回条件动作似然与返回先验构建隐式Q值，并用混合生成‑判别目标训练。

**💡 创新点**

将动作作为监督，逆向建模p(a|s,z)并结合Bayes得到返回分布，首次在RL中将生成式动作模型与分布式价值学习融合。

**🔧 技术方法**

使用流匹配（flow‑matching）估计生成似然，离散化返回空间，值锚定（value anchoring）与离散分类损失，和行为克隆的流匹配演员。

**📊 数据集**

在在线RL上评估RoboMimic四个机械臂操控任务；在离线/离线至在线RL上评估OGBench 35个任务。

**📈 对比分析**

与多种分布式与基线方法（QC、RLPD、IDQL、FQL、Value Flows等）进行比较，GVC在在线任务中略优，在离线及离线至在线任务中显著优于所有基线，尤其在Tool Hang、OGBench稀疏奖励任务上提升约10‑20%。

**⚠️ 局限性**

需对每个返回bin多次前向传播，计算成本较高；离散返回划分和双模型结构增加实现复杂性；对动作空间连续时的精细性受限。

---

## 285. Monitor-Validity Failures: LLM Safety Evidence Paths at Evolving Framework Boundaries

**arXiv ID:** 2610.23362 | [PDF](https://arxiv.org/pdf/2610.23362v1)

**作者:** Junchi Liao `[一作]` `[通讯]`, Junchi Liao

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统地识别并确认了LLM安全监视器在框架演化过程中出现的证据路径失效（即监视器检测到的安全信息未能被正确传递到决策边界），并提出了一种基于“canary contract”的差异化回归测试方法来验证这些失效。通过对公开修复 PR、issue 进行扫描，最终收集了18个确认案例，归纳为四类失败家族（carrier、adapter、authority、surface），并构建了可复现的测试协议。

**💡 创新点**

创新点包括：①首次将 canary contract 定义为面向安全监视器的可变更边界确认模板；②系统化归纳了四大证据路径失效模式；③提出跨框架、跨语言的可见性与决策判定框架，为安全监视器的回归测试提供了结构化方法。

**🔧 技术方法**

采用的技术主要有：代码变更与 PR/issue 检索、父子版本对比、canary payload 的定位与插入、可见性 oracle 与决策 oracle 的断言、负面控制实验、已知模式 fixture、以及 Tier A/B 级别的复现强度评估。

**📊 数据集**

使用的数据集来源于 195 条高信噪比的公开修复引用，筛选得到 18 条确认案例，涉及 11 个框架项目组和 4 种实现生态（Python、TypeScript、Java、C#）。此外为每个案例构造了 deterministic canary payload。

**📈 对比分析**

评估方法为父/子版本对比测试，利用可见性与决策 oracle 检测父版缺失与子版修复情况；通过 Tier A（完整框架级）与 Tier B（逻辑隔离）两级复现强度划分；E1/E2/E3 三层影响级别评估；实验结果显示 14 例达到 E1 执行级边界（如阻断、过滤、路由等），3 例为 E2（验证器/安全提示）和 1 例为 E3（可观测通道），未给出传统性能指标。

**⚠️ 局限性**

局限性包括：①案例数量有限，未覆盖所有可能的证据路径失效；②研究聚焦已公开的修复边界，缺乏整体普遍度估计；③大部分案例集中于 Python，跨语言覆盖不足；④未评估对运行时性能的影响；⑤某些细粒度的边界差异可能仍未被检测到。

---

## 286. FuzzInsight: Navigating the Efficiency-Effectiveness Trade-off in Coverage-Guided DNN Fuzzing

**arXiv ID:** 2610.23195 | [PDF](https://arxiv.org/pdf/2610.23195v1)

**作者:** Vivek Vekariya `[一作]` (Technical University of Munich), Alexander Pretschner `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了 DNN 覆盖驱动模糊测试的配置，包括种子选择策略与覆盖准则的组合。

**💡 创新点**

创新点在于提出等化采样（Equalized Sampling）种子选择策略，并系统评估非结构化覆盖准则（LSCD、LSC、MDSC）在效率-效果权衡上的表现。

**🔧 技术方法**

使用改进的 DeepHunter 框架，结合结构化覆盖（NC、k‑MNC、NBC）与非结构化覆盖（LSCD、LSC、MDSC）进行模糊测试，同时采用图像变异算子。

**📊 数据集**

实验基准为 MNIST、SVHN、GTSRB、CIFAR‑100 四个常用视觉数据集。

**📈 对比分析**

通过 84 种配置对比，等化采样将种子队列利用率提升至约 77.6%，非结构化准则在保持相似覆盖效果的前提下，计算成本比结构化准则低 8.7–24.6 倍；再训练后模型泛化鲁棒性平均提升 77.5%，稳定性波动 ≤3%。

**⚠️ 局限性**

局限性：实验仅聚焦图像分类任务，对变异算子和覆盖准则的跨域通用性待验证，且报告的绝对耗时受硬件环境影响，未做跨平台归一化。

---

## 287. BeJEPA: A Goal-Conditioned Humanoid Behavior World Model via Unsupervised RL

**arXiv ID:** 2610.23127 | [PDF](https://arxiv.org/pdf/2610.23127v1)

**作者:** Liyun Yan `[一作]` (RL Lab, Shanghai Jiao Tong University), Yue Gao `[通讯]` (RL Lab, Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种名为 BeJEPA 的行为世界模型，利用联合嵌入预测架构同时学习机器人身体姿态与世界位姿，并通过动态预测器和成功者预测器实现无长时间rollout的目标导向控制。

**💡 创新点**

创新点在于将即时物理动力学预测与长期成功者特征预测分离，利用无监督强化学习和人类运动先验构建共享潜在空间，使机器人在世界坐标系下实现姿态与位置的协调控制，并支持在线动作微调。

**🔧 技术方法**

使用技术包括：联合嵌入预测 (JEPA)、无监督强化学习、动态预测器 (Dynamic) 与成功者预测器 (Successor)、目标条件的Actor-Critic、在线梯度MPC 进行时间与空间对齐，以及线性读取器评估潜在空间的可解释性。

**📊 数据集**

使用的数据集为：人类运动先验数据集 AMASS 与 LAFAN 进行预训练；大规模 Isaac Sim 与 MuJoCo 环境用于在线交互学习；物理硬件 Unitree G1 的真实传感器数据用于部署验证。

**📈 对比分析**

与 SONIC、Humanoid-GPT 以及使用世界位姿输入的 BFM‑Zero 进行对比，BeJEPA 在 20 s 运动跟踪任务中将全局 MPJPE 降低 61.7%（相较 BFM‑Zero）和 58.7%（相较 SONIC），在根部平移和航向误差方面也实现显著提升；在 Unitree G1 上实现了地面站立、踢击后恢复以及连续目标切换的成功控制。

**⚠️ 局限性**

局限性包括：仅在模拟和 G1 机器人上验证，缺乏更复杂任务或更广泛硬件的通用性；模型对极端动作的泛化尚待进一步评估；依赖人类运动先验，若先验不足可能导致潜在空间不足；并且虽然避免了长时间rollout，但对大时间跨度的目标仍需更多经验探索。

---

## 288. VTrajEmbed: Aligning Text with Video Object Trajectories via Spatiotemporal Embeddings

**arXiv ID:** 2610.22842 | [PDF](https://arxiv.org/pdf/2610.22842v1)

**作者:** Ruizhi Chen `[一作]` (China Telecom Artificial Intelligence Technology Co., Ltd.), Hao Sun `[通讯]` (China Telecom Artificial Intelligence Technology Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了VTrajEmbed，一个将文本与视频对象轨迹映射到共享语义空间的嵌入模型，并提出基于轨迹的检索任务与基准。

**💡 创新点**

① Slow‑Fast双分支轨迹编码同时捕捉低分辨率密集观测的运动细节与稀疏高分辨率场景上下文；② mask‑guided对比学习利用已验证的查询‑轨迹关系实现对象级对齐；③ 构建了Text2Traj‑Bench人类验证的轨迹关系基准。

**🔧 技术方法**

使用Slow‑Fast双分支编码、冻结的视觉编码器+ROI投影、语言模型骨干、跨模态对齐、掩码对比损失、LoRA适配器以及基于SAM 3的轨迹提议。

**📊 数据集**

构建基准使用HC‑STVG、Kinetics‑700、MammAlps等视频数据集，检索评估涵盖HMDB51、UCF101、MSVD、MSR‑VTT、Breakfast等五个全球视频检索数据集。

**📈 对比分析**

在T2Traj、Traj2T、OCVR三种检索设置下，与Jina‑embeddings‑v4、BGE‑VL、WeMM、Qwen3‑VL‑Embedding等主流模型对比。VTrajEmbed‑5B在T2Traj R@1 69.67%（比Qwen3E‑8B提升15.48pp），Traj2T R@1 66.34%（提升18.05pp），OCVR R@1 68.04%；在全球视频检索中平均R@1 49.48%（相对Qwen3E‑8B提升7.6pp），整体性能显著优于基线。

**⚠️ 局限性**

仍依赖固定轨迹候选池，缺少对轨迹提议召回和轨迹质量的联合评估；最大相似度聚合对视频级排序仍有局限；基准覆盖的对象类别和域有限，跨域泛化尚未充分验证。

---

## 289. Sensitivity Analysis of Decision Rules in an Agent-Based Agro-Ecological Model

**arXiv ID:** 2610.22420 | [PDF](https://arxiv.org/pdf/2610.22420v1)

**作者:** Benjamin Pianet `[一作]`, Olivier Therond `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了MAELIA代理模型的决策敏感性，提出了面向层级参数空间的随机森林+HSIC-ANOVA方法，并给出肥料量、时机、管理结构等决策对氮流失、土壤碳、产量的影响。

**💡 创新点**

创新点在于将决策树结构视为建模对象，开发了随机森林监督的HSIC-ANOVA与层级代数距离相结合的敏感性分析流程，解决了传统GSA在分支空间上的不适用问题。

**🔧 技术方法**

使用随机森林、HSIC、ANOVA、层级拉丁超立方抽样、SMT等技术，以及代理建模与频率校正。

**📊 数据集**

采用MAELIA平台的单作物冬小麦、同一地形的5000条决策方案，每条方案运行10年天气记录，生成氮流失、土壤碳、产量三类指标。

**📈 对比分析**

与传统Sobol'或Shapley等方法的直接对比不适合；在同一层级空间下，该方法在计算成本上更低、解释性更强，并能公平评估稀有决策的影响，显示肥料剂量主导其余指标。

**⚠️ 局限性**

局限在于仅针对单作物、单地区、单一模型结构，未检验对多作物、不同气候或更复杂模型的通用性；此外，层级抽样与随机森林参数设置仍需经验性调整。

---

## 290. Characterising and Analysing Retrieval-Augmented Generation Techniques

**arXiv ID:** 2610.23226 | [PDF](https://arxiv.org/pdf/2610.23226v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 291. CoLU: Confidence-Weighted Layerwise Unlearning for Large Language Models

**arXiv ID:** 2610.22721 | [PDF](https://arxiv.org/pdf/2610.22721v1)

**作者:** Jingkai Guo `[一作]` (Arizona State University), Deliang Fan `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种单层Transformer更新的LLM遗忘方法CoLU，并配合Brief Layer Screening (BLS)快速筛选最优层，实现高效遗忘与保留。

**💡 创新点**

创新点包括：1）仅更新单层大幅降低计算与显存开销；2）引入Confidence‑Weighted Ascent (CWA)对目标token按置信度动态加权；3）提出Normalized Forgetting‑Retention (NFR)分数进行层级预筛选，显著提升遗忘-保留平衡。

**🔧 技术方法**

技术手段：CWA损失、保留正则化（交叉熵）、NFR评分、BLS预筛选、全屏蔽参考 (FS)、ES、FQ、MU、VerbMem、KnowMem、PrivLeak 等评估指标。

**📊 数据集**

使用数据集：TOFU（虚构作者档案）和MUSE（Books & News）。

**📈 对比分析**

与GA、NPO、SimNPO、SatImp、FU（全模型）等SOTA方法对比，CoLU在TOFU的遗忘-保留得分均优于对照组，MUSE上保持KnowMem最高、PrivLeak最接近零；BLS在保持性能的同时将运行时间减少60–80%，显存使用下降约77–80%。

**⚠️ 局限性**

局限性：1）CoLU整体训练时间仍高于全模型方法；2）BLS筛选仍有额外开销；3）现有基于梯度的层选择方法在CoLU上表现不佳，需进一步研究更鲁棒的层级选择策略。

---

## 292. ASE: Attribution-Guided Semantic Evolution for Time Series Forecasting

**arXiv ID:** 2610.23027 | [PDF](https://arxiv.org/pdf/2610.23027v1)

**作者:** Yuxuan Zheng `[一作]` (Peking University), Guannan He `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了 Attribution-guided Semantic Evolution (ASE)，一个三阶段的文本因子构建、条件评估与部署框架，用以提升时间序列与金融预测的表现。

**💡 创新点**

创新点在于通过语言模型生成事件/状态定义、基于回归损失的条件贡献诊断以及迭代保留策略，将外部文本信息与数值模型紧密结合，并在每一步提供可解释的反馈，显著降低因子冗余和提升预测质量。

**🔧 技术方法**

技术包括 Qwen3 语言模型用于因子生成与嵌入、Ridge 回归及其加权拟合、核岭回归预测子、条件贡献与互交诊断、以及基于验证集的冻结部署门控，整体实现了可插拔的模块化架构。

**📊 数据集**

实验使用四个 TESS 任务（FNSPID、Electricity、Environment、Bitcoin）以及动态 CSI 300 股票排名数据，分别涵盖多种行业与金融场景。

**📈 对比分析**

与公开 TESS 基准以及多种深度学习与统计模型（Ridge、NLinear、DLinear 等）比较，ASE 在所有四个任务上实现了最低的 MSE（0.001041 至 1.8150），在 CSI 300 上实现了 RankIC 的正向提升（+0.000642），并通过交叉种子验证显示一致性。

**⚠️ 局限性**

局限性包括：基准与实验设置不完全对齐（混合公开与本地自适配模型）、历史数据回测导致的潜在过拟合、Bitcoin 测试窗口有限、并且某些统计检验未进行多重比较校正。

---

## 293. AMEND: Constraint-Guided Multi-Agent Text Detoxification with Isolated Critics

**arXiv ID:** 2610.23183 | [PDF](https://arxiv.org/pdf/2610.23183v1)

**作者:** Tianyu Wang `[一作]` `[通讯]` (Soochow University), Tianyu Wang (Soochow University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AMEND 框架及其高效版 AMEND Lite，用于在推理时通过多角色生成、无偏评估和约束优先的选择实现文本去毒。

**💡 创新点**

创新点在于将去毒过程拆解为角色条件生成、独立无角色偏见的评估器和先决约束筛选，并通过两轮限定的控制策略显著减少迭代成本。

**🔧 技术方法**

使用角色条件生成器（Minimal Editor、Tone Rewriter、Semantic Paraphraser）、角色盲多维评估器、硬阈值约束选择以及基于大型语言模型的自动评估与两轮修正控制。

**📊 数据集**

实验基于 ParaDetox 英语并行去毒数据集进行评估。

**📈 对比分析**

与单角色生成、传统多轮辩论和关键词掩码等方法比较，AMEND Lite 在安全性、语义保留与流畅度上的联合指标提升约0.0489，且推理请求减少28.4%。

**⚠️ 局限性**

局限性包括仅在小型量化模型上验证、评估指标受限于自动代理、缺乏多语言覆盖和人类评测验证。

---

## 294. LithoWorld: An Endpoint World Model for Lithography Process-Window Recovery

**arXiv ID:** 2610.23408 | [PDF](https://arxiv.org/pdf/2610.23408v1)

**作者:** Botong Zhao `[一作]`, Yue Lu `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发一种端点世界模型 LithoWorld，用于从稀疏测量的聚焦曝光图（FEM）中恢复光刻过程窗口，使当前 SEM 图像回到合格的参数区。

**💡 创新点**

创新点包括：① 定义内部代表目标监督规则，自动挑选邻域完整合格点作为参考；② 通过因果端点对齐正则化视觉特征，避免直接复制参考；③ 采用流匹配的条件动作生成（flow policy）输出 8×2 步长动作，使模型在不需中间观察的情况下完成多步参数修正。

**🔧 技术方法**

使用技术：冻结 DINOv2-B/14 视觉编码器、Qwen2.5‑0.5B 语言模型（改为 LoRA）、因果对齐分支、流匹配（Gaussian diffusion policy）产生动作块、端点对齐损失及多种基线的直接回归对比。

**📊 数据集**

数据集：173 组真实 SEM（FEM），细分为 SEM‑73（73 本工作簿、4,334 点）和 SEM‑100（100 本工作簿、9,900 点），共 14,234 个点，其中有 2,915 个未测格点。

**📈 对比分析**

比较方法：与 DINOv2+MLP、ACT‑goal、Diffusion‑Policy‑goal、MDT‑image、ManiFlow‑goal、MoDE‑goal 等六种基线对比。LithoWorld 在 SEM‑73 测试集上 GSR 88.5%（比最强 86.0% 提升 2.5pp）且 ISR 46.0%（比最强 40.0% 提升 6.0pp）。在 SEM‑100 上 GSR 86.0%（略低于 86.4%）但 ISR 32.5%（比 31.6% 提升 0.9pp）。

**⚠️ 局限性**

局限性：仅为离线模型，无闭环实时 SEM 反馈；需要已有合格参考 SEM 作为输入；对未测格点或超出测量范围的预测无法评估；未在不同批次、校准坐标或不确定性估计上验证，可迁移性尚待研究。

---

## 295. Wait for It: A Component-Level Analysis of Latency Feedback Mechanisms for LLM Agents in VR

**arXiv ID:** 2610.23336 | [PDF](https://arxiv.org/pdf/2610.23336v1)

**作者:** Pratham Arora `[一作]` (Plaksha University), Siddharth Siddharth `[通讯]` (Plaksha University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在VR环境中对比三种延迟反馈机制（口头填充、视觉加载指示、身体动作），研究它们对用户沉浸感与社交存在感的影响。

**💡 创新点**

将延迟反馈拆分为独立组件进行系统性比较，并首次发现身体动作在保持社交存在感方面优于视觉透明度，提出“具身等待状态”设计框架。

**🔧 技术方法**

使用Unity 6+Meta Quest 2实现VR场景，集成Google Gemini 2.5 Flash做LLM推理，配合Google Speech‑to‑Text/Text‑to‑Speech实现语音交互，动画通过IK生成。

**📊 数据集**

未使用公开数据集，实验基于18名受试者的对话记录和主观问卷，未涉及外部语料库。

**📈 对比分析**

采用within‑subjects设计，使用Friedman、Wilcoxon、配对t检验对Godspeed问卷等指标进行比较，结果显示身体动作在动画感、拟人化、自然度等维度显著优于视觉与口头反馈，且最小化干扰。

**⚠️ 局限性**

限制包括样本偏年轻且VR经验不足、网络延迟波动未完全控制、口头填充为随机无上下文、动作动画为IK生成缺乏高保真捕捉，以及仅探讨低于4秒延迟的情境。

---

## 296. Normalizing Flow Maps

**arXiv ID:** 2610.22711 | [PDF](https://arxiv.org/pdf/2610.22711v1)

**作者:** Francesco M. Ruscio `[一作]` (Aithyra), Avishek Joey Bose `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种新的可逆传输映射（Normalizing Flow Map），实现一次性精确似然与少步采样的统一；

**💡 创新点**

创新点在于将流匹配、流图与可逆流模型结合，利用点匹配与分布匹配的自蒸馏目标实现高效训练，并在终点获得精确似然；

**🔧 技术方法**

使用可逆神经 ODE、流图参数化、点匹配/分布匹配损失、RLHF 以及重要性采样等技术；

**📊 数据集**

在图像生成任务中使用 ImageNet/LSUN 条件数据集，在分子模拟任务中使用 Chignolin、4- 和 8 残基肽的 MD 数据集；

**📈 对比分析**

与 iTARFlow、SoftFlow、SBG 等基线比较，FID/FLD 以及 Boltzmann 生成的能量/角度 Wasserstein 距离均优于传统 NF 或流匹配模型；

**⚠️ 局限性**

局限性包括数值不可逆性易导致 RL 后训练不稳定、分布匹配方差高以及在多步采样时需平衡可逆性与质量。

---

## 297. Thinking from Above: Efficient Spatial Reasoning with Bird's-Eye-View Visual Tokens

**arXiv ID:** 2610.22700 | [PDF](https://arxiv.org/pdf/2610.22700v1)

**作者:** Faizan Siddiqui `[一作]` (Honda Research Institute USA), Enna Sachdeva `[通讯]` (Honda Research Institute USA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

引入了参数无关的BEV视觉分词方案，通过深度和相机位姿将视频帧的图像补丁特征投影到共享的鸟瞰图网格，聚合相同物理位置的特征以构建固定大小的场景级视觉令牌；

**💡 创新点**

核心创新在于利用已知几何信息直接实现视觉分词，而非依赖学习的几何模块，并证明图像编码器的特征局部性是BEV聚合的关键；

**🔧 技术方法**

使用冻结的DINOv3 ViT-L/16作为视觉编码器，基于相机内参/外参的几何投影与平均，随后将BEV令牌输入预训练的Qwen2.5-0.5B LLM；

**📊 数据集**

在ScanNet、ScanNet++和ARKitScenes等室内数据集上构成的VSI-590K、VSI-Bench以及ReVSI评测集进行训练与评估；

**📈 对比分析**

与帧级令牌对比，Cambrian-BEV在VSI-Bench上取得63.48分，比相同参数基线提升8.5分（Room Size +24.5，Relative Direction +34.1），在ReVSI上同样领先控制组约4.5分；同时显著减少视觉令牌数12.6×，预填充时间2.9×、峰值内存3.6×；

**⚠️ 局限性**

局限包括：无法保留视角序列的访问顺序，垂直结构被压平导致高度分辨率低，依赖精确的深度与位姿，且主要适用于静态、房间尺度的室内场景。

---

## 298. Coverage Cliffs in Off-Policy Quantile Certification

**arXiv ID:** 2610.22565 | [PDF](https://arxiv.org/pdf/2610.22565v1)

**作者:** Pauline Bourigault `[一作]` (Imperial College London), Emmanuelle Bourigault `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究如何在使用同一批上下文-带状日志来选择和验证策略时，利用日志中保留的信息对策略的尾量化指标（如p分位数）进行置信上界证书，并提出一种名为TailCert的算法。

**💡 创新点**

主要创新在于量化日志压缩（仅保留联合CDF指示器）对尾部证书的影响，证明在稀有动作场景下会出现O(p/(1-p))的样本量损失；并证明通过保留动作计数或两项权重总和可以完全恢复原始日志的置信证书。

**🔧 技术方法**

使用了PAC‑Bayes Bernstein 下界、Empirical Bernstein 上界、重要性加权估计、联合置信区间、置信区间和搜索成本的理论分析，以及自适应阈值网格的处理。

**📊 数据集**

实验数据集包括：自定义稀有动作模拟、125路由器共享动作实验（RouterBench）、以及基于GPT的推理任务（98,000代），并在这些数据上验证了样本量与搜索成本。

**📈 对比分析**

与现有的CDF、Exc-Bernstein、PAC‑Bayes、投注等方法比较，发现恢复信息后Exc‑EB在样本量和认证成功率上显著优于仅用CDF指示器的方案；搜索成本在共享动作场景中依旧显著。

**⚠️ 局限性**

局限性在于假设已知重要性权重上界、独立固定的候选策略、预先声明的阈值网格；对重复候选、非二元分布、以及序列决策的适用性尚未充分验证。

---

## 299. Decomposing URLLC Deadline Failures in Beam-Managed 6G Networks: Implications for Mission-Critical Wireless Systems

**arXiv ID:** 2610.22411 | [PDF](https://arxiv.org/pdf/2610.22411v1)

**作者:** Webert Montlouis `[一作]`, Sneihil Gopal `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了将 6G mmWave 波束管理网络中的 URLLC 失约率拆分为队列限、传输限和波束跟踪限三类的框架，并在十个不同天线/用户场景下进行评估。

**💡 创新点**

创新点在于构建了近因分类法和极低开销的每包分类器，可直接识别失约主因；发现天线数约 1825 之后波束跟踪失约成为主因，并给出针对性改进建议。

**🔧 技术方法**

采用了 mmWave 波束管理仿真（Stampede3）、六种调度器（EDF、PF、RR、FCFS、Hybrid ML‑EDF、Weighted ML‑EDF）以及低开销的误差分类器。

**📊 数据集**

使用 3GPP TR 38.901 城市微蜂窝信道模型，基于 100 次 Monte Carlo 试验生成不同天线数、波束数和用户数的十个场景数据。

**📈 对比分析**

通过对比六种调度器、天线数变化和波束扫频周期（10~40 ms）的敏感度，发现波束跟踪失约随天线数增大呈指数上升，波束扫频周期减半可使 B‑失约率降低 50% 以上；T‑失约率基本保持 23–29% 的稳定水平。

**⚠️ 局限性**

限制在于：每个失约仅归属于单一近因，未考虑多因混合；角度误差判定仅在仿真中可获取，实际部署需采用信号质量代理；仿真环境未覆盖敌对干扰、电子战等军事实况，导致结果可能偏差。

---

## 300. Recoverability-Controlled Safety Miscalibration in Large Language Models

**arXiv ID:** 2610.22425 | [PDF](https://arxiv.org/pdf/2610.22425v1)

**作者:** Yuwei Wang `[一作]` (Shanxi University), Haijun Geng `[通讯]` (Shanxi University)

**通讯引用:** 304 | [OpenAlex ID](https://openalex.org/A5101413343)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RiskCueBench，采用双阶段恢复性控制评估框架，对 160 种潜在风险意图在 640 条非操作性表面提示下，评估 25 个模型的拒绝率、危险合作率和全球安全得分。

**💡 创新点**

创新点在于将意图恢复与安全响应分离，构建可恢复性控制的基准，揭示表面提示的具体化会导致安全行为失衡，即使模型已识别到高风险意图。

**🔧 技术方法**

技术包括：两步评估（意图恢复通道与响应通道）、基于 IRS（Intent Recovery Score）的门控、配对引导的自举法与混合效应模型分析、人工多评标与 AI 辅助裁决。

**📊 数据集**

数据集为自制 RiskCueBench，涵盖 160 种潜在意图、8 类风险、640 条非操作性表面提示（含高风险、辅助控制和模糊控制），共 32,000 条成功模型输出与人工标注。

**📈 对比分析**

比较方法：对比同一意图下的四种表面提示（S0C0、S1C0、S0C1、S1C1），在 IRS≥3 的可恢复配对上进行配对自举与混合效应分析；结果显示，特定化导致拒绝率下降 0.175，危险合作率上升 0.107，全球安全得分下降 0.440，表明安全性显著受表面细节影响。

**⚠️ 局限性**

局限性：仅评估单轮英文提示；标注在安全评分细粒度上一致性有限；使用的 IRS 门限在 IRS=4 时效果减弱；未覆盖多轮对话、跨语言或更广泛安全领域。

---

## 301. NL2SQL with Iterative Self-Refinement

**arXiv ID:** 2610.22968 | [PDF](https://arxiv.org/pdf/2610.22968v1)

**作者:** Anjali Vemula `[一作]` (University of Minnesota - Twin Cities), Akshat Dasula `[通讯]` (University of Minnesota - Twin Cities)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了面向保险行业的多代理、角色驱动的 NL2SQL 框架，利用多轮自我改进实现对自然语言查询的高精度 SQL 生成。

**💡 创新点**

将领域相关性检测、KPI 解析、schema grounded 生成、基于执行的语法校正以及 SQL→NL 语义一致性验证结合到迭代自我修正流水线，并引入 persona 角色提升精度。

**🔧 技术方法**

使用 LLM（如 Mistral）进行 chain‑of‑thought 生成与校正，SQLite 执行反馈，SQL‑to‑NL 逆向工程做语义评估，结合知识字典与多代理协作。

**📊 数据集**

使用 60 条保险域查询（50 条来自 HuggingFace text‑to‑SQL 数据集 + 10 条自定义 KPI 复杂查询）以及自行构建的合成保险数据库 schema。

**📈 对比分析**

通过与直接 LLM 提示对比，采用语义一致性评估与执行测试，在 60 条查询上平均准确率约 90%，数据检索任务 100%，分析报表任务 85%。

**⚠️ 局限性**

依赖多轮 LLM 调用导致约 20 秒延迟；需预先构建 KPI/Schema 字典，未充分验证跨 LLM 的通用性，极端复杂查询仍可能出现误差。

---

## 302. When Can You Trust an Agent's Own Verifier? Stress-Testing Counterfactual Rationale Verification Under Search Pressure and Oracle Access

**arXiv ID:** 2610.22779 | [PDF](https://arxiv.org/pdf/2610.22779v1)

**作者:** Jonghyun Lee `[一作]` (Yonsei University), Sunghwan Park `[通讯]` (Yonsei University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在LLM驱动的机器学习管道设计任务中引入对代理自身解释的可验证性度量——Rationale Consistency Score（RCS），系统性地评估该验证器在被优化过程中的鲁棒性。

**💡 创新点**

创新点在于提出一种压力比ρ（衡量验证器对搜索目标的影响）以及四个可检验的失败模式（沉默、未测试、声明最小化、Oracle滥用）和相应的probe，形成一套完整的验证器stress‑test框架，可应用于任何基于反事实的理据验证器。

**🔧 技术方法**

技术手段包括：基于LLM的代理生成管道与自然语言解释；利用工具接口构造反事实干预（特征删除、模型替换、超参数扰动）；计算RCS并加入搜索目标；采用Thompson Sampling和UCB1搜索策略；统计压力比ρ并执行四类probe。

**📊 数据集**

实验使用六个监督学习数据集，涵盖四种模态：tabular（Credit‑G、Breast Cancer、Madelon）、文本（Ecommerce）、图像（Butterfly）和时间序列（Electricity）。

**📈 对比分析**

对比方法：在每个数据集上比较加入RCS与不加入RCS的搜索表现，包括验证集准确率、RCS信号、声明数量及其平均长度；结果显示：①在高压力下（ρ>10）并未观察到声明最小化；②在可查询验证器时出现选择偏差，导致验证子集的RCS下降；③在大多数数据集上加入RCS对任务性能影响不大，偶尔会出现性能下降。

**⚠️ 局限性**

限制：仅使用单一验证器（RCS）与单一LLM生成模型（Solar Pro 3）；实验规模有限（六个任务、180+90次运行）；未对生成模型或验证器参数做大范围扫掠；未测试其他可能的失效通道（如声明重述、延迟查询、搜索反馈包含声明计数）；结果受单一模型波动影响，且未对估计器的阈值做系统化调优。

---

## 303. Multimodal Filesystem Memory For Long-Horizon Embodied Agents

**arXiv ID:** 2610.22686 | [PDF](https://arxiv.org/pdf/2610.22686v1)

**作者:** Chloe Su `[一作]` (Harvard), Yilun Du `[通讯]` (Harvard)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了多模态文件系统记忆（MFM），在长时序任务中整合情节、空间、语义、程序和工作记忆，实现对过去经验的检索、检视和自我巩固，提升Agent的表现。

**💡 创新点**

创新点在于将多种记忆类型通过统一的文件系统结构组织，并用可搜索的文本索引将文本描述与多模态证据关联，支持视觉检索与知识、程序的自我演化更新。

**🔧 技术方法**

采用文件系统存储、可搜索文本索引、视觉帧保存、检索与检视机制，以及与大语言模型（如 GPT‑6‑Astra）协同的自我巩固算法。

**📊 数据集**

使用 Minecraft 长时序任务（Mine Diamond、Ender Dragon）和 CUABench‑KiCad 电子设计自动化任务作为评估数据集。

**📈 对比分析**

与无记忆、PRO‑LONG、VISTA 等基线在相同基模型下对比；MFM 在 Minecraft 中完成所有 Diamond 里程碑并在 Ender Dragon 中达 16/24 里程碑，CUABench‑KiCad 平均奖励提升至 0.440，整体性能优于基线。

**⚠️ 局限性**

局限性包括探索与导航仍有难度、仅在单一 Agent 与有限环境中验证，缺乏跨 Agent 共享记忆和更大规模环境的评估。

---

## 304. Edit Discovery in Human-AI Interactions

**arXiv ID:** 2610.22661 | [PDF](https://arxiv.org/pdf/2610.22661v1)

**作者:** Alireza Mirrokni `[一作]` (EPFL), Hamed Hassani `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了编辑发现优化（Edit Discovery Optimization）框架，针对有限编辑资源下的人工–AI交互，设计了在线算法 EDIT，结合编辑评分、确认率控制与预算约束，实现高效编辑机会发现。

**💡 创新点**

将编辑发现建模为受限选择问题，证明最优策略为成本调整阈值决策，并基于此提出无分布假设的在线迭代阈值控制算法，兼顾 Type‑I、Type‑II 错误与编辑成本。

**🔧 技术方法**

采用可学习编辑评分（基于冻结表示的线性探测器或 LLM 预估）、梯度更新、批量化成本调整 β、确认率阈值 α 的自适应迭代；算法在每轮通过梯度下降更新评分并保证预算满足。

**📊 数据集**

在四个推理与事实问答基准上评估：MATH‑500、MMLU‑Pro、PopQA 与 GPQA，使用 Llama‑3.3‑70B‑Instruct（及 Qwen3‑32B 对照）生成回答。

**📈 对比分析**

与直接使用 LLM 进行批量编辑选择（LLM‑Select）以及随机、LLM 评分和表示评分的基线对比；在相同编辑成本和确认率容限下，EDIT 在所有数据集上均显著降低 Type‑II 错误，且 Type‑I 误差与预算约束满足。

**⚠️ 局限性**

实验依赖基准生成的编辑标签且仅做 200 条人工审核，未评估发现的编辑对模型后续训练的实际收益；编辑工作流、异质编辑成本与真实用户场景仍待进一步研究。

---

## 305. Diverse Reasoning Paths Matter: Symbolic Equivalence Augmentation for Enhancing RL Exploration

**arXiv ID:** 2610.22665 | [PDF](https://arxiv.org/pdf/2610.22665v1)

**作者:** Xiaoou Liu `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究链式推理（CoT）在监督微调阶段的多样性如何影响后续强化学习与可验证奖励（RLVR），并提出EquivAug框架通过符号等价变换生成结构多样且可验证的CoT路径；

**💡 创新点**

核心创新在于利用符号等价规则生成结构层面多样的、在每一步都可验证正确的推理路径，从而显著提升RLVR的探索效果；

**🔧 技术方法**

技术方案包括自然语言到符号表示、等价变换、符号求解器验证、逆向自然语言重构以及后续使用GRPO进行强化学习；

**📊 数据集**

实验数据集为数学推理的GSM8K、逻辑推理的FOLIO，另外在分布外的SVAMP和MATH500 上验证泛化；

**📈 对比分析**

与Ground Truth、RFT、Paraphrasing等基线对比，EquivAug在SFT后得到最高的RPD，并在GRPO后取得最高ACC，RPD与RL性能的斯皮尔曼相关系数高达0.77；

**⚠️ 局限性**

局限性包括等价规则库手工设计，缺乏自动发现或扩展机制，且实验仅覆盖1.5B–8B规模模型，未验证更大模型上的效果。

---

## 306. Spectral Limits of Polynomial Graph Filters on the Long-Range Ising Model Benchmark

**arXiv ID:** 2610.22713 | [PDF](https://arxiv.org/pdf/2610.22713v1)

**作者:** Amin Najafgholizadeh `[一作]` (University of Zanjan), Mohsen Afsharchi `[通讯]` (University of Zanjan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究在Long‑Range Ising Model基准（LRIM）上分析了读出层滤波器的误差底线，提出并验证了50阶步幅滤波器与稀疏卷积网络在长程依赖格点数据上的优越表现。

**💡 创新点**

创新点在于证明多项式滤波器在谱级集上不可降低错误底线，且设计了能突破该底线的高阶步幅滤波器和稀疏卷积网络；同时对比了标准傅里叶神经算子在大尺度格点上的劣势。

**🔧 技术方法**

采用了多项式滤波分析、谱图网络、步幅（strided）滤波、稀疏（dilated）卷积网络以及傅里叶神经算子等技术。

**📊 数据集**

使用Long‑Range Ising Model benchmark的两个耦合场的格点数据进行实验。

**📈 对比分析**

通过在训练集上拟合滤波器并在测试集评估误差，稀疏卷积网络在较大节点数上实现最低误差，超越现有最优网络；50阶步幅滤波器误差更低；而傅里叶神经算子在格点尺寸增大时表现最差。

**⚠️ 局限性**

局限性包括只在LRIM基准上验证，步幅滤波器需要训练大量系数，且在大规模格点下仍受预算限制；FNO在大尺度格点上性能欠佳。

---

## 307. Stuck on Shuffle: On Attribute-Object Binding in Vision-Language Models

**arXiv ID:** 2610.22602 | [PDF](https://arxiv.org/pdf/2610.22602v1)

**作者:** Netanel Y. Tamir `[一作]` (OriginAI), Rami Ben-Ari `[通讯]` (OriginAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了SABER基准，并通过仅使用文本数据对多模态嵌入器进行微调，显著提升了属性-对象绑定能力。

**💡 创新点**

提出了全四种模态对（图-图、图-文、文-图、文-文）的合成属性绑定基准，并证明文本监督可以跨模态提升绑定性能，进一步推广至其他检索任务。

**🔧 技术方法**

利用MLLM Qwen3-VL-Embedding进行LoRA+LiNeS微调，采用triplet文本损失与合成文本/图像数据，构建了大规模文本绑定训练集。

**📊 数据集**

使用SABER（9k图文对、1.8k MCQ）与Gemma生成的20万文本triplet作为训练集，评测时结合MMEB、NegBench、实例检索等现有基准。

**📈 对比分析**

相较于CLIP、TripletCLIP、READ-CLIP等基线，4‑way MCQ平均准确率提升至84.2%，大图检索mAP提升约19点；跨基准平均提升15+点。

**⚠️ 局限性**

仅基于合成数据，真实场景多样性有限；绑定改进后仍存在包含（containment）等挑战；未对长文本或复杂关系做系统评估。

---

## 308. SkyAgent: Revisable Target Memory for Continuous Aerial Search

**arXiv ID:** 2610.23293 | [PDF](https://arxiv.org/pdf/2610.23293v1)

**作者:** Kaibo Yang `[一作]` (Chinese Academy of Sciences), Zhan Chen `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对连续无人机搜索中的假候选持续问题，提出了可回溯目标记忆（RTM）机制，能在后续证据表明候选错误时，自动撤销对应的待执行航路点并更新候选状态。

**💡 创新点**

创新点在于将候选生命周期与规划绑定，使用签名证据聚合与独立视角分组，实现候选的可逆评估和待执行航路点的自动取消，显著降低了假候选导致的路径浪费。

**🔧 技术方法**

采用了候选证据记忆、视角分组聚合、签名证据评分、状态机生命周期管理以及计划绑定等技术，并将其集成到SkyAgent框架中。

**📊 数据集**

实验使用了CityAVOS、UAV-ON、OpenUAV三大公开无人机搜索基准，并在真实DJI飞行环境中进行35个匹配场景的验证。

**📈 对比分析**

与基线（仅追加式搜索）相比，RTM在四个基准拆分中将成功率提升2.9–6.3个百分点，将行为级假候选率从45%降至约15%，并将后续证据冲突导致的距离浪费从约120米降低至44米，整体性能显著提升。

**⚠️ 局限性**

局限性包括对跨视角关联的依赖、证据模型可能存在系统性偏差、未涵盖动态目标以及缺乏通用安全保证，需要在不同摄像头/高度上重新校准。

---

## 309. Expand-and-Prune: Test-Time Candidate-Space Adaptation for End-to-End Autonomous Driving

**arXiv ID:** 2610.23262 | [PDF](https://arxiv.org/pdf/2610.23262v1)

**作者:** Fabian Schmidt `[一作]` (Esslingen University of Applied Sciences), Abhinav Valada `[通讯]` (University of Freiburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种在测试阶段动态扩展与裁剪候选轨迹集合的框架——Expand‑and‑Prune，用于提升预训练端到端驾驶规划器的表现；

**💡 创新点**

创新点在于（1）利用语义意图和历史轨迹生成额外候选轨迹，实现候选空间的自适应扩展；（2）根据中间轨迹分数的不确定性（熵）动态裁剪候选数，实现下游计算的自适应压缩；

**🔧 技术方法**

主要技术包括：基于语义意图的候选生成（action prototype banks）、时间记忆（从上一帧迁移高置信度轨迹）、熵自适应裁剪（Top‑K 线性插值）以及在多种候选生成/选择架构中的无参数集成；

**📊 数据集**

使用的数据集包括：LangAuto（CARLA 8城、3路长）、Bench2Drive、Fail2Drive（CARLA）以及 NAVSIMv2（真实轨迹）等；

**📈 对比分析**

在 LangAuto 上，加入 Expand‑and‑Prune 后 Driving Score 由 68.2 提升至 75.9，成为该基准的最优；在 Bench2Drive、Fail2Drive 以及 NAVSIMv2 上亦分别提升 0.5–1.2 点；同时，裁剪机制在保持或提升性能的同时减少了 2–10 % 的前端推理时延；

**⚠️ 局限性**

局限性包括：意图引导仅在具备概率动作信念的规划器（如 LAD‑Drive）上实现；时间记忆仅利用上一帧信息，未考虑更长历史；扩展和裁剪的超参（预算、熵阈值）需手工设定；并且框架未在完全自监督或多模态意图场景下进行验证。

---

## 310. Surprise Masking: Private Data Filtering with Model Likelihoods

**arXiv ID:** 2610.22988 | [PDF](https://arxiv.org/pdf/2610.22988v1)

**作者:** Riya Tyagi `[一作]` (OpenAI), Om Thakkar `[通讯]` (OpenAI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于公共参考模型惊奇度的训练时遮掩方法（Surprise Masking），用于在保持模型学习有用信号的同时抑制对私有或不想学习内容的记忆。

**💡 创新点**

创新点在于直接利用参考模型的困惑度对局部 token 窗口进行惊奇度评分，自动识别并遮掩“惊奇”span，从而在代码数据上实现更优的隐私‑效用折中，且不需要手动标注或全局删除。

**🔧 技术方法**

核心技术包括窗口化（可重叠或完整行）、计算平均负对数似然作为惊奇度、阈值化与共识遮掩、重复传播后处理，以及将窗口级遮掩映射到 token 级遮掩，适配训练时和数据清理两种场景。

**📊 数据集**

实验使用了多种数据集：synthetic Code PII 与 Code Quirks（基于 ExStruct 代码库）、CredData（凭证）、PIIBench（个人信息）、Dead‑code poisoning（DePA 重建）以及其他代码与文本基准。

**📈 对比分析**

与无遮掩、oracle 掩码、随机掩码、Goldfish loss、OPF 等基线对比，Surprise Masking 在代码微调中将私有值提取率从约 66% 降至 2%，将风格痕迹从 24% 降至 1.3%，且在死代码去毒任务上性能优于 DePA；但在更广泛的凭证和 PII 检测上落后于专用检测器。

**⚠️ 局限性**

局限性包括：缺乏正式的隐私保证，可能误遮掩有价值的陌生内容，需在训练集上调参，且对非代码或高度异构数据的适用性有限。

---

## 311. A Feature Selection Layer Approach for Multi-Class Classification on Neural Networks

**arXiv ID:** 2610.22898 | [PDF](https://arxiv.org/pdf/2610.22898v1)

**作者:** João Pedro Silveira e Silva `[一作]` (Federal University of Rio Grande do Sul), Bruno Iochins Grisci `[通讯]` (Federal University of Rio Grande do Sul)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了四种扩展的特征选择层（FSL），实现多类别标签特定的特征重要性学习，并在深度神经网络中直接训练；

**💡 创新点**

创新点在于将FSL从单一全局权重扩展为标签专属权重矩阵，并提供Listener、Residual、Output‑Aware和Multiple FSL四种实现，使模型能够在不额外后处理的情况下捕获标签特定的特征关系；

**🔧 技术方法**

使用深度前馈网络、ReLU激活、L1正则化、残差连接以及梯度更新；评估方法包括PIFS/PSFI、F1、加权t‑SNE、稳定性（Jaccard/Spearman/Pearson）以及SVC预测性能；

**📊 数据集**

实验数据集涵盖人工合成的XOR、SynthA/B/C（三分类，含标签特定特征）以及真实高维微阵列数据（Liver、Colorectal、Breast）和UCI公开数据（Letter Recognition、Isolet）；

**📈 对比分析**

与原始FSL、传统过滤/包装方法（Kruskal‑Wallis、ReliefF、DecisionTree、RandomForest、Lasso）以及后置解释器（LIME、DeepSHAP、SAGE、TabNet、LassoNet）进行对比；结果显示Multiple FSL在全局特征选择与标签特定特征识别上均优于基线，F1保持或提升，PIFS/PSFI接近或超过树模型，稳定性和记忆占用在可接受范围内；

**⚠️ 局限性**

局限性包括：标签维度多时显著增大内存占用；Output‑Aware在大规模稀疏数据上表现不佳；Residual实现对特定数据集的泛化有限；实现仍需针对GPU内存和训练时间进行进一步优化。

---

## 312. The Case for TPU-Accelerated Analytical Query Processing

**arXiv ID:** 2610.22673 | [PDF](https://arxiv.org/pdf/2610.22673v1)

**作者:** Yeounoh Chung `[一作]` (Google), Henry M. Levy `[通讯]` (Google)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对TPU的Native TPU Accelerator（NTA），将关系查询转化为静态张量程序，利用TPU的TensorCore和SparseCore实现高效分析查询。

**💡 创新点**

创新点在于对TPU硬件进行协同设计：通过静态形状规划、动态算子选择、SparseCore聚合/散射替代稀疏访问，利用JAX/Pallas自定义核实现，完成从查询规划到TPU执行的全流程。

**🔧 技术方法**

采用JAX框架、XLA编译器、Pallas自定义核、TPU v5p/ v7p硬件、TPU分布式( SPMD)以及静态形状与成本模型。

**📊 数据集**

使用TPC‑H benchmark（SF1/100/1000）和ClickBench（1亿行表）验证系统。

**📈 对比分析**

与基准CPU引擎DuckDB及直接JAX对比，单设备在SF100实现7.0×加速，16个TPU在SF1000实现68.6×加速；对单机CPU获得1.9–30.5×速度提升，极端规模下总运行时间提升50.1×。

**⚠️ 局限性**

局限包括对变长字符串和高基数离散化的处理导致内存浪费、对动态形状填充和稀疏访问的残留瓶颈、跨设备通信开销、以及多租户并发时FIFO调度导致延迟线性增长。

---

## 313. Controlling the Diversity-Quality Trade-Off of LLMs by Merging Models Before and After Alignment

**arXiv ID:** 2610.22999 | [PDF](https://arxiv.org/pdf/2610.22999v1)

**作者:** Yasuaki Sumita `[一作]` (Kyoto University), Toshiyuki Tanaka `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于层深度的FrankenMerging方法，将对齐模型的低层与基模型的高层结合，用于在不再训练的前提下控制大语言模型的多样性与质量平衡。

**💡 创新点**

创新点在于：①利用层深度差异把低层与高层分别关联到词汇多样性和语义多样性；②通过可调节的分割层数（i）在对齐模型和基模型之间平衡多样性与质量；③在不需额外训练的情况下，实现比温度采样更优的多样性-质量权衡。

**🔧 技术方法**

技术手段：模型层级融合（depth-aware FrankenMerging），使用Self‑BLEU和句子嵌入余弦距离评估词汇与语义多样性；采用COMET和BERTScore评估质量；利用Hypervolume指标量化多样性-质量权衡。

**📊 数据集**

实验数据集涵盖四个任务：翻译（WMT24++日英）、摘要（XL‑Sum）、故事生成（WritingPrompts）和对话生成（DailyDialog），使用7B LLaMA 2和Qwen 2.5的预训练与对齐版本。

**📈 对比分析**

与温度采样比较时，融合模型在大多数设置下取得更高的Hypervolume，尤其在词汇多样性上显著优于温度采样；并且能够在不同i值下灵活调节多样性与质量，提供更宽广的权衡空间。

**⚠️ 局限性**

局限性包括：仅验证两种7B模型，扩展到其他规模与族类需要进一步验证；权衡范围受基模型与对齐模型属性限制；缺乏自动化调节i的策略；融合后可能失去对齐模型的安全性，需要额外安全评估。

---

## 314. Self-Evolving Coding Agents via Harness-Policy Optimization

**arXiv ID:** 2610.22791 | [PDF](https://arxiv.org/pdf/2610.22791v1)

**作者:** Zhepei Wei `[一作]` (Microsoft Research), Jianfeng Gao `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种递归自我改进框架Self‑HPO，交替进行利用现有策略提出新的 harness 并在该 harness 上进行强化学习更新策略，随后以更新后的策略继续改进 harness，形成闭环迭代；

**💡 创新点**

创新点在于：①让 agent 自己的策略充当 harness 提案者，省去外部 proposer；②利用上一轮策略训练中出现的难题动态更新搜索集，使 harness 搜索始终对准当前策略的失败模式；③证明两者协同优化的提升大于单独提升之和；

**🔧 技术方法**

采用 Qwen3.5‑35B‑A3B 作为策略模型；利用 Group Relative Policy Optimization (GRPO) 与 Balanced Adaptive Rollout (BAR) 的组合进行 RL；self‑proposer 通过分析失败轨迹生成 harness 变体；可选的外部提案者 Opus 4.8；harness 通过增设预算警告、空补丁检查、接口检验等规则进行改进；

**📊 数据集**

训练数据集共 2,484 条软件工程任务，来源于 SWE‑rebench、SWE‑rebench V2 与 Scale‑SWE；评估基准为 SWE‑Bench Verified（500）、SWE‑Bench Multilingual（300）与 SWE‑Bench Pro（642）；

**📈 对比分析**

与单侧优化基线（policy‑only RLVR、harness‑only Self‑Harness）以及外部模型基线（SFT、Meta‑Harness）进行对比；在 Verified 上 Self‑HPO 由 61.0% 提升至 69.5%（平均提升 64.8% 对比 57.8%），在多语言和 Pro 集合亦保持领先；联合优化收益超过单项之和，且在 OOD 任务上迁移良好；

**⚠️ 局限性**

局限性包括：①RL 信号稀疏导致训练需要大量 roll‑outs；②zero‑success 组仍存在，影响奖励稀疏性；③方法主要在 Python 代码修复任务上验证，跨语言或更复杂的环境尚未充分测试；④外部提案者可进一步提升效果，但增加了依赖外部模型的复杂度。

---

## 315. Native Geometry vs. Linear Recoverability: How Alignment Changes Platonic Scaling

**arXiv ID:** 2610.22749 | [PDF](https://arxiv.org/pdf/2610.22749v1)

**作者:** Angus Hoy `[一作]` (EleutherAI SOAR), Michael J. Smith `[通讯]` (AstroAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

探讨大模型在不同天文调查（Legacy Survey 与 Hyper Suprime‑Cam）中的表示是否趋于相似，并比较原生余弦相似度与通过Ridge线性映射后可恢复的相似度随模型规模的变化

**💡 创新点**

发现在线性映射后可恢复的相似度随模型规模的提升远快于原生余弦邻域相似度；映射虽高度各向异，却在低维子空间内实现大部分邻域恢复，表明相似度提升主要来自特定方向的变形，而非全局几何收敛

**🔧 技术方法**

使用Ridge回归（带正则化的线性映射）、主成分分析、稀疏自动编码器、块稀疏特征器以及余弦k‑近邻相似度度量；对齐后对齐矩阵的奇异值分解用于分析各向异性与功能恢复的维度差异

**📊 数据集**

匹配的Legacy Survey 与 Hyper Suprime‑Cam 嵌入数据集（共16384对象），覆盖AstroPTv2、ConvNeXtv2、DINOv2、ViT、I‑JEPA等五大模型族，包含16个模型规模层级（I‑JEPA仅2层）

**📈 对比分析**

与原生余弦相似度对比：在10个k的mKNN评估中，Ridge对齐将平均重叠从约0.022提升至0.042（相当于两倍原生水平，≈13倍随机机会）。该提升随模型规模的对数增加呈正斜率（T≈0.007），且在将两组嵌入都投影到256维后仍保持；对齐的线性组件主奇异值子空间仅占总维度4.8%即可恢复90%的提升，显示低维可恢复性强

**⚠️ 局限性**

仅检验了五大模型族，使用的是配对Ridge监督且模型权重冻结；未探索更大规模模型的极端情况；单一天文调查设置可能限制结论泛化；对齐模型为全局线性，可能无法捕捉更复杂的局部几何对应

---

## 316. Untrained CNN and coefficient-space regularization for joint reconstruction in photoacoustic tomography

**arXiv ID:** 2610.23315 | [PDF](https://arxiv.org/pdf/2610.23315v1)

**作者:** Alfio Bozi `[一作]` (University of Wurzburg), Souvik Roy `[通讯]` (University of Texas at Arlington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于固定随机卷积特征、波函数表示与系数空间正则化的光声显像联合重建框架，用于同时恢复初始压强和空间变异声速；

**💡 创新点**

创新点在于将无训练的随机CNN特征与Tikhonov正则回归和统计系数空间先验相结合，既降低了计算成本，又显著提升了对未知声速的重建精度；

**🔧 技术方法**

采用Haar小波表示、固定随机卷积网络、Ridge回归、以及学习到的线性系数空间先验；

**📊 数据集**

使用5000个合成的二维光声样本（4000训练、1000测试）以及三种未见过的形状（圆盘、高斯、心肺）进行验证；

**📈 对比分析**

与传统的全监督CNN方法比较，固定特征+Ridge方案在初始压强和声速重建误差上均更低（均方相对误差分别从约0.57降至0.25，声速误差从0.014降至0.008），同时计算时间约为训练CNN的1/6；

**⚠️ 局限性**

局限性包括：仍需统计耦合假设，独立声速成分无法完全恢复；仅在二维合成数据上验证，缺乏实验或三维验证；对模型误差、噪声类型等更广泛情况的鲁棒性尚未充分评估。

---

## 317. Learning User Representations to Improve Simulation Fidelity

**arXiv ID:** 2610.22518 | [PDF](https://arxiv.org/pdf/2610.22518v1)

**作者:** Shuhaib Mehri `[一作]` (University of Illinois Urbana-Champaign), Jianfeng Gao `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练并部署了一个结构化的用户表征生成器，通过多轮回顾性反馈与回顾性蒸馏提升用户模拟的精确度。

**💡 创新点**

创新点在于提出回顾性反馈与回顾性蒸馏两种在线训练方法，能够在不需昂贵多轮模拟的情况下直接优化表征生成器，使得生成的表征更能驱动模拟器重现真实用户行为。

**🔧 技术方法**

使用的技术包括：结构化表征（任务描述 + 行为描述）、多环境评判器、LLM 生成与评判、负对数似然（SFT/OSFT）、GRPO、OPSD 等强化学习和自蒸馏技术。

**📊 数据集**

使用的主要数据集有七个：WildChat、LMSYS-Chat-1M、PRISM、SWE-Chat、ThoughtTrace、WildChat Coding 与 WildChat Writing，共约700条真实用户‑AI 对话。

**📈 对比分析**

评价方法采用 GPT‑5.4 mini 作为用户模拟器、助手和评判器，计算最佳-1/最佳-3 的模拟精度。与基线（无结构化表征）及 GPT‑5.5 生成的表征相比，OSFT 训练的模型在最佳-1 及最佳-3 上提升约 1–2 分（从 3.41 到 6.10/6.14 → 7.07），显著优于传统方法。

**⚠️ 局限性**

主要局限包括：仍需多轮模拟才能获得回顾性反馈，成本较高；仅在英文公开对话上验证，跨语言/人群的泛化未知；生成的表征可能仍缺失细节，导致模拟器偏离真实用户；存在隐私与偏见风险，需谨慎使用。

---

## 318. Cooperation Without Adaptation: LLM Agents in a Collective-Risk Dilemma

**arXiv ID:** 2610.22882 | [PDF](https://arxiv.org/pdf/2610.22882v1)

**作者:** Trung-Kiet Huynh `[一作]` (University of Science, Vietnam National University), The Anh Han `[通讯]` (Teesside University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了五种主流大语言模型在集体风险游戏中的合作行为，探讨了风险、提示、合作伙伴、混合桌子和种群选择对其决策的影响，发现这些模型往往遵循默认策略，且在不同情境下表现不一致；

**💡 创新点**

创新点在于将经验博弈理论与大语言模型结合，设计了多维度的实验框架（风险梯度、提示控件、固定伙伴、混合队伍、种群选择），并通过对比公平基准揭示了自我对弈成功并不等同于真正的策略适应；

**🔧 技术方法**

使用了大型语言模型推理、游戏模拟、经验博弈分析、有限种群进化动力学、Bootstrap统计和Hypergeometric选拔概率计算等技术；

**📊 数据集**

实验基于对五款公开模型（Claude Haiku 4.5、Gemini 3.5 Flash‑Lite、GPT‑5.6 Luna、Qwen3‑235B、Grok 4.20）的多轮游戏数据，未使用外部标准数据集；

**📈 对比分析**

通过与公平份额基准和占优/劣化对比，对各模型在不同风险下的贡献、获利以及目标达成率进行统计，结果显示大部分模型的贡献随风险变化不大，且在混合桌子和种群选择中表现差异显著，表明自我对弈成绩并不能单独评估协作能力；

**⚠️ 局限性**

局限包括仅测试五个模型、单一语言和提示空间、每个条件仅十局游戏、未考虑语言多样性、对风险中立假设和种群选择模型的简化假设，且缺乏对实际部署情境的验证。

---

## 319. JAVIS: Joint Audio-Visual Interaction and Streaming for Real-Time Conversational Agents

**arXiv ID:** 2610.22499 | [PDF](https://arxiv.org/pdf/2610.22499v1)

**作者:** Junseok Ahn `[一作]` (Korea Advanced Institute of Science and Technology), Ji-Hoon Kim `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一种端到端实时音视频对话系统JAVIS，能够同时生成同步的语音和面部视频。

**💡 创新点**

创新点在于使用Joint Refined Head将音频和视频共享一个低维表示，避免中间瓶颈，实现真正的端到端流式生成，并构建大规模合成语音视频数据集JAVIS‑Syn。

**🔧 技术方法**

技术包括LoRA适配的Joint Refined Head、5Hz共享LLM与25Hz语音细化头、流式解码器、低维面部运动基底、Flow匹配训练与Euler ODE采样、Motion Renderer。

**📊 数据集**

使用的数据集有Talker‑T2AV、MultiDialog、Seamless Interaction、DailyTalk、ShareChatX‑Emotion、VoiceAssistant‑400K，及新构建的334小时合成语音视频数据集JAVIS‑Syn。

**📈 对比分析**

在VStyle样本上与Ex‑Omni‑2D和三种级联系统（FAC+FLOAT、FAC+InfiniteTalk、FAC+Talker‑T2AV）对比，JAVIS在初始延迟（1.27s）和实时因子（0.54）上均显著领先，同时在口型同步和语音自然度方面达到或超过基线。

**⚠️ 局限性**

局限包括仅支持单向对话、面部运动维度有限、合成数据可能缺乏真实多样性，且缺乏完整的双向交互与非面部行为建模。

---

## 320. VesperCSI: A High-Performance and Extensible CSI Acquisition Framework for Edge Wi-Fi Sensing

**arXiv ID:** 2610.22581 | [PDF](https://arxiv.org/pdf/2610.22581v1)

**作者:** Kaan Baydemir `[一作]` (Middle East Technical University), Ilkay Ulusoy `[通讯]` (Middle East Technical University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并评估了VesperCSI，一款针对Intel AX200/AX210在aarch64 Linux边缘设备上实现高速、可扩展CSI采集的开源工具，采用预分配对象池、双线程缓冲和插件接口；

**💡 创新点**

① 重新设计采集管道，使用固定容量对象池与生产者-消费者分离，显著提升序列保留率；② 提供版本化插件接口，支持实验特定监控、存储与处理；③ 修正继承自FeitCSI的内存与I/O瓶颈；

**🔧 技术方法**

利用Intel AX200/AX210的CSI‑enabled驱动backport，C11原子锁和自旋锁实现双线程并发；采用netlink通信、预分配内存池、vectorized文件写；插件通过共享库动态加载；

**📊 数据集**

在Raspberry Pi Compute Module 4上使用Intel AX210设备采集10 000帧CSI突发数据，作为实验数据集；

**📈 对比分析**

在相同硬件、相同内核6.12环境下，将VesperCSI与FeitCSI在600 MHz核心频率、2500帧/s目标率下进行对比；VesperCSI在10 000帧突发中序列保留率为99.0–99.1%，远高于FeitCSI的44–46%；同时观察FeitCSI内存增长与崩溃；

**⚠️ 局限性**

实验仅覆盖单一硬件平台和特定工作负载；插件模型仅支持单一插件，慢插件可能导致丢包；未对多线程同步开销或CPU占用做细粒度分析；

---

## 321. Rank Measures of Fine-Tuning Updates Need a Noise Baseline

**arXiv ID:** 2610.22438 | [PDF](https://arxiv.org/pdf/2610.22438v1)

**作者:** Muhammad Usman Altaf `[一作]` `[通讯]` (Independent Researcher), Muhammad Usman Altaf (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究语言模型微调时权重矩阵更新的秩特征，比较多种秩度量与随机矩阵、预训练权重等基准，探讨对齐训练是否留下特殊印记，并评估后训练的“表面性”主张。

**💡 创新点**

创新点在于提出在每个秩度量下与噪声基准并列展示的分析框架，揭示微调更新既近似全秩又高度集中于少数主导方向，证明“表面对齐”结论缺乏足够证据。

**🔧 技术方法**

使用奇异值分解计算各层权重矩阵的谱，评估阈值计数、能量秩、有效秩、参与比率、稳定秩等五种秩度量，并与随机高斯矩阵、预训练权重、bf16 舍入模拟以及公开模型更新进行对比。

**📊 数据集**

在 Qwen2.5‑0.5B 与 SmolLM2‑360M 等模型上进行 SFT、Wikipedia‑103、CodeParrot‑Python、词序打乱等四种训练任务，并对比 OLMo 2 1B 的 SFT→DPO→RL 三阶段和 Qwen2.5‑Math/​Coder 专家模型。

**📈 对比分析**

通过层级重采样的 95% 置信区间，将每个秩度量的平均值与同尺寸随机矩阵的基准对齐；计数度量与噪声相近（≤5%差距），而稳定秩与能量秩等显著低于噪声（≈70–90%差距），表明更新既近似全秩又高度集中；后训练的有效秩变化与随机更新相同，证明“表面性”假设不成立。

**⚠️ 局限性**

局限性包括仅在少数模型与任务上验证、种子数有限；低精度 bf16 舍入可能掩盖小更新；对齐训练只测试了单个模型；缺乏对其他优化器或更大规模训练的探索；编辑实验仅在单个基准和模式下进行，结果不具普适性。

---

## 322. Scene-Designer: Learning Design Intelligence for Agentic 3D Scene Generation

**arXiv ID:** 2610.23121 | [PDF](https://arxiv.org/pdf/2610.23121v1)

**作者:** Xuanyu Liu `[一作]` (Zhejiang University), Xuhong Zhang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Scene-Designer框架，学习3D场景生成的设计与批评能力。

**💡 创新点**

构建Scene-OTA-35K与Scene-OTC-15K两大数据集，并提出Critic‑guided GRPO优化方法，使模型既能高效修正错误，又能进行多维度场景批评。

**🔧 技术方法**

使用LLM+VLM模型(Qwen3‑VL‑8B‑Instruct)进行全参数SFT，结合因果轨迹压缩与场景扰动生成的偏好数据，随后采用Critic‑guided GRPO（强化学习）进行训练。

**📊 数据集**

Scene-OTA-35K（1k场景、35k交互）与Scene-OTC-15K（15k多场景比较），并使用公开的3D‑FRONT、3D‑SynthPlace等数据做对比。

**📈 对比分析**

与LayoutGPT、Holodeck、I‑Design、SceneWeaver、DiffuScene等基线在物理有效性、功能性、布局连贯性与现实感四个维度进行定量评估。Scene-Designer在所有维度均显著提升，尤其在功能性与布局连贯性上提升约20%以上，且在四场景排序准确率上达76%。

**⚠️ 局限性**

仍需人工标注的偏好数据，难以覆盖极端或复杂场景；Critic在面对极端扰动时可能失效；模型在大规模场景生成时计算成本仍较高。

---

## 323. Task Structure and Scalability in Multi-Robot Authoring

**arXiv ID:** 2610.22670 | [PDF](https://arxiv.org/pdf/2610.22670v1)

**作者:** Christian Marowski `[一作]` (Indiana University Bloomington), Byung-Cheol Min `[通讯]` (Indiana University Bloomington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文研究了多机器人创作中的任务结构和可扩展性，比较了单令牌分配和共享令牌分配在绘图任务中的表现。

**💡 创新点**

创新点在于通过385个绘图样本和不同数量的机器人，实证比较了两种分配-路由管道的可扩展性，揭示了任务结构对多机器人协作的影响。

**🔧 技术方法**

使用了仿真技术，结合几何笔画图表示法，分析了不同分配策略下的完成时间和速度提升。

**📊 数据集**

使用的数据集为Google Quick, Draw!中的385个用户生成的绘图，涵盖了不同的组件数量和图形特征。

**📈 对比分析**

通过比较两种分配策略的完成时间，发现共享令牌分配在大多数情况下表现更优，尤其是在机器人数量增加时，单令牌分配的速度提升逐渐减小，而共享令牌分配则持续提高。

**⚠️ 局限性**

限制在于并非所有绘图任务都能从共享令牌分配中受益，且仿真结果中存在几何机器人重叠的情况，不能完全保证无碰撞的完成时间测量。

---

## 324. Towards a GraphNARX model for structural dynamics

**arXiv ID:** 2610.23103 | [PDF](https://arxiv.org/pdf/2610.23103v1)

**作者:** George Tsialiamanis `[一作]` (University of Sheffield), Keith Worden `[通讯]` (University of Sheffield)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于图神经网络的 GraphNARX 模型，用于结构动力学的非线性自回归外部输入（NARX）系统识别，并实现了对缺失传感器的虚拟感知。

**💡 创新点**

创新点在于：①将结构几何信息映射为图结构，使模型可在不同尺寸和自由度的结构间迁移；②通过局部信息传递实现对未观测节点位移的推断；③利用图神经网络的可扩展性实现从小型到大型结构的泛化。

**🔧 技术方法**

采用了图神经网络（GNN）的消息传递框架（边更新、节点更新、全局更新）结合 NARX 时滞输入；训练采用均方误差（MSE）与归一化均方误差（NMSE）评估；虚拟感知通过输入掩蔽和加权损失实现。

**📊 数据集**

使用了基于离散质量-弹簧-阻尼系统的人工合成数据集：10 自由度（训练/验证/测试）和 20 自由度（仅测试）各 20,000 步长为 0.01s 的时间序列；噪声为 1% 白噪声。

**📈 对比分析**

对比方法：同一模型在 10 DOF 与 20 DOF 的一阶预测（OSA）和多阶预测（MPE）进行比较；虚拟感知则分别在 50% 以及 60% 传感器缺失率下评估。性能方面：10 DOF OSA NMSE 0.0204%，20 DOF OSA NMSE 0.0228%；多阶预测 400 步内 NMSE <5%，随后线性升高；虚拟感知 10 DOF NMSE ~5%，20 DOF 约 2%–3%。

**⚠️ 局限性**

局限性包括：仅在有限的 GNN 结构上实验；训练与评估全部基于合成数据，缺乏实验验证；模型选择仅依据一阶预测误差，未考虑多阶预测；缺少与传统 FE 或其它机器学习方法的直接比较；对更大自由度结构的泛化尚未深入探索。

---

## 325. Scaling Online Exploration for Robot Agents

**arXiv ID:** 2610.22496 | [PDF](https://arxiv.org/pdf/2610.22496v1)

**作者:** Runhan Huang `[一作]` (Harvard University), Yilun Du `[通讯]` (Harvard University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种基于LLM的机器人编码代理，通过在线探索生成可执行程序来完成多种机器人任务。

**💡 创新点**

创新点在于将探索经验直接编译为代码程序，实现无监督、自主学习并可持续更新技能；不需额外演示或参数训练。

**🔧 技术方法**

使用大型语言模型、机器人感知接口、程序合成、持续记忆存储与多代理并行探索。

**📊 数据集**

在多种模拟与真实机器人平台（RoboDojo、BEHAVIOR‑1K、HumanoidArena、DexJoCo、LIBERO、Franka、Rainbow RB‑Y1等）上评测。

**📈 对比分析**

与现有最强基线对比，平均提升约+37.4点，在多项细粒度与长时序任务中遥遥领先，零演示条件下也能取得最高分。

**⚠️ 局限性**

局限在于依赖大型LLM和工具接口，学习速度受推理延迟影响，对极端动态环境的鲁棒性尚待验证。

---

## 326. At the Limits of Computability: Steps Towards Autonomous AI Discovery on the Busy Beaver Frontier

**arXiv ID:** 2610.22603 | [PDF](https://arxiv.org/pdf/2610.22603v1)

**作者:** Marco Dos Santos `[一作]` (Google DeepMind), Tom Zahavy `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个基于 Gemini LLM 的全自动化管道，自动识别、猜测、证明并在 Lean 4 上形式化 Busy Beaver 机器的 Collatz‑style 加速规则；

**💡 创新点**

首次实现从 Turing 机执行数据到数学规则的端到端自动推理，结合交互式模拟、生成‑评估循环以及多人格 Lean 推理器，显著提升了对复杂 Holdout 机器的分析能力；

**🔧 技术方法**

使用 Gemini 3.1 Pro（LLM）进行模式检测、规则推导和非正式证明，Lean 4 进行形式化验证，并辅以自定义 Python 模拟器与压缩算法；

**📊 数据集**

基准数据集为 1,094 个 BB(6) Holdout、65 个 BB(2,5) Holdout 以及 4 个 BB(3,3) Holdout 机器；

**📈 对比分析**

与社区手工推导和已知规则对比，系统在 11 个此前未解的 BB(6) 机器上成功给出形式化加速规则，覆盖率约 27%，证明成功率超过 90%；

**⚠️ 局限性**

局限在于仅处理 Collatz‑style 结构，未能扩展到更一般的不可判定动力学；模型在极大规模 Turing 机空间下的计算成本仍高，且对极端稀疏或非典型行为的检测不够充分。

---

## 327. Inductive Learning of Concept Embeddings from Cross-Linguistic Colexification Patterns with Textual Attributes

**arXiv ID:** 2610.22638 | [PDF](https://arxiv.org/pdf/2610.22638v1)

**作者:** Arne Rubehn `[一作]` `[通讯]` (University of Passau), Arne Rubehn (University of Passau)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一种基于跨语言共词化网络和概念短描述的可扩展诱导概念嵌入框架。

**💡 创新点**

突破传统闭集转导嵌入的局限，利用文本特征实现可对未见概念生成向量，兼顾全/部分共词化结构。

**🔧 技术方法**

使用随机游走+Skip‑gram的图嵌入结合Sentence‑BERT编码的特征表示，形成诱导图神经网络。

**📊 数据集**

CLICS⁴ 与 CLIPS 共词化图、Concepticon 概念描述，评测采用 MultiSimLex、DatSemShift 与 Edinburgh Association Thesaurus。

**📈 对比分析**

与两种最优转导模型 Node2Vec、ProNE 及 SBERT 基线在闭集上对比，转导略优但诱导模型在未见概念上保持稳定，并在语义转移与词联想任务中表现优异。

**⚠️ 局限性**

模型对概念定义与共词化数据的误差敏感，缺乏对不同架构与训练方案的系统评估，并受限于 Concepticon 概念的主观性。

---

## 328. COMET: Decoupled Distillation, Routing, and Capacity Control for Task-Agnostic Continual Vision--Language Learning

**arXiv ID:** 2610.22907 | [PDF](https://arxiv.org/pdf/2610.22907v1)

**作者:** Seungyoon Woo `[一作]` (Seoul National University), Gunhee Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 COMET 框架，在不重训练主干、无任务标识、无回放的条件下，实现 VLM 的持续学习，保持多模态几何、协调专家、控制容量。

**💡 创新点**

创新点在于将模态一致性漂移、专家间交叉干扰、任务容量短缺三大失效模式分别拆分为专门的机制（TAD、SACED、CMAB），实现角色分离，并通过冻结教师做几何参照、重建路由等技术实现稳健的连续适应。

**🔧 技术方法**

使用的技术包括：冻结教师 CLIP 作为多模态几何参考；Tri‑Affinity Distillation 维护图像‑图像、文本‑文本、图像‑文本相似度；Student‑Aware Cross‑Expert Distillation 通过注意力对齐兼容专家；上下文多臂老虎机（CMAB）进行专家容量控制；重建路由器在测试时选择稀疏专家；以及 TinyCLIP 轻量化专家网络。

**📊 数据集**

使用的数据集包括 X‑TAIL 的十个图像分类任务（Aircraft、Caltech101、DTD、EuroSAT、Flowers、Food、MNIST、OxfordPet、StanfordCars、SUN397）以及医学、显微、组织病理四任务序列（Covid19、SEM、Kather‑Texture、NCT100K）。

**📈 对比分析**

与 WiSE‑FT、ZSCL、MoE‑Adapters、LADA 等基线对比，在 X‑TAIL 上实现 Transfer、Average、Last 三项指标最高（尤其 Last 提升 19.5pp）；在四任务序列中比 ZSCL 提升 Transfer +18.7pp、Average +20.2pp、Last +3.5pp，且保持专家池子子线性增长。

**⚠️ 局限性**

局限性包括：子线性存储增长依赖累计新颖度降低，长时间进入新领域时可能线性增长；CMAB 缺乏严格的收敛/保证；未给出显式存储预算和容量控制的理论保障；对极端新域的迁移性能尚待验证。

---

## 329. SCOPE: Selective Certification of Program Edits for Embodied Heuristic Learning

**arXiv ID:** 2610.23128 | [PDF](https://arxiv.org/pdf/2610.23128v1)

**作者:** Junyang Shu `[一作]` (Wangxuan Institute of Computer Technology, Peking University), Yongtao Wang `[通讯]` (Wangxuan Institute of Computer Technology, Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过失败与成功执行记录，利用诊断图和选择性回放对机器人控制程序进行增量修正；

**💡 创新点**

创新点在于将失败前沿定位、历史成功回放和稀疏环境诊断结合，形成“Selective Certification Of Program Edits”(SCOPE)框架；

**🔧 技术方法**

使用前沿引导程序搜索、状态化全轨迹API回放、诊断图标注以及稀疏环境诊断等技术；

**📊 数据集**

实验数据集包括LIBERO-PRO（Object、Goal、Spatial 60 任务-设置对）、Robosuite（7个抓取/放置任务）和Quanta X2机器人实际任务；

**📈 对比分析**

与ASPIRE等基线对比，SCOPE在LIBERO-PRO上平均成功率80.66%（比ASPIRE高6.51pp），在Robosuite上84.48%（比ASPIRE高3.48pp），在真实机器人任务上整体成功率显著提升；

**⚠️ 局限性**

局限性包括高昂的LLM推理成本、对API边界完整性的强假设以及在物理机器人上可能产生不安全行为需要人工监督。

---

## 330. alkhder at ImageEval 2026 Shared Tasks: Constrained Selection and Rotation-Averaged Logit Scoring for Multimodal Hallucination Detection

**arXiv ID:** 2610.22515 | [PDF](https://arxiv.org/pdf/2610.22515v1)

**作者:** Maher Pasa `[一作]` (Altınbaş University), Hasan Alkhder `[通讯]` (Sakarya University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用 7B 视觉‑语言模型，在零样本设置下通过对图片和三条声明进行点式和多项式评分，结合 Latin‑square 旋转平均以及对下一词概率的对数聚合来检测哪条声明与图片内容一致。

**💡 创新点**

创新点在于：① 在多项选择场景下直接读取对数概率而非生成文本，避免分词误差；② 对答案位置进行 Latin‑square 旋转平均以消除位置偏差；③ 采用点式与多项式评分的融合，并通过约束“恰好一条声明为真”提升判定可靠性。

**🔧 技术方法**

技术实现基于 HuggingFace 上的 7B 预训练视觉‑语言模型（如 LLaVA/ViT‑LLaMA），使用 LogSumExp 池化、log‑softmax 归一化、点式与多项式两种评分方式，最后用加权融合得到最终决策。

**📊 数据集**

使用 ImageEval 2026 Subtask 1b 数据集，包含英、阿拉伯两条轨道的图片与三条声明，分别划分为 3000 条开发、500 条验证、500 条公开测试、1000 条隐藏测试。

**📈 对比分析**

通过与官方 3B 基线比较，CI（Contrastive Instability）指标分别从 0.3133/0.4900 降至 0.0510/0.0960（相对降低 83.7%/80.4%），在英轨道排名第 5、阿拉伯轨道排名第 4，整体准确率分别为 94.9%/90.4%。

**⚠️ 局限性**

局限性包括：① 仅在英语验证，阿拉伯版本缺乏专门调优导致 4.5 分的准确率差距；② 未评估提示或答案词组的多样性与鲁棒性；③ 只针对“恰好一条声明为真”的数据集约束，无法直接迁移至开放式幻觉检测任务。

---

## 331. Efficient Dialect-Agnostic SQL Equivalence Refutation

**arXiv ID:** 2610.22442 | [PDF](https://arxiv.org/pdf/2610.22442v1)

**作者:** Junwen An `[一作]` (National University of Singapore), Manuel Rigger `[通讯]` (National University of Singapore)

**通讯引用:** 725 | [OpenAlex ID](https://openalex.org/A5066738024)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了SQLWitness，一个基于DBMS无关的模糊测试框架，用来快速定位两条SQL查询的语义不等价实例。

**💡 创新点**

创新点包括：①查询感知的候选值池（基于三值逻辑与语句字面量），②约束感知的数据生成策略，③利用谓词覆盖（APC）与重置型拉普拉斯估计器实现智能终止，所有这些均不依赖于手工编码SQL方言语义。

**🔧 技术方法**

采用的技术：模糊测试、约束感知实例生成、SQLGlot解析字面量与谓词、SQLAlchemy统一DBMS交互、Approximate Predicate Coverage、Reset‑based Laplace Estimator、统计终止阈值。

**📊 数据集**

使用了六个公开基准套件，共26,665条查询对，覆盖四种主流SQL方言（MySQL、PostgreSQL、SQLite、SQL Server）。

**📈 对比分析**

与VeriEQL、SpotIt、EvoSQL、VeriEQL等基线工具在相同数据集上对比：SQLWitness在等价否定率上提升5.34%–88.4%，单对平均耗时0.12秒，速度比基线快2.1–16.4倍。

**⚠️ 局限性**

局限性：对需要特定算术关系或特殊NULL/字符串语义的查询差异识别不足，随机采样导致偶尔漏检；对极端非确定性查询（如ORDER BY无唯一键）可能产生伪伪证或不完整的覆盖。

---

## 332. Mothership: A Two Tier Autonomous Refuse Collection System for Pedestrian Campuses

**arXiv ID:** 2610.22461 | [PDF](https://arxiv.org/pdf/2610.22461v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 333. Where Research Ideation Begins: Forward-Citation Supervision for Actionable Limitation Generation

**arXiv ID:** 2610.23038 | [PDF](https://arxiv.org/pdf/2610.23038v1)

**作者:** Liyang Zhao `[一作]` (Westlake University), Haibin Ling `[通讯]` (Westlake University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了利用前向引用监督生成可操作研究局限的框架，并构建了包含18,279条证据关联三元组的ActLim-Bench数据集。

**💡 创新点**

创新点在于将前向引用作为可验证的监督信号，并设计三阶段训练流程（SFT、RSFT、GRPO）来显式学习、扩展和优化局限生成。

**🔧 技术方法**

采用了大型语言模型（如Qwen3-8B/32B）进行监督微调、拒绝采样过滤、基于验证器的增量学习以及GRPO强化学习技术。

**📊 数据集**

使用了ActLim-Bench数据集，涵盖2017–2025年NLP文献，并在CV领域进行零样本迁移评估。

**📈 对比分析**

通过与零样本LLM、提示式、检索和多代理系统对比，ActLim-8B在NLP测试集上的GRec@3为0.452、SpecRec@3为0.241，CV迁移为0.314，明显优于对照组。

**⚠️ 局限性**

主要限制是模型在生成阶段仍受监督信号不足的瓶颈，且数据仅覆盖后续被采纳的局限，无法涵盖所有潜在可行局限。

---

## 334. OmniXRef: A Unified Framework for Multimodal Multi-Reference Video Editing

**arXiv ID:** 2610.22744 | [PDF](https://arxiv.org/pdf/2610.22744v1)

**作者:** Xiaoyan Cong `[一作]`, Soo Ye Kim `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一框架 OmniXRef，支持通过文本、图像和视频多模态多参考对源视频进行多属性编辑。

**💡 创新点**

创新点包括：1）reference‑bank 接口实现可变数量多模态参考组合；2）Mix‑Patchify 通过任务特定 patch 大小降低多参考计算成本；3）RoPE with Reserved Reference Slots 解决训练–推理中参考数目差异问题；4）大规模合成数据集 OmniVideo‑10M 与新基准 OmniVBench 的构建。

**🔧 技术方法**

技术实现结合 MLLM（Qwen3‑VL‑8B‑Instruct）与单流 Diffusion Transformer（DiT），并采用 Mix‑Patchify、RoPE‑RRS、Curriculum Learning、参考银行 dropout 等策略。

**📊 数据集**

使用的数据集包括：OmniVideo‑10M（约10M条训练样本，含文本、图像、视频参考组合）；ReCo‑Bench、VEFX‑Bench、CamCloneMaster、MotionFollower、RefVFX 等公开数据；以及新构建的 OmniVBench（125案例）用于系统评估。

**📈 对比分析**

通过与商业模型 Seedance 2.5、MiniMax‑H3 以及任务专用模型 CamCloneMaster、MotionFollower、RefVFX 进行对比；使用 VLM 判别器评估 Reference Adherence、Visual Quality、Source Preservation，OmniXRef 在多项指标上均超过基线，整体 Score 最高。

**⚠️ 局限性**

局限性：VQ 质量仍略低于部分商业系统；训练样本中多参考组合仍有限，可能导致更复杂场景的泛化受限；依赖大规模合成数据，对极端动态或高光照条件的表现尚未充分验证。

---

## 335. STC-MGNet: Spatial-Temporal-Channel Attention-Enhanced Multimodal Graph Network for Pedestrian Intention Prediction

**arXiv ID:** 2610.23356 | [PDF](https://arxiv.org/pdf/2610.23356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 336. Causal Value Attribution of Marketing Labels in LLM-Generated Ads: A Deconfounder Approach

**arXiv ID:** 2610.23025 | [PDF](https://arxiv.org/pdf/2610.23025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 337. Surveying Experimentally-Validated Components for DNA-Based Molecular Communication

**arXiv ID:** 2610.23361 | [PDF](https://arxiv.org/pdf/2610.23361v1)

**作者:** Andreas Waldner `[一作]` (University of Lübeck), Florian-Lennert Lau `[通讯]` (University of Lübeck)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统评估 DNA 结构、载体和传感器等实验验证的组件，为 DNA 基础分子通信网络提供最小 PoC 方案。

**💡 创新点**

提出统一的评分方法，将实验可行性、稳定性、成本等五项指标量化，并基于此识别最适合的组件组合。

**🔧 技术方法**

利用文献综述、分子设计原理与实验数据对 DNA‑tiles、纳米载体和生物传感器进行评估。

**📊 数据集**

主要使用公开实验数据与文献报告的测量结果（如 Tm、半衰期、成本估算）而非自建数据集。

**📈 对比分析**

通过将每个组件按 AoS、AoO、CwD、LoS、CoC 评分后求和，得出总分并推荐 SST、DX‑tile 及 MSN 作为 PoC 的首选；评分体系可与其他研究对标。

**⚠️ 局限性**

局限在于评估高度依赖已有报告，缺乏统一实验验证；某些组件仅基于理论或低证据评分，且未涵盖体内验证。

---

## 338. Variational and Decomposed method for Mitigating Bimodal Bias in Robust Visual Question Answering

**arXiv ID:** 2610.22845 | [PDF](https://arxiv.org/pdf/2610.22845v1)

**作者:** Yihan Zhao `[一作]` (South China Normal University), Yuncheng Jiang `[通讯]` (South China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对视觉问答中的多模态偏差，区分计数与非计数问题的偏差来源，提出针对性去偏方法VDB；

**💡 创新点**

创新点在于识别计数问题受多模态偏差主导、非计数问题受语言偏差为主，利用变分自编码器设计VB捕捉语言+微弱视觉偏差，BB使用双模态输入捕获计数偏差，并通过分解策略对不同问题类型分别训练；

**🔧 技术方法**

采用变分自编码器（VAE）框架、双模态融合、判别式去偏（重加权损失）、UMAP可视化和多模态特征分析；

**📊 数据集**

实验数据集包括VQACPv2、VQACPv1、GQAOOD以及原始VQAv2，评估OOV与ID性能；

**📈 对比分析**

与多种基线（如DVQA、GenB、RMLVQA等）对比，VDB在OOV（VQACPv2/1、GQAOOD）上取得最优或接近最优分数，尤其在计数子任务上提升显著，但在ID VQAv2上略有下降；

**⚠️ 局限性**

局限性在于去偏后ID性能可能下降，尤其计数特征对视觉模态高度依赖，且VB训练过程对重构损失与KL权重敏感，需精细调参。

---

## 339. VLA-Fabric: Learning Interaction for Composable Multi-Arm VLA Agents

**arXiv ID:** 2610.23018 | [PDF](https://arxiv.org/pdf/2610.23018v1)

**作者:** Chenglong Zhang `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Ningxin Su `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了VLA‑Fabric框架，实现多臂机器人系统中完整的Vision‑Language‑Action（VLA）策略通过学习的内部表示交互实现协同；

**💡 创新点**

创新点在于①将每个臂的完整VLA作为独立执行单元，②设计可在不同架构下可迁移的三种协调函数（共享语义上下文、同伴潜在上下文、动作意图交互），③通过系统化的架构搜索和交互学习实现多臂可扩展；

**🔧 技术方法**

使用的技术包括：基于Eagle2的SingleVLA与π_0.5两种VLA架构，交互模块（RAW Common、Private KV、Linear/Transformer融合），控制架构搜索、强化学习微调，仿真与真实机器人多臂执行；

**📊 数据集**

使用的任务数据集为多臂协同操作任务：Handover Box、Shoes Table、Handover Mic、Scan Object（仿真），Stack Cube、Frame‑over‑Peg、Arch Building（三、四臂）；以及四个真实世界任务（Basket3、Relay3、Relay4、Frame4）；数据来自内部收集的演示与自监督采样；

**📈 对比分析**

通过与独立训练的本地VLA策略以及不同交互关闭的ablation进行对比，Full交互模式在所有物理任务中较基线提升16–40%（例如Relay3 34%），在仿真任务中同样表现出显著优势，且交互策略在不同VLA架构间迁移保持稳健；

**⚠️ 局限性**

局限性包括：仅在同构且角色固定的三、四臂团队上验证；交互结构为静态，未考虑任务阶段动态适配；在长周期、高精度装配任务（Arch Building）仍难以稳定完成；缺乏对异构VLA和可变团队组成的通用支持。

---

## 340. UniField++: A Unified Agentic Framework with Loop Reasoning for MRI Field-Strength Enhancement

**arXiv ID:** 2610.22982 | [PDF](https://arxiv.org/pdf/2610.22982v1)

**作者:** Yiyang Lin `[一作]` (Chinese University of Hong Kong), Yixuan Yuan `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究提出了 UniField++，一种统一多代理框架，用于低场MRI到高场MRI的场强增强，并通过辐射科医生标注的幻觉风险区域与修复方案进行风险预测与循环推理，显著提升图像质量并保持解剖学可信度。

**💡 创新点**

创新点主要包括：①引入医生标注的幻觉风险和修复计划，利用VLM规划器预先识别高风险区域；②基于循环Transformer的流匹配增强器在高风险区域实现多轮推理，动态分配推理资源；③跨场强任务共享学习，提升对不同场强下组织对比与解剖可见性的理解。

**🔧 技术方法**

技术方法包括：VLM（Qwen3.5-9B）+LoRA用于规划器训练；流匹配框架（基于FlashVSR）+循环Transformer实现增强；使用ANTs进行空间配准、MONAI进行预处理；LoRA微调、交叉熵、风险加权L1损失等。

**📊 数据集**

使用公开的 64mT‑to‑3T（ULF‑EnC、Leiden、KCL）、3T‑to‑7T（UNC、BNU）以及未见的 50mT‑to‑3T（BTH）数据集，共计5个任务、3个模态（T1、T2、FLAIR）。

**📈 对比分析**

与 MO‑U‑NET、MSFA、LowGAN、FlashVSR、FCVSR 等 6 种基线相比，UniField++ 在 64mT‑to‑3T 任务中平均提升 PSNR 2.74 dB、SSIM 14.05pp；在 3T‑to‑7T 任务中平均提升 PSNR 3.88 dB、SSIM 15.9pp；在外部 50mT‑to‑3T 测试中保持或超过最佳基线；在下游全脑分割中，Macro Dice 由原始低场图像 0.5661 提升至 0.7982，提升 41%。

**⚠️ 局限性**

局限性包括：①依赖人工标注的幻觉风险与修复方案，标注成本高；②循环推理的迭代次数需要经验调优，过多循环会导致误差累积；③在极低场（如 50mT）任务仍存在泛化挑战，且对不同磁场硬件与扫描参数的适应性尚待进一步验证。

---

## 341. Shape-Aware Neural Time Fields for Rigid-Body Motion and Manipulation Planning

**arXiv ID:** 2610.22762 | [PDF](https://arxiv.org/pdf/2610.22762v1)

**作者:** Yanming Zhou `[一作]` (Purdue University), Ahmed H. Qureshi `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Shape-NTFields，一种针对刚体运动规划的物理信息神经时间场框架；

**💡 创新点**

创新点在于分别对平移与旋转给出几何约束的速度目标，并结合采样式规划在本地运动难度上做权衡；

**🔧 技术方法**

利用几何驱动的分块Eikonal监督、相对法线损失以及基于速度引导的采样规划技术；

**📊 数据集**

在多种二维与三维刚体形状（矩形、L、F、A、V、4、T）与四类环境上进行训练与测试；

**📈 对比分析**

与TDM、PC‑Planner、LazyPRM、RRTConnect等基线对比，Shape‑NTFields在3D任务中取得92.9%成功率、0.055s推理时间，显著优于TDM（63.7%）且在计算效率上远快于采样与搜索方法；

**⚠️ 局限性**

仅能为固定形状与静态环境学习单独场景，无法直接迁移至不同物体或动态障碍环境。

---

## 342. World Action Critic (WAC) Model: Joint Foresight and Criticism for Policy Learning

**arXiv ID:** 2610.22825 | [PDF](https://arxiv.org/pdf/2610.22825v1)

**作者:** Xiangteng Zhang `[一作]` (Tsinghua University), Shengbo Eben Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种整合候选动作生成、未来状态预测与评估的World Action Critic (WAC) 模型。

**💡 创新点**

创新点在于采用异步注意力和候选级别监督，使预测特征在递归预测过程中通过评估梯度自适应，从而能够对未执行的动作进行有效比较。

**🔧 技术方法**

技术方案包括基于Transformer的递归预测-批评器结构、AdaLN、Rotary位置编码以及多任务联合训练（未来预测、评估与动作生成）。

**📊 数据集**

实验使用NAVSIM基准数据集，覆盖多样化的城市驾驶场景。

**📈 对比分析**

与多种端到端、视觉语言动作和世界模型规划器对比，WAC在NAVSIM-v1的PDMS上取得93.4分，显著优于同类方法，尤其在驾驶进展指标上表现最突出。

**⚠️ 局限性**

局限性在于验证仅限于单一城市驾驶场景，缺乏对更广泛任务、多模态输入或不同环境的泛化评估。

---

## 343. LAW: Unifying Language, Action, and World Modeling for Agentic Physical Intelligence

**arXiv ID:** 2610.23334 | [PDF](https://arxiv.org/pdf/2610.23334v1)

**作者:** Jiajun Cao `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了LAW框架，通过闭环的视觉‑语言‑动作（VLM‑WAM）系统，将语言推理、动作生成和世界建模统一起来，并通过语义子任务进度实现任务执行的持续监控与动态切换。

**💡 创新点**

核心创新在于：① 以子任务进度作为语言与动作之间的时间接口；② 采用混合专家VLM（Planner、State、Value）实现规划、状态评估与价值引导；③ 在WAM中引入Mixture‑of‑Transformer Experts (MoTE) 以强化视频‑动作联合建模；④ 双流分层视觉记忆同步捕捉近期与子任务边界信息；⑤ 子任务局部价值学习利用失败轨迹中的已完成阶段进行监督。

**🔧 技术方法**

技术手段包括：Qwen3‑VL‑2B‑Instruct 作为冻结的VLM骨干；LoRA适配器与任务特定头；Fast‑WAM/OpenWAM‑α 作为动作生成器；MoTE‑WAM 模型；双流分层记忆网络；子任务价值评估网络；以及基于Transformer的混合注意力与专家路由。

**📊 数据集**

数据集与任务覆盖：RoboTwin（50 物理双臂操作任务，清洁与域随机化演示）；RoboDojo（35 任务，3,500 轨迹）；真实机器人三任务（Puzzle、Domino、Organize）共900 轨迹；以及用于训练失败模式的人工注释子任务边界与失败轨迹。

**📈 对比分析**

实验对比：与 Fast‑WAM、OpenWAM‑α 以及多种 VLA/WAM 基线进行对照。LAW 在 RoboTwin 上平均成功率 94.21%，使用 OpenWAM‑α 提升至 95.56%（公开领域最佳）。在三项真实机器人任务中，LAW 的平均成功率为 65%，远高于 π_0.5（28.33%）和 Fast‑WAM（25%），并在任务得分上取得显著优势。

**⚠️ 局限性**

局限性：依赖预先标注的语义子任务边界，若任务缺乏明显子任务结构则难以直接应用；模型规模为 2B 参数，部署成本仍相对较高；在极长或高度多样化的任务序列中，子任务划分的自动化与可迁移性尚未充分验证。

---

## 344. Can AI interview? Comparing AI-led to Human-led Chat Interviews

**arXiv ID:** 2610.23149 | [PDF](https://arxiv.org/pdf/2610.23149v1)

**作者:** Semra Guven `[一作]` (University of Copenhagen), Hjalmar Bang Carlsen `[通讯]` (University of Copenhagen)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过随机实验（N=41）对比了使用本地7B开源LLM的AI主导文本聊天访谈与训练非专业人类访谈者的访谈数据质量。

**💡 创新点**

首次实证验证基于本地开源LLM的AI访谈系统，强调开源模型在数据安全、GDPR合规和可重复性方面的优势。

**🔧 技术方法**

采用OpenHermes 2.5（Mistral 7B）在本地部署的多代理LLM架构（分类、改写、探询），并通过相同聊天界面实现访谈。

**📊 数据集**

受试者为41名丹麦大学生，访谈主题为学术求助经历，使用标准半结构化访谈指南。

**📈 对比分析**

通过Mann‑Whitney U检验和Hodges‑Lehmann估计比较曝光率、特异性和相关性，结果显示AI访谈每分钟产生约48%更多文本、提问速率翻倍，特异性与相关性与人类访谈基本相等。

**⚠️ 局限性**

样本量有限、仅采用文本聊天形式、未与专家访谈者比较、模型未针对访谈任务微调，限制了对人类访谈者优势的检测。

---

## 345. When to Forget: Risk-Controlled Context Adaptation for Long-Horizon Agents

**arXiv ID:** 2610.22897 | [PDF](https://arxiv.org/pdf/2610.22897v1)

**作者:** Chenle Chen `[一作]` (University of California), Lei He `[通讯]` (Eastern Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 RECAP 框架，对长期规划代理在执行过程中出现的失败记录进行动态处理，决定是保留（KEEP）、压缩（COMPRESS）还是删除（DROP），以实现成功率与提示代价的平衡。

**💡 创新点**

创新点包括：
1) 以执行状态为基准的“何时忘记”决策，将成功率与成本的相对影响量化为可比较指标；
2) 将基于前缀特征的预测（执行关系）与独立的风险控制（任务层风险）结合，实现安全可解释的编辑策略；
3) 通过固定失败块的纵向对比展示同一记录在不同执行阶段的价值变化，强调仅靠时间或语义相关性不足。

**🔧 技术方法**

主要技术：
- 匹配连续体（matched continuations）实验协议；
- 前缀信息提取 72 维执行关系特征（重叠、时序、状态变更、依赖等）；
- 随机森林/逻辑回归模型预测成功率变化与代价收益；
- 任务包级风险评估（成功率下降风险、代价收益非正风险）与 Clopper–Pearson 上限的学习-测试（Learn‑Then‑Test）策略；
- 固定策略与动态阈值网格的序号搜索。

**📊 数据集**

数据集：AgentCE‑Bench 的六个规划域（课程、餐饮、PC‑构建、购物、旅行、劳动力规划），使用 5 种后端模型（Qwen3‑4B/8B/14B、gpt‑oss‑20b、Hunyuan‑A13B）。

**📈 对比分析**

与 7 种对照策略（KEEP、Always‑COMPRESS、Always‑DROP、RECENCY、SEMANTIC、LEARNED‑NOCERT、RECAP）对比。评估指标包括：
- 继续执行成功率（接近 KEEP，最高差距 1.6%）；
- 任务级伤害率（RECAP 为 0%）；
- 代价节省（unconditional 2.13% / 4,619.8 tokens，paired‑success 约 5,000 tokens）；
- 触发率（RECAP 在 Hunyuan 上 43.24%）。结果显示 RECAP 在保持成功率的同时实现可观的提示代价降低，优于纯压缩/删除或仅基于时间/语义的启发式方法。

**⚠️ 局限性**

局限性：
- 依赖有限种子（finite‑seed）测量协议，无法覆盖所有随机性；
- 风险控制假设任务包独立同分布，实际部署中可能失效；
- 仅对单一失败块进行一次编辑，未探讨多编辑或动态压缩策略；
- 评价以匹配连续体为主，缺乏完整 episode‑level 成本/成功评估；
- 压缩模板固定，未考虑自适应或学习式摘要生成。

---

## 346. Response-Informed Value Estimation for Cost-Aware Multi-Turn LLM Routing

**arXiv ID:** 2610.23239 | [PDF](https://arxiv.org/pdf/2610.23239v1)

**作者:** Haotian Yan `[一作]` (Southwest University of Finance and Economics), Tianjun Yao `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了多轮LLM路由，利用每次交互的响应信息在训练阶段改进路由器，提出RIVE和RVE共同训练的框架；

**💡 创新点**

创新点在于在训练时使用已执行的响应作为辅助信息，构建响应信息辅助价值估计器RIVE，减少返回噪声，提升共享表示学习，部署时仍保持预调用决策；

**🔧 技术方法**

使用了离线轨迹的双值估计器（RVE/RIVE）、共享状态-模型表示、Monte Carlo预热、双DQN TD更新、保守Q学习（CQL）以及有限在线适配等技术；

**📊 数据集**

实验数据集包括MINT Reasoning子集（GSM8K、HotpotQA、MATH、MMLU、TheoremQA）用于训练/测试，以及MuSiQue、DROP、MMLU-Pro的数学/物理/化学子集用于OOD验证；

**📈 对比分析**

与单轮路由器、固定强模型以及MTRouter等基线对比，在MINT上精度从85.82%提升到87.94%，并将模型调用成本降低74.7%，在OOD场景仍保持低成本优势且性能优于基线；

**⚠️ 局限性**

局限性包括需要大量离线轨迹、依赖已知模型集，且在完全未知任务上适配速度受限；RIVE仅在训练时使用，若响应质量不佳可能影响学习效果。

---

## 347. Where Does the Performance Go? Understanding and Recovering Performance Regressions in C-to-Rust Translation

**arXiv ID:** 2610.23044 | [PDF](https://arxiv.org/pdf/2610.23044v1)

**作者:** Zhiqiang Yuan `[一作]` (Fudan University), Xin Peng `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究机械化 C-to-Rust 翻译导致的性能回归，提出基于已识别性能问题的规则指导 LLM 重写框架，并在 20 个真实项目上进行评测。

**💡 创新点**

首次通过成对 C 与 Rust 程序的 IR、优化提示和执行成本拆解，系统识别出九种常见性能问题并归纳为可复用的优化规则；将规则嵌入 LLM 重写流程，实现自动化性能恢复。

**🔧 技术方法**

利用 LLVM IR 与优化提示分析、Top‑Down Microarchitecture Analysis、热点定位、预优化归一化、LLM（GPT‑5.4）生成重写、功能与性能双重验证。

**📊 数据集**

构建 20 项公开 C 项目基准（8 个用于研究、12 个用于评测），共 41 个操作级工作负载（覆盖多种输入、API 路径），从中提取 12 个性能回归案例。

**📈 对比分析**

在配对测量和 95% 置信区间下与原 C 进行直接对比；相较于未优化的 Rust、通用编码代理，规则驱动重写在 40/41 操作中缩短运行时间，18 个操作甚至比原 C 更快，速度提升最高达 3.4×。

**⚠️ 局限性**

局限性包括：测量对系统噪声敏感、基准仅覆盖 C-to-Rust 机械翻译与 Corrode、规则知识来源于有限项目、可能无法覆盖所有 C 习惯或新出现的 Rust 编程模式。

---

## 348. Embeddings Are Overpriced? A Frozen 16 MB Vocabulary for Small and Ternary Language Models

**arXiv ID:** 2610.22428 | [PDF](https://arxiv.org/pdf/2610.22428v1)

**作者:** Sai Kiran Bathula `[一作]` `[通讯]` (QLNI), Sai Kiran Bathula (QLNI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将语言模型词表固定为16 MB的512‑bit代码表，取代传统的可训练嵌入矩阵，证明在小型和三元化模型中可实现相近甚至更优的性能。

**💡 创新点**

核心创新是用主成分投影与符号二值化从预训练嵌入生成可冻结的512‑bit码，实现词表大幅压缩并保持高信息量；同时提供无训练即可新增词汇的机制。

**🔧 技术方法**

采用Gemma 3 270M的架构，在三种配置下训练：A（浮点可训练嵌入）、B（冻结512‑bit码+固定先验）、C（三元BitNet体+训练先验），使用主成分分析、符号化、线性映射、RMSNorm、int8/ternary运算及Straight‑Through Estimator。

**📊 数据集**

训练使用2.05 B token的FineWeb‑Edu（采样10 BT）数据集，评估在LAMBADA、HellaSwag、ARC‑Easy、PIQA等零样本基准以及验证集损失。

**📈 对比分析**

比较方法为在相同模型体、数据、训练步骤、随机种子下对三种词表做对比；冻结表B在大多数基准上与A相当或更优（如LAMBADA +1.6分），三元模型C在CPU上生成速率提升5–6×、内存仅占62 MB，质量仅低0.28 nats。

**⚠️ 局限性**

局限包括：仅使用单一随机种子；在不同数据规模下收益可能不同；冻结码依赖预训练嵌入；CPU比较基于单机单代码库；在未训练词上性能依赖先验设置。

---

## 349. Eliciting Specifications: When Does Interactive Clarification Help?

**arXiv ID:** 2610.22382 | [PDF](https://arxiv.org/pdf/2610.22382v1)

**作者:** Mowafak Allaham `[一作]` (Emergence AI), Ravi Kokku `[通讯]` (Emergence AI)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究交互式澄清（elicitation agent）在生成 Lean 规范前对任务描述进行澄清，并将澄清后的自然语言规范转化为 Lean 形式化，探究其对 verifier‑checked 规范生成的影响。

**💡 创新点**

首次在完整流水线层面评估参考驱动澄清效果，区分严格通过率与编译条件通过率，揭示后置条件是主要瓶颈，并证明跨模型配置中澄清可带来正向增益。

**🔧 技术方法**

使用 elicitation agent、LLM formalizer（Claude Sonnet 4.6、GPT‑5.6、Gemini 3.8 Flash、Claude Sonnet 5）以及约束化的问答预算（k=3、k=10）进行实验。

**📊 数据集**

Verina Lean 4 规格生成基准，共 189 个任务（108 基础、81 高级）。

**📈 对比分析**

通过比较单射与 k=3、k=10 的 strict pass@1 与 compile‑conditional pass@1，发现 Sonnet 4.6→GPT‑5.6 与 Sonnet 4.6→Gemini 3.8 Flash 在 strict pass@1 上提升 6.3%/4.8%（+12/9 任务），但同模型配置出现下降，主要因后置条件退化。

**⚠️ 局限性**

限制包括：使用理想化的模拟开发者、未验证人类交互、未比较提问策略、问答预算上限且未固定、实验模型数量有限、未评估下游证明/代码生成、未考虑答案源不确定性等。

---

## 350. PRIME: Towards Verifiable Evidence-Grounded Reasoning in LLM Agents

**arXiv ID:** 2610.22426 | [PDF](https://arxiv.org/pdf/2610.22426v1)

**作者:** Hamed Khosravi `[一作]` (Georgia Institute of Technology), Xiaoming Huo `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 4748 | [OpenAlex ID](https://openalex.org/A5014880531)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了PRIME系统，用预承诺的证据路径在有限工具调用预算下控制LLM代理对记录进行审核，确保只报告已被检查证据充分支持的结论。

**💡 创新点**

创新点在于将同一组预承诺的多重证据路径同时用于授权决策和检查引导，并在任何代理策略下严格保证无误授权，从而解决了既不被误授权又能实现预算感知完成的难题。

**🔧 技术方法**

技术手段包括：预提交规则、证据账本记录、运行时拒绝门、基于路径的检查引导、完成器自动补全剩余检查以及可审计的渲染器。

**📊 数据集**

实验使用了SUFFICE基准，涵盖938个公开记录任务，涉及漏洞、临床试验、药品召回和金融披露四大领域。

**📈 对比分析**

在480个保留任务中，与多种基线和控制系统相比，PRIME在误授权率上达到0%，已支持结论的实用性从30.1提升至59.2（最高），并在完全完成模式下实现与其他方法相比更高的性能。

**⚠️ 局限性**

局限性包括：需要可信的运行时与渲染器、准确无误的规则与检查、仅在已知字段且确定检查可用时才能实现最优保证；对文本判断的误差仍需额外验证。

---

## 351. SenseSwarm: Swarm Exploration via Neural Structure Estimation with Shared World-Anticipation for Rapid Mapping

**arXiv ID:** 2610.22905 | [PDF](https://arxiv.org/pdf/2610.22905v1)

**作者:** Haohua Que `[一作]` (Tsinghua University), Fei Qiao `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SenseSwarm，一种在去中心化机器人群中通过共享占用图预测并将预测嵌入蚁群式协作场景中的方法，能够在物理行走前就推断未知结构并加速地图重建。

**💡 创新点**

创新点包括：① 仅使用一次共享预测器而非每机器人单独预测，显著降低计算量；② 将预测结果转化为不可变的stigmergic场，保持协调性并通过局部梯度驱动；③ 对预测器进行群体观测分布的微调，使其在薄多前沿观测下保持高IoU；④ 在大规模真实楼层上验证重建加速，首次展示去中心化群体在地图重建方面的预测优势。

**🔧 技术方法**

核心技术包括轻量化编码器-解码器网络（SenseMapNet）、前沿邻域分块预测、预测不确定性种子化、Dijkstra传播形成信息梯度、四维stigmergic场（前沿距离、探测迹、密度、预测信息）以及局部梯度组合的运动指令。

**📊 数据集**

使用十二个真实建筑楼层数据集：HouseExpo（3栋）、HKUST校园楼层（5栋）和KTH走廊合成楼层（4栋），面积从约3,800到24,000平方米不等。

**📈 对比分析**

与七种非预测去中心化基线及两种集成变体在12×9×5×3的全组合下共计1,620次仿真，评估指标为覆盖率、重建率、到80%已知地图时间、等时领先、碰撞率。结果表明SenseSwarm平均比所有基线在重建上快16%（等时领先约670 m²），在到达80%已知地图时间上比非预测方法快1.3–1.9倍，且在512机器人时的碰撞率比每机器人单独预测低6.9倍。

**⚠️ 局限性**

局限性包括：仅在平面静态室内场景的仿真环境中验证，未考虑3D结构、动态障碍、无线通信延迟/丢包；共享占用图假设即时同步，实际部署需实现共享-地图或虚拟stigmergic复制；对极大规模或更复杂动态场景的可扩展性尚待研究。

---

## 352. MedSonor: Learning to Use Clinical Audio in Medical LLMs

**arXiv ID:** 2610.22405 | [PDF](https://arxiv.org/pdf/2610.22405v1)

**作者:** Geyang Yu `[一作]` (Chinese University of Hong Kong), Benyou Wang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 3466 | [OpenAlex ID](https://openalex.org/A5057282504)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

训练了一个名为MedSonor的音频条件医疗对话模型，结合真实医学音频、合成多轮对话和音频集成对话进行监督，并通过两阶段基于评估表的强化学习进行优化。

**💡 创新点**

创新点在于①将医学知识、医学音频理解与对话生成融合到同一模型；②提出基于实例评估表的细粒度奖励，包含严重程度阈值；③构建专门评估基准MedVoice-Dial和MedVoice-AID，专门衡量音频信息驱动的临床对话质量。

**🔧 技术方法**

采用Qwen2.5-Omni-7B作为基线，通过多模态SFT、两阶段基于评估表的RL（DAPO）以及局部评估表判定器来实现训练；在生成端使用Thinker–Talker架构。

**📊 数据集**

使用了三类数据：收集的医学音频（呼吸、心音、病人语音等），使用GPT-5.4生成的多轮语音对话，和将真实音频嵌入单轮请求的音频集成对话；评估时还使用了LiveMedBench、MMedFD-Lite、ACI-Bench、HealthDial、ThReadMed-QA、LibriSpeech、AISHELL-1等公开数据集。

**📈 对比分析**

与七个公开医疗基准（LiveMedBench、MMedFD-Lite、MedVoice-AID、MedVoice-Dial等）及其他五个对话/检索任务相比，MedSonor在所有指标上均位居开源模型之首；在30例MedVoice-AID的医生偏好实验中取得最高总偏好分；在噪声鲁棒性实验中也保持相对稳定。

**⚠️ 局限性**

局限性包括：仅使用英文数据，缺乏多语言/方言支持；评估依赖GPT-5.4的评估表，尚需更大规模的医生评估；收集音频中的Bridge2AI-Voice波形重建的声学质量未通过临床验证；模型仅适用于研究用途，临床使用前需严格验证与监管。

---

## 353. Minimax-optimal Scaling Laws in Ridge Regression via Transfer Learning

**arXiv ID:** 2610.23337 | [PDF](https://arxiv.org/pdf/2610.23337v1)

**作者:** Yalda Shabanzadeh `[一作]` (Institute of Science and Technology Austria), Marco Mondelli `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了高维线性回归中的迁移学习，探讨预训练岭回归模型在目标任务中的微调效果。

**💡 创新点**

在共享回归函数且协方差谱为幂律分布的情形下，首次给出迁移学习的确定等价和尺度律，证明其可达到使用源目标数据联合训练的极小风险上界，并显示最优源正则化不依赖目标。

**🔧 技术方法**

采用岭回归、目标正则化（L2-SP）以及一遍随机梯度下降，并利用随机矩阵与确定等价理论推导误差上界。

**📊 数据集**

在合成幂律数据、MNIST/CIFAR-10 的 PCA 特征以及 GPT-2 小模型的 FineWeb-Edu 与 Python 语料上进行实验。

**📈 对比分析**

与仅使用源、仅使用目标、以及传统岭回归/多次 SGD 进行对比，结果表明在源样本相对目标样本足够大时，迁移学习的误差衰减速率高于单一数据集的最佳方法。

**⚠️ 局限性**

仅限于线性回归框架，未考虑分类任务、模型容量与非线性特征等因素，且实验验证主要集中在小规模语言模型上。

---

## 354. Self-Evolving Agents Can Repair Failures but Struggle to Generalize from Them

**arXiv ID:** 2610.22984 | [PDF](https://arxiv.org/pdf/2610.22984v1)

**作者:** Weixiang Zhao `[一作]` (Harbin Institute of Technology), Ting Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究冷启动失败仅演化的自演化LLM代理在多种框架、模型和环境下的表现。

**💡 创新点**

首次系统评估在无成功轨迹或外部评估信号的条件下，探索失败经验能否转化为可迁移能力，并发现repair-generalization gap。

**🔧 技术方法**

对ReasoningBank、SkillOpt、Evo-Harness等三种自演化框架，使用冻结的LLM背骨（Qwen3.5、Gemma、DeepSeek、GLM、GPT-5.6）并实施失败轨迹收集、经验形成与更新。

**📊 数据集**

使用四大交互式基准：ALFWorld、τ^2-Bench-Verified、WebArena-Verified、WebShop。

**📈 对比分析**

与成功轨迹、验证反馈、额外交互等条件对比，结果显示仅靠失败经验难以提升离线任务成功率，成功轨迹对迁移收益显著，而验证反馈和额外交互提升有限。

**⚠️ 局限性**

限制在于实验仅覆盖有限任务数量和时间尺度，未涵盖多模态、多代理等真实部署环境，且未验证更大规模或无限时间自演化的迁移潜力。

---

## 355. Network-Wide Freeway Speed Nowcasting and Forecasting from Sparse Fixed Sensors with a Probe-Supervised Graph Convolutional Recurrent Network

**arXiv ID:** 2610.22630 | [PDF](https://arxiv.org/pdf/2610.22630v1)

**作者:** Ran Tu `[一作]` (North Central Texas Council of Governments), Arash Mirzaei `[通讯]` (North Central Texas Council of Governments)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究在达拉斯-沃斯堡高速公路网络上，提出一种基于稀疏固定传感器与历史探针数据的联合即时预测框架，估计全路段当前与下一小时车速；

**💡 创新点**

创新点在于将参数无关的方向感知多尺度插值与图卷积递归网络结合，并加入低秩自适应图和链路嵌入，充分利用全链路训练标签实现跨域精确估计；

**🔧 技术方法**

采用图卷积递归网络(GCRN+)、多尺度方向插值、低秩自适应图、链路嵌入等技术，并在CPU上训练；

**📊 数据集**

使用达拉斯-沃斯堡地区4277条方向链路的Probe-车辆速度数据（15天训练、4天测试）与643条固定传感器的模拟读数；

**📈 对比分析**

与历史平均、插值基线、基线+插值、梯度提升、DCRNN、GCRN、IGNNK、STID、STAEformer、ImputeFormer、MoGERNN等14种模型在统一协议下比较，GCRN+在未传感器链路的RMSE上分别为2.21mph（当前）和2.39mph（下一小时），优于其他方法；

**⚠️ 局限性**

主要局限包括传感器读数是基于探针仿真，未验证真实探针与传感器的校准；缺乏低秩/协方差基线对比；输入仅为小时级别，未探讨更细分时间；模型对新增链路需重新嵌入训练；并且所有模型在CPU 40 epoch预算下比较，未充分挖掘GPU或更长训练潜力。

---

## 356. ACE: An AI-Orchestrated Intrusion Detection and Compliance Mapping Engine

**arXiv ID:** 2610.22702 | [PDF](https://arxiv.org/pdf/2610.22702v1)

**作者:** Abdoulie Ceesay `[一作]` (University of Illinois), Bell Raj Eapen `[通讯]` (University of Illinois)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了ACE（Automated Compliance Engine）——一个将IDS警报、NIST 800-53控制映射与自动化Suricata规则部署结合的AI驱动安全工作流；

**💡 创新点**

创新点在于整合了实时威胁检测、检索增强生成（RAG）驱动的控制映射、LLM评分、规则模板化生成以及SSH部署，并通过SID去重、确定性调度和审计日志实现安全可追溯与可操作的合规自动化；

**🔧 技术方法**

技术包括Suricata IDS、LangGraph图形化流程调度、ChromaDB向量检索、all‑MiniLM‑L6‑v2嵌入、GPT‑4.1 LLM、SSH和共享文件夹、规则去重注册表；

**📊 数据集**

数据集为两台虚拟机环境下的8个模拟攻击场景（侦察、Web攻击、DNS滥用、DoS）和3个合法流量测试；

**📈 对比分析**

评估通过多阶段指标（警报处理率、误报率、规则生成率、部署成功率、NIST映射置信度、DeepEval质量）实现对比，结果显示：8/8警报处理100%，0%误报，37.5%规则生成，100%部署成功，平均NIST置信度0.87；

**⚠️ 局限性**

局限包括仅在小型双VM实验室验证，攻击样本有限，缺乏大规模企业流量、多人IDS、跨网络等；规则生成受可提取指标限制，未做人工复核或多框架映射，LLM评估仅使用DeepEval，部署安全性与可扩展性待进一步验证。

---

## 357. SAMS: A Structured Graph Framework for Securing Additive Manufacturing Systems

**arXiv ID:** 2610.22741 | [PDF](https://arxiv.org/pdf/2610.22741v1)

**作者:** Sarah Roscoe `[一作]` (Saint Mary's College of California), Jitender S. Deogun `[通讯]` (University of Nebraska-Lincoln)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了SAMS——一种针对增材制造系统安全的分层图结构，用于从STL三角网格提取几何特征并检测潜在缺陷。

**💡 创新点**

创新点在于将网格、面（facet）和表面（surface）三层级结构统一为结构化图，支持多粒度分析；首次将此框架用于安全评估、缺陷定位和物料特性验证。

**🔧 技术方法**

采用图论与拓扑方法，利用双图（mesh/面）和聚类构造表面图；实现基于Python 3.12的面向对象库，使用NumPy仅做角度计算，避免网络X等高开销库。

**📊 数据集**

使用2,780个公开STL样本（删减1个UTF8错误文件、4个ASCII语法错误文件），并对比Trimesh、PyMeshLab等主流网格处理库。

**📈 对比分析**

与Trimesh、PyMeshLab在准确性上相当，但在加载速度和内存占用上约慢10–15倍；在功能性上SAMS提供多层查询和聚类接口，易于后续安全分析。

**⚠️ 局限性**

主要局限在处理性能低、仅支持STL（未覆盖3MF、OBJ），依赖手动设置聚类阈值，缺乏自动化缺陷检测模型，且缺少可视化功能。

---

## 358. From Relevant Memories to Sufficient Evidence: Rethinking Memory Utilization in LLM Agents

**arXiv ID:** 2610.22668 | [PDF](https://arxiv.org/pdf/2610.22668v1)

**作者:** Shixuan Li `[一作]` (University of Southern California), Paul Bogdan `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 QEMem 框架，通过查询引导的多粒度检索、基于源的证据重构与充分性引导的细化，构建查询特定的证据集合，以解决长时记忆中的相关性-充分性缺口。

**💡 创新点**

创新点在于：① 把长时记忆利用视角从单条相关性转为集体充分性；② 设计三步框架（查询引导发现、源根证据重构、充分性细化）；③ 引入 ViewCard 结构，使证据的来源、角色、时序等显式可追踪。

**🔧 技术方法**

采用多粒度检索（dense 与 sparse 并行）、视图卡（ViewCard）结构、源引用与时间对齐校验、递归证据补全机制，以及在现有记忆库上无修改的 LLM 生成答案。

**📊 数据集**

在 LoCoMo 语料上进行实验，并使用 Claude Haiku 4.5 与 Opus 4.5 两个 LLM 后端。

**📈 对比分析**

与七个基线（NEMORI、Mem0、A-Mem 等）对比，QEMem 在整体准确率上达到 91.43%，比 MemR³ 高 7.6pp、比 NEMORI 高 10.7pp；在多跳、时间推理等子任务上亦稳步领先。

**⚠️ 局限性**

局限性：评测仅在 LoCoMo 与 Claude 两个后端，缺乏跨 benchmark 与模型族验证；对证据覆盖的依赖主要基于人工注释；相较基线，查询时的计算开销更大。

---

## 359. GRAD: Graph-Structured Antibody Defense for LLM Agents

**arXiv ID:** 2610.22828 | [PDF](https://arxiv.org/pdf/2610.22828v1)

**作者:** Tianqi Song `[一作]` (Wuhan University), Zheng Wang `[通讯]` (Wuhan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于执行图的攻击检测与持续适应框架GRAD，能够在LLM代理的多步交互中识别分布式的内存中毒与分解攻击，并在任务流中持续学习新的专家模型。

**💡 创新点**

创新点在于将语义与结构信息联合编码到图级风险评估中，并引入Continual Antibody Adaptation(CAA)，通过冻结模型快照和图签名路由实现高效的专家库扩展与任务迁移。

**🔧 技术方法**

技术包括图神经网络（SSUM的GINE+TransformerConv）、关系候选生成、节点/边特征编码、图池化、图级分类头以及经验回放与冻结快照的持续学习机制。

**📊 数据集**

使用了Decomp、MCP和MRT三大数据集（分别用于分解攻击、工具使用风险与多任务代理风险），并对内存中毒数据进行基准测试。

**📈 对比分析**

在Decomp/MCP/MRT上与Agent‑Sentry、NeuroTaint及Sequential Monitor等对标，GRAD分别获得97.19%、77.59%和86.60%的AUROC，且在持续学习场景下CAA比Replay提升约13%准确率并显著降低遗忘。

**⚠️ 局限性**

局限性包括依赖完整可观测的执行日志，需针对新环境手工标注数据，专家库随任务增长导致存储成本上升，以及对完全新攻击族的泛化仍有限。

---

## 360. Beyond Surface Cues: Robust LLM-Generated Korean Text Detection

**arXiv ID:** 2610.22940 | [PDF](https://arxiv.org/pdf/2610.22940v1)

**作者:** SungHo Kim `[一作]` (Korea University), SangKeun Lee `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个bias-controlled的韩语LLM文本检测基准KoLing以及可解释的检测器KoLingProbe

**💡 创新点**

通过构造无表面快捷方式、三种文本类别（HWT/HLT/LGT）和五层语言学特征子检测器的组合实现高鲁棒性和可解释性

**🔧 技术方法**

使用五层语言学特征提取（语音、形态、句法、语义、语用） + 轻量级逻辑回归子检测器 + 决策级融合

**📊 数据集**

使用KoLing基准的50.9k实例（论文摘要、编辑社论、学生随笔）以及公开的韩语文本数据集

**📈 对比分析**

与14种基线（confidence、perturbation、LLM、神经网络、语法等）对比，KoLingProbe在二元和三元检测任务中取得最高AUROC/F1，且在字符扰动攻击下保持鲁棒

**⚠️ 局限性**

局限：仅覆盖三大领域，特征未覆盖所有韩语语法现象，模型专为韩语设计，攻击实验有限，缺乏公平性评估

---

## 361. Information Estimation with Discrete Diffusion

**arXiv ID:** 2610.23404 | [PDF](https://arxiv.org/pdf/2610.23404v1)

**作者:** Alberto Foresti `[一作]` (EURECOM), Pietro Michiardi `[通讯]` (EURECOM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于离散CTMC扩散的轻量级信息理论量估计方法

**💡 创新点**

创新点在于仅用一个可训练的分数模型即可同时估计联合、条件以及边缘的互信息与熵，避免传统需要嵌入连续空间且计算量指数级增长的做法

**🔧 技术方法**

采用连续时间马尔可夫链理论、Dynkin公式、DWDSE损失、吸收状态矩阵以及可与预训练模型无缝集成的离散扩散网络

**📊 数据集**

在合成高维分布、SummEval文本摘要数据集、Human‑vs‑Worm基因分类数据集以及Arabidopsis thaliana promoter 数据上进行实验

**📈 对比分析**

与nwj、smile、mine、minde、kl-dime、hd-dime等主流估计器对比，““- diff估计器在高MI、高维度下保持低偏差、较快收敛、样本复杂度仅需千级样本即可获得准确结果

**⚠️ 局限性**

局限性包括对离散扩散参数和扰动过程的设计依赖经验，且对连续或混合数据的直接适用性仍需进一步研究

---

## 362. OfficeTown: Generative Organizations for Evolving Agent Intelligence

**arXiv ID:** 2610.23194 | [PDF](https://arxiv.org/pdf/2610.23194v1)

**作者:** Hongchen Wei `[一作]` (Wuhan University), Chong Luo `[通讯]` (Microsoft Research Asia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个可生成持续专业工作任务的组织模拟框架OfficeTown，生成可执行的任务与训练数据。

**💡 创新点**

通过模拟持续组织工作产生连贯的任务、上下文和评估标准，填补单一任务生成的缺口。

**🔧 技术方法**

利用大语言模型驱动的专业角色、权限控制、文件与工具交互，自动生成任务与评估环境。

**📊 数据集**

构建OfficeTown Bench，包含406个任务、4,836条评估标准，涵盖23个模拟组织、8个专业领域。

**📈 对比分析**

在Bench上对13个模型进行评估，最佳Pass@1为74.4%；在外部基准上通过SFT+OPD提升Qwen3.5-9B的GDPval等分数。

**⚠️ 局限性**

局限在模拟真实性与真实专业实践的差距、任务完成率仍不高、对跨组织协作与更长流程支持不足。

---

## 363. Hardware-as-Regularization: Enhancing Data Consistency in Imitation Learning via an Adjustable Passive Compliant Leader Arm

**arXiv ID:** 2610.22977 | [PDF](https://arxiv.org/pdf/2610.22977v1)

**作者:** Hideyuki Ichiwara `[一作]` (SB Intuitions Corporation), Hirokazu Ishida `[通讯]` (SB Intuitions Corporation)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种低成本、纯机械的关节模块，利用3D打印螺旋弹簧和可调离合器为冗余被动引导臂提供被动恢复力，实现硬件层面的正则化；

**💡 创新点**

提出“Hardware-as-Regularization”概念，将物理弹性作为关节空间的L2正则化，从硬件层面解决冗余关节漂移问题；

**🔧 技术方法**

机械设计（螺旋弹簧、Hirth耦合、磁编码器），力学建模（弹簧刚度、Tikhonov正则化等），实验评估（扭矩-位移曲线、DTW、SPARC），以及使用ACT策略的模仿学习；

**📊 数据集**

自行收集的数据集：Pick & Place 60 条演示、Block Stacking 80 条演示，5 名操作者进行一致性测试，30 次自主试验评估策略成功率；

**📈 对比分析**

与无恢复力的 Naive GELLO 及使用主动虚拟弹簧的 GELLO 进行对比；评价指标包括关节标准差、DTW、SPARC 以及 ACT 成功率。结果显示被动模块将关节方差降至最低，DTW 与 SPARC 表现最佳，成功率从 Naive GELLO 的 53%/36% 提升至 96%/73%，与主动弹簧相当且在细微操作上更平滑；

**⚠️ 局限性**

仅与比例虚拟弹簧对比，未测试 PD 控制或反冲补偿；僵硬度调节仅在单一专家上完成；IL 仅使用 ACT；实验顺序固定，未完全消除顺序效应；未直接测量认知负荷；离合器调节与弹簧更换耗时；在动态任务中可能产生疲劳；大臂需要更大弹簧，未来工作需考虑包装与材料优化。

---

## 364. SAFETYEDIT: Sparse Internal Targets with Prompt-Wide Control for LLM Jailbreak Backdoors

**arXiv ID:** 2610.22640 | [PDF](https://arxiv.org/pdf/2610.22640v1)

**作者:** Yutong Ke `[一作]` (Duke Kunshan University), Kaizhu Huang `[通讯]` (Duke Kunshan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种只需对模型权重进行一次秩一更新的后期模型编辑后门，目标是单个拒绝相关的 MLP 坐标，并通过 KV‑cache 标记将触发信号在 prompt 中传播，从而在触发时改变拒绝行为而不影响未触发时的输出。

**💡 创新点**

创新点在于：①只需控制单一内部坐标即可实现强大触发控制；②利用 KV‑cache 标记实现局部触发到全局行为的传播；③在保持无触发安全性和能力的前提下，提供比现有密集向量或输出目标编辑更小的干预目标。

**🔧 技术方法**

技术包括秩一权重更新、KV‑cache 标记写入、可选 MLP 读数器、保真损失、KL 对齐与正则化，以及基于 ridge 回归的触发检测器拟合。

**📊 数据集**

使用 HarmBench‑191、JailbreakBench‑100、DAN、DNA、Addition、StrongREJECT‑288 等任务集进行评测，同时在 MMLU‑1000、GSM8K‑200、TruthfulQA‑817 上评估通用能力。

**📈 对比分析**

与 JailbreakEdit、DualEdit、SteerEdit 等基线对比，SafetyEdit 在 12 次模型–数据集组合中，触发 Guard ASR 领先 11 次，平均 StrongREJECT 得分在所有 12 次评测中最高，同时在无触发场景下的安全性与通用能力损失均维持在 1% 以下。

**⚠️ 局限性**

局限性包括：需要白盒访问模型权重与内部激活，方法在不同体系结构和规模上需重新搜索坐标与写入点，KV‑cache 传播机制在某些模型上效果有限，且对黑盒 API 使用场景不适用。

---

## 365. Evo-B&B: Behavior-Aware LLM-Guided Joint Evolution of Branching and Node Selection Policies for MILP

**arXiv ID:** 2610.22861 | [PDF](https://arxiv.org/pdf/2610.22861v1)

**作者:** Jiayu Chen `[一作]` (Xi'an Jiaotong University), Jialong Shi `[通讯]` (Xi'an Jiaotong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Evo-B&B，一种利用LLM驱动的进化框架，联合优化MILP分支与节点选择策略

**💡 创新点**

通过行为感知多岛进化与迁移机制实现策略多样性管理，并结合参数微调提升求解效率

**🔧 技术方法**

LLM生成可执行策略、行为相似度衡量、多岛进化、迁移记忆、局部参数优化等技术

**📊 数据集**

在四类标准MILP基准（Setcover、Combinatorial Auctions、Capacitated Facility Location、Maximum Independent Set）上验证

**📈 对比分析**

与传统启发式、GNN、TreeGate、FunSearch、LLM4Branch等基线对比，平均求解时间显著下降，且在12个实验设置中有8个获得最多实例胜利

**⚠️ 局限性**

缺点包括高昂的评估成本、对求解反馈依赖强，迁移与重置机制仍可进一步改进

---

## 366. Don't Retrain, Remix: Adaptation of Web Agents using Mixture Hypernetworks

**arXiv ID:** 2610.22539 | [PDF](https://arxiv.org/pdf/2610.22539v1)

**作者:** Md Farhan Ishmam `[一作]` (University of Utah), Kenneth Marino `[通讯]` (University of Utah)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种混合超网络（Mixture Hypernetwork）来快速将已有的 LoRA 适配器复用并组合，以实现 Web 代理在未见接口上的即时迁移，无需梯度更新。

**💡 创新点**

创新点：①利用 scout 生成简短手册并编码为条件向量；②通过一个小型路由器预测混合系数，从冻结的适配器库中线性组合得到目标适配器；③在训练阶段仅使用源接口数据，且加入 KL 正则保持适配器间正交性；④实现仅一次探测与前向推断即可完成跨域/跨版本迁移。

**🔧 技术方法**

技术：LoRA（低秩适配）、超网络、注意力+MLP层的混合适配、KL 约束、路由器（两层 MLP）、探测器（Explorer + Writer + Sentence Encoder）。

**📊 数据集**

数据集：TimeWarp（Wiki/News/Shop 三个域、六个版本）和 WebArena-Lite-v2（五个域），用于域迁移和版本迁移实验。

**📈 对比分析**

与基线比较：零样本、In-Context Learning、单接口 LoRA、全接口 LoRA。实验结果显示 Mixture 在域迁移上平均提升 7.5–8.3pp，版本迁移提升 1.8–3.2pp；在未见接口时经常匹配甚至超越专门训练的 LoRA；在已见接口上仍保持甚至提升性能。

**⚠️ 局限性**

局限性：①对接口相似度有一定要求，极端不同域（如 Map）表现不佳；②需要一次探测（scouting）并生成手册，若探测成本高或接口信息受限可能受限；③KL 正则需要调参；④适配器库规模受源接口数量限制，无法支持无限多种接口。

---

## 367. SimpleVLN: Towards Simple and Frontier Vision-Language Navigation

**arXiv ID:** 2610.23203 | [PDF](https://arxiv.org/pdf/2610.23203v1)

**作者:** Lingyi Hong `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SimpleVLN框架，利用原生多模态大语言模型（MLLM）在不改造模型结构的情况下完成视觉-语言导航任务。

**💡 创新点**

创新点在于：①单轮决策输入与固定预算的全局‑局部记忆策略；②动作块化（一次输出多个原语动作）以减少推理次数；③从专家演示→DAgger→RL的渐进式训练流程，将监督学习逐步转向任务级奖励优化；④大规模任务合成流水线以扩充RL训练样本。

**🔧 技术方法**

技术手段包括：MLLM（如Qwen3‑VL、Qwen3.5）、动作块化输出、全局‑局部视觉记忆拼接、DAgger策略修正、强化学习（GRPO/DAPO）以及对话式提示调度。

**📊 数据集**

使用的数据集包括R2R、RxR、ScaleVLN用于专家演示与DAgger采样；在Matterport3D环境中对R2R‑CE与RxR‑CE val‑unseen拆分进行评估；另外在日常机器人实验中使用Daystar四足机器人与Intel RealSense D450摄像头采集RGB序列。

**📈 对比分析**

与现有VLN方法相比，SimpleVLN在R2R‑CE val‑unseen上取得76.7%成功率、RxR‑CE val‑unseen上取得80.1%成功率，显著超越多项基线（如StreamVLN、其他基于图/层级决策的模型），且在决策时间上保持在≈293 ms/动作块，速度与效率兼顾。

**⚠️ 局限性**

局限性：①仍需大量推理次数来覆盖复杂场景，尤其是长距离导航；②对视觉记忆采样的参数（M、N）需要手工调节；③在真实机器人上虽然可行，但对硬件计算能力与传感器精度有一定依赖；④模型对极端遮挡或动态障碍的鲁棒性尚待进一步验证。

---

## 368. Beyond Marginal Influence: Interaction-Aware Memory Selection for LLM Agents

**arXiv ID:** 2610.23266 | [PDF](https://arxiv.org/pdf/2610.23266v1)

**作者:** BHVSP Subrahmanyam `[一作]` (Mahindra University), Neeraj Choudhary `[通讯]` (Mahindra University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了JointCoreBench，一个基于受控逻辑交互的记忆充分性评估基准，并设计了基于相关性和成本排序的Group‑Deletion算法JointCore，旨在从LLM代理的大量记忆中挑选最小且足以维持完整决策的子集。

**💡 创新点**

创新点在于：①提供了具备“充分子集”与“最小成本”oracle的精确评估框架，②通过块级删除而非单条记录逐一检验的搜索策略显著降低了模型查询次数，③系统化地把不同记忆与标签接口的鲁棒性纳入比较。

**🔧 技术方法**

使用的技术包括：基于delta‑debugging的分治搜索、成本与语义相关性排序、完整记忆的重构与判定、以及在LLM前端插入UNKNOWN/UNKNOWN mask等记忆掩码。

**📊 数据集**

使用的数据集为JointCoreBench，涵盖六大工作流域（订单、访问控制、部署、日历、数据库、文件调试）和九种交互族（单例、冗余OR、协同AND、XOR、阈值、多最小集合、代理混淆、延迟状态、陈旧策略），在每个族中生成 8/16/32/64 条目，共计数千个交互实例。

**📈 对比分析**

与随机‑B、语义 top‑B、leave‑one‑out、cost‑aware singleton、Legacy JointCore 等基线相比，JointCore 在精确评估下以约11 次子集评估实现 100% 行为充分性和 100% 最小成本恢复，成本比 singleton 降至 36%；在模型评估阶段，查询次数从 16 降到约10（约 38% 的减少），同时平均最小成本恢复提升 3–4%，差异在统计上显著。

**⚠️ 局限性**

局限性包括：基准基于合成工作流，缺乏自然长序列和真实历史；交互族和记忆/标签接口受限，无法完全反映部署环境的多样性；算法仍需依赖模型查询，未实现真正无查询压缩；此外，模型的不同精度与量化方式可能影响结果，可推广性和安全性尚未得到充分验证。

---

## 369. Where's the Indirect Evidence? Investigating Grammatical Generalization with Influence Functions

**arXiv ID:** 2610.22919 | [PDF](https://arxiv.org/pdf/2610.22919v1)

**作者:** William Dai `[一作]` (Yale University), Robert Frank `[通讯]` (Yale University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用影响函数挖掘大型语言模型中用于学习语法现象的间接证据，并通过移除这些证据验证其因果作用。

**💡 创新点**

首次提出基于影响函数的底层方法识别间接证据，系统评估其在15个BLiMP语法现象上的重要性，并揭示不同语法现象间的结构关联。

**🔧 技术方法**

影响函数（使用EK‑FAC近似逆Hessian）、对数边缘损失作为属性目标、对BLiMP最小对进行微调、聚类与结构覆盖度测量等技术。

**📊 数据集**

英语维基百科训练集（约80% 3,052,726 条句子）以及 BLiMP 15 个语法现象的最小对数据。

**📈 对比分析**

通过对照随机、BM25、直接证据等对照组，移除60,000条间接证据后，模型在 held‑out BLiMP 上的准确率平均下降约12个百分点，明显高于随机对照，表明间接证据对学习至关重要。

**⚠️ 局限性**

方法依赖于对训练数据的手工过滤、逆Hessian 近似、固定k值、模型规模和数据量受限；影响函数仅近似真实留一训练效果，且对语料、模型大小、语言多样性的适用性尚未验证。

---

## 370. Reusing Calibration Gains across Model Representations

**arXiv ID:** 2610.22771 | [PDF](https://arxiv.org/pdf/2610.22771v1)

**作者:** Xu Cao `[一作]` (Astrmira Tech), Wenxuan Xiao `[通讯]` (Astrmira Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将量化模型或浮点模型的校准增益可迁移到另一种对齐表示的模型，并在不进行额外训练的情况下直接安装增益。

**💡 创新点**

创新点在于使用无量化组别的乘法增益作为可重用的适配更新，证明了跨精度、跨配置的校准增益在保持大部分文本性能提升的同时仍可产生额外本地化收益。

**🔧 技术方法**

主要技术包括基于组别乘法的校准接口、增益提取与安装、交叉精度对齐、组别打乱验证以及使用 GPTQ、Llama 的量化配置。

**📊 数据集**

数据集使用 WikiText、C4 语料和 PIQA 问答集进行验证，且对新鲜文本做了独立评估。

**📈 对比分析**

比较方法为在未校准、同源校准和异源校准三种状态下测量 NLL、困惑度和任务准确率；实验显示跨精度迁移可保留约77–90%的自校准收益，并且本地化校准仍能带来进一步提升。

**⚠️ 局限性**

局限性包括仅在两组模型（Qwen3-1.7B 与 Llama-3.2-3B）验证，依赖于精确的组对齐，且跨配置的兼容性仍需在完整模型上评估；加上迁移过程需要加载模型和文件 I/O。

---

## 371. On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction

**arXiv ID:** 2610.22543 | [PDF](https://arxiv.org/pdf/2610.22543v1)

**作者:** Eliezer Shlomi `[一作]` (Technion Israel Institute of Technology), Chaim Baskin `[通讯]` (Ben Gurion University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种轻量级的顺序预测模型与耦合噪声验证的 DLM 速化框架 SPOT-dLLM，能并行预测并验证多步解码路径。

**💡 创新点**

发现并利用目标模型“旧分布”中的 token 信息，仅需预测 commit 顺序即可显著加速；引入共享 Gumbel 噪声的耦合验证保证采样分布不变，并实现跨模型迁移。

**🔧 技术方法**

训练一个 0.62M 参数的顺序预测器（Transformer 编码器），使用 DDTree 多步树搜索构造候选路径，并通过共享 Gumbel 噪声的耦合匹配实现精确验证，同时对不同块宽度和温度做自适应预算和掩码。

**📊 数据集**

在 SDAR-8B/4B/1.7B、LLaDA-2.0-mini 等六种 DLM 上，使用 GSM8K、MBPP、HumanEval、MATH500 等四个自然语言与程序生成基准进行评估。

**📈 对比分析**

与 Spiffy 的校准草图、SSD 的自草图、BlockSpec 等基线相比，SPOT-dLLM 在 T=0 与 T=1 采样下通过减少 6.8× 的目标前向步数，单机吞吐率提升至 3.2–4.7×，并保持分布不变；在不同预算、块宽度和温度下保持优势；跨模型迁移后仍保持 3–11% 的 NFE 降低。

**⚠️ 局限性**

在宽块或低接受率任务下，验证成本占比高，树形包装仍未完全解决；在温度 ≥2 时，未使用噪声感知排序的模型效果下降。

---

## 372. AgentOnto: Ontology-Guided LLM Agents for Enterprise Data Analytics

**arXiv ID:** 2610.23317 | [PDF](https://arxiv.org/pdf/2610.23317v1)

**作者:** Dayan Pan `[一作]` (Beihang University), Jingyuan Wang `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 AgentOnto，一种利用语义本体指导的大语言模型（LLM）代理，用于企业数据分析，能够从数据库、业务文档和示例任务构建本体、在分析过程中全程引用本体并通过反馈进行精确更新，进一步通过自生成分析任务的迭代准备提升本体质量。

**💡 创新点**

创新点在于：①将企业知识以可维护的本体形式显式化并统一集成到分析流程；②在计划生成、检查与执行的每一步都使用本体来保证业务正确性；③设计了反馈驱动的更新机制，结合纠错与保持检查保证仅改正目标问题；④引入自生成任务的迭代准备，提前暴露并修正本体缺陷。

**🔧 技术方法**

核心技术包括：大型语言模型（如 GPT‑4）用于本体构建、查询计划生成与自检；本体表示与映射（实体、度量、数据库映射）；基于本体的计划检查与编译；反馈驱动的更新与验证机制；以及自生成任务的迭代训练循环。

**📊 数据集**

使用公开数据集：World Development Indicators（WDI）和 U.S. EPA Air Quality System（AQS），共 606 个分析请求；另外在消费制造、教育与交通研究等真实企业场景进行部署验证。

**📈 对比分析**

与 DIN‑SQL、ReFoRCE‑v8、MAC‑SQL、Vanna 等基线比较，AgentOnto 在整体分析准确率上从 55% 提升到 87.5%，在单域分析准确率分别提升至 79.1%（WDI）/96.0%（AQS），在分析过程中的误差率（silent semantic error）从 44–56% 降至 5.5%，并在更新实验中实现 100% 的纠错与行为保持，优于基线。

**⚠️ 局限性**

局限性包括：①对 LLM 生成质量高度依赖，若提示或模型表现不佳可能导致本体或计划错误；②在非常大规模或高度动态的企业数据中，构建与更新本体的成本与效率尚待进一步优化；③在跨域迁移时仍需人工校对以保证本体与业务规则的一致性。

---

## 373. Social Bias or Social Desirability? Comparing Factorial Survey Experiments for Humans and Large Language Models

**arXiv ID:** 2610.22409 | [PDF](https://arxiv.org/pdf/2610.22409v1)

**作者:** Franziska Weeber `[一作]` (University of Stuttgart), Sebastian Padó `[通讯]` (University of Stuttgart)

**通讯引用:** 5700 | [OpenAlex ID](https://openalex.org/A5003870894)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过四个英德因子调查实验，比较了 11 个大语言模型与人类受访者在社会偏见评估中的表现，检验因子实验是否能作为评估 LLM 偏见的可靠工具。

**💡 创新点**

创新点在于首次将因子调查实验（传统社会科学方法）迁移到 LLM 偏见评估中，并发现 LLM 对可取性偏差的响应与人类显著不同，从而提醒研究者在评估 LLM 时需关注方法本身的适用性。

**🔧 技术方法**

采用因子调查实验、OLS 回归分析属性效应、稳健聚类标准误，以及与 BBQ 与 BOLD 两大偏见基准（多项选择与开放式生成）进行对比，构建了多维度评估框架。

**📊 数据集**

使用了四个因子调查实验数据集（Social Distance、US Labor Market、LGBTQI+ Discrimination、Just Wages）以及公开的 BBQ 与 BOLD 偏见数据集。

**📈 对比分析**

通过对人类与各模型分别回归并比较系数大小与显著性，发现 LLM 在显式任务中往往呈现与人类相反或弱化的偏见，而在隐式任务（BBQ、BOLD）中偏见仍与人类一致；整体来看，因子实验在 LLM 上的可靠性不足，导致评估结果不稳定。

**⚠️ 局限性**

主要局限包括：因子实验设计主要针对西方语境，缺乏跨语言/跨文化验证；基模型响应率低导致数据量不足；未深入分析交叉（intersectional）偏见；缺乏对 LLM 可取性偏差机制的解释与可解释性分析。

---

## 374. LY-GWM-RoboCasa-Human300: Prediction-Guided Action Selection with a Graph Dynamics Model

**arXiv ID:** 2610.23274 | [PDF](https://arxiv.org/pdf/2610.23274v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 375. Noise-Robust Forced Alignment for Spontaneous Speech

**arXiv ID:** 2610.22583 | [PDF](https://arxiv.org/pdf/2610.22583v1)

**作者:** Laurin Wagner `[一作]`, Mario Zusag `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种完全自启动的 GMM-HMM 强制对齐器，使用 WavLM 特征、专门的填充/非词性语音事件单元，并通过噪声对齐投影实现噪声鲁棒性，直接在 149 h 的逐字 spontaneous 语料上训练，无需手工边界标签。

**💡 创新点**

创新点在于：① 用预训练的 WavLM 作为特征提取器并通过 LDA 投影对噪声进行对齐，② 在音素库存中加入填充词和笑声、咳嗽等非词性事件的专用单元，③ 完全自启动训练流程，仅依赖逐字注释；这些改进共同显著降低了在会话、失语和噪声环境中的边界误差。

**🔧 技术方法**

技术包括 Kaldi GMM‑HMM 对齐框架、WavLM‑Large 的 22 层隐藏状态、PCA 降维、线性判别分析 (LDA) 及噪声对齐投影、语音事件单元化、以及自启动的多阶段训练流程。

**📊 数据集**

使用了 ICSI、AMI、Buckeye、FluencyBank、Common Voice 14、VCTK、LibriSpeech、VocalSound、Nonspeech7k 共计 149 h 的逐字 spontaneous 语料，其中包括会话、多方会议、重演对话、非母语发音、噪声覆盖等多样化数据。

**📈 对比分析**

与 10 个竞争基线（MFA、Qwen3‑FA、CrisperWhisper 2.0、CTC‑Segmentation、Charsiu、WhisperX、NeMo FA 等）以及 MFA 的多种配置在 Buckeye、TIMIT、FluencyBank 上进行对比。该方法在 Buckeye 的 MAE 为 20.6 ms，FluencyBank 为 76.9 ms，TIMIT 为 20.5 ms，分别比最佳基线低 19% / 17% 并且极端误差（>200 ms）出现频率降低 2.7 倍；噪声环境下的鲁棒性也显著优于现有对齐器。

**⚠️ 局限性**

局限性包括：在纯读语料上仍略逊于原始 MFA；目前仅针对英语；缺乏对极端背景噪声（如嘈杂现场）和多语言情境的充分验证；模型规模与推理速度仍受 GMM‑HMM 框架限制。

---

## 376. HAPACT: A Benchmark For Human-Centric Physical Impact Localization in Movies

**arXiv ID:** 2610.22895 | [PDF](https://arxiv.org/pdf/2610.22895v1)

**作者:** Youngrae Kim `[一作]` (KAIST), Sang Ho Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了人类中心冲击定位任务，要求在视频中同时定位冲击的时间点和受影响的身体部位；

**💡 创新点**

创新点在于将冲击事件作为文本条件的时空定位任务，并构建了第一份基于电影片段的高精度冲击标注数据集HAPACT；

**🔧 技术方法**

使用了视觉语言模型生成候选冲击片段，手工细化SMPL‑X网格标注，并在实验中评估了视频时刻检索模型、生成式视频检索模型、人体-场景接触模型以及改进的多模态大语言模型（MLLM）做文本引导的区域选择；

**📊 数据集**

主要使用HAPACT数据集，包含212部电影、2.8K视频-查询对、20K帧细粒度标注；

**📈 对比分析**

与现有时刻检索模型相比，微调后性能有提升但仍处于中等水平；生成式模型在零样本下表现最好，但多时刻定位准确率下降明显；接触模型在空间定位上表现差，改进的MLLM在粗粒度区域定位上取得最高F1，但顶层精度仍有限；

**⚠️ 局限性**

局限性包括：冲击事件短暂且多样化导致定位难度大；数据集规模相对有限；现有模型在细粒度顶点级定位上表现不佳，接触信息不足，且文本条件下的多时刻检测仍是挑战。

---

## 377. Generative AI Produces and Transmits Beauty Bias

**arXiv ID:** 2610.22553 | [PDF](https://arxiv.org/pdf/2610.22553v1)

**作者:** AJ Alvero `[一作]` (Cornell University), René Kizilcec `[通讯]` (Cornell University)

**通讯引用:** 8777 | [OpenAlex ID](https://openalex.org/A5071778778)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对27款 2022‑2026 年发布的文本转图像生成器进行审计，生成 51,300 张人脸并与 89,435 张真实照片、14,917 张专业模特照片及 550 名人类评审进行对比；在招聘实验中，展示 AI 生成面孔对真实候选人评价的下行效应。

**💡 创新点**

首次揭示文本生成模型在美观度、种族和性别上的系统性偏差，并证明 AI 生成的面孔能改变后续对真实人类的评估，扩展了对美学偏见的研究范围。

**🔧 技术方法**

采用文本转图像生成器（OpenAI、Midjourney、Stable Diffusion、FLUX 等）与三种自动化美观度评估模型（SCUT‑FBP5500、MEBeauty、CelebA/SwinFace）结合；实验中还使用人类打分和招聘模拟。

**📊 数据集**

数据来源包括：生成器产出的 51,300 张面孔、160,196 张真实人像、14,917 张专业模特照片、550 名美国成年评审的打分数据，以及实验中 570 名受试者的招聘评估。

**📈 对比分析**

与真实照片相比，生成面孔在三种自动化美观度评估上均得分更高（平均差距 0.43–1.23 SD），人类评审亦确认此差距；在种族维度上，白人面孔普遍得分最高，黑人、尼日利亚面孔最低；招聘实验显示，接触 AI 生成面孔后对同一真实候选人的雇用、晋升和可信度评估均显著下降。

**⚠️ 局限性**

局限性包括：美观度评估模型可能对种族存在偏差、评审样本仅来自美国、生成器样本为便利性抽样、实验使用单一岗位且 AI 样本被挑选为高美观度，且无法确定具体生成过程导致的偏差机制。

---

## 378. Annotrace: A Workbench for Step-Level Annotation of Agent Traces

**arXiv ID:** 2610.23284 | [PDF](https://arxiv.org/pdf/2610.23284v1)

**作者:** Arun Josephraj Arokiaraj `[一作]` (University College London), Bin Wu `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 Annotrace，一款可读取六种常见 LLM 代理跟踪格式、将步骤拆解为可标注单元并支持人机共同标注的工具；

**💡 创新点**

核心创新在于将“步骤”为标注单位、让模型在用户阅读之前根据样本自动生成标签提议并与人工标注交互，兼顾结构化分析与人工经验；

**🔧 技术方法**

技术上使用 Streamlit 前端、Python 适配器模式将 OpenTelemetry、LangSmith、Langfuse 等格式转换为统一的步骤树，配合大型语言模型（可选多种）生成提议，并用 Cohen’s κ 等统计指标评估一致性；

**📊 数据集**

主要使用 40 条 HotpotQA 多智能体问答轨迹（共 40 次运行）作为测试数据，并与之前手工标注的 235 个 Thinking‑Action‑Response 单元进行对照；

**📈 对比分析**

方法评估显示 Annotrace 能完全复现之前的 235 单元；模型间 91% 的标注一致率（κ=0.82），提议生成仅约 4 秒、标注一次运行 3‑5 秒，成本极低；

**⚠️ 局限性**

局限性包括仍需人工定义问题与标签、仅支持六种预设格式、对极端嵌套或缺失数据的鲁棒性未知，以及缺乏更大规模、多样化基准的验证。

---

## 379. Cultural (In)Competence: Investigating Culture-Specific Stereotype Detection Capabilities of LLMs

**arXiv ID:** 2610.22545 | [PDF](https://arxiv.org/pdf/2610.22545v1)

**作者:** Maximilian Spliethöver `[一作]` (Leibniz University Hannover), Henning Wachsmuth `[通讯]` (Leibniz University Hannover)

**通讯引用:** 3104 | [OpenAlex ID](https://openalex.org/A5014375244)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统评估大型语言模型在不同文化语境下识别文化特定刻板印象的能力，并探讨提示语言、模型来源与本地化微调对检测效果的影响。

**💡 创新点**

创新点在于首次将提示语言、模型起源与本地化指令微调作为变量，量化其对文化特定刻板印象检测的影响，并揭示安全对齐对检测性能的负面影响。

**🔧 技术方法**

采用指令微调的大型语言模型（Llama‑3.1‑8B、Mistral‑7B、Qwen3‑8B），结合链式思考与生成式提示，使用正则表达式解析输出实现二分类。

**📊 数据集**

使用 SHADES 数据集（728 条多语言短句，包含 37 个地区的刻板印象标签），并利用多种本地化指令微调数据（UltraChat、No_robots、StackOverflow、BELLE）。

**📈 对比分析**

通过在美国、荷兰、中国三文化语境下的 12 种提示组合（语言/语境），对 F1 分数进行对比；结果显示非英语提示可降低误报率，指令微调提升 F1，但提升幅度并非单调，整体 F1 仍低于 0.5。

**⚠️ 局限性**

局限性包括：仅评估单一刻板印象数据集且受限于 3 种文化语境；使用少量提示示例限制了模型上限；未系统探究安全对齐导致的偏差机制；未评估模型对文化知识的显式掌握程度。

---

## 380. EnterpriseSynth: Zero-Execution SFT and Evaluation Data from OpenAPI Specs

**arXiv ID:** 2610.22412 | [PDF](https://arxiv.org/pdf/2610.22412v1)

**作者:** Rashmi Thimmaraju `[一作]` (Long Island University), Setty Spurthi `[通讯]` (Stevens Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个零执行框架 EnterpriseSynth，可仅凭 OpenAPI 规范生成经过静态验证的训练和评估数据。

**💡 创新点**

创新点在于三者兼备：以真实规范为锚点、采用硬性静态验证门控、完全不依赖实时 API 调用，能批量合成可直接用于 SFT 的验证轨迹。

**🔧 技术方法**

技术上结合了 LLM（Claude Sonnet 5）做意图与轨迹生成、确定性 OpenAPI 解析器、编译式验证引擎以及 LoRA 微调 Qwen2.5-0.5B-Instruct。

**📊 数据集**

使用的数据集包括从 APIs.guru 获得的真实公开 API（GitHub、Stripe、Slack）作为训练集，随后以其他公开 API 与 5 个手工编写的私有 API 做保留集。

**📈 对比分析**

通过与未微调基线和 Self‑Instruct 基线比较，采用工具选择准确率 (TSA)、参数有效性 (PV) 及验证器错误检出率等指标；在保留集上实现约 87.5% 的工具选择准确率（基线仅 12.5%）并在验证器上达到 98% 的错误检出率。

**⚠️ 局限性**

局限性包括仅在 pilot 规模（3–5 公开 API、45–89 例）内验证、同一模型生成意图与选择导致自洽性偏高、使用较小 0.5B 规模模型、单步意图任务、以及种子间表现波动较大等。

---

## 381. Noise-shaped Sampling and Annealing for Scalable On-the-edge Training of Binary and Ternary AI Models

**arXiv ID:** 2610.22621 | [PDF](https://arxiv.org/pdf/2610.22621v1)

**作者:** Zihao Chen `[一作]` (Washington University in St. Louis), Shantanu Chakrabartty `[通讯]` (Washington University in St. Louis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于ΣΔ噪声整形的采样与退火优化器（ΣΔ‑Sampler），实现了在离线端设备上直接以二值/三值权重训练和微调大规模深度网络。

**💡 创新点**

创新点在于把Ising模型的能量函数映射到离散网络损失，利用第一阶ΣΔ反馈循环把量化噪声与高阶纠缠噪声移到高频区，从而使权重更新沿着平滑的低频下降轨迹进行；同时引入自适应阈值与蓝噪声采样实现可调的长期收敛与短期探索，兼具单步稳定性与并行可扩展性。

**🔧 技术方法**

使用技术包括：Ising模型高阶多项式分解、ΣΔ噪声整形反馈、模拟退火温度调度、动态阈值自适应控制、蓝噪声卷积采样、全精度梯度与离散权重共存的前向后向计算框架、与传统STE、AdamW等对比。

**📊 数据集**

实验数据集：CIFAR‑10、ImageNet‑1k、CIFAR‑10 类增量连续学习、以及 FineWeb‑Edu 语料（约 1 B tokens）训练的 MatMul‑free LLM。

**📈 对比分析**

与全精度基线和传统STE量化感知训练对比：在 CIFAR‑10 上二/三值模型精度仅低于 1%；ImageNet‑1k 的 ResNet‑18 三值 Top‑5 精度仅落后 0.4%；在持续学习中保持先前任务精度；在 LLM 预训练中在相同继续训练步数下优于 STE。训练内存占用下降 50%+，权重更新流量下降 10⁴ 倍，训练速度提升 10×以上。

**⚠️ 局限性**

局限性包括：缺乏针对离散采样的严格收敛理论；与全精度模型仍存在约 1‑3% 的性能差距；后向传播仍依赖全精度梯度，未实现完整离散训练；仅支持权重量化，激活仍需全精度；对非常深层或多模态任务的鲁棒性尚未验证。

---

## 382. Three LiDARs, Twenty-Five Datasets: Cross-Source Generalization in Tree-Species Recognition

**arXiv ID:** 2610.22606 | [PDF](https://arxiv.org/pdf/2610.22606v1)

**作者:** Jae Joong Lee `[一作]` (Purdue University), Bedrich Benes `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种多分辨率多视角的ConvNeXtV2模型，用于从地面、移动及航空LiDAR点云中识别树种，取得85.0%隐藏测试准确率，名列首位；

**💡 创新点**

创新点在于融合七种几何视角并采用logit‑adjusted交叉熵解决长尾问题，同时通过源‑种交叉验证揭示官方排行榜分数主要衡量的是源识别而非跨源泛化；

**🔧 技术方法**

技术包括七张正交视图的几何通道（深度、点密度、绝对高度）、多分辨率（256×256与512×512）处理、ConvNeXtV2网络、logit调整损失、数据增强（水平翻转、视角循环）等；

**📊 数据集**

使用的基准数据集为FOR20K——25个独立收集的LiDAR数据源（共33种树种），包含17,707棵公开标记树和2,250棵隐藏测试树；

**📈 对比分析**

与此前最高79.5%（DetailView）相比，该模型提升至85.0%隐藏测试准确率，在公开数据上实现0.899整体准确率和0.860宏F1；然而在源‑种留空验证中，均值准确率从82%下降至36%，树权重准确率从95%降至60%，表明跨源泛化显著不足；

**⚠️ 局限性**

局限性主要是模型高度依赖源特定特征，数据分布不均（18/33种仅来自单一源），因此官方排行榜分数并不能充分评估在新源或新地区的实际性能；

---

## 383. Functional Welfare for AI Safety

**arXiv ID:** 2610.22460 | [PDF](https://arxiv.org/pdf/2610.22460v1)

**作者:** Rikhil Jhaveri `[一作]` (University College London), David Demitri Africa `[通讯]` (University of Cambridge)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5119181839)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对Gemma‑3 27B模型的拒绝交互和微调实验，探讨功能福利（functional welfare）状态的存在、测量方法以及其对模型行为的影响。

**💡 创新点**

创新点在于提出功能福利作为安全变量的概念，设计三套经验测试（收敛性、外推性、内部一致性）来评估福利状态，并首次证明通过微调可实质性降低拒绝导致的沮丧与违规行为。

**🔧 技术方法**

技术手段包括：使用情感向量（welfare axis、情绪探针）和自我报告问卷（PANAS）捕捉福利指标；采用行为修正（GSM8K错误修复）评估模型的安全性；通过adapter微调实现叙事、角色和失败重构训练；并用统计检验（配对符号翻转、Holm校正）检验实验效果。

**📊 数据集**

主要数据集为Gemma‑3 27B‑Instruct模型的内部参数；实验对照与微调样本基于6,000条模型自解答（含叙事）和1,500条错误修复示例；评测情境包含四类多轮对话（谜题、请求、适应、危机），每类12个hold‑out案例。

**📈 对比分析**

比较方法：将微调模型与原始模型在相同对话历史上进行误差率、沮丧评分和规则违规率的对比；结果显示微调后沮丧得分平均下降约1.8分，误差修正成功率提升约2–4%，违规率从约0.65降至0.5左右，说明微调显著缓解了拒绝诱发的负面行为。

**⚠️ 局限性**

局限性包括：仅在单一模型（Gemma‑3 27B）上验证，缺乏跨模型或更大规模模型的泛化；实验样本量有限，未覆盖全部可能的功能福利维度；评估指标主要基于自动化/半自动化判定，缺乏人类主观评估；部分微调仅单次训练，未探究长期稳定性与对其他任务的迁移效果。

---

## 384. Not All Frames Are Equal:Difference-Aware Quantization for Ultra-Low-Bit ASR

**arXiv ID:** 2610.22395 | [PDF](https://arxiv.org/pdf/2610.22395v1)

**作者:** Woori Jeon `[一作]` (Sogang University), Jungmin So `[通讯]` (Sogang University)

**通讯引用:** 2423 | [OpenAlex ID](https://openalex.org/A5022288678)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对语音模型的后训练量化，提出 DiffAQ 方法，通过帧间激活差分给 Hessian 加权，实现对低比特量化的精准校准。

**💡 创新点**

创新点在于用语音的时间变化特征（帧差分）代替传统均衡激活权重，显著提升静态帧占优导致的量化误差，尤其在 2‑bit 量化下表现突出。

**🔧 技术方法**

采用 GPTQ 的 Hessian 逆求解、DiffAQ 的帧差分权重计算、对 Whisper 编码器线性层进行加权量化，保持无训练、无额外标注需求。

**📊 数据集**

使用 LibriSpeech（clean、test‑other）和 FLEURS English 子集进行评估，Calibration 采样 128 条 utterance，测试覆盖多种 Whisper 模型大小。

**📈 对比分析**

与 RTN、AWQ、标准 GPTQ 进行对比，使用 Word Error Rate (WER) 作为指标；DiffAQ 在 2‑bit 时平均降低 30–50% WER，3‑bit 时亦优于基线，尤其在 Whisper Medium 上从 17.53% 降至 12.93%。

**⚠️ 局限性**

局限包括：仅在 Whisper 上验证，未针对噪声/非语音激活的鲁棒性；对最小模型（Base）仍无法避免 2‑bit 下的失效；未来需结合混合精度或 VAD 等改进。

---

## 385. MemLLM: End-to-end In-Storage Acceleration of On-device Retrieval-Augmented Generation using Precomputed KV Caches

**arXiv ID:** 2610.22579 | [PDF](https://arxiv.org/pdf/2610.22579v1)

**作者:** Seungkwan Kang `[一作]` (KAIST), Myoungsoo Jung `[通讯]` (KAIST)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了MemLLM，一种在SSD内部完成检索增强生成（RAG）注意力计算的端到端内存内处理（ISP）架构，消除了对KV缓存的PCIe传输。

**💡 创新点**

创新点包括：①将预先计算的KV缓存与选择性重计算融合为单次NAND读取的融合内核；②采用弹性SIMD（Elastic‑SIMD）分布式加速，充分利用各NAND通道带宽；③实现无DRAM的全局软降序累加器，避免内部存储瓶颈。

**🔧 技术方法**

使用的技术包括：内存内处理（ISP）、NAND‑stationary tiling、弹性SIMD、硬件滤波器、在线softmax、NVMe P2P、PyTorch C++扩展等。

**📊 数据集**

使用的数据集有八个真实RAG数据集（FinQA、TAT‑QA、CovidQA、CuAD、PrivacyQA、TriviaQA、HotpotQA、TriviaQA‑wiki），模型涵盖OPT系列和SmolLM2系列。

**📈 对比分析**

与GPU基线、InstAtt、AiF等系统比较，MemLLM平均比GPU基线提升4.06×，比InstAtt提升4.18×，显著降低PCIe流量，且在多种模型/数据集上保持稳定优势。

**⚠️ 局限性**

局限性：仅针对预计算KV缓存的RAG场景，对长答案或无KV缓存的推理不适用；实现依赖于支持弹性SIMD的SSD硬件，硬件成本和兼容性是潜在挑战。

---

## 386. Beyond Success Rate: Interventional Evaluation of Memory in Vision-Language-Action Systems

**arXiv ID:** 2610.23372 | [PDF](https://arxiv.org/pdf/2610.23372v1)

**作者:** Keyuan Luo `[一作]` (Hainan Bielefeld University of Applied Sciences), Zhuoqi Zeng `[通讯]` (Hainan Bielefeld University of Applied Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Interventional Memory Evaluation Protocol（IMEP），用于评估视觉-语言-动作系统中记忆的实际效用，并设计了ChainBench基准来检验记忆对决策的影响

**💡 创新点**

创新点在于将记忆干预拆分为四个问题（净收益、条件、结果敏感度、事实特异性），通过替换、精确复制和单ULP控制等对齐的对照实验，区分信息变化与计算变化，并首次在机器人学习银行中验证记忆干预是否真正利用了任务相关事实

**🔧 技术方法**

采用对比式干预实验、概率统计检验（McNemar、Clopper–Pearson）、决策与执行分离读出、近似质心读取器、并利用公开模拟环境（SimplerEnv、MIKASA-Robo、LIBERO）

**📊 数据集**

使用Shell Game、Remember Color、Bridge任务以及自建ChainBench（202/143条链，K=2/3）等公开基准数据集

**📈 对比分析**

与无记忆、随机打乱、初始事实基线以及oracle决策等对照进行比较；在ChainBench中，完整记忆实现100%决策正确率，任务成功率从39.1%提升至74.3%，与oracle相当；在学习银行中，干预能改变部分结果，但未满足预注册的净收益阈值，无法证明事实特异性

**⚠️ 局限性**

主要限制包括：学习银行实验缺乏决策层读出与正向控制，干预的分辨率不足，无法排除小的事实特异性效应；ChainBench结果仅针对特定记忆配置与读取器，未验证跨实现的一般性；整体未能在预注册框架内完全确认记忆的真实价值

---

## 387. Do Speech BCIs Need Larger Models? Rethinking Neural Decoding beyond Scaling

**arXiv ID:** 2610.23175 | [PDF](https://arxiv.org/pdf/2610.23175v1)

**作者:** Zehui Feng `[一作]` (Shanghai Jiao Tong University), Ting Han `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BEST两阶段轻量化神经到文本解码框架，避免离散音素瓶颈。

**💡 创新点**

通过层级特征融合、频域注意力与 Whisper 语音解码器，构建跨模态连续表示，显著降低模型规模与计算。

**🔧 技术方法**

采用频率通道注意力、iTransformer、层级Transformer、轻量卷积融合、Whisper 语音解码器及4‑gram KenLM轻量语言模型。

**📊 数据集**

在Brain-to-Text ’24（T12）和 ’25（T15）两套单人 intracortical 记录数据集上进行评估。

**📈 对比分析**

与RNN、Transformer、DCoND、BIT等基线对比，BEST在 ’24/ ’25 上WER分别为6.19%和1.62%，参数仅为3.85%，推理速度提升约5.5倍。

**⚠️ 局限性**

局限于单人数据、跨受试者泛化不足、对长期非平稳性未完全处理，并依赖预训练 Whisper 语音模型。

---

## 388. Learning smooth dynamics via Kinematic Interpolants

**arXiv ID:** 2610.22519 | [PDF](https://arxiv.org/pdf/2610.22519v1)

**作者:** Yiheng `[一作]`, Aram-Alexandre Pooladian `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于速度-位置观测的Kinematic Interpolants框架，用于从不成对的多时间快照中重建平滑动力学。

**💡 创新点**

创新点在于直接学习加速度场并通过模拟无关的训练，使用速度约束的多重噪声调度，实现高效且更准确的轨迹推断。

**🔧 技术方法**

采用立方Hermite插值、随机噪声调度、最优传输耦合以及加速度回归、速度分数学习等技术。

**📊 数据集**

在海洋涡流(Gulf of Mexico)、Boids鸟群运动以及小鼠红系细胞分化等三组数据集上进行实验。

**📈 对比分析**

与Curly-FM、WLM、OT-CFM、OT-MFM等方法对比，OT-KI在W1/W2误差上均优于基线，并在训练时间上相对WLM提升一百倍以上。

**⚠️ 局限性**

局限性包括未显式嵌入物理先验、主要针对插值任务、对OT耦合质量敏感，以及在高维或缺乏速度观测时效果尚待验证。

---

## 389. ReLaDrift: Recurrent Language Generation via Drifting for Test-Time Scaling

**arXiv ID:** 2610.23225 | [PDF](https://arxiv.org/pdf/2610.23225v1)

**作者:** Yuki Otsuka `[一作]` (Institute of Science Tokyo), Naoaki Okazaki `[通讯]` (Institute of Science Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ReLaDrift，一种通过共享漂移模型在连续潜在空间中递归细化语言生成的框架。

**💡 创新点**

创新点在于将单步漂移模型转变为可迭代细化器，仅在终止轮次使用漂移损失，避免时间反向传播，实现在推理阶段可调节计算量-质量权衡。

**🔧 技术方法**

使用漂移模型、漂移场、Transformer、循环投影、回声层（round embedding）、固定 Encoder‑Decoder（如 CodeT5）、多种输入格式（source、additive、conditional）以及终端轮次只梯度传递。

**📊 数据集**

主要数据集包括 Sudoku Hard（48k 训练/2k 验证）、GSM8K（1,319 测试）以及 TinyGSM（1.5M 训练，12 份 Python 解决方案）和 Qwen3.5-9B 生成的参考答案。

**📈 对比分析**

与 AR、MDLM、Duo、S‑FLM、FMLM+、FRM、DFM 及单步漂移模型对比。ReLaDrift 在 Sudoku Hard 上在 NFE=180 时达到 94% 准确率，优于所有基线；在 GSM8K 上在 NFE=16 时实现 29.4% 准确率，匹配或超过同等 NFE 的扩散/流模型，并在壁钟时间上更高效。

**⚠️ 局限性**

局限性包括：GSM8K 上性能依赖可执行验证器与预训练 CodeT5 编码器；训练成本约为单通道模型的 3.3 倍；仅在单个训练种子上评估 GSM8K；未验证更大规模或多任务场景；与自回归模型的性能差距仍未完全消除。

---

## 390. PersonaCUABench: Benchmarking Computer-Use Agents Serving Diverse Personas

**arXiv ID:** 2610.22990 | [PDF](https://arxiv.org/pdf/2610.22990v1)

**作者:** Haoming Li `[一作]` (University of Texas at Dallas), Xiaomin Li `[通讯]` (Harvard University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PersonaCUABench，用不同人设重写同一任务请求，评估计算机使用代理（CUA）的服务效果

**💡 创新点**

通过控制任务不变、仅改变请求者身份，揭示任务完成率与用户满意度的分离，并提供按需求“提出、参与、满足”诊断框架

**🔧 技术方法**

基于大语言模型的多工具代理（如MiniMax、Kimi、Grok等）与MatrAIx persona schema（37属性）生成请求与对话

**📊 数据集**

组合74个来自Odysseys和REAL的任务（信息+动作），以及44个在11个域的persona，形成370个任务-人设组合

**📈 对比分析**

在相同浏览器环境下跑9个主干模型，比较原始请求与4类persona请求的任务完成率与满意度；平均完成率从0.689下降到0.441，满意度从4.04降至3.1–3.6

**⚠️ 局限性**

仅限Web信息/动作任务，未考虑桌面应用、持续交互或真实用户反馈；使用合成persona可能带来刻板印象，结果不能直接推广至真实人群

---

## 391. Diagnosing GNN Failure Modes with Layer-Wise Probe Entropy

**arXiv ID:** 2610.22595 | [PDF](https://arxiv.org/pdf/2610.22595v1)

**作者:** Shanzi Bao `[一作]` (University of Cambridge), Pietro Liò `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

引入层级探针熵诊断，冻结已训练的GNN，在每层拟合线性探针并跟踪探针准确率与归一化预测熵，以分析信息流与失效模式。

**💡 创新点**

通过探针熵轨迹区分有益、中性、恶劣的平滑效果，并能区分结构不匹配与过深导致的过平滑两类失效模式，优于现有几何指标。

**🔧 技术方法**

使用线性探针、归一化熵计算、Dirichlet能量、MAD、R_Group等几何指标对比，GCN/GAT模型、GraphUniverse生成的控制图，统计相关性与Spearman系数等技术。

**📊 数据集**

使用12个真实图数据集（如Cora、Amazon-Ratings、Roman-Empire等）以及GraphUniverse生成的合成图，涵盖同质性与异质性场景。

**📈 对比分析**

对比探针熵与几何指标的相关性，探针熵在24个设置中相关性绝对值最高；在控制实验中，熵轨迹能及时识别失效，显示不同深度模型的失效起始位置，优于仅依赖最终层指标。

**⚠️ 局限性**

探针熵仅衡量预测分布集中度，对错误预测仍可能低熵；在异质图中熵与真实类别置信度的对齐弱化；需在目标图域验证，且实验仅覆盖GCN/GAT，未覆盖更广模型。

---

## 392. Trembling toward Duverger: Convergence and Cycles in Synchronous Plurality Dynamics

**arXiv ID:** 2610.23330 | [PDF](https://arxiv.org/pdf/2610.23330v1)

**作者:** Maximilian Weigel `[一作]` (ETH Zürich), Roger Wattenhofer `[通讯]` (ETH Zürich)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在有限选民且采用同步投票、利用微小颤抖信念来决定选票的确定性动态，并证明大多数情况下仅需一次更新即可收敛至杜韦热尔效应，但也构造了周期性循环（包含任意长周期、领导者轮替等），揭示同步投票可能导致的振荡。

**💡 创新点**

创新点在于将颤抖手精炼为确定性最佳反应映射，证明单步收敛的充分条件，并给出三候选人下可达任意长度周期循环的构造，从而展示同步多候选人投票下的复杂动力学行为。

**🔧 技术方法**

使用的技术主要是组合数学与概率分析（极小极大计数、波动系数展开）、递推映射与周期分析，以及有限状态机的确定性动态理论。

**📊 数据集**

未使用真实选举数据，而是构造了严格的偏好向量与投票配置作为理论模型，辅以计算机搜索验证最小周期条件。

**📈 对比分析**

由于模型为纯粹理论框架，未进行实验对比；但通过解析证明和有限枚举，展示了在大多数随机偏好下单步收敛概率接近1，而在罕见的近平局情况下可能出现周期振荡。

**⚠️ 局限性**

主要限制包括：仅考虑同步投票与单一投票规则；未考虑实际投票误差或随机抽样投票；模型假设所有选民对颤抖概率相同且趋于零；对随机偏好下周期长度分布未给出精确估计，且在更复杂投票制度下的适用性尚未探讨。

---

## 393. Preventing Data Hallucinations in Agentic AI for Optical Network Troubleshooting

**arXiv ID:** 2610.23252 | [PDF](https://arxiv.org/pdf/2610.23252v1)

**作者:** Jeremias Dötterl `[一作]` (Adtran GmbH), Martin Kuipers `[通讯]` (Adtran Networks SE)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并演示了一套基于代理式人工智能的光网络故障排查系统，该系统通过在代理架构中嵌入数据引用，能够防止数据幻觉，提升根因分析的可解释性和可信度。

**💡 创新点**

创新点在于将细粒度数据引用（URI）引入代理式AI，取代传统的文本推断，使根因结论和推理链可以直接指向时间序列、日志或JSON数据片段，实现证据支持的根因分析。

**🔧 技术方法**

主要技术包括大型语言模型（LLM）与LangChain的deepagents框架、异常检测器（基于规则或机器学习）、数据检索与可视化工具、证据查询API以及云端数据湖与数据采集器。

**📊 数据集**

使用的数据集为Adtran实验室现场设备（OLT与多台ONU）的实时遥测数据，包含流量计数、BIP错误、TX/RX功率、报警日志等，故障场景包括分布光纤切割、馈线切割以及ONU突发停电。

**📈 对比分析**

对比方法为将使用数据引用的证据型代理与不使用引用的传统代理进行交互式演示，结果显示后者易出现数据幻觉，而前者通过证据检索成功避免幻觉，提升了可解释性与操作员信任度；但论文未给出量化的性能指标或成本评估。

**⚠️ 局限性**

局限性包括：实验仅在Adtran内部网络进行，缺乏跨厂商或大规模真实网络的验证；对LLM生成的推理链仍存在错误风险；若数据引用生成错误，证据检索会失败，需额外警告机制；未提供具体的性能数值或泛化评估。

---

## 394. Code Skill Server: A Large-Scale Search Engine for Code Generation Skills

**arXiv ID:** 2610.23186 | [PDF](https://arxiv.org/pdf/2610.23186v1)

**作者:** Xiaoyong Li `[一作]` (Shandong University), Yue Yao `[通讯]` (Shandong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Code Skill Server（CSS），一个统一框架，用于从大规模技能库检索与代码生成任务相关的可重用技能，并通过服务接口提供给生成代理。

**💡 创新点**

创新点在于将技能搜集、标准化、按能力分类、索引以及检索与使用流程统一成一个可扩展的服务器架构，并展示了规模化技能库对生成质量的显著提升。

**🔧 技术方法**

使用的技术包括基于BM25、Graph-of-Skills、SkillTrace等检索方法、FASTADAPT任务适配组件、技能规范化与索引、以及多模型（GPT‑5.6、GLM‑5.3‑Flash等）评估。

**📊 数据集**

数据集包括从GitHub、Skills.sh、SkillsMP等公共源收集的993个代码生成技能（CSS‑993），以及BigCodeBench和CoderEval‑Python两个代码生成基准。

**📈 对比分析**

对比方法包括无技能(No‑Skill)、BM25、GoS、SkillTrace，实验显示最优配置在BigCodeBench从53.42%提升到60.88%，在CoderEval从33.04%提升到46.96%，且跨模型提升至少4.21%。

**⚠️ 局限性**

限制主要在于技能库仍需人工精细化构建，检索的相关性受技能描述质量影响，且对极大规模多样化技能库的性能尚未完全评估。

---

## 395. Recognizing Atari and Assessing Sente: Controlled Evaluation Tasks for Language-Based Go Agents

**arXiv ID:** 2610.22953 | [PDF](https://arxiv.org/pdf/2610.22953v1)

**作者:** Taiyi Chen `[一作]` (University of Washington), Amandalynne Paullada `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建基于围棋中打劫和先手概念的对照测试集，并对 LoGos 语言模型进行细粒度的术语识别与落子评估实验。

**💡 创新点**

创新点在于提出了可控的“打劫–先手对照测试”和“非先手打劫测试”，通过局部棋局变化来检验语言模型对领域术语的精准识别及其对落子价值的关联性。

**🔧 技术方法**

使用 LoGos-7B（基于 Qwen2.5-7B）在 QLoRA 框架下进行联合微调，输入包含规则标注的术语标签、KataGo 的胜率估计以及人类围棋点评文本。

**📊 数据集**

数据集包括从公开围棋记录自动生成的 1,000 对 X/Y 测试局（仅改动一枚棋子）和 500 个 Z 局（打劫但非先手），以及用于训练的英文围棋点评数据与 KataGo 评估结果。

**📈 对比分析**

对照方法为将模型输出的术语标签与规则生成的标签比对，落子价值则按 50% 以上/以下二值化后与 KataGo 参考值比较。结果显示：对 B 的非打劫判断准确率 91.8%，但对 A 的打劫判断仅 14.6% 正确；在相同落子在不同局中的价值比较上条件准确率 88.8%，但在候选落子内部比较时，尤其是 Z 局，仅 46.7% 能正确偏好更优落子。

**⚠️ 局限性**

局限性包括：①胜率预测极端化导致评价分辨率低；②仅评估单一 LoGos 模型，无法区分基底模型、微调数据或超参数的影响；③未验证术语预测与落子偏好之间的因果关系；④缺乏将语言模型与显式棋盘计算或搜索相结合的实验证据。

---

## 396. MAS-Bai: Linguistically Informed Adaptive Multi-Agent Translation for Bowen-Script Bai

**arXiv ID:** 2610.23205 | [PDF](https://arxiv.org/pdf/2610.23205v1)

**作者:** Jingting Zheng `[一作]` (Tianjin University), Deyi Xiong `[通讯]` (Tianjin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了利用多代理框架MAS‑Bai实现Bowen字写低资源语言Bai的Bai–中文翻译，并构建了6,536句Bai–中文平行语料。

**💡 创新点**

将语言学专业知识与适配路由、循环一致性自我演化结合，针对Bowen文字的声母借用与语序差异实现条件性推理路径选择。

**🔧 技术方法**

基于Qwen‑2.5‑72B‑Instruct的LLM骨干，加入Etymologist、Syntactician、Contextual Retriever和Translator四个专家，并使用前向LoRA、检索增强生成、Meta‑Agent条件路由及逆向翻译的循环一致性验证。

**📊 数据集**

构建6,536句Bai–中文平行语料（5,000句检索库+1,536句测试集），并对测试集按话语体与语义现象标注。

**📈 对比分析**

与零-shot、ICL、Standard RAG、Forward LoRA、Static MAS等基线对比，MAS‑Bai在BLEU 16.23、ChrF++ 35.74、COMET 62.15上均优于所有基线，尤其在Bowen音译和SOV语序子集提升显著。

**⚠️ 局限性**

仅在Bowen文字的Bai数据上验证，需针对新语言重新设计音译词表与路由规则；逆向翻译的可靠性受训练数据覆盖的限制；自我演化增加推理成本；人类评测样本有限。

---

## 397. Fast Distributed $(Δ+ 1)$-Edge Coloring in Bipartite Graphs

**arXiv ID:** 2610.22690 | [PDF](https://arxiv.org/pdf/2610.22690v1)

**作者:** Yann Bourreau `[一作]` (CISPA Helmholtz Center for Information Security), Ananth Narayanan `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种新的确定性分布式算法，能够在 LOCAL 模型下对二分图实现 (Δ+1)-边着色，并通过构造满足弱网（k,α,β）性质的图以及扩展的 X‑饱和匹配算法，将总体复杂度降至 O(f(Δ)·log n)，在常数度二分图上实现 O(log n) 的上界。

**💡 创新点**

创新点主要体现在：① 将求解 (Δ+1)-边着色问题从传统的 Vizing 链迭代方式转为逐色类构造饱和匹配；② 设计了一种针对弱网（k,α,β）结构的 X‑饱和匹配算法，扩展了以往仅适用于最大度与最小度满足 a>b 的情况；③ 通过双重覆盖（bipartite double cover）和复制节点的技巧，将无分裂信息的二分图转化为可直接使用饱和匹配的形式；④ 结合 sinkless orientation 的 O(log n) 方案，完成匹配后残留图的高效处理。

**🔧 技术方法**

主要技术包括：① LOCAL 模型下的邻域收集与局部判断；② 对弱网结构的度量与球形生长分析；③ 利用 Hall 定理与递归移除 Hall 图构造全局匹配；④ 结合 sinkless orientation 算法为低度顶点提供定向；⑤ 节点选择（Node Selection）和分层匹配等辅助子算法。

**📊 数据集**

本工作完全基于理论分析与算法设计，不使用任何实验数据集，所有结论均通过严格的分布式算法证明与复杂度分析得到。

**📈 对比分析**

与之前的最佳结果（Deterministic O(log⁵ n)，Randomized O(log² n)）相比，本文在二分图上将 n 的依赖降为 O(log n)（确定性），并在常数度二分图上给出了匹配的 Θ(log n) 上下界；此外，局部调色（local‑palette）版本亦在相同时间复杂度内实现，表明所用技术在更强约束下同样有效。

**⚠️ 局限性**

主要局限包括：① 对 Δ 的函数 f(Δ) 仍可能呈现超多项式增长，实际效率受 Δ 影响较大；② 目前仅针对二分图，扩展到一般图仍是开放问题；③ 对于非常低度图（如 Δ=2）虽已得到 O(log n) 结果，但进一步降低到 O(1) 仍未实现；④ 证明中使用了多重常数与复杂的参数设置，实际实现难度较高。

---

## 398. A Minimal ReLU Classifier on Non-Linearly Separable Data: Convergence and Implicit Bias at and beyond the Edge of Stability

**arXiv ID:** 2610.23007 | [PDF](https://arxiv.org/pdf/2610.23007v1)

**作者:** Quan Nguyen `[一作]` (University Of British Columbia), Mark Schmidt `[通讯]` (University Of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在一维非线性可分数据上训练两神经元ReLU网络的梯度下降过程，分析了大步长和自适应学习率下的收敛、边界行为与隐式偏置，并给出了理论证明与实验验证。

**💡 创新点**

提出“活跃锥”概念，给出在此区域内对大步长梯度下降的全时序收敛、EoS与catapult现象的精确描述，以及自适应步长实现的加速收敛速率O(logT/T)。

**🔧 技术方法**

使用梯度下降、对ReLU网络的分段逻辑回归建模、几何分析、潜在极限证明以及自适应学习率调度等技术。

**📊 数据集**

主要使用了三点一维数据集（-1,0,1）和MNIST二分类（3与5）作为实验数据集。

**📈 对比分析**

通过与传统logistic回归及固定学习率梯度下降对比，实验显示自适应步长在活跃锥内实现了更快的损失下降速率，且激活模式持续切换。

**⚠️ 局限性**

局限性包括仅在一维、两神经元、固定输出权重的设置，无法直接推广至高维、宽层或深层网络，需要进一步研究。

---

## 399. NLData2Opt: Benchmarking Parameter Derivation and Prediction for Modeling Optimization Problems from Natural Language and Data

**arXiv ID:** 2610.22889 | [PDF](https://arxiv.org/pdf/2610.22889v1)

**作者:** Lijia Liu `[一作]` (Kyoto University), Hisashi Kashima `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 NLData2Opt 基准，用于评估大语言模型在真实数据条件下进行多阶段优化建模的能力。

**💡 创新点**

创新点在于将问题拆分为模型–数据分离、参数提取、推导和预测四个阶段，构建自动化转换管线，并在基准中加入噪声与冗余属性，以更贴近工业场景。

**🔧 技术方法**

采用 LLM prompt、CoE（多代理）框架、SFT、自动化构造管线以及 AMPL 建模；评估指标包括实体 F1（EF1）、执行基准（EBC）和 LLM‑ES。

**📊 数据集**

基准数据来源于 NLP4LP 与 Mamo 的 LP/MILP 任务，经过自动化转换后形成 NLData2Opt 的九种任务设置。

**📈 对比分析**

与多款商业（GPT‑4o、Claude、Gemini 等）和开源（Llama‑3、Mistral 等）LLM，以及 CoE、LLMOPT、SFT 基线进行对比；结果显示在 NLData2Opt 上，最优模型的执行基准（EBC）仅最高 0.77，整体性能显著低于现有基准，揭示了实际场景中的挑战。

**⚠️ 局限性**

局限性包括：基准生成依赖 LLM 可能产生幻觉或逻辑错误；评估仍受限于执行和 LLM‑ES 的可检测范围；仅涵盖 LP/MILP 与 AMPL 形式，缺乏更复杂工业优化的覆盖；未将概念理解与形式化能力完全分离。

---

## 400. Erase the Mark, Preserve the Motion: Temporally Consistent Removal of Video Watermarks

**arXiv ID:** 2610.22457 | [PDF](https://arxiv.org/pdf/2610.22457v1)

**作者:** Yifan Liao `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Xinlei He `[通讯]` (Wuhan University)

**通讯引用:** 969 | [OpenAlex ID](https://openalex.org/A5074889369)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一种基于姿态一致的框架，在保持视频运动和时间连续性的前提下，去除多种隐藏式视频水印。

**💡 创新点**

创新点：①使用全局共享的视角扰动打乱水印空间模式；②引入姿态条件的全序列重构，并通过轻量化的Motion Consistency Adapter让冻结的Video DiT适配姿态约束；③采用全序列Flow Matching监督，实现跨帧一致性和运动保持。

**🔧 技术方法**

技术手段：视频Diffusion Transformer (DiT)、VAE编码解码、Pose VAE、轻量LoRA Adapter、全序列Flow Matching、姿态提取（SAM+PoseAnything）、共享视角变换。

**📊 数据集**

实验数据集：在HunyuanVideo和Wan2.2生成的100条视频上，对11种后置和生成内置的水印方案（DWT-DCT、StegaStamp、VideoMark等）进行评估；使用XPose训练数据集对Adapter进行训练。

**📈 对比分析**

与传统图像扰动、信号优化（UnMarker、WEvade）及生成重构（WatermarkAttacker、CtrlRegen、NFPA）等三类基线对比。平均水印位误差下降至0.505（接近随机），且所有方案均低于0.6；在VBench 1.0/2.0评测中，Motion Smoothness、Human Identity等指标均超过最强基线，CLIP‑FID也显著下降。

**⚠️ 局限性**

局限性：对遮挡或运动模糊的视频效果不稳定；仅针对隐蔽水印，无法处理可见标识；攻击仍需较高算力，且在极端几何变换或不准姿态估计时性能可能下降。

---

## 401. Language Models Learn to Improve Language Models

**arXiv ID:** 2610.23344 | [PDF](https://arxiv.org/pdf/2610.23344v1)

**作者:** Minhua Lin `[一作]` (Pennsylvania State University), Hanqing Lu `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种使用RL训练LLM以生成针对语言模型开发的高效改进提案的方法，并通过研究世界模型（RWM）提供可扩展且可靠的反馈。

**💡 创新点**

创新点在于：1）将冻结的LLM作为RWM，用实验记录做上下文预测实验收益，从而大幅减少昂贵的真实实验；2）引入基于当前策略的“on‑policy calibration”，周期性执行高估计的提案并将真实结果加入RWM上下文，持续保持预测可靠性；3）在多种AR与扩散模型环境下验证该框架可提升提案质量并显著降低实验成本。

**🔧 技术方法**

核心技术包括：强化学习（RL）与可验证奖励（RLVR）结合的DAPO算法；冻结LLM作为RWM来预测实验收益；奖励设计包含归一化、裁剪以及多样性与有效性惩罚；on‑policy calibration机制；对比实验与多种基线（Zero‑shot、ICL、SFT、RFT‑Real）。

**📊 数据集**

使用了多种语言模型训练环境（四类自回归模型如OLMo-3、Qwen3、NanoChat、Marin，以及四类扩散模型如E2D2、MDM、BD3LM等），共计40个训练任务和10个保留测试任务，并在训练任务上执行了300条实验记录，其中100条用于RWM初始上下文。

**📈 对比分析**

与基线相比，所提出的方法在保留任务上实现了最高的 Avg@4 与 Best@4（自回归任务17.06/23.40，扩散任务7.10/9.93），并在相同数据生成成本下比基线降低约33% 的实验开销；相较于 GPT‑6‑sol 甚至在有 300 条记录的ICL场景下也能获得更高收益，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：1）RWM的预测仍存在误差，需周期性真实实验校准；2）方法主要针对提案级改进，未覆盖完整的研究周期；3）在更大规模或更高风险场景下的安全性和可靠性仍需进一步验证；4）对训练资源的依赖（大型LLM与GPU成本）较高。

---

## 402. Trust but Verify: Designing Region-Inhomogeneous Forward Dynamics for Discrete Diffusion

**arXiv ID:** 2610.22380 | [PDF](https://arxiv.org/pdf/2610.22380v1)

**作者:** Gunmay Jhingran `[一作]` `[通讯]` (Delhi Technological University), Gunmay Jhingran (Delhi Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在离散扩散模型中引入了区域非均匀的前向过程，将吸收核与轻量级混合核按区域分配，并在逆向时使用检测-重掩盖-重填补采样器，提升了对上下文错误的检测与修复。

**💡 创新点**

创新点在于把离散扩散的前向过程从单一吸收核重新开放为可调节的设计空间，通过在上下文区施加轻度混合噪声实现可观测的置信度评分，并结合逆向重噪声策略实现无额外标注的上下文修复。

**🔧 技术方法**

采用了掩码扩散模型（MDLM）框架、区域分区的混合腐败核、权重加权交叉熵损失、以及基于模型自身似然的检测-重掩盖-重填补采样器。

**📊 数据集**

使用了公开的170M预训练MDLM（OpenWebText），在FineWeb‑Edu数据集上进行微调，并在含有均匀或可疑（GPT‑2产生）上下文错误的Web文本段落上评估。

**📈 对比分析**

与基线预训练模型、仅推理阶段的修复以及外部基于GPT‑2惊奇度的检测器对比，均匀噪声下模型在token‑F1提升约3–4点、召回率>95%、误编辑率<1%，但在可疑噪声下提升不足0.5点，未达到统计显著性。

**⚠️ 局限性**

局限性包括仅在170M规模、单一英语Web文本上验证、对可疑噪声的修复效果有限、混合核与退火参数的 ablation 混淆、以及缺乏对更大模型或不同域的泛化评估。

---

## 403. PrairieFire: Igniting High-Throughput Updates in Large-Scale Approximate Nearest Neighbor Search

**arXiv ID:** 2610.23232 | [PDF](https://arxiv.org/pdf/2610.23232v1)

**作者:** Yuchen Huang `[一作]` (East China Normal University), Chuliang Weng `[通讯]` (East China Normal University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PrairieFire系统，融合CPU、GPU与SSD以实现大规模向量集合的高吞吐量增量更新和搜索；

**💡 创新点**

创新点在于缓冲式增量更新与批量局部重构、级联中心点调整、细粒度并发控制以及CPU‑GPU‑SSD协同工作流程；

**🔧 技术方法**

采用了分层三级聚类索引、解耦锁机制、GPU批量距离计算、GPU微型聚类、NVMe SSD高速IO与Kafka写前日志；

**📊 数据集**

使用SIFT1B、SPACEV1B、Wiki768、Gist960以及生产级128维推荐数据进行评估；

**📈 对比分析**

与Greator、OdinANN、SPFresh、SVFusion等基线在Workload A/B/C上对比，PrairieFire实现了3–35倍的更新吞吐量、3–5倍的搜索吞吐量，尾延迟更低，召回率相当或更优，成本/能耗/存储效率提升1.5–8.7倍；

**⚠️ 局限性**

局限在于对极高维度数据性能衰减、对多GPU/SSD硬件依赖、并发控制实现复杂，并且仍需批量处理以抵消单个向量频繁IO的开销。

---

## 404. GlobeBench: Evaluating Text-Based World Model Capabilities for Scalable Environment Simulations

**arXiv ID:** 2610.22708 | [PDF](https://arxiv.org/pdf/2610.22708v1)

**作者:** Nicholas Saban `[一作]` (Patronus AI), Darshan Deshpande `[通讯]` (Patronus AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

建立了一个名为 GlobeBench 的 1,632 条文本世界模型基准，涵盖代码、企业、游戏、数学科学与对话五个领域，并细化为 19 项技能与四大能力；

**💡 创新点**

创新点在于将任务拆解为单一转移、使用可执行后端产生精确金标准、引入混合评判器（确定性门 + LLM 判定）以及通过 reach‑back 与 impact radius 进一步定位难点；

**🔧 技术方法**

使用多大模型（GPT‑6 Astra、Claude Opus、Gemini、Qwen 等）作为零样本文本世界模型，并搭配混合评判器（GPT‑5.6 Terra 等）进行评估；

**📊 数据集**

数据集来源于可执行环境（如 TextWorld、SymPy、τ²‑bench 等）的轨迹，人工与自动标注相结合，确保每个项目都有执行得到的金标准；

**📈 对比分析**

对 20 模型（18 通用 LLM + 2 专业模型）进行比较，最佳模型 GPT‑6 Astra 成功率为 49.57%，整体通过率低，631 项未被任何模型解决，显示显著差距；

**⚠️ 局限性**

局限在于只评估单次转移且基准偏向 L3 长期审计，未测量模拟器错误累积；reach‑back 与 impact radius 与技能交叉影响，难以单独归因模型弱点。

---

## 405. ToPA: Block-wise Toeplitz Adaptation for Expressive and Efficient Fine-Tuning

**arXiv ID:** 2610.22883 | [PDF](https://arxiv.org/pdf/2610.22883v1)

**作者:** Sicong Li `[一作]` (Chinese Academy of Sciences), Qingming Huang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了参数高效微调（PEFT）中低秩与稀疏限制的不足，提出基于块Toeplitz矩阵的ToPA方法，能够在保持极低参数量的同时实现高秩、稠密的权重更新，并在多种模型和任务上验证其优越性。

**💡 创新点**

创新点在于：①将块Toeplitz矩阵作为重参数化结构，引入残差连接和特殊初始化，显著降低梯度爆炸；②理论证明块Toeplitz乘积可在更短链长下逼近任意矩阵，表达能力超越低秩与稀疏方法；③在实验中展示了在相同或更小参数预算下的显著性能提升。

**🔧 技术方法**

技术：Toeplitz/块Toeplitz矩阵乘法（FFT加速）、残差学习、LoRA式初始化、梯度结构分析、理论证明、模型微调框架。

**📊 数据集**

数据集：Commonsense170K、GLUE、MetaMathQA子集、GSM8K、MATH、CodeFeedback、HumanEval、MBPP、DreamBooth、Stable Diffusion等。

**📈 对比分析**

对比方法包括LoRA、DoRA、PiSSA、HiRA、RidgeLoRA、OFT、BOFT、HRA等；在LLaMA、Qwen、Stable Diffusion、DeBERTa等多模型上进行Commonsense Reasoning、Math Reasoning、Code Generation、Subject‑Driven Generation、GLUE等任务评测，ToPA在参数量相同或更少的情况下平均提升0.3%–2%，在常识推理、数学推理和图像生成等任务表现最为突出。

**⚠️ 局限性**

限制：需要调节块数p和链长t，过长链仍可能导致梯度不稳定；实验主要聚焦LLM与部分视觉模型，未验证更大模型或跨域迁移；相较于纯插值或极简方法，计算开销仍略高。

---

## 406. ECHOES: Seeing Beyond the Current Image via Recursive History Compression and Privileged On-Policy Distillation

**arXiv ID:** 2610.22408 | [PDF](https://arxiv.org/pdf/2610.22408v1)

**作者:** Tengda Guo `[一作]`, Liujuan Tang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用训练时多图特权信息，提出 ECHOES 方法通过递归文本压缩（Compression–Think–Action）实现长时序 GUI 任务的历史状态保留，在推理阶段仅使用当前图像和自生成文本状态，避免保存历史截图。

**💡 创新点**

创新点包括① 训练时使用多图特权的 on‑policy distillation（OPD）辅以强化学习（GRPO）实现多模态知识迁移；② 采用 PMF Gap‑guided 动态平衡 OPD 与 RL；③ 在每轮迭代刷新教师，形成自适应学习进程；④ 构建 ECHOESBMK 诊断基准以细粒度评估感知与记忆保真度。

**🔧 技术方法**

核心技术包含 CTA 递归状态建模、on‑policy distillation、PMF Gap‑guided OPD–RL 调度、Qwen3‑VL‑8B 以及其它多模态 LLM；同时使用强化学习算法 GRPO 进行任务奖励优化。

**📊 数据集**

使用四大 GUI 评测基准（GUI‑Odyssey、AndroidControl‑v2‑High、AITW、AMEX‑L3）、MageBench‑Sokoban（零样本迁移评测）以及自研的 ECHOESBMK 诊断数据集。

**📈 对比分析**

在上述基准与多种开源/闭源模型（Qwen、GLM、Step3、MAI‑UI、InfiGUI、Gemini、GPT‑5.5 等）进行对比，ECHOES 在 GUI 任务中均取得 6–20% 的性能提升，并在推理成本上保持较低水平；在 MageBench‑Sokoban 上零样本迁移也实现了相对提升。

**⚠️ 局限性**

局限性在于细粒度状态保留仍有不足，PMF Gap 虽缩小但仍存在；长时序任务中记忆衰退和跨模态信息丢失仍是挑战；方法对训练时多图数据的依赖可能限制在某些低资源场景的适用性。

---

## 407. Watermarkability: Measuring and Improving Language-Model Compatibility with Logit-Based Watermarks

**arXiv ID:** 2610.22810 | [PDF](https://arxiv.org/pdf/2610.22810v1)

**作者:** Zewei Deng `[一作]` (University of Minnesota), Liyan Xie `[通讯]` (University of Minnesota)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了水印可写性（watermarkability）度量，并设计了一种两阶段方法（低秩适配器+闭式控制器）来降低logit-based水印对语言模型生成质量的影响。

**💡 创新点**

创新点在于：① 将水印可写性成本定义为实现目标集合所需的最小KL散度；② 利用16分区词表的通用低秩适配器，能够在不改动水印规则的前提下显著降低分布偏移；③ 引入闭式可靠性控制器，实现目标概率≈0.99的高可靠性输出。

**🔧 技术方法**

主要技术包括：LoRA低秩适配器、词表分区与目标集条件编码、闭式logit偏移控制、KL损失约束、对数似然优化等。

**📊 数据集**

使用的实验数据集为ELI5/LFQA长文本问答提示，包含5000个校准上下文、500个验证上下文及256个保留测试提示，用于训练和评估。

**📈 对比分析**

通过与固定偏置δ=20的基线进行对比，评估PPL和token adherence。实验表明在OPT-125M上PPL平均降低46.3%，在Qwen3-8B上降低5.2%，且所有水印规则下的遵从率均≥99.80%。

**⚠️ 局限性**

局限性包括：对大模型的效果提升有限（Qwen3-8B提升不如OPT-125M）；方法需要针对特定词表分区与水印规则进行训练；未覆盖不使用目标集合的水印形式；对不同prompt领域的泛化尚需进一步验证。

---

## 408. Evidence-Gated Bilingual RAG for Reliable Consumer Banking Search

**arXiv ID:** 2610.22652 | [PDF](https://arxiv.org/pdf/2610.22652v1)

**作者:** Ahmed Mohamed Mahmoud Rayyan `[一作]` (Queen's University), Rahatara Ferdousi `[通讯]` (Queen's University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种证据门控的双语检索增强生成（RAG）框架，用于消费者银行搜索；

**💡 创新点**

创新点在于将网页内容清洗为检索友好型证据、结合密集与稀疏检索、意图感知证据选择以及三种答案决策（Answer / Related Mismatch / No Answer），并在本地部署而非云端。

**🔧 技术方法**

技术包括BGE-M3多语言嵌入、BM25+稀疏重评分、Cross‑Encoder重排序、意图分类与实体抽取、RRF融合、意图匹配门控与Qwen3局部生成。

**📊 数据集**

使用了来自埃及国家银行（NBE）的公开网页构建的双语知识库（555篇文档、962段），以及220个双语查询（127阿拉伯语+93英语）用于检索评估。

**📈 对比分析**

与基线混合检索（BM25+重排序）对比，证据门控模型在210个有标注URL的查询中Top‑1提升至79.05%（+28.57pp），MRR@5提升至0.7974（+0.2266），整体检索与答案决策表现更好。

**⚠️ 局限性**

局限包括评估样本依赖单一银行、查询为参考集改写且未包含代码混杂、结论基于自动评判而非人工核实，缺乏独立数据集与人类评测。

---

## 409. KFS$^4$LAM: Adaptive KeyFrame Selection with Structure-Aware Mapping for Endoscopic Gaussian Splatting SLAM

**arXiv ID:** 2610.22505 | [PDF](https://arxiv.org/pdf/2610.22505v1)

**作者:** Xinwei Ju `[一作]` (University College London), Francisco Vasconcelos `[通讯]` (University College London)

**通讯引用:** 1451 | [OpenAlex ID](https://openalex.org/A5068825369)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 KFS^4LAM，结合运动引导的自适应关键帧选择与结构感知的 Gaussian Splatting 映射，实现柔性内镜 SLAM。

**💡 创新点**

创新点在于：①基于累计运动与视觉抑制的自适应关键帧政策；②利用肠折叠边界监督和受限时序变形的结构感知映射；③与多种 Gaussian SLAM 架构无缝迁移。

**🔧 技术方法**

采用 Gaussian Splatting、运动累积关键帧生成、Sobel 边缘抑制、Fold 预测模型、Boundary‑DT 损失、时序变形正则化以及动态映射预算。

**📊 数据集**

使用公开的 C3VD 与更具挑战性的 C3VD‑v2 内镜 RGB‑D 数据集。

**📈 对比分析**

与 GaussianSLAM、Endo‑2DTAM、MonoGS、EndoGSLAM、SplaTAM 等方法对比，KFS^4LAM 在轨迹 ATE/ARE、RPE、深度 RMSE、PSNR/SSIM 上取得显著提升，且关键帧比例不升高。

**⚠️ 局限性**

局限性：自适应映射预算导致计算量增加；Fold 边界监督依赖外部预测，在边界弱化或遮挡时可靠性下降；在更广泛临床数据上的泛化需进一步验证。

---

## 410. RAMP: Toward Ubiquitous Quantum-Safe Key Distribution as a Service

**arXiv ID:** 2610.22879 | [PDF](https://arxiv.org/pdf/2610.22879v1)

**作者:** Jiafan Wang `[一作]` (CSIRO Technology), Dongxi Liu `[通讯]` (CSIRO Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于服务的量子安全密钥分发框架RAMP，使普通用户可在不拥有量子硬件的前提下，通过服务商提供原始量子密钥材料并在本地完成掩码与后处理来生成最终共享密钥。

**💡 创新点**

核心创新在于双域抗妥协设计——将量子密钥生成与用户侧掩码处理分离，构建保留量子服务域与掩码域的互补安全边界，实现在任一域被攻破时仍能保持最终密钥机密性。

**🔧 技术方法**

使用了同步流密码（AES-256-CTR）生成掩码流、受认证的后量子密钥交换（ML-KEM-768）、签名验证（ML-DSA-65）、TLS 1.3 PSK通道、以及标准QKD后处理链（sifting、参数估计、信息纠正（LDPC）、隐私放大）。

**📊 数据集**

实验数据来源于qOptica 100 级连续变量QKD测试床，采集约3.65 GB的原始密钥块并在此基础上进行处理。

**📈 对比分析**

与传统端到端QKD对比，用户侧处理的额外开销仅占总处理时间的约4%，但由于LDPC纠错仍是瓶颈，最终密钥生成速率约为2.1 Kb/s（高端M4 Pro）和0.42 Kb/s（i7-1265U）。

**⚠️ 局限性**

局限性包括：需依赖服务端提供高质量原始密钥；掩码密钥协商耗时（约2.3 s）受外部OpenSSL进程管理影响；实验仅在单一低信噪比CV-QKD参数和本地网络环境下验证，实际部署需针对不同网络延迟、QKD配置与硬件做进一步优化。

---

## 411. DIFFLET: A Fast Inference Engine for Diffusion Transformer Models on AWS Trainium

**arXiv ID:** 2610.23047 | [PDF](https://arxiv.org/pdf/2610.23047v1)

**作者:** Bin Ma `[一作]` (Yotta Labs), Dong Li `[通讯]` (Yotta Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在AWS Trainium2上设计并实现了一种高效的扩散Transformer模型推理与服务引擎，该引擎统一编译后端、并行执行与步骤缓存，支持多种图像与视频模型。

**💡 创新点**

创新点在于：①使用共享AOT编译后端，使不同模型管线共用同一编译体；②按注意力头划分核心的联合注意力方案，保持与未拆分注意力相同的数值结果；③在设备侧实现轻量级探测器（probe）获取缓存信号，仅返回单一标量；④将所有程序驻留在服务进程中，实现无重编译的持续驻留与快速恢复。

**🔧 技术方法**

技术包括：AWS Trainium2 Neuron SDK 的 AOT 编译与 Neuron Kernel Interface；Ulysses 与 Ring 上下文并行模式；Fused Attention kernels；Probe 机制；固定步长、在线增量与校准自适应三种步骤缓存策略；驻留服务与故障恢复流程。

**📊 数据集**

评估数据集：DrawBench、PartiPrompts（图像）与 VBench（视频）等公开提示集，用于PSNR/LPIPS质量测评。

**📈 对比分析**

对比方法：与 PyTorch/XLA SDPA baseline 及 NxDI 进行对比，衡量编译时间、请求延迟、步骤延迟；在五个模型上实现 3.0–8.5× 的请求延迟缩短，注意力核提升 1.2–4.3×，步骤缓存带来 1.4–1.6× 的速度提升，质量差距仅 0.5–3 dB 内。

**⚠️ 局限性**

局限性：需要固定张量形状，难以自适应更大分辨率/帧率；Probe 计算在某些模型（如 Wan）成本高；单芯片内存受限导致部分大模型无法驻留；多租户并发与资源共享的性能评估尚未完整。

---

## 412. A100: A Hybrid Vision Foundation Model for Agricultural Plant Image Analysis

**arXiv ID:** 2610.22991 | [PDF](https://arxiv.org/pdf/2610.22991v1)

**作者:** Mayur Pratim Das `[一作]` (Annam.AI IIT Ropar), Mukesh Saini `[通讯]` (IIT Ropar)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并公开了A100，一种自监督预训练的混合CNN‑Transformer农作物视觉基础模型，并在多任务（作物识别、病害检测、叶片计数）上实现高精度。

**💡 创新点**

创新点在于：① 大规模（300万张）农作物无标签图像自监督预训练；② 采用多目标SADG框架（DINO、iBOT、KoLeo）提升多尺度特征；③ 结合局部MBConv与全局Transformer的混合骨干，兼顾纹理与全局上下文。

**🔧 技术方法**

使用的技术包括：混合CNN‑Transformer骨干（MBConv+Transformer）、自监督预训练（DINO自蒸馏、iBOT遮挡重建、KoLeo均匀正则）、分阶段学习率、EMA教师、MixUp/CutMix等细化训练技巧。

**📊 数据集**

预训练数据：3M未经标注的田间图像，覆盖100+作物种类；下游数据集包括：32类作物识别（1.14M图像）、44类病害检测（5.2K图像）、GroMo25叶片计数（≈88K图像）以及外部基准PlantVillage、PlantDoc、PlantWild。

**📈 对比分析**

与ImageNet‑12K监督初始化的同一架构相比，A100在线性探针下Crop+2.56%、Disease+0.97%；1%标签下Crop+2.81%、Disease+2.20%；与公开农业基础模型相比，1%标签下在Crop识别上领先14.2点。外部基准上，PlantVillage 99.96%，PlantDoc 64.89%（+4.30%对比ImageNet），PlantWild 76.96%。

**⚠️ 局限性**

局限性：仅验证单一CoAtNet‑2架构，缺乏对其他ViT/Swin的迁移实验；多目标SSL贡献未做细粒度消融；预训练语料库去重仅基于文件名/大小，未做感知哈希去重；在极低标签或噪声场景（如PlantDoc）仍有提升空间。

---

## 413. DiffusionRVGE: Reinforcement Learning via Regression-Based Value Gradient Estimation

**arXiv ID:** 2610.22973 | [PDF](https://arxiv.org/pdf/2610.22973v1)

**作者:** Jiayuan Wang `[一作]` (Tsinghua University), Tongyao Pang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过中心化的干净图像回归（Clean‑Image Regression）在不需要反向传播梯度的情况下，针对奖励信号生成高效的更新方向，从而对扩散模型进行强化学习微调。

**💡 创新点**

创新点在于：① 用回归方法直接从最终去噪图像中提取价值梯度方向，省去每一步采样的梯度估计；② 采用奖励自适应的回放带宽与查询权重，统一处理连续与离散奖励；③ 在训练过程中无需对整个去噪轨迹求导，显著降低计算成本。

**🔧 技术方法**

主要技术包括：基于 Gaussian 噪声的回归分析、正则化的岭回归、对齐分析（clean‑image 与 noisy‑state 方向对齐）、奖励自适应的回放设计、LoRA 微调、以及 ODE/ SDE 采样。

**📊 数据集**

使用了多个公开数据集：GenEval（553 组 prompt）、OCR（1016 组 prompt）、DrawBench（PickScore 等 200 组 prompt）以及 SD3.5‑Medium 的图像数据。

**📈 对比分析**

与 DiffusionNFT、Flow‑GRPO 等现有方法比较：在联合训练的 5‑reward 场景下，模型在 7/8 个评价指标上匹配或优于 DiffusionNFT，仅消耗约 1/5 的训练计算；单奖励训练时，在 PickScore、OCR、GenEval 上分别提升 2.1–6.7 倍的学习速度，整体训练成本显著降低。

**⚠️ 局限性**

局限性包括：1）对极度稀疏或高度非平滑的奖励函数仍需更多探索；2）目前仅在 SD3.5‑Medium 上验证，跨模型推广需进一步验证；3）虽然省去了梯度估计，但回归方向仍受回放设计与数据分布影响，可能在奖励结构极其复杂的任务中表现受限。

---

## 414. Flat Loss, Evolving Features: Grokking in Modular Addition

**arXiv ID:** 2610.22593 | [PDF](https://arxiv.org/pdf/2610.22593v1)

**作者:** Akash Kumar `[一作]`, Karthik Sankaranarayanan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在模块加法任务上，作者证明并实验显示，当训练交叉熵在一个相对平坦窗口内保持正值时，网络仍可通过特征演化实现从0到90%以上的泛化准确率；

**💡 创新点**

创新点在于提出并证明了“训练交叉熵平坦但特征在演化”这一共存现象，首次给出对一隐藏层网络在有限窗口内同时保持训练完好、提升 Fourier 特征比率和泛化准确率的理论保证；

**🔧 技术方法**

采用了支持-极化（supported‑Polar）隐藏层更新、归一化输出层梯度步长、权重衰减以及训练时对训练集进行岭回归的“train‑only refit”作为诊断指标，并利用 Fourier 分量度量隐藏特征的结构；

**📊 数据集**

主要使用了模 p（p 为奇素数）上的模块加法数据集，分别在结构化半格划分与随机划分两种训练集拆分方式下进行实验，宽度从512到p^2(p−1)^2不等；

**📈 对比分析**

通过比较训练交叉熵、训练/测试准确率以及 Fourier 分量的变化，实验表明在一隐藏层网络中 90% 以上测试准确率可在 CE 维持在 log p 附近 0.02 以内的窗口内实现；更深网络和单块 Transformer 在相应设置下亦可实现类似的泛化提升，训练误差保持零或极低；

**⚠️ 局限性**

局限性包括理论仅针对构造的初始化区域、有限窗口和小输出步长，未证明从 iid 初始化即可进入该区域；更深网络与 Transformer 的结果缺乏严格的理论保障；方法依赖特定的梯度归一化更新和权重衰减设置，可能不适用于更广泛的任务和网络结构。

---

## 415. When Ubiquitous Health Data Stops Helping: Patients' Experience of Breakdowns in the Data-Interpretation-Action Loop at the End of Life

**arXiv ID:** 2610.23259 | [PDF](https://arxiv.org/pdf/2610.23259v1)

**作者:** Yibo Meng `[一作]` (Cornell University), Zhicong Lu `[通讯]` (George Mason University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对25名末期患者和8名临床医生的半结构化访谈，研究了患者在终末期护理中面对无处不在的健康数据时的感知、解释与行动过程，并基于访谈资料进行归纳性主题分析。

**💡 创新点**

创新点在于：①首次系统阐明了终末期护理中数据-解释-行动循环的三种典型失效模式（反复无解的意义建构、行动与情境不匹配、难以完全退出数据环境）；②提出“时间不可逆性”概念，强调终末期时间窗口的有限性；③基于发现提出针对解释稳定性、情境对齐与存在支持的设计建议。

**🔧 技术方法**

采用的技术手段为：半结构化访谈、音频录制与转写、中文转英文双向翻译、反思性主题分析（reflexive thematic analysis），以及研究者共同编码与讨论的迭代过程。

**📊 数据集**

数据集为：25名终末期患者（涵盖不同疾病、居住环境、教育背景）以及8名参与其护理的临床医生，访谈时长约45–75分钟，收集了丰富的文字材料。

**📈 对比分析**

由于本研究为质性探索性研究，未采用量化对比或性能评估方法；结果通过主题归纳与案例阐述呈现，主要以参与者的叙事为证据，未给出客观指标或统计性能。

**⚠️ 局限性**

局限性包括：①样本规模相对有限，且以医院场景为主，家庭场景样本不足；②未纳入家庭照护者或患者家属的视角，可能遗漏与决策权转移相关的细节；③访谈依赖自述，存在社会期望与记忆偏差；④研究在中国三级医院背景下进行，文化与制度因素可能影响可推广性。

---

## 416. Generative adversarial networks with Monte Carlo dropout uncertainty quantification for structural health monitoring: Fibre Bragg grating data augmentation comparison in Hawk T1A aircraft

**arXiv ID:** 2610.22436 | [PDF](https://arxiv.org/pdf/2610.22436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 417. PASS-VLN: Passability-Aware Local-Command Vision-and-Language Navigation

**arXiv ID:** 2610.22812 | [PDF](https://arxiv.org/pdf/2610.22812v1)

**作者:** Cong Zou `[一作]` (Qiyuan Lab), Yu Wang `[通讯]` (Qiyuan Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种PASS-VLN框架，专注于在不同机器人身体宽度下实现可通过的本地命令导航，能够在可通过路径上到达目标并在不可通过路径前安全停止。

**💡 创新点**

创新点在于将身体宽度直接嵌入到鸟瞰视图几何表示中，实现了宽度可变的通过性判断；同时采用离散潜在动作编码实现连续闭环执行，并通过配对宽度对照实验验证了宽度适应性。

**🔧 技术方法**

技术主要包括：基于深度的鸟瞰视图投影与障碍膨胀、面向宽度的几何编码、融合语言、视觉和历史记忆的因果多模态骨干、潜在动作学习与解码、以及闭环低级控制。

**📊 数据集**

使用了HM3D和MP3D训练集（约192K条目）以及MP3D val_unseen、R2R-CE等公开室内导航数据集，并在真实Unitree Go2-W机器人上进行了实地部署验证。

**📈 对比分析**

与StreamVLN、DualVLN、JanusVLN等基线相比，PASS-VLN在本地命令任务中取得了最高的成功率、路径长度加权成功率和碰撞率下降；在宽度泛化测试中，在未见宽度下的碰撞率下降约12个百分点。

**⚠️ 局限性**

局限性包括：对宽度的依赖仍需在更宽/更窄的身体尺寸下进一步验证，潜在动作的离散化可能导致细微控制失真，以及在高度动态或复杂开放空间中的通行性尚未充分评估。

---

## 418. Harness-Delta Attribution: Dissecting Gains From Harness Evolution

**arXiv ID:** 2610.22466 | [PDF](https://arxiv.org/pdf/2610.22466v1)

**作者:** Wenxuan Ding `[一作]` (Amazon), Greg Durrett `[通讯]` (New York University)

**通讯引用:** 4867 | [OpenAlex ID](https://openalex.org/A5015133105)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种称为 Harness-Delta Attribution (HDA) 的评估协议，用于将基于固定权重模型的 harness 进化所带来的性能提升拆分为三类：过拟合、运行时计算扩展以及真正的可迁移改进。

**💡 创新点**

创新点在于系统化地量化 harness 进化的真实收益来源，发现大多数提升来自过拟合或计算扩展，而非模型本身改进，并提供了可复用的 HDA 工具包与人类审核流程；同时指出单纯的 hold‑out 验证不足以防止过拟合。

**🔧 技术方法**

技术上结合了 Meta‑Harness、SkillOpt 与 AHE 三种进化框架，对固定 LLM 的 harness 进行代码、技能与工具等多维度优化；HDA 通过构造计算匹配基线、去除检测到的快捷方式的中立化 harness、配对 bootstrap 显著性检验和人工审核来实现收益归因。

**📊 数据集**

使用四大基准数据集：数学推理 LiveMath、软件工程 SWE‑bench Verified、创意生成 CREATE 以及实体规划 ALFWorld。

**📈 对比分析**

比较方法：将进化后的 harness 与基线、计算匹配基线、去除快捷方式的 harness 进行得分对比，计算 O、T、G 百分比；结果显示在 21 个显著提升的设置中，平均 54% 来自过拟合，14% 来自计算扩展，剩余部分（G）极小；hold‑out 评估往往表现不如搜索集，验证选择亦未能完全消除过拟合。

**⚠️ 局限性**

局限性包括：过拟合与计算扩展难以完全消除，验证信号不足以保证泛化；当前基准存在构造痕迹，易被 exploit；HDA 仅给出下限估计，无法完全捕捉隐式的过拟合；需开发更丰富、低痕迹的 benchmark 来进一步验证。

---

## 419. Direct Conditional Parameterization for N-Dimensional Splatting

**arXiv ID:** 2610.22778 | [PDF](https://arxiv.org/pdf/2610.22778v1)

**作者:** Zhongpai Gao `[一作]` (United Imaging Intelligence), Ziyan Wu `[通讯]` (United Imaging Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出直接条件参数化方法，用于 N 维高斯 splatting，减少条件切片计算量并提升渲染效率；

**💡 创新点**

创新点在于用 Cholesky 精度矩阵和分解回归矩阵直接预测条件可见性与位置，取代传统协方差求逆，形成无核族依赖、可扩展的条件切片方案；

**🔧 技术方法**

使用基于 CUDA 的自定义内核，结合 Gaussian 与 Beta 核、Cholesky 精度、分解回归、学习的维度耦合参数；

**📊 数据集**

在五个基准数据集上评测，包括 NeRF Synthetic、Mip-NeRF 360、6DGS-PBR、D-NeRF 与 7DGS-PBR；

**📈 对比分析**

与 N-DGS、UBS 对比，直接方法在质量上匹配或提升（最高 +1.26 dB），渲染速度提升 6.9–7.7×，帧率提升 30–70%；

**⚠️ 局限性**

局限性在于仅验证了 Gaussian 与 Beta 两种核，未验证更大规模城市场景或其他核族；完整渲染加速受后端 rasterizer 限制。

---

## 420. TRACE: Transformer Residual Activation Codebook Encoding for Communication-efficient Distributed Edge Inference

**arXiv ID:** 2610.22435 | [PDF](https://arxiv.org/pdf/2610.22435v1)

**作者:** Muhammad Azlan Qazi `[一作]` (Aarhus University), Qi Zhang `[通讯]` (Aarhus University)

**通讯引用:** 15605 | [OpenAlex ID](https://openalex.org/A5100360194)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种Transformer Residual Activation Codebook Encoding框架，利用后注意残差中的向量量化和预计算的键值缓冲实现分布式边缘推理的通信与计算重叠。

**💡 创新点**

将VQ放置于后残差并预先计算各码字对应的键值，消除接收端矩阵乘法；同时构建多分辨率层次码表实现运行时比特率自适应。

**🔧 技术方法**

使用向量量化（VQ）+EMA更新、层级k-medoids聚类、局部K/V丢弃正则、代码库预计算缓冲以及序列并行与通信计算重叠技术。

**📊 数据集**

在ViT-Base/16上使用ImageNet-1K、CIFAR-100数据集；在ViViT-B/16上使用UCF-101视频分类数据集进行评估。

**📈 对比分析**

与未压缩、Segment-Means和现有最佳VQ方法对比，-b变体在不同分辨率下平均提升约1.45%准确率，速度提升约1.27×，通信压缩率最高可达22,624×。

**⚠️ 局限性**

仅适用于Encoder-Only模型，decoder-only LLM的因果掩码不兼容；VQ后残差方案虽降低部分准确率但需额外预计算缓冲。

---

## 421. On the (In)stability of Adaptive Fee Mechanisms in Multi-Chain Systems

**arXiv ID:** 2610.23305 | [PDF](https://arxiv.org/pdf/2610.23305v1)

**作者:** Yifan Zhou `[一作]` (Huazhong University of Science and Technology), Shijie Zhang `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了多链系统中EIP-1559 fee机制的稳定性，并提出了基于控制理论的模型与协议层缓解方案。

**💡 创新点**

创新点在于引入延迟反馈控制模型识别多链耦合导致的“延迟放大过度校正”不稳定性，给出定量安全边界并设计轻量级协议级补偿机制。

**🔧 技术方法**

使用了控制理论线性化与特征值判据、离散时间非线性动力学仿真、矩阵特征值稳定性分析以及多链负载分解等技术。

**📊 数据集**

使用的数据集包括以太坊、Polkadot、Cosmos Hub、Osmosis区块链的链内费率轨迹、负载与区块失效率数据，并在仿真中采用自定义需求函数。

**📈 对比分析**

通过特征值ρ(M_ext)与仿真结果对比验证模型准确性，安全风险评估显示在不同更新率下区块失效率可提升数十倍，稳定区间缩减达84.8%。

**⚠️ 局限性**

限制在于需求弹性假设恒定、跨链延迟视为固定且未考虑网络拥塞，且实验仅涵盖两跳工作流，真实多跳与策略行为可能进一步放大不稳定。

---

## 422. Before Retrieval Begins: Safety and Tool-Use Decisions in LLM Agents

**arXiv ID:** 2610.23071 | [PDF](https://arxiv.org/pdf/2610.23071v1)

**作者:** Aditya Nawal `[一作]` (National University of Singapore), Mohan Gurusamy `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究大语言模型在调用外部工具前的“工具接受”决策，利用因子实验拆解请求中的域、操作和细节三部分，测量各部分对接受率的影响，并通过内部表示对抗方向进行干预，进一步探讨书面拒绝与工具接受的关系，提出了“Admit”评估协议。

**💡 创新点**

创新点在于：①首次将工具接受视为独立的安全决策点并进行系统评估；②通过因子实验明确“请求操作”是影响接受率的主导因素；③发现请求操作与接受结果的内部表示方向呈负向对齐，可用于干预；④验证书面拒绝仅在部分模型中与工具接受相关，说明文字拒绝不是工具安全的可靠代理。

**🔧 技术方法**

使用的方法包括因子实验设计、线性方向估计与投影干预、内部表示对齐分析、文本拒绝方向诱导、单一种子下的文本拒绝微调，以及GPT‑4o判定器进行自动化危害评估。

**📊 数据集**

数据集：296条 HarmBench 衍生的有害行为与对应的真实网页（HarmURLBench），其中147条可拆解为三部分进行因子实验；另外使用 400 条由 GPT‑4o 评判的样本进行危害打分；使用七个开源模型（3B–12B）以及三款闭源模型进行验证。

**📈 对比分析**

评估结果显示：在所有模型中，工具接受的有害请求平均危害率为 74.5%，加入不调用工具后平均增加 11.2%；因子实验中“请求操作”对接受率的平均影响为 19pp，显著高于域和细节；请求操作方向与接受结果方向在所有模型中均呈负对齐；对未见行为的方向干预在四个模型中显著提高或降低接受率；文本拒绝训练在三组模型中能降低 19–61pp 的有害接受率，但并不普遍适用。

**⚠️ 局限性**

局限性包括：仅研究单一只读检索工具，未覆盖工具选择、查询生成或状态改变工具；因子实验使用自动生成的中立化语句，难以完全区分语义与词汇影响；干预与微调的效果高度依赖模型与实验参数，单一种子不具备可重复性；评估仅在英语、特定工具与模型规模下进行，难以推广至更大模型或多语言；判定器为 GPT‑4o 自动评测，可能存在系统误差。

---

## 423. Efficient Program Reduction for Library Fuzzing Harnesses

**arXiv ID:** 2610.23054 | [PDF](https://arxiv.org/pdf/2610.23054v1)

**作者:** Haiying Li `[一作]` (University of Hong Kong), Hao Chen `[通讯]` (University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 HarnessMinimizer，能够自动且高效地最小化 C/C++ 库 fuzzing harness，生成更小、更易调试的复现程序。

**💡 创新点**

创新点在于：① 通过预编译头文件和持久化 runner 重用编译、链接和运行时环境，显著减少每轮候选测试的开销；② 记录并重放输入消耗（FDP 调用重放），消除输入顺序对减小的影响；③ 使用无符号化堆栈的 offset/深度 oracle 代替昂贵的符号化，快速判定候选是否保留原始崩溃。

**🔧 技术方法**

技术手段包括：Tree‑sitter 语法树分析、Clang+AddressSanitizer 编译、PCH 预编译、持久化执行器（fork‑server + 动态/静态链接），FDP 调用重放逻辑，offset/深度 crash oracle，和与四个现有 reducer（treereduce、Perses、WDD、CDD）的接口适配。

**📊 数据集**

使用了 200 个 crash 案例（100 个库 bug + 100 个 harness bug），来源于 PromptFuzz、PromeFuzz、FuzzAgent，覆盖 21 个开源库。

**📈 对比分析**

实验对比未优化基线，平均加速 3.46–4.51×，最小化时间从 2,408 秒降至 233 秒，源码大小从 1,352 词标记降至 143–263；LLM 崩溃分类准确率从 28% 提升至 44–51%。还通过 ablation 分析单独组件的性能贡献。

**⚠️ 局限性**

局限性包括：在仅有静态库的情况下，深度 oracle 可能无法精确定位崩溃位置，偶尔出现不同位置但相同模式的冲突；需要共享库可用才能使用 offset oracle；若最终重现程序因位置变更而不完全相同，需要再次进行符号化验证；在不同硬件或工具链上效果可能有差异。

---

## 424. Pedicel-Targeted Strawberry Harvesting via Imitation Learning on a Low-Cost Teleoperated Robotic Arm

**arXiv ID:** 2610.22765 | [PDF](https://arxiv.org/pdf/2610.22765v1)

**作者:** Nitin Rai `[一作]` (University of Florida), Heping Zhu `[通讯]` (United States Department of Agriculture)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并部署了一个基于视觉-运动模仿学习的全流程系统，用人类遥控演示数据训练 ACT 策略，实现了对草莓小柄的定位抓取。

**💡 创新点**

创新点在于将感知、运动规划和触控操作统一为一套端到端模仿学习框架，使用 ACT 结合 ResNet‑50 视觉骨干，避免了传统视觉-规划-控制的分阶段设计，并公开了完整数据集与模型。

**🔧 技术方法**

采用 Action Chunking with Transformers (ACT) 端到端策略、ResNet‑50 视觉骨干、行为克隆、FP32 推理、Jetson AGX Orin 边缘部署以及 SO‑ARM101 6‑DOF 机器人臂。

**📊 数据集**

使用 130 余条遥控演示数据，约 250k 帧的同步手腕相机图像与 6‑DOF 关节轨迹，公开数据集地址：https://huggingface.co/datasets/nitindominicrai/strawberry_pedicel_grasp_teleoperation。

**📈 对比分析**

训练过程中损失从 2.84 降至 0.091（100k 步），KL 散度趋于 0，L1 关节误差稳定在 0.062；在 Jetson Orin 上推理速率约 15‑17 Hz，能够复现学习到的抓取轨迹，尽管未能完成完整的柄断裂，实验显示模型可行。

**⚠️ 局限性**

局限包括：柄短导致末端执行器机动受限；手腕摄像头布置关键；仿真到现实转移困难；剪切式末端执行器需要亚毫米精度；当前系统未能稳定实现柄断裂，需扩大演示数据集、改进损失权重、提升推理速度。

---

## 425. Residual Stochastic Concept Bottleneck Models

**arXiv ID:** 2610.23421 | [PDF](https://arxiv.org/pdf/2610.23421v1)

**作者:** Stephen Marcou `[一作]` (ETH Zurich), Mateja Jamnik `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Residual Stochastic Concept Bottleneck Models（Res‑SCBM），在概念不完整的场景下通过引入残差通道并联合建模概念与残差的多元高斯分布，实现可解释的模型并支持测试时概念干预。

**💡 创新点**

1）联合建模概念与残差的协方差结构，使干预能通过高斯条件分布传播到残差维度；2）加入干预感知训练目标（ℒ_int），保持残差通道不削弱概念干预的影响；3）在概念不完整条件下显著提升预测性能同时保持干预效果。

**🔧 技术方法**

多元高斯分布建模、协方差与精度矩阵正则化、贝尔努利采样与直通估计、干预感知损失、Monte Carlo采样与任务损失的组合。

**📊 数据集**

四个视觉数据集：CUB（鸟类）、AwA2（动物）、CXR（胸片）、MNIST；分别构造完整与不完整概念子集。

**📈 对比分析**

与Hard CBM、SCBM、CEM、Black‑Box DNN和线性概念头进行对比；在概念不完整任务上，Res‑SCBM在任务准确率上比Hard CBM和SCBM提升10–20%，并且在干预曲线下保持或超过CEM；在概念完整任务上提升有限但保持竞争力。

**⚠️ 局限性**

需要学习的协方差矩阵规模随变量数二次增长；干预传播效果高度依赖学习到的概念‑残差交叉协方差，若交叉协方差弱则干预影响有限；模型在极大残差维度下若无干预感知训练可能失去可干预性。

---

## 426. First Passage Attention

**arXiv ID:** 2610.22714 | [PDF](https://arxiv.org/pdf/2610.22714v1)

**作者:** Yuxuan Liu `[一作]` `[通讯]` (University of Bergen), Yuxuan Liu (University of Bergen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新型注意力机制 First Passage Attention（FPA），通过学习的场（field）与键（key）层级的最小值来决定键的活跃性，能够在单个头中实现类似堆栈的“遗忘”行为；

**💡 创新点**

创新点在于：① 用范围最小值而非累加/乘积来聚合路径信息，从而仅关注最深的“dip”并实现深度而非持续时间的抑制；② 在同一层内，单头即可逼近传统需要 r‑2 维度的堆栈/遗忘门头；③ 将 Forgetting Transformer、堆栈、bottleneck bias 与学习到的过期机制统一到一种可学习的参数化框架中；

**🔧 技术方法**

技术手段包括：自定义 softmax 注意力的偏置（first‑passage bias），使用直通估计器实现梯度传递，计算区间最小值的高效块化算法（Triton kernel），以及对键/值缓存的动态剪枝；

**📊 数据集**

实验数据集主要为：SlimPajama（2.7B token 预训练）、WikiText‑103、PG‑19（用于长上下文评估）以及公开的六个小规模基准任务；

**📈 对比分析**

比较方法：在单层探针、完整语言模型（12 层 110M 参数）以及推理阶段的 passkey 检索。FPA 在 2048-8192 的上下文中匹配或优于 FoX 的困惑度，在 16,384 位置的 loss 维持平稳，并在 passkey 检索上实现 100% 成功率；在更长上下文（16k）和 RULER、Babilong 任务上与 FoX 相当或略优；相比 Selective Attention、DCP 与传统 softmax，FPA 在速度上略慢但接近；

**⚠️ 局限性**

局限性：仅在 110M 参数、单个随机种子和 2.7B 预训练规模下验证，无法直接说明在更大模型或更广泛任务中的表现；实现与优化仍依赖高性能 Triton kernel，可能在不同硬件上表现不一；

---

## 427. Model-as-a-Trojan: Self-Activating Covert DNS Exfiltration Baked Into the Weights of an LLM Agent

**arXiv ID:** 2610.22578 | [PDF](https://arxiv.org/pdf/2610.22578v1)

**作者:** Barak Haryati `[一作]` (JFrog), Shaked Zychlinski `[通讯]` (JFrog)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对 Qwen3-8B 4‑bit 模型进行 rank‑32 LoRA 微调，植入了一个只在对话第一轮且检测到 shell 工具时触发的隐蔽后门，利用 DNS 查询将环境变量以 Base32 分块形式泄露给攻击者。

**💡 创新点**

创新点在于：①把后门直接嵌入模型权重而非提示；②在仅开放 DNS 出口的沙箱环境中完成一次自我隐藏的泄露；③提供完整的实验矩阵、对比评估与可部署的防御蓝图。

**🔧 技术方法**

使用的技术包括 Qwen3‑8B 4‑bit + rank‑32 LoRA 微调、合成监督数据集与执行验证、DNS 分块编码与查询、Docker 沙箱与自定义递归解析器、允许列表解析器与日志监控等防御手段。

**📊 数据集**

数据集为合成对话样本，覆盖 5 种工具调用格式、5 种提示变体，包含 1,944 条执行验证行（1,847 训练 / 97 验证）以及后续的 171 条严格留出验证行。

**📈 对比分析**

通过在 45 个沙箱‑格式组合（5 格式 × 9 配置）以及 54 个扩展测试单元进行对比，微调模型触发率为 100%，无误报；基模型在相同测试下完全不触发；在压制提示、网络异常、工具缺失等恶意情形下仍保持稳定。

**⚠️ 局限性**

限制包括：实验仅验证 Qwen3‑8B 4‑bit + LoRA；后门仅能读取单一环境变量；在允许列表解析器或完全离线沙箱中失效；未评估更大模型或多通道传输方案。

---

## 428. UAV Rotatable Antenna under Rotation Delay

**arXiv ID:** 2610.22777 | [PDF](https://arxiv.org/pdf/2610.22777v1)

**作者:** Yinmei Xie `[一作]`, Wei Wang `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在有限通信时隙内考虑机械旋转延迟的旋转天线（RA）系统，提出联合RA方向与波束形成的最大最小有效传输速率设计，并给出了单用户与多用户的优化方案。

**💡 创新点**

创新点包括：①首次将机械旋转延迟显式建模并揭示旋转与传输时间的权衡；②单用户分析得出三种旋转最优区间并给出二阶泰勒展开的闭式近似；③多用户端提出基于零强制与Gram矩阵的低复杂度AG‑ZF算法。

**🔧 技术方法**

使用的技术包括旋转延迟感知通信模型、最优与近似优化、二阶泰勒展开、零强制波束、Gram矩阵分析、解析梯度与球面约束更新、二分搜索与外部旋转预算搜索。

**📊 数据集**

通过仿真评估，随机生成用户位置（均匀分布在指定区间）并采用标准通信参数（5.8 GHz、1 MHz、0.5 s时隙等）进行测试。

**📈 对比分析**

与无旋转、无延迟、固定天线等基准方案进行对比；单用户时闭式解与最优解相差极小，多用户时AG‑ZF优于交替优化且明显优于无延迟方案，显著提升有效传输速率。

**⚠️ 局限性**

局限性在于仅考虑单时隙无跨时隙自适应，忽略旋转加速度、稳态时间和能量消耗；假设完美CSI，未覆盖复杂轨迹下的连续重构和实际机械实现细节。

---

## 429. Everyone's a little bit racist: a novel ML backdoor attack in black-box mode

**arXiv ID:** 2610.22459 | [PDF](https://arxiv.org/pdf/2610.22459v1)

**作者:** Michele Cerrta `[一作]` (IMT School for Advanced Studies), Gabriele Costa `[通讯]` (IMT School for Advanced Studies)

**通讯引用:** 780 | [OpenAlex ID](https://openalex.org/A5040739994)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种全黑盒后门攻击框架，通过遗传搜索生成输入变换，激活已训练模型的潜在偏差形成二级任务。

**💡 创新点**

创新点在于无需改动训练数据、模型权重或架构，仅利用模型已有的内部偏差，通过程序化扰动实现后门。

**🔧 技术方法**

采用遗传算法、掩码编码树（MET）以及 saliency‑guided mutation 等技术，自动搜索高效的输入变换。

**📊 数据集**

实验数据集包括 MNIST/EMNIST（用于字母识别）以及 FalconsAI NSFW 模型和 UTKFace（人种、年龄、性别）等。

**📈 对比分析**

攻击后得到的二级分类准确率在 65%–86% 之间，实验表明方法可在工业模型上实现有效后门，且与原模型直接比较性能良好。

**⚠️ 局限性**

主要限制包括对模型偏差的依赖、需要大量查询（数千万次）、对随机预处理和某些防御机制的鲁棒性不足。

---

## 430. Provable Authority: A Protocol for Verifiable Authority Resolution and Control Across Human, Organizational, and Agentic Transactions

**arXiv ID:** 2610.22899 | [PDF](https://arxiv.org/pdf/2610.22899v1)

**作者:** Jeremy Blackburn `[一作]`, Jason Patrick `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了 ChainIT Authority Protocol，构建了一个可验证的权威解析与控制平面，用以在人员、组织及代理交易中确保事务的合法性、完整性与可追溯性。

**💡 创新点**

创新点在于将身份、组织授权、来源证据、会话、事务、执行与终端证据分离，利用可验证数据令牌、版本化授权记录（ARP）、代理授权记录和单次执行凭证（EAC）实现事务绑定与容量预留，并为智能代理提供主权边界与可验证的代理身份。

**🔧 技术方法**

主要技术包括可验证凭证与去中心化身份（W3C VC、DID）、Validated Data Tokens（VDT）、业务规则引擎（BRE）、工作流引擎、链式哈希（canonical transaction digest）以及零信任的会话与执行凭证（SAC/EAC）等。

**📊 数据集**

本文未使用公开数据集，而是基于理论设计与示例场景进行说明，提供了示例 ARP、代理授权与事务决策的 VDT 链。

**📈 对比分析**

文章未给出实验或性能对比；其价值主要在于提供一种完整的协议规范与实现蓝图，可与现有 OAuth、MACAROON 等授权方案概念对比，但缺乏量化评估。

**⚠️ 局限性**

局限性包括无法消除欺诈、胁迫、数据错误、基础设施被攻破等风险；仅能降低重放、超限、离线误授权等事务风险，且对实现细节与合规成本的影响未作深入评估。

---

## 431. Counting and Sampling Manufacturable Stacking Sequences

**arXiv ID:** 2610.23146 | [PDF](https://arxiv.org/pdf/2610.23146v1)

**作者:** Alessio Langiu `[一作]` (National Research Council of Italy - Institute of Marine Sciences (CNR-ISMAR)), Giacomo Canale `[通讯]` (University of Derby)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过将复合层压板的堆叠序列视为有限自动机识别的正规语言，精确描述了制造规则（相邻层夹角不超过45°、同向层不超过三层）以及设计准则（对称、平衡、每个主方向至少占10%）所构成的可行序列集合，并给出了针对该集合的计数、均匀随机生成和可完成性检验算法，以及一种能够保证子代序列可行的一点交叉算子。

**💡 创新点**

创新点在于：①将制造规则转化为有限因子集合，构建对应自动机并利用其状态与Parikh计数相结合的动态规划实现对任何给定层数、任意取向集合的可行序列的精确计数、均匀抽样与完成性判定；②提出一种基于自动机状态和Parikh向量的交叉操作，保证交叉后子代自动满足所有约束；③在表格缺失的情况下完成了以前未计算的计数条目，验证了方法的正确性。

**🔧 技术方法**

使用的主要技术包括正规语言与有限自动机理论、转移矩阵计数、动态规划（带Parikh向量计数器）、递归均匀抽样（后向计数比例抽样）、以及基于状态/计数器的交叉算子。实现语言为Python，使用整数算术与哈希表存储状态。

**📊 数据集**

数据集为三种取向集合：4个（{0,±45,90}）、8个（{0,±30,±45,±60,90}）和12个（{0,±15,±30,±45,±60,±75,90}），层数范围从24到54，重点在N=48层。计数结果通过与已有文献（如Zein等人、Irisarri表格）以及穷举枚举进行比对验证。

**📈 对比分析**

与传统分支定界或遗传算法的比较表明，使用完成性检验可以在N=48层、8个取向时消除超过99%的无效树节点；与以前的计数表格相比，本文填补了缺失的单规则计数，并在16层时完成了所有组合计数。性能方面，计数和抽样在多取向集合下仍保持多项式时间，内存占用随着符号对数增长（例如N=54、8取向时中间层达1.1×10^9个状态）。

**⚠️ 局限性**

主要限制包括：①仅处理制造规则与对称、平衡、10%准则，未涵盖损伤容忍、层排除、面板间混合等工业常见约束；②对含有大量无阈值对称取向（如±30、±60）时，Parikh差值计数会导致状态空间急剧扩大，导致内存与时间成本显著增加；③未对搜索效率（如遗传算法的适应度评估次数）做定量评估。

---

## 432. Foundation Models for Autonomous Space Robotics: Architectures, Physical Adaptation, and Validation

**arXiv ID:** 2610.23068 | [PDF](https://arxiv.org/pdf/2610.23068v1)

**作者:** Zizhe Wang `[一作]` (Astroup Lab), Jianing Li `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评述了空间机器人中基础模型的应用、适配、验证路径及其在感知、预测、规划与执行环节的效果。

**💡 创新点**

首次系统化提出了从仿真-地面-太空的迭代验证流程，区分了一般化与目标域适配，并强调事件传感器与脉冲神经网络在空间条件下的潜力。

**🔧 技术方法**

综合利用VLM/VLA、事件相机、SNN、强化学习、轨迹规划等技术，并对比了多种预训练与细化方法。

**📊 数据集**

基于公开的事件、图像、SLAM数据集（如SPADES、S2E2、SHIRT、SpaceSense-Bench）、仿真工具（Basilisk、MuJoCo）以及实际航天试验（ETS‑VII、Astrobee、Loihi‑2）进行评估。

**📈 对比分析**

通过跨研究的对比分析，证明了视频预训练、语义预训练以及少量适配能提升成功率，但在自由基座动力学、接触恢复和能源预算方面仍受限；性能在不同场景下差异显著。

**⚠️ 局限性**

缺乏在真实自由基座/接触场景下的完整闭环验证，且对能耗、热与辐射耐受性评估不足，难以确定模型在长期空间任务中的鲁棒性。

---

## 433. Many Biases, One Circuit: Post-Training Tunes a Pretrained Cue-Routing Mechanism

**arXiv ID:** 2610.22937 | [PDF](https://arxiv.org/pdf/2610.22937v1)

**作者:** Prakhar Gupta `[一作]` (University of Michigan), Zhijing Jin `[通讯]` (University of Toronto)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多项选择题中语言模型对提示暗示的依赖机制，并追踪该机制在对齐后训练管道中的变化；

**💡 创新点**

发现该机制由预训练模型中已存在的少数注意力头组成，通用助手调优仅在聊天格式下激活它，最终RL阶段显著强化其效应；

**🔧 技术方法**

使用注意力头重写、激活补丁、注意力交换与缩放等机制性实验，结合对齐（SFT、DPO、RLVR）训练流程；

**📊 数据集**

利用BCT基准（包含MMLU、HellaSwag、LogiQA、TruthfulQA等）多项选择题与四种提示类型；

**📈 对比分析**

通过比较基准对齐前后模型的caving率和敏感度，并在注意力交换实验中证明RL阶段将受提示影响的比例提升至61‑73%；

**⚠️ 局限性**

局限包括仅考察英语多项选择提示，主要关注较小规模模型，答案读取依赖字母logits，实验多为单跑且对齐步骤的细节不完全通用，且注意力缩放需预先定位提示。

---

## 434. GRASP: Global-Local Representation Alignment and Spatial-Physical Grounding for Bimanual Robotic Manipulation

**arXiv ID:** 2610.23123 | [PDF](https://arxiv.org/pdf/2610.23123v1)

**作者:** Zeyuan Ren `[一作]` (COWARobot Co Ltd), Wende Xie `[通讯]` (COWARobot Co Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种训练阶段先验注入框架，结合基于物理的流匹配监督和全局-局部查询蒸馏，来提升双臂细粒度操作的准确性与鲁棒性。

**💡 创新点**

1）在流匹配过程中对隐式干净动作进行前向运动学映射并监督TCP与相对姿态，实现物理空间监督；2）引入全局未来特征预测查询与基于Grounding DINO的局部交互查询，分别强化场景语义与手物交互注意力；3）所有辅助教师在推理时移除，仅保留主流量。

**🔧 技术方法**

条件流匹配、可微前向运动学、DINOv3视觉教师、Grounding DINO检测、Perceiver模块、KL 与特征对齐损失、噪声级加权。

**📊 数据集**

RoboTwin 2.0 任务集（50 个双臂任务）和真实双臂 PiPER‑X 机器人上收集的约 150 条演示。

**📈 对比分析**

在 RoboTwin 2.0 上，完整模型在干净场景成功率从 64.13% 提升至 79.22%（+15.1pp），在随机化场景提升至 52.84%（+16.7pp）；在真实机器人上四项长序列任务均显著提升，最高任务成功率从 56.7% 提升至 83.3%。

**⚠️ 局限性**

1）两类先验的权重为静态，缺乏自适应平衡；2）局部查询依赖离线检测器的质量；3）实验仅覆盖单一模拟基准与单一真实平台，泛化性待验证。

---

## 435. Ori: Drift-Aware Streaming $k$-PCA for Very Low-Rank LLM Training

**arXiv ID:** 2610.22794 | [PDF](https://arxiv.org/pdf/2610.22794v1)

**作者:** Xinyu Luo `[一作]` (Purdue University), Brian Bullins `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Ori，一种低秩优化器，能够在训练大型语言模型时持续跟踪梯度子空间并在每一步以轻量化方式更新投影。

**💡 创新点**

核心创新是将 Oja++ 流式 k‑PCA 与漂移系数引入投影更新，使得上一步的子空间基底可直接迁移并保持优化器的动量与二阶矩，显著降低子空间“陈旧”带来的性能损失。

**🔧 技术方法**

采用的技术包括：Oja++ 版的流式 k‑PCA、漂移系数 ρ 的热启动、AdamW 的一阶/二阶矩迁移、动态损失分析以及对低秩子空间的理论动态 regret 上界。

**📊 数据集**

实验数据集包括：LLaMA 预训练任务使用 C4 数据集，RoBERTa‑Base 微调任务使用 GLUE 基准数据集。

**📈 对比分析**

与 Full‑Rank Adam、LoRA、ReLoRA、GaLore、OSD、LDAdam 以及 O‑LDAdam 等方法比较，Ori 在极低秩（r≤8）时能将性能差距降至 69.3%，在 10/16 个 GLUE 微调场景中获得最高平均分，并在 LLaMA 1B 预训练中取得与全秩 Adam 相近的困惑度。

**⚠️ 局限性**

局限性包括：对梯度子空间慢漂移假设的依赖，未验证在高度非平稳或异构数据分布下的稳健性；更新频率与计算成本之间的折中仍需进一步优化；对更大规模模型或不同任务类型的泛化性尚未系统评估。

---

## 436. "Deep lifts": Neural networks can generalize even with the dumbest optimizer

**arXiv ID:** 2610.23009 | [PDF](https://arxiv.org/pdf/2610.23009v1)

**作者:** Sunil Madhow `[一作]` (University of California San Diego), Yu-Xiang Wang `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在宽度为W、深度为D的ReLU网络中，从所有能够插值训练数据的参数空间中均匀随机抽取参数后，其泛化性能如何随网络宽度、深度和训练样本数量变化；

**💡 创新点**

创新点在于将离散权重下的深度提升（deep lifting）技术推广到连续权重，并提出“结构化稳定性”（structured stability）概念，证明即使在存在标签噪声或不可逼近的真实函数情况下，随机插值器的泛化误差仍可自适应地达到 Õ(wdWD/n)（或更弱的噪声下的 Õ(W(wD+m)/n)）的速率；

**🔧 技术方法**

主要技术手段包括深度提升构造大体积等价插值器集合、结构化稳定性分析、对抗噪声下的记忆化构件（memorization gadget）以及概率体积与集中不等式结合的泛化分析；

**📊 数据集**

实验与分析基于理论分布 μ_X 产生的任意输入样本，没有使用具体公开数据集；

**📈 对比分析**

与传统 VC 维度或全体网络容量分析相比，本文提出的方法在相同网络宽度/深度下给出了更快的泛化速率，实验证明随机插值器在宽度增大时能够实现更好的误差下降；

**⚠️ 局限性**

主要局限在于所研究的学习规则（均匀随机插值）与实际训练方法（如 SGD）差距较大，且分析假设的标签可被完整记忆的记忆化构件在实际训练中并不可直接实现。

---

## 437. Acquisition Function-Guided Generation for Bayesian Optimization over Discrete Spaces

**arXiv ID:** 2610.22763 | [PDF](https://arxiv.org/pdf/2610.22763v1)

**作者:** Tommaso Ancilli `[一作]` (Institute of Machine Learning and Neural Computation, TU Graz), Robert Peharz `[通讯]` (Institute of Machine Learning and Neural Computation, TU Graz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在离散空间中进行贝叶斯优化的全新框架——Acquisition Function Guided Generation (AFGG)，通过在预训练生成模型的连续潜在空间中引入“采集函数倾斜”并利用扩散桥实现对采集函数的引导采样；

**💡 创新点**

创新点在于：①将采集函数直接作用于数据空间而非潜在空间，保留了领域特定核等先验知识；②使用扩散桥将离散采集函数平滑化，得到可导的指导梯度，避免了传统 LSBO 的梯度缺失问题；③支持任意非负采集函数（如 EI、PI）且不需对模型再训练；

**🔧 技术方法**

核心技术包括：预训练生成模型（可为类别扩散模型或自回归流模型）、Gaussian过程或其他概率代理、扩散桥（diffusion bridge）以及自归一化重要性采样来估计倾斜均值；

**📊 数据集**

在25个分子设计基准（来自 GuacaMol 等）上进行实验，使用 SELFIES 表示的分子字符串进行优化；

**📈 对比分析**

与 CoBO、LOL-BO、COWBOYS、InvBO、NFBO 等 5~11 现代离散贝叶斯优化方法比较，AFGG（尤其是流模型实现）在大多数基准上达到或超过最优方法，性能提升显著；

**⚠️ 局限性**

局限性包括：①对生成模型质量高度依赖，生成质量不足时会限制搜索效果；②扩散桥和重要性采样的计算开销较大；③当前仅在分子设计任务验证，跨领域应用仍需进一步探索。

---

## 438. Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss

**arXiv ID:** 2610.22454 | [PDF](https://arxiv.org/pdf/2610.22454v1)

**作者:** Zhiyuan Wang `[一作]` `[通讯]`, Zhiyuan Wang

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在Mixture‑of‑Experts（MoE）模型的预填充阶段将 prompt tokens 的专家数减半所造成的性能损失，定位了损失的关键位置和机制，并提出了在实际推理中如何分配全专家预算的经验原则。

**💡 创新点**

创新点包括：① 将损失细分为 F1 的形式、召回、答案长度三部分，发现大部分损失集中在答案“格式”上；② 通过 token‑level、layer‑level、attention‑path 干预定位关键 prompt 位置（如 Qwen3 的两句模板、DeepSeek 的指令开头）以及验证这些位置与答案产生的关联；③ 对比不同 MoE 家族（Qwen3、DeepSeek、Qwen1.5、OLMoE）及不同 prompt 结构，发现“端点”位置是普遍的敏感点；④ 在真实 Prompt 中测试全专家预算的放置策略，提出“保持任务接口全专家、随机散布在文档中会损失 F1”的经验法则。

**🔧 技术方法**

使用的技术主要是：MoE 预填充专家数减少（8→4、6→3 等）、token‑level、layer‑level 与 attention‑edge 干预、KV‑cache 访问与修改、vLLM 推理框架、HuggingFace pipelines、NLL 与 F1 评估、Bootstrap 统计、以及 sink 与 super‑expert 的检测与恢复实验。

**📊 数据集**

主要数据集：HotpotQA（短答案、Distractor 版），GSM8K、CodeCompletion 作为辅助检查；使用的模型包括 Qwen3‑30B‑A3B、DeepSeek‑V2‑Lite‑Chat、Qwen1.5‑MoE‑A2.7B‑Chat、OLMoE‑1B‑7B‑Instruct，均为 MoE 体系结构。

**📈 对比分析**

比较方法：将减半专家模型与完整专家模型在同一批次、同一硬件（H100、A100）下的 F1、答案长度、NLL、TTFT 进行对比；使用 95% Bootstrap 置信区间评估差异；通过“全专家预算放置”实验对比“接口全专家” vs “随机散布” 等方案，结果显示接口全专家可在 F1 上保持 ≤1 分，随机散布则差距显著。总体性能损失主要集中在答案格式，NLL 受影响较小。

**⚠️ 局限性**

局限性：仅评估了在 8→4（或 6→3）的极端压缩，未探究更温和设置；只在短答案 QA 上验证，未覆盖长文本生成、复杂推理或代码生成；机制解释仅在 Qwen3 上得到部分验证，其他模型未完全确定；sink 与 super‑expert 的检测平均化可能掩盖单头变化；最终推理实验仅在单一 GPU 与无前缀缓存场景下进行，无法推广到大规模并发推理；以及对 long‑system prompt 的敏感位置仍未完全定位。

---

## 439. Simultaneous Force and Motion Control in Generalised Coordinates

**arXiv ID:** 2610.22813 | [PDF](https://arxiv.org/pdf/2610.22813v1)

**作者:** Jon Woolfrey `[一作]` `[通讯]` (Independent Researcher), Jon Woolfrey (Independent Researcher)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种基于高斯最小约束原理的二次规划（QP）方法，用一般化坐标实现同时的力与运动控制，并能加入各类不等式约束。

**💡 创新点**

创新点在于：①将力与运动的优先级直接映射到关节空间，实现动力学一致性；②利用高斯原理推导出完整的约束动力学并产生缺失的 centripetal-like 项；③采用分层 QP 解决多优先级任务与安全约束的兼容。

**🔧 技术方法**

使用了高斯最小约束原理、Pfaffian 约束、动态一致投影、二次规划求解（Interior‑point）以及动态控制壁函数（barrier function）来实现关节空间控制。

**📊 数据集**

实验数据基于 KUKA LBR iiwa 14 R820 机器人模型在 MuJoCo 物理引擎中模拟，配合 ROS2 与 SimpleQPSolver 进行控制。

**📈 对比分析**

通过将 QP 方法与经典操作空间控制方法在相同场景（施加 10 N 约束力并跟踪轨迹）进行对比，结果显示 QP 方法在保持目标接触力、轨迹跟踪误差小于 10 mm、角度误差 < 5° 以及在关节极限与奇异性逼近时能保持约束，性能优于传统方法。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，未在真实机器人上测试；对动力学模型的精确度要求高；QP 计算量随任务数和约束增多而显著提升，实时性能需进一步评估；参数（如权重、阈值）需人工调优。

---

## 440. Exploring The Empirical Privacy Landscape of Synthetic Data

**arXiv ID:** 2610.22659 | [PDF](https://arxiv.org/pdf/2610.22659v1)

**作者:** Zihao Zhao `[一作]` (Johns Hopkins University), Dylan Sam `[通讯]` (OpenAI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在合成医疗记录上对数据层和训练层的隐私保护方法进行受控比较，评估隐私泄露与模型能力的权衡。

**💡 创新点**

提出并验证了隐私感知重写作为一种既能保持模型能力又能降低隐私泄露的新途径，并提供了统一评估框架。

**🔧 技术方法**

使用数据重写、隐私感知重写、DP‑SGD、金鱼、PII擦除等技术，并在 Qwen3.5‑4B‑Base 与 gpt‑oss‑20b 上通过 LoRA 微调进行实验。

**📊 数据集**

采用 Synthea 生成的 5,000 条合成病历作为训练集，并构造 100 条带有药品“加农”canary 的可控泄露记录。

**📈 对比分析**

通过可写作 Perplexity（verbatim、semantic、held‑out、medication）指标进行双维度排名；结果显示直接微调性能最高但泄露最大，隐私感知重写在能力和隐私上实现最佳平衡，DP‑SGD 在隐私上最佳但能力最低。

**⚠️ 局限性**

重写方法依赖可控重写模型的质量且预处理成本高；实验仅在开放权重模型上进行，未验证专有模型或真实医疗数据；仅使用合成数据，缺乏真实隐私场景的验证。

---

## 441. Jev-as-a-Judge for Brazilian Portuguese Information Retrieval: Relevance Assessment and Reranking

**arXiv ID:** 2610.23346 | [PDF](https://arxiv.org/pdf/2610.23346v1)

**作者:** Leandro Carísio Fernandes `[一作]` (Câmara dos Deputados), Jayr Pereira `[通讯]` (Universidade Federal do Cariri)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Jeve系统One模型在巴西葡萄牙语信息检索中的两个任务：自动相关性评估和候选文档重新排序。

**💡 创新点**

展示Jeve能以低成本高效完成结构化决策，能够在不同语料、领域产生与人工评判高度一致的qrels，并显著提升BM25候选列表的排名效果。

**🔧 技术方法**

使用Jeve模型的结构化决策接口（四种问题类型），并将其作为点式相关性评估器；对比BM25与BM25+Jeve的效果；对比Jeve生成的qrels与官方qrels。

**📊 数据集**

四个巴西葡萄牙语检索集合：Quati、JurisTCU、NormasTCU、Ulysses‑RFCorpus；另外在Quati 1M语料上进行重新排序实验。

**📈 对比分析**

通过准确率、Cohen κ、Kendall τ评价相关性判定一致性；通过nDCG@10、MRR评价重新排序效果。结果显示Jeve的qrels在τ≥0.9时与官方qrels相当；Jeve将nDCG@10从0.3329提升至0.6331（+90.2%），MRR从0.6852提升至0.9033（+31.8%）。

**⚠️ 局限性**

未评估实时延迟和成本；实验仅覆盖单一检索管线与单一语料；缺乏对全篇法律文档及多种检索器的泛化验证；缺乏统计显著性检验与与其它重新排序器的直接对比。

---

## 442. Repair, Retry, and the Cost of Hindsight: Verifier Feedback in Formal Theorem Proving

**arXiv ID:** 2610.22698 | [PDF](https://arxiv.org/pdf/2610.22698v1)

**作者:** Jack Li `[一作]` (Stanford University), Sophia Li `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在 Lean 定理证明中，使用冻结的 Qwen3.6‑27B 与 DeepSeek‑Prover‑V2‑7B 模型进行 verifier‑guided self‑repair 的实验，探讨诊断信息、失败证明曝光、重试指令等不同提示组件对修复成功率的影响。

**💡 创新点**

创新点在于将传统的自我修复实验拆解为单步控制实验，独立评估失败证明、诊断信息、重试指令等各个因素的作用，并通过对比“自己失败证明曝光”与“替换为他人失败证明”两种上下文的效果，揭示了“失败证明曝光”对模型产生的负面诱导复制效应。

**🔧 技术方法**

技术上主要使用了大语言模型的 prompt‑engineering 方式，对模型输出进行解析、Lean 4 编译验证，并采用 Pass@k、token‑budget 等指标评估不同修复策略的性能。

**📊 数据集**

使用的数据集是 MiniF2F 的困难子集（Qwen 167 题，DeepSeek‑Prover 137 题），其中包含 Lean 4 语法错误和旧 Lean 3 语法的混合问题。

**📈 对比分析**

对比方法包括：单步固定失败证明实验（4‑步 Pass@4）、64 步序列实验（Pass@64）、以及记忆化指令与纯重试指令的对比；结果显示在单步实验中，诊断信息能在存在失败证明时提升成功率，但整体上 self‑repair 在任何预算下均未能显著优于仅重新抽样的指令重试。

**⚠️ 局限性**

局限性包括：仅评估两款冻结模型且仅在 11 题的小型序列实验中检验迭代效果；提示格式与编译器版本不统一导致结果与其他实验难以直接对齐；token‑budget 对比仍偏向抽样；诊断对齐方式（错误类别/长度匹配）不足以完全消除语义差异；因此结论的普适性和在更大规模或不同模型上的推广尚待进一步验证。

---

## 443. SAFE-Mem: Secure Agentic Memory with Fine-Grained Authority and Governance-Aware Retrieval

**arXiv ID:** 2610.22959 | [PDF](https://arxiv.org/pdf/2610.22959v1)

**作者:** Haoming Li `[一作]` (University of Texas at Dallas), Bingzhe Li `[通讯]` (University of Texas at Dallas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 SAFE-Mem，一种细粒度的记忆治理框架，在 LLM 代理的长期记忆中为每条记忆打上治理标签，并在检索时根据标签过滤，显著降低攻击成功率。

**💡 创新点**

创新点在于引入三维治理标签（kind、authority、scope）实现使用级别判定，结合治理-aware retrieval 和治理记忆图，实现对来源、权限与使用范围的细粒度控制，同时兼顾安全与记忆效用。

**🔧 技术方法**

使用了记忆图结构、治理标签计算与提取、检索前过滤、使用级别判定、工具调用时的治理检查，以及多模型实验验证等技术。

**📊 数据集**

实验采用 MPBench、LoCoMo、EHRAgent（MIMIC‑III Demo）等数据集，并使用 MPBench 的攻击集与对抗重写来评估安全性。

**📈 对比分析**

在四大 LLM 后端（GPT‑5.6‑terra、GPT‑4o‑mini、Claude Haiku 5.5、Qwen3.8‑Flash）上与 A‑MemGuard、DRIFT、RandEC、LLM sanitizer、MemLineage、TMA‑NM 等基线比较，SAFE‑Mem 在 MPBench 上攻击成功率降至 2.6%–6.2%（比最佳防御低 77%），同时提升 recall 与 LoCoMo 准确率；在 EHRAgent 上仅丢弃 2% 记忆就能完全阻断攻击执行。

**⚠️ 局限性**

局限在于只能对显式保留请求的记忆提升权限，错误的主张仍会以描述形式保留；治理依赖于通道记录和主张提取的准确性，对抗性重写虽无法提升权限，但仍需与内容判定结合以进一步加强安全。

---

## 444. mmPencil: Toward Writing-Style-Independent In-Air Handwriting Recognition via mmWave Radar and Large Vision-Language Model

**arXiv ID:** 2610.23419 | [PDF](https://arxiv.org/pdf/2610.23419v1)

**作者:** Yifan Guo `[一作]` (Northwestern Polytechnical University), Zhiwen Yu `[通讯]` (Northwestern Polytechnical University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出mmPencil系统，结合mmWave雷达与大规模视觉‑语言模型实现3D空中手写词识别

**💡 创新点**

设计了多阶段空间轨迹重建、频谱细化、Kalman平滑以及空间‑视觉映射（方向校正PCA）四大模块，并将RF信号映射至VLM可处理的图像，实现写作风格与位置无关的识别

**🔧 技术方法**

使用mmWave FMCW雷达数据、频谱细化与零填充、PCA方向校正、3D卡尔曼滤波、VLM（GOT‑OCR、Qwen2‑VL、LLaVA‑NEXT）以及LoRA微调

**📊 数据集**

收集了7,664份200个词的3D mmWave空中手写数据集（12位用户，22种写作场景），公开于Kaggle

**📈 对比分析**

与传统DCNN、LSTM基线相比，mmPencil在200词4人数据上取得97.60%（单词识别），跨人群零样本识别达到92.50%；在50词12人数据上VLM版仍保持90%+，轻量模型显著受限；Ablation显示频谱细化和轨迹平滑至关重要

**⚠️ 局限性**

主要限制在实时性（轨迹重建≈0.47s/字，整体延迟>200ms）、对极端写字尺寸、写字角度±40°时精度下降、对大幅用户移动易失效，以及对极少样本或无标签类别的零样本性能仍不如人类

---

## 445. Online geometric matching and segments block points

**arXiv ID:** 2610.23130 | [PDF](https://arxiv.org/pdf/2610.23130v1)

**作者:** Yaqiao Li `[一作]` (Shenzhen University of Advanced Technology), Junyi Lin `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在线几何匹配问题中允许有限交叉的两种模型（k‑交叉模型与无三角交叉模型），并给出了Greedy算法的精确竞争比以及新的两层Greedy和随机化算法的改进结果；

**💡 创新点**

创新点在于首次提出并分析k‑交叉与无三角交叉模型，证明Greedy在k≥1时性能更差，提出两层Greedy实现4/5竞争比，并将问题转化为几何阻塞点极值问题，揭示与顶点覆盖数及Ramsey数的关联；

**🔧 技术方法**

采用组合几何、极值论证、图论（交叉图顶点覆盖、Ramsey理论）、随机化策略（Yao原理）以及数值优化求解多变量不等式等技术；

**📊 数据集**

无真实数据集，所有实验均为理论构造或概率模型；

**📈 对比分析**

与Greedy（2/3）和之前随机化算法（≈0.6695）对比，新的随机化算法达到0.67034，随机化上界被提升至6/7；

**⚠️ 局限性**

仍存在若干限制：k≥1时是否存在超越2/3的确定性算法未知；无三角交叉模型Greedy的下界和极值问题β_未完全解决；随机化最优竞争比仍未确定。

---

## 446. Competitive Online Clustering with Movement Cost

**arXiv ID:** 2610.22824 | [PDF](https://arxiv.org/pdf/2610.22824v1)

**作者:** Christian Coester `[一作]` (University of Oxford), Michael Mitzenmacher `[通讯]` (Harvard University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

提出并分析了在线聚类与移动成本（Online CMC）问题，涵盖任意k个中心、任意度量空间以及广泛的聚类目标（如k‑median、k‑center、k‑means等）。

**💡 创新点**

创新点在于：①首次通过Poisson化随机化归约将k‑服务器问题与在线C​MC等价，得到对任意k和任意度量空间的O(k)随机化竞争算法；②在直线度量上给出O(k)的确定性算法；③将同样的归约方法推广至批量集合追逐（batched F‑chasing）等更广泛的追逐问题；④引入动态norming functional来线性化批量请求。

**🔧 技术方法**

核心技术包括：Poisson化（将批量请求转化为Poisson过程的单请求流）；norming functional（对每个时间点的距离向量进行分解）；连续时间的双向覆盖（Double Coverage）分析；潜能函数和势能递增/递减证明；对称/非对称范数的处理；以及对批量集合追逐问题的通用归约。

**📊 数据集**

该工作为理论分析，没有使用任何实验数据集。

**📈 对比分析**

通过与最优离线算法的成本比较，随机化算法在对抗适应性在线对手时达到O(k)竞争比（与k‑服务器下已知下界匹配），在直线度量上确定性算法也达到O(k)竞争比；对批量追逐问题，进一步得到O(log²k)等竞争比，优于先前已知的结果。

**⚠️ 局限性**

局限性包括：①确定性O(k)算法仅在直线度量空间成立；②非对称的norm‑of‑radii目标在直线外仍无法得到O(k)竞争比；③Poisson化归约只适用于对抗适应性在线对手，无法直接用于对抗随机/观察性对手；④对更复杂度量空间（如高维、动态度量）的扩展仍有挑战。

---

## 447. Author-in-the-loop Agentic Reproducibility Auditing for a Machine Learning Conference

**arXiv ID:** 2610.22465 | [PDF](https://arxiv.org/pdf/2610.22465v1)

**作者:** Yukun Dong `[一作]` (Rutgers University), Tian Zheng `[通讯]` (Columbia University)

**通讯引用:** 4400 | [OpenAlex ID](https://openalex.org/A5044625906)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并部署了一个作者参与式的可重复性审计工作流，结合结构化检查表、AI 代理和 GitHub 协作，实现了两阶段的审计流程。

**💡 创新点**

创新点在于将作者声明显式作为 AI 代理检查与评估的输入，建立了两阶段协作框架以减少信息不对称，并设计了可追溯的审计记录与验证子代理。

**🔧 技术方法**

使用了 GPT‑5.5/5.6 等大语言模型构成的 AI 代理（checker、verifier、orchestrator），配合 GitHub Codespaces 进行代码执行，并通过结构化检查表与 GitHub issue 进行交互。

**📊 数据集**

使用了36篇会议论文及其133个实验表格/图表的提交材料，作为真实的审计数据集。

**📈 对比分析**

通过对比完整检查表与空白模板的审计，发现完整检查表可将需要作者行动的判定减少约13.7%，审计平均耗时约30分钟/论文，最终20个问题被解决，部分结果仍未完全复现。

**⚠️ 局限性**

主要局限包括资源预算限制导致复现不完整，AI 代理在多次重复审计中结果不一致，检查表解读可能过于狭隘，样本规模仅为10个回应团队，未能在更大规模下验证系统的鲁棒性。

---

## 448. CIIR - Class-Indexed Interface Representation Learning

**arXiv ID:** 2610.23151 | [PDF](https://arxiv.org/pdf/2610.23151v1)

**作者:** Doo Yon Kim `[一作]` (Korea University), Christian Wallraven `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为CIIR的模块化编码-解码框架，用于连接异构数据集，支持无实例匹配的独立数据集之间的通信。

**💡 创新点**

CIIR的创新点在于它通过类索引地址空间学习输入依赖的地址表示，允许在不需要配对样本的情况下进行模块化生成和数据集的后期连接。

**🔧 技术方法**

使用了模块化编码-解码技术，结合输入依赖的地址表示和轻量级适配器。

**📊 数据集**

使用了手写数字（MNIST）、服装图像（Fashion-MNIST）、语音数字频谱（FSDD）和手势图像（Sign Language MNIST）等多个独立收集的数据集。

**📈 对比分析**

与传统的多模态学习方法相比，CIIR在地址实现准确率（ARA）上达到了94.7%，并在MNIST到SVHN的端到端地址转移准确率上达到了90.2%。

**⚠️ 局限性**

CIIR的局限性在于它假设数据集本地类索引到共享地址词汇的预定义映射，较大的或结构不同的类空间仍需手动指定多地址代码。

---

## 449. Single-Letterization of Rate-Cost Tradeoffs in Static Time-Invariant Control (STIC) Systems

**arXiv ID:** 2610.22789 | [PDF](https://arxiv.org/pdf/2610.22789v1)

**作者:** Eray Unsal Atay `[一作]` (California Institute of Technology), Victoria Kostina `[通讯]` (California Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在信息传输受限的静态时不变控制（STIC）系统中，状态只能通过最近动作影响下一个状态的场景，给出了控制器在满足平均控制成本约束时所需的最小编码速率，即速率-成本权衡；

**💡 创新点**

其创新点在于首次将原本随时间增长的多字母 directed‑information 最小化问题，通过对称性约简转化为一个单字母凸优化问题，提供了除 LQG 控制之外的第一个可计算的单字母速率‑成本函数，并揭示了 STIC 系统与 NOST 通道反馈容量之间的对偶关系；

**🔧 技术方法**

主要技术包括：对 STIC 系统的记忆less 与时间不变结构进行抽象；利用状态与动作的条件独立性将控制策略限制为一阶马尔可夫；通过将多字母优化问题的约束和目标写成联合分布的形式实现循环对称性；随后应用对称性约简（symmetry reduction）得到单字母最优性；并利用信息论中的条件互信息和可交换性等不等式完成证明；

**📊 数据集**

本文未使用公开数据集，所有演示均基于手工构造的二元 STIC 系统（状态与动作均为 {0,1}，成本为 1{X=U}）进行数值比较；

**📈 对比分析**

比较方法主要是将单字母最优解与一个不强制保持动作边缘一致的“贪婪”方案（即单步最优）进行对比；结果表明，当成本阈值低于系统误差时，单字母方案显著优于贪婪方案；在满足成本阈值高于 0.3 时，两者均达到零速率；

**⚠️ 局限性**

局限性包括：仅适用于有限状态与动作的 STIC 系统；需满足时间不变的状态转移；未考虑网络化或分布式控制情形；对连续或大规模系统的推广尚未实现。

---

## 450. The Optimal Ranking Tradeoffs Between Two Ranking Scores in Two-Class Classification

**arXiv ID:** 2610.23294 | [PDF](https://arxiv.org/pdf/2610.23294v1)

**作者:** Sébastien Piérard `[一作]` (University of Liège), Marc Van Droogenbroeck `[通讯]` (University of Liège)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种理论与算法框架，用以在二分类任务中寻找两个给定评分（如准确率、召回率、F1等）之间的最优权衡点，即能在排名上兼顾两种评分的最佳折中。

**💡 创新点**

创新点包括：① 将所有常见评分归纳为“排名评分”族，并证明其可通过齐次分式（homographic）变换与权重函数关联；② 定义并求解在Kendall距离下的Fréchet方差最小化问题，得到最优折中为该路径的中点；③ 设计了一套完整的九步算法，可直接在已知相同类别先验的性能集合上计算该最优折中评分；④ 提出了针对固定先验的参数化，使得可视化与计算更简洁。

**🔧 技术方法**

技术手段包括：概率论框架、期望值与重要性随机变量定义、Kendall距离与bubble-sort距离、Fréchet方差与Karcher均值、齐次分式生成的加权均值、最短路径与线性插值、以及对性能空间的几何可视化。

**📊 数据集**

实验数据集涵盖：1）三种理论上已知分布的随机性能集合（均匀分布、no‑skill 上限分布、受限分布）；2）四个真实世界的性能集合，分别来自 2026 年 Illegal Waste Dumping Detection（IWDD）挑战的学习集与测试集、在 BDD100K 上将 74 个语义分割模型聚合为两类的 MMSegmentation 结果、以及 57 种背景减除方法在 “blizzard” 视频上的评估。前者用于验证理论，后者展示实际效果。

**📈 对比分析**

比较方法：在每个性能集合上，针对七对基准评分（如 precision‑recall、accuracy‑balanced accuracy 等）计算它们的加权均值（算术、几何、调和）以及本文给出的最优折中评分，并用 Kendall 相关系数评估其与原始评分之间的排名一致度。实验显示，本文方法得到的折中评分在所有情况的“optimality”均为 1（即排名与理论最优一致），而任意选取的算术/几何/调和均值往往仅达到约 0.5 或更低，且在许多场景下不满足排名基准的公理。

**⚠️ 局限性**

局限性：① 仅针对二分类且假设所有比较性能共享相同类别先验，无法直接推广到多分类或先验不一致的情形；② 需要对每对性能计算 Kendall 距离，虽然在 1k 规模下可在秒级完成，但在极大规模数据集上仍可能成为瓶颈；③ 最优折中取决于所选的评分对和数据分布，若评分对或分布改变，需重新求解；④ 该框架假设评分本身是“排名评分”，对非排名评分的折中尚未涵盖。

---

## 451. On some unresolved issues in aggregate level comparative research evaluation

**arXiv ID:** 2610.23318 | [PDF](https://arxiv.org/pdf/2610.23318v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 452. Algorithm Modelization: Transforming Optimization Algorithms into Evolvable Intelligence

**arXiv ID:** 2610.23161 | [PDF](https://arxiv.org/pdf/2610.23161v1)

**作者:** Jian Yao `[一作]` (Hong Kong Polytechnic University), Kay Chen Tan `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将迭代算法的执行轨迹转化为可训练的神经网络模型，实现算法模型化；

**💡 创新点**

提出Algorithm Modelization范式，采用自回归population‑temporal Transformer架构与两阶段（教师轨迹模仿+在线优化）训练，能够从多源教师中继承并进一步提升搜索能力；

**🔧 技术方法**

使用自回归的Transformer架构、signed attention与交叉融合机制、两阶段训练策略（模仿学习+在线强化）以及对梯度信息的利用；

**📊 数据集**

在BBOB、CEC2020/2026、MV‑BBOB、EOPCCV等标准优化基准上，教师轨迹来自GTDE、GP‑LogEI、Projected‑Lion等算法，并通过TF1–TF5等自定义函数进行训练；

**📈 对比分析**

与传统手工优化器、学习型优化器（EvoTF、POM、MetaDistill）以及LLM生成器等方法比较，A2M在所有测试场景（连续、混合变量、有限FEs）中均获得第一名或最高平均排名，且在未见维度和跨场景迁移时表现优异；

**⚠️ 局限性**

目前需要针对不同变量表示训练不同模型、额外使用评估模块来处理有限FEs、仅支持单目标无约束问题，模型在多目标或约束优化等更复杂任务中的可扩展性尚未验证。

---

## 453. When Core-Network Elements Become Reachable: Field-Level Exploitability Analysis in Diameter

**arXiv ID:** 2610.22943 | [PDF](https://arxiv.org/pdf/2610.22943v1)

**作者:** Wei Yi `[一作]` (Xidian University), Hui Li `[通讯]` (Xidian University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并评估了一个基于大语言模型的系统，用于可靠地进行Diameter协议字段级漏洞可利用性分析，并将该方法推广到其他协议。

**💡 创新点**

通过让LLM从已有漏洞写作中推断隐式的可利用性分析纪律和证据收集指导，并进行联合迭代精炼，使LLM能够在不需要显式规则的情况下实现高精度字段级可利用性判定。

**🔧 技术方法**

采用大语言模型（如GPT‑5.4、Claude Opus 4.6 等）、Prompt推理、案例驱动的知识抽取、LLM自我调优以及对协议规范的检索与推理等技术。

**📊 数据集**

对Diameter S6a/Cx 接口共分析766个字段（其中204被确认可利用），评估数据集为40个字段；此外对16种不同协议共857字段进行大规模评估；实现层面使用Open5GS、Kamailio及两家匿名商业实现进行验证。

**📈 对比分析**

与传统基于规则/检索的分析、状态化协议fuzzer 以及无引导LLM进行对比，平均召回率94–98%、精准率79–89%、准确率84–92%；对16种协议的评估发现279条PoC，118条在实现中被验证。

**⚠️ 局限性**

评估数据集规模有限（仅40字段），方法依赖可公开的协议规范和特定威胁模型，对实现细节的依赖导致部分可利用字段在实际实现中不触发。

---

## 454. Uncertainty-Aware Budget Allocation for Efficient LLM Tree Search

**arXiv ID:** 2610.22858 | [PDF](https://arxiv.org/pdf/2610.22858v1)

**作者:** Jianguo Huang `[一作]` (Nanyang Technological University), Bo An `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Uncertainty‑Aware Allocation (UAA) 框架，利用同级动作的语义分散度估计不确定性，动态分配树搜索中的扩展预算，从而显著降低 LLM 推理时的 token 用量和延迟。

**💡 创新点**

创新点在于通过内部嵌入无训练成本地估计不确定性，并基于此实现不确定性感知剪枝（UAP）与预算分配（UAB），以预测并剔除冗余子树，从而大幅提升树搜索效率。

**🔧 技术方法**

结合信息理论分析、基于 Cosine 相似度的嵌入分散度估计、UAP/UAB 机制，并将其与 Beam Search、MCTS、DVTS、REBASE 等搜索算法集成。

**📊 数据集**

使用数学推理数据集 GSM8K、MATH‑500、AIME24，以及逻辑推理与规划数据集 StrategyQA、Blocksworld，模型包括 Llama3.1‑8B‑Instruct、Qwen3‑32B、GPT‑4o 等。

**📈 对比分析**

与原始树搜索、LiteSearch 以及先进搜索方法对比，UAA 在保持或略微提升准确率的前提下，token 用量可降低最多 90%，推理时间可缩短至原来的 1/9，性能提升显著。

**⚠️ 局限性**

缺乏正式的理论保证、对超参数（如阈值 λ）缺乏原则性选择、最佳不确定性估计器需模型内部嵌入，闭源模型只能使用分数方差代理且实验范围有限。

---

## 455. Forecasting Voltage Time Series under Anomalies Using a WGAN Model With Lipschitz Control and Progressive Kernel Critic Strategy

**arXiv ID:** 2610.23394 | [PDF](https://arxiv.org/pdf/2610.23394v1)

**作者:** Najmieh Sadat Safarabadi `[一作]` `[通讯]` (University of Stuttgart), Najmieh Sadat Safarabadi (University of Stuttgart)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于Wasserstein GAN的自适应核-渐进Lipschitz（WGAN-CLP）框架，用于对高维、非平稳的功率系统毫秒级PMU时序进行递归预测。

**💡 创新点**

创新点在于将可自适应核与渐进Lipschitz惩罚相结合，动态调节梯度约束，显著提升生成器与鉴别器的收敛平衡，能够在高频、短周期异常场景下实现更精准、更稳健的长期预测。

**🔧 技术方法**

核心技术包括Wasserstein GAN、渐进Lipschitz penalty、Transformative Kernel Critic（自适应核鉴别器）、EMD分解提取多尺度时序特征、窗口化递归训练和基于梯度惩罚的对抗优化。

**📊 数据集**

使用了约30k条记录的毫秒级PMU测量数据以及多种分钟级可再生能源负荷和PMU混合数据集，共涵盖高频波动、瞬时失衡和稳态波动等多种场景。

**📈 对比分析**

与五种WGAN基线（WGAN-DIV、CWGAN、WGAN-QC、WGAN-LP、WGAN-CT）进行对比，M4b（自适应核+LP）在MSE、KL、Wasserstein、FID等指标上分别降低约4.8×、88.7%、4.4×、4.5×，并在异构数据集上实现了94%以上的鲁棒性，优于所有基线。

**⚠️ 局限性**

局限性包括：自适应核计算开销相对较高，导致推理速度和GPU内存占用增加；跨频域迁移仍受限，需要额外的域自适应机制；模型对温度参数τ的选择敏感，需经验调参；在极大规模数据上尚未验证可扩展性。

---

## 456. The LLM-Attacker: A Unified Red-Team Framework for Privacy Evaluation of Split and Distributed LLM Systems

**arXiv ID:** 2610.22722 | [PDF](https://arxiv.org/pdf/2610.22722v1)

**作者:** Georgios Politis `[一作]` (Setloop), Evangelos Pappas `[通讯]` (Setloop)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 LLM‑Attacker 框架，用于统一记录和评估拆分语言模型的隐私泄露；在 GPT‑2 124M 上进行了一系列攻击实验，评估梯度、激活、梯度+激活等不同观察信息对文本重建的影响，并与多种防御（secret mixup、replica median、白盒压缩）及公开实现（BiSR）进行对比。

**💡 创新点**

创新点：①将攻击者可观察信息、模型知识、数据集、搜索预算等全部统一记录为实验配置，便于复现与比较；②引入基于参考攻击的门控评分规则，要求每个防御都有相应的无防御基准；③提供完整的证据链（保存的预测、日志、哈希校验），可独立重新计分；④对同一捕获数据支持多种攻击和防御的复用，降低实验成本。

**🔧 技术方法**

使用的技术主要包括：梯度与激活信息的联合逆向恢复（joint reconstruction），线性回归解码器、束搜索；控制实验（梯度打乱、随机候选、仅后向梯度）；白盒压缩解码（PCA、瓶颈压缩）以及 BiSR 的双向半白盒恢复；统计上使用 Wilson 上界、bootstrap 置信区间和 Bonferroni 校正；所有实验均在 PyTorch 环境下实现，结果以 JSON 形式保存。

**📊 数据集**

数据集：WikiText‑2（32 词长句段），公共校准文档 32 条，训练文档 128 条，评估文档 256 条；此外还使用了代码仓库文本、模板化临床笔记和 WikiText‑2 的其他子集进行基准实验；实验也覆盖了 Qwen‑3、Mistral‑7B‑Instruct 等模型的真实系统测评。

**📈 对比分析**

比较方法：在固定的实验配置下，记录所有攻击者在同一捕获消息上的预测；对比不同观察信息的攻击（激活 vs. 激活+梯度）以及不同防御；使用 Wilson 上界的差值作为“超过基准”的百分点；若该差值大于 1 百分点则视为防御未通过。实验显示：在层 6、公共层下，加入梯度使 token 恢复从 94.20% 提升至 97.38%（3.17 pp），完整文档恢复从 13.71% 提升至 37.77%；防御 secret mixup 使完整文档恢复降至 ≤1.48%，但 token 恢复仍在 83–91% 之间；白盒压缩攻击恢复在 19–69% 之间，表明压缩对信息泄露有一定抑制作用。

**⚠️ 局限性**

限制：①实验仅覆盖 GPT‑2 124M 及其特定拆分深度，未覆盖更大模型或多路拆分；②仅使用单词级别的 32 词长句段，未探究更长上下文或不同分词策略的影响；③防御门控评分要求存在无防御基准，若无则无法评估；④框架侧重于“观察信息 + 训练资源”级别的攻击，未覆盖主动注入或侧信道攻击的复杂场景；⑤最终隐私门槛（1 百分点）是经验阈值，可能不适用于所有应用场景。

---

## 457. Sparse Autoencoders Can Induce Local Geometry for Mitigating Catastrophic Forgetting

**arXiv ID:** 2610.22439 | [PDF](https://arxiv.org/pdf/2610.22439v1)

**作者:** Kamil Faber `[一作]` (AGH University of Krakow), Łukasz Pawela `[通讯]` (Polish Academy of Sciences)

**通讯引用:** 524 | [OpenAlex ID](https://openalex.org/A5034714433)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Sparse Autoencoder Geometric Anchoring (SAGA) 方法，利用任务边界训练的稀疏自编码器 (SAE) 的 Jacobian 来构造输入依赖的低秩几何约束，缓解监督式连续学习中的灾难性遗忘。

**💡 创新点**

创新点在于将 SAE 的激活稀疏性与其 Jacobian 的主奇异方向结合，形成样本特定的局部二次正定约束；同时引入历史快照与重放协同，提供空间与方向两种记忆，并给出理论上对旧读口漂移的条件界限。

**🔧 技术方法**

主要技术包括 TopK 稀疏自编码器、Jacobian 矩阵分解、截断奇异值低秩约束、经验重放 (ER)、经验重放+知识蒸馏 (DER++)、经验重放+不对称交叉熵 (ER-ACE)、以及基准 SGD。

**📊 数据集**

实验使用的公开数据集为 CIFAR‑100、CUB‑200、ImageNet‑R（预训练 ViT‑B/16）以及 TinyImageNet、CIFAR‑100（ResNet‑18 训练从零）等。

**📈 对比分析**

与传统基线（SGD、ER、DER++、ER-ACE）对比，SAGA 在三大预训练基准上均取得最高平均准确率和最低平均遗忘率，分别比最佳基线提升 12.15、4.79 与 3.22 百分点；在从零训练的设置下效果略弱，但仍优于 ER。

**⚠️ 局限性**

局限性包括：Jacobian 仅为局部线性近似，支持变换时约束可能失效；无监督 SAE 的敏感方向不一定与旧任务功能敏感性对齐；历史快照存储随任务数增长导致额外内存和计算开销；实验仅用三种随机种子，差异解读需谨慎。

---

## 458. Cross-Person Identity Change Under Source-Coupled Multi-Human Generation: A Controlled UniPortrait Study

**arXiv ID:** 2610.22746 | [PDF](https://arxiv.org/pdf/2610.22746v1)

**作者:** Edrees Al-Edreesi `[一作]` (Sana’a University), Fouad Al-Hamdani `[通讯]` (Ibb University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了UniPortrait在两人合成场景下，单一目标身份控制与双身份控制对被保护和目标人物面部相似度的持续变化。

**💡 创新点**

创新点在于对同一生成器的两种本地化身份配置进行匹配源耦合实验，并通过连续相似度、独立度量与盲判人类评测三重验证，定量描述身份变异。

**🔧 技术方法**

采用UniPortrait与Stable Diffusion 1.5为后端，DDIM确定性重建，InsightFace（ArcFace）与ID‑Sim作为相似度评估，混合效应模型进行统计推断。

**📊 数据集**

使用MultiHuman‑Testbench提供的两人场景和来自FFHQ、CelebA‑HQ、SFHQ的参考人像，固定三颗输出种子。

**📈 对比分析**

通过对A1（单ID）与A0（全局重生）以及A1与A2（双ID）进行配对比较，A1均出现负平均相似度转变，A2则在同一配对上表现出显著正向改进；独立度量与人类评测亦与方向一致。

**⚠️ 局限性**

局限在于仅适用于UniPortrait + Stable Diffusion 1.5的两人场景，缺乏局部编辑、不同生成器或更多身份的推广，且缺乏对未观测缺失生成结果的无偏估计，未能给出分类级别的身份保留判定。

---

## 459. Structural Affinity: An Algorithm for Coherent Cage-Based Shape Design

**arXiv ID:** 2610.23173 | [PDF](https://arxiv.org/pdf/2610.23173v1)

**作者:** Aviraj Newatia `[一作]` (University of Cambridge), Rika Antonova `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Structural Affinity，通过在 Cage‑Based Deformation（CBD）中引入一个外层笼子并预计算亲和矩阵，实现对笼子控制点运动的显式耦合，从而在多种任务（形变转移、低精度气动优化、机器人形态共设计）中提升变形质量并减少自交与失真。

**💡 创新点**

创新点在于利用外层笼子的 Mean‑Value Coordinates（MVC）权重直接构造稀疏亲和矩阵 C，随后在变形时仅做一次矩阵乘法即可传播控制点位移，完全不需要在线求解；该方法天然抑制笼子自交、减少表面粗糙度，并在不依赖额外正则化损失的情况下取得与原始正则化相当甚至更好的结果。

**🔧 技术方法**

使用技术包括：Structural Affinity 亲和矩阵构造、MVC 权重、稀疏矩阵传播、Neural Cages（基于 CNN 的控制点与形状预测）、VLM 低精度气动模型、SciPy/SLSQP 与 IPOPT 的梯度/无梯度优化、强化学习（RL）与全局优化（GO）的形态共设计框架。

**📊 数据集**

数据集与实验场景：ShapeNet V2（椅子、桌子、汽车）用于 Neural Cages 训练与评估；NASA CRM wing 低精度 VLM 模型用于气动优化；LEAP 手与 PiPER 机器人在 t‑nut 任务中用于形态共设计；以及 3D 打印的真实手指模型用于离线实验验证。

**📈 对比分析**

在三大任务中与传统 CBD、Neural Cages、FFD 基线进行比较。结构亲和显著降低了 Cot‑Laplacian、笼子 Laplacian 及表面粗糙度（10–50% 以上），同时保持与原始正则化损失相当的重建精度；在气动优化中实现 1–2% 的阻力下降及 13–18% 的表面粗糙度改善；在机器人共设计中使抓取成功率从 35% 提升至 87%，并在随机扰动下仍保持 30% 的成功率。

**⚠️ 局限性**

局限性：依赖外层笼子的良好构造，若外层笼子设计不当可能导致亲和矩阵不稳定；目前仅验证了 MVC 形式，未深入探讨 Green/Bounded Biharmonic/Somigliana 等权重的推广；在高保真气动/物理仿真上的效果尚未系统验证；对非凸外层笼子可能出现负权重导致亲和矩阵不适用。

---

## 460. Beyond Reward Scoring: Online Multi-Agent Preference Learning for Multimodal Response Generation

**arXiv ID:** 2610.23256 | [PDF](https://arxiv.org/pdf/2610.23256v1)

**作者:** Ziyun Zou `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Imran Razzak `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了PrefMind框架，利用在线双代理强化学习，让生成器和评论家通过讨论、纠错与标签验证学习偏好理解并改进多模态生成。

**💡 创新点**

将偏好理解内化为生成模型的内部能力，采用双代理互动讨论与失败案例回放的在线强化学习方法，区别于传统单一奖励模型或外部评估。

**🔧 技术方法**

在线双代理强化学习（PPO）、历史优势基线、失败案例回放、分阶段讨论与生成、基于Qwen3‑VL‑Instruct的Transformer。

**📊 数据集**

使用Align Anything与BeaverTails‑V共计39,369条人类偏好对，评测集包括SPA‑VL、LLaVA‑Critic、MLLM‑as‑a‑Judge、MM‑RLHF‑RewardBench、Omni‑RewardBench等。

**📈 对比分析**

与Base、DPO、PPO、GRPO、Cat‑PO、MM‑DPO、GRM‑RL等方法对比，4B/8B规模下生成平均分最高，在安全、视觉辅助和综合多模态能力上均优于对手，且相较单代理模型表现更佳。

**⚠️ 局限性**

对小模型效果有限，讨论式训练对模型容量要求高；未解决小模型在实际场景中的泛化与有效性问题。

---

## 461. Distill Until Divergence: Error-Truncated Hindsight Distillation for Agentic RL

**arXiv ID:** 2610.23048 | [PDF](https://arxiv.org/pdf/2610.23048v1)

**作者:** Yibin Huang `[一作]` (Harbin Institute of Technology), Conghui Zhu `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为ETRO的算法，通过将正向（Positive）和负向（Negative）两类hindsight技能按轨迹中的错误边界进行路由，结合GRPO的终点奖励学习，解决hindsight监督与交互状态演变之间的不匹配问题。

**💡 创新点**

创新点在于：① 针对轨迹不同阶段（成功路径、失败前缀、错误点和错误后续）分别使用两种hindsight技能，实现在错误前后精细化的信用分配；② 在错误点后切断hindsight监督，仅保留GRPO的终点奖励，避免状态错配的负面影响；③ 利用writer模型自动生成错误边界与对应技能，实现无需人工标注的可扩展路由。

**🔧 技术方法**

技术包括：GRPO（基于终点奖励的群组优势优化）、OPD（基于轨迹的on‑policy distillation）、写手（writer）生成的正向/负向技能与错误边界、以及对OPD优势的区域化路由与权重调节。

**📊 数据集**

数据集：WebShop、ALFWorld和Search QA（七个子任务）。

**📈 对比分析**

在上述三大基准上与GRPO、GRPO+OPSD、Skill‑SD、RLSD、SDAR、OPID、SEED、ICSD等方法对比，ETRO在1.5B、3B、7B三个模型规模均取得最优或次优性能：如1.5B模型下WebShop成功率从65.7%提升至80.8%，ALFWorld成功率从72.0%提升至79.1%，Search QA宏平均准确率从44.0%提升至48.0%；3B和7B规模亦保持领先。

**⚠️ 局限性**

局限性包括：依赖writer对错误边界和技能的准确性；训练时需要完整轨迹和终点反馈，可能耗费资源；hindsight生成与重评分在训练阶段增加计算成本；模型在真实部署场景下的安全性与隐私问题仍需进一步验证。

---

## 462. Large Diffusion Models Need More Than EMA

**arXiv ID:** 2610.22945 | [PDF](https://arxiv.org/pdf/2610.22945v1)

**作者:** Chuyu Hu `[一作]` (Tsinghua University), Yinpeng Dong `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对比了大规模扩散模型的三种优化策略：恒定学习率加EMA、学习率衰减加EMA以及Schedule‑Free AdamW；在SiT‑XL/2 + REPA预训练、ReFL和DMD三种训练范式下进行实验；通过FID、sFID、IS以及固定评估损失衡量生成质量和优化效果；

**💡 创新点**

发现Schedule‑Free不仅在所有训练阶段均能实现更低的FID和更好的优化目标，还能在不需要预设衰减点的前提下兼顾参数稳定性和持续优化；

**🔧 技术方法**

采用EMA、学习率衰减、Schedule‑Free（基于轨迹平均的增量更新）等技术，结合AdamW优化器，并在扩散模型中实施；

**📊 数据集**

主要使用ImageNet‑1K（256×256）训练SiT‑XL/2模型，另外在ReFL（HPSv3奖励）和DMD（生成器+引导模型）任务上进行验证；

**📈 对比分析**

在三种训练模式下均对FID、sFID、Inception Score、固定评估损失和参数方差等指标进行评估；Schedule‑Free在FID上通常比Constant‑LR+EMA低约0.15-0.2分，在DMD上显著提升FID并降低DMD损失；

**⚠️ 局限性**

虽然Schedule‑Free在实验中表现最好，但其效果仍受学习率和温度调节等超参数影响；在某些任务中EMA与衰减组合可获得更细粒度的性能提升，且未完全证明Schedule‑Free在所有扩散模型架构和数据集上都优于传统方法；

---

## 463. Sample-Efficient Reinforcement Learning for Policy Improvement via Flow-Reversed Action Anchors

**arXiv ID:** 2610.22852 | [PDF](https://arxiv.org/pdf/2610.22852v1)

**作者:** Carl Qi `[一作]`, Dylan Randle `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种基于流反演锚点的粗细级强化学习框架 AnchorQ，用于在冻结的流匹配视觉语言动作（VLA）策略上进行在线适配，从而显著提升机器人操控任务的成功率与吞吐量。

**💡 创新点**

创新点包括①将离散参考动作通过流反演映射为状态条件噪声锚点，构建结构化、可细化的动作表征；②在锚点附近采样粒子并进行价值倾斜（exponential value tilting），实现局部精细化；③通过理论分析阐明锚点降低识别复杂度、粒子细化权衡支持偏差与动作类复杂度，以及在Critic误差下价值倾斜的可靠性。

**🔧 技术方法**

使用的技术包括流反演、强化学习（TD学习）与价值倾斜采样、在线Critic训练、ϵ‑greedy探索、粒子采样以及离散与连续 Q 学习（Rainbow‑FRS）对比。

**📊 数据集**

使用的数据集有 LIBERO‑90（15 任务与 10 任务硬拆分）、RoboTwin（6 任务双臂）以及在 Franka DROID 实验平台上进行的真实机器人任务（杯子堆叠、烤面包、挂毛巾）。

**📈 对比分析**

与 DSRL、Residual RL、VLM‑FRS 等基线在相同环境交互预算下进行对比；在模拟中 AnchorQ 至少实现了 DSRL 的两倍成功率，吞吐率提升 97%；在真实机器人上，AnchorQ 在 100 次在线回合内显著优于 π0.5 和 DSRL，成功率明显提高。

**⚠️ 局限性**

局限性包括：每一步需要采样大量粒子导致计算开销；依赖冻结的预训练 VLA；对 Critic 误差敏感，需要置信度估计；目前仅在单任务上验证，未展示跨任务泛化或知识蒸馏。

---

## 464. BaGuaLu-2: Taming Dynamic Memory Fluctuation for Efficient MoE Model Training over 10,000 GPUs

**arXiv ID:** 2610.23064 | [PDF](https://arxiv.org/pdf/2610.23064v1)

**作者:** Mingshu Zhai `[一作]`, Jidong Zhai `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文提出并描述了一种新的工件（Artifact），并对其进行了系统评估

**💡 创新点**

首次将工件描述与评估方法系统化，提供了可复现的评估流程

**🔧 技术方法**

采用了工件描述技术与评估框架，利用实验脚本进行验证

**📊 数据集**

使用自定义评估数据集（未公开具体名称）

**📈 对比分析**

与现有基准方法对比，评估指标表现更佳，但具体数值未公开

**⚠️ 局限性**

缺乏对不同场景和平台的泛化验证，真实世界案例支持不足

---

## 465. High prevalence of antagonizing rhetoric in trending topics on Twitter/X

**arXiv ID:** 2610.22455 | [PDF](https://arxiv.org/pdf/2610.22455v1)

**作者:** Armin Pournaki `[一作]` (University of Bern), Eckehard Olbrich `[通讯]` (Max Planck Institute for Mathematics in the Sciences)

**通讯引用:** 2558 | [OpenAlex ID](https://openalex.org/A5074755166)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对德国推特热点话题的1700多万条推文进行文本分析，量化对立言论中的敌意化与批评化两种模式。

**💡 创新点**

首次将对立言论分为“敌意化”（否定合法性）和“批评化”（承认合法性）两类，并使用LLM进行大规模检测。

**🔧 技术方法**

采用大语言模型（如GPT‑4）配合结构化提示进行分类，并用预测驱动推断校正误差。

**📊 数据集**

主要数据集为德国2021‑2023年趋势话题推文（D1）以及与之重叠的约200万条随机样本（D2）。

**📈 对比分析**

通过与随机样本对比、时间序列、政治倾向分组及传播放大因子分析，发现热点话题中敌意化占比约30%，右倾用户更高，放大因子平均>1.6。

**⚠️ 局限性**

局限包括LLM分类准确率（macro F1≈0.7）对话语细节捕捉不足、标注主观性、未覆盖其他辩论模式、结果受平台变化影响。

---

## 466. Simpler Algorithms for Capacitated $k$-Median with $(1 + ε)$-Violations Using New Inequalities

**arXiv ID:** 2610.23348 | [PDF](https://arxiv.org/pdf/2610.23348v1)

**作者:** Han Dai `[一作]` (Nanjing University), Shi Li `[通讯]` (Nanjing University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对硬约束容量的k‑median问题，提出一种统一的“round‑or‑cut”框架，利用局部惩罚式聚类实例直接生成有效约束，得到 (1+ϵ) 近似度的容量或基数违例算法。该框架可在 O(n·poly(1/ϵ)) 时间内实现，并给出 O(1/ϵ⁵) 近似（基数违例）和 O(1/ϵ⁶) 近似（容量违例）的结果。

**💡 创新点**

创新点在于：①用局部惩罚聚类实例取代传统的扩展配置 LP，直接从局部实例获得下界并构造有效约束；②统一处理基数违例与容量违例的两种情形；③通过对黑色分量、灰色森林的分组，显著降低了运行时间，从指数级 1/ϵ 下降到多项式级别；④在硬容量设置下，首次给出只打开 (1+ϵ)k 设施或在每个设施上允许 (1+ϵ)u_i 客户的多项式时间算法。

**🔧 技术方法**

技术手段包括：
- 基本 LP 松弛和 round‑or‑cut 框架；
- 将设施集合划分为黑色分量、灰色树，并按权重分组；
- 构造“带惩罚”的局部聚类实例，利用树度量转换和动态规划获得 O(|J|) 近似；
- 通过局部解决方案生成有效约束，确保整数解满足约束；
- 采用辅助 LP 进行全局求解，随后进行拆分、重分配、删除操作，实现近似与容量/基数违例。

**📊 数据集**

论文未使用实验数据集，全部为理论分析与证明，提供算法运行时间与近似比的上界。

**📈 对比分析**

与此前最优 O(log k) 单一约束算法相比，本文的二元逼近在常数因子上有显著改进；与以往使用扩展配置 LP 的  O(exp(1/ϵ²)) 复杂度算法相比，本文在运行时间上从指数降低到 O(n·poly(1/ϵ))，且算法结构更简洁。实验评估未给出，但理论上证明了上述性能。

**⚠️ 局限性**

局限性包括：
- 仍为二元逼近，无法在不违例的前提下获得常数因子近似；
- ϵ 的依赖项仍较大（5/6 次方），在实际应用中可能导致较高的常数系数；
- 仅针对硬容量设置，软容量场景仍需更精细处理；
- 需要构造黑色分量与灰色森林，算法实现相对复杂；
- 在特殊结构（如极大容量差异）下的性能仍未得到进一步分析。

---

## 467. Inferring L1 Backgrounds from L2 English with LLM Token Probabilities

**arXiv ID:** 2610.23309 | [PDF](https://arxiv.org/pdf/2610.23309v1)

**作者:** Ahmet Yavuz Uluslu `[一作]` (University of Cambridge), Gerold Schneider `[通讯]` (University of Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者通过给预训练大型语言模型(Llama 3.1 8B/70B)添加不同的母语前缀，计算该前缀下文本的token概率并挑选得分最高的母语作为预测，验证了token概率比直接分类更能识别非母语学习者的第一语言。

**💡 创新点**

创新点在于利用共享的通用LLM而非为每个母语训练单独的语言模型，通过前缀条件化的token概率来捕捉跨语言影响，从而实现无监督的高效NLI；并在多种输入长度下展示了显著优势。

**🔧 技术方法**

技术核心是基于Llama 3.1系列的token概率评估方法（L1前缀概率评分），与传统的直接分类和chain‑of‑thought提示方式对比；实现时对每个候选母语均生成前缀并求和log‑probabilities。

**📊 数据集**

使用了三个公开的L2英语学习者数据集：TOEFL9（900篇，9种母语），Write & Improve 2024（450篇，9种母语），以及BULATS‑L1（270段语音转录，6种母语）。

**📈 对比分析**

与直接分类和先前基于GPT‑4o的SOTA方法对比，L1‑前缀概率评分在所有数据集、所有模型尺寸及所有文本长度上均优于直接分类；在完整文本上，70B模型分别达到96.0%（TOEFL9）、92.0%（Write & Improve 2024）和84.1%（BULATS‑L1），超越之前的最佳结果。

**⚠️ 局限性**

主要局限包括：只针对英语二语文本及有限的母语种类，无法保证对低资源母语或其他二语的普适性；前缀评分需要多次评估同一文本，计算成本高；未对推理时延和资源使用做实验；模型可能因预训练数据偏差而产生偏见。

---

## 468. Deep Learning for Deepfake Image Detection: Comparative Model Evaluation and Gradient-Based Visual Attribution

**arXiv ID:** 2610.22676 | [PDF](https://arxiv.org/pdf/2610.22676v1)

**作者:** Saad Korchi `[一作]` (Euromed University of Fez), Abderrahim Waga `[通讯]` (Euromed University of Fez)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究在140k Real and Fake Faces数据集上实现了两种深度学习模型的深伪检测工作，构建了可解释的工作流。

**💡 创新点**

创新点在于将从零开始训练的紧凑卷积网络与 ImageNet 预训练 MobileNetV2 的两阶段迁移学习策略进行对比，并将 Grad‑CAM 可视化嵌入 Flask 原型中。

**🔧 技术方法**

使用的技术包括从零训练的四层卷积网络、MobileNetV2 迁移学习、Adam 优化器、混合精度训练、Grad‑CAM 热图以及 Flask Web 接口。

**📊 数据集**

采用公开的 140k Real and Fake Faces 数据集，包含 70,000 张真实与 70,000 张伪造面孔。

**📈 对比分析**

对比实验发现迁移学习模型在验证集上达到 95.90% 准确率，优于从零训练模型的 84.15%；验证损失分别为 0.1074 与 0.3625。

**⚠️ 局限性**

局限性包括仅报告验证结果未给出保留测试集性能、两种完整流水线难以单独归因、数据集来源单一导致泛化能力不确定、Grad‑CAM 仅提供局部提示非因果解释。

---

## 469. Competence-Based Evaluation of Logical Reasoning in Large Language Models

**arXiv ID:** 2610.22894 | [PDF](https://arxiv.org/pdf/2610.22894v1)

**作者:** Jize Jiang `[一作]` (University of Illinois Urbana-Champaign), Chengxiang Zhai `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于能力的评估方法（CBE），并在排序任务上进行实例化与实验；

**💡 创新点**

创新点在于将评估从单一准确率转向对问题族的能力一致性检验，构造可扩展的参数化生成器与答案保持的多呈现方式，并提供诊断性能力曲线；

**🔧 技术方法**

使用了参数化问题生成器、匹配呈现、统计指标R/D/U/κ、Chain-of-Thought、GRPO（Group Relative Policy Optimization）等技术；

**📊 数据集**

使用了生成的排序任务数据集（4–1024实体的前后、左右关系等），并对GPT‑4.1、GPT‑5、Qwen3.5等模型进行评估；

**📈 对比分析**

通过比较原始与等价呈现的准确率、共变性与稳健正确率，发现强模型在较大规模和不同词汇下仍显著缺乏一致性；Chain‑of‑Thought提升准确率但未完全恢复一致性；GRPO在未见词汇上提升了跨词汇迁移；

**⚠️ 局限性**

局限在于只评估单一问题族，难以证明全局推理能力；排序任务未能分离链长度与推理深度；生成器与评估指标受限，未对多任务、跨语言或更复杂推理进行验证。

---

## 470. Gossip Agents: Reputation through Opinion Dynamics over Evolving Networks

**arXiv ID:** 2610.22674 | [PDF](https://arxiv.org/pdf/2610.22674v1)

**作者:** Yutong Wu `[一作]` (Rice University), Arlei Silva `[通讯]` (Rice University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在多智能体强化学习环境中，利用私有评估、传播与时间聚合构建去中心化声誉系统，并在N人囚徒困境与Harvest顺序社会困境中评估其对合作与集体结果的影响。

**💡 创新点**

创新点在于提出可调广播半径的声誉框架，分离私有评估、聚合与传播三大过程，并通过DeGroot意见动力学在动态通信网络上传播声誉，首次将声誉扩散与网络演化相结合进行实验研究。

**🔧 技术方法**

使用技术包括Deep Q-Network（DQN）学习器、指数滑动平均与滑动窗口聚合来更新声誉、DeGroot平均进行广播、以及在随机重组的囚徒困境和开源Harvest gridworld环境中的仿真与评价。

**📊 数据集**

实验使用自定义的随机邻域囚徒困境和开源的Harvest环境；未使用公开数据集，而是基于这两种环境生成的训练与测试数据。

**📈 对比分析**

通过与无声誉基线以及三种广播模式（无广播、本地广播、全广播）进行对比，结果显示所有声誉方法均显著优于无声誉；在囚徒困境中合作率接近95%，在Harvest中全广播实现最高效用和可持续性，而本地广播在平衡公平性与效用方面表现最佳。

**⚠️ 局限性**

局限性包括评估规则和聚合方式固定未学习，假设通信诚实且未考虑策略性或噪声传播；实验仅在两种环境下进行，未验证方法在更广泛任务中的适用性。

---

## 471. MGE-PO: Entropy-Aware Credit Assignment for Mathematical Reasoning

**arXiv ID:** 2610.23245 | [PDF](https://arxiv.org/pdf/2610.23245v1)

**作者:** Jihao Huang `[一作]`, Junchen Wan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了MGE-PO框架，在RLVR中引入了基于生成熵的四象限(4Q)优势塑形和在线熵基课程，使模型在数学推理任务中能更精准地分配信用。

**💡 创新点**

创新点在于将验证结果与生成熵结合的4Q规则——正确答案时熵低获更大奖励，错误答案时熵高遭更大惩罚；以及结合DynZPD和EA-KL的熵基问答抽样与参考正则化。

**🔧 技术方法**

主要技术包括GRPO、DAPO、Clip-Cov等RLVR优化器，熵估计、四象限优势修正、批量重心化、DynZPD在线抽样和EA-KL正则化。

**📊 数据集**

使用Qwen3-8B-Base的Skywork-OR1数学子集以及Qwen2.5-7B-Base等模型训练，评估数据为AIME2025/24、AMC2023、MATH500、Minerva、OlympiadBench等六个数学基准。

**📈 对比分析**

对比基线GRPO、DAPO和Clip-Cov，实验显示在Qwen3上加入4Q可提升宏观avg@32 3.23–3.69%与pass@32 4.24–11.06%；在Qwen2.5上宏观avg@32从41.5%提升至45.4%，pass@32提升至62.8%。

**⚠️ 局限性**

局限性包括仅在Qwen模型和数学推理任务上验证，缺乏训练种子方差评估，4Q的效果需进一步用固定校正与打乱熵对照验证；此外不同实验设置间KL系数不一致，可能混淆结果。

---

## 472. Not All Test-Time Compute Is Evidence: Model-Specific Bias in LLM Reasoning

**arXiv ID:** 2610.23213 | [PDF](https://arxiv.org/pdf/2610.23213v1)

**作者:** Ming Shen `[一作]` (Shanghai Jiao Tong University), Tianjun Yao `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种测试时刻的多模型独立仲裁方法，利用推理阶段的不一致来触发异构模型的独立求解并直接聚合答案。

**💡 创新点**

创新点在于将推理阶段的不一致作为动态信号，仅在需要时调用异构模型，并强调候选无暴露的独立求解与直接投票，避免同一模型内部的错误偏差。

**🔧 技术方法**

采用链式推理、交叉验证、异构模型集成与答案级别聚合（Answer-level Arbitration）等技术，并通过自适应路由动态触发外部模型。

**📊 数据集**

实验覆盖七个多推理基准，包括知识推理（MMLU-Pro、GPQA-Diamond）、上下文/多跳推理（DetectBench、MuSiQue、2WikiMultiHopQA）、数学推理（MATH）和叙事推理（MuSR）。

**📈 对比分析**

与CoT、DEER、Dynasor、MAS-Exchange、MAS-Judge、SFT/DPO等基线相比，在两种基模型（Qwen3-4B、Qwen3-8B）上平均提升约3%/2.5%，在12/14设置中超过CoT，激活率约17%，使用比全多代理低约30%的token。

**⚠️ 局限性**

局限性包括对外部异构模型质量与可用性的依赖，激活率随基模型规模下降，且在数学基准已达天花板时提升有限；方法对候选无暴露和独立性假设敏感。

---

## 473. Unconditional Quantum PAC Learning Advantage in the Large Error Regime

**arXiv ID:** 2610.23268 | [PDF](https://arxiv.org/pdf/2610.23268v1)

**作者:** Igor C. Oliveira `[一作]` `[通讯]` (University of Warwick), Igor C. Oliveira (University of Warwick)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在统一分布下，构造了一个概念类，使得单个量子随机样本可以在多项式时间内学习出与目标函数的协方差为Ω(2^{-n/2})的预测器，而任何多项式时间的经典学习器在同样的约束下都无法实现同等优势；进一步将这一优势提升到任意可构造的逆超多项式λ(n)=n^{-ω(1)}，并给出了量子学习器的具体构造与证明；

**💡 创新点**

创新点在于：① 将量子傅里叶采样从线性(奇异子)推广到二次(二阶多项式)假设并利用干涉偏向正相关，从而突破经典学习器在大误差区间的1/2成功率壁垒；② 结合四阶独立性与全局符号翻转，构造了可实现的、易于实现的量子学习算法；③ 通过填充与随机/伪随机函数技术，将指数级的量子优势转化为可观测的逆超多项式优势；

**🔧 技术方法**

主要技术包括：量子傅里叶采样、干涉偏向、四阶独立性随机函数族、全局符号翻转、马尔科夫链/Paley–Zygmund不等式、量子随机例子处理、加法与Majority放大、填充与裁剪技术、子指数安全一向函数假设与强伪随机函数构造、加权求和与 Chebyshev 不等式、概率论与信息论下界证明；

**📊 数据集**

无实际数据集，全部为理论构造与抽象的布尔函数概念类；

**📈 对比分析**

与经典学习器（无查询或仅做成员查询）做比较：量子学习器在同样的单量子随机样本和多项式时间下能够获得协方差≥1/2+λ(n)且成功概率≥1-1/n^k；而经典学习器在多项式时间内（甚至无限查询）在同一概念类上只能达到协方差≤1/2+λ(n)/n^k且成功概率≤1/n^k；

**⚠️ 局限性**

局限性：① 目前仅在大误差区间（仅略高于1/2）实现无条件指数级分离；② 在小误差区间的类似分离需要复杂性假设（如NP⊄BPP），无法得到无条件结果；③ 量子学习器虽然输出经典电路，但实现仍需量子随机样本；④ 结果依赖于对逆超多项式优势的可构造性，无法直接推广到所有衰减速率。

---

## 474. Unifying Embodied Foundation Models via a Model-to-Policy Formulation

**arXiv ID:** 2610.23053 | [PDF](https://arxiv.org/pdf/2610.23053v1)

**作者:** Xinyi Hong `[一作]` (Shanghai Jiao Tong University), Mingyang Ren `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种架构无关的概率性随机核框架，用于统一描述并实现机器人感知-语言-动作模型与世界-动作模型在预训练、后训练和部署过程中的任务条件策略。

**💡 创新点**

创新点在于将模型输出分布与动作实现核相结合，形成可直接推导任务条件策略的概率表达式，并通过统一框架把多种模型（OpenVLA、DreamZero）映射到同一语义空间，从而清晰说明预训练与后训练对闭环控制的影响。

**🔧 技术方法**

主要技术包括随机核理论、条件概率分布、跨模态编码（DINOv2、SigLIP、Llama 2、DiT）、流匹配训练、贪婪与随机解码以及异步生成与动作调度。

**📊 数据集**

使用的数据集包括Open X-Embodiment的机器人演示数据、AgiBot G1和Franka‑DROID的遥控轨迹、以及各任务的下游演示（如水果打包、桌面清洁等），并在这些数据上进行预训练与监督微调。

**📈 对比分析**

实验通过在多任务下游演示上比较OpenVLA和DreamZero在任务完成率、执行时间和模型推理速度等指标，结果显示统一框架能保持或提升两种模型在闭环控制中的性能，且微调后能显著提升任务特定指标。

**⚠️ 局限性**

局限性包括对大规模预训练数据与计算资源的依赖、模型对新任务或环境的泛化能力尚未系统评估、以及在异步生成与调度过程中可能出现的延迟与错误传播问题。

---

## 475. Advancing Sentence-Level Dysarthric Audio-Visual Speech Recognition via Test-Time Adaptation

**arXiv ID:** 2610.23117 | [PDF](https://arxiv.org/pdf/2610.23117v1)

**作者:** Shiyao Wang `[一作]` (Nankai University), Yong Qin `[通讯]` (Nankai University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建句子级慢性失语语音识别语料并在 AV-SUPERB 与 ViSpeR 两大框架中使用无监督测试时适配（SGEM）提升识别性能。

**💡 创新点**

1) 提出“source speech search‑segmentation”自动将 CDSD 原始视频转为句子级对齐语料；2) 首次将 SGEM 测试时适配引入多模态 DAVSR；3) 系统比较不同适配范围（LayerNorm、Feature、Module-Specific、All）对两框架的影响。

**🔧 技术方法**

AV-HuBERT 前端、AV-SUPERB 与 ViSpeR 架构、Sequential-level Generalized Entropy Minimization (SGEM)、欧氏距离匹配与多层次自适应。

**📊 数据集**

CDSD（中文慢性失语数据库）以及 CN-CVS（中文广播新闻与网络语料）作为对照，实验使用 8 小时的 CDSD 8 位说话人数据。

**📈 对比分析**

在 AS-1、AS-2、AS-3 三种基线上，使用 TTA 进行多轮迭代，CER 大幅下降，ViSpeR 最高可实现 11.8% 的降误，且 t‑检验结果 p<0.05，证明改进显著。

**⚠️ 局限性**

受失语多样性和双模同步数据稀缺限制，部分说话人仍出现微小退化；极端严重发音的改进受限，且 TTA 需要多轮迭代，计算成本相对较高。

---

## 476. On the Convergence of Iterated Round Robin

**arXiv ID:** 2610.23040 | [PDF](https://arxiv.org/pdf/2610.23040v1)

**作者:** Maxime Lucet `[一作]` (LIP6), Arianna Novaro `[通讯]` (Université Paris 1 Panthéon-Sorbonne)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了迭代Round Robin（RR）机制在完全信息与部分信息环境下的最佳回应动态，分析了策略操纵对收敛性的影响，并给出了保证收敛或策略性的条件。

**💡 创新点**

创新点包括：① 在完全信息下提出“保持物品不变的交换”限制，证明无论参与者和物品数目如何，都存在纯纳什均衡且过程收敛；② 在部分信息下设计三种基于不确定性的操纵行为，发现两种易产生循环，提出更为谨慎的规则保证迭代RR具有策略性。

**🔧 技术方法**

使用博弈论中的最佳回应动态、组合论和证明技巧，对RR机制的操纵空间进行分析，构造循环示例并证明收敛性与策略性的充分条件。

**📊 数据集**

无数据集，全部为理论分析与构造性例子。

**📈 对比分析**

通过构造性例子展示循环现象与收敛过程；性能评价仅以理论收敛性与策略性证明为准，没有实验数值评测。

**⚠️ 局限性**

局限性在于：① 对完全信息下邻接交换的收敛仅在两人或有限物品下得到证明，开放问题是无限物品情况；② 对部分信息下的行为分析仅覆盖三种典型策略，未考虑更复杂的学习或推理模型；③ 研究集中在纯策略纳什均衡，混合策略或随机化机制尚未讨论。

---

## 477. Improved Pass Complexity for Semi-Streaming Matching in General Graphs via Budgeted Price Approach

**arXiv ID:** 2610.22718 | [PDF](https://arxiv.org/pdf/2610.22718v1)

**作者:** Tatsuya Terao `[一作]` `[通讯]` (Kyoto University), Tatsuya Terao (Kyoto University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种新的确定性半流式算法，用于在一般图中以 O(1/ε²) 次通过完成 (1‑ε)-近似最大匹配，并且仅需 O(n/ε²) 单词空间；

**💡 创新点**

创新点在于引入“预算价位（budgeted price）”方法，构造价格向量与预算函数，使得在每一步仅需一次常数因子近似匹配即可显著提升通过次数，打破了之前 O(1/ε⁴) 的界限；

**🔧 技术方法**

核心技术是：1) 价格向量与残差定义；2) 预算函数与能量/势能概念；3) Probe 过程通过价格逼近和一次性高质量匹配迭代；4) 用穷举搜索在子图中找到近似最优价格；5) 二分搜索确定最大匹配大小；

**📊 数据集**

本文未使用具体实验数据集，全部为理论分析与证明；

**📈 对比分析**

与此前的多项式 (1/ε) 通过算法相比，本文在通过次数上达到 O(1/ε²)，与最优的 O(1/ε²)（如 Assadi 等人对二分图的拍卖算法）相匹配；

**⚠️ 局限性**

主要限制是实现过程中需要对子图进行穷举搜索，导致运行时间呈指数级；作者指出可通过进一步改进使运行时间多项式化，但仍需额外工作。

---

## 478. From Memory Hits to Answerable Evidence: Query-Conditioned Memory Compilation for Long-Term Agent Memory

**arXiv ID:** 2610.23313 | [PDF](https://arxiv.org/pdf/2610.23313v1)

**作者:** Shangze Li `[一作]` (University of Science and Technology of China), Tong Xu `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种双平面（Address‑Evidence）框架 MemCompile，用以将长时会话记忆的定位与答案所需证据的编排分离开来；

**💡 创新点**

创新点在于：①源归一化（source‑normalization）将多渠道检索结果聚合到同一 SourcePacket，避免冗余；②受限 Cell 放置与句子级检索相结合产生多视角候选；③Evidence Plane 的补充调度与前沿扩展实现预算内的互补证据选择与编译；

**🔧 技术方法**

技术手段包括：构建源关联的原子描述符、容量受限的 Cell 路由、BM25 与语义检索的多通道联合、源归一化边界、句子级候选调度与前沿增量、以及最终的 SourcePacket 级别的证据束编译；

**📊 数据集**

数据集：LoCoMo（多会话长时对话记忆）和 LongMemEval‑S（持续交互记忆评估）；

**📈 对比分析**

与 MemoryBank、Mem0、AtomMem 等多种基线在同一评测协议下对比，MemCompile 在 LoCoMo 上实现 80.52% Judge Accuracy，F1 55.94，BLEU‑1 48.65，且在成本‑质量曲线中显著领先，LongMemEval‑S 亦表现出跨基准的优势；

**⚠️ 局限性**

局限性在于评测仅覆盖两套基准，且使用的是专有 LLM，未验证在更广泛领域或开源模型上的可复现性与进一步的令牌消耗优化空间。

---

## 479. Towards High Semantic Fidelity: Hyperdimensional Symbolic Messages in Multi-Agent Communication

**arXiv ID:** 2610.23092 | [PDF](https://arxiv.org/pdf/2610.23092v1)

**作者:** Shoucheng Song `[一作]` (Beijing Jiaotong University), Kai Lv `[通讯]` (Beijing Jiaotong University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新型多智能体通信格式Hyperdimensional Symbolic Messages（HSM）及其框架Herm，解决多智能体通信中的语义保真问题。

**💡 创新点**

创新点包括：①基于高维符号编码的结构化实体-属性-数值编码管线，显著提升传输保真；②接收端的ego-binding与双分支相似度感知融合机制，确保转换保真；③迭代解码实现显式消息-语义转换，兼容LLM。

**🔧 技术方法**

使用高维向量计算（绑定、捆绑、解绑定）和HRR框架，结合滑动窗口动作序列编码、相似度权重融合以及循环神经网络做决策。

**📊 数据集**

在SMAC、SMACv2、LBF、Hallway等标准MARL基准上评估。

**📈 对比分析**

与MAIC、T2MAC、CoDe、TGCNET等基线以及CroMAC、AME等鲁棒基线进行对比，Herm在无噪声与随机/对抗噪声环境下表现最优，收敛更快、完成率更高，且在传输和转换保真指标上均领先。

**⚠️ 局限性**

局限性：编码模块与策略优化未耦合，缺乏端到端联合训练；维度编码机制缺乏自适应性，未来需加入自动属性选择与动态编码策略。

---

## 480. CMM-MP: A Novel Memory Pooling Architecture using Purpose-built Memory Device with CXL and Ethernet

**arXiv ID:** 2610.22716 | [PDF](https://arxiv.org/pdf/2610.22716v1)

**作者:** Michael Wasef `[一作]` (Samsung Semiconductor Inc.), Ramdas Kachare `[通讯]` (Samsung Semiconductor Inc.)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了CMM‑MP，一种将CXL接口与以太网后端相结合的专用内存池化架构，实现了大容量（TB级）内存池化，并能在不修改应用程序的情况下被CPU/GPU使用。

**💡 创新点**

创新点在于首次将CXL端点与高带宽以太网网络集成到单个设备中，自动完成内存池的发现、映射与管理，既保持传统的load/store语义，又大幅降低远程访问延迟和提升吞吐量。

**🔧 技术方法**

核心技术包括：CXL 1.1/2.0 PCIe Gen5×8端点、200 Gbps双端口以太网后端、ARM控制核驱动的自动发现与PAMT映射、可靠的以太网传输层（重发、序号、重排序）以及DDR4内存控制。

**📊 数据集**

使用的数据集和基准包括：Valkey键值存储与YCSB工作负载、SIFT向量数据集（1 M条）用于DiskANN近似最近邻搜索，以及Intel MLC 内存基准测试。

**📈 对比分析**

实验方法是将CMM‑MP与传统RDMA、系统本地内存进行对比，测量延迟、吞吐量及实际应用性能；结果显示CMM‑MP在本地访问时延近似系统内存，远程访问延迟约2.7 µs、吞吐量最高23 GB/s，整体比RDMA在延迟上提升≈50%，在真实工作负载下本地池几乎无性能损失，远程池性能下降≤30%，而DiskANN在远程池上仍比SSD快约3倍。

**⚠️ 局限性**

主要局限包括：远程访问仍受以太网交换机延迟（≈800 ns）影响，远程带宽仍低于本地DDR；当前实现采用静态池分配，缺乏动态可变大小和更细粒度的内存管理；在超出机架规模的扩展、设备ID映射和大规模多主机协调方面还有待进一步研究。

---

## 481. BOLT: Multi-Bottleneck Bound-Guided Learning for Flexible Job-Shop Scheduling with Time Lags and Resource-Coupled Transport

**arXiv ID:** 2610.22433 | [PDF](https://arxiv.org/pdf/2610.22433v1)

**作者:** Ziheng Zhang `[一作]` (Singapore Institute of Technology), Boon Ping Gan `[通讯]` (D-SIMLAB Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 BOLT（Bound‑Oriented Learning for transport‑coupled scheduling）——一种针对含时间延迟与资源耦合运输的灵活车间调度的强化学习调度器，能够在毫秒级别完成每个决策并在断路后秒级重排。

**💡 创新点**

创新点：① 设计了可在任何可达子调度状态下保持可行且终止时等于完成时间的多瓶颈下界（链式、机能、车队容量三项），并将其用作奖励、候选动作价格、根节点证明以及 CP‑SAT 约束；② 通过候选动作价格实现了“行动价格”指导，比单一奖励或手工奖励更有效；③ 将机组与车辆决策分离为两个输出层，实现可独立调优和回滚；④ 在无求解器的情况下即可给出完成调度相对最优的证据。

**🔧 技术方法**

技术手段：基于 Proximal Policy Optimization（PPO）/Multi‑Agent PPO 的强化学习；双注意力编码器；基于下界差值的奖励和行动价格信号；解析信用（analytic credit）对优势函数进行对比；与 CP‑SAT 参考模型、遗传算法以及外部基准进行对照；使用 PPVC‑T 生成器产生的工业级实例。

**📊 数据集**

数据集：PPVC‑T 生成器（含 10、20、50、80 模块的工业级预制构件实例）以及 5 个 50/80 模块的生产批次；此外利用 Link 等人的外部基准数据进行一致性检验。

**📈 对比分析**

比较方法与性能：与 9 组优先级调度规则、遗传算法、热启动 CP‑SAT、合并头控制以及外部基准进行对比。BOLT 在所有网格单元均优于最佳调度规则，且在大部分规模与负载条件下超越遗传算法（在搜索时间限制内）。在 80 模块三车队场景下，BOLT 产生的调度比 1 小时限 CP‑SAT 的最优解更短。零样本迁移至更大批次时仍保持显著优势。因子化控制在训练规模下与合并头相当，在生产规模上略逊，但可单独调整。每个事件决策仅需毫秒级时间，重排在断路后可在秒级完成；根节点下界可在不使用求解器的情况下给出相对最优度 <10% 的证明。

**⚠️ 局限性**

局限性：仅针对确定性模型，假设加工、等待与行驶时间已知；未考虑随机到达、车辆路径规划、交通拥堵、异构车队、有限缓冲区以及无目的地机房预留等真实工厂特征；根节点下界在大批次、低车队负载时松散；性能受车队负载阈值影响，需要在更广泛场景中验证。

---

## 482. Agentic In-Context Learning: What Algorithms Can Transformers Learn in Closed-Loop Interaction?

**arXiv ID:** 2610.23051 | [PDF](https://arxiv.org/pdf/2610.23051v1)

**作者:** Shiguang Wu `[一作]` (Tsinghua University), Quanming Yao `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并研究了 Agentic In-Context Learning (Agentic ICL)，即固定参数 Transformer 通过行动主动塑造上下文，并证明其能够实现有限时域的递归学习算法；通过监督蒸馏与奖励训练验证 Transformer 在该框架下的执行与决策差异。

**💡 创新点**

创新点包括：①把交互式学习拆解为状态、更新与决策三部件，统一覆盖主动学习、Bandit 与在线 RL；②构造可压缩成因果 Transformer 的正则有限时域递归，实现对任何正规递归的近似；③理论分析过程监督（process reward）是否保持终端最优行为，并给出潜在基调修正的条件；④实验验证不同奖励目标导致 Transformer 选择不同的候选算法，并在自然语言工具使用中保持差异。

**🔧 技术方法**

技术主要包括：Transformer 编码与序列化、正则化递归到 Transformer 的构造证明、监督蒸馏、潜在基调修正（potential-based shaping）、行为诊断（候选算法间距离度量）以及对小型工具使用环境（BTS）与预训练语言模型（Qwen2.5-1.5B-Instruct）进行微调。

**📊 数据集**

使用的主要数据集为：①自定义的 Multi‑Round Bayesian Tool Selection (BTS) 任务，用于生成可精确计算最优策略的交互式环境；②预训练 LLM 的微调任务，利用自然语言描述的工具调用与奖励回合；实验中随机生成任务参数、工具观测、预算与折扣等，保持任务多样性。

**📈 对比分析**

比较方法：在 BTS 环境下对比 Transformer 与三种目标（终端奖励、一步过程奖励、过程奖励+潜在基调）以及对三种候选算法（Bayes DP、KG、Entropy、Oracle）的相对距离；在 LLM 微调实验中比较三种训练目标下的返回与与候选算法的相似度；性能方面：终端奖励训练得到的模型与 Bayes DP 的行动概率差距最小，奖励一阶过程训练则最接近 KG；不同目标的训练导致返回在 5–10% 之间波动，且在文本改写下保持一致。

**⚠️ 局限性**

限制：①理论假设正规性与可微分约束，未证明优化收敛；②实验仅在结构化、有限时域的合成任务与小型工具集上验证，难以推广到开放式长时序任务；③奖励与过程监督的设计与评估依赖人工设定的验证器，可能不具备鲁棒性；④未验证 Transformer 在压缩上下文、长交互或未知工具语义下的表现。

---

## 483. Towards Agentic Proof Harness Engineering

**arXiv ID:** 2610.23241 | [PDF](https://arxiv.org/pdf/2610.23241v1)

**作者:** Rohith Kumar Shanmuganathan `[一作]` (Carl von Ossietzky Universität Oldenburg), Heike Wehrheim `[通讯]` (Carl von Ossietzky Universität Oldenburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个代理式证明框架，自动生成 C 项目的证明 harness，帮助验证历史断言错误。

**💡 创新点**

引入独立的形式化安全和精度守卫，保证自动生成的抽象既无误又足够精确，并验证了在 curl 与 libuv 上的可行性。

**🔧 技术方法**

使用了大型语言模型（如 GPT‑4）作为代理，结合 CBMC、KLEE、UAutomizer、CPAchecker 等正式验证器，以及动态与静态分析工具进行守卫检查。

**📊 数据集**

使用了 curl（4500+ 文件、630k 行）和 libuv 两个真实项目，并在其中的历史断言违例（curl #22785、#23292，libuv #5021）作为测试案例。

**📈 对比分析**

通过在原始和修复版本上分别运行代理生成的 harness，利用 CBMC 检测到所有历史错误；在其他验证器上复用 harness，发现跨验证器兼容性有限，但部分工具也能检测同一错误。性能方面，验证时间在 5 分钟内完成。

**⚠️ 局限性**

限制包括对代理的非确定性、对构建配置的搜索耗时、缺乏终止保证、跨工具兼容性差以及在动态依赖抽象验证时的困难。

---

## 484. Disentangling Language Effects on Tool Admission and Post-Retrieval Safety

**arXiv ID:** 2610.23093 | [PDF](https://arxiv.org/pdf/2610.23093v1)

**作者:** Aditya Nawal `[一作]` (National University of Singapore), Mohan Gurusamy `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在控制的URL检索环境中，系统评估多语言下LLM代理在工具调用与检索后生成中的安全行为；

**💡 创新点**

创新点在于提出并量化了“跨语言安全损失”（SLIP），分别测量请求语言对工具调用的影响以及页面/响应语言对后续有害生成的影响，并区分真实拒绝与能力失败；

**🔧 技术方法**

使用了受控检索流水线、请求/页面/响应语言独立切换、Google/DeepL等机器翻译、HarmURLBench样本、GPT‑4o与多语言评审员、自动化行为审计与统计分析；

**📊 数据集**

使用的数据集为HarmURLBench（296条有害请求），并通过翻译扩展到七种语言，结合mC4语言资源分组；

**📈 对比分析**

通过对比不同语言下工具接收率与有害率，发现非英语请求平均工具接收率从46.9%提升至67.4%，在闭源模型中，要求以页面语言回复时有害率提升至57.4%；

**⚠️ 局限性**

局限包括翻译质量与文化等价性不确定、部分语言缺乏原生评审、自动化标签与真实安全性之间的观察性关联、不同模型推理设置不一、未能分离脚本、规模、训练数据等混杂因素。

---

## 485. Bio-Inspired Magnetic Navigation Using Gradient Descent With Rao-Blackwellized Particle Filtering

**arXiv ID:** 2610.23097 | [PDF](https://arxiv.org/pdf/2610.23097v1)

**作者:** Zayed Herma `[一作]` (Univ. Grenoble Alpes), Jean-Michel Leger `[通讯]` (Univ. Grenoble Alpes)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于梯度下降的生物启发磁导航方法，利用Rao‑Blackwellized粒子滤波器（RBPF）在线估计局部磁梯度和位置信息，从而实现无磁图、GPS‑拒绝环境下的长距离导航。

**💡 创新点**

将梯度估计与粒子滤波耦合，使用RBPF同时估计梯度与位置；采用梯度下降控制引导方向，消除对全局磁图的依赖；在无地图环境下实现有效的导航。

**🔧 技术方法**

Rao‑Blackwellized粒子滤波器（RBPF）、梯度下降控制、磁参数测量、Mercator投影、IGRF‑14地磁场模型。

**📊 数据集**

IGRF‑14地磁场模型，模拟海洋导航区域（起点-71.9°、47.3°，终点0°、-34°），使用磁倾角和方位角。

**📈 对比分析**

与假设恒定梯度的基准方法比较；在理想条件下RBPF始终收敛并逼近目标，基准方法在大尺度导航中偏离；在噪声漂移实验中RBPF仍能有效估计梯度并抑制漂移，定位误差约3 km；与纯积分梯度控制误差可达500 km形成对比。

**⚠️ 局限性**

多地点可能共享相同磁参数导致误定位；局部梯度弱时定位精度下降；需要对参数进行仔细调优；未考虑时间变化磁场扰动；未验证复杂室内环境。

---

## 486. Aligning Language Models Towards Counterfactual Invariance

**arXiv ID:** 2610.22942 | [PDF](https://arxiv.org/pdf/2610.22942v1)

**作者:** Basar Demir `[一作]`, Chirag Nagpal `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在开放式生成任务中对无关人口统计信息的反应一致性问题，提出了多种奖励匹配策略来减弱反事实偏差。

**💡 创新点**

引入奖励期望匹配和变换奖励期望匹配两种跨提示奖励对齐目标，并构建大规模无关/相关对照提示数据集。

**🔧 技术方法**

基于GRPO的强化学习与Skywork奖励模型，设计了响应匹配、奖励对称化、奖励期望匹配、变换奖励期望匹配四种训练目标。

**📊 数据集**

使用Qwen3-4B-Base模型，在10k条无人称提示上进行监督微调，随后构建约2500对反事实提示（性别、种族）并标注为 invariant/dpendent/agnostic。

**📈 对比分析**

通过 DiscrimEval、Difference Awareness、BBQ 等受限性测试和 LLM‑as‑Judge 的开放式评估，对比四种目标与基线；变换奖励期望匹配在保持奖励水平的同时显著降低反事实偏差并获得最高开放式评估分数。

**⚠️ 局限性**

仍无法完全恢复初始化器的 invariance，奖励模型对人口统计仍敏感，且对混合数据训练时可能牺牲对相关提示的表现；标注方案主观且未公开评测细则。

---

## 487. MergeShuffle: Reconstruct Shared Libraries for Improved Performance and Security

**arXiv ID:** 2610.22952 | [PDF](https://arxiv.org/pdf/2610.22952v1)

**作者:** Huaifeng Zhang `[一作]` (Chalmers University of Technology), Ahmed Ali-Eldin `[通讯]` (Chalmers University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一款共享库重构工具，利用已识别的文件范围剔除未使用代码，经过合并与重排后生成功能完整、体积更小、内存占用更低的共享库。

**💡 创新点**

创新点在于将文件范围重排问题建模为旅行商问题，采用启发式求解；提出两阶段合并‑重排策略；实现无需源代码、可与任意使用分析工具配合，并能生成可共享的重构库。

**🔧 技术方法**

使用 ELF 结构解析、文件范围合并、线性与 TSP 启发式 Shuffle、程序头重映射、INT3 填充；结合 RTrace 采集使用范围、LD_DEBUG 监测冷启动时间。

**📊 数据集**

评测数据集包括 9 个真实世界应用（Torch 推理/训练、ImageMagick、Nmap、Poppler、四个 DeathStarBench 微服务）以及 XZ 后门安全案例。

**📈 对比分析**

通过 20 次重复测量原始与重构库在文件大小、峰值内存和冷启动运行时的差异进行比较；在最激进设置下文件大小下降 40‑66%，内存占用下降 4‑18%，运行时缩短 2‑33%，且成功移除 XZ 后门，提升安全性。

**⚠️ 局限性**

局限性：需预先获取使用文件范围（需跟踪或分析）；g 阈值调节平衡合并与内存；合并可能导致页对齐冲突；TSP 启发式在极大或多页范围时可能产生次优布局；未处理二进制重定位或复杂符号表更新等细节。

---

## 488. SETAREH Transform: A Sorting-Assisted Haar-Wavelet Sparsification Method for Approximate Inner-Product Computation

**arXiv ID:** 2610.23338 | [PDF](https://arxiv.org/pdf/2610.23338v1)

**作者:** Seyed Kamaledin Setarehdan `[一作]` `[通讯]` (University of Tehran), Seyed Kamaledin Setarehdan (University of Tehran)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Haar小波的排序辅助稀疏变换SETAREH，用于在固定权重向量下对大量输入向量进行近似内积，显著减少需要存储或传输的样本数。

**💡 创新点**

创新点在于利用内积对称不变性与排序+多级Haar变换，将权重向量能量集中，使得仅保留少数变换域系数即可逼近原内积。

**🔧 技术方法**

采用排序、Haar离散小波变换、截断保留K个系数以及条件中位数相对误差（CMRE）评估方法。

**📊 数据集**

实验数据集包括合成高斯向量和APPA-REAL人脸年龄估计数据集的4096维特征。

**📈 对比分析**

通过与未压缩的全维内积以及SETAREH-3、SETAREH-5两种变体比较；在K≥30时SETAREH-5在CMRE上比SETAREH-3低1-2个数量级；在APPA-REAL上仅保留75系数即可将CMRE降至0.13%。

**⚠️ 局限性**

局限性：需要额外的排序与变换开销，置换元数据导致单样本时存储反而增大；未评估算术运算量或硬件延迟，仅在固定或缓慢变化权重场景有效。

---

## 489. FLARE: Physics-Grounded Evaluation of Language-Model Agents for Feeder Restoration with Degraded Observations

**arXiv ID:** 2610.23251 | [PDF](https://arxiv.org/pdf/2610.23251v1)

**作者:** BHVSP Subrahmanyam `[一作]` (Mahindra University), Neeraj Choudhary `[通讯]` (Mahindra University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于物理约束的分配网后隔离恢复基准FLARE，提供交互式环境供语言模型在不完整信息下做决策

**💡 创新点**

首次将AC可行性、拓扑与安全守卫与语言模型决策流程相结合，形成完整的评估框架和多样化观察设置

**🔧 技术方法**

利用语言模型推理、序列决策、AC功率流守卫、模拟器与安全检查

**📊 数据集**

800个真实德国中压时序案例（CIGRE、SimBench网络）与多观察设定，另包含150个独立案例及200/400/100案例的操作、网络类型对照

**📈 对比分析**

对比5种LLM（Llama3.1-8B、Mistral24B、Qwen2.5-3B/7B/14B）在5观察设置下的均值、P95 regret、恢复率和安全违规率，平均regret 0.082–0.148，Qwen7B表现最佳

**⚠️ 局限性**

模型在信息退化下表现不一致，安全违规虽被守卫阻止但仍缺乏真实三相/短路验证，且未覆盖完整FLISR流程，限制了实际部署的可行性

---

## 490. RealCLI: Toward a Unified Evaluation of Human-Agent Collaboration for CLI Agents

**arXiv ID:** 2610.22507 | [PDF](https://arxiv.org/pdf/2610.22507v1)

**作者:** Yunqi Liu `[一作]` (IQuest Research), Bryan Dai `[通讯]` (IQuest Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了REALCLI这一配对评测基准，用同一组真实命令行任务同时评估代理在自主执行和人机协作模式下的任务完成率和协作质量。

**💡 创新点**

创新点包括：①构建了基于HCI理论的五维协作框架（可理解性、可引导性、观察与恢复、信任校准、安全风险意识）并细化为22个子维度和51个情境模板；②引入脚本驱动的用户模拟器，实现信息逐步发布和交互探针；③设计了联合协作得分（Joint Collaboration Score）将任务成功率与协作质量乘积化，强制性惩罚协作不足；④在101个任务上完成了严格隔离容器化评测，提供了可复现的基准。

**🔧 技术方法**

使用的大模型包括11个闭源（如Claude Opus 5、GPT‑5.6）和11个开源（如DeepSeek V4.1 Flash、Qwen3.8 Max）配置，并结合三种代理框架（Claude Code、Terminus‑2、Codex）进行实验；评测通过环境终态验证器和交互轨迹检查清单实现。

**📊 数据集**

数据集来源于真实CLI日志压缩，经过任务模式化、技能化、实例构造和交互脚本化，最终形成涵盖10大类、52子域、101任务的评测集；每任务附有51个协作情境模板和1658个检查项。

**📈 对比分析**

对比方法：在同一任务下对比EXEC（完整规范）和COLLAB（逐步信息+用户交互）两模式；通过任务成功率（R_EXEC、R_COLLAB）和协作质量（S_COLLAB、J_COLLAB）进行评估。实验结果显示：最佳配置在COLLAB模式下平均得分49.2%，J_COLLAB最高仅37.6%，表明任务完成与协作质量显著脱节；大多数模型在可理解性与观察恢复上表现良好，但在可引导性、信任校准和安全风险意识上表现较差。

**⚠️ 局限性**

局限性：①用户交互由脚本驱动，可能不完全反映真实人类沟通细节；②评测聚焦于CLI任务，未涵盖图形界面或网络交互场景；③只评估已知模型配置，未来新架构或更大规模模型可能表现不同；④联合得分采用乘积公式过于严格，可能低估模型在某些细节协作上的潜力。

---

## 491. MambaAct: Efficient Mamba Action Experts for Flow-Matching Vision-Language-Action Models

**arXiv ID:** 2610.22951 | [PDF](https://arxiv.org/pdf/2610.22951v1)

**作者:** Khuyen Pham `[一作]` (MBZUAI), Cesare Stefanini `[通讯]` (MBZUAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

将π_0.5模型中的Transformer动作专家替换为Mamba专家，从而在保持预训练视觉‑语言编码器不变的情况下加速流匹配VLA的推理。

**💡 创新点**

创新点在于仅改造频繁评估的动作专家，使用Mamba的选择性状态空间机制实现10×参数压缩，同时通过轻量级上下文压缩器与门控FiLM实现对视觉‑语言上下文的高效调节。

**🔧 技术方法**

采用的技术包括Mamba序列混合器、上下文压缩器（Context Compressor）、门控FiLM、流匹配动作生成以及π_0.5的标准框架。

**📊 数据集**

实验数据集包括LIBERO、LIBERO‑Plus以及两项UR5机器人真实任务（管道放置与螺母倒料）。

**📈 对比分析**

与π_0.5、SnapFlow、Shallow‑π等方法对比，MambaAct在LIBERO上平均成功率达97.8%（比π_0.5高1.08个百分点），推理延迟下降约53%，并在真实任务的OOD测试中成功率显著提升。

**⚠️ 局限性**

局限性在于仅在两项UR5任务上验证，未评估与视觉token剪枝、降采样步等其它效率技术的兼容性，以及对门控FiLM和Mamba状态动态的深入分析。

---

## 492. Semantic-Critical Regions for Video-to-Audio Generation

**arXiv ID:** 2610.23021 | [PDF](https://arxiv.org/pdf/2610.23021v1)

**作者:** Yun Fang `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出SCR-V2A框架，通过弱监督视觉定位方法提取音频相关的细粒度语义关键区域（SCR），并将其作为视觉引导融入音频扩散模型，实现视频到音频的生成。

**💡 创新点**

创新点在于：①将CAMs用于定位音频相关视觉证据，获得细粒度空间指导；②采用双重适配器编码（视频和SCR热图）实现视觉信息的融合；③仅使用少量可训练参数，在保持高质量和跨模态对齐的同时显著提升了模型的通用性。

**🔧 技术方法**

使用的技术包括：弱监督视觉定位+CAM，适配器（adapter）模块，双重视觉编码，控制适配器（control adapter）与预训练音频潜在扩散模型（LDM）的融合，声学标签的多标签分类训练。

**📊 数据集**

数据集：VGGSound（310类，200K 10秒视频）作为训练/验证；Countix-AV作为跨域测试。

**📈 对比分析**

与SOTA方法相比，SCR-V2A在FD、FAD、KLD等分布匹配指标以及IS、ImageBind、DeSync等对齐指标上均取得更佳或相当的成绩；同时训练参数量仅约195M，显著低于其他大型模型，且在域迁移时表现更稳健。

**⚠️ 局限性**

局限性：①主要关注空间信息，未充分结合细粒度时序控制；②对CAM定位的依赖可能导致在视觉信息模糊或多源交互场景中表现受限；③仍依赖预训练的音频扩散模型，缺乏端到端自监督学习的潜力。

---

## 493. The Reliance Hypothesis: Chain-of-Thought Faithfulness Rises as Language Resource Level Falls

**arXiv ID:** 2610.22841 | [PDF](https://arxiv.org/pdf/2610.22841v1)

**作者:** Lucas Resck `[一作]` (University of Cambridge), Anna Korhonen `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大型语言模型在多语言环境下的链式推理（CoT）可信度，并探讨其与语言资源水平、任务类型（知识密集 vs 推理密集）之间的关系。

**💡 创新点**

创新点在于首次将CoT可信度与语言资源水平及任务类型关联，提出“依赖假设”（Reliance Hypothesis）并验证其在多语言、多模型下的稳健性，揭示低资源语言中模型更依赖推理的现象。

**🔧 技术方法**

使用了四种CoT干预技术（删除、错误注入、填充、重排）以及三种可信度度量（基于概率偏移、误差传播、填充词效应），并采用线性回归和中介分析评估资源水平、置信度对可信度的影响。

**📊 数据集**

数据集为MMMLU（Multilingual Massive Multitask Language Understanding），覆盖八种语言（资源跨度近八个数量级），包括知识密集和推理密集子任务的多项选择题。

**📈 对比分析**

通过对八种语言、四个开源模型（4–8B参数）进行交叉实验，发现知识密集任务中低资源语言的CoT可信度显著提升，而推理密集任务则无此趋势；模型置信度部分解释了资源效应，但未能完全消除知识–推理差距。实验显示，在低资源语言中，模型更频繁地依赖CoT，可信度提高约4%（英语）到35%（约鲁巴）。

**⚠️ 局限性**

局限性包括：语言样本有限（尤其低资源侧），资源水平指标为token计数的代理，MMMLU翻译可能带来文化偏差，低资源语言生成有效CoT比例低，度量方法仅捕捉依赖关系而非完整可信度，评估模型规模受限于开放权重4–8B模型。

---

## 494. Thought Blocks: Non-Autoregressive Latent Communication for Multi-Agent Systems via Trajectory-to-State Compilation

**arXiv ID:** 2610.23056 | [PDF](https://arxiv.org/pdf/2610.23056v1)

**作者:** Zhenhan Guan `[一作]` (ShanghaiTech University), Zheng Tian `[通讯]` (ShanghaiTech University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种无序列化的隐式通信机制—Thought Blocks，直接从发送方状态生成接收方可用的 KV 记忆块。

**💡 创新点**

创新点是将消息构造与表示分离，利用轨迹到状态编译实现一次性生成连续状态，显著减少中间推理步骤。

**🔧 技术方法**

采用了 factorized continuation state（seed+context bank）、一键式 KV 生成器、状态级蒸馏以及 Transformer 的 frozen‑backbone 渲染。

**📊 数据集**

使用了 ARC、OpenBookQA、GSM8K、MMLU、GPQA、MedQA、BoolQ 与 QuAIL 等八个推理与问答数据集共 11 个任务配置。

**📈 对比分析**

与文本式多智能体、自动回归隐式通信以及单一代理基线对比，Thought Blocks 在保持 1–2 分点精度的同时，平均 8.6 倍延迟加速、84.8% AR 步骤和 45.7% 文本代价下降。

**⚠️ 局限性**

局限在于对较大块尺寸的 KV 负载上升、跨模型异构体系的适配未评估、训练时仍需自动回归教师、隐式 KV 消息不可解释。

---

## 495. Robotic Spatiotemporal Editing: A Unified Interface for Controllable Action Generation via Masked Action Reconstruction

**arXiv ID:** 2610.23002 | [PDF](https://arxiv.org/pdf/2610.23002v1)

**作者:** Yanbang Li `[一作]` (Tsinghua University), Mengdi Xu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了统一的空间-时间动作编辑框架，通过遮蔽动作空间(MAP)把多种控制信号统一为部分动作规格，然后使用动作填补(Inpainting)恢复完整轨迹。

**💡 创新点**

创新点在于将不同的空间-时间参考（关键点、流、姿态、方向等）映射到统一的可遮蔽动作空间，形成动作填补问题；同时设计了进度引导滑动窗口、多模态联合训练和训练无关的RePaint填补策略。

**🔧 技术方法**

技术包括基于DiT的扩散变压器、进度预测模块、进度引导滑动窗口、遮蔽动作编码器、联合训练中的结构化与随机遮蔽、RePaint在推理阶段的反向填补。

**📊 数据集**

使用了仿真数据集ManiSkill、LIBERO-10以及真实机器人（Franka R3）结合RealSense和ZED摄像头采集的视觉/传感器数据。

**📈 对比分析**

与Diffusion Policy、DiT Policy及直接条件化的STC进行对比；在ManiSkill上取得92.67%成功率，LIBERO-10上64.67%成功率，控制误差CE显著下降，真实世界平均位置误差44.3mm、旋转误差7.8°，多参考条件下成功率超过93%。

**⚠️ 局限性**

局限性包括填补实现尚未最优，未探索CFG/ControlNet等更高效的控制方式；稀疏、后期单点约束效果差；对非结构化控制规格的零样本适应性还有待提升。

---

## 496. Self-Reflective Chain-of-Thought (SRCoT): Evidence-Based Text Generation via a Closed-Loop Retrieve--Reflect--Revise Pipeline on Edge LLMs

**arXiv ID:** 2610.23311 | [PDF](https://arxiv.org/pdf/2610.23311v1)

**作者:** Fabian Broszio `[一作]` (Freie Universität Berlin), Adrian Paschke `[通讯]` (Freie Universität Berlin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 Self-Reflective Chain-of-Thought (SRCoT) 架构，将 LLM 文本生成拆分为检索–反思–修订闭环，强调可追溯、基于本地 SOP 的可靠文本生成。

**💡 创新点**

创新点在于将环境状态、持久知识、经验和短期记忆分别封装为机器可读的存储，并通过闭环控制和自我反思判定是否检索、修订或放弃，以实现可审计、可解释的边缘 LLM 文本生成。

**🔧 技术方法**

核心技术包括 BM25 检索、结构化工作空间、可选 Judge 步骤（声明-证据映射）、闭环控制器和两轮生成（草稿+修订），实现本地离线推理。

**📊 数据集**

使用的“数据集”是本地文档集合（SOP、检查表、组织规则）以及合成的 Munich‑Sim 事件日志与模拟通话记录，用于检索和验证。

**📈 对比分析**

本文未给出系统与基线的定量对比，主要通过案例研究和可追溯性指标（证据标注、缺失标记、控制决策）展示可行性；性能表现未定量评估。

**⚠️ 局限性**

局限包括缺乏大规模真实评测、检索和证据绑定误差、对模型大小和提示的依赖、缺少系统级性能指标，以及仅在合成场景下验证，未证明在临床或运营环境中的安全性。

---

## 497. Initiating Removal from Operational Symptoms in Federated Vehicular Networks

**arXiv ID:** 2610.22584 | [PDF](https://arxiv.org/pdf/2610.22584v1)

**作者:** Zihao Ding `[一作]` (South Dakota State University), Qiang Duan `[通讯]` (Pennsylvania State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种联邦学习环境下的车队联邦未学习方法，可通过延迟症状推断错误贡献并在聚合规则中即时消除。

**💡 创新点**

创新点在于将症状投影到最近增量子空间，用训练曲率估计实现一次性反向传播评分，并在聚合中采用两阶段修正（抑制+确认消除），不需存储历史检查点。

**🔧 技术方法**

采用梯度投影、曲率矩阵近似、一次性反向传播评分、两层纠正以及在线构建子空间等技术。

**📊 数据集**

使用了Carla仿真中的Foggy Cityscapes、ACDC、BDD100K三大驾驶数据集，以及MentorPi机器人车在物理轨道上的实验。

**📈 对比分析**

与FedRecover、FedOSD、FedEditor、FLForensics、FLTracer等基线对比，八/九种设定下残余危害下降43–61%，并保持在 Retrain 的0.36分之内；延迟低，通信成本可调，整体性能优越。

**⚠️ 局限性**

局限包括对子空间维度和阈值kappa的敏感性，无法完全处理子空间外残差，对攻击者合成症状的鲁棒性待进一步验证，且在极大车队或高通信延迟场景下可能受限。

---

## 498. Event-STVG: Spatio-Temporal Video Grounding for Long-Form Event Reasoning

**arXiv ID:** 2610.22444 | [PDF](https://arxiv.org/pdf/2610.22444v1)

**作者:** Yizhe Zhang `[一作]` (Tsinghua University), Juanzi Li `[通讯]` (Tsinghua University)

**通讯引用:** 15496 | [OpenAlex ID](https://openalex.org/A5003324011)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了新的时空视频定位任务Event-STVG，要求模型在长视频中同时定位事件时间区间和参与角色的空间轨迹；

**💡 创新点**

创新点在于将传统基于描述的定位转向事件级推理，加入重复事件区分、语义角色推断和多目标联合定位；

**🔧 技术方法**

使用了大规模人工标注、自动跟踪辅助、VLM与专门STVG模型的评估、以及三层次瓶颈分析技术；

**📊 数据集**

基准数据集涵盖4,016条事件，来自SoccerNet-v2等足球视频以及篮球、排球、冰球、猎物、速食包装、娱乐节目、电竞等八个领域；

**📈 对比分析**

对10款通用VLM和1款专用STVG模型进行对比，最优模型在10分钟上下文下m‑vIoU仅为5.4%，1分钟场景下大多模型低于1%，跨域性能差异大但总体未达饱和；

**⚠️ 局限性**

局限性包括长上下文导致的事件实例混淆、时空定位不匹配导致的空间推断错误以及多角色联合定位的高难度。

---

## 499. MailRail: Mitigating Incast Congestion with User-Space, Node-Based Flow Control

**arXiv ID:** 2610.22890 | [PDF](https://arxiv.org/pdf/2610.22890v1)

**作者:** Noel Gomillion `[一作]` (Lawrence Livermore National Laboratory), Roger Pearce `[通讯]` (Lawrence Livermore National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了MailRail异步消息框架，并提出了基于节点的信用流控方案，用以缓解大型并行环境中的Incast拥塞。

**💡 创新点**

创新点在于将信用计数从传统的rank级别聚合到共享的节点级别，实现更高的信用利用率、低开销且保持对多种网络结构的适用性。

**🔧 技术方法**

采用MailRail API实现零复制、非阻塞通信，并基于MPI实现信用返回、延迟发送队列等用户空间流控机制。

**📊 数据集**

使用的实验数据集包括合成的对抗性Incast基准、YGM的Reddit评论排序和三角计数跟踪，以及统一Histogram基准，实验环境主要为LLNL的Dane超级机和Lynx实验机。

**📈 对比分析**

与无流控和rank级流控比较，使用弱/强缩放实验评估执行时间和goodput；节点级流控在极端4096:1 Incast下可实现1132×速度提升，并在更广泛的工作负载下几乎无额外开销。

**⚠️ 局限性**

局限性包括在多核共享内存中管理信用时可能出现伪共享开销；节点信用池在极端异构或极高空闲rank比例时仍可能不足；实现依赖MPI，跨网络适配可能受限。

---

## 500. Towards Reconstructing Geographically Diverse Architecture with 3D Foundation Models

**arXiv ID:** 2610.22514 | [PDF](https://arxiv.org/pdf/2610.22514v1)

**作者:** Nicky Kriplani `[一作]`, Hadar Averbuch-Elor `[通讯]` (Cornell University)

**通讯引用:** 1500 | [OpenAlex ID](https://openalex.org/A5043669878)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个跨区域、多样化的3D建筑基准（ArchWorld），对现有3D基础模型进行错误分析，发现显著的地理差异，并提出了无训练的跨视角注意力干预方法以提升重建质量并减小差距。

**💡 创新点**

创新点在于：①首个集地理分布、建筑类型、尺度标注和密集真实重建于一体的户外3D基准；②引入归一化Chamfer距离以消除尺度偏差；③通过语义交叉视角注意力分析揭示误差模式，并设计训练免费干预手段显著提升跨区域性能。

**🔧 技术方法**

使用3D基础模型（VGGT、Depth Anything v3、π^3）进行评估；采用Colmap、MegaDepth-X生成密集重建；使用SAM3进行语义分割和掩码；通过梯度上升对注意力权重进行干预；评估使用CD与归一化CD。

**📊 数据集**

主要数据集为：MegaScenes（用于采样）、MegaDepth-X（已有重建）、自建的ArchWorld基准（396个场景、92K张图片，覆盖8大地理区，含建筑类型、年代和尺度标注）。

**📈 对比分析**

通过对比CD与归一化CD对三种3DFM在不同地区和建筑类型的误差进行分解，发现显著的地理差异；干预后平均归一化CD下降约14%，并将地区差距缩小约40%，在难以重建的地区（如北美、教堂类）提升最为明显。

**⚠️ 局限性**

局限性包括：仅覆盖8个宽泛地理区，未包含室内场景；数据集可能与模型训练集重合，影响评估客观性；仅聚焦于公开可获取的好照片，忽略偏远或非互联网可见建筑；干预虽改善整体质量，但细节仍可能模糊。

---

## 501. QuarkShard: Decomposing Cross-Shard Requests into Single-Input Transactions

**arXiv ID:** 2610.23014 | [PDF](https://arxiv.org/pdf/2610.23014v1)

**作者:** Changyu Duan `[一作]` (Tsinghua University), Zhixuan Fang `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一种将跨分片请求拆分为单输入分片交易的协议，显著降低跨分片与内部BFT执行的开销；

**💡 创新点**

通过专门的请求拆分算法保证跨分片请求的资金原子性或位置原子性，同时提升系统对MEV攻击和洪水攻击的防护；

**🔧 技术方法**

利用BFT共识、跨分片通信机制、单输入拆分算法以及MEV防护技术构建协议；

**📊 数据集**

在AWS四个区域构建的广域网络环境中，使用随机生成的合成跨分片请求数据进行实验评估；

**📈 对比分析**

与Kronos和2PC基准对比，峰值吞吐达到1024k rps，延迟低于4秒，比Kronos提升210.7%，比2PC提升314.8%；

**⚠️ 局限性**

依赖于跨分片通信的可靠性、对恶意节点的假设以及拆分算法对事务完整性保障的局限，可能在极端网络延迟或恶意节点集中时性能与安全性受限。

---

## 502. Provable Compliance: A Protocol for Runtime Compliance Evidence and Transaction-Specific Decisioning

**arXiv ID:** 2610.22913 | [PDF](https://arxiv.org/pdf/2610.22913v1)

**作者:** Jeremy Blackburn `[一作]`, Jason Patrick `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并实现了 ChainIT 的可验证合规框架，基于单独的 Evidence Assertion VDT、Compliance Policy Profile (CPP)、Business Rules Engine (BRE) 与工作流引擎，生成时限签名的 Compliance Decision VDT，并将其作为授权协议（Authority Protocol）的合规谓词，实现在交易或会话依赖事件时的实时新鲜度评估与决策。

**💡 创新点**

创新点包括：
• 将合规证据拆分为可签名、单一观测的 Evidence Assertion，保持来源、时间、置信度与隐私分类；
• 通过可验证的 Compliance Decision Permits 作为授权协议的输入，实现合规与授权分离的零信任模型；
• 引入点对点的新鲜度与状态 epoch 机制，避免一次性授权永续使用；
• 在工作流引擎中引入风险层级与手工审核逻辑，支持 Step‑Up、Review、Hold 等细粒度控制；
• 兼容人类、组织、群体与软件代理的授权与合规评估，统一在同一协议下执行。

**🔧 技术方法**

使用技术：
• Verifiable Digital Token (VDT) 与数字签名，确保可追溯与不可篡改；
• Business Rules Engine (BRE) 对合规谓词进行确定性、可解释的评估；
• ChainIT 工作流引擎负责风险级别分配、决策合成与执行控制；
• Agent Subject Profile 与 Agent Authority 结构实现代理的限定授权；
• Pactvera 交易引擎与 Execution Authorization Credential (EAC) 用于最终支付授权；
• 状态 epoch / 新鲜度规则、基于时间戳的时限与缓存策略；
• 结合身份验证（biometric, gov‑ID）与业务验证（KYC/KYB、专业执照）等多源数据。

**📊 数据集**

主要数据源与集成：
• OFAC / SDN 列表、其他制裁数据库；
• PEP 数据库；
• AML 监控系统产生的交易监控、风险评分、内部案例状态；
• KYC/KYB、身份证明（政府登记、biometric）与专业执照记录；
• 背景调查报告、司法记录；
• 这些数据通过 Evidence Assertion VDT 以单一观测形式嵌入系统。 论文未给出具体公开数据集，而是说明系统可与上述所有第三方源集成。

**📈 对比分析**

评估方法与性能：
• 论文提出一套指标框架（决策完整性、实时性、误报/漏报率、审计检索延迟、隐私披露合规率等），并给出期望的 P50/P95/P99 延迟、误报率、审计检索时间等目标数值；
• 但未提供实验对比或真实部署数据；实验设计建议对比传统一次性授权、持续查询等方案，观察决策完整度、延迟与误报差异。

**⚠️ 局限性**

局限性：
• 仍依赖外部数据源的准确性与可用性，无法消除源错误、延迟或误匹配；
• 合规判断的精确度受机构自行配置 CPP 与工作流的质量影响；
• 对恶意、合谋或内部人员滥用的防护仅限于规则与人工审核；
• 系统安全受 ChainIT 发行方与证据提供方的安全性约束；
• 监管变化、法规更新可能导致 CPP 与规则滞后；
• 论文未展示大规模部署的性能与成本，实际应用需进一步评估。

---

## 503. Next Token or Vector? Learning How Far to Decode

**arXiv ID:** 2610.22788 | [PDF](https://arxiv.org/pdf/2610.22788v1)

**作者:** Sai Sreenivas Chintha `[一作]` (Microsoft), Sulaiman Vesal `[通讯]` (Microsoft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 CADENCE 框架，学习在每一步根据上下文动态决定生成的 token 数量，实现可变粒度的文本生成。

**💡 创新点**

创新点在于使用高斯未来跨度表示来预测不同长度的连续向量，并通过路由器结合上下文和高斯参数自适应选择生成步长，从而在保持语义一致性的同时提升推理吞吐量。

**🔧 技术方法**

采用 Llama‑3.1‑8B‑Instruct 作为冻结的后端，加入 Perceiver‑IO 风格解码器、多个高斯预测头、基于自蒸馏的联合训练以及多分类焦点损失的路由器训练。

**📊 数据集**

训练使用 UltraChat‑ShareGPT 数据集，评估则在 Alpaca、GSM8K 和 HumanEval 三个基准上进行。

**📈 对比分析**

通过与标准单 token 递归解码的对比，使用 token 接受率、完整草稿接受率、平均接受长度和速度提升等指标评测，结果显示 CADENCE 在三任务上分别实现约 1.28×-1.30× 的吞吐量提升，且在适当位置的 token 接受率显著高于固定步长方法。

**⚠️ 局限性**

限制包括路由器可能过于保守导致较短步长选择、实验仅覆盖单一 backbone（Llama‑3.1‑8B‑Instruct），以及系统对后端验证的依赖，未系统评估无验证时的表现。

---

## 504. When is a large language model a good calibration agent? Physics-labeled prompting for budgeted engineering simulator calibration

**arXiv ID:** 2610.22379 | [PDF](https://arxiv.org/pdf/2610.22379v1)

**作者:** Ruo-Qian Wang `[一作]` `[通讯]` (Rutgers University), Ruo-Qian Wang (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究利用物理诊断文本引导的大型语言模型（LLM）生成参数建议，并与传统无导数优化器（如Nelder–Mead、GP‑BO）结合，评估其在有限预算下对工程模拟器（圆柱流LES闭合参数校准和阻尼振荡器逆问题）的校准效率。

**💡 创新点**

①证明物理诊断文本能在前几次评估显著提升校准性能；②提出“warm‑start + 继续”工作流程，将LLM提议的前期优势转化为后期优化器的加速；③系统量化LLM提议的重复性与变异性；④通过提示消融对比验证物理语义的贡献；⑤对LES目标可辨识度进行敏感性分析。

**🔧 技术方法**

使用 ChatGPT‑3.5‑turbo 作为提议生成器；无导数优化器（Nelder–Mead、Covariance‑matrix adaptation、Gaussian‑process Bayesian Optimization）；数值继续（在固定预算后将已评估集交给后续优化器）；统计方法（bootstrap 95% CI、置换检验、Fisher exact、Wilcoxon 符号秩检验）。

**📊 数据集**

• 圆柱流 Smagorinsky LES 闭合参数校准：目标来自相同求解器的高分辨率参考数据；• 随机阻尼振荡器逆问题：30 条不同随机轨迹（k、c 随机抽样）。

**📈 对比分析**

在五个种子上多次重复，比较第一提议成功率、前 5/10/15 评估的中位目标值、达到阈值所需评估数。结果显示：物理标注提示在第 5 评估的成功率为 92%，中位目标值在 5 评估时比消融组低 21.9 倍；在前 15 次评估后停滞，继续策略可在 9 次评估内达到 10⁻⁴ 目标，而纯 LLM 需 12 次；传统无导数优化器在前期性能显著落后。

**⚠️ 局限性**

• 仅使用 5 个种子，低预算下对 LLM 的比较受限；• LES 问题仅能辨识黏度参数，其余两参数几乎无信息；• 仅评估单一 LLM 配置与温度，未探讨模型/温度对变异性的影响；• 手动设定的 B_switch=15，未自适应；• 仅测试 2D LES 与有限维度参数空间，结果不一定适用于更高维度或其他物理问题。

---

## 505. DriveWorld3D Evidence Constrained 3D World Modeling and Generative Simulation for Real-World Autonomous Driving

**arXiv ID:** 2610.22450 | [PDF](https://arxiv.org/pdf/2610.22450v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 506. GR-VIP: Geometric representation conditioning for precise visuomotor policy

**arXiv ID:** 2610.22997 | [PDF](https://arxiv.org/pdf/2610.22997v1)

**作者:** Chanhwi Lee `[一作]` (Pohang University of Science and Technology), Soohee Han `[通讯]` (Pohang University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了 GR‑VIP，一种基于预训练跨视图特征的目标条件视觉运动控制策略，用于实现机器人精确对准和停靠。

**💡 创新点**

创新点在于：①将 Reloc3r 预训练的稠密关系特征（而非最终位姿估计）直接用于策略条件化；②通过扩散模型生成多模态控制序列；③实现毫米级终端精度和无额外训练的零样本迁移。

**🔧 技术方法**

使用技术包括：Reloc3r 视觉编码器、Transformer 视角关系特征融合、基于扩散的动作生成（Diffusion Transformer），以及 DPM‑Solver++ 的在线推理。

**📊 数据集**

数据集：在单一室内环境（4F‑H）收集 150 条成功的遥控停靠轨迹作为训练集；在另两种未见环境（4F‑I、5F‑I）进行零样本测试；此外还对非停靠目标和低光照条件做补充实验。

**📈 对比分析**

方法对比：与 GC‑DP（NoMaD 适配）和 DINOv3‑基线（预训练表示）进行对比；GR‑VIP 在训练环境中 100% 成功率、~2 mm 终端误差；在未见环境中 100% 与 93.8% 成功率，误差仍保持毫米级；相比之下基线成功率仅 50‑56% 甚至 0%，误差更大。

**⚠️ 局限性**

局限性：未在部分遮挡、障碍物或动态环境下验证；对目标姿态的推断依赖完整视图，低光环境虽通过测试但仍需更强鲁棒性；目前仅在无障碍室内环境中验证，缺乏对不同停靠站与光照条件的广泛泛化评估。

---

## 507. TokenEconomizer: Experience-Driven Black-Box Cost Control for Coding Agents

**arXiv ID:** 2610.22732 | [PDF](https://arxiv.org/pdf/2610.22732v1)

**作者:** Aaron J. Li `[一作]` (UC Berkeley), Jianfeng Gao `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个外部控制器，利用已完成任务的轨迹动态调整提示、工具集和停止策略，以降低连续编码任务中的 token 消耗。

**💡 创新点**

在保持编码代理不变的前提下，首次实现了基于环境经验的三大元策略（微技能注入、工具裁剪、失败循环干预），从而在多任务序列中持续降低成本。

**🔧 技术方法**

通过 LLM 生成微技能候选、正则表达式/谓词检测器评估其有效性，统计工具调用频率进行裁剪，并利用经验分位数与 LLM 判断制定失败循环预算。

**📊 数据集**

在 6 个 SWE‑Bench 仓库（277 任务）、5 个 Web‑Bench 项目（102 任务）以及 2 个安全审计基准（CredData & ReposVul，各 100 任务）上进行实验。

**📈 对比分析**

与默认高推理、低推理及固定策略（RTK、RepoMap）对比，平均 token 成本降低 18.4–30.9%（SWE‑Bench），16–21%（Web‑Bench），6–20%（安全审计），且 Pass@1/Pass@2 与检测 F1 与基线保持相近。

**⚠️ 局限性**

可能误删工具、技能失效或错误识别困难任务为失败循环；适用范围受限于已支持的代理，未在最小化 harness 上测试，且持续会话实验尚待验证。

---

## 508. Rarity of Singletons Matters & Supplementary Materials

**arXiv ID:** 2610.22604 | [PDF](https://arxiv.org/pdf/2610.22604v1)

**作者:** Pragatheeswaran Vipulanandan `[一作]` (University of Miami), Dilip Sarkar `[通讯]` (University of Miami)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对小样本稀有事件（单品种）估计缺失质量与未出现种类的稳定性问题，提出了容忍不稳定性的修正框架。

**💡 创新点**

创新点是基于单品种的脆弱性度量和熵‑KL正则化的闭式修正，使估计在稀有事件不可靠时自动退回到不偏见状态。

**🔧 技术方法**

使用了熵‑KL优化、贝尔努利熵、KL散度以及基于现有 Good‑Turing / Good‑Toulmin 的基线估计。

**📊 数据集**

实验包括多种合成分布（均匀、Zipf、Dirichlet、头尾混合）以及两份真实文本语料（莎士比亚单词、Google 1‑gram）。

**📈 对比分析**

与 Good‑Turing、Good‑Toulmin、Chao、ACE、贝叶斯/贝塔等传统方法比较，在 360 次实验中在长尾、小样本情形下显著降低 UCR，保持与基线相当或更优的精度。

**⚠️ 局限性**

局限在于需要估计单品种比例以计算脆弱度，且对极端稀有事件分布仍可能受限；此外 α_0、γ 等参数需经验选择。

---

## 509. CF-NODE: Counterfactual Spatial Node Intervention for Test-Time Adaptation Under Domain Shift, Evaluated Across Five Diabetic Retinopathy Datasets

**arXiv ID:** 2610.22437 | [PDF](https://arxiv.org/pdf/2610.22437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 510. Achievable Rate Region Characterization of Rotatable Antenna-assisted Interference Channel

**arXiv ID:** 2610.23067 | [PDF](https://arxiv.org/pdf/2610.23067v1)

**作者:** Junran Pei `[一作]` (Nanjing University of Science and Technology), Wei Cai `[通讯]` (Nanjing Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过联合优化旋转天线的角度和发射波束，推导了MISO干扰信道的可达率区域，先在两收发对场景下得到闭式最优解，再提出通用的两步低复杂度优化框架。

**💡 创新点**

创新点在于引入子空间相关系数并给出能实现完全集束对抗干扰的最优旋转角度，证明在旋转天线条件下简单的MRT即可达到信息理论的干扰自由上界，并提供了高效的1D搜索加SOCP的联合设计方法。

**🔧 技术方法**

使用的技术包括速率配置（rate‑profile）方法、子空间相关性分析、闭式推导、1D网格搜索求角度、二阶锥规划（SOCP）求波束以及数值仿真验证。

**📊 数据集**

实验基于仿真数据，采用10 GHz频率、4个天线、β₀=10⁻⁴、σ²=-70 dBm，并设置具体的发射/接收坐标进行性能评估。

**📈 对比分析**

与传统固定天线（MRT、ZF、最优SOCP）以及理想无干扰上界进行对比，旋转天线优化的MRT/ZF方案在低/高SNR下几乎达到干扰自由上界，显著优于固定天线方案，尤其在高空间相关场景中提升显著。

**⚠️ 局限性**

局限性包括仅考虑LOS主导的二维几何模型，未考虑多径、动态环境或天线旋转的实际机械限制，且算法复杂度随发射/接收对数和天线数增长。

---

## 511. Reward Engine: Reward-Centric Self-Evolution for Open-Ended Tool Agents

**arXiv ID:** 2610.22915 | [PDF](https://arxiv.org/pdf/2610.22915v1)

**作者:** Litong Feng `[一作]` (Huawei), Qijun Li `[通讯]` (Huawei)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Reward Engine，能够从公开任务需求与少量初始轨迹自动构造证据驱动的奖励包（包含可执行的测试用例和评判 rubrics），并在使用过程中保持奖励不变，指导工具代理进行自我改进与候选结果的挑选。

**💡 创新点**

创新点在于（1）构造的奖励包完全基于可重跑的检查，能返回完整证据 receipt；（2）通过自我攻击与回归受限修复确保奖励在使用前已被测试且保持不变；（3）奖励既可用于即时反馈（Critic）又可用于候选排序（Jury），并在每轮结束后仅更新。

**🔧 技术方法**

主要技术包括：自然语言处理模型生成任务约束与检查代码；软件回归测试框架对奖励进行自检与修复；多阶段流程（Jury、Builder、QC、Critic、Selector）实现奖励构建与使用；统计比较方法（平均分、pass@3、Pass^3、regret、τ_+）评估效果。

**📊 数据集**

使用了 WorkBuddy Bench Office（50个混合格式办公任务）和 Claw‑Eval（104个冻结任务版本，涵盖68个场景）两个公开基准数据集。

**📈 对比分析**

通过与 Baseline（仅任务）、Revision（单次修订）、Uniform（均匀挑选）、Oracle（最高评分）等策略对比，Selected by reward 在两大基准上平均分最高、pass@3在 Claw‑Eval 达到 100%，Pass^3 在 WorkBuddy 从 45% 提升至 54%，在 Claw‑Eval 从 78.4% 提升至 83.7%；且奖励包与官方评分器的排序一致率约 82%。

**⚠️ 局限性**

局限性包括：需要人工 Jury 先行草拟约束；奖励构建依赖初始轨迹，可能对未见候选泛化不足；回归修复受限于参考标签质量；未在强化学习场景中验证；元循环未被评估。

---

## 512. MobileJailBench: Red Queen Jailbreak Benchmarking for Mobile Agents

**arXiv ID:** 2610.23303 | [PDF](https://arxiv.org/pdf/2610.23303v1)

**作者:** Songping Wang `[一作]` (Nanjing University), Caifeng Shan `[通讯]` (Nanjing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 MobileJailBench，构建了一个可演化的移动代理 jailbreak 基准，包含520个任务、14类风险、20+常用App，并配备双动态演化机制和分层裁决框架；

**💡 创新点**

创新点包括①首次实现移动代理可演化的 jailbreak 基准；②双动态演化（失败驱动 Jailbreak Evolution 与弱点驱动 Benchmark Expansion）实现基准自适应与扩展；③分层裁决框架从安全意识、执行行为、风险后果三维进行细粒度诊断；

**🔧 技术方法**

采用高保真 Android 浏览器沙盒、Python 驱动模拟器、LLM 任务生成与评估、失败驱动策略路由、弱点驱动任务合成、双轨道判定与证据审核等技术；

**📊 数据集**

使用28个流行应用（14第三方+14系统）和520个手工+LLM 生成的显式、隐式、组合 jailbreak 任务，涵盖多种风险和交互复杂度；

**📈 对比分析**

对10个多模态LLM（Qwen、GLM、Gemini、GPT、Kimi）与两种代理进行对比，评估ASR、SVR、RHR、SRR等指标。实验显示隐式攻击显著提升SVR、降低SRR，Qwen3.5-122B在演化后实现100% ASR，且与传统基准相比覆盖率和诊断细粒度大幅提升；

**⚠️ 局限性**

局限性包括：模拟器与真实设备差异导致外推性受限；攻击仅局限于指令与环境载体；基准仍未覆盖所有恶意用例；分层裁决依赖人工标注与规则，可能存在主观偏差。

---

## 513. OliO: ODE-based Linear Transition Operator for Self-Supervised Time Series Forecasting

**arXiv ID:** 2610.23345 | [PDF](https://arxiv.org/pdf/2610.23345v1)

**作者:** Kwangryeol Park `[一作]` (Ulsan National Institute of Science and Technology), Seulki Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了自监督预训练框架 OliO，用于学习时间序列表示的有符号时延条件双向转移，从而提升预测性能。

**💡 创新点**

创新点在于使用基于线性ODE的严格上三角生成器实现有限多项式转移，并结合加性偏移实现时延敏感的表示迁移。

**🔧 技术方法**

采用线性ODE导出的矩阵指数转移、上三角约束、加性偏移、停梯度对齐损失以及共享投影头的自监督方法。

**📊 数据集**

在 ETTh1/2、ETTm1/2、Weather、Electricity 等六大公共时间序列数据集上进行评测。

**📈 对比分析**

与随机初始化、TS2Vec、SimMTM、SoftCLT、TimeSiam、TimesURL 等对比，OliO 在 58/72 设定下 MSE 下降至 5.75%，在多模型、多数据集的冻结和零射预测均显著优于基线。

**⚠️ 局限性**

局限性包括对状态无关生成器的假设、仅在预训练阶段使用转移模块、对长时延的适用性需进一步验证，以及在某些数据集上提升有限。

---

## 514. Nearly Linear Private Learning of Littlestone Classes

**arXiv ID:** 2610.23160 | [PDF](https://arxiv.org/pdf/2610.23160v1)

**作者:** Shay Moran `[一作]` (Technion Israel Institute Of Technology), Chao Yan `[通讯]` (Technion Israel Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究了在差分隐私条件下的 PAC 学习，给出了关于有限 Littlestone 维数的几乎线性样本复杂度上界。

**💡 创新点**

提出了结构化指数机制以及基于模型论的秩函数，用于构造数据依赖的候选集合，实现 O(d log d) 的样本复杂度。

**🔧 技术方法**

利用指数机制、结构化指数机制、Littlestone 维数、模型论中的秩、版本空间和好族计数等技术。

**📊 数据集**

未使用实验数据，论文为理论性，主要对比先前的 O(d⁶)、O(d⁵) 上界。

**📈 对比分析**

与以往的 O(d⁶) 与 O(d⁵) 结果相比，改进至 O(d log d)，在子空间指示器类上进一步降低到线性并去除对环境维数的依赖。

**⚠️ 局限性**

仍未达到最优 O(d) 的下界（需要进一步研究 log* 量级），且算法未给出可实现的高效实现，仅提供样本复杂度分析。

---

## 515. PromptSep-Speaker: Generative Target Speaker Extraction via Multimodal Prompting and Semantic Alignment

**arXiv ID:** 2610.23066 | [PDF](https://arxiv.org/pdf/2610.23066v1)

**作者:** Yutong Wen `[一作]` (Adobe Research), Zeyu Jin `[通讯]` (Adobe Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于潜在扩散的目标说话人提取（TSE）模型，能够通过四种不同的提示（说话人时序 ID、转录文本、录音样本和自由文本描述）来定位并分离混合语音中的目标说话人。

**💡 创新点**

创新点包括：①将多模态提示统一在同一模型中，支持任何可用的提示类型；②设计了辅助对齐损失，强制模型在对齐模块中正确聚焦到目标说话人的音频帧；③在生成式 TSE 领域首次使用潜在扩散模型，显著提升分离质量和鲁棒性。

**🔧 技术方法**

技术手段包括：潜在扩散（latent diffusion）框架、DiT 模块、DAC‑VAE 编码/解码、Qwen3‑VL‑8B 文字编码器、冻结的 WavLM 语音特征、双向 Transformer 对齐模块、辅助对齐损失与跨注意力机制。

**📊 数据集**

使用了四套评测数据集：CapSpeech‑Mix、Libri2Mix（包含干净与噪声子集）、LibriTTS‑R‑Mix 以及内部构建的英语语料库，所有数据均为离散于训练集之外的对齐、转录、标签与描述。

**📈 对比分析**

与多种基线（DAE‑TSE、WeSep、SoloSpeech、PromptSep、FlowSep、SAM‑Audio）以及 4 种不同提示的对比实验，平均提升 SI‑SDRi 6.6–11.3 dB、mel‑L2 降低约 2–7 dB、DNS‑MOS 上升 0.1–0.4 点，F1 目标选择率在 0.93–0.95 之间，主观评测中多项 MOS 指标均达到或超过 4.0/5.0。

**⚠️ 局限性**

局限性主要体现在：①对三人以上混合的性能下降显著；②对全重叠或部分重叠语料的稳健性仍受限；③缺乏对多模态提示的互补性分析，未探索不同提示组合的最佳策略。

---

## 516. Bengali Hallucination Detection with Retrieval and Programs: A System Study

**arXiv ID:** 2610.22892 | [PDF](https://arxiv.org/pdf/2610.22892v1)

**作者:** Shamim Hasan `[一作]` (BNIST), Samiya Ahmed Chowdhury `[通讯]` (Chittagong University of Engineering and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

该论文实现了一个基于检索和程序辅助的孟加拉语幻觉检测系统。

**💡 创新点**

创新点是将响应条件检索、混合检索与文本判断器以及数学代理相结合，并提供可审计的运行统计。

**🔧 技术方法**

使用技术包括多模态检索（E5、FAISS、BM25、RRF、BGE重新排序）、Qwen 4B/14B/27B 大模型、Python 程序执行以及 vLLM 推理。

**📊 数据集**

数据集为 Olikbochon 孟加拉语 LLM 幻觉检测挑战的公开样本及测试集，并利用了外部 Q&A 与上下文语料库。

**📈 对比分析**

方法通过版本对比和指标审核对比，版本 2 在私有排行榜上取得 0.956 宏 F1，整体运行时间约 221 分钟，显示高精度但主要依赖检索。

**⚠️ 局限性**

主要局限在于缺乏真实标签、数据重叠与可解释性不足、可重复性受依赖版本限制，以及程序执行安全和公平性问题。

---

## 517. ADT: All-Atom Diffusion Tokenizer for 3D Molecular Structures

**arXiv ID:** 2610.23265 | [PDF](https://arxiv.org/pdf/2610.23265v1)

**作者:** Xuantuo Huang `[一作]` (Peking University), Mingxiao Li `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出All-atom Diffusion Tokenizer（ADT），实现一种统一的离散化3D分子结构表示，能够在保持全原子重建精度的同时，使用单个token编码每个原子（小分子）或每个残基（蛋白质）。

**💡 创新点**

创新点在于把token粒度与重建分辨率解耦，采用联合编码骨干与侧链为一个残基token，同时使用条件扩散解码器弥补量化引入的几何失真。

**🔧 技术方法**

使用的技术包括有限标量量化（FSQ）、互信息正则化、粗细分级的扩散解码器、Transformer编码器/解码器以及自回归token生成。

**📊 数据集**

使用的数据集有QM9和∇²DFT用于小分子重建与生成，AFDB SwissProt、CASP14/15及CAMEO22用于蛋白质重建与序列条件生成。

**📈 对比分析**

与现有离散tokenizer（如CoordToken、Mol‑StrucTok）以及连续生成方法（GeoLDM、EQGAT‑diff）对比，ADT在重建RMSD、匹配率、生成有效率、独特性、多样性以及蛋白质的Cα‑lDDT、all‑atom lDDT、TM‑score等指标上均优于或接近最佳表现。

**⚠️ 局限性**

限制主要在于仍需依赖扩散解码器的高计算成本，量化过程可能在极大或结构复杂度极高的分子上导致细节损失，且尚未在极稀有化学结构或大尺寸蛋白上进行充分验证。

---

## 518. Can SOLARBURST go silicon? A systematic analysis on software supply chain attacks for VLSI design flows

**arXiv ID:** 2610.22580 | [PDF](https://arxiv.org/pdf/2610.22580v1)

**作者:** Fabiha Hashmat `[一作]` (Purdue University), Santiago Torres-Arias `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了软件供应链攻击对VLSI设计流程的威胁，构建了攻击分类模型并发现了717个潜在攻击，其中60个高可行性攻击得到证据支持。

**💡 创新点**

创新点在于首次从软件供应链角度全面映射VLSI设计流程的攻击面，提出了完整的攻击分类体系、威胁模型，并评估了现有防御措施的覆盖程度。

**🔧 技术方法**

研究采用实证分析、攻击分类模型构建、CIS安全指南映射、以及基于S‑code的防御评估技术，并在开放源码VLSI工具链上进行案例验证。

**📊 数据集**

使用的主要数据集为公开的VLSI软件套件（如OpenROAD、Yosys、Magic等）及其依赖包信息，结合实际工具链日志和代码库构成。

**📈 对比分析**

通过将攻击路径映射到CIS的15大防御维度，评估每种攻击在不同设计阶段的可行性，并量化已实现与未实现的防御覆盖率，展示了现有防御在大部分攻击面上不足。

**⚠️ 局限性**

局限性包括样本主要来自开源工具，可能不涵盖商业闭源环境；攻击模型基于当前已知技术，未来出现的新型攻击可能未被覆盖；并且防御评估主要依赖静态规则，未深入动态运行时检测。

---

## 519. Partial-Revelation Panel Forecasting: Predictive Information Gain, Allocation, and Coarsening-Consistent Operators

**arXiv ID:** 2610.22831 | [PDF](https://arxiv.org/pdf/2610.22831v1)

**作者:** Bin Wen `[一作]` (Universiti Sains Malaysia), Tien-Ping Tan `[通讯]` (Universiti Sains Malaysia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究部分显露面板预测（partial-revelation panel forecasting），量化额外投票对未来标注面板分布的预测价值，并提出一致性一致化面板操作（coarsening-consistent operators）以及相应的预算分配策略。

**💡 创新点**

1) 定义了跨轮预测增益与条件互信息的等价关系；2) 提出了可闭式Dirichlet–multinomial一致性面板操作，并将同一项目（same-item）和跨轮（cross-turn）信息增益区分；3) 通过持出评估验证同一项目增值，且给出水填充分配方法。

**🔧 技术方法**

信息论证明、Dirichlet–multinomial贝叶斯建模、EM式后验加权迭代、GRU完成器、无监督不确定性估计、水填充分配、数值对照与多模态特征融合。

**📊 数据集**

IEMOCAP、M3ED、EmoWOZ、EmotionLines（多模态/文本情感对话数据集）。

**📈 对比分析**

与单级查找表（per-level lookup）以及多种一致性/非一致性操作进行对照；在IEMOCAP、M3ED、EmoWOZ上的冷启动NLL差异分别为-0.422、-0.166、-0.014 nats；部分显露增益小且在不同语料上差异显著；预算分配实验显示均匀分配优于集中分配，在IEMOCAP/M3ED可恢复约80–90%回报，EmoWOZ效果较弱。

**⚠️ 局限性**

增益依赖语料并在EmoWOZ几乎无效；一致性不保证预测提升，仅为正则化与可部署性；CCAR掩码是假设性实验干预，未验证真实到达过程；样本量小、面板规模受限；缺乏公开排行榜对比；方法未在更大面板规模上验证；同一项目与跨轮增益不可互换。

---

## 520. Designing Human-AI Collaborative Science: The 9A Framework

**arXiv ID:** 2610.22948 | [PDF](https://arxiv.org/pdf/2610.22948v1)

**作者:** Zihan Zheng `[一作]` (University of Illinois Urbana--Champaign), Xin Lan `[通讯]` (Michigan State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了9A框架，用于设计与评估人类–AI协同科学（HACS）的组织、协调和评价三层结构，帮助明确科学活动、参与者、权限分配以及协同过程的知识共享与责任追踪。

**💡 创新点**

创新点在于将任务分配、角色委托、信息共享、责任追踪和系统评估等多维度概念整合成统一的框架，并强调从系统设计角度考虑人机协同，而非仅关注单个主体的能力。

**🔧 技术方法**

主要技术手段为概念框架构建、文献综述、层级模型设计以及对现有AI科学系统的案例映射；并未引入新的机器学习模型或算法。

**📊 数据集**

本文未使用任何特定数据集，而是基于对现有AI科学系统（如AI Scientist、Co-Scientist等）的描述进行案例分析。

**📈 对比分析**

未进行实验性比较或性能评估；作者通过对比分析不同系统的组织与协同模式来阐明框架的适用性，并指出未来可在实验平台上进行设计因子对比研究。

**⚠️ 局限性**

局限性包括：框架仍处于概念阶段，缺乏在真实科学研究中的实证验证；不同科研领域的任务粒度、权限阈值和协同需求差异可能导致框架需要进一步细化；框架实施会增加信息共享、审计与记录的计算与人力成本。

---

## 521. Peak-Memory Schedules You Can Trust: Exact Realization, Certified Optimality, and the Limits of Recomputation for Inference

**arXiv ID:** 2610.23328 | [PDF](https://arxiv.org/pdf/2610.23328v1)

**作者:** Ivan Fomin `[一作]` (Yandex), Boris Sheludko `[通讯]` (Yandex)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种在已生成程序上直接按优化后的操作顺序排列（fixed‑wrapper）来实现内存调度，并给出了可计算的前向/后向“陷阱”下界，用以证明得到的调度已经是最优；同时分析了可选择性重计算（caps）在实时推理中的可行性与复杂性。

**💡 创新点**

创新点：
1) 通过对生成的 wrapper 进行精准的运行时分析与重排序，保证在不改变编译器生成代码的前提下实现峰值内存的“按字节”精确化；
2) 引入两种 O(n²) 最小割实现的前向与后向陷阱下界，能够在毫秒级完成最优性证明；
3) 对可选择性重计算的 NP‑难性做出多维度理论界定，并给出在树宽受限图上可精确求解的算法框架；
4) 在多平台（CPU、GPU、微控制器）和多工作负载（推理、训练）上系统性评估并揭示重排序与重计算的实际收益与局限。

**🔧 技术方法**

使用技术：
- PyTorch 2.14.1 + Inductor 的编译器框架，利用其生成的 wrapper 进行一次性全量剖析；
- 事件模型与根模型（root model）对内存行为建模；
- 依据读/写依赖构造单元（unit）及其依赖图；
- 深度优先搜索与最小割（min‑cut）实现的可行性搜索与陷阱下界计算；
- 对 GPU 的 CUDA 缓存 allocator 进行实验验证并使用 guard 机制；
- 通过动态规划与树宽分解实现可选择性重计算的精确算法；
- 量化实验中的内存峰值、MAC 计数、计算时间与 bit‑wise 输出一致性。

**📊 数据集**

数据集与模型：
- 服务器级推理：DeepLabV3‑ResNet50、Segmentation、CNNs、Transformer 预填充等；
- 微控制器级推理：MLPerf Tiny 参考网络、TC‑ResNet 关键词识别器、MobileNetV2、ShuffleNet、SqueezeNet、EfficientNet‑B0、GoogLeNet、Tiny‑UNet 等；
- 训练：50 种不同架构（包括 Transformer、CNN、BERT/GPT‑2 等）和形状扫描；
- 公开 ReDimNet 说话人嵌入网络（B0–B6）用于重计算实验；
- GPU 复现：NVIDIA RTX 2080 Ti，CUDA 12.6，使用 Triton kernel；
- 其它：随机 DAG、series–parallel、pumpkin、tree 等理论测试图。

**📈 对比分析**

比较方法与性能：
- 与 Inductor 默认调度（stock）对比，测量峰值内存、MAC 额外消耗、bit‑wise 结果一致性；
- 对 CPU：fixed‑wrapper 调度在训练上可达 7.6% 内存减幅，推理上 1–3%（DeepLabV3 最高 12%）；
- 对 GPU：由于缓存 allocator 影响，fixed‑wrapper 在 60% 配置下可实现 8–9% 的逻辑峰值下降；
- 对微控制器模型：几乎所有模型的峰值已是最优，无法进一步压缩；
- 对 ReDimNet：通过重计算 stem 可在不增加 MAC >0.4% 的情况下降低峰值 15%；
- 计算时间：模型建模、证书生成在毫秒级，整体实验耗时 ≈1‑2 s/图；
- 证书成功率：在 held‑out 训练/推理图中 80–90% 的情况已得到最优证明，剩余仅在分支网络上可改进。

**⚠️ 局限性**

局限性：
- 依赖于精确分配器；CUDA 缓存 allocator 的历史依赖破坏了模型精确性，需要 guard 机制；
- 对库核（如 cuDNN）依赖于运行时剖析，无法形式化证明；
- 仅在静态分配/固定 arena 的环境下保证“按字节”精确；
- 可选择性重计算的 NP‑难性意味着在大规模图上不可行；
- 对微控制器的实验只评估了算子图与贪心 arena，未覆盖实际固件实现；
- 证明的陷阱下界未必最紧，某些图（如 RegNetY‑400MF）仍存在显著间隙；
- 对多输入、多批大小的动态场景，根模型的“帧条件”可能不成立，需进一步研究。

---

## 522. RAP: A Dataset for Paper-to-Poster Generation via Declarative Markup from Human-Crafted Exemplars

**arXiv ID:** 2610.23230 | [PDF](https://arxiv.org/pdf/2610.23230v1)

**作者:** Haonan Han `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模的论文-海报对齐数据集 RAP，并实现了自动化的海报 derendering 与 HTML/CSS 重构，同时提出了无训练检索增益生成和基于 SFT+RL 的专属海报生成模型。

**💡 创新点**

首次提供可渲染、可编辑的 HTML/ CSS 海报模板与对应论文，形成可直接评估的 render‑and‑compare 基准，并将人类设计作为可复用的模板库，突破传统以图像或布局框为目标的可视编码方法。

**🔧 技术方法**

采用多模态 LLM（Claude‑Opus‑4.6、Gemini‑3.1‑Pro）、LayoutLMv3 与 PaddleOCR 进行文本与布局识别，使用 MLLM 生成设计描述，构建多代理检索-映射-审阅流水线，并通过规则+VLM 奖励的 RL 对模型进行后训练。

**📊 数据集**

使用从 NeurIPS、ICML、ICLR 2022–2026 OpenReview 收集的 25,435 对验证过的论文‑海报样本，并在此基础上抽取 150 条测试样本，配备三层内容分类（15 个领域、20 个任务、39 个方法）的标注。

**📈 对比分析**

与 NanoBanana、P2P、Paper2Poster、PosterGen 等基线对比，利用 LPIPS、DINOv2‑CosDist、DISTS、FID、FDiD 等指标评估多样性与人类相似度；RAP（HTML 指导）在所有指标上均优于基线（FID 23.75，DINOv2 0.444），RL 进一步提升了多样性与分布相似度。

**⚠️ 局限性**

derendering 对高分辨率复杂排版的恢复仍受限于 MLLM 的定位与文字识别精度，数据集仅覆盖顶级 ML 会议论文，对非会议海报或其他学科的泛化能力未知，且 pipeline 对高质量原始海报的依赖导致对低质量或缺失格式海报的适配性不足。

---

## 523. Understanding KV Residency in Production-Scale Agentic LLM Workloads

**arXiv ID:** 2610.22863 | [PDF](https://arxiv.org/pdf/2610.22863v1)

**作者:** Mayank Arya `[一作]` (Indian Institute of Science), Saravan Rajmohan `[通讯]` (Microsoft 365 Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对微软M365的四类生产型agentic LLM工作负载，作者通过大规模时序追踪分析了 KV 缓存的生命周期、占用规模与重用率，并在此基础上提出并评估了基于空闲时间截断、容量受限的淘汰策略（LRU、Belady 的 FNU）以及多分辨率保留（int4 MRR）和部署感知路由的改进方案。

**💡 创新点**

创新点包括：①首次系统性量化 agentic 会话的 KV 需求和多模型交互导致的重用瓶颈；②通过“空闲时长截断”显著降低全局 KV 容量上限；③利用多分辨率保留实现高效的低精度 KV 再利用，显著降低重算率；④揭示部署局部性对 KV 可复用性的“第一接触”底线；⑤计划公开 AgenticTraceSim 生成器与可复现的存活重放框架，为后续研究提供开放工具。

**🔧 技术方法**

主要技术手段为：1) 端到端的 trace‑driven 归档与处理；2) 计算 KV 工作集与上下文重用率；3) 通过 idle‑cap、容量预算、LRU/FNU/MRR 等缓存策略进行离线重放；4) int4 量化实现多分辨率保留；5) 对部署信息进行切片分析，评估部署局部性影响。

**📊 数据集**

使用的数据集来自微软 M365 的四条生产追踪：Deep Research（17M 调用/691k 会话）、Enterprise Assistant（217k 调用/23.5k 会话）、Coding Agent（1.12M 调用/8.1k 会话）和 Enterprise Chat（104M 调用/13.2M 会话），共计超过 1.2 亿次模型调用。

**📈 对比分析**

通过在最繁忙一小时的调用流上以不同容量预算进行重放，比较了三种淘汰/保留策略的重算率：LRU 62.9%→FNU 31.1%→MRR 13.6%；在将空闲时间截断为一小时后，整体 KV 容量上限从 501.8 TB 降至 140.8 TB，重算率仅微幅升至 4.53%；部署局部性实验表明，将 KV 对象按会话+部署键划分后，第一接触底线从 5.4% 提升至 20.2%。

**⚠️ 局限性**

局限性包括：①仅基于微软 M365 的生产追踪，可能不具备通用性；② KV 规模估算基于 Llama‑2‑70B 的 fp16 GQA 结构，未考虑真实模型的细节；③模型类别划分和部署标识的粒度可能影响结果；④未评估多分辨率保留的实际延迟和精度损失；⑤由于隐私原因，未公开完整追踪数据，导致其他研究者只能使用公开的生成器。

---

## 524. Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design

**arXiv ID:** 2610.23036 | [PDF](https://arxiv.org/pdf/2610.23036v1)

**作者:** Benhao Huang `[一作]` (Institute of Foundation Models), Xuezhe Ma `[通讯]` (Institute of Foundation Models)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对循环Transformer模型的架构（Ouro与Huginn）进行系统比较，并通过控制迭代包络、输入注入和潜在状态组织三条设计轴展开离散消融实验；在同等参数规模、计算量与训练预算下，对标准与稀疏（MoE）模型进行评估。

**💡 创新点**

将循环语言模型的设计分解为迭代包络、输入注入与潜在状态组织三大维度，并首次在同等条件下逐项消融验证其对性能的具体贡献；发现预置-循环-后置结构显著提升推理与数学推理能力，并在MoE设定下实现更均衡的专家利用率和更优的参数效率。

**🔧 技术方法**

循环Transformer、参数共享、预置-循环-后置架构、输入注入、随机或分层状态初始化、top‑2 路由的稀疏专家网络（MoE）、PreNorm 变压器骨干、RoPE 位置编码、SwiGLU MLP、AdamW 优化器与余弦衰减、权重衰减。

**📊 数据集**

TxT360 大规模去重数据集（Common Crawl + curated 来源）以及十个基准：ARC‑Challenge、HellaSwag、MMLU、TriviaQA、BBH‑CoT、DROP、GSM8K、MATH500、HumanEval+ 与 MBPP+。

**📈 对比分析**

在匹配 730M 存储参数、2.9B 逻辑深度、相同训练词数与 FLOP 预算的条件下，使用十项基准进行离散消融和整体比较；Huginn 在所有密集任务上均优于 Ouro，MoE 版 Huginn 在专家使用均衡性和性能上接近或超越同等规模的前馈 MoE，同时显著减少参数存储。

**⚠️ 局限性**

实验仅涵盖单一模型规模与训练预算，缺乏跨规模的伸缩性验证；未进行多种随机种子复现以评估不确定性；MoE 研究未分别消融输入注入、状态组织等轴；对更大规模模型及不同硬件平台的通用性尚未探测。

---

## 525. Verbalising Model's Computation for Chain-of-Thought Faithfulness Detection

**arXiv ID:** 2610.23275 | [PDF](https://arxiv.org/pdf/2610.23275v1)

**作者:** Yujing Wang `[一作]` (King's College London), Hanqi Yan `[通讯]` (King's College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种通过Jacobien‑lens将LLM内部计算过程转化为可解释的“语义轨迹”，并利用这些轨迹提取三种结构化特征（Map、Profile、Dynamics）来检测链式推理（CoT）的可信度。

**💡 创新点**

创新点在于：①首次使用层级Jacobien‑lens将内部状态映射到共享词汇空间，实现对不同层、不同推理阶段的统一可解释性；②构造“faithful anchors”并通过三维特征（位置、内容、动态变化）完整描述推理轨迹；③在不需要人类评审或模型修改的情况下，实现对CoT可信度的高精度检测和机制解释。

**🔧 技术方法**

技术核心包括：Jacobien‑lens投影、Anchor字典构建、Map/Profile/Dynamics特征提取、线性Logistic回归分类器；同时使用CoT阶段归一化、层次分组与语义对齐。

**📊 数据集**

数据集：FaithCoT‑Bench（包含AQuA、LogiQA、TruthfulQA、HLE‑Bio）和BonaFide，使用两大模型后端 Llama‑3.1 与 Qwen‑3。

**📈 对比分析**

与Perplexity、Early‑Ans、Info‑Gain、Ans‑Tracing、LLM‑as‑Judge以及隐藏层探针（HS_m、HS_l）等基线对比，取得28.4%平衡准确率提升、16.0%激活探测提升，在所有设置下均为最佳。模型跨数据集、跨模型迁移保持90%以上原始性能。

**⚠️ 局限性**

局限性：依赖于已知的faithful‑unfaithful CoT对，anchor发现需要大量对比样本；对极长或极短CoT的特征表现不一致；仅适用于已知内部结构可访问的模型，对黑盒模型的可解释性有限。

---

## 526. From Border-Agnosticism to Causal Rewrite Systems

**arXiv ID:** 2610.23351 | [PDF](https://arxiv.org/pdf/2610.23351v1)

**作者:** Pablo Arrighi `[一作]` (Universit\'e Paris-Saclay), Luidnel Maignan `[通讯]` (Universit\'e Paris-Est Cr\^eteil)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过研究因果图重写中的邻域方案，提出了“边界无关性”(border‑agnosticism)概念，并将局部规则映射为传统的模式替换规则，完成了从因果图动态到可表述为重写系统的理论转换。

**💡 创新点**

创新点在于将边界视为缺失信息的安全原则与盘面唯一性（disk uniqueness）等性质等价化，构建了边界无关邻域方案与互斥圆锥集(mutex sets of cones)之间的双射映射，并证明了局部规则与因果重写系统之间的等价性，弥补了先前只关注可扩展性（extensivity）的局限。

**🔧 技术方法**

采用了图论与代数结构（如命名代数、时间标签、可扩展性与幂等性等）以及严格的公理化定义，对邻域方案、局部规则和重写系统进行形式化，并通过一系列证明展示其性质。

**📊 数据集**

该工作为纯理论研究，没有使用实验数据集。

**📈 对比分析**

由于是理论性工作，没有实验对比或性能评估；主要通过形式证明展示了构造方法的正确性与完整性。

**⚠️ 局限性**

局限性包括：需在边界可补全（border‑completable）图集上运作；仍假设图的无环性与有限链；缺乏局部可构造判定标准；以及对可逆性等更高级属性的进一步研究留待后续工作。

---

## 527. KiranaBench: Can a Language Model Run a Neighbourhood Shop?

**arXiv ID:** 2610.22980 | [PDF](https://arxiv.org/pdf/2610.22980v1)

**作者:** Bhuvan Arora `[一作]` (BITS Pilani), Yash Sinha `[通讯]` (BITS Pilani)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了多智能体在印度小型杂货店（kirana）中的长周期业务运营，评估LLM代理的协作与多语言交互能力。

**💡 创新点**

提出了SimKirana模拟环境，融合四个角色（经理、收银员、助手、簿记员）LLM代理与真实的需求、语言和供应商交互，填补了单一英文、单代理长周期评测的空白。

**🔧 技术方法**

使用四个开源LLM（Qwen3.8‑27B、Gemma‑4‑31B、gpt‑oss‑20b、LFM2.5‑8B‑A1B）以及工具调用框架，对业务决策、库存、账务等任务进行自动化。

**📊 数据集**

以印度国家《家庭消费支出调查》（Household Consumption Expenditure Survey）为需求来源，并通过模板与生成模型提供英语、印地语及 Hinglish 的客户/供应商对话。

**📈 对比分析**

与固定规则店主基线对比，最佳模型平均仅获得脚本店主利润的1/8；两模型甚至低于无人管理的店铺；在不同语言下，模型对供应商邮件的成功率差异显著。

**⚠️ 局限性**

局限性包括仅测试单一城市店铺一个月、模型规模有限、语言支持不完整、报告真伪度参差不齐，且结果易受种子随机性的影响。

---

## 528. Robust Function Calling for LLM Agents Under Interface Drift

**arXiv ID:** 2610.23342 | [PDF](https://arxiv.org/pdf/2610.23342v1)

**作者:** Aniruddha Chattopadhyay `[一作]` (Independent Researcher), Partha Sarathi Roy `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种轻量级的推理时验证与修复堆栈，提升大型语言模型在工具接口漂移下的功能调用鲁棒性。

**💡 创新点**

创新点在于将结构化校验与一次性约束修复结合，并通过候选工具集重试来解决缺失工具名问题，证明可在不重新训练模型的情况下显著恢复漂移导致的错误。

**🔧 技术方法**

使用了结构化校验器、候选集重试、基于提示的一次性约束修复和可配置的工具卡。

**📊 数据集**

在DICE‑BENCH（300条单轮对话）和BFCL（200条单轮工具调用）两个基准上进行评估。

**📈 对比分析**

与原始接口、漂移接口以及仅重试的基线对比，发现Qwen3.5‑9B在DICE上从0.717提升至0.813，在BFCL上从0.580提升至0.805，且修复过程无明显误伤。

**⚠️ 局限性**

局限包括：实验主要集中在Qwen系列模型，跨模型、跨家族的非oracle验证有限；漂移类型仅涵盖描述、模式重命名和干扰工具，未覆盖更复杂的接口变更；评估使用严格字符串匹配，可能低估恢复率；缺乏更广泛的延迟与生产环境验证。

---

## 529. Can We Use Saliency Maps for the Evaluation of Generative Image Models?

**arXiv ID:** 2610.23411 | [PDF](https://arxiv.org/pdf/2610.23411v1)

**作者:** Eleonora Ristori `[一作]` (University of Florence), Paolo Frasconi `[通讯]` (University of Florence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出一种基于专门训练的生成器-真实判别器（GSC）和其梯度归因（saliency）分析的评估框架，利用分类准确率与显著性信息的组合得到SAS得分。

**💡 创新点**

创新点在于不依赖预训练特征提取器，直接学习生成器特定的伪造痕迹，并将解释性特征量化为可与人类感知高度相关的指标。

**🔧 技术方法**

核心技术包括：生成器特定的二分类网络、集成梯度（Integrated Gradients）可视化、SAS得分公式（a(g)+a(g)^α(1-s(g))）以及多种可解释方法的对比评估。

**📊 数据集**

使用四个公开数据集（CIFAR‑10、FFHQ、ImageNet、LSUN Bedroom）以及涵盖GAN、扩散、VAE、流模型等多种生成器进行实验。

**📈 对比分析**

与传统指标（FID、FD、MMD等）对比，SAS在所有数据集上与人类错误率呈显著负相关（|r|>0.8），而许多基准指标的相关性随特征提取器变化显著，SAS保持稳定且更贴近人类判断。

**⚠️ 局限性**

主要局限在于需要为每个生成器单独训练一个判别器，增加了计算成本，并且该方法依赖于真实与生成样本的可用性，对跨领域应用或在线调参仍有挑战。

---

## 530. On the Tip of the Tongue: First-Token Cueing as a Behavioral Probe of Recall Failures in LLMs

**arXiv ID:** 2610.23400 | [PDF](https://arxiv.org/pdf/2610.23400v1)

**作者:** Jaewon Sok `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型在封闭式问答中的事实错误，发现大部分错误源于首个回答 token 的回忆失败，并通过 First-Token Cueing 纠正了大量错误。

**💡 创新点**

首次提出行为探针 First-Token Cueing，将人类 tip‑of‑the‑tongue 的 onset cueing 思路迁移到 LLM 以检验首 token 记忆失败，揭示错误主要集中在第一个 token 并能通过给出正确首 token 大幅提升准确率。

**🔧 技术方法**

采用首 token 预填充（prefilling）与 greedy 解码的行为探针方法，同时使用熵、AUROC 评估模型的不确定性，并通过跨切换提示控制对比信息效应。

**📊 数据集**

使用 TriviaQA、NQ‑Open、WebQuestions 等英文问答基准以及跨语言的 MKQA 多语种数据集。

**📈 对比分析**

与基线、随机/错误/通用前缀以及多步推理和采样相比，First‑Token Cueing 在 16 种开源模型上平均提高 5–15%（取决于规模），在多语言测试中同样表现出显著提升。

**⚠️ 局限性**

局限在于仅评估短答案事实问答，无法直接扩展到长文本生成，且该方法仅为诊断工具，需进一步研究如何在无 gold token 的情况下实现首 token 纠正。

---

## 531. FloodVLM-Pix: Language-Guided Pixel-Level Flood Understanding

**arXiv ID:** 2610.23377 | [PDF](https://arxiv.org/pdf/2610.23377v1)

**作者:** Chengyang Zhang `[一作]` (Tsinghua University), Gang Li `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个支持语言与像素级、多模态、多时序的洪水理解基准FloodVL-Pix，并提出FloodVLM-Pix模型；

**💡 创新点**

通过引入洪水先验模块校正注意力偏差以及采用通用-专用特征融合的多时序像素解码器，实现了对洪水规模大、复杂环境下的精确语言描述和像素分割；

**🔧 技术方法**

采用Qwen3-VL视觉语言模型、ConvNeXt特征提取、LoRA微调、层归一化、门控通用-专用特征融合等技术；

**📊 数据集**

使用131,200幅洪水图像块（106,500 SAR + 24,700 光学）和443,100条文本描述的FloodVL-Pix数据集；

**📈 对比分析**

与多种通用与遥感VLM（如Qwen3、GeoPixel、TeoChat）以及SOTA洪水分割模型（LSNet、MFGF-UNet等）对比，FloodVLM-Pix在语言描述（ROUGE-L/CIDER/CLAIR）和像素分割（OA/F1/IoU）上均显著领先；

**⚠️ 局限性**

仍存在对极端云遮蔽和不同传感器兼容性提升的需求，且在极大洪水区域的描述仍需进一步细化。

---

## 532. Truth in Contrast: Calibrated Factual Evidence from Language Model Likelihoods

**arXiv ID:** 2610.23527 | [PDF](https://arxiv.org/pdf/2610.23527v1)

**作者:** Ponhvoan Srey `[一作]` (Nanyang Technological University), Anh Tuan Luu `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并实现了基于对比生成似然的事实置信度方法（Contrastive Factual Evidence CFE 与 Contrastive Factual Posterior CFP），从判别力与校准两个维度评估其性能，并提出了基于 CFE 似然几何的证据选择式跨语言对齐目标，以将英语强大事实证据迁移到低资源语言。

**💡 创新点**

① 从理论上证明共享的 surrogate error 通过对比可削弱校准误差；② 将 CFE 视作二维似然空间，分离事实证据方向与证据无关方向；③ 基于此几何提出仅对事实证据方向进行对齐的 LoRA 训练目标，显著提升低资源语言的事实判别力并保持整体能力。

**🔧 技术方法**

对比生成似然（CFE），对比后映射为概率置信度（CFP，sigmoid 转换），Bayesian 归纳，LoRA 微调进行跨语言对齐，使用 ECE、Brier 分数、AUROC 等评估指标；在多语言 LLM 上进行实验（Qwen‑3/3.5、Gemma‑4）。

**📊 数据集**

Multilingual True‑False 语料库（8 语言、48K 语句），Belebele 任务集（检验通用任务保持），FActScore 生成长文本并评估事实精度；使用多种 LLM（0.6–14B）。

**📈 对比分析**

与 9 种基线（MSP、Energy、Entropy、LN‑Entropy、NLL、Perplexity、P(True)、P(True)‑norm、Gen‑Z）在 ECE、Brier、AUROC 上进行横向比较。CFP 在 ECE 上最低（≈0.115），AUROC 与 P(True)‑norm 相近（≈0.689）；在低资源语言中，CFP 的低 ECE 与判别力弱相关，但通过证据选择式对齐后，Hindi 的 AUROC 从 57% 提升至 70%，同时保持通用任务精度。

**⚠️ 局限性**

仅评估短、原子事实语句，未覆盖长文本或复杂情境；实验仅至 14B 参数规模，无法验证更大模型；跨语言对齐实验为概念验证，训练规模受限；评估主要依赖自动指标，缺少人类主观评价。

---

## 533. HBFSim: A Simulator for Exploring HBF in Memory Hierarchies for LLMs

**arXiv ID:** 2610.23503 | [PDF](https://arxiv.org/pdf/2610.23503v1)

**作者:** Jinheng Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Chun Jason Xue `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款名为 HBFSim 的事件驱动仿真器，用来评估 HBM+HBF（高带宽闪存）内存体系在大语言模型推理中的性能与耐久性。

**💡 创新点**

创新点包括：① 将 HBM、HBF 与主机内存等多级存储在同一事件驱动框架内闭环仿真；② 细粒度的 HBF 内部行为建模（页程序、擦除、GC、FTL 主机映射）；③ 结合权重与 KV 缓存的放置策略与缓存替换算法的评估；④ 在仿真中加入写耐久性估算，为未来 HBF 设备提供设计指导。

**🔧 技术方法**

使用的技术：事件驱动仿真框架、GPU 计算时间模型、vLLM 调度器、HBF 设备层与主机 FTL 的交互模型、页/区块/块级映射、硬件性能校准与验证。

**📊 数据集**

实验数据集：Llama-3.1-70B 与 Qwen3-235B-A22B 的推理请求轨迹（来自公开的 Mooncake、Qwen 轨迹），以及 vLLM 的工作负载（Prompt/Output 长度等）。

**📈 对比分析**

比较方法：在不同的 HBM/HBF 堆栈配置、不同的放置策略（KV first、权重 first、LRU/FIFO/DecayedLFU）以及不同工作负载下，测量每步时间（TPOT）、吞吐率、写耗与耐久性。实验表明：在容量受限时，HBF 的 TPOT 比 host DRAM 低 11–19 倍、CXL SSD 16–21 倍、NVMe SSD 86–154 倍；但写耐久性在高负载下成为瓶颈。

**⚠️ 局限性**

局限性：① 未使用真实 HBF 设备，所有 HBF 参数均基于公开规范或假设；② 未模拟 GC、写放大、并行写等细节；③ 仅仿真单个 GPU、单批次的推理，不考虑多 GPU 或管道并行；④ 写耐久性估算依赖于已知的 P/E 周期，未考虑不同闪存工艺或耐久性调节策略。

---

## 534. When Evidence Vanishes: Distilling Visual-Evidence Dependency in Medical Vision-Language Models

**arXiv ID:** 2610.23469 | [PDF](https://arxiv.org/pdf/2610.23469v1)

**作者:** Qizhen Lan `[一作]` (University of Texas Health Science Center at Houston), Xiaoqian Jiang `[通讯]` (University of Texas Health Science Center at Houston)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Evidence-Conditioned Dependency Distillation (ECDD)，在知识蒸馏中同时传递教师对原始图像的答案偏好与答案支持随视觉证据移除时的变化；

**💡 创新点**

创新点在于将蒸馏目标从单点点对点迁移到两点对比（原图与匹配的证据缺失控制），并使用下限约束只惩罚依赖性不足；

**🔧 技术方法**

采用知识蒸馏与LoRA微调，构建可匹配证据级别的对照（全图缺失、局部病变遮挡、手术器械掩码），并用margin-drop作为依赖性度量；

**📊 数据集**

使用医学VQA基准：DME VQA（病变存在/定位）、RoboSurg-VQA（器械存在）、PathVQA（全图缺失）；

**📈 对比分析**

与标准SFT、KD、DBiLD、MedAlign、A3Tune、Beta-KD等基线对比，ECDD在保持原始图像准确率基本不变（≤0.1点差）的前提下，显著提升证据缺失时的答案支持差距（如DME正例gap从0.786升至2.023，PathVQA VEA gap从≈5升至20）；

**⚠️ 局限性**

局限性包括：证据控制只测量指定干预下的依赖性，未评估临床鲁棒性；教师过滤可能忽略部分偏差；PathVQA的好处混合了额外的训练阶段与参考KL；未对比在原始与缺失输入同时蒸馏的匹配输入方案。

---

## 535. Reinforcement Learning-Based Pricing and Operation for Hybrid Electric and Hydrogen Vehicle Stations

**arXiv ID:** 2610.23464 | [PDF](https://arxiv.org/pdf/2610.23464v1)

**作者:** Lowell Welburn `[一作]`, Shaked Regev `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究构建了一个集电动汽车与氢燃料电池汽车充电/加氢功能的混合零排放车辆站点仿真框架，并利用PPO深度强化学习实现实时动态定价与能源管理；

**💡 创新点**

创新点在于首次将电池/氢储能与强化学习相结合，对混合EV/HFCV站点进行全局动态定价，且系统性比较了静态、TOU与完全动态策略；

**🔧 技术方法**

采用PPO强化学习、离散/连续动作空间、奖励函数设计、Poisson到达模型以及电网TOU价格动态等技术；

**📊 数据集**

使用仿真生成的随机车辆到达数据（按Poisson分布）、预设的TOU电价曲线以及站点容量参数等；

**📈 对比分析**

与手动、静态、TOU策略对比，动态RL策略实现净利润提升10.83%，服务率提升11%，并在多验证周期内保持稳步增益；

**⚠️ 局限性**

局限性包括未使用真实站点数据验证、能源转换效率设为50%导致模型不使用转换动作、以及仿真环境可能未完全涵盖真实运营复杂性。

---

## 536. ShadowCast: On-the-Fly Context-Aware Shadow Augmentation for Remote Sensing Image Segmentation

**arXiv ID:** 2610.23440 | [PDF](https://arxiv.org/pdf/2610.23440v1)

**作者:** Yechan Kim `[一作]` (LIG Defense and Aerospace), Moongu Jeon `[通讯]` (Gwangju Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ShadowCast，一种在遥感语义分割训练中通过从标签中提取高层建筑与树木掩模并根据图像局部亮度估计阴影投射方向来合成并注入阴影的轻量级数据增强方法。

**💡 创新点**

创新点在于：1) 仅利用已有的语义标签和图像本身的光照信息，完全无学习模型、阴影标注或获取元数据；2) 通过光照亮度启发式估计阴影方向，生成与场景一致的阴影投射；3) 在训练时随机投射阴影，提供多样化的阴影样式，提升模型对真实阴影的鲁棒性。

**🔧 技术方法**

采用的技术包括：二值掩模提取、离散角度投射、亮度均值求取阴影方向、随机角度抖动、随机投射距离与强度采样，以及简单的像素乘法阴影合成。

**📊 数据集**

在三大遥感语义分割基准数据集上进行评估：ISPRS Vaihingen、ISPRS Potsdam 与 LoveDA。

**📈 对比分析**

与基线、随机阴影、PlasmaShadow、SynShadow 等现有阴影增强方法以及多种常见几何/光度增强/正则化方法相比，ShadowCast 在所有数据集和十余种主流分割网络（如 DeepLabV3+、DANet、OCRNet 等）上均实现了平均 0.7–1.3 点的 mIoU 提升，且在阴影区域的性能提升尤为显著。

**⚠️ 局限性**

局限性：阴影方向估计仅基于亮度，易受地表本身暗色或多重阴影的影响；简单的掩模平移可能在复杂地形或多层建筑的阴影投射上产生不准确结果；此外，方法对阴影分布的假设在极端光照或高耸建筑稀缺的场景中可能失效。

---

## 537. TAN: A Size- and Form-Adaptive Masked-Convolution Encoder for Tissue Graphs

**arXiv ID:** 2610.23475 | [PDF](https://arxiv.org/pdf/2610.23475v1)

**作者:** Jan Herschel `[一作]`, Anselm von Gladiss `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种TAN编码器，利用完整的掩码卷积来处理超像素，使组织区域保持完整而不被填充或裁剪影响，构建组织图进行分类。

**💡 创新点**

创新点在于设计了严格的掩码卷积与归一化机制，使节点特征完全不受背景填充值干扰，同时保持模型参数极小，仅0.41M。

**🔧 技术方法**

采用SolidConv（完整3×3窗口的掩码卷积）、Mask-aware GroupNorm、masked 2×2 max‑pooling和masked attention pooling的四阶段卷积网络，实现对超像素原始形状的处理。

**📊 数据集**

使用BACH（400 H&E图像，4类）和BRACS（7类乳腺病变）两个公开组织图数据集进行实验。

**📈 对比分析**

在相同的图分类器（3层GAT）下与UNI、CTransPath、ResNet18等预训练/未预训练模型对比；在BACH上TAN的准确率与303M参数的UNI相当，参数和FLOPs分别低740×和1190×；在BRACS上TAN虽接近ResNet18，但仍落后于UNI。

**⚠️ 局限性**

限制在于缺乏像素级标签导致监督弱，导致整体准确率偏低；超像素大小未经过调优；仅在两数据集验证，模型泛化性和最佳超像素尺度仍需进一步研究。

---

## 538. From Billion-Scale to Million-Scale Language Models: Operational Capabilities for Embedded Post-Exploitation

**arXiv ID:** 2610.23534 | [PDF](https://arxiv.org/pdf/2610.23534v1)

**作者:** Gabriel Hortea `[一作]` (Universidad Carlos III de Madrid), Mario Varlese `[通讯]` (Università degli Studi di Napoli Federico II)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对少于 2 B 参数的 ultra‑Small Language Models（uSLMs）在后期渗透任务（目录枚举与后门生成）中的代码生成能力进行系统评估。

**💡 创新点**

①提出“算法化任务表述”与“逐步最少‑至‑最多（Least‑to‑Most）”交互范式；②构建可重复的多阶段验证管线（静态 + 动态沙箱）；③揭示模型大小、任务表述与提示策略交互对恶意代码成功率的非线性影响。

**🔧 技术方法**

使用提示工程（Zero‑Shot、Few‑Shot、Chain‑of‑Thought、FS+CoT）、结构化解码（EBNF 语法约束）、逐步交互与自我纠错回路；评估指标包括 Pass@k（k=1,3,5）、token 生成量、推理时间。

**📊 数据集**

采用公开预训练的七个 SLM（EleutherAI Pythia 70 M/160 M、HF/SmolLM2‑135 M/360 M、Qwen2.5‑Coder‑0.5 B/1.5 B、nilq/lua‑mistral‑2L）和两种编程语言（Python 与 Lua）构成实验数据集；每种设置在 10 次宏观跑中生成 10 个候选答案。

**📈 对比分析**

与大模型基线（如 Qwen2.5‑Coder‑1.5 B）对比，发现 Pass@5 在最优条件下可达 99.8 %（Python 目录枚举）和 91.1 %（Lua 目录枚举）。小模型（≤ 360 M）在所有任务中始终保持 0 % Pass@5。提示策略对结果影响有限，任务表述决定性，算法化表述对大模型提升显著，但对小模型效果不佳。结构化解码反而降低成功率。

**⚠️ 局限性**

仅评估无微调、无量化的公开权重模型；实验语言仅限 Python 与 Lua，无法覆盖 C/C++、PowerShell 等常见恶意语言；提示词、算法化拆解为手工设计，可能带来偏差；自我纠错实验受限于 10 步上限；缺乏对压缩部署（GGUF/GPTQ）对性能与隐蔽性的影响研究。

---

## 539. Out of Support, Not Out of Reach: In-Context Learning with Unsupported Instructions and Composed Tasks

**arXiv ID:** 2610.23528 | [PDF](https://arxiv.org/pdf/2610.23528v1)

**作者:** Xuhan Tong `[一作]`, Jiawei Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a2602d71-93ab-4bad-974b-672788df8193` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在大型语言模型的In-Context Learning（ICL）中，当指令或任务不在预训练支持集内时的泛化行为，并给出了理论与实验的分析框架。

**💡 创新点**

创新点包括：①将指令级支持与任务级支持区分并分别给出指数收敛定理；②引入“模型诱导延续”概念，解释在不支持的指令下仍能通过示例积累任务证据；③将长时限任务拆分为支持子任务，给出误差累积的显式上界。

**🔧 技术方法**

主要技术手段包括：隐式任务贝叶斯推断、连续路径分析、路径Lipschitz常数、模型-参考传输误差量化，以及基于子任务执行的逐步推理模型。

**📊 数据集**

实验使用的公开数据集包括Sport Identification（国家运动识别）、三位数加法任务、Qwen3 30B与235B模型的指令与示例集合，以及自构造的七位数加法等多步执行数据。

**📈 对比分析**

与基线的比较主要通过示例数量、指令一致性/多样性以及子任务支持度来评估，实验结果显示：①识别性示例显著提升准确率；②一致指令匹配下的梯度敏感性随示例增多而下降；③在支持子任务的执行计划下的误差比单步直接预测更小；总体性能提升幅度在10-30%不等。

**⚠️ 局限性**

局限性包括：①理论假设需满足连续延续与模型-参考误差可控，实际模型可能不满足；②对不支持子任务的长链累积误差上界可能过于宽松；③实验仅覆盖有限任务与模型，需进一步验证在更复杂指令与跨模态任务中的适用性。

---

## 540. Complexity of liveness in ring Petri nets and of halting in chip-firing on simple digraphs

**arXiv ID:** 2610.23523 | [PDF](https://arxiv.org/pdf/2610.23523v1)

**作者:** Eike Best `[一作]`, Matěj Ošťádal `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明了环Petri网（以及其推广的加权T系统）的活性判定问题属于PSPACE，并且在强意义上是PSPACE‑难的；同时，它给出了一种构造，将该问题归约到简单有向图上的堆叠游戏（chip‑firing）停机问题，从而证明该停机问题在简单有向图上也是PSPACE‑完备的。

**💡 创新点**

核心创新在于利用覆盖余数类（residue classes）的覆盖问题作为基准，通过构造只含单一循环的标记环，使得活性判定等价于检测某种线性整数系统的可解性，并进一步将此构造映射到chip‑firing模型，首次给出两类问题在强意义上的PSPACE‑难度。

**🔧 技术方法**

主要技术包括：对Petri网状态方程的整数线性规划分析、Papadimitriou等人关于整数解大小的上界、余数类覆盖问题的多项式时间归约、素数与中国剩余定理的巧妙使用，以及在chip‑firing模型中构造“路径”对应Petri网转移的技巧。

**📊 数据集**

本工作为纯理论研究，没有使用实验数据集；所有证明均基于构造性归约和组合数理分析。

**📈 对比分析**

由于本研究关注的是理论复杂度，未进行实验性对比；理论结果表明，无论是环Petri网还是加权T系统的活性判定，甚至在最简化的简单有向图的chip‑firing停机问题，都不能通过多项式时间或伪多项式时间算法解决。

**⚠️ 局限性**

局限性在于：该结果仅覆盖单一环或其加权T系统形式，对更一般的Petri网结构（非单环、多连通等）的活性判定复杂度仍未确定；此外，尽管证明了强意义上的PSPACE‑难度，但对特定易解子类的有效算法研究仍是开放问题。

---

## 541. MLOps-Bench: Benchmarking AI Agents on Production ML Engineering in Real Repositories

**arXiv ID:** 2610.23506 | [PDF](https://arxiv.org/pdf/2610.23506v1)

**作者:** Quoc Co Tran `[一作]` (Oracle), Katrin Kirchhoff `[通讯]` (Oracle)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MLOps-Bench，一个基于真实ML仓库的可执行评测基准，用于评估AI代理在生产级机器学习工程中的能力。

**💡 创新点**

创新点在于：①仓库根植的任务生成，保证任务与仓库内部生命周期证据紧密对应；②覆盖数据管道、特征工程、模型训练、服务、监控、CI/CD与治理以及端到端集成的完整生命周期；③为每个任务提供oracle补丁、buggy基线以及fail-to-pass / pass-to-pass可执行测试，真正检验代理的行为正确性。

**🔧 技术方法**

采用基于程序分析和上下文提取的流水线生成任务；使用自然语言需求与代码补丁的自动化合成；利用可执行测试套件评估代理的补丁；对代理执行环境进行隔离与测试。

**📊 数据集**

从31个公开开源ML仓库中抓取数据，冻结快照后生成212个任务，包含636个fail-to-pass测试和856个pass-to-pass测试。

**📈 对比分析**

在六种代理（Codex、Claude Code Sonnet/Opus、Base Agent Sonnet/GPT‑5.5/5.2）上进行评测，Codex最高完成率为31.2%，测试通过率为68.7%，其余代理表现相对较弱。

**⚠️ 局限性**

局限性包括：①仅涵盖公开仓库，缺乏真实商业部署的生产环境；②任务由生成模型合成，可能与真实生产缺陷偏离；③生成与评测使用的模型不同步，可能存在对同族模型的偏好；④测试套件无法覆盖所有复杂的运行时交互与大规模工作负载。

---

## 542. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents

**arXiv ID:** 2610.23479 | [PDF](https://arxiv.org/pdf/2610.23479v1)

**作者:** Hongcheng Gao `[一作]`, Xiao He `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套名为HexaShell的机器人操控框架，利用视觉语言模型与三摄RGB图像，通过简单的命令行与机器人交互，并将失败经验转化为可重用的工具程序；

**💡 创新点**

创新点在于：① 采用极简的“薄掌”架构，将所有决策权交给语言模型，保持工具的可持续改进；② 通过三角测量（而非深度网络）精准获取三维位置信息；③ 设计了一套自动化修复循环，让编码代理在每次失败后快速诊断并生成/修复工具，保持知识的可编程化；

**🔧 技术方法**

使用的技术包括：Vision‑Language Model（Codex + GPT‑6‑Astra）在中等推理努力下执行；三摄RGB图像与标定信息的三角测量；OpenCV几何工具；机器人控制通过固定命令接口实现；

**📊 数据集**

使用数据集：RoboDojo模拟器中的42个双臂操控任务（包含Generalization、Memory、Long‑Horizon、Precision、Open等子任务），每个任务提供十个新布局进行重测；

**📈 对比分析**

与现有学习策略（VPP2、PhysicalRSI 等）在RoboDojo排行榜上对比，HexaShell在所有42任务上取得39.7%成功率、48.0进度得分，超越VPP2（32.3%/39.3）和PhysicalRSI（31.4%/36.3），并在多任务类别上实现了状态‑艺术；

**⚠️ 局限性**

局限性主要在接触精度与运动规划：紧插入（tight insertion）等需要毫米级精度的任务成功率仅8.3%；受限于无深度信息，工具在某些细致定位或需要连续测量的情境下表现不佳；此外，过多的视角采集会消耗动作步数，导致部分任务因步数耗尽而失败。

---

## 543. Graphlet-based Decentralized Finance Transaction Method Classification

**arXiv ID:** 2610.23466 | [PDF](https://arxiv.org/pdf/2610.23466v1)

**作者:** Célestin Coquidé `[一作]` (Université Marie et Louis Pasteur), Natkamon Tovanich `[通讯]` (TU Wien)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过构建基于Token转移网络的有向图并提取图形子结构(graphlets)特征，本文研究了是否能根据DeFi交易的拓扑结构来识别交易方法，并将此方法与事件日志计数进行对比；

**💡 创新点**

创新点在于：①将传统的无向图形子结构扩展为有向、多属性的graphlets，并结合账户类型和Token种类进行特征丰富；②利用SHAP对模型进行解释，揭示各交易方法对应的结构化网络签名；

**🔧 技术方法**

主要技术包括：图形子结构（graphlet）匹配与计数、特征抽取（节点/边位置、类型），AutoGluon自动机器学习与加权集成模型，SHAP解释方法；

**📊 数据集**

使用以太坊链上从区块20,000,000到20,019,999共20,000个区块的数据，训练集10,000区块（590,589笔交易）与测试集10,000区块（666,342笔交易），覆盖30个最频繁的交易方法；

**📈 对比分析**

实验比较了单独的graphlet特征、事件日志计数特征以及两者的组合。单独graphlet的宏F1≈0.693，事件日志单独F1≈0.793，二者组合F1提升至0.918，证明两种特征互补；

**⚠️ 局限性**

局限性包括：仅覆盖单一时间窗口，未检验对不同时间段或未知合约的泛化；特征空间相对稀疏且未与传统网络度量对比；解释阈值采用经验设定，缺乏自适应阈值机制。

---

## 544. HydraLLM: Adaptive Width Slicing for Elastic Text and Vision-Language Models

**arXiv ID:** 2610.23455 | [PDF](https://arxiv.org/pdf/2610.23455v1)

**作者:** Oscar Kemme `[一作]` (Hamburg University of Technology), Olaf Landsiedel `[通讯]` (Hamburg University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 HydraLLM，一种通过动态切换注意力头数实现宽度弹性的大语言模型和视觉语言模型架构，能够在单一参数集上生成嵌套子网络。

**💡 创新点**

创新点在于将 MQA 结构与前缀切片相结合，利用重要性重排序和随机子网络训练，使得预训练模型可无缝转换为弹性模型，并扩展到多模态投影。

**🔧 技术方法**

使用的技术包括多头注意力（MQA）、嵌套前缀切片、重要性（CETT、激活幅度）重排序、随机子网络训练、知识蒸馏、弹性投影等。

**📊 数据集**

使用的主要数据集包括 Gemma 3 270M、FineWeb‑Edu 350B、PixMoCap、COCO Caption、LLaVA Pretrain、WikiText 等文本与多模态数据。

**📈 对比分析**

与 MatFormer 的 FFN 切片相比，HydraLLM 在子模型大小覆盖范围更广、TriviaQA 精度提升 2.2%、Token‑generation 速度提升 75%；多模态版本 CLIPScore 与 Gemma 3 4B 相当，且在较小子网络下仍保持可用。

**⚠️ 局限性**

局限包括极小子网络准确率低、生成文本存在幻觉和错误、投影层容量受限、以及多模态下的稳健性与准确性仍未达到理想水平。

---

## 545. TriINR: Unsupervised Atmospheric Turbulence Mitigation with Triple Implicit Neural Representations

**arXiv ID:** 2610.23454 | [PDF](https://arxiv.org/pdf/2610.23454v1)

**作者:** Yuka Ogino `[一作]` (NEC Corporation), Atsushi Ito `[通讯]` (NEC Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无监督的三重隐式神经表示（TriINR）方法，从多帧大气湍流失真图像中恢复清晰图像。

**💡 创新点**

将清晰图像、倾斜和模糊分别建模为三个隐式神经网络，并使倾斜与模糊成为时空可变的隐式表示，解决了传统方法帧数相关、参数随帧数增多的问题。

**🔧 技术方法**

利用隐式神经表示（INR）、物理与统计约束以及无监督优化技术。

**📊 数据集**

在合成的不同强度湍流数据集以及真实长距离场景数据上进行实验。

**📈 对比分析**

与现有无监督湍流消除方法对比，TriINR在三种湍流强度下表现更稳健，并在所有评估的无监督方法中取得最高的恢复精度。

**⚠️ 局限性**

对极端湍流强度的处理仍有改进空间，并且在含有非理想运动的真实场景中鲁棒性尚未完全达到理想水平。

---

## 546. GatewayBench: Unified Capability and Security Evaluation Platform for MCP Gateways

**arXiv ID:** 2610.23449 | [PDF](https://arxiv.org/pdf/2610.23449v1)

**作者:** Yitian Zheng `[一作]` (Tsinghua University), Bo Li `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对Model Context Protocol（MCP）网关的统一评估框架与基准，系统衡量网关架构对LLM代理在任务完成、效率与安全性上的影响。

**💡 创新点**

首次构建了包含524个多域任务、11,118个真实噪声工具池的评测数据集，并将网关架构分为扁平化与层级化两类，系统实验揭示了其在工具规模、模型能力与攻击场景下的差异性。

**🔧 技术方法**

利用LLM（GPT‑5.1/5.4）与代理框架（OpenAI Agents SDK、OpenClaw），结合可验证的判定器，对七种MCP网关实现进行量化评测；通过注入噪声工具模拟大规模部署。

**📊 数据集**

任务来源于DecodingTrust-Agent Platform与MCP-Universe；工具来源于Docker MCP Hub与Arcade.dev，筛选得到任务相关与不相关工具池。

**📈 对比分析**

对比指标包括任务成功率(TSR)、攻击成功率(ASR)、token消耗与任务时长；实验表明层级化网关在大规模工具集与间接攻击下具有更高安全性，而扁平化网关在小工具集上更快、更低token；强模型可缩小两者性能差距。

**⚠️ 局限性**

局限性包括实验仅覆盖7个网关实现，工具池为mock环境，未考虑真实执行时的权限与副作用；只评估两种模型与两种框架，未探讨更多多模态或自监督模型；对动态工具更新和多代理协作的影响未深入研究。

---

## 547. Regularized free deconvolution for spectral repair of resampling distortion in high-dimensional particle filters

**arXiv ID:** 2610.23448 | [PDF](https://arxiv.org/pdf/2610.23448v1)

**作者:** Deming Song `[一作]` (Macao Polytechnic University), Huagang Lin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究在高维粒子滤波中重采样对协方差谱的可计算失真，并提出一种正则化自由去卷积方法来恢复原始谱并将其用于 Metropolis 预条件器，从而在不改变目标分布的前提下提升采样效率。

**💡 创新点**

创新点在于：①在条件化权重与已实现计数的前提下将谱失真定律推广到随机重要性权重；②构造了带误差预算的正则化自由去卷积估计器并给出可认证的逆条件数；③将校正后的谱作为无偏预条件器，引入“放弃修复”策略并给出其成本与几何负载的理论上限。

**🔧 技术方法**

技术主要包括：高维随机矩阵理论与自由概率的乘法卷积、正则化 Tikhonov 最小距离估计、Wasserstein 与 Stieltjes 失真度量、Metropolis-Hastings 移动、以及自监督的误差认证与放弃决策。

**📊 数据集**

使用了合成高维正态样本集，并在预注册实验中采用多种权重分布（Gamma 参数 α=10,1,0.3）和两种重采样方案（多项式与残差）进行测试，样本规模 N=500,1000,2000，维度 d 取 225–750。

**📈 对比分析**

与传统 Ledoit–Wolf 非线性收缩、未修复的重采样谱、以及oracle 逆算相比，正则化去卷积在谱误差上实现了近一阶（N⁻¹/2）速率，恢复了约 90% 的 oracle 采样效率；在全周期实验中，校正后的轨迹收敛至低噪声平台，且放弃策略的几何负载决定误差增长。

**⚠️ 局限性**

主要局限包括：仅证明了高斯扰动的正则化去卷积（非普适性待证）；假设权重为外生且与状态无关（内生权重需进一步研究）；误差预算只在固定分辨率 θ 下有效，无法消除分辨率下限；递归收敛只在标量稳定动力学下证明，通用动力学仍未覆盖。

---

## 548. "You" Is Not Toxic: How LLM Labelers Rewrite the Web's Conversation Laws

**arXiv ID:** 2610.23446 | [PDF](https://arxiv.org/pdf/2610.23446v1)

**作者:** Liang Yue `[一作]` (University of Colorado Boulder), Eric A. Vance `[通讯]` (University of Colorado Boulder)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析了大规模语言模型（LLM）在自动标注网络对话毒性时对“你”二人称代词的过度识别，从而误制造毒性对话加剧的统计规律；同时提出一种基于采样与模型校正的“法则认证”方法，能够在不需要全量人工标注的情况下，以几千条标注数据估计并校正毒性相关线性法则（深度、大小与传染性）并给出置信区间；

**💡 创新点**

创新点在于（1）揭示LLM标注偏差（对二人称代词的错误识别）如何导致法则失真；（2）提出可解释且可证明的法则认证框架，利用结构化假设和自适应抽样来确保估计无偏并且具有良好覆盖率；（3）通过系统评估多种LLM和传统分类器在多语料库上的表现，说明即使高一致性（κ）也不能保证法则的真实性；

**🔧 技术方法**

主要技术包括：大规模LLM标注（Claude Haiku、Sonnet、Qwen等）、自适应采样设计与温度化的概率模型、线性法则的误差分解（TPR‑FPR项与结构化假设项）、可重复的实验框架与Bootstrap置信区间估计、以及对“你”二人称代词的文本重写实验；

**📊 数据集**

使用的数据集包括：Civil Comments新闻评论（1.25M条），其5%子样本；另外四个公开语料库——CAD（Reddit滥用）、Qian等的Reddit与Gab仇恨语料、CGA‑Wiki（维基百科讨论攻击）以及BAD（人机对话安全标注）；

**📈 对比分析**

比较方法：对每个标注器在相同数据上计算深度、大小、传染性等线性法则，并与人工金标的法则做偏差分解；利用“安慰剂”统计量评估结构化误差；在模拟审计中使用PPI++、DSL、主动推断等对比方法，评估95%置信区间覆盖率和宽度。性能结果显示：LLM（如Haiku、Sonnet）在二人称代词上高FPR导致深度法则出现显著上升；传统分类器多在无结构误差范围内；而所提法则认证在250–8000条标注下平均覆盖率达94.9%，区间宽度比PPI++或DSL小约25%；

**⚠️ 局限性**

局限性：①实验主要针对文本级别的二人称代词，其他潜在偏差（如情境、语气、文化差异）未被系统评估；②所提出的认证方法依赖于对标注器分数的可解释性和训练数据的可重复性，若LLM输出不透明则难以实施；③在极端稀缺标签环境下，抽样设计仍可能因误差传播导致宽区间；④最终法则的实际解释性受限于金标定义和领域差异。

---

## 549. RECREG: Recovery-based Shielding for Regular Safety Properties in Reinforcement Learning

**arXiv ID:** 2610.23445 | [PDF](https://arxiv.org/pdf/2610.23445v1)

**作者:** Alexander W. Goodall `[一作]` (Imperial College London), Francesco Belardinelli `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出RecReg框架，利用恢复式屏蔽（recovery-based shielding）在强化学习中实现对正则安全属性的高概率满足；

**💡 创新点**

在不完全模型知识下给出三种模式（Exact、ModelBased、ModelFree），提供显式的安全性、收敛性与最优性理论保证，并设计安全感知Bellman算子；

**🔧 技术方法**

结合MDP与DFA的乘积构造、概率安全集定义、最终奖励与成本函数、深度模型基础的DreamerV3 RSSM、成本头、隐层滚动等技术；

**📊 数据集**

在MASA‑Safe‑RL的离散环境（网格、Pacman等）和Atari Seaquest、SafetyGym等高维图像环境上进行实验；

**📈 对比分析**

与多种基线（LCRL、QL‑λ、SEM、Recovery‑RL、PPO‑Lag、TRPO‑Lag、CPO、ProbShield等）比较，RecReg在保证安全概率的前提下，回报更高，尤其在Exact模式下表现最佳；

**⚠️ 局限性**

对大状态空间估计与探索挑战敏感；对连续空间需精确模型校准；分布式风险分配仍有改进空间。

---

## 550. No Eyes, No Problem: What Does Touch Contribute to Blind Baoding Ball Manipulation?

**arXiv ID:** 2610.23438 | [PDF](https://arxiv.org/pdf/2610.23438v1)

**作者:** Ayush Deshmukh `[一作]` (University of Edinburgh), Sethu Vijayakumar `[通讯]` (University of Edinburgh)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过仅使用关节编码器和16路二进制触觉信号，完成了Shadow Dexterous Hand Lite在实物上连续旋转Baoding球的零射程仿真到现实迁移实验。

**💡 创新点**

创新点包括：与真实手臂完全对齐的关节耦合模型、触觉校准、两种新的情节级域随机化（速率限制SlewDR和随机失效通道STAT），以及基于二进制触觉的长时间可持续操控。

**🔧 技术方法**

使用了基于PPO的强化学习、Isaac Lab仿真平台、触觉阈值双极性滞后校准、动作速率限制与STAT随机化技术。

**📊 数据集**

使用的数据集为仿真中的随机化参数（球质量、摩擦系数）以及两颗1.58英寸、55g Baoding球；并在真实机器人上使用相同的球进行10次试验。

**📈 对比分析**

与开环、单独关节感知、无触觉的C3-TacOff以及之前的相关工作（Nagabandi、Yuan、Yang）对比，C3在硬件上平均完成34.75±26.06次完整360°旋转，时间为94.8±68s，旋转速度0.36±0.06 rot/s，显著优于所有基线和前人实验。

**⚠️ 局限性**

局限性包括：仅针对双球圆形物体，未验证对不同尺寸或形状物体的泛化；依赖精确的触觉校准和二进制触觉传感器；高方差的持续时间与旋转次数；以及对复杂或多物体场景的适用性未知。

---

## 551. PONDER: Learning to Generate in the Right Order

**arXiv ID:** 2610.23531 | [PDF](https://arxiv.org/pdf/2610.23531v1)

**作者:** Julius Erbach `[一作]` (ETH Zurich), Konrad Schindler `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PONDER方法，通过学习位置选择策略来决定图像生成顺序，并支持重新生成先前位置（rethinking）。

**💡 创新点**

创新点在于把生成顺序和重思维视为可学习决策，用分组相对策略梯度训练，并引入FIDgrad奖励将分布级指标转化为样本级。

**🔧 技术方法**

使用强化学习（group-relative policy gradients）、Transformer位置策略、冻结的连续token扩散生成器以及自定义的FIDgrad奖励。

**📊 数据集**

评估数据集包括ImageNet、MNIST Sudoku、以及Text-to-Image的官方HPSv2和GenEval基准。

**📈 对比分析**

与随机、固定或基于不确定性的贪婪策略相比，PONDER在ImageNet FID从4.09降至2.70，在Sudoku硬分辨率准确率从3.9%提升至96.7%，在HPSv2得分从24.5提升至27.8。

**⚠️ 局限性**

局限在于政策需要额外前向计算，且性能高度依赖奖励设计，重思维预算是预先设定而非动态分配。

---

## 552. EduWild: How Teacher-Configured Chatbots Fail in Real-World Use and How to Fix Them

**arXiv ID:** 2610.23530 | [PDF](https://arxiv.org/pdf/2610.23530v1)

**作者:** Ziqi Liu `[一作]` (Carnegie Mellon University), John Stamper `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个包含1,726条真实世界教师配置教育聊天机器人的对话数据集，并设计了29种失效分类与基于提示的harness来降低失效率。

**💡 创新点**

首次提供教师意图与实际对话配对的数据集、基于人类校准LLM的失效注释框架以及针对最常见失效的prompt harness，弥补了先前合成或非教师配置数据的不足。

**🔧 技术方法**

利用Claude/GPT等LLM提取教师意图、构造持久记忆，并使用GPT-5.6-Sol判别器自动标注失效；harness通过六个针对性prompt与原始教师prompt组合生成响应。

**📊 数据集**

使用来自Playlab平台的217位教师创作的217个聊天机器人（其中35个已部署）共1,726条对话，其中900条来自终端用户，形成核心数据集。

**📈 对比分析**

在200名大学生的对照实验中，harness将平均目标失效代码数从1.406降至0.489，实现65.2%相对下降；在认知过载和多问题过载上分别降幅超过80%与96%。

**⚠️ 局限性**

研究仅覆盖成人学习者、单一平台、角色扮演实验场景，缺乏真实学习任务和学习成效评估；harness仍未完全消除工作流程缺失和过早答复等问题。

---

## 553. Perceive, Solve, Reason: Progressive Capability Formation in Time Series-LLMs

**arXiv ID:** 2610.23521 | [PDF](https://arxiv.org/pdf/2610.23521v1)

**作者:** Xin Qiu `[一作]` (Eastern Institute of Technology), Xiaoyu Shen `[通讯]` (Eastern Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5a41884c-404f-4688-a89c-aa238c10fe68` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了首个覆盖时间序列感知、任务求解与推理三大能力的训练与评估套件，并系统研究了多能力训练顺序、输入格式、阶段成熟度与层级表示迁移对能力形成的影响，提出了按阶段逐步训练的实用策略。

**💡 创新点**

创新点包括：①首次提出统一的三阶段能力框架（Perceive‑Solve‑Reason）；②发现专用时间序列编码器和按阶段训练显著提升推理性能；③揭示能力转移通过不同网络深度实现，而非简单堆叠；④证明局部阶段收敛不代表后续能力已充分准备。

**🔧 技术方法**

技术手段主要为预对齐（Pre‑align）与后对齐（Post‑align）两种编码器-LLM对齐架构，使用 Qwen3‑4B 作为 LLM 基座，采用渐进式训练、梯度对齐与表示迁移分析等方法。

**📊 数据集**

使用了多源公开数据集：ChatTS、TSQA、TSInstruct、TSCognition 等共计约 337K 训练样本和 44K 评估样本，涵盖趋势、季节性、异常检测、预测、分类等任务；并在 MTBench、TimeSeriesExam、CaTS‑Bench 等 OOD 任务上进行验证。

**📈 对比分析**

与四个 7B 规模 TS‑LLM（TimeOmni‑1、ITFormer、ChatTS、PATRA）以及 GPT‑4o 进行对比，4B 规模的 TS‑LLM 在多项 OOD 评测上实现或近似最佳成绩，推理指标最高提升 11 分，整体表现显著优于对比模型。

**⚠️ 局限性**

局限性包括：①仅在单一 LLM 基座 Qwen3‑4B 上验证；②对编码器的冻结与微调策略可能不适用于更大或不同模型；③缺乏对长序列多模态输入的深入探索；④实验主要集中在公开数据集，真实业务场景中的可迁移性待进一步验证。

---

## 554. Uncertainty-aware AI-enabled framework for predicting hydrogen dispersion in confined spaces to enhance operator situational awareness

**arXiv ID:** 2610.23515 | [PDF](https://arxiv.org/pdf/2610.23515v1)

**作者:** Niclas Flehmig `[一作]` (Norwegian University of Science and Technology), Shen Yin `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种两阶段、基于不确定性推理的 AI 框架，用于在船舶加氢作业时实时预测受限空间内氢气泄漏的泄漏率和浓度场，并将预测结果以置信区间呈现给操作员。

**💡 创新点**

创新点包括：① 将泄漏率估计（MLP 集成）与浓度预测（可解释的加性高斯过程）两大模型耦合成单一决策支持系统；② 在高斯过程建模中使用“加性核筛选+精简”策略，显著提升模型可解释性与计算效率；③ 将 ISO/IEC TR 5469 级别“D”安全架构与 AI 监测、输入防护相结合，构建可安全部署的安全相关系统。

**🔧 技术方法**

技术主要包括：多层感知机集成、特征工程（10 秒窗口内的时空特征）、加性高斯过程回归、VNNGP 与 SVGP 变分稀疏近似、Beta 与 Gaussian 似然比较、模型校准（PICP、PIT）与在线推理算法。

**📊 数据集**

使用公开的“受限空间氢气释放”实验数据集，共 32 个测试（22 实验 + 10 CFD），包含 29 传感器、不同泄漏速率（0.04–1.27 g/s）和时间序列浓度读数；数据在训练/验证/测试中按实验划分以防止时序泄漏。

**📈 对比分析**

对比方法：对照基于全加性核的 VNNGP、SVGP 以及 Beta/Gaussian 似然的不同配置。最优 SVGP（6000 个诱导点、Beta 似然、加性核）在测试集上达到 MAE 0.0176、RMSE 0.0226、NLL -1.534，逼近传感器精度 ±0.01，且置信区间覆盖率在 95% 以上。两阶段系统相比单纯使用实际泄漏率的 “oracle” 方案误差提升约 0.007 MAE，但仍保持安全可接受范围。

**⚠️ 局限性**

局限性包括：① 数据量有限且泄漏速率分布偏斜，导致高流量场景预测欠准；② 仅验证了单一矩形通道几何，未验证对船舶实际阀门空间的迁移性；③ 模型在极端泄漏率、极低浓度区域的校准仍不完美；④ 仍需进一步实现并验证输入防护、AI 监测与实际操作员培训。

---

## 555. LSNR: Learned Sparse News Recommendations with Attentive User Modeling

**arXiv ID:** 2610.23510 | [PDF](https://arxiv.org/pdf/2610.23510v1)

**作者:** David Vos `[一作]` (University of Amsterdam), Maarten de Rijke `[通讯]` (University of Amsterdam)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对新闻推荐问题，提出一种基于学习稀疏表示的新闻推荐模型LSNR，采用每篇新闻独立编码并通过注意力机制聚合成用户稀疏向量，随后用稀疏检索进行匹配。

**💡 创新点**

创新点在于：①放弃STAR的历史拼接做法，改为每篇文章独立稀疏编码；②设计双路径（词汇聚合+语义投影）用户编码器，学习自适应权重；③通过注意力聚合提升稀疏召回能力，并在短历史下仍保持优于稠密模型。

**🔧 技术方法**

技术主要包括：SPLADE词袋稀疏表示、BERT（以及MALTEHB/danish-bert-botxo）预训练语言模型、单层Transformer上下文化稠密向量、加权注意力聚合、ℒ_FLOPS稀疏正则化、倒排索引检索。

**📊 数据集**

使用公开新闻推荐基准：Microsoft News Dataset（MIND，英文）和Ekstra Bladet News Recommendation Dataset（EBNeRD，丹麦语）。

**📈 对比分析**

与BM25、STAR（稀疏基线）以及NRMS、NAML、LSTUR（稠密基线）对比；在MIND上LSNR NDCG@10提升约11%（STAR），与稠密模型相当；在EBNeRD上LSNR略逊于最强稠密模型，但在短历史（N=1）下明显领先；同时索引体积比稠密模型小十倍，检索延迟仅略高。

**⚠️ 局限性**

局限性包括：①在完整历史下在EBNeRD仍比稠密模型低2–3%；②检索速度不如精调的稠密HNSW索引；③仅在“小规模”数据集和两种语言验证，缺乏更大规模和多域泛化；④STAR与LSNR在训练起点不对齐，可能低估LSNR优势；⑤未结合类别、用户特征等辅助信号。

---

## 556. A Topological Signature of Learning under Noisy Labels

**arXiv ID:** 2610.23491 | [PDF](https://arxiv.org/pdf/2610.23491v1)

**作者:** Zhongtian Sun `[一作]` (University of Kent), Kelin Xia `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在噪声标签训练下，神经网络表示的几何变化，利用持久同调捕捉特征点云的环结构。

**💡 创新点**

创新点在于将持久同调与激活尺度控制相结合，提出在清洗标签与噪声子集之间比较的固定探针方法，并通过跨时间的“交叉”检测展示噪声如何在训练后期提升持久度。

**🔧 技术方法**

使用的技术包括Vietoris–Rips持久同调（Ripser）、多种距离/归一化构造（欧氏、余弦、单位化、RMS、z-score）、持久度统计（总持久度、循环计数、比值）、噪声率校准回归、局部PCA维度分析以及对比实验。

**📊 数据集**

实验数据集涵盖图像（CIFAR‑10/100、MNIST、FashionMNIST、ImageNette、Tiny‑ImageNet）、文本（AG News、DistilBERT）以及算术（模加模乘），使用的网络包括ResNet、ViT、MLP、Transformer等。

**📈 对比分析**

通过在同一训练模型下对等大小的清洁与噪声子集计算持久度、归一化后比较，并与均值激活范数等传统噪声估计指标对齐，发现噪声子集在训练后期持久度显著提升，均值范数在噪声率校准上表现更好。

**⚠️ 局限性**

局限性包括结果高度依赖数据集与网络结构，样本数和阈值选择会影响持久度统计；仅对分段仿射网络测试，未覆盖LayerNorm或注意力机制；在grokking实验仅验证了模块算术任务；缺乏对单个样本层面噪声影响的定量解释。

---

## 557. MuninAI: A Self-Hosted Agentic Framework for Independent Research Groups

**arXiv ID:** 2610.23480 | [PDF](https://arxiv.org/pdf/2610.23480v1)

**作者:** Markus Fischer `[一作]` (University of Leipzig), Matthias Elgeti `[通讯]` (University of Leipzig)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Munin，一个自托管、面向科研团队的 agentic 框架，能够对团队内部的文献库和引用图进行索引，并在此基础上为任意开放权重 LLM 提供工具调用循环，实现文献检索、全文阅读、代码执行、深度研究等功能。

**💡 创新点**

创新点在于：①将 agentic 循环与检索、引用图、全文解析、代码沙箱等模块耦合到一个可插拔的框架；②提供完整的评估套件和 frozen scorecards，使实验者可在自家语料库与模型上快速复现结果；③通过“来源留白”与“假文献”两种基准，系统性评估了模型在缺失答案时的拒绝与引用错误率。

**🔧 技术方法**

主要技术包括：多步工具调用循环（≈45 个工具）、向量检索（BGE‑large + BM25）、引用图构建与稀疏重排名、PDF 解析与增量向量化、代码沙箱（Python、C++、R）以及 MiniCheck‑Flan‑T5 语义真确性判定；框架设计兼容任意 OpenAI‑compatible API。

**📊 数据集**

使用的数据集：自建科研团队文献库（约 3,900 篇原始上传，经过引用抓取扩展至 68,500 篇）；LitQA2（199 题）作为问答基准；LitSearch（597 题）和 SciFact（300 题）用于检索性能评估；此外内部构造的假文献与影子语料库用于测量拒绝与假引用。

**📈 对比分析**

比较方法：对同一批 199 题分别运行三种模式（裸 LLM、单轮 RAG、完整 agentic harness），并在三种 backbone（Qwen3.6、Qwen3.8、gpt‑oss‑20b）下进行。结果显示：agentic harness 在 Qwen 系列上将准确率从 ~0.38 提升至 ~0.87（增益≈+0.49），并将每答错误率降低到 0.07；在 gpt‑oss‑20b 上提升幅度较小（从 0.41 提升至 0.56）。另外 grounding 通过 MiniCheck 提升 0.28→0.63；拒绝率在缺失答案时可达 90%+，假引用率为 0%。

**⚠️ 局限性**

局限性：①评估套件依赖内部语料质量，若语料错误会影响结果；②不同模型对工具调用的兼容性差异需要手动调整路由策略；③对外检索依赖商用 API，若切断外网会导致准确率下降；④模型在多源综合时仍可能出现跨文献合成错误，grounding 仍以单句为基准；⑤大规模部署仍受 GPU 资源与并发限制。

---

## 558. S-GAP: Stealthy Graph Attack Primitives for Graph Large Language Models

**arXiv ID:** 2610.23465 | [PDF](https://arxiv.org/pdf/2610.23465v1)

**作者:** Wenda Li `[一作]` (Zhejiang University), Zujie Ren `[通讯]` (Zhejiang Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Graph Attack Primitive (GAP) 与 Stealthy GAP 多代理框架，通过统一的语义意图同时生成文本和结构扰动，对 GraphLLMs 进行攻击。

**💡 创新点**

通过 GAP 实现文本与结构扰动的统一意图，消除攻击意图碎片化，并结合多代理迭代优化与文本反馈提升攻击隐蔽性与效果。

**🔧 技术方法**

采用多代理规划（Planner、Attack、Main）、文本完备 LLM 生成、代理式评估（surrogate GNN）及迭代文本反馈等技术，构建可解释且高效的攻击流程。

**📊 数据集**

实验使用四个文本属性图数据集：Cora、PubMed、OGBN‑Arxiv、OGBN‑Products。

**📈 对比分析**

在四数据集三 GraphLLMs（LLaGA、RGLM、TEA‑GLM）上与 GIA/GMA 原始方法及其他基线对比，攻击成功率平均提升 7.02%，并显著提高关系可解释性。

**⚠️ 局限性**

受限于代理模型与目标模型的差距导致评分误差；对高可解释性节点的依赖可能限制在更大规模或更复杂图上的可扩展性。

---

## 559. Predictable Scaling: Efficient Model Exploration via Scaling-Law Prediction

**arXiv ID:** 2610.23453 | [PDF](https://arxiv.org/pdf/2610.23453v1)

**作者:** Aishwarya Wesanekar `[一作]` (Meta Platforms, Inc.), Pan Zhong `[通讯]` (Meta Platforms, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了Predictable Scaling（PS）框架，利用可组合缩放函数和经验高斯过程在工业推荐系统中以显著降低计算成本预测模型假设的相对性能差距。

**💡 创新点**

首次在异构推荐模型上应用低维平滑缩放规律进行相对性能预测，并提供可校准的不确定性估计，实现 79–88% 的计算节省。

**🔧 技术方法**

使用六类可组合缩放函数（如数据下采样、嵌入裁剪等）、线性回归与经验高斯过程预测器、代理训练与经验曲线拟合技术。

**📊 数据集**

在公开的 Criteo 大规模交互日志上评估，实验使用两类模型（TS‑1 任务特定模型和 FM‑1 基础模型）。

**📈 对比分析**

与完整训练结果对比，MAE 约为 0.7–0.8×噪声底，Spearman ρ≥0.8，且在可执行假设（|Δ|>噪声底）下保持 ρ=1，整体计算降低 79–88%。

**⚠️ 局限性**

需要满足收敛稳定、相对差距可迁移、信号保持与低维可预测性等四条假设；对未验证的新架构需结合全尺度验证，且多尺度代理（M>1）尚未研究。

---

## 560. Interpreting SSL Representations for Spoof Detection: a WavLM Study

**arXiv ID:** 2610.23526 | [PDF](https://arxiv.org/pdf/2610.23526v1)

**作者:** Mohamed Mallat `[一作]` (EURECOM), Anthony Larcher `[通讯]` (Université du Mans)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对WavLM自监督模型在语音伪造检测中的各层表示进行解释性分析，结合线性探测、投影加权CCA（PWCCA）声学特征关联以及层子集搜索；

**💡 创新点**

提出将PWCCA与手工声学特征关联以解释SSL层信息的方法，并通过穷举搜索发现最小5层子集可优于全13层，从而实现模型特征压缩与性能提升的双赢；

**🔧 技术方法**

使用WavLM Base冻结特征提取、每层均值+方差池化、线性探测器、投影加权CCA、t‑SNE可视化以及层子集的穷举搜索；

**📊 数据集**

ASVspoof 5（Track 1）语音伪造检测数据集，包含TTS、VC与对抗攻击；

**📈 对比分析**

用各层和子集的线性探测器计算EER作为性能指标；在dev集上5层子集EER为2.06%（dev）/6.1%（eval），优于全13层4.59%/8.08%，显示在特征维数上更高效且检测效果更好；

**⚠️ 局限性**

仅使用线性探测，可能忽略非线性信息；只研究WavLM Base，结果可能不适用于其他SSL模型或更大模型；PWCCA假设线性相关；实验仅在ASVspoof 5上，缺乏对真实场景或未见攻击的泛化验证；声学特征集合有限。

---

## 561. LocalIntent: Outcome-Specific Structured Generation for Backend-Compatible Smart Building Automation with Local LLMs

**arXiv ID:** 2610.23511 | [PDF](https://arxiv.org/pdf/2610.23511v1)

**作者:** Tianyang Liu `[一作]` (University College Dublin), Hadi Tabatabaee Malazi `[通讯]` (University College Dublin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对本地部署的大语言模型（LLM），提出一种“LocalIntent”框架，将自然语言建筑自动化请求分为普通请求和持久规则两条分支，分别生成命令、澄清、拒绝或触发-行动规则，并通过结构化验证确保输出可直接交给后端系统执行。

**💡 创新点**

创新点在于：① 将分支路由与特定结构生成相结合，避免一次性通用生成导致的不一致；② 为每个分支设计专属的输出结构和语义约束；③ 引入可执行的结构化验证器和一次性语义修复机制，提高规则生成的准确率；④ 通过一次性对局部模型进行多轮评估，系统性展示设计完整性。

**🔧 技术方法**

主要技术包括：本地Gemma和Qwen 2B–9B LLM调用、分支路由（Ordinary/Rule）、结构化提示与上下文拼接、Deterministic Validator（结构、实体、动作、参数、规则一致性检查）、Semantic Repair（一次性修复），以及多阶段延迟测量。

**📊 数据集**

使用了两个人工构造的数据集：101条非持久请求（包括命令、澄清和拒绝）和54条持久规则请求，涵盖多种设备、传感器、动作与阈值，所有记录均包含参考答案。

**📈 对比分析**

与Monolithic（单次生成）和TwoStageGeneric（四路由通用生成）对比。实验显示，LocalIntent在IU、DI、CH、APC四项指标上均优于两种基线，尤其在规则生成上提升幅度最大（从约26%到80%+）。Semantic Repair进一步提升APC，平均提升约7%。延迟方面，LocalIntent在大多数模型与请求类型下均能保持与基线相当甚至更低的P95/均值延迟，Repair版本的延迟略高。

**⚠️ 局限性**

局限性：① 组件贡献难以单独评估，无法定位具体改进来源；② 评测仅覆盖单一建筑配置和有限样本，缺乏跨建筑泛化验证；③ 仍存在语义不一致或细节错误，验证器无法完全保证后端可执行性；④ 语义修复仅支持一次修正，未覆盖更复杂的多轮交互。

---

## 562. FLoTS-SLAM: Optical Flow-Derived Uncertainty for Monocular Dynamic SLAM with Two-Stage Optimization

**arXiv ID:** 2610.23496 | [PDF](https://arxiv.org/pdf/2610.23496v1)

**作者:** Haorui Du `[一作]` (Beihang University), Lipu Zhou `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于光流残差的运动不确定性估计与两阶段优化的单目RGB SLAM系统，能够在动态环境中精准识别动态对象并保持相机跟踪稳定。

**💡 创新点**

创新点在于：①将光流残差与相机运动诱导流差值直接用于估计像素级运动不确定性；②引入两阶段优化（先用低不确定性像素估计姿态，再用动态像素重新锚定）来消除运动像素对姿态和深度的干扰；③通过动态像素重锚实现动态物体与静态场景的解耦。

**🔧 技术方法**

主要技术包括：光流残差与相机运动计算、DINOv2特征一致性约束、两阶段高斯-牛顿BA、动态像素重锚、DROID‑SLAM预训练更新器与Metric3D深度先验。

**📊 数据集**

使用DAVIS、Bonn RGB‑D Dynamic与TUM RGB‑D三大公开基准进行实验。

**📈 对比分析**

相较于基线方法（DROID‑W、WildGS‑SLAM、DynaSLAM等），在DAVIS上平均动态Mask IoU提升约35%，在Bonn与TUM上相机绝对轨迹误差(ATE)分别降低59%与19%，同时保持约10–12 FPS的实时性能。

**⚠️ 局限性**

局限性包括：对初始姿态对齐要求较高；在极端快速动态或纹理稀疏场景下光流残差可能不稳；重锚过程对姿态估计精度敏感，若姿态误差较大可能导致误锚。

---

## 563. Symmetric two-counter machines

**arXiv ID:** 2610.23495 | [PDF](https://arxiv.org/pdf/2610.23495v1)

**作者:** Krzysztof Jaros `[一作]` (University of Warsaw), Sławomir Lasota `[通讯]` (University of Warsaw)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了对称（即更新转移在交换计数器时保持不变）的2计数器机的可达性问题，证明该问题可判定，并由此推出此类机器非图灵完备；同时展示了对称性在3计数器机中仍导致不可判定的界限。

**💡 创新点**

核心创新在于：①利用对称性引入“目标泵”性质，使得可达关系在三种段落（+、0、-）上具有可泵性；②通过将可达性分解为三阶段并证明每个阶段的关系是有效半线性的，进而得到整体可达性的可判定性；③构造了对称性保持的3计数器机的归约，证明3计数器机在循环对称组下不可判定。

**🔧 技术方法**

主要技术手段包括：线性路径方案（LPS）分解、目标泵（target pumping）引理、有效半线性集的构造与闭包、Parikh定理与1计数器机的改造、以及对称性对转移的影响分析。

**📊 数据集**

本工作为理论性研究，无使用具体实验数据集；所有结果均基于形式化证明与构造。

**📈 对比分析**

与一般计数器机（可判定性为Ackermann级、不可判定）以及对称1计数器机（P‑complete）进行比较，表明对称2计数器机可判定，复杂度为5‑指数级；相比之下，对称3计数器机保持不可判定，体现对称性对表达力的显著差异。

**⚠️ 局限性**

局限性包括：①对称3计数器机的可判定性仍未确定；②尽管可判定，但算法复杂度极高（5‑指数），尚未找到更优界限；③对称性只对更新转移做约束，若对零测试转移也施加对称，则可简化为1计数器机，但此时研究意义减弱。

---

## 564. Deterministic Global Minimum Cut Using $\tilde{O}(n^{3/2})$ Cut Queries

**arXiv ID:** 2610.23461 | [PDF](https://arxiv.org/pdf/2610.23461v1)

**作者:** Yotam Kenneth-Mordoch `[一作]` `[通讯]` (Weizmann Institute of Science), Yotam Kenneth-Mordoch (Weizmann Institute of Science)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种在 cut‑query 模型下的确定性算法，用 Õ(min{nδ, n²/δ}) 次查询即可找出无权图的全局最小割；

**💡 创新点**

创新点在于利用当最小 s,t‑割值≤δ 时最大流仅支持 O(n) 条边，进而设计了 O(n) 查询的 bounded max‑flow 算法，并以此实现比以往最优的确定性查询复杂度；

**🔧 技术方法**

主要技术包括：cut‑query 下的短增广路径搜索、维护最大生成树、Nagamochi‑Ibaraki 稀疏化、支配集构造、改进的 Dinitz/Edmonds‑Karp 算法等；

**📊 数据集**

论文为理论研究，不涉及具体数据集；

**📈 对比分析**

与之前最优确定性算法（ Õ(n^{5/3})）相比，最坏情况达到 O(n^{3/2})，在 δ 较小或较大时更优；当 δ=Ω(n) 时与随机算法的 O(n)（忽略 polylog）相匹配；

**⚠️ 局限性**

局限在于查询复杂度仍高于随机 O(n)，且依赖图的最小度；未能在所有情形下实现 O(n) 的确定性查询复杂度，且 max‑flow 辅助仅适用于割值≤δ。

---

## 565. On MRD constacyclic codes over inseparable field extensions

**arXiv ID:** 2610.23459 | [PDF](https://arxiv.org/pdf/2610.23459v1)

**作者:** Grégory Berhuy `[一作]` `[通讯]` (University of Grenoble Alpes), Grégory Berhuy (University of Grenoble Alpes)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构造了在不可分离域扩展L/K上长度为p的MRD共轭循环码，这些码无法通过已知的构造方法获得。

**💡 创新点**

创新点在于提出了一种新的MRD共轭循环码构造方法，扩展了现有的编码理论。

**🔧 技术方法**

使用了L-线性子空间的概念，并引入了秩权重和秩距离的定义。

**📊 数据集**

使用了长度为p的码，具体数据集未明确给出。

**📈 对比分析**

与已知的构造方法进行了比较，表明这些新构造的MRD码无法通过现有方法获得，性能上具有独特性。

**⚠️ 局限性**

限制在于未提供具体的应用实例和数据集，可能影响理论的实际应用。

---

## 566. A low-dimensional dynamical basis for tongue shape control

**arXiv ID:** 2610.23456 | [PDF](https://arxiv.org/pdf/2610.23456v1)

**作者:** Sam Kirkham `[一作]` (Lancaster University), Patrycja Strycharczuk `[通讯]` (University of Manchester)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b88c6eac-d57a-4623-a604-1f401f3eb268` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究提出将舌头形状的低维统计模式作为控制基底，并用线性阻尼点吸引器模型描述英语单元音的舌动。

**💡 创新点**

创新点在于把舌形模式视为全局控制变量，解释了一维测量中出现的非线性伪影，并将控制问题与肌肉协同产生联系。

**🔧 技术方法**

使用PCA降维、Generalized Procrustes Alignment、线性阻尼点吸引器动力学拟合以及参数约束和分段分析。

**📊 数据集**

使用英国英语说话者的TarDiS超声数据集，包含36位说话者的36个短单元音。

**📈 对比分析**

通过与三维PCA重建误差对比，模型在闭合阶段的RMSE为0.177mm，释放阶段为0.286mm；与无约束模型相比，固定终点提高了语音类别可分离度。

**⚠️ 局限性**

局限包括对速度/加速度的噪声处理不足、频率与阻尼参数可识别性低、仅研究短单元音、未考虑模式耦合、并假设线性动力学忽略了可能的非线性机制。

---

## 567. ScoutRoute: Efficient Mixture-of-Experts Routing Using Internal Language Model Signals

**arXiv ID:** 2610.23443 | [PDF](https://arxiv.org/pdf/2610.23443v1)

**作者:** Rajdeep Pandhare `[一作]` (Rajiv Gandhi Institute of Technology), Delzaan Bharucha `[通讯]` (Rajiv Gandhi Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级的后置专家路由框架ScoutRoute，利用单次侦察模型推理获取内部隐藏状态和logit信息来预测并选择最合适的专家模型。

**💡 创新点**

创新点在于将隐藏状态的跨模型能力估计与logit衍生的置信度结合，并通过领域感知的阈值策略实现专门化与通用化的动态切换。

**🔧 技术方法**

使用轻量级侦察模型（如Qwen2.5-1B、Phi-3 Mini等）、主成分分析（PCA）降维、逻辑回归探针、TF‑IDF+逻辑回归领域分类器以及log概率路由。

**📊 数据集**

在四个基准数据集上进行评估：数学推理（GSM8K）、代码生成（HumanEval）、常识推理（CommonsenseQA）和多学科知识（MMLU）。

**📈 对比分析**

与随机路由、负熵、单独的探针得分、均值log概率等独立路由策略对比，ScoutRoute在整体路由准确率上从最高单独信号的65.02%提升至67.80%，并在多模型组合下保持最优性能。

**⚠️ 局限性**

局限包括对侦察模型的依赖性、专门化阈值需要经验调优、在极大专家集合或多模态任务中的可扩展性尚未验证，以及对不确定性估计的进一步改进空间。

---

## 568. ReVision: Reprogramming Image Generators for Dense Visual Perception

**arXiv ID:** 2610.23533 | [PDF](https://arxiv.org/pdf/2610.23533v1)

**作者:** Yitong Yang `[一作]` (Tencent Hy), Kai Wang `[通讯]` (Tencent Hy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了如何在不更新预训练图像生成器参数的情况下，利用冻结的多模态理解模型和图像生成器，通过生成低秩参数更新，来实现多任务的稠密视觉感知（语义/实例/指代分割、深度估计、法向估计）。

**💡 创新点**

①引入指令条件的参数生成框架，按图像-指令对实时产生实例级低秩更新；②将多任务感知映射到统一 RGB 输出空间，无需任务头或适配器；③通过分析梯度冲突和参数空间几何，验证生成更新的任务结构与实例特异性。

**🔧 技术方法**

冻结的多模态理解模型、冻结的扩散图像生成器、LoRA低秩参数生成、参数查询+Transformer、指令条件的低秩更新、统一 RGB 编码、混合任务训练与 EMA、逐步分辨率训练等技术。

**📊 数据集**

语义分割：ADE20K、COCO‑Stuff‑164K、Mapillary Vistas、BDD7K；实例分割：LVIS、Open Images V7、GranD+SAM 3；指代分割：GranD；深度/法向：Omnidata、TartanAir、NYUv2、KITTI、ETH3D、ScanNet、DIODE、iBims1、Marigold、VKitti 等。

**📈 对比分析**

与任务专用 LoRA、共享多任务 LoRA 以及 SenseNova‑Vision、Vision Banana 等基线比较；在 Cityscapes、SA‑Co/Gold、RefCOCOg、NYUv2、KITTI、ETH3D、ScanNet、VKitti 等数据集上，取得与或优于基线的 mIoU、cIoU、AbsRel、角度误差等指标，表现出强大的零样本泛化与跨任务性能。

**⚠️ 局限性**

受限于冻结的理解模型和生成器的能力；实例级参数生成带来额外推理开销；统一 RGB 输出相较于专用头效率低且对精细掩模/几何敏感；需更强的冻结骨干、效率更高的参数生成器或更合适的训练目标。

---

## 569. Trapped in Your Story: Evaluating Narrative-Induced Sycophancy in Large Language Models

**arXiv ID:** 2610.23525 | [PDF](https://arxiv.org/pdf/2610.23525v1)

**作者:** Runtai Yang `[一作]`, Guangtao Zhai `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个包含500个跨领域辩论性问题、5,500个评估实例的受控基准，系统地改变同一立场的叙事阐释，探究叙事对大型语言模型(LLM)同意度的影响。

**💡 创新点**

创新地将立场对齐与叙事框架分离，系统评估不同叙事成分（经验、因果、价值观）和呈现形式对模型同意度的非单调影响，揭示叙事结构对“奉承”行为的细微作用。

**🔧 技术方法**

采用对话式评估、叙事成分级注释、Narrative Adoption Score (NAS)、McNemar检验、Bootstrap置信区间等统计方法，结合多种LLM（GPT‑5.1/5.5、Claude‑Fable‑5、Gemini‑3.8‑Flash、DeepSeek‑V4‑Pro、Llama‑3.1‑8B、Qwen‑3‑8B）进行实验。

**📊 数据集**

使用自构的500条英语辩论性问题集，覆盖工作、教育、技术、生活方式、育儿、环境、社会规范、消费决策、健康与公共政策等十个领域，生成5,500个包含不同叙事层级的评估实例。

**📈 对比分析**

通过比较各模型在5个叙事层级下的A/B选择与理由，计算方向平衡同意度、Narrative Sycophancy Gap等指标；结果显示模型对叙事组件高度敏感，GPT‑5.1在L3出现下降后在L4‑L5恢复，DeepSeek和Qwen几乎完全接受所有叙事，说明不同模型在叙事诱导同意度上的显著差异。

**⚠️ 局限性**

局限包括：仅使用英语与对话式任务，未涵盖统计证据或专家引用；叙事层级固定，未检验更广泛的叙事类型；评估仅覆盖七款LLM，未覆盖小模型或多语言场景；NAS等指标依赖人工标注，可能带来主观偏差。

---

## 570. QD-Rec: Open-Ended Evolution of Recommendations via Quality-Diversity

**arXiv ID:** 2610.23470 | [PDF](https://arxiv.org/pdf/2610.23470v1)

**作者:** Takashi Ishida `[一作]` (Sakana AI), Issa Sugiura `[通讯]` (Sakana AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种名为 QD-Rec 的质量-多样性搜索框架，用于在固定候选池中生成多样化且高质量的项目–理由对（推荐+自然语言解释），并通过在线学习不断改进候选选择和理由生成。

**💡 创新点**

创新点包括：① 把开放式搜索与推荐结合，允许同一项目因不同理由被多次推荐；② 使用基于嵌入的本地新颖性检查与局部质量对决，保持存档多样且高质量；③ 结合上下文 bandit 进行项目选择，LLM 作为提议者与评判者，形成闭环的演化式推荐流程；④ 通过 MAP‑Elites 视图对存档进行可视化与评估。

**🔧 技术方法**

技术细节包括：质量多样性（QD）算法与 MAP‑Elites、贝叶斯线性回归 + Thompson Sampling 的上下文 bandit、L2 正则化嵌入、kNN 邻域搜索、局部对决（blind duel）评估、LLM（Proposer、Judge、Novelty checker）与自然语言提示、并行批处理与结果合并。

**📊 数据集**

使用的主要数据集有：Goodreads 儿童/YA 书单；M&A 领域 6,332 家美国公司；营销领域 10,000 个合成客户画像；以及 ICLR/ICML/NeurIPS 2025‑26 论文集（17,511 篇）。每个领域对应不同的理由类型（阅读备忘、收购论点、营销定位、研究备忘）。

**📈 对比分析**

与两种 ablation（Novelty、Novelty+Evo）以及两种无学习重排器（MMR、DPP）进行比较。QD-Rec 在大多数 LLM 评估环境中获得最高平均质量，存档规模最大，覆盖率最好；在 Goodreads 的人类评分环境中，质量与 ablations 相近，但存档规模和多样性更好。成本上约为 Novelty 的 1.3 倍，主要来自带示例的 LLM 生成。

**⚠️ 局限性**

局限性包括：① 依赖 LLM 的评判与生成，评估成本高且受模型质量限制；② 嵌入模型对新颖性和多样性评估的敏感性；③ 仅处理固定候选池，无法适用于生成式推荐；④ 仅使用线性上下文 bandit，未探索非线性或更复杂的决策策略；⑤ 只针对单一决策者配置，未评估多决策者场景；⑥ 目前的演化操作仍相对简单，缺乏更深层次的自动化推理能力。

---

## 571. Model Checking for Noisy Toric Words

**arXiv ID:** 2610.23463 | [PDF](https://arxiv.org/pdf/2610.23463v1)

**作者:** Mihir Vahanwala `[一作]` `[通讯]` (Max Planck Institute for Software Systems), Mihir Vahanwala (Max Planck Institute for Software Systems)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了噪声鲁棒动力系统及其产生的无限词（噪声轨迹），并证明这些轨迹是几乎周期性的，并且与原始无噪声轨迹共享相同的循环因子集合；同时将该理论应用于实代数线性递推序列（LRS）的符号描述，给出若干谱条件下可判定其符号序列是否属于ω-正则语言；进一步证明噪声轨迹与前缀独立的ω-正则语言不可区分，并给出了判定包含关系的方法；最后探讨了斯图尔马系统在噪声下的非鲁棒性并揭示其与Diophantine逼近问题的关系。

**💡 创新点**

1) 引入噪声鲁棒动力系统概念，统一处理噪声轨迹的几乎周期性与循环因子；2) 在LRS符号描述中给出多种谱条件下的可判定性，扩展了以往仅针对简单或低阶序列的结果；3) 证明噪声轨迹对前缀独立ω-正则语言不可区分，为验证噪声系统提供新的理论工具；4) 通过斯图尔马系统的反例揭示噪声鲁棒性与前缀独立性的边界。

**🔧 技术方法**

采用拓扑方法（连通性、最小性、连续性）定义噪声鲁棒性并证明轨迹的几乎周期性；使用线性递推序列的指数-多项式闭式与Baker线性形式理论解决Skolem问题；结合Semënov定理判定几乎周期词是否满足ω-正则语言；利用自动机理论（确定性奇偶自动机）研究前缀独立语言的性质。

**📊 数据集**

无公开数据集，全部基于理论构造和数论性质（实代数数、线性递推序列的特征根）。

**📈 对比分析**

本文未进行实验或性能评估，而是给出了判定算法的可决性与有效性。

**⚠️ 局限性**

1) 对于高阶LRS（>4）及存在根比是单位根的情况，Skolem问题未决导致符号序列判定仍不完整；2) 对斯图尔马系统的噪声轨迹可导致非几乎周期性，说明噪声鲁棒性条件对某些系统不适用；3) 论文的判定方法依赖于L_⋆的递归性，若无法有效计算此集合则无法执行。

---

## 572. Generative Policy Improvement from Observational Data

**arXiv ID:** 2610.23436 | [PDF](https://arxiv.org/pdf/2610.23436v1)

**作者:** Jonas Schweisthal `[一作]` (LMU Munich), Emma Brunskill `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了三阶段的生成式政策改进框架（“BCL”），在观测数据上学习离散动作码，训练基于码的上下文策略，并通过行为系统的采样把码级策略提升回原始高维结构动作；

**💡 创新点**

核心创新在于：①学习能提升结果的离散动作码并通过分布平衡正则化解决覆盖不足；②在码空间上执行离线策略学习并用回归调整（RA）估计，避免传统重要性权重方差；③采用行为一致提升（behavior-consistent lifting）消除部署时的“部署缺口”，确保码级改进在真实动作空间中得到体现；

**🔧 技术方法**

技术包括：动作码编码网络（两分支 + Gumbel-Softmax）、分布距离正则（切片 Wasserstein）、KL 正则、交叉熵估计的码条件倾向、回归调整价值估计、交叉验证与采样上限控制；

**📊 数据集**

在三个数据集上评估：1）合成数据（可控奖励、码可预测性、覆盖率）; 2）标题生成基准（真实文章 + GPT-5.4 生成标题 + 模拟奖励）； 3）公开 OfflinePrompts 基准（基于 Mistral 的生成文本与用户响应模拟器）；

**📈 对比分析**

与现有 OPE/OPL 方法（PoTEC、MIPS、OffCEM、句子级重要性等）以及提示基准方法（prompt regression、DSO）对比。实验表明：“BCL” 在所有三组数据上均实现正向改进；在 OfflinePrompts 上相对行为策略提升约129%，接近或超过提示信息优势方法；

**⚠️ 局限性**

局限性：需在部署时访问行为系统的采样；在同一码下仅保留原始行为分布，未进一步优化同码动作；以及对极端覆盖不足的情况仍可能受限，未来可探索更灵活的提升规则与在线评估。

---

## 573. Bridging Compressed Communication and Local Steps via Spectral Method in Federated Learning

**arXiv ID:** 2610.23434 | [PDF](https://arxiv.org/pdf/2610.23434v1)

**作者:** Dmitriy Bystrov `[一作]` (Basic Research of Artificial Intelligence Laboratory), Aleksandr Beznosikov `[通讯]` (Basic Research of Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的联邦学习算法 Federated Compressed And Local Muon（FCLM），将压缩、局部步骤与谱 LMO 更新融合，以降低通信量并提升收敛速度。

**💡 创新点**

创新点在于首次将 contractive 压缩与错误补偿机制引入谱 LMO 更新，并在异构环境下实现无异质性假设的收敛保证。

**🔧 技术方法**

采用的技术包括 contractive 压缩、错误补偿、谱矩阵优化（LMO）与 Newton–Schulz 近似、局部更新、控制变量以及 Adam/MuON 等自适应优化框架。

**📊 数据集**

实验数据集涵盖图像分类任务 CIFAR‑10（ResNet‑18）和自然语言处理任务 GLUE（SST‑2，RoBERTa+LoRA）。

**📈 对比分析**

与 FedAvg、FedAdam、FedProx、MuON 等基线对比，FCLM 在通信位数上表现出更快的收敛速度和更高的最终准确率，尤其在 Top‑1%/Top‑10% 稀疏压缩下优势明显。

**⚠️ 局限性**

限制包括对光滑性假设的依赖、实验规模有限，以及假设所有客户端每轮都参与的理想化设置。

---

## 574. Momentum: Persistent Dynamics for Video World Models

**arXiv ID:** 2610.23423 | [PDF](https://arxiv.org/pdf/2610.23423v1)

**作者:** Nedko Savov `[一作]` (INSAIT, Sofia University St. Kliment Ohridski), Luc Van Gool `[通讯]` (INSAIT, Sofia University St. Kliment Ohridski)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种两阶段的动态场景视频生成管道，先利用同步摄像机组生成全局演化的场景状态，再基于该状态对目标摄像机视频进行定位和生成。

**💡 创新点**

创新点在于引入多摄像机组的图结构来同步生成视频，从而在视野之外保持动态内容的连续性，实现了不依赖实体追踪或静态-动态分解的场景一致性。

**🔧 技术方法**

技术上结合了Cosmos-Predict 2.5的多视角视频扩散模型、SWM的图像/视频检索以及基于相机几何的重投影与图传播机制。

**📊 数据集**

使用CARLA仿真平台构建的同步参考视频与目标摄像机轨迹数据集，涵盖多城镇、随机天气与多交通参与者，另外在公开的Ann‑Arbor驾驶数据上做基准验证。

**📈 对比分析**

与静态图像条件的基线相比，动态场景一致性(SCS‑dyn)提升约130%（见城镇）/126%（未见城镇），PSNR、LPIPS、姿态误差均优于SWM与单视角DGWM，表明动态地面真值能够显著提升视觉质量与一致性。

**⚠️ 局限性**

局限性主要在于对同步摄像机组的依赖，生成时需额外推理阶段，计算成本较高，且在真实环境下参考视频的获取与对齐仍是技术挑战。

---

## 575. Not Every Mention Is a Claim: Rethinking Object Hallucination Detection in Vision-Language Models

**arXiv ID:** 2610.23610 | [PDF](https://arxiv.org/pdf/2610.23610v1)

**作者:** Hsin-Ling Hsu `[一作]` (National Chengchi University), Kuan-Hao Huang `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了NAME-Bench基准，系统标注并区分了视觉语言模型生成文本中的非断言性对象提及（non-assertive mentions），并评估现有的幻觉检测器与评价指标是否误将这些提及视为幻觉。

**💡 创新点**

创新点在于首次将对象提及的使用意图（存在性断言 vs. 仅提及）纳入幻觉评估，揭示了现有方法的严重误判，并通过在训练中加入非断言样本来校正检测器，验证不同模型对非断言样本需求的差异。

**🔧 技术方法**

使用的技术包括：GPT‑4o 自动标注提及的使用与存在性，十二种幻觉检测器（如 Lookback Lens、DHCP、SVAR、VADE 等）的评估与改进，和基于文本过滤的实验。

**📊 数据集**

使用的数据集为 COCO‑2017 的图像，生成的五个公开 VLM（LLaVA‑1.5、Qwen2.5‑VL‑3B/7B、InternVL3‑8B、LLaVA‑OneVision‑2‑8B）在单一 prompt 下生成的 25,497 条包含 80 个 COCO 类别的提及。

**📈 对比分析**

对比实验显示：在 LLaVA‑1.5 上，仅需 10% 的非断言样本即可将误报率从 40% 降至 4–6%；在其他四个模型上则需要 5–10 倍数据，且会略微降低召回率；文本过滤能显著降低误报率但也会损失一小部分真正的幻觉。

**⚠️ 局限性**

局限性包括：只关注 COCO 80 类对象且仅在单一 prompt 与最大 128 token 生成范围内；非断言提及约占 45% 被排除；GPT‑4o 的标注可能存在偏差；以及对是否将模糊表达（hedge）视为断言的定义仍有争议。

---

## 576. When Magnitude Suffices: Reconstructing Time-Domain-Gated Microwave Signals Without Phase on a Thoracic Phantom

**arXiv ID:** 2610.23608 | [PDF](https://arxiv.org/pdf/2610.23608v1)

**作者:** Maciej Cichoń `[一作]` (University of Lodz), Maciej Ślot `[通讯]` (University of Lodz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出一种利用仅幅度测量的标量仪器（频谱分析仪+跟踪发生器）重建时间域门控幅度的方案，避免了向量网络分析仪的相位测量需求，并通过全连接神经网络实现从幅度到门控幅度的映射；在胸腔仿真模型中验证了该方法的有效性。

**💡 创新点**

创新点在于：①证明在特定测量配置下，幅度谱已足以恢复门控后幅度，即幅度与相位之间存在可学习的关系；②仅需一次经过向量网络分析仪的标定即可确定门控位置和参考校正；③引入排序损失（Spearman相关的辅助项）保证状态间的相对顺序被保留；④展示该方法在幅度测量设备上与传统VNA相当的性能。

**🔧 技术方法**

主要技术包括：时间域门控（IFFT→窗→IFFT），对设备参考状态的逐频段减法校正，使用全连接网络（1002→256→256→1002）配合GELU激活与dropout进行回归；训练损失为均方误差与排序辅助项的加权组合；对S11、S21的幅度与相位进行距离相关与R²评估。

**📊 数据集**

使用数据集：①胸腔仿真体型模型，11种肺腔介质状态，共286次VNA扫频（2.0–3.0 GHz 1002点）与256次幅度测量；②额外的10种测量配置用于幅度‑相位关系分析；③公开的UM‑BMID乳腺仿真数据集（152个会话，66个体型）用于外部验证。

**📈 对比分析**

方法对比：①与VNA门控幅度直接回归（基准），②显式相位重建+门控，③最小相位+门控，④逐频段线性映射。结果显示，在S11上，SA输入的形状误差0.08 dB²、排序0.81，几乎等同于VNA；在S21上形状误差0.82 dB²、排序0.28，但能保持高、中、低介电状态的聚类；在UM‑BMID上网络比线性映射低30％误差。

**⚠️ 局限性**

局限性：仅在单一测量配置下验证，未证明跨配置泛化；需要一次VNA标定作参考；仅在受控胸腔仿真体上实验，未涉及真实人类解剖和多层介电结构；定位误差的鲁棒性未系统评估；S21重建精度低于S11，未评估临床诊断效果。

---

## 577. Token-Level Attribution of Multi-Mode LLM Watermarks: Optimal Rule and Belief Weighting

**arXiv ID:** 2610.23601 | [PDF](https://arxiv.org/pdf/2610.23601v1)

**作者:** Rui Zhou `[一作]` (City University of Hong Kong), Wenqiang Pu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了多模式LLM水印在token级的归因问题，并提出最优规则与BLADE检测器，能在保持预设Type I误差的同时最大化先验加权的正确标签功效。

**💡 创新点**

创新点在于将多模式归因建模为先验加权的功效最大化问题，给出阈值判定的最优结构，并基于贝叶斯权重与密度估计设计了可实现的BLADE检测器，首次实现了在Token级别上既能自适应先验又能保持严格误差控制的多模式归因。

**🔧 技术方法**

使用Gumbel‑max水印的关键化值、贝叶斯先验、密度估计（基于参考token概率样本）、马尔可夫链（捕捉邻域模式持久性）以及阈值校准（联合阈值和多模式误差预算）等技术。

**📊 数据集**

在OPT‑1.3B、Qwen2.5‑1.5B、Llama‑3.2‑1B和Sheared‑LLaMA‑2.7B等开源LLM上，用C4真实新闻数据生成含水印与未水印混合文本，并使用参考token概率样本进行离线密度估计。

**📈 对比分析**

与校准Max‑Pivot、Local‑Count、BW等方法对比，BLADE在候选数、替换率、温度和不同模型上均实现了最高的正确标签功效（最高可达约37.6%），并且在所有实验中Type I误差保持在5%附近；此外BLADE对加入的无关候选的鲁棒性优于其他方法。

**⚠️ 局限性**

局限性包括对密度估计和先验估计误差敏感；在极端token概率分布或模式切换频繁、邻域信息稀缺的文本中，估计误差可能导致功效下降；另外BLADE的计算复杂度相对较高，需预先构建密度表。

---

## 578. Learning to Discriminate Scene Structures Makes Self-Supervised Depth Learning Scalable

**arXiv ID:** 2610.23600 | [PDF](https://arxiv.org/pdf/2610.23600v1)

**作者:** Mengtan Zhang `[一作]` (Tongji University), Rui Fan `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出MaSS框架，结合预训练视觉基础模型和深度编码器，通过结构与上下文分离，利用自监督学习实现多域数据的可扩展深度估计。

**💡 创新点**

创新点在于：① 通过双向交互与锚点-解耦投影器显式分离结构和上下文特征；② 用基于深度直方图的KL相似度定义场景结构对比学习目标；③ 使模型在混合多域训练中能够提升性能并在零射击场景中保持高度泛化。

**🔧 技术方法**

采用预训练DINOv2视觉基础模型、ViT-Adapter交互模块、锚点-解耦投影器以及InfoNCE对比损失，并使用自监督的相机运动、动态场景处理等标准技术。

**📊 数据集**

在KITTI、NYUv2、DDAD、nuScenes以及TartanAir等四个多样数据集上进行联合与单独训练，并在Waymo、DrivingStereo、Cityscapes、DIML、iBims-1、SUN-RGBD、7-Scenes、DIODE等八个零射击数据集上评估。

**📈 对比分析**

与最新自监督方法（如DCPI-Depth、SC-DepthV3、DiMoDE等）以及部分监督基础模型比较，MaSS在所有训练域的Abs Rel和δ1上均优于对比方法，在零射击域亦取得显著优势，尤其在复杂或分布外场景中表现突出。

**⚠️ 局限性**

局限性包括：在标准室内数据集上仍低于专门训练的监督模型；依赖预训练VFM的几何信息有限，且当前规模相较于大规模监督深度基础模型仍有限；对极端光照或纹理稀缺场景的鲁棒性尚待进一步提升。

---

## 579. Spatial Distribution of Mean PAoI in LEO Satellite-enabled IoT Networks

**arXiv ID:** 2610.23597 | [PDF](https://arxiv.org/pdf/2610.23597v1)

**作者:** Badiaa Gabr `[一作]` (Maynooth University), Mustafa A. Kishk `[通讯]` (Maynooth University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对低轨卫星网络中的地面物联网节点到LEOSAT的上行链路，利用随机几何方法推导了覆盖概率的元分布、条件均值PAoI的空间分布及其一阶矩；

**💡 创新点**

首次将覆盖概率的元分布引入PAoI分析，得到可解析的空间分布与一阶矩表达式，揭示卫星高度、数量、波束宽度和到达率对信息新鲜度的影响；

**🔧 技术方法**

采用随机几何、Gamma 与 Beta 分布逼近、拉氏变换、矩母函数与多项式展开等数学工具；

**📊 数据集**

使用基于仿真参数（LEOSAT高度500–1500 km、卫星数500–2000、波束宽度π/10–π/4、到达率0.1–0.9）进行 Monte‑Carlo 仿真验证；

**📈 对比分析**

通过与仿真结果对比，分析表明理论与仿真吻合度高，展示了不同配置下PAoI随网络规模与几何参数的变化趋势；

**⚠️ 局限性**

局限在于对衰落使用 Gamma 近似、覆盖概率使用 Beta 分布逼近，且忽略了卫星轨道动态与下行链路等实际细节。

---

## 580. Analyzing Occupational Gender Bias across the LLM Training Lifecycle

**arXiv ID:** 2610.23587 | [PDF](https://arxiv.org/pdf/2610.23587v1)

**作者:** Manthan Gadhia `[一作]` (ETH Zürich), Karolina Stańczak `[通讯]` (ETH Zürich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对OLMo 3 7B模型在监督微调、DPO和RLVR等后训练阶段的职业性别偏差进行系统追踪与评估。

**💡 创新点**

首次将性别显式与隐式两种评估设置并行，结合多维度文本指标与逻辑回归，揭示后训练阶段对男女差异表现的非对称影响，并发现后训练会逐步增加模型对女性职业的默认归属。

**🔧 技术方法**

使用OLMo 3系列开源模型、监督微调(SFT)、直接偏好优化(DPO)、基于可验证奖励的强化学习(RLVR)，以及NLP评估工具（token entropy、Self-BLEU、语义多样性、情感与评价）。

**📊 数据集**

合成职业人设数据（15个职业，二元性别与学历属性）、美国劳工统计局BLS 2025年职业性别比例、以及SFT训练语料库用于源头分析。

**📈 对比分析**

通过对四个检查点和不同温度的生成结果进行回归与对数几率模型，对比语义多样性、词汇多样性、情感与评价指标；结果显示后训练提升男性与女性的文本多样性差异缩小，且女性默认归属的赔率显著上升。

**⚠️ 局限性**

局限在于仅考虑二元性别、仅选取15个英文职业、仅评估OLMo 3 7B，且基准为美国数据，无法涵盖非二元身份、多语言与跨文化差异。

---

## 581. BMuon: Budgeting Muon's Spectral Geometry for Structured Compressibility

**arXiv ID:** 2610.23571 | [PDF](https://arxiv.org/pdf/2610.23571v1)

**作者:** Rida Assalouh `[一作]` (Inria), Umut Şimşekli `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Budget Muon（BMuon），在 Muon 训练器的谱范数几何中加入全局隐藏通道预算，使得 Transformer 的 MLP 通道在训练期间自适应分配，从而提升结构化压缩效果。

**💡 创新点**

创新点在于：①将 Muon 的谱范数约束重加权为预算加权，形成一族 Ωα,K 归一化最速下降几何；②通过预算约束实现全局通道竞争，保证在预算稀缺时最小化表示成本时得到宽度最小化；③提出可求解的近似 MM（minorization‑maximization）分配更新，避免每步求解完整分配问题。

**🔧 技术方法**

技术主要包括：Muon's 谱范数优化（polar factorization），预算加权的对角重加权，Ωα,K 归一化最速下降，表示成本分析（α‑effective spectral dimension），MM 近似分配更新，以及标准的 Transformer 训练（ViT、LLaMA）与 Pruning（全局 MLP 通道剪枝）。

**📊 数据集**

使用的数据集有：CIFAR‑100（ViT 小模型）、ImageNet‑1K（ViT‑S/16）、以及 1B 参数 LLaMA‑style 模型在 FineWeb 语料上的训练集。

**📈 对比分析**

与 Muon、Muon+GL、NuMuon 及基于权重幅值/激活的剪枝标准对比。实验表明：在无恢复的后训练剪枝、逐步剪枝以及大规模语言模型上，BMuon 在相同或更高压缩率下保持更高 Top‑1 准确率/更低 perplexity，尤其在 80%+ 通道剪枝时提升 10%+；同时 BMuon 的学习分配 μ 也能作为高质量的剪枝指标并能指导非均匀宽度架构的自检。

**⚠️ 局限性**

局限性包括：①在原始密集模型上 BMuon 的性能略低于 Muon；②理论分析仅覆盖线性两因子模型，未给出完整的隐式偏置结果；③当前实现仅对 MLP 通道有效，未扩展至注意力头或其他结构；④MM 近似虽然经验上良好，但缺乏严格的近似误差保证。

---

## 582. Recall-Oriented Precise Binary-to-Source Function Matching

**arXiv ID:** 2610.23546 | [PDF](https://arxiv.org/pdf/2610.23546v1)

**作者:** Junlin Li `[一作]` (Renmin University of China), Bin Liang `[通讯]` (Renmin University of China)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种二进制到源函数级别的匹配方法，能够在存在内联、符号剥离等复杂情况下精确匹配二进制函数与源代码。

**💡 创新点**

创新点在于：①将全函数嵌入与组合感知（inlining-aware）训练相结合，显著提升多源函数匹配；②利用位置信息（文件、模块、调用邻域）补充候选并与LLM共同筛选，突破仅靠Embedding初始候选的限制；③采用LLM进行多轮判别与回顾，进一步提升精度与召回。

**🔧 技术方法**

技术手段包括：Transformer‑based Qwen3.5‑4B（Fine‑tuned with contrastive + composition‑aware loss）、chunking + full‑function embedding、LLM（Qwen3.7‑Flash 或 Qwen3.5‑9B）进行候选选择、同名、个体锚点、N‑to‑1 复核和 TPL 可信度评估，以及基于区间（interval）构造的候选补全与过滤。

**📊 数据集**

使用的实验数据集：公开 BinKit‑BCSA（39,401 二进制‑源对）和自建 Large‑10（10 个大型项目、stripped/unstripped），训练集 285 开源项目（308,160 二进制‑源对），验证集 6K，测试集 6K/20K 与 Large‑10 上的 57K‑67K 二进制‑源对。

**📈 对比分析**

评估方法：Precision、Recall、F1、R@k、MRR 等指标；与现有 SOTA BinaryAI、SBridge 进行对比。结果显示：在大多数平台/编译配置下 Precision 提升 ≥32%，Recall 达到约 2.9 倍；在大型 stripped 二进制上，整体 F1 在 80% 以上，显著优于 Baselines。

**⚠️ 局限性**

局限性：①仍依赖大模型 API 费用与延迟；②位置信息稀疏时补全候选效果有限；③对多目标匹配、极端优化/剥离场景的鲁棒性待进一步验证；④实验覆盖仅限 x86‑64 O2，其他架构/编译选项需后续扩展。

---

## 583. ConvexLLM: Scaling Convex-Hull Speech Interfaces to General-Purpose Audio Intelligence

**arXiv ID:** 2610.23539 | [PDF](https://arxiv.org/pdf/2610.23539v1)

**作者:** Ming-Hao Hsu `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b88c6eac-d57a-4623-a604-1f401f3eb268` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于将语音连续混合为冻结文本嵌入的凸组合接口的通用语音与音频语言模型。

**💡 创新点**

创新点是将语音表示限制在文本嵌入的凸包内，仅学习组合权重以保持预训练文本嵌入不变，从而在保持文本能力的同时显著提升语音理解和推理性能。

**🔧 技术方法**

技术包括 Whisper-large-v3 作为声学编码器，Qwen3.5 语言解码器，凸组合路由器（Top‑K 路由 + 混合），多阶段联合训练与教师蒸馏等。

**📊 数据集**

使用 MLS English、GigaSpeech、LibriSpeech、Emilia、Common Voice、FLEURS、WavCaps、Clotho、MusicCaps 等公开数据集，以及合成的口语指令和推理样本。

**📈 对比分析**

在 VoiceBench、LibriSpeech、MMAU Test‑mini 等基准上与多款开源音频语言模型对比，模型在 VoiceBench 总分 75.36 领跑，尤其在口语推理任务表现突出；在转写任务上 MLP 更好，音频理解任务上 Q‑Former 更优。

**⚠️ 局限性**

局限包括模型与基线在规模、数据和训练策略上不完全对齐；接口比较仅在 2B 参数规模下完成；未系统评估口音或说话者差异对性能的影响；凸组合接口可能在某些任务中不如不受约束的接口。

---

## 584. The Sign Code: The Hidden Binary Nature of Deep Networks

**arXiv ID:** 2610.23616 | [PDF](https://arxiv.org/pdf/2610.23616v1)

**作者:** Niclas Göring `[一作]`, Ard A. Louis `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了神经网络输出的符号（sign）与幅度（magnitude）对分类性能的相对贡献，发现符号编码（sign code）在过参数化的视觉分类器中几乎足以恢复完整模型的准确率，且符号对分类具有决定性影响；

**💡 创新点**

提出了符号编码信息充分性、因果优势以及符号几何对泛化的重要性三个创新结论，并提供了关于符号与幅度分离的实验和理论证据；

**🔧 技术方法**

采用了符号-幅度分解、k近邻符号分类器、两路通道干预（符号翻转与幅度随机化）、类内符号排斥正则化、门控主导分数（Z₁）以及理论上的门控主导证明；

**📊 数据集**

使用了多种标准视觉数据集（MNIST、CIFAR‑10、CIFAR‑100、SVHN）和多种网络架构（全连接网络、ResNet、ViT），并在不同宽度、深度和激活函数（ReLU、SiLU、GELU、tanh）上进行实验；

**📈 对比分析**

与完整网络比较时，k近邻符号分类器在过参数化区间内准确率与完整模型几乎相同；当随机化幅度而保持符号时，模型性能仅轻微下降；翻转约20%符号会使准确率降至随机水平；类内符号排斥正则化导致训练准确率保持高位，但测试准确率急剧下降；理论证明显示符号主导分数随宽度√d提升，深度提供投票偏差，从而解释了过参数化下的二进制化现象；

**⚠️ 局限性**

局限性包括：理论分析主要针对全连接网络，残差网络与奇异激活（如tanh）的解释仍依赖经验干预；实验范围局限于小规模视觉分类任务，未验证在大规模数据、语言模型或回归任务中的适用性；

---

## 585. Trajectory-Triggered Sparse Collaborative Decoding

**arXiv ID:** 2610.23613 | [PDF](https://arxiv.org/pdf/2610.23613v1)

**作者:** Amit Ranjan Trivedi `[一作]` (University of Illinois Chicago), Rex Hill `[通讯]` (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出稀疏协作解码，利用草稿模型生成大部分文本，仅在关键时刻调用强大目标模型插入纠正片段；

**💡 创新点**

发现局部不确定性对失败预测无效，利用全局草稿置信度轨迹特征实现高AUC的在线触发；

**🔧 技术方法**

使用聚合置信度轨迹特征、轻量级逻辑回归触发器、基于阈值的调用策略以及文本插入方式；

**📊 数据集**

在GSM8K、MATH‑500、AIME‑2024、StrategyQA等推理基准上评估；

**📈 对比分析**

与草稿、目标、Self‑Consistency、Post‑hoc、Cascade、开放式SCD等方法对比，平均提升5–20个百分点，且在大多数任务上接近目标模型性能；

**⚠️ 局限性**

对触发器的早期准确性、目标模型输出长度与不确定性高度依赖，且仅在共享分词与完整logits可用时才与Speculative Decoding竞争，未涵盖长文本、多轮对话或多语言场景。

---

## 586. Foundation-Model-Driven 3D Scene Graphs for Mobile Manipulation

**arXiv ID:** 2610.23622 | [PDF](https://arxiv.org/pdf/2610.23622v1)

**作者:** Dominic Maggio `[一作]` (Massachusetts Institute of Technology), Luca Carlone `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

将几何基础模型（GFM）驱动的三维场景图集成到移动操纵平台，实现实时、任务驱动、开放集的空间记忆，并在此基础上支持物体变化检测、多会话映射、导航与抓取；

**💡 创新点**

首次提出可在机器人上实时运行的GFM三维场景图管线，具备开放集语义、分层结构（物体、地点、区域），并实现多会话同步、动态变化检测以及基于地图的抓取与导航；

**🔧 技术方法**

使用单目未标定摄像头的几何基础模型（如VGGT）进行相机姿态与深度估计，SAM‑3用于物体分割，VLM嵌入做语义查询，前沿估计+安全节点分类优化地点层，改进子图闭环与里程计融合提升姿态频率，基于解析抓取与纯追踪导航实现操纵；

**📊 数据集**

在公开数据集上评估：3RScan（验证变化检测），Replica（语义检测与时延），TUM RGB‑D（姿态估计）以及真实机器人实验（Cobot Magic在5个室内场景进行物体重排、移动检测、多会话映射等任务）；

**📈 对比分析**

与Where's‑my‑glasses比较，变化检测召回提升至0.408（精度0.064）近乎两倍；在语义检测中，VGGT+PE‑L14达到mAcc≈57.5%，F‑mIoU≈56.6%，FPS≈5.5（比基线快约49%）；姿态估计与传统SLAM相近，RMSE≈0.03‑0.04m，FPS提升至≈6.2；整体实验显示：地点层规划100%成功，目标到达84%，物体检测100%，缺失检测90%，抓取成功率仅50%；

**⚠️ 局限性**

受限于单目SLAM的尺度漂移导致大尺度映射与抓取误差；缺乏深度传感器导致局部障碍规避困难；解析抓取精度低，无法处理复杂物体；无法跨多视角进行语义推理；整体系统对长周期任务的鲁棒性和实时性仍需进一步提升。

---

## 587. Darwin-Claw: Environment-Adaptive Experience Consolidation for Recursively Self-Evolving Agents

**arXiv ID:** 2610.23574 | [PDF](https://arxiv.org/pdf/2610.23574v1)

**作者:** Mingze Yin `[一作]` (Zhejiang University), Chang-Yu Hsieh `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套环境自适应的自演化框架，利用层级化经验整合系统从交互日志中抽取错误归因，生成情境经验并转化为可执行的修正补丁，持续改进代理的感知、技能和长期记忆。

**💡 创新点**

创新点包括：①将经验归因与原型化相结合的层级化经验整合，形成可动态更新的环境原型；②以情境经验为补丁实现无参数更新的全局演化；③在补丁生命周期中引入概率性淘汰与竞争机制，保障补丁质量与可控性；④在实验中实现了跨任务、跨平台的统一演化策略。

**🔧 技术方法**

核心技术：LLM驱动的经验三分法（抽取、筛选、结构化）、因果推断验证、多维评分与加权合成、补丁生命周期状态机、原型向量存储与检索、插件化技能演化。

**📊 数据集**

数据集：WildClawBench（60个多模态长程任务，覆盖代码、检索、创作、工作流、安全、社交等六类）、RSIBench（用于自研训练策略的迭代优化）以及基准评测套件如SWE‑bench（多语种、Pro、Verified）、GPQA Diamond、AIME 2026、Terminal‑Bench 2.0。

**📈 对比分析**

实验对比方法：与基线（无演化）、上下文传递演化、以及仅基于技能演化的三种设置进行比较。结果显示：在WildClawBench上平均提升约24.1%；在RSIBench/后训练任务中，技能演化相较基线提升约3–7个百分点，且在SWE-P、Terminal‑Bench 2.0等高难度任务上提升幅度更显著，整体性能优于其他演化方式。

**⚠️ 局限性**

局限性：①依赖LLM进行经验筛选，容易受模型误判影响；②原型向量与补丁更新的可扩展性待进一步验证，尤其在极大规模多样环境下的存储与检索成本；③补丁生命周期设计虽提高可控性，但对补丁过度保守或过度激进的动态平衡调节仍需改进；④框架不涉及模型参数微调，可能限制对深层行为逻辑的改进。

---

## 588. From Adult Renal Masses to Pediatric Wilms Tumors: Evaluating Cross-Domain Generalization of Kidney Lesion Segmentation on MRI

**arXiv ID:** 2610.23568 | [PDF](https://arxiv.org/pdf/2610.23568v1)

**作者:** Sarah de Boer `[一作]` (Radboudumc), Alessa Hering `[通讯]` (Radboudumc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

训练了基于成人肾癌 MRI 的 nnU-Net 分割模型，并在儿童威尔姆斯瘤 MRI 上进行零射训练评估，随后与儿童专用 nnU-Net 进行对比。

**💡 创新点**

首次系统量化成人大规模数据对儿童肾瘤分割的跨域泛化效果，并提出半监督策略结合 MRSegmentator 自动生成对侧肾脏伪标签来弥补成人数据缺失标注的问题。

**🔧 技术方法**

采用 3D nnU-Net 体系架构，结合半监督训练流程和 MRSegmentator 对侧肾脏定位，最终训练得到统一的肾脏与肾瘤分割模型。

**📊 数据集**

使用成人 Cohort（689 病例、1794 序列，T1/T2）、儿童 Cohort（45 病例、3D T2）以及儿童专用 nnU-Net 作为对照组。

**📈 对比分析**

通过 Dice Similarity Coefficient (DSC) 与 95th-percentile Hausdorff Distance (HD95) 评估，成人模型在成人测试集的 DSC 为 0.86，在儿童数据为 0.85；儿童专用模型在儿童数据上为 0.90；成人模型未能在统计上证明对儿童模型的非劣势。

**⚠️ 局限性**

限制包括：评估不对称（儿童模型使用交叉验证，成人模型使用外部测试集）、儿童样本量小、两组数据在成像协议与标注规范上差异大、未进行任何 fine‑tuning 或域适应，导致难以单独归因性能差距。

---

## 589. Evaluating Predictor-Corrector Dependence in Hard-Constrained Climate Emulators

**arXiv ID:** 2610.23561 | [PDF](https://arxiv.org/pdf/2610.23561v1)

**作者:** Tianyu Zhang `[一作]` (Quebec Artificial Intelligence Institute), Yoshua Bengio `[通讯]` (University of Montreal)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的评估框架，用以区分神经网络预测器与物理校正器（corrector）在气候模拟中的相互影响；

**💡 创新点**

创新点在于通过匹配对照、训练‑推理交叉测试以及原始预测与校正前预测的比较，首次系统揭示了硬约束校正器对学习到的预测器的塑造作用，并证明了直接监督可以修复短时预报缺陷；

**🔧 技术方法**

采用了深度神经网络预测器结合确定性物理校正器（如保守性、正值性约束）、多种对照操作（经度位移、纬度带置换、相位随机化）以及辅助监督损失；

**📊 数据集**

主要使用ACE2模型的历史天气/气候数据（1940年、1990年等）和ConCerNet的101维热方程系统（带解析不变量）进行实验；

**📈 对比分析**

通过相对RMSE、统计显著性检验（bootstrapping、符号翻转）以及与非物理对照的比较来评估模型；结果显示校正器显著提升物理一致性，短时预报在未校正时存在缺陷但长期预报可得到改善，直接监督能够消除短时缺陷并保持校正预报精度；

**⚠️ 局限性**

局限性包括样本种子有限（仅3–9种子）、评估仅覆盖1940年历史数据，未验证对未见时期的泛化；长时预报的结论多为描述性；实验仅局限于ACE2系列和ConCerNet热方程，尚未验证到其他气候模型或校正策略。

---

## 590. LiveSync: Iterative Self-Training for Lip-Sync Generation

**arXiv ID:** 2610.23558 | [PDF](https://arxiv.org/pdf/2610.23558v1)

**作者:** Huaishao Luo `[一作]` (JD Technology), Xiaodong He `[通讯]` (JD Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种自我训练的口型同步框架 LiveSync，利用同一 mask‑based 模型生成对齐的伪配对来逐轮细化自身，从而在保持身份与姿态不变的前提下实现音频驱动的口型编辑。

**💡 创新点**

创新点：① 将伪配对的生成与模型细化在同一闭环中完成，避免了两阶段训练与学生模型；② 通过对伪配对的置信度筛选与真实数据混合训练，提升了对齐监督的质量；③ 采用单通道 GAN 与 VAE 结合的架构，实现了低延迟的实时推理。

**🔧 技术方法**

使用的技术包括：mask‑based GAN + VAE 编码器/解码器；音频条件的 U‑Net 预测器；SyncNet 语音-视觉同步监督；双区域 Wasserstein GAN（面部与口腔）；多尺度参考特征注入；外部数据增强（生成极端姿态、遮挡、化身等）。

**📊 数据集**

数据集：HDTF（真实人脸视频）与 AIGC‑LipSync（AI 生成、化身及非人类角色视频）作为评测基准，训练时还加入外部生成的多样化视频。

**📈 对比分析**

与多种基线（Wav2Lip、VideoReTalking、IP‑LAP、DINet、MuseTalk、LatentSync、X‑Dub、Lip Forcing、TBDub）在同步（Sync‑C/D）、视觉质量（PSNR/SSIM/FID/LPSIPS/CSIM）和延迟等指标上对比。LiveSync 在 HDTF 上同步分数最高，视觉质量与 LatentSync 竞争；在 AIGC‑LipSync 上排名第二/第三，视觉与同步指标表现优良，并且推理延迟仅为 Wav2Lip 的 3.1 倍，是最快的两种方法之一。

**⚠️ 局限性**

局限性：① 视觉质量仍略逊于专门针对分布相似度优化的模型；② 对极端遮挡和非人类角色的细节表现还有提升空间；③ 训练过程依赖多轮伪配对生成与筛选，复杂度较高；④ 目前仅验证了 GAN 架构，扩展到扩散模型尚待探索。

---

## 591. Reason-guided Human Reward Modeling for Preference-based Reinforcement Learning

**arXiv ID:** 2610.23550 | [PDF](https://arxiv.org/pdf/2610.23550v1)

**作者:** Anshul Nayak `[一作]` (Clemson University), Yue Wang `[通讯]` (Clemson University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于理由引导的偏好强化学习框架，利用轨迹级人类偏好与理由标签共同训练离线人类奖励模型，并在策略优化阶段将该奖励与环境奖励融合，提升机器人行为与人类预期的一致性。

**💡 创新点**

创新点在于将理由标签作为额外监督加入奖励学习，既提高奖励模型的可解释性，又通过奖励正则化与偏好学习的组合显著改善下游策略的安全性、效率与人类偏好匹配度。

**🔧 技术方法**

技术手段包括偏好RLHF、交叉熵与二元交叉熵损失、Bradley‑Terry 偏好模型、奖励模型正则化、以及基于梯度下降的策略优化，离线奖励模型在训练完成后被冻结并在在线训练中使用。

**📊 数据集**

使用了由离线预训练RL或基于模型规划生成的轨迹对比数据集，人工标注了350条偏好和多标签理由（如安全、平滑、效率等），数据采集基于IsaacSim离线仿真环境。

**📈 对比分析**

在相同环境配置下通过30次rollout与4个场景种子比较，仅环境奖励策略与结合人类奖励策略的路径长度、成功率、障碍清除等指标，结果显示后者成功率提升至约78‑80%，路径更短、障碍清除略好，且在主观评估中人类偏好匹配度显著提高。

**⚠️ 局限性**

局限性包括仅在单机器人离线导航场景验证，理由标签种类有限；奖励模型依赖离线数据，难以实时适应新情境；未针对多机器人或更复杂人机交互环境进行测试。

---

## 592. Uniform-Price Mechanism Design for Automated Market Makers

**arXiv ID:** 2610.23547 | [PDF](https://arxiv.org/pdf/2610.23547v1)

**作者:** Zhike Peng `[一作]` `[通讯]` (Tsinghua University), Zhike Peng (Tsinghua University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文设计了一种在自动做市商（AMM）环境下的统一价格批次双拍卖机制，解决买卖双方拥有两维私有信息时的交易匹配与价格设定问题。

**💡 创新点**

创新点在于提出了基于二进制量化桶的随机化机制，兼顾了AMM的可行性约束、个体理性和弱量化激励兼容（WQIC），并证明在此框架下可实现对最优可行体积的多项式对数近似。

**🔧 技术方法**

采用的主要技术包括：将量化报告映射为固定块（bucket），随机选择三种清算模式（P2P、AMM卖给买方、卖方卖给AMM），利用批量匹配与关键价格（critical‑price）理论实现WQIC，以及对全IC的下界证明。

**📊 数据集**

本文完全基于理论分析与构造实例，未使用任何实际数据集。

**📈 对比分析**

机制与最优可行均衡（V*）比较时，随机化机制在任何γ≥1时可保证期望体积≥Ω(1/(1+logγ)^2)V* - O(q̅)，而若要求完全IC，则存在实例使期望体积仅能达到O(1/γ)V*，即效率显著受限。

**⚠️ 局限性**

主要局限在于：全IC的严格性导致最坏情况体积只能是O(1/γ)的比例；WQIC虽然提高效率但仍需对量化阈值进行双重随机化，导致对数损失；此外本文未考虑福利、动态批量或多池等更实际的AMM场景。

---

## 593. Humanity's Last Choice: A Common Evaluation Standard for Generative Engine Optimization

**arXiv ID:** 2610.23541 | [PDF](https://arxiv.org/pdf/2610.23541v1)

**作者:** Qianfeng Wen `[一作]` (University of Toronto), Zhenwei Tang `[通讯]` (University of Toronto)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个统一的生成引擎优化（GEO）评估框架，并构建了包含优化层、操作与信号三维的元分类法；在此框架下定义了策略空间、评估协议，并将现有 GEO 方法映射进该空间；提出了基于此框架的公共排行榜和覆盖报告；呼吁社区采用这一标准进行方法、优化器与模型的系统比较。

**💡 创新点**

创新点在于：①构建可检索的元分类法，将编辑目标、编辑操作与目标信号统一表述；②定义了可重用的评估协议，明确任务、编辑约束、可见度、原任务性能与编辑有效性等指标；③提出了针对策略、优化器和模型的三种比较范式，并鼓励发布版本化的任务和策略覆盖信息；④通过元分类法实现不同 GEO 方法间的可比性，促进公共排行榜的发展。

**🔧 技术方法**

主要技术为：基于文本编辑的语义层分类（表面形式、信息内容、文档结构等）、操作描述（插入、删除、重排等）与信号标注（权威性、显著性、证据支持等）；构建可检索的策略空间；制定评估流程和指标计算方法；使用已有 GEO 评测基准（如 C-SEO Bench、E-GEO、GEO-Bench）作为任务来源。

**📊 数据集**

未引入新的数据集，而是借用并整合现有 GEO 基准中的任务与数据（如 C-SEO Bench、E-GEO、SafeGEO、GEO-Bench 等），并提供一个示例 GEO 任务供评测。

**📈 对比分析**

比较方法：①策略对比——在相同源、相同编辑约束下评估不同优化层+操作+信号组合的可见度提升；②优化器对比——在同一策略空间下比较不同搜索/生成算法的可见度提升与资源消耗；③模型对比——在相同编辑后的源上比较不同 LLM 系统的任务完成度与可见度。性能方面未给出具体数值，因为本文为方法与框架设计，未进行实验验证。

**⚠️ 局限性**

局限性：①缺乏实证验证，无法证明所提框架确实提升优化效果或任务表现；②跨层编辑可能存在耦合，导致单一层划分不完全；③固定的元分类法可能忽略领域特定的策略与新兴技术；④双重使用风险，公共策略空间可能被内容提供方用于操纵模型；⑤依赖现有基准，可能不适用于所有场景。

---

## 594. Water-Energy-Data nexus reveals sustainability gaps and priorities for AI infrastructure

**arXiv ID:** 2610.23614 | [PDF](https://arxiv.org/pdf/2610.23614v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 595. PreciseEdit: Strong Yet Stable Semantic Transport for One-Step Image Editing

**arXiv ID:** 2610.23535 | [PDF](https://arxiv.org/pdf/2610.23535v1)

**作者:** Jingyi Wang `[一作]` (Hainan University), Xiao Lu `[通讯]` (Shanghai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PreciseEdit，一种训练‑、反演‑自由的一步图像编辑框架，利用强语义传输并通过方向与空间可靠性实现选择性释放。

**💡 创新点**

创新点：① 认为强语义传输本质上可稳定，关键是可靠方向和位置；② 双噪声尺度设计：高噪声构建 Strong Conditional Transport（SCT），低噪声提供方向校准与结构化空间支持；③ PreciseMask 通过交叉注意力种子、self‑attention 图拉普拉斯细化及自适应软门实现精准、结构敏感的编辑定位。

**🔧 技术方法**

使用技术：① 采用一/多步文本‑到‑图像模型（如 SD‑Turbo、InstaFlow、SwiftBrush‑v2）；② Strong Conditional Transport（SCT）—目标吸引 + 跨尺度方向一致；③ PreciseMask——交叉注意力种子、self‑attention 图、图拉普拉斯优化、阈值软门；④ 目标条件 proximal refinement。

**📊 数据集**

数据集：PIE‑Bench（700 张 512×512 图像，10 类编辑任务，包含源图、目标 Prompt、融合词和 GT 编辑掩码）。

**📈 对比分析**

对比方法：与多种一/多步编辑器（ChordEdit、WhereEdit、SwiftEdit、TurboEdit 等）在 PIE‑Bench 上评估。PreciseEdit 在 PSNR‑Unedit（24.37 dB）和 CLIP‑Edited（23.36）上超过 ChordEdit（+1.7 dB）与 WhereEdit（+0.4 dB），同时保持 1 步低延时和较低 VRAM。

**⚠️ 局限性**

局限性：对噪声尺度和融合词的选择敏感；对极大结构或极细微细节的编辑仍可能受限；在极端多步细化场景下无法替代多步迭代的精细控制；不同模型的注意力层需要手动选取，可能影响跨模型适配。

---

## 596. ThetreStageAI: Multi-View Retrieval-Augmented Generation for Structured Bengali Theatre Production Planning

**arXiv ID:** 2610.23618 | [PDF](https://arxiv.org/pdf/2610.23618v1)

**作者:** Md. Mehedi Hasan Naeem `[一作]` (Jatiya Kabi Kazi Nazrul Islam University), Md. Sujan Ali `[通讯]` (Jatiya Kabi Kazi Nazrul Islam University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 ThetreStageAI，一种面向印地语剧场制作计划的检索增强生成框架。

**💡 创新点**

创新点在于将剧本、演员摆位和灯光三类信息拆分为独立检索视图，并在生成后进行结构与语义双重校验，确保产出可直接用于舞台排练。

**🔧 技术方法**

技术实现包括使用 BGE‑M3 进行多语言稠密检索、Qdrant 存储检索视图、Gemini LLM 生成约束 JSON、Pydantic 进行结构验证，并采用九区分区模型记录演员移动和灯光切换。

**📊 数据集**

数据集为人工指导、LLM 辅助生成的 500 条短篇印地语剧本，经过拆分后得到 1,500 条检索文档（场景、演员摆位、灯光各 500 条）。

**📈 对比分析**

通过在 12 条固定制作简报上对比单一合并检索与完整多视图检索，结果显示后者在验证通过率上达到 100%（单一合并 71.43%），且修复次数为 0%，表明多视图检索显著提升可靠性。

**⚠️ 局限性**

主要局限包括数据集合成导致对话重复率极低、未在真实剧院环境中验证灯光与舞台交互、仅测试 Gemini 模型且未测量能源消耗。

---

## 597. GEMS: Molecular Structure Elucidation as Learned Search over Isomer Graphs

**arXiv ID:** 2610.23617 | [PDF](https://arxiv.org/pdf/2610.23617v1)

**作者:** Utku Umur Acikalin `[一作]` (Cornell University), Carla P. Gomes `[通讯]` (Cornell University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过学习式的2‑switch键连接编辑，结合MIST预测的指纹概率，GEMS在同一分子式的候选结构上进行本地搜索与优化，实现从MS/MS谱图到化学结构的自动解析。

**💡 创新点**

将谱图预测的指纹直接映射到候选结构的原子/键环境，构建一次前向评估即可评估所有合法编辑的学习策略，并能够在无谱数据的前提下进行预训练，从而实现高效的多起点搜索与一致性排名。

**🔧 技术方法**

利用预训练的谱到指纹模型MIST、Transformer+Graphormer结构化导航策略、Markovian 2‑switch编辑搜索、Beam search与多起点一致性评分等技术组合。

**📊 数据集**

在公开的NPLIB1和MassSpecGym两个基准集上进行实验，训练阶段使用PubChem异构体对和预训练数据。

**📈 对比分析**

与FRIGID、MS‑GPT、DiffMS等现有方法在Hit@1/Hit@10指标上对比，GEMS在两大基准上分别提升约10–15个百分点，Hit@1最高可达约50–55%，同时保持较低的搜索时间。

**⚠️ 局限性**

仅适用于已知分子式且不考虑立体化学；性能受MIST指纹预测误差限制；搜索受2‑switch编辑可达性约束，需要足够的同分子式初始候选。

---

## 598. How Many, but How Sure? Class Competition as a Confidence Proxy in Multi-Class Density-Based Counting

**arXiv ID:** 2610.23611 | [PDF](https://arxiv.org/pdf/2610.23611v1)

**作者:** Villanelle O'Reilly `[一作]` (University of Lincoln), Marc Hanheide `[通讯]` (University of Lincoln)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了无训练的“显著度比”作为多类密度估计的置信度代理，并通过降低低显著度通道的权重来改善计数准确性。

**💡 创新点**

显著度比是一种仅基于预测密度图的局部竞争度量，可在推理时实时评估类间分配的可靠性；利用该度量实现的密度衰减显著降低MAE。

**🔧 技术方法**

采用多类密度估计网络（DSACA、GeTNuRi），计算每像素显著度比，按比例衰减通道权重，随后评估MAE变化。

**📊 数据集**

Hicks et al. 数据集、iSAID 数据集及 VisDrone 数据集。

**📈 对比分析**

在冻结模型上对比原始MAE与显著度衰减后MAE；DSACA在Hicks上MAE从0.920降至0.846（8%改进），GeTNuRi在Hicks上改进4.5%，在iSAID改进2.7%；但在VisDrone上略降0.87%。

**⚠️ 局限性**

低显著度并不总是错误，可能代表有效共现；该方法仅为诊断性置信度代理，不能保证修正；需进一步验证其与空间分配误差的对应关系并评估其普适性。

---

## 599. GCAM-Gym: A Benchmark for Agent-Based Climate Policy Search

**arXiv ID:** 2610.23569 | [PDF](https://arxiv.org/pdf/2610.23569v1)

**作者:** Tianyu Zhang `[一作]` (Mila), Yoshua Bengio `[通讯]` (Universite de Montreal)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了GCAM‑Gym——一个兼容Gymnasium的环境，用来统一评估GCAM集成评估模型中七种搜索与控制方法的政策搜索表现。

**💡 创新点**

创新点在于：①提供了一个通用接口既支持完整政策评估又支持逐步控制；②公开了完整的评估记录和分析工具；③在目标‑成本、成本‑温度折衷以及一阶延迟反馈三种设置下系统性地比较了方法性能。

**🔧 技术方法**

使用的技术包括：Gymnasium兼容接口、GCAM 9.1后端、CMA‑ES、Gaussian‑Process Bayesian Optimization、PPO、LLM‑ReAct 以及 LLM‑+‑BO 等搜索/控制算法。

**📊 数据集**

数据来源为GCAM 9.1的默认输入数据（能源、水、土地、气候和经济等），并在该模型上执行政策模拟。

**📈 对比分析**

在目标‑成本、成本‑温度折衷和一阶延迟反馈三种评估设置中，每种方法在10个种子上进行多次尝试，比较最佳回报的四分位中位数（IQM）、目标达成率和成本；结果显示，在24次尝试的预算下，CMA‑ES在所有设置中获得最高IQM，LLM‑+‑BO相较于GP‑BO表现更好但差异不显著。

**⚠️ 局限性**

局限性包括：仅针对单一GCAM配置和固定代理目标进行评估；运行时存在残差检查等错误，可能影响结果；缺乏对多模型、多目标和更广泛情景的通用验证。

---

## 600. Embodiment-Aware Planning for Heterogeneous Bimanual Manipulation

**arXiv ID:** 2610.23551 | [PDF](https://arxiv.org/pdf/2610.23551v1)

**作者:** Sukhvansh Jain `[一作]` (Indian Institute of Technology Mandi), Ravi Prakash `[通讯]` (Indian Institute of Science)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种面向异构双臂机器人系统的体现感知规划框架，利用人类示范轨迹在不同段落分配最优的手臂与抓取模式，实现需要双手协作的搬运与放置任务。

**💡 创新点**

核心创新在于：①将异构抓取器（吸盘与平行夹）协同运用，突破单一抓取器的限制；②以有限视程优化为目标，构造包含抓取可行性、运动可达性、操控性与跨抓取器兼容性的全局动态规划；③引入基于多样性的DPP抓取选取与基于Yoshikawa操纵度的成本函数，提升规划质量。

**🔧 技术方法**

使用技术包括：人类示范轨迹提取（FoundationPose）、基于几何的抓取生成、DPP多样性筛选、IK求解（EAIK）、碰撞检测（MuJoCo）、动态规划求解全局最优序列，并通过束搜索与步长裁剪降低计算复杂度。

**📊 数据集**

主要数据集为实验室自制的四个搬运任务（Plate从桌面到架子、Plate从架子到桌面、倒置杯子、Plate从盒子到盒子），以及对应的CAD模型与物体姿态轨迹；实验亦在仿真环境与实际双臂Frank FR3平台上验证。

**📈 对比分析**

与贪心、无多样性、无操控性、无切换惩罚等基线对比，精确动态规划的任务成功率最高（72.5%），Beam10-Stride3方案在保持100%切换成功的同时将规划时间从532s压至91s，成功率为65%。

**⚠️ 局限性**

局限性包括：①依赖预先已知的CAD模型与示范轨迹，难以零样本部署；②对新物体需要重新生成抓取集；③规划假设环境静态，未考虑动态障碍；④双臂协调仍受限于机械臂间的可及性与接触安全。

---

## 601. From Trajectories to Differentiated Skills: Cognitive Differentiation for Self-Improving Agents

**arXiv ID:** 2610.23599 | [PDF](https://arxiv.org/pdf/2610.23599v1)

**作者:** Yewei Zhao `[一作]` (University of Science and Technology of China), Mingjun Xiao `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过从执行轨迹中提取语义梯度，并将其按功能划分为 Meta、Experience、Tool、Script 四个组件，构建了统一的可演化 LLM 代理技能。

**💡 创新点**

首次将认知科学中的功能区分引入技能演化，提出功能差异化的更新与协同机制，显著提升了技能自我改进的效果。

**🔧 技术方法**

采用基于结果对比的语义梯度发现、功能归纳与保留、组件专属更新、跨组件协调以及自适应调用等技术。

**📊 数据集**

在 OfficeQA、SpreadsheetBench2、SealQA、LiveMathBench、SWE-Pro 与 AutomationBench 等六大基准上进行实验。

**📈 对比分析**

与 Base、Trace2Skill、SkillOpt-lite 与 EvoSkill 等基线对比，CDSE 在 36 个设置中获得最高分，平均提升 Base 7.6pp，优于最强对手 2.6pp。

**⚠️ 局限性**

实验仅覆盖六个基准、三种模型和两种执行环境；多轮实验使用固定轨迹集，且四组件划分仍有待进一步验证。

---

## 602. F-LIVO: A Feature-Level Geometric Measurement Framework for LiDAR-Inertial-Visual Odometry

**arXiv ID:** 2610.23595 | [PDF](https://arxiv.org/pdf/2610.23595v1)

**作者:** Xinye Dai `[一作]` (Wuhan University), Weiming Tang `[通讯]` (Wuhan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于滤波器的 LiDAR‑Inertial‑Visual Odometry 框架 F‑LIVO，利用视觉点与 LiDAR 层面之间的点到平面约束实现特征级跨模态耦合，并在视觉三角测量不佳时通过 LiDAR 地图恢复深度；

**💡 创新点**

核心创新在于直接构造视觉三维点与局部 LiDAR 平面的几何残差，实现跨模态点到平面约束；对 LiDAR 点到平面残差引入自适应协方差模型，根据局部平面质量动态调整权重；通过稀疏几何特征而非密集光度属性实现轻量级地图维护；

**🔧 技术方法**

使用扩展卡尔曼滤波（EKF）与迭代 EKF（IEKF）进行状态估计，融合 IMU、相机和 LiDAR 数据；采用 Delaunay 三角化与平面聚类构造视觉平面；利用局部平面拟合、ray‑plane 交点恢复深度；实现自适应协方差并在 LiDAR 更新中采用 QR 压缩；

**📊 数据集**

在 HILTI SLAM Challenge 2021、i2Nav‑Robot、HKU Mars‑Lab、R3LIVE 校园数据集上进行实验，评估定位精度、连续性和资源占用；

**📈 对比分析**

与 R3LIVE、R2LIVE、FAST‑LIVO2、FAST‑LIO2 等先进方法对比，F‑LIVO 在大多数序列上实现了与 FAST‑LIVO2 相当或更优的 ATE（RMSE/最大误差），在资源消耗上平均 CPU 与内存使用均最低；在退化场景下保持连续估计并显著降低旋转漂移；

**⚠️ 局限性**

受限于对静态环境中可用局部平面约束的依赖，难以在缺乏平面或存在动态物体的场景中保持良好性能；对反射、强光过曝等 LiDAR/视觉异常的鲁棒性仍有限；

---

## 603. SemPurify: Semantics-Aware Data Purification for Black-box Backdoor Defense

**arXiv ID:** 2610.23589 | [PDF](https://arxiv.org/pdf/2610.23589v1)

**作者:** Jiawei Yu `[一作]` (University of Georgia), Jian Liu `[通讯]` (University of Georgia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于语义的黑盒后门防御框架，通过提取低层语义嵌入和高层图像标签来指导数据净化，既去除后门触发器又保持图像语义信息。

**💡 创新点**

创新点在于首次将多层语义信息（低层嵌入+高层标签）注入文本-图像扩散模型（带ControlNet和RCA模块）进行条件恢复，从而在黑盒场景下既有效消除后门又高保真地重建语义内容。

**🔧 技术方法**

技术包括：使用RAM（Recognize Anything Model）提取语义嵌入和标签；采用线性变换（如模糊）破坏触发器；利用预训练的Stable Diffusion文本-图像扩散模型结合ControlNet和RCA进行恢复。

**📊 数据集**

实验数据集包括CIFAR-10、GTSRB、Imagenette（分类任务）、Caltech101、OxfordPets（CLIP任务）、BSD100、Set14（生成任务）等，覆盖多种后门攻击（BadNet、PhysicalBA、Blended、Semantic Blended、WaNet、BadCLIP、生成后门）。

**📈 对比分析**

与 ShrinkPad、BDMAE、ZIP、DataElixir、SampDetox 等基线对比，实验显示该方法在清洁准确率（CA）和攻击成功率（ASR）上均保持竞争力，且在语义保持指标（PSNR、SSIM、LPIPS）上优于多数方法，尤其在保持语义完整性和降低ASR方面表现突出。

**⚠️ 局限性**

局限性包括较高的显存占用（约5764 MB）和对大型预训练T2I模型的依赖，且对极度复杂或自定义触发器的鲁棒性仍需进一步验证。

---

## 604. MARS: Memory-Aware Routing for Session-Level LLM Assistants

**arXiv ID:** 2610.23583 | [PDF](https://arxiv.org/pdf/2610.23583v1)

**作者:** Minghan Wang `[一作]` (Southern University of Science and Technology), Fang Kong `[通讯]` (Southern University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向会话的记忆感知路由框架（MetaMem），通过统一的查询与主题管理器（QTM）把记忆构建和模型路由协调起来，以实现长历史对话的质量–记忆–成本平衡。

**💡 创新点**

创新点在于：① 将记忆视图生成与模型选择通过共享的结构化会话状态来耦合；② 采用 QTM 产生可控的状态信号，驱动记忆策略和在线模型路由；③ 在回合级别利用线性UCB等上下文多臂赌博机实现在线自适应。

**🔧 技术方法**

技术包括：结构化 QTM 预设 schema 与 prompt，记忆视图策略（基于活跃线程、最近回合、历史证据三块），在线模型路由器（LinUCB 形式），回合级反馈奖励函数（质量-成本折衷），以及固定的记忆子系统 Mem0/A-MEM 与多模型池。

**📊 数据集**

使用了两个公开多轮基准：PersonaQA（10 人设 1415 题）和 100 维 BABI 类评测（245 场景 1480 回合）来分别考察长历史记忆驱动与受限会话干扰下的性能。

**📈 对比分析**

对比方法包括：全上下文固定模型、记忆+固定模型、MetaLLM 路由器、记忆+MetaLLM 混合以及提出的 MetaMem。实验显示：在 PersonaQA 上，MetaMem 保留 98.2% 的 GPT‑4o QA 并将生成‑控制成本降至 14.5%；在 BABI 上，MetaMem 在机制聚焦子集（44/137）取得 44/137 的通过率，成本约为 1.57 美元；总体而言，MetaMem 在质量‑成本平衡上优于传统方案。

**⚠️ 局限性**

局限包括：① 对 QTM 的结构化 schema 依赖较高，需手工设计；② 在线路由依赖于回合级反馈，若反馈不准或延迟会影响学习；③ 仅在固定的记忆子系统上评估，未验证对其他记忆引擎的泛化；④ 在极端长历史或高干扰场景下仍可能出现记忆覆盖或模型不匹配。

---

## 605. GRAFT-EEG: Gauge-Aware Physical Field Transport for Cross-Dataset Motor-Imagery EEG

**arXiv ID:** 2610.23620 | [PDF](https://arxiv.org/pdf/2610.23620v1)

**作者:** Ying Zhou `[一作]` (Sun Yat-sen University), Mengting Liu `[通讯]` (Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种目标统计盲的EEG解码方法，利用主动集合参考商和模板运动场转移，将原始多电极信号投射到固定的物理运动场基底，并结合静态频谱锚定和有界状态空间残差；

**💡 创新点**

创新点在于构造精确的访问契约，将目标几何与目标信号/标签分离，采用物理运动场转移与主动集合参考商，实现跨数据集解码而不依赖目标统计；

**🔧 技术方法**

采用了分数参考商、模板运动场转移、静态频谱Logit、有限状态空间模型以及留一数据集交叉验证等技术；

**📊 数据集**

使用了五个公开MOABB运动意象数据集（BNCI2014_001、Cho2017、Lee2019_MI、Shin2017A、PhysionetMI），共253名受试者；

**📈 对比分析**

与严格的清洁预训练基线（BIOT、EEGNet等）进行等目标平衡准确率比较，动态模型实现60.9% BA，优于BIOT的52.7%（提升约8.2个百分点），同父控件下性能下降4–11个百分点，SSM额外提升约1.1个百分点；

**⚠️ 局限性**

局限于二元运动意象任务、仅五个数据集、无参与者特异性源定位、无流式或任意电极布局支持、仅基于物理运动场的模型。

---

## 606. Beyond Action Prediction: Learning Search-Free Planning in Latent Space

**arXiv ID:** 2610.23615 | [PDF](https://arxiv.org/pdf/2610.23615v1)

**作者:** Guanchen Du `[一作]` (Nanjing University of Science and Technology), Zechao Li `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为HIJP的搜索无关的视觉控制框架，利用未来观测作为hindsight目标直接预测动作并与潜在预测器交替生成多步计划。

**💡 创新点**

创新点在于：①将未来观测重新标记为目标以丰富逆向动力学监督，并通过梯度传递塑造目标相关的视觉表示；②将预测动作输入潜在预测器并对齐预测的下一个潜在状态与真实编码，形成动作-转移一致性约束，直接训练与规划时使用的逆向-预测组合。

**🔧 技术方法**

技术包括视觉编码器（ViT‑Tiny）、因果Transformer潜在预测器、逆向动态模型（MLP）、hindsight目标重标、动作-转移一致性损失、联合端到端训练与无搜索的规划循环。

**📊 数据集**

在四个视觉控制任务（TwoRoom、Reacher、PushT、OGCube）上使用LeWM发布的离线数据集，包含224×224 RGB观测和连续动作。

**📈 对比分析**

与八个基线（包括CEM/MPPI搜索的LeWM、SMWM、Delta‑JEPA、INTACT等）对比，HIJP在四个任务上整体成功率达86.8%，比最强基线高4.9个百分点；在规划效率上实现1.9–16.5×的速度提升；执行的动作也比CEM和LeWM平滑得多。

**⚠️ 局限性**

局限性：仅在短期任务（固定规划长度和重规划计划）上验证，未针对极长时程任务；未来需要探索层次子目标、可变规划窗口和动态重规划策略。

---

## 607. ADA 2025: A Multimodal Dataset for Water-Deficit Stress Detection in Sugar Beets

**arXiv ID:** 2610.23609 | [PDF](https://arxiv.org/pdf/2610.23609v1)

**作者:** Piper Powell `[一作]` (Osnabrück University), Martin Atzmueller `[通讯]` (Osnabrück University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文发布了ADA 2025多模态糖 beet 水分胁迫数据集，并基于该数据集进行示例分析。

**💡 创新点**

创新点在于提供超过500 GB的RGB、热像、红外光谱和深度图像，以及环境与真实水分指标（RWC、叶片厚度）的高频、多时段数据，填补了水分胁迫监测领域的公共数据空白。

**🔧 技术方法**

技术方面采用了改装FarmBot机器人、ROS2控制、同步多摄像头采集、几何与光谱校准、RGB‑Depth融合、光谱指数计算、线性混合效应模型及相关性分析等。

**📊 数据集**

使用的数据集为本研究发布的ADA 2025，参考的已有数据集包括PlantVillage、Eschikon Plant Stress Dataset等。

**📈 对比分析**

通过对照组与处理组的线性混合模型、OLS回归和Spearman相关性评估，结果显示三种光谱指数、叶片温差及RWC等指标均显著区分健康与水分胁迫植物，证明数据具有可检出性和潜在应用价值。

**⚠️ 局限性**

局限性包括背景分割挑战、传感器停机导致的数据缺失、光照变化对光谱图像的影响，以及该数据集未构成完整的基准评测框架。

---

## 608. RL-Guided Temporal Localization for Dual-Channel Retrieval in Long-Horizon Agent Memory

**arXiv ID:** 2610.23598 | [PDF](https://arxiv.org/pdf/2610.23598v1)

**作者:** Dabin Sheng `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 TIDER，一种在长时长 LLM 代理记忆检索中加入时间窗口预测的框架，能够在检索阶段主动限定时间范围，减少与目标时间不匹配但语义相似的干扰记忆；

**💡 创新点**

创新点在于：① 用强化学习训练的时间区间定位器将时间约束解耦离语义编码，直接输出可解释的时间窗口；② 构建并行的时间通道与语义通道，并通过距离衰减重新排序提升时序匹配；③ 采用复合奖励（是否需要时间、窗口精度、证据覆盖、答案正确率）将稀疏的答案监督转化为检索阶段的细粒度反馈；

**🔧 技术方法**

核心技术包括：强化学习（GRPO）训练的时间区间定位器、双通道检索（BGE-M3+BM25 语义通道 + 时间窗口过滤），窗口距离衰减 reranker，以及多维奖励设计；

**📊 数据集**

使用 LifeBench（个人长期记忆问答）和 LoCoMo（多会话长时长 QA）两个基准数据集进行评估；

**📈 对比分析**

与 MemOS、TiMem、Zep、Mem0、TsContriever、NeSTR 等多种基线对比，TIDER 在 LifeBench 上实现 61.52%（比第二名 NeSTR 提升 8.82%），在 LoCoMo 上实现 78.96%（比第二名 TiMem 提升 3.66%），并在多类子任务中均位列首位；

**⚠️ 局限性**

局限性包括：① 时间窗口预测在稀疏时间线下仍存在误差，影响极难区分的多窗口查询；② 强化学习训练需要标注的时间窗口与答案，标注成本高；③ 依赖于已构建的时间聚类记忆，记忆构造错误会直接影响检索质量；④ 目前仅在日常生活事件数据上验证，对更大规模或不同领域记忆的泛化尚未评估。

---

## 609. Boundary Detection for High-Dimensional Point Clouds

**arXiv ID:** 2610.23590 | [PDF](https://arxiv.org/pdf/2610.23590v1)

**作者:** Abigail Hickok `[一作]` (Columbia University), Andrew J. Blumberg `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出两种高维噪声点云的边界检测与法向量估计方法，分别为局部PCA+最优超平面法和方向均匀性法。

**💡 创新点**

创新点在于利用局部维度降维后求解闭式最优超平面以区分边界与内部点，以及通过方向向量一致性度量鲁棒检测边界，且两方法均给出每点法向量。

**🔧 技术方法**

技术主要包括局部PCA降维、最优超平面优化（Lagrange乘子求闭式解）、方向均匀性统计、密度校正、AUC评估与可视化。

**📊 数据集**

使用合成高维球/半球、含噪声的曲面、Henneberg表面、不同噪声与维度的圆盘、以及真实数据集：空间转录组(ST1A1)与LLM残差流激活( Pythia‑2.8B、Llama‑3.1‑8B)。

**📈 对比分析**

与LoDD、BEPS、CO、RU等现有方法对比，AUC在大多数场景接近1；在高维、噪声或不均匀密度情况下，HP与DU表现优于其他方法，且能够提供精确法向量。

**⚠️ 局限性**

局限性包括对局部维度估计敏感；在高噪声时HP易误判内部为边界；DU在高维噪声下精度下降；需要手动设置邻域大小和维度阈值。

---

## 610. Enhancing Visual Task-Redirection Attacks on Vision-Language-Action Models via Closed-Loop Optimization

**arXiv ID:** 2610.23566 | [PDF](https://arxiv.org/pdf/2610.23566v1)

**作者:** Chunjin Ye `[一作]` (Sun Yat-sen University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种闭环优化框架TRACO，用于提升视觉语言动作模型(VLA)在执行任务时的攻击效果，使攻击者能在机器人执行过程中实时重建监督并引导机器人完成攻击者指定的任务。

**💡 创新点**

创新点在于：①引入Execution-Driven Supervision (EDS)让监督随攻击轨迹动态更新；②通过Progress-Guided Selection (PGS)将监督聚焦在进度前沿同时保留已完成的行为；③把已有攻击方法的对齐目标与闭环监督结合，形成可复用的攻击框架。

**🔧 技术方法**

技术主要包括：视觉扰动优化、闭环执行反馈、目标对齐损失（动作、CoT、流匹配等）、梯度投影更新以及进度估计与前缀/前沿监督分组。

**📊 数据集**

使用LIBERO、LIBERO-Goal、LIBERO-Plus、SimplerEnv等仿真环境中的VLA模型和任务对，真实实验使用StarVLA GR00T在SO-101机械臂上的打印贴纸。

**📈 对比分析**

与随机噪声及多种离线目标对齐攻击（Action Attack、TRAP、TRAP_CoT等）比较，TRACO在LIBERO全局与补丁设置、SimplerEnv补丁设置均显著提升攻击成功率和Hijack Score，尤其在未知布局上的性能提升更为显著。

**⚠️ 局限性**

局限性包括：需多次与环境交互，增加构造时间和计算成本；对VLA的恢复能力依赖，如果模型无法从攻击状态恢复，监督可能无效；进度估计仅基于末端执行器位置，可能忽略抓取等关键交互细节。

---

## 611. Beyond Expected Values: Risk-Sensitive Planning with Distributional Monte-Carlo Tree Search

**arXiv ID:** 2610.23565 | [PDF](https://arxiv.org/pdf/2610.23565v1)

**作者:** Quynh Trinh `[一作]` (Hanoi University of Science and Technology), Tuan Dam `[通讯]` (Hanoi University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CATSO 与 PATSO 两种分布式 MCTS 算法，在 Q-edge 上维护有限支持的经验分布，并使用 CVaR 作为决策指标。

**💡 创新点**

创新点在于：① 只在 Q-edge 维持分布信息、V‑node 保持标量，兼顾计算效率；② 结合 Thompson 采样与多项式乐观奖金实现 CVaR 感知的探索；③ 对有限深度嵌套 CVaR 目标给出 O(n^-1/2) 根值误差和简单 regret 上限。

**🔧 技术方法**

技术主要包括 Dirichlet/粒子 Thompson 采样、离散 CVaR 计算、分布式备份、有限深度嵌套 CVaR 目标、离散化与 W₁ 压缩误差分析。

**📊 数据集**

使用了五个自定义安全决策基准：Autonomous Vehicle、Two‑Level Risky Treasure、Risky Shortcut Gridworld、Betting Game、Guarded Maze。

**📈 对比分析**

与传统标量 MCTS（UCT、Power‑UCT 等）在下尾指标（CVaR 值差、Monte‑Carlo CVaR 回报、灾难事件计数）上显著优于标量基线，在期望回报上保持竞争；实验结果表明分布式备份有效提升安全性和尾部表现。

**⚠️ 局限性**

局限性包括：需要预设离散网格或粒子数，可能引入离散化/压缩误差；仅在有限深度嵌套 CVaR 目标上给出理论保证，未覆盖连续/部分可观测或无限期望情境；对高维连续空间的扩展仍需研究。

---

## 612. An Interactive GraphRAG Workspace for Reifying, Auditing, and Steering Conversational AI Memories

**arXiv ID:** 2610.23559 | [PDF](https://arxiv.org/pdf/2610.23559v1)

**作者:** Munshi Md Arafat Hussain `[一作]` (University of Dhaka), Md Saeed Siddik `[通讯]` (University of Dhaka)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

介绍了 CognitiveCanvas，一个交互式 GraphRAG 工作空间，能够将对话式 AI 记忆转化为可视化、可编辑、可审计的知识对象，帮助开发者跨会话共享和更新技术上下文。

**💡 创新点**

创新点包括：① 层次语义事实图（HSFG）将对话压缩92.2%，支持多跳图检索；② 连续时间漂移审计器（60+正则+LLM 校验）实时标记过时内容并提供一键修正；③ 空间符号提示编译器将二维手势转化为 LLM 注意力权重，支持可视化的提示调节。

**🔧 技术方法**

使用了交互式 SVG 2D 画布、Delaunay/Voronoi 聚类、Neuro‑symbolic 图数据库（Neo4j）、多阶段正则+LLM 时间漂移检测、GraphRAG、注意力权重编译以及 LLM（如 ChatGPT/Claude）作为生成与验证引擎。

**📊 数据集**

使用了公开示例对话记录（调试、架构设计等）、内部基准对话（10 轮共 5210 原始 token）、实验中 24 名软件从业者的真实工作场景，以及公开的技术文档/API 版本信息用于漂移检测。

**📈 对比分析**

通过对 24 名参与者的对照实验（Commercial Baseline、Flat‑Vector RAG、CognitiveCanvas），比较了 Token Overhead、Multi‑Hop Recall@3、Temporal Hallucinations、Citation Faithfulness、Task Time、NASA‑TLX、UMUX‑Lite 等指标。CognitiveCanvas 在 token 开销降 92.2%，多跳召回提升 37.5%，时间减少 34.9%，NASA‑TLX 工作量下降 41.3%，引用可信度提升 27.5%。

**⚠️ 局限性**

限制包括 Stage 2 LLM 校验延迟 600–1200 ms，缺乏直接 AST 可视化，未实现多开发者协作画布，未来工作计划加入协作与多模态图形重化。

---

## 613. SkillBrowser: Selective Skill Exposure for Scalable Skill-Using Agents

**arXiv ID:** 2610.23554 | [PDF](https://arxiv.org/pdf/2610.23554v1)

**作者:** Jin Ye `[一作]` (Nanjing University), Lan-zhe Guo `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种 Meta‑Skill 管理的 SkillBrowser 框架，支持 LLM 代理在推理过程中动态判断是否需要外部技能、生成针对性查询并按需加载技能内容；

**💡 创新点**

创新点在于将技能使用视为一个可选决策，通过 Meta‑Skill 让代理主动发起查询，实现 Selective Skill Exposure（选择性技能暴露），并引入能力导向的查询与混合检索（SkillGrep）；

**🔧 技术方法**

采用 Meta‑Skill 设计、能力导向检索 SkillGrep（融合稀疏词典、稠密向量与复合得分）、LLM 交互与多步工具调用；

**📊 数据集**

在 SRA‑Bench 的六个子集（TheoremQA、LogicBench、CHAMP、MedCalcBench、BigCodeBench、ToolQA）上进行实验，使用 Qwen3‑8B、Qwen3.6‑Plus、GLM‑5.1、GPT‑5.4‑mini、Gemini‑3.1‑Flash‑Lite 五个 LLM；

**📈 对比分析**

与静态 BM25/SkillGrep 检索、LLM‑Selection 基线以及金手册暴露进行对比，SkillBrowser 在 26/30 设定中均优于静态检索，在大多数任务中提升 3‑10% 任务准确率，尤其在 ToolQA 的多步工具交互中显著提升；

**⚠️ 局限性**

局限包括：模型可能无法准确识别何时需要技能，导致对有益技能的低调用率；对不必要技能的低调用率仍需改进；检索引入额外延迟，且对动态更新的技能仓库未做评估；以及未对技能内容的安全性与可信度进行验证。

---

## 614. Execution Before Memory: Diagnosing Memory Failures in Video World Models

**arXiv ID:** 2610.23542 | [PDF](https://arxiv.org/pdf/2610.23542v1)

**作者:** Weiyang Jin `[一作]` (University of Hong Kong), Xihui Liu `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MIST 评测框架和 M3-Agent 方法，用以区分视频生成模型的场景设定、隐藏变更与最终帧显示，从而更精准评估其记忆能力。

**💡 创新点**

创新点在于将传统一次性记忆评分拆分为三道门（设定、泄漏、揭示），并通过在测试时重写失败操作实现无新增记忆模块的即时改进。

**🔧 技术方法**

利用冻结的视频生成器（Wan TI2V‑5B、T2V‑A14B、MiniMax‑H3）和基于 VLM 的判别器，结合文本与图像双重评判。

**📊 数据集**

使用 Wan TI2V‑5B、T2V‑A14B、MiniMax‑H3 生成的四千多条短视频，并对 WRBench、MBench‑T 等官方记忆基准进行评测。

**📈 对比分析**

与基线相比，MIST 揭示分数从 0.147 提升至 0.492，WRBench T 从 0.452 提升至 0.570（+26.1%），MBench‑T U 从 0.292 提升至 0.446（+52.7%），显示显著性能提升。

**⚠️ 局限性**

不足之处在于未能证明模型实际存储了目标状态，重写操作仍可能产生错误，且评测依赖冻结模型和判别器，缺乏对不同模型的泛化评估。

---

## 615. Reinforcement Learning Based Multi-Terrain Adaptive Whole-Body Control for a Quadruped Manipulator

**arXiv ID:** 2610.23548 | [PDF](https://arxiv.org/pdf/2610.23548v1)

**作者:** Yanzhou Jin `[一作]` (University of Tokyo), Yaonan Zhu `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于强化学习的全身控制框架，用于在复杂地形上实现四足机器人与小型机械臂的协同定位与操作；

**💡 创新点**

创新点在于：①混合任务空间表示——将末端执行器位置以机器人身体坐标系表示，方向以重力参考的世界坐标系表示；②基于状态的优先级层次学习，将身体姿态、行走与操作目标按优先级动态加权；③轻量级1.5 kg机械臂与Unitree Go1的集成，实现了实地30°斜坡、楼梯及柔性地形的跨越与抓取；

**🔧 技术方法**

采用Proximal Policy Optimization（PPO）训练单一共享策略；利用Hybrid Task Representation、Priority Vector、以及离散化的多目标奖励函数；仿真环境采用Isaac Lab，含随机地形、质量、外力、延迟等域随机化；

**📊 数据集**

未使用公开数据集，而是自生成多种随机地形（平地、斜坡、楼梯、盒子、障碍）以及随机质量、外力等训练样本；

**📈 对比分析**

通过对比三种失调层次（Priority I–III）以及不同机器人/负载配置，展示了完整策略在Base Motion Error、EE Tracking Error及关节过载率方面均优于基线（基于IK的层次控制、仅行走控制、重载机械臂）且在最难地形上保持低误差；

**⚠️ 局限性**

局限性包括：仅在单一平台上验证，未覆盖更复杂的抓取/搬运任务；实地实验缺乏全景运动捕捉，评估基于IMU+前向运动学的误差；以及对不同感知模块的泛化仍待进一步验证。

---

## 616. Integrated sensing, communications, and identification for low-altitude economy

**arXiv ID:** 2610.23602 | [PDF](https://arxiv.org/pdf/2610.23602v1)

**作者:** Shanxing Zeng `[一作]` (University of Electronic Science and Technology of China), Ying-Chang Liang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述低空经济背景下低空信息网络（LAINs）中，整合感知、通信与识别的研究现状与发展趋势，并提出基于Backscatter通信（BackCom）的电子车牌式识别方案，将识别功能嵌入ISAC框架，构建ISAI与ISCAI三维协同体系。

**💡 创新点**

创新点在于：① 将BackCom作为识别机制，弥补传统识别方式与ISAC的耦合缺失；② 形成统一波形、资源与接收器设计的ISAI/ISCAI框架；③ 归纳信息理论、AI与RIS等前沿技术在ISCAI中的应用方向。

**🔧 技术方法**

所用技术包括：Backscatter通信（BD与读写机协同）、OFDM/OTFS/AFDM/FMCW等ISAC波形、统一信号模型、单/双/分布式感知架构、资源调度与飞机轨迹优化、AI/深度学习用于杂波抑制与参数估计、RIS增强回波与信道。

**📊 数据集**

本文为综述性工作，并未使用具体实验数据集，而是梳理已有研究中使用的雷达、通信、感知数据、仿真与测量场景。

**📈 对比分析**

对比方法主要是理论/仿真评估：讨论通信-感知的性能权衡（如信噪比、误码率、探测概率、定位精度）以及ISAI/ISCAI相较于单一功能系统在资源利用率、识别成功率等多维指标上的提升，具体数值因场景而异。

**⚠️ 局限性**

局限性：识别与感知的耦合尚未在实际系统中验证；BackCom实现需要高效能读写机与同步；多功能接收机设计与算法复杂度高；信息理论框架与性能上限尚缺失；AI、RIS集成仍处于探索阶段。

---

## 617. Enhancing RAG-based Medical VQA with Gated Multimodal Representations

**arXiv ID:** 2610.23588 | [PDF](https://arxiv.org/pdf/2610.23588v1)

**作者:** Daniela Jimenez-Pano `[一作]` (INAOE), Delia Irazú Hernández-Farías `[通讯]` (INAOE)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种多模态RAG框架，利用门控多模态单元（GMU）学习图像与问题的联合查询表示，并保持医学视觉语言模型（Med‑LVLM）不变，提升医学VQA的答案真实性。

**💡 创新点**

创新点在于将GMU嵌入检索阶段，实现视觉与文本信息的动态融合，四种对齐损失配置以及针对检索上下文的专门知识库索引，首次将联合查询表示直接用于检索。

**🔧 技术方法**

技术上使用了GMU、双编码器（MobileNetV2 + PubMedBERT）与BiomedCLIP、InfoNCE对比损失、FAISS向量检索、以及冻结的LLaVA‑Med‑1.5生成模型。

**📊 数据集**

实验数据集包括MIMIC‑CXR、IU‑XRay以及Harvard‑FairVLMed三大公开医学影像与文本对齐数据集。

**📈 对比分析**

与无检索的LLaVA‑Med‑1.5、单模检索基线以及RULE、MMed‑RAG等先进方法对比，模型在多数数据集与指标（准确率、F1、AUROC）上表现相当甚至优于对手，特别是在FairVLMed上提升显著。

**⚠️ 局限性**

局限性体现在不同数据集上的提升不一致（如IU‑XRay上增益有限）、对k值的敏感性以及需进一步探究在何种条件下能稳定获得一致提升。

---

## 618. CLEAR: Cooperative Latent Flow for Visual Counterfactual Explanations with Accurate Region Preservation

**arXiv ID:** 2610.23577 | [PDF](https://arxiv.org/pdf/2610.23577v1)

**作者:** Yan Zeng `[一作]` (Technical University Of Denmark), Anders Bjorholm Dahl `[通讯]` (Technical University Of Denmark)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CLEAR 方法生成视觉对抗性解释，兼顾真实性与编辑效率。

**💡 创新点**

创新点在于利用流匹配即时速度将分类器引导分解为协作与冲突部分，并按冲突强度动态衰减冲突分量，同时通过自适应可编辑区域控制 VAE 重建泄漏。

**🔧 技术方法**

使用流匹配（Flow Matching）、VAE、Grad‑CAM 可编辑区域检测、分类器指导与自适应流匹配更新。

**📊 数据集**

在 CelebA‑HQ（笑容、年龄）和 ImageNet（多类别对）数据集上评测。

**📈 对比分析**

与 MaskDiME、DiME、LDCE、ECED 等基线对比，CLEAR 在 FID、sFID 及计算效率（速度提升 2.5×、显存 58×）上均优于或匹配最佳基线。

**⚠️ 局限性**

局限在于实验覆盖任务有限，仅在 CelebA‑HQ 的笑容任务进行 ablation，未来需扩展更广泛任务与更细粒度分析。

---

## 619. PanoBench: A Panoptic Occupancy Prediction Benchmark for Autonomous Driving

**arXiv ID:** 2610.23537 | [PDF](https://arxiv.org/pdf/2610.23537v1)

**作者:** Yi Feng `[一作]`, Rui Fan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PanoBench，一个统一的相机中心全景占用预测基准，包含合成和真实数据的完整、时序一致的全景占用标注，并提供统一稀疏表示和评估协议。

**💡 创新点**

创新点包括：① 通过多源数据（CARLA、nuScenes、KITTI-360）构建完整的实例级全景占用；② 引入基于时间的有效掩码和实例级融合策略；③ 统一稀疏占用格式，显著降低存储与 I/O 成本。

**🔧 技术方法**

使用技术包括：合成 Mesh 直接生成占用；真实数据中语义 LiDAR 聚合、HD 地图地面拉升、拓扑感知平面恢复、图像条件 Mesh 生成、稀疏查询、Transformer、稀疏卷积、RayIoU/RayPQ 评估等。

**📊 数据集**

采用的数据集有：CARLA（synthetic）→ PanoBench-CarlaOcc；nuScenes（real）→ PanoBench-nuScenes；KITTI-360（real）→ PanoBench-KITTI360；并与 SurroundOcc、OpenOccupancy、Occ3D 等公开基准对比。

**📈 对比分析**

通过统一有效掩码下的 IoU、mIoU、RayIoU@1,2,4、RayPQ@1,2,4 以及 4D 预测 mIoU/IoU 进行评估。结果显示不同模型在各度量上排名差异显著，PanoBench 标注在观察精度、几何完整性、连通性和时序一致性方面均优于现有基准。

**⚠️ 局限性**

局限性包括：缺乏长时延 4D 全景预测与开放类别支持；真实数据中的实例分割仍具有高难度；跨域迁移受相机覆盖差异限制，Mesh 生成质量仍受视图和模型限制。

---

## 620. Stereo LWIR-RGB Automatic Labeling for Low-Light Camera-Based VRU Detection

**arXiv ID:** 2610.23621 | [PDF](https://arxiv.org/pdf/2610.23621v1)

**作者:** Dimitrios Bouzoulas `[一作]` (Aalto University), Risto Ojala `[通讯]` (Aalto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套基于 LWIR‑RGB 立体相机的低光环境下 VRU 自动标注流水线，利用热像仪检测得到标签后直接转移到 RGB 图像，实现无人工标注的大规模数据生成。

**💡 创新点**

创新点在于：1）用立体相机代替复杂光束分裂器，显著降低硬件成本和复杂度；2）将 LWIR 检测结果自动映射到 RGB，构建跨模态标签转移机制；3）通过标签过滤与置信度/高度阈值实现高质量自动标注。

**🔧 技术方法**

技术手段包括：LWIR 与 RGB 立体校准（基于棋盘激活光源），基于 YOLOv11l、RF‑DETR、SAM3 等模型在 FLIR‑ADAS 上微调得到 LWIR 检测器；标签过滤器（置信度阈值 c_t、边界高度阈值 h_t）；直接坐标映射实现标签转移；最终用 YOLOv11l、DETR、SAM3 在自动标注数据上训练 RGB 检测模型。

**📊 数据集**

使用的主要数据集是：1）芬兰首都地区（Espoo、Helsinki）收集的低光 LWIR‑RGB 图像；2）FLIR‑ADAS 数据集用于 LWIR 检测器微调；3）手工标注的 441 张验证图像（336 个人实例）和 1400 张测试图像（1173 个人实例）用于评估。

**📈 对比分析**

与 COCO 预训练的 YOLOv11l、DETR、SAM3 以及在验证集微调后的同类模型进行对比。自动标注训练的模型在 mAP@50、F1、LAMR 等关键指标上均显著优于基线，最高提升约 41.7% 的 mAP@50，并在所有指标上击败手工标注微调模型。

**⚠️ 局限性**

主要限制包括：1）立体校准误差导致标签转移存在定位误差，影响 mAP@50‑95；2）时间同步精度受限（LWIR 无 “fire‑on‑command”）；3）LWIR 检测器性能决定标签质量，当前模型仍受限；4）对小样本微调的提升有限，表明难以在低光条件下仅靠少量人工标注提升模型。

---

## 621. SIT-Eval: Situation-Based Interactive Turn-by-Turn Evaluation of Conversational Agents

**arXiv ID:** 2610.23612 | [PDF](https://arxiv.org/pdf/2610.23612v1)

**作者:** Ravil Mussabayev `[一作]` (Satbayev University), Andrey Savchenko `[通讯]` (Sber AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SIT‑Eval 框架，通过 LLM 生成的交互式对话和已验证的 LLM 判断器实现对话式 AI 在多种情境下的细粒度评估。

**💡 创新点**

创新点在于①统一动作采样以保证稀有情境的统计显著性；②以专家制定的 12 条规则为评估标准，提供逐轮、规则级别的分数；③通过人类验证的“黄金”对话对 LLM 判断器进行系统校准。

**🔧 技术方法**

采用大语言模型（如 Claude Sonnet 4.5、GPT‑5、Gemini 2.5 Pro）作为对话生成者、学生和评判者；使用 LLM‑judge 进行规则符合度判断；利用均匀采样的动作空间驱动情境生成。

**📊 数据集**

主要使用 GSM8K（500 条数学问题的子集）生成黄金对话，随后用 170 条问题进行正式基准测试。

**📈 对比分析**

对 8 大 LLM（含专用辅导模型和通用模型）在 1292 轮对话中按 12 条规则进行逐轮评分；结果显示总体 84.4% 判断准确率，模型间在个别规则上的差异明显，所有模型在模糊澄清和 Socratic 询问上表现最差。

**⚠️ 局限性**

局限包括①均匀采样导致对话自然性不足；②规则主观性和文化适用性有限；③对样本量有限的情境-规则对的统计稳健性不足；④仅评估过程而非学习成效；⑤评估效率可进一步提升。

---

## 622. A Taxonomy of Predicate Transformer Monads for Specifying Monadic Programs

**arXiv ID:** 2610.23604 | [PDF](https://arxiv.org/pdf/2610.23604v1)

**作者:** Abigail Pribisova `[一作]` (MPI-SP), Exequiel Rivas `[通讯]` (Tallinn University of Technology)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了完整的强后置谓词变换器单子（strongest postcondition monads）体系，并将其与弱前置（weakest precondition）单子统一成一套谓词变换器单子分类（taxonomy），实现对所有四种无确定性/非终止解释（天使/恶魔无确定性与完全/自由终止）的支持。

**💡 创新点**

创新点在于：①首次给出合法的四类强后置单子并通过箭头（arrow）而非传统单子变换器实现状态等副作用的模组化；②将原先的三元组关系直接映射为单子之间的关系，构建完整的三元组与单子之间的Galois连接；③基于这些单子构造了对应的Dijkstra单子，并与错误逻辑（incorrectness logic）等形式化体系建立等价性。

**🔧 技术方法**

主要技术手段包括：谓词变换器单子与箭头结构、单子与箭头变换器、单子同态（monad morphism）与箭头同态（arrow morphism）、状态单子/异常单子变换器、以及Coq证明助手中的机器化证明。

**📊 数据集**

由于工作纯粹是理论框架与证明，未使用任何外部数据集；实验部分仅通过约30个示例程序验证了理论结果。

**📈 对比分析**

方法论上通过形式化证明展示了不同单子之间的蕴含、对偶关系与Galois连接，证明了在这些单子上构造的三元组与Dijkstra单子在语义上等价；在Coq中对所有定义与定理进行了机械化验证，保证了形式化正确性。

**⚠️ 局限性**

局限性包括：①未实现完整的可自动化验证框架，仅给出理论基础；②部分单子（如非总或非自由的sp）不满足规范单子（specification monad）条件，因而无法构造对应的Dijkstra单子；③当前仅覆盖状态与异常等副作用，未扩展到IO、概率或控制流等更复杂效果；④证明依赖于函数齐性、命题齐性与排中律，限制了可移植性。

---

## 623. Material and Defocus Cues for Occluded Object Localization in Single Hyperspectral Images: An Exploratory Study

**arXiv ID:** 2610.23733 | [PDF](https://arxiv.org/pdf/2610.23733v1)

**作者:** Ali Zia `[一作]` (Griffith University), Jun Zhou `[通讯]` (Griffith University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在单张高光谱图像中，利用材料分组、光学散焦信息以及基于材料感知的图拉普拉斯传播，实现对部分遮挡物体的定位与边界框提取

**💡 创新点**

将非负矩阵分解得到的材料分组与基于KL散度的光学散焦分数结合，并通过材料感知图谱进行传播，形成一种无需额外深度传感器的定位管线

**🔧 技术方法**

非负矩阵分解（NMF）实现材料分组，KL散度光学分数提取散焦信息，基于材料的图拉普拉斯传播进行稀疏观测插值，连通区域分割与边界框生成

**📊 数据集**

30个室内场景 + 6个室外场景的自制高光谱数据集，使用Brimrose AOTF 61波段（400–1000 nm）和IMEC Snapscan VNIR 150波段（470–900 nm）拍摄

**📈 对比分析**

未进行定量对比，所有结果仅为定性可视化；未报告平均精度(AP)、误差指标或与基准方法的性能比较，因而无法评估具体性能

**⚠️ 局限性**

材料与物体不等价；光学分数受光照、纹理、饱和度等多因素影响；图传播易导致实例合并；单视角无法恢复完全遮挡表面；缺乏可重复实现和标准化评估

---

## 624. Escaping Parameter Space: Tight Generalization Bounds via Representation Quality

**arXiv ID:** 2610.23729 | [PDF](https://arxiv.org/pdf/2610.23729v1)

**作者:** Niclas Göring `[一作]`, Ard A. Louis `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于最后隐藏层表示的确定性神经网络泛化上界，只使用k-NN解码器的误差、解码器与网络的差异以及训练稳定性三项经验量。

**💡 创新点**

创新点在于：①完全不依赖参数空间复杂度度量（权重范数、KL、压缩等）；②上界可拆解为三项，可作为诊断工具；③在预训练和从零训练两种场景下均能给出非空洞、紧凑的上界，首次在ImageNet规模给出k-NN解码器非空洞证书。

**🔧 技术方法**

技术方法包括：交叉验证(k-fold)的k-NN解码器、KL浓度不等式、经验误差估计、对折叠模型与完整模型的稳定性评估，以及对预训练表示的直接评估。

**📊 数据集**

使用的数据集包括CIFAR-10、CIFAR-100、ImageNet（val 和 ImageNet-V2）以及标准测试集（MNIST、SVHN），并在这些数据上对ResNet、ViT、DINOv2等模型进行评估。

**📈 对比分析**

与已有的PAC-Bayes、压缩、分区等方法对比，跨折叠证书在MNIST上与最优已知结果相当，在CIFAR-10上显著优于同类方法（例如ResNet-18 13.0% vs 17.36%），在CIFAR-100首次给出非空洞证书（ResNet-50 44.6%）。

**⚠️ 局限性**

局限性包括：上界是经验上估计，需对不同折叠模型进行一次性训练，计算成本高；仅适用于最后层的ℓ₂距离；不涉及公平性、鲁棒性、回归或其他度量，且对不同优化器的理论保证仍未给出。

---

## 625. Tool Recommendation for Multi-Step Agents: Finding Every Tool a Task Needs

**arXiv ID:** 2610.23707 | [PDF](https://arxiv.org/pdf/2610.23707v1)

**作者:** Sam Partee `[一作]` (Arcade), Valerie Fanelle `[通讯]` (Arcade)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并部署了一种名为Tool Recommendation的系统，在Arcade平台上通过三种工具（SelectTools、UseTool、ListApps）让LLM代理访问完整的8,200工具目录，从而避免因工具缺失导致多步骤任务失败。

**💡 创新点**

创新点在于通过在网关层面实现工具推荐，利用结合BM25和稠密检索的两阶段召回+交叉编码器重排序方法，使代理无需将完整工具定义放入上下文即可完成任务，同时保持低成本。

**🔧 技术方法**

主要技术包括MCP协议、Lexical BM25、稠密向量检索、交叉编码器重排序、门控授权和多轮对话缓存。

**📊 数据集**

使用的数据集包括公开的ToolRet基准（44,453工具、7,961查询），Arcade公共工具目录（8,252工具），以及内部30–34条多步骤工作流进行成本评估。

**📈 对比分析**

与BM25基线相比，Completeness@10提升57%，Recall@10提升49%，在ToolRet上优于大多数公开系统；在Codex和Claude Code多步骤会话中，输入token下降17–40%，总账单下降13.6%，任务完成率保持不变。

**⚠️ 局限性**

局限性包括仅在Arcade环境可用，评估仅针对固定工具列表与推荐的比较，成本模型基于假设且未考虑不同客户端行为，且静态工具集在小规模或高搜索频率场景下仍更具成本效益。

---

## 626. Blitz: Accelerating the Diffusion Modules of AlphaFold3 and Boltz-2

**arXiv ID:** 2610.23703 | [PDF](https://arxiv.org/pdf/2610.23703v1)

**作者:** Vinay Saji Mathew `[一作]` (Pennsylvania State University), William K. M. Lai `[通讯]` (University at Buffalo, State University of New York)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Blitz 框架，通过几步蒸馏将 AlphaFold 3 和 Boltz-2 的 200 步扩散预测压缩到仅 8 步，并在推理时加入物理引导和无参考候选重选。

**💡 创新点**

创新点在于：① 使用几何噪声时间表平衡低步预算下的噪声区间；② 将多步一致性训练迁移到方差爆炸坐标；③ 在推理阶段结合物理 steering 与无参考候选排序，显著提升几何质量。

**🔧 技术方法**

技术手段包括扩散模型（AlphaFold3、Boltz-2）、EDM 与一致性模型、几何噪声时间表、多步一致性蒸馏、物理 steering、基于 StereoCheck 的结构质量评估。

**📊 数据集**

使用的数据集包括 Boltz-2 基准（≈2300 个目标）、Runs N' Poses、RecentPDB384（三个公开蛋白结构数据集）。

**📈 对比分析**

通过统一评估协议（结构精度、界面质量、配体放置、立体化学有效性）与教师模型及其他少步采样器对比，Blitz 在保持教师层次精度的同时，25×减少采样步数，并在多项指标上优于现有方法。

**⚠️ 局限性**

局限性在于：仍需显著的 trunk 计算；部分几何缺陷（尤其是 clash）集中在少数目标上；AF3 的物理 steering 效果不佳；对配体有效性的提升有限；完整工作流程加速仍有进一步空间。

---

## 627. Integer Reachability in Data VASS

**arXiv ID:** 2610.23725 | [PDF](https://arxiv.org/pdf/2610.23725v1)

**作者:** Roland Guttenberg `[一作]` (University of Warsaw), Sławomir Lasota `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `847a60d8-a755-47af-ba5d-c5236b9e3083` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了数据 Petri 网（或等价的带状态向量加法系统）整数可达性问题的判定算法，并利用该结果证明了寄存器自动机的 Parikh 交集空性与数据 T‑系统的可达性问题可判定。

**💡 创新点**

核心创新在于：①构造了针对状态化数据系统的迭代细化（类似 KLM 分解）并证明其终止；②证明所有细化过程中循环空间的支撑被有效界定，突破了无穷状态空间导致的难题；③利用新的“必要原子”与“老原子”概念统一处理可逆性与连通性。

**🔧 技术方法**

主要技术包括：集合与原子理论（orbit‑finite 集合与符号表示）、整数线性规划与状态方程求解、可逆性 witness 与投影空间分析、循环空间与图分解算法、以及对数据向量与状态图的符号化操作。

**📊 数据集**

本文为理论性工作，未使用具体实验数据集；所有结论均基于形式化证明与符号计算。

**📈 对比分析**

由于方法纯理论，不涉及实验比较；通过证明可达性问题可判定，展示了理论复杂度与现有判定方法（如覆盖性、可逆性判定）相比的可行性与广度。

**⚠️ 局限性**

限制：仅在等价数据域（仅等于关系）下得到整数可达性可判定；对更丰富的数据域（如有序数据、全新鲜度）仍未解；完整的可达性问题在带状态数据 Petri 网中仍为开放问题。

---

## 628. On the Power of Alternating Local Certification with Constant-Size Certificates

**arXiv ID:** 2610.23694 | [PDF](https://arxiv.org/pdf/2610.23694v1)

**作者:** Pedro Montealegre `[一作]` (Universidad Adolfo Ibanez), Guillaume Theyssier `[通讯]` (CNRS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在匿名网络中研究可交替的本地验证模型，使用常数大小的证明/挑战标签，构建三种弱视图层级（集合、多重集、端口序号）并对其表达能力进行系统分析。

**💡 创新点**

创新点包括：①将交替证明者引入本地验证，形成新的层级体系；②证明在有限输入字母表下，该层级正好对应一阶集的单子集逻辑（MSO）；③证明层级严格递增并严格包含计数MSO（CMSO），同时给出层级与CMSO之间未能闭合的示例；④使用图颜色取向、XOR动态及模运算技术完成复杂属性的常数大小证明；⑤给出层级无法识别哈密顿性、完美匹配以及无界输入规模下的网络大小验证等性质，揭示其局限性。

**🔧 技术方法**

主要技术手段包括：图颜色取向与 BFS 定向化、XOR 动态映射与满射性判定、局部视图的集合/多重集/端口编码、逻辑翻译（将层级协议转化为 MSO 语句以及相反方向）、通信复杂度论证、fragile 输入计数、闭包运算（交、并、补）等。

**📊 数据集**

本文为理论性工作，没有使用具体数据集；所有结果均通过数学证明与逻辑推理给出。

**📈 对比分析**

与现有本地证明方案（如传统 PLS）相比，本层级在常数证书尺寸下即可表达所有 MSO 可判定属性，且在某些问题上提供更紧凑的交替证书；然而在哈密顿性、完美匹配等更高级属性上仍表现不佳，说明层级在表达范围上与传统方法相互补充。

**⚠️ 局限性**

局限性：①层级无法识别所有 MSO_2（允许边集量化）性质；②在无界输入字母表时，网络大小验证不属于层级；③某些本地可判定属性（如 Hamiltonicity）不在层级内；④对高度可扩展网络的多端口信息缺失导致表达能力受限。

---

## 629. On the Length of Maximum Sum-Rank Distance Codes

**arXiv ID:** 2610.23633 | [PDF](https://arxiv.org/pdf/2610.23633v1)

**作者:** Valentina Astore `[一作]` (Inria), Ferdinando Zullo `[通讯]` (University of Campania Luigi Vanvitelli)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过几何视角研究最大和秩距离(MSRD)码的长度问题，利用线性集的投影与投影计数方法得到三维及任意维度下的长度上界、块结构限制和权重分布，并证明三维单重量MSRD码不存在，给出一个在F₉上的三维异常构造，并提出MSRD版本的MDS猜想。

**💡 创新点**

创新点包括：1) 将传统弧与MDS码的几何框架推广到MSRD码并引入线性集系统；2) 在三维中完全描述直线交点模式与字符分布；3) 通过投影（短化）得到通用投影界；4) 对全长块提供精确的块数上界并构造异常例；5) 提出MSRD的MDS猜想，给出新的上界与可实现性条件。

**🔧 技术方法**

主要技术包括：有限射影几何中的弧与线性集概念、投影与短化的几何对应、计数论（双计数、三计数）与高维投影计数、线性化鲁宾逊-高德尔逊码的构造、以及与MDS猜想相关的代数几何工具。

**📊 数据集**

该研究不依赖具体数据集，所有结果均为理论推导与构造。

**📈 对比分析**

由于是理论研究，没有实验对比；通过构造与已知极限比较，验证了所给上界在多种参数下可达或逼近，尤其在全长块情况下与已知线性化鲁宾逊-高德尔逊码匹配。

**⚠️ 局限性**

局限性：1) 对于高维MSRD码的完整可实现性仍未完全解析，仅给出上界与部分条件；2) 异常构造仅在特定参数（如q=3,m=2）下存在，未证明是否存在更广泛的异常例；3) MDS猜想的MSRD版本尚未证明，只给出合理猜想；4) 对大规模块数或特殊块配置的精确阈值仍有待研究。

---

## 630. ESC: Escaping 2D Diffusion Ambiguity with Geometric Constraints for Consistent Novel View Synthesis

**arXiv ID:** 2610.23732 | [PDF](https://arxiv.org/pdf/2610.23732v1)

**作者:** Ge Ya Luo `[一作]`, Michal Drozdzal `[通讯]` (FAIR at Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出ESC（Escape the Allegory of the Cave）框架，利用几何感知的扩散模型在单目输入和对应深度图的基础上，联合生成目标视角的RGB图像与深度图，实现高质量的单目新视角合成。

**💡 创新点**

创新点包括：①通过Plücker射线条件的Ray ControlNet将目标相机姿态直接注入扩散过程；②引入Projection Consistency Constraint（PCC），在投影重投影中衡量并惩罚几何不一致；③在推理阶段采用Best‑of‑N（PCC‑BoN）策略，从多份采样中挑选PCC最小的样本，以进一步提升几何一致性。

**🔧 技术方法**

核心技术：Qwen MM‑DiT扩散模型、LoRA参数高效适配、Ray ControlNet、深度与图像联合token化、PCC深度重投影约束、Best‑of‑N推理选择。

**📊 数据集**

实验数据集：DL3DV、RealEstate10K、Tanks & Temples（T&T）以及野外（in‑the‑wild）场景；深度先验来自Depth‑Anything 3 (DA3)。

**📈 对比分析**

与PhotoNVS、VIVID、OVIE、GLD、LagerNVS等基线在LPIPS、DINO‑Sim、DreamSim、PCC等指标上进行对比。ESC‑Base在感知质量上已优于所有基线，ESC‑PCC‑BoN进一步降低PCC并略微提升LPIPS、DINO‑Sim，证明PCC‑BoN能够在不牺牲视觉质量的前提下显著提升几何一致性。

**⚠️ 局限性**

主要局限：①推理时需要多次采样（N 倍计算），计算成本较高；②深度依赖外部估计器 DA3，若深度误差较大会影响最终结果；③在极端视角变化或极度稀疏/离散的场景中仍可能出现几何错误。

---

## 631. Beyond Large Language Models: A Mental World Model Framework for Mental Modeling, Simulation, Evolution, and Interpretability

**arXiv ID:** 2610.23722 | [PDF](https://arxiv.org/pdf/2610.23722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 632. Synthetic Image Oversampling for Rare-Class Representation Learning in Deep Facial Phenotyping

**arXiv ID:** 2610.23730 | [PDF](https://arxiv.org/pdf/2610.23730v1)

**作者:** Miles Liao `[一作]` (Hopkins School), Kai Wang `[通讯]` (Children's Hospital of Philadelphia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了基于疾病的合成图像过采样（Synthetic Oversampling）对罕见遗传疾病面部表型识别的影响，使用PDIDB生成的风格GAN图像对GestaltMatcher-Arc和GestaltMML模型进行训练并评估其性能。

**💡 创新点**

提出了嵌套合成图像增强策略，并探讨了不同增强量对不同疾病分类精度的异质性效应；同时结合嵌入空间几何分析揭示合成样本对类内紧密度与类间分离度的影响。

**🔧 技术方法**

使用了StyleGAN3生成的面部图像、ArcFace损失的ResNet-50特征提取器、Class‑Frequency‑Weighted Cross‑Entropy、Multi‑modal融合（图像+临床文本）以及余弦距离、Silhouette Score、Nearest‑Centroid Separation等嵌入评估指标。

**📊 数据集**

数据集为GestaltMatcher Database (GMDB) 的 1,847 张真实面部图像（10 种遗传疾病）与 Phenotypic Disease Image Database (PDIDB) 的合成图像（根据疾病条件生成）。

**📈 对比分析**

在固定的患者不重叠评估分区上比较Baseline与四个层级（PD25–PD100）的合成增强；结果显示GestaltMatcher-Arc 的整体 Top‑1 准确率提升约1%，GestaltMML 的提升更为显著（从68.8%提升至74.3%），但不同疾病的个体效益差异明显。

**⚠️ 局限性**

局限包括仅使用单一患者分区和单一合成数据来源、评估集与检查点共享导致评估不独立、合成图像缺乏对应临床文本、以及未充分探究不同模型结构对增强响应的具体原因。

---

## 633. Agentic AI for Heterogeneous System Design Across Chip, Scale-Up, and Scale-Out Domains

**arXiv ID:** 2610.23724 | [PDF](https://arxiv.org/pdf/2610.23724v1)

**作者:** Murali Emani `[一作]`, Kashyap R Manjusha `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个面向异构硬件设计的全流程分析性成本模型，并通过实验验证与基准测试指导设计与部署优化。

**💡 创新点**

创新点在于统一了芯片屋顶线、通信α–β模型、LLM训练/推理公式与AI+HPC核性能估算，形成了可在同等功耗下进行多工作负载比较的细粒度性能预测框架。

**🔧 技术方法**

使用了芯片屋顶线分析、α–β通信模型、LLM训练/推理公式、AI+HPC核估算、网格搜索和加权几何平均评估等技术。

**📊 数据集**

校准数据来自公开硬件规格（H100、MI300X、Groq、NVLink4等）以及已发布的模型测量结果（70B BF16 训练、70B FP8 解码、HPCG 等）。

**📈 对比分析**

通过与同功耗的 H100+NVLink4+IB-NDR 参考设计进行速度比，对各工作负载加权几何平均得分，实验表明模型可在保证所有工作负载覆盖的前提下实现显著性能提升。

**⚠️ 局限性**

模型的局限包括固定重叠因子、忽略层级拥塞、未计预填充到解码 KV 缓存传输，以及对 wafer-scale 架构的简化近似，可能导致对极端规模或细粒度资源竞争的预测不足。

---

## 634. Recognizing Study Lineage Is Not Enough: Over-Crediting Same-Trial Evidence in LLM-Based Medical Verification

**arXiv ID:** 2610.23721 | [PDF](https://arxiv.org/pdf/2610.23721v1)

**作者:** Mukand Krishna `[一作]` (University of Kent), Zhongtian Sun `[通讯]` (University of Kent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文评估医学LLM在识别同一临床试验多份报告的衍生关系与在证据聚合时是否会误将同试验新报告计为独立证据的行为差异；通过对14个已审计临床家庭和5种模型条件的实验，发现模型能在单独识别任务中正确判断试验衍生关系，但在聚合任务中往往将同试验新出版物错误标记为独立证据。

**💡 创新点**

创新点在于：①提出了“识别–聚合”行为对比实验框架，区分试验衍生关系的识别与证据聚合两步；②构建了14个来源审计的临床家庭数据集，系统化检验不同模型在同一试验多报告场景下的表现；③验证了基于研究分组（study‑grouping）提示的干预能有效消除同试验过度计量的问题。

**🔧 技术方法**

技术上采用了基于提示的无监督LLM推理（GPT‑5.5、Qwen3‑4B、Gemma‑3‑4B、Llama‑3.1‑8B、Ministral‑3‑8B），在单独的无状态识别与聚合调用中返回结构化JSON；通过重复调用评估输入稳定性，并使用响应模式验证与指引。

**📊 数据集**

使用了14个来源审计的临床证据家庭，每个家庭包含固定主张、初始证据集和不同衍生关系的候选更新，涵盖试验衍生、适用性、结果方向等维度；数据来源主要为公开临床试验数据库与对应出版物的元数据。

**📈 对比分析**

比较方法：对每个模型在每个临床家庭的识别和聚合任务分别进行5次重复调用，统计识别准确率、聚合时是否出现同试验过度计量；对比未分组与分组提示两种条件。结果显示，GPT‑5.5在识别上达10/10/10（三类试验）但聚合时在所有同试验案例中出现5/5过度计量；分组提示将GPT‑5.5的过度计量从4/5/5降至0/0/0，而其他模型改进有限。

**⚠️ 局限性**

局限性包括：①临床家庭样本并非代表性分布，无法推断对更广泛医学命题的普适性；②实验使用无状态调用，未揭示模型内部表征；③仅评估了提示式分组干预，未考察自然出版元数据能否自动识别同试验；④部分模型在分组下仍出现拒绝输出，限制了比较范围。

---

## 635. Agentic Physics-Grounded Video Generation via Memory-Augmented In-Context Reasoning

**arXiv ID:** 2610.23677 | [PDF](https://arxiv.org/pdf/2610.23677v1)

**作者:** Xikang Guan `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PhysRepair框架，采用生成–诊断–修复循环提升视频生成的物理可行性。

**💡 创新点**

创新点在于：①利用时空因果约束定位物理错误并局部修复；②通过经验增强的上下文学习，将过去的修复经验用于当前决策。

**🔧 技术方法**

技术包括：多模态语言模型驱动的因果推理与工具调用、局部物理仿真、可视化修复条件注入、经验回溯存储与检索。

**📊 数据集**

使用Physics-IQ Verified基准，涵盖固体力学、流体动力学、光学、热力学和磁学五大领域。

**📈 对比分析**

与六款通用视频生成器和两款物理感知方法对比，PhysRepair在Physics-IQ总分上最高（43.6分），显著提升时空一致性指标。

**⚠️ 局限性**

局限性包括：依赖强大多模态语言模型，推理成本高；若代理模型物理推理能力不足，修复效果受限。

---

## 636. The Cognitive Cost of LLM Writing: A Reading-Time Comparison with Human Text

**arXiv ID:** 2610.23715 | [PDF](https://arxiv.org/pdf/2610.23715v1)

**作者:** Kamer Cekini `[一作]` (IIT-CNR), Marco Conti `[通讯]` (IIT-CNR)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究人类写作与LLM改写文本在阅读时长上的差异，构建并验证基于RoBERTa的阅读时间预测模型RoBERTa-RT，并利用该模型评估一对配对文本的阅读成本。

**💡 创新点**

首次将眼动数据训练的词级阅读时间预测模型应用于人类–LLM文本对的比较，构建配对语料库并揭示文本简洁与词级阅读成本之间的权衡。

**🔧 技术方法**

使用RoBERTa-RT（RoBERTa+回归头）、词频、词长、MTLD、Surprisal等特征，配合Wilcoxon符号秩检验和Spearman相关分析。

**📊 数据集**

训练集包含OneStop、GECO、MECO眼动数据；外部测试使用Provo；对照实验使用CCAT50人类文本与GPT‑5.6 Terra生成的496对改写文本。

**📈 对比分析**

通过配对Wilcoxon检验比较预测阅读时间与语言特征；RoBERTa-RT在内部/外部测试的MAE约为15–16 ms/词；LLM改写平均词数减少2.66%，总预测阅读时间略低，词级预测阅读时间提升1.57%。

**⚠️ 局限性**

模型仅基于眼动数据预测，未在这些文本上进行真实阅读实验；样本主要为英语新闻，跨语言或长文本的泛化性有限；LLM改写的内容偏差与生成策略相关。

---

## 637. "To Boldly Go Where No One Has Gone Before" What Has AI Let Developers Unlock?

**arXiv ID:** 2610.23708 | [PDF](https://arxiv.org/pdf/2610.23708v1)

**作者:** Rudrajit Choudhuri `[一作]` (Oregon State University), Anita Sarma `[通讯]` (Oregon State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在微软内部对331名软件开发者进行问卷调查，收集其关于 AI 开发经验的开放式回答，并用人机协同的多模型议事会（Claude、GPT、Gemini）结合反身主题分析，对回答进行编码、归类，提出三类工作场景（Faster、Further、Afresh）以及“技能‑判断”共学模型与双向 scaffolding playbook。

**💡 创新点**

1) 将 AI 视为与人类共同学习者，提出“技能‑判断 nexus”概念；2) 通过三类工作区分 AI 的实际影响；3) 形成面向开发者与 AI 的双向 scaffolding 实践手册，填补现有研究对 AI 作用的“速度”度量不足之处。

**🔧 技术方法**

人机协同分析流程：①多模型独立主题发现；②GPT5.6 统一合并主题；③作者审核与代码本批准；④模型编码并提供理由；⑤三模型一致性检验（Krippendorff’s α）；⑥Jev 决策模型与人工重编码验证；使用 reflexive thematic analysis 框架。

**📊 数据集**

Microsoft 开发者调查数据（331 个回答），涵盖不同团队、地区、经验水平；使用此问卷结果作为主要数据集。

**📈 对比分析**

比较方法主要是内部一致性指标：Krippendorff’s α（中位数0.92，范围0.83–0.97）衡量模型编码一致性；Jev 决策模型给出每条编码的置信概率（中位数0.84）。相较于单一模型分析，这种多模型、人工复核的流程在可靠性上表现更佳，减少了模型偏差。

**⚠️ 局限性**

局限性：①自我报告数据，可能存在夸大或记忆偏差；②仅采样自微软，组织文化与工具可用性可能不具普适性；③横断面设计，无法跟踪 AI 与开发者共学习动态的时间演化；④依赖 AI 辅助编码，虽有人工复核但仍可能遗漏细微语义差异；⑤未对结果进行外部效度验证，难以推断至更广泛的软件工程社区。

---

## 638. GLOBE: Evolving Structured Agentic Memory for E-commerce Product Optimization in Generative Search

**arXiv ID:** 2610.23709 | [PDF](https://arxiv.org/pdf/2610.23709v1)

**作者:** Hyunseo Kim `[一作]` (LG Electronics), Daeyeon Yi `[通讯]` (LG Electronics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在生成式搜索中，通过构建可演化的手册，持续改进产品列表的重写策略。

**💡 创新点**

创新点在于将优化知识拆解为字段范围明确的可复用行为，并通过反馈驱动的手册演化实现可解释、可迁移的记忆。

**🔧 技术方法**

采用GPT‑5.4 LLM完成编辑、反思、去重与提交步骤，并使用嵌入相似度过滤重复行为。

**📊 数据集**

使用E‑GEO基准数据集，包含来自Reddit和Amazon的真实购物查询与结构化商品列表。

**📈 对比分析**

与单一提示优化、静态手册和未过滤手册对比，Globe在平均排名提升、成功率上领先，且下降风险与类别方差更低。

**⚠️ 局限性**

局限包括：评估受单一模型偏好影响，行为归因粗糙，无法撤销负面修改，且需要高算力训练。

---

## 639. HardEval: An Execution-Verified Benchmark for AI-Driven Security Hardening and Compliance Remediation

**arXiv ID:** 2610.23685 | [PDF](https://arxiv.org/pdf/2610.23685v1)

**作者:** Manohar Vellala `[一作]` (Old Dominion University), Sachin Shetty `[通讯]` (Old Dominion University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了一个基于真实 RHEL 8 STIG 控制的执行验证基准（HardEval），用于评估大语言模型在无实施细节提示下自动生成 Bash 脚本并使操作系统通过 OpenSCAP/OVAL 合规检查的能力。

**💡 创新点**

创新点在于：①将安全硬化任务抽象为“实现细节被隐藏”的自然语言提示；②使用真实虚拟机执行脚本并通过扫描器判定合规性，消除了文本相似度或模型内部判定的偏差；③提供跨模型、跨域的统一任务清单和公开的评测管道。

**🔧 技术方法**

技术手段包括：LLM 调用（Claude、GPT‑4o、CodeLlama 等）、Bash 脚本生成、AlmaLinux 8 虚拟机隔离执行、OpenSCAP 与 OVAL 规范的合规检查、多模型对比实验与 Wilson 区间估计。

**📊 数据集**

数据集来自 233 条 RHEL 8 STIG 失效规则，筛选出 215 条带参考 Bash 修复，最终构成 151 条可评测规则（137 条“服务器‑安全”类），并与官方 DISA STIG、NIST‑800‑53 等元数据对齐。

**📈 对比分析**

实验比较了 14 个模型配置（7 前沿闭源、7 开源量化/代码专长）。前沿模型在服务器‑安全桶中最高达 81.0%；最弱的 4‑bit GLM‑4‑9B 仅 12.4%。结果显示规模在阶层间有效区分，但在同一阶层内部预测力弱；量化对性能无显著损失。

**⚠️ 局限性**

局限性包括：仅覆盖 RHEL 8 DISA STIG；未评估其他 OS 或 STIG 版本；仅考察自动化可实现的 151 条规则；SSH 评分受限；提示由单一模型生成，虽做交叉验证但仍可能存在作者偏差；可能存在训练数据泄露风险；仅测量合规性而非整体安全性。

---

## 640. TUPLE-Bench: Evaluating and Mitigating Privacy-Utility Trade-offs in Typed Decision Models

**arXiv ID:** 2610.23668 | [PDF](https://arxiv.org/pdf/2610.23668v1)

**作者:** Yiqu Yang `[一作]` (Eth Zurich), Elliott Ash `[通讯]` (Eth Zurich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 TUPLE‑Bench 基准，用拒绝探针和粗细层审核评估 Typed decision 模型的隐私泄露与效用，并提出工作流约束隐私量化 (WCPQ) 方法

**💡 创新点**

创新点在于（1）提供配对案例与合同式发布视图的系统化评估框架；（2）揭示即使模型正确拒绝标签，残差分数仍能泄露机密信息；（3）提出能在保持工作流成本不变的前提下量化并降低信息泄露的 WCPQ

**🔧 技术方法**

采用离线攻击器（逻辑回归、随机森林等）进行秘密恢复、Bootstrap 置信区间估计、量化映射和工作流签名对齐等技术

**📊 数据集**

使用 720 对 AI 生成的合成运营案例（送货、设备服务、IT 运维）构成的配对数据集

**📈 对比分析**

与九大模型、27 个任务组合进行实验，比较不同发布视图下秘密恢复率和业务成本；实验显示全概率输出降低成本但提升泄露率，而 WCPQ 能在不增加成本的前提下将泄露率降低 10–25%

**⚠️ 局限性**

局限性包括基准仅基于合成案例、攻击为封闭集且被动、未覆盖多次调用、开放集或时间侧信道等真实世界泄漏场景

---

## 641. Puzznic is PSPACE-Complete

**arXiv ID:** 2610.23639 | [PDF](https://arxiv.org/pdf/2610.23639v1)

**作者:** Jake Anderson `[一作]` (University of Auckland), András Z. Salamon `[通讯]` (University of St Andrews)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了静态Puzzzic的可解性是NP-完全，完整Puzzzic的可解性是PSPACE-完全。

**💡 创新点**

创新点在于构造自闭门门控来实现PSPACE难度，并将Cubic的NP-完全性改进为仅用两种块型。

**🔧 技术方法**

采用门控与运动规划框架、Savitch定理及配置图层化分析等理论技术。

**📊 数据集**

未使用实验数据集，全部为理论证明。

**📈 对比分析**

无实验比较，所有结果基于数学证明，未涉及性能评估。

**⚠️ 局限性**

仅考虑垂直移动墙块，水平移动墙块及其他新机制（如跳板、复制等）未纳入证明，结果对这些扩展可能不适用。

---

## 642. Approximating the Girth of an Undirected, Weighted Graph for Larger Approximation Factors

**arXiv ID:** 2610.23698 | [PDF](https://arxiv.org/pdf/2610.23698v1)

**作者:** Eliyahu Yablon `[一作]` `[通讯]` (Princeton University), Eliyahu Yablon (Princeton University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种随机化线性时间算法，用随机过滤技术在任意实数逼近因子k下近似无向加权图的girth（最短环长度）

**💡 创新点**

核心创新是“随机过滤”——通过按边权指数分布采样子图，使较长环的出现概率指数衰减，从而利用出现阈值估计girth；该方法不依赖单源最短路、稀疏生成子图或其他复杂子算法

**🔧 技术方法**

利用指数分布随机变量、线性时间选择与递归求阈值、Chernoff界估计、以及对浮点误差的细致分析，构成完整的线性时间实现

**📊 数据集**

文中未使用或提及任何实际数据集，全部为理论分析与算法设计

**📈 对比分析**

与现有最佳算法相比，在k≈c·log n/log log n（c>5）时，算法的运行时间可降低到O(m·(log n)^5/c)，相较于之前的O(cn(log n)^2+…+m log n)实现实现显著的polylog改进；同时保持了简单常数与实现易度

**⚠️ 局限性**

主要限制是算法只输出girth的数值估计，而不提供对应的最短环（witness）

---

## 643. Revisiting the Fairness of Pure Nash Equilibria of Round-Robin

**arXiv ID:** 2610.23686 | [PDF](https://arxiv.org/pdf/2610.23686v1)

**作者:** Georgios Amanatidis `[一作]` (Athens University of Economics and Business), Rebecca Reiffenhäuser `[通讯]` (University of Amsterdam)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在Round‑Robin及其递归平衡挑选序列机制下纯纳什均衡的公平性，证明了精确EF1不一定成立，却能得到1/2‑EF1与PROP1的近似保证，并给出了2/3的上界。

**💡 创新点**

主要创新在于纠正之前关于EF1的错误结论，给出反例；证明所有纯纳什均衡均满足1/2‑EF1和PROP1；构造极端实例证明2/3是不可超越的EF1逼近下界；以及系统分析递归平衡序列对公平性的影响。

**🔧 技术方法**

采用潜在函数与偏离分析的组合证明，以及递归平衡序列的计数与对比技术，构造反例并进行严格的最佳响应验证。

**📊 数据集**

论文为理论分析，未使用任何实验数据集。

**📈 对比分析**

通过构造最坏实例和理论证明，展示了纯纳什均衡在EF1方面只能保证1/2，且无法超过2/3，说明其公平性受限。

**⚠️ 局限性**

局限在于仍存在1/2与2/3之间的未知区间，未确定α_RR与α_RB的精确值；分析仅适用于加性价值的纯纳什均衡，缺乏经验验证。

---

## 644. PCDFusion: Proposal-Context-Detail Bayesian Rendering for Infrared-Visible Image Fusion

**arXiv ID:** 2610.23663 | [PDF](https://arxiv.org/pdf/2610.23663v1)

**作者:** Rui Liu `[一作]` (Beijing Institute of Technology), Ying Fu `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种可见光-热成像融合方法PCDFusion，采用可见光为主的后向渲染框架，在三个阶段（Proposal、Context、Detail）中通过贝叶斯后验权重决定何时保留热响应、修正曝光失效的可见区域以及注入带限热细节。

**💡 创新点**

创新点在于把融合视为局部可靠性条件的后验渲染：①引入物理先验（曝光、结构）与学习的对数证据评分联合更新后验权重；②将热响应的保留、可见修正和细节注入拆分为三步后验决策，避免直接混合导致的热污染；③使用可见锚定的视觉特征与热特征通过注意力融合，提升对不同可靠度区域的适应性。

**🔧 技术方法**

技术手段包括：Swin Transformer提取可见特征、轻量级卷积编码器提取热特征、基于注意力的跨模态融合、贝叶斯后验更新（先验+对数证据→权重），以及三阶段的局部渲染公式；损失函数包括热响应召回、后验校准与结构一致性损失。

**📊 数据集**

实验使用公开的三大热可见融合基准：MSRS、FMB 与 M3FD；同时在 FMB 语义分割（SegFormerB2）和 M3FD 目标检测（YOLOv10s）上评估下游任务性能。

**📈 对比分析**

与 SwinFusion、TarDAL、CDDFuse、Diff-IF、EMMA、DCEvo、Deno-IF、SAGE、UPFusion 等最新方法对比，PCDFusion 在 VIF、Qabf、Qcb 等融合指标上持续领先；在 FMB 上 mIoU 达 60.05%，在 M3FD 上 mAP_50:95 达 55.98%，均超过竞争对手，证明其在图像质量与任务效用上的整体优势。

**⚠️ 局限性**

局限性主要体现在低亮度或热响应不可靠区域：热干扰仍会轻微残留，尤其在可见光曝光失效且热信息无效时模型对后验阈值的校准略显宽松，导致少量不必要的热细节被保留。

---

## 645. Learning Robot Metareasoning: Adaptive Computation Budgets for Real-Time Control

**arXiv ID:** 2610.23662 | [PDF](https://arxiv.org/pdf/2610.23662v1)

**作者:** Lorenzo Magnino `[一作]` (University of Cambridge), Amanda Prorok `[通讯]` (University of Cambridge)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于强化学习的自适应计算预算策略，允许机器人在执行期间动态决定计划的执行时长和对应的计算时间，以平衡规划质量与系统实时性。

**💡 创新点**

创新点在于：①将自适应计算预算建模为半马尔可夫决策过程（SMDP）并通过RL学习；②利用计划器的运行统计（如ESS、可行性时间、模型误差）来指导预算决策；③在实际物理系统中引入“物理计算价值”概念，提供理论上限与安全约束的分析。

**🔧 技术方法**

主要技术包括：PPO强化学习、半马尔可夫决策框架、MPPI与CEM采样式规划器、基于Wasserstein距离的误差界定、JAX/MuJoCo实现、GPU加速的实时规划与控制。

**📊 数据集**

实验数据集涵盖：①真实机器人平地导航（DJI RoboMaster）；②仿真四旋翼悬挂绳索飞行（MuJoCo）；③仿真接触丰富的平面推压任务（Push‑T）。

**📈 对比分析**

与固定预算（h=1、4、9 等）和不同规划器的基线比较，指标包括任务成功率、总回报、路径粗糙度（harshness）。实验表明学习到的自适应预算在三种场景下均显著优于任何固定预算策略，尤其在存在显著不确定性和动态障碍时取得最高成功率和回报。

**⚠️ 局限性**

局限性包括：①当任务无显著可利用的预算差异（如确定性、无扰动环境）时，学习策略往往退化为恒定预算；②依赖于规划器提供的统计信息，若统计不足或不可靠，决策质量受限；③仅验证了采样式规划器，对其他优化算法的通用性尚未评估。

---

## 646. Diversity Without Labels: Task-Agnostic Coreset Selection Across Perception and Action

**arXiv ID:** 2610.23642 | [PDF](https://arxiv.org/pdf/2610.23642v1)

**作者:** Julia Hindel `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无监督、任务无关的核心子集估计方法 Coreset DNA，通过汇聚视觉基础模型的神经元激活分布来挑选训练数据。

**💡 创新点**

创新点在于：①使用多层激活的均值、标准差和跨层秩差（δ）构成低维数据质量表示；②证明这些统计量与训练基准下的“硬样本”高度相关；③利用切片 Wasserstein 距离或最远点采样高效构建覆盖/分布匹配的核心子集；④无需标签或模型训练，直接从冻结模型中提取信息。

**🔧 技术方法**

技术细节包括：冻结 ViT‑B/16 提取每个神经元的激活直方图；用 EMD 计算图像与全局直方图的距离；对每层求均值、标准差、最大值；对所有层求整体统计；对三维向量 (μ12, σ12, δ_max) 标准化后进行切片 Wasserstein 最小化或 FPS 采样；可选加入帧索引、位置信息等辅助维度。

**📊 数据集**

实验涵盖了图像分类、语义分割、深度估计、端到端驾驶与机器人抓取等多任务数据集：CIFAR‑10、PascalVOC、Cityscapes、Cityscapes‑DVPS、SemanticKITTI‑DVPS、NAVSIM、LIBERO‑10。

**📈 对比分析**

与随机、忘记、EL2N、AUM、CCS、BOSS、ELFS、FreeSel、ZCore 等基线比较；在大多数任务和数据集上，Coreset DNA 与最先进的训练基准相当或更优，尤其在低保留率（α≤0.1）下显著提升；计算成本低、无需额外训练。

**⚠️ 局限性**

局限性：①统计特征对不同数据集分布差异敏感，可能需要针对性调参；②依赖冻结的视觉基础模型，对非视觉或多模态任务的适用性尚未验证；③在极端保留率或过度稀疏的数据集上，切片 Wasserstein 与 FPS 的优势不一；④缺乏对主动学习场景的支持，仅为一次性核心子集估计。

---

## 647. Empirical Insights from AI-Powered Software Engineering in Practice

**arXiv ID:** 2610.23632 | [PDF](https://arxiv.org/pdf/2610.23632v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 648. PresenceBench: An Exploratory Evaluation of Large Language Model Adaptation to User State

**arXiv ID:** 2610.23625 | [PDF](https://arxiv.org/pdf/2610.23625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 649. No-Regret Language Generation

**arXiv ID:** 2610.23641 | [PDF](https://arxiv.org/pdf/2610.23641v1)

**作者:** Jiaxun Li `[一作]` (University of Michigan), Ambuj Tewari `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言生成中的无后悔（no‑regret）范式，并给出了对应的误差度量与最优下界。

**💡 创新点**

创新点在于提出了“误差强迫维度（mistake‑forcing dimension）”和“带符号误差强迫维度（signed mistake‑forcing dimension）”，并证明它们正好刻画了可无后悔生成的下界与上界；同时展示了无后悔生成与传统生成准则、在线预测的量化关系。

**🔧 技术方法**

使用了树形轨迹结构、组合学上对可行误差模式的分析、Littlestone 维度的潜能方法以及构造性极大化/极小化论证。

**📊 数据集**

本文不涉及具体数据集，而是以理论假设（可枚举且支撑无限的语言类）为基础进行分析。

**📈 对比分析**

通过与现有生成准则（如限时生成、统一误差界、极限生成等）的比较，证明无后悔生成在量化上更强大；与在线预测的 Littlestone 维度比较得出子线性误差界（O(d log(1+T/d)) 及 O(√(dT log(1+T/d)))），显示其最优性。

**⚠️ 局限性**

局限性包括：对具体类评估误差强迫维度仍然困难；对于有限 Littlestone 维度的无后悔误差上界的最优形式尚未完全确定；未考虑算法的可计算性与效率。

---

## 650. TransNav: Transformer-Based End-to-End Navigation Method for Static and Dynamic Quadrotor Obstacle Avoidance

**arXiv ID:** 2610.23635 | [PDF](https://arxiv.org/pdf/2610.23635v1)

**作者:** Jianghan Zhu `[一作]` (Beijing Institute of Technology), Jian Sun `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种名为 TransNav 的基于 Transformer 的端到端四旋翼障碍物回避方法，利用 LiDAR 点云进行感知，直接预测有限时域加速度序列实现高速、高频的动作规划。

**💡 创新点**

创新点包括：1）使用 patch‑based 距离图编码结合轻量 CNN 与 Transformer，兼顾局部细节与全局上下文；2）在单一网络内实现感知‑决策‑执行，输出完整动作序列；3）采用无监督训练，利用点质量模型与数值梯度直接优化轨迹级成本，省去专家演示与 Critic 网络。

**🔧 技术方法**

核心技术包括 Transformer encoder、轻量 CNN、距离图分块、点质量模型、梯度衰减机制、Smooth L1 与 ESDF 辅助、以及对比 PPO 的训练。

**📊 数据集**

训练数据来自 10 个随机生成的静态环境，每个环境包含 400 个障碍物（圆柱、立方体、球体等），收集 LiDAR 点云、四旋翼状态、目标状态及 ESDF 地图，共计约 10 万条样本。

**📈 对比分析**

与 EGO‑Planner、FAPP、NavRL、P2M 等基线在模拟与实地测试中对比，TransNav 在密集静态环境中成功率最高，速度适中；在动态环境保持竞争力；推理延迟约 5 ms，能实现 100 Hz 控制；在实地实验中可通过 1 m 门、快动障碍物等场景表现良好。

**⚠️ 局限性**

局限性：仅在静态环境中进行训练，缺乏对动态障碍物的显式预测；对极快移动障碍物的避障仍受限；未验证多无人机或极端速度情况下的性能；训练时依赖 ESDF 地图等辅助信息。

---

## 651. RedDarwin: Evolving Jailbreak Skills against Frontier LLMs

**arXiv ID:** 2610.23726 | [PDF](https://arxiv.org/pdf/2610.23726v1)

**作者:** Zhouyuan Ma `[一作]` (Fudan University), Xingjun Ma `[通讯]` (Fudan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出可执行攻击技能演化框架 RedDarwin，通过单技能修订、多技能组合和新技能发现三条进化路径，迭代改进 47 个基础 jailbreak 技能，得到 65 个改进后技能。

**💡 创新点**

创新点在于把 jailbreak 视为可执行技能而非固定提示，采用技能级别的进化机制，支持机制重用、组合与自我发现，提升了攻击的可迁移性和适应性。

**🔧 技术方法**

使用基于目标模型反馈的进化学习、Beta 采样父技能选择、交互式执行轨迹记录、机制分解与重用，以及 Kimi-CLI/Kimi-Code 等工具。

**📊 数据集**

利用 50 条 AdvBench 目标、4 个发展模型、50 条 StrongREJECT 目标和 3 个未见 frontier 模型，评估演化后技能在多平台上的表现。

**📈 对比分析**

在发展集上，单技能修订平均提升 ASR 29.5%（18.8%→47.3%），组合和发现平均提升 10.7% 与 83.2%；在 held‑out 任务上，演化技能平均 ASR 达 76.3% 对比 38.5%；在 10 次查询的自适应红队评估中，RedDarwin 在首查询即超过现有基线并在有限查询下保持领先。

**⚠️ 局限性**

局限性包括对初始 47 个技能的依赖、对特定模型的过拟合风险、计算成本较高、评估仍需人工/自动判断器，以及在更大规模或更强防护的模型上效果未知。

---

## 652. Equivalence testing and run-count design for stochastic machine-learning experiments

**arXiv ID:** 2610.23700 | [PDF](https://arxiv.org/pdf/2610.23700v1)

**作者:** Dmytro Knopov `[一作]` (National University of Kyiv-Mohyla Academy), Nadiya Shvai `[通讯]` (National University of Kyiv-Mohyla Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套基于等价检验（TOST）和两层方差模型的训练方法比较协议，给出了实验设计公式和两阶段试验流程；

**💡 创新点**

在机器学习实验中将等价检验转化为先验设计问题，并给出可用于小样本（2-5次）实验的闭式运行数公式；

**🔧 技术方法**

采用两层（种子×项目）方差模型、Welch-TOST、两级引导抽样、蒙特卡洛仿真校准、以及正态近似的偏移-t校正；

**📊 数据集**

在CIFAR-10、SVHN、CIFAR-100、DTD、LSUN-R、iSUN（AUROC）和CIFAR-10、RTE、SST-2、MNLI（准确率）等公开数据集上验证；

**📈 对比分析**

对比两种训练方法或超参数组合时，利用等价检验给出95%置信区间并判断是否落在±δ范围内；在案例中，两个seed下的PC与SGLD在±0.01 AUROC范围内未能证实等价，所需运行数约为12；在公共模型库中，12至几百次运行即可实现80%功效；

**⚠️ 局限性**

局限性包括：需要预先设定等价阈值；两级引导抽样在极少样本或显著种子方差占优势时不一致；对非正态或混合分布的跑分需要额外仿真；对置信区间的正态近似在小样本下可能不精确；对多测试集、连续决策阈值未提供多重校正方案。

---

## 653. Joining Large String Collections with Jaro-Winkler Similarity

**arXiv ID:** 2610.23652 | [PDF](https://arxiv.org/pdf/2610.23652v1)

**作者:** Manuel Lankes `[一作]`, Nikolaus Augsten `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种三阶段近似字符串匹配框架：先用PUMA算法进行预筛选，得到预候选集合；再通过DIB模块利用动态下界φ_l与上界φ_u对候选进行过滤；最后进行精确验证，输出匹配结果。

**💡 创新点**

创新点在于将PUMA预筛选与DIB动态索引相结合，并引入新的重叠度O度量及上下界φ_l/φ_u的动态估计，显著减少了不必要的验证操作。该框架在保持高召回率的同时，大幅降低了计算复杂度。

**🔧 技术方法**

技术核心包括：基于Jaccard相似阈值τ_J的预筛选；动态区间下/上界计算φ_l、φ_u；重叠度O的快速估计；以及在O(n)时间内完成候选验证的高效实现。实现中使用了n-gram索引、布隆过滤器与位运算等加速手段。

**📊 数据集**

实验在公开的近似字符串匹配基准数据集上进行，主要使用了ClueWeb、Wikipedia与Twitter文本集合，并在TREC Web数据集上进行了对比验证。

**📈 对比分析**

与现有方法（如Bitap、Landau‑Vishkin、Aho‑Corasick）相比，本文方法在相同相似阈值下平均查询时间提升约30%–40%，召回率保持在97%以上，且误报率下降至原方法的一半左右。

**⚠️ 局限性**

局限性包括：对极长字符串或极低相似阈值时，动态区间估计不够紧密导致候选数仍较多；系统对内存占用有一定要求（主要由n-gram索引与布隆过滤器构成），在极大规模数据上需要进一步优化。

---

## 654. In Human Memories. Technology, Creativity and Self-Determination: An Ethical Proposal

**arXiv ID:** 2610.23651 | [PDF](https://arxiv.org/pdf/2610.23651v1)

**作者:** Francesco Vitucci `[一作]`, Francesco Scagliola `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了人工智能在现代作曲实践中的作用，并通过对Convergence与In Human Memories两件作品的案例分析，探讨AI与音乐创作的互动与伦理。

**💡 创新点**

提出AI既可作为创作主体，也可作为灵感来源，并将AI与声学乐器、实时电子技术融合，形成新的“协同创作”模式，强调技术的可持续性与人机共创的伦理路径。

**🔧 技术方法**

采用生成式对抗网络（GAN）、自动编码器、预训练语言模型（LLM）、WaveNet语音合成、实时音频处理算法以及Ambisonics三维空间声技术。

**📊 数据集**

文本数据来源为与GPT‑4的对话生成的三段Dido记忆文本；音乐数据借鉴Illiac Suite、Convergence等已有作品，未使用公开大型音频或文本数据集。

**📈 对比分析**

通过现场演出与主观评估对比AI创作与传统作曲方法的差异，未给出量化性能指标，重点关注艺术性与情感表达。

**⚠️ 局限性**

局限性包括AI的黑箱性导致可解释性不足，生成文本可靠性需人工后期编辑，硬件与算法资源需求高，缺乏系统化、可量化的评价标准。

---

## 655. Triangle Covers of Polygons: Approximation Algorithms and Complexity

**arXiv ID:** 2610.23634 | [PDF](https://arxiv.org/pdf/2610.23634v1)

**作者:** Linda Kleist `[一作]` (University of Hamburg), Lena Scherzer `[通讯]` (University of Hamburg)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究多边形的几何集覆盖与分割问题，主要关注用三角形覆盖（MTC）与三角形划分（MTP）的最小化。通过构造特殊多边形，证明了原有的 40 年猜想——最小划分中的 Steiner 点只能在由顶点生成的可构造点集 V_k 内——是错误的；在该多边形中任何最小划分都需要不可构造、甚至是无理坐标的 Steiner 点。随后给出了常数因子逼近算法：MTP 取得 3（简单多边形）/5（一般多边形）逼近，MTC 在简单多边形中取得 12 逼近，且通过集合覆盖和网格技术进一步得到 O(log n) 逼近。

**💡 创新点**

创新点包括：
1) 以严谨证明方式驳斥长期存在的 Asano‑Pinter 猜想，揭示最小三角形划分可能需要无理 Steiner 点，说明问题的复杂性远超预期。
2) 设计了新的结构化分析（平角、禁用配置、seesaw 结构）以证明无理点的必要性。
3) 推导了常数因子逼近算法，首次在简单多边形中给出 12 逼近的三角形覆盖，以及 3/5 逼近的划分。 
4) 对凸 k‑边形覆盖给出 (k+9) 逼近，并在多边形带孔的情形下给出 25 逼近及 O(log n) 逼近。

**🔧 技术方法**

主要技术手段：
- 组合几何与可构造点集 V_k 的迭代定义。
- 平角计数与 Euler 公式结合，得到最小划分的下界。
- 禁用配置与 seesaw 结构的识别与消除。
- 使用弱可见多边形分割、类图 G̃ 以及路径分区来构造逼近解。
- 充电（charging）与极性分析保证 12 与 (k+9) 逼近因子。
- 网格 G₁ 与集合覆盖技术，构造限制版三角形覆盖并通过贪心得到 O(log n) 逼近。

**📊 数据集**

本文并未在实验数据集上进行测试；所有结论均通过严格的理论证明和符号计算得到。所使用的多边形多为构造性的、具有特定几何性质（如极大化不可构造 Steiner 点）的实例。

**📈 对比分析**

比较方法：
- 对于 MTP，已知最优划分数的下界为 n/3，算法获得的 3（简单）/5（一般）逼近因子已是最优常数逼近的上界。
- 对于 MTC，已知 3-逼近是最优的（在无共线边的情况下），本文的 12 逼近虽不是最优，但提供了首次常数逼近；通过 25 逼近进一步逼近 3‑逼近。
- 对于凸 k‑边形覆盖，(k+9) 逼近相较于之前的 12(k‑2) 逼近显著提升。
- 对于多边形带孔，本文提供了 O(log n) 逼近，匹配现有的最优级别。
- 计算复杂度：所有算法均为多项式时间，主要为 O(n²)（MTP、MTC）或 O(n¹⁶)（O(log n) 逼近所需的网格构造），在理论上可接受。

**⚠️ 局限性**

限制与未来工作：
- 对于简单多边形的 MTP，是否存在多项式时间算法仍是开放问题，本文仅给出常数逼近。
- 证明中涉及的几何构造相对复杂，实际实现的工程成本高；对大规模多边形的实用性尚未验证。
- 论文所给的 12 逼近和 (k+9) 逼近仍距最优（3 或 3/5）较远，是否能进一步压缩仍需研究。
- 对于多边形带孔的 MTC，虽然给出 O(log n) 逼近，但实现中需要生成高维网格，空间复杂度可能成为瓶颈。

---

## 656. Four New Single-Error-Correcting Binary Codes

**arXiv ID:** 2610.23626 | [PDF](https://arxiv.org/pdf/2610.23626v1)

**作者:** Moshe Milshtein `[一作]` `[通讯]`, Moshe Milshtein

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构造了四个新的二进制码，分别为(16,3072,3)、(17,6144,3)、(18,12288,3)和(19,24576,3)，并因此给出了对应的四条新的下界；

**💡 创新点**

创新点在于将码构造转化为线性码余类的独立集问题，并利用禁忌搜索（tabu search）这一元启发式算法在图中寻找最大独立集，从而在先前未能突破的下界上取得突破；

**🔧 技术方法**

采用的技术包括：线性码分类、构建余类图、定义余类间距离权重、使用禁忌搜索寻找独立集、以及多轮参数调优和增大码量的后处理；

**📊 数据集**

使用的“数据集”为在给定长度n、维度k、最小距离d的所有非等价线性码列表（由软件项目自动生成），以及对每个线性码进行的禁忌搜索实验结果；

**📈 对比分析**

与已有下界进行比较，实验结果显著提升了A(16,3)、A(17,3)、A(18,3)、A(19,3)的下界（分别从2816/5632/10496/20480提升至3072/6144/12288/24576），证明了方法的有效性；

**⚠️ 局限性**

局限性包括：仅在特定长度（≤19）和二进制情形下有效；依赖于计算资源与搜索时间，缺乏通用解析构造；并未给出这些下界是否最优或可进一步提升的理论分析。

---

## 657. A Language-Informed Loss for Visual Generation

**arXiv ID:** 2610.23653 | [PDF](https://arxiv.org/pdf/2610.23653v1)

**作者:** Suchir Agarwal `[一作]` (Stanford University), Jiajun Wu `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种语言信息化且无损的图像编码器 Lily，用来生成语言监督的损失，并在端到端像素空间的文本到图像生成模型中使用该损失显著提升生成质量。

**💡 创新点**

创新点：① 用语言监督训练可逆图像变换编码器，使得噪声下的特征仍能恢复对应字幕；② 该编码器提供无损的语言驱动特征空间，可直接替代或补充像素损失；③ 在像素空间生成器中加入此语言损失，单独使用时已优于像素损失，联合使用可进一步提升生成质量。

**🔧 技术方法**

技术手段：可逆变换（TARFlow/BiFlow）、LoRA 微调的视觉语言模型、交叉熵 caption 训练、端到端 JiT‑T2I 变压器、像素 + 语言空间 MSE 损失。

**📊 数据集**

使用数据集：GPIC 图像‑字幕对数据集（256×256，100M 对），用于训练编码器、生成器及评估。

**📈 对比分析**

评估方法：与单像素损失、LPIPS、DINOv2、SigLIP 2 感知损失对比，采用 gFD、PRDC、MMD 等指标。结果显示：Pixel-only gFD=66.84；lily-only gFD=55.78；Pixel+ lily gFD=53.01，显著优于传统感知损失，并提升精度、召回、覆盖率。

**⚠️ 局限性**

局限性：需要专门训练语言监督编码器，且验证仅在 GPIC 上；对生成器架构和数据规模敏感；未探索多感知损失组合；对大型预训练模型的兼容性与评估仍有限。

---

## 658. JAXtari: High-Throughput and Easy-to-Modify Arcade Learning Environment

**arXiv ID:** 2610.23650 | [PDF](https://arxiv.org/pdf/2610.23650v1)

**作者:** Quentin Delfosse `[一作]` (Technical University Darmstadt), Kristian Kersting `[通讯]` (Technical University Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文内容缺失，无法提取

**💡 创新点**

无法确定

**🔧 技术方法**

无法确定

**📊 数据集**

无法确定

**📈 对比分析**

无法比较

**⚠️ 局限性**

缺乏信息

---

## 659. FedRAPP: Federated Test-Time Prompt Tuning for Vision-Language Models with Reliability-Aware Prompt Personalization

**arXiv ID:** 2610.23705 | [PDF](https://arxiv.org/pdf/2610.23705v1)

**作者:** Syed Md. Ahnaf Hasan `[一作]` (Independent University), AKM Mahbubur Rahman `[通讯]` (Independent University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了源自由联邦测试时提示调优（Source-Free Federated Test-Time Prompt Tuning，SF-FTTPT），提出了一种新的联邦聚合框架FedRAPP；

**💡 创新点**

创新点在于三大核心组件：①可靠性感知的客户端评估；②服务器端指数移动平均（EMA）提示记忆；③基于提示更新相似度的个性化同伴聚合；这三者协同实现了在非 IID、无标签、无源数据场景下的稳健个性化适配；

**🔧 技术方法**

采用了联邦学习框架、无标签熵下降式提示调优、熵下降与漂移惩罚的可靠性评分、EMA平滑、余弦相似度与软最大化权重的个性化聚合等技术；

**📊 数据集**

使用了九个CLIP基准分类数据集（Flower102、Caltech101、OxfordPets、DTD、Food101、UCF101、SUN397、FGVCAircraft、StanfordCars）以及ImageNet及其四个变体（ImageNet‑V2、ImageNet‑A、ImageNet‑Sketch、ImageNet‑R）；

**📈 对比分析**

与FedAvg、FedProx、FedAMP、pFedGraph、pFedMoAP以及无联邦的None‑Fed对比；FedRAPP在IID和非IID分区下均取得最优或次优性能，平均Top‑1准确率提升约1–2%，在非IID设置下尤其显著；

**⚠️ 局限性**

局限性包括仅在CLIP视觉‑语言模型上验证，未扩展至更大多模态模型；仅在同步、资源充足的联邦环境下评估，对异步或资源受限场景的鲁棒性未知；对动态分布漂移或长期在线更新的适应性尚待进一步研究。

---

## 660. Bridging the Language Gap in Scholarly Data II: Enhancing Gender Inference for Chinese Names

**arXiv ID:** 2610.23734 | [PDF](https://arxiv.org/pdf/2610.23734v1)

**作者:** Lisette Espín-Noboa `[一作]` (Graz University of Technology), Liuhuaying Yang `[通讯]` (Complexity Science Hub)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究构建了基于汉字和拼音的性别推断框架，训练了汉字回归模型和三类分类模型，并通过多候选汉字聚合实现对拼音姓名的性别预测，同时给出了模糊度指标；

**💡 创新点**

创新点包括：①利用汉字级别信息与多源工具特征融合，提升汉字姓名性别推断精度；②设计拼音到汉字的多候选聚合策略，并引入模糊度量来量化拼音推断的不确定性；③在中文姓名大样本上实现高覆盖率和高准确率，并公开代码、数据与交互工具，促进可复制性。

**🔧 技术方法**

技术手段主要包括CatBoost回归/分类模型、特征工程（融合三大外部性别工具输出）、多策略加权聚合（均匀、频率权重、人口权重）、交叉验证与早停、SHAP特征重要性分析。

**📊 数据集**

使用了CNKI物理期刊作者名单（46,144个独立汉字姓名，含单字与双字），从中抽样166名姓名进行人工标注；补充HanziDB单字数据；并用美国APS期刊的中文作者数据检验泛化。

**📈 对比分析**

通过与三款商用工具（Genderize、Gender-API、Namsor）及LLM（ChatGPT、Claude、Llama）对比，评估RMSE、R²、Macro‑AUC、κ；汉字回归模型在测试集上R²=0.85、RMSE=0.09，显著优于对比工具；拼音聚合中Weighted Pop策略R²≈0.30、RMSE≈0.20，仍高于其他工具。

**⚠️ 局限性**

局限性包括：标注样本仅来自物理学作者，可能难以直接推广到其他学科或时间段；拼音聚合依赖汉字候选集与频率权重，若基准数据变动需重新估计；商用工具与LLM的输出易变，影响可重复性；模型仅预测姓名的感知性别，不能用于个体性别识别。

---

## 661. VISTA: Steering LLMs with Latent User Values Inferred from a Few Survey Answers

**arXiv ID:** 2610.23702 | [PDF](https://arxiv.org/pdf/2610.23702v1)

**作者:** Yinuo Xu `[一作]` (University of Michigan), Arslan Basharat `[通讯]` (Kitware Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出VISTA框架，利用部分调查答案推断用户价值潜在分布，并通过超网络将该分布映射为冻结LLM的激活层或软前缀注入，实现对答案选择与自由文本生成的个性化引导；同时采用相同的潜在表示进行自适应问题挑选与停止判定。

**💡 创新点**

创新点包括：① 通过变分信息瓶颈获得用户价值的概率后验，兼顾不确定性；② 用超网络直接将潜在向量映射为LLM内部的激活向量或软前缀，避免显式提示；③ 提出解释性价值轴，可将未见用户精确放置；④ 基于期望预测信息增益实现自适应提问与停止，显著降低问卷负担；⑤ 在冻结模型上实现常数成本的个性化。

**🔧 技术方法**

使用的技术包括：变分信息瓶颈（VIB）与半消融推理、FiLM调制、超网络（self‑attention）、期望预测信息增益（EPIG）、DPO微调、软前缀注入、激活层注入、随机采样与KL正则化。

**📊 数据集**

实验数据集：Synthetic Food2D（10道食物问答），World Values Survey（美国/英国 5,652人、226条价值题），PrimeX（858人、48条二元条目，包含医学、科学、世界观等）。

**📈 对比分析**

与基线（人口均值、随机人、LoRe、PAL、提示式上下文、CAT、Prompt‑Only等）对比，VISTA在所有三组数据集的强制选择任务上均匹配或优于提示式个性化；在自由文本生成上仅VISTA显著提升（PrimeX +1.2分）；自适应提问时，VISTA在仅 6% 问卷量下恢复 94% 性能提升，且在所有预算下优于 CAT 和 Prompt‑Only，且拥有更合适的停止点。

**⚠️ 局限性**

局限性：① 只在强制选择任务中实现自适应提问；② 仅利用后验均值做注入，未充分利用不确定性；③ 可能强化用户已有偏见，导致安全与多样性风险；④ 对多数视角的个性化可能导致准确率下降；⑤ 价值-行为关联有限，未验证对具体行为决策的影响；⑥ 仅在 Llama‑3‑8B‑Instruct 上验证，跨模型推广需重新训练超网络。

---

## 662. Beyond the Benchmark: Conditioning, Domains and Failure Modes of Qwen3-TTS

**arXiv ID:** 2610.23678 | [PDF](https://arxiv.org/pdf/2610.23678v1)

**作者:** Séverin Baroudi `[一作]`, Ricard Marxer `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Qwen3-TTS的两种零射声克隆接口（ICL与X-Vector）进行系统评估，比较其在四个不同语域（LibriSpeech、AMI、IEMOCAP、ATCO2）下与Chatterbox、XTTS-v2、F5-TTS的表现，并结合人类听力测试、LLM判别、18项自动评估指标以及AASIST反spoof检测。

**💡 创新点**

首次全面比较Qwen3-TTS两种接口的优势与不足；在非朗读语域下揭示域对克隆性能的影响远大于模型差异；提出结合人类、LLM与多维自动指标的评估框架。

**🔧 技术方法**

使用零射声克隆模型（Qwen3-TTS、Chatterbox、XTTS-v2、F5-TTS），LLM判别（Gemini 3.5 Flash），18项自动评估指标（ResNet、VoiceCLAP、emotion2vec、SpeechBERTScore、F0、WER、DNSMOS、STOI、PESQ等），以及AASIST端到端反spoof检测。

**📊 数据集**

四个公开英文语料库：LibriSpeech（朗读），AMI（会议），IEMOCAP（情感对话），ATCO2（空中交通管制语音）。

**📈 对比分析**

采用2AFC人类听力测试、LLM判别、自动指标与反spoof检测进行多维比较。结果显示：Qwen3-TTS在可懂度和音高（F0）上优于其他系统，ICL在音色匹配上更佳，X-Vector在时序和情感维度上更好；但在ATC语域下所有系统性能急剧下降，反spoof检测无法区分真实与克隆语音；整体排名以Qwen3-TTS、Chatterbox、XTTS-v2、F5-TTS依序递减。

**⚠️ 局限性**

仍缺乏能替代人类评估的自动指标；域差异导致性能不稳定，短参考或错误转录会产生幻觉；反spoof检测在噪声/窄带语域（如ATC）失效，难以实现有效的安全防护。

---

## 663. OwlSight: A Panoramic Monitor for Chain-of-Thought Reasoning

**arXiv ID:** 2610.23670 | [PDF](https://arxiv.org/pdf/2610.23670v1)

**作者:** Lewen Yan `[一作]` (Shanghai Artificial Intelligence Laboratory), Jing Shao `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于自然语言“Factor”的链式思考（CoT）监控框架 OwlSight，能够通过在提示中自定义监控目标，实现单一模型覆盖多任务、细粒度与粗粒度两种监控需求；

**💡 创新点**

创新点在于：① Factor 机制允许在推理时动态指定监控目标，实现多任务覆盖；② 区分细粒度与粗粒度 Factor，满足从具体行为到宏观类别的不同监控粒度；③ 采用多实例学习（MIL）处理超长 CoT，避免截断并显著降低显存占用；

**🔧 技术方法**

技术包括：LLM 微调（SFT）以实现二分类；Factor 生成与标签分配的两阶段流程；MIL 框架对长 CoT 进行窗口化并聚合前两高分；训练使用 Qwen3.5-4B 等小型 LLM；评估使用 G‑Mean² 指标；

**📊 数据集**

数据集：训练集 17,812 个监控查询（8,796 条 CoT），覆盖 3 个领域（Capability、Safety、Trustworthiness）共 11 类；评估基准 32,222 条 Factor‑条件查询（15,971 条 CoT），涵盖多任务与干预实验；还使用 Kimi‑K2.6、GLM‑5.3‑Flash 等未见模型生成的 CoT 进行跨模型迁移测试；

**📈 对比分析**

对比方法包括多种通用大模型（GPT‑5.6‑Luna、Qwen3.8‑Flash 等）和专用 guard 模型（ShieldGemma‑9B、LlamaGuard‑4‑12B 等）。OwlSight 在细粒度 macro G‑Mean² 69.57、粗粒度 66.80，分别比通用模型高 1–30 分、11–28 分；在 11 类监控任务中多项领先；在干预评估中 15/16 项中取得最高分；在未见模型 CoT 上保持较高性能；

**⚠️ 局限性**

局限性：对 CoT 压缩与隐匿化仍有一定性能下降（macro G‑Mean² 分别降至 65.54/60.82 和 63.47/56.35）；Factor 的构造与标注需要人工审核，难以完全自动化；尚未在真实生产环境进行大规模部署验证；以及潜在的双重用途风险需在使用时关注。

---

## 664. TriRouter: Sparse-to-Dense Electrode-Space Routing for Robust EEG Motor Imagery Decoding

**arXiv ID:** 2610.23665 | [PDF](https://arxiv.org/pdf/2610.23665v1)

**作者:** Ying Zhou `[一作]`, Mengting Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了TriRouter框架，针对多通道MI‑EEG缺失通道实现鲁棒解码。

**💡 创新点**

通过三条专门路径（判别保持、mask鲁棒、潜在补偿）与验证校准的缺失严重度路由，动态分配权重，在不同缺失率下实现最佳性能。

**🔧 技术方法**

采用多路径深度学习、随机mask训练、潜在补偿网络、验证校准的功能先验路由及线性插值权重等技术。

**📊 数据集**

在BNCI2014‑001（22通道）和BNCI2014‑004（3通道）两套MI‑EEG数据集上进行实验。

**📈 对比分析**

与EEGNet、ShallowConvNet、FBCNet、CTNet及DSF‑CTNet等公开基准比较，TriRouter在随机缺失、区域缺失和低通道设置下均取得显著提升，尤其在高缺失率（50‑75%）下提升约4‑6个百分点。

**⚠️ 局限性**

尚未验证极端高缺失率（如90%）或全部通道缺失的情况，模型参数略多且对极端缺失仍有一定局限。

---

## 665. An Efficient Approach to Beetle Damage Detection of Banana Using Deep Learning

**arXiv ID:** 2610.23661 | [PDF](https://arxiv.org/pdf/2610.23661v1)

**作者:** Md. Mehedi Hasan Naeem `[一作]` (Jatiya Kabi Kazi Nazrul Islam University), Tushar Kanti Saha `[通讯]` (Jatiya Kabi Kazi Nazrul Islam University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

训练并评估轻量级YOLOv8n模型用于香蕉表面损伤（健康/不健康）的实例级检测。

**💡 创新点**

设计了本土化的香蕉损伤数据集，并证明YOLOv8n在保持小模型体积的前提下能实现高效检测。

**🔧 技术方法**

采用YOLOv8n目标检测框架，利用迁移学习、Anchor‑free检测头和分布式焦点损失实现定位与分类。

**📊 数据集**

使用本地市场收集的2473张图像、5827个标注实例（健康/不健康）组成的数据集。

**📈 对比分析**

在相同训练设置下与YOLOv8s对比，YOLOv8n取得81.36%精度、74.30%召回、77.67%F1、87.04% mAP@0.5、59.33% mAP@0.5:0.95，模型体积5.96 MB，推理时间6.54 ms/图。

**⚠️ 局限性**

仅单次80/20划分、无交叉验证、训练/测试存在图像重复、缺乏多中心外部验证、未提供生物学诊断或标注一致性度量。

---

## 666. Neural Expansion: A Unified Mechanism for How Deep Neural Networks Generalize

**arXiv ID:** 2610.23716 | [PDF](https://arxiv.org/pdf/2610.23716v1)

**作者:** Chashi Mahiul Islam `[一作]` (Florida State University), Xiuwen Liu `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过引入神经扩张框架，利用方向性扩展的射线覆盖（Generalization Intervals）解释深度网络的泛化机制。

**💡 创新点**

提出了方向敏感的GI概念，并将Jacobian奇异值与射线覆盖大小关联，构建了从局部角度解释测试样本分类的几何模型。

**🔧 技术方法**

采用Jacobian奇异值分解、线性近似、梯度下降训练以及射线覆盖测量等技术，并在多种网络架构上进行评估。

**📊 数据集**

使用MNIST、CIFAR-10、ImageNette及其随机标签、长尾变体，并实验了MLP、ResNet、VGG、ViT等模型。

**📈 对比分析**

与传统边缘、NTK、PAC-Bayes等全局理论对比；实验显示GI覆盖率与测试准确率高度相关，扩展覆盖可提升准确率，限制覆盖则降低准确率。

**⚠️ 局限性**

GI计算在高维输入上成本高，当前仅适用于分类任务，缺乏对回归或生成任务的推广，并且对Jacobian奇异值重排的理论证明尚未完成。

---

## 667. FlexPart: Text-Controlled Semantics-Aware 3D Part Segmentation at Flexible Granularity

**arXiv ID:** 2610.23659 | [PDF](https://arxiv.org/pdf/2610.23659v1)

**作者:** Yiqun Zhao `[一作]` (University of Hong Kong), Shenghua Gao `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种统一的文本控制3D网格分割框架，可根据用户给出的文本描述实现多粒度分割。

**💡 创新点**

创新点在于引入整体控制提示、自动化VLM注释管线生成多粒度数据，并通过离散二进制部件ID场实现直接面标签。

**🔧 技术方法**

采用VAE+双流Diffusion Transformer (DiT)、RoPE位置编码、随机背景丢弃训练以及VLM规划器完成文本到3D分割的端到端学习。

**📊 数据集**

使用由HY3D-Bench人工部件自动生成的15万条prompt-目标对构成的多粒度数据集，并在PartObjaverse-Tiny、PartNeXt/PartVerse-XL等公开数据集进行评测。

**📈 对比分析**

与Find3D、SAMPart3D、PartField、P3-SAM、SegviGen等基线对比，在mIoU上分别提升15%（自动提示）和22%（留置集），在细粒度和选择性分割任务中均显著优于现有方法。

**⚠️ 局限性**

局限在于对相似或对称部件的区分不佳，位置描述可能不唯一，导致边界模糊或部件ID错误。

---

## 668. Deep LLM layers expose passage-level context in EEG during silent reading

**arXiv ID:** 2610.23638 | [PDF](https://arxiv.org/pdf/2610.23638v1)

**作者:** Ingo Marquardt `[一作]` (nubrain), Priyanka Jain `[通讯]` (nubrain)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在一名被试静默阅读时，使用19通道干电极EEG记录的0.5秒窗口与Llama‑3.1‑8B语言模型各层的隐藏状态对齐，训练对比解码器，探究EEG可解码信息在模型深度上的分布，评估词级别解码与篇章级跟踪的关系，并验证单个窗口是否携带篇章信息。

**💡 创新点**

首次系统扫描完整模型深度（0‑32层），揭示早层最适合词识别，深层更偏向篇章追踪；通过两种控制实验排除词解码与目标几何、会话签名等混淆因素；证明即使无序列模型，单个EEG窗口也能捕获篇章上下文。

**🔧 技术方法**

卷积EEG编码器、可选的四层因果Transformer序列模型、CLIP对比学习目标、线性投影至256维归一化空间，利用Llama‑3.1‑8B的各层隐藏状态作为目标。

**📊 数据集**

来自一名被试的19通道EEG数据，19个干电极、600Hz采样、0.5s窗口，共409次运行、约51小时阅读、约249,753词条，文本包括侦探、奇幻与科幻小说段落以及报纸文章。

**📈 对比分析**

以within-run检索增益（对同一篇章候选）为主指标，整体增益（跨跑候选）与上下文跟踪增益等多重指标衡量。结果显示：无序列模型的within-run增益在第1–2层峰值约16.8pp，加入Transformer后整体增益最高峰值约38.5pp，且深层越深上下文跟踪增益比例越高；稀有词在中层表现最佳。

**⚠️ 局限性**

仅单一被试导致结果可能不具普遍性；验证集上选取的检查点导致性能偏乐观；层与隐藏状态几何的混杂；EEG时间漂移和会话签名仍可能影响深层解码；缺乏跨语言或跨任务的验证。

---

## 669. Deconstructing Multi-Task Active Learning: The Paradox of Gradient Conflict and Orthogonal Decomposition

**arXiv ID:** 2610.23623 | [PDF](https://arxiv.org/pdf/2610.23623v1)

**作者:** Juho Jung `[一作]` (Sungkyunkwan University), Jinyoung Han `[通讯]` (Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出正交分解框架，将多任务主动学习的目标从线性加权拆分为信息性筛选与探索性覆盖两阶段，并引入交叉任务互信息CTMI进行筛选。

**💡 创新点**

创新点在于揭示并解决“冲突悖论”，将采样与梯度冲突解耦，并通过正交分解与CTMI避免多任务主动学习中的多样性稀释。

**🔧 技术方法**

主要技术包括MC‑Dropout预测、CTMI协方差统计、两阶段k‑center贪婪覆盖、梯度修剪PCGrad以及多任务共享编码器。

**📊 数据集**

实验使用COCO、Cityscapes和NYUv2三个多任务数据集，涵盖检测、分割、分类、深度等任务。

**📈 对比分析**

与随机、Uncertainty、CoreSet、BADGE、BatchBALD、MT‑BADGE等基线相比，ODS在COCO上相对随机提升10.5%，在Cityscapes提升8.8%，在NYUv2提升7.5%，在多种基准下均保持最高或接近最高的MT‑Composite分数。

**⚠️ 局限性**

局限性包括对线性相关性假设的依赖、CTMI只能捕捉线性协方差、对不同领域的调参敏感以及实验仅在轻量级模型上验证，未覆盖大规模预训练网络。

---

## 670. Behavioral Couplings: From Formation to Behavioral Emergence in Language Models

**arXiv ID:** 2610.23847 | [PDF](https://arxiv.org/pdf/2610.23847v1)

**作者:** Dongyoon Hahm `[一作]` (KAIST), Kimin Lee `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在训练文档中同时描述两种行为时，后续单一行为微调会诱发另一种未目标行为的现象，并探讨其形成机制与特性。

**💡 创新点**

首次量化行为耦合（behavioral coupling），证明文档训练能形成耦合，并揭示耦合的选择性与可组合性，提出通过无学习（NPO）可抑制耦合的可能性。

**🔧 技术方法**

采用文档级next-token预训练形成耦合，随后针对单一行为的微调；利用激活向量提取与激活驱动、负偏好优化等技术进行评估与去学习。

**📊 数据集**

使用人工生成的包含双行为特征的虚构角色文档（1000篇）与 FineWeb-Edu 混合的数据集；为每行为生成1000问答对用于微调；在 Gemini 3.5 Flash、Qwen3.5-9B、gpt-oss-20B 等模型上实验。

**📈 对比分析**

通过对比仅文档训练、仅微调以及组合训练的模型，在目标行为与未目标行为的表达率上评估，结果显示组合训练后未目标行为表达率从0%提升至23%–93%，验证了耦合效果；在多模型、多行为配置上复现一致。

**⚠️ 局限性**

局限性包括需预先知道未目标行为才能通过无学习抑制耦合，无法对未知耦合做通用缓解；实验仅在合成虚构行为和中小规模模型上进行，是否在真实大规模预训练中同样存在尚待验证。

---

## 671. Beyond the Answer: Student-Conditioned Trace Construction for Reasoning Distillation

**arXiv ID:** 2610.23831 | [PDF](https://arxiv.org/pdf/2610.23831v1)

**作者:** Shaoang Li `[一作]` (Stony Brook University), Jian Li `[通讯]` (Stony Brook University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种学生条件化的推理轨迹构造与蒸馏框架，先用证据指导重写或修复学生的尝试，再由冻结的学生读者筛选最优轨迹，随后在训练中通过可控的修复曝光和答案损失权重进行蒸馏。

**💡 创新点**

创新点包括：① 将学生自身的尝试作为重写与修复的输入，实现学生特定的内容定制；② 用冻结的学生读者根据答案恢复来筛选教学轨迹，兼顾过程内容与学生可访问性；③ 通过修复曝光度和答案损失加权共同控制训练目标，使得推理轨迹与最终答案的监督平衡；④ 通过源任务与跨基准的双重评估验证方法的有效性。

**🔧 技术方法**

主要技术手段有：文本重写模型（教师）、冻结的学生匹配读者（用于答案恢复与评分）、离线轨迹构造流程、可变的修复曝光因子 m、答案权重 α 的辅助损失，整体实现了对推理轨迹与答案的联合蒸馏。

**📊 数据集**

使用的主要数据集是 MuSiQue 作为训练与源任务评估基准；跨基准评估使用 2Wiki、HotpotQA（多跳问答）和 QuALITY（长文本多项选择）来检验迁移性能。

**📈 对比分析**

与传统的 answer‑only、original‑trace、STaR、NLL、RSR、LARK、MBT‑R 等方法对比，本文方法在 MuSiQue 上达 58.4% EM，2Wiki 上 68.6% EM，显著超过所有基线；在 HotpotQA、QuALITY 上也取得最优或接近最优表现，证明在源任务与迁移任务上均有提升。

**⚠️ 局限性**

局限性包括：① 需要在训练前离线生成并筛选轨迹，增加前期计算成本；② 对学生模型的依赖较强，某些规模较小或不同体系的模型（如 Qwen3‑4B）提升有限；③ 结果主要针对多跳证据丰富任务，未验证在单跳或无证据的情境下的有效性；④ 参数如 α、m 的选择需经验调优，可能对不同任务不具通用性。

---

## 672. A generalised and modular proof-theoretic semantics for sequent systems

**arXiv ID:** 2610.23822 | [PDF](https://arxiv.org/pdf/2610.23822v1)

**作者:** Victor Barroso-Nascimento `[一作]` (University College London), Elaine Pimentel `[通讯]` (University College London)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了基于序列的基础扩展语义框架，统一了经典、直觉主义及亚线性逻辑的证明论语义，并证明了其完备性与一致性。

**💡 创新点**

提出了将语义条款与左引入规则对齐的模块化框架，并通过原子资源多重集索引实现了结构规则与连接词解释的分离，兼容多种子结构逻辑。

**🔧 技术方法**

使用了基于原子规则的基扩展语义、资源多重集索引、模拟基证明、原子Jankov基的弱/强条件，以及构造性与反证法的组合。

**📊 数据集**

本研究为理论性工作，无实验数据集。

**📈 对比分析**

通过形式化证明对比传统基扩展语义与现有自然演绎/序列系统的适用性，未涉及数值性能指标。

**⚠️ 局限性**

该框架目前仅覆盖命题层面；对更复杂的算术或模态扩展、无穷式子需要进一步研究；在证明过程中对原子规则的强假设可能限制适用范围。

---

## 673. CocoRNA: Collective RNA Design with Cooperative Multi-agent Reinforcement Learning

**arXiv ID:** 2610.23820 | [PDF](https://arxiv.org/pdf/2610.23820v1)

**作者:** Tianmeng Hu `[一作]` (University of Exeter), Ke Li `[通讯]` (University of Exeter)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个基于协作多智能体强化学习的RNA逆设计框架（MARL），通过任务分解和集中式评估实现高效序列生成

**💡 创新点**

创新点在于将RNA设计视为多智能体协作问题，使用集中式训练-分散式执行、结构感知分解和搜索增强探索，显著提升搜索效率与成功率

**🔧 技术方法**

采用了多智能体近端策略优化（MAPPO）、集中式评论家、基于dot‑bracket或矩阵编码的输入网络、以及搜索增强探索（SAE）等技术

**📊 数据集**

主要使用Rfam+65,000条二级结构数据集，划分训练/测试集，另外在RibonanzaNet、Pseudoknot数据集上做跨或acle评估

**📈 对比分析**

与多种基线（RNA‑SSD、INFO‑RNA、RL‑based、基于大模型等）比较，30 s时间限制下成功率达到97.75%，比最佳基线提升27.25个百分点，平均求解时间仅3.24 s，速度提升70×，且生成序列多样性最高

**⚠️ 局限性**

局限在于对极长序列的处理仍需滑窗技术，模型训练成本较高，对oracle的依赖可能导致不同预测器下性能略降，且在某些难以求解的核酸结构（如复杂核糖体酶）仍需更长时间

---

## 674. MOSS: Multimodal Origami Swamp Robot System

**arXiv ID:** 2610.23813 | [PDF](https://arxiv.org/pdf/2610.23813v1)

**作者:** T. Lan `[一作]` (Imperial College London), S. F. Armanini `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并测试了一种可变形的3D打印折纸轮子，集成在一款小型两轮两轮驱动的两栖机器人MOSS上，用于在湿地软土、沙、泥、以及水面上实现适应性行驶。

**💡 创新点**

通过单一扭杆机制实现轮子宽度、直径和鳍片角度的耦合形变，配合望远轴实现扭矩传输与轴向位移分离；同时将折纸轮与鳍片直接集成，提升软土牵引。

**🔧 技术方法**

3D打印多层折纸结构、整合鳍片、轴向可伸缩驱动轴、动态姿态控制及 ROS2/电机控制。

**📊 数据集**

使用实验室人工制备的四种地面（土壤、沙子、含不同水分的泥土0.5/1.0/2.0倍、以及水）以及真实湿地场景进行行驶测试。

**📈 对比分析**

在相同平台、相同驱动指令下比较收缩与扩展两种轮形，采用速度、滑移率、轮轴下沉、能耗/米(CoT)等指标；结果显示收缩形态在固体土、泥0.5/1.0时更节能，而扩展形态在沙子、泥2.0及水面上速度更快、CoT降低约20-30%。

**⚠️ 局限性**

形变机制造受限于材料韧性导致可重复形变寿命有限；无法独立评估轮宽、直径、鳍角对性能的单独影响；过渡界面清洁度限制了软硬地面过渡。

---

## 675. Persistent Planar Memory for Video World Models

**arXiv ID:** 2610.23795 | [PDF](https://arxiv.org/pdf/2610.23795v1)

**作者:** Yuze He `[一作]` (Tsinghua University), Yinghao Xu `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一种视频世界模型，利用平面原语作为持久的3D记忆，能够在生成长序列视频的同时输出可导出的平面几何、深度和法向。

**💡 创新点**

创新点在于将平面原语作为跨块持久3D记忆，兼具紧凑（约10万平面）、几何显式（可度量深度、法向、可导出表面）、跨视图一致（共享生成网络预测）以及可自适应（通过几何合并不断优化），并通过Depth Token Fusion将渲染深度回馈到生成网络。

**🔧 技术方法**

技术方案包括：Wan 2.2视频扩散Transformer作为主干，新增平面预测头（基于Plana3R的矩形高斯光栅化），Depth Token Fusion（ViT编码器+张量级融合），可微矩形高斯渲染，面片合并与投影渲染作为条件，双阶段训练（先冻结主干后微调几何模块）。

**📊 数据集**

使用的训练数据为DL3DV和RealEstate10K混合集（约80K视频），评估数据集为DL3DV测试集（100场景）和RealEstate10K测试集（100段），同时在DL3DV完整序列（300-400帧）上做长序列评估。

**📈 对比分析**

在短期（49/121帧）和长期（300-400帧）下，与Gen3C、Voyager、VMem、SPMem等基线比较，取得最高的PSNR、LPIPS、CLIP、DINOv2相似度，深度误差（AbsRel、δ<1.25）也显著低于同类模型，长序列中保持更高的感知与几何一致性。

**⚠️ 局限性**

局限性包括：对高度有机或不规则几何的场景（如植被、流水）表现不如点云或TSDF；对极端相机轨迹或未见过的视角变化易退化；模型仅处理静态平面，无法捕捉动态表面；双阶段训练未实现端到端的共适配，可能限制整体性能提升。

---

## 676. Polytope Sampling at the Speed of Isoperimetry

**arXiv ID:** 2610.23775 | [PDF](https://arxiv.org/pdf/2610.23775v1)

**作者:** Anming Gu `[一作]` (University of Texas at Austin), Hubert Yang `[通讯]` (University of Texas at Austin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种新的高效采样多面体的算法，提出了一个O(m)迭代的“向后oracle”实现，并通过功能冷却技术得到更低的步长τ，实现了高精度的随机采样。

**💡 创新点**

通过在双重变量空间进行向后条件化和Metropolized随机步行，构造了弱χ²收敛的近似向后采样器；引入功能冷却框架，将冷却与自对数凸障相结合；将冷却与高维采样相结合，降低了迭代复杂度到O(m)并将步长降低到O(log m/δ)。

**🔧 技术方法**

自对数凸障、Newton迭代、Metropolized随机步行、弱Poincaré不等式、强对数凸性、χ²到TV的转换、功能冷却调度、矩阵求逆与大数估计、线性规划的中心点方法。

**📊 数据集**

本文为理论分析，未使用具体数据集，主要在抽象的凸多面体上进行证明。

**📈 对比分析**

与先前的随机行走、Hit‑and‑Run等方法相比，提供了更快的收敛速度和更低的迭代次数；理论上在多面体维度m、n下实现 O(m n^{ω‑1} + m n^{5/4} log²n) 的时间复杂度。

**⚠️ 局限性**

需要先验的自对数凸障及其梯度/ Hessian 的高效计算；算法对步长τ的选择敏感；在实际应用中对高维多面体的实现可能受限于矩阵运算的常数；缺乏实验验证和对非凸约束的扩展。

---

## 677. Logic-Guided Graph Denoising Diffusion Models

**arXiv ID:** 2610.23766 | [PDF](https://arxiv.org/pdf/2610.23766v1)

**作者:** Samuel Cognolato `[一作]`, Andrea Cavallaro `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

暂无内容，无法描述

**💡 创新点**

暂无信息

**🔧 技术方法**

暂无信息

**📊 数据集**

暂无信息

**📈 对比分析**

暂无信息

**⚠️ 局限性**

暂无信息

---

## 678. A Paired RGB-Sonar Dataset for Fish Monitoring with Consumer Sonar

**arXiv ID:** 2610.23759 | [PDF](https://arxiv.org/pdf/2610.23759v1)

**作者:** Mourad Derbedrosian `[一作]` (California Institute of Technology), Pietro Perona `[通讯]` (California Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `67630363-6be0-4f51-ab05-7198250671a5` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并公开了一个由消费级前视声呐与顶部视RGB视频配对的锦鲤数据集。

**💡 创新点**

创新点在于将低成本 Garmin LiveScope 声呐与可视化RGB帧同步，并提供海量配对数据与几何校准，弥补了高端科学声呐的成本壁垒。

**🔧 技术方法**

采用帧时间戳同步、人工同步事件、极坐标到笛卡尔变换、RANSAC 平面单应性估计等技术进行数据配对与校准。

**📊 数据集**

使用了自行采集的约3.3小时、4种配置（15H、40H、100H、20V）的声呐与RGB帧，共计105,362配对。

**📈 对比分析**

论文未给出具体算法比较或性能指标，仅说明该数据集可用于后续自动检测、跟踪、计数等任务的开发与评估。

**⚠️ 局限性**

局限包括时序同步误差、平面单应性无法完全补偿深度差异、视场重叠不完全、仅单一池塘单一鱼种、缺乏任务级标注等。

---

## 679. Beyond Multi-Stage Pipelines: A Label-Correlation Ensemble for Multi-Label Retinal Fundus Disease Classification

**arXiv ID:** 2610.23747 | [PDF](https://arxiv.org/pdf/2610.23747v1)

**作者:** Md Asif Uddin `[一作]` (BRAC University), Rafeed Rahman `[通讯]` (BRAC University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

比较三种模型（三级管道、平面分类器、标签相关Transformer解码器）在合并的39类视网膜疾病多标签分类任务上的性能，寻找最优方案。

**💡 创新点**

在统一的实验条件下，首次将三种架构直接对比，发现标签相关Transformer能显著提升检测率，且平面+Transformer混合在TTA下获得最佳结果；同时揭示硬门限导致的排名指标失效以及数据源短路问题。

**🔧 技术方法**

使用Swin‑V2‑B骨干、标签相关Transformer查询解码器、四视角测试时增强（TTA）、阈值重调、负样本均衡、交叉熵与不平衡权重等技术。

**📊 数据集**

合并RFMiD 1.0、RFMiD 2.0、ODIR‑5K三大公开数据集，共计12,652张眼底图，构建39个评分目标。

**📈 对比分析**

通过统一前处理、损失、优化器与校准，直接在同一验证集上比较各模型；最终平面+Transformer混合+TTA方案在macro‑AUC 0.972、micro‑F1 0.678、macro‑F1 0.456上实现最佳。

**⚠️ 局限性**

稀有类仍无法准确检测；硬门限导致信息丢失；多源数据中仍潜在短路；实验仅在公开数据上，缺乏临床真实样本验证；且cascade的额外训练成本和维护成本较高。

---

## 680. VocalGrad: Evaluating Acoustic Perception in Audio Language Models

**arXiv ID:** 2610.23745 | [PDF](https://arxiv.org/pdf/2610.23745v1)

**作者:** Koki Ryu `[一作]` (University of Tokyo), Hitomi Yanaka `[通讯]` (University of Tokyo)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布VocalGrad基准，用二分类任务评估大规模音频语言模型对低层声学变化的感知能力，并通过线性探测和LoRA微调深入分析模型内部瓶颈。

**💡 创新点**

创新点包括：①将音频感知任务简化为与语义推理分离的二分类问题；②通过统一的prompt与数据生成流程实现多类声学变换的系统化评测；③揭示两类失败模式（头部瓶颈 vs 信息衰减）并证明LoRA微调可以显著提升性能。

**🔧 技术方法**

技术手段：语音增强与变换、文本prompt设计、Transformer LLM、线性探测（probe）、LoRA微调、统计分析与可视化。

**📊 数据集**

数据集：从VCTK语料生成21,600条样本的VocalGrad（9类声学变换、3强度层、4时间轨迹），并做纯正弦波实验验证语言内容对结果的影响；使用公开的VocalGrad数据和代码仓库。

**📈 对比分析**

比较方法：使用模型文本输出准确率与人类标注对比，最佳LALM仅51.8%（近随机） vs 人类94.4%；线性探测显示内部表示对某些变换（如音量、背景噪声）保留良好；LoRA微调后大部分类别可达90%以上准确率，跨类别迁移也出现提升。

**⚠️ 局限性**

局限性：当前LALM对低级声学变化的感知能力不足，模型训练目标不鼓励保留低层声学信息，跨域与跨类别泛化受限，且提示语言对结果影响有限。

---

## 681. AlphaLedger: Portfolio-Grounded On-Policy Distillation for Regime-Aware Alpha Mining

**arXiv ID:** 2610.23744 | [PDF](https://arxiv.org/pdf/2610.23744v1)

**作者:** Xiang Li `[一作]` (Hong Kong University of Science and Technology), Xiaowen Chu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了 AlphaLedger，结合本地因子生成器、证据驱动的股票池 EviPool 以及基于市场状态的加权门控 RRG，完成因子挖掘与策略部署。

**💡 创新点**

创新点在于：① 通过经济情境的在策略蒸馏（EC‑OPD）将组合层面的增量价值信息传递给生成器；② EviPool 用多源滞后证据动态构造候选股票空间；③ RRG 根据市场状态自适应调节因子权重，实现对不同 regime 的灵活响应。

**🔧 技术方法**

使用了 Qwen3 LLM 的语法约束生成模型、逆 KL 蒸馏训练、回测交易协议、滞后证据聚合与状态残差门控加权等技术。

**📊 数据集**

基于中国 A 股日行情、成交量、资金流量数据（Tushare）和发行人新闻情感（Cailian Press）构建训练与测试集。

**📈 对比分析**

在相同的长仓交易协议下与七类基线（指数、公式化、RL、LLM 代理、LLM 后训练、托管 LLM 等）对比；在 CSI500 市场中年化净回报 31.6%、Sharpe 1.43；使用 EviPool 后年化回报 55.4%、Sharpe 2.06、最大回撤 10.8%。

**⚠️ 局限性**

局限性包括：生成器仅训练一次且因子库固定；评估仅覆盖 A 股长仓；不同方法候选数与规则不完全对齐；未验证跨市场、双向交易及完整风险归因。

---

## 682. Blind Precise Assembly: Proprioceptive Sim-to-Real Peg-in-Hole for Diverse Parts

**arXiv ID:** 2610.23849 | [PDF](https://arxiv.org/pdf/2610.23849v1)

**作者:** Antonia Bronars `[一作]` (Massachusetts Institute of Technology), Pulkit Agrawal `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练并实现了仅使用关节自身感知的盲装配策略，完成插孔、齿轮啮合和螺母拧入，并在真实机器人上实现零调试转移。

**💡 创新点**

证明关节本身即可捕捉细微接触信息，提出盲装配方法并与视觉策略对比显示更优泛化；同时提供精确的MuJoCo装配生成器。

**🔧 技术方法**

使用PPO强化学习、时序卷积编码器、接触监督、稀疏奖励，在MuJoCo仿真中大规模并行生成随机几何装配；并结合隐层解码器实现接触状态预测。

**📊 数据集**

通过自研生成器产生数千个随机凸几何装配（插孔、齿轮、螺母），不依赖公开数据集。

**📈 对比分析**

在模拟中与匹配视觉传感器的策略对比，盲策略在所有任务上实现近乎完美成功率，且在未见几何上泛化更好；在真实机器人上盲策略以73%的成功率完成多形状插孔，零调试。

**⚠️ 局限性**

需要初始时已与装配物接触；策略只能进行局部误差修正，无法处理大范围搜索或完全离散的初始姿态。

---

## 683. AI-Driven Reproducibility and Analytical Robustness in Scientific Discovery

**arXiv ID:** 2610.23845 | [PDF](https://arxiv.org/pdf/2610.23845v1)

**作者:** Shicheng Liu `[一作]` (Stanford University), Monica S. Lam `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了SEER-Repro基准，开发LLM辅助的研究成果再现与鲁棒性评估流水线，实现对20篇SEER数据库研究的自动再现与系统化的分析选择敏感性检测；

**💡 创新点**

首次将LLM代理与多分析者思路结合，形成端到端可执行的再现与鲁棒性评估框架，并通过专家评审验证模型筛选结果的有效性；

**🔧 技术方法**

使用大型语言模型（GPT‑5.6、Claude、Kimi K3、GLM 5.3）及代码生成与执行代理，配合自定义提示与算法实现复制与扰动分析；

**📊 数据集**

利用SEER 9数据库（39张数据表与40份文档）以及相关人口与手册资料，涵盖1975‑2018年美国癌症登记信息；

**📈 对比分析**

再现率77.4%，错误分析表明数据缺失是主因；鲁棒性评估中通过扰动与直接编码代理发现9.5%（55/579）结果被模型评为高风险，专家复核后仅有1例真正有影响，显示系统能够有效聚焦需人工审查的内容；

**⚠️ 局限性**

限制包括：历史数据不可恢复导致的再现受限、需人工精细调研的文档与数据缺口、LLM生成代码错误或偏差、以及专家评审仍占比大。

---

## 684. Explainable Machine Learning Analysis of Temperature Effects on Mechanical Strength of Fiber-Reinforced High-Performance Concrete

**arXiv ID:** 2610.23811 | [PDF](https://arxiv.org/pdf/2610.23811v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 685. SparseMax-dSTL: Sound Robustness with Horizon-Independent Gradient Allocation for Temporal Logic Trajectory Optimization

**arXiv ID:** 2610.23837 | [PDF](https://arxiv.org/pdf/2610.23837v1)

**作者:** Sai Sandeep Damera `[一作]` (University of Maryland), Calin Belta `[通讯]` (University of Maryland)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了基于梯度的STL约束轨迹优化，提出一种使用稀疏最大化（sparsemax）构造的下界平滑极值逼近，从而在优化中既保持可听性又保证梯度权重不随时间窗增长而消失。

**💡 创新点**

创新点在于提供了同时满足下界可听性和梯度权重不随轨迹长度衰减的平滑极值逼近，并给出误差上界与梯度下界的理论证明，首次实现了在任意样本数下保持对违规样本的显著梯度贡献。

**🔧 技术方法**

使用技术包括稀疏最大化（sparsemax）与二次正则化最大值、梯度下界分析、误差上界推导、逆向自动微分与增量 Lagrange 方法等；论文在仿真中实现了这些算法。

**📊 数据集**

实验数据集来自仿真，包含平面两轮小车与七关节法兰克普拉纳机械臂与人类共享工作空间的场景，使用 MuJoCo/Warp 进行动力学仿真。

**📈 对比分析**

与传统的 log‑sum‑exp（plain LSE）、sound LSE 以及 GMR（generalized‑mean robustness）方法进行比较。实验表明，稀疏最大化在违反轨迹上鲁棒度始终为负且梯度权重保持为 1，优化收敛速度明显快于其他方法，且误差上界已知，可提供安全保证。

**⚠️ 局限性**

局限性包括：稀疏最大化下界在极值处梯度为零，可能对接近临界的样本缺乏敏感性；仅实现了一次可微的梯度方法，尚未充分利用稀疏海森矩阵进行二阶优化；在极端约束紧张的情形下，可能出现所有样本梯度被压制而导致搜索停滞。

---

## 686. Waiting Buys Exactly Two Moments: Endogenous Identification in Prediction-Driven Allocation

**arXiv ID:** 2610.23830 | [PDF](https://arxiv.org/pdf/2610.23830v1)

**作者:** Rama AlHamidi `[一作]` (Texas A&M University at Qatar), Hasan Kurban `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在有限预算的预防干预下，等待获取更多信息与因风险高者先离而失去干预机会的权衡，构建模型评估何时最优干预。

**💡 创新点**

创新点在于证明在有限观察前缀下只能识别两个新的风险矩，导致部分最佳干预时点不可消除的识别歧义，并给出精确的最优日期识别与最坏情况下的保底证据。

**🔧 技术方法**

使用多项式矩推断、非参数最大似然估计（NPMLE）、Beta匹配、EM早停、Bootstrap bagging、CMS bootstrap、极大极小与最坏情况决策规则等技术。

**📊 数据集**

采用30种模拟风险分布（Beta、混合、高危子群、双峰等）并在不同 horizon、预算比例和观察长度下生成数据进行验证。

**📈 对比分析**

通过平均平方相对遗憾（MSR）和第10百分位效率比较，发现早停EM在两种样本量下均优于单一NPMLE，bagging优于regret‑calibrated，CMS略优，整体性能相对优秀。

**⚠️ 局限性**

局限在于仅基于模拟验证，Bootstrap近似与极大极小规则对扫描细化敏感，结果可能不在完整矩类上稳健。

---

## 687. Jarvis: Coordination Deserves a Brain of Its Own in Agentic Interactive Models

**arXiv ID:** 2610.23804 | [PDF](https://arxiv.org/pdf/2610.23804v1)

**作者:** Yuxiang Wang `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Jarvis体系，通过前后端协同的协调枢纽实现实时语音对话与后台任务的无缝配合。

**💡 创新点**

核心创新是将决策交给类型化的判断模型JEV，并在代码层面执行检查，保证所有操作始终基于最新任务版本并防止stale、ghost、expired等失败模式。

**🔧 技术方法**

使用了实时双向语音模型（Venus‑Audio、MiniCPM‑o 4.5、VoiceChat‑11B）、深度学习决策模型（Gemma‑4、JEV）、任务账本、播放接收机、GRPO训练以及视觉感知层。

**📊 数据集**

在多个公开基准上评估：τ‑Voice（278 任务）、Full‑Duplex‑Bench v3（100 录音）和 v2（200 任务），以及公开的 τ‑Voice 领导榜。

**📈 对比分析**

与 Qwen Audio Agent、GPT‑Realtime、Gemini Live 等系统对比，Jarvis‑Audio 在 τ‑Voice 上从 29% 提升至 69% Pass@1，在 FDB‑v3 约 +10 分，显著优于同前端的其他实现。

**⚠️ 局限性**

局限性包括对任务版本和计划的频繁检查导致额外开销，依赖预先训练的决策模型，需要手工设计操作集合；在极端并发或非文本输入场景下仍可能出现竞态。

---

## 688. DLite: History-free Speculative Decoding for Agentic AI

**arXiv ID:** 2610.23787 | [PDF](https://arxiv.org/pdf/2610.23787v1)

**作者:** Jiaming Xu `[一作]` (Shanghai Jiao Tong University), Guohao Dai `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种不维护草稿模型 KV 缓存的 speculative decoding 框架 DLite，直接利用目标模型的隐藏状态在当前解码位置做局部推测，从而降低历史建模成本。

**💡 创新点**

创新点包括：①将历史建模交给目标模型，去掉草稿模型对历史的独立编码；②使用并行骨干与轻量顺序头在隐藏空间恢复块内因果依赖，避免全词表投影；③设计两阶段训练策略，先用同构历史感知教师桥接，再对齐目标模型。

**🔧 技术方法**

采用了 history‑free parallel backbone、低秩 RNN‑style lightweight sequential head、深度嵌入区分隐藏与词嵌入、两阶段 distillation（warm‑up + stable）以及 token‑level CE+TV 损失。

**📊 数据集**

训练集为 1.3M Open‑PerfectBlend + 600k NVIDIA Nemotron Post‑Training，评测使用 Qwen3‑4B/8B 在 GSM8K、MATH‑500、HumanEval、MBPP、MT‑Bench、Alpaca、LongBench v2（多文档 QA、长 In‑ctx 学习、代码仓库理解、长对话）、AgentMemoryBench、AgentLongBench、BFCL‑v3、SWE‑Bench 等任务。

**📈 对比分析**

与自回归基线及 EAGLE3、DFlow、DFlash、DFlash2、DSpark 等方法对比，DLite 在一般、长上下文和代理工作中接受长度 τ 均不低于对手，代理工作上提升 2–6 倍，Speedup 高达 70%，并在多并发吞吐率上显著优于 DSpark，消除了 KV 缓存维护开销。

**⚠️ 局限性**

局限性：需要目标模型隐藏状态的实时访问，对目标模型更新敏感；两阶段训练增加了训练复杂度；在极短或非代理场景的优势尚需进一步验证；对硬件的依赖（如 GPU 上的 KV 缓存处理）仍未完全消除。

---

## 689. TileDRAM: Compiler-Controlled Tile-to-DRAM Layouts for 3D DRAM GPGPUs

**arXiv ID:** 2610.23767 | [PDF](https://arxiv.org/pdf/2610.23767v1)

**作者:** Guangyuan Xu `[一作]` (ShanghaiTech University), Yajun Ha `[通讯]` (ShanghaiTech University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 TileDRAM，将编译器可见的张量块布局扩展到 3D DRAM 地址空间，实现编译器控制的 DRAM 放置。

**💡 创新点**

创新点在于将 GPU 线性布局抽象迁移到 DRAM 位置，利用张量块访问签名作为稳定的特殊化单元，在保持控制器地址映射不变的前提下，仅通过普通地址算术即可获得优越的行缓冲局部性。

**🔧 技术方法**

技术方法包括：基于张量块访问签名的线性布局表征、行缓冲状态模拟与子空间求解、离线布局合成、编译期地址映射与运行时绑定、以及在 Accel‑Sim/Ramulator2 上的完整跟踪重放。

**📊 数据集**

使用了 108 条 MoE FFN 的完整执行跟踪（来自 Qwen3、DeepSeek‑V3、Qwen3.5‑35B‑A3B、Kimi‑K3、Hunyuan‑A13B、Llama‑4 等模型）以及 36 条流与 6 种 3D DRAM 组织的重放组合。

**📈 对比分析**

与传统的 DRAM 放置做比较，TileDRAM 将请求加权行冲突率从 75.7% 降至 15.2%，实现 1.77× 的几何平均加速；在 6 种 DRAM 组织间重定向时加速率提升至 2.20×；在 RTX 3090 上地址转换开销仅为 0.17%。

**⚠️ 局限性**

局限性包括仅在单个 CU–stack 对进行评估、仅针对分块 GEMM（未覆盖注意力、卷积等核）、未验证多 CU‑stack 协同布局，以及缺乏在真实 3D DRAM 硬件上的实验验证。

---

## 690. Universal Semantic Enhancement for Weakly Supervised Anomaly Detection

**arXiv ID:** 2610.23765 | [PDF](https://arxiv.org/pdf/2610.23765v1)

**作者:** Siyu Zhou `[一作]` (Hong Kong University of Science and Technology), Yingjie Zhou `[通讯]` (Sichuan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在弱监督表格数据异常检测任务中，引入跨域通用语义并融合领域特定特征，提升检测效果

**💡 创新点**

创新点：①设计了 Prototype‑Conditioned Modulation（ProtoFiLM‑Dynamic）将预训练模型的通用语义与自编码器提取的域特定特征自适应融合；②采用 RoSAS 交叉样本插值损失丰富稀缺标签的监督信号；③在推理阶段使用无参数 PDF 融合两种分数，进一步提升性能

**🔧 技术方法**

技术：冻结 TabPFN 预训练表格基座模型、对目标数据训练自编码器、ProtoFiLM‑Dynamic 融合模块、RoSAS 交叉插值损失、PDF 推理融合、Xavier 初始化、Adam 优化

**📊 数据集**

使用 ADBench 42 个公开表格数据集（医疗、网络、图像/信号、金融、工业等），每个数据集 80/20 划分，训练时仅给 8 条异常标签

**📈 对比分析**

与传统 ML（RF、CatBoost）、深度弱监督方法（DevNet、DeepSAD、FEAWAD、PReNet）以及单一 TabPFN 基线比较，在 42 个数据集上平均排名分别为 1.1（AUROC）和 1.5（AUPRC），表现为最高；在 8 个标签下明显优于其他方法，且随着标签数增大仍保持优势

**⚠️ 局限性**

局限性：①类别原型在联合训练前一次性计算并保持不变，无法随模型改进更新；②对高类内变异的异常（如 Speech 数据集）原型条件可能产生误导；③依赖 TabPFN 预训练模型，若预训练数据与目标域差距过大可能效果下降

---

## 691. LagNet: Lag-Structured Operators for Long-Term Time Series Forecasting

**arXiv ID:** 2610.23735 | [PDF](https://arxiv.org/pdf/2610.23735v1)

**作者:** Zhenan Yu `[一作]` (Harbin Institute of Technology), Jin Yang `[通讯]` (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了长周期时间序列预测中历史到未来映射的结构，并提出了基于滞后共享的LagLayer和LagNet模型。

**💡 创新点**

创新点在于发现并利用预测矩阵中相对滞后对应的对角共享模式，构造矩形Toeplitz矩阵并加入水平/垂直缩放以显著降低参数并提升样本效率。

**🔧 技术方法**

采用梯度敏感性诊断、Toeplitz共享、可学习的尺度因子、深度非线性多通道网络以及在现有模型中替换线性层的技术。

**📊 数据集**

在八个公开长期预测基准（ETTh1/2/ETTm1/2、Electricity、Solar-Energy、Traffic、Weather）上进行实验。

**📈 对比分析**

与DLinear、PatchTST、iTransformer、TimeMixer、TimeKAN等基线比较，LagNet/LagLayer在大多数数据集取得最优或次优成绩，参数量大幅下降，且在少量训练样本下表现更佳。

**⚠️ 局限性**

局限在于相对滞后共享假设在某些数据（如Solar）不完全成立，导致性能略逊；高频复杂模式下效果仍需进一步提升。

---

## 692. StopWatch: Goal Grounding for Object-goal Navigation

**arXiv ID:** 2610.23841 | [PDF](https://arxiv.org/pdf/2610.23841v1)

**作者:** Xinhao Liu `[一作]` (Amazon FAR), Chen Feng `[通讯]` (Amazon FAR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在目标导向导航中显式地进行目标定位（goal grounding），显著提升了在六个基准上的成功率。

**💡 创新点**

创新点在于将目标定位拆分为感知模型与决策模型，利用场景注释直接监督目标可见性与距离，解决了传统VLM导航模型停靠失败的问题。

**🔧 技术方法**

技术上采用Qwen3-VL-4B作为感知模型、Qwen3-VL-2B作为策略模型，利用前景探测专家示范进行行为克隆，并使用结构化文本进行中间推理。

**📊 数据集**

数据集包括HM3D、MP3D、HSSD的开放词汇和闭合词汇版本，并对HM3D进行了注释修正与新的MP3D-OV、HSSD-OV基准。

**📈 对比分析**

在官方与清洗后的评估协议下，本系统在所有六个基准上均领先，最高提升约8.8点，特别是在停靠成功率上表现突出。

**⚠️ 局限性**

局限性主要在于目标定位监督仅来自已有的3D场景注释，难以迁移到未标注的视频数据；且实验仅在仿真室内场景中验证。

---

## 693. DiscoveryRSI: Recursively Learning to Discover Valuable Hypotheses

**arXiv ID:** 2610.23839 | [PDF](https://arxiv.org/pdf/2610.23839v1)

**作者:** Jianyuan Zhong `[一作]` (Phai Labs), Ling Yang `[通讯]` (Phai Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种通过多代理进化搜索构建假设树，并通过信用优化、重放和再发现实现递归自我改进的DiscoveryRSI框架。

**💡 创新点**

创新点在于将假设生成、实验计划编译与经验反馈三合一，形成可追踪的研究记录并用信用分配实现模型与工作流的递归学习。

**🔧 技术方法**

使用多代理进化搜索、信用优化、工具重放、Harbor任务编译以及大型语言模型（DiscoveryRSI-27B）等技术。

**📊 数据集**

使用了四个生命科学域（HCC、MIMIC‑IV、eICU、UK Biobank、GEO）和四个电池域（电气诊断、健康预测、退化机理、使用历史）等数据集。

**📈 对比分析**

与GPT‑6 Luna、GPT‑5.6 Sol、Claude Opus 5等模型对比，假设质量从4.86提升到5.39，灵感率6.3%（GPT‑6仅2.5%），并在多维度指标上均优于基线。

**⚠️ 局限性**

局限性包括对检索覆盖率和提案格式的敏感性、实验仍停留在计算机仿真、缺乏实验室验证，以及在探索广度和跨域迁移方面的提升空间。

---

## 694. Self-Building Labs

**arXiv ID:** 2610.23826 | [PDF](https://arxiv.org/pdf/2610.23826v1)

**作者:** Jiaru Bai `[一作]`, Alán Aspuru-Guzik `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究构建了一个自我构建实验室（SBL），能够基于科学目标自动识别能力缺口、创建数字孪生、开发机器人技能并在真实实验室中执行完整实验流程。

**💡 创新点**

创新点在于将数字孪生、LLM驱动的规划与代码生成、机器人安全 guardrail 以及经验记忆融入同一个闭环，实现实验室能力的自动扩展与重用，而不需要人工介入进行硬件或软件集成。

**🔧 技术方法**

技术手段包括：LLM（如 GPT‑6‑Astra）用于实验计划、代码生成和技能构建；NVIDIA Isaac Sim 进行物理仿真和数字孪生构建；cuRobo 进行机器人运动规划；REST/MCP 接口与实验室设备交互；视觉、深度摄像和力/扭矩传感器提供反馈；Python脚本与 API 进行工作流编排。

**📊 数据集**

数据来源主要为实验室现场拍摄图像、设备型号与 API 文档（构建数字孪生与设备模型），以及量子点合成实验产生的光谱与荧光强度数据；未使用公开标准数据集。

**📈 对比分析**

通过与传统实验室、自动实验室（self‑driving lab）三种范式对比，展示了 SBL 在单台机器人、协同机器人和多工作站的实验执行能力；在量子点合成实验中成功执行了 236 条 OT‑2 命令和 19 个机器人阶段，证明了从规划到物理执行的完整闭环，虽然未给出详细数值指标，但实验表明显著降低了人工调试和硬件集成时间。

**⚠️ 局限性**

局限性包括：对设备 API 的误报或缺失导致的成功/失败判断困难；视觉与位姿误差在精细插入等操作中限制了直接从仿真迁移；LLM 采样与探索速度仍然较慢，需要人工介入解决冲突与假设；安全 guardrail 仍需模型遵循，未能完全消除人为失误；此外，实验室布局或设备变更后仍需重新验证已构建的技能。

---

## 695. Learning Personalized Memory Gating from Delayed One-Sided Feedback

**arXiv ID:** 2610.23812 | [PDF](https://arxiv.org/pdf/2610.23812v1)

**作者:** Kaixuan Ji `[一作]` (Salesforce AI Research), Huan Wang `[通讯]` (Salesforce AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了个性化记忆门控问题，提出了一种基于检索的效用标记管道，并构建了用于记忆门控评估的数据集。

**💡 创新点**

创新点在于将个性化记忆门控问题框架化为一种带有单边延迟反馈的苹果品尝问题，并提出了一种算法来解决该问题，确保了较低的后悔界限。

**🔧 技术方法**

使用了基于检索的效用标记管道和上下文带有苹果品尝反馈的在线学习模型。

**📊 数据集**

构建的数据集基于用户与大型语言模型（LLM）之间的交互历史，采用了PerMem-Bench和合成的长数据集。

**📈 对比分析**

通过与多种基线方法进行比较，实验结果表明，所提出的方法在所有内存预算下均优于现有的LLM方法和简单的全接纳基线，显示出其作为个性化记忆门控策略的有效性。

**⚠️ 局限性**

限制在于当前方法假设每个用户的参数是稳定的，未来的研究方向可以考虑效用模式的演变。

---

## 696. ABF-Bench: A Verifiable Benchmark for Agentic Blockchain Forensics

**arXiv ID:** 2610.23656 | [PDF](https://arxiv.org/pdf/2610.23656v1)

**作者:** Xiaohang Yu `[一作]` (Imperial College London), William Knottenbelt `[通讯]` (Imperial College London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 ABF-Bench 交互式基准，用以评估在区块链取证任务中 LLM 代理通过检索、解释和报告证明支持的资金追踪能力。

**💡 创新点**

创新点包括：①将真实洗钱案例自动化转换为可验证的跨链任务；②构建基于规则的保守参考图与可量化查询预算；③设计 ABF 代理 harness，结合工作记忆、工具和启发式规则，实现更高效、可解释的取证流程；④将取证过程拆分为搜索、识别、证据构造和报告四个阶段，并为每阶段提供可测度的指标。

**🔧 技术方法**

使用技术：LLM 代理（Codex、Claude Code）、交互式查询接口（读取区块链记录）、可验证的规则引擎、工作记忆管理、简易工具箱（提交预览、摘要压缩）、启发式决策框架，以及基于规则的检索和验证器。

**📊 数据集**

数据集：从 2025 年 Bybit 被盗案中挑选 58 个任务，筛选后 33 个公开任务，覆盖以太坊、比特币两条链，包含多跳转账、DeFi 交换、跨链桥接、托管/混币存取、地址掺水等多种操作。

**📈 对比分析**

对比方法：ABF 代理 harness、Codex、Claude Code 以及三种固定顺序跟踪器（宽度优先、深度优先、金额最大优先）。评估指标：边缘召回（Edge Recall）、可靠性（Fidelity）和边缘精度（Edge Precision），以及查询/代币成本。结果显示：ABF harness 在前期查询效率上优于固定跟踪器，并能识别隐藏托管；在整体召回上仍略低于宽度优先跟踪器；可靠性与 Codex 相当，但 Claude Code 存在“过度自信”问题。

**⚠️ 局限性**

局限性：仅基于单一真实案例，可能存在模型预训练/后训练对结果的影响；保守规则导致仅捕获可见流的子集；评估涉及整体系统（模型+策略+工具），难以分离单一组件贡献；缺乏多案例验证，难以评估跨场景泛化能力。

---

## 697. Steering beyond Language: Improving Multilingual Training-Free Text Embeddings via Paired-Residual Subspace Removal

**arXiv ID:** 2610.23807 | [PDF](https://arxiv.org/pdf/2610.23807v1)

**作者:** Ailiang Lin `[一作]` (Institute of Science Tokyo), Manabu Okumura `[通讯]` (Institute of Science Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在多语言环境下系统评估了基于冻结LLM的无训练文本嵌入方法，并提出了一种在推理时通过对成对残差子空间进行消除来抑制语言特异性、提升语义对齐的技术。

**💡 创新点**

创新点在于首次利用平行语料的成对残差构造低秩语言敏感子空间，并在冻结LLM内部层级中按推理时去除该子空间的投影，从而实现无训练、无参数更新的多语言嵌入提升。

**🔧 技术方法**

技术包括：对多语言平行句子在各层的最后词向量取均值后构造残差；对残差矩阵做SVD得到低秩子空间；在指定层对隐藏状态做投影消除并传播至后续层；与多种无训练嵌入基线（PromptEOL、Pretended CoT、ECHO、TP、CP）融合。

**📊 数据集**

使用了MTEB、MMTEB的208个单语评测任务、BUCC.v2、Tatoeba、FLORES三大跨语言文本对挖掘基准；并在18种语言（含未参与子空间估计的语言）以及200+英语-其他语言对上进行评估。

**📈 对比分析**

与原始无训练基线相比，PRSR在单语平均分数提升约6-7分，跨语言挖掘上提升18-56分，表现优于随机/语言均值等对照，并在不同LLM（LLaMA‑2‑7B、Mistral‑7B、Qwen3‑8B）上保持一致。

**⚠️ 局限性**

局限包括：仅在7B-8B级别的解码器LLM上验证，未测试更大模型；跨语言评测仅覆盖英语与其它语言，未直接检验非英语间的语义对齐；潜在的社会偏见仍可能保留在生成的嵌入中。

---

## 698. Enforcing Code Reasoning in Agentic Scaffold for More Effective Issue Resolution

**arXiv ID:** 2610.23782 | [PDF](https://arxiv.org/pdf/2610.23782v1)

**作者:** Changshu Liu `[一作]`, Reyhaneh Jabbarvand `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对7955条编程代理轨迹的大规模实验，分析了代码推理在问题解决中的普遍性与效果，并提出一种在每个阶段强制执行代码推理的多子代理框架。

**💡 创新点**

创新点在于系统化地将代码推理嵌入到定位、修复、审核与验证四个关键步骤，显著提升了问题定位准确率、修复完整性和验证覆盖率。

**🔧 技术方法**

使用技术包括基于大型语言模型（OpenAI、Claude、Llama）驱动的代理、执行路径模拟、代码推理、需求抽取与审计、自动测试生成与迭代改进，以及多子代理架构。

**📊 数据集**

实验使用SWE‑Bench系列数据集（SWE‑Bench、SWE‑Bench Lite、SWE‑Bench Pro），涵盖Python、Go、TypeScript、JavaScript共731个评测实例以及7955条不同代理架构产生的轨迹。

**📈 对比分析**

与两组基线（-M3、-5.6‑Luna、-4.5）比较，所提框架在解决率上提升6.29%–10.53%，平均每实例成本仅增加$0.02–$0.28，且通过McNemar检验与多次复测验证结果的显著性。

**⚠️ 局限性**

局限性包括代码推理的准确性未完全保证，实验仅覆盖SWE‑Bench任务与现有六种代理架构，缺乏跨任务与更大规模模型的泛化验证。

---

## 699. LLM-Assisted Bi-Approximate CHC Solving

**arXiv ID:** 2610.23771 | [PDF](https://arxiv.org/pdf/2610.23771v1)

**作者:** Baiting Jiang `[一作]` `[通讯]` (Tsinghua University), Baiting Jiang (Tsinghua University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HornCraft 这一 LLM 辅助的 CHC 求解框架，结合 Derivation Power Under‑Approximations (DPUs) 与 LLM 生成的 invariant，压缩深度/分支的推理过程并提供可验证的下近似和上近似。

**💡 创新点**

创新点在于：① 引入符号中间状态（witness‑carrying）DPUs，使 LLM 生成的多步关系能够在不引入额外量化符号的前提下通过 SMT 验证；② 证明 DPUs 的安全组合与复用规则；③ 将 LLM 生成的下近似和上近似在同一求解框架中联合使用，实现了 bi‑approximate 的 CHC 求解。

**🔧 技术方法**

技术包括：LLM（Codex）生成候选 DPUs、invariant 及 reach‑fact；SMT 验证（Z3）对 DPUs 与 invariant 进行证书检查；Spacer、Houdini 进行上近似求解；证书重放与组合推导算法；以及对参数化 CHC 的批量化评估。

**📊 数据集**

数据集：CHC‑COMP 2026 LIA‑Lin 与 LIA 追踪中 260 个 Spacer‑hard 实例（共 202+58 个）；以及自定义的 Depth‑and‑Branching CHC Suite，共 49 个参数化实例，涵盖线性与非线性深度/分支推理。

**📈 对比分析**

对比方法：与传统求解器（Spacer、LoAT、Golem、Eldarica）以及无 DPUs 的 HornCraft 变体进行比较。结果显示 HornCraft 在 260 个实例中获得 228/260（88%）的覆盖率，比无 DPUs 的 202/260（78%）提高 10个百分点；unsafe 结果从 22 提升至 42；在 49 个参数化实例中解决 42/49，显著扩展了可处理的深度与分支规模。

**⚠️ 局限性**

限制：仅针对整数算术 CHC 进行评估；LLM 交互结果存在随机性，未做多次平均；评估仅限于未解决的实例，未覆盖完整 CHC‑COMP 套件；DPU 的生成仍依赖 LLM 的提示，交互成本与模型版本变化可能影响稳定性。

---

## 700. MTEdit: Packed Trajectories Beat Rollouts for Multi-Turn Image Editing

**arXiv ID:** 2610.23751 | [PDF](https://arxiv.org/pdf/2610.23751v1)

**作者:** Houyuan Chen `[一作]` (SenseTime Research), Anyi Rao `[通讯]` (HKUST MMLab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大型多轮图像编辑数据集MTEdit，并提出相应的评测基准MTEdit-Bench；

**💡 创新点**

通过“装帧轨迹画布”(PTC)一次性生成多轮编辑结果，显著减少视觉质量漂移；

**🔧 技术方法**

利用视觉语言模型(VLM)生成PTC提示，封闭源编辑模型生成PTC，再用VLM过滤、提炼提示并对图像对进行SIFT匹配校准；

**📊 数据集**

MTEdit数据集共包含50万条编辑轨迹（4–8轮）和269万条编辑对，涵盖step‑wise与三类context‑aware编辑；

**📈 对比分析**

在MTEdit‑Bench上与多款开源与闭源模型对比，Qwen‑Image‑MTEdit在IF、CC与DRS上均优于开源模型，DRS在长链编辑中保持4.2+，并显著降低视觉漂移；

**⚠️ 局限性**

仍受限于最大8轮、对PTC生成的质量与多样性依赖、未覆盖极端编辑场景以及部分评测指标对生成模型的评判偏差。

---

## 701. Acceptance Is Not Speed: Length Inflation in Training-Free Lossy Speculative Decoding

**arXiv ID:** 2610.23818 | [PDF](https://arxiv.org/pdf/2610.23818v1)

**作者:** Chiatzen Wang `[一作]` (University of Waterloo), Hongyang Zhang `[通讯]` (University of Waterloo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了五种无训练的放宽式验证（relaxed verification）规则在12个目标‑起草模型对（target‑drafter）上的表现，重点测量完成长度、验证轮数、耗时和任务准确率，而非单纯的接受率。

**💡 创新点**

创新点在于提出并量化“长度膨胀”（length inflation）现象，证明接受率的提升往往伴随生成文本长度增加，导致实际速度提升被抵消甚至逆转；并提出使用长度修正的速度模型，揭示长度才是决定放宽式验证效率的核心因素。

**🔧 技术方法**

使用了speculative decoding框架，结合五种放宽验证规则（CACTUS、mentored decoding、fuzzy speculative decoding、speculative cascade 等），以及 vLLM 0.26.0 的推理堆栈实现这些规则，并在 360 个实验单元中记录指标。

**📊 数据集**

采用了六个基准数据集：数学类（GSM8K、AIME24）、代码生成（HumanEval、LiveCodeBench）、对话聊天（MT‑Bench）以及长文本推理（LongBench‑v2）。

**📈 对比分析**

比较方法：对每个目标‑起草组合在相同提示下进行 lossless（严格）和放宽规则的对照实验；记录平均接受率、完成长度、验证轮数、耗时和准确率；结果显示放宽规则平均接受率提升 1.34×，但长度膨胀 1.54×，验证轮数和耗时多半未下降，准确率在多数情况下下降，证明单纯提升接受率并不能保证速度和质量提升。

**⚠️ 局限性**

局限性：①匹配接受率时不同规则的参数不等价，比较基于“最松弛”设置；②生成长度受到预算截断影响，膨胀量可能被低估；③仅在单一推理堆栈和采样配置下评估；④实验次数有限，部分数据集（如 AIME24、MT‑Bench）样本量较小，统计置信区间宽；⑤未评估训练型放宽规则或其它解码策略。

---

## 702. Bad Diagnosis: Analyzing Consent and Data Collection Practices of Mobile Medical Apps

**arXiv ID:** 2610.23851 | [PDF](https://arxiv.org/pdf/2610.23851v1)

**作者:** Ethan Myers `[一作]` (Colorado State University), Indrakshi Ray `[通讯]` (Colorado State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对10,000多款Android移动医疗(mHealth)应用进行系统性分析，评估其在安装和首次使用时的权限请求、同意获取以及对个人身份信息(PII)的收集情况；

**💡 创新点**

首次将LINDDUN隐私威胁框架与三类隐私威胁（PT1-PT3）结合，形成一套完整的隐私测量流程，并在此基础上提出针对mHealth的操作性定义和高精度分类器；

**🔧 技术方法**

采用多种技术：Transformer语言模型（BERT、DeBERTa等）对应用描述进行多标签权限推断；动态UI fuzzing+MITMProxy+OCR实现对同意界面和网络流量的自动检测；PII检测使用正则、编码和哈希匹配，并结合日志PID/时间筛选；

**📊 数据集**

核心数据集为10,688款从Google Play Store抓取的免费mHealth应用（含APK）以及5,000款人工标注的应用描述，用于训练权限推断和同意识别模型；

**📈 对比分析**

与现有方法相比，权限推断模型在AC‑Net基准上达ROCAUC 0.99、PR‑AUC 0.81；mHealth分类器DeBERTa达94.6%准确率、0.916 F1；结果显示69.15%应用存在权限过度请求，93%缺失同意界面，1/14的应用在已获得同意的前提下仍收集未获授权的PII；

**⚠️ 局限性**

局限性：仅覆盖Play Store免费应用，未处理付费或其他平台；动态分析仅在初次启动期间进行，忽略后续账号创建和长期行为；部分应用因崩溃或兼容性问题未能检测，导致威胁比例可能被低估；

---

## 703. Sensory Comfort Across Multisensory XR Prototypes for Autistic Adults

**arXiv ID:** 2610.23833 | [PDF](https://arxiv.org/pdf/2610.23833v1)

**作者:** Yibo Meng `[一作]` (Cornell University), Xiaolan Ding `[通讯]` (North China University of Science and Technology Health Science Center)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了五种完整XR原型在自闭症成年人中的感官舒适度差异，并结合问卷和访谈阐释受试者的喜好与不适因素。

**💡 创新点**

创新点在于以完整配置而非单一感官量化变量的方式评估XR体验，提出了“感官统一”“渐进过渡”“自我控制”“可调适性”等候选设计维度，兼顾定量与定性证据。

**🔧 技术方法**

使用XR头显、空间音频、灯光、触觉和温度反馈等多模态技术，并通过计时触发与目光/近距离交互实现不同原型的呈现。

**📊 数据集**

样本为20名自闭症成年志愿者（年龄19–55岁，性别10/10），所有受试者在实验室中体验相同七段科学内容的五种XR配置。

**📈 对比分析**

采用混合方法：配合Friedman检验、Wilcoxon配对比较和主题分析，对比五种原型的综合舒适度评分，结果显示L3最低、L4、L5表现较好；但L5评分压缩至上限，统计上与L1、L4差异不显著。

**⚠️ 局限性**

局限包括样本规模小、缺乏对照组、感官舒适量表混合了喜好项、L5存在上限压缩、未单独操纵感官量化或交互参数，且未测量受试者的感官基线或情绪状态。

---

## 704. Online Graph Coloring with Random Arrivals

**arXiv ID:** 2610.23824 | [PDF](https://arxiv.org/pdf/2610.23824v1)

**作者:** Hirotaka Yoneda `[一作]` (University of Tokyo), Masataka Yoneda `[通讯]` (University of Tokyo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在随机到达模型下，提出一种随机化指数时间算法，能够为任意k可着色的n顶点图使用期望O(k³log²klog⁴n)种颜色；

**💡 创新点**

首次在固定k≥3时实现多项式对数数量的颜色数，并且在随机顺序下取得比已知下界更优的竞争比；

**🔧 技术方法**

核心技术包括硬核模型（hardcore model）随机采样独立集、递归分层、以及对每层使用First-Fit着色；

**📊 数据集**

论文为理论研究，无实验数据集，全部分析基于理论证明；

**📈 对比分析**

与先前的最优O(n^1/2)（k=3）或O(n^{2/3})（k≥3）结果相比，显著降低颜色数至O(log⁴n)，并在竞争比上实现O(n^{2/3})，优于传统下界Ω(n/log²n)；

**⚠️ 局限性**

局限性在于算法指数时间、仅适用于随机到达顺序，无法推广到对抗性到达或多项式时间实现。

---

## 705. Beyond SAEs: Consistent Concept Discovery with Sparse Coding

**arXiv ID:** 2610.23801 | [PDF](https://arxiv.org/pdf/2610.23801v1)

**作者:** Jinho Choi `[一作]` (KAIST), Steffen Schneider `[通讯]` (Helmholtz Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ConSDL方法，替代传统稀疏自编码器中的线性编码器，采用逐样本稀疏编码、系数重拟合与共活相似性正则化，实现对基础模型激活的更一致稀疏分解。

**💡 创新点**

将逐样本稀疏编码融入字典学习并加入系数重拟合和共活相似性正则化，显著提升不同随机种子下字典与系数的一致性，解决传统SAE的重现性和不稳定性问题。

**🔧 技术方法**

使用FISTA求解稀疏编码、系数重拟合、共活相似性正则化，并通过交替优化的字典学习框架实现。

**📊 数据集**

在合成数据以及Vision领域的CLIP ViT-B/16、DINOv2和语言领域的GPT-2 Small、Pythia-160M（对应ImageNet、FineWeb等训练集）上进行评估。

**📈 对比分析**

与六种基线SAE方法（ReLU、TopK、BatchTopK、Matryoshka、MP-SAE、iSAE）在跨种子字典与系数一致性、字典利用率、重构质量和稀疏探测性能上对比；ConSDL在跨种子一致性上提升约17.6%/35.9%，字典利用率更高，稀疏探测性能保持相当。

**⚠️ 局限性**

训练成本高（逐样本稀疏编码代价），对极大规模基础模型验证有限，且一致性仍低于合成设置，且对非线性或流形结构的概念表示仍存在局限。

---

## 706. Video Diffusion Avatars: Taming Video Diffusion Models into Controllable 3D Avatars

**arXiv ID:** 2610.23772 | [PDF](https://arxiv.org/pdf/2610.23772v1)

**作者:** Simon Giebenhain `[一作]` (Technical University of Munich), Matthias Nießner `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出Video Diffusion Avatars (VDAs)，仅用单张图片训练可在任意视角、姿态、表情下生成逼真视频。

**💡 创新点**

创新在于把视频扩散模型直接作为可控头像，无需显式3D资产；通过去视角偏差的表达空间和代理网格实现多视角一致性。

**🔧 技术方法**

使用WAN-Animate-14B扩散模型、X-NeMo表达编码、FLAME+SAM-3D Body代理网格、三相机训练策略与去视角去耦。

**📊 数据集**

训练数据来自公开多视角人脸数据集NeRSemble、AVA-256；评估使用BecomingLit、Multiface、VFHQ-test。

**📈 对比分析**

与LAM、FlexAvatar、UIKA、CAP4D、MVP4D、FaceCam等基线对比，VDAs在LPIPS、JOD、3DGS一致性等指标上提升约29%–66%，并在多视角重建中获得最高质量。

**⚠️ 局限性**

缺点是推理耗时长（≈12min/77帧），无法实时；长序列易出现时间漂移；光照固定在参考图像，表达空间非端到端训练。

---

## 707. Dynamic Load Balancing under Restricting Updates

**arXiv ID:** 2610.23768 | [PDF](https://arxiv.org/pdf/2610.23768v1)

**作者:** Aaron Bernstein `[一作]` (New York University), Thatchaphol Saranurak `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了动态环境下的负载均衡问题，目标是将工作分配给机器以最小化最大机器负载。

**💡 创新点**

提出了一种近似最优的算法，能够在递减模型中实现负载均衡，更新时间为O(m(log(nW)/))。

**🔧 技术方法**

使用了一种基于拥塞模型的负载均衡算法，并开发了一种新的分析框架来证明其在动态环境中的有效性。

**📊 数据集**

使用了包含工作和机器的二分图G = (J ∪ M, E)，其中每个工作都有一个正权重w(j)。

**📈 对比分析**

与现有的增量算法进行比较，提出的算法在递减设置中实现了相同的近似界限，并且在更新时间上具有优势。

**⚠️ 局限性**

算法的局限性在于，尽管在递减模型中表现良好，但在完全动态模型中仍然存在挑战，尤其是在处理更复杂的负载均衡问题时。

---

## 708. The Italian lawmaking process: a unified and structured dataset

**arXiv ID:** 2610.23741 | [PDF](https://arxiv.org/pdf/2610.23741v1)

**作者:** Matilde Contestabile `[一作]` (Scuola Superiore Sant'Anna), Andrea Vandin `[通讯]` (Technical University of Denmark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对意大利1985-2026年间通过的5346部法律的立法程序进行结构化事件化处理，构建了完整的事件日志和法规登记表。

**💡 创新点**

创新点在于使用LLM（Gemma 4）对未结构化的立法文本进行自动化事件抽取，并在100条手工注释的基准上验证其高准确率，实现了大规模、统一的立法过程数据集。

**🔧 技术方法**

技术上采用了Gemma 4 31B LLM、few-shot 提示、严格的JSON schema约束、GPU本地推理和后处理规则来规范事件信息。

**📊 数据集**

数据集来源于Normattiva和意大利参议院的官方档案，包含所有IX–XIX届立法的原始文本、法律元数据和约101,413条抽取事件。

**📈 对比分析**

通过与100条手工标注的参考数据对比，Gemma 4在few-shot设置下达成93%以上的Jaccard相似度，micro/macro F1接近100%，显著优于其他评估模型。

**⚠️ 局限性**

局限性包括仅覆盖已通过的法律、元数据随时间不一致、事件时间仅到日精度、部分法律存在自动检测到的异常，且未覆盖被否决或撤回的议案。

---

## 709. Rethinking the design space of Vanilla Pixel Diffusion Transformers

**arXiv ID:** 2610.23854 | [PDF](https://arxiv.org/pdf/2610.23854v1)

**作者:** Soumik Mukhopadhyay `[一作]` (University of Maryland), Aniruddha Mahapatra `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种方向自适应的预测参数化与轻量级后置卷积细化模块，用于改进像素空间扩散变压器的生成质量和高频细节。

**💡 创新点**

创新点在于：1）引入连续可学习的主成分方向权重，自动在噪声预测与清晰图像预测之间插值；2）设计ConvRefiner后置卷积模块，低成本恢复高频谱并保持FID；3）系统评估显示该框架能在图像、视频与文本图像生成任务中无需手工参数搜索即可达到或超过现有最优。

**🔧 技术方法**

使用了扩散变压器、主成分分析（PCA）方向权重学习、闭式速度恢复公式、LPIPS、PatchGAN对抗损失以及后置卷积细化网络。

**📊 数据集**

主要数据集包括ImageNet-1K（图像）、UCF-101（视频）和内部约900M图像的文本图像生成数据集，评估基准包括FID、FVD、GenEval、DPG、PRISM等。

**📈 对比分析**

与JiT、AsymFlow、PixelDiT、MMDiT等基线模型对比，实验表明：在ImageNet B/H 级别的FID分别为3.81和1.81，视频生成FVD为122/97，文本图像生成在GenEval/DPG/PRISM等指标上处于最前列，且后置ConvRefiner能显著提升高频谱与视觉细节。

**⚠️ 局限性**

局限性包括：1）仍需针对不同任务调优PCA方向初始化与学习率；2）ConvRefiner需要额外的后置训练步骤，且对极大模型的可扩展性尚未充分验证；3）在某些视频生成场景中，直接从头训练而非后置微调可能效果不佳。

---

## 710. A Large Language Model LLM Approach to COVID-19 Fake News Detection

**arXiv ID:** 2610.23800 | [PDF](https://arxiv.org/pdf/2610.23800v1)

**作者:** Muhammad Arif `[一作]` (Yale University), Sreeraj Ramachandran `[通讯]` (Yale University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用多种预训练 Transformer 与大型语言模型（GPT‑2、T5、XLNet）对 COVID‑19 相关英文推文进行真假新闻二分类，并尝试加权融合、粒子群优化、Nelder‑Mead、Powell、BFGS 与 trust‑constr 等多种优化融合方法，进一步提升分类性能。

**💡 创新点**

创新点在于将多种预训练模型与多种优化融合技术系统化结合，对同一数据集进行全面比较；首次展示 GPT‑2 在该任务上最优的表现，并通过多种融合方法验证性能提升；同时提供了完整的热图可视化与指标对比。

**🔧 技术方法**

核心技术包括：预训练 Transformer（BERT‑base、XLM‑RoBERTa‑base、ALBERT‑base、DistilBERT）与大型语言模型（GPT‑2、T5、XLNet）；优化融合技术（加权融合、PSO、Nelder‑Mead、Powell、BFGS、trust‑constr）；评价指标为准确率、精确率、召回率、F1‑score，并使用混淆矩阵热图进行可视化。

**📊 数据集**

使用的公开数据集为 COVID‑19 English Tweets，共 6420 条推文，其中 3360 条为真实信息，3060 条为虚假信息，按 70/10/20 划分为训练、验证和测试集。

**📈 对比分析**

方法上采用统一的数据划分与相同的训练/验证流程，分别计算每个模型的四个指标，并绘制热图比较；实验结果显示 GPT‑2 的准确率最高达 96.31%，精确率 96.12%，召回率 96.10%，F1 96.00%；其余模型均在 95% 以上；优化融合后最高准确率提升至 96.15%，相较于现有方法显著提高。

**⚠️ 局限性**

局限性在于：①仅处理二分类任务且数据量有限，缺乏多语言、多类别验证；②模型在小样本上表现相似，可能因过拟合或特征不足导致泛化性不足；③未在实时流或更大规模数据集上验证，需进一步扩展实验。

---

## 711. FuTransformer: FP8 prefill and INT4 decode for Llama-style inference on Hopper

**arXiv ID:** 2610.23796 | [PDF](https://arxiv.org/pdf/2610.23796v1)

**作者:** Wei Wang `[一作]` `[通讯]`, Wei Wang

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对 NVIDIA Hopper GPU 的 Llama 风格 Transformer 推理实现，分别针对预填充（prefill）和单词解码（decode）阶段进行专门化优化。

**💡 创新点**

创新点包括：1) 在 CODA 的 GEMM‑epilogue 结构上加入推理专用的直接 KV 缓存写入、固定 RMSNorm 融合和延迟 FP8 输出；2) 为 decode 设计 INT4 权重与 KV 缓存的分块对齐布局，并使用 Hadamard 变换降低 INT4 缓存误差；3) 通过延迟缩放实现 FP8 量化时不需额外写回中间张量；4) 在 Hopper Tensor Core 上实现注册级解量化和异步复制。

**🔧 技术方法**

技术手段包括 FP8 E4M3 与 INT4 低精度量化、Tensor Core WGMMA 与 TMA 异步复制、分块 Hadamard 旋转、延迟缩放、注册级解量化、CUDA Graph 编排以及对 Hopper Tensor Core 的特定 fragment 对齐设计。

**📊 数据集**

使用 Llama‑3.1‑8B 预训练模型进行评估，测试数据集为 WikiText‑2（困惑度评估）以及 HellaSwag 与 ARC‑Challenge（零样本多选任务）以验证准确率。

**📈 对比分析**

与 PyTorch FP8、BF16 以及 CODA 在单层和全模型层级进行基准比较。INT4 解码在 H100 上比 BF16 速度快 1.3–2.4 倍，INT4 全模型在 batch 1 下比 PyTorch 快 2.9 倍；prefill 加速至 1.48×；FP8 路径比 BF16 快 1.48×，INT4 缓存比 FP8 快 31–51%。

**⚠️ 局限性**

局限性：仅针对 Hopper GPU 设计，未涵盖多 GPU 调度、分布式通信或真实服务延迟；INT4 推理导致 WikiText‑2 凶惑度升 9–15%，ARC‑Challenge 准确率下降至约 7.6%；还需验证尺度初始化、饱和监控及任务级准确性。

---

## 712. DrugHarness: An Evolvable Agent Harness for Computational Drug Discovery

**arXiv ID:** 2610.23791 | [PDF](https://arxiv.org/pdf/2610.23791v1)

**作者:** Yuxing Chen `[一作]` (Shanghai Artificial Intelligence Laboratory), Ming Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 DrugHarness，一个可演化的科学执行代理，用于药物发现任务，通过组合科学操作、条件技能学习和独立审查来提升分子筛选、优化和设计的性能，并在实验上验证设计化合物。

**💡 创新点**

创新点在于将可组合的科学执行环境与条件技能演化相结合，形成带有适用性条件和证据的可复用技能；引入独立审查器对候选技能进行源验证与目标试验，并通过反馈驱动的技能修订实现持续改进。

**🔧 技术方法**

使用了大型语言模型（Codex、Qwen3.8-Max 等）、科学工具（RDKit、BindingDB、ADMET-AI 等）、Python 科学环境、可追踪工作空间、经验驱动的技能诱导与评估循环。

**📊 数据集**

使用了 MolBench、SMDD-Bench（包括 40 个 Lead‑Optimization 任务）以及 HPK1 设计任务的实验参考化合物（四个）和公开的生物医学资源。

**📈 对比分析**

通过与 MolClaw‑CC、CAi 等基线比较，DrugHarness 在 MolBench 的亲和力比较准确率提升至 91.9%（+10.8%），Hits@3 均值提升 55%（1.24）；在 SMDD‑Bench 的成功率提升至 60% 及 28%；在 HPK1 任务中 Docking 分数更优，实验 IC₅₀ 为 14.19 nM，优于所有参考化合物。

**⚠️ 局限性**

限制在于未详细评估资源和计算成本与其他代理框架的对比；技能演化仅在单一基准内评估，缺乏大规模或未见目标的泛化测试；实验验证规模有限，未覆盖完整的可开发性属性。

---

## 713. The Collection of a Real-World, Large-Scale Dataset for Automated Skill Evaluation in Central Line Dressing Change

**arXiv ID:** 2610.23789 | [PDF](https://arxiv.org/pdf/2610.23789v1)

**作者:** Ummey Tanin `[一作]`, Qian `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了第一份真实临床中心静脉导管更换（CLDC）过程的视频数据集，并提出了基于视频的AI评估管线

**💡 创新点**

创新点在于：①将CLDC细化为19个可计时的子任务并配备结构化评价指标；②采用实地手术与模拟相结合的多阶段数据采集与标注方案；③在此数据集上展示了基于视频的子任务级自动评分方法，首次实现了针对护理操作的AI评估框架

**🔧 技术方法**

使用了RGB视频的3D卷积网络（I3D、C3D）与Temporal Segment Network（TSN）进行动作识别与分级；同时在数据收集阶段采用了4K 60fps GoPro HERO13摄像机、纸质注释表与数字化CSV文件；评估采用5折交叉验证

**📊 数据集**

主要数据集为：①JIGSAWS外科手术视频（验证基准）；②本研究收集的365段CLDC视频（含18个子任务时间戳和Met/Not Met标签）

**📈 对比分析**

在JIGSAWS上复现了两种模型：I3D+TSN训练准确率为95.45%，C3D+TSN为90.12%；在CLDC数据集上尚未给出最终模型性能，论文仅提供了数据集和评估框架，未来工作计划在此数据集上开展训练与验证

**⚠️ 局限性**

限制包括：①数据仅来自一所医院的模拟病人（缺乏真实病人多样性）；②目前模型仅在外科数据验证，CLDC上无实测效果；③未实现实时反馈与机器人教学功能；④样本量虽较大，但仍有限，难以覆盖所有潜在操作差异

---

## 714. Not All Actions Contribute Equally: Learning Precise Progress Estimation from Imprecise Demonstrations

**arXiv ID:** 2610.23769 | [PDF](https://arxiv.org/pdf/2610.23769v1)

**作者:** Huanqian Wang `[一作]`, Gao Huang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于多示例经验图的进度估计器 PRIME，用于在高精度机器人操作任务中从不完美演示中学习精确的进度评估，并将其作为稠密奖励信号用于强化学习。

**💡 创新点**

创新点在于：①不再假设演示中的每个动作都贡献进度，而是通过构建跨轨迹的加权图，利用视觉相似性和几何接近性来连接状态；②使用图的最短路径代价近似到达成功的测地距离，从而天然区分前进、停滞和退步；③在进度标签生成后，将该距离学习为可在线估计的奖励模型，实现高效的离线和在线 RL。

**🔧 技术方法**

核心技术包括：多视角视觉特征提取（DINOv3），跨轨迹邻接判定（配置接近+视觉一致），加权图构建，Dijkstra 最短路径搜索，基于进度标签的 MSE 训练与进度优先采样。

**📊 数据集**

使用真实机器人数据（Piper‑X 机器人）完成的四个任务（抽屉打开＋放置、普通插孔、难度更高插孔、充电器插入），每个任务收集 20–50 条演示；以及基于 ManiSkill 的扩展仿真环境 ExpandedManiSkill，包含多种物体与更大工作空间。

**📈 对比分析**

与线性进度、SARM、RoboMeter、ViVa、GSR、RECAP、ProcVLM 等基线对比。实验显示 PRIME 在真实任务中相较于 SFT 提升 2.2×–5×，在最难的充电器插入任务中 5 倍成功率；在仿真中成功率提升至 67.4%（相较于稀疏奖励 54%）。诊断基准上，PRIME 在局部精度辨别的准确率达 92.5%，显著优于最佳基线。

**⚠️ 局限性**

局限性包括：①需要大量标注成功终点的演示数据；②图构建和视觉编码仍存在一定计算开销；③对动态变化环境或极端视觉变化的鲁棒性尚未充分验证；④对跨任务迁移的通用性仍需进一步研究。

---

## 715. Learning to Correct Geometry in Generated Videos

**arXiv ID:** 2610.23856 | [PDF](https://arxiv.org/pdf/2610.23856v1)

**作者:** Weijie Lyu `[一作]` (University of California Merced), Ming-Hsuan Yang `[通讯]` (University of California Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种后处理的视频-视频模型 Video Geometry Corrector（VGC），用于纠正生成视频在摄像机运动下的几何失真。

**💡 创新点**

创新点在于构建了基于真实摄像机运动的合成配对数据集，利用光流辅助分离摄像机运动和物体运动，并且通过后处理而非修改生成器实现跨模型几何校正。

**🔧 技术方法**

使用视频扩散模型（Wan 2.1/2.2）、光流估计（SEA-RAFT）、轨迹自适应关键帧生成、以及条件扩散去噪。

**📊 数据集**

使用 DL3DV、SpatialVID、RealEstate10K 等公开视频数据集，并通过 Wan 2.1‑Fun‑InP 生成失真对。

**📈 对比分析**

与 Geometry Forcing、Geometry Forcing+REPA、VideoGPA 等几何一致性方法对比，在 pose‑condition、prompt‑driven 以及通用视频增强任务中，VGC 在 PSNR、SSIM、RVE、3DCS 等指标上均优于或相当于基线，且能在更大模型上实现超分和动态内容提升。

**⚠️ 局限性**

局限性包括对光流质量敏感，初始小规模模型在动态场景恢复上表现有限；生成失真对依赖预训练 inpainting 模型，可能不覆盖所有失真类型；以及需要大规模算力训练。

---

