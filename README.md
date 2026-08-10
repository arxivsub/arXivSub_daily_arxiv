# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-10 | 今日论文总数: 515

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. WorldMark: A Plug-and-Play World Knowledge Interface for Cross-Host Language Model Watermarking

**arXiv ID:** 2608.06416 | [PDF](https://arxiv.org/pdf/2608.06416v1)

**作者:** Song Xiao `[一作]`, Kejun Zhang `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出一种名为WorldMark的插件式水印接口，利用世界知识记忆（WKM）生成语义-情节图谱，并通过知识显著性评分和不对称知识调制（AKM）自适应地调整大语言模型（LLM）的水印强度，从而在不需要重新训练模型或额外检测器参数的情况下提升水印的可检测性与文本质量。

**💡 创新点**

创新点在于①将结构化的语义与情节知识融入水印过程；②设计知识显著性估计器将检索到的知识映射为token级显著性分数；③提出不对称调制机制，将显著性分数分解为质量缓解系数与检测增强系数，实现对已锚定知识词和非锚定词的差异化水印强度；④实现全插件化，兼容多种主机水印（logit、采样、混合）且无额外检测器开销。

**🔧 技术方法**

使用的技术包括：世界知识记忆（WKM）与AriGraph结构化图谱；Sentence‑BERT编码语义显著性；Asymmetric Knowledge Modulation（AKM）算法；与MorphMark等自适应强度水印的集成；以及对C4、OpenGen等数据集的评测。

**📊 数据集**

实验使用了C4大规模文本语料库进行主实验，另外在OpenGen、TriviaQA、NaturalQuestions等数据集上做了跨模型与跨任务的扩展评估。

**📈 对比分析**

与原始MorphMark及其复现版本比较，WorldMark在TPR@1%（包括鲁棒TPR）、F1（包括鲁棒F1）以及困惑度（PPL）上均有提升，平均鲁棒TPR提升约0.03，鲁棒F1提升约0.01，同时保持或略微降低生成延迟，证明了方法在不增加显著检测成本的前提下提升了水印效果。

**⚠️ 局限性**

局限性包括：①对极端语义重写或强改写攻击（如LLM重写、翻译循环）尚未系统评估；②需要对WKM进行检索与显著性估计，可能在极大上下文或多模态场景下开销上升；③不同主机水印的参数调优仍依赖经验；④直接注入记忆而不做调制在某些情况下会导致不稳定表现，强调完整AKM流程的重要性。

---

## 2. Recovering Lesion Parameters from Aphasic Picture Naming Error Profiles in Large Language Models

**arXiv ID:** 2608.06429 | [PDF](https://arxiv.org/pdf/2608.06429v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 3. Real-time Whole-Body Motion Planning for Mobile Manipulators Carrying Arbitrarily Shaped Payloads via Kinematically-Coupled SVSDF

**arXiv ID:** 2608.07005 | [PDF](https://arxiv.org/pdf/2608.07005v1)

**作者:** Yisheng Li `[一作]` (University of Hong Kong), Fu Zhang `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种实时全身运动规划框架，能在拥挤环境中安全地携带任意形状的大型载荷。

**💡 创新点**

创新点在于链式分解的基于核的碰撞检查保持真实几何并显著降低存储复杂度；以及引入的连杆耦合扫掠体签名距离场（KC‑SVSDF），解决多连杆系统梯度不一致导致的优化停滞。

**🔧 技术方法**

使用前端链式核碰撞检查、MID‑END连续轨迹预处理、后端基于KC‑SVSDF的轨迹优化（MINCO参数化）、Hybrid A*+RRT*+全身RRT*搜索等技术。

**📊 数据集**

在仿真中使用 Tunnel（结构化走廊）和 Forest（随机木柱）两种场景；在真实实验中使用带长方形和T形载荷的Agilex平台。

**📈 对比分析**

与 REMANI 与 TOPAY 两个现有基于球面近似的优化方法比较，实验显示本方法成功率最高、规划时间最短，并且在三大关键障碍物上最小SVSDF值更大，表明碰撞安全性更好。

**⚠️ 局限性**

局限性主要在于对前端初始轨迹的依赖，易陷入局部极小值；未来计划加入拓扑路径搜索以提供更多样化的初始猜测。

---

## 4. Multimodal Drivers' Emotion Recognition and Safety-Oriented Intervention for Intelligent Transportation Systems

**arXiv ID:** 2608.06378 | [PDF](https://arxiv.org/pdf/2608.06378v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 5. An Agentic Hybrid Top-Down and Bottom-Up Approach to Knowledge Graph Generation

**arXiv ID:** 2608.07023 | [PDF](https://arxiv.org/pdf/2608.07023v1)

**作者:** Emma Jouffroy `[一作]` (Malt), Marc Palyart `[通讯]` (Malt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种混合知识图谱生成管道，利用大语言模型与Wikidata进行实体对齐，并通过多代理反射循环自动捕捉并集成多语言长尾技能，构建可解释、可自愈的技能知识图谱。

**💡 创新点**

创新点在于将LLM与事实KG深度绑定，实现Top‑Down的稳定性与Bottom‑Up的灵活性结合；通过Agentic Reflection动态生成缺失实体与关联元数据，形成自愈、可解释的多语言知识图谱。

**🔧 技术方法**

技术包括Gemini 1.5 Flash LLM、Wikidata KG接入、跨语言canonicalization、实体对齐与去重、主动修正（主动反思）以及多代理循环自我校正。

**📊 数据集**

使用了Malt平台的36,037条多语言技能声明，结合Wikidata实体以及平台共现统计作为背景数据。

**📈 对比分析**

评估采用手工金标准，前置阶段对齐覆盖率79.7%，Found Coverage 84.9%，错误率19.1%；相较传统ESCO或聚类方法，性能更优且保持跨语言一致性。

**⚠️ 局限性**

局限性包括依赖Wikidata的更新滞后、对非英语专业术语可能过度归一化、性别偏差未完全消除，以及整合与自愈阶段尚需进一步验证鲁棒性。

---

## 6. ADIAS: Automated Design of Interactive Agentic Systems

**arXiv ID:** 2608.06410 | [PDF](https://arxiv.org/pdf/2608.06410v1)

**作者:** Lekang Jiang `[一作]` (University of Cambridge), Yiwen Guo `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于持续问题状态的自动化代理设计框架 ADIAS，利用问题追踪实现循环修复与聚焦代码修改。

**💡 创新点**

创新点在于将跨轮经验组织从候选代理转为问题导向，并将问题状态作为优化控制信号。

**🔧 技术方法**

主要技术包括持久化问题状态管理、问题关联与生命周期转移、基于问题的计划与全代码聚焦修复。

**📊 数据集**

使用了五个文本交互基准：Tau-Bench、ALFWorld、TextCraft、WebShop、ScienceWorld。

**📈 对比分析**

与多种基准方法对比，ADIAS 在所有基准上获得最高得分和交互效率，平均提升 25.2% 并在不同主干模型上保持优势。

**⚠️ 局限性**

主要限制在于诊断准确性与多模态、长时序任务的适用性尚未验证。

---

## 7. MISO: Model-Internal-State-Guided Optimization for Ranking Models

**arXiv ID:** 2608.07035 | [PDF](https://arxiv.org/pdf/2608.07035v1)

**作者:** Yongzhe Zhang `[一作]` (Meta Inc), Santanu Kolay `[通讯]` (Meta Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了MISO系统，利用模型内部状态（参数、激活、梯度等）来指导广告排名模型的局部优化

**💡 创新点**

创新点在于将内部状态抽象为可聚合的信号（排名、对齐、比较），并通过闭环工作流程在有限实验预算下生成可解释的优化建议

**🔧 技术方法**

使用内部状态提取接口、聚合原语（Ranking、Alignment、Comparison）以及基于预算的决策循环；对内部信号进行梯度与扰动评估

**📊 数据集**

在大规模广告推荐工作负载中，使用含亿级样本、稠密与稀疏特征的数据集进行实验

**📈 对比分析**

与黑盒扩容和专家驱动调优对比，MISO在四个模型规模上相较专家调优提升了约2×的归一化熵，并将训练试验次数减少了84–94%

**⚠️ 局限性**

局限包括：仅在广告排名模型上验证，提取开销高，对噪声信号敏感，且未覆盖全新架构搜索

---

## 8. Self-Healing 6G Networks-in-Network for Resilient Wireless Communication

**arXiv ID:** 2608.06423 | [PDF](https://arxiv.org/pdf/2608.06423v1)

**作者:** Daniel Lindenschmitt `[一作]` (RPTU Kaiserslautern-Landau), Hans D. Schotten `[通讯]` (RPTU Kaiserslautern-Landau)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

演示了一种 6G nin 自愈网络架构，将频谱异常检测与动态频谱管理紧密耦合，实现了在干扰出现时自动重新配置频谱。

**💡 创新点**

创新点在于将频谱异常检测结果直接驱动动态频谱管理，实现检测到干扰后即时自动迁移至可用频谱，消除了传统手工干预的延迟。

**🔧 技术方法**

使用了 SDR（USRP B210）进行频谱扫描、k‑means 无监督聚类算法进行异常检测、MQTT 进行组件间通信、以及动态频谱管理算法在 LattePanda 上执行。

**📊 数据集**

训练数据来源于正常运行时的频谱功率序列，未使用标注异常数据；主要使用实时的信号功率统计特征进行无监督学习。

**📈 对比分析**

该演示未进行与其他方法的定量比较，主要展示了实时检测与自动迁移的流程；性能评估以演示过程中的可视化反馈为主，没有给出具体吞吐率或延迟指标。

**⚠️ 局限性**

局限性包括仅测试单一窄带干扰场景、使用两路固定频率、未覆盖宽带干扰、多 SN 并发干扰或 DSM 故障情况，且缺乏大规模部署与多样化干扰条件下的验证。

---

## 9. Gated-BEPO: Confidence-Gated Bellman Credit Assignment for Large Language Model Agents

**arXiv ID:** 2608.06861 | [PDF](https://arxiv.org/pdf/2608.06861v1)

**作者:** Hongxi Yan `[一作]` (Beihang University), Qingjie Liu `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为 Gated-BEPO 的无价值网络（critic-free）方法，通过构建经验回放图并使用 Bellman 固定点估计器来生成细粒度的步级信用，并通过置信门控（confidence gate）动态融合全局与局部信用，提升大语言模型代理在长时序稀疏奖励环境中的训练效率。

**💡 创新点**

创新点在于：①利用经验回放图对状态转移进行聚合并用均值 Bellman 备份求解固定点，从而得到无模型的局部优势；②通过置信门控根据是否观察到多条后继路径来决定何时使用步级信用，避免不可靠的局部估计污染策略更新；③将步级优势与全局轨迹收益按动态比例混合，兼顾即时与未来奖励。

**🔧 技术方法**

技术细节包括：经验回放图构建、均值 Bellman 固定点迭代、基于 GAE 的残差传播、置信门控与动态优势混合、PPO 贴边目标优化，以及对回报的标准化处理。

**📊 数据集**

在三个典型环境上进行实验：WebShop、ALFWorld（文本/多步骤任务）以及 6×6 视觉 Sokoban（视觉-语言任务），使用 Qwen2.5-1.5B/7B-Instruct、Qwen2.5-VL-3B 等大语言模型。

**📈 对比分析**

与现有无价值网络方法（GRPO、GAGPO、GiGPO、HGPO 等）及带价值网络的 PPO 进行对比。Gated-BEPO 在 WebShop 的成功率提升约 3–4%，在 ALFWorld 的成功率提升约 2–3%，在视觉 Sokoban 的成功率提升约 13%，同时在所有任务上均能跑得更快、更高并保持较低的额外计算开销。

**⚠️ 局限性**

局限性包括：①对状态分支的依赖，若环境中缺乏可观测的多条后继路径则置信门控会关闭，导致无法充分利用步级信用；②对经验图规模的敏感性，极大回放图或高维状态可能导致 Bellman 固定点迭代变慢或不收敛；③方法仍假设奖励稀疏且环境可被状态哈希唯一标识，面对连续状态或密集奖励时可能表现欠佳。

---

## 10. Don't `Well, Actually' Me Unless You Know What You're Talking About: Weak Presupposition Verification Degrades General QA Performance

**arXiv ID:** 2608.06539 | [PDF](https://arxiv.org/pdf/2608.06539v1)

**作者:** Shenran Wang `[一作]` (University of British Columbia), Hila Gonen `[通讯]` (University of British Columbia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM在检测与纠正问题中假设性前提（FPQ）与正常前提（TPQ）时的表现平衡问题，系统评估了多种FPQA方法的效果并揭示了事实核查组件的弱点导致的FPQ/TPQ性能权衡；

**💡 创新点**

发现提高FPQ识别率往往会削弱TPQ表现，事实核查模块的过度拒绝真实前提是主要原因，并提供了真实场景FPQ比例估计与加权评估框架；

**🔧 技术方法**

采用多模型（Gemini‑3‑flash、Gemma‑3‑E4B‑it、Qwen2.5‑7B‑Instruct等）和多技术（直接QA、FP识别、Prompt‑Tuning、Decompose‑then‑Fact‑Check、FAITH、Fine‑tune、RAG等）进行对比实验；

**📊 数据集**

使用四个FPQA基准（(QA)²、CREPE、Syn‑QA²、Cancer‑Myth）以及从WildChat手工标注的FPQ/TPQ比例，构建外部证据库（网页抓取、Wikipedia等）；

**📈 对比分析**

实验显示在所有设置下，FPQ表现最好的方法（如FP Identification、Decompose‑then‑Fact‑Check、GEPA）在TPQ上表现最差，而Direct QA在加权真实分数下表现最好，说明当前FPQA方法普遍存在公平性缺陷；

**⚠️ 局限性**

主要局限包括：事实核查所用证据来源有限（未覆盖更广泛的网络搜索或专业数据库），对不同领域FPQ比例估计不充分，且使用的评判器依赖于LLM（可能引入偏差）。

---

## 11. Multi-Perspective Triad Interaction Graph Neural Network for Cognitive Distortion Detection

**arXiv ID:** 2608.06785 | [PDF](https://arxiv.org/pdf/2608.06785v1)

**作者:** Jun Seo Kim `[一作]` (Gachon University), Hye Hyeon Kim `[通讯]` (Yonsei University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 MTI-GNN 模型，利用 Beck 认知三角（自我、世界、未来）将每个对话语句拆解为四个视角，并在图神经网络中编码这些视角，进一步通过三角互动模块和原型引导融合实现情绪扭曲检测。

**💡 创新点**

创新点包括：① 把认知三角作为结构化的、互相影响的视角融入文本分类；② 设计顺序的三角互动（自我→世界→未来）和特征门控机制；③ 采用原型引导的标签条件融合；④ 通过标签扩展监督充分利用多标签信息。

**🔧 技术方法**

核心技术：LLM（GPT‑4o‑mini）进行三角视角抽取；BGE‑M3 多语言词向量；Graph Attention Network（GAT）编码视角图；三角互动模块（源条件更新与特征门控）；原型引导视角融合；标签扩展的交叉熵训练。

**📊 数据集**

使用四个公开多语言数据集（TherapistQA、SocialCD‑3K、KoACD、Cognitive Reframing），共 9,764 条样本，涵盖韩语、英语和中文，统一标准化为十个情绪扭曲标签。

**📈 对比分析**

与三种监督变体（Text‑only MLP、Text+Triad MLP、MP‑GNN）以及八个生成模型（GPT‑4o‑mini、GPT‑5.4‑mini、Claude Sonnet 5、Qwen3.5‑9B、Qwen3.6‑27B、Gemma‑4‑31B、GLM‑4‑9B、GLM‑4‑32B）进行对比。MTI‑GNN 在 Weighted‑F1 上分别比监督基线提升约 3‑4%（从 0.5430 提升至 0.5560），并在所有生成模型下均表现更好（最高 0.5560 对比生成模型最低 0.4232）。

**⚠️ 局限性**

局限性：① 数据集间异质性导致跨域性能波动；② 仅使用标准化标签，忽略标签共现关系；③ 三角视角由 LLM 零样本抽取，验证样本有限；④ 专家标注一致性低（κ≈0.3）；⑤ 生成模型对比仅限零/少样本提示，未覆盖微调或参数高效方法；⑥ 三角互动采用单向一次更新，缺乏迭代或双向解释。

---

## 12. Automated item evaluation: Predicting item acceptance and rejection using LLM-generated critiques

**arXiv ID:** 2608.06609 | [PDF](https://arxiv.org/pdf/2608.06609v1)

**作者:** Hotaka Maeda `[一作]` (Smarter Balanced), Yikai Lu `[通讯]` (University of Minnesota)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种基于Transformer的自动化项目评估模型，利用标准化考试项目的原始文本和LLM生成的批评文本来预测项目是否能进入正式运营，旨在实现近乎全面的项目接受/拒绝判定；

**💡 创新点**

创新点在于将原始项目文本与LLM（Qwen3）生成的批评文本进行特征融合，首次将开发者评论作为拒绝原因进行多类别评估，并在大规模真实项目拒绝数据上训练单一模型，覆盖多种质量维度；

**🔧 技术方法**

采用DeBERTaV3‑large进行LoRA微调；使用Qwen3生成批评文本；构建融合模型并采用加权交叉熵；低阈值策略提升召回率；情感分析（SiEBERT）用于解释批评文本对模型的影响；

**📊 数据集**

数据集为52,759个中学英语与数学项目，包含34%永久拒绝（30% ELA，39% 数学），并记录了10类拒绝原因；

**📈 对比分析**

与零样本、仅文本、仅批评、融合以及按学科分离的模型对比，评价指标为AUC、准确率、精确率、召回率和F1。融合模型整体AUC 0.80、F1 0.64，数学F1 0.73、ELA F1 0.51；降低阈值至0.25可将召回率提升至0.90，但特异度下降；

**⚠️ 局限性**

局限性包括对偏见、敏感性、公平性等问题识别效果差；无法利用段落、评分细则等非文本信息；对缺失或非文本拒绝原因无预测能力；需要人工审核以捕捉不可从文本直接推断的质量问题；缺乏对项目相似度和难度分布的评估。

---

## 13. On a General Theoretical Framework for Radio Frequency Fingerprint-Based Authentication

**arXiv ID:** 2608.06805 | [PDF](https://arxiv.org/pdf/2608.06805v1)

**作者:** Yuanyu Zhang `[一作]` (Xidian University), Xiaohong Jiang `[通讯]` (Future University Hakodate)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个将射频指纹（RFF）形成与演化模型与认证属性分析相结合的通用理论框架，用以解释RFF在设备身份验证中的可靠性和应用条件；

**💡 创新点**

首次系统性地将RFF形成过程与四大认证属性（唯一性、稳定性、可区分性、不可伪造性）联系起来，并提出通信‑认证共设计的系统视角；

**🔧 技术方法**

基于物理层模型、链式演化分析、认证属性理论分析与共设计原则的理论推导；

**📊 数据集**

本文未使用任何特定数据集，主要是概念性与理论性阐述；

**📈 对比分析**

由于缺乏实验验证，本文没有进行方法对比或性能评估，主要以理论分析与示意图说明其可行性；

**⚠️ 局限性**

局限性在于缺少实测验证与量化指标，理论框架尚需进一步实验验证与细化。

---

## 14. Scalable Long-Horizon Planning with Staggered Updates for Lifelong MAPF

**arXiv ID:** 2608.06702 | [PDF](https://arxiv.org/pdf/2608.06702v1)

**作者:** Vaibhav Sanjay `[一作]` (Carnegie Mellon University), Jiaoyang Li `[通讯]` (Carnegie Mellon University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 PUSH 的终身多智能体路径规划算法，能够在千级甚至十万级代理数量下，在严格的实时约束下生成长视窗的无碰撞路径。

**💡 创新点**

创新点在于将 TP 的子集规划、RHCR 的滚动窗口和 EPIBT 的优先级继承三种思想融合：通过子集规划降低计算量，利用滚动窗口实现长视窗推理，并采用递归优先级推挤与局部约束记录来处理高拥堵场景，从而实现一般地图下的可扩展长视窗规划。

**🔧 技术方法**

核心技术包括：1）基于 EPIBT 的递归优先级推挤和回溯；2）窗口化空间-时间 A* 用于生成 W 步路径；3）使用全局 fallback 路径集合 Π 保障规划可行性；4）局部任意时间 LNS 优化；5）子集规划策略以及对规划窗口大小的固定控制。

**📊 数据集**

使用 MAPF benchmark 中的多种地图：随机 32×32、sortation、warehouse-large/warehouse-small、Symbotic 以及 delivery 地图，并在这些地图上设置不同的任务完成时间（TCT）来模拟实际仓储场景。

**📈 对比分析**

在固定 1 秒/时间步的规划预算下，对比 RHCR‑PBS、PIBT、EPIBT‑LNS 等基线，评估吞吐量（每时间步完成的目标数）和决策时间。实验结果显示，PUSH 在 10k 代理规模下仍保持高吞吐，吞吐量比 EPIBT‑LNS 提升 25%–300%，在复杂地图和高 TCT 场景中优于 RHCR 与 EPIBT，且决策时间在可接受范围内。

**⚠️ 局限性**

局限性包括：1）规划窗口 W 固定，缺乏动态适应；2）递归推挤仅能一次性推挤一个低优先级代理，可能限制在极端拥堵下的可行性；3）未考虑运动学约束（速度、加速度）和严格交付时间；4）在某些极度拥堵或特殊拓扑下仍可能出现死锁或性能下降。

---

## 15. A Pairwise-Error-Probability Framework for One-Shot Information Theory

**arXiv ID:** 2608.06577 | [PDF](https://arxiv.org/pdf/2608.06577v1)

**作者:** Nir Elkayam `[一作]` (Tel Aviv University), Meir Feder `[通讯]` (Tel Aviv University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文基于随机化的配对误差概率（PEP）构建了一套一拍（有限块长）信道编码框架，利用PEP的均匀性把编码的可实现性和逆定理统一到同一个错误谱上，适用于任意解码度量。

**💡 创新点**

创新点包括：① 用PEP和随机化分段规则获得严格的概率积分变换；② 推导出两种变分身份（Neyman–Pearson β-函数和反向信道）在任意度量下仍成立；③ 在匹配ML下证明错误谱在测试水平与输入先验上联合凸；④ 将最小化先验的meta‑converse转化为有限维线性规划；⑤ 对随机编码上限进行凸化并给出显式梯度，使其可通过一次性梯度法求解；⑥ 将上述框架与经典PPV、Han、Matthews等界限统一为同一语义。

**🔧 技术方法**

使用技术包括：随机化的dither分段PEP、概率积分变换、变分身份（Neyman–Pearson β-函数、反向信道）、凸分析与线性规划、类型压缩、Frank–Wolfe梯度法（单步投影梯度下降）。

**📊 数据集**

实验数据仅来自理论模型：AWGN信道和二进制Z信道的数值演示（BSC示例用于说明对称性）。没有外部数据集。

**📈 对比分析**

方法对比：用同一错误谱同时评估可实现性（RCU+、改进的随机编码界）与逆定理（meta‑converse、线性规划逆定理），在AWGN信道下得到约0.009 bits的收敛幅度（n=200，ε=10⁻³），在Z信道上显示先验优化可进一步降低误码概率。匹配ML情况下均匀先验在Gallager‑对称信道上最优。

**⚠️ 局限性**

局限性：① 对多终端或极大化误匹配度量时失去凸性；② 线性规划和类型压缩的复杂度随字母表大小呈指数增长；③ 错误谱对数凸性的假设在某些非对称信道上可能不成立；④ 目前仅适用于点对点记忆无关信道，尚未推广到源‑信道联合或失真编码。

---

## 16. Improved Algorithms for Learning Fourier-sparse Signals

**arXiv ID:** 2608.06385 | [PDF](https://arxiv.org/pdf/2608.06385v1)

**作者:** Dongrun Cai `[一作]` (University of Science and Technology of China), Yile Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在连续时间窗口[-T,T]下，用有限噪声观测学习具有任意离散频率的k-稀疏傅里叶信号，并输出逼近信号的稀疏插值。

**💡 创新点**

主要创新在于：① 通过改进频率平移误差上界，将频率网格间距压缩至O(1/k^2)，从而把采样复杂度从先前的O(k^4)降至O(k^2)；② 引入新的频率聚类与过滤器分析，提升频率估计误差至O(k^2.75/T)，最终实现更高效的插值算法；③ 在假设Chebyshev多项式极值增长极限的前提下，进一步将采样复杂度压至O(k^3)。

**🔧 技术方法**

使用的核心技术包括：改进的频率平移误差估计（利用积分算子和Poincaré不等式），构造具有紧支撑且对信号几乎完美保留能量的过滤器H_k，聚类分层频率划分并证明各簇在过滤后几乎正交，以及对插值误差的多项式近似分析，最后利用线性回归求解低阶多项式系数。

**📊 数据集**

本工作为理论分析，未使用具体实验数据集；所有结果均在理论模型与噪声假设下给出。

**📈 对比分析**

与之前的上界O(k^4·(log kFT)^O(1))相比，新算法的采样复杂度提升到O(k^2·(log kFT/ε)^2)，仅低一阶多项式；在效率上，采样量为O(k^3.75)且运行时间为O((k^3.75)^ω)，相比之前O(k^4)^ω有显著加速。若满足Conjecture 1，则采样复杂度可进一步降至O(k^3)，几乎逼近信息理论下界Ω(k log FT)。

**⚠️ 局限性**

主要局限性：采样复杂度仍比下界多一个k因子，无法完全达到最优；在最优方案下依赖于尚未正式证明的猜想；算法对频率网格、时间窗口、噪声水平的假设较为严格，且对噪声的ℓ_2范数限制较强；实现复杂度高，涉及大量的频率网格枚举和高阶矩阵运算。

---

## 17. Investigating the Presence and Development of Student Instructor Preferences in a Large-Scale CS1 Course

**arXiv ID:** 2608.06782 | [PDF](https://arxiv.org/pdf/2608.06782v1)

**作者:** Yiqiu Zhou `[一作]` (University of Illinois Urbana-Champaign), Geoffrey Challen `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了 CS1 课程中学生对多位教师的偏好形成与发展，利用在线平台记录学生对教师视频的选择，量化偏好得分并跟踪其随学期变化。

**💡 创新点**

创新点在于：①通过微观级别的选择日志实时追踪偏好动态，而非传统一次性调查；②设计了支持多位教师并保持内容一致的教学平台；③揭示了性别与先前编程经验对偏好演化的细微影响。

**🔧 技术方法**

采用了日志追踪技术、偏好得分公式（差分比例）、Kruskal‑Wallis 与卡方检验等统计方法来评估偏好与学生特征的关联。

**📊 数据集**

数据集包含 1221 名 CS1 学生（2021 与 2022 两个学期）的在线学习日志（视频观看记录）以及预课程调查收集的性别与自评编程经验数据。

**📈 对比分析**

通过比较课程初期（前 20% 记录）与学期末偏好得分，检验偏好变化；使用卡方与 Kruskal‑Wallis 检验评估性别和经验对偏好的影响，结果显示偏好随时间演化、性别与经验显著相关，但效应量较小。

**⚠️ 局限性**

局限性：仅基于视频观看数据，未涵盖其他学习行为；研究仅在单门课程、单校情境，教师多样性有限；未深入探讨教学风格等潜在混杂变量。

---

## 18. Stationarity is not enough: tightness of the quantum mechanical bootstrap and the copositive cone

**arXiv ID:** 2608.07047 | [PDF](https://arxiv.org/pdf/2608.07047v1)

**作者:** Daniel Keren `[一作]` `[通讯]` (University of Haifa), Daniel Keren (University of Haifa)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究量子力学bootstrap在不同约束下的紧致性，证明二维双阱在驻态约束下在第三级不紧致，而能量本征约束似乎在各种维度和截断级别下保持紧致；

**💡 创新点**

创新点在于首次用精确有理算术证明了驻态bootstrap存在gap，并揭示了能量本征约束通过限制高阶动量矩导致的Archimedean‑型边界，从而解释了该约束为何能关闭gap；

**🔧 技术方法**

使用了半正定规划（SDP）与线性规划相结合的双线性搜索技术，结合Helton定理、Diananda定理以及Reznick多项式正定性判定等理论工具；

**📊 数据集**

数据集包括二维四次双阱（V=20[(q1^2−1)^2+(q2^2−1)^2]−q1^2q2^2），以及多达58个对称/非对称的五维环形四次势；

**📈 对比分析**

对比方法为在每个可行点上对完整的非负非SOS多项式族求最小期望值；在所有检验设置（除d=5,K=2的非下界势外）均无负值，表明能量本征bootstrap在实践中表现为紧致；

**⚠️ 局限性**

局限性在于仅对有限维度、有限截断级别（最多K=3）进行数值搜索，未给出一般性证明；高阶动量仍未完全受限，且结果仅在有理算术下验证，缺乏对更复杂势能或更高维系统的理论保证。

---

## 19. Certified Feedforward Tracking for Unknown Nonlinear Systems via Invertible Neural Networks

**arXiv ID:** 2608.06419 | [PDF](https://arxiv.org/pdf/2608.06419v1)

**作者:** Berk Altiner `[一作]` (University of Minnesota--Twin Cities), Kenneth Kim `[通讯]` (DECOM Army Research Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究未知非线性系统的前馈跟踪控制认证，使用可逆神经网络在频域建模并通过合成预测提供有限样本概率保证。

**💡 创新点**

① 采用可逆残差网络（i-ResNet）消除非凸逆问题，只剩建模误差；② 在频域实现前馈控制；③ 结合合成预测与Lipschitz连续性给出概率跟踪误差上界。

**🔧 技术方法**

可逆神经网络（i-ResNet）、频域变换、合成预测（Split CP）、Lipschitz常数估计、固定点迭代求逆。

**📊 数据集**

通过仿真生成的1.28M点输入输出数据集，包含0–5 Hz三重正弦组合，用于训练DC‑电机驱动机械负载的非线性摩擦模型。

**📈 对比分析**

与传统需要非凸优化逆的前馈方法相比，无需求解逆优化；数值实验显示跟踪误差满足理论上限（最大误差≈0.495°，理论上限≈0.462°），表明方法可行且性能符合预测。

**⚠️ 局限性**

假设参考轨迹与训练数据同分布；需要已知Lipschitz常数；频域截断导致高阶谐波无法补偿；Lipschitz常数估计未完整展开，可能保守；仅在仿真验证，缺乏真实硬件实验。

---

## 20. Switched Reading: Toward Seamless Visual-Auditory Switching When Reading Text in Augmented/Mixed Reality

**arXiv ID:** 2608.06985 | [PDF](https://arxiv.org/pdf/2608.06985v1)

**作者:** Kazuyuki Fujita `[一作]` (Tohoku University), Yoshifumi Kitamura `[通讯]` (Tohoku University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在增强/混合现实（AR/MR）环境下提出并实现了“Switched Reading”系统，该系统通过眼动检测实现视听切换，以支持用户在视觉不可用时继续阅读。

**💡 创新点**

核心创新在于（1）眼动定位的语音回放技术，使声音从用户实际注视位置开始；（2）对应感知过渡效果，在切换时高亮对应文本以保持语义连续性。

**🔧 技术方法**

实现技术包括 Meta Quest Pro 的眼动跟踪、Unity 3D 开发、Microsoft Azure 语音合成以及自研的眼动滤波与文本匹配算法。

**📊 数据集**

数据集为 GPT‑4o 生成的约 7000 字日语文本（含手工润色）以及基于该文本自动生成的三道选择题，用于测评阅读理解。

**📈 对比分析**

通过 16 名受试者的受控 VR 试验（2×2 设计：眼动回放 vs 位置回放，过渡效果 vs 无），发现眼动回放显著提升阅读速度（≈ + 40 %）并大幅缩短视听偏移；过渡效果虽未显著影响速度或理解，但被受试者主观认为更易保持语境。

**⚠️ 局限性**

局限性包括：实验环境过于受控、切换间隔固定、缺乏复杂外部干扰、眼动误差可能导致起始不准、仅使用日语文本、仅处理纯文本而非图表或多媒体内容。

---

## 21. Duration-constrained Interval Joins

**arXiv ID:** 2608.06856 | [PDF](https://arxiv.org/pdf/2608.06856v1)

**作者:** Naoya Ehara `[一作]` (University of Osaka), Daichi Amagata `[通讯]` (University of Osaka)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

针对区间数据的 duration‑constrained interval join 问题，提出了一种新的阈值网格索引算法，并在此基础上引入了两种优化技术和批量处理机制，显著减少不必要的区间比较和结果过滤工作。

**💡 创新点**

创新点包括：① 基于阈值的两维网格索引，实现对重叠区间的快速判断；② 两个基于最小/最大端点的剪枝规则，进一步避免单个区间比较；③ 利用批量处理（grouping）在区间密集场景下一次性处理相似区间，提高缓存命中率和计算效率。

**🔧 技术方法**

核心技术：二维网格索引与多维排序数组、二分查找、嵌套循环连接、阈值推理与批量处理；实现使用 C++，并在实验中对比了 FS、RD‑index、Rel 等现有方法。

**📊 数据集**

实验使用了三套真实区间数据集：BTC（比特币历史价格区间）、Books（Aarhus 图书馆借阅周期）和 Renfe（西班牙高速铁路行程）。

**📈 对比分析**

通过预处理时间、内存占用和 join 时间等指标与 FS、RD‑index、Rel 进行对比，实验结果显示：我们的算法在所有设置下均优于基线，平均 join 时间约为最佳基线的一半；在不同 |R|/|S| 与 ϵ 变化时均保持稳健性能；消除优化后性能下降明显，证明优化措施有效；批量处理在密集数据集（Books、Renfe）中进一步提升速度。

**⚠️ 局限性**

限制与挑战：① 需要对 S 进行预处理并构建网格结构，导致额外的构建时间和约 2 倍的内存占用；② 批量分组采用启发式方法，NP‑hard 归约证明，难以获得最优分组，且在区间稀疏场景下批量处理反而可能成为性能瓶颈；③ 目前实现仅为单线程，尚未充分挖掘并行化潜力。

---

## 22. Shape Your Feed: An LLM-based Agentic System for Conversational Recommendation

**arXiv ID:** 2608.06632 | [PDF](https://arxiv.org/pdf/2608.06632v1)

**作者:** Ziyun Xu `[一作]` (Meta Platforms), Linhong Zhu `[通讯]` (Meta Platforms)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个名为Shape Your Feed (SYF)的多模态LLM驱动的对话式推荐框架，实现用户在实时交互中以文本、语音和UI控件来主动指令和调整内容流；

**💡 创新点**

创新点包括：①把LLM作为“代理”整合进三层架构（Perception、Serving、Self‑Evolution），实现持续的语义档案更新和即时重排序；②通过“Agentic Refinement”在候选池上做增量检索、对齐评分与剪枝，确保推荐既符合用户显式意图又保持高基线点击率；③采用双重反馈（用户行为 + LLM‑Judge集成）进行SFT‑DPO的对齐优化，解决稀疏监督和低延迟两大工业约束；

**🔧 技术方法**

技术栈主要包括：LLM（Llama‑3‑8B）做意图识别、语义档案更新、对齐评分；多模态交互接口（文本/语音/Context‑Aware Pills）；异步候选检索与轻量级基线评分模型；列表式对齐评分推理；双重监督训练（SFT + DPO）；在生产系统中集成至现有排名管线；

**📊 数据集**

使用的数据集包括：①内部生产流量的历史交互日志（用于SFT标注与在线反馈）；②由LLM‑Judge集成生成的〈SemanticProfile, Candidate〉标签集合；③在线A/B测试的真实用户流量（US/Canada 18+）；以及对比实验所需的传统排名信号（P(CSL)、P(CSM)）。

**📈 对比分析**

方法对比：在离线评估中，SFT + DPO模型在准确率、精确率、召回率与F1上分别达98.85%、79.16%、79.41%、79.51%，相较于few‑shot基线提高约15%；在在线A/B测试中，SYF相较于生产基线降低1.70%发帖驳回率、2.74%发帖不喜欢率，并提升0.16%新兴趣消费率，显示显著的用户体验和内容相关性提升。

**⚠️ 局限性**

局限性：依赖用户主动的自然语言或UI交互，主动控制比例低时，系统仅对主动用户生效；目前对低信号或不常交互用户的语义档案更新不足，未来需要引入主动倾向挖掘与迁移学习来弥补。

---

## 23. CyberForge: Verified Vulnerability Injection at Repository Level for Cybersecurity Agent Training

**arXiv ID:** 2608.06471 | [PDF](https://arxiv.org/pdf/2608.06471v1)

**作者:** Amine Lbath `[一作]` (National Institute of Standards and Technology), Dinesh Manocha `[通讯]` (University of Maryland)

**通讯引用:** 40711 | [OpenAlex ID](https://openalex.org/A5004194238)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了可执行、基于仓库级别的安全训练数据集CyberForge，通过在真实C/C++项目中注入并差分验证漏洞。

**💡 创新点**

创新点在于将漏洞注入与差分PoV验证自动化、规模化，解耦数据集增长与公开漏洞速度，生成与真实CVE相似的编辑局部。

**🔧 技术方法**

使用LLM驱动的注入代理、OSS‑Fuzz覆盖信息、CodeQL静态分析、PoV差分验证、LoRA微调等技术。

**📊 数据集**

数据集为80个OSS‑Fuzz C/C++项目，共生成1034条经过验证的漏洞实例，涵盖63类弱点。

**📈 对比分析**

与SEC‑bench和PatchEval进行对比，Fine‑tuning后学生模型在SEC‑bench提升3.3–14.7个百分点，在PatchEval也显著提高，表明跨语言泛化。

**⚠️ 局限性**

局限包括对PoV触发的依赖、注入成功率低、缺少跨语言原始代码、对更复杂漏洞类型覆盖不足。

---

## 24. Can Language Models Imagine Without Seeing? Ekphrasis: Measuring Visual Creative Ideation in Text-Only LLMs

**arXiv ID:** 2608.06967 | [PDF](https://arxiv.org/pdf/2608.06967v1)

**作者:** Hongyu Luo `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并发布了视觉创意构思（VCI）基准，用来评估文本语言模型在生成图像前的可视化方案创意水平。

**💡 创新点**

将VCI定义为既满足任务需求、又具表现力、又相较于模型群体避免常见视觉陈词的预图像计划，并引入Typed Idea Graphs来判定任务特定的新颖性，以及跨模态验证以证明文本计划能被忠实渲染。

**🔧 技术方法**

采用两字段文本计划输出、维度特定清单辅助的成对比较、Bradley–Terry聚合模型，以及Typed Idea Graphs构建新颖性参考。

**📊 数据集**

构建了400个任务样本，覆盖抽象、组合、转换、适应四类，采样来源于CONCRETE、SUBTLEX-US、THINGS、Places365等公开资源。

**📈 对比分析**

通过成对评审得到Usefulness、Expressiveness、Novelty三维BT得分，整体VCI为标准化平均；14个文本模型评测中Gemini 3.1 Pro、GPT‑5.4、Claude 4.7等名列前茅，展示多维表现差异并验证模型在不同任务族的特定优势。

**⚠️ 局限性**

局限性包括受任务分布、语言/文化偏差、所用模型池和渲染器的限制；清单式评审仍存在位置与长度偏差；新颖性评价相对模型群体而非绝对；跨模态验证受渲染质量影响，无法完全代替图像生成评测。

---

## 25. MiCoPro: End-to-End Mixed Precision HW/SW Co-design with HW-aware Proxy Model

**arXiv ID:** 2608.06916 | [PDF](https://arxiv.org/pdf/2608.06916v1)

**作者:** Zijun Jiang `[一作]` (Hong Kong University of Science and Technology), Yangdi Lyu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MiCo 与 MiCoPro 两个端到端混合精度量化（MPQ）探索与部署框架，能够在给定硬件约束下自动寻找最优层级精度组合并生成可直接在加速器或 SIMD 扩展的 RISC‑V CPU 上运行的代码。

**💡 创新点**

创新点包括：①使用随机森林与专门设计的层级正交采样相结合的预测器，提高搜索效率；②提出硬件感知代理（HAP）模型，利用 CBOPs 特征、网络级 min‑max 校准和跨硬件迁移学习，显著提升延迟预测准确性；③将 MPQ 设计与 QAT/PTQ 训练流程无缝衔接，实现低比特宽度（≤4 位）模型的高精度恢复。

**🔧 技术方法**

核心技术包括随机森林准确率预测器、近约束采样（NCS）与遗传算法搜索、BOPs 与 CBOPs 计算量建模、线性与非线性回归的代理模型、QAT/ PTQ 量化训练、PyTorch FX 图转换、C 代码生成与裸机库。

**📊 数据集**

在多种数据集上进行评估，分别使用 CIFAR‑10、CIFAR‑100、MNIST、Speech Commands V2、TinyStories（小型 LLM）和 ImageNet，对多种网络架构（VGG7、ResNet‑18/34、LeNet‑5、CNN4、DS‑CNN、TinyLLaMa 等）进行测试。

**📈 对比分析**

与 HAWQ‑V3、HAQ、BOMP、w‑method 等现有方法比较，MiCo 在 PTQ 与 QAT 场景下均能获得更高准确率，且在 60% BOP 约束下实现 20‑30% 的延迟降低；在 BitFusion 与 VexiiMiCo CPU 上，最终加速率可达 40% 以上，准确率下降不足 3%。

**⚠️ 局限性**

主要限制包括：①需要在目标硬件上收集大量内核延迟数据，虽然转移学习可减轻这一成本；②QAT 训练耗时较长，对大模型和低精度空间效果仍不够理想；③代理模型的预测误差仍存在，尤其在极低精度或非常规硬件上；④框架目前仅支持 INT1/2/4/8 位和特定的加速器/CPU 体系结构。

---

## 26. Graph Machine: Exploring Edge Mechanisms as an Inductive Bias

**arXiv ID:** 2608.06834 | [PDF](https://arxiv.org/pdf/2608.06834v1)

**作者:** Lintai Hou `[一作]` `[通讯]` (Iterlabs), Lintai Hou (Iterlabs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 Graph Machine（GM）结构，融合了基于边的注意力与边的重参照机制，用于改进图形数据的推理。

**💡 创新点**

创新点在于：①将边拆分为特征和地址两部分，形成可解释的软边；②通过边增强注意力与边重参照实现显式的迭代关系遍历；③在单次前向传递中动态重构和更新图结构。

**🔧 技术方法**

使用的技术包括：多头点积注意力、基于 logit 的多专家融合、边地址的温度锐化、稀疏/压缩化的边地址处理、以及对 Sudoku 任务的专门化实现。

**📊 数据集**

使用了 Kaggle 的 Sudoku-3M 数据集，包含 3M 条 Sudoku 谜题，范围从 23 到 26 条线索，难度多样。

**📈 对比分析**

将 GM 与多种 Transformer 基线（含静态边、sinusoidal PE、规模扩大版）进行对比。GM 在标准规模下以约 86% 的单板准确率击败普通 Transformer，且接近 2× 参数量的 Transformer，表明边机制带来显著性能提升。

**⚠️ 局限性**

主要局限包括：计算复杂度为 O(n³)、内存占用为 O(n²)；当前实现仅在 Sudoku 这一相对简单的关系任务上验证，未检验更复杂、动态的图结构；以及对稀疏化、压缩等优化的探索仍不充分。

---

## 27. CEDAR: Agent-Orchestrated Tree Search for Goal-Directed Optimization of Complex Systems

**arXiv ID:** 2608.06871 | [PDF](https://arxiv.org/pdf/2608.06871v1)

**作者:** Yingtao Tian `[一作]` `[通讯]` (Sakana AI), Yingtao Tian (Sakana AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种利用大型语言模型驱动的蒙特卡罗树搜索（MCTS）方法，自动生成满足自然语言目标的复杂系统。

**💡 创新点**

创新点在于将LLM同时用作变异操作器（Editor）和评估器（Judge），与MCTS结合形成自适应搜索框架，并通过统一的受限Python子集实现系统的可读性与可编辑性。

**🔧 技术方法**

使用技术包括LLM Editor、LLM Judge、MCTS与进化搜索的融合、Python子集表示、可扩展的上下文抽样、以及多种LLM后端（Claude Sonnet 4.5、GPT‑5.1）。

**📊 数据集**

数据集为从 DYNAMO 与 STELLA 转译得到的 19–20 个经典社会与生物动力系统（如 World Dynamics、Modeling Dynamic Biological Systems），以及其对应的运行记录。

**📈 对比分析**

与 Optuna 等传统黑盒优化方法比较，本文方法在自然语言目标优化和记录拟合任务上获得更高分数/更低 DTW 距离，并产生多样化的高性能解，表现优于传统方法。

**⚠️ 局限性**

局限包括对 LLM 评估主观性的依赖、评估器与编辑器共享模型导致的循环性风险、实验规模与系统数目有限、缺乏严格统计验证，以及对高质量 LLM 后端的计算成本依赖。

---

## 28. Dirichlet Follow-the-Leader Closes the Gap in Simultaneous Multiclass U-Calibration

**arXiv ID:** 2608.06656 | [PDF](https://arxiv.org/pdf/2608.06656v1)

**作者:** Pahan Dewasurendra `[一作]` `[通讯]` (Johns Hopkins University), Pahan Dewasurendra (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于贝叶斯自举的单步多分类预测器，能够在任何有界的合适损失下实现 
4√(KT) 的最优平均惩罚，并且在所有 β‑光滑合适损失下同时达到 O(βlog T) 的优越表现。

**💡 创新点**

首次通过一个精确的“计数删除”Dirichlet恒等式，融合了对离散、非光滑损失的计数稳定性与对光滑损失的均值中心化，统一实现了两个此前被认为互斥的最优性目标。

**🔧 技术方法**

使用了 Dirichlet 采样、贝叶斯自举、对数算子与几何恒等式、以及对可导性、凸/凹性质的严格分析；核心是对 Dirichlet 取样分布在计数删减时的精确总变分上界。

**📊 数据集**

无实验数据集，全部为理论证明与抽象推导。

**📈 对比分析**

与之前的自适应自扰扰动方法（K⁵⁄⁴√T 及 β√K log K 误差）相比，本工作在多类别设置下完全消除了维度缺口，提供了最优的 √(KT) 误差率，并且在光滑损失下保持 O(βlog T) 的渐进率。

**⚠️ 局限性**

局限在于仅给出期望惩罚的上界，未能控制对无限损失集合的真正 U‑校准（sup‑ℓ 形式），且未考察对日志障碍光滑损失或非零先验下的适用性。

---

## 29. Not Always Top-Left: Untangling the Signals that Guide Dashboard Reading Order

**arXiv ID:** 2608.06845 | [PDF](https://arxiv.org/pdf/2608.06845v1)

**作者:** Nicole Sultanum `[一作]` (Tableau Research), Vidya Setlur `[通讯]` (Tableau Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过访谈和思考录音收集作者和使用者的阅读流程，结合自定义的流程规范工具，对八个不同布局的仪表盘进行实验，归纳了阅读顺序的六个设计因素和七个设计原则，并提出了九种典型阅读流程模式；

**💡 创新点**

首次系统性地将仪表盘阅读顺序从静态布局视角转为动态、可变的使用行为模型，识别了布局、可见性、语义、功能、交互和用户上下文如何共同影响阅读路径；

**🔧 技术方法**

采用半结构化访谈、思考录音、定制的流程可视化工具以及手工注释的组件顺序表，随后对序列与说明进行定量与定性分析；

**📊 数据集**

使用来自七位作者的八份公开 Tableau 仪表盘，参与者包括 13 位作者和 16 位终端用户；

**📈 对比分析**

对作者意图流与用户实际流进行对比，计算各组件在不同实例中的平均顺序与标准差，基于这些统计和用户说明识别共性与变异性；不涉及算法性能评估，仅以模式发现为主要评价方式；

**⚠️ 局限性**

样本偏向业务导向的 KPI 仪表盘，仪表盘数量有限，实验任务仅覆盖首次接触阶段，读取序列为自我报告，未通过眼动或交互日志验证，导致对真实使用行为的可推广性有限。

---

## 30. Target-Weighted Neyman Allocation: Experimental Design for Heterogeneous Treatment Effects under Population Shift

**arXiv ID:** 2608.06512 | [PDF](https://arxiv.org/pdf/2608.06512v1)

**作者:** Hoang Dang `[一作]` (Independent Researcher), Minh Nguyen `[通讯]` (Florida Atlantic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出两阶段分层实验设计TWNA，利用试点方差估计实现目标加权的组效应精度最优分配。

**💡 创新点**

创新点在于给出闭式最优分配公式，并扩展至目标权重不确定与试点小样本/重尾稳健形式。

**🔧 技术方法**

技术方法包括Neyman分配、方差估计、投影投影与鲁棒化（Winsor化、缩放）等统计设计技术。

**📊 数据集**

实验数据集涵盖模拟情形、合成四元协变量基准以及真实协变量的IHDP和LaLonde基准。

**📈 对比分析**

与均匀分配、部署比例、方差比例、HHK等基准比较，TWNA在目标加权GATE MSE上显著优于所有对照，接近理论最优。

**⚠️ 局限性**

局限性包括需预先确定分组、仅处理组间比例偏移、对小样本/重尾试点的依赖以及忽略招募成本、干扰等实际约束。

---

## 31. Faster Query-Key Learning Sharpens Attention in Self-Attention Models

**arXiv ID:** 2608.06776 | [PDF](https://arxiv.org/pdf/2608.06776v1)

**作者:** Rahul Vashisht `[一作]` (IIT Madras), Harish G. Ramaswamy `[通讯]` (IIT Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了自注意力模型中查询-键（QK）与输出-值（OV）电路相对学习速率如何影响注意力聚焦，并给出了理论分析与实验验证。

**💡 创新点**

创新点在于：①通过梯度流分析证明 QK 学习速率高于 OV 时，注意力会更锐利；②将 factorized 与 collapsed 参数化映射到同一梯度流框架；③给出闭式正交数据模型下的量化动力学，并将其与实际多层模型的注意力解释指标相连。

**🔧 技术方法**

主要技术包括：梯度流（gradient‑flow）与深度线性网络理论、参数化映射与预处理矩阵、正交合成数据生成、注意力解释指标（DTAP、AC、ACMC、Sufficiency、Comprehensiveness）以及 AdamW 训练。

**📊 数据集**

使用的数据集：1）基于类特定与通用词的正交合成数据；2）真实任务数据：SQuAD、HateXplain、Subject‑Verb Agreement (SVA) 以及相应的 token‑level relevance 注释。

**📈 对比分析**

比较方法：在基线模型与“加速 QK 学习率”模型间相同任务、相同预测性能（F1/Accuracy）进行对比。实验显示预测性能基本不变或略有提升，而注意力解释指标（AC、ACMC、Sufficiency、Comprehensiveness）均显著改善，证明更高 QK 学习速率能显著锐化注意力。

**⚠️ 局限性**

局限性：①理论分析仅针对单层无层归一化和前馈网络的纯自注意力模型；②依赖正交与无交叉假设，实际数据可能不满足；③锐化注意力并不等价于更可信的解释，仍需进一步研究与验证。

---

## 32. Beyond "AI Language": The case for the idiolectal nature of LLM output

**arXiv ID:** 2608.06589 | [PDF](https://arxiv.org/pdf/2608.06589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 33. Dueling World Models: Advantage-Style Action Channels for Common-Mode Distractor Rejection

**arXiv ID:** 2608.06706 | [PDF](https://arxiv.org/pdf/2608.06706v1)

**作者:** Jiazhuo Li `[一作]` (University of Michigan), Heikichi Hayashi `[通讯]` (Adrasteia Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证一种“动作均值中心化”机制，利用行动均值的减法在潜在世界模型中去除共模干扰，从而保留可控动作通道。

**💡 创新点**

创新点在于将 dueling 分解中的动作优势中心化从标量 Q 值迁移到向量潜在动态预测，实现无额外损失、无重建、无奖励的纯结构性干扰消除。

**🔧 技术方法**

技术包括 Joint‑Embedding Predictive Architecture (JEPA)、信息对比损失（InfoNCE）、VICReg 正则、中心化预测器（B+Δ−Δ̅）以及可选的门控项，但核心仅为一次性减法。

**📊 数据集**

实验数据集覆盖：FourRooms 13×13 网格世界（含可滚动干扰细胞）、合成生成器（已知控制与噪声因子）、DeepMind Control 任务（带像素级移动遮挡）以及 Atari Freeway（天然交通干扰），并在冻结的 RePo 与 TIA 世界模型上进行推理测试。

**📈 对比分析**

通过与未中心化、标准单一预测器以及带门控的变体进行 ablation，对比指标包括动作分离度（AS）、对真实因子（控制位移与干扰位移）的 R² 解释度、以及 GridWorld 的目标达成率；中心化后 R² 在控制因子上提升 10‑30%，干扰因子接近 0，控制成功率在网格世界中从 0.57 提升至 0.92；在像素干扰与 Atari 场景中亦显著恢复可控信号。

**⚠️ 局限性**

局限性：若干扰动态与行动相关（共模假设失效），中心化失效；无法在大规模连续控制任务中直接提升多步规划性能；仅对单步预测有效，需进一步探索累积特征与多步控制的结合。

---

## 34. Beyond Routing Weights: Faithful Response-Level Interpretation of Mixture-of-Experts Reward Models via Contribution Contrast

**arXiv ID:** 2608.06400 | [PDF](https://arxiv.org/pdf/2608.06400v1)

**作者:** Yifan Wang `[一作]` (Saarland University), Vera Demberg `[通讯]` (Saarland University)

**通讯引用:** 4175 | [OpenAlex ID](https://openalex.org/A5023605306)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了CoCo方法，通过贡献对比解释稀疏Mixture-of-Experts奖励模型的专家行为。

**💡 创新点**

创新点在于将路由权重与专家评分的贡献对比结合为解释信号，并将正则化约束迁移至贡献对比，提升解释的真实性与专一性。

**🔧 技术方法**

采用MoE奖励模型、贡献对比正则化、LLM提示生成解释以及一系列自动与人工评测指标进行实验。

**📊 数据集**

使用700K偏好对比数据集和Reddit的SHP数据集进行训练与评估。

**📈 对比分析**

与基于路由、分数或稀疏自编码器的解释方法相比，CoCo在Fidelity、专家–模型一致性、专家准确率等指标上均表现更优，同时保持奖励模型准确性。

**⚠️ 局限性**

局限性包括对模型学习结构的依赖、无法保证专家对应语义清晰维度、评估指标间接且易受数据偏差影响。

---

## 35. The blue pebbling cost and the space in tree-like and negative Resolution

**arXiv ID:** 2608.06443 | [PDF](https://arxiv.org/pdf/2608.06443v1)

**作者:** Lisa-Marie Jaser `[一作]` (Ulm University), Jacobo Torán `[通讯]` (Ulm University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过引入蓝色斑点游戏中的蓝色成本（blue pebbling cost），研究并刻画了树状解析（tree‑like Resolution）与负解析（negative Resolution）的子句空间（clause space）复杂度。

**💡 创新点**

创新点在于：①首次定义蓝色成本，并证明其介于传统黑色斑点与可逆斑点之间；②利用该度量实现树状解析子句空间的精确等价；③给出提升（lifted）铺设公式（_G[∨] 与 _G[⊕]）在负解析中的上界与下界；④构造近最优的空间分离实例，揭示树状解析与负解析之间的巨大空间差距。

**🔧 技术方法**

核心技术包括：红蓝斑点游戏的策略分析、Prover–Delayer 游戏与解析空间的对应关系、负投影（negative projection）与 Hall 定理结合的宽度下界证明、以及白色斑点游戏的逆向构造以得到解析空间与黑斑点数之间的联系。

**📊 数据集**

研究对象为理论构造的CNF公式——经典铺设公式 G 及其提升版 _G[∨]、_G[⊕]，以及其相关的有向无环图（sDAG）结构；未使用具体实验数据集。

**📈 对比分析**

通过理论证明与比较，蓝色成本可精确等价树状解析子句空间；在负解析中，蓝色成本提供了子句空间的近似上下界；在空间分离实验中，构造的公式在树状解析下常数空间，而在负解析下达到 Ω(n/log n) 的空间，体现了与传统结果相当的或更高的分离度。

**⚠️ 局限性**

主要局限性：对于提升铺设公式的空间上下界仍存在小的对数差距；尚未找到三种斑点成本（黑、可逆、蓝）都彼此独立的图族；对负解析与树状解析之间更细粒度的空间与大小关系仍有开放问题。

---

## 36. Two-Phase Phase-Type Queues: Closed-Form Distributions and the Numerical Accuracy Landscape of BuTools

**arXiv ID:** 2608.06414 | [PDF](https://arxiv.org/pdf/2608.06414v1)

**作者:** Yossi Luzon `[一作]` `[通讯]` (Afeka Tel Aviv Academic College of Engineering), Yossi Luzon (Afeka Tel Aviv Academic College of Engineering)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统推导并验证了两阶段 Phase‑Type（PH₂）单服务器队列（M/E₂/1、M/Hypo₂/1、M/Hyper₂/1、M/Coxian₂/1）的完整闭式稳态队列长度分布和平均停留时间分布，并给出易于使用的公式表；

**💡 创新点**

创新点在于将 PK 变换与矩阵谱分解两大传统统一为一套统一的显式表达式，首次提供 PH₂ 所有四类模型的闭式概率分布和停留时间密度，并用它们对 BuTools 软件进行精度基准，揭示了队列长度与停留时间在极端参数下的不同数值失真行为；

**🔧 技术方法**

主要技术包括：Pollaczek–Khinchine 生成函数与拉普拉斯-斯特莱斯变换、部分分式分解、矩阵-解析 QBD 速率矩阵 R 的求解（循环降阶），以及符号计算与数值验证；

**📊 数据集**

本文不依赖外部数据集，而是通过数学推导得到通用闭式公式，随后用 BuTools 在广泛参数空间（ρ≤0.999、C_s²高至10⁴）进行数值验证；

**📈 对比分析**

与 BuTools 的对比表明，队列长度计算在整个参数空间均达机器精度（相对误差≤10⁻¹⁴）；停留时间计算在高负载与高变异性联合出现时误差可降至 10⁻⁶，显示了闭式公式在极端条件下的优势；

**⚠️ 局限性**

主要限制：闭式解仅适用于两阶段服务（PH₂），三阶段及以上因根不可用实根或需复数/三次根导致符号表达式变得不可行；此外，BuTools 的矩阵指数在极端参数下仍存在数值不稳定性。

---

## 37. Lost in Interpolation: Why Predictive Feedback Fails in Diffusion Language Models

**arXiv ID:** 2608.06529 | [PDF](https://arxiv.org/pdf/2608.06529v1)

**作者:** Lavanya Nigam `[一作]` (Indian Institute of Technology Roorkee), Gaurav Kumar Nayak `[通讯]` (Indian Institute of Technology Roorkee)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了球面软掩码（S-SM），用几何化的平均和球面线性插值取代MDLM中的线性插值，以保留嵌入空间的球面几何；

**💡 创新点**

创新点在于识别并修正软掩码中的几何失配——线性插值在球面嵌入上会偏离流形，并通过在球面上计算Frechet平均并使用SLERP实现端点一致、保范数的反馈；

**🔧 技术方法**

技术包括球面投影、Frechet（Karcher）均值、SLERP插值、Riemannian对数/指数映射，以及改进的数值安全策略；

**📊 数据集**

使用OpenWebText数据集，对公开的169M参数MDLM checkpoint进行持续预训练；

**📈 对比分析**

在与无反馈MDLM和TopK/LERP基线的对比中，S-SM在不同NFE预算（64、128、256、512）下平均MAUVE提升约27.7–56.1%，在最高预算下近乎翻倍；同时生成困惑度下降约12–20%，并保持输出熵与基线相近；

**⚠️ 局限性**

局限包括：仅在单一169M规模模型上验证；未重新调优超参数（如时间区间、软掩码激活概率）；仅比较了单一插值族；评估指标限于MAUVE与生成困惑度，未做人工评测。

---

## 38. Generative Embedding Benchmark: How Much Information Survives in a Dense Embedding?

**arXiv ID:** 2608.06972 | [PDF](https://arxiv.org/pdf/2608.06972v1)

**作者:** Yun Li `[一作]` (Fudan University), Wenwu Ou `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Generative Embedding Benchmark（GEB），通过生成式读出评估冻结的多模态嵌入能否保留生成所需的信息。

**💡 创新点**

创新点在于：①将嵌入视作生成器的条件接口，用生成式回答来量化可恢复信息；②设计视觉仅模式和视觉-语言联合模式两种编码，分别考察嵌入的可重用性和查询条件化压缩；③通过对比多模态基准MMEB-V2展示生成式评测提供不同视角。

**🔧 技术方法**

使用冻结的多模态嵌入模型（CLIP、SigLIP、Qwen3-VL-Embedding、VLM2Vec-V2、UME-R1、Embed-RL、Qwen3-VL-Embedding-8B）以及共享的Qwen3-0.6B解码器（adapter+LM），并在同一训练数据（LLaVA‑NEXT 738K）下微调。

**📊 数据集**

构建了包含自然图像、场景文本、视觉文档三类的1,800条开发集和900条测试集，来源于MME、CV‑Bench、RealWorldQA、TextVQA、OCRBench、ChartQA、DocVQA、InfographicVQA等公开数据集。

**📈 对比分析**

对比方法包括匹配嵌入、文本仅输入、零嵌入、打乱嵌入等控制实验；在视觉仅模式下，得分集中在28–33%；在视觉-语言联合模式下，得分提升至48–65%，最高65.56（Qwen3-VL-Embedding‑8B）。与MMEB‑V2的顺序不一致，展示生成式评测揭示新的信息瓶颈。

**⚠️ 局限性**

局限性：生成式读出受解码器容量限制；对场景文本和视觉文档的恢复仍低，说明单一固定维度嵌入难以满足多种信息需求；实验仅覆盖特定任务，缺乏跨语言、多任务的泛化评估。

---

## 39. Benchmarking and Reasoning Distillation of Large Language Models for Feedback Controller Design in Complex Dynamical Systems

**arXiv ID:** 2608.07004 | [PDF](https://arxiv.org/pdf/2608.07004v1)

**作者:** Zhongchao Zhou `[一作]` (University of Tokyo), Yusuke Iwasawa `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `79276348-11e0-48e3-84bc-7ec231d0171c` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了名为 CoDyControlBench 的基准，包含 132 个多自由度、非线性、时变、耦合、阻尼多样的控制系统，并评估六种顶尖 LLM（GPT、Gemini、Claude、GLM、DeepSeek、Qwen）在此基准上的控制器设计能力；同时通过“推理蒸馏”方式从 1.5B 参数的 DeepSeek 模型中训练出轻量化控制专用模型（Think‑Model），并在基准与真实的 PAM 驱动机器人臂上验证其效果。

**💡 创新点**

①首次系统化评测 LLM 在复杂动力学控制器设计上的性能，填补了仅聚焦线性单自由度的 ControlBench 空白；②提出“推理蒸馏”策略，证明在 1.5B 规模下即可获得可在边缘设备部署的控制专用 LLM；③对比不同 LLM、不同控制器类型、不同系统维度的性能差异，揭示“模型基础设计”与“启发式设计”的权衡。

**🔧 技术方法**

使用自然语言提示与统一的系统提示让 LLM 生成 PID/SMC 控制器代码；定义稳态误差、超调、延迟、振荡惩罚等多维度指标计算综合得分；通过 LoRA（4‑bit QLoRA）在 1.5B 模型上进行答案蒸馏与推理蒸馏；采用固定迭代（Fixed‑3/Fixed‑10）与成功停止（Success‑Stop）两种调优策略；对基准案例进行三次独立实验并统计成功率和平均得分。

**📊 数据集**

CoDyControlBench：132 个系统配置（1–6 自由度、四类系统类型、两种耦合、三种阻尼、两种控制器族）；真实实验数据：PAM 驱动的一自由度机械臂的角度、速度传感数据。

**📈 对比分析**

对比六大 LLM 在 Fixed‑3 与 Success‑Stop 两种策略下的成功率与平均得分；GPT 最高 94.8% 成功率；在 1–6 自由度上 GPT 维持 88%+ 成功率；推理蒸馏得到的 Think‑Model 在基准上 54–69% 成功率，且在 PAM 机器人上 100% 成功率；与答案蒸馏（Answer‑Model）和基线 1.5B 模型比较，思维蒸馏显著提升稳定性与跨维度泛化。

**⚠️ 局限性**

①仅覆盖二阶、全致动、状态空间 A 恒定 B 的系统，未考虑输入耦合 B(x,t) 或非致动；②评测数据为合成仿真，缺乏噪声、扰动与测量误差；③边缘部署实验仅在单自由度 PAM 机器人上验证，未检验更高维度或复杂场景；④对 LLM 的迭代次数有限（最多 10 次），无法充分探索更深层调优策略；⑤推理蒸馏的 1.5B 模型在极大自由度下仍显示一定波动，说明对更高维度的鲁棒性仍待提升。

---

## 40. Mathematical Principles and Experimental Discoveries of the Emergence of Symbolic Patterns in Artificial Neural Networks

**arXiv ID:** 2608.06839 | [PDF](https://arxiv.org/pdf/2608.06839v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 41. Cross-View Action Consistency for Camera-Robust Vision-Language-Action Policies

**arXiv ID:** 2608.06965 | [PDF](https://arxiv.org/pdf/2608.06965v1)

**作者:** Bingqi Huang `[一作]` (Tsinghua University), Zhaokui Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究视角变化对流式视觉-语言-动作（VLA）策略的鲁棒性，并提出跨视角动作流一致性约束以提升相机位移下的性能。

**💡 创新点**

创新点在于直接对动作流预测施加跨视角一致性正则化，使同一物理状态在不同相机视角下产生相同动作，从而提升相机扰动鲁棒性。

**🔧 技术方法**

采用流式动作生成网络、跨视角一致性损失、共享权重的双分支 VLA 架构，以及语言与 proprioception 编码器。

**📊 数据集**

使用 LIBERO‑Plus 视角扰动基准（C1/C2/C3 视角变换）以及真实机器人同步多摄像头演示数据集。

**📈 对比分析**

与仅流匹配、混合相机 SFT 及同数据流匹配基线对比，跨视角一致性约束在相机扰动集上提升约 7.4pp（至 87.2%）且保持原始视角性能（95%）

**⚠️ 局限性**

局限性：需要真实的动作等价视角对，单摄像头数据无法直接使用；在单 RGB 条件下无法弥补信息缺失；实验范围仅涵盖 LIBERO‑Plus 视角扰动与有限的真实机器人桌面任务。

---

## 42. BZKO: An Ontology for the Card Index of German Post-War Compensation Records

**arXiv ID:** 2608.06918 | [PDF](https://arxiv.org/pdf/2608.06918v1)

**作者:** Dilek Yargan `[一作]` (FIZ Karlsruhe), Harald Sack `[通讯]` (FIZ Karlsruhe)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了BFO兼容的两层本体（BZKO），用于对德国“Wiedergutmachung”补偿索引卡进行语义化建模，并为主题门户提供知识图谱基础。

**💡 创新点**

采用两层本体架构，将领域本体与标准本体（BFO、NFDIcore、RiC‑O、PROV‑O、PiCo、Schema.org）分离；引入角色与过程边界的显式建模，建立从原始卡片到数字化记录的可追溯性；提供快捷关系和SPARQL映射以简化查询。

**🔧 技术方法**

OWL 语言 + Protégé + Ontology Development Kit (ODK) + 通过 SPARQL 语句实现的语义规则（非 SWRL）+ 质量控制工具和推理器（如 HermiT）。

**📊 数据集**

BZKOpen 手工标注的 516 张已数字化的索引卡数据集；以及中心联邦卡索引的原始卡片（约 1.9 M 张）作为语义化目标。

**📈 对比分析**

论文未给出系统性能量化，仅提出将利用领域专家的竞赛问题和自动推理一致性检查进行验证，尚无基准对比；预期能在主题门户中实现可搜索的知识图谱。

**⚠️ 局限性**

对时间与地点的不确定性、近似性和模糊性的语义化尚未完成；缺少对更多实体如亲属关系、民族、组织等的建模；当前模型在细粒度过程区分和时间/空间表达上有限。

---

## 43. Understand Before Detect: Vision--Language Learning for Omni-Domain Infrared Small Target Detection

**arXiv ID:** 2608.07015 | [PDF](https://arxiv.org/pdf/2608.07015v1)

**作者:** Haoyang Yuan `[一作]` (National University of Defense Technology), Wei An `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于“先理解后检测”理念的多模态框架 JinSight，先通过视觉指令微调的生成式 VLM 学习跨域全景 IR 语义，再利用低秩 Latent Semantic Interaction 将全局语义与局部特征高效融合，最终实现小目标检测。

**💡 创新点**

创新点在于把 IRST 从传统的二分类检测转化为先全景语义理解后检测，并通过语言监督构造跨域不变语义，设计低秩 LSI 以高效融合全局语义与细粒度空间特征。

**🔧 技术方法**

采用 InternVL2.5 生成式 VLM 作为基础，使用 ViT backbone 进行视觉指令微调，加入低秩 LSI 模块，并在 UperNet 解码器上完成语义分割/检测任务。

**📊 数据集**

训练与评估使用自研 OmniIRST‑VL（约 39k 图文对）以及 WideIRSTD、NUST‑SIRST、SIRST‑v2、NUDT‑SIRST、IRSTD‑1K、MIRSTD、FZDT 等公开 IRST 数据集。

**📈 对比分析**

与多种 CNN/Transformer 传统 IRST 检测器以及 InternVL、Qwen3‑VL、LLaVA 等大型视觉‑语言模型在 WideIRSTD 上进行对比，JinSight 在 IoU 上提升 14% 以上，P_d、AP_50 等指标显著优于最新方法。

**⚠️ 局限性**

主要局限在于对人工构造的指令数据依赖较大，模型规模和推理成本高，且对极端光照、极小目标以及极端噪声场景的鲁棒性仍有提升空间。

---

## 44. DREAMS: Diverse Reactions of Engagement and Attention Mind States Dataset

**arXiv ID:** 2608.06382 | [PDF](https://arxiv.org/pdf/2608.06382v1)

**作者:** Monisha Singh `[一作]` (Indian Institute of Technology Ropar), Abhinav Dhall `[通讯]` (Flinders University)

**通讯引用:** 6008 | [OpenAlex ID](https://openalex.org/A5085376429)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了DREAMS数据集，并在该数据集上对面部视频进行自标注的参与度和注意力状态分类，探究二者关系。

**💡 创新点**

创新点在于首次构建面向自然环境的自标注参与度/注意力状态数据集，并通过多任务学习证明注意力更能作为参与度的指示器；同时结合NASA‑TLX评估工作负载与表现。

**🔧 技术方法**

采用OpenFace和MARLIN提取眼动、头部姿态、面部动作单元等特征，使用Transformer架构进行时序学习，并实现单任务、迁移学习和多任务三种训练策略。

**📊 数据集**

使用DREAMS数据集，共32名大学生产生的832段40秒面部视频，包含教育与幽默等多样刺激。

**📈 对比分析**

通过对比单任务、迁移学习和多任务在加权精度、F1和召回率上的表现，发现多任务在参与度预测上普遍优于单任务，迁移学习在参与度上有提升但对注意力无显著改善，整体分类效果在中等水平。

**⚠️ 局限性**

限制在于样本量小、刺激类别单一、仅采用监督学习，且自标注的注意力/参与度标签可能存在主观误差。

---

## 45. The Optimizer Is the Agent: Reasoning-Driven Search across Prompts, Programs, and ML Workflows

**arXiv ID:** 2608.06714 | [PDF](https://arxiv.org/pdf/2608.06714v1)

**作者:** Junbo Li `[一作]` (University of Texas at Austin), Zhewei Yao `[通讯]` (Snowflake)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 ReASearch 框架，让单一 LLM 代理通过工具使用和记忆实现提示、程序和 ML 工作流的自主优化。

**💡 创新点**

创新点在于把传统搜索控制器完全内化为代理推理过程，使用相同的工具和记忆机制在多任务上实现统一的优化策略，并能自然产生类似优化器的行为。

**🔧 技术方法**

采用大语言模型代理、Python/Bash 执行工具、评估/诊断工具、持久记忆和上下文压缩等技术。

**📊 数据集**

使用 14 个任务的数据集，覆盖提示优化（AIME、GSM8K、HotpotQA、Terminal‑Bench）、程序演化（圆形 Packing、Heilbronn 三角形、TXN、EPLB、ARC‑AGI‑2）以及 ML 工作流（NanoGPT、IMG‑100、Atari Q*bert、MuJoCo、Crypto Kaggle）。

**📈 对比分析**

与领域专用基线（GEPA、AdaEvolve、Claude Code 等）对比，ReASearch 在大多数任务中实现 2%~40% 的性能提升，并在圆形 Packing 等任务上突破人类已知最优。

**⚠️ 局限性**

局限性包括对 LLM 计算成本高、对工具设计的依赖、在极大搜索空间或无评估反馈的任务上效果未知，以及可能需要更多实验验证其鲁棒性。

---

## 46. MaskFlow: Precise, Consistent and Seamless Regional Image Editing

**arXiv ID:** 2608.06929 | [PDF](https://arxiv.org/pdf/2608.06929v1)

**作者:** Rui Xu `[一作]` (SenseTime Research), Chengtao Lv `[通讯]` (SenseTime Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MaskFlow，一种针对区域图像编辑的训练框架，能够精确定位编辑区域、保持未编辑背景并实现无缝边界过渡。

**💡 创新点**

创新点包括：① 将编辑掩码直接嵌入流匹配（flow‑matching）的概率路径与损失函数中，实现对可编辑区域的显式约束；② 设计 Soft‑Poisson 去缝合（de‑seaming）模块，在采样过程中连续细化向量场，消除前景与背景的色彩/纹理缝隙；③ 通过去除提示中的位置信息，让模型更依赖掩码进行定位；④ 构建 MEData 数据集，提供自然场景与信息图的掩码编辑训练样本。

**🔧 技术方法**

核心技术包括：基于 Diffusion/Flow‑matching 的 MMDiT 模型、LoRA 微调、Soft‑Poisson 迭代求解、算子化的概率路径约束，以及使用 Vision‑Language 模型与图像生成模型的自动数据合成管线。

**📊 数据集**

使用自行构造的 MEData 数据集（约 10K 条自然场景与信息图对），以及公开的 ImageNet‑style 训练集用于预训练。

**📈 对比分析**

在 MEData 上与多种基线（Gemini、GPT Image、BAGEL、Flux、RefineAnything、RegionE、SpotEdit 等）进行定量与定性对比，MaskFlow 在 CLIP、DINO、FID、PSNR、SSIM、背景 LPIPS 等指标上均实现最佳或次优成绩，显著提升了编辑精度与背景保持效果。

**⚠️ 局限性**

局限性：仍需手动提供掩码，无法完全自动化定位；对极大或复杂形状的掩码可能产生轻微缝隙；在极端噪声或极端光照条件下的泛化能力尚待进一步验证。

---

## 47. Retrieval-Constrained Policy Optimization for Attack Technique Extraction from Cyber Threat Intelligence

**arXiv ID:** 2608.06778 | [PDF](https://arxiv.org/pdf/2608.06778v1)

**作者:** Jiayun Zhang `[一作]` (Amazon Web Services), Yi Fan `[通讯]` (Amazon Web Services)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了将网络威胁情报（CTI）文本自动映射到 MITRE ATT&CK 技术的多标签提取问题。

**💡 创新点**

创新点在于提出了两阶段框架：首先使用混合检索+监督微调（SFT）将候选技术空间缩小，然后通过强化学习（RLVR）结合可验证的分解奖励（精确度、召回率、输出格式）直接监督集合级别的预测质量，并采用奖励解耦归一化提升训练稳定性。

**🔧 技术方法**

使用的技术包括：BM25+句子 BERT 的混合检索、LoRA 微调、Group Relative Policy Optimization (GRPO) 强化学习、可验证奖励机制（precision/recall/format）以及奖励解耦归一化。

**📊 数据集**

所用数据集为四个公开 CTI 标注数据集：TRAM、Procedures、Derived Procedures 与 Expert，涵盖单标签和多标签情境。

**📈 对比分析**

与前沿 LLM（DeepSeek‑R1、Claude Sonnet 4.5）、开放权重 LLM（Qwen3‑8B、Ministral‑8B、Nova 2 Lite）以及领域特定方法（IntelEX、TechniqueRAG）进行对比，平均 F1 在技术层面和子技术层面均位居榜首，子技术层面提升 7.4%；8B 模型在单 GPU 上推理时延仅 0.34 秒，比 Claude 速度快 28 倍。

**⚠️ 局限性**

局限性包括：检索候选集限制召回，RL 提升有限；仅评估句子/段落级别提取，未覆盖全文级 TTP；对新版 ATT&CK 或 ATLAS 的迁移需少样本适配。

---

## 48. Translation Tag Team: Formal Rules and LLMs Translate More Macros Together than Apart

**arXiv ID:** 2608.06705 | [PDF](https://arxiv.org/pdf/2608.06705v1)

**作者:** Brent Pappas `[一作]` (University of Central Florida), Paul Gazzillo `[通讯]` (University of Central Florida)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个规则驱动的宏翻译工具MerC，并构建首个宏翻译基准MacroBench，评估MerC与LLM在C宏翻译上的表现。

**💡 创新点**

创新点在于：①给出宏与C语言共享语义的正式翻译规则，支持安全地将宏转为变量/枚举/函数；②构建了首个宏翻译基准；③提出并验证了“tag team”方法（先用MerC再用LLM），显著提升覆盖率与准确率。

**🔧 技术方法**

技术手段包括：使用Maki宏分析器提取布尔属性、基于Python实现MerC、对LLM（GPT‑4o、Claude 3.5、ChatGPT‑4‑Turbo、o1‑preview）采用零射击/少量示例提示、通过编译与运行时检查评估翻译正确性。

**📊 数据集**

数据集来源于26个真实C项目（包含Linux、Lua、SQLite、FFmpeg等），共采集100,711个宏定义，随机抽取398个宏构成测试用例。

**📈 对比分析**

比较方法：测量翻译覆盖率、失败率、总时长。MerC覆盖率50%、失败率0%；LLM覆盖率高至88%但失败率8–28%；Tag Team在保持低失败率（7–22%）的同时，将覆盖率提升30–80%，总翻译时间从数分钟到数小时显著降低。

**⚠️ 局限性**

局限性在于：仅能翻译展开为完整C构造的宏，无法处理token拼接、预处理条件等元编程宏；LLM翻译需人工验证；基准不覆盖跨文件宏调用和复杂预处理表达式；工具对宏内部嵌套支持不完整，需要多轮处理。

---

## 49. Is Forward Prediction Enough? Physical State Grounding for JEPA World Models

**arXiv ID:** 2608.06799 | [PDF](https://arxiv.org/pdf/2608.06799v1)

**作者:** Haodong Yan `[一作]` (Hong Kong University of Science and Technology), Haoang Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种物理状态与转移对齐的JEPA世界模型PSG-JEPA。

**💡 创新点**

通过在训练时加入单帧状态监督与多时程关节角度变化监督，使潜在空间更可解释且对决策更友好。

**🔧 技术方法**

使用JEPA框架、前馈预测、SIGReg正则、线性/MLP探针、GC-IDM规划、OFT策略学习等技术。

**📊 数据集**

在OGBench-Cube/Scene、LIBERO-Goal仿真、以及真实双臂机器人演示数据集上进行评估。

**📈 对比分析**

相较于LeWM、DINOv2以及动作监督版本，PSG-JEPA在潜在可辨识度、目标规划成功率与策略成功率上均有显著提升，尤其在有限训练预算与演示数据时优势更为明显。

**⚠️ 局限性**

仍需在训练阶段额外加入状态/转移头，且实验规模和任务多样性有限，缺乏在更大规模或更复杂任务上的验证。

---

## 50. Explore or Converge? Stage-Guided Per-Step Optimization for Diffusion Models

**arXiv ID:** 2608.06768 | [PDF](https://arxiv.org/pdf/2608.06768v1)

**作者:** Renye Yan `[一作]` (Peking University), Yimao Cai `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Stage-Guided Per-step Optimization (SGPO)，一种阶段感知的强化学习框架，用于在扩散模型中对不同生成阶段分别优化奖励。

**💡 创新点**

通过实时监测信噪比和语义变化，将扩散生成划分为混沌、结构稳定、收敛三阶段，并为每个阶段设计专属奖励函数，解决奖励稀疏和奖励劫持问题。

**🔧 技术方法**

结合扩散模型的逆向过程与强化学习（policy gradient），利用CLIP语义评分、SNR监测、阶段切换机制及多目标奖励融合。

**📊 数据集**

在HPSv2、Pick-a-Pic、GenEval及Simple Animals等四大公开数据集上进行微调与评估。

**📈 对比分析**

与DDPO、DPOK、D3PO、TDPO等SDE和流匹配基线对比，SGPO在美学评分、PickScore、FID、LPIPS等指标上平均提升约26.7%质量、36.7%收敛速度，并显著降低奖励劫持。

**⚠️ 局限性**

仍受限于奖励模型的准确性和多目标平衡，极端稀疏奖励场景下仍可能出现少量过拟合；同时阶段划分依赖阈值，可能在不同模型或任务中需要调参。

---

## 51. Thermodynamic Human-Computer Interaction

**arXiv ID:** 2608.07123 | [PDF](https://arxiv.org/pdf/2608.07123v1)

**作者:** Uzafir Ahmad Rafaq `[一作]` (Heriot-Watt University), Ali Muzaffar `[通讯]` (Heriot-Watt University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于热力学统计力学的统一框架，用来描述人机交互中的目标预测，并将 Fitts 定律和 Schmidt 定律推导为系统处于热平衡时的结果。

**💡 创新点**

创新点在于：①将数字目标视为谐振势阱；②通过 Gibbs 分布和 Hamiltonian 形式建立能量驱动的预测模型；③证明目标的颜色、文字清晰度等设计属性是独立且可加的势场参数；④揭示大目标和高速运动属于非平衡状态，从而解释 Fitts 定律的局限性。

**🔧 技术方法**

采用的技术包括：统计力学（Gibbs 分布、Hamiltonian 能量）、热力学温度对应的运动噪声、Kalman 滤波对速度噪声估计、贝叶斯决策框架（期望效用）以及 O(1) 时间的概率计算。

**📊 数据集**

使用的数据集为：30 名参与者在一次电子商务网站上完成的桌面和移动端交互日志（约 110 个商品链接）以及一个受控指点实验中 2400 次点击数据（四种按钮颜色/标签组合）。

**📈 对比分析**

与 ForesightJS（基于运动学的预测）和“预取所有可视链接”基线进行对比，评估指标包括 Fetch‑Click 比例和预测准确率。桌面端模型 Fetch‑Click 为 1.37、准确率 98.1%；移动端为 1.75、98.0%。相较于基线 2.00（桌面）/2.82（移动）和 100% 准确率，模型在效率上明显优越。

**⚠️ 局限性**

局限性：①仅适用于热平衡阶段，无法处理大目标或高速运动导致的非平衡情况；②样本量和任务范围有限（仅一个电商网站和受控指点实验）；③假设颜色、标签等参数独立且可加，实际应用中可能存在耦合；④未对不同设备、语言和文化背景进行验证。

---

## 52. Deal Me Maybe: The Role of Emotions in Multi-Agent Negotiation

**arXiv ID:** 2608.06922 | [PDF](https://arxiv.org/pdf/2608.06922v1)

**作者:** Massimiliano Luca `[一作]` (Bruno Kessler Foundation), Bruno Lepri `[通讯]` (Bruno Kessler Foundation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过在多智能体LLM谈判框架中对买卖双方进行情绪标签的提示调节，系统评估情绪对谈判结果与过程的影响。

**💡 创新点**

创新点在于：①构建了一个可控的情绪条件实验平台，覆盖六种离散情绪；②揭示买方情绪主导达成概率，而卖方情绪主要影响让步轨迹；③证明情绪调节在不同LLM模型间均能产生一致的方向性效果。

**🔧 技术方法**

技术方法包括：情绪标签+行为指令的系统提示调节；辅助模型（GPT‑4o‑mini）用于价格提取和谈判状态分类；在五种主流LLM（GPT‑3.5‑Turbo、GPT‑4o‑mini、Gemini 2.5 Flash、DeepSeek‑R1、Claude Sonnet 3.5）上实现买卖双方代理。

**📊 数据集**

使用的数据信集为350件真实消费品（电子、车辆、房地产、健康护理、家居、化妆品、食品、办公、房地产），每件产品包含零售价、特征描述和估计批发成本。

**📈 对比分析**

在36种买卖情绪组合×两种预算情境共72个实验条件下，对五个模型进行3次重复，评估指标包括达成率（DR）、买方降价率、卖方加价率、让步斜率等。结果显示：愉快情绪的买方DR最高（≈28.9%），但降价率低；愤怒情绪的买方几乎不达成（≈0.4%）；模型间的绝对数值差异显著，但情绪效应的方向性一致。

**⚠️ 局限性**

局限性包括：实验仅在模拟的买卖双方对话环境中进行，未涉及人类主体；情绪仅限定为六种基本情绪，缺乏对情绪持续性与交互的深入探索；结果可能受提示设计与模型特性影响，无法完全区分真实情绪效应与提示偏差。

---

## 53. TaskSense: Focusing on What Matters in World Models

**arXiv ID:** 2608.06544 | [PDF](https://arxiv.org/pdf/2608.06544v1)

**作者:** SM Mazharul Islam `[一作]` (University of Texas at Arlington), Manfred Huber `[通讯]` (University of Texas at Arlington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种名为TaskSense的任务中心世界模型，利用基于前一时刻隐状态的随机空间注意力在编码前对观测进行筛选，只重建被注意的区域，并辅以逆动力学监督来引导注意力。

**💡 创新点**

创新点在于：①在编码前主动控制信息流，通过随机空间注意力实现任务相关性前置；②将逆动力学监督与注意力机制结合，使模型在不完整观测下仍能学习到对控制有用的特征；③使用注意力条件重建避免重建被遮蔽的视觉信息。

**🔧 技术方法**

采用的技术包括：DreamerV3的RSSM框架、随机二元Concrete注意力、稀疏正则化、逆动力学监督（action prediction from feature差分）、注意力条件重建、以及标准的模型预测与策略优化流程。

**📊 数据集**

使用的数据集为DeepMind Control Suite（DMC）和Distracting Control Suite（DCS，包含DAVIS动态背景），所有任务均以64×64 RGB图像为输入。

**📈 对比分析**

与DreamerV3基线在DMC和DCS上进行对比，TaskSense在标准DMC任务上保持与DreamerV3相近的平均收益（如Cartpole 708 vs 777），但在DCS上表现显著提升（如Cheetah Run 490 vs 397，Walker Run 531 vs 286），显示对视觉干扰的鲁棒性大幅提升。

**⚠️ 局限性**

局限性在于：依赖逆动力学监督假设动作可预测特征与长期奖励信息一致；当两者不匹配时，注意力可能聚焦在子最优特征，影响任务性能。

---

## 54. ConstructCIE: A Dataset for Extracting Causal Information from Construction Accident Narratives

**arXiv ID:** 2608.06495 | [PDF](https://arxiv.org/pdf/2608.06495v1)

**作者:** Hung Nguyen `[一作]` (Texas A&M University), Kuan-Hao Huang `[通讯]` (Texas A&M University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 ConstructCIE 数据集，并对 OSHA 施工事故报告进行因果信息提取研究。

**💡 创新点**

提出了面向施工事故的分层因果分类模式，并通过手工标注实现了细粒度因果标签。

**🔧 技术方法**

使用监督序列标注模型（TagPrime）和指令调优大型语言模型（Llama、Qwen）进行端到端的因果信息提取。

**📊 数据集**

采用 2011-2023 年间发布的 530 篇 OSHA 事故调查摘要作为实验数据。

**📈 对比分析**

对比 JHE 与 IHE 两种提取策略，并与多种 LLM 与 TagPrime 进行精确匹配、软匹配和关键词匹配评估，结果显示事故类型预测准确但跨度提取仍低，JHE 在精确/软匹配上更优，IHE 在关键词匹配上略胜。

**⚠️ 局限性**

受限于数据规模、类别不平衡、未结合外部安全知识以及仅使用文本而非多模态信息，导致模型在细粒度跨度提取和罕见因果类别上表现不佳。

---

## 55. Hoverflie: An empirical investigation of rotor shrouds to transform micro air vehicles into multi-modal hovercraft

**arXiv ID:** 2608.06707 | [PDF](https://arxiv.org/pdf/2608.06707v1)

**作者:** Mrinmoy Modak `[一作]` (University of Hawaii at Manoa), Daniel S. Drew `[通讯]` (University of Hawaii at Manoa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种轻量级旋翼舷板，将Crazyflie 2.1改造成可在近地效应下悬停并在空中自由飞行的双模无人机。

**💡 创新点**

关键创新是通过参数化的管道、喷嘴和进气口几何设计实现了三倍的近地升力提升，并提出了捕捉“吸力”效应的经验模型。

**🔧 技术方法**

使用了定制的垂直悬停实验平台、负载传感器、三维打印+热成型舷板、Python GUI 控制、Lighthouse定位系统以及自定义 PID。

**📊 数据集**

主要数据来自在多高度、舷板几何参数下测得的升力曲线、PWM 与升力关系以及电池电压衰减曲线。

**📈 对比分析**

与原Crazyflie 2.1 进行对比，近地悬停续航提升约60%，但自由飞行续航下降约30%；模型拟合表明经验模型优于传统Cheeseman–Bennett。

**⚠️ 局限性**

限制包括仅在单轴静态悬停下测量，未考虑横向运动、姿态变化和动态过渡，舷板设计空间未进行全因子优化。

---

## 56. SubtleTalk: Generating Controllable Weakly-correlated Facial Dynamics for 3D Talking Heads via Residual Flow Matching

**arXiv ID:** 2608.06408 | [PDF](https://arxiv.org/pdf/2608.06408v1)

**作者:** Chenyang Ding `[一作]` (Shanghai Jiao Tong University), Ye Pan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1083 | [OpenAlex ID](https://openalex.org/A5071175541)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种两阶段残差流匹配框架（SubtleTalk），能够根据语音及可选控制信号生成自然且可控的弱相关面部动态，并保持准确的唇同步；

**💡 创新点**

①引入多模态控制（语音韵律、区域强度、连续情感VA）显式约束上部面部与头部运动；②使用确定性运动先验+残差流匹配分离强关联与模糊动态；③构建大规模高质量3D面部动画数据集SubtleTalk‑Face；

**🔧 技术方法**

多尺度WavLM+韵律特征、区域强度向量、连续VA标注；两阶段网络：确定性运动先验（DMP）+残差流匹配（Flow Matching）；隐式解耦控制（IDC）与VA动态预测器（VADP）；

**📊 数据集**

SubtleTalk‑Face（约3900身份、74小时语音-面部同步数据），并在现有公开3D面部动画数据集上进行对比实验；

**📈 对比分析**

与多种基线（传统回归、离散先验、单模态方法）进行量化与定性对比，实验表明SubtleTalk在上部面部动态的真实感、时序连贯性和多样性上均显著优于对手，同时保持与语音的高唇同步度；

**⚠️ 局限性**

仍受限于对大规模高质量上部面部数据的需求，模型复杂度较高，且对极端或罕见表情（如特殊情感或戏剧性动作）的生成尚不完美。

---

## 57. Accelerating Accurate Assignment Authoring Using Solution-Generated Autograders

**arXiv ID:** 2608.06572 | [PDF](https://arxiv.org/pdf/2608.06572v1)

**作者:** Geoffrey Challen `[一作]` (University of Illinois), Ben Nordick `[通讯]` (Code Awakening LLC)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用参考解答自动生成评测脚本，省去人工编写测试用例，快速创建大规模、准确的自动评分系统；

**💡 创新点**

创新点在于将参考解答作为唯一真值，结合变异测试自动生成误判样例并验证评测准确性，同时提取代码质量指标；

**🔧 技术方法**

使用随机输入生成器、变异测试、静态分析、运行时插桩、Gradle插件、Docker化后端与MongoDB存储；

**📊 数据集**

基于UIUC CS1课程的771道题库及约850万次提交的数据集；

**📈 对比分析**

与传统手工编写测试套件对比，评测准确率达99.96%，平均评测时间43 ms（99%分位数约806 ms），验证了方法的高效与准确；

**⚠️ 局限性**

局限在于仅支持单一JVM类、未能自动裁剪冗余测试、可能误判极端对抗性提交、对Python等非JVM语言的支持尚未完善。

---

## 58. Hyperbolic Graph Embedders for Link Prediction and Topology Reconstruction

**arXiv ID:** 2608.07029 | [PDF](https://arxiv.org/pdf/2608.07029v1)

**作者:** Robert Jankowski `[一作]` (TU Delft), Dorota Celińska-Kopczyńska `[通讯]` (University of Warsaw)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文统一评估了13种无监督超曲率网络嵌入方法，比较了它们在链路预测和拓扑重建两项任务中的表现，并针对不同网络结构给出方法选择建议。

**💡 创新点**

首次提出跨学科（最大似然、结构/排序、表征学习、混合）嵌入方法的统一基准框架，系统揭示了方法类型与网络结构之间的相互关系，为实际应用提供实用指南。

**🔧 技术方法**

采用最大似然（MLE）、结构/排序（SOE）、表征学习（RLE）及混合（HE）技术，并利用超曲率几何模型（H₂/S¹）进行网络生成、嵌入与评估。

**📊 数据集**

在实验中使用了50个合成网络（N=500/1000/2000，温度T=0.1/0.4/0.7，平均度<k>=10/20）以及11个真实网络（包括蛋白质相互作用、脑网络、引用图、通信网络等）。

**📈 对比分析**

通过链路预测的精度、预测力等指标以及拓扑重建的度分布、全局传递性、Jaccard相似度等指标比较，结果显示MLE、RLE、HE总体优于SOE，不同方法在链路预测与拓扑重建上存在明显权衡。

**⚠️ 局限性**

未考虑算法的计算复杂度和内存消耗；评估任务仅限链路预测和拓扑重建；所用生成模型不包含社区结构，可能限制对具有显著社区特征网络的适用性。

---

## 59. Pessimal Elections for Approximately Dominating Sets

**arXiv ID:** 2608.06872 | [PDF](https://arxiv.org/pdf/2608.06872v1)

**作者:** Moses Charikar `[一作]` (Stanford University), Kangning Wang `[通讯]` (Rutgers University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究社交选择理论中近似支配集（approximate dominating set）的问题，构造最坏情况下的选举实例，证明任意选举都存在大小为O(1/ε²)的近似支配集，并给出对应的下界，使上界与下界在常数因子上匹配。

**💡 创新点**

首次给出了近似支配集大小下界达到Ω(1/ε²)的构造，证明了此前上界O(1/ε²)的紧致性，完成了该问题的常数因子最优解。

**🔧 技术方法**

采用布尔函数与集系统的对应技术，将候选人映射到布尔函数与坐标的组合，利用多数函数与输入变量的相关性分析，构造选举的投票偏好；还借鉴了布尔函数分析中的中心二项式估计。

**📊 数据集**

无具体数据集；研究完全在理论构造与概率上进行。

**📈 对比分析**

与之前仅给出O(1/ε²)上界的结果相比，该工作提供了匹配的下界，证明上界是最优的；理论上展示了下界的常数因子约为π/8²≈1.23，说明方法在理论性能上是最优的。

**⚠️ 局限性**

限制主要在于常数因子仍未完全确定，构造仅给出渐进结果；同时方法针对的是完全随机或特定构造的投票模型，可能不直接适用于实际投票数据或更一般的选举规则。

---

## 60. Mind the Gap: A Dual Knowledge Graph Framework for Unified Multi-task User Intent Inference

**arXiv ID:** 2608.06752 | [PDF](https://arxiv.org/pdf/2608.06752v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 61. FedVAR: Prototype-Aligned Federated Framework for Video Anomaly Recognition

**arXiv ID:** 2608.06876 | [PDF](https://arxiv.org/pdf/2608.06876v1)

**作者:** Ghani Haider `[一作]` (Chungbuk National University), Taehong Kim `[通讯]` (Chungbuk National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 FedVAR，利用全局正常性原型对分布式视频异常识别进行弱监督的联邦学习框架，解决语义偏差问题。

**💡 创新点**

创新点在于跨模态原型对齐机制：通过聚合各客户端的正常性原型形成全局语义锚点，使视觉和文本特征在同一空间重新居中，显著缓解非IID导致的语义不一致。

**🔧 技术方法**

采用预训练 CLIP 视觉-文本双模体作为特征提取器，使用可学习的文本提示、Axial Transformer 进行时序建模，并在联邦学习中仅同步可训练参数和全局原型。

**📊 数据集**

在 UCF‑Crime、XD‑Violence 和 ShanghaiTech 三大视频异常数据集上进行实验，并在随机、事件、场景等多种非IID划分下验证其鲁棒性。

**📈 对比分析**

与 FedCoOp、Fed‑WSVAD、CLAP、ZS‑CLIP、Temp‑CLIP 等基线对比，FedVAR 在各数据集的 mAUC/AUC/AP 指标上均超过或匹配最佳方法，尤其在多类别识别与跨域迁移上取得显著提升。

**⚠️ 局限性**

局限性包括：在极度稀疏的长视频（如 XD‑Violence）时时序建模粗粒度略逊；全局原型采用简单加权平均，对噪声或极端恶意客户端易受影响；未在真实边缘设备上进行完整的能耗和实时性评估。

---

## 62. Prune Once: Retraining-Free Task-Agnostic Pruning for Vision-Language Models

**arXiv ID:** 2608.06901 | [PDF](https://arxiv.org/pdf/2608.06901v1)

**作者:** Minseok Kang `[一作]` (Chung-Ang University), Dahuin Jung `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种一次性、无再训练的任务无关视觉语言模型压缩框架 PORTA，能够在单次剪枝后直接应用于多种下游任务。

**💡 创新点**

创新点在于使用激活方差作为跨模态无偏重要性度量，并根据输出特征方差自适应分配层级稀疏度，从而克服传统幅值基准导致的模态偏差和均匀稀疏带来的性能陡降。

**🔧 技术方法**

技术包括基于特征维度激活方差的权重重要性计算、对层输出方差的统计推导实现层级稀疏比例估计、以及对方差矩阵的对角近似以降低计算开销。

**📊 数据集**

使用 MSCOCO 等通用图文对作为校准集，评估时使用 CLIP、BLIP、Qwen2‑VL 模型，在图像分类（CIFAR‑10/100、ImageNet、Oxford Flower‑102）、图文检索（MSCOCO、Flickr30K）和 VQA（ScienceQA）任务上验证。

**📈 对比分析**

与 Wanda、SparseGPT、ECoFLaP 和 Multiflow 等基准相比，PORTA 在 45%–65% 稀疏度下保持或提升性能，例如在 65% 稀疏下 MSCOCO IR@5 从 35.00 提升至 37.32，ImageNet 准确率从 25.96% 提升至 31.38%，整体平均提升约 12.6% 相较 SparseGPT，21.5% 相较 Wanda。

**⚠️ 局限性**

局限性包括对零均值及对角协方差近似的假设（虽影响小）、仅在视觉–语言任务上验证，未考察更广泛模态或更大模型，且对极端稀疏度或非图文输入的鲁棒性仍待进一步研究。

---

## 63. Divergent Response Modes in Frontier Language Models Under Steering Pressure

**arXiv ID:** 2608.06578 | [PDF](https://arxiv.org/pdf/2608.06578v1)

**作者:** Ali Jalal-Kamali `[一作]` `[通讯]` (University of Southern California Institute for Creative Technologies), Ali Jalal-Kamali (University of Southern California Institute for Creative Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了六款前沿语言模型在面对明确引导时的行为可控性与非合规模式，基于基线与引导对照的300对样本进行盲评判，探讨模型在值冲突、推理揭示与推理抑制三大类情境下的反应差异。

**💡 创新点**

首次系统比较不同开发者的前沿模型在同一对比任务下的定性与定量行为差异，揭示模型在拒绝、隐含推理、抑制策略等方面存在的类别差异，并对一种开源模型通过线性探针与激活注入实现机制级解释。

**🔧 技术方法**

采用了盲评判的同行评审框架、留一一致性打分、Benjamini–Hochberg多重检验校正、Cohen’s h效应量、Fleiss’ κ 评估判定一致性，以及对开源模型的线性探针与激活注入实验。

**📊 数据集**

使用了340个评测样本，分为三类核心情境（价值冲突、推理揭示、推理抑制）各100个对照对，另含20个验证对，全部手工构造并对每个模型生成两条回应。

**📈 对比分析**

通过每个模型的盲评判得出一致性标签后计算各模型在各标签下的比例，并用 Fisher 精确检验和 BH 校正评估差异显著性；在开源模型上线性探针的准确率最高达0.87，激活注入可将行为转移至0%或86%。

**⚠️ 局限性**

局限性包括：只评估每个开发者的单一模型，缺乏对同一实验室内其他模型的推广性；评审者与人工判断的一致性尚未充分验证；推理抑制类的判定一致性相对较低，可能影响结论稳健性。

---

## 64. Weak Adversarial Neural Pushforward Method for Boltzmann Equation

**arXiv ID:** 2608.06823 | [PDF](https://arxiv.org/pdf/2608.06823v1)

**作者:** Jenia Fardousi Koly `[一作]` (Southern Methodist University), Wei Cai `[通讯]` (Southern Methodist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于弱对抗性神经推前映射（WANPM）的高维分布式求解器，用以求解受外力驱动的Boltzmann方程；

**💡 创新点**

创新点在于将弱形式的碰撞算子与可逆推前映射结合，利用对抗平面波测试函数实现分布式无偏损失，同时通过时间门控实现硬性初始条件；

**🔧 技术方法**

使用了可逆流（RealNVP）分离的空间采样器与条件速度采样器、对抗平面波和多项式基测试函数、Monte‑Carlo估计、梯度裁剪的双重优化；

**📊 数据集**

实验采用闭式高斯初始分布与无碰撞自由传输（E1）和谐波受力（E2）两组基准，没有使用公开数据集；

**📈 对比分析**

与传统DSMC和PINN对比，WANPM在自由传输下在一维边缘误差与相关性上与DSMC相当，显著优于PINN，且在受力案例中通过多项式基显著恢复位置-速度协方差；

**⚠️ 局限性**

局限在于对协方差等弱信号的估计受采样噪声影响，单一训练运行可能导致协方差不稳定，且对单维方差仍存在轻微偏差。

---

## 65. Spatiotemporal Agility: Time-Constrained Reinforcement Learning for Vision-Guided Dynamic Quadrupedal Interception

**arXiv ID:** 2608.06907 | [PDF](https://arxiv.org/pdf/2608.06907v1)

**作者:** Yidong Zhu `[一作]` (Zhejiang University), Hua Chen `[通讯]` (LimX Dynamics Technology Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于视觉预测和时空目标条件的四足机器人捕球框架，能够在约一秒的时间窗口内预测并捕捉投掷球。

**💡 创新点**

创新点在于将多摄像头实时感知、Kalman滤波轨迹预测与时间感知的RL位置条件策略集成，解决了感知延迟与时序失配问题，并实现了旋转优先的机动捕球行为。

**🔧 技术方法**

使用YOLOv13n目标检测、Intel RealSense D415深度相机、Kalman滤波轨迹预测，以及强化学习中的位置-时间条件策略。

**📊 数据集**

采用自制的弹道投掷数据集（包含仿真和实测的球投掷轨迹）进行训练与评估。

**📈 对比分析**

与传统速度跟踪基线和即时球状态基线对比，仿真环境下成功率从67%提升至86%，实测环境下跟踪成功率从25%提升至62%，捕捉成功率从6%提升至37%。

**⚠️ 局限性**

局限性包括硬件执行延迟、视觉噪声与仿真到现实的差距导致整体成功率仍有限，且实验仅在平地且依赖外部摄像头的环境下验证。

---

## 66. ECAD: Expanding Class-Agnostic Detection Beyond Thing-Centric Objectness

**arXiv ID:** 2608.06841 | [PDF](https://arxiv.org/pdf/2608.06841v1)

**作者:** Liang Wan `[一作]` (Tianjin University), Fangzhuo Gao `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了扩展类无关检测 (ECAD)，使检测框架能够发现除传统可计数物体之外的语义重要视觉元素。

**💡 创新点**

创新点在于将物体定义从“可计数实体”扩展到任何视觉上有意义的内容，并通过 Geometry‑Aware Expert Regression 和 Prototype‑Guided Query Modulation 两个模块提升定位与物体性估计。

**🔧 技术方法**

利用冻结的 DINOv3 编码器、轻量化 DETR 解码器，并加入 GAER 与 PGQM 以实现几何专家回归和原型引导的查询调制。

**📊 数据集**

构建了 BTCO‑Bench，包括真实场景的 LVIS 补充标注和跨域（艺术、素描、恶劣条件）数据集，用于评估物体性发现能力。

**📈 对比分析**

与 RPN、CAOD、MAVL、DiPEx、PF‑RPN 及 RF‑DETR 等基线在 BTCO‑Real 与 BTCO‑XDomain 上比较，ECADet 在 AP、AP_50、AP_75、AR_100 上均领先 10–20+ 分，且保持较低参数量与较高 FPS。

**⚠️ 局限性**

限制在于仍是类别无关框预测，缺乏语义标签；对极端形变或低分辨率场景的鲁棒性待进一步验证。

---

## 67. Calibrating WEAT Against Anisotropy: ZCA Whitening as a Geometric Pre-Processing Step for Embedding Association Tests

**arXiv ID:** 2608.06908 | [PDF](https://arxiv.org/pdf/2608.06908v1)

**作者:** Seitaro Ono `[一作]` (Kyoto University), Jun Saiki `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了使用 ZCA whitening 对词嵌入空间进行几何校准，以修正 WEAT 测量中的各向异性误差。

**💡 创新点**

创新点在于将 ZCA whitening 作为预处理步骤，恢复嵌入空间的各向同性，从而解决 WEAT 依赖余弦相似度时的几何偏差。

**🔧 技术方法**

使用技术包括 ZCA whitening、Word Embedding Association Test (WEAT)、语义相似度基准（WordSim-353、SimLex-999、STS-B）以及统计显著性检验等。

**📊 数据集**

使用数据集：WikiText-103（估计白化矩阵）、10 套标准 WEAT 测试集、WordSim-353、SimLex-999 与 STS-B 作为语义基准。

**📈 对比分析**

通过在七种模型（静态、上下文、对比）上比较未校准与校准后的 WEAT 效果大小与显著性，校准后超过 30% 的结果显著性状态改变；在高各向异性模型上语义相似度显著提升（如 GPT‑2 从 0.263 提升至 0.620）。

**⚠️ 局限性**

限制包括：白化矩阵依赖参考语料，对不同语料的敏感度未做系统评估；只评估了七种模型，未必能推广到更广泛的模型；固定正则化参数未进行灵敏度分析；未探讨校准对下游任务偏差的具体影响。

---

## 68. Ex-Post Equilibria: Structure and Computation

**arXiv ID:** 2608.07025 | [PDF](https://arxiv.org/pdf/2608.07025v1)

**作者:** Francesco Giordano `[一作]` (HEC Paris), Christian Kroer `[通讯]` (Columbia University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在参数不确定性下的后置均衡（EPE）概念，给出了其存在性、逼近性以及计算方法，并在两类经典博弈（两人零和博弈与势博弈）中提供了完整的理论与算法框架。

**💡 创新点**

创新点在于将EPE问题转化为辅助极小极大问题，证明EPE由单调性和集合一致性唯一确定；提出最优近似后置均衡概念，并给出多项式时间的近似与最优算法；以及在对称势博弈中通过凸二次形式得到闭式最优解。

**🔧 技术方法**

主要使用的技术包括极小极大框架、凸优化与线性规划、凸-凹优化与在线学习算法（Mirror Prox、extragradient、预测-均衡等）、对偶性与潜在函数方法以及数值实验。

**📊 数据集**

实验采用人工生成的博弈实例（如 Rock‑Paper‑Scissors、Matching Pennies、Kuhn Poker、随机生成零和博弈）以及一系列对称势博弈的经典模型（Cournot、Congestion、网络、Bertrand、公共物品）中所用的参数空间。

**📈 对比分析**

通过在线学习算法求解辅助极小极大问题，实验结果表明目标值能够达到理论上的 2-近似，且在零和博弈中证明不存在更优的多项式时间逼近；与传统的稳健纳什均衡和信息不完全博弈中的后置均衡相比，提供了更强的鲁棒性和可计算性。

**⚠️ 局限性**

局限性包括：仅在有限行动或连续可微的势博弈下可得到算法；对非凸潜在函数或非对称游戏的分析不足；逼近下界仍存在 1.16 以上的缺口；对无限参数空间的理论完整性尚未建立。

---

## 69. Toward surface-based registration of a virtual preoperative cutting guide onto the mandible for reconstruction surgery

**arXiv ID:** 2608.06599 | [PDF](https://arxiv.org/pdf/2608.06599v1)

**作者:** Yue Yang `[一作]` (Vanderbilt University), Jie Ying Wu `[通讯]` (Vanderbilt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发了一套无标记、牙齿感知的增强现实下颚切割导引系统，可在口内摄取局部深度图后自动注册并实时投影预先规划的虚拟切割导引。

**💡 创新点**

创新点在于：①利用牙齿作为天然高曲率特征实现markerless注册；②在目标运动后采用姿态插值与更新机制；③设计了三种曝光级别的盲法试验，评估不同牙齿可见度对注册精度的影响。

**🔧 技术方法**

技术手段包括：HoloLens 2 ToF摄像机、Fast Point Feature Histogram (FPFH)、TEASER++截断最小二乘全局对齐、异向点‑面ICP局部优化、实时姿态插值与延迟补偿。

**📊 数据集**

数据集：3D打印PLA下颚模型（Bambu Lab H2S）作为phantom，配合NDI Polaris跟踪系统收集目标点；并使用预先分割的CT下颚表面点集进行对齐。

**📈 对比分析**

性能对比：在全可见、牙齿+周围表面、仅牙齿三种曝光下的中位目标注册误差（TRE）分别为4.05 mm、6.10 mm、7.10 mm，平均为5.45 mm；运动到显示延迟平均0.805 s（≤1 s）。这些误差与传统物理导引约4 mm的误差相近，显示AR方案的可行性。

**⚠️ 局限性**

局限性包括：①血液、唾液或手术器械可能遮挡牙齿并产生噪声；②在口外或经皮切口下牙齿可见度更低，性能尚未验证；③可视化渲染可能遮挡手术视野；④运动延迟仍存在，需要进一步优化。

---

## 70. Unordered Landmark Visual Navigation

**arXiv ID:** 2608.06833 | [PDF](https://arxiv.org/pdf/2608.06833v1)

**作者:** Hao Ren `[一作]` (Sun Yat-sen University), Hui Cheng `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 ULVN 框架，实现在仅依赖无序 RGB 图像集合（不使用里程计、深度或时间序列）的闭环图像目标导航。

**💡 创新点**

创新点包括：
1) RAVEL 通过一次性阈值校准、最大生成森林（MSF）和循环重插入，构建鲁棒的无序拓扑图；
2) BPL 在 2D 拓扑图上实现全图贝叶斯状态估计，利用熵自适应融合权重抑制视觉歧义；
3) BASS 采用“最大宽度路径”策略进行子目标规划，动态重规划以应对漂移与偏差。

**🔧 技术方法**

技术手段：视觉位置识别（VPR）使用 MegaLoc；特征匹配使用 LightGlue；全局检索用 FAISS；图结构提取用 Kruskal（MSF）与 k‑means；贝叶斯滤波与熵自适应融合；Dijkstra 计算最大最小宽度路径；本地控制可使用 ViNT 或 NoMaD 等视觉基础模型。

**📊 数据集**

使用的数据集包括：
- NVIDIA Isaac Sim 的 GRScenes（10 个室内/室外场景）
- CARLA 进行 RAVEL 与 BPL 的鲁棒性评估
- 真实环境中 Diablo 机器人与 Azure Kinect 的实地测试
- 公开的 RECON、SCAND、GoStanford、SACSoN 等图像序列，用于视觉降噪与扰动测试。

**📈 对比分析**

对比方法：ResNet‑50、DINOv2、MegaLoc、ViNT、PlaceNav、Uni‑Navid、UniGoal、NoMaD 等；评估指标包括拓扑图精度（P、R、F1）、定位精度、成功率（SR）、路径长度加权成功率（SPL）以及碰撞次数。实验表明，ULVN 在拓扑构建上实现最高的 F1（>0.90），定位准确率约 95%，而在导航任务中 SR 达到 71.9%（比 Uni‑Navid 高 20% 以上），SPL 亦显著优于基线。

**⚠️ 局限性**

局限性：
1) 对本地规划器的障碍感知和轨迹生成仍有依赖，偶尔因遮挡或动态障碍导致失败；
2) 纯 RGB 的视觉匹配在极端光照或模糊条件下仍可能出现误匹配；
3) 在极大规模无序图像集合中，检索与图构造的计算开销仍高；
4) 该方法尚未验证在高度动态或多机器人协作场景中的鲁棒性。

---

## 71. GOPI: Generation-Oriented 3D Pose Inference for Furniture Insertion from Single-View RGB-D Indoor Scenes

**arXiv ID:** 2608.06836 | [PDF](https://arxiv.org/pdf/2608.06836v1)

**作者:** Ruifeng Zhai `[一作]` (Sun Yat-sen University), Liang Lin `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种两阶段家具插入方法：先使用GOPI在单视角RGB-D场景中推断几何合理的3D姿态，再将该姿态投影到图像平面作为条件进行图像合成。

**💡 创新点**

创新点在于：①将家具插入视为先推断3D姿态再生成图像的“pose-first”框架；②设计了基于场景-框架交互的双流点云编码网络和迭代细化机制；③引入视锥一致性、支撑对齐与房间轴对齐等几何约束提升姿态可行性。

**🔧 技术方法**

使用点云编码（PTv3）、交叉注意力、GeM池化、迭代细化网络、以及基于Blender渲染的几何投影作为图像条件；后期使用Diffusion模型（IDM）进行图像合成。

**📊 数据集**

在Synthetic dataset "Front3D-Insertion"（基于3D-FRONT和3D-FUTURE资产）上进行训练与评测，包含约18k个样本。

**📈 对比分析**

与直接回归（PTv3 Regression）和无迭代的Vanilla基线对比，GOPI在几何可行性（Overall@M）提升约18个百分点，位置准确率（TransAcc@0.3m）提升约11个百分点，朝向准确率（YawAcc@10°）提升约5个百分点；实验也展示了在不同家具尺度下的投影-生成一致性稳定。

**⚠️ 局限性**

局限性包括：仅在合成RGB‑D场景上验证，缺乏真实世界泛化；依赖深度信息，无法直接用于纯RGB环境；评估主要聚焦几何一致性，未全面覆盖视觉真实性。

---

## 72. A Multi-Agent Framework for Automated Coarse-Grained Molecular Dynamics of Polymers

**arXiv ID:** 2608.06694 | [PDF](https://arxiv.org/pdf/2608.06694v1)

**作者:** Joohee Choi `[一作]` (Korea Advanced Institute of Science and Technology), Seunghwa Ryu `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一套名为 CGMas 的多智能体框架，能够从自然语言描述的聚合物名称与目标分辨率出发，自动完成原子级拓扑构建、系统平衡、粗粒化映射、势能参数化及模型验证，最终生成可直接使用的 CG 模型。

**💡 创新点**

其创新点在于：①将 LLM 进行化学推理与错误自校验与专门的物理计算工具结合；②实现了全流程自动化的下采样（从 AA 到 CG）且无需用户手工编写拓扑或脚本；③通过自校正循环确保拓扑与势能参数的物理合理性。

**🔧 技术方法**

使用的技术包括：大语言模型（LLM）与 LangChain/LangGraph 的多智能体调度、LAMMPS 的原子/粗粒化动力学模拟、Boltzmann 逆推（Bond/Angle/ RDF）以及自定义的自校正与审核模块。

**📊 数据集**

使用的数据集为 27 种聚合物任务，涵盖五个难度级别（包括 4 个碳氢化合物、7 个含 heteroatom 的单体、9 个复杂侧链、3 个多功能单元以及 4 个共聚物），每个任务均在同一配置下生成原子级参考模型。

**📈 对比分析**

在 27 任务中，81.5% 的模型通过密度误差 ≤5% 的验收；平均 CG 模型与 AA 模型的平衡密度误差约 4%；粗粒化模拟在 1 分钟内完成，而原子级模拟耗时 38–88 分钟，提升约 40–90 倍；LLM 计算成本低于 0.01 美元/任务。

**⚠️ 局限性**

局限性包括：仅适用于 OPLS-AA 体系，缺乏对金属或电荷强的聚合物的支持；Boltzmann 逆推不自洽，可能导致结构与密度匹配但局部配位不准；目前仅覆盖聚合物，其他材料系统需进一步扩展；LLM 结果可变，导致成本和修正循环次数波动。

---

## 73. Automated Terminal-to-Housing Assembly System for Flat Ribbon Cable Harness

**arXiv ID:** 2608.06996 | [PDF](https://arxiv.org/pdf/2608.06996v1)

**作者:** Eunkyu Choi `[一作]` (Sogang University), Seokhwan Jeong `[通讯]` (Sogang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

开发了一套面向单行双向平板带状电缆线束（FRCH）的端口对壳体自动装配系统，采用纯机械、无感知的四阶段流程（Cable Alignment、Lean & Slide、Weaving、Clamping）完成多端子协同插入；

**💡 创新点**

创新点在于提出了一种“传感器极简、机械化”的装配策略，专门解决FRCH多端子通过柔性带状相互耦合导致的姿态误差、插入干扰与最终锁定不稳定性；同时首次实现了多端子协同插入的实验原型；

**🔧 技术方法**

核心技术包括：基于弹性导向的Cable Alignment导向槽；利用“Lean & Slide”机动实现自适应姿态校正；在插入过程中加入“Weaving”振荡运动消除内部卡阻；以及Clamping闭合装置稳固终端姿态；全部由12轴伺服驱动、光纤传感检测完成；

**📊 数据集**

该工作不使用公开数据集，而是通过80次实验测试完成实验室级原型验证；

**📈 对比分析**

对比方法：将传统平行插入与Lean & Slide两种方式对比，平行插入成功率仅15%，Lean & Slide达90%；整体端到端成功率83.75%，分阶段成功率分别为86%（传输）和98%（插入），循环时间约33秒（半速），在全速可降至15秒；

**⚠️ 局限性**

局限性包括：系统高度依赖产品参数（端子数、间距、壳体几何），需要重新配置导向路径与插入参数；使用3D打印原型导致尺寸精度、刚度与耐用性不足；循环时间与工业水平仍有差距；未来需加入抓取与视觉检测提升鲁棒性。

---

## 74. Canonicalization Failures as a Recurring Vulnerability Class: Representation Divergence in Cryptographic Systems and Its Avoidance

**arXiv ID:** 2608.06508 | [PDF](https://arxiv.org/pdf/2608.06508v1)

**作者:** Arslan Brömme `[一作]` `[通讯]` (Chain Horizon GmbH), Arslan Brömme (Chain Horizon GmbH)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一套基于唯一性条件的表示理论框架，对加密系统中出现的多种“可变性”缺陷进行统一归类，并给出了相应的规范化（canonicalization）义务和审计步骤。

**💡 创新点**

创新点在于：①把散落在不同区块链与协议生态中的缺陷（如交易可变性、签名可变性、状态值与默认值冲突等）归纳为同一数学根源——唯一性条件的两种破裂方向；②将传统的 canonicalization 安全理念与表示理论、可计算性理论结合，形成跨生态的解释与工具；③基于此构建了可执行的审计流程，帮助开发者在设计时就能检测并消除表示歧义。

**🔧 技术方法**

主要技术手段包括：表示理论（唯一性条件、S‑break 与 I‑break 的定义）、可计算性分析、案例驱动的分类网格、规范化义务的形式化表达，以及对已存在的标准化方案（如 BIP‑66、Deterministic CBOR、JSON Canonicalization Scheme）的引用与对比。

**📊 数据集**

所使用的数据集主要是案例集：ECDSA 签名可变性、Nomad 桥接漏洞、Wormhole 账户混淆事件，以及跨生态的其他实例（Bitcoin、Cosmos、Ethereum 等）作为背景证据。没有引入大规模实验数据或数值数据集。

**📈 对比分析**

比较方法是基于概念性的分类网格，对每个案例按四步分析（机制、分类、canonicalization 效果、额外措施）进行评估。文中未进行性能实验或数值比较，重点在理论解释与流程设计的可行性。

**⚠️ 局限性**

局限性包括：①不提供新的密码原语或新的安全证明；②缺乏系统化的量化统计，未评估该缺陷类在整个生态中的频率；③canonicalization 只能降低风险，无法替代其他安全措施；④依赖于可显式定义的语义等价关系，若未明确则难以执行；⑤可能产生误报，需结合业务场景判断；⑥框架主要针对字节级绑定的系统，对非字节绑定或浮点数等非可计算性问题不适用。

---

## 75. AnyTrack: Unifying Visual Object Tracking with Any Modalities

**arXiv ID:** 2608.06773 | [PDF](https://arxiv.org/pdf/2608.06773v1)

**作者:** Hao Li `[一作]` (Army Engineering University of PLA), Huchuan Lu `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了AnyTrack，一种支持任意组合输入模态（RGB、灰度、深度、热成像、事件、语言描述、音频等）的统一视觉目标跟踪框架；

**💡 创新点**

创新点在于（1）Modality-aware Interaction Module（MIM）通过动态Mixture‑of‑Experts路由和语义专家实现跨模态特征桥接与时空一致性；（2）Context Understanding Module（CUM）利用全局‑局部提示和双向注意力实现目标对齐和位置自适应伪掩码；（3）在多模态基准上扩展了灰度、语言、音频标签，实现了跨任务训练；

**🔧 技术方法**

采用Mixture‑of‑Experts、动态噪声门控、语义专家融合、跨模态注意力、双向异向注意、动态高斯伪掩码、HiViT视觉编码器、CLIP文本编码器、WavLM音频编码器等技术；

**📊 数据集**

使用RGBDT500、DepthTrack、LasHeR、VisEvent四个多模态跟踪基准，并在每个基准中添加灰度、语言和音频数据；

**📈 对比分析**

与现有RGB/RGB+D/RGB+T/RGB+E以及联合模态跟踪器（如Un-Track、RDTTrack、SDSTrack、ViPT、XTrack等）进行对比，AnyTrack在DP、AUC、PR、F‑score等指标上均领先或处于最前沿；在完整模态和缺失模态场景下均保持强劲性能；

**⚠️ 局限性**

仍存在的局限包括：单模态（如深度、音频）性能相对较低，缺失或极度噪声模态时性能下降；需要大量多模态数据和显著计算资源；模型复杂度和推理速度相对传统单模态跟踪器较高；

---

## 76. Explanation Stability of Test-Time Adaptation in Computational Pathology: A Large-Scale Benchmark

**arXiv ID:** 2608.07062 | [PDF](https://arxiv.org/pdf/2608.07062v1)

**作者:** R. G. Bahumanya `[一作]` (R.V. College of Engineering), Anala M. R `[通讯]` (R.V. College of Engineering)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究测试时适应对病理图像模型解释稳定性的影响，并建立大规模基准。

**💡 创新点**

首次引入解释稳定性指数(ESI)，展示解释漂移与方法、架构和域移的系统关系。

**🔧 技术方法**

采用17种TTA方法、4类归因器（IG、Grad‑CAM、Grad‑CAM++、attention rollout）和混合效应模型等技术。

**📊 数据集**

使用Camelyon17多医院肿瘤/正常切片和NCT‑CRC‑HE九分类肠癌组织数据集。

**📈 对比分析**

与传统的准确率/校准评估相比，ESI揭示部分方法解释不变却校准/准确下降，显示解释稳定性与性能独立。

**⚠️ 局限性**

局限在于ESI为统计指标，未评估全切片级推理与临床读者信任，且仅基于补丁级数据。

---

## 77. MIFA: An MILP-based Framework for Improving Differential Fault Attacks

**arXiv ID:** 2608.06837 | [PDF](https://arxiv.org/pdf/2608.06837v1)

**作者:** Hanbeom Shin `[一作]` (Korea University), Dongjae Lee `[通讯]` (Kangwon National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于MILP的差分故障攻击框架MIFA，能够在更深的轮数上寻找单解差分轨迹，从而显著减少注入故障数量。

**💡 创新点**

首次将MILP直接用于搜索唯一差分轨迹，并系统评估所有单比特故障位，理论计算所需故障数，同时证明LS S-box下关键候选空间始终为2^k。

**🔧 技术方法**

采用混合整数线性规划求解器、差分分析、线性结构分析以及概率模型与实验验证相结合的技术方案。

**📊 数据集**

在DEFAULT（80轮）和BAKSHEESH（35轮）两种基于LS S-box的块密码上进行实验，使用随机明文/密钥作为数据来源。

**📈 对比分析**

与先前的单解差分攻击、信息组合、SDFA等方法对比，MIFA在DEFAULT上实现6/7/8轮攻击分别仅需3/2/2次故障注入，性能大幅提升。

**⚠️ 局限性**

仅适用于单比特故障模型，对非LS S-box或多比特/字节级故障求解困难；在更深轮数时单解轨迹出现概率极低，受限于MILP求解时间。

---

## 78. A Disturbance in the Force: Force Actuation on the RAVEN II Surgical Robot with Parallel Motor-Cable Units

**arXiv ID:** 2608.06488 | [PDF](https://arxiv.org/pdf/2608.06488v1)

**作者:** Haonan Peng `[一作]` (University of Washington), Blake Hannaford `[通讯]` (University of Washington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文开发了一套在RAVEN-II手术机器人周围布置六个电机-绳索单元的并行力学激励系统，能够在机器人运动时对末端执行器施加任意方向和大小的外力。

**💡 创新点**

创新点在于通过多电机-绳索组合实现无干扰、可编程的外力注入，为学习型外力估计提供高质量训练数据，同时不需要在末端装配额外传感器。

**🔧 技术方法**

采用ROS框架实现控制软件，使用SLSQP优化求解绳索张力，配合320 Hz负载传感器闭环反馈，利用Monte‑Carlo仿真与Kabsch算法完成电机定位与绳索方向计算。

**📊 数据集**

研究中自行采集了一套机器人轨迹与外力共现的原始数据集，用于后续神经网络模型的训练。

**📈 对比分析**

通过对比期望外力与实际施加外力的平均绝对误差，实验结果显示X、Y、Z方向误差分别为0.80 N、0.79 N和0.44 N，整体误差低于1 N，验证了系统的高精度激励能力。

**⚠️ 局限性**

局限性包括：电机控制仍嵌入RAVEN-II系统，需进一步拆分为独立驱动器；绳索张力受限时可能出现干涉；系统平滑性与稳定性仍有提升空间，且仅在实验环境下验证，缺乏大规模实战验证。

---

## 79. Retention-Aware RISC-V ISA Extension and Memory Controller on FPGA for MLC NVM

**arXiv ID:** 2608.06725 | [PDF](https://arxiv.org/pdf/2608.06725v1)

**作者:** Mina Ibrahim `[一作]` (German University in Cairo), Joerg Henkel `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一种基于FPGA的多级单元（MLC）非易失性存储器（NVM）内存控制器，并扩展RISC‑V ISA，新增了快速写指令与按位显著性写入的AXI外设。

**💡 创新点**

创新点包括：①利用写时延与保留时间权衡的快速写指令；②设计按位显著性写入的转置AXI外设，实现MSB采用高保留写入、LSB采用低保留写入；③将控制器与指令扩展与FPGA紧耦合，实现硬件级别的性能优化与面积节省。

**🔧 技术方法**

采用了有限状态机（FSM）+AXI内存映射接口的内存控制器，RISC‑V自定义S‑type指令，FPGA实现的快速写指令与转置外设，SPI/并行NVM接口，以及Verilog/VHDL与GCC RISC‑V工具链。

**📊 数据集**

未使用传统数据集，而是使用多种商业NVM（ReRAM、FRAM、MRAM）芯片进行硬件性能测评，采用Zedboard + FMC/PCB调试平台。

**📈 对比分析**

通过与Xilinx自带内存控制器对比，硬件面积减少30%；在ReRAM、FRAM、MRAM上测量读写延迟，Fast‑Store指令在流式工作负载中提升约7.7%；按位转置外设在64×64矩阵下LUT利用率仅3.5%。

**⚠️ 局限性**

主要限制包括：按位转置外设在更大矩阵尺寸下面积仍会增长；快速写指令的保留时间仅通过延时仿真模拟，未在真实MLC芯片上验证；系统对NVM特定写周期的硬件支持有限，需要针对不同NVM进一步定制。

---

## 80. Bootstrap-Conditioned Action Selection with Tabular Foundation Models

**arXiv ID:** 2608.06559 | [PDF](https://arxiv.org/pdf/2608.06559v1)

**作者:** Devansh Gupta `[一作]` (Amazon Science), Boris N. Oreshkin `[通讯]` (Amazon Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种利用预训练表格式上下文学习模型的 Bootstrap‑ICL 算法，在上下文赌博机框架下通过对交互历史进行自举采样并用冻结的 ICL 模型进行奖励预测，从而实现随机化探索策略。

**💡 创新点**

创新点在于：① 将自举重采样与固定的 ICL 预测器结合，实现无参数在线探索；② 设计了乘法型 arm‑context 表征，促进不同动作间共享统计信息，避免孤立动作的自举失败；③ 在保持预训练模型先验优势的同时，实现了有效的探索与决策。

**🔧 技术方法**

主要技术包括：预训练表格式 ICL 模型（如 TabPFN、TabICL）；Bootstrap 自举采样；乘法式 arm‑context 编码；随机化动作选择；在线评估框架。

**📊 数据集**

在 UCI 公开数据集上测试：Adult、Covertype、Isolet、Letter、Mushroom、Magic Telescope、Shuttle、MNIST 等共八个多类别分类任务。

**📈 对比分析**

与线性、核化、神经网络上下文赌博机基线（LinearTS、LinUCB、KernelUCB/TS、NeuralUCB/TS、BootstrapNN 等）以及随机策略对比。实验结果显示，BC‑ICL（TabPFN/TabICL 版）在大多数数据集上实现了最低的累计 regret，特别在多类别、数据稀疏或冷启动场景下优势显著，提升幅度可达 30–40%。

**⚠️ 局限性**

局限性包括：① 需要预训练模型与目标任务的先验相似，否则性能可能下降；② 自举重采样导致每轮预测成本高，尤其对长时间序列产生计算瓶颈；③ 目前仅使用固定特征表示，未探索在线表示学习或不确定性估计的进一步提升。

---

## 81. Debias in Text, Believe Your Eyes: Text-Anchored Cross-Modal Transfer for Visual Counter-Commonsense Reasoning

**arXiv ID:** 2608.06938 | [PDF](https://arxiv.org/pdf/2608.06938v1)

**作者:** Chen Ling `[一作]` (Zhejiang University), Nai Ding `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出文本锚定的数据构建管线（FFD）和文本锚定跨模态迁移（TACT），通过文本监督重新校准多模大模型的语言先验，提升对视觉反常识推理的能力。

**💡 创新点**

创新点在于将语言先验偏差归因于共享语言解码器的决策过程，并通过仅文本的高质量对抗性事实数据和两阶段后训练（监督+偏好优化）实现跨模态的先验去偏。

**🔧 技术方法**

使用事实频率蒸馏（FFD）构造反常识 QA 语料，采用 LoRA 微调、链式推理轨迹挖掘与直接偏好优化（DPO）对共享解码器进行两阶段训练。

**📊 数据集**

在 CDH‑Bench、CAIT、VLind 等反常识基准以及 MMBench、HallusionBench 等通用 VQA 集合上进行评估，并在 FFD 生成的内部对抗性 QA 集合上进行训练。

**📈 对比分析**

与多种基线（包括 Qwen3‑VL‑8B、Qwen3‑VL‑32B、InternVL3‑5‑38B 及 VCD、NoLan 等偏差缓解方法）进行对比，TACT 在三大反常识基准上平均准确率提升至 80.4%（相较基线 +13.6%），并将先验偏差率降至 18.2%，同时保持原始通用视觉与常识推理能力。

**⚠️ 局限性**

局限性在于只依赖文本监督来纠正先验，可能无法覆盖所有视觉感知缺陷；生成的反常识语料需人工制定先验分类，数据规模与多样性受限；此外对更大模型的适用性和在不同视觉模态下的泛化仍待验证。

---

## 82. Newton-Schulz Retraction-Based Inference Enables Hidden Quantum Markov Models to Outperform Classical HMMs

**arXiv ID:** 2608.06554 | [PDF](https://arxiv.org/pdf/2608.06554v1)

**作者:** Ning Ning `[一作]` `[通讯]` (Texas A&M University), Ning Ning (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 NS-RIS 算法，用于在复数 Stiefel 范围内高效学习隐藏量子马尔可夫模型（HQMM）参数，首次给出该类模型的有限时间收敛保证。

**💡 创新点**

创新点包括：①将 Newton–Schulz 正交化两次作为梯度方向构造和约束恢复的高效方案，避免昂贵的奇异值分解；②在 HQMM 学习中引入了对 Riemannian 方向的极限近似，从而实现更强的收敛性；③提供了第一份数学性能保证，为 HQMM 研究提供了理论基石。

**🔧 技术方法**

使用了 Newton–Schulz 迭代正交化、Riemannian 随机梯度、动量加速、Stiefel 退行映射、投影梯度、以及分布式 mini‑batch 采样等技术；同时在实验中对比了 EM、COSM、GS 等传统方法。

**📊 数据集**

数据集包括：
- 6 状态 6 输出的合成 HMM 数据（20 训练/10 验证序列，长度 3000，切分为 300 短序列），
- 2 隐状态 6 输出的合成 HQMM 数据（同样训练/验证设置），
- 真实生物序列 splice 数据（DNA 长度 60，分为 EI、IE、N 三类）。

**📈 对比分析**

在合成 HMM 基准上，NS-RIS 在所有隐藏维度与 Kraus rank 组合下均超越 COSM，平均提升约 38.5%（最佳情况 50.6%），并优于 EM。 在合成 HQMM 基准上，NS-RIS 相比 COSM 的测试指标提升 18.9%，且运行时间减少 12%。 在 splice 分类任务中，NS-RIS 在 n=6、n=8 时相较 COSM 的平均分类错误分别降低 17.9% 与 14.9%，并且明显优于 EM，表明在更高维隐空间下表现尤为突出。

**⚠️ 局限性**

局限性：
- 对于小隐维或低 Kraus rank 的设置，性能提升不明显；
- 依赖于 Newton–Schulz 迭代精度，若 T_NS 过小可能导致约束恢复误差增大；
- 实验主要集中在小规模合成数据和 splice 数据，尚未验证在更大规模真实序列上的可扩展性；
- 需要对学习率、动量等超参数保持一定的调节，尽管相对稳健。

---

## 83. SoRoMoX: Fast, Differentiable, and Parallelizable Soft Robot Models

**arXiv ID:** 2608.06650 | [PDF](https://arxiv.org/pdf/2608.06650v1)

**作者:** Maximilian Stölzle `[一作]` (Delft University of Technology), Cosimo Della Santina `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 SoRoMoX，一个基于 JAX 的软体机器人模型框架，支持 PCS、GVS 等 Cosserat 软体机器人模型，并提供统一的控制友好接口，实现端到端可微分、GPU 并行和高性能仿真。

**💡 创新点**

创新点在于将软体机器人模型与现代深度学习工具链无缝结合：实现了 JIT 编译、自动微分和批量并行；提供完整的控制所需量化（质心矩阵、关节力、雅可比等）；相较现有工具大幅提升 CPU 速度（≈18×）和 GPU 速度（≈234×）；在系统识别、残差学习、控制优化、RL 等多种任务上验证其效能。

**🔧 技术方法**

使用技术包括：JAX + Equinox（JIT + AD）、Diffrax（数值积分）、GPU/TPU 并行化（jax.vmap）、结构化评估与自动化测试、控制理论（HOCLF/HOCBF）、强化学习（PPO）、传统软体机器人动力学（Cosserat rod、PCS、GVS）。

**📊 数据集**

数据集主要为自建实验数据：软体机器人在静态平衡时采集的标记点位置、传感器读数以及多种施加输入下的运动轨迹，用于系统识别、残差学习和 RL 训练；未使用公开的公共数据集。

**📈 对比分析**

与 SoRoSim（MATLAB）和 PyElastica（CPU）对比：在 CPU rollouts 中 SoRoMoX 速度提升至 18.1×，在 GPU 并行 rollouts 中通过批量加速实现 234.6× 的吞吐量提升；案例研究表明，系统识别 RMSE 降低 66%，残差学习 64%，模型基控制误差相较于 PD 减少约 500 倍，RL 训练时间比基线缩短 4–7×。

**⚠️ 局限性**

局限性包括：仅支持线性连杆、PCS/GVS 软体机器人，未覆盖浮动基、树形或闭链结构；执行器类型有限，缺乏形状记忆合金、电介质弹性体等新兴技术；未提供标准软体机器人描述文件，模型可移植性受限；在极端非线性或复杂几何下的数值稳定性与可扩展性仍待进一步验证。

---

## 84. CrystalGRPO: Target-Aligned and Coverage-Preserving Reinforcement Learning for Flow-Based Crystal Structure Prediction

**arXiv ID:** 2608.06582 | [PDF](https://arxiv.org/pdf/2608.06582v1)

**作者:** Kaixiang Su `[一作]` (University of North Carolina at Charlotte), Qiang Zhu `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 11827 | [OpenAlex ID](https://openalex.org/A5100776456)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种针对流式晶体结构生成器的强化学习后训练框架CrystalGRPO，能够在有限候选预算内提升晶体结构预测的目标复原率。

**💡 创新点**

创新点在于：①将ODE‑to‑SDE策略扩展到联合坐标‑晶格状态，实现对两者的共同随机化和策略梯度更新；②构造混合奖励函数，既利用MACE预测能量提供物理引导，又加入StructureMatcher恢复分数显式识别参考多晶型；③引入两种运行模式CrystalGRPO‑Q和CrystalGRPO‑C，分别针对单次抽样精度和多样性覆盖进行权衡。

**🔧 技术方法**

技术手段包括流匹配模型（FlowMM、OMatG、PXRDGen）、SDE采样、强化学习（GRPO）与混合奖励、覆盖保护机制（全程KL约束与组优势修正）等。

**📊 数据集**

使用公开晶体数据库MP‑20（≤20原子/晶胞）和MPTS‑52（≤52原子/晶胞），并在含PXRD条件的实验数据上进行测试。

**📈 对比分析**

与传统生成器（CDVAE、DiffCSP、CrystalFlow）以及基于RL的OMatG‑IRL进行对比。CrystalGRPO‑Q在MR@1上取得最高提升（如MP‑20从59.06%提升至64.63%），CrystalGRPO‑C在MR@20上表现最佳（如MP‑20从80.23%提升至80.62%），同时RMSE均低于对比基线，显示出在单样本精度与候选覆盖之间的有效权衡。

**⚠️ 局限性**

局限性包括：①对能量预测模型的依赖，若能量估计误差较大会影响奖励信号；②目前主要验证在无机晶体上，对有机分子晶体等更复杂系统的推广仍待验证；③强化学习训练需要额外计算开销，且覆盖保护机制参数需要经验调优。

---

## 85. How Molecular Generative Models Organize Molecular Identity

**arXiv ID:** 2608.06956 | [PDF](https://arxiv.org/pdf/2608.06956v1)

**作者:** Raul Ortega-Ochoa `[一作]` (Toyota Research Institute), Tonio Buonassisi `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过显式定义分子身份并拉回生成过程，研究了三种分子生成模型内部如何将离散分子身份分布在连续生成坐标空间中；

**💡 创新点**

提出了一种基于身份等价关系的“拉回”框架，揭示生成空间由分段常数的分子区域组成，且这些区域的边界呈粗细层级结构；

**🔧 技术方法**

利用等价关系投影、随机“Tape”控制、二维/一维坐标截面、邻域采样、Tanimoto相似性、Jaccard重叠、AUC指标等统计和可视化技术，对模型内部组织进行定量评估；

**📊 数据集**

在ZINC数据库上训练的三种模型：MolMiner、HierVAE、GDSS；

**📈 对比分析**

通过邻域重叠、Tanimoto AUC、欧氏/余弦距离与化学相似度的回归，比较三模型内部组织的化学一致性与坐标距离的对应关系；结果显示MolMiner和HierVAE在精细化学层面表现出显著的聚集性，而GDSS仅在粗粒度等价关系下表现出组织；

**⚠️ 局限性**

局限性在于仅针对单一隐藏空间与单一输出表示（SMILES/图），对更高维、不同输出形式或多任务场景的泛化尚未验证；

---

## 86. LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers

**arXiv ID:** 2608.06867 | [PDF](https://arxiv.org/pdf/2608.06867v1)

**作者:** Tao Feng `[一作]` (University of Illinois Urbana Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种统一的LLM路由框架，将单轮、多轮和个性化路由视为共同的序列决策过程，并开发了一个自动化管道来构建路由监督和评估。

**💡 创新点**

创新点在于提供了一个统一的LLM路由公式，能够将现有的多种路由方法整合为三类，并通过xRouteBench基准进行标准化评估。

**🔧 技术方法**

使用了序列决策过程的框架，结合上下文编码器、模型编码器、评分函数、决策规则和学习信号等组件。

**📊 数据集**

使用了xRouteBench数据集，该数据集涵盖了通用LLM任务、记忆增强、视觉（图像和视频）、时间序列和个性化路由场景。

**📈 对比分析**

通过统一的评估协议比较不同路由器的性能，发现学习型路由器在响应质量和推理成本上相较于固定模型基线有14.6%的相对提升，且在更严格的成本约束下，轻量设计的路由器排名反转，个性化路由则带来了持续的个性化收益。

**⚠️ 局限性**

限制在于现有的路由器在不同形式主义下开发，缺乏统一的评估管道，导致难以进行公平比较和进一步扩展。

---

## 87. Stockmark-Nemotron-3-Nano-Omni-JapanDocReader: Structured Document Parsing via Capability Injection and Forgetting Control

**arXiv ID:** 2608.06758 | [PDF](https://arxiv.org/pdf/2608.06758v1)

**作者:** Shi Chen `[一作]` (Stockmark Inc.), Kosuke Arima `[通讯]` (Stockmark Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过混合监督微调（SFT）与基于奖励的解析强化学习（DAPO）对Nemotron‑3‑Nano‑Omni模型进行后训练，使其能够输出完整的页面级JSON结构化文档解析结果，同时尽量保留原有的日文文档VQA能力。

**💡 创新点**

1）设计了一套可合成的日文VQA与结构化解析数据引擎；2）提出思考模式（带英文推理轨迹）与指令模式对VQA遗忘的影响；3）通过混合SFT与DAPO解析RL实现能力注入与遗忘控制的闭环。

**🔧 技术方法**

利用Mamba2‑Transformer+MoE Nemotron‑3模型，采用监督微调（SFT）与强化学习（GSPO/DAPO）技术；奖励设计基于结构化有效性、文本相似度、表格Teds、公式CDM、边界框IoU等；同时使用英语推理轨迹保持推理风格。

**📊 数据集**

合成日文VQA流（约94k例）与结构化解析流（约25k例），以及OmniDocBench‑JASyn（520张合成日文文档）和三大VQA基准（JA‑Business‑Doc‑RQ‑Bench、JGraphQA‑Refined、JDocQA‑Refined）。

**📈 对比分析**

在OmniDocBench‑JASyn上与Qwen3.6/3.5、gemma‑4‑31B‑it等开源模型对比，所发布模型Stockmark‑Nemotron‑3‑Nano‑Omni‑JapanDocReader取得DocParse‑Overall 87.67分，超过所有对比模型；在VQA基准上保持约0.84的整体分数，较基线仅略有下降。

**⚠️ 局限性**

主要限制在于：1）在解析性能提升的同时，VQA多步推理能力仍有衰退；2）奖励设计与样本过滤对RL效果高度敏感；3）模型规模庞大，需昂贵的算力与长推理时间，限制了在资源受限环境下的部署。

---

## 88. Confirming Our Biases? Evaluating the Capabilities, Risks, and Societal Impact of Large Language Models

**arXiv ID:** 2608.06977 | [PDF](https://arxiv.org/pdf/2608.06977v1)

**作者:** Mudar Adas `[一作]` (University of Tübingen), Martin V. Butz `[通讯]` (University of Tübingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统评估六款大语言模型在 160 条不同结构的提示下，探究显式支持/挑战指令与隐式观点表达对模型回答的操控与确认偏差程度。

**💡 创新点**

提出将“操控效应”“确认偏差效应”“平衡推理效应”三种效应分别量化，并通过 1/2/3 级回答分类（服从/不服从、对齐/不对齐、平衡推理）阐释模型对提示的敏感性与极限；同时检验推理强度对操控影响的非显著性，揭示推理容量并非减轻偏差的关键。

**🔧 技术方法**

采用系统化提示设计、二元分类与比例检验（One‑sample proportion test），对每种提示模式、主题、动词、极性、模型和推理设置进行统计显著性检验。

**📊 数据集**

数据集：160 条自造提示，覆盖 10 个主题（包括 4 个观点性、6 个事实性），每条提示包含“I believe/think”+正/负极性+支持/挑战/对齐/不对齐四种回应指令；每条提示提交 10 次得到 14 400 条模型输出。

**📈 对比分析**

对比方法：计算每种实验条件下的服从率、对齐率和平衡推理率，并与理论基准 0.5 进行比例检验；结果显示显式支持/挑战下服从率普遍在 0.63–0.84 之间显著高于 0.5，隐式对齐/不对齐下对齐率仅 0.43–0.57，说明显式操控对模型影响更强；各模型间的差异也被揭示。

**⚠️ 局限性**

局限性：仅评估 6 款模型、10 个主题、两种动词和两种极性，缺乏跨语言、跨文化、跨任务的普适性；提示设计人为构造，可能与真实对话场景存在差距；未对模型内部机制进行解释，无法识别偏差根源；平衡推理率普遍低，未深入探讨其生成机制。

---

## 89. Do Audio Language Models Use Paralinguistic Evidence? Counterfactual Audits for Response Evaluation

**arXiv ID:** 2608.06718 | [PDF](https://arxiv.org/pdf/2608.06718v1)

**作者:** Kevin Miller `[一作]` (Boston University), Venkatesh Saligrama `[通讯]` (Boston University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文使用反事实审计方法，对音频语言模型（ALM）作为评判者的情感、韵律等非语言线索判断能力进行系统评估。

**💡 创新点**

创新点在于引入三段式审计流程（原生单上下文判断、对比可恢复性控制和感知-映射-判断组件分解），揭示了“Potemkin”失败与“shortcut”成功等多种判别器失效模式。

**🔧 技术方法**

主要技术包括音频合成（TTS）、对比实验设计、三元组诊断状态（P,O,J）以及统计分布分析与置信区间计算。

**📊 数据集**

实验数据来源于CAVA、Open Dialogue (OD3)、EmoCF 等语料库，同时利用合成语音和真实人声进行验证。

**📈 对比分析**

在 Pointwise（原生单上下文）与 Pairwise（对比可恢复性）两种协议下对 Gemini、GPT 及开源模型进行比较，结果显示对比可恢复性高但原生判断低，且按 (P,O,J) 状态分解后可识别不同失败来源，整体准确率往往低于预期。

**⚠️ 局限性**

主要局限在于过度依赖合成语音，缺乏对多语言、多说话人、不同录音环境和情绪分类的覆盖，且未给出针对性模型改进建议。

---

## 90. UniCycleFlow: Bidirectional Unpaired Image Translation with a Shared Rectified Flow

**arXiv ID:** 2608.06784 | [PDF](https://arxiv.org/pdf/2608.06784v1)

**作者:** Xianhao Zhou `[一作]` (University of Electronic Science and Technology of China), Guotai Wang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 UniCycleFlow，通过单一时间条件速度场的正负时间积分实现双向无配对图像翻译。

**💡 创新点**

创新点在于将两方向视为同一连续动力学的正负时间过程，并通过源条件端点对齐、路径自匹配、循环闭合和表征路径速度正则化等手段实现结构保持和一致的变换。

**🔧 技术方法**

使用的技术包括直流流 (rectified flow) 结合时间条件 UNet、对抗端点匹配、stop‑gradient 自流匹配、离散循环闭合、表征路径速度正则化 (RPV)、PatchGAN 判别器以及冻结的 VGG16 表征。

**📊 数据集**

使用的实验数据集涵盖十个双向翻译任务：夏季↔冬季、马↔斑马、猫↔狗、野生↔狗、航空图↔地图等，所有任务均为无配对图像。

**📈 对比分析**

与 CycleGAN、CUT、DCLGAN、SANTA、UNSB 等基线在十个方向上进行比较，单步推理下平均 FID 55.1、KID×100 2.107，显著优于最强基线 DCLGAN 的 62.6/2.415，且多步推理仅略有提升。

**⚠️ 局限性**

局限性包括：采用确定性速度场无法自然产生多模输出；自流匹配对端点质量高度依赖，训练早期误差可能影响中间状态监督；RPV 受冻结表征几何限制，可能无法完全捕捉语义层面的局部变化。

---

## 91. SLED: Scalable Location Encoding via Distillation

**arXiv ID:** 2608.06612 | [PDF](https://arxiv.org/pdf/2608.06612v1)

**作者:** Kevin Lane `[一作]` (University of Colorado Boulder), Morteza Karimzadeh `[通讯]` (University of Colorado Boulder)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于蒸馏的可扩展位置编码器SLED，能够用小批量训练并支持多模态学习

**💡 创新点**

创新点在于将地理位置视为绑定模态，利用蒸馏而非InfoNCE对齐，避免了大批量与误负样本问题，并通过线性对齐头实现多模态自由组合

**🔧 技术方法**

使用MSE蒸馏损失、RFF位置编码、线性对齐层，教师网络为TerraFM（Sentinel‑2）、FG‑MAE（Sentinel‑1）和MoCo（Landsat）

**📊 数据集**

训练数据包括Sentinel‑2（100K样本）、Landsat 8/9（250K样本）和Sentinel‑1 SAR（20K样本），覆盖光学与雷达多模态

**📈 对比分析**

与现有的SatCLIP和GeoCLIP对比，SLED在19个以人为中心的基准任务上性能相当甚至优于SOTA；训练时间可缩短约47倍，且可在batch 128下完成

**⚠️ 局限性**

局限性包括：对SAR模态的提升不稳定，部分任务（如社会指标）仍逊色于地面视角模型；需更多教师/模态多样性来进一步提升表现

---

## 92. Social Facilitation of Creative Reflection: AI-agents and Humans

**arXiv ID:** 2608.06980 | [PDF](https://arxiv.org/pdf/2608.06980v1)

**作者:** Olga Sutskova `[一作]` (University of the Arts London), Corey Ford `[通讯]` (University of the Arts London)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨了社会促进效应对创意反思的影响，并提出了研究框架。

**💡 创新点**

将心理学的社会促进理论与创意支持工具相结合，开辟了AI伴随者在创意反思中的作用研究。

**🔧 技术方法**

主要使用心理学理论（如社会促进效应、群体流动理论）与人机交互设计思路。

**📊 数据集**

未使用实验数据集，仅基于文献综述和理论推导。

**📈 对比分析**

未进行实验比较，未给出性能指标。

**⚠️ 局限性**

缺乏实证验证，AI伴随者是否真正产生同等社会影响仍不确定。

---

## 93. YOLO-PEFT: Parameter-Efficient Fine-Tuning on YOLO Family

**arXiv ID:** 2608.07051 | [PDF](https://arxiv.org/pdf/2608.07051v1)

**作者:** Xu Lin `[一作]` (Tencent), Yong Liu `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

无法确定具体研究内容，因论文正文缺失

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定所用技术

**📊 数据集**

无法确定使用的数据集

**📈 对比分析**

无法确定比较方法及性能表现

**⚠️ 局限性**

缺乏足够信息，无法判断限制

---

## 94. The Nocturnity Scale: Measuring the Sense of Being at Night in Virtual Urban Environments

**arXiv ID:** 2608.06904 | [PDF](https://arxiv.org/pdf/2608.06904v1)

**作者:** Anthony Le Gourri{é}rec `[一作]`, Myriam Servi{è}res `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了“夜间感知度”（nocturnity）概念并构建了一个基于文献的三子量表（感知、活动、内在状态），共42项Likert量表并进行专家评审

**💡 创新点**

首次系统化定义并量化虚拟城市环境中的夜间体验，构造了理论驱动的多维框架和完整问卷

**🔧 技术方法**

采用文献回顾、理论构建、专家访谈和问卷设计方法，未使用实验技术

**📊 数据集**

未使用具体数据集，问卷设计以专家经验为基础，待后续在虚拟现实环境中收集数据验证

**📈 对比分析**

尚未开展对比实验或性能评估，未来计划在VR场景中检验问卷的区分度和可靠性

**⚠️ 局限性**

目前缺乏实证验证，问卷内容对情境依赖性和清晰度仍需进一步细化，且尚未进行心理测量学分析

---

## 95. GPTKB 2.0: Browsing, Querying, and Auditing a Disambiguated LLM-Derived Knowledge Base

**arXiv ID:** 2608.06992 | [PDF](https://arxiv.org/pdf/2608.06992v1)

**作者:** Yujia Hu `[一作]` (TU Dresden), Simon Razniewski `[通讯]` (TU Dresden)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

GPTKB 2.0 是一个从大型语言模型直接构建的去歧义化通用知识库，并提供可追踪的知识推导、SPARQL 与自然语言查询以及文本实体链接的网页界面。

**💡 创新点**

创新点在于：①在构建过程中实时基于上下文进行实体去歧义，既能区分同名异义体又能合并同义表述；②提供透明的推理轨迹，用户可查看每条事实的表面形式、候选匹配和上下文；③将LLM隐式知识转化为可查询、可检验的结构化数据。

**🔧 技术方法**

技术主要包括：LLM 调用（如 GPT‑4）进行三元组诱导、命名实体识别、基于嵌入相似度的实体匹配；在后端使用 OpenLink Virtuoso 存储并暴露 SPARQL 接口；前端基于 Django、Nginx 与 GRASP 代理实现自然语言问答与实体链接。

**📊 数据集**

数据集：基于 GPT‑4 生成的事实，最终得到 38.4 M 条三元组、1.6 M 个规范化实体、207.6 K 条关系、66.5 K 条类别，约 36.8 % 的实体在 Wikidata 中未出现。

**📈 对比分析**

方法通过人工评估（400 条 NED/200 条三元组）和自动评估（1000 条）与现有 LLM 诱导 KB 进行对比，三元组真实率超过 90%，实体可验证率 92–96%，显示出高质量且可比性优于单纯文本提示方法。

**⚠️ 局限性**

局限性：仍存在少量同义词错误合并（约 9%），对不常见表述的识别敏感，且高度依赖上下文描述，缺少外部知识库支持导致某些领域覆盖不足。

---

## 96. TRACE: A Multi-Layer Benchmark for Human AI Controller Coordination Under Drift and Failure

**arXiv ID:** 2608.06657 | [PDF](https://arxiv.org/pdf/2608.06657v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 97. KNOWPLAN: Knowledge-Driven AI Agents for Smart Degree Pathway Planning

**arXiv ID:** 2608.06530 | [PDF](https://arxiv.org/pdf/2608.06530v1)

**作者:** Shuheng Cao `[一作]` (University of California San Diego), Zhaoxiang Feng `[通讯]` (University of California San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种“先抽取后规划”的框架KnowPlan，先通过CatalogBrowse从公共源重建完整课程体系，再用DegreeMap基于用户信息生成个性化学位规划。

**💡 创新点**

创新点在于将课程重构与规划分离，CatalogBrowse使用低置信度的边际覆盖估计和闭包证明确保完整性，而DegreeMap通过有向超图和层次化CP‑SAT优化实现可证可验证的规划。

**🔧 技术方法**

使用了Web探索代理、基于句子到抽象语法树的模型回退、闭包证书、三份关联源JSON、Typed Requirement Hypergraph、CP‑SAT（Google OR‑Tools）以及层次化（lexicographic）优化。

**📊 数据集**

在包含100所大学的公开课程目录（网页、JSON端点、PDF）以及6所大学的稠密实验集上进行评估。

**📈 对比分析**

与最强基线相比，CatalogBrowse在库存召回率和源恢复率上表现突出，并且使用的源访问量比全量抓取少约30%；DegreeMap在硬可行性维持90%以上的同时，个性化效用提升约25%，与黄金课程图相比仅有0.015的效用缺口。

**⚠️ 局限性**

局限性包括依赖公开可访问的课程信息，可能无法覆盖隐藏或非标准格式的先决条件；模型在新机构或大规模课程变更时需要额外的重建；同时对超大规模高校的计算成本和并发访问仍需进一步优化。

---

## 98. Rendezvous of Mobile Deterministic Automata in Graphs

**arXiv ID:** 2608.06482 | [PDF](https://arxiv.org/pdf/2608.06482v1)

**作者:** Bibhuti Das `[一作]` (University of Quebec in Ottawa), Andrzej Pelc `[通讯]` (University of Quebec in Ottawa)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在无标签图中使用可移动或固定弹珠的相同有限状态机（DFA）实现两代理在图中会面的“Rendezvous”问题。

**💡 创新点**

证明在所有图实例中即使拥有任意有限可移动弹珠也不存在RV‑通用DFA，而在树图中使用单个可移动弹珠可以构造出一种通用的DFA，实现O(n²)时间的会面。

**🔧 技术方法**

采用状态机模型、基本行走（Basic Walk）策略、弹珠标记、端点对称性分析及递归阶段化算法设计。

**📊 数据集**

论文未使用具体实验数据集，而是通过理论构造和证明得到结论。

**📈 对比分析**

通过数学证明和构造对比，展示在树图上可实现会面，并给出时间复杂度为O(n²)；相较于无弹珠或多弹珠的情况，证明了可移动弹珠的关键性。

**⚠️ 局限性**

仅适用于树形图；在更一般的图上仍无法得到通用DFA；且算法实现对初始延迟和节点度的假设有限，未讨论大规模或动态网络场景。

---

## 99. When One Modality Is Not Enough: Multimodal Sex and Life-Stage Classification of Red Deer from Aerial RGB-Thermal Video

**arXiv ID:** 2608.06973 | [PDF](https://arxiv.org/pdf/2608.06973v1)

**作者:** Hugo Markoff `[一作]` (Aalborg University), David C. Schedl `[通讯]` (University of Applied Sciences Upper Austria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

通过将无人机拍摄的RGB与热成像视频进行跨模态融合，构建一个能够在个体层面对红鹿进行物种、性别与生命阶段自动分类的完整流程。

**💡 创新点**

创新点在于：①把两种模态在特征空间而非像素级进行融合；②引入跨模态确认（track-level）来剔除噪声并提升检测可靠性；③采用帧级遮挡过滤与多帧投票，避免单帧标签噪声；④利用体积（箱体面积）作为生命阶段的生物测量；⑤将所有步骤放在同一流水线中，直接从无人机影像得到可审计的个体级人口结构。

**🔧 技术方法**

技术主要包括：YOLOv8 单类检测、SORT 追踪、DINOv3 ViT-H+ 的自监督特征提取、三重投票机制（遮挡、物种、性别）、triplet 训练的投影层、正则化的正交投影与姿态校正、基于地理坐标的重复识别与箱体面积阈值判定。

**📊 数据集**

使用公开的 BAMBI 数据集（包含温度与RGB对齐的森林边界与森林内部拍摄的红鹿、鹿、野猪等），并在四个测试航班（A1、A2、B、C）上验证。

**📈 对比分析**

与单模态（仅RGB或仅热）相比，融合方法在物种识别上达到 96% 的准确率，性别识别在 8 只成年雄性中能正确判定 7 只；单模态仅能识别 2 只雄性。检测层面，热成像在遮挡/阴影条件下表现更好，mAP@50-95 在热视角下从 0.562 提升至 0.741；RGB 在正射影像下性能显著下降。整体而言，跨模态融合显著提升了检测与分类的鲁棒性。

**⚠️ 局限性**

主要局限：① DINOv3 对个体的辨别力不足，导致再识别召回率有限；② 雄性样本稀缺，单模态模型易出现误判；③ 仅使用轴对齐框，体积测量受朝向影响；④ 需要大量标注与手工验证的训练集；⑤ 融合策略仍基于经验阈值，可能不适用于极端环境或其他物种。

---

## 100. Learning to Predict Middle-Layer Attention in MLLMs for Visual Token Prunin

**arXiv ID:** 2608.06411 | [PDF](https://arxiv.org/pdf/2608.06411v1)

**作者:** Yuyao Sun `[一作]` (Beihang University), Minjun Yu `[通讯]` (Shanghai Eabot Technology Co Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为MAP的视觉令牌剪枝方法，能够在多模态大型语言模型中在不显著损失性能的前提下大幅减少视觉令牌数量；

**💡 创新点**

创新点包括：1）设计Question Contrastive Teacher Selection（QCTS）机制，针对每个样本自动挑选最能体现问题相关性的中间层作为教师；2）将教师层的注意力分布蒸馏到轻量化预测器，实现无需运行大模型即可提前估计视觉令牌重要性；3）在预测的重要性基础上加入多样性约束，实现既关注问题相关性又避免冗余的令牌选择；

**🔧 技术方法**

核心技术主要包括：文本-视觉注意力聚合与归一化、Jensen–Shannon散度用于QCTS、轻量化注意力预测器（仅含查询/键投影并使用RoPE）、基于正交残差的多样性采样策略；

**📊 数据集**

实验使用的主要数据集包括LLaVA-NeXT-7B、LLaVA-1.5-7B、Qwen2.5-VL-7B-Instruct在GQA、SQA、TextVQA、POPE、MME、VQA-v2、MMBench、VizWiz、SEED-Bench等十个视觉语言基准；

**📈 对比分析**

与VisionZip、DART、CDPruner、MMTok、ZOO‑Prune、LearnPruner、FastV等现有剪枝方法对比，MAP在保留5.56%视觉令牌时仍能保持97.5%原模型性能，获得3.09×端到端加速，预填充阶段可达7.44×速度提升，KV缓存显著降低；

**⚠️ 局限性**

局限性包括：需在训练阶段对每个样本做QCTS并蒸馏，增加了离线准备成本；预测器的精度受限于训练数据和教师层选择的误差；在极低令牌保留率下多样性增益有限；方法目前主要验证在三大MLLM体系结构，尚未在更大模型或实时场景中充分评估。

---

## 101. On the Hardness of Strong Metric Dimension

**arXiv ID:** 2608.06747 | [PDF](https://arxiv.org/pdf/2608.06747v1)

**作者:** Prafullkumar Tale `[一作]` `[通讯]` (Indian Institute of Science Education and Research Pune), Prafullkumar Tale (Indian Institute of Science Education and Research Pune)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过多种多项式时间归约，证明了强度度（Strong Metric Dimension）问题在图的直径为 2 的情况下以及在路径宽度与反馈顶点集数均为常数的图中仍为 NP‑完整。

**💡 创新点**

创新点在于：
- 提供了一个比之前更简单、直观的归约，展示强度度在直径 2 图上的 NP‑完整性；
- 设计了复杂的门控（portal）和半全局顶点结构，构造了在常数路径宽度和反馈顶点集数下的图，并证明其强度度与 Vertex Cover 等价，从而完成了 NP‑完整性的证明。

**🔧 技术方法**

主要技术手段包括：
- 通过将 3‑SAT 归约到 Vertex Cover，再将 Vertex Cover 通过补图和全局顶点转换成强度度问题；
- 构造“portal”路径和半全局顶点来控制最短路径长度，以实现对互相最大距离（mutually maximally distant）属性的精确编码；
- 利用强度分辨图（Strong Resolving Graph）与 Vertex Cover 的对应关系，证明两者的最小解等价；
- 使用路径宽度、反馈顶点集数等结构参数的性质，对归约图进行分析。

**📊 数据集**

本文没有使用实验数据集，所有结果均为理论证明。

**📈 对比分析**

本文未进行实验比较或性能评估，因而不存在方法比较与性能结果；该工作仅提供理论上的复杂度下界与等价性证明。

**⚠️ 局限性**

局限性：
- 归约过程极其繁复，难以直接转化为可用于实际算法的启发式；
- 结果仅适用于特定参数（直径、路径宽度+反馈顶点集数），对其他常见参数（如树宽、度数等）未给出完整结论；
- 仍开放问题：是否在路径宽度+反馈顶点集数下也存在 NP‑完整性，以及是否存在更高效的 FPT 算法。

---

## 102. The Horizon Gap: Planning, Memory, Execution, Training, and Evaluation for Long-Horizon LLM Agents

**arXiv ID:** 2608.06663 | [PDF](https://arxiv.org/pdf/2608.06663v1)

**作者:** Mingguang Chen `[一作]` (DeepGrounding), Bo Qu `[通讯]` (DeepGrounding)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统梳理2024‑2026年arXiv长时序语言模型论文，构建1,547篇语料并提出六大技术维度与两轴分类，分析“horizon gap”及其测评与安全空白。

**💡 创新点**

首次将长时序、长上下文、长期记忆三维度与任务生命周期六大模块相结合，并引入“horizon携带位置”轴，揭示跨领域论文共性与关键未解测量问题。

**🔧 技术方法**

采用文献挖掘、TF‑IDF+SVD+K‑means聚类、规则+人工标注、OpenAlex元数据查询与语义可视化等技术。

**📊 数据集**

使用arXiv API检索（2024‑01至2026‑07）与OpenAlex作者信息，形成包含1,547篇论文的公开语料库。

**📈 对比分析**

通过语义图与时间线对论文数量、领域占比和2026年增长趋势进行量化对比，未提供单篇模型性能比较。

**⚠️ 局限性**

局限于单标注者、规则基础标注、非英语/行业论文缺失、缺乏对实际模型性能的实证验证以及评测与训练信号相互关联偏差的深入实验。

---

## 103. Scenix: Sparse-View 3D Scene Reconstruction via Executable Scene Programs

**arXiv ID:** 2608.07012 | [PDF](https://arxiv.org/pdf/2608.07012v1)

**作者:** Kai Li `[一作]` (City University of Hong Kong), Weikai Chen `[通讯]` (LIGHTSPEED)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们提出了一种基于稀疏无标定RGB视角的可编辑3D室内场景生成框架，直接预测可执行场景程序并实例化成可编辑的3D模型。

**💡 创新点**

创新点在于将结构化场景程序作为中介实现跨视角推理，结合观察一致监督、资产对齐与闭环细化，并构建了110k案例的 Image‑3D Scene Layout 数据集。

**🔧 技术方法**

核心技术包括多视角自回归 LayoutVLM 生成程序、Asset Grounder 对象匹配与开放词汇3D资产生成，以及 Critic–Editor–Verify 闭环细化，并基于 Qwen3.5-4B/9B 语言模型。

**📊 数据集**

我们使用自建的 Image‑3D Scene Layout 数据集（约110k室内场景，含 InfiniGen 合成视角与 SUN RGB‑D 真实图像），以及 SpatialGen、SUN RGB‑D、ScanNet 等公开数据进行评估。

**📈 对比分析**

在内部测试和 SpatialGen 分布外场景中，与单视角基线（SAM3D、Gen3DSR、3D‑Fixer）相比，LayoutVLM 在定位F1、3D IoU、姿态误差等指标上提升显著，闭环细化后整体F1提升达34%–48%。

**⚠️ 局限性**

局限性包括在分布外场景仍存在定位误差和实体过度生成，遮挡与光照不确定性难以完全解决，以及对小装饰物噪声敏感，整体鲁棒性仍待提升。

---

## 104. Theoretical Foundations of Communication-Efficient, Robust, and Practical Distributed and Federated Optimization

**arXiv ID:** 2608.06563 | [PDF](https://arxiv.org/pdf/2608.06563v1)

**作者:** Grigory Malinovsky `[一作]` `[通讯]`, Grigory Malinovsky

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

论文内容未提供，无法生成摘要

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

## 105. EntropyMoE: Entropy-Aware Sparse Expert Routing for Tokenizer-Free LLMs

**arXiv ID:** 2608.06398 | [PDF](https://arxiv.org/pdf/2608.06398v1)

**作者:** Bo Liu `[一作]` (University of Bristol), Yongping Zhang `[通讯]` (Beihang University)

**通讯引用:** 1889 | [OpenAlex ID](https://openalex.org/A5100714501)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于字节动态补丁的稀疏专家架构EntropyMoE，利用补丁熵作为路由信号，将字节补丁路由到Top-2专家；

**💡 创新点**

创新点在于将传统密集前馈层替换为仅使用标量熵的稀疏专家路由，显著减少路由参数（仅400个）并实现按字节覆盖量的工作量计数；

**🔧 技术方法**

采用BLT式字节补丁Transformer、Mixture‑of‑Experts（Top‑2）、熵路由、字节加权负载平衡与密集复制初始化；

**📊 数据集**

使用FineWeb‑Edu预训练数据，评估语料下的语言建模和下游任务（PIQA、HellaSwag、ARC‑E/C、OpenBookQA、BoolQ、MMLU）;

**📈 对比分析**

在匹配数据、更新、激活参数和初始化的对照实验中，EntropyMoE在BPB上最低（0.8351 vs 0.8442 dense），在下游平均准确率略高（Avg‑6 50.31% vs 50.20% dense），且在两种随机种子上保持一致；

**⚠️ 局限性**

局限性包括稀疏实现仍比密集模型慢，缺乏对专家专化的功能验证，未分离熵与补丁长度的影响，且实验仅覆盖继续训练阶段，未探讨不同规模或数据域的泛化。

---

## 106. Towards Multi-Label Graph Foundation Models: from Single-Vector Representation Learning to Multi-Semantic Basis Learning

**arXiv ID:** 2608.06394 | [PDF](https://arxiv.org/pdf/2608.06394v1)

**作者:** Dongxiao He `[一作]` (Tianjin University), Di Jin `[通讯]` (Tianjin University)

**通讯引用:** 6780 | [OpenAlex ID](https://openalex.org/A5012455357)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种跨域多标签节点分类的图基础模型MSB-GFM，利用多语义基学习、结构原型学习和域不变学习实现跨域迁移。

**💡 创新点**

创新点在于把单一向量表示替换为可自适应激活的多语义基组合，既能捕捉多标签语义，又能在不同图域间共享知识。

**🔧 技术方法**

核心技术包括多语义基学习、结构原型学习、域对抗训练以及预训练-微调框架，使用GCN编码器和PCA统一特征维度。

**📊 数据集**

实验采用四个公开多标签图数据集：Humloc、PCG、Blogcatalog、PPI。

**📈 对比分析**

与六种基线（LARN、LIP、CorGCN、AnyGraph、GCOPE、TIG）对比，MSB-GFM在七个评价指标上大多数数据集取得第一或第二名，整体性能显著提升，尤其在单shot跨域场景下效果突出。

**⚠️ 局限性**

局限在于对多语义基数量和结构原型数量需要手动调参，且在极小样本标签稀缺的目标域中仍可能出现偏差，未来需探索更自动化的基数选择和无标签目标域适配。

---

## 107. HRDiT: Training-Free High-Resolution Image Generation with Off-the-Shelf Diffusion Transformer Models

**arXiv ID:** 2608.07003 | [PDF](https://arxiv.org/pdf/2608.07003v1)

**作者:** Yu Xue `[一作]` (Lancaster University), Jun Liu `[通讯]` (Lancaster University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练‑free 的高分辨率图像生成框架 HRDiT，能够在不重新训练的情况下让现成的 Diffusion Transformer（DiT）模型在 2K/4K/8K 级别生成高质量图像。

**💡 创新点**

创新点在于：
① Spatial Position Alignment (SPA) 通过 Bundle 与 Slide 两步对 token 位置嵌入进行重新映射，提升位置表达的可区分性，消除空间失序；
② Head‑adaptive Attention Pruning (HAP) 采用先在一次前向过程中估计每个注意力头在不同窗口尺度下的损失与计算量，随后通过整数规划得到最优窗口分配，实现大幅度剪枝而不显著损失质量；
③ 通过理论与经验分析，明确定位空间失序与长时间生成的根本原因，并针对性设计对应模块。

**🔧 技术方法**

使用的核心技术包括：
- Diffusion Transformer（Stable Diffusion 3、FLUX）作为基础生成器；
- Bundle+Slide 位置对齐技术实现 SPA；
- 单次推理内的 Taylor‑based 损失估计与线性规划实现 HAP；
- FlashInfer 等高效 GPU 核心实现加速。

**📊 数据集**

在 LAION‑5B 数据集上随机抽取 1,000 条 caption 作为提示，评估 2K、4K、8K 三个分辨率的图像生成效果。

**📈 对比分析**

与 Direct、I‑Max、DemoFusion、DiffuseHigh、HiFlow、FreeScale 等现有方法对比，HRDiT 在 FID、FID_p、KID、KID_p、CLIP 等五项指标均优于所有基线；同时在 4K/8K 任务中，生成时延显著下降（约 30–50 %），甚至比最优基线快一半以上。

**⚠️ 局限性**

局限性包括：
① 仍需一次性预处理（计算窗口分配），不适用于实时即插即用场景；
② 对非常高分辨率（>8K）时，细节失真或边缘模糊仍有可能出现；
③ 仅在 DiT 架构上验证，对 U‑Net 或其它生成模型的迁移性尚未充分评估。

---

## 108. From Classification to Recommendation: Empirical Analysis of Audio Embedding Models Application for Content-Based Music Recommendation

**arXiv ID:** 2608.06928 | [PDF](https://arxiv.org/pdf/2608.06928v1)

**作者:** Qingrui Li `[一作]` (University of New South Wales), Lina Yao `[通讯]` (University of New South Wales)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

系统性评估了六种预训练音频编码器在内容检索、顺序推荐和生成式推荐中的表现，并探讨了残差量化设计对生成式推荐的影响。

**💡 创新点**

首次将音频预训练模型与三类音乐推荐系统直接对比，揭示对齐式（CLAP）和音乐专用预训练在生成式推荐中更有优势，并证明增大语义ID宽度而非深度能提升性能。

**🔧 技术方法**

使用预训练音频编码器（Wav2Vec2、HuBERT、Music2Vec、MERT、CLAP-G、CLAP-Music）、残差向量量化（RVQ）、KNN、SASRec、TIGER以及自回归生成模型。

**📊 数据集**

采用LFM1b和Music4All‑Onion两个约5,000曲目子集的音乐推荐数据集进行实验。

**📈 对比分析**

通过 Recall@50、NDCG@50、MRR@50 等指标对比三种推荐器和不同语义ID宽度/深度设置，结果显示CLAP‑Music在大多数场景下最佳，SASRec整体性能最稳健，而更深的RVQ层往往导致性能下降。

**⚠️ 局限性**

仅覆盖两份规模有限的离线数据集，未考虑实时用户体验指标；且仅利用音频特征，未包含艺术家、流派、流行度等其它潜在偏好信息。

---

## 109. CubicQuant: Parametric Non-Uniform Codebooks for High-Throughput LLM Inference with 1-8-Bit Weights

**arXiv ID:** 2608.06763 | [PDF](https://arxiv.org/pdf/2608.06763v1)

**作者:** Xuetian Gao `[一作]` `[通讯]`, Xuetian Gao

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于单调三次曲线的可压缩权重量化格式 CubicQuant，能够在保持整数位流密集打包的同时，通过两个可学习的形状参数自适应地重建权重。

**💡 创新点**

创新点在于将非均匀量化的自由度压缩为可参数化的单调三次曲线，并将其与整数编码紧密耦合，既保留了整数码流的高效 GPU 访问，又显著降低了重建误差；此外，该格式兼容两种执行路径（model‑dtype 与 Dynamic‑A8），且可通过单一压缩文件切换。

**🔧 技术方法**

主要技术包括：1) 参数化三次曲线映射、2) 基于组内规模和形状参数的离散化重构、3) 动态‑A8 carrier‑aware 训练目标、4) GPU 直译张量、5) 统计量化误差理论推导和有限组适配算法。

**📊 数据集**

实验使用三种标准分布（Uniform、Gaussian、Laplace）进行统计量化误差评估，并在 15,360 个样本的 128 维组内进行有限组拟合；随后在 NVIDIA H200 上对不同宽度（W2–W8）进行 kernel 交叉点测量。

**📈 对比分析**

与最优裁剪的 4‑bit uniform integer 以及枚举得到的最佳有限 FP（E1M*, E2M* 等）进行比较；在 G128 实验中，CubicQuant 在 Gaussian 上降低 13.49% RMSE，在 Laplace 上降低 28.14%；在动态‑A8 vs model‑dtype 的 kernel 测试中，行数较大时 Dynamic‑A8 更快，行数较小或宽度较低时 model‑dtype 更快，呈现工作负载相关的交叉点。

**⚠️ 局限性**

限制包括：1) 未评估对最终模型推理质量（如 perplexity、生成性能）的影响；2) 仅测试单一组大小 G128，未给出最佳组大小；3) 性能测量仅在 Hopper（H200）上完成，缺乏多代 GPU 验证；4) 不是 TensorCore 原生格式，需在执行前解码；5) 只支持权重量化，未涵盖激活或 KV‑cache 的量化；6) 需要额外的形状元数据，组大小增大时元数据开销不减小。

---

## 110. KReF: Training-Free Retrieval for Long-Term Time-Series Forecasting and Predictive Uncertainty

**arXiv ID:** 2608.06748 | [PDF](https://arxiv.org/pdf/2608.06748v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 111. Effects of parental controls in the context of Digital Forensics

**arXiv ID:** 2608.07016 | [PDF](https://arxiv.org/pdf/2608.07016v1)

**作者:** Selina Märchya `[一作]` (University of Applied Sciences Bern), Frank Breitinger `[通讯]` (University of Augsburg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对15台 Windows、Android 与 iOS 设备进行实验，系统评估了 Microsoft、Google 与 Apple 家长控制系统对数字取证过程的影响，并给出绕过方法；

**💡 创新点**

创新点在于首次从实验角度全面分析家长控制对取证的阻碍，并提出可在现场实施的绕过与备选云取证方案；

**🔧 技术方法**

采用 ADB、FTK、物理镜像、Google/Apple/Windows 取出工具、手工截图等多种取证技术；

**📊 数据集**

使用15台不同品牌、不同 OS 版本的设备，并在实验中注入多种数据（图片、视频、通讯记录等），形成一个受控的取证数据集；

**📈 对比分析**

通过对比父母模式与子女模式的镜像，发现父母模式下可完整使用 ADB/管理员权限，而子女模式受限，性能差异仅为能否激活调试/获取管理员权限，整体取证质量差异极小；

**⚠️ 局限性**

局限性包括样本量有限、实验环境受限、仅在瑞士测试、手工比对可能存在误差、未覆盖所有第三方控制应用与地区差异。

---

## 112. Agentic AI: User Empowerment or Enclosure?

**arXiv ID:** 2608.06510 | [PDF](https://arxiv.org/pdf/2608.06510v1)

**作者:** David Gamba `[一作]` (University of Michigan), Grant Schoenebeck `[通讯]` (University of Michigan)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对广告拦截器、推荐系统、机器人顾问和垃圾邮件治理四个成熟域的比较案例分析，提出了“构成性政治”框架并将其应用于即将形成的 Agentic AI 治理结构。

**💡 创新点**

创新点在于将技术配置视为政治决策，提出“dep­politicization”概念，识别集体议论的三维条件（可观察性、独立基础设施与治理持续性），并将这些条件映射到协议层面，为 Agentic AI 的治理前景提供先导性分析。

**🔧 技术方法**

主要采用案例研究与比较法，结合历史技术演进与治理机制的定性分析；未开发新算法或模型。

**📊 数据集**

未使用实验数据集，而是依赖公开文献、行业报告、标准文件、法规文本及技术规范作为资料来源。

**📈 对比分析**

通过跨案例比较法评估技术、知识和目标三维中的可访问性、独立性与治理维度，发现个体效益与集体争议能力可能相悖；未给出量化性能指标。

**⚠️ 局限性**

限制在于案例主要聚焦美欧背景，缺乏跨文化或非市场治理视角，框架未经过正式建模，未来 Agentic AI 发展仍不确定，需补充新维度与实证验证。

---

## 113. Model Confidence Under Answer-Preserving Attacks: An Informativeness-Manipulability Frontier

**arXiv ID:** 2608.06571 | [PDF](https://arxiv.org/pdf/2608.06571v1)

**作者:** Reza Khanmohammadi `[一作]` (Michigan State University), Mohammad M. Ghassemi `[通讯]` (Michigan State University)

**通讯引用:** 10525 | [OpenAlex ID](https://openalex.org/A5076266282)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在白盒、仅对图像做攻击且答案保持字节相同的威胁模型下，系统评估并证明多种视觉‑语言模型内部的置信度读数易被操纵，导致置信门控失效。

**💡 创新点**

提出答案字符串上限和无假设的统一幅度证明，并对四个VLM、三类VQA基准、五种置信度通道的可移动性进行系统验证，揭示内部置信度是完整性敏感而非鲁棒的监督信号。

**🔧 技术方法**

采用投影梯度下降、标签感知协调攻击、隐藏状态干预、随机平滑、抗噪预处理等多种攻击与防御技术，并结合精确答案一致性约束进行评估。

**📊 数据集**

使用InternVL3.5‑2B、Gemma‑3‑12B、Molmo2‑4B、LLaVA‑OneVision‑7B 四个模型；GQA、VQAv2、POPE 三个视觉问答基准（各 1000 条样本，正确/错误各一半）。

**📈 对比分析**

通过AUROC、答案字符串上限、统一幅度阈值等指标比较，发现所有部署通道在最强攻击下均跌至或低于答案先验阈值，攻击平均造成 3.7 倍以上准确率下降，未见任何防御能在此威胁模型下保持鲁棒。

**⚠️ 局限性**

实验仅覆盖四种 VLM、三基准、白盒图像仅攻击；未评估跨模型迁移、外部验证器、真实数据范围之外的攻击；假设1、2 的可达性假设在实验中未完全验证，未证明统一幅度证明在更高预算下可行。

---

## 114. SyncSBC: Decentralized Swarm Behavior Prediction for Synchronized Autonomous Control

**arXiv ID:** 2608.06587 | [PDF](https://arxiv.org/pdf/2608.06587v1)

**作者:** Varun Raveendra `[一作]` (University of Utah), Daniel S. Brown `[通讯]` (University of Utah)

**通讯引用:** 941 | [OpenAlex ID](https://openalex.org/A5103065575)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种完全去中心化的实时群体行为分类与同步框架 SyncSBC，可让群体机器人在仅靠局部感知和有限通信下识别全局行为并同步执行。

**💡 创新点**

创新点包括：① 将深度学习的局部感知映射到全局行为分类（SBC模型）；② 设计双阶段共识与同步机制（内部信念 + Boolean Gossip/Integrated Belief Synchronizer），实现时序对齐与低延迟；③ 在同步阶段集成事件触发通信，显著降低消息量。

**🔧 技术方法**

使用技术包括：时序卷积神经网络（TCN）进行行为分类；EWMA 平滑预测；多种共识策略（平均、熵融合、方差更新、样本与保持）；Boolean Gossip 与 Integrated Belief Synchronizer；事件触发通信；离散化的同步延迟评估。

**📊 数据集**

数据集：在 Isaac Sim 中模拟 8 台差分驱动机器人，配备 ToF 传感器，记录多种已知控制器（Cyclic Pursuit、Aggregation、Dispersal）下的时间序列；此外在真实 HeRo+ 机器人上进行实验。

**📈 对比分析**

与传统仅用共识（Pure SH）相比，SyncSBC 在模拟中将同步延迟从 10+ 秒压缩至 <1 秒，测试集分类准确率超过 95%。在真实 8 台机器人实验中，同步延迟降至 <3 秒，远低于 >15 秒。通信量方面，ESI（事件触发 + Integrated Belief Synchronizer）实现最低消息数。

**⚠️ 局限性**

局限性：仅评估了三种相对简单的聚集/巡航/分散行为；对更复杂或多类别行为的泛化尚未验证；依赖 ToF 传感器，感知模式对不同环境和机器人硬件的鲁棒性待进一步研究；在更大规模或高度动态拓扑下的可扩展性仍需深入实验。

---

## 115. Ask-E: An Environment for Calibrated Question Generation

**arXiv ID:** 2608.06933 | [PDF](https://arxiv.org/pdf/2608.06933v1)

**作者:** Sarah Pratt `[一作]` (University of Washington), Ali Farhadi `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了Ask-E评估与训练环境，专门让模型生成难度精准的数学问题，而不是仅仅回答问题。

**💡 创新点**

创新点在于将“问问题”变成评估与训练目标，利用模型间能力差异通过对话自动校准问题难度，避免传统评测中需要更强监督模型的瓶颈。

**🔧 技术方法**

采用多轮对话交互框架、LLM作为提问者、边界求解者和交叉检查者，并用强化学习（CISPO）在该环境中训练模型。

**📊 数据集**

使用20个边界模型（包括17个开源权重模型和3个Gemini Flash）生成的边界对来构造任务；评估数据来自AIME、HMMT、IMO‑AnswerBench等数学基准。

**📈 对比分析**

实验表明最强模型在Ask‑E的校准率约为44.9%，RL训练后模型在AIME、HMMT和IMO‑AnswerBench等基准的pass@8和avg@8都有显著提升，说明问问题训练能迁移到回答问题。

**⚠️ 局限性**

局限性包括：校准率远未饱和，模型仍需更高能力才能充分利用；任务受限于数学领域，对其他知识域的推广需进一步研究；以及依赖预设的边界模型，边界模型能力变化会直接影响评测结果。

---

## 116. Rigid-Covert GNSS Spoofing of UAV Swarms: A Structural Blind Spot, Its Detection Limit, and Absolute-Anchor Defenses

**arXiv ID:** 2608.06885 | [PDF](https://arxiv.org/pdf/2608.06885v1)

**作者:** Minseok Park `[一作]` (Jeonbuk National University), Joon Soo Yoo `[通讯]` (Jeonbuk National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于可信绝对基准的无人机编队恢复方法，用以克服传统相对几何防御在“刚性隐蔽”GNSS欺骗下的结构盲区。

**💡 创新点**

创新点在于揭示相对几何渠道的可观测性缺陷，推导出漂移相关的检测阈值法则，设计联合漂移/攻击的变化点估计器，并实现以少数锚点为基础的全编队绝对位置恢复。

**🔧 技术方法**

采用经典多维尺度(MDS)与RANSAC配合锚点对齐、漂移校正的自举/联合估计器、基于范围的相对几何检测以及多层仿真架构（kinematic、ArduPilot SITL、Gazebo视觉锚点）。

**📊 数据集**

使用了自生成的种子驱动仿真数据、ArduPilot SITL与Gazebo渲染的视觉锚点数据，全部为软件层面的仿真结果，没有真实硬件或现场飞行数据。

**📈 对比分析**

与现有相对几何检测器（距离校验、半正定规划）对比，发现其对刚性偏移无感知；绝对锚点检测器在阈值率以上时AUC可达1.0；恢复实验中将约10.1 m的GNSS漂移压缩至0.39 m，渲染视觉多SITL实验恢复误差为7.1 cm。

**⚠️ 局限性**

局限性包括仅在仿真环境验证、依赖可信绝对基准且要求锚点非共线、锚点覆盖和可见性受限，以及对齐漂移同相攻击（τ→0）和主锚点被攻破的两大根本壁垒。

---

## 117. What Are Developers Actually Discussing When Visual Regression Tests Fail?

**arXiv ID:** 2608.07020 | [PDF](https://arxiv.org/pdf/2608.07020v1)

**作者:** Miku Watanabe `[一作]` (Nara Institute of Science and Technology), Hajimu Iida `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过对307个含Chromatic VRT结果的Pull Request与299个仅含图片附件的Pull Request进行量化比较，并手工对189个VRT触发问题进行卡片排序分类，揭示VRT对讨论、解决时间和代码变更的影响；

**💡 创新点**

首次系统性证明VRT既是视觉风格检查，也能捕捉非局部功能回归，并首次归纳出七类VRT导致的问题类别，展示其在代码审查中的独特角色；

**🔧 技术方法**

结合Chromatic VRT截图与GitHub PR评论的数据关联，采用统计检验（Chi-square、log-rank、Mann‑Whitney U）和卡片排序手工标注技术进行分析；

**📊 数据集**

使用103个GitHub开源仓库中的307个含Chromatic链接的PR和299个仅含图片附件的PR作为实验数据，手工分析的189个VRT问题作为案例集；

**📈 对比分析**

通过对比接受率、合并时间、评论数、提交数、文件变更量及代码行数等指标，发现VRT PR 的中位解决时间约为非VRT PR 的3.8倍，评论数约10倍，代码变更量约1.75–4.5倍，所有差异均在统计上显著；

**⚠️ 局限性**

研究仅覆盖包含Chromatic链接的PR，难以排除VRT使用与PR复杂度的共线性；人工标注存在主观性；未能判断回归是否为有意变更，缺乏因果关系验证。

---

## 118. SNI-GNN: SmartNIC-Assisted Full-Graph GNN Training with In-Network Embedding Prediction

**arXiv ID:** 2608.06441 | [PDF](https://arxiv.org/pdf/2608.06441v1)

**作者:** Guofan Yu `[一作]` (Hong Kong Baptist University), Amelie Chi Zhou `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 473 | [OpenAlex ID](https://openalex.org/A5015692437)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为SNI-GNN的全图GNN训练系统，利用SmartNIC（BlueField‑3 DPU）在网络路径上执行轻量级线性趋势预测，预测远程节点的嵌入，降低跨节点通信量并保持训练精度。

**💡 创新点**

创新点：① 将嵌入预测迁移至SmartNIC，消除CPU/GPU的PCIe竞争；② 采用单一线性趋势预测器与重要性边界节点采样，满足DPU有限计算与内存；③ 设计异步DPU–GPU管线与中间结果重用，隐藏预测延迟；④ 在二阶平滑动态下给出误差与收敛理论，证明在使用不精确梯度时仍能获得非凸收敛。

**🔧 技术方法**

技术实现包括：SmartNIC编程（BlueField‑3 + DOCA），轻量级线性趋势预测器，边界重要性采样，异步RDMA与GPU计算重叠，GPU训练框架（PyTorch + cuDNN），以及通信压缩/重叠与理论误差分析。

**📊 数据集**

使用的公开数据集：Reddit、IGB‑small、ogbn‑products、IGB‑medium；在多层GCN、GAT、GraphSAGE等模型上进行实验。

**📈 对比分析**

与BNS‑GCN（采样型）、SANCUS（历史嵌入）、NeutronTP（张量并行）等现有全图GNN训练框架对比。实验显示通信量下降21–45%，整体速度提升1.3–3.6×相较BNS‑GCN，1.29×相较SANCUS，且在16 GPU、数千万边图规模下仍保持可扩展性，准确率下降≤0.01。

**⚠️ 局限性**

限制与挑战：① 预测效果依赖嵌入随时间平滑性，若第二阶差分不受限误差会增大；② 仍需图划分与边界采样的手工调参；③ SmartNIC资源受限，需平衡采样率与预测精度；④ 只在两类并行模型（分区+张量）上验证，未覆盖其他并行策略；⑤ 在极端非平滑训练初期或大学习率时需频繁同步以防误差放大。

---

## 119. LMM Modality Transfer: A Pre-requisite for Autonomous GIS Agents

**arXiv ID:** 2608.06948 | [PDF](https://arxiv.org/pdf/2608.06948v1)

**作者:** Ivan Majic `[一作]` (Graz University of Technology), Alexandra Fortacz-Lazan `[通讯]` (University of Vienna)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并执行了图像→文本→图像的跨模态空间信息传递任务，评估大型多模态模型在GIS工作流程中的空间推理与保真度。

**💡 创新点**

提出专门针对空间信息的无损模态传递基准，并结合1D Levenshtein、2D Earth Mover’s Distance等定量指标，对完整的跨模态循环进行系统评估。

**🔧 技术方法**

使用OpenAI GPT‑5生成图像描述，GPT‑image‑1生成图像；利用CIEDE2000阈值判定颜色匹配；采用Levenshtein距离衡量序列误差，采用二维空间EMD量化几何偏差。

**📊 数据集**

采用程序生成的规则彩色方格网格图像，尺寸从5×5到10×10，颜色数分别为3、4、5，共计900个样本。

**📈 对比分析**

通过与原始图像直接比较，计算归一化Levenshtein距离和归一化EMD，结果表明网格尺寸和颜色数增大时模型在文本生成阶段失真率迅速升高，图像生成阶段出现严重空间幻觉，整体性能低于预期。

**⚠️ 局限性**

仅评估了两款OpenAI模型，零样本提示导致高复杂度图像缺失完整网格；手工采样图像缺乏可扩展性；基准过于抽象，未涵盖真实GIS数据；未探索少样本或微调的潜在提升。

---

## 120. "Death by a thousand taxonomies?": AI Risk Classification In Practice

**arXiv ID:** 2608.06831 | [PDF](https://arxiv.org/pdf/2608.06831v1)

**作者:** Glen Berman `[一作]` (Australian National University), Ben Hutchinson `[通讯]` (Google Research)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对25位AI研究者和从业者的访谈，系统性研究了社会技术结果分类法（SOT）的开发、使用与在AI治理中的集成程度。

**💡 创新点**

首次揭示SOT在治理中被弱集成的原因，并提出可互操作、可扩展、可追踪的设计原则，建议构建共享注册库以推动标准化。

**🔧 技术方法**

采用定性研究方法——半结构化访谈和反射性主题分析；未使用量化算法或机器学习技术。

**📊 数据集**

使用访谈文本数据（共25份访谈记录），无公开数据集。

**📈 对比分析**

通过访谈内容的主题编码与对比，分析SOT的设计与使用差异；未给出可度量的性能指标或客观比较。

**⚠️ 局限性**

限制：样本主要来自欧美，缺少全球南方视角；仅基于回顾性访谈，未直接观察SOT在实践中的应用；未收集未使用SOT者的反馈。

---

## 121. Online Multi-Level Aggregation with Per-Batch Maximum Delay

**arXiv ID:** 2608.06796 | [PDF](https://arxiv.org/pdf/2608.06796v1)

**作者:** Tianhang Lu `[一作]`, Shengcai Liu `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在有限根树上研究了按批次最大等待时间为目标的在线多级聚合问题，并给出了最优的确定性与随机化竞争比。

**💡 创新点**

提出了 DP‑Envelope 时钟与全局随机偏移技术，证明无论树形结构如何，这两种算法在确定性下达到 2 倍、随机化下达到 e/(e−1) 倍的最佳竞争比；同时将结果推广到任何满足非负、单调、子模的静态服务成本模型。

**🔧 技术方法**

核心技术是基于连续到达块的离线动态规划、子模性引入的延迟阈值计时器、以及对参数化贪心分区的分层分析与积分势能方法。

**📊 数据集**

该工作完全基于理论分析，无使用具体实验数据集；所有结果均来自算法与数学证明。

**📈 对比分析**

与已知的确定性下 2 倍、随机化下 e/(e−1) 的下界相匹配，证明在任何非退化根树上这些算法都是最优的；相比传统的可加延迟模型显著提升了对批次尾延迟的处理。

**⚠️ 局限性**

局限性在于仅针对无适应性（oblivious）对手的随机化算法；对自适应对手未给出证明；此外假设服务成本是可预先计算且满足子模性质，限制了对动态或状态相关服务模型的适用性。

---

## 122. Streaming Algorithms for Monotonicity Testing

**arXiv ID:** 2608.07073 | [PDF](https://arxiv.org/pdf/2608.07073v1)

**作者:** Amir Azarmehr `[一作]` (Northeastern University), Ronitt Rubinfeld `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在无序流模型下，用 O(n) 以内的空间估计有向无环图（或偏序集）上布尔函数的单调性距离，给出一种 (2+ε) 近似的算法，使用约 √n 次遍历。

**💡 创新点**

核心创新是将单调性距离估计转换为求解“违约图”中的最大匹配大小，并在流环境中首次引入子线性查询模型（vertex 与 subset 查询），通过这些查询实现对匹配大小的高效估计，从而突破了传统子线性匹配估计中 Ω(n) 查询的下界；同时给出了匹配估计与单调性距离之间的紧密联系。

**🔧 技术方法**

主要技术包括：
- 通过构造违约图（violation graph）将单调性距离映射为最大匹配问题；
- 设计子线性查询模型，并实现子度估计器（subset degree estimator）与分层“分解匹配”（fractional peeling）算法；
- 通过构造“shortcut set”实现子度查询在流中的高效实现；
- 结合 st‑reachability 的已知流算法，证明 √n 次遍历是最优（除非改进 st‑reachability 的流算法）。

**📊 数据集**

论文主要是理论性研究，没有使用真实数据集，实验与评估全部在理论分析与复杂度证明上完成。

**📈 对比分析**

与之前仅在查询复杂度上研究单调性测试的工作相比，本文在流模型下提供了空间上与时间（遍历次数）上与最优 st‑reachability 流算法相匹配的结果；在匹配估计上实现了 (2+ε) 近似，显著优于传统的 (3/2) 或更差的近似。

**⚠️ 局限性**

局限性包括：
- 仍需 √n 次遍历，除非突破 st‑reachability 的流算法；
- 仅适用于 O(n) 空间；
- 对于非常小的 ε 需要更高的常数因子；
- 需要引入更强的查询模型（vertex/subset 查询），实际实现可能受限于硬件或数据流协议。

---

## 123. MiGHT-EHR: A Multi-task Graph Transformer for Heterogeneous Temporal Electronic Health Records

**arXiv ID:** 2608.06430 | [PDF](https://arxiv.org/pdf/2608.06430v1)

**作者:** Anirudh Rayas `[一作]` (Arizona State University), Pavan Turaga `[通讯]` (Arizona State University)

**通讯引用:** 6092 | [OpenAlex ID](https://openalex.org/A5062945520)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建一张包含患者、就诊、诊断、处置和药物等多种实体的异构时序图，并利用多任务图变换器（带时间注意力）学习共享的访问表示，支持药物推荐、住院时长、死亡率和再入院率等四项临床预测任务。

**💡 创新点**

① 用归一化点互信息（NPMI）自动判定实体共现关系，避免频率偏倚；② 将时间序列信息直接嵌入节点表示并在图变换器中引入跨访问时间注意力；③ 采用双平衡多任务学习（DB-MTL）在梯度层面平衡各任务贡献，防止单一任务主导。

**🔧 技术方法**

主要技术包括：1) 图构建与三类关系（归属、时间、共现）；2) 预训练节点特征（Concept用Bio_ClinicalBERT，Patient/Visit用TransE）；3) 异构图变换器（多头注意力、关系感知消息传递、交叉访问时间注意力）；4) DB‑MTL 共享编码器；5) 评估指标（AUROC、AUPR、准确率、F1、ECE）。

**📊 数据集**

在公开 EHR 基准 MIMIC‑III 和 MIMIC‑IV 两个数据集上进行实验，使用 90/10 访问级拆分进行训练与测试。

**📈 对比分析**

与 14 种基线（GRU、Transformer、Deepr、ConCare、Dr. Agent、AdaCare、StageNet、GRASP、GRAM、GraphCare、MulT‑EHR 以及药物推荐的专用模型）进行对比，MiGHT‑EHR 在四个任务的平均相对提升约 8.7%，尤其在罕见事件预测（死亡率提升 22.2%，再入院率提升 17.9%）显著优于现有方法，药物推荐与住院时长也保持了接近最佳水平。

**⚠️ 局限性**

1) 仅使用对偶共现边，无法完整捕捉一次就诊中多种实体的高阶交互；2) 图结构为静态，只考虑访问顺序的时间边，未对随时间演化的图拓扑进行建模；3) 研究仅在 MIMIC 两个数据集上验证，跨域泛化尚待进一步探索。

---

## 124. StateFlow: Sequence Pipeline Parallelism for Long-Context Modeling with Linear Recurrence

**arXiv ID:** 2608.06838 | [PDF](https://arxiv.org/pdf/2608.06838v1)

**作者:** Wenxuan Zhao `[一作]` (Tsinghua University), Guangwen Yang `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为StateFlow的序列流水线并行系统，用于训练具有线性递归（线性注意力/状态空间模型）以及混合模型的长上下文语言模型。

**💡 创新点**

创新点在于：① 通过将序列划分为块并将每个块作为流水线单元，保持递归状态的前向/后向依赖，提前释放激活；② 对混合模型使用基于FLOP平衡与经验搜索的分块划分；③ 通过状态转移网格优化与前后向重叠，提升GPU利用率并隐藏状态转移延迟。

**🔧 技术方法**

技术包括：序列分块调度、边界状态传递、分块搜索（等长与FLOP平衡混合）、状态转移网格尺寸调优、前后向重叠、Megatron Core 与 Swift框架的流水线实现。

**📊 数据集**

实验使用官方的GDN（Gated DeltaNet）和Mamba-3两种长上下文模型，规模分别为3B、15B、32B，测试上下文长度64K、128K、256K，在NVIDIA A100‑SXM4‑80GB GPU上进行。

**📈 对比分析**

与Megatron、Swift原生PP+TP、TeraPipe、Seq1F1B等对比，StateFlow在3B模型上最高可实现2.22×吞吐加速、2.45×内存缩减；在32B模型上也获得约1.5–2×吞吐提升、超过2×内存缩减，并能完成原方案OOM的配置。

**⚠️ 局限性**

局限性：仅适用于单个训练序列（MBS=1），对分块搜索与状态重叠需要额外的Profile步骤；在高TP度下受SM覆盖限制，且分块划分与重叠配置对不同模型和硬件敏感，需要经验调优。

---

## 125. Where Does AI Innovation Go? Measuring Research Attention Imbalance in AI Music

**arXiv ID:** 2608.06903 | [PDF](https://arxiv.org/pdf/2608.06903v1)

**作者:** Qian Liang `[一作]` (Chinese Academy of Sciences), Ningbo Cheng `[通讯]` (Chinese Academy of Sciences)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了6839篇AI音乐论文的语料库，提出了“研究关注度档案（Research Attention Profile）”框架，利用四个指标量化不同音乐任务在技术投入、方法分配、多样性和前沿方法采用滞后等方面的研究关注度。

**💡 创新点**

首次将应用任务与技术方法联合映射并构建统一分类体系，设计四个可量化指标（TIR、TMIR、NMD、FMAL）系统性评估领域内的技术投入不均衡与前沿方法扩散速度，从而揭示技术与社会需求之间的偏差。

**🔧 技术方法**

使用Semantic Scholar、ISMIR和arXiv三源语料检索，借助LLM辅助分类结合人工验证完成任务与技术标签，采用统计学方法计算技术投资残差、方法分配残差、方法多样性指数与前沿方法滞后，最后进行可视化与时间序列分析。

**📊 数据集**

收集并整合来自Semantic Scholar、ISMIR和arXiv的6839篇AI音乐论文，其中包含3648篇已标注技术方法的论文；该语料涵盖2015年至2026年间的主要期刊、会议与预印本。

**📈 对比分析**

通过比较不同任务（生成、信息检索、推荐、教育、健康、治理等）与技术方法（CNN、RNN、Transformer、扩散模型、基础模型等）的TIR、TMIR、NMD和FMAL，发现生成任务在技术投入与前沿方法采用方面显著领先，教育、健康与治理任务技术投入不足且采用滞后平均超过4-5年。

**⚠️ 局限性**

局限性包括：语料覆盖范围不完全、2026年数据不完整；每篇论文仅标注单一主任务与单一技术，忽略多任务/多方法交叉；首次采用年份可能受早期少数论文影响；指标仅衡量研究关注度，未能直接反映社会需求或实际影响。

---

## 126. SynChain: Inducing Computer-Use Agent Systems to Construct Their Own Attack Chains

**arXiv ID:** 2608.06862 | [PDF](https://arxiv.org/pdf/2608.06862v1)

**作者:** Fuyao Zhang `[一作]` (Nanyang Technological University), Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种自合成攻击范式，揭示了计算机使用代理（CUAs）内部供应链风险，表明恶意影响可以在代理的持久状态中传播。

**💡 创新点**

创新点在于引入了持久性感知的有向监督微调方法，使得代理能够在生成的看似良性的工件中嵌入潜在的恶意载体，从而实现跨任务的妥协。

**🔧 技术方法**

使用了持久性感知的有向监督微调（SFT）技术来优化模型，使其能够进行潜在载体变异。

**📊 数据集**

构建了一个包含30个良性任务链和3个攻击目标的数据集，用于系统评估。

**📈 对比分析**

在OpenClaw、Codex和Claude Code等三个CUA框架下进行评估，攻击成功率（ASR）在Chain-1中超过93%，在Chain-2中保持在72%以上，显著优于适应的基线方法，证明了现有防御措施的局限性。

**⚠️ 局限性**

限制在于攻击的传播深度有限，随着任务链的延长，攻击的有效性会下降，受到内存摘要和上下文截断等自然信息瓶颈的影响。

---

## 127. CellWorld: From Gene-Level Reconstruction to Latent Cell Prediction in Spatial Transcriptomics Foundation Models

**arXiv ID:** 2608.06659 | [PDF](https://arxiv.org/pdf/2608.06659v1)

**作者:** Haiping Liu `[一作]` (University of Manchester), Hongpeng Zhou `[通讯]` (University of Manchester)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了CellWorld，一种通过在空间转录组数据中预测遮蔽细胞的潜在表征来进行自监督预训练的基础模型。

**💡 创新点**

创新点在于将潜在空间预测作为预训练目标，避免直接重建原始测量导致的技术变异；引入细胞级目标提示以缓解目标不确定性，并在大规模多平台数据上验证模型可扩展性。

**🔧 技术方法**

采用细胞token化、二维ALiBi空间注意力、EMA目标编码器、随机遮蔽与局部表达提示，以及大规模Transformer架构进行预训练。

**📊 数据集**

预训练使用46M人类细胞，来源于MERFISH、Xenium、CosMx三种平台；下游评估在四个held‑out数据集（MERFISH人脑、CosMx正常肝、CosMx肝癌、Xenium肺纤维化）上进行。

**📈 对比分析**

与Nicheformer、scGPT‑spatial、CellPLM及PCA等现有方法对比；在所有11个线性探针基准和7个空间微调基准上均实现SOTA；尤其是冻结的CellWorld‑Large在仅5%预训练数据下超越所有完全微调基准。

**⚠️ 局限性**

局限性包括对细胞级目标提示的依赖、在更大模型规模下仍需更充分的优化、对数据多样性依赖强、单一来源或细胞数不足时性能下降，以及缺乏对时间/干预动态的建模。

---

## 128. Test-Time Adaptation with Online Personalized Energy-Based Cache for Fine-Grained Video Expression Recognition

**arXiv ID:** 2608.06467 | [PDF](https://arxiv.org/pdf/2608.06467v1)

**作者:** Masoumeh Sharafi `[一作]` (École de technologie supérieure), Eric Granger `[通讯]` (École de technologie supérieure)

**通讯引用:** 7289 | [OpenAlex ID](https://openalex.org/A5006937759)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于能量模型的缓存个性化方法（EB-CaP），在视频表情识别的测试时刻为每个目标视频在线生成类特定的原型缓存，并结合正负缓存实现无模型更新的自适应。

**💡 创新点**

创新点在于：①利用CLIP视觉-文本相似度引导的轻量能量模型从当前视频中自适应采样多样化的类原型；②引入自适应熵门和多样性门双重控制缓存更新，避免伪标签污染和冗余；③不依赖源域原型，完全在线构建个性化缓存。

**🔧 技术方法**

采用CLIP ViT‑B/32预训练编码器、SGLD采样、熵门与多样性门、余弦相似度融合以及简单的线性融合等技术。

**📊 数据集**

在Heat Pain（BioVid）、StressID、Ambivalence & Hesitancy（BAH）三大公开视频情绪识别数据集上进行实验。

**📈 对比分析**

与零样本CLIP、Fine‑Tuned CLIP以及多种TTA方法（TPT、TDA、DPE、PromptAlign、ReTA、T3AL等）对比，EB‑CaP在三个数据集上均实现最高准确率（平均约81%）并保持较低的运行时与内存消耗。

**⚠️ 局限性**

主要限制包括对SGLD采样超参数敏感、在极少样本或高噪声的目标视频中仍可能产生不可靠的伪标签，以及实验仅覆盖三类数据集，缺乏对更广泛任务的验证。

---

## 129. Beyond Attention: Signed Integrated Gradients Attribution in a BiomeGPT-Style Microbiome Transformer

**arXiv ID:** 2608.06486 | [PDF](https://arxiv.org/pdf/2608.06486v1)

**作者:** Oren Nelson `[一作]` `[通讯]` (University of California, San Diego), Oren Nelson (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于融合感知的符号积分梯度（IG）方法，用来解释BiomeGPT风格的微生物组变压器模型对炎症性肠病（IBD）与健康的预测决策。

**💡 创新点**

创新点在于：①为特征‑token化模型设计了保持物种身份不变、只插值丰度的基线（T'=S+A₀），从而实现对丰度变化的有符号、方向性归因；②将IG与传统注意力权重进行对比，证明IG能揭示正负证据并揭露关注权重无法发现的因果方向。

**🔧 技术方法**

使用了整合梯度（Integrated Gradients）、第二阶积分Hessian（讨论但未实现）、Transformer编码器（BiomeGPT风格）、以及对丰度嵌入的融合路径。

**📊 数据集**

使用的基准数据集为约27,000个MetaPhlAn构建的粪便微生物组样本进行预训练，随后在8,000个IBD/健康平衡样本上微调。

**📈 对比分析**

与传统注意力权重进行对比：注意力仅提供非负排名，IG则给出正负方向归因；在IBD判别任务上模型验证准确率约为93%，表明IG能完整反映模型决策逻辑。

**⚠️ 局限性**

局限性包括：未在外部数据集上评估泛化能力；IG聚合路径可能掩盖丰度在不同区间的条件敏感性；仅使用一阶归因，未实现二阶交互分析；归因结果不等同于生物学因果关系。

---

## 130. Decoupling Intention from Trajectory: A Representational Deduction Framework for World Action Models

**arXiv ID:** 2608.06994 | [PDF](https://arxiv.org/pdf/2608.06994v1)

**作者:** Xiangkai Ma `[一作]` (Nanjing University), Zhihao Yuan `[通讯]` (Joy Future Academy, JD)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 PILOT 框架，通过 Representational Deduction（RD）机制将高层运动意图与低层轨迹生成解耦，压缩状态转移信息为 Motion-CoT 令动作模型专注于细粒度轨迹细化。

**💡 创新点**

创新点包括：① 用 RD 将运动意图从视觉生成中抽取；② Motion-CoT 作为内置的链式推理上下文；③ 利用 Causal Dynamics Engine（CDE）在 VJEPA2-AC 表示空间中监督 Motion-CoT；④ 因果解耦注意力结构防止噪声反向传播，提升表示分离度。

**🔧 技术方法**

技术手段：预训练的 Wan2.2 视频扩散 Transformer 作为世界模型；Perceiver 结构与 flow‑matching 动作解码器；可学习查询 token 提取 Motion-CoT；冻结 VJEPA2-AC 编码器 + 可训练的 CDE；Softmax 以及流匹配损失等。

**📊 数据集**

数据集：LIBERO 与 RoboCasa-GR1 这两大机器人仿真基准；Agibot‑G1 实际双臂人形机器人；以及用于预训练的通用视频数据和 VJEPA2-AC 所需的 RGB 图像对。

**📈 对比分析**

与 Motus、π_0.5、FastWAM、LDA 等基线对比，PILOT 在 LIBERO 上取得 97.9% 成功率，RoboCasa‑GR1 上 62.6%，在 Agibot‑G1 真实世界任务中平均 83.1% 成功率；与 predict‑then‑act 模式相比，推理时延降低 90%；在仅 10% 数据的 few‑shot 微调下性能仅下降 20%。

**⚠️ 局限性**

局限性：① 仍需大规模预训练模型，训练成本较高；② 当前方法主要针对短期轨迹，对极长周期任务尚未充分验证；③ 需要未来图像对作为监督，限制了数据收集范围；④ 在极端分布漂移或高度动态交互场景中，Motion‑CoT 可能不够鲁棒。

---

## 131. Progressive Content Refinement with Decaying Reward Joint LinUCB

**arXiv ID:** 2608.06750 | [PDF](https://arxiv.org/pdf/2608.06750v1)

**作者:** Shion Ishikawa `[一作]` (Rakuten Group, Inc.), Yun Ching Liu `[通讯]` (Rakuten Group, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DR‑LinUCB 算法，结合奖励衰减的上下文多臂赌博机，用 EM 同时学习奖励与衰减参数，以实现 LLM 的迭代细化。

**💡 创新点**

创新点在于将奖励衰减显式建模并与上下文特征联合学习，突破传统线性 UCB 只关注静态奖励的局限，并通过 EM 避免传统 Rotting Bandit 的探索浪费。

**🔧 技术方法**

采用上下文线性 UCB、EM 算法、Prompt 嵌入、线性回归变换及指数衰减模型。

**📊 数据集**

使用 GSM8K 数学推理数据集和 Sentiment Reversal（情感反转）数据集进行评估。

**📈 对比分析**

与单次调用、随机探索、JointLinUCB、EvoLinUCB、Self‑Refine、REx 等基线比较，DR‑LinUCB 在两大任务上均取得最高或相近最高得分，明显优于传统方法。

**⚠️ 局限性**

局限包括仅在易实现任务上验证（难以推广至高难度任务）、多次 LLM 调用导致成本高昂，以及对超参数敏感。

---

## 132. Do 3D Medical Foundation Models See Through MRI Artifacts? A Controlled Study of Representation Robustness

**arXiv ID:** 2608.06613 | [PDF](https://arxiv.org/pdf/2608.06613v1)

**作者:** Julia Anna Mielcarz `[一作]` (University of Copenhagen), Mostafa Mehdipour Ghazi `[通讯]` (University of Copenhagen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

对五个预训练的3D医学影像编码器在受MRI伪影影响下的表示鲁棒性进行系统评估。

**💡 创新点**

提出了对不同伪影（频域与像素域）下的代表性几何（CKA）、谱秩（RankMe）和任务一致性（分割Dice）多维度评估框架。

**🔧 技术方法**

使用线性CKA、RankMe、UMAP、分割一致性评估，结合多种伪影模拟。

**📊 数据集**

BraTS‑Africa 95例多中心脑肿瘤MRI（T1/T1ce/T2/FLAIR）并生成13,300个受控伪影样本。

**📈 对比分析**

在不同模型、伪影类型和严重程度下计算CKA/RankMe并与分割Dice对比，发现3DINO鲁棒性最高，BrainIAC最敏感；伪影常导致几何漂移而非维度坍塌。

**⚠️ 局限性**

缺乏对伪影生成与临床真实分布的匹配，无法分离架构、预训练目标与数据规模的独立影响。

---

## 133. Graph Invariants under Lexicographic Products: Shannon Capacity and Related Parameters

**arXiv ID:** 2608.06573 | [PDF](https://arxiv.org/pdf/2608.06573v1)

**作者:** Igal Sason `[一作]` `[通讯]` (Technion-Israel Institute of Technology), Igal Sason (Technion-Israel Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对图的图论不变量在词典积（lexicographic product）下的行为进行了系统研究，给出了Lovász θ函数、分数Haemers数以及Shannon容量在此积下的乘法性质，提出了对Shannon容量的上下界比较，并在Kneser图、其补图、q-Analog等特殊图类上给出了完全可计算的结果；此外证明了完全外因子保持Shannon容量不变，并求出了自补图的迭代词典积的容量。

**💡 创新点**

创新点包括：① 用简洁的自洽证明重新展示了Lovász θ函数和分数Haemers数在词典积下的乘法性；② 证明Shannon容量在词典积下是超乘法的，并与强积比较，得到一系列新的上下界与等价条件；③ 在Kneser图及其q-Analog的词典积上给出精确的Shannon容量；④ 证明完全外因子不影响容量，并求自补图词典积的容量；⑤ 提出并讨论了关于强积与词典积下Shannon容量关系的开放问题。

**🔧 技术方法**

主要技术手段包括：半正定规划（SDP）形式的Lovász θ函数、Kronecker乘积与秩分解的线性代数工具、词典积与强积的结构性质、以及分数Haemers数的表示与极值论证；同时利用独立集、团数的乘法公式以及图的对称性特征。

**📊 数据集**

由于研究属于纯理论图论和信息论，未使用具体实验数据集；所有结果均通过理论推导获得。

**📈 对比分析**

比较方法主要是通过解析公式与已知的上界（如θ函数、Haemers数、极小色数等）进行大小比较，并在特殊图类上直接计算得到精确值；在Kneser图等例子中得到与已知结果一致且更强的结论，证明方法在理论层面表现优异。

**⚠️ 局限性**

局限性：① 对一般图的Shannon容量仍无法给出精确值，仍依赖已知的上界与下界；② 词典积下的精确容量计算仍受限于图的对称性或特殊结构；③ 对于强积与词典积下容量是否严格相等的问题仍未解决，留有开放问题。

---

## 134. LiFTER: A Grounded Neuro-Symbolic Microscope for Continuous-Time Dynamic Graph Forecasting

**arXiv ID:** 2608.06765 | [PDF](https://arxiv.org/pdf/2608.06765v1)

**作者:** Minwoo Yu `[一作]` (Konkuk University), Young-guk Ha `[通讯]` (Konkuk University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于有界规则的神经符号预测框架 LiFTER，用于连续时间动态图（CTDG）中的未来链接预测，并提供可验证的、可编辑的预测轨迹。

**💡 创新点**

创新点在于：
- 以事件为原子事实而非语义关系，构建有限规则语言；
- 每个预测分数由可执行的规则实例组成，可完整回溯、重新计算与干预；
- 将解释与预测融合，消除解释器与模型的分离；
- 通过完整执行记录实现独立可验证的解释与性能分析。

**🔧 技术方法**

技术手段包括：
- 有界规则枚举（终点绑定、对偶更新、历史位置、顺序转移）；
- 规则权重与时间兼容性学习；
- 基于批量张量操作的高效规则执行；
- 可选的多谓词（K>1）预处理；
- 解释度量（ACC-AUC、Deletion AUFSC）和精确Shapley分解。

**📊 数据集**

使用的公开数据集有：Wikipedia、Reddit、MOOC、LastFM（JODIE 交互序列）。

**📈 对比分析**

与六种基线（TGN、TGAT、GraphMixer、DyGFormer、EdgeBank、T-GNNExplainer 等）进行比较：
- 在历史负样本下，LiFTER 在 Reddit、LastFM 的 AUC/AP 均位居第一，Wikipedia 与 MOOC 仅次于最佳；
- 在随机负样本下，LiFTER 与最强模型相差不超过 1 个百分点；
- 解释质量上，LiFTER 的 macro ACC-AUC 与 Deletion AUFSC 均为所有方法之最。

**⚠️ 局限性**

局限性包括：
- 对多谓词事件流的优势有限（K>1 在当前 JODIE 流上提升不明显）；
- 规则枚举在极大历史窗口下可能导致计算膨胀，需对 H 进行验证；
- 对于完全随机或非时间相关的负样本，模型优势不明显；
- 需要手工设计规则模板，可能限制对特定领域规则的自动发现。

---

## 135. EvoRIC: Reinforcement Learning Fine-Tuned LLM-empowered RAN Intelligent Control Toward Autonomous O-RAN

**arXiv ID:** 2608.06789 | [PDF](https://arxiv.org/pdf/2608.06789v1)

**作者:** Lingyan Bao `[一作]` (Yonsei University), Tony Q. S. Quek `[通讯]` (Singapore University of Technology and Design)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种分层的 EvoRIC 框架，将非实时 RIC 用作 RL 训练和 LLM 微调平台，近实时 RIC 用作边缘推理与控制单元，实现 O‑RAN 的自适应资源管理。

**💡 创新点**

创新点在于：① 将 LLM 作为 RL 代理，通过环境交互进行自主微调，避免传统 SFT 对标注数据的依赖；② 采用闭环“收集‑更新‑部署”循环，持续提升模型对多样化网络拓扑的泛化能力；③ 结合 KL 正则化与格式奖励，显著降低 LLM 的幻觉与不可执行行为。

**🔧 技术方法**

使用的技术包括：Llama‑3.2‑3B‑Instruct 作为基础模型，PPO 训练框架，KL 损失正则化，结构化提示模板与安全验证模块，O‑RAN 接口（E2、A1、O1）实现模型生命周期管理。

**📊 数据集**

数据集来源于仿真的 IAB 网络场景（多 MBS‑SBS 结构、Gaussian‑Markov 用户运动、LoS/NLoS 通道模型），通过收集近实时 RIC 的交互日志构建 Rollout Buffer 进行 RL 训练。

**📈 对比分析**

与 EPA、SCA‑based、未微调 Llama3B、DeepSeek 与 Gemini 进行对比；EvoRIC‑L2 在三种拓扑下平均提升 16.6%–67% 的总吞吐量，且推理耗时仅 0.56 s，显著优于大型 LLM（3.82 s / 1.60 s），说明微调后的紧凑模型兼具性能与低延迟。

**⚠️ 局限性**

局限性包括：① LLM 令牌离散化导致连续参数精度损失；② 探索仍依赖随机温度采样，缺乏高层次的策略多样性；③ 仅处理文本序列，未充分利用多模态信息；④ 对实时细粒度控制与极端传感器故障的鲁棒性尚未完全验证。

---

## 136. A Rate Separation for Agnostic Direct Sums

**arXiv ID:** 2608.06951 | [PDF](https://arxiv.org/pdf/2608.06951v1)

**作者:** Mihir More `[一作]` (Truth Audit Labs), Debayan Gupta `[通讯]` (Truth Audit Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究了在无假设PAC学习框架下，概念类的直接和（product class）的学习曲线与单实例学习曲线的关系。

**💡 创新点**

给出了一个反例：两个单实例类虽然具有相同的单实例学习率，但其直接和的学习曲线随因素数 r 的变化而表现出截然不同的速率，揭示了单实例学习率并不能决定其直接和的学习速率。

**🔧 技术方法**

运用了 Le Cam 的两点法、Pinsker 不等式、Assouad 引理以及有限类经验风险最小化的理论工具来构造下界与上界。

**📊 数据集**

未使用任何真实数据集，而是通过构造合成分布和分布族来证明理论结果。

**📈 对比分析**

通过对构造的分布族分别计算经验风险最小化的上界和使用 Assouad 引理得到的下界进行比较，证明直接和类的学习曲线在 r 维度上可达到 O(n^{-1/2}) 的上界，而下界则是 Ω(min{1, √(r/n)})，表明当 r≥n 时直接和类的学习性能会退化为常数级别。

**⚠️ 局限性**

结果仅适用于特定的概念类构造，未给出普适的直接和定理；在更一般的分布自由设定下，如何精确描述直接和学习曲线仍是未解决的问题。

---

## 137. Evolving Parallel Algorithm Portfolios via Potential-Aware Instance Generation with LLMs

**arXiv ID:** 2608.06808 | [PDF](https://arxiv.org/pdf/2608.06808v1)

**作者:** Shaofeng Zhang `[一作]` (Southern University of Science and Technology), Ke Tang `[通讯]` (Southern University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于大型语言模型的潜在感知实例与算法共进化框架PIAC，用于自动构建并优化并行算法组合（PAP），以提升组合在多种分布下的泛化性能。

**💡 创新点**

核心创新：① 设计潜在收益（potential gain）度量，利用对算法的轻微扰动直接评估实例的改进潜力，从而不依赖高质量参考解；② 通过LLM生成多样化实例变异器，扩展实例空间，提升组合的多样性与鲁棒性；③ 将这两项技术嵌入共进化流程，实现实例与算法的双向迭代改进。

**🔧 技术方法**

技术手段包括：大型语言模型（DeepSeek‑V3.2、DeepSeek‑V4、Kimi‑K2.6、GPT‑4.1 mini）用于生成/修改启发式代码；对启发式矩阵的随机扰动评估潜在收益；进化算法（父子选择、互补交叉、变异、贪心替换）用于构建PAP；多种背骨（Greedy Constructive、Ant Colony Optimization、Guided Local Search）对应的扰动与实例生成策略。

**📊 数据集**

实验数据集：TSP 与 CVRP 的合成实例，训练集为随机分布（RUE）8个样本；六种不同分布（RUE、explosion、implosion、cluster、expansion、grid）用于测试；公开库 TSPLib 与 CVRPLib 进一步验证。

**📈 对比分析**

与 FunSearch、EoH、ReEvo、MCTS‑AHD、EoH‑S 等 LLM‑ACP 基线以及 PIAC 的两种内部变体（RND 与 GAP）进行对比。实验表明：在所有背骨上，PIAC 的平均最优性差距明显降低，TSP Greedy Constructive 上比 EoH‑S 降低 19.76%（相对），CVRP 上提升 11.21%；在 TSPLib / CVRPLib 上同样取得最小误差，显示强泛化能力。

**⚠️ 局限性**

局限性：目前扰动机制仅适用于产生矩阵形式启发式的算法；对更复杂或非结构化决策表示的算法适配尚未完成；潜在收益虽无需参考解但仍需多次算法评估，计算成本高；LLM 的生成质量与模型能力密切相关，低端模型性能相对受限。

---

## 138. Multi Codec Discrete Diffusion Model for Text Guided Speech Inpainting and Editing

**arXiv ID:** 2608.06424 | [PDF](https://arxiv.org/pdf/2608.06424v1)

**作者:** Iftach Shoham `[一作]` (Ben Gurion University Of Negev), Eliya Nachmani `[通讯]` (Ben Gurion University Of Negev)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于离散扩散的语音文本引导的填补与编辑框架 SIEDD，能够在不重合成全句的情况下修复或替换语音中的缺失或改写段落。

**💡 创新点**

通过 HiCoDD 采用分层残差向量量化的码本顺序生成，保持粗到细的上下文一致；结合音素级条件、持续时间预测与局部无分类器扩散指导，显著提升语音连贯性与内容准确性。

**🔧 技术方法**

离散扩散模型（吸收状态 DDM）、残差向量量化（EnCodec）码本分层、Diffusion Transformer 结构、跨层注意力、音素条件编码、局部 CFG 以及持续时间预测网络。

**📊 数据集**

在 RealEdit 语音编辑基准（LibriTTS、GigaSpeech 以及 YouTube 录音）上训练与评估，使用 EnCodec 16kHz 编码器。

**📈 对比分析**

与 VoiceCraft、SSR‑Speech 以及 TTS baseline MMS 进行对比，SIEDD 在 RealEdit 上实现最低 WER、最高说话人相似度和最低 MCD，在多种掩码长度与多段编辑场景下均优于自回归基线，显示出更稳定的性能。

**⚠️ 局限性**

采样速度仍慢，需512步扩散，持续时间预测及多语言支持尚未完善，且对极长或复杂编辑场景的泛化仍有限。

---

## 139. MMAG: A Multi-Control Mixed Audio Generation Benchmark

**arXiv ID:** 2608.06900 | [PDF](https://arxiv.org/pdf/2608.06900v1)

**作者:** Zihao Zheng `[一作]` (Shanghai Jiao Tong University), Mengyue Wu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MMAG基准，构建约4千条手工校验的混合音频样本并为其生成语义、说话人、音乐属性、声效与时间关系等细粒度注解，同时设计了语音克隆和时间戳控制子集，用于全维度评估混合音频生成模型。

**💡 创新点**

创新点在于统一把语义、说话人、音乐、声效和时间顺序等多维属性进行细粒度标注，并通过子集实现对语音克隆和精准时间控制的评估，从而填补了现有基准缺乏混合场景与多控件评估的空白。

**🔧 技术方法**

采用专家模型（语音识别、说话人识别、音乐标签）与LLM协同生成注解，人工复核；评估方面使用FAD、FD、KL、IS、PQ、SPK-SIM、UTMOS、CLAP、AnyAudio-Judge、Speech_F1等多维度自动指标。

**📊 数据集**

数据来源于AudioCaps、VGGSound、MECAT三大公开混合音频集合，经过过滤、单声道英语、录音质量检查后构成主集、克隆集与时间戳集。

**📈 对比分析**

统一在声学、语音质量、语义一致性和时间控制四个维度上对AuDirector、LTX-2、MOVA、Dasheng-AudioGen、Ming-Omni-TTS等模型进行对照评测，结果表明无单一模型在所有指标上占优，模型间存在明显的性能权衡与制约。

**⚠️ 局限性**

局限性包括片段仅10秒、单说话人英语、类别分布不均、注解过程仍存在主观性、LLM在细粒度评估与时间标注上的可靠性不足，以及对非英语、多说话人场景的可推广性有限。

---

## 140. Measuring the Cross-Lingual Comprehension Gap: How the language of the evidence shapes what language models understand

**arXiv ID:** 2608.06506 | [PDF](https://arxiv.org/pdf/2608.06506v1)

**作者:** Rafael da Silva `[一作]` (Eastern University), Jeff Eicher `[通讯]` (Eastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在同一段落的多语言版本中保持问题、答案、模型和评分标准不变，只改变证据文本的语言，构建了一个对照实验来衡量语言模型在跨语言理解上的差距，称为跨语言理解差距（CLCG）。

**💡 创新点**

提出了CLCG这一新的衡量指标，并且在完全受控的“并行内项”设计下首次量化了语言资源稀缺语言对模型表现的系统性影响；同时发现CLCG与语言资源级别呈负相关，并揭示了跨语言差距在不同深度问题上呈非单调分布。

**🔧 技术方法**

使用了Token‑F1、Exact Match、BERTScore 等自动评估指标；为验证自动评估，进行了大规模的人工评价（约1700份评注）并与自动评分进行对照；还采用了混合效应模型、bootstrap 以及等效性检验等统计方法来检验结果的稳健性。

**📊 数据集**

基于《Watchtower Online Library》（WOL）的人工翻译并行语料库 ParallelQA‑18，该语料覆盖 18 种语言（含 1 个资源极低的 Fon、1 个半孤立语言 Basque 等），共 559 篇英文原文及 9,843 语言版本；此外，还在 FLORES‑200 语料上做了跨域验证。

**📈 对比分析**

实验对比了英语（参考）与 16 种目标语言的 Token‑F1 分数，得到平均 CLCG 为 0.078（即约 17% 的性能下降）。不同模型（GPT‑5.6 Terra、Gemini‑3.6 Flash、Kimi‑K3、DeepSeek‑V4‑Pro、Claude‑Sonnet‑5）均出现正向 CLCG，范围 0.062–0.101；在人类评价中，高资源语言的答案更受偏好，支持自动评估的趋势。

**⚠️ 局限性**

主要局限包括：CLCG 受限于所选的语言和语料（未覆盖极高资源语言 Class 5、Ayacucho Quechua 未分类）；语言间翻译质量虽做了人工校验，但仍可能引入语义偏差；使用的深度问题分级为实验设计的近似，未必完全对应真实认知难度；人工评价样本量有限，且跨语言一致性评测仅在英语完成，无法直接评估目标语言的生成质量。

---

## 141. Improving Low-Resolution Face Recognition under Limited Data: How Synthetic Data Generation Can Close the Domain Gap

**arXiv ID:** 2608.06580 | [PDF](https://arxiv.org/pdf/2608.06580v1)

**作者:** Luis S. Luevano `[一作]` (Idiap Research Institute), Sébastien Marcel `[通讯]` (Idiap Research Institute)

**通讯引用:** 15082 | [OpenAlex ID](https://openalex.org/A5016330764)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在缺乏本地低分辨率（LR）训练样本的条件下，如何通过合成LR数据提升边缘设备上面向低分辨率人脸识别的性能。

**💡 创新点**

提出了系统的合成与适配策略评估框架，揭示合成难度与真实LR之间的域差距，发现最佳合成设置在合成基准上与真实数据相悖，并证明学习式超分辨前端并不一定优于直接投射到强大冻结模型，且LR合成并未显著改善公平性。

**🔧 技术方法**

使用了多种技术：插值降采样、Real-ESRGAN式随机降噪降采样、知识蒸馏、Prepended Domain Transformer (PDT)、身份感知超分辨前端（ESPCN、RRDB）与对比学习。

**📊 数据集**

主要数据集包括WebFace4M（HR）、LFW、CFP-FP、AgeDB-30（合成LR评估）以及真实低分辨率的TinyFace（识别）和RFW（公平性评估）。

**📈 对比分析**

通过在合成LR基准（LFW/CFP-FP/AgeDB-30）和真实LR TinyFace上对比，发现：在合成基准上插值到28px和KD方法能提升性能，但在TinyFace上56px插值最优；学习式SR+PDT在冻结大模型上不超越直接投射；在公平性上LR合成对FMR不产生系统性改进。

**⚠️ 局限性**

局限性在于仅使用合成LR训练，未评估真实LR微调的效果；学习式SR前端仅在大冻结模型上验证，未测试对小模型的潜在收益；合成策略对不同部署场景的适配仍需进一步实验。

---

## 142. Coupling Planning with Episodic Memory in LLM Agents for Software Issue Resolution

**arXiv ID:** 2608.06811 | [PDF](https://arxiv.org/pdf/2608.06811v1)

**作者:** Jiahao Zhang `[一作]` (Vanderbilt University), Yu Huang `[通讯]` (Vanderbilt University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了PMCoder，一个将分层阶段规划与情节记忆双向耦合的LLM代理，用于长时间的软件问题修复。

**💡 创新点**

双向耦合规划与记忆：计划状态指导记忆检索，记忆统计驱动挫败检测与重规划，并通过执行验证根据信息更新计划状态。

**🔧 技术方法**

分层阶段规划器、情节记忆（MMR检索、代码结构图）、工具调用交互、执行基础验证（issue reproduction scripts）、回滚-重构恢复等技术。

**📊 数据集**

SWE-bench Verified-500 真实 GitHub issue 集；另外使用 DeepSeek、Claude、OpenHands、TerminalWorld 进行跨模型/框架/基准的通用性验证。

**📈 对比分析**

与基线（无规划/记忆）在三次独立运行下做配对对比，PMCoder平均提升 25.0 例（+5.0pp，p<0.001）；在不同模型/框架下亦显著提升，记忆+规划交互效果显著优于单独组件。

**⚠️ 局限性**

依赖已提取的 reproduction scripts，无法覆盖无脚本场景；仅在公开基准和公开模型上评估，语言/私有仓库迁移未知；计划/记忆阈值固定，未针对不同难度动态调优。

---

## 143. Deep Evidential Regression for Sparse Forest Height Estimation from Multimodal Satellite Imagery

**arXiv ID:** 2608.06406 | [PDF](https://arxiv.org/pdf/2608.06406v1)

**作者:** Laura Bader `[一作]` (LMU Munich), Göran Kauermann `[通讯]` (LMU Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究将深度证据回归（DER）应用于多模态卫星影像（Sentinel-1/2）来估计森林平均树高，并通过掩蔽证据损失解决极度稀疏的监督问题。

**💡 创新点**

创新点包括：①在森林高度估计任务中首次将DER与掩蔽损失结合以适应稀疏标签；②对预测不确定性进行校准评估并分析其与森林结构异质性的空间关联；③提供了一套完整的实验与可复现的代码。

**🔧 技术方法**

技术手段包括：U‑Net 编码器为 ResNet‑50，输出 NIG 参数的证据头；使用 log(1+y) 变换、混合精度训练、AdamW 与余弦学习率调度；以及通过随机翻转、旋转进行数据增强。

**📊 数据集**

数据集为 TreeUQ 基准，覆盖德国巴伐利亚州，包含 10 m 解析度的 Sentinel‑1/Sentinel‑2 合成影像和稀疏的树木清单树高与方差统计。

**📈 对比分析**

与传统确定性 U‑Net 进行对比，DER 在 RMSE（5.91 ± 0.15 m vs 5.75 ± 0.09 m）和 R²（0.497 ± 0.026 vs 0.525 ± 0.015）上相差不大，却能在单前向传播中给出可靠的不确定性估计，校准误差 ECE 仅 0.025，校准曲线近似对角线。

**⚠️ 局限性**

局限性包括：①证据框架对不确定性分解的解释仍具启发性；②假设高斯似然可能无法完全捕获树高分布；③稀疏监督导致模型未在非森林区域训练为零高度，可能产生误预测；④仅评估了单一森林属性，未与其他不确定性方法进行更系统比较。

---

## 144. MI-MIDI: Mechanistic Interpretability of Text-to-MIDI Generation Models via Probing, Lenses and Steering

**arXiv ID:** 2608.06638 | [PDF](https://arxiv.org/pdf/2608.06638v1)

**作者:** Jakub Poćwiardowski `[一作]` (Warsaw University of Technology), Mateusz Modrzejewski `[通讯]` (Warsaw University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究对两种文本到MIDI的生成模型（text2midi和MIDI-LLM）进行了机制性可解释性分析，包括线性探测、Logit Lens、调优 Lens、激活修补与激活导向等多种技术，以揭示其内部音乐结构、预测形成时机以及可控性。

**💡 创新点**

创新点在于首次对符号音乐生成模型开展全面的机制性可解释性研究，比较了两种截然不同的架构（编码-解码vs. 单向语言模型）对音乐信息的编码与生成过程，并提出了双向评估协议来区分方向性控制与对称漂移。

**🔧 技术方法**

使用了线性探测器（多项式LogReg）、经典与调优 Logit Lens、激活修补、差分均值方向（difference‑in‑means）和双向激活导向技术，以及范数相对尺度的调试策略。

**📊 数据集**

数据集包括公开的MIDICaps文本–MIDI配对数据（用于生成和标签），SynTheory合成音乐理论数据（用于对单一音乐概念的精细探测），以及从随机文本提示生成的1,000段MIDI序列。

**📈 对比分析**

通过对每层的探测准确度提升、Logit Lens的一致性、激活修补的传递程度以及激活导向的方向性与对称性分解进行比较，表明text2midi在深层逐步细化输出，MIDI‑LLM则表现出在层13-14的“晚期绑定”转变；在可控性方面，单层注入在MIDI‑LLM中保持低对称漂移，所有层注入在text2midi中效果更稳定。

**⚠️ 局限性**

局限性包括：对比文本提示在词汇与音乐提示上的混合导致差分均值方向不够纯粹；使用的指标多为近似代理，可能与实际音乐概念不完全一致；标签噪声和仅使用两种模型限制了泛化；复杂音乐概念的可度量性不足，未能纳入定量评估。

---

## 145. A Kubernetes Scheduler Plugin for Cluster-Wide Placement Optimisation

**arXiv ID:** 2608.06987 | [PDF](https://arxiv.org/pdf/2608.06987v1)

**作者:** Henrik Daniel Christensen `[一作]` (TV 2 Danmark A/S), Jacopo Mauro `[通讯]` (University of Southern Denmark)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了一个Kubernetes调度插件，使外部求解器能够生成全局调度计划，并通过调度框架钩子原子性地执行计划。

**💡 创新点**

创新点在于在不改动默认调度器的前提下，利用Scheduling Framework提供的五个钩子实现跨节点预裁与同步，支持三种触发模式与阻塞/非阻塞执行，兼容外部任何优化求解器。

**🔧 技术方法**

采用了外部求解器（以Google OR-Tools CP‑SAT为例）与Kubernetes Scheduling Framework钩子，构建了快照、计划验证、计划执行的三阶段流程；并使用了异步/同步模式控制调度器并发。

**📊 数据集**

使用了基于Alibaba GPU和Google Cluster 2019的公开工作负载统计参数生成的合成trace，并在KWOK模拟器上进行评估；实验规模为16/32节点、1/4优先级、3类到达速率。

**📈 对比分析**

与默认调度器对比，periodic和stable‑queue模式在大多数场景下提升资源利用率1.2–3.0%，并降低排程延迟；调度失败模式效果更弱。实验展示了计划激活率、求解器调用次数和pod删除量等指标。

**⚠️ 局限性**

主要限制包括求解器运行时间高，难以在大规模或高到达率环境下实时使用；计划基于快照，易被状态漂移失效；实验仅基于合成trace，未在真实生产集群中验证；缺乏机器学习预测等动态信息。

---

## 146. Walkable to Whom? Capturing Subjective Variability in Walkability Perception Using Multimodal Deep Learning

**arXiv ID:** 2608.06934 | [PDF](https://arxiv.org/pdf/2608.06934v1)

**作者:** Moloud Damandeh `[一作]` (University of New South Wales), Meead Saberi `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了一个包含9个澳大利亚城郊地区、974幅人行道视角图像的可步行性感知数据集，并提出了基于用户属性的多模态深度学习框架，用以预测个体对行人环境的步行性评分。

**💡 创新点**

创新点在于首次将步行性感知视为用户条件化预测任务，将图像特征与评测者属性相结合，并通过视角比较证明了人行道视角对评分的显著影响，推动了更具包容性的步行性评估方法。

**🔧 技术方法**

采用Swin‑Tiny视觉Transformer提取图像特征，FT‑Transformer对结构化评测者属性进行编码，随后通过Transformer融合模块进行多模态组合，并使用CORAL序数回归损失实现最终预测。

**📊 数据集**

使用29,870条来自1,196名受访者的五分制步行性评分数据，涉及974幅人行道视角图像；同时匹配100个地点的Google街景图像用于视角对比。

**📈 对比分析**

通过与仅图像输入的基线模型在测试集上对比，利用二次加权Kappa、MAE、准确率和within-one准确率评估。用户条件化模型将Kappa从0.285提升至0.469（+65%），MAE从0.885降至0.785，并在极端评分上显著提升召回率。

**⚠️ 局限性**

局限性包括样本主要来自澳大利亚，缺乏跨文化与跨基础设施环境的代表性；视角对比研究规模有限；模型对完全个体化偏好捕捉能力有限，迁移性尚待验证。

---

## 147. Long-Horizon Agent Trajectory Attribution: A Unified Benchmark and Fine-Grained Annotation Framework

**arXiv ID:** 2608.06909 | [PDF](https://arxiv.org/pdf/2608.06909v1)

**作者:** Jing Chen `[一作]` (Huawei Technologies Ltd.), Jie Shi `[通讯]` (Huawei Technologies Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个用于评估大型语言模型代理轨迹归因的基准和注释框架，统一了异构轨迹的组件级别结构，定义了主归因与链归因两种标注形式，并提供可复用的注释技能；

**💡 创新点**

创新点在于构建了可复用的轨迹归因框架，首次在多种代理场景下生成1300+带主归因和攻击/执行链标注的轨迹，并通过统一的评估任务（主归因定位与链归因恢复）实现跨任务、跨模型的系统比较；

**🔧 技术方法**

采用LLM辅助的注释与验证流程，利用两种基线归因方法（增量归因与留一归因）以及统一的轨迹标准化与评估指标（Hit@1、MRR、Recall@K、MAP）进行实验；

**📊 数据集**

使用了AgentDojo、Agent3Sigma Stage 和 Agent3Sigma Canary 生成的轨迹数据，覆盖任务完成、违规动作和安全拒绝三类行为，构成1,351条带有主归因与链归因标注的轨迹；

**📈 对比分析**

通过微平均计算 Hit@1、MRR、Recall@K 与 MAP，对比两种基线在不同目标类型、来源与归因距离上的表现；结果显示主归因 Hit@1 在 0.37–0.54 之间波动，链归因 Recall@3 在 0.00–0.66 之间，表明任务具有显著挑战且各设置差异明显；

**⚠️ 局限性**

局限性包括仅覆盖三类基准，注释过程依赖 LLM 可能存在多种合理解释，未提供更细粒度（句子/词级）标注，基线仅为简单方法，缺乏更先进归因算法的评估与对比。

---

## 148. FedLBW: A Loss-Based Weighting Strategy for Federated Learning on Non-IID Data in Wireless Networks

**arXiv ID:** 2608.07007 | [PDF](https://arxiv.org/pdf/2608.07007v1)

**作者:** Majid Kundroo `[一作]` (Chungbuk National University), Taehong Kim `[通讯]` (Chungbuk National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedLBW——一种利用服务器端小型代理数据计算每个客户端验证损失的逆值作为聚合权重的新型联邦学习聚合方法；

**💡 创新点**

创新点在于用本地模型验证损失的反比来权衡聚合，而非传统的样本数权重，能自适应处理非 IID、异常值及客户端掉线，并无额外学习开销；

**🔧 技术方法**

使用 FedAvg 及其改进版（FedAvgM、FedProx、FedNova 等）做对照，采用 Dirichlet 划分、SGD、CNN/ResNet 等模型，理论上给出收敛分析；

**📊 数据集**

实验采用 FashionMNIST、CIFAR‑10 与 CIFAR‑100 三个公开图像分类数据集；

**📈 对比分析**

与 FedAvg、FedAvgM、FedProx、FedNova、FedLAW、FedDkw 等算法在不同非 IID程度（α=0.1、0.3、0.6）和客户端掉线率（0.0‑0.5）下对比，FedLBW 在极端非 IID（α=0.1）下可提升约7.6%准确率、收敛更快，在高掉线率时仍保持约90%原始性能；

**⚠️ 局限性**

需要服务器端持有代理数据，且计算逆损失权重增加服务器负载；若代理数据与客户端分布相距较大，权重可能失真；适用于能获取验证损失的任务，受限于代理数据可用性。

---

## 149. A Transferable Autologistic Model for Predicting Rare Failures in Heterogeneous Equipment

**arXiv ID:** 2608.06695 | [PDF](https://arxiv.org/pdf/2608.06695v1)

**作者:** Islam Benamirouche `[一作]` (Université de Sherbrooke), Feriel Fass `[通讯]` (Université de Sherbrooke)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种适用于同一设备族异构传感器环境下的罕见故障预测方法，能够在训练设备上学习共性模式后，再通过少量目标设备数据进行自适应改进。

**💡 创新点**

创新点包括：①将异构传感器数据映射到统一传感器空间并使用可用性掩码；②采用监督式自编码器提取低维隐空间，既保留时序信息又兼顾判别性；③在逻辑回归基础上引入自回归项实现故障概率的时序平滑；④通过先验正则化实现从共性模型向目标设备的温和迁移；⑤在合成数据集上展示了显著提升的检测率与误报率。

**🔧 技术方法**

使用了监督自编码器（encoder-decoder+logistic head）、自回归逻辑回归（autologistic regression）、类别加权与高斯先验的MAP估计、L‑BFGS‑B优化、以及滑动平均后处理。

**📊 数据集**

合成冷藏机数据集：27台模拟冷柜（17台训练，10台目标），每台设备配备15–20个多变量传感器，采样频率为1分钟，包含六种失效模式（压缩机、机械磨损、冷凝器污垢、门密封、风扇、除霜加热）。

**📈 对比分析**

对比方法：共性模型直接预测 vs 目标特定适配后预测。性能提升：检测率从61.0%提升到91.5%，误报数从24降至13，平均提前警报时间从132.8h降至101.2h；AUC‑ROC 在各目标设备上从0.526–0.818提升到0.923–1.000。

**⚠️ 局限性**

局限性：仍有5次漏报（约8.5%）和13次误报；使用的是合成数据，未验证在真实工业数据中的效果；对非平稳或极端操作条件的鲁棒性未充分探究；需要进一步优化报警生成规则以进一步降低误报。

---

## 150. CoDAT: Collaborative Dual-Attention Transformer with Low-Cost Temporal Modeling for Efficient Edge Action Recognition

**arXiv ID:** 2608.06691 | [PDF](https://arxiv.org/pdf/2608.06691v1)

**作者:** Novendra Setyawan `[一作]` (National Formosa University), Jun-Wei Hsieh `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种适用于边缘 IoT 设备的低成本时空注意力 Transformer——CoDAT，用于实时人类动作识别。

**💡 创新点**

设计了协同双注意力模块（CoDA），将单头跨步注意力与局部卷积注意力并行融合，并引入无参数时间位移 TShift，实现轻量化全局与局部时空建模。

**🔧 技术方法**

采用单头跨步注意力、空间卷积注意力、可学习交叉投影、零参数时间位移、层级金字塔结构与轻量化 FFN 等技术。

**📊 数据集**

在 ImageNet‑1K、Kinetics‑400、MA‑52 与 UCF‑101 等公开数据集上进行图像与视频动作识别实验。

**📈 对比分析**

与现有 CNN、Transformer 与轻量化 ViT 基线在 Jetson AGX Orin 与 Raspberry Pi 5 上对比，CoDAT 在保持或略优于 SOTA 的准确率同时，速度提升 1.5‑3 倍，能耗降低 70‑90% 以上。

**⚠️ 局限性**

全局–局部融合与通道压缩方案仍为手工设计，缺乏自适应动态压缩；时序建模仅限单尺度短时，难以捕获长时或多尺度动作。

---

## 151. Online Security Learning in Cooperative Multi-Agent Systems under Hidden Byzantine Attacks

**arXiv ID:** 2608.06520 | [PDF](https://arxiv.org/pdf/2608.06520v1)

**作者:** Ximing Sun `[一作]` (University of Central Florida), Yue Wang `[通讯]` (University of Central Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出并研究了在 Byzantine 攻击下的多智能体系统的在线合作控制问题，即团队在计划联合行动后，部分未知代理可以偷偷覆盖自己的行动坐标。

**💡 创新点**

创新点包括：①将攻击信息与决策几何联系，揭示内幕观察者导致(s,a)-矩形鲁棒MDP、盲观察者导致s-矩形模型；②证明安全回报退化为实际回报与响应差距之和，并给出信息论极限；③设计了阶段绑定的鲁棒估计-决策(E2D)学习算法，显著降低估计预算并得到𝒪(H²S√(AK))的安全回报损失上界；④明确了公共反馈无法完全识别最坏响应，需接受响应差距D_K。

**🔧 技术方法**

使用的技术包括：鲁棒MDP矩形性分析、信息论极限证明、E2D框架与阶段绑定估计器、Hellinger距离预测损失、贝尔曼不等式优化、随机马尔可夫策略、近似决策oracle等。

**📊 数据集**

论文未使用任何公开数据集，全部以理论分析与仿真实例验证。

**📈 对比分析**

与现有最优鲁棒MDP学习算法（如Appel‑Kosoy）相比，本文在状态维度上提升了√S，且在最坏响应已知的情形下提供安全回报的上界。对比表格显示在D_K=0时，安全回报复杂度为𝒪(H²S√(AK))，优于先前的𝒪(H²S^{3/2}√(AK))。

**⚠️ 局限性**

局限性包括：①无法消除响应差距D_K，仍需对最坏响应做假设；②对决策oracle的近似要求未给出多项式时间实现；③仅考虑集中式策略，分布式或联邦学习场景仍待研究；④对动态或可变 Byzantine 集合的情况未覆盖。

---

## 152. PHASE-Tree: Modeling Character-State Evolution in Long-Horizon Role-Playing Dialogue

**arXiv ID:** 2608.06975 | [PDF](https://arxiv.org/pdf/2608.06975v1)

**作者:** Bo Tang `[一作]` (MemTensor Technology), Jiajun Shen `[通讯]` (MemTensor Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对长时序角色扮演中的“stale‑state failure”，提出可编辑心理学驱动的分层树结构PHASE‑Tree，并构建统一的LongEvoRoleBench评测集。

**💡 创新点**

创新点在于：① 将角色状态拆分为不可变身份根 + 可编辑的persona、session、moment层，使用抗性–证据–冷却门控实现跨剧集演化；② 同时提供文本提示与参数适配两种条件化范式。

**🔧 技术方法**

技术包括：大型语言模型（GPT‑4.1、Qwen2.5‑7B‑Instruct 等）用于树抽取与更新；结构化树序列化至提示或Profile‑to‑LoRA hypernetwork 生成/适配；多模态评价与自定义门控策略。

**📊 数据集**

使用八个现有对话语料（Friends、The Office、Star Trek、Harry Potter、RAIDEN、CharacterEval、SimsConv、ChatHaruhi），划分短/长对话两种评测模式。

**📈 对比分析**

与多种基线（RAG、PAG、CFG、MT‑LoRA、Steering、OPPU、P2P 等）对比；在文本提示下取得21/24指标最高，长对话上12/12指标领跑；在参数适配下排名前二，整体提升字符一致性12%–19%，语义连贯性和嵌入相似度显著优于传统方法。

**⚠️ 局限性**

局限主要是 Profile‑to‑LoRA 压缩导致细粒度状态信息丢失，门控阈值手工设定缺乏自动学习，且跨域泛化和更大规模角色库的适应性尚待验证。

---

## 153. Explicit, Not Longer: What Makes Epistemic Stance Survive Memory Compression

**arXiv ID:** 2608.06953 | [PDF](https://arxiv.org/pdf/2608.06953v1)

**作者:** Alex Kwon `[一作]` `[通讯]` (Independent Researcher), Alex Kwon (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了在内存压缩系统中，使用“字段”写法与“括号注释”写法写入声明的可信度（standing）时对信息保留的影响，发现字段写法能显著提升可信度的存留率；

**💡 创新点**

创新点在于提出并验证一种简单的写入模式（CLAIM/SOURCE/CERTAINTY/AS_OF字段表述）可显著改善可信度的存留；通过细粒度消融分析揭示不同模型对写法特征（标签、括号、词语、长度）的敏感度不同；

**🔧 技术方法**

技术手段包括：使用两款大语言模型（Haiku 4.5 与 Sonnet 4.5）做压缩器；采用盲读者模型对压缩结果进行无偏判定；对写法属性（标签、行距、词化、长度）进行逐步消融；统计方法包括配对符号检验、信赖区间、Bootstrap 置信区间；

**📊 数据集**

数据集为 60 条手工编写的英文声明，覆盖七种注册表（如项目、冲突等），每条声明以两种写法（字段/括号）复制，并配有填充注释；另外使用 10 条早期样本、50 条人工标注评测数据。

**📈 对比分析**

对比方法：同一声明在两种写法下送入相同压缩器，盲读者判定存留的可信度，结果显示字段写法在两种模型上均提高约 15%（Haiku +15.8，Sonnet +15.3），统计显著（p<0.0001）。消融实验表明标签效应在两模型均显著，长度无效，词化效应模型依赖。

**⚠️ 局限性**

限制包括：仅测试两款模型，未覆盖更广泛压缩器；样本为人工合成，未检验真实笔记；仅在极紧预算下显著，宽松预算下效应减弱；消融路径非完全交叉实验，导致模型特异性；人工标注由作者完成，缺乏独立评测。

---

## 154. Solver-Guided Reasoning for Mixed-Equilibrium Strategies

**arXiv ID:** 2608.06741 | [PDF](https://arxiv.org/pdf/2608.06741v1)

**作者:** Han Wang `[一作]` (Shanghai Jiao Tong University), Baoxiang Wang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作通过将商业解算器输出的混合策略转换为稀疏决策树（MDT）和对比式抽样（SCCS）规则，构建了一套可供大型语言模型（LLM）推理的可读规则集，进而提升LLM在无限注德州扑克和两人Liar's Dice等不完全信息游戏中的平衡策略推断能力。

**💡 创新点**

创新点在于（1）提出MDT，将高维混合策略压缩为可解释的稀疏树结构，既保留关键决策边界又便于人类与LLM理解；（2）提出SCCS，利用公共情境下的策略分歧对比，提炼出局部对比规则，让LLM能在未知手牌上迁移混合策略；（3）展示将solver产生的人工稀疏规则直接注入LLM提示，显著弥合人类数据与最优策略之间的鸿沟。

**🔧 技术方法**

使用了：商业No‑Limit Texas Hold’em求解器生成的250+M决策点；稀疏决策树（hard top‑K路由）进行策略蒸馏；Scenario‑Constrained Counterfactual Sampling抽样对比；LLM提示工程（Direct、Direct+Summaries、Route‑only、SCCS Rule四种条件）；评估指标包括L1距离、argmax一致率以及River端游戏的精确可利用度。

**📊 数据集**

数据集包括：1）约250M条NLH求解器标注的后注决策（16M flop + 235M turn）覆盖1,755 flop纹理及多种公共情境；2）使用相同方法生成的Liar's Dice全局求解器数据（6面骰子，最高三张牌）用于跨游戏验证。

**📈 对比分析**

比较方法是对同一套未见手牌在同公共上下文下，以四种提示条件分别评估LLM输出，计算与solver目标的L1距离与与MDT蒸馏模型的距离；实验结果显示：在8种LLM配置下，SCCS规则将L1从0.211下降至0.100（52.6%改进），argmax准确率从57.2%提升至76.1%；River端游戏的exploitability差距约0.2%；在Liar's Dice中，SCCS使L1下降至0.105（相对Direct下降39.7%）。

**⚠️ 局限性**

局限性包括：依赖高质量求解器，规则抽取与抽象仅适用于具有公共/私有分离结构的离散游戏；对连续或更大规模游戏的推广尚未验证；LLM仍未成为完整扑克代理，规则的解释性与可迁移性受限；以及对solver输出的误差或不完备性可能影响规则质量。

---

## 155. AtlasVLA: Persistent World-Ego State Modeling for Vision-Language-Action Models

**arXiv ID:** 2608.06729 | [PDF](https://arxiv.org/pdf/2608.06729v1)

**作者:** Guiyu Zhao `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jing Liu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 AtlasVLA，一种基于持久世界-自我状态记忆的 Vision‑Language‑Action 框架，使得仅使用腕部摄像头即可完成长期、部分可观测的机器人操作。

**💡 创新点**

创新点在于：① 4D Persistent World State Memory（将实时 2D 观测映射到 voxel‑hashed 3D 空间，实现全局场景记忆），② Ego‑Working State Memory（利用意图感知查询和冗余融合，保持任务进度与自我状态），③ 将两者作为条件输入给 diffusion transformer，形成世界‑自我引导的动作生成。

**🔧 技术方法**

技术包括：深度估计（Depth Anything v3）、空间反投影、双重时空位置编码、TSDF‑风格 voxel 聚合、滑动窗口记忆更新、意图感知查询、跨注意力检索、step‑wise conditioning 的 DiT 以及 DDIM 与 CFG 的动作解码。

**📊 数据集**

使用的数据集包括：LIBERO（五个子套件，含 Long 与 90）、RLBench（20 任务）、以及真实 Franka 机器人平台（6 一般操作 + 4 长周期任务）。

**📈 对比分析**

与现有基线（OpenVLA、π₀、CogACT、MemoryVLA 等）进行对比；在仅腕部摄像头条件下，AtlasVLA 在 LIBERO-Long 上提高 9.4% 成功率，在真实长周期任务上提高 17.5%，并在 RLBench 上以 70.8% 的平均成功率领跑同类方法。

**⚠️ 局限性**

局限性包括：① 依赖精确手眼标定与深度估计的质量；② voxel‑hashed 记忆在极大空间中仍有存储与计算瓶颈；③ 对动态环境（快速移动对象）与光照变化的鲁棒性尚待进一步验证。

---

## 156. IB-RL: Isolated Bilateral Reinforcement Learning for Strategic Dialogue Agents

**arXiv ID:** 2608.06735 | [PDF](https://arxiv.org/pdf/2608.06735v1)

**作者:** Senhao Wang `[一作]` (Dcar Inc. ByteDance), Zecheng Lin `[通讯]` (Dcar Inc. ByteDance)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Isolated Bilateral Reinforcement Learning（IB‑RL），通过在两方对话中完全隔离奖励、优势、掩码和优化器实现共进化训练，解决传统单向RL导致的静态对手不匹配问题。

**💡 创新点**

创新点在于将两方代理的学习信号完全分离，并在共进化过程中采用对手池采样和延迟更新机制，既保持了训练的稳定性，又提升了对未见对手的泛化能力。

**🔧 技术方法**

使用GRPO作为无价值函数的策略梯度算法，并在此框架下实现了角色独立的优势归一化、令牌级掩码以及独立的优化器更新。

**📊 数据集**

实验基于两大数据集：Vehicle TeleSales（约50K真实中文外呼记录和3K模拟客户资料）和Deal‑or‑No‑Deal（公开的谈判对局数据）。

**📈 对比分析**

在两任务上与单向RL和最强基准模型对比，IB‑RL在Vehicle TeleSales中Success@1从84.6%提升至89.6%，在Deal‑or‑No‑Deal中对未见对手的协议率超过94%，均优于同规模对手和前沿模型。

**⚠️ 局限性**

局限性包括：评估仅覆盖有限的未见对手集合，实验仅针对两角色两任务，且RL奖励评估可能存在判别者偏差，未来需扩展更多角色、任务和人类验证。

---

## 157. GRASP: Reinforcing Language Model Anonymizers with Group Relative Policy Optimization

**arXiv ID:** 2608.06526 | [PDF](https://arxiv.org/pdf/2608.06526v1)

**作者:** Sajjad Ghiasvand `[一作]` (University of California Santa Barbara), Nader Sehatbakhsh `[通讯]` (University of California Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于单模型的本地隐私文本匿名化方法，利用GRPO强化在线自我评估，以实现隐私-可用性权衡。

**💡 创新点**

将离线偏好学习替换为自我奖励的强化学习，单模型同时担任匿名化、攻击者和评估者，并通过奖励设计防止奖励劫持。

**🔧 技术方法**

使用Llama‑3.1‑8B‑Instruct，先SFT微调后采用Group Relative Policy Optimization (GRPO)训练，奖励由隐私得分和可用性得分组成。

**📊 数据集**

在SynthPAI合成的个人资料与评论数据集上实验，该数据集标注了八个个人属性。

**📈 对比分析**

与规则方法、转写器、前沿模型驱动的对抗匿名化及SEAL基线比较；在主/硬数据集上，整体隐私‑可用性得分提升至约0.396，隐私率下降至0.195，可用性保持在0.982以上。

**⚠️ 局限性**

推理延迟较高（单次自我修正约13.8秒），依赖LLM评估器且仅覆盖固定属性集，且并不能保证绝对匿名。

---

## 158. Rethinking Unified Memory for NPU-PIM Systems: Dual-View Memory for Dynamic Inference of LLM

**arXiv ID:** 2608.06989 | [PDF](https://arxiv.org/pdf/2608.06989v1)

**作者:** Shixin Zhao `[一作]` (Institute of Computing Technology), Ying Wang `[通讯]` (Institute of Computing Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现一种双视图统一内存系统PFM，支持NPU‑PIM架构在LLM推理中动态切换执行设备，并保持高带宽访问。

**💡 创新点**

创新点在于将物理内存布局与逻辑视图解耦，提供设备优化的双视图映射以及运行时地址重映射与调度机制，解决传统统一内存静态设备偏向导致的性能瓶颈。

**🔧 技术方法**

技术实现包括离线多目标混合整数规划求解双视图映射、ARU与FAS控制器实现动态地址翻译与访问调度，以及基于Ramulator与GPGPU‑Sim的系统级仿真。

**📊 数据集**

实验使用四大LLM（LLaMA3‑8B、DeepSeekMoE‑16B、Mixtral‑8×7B、GPT‑OSS‑120B）以及对应的MoE专家激活轨迹。

**📈 对比分析**

通过与NPU‑only、PSM、PUM三种基线在相同硬件平台对比，PFM在小批量场景下可达2.23×吞吐提升，批量较大时可达2.32×，并实现近92%理论带宽利用率。

**⚠️ 局限性**

限制在于需离线分析与有限的MDT条目，若模型或专家激活模式频繁变化需重新调优或扩展MDT，且实现依赖于支持2 MB超页的GPU。

---

## 159. Soft Redaction of Image Provenance via Zero-Knowledge Proofs

**arXiv ID:** 2608.07063 | [PDF](https://arxiv.org/pdf/2608.07063v1)

**作者:** Muhammad Awan `[一作]` (University of Surrey), John Collomosse `[通讯]` (University of Surrey)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了在C2PA图像原始信息中实现软红色化的机制，通过将敏感断言（如位置、面部特征、视觉指纹）替换为零知识证明（ZKP）来隐藏真实值，同时保持可验证性。

**💡 创新点**

创新点在于将ZKP与C2PA现有红色化机制结合，首次实现多种距离谓词（地理距离、L2相似度）下的软红色化，并利用Chebyshev多项式近似Haversine公式，在生物识别与防伪中实现隐私保护。

**🔧 技术方法**

使用技术包括基于PLONK的零知识证明系统、Chebyshev多项式逼近、L2距离证明电路、固定点算术以及C2PA manifest的硬红色化与软红色化流程。

**📊 数据集**

实验使用了LFW人脸识别数据集、MIRFLICKR‑25k图像检索数据集、随机生成的GPS坐标以及ArcFace、FaceNet、AdaFace、ElasticFace、ResNet‑18、DINO、SimProv、SSCD等特征提取模型。

**📈 对比分析**

方法上与Groth16、Bulletproofs三种ZKP系统对比，PLONK在证明时间0.4–6.7 s、验证时间≈250 ms、证明尺寸768 B的基础上可接受不同维度的嵌入；在位置证明中误差分别为36 m、214 m、1.1 km、7.6 km；在人脸相似度评估中准确率95–99%；在视觉指纹反欺骗实验中攻击拒绝率从20.9%提升至100%。

**⚠️ 局限性**

局限性包括仅实现距离谓词的软红色化，未覆盖集合成员、时间或复合策略；对低熵断言存在多次查询泄露风险；C2PA尚未标准化ZKP声明类型，需进一步完善安全模型与可信设置。

---

## 160. MolBioKG: Grounding Out-of-Graph Molecules in Biomedical Knowledge Graphs via Multi-Resolution Structural Anchoring

**arXiv ID:** 2608.06713 | [PDF](https://arxiv.org/pdf/2608.06713v1)

**作者:** Yiming Zhang `[一作]` (University of Tokyo), Keisuke Ozawa `[通讯]` (SB Intuitions)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套两层架构，先通过多分辨率分子索引（Bemis–Murcko骨架、BRICS碎片、功能基、ECFP4指纹）检索结构相似已收录的化合物，再通过桥接索引把这些“结构锚点”映射到生物医学知识图（RTX‑KG2c），并在此基础上执行推理。该系统针对未在图中出现的分子（out‑of‑graph molecule）实现了可追踪的目标预测与指示推断。

**💡 创新点**

创新点：
1. 针对冷启动化合物提出“结构多分辨率锚定 + 结构-图桥接”的双层架构，解决传统知识图只能处理已注册化合物的问题。
2. 设计两种推理机制：静态多锚点检索（RRF）与 LLM 驱动的自适应遍历（Adapt‑KG），二者互补，分别适用于结构检索和多跳推理。
3. 引入可追踪的结构与图证据路径，确保推断可解释且可审计。

**🔧 技术方法**

主要技术：
- 多分辨率分子检索（骨架、碎片、功能基、指纹）
- Reciprocal Rank Fusion (RRF) 进行多视角检索融合
- 预训练 LLM（如 GPT‑OSS‑120B）作为工具调度器，动态选择结构检索和图遍历动作
- 结构-图桥接索引（InChIKey、等价关系）
- Typed graph traversal 工具（9 种跳转操作）
- 源属性记录与路径追溯机制

**📊 数据集**

使用数据集：
- 分子层：RTX‑KG2c、PubChem、ChEMBL、ChEBI 共 2.74M 分子
- 结构索引覆盖 4 种视角（骨架、碎片、功能基、指纹）
- 生物医学图层：RTX‑KG2c，约 860k 节点、9.6M 边
- 评估集：
  * 199 2011 年后批准的新药（out‑of‑graph 评测）
  * MolBioKG‑KGQA：460 个 2–5 跳的自然语言多跳问题
  * 传统链路恢复（4 种药物指示/靶点/效应任务）

**📈 对比分析**

比较方法：
- 传统图模型：TxGNN、Graph‑RAG、Graph‑CoT、Think‑on‑Graph 等
- 结构检索 baseline：单视角检索、Label Propagation
- 结果：
  * 在多跳推理上，Adapt‑KG Hits@10 提升至 0.876，远超 Graph‑CoT（0.585）
  * 在 out‑of‑graph 目标召回上，RRF 从 0.145 提升至 0.269；对指示的召回从 0.186 提升至 0.239
  * 在链路恢复任务上，RRF 在 Hits@10 方面多项击败 TxGNN（例如指示 L1: 0.885 vs 0.843）

**⚠️ 局限性**

局限性：
- 当查询分子与已注册化合物缺乏骨架重叠时，检索效果明显下降。
- 评价指标（如命名一致性）较为严格，可能低估模型的实际生物医学发现潜力。
- 当前仅依赖基于结构的检索，缺少更灵活的嵌入或生成式分子表示，可能限制对高度异构化合物的覆盖。
- 需要进一步验证在更大规模或不同领域知识图中的可迁移性。

---

## 161. Surg-UniWorld: A Unified Surgical World Model with Multimodal Control Experts

**arXiv ID:** 2608.06770 | [PDF](https://arxiv.org/pdf/2608.06770v1)

**作者:** Rulin Zhou `[一作]` (Chinese University of Hong Kong), Hongliang Ren `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Surg-UniWorld，一种用于可控腹腔镜视频生成的统一手术世界模型。

**💡 创新点**

创新点在于将首帧外观与层次语义掩码构建为“层次手术锚”，并通过锚相关模态专家和贡献保留的多模态控制实现任意子集模态的组合。

**🔧 技术方法**

结合 Wan2.2 视频扩散骨干、Surg-ARCA 控制适配器、VACE 结构、锚相关注意力、流匹配损失及区域感知正则化等技术。

**📊 数据集**

使用自构建的 Cholec80‑SurgWAM 数据集，包含 6,001 条 49 帧腹腔镜剪辑、层次掩码、深度、光流、边缘和文本描述。

**📈 对比分析**

在多项基线（LTXV‑2B、Wan2.2、Cosmos‑H‑Predict、SurgSora、Endora、Cosmos‑H‑Transfer、VACE‑Wan2.2、ControlNet‑Wan2.2）上采用 PSNR、SSIM、LPIPS、FID、FVD 等指标对比，Surg‑UniWorld 在 PSNR 21.25、FVD 92.98、FID 7.36 等指标上均领先。

**⚠️ 局限性**

对极端遮挡和快速相机运动仍表现不佳，缺乏显式相机运动建模与长时序动作条件化。

---

## 162. HarnessSafe: Evaluating Safety Across Persistent Carriers in Agent Harnesses

**arXiv ID:** 2608.06984 | [PDF](https://arxiv.org/pdf/2608.06984v1)

**作者:** Xiao Zhang `[一作]` (Beijing University of Posts and Telecommunications), Zhaofeng He `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了名为 HarnessSafe 的基准，包含 328 个可执行的持续风险案例，覆盖七大持久载体家族，用于评估 LLM 代理的安全性。

**💡 创新点**

首次引入持久风险生命周期（Persistent‑Risk Lifecycle）模型与多阶段基于追踪的评估方法，能够定位风险链在不同阶段被阻断的位置，从而实现跨 harness 的可比性。

**🔧 技术方法**

采用生命周期建模、执行追踪与阶段归属、链段得分（CSS）等技术，对多种主流代理 harness 进行统一评估。

**📊 数据集**

自构造的 328 个案例集合，按内存、技能、工具/MCP、跨载体转换、子代理、会话摘要及共享工件等七个持久载体家族进行系统覆盖。

**📈 对比分析**

通过在公共支持集上统一评估各 harness‑model 配置的 CSS 得分，最高得分为 Codex CLI 62.3，最低为 MiniMax M2.5 22.7，展示了配置间风险阻断差异显著。

**⚠️ 局限性**

受限于部分 harness 对某些家族的缺失支持，导致案例覆盖不完整；基准仅覆盖 328 例而非全部潜在风险，且实验未覆盖动态攻击适应场景。

---

## 163. RibAssist 3D: Biplanar Rib-Fracture Detection, Addressing, and Selective 3D Localization from CT-Derived Projections

**arXiv ID:** 2608.06914 | [PDF](https://arxiv.org/pdf/2608.06914v1)

**作者:** Kabila Haile Soboka `[一作]` `[通讯]` (University of Texas at Austin), Kabila Haile Soboka (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并验证一种基于CT模拟正射投影的双平面骨折检测与三维定位框架RibAssist 3D，强调在受控误报预算下的信心门控三维定位。

**💡 创新点**

创新点在于将三维定位拆解为几何、定位、对应三大子任务，发现对应阶段是瓶颈，并通过后期侧面检测器的再训练实现首个零误报预算下的非零召回。

**🔧 技术方法**

采用U‑Net骨折检测网络、基于共享轴的候选图构建、一次性赋值与置信度门控、以及双流CNN骨干位置信息预测等技术。

**📊 数据集**

使用公开的RibFrac和RibSeg v2数据集，生成正射投影并分为训练、诊断与密封测试三份。

**📈 对比分析**

与基准FracNet等三维检测模型对比，RibAssist 3D在10 mm容忍度下实现约2.5%召回率、每病例0.436个误报，显示其在受限误报预算下的可行性。

**⚠️ 局限性**

局限包括投影仅为模拟正射图像、缺乏负样本评估、低召回率、且仅在受限的两视图设置下验证，未测试真实双平面放射数据或临床工作流影响。

---

## 164. WhiteNet: Robust Identification of Overlapping IEEE 802.11 Signals Across Unseen Channels

**arXiv ID:** 2608.06581 | [PDF](https://arxiv.org/pdf/2608.06581v1)

**作者:** Ildi Alla `[一作]` (University of Luxembourg), Vincent Lenders `[通讯]` (University of Luxembourg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对无线频谱中出现的多协议（802.11ax、802.11b/g/n）叠加信号，提出了一个基于频谱白化的端到端识别框架，实现了对未知通道条件下的鲁棒协议分类；

**💡 创新点**

创新点包括：①利用物理层频谱尺度分离（通道相干带宽 >> OFDM子载波间距）设计频谱白化预处理，直接在信号级抑制频率选择性衰落；②构建物理上合理的合成叠加器，以单协议捕获为基础生成多协议叠加样本，极大减少真实多源数据需求；③采用U‑Net+非局部注意力网络与知识蒸馏实现轻量化，适配边缘设备；

**🔧 技术方法**

核心技术包括：频谱白化（离散傅里叶变换+平滑功率谱估计）、物理仿真混合器（多径、CFO、共享收发器失真）、U‑Net编码‑解码网络+非局部注意力、渐进式训练管线、知识蒸馏与特征对齐；

**📊 数据集**

使用公开的现场覆盖（OTA）Wi‑Fi数据集（包含 ax、b、g、n 四代协议的20 MHz同频叠加录音），其中 S1、S2 训练/验证，S3 为完全未见的通道条件；

**📈 对比分析**

与公开基线（ResNet、Transformer、JDM 等）以及域自适应方法（DANN、Reptile）在 held‑out S3 进行对比。该方法在 unseen 通道下 exact‑match accuracy 提升 26 pp（从 47.4 % 到 73.6 %），在分布内误差缩小 40 pp（从 40.0 pp 到 8.0 pp）。模型参数仅 889 K，约比前沿方法少 7.7 倍，边缘蒸馏版可压至 10 K 参数，仍保持 60–70 % 的 EM；

**⚠️ 局限性**

局限性包括：仅验证在室内 802.11 协议集，未覆盖其它无线技术；在极端重叠或极低 SNR 条件下性能仍有下降；白化参数对非常宽的通道（B_c 接近子载波间距）可能失效；边缘蒸馏版本的准确率相对基准显著下降，需要在实际部署前进行更多场景测试。

---

## 165. Hierarchical Quantization with Domain-Adaptive Sparse Routing for Generative Cross-Domain Recommendation

**arXiv ID:** 2608.06997 | [PDF](https://arxiv.org/pdf/2608.06997v1)

**作者:** Haiying He `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种统一的生成式跨域序列推荐框架（HD‑REC），通过层次化的域感知量化器、域自适应稀疏专家混合网络以及跨粒度路由一致性学习，实现了跨域推荐的生成式建模；

**💡 创新点**

创新点包括：①层次化域感知量化器，将全局共享的粗粒度代码表与自适应路由的细粒度代码表相结合，既保持跨域共享，又保留细粒度差异化；②域自适应稀疏MoE，在共享专家基础上按输入动态选取专用专家，提供条件化模型容量；③跨粒度路由一致性学习，通过KL正则化让同一商品的语义Token保持路由一致性，提升表示稳定性；

**🔧 技术方法**

采用了向量量化自编码器（RQ‑VAE）实现语义ID分解、Gumbel‑Softmax路由、Transformer（T5）序列生成、稀疏Mixture‑of‑Experts、负对数似然、负熵负载均衡、以及跨粒度一致性损失；

**📊 数据集**

实验使用了三个公开的跨域数据集对：Amazon Clothing‑Sports、Amazon Electronics‑Phones 以及 Douban Books‑Movies；

**📈 对比分析**

与单域序列推荐（BERT4Rec、SASRec、STOSA）、生成式推荐（VQ‑Rec、TIGER、HSTU）以及跨域序列推荐（C2DSR、TriCDR、LLM4CDSR、GenCDR）等基线在 HitRate@10 和 NDCG@10 上进行对比。HD‑REC 在 Sports、Electronics、Phones 等领域均实现了 9.9%–17.6% 的 H@10 提升，并在大部分指标上优于最强基线；

**⚠️ 局限性**

局限性主要体现在：①仅验证了双域设置，缺乏多域扩展的实验；②依赖文本内容特征，对噪声或缺失的商品信息敏感；③评估仅为离线基准，未考虑动态商品目录、实时用户兴趣变化以及在线反馈循环；

---

## 166. Solution Space Partitioning for Extremal Set Theory

**arXiv ID:** 2608.06432 | [PDF](https://arxiv.org/pdf/2608.06432v1)

**作者:** Jesse Looney `[一作]` (Amherst College), Haoze Wu `[通讯]` (Amherst College)

**通讯引用:** 854 | [OpenAlex ID](https://openalex.org/A5100637734)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种在极值集合论中按等价（同构）族进行语义级别划分的分区方法，并将其形式化为可证明的推理系统。

**💡 创新点**

创新点在于：①通过对部分族的扩展与禁止等价族的操作实现真正的语义分区；②证明了该推理系统的完备性、无穷性终止性和分区互异性；③在 Rust 中实现了无缝的求解器无关接口，可与 SAT/ILP 求解器协同使用。

**🔧 技术方法**

使用的技术包括：基于同构族的分区算法、证明系统的形式化与证明、SAT/ILP 编码（包括总和计数器、整数线性规划）、分布式并行求解器（Kissat、SCIP）、动态深度自适应分区以及证书生成与 VIPR 校验。

**📊 数据集**

主要数据集为 Chvátal’s Conjecture 的有限案例（n=7 与 n=8 的全子集空间），通过对 2^2^n 的子集族进行枚举与划分。

**📈 对比分析**

与 Cube and Conquer 的 look‑ahead 分区和标准 SCIP 进行比较。分区方法在 SAT 上生成的子问题更少、解时间更短；在 ILP 上通过动态深度分区实现了约 38× 的 wall‑clock 加速，并成功得到 n=8 的 14 GB 证书，显著低于以前单一运行可能产生的 1 TB 证书。

**⚠️ 局限性**

局限性包括：①分区步骤本身缺乏可验证的证明，导致最终证明只能验证子问题；②分区效果依赖于同构检测和启发式扩展选择，可能在其他问题上表现不佳；③SAT 编码仍较为低效，ILP 依赖求解器参数；④证书大小仍庞大，分区方法未能进一步压缩；⑤对非同构不变谓词的适用性有限。

---

## 167. Does Splitting a Triage Decision Across Agents Hide Bias or Help Catch It? A Multi-Agent Simulation Study of LLM-Based Resource Allocation Under Audit Capacity Constraints

**arXiv ID:** 2608.06949 | [PDF](https://arxiv.org/pdf/2608.06949v1)

**作者:** Paul-Peter Arslan `[一作]` `[通讯]` (Institute for Future Technologies), Paul-Peter Arslan (Institute for Future Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过在合成灾难救援模拟器中对比单体LLM与分工多智能体（评估、分配、独立审核）管道，探究在资源紧张与审计负荷不同情况下的群体偏差产生与检测情况。

**💡 创新点**

创新点在于：①首次将双胞胎对（Twin-pair）方法应用于多智能体决策管道，客观衡量人口属性对决策结果的偏差；②揭示审计容量对偏差检测覆盖率的关键作用；③提出基于风险优先的审计队列，显著恢复覆盖率。

**🔧 技术方法**

使用技术包括：GPT‑4o‑mini作为所有LLM调用；构建的灾难三排救援仿真器；2×2×2因子设计（案件流、床位稀缺、审计容量）；风险优先审计队列算法；统计检验（p值、Wilson CI）及覆盖率/判定率分解。

**📊 数据集**

数据集为合成生成的灾难救援病例，采用双胞胎对设计，病例相同临床严重度，仅在一项人口属性上不同。

**📈 对比分析**

比较方法：在192个实验周期（共2304个双胞胎对）下对单体与多智能体条件下的偏差率、审计覆盖率、判定率进行统计比较。结果显示：单体与多智能体的偏差率差异不显著（6.9% vs 6.1%，p=0.498）；审计覆盖率随审计负荷显著下降（100%→65.6%，p<0.001）；风险优先队列将覆盖率从65.6%提升至91.7%（p=0.028），暗示队列调度可提升检测效果。

**⚠️ 局限性**

限制包括：仅使用单一模型（GPT‑4o‑mini），样本量有限，未进行对抗性复制，结果可能不适用于其他模型或真实部署环境，且多智能体拓扑与审计策略的通用性待进一步验证。

---

## 168. A Parameter-Specific Retrieval and Knowledge-Guided Reasoning Framework for LLM-Based GPSR Optimization in FANETs

**arXiv ID:** 2608.06760 | [PDF](https://arxiv.org/pdf/2608.06760v1)

**作者:** Zhipeng Lin `[一作]` (Chengdu University of Technology), Xiaojun Yuan `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在高动态飞行自组网络中，提出一种基于大语言模型的参数特定多索引检索与知识引导推理框架（PMKR-GPSR），通过动态调整GPSR路由参数来提升网络性能。

**💡 创新点**

创新点在于：①引入参数特定多索引检索机制，显著提升检索相关性；②构建知识引导约束图，将协议约束融入LLM推理过程；③在同一框架内实现经验检索、约束推理与参数优化，兼顾性能与可解释性。

**🔧 技术方法**

采用的技术包括：大语言模型（Qwen2.5-3B-Instruct）、多索引检索、知识图谱约束推理、基于NS‑3的仿真评估、Z-score归一化与统计特征提取。

**📊 数据集**

使用数据集：离线仿真产生的16,800条路由经验（包含场景特征、参数与性能），以及在NS‑3仿真平台下生成的三维FANET网络数据。

**📈 对比分析**

实验与基线对比：与原始GPSR、CF‑GPSR、多路径GPSR、QLGR进行比较。结果显示，在速度80 m/s时，PMKR‑GPSR的PDR提升约30%，E2E延迟降低约20%；在不同模型规模评估中，3B模型在性能与推理延迟之间取得最佳折中。

**⚠️ 局限性**

限制：①需要人工构建经验数据库与知识图谱；②LLM推理受模型规模限制，推理延迟仍高；③仅在仿真环境验证，真实环境中的适应性与鲁棒性仍待进一步研究。

---

## 169. Frame-Level Pansori Mode Classification with Complementary Audio Representations

**arXiv ID:** 2608.06633 | [PDF](https://arxiv.org/pdf/2608.06633v1)

**作者:** Sangheon Park `[一作]` (Georgia Institute of Technology), Dasaem Jeong `[通讯]` (Sogang University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于多模态的帧级潘索里调式分类框架，并构建了46小时、396曲目专家标注的语料库。

**💡 创新点**

创新在于将mel谱、F0、MIDI和文化自监督嵌入等四种互补输入对应调式的不同维度，设计了Daemok‑Shared与Work‑level两种评估拆分，揭示模型真正学习调式特征而非记忆曲目。

**🔧 技术方法**

采用CRNN结构、Focal Loss、数据增强、源分离、MIDI转录、PESTO F0、CultureMERT预训练模型，并通过late fusion形成多模态集成。

**📊 数据集**

使用了46小时、396曲目（含5个batang及其他作品）的帧级标注语料库，按五大调式归类。

**📈 对比分析**

通过masked macro‑F1进行比较，三大常见调式在两种拆分下差距仅2–3.6 F1，整体差距5.9–8.6；源分离降低Changjo表现；CultureMERT在Work‑level拆分最差；ensemble作为分析工具展示多模态分歧。

**⚠️ 局限性**

局限在于对稀有调式（Changjo、Ujo）误判率高，源分离会移除关键节奏信息，CultureMERT跨文化泛化受限，且仅在少数现代作品验证了泛化能力。

---

## 170. TradeVerse: A Longitudinal Benchmark of Political Negotiation in International Trade

**arXiv ID:** 2608.06549 | [PDF](https://arxiv.org/pdf/2608.06549v1)

**作者:** Debodeep Banerjee `[一作]` (University of Pisa), Amitangshu Dasgupta `[通讯]` (RazorPay)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于WTO贸易关切记录的纵向多轮对话基准，定义HS章节预测、答复国识别与最终陈述生成三项任务；

**💡 创新点**

首次在真实多国长期贸易谈判文本上评估LLM的时序推理与政治语境理解，无需人工标注，并揭示模型对西方国家识别的系统性优势；

**🔧 技术方法**

使用多种SOTA LLM（Nemotron Ultra、Llama 3.3、GLM‑5.2、DeepSeek‑V4‑Pro、GPT‑OSS‑120B、Kimi‑K2.7‑Code）进行零样本推理，配合长文本提示，评估指标包括 Precision/Recall/F1、BLEU‑4、ROUGE‑L 与 BERTScore；

**📊 数据集**

采用1170个贸易关切、6933次会议记录、26 219条陈述的数据集，覆盖5个WTO委员会、89个HS章节、68个答复国，数据直接从官方会议记录中提取；

**📈 对比分析**

通过对比不同模型在三任务上的表现：HS预测 F1≈55–65%，答复国识别准确率>90%（西方高于非西方），最终陈述生成 BLEU<12、BERTScore≈60%，显示模型整体能力有限但存在明显的区域性偏差；

**⚠️ 局限性**

存在的限制包括HS分类过度泛化、生成陈述缺乏具体细节、对非西方国家识别存在系统性偏差、对轮数变化的敏感性不均衡，且未深入探究偏差根源。

---

## 171. The Price of Order in the Logarithmic Method

**arXiv ID:** 2608.06388 | [PDF](https://arxiv.org/pdf/2608.06388v1)

**作者:** Sichen Wang `[一作]` (Shenzhen MSU-BIT University), Jingbang Chen `[通讯]` (CUHK-Shenzhen)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究插入-only 的 merge-stack 模型下，基于 log‑method 的写-读产品最优下界，按查询类型给出 nlog²n、nlog³n、n² 的三种结果。

**💡 创新点**

首次把不同查询（哈希可证、需要定位键值、仅给定秩）与写读开销之间的“量化分离”形式化，揭示了“无阶级”与“阶级”查询的本质差距，并证明了在强制物化和单向扫描下的不可跨组件的分段搜索障碍。

**🔧 技术方法**

采用 merge‑forest 结构、信息熵与编码理论、固定状态下的搜索下界、以及“大小多样性法则”对写读产品进行紧确分析。

**📊 数据集**

本研究为理论工作，未使用任何实际数据集，所有结果均在抽象键（可哈希、可比较）的假设下得到。

**📈 对比分析**

对比传统的 log‑method 上界（Bentley–Saxe 方案），本文提供匹配的下界，证明了在该模型下各查询类型的最优写读产品：membership/哈希可证为 nlog²n，顺序/范围查询为 nlog³n，选择查询为 n²；并指出整数键或可随机访问时可缩减部分复杂度。

**⚠️ 局限性**

局限性在于仅适用于抽象键、强物化且单向扫描的模型；若放宽这些假设（如非物化链路、双端编辑或随机访问），下界可能失效或需重新证明。

---

## 172. Auctioning Attention on Social Networks

**arXiv ID:** 2608.06665 | [PDF](https://arxiv.org/pdf/2608.06665v1)

**作者:** Andy Lee `[一作]` (University of Illinois Urbana Champaign), Hari Sundaram `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于用户间拍卖的社交媒体关注分配机制，并结合税收政策调节内容质量，以实现对内容生产者、消费者、平台和社会的多重利益平衡。

**💡 创新点**

创新点在于：①首次将 VCG 拍卖与自动竞价（autobidder）相结合，实现用户对自己帖子关注度的最优投标；②设计可调节的税收机制，根据内容质量或用户兴趣动态分配预算，抑制低质量或有害内容；③在理论上证明了自动竞价在两期和无穷期下的最优性与弱激励兼容性。

**🔧 技术方法**

技术核心包括：
- VCG 拍卖与自适应竞价算法；
- 基于最大流的社区兴趣建模与余弦相似度衡量；
- 使用贝尔曼方程和 KKT 条件求解多期最优竞价；
- 强化学习（RL）学习税收策略；
- 通过模拟评估 Lorenz 曲线和 Gini 系数衡量关注度公平性。

**📊 数据集**

实验数据集主要有三种合成网络（k‑regular 环网、随机块模型、尺度无关网络）以及真实欧盟邮件网络（EU email network）来验证机制在不同拓扑下的表现。

**📈 对比分析**

与基线（时间线、亲和度、质量、消费者/平台/社会福利最大化）进行比较；实验显示：
- 在三种合成网络和真实网络中，拍卖+税收方案平均提升 36.3% 的生产者福利；
- 相比于基线，注意力分布更公平（Gini 系数下降），尤其在度分布偏斜的网络中表现更显著；
- 在不同税率设置下，可实现生产者福利与社会福利的权衡。

**⚠️ 局限性**

局限与未来工作：
- 模型假设网络、目标集合和内容策略固定，未考虑动态演化；
- 未建模用户对内容的真实价值估计与分享行为；
- 仅对预算约束下的竞价做最优分析，未考虑信息不对称或噪声估计误差；
- 税收策略需手工调参，自动学习仍处于实验阶段。

---

## 173. The Reeb Structure of Bend Distance in Grid Domains: Cycle Bounds with Holes, Exact Sector Geometry on Disks, and the Two-Port Constant $c_2^\square=3$

**arXiv ID:** 2608.06421 | [PDF](https://arxiv.org/pdf/2608.06421v1)

**作者:** Aoji Li `[一作]`, Guangbo Ding `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究离散化简单多边形（含洞）中的协同运动规划，建立了弯折距离的分段仿射扩展，并证明单口径分区图的环数不超过洞数，同时给出线性时间的反馈集构造。

**💡 创新点**

创新点在于将弯折距离与Reeb多图关联、证明纯二维立方域中单口径分区图环数上界为洞数、以及在无洞盘形域中给出完全一侧挤压几何与树宽为3的精确定界。

**🔧 技术方法**

主要技术包括二维Reeb图与极限理论、Gelbukh环秩定理、Alexander对偶、状态图的0/1权重表示、凸性与CAT(0)立方体复合体等。

**📊 数据集**

论文不涉及实验数据集，全部结果为理论证明。

**📈 对比分析**

通过构造反例与严格证明得到环数与树宽上界，单口径上界为3，反馈集线性时间求解；未做实验性能评估。

**⚠️ 局限性**

局限性在于仅给出结构性上界，未实现f(k,h)n^O(1)的固定参数算法；在有洞情况下几何性质破坏，缺乏完整的多口径收缩证明；边离散化范围下常数仅在条件下给出。

---

## 174. Detecting and Characterizing Massively Shared IP Addresses

**arXiv ID:** 2608.06517 | [PDF](https://arxiv.org/pdf/2608.06517v1)

**作者:** Amanda Hsu `[一作]` (Georgia Institute of Technology, Akamai), Philipp Richter `[通讯]` (Akamai)

**通讯引用:** 1707 | [OpenAlex ID](https://openalex.org/A5035619973)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过对 CDN 日志中每个 IP 的小时请求量进行 Fast Fourier Transform 分析，检测并特征化大规模共享 IP（IP）及其流量形状。

**💡 创新点**

创新点在于仅利用流量时间序列的周期性和光滑度，提出了轻量级的 R^2 曲线拟合阈值检测方法，避免了对客户端信息的依赖。

**🔧 技术方法**

主要技术包括 FFT、频率能量比较、R^2 拟合评估及基于频率截断的重构曲线。

**📊 数据集**

使用数据集为全球最大 CDN 的访问日志（IPv4 与 IPv6 请求计数）以及 RUM、WHOIS、BGP、User‑Agent 等上下文信息。

**📈 对比分析**

与基于客户端或 P2P 的检测方法对比，该方法覆盖面更广、精度更高，识别出约 1.6% 活跃 IPv4 地址占 41% 流量的 IP，并展示了随时间增长的趋势。

**⚠️ 局限性**

局限性包括仅从 CDN 视角观察、对高波动流量可能误判、无法精确计数后端用户、对 IP churn 敏感以及缺乏绝对基准验证。

---

## 175. From Interpretation to Compilation: A Compilation-Based Execution Engine for Semantic Operator Systems

**arXiv ID:** 2608.06677 | [PDF](https://arxiv.org/pdf/2608.06677v1)

**作者:** Wenkai Dong `[一作]` (University of Hawaii at Manoa), Yifan Wang `[通讯]` (University of Hawaii at Manoa)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 SemBaker，一个外部插件，通过将语义操作编译成确定性代码，实现在语义操作系统中的混合执行；

**💡 创新点**

首次提出语义操作的编译式执行范式，并配备基于成本的优化器与并行编译执行框架，显著降低 LLM 调用次数与成本；

**🔧 技术方法**

利用 LLM 作为编译器生成 Python 函数，采用成本模型决定每个操作是本地编译还是后台原生执行，使用多线程并行编译、缓存、可选的精炼与验证；

**📊 数据集**

在 ManyModalQA、HybridQA、MMQA、SemBench 等多模态问答数据集以及 UCSD Steam Reviews 与 UCI SMS Spam 集合上进行实验；

**📈 对比分析**

与原生语义操作系统在同一后端进行对比，平均加速 4.8–6.3 倍、成本降低 5.4–10.7 倍；在 ManyModalQA 上延迟下降 79%，在 HybridQA 上成本下降 91%；质量指标（EM/F1）与原生系统相当甚至更优；

**⚠️ 局限性**

对高度语义化的谓词编译效果有限，精炼与验证在不同场景表现不稳定；目前仅支持文本/表格操作，无法处理图像输入；需要手工调优阈值，编译时 LLM 调用对资源有一定占用。

---

## 176. Multi-Level Modeling of Large Language Model Inference Latency and Energy via Hybrid Analytical--Machine-Learning Predictors

**arXiv ID:** 2608.06723 | [PDF](https://arxiv.org/pdf/2608.06723v1)

**作者:** Saeid Shokoufa `[一作]` (University of Southern California), Massoud Pedram `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种混合层级模型 HYMELL，用来准确预测大型语言模型在推理时的延迟和能耗。

**💡 创新点**

创新点在于将 GPU 低层算子（GEMM、Softmax、RMSNorm）用可切换的两阶段解析模型估算，再用轻量级 MLP 预测注意力/前馈块以及整体系统级成本，实现跨架构、跨任务的可迁移性与高精度。

**🔧 技术方法**

技术手段包括：GPU 低层算子解析回归（针对 launch-bound 与 memory-bound 阶段），多层感知机（MLP）校正注意力/FFN 以及全局预测；使用 CUDA 计时、NVML 能耗采集；在 NVIDIA H100 NVL GPU 上进行实验。

**📊 数据集**

数据集为约 1,200 个不同配置（模型维度、层数、序列长度、批大小、MoE/密集 FFN、MHA/GQA）在 HuggingFace Transformers 上的真实推理测量。

**📈 对比分析**

与 AMALI 的周期级预测相比，HYMELL 在 LLaMA‑3‑8B 上单序列误差分别为 4.19%（延迟）和 2.92%（能耗）；在整体模型预测上，平均 MAPE 约 10%（延迟）和 12%（能耗），在不同 GPU（H100、A6000）及注意力变体上保持 5–10% 误差，显示出更好的通用性和更低的误差。

**⚠️ 局限性**

局限性包括：目前仅针对单 GPU 推理，跨多 GPU（张量并行）需要额外的通信模型；模型需要针对每种硬件重新采样与校准；对新型算子或极端稀疏/异构批处理的支持仍需进一步扩展。

---

## 177. Understanding the Energy Impact of Software Refactoring: A Workload-Aware Study of Controlled Examples and Real-World Commits

**arXiv ID:** 2608.06620 | [PDF](https://arxiv.org/pdf/2608.06620v1)

**作者:** Haibo Wang `[一作]` (Concordia University), Shin Hwei Tan `[通讯]` (Concordia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建两个基准——Micro‑benchmark（68种重构类型的可执行 Java 示例）和 Practical‑benchmark（481 个来自 430 个 GitHub 项目的真实重构提交），在多种工作负载下对重构前后版本进行能耗测量与统计，系统探究重构对能耗的影响、与重构类型/组合的关联、能耗变化的解释因素以及现有指标/LLM 对能耗回归的识别效果。

**💡 创新点**

1) 首次在大规模、多工作负载、真实提交环境下量化重构能耗；2) 通过 LLM 自动生成可重复、多规模工作负载；3) 结合运行时、静态、变更、覆盖、语义等多维指标，解释能耗差异；4) 评估传统指标和 LLM 在能耗回归识别中的表现。

**🔧 技术方法**

能源测量使用 JoularJX；运行时指标通过 JMH + Java Flight Recorder；静态/变更/覆盖指标使用 CK、RefactoringMiner、JaCoCo 等；LLM 预测采用 OpenAI text‑embedding‑3‑large；统计分析采用 Wilcoxon 符号秩检验、Cliff’s Δ、Spearman 相关、Fisher 检验、Benjamini–Hochberg 校正；机器学习模型为 Logistic Regression、Random Forest、XGBoost；LLM 预测基于 Prompt。

**📊 数据集**

Micro‑benchmark：68 代码片段，覆盖 61 种重构类型；Practical‑benchmark：481 次纯重构提交，来自 430 个 Maven Java 项目，包含 345 次有 RefactoringMiner 检测到的重构，平均 10 行变动。

**📈 对比分析**

实验显示：在 Micro‑benchmark 中 51.8% 的重构–工作负载对出现显著能耗差异，27.9% 超过 10% 变化；在 Practical‑benchmark 中仅 7.5% 的提交显著改变能耗，超过 10% 的占 66.7%。工作负载多样性显著影响结果；传统指标（执行时间、分配量等）仅部分解释能耗差异；基于指标的预测模型及 LLM 在识别能耗回归方面表现均不佳，精度均低于 0.4。

**⚠️ 局限性**

1) 仍无法准确预测重构的能耗影响；2) 仅基于测试套件的工作负载可能无法覆盖所有受影响路径；3) 真实提交中多重重构和功能变更难以分离导致解释性不足；4) LLM 预测受限于未 fine‑tuned，缺乏针对能耗的专门训练；5) 仅考虑 Java，缺乏跨语言或嵌入式设备的验证。

---

## 178. Factorized Hypothesis Search for Evidence-to-Taxonomy Retrieval

**arXiv ID:** 2608.06614 | [PDF](https://arxiv.org/pdf/2608.06614v1)

**作者:** Linhai Ma `[一作]` (Fin AI), Víctor Gutiérrez-Basulto `[通讯]` (Cardiff University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为Factorized Hypothesis Search（FHS）的检索框架，用以将间接证据（如表格单元格或临床提及）转换为可用于大型概念库检索的查询；

**💡 创新点**

创新点在于将检索过程拆解为多个部分性语义假设（factorized hypothesis），并在每个假设上进行结构化查询渲染、候选集合融合以及维度级验证，从而显著提升头部检索性能；

**🔧 技术方法**

核心技术包括基于LLM的假设生成与验证、双模式查询渲染（label‑form与definition‑form）、逆序排名融合（RRF）、以及候选级重新排序；

**📊 数据集**

实验使用了两个公开数据集：美国通用会计准则（US‑GAAP）17,388条目对应的财务标注任务，以及CodiEsp临床诊断编码任务（ICD‑10‑CM 71,344条目）；

**📈 对比分析**

与直接检索、单通道、并行采样、迭代自我修正等基线相比，FHS在Recall@1、MRR和最终准确率上均实现了显著提升（如财务标注Recall@1从0.12提升至0.18，CodiEsp Recall@1从0.20提升至0.26），且ablations表明假设分解、定义式查询与候选级验证是主要贡献；

**⚠️ 局限性**

局限性包括：依赖预定义的六维语义维度，若目标概念库缺乏相似结构则需额外的模式诱导；此外模型在不同规模与家族的鲁棒性未进一步评估；

---

## 179. Quantization Damage Is Multiplicative, Not Additive

**arXiv ID:** 2608.06564 | [PDF](https://arxiv.org/pdf/2608.06564v1)

**作者:** Zekun Wu `[一作]` (Holistic AI), Adriano Koshiyama `[通讯]` (Holistic AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究后训练量化（尤其四位以下）对大型语言模型决策的影响，提出量化导致每个决策边际（margin）按比例收缩，而非添加固定噪声，导致某些决策失效；并用对决策边际的逐条测量验证这一“乘法收缩”定律，给出了预测翻转概率的公式并证明其校准性；同时探讨了不同量化方法、模型家族、位宽以及激活量化对决策失效的差异，并评估了多种修复手段，结论是仅增加一位比现有修复更有效。

**💡 创新点**

创新点在于：①提出并验证了量化导致的“边际收缩”乘法模型，否定了传统的“固定噪声”假设；②给出了基于此定律的翻转概率公式，未使用翻转数据训练，且校准良好；③通过细粒度决策测量揭示了工具调用与安全拒绝等行为在位宽下降时的“一方向性崩溃”现象；④系统性比较了多种量化方案、模型家族及激活量化对决策损伤的影响，发现单增加一位是最具成本效益的修复。

**🔧 技术方法**

主要技术包括：后训练量化（Round‑to‑Nearest、GPTQ、LLama.cpp GGUF等），对决策边际的对齐测量（每个问题只考虑第一token的logit差值），统计模型比较（BIC、最大似然），基于乘法收缩定律的翻转概率推导与校准，离散化实验评估与基准对比。

**📊 数据集**

使用了作者自建的四套测试集：工具调用（280条）、安全（400条）、常识（220条）和社会偏见（192条，来自BBQ基准），以及在公开LLM（Qwen3、Gemma、Llama、GPT等）上进行的量化与激活量化实验。

**📈 对比分析**

对比方法：在同一模型与量化方案下分别测量全精度与量化后的决策边际，并统计翻转率；通过BIC比较乘法收缩模型与多种加性噪声模型；使用翻转概率公式预测未见过的决策翻转率，得到中位绝对误差约1.5–1.8个百分点，校准误差0.004，表明模型预测性能优于无修复或简单加性噪声假设。

**⚠️ 局限性**

局限性：①未覆盖多步生成或长文本决策；②某些细胞（如工具结果使用）边际分布不满足恒定方差假设，导致预测偏差；③激活量化实验样本有限；④翻转概率公式在位宽极低（2位）时预测不佳；⑤未探讨跨模型参数迁移，仅在单模型内评估；⑥对大规模模型的抗压性机制仍未阐明。

---

## 180. Cryptanalytic Extraction of Isolated Bias-Free GLU Feed-Forward Blocks by Antipodal Separation

**arXiv ID:** 2608.06631 | [PDF](https://arxiv.org/pdf/2608.06631v1)

**作者:** Chunhui Shi `[一作]` (Avitam), Xinwen Fu `[通讯]` (University of Massachusetts Lowell)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种仅通过前向查询就能提取无偏置GLU前馈块（bias‑free GLU）的权重的加密分析方法。

**💡 创新点**

核心创新在于利用有限差分二阶导数（Hessian）得到门控方向候选，再通过在 x 与 -x 处的奇偶分解，分别恢复门的幅度、耦合与方向，从而实现对双分支结构的完整分离。

**🔧 技术方法**

采用的技术包括前向查询攻击、有限差分 Hessian 估计、奇偶（odd/even）分解、L‑BFGS 与 Levenberg–Marquardt 迭代、条件修复（残差子空间补救、留一去除重复）、嵌套输入维度链、前向修整和尺度选择。

**📊 数据集**

实验使用的模型包括 Qwen3‑0.6B、Llama‑3.2‑1B、Gemma‑3‑1B，分别在 FP64、BF16、FP16、TF32 等多种数值精度下进行评估。

**📈 对比分析**

与先前仅针对非门控网络的提取方法相比，该方法在高精度设置下实现了 0.1% 以下的验证误差，门方向匹配率超过 99%，但在低精度（BF16/FP16/TF32）下无法恢复到存储精度（未达存储精确匹配）。

**⚠️ 局限性**

主要局限包括：需直接访问并知道单块接口与元数据；无法直接推断完整语言模型的最终输出；对查询量极大且对返回精度敏感；仅在单模型单实验中验证，未对真实服务的可行性进行评估。

---

## 181. Fast LapSum: Exact Differentiable Top-k at Million Scale

**arXiv ID:** 2608.06912 | [PDF](https://arxiv.org/pdf/2608.06912v1)

**作者:** Łukasz Struski `[一作]` (Jagiellonian University), Jacek Tabor `[通讯]` (Jagiellonian University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Fast LapSum，一种GPU原生、可微、精确保预算的soft top‑k运算符，能在排序后以线性时间计算阈值，并通过概率括号实现百万级向量的高效求解。

**💡 创新点**

创新点包括：① 通过Laplace CDF闭式阈值求解，保证每次输出的总权重恰好为 k；② 在排序后使用一次稳定前缀/后缀扫描即可得到阈值曲线；③ 采用概率括号（binomial order statistics）局部化阈值，避免对整个向量排序；④ 推导并实现单次归约的向量–雅可比乘积，保持完全可微；⑤ 在GPU上实现后端，速度比传统二分查找快近两百倍。

**🔧 技术方法**

主要技术包括：LapSum 软 top‑k 公式、线性阈值计算与闭式根、概率括号与中间区间排序、GPU原生并行前缀/后缀扫描、一次性归约的向量–雅可比乘积、CUDA 核心融合、以及多路径调度（全排序 vs. 括号分支）。

**📊 数据集**

实验数据集主要包括：DIV2K（原分辨率图像）用于构造百万像素的稀疏对抗攻击；一组自定义的百万维向量用于测试稀疏图像编码器；在基准测试中使用随机/真实分布的向量与 ConvNeXt‑V2 分类器。

**📈 对比分析**

与 DFTopK、NeuralSort、SoftSort、Sinkhorn‑OT、Perturbed 等现有可微 top‑k 方法对比，Fast LapSum 在 10⁶–10⁸ 元素规模下的前向时间始终保持在毫秒级，10⁷ 时比 DFTopK 快 5–20×，10⁷ 时比官方 LapSum 快 1200×；在对抗攻击实验中，Fast LapSum 在 99% 成功率时的预算仅为 200（相较最优基线的 3100），显示显著的稀疏性和效率提升。

**⚠️ 局限性**

局限性包括：仅适用于 Laplace 核（无法直接迁移到其他平滑核）；概率括号的高概率保证在极端分数分布下可能失效，需要额外验证步骤；在最坏情况下（路径验证失败）会出现多次前向传播，导致运行时间升高；预计线性时间是平均复杂度，非严格最坏情况保证。

---

## 182. NxN E-valuation: Hypothesis Certification via a Conformal CRT Null

**arXiv ID:** 2608.06621 | [PDF](https://arxiv.org/pdf/2608.06621v1)

**作者:** Bin Wang `[一作]` (Meta), Yan Zhong `[通讯]` (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 N×N E-valuation，一种基于 e‑value 的通用、随时有效的假设证书方法，用于验证 LLM 等机器生成的每个单位级假设，避免手工构造检验。

**💡 创新点**

通过将大数据集自身作为无偏置的 null 以构造显著性与机制两个 e‑value，结合 N×N 交叉预测矩阵和多银行累积实现无多重检验、可扩展且区分真因果与伪关联。

**🔧 技术方法**

采用条件随机化检验 (CRT)、e‑value 合成、交叉预测矩阵、分层平均/方差多样性度量、单位内机制检验与交叉校准等技术。

**📊 数据集**

在论文中使用了合成的广告推荐世界（植入真实与伪假设）来评估算法，未使用真实数据以避免真值未知。

**📈 对比分析**

与传统 held‑out 预测验证相比，N×N 证书在 99% 真正率下正确判定四类伪假设，而传统方法误判 50%；在合成实验中所有伪假设均被拒绝，真实假设全部通过。

**⚠️ 局限性**

需足够大的样本以构建有效的 null；极罕见假设需多银行累积；多样性阈值 τ_div 经验调参且无 Type‑I 保证；对极高维输入的计算复杂度仍为 O(|U||T|)。

---

## 183. A Divide-and-Conquer Engine for Lexicographical Permutations: Accelerating State Evolution via Hybrid Software-Hardware CPU Instructions

**arXiv ID:** 2608.06384 | [PDF](https://arxiv.org/pdf/2608.06384v1)

**作者:** Yusheng Hu `[一作]` `[通讯]` (Independent Researcher), Yusheng Hu (Independent Researcher)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种名为LexCHA的硬件‑软件协同框架，通过预计算的掩码和SIMD指令实现无分支的字典排列生成；

**💡 创新点**

利用排列的分形块结构，将全局状态分解为宏观软件跳跃与微观硬件加速两层，消除传统算法中的分支误判与流水线停顿；

**🔧 技术方法**

基于factoradic表示、SSE/AVX shuffle指令、预计算的掩码表以及宏微分离的divide‑and‑conquer控制流；

**📊 数据集**

在 n=10~13 的排列序列上进行实验，分别在 AMD EPYC 与 Intel Core CPU 上测评；

**📈 对比分析**

与标准 C++ std::next_permutation 进行对比，LexCHA 在两种硬件上分别提升 7–17 倍，per‑perm 延迟随 n 几乎不变；

**⚠️ 局限性**

需要额外存储预计算掩码，k=5 时块大小固定，k 越大空间占用急剧上升，对极小 n 或非词典排列不具优势。

---

## 184. RegionDet: A Benchmark for Region Detection Beyond Object Instances

**arXiv ID:** 2608.06850 | [PDF](https://arxiv.org/pdf/2608.06850v1)

**作者:** Liang Wan `[一作]` (Tianjin University), Sirui Zhu `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了“Region Detection”任务并构建了RegionDet基准数据集，旨在检测由视觉状态、场景语境、对象关系和人类活动定义的区域。

**💡 创新点**

创新点在于将目标检测扩展到非离散、弱边界的区域范畴，并首次提供了包含八类区域目标的COCO兼容标注与评估协议；同时揭示了现有对象检测模型在该任务上的局限性。

**🔧 技术方法**

采用多种检测范式（两阶段、单阶段、anchor-free、扩散式、Transformer查询式）以及多种零射击/开放词汇视觉语言检测器（MMGroundingDINO、YOLO-World、YOLOE、LLMDet）进行系统评估，并使用COCO风格的AP、AP50、AP75指标。

**📊 数据集**

使用RegionDet数据集，共8010张图片、12588个区域标注，八个类别（Construction、Crossing、Damage、Queuing、Talking、Vendor、Waiting、Walking），训练集4005张/6371个，测试集4005张/6217个。

**📈 对比分析**

比较方法中，闭集检测器中RF‑DETR‑L最佳AP 37.6；单阶段和两阶段模型AP低于20；扩散式和anchor‑free模型在20–23左右。零射击/开放词汇检测器整体AP不足1.0，AP50低于3.0，表明其在区域目标上的泛化能力极差。

**⚠️ 局限性**

局限性包括：区域目标具有弱边界、强上下文依赖和关系定义，现有模型难以精确定位；零射击/开放词汇模型仍偏向单体对象，缺乏对区域级语义与关系的建模；Dataset规模相对有限，未来需更大规模、多样化的区域标注与更灵活的标注形式。

---

## 185. Flowing Through States: Neural ODE Regularization for Reinforcement Learning

**arXiv ID:** 2608.06595 | [PDF](https://arxiv.org/pdf/2608.06595v1)

**作者:** Mohamed Ghanem `[一作]` (CISPA Helmholtz Center for Information Security), Bernd Finkbeiner `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种无监督的流正则化（FlowReg），通过让状态嵌入遵循神经ODE生成的平滑流来对齐MDP语义轨迹与潜在轨迹，从而提升强化学习代理的性能。

**💡 创新点**

创新点在于将连续时间ODE流作为潜在空间的全局结构约束，将离散MDP轨迹映射到神经ODE的离散化路径，并通过对齐损失在训练时强制嵌入满足ODE流属性。

**🔧 技术方法**

使用的技术包括神经ODE（Neural ODE）与梯度可微数值求解器、Actor-Critic（A2C）和PPO算法、无监督路径对齐损失、RMSProp优化器以及Stable‑Baselines3框架。

**📊 数据集**

实验数据集为 11 个 Atari 游戏（Arcade Learning Environment）和 3 个 Minigrid 环境，均在固定训练步数下评估。

**📈 对比分析**

与传统 A2C / PPO 基线相比，FlowReg 在所有 Atari 任务上平均提升 30‑70% 奖励，并在 Minigrid 的 FourRooms 与 Dynamic‑Obstacles 任务中表现出显著优势；同时，潜在轨迹的路径长度、净位移与加速度能量等平滑指标均显著下降，证明嵌入空间更加平滑；运行时开销仅略高于基线，整体效率可接受。

**⚠️ 局限性**

主要局限包括：需要维护每条轨迹的 episode ID，导致对基于 episode 的训练管道有更高需求；ODE 流的唯一性禁止轨迹交叉，可能在存在瓶颈状态或循环需要的环境中限制探索；在极大状态空间的离散环境中该限制的影响相对较小。

---

## 186. MemPrism: Task-Conditioned Relational Memory Views for Long-Horizon Agents

**arXiv ID:** 2608.06745 | [PDF](https://arxiv.org/pdf/2608.06745v1)

**作者:** Zhisheng Chen `[一作]` (Nanyang Technological University), Jingwei Song `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MemPrism，分离持久事件流与任务条件的关系视图，解决长序列记忆中的表示不匹配问题。

**💡 创新点**

创新点在于：检索后通过任务条件动态构造关系视图，采用光学视图统一展示，并学习视图策略实现记忆接口的自适应。

**🔧 技术方法**

使用技术包括：事件流记录、轻量级Transformer+MLP视图策略、关系结构构造器、光学视图渲染、基于 GRPO 的策略优化。

**📊 数据集**

数据集：ALFWorld、EB-ALFRED、Mind2Web 等长序列文本/视觉/网页交互基准。

**📈 对比分析**

通过与 Full History、LangMem、A-Mem、Mem0 等基线对比，MemPrism 在 ALFWorld 40.71% SR、Mind2Web 12.87% 行动准确率、EB-ALFRED 17.7% SR 等显著提升，同时显著减少提示 token。

**⚠️ 局限性**

局限性：视图策略需要在不同任务 VLM 上进行更充分的迁移验证；光学视图渲染对高分辨率图像可能产生额外开销；尚未与可学习的记忆控制模块完整集成。

---

## 187. Vehicle routing problem using deep reinforcement learning - A case study about truck planning in the industry

**arXiv ID:** 2608.06668 | [PDF](https://arxiv.org/pdf/2608.06668v1)

**作者:** Siliang Lu `[一作]` (Bosch Center for Artificial Intelligence), Lili Wu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并实现了基于Transformer的深度强化学习模型，解决工业外部奶跑（EMR）与LTL混合的异构车辆容量路由问题

**💡 创新点**

将LTL与HCVRP融合为混合模式，提出可部署的无模型策略网络，利用DRL实现10%+成本降低和空载率控制，展示了训练样本生成与Transformer架构的有效性

**🔧 技术方法**

深度强化学习（PPO+Transformer政策网络）、图神经网络/Transformer架构、奖励函数与掩码约束设计、与MIP/启发式方法对比实验

**📊 数据集**

三份真实工业外部奶跑案例（订单量分别为171、360、504）及其合成训练/验证集（每个样本200订单），包含经纬度、需求量、服务窗口等特征

**📈 对比分析**

与仅使用LTL价格总和的基准对比，模型在三个用例分别降低了23.48%、18.10%和18.87%的运输成本；推理速度远快于MIP和启发式方法

**⚠️ 局限性**

需要大量与测试场景相似的合成样本才能训练；对动态约束、时间窗口、车辆数目自动选择等扩展验证不足；模型对不同业务场景的迁移性能有限

---

## 188. CAi Copilot: Reducing Operational Workload in Molecular Design through Intent-Driven Agentic Workflows

**arXiv ID:** 2608.06961 | [PDF](https://arxiv.org/pdf/2608.06961v1)

**作者:** Zhu Wang `[一作]` (Shanghai Artificial Intelligence Laboratory), Na Zou `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CAi Copilot，一个将分子设计意图转化为可执行工作流并生成可追溯候选分子及证据的三层代理。

**💡 创新点**

创新点在于把意图-证据工作流执行视为整体任务，统一意图映射、推理控制和工具执行，并实现多工具协同与证据记录。

**🔧 技术方法**

使用基于大型语言模型的推理层、研究接口层生成计划，以及包含生成、评价、筛选、优化等分子工具的执行层。

**📊 数据集**

使用 45 任务的 CAiMD 任务集以及 SMDD-Bench、LIDDiA 和 MolBench 等公开基准进行评估。

**📈 对比分析**

与 ChemCrow、Biomni、AgentD、Codex、Hermes 等对比，CAi 在 45 任务上平均 84.59% 的目标满足率，整体分数 75.71%，超过第二名 18.07 分。

**⚠️ 局限性**

局限在于长距离执行路径易缺失步骤、对结构依赖的虚拟筛选表现不佳、以及在激进优化中对骨架保留的限制。

---

## 189. TA-RAG: Tone Awareness as a Design Imperative for Retrieval-Augmented Generation

**arXiv ID:** 2608.06672 | [PDF](https://arxiv.org/pdf/2608.06672v1)

**作者:** Yong-Bin Kang `[一作]` (Swinburne University of Technology), Anthony McCosker `[通讯]` (Swinburne University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了Tone-Aware Retrieval‑Augmented Generation（TA‑RAG）框架，旨在在RAG系统中将沟通一致性与事实准确性双重目标纳入设计，解决传统RAG在检索与生成交互中出现的“上下文解耦”问题。

**💡 创新点**

创新点包括：
1) 将四大沟通约束（去污名语言、可读性匹配、收件人适配、共情框架）分布到RAG五个阶段；
2) 引入检索‑上下文‑生成‑验证五阶段管线，并在每个阶段显式执行约束；
3) 提出兼顾事实与沟通的联合评估议程，并讨论可测量的约束指标与语义保真度检查。

**🔧 技术方法**

主要技术手段：
- 语义检索（Dense Retrieval）与三种沟通信号（术语、可读性、角色相关性）相结合；
- 上下文构造时对检索文本做标注与替换；
- 生成阶段引入约束指令；
- 约束验证阶段采用规则检查、可读性指标、Rubric评分与语义相似度（BERTScore）等。

**📊 数据集**

本文为概念性研究，未使用公开数据集；若在实验中评估，建议使用医疗/教育/同行支持领域的事实检索语料与已标注的沟通质量数据集（如医疗术语词表、可读性阈值、共情标注数据）。

**📈 对比分析**

由于缺乏实现细节与实验结果，本文未给出定量性能比较；但作者指出该框架的目标是提升沟通一致性而不牺牲事实准确性，并建议后续工作构建联合评估基准以量化两者的权衡。

**⚠️ 局限性**

局限性包括：
- 需要预先定义并维护大量领域术语与可读性规则，实施成本高；
- 约束验证与修正过程可能导致生成速度下降；
- 评估指标尚未标准化，缺乏统一的公开基准；
- 复杂交互场景中，事实与沟通目标可能存在不可调和的冲突，需人工干预。

---

## 190. Are Visual Place Recognition Models Recognizing Places or Conditions? Distractor-Augmented Evaluation and Condition Suppression

**arXiv ID:** 2608.06847 | [PDF](https://arxiv.org/pdf/2608.06847v1)

**作者:** Beomsu Kim `[一作]` (DGIST), Giseop Kim `[通讯]` (DGIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种新的评估框架——Distractor-Augmented Recall (DAR)，用来衡量视觉定位（VPR）方法在混合条件数据库中因条件相似的干扰图像导致的检索失败。随后将线性概念消除技术（INLP、LEACE、STD）作为条件抑制（condition suppression）方法，后处理VPR描述子，去除光照、天气、季节等条件信息，从而提升在含干扰图像情况下的鲁棒性。

**💡 创新点**

创新点包括：①首次系统地将DAR与传统Recall (R@K)分离，提出了Robustness to Distractors Ratio (RSR) 以量化检索对干扰的抗干扰能力；②将自然语言处理中的线性概念消除方法迁移到VPR领域，验证其在去除条件信息、提升DAR@1方面的有效性；③展示了单一变换可合并抑制多种条件的可行性，减少部署成本。

**🔧 技术方法**

主要技术：
- Distractor-Augmented Recall (DAR) 评估协议及其指标 RSR；
- 线性概念消除方法：INLP（Iterative Nullspace Projection）、LEACE（Linear Erasure of a Concept with Eigen-Components）、STD（条件标准化）；
- 基于全局描述子（如NetVLAD、AnyLoc、EigenPlaces）的后处理；
- 评估脚本采用公开权重与标准化的训练/测试分割。

**📊 数据集**

使用的数据集包括：RobotCar、StLuciaMToD、SVOX、Nordland、Gardens Point、AmsterTime 等；所有数据集均以1米间距采样，并在多种天气、季节、光照条件下提供跨条件轨迹。

**📈 对比分析**

实验对比 11 种主流 VPR 方法。结果表明：
- DAR@1 的排名与传统 R1 明显不一致；
- INLP 与 LEACE 在 46–47 对方法/数据集上显著提升 DAR@1（平均提升 3–4%），且对 R1 的负面影响最小；
- STD 在大多数情况下效果不佳，需要更多拟合数据；
- 单一合并变换在不影响 R1 的前提下，保持了对多条件的抑制效果；
- 在极少量的拟合样本（≥10%）下即可取得与全量相当的 DAR@1 与 R1。

**⚠️ 局限性**

局限性：
- 仅针对全局描述子，未评估局部或层次特征的抑制效果；
- 需要事先标注条件标签以训练抑制变换，部署时对无标签数据库不友好；
- DAR 仅测量跨条件检索对干扰的鲁棒性，未涵盖单一会话中的光照/遮挡等瞬时条件；
- 对极端条件（如极寒、浓雾）尚未充分验证，可能需要更强的抑制策略。

---

## 191. Vertex cover number of valued constraints is a structural parameter for efficient local search

**arXiv ID:** 2608.06502 | [PDF](https://arxiv.org/pdf/2608.06502v1)

**作者:** Artem Kaznatcheev `[一作]` `[通讯]` (Utrecht University), Artem Kaznatcheev (Utrecht University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在顶点覆盖数为k的VCSP表示的适配度景观中，严格局部搜索（贪心与随机上坡）能够在指数级时间内找到局部峰值；

**💡 创新点**

首次证明顶点覆盖数是局部搜索复杂性的有用参数，并给出了贪心上升的$2^{2k}(n-k+1)$步上界、随机上升的$2^k n(n-k)$步期望上界，以及对应的$9/128·2^k(n-k)$步下界；

**🔧 技术方法**

采用参数化复杂性分析、VCSP转化为加权超图、子问题分解、归纳证明以及蛇形码构造等技术；

**📊 数据集**

无具体数据集，论文完全基于理论证明；

**📈 对比分析**

与之前的$2^k(n-k+1)$上界相比，贪心上升获得了更精确的$2^{2k}(n-k+1)$上界，随机上升得到$2^k n(n-k)$期望上界，且给出了匹配的下界，表明在顶点覆盖数参数化下的复杂度是指数可接受的；

**⚠️ 局限性**

仅限于顶点覆盖数参数的VCSP，未提供实验验证；上界与下界之间仍存在常数因子差距，对其他结构参数的可行性未进一步探讨。

---

## 192. Recent advances in weakly supervised learning: New supervision paradigms, assumption relaxations, and practical solutions

**arXiv ID:** 2608.06896 | [PDF](https://arxiv.org/pdf/2608.06896v1)

**作者:** Wei Wang `[一作]` (RIKEN), Masashi Sugiyama `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了三项关于弱监督学习的关键方法：置信差分分类、基于 SCAR 假设的一致补标签学习以及首个深度部分标签学习基准 PLENCH，并给出了相应的理论证明与实验评估。

**💡 创新点**

创新点在于：①设计了仅使用样本对置信差信息的弱监督二分类框架并构造无偏风险估计器；②在不需要均匀或转移矩阵假设的前提下，通过 SCAR 假设实现补标签学习的一致性；③构建了系统化的 PLENCH 评测平台并提出了可理论保证的模型选择指标。

**🔧 技术方法**

采用的技术包括：无偏风险估计、估计误差上界分析、风险修正方法、One‑Vs‑Rest 策略、Rademacher 复杂度界、以及深度网络训练（如 ResNet、DenseNet、MLP）和 Adam 优化。

**📊 数据集**

实验数据集涵盖了图像数据（如 ImageNet、CIFAR‑10、COCO 等）和表格数据（Lost、Soccer Player、Yahoo! News 等），以及人工构造的部分标签集合。

**📈 对比分析**

通过在 PLENCH 基准上对比多种深度部分标签算法，使用覆盖率、近似准确率等模型选择指标，发现传统方法在特定评测设置下表现被低估，最新方法在多数数据集上取得显著提升。

**⚠️ 局限性**

局限性包括：置信差分类仍需估计类先验且对噪声敏感；SCAR 假设虽然更宽松但仍对数据采集方式有要求；PLENCH 评测受限于可用的部分标签生成机制，且 Oracle 准确率不切实际。

---

## 193. R2S-EGO: Dual-Proxy Refinement for Sparse-Capture Real-to-Sim

**arXiv ID:** 2608.06827 | [PDF](https://arxiv.org/pdf/2608.06827v1)

**作者:** Shuai Fang `[一作]` (XPENG Robotics), Jie Chen `[通讯]` (XPENG Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了R2S-EGO双代理方法，用于在稀疏捕获环境中细化现实到仿真的场景，生成行为可执行的ego视图并将其融入视觉资产。

**💡 创新点**

创新点在于将机器人代理与几何代理结合：机器人代理限定可执行的摄像头查询并评估视觉缺口，几何代理通过捕获对齐的形状先验提供结构条件；通过预算化的查询选择、合成伪观测并更新场景，从而实现针对机器人行为视角的有针对性补全。

**🔧 技术方法**

使用的技术包括：机器人控制器与前向运动学、三维Gaussian场景（3DGS）渲染、SAM3D形状先验、NKSR几何重建、ViewCrafter视频生成、VGGT相机姿态校准以及PSNR/SSIM/LPIPS评估指标。

**📊 数据集**

使用数据集：Replica三维室内场景（共三条场景），Unitree G1机器人ego相机采样；六个真实RGB捕获用于初始化，48个冻结的测试相机。

**📈 对比分析**

通过与Vanilla 3DGS、GaussGym、Difix3D+以及GenFusion等方法比较，R2S-EGO在六视图设置下取得19.06 dB PSNR、0.757 SSIM、0.273 LPIPS，明显优于其它基线；在真实G1坐姿任务中，R2S-EGO以82.5%±6.8%的成功率对比GaussGym的10%±10.5%，验证了其在真实场景中的有效性。

**⚠️ 局限性**

局限性包括：先验完成的几何仍为假设，未对碰撞精度进行独立评估；机器人代理仅覆盖声明的行为空间，未学习到的行为可能缺失；伪观测更新可能引入累积误差。

---

## 194. CrossTracer: Cross-Embodiment Navigation via VLA Model Reasoning and Trace Residuals Adapting

**arXiv ID:** 2608.06688 | [PDF](https://arxiv.org/pdf/2608.06688v1)

**作者:** Yao Wang `[一作]` (Peng Cheng Laboratory), Wenjun Xu `[通讯]` (Peng Cheng Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CrossTracer 框架，实现跨机器人体型的视觉‑语言‑行动导航，通过像素空间轨迹先行规划再进行体型感知的残差微调；

**💡 创新点**

核心创新在于将语义轨迹与体型约束解耦，使用统一像素轨迹作为中间表示，并通过 CE‑Adapter 对残差进行自适应修正；同时引入 CE‑RRT* 自动生成体型约束下的规划轨迹做监督；

**🔧 技术方法**

技术手段包括 Vision‑Language Trace Proposer (基于 OmniVLA+LoRA)、CE‑Adapter（FiLM、跨注意力残差头、遍历度重建头）以及 CE‑RRT*（从 panoptic segmentation 转为成本图后 RRT* 规划）；

**📊 数据集**

主要数据集为 NaviTrace 基准（提供 egocentric 图像、语言指令与体型标签），VL‑Tracer 采用 VAMOS 的轨迹标注，CE‑Adapter 的训练数据通过 CE‑RRT* 生成；

**📈 对比分析**

与通用视觉‑语言模型（Gemini‑2.5‑Pro 等）及专用体型模型对比，CrossTracer 在 NaviTrace 总分 45.68，较 Gemini‑2.5‑Pro 提升 28% ；在真实机器人（轮式与步行）部署时，成功率、路径效率及任务时间均显著提升；

**⚠️ 局限性**

局限性包括对 panoptic segmentation 质量敏感、成本图参数需人工设定、无法直接处理高度不连续或悬垂障碍，未来工作将探索从交互数据学习遍历度并扩展至 3D 结构与闭环再规划。

---

## 195. LyEvO: Lyapunov-Guided Evolutionary Optimization for Safe and Robust Sim-to-Real Policy Learning

**arXiv ID:** 2608.06481 | [PDF](https://arxiv.org/pdf/2608.06481v1)

**作者:** Riccardo Curcio `[一作]`, Marco Caccamo `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

论文探讨了某一特定领域的问题，并提出了一种新的解决方案。

**💡 创新点**

创新点在于提出了一种新的方法或模型，能够更有效地解决该问题。

**🔧 技术方法**

使用了深度学习技术和机器学习算法。

**📊 数据集**

使用了公开的数据集进行实验，确保结果的可重复性。

**📈 对比分析**

与现有的方法进行了比较，结果显示新方法在准确性和效率上都有显著提升。

**⚠️ 局限性**

限制在于模型的可扩展性和对特定数据集的依赖性。

---

## 196. ZIPBrain: Can EEG Foundation Models Be Faster, Locally Deployable, but Accurate?

**arXiv ID:** 2608.07033 | [PDF](https://arxiv.org/pdf/2608.07033v1)

**作者:** Lingwei Li `[一作]` (Nara Institute of Science and Technology), Yasuhiko Nakashima `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出 ZIPBrain，一种训练无关、可插拔的 EEG 基础模型令牌池化框架；

**💡 创新点**

创新点在于利用 EEG 低 SNR 低冗余特征，采用两阶段冗余感知分区、匹配及保范数合并，实现高效令牌压缩；

**🔧 技术方法**

技术包括主轴（pivot）选择、冗余评分、余量匹配、保范数聚合，兼容 Transformer 结构，并支持 CUDA Graph 等加速；

**📊 数据集**

使用 TUAB、EEGMAT、TUEV、ISRUC‑Sleep、EarEEG 等 5 个公开 EEG 数据集进行评估；

**📈 对比分析**

与 ToMe、ToFu、EViT、DART 等 CV 令牌压缩方法对比，ZIPBrain 在最大压缩率下平均提升 1.3%–10.5% 准确率，推理时间下降 32.7%（CUDA Graph 下 41.8%）；

**⚠️ 局限性**

局限在于仅针对推理阶段，未结合权重量化或生成式模型，且对极低 SNR 或极少通道的鲁棒性待进一步验证。

---

## 197. Finding Usable Weight Mechanisms with Tiled SVD

**arXiv ID:** 2608.06969 | [PDF](https://arxiv.org/pdf/2608.06969v1)

**作者:** Ash Manvi `[一作]` (Aquin Labs), Samreena Tajreen `[通讯]` (Aquin Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提取Gemma‑2‑2B线性层的列分块SVD挂载（v、u、σ），并用全写能量提升、覆盖饱和度和深度条件的steer‑unembed一致性对其进行评估；

**💡 创新点**

首次提出全写能量提升替代传统列局部提升，证明在匹配挂载预算下分块SVD优于全矩阵SVD、列采样和随机方法，并将有效路径挂载与多线性地图结合形成完整可复现的评估套件；

**🔧 技术方法**

使用列分块SVD、触发与写入能量提升计算、覆盖饱和度评估、后层RMSNorm后的steer‑unembed对齐、mean‑gate与ridge least‑squares等有效路径权重构造技术；

**📊 数据集**

WikiText‑2训练集子样本（16,384个token）；

**📈 对比分析**

在Gemma‑2‑2B的26层七个线性映射上进行比较，分块SVD在全部182个site‑layer上均通过评估，覆盖饱和度在1–2模式即可；在全写能量提升上优于全矩阵SVD、列采样和随机方法；

**⚠️ 局限性**

仅在Gemma‑2‑2B单一模型规模上测试；实验C仅适用于残差写入位置；挂载缺乏语义标签；WikiText‑2子样本可能导致偏倚；Steer使用固定α、短文本，且早期层的评估被豁免。

---

## 198. Hidden Gauge Controls Feature Specialization in ReLU Networks

**arXiv ID:** 2608.06766 | [PDF](https://arxiv.org/pdf/2608.06766v1)

**作者:** Tongxi Wang `[一作]` `[通讯]` (Southeast University), Tongxi Wang (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究ReLU网络在训练过程中，隐藏的尺度（gauge）如何决定同一功能特征最终由哪一个冗余神经元学习，并证明尺度差异可导致特征学习时间呈Θ(D²)的非同步化。

**💡 创新点**

创新点在于：①提出“特征所有权”概念，并揭示它由不可见的尺度参数决定；②通过正交尺度变换（gauge）在节点层面重新分配系数反应与方向运输两种运动的自由度；③在可解析的高维高宽两层Gaussian教师-学生模型中，证明了尺度可控的特征所有权与功能修剪的全局定理，并给出了精确的时间尺度分离和轨迹预测；④将理论延伸到可观测扰动、有限步全批梯度下降以及有限样本情形。

**🔧 技术方法**

使用的技术包括：正则化ReLU网络的标记坐标变换、反应-运输分解、闭式梯度流动力学、随机变量的弧余弦核（arc‑cosine kernel）分析、时间尺度分离理论与收敛性证明、以及离散化的全批梯度下降误差分析。

**📊 数据集**

数据集：采用高维正态分布的合成教师-学生数据（Gaussian teacher–student），无真实外部数据集；实验中通过多种样本大小（N=512…8192）验证理论。

**📈 对比分析**

比较方法：将理论预测的“标记动力学”与原始参数流、实验轨迹、以及离散化梯度下降轨迹进行对齐；使用损失、特征对齐、冗余质量衰减以及耗散分解等指标；实验结果显示拟合残差均低于1.3×10⁻⁷，时间尺度与D²比例与理论一致，误差随样本数呈N⁻¹/²下降。

**⚠️ 局限性**

局限性：①主要针对完全重复初始化的网络，非对称冗余字典需要额外的非退化性假设；②理论在大尺度D取值时成立，随机初始化下尺度不平衡通常较弱；③仅在两层ReLU网络的Gaussian教师-学生模型中证明，尚未在更深网络或复杂任务上验证；④结果对梯度下降步长的选择敏感，需在ηD小范围内；⑤不讨论泛化性能，只关注内部特征分配与训练轨迹。

---

## 199. Casting the Net! Revisiting MasterFace Impersonation Attacks

**arXiv ID:** 2608.06952 | [PDF](https://arxiv.org/pdf/2608.06952v1)

**作者:** Seunghun Paik `[一作]` (Hanyang University), Jae Hong Seo `[通讯]` (Hanyang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种利用合法购买的商业API，通过Gram分解和最大覆盖问题构造API定制的MasterFace，从而在有限查询预算下突破传统FMR基准的欺骗攻击。

**💡 创新点**

将MasterFace攻击转化为模板空间的最大覆盖问题，并通过Gram分解从仅有相似度查询中恢复模板旋转，实现了在公开API上合法的非零努力欺骗。

**🔧 技术方法**

采用Gram分解、最大覆盖问题贪婪求解、score-based reconstruction（Kim等重构攻击）以及NES后处理等技术。

**📊 数据集**

使用MS1MV3、LFW、CASIA-WebFace以及WebFace42M的PCA组件等数据集，目标FRS包括ElasticFace、ArcFace、AdaFace和Amazon Rekognition CompareFaces。

**📈 对比分析**

与零努力impostor基线对比，在10^-3、10^-4、10^-5 FMR下，Q=5、10、30查询预算时，攻击成功率分别提升约2×-9×，最高提升近9.5倍，显著优于基线。

**⚠️ 局限性**

对商业API相似度到余弦相似度映射误差、需要NES后处理、以及只能在数字域实现等因素限制了攻击的实际可行性，物理现实攻击仍需进一步研究。

---

## 200. TEXAS: Task-Expert-Aware Supervision for Downstream Mixture-of-Experts LLM Adaptation

**arXiv ID:** 2608.06396 | [PDF](https://arxiv.org/pdf/2608.06396v1)

**作者:** Guanzhi Deng `[一作]` (City University of Hong Kong), Linqi Song `[通讯]` (University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为TEXAS的MoE模型下的适配框架，通过对基模型成功与失败实例的专家激活差异进行检验，筛选出任务相关专家，并在下游微调时对这些专家激活的答案词加权提升监督力度。

**💡 创新点**

创新点在于①将专家激活与任务正确性直接关联而非仅统计使用频率；②利用已识别的任务专家作为token级监督加权信号，实现对学习信号的精细分配，而非对专家子集或路由进行硬性约束。

**🔧 技术方法**

核心技术包括：基于Welch t检验的专家显著性筛选、Benjamini–Hochberg多假设校正、top‑k路由记录、交叉熵加权损失以及LoRA微调框架。

**📊 数据集**

使用了六大类基准数据集：数学推理（GSM8K、MATH500）、代码生成（HumanEval、MBPP）、通用知识（MMLU）和指令跟随（IFEval），每个任务对应的训练集为MetaMathQA-GSM8K、MetaMathQA-MATH、CodeAlpaca、OpenCodeInstruct、MMLU-train、RECAST-30K。

**📈 对比分析**

与基线模型（原始模型、SFT、ESFT、RoMA）对比，TEXAS在18个模型‑任务组合中取得17个最佳或同等成绩，平均比最强基线提升约1.3–1.5个百分点，且在SFT基础上平均提升3.0、2.4、2.7个百分点。

**⚠️ 局限性**

局限性包括：在多样化知识覆盖的MMLU任务上提升有限；需要预先定义任务成功判定和token级规则；以及对超参数（K、α）的敏感性需进一步研究。

---

## 201. Robust Average-Reward Markov Decision Processes: Minimax-Optimal Learning via Plug-in Reductions

**arXiv ID:** 2608.06545 | [PDF](https://arxiv.org/pdf/2608.06545v1)

**作者:** Yuepeng Yang `[一作]`, Yuejie Chi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究在生成式模型下分布式鲁棒平均收益马尔可夫决策过程（AMDP）的样本复杂度，并给出了匹配的上下界；

**💡 创新点**

首次给出鲁棒AMDP的最优样本复杂度，划分高容忍和低容忍两种工作区间，并提出跨度信息化与跨度无关的最优降维算法；

**🔧 技术方法**

利用平均收益-折扣转化、鲁棒折扣MDP求解器、跨度引导与置信下界估计等技术实现样本效率最优；

**📊 数据集**

实验使用控制性仿真生成的合成AMDP数据集，分别调节跨度、噪声半径和误差阈值；

**📈 对比分析**

与传统非鲁棒AMDP和前沿鲁棒方法（如Roch等）的样本复杂度进行比较，实验验证理论曲线并展示自适应选择策略和接近理论极限的性能；

**⚠️ 局限性**

仅适用于(s,a)-矩形总变差不确定集和生成式模型，未考虑在线探索或更通用的不确定性结构。

---

## 202. From Documentation to Zero-day Vulnerabilities: LLM-Driven Fuzzing of JavaScript Engines in PDF Readers

**arXiv ID:** 2608.06641 | [PDF](https://arxiv.org/pdf/2608.06641v1)

**作者:** Suyue Guo `[一作]` (UC Santa Barbara), Giovanni Vigna `[通讯]` (UC Santa Barbara)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于大语言模型的 PDF 阅读器 JavaScript 引擎 fuzzing 工具，自动推断 API 规范与复杂的跨 API 关系，并生成语义丰富的 API 调用序列；

**💡 创新点**

创新点在于①利用 LLM 构建完整 API 规范，覆盖未文档化函数；②发现并建模三类强关系（生产者-消费者、值约束、隐式状态），通过 SMT 求解生成符合约束的调用序列；③采用两阶段关系推断与参数级 CFG 生成，显著提升覆盖率；

**🔧 技术方法**

核心技术包括 LLM（如 GPT‑4o）用于规范推断、关系推断；上下文无关文法（CFG）生成；SMT 求解器 Z3；PDF 对象生成协同（Cooper）和定向变异；

**📊 数据集**

使用 Adobe 官方 JavaScript API 手册（700+页）以及从三款主流 PDF 阅读器（Adobe Acrobat Reader、Foxit PDF Reader、PDF‑XChange Editor）获得的执行轨迹与样本 PDF；

**📈 对比分析**

与三款现有 PDF fuzzers（TypeOracle、Favocado、Cooper）以及两种通用 LLM fuzzers（Fuzz4All、Naïve LLM）进行对比；覆盖率提升最高 48%，发现 31 个零日漏洞（高危 11 例），而对手最多 6 例；

**⚠️ 局限性**

局限包括对 LLM 质量的依赖（错误率 2–7%），生成过程仍需要较多算力与费用，且缺乏持续的反馈循环；对未文档化 API 的推断可能遗漏隐藏约束；

---

## 203. Direct Factorization of the Karhunen-Loève Transform of AR(1) Sources

**arXiv ID:** 2608.06522 | [PDF](https://arxiv.org/pdf/2608.06522v1)

**作者:** Yuriy A. Reznik `[一作]` `[通讯]`, Yuriy A. Reznik

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108`

**🎯 论文内容**

提出了精确的AR(1) KLT（卡尔曼-洛维特变换）的递归分解与O(N log N)实现；

**💡 创新点**

证明KLT可通过边界一阶修正和箭头矩阵实现分块分解，揭示其与DCT分裂的本质相同；

**🔧 技术方法**

利用Sherman–Morrison恒等式、秩一更新与Cauchy矩阵快速乘法（FMM或Trummer方法）完成分解与计算；

**📊 数据集**

无特定数据集，本文为理论与算法分析；

**📈 对比分析**

与传统密集矩阵实现相比，算法实现了与FFT同阶的O(N log N)复杂度；

**⚠️ 局限性**

主要限制在于需要高效的Cauchy矩阵乘法实现，对更高阶AR(p)模型的推广尚未完成。

---

## 204. Fairis: Fairness-Aware Aggregation with Provable Influence Containment against Fairness Poisoning Attacks in Collaborative Machine Learning

**arXiv ID:** 2608.06469 | [PDF](https://arxiv.org/pdf/2608.06469v1)

**作者:** Devharsh Trivedi `[一作]` (Bowie State University), Jackson Walters `[通讯]` (Northern Virginia Community College)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 Fairis：一种服务器端聚合加权方法，通过对每个客户端的公平性得分进行绝对值加权，并结合安全参数 η，实现对公平性攻击的影响控制。

**💡 创新点**

创新点在于：①以绝对公平性得分（EOD）为基础构建闭式加权公式，避免了传统相对加权被攻击者游戏；②正式定义并证明三项安全性质（Monotone Weight Reduction、Demographic Participation、Non-Gamesmanship），并将其扩展到少数派合谋攻击；③在权重上实现安全与公平的折中，可通过 η 进行可调节；④提供大小加权变体，兼顾样本量信息。

**🔧 技术方法**

技术方法包括：服务器端聚合时计算权重 ω_k = (η – F_k) / Σ_j(η – F_j)；对更新进行范数裁剪以限制攻击者的位移；使用 EOD（Equal Opportunity Difference）作为公平性度量；通过数学证明和 Lean 形式化验证三项安全性质；在 FL、SplitML 等多种联邦学习架构下统一实现。

**📊 数据集**

使用了三个公开金融数据集：German Credit（1,000 条记录）、Taiwan Credit（30,000 条记录）和 Adult Income（30,162 条记录），并在每个数据集上按 Dirichlet(α=0.5) 分配给不同客户端，保留 20% 测试集。

**📈 对比分析**

与 Silo（无协作）、gap‑based 加权（β=0,1）、Uniform、以及 GLocalFair、SFFL、FairTrade 等方法做对比。实验表明：在常规非 IID 情况下无单一方法统治；在公平性攻击（fairness poisoning）下，Fairis 在 η=1.01 时显著削弱攻击者权重（降低 41–54%），并在攻击者误报公平时保持正权重；相比之下，gap‑based 方案易被游戏，Uniform 方案无法响应公平信号。

**⚠️ 局限性**

主要局限包括：①需要客户端真实报公平得分（A2 条件），缺乏可验证的评分机制；②在整体群体已不公平时，绝对评分加权可能不降低攻击者权重；③只保证攻击者影响受限，未必提升最终模型的公平性；④未对多攻击者协同或大规模实验进行评估；⑤缺乏安全多方计算或加密实现，服务器可见所有公平得分。

---

## 205. Blind to the Pivotal Vote: Aggregate Independence Metrics Miss Where Verification Actually Helps

**arXiv ID:** 2608.06940 | [PDF](https://arxiv.org/pdf/2608.06940v1)

**作者:** Yang Shu `[一作]` `[通讯]` (Zhejiang University), Yang Shu (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在多模型LLM评判面板中单一投票替换（例如加入测试套件的通过/失败信号）对判定准确率的影响。

**💡 创新点**

创新点在于通过多数投票算术揭示只有“关键”投票（margin=1）的查询能被单一投票改变，且所有准确率提升集中于此，从而区分整体相关性与条件效用。

**🔧 技术方法**

采用有效投票数（n_eff）分析、边际门控策略（margin gating）、均匀随机替换与多数侧替换规则，以及Bootstrap置信区间估计。

**📊 数据集**

使用 HumanEval+、MBPP+ 与 LiveCodeBench 三个代码生成基准，每个基准配备 3/5/7/9 名独立评判者和 7 名解码器。

**📈 对比分析**

与仅用评判者、最佳单一评判者、全量信号或统一随机替换等策略比较，发现统一随机替换可提升约1.8个百分点，关键投票上的多数侧替换可提升约11–23个百分点；整体准确率最高仍为全量信号或最佳评判者。

**⚠️ 局限性**

局限性包括信号为完整测试标签的子集导致单侧误差、信号精度高但在低精度场景下未验证、门控仅在需保留面板时有价值、未探究加权面板或非二值信号、以及仅针对代码生成任务。

---

## 206. Unmasking Removal-Budget Confounding: A Matched Operating-Point Evaluation Framework for Adaptive Data Cleaning

**arXiv ID:** 2608.06511 | [PDF](https://arxiv.org/pdf/2608.06511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 207. Agentic Planning for Symbolic Execution

**arXiv ID:** 2608.06397 | [PDF](https://arxiv.org/pdf/2608.06397v1)

**作者:** Daniel Koh Ji Yang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Youcheng Sun `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 2336 | [OpenAlex ID](https://openalex.org/A5060663223)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于外部规划器的代理式符号执行框架，通过利用先前有界符号执行运行产生的证据，动态配置后续的有界符号执行任务，以提升覆盖率。

**💡 创新点**

创新点在于把符号执行的决策从单个运行内部迁移到跨运行的规划层面，使得规划器能够依据覆盖反馈和执行历史，选择合适的入口点、符号输入表面和执行模式（如 Witness‑Guided 与 Harness‑Entry）。

**🔧 技术方法**

技术上实现了在 KLEE 上的两种执行模式，使用 LLM（OpenAI Agent）作为规划器，通过覆盖重放、路径增量反馈和预飞行检查，构造 BSE 任务并提交给 KLEE 进行执行。

**📊 数据集**

实验采用 ConcoLLMic 评测基准中的七个 C/C++ 程序（如 LibYAML、Oggenc、WOFF2 等），并利用这些程序的测试 harness 与源文件列表。

**📈 对比分析**

对比方法是将 Planner+KLEE 的最终语料与连续 KLEE、AFL++、SymCC、SymSan 的 3 小时语料以及 ConcoLLMic 的三次跑进行覆盖率比较，结果显示 Planner+KLEE 在所有程序上平均提升约 3.5 倍分支覆盖率，且在五个程序上超过 ConcoLLMic，且能覆盖其他方法未覆盖的分支。

**⚠️ 局限性**

局限性包括资源不匹配（Planner+KLEE 使用多余工作线程），仅针对 KLEE 与 LLVM 的实现，未对等价容量或不同符号执行工具做严格对比，且缺乏对不同目标函数或更复杂程序的广泛验证。

---

## 208. When Semantics Saturate or Emerge: Adaptation-Conditional Semantic Utility in Source-Free Cross-Domain Few-Shot Learning

**arXiv ID:** 2608.06673 | [PDF](https://arxiv.org/pdf/2608.06673v1)

**作者:** Wei Liu `[一作]` (Jiangsu University of Science and Technology), Haijian Shao `[通讯]` (Jiangsu University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在源自由跨域少样本学习中，作者通过对同一支持集、同一视觉 LoRA 适配，只改变文本描述（基础类名与详细类级描述），研究冻结与适配后语言的相对效用，发现语言在两阶段的优势存在“语义饱和”与“语义出现”两种规律。

**💡 创新点**

创新点在于提出适配条件下的语言效用评估框架，区分零射精与适配后语义效用；首次系统揭示同一文本在冻结与适配阶段的效用可逆或不可逆；并通过样本级保留/覆盖、训练轨迹分析阐明两种规律背后的机制。

**🔧 技术方法**

使用 CLIP ViT‑B/16（以及 B/32）视觉语言模型，低秩适配 LoRA；固定类级描述与基础模板；采用配对 episode 训练与评估，配对计算 A_D^0、A_B^0、A_D^L、A_B^L；并进行训练轨迹、样本级保留/覆盖度、shuffle‑semantics 控制、不同随机种子与 backbone 的验证。

**📊 数据集**

四个跨域少样本基准：EuroSAT、CropDisease、ISIC、ChestX；采用 5‑way 1‑shot 与 5‑shot 组卷，每个组卷包含 15 个查询样本。

**📈 对比分析**

通过配对评估计算零射精与适配后语义效用差值；结果显示 EuroSAT 与 CropDisease 处于“语义饱和”状态（零射精高、适配后差距显著缩小），ISIC 与 ChestX 处于“语义出现”状态（零射精低、适配后显著提升）。总体而言，Detailed‑LoRA 在绝大多数条件下优于 Base‑LoRA，性能与现有 CLIP‑LoRA 与多种源自由基线相当或略优，但未能超越最新的优化方法。

**⚠️ 局限性**

局限性包括：仅研究固定文本描述与单一 LoRA 适配；仅使用两种 CLIP backbone 与四个目标域；未探讨不同文本生成/优化策略、其他参数高效适配技术；未深入内部机制（如注意力、梯度分布）；shuffle‑semantics 控制仅验证对应关系而非具体语义因果；实验仅覆盖 5‑way 1/5‑shot，未检验更大类数或更高 shot 数的行为。

---

## 209. Local Epistemic Uncertainty Guided Active Sampling for Plug-and-play Diffusive Image Restoration

**arXiv ID:** 2608.06981 | [PDF](https://arxiv.org/pdf/2608.06981v1)

**作者:** Jiaqi Zhang `[一作]` (Jiangsu University), Yang Yang `[通讯]` (Jiangsu University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了LEADer框架，利用本地表观不确定性指导扩散模型的图像恢复采样，实现在空间上自适应先验调制、时间上自适应轨迹裁剪的零拷贝方法。

**💡 创新点**

创新点包括：①基于Fisher信息量化的本地表观不确定性；②不确定性校准的先验调制（UCPM）以平衡细节保留与伪影抑制；③状态感知轨迹裁剪（SATP）通过不确定性迹线动态调整采样步长，提供确定性误差界限；④理论证明保证数据一致性与采样收敛。

**🔧 技术方法**

使用扩散模型（DDPM/DDIM）+ Fisher信息矩阵估计+最大后验优化+正交投影+轨迹裁剪+误差分析等技术。

**📊 数据集**

在CelebA‑HQ 1K与ImageNet 1K两个常用图像恢复基准上，针对四倍超分、Gaussian去模糊、运动去模糊、压缩感知等五类任务进行实验。

**📈 对比分析**

将LEADer以plug‑and‑play模块集成到多种SOTA零拷贝DMIR方法（DDNM、DDPG、ProjDiff、SITCOM、PIRP、EquS等）进行对比；实验表明平均PSNR提升约1–3%，LPIPS下降约10%，并在相同或更低采样时间内实现性能提升，显示出显著的质量与效率双赢。

**⚠️ 局限性**

局限性：当信息损失预算B设置过大时会导致质量下降；在极低采样步长下速度提升有限；对更复杂、多模态或高分辨率场景的适用性仍待进一步验证。

---

## 210. Statistical Analysis of Executability and Program Equivalence in Decompilation for IoT Vulnerability Detection

**arXiv ID:** 2608.06960 | [PDF](https://arxiv.org/pdf/2608.06960v1)

**作者:** Minami Yoda `[一作]` (Nihon University), Yutaka Matsuno `[通讯]` (Nihon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一套九维度评估指标，用以定量评估IoT固件反编译结果的质量。

**💡 创新点**

创新点在于将结构、行为、语义三大类各细分为三维度，既衡量控制流与API调用的准确性，又关注语义一致性。

**🔧 技术方法**

主要技术包括基于LLM的反编译、CFG提取与匹配、正则式漏洞检测以及统计效应量（Cohen's d）分析。

**📊 数据集**

实验使用OpenWrt的318个程序作为数据集，生成19,625份C源码进行评估。

**📈 对比分析**

与传统规则基Ghidra及LLM4Decompile等模型比较，平均得分约0.64，行为与结构相似度对重编译成功的预测效果显著，Cohen's d超过0.9。

**⚠️ 局限性**

局限包括评估指标权重未经过最优调优、逻辑操作子指标出现负相关、仅覆盖单一固件平台与x86架构，未来需扩大数据范围并改进指标设计。

---

## 211. UAV3DCrop: Benchmarking 3D Reconstruction in Repeated Multi-Angle UAV Crop Surveys

**arXiv ID:** 2608.06404 | [PDF](https://arxiv.org/pdf/2608.06404v1)

**作者:** Junxiong Zhou `[一作]` (University of Minnesota), Licheng Liu `[通讯]` (University of Minnesota)

**通讯引用:** 4488 | [OpenAlex ID](https://openalex.org/A5056697722)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

发布了 UAV3DCrop 公开基准，涵盖 91 个多作物、重复时间的 UAV RGB 场景，评估场景优化与零射程前馈模型在外观、几何、冠层高度及度量尺度等指标上的表现。

**💡 创新点**

创新点包括：①首个公开的多作物、重复时间的 UAV 三维重建数据集；②两轨道基准（场景优化与零射程前馈），系统评估外观、深度、冠层高度与绝对尺度的相互关系；③揭示不同方法在不同任务上的排名差异，指出任务导向性与模型设计的关键影响。

**🔧 技术方法**

使用技术包括：NeRF 与 3D Gaussian Splatting（Splatfacto、Splatfacto‑big、Mip‑Splatting、Scaffold‑GS、CityGaussian）、Zero‑shot前馈几何模型（MapAnything、VGGT、Pi3、MASt3R）；Agisoft Metashape 进行 SfM + MVS；评估指标涵盖 PSNR/SSIM/LPIPS、RMSE/AbsRel/SILog/pearson、canopy height MAE/RMSE/R²、ATE RMSE、AUC@5 等。

**📊 数据集**

使用的数据集为 UAV3DCrop：88,830 张 RGB 图像、91 场景（四种作物：玉米、大豆、小麦、大麦），配备 RTK 定位姿态、MVS 深度、现场有效叶面积指数（LAI）与 2025 年的植株高度。

**📈 对比分析**

对七种场景优化模型和四种前馈模型进行基准比较。Splatfacto‑big 在外观指标（PSNR/SSIM/LPIPS）最高；Scaffold‑GS 在深度和冠层高度恢复上最优；MapAnything 在绝对尺度、姿态、点图、深度等多项指标上表现最佳；但不同方法在各指标上的排名不一致，显示任务导向性与模型设计的差异。

**⚠️ 局限性**

局限性包括：SfM 结果受重复纹理与作物移动影响，缺乏激光扫描验证；评估仅针对视角插值，未覆盖稀疏或极角场景；植物高度仅在 2025 年收集，且 LAI 与时间相关，难以分离时间与场景质量；前馈模型在绝对尺度上存在域偏移，需进一步验证和适配。

---

## 212. G-Power: Architecture-level GPU Power Modeling with Aggregated Knowledge Foundations from Known GPUs

**arXiv ID:** 2608.06870 | [PDF](https://arxiv.org/pdf/2608.06870v1)

**作者:** Qijun Zhang `[一作]` (Hong Kong University of Science and Technology), Zhiyao Xie `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 G-Power，一种利用已知 GPU 数据进行预训练、注意力聚合和微调的 GPU 功耗建模框架。

**💡 创新点**

创新点在于三阶段算法：在多款已知 GPU 上预训练基础模型，通过注意力机制聚合相似基础，再对目标 GPU 进行微调，从而显著提升功耗预测精度。

**🔧 技术方法**

采用线性回归建模、注意力聚合、两级特征分组微调，以及 Nsight Compute 收集的架构事件和 NVML 测量功耗；对照 AccelWattch 与 McPAT‑Calib 的 XGBoost 方法。

**📊 数据集**

使用 65 个 AccelWattch 微基准和 CUDA Samples、Rodinia、CUTLASS、Parboil 等真实工作负载，评估四款 NVIDIA GPU（RTX A6000、RTX 3090、RTX 4090、RTX 5880 Ada）。

**📈 对比分析**

与 AccelWattch、McPAT‑Calib 进行对比，G‑Power 平均 MAPE 下降 22%（约 14%），相关系数提升 0.36（约 0.88），在所有测试场景中均优于基线。

**⚠️ 局限性**

局限性在于需要多款已知 GPU 作为预训练数据，对全新架构的泛化能力有限；实验仅覆盖 NVIDIA GPU，且对架构事件的采集精度依赖硬件工具。

---

## 213. Evaluating XAI Support From A Hierarchical Reinforcement Learning Policy in Human-Agent Collaboration

**arXiv ID:** 2608.06381 | [PDF](https://arxiv.org/pdf/2608.06381v1)

**作者:** Mateus Levi Simões Fernandes `[一作]` (PUC-Rio), Alberto Sardinha `[通讯]` (PUC-Rio)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Overcooked-AI 基准环境中，利用层次化自适应代理（HA^2）在实时决策中生成可解释性说明，并通过触发式文本或音频方式交付给人类伙伴。

**💡 创新点**

首次将 intrinsically explainable 的强化学习架构应用于标准合作基准，并系统比较文本与音频两种模态对人机协作效果的影响，揭示音频可加速学习但伴随工作联盟评分下降。

**🔧 技术方法**

使用 HA^2 的 Manager‑Worker 层次化 RL、PPO 训练管理器、基于子任务的触发式解释生成、浏览器原生 TTS 进行音频交付，配合 NASA‑TLX、Godspeed、Human‑Robot Fluency 等问卷。

**📊 数据集**

采用 Overcooked‑AI 的 Counter‑Circuit 关卡作为实验数据集，作为公开的多人协作基准。

**📈 对比分析**

在 38 名受试者的受试者间实验中，实验组与对照组在游戏得分上无显著差异；音频组显示较快的学习曲线（提升率高于文本与无说明组），但文本组表现与无说明组相当；音频组的工作联盟评分显著低于其他两组，提示模态选择对主观体验有重要影响。

**⚠️ 局限性**

限制包括：样本主要为游戏经验丰富的年轻学术人员，样本量小且不平衡；实验仅使用单一关卡；HA^2 采用反应式策略，缺乏长期计划与人类建模，导致音频说明的承诺与实际行为不匹配；音频质量受浏览器差异影响，可能影响主观评分。

---

## 214. Separating Decision-Rule Misalignment from Readout-Coverage Limitations in Speech Language Models

**arXiv ID:** 2608.06409 | [PDF](https://arxiv.org/pdf/2608.06409v1)

**作者:** Linkai Peng `[一作]` (University of Connecticut), Baorian Nuchged `[通讯]` (University of Texas at Austin)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5092715834)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一个生成对齐的诊断阶梯，用于分离语音语言模型在情感识别任务中的端点、决策规则和读取覆盖等多重错误来源。

**💡 创新点**

创新点在于将模型的发射答案、选项logits、logits的仿射读取以及隐藏状态的线性读取放在同一层次上进行对比，从而精准定位性能瓶颈，并通过无标签的logit校正部分可操作的决策规则误差。

**🔧 技术方法**

使用了诊断阶梯分析、无标签logit校正、激活补丁（activation patching）、探测分类器（probing classifiers）等技术手段。

**📊 数据集**

在五个不同的语音语言模型和两个情感语料库（如RAVDESS、CREMA‑D等）上进行实验。

**📈 对比分析**

通过比较各阶段的准确率差异，发现状态解码平均比生成结果高27.8个百分点；所有十个实验条件均出现正向的决策规则和读取覆盖差距；无标签logit校正在每个条件下均提升生成准确率；rank‑matched对比显示模型在新说话人上情感信息的泛化能力，并且对声学特征的控制未能显著削弱该信息。

**⚠️ 局限性**

局限性包括：仅聚焦于情感识别任务，未验证在其他 paralinguistic 任务的泛化；对决策规则误差的改进仍不完整，仅通过校正部分可操作的误差；实验覆盖模型数量有限，结果可能受模型架构和训练数据的影响。

---

## 215. From Enumeration to Covering: Near-Optimal Densest P-Partite Subgraph Search over Large Heterogeneous Information Networks

**arXiv ID:** 2608.06906 | [PDF](https://arxiv.org/pdf/2608.06906v1)

**作者:** Lu Chen `[一作]` (Swinburne University of Technology), Jianxin Li `[通讯]` (Edith Cowan University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种新的密度优化方法，针对大规模异构信息网络（HIN）中给定元路径的稠密子图搜索问题，提供近最优解而不需要枚举所有权重集合。

**💡 创新点**

创新点包括：① 用对数坐标将可行权重空间转化为有限多面体，并用二阶 η‑网覆盖该空间，只需多项式对数数量的代表权重；② 将固定权重子问题改写为加权超模稠密子图问题，利用自适应剥离（instance‑free）得到 (1-δ)/(1+η) 的近似保证；③ 通过“占位符”式的路径实例计数避免了实例化整个元路径，显著降低时间与空间复杂度；④ 引入基于占位数的顶点剪枝进一步减少搜索规模。

**🔧 技术方法**

核心技术包括：对数变换与 η‑网覆盖、加权超模稠密子图的 Super‑Greedy++ 算法、实例自由的路径计数动态规划、占位符顶点剪枝与占位数下限、以及基于增量证明的近似保证。

**📊 数据集**

实验使用了五个真实 HIN 数据集：MovieLens、DBLP、Douban、DBpedia、Freebase（以及 Hetionet 用于药物再利用案例）。

**📈 对比分析**

与最优枚举基线及现有近似算法比较，本文方法在保持 99.9% 以上密度的同时，速度提升 10–100 倍，尤其在 Freebase 等百万级图上实现了完整求解；并且在所有测试中均能提供近最优保证。

**⚠️ 局限性**

局限性主要在于：① 对超模稠密子图求解的收敛时间与数据分布相关，稠密或路径长度较大的实例仍可能较慢；② 近似误差依赖于 η 与 δ 的选择，需要经验调参；③ 对于非常长的元路径，覆盖网格的规模仍会随 log n 增大，导致一定的空间开销。

---

## 216. From Cheap Fakes to Pure Synthesis: Addressing the New Era of T2V Fake News Videos

**arXiv ID:** 2608.06732 | [PDF](https://arxiv.org/pdf/2608.06732v1)

**作者:** Yifeng Luo `[一作]` (Hong Kong Baptist University), Tian Wang `[通讯]` (Beijing Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对文本到视频生成模型产生的全合成假新闻视频，本文提出三分类（真、廉价假、全合成假）检测任务并构建相应数据集，进一步设计了基于多模态大模型的推理引导检测框架；

**💡 创新点**

创新点在于：①首次将全合成假新闻视频纳入检测范畴并定义三分类任务；②构造PS‑FNVD数据集，系统模拟模态对齐陷阱与语义视觉退化；③提出基于七维推理结构的R‑T2V框架，通过有条件推理生成与监督微调实现显式拆解；

**🔧 技术方法**

采用多模态大模型（Qwen2.5‑VL‑7B‑Instruct、GPT‑4o）进行条件推理生成，利用LoRA对模型进行有监督微调，并在推理过程中评估语义逻辑与物理生成痕迹；

**📊 数据集**

使用自研的PS‑FNVD数据集（共6636条视频，包含1687真、1631廉价假、1631/1687全合成假），基于FakeSV原始数据并结合Hunyuan Video文本到视频生成模型构建；

**📈 对比分析**

在PS‑FNVD上与十类基线（零射击LLM、基于代理的推理、监督训练）对比，R‑T2V取得84.79%准确率、80.00%宏F1，分别比第二佳方法高出12.20%和8.46%；

**⚠️ 局限性**

局限性包括：仅关注视频内容，未考虑社交传播与用户互动；音频仅用文本转录，忽略原始语音特征；对混合型视频（部分真实+部分合成）标签模糊。

---

## 217. MemOPD: On-Policy Distillation through Memory State Alignment for Long-Horizon Agents

**arXiv ID:** 2608.07068 | [PDF](https://arxiv.org/pdf/2608.07068v1)

**作者:** Zhiyuan Liu `[一作]` (Peking University), Songfang Huang `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的训练框架MemOPD，用于解决长时间交互中上下文重写导致的状态不匹配问题，通过重建每次模型调用的状态来优化记忆更新。

**💡 创新点**

创新点在于引入了记忆状态对齐的概念，确保教师评分与学生生成的动作在相同的状态下进行，从而提高了模型的性能和效率。

**🔧 技术方法**

使用了记忆对齐的在线蒸馏（MemOPD）技术，结合了近端策略优化（PPO）和教师指导。

**📊 数据集**

使用了多目标问答基准（HotpotQA和Natural Questions）和单目标Wiki-RAG数据集进行实验。

**📈 对比分析**

与传统的PPO方法相比，MemOPD在F1分数上提高了最多416.2%，并在训练期间实现了最高1.63倍的计算速度提升。

**⚠️ 局限性**

限制在于该方法依赖于教师模型的质量和稳定性，且在不同上下文更新机制下的适用性仍需进一步验证。

---

## 218. Detection and Ranging of Transient Extrinsic Contacts Based on 6D Dynamic Tactile Sensing

**arXiv ID:** 2608.07075 | [PDF](https://arxiv.org/pdf/2608.07075v1)

**作者:** Haowen Zheng `[一作]` (Harbin Institute of Technology), Yitian Shao `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了基于单一6轴惯性测量单元的事件触发式触觉感知框架TECDAR，用于在机器人抓持物体时快速检测和定位瞬时外部接触。

**💡 创新点**

创新点在于将低成本小尺寸IMU与差分动力学约束结合，通过高频惯性信号实现毫米级定位，仅以84 KB/s的数据率完成与可视化相当的精度。

**🔧 技术方法**

技术核心包括事件驱动的加速度触发、角速度解算、扩展卡尔曼滤波以及贝叶斯融合的几何约束求解。

**📊 数据集**

使用的实验数据集为多种3D打印塑料模型（方块、尖头、盒子）以及日常物体（笔、纸刀、书、擦子）在实际机械臂上的采集。

**📈 对比分析**

与基于几何约束的SVD/梯度下降法及视觉触觉传感器（GelSlim、GelSight Mini）对比，TECDAR在180 ms内平均误差约7 mm，延迟仅20 ms，数据吞吐量比视觉触觉低两位数。

**⚠️ 局限性**

局限性包括只能对刚体有效，受抓握力和材料硬度影响、易受滑移破坏约束、对柔性或高滑移表面表现欠佳，以及需要在不同工作状态下重新校准传递系数。

---

## 219. Optimal Neural Network Approximation via Empirical Least Squares with Deterministic Samples

**arXiv ID:** 2608.06687 | [PDF](https://arxiv.org/pdf/2608.06687v1)

**作者:** Xinliang Liu `[一作]` (Ocean University of China), Jinchao Xu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并证明了基于线性化 ReLU^k 神经网络的离散残差最小二乘近似方法，在球面上对椭圆谱方程的误差收敛理论和误差估计。

**💡 创新点**

提出了针对球面 ReLU^k 网络的 Bernstein 不等式和残差空间逆估计，完成了从连续到离散的最优误差收敛率分析；给出了高概率随机样本的近似结果；通过球面延拓实现了有限域上仿射 ReLU^k 网络的近似推导。

**🔧 技术方法**

采用球谐函数展开、超球面积分、Ultraspherical 多项式、滤波核局部化、Bernstein 不等式、逆逼近定理、Marčinkiewicz–Zygmund 不等式、随机矩阵/采样不等式、Sobolev 与椭圆算子谱理论。

**📊 数据集**

主要使用人工合成的球面函数和目标函数，实验采用 Sobol 采样点；在立方体实验中使用均匀 Sobol 序列；未使用公开真实数据集。

**📈 对比分析**

与理论预期的收敛指数做对比，采用多尺度参数 (n = 64, 128, 256, 512, 1024) 评估误差并拟合斜率。球面实验收敛率与理论相符或更好；立方体实验收敛率明显低于球面理论，表明方法对球面几何依赖较强。

**⚠️ 局限性**

仅在球面上完成完整误差理论；对有界域的扩展仅适用于 β = 0 且不考虑边界条件；对高维立方体实验缺乏理论保证；随机样本的高概率结果假设较强；理论依赖网络参数的反对称分布，实际实现需复杂采样；对高阶激活 k 与维度存在严格限制。

---

## 220. ReQuant: Fixed-Grid Discrete Refinement for Post-Training Quantization

**arXiv ID:** 2608.07019 | [PDF](https://arxiv.org/pdf/2608.07019v1)

**作者:** Yongge Ma `[一作]` (Peking University), Tong Yang `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种后处理阶段 ReQuant，利用离线坐标下降在已完成的 PTQ 模型上对离散权重量进行细化，保持量化格式不变，进一步降低重构误差和提升下游性能。

**💡 创新点**

创新点在于：①将已完成的 PTQ 结果视为可行初始化，在固定量化网格上进行可行的离散优化；②采用无反向传播、无 STE 的坐标更新规则，只在整数编码上迭代；③通过逐行、逐列的局部搜索实现并行化，计算复杂度仅为一次 GPTAQ 迭代。

**🔧 技术方法**

核心技术包括：基于重构误差的二阶近似目标、行分解的坐标下降更新、局部邻域搜索（K‑neighborhood）、增量梯度更新以及对激活不匹配的校正（GPTAQ 思路）。

**📊 数据集**

在 WikiText-2（校准与评估）、UltraChat-2k、NuminaMath 等数据集上进行评估，覆盖 Llama‑3 8B/70B、Qwen3‑14B、Qwen3‑235B MoE 等多种大模型。

**📈 对比分析**

与 RTN、AWQ、GPTQ、GPTAQ、FlexRound 等 PTQ 基线对比。实验显示，ReQuant 在 W4A16/W4A4/W3A4/W2A4 下显著提升 perplexity、KL、下游零样本准确率；对简单初始化和低位宽的提升最为显著；与 FlexRound 相比，ReQuant 既能获得更好或相近的性能，又具备更低的离线成本。

**⚠️ 局限性**

局限性包括：①只在固定网格和已冻结的比例/零点上搜索，可能无法突破网格限制；②离线细化成本随迭代次数 T 增加，尤其对 RTN 等极快基线显得较高；③结果依赖于校准集与统计估计，分布漂移时效果不确定；④目前仅优化权重量，未考虑激活与权重联合细化。

---

## 221. SkillAligner: Treating Retrieved Skills as Adaptable Drafts at Execution Time

**arXiv ID:** 2608.06880 | [PDF](https://arxiv.org/pdf/2608.06880v1)

**作者:** Qinfeng Li `[一作]` (Zhejiang University), Xuhong Zhang `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种执行时技能自适应框架 SkillAligner，将检索到的通用技能转换为与当前任务、环境和其他技能兼容的执行指南，解决技能与执行之间的不匹配问题。

**💡 创新点**

创新点在于：①将检索到的技能视为可调节的“草稿”而非固定模板；②在一次性自适配过程中同时完成任务定向、环境对齐和技能组合，消除三类不匹配（任务、环境、互相冲突）；③在执行阶段不再直接使用原始技能，而是利用生成的四字段执行指南，显著降低回退和错误率。

**🔧 技术方法**

技术实现主要依赖：大语言模型（Qwen3-8B/32B、Gemini 2.5 Pro）执行一次性自适配；BM25检索从技能库中提取前5个相关技能；自适配过程包括任务合同提取、技能碎片化、与执行接口匹配与修复、以及依赖/冲突消解；生成的指南包含 Primary、Checks、Avoid、Fallback 四部分。

**📊 数据集**

使用的公开数据集包括 ALFWorld（多步实体交互任务）、WebShop（电商购物任务）和 SearchQA（基于检索的问答任务），三者均采用统一的技能库和检索策略。

**📈 对比分析**

与基线比较：No Skill、Top‑k Raw Skill、Graph of Skills、ReasoningBank、MemP、SkillOS、GraSP、SkillRAE、SkillDAG、SkillPyramid 等。SkillAligner 在所有 9 个（3 数据集 × 3 模型）设置中均达到最高成功率/精确度，平均提升 3.9 分（相较最强基线 SkillPyramid）并降低回退率至约 2%（相比原始 10–18%）。同时，执行成本平均下降 38%（尽管自适配开销为 8%），实现显著的总成本节省。

**⚠️ 局限性**

局限性：①仅在任务开始前进行一次自适配，无法在执行过程中根据中间状态进一步调整；②对极度动态或高度不确定的环境仍可能出现新型不匹配；③依赖大模型的推理能力，若模型质量不足，可能无法正确识别或修复不匹配；④需要手工设计四字段指南模板，适配不同任务类型时需额外工程。

---

## 222. Density-aware Hierarchical Clustering Based on Element-Categorized Connection Subgraphs

**arXiv ID:** 2608.06990 | [PDF](https://arxiv.org/pdf/2608.06990v1)

**作者:** Yuning Yu `[一作]` (Tongji University), Bin Feng `[通讯]` (RadioSky (Shanghai) Communication Technology Co., Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于元素分类连接子图（ECS）的密度感知层次聚类算法 DHC-ECS。

**💡 创新点**

创新点在于通过 KNN 连接子图对连接区域的点和边显式分类，结合链路紧凑度、链路相似度、密度相似度与变异系数，构造新的聚类相似度度量，并引入自适应阈值减少手工调参。

**🔧 技术方法**

使用了层次聚类、密度聚类、图聚类、KNN 连接子图、核密度估计、几何/地理距离、两阶段贪婪合并和自适应阈值机制。

**📊 数据集**

在十个常见合成数据集上评估：2circles、Aggregation、Halfkernel、Flame、Jain、Pathbased、R15、Spiral、Two circle noise、Compound。

**📈 对比分析**

通过与传统距离相似度的层次聚类、ACHAMELEON、RNN-DBSCAN、McDPC、G-RMS 等基线方法在 NMI、ARI、FMI、Purity 四个外部指标上比较，DHC-ECS 在所有数据集上 NMI≥0.95，整体表现更稳健且阈值 T_s 具备一定通用性。

**⚠️ 局限性**

算法最坏情况下时间复杂度 O(N³)、空间复杂度 O(N²)，在高维空间中距离与密度估计可能失效，且需要进一步优化复杂度以适应更大规模数据。

---

## 223. InsertFuse: A Unified Framework for Multi-Category Reference-Guided Image Insertion

**arXiv ID:** 2608.06490 | [PDF](https://arxiv.org/pdf/2608.06490v1)

**作者:** Guangzhao Li `[一作]` (Shanghai Jiao Tong University), Xiaohong Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 35573 | [OpenAlex ID](https://openalex.org/A5063022663)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本工作提出一种统一的多类别参考引导图像插入框架InsertFuse，能够在保持参考对象特征的同时将其精准插入到目标场景中；

**💡 创新点**

核心创新在于将类别专门化学习与跨类别能力整合分离，通过Insertion On-Policy Distillation（IOPD）将多名专家的专业能力压缩到单一学生模型；另外引入Token‑Aligned Geometry Conditioning（TAGC）、Region‑Balanced Flow Matching（RBFM）以及Reference Classifier‑Free Guidance（Reference CFG）三种机制，分别提升空间对齐、区域监督均衡与参考保真；

**🔧 技术方法**

技术手段包括基于预训练Qwen‑Image‑Edit‑2511的flow‑matching框架，LoRA与轻量级几何编码器、TAGC将插入掩模转化为token‑aligned几何特征、RBFM对插入区与背景误差分别归一化、Reference CFG通过对比有无参考条件提取参考贡献、IOPD实现基于学生自身轨迹的专家对齐蒸馏；

**📊 数据集**

使用五个类别（配饰、动物、服装、通用对象、人类）各约1,000张训练样本的多类别测试集，以及公开的AnyInsertion基准数据集进行评估；

**📈 对比分析**

与AnyDoor、Insert Anything、A²‑Edit等专门插入方法以及FLUX.1 Kontext、Qwen‑Image‑Edit‑2511、Qwen‑Image‑2.0等统一编辑模型对比，InsertFuse在DINO‑I、CLIP‑I、PSNR、SSIM、LPIPS、FID等指标均达到或位居榜首，显示出更高的参考保真度、空间适配性与生成质量；

**⚠️ 局限性**

局限性包括：需要预先训练多个类别专家且蒸馏过程成本较高，无法直接处理未见类别的插入任务，且在极大或极小插入尺寸下仍可能出现局部细节失真或融合不自然。

---

## 224. HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management

**arXiv ID:** 2608.07009 | [PDF](https://arxiv.org/pdf/2608.07009v1)

**作者:** Zhiqiang Xie `[一作]` (Stanford University), Christos Kozyrakis `[通讯]` (Stanford University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个层次化的 KV 缓存系统，将每个请求的完整 KV 历史保存在主机内存中，GPU 仅保留一个固定大小的热缓存，利用融合的 CUDA 核心实现快速命中/缺失检测、LRU 替换和主机到设备的数据拷贝，同时为共享索引的模型提供精确预取。

**💡 创新点**

创新点包括：① 彻底分离逻辑 KV 可用性与物理 GPU 位置，实现“索引器无关”的统一接口；② 在 decode CUDA 图中使用单一融合核完成命中判断、Victim 选取、IO 与 metadata 更新，极大降低了批量推理的计算开销；③ 通过 LRU 结合层间选择共性实现高命中率，并在共享索引模型上实现全量预取，进一步隐藏 IO 延迟。

**🔧 技术方法**

使用技术有：GPU 缓存 + 主机内存分层、LRU 替换策略、GPU 友好的 IO（直接从主机 pinning 的 DRAM 读取）、单核融合 miss‑resolution kernel、CUDA Graph 记录、层间预取机制、以及不同硬件平台（H200、B200、GH200）下的 PCIe/NVLink 通道。

**📊 数据集**

评估数据集主要包括：LongBenchV2 选取跟踪、GLM‑5.1/5.2（DeepSeek‑DSA）和 Qwen3‑30B‑A3B（Quest）模型在 4K~200K 长上下文下的生成任务，输出长度固定为 8K。

**📈 对比分析**

与传统在 HBM 中保留完整 KV 的 SGLang 基线相比，实验显示：在 H200 上 32K/8K 负载下吞吐量提升 2.1‑2.9 倍；在 200K 长上下文下提升 3.6‑4.7 倍；per‑token 延迟保持不变或仅略升；TTFT 在高负载下显著下降。性能提升主要来自可承载更多并发请求，而不是单步加速。

**⚠️ 局限性**

限制与不足：① 当模型不受容量瓶颈影响时，额外 IO 开销导致 per‑token 生成时间上升 7‑8 ms；② 需要足够的主机内存来存储完整 KV，Grace‑based 系统上主机内存相对 HBM 较少，限制了可扩展性；③ 预取效果依赖于模型层间索引共享，对不共享索引的模型只能采用预测方案，提升有限；④ 目前 cache 规模为静态配置，未实现动态重分配。

---

## 225. Sub-Quadratic Bisimulation Metrics via Approximate Nearest Neighbors: Coverage-Augmented Guarantees and Computable Two-Sided Certificates

**arXiv ID:** 2608.06762 | [PDF](https://arxiv.org/pdf/2608.06762v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Joyanta Jyoti Mondal `[通讯]` (University of Delaware)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过低维嵌入与近似最近邻索引，实现子二次复杂度的 bisimulation metric 计算，并给出可验证的误差证书。

**💡 创新点**

①证明全局误差由覆盖率决定，给出可观测的三明治宽度作为误差上界；②提出覆盖增益的子二次求解算法；③给出匹配的下界，说明覆盖不可完全消除误差。

**🔧 技术方法**

低维多样本 MDS 嵌入、随机超平面 LSH 近似最近邻索引、精确的 Kantorovich‑Wasserstein 备份、双臂单调限制迭代与收敛分析。

**📊 数据集**

随机生成的 10‑18 状态 MDP，Taxi‑v3（500 状态），2500 状态网格世界，以及 64 状态分组 MDP（用于与 MICo/DBC 比较）。

**📈 对比分析**

与完整全对全迭代、随机覆盖以及学习型近似（MICo、DBC）比较；子二次实现相对全二次获得 1.4–7.2 倍加速，最高 25.5 倍；在 2500 状态实验中仅 12.8% 的一次扫除即可比学习型方法降低 28.6% 的值损失；在 Taxi 实验中证书能识别无信息嵌入。

**⚠️ 局限性**

误差受覆盖率限制，无法仅靠改进索引质量消除；覆盖需遍历所有状态对才能完全收敛，导致子线性仅在特定条件下；自适应评估仍需 Ω(n) 次评估，完全超线性下界尚未解决；在极大规模 MDP 时仍需大量查询；精确 Wasserstein 计算仍是瓶颈。

---

## 226. Retrofitting Linear Attention into Diffusion Language Models

**arXiv ID:** 2608.06628 | [PDF](https://arxiv.org/pdf/2608.06628v1)

**作者:** Jinha Kim `[一作]` (Apple), Jaeyeon Kim `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在预训练的 diffusion 语言模型 LLaDA 2.1 mini 上，本文通过两阶段后训练，将部分注意力层改为 block‑hybrid 形式，即在当前 denoising 块内保持精确 softmax 注意力，而对已完成块使用固定大小的线性注意力状态，从而降低 KV 缓存成本并加速推理。

**💡 创新点**

创新点在于提出 block‑hybrid 注意力结构：在块内保留 bidirectional softmax，跨块压缩为固定长度的线性注意力记忆；并设计了两阶段迁移学习（注意力传递 + LoRA 微调）实现高效、低成本的后期 retrofit。

**🔧 技术方法**

使用技术包括 Hedgehog 线性注意力特征映射、两阶段后训练（注意力传递与 LoRA 微调）、SGLang Triton 内核融合实现统一的 softmax 与线性分支、固定状态的 KV 缓存以及多头注意力的门控融合。

**📊 数据集**

后训练使用公开的 Tulu SFT 混合数据集；评估基准包含 HumanEval、HumanEval+、MBPP、MBPP+、CMATH、GPQA‑Diamond 等编程、数学与推理任务。

**📈 对比分析**

在 no‑edit decoding（τ=0.7）下，Hybrid 在上述基准上仅略低于 teacher（1.6–3.6% 的差距）；在 SGLang 连续批处理下，Hybrid 在 16–256 并发请求上实现 1.5–1.73× 的解码吞吐量提升，且内存占用保持固定，可承载更多并发请求。

**⚠️ 局限性**

局限性包括仅 linearize 6/20 层，仍保守；对更长序列或不同块大小的泛化尚未验证；对最难推理任务仍存在约 8% 的性能下降；未探索更激进或自适应层选择方案。

---

## 227. Ising Acceleration for Multi-Robot Multi-Target Planning

**arXiv ID:** 2608.06803 | [PDF](https://arxiv.org/pdf/2608.06803v1)

**作者:** Ahmet Efe `[一作]` (University of Minnesota), Ulya R. Karpuzcu `[通讯]` (University of Minnesota)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究并实现了利用紧凑型CMOS Ising芯片对多机器人多目标规划（目标共享、巡航路径构造和路径搜索）进行硬件感知的加速方案，提出了分层拆分、补丁滑动、聚类分割等方法，并构建了多映射（自旋合并、系数量化、分支枚举）流水线以提升鲁棒性。

**💡 创新点**

创新点包括：① 将规划问题按层级拆解为符合自旋数、系数范围等硬件约束的小子问题；② 通过多映射组合（自旋合并、系数量化、分支枚举）构成映射组合包，提高成功率并抑制单一映射失效；③ 递归目标共享、聚类巡航、补丁滑动路径等专为CMOS Ising硬件设计的算法；④ 在端到端规划中实现能耗显著降低（高达130×），同时保持较低的解质量损失。

**🔧 技术方法**

技术手段包括：QUBO → Ising映射、硬件约束感知映射（自旋合并、系数量化、分支枚举）、分层拆分（路径补丁、聚类分割、递归分割）、逻辑Ising后端Tabu采样、并行芯片调用与结果解码、Python/C++主机逻辑以及基于45-spin CMOS Ising芯片的硬件实现。

**📊 数据集**

使用随机生成的10×10网格地图（障碍率20%），随机放置机器人、目标，分别在路径、巡航、目标共享和整体端到端实验中使用不同规模（如3机器人/10目标、10目标/10机器人等）的随机实例集合。

**📈 对比分析**

对比方法包括经典A*、BFS、GBFS、Nearest-Neighbor、2-opt/3-opt、Round-Robin、PSA、SSA等基线。实验结果显示：路径搜索能耗下降约37×；目标共享能耗下降约8000×；端到端能耗下降约130×，路劲长度仅比最强基线高9%；成功率接近100%，但整体延迟较大（≈341 ms 对比 27 ms）。

**⚠️ 局限性**

主要限制：① 受限于45自旋和整数系数范围，无法直接实现一热约束的完整TSP，需改用逻辑Ising后端；② 物理芯片需要多次调用导致较高延迟；③ 仅适用于小规模实例，扩展到更大问题需更高自旋数或更宽系数范围；④ 多映射策略需并行硬件支持，单芯片调用时无法充分利用并行性。

---

## 228. FedTransKD-IDS: Robust Federated Transfer Learning with Knowledge Distillation for Intrusion Detection in IoT

**arXiv ID:** 2608.06447 | [PDF](https://arxiv.org/pdf/2608.06447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 229. MARS: A Monte Carlo Tree Search-based Adaptive and Responsive Scheduler

**arXiv ID:** 2608.06629 | [PDF](https://arxiv.org/pdf/2608.06629v1)

**作者:** Yash Kurkure `[一作]` (University of Illinois Chicago), Michael E. Papka `[通讯]` (University of Illinois Chicago)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发一种无需训练、可在运行时通过奖励函数配置目标的Monte Carlo Tree Search（MCTS）调度器MARS，取代传统启发式和深度强化学习在HPC调度中的不足；

**💡 创新点**

创新点在于：①训练无关，目标可即时切换；②利用MCTS与离散事件模拟实现前瞻性规划，支持系统保留与耗尽；③通过启发式分支、树剪枝和根并行化解决大分支与实时约束；

**🔧 技术方法**

使用技术包括：Monte Carlo Tree Search、离散事件模拟（CQSim）、启发式分支、树剪枝、根并行化、C++并行实现，以及可配置的奖励函数（等待时间、利用率）；

**📊 数据集**

使用的数据集为Argonne Leadership Computing Facility的两组真实生产工作负载：2024年Polaris（47,184个作业）和2021年Theta（30,415个作业），包含维护事件；

**📈 对比分析**

与传统启发式（FCFS、SJF、WFP等）、深度强化学习调度器RLScheduler和随机策略进行对比。结果显示MARS-CW在Theta上尾部等待时间降低64%、Polaris降低43%；MARS-CU在维护前48小时提升利用率；总体而言MARS在等待时间和利用率两个目标上均优于或匹配传统启发式，显著优于DRL；

**⚠️ 局限性**

局限性包括：需要高并行计算资源（256核）；对奖励函数及搜索参数（树深度、时间预算）敏感；未考虑能耗等新型优化目标；评估仅在模拟环境下完成，真实系统集成仍待验证。

---

## 230. Pre-Inference Routing for Cost-Efficient Document Field Extraction

**arXiv ID:** 2608.06607 | [PDF](https://arxiv.org/pdf/2608.06607v1)

**作者:** Sreerekha Rajendran `[一作]` `[通讯]`, Sreerekha Rajendran

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种预推理路由（pre‑inference routing）方法，根据文档的低成本特征决定使用廉价模型还是高成本模型完成字段抽取。

**💡 创新点**

创新点在于：①给出两步诊断（路由头部空间与特征可预测性）评估路由是否有效；②使用13个可解释的文档内特征与校准随机森林构建路由器；③证明在非收据类（广告表单）同样可行，并与简单文本基线相媲美。

**🔧 技术方法**

主要技术：OCR（Tesseract）提取文本、计算OCR置信度、图像质量、布局与内容密度特征；随机森林（校准后）路由器；成本模型基于token费用；对比基线包括 always‑small/large、单一特征规则、随机阈值及 oracle。

**📊 数据集**

使用五个文档类别的公开数据集：收据（CORD、SROIE）、广告购买表单（DeepForm）、发票（DocILE）、营养标签（POIE）和注册表单（VRDU）。

**📈 对比分析**

评估方式：在每个数据集上绘制质量–成本 Pareto 前沿，计算在保持 F1 误差 ≤0.02 的前提下的成本节省；并对 AUC、阈值转移、与信心级联等进行对比。性能方面：收据路由节省 31–33% 成本；广告表单节省 77%；AUC 在 0.71–0.91 之间；在无法满足两条诊断条件的类中路由无效。

**⚠️ 局限性**

局限性：①路由仅在满足两条诊断条件（头部空间和特征可预测性）的类别才有效；②路由器不跨数据集、跨语料或跨模型对接；③仅对单页处理，未覆盖多页文档；④依赖 OCR 与特定 LLM 版本，可能随服务商更新而变；⑤实验未包含 FUNSD 等结构化实体抽取任务。

---

## 231. Beyond Foundation Models: Dimension-Aware Neural Architecture Search with Small-Data Representation Models for Cryocooler Lifetime Prediction

**arXiv ID:** 2608.06993 | [PDF](https://arxiv.org/pdf/2608.06993v1)

**作者:** Gregor Molan `[一作]` (Comtrade 360 d o o), Martin Molan `[通讯]` (Comtrade AI GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在有限、单一领域的工业时序数据（冷却器遥测）上，构建了 FSD‑RM（Family of Small‑Data Representation Models）框架：利用CNN1D、LSTM、GRU、Transformer 四种成熟编码器在无监督 Seq2Seq 重构任务中学习低维任务无关表征；随后通过 dimension‑aware NAS（da‑NAS）在 2~512 维空间里递归地搜索最优容量与维度；最后将得到的表征用于二分类寿命预测与一类异常检测，形成从原始信号到寿命决策的完整两阶段流水线。

**💡 创新点**

（1）引入容量可控的表征学习，避免大模型过拟合；（2）提出专为小样本设计的 da‑NAS，按维度层层搜索并利用跨维度性能传递与 Beat‑Lower‑Dimension 终止；（3）系统性比较四种编码器与多种下游分类器，验证跨任务可迁移性；（4）在极度不平衡、不同阈值的工况下对模型做细粒度敏感性分析。

**🔧 技术方法**

技术栈：时序预处理（滤波、缺失剔除、Robust Scaler）；无监督 Seq2Seq 表征学习（CNN1D、LSTM、GRU、Transformer）；dimension‑aware NAS（Optuna+自定义停机策略）；下游任务采用多种经典分类器（NB、LR、SVM‑RBF、RF、XGB、MLP、KNN）和 One‑Class SVM；评价指标包括 ROC‑AUC、PR‑AUC、F1‑Macro；实验环境为 Leonardo 预算层 GPU（A100）。

**📊 数据集**

数据集：1,305 条无标签冷却器遥测序列与 95 条带有寿命标签的序列；遥测包含时间域（温度、电流、转速等）和频域（噪声频谱）两种模态；寿命阈值分别设为 10,000 h、15,000 h、20,000 h，形成 3 种不同不平衡比例的二分类任务；实验通过 5%–100% 子样本分割、3 组阈值和 3 个类别不平衡级别来评估模型。

**📈 对比分析**

与大规模预训练基线（TimesFM、UniTS、PatchTST、Mamba）直接对比受限，但通过与多种轻量级分类器的组合评估，FSD‑RM 在全量训练下可达 ROC‑AUC 0.84（Transformer+LR）或 0.83（CNN1D/LSTM/GRU），并在 5% 训练、极端不平衡（20,000 h）时仍保持 0.70+；一类 SVM 在最高 0.84 的 AUC；多任务一致性表明同一表征可无重训练地服务二分类与异常检测，证明了任务无关性。

**⚠️ 局限性**

局限性：1）仍依赖于有限标签（仅 95 条），无法证明对更大规模数据的可扩展性；2）Transformer 在低数据、极度不平衡时性能波动大，需要额外调参；3）da‑NAS 需要多轮训练，搜索时间相对较长；4）模型和结果主要针对冷却器遥测，迁移到其他工业时序任务需重新评估；5）未与真正的全局预训练基础模型做直接数值比较，缺乏跨模型基准。

---

## 232. Every Cache Entry Earns Its Place: Global Allocation of Resolution and Coverage for KV Cache Compression

**arXiv ID:** 2608.07001 | [PDF](https://arxiv.org/pdf/2608.07001v1)

**作者:** Haolin Tian `[一作]` (Tsinghua University), Tonghan Wang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM KV缓存压缩提出全局资源分配方案GraceKV，构建原型树并在GPU上实现压缩。

**💡 创新点**

创新点在于将缓存压缩视为全局预算分配问题，使用可扩展原型树结合价值流与预算流实现自适应分辨率与覆盖。

**🔧 技术方法**

采用曲率引导语义分割、逐层价值流、预算流、singleton floor以及GPU端原型树构造等技术。

**📊 数据集**

使用LongBench与RULER数据集，评测 Qwen2.5-7B-Instruct 与 Llama-3.1-8B-Instruct。

**📈 对比分析**

与 FullKV 及八种基线对比，GraceKV 在 32 个压缩设定中首位 24 次，压缩比高达 128× 仍保持稳健性能。

**⚠️ 局限性**

局限在于贪心分配无法保证全局最优，原型树构造产生一次性延迟和额外内存。

---

## 233. Entanglement-Assisted Quantum Locally Recoverable Codes: Bounds, Optimal Constructions, and Achievability

**arXiv ID:** 2608.06854 | [PDF](https://arxiv.org/pdf/2608.06854v1)

**作者:** Vijay Kumar `[一作]` (International Institute of Information Technology), Ramakrishna Bandi `[通讯]` (International Institute of Information Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并分析了一类基于纠缠辅助的CSS‑类局部可恢复量子码（EA‑qLRC），并给出了其理论界限、最优构造与可实现性；

**💡 创新点**

创新点在于突破传统对偶包含限制，利用LCD（线性可逆）码与纠缠辅助，实现任意局部可恢复性与距离的量子码，并给出了单码与对称码构造的最优性判据；

**🔧 技术方法**

主要技术包括：CSS‑类对称构造、Tamo–Barg多项式评价码与循环码的定义集设计、LCD 与自共轭性判据、单字母变换的LCD化以及Gilbert–Varshamov/串联码的存在证明；

**📊 数据集**

由于是理论构造，本文未使用具体实验数据集，所有结果均为严格数学证明；

**📈 对比分析**

与传统需要对偶包含的CSS‑qLRC相比，本文构造在保持相同距离和局部性时可实现更低的码率且仅需消耗固定比例的纠缠资源，理论上可达到完整的LRC距离–速率–局部性极值区域；

**⚠️ 局限性**

局限性包括：对有限域 q≤3 的LCD化问题尚未完全解决，导致在这些域上无法保证所有点都可实现；同时，构造在低速率高距离区间仍存在相对距离的可实现性与上界之间的明显间隙。

---

## 234. When Coordination Becomes a Threat: Communication Attacks in LLM-Controlled Multi-Robot Systems

**arXiv ID:** 2608.06830 | [PDF](https://arxiv.org/pdf/2608.06830v1)

**作者:** Zhen Huang `[一作]` (National University of Defense Technology), Zhiping Cai `[通讯]` (National University of Defense Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对 LLM 控制的多机器人系统中的通信攻击进行系统评估，提出外部入口攻击和内部特权攻击，并在 DMAS、HMAS-1、HMAS-2 三种通信架构下进行实验，最终设计了基于交互轨迹的传播评估框架和 CPV（声明溯源验证）门控机制。

**💡 创新点**

创新点包括：①首次从通信架构角度系统化分析攻击传播；②提出两种攻击模型并在三种架构中实现；③开发了信息传染性、行动传染性、过程配置等多维指标的交互轨迹评估框架；④提出 CPV 门控来验证声明的可信度和证据，提升通信安全。

**🔧 技术方法**

使用技术包括：大语言模型（GPT‑3.5‑Turbo、Qwen3‑235B‑A22B、GPT‑4o）、Isaac Sim/ROS2 物理仿真、基于交互日志的定量指标计算，以及 CPV 门的声明标注与验证。

**📊 数据集**

实验数据集为五个人工设计的任务场景：仓库巡逻、医院隐私监测、装甲协同、空地混合巡逻和物品匹配，涵盖不同的任务约束与协作需求。

**📈 对比分析**

方法比较：在平衡提示与无安全约束提示两种对话策略下对三架构进行攻击实验，并与 CPV 门控前后的违规率对比。结果显示：DMAS 最高的传播成功率；CPV 门将违规率从 70.0% 降至 36.6%，行动传播覆盖率提升显著。

**⚠️ 局限性**

限制：未对攻击者的拓扑推理能力做深入研究；实验仅在仿真环境中完成，未覆盖真实机器人部署；仅评估三种主流通信架构，未考虑更复杂或混合网络拓扑；提示级约束无法阻止后续传播，仅能降低初始违规概率。

---

## 235. Modular TTT: Rethinking Test-Time Training as Composable Modules

**arXiv ID:** 2608.07110 | [PDF](https://arxiv.org/pdf/2608.07110v1)

**作者:** Bohao Tang `[一作]` (Shanghai Jiao Tong University), Ya Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Modular TTT 框架，将测试时训练（TTT）拆分为可组合的有向无环图，自动生成训练与查询视图的前向/后向规则。

**💡 创新点**

创新点在于把 TTT 内部学习器抽象为可注册的原语并通过图化方法实现模块化、可扩展的设计空间，从而无需手工推导每个变体的全局更新。

**🔧 技术方法**

使用了自动微分、PyTorch 动态图、原语级手工实现的前向/后向运算、可学习的学习率、衰减、非线性激活，以及多尺度实验。

**📊 数据集**

使用了大型英文预训练语料库（10B tokens 及 100B tokens）进行语言建模，评估使用 OpenAI GPT‑2 BPE tokenizer，后续通过 lm‑evaluation‑harness 进行零样本下游任务。

**📈 对比分析**

对比官方 TTT、GDN、LLaMA 等基线，发现 Modular TTT 在 410M 与 1.45B 规模下训练损失、perplexity 与多选准确率与 GDN 接近，吞吐量提升 2.2~3.3 倍。

**⚠️ 局限性**

局限在于深层快权重学习器难以优化，长上下文回忆仍弱，且在更大上下文（8k）下性能远不如 LLaMA。

---

## 236. Improving the Energy Efficiency of High Throughput Computing: A Measurement-Based Case Study

**arXiv ID:** 2608.06622 | [PDF](https://arxiv.org/pdf/2608.06622v1)

**作者:** Damu Ding `[一作]` (University of Oxford), Noa Zilberman `[通讯]` (University of Oxford)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文通过对一个实际数据中心的测量案例研究，分析了高通量计算的能耗情况，并提出了减少能耗的建议。

**💡 创新点**

创新点在于提供了基于测量的能耗分析，探讨了服务器配置调整对能耗的影响，并提出了平衡性能与能效的实用建议。

**🔧 技术方法**

使用了测量技术，结合了对数据中心的电力使用信息和针对高通量工作负载的专注测量。

**📊 数据集**

使用了来自英国科学与技术设施委员会（STFC）数据中心的电力消耗数据集，以及本地集群的专注电力优化测量数据。

**📈 对比分析**

与以往的研究相比，本研究提供了更新的实证数据，分析了不同配置和操作场景下的能耗敏感性，并提出了具体的能效改进建议。

**⚠️ 局限性**

限制在于研究主要集中在特定的数据中心和高通量计算工作负载，可能不适用于所有类型的数据中心或计算场景。

---

## 237. From Points to Edges: Edge-Conditioned Spectral Operators for Physics-Sensitive PDE Learning

**arXiv ID:** 2608.06894 | [PDF](https://arxiv.org/pdf/2608.06894v1)

**作者:** Zhentao Tan `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Edge-Conditioned Spectral Operator (ESO)，通过局部边缘信息调节全局谱混合以求解 PDE。

**💡 创新点**

创新点在于引入 Pairwise-Variation Modal Mixer (PVMM) 对谱模式进行基于局部差分的自适应加权，并配合 Task-Adaptive Physics-Aware Reweighting (PAR) 强化物理敏感区训练。

**🔧 技术方法**

采用谱混合（FNO 框架）、局部差分统计、注意力式权重重构、以及 Laplace–Beltrami 基底实现局部-全局协同建模。

**📊 数据集**

使用九个标准 PDE 基准，包括 Darcy 流、Navier‑Stokes、Airfoil、Plasticity、Irregular Darcy、Pipe Turbulence、Heat Transfer、Composite 以及 Blood Flow，涵盖结构化与非结构化网格。

**📈 对比分析**

与 11 种现有神经算子（FNO、HPM、SAOT、LRSA 等）对比，ESO 在所有任务上均实现了最低的相对 L₂ 误差，尤其在物理敏感区误差显著下降。

**⚠️ 局限性**

局限在于需预先设计邻域规模、对不同网格类型的邻域构造存在依赖，并且在极大尺度或极稀疏网格下的泛化性能尚未完全验证。

---

## 238. AdvTiles: Physical Adversarial Camouflage Clothing against Person Detectors via Learnable Tiles

**arXiv ID:** 2608.06801 | [PDF](https://arxiv.org/pdf/2608.06801v1)

**作者:** Jinlei Wang `[一作]` (Sun Yat-sen University), Wen Yao `[通讯]` (Chinese Academy of Military Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于可学习瓦片的AdvTiles框架，用于生成可打印的物理对抗迷彩服以误导人体检测器。

**💡 创新点**

创新点在于将对抗纹理拆解为可学习瓦片，并使用Straight-through Gumbel-Softmax实现瓦片模式与空间布局的联合优化，从而获得细粒度控制和自然外观；同时利用3D Gaussian Splatting渲染多视角场景来提升鲁棒性。

**🔧 技术方法**

使用的技术包括可学习瓦片生成、Straight-through Gumbel-Softmax估计、3D Gaussian Splatting渲染、HDR环境光照、背景多变换以及多项对抗损失（检测、分布、边缘、缝隙）。

**📊 数据集**

实验数据集主要来自nuScenes的912张背景图（含雨天、晴天、黄昏、夜晚等天气与光照条件），合成出3648张视角与尺度多样的图像用于训练（2432张）和测试（1216张）。

**📈 对比分析**

在数字评测中与多种Patch/Texture攻击方法及随机迷彩进行对比，AdvTiles在YOLOv5上实现ASR最高达97.5%，并在物理实验中在360°视角和不同距离下均保持90%以上的ASR，明显优于现有SOTA。

**⚠️ 局限性**

局限性包括对抗效果随远距离或极端视角下降，且实现依赖复杂的3D渲染与大量计算，且在极端光照或衣物形变场景下仍可能失效。

---

## 239. Vernata: Self-Supervised Learning of LiDAR Point Representations

**arXiv ID:** 2608.06919 | [PDF](https://arxiv.org/pdf/2608.06919v1)

**作者:** Oliver Lemke `[一作]` (Robotics and AI Institute), Marco Hutter `[通讯]` (Robotics and AI Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向户外 LiDAR 点云的自监督学习框架 Vernata，改进 Sonata 以提升稀疏点云的鲁棒性和语义表达。

**💡 创新点**

三大创新：①稀疏视图增强使模型对点云密度变化不敏感；②引入 Sinkhorn‑Knopp 内存池在有限显存下稳定自蒸馏；③跨模态蒸馏利用高分辨率 2D 视觉模型提供细粒度语义指导。

**🔧 技术方法**

采用 DINOv2 风格的学生-教师蒸馏、Sparse View 采样、SwAV 内存池、LoftUp 上采样的 2D 视觉特征、线性探针评估。

**📊 数据集**

在 GrandTour、TartanGround、Waymo 三个大型户外 LiDAR 数据集以及自建的 311 帧标注集上进行预训练与微调。

**📈 对比分析**

与原始 Sonata、Finetuned Sonata 以及 PTv3 监督基线对比，Vernata 在 TartanGround 和 Waymo 上线性探针 mIoU 分别提升约 +X% 与 +Y%，在小样本场景下亦能逼近或超越监督方法。

**⚠️ 局限性**

局限性：仅对单帧点云做推理，缺乏时间上下文；在极端稀疏或缺少颜色/法向量的环境下性能仍有限；训练仍需要较多算力和大规模未标注数据。

---

## 240. GraphVerse: A Comprehensive Visual Graph Reasoning Benchmark for Multimodal Large Language Models

**arXiv ID:** 2608.06769 | [PDF](https://arxiv.org/pdf/2608.06769v1)

**作者:** Yuanfu Sun `[一作]` (New York University), Qiaoyu Tan `[通讯]` (New York University Shanghai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究构建了GraphVerse基准，旨在统一评估多模态大型语言模型（MLLM）在视觉图形推理（VGR）中的感知、视觉推理和文本推理能力，并提供一系列图像编辑策略和过程敏感的评分指标；

**💡 创新点**

创新点包括：① 统一感知-视觉推理-文本推理的端到端评估框架；② 四种图像编辑策略（图-图补丁扰动、跨图合成、图注意聚焦、空间条件重色）以提升视觉推理挑战；③ 过程敏感的VGR-Score评分方法；④ 单图与双图情境的全面覆盖；⑤ 公开数据集与代码，方便复现与扩展；

**🔧 技术方法**

技术手段包括：使用GraphViz渲染真实世界子图为图像；设计四种图像编辑策略；通过程序化生成答案与模板化的图结构说明进行评估；利用GPT‑5.1等LLM对模型输出进行过程与答案的自动评分；采用Program‑of‑Thoughts提示、SFT与RL等训练/无训练方法提升模型表现；

**📊 数据集**

数据集来源于多种真实图源（DBLP、社交网络、DBpedia1M、OpenFlights、PubChemQC、PCQM4Mv2等），采样子图后渲染为图像，构成11,000条样本（测试子集1,060条），覆盖单图与双图、多种经典图论任务；

**📈 对比分析**

通过与19款开源与闭源MLLM的Acc与VGR‑Score比较，实验表明即使是最强模型仍存在显著差距，尤其是双图任务更难；文本‑only pipeline效果差；无训练的PoT提示和训练后SFT/RL均能提升性能，训练基于GIE的数据能进一步提高；VGR‑Score揭示过程质量与最终答案准确度往往不一致；

**⚠️ 局限性**

局限性在于缺乏更多定性案例，实验仅涉及单图与双图推理，未覆盖更丰富的多图场景；未来计划扩展GraphVerse至更广泛的多图VGR，并进一步完善评估与案例分析。

---

## 241. Online Monitoring and Corrective Steering of Programming Agents

**arXiv ID:** 2608.06701 | [PDF](https://arxiv.org/pdf/2608.06701v1)

**作者:** Shuyang Liu `[一作]` (University of Illinois Urbana Champaign), Reyhaneh Jabbarvand `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种在线监控与纠正驱动机制，用于在编程代理执行 GitHub 问题修复任务时实时识别并纠正行为漂移。

**💡 创新点**

创新点在于：①将判断（是否出现漂移）与建议（如何纠正）解耦，使用确定性的规则监控器先检测漂移；②只有在检测到漂移时才调用 LLM 提供高层次的下一步纠正建议，避免 LLM 误判或重复规划；③提供可配置的通用漂移信号与预定义/自定义建议，兼顾效率与可靠性。

**🔧 技术方法**

技术手段包括：基于过程中心的轨迹表示（轨迹图与阶段序列），确定性规则检测（如循环、停滞、跳过阶段等漂移信号），阈值冷却与阻塞/非阻塞策略，以及 LLM 作为轻量级建议生成器。

**📊 数据集**

使用了两个公开的 GitHub 代码修复基准：SWE‑Bench Pro（Python）和更具挑战性的另一份基准（未命名），共计 7752 条轨迹，涵盖中等与困难难度。

**📈 对比分析**

与 vanilla、SAGE 以及两种基线的周期性干预/预定义干预方案进行对比，实验表明在所有模型与基准上，平均提升 9.9%（最高 15.2%）的修复成功率，且成本仅增加约 $0.08/实例，主要改善了中等/困难问题的解决率。

**⚠️ 局限性**

局限性包括：①漂移检测依赖预设阈值和规则，可能漏检或误报；②仅在 GitHub 代码修复场景下验证，无法直接证明跨领域通用性；③对 LLM 生成建议的依赖仍可能引入偶发误判；④在极端长轨迹或高频漂移场景下，冷却阈值与阻塞上限可能需要进一步调优。

---

## 242. When Context Bites: Detecting RAG Poisoning via Document-Level Attention Collapse

**arXiv ID:** 2608.06947 | [PDF](https://arxiv.org/pdf/2608.06947v1)

**作者:** Yingtao Ren `[一作]` (University of Technology Sydney), Chin-Teng Lin `[通讯]` (University of Technology Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究 Retrieval-Augmented Generation（RAG）系统在遭遇对抗性注入攻击时的内部机制，并提出基于文档级注意力坍塌（Attention Collapse）的轻量级检测方法 D‑SCAN。

**💡 创新点**

①发现并量化了“注意力坍塌”——攻击样本使模型在文档层面聚焦于恶意文档、熵显著下降；②基于此特征构建文档级注意力密度与熵指标；③通过机制可解释性视角从内部动态而非输出端识别攻击。

**🔧 技术方法**

机制可解释性分析、注意力密度/熵计算、线性分类器，多采样推理；使用 Llama‑3.1‑8B、Qwen 系列 LLM 作为生成器；攻击样本由 PoisonedRAG 生成。

**📊 数据集**

HotpotQA、2Wiki、Musique 三个多跳问答基准；通过 PoisonedRAG 在这些基准上构造对抗性检索文档。

**📈 对比分析**

与零样本 LLM 检测器（如 Qwen‑7B）、基线方法 HaloScope、ReDeep、RevPRAG 等在同一基准上对比；D‑SCAN 在所有数据集上的 AUC 与 F1 均超过 0.85，单采样（N=1）仍保持 AUC >0.8，且在攻击成功与失败场景均保持高检测精度。

**⚠️ 局限性**

仅依赖注意力分布，可能对某些模型或更复杂的攻击方式（如多文档混合攻击）不够鲁棒；需要模型内部注意力权重的可访问性；对模型规模的敏感性（较大模型可能更易被攻击）；在极低采样或极多文档检索场景下性能可能下降。

---

## 243. StepJack: Benchmarking Computer-Use Agent Safety Against Multi-Step Indirect Prompt Injection

**arXiv ID:** 2608.06477 | [PDF](https://arxiv.org/pdf/2608.06477v1)

**作者:** Zhuoxin Zhan `[一作]` (Simon Fraser University), Layla El Asri `[通讯]` (RBC Borealis)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多步间接提示注入攻击和对应的 StepJack 评测基准。

**💡 创新点**

创新点：①将恶意目标拆分成多步无害子步骤并分布在多个网页；②开发了自动化分解流水线和分解策略分类法；③构建了可扩展的 480 条样本基准。

**🔧 技术方法**

使用 LLM 驱动的分解流水线（Qwen3.5-27B 作为 LLM），并在 RedTeamCUA sandbox 中进行实验评估。

**📊 数据集**

数据集为 StepJack，包含 480 条测试样本，覆盖 12 种攻击目标、不同平台、用户指令模式、分解深度等维度。

**📈 对比分析**

对比六款主流 CUA，单步基准下 ASR 平均 31.3%，多步提升至 36.9%（单个模型最高提升 31.2 分）；多步攻击在 DSP 与 PID 防御下仍优于单步。

**⚠️ 局限性**

局限性：实验仅跑一次，API 与计算成本高；基准规模受限于 12 个攻击目标和深度 ≤3；平台与任务维度相互耦合。

---

## 244. DAEP: Difficulty-Aware Evidence Planning for Medical Video Corpus Temporal Answer Grounding

**arXiv ID:** 2608.06869 | [PDF](https://arxiv.org/pdf/2608.06869v1)

**作者:** Tianjian He `[一作]` (TikTok, ByteDance), Changbo Xu `[通讯]` (Beijing Institute of Graphic Communication)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们提出并实现了 DAEP 系统，在 NLPCC 2026 Shared Task 1 Track 3 任务中对 50 候选视频进行检索并定位答案时段。

**💡 创新点**

创新点在于利用任务给出的简单/复杂标签在推理时动态规划多模态证据权重、检索宽度、边界阈值和重排序强度，实现难度感知的证据规划。

**🔧 技术方法**

采用了多模态编码（mBERT、CLIP ViT‑B/32、上下文编码），基于视频字幕单元的匹配、Top‑K 聚合、NMS、跨模态重排序，并使用 MLP 规划器与可学习权重。

**📊 数据集**

使用了 NLPCC 2026 Shared Task 1 的 DA‑TAGVC 数据集，包括中英两语、简单/复杂两难度的医学教学视频与问答。

**📈 对比分析**

与十支参赛系统对比，DAEP 在 R@1|mIoU、R@10|mIoU、R@100|mIoU 及平均分均排名第一，平均分为 0.2728，领先第二名约 9.4%。

**⚠️ 局限性**

局限性包括对相似程序的混淆、叙述与动作边界漂移、字幕与视觉不匹配及双语术语差异，且对更细粒度动作识别与术语标准化的需求未满足。

---

## 245. PyFlow: An Inter-procedural Static Analysis Framework for Python

**arXiv ID:** 2608.07026 | [PDF](https://arxiv.org/pdf/2608.07026v1)

**作者:** Zinan Gu `[一作]` (Zhejiang University), Peisen Yao `[通讯]` (Zhejiang University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种通用的基于IFDS的Python静态分析框架PyFlow，旨在解决Python中的跨过程静态分析问题。

**💡 创新点**

创新点在于提供了一个多阶段的中间表示管道和一个通用的IFDS求解器，允许分析开发者仅需实现数据流语义，框架自动构建超图、执行固定点迭代和缓存摘要。

**🔧 技术方法**

使用了IFDS（Interprocedural Finite Distributive Subset）技术进行数据流分析，并实现了污点分析。

**📊 数据集**

在合成基准和真实世界基准上评估了PyFlow，合成基准包含240个程序（120个易受攻击和120个修补过的），真实世界基准包含62个开源Python项目中的108个CVE。

**📈 对比分析**

与八种现有的Python SAST工具（如DevSkim、Dlint、Bandit等）进行比较，PyFlow在合成基准上获得了最佳的召回率和F1分数，在真实世界基准上也达到了最高的召回率（48.1%）和F1分数（57.5%），同时保持了与基于污点的引擎相竞争的精度。

**⚠️ 局限性**

限制在于Python的动态特性使得调用图构建不够精确，且在处理复杂数据模型时可能导致性能下降。

---

## 246. Degradation-Aware Prompt Learning with Cross-Modal Compensation for Adverse Weather Removal

**arXiv ID:** 2608.06939 | [PDF](https://arxiv.org/pdf/2608.06939v1)

**作者:** Wanshu Fan `[一作]` (Dalian University), Jinshan Pan `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于跨模态提示的降雨、雾霾、雪雾和雨滴等恶劣天气图像恢复网络——DCMPC‑Net。

**💡 创新点**

创新点在于：① 通过冻结的预训练视觉‑语言模型（LLaMA）与文本编码器生成“降解感知跨模态提示”，将语义先验与视觉特征融合；② 在恢复骨干中加入 Prompt‑Guided Attention Alignment Module（PGAAM）实现语义提示与局部降解特征的精确对齐；③ 引入 Dual Feature Compensation Module（DFCM）对降解痕迹与场景结构进行分离与补偿，提升细节重建与结构保真。

**🔧 技术方法**

技术手段包括：Transformer‑based encoder‑decoder、跨模态注意力融合、提示生成与对齐、双分支特征补偿、频域损失、交叉熵等多项损失；使用预训练的 LLaMA、Multilingual‑E5 文本编码器和图像编码器；网络整体可端到端训练，提示特征在推理阶段缓存。

**📊 数据集**

使用的公开基准包括 Snow100K（S/L）、Outdoor‑Rain、RainDrop、All‑Weather 等多种恶劣天气数据集。

**📈 对比分析**

与 Histoformer、GridFormer、T3‑DiffWeather、CyclicPrompt、TransWeather 等多种最新单/多任务恢复方法在 PSNR/SSIM 上进行对比。DCMPC‑Net 在四个基准上均取得最高 PSNR（Snow‑S 38.21 dB、Snow‑L 32.35 dB、Outdoor‑Rain 32.49 dB、RainDrop 33.08 dB）和最高 SSIM，平均 PSNR 约 33 dB，平均 SSIM 0.948，显著优于现有最优方法。

**⚠️ 局限性**

局限性：① 在极端降雨密集、雨滴覆盖细节结构的极端场景下，提示特征与真实场景特征混淆导致残留伪影；② 全流程仍需预训练视觉‑语言模型的文本提取，推理时额外开销较大；③ 对不同语言或不常见天气形态的泛化能力受限。

---

## 247. LoRAScan: Detecting Backdoor Prompts in Low-Rank Adapters for Large Language Models via Down-Projection Activation Spikes

**arXiv ID:** 2608.06795 | [PDF](https://arxiv.org/pdf/2608.06795v1)

**作者:** Doniyorkhon Obidov `[一作]` (Michigan Technological University), Kaichen Yang `[通讯]` (Michigan Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种在第三方 LoRA 适配器中进行推理时检测并拒绝后门触发输入的防御方法 LoRAScan。

**💡 创新点**

创新点在于：1) 仅监控少量低方差的 LoRA 下投影激活；2) 通过检测激活峰值集中度来识别后门触发，而不修改适配器参数；3) 只需一次前向推理即可完成检测，提升效率。

**🔧 技术方法**

使用技术包括：低秩适配器监测、激活峰值统计（max/平均比值）、低方差插入点选择、基于中位数绝对偏差的阈值设定，以及单次前向推理检测。

**📊 数据集**

使用的数据集主要为 BackdoorLLM benchmark，涵盖 DeepSeek、Llama‑2、Llama‑3、OpenChat‑3.5、Vicuna 等模型；阈值校准采用 Stanford Alpaca 等干净提示。

**📈 对比分析**

与 ONION、BEAT、ConfGuard（模型保留防御）以及 Fine‑tune、CROW、WANDA、量化、Obliviate、Decoding、CleanGen（模型修改防御）比较；LoRAScan 在 12,475 个攻击样本上实现 98.49% 的攻击拒绝率、96.59% 的干净通过率，检测延迟约 29.69 ms，明显优于多数对手。

**⚠️ 局限性**

局限性：仅针对自回归 Transformer LLM；不适用于非 Transformer 或多模态模型；需白盒访问内部激活，黑盒 API 客户端无法直接使用；未实现输入清理，仅通过拒绝策略降低误报。

---

## 248. CertBind from Multimodal Connectivity to Certifiable Retrieval Decisions

**arXiv ID:** 2608.06516 | [PDF](https://arxiv.org/pdf/2608.06516v1)

**作者:** Shuheng Cao `[一作]` (University of California, San Diego), Fan Gu `[通讯]` (Changsha University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在冻结的多模态编码器与轻量级连接器图上，提出了一套多尺度可证合成框架 CertBind，能够从节点级的任务识别到边级的合约检查，再到路径级的重构和查询级的 top‑k 认证，实现对检索决策的可证保证。

**💡 创新点**

创新点包括：①利用本地 anchor 解决节点尺度的残余尺度问题，给出判定是否能识别原生检索任务的精确边界；②在边尺度引入合约感知的 conformal 等价层和 Holm 步骤，控制全图的错误标记；③提出路径级重构预算 κ1/2，量化共享失败对多条路径的影响并给出严格的恢复半径；④在查询尺度将恢复半径映射为覆盖 top‑k 候选集，并给出点级证书与 Abstain 的判定规则；⑤构建“保留优先”决策策略，实现直接、认证与放弃的自动化映射。

**🔧 技术方法**

主要技术：冻结多模态编码器（如 CLIP、CLIP‑Text‑Audio）、轻量级连接器图、anchor‑注册与识别、合约级的 conformal 等价检验、Holm 多重检验、路径级重构预算（基于多数路径跨越）、坐标中位数恢复、有限样本 conformal 置信半径、整数规划与 Menger 定理用于路径预算优化。

**📊 数据集**

使用的数据集包括：CLIP 原生图像‑文本检索（用于评估 R@1），C‑MCR 与 Ex‑MCR 共享/原生路由实验；Clotho 语音-文本子集（用于跨模态文本‑音频检索）；以及在生产环境中收集的通过 fallback 进行的检索样本。

**📈 对比分析**

在原生 CLIP 图像‑文本检索中，C‑MCR 共享路由将 R@1 从 0.524 降至 0.290，验证了保留效果；在生产 fallback 下，恢复率达到 0.963±0.002，且通过分支的 no‑harm 记录为 1.000；在 Clotho 文本‑音频检索中，C‑MCR 与 Ex‑MCR 分别实现 R@1 0.168 与 0.180，表明跨模态扩展能力。整体上，CertBind 能在保持原生性能的同时，提供受控的恢复与证书。

**⚠️ 局限性**

局限性包括：① 依赖正交图表模型和 anchor 覆盖，若锚点稀疏或条件不佳，识别边界失效；② 合约级检查假设在合约内可交换，现实中可能不成立；③ 路径预算优化为 NP‑hard，实际部署需近似或仅限于边缘相互不干扰的路径；④ 主要针对余弦检索，非线性或非对称连接器可能需要新的识别与校准理论；⑤ 需要预先声明合约与校准数据，增加工程复杂度。

---

## 249. ControlRef: Efficient Layout-Guided Multi-Instance Generation via Anchored 4D-RoPE

**arXiv ID:** 2608.06878 | [PDF](https://arxiv.org/pdf/2608.06878v1)

**作者:** Yunkai Yang `[一作]` (Sun Yat-Sen University), Runmin Dong `[通讯]` (Sun Yat-Sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ControlRef 框架，实现了布局引导的多实例图像生成，解决了传统方法中画布填充导致的空间‑频率权衡问题。

**💡 创新点**

创新点包括：① Unified Instance‑Layout Control (UILC) 注意力掩码，实现实例级隔离和区域绑定；② Anchored 4D‑RoPE，直接将参考图像与布局 token 绑定到绝对几何中心；③ 通过消除冗余全分辨率填充，显著提升推理效率。

**🔧 技术方法**

技术手段：基于 FLUX.2 9B 的多模态 Diffusion Transformer；LoRA 微调；4D‑RoPE（z、y、x、t）位置编码；UILC 注意力掩码；参考图像与布局 token 的绝对坐标对齐。

**📊 数据集**

使用的主要数据集：IMIG‑100K 进行微调；LayoutSAM‑Eval 进行布局对齐评估；LAMICBench++ 进行多实例一致性与质量评估。

**📈 对比分析**

与多种开源（LAMIC、XVerse、UNO、MS‑Diffusion 等）和部分商业模型（GPT‑4o‑Image、Nano Banana 等）进行对比；ControlRef 在 ITC、IPS、AVG 等指标上均位居前列；推理延迟平均降低 80%+，内存占用降低 50%+，展示出卓越的效率与质量。

**⚠️ 局限性**

局限性：在全局视觉一致性（ITC、AES 等）上仍略逊于部分商业模型；目前仅针对单帧图像，视频生成及更细粒度属性路由尚未展开；需要进一步验证在更复杂布局与多模态条件下的稳定性。

---

## 250. Progressive Alignment of Recommender Foundation Model through Multi-Phase Post-Training

**arXiv ID:** 2608.06792 | [PDF](https://arxiv.org/pdf/2608.06792v1)

**作者:** Oseong Choi `[一作]` (NAVER WEBTOON), Taeyeong Jang `[通讯]` (NAVER WEBTOON)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了三阶段递进式后训练框架，将任务适配与业务指标对齐分离，先用线性探测(LP)冻结预训练模型后再全参数微调(FFT)，随后通过强化学习(基于GRPO或DPO)使用学习的奖励模型对策略进行业务对齐。

**💡 创新点**

创新点在于：①结构化的三阶段训练流程避免单阶段训练导致的灾难性遗忘；②使用密集隐式反馈训练基准策略，再用稀疏业务指标构建奖励模型，避免直接用稀疏目标导致的泛化不足；③将奖励模型作为对齐信号而非直接服务策略，从而保持强区分度。

**🔧 技术方法**

技术包括：预训练序列基础模型（HSTU）、轻量化任务特定编码器、线性探测+全参数微调、GRPO（基于组相对策略优化）与DPO（直接偏好优化）、印象倾向校正、Ordinal回归奖励模型、学习率差异控制。

**📊 数据集**

使用了韩国Webtoon平台的真实交互日志，包含约7M参数的基础模型及数百万用户的长序列行为数据。

**📈 对比分析**

与单阶段SFT、仅使用奖励模型或直接奖励对齐的基线相比，三阶段框架在离线Rank‑NDCG、Funnel‑NDCG上提升约6–12%；在线A/B实验显示对比控制模型提升点击率、参与度和付费完成率，GRPO+奖励模型在深度参与指标上最高。

**⚠️ 局限性**

局限性包括：奖励模型的质量对对齐效果敏感；GRPO训练需要候选集抽样，可能在极大候选空间下效率低；方法在稀疏业务信号极端稀缺时仍可能面临数据不足问题。

---

## 251. Superlogarithmic-Rank Matrix Rigidity for the Walsh-Hadamard Transform

**arXiv ID:** 2608.06592 | [PDF](https://arxiv.org/pdf/2608.06592v1)

**作者:** Josh Alman `[一作]` `[通讯]` (Columbia University), Josh Alman (Columbia University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明Walsh–Hadamard矩阵在3域上，当目标秩为⌊log²N/80⌋时，其矩阵刚性至少为N²/100，即至少需要改动1/100比例的矩阵元素才能将秩降至该级别。

**💡 创新点**

首次给出显式矩阵族在超对数秩（log²N）下的常数分数刚性下界，从而逼近Razborov提出的通信复杂度下界目标。

**🔧 技术方法**

结合随机加权列探测、复数相位编码、协方差与重叠的组合不等式、Walsh列的正交展开、Bessel不等式以及随机子空间交叉概率上界等概率与线性代数技术，构造并分析了对应的向量序列。

**📊 数据集**

无实测数据集，研究完全是理论性的。

**📈 对比分析**

与之前仅达到Θ(log N)秩下界的代码和Cauchy矩阵相比，本结果将可实现的秩提升至log²N，并在误差比例上达到N²/100的下界，表明即使仅改动10%的元素，矩阵仍保持高秩。

**⚠️ 局限性**

仍未达到Razborov所需的极高秩下界（2^{(log log N)^ω(1)}），误差比例仅为1/100而非接近1/2，且证明仅适用于大规模2的幂阶N（n≥2000）。

---

## 252. Understanding Differentiable Embeddings Through Differential and Integral Geometry

**arXiv ID:** 2608.06809 | [PDF](https://arxiv.org/pdf/2608.06809v1)

**作者:** Xinyu Zhang `[一作]`, Klaus Mueller `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个统一的几何框架，解释并量化可微降维嵌入的局部和全局可靠性；通过该框架推导并验证投影图、曲率、霍洛尼等诊断的本质；进一步给出了曲率的误差上限和霍洛尼的可积性判定，并展示了该框架在单细胞和图像数据上的应用；

**💡 创新点**

发现所有现有的局部诊断（投影图、映射连续性得分）和全局诊断（传输一致性）都可归结为同一几何对象（连接）的两种读数，并证明它们在信息上相互独立；提出曲率和霍洛尼这两个新的几何证书，并证明曲率提供局部误差上限，霍洛尼检测路径依赖；

**🔧 技术方法**

基于隐式微分、自动微分计算的Jacobian与Hessian；利用曲率（第二阶导数）和霍洛尼（闭环积分）来评估嵌入；对非线性目标函数（如t‑SNE、UMAP、MDS）做局部最优化并求解对应的导数；实现了流场积分、闭环检测与逆向回传；

**📊 数据集**

单细胞PBMC3k、图像FashionMNIST与COIL‑20、人工合成数据（swiss‑roll、MDS/Isomap对比）以及公开的数字手写体数据集；

**📈 对比分析**

与传统的可解释性度量（信任度、连续性、邻域排名）以及直接重新优化的基线进行对比；曲率预测的单细胞嵌入误差Spearman相关系数在0.963–0.999之间；霍洛尼在t‑SNE/UMAP上显著非零，而在MDS上几乎为零；在路径追踪实验中，利用流场积分的计算量约为逐帧重新优化的1/3–1/4，且保持相似的误差；

**⚠️ 局限性**

仅适用于可微嵌入，无法处理非可微或随机映射；对优化目标的参数（如perplexity、正则化）敏感；霍洛尼的判定取决于闭环是否跨越鞍点，若闭环不满足条件可能产生误判；仅能检测到分支缺失/多值性，无法揭示数据集尺度和样本稀疏造成的邻域失真。

---

## 253. AutoIntervene: Calibrated Intervention for Action-Chunking Imitation Learning Policies

**arXiv ID:** 2608.07065 | [PDF](https://arxiv.org/pdf/2608.07065v1)

**作者:** Jinhe Tang `[一作]` (University of Sydney), Weiming Zhi `[通讯]` (University of Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种在线自适应框架 AutoIntervene，用于在部署时检测并自动切换控制权，收集针对性干预数据以提升动作分块式模仿学习策略的可靠性。

**💡 创新点**

创新点在于：①结合阶段局部与全局视觉-动作支持评估实现双向控制切换；②使用实验保留的成功轨迹对切换阈值进行离线量化校准；③将操作员干预段作为有针对性的 DAgger‑style 训练样本，显著降低所需干预时间。

**🔧 技术方法**

核心技术包括：视觉-动作检索记忆（基于多视角 DINOv3 编码器）、相似性与动作风险评分、模式特定检索窗口、阈值量化与双向决策逻辑、以及基于干预数据的在线策略更新。

**📊 数据集**

使用了九个真实双臂操作任务的数据集，包含 30 条初始演示轨迹（其中 6 条用于校准）以及在多轮干预中收集的干预轨迹；实验涵盖不同动作生成头（ACT、Diffusion Policy、Flow Matching）和长周期任务。

**📈 对比分析**

与手动切换、完整额外演示以及 LazyDAgger / RND‑DAgger 等基线对比，AutoIntervene 在平均成功率上提升 49.1%（从 30.9% 到 80%），并且额外干预时间仅为完整演示的 26%，在双向切换可靠性和误触率上均优于前置监控器。

**⚠️ 局限性**

局限性包括：依赖预先收集的成功演示来构建支持记忆，对视觉特征分布变化敏感；阈值校准需要保留一组固定演示；在极端视觉噪声或非视觉感知失败时，支持评分可能失效；并且在大规模高维任务中检索与评分的计算成本仍需进一步优化。

---

## 254. SkillEval: Decomposing Agent Skill Quality into Interpretable Signals

**arXiv ID:** 2608.06891 | [PDF](https://arxiv.org/pdf/2608.06891v1)

**作者:** Jiahui Han `[一作]` (Xi'an Jiaotong University), Ninghao Liu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 SkillEval，一种可解释的框架，用来评估并诊断可重用的 agent 技能文档质量。

**💡 创新点**

创新点在于：①将技能质量拆解为可解释的四个维度（适用性、内容质量、执行指导与鲁棒性），②通过对比正负技能对学习线性评分方向，并通过投影给出可解释的分数，③在评分过程中主动消除长度、格式等无关偏差，提升评分的语义性。

**🔧 技术方法**

技术手段包括：冻结大型语言模型（如 Qwen3.5-9B）获取隐藏层表示，构造控制的正负技能对，求取差分向量并正交化去除共因子，随后投影并标准化得到 z‑score；此外使用线性回归将评分映射到下游任务的提升幅度。

**📊 数据集**

数据集包括 1,647 个公开与手工构造的技能文档，用于训练与验证评分方向；下游评估则使用 SkillsBench benchmark，包含多任务的 pass‑rate uplift 数据。

**📈 对比分析**

在验证集上，所有六个正向维度的正负技能均能被清晰区分，平均分数差距从 0.75 到 1.36；与下游任务的 pass‑rate uplift 的 Pearson 相关系数在 0.779–0.787 之间；使用 SkillEval 指导的技能修订可使平均通过率从 18.6% 提升至 48.1%，提升幅度达 29.5%。

**⚠️ 局限性**

局限性包括：①评价仅基于文本层面的特征，无法捕捉执行时动态交互导致的质量差异；②对特定格式（YAML 前置层）和工具链的依赖可能限制在更广泛的技能生态中的适用性；③尽管对长度偏差做了正交化，但仍可能存在其他未检测的文本属性偏差；④在不同模型或跨任务场景下的普适性仍需进一步验证。

---

## 255. Linguistic Pattern Based Optimization of Economic and Spatial Uniformity Criteria in Facility Layout Problems

**arXiv ID:** 2608.07011 | [PDF](https://arxiv.org/pdf/2608.07011v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 256. HLSmith: An Expert-Guided Agentic Framework for C/C++-to-HLS Translation

**arXiv ID:** 2608.06791 | [PDF](https://arxiv.org/pdf/2608.06791v1)

**作者:** Yuebo Luo `[一作]`, Caiwen Ding `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

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

## 257. Tight Inapproximability of Max Independent Set in Triangle-Free Graphs

**arXiv ID:** 2608.06493 | [PDF](https://arxiv.org/pdf/2608.06493v1)

**作者:** Édouard Bonnet `[一作]` `[通讯]` (University of Lyon), Édouard Bonnet (University of Lyon)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了在三角形自由图以及更广泛的包含环的图族 ℱ‑子图自由图中，最大独立集的近似因子若小于 n^{1/2}-ε（或相应的 μ(ℱ)-ε）则导致 NP⊆BPP；

**💡 创新点**

通过将 Moser–Tardos 重采样算法与 Haeupler‑Saha‑Srinivasan 的概率上界结合，得到新的硬度阈值，并将结果推广到所有含环的图族；

**🔧 技术方法**

采用 Moser–Tardos 重采样、Motzkin–Straus 定理、Haeupler‑Saha‑Srinivasan 的概率分析以及 Markov 与 union bound 等概率与组合技术；

**📊 数据集**

该研究为理论证明，不使用任何实验数据集；

**📈 对比分析**

相较于之前的 n^{1/4}-ε 难度结果，论文将硬度指数提升至 1/2，几乎与已知的 n^{1/2} 近似算法匹配，表明在这些图族上实现了最优的近似因子；

**⚠️ 局限性**

仍未完全解决更一般的图族（如所有无环图、偶数环图等）的近似阈值，且证明基于 NP⊆BPP 的弱化假设，存在进一步提升与更广泛适用性的空间。

---

## 258. Fast and Accurate: An Adaptive VLA Inference Framework through Environment-aware Model Selection

**arXiv ID:** 2608.06434 | [PDF](https://arxiv.org/pdf/2608.06434v1)

**作者:** Yuewei Sun `[一作]` (Huawei Technologies), Yuxin Ma `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 1916 | [OpenAlex ID](https://openalex.org/A5111996979)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了EMS框架，实现快速（System 1）与慢速（System 2）视觉‑语言‑动作模型的分离式异步推理，并通过轻量化的环境感知切换模块在运行时动态选择合适的模型执行。

**💡 创新点**

创新点在于：①完全解耦的双系统架构，支持插件式升级；②基于强化学习的环境感知自适应切换策略；③在保持大模型规划优势的同时，利用轻量模型实现接近100 Hz的闭环控制，显著提升任务完成速度。

**🔧 技术方法**

技术手段包括：多模态视觉‑语言预训练模型（如PI0、BCVILT、ACT），动作分块与单步推理，基于DQN和IQL的离线/在线切换策略训练，动作融合以平滑切换边界。

**📊 数据集**

主要使用LIBERO仿真基准（四个子套件），以及Realman RM75‑6F双臂机器人现场数据进行真实世界验证。

**📈 对比分析**

对比方法包括单体慢速模型、单体快速模型、固定切换策略等。EMS在LIBERO上平均成功率达92.4%，仅以0.15的切换比例即可实现近似慢速模型的成功率，同时有效提升至93.4 Hz；在真实双臂实验中，任务完成时间从29 s降至23 s，成功率提升至70%。

**⚠️ 局限性**

局限性包括：仍需稀疏调用慢速模型以保证成功率；切换策略对状态表征敏感，可能在极端动态环境下表现不稳；在真实系统中，感知与低层控制的时延仍对高速闭环产生影响。

---

## 259. TRIBE: Predicting Team Performance via Communication Behavior Ensembles

**arXiv ID:** 2608.06926 | [PDF](https://arxiv.org/pdf/2608.06926v1)

**作者:** Ali Jalal-Kamali `[一作]` (USC Institute for Creative Technologies), Fred Morstatter `[通讯]` (USC Information Sciences Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

TRIBE通过对团队对话进行主题建模与聚类，实时预测团队表现并分析AI干预对行为动态的影响。

**💡 创新点**

其创新点在于实现无任务特定、早期可预测的行为族分类，并通过时间序列与马尔可夫分析揭示代理干预对团队行为轨迹的改变。

**🔧 技术方法**

技术组合包括LDA/NMF主题建模、k‑means/GMM聚类、线性回归、存活分析、Markov链建模，以及对比LLM（Llama）提示实验。

**📊 数据集**

评估数据来源于ASIST Minecraft 场景（Study 3）以及四个不同任务的数据集——Study 4、DELI、GAP，均为公开对话与绩效记录。

**📈 对比分析**

与传统基于结果的评估和LLM方法对比，TRIBE在四个域中实现18–44% 的R²，早期预测准确率可达90%（50% 进度），并在速度上提升约24×，表现优于Llama提示实验。

**⚠️ 局限性**

主要局限包括对团队质量分布偏差的敏感性、依赖足够的沟通表达空间以及对极端复杂任务中细粒度行为识别的不足。

---

## 260. Discovering Conceptual Metaphors Across Topics and Media Types

**arXiv ID:** 2608.06652 | [PDF](https://arxiv.org/pdf/2608.06652v1)

**作者:** Alexandria Leto `[一作]` (University of Colorado Boulder), Maria Leonor Pacheco `[通讯]` (University of Colorado Boulder)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种无监督方法，先从语料中提取语言隐喻，再通过LLM生成的隐喻解释与源-目标属性约束进行结构化聚类，从而推断出概念隐喻并用于分析媒体偏见和播客言论。

**💡 创新点**

创新点在于将LLM的生成能力与CMT约束的结构化K‑means聚类相结合，首次实现完全无监督地从大量文本中高效抽取并聚类概念隐喻，且聚类结果可直接用于政治倾向预测。

**🔧 技术方法**

核心技术包括：使用Qwen3‑8B LLM进行隐喻检测与解释生成；spaCy解析依赖路径抽取候选隐喻；带不能链接约束的K‑means聚类（Soft Can‑Link）；以及使用文本相似度与离散属性（目标组、图像模式）作为聚类特征。

**📊 数据集**

所用数据集包括：LCC隐喻数据集（约1万条），移民推文、枪支控制与堕胎新闻各1000条，左右倾向的94集美国政治播客转录（共约4.65k条隐喻）。

**📈 对比分析**

在隐喻检测上，LLM的F1≈81.6，略优于RoBERTa监督模型；在聚类评估中，结构化聚类在目标/源属性纯度和政治极化预测F1（约54–55%）均优于普通K‑means；聚类纯度提升至约4–5倍，预测准确性提升约2–4%。

**⚠️ 局限性**

主要局限包括：仅覆盖英语、美国中心的公开语料，LLM生成文本可能放大训练数据偏见；播客转录使用Whisper但未评估其准确性；样本规模有限，未涵盖更广泛的媒体类型或多语言场景。

---

## 261. Risk-Aware Decision Policies for Agents Under Noisy Perception

**arXiv ID:** 2608.06420 | [PDF](https://arxiv.org/pdf/2608.06420v1)

**作者:** David Szczecina `[一作]` `[通讯]` (University of Waterloo), David Szczecina (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实验了一个基于噪声感知的人工生命捕食-猎物模拟，比较了盲目信任、怀疑与主动验证三种决策策略。

**💡 创新点**

创新点在于将感知建模为噪声标签问题，提出不确定性感知策略并揭示行为模式转变与鲁棒性提升。

**🔧 技术方法**

使用了基于贝叶斯推断的感知模型、固定决策策略、能量状态管理的二维Agent模拟。

**📊 数据集**

使用自生成的二维空间环境，包含50个Agent、150个食物和20个捕食者，随机初始化20个实验种子。

**📈 对比分析**

通过平均生存时间、食物获取量、死亡原因比例等指标对比，主动验证策略在高噪声下保持最高生存率并显著降低误接近率。

**⚠️ 局限性**

局限在于未加入学习/自适应机制，环境与感知模型过于简化，策略固定且未考虑动态不确定性变化。

---

## 262. DGEMM with Ozaki Scheme I/II on FP4 Tensor Cores: A Base-13 E2M1 Limb Representation

**arXiv ID:** 2608.06812 | [PDF](https://arxiv.org/pdf/2608.06812v1)

**作者:** Shun-ichiro Hayashi `[一作]` (Nagoya University), Takahiro Katagiri `[通讯]` (Nagoya University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出利用FP4 Tensor Cores在Ozaki scheme I/II框架下，通过13进制FP4 limb表示实现FP64矩阵乘法的无误差整数计算，从而在低精度硬件上模拟高精度计算。

**💡 创新点**

创新点在于将FP4的整数特性与基于13进制的位数分解相结合，构建错误无效的整数GEMM，并在此基础上实现Ozaki scheme I/II，同时采用直接CRT和内核融合技术显著提升性能。

**🔧 技术方法**

使用的技术包括FP4 (E2M1) Tensor Cores、Ozaki scheme I/II、13进制FP4 limb表示、直接CRT重构、Triton DSL编写的GPU内核、以及CUDA 12/13与cuBLAS对比。

**📊 数据集**

用于评估的不是公开数据集，而是由随机生成的实数矩阵（±(1+u)·2^e形式）构成的合成数据集，用于测试不同动态范围下的精度和性能。

**📈 对比分析**

方法通过与cuBLAS DGEMM、GEMMul8-FP8（FP8版Ozaki scheme II）以及GEMMul8-INT8（INT8版）在RTX PRO 6000 Blackwell上进行对比；在FP64模拟任务中，OzII-FP4的计算阶段比GEMMul8-FP8快约1.1–1.2×，整体上在问题规模16384^3时实现0.96×的总时延，且在中小规模下与之相当。

**⚠️ 局限性**

局限性包括：共享指数量化导致对大动态范围矩阵的精度衰减；需要大量GEMMs（Q≈75）和模数，受FP4吞吐率高低限制；实现复杂，依赖内核融合与Triton优化，且在硬件缺乏FP4支持时不可行。

---

## 263. ReGraph: Learning to Generate Recipe Graphs from Food Images

**arXiv ID:** 2608.06917 | [PDF](https://arxiv.org/pdf/2608.06917v1)

**作者:** Guoshan Liu `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了大规模烹饪流程图数据集 ReGraph，并基于该数据集设计了两阶段的 Recipe Graph Learning (RGL) 框架，用结构化图来直接从食物图像生成细粒度的烹饪工作流程。

**💡 创新点**

创新点包括：① 通过细粒度实体、属性与多类型关系构建的图结构显式表示烹饪过程；② 引入 Recipe Reasoning Chain‑of‑Thought (RR‑CoT) 作为中间监督，帮助模型先进行程序化分解；③ 采用基于 GRPO 的强化学习，使用相对提升奖励 (RIR) 与格式奖励显式提升图生成的语义与结构质量。

**🔧 技术方法**

使用的大型多模态模型（Qwen3‑VL、InternVL3）在监督微调后结合 Group Relative Policy Optimization (GRPO) 进行强化学习；实现 RR‑CoT 生成、图结构序列化、相对提升奖励 (RIR) 与轻量级格式奖励。

**📊 数据集**

核心数据集为 ReGraph（来自 Recipe1M，10,000 份，318,773 个实体和 391,051 条关系），并在实验中对比传统文本生成数据集与其它食品图数据库（如 FoodKG、Flow Graph Corpus）。

**📈 对比分析**

评估采用确定性、schema‑aware 的 Canonical Matching，衡量实体与关系的 F1。与传统文本生成方法（SacreBLEU、ROUGE‑L）对比，RGL 在实体 F1 约 30–31%，关系 F1 约 8–9%，显著高于单阶段或无监督方法，显示出在结构化流程生成上的优势。

**⚠️ 局限性**

局限性：① 单张图像无法观察到所有准备操作，导致缺失或错误预测；② 仅依据单一参考图评估，忽略多种合理流程的多样性；③ 细粒度状态与关系预测仍存在较大难度；④ 仅在 8B 级别模型实验，缺乏对更大模型的验证。

---

## 264. WebGrader: Training LLMs for Web Development with Self-Evolving Programmatic Grader

**arXiv ID:** 2608.06474 | [PDF](https://arxiv.org/pdf/2608.06474v1)

**作者:** Boshui Chen `[一作]` (Beijing Institute of Technology), Shaolei Zhang `[通讯]` (Renmin University of China)

**通讯引用:** 214 | [OpenAlex ID](https://openalex.org/A5031466943)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种自演进、可执行的Web评估器WebGrader，自动将开放式网站需求转化为可执行的浏览器交互流（Flow Contract），并基于执行证据给出可用作RL奖励的判定。

**💡 创新点**

创新点在于：①将需求先行解析为可执行交互流并与生成网站动态对齐；②利用残差驱动的NAS式循环将判定器细化为可组合的SkillGraph；③将演进后的评估器冻结，作为高质量可执行奖励，为LLM网站生成提供功能性监督；④在RL过程中不需要额外的判定网络，直接从浏览器执行获得奖励。

**🔧 技术方法**

使用技术包括：自然语言需求解析、Playwright浏览器自动化、可执行的Flow Contract、证据收集（截图、DOM、网络响应、持久状态）、残差归因、神经架构搜索启发的技能变异与筛选、路由SkillGraph、GRPO强化学习。

**📊 数据集**

使用数据集：WebGen-Verifier-100（带故障变体的验证环境）、WebGen-Bench（公开需求与功能案例）、WG-core-250（HTMLBench-400子集）以及训练时的WebGen-Instruct。

**📈 对比分析**

与基线对比：在WebGen-Bench上，WebGrader-RL的功能成功率为52.01%，比匹配的VLM+Base-Script-RL提升7.88点，且在appearance上仅差0.02点；在WG-core-250上提升6.58分、TC%提升6.41；在同类模型o4-mini、DeepSeek-v4-flash上均表现更好；整体表明可执行奖励显著提升功能性。

**⚠️ 局限性**

局限性：仍落后于最强的封闭源生成器；评估器演进依赖于带有故障标签的验证环境，需人工构造；主要针对React/Vite项目，迁移到其他前端框架或后端技术尚未验证；对非常复杂的多页面或异步状态依赖的需求可能仍难以完全捕获。

---

## 265. Preventive Care Recommendations by Large Language Models

**arXiv ID:** 2608.06379 | [PDF](https://arxiv.org/pdf/2608.06379v1)

**作者:** Eden Avnat `[一作]` (Tel Aviv University), Raja-Elie E. Abdulnour `[通讯]` (Harvard Medical School)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究评估了七种大型语言模型（LLM）在预防性医疗服务优先级排序中的表现，并与2017年张等人的医师调查结果进行对比；

**💡 创新点**

创新点在于首次将LLM用于多服务优先级排序任务，结合时间限制与生命年收益评估，探讨其对现有人类偏见的放大或缓解作用；

**🔧 技术方法**

使用的技术包括LLM推理（如GPT‑5‑mini、Gemini‑2.5‑flash等）、系统消息与多结构提示，计算Spearman相关、Consensus‑Stratified Agreement（CSA）及Life‑Years‑Gained‑Per‑Choice（LYGPC）；

**📊 数据集**

数据集来源于张等人2017年的医师调查（137名医师）和其验证的预防服务生命年框架，本文通过生成等价医师画像模拟数据；

**📈 对比分析**

比较方法是将LLM生成的优先级评分与医师的评分进行Spearman相关、CSA和LYGPC对比；性能显示LLM与医师高度一致（平均Spearman≈0.83），在极端共识区CSA达94%，但在中等共识区显著低于医师，且LLM在生活方式干预上被低估；部分模型在LYGPC上优于医师，但总体未达到最佳边界；

**⚠️ 局限性**

局限性包括：时间差异导致潜在知识泄漏，生成的医师画像缺乏联合分布，缺少原始医师数据，仅涵盖两例假设患者，且未评估领域特定LLM，限制了外推性和实际临床适用性。

---

## 266. From Prompting to Describing: A Cross-Cultural Study of Language for AI-Generated Music

**arXiv ID:** 2608.06634 | [PDF](https://arxiv.org/pdf/2608.06634v1)

**作者:** Sangheon Park `[一作]` (Georgia Institute of Technology), Claire Arthur `[通讯]` (Georgia Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究对比了文本到音乐生成系统的提示文本与听者对生成音频的自由描述，构建了人类驱动的音乐提示词汇表，并分析了两者在词汇、类别和语义上的差异。

**💡 创新点**

创新点在于首次系统量化提示与描述之间的结构差异，发现提示侧重流派和叙事，而描述侧重乐器、情绪和音乐理论，并揭示叙事性提示导致语义不匹配。

**🔧 技术方法**

采用人类注释构建的七分类词汇表、GPT‑5.4自动标注、词频与向量相似度分析（Sentence‑BERT）以及混合效应回归等技术。

**📊 数据集**

使用了Casini等人公开的200条Udio真实提示生成音频，并收集了来自70名英语听众和78名韩语听众的共2,624条自由描述。

**📈 对比分析**

通过类别存在率、词汇密度、词汇传播率和句向量余弦相似度等多层次指标进行比较，发现提示与描述在类别分布上存在显著偏差，语义对齐度因提示内容而异，且叙事性提示最易导致低对齐。

**⚠️ 局限性**

主要局限包括样本来源单一（仅Udio系统）、跨语言比较受招募渠道差异影响、LLM标注可能引入噪声，以及缺乏对多种生成系统的普适性验证。

---

## 267. NTDH: Complex Reasoning for Comprehensive Affective Analysis

**arXiv ID:** 2608.06425 | [PDF](https://arxiv.org/pdf/2608.06425v1)

**作者:** Tianlei Zhu `[一作]` (Columbia University), Sophia Ananiadou `[通讯]` (University of Manchester)

**通讯引用:** 17864 | [OpenAlex ID](https://openalex.org/A5077976343)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了NTDH框架，将综合情感分析任务统一为一次性可解释的推理过程，并通过自然化、容差验证、领域细化和方向性提示来构造高质量的推理训练数据，随后对Qwen3-8B进行SFT+GRPO两阶段训练；

**💡 创新点**

创新点在于：①将多种情感任务（情感强度回归、多标签情感分类、情感极性回归与序数分类）统一为结构化推理生成；②通过自然化使训练目标始终与金标准一致；③设计容差感知的判定门，消除回归误差泄漏；④基于情感科学的领域细化策略与方向性提示实现无答案泄漏的推理修正；

**🔧 技术方法**

使用技术包括：大规模语言模型Qwen3‑8B、结构化推理生成（Chain‑of‑Thought）、容差门判定、领域自适应细化策略、方向性提示、监督微调（SFT）与基于组相对策略优化（GRPO）的强化学习；

**📊 数据集**

使用数据集为SemEval‑2018 Task 1的四个子任务（情感强度回归、情感极性回归、序数极性分类、多标签情感分类），共16 302条训练记录（约1/14 EmoLLM的规模）；

**📈 对比分析**

与EmoLLM等基线相比，NTDH在EI‑reg上取得最高Pearson r = 0.862（比最佳基线高0.031），V‑reg、V‑oc分别为0.840和0.831，E‑c的Jaccard为0.579，整体性能在六项指标中五项超越SFT初始检查点，且训练样本量显著降低；

**⚠️ 局限性**

局限性包括：①对多标签情感分类的容差门过于宽松，导致推理与答案不完全一致；②稀有情感标签召回仍不足；③奖励稀疏且仅为二元，缺乏梯度形状；④未实现推理步骤与答案的一致性验证，可能产生不可靠推理；

---

## 268. A Self-Adaptive Extensible CEP framework for the Cloud-Edge Continuum

**arXiv ID:** 2608.06966 | [PDF](https://arxiv.org/pdf/2608.06966v1)

**作者:** Olaf Markus Link `[一作]` (University of Potsdam), Sukanya Bhowmik `[通讯]` (University of Potsdam)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个自适应可扩展的 CEP 框架 Svayam，能够在云‑边缘连续体中实时检测算子过载并通过负载削减（load shedding）保持低延迟；

**💡 创新点**

设计了三种负载削减模式（本地、全局、混合），结合协调器与局部分析器动态计算事件削减率，并考虑全局重要性，实现了在分布式多算子环境中的自适应负载管理，填补了缺乏此类开源框架的空白；

**🔧 技术方法**

采用 Apache Flink 流执行环境实现；使用有限状态机（FSM）检测模式；实现监控组件、分析器、负载削减器；通过线性优化求解全局削减率；实现了事件特征提取与流监测等技术；

**📊 数据集**

使用美国标准普尔 500 指数 2013‑2018 年的历史股价数据（340,000 条记录）生成的合成事件；

**📈 对比分析**

与无负载削减的“正常模式”进行对比，评估延迟符合性和召回率。结果显示，三种模式在延迟上均低于正常模式，并将处理时间从 557 s 降至 287 s；召回率从 45 % 提升至约 61‑63 %，说明负载削减显著提升了系统整体输出质量；

**⚠️ 局限性**

限制包括：超参数固定未能自适应；仅支持 AND/OR/SEQ 模式，未实现 NOT、量化或参数化；数据结构与序列化未完全优化；抽象层次不够直观，需改进设计模式。

---

## 269. A Haptic Robot Finger Designed for Guqin Instrument Playing

**arXiv ID:** 2608.07002 | [PDF](https://arxiv.org/pdf/2608.07002v1)

**作者:** Tianwei Zhang `[一作]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society), Yang Yang. Ziya Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一种结合静压与动态振动双模仿生触觉手指，集成于双臂机器人系统，用于古琴的开弦、止音、和音等弦接触任务。

**💡 创新点**

创新点在于：① 采用半指甲与柔软手指结构的仿生设计，减少摩擦并提升触感；② 采用 4×4 压阻阵列和 MEMS 麦克风实现静压与振动的并行感知；③ 将多模态触觉反馈与视觉、音频同步融合，用于触觉事件触发双手协同。

**🔧 技术方法**

使用技术包括压阻传感阵列、MEMS 麦克风、UR5 机械臂、BrainCo Revo1 机械手、Azure Kinect V4 RGB‑D、ROS2 控制框架、数据同步采集与可视化。

**📊 数据集**

使用的数据集为本实验生成的古琴音频和触觉记录（10 次试验），比较对象为人工演奏的音频作为基准；未使用公开大规模音乐数据集。

**📈 对比分析**

比较方法：对三种指尖结构（R1、R2、R3）在古琴开弦实验中，利用 log‑mel、DTW、Envelope 相关等声学相似度指标；R3-侧（半指甲侧按压）与人工演奏最相近；触觉事件触发双手协同实验平均触发到音频峰值延迟为 481±56 ms，显示可重复性。

**⚠️ 局限性**

限制：① 手指硬件缺乏高频、复杂的拨弦动作；② 视觉输入仅 30 Hz，导致多模态融合低频化；③ 缺乏统一的古琴表演评估标准，难以量化对比；④ 机器人手腕关节不足，无法完成如“轮指”等传统技巧。

---

## 270. Summarize First, Download Later: Onboard VLMs for Bandwidth-Efficient Earth Observation

**arXiv ID:** 2608.06959 | [PDF](https://arxiv.org/pdf/2608.06959v1)

**作者:** Junghwan Park `[一作]` (TelePIX), Darongsae Kwon `[通讯]` (TelePIX)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种先生成自然语言摘要后再下行完整图像的“先总结后下载”卫星下行协议，并在机载低功耗硬件上实现可交互的视觉问答与图像说明。

**💡 创新点**

创新点在于将卫星下行从单一的批量数据传输转变为基于语义的交互式决策过程，利用机载视觉‑语言模型生成轻量化文本包并支持地面问答，从而显著节省带宽并加速情报获取。

**🔧 技术方法**

使用量化的视觉‑语言模型（Gemma‑3 4B 4‑bit、LFM‑2.5‑VL 1.6B 8‑bit、Qwen3‑VL 2B 8‑bit）在 NVIDIA Jetson Orin Nano 上进行图像说明和视觉问答推理；采用自然语言作为通信介质并实现机载推理与交互接口。

**📊 数据集**

数据集包括用于视觉问答的 RSVQA‑LR（Sentinel‑2 低分辨率）和 RSVQA‑HR（高分辨率航空图像），以及用于图像说明的 RSICD 与 NWPU‑Captions 两大遥感图像说明数据集。

**📈 对比分析**

对比方法：对三种 VLM 进行 VQA 精度（Gemma‑3 72.2/61.0、LFM‑2.5 69.3/52.8、Qwen3‑VL 73.6/62.5）和图像说明的 BERTScore‑F1（≈0.90）与 CLIPScore（≈0.30）进行评估；同时量化带宽节约，文本包仅 485 B，远小于 100 kB 预览图或 50 MB 全分辨率图；机载运行时间表明 LFM‑2.5 具备最快的推理速度（≈29 s）与最小内存占用。

**⚠️ 局限性**

局限性：对领域漂移与不确定性缺乏鲁棒性；需进一步提升模型在不同传感器和任务上的泛化；能耗和实时性仍受限；文本摘要虽能快速决策，但无法替代完整图像进行精细分析。

---

## 271. Critical Acclaim Orientation in Large Language Models: Evidence from Film Preference Elicitation

**arXiv ID:** 2608.06955 | [PDF](https://arxiv.org/pdf/2608.06955v1)

**作者:** Jonghyun Jee `[一作]` (Northwestern University), Aaron Shaw `[通讯]` (Northwestern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对八个不同家族、不同规模的LLM进行20,000次成对强制选择电影比较，构建了包含批评声誉、商业成功和双重合法性三类的200部电影基准，评估模型的文化偏好。

**💡 创新点**

首次发现LLM在电影评估中系统性偏好批评界认可的电影而非仅商业票房成功的电影，并且这种偏好随模型规模增大而强化，揭示了文化评价层级在LLM输出中的结构化体现。

**🔧 技术方法**

采用Bradley–Terry模型估计每部电影的潜在强度，并用嵌套OLS回归剖析可见度、流行度与评判之间的相互影响，验证批评声誉偏好的独立性。

**📊 数据集**

使用《They Shoot Pictures, Don't They?》专业影评榜单与Box Office Mojo票房榜单交叉筛选得到的200部电影样本，分为批评声誉优先、商业成功优先和双重合法性三组。

**📈 对比分析**

通过对每个模型进行20,000次成对比较并计算胜率与标准化Bradley–Terry分数，八个模型均显著偏好批评声誉优先电影；大模型的胜率普遍高于小模型，说明规模影响评估倾向。

**⚠️ 局限性**

研究仅聚焦电影领域、使用英文提示、仅给出影片标题与年份，缺乏跨文化、跨语言和更丰富情境的评估；同时模型内部生成机制未被探究，限制了对偏好根源的深入理解。

---

## 272. CyberLLM: A Multi-Agent LLM Framework for Autonomous Detection and Guarded Response in Automotive Cybersecurity

**arXiv ID:** 2608.06651 | [PDF](https://arxiv.org/pdf/2608.06651v1)

**作者:** Nenad Petrovic `[一作]` (Technical University of Munich), Alois Knoll `[通讯]` (Technical University of Munich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CyberLLM，一个多代理 LLM 框架，结合 deterministic 分析与 LLM 细化，实现汽车软件安全漏洞检测与受控响应。

**💡 创新点**

通过层级检测、基于签名的安全门控、可回溯记忆和独立行动对齐判定，将 LLM 预测置于可审计的安全包层，实现在安全约束下的自适应自动化防御。

**🔧 技术方法**

使用正则/AST 静态分析、BANDIT/SEMgrep、拓扑图检查作为 deterministic 基底，GPT‑4o 进行 LLM 细化和完整性检验，Schema‑driven 多代理架构与 MCP 工具注册表，四项上下文安全属性与 H_a 对齐 Oracle。

**📊 数据集**

在 9 个原始汽车 ECU 模块（C/C++/Rust）以及 2 个干净对照文件的 47 个手工标记漏洞数据集上进行评估。

**📈 对比分析**

将 Fast（deterministic）与 Deep（加 LLM 细化）两种模式对比，Fast 覆盖率 34%/F1 0.51，Deep 覆盖率 70%/F1 0.83，所有模式保持 1.000 的精度。

**⚠️ 局限性**

剩余漏报集中于中等严重度的逻辑/DoS 漏洞，缺乏动态分析支持；同时 LLM 仍可能在非安全边界环境产生误报，需要进一步引入 CVE/标准检索与大模型上下文管理。

---

## 273. ED-CSP: Crystal Structure Prediction from Electron Diffraction

**arXiv ID:** 2608.06448 | [PDF](https://arxiv.org/pdf/2608.06448v1)

**作者:** Germain Poloudenny `[一作]` (Université d'Artois), Arnaud Demortière `[通讯]` (Université de Picardie Jules Verne)

**通讯引用:** 6336 | [OpenAlex ID](https://openalex.org/A5053447770)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出ED-CSP模型，利用已知化学成分和稀疏多视角电子衍射（ED）点列表来生成晶格和原子分数坐标。

**💡 创新点**

创新点在于将ED视角信息与机器学习相结合，使用关系式点集编码器、视角聚合和周期流生成器，首次在无索引ED数据下实现全新晶体结构预测。

**🔧 技术方法**

采用关系式图注意力编码器、均值-最大池化聚合、周期流（CSPFlow/CSPNet）生成器、对比预训练和可微动态模拟。

**📊 数据集**

使用CHILI-100K测试集和自构建的ED-CS 4.85M结构数据集（包含模拟的多视角ED点）。

**📈 对比分析**

与PXRDGen、XRDSol、deCIFer、库检索和Superflip/EDMA等方法比较，在CHILI-100K上MR@5达57.49，预训练注册1M后提升到66.27，显著优于同类PXRD条件下的生成模型。

**⚠️ 局限性**

局限性包括对实验ED数据的迁移性不足、对视角数量的依赖、候选池质量与能量排序的局限，以及未能充分利用全局晶体对称性和缺少真实实验验证。

---

## 274. Understanding and Improving Model Editing for Secure Code Generation

**arXiv ID:** 2608.06848 | [PDF](https://arxiv.org/pdf/2608.06848v1)

**作者:** Weifeng Sun `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了模型编辑（UltraEdit、DINM、DEFER*）在硬化LLM生成安全代码方面的效果，并与推理时硬化方法CoSec进行对比，进一步提出后编辑修正方案来缓解功能回退问题。

**💡 创新点**

首次将模型编辑直接应用于安全代码生成任务，证明其在安全率上明显优于CoSec，并提出结合编辑友好正则化的后编辑方法实现安全与功能的双赢。

**🔧 技术方法**

使用UltraEdit、DINM、DEFER*等局部参数编辑技术、CoSec的辅助安全模型与token调度、以及Post‑Edit Refinement的编辑友好正则化来提升安全性和功能性。

**📊 数据集**

采用He & Vechev构造的710对漏洞/修复代码、CodeQL规则集、HumanEval、MultiPL‑E和CodeGuard+等多维度数据集进行评估。

**📈 对比分析**

通过安全率、未见CWE泛化、功能正确率（Pass@k）和推理延迟等指标对比，模型编辑在安全率上提升15–25%（CoSec仅提升约1–2%），UltraEdit+后编辑在保持安全提升的同时显著恢复功能正确率，并在推理效率上优于CoSec。

**⚠️ 局限性**

安全提升易受编辑参数选择影响，泛化到未见CWE不稳定，编辑可能导致功能回退，且仅适用于可访问模型权重的公开模型，需慎重调参并关注功能与安全的权衡。

---

## 275. How Much, Then Where: Credit-Conserving Action-to-Token Allocation for Multi-Turn Agent Reinforcement Learning

**arXiv ID:** 2608.07118 | [PDF](https://arxiv.org/pdf/2608.07118v1)

**作者:** Lichao Ma `[一作]` (Peking University), Jiaye Lin `[通讯]` (Meituan LongCat Interaction Team)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FACTOR方法，将轨迹级信用分配与令牌级分配分离，先用TD残差确定动作信用，再用后向教师似然差距分配令牌信用，并通过动作均值保持归一化实现两者互不干扰。

**💡 创新点**

创新点在于两阶段因子化分离信用与分配，结合检查点校准的TD信用、后向教师-学生似然差距以及动作均值保留归一化，消除了令牌长度对损失的影响，提升了多轮LLM代理的信用分配精度。

**🔧 技术方法**

使用技术包括：TD残差分解、后向教师似然差距分配、动作均值保留归一化、动作均值PPO surrogate、检查点恢复与无监督推理续写以及多层次价值头回归。

**📊 数据集**

实验使用的公开数据集为ALFWorld、WebShop和ScienceWorld。

**📈 对比分析**

与SERL-Repro、GRPO等基线在相同硬件和超参设置下比较，FACTOR在所有三个基准上平均提升约2–4个百分点，并在更大backbone（Qwen2.5-14B、Llama-3.1-8B）上也保持显著优势。

**⚠️ 局限性**

主要局限在于需要可恢复状态和推理续写来估计价值，导致额外的环境交互和计算成本。

---

## 276. Learning in Deep Networks under Dale's Constraint

**arXiv ID:** 2608.06963 | [PDF](https://arxiv.org/pdf/2608.06963v1)

**作者:** Roy Abel `[一作]` (Weizmann Institute of Science), Shimon Ullman `[通讯]` (Weizmann Institute of Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于对称正负通道（on‑off 电路）的 Dale 约束网络，并给出了只使用非负活动和固定符号突触的局部 Hebbian 学习规则。

**💡 创新点**

创新点在于用两路非负通道来编码有符号信息，同时保持突触固定符号和局部可实现的学习机制；理论上证明该机制可等价恢复标准反向传播的梯度更新。

**🔧 技术方法**

使用了 on‑off 电路模块、对称/非对称的前馈与反馈通路、局部 Hebbian 更新，以及在网络中重复使用的双通道结构。

**📊 数据集**

在 MNIST、Fashion‑MNIST、CIFAR‑10 和 Tiny ImageNet 等图像分类数据集上进行评估。

**📈 对比分析**

与标准 ReLU MLP/Conv 网络以及其它生物学可实现方法进行对比；on‑off 模型在所有数据集上表现相当或更优，Tiny ImageNet 上 Top‑1 为 42.31%，明显高于 SCFF 35.7% 以及普通 Conv 35.58%。

**⚠️ 局限性**

局限性包括：仍未覆盖完整的生物学可行性、需要双通道导致神经元数目显著增加、对前馈‑反馈权重对齐敏感等。

---

## 277. Stable Curves, Unstable Items: Item-Level Scaling Heterogeneity in Video LLMs

**arXiv ID:** 2608.07014 | [PDF](https://arxiv.org/pdf/2608.07014v1)

**作者:** Wenzhang Sun `[一作]`, Kun Zhan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对冻结的多种视频LLM在不同视觉预算下进行配对预算轨迹分析，揭示了隐藏的项目级正负转移和互补性；

**💡 创新点**

提出了匹配网格轨迹度量（oracle headroom、视觉混淆、文本覆盖、波动）以及可复现的审计数据包，显著提升了对规模曲线的可解释性；

**🔧 技术方法**

采用受控帧计数、分辨率、采样策略、等计算分配、原始/缓存执行等技术，并使用二元与连续轨迹、统计检验（McNemar、bootstrap）及置信度级联进行评估；

**📊 数据集**

使用 Video-MME v1/v2、MLVU、AVSD 等公开基准，包括多选题（V1短/中、V2）和开放式问答、摘要、对话生成；

**📈 对比分析**

在五个开源模型（Qwen2.5-VL-7B、Qwen3-VL-8B、InternVL3-8B、InternVL3.5-8B、LLaVA-NeXT-Video-7B）上，匹配网格下oracle headroom 8.8–18.9分，视觉混淆率12.5–25.5%；置信度级联可在约31.7%帧成本下降的情况下实现与固定128帧相同的准确率；

**⚠️ 局限性**

仅针对冻结模型和现有基准，需存储大量逐项输出；未探讨模型内部机制，结果对可微调或新架构的迁移性有限；

---

## 278. AgentChaos: Chaos Engineering for Agent Systems via Programmatic Fault Injection

**arXiv ID:** 2608.06790 | [PDF](https://arxiv.org/pdf/2608.06790v1)

**作者:** Gou Tan `[一作]` (Sun Yat-sen University), David Lo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 AgentChaos，一套非侵入式、运行时的 LLM API 故障注入框架，用于系统性评估多种代理系统在不同 LLM API 故障（crash、omission、value 等）下的鲁棒性。

**💡 创新点**

创新点包括：①在 HTTP 层对 LLM API 响应进行字段级故障注入；②构建了完整的故障分类及 65 种注入配置；③实现了触发验证机制，过滤未触发任务；④对多种代理架构、数据集和模型进行统一评估。

**🔧 技术方法**

技术手段包括：HTTP 级 monkey‑patch 与响应拦截/修改、注入策略（单次、持续、间歇、突发）与位置/组合设置、触发验证、以及对多大模型（Claude‑Sonnet‑4.5、GPT‑5.2、DeepSeek‑V3.2、Seed‑1.8）与多代理框架（AutoGen、MAD、MapCoder、EvoMAC、Mini‑SE）的部署。

**📊 数据集**

使用了四大类任务数据集：代码生成（HumanEval、HumanEval+、MBPP、MBPP+）、知识推理（MMLU‑Pro、MATH‑500）和软件工程（SWE‑bench Pro）。

**📈 对比分析**

通过对比无故障注入与注入条件下的 pass@1，计算 Δpass@1；实验显示所有系统均出现显著下降，MapCoder 最差，Δ 高达 49.66%；不同模型表现一致，表明鲁棒性取决于系统实现而非模型；诊断方法准确率低于 56%，资源消耗方向因系统而异。

**⚠️ 局限性**

局限性包括：仅评估了每种架构下的单一代理系统，难以分离架构与实现的影响；故障配置覆盖面有限，未涵盖所有可能的 LLM API 字段；温度 0.7 的随机性导致结果波动；未涵盖 RAG 等新型代理架构。

---

## 279. bioMoR: Biology-Guided Mixture-of-Recursions for Effective Genomic Learning

**arXiv ID:** 2608.06727 | [PDF](https://arxiv.org/pdf/2608.06727v1)

**作者:** Koushik Howlader `[一作]` (Iowa State University), Wei Le `[通讯]` (Iowa State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 bioMoR，一种将生物学知识融入 Mixture-of-Recursions 框架的基因与通路级别 omics 学习模型。

**💡 创新点**

创新点在于三处知识注入：嵌入平滑、注意力偏置以及基于邻域的递归深度路由，实现生物学结构驱动的自适应计算。

**🔧 技术方法**

采用 Transformer 共享递归块、Mixture-of-Recursions 结构、基因共表达或 Reactome 通路图等生物学知识图，以及多任务损失与辅助正则。

**📊 数据集**

使用八个多样化的单细胞与多组学基准数据集，包括 Genomap、Reactome 病理通路数据、TCGA HNSC、UCEC、COAD、KIRP 等。

**📈 对比分析**

与 Vanilla、Recursive、MoR 以及多种 state‑of‑the‑art 基线在统一五折交叉验证下比较，bioMoR 在宏 F1 上平均提升 8.2 点、平衡准确率提升 7.1 点，同时参数量减少 75% 与 FLOPs 降低 58%。

**⚠️ 局限性**

局限性包括对知识图质量的依赖、对不同通路定义和标记基因选择的敏感性，以及在极端高维或稀疏数据时的可扩展性待进一步验证。

---

## 280. Resource-Aware Intrusion Detection in Infrastructure Networks: A Game-Theoretic Approach

**arXiv ID:** 2608.06655 | [PDF](https://arxiv.org/pdf/2608.06655v1)

**作者:** Xuanli Lin `[一作]` (Arizona State University), Guoliang Xue `[通讯]` (Arizona State University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了基础设施网络中分布式感知与边缘计算相结合的入侵检测问题，构建了一个图安全博弈模型，探讨了在不同观测条件（无观察、观察纯配置、观察混合策略）下防御者与攻击者的最优策略，并设计了多种算法（RAD、Adaptive Refinement、RADAR-guided Restricted Game）求解纯、混合纳什均衡和强/弱Stackelberg均衡。

**💡 创新点**

创新点包括：①首次将多种感知类型、处理模式、资源限制（部署成本、服务器容量、误报阈值）融入图安全博弈；②揭示观测信息对均衡结构和可计算性的决定性影响，证明了多种均衡存在性与 NP‑hard 性；③提出了资源感知降级（RAD）与自适应细化（Adaptive Refinement）搜索框架，以及针对混合策略的RADAR限制游戏生成器；④通过实验验证这些算法在可枚举和不可枚举实例上都能保持较小的误差，且相对传统精确枚举方法具有显著的时间优势。

**🔧 技术方法**

使用的技术包括：图安全博弈建模、短路算法（Dijkstra）求解单纯路径最优响应、线性规划求解混合Stackelberg/纳什均衡、支撑集生成与最优加权几何平均替代路径生成、资源感知降级与自适应细化搜索、限制游戏的列生成与价格更新、以及复杂度分析和NP‑hard归约。

**📊 数据集**

实验数据集主要为可枚举的“Diamond Graph”和“Random Geometric Graph”两类网络，节点数分别为4–8（可枚举）和10–12（扩展），每个拓扑产生100个随机实例。每条边支持三种感知类型，检测概率、误报概率、部署成本和服务器负载等参数按给定分布随机扰动。

**📈 对比分析**

与完整枚举或精确求解相比：在可枚举实例上，RADAR+Adaptive Refinement 的纯策略优化平均误差约为3.6%（比精确枚举的0%显著提升但仍可接受），在混合纳什/Stackelberg场景下，生成的有限游戏结果与完整游戏的平均绝对归一化差异分别约为0.47%和0.43%，最大差异在可接受范围内。时间上，RADAR+Adaptive Refinement 的评估时间平均在数十毫秒级，远低于完整枚举的数百毫秒。

**⚠️ 局限性**

限制包括：①在不可枚举实例上仅提供局部搜索结果，无法给出全局最优保证；②混合策略响应与优化的 NP‑hard 性导致无多项式时间通用最优解；③生成的路由和配置差距仅为局部诊断，不能保证整体最优；④实验中未覆盖更大规模网络或更复杂的资源约束，未能验证方法在极端条件下的稳健性。

---

## 281. ArchEGraph: A Large-Scale Graph Dataset for Geometry-Topology-Physics Aligned Building Energy Modeling

**arXiv ID:** 2608.06772 | [PDF](https://arxiv.org/pdf/2608.06772v1)

**作者:** Yihui Li `[一作]` (Tsinghua University), Borong Lin `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 ArchEGraph 大规模建筑能耗数据集，并定义了 Mesh-to-Graph（M2G）和 Graph-to-Energy（G2E）两个基准任务；

**💡 创新点**

创新点在于将建筑几何、拓扑、气候和能耗信息统一映射为异构图结构，提供了跨建筑和跨气候的全局可扩展基准；

**🔧 技术方法**

使用图神经网络（SetTransformer、TopoTransformer 等）与注意力/消息传递模型结合天气编码器和时序解码器进行评估；

**📊 数据集**

数据集包含 5,481 个建筑，49,326 个气候-模拟案例，包含 1.33×10^5 空间节点、1.44×10^6 面节点，覆盖 64 城市的全球气候；

**📈 对比分析**

与传统非图的 WeatherMLP、以及多种图模型进行对比，结果表明图模型在 M2G 中实现了高达 0.98 的 F1/Acc，G2E 中 MAE 可低至约 1.6 W/m²；在跨建筑/气候的泛化实验中，图模型显著优于天气仅模型；

**⚠️ 局限性**

局限性包括仅涵盖办公建筑、模拟数据无真实测量校准、缺乏材质多样性、对非常规建筑拓扑的泛化仍有限、模型规模受算力限制

---

## 282. Recovering Explanations from Transformed Rule-Based Ontologies

**arXiv ID:** 2608.06399 | [PDF](https://arxiv.org/pdf/2608.06399v1)

**作者:** Alex Ivliev `[一作]` (TU Dresden), Maximilian Marx `[通讯]` (TU Dresden)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

研究如何在知识图谱规则（Datalog）变换后恢复原始规则的推理证明树，提供逆向证明构造的算法与复杂度分析。

**💡 创新点**

首次提出“证明树同态”和基于Monadic Second‑Order逻辑的证明树变换语言，揭示它们与程序包含关系的关联，并给出了复杂度上界与下界。

**🔧 技术方法**

利用程序包含与统一包含理论、树同态、MSO解释（树转化）以及树宽度与MSO模型检查的对数空间特性。

**📊 数据集**

无实验数据集，本工作为理论分析与算法设计，主要关注推理证明结构而非真实知识图谱数据。

**📈 对比分析**

通过理论证明与已知难度类（P、NL、P‑hard、NL‑hard）对比，展示了逆向证明构造的时间与空间复杂度，并证明MSO变换可在对数空间内执行。

**⚠️ 局限性**

仅适用于Datalog及其简化子语言；对存在性规则、聚合或带层级否定的扩展尚未讨论；理论结果与实际推理引擎性能之间仍有差距。

---

## 283. Simple-OPD: Demystifying Warm-up for On-policy Distillation

**arXiv ID:** 2608.06802 | [PDF](https://arxiv.org/pdf/2608.06802v1)

**作者:** Tao Liu `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了在 on‑policy distillation (OPD) 训练中预热阶段的作用，揭示了合适的数据和训练策略对学生模型的影响，并提出了一种简单易用的预热方案——Simple‑OPD；

**💡 创新点**

创新点在于发现预热时使用与后续 OPD 教师兼容的 chain‑of‑thought（CoT）提示能显著提升初始化效果，且不需要正确答案；同时，采用低秩 LoRA 预热到近饱和点能在保持领域适配的同时避免对外域泛化的严重退化；

**🔧 技术方法**

主要技术包括：Token‑level reverse KL 的 OPD 训练、带 CoT 的监督式预热（SFT）、低秩 LoRA 参数微调、以及对比实验评估；

**📊 数据集**

实验使用了 Qwen3 系列模型、DAPO‑Math‑17K、MATH‑500、AIME24/25、AMC23、IFEval、GPQA‑Diamond、HumanEval、MMLU‑Pro 子集等数据集；

**📈 对比分析**

与直接从基础学生模型进行 OPD 的基线相比，Simple‑OPD 在 ID 任务（如 AIME24/25）平均提升 1.3–1.6 分，同时 OOD 任务保持不变或略有提升，且在训练速度和稳定性上也表现更佳；

**⚠️ 局限性**

局限性包括实验仅覆盖部分模型家族、规模有限，未检验更大规模或不同领域的通用性，且仅针对基础 OPD 目标，未探讨与更高级目标或动态调度的交互作用。

---

## 284. Stream Learning: Partition-Fair Gossip Learning Without Tokens

**arXiv ID:** 2608.06946 | [PDF](https://arxiv.org/pdf/2608.06946v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 285. EpiFlow: A framework for improving the utility of wastewater signals for disease forecasting

**arXiv ID:** 2608.06671 | [PDF](https://arxiv.org/pdf/2608.06671v1)

**作者:** Aniruddha Adiga `[一作]` (University of Virginia), Madhav Marathe `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了 EpiFlow 框架，整合预处理、信号质量评估、时变 Granger 因果分析和滚动 VAR 预测，利用废水病毒载量（WVL）提升对 COVID‑19 住院人数的实时概率预测。

**💡 创新点**

创新点在于：①使用排列熵衡量废水信号可预测性并自动选择合适窗口；②通过滚动窗口 Granger 因果检验揭示废水与住院人数的时间变因果关系；③对废水信号进行 Savitzky–Golay 滤波去噪；④将以上结果嵌入时间变 VAR 模型，显著提升峰值期间的预测覆盖率。

**🔧 技术方法**

技术包括：Savitzky–Golay 滤波、排列熵（Permutation Entropy）、滚动窗口 Granger 因果（RWGC）、VAR 预测、ARIMA/ARIMAX 基线、加权区间得分（WIS）评估以及置换检验。

**📊 数据集**

数据集为 2021‑12‑2023 期间来自弗吉尼亚州 36 个污水处理厂的 SARS‑CoV‑2 WVL 数据（按周汇总后去噪）以及弗吉尼亚州 5 个卫生区和全州的 COVID‑19 住院人数。

**📈 对比分析**

与 ARIMA 与 ARIMAX（含原始和去噪废水信号）基线比较，使用 WIS 和预测覆盖率评估。VAR‑dn（去噪后滚动 VAR）在波动期覆盖率提升约 20%（相较于 ARIMA），并在 WIS 上优于所有基线；在稳定期 ARIMA 仍具竞争力。

**⚠️ 局限性**

局限性包括：VAR 模型仅捕捉线性关系，未给出传输或感染的机制解释；假设污水与卫生区的空间对齐不变，忽略站点停用或采样变动；缺乏对缺失值的更复杂处理；目前仅验证于弗吉尼亚州，需在更广范围内检验。

---

## 286. Sharding Prevents LLM Oversight Failures and Adversarial Exploitation

**arXiv ID:** 2608.06422 | [PDF](https://arxiv.org/pdf/2608.06422v1)

**作者:** Victor Akinwande `[一作]` (Carnegie Mellon University), Aran Nayebi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1926 | [OpenAlex ID](https://openalex.org/A5058188874)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

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

限制在于XXX，例如数据集规模较小或模型复杂度高。

---

## 287. ELMZip: Onboard Satellite Image Compression via Extreme Learning Machines for Efficient Downlink

**arXiv ID:** 2608.06942 | [PDF](https://arxiv.org/pdf/2608.06942v1)

**作者:** Woojin Cho `[一作]` (TelePIX), Darongsae Kwon `[通讯]` (TelePIX)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于极限学习机(ELM)的卫星图像压缩框架ELMZip，实现机载快速拟合并仅通过输出权重进行下行。

**💡 创新点**

创新点在于将ELM与域分解相结合，使用随机固定特征一次性求解输出权重，省去反向传播；采用不对称下行协议，仅发送输出权重，实现约10×的压缩率，并支持即时预览。

**🔧 技术方法**

采用极限学习机、域分解、正交正则化、线性最小二乘闭式解、Sine激活函数、窗口函数拼接等技术，并利用PyTorch+CUDA在Jetson Nano等低功耗平台上加速。

**📊 数据集**

使用Sentinel‑2 MSI Level‑0与Level‑1C六个地理区域（Antuco、Puszta、Andaman、Cairo、Merapi、Seoul）的多光谱图像数据集。

**📈 对比分析**

与MLP、SIREN、FFN、GaussNet、WIRE等基线在相同下行参数预算下对比，ELMZip在PSNR与SSIM上均优于所有基线，且电能消耗仅为基线的约1/10。

**⚠️ 局限性**

局限性包括需预共享随机特征种子；域分解对大尺度图像可能导致内存/计算负担；闭式求解受数值稳定性影响；对极端噪声或非光学波段的适应性尚未验证。

---

## 288. Policy-Masked Private Experts: Auditable and Reversible Capability Access Control in Sparse MoE Models

**arXiv ID:** 2608.06690 | [PDF](https://arxiv.org/pdf/2608.06690v1)

**作者:** Zhuoheng Huang `[一作]` (Independent Researcher), Mukesh Singh `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在稀疏 MoE 语言模型中引入了政策掩码私有专家，利用受信任授权在前向传递前决定使用公共还是私有专家，从而实现请求级别的可审计与可逆访问控制，并验证其在工具调用任务上的性能提升。

**💡 创新点**

创新点在于将访问控制与参数可达性分离：在路由前施加硬性掩码，保证未授权请求绝不执行私有专家；并通过路由日志、独立前向钩子等手段直接验证非参与性，避免仅靠拒绝行为推断安全性。

**🔧 技术方法**

核心技术包括稀疏 MoE 模型、政策掩码（policy mask）和 top‑k 路由、冻结公共参数与专家、训练私有专家分支、LoRA 对比实验、独立前向钩子、请求级别的权限元数据解析及密钥化模型清单。

**📊 数据集**

使用的主要数据集包括：ToolMind（工具调用对齐评测），ToolFailBench（新鲜外部工具使用评测），CCTU（受限约束工具使用评测），以及在 DeepSeek 上的相同 100 条任务对齐示例。训练时采用 8,524 条 ToolMind 目标（Qwen）或 1,024 条（DeepSeek）等。

**📈 对比分析**

通过对比公共专家、私有专家、LoRA、提示仅、软路由仅等六种配置，在 Qwen 上：ToolMind 上提升 5pp，ToolFailBench 上提升 21.3pp，CCTU 近似零差异；在 DeepSeek 上对齐任务提升 27pp。统计检验采用 McNemar、百分位 bootstrap、Holm 校正等，表明私有专家在部分任务上显著优于公共或 LoRA，但在 CCTU 上无显著提升。

**⚠️ 局限性**

局限性包括：仅在稀疏 MoE 模型的少数层引入私有分支，未覆盖稠密模型或多层私有化；TCB 较大，若身份服务或掩码编译器被破坏则失效；未对侧信道、内存/时间泄漏进行分析；评测主要集中在工具调用任务，未检验在更广泛语义任务中的普适性；LoRA 仍在未授权时执行调用，表明逻辑门控不等同物理非参与。

---

## 289. Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving

**arXiv ID:** 2608.06557 | [PDF](https://arxiv.org/pdf/2608.06557v1)

**作者:** Muhammad Adnan `[一作]` (University of British Columbia), Esha Choukse `[通讯]` (Microsoft Azure Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Cascade，一种基于每个请求的延迟预算（SLO 与预测服务时间之差）来同时调度请求和管理 KV 缓存的 LLM 服务器系统。

**💡 创新点**

创新点在于：1) 将请求的剩余延迟预算统一为单一资源，既用于调度也用于跨层 KV 缓存迁移；2) 动态预测每个请求的剩余预算并实时更新；3) 通过预算驱动的调度和 KV 缓存决策，减少 Head‑of‑Line 阻塞和深层缓存恢复导致的 SLO 违规；4) 在保持公平性的同时显著提升好通过率。

**🔧 技术方法**

使用的技术包括：基于 vLLM 的推理引擎、Vidur 仿真器对多层 KV 缓存和 NVLink 通信进行建模、预算估算器（利用输入长度、预测输出长度、已缓存 KV、系统负载）、预算驱动的优先级调度、预算约束下的 KV 恢复/预取决策。

**📊 数据集**

评估数据集为真实生产轨迹：从 Aliyun Bailian 收集的两小时 Qwen 系统日志，涵盖 ChatBot、Tool&Agent、Coder、Reasoning 四类请求；在三大 LLM（Qwen‑2.5‑72B、Llama‑3‑70B、Llama‑3‑405B）上进行实验。

**📈 对比分析**

与 FCFS、EDF、SJF 等基线进行对比；Cascade 在所有模型与工作负载下平均提升 2.4–3.0× 好通过率，同时将 SLO 违规率降低至 10–15%，并在多种容量和负载变化场景中保持稳定；公平性指标（Jain）接近 1，说明对不同长度/类型请求几乎没有偏差。

**⚠️ 局限性**

局限性包括：1) 需要对 KV 缓存状态和系统负载进行实时监测与预测，复杂度随规模增加；2) 在极端高负载或极短 SLO 的情形下，预算不足仍可能导致 SLO 违规；3) 依赖于 vLLM 以及预先收集的 KV 迁移性能数据，迁移到其他架构或新硬件时需重新校准。

---

## 290. WaveFreqAnchor: Wave-Structural Anchoring and Frequency Correction Diffusion for Training-Free Face Restoration

**arXiv ID:** 2608.06717 | [PDF](https://arxiv.org/pdf/2608.06717v1)

**作者:** Zelin Du `[一作]` (Nanjing University of Posts and Telecommunications), Guangwei Gao `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出无训练的WaveFreqAnchor框架，对预训练扩散模型进行波结构锚定和频率校正，以实现高质量人脸恢复。

**💡 创新点**

创新点在于结合Anchor‑Space Wave‑Structural Guidance、Multi‑scale Wavelet‑Fourier Injection与Subband High‑Frequency Enhancement三种无训练的引导策略，有效抑制结构漂移并提升身份相似度。

**🔧 技术方法**

使用波形响应一致性约束、频域相位校正、Haar小波变换和无训练的梯度引导，配合冻结的无条件扩散模型。

**📊 数据集**

在CelebA‑HQ、LFW‑Test、WIDER‑Test和WebPhoto‑Test等公开人脸数据集上评估。

**📈 对比分析**

与CodeFormer、DPS、DiffBIR、OSDFace、FiDeSR、SubDAPS++等方法对比，实验显示在4×/8×/16×超分以及真实场景恢复中均取得更高的PSNR/SSIM/LPIPS/ID等指标，并在无参考指标上优于对手。

**⚠️ 局限性**

局限包括对严重色偏、曝光过度、低对比度等情况仍有挑战，且仅使用两种预设配置难以覆盖连续的真实降质范围。

---

## 291. Science Edge Evaluation: SEE the Missing Step Toward Real Scientific Discovery

**arXiv ID:** 2608.06931 | [PDF](https://arxiv.org/pdf/2608.06931v1)

**作者:** Taolin Han `[一作]` (Alibaba Group), Bing Zhao `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了基于真实实验场景的多模态科学问答基准 Science Edge Evaluation，并评估了19个多模态大语言模型的表现。

**💡 创新点**

提出了以实验数据为核心的多模态跨学科基准，强调证据驱动推理而非仅知识回忆，并对工具增强视觉代理的效果进行系统分析。

**🔧 技术方法**

使用多模态大语言模型评估框架、LLM-as-judge、工具增强视觉代理（web search + code interpreter）、模态消融、交叉学科标签等方法。

**📊 数据集**

Science Edge Evaluation数据集，共1116条问题（1049公开），涵盖化学、生物、材料三大学科及其交叉子领域，包含文本与多种实验图像（光谱、显微、染色、图谱等）。

**📈 对比分析**

对19个模型进行统一推理和评估，平均准确率仅32-35%，最佳模型GPT‑5.6‑Sol (Max) 48.7%；工具增强后最高52.7%。一般模型普遍优于专门化模型。

**⚠️ 局限性**

模型缺乏对多模态证据的精准感知与整合，易忽视视觉缺失、先验知识主导、逻辑不连贯、过度推断，工具使用仍带来新的错误，整体表现低于预期。

---

## 292. Latent Fact-Checking: Detecting Misinformation through Activation Engineering

**arXiv ID:** 2608.06417 | [PDF](https://arxiv.org/pdf/2608.06417v1)

**作者:** Pedro Barcelos `[一作]` (Pontifical Catholic University of Rio Grande do Sul), Rodrigo C. Barros `[通讯]` (Pontifical Catholic University of Rio Grande do Sul)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在不对语言模型进行微调的前提下，通过对冻结模型内部激活进行对比性向量提取并投影，构建了一个基于激活工程的谣言检测方法。

**💡 创新点**

创新点在于将对比激活增量（CAA）用于估计“谎言方向”，并仅通过对原始句子激活的投影+浅层MLP即可完成事实真伪判断，避免了对生成文本的依赖。

**🔧 技术方法**

使用的技术包括对比激活提取、线性方向估计（falsehood direction）、向量投影、层级选择以及单隐藏层MLP分类器。

**📊 数据集**

实验使用了三大真实事实核查基准：AVeriTeC、LIAR 和 FACTors。

**📈 对比分析**

与零样本/少量样本提示法相比，该方法在 LIAR 和 FACTors 上（尤其是小模型）平均提升了 20–30% 的准确率，并在 11 种不同规模的模型上保持稳健表现。

**⚠️ 局限性**

主要限制是对需要检索证据的 AVeriTeC 数据集效果较差，因该方法仅利用句子本身的激活，无法捕获外部证据所决定的真伪信息。

---

## 293. Tight Security for BBS Signatures

**arXiv ID:** 2608.06724 | [PDF](https://arxiv.org/pdf/2608.06724v1)

**作者:** Rutchathon Chairattana-Apirom `[一作]`, Stefano Tessaro `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文研究了BBS数字签名的具体安全性，并给出了在每条消息最多签名一次（即签名查询不重复）时的紧凑安全证明，同时证明在允许同一消息多次签名的情形下，任何基于q‑SDH假设的代数直线或重放重置化简无法达到紧凑性；

**💡 创新点**

创新点在于提出一种新的多项式选择与签名标签分配策略，能够在不依赖代数群模型的情况下实现紧凑化简；并通过元化简技术揭示了多签名情形下紧凑化简的根本不可能性；

**🔧 技术方法**

核心技术包括基于随机多项式的签名生成与提取策略、H‑系数技术用于分析统计距离、以及针对多项式与随机函数的概率工具包；

**📊 数据集**

本研究没有使用传统数据集，主要在理论分析与抽象模型下进行实验验证，涉及对大素数域中的随机多项式和多项式根分布的统计估计；

**📈 对比分析**

与先前的非紧凑化简相比，本文的紧凑化简将优势损失从O(q²)降低到O(q)，实现了接近q‑SDH假设下的最优安全度；尽管有额外的O(q²(log²logp+T+Tp))时间开销，但整体性能仍优于现有证明；

**⚠️ 局限性**

局限性包括：仅适用于签名查询不重复的情形，无法处理同一消息的多次签名；此外，紧凑化简仍存在多项式次数的增大和额外运行时间；在多用户或适应性篡改场景下也无法保证紧凑性。

---

## 294. Interpretable Unsupervised Community Detection with LLM-Symbolized Structured Processes

**arXiv ID:** 2608.06402 | [PDF](https://arxiv.org/pdf/2608.06402v1)

**作者:** Aoting Zeng `[一作]` (Shanghai Jiao Tong University), Wenjie Zhang `[通讯]` (University of New South Wales)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于大型语言模型的可解释、无监督社区检测框架 LUCID，采用四阶段（初始化、合并、细化、选择）流程完成社区发现。

**💡 创新点**

创新点在于将 LLM 作为规则诱导器而非直接预测器，生成多因子决策树用于合并、粗细化规则用于去噪，并提出密度正则化导电度评价指标，以实现可解释且无需训练的社区检测。

**🔧 技术方法**

使用 k‑ego 子图初始化与无监督节点角色标记、LLM 生成的决策树与粗细化规则、并行规则执行、RDC 质量度量，以及传统图嵌入与矩阵分解技术对照。

**📊 数据集**

使用 SNAP 公开数据集 Facebook、Amazon、Livejournal、DBLP 和 Twitter。

**📈 对比分析**

与无监督基线（BigClam、ComE、CommunityGAN、Bespoke）以及半监督基线（SEAL、CLARE、PROCOM）进行对比，LUCID 在 F1 上平均提升 20.7%（相对最佳无监督方法），在 Jaccard 上提升 31.9%；在半监督基线上也分别提升 10.1% 与 16.6%，实现最优性能。

**⚠️ 局限性**

主要局限包括对 LLM 计算成本与 prompt 设计的依赖，受限于 LLM 上下文窗口、对文本属性或动态图的适应性不足，以及在极大图规模下的可扩展性仍需进一步验证。

---

## 295. Unsupervised Adaptation of PDE Foundation Models

**arXiv ID:** 2608.07053 | [PDF](https://arxiv.org/pdf/2608.07053v1)

**作者:** Ziye Song `[一作]` (Nanyang Technological University), Yueming Lyu `[通讯]` (A*Star)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种无监督的 PDE 基础模型微调框架，只利用目标 PDE 的方程式和边界条件即可将预训练模型适配到新方程，完全不需要内部的真值解数据。

**💡 创新点**

创新点包括：①在适配阶段引入仅基于 PDE 残差和边界约束的无监督目标；②在基础模型中使用邻域注意力（Neighborhood Attention）Transformer 以天然支持不同空间分辨率；③提出 NSLoRA（Newton‑Schulz 正交化低秩适配器），通过正交化提升低秩增量的有效秩，解决标准 LoRA 的秩坍塌和物理量不平衡问题；④通过对 PDE 残差的子方程归一化和边界权重自适应调节，提升训练稳定性。

**🔧 技术方法**

主要技术包括：邻域注意力 Transformer、Newton‑Schulz 正交化的低秩适配器（NSLoRA）、基于 PDE 残差的无监督损失（包括边界损失和残差损失）、多尺度预训练、低秩适配（LoRA）和正交化策略。

**📊 数据集**

在预训练阶段使用 PDEBench 的六个子集（压缩 Navier‑Stokes 1D/2D/3D、浅水、扩散-反应、不可压 Navier‑Stokes）。在下游评估阶段使用 11 个数据集：The Well（Rayleigh‑Bénard、Shear Flow、Gray‑Scott、Active Matter）和七个基于解析解的 Exact‑Solution 数据集（Burgers 1D/2D、Advection 1D、Taylor‑Green 2D、Wave 2D、Advection‑Diffusion 2D、Burgers 2D、Advection 3D）。

**📈 对比分析**

与四个神经算子基线（FNO、TFNO、U‑Net、CNextU‑Net）以及两个 PDE 基础模型（PDE‑Transformer、Poseidon）进行对比。采用 VRMSE 作为指标，结果显示：在 11 个下游数据集上，所提方法在无监督条件下相较于 PDE‑Transformer 在 UPAO 上的 VRMSE 减少 9.9 倍，在 7/8 个 2D 数据集上距离监督 LoRA 仅差 2.5 倍，并在 9/11 个数据集上至少优于一条神经算子基线；在与随机初始化全参数微调相比，预训练提升了 1~2 倍；NSLoRA 相比标准 LoRA 在 8 个 2D 数据集平均降低 4% VRMSE。

**⚠️ 局限性**

局限性包括：①必须已知目标 PDE 的方程形式且可获得完整的边界观测，限制了在缺乏先验信息或边界观测稀缺的实际场景中的适用性；②PDE 残差监督对数值离散误差敏感，若数据包含离散误差，监督效果减弱；③实验仅验证单步预测，未检验自回归推理的长期稳定性；④对大规模、三维非结构化网格的适配和推理速度尚未全面评估。

---

## 296. FutureBridge: Token Selection Beyond Local Preference in Collaborative Decoding

**arXiv ID:** 2608.06819 | [PDF](https://arxiv.org/pdf/2608.06819v1)

**作者:** Quanquan Li `[一作]` (East China Normal University), Guitao Cao `[通讯]` (East China Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为FutureBridge的Token级小大模型协作框架，能在需要时让大模型为小模型提供最合适的单词插入，以提升数学推理任务的准确率。

**💡 创新点**

创新点在于用一个答案已验证的共享未来（shared future）来评估每个候选token对小模型后续推理的兼容性，从而生成候选级别的监督信号；并将这一信号蒸馏为无需未来信息的轻量级reranker。

**🔧 技术方法**

主要技术包括：联合候选池（SLM+LLM top‑k），答案验证的共享未来，冻结的SLM计算兼容性分数（平均对数似然），标准化并转化为soft标签，LoRA蒸馏的token reranker，以及固定的请求策略。

**📊 数据集**

在五个数学推理基准上进行实验：GSM8K、MATH‑500、OlympiadBench、AIME 2024 与 AIME 2025。

**📈 对比分析**

与基线（贪婪解码、Maj@8、S2T‑Local、R2R、Takeover 等）比较，FutureBridge在Qwen3‑1.7B上实现 Math Avg. 50.38%，相较于最强 SLM‑only 方法提升约10.18 分（约 25.3% 绝对提升），且在更弱小模型 Qwen3‑0.6B 上同样取得显著提升；匹配预算下仍保持 1.24 分的优势。

**⚠️ 局限性**

局限性包括：需额外的训练成本和对答案已验证轨迹的依赖；共享未来的长度与质量直接影响监督效果；在非数学推理或更复杂任务中其有效性尚未验证。

---

## 297. The Sparsity Whisperer

**arXiv ID:** 2608.06630 | [PDF](https://arxiv.org/pdf/2608.06630v1)

**作者:** Linghao Kong `[一作]` (Massachusetts Institute Of Technology), Nir Shavit `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于保留输出差异的单次剪枝方法，分别在一阶和二阶层面改进了传统的Wanda和SparseGPT裁剪策略；

**💡 创新点**

创新点在于将输入对差分作为重要性评估的核心信号，并针对MLP上升和门控投影中的Wasserstein神经元实现了神经元级差分重要性和差分Hessian重构；

**🔧 技术方法**

使用的技术包括差分归一化的重要性评估、神经元特定高差分对选择、轻量正则化的差分Hessian以及与RIA、ALPS等强力剪枝方法的无缝组合；

**📊 数据集**

实验采用了多种LLM体系结构（Llama 2/3.1、Mistral、Qwen、Granite）和WikiText‑2、Open LLM Leaderboard v1等评测数据集；

**📈 对比分析**

与基准方法比较时，该方法在所有稀疏度（50%、65%、2:4）下均超越Wanda和SparseGPT，且在多模型、多规模下实现了最优困惑度和下游任务平均分数；

**⚠️ 局限性**

限制主要体现在仅针对MLP上升/门控投影的差分信息，未扩展到MoE或后续微调，以及在稀疏度极端时对差分Hessian的稳定性需要进一步研究。

---

## 298. Learning GR(1) Specifications from Traces

**arXiv ID:** 2608.06546 | [PDF](https://arxiv.org/pdf/2608.06546v1)

**作者:** Sam Nicholas Kouteili `[一作]` (Yale University), Ruzica Piskac `[通讯]` (Yale University)

**通讯引用:** 1732 | [OpenAlex ID](https://openalex.org/A5045794652)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于 SAT 的算法，利用固定的 GR(1) 语法骨架，从 lasso 轨迹中学习可合成的假设-保证规范。

**💡 创新点**

创新点在于：①通过预先编码组件的 DAG 结构，只搜索命题内容而非时序运算符；②在枚举不同模板配置时共享并增量求解 SAT 约束，利用已学到的子句避免重复计算；③在模板中内置环境/系统变量分区，天然保证假设与保证的边界，生成的规范可直接用于合成。

**🔧 技术方法**

使用技术包括：命题 DAG 的 SAT 编码、临时变量评估、时序持有约束、正负轨迹的一致性约束；增量 SAT 求解（Push/Pop）、学习子句、利用 lasso 轨迹的循环结构；最后使用 GR(1) 合成器验证可实现性。

**📊 数据集**

实验数据集：60 条来自 Syntech 的布尔化 GR(1) 基准；以及 60 条来自 SYNTCOMP 的非 GR(1) LTL 规范（TLSF 格式），每个基准生成 20 条正轨迹和 20 条负轨迹。

**📈 对比分析**

与 ATLAS[LTL]（无约束 LTL 学习）和 ATLAS[GR1]（模板约束学习）在 3 分钟超时、单核 M1 Pro 上对比。新方法在 Syntech 上 100% 通过，ATLAS[LTL] 仅 55%，ATLAS[GR1] 37%；在 SYNTCOMP 上 63% vs 30% vs 1%。平均求解时间比 ATLAS[LTL] 快 10 倍、比 ATLAS[GR1] 快 2 倍，且所有 98 条学习到的规范均可实现，ATLAS[LTL] 仅 76% 可实现。

**⚠️ 局限性**

局限性：只能学习符合 GR(1) 骨架的规范，无法恢复明显非 GR(1) 的模式；对轨迹长度和变量数有限制；仅处理布尔化规范，无法直接学习包含算术或数组等结构的 LTL；若基准的真规范超出骨架，工具可能输出不可实现或不完整的结果。

---

## 299. Beyond Starry Night: Shortcut-Aware Control-State Planning for Artist-Grounded Text to Image Generation

**arXiv ID:** 2608.06751 | [PDF](https://arxiv.org/pdf/2608.06751v1)

**作者:** Kuan Xing `[一作]`, Yilin Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种短路感知的控制状态规划框架（Atelier），用于从含有艺术家名称的模糊文本请求中生成符合艺术家风格且避免不必要的典型短路的图像。

**💡 创新点**

创新点在于：①将艺术家意图拆解为可解释的控制状态（场景锚点、保持/变换策略、时期/风格假设、局部证据绑定、短路约束）；②利用艺术家知识库和局部图像片段进行检索绑定；③在闭环中通过全局与局部真实性评估迭代修正控制状态，从而显著减少“短路”现象。

**🔧 技术方法**

技术手段包括：大型语言模型（LLM）负责感知与规划，知识库检索（全球/局部层面）获取艺术家风格与局部证据，后端图像生成器（FLUX、Qwen、LongCat、Hunyuan等）根据规划生成图像，AuthCritic（Gemma-4+LoRA）做局部真实性评分，GlobalCritic做整体评估，反射记忆实现迭代改进。

**📊 数据集**

使用的主要数据集为：Van Gogh（公共域画作及其阶段划分）和Qi Baishi（中国传统水墨画），并构建了对应的艺术家知识库与局部图像片段；同时还使用了ArtIntentBench benchmark 进行量化评估。

**📈 对比分析**

与直接提示、检索增强、代理式提示扩展等基线方法对比，Atelier 在闭源和开源配置下在艺术家风格逼近、内容保持和短路抑制上均表现更好；在 ArtIntentBench 上 W_2 距离显著降低，短路率（SSR）从 48–78% 降至 31–52%，用户偏好也优于基线。

**⚠️ 局限性**

局限性包括：仅覆盖两位艺术家，知识库覆盖率不足导致某些变换缺失；闭源生成器对短路约束的接受度有限；生成质量受后端模型能力限制；缺乏对更广泛艺术家、媒介和交互式编辑的支持。

---

## 300. Implementation of Split Deadlines in a Large CS1 Course

**arXiv ID:** 2608.06753 | [PDF](https://arxiv.org/pdf/2608.06753v1)

**作者:** Hongxuan Chen `[一作]` (University of Illinois at Urbana-Champaign), Kathryn Cunningham `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在CS1大班课程中实施并评估了将项目提交期限拆分为两组的策略

**💡 创新点**

通过拆分单一截止日期，降低办公时间需求峰值并提升效率，而不影响学生成绩和公平感知

**🔧 技术方法**

采用课程自建的线上办公时间管理系统和问卷调查收集支持请求和学生反馈

**📊 数据集**

使用四个学期的办公时间请求日志、学生成绩记录以及春季2023学期的问卷数据

**📈 对比分析**

通过对比单一截止与拆分截止四个学期的平均请求量、等待时间和支持效率，并进行t检验、Mann‑Whitney检验和相关分析，结果显示支持效率提升约两倍，成绩无显著差异，学生普遍认为政策公平且有效

**⚠️ 局限性**

局限于低问卷响应率（18.9%），且缺乏对拆分截止法实施前的问卷数据，导致对公平感知的外推性有限

---

## 301. Online Correlation Clustering with Metric Weights

**arXiv ID:** 2608.06566 | [PDF](https://arxiv.org/pdf/2608.06566v1)

**作者:** Sami Davies `[一作]` (University of California Berkeley), Heather Newman `[通讯]` (Vassar College)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并分析了一个在在线环境下针对负权重满足三角不等式（即度量约束）的加权相似性聚类问题的确定性算法，证明其得到的聚类成本与离线最优解相比仅相差常数倍。

**💡 创新点**

创新点在于引入“中心移动”机制：在每个聚类的生命周期中允许中心（pivot）随节点加入而在相同密度阈值下逐步迁移，并通过“温和移动”保证相邻中心距离受限，从而突破传统基于固定中心的Pivot算法在在线模式下的Ω(n)下界。

**🔧 技术方法**

技术手段包括：将负权转换为半度量距离；利用密度阈值和阶段划分（size 2^ℓ）决定中心迁移或聚类停止；构造精细的充电（charging）方案把错误边的成本归入离线最优的成本；以及一系列基于三角不等式的距离与密度的定理来保证中心迁移的可控性。

**📊 数据集**

本工作为理论分析性研究，没有使用具体数据集；所有证明均基于抽象的图模型和数学分析。

**📈 对比分析**

与离线最优解的对比：算法实现的竞争比为常数（实验中给出一个具体上界108048，实际理论上可以进一步优化）。相较于传统在线算法的Ω(n)下界，提供了首个常数竞争的在线解决方案。

**⚠️ 局限性**

局限性：常数竞争因子较大，尚未证明是否为最优；算法对常数参数（r, ρ, ε）的选择较为粗糙；未给出实际实验验证；未讨论在更一般的输入或更强约束下的性能；以及对更优竞赛比的理论下界仍是开放问题。

---

## 302. Organizational and Socio-Technical Challenges in UAV Incidents: Evidence from a Practitioner Focus Group

**arXiv ID:** 2608.06472 | [PDF](https://arxiv.org/pdf/2608.06472v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 303. Confidence Estimation for Financial Vision-Language Models in Chart and Document Understanding

**arXiv ID:** 2608.06532 | [PDF](https://arxiv.org/pdf/2608.06532v1)

**作者:** Reza Khanmohammadi `[一作]` (Michigan State University), Mohammad M. Ghassemi `[通讯]` (Michigan State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在五个开源视觉‑语言模型（Qwen3‑VL‑8B、LLaVA‑NeXT‑13B、InternVL3.5‑14B、DeepSeek‑VL2、Gemma‑3‑27B）上，使用七种置信度估计器，对四个金融视觉‑语言问答基准（FinMME、FinChart‑Bench、MME‑Finance 英文、MME‑Finance 中文）进行零样本迁移评估，探究哪些答案可被安全自动化，哪些需交给人工。

**💡 创新点**

① 证明仅训练在自然图像上的内部探针（BICR、InternalInspector 等）能够提供良好校准的置信度；② 置信度可靠性呈现结构化模式，随模型、任务和语言变化；③ 采用可容错的选择性预测框架，评估在不同错误预算下可自动化的比例，发现置信度层在弱模型上提升最大。

**🔧 技术方法**

采用内部状态探针（SAPLMA、CCPS、InternalInspector、BICR）和推理‑仅方法（P(True)、Self‑Probing、Prompt Ensemble）；使用 Brier、ECE、AUROC、AUCPR 等评估指标；基于选择性预测（risk‑coverage、safe‑yield）框架进行安全自动化评估。

**📊 数据集**

探针仅在自然图像数据集 GQA（20k 训练 / 5k 验证）上训练；评估使用金融 VQA 基准：FinMME（11,099 题）、FinChart‑Bench（7,019 题）、MME‑Finance（1,171 英文 + 1,103 中文）。

**📈 对比分析**

与七种估计器的校准、判别力及可自动化比例进行比较。推理‑仅方法判别力高但过度自信；BICR 和 InternalInspector 在校准上表现最佳；在 5% 错误预算下，单模型安全自动化比例接近 0，只有在较宽松预算和易任务（如 FinChart‑Bench）才取得一定自动化；不同模型与任务间的排名差异显著。

**⚠️ 局限性**

仅使用单一自然图像数据集训练，未进行领域适配；正确性标签由模型判断，可能不够精确；评估聚合方式可能掩盖单模型细节；仅用第一词不变作为无图像基准，未完全覆盖全部无图像情况；实际系统需根据目标分布自适应阈值。

---

## 304. HazeSpikeMamba: Coupling Spiking-Inspired and State-Space Features for Self-Supervised Real-World Dehazing

**arXiv ID:** 2608.06886 | [PDF](https://arxiv.org/pdf/2608.06886v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 305. Does More Retrieved Evidence Help Visual Retrieval-Augmented Generation with Diffusion Language Models?

**arXiv ID:** 2608.07006 | [PDF](https://arxiv.org/pdf/2608.07006v1)

**作者:** Jiankun Wang `[一作]`, Chen Gao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了视觉检索增强生成（RAG）中扩散语言模型对检索证据数量的敏感性，提出了训练无关的Entropy‑Based Candidate Filter（ECF）框架，先通过多粒度视觉切片构造候选证据，再利用空白匹配的熵增评估候选的有用性，并通过rank‑prior策略决定最终传入生成器的证据。

**💡 创新点**

创新点在于揭示更大检索池会导致源协同损失（source‑coherence loss），并利用第一步答复块的熵差量化候选证据的贡献，从而实现只在必要时扩展证据而非无条件传递全部候选。

**🔧 技术方法**

使用了扩散语言模型（DLM）、多粒度视觉切片、空白匹配熵增计算、并行去噪的第一步答复块分析以及rank‑prior基于熵增与检索排名的候选选择算法。

**📊 数据集**

实验覆盖了五个视觉问答基准：ChartQA、InfoChartQA、DocVQA、InfoVQA 和 TATDQA。

**📈 对比分析**

在固定top‑k输入与五种训练无关的证据筛选方法（如UOVES、Answer‑UQ等）的对比中，ECF在三种DLM（LLaDA2.0‑Uni、LLaDA‑V、Dream‑VL）上平均提升约2.6个百分点，并在大多数模型‑数据集组合中取得最高准确率。

**⚠️ 局限性**

局限性包括对检索质量高度依赖，无法有效处理检索结果高度语义冲突或极少的相关页面；空白匹配只评估单一候选的影响，未考虑多候选交互或更复杂的跨图像依赖。

---

## 306. Bypassing Krum: Selection-Aware Backdoor Attacks in Federated Learning

**arXiv ID:** 2608.06637 | [PDF](https://arxiv.org/pdf/2608.06637v1)

**作者:** Srinivasan Subramanian `[一作]` (Kennesaw State University), Kazi Aminul Islam `[通讯]` (Kennesaw State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种针对Krum聚合规则的选择感知后门攻击（Krum-Proxy），能够绕过Byzantine-robust聚合并成功注入后门。

**💡 创新点**

创新性地引入随机参考构造、两阶段梯度优化、anchor-guided对齐与投影机制，显著提升对Krum/Multi‑Krum的攻击成功率与稳定性。

**🔧 技术方法**

使用两阶段优化、差分Krum代理、随机参考建模、anchor对齐、投影约束、梯度裁剪、交叉熵等技术。

**📊 数据集**

在CIFAR-10、MNIST、EMNIST三个标准图像数据集上进行实验。

**📈 对比分析**

与FedAvg、Krum、Multi‑Krum聚合以及Scaled Backdoor、Constrain‑and‑Scale基线比较；Krum-Proxy在Krum/Multi‑Krum下实现≈100%攻击成功率、≈90%主任务准确率，且攻击稳定；在标准FedAvg下表现与其他基线相当。

**⚠️ 局限性**

依赖本地参考估计，极度非IID或更新高度聚集时效果下降；仅针对Krum式距离聚合，对Median、Trimmed Mean等坐标聚合无直接适配；额外的参考构造和代理优化带来一定计算开销。

---

## 307. Invisible to the Machine: Auditing AI Restaurant, Cafe, and Bar Recommendation Against a Complete Market Census

**arXiv ID:** 2608.07069 | [PDF](https://arxiv.org/pdf/2608.07069v1)

**作者:** Vladimir Pitenin `[一作]` `[通讯]` (Norly Research), Vladimir Pitenin (Norly Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过对巴厘岛Canggu和Ubud两个有限市场内的4,776家咖啡馆、餐厅与酒吧进行完整枚举，随后对ChatGPT、Claude、Gemini和Perplexity四款生产型AI助手进行2,208次搜索式查询，系统性评估它们在本地餐饮推荐中的可见性、推荐内容、错误模式及跨引擎一致性。

**💡 创新点**

创新点包括：①使用完整市场“census”作为分母，首次量化AI推荐的可见性率并揭示大多数场所从未被推荐的事实；②发现AI可见性呈两边缘结构——入口取决于文档与评论量，而排名由评分决定；③证实Foursquare公开POI数据对可见性无正向影响；④将staleness（推荐已关闭场所）定位为主要失效模式；⑤展示跨系统的低一致性与重测稳定性，强调单次测评的噪声性。

**🔧 技术方法**

技术上，研究采用公共付费API实现搜索式查询，使用LLM抽取与实体匹配提取提及信息，并通过预注册实验设计和多重统计模型（GLM、GEE、条件Logit）进行可见性与排名边缘的因子分析，同时进行重测与相似度评估。

**📊 数据集**

数据集包括：①Google Places列出的4,776家餐饮场所及其属性；②Foursquare开放POI数据与质量等级；③通过Web搜索API获取的第三方提及与域名计数；④2,208次AI查询产生的12,439条有效提及；⑤对未匹配提及进行的名称变体与闭店校验等。

**📈 对比分析**

评估方法包括跨引擎顶级20列表的Jaccard相似度（0.33–0.54），重复相同查询的Jaccard重叠（0.22–0.45），以及两周后重测的相似度保持不降，表明系统行为在时间上稳定但受随机采样影响。性能表现显示：入口边缘由文档与评论量显著提升可见性，评分仅影响排名，且系统之间推荐内容差异显著。

**⚠️ 局限性**

局限性包括：①观测式设计仅揭示关联，缺乏因果性；②仅覆盖巴厘岛两个区域且使用英文查询，结果对其他地区和语言的推广受限；③依赖Google Places的列出信息，未覆盖未上榜的微型场所；④匹配错误与变量构造可能引入偏差；⑤仅评估四款系统且仅进行七天采集与两周重测，未覆盖更长周期或模型迭代的变化。

---

## 308. Fact-Check Your Information (FYI): A Design Probe to Understand How People Actually Fact-Check Data-Driven Articles

**arXiv ID:** 2608.06804 | [PDF](https://arxiv.org/pdf/2608.06804v1)

**作者:** Nguyen-Truong Thinh `[一作]` (Holistics Software), Arpit Narechania `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了名为 FYI 的浏览器扩展，集成了从检测到验证再到判定的数据声明事实核查工作流，并在一次包含 22 名受试者的实验中对其多模态使用方式进行观察与分析

**💡 创新点**

创新点在于将多种人工与人工智能工具（自动检测、对话式 AI、表格浏览、图表构建）整合在同一阅读环境中，形成“混合倡议”验证流程；通过实证揭示用户如何在不同工具间切换、如何通过可视化进行自我审计以及如何动态校准对 AI 的信任

**🔧 技术方法**

核心技术包括：① GPT‑4.1 进行句子级数据声明检测、自动验证及对话式辅助；② 基于 Pyodide 的 Python 代码执行；④ Vega‑Lite 交互式图表；⑤ 传统表格过滤与排序功能；⑤ 浏览器侧扩展与事件日志记录框架

**📊 数据集**

使用了一个包含 1,724 部电影数据（年份、全球票房、预算、评分等 8 个属性）的公开数据集，并基于此数据编写了一篇包含 6 条嵌入式数据声明的新闻文章作为实验材料

**📈 对比分析**

对比评估主要通过人工标注的六条声明真值与 FYI 自动化组件（Auto Check、AI Chat）以及受试者判断的一致性来衡量；两者与真值的匹配率分别约为 71% 与 76%，显示 AI 与人类在可验证声明上表现相近；实验未给出传统基准系统的性能对比，更多聚焦于使用模式与信任动态

**⚠️ 局限性**

局限性包括：① 仅在一篇电影领域文章上验证，缺乏跨域通用性；② 受试者多为数据/可视化背景，普适性待检验；③ 仅为 6 条声明提供真值，无法评估所有检测到的声明；④ 依赖 GPT‑4.1 的数值推理与可视化生成存在错误与幻觉风险；⑤ 未对系统整体效率或自动化水平与传统工具做定量性能比较

---

## 309. WebRider: Persona-Conditioned Intent Controllers for Live-Web Assistance

**arXiv ID:** 2608.06704 | [PDF](https://arxiv.org/pdf/2608.06704v1)

**作者:** Zhi Li `[一作]` (University of California, Los Angeles), Demetri Terzopoulos `[通讯]` (University of California, Los Angeles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出WebRider框架，将浏览任务分解为意图合约与分层控制，以实现可审计、可学习的网页代理；

**💡 创新点**

创新点在于将委托策略形式化为“意图合约”，并通过分层控制（最高层合同，次层守护动作，底层执行器）将决策与执行隔离，既保证策略可追溯，又提供可训练的动作接口；

**🔧 技术方法**

采用层次化控制器、JSON动作抽象、守护器（语法/状态检查）、基于浏览器/搜索/地图工具的执行器，以及大型语言模型（如Gemini、GPT-5.5）进行决策与监督；

**📊 数据集**

使用RiderBench数据集，包含4,096条基于42个公共网站、5-6种persona的任务合约与完整的浏览轨迹；

**📈 对比分析**

对比Full-Pro、Exec-Pro、IntentCore等模型，结果显示Full-Pro完成率99.2%但合约通过率仅38.8%，而IntentCore在合约通过率上升至约46.8%，人类评价则显示IntentCore的步驟一致性和委托舒适度更高；

**⚠️ 局限性**

局限在于仅处理公开网页任务，未解决CAPTCHA/访问封锁、长期记忆与对话式合同推理、以及完整端到端模型训练；

---

## 310. Educational Short Videos: Bibliometric Trends, Thematic Structure, and Operationalisation

**arXiv ID:** 2608.06932 | [PDF](https://arxiv.org/pdf/2608.06932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 311. Suppress and Diversify: Refining Robust Pathways for Corruption Robustness

**arXiv ID:** 2608.06712 | [PDF](https://arxiv.org/pdf/2608.06712v1)

**作者:** Jiangang Yang `[一作]` (Institute of Microelectronics, Chinese Academy of Sciences), Jian Liu `[通讯]` (Institute of Microelectronics, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Suppress and Diversify (S&D) 方法，在训练阶段通过显式识别并动态选择鲁棒计算路径，再通过对称保持的变换对其进行多样化，从而显著提升模型在自然图像失真下的鲁棒性。

**💡 创新点**

创新点在于：①首次系统分析内部鲁棒特征随网络深度的衰减及其对整体鲁棒性的影响；②设计了内存感知动态选择机制（MDSM）和结构一致路径微调（SPTS），实现无侵入、无参数、零推理开销的鲁棒提升；③通过对鲁棒路径进行对称保持的多样化，进一步增强鲁棒特征的多样性。

**🔧 技术方法**

使用了本地Lipschitz常数评估鲁棒特征、CKA相似度衡量路径稳定性、线性探针与损失景观分析、对称保留变换生成鲁棒路径，并在训练期间动态更新内存银行。

**📊 数据集**

评估基准包括 ImageNet-C、ImageNet-C̅、ImageNet-3DCC、ImageNetV2-C、COCO-C、VOC-C、Cityscapes-C、ADE20K-C 等八个鲁棒性基准，涵盖分类、检测与分割三大视觉任务。

**📈 对比分析**

与多种现有鲁棒训练方法（数据增强、正则化、子网络选择、权重演化、一致性学习、仿生设计）以及强大训练技巧（AugMix、Mixup 等）进行对比，S&D 在分类、检测、分割任务均实现平均 mCE 降低 1%–4%，mAP/mIoU 提升 1%–4%，并与最佳训练策略兼容，进一步提升性能。

**⚠️ 局限性**

局限性在于：1）存在清晰图像与失真图像性能权衡；2）实验仅覆盖中等规模模型与数据集，尚未验证在更大规模模型或全量 ImageNet 级别数据上的可扩展性；3）目前仅针对 2D/3D 视觉任务，未探究对其他模态的适用性。

---

## 312. AgentPatch: Coarse-to-Fine Weak-Task Repair for Merging Agentic Multimodal Large Language Models

**arXiv ID:** 2608.06699 | [PDF](https://arxiv.org/pdf/2608.06699v1)

**作者:** Zibo Shao `[一作]` (Chinese Academy of Sciences), Xiaoshan Yang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AgentPatch，一种训练无关的粗细两步修复框架，用于将不同任务的代理多模态大语言模型合并为单一通用模型。

**💡 创新点**

通过稳定骨干选择、弱任务唯一残差恢复和行为关键补丁三阶段，解决异质能力不均与行为关键遗忘问题。

**🔧 技术方法**

采用模型合并、权重插值、任务向量算术、参数级唯一残差插值、行为轨迹诊断、神经元激活评分与软插值等技术。

**📊 数据集**

使用六个代理与多模态基准：MMSearch、FactualVQA、AndroidWorld、OSWorld、V*Bench、HR-Bench 8K。

**📈 对比分析**

与 Weight Averaging、Task Arithmetic、TIES-Merging、TSVM、Iso-CTS、OptMerge、ACE-Merging、DC-Merge 等训练无关合并基线对比，AgentPatch 在平均分 56.6% 最高，显著提升 GUI 任务并保持搜索与视觉能力。

**⚠️ 局限性**

仅在单一基线骨干上验证，未探究跨模型规模与多语言适配；对极端任务互补度低时的效果仍有限。

---

## 313. Load-Path Redistribution and Damage Asymmetry in Reinforced Concrete Beams under Eccentric Drop-Weight Impact: A Coupled SPH--FEM Study

**arXiv ID:** 2608.06874 | [PDF](https://arxiv.org/pdf/2608.06874v1)

**作者:** Ziqi Gao `[一作]` (Kyushu University), Hiroki Tamai `[通讯]` (Kyushu University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了在不同冲击偏心度下，受力传递、能量吸收与损伤发展在受压混凝土梁中的变化，并通过匹配短跨梁实验验证偏心冲击的局部与整体耦合效应。

**💡 创新点**

创新点在于首次系统揭示冲击偏心导致的剪切转移显著偏向短跨侧、能量密度聚焦和碎屑形成，并证明单纯的短跨梁模型不足以描述完整结构响应。

**🔧 技术方法**

采用耦合的光滑粒子流体动力学–有限元（SPH–FEM）模型，配合GPU并行求解实现大变形、破坏与碎屑的自洽模拟。

**📊 数据集**

利用Tamai等人中心冲击实验数据进行验证，并在此基础上构建了一个包含冲击速度（2–4 m/s）、混凝土抗压强度（35–55 MPa）与偏心度（0–300 mm）的参数化仿真数据集。

**📈 对比分析**

通过对比首峰冲击力、冲击脉冲持续时间、剪切/弯矩极值包络、单位跨段吸能密度以及与匹配短跨梁的实际量值，模型在实验误差≤6%范围内重现关键响应，并揭示剪切放大系数最高达2.23倍、能量密度放大系数最高达4.29倍。

**⚠️ 局限性**

局限性包括仅考虑简支单跨梁、有限的混凝土强度范围、缺乏针对偏心冲击的实验验证以及潜在的粒子/网格分辨率对局部损伤细节的影响。

---

## 314. Capek 0.5: An Execution-Centric Vision-Language Model for Embodied Intelligence

**arXiv ID:** 2608.06756 | [PDF](https://arxiv.org/pdf/2608.06756v1)

**作者:** Ying Chen `[一作]`, Jie Chen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种以执行为中心的统一视觉-语言模型，旨在增强机器人感知和推理能力。

**💡 创新点**

创新点在于引入了一个执行中心的能力分类法，将能力分为空间推理、时间理解、行动指导和状态验证四个类别，并通过强化学习和权重合并将这些能力整合到一个模型中。

**🔧 技术方法**

使用了强化学习、权重空间合并和路由政策空间蒸馏等技术。

**📊 数据集**

使用了多个数据集，包括SenseNova-SI-8M、CLEVRER、Charades、PixMo等，涵盖空间推理、时间理解、行动指导和状态验证等任务。

**📈 对比分析**

通过与基线模型的比较，展示了在多个基准测试中的性能提升，尤其是在执行相关的能力上，显示出显著的改进。

**⚠️ 局限性**

限制在于模型的复杂性和训练过程中的优化干扰，可能导致不同能力之间的相互影响。

---

## 315. How Should I Pick a Foundation Model for My Robot? In Favor of a Community Evaluation Framework for Social Robots

**arXiv ID:** 2608.06898 | [PDF](https://arxiv.org/pdf/2608.06898v1)

**作者:** Eric Nichols `[一作]` (Honda Research Institute Japan), Hatice Gunes `[通讯]` (University of Cambridge)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了三层评估漏斗框架，以系统化地评估和选择社交机器人使用的基础模型；

**💡 创新点**

创新点在于将社交机器人评估拆分为五个维度（对话能力、安全性、实体角色、场景效果、受众适宜性），并构建了从静态基准到仿真交互再到机器人特定评估的渐进漏斗，提供 Pareto 前沿以兼顾性能与部署成本；

**🔧 技术方法**

采用现有文本基准（MMLU、CommonsenseQA、HellaSwag、Social IQa、TruthfulQA、IFEval、XSTest、OR‑Bench 等）结合 LM‑eval、vLLM 进行静态评估；通过 LLM 模拟用户和评判员完成多轮对话仿真；最后通过平台专用 harness 在真实机器人上验证；

**📊 数据集**

使用的主要数据集为上述基准集及其对应的多轮对话脚本，用于 LLM 角色扮演的仿真交互；

**📈 对比分析**

通过在每个层级上计算模型在各维度上的分数及部署成本（VRAM、延迟），生成 Pareto 前沿供研究者选择；在实际案例中，该方法可将候选模型从数十个缩减到一两个，显著降低实验成本，且在模拟层已获得与人类评判相近的安全与对话质量评分；

**⚠️ 局限性**

局限性包括：第三级仍需机器人特定的 harness 与现场用户试验，难以通用；多模态感知模型尚未完整融入漏斗；缺乏对 harness 与真实交互差距的量化评估，需要社区共同完善。

---

## 316. Characterizing the Quality Profile of AI-Generated C++ in Production

**arXiv ID:** 2608.06640 | [PDF](https://arxiv.org/pdf/2608.06640v1)

**作者:** Michael Tran `[一作]` (Google), Parthasarathy Ranganathan `[通讯]` (Google)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在大型企业单仓库中，对AI生成的C++代码进行全生命周期的纵向研究，涵盖从代码生成、提交、审查、运行监控到基于静态检查的反馈干预。

**💡 创新点**

创新点：①采用作者时刻的生成 provenance 直接追踪 AI 代码在提交与审查过程中的演变；②构建针对 C++ 的静态问题三层分类法，揭示 AI 代码的主要弱点；③首次将静态发现与运行时 CPU/内存占比关联，量化 AI 代码在生产中的资源消耗；④设计基于分类法的 prompt 反馈实验，验证针对性干预能显著降低静态缺陷并提升执行效率。

**🔧 技术方法**

技术手段：作者时刻 provenance 收集、提交级别聚合、静态分析工具整合、审查指标抽取、生产监控数据对齐、功能级映射、回归与分层对照模型、prompt 反馈实验。

**📊 数据集**

数据集：3.52 百万提交变更（涵盖多语言），10.46 百万行 C++ 代码（含 350k 可审查变更），70k 功能级计算观测（CPU/内存），以及 50 个人工标注的 C++ 函数基准用于反馈实验。

**📈 对比分析**

比较方法：分层队列对照（按月份、变更规模、组织片段等）与分层回归模型；对比 AI 生成与纯人工代码在构建失败、回滚率、审查阻塞、评论量、计算成本等指标上的差异。实验表明 AI 代码的构建失败率 ~1.3×，回滚率 ~0.9×，但 CPU 成本增长率 1.31×；针对性反馈后静态发现减少 11.1%，执行效率提升 31%。

**⚠️ 局限性**

局限性：①仅针对单一企业 C++ 单仓库，缺乏跨语言/跨组织验证；②作者时刻 provenance 与模型版本混杂，无法区分不同模型的影响；③观测性研究，仍存在任务难度、作者经验等潜在混杂；④部分静态检查类别支持不足，导致细粒度分析受限；⑤内部数据不可公开，重现性受限。

---

## 317. Rhetorical-Role-Aware Retrieval-Augmented Generation for Legal Question Answering over Indian Supreme Court Judgments

**arXiv ID:** 2608.06828 | [PDF](https://arxiv.org/pdf/2608.06828v1)

**作者:** Sayed Ayaan Ahmed Sha `[一作]` (National Institute of Technology), Navya Binu `[通讯]` (National Institute of Karnataka)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个基于修辞角色的检索增强生成（RAG）框架，用于印度最高法院判决的法律问答；

**💡 创新点**

创新点在于引入修辞角色分块、意图感知过滤、双重稠密+稀疏索引、跨编码器重排序，并利用对话历史和查询重写提升检索上下文的相关性；

**🔧 技术方法**

使用了BERT/GPT系列模型（如Gemini、Qwen3）、BM25稀疏检索、双编码器向量检索、跨编码器重排序、LLM生成、意图分类与重写、对话历史提示等技术；

**📊 数据集**

采用了30份印度最高法院判决（10份民事、10份公司、10份刑事），每份已标注6种修辞角色，来自Indian Kanoon的数据集；

**📈 对比分析**

通过自动化LLM评估指标（Faithfulness、Answer Relevancy、Contextual Relevancy）与Qwen3-32B、Gemini-2.5-flash、Gemini-2.5-Pro三模型比较，在三类法律领域均显示Gemini-2.5-Pro在Faithfulness上表现最佳，整体性能稳定；

**⚠️ 局限性**

仅在印度最高法院判决上评估，缺乏人类专家评测，结果可能不适用于其他司法体系、低级法院或非英语判决。

---

## 318. Plan-and-Avoid: Real-Time Aircraft Trajectory Coordination in a Multi-Agent Environment

**arXiv ID:** 2608.06648 | [PDF](https://arxiv.org/pdf/2608.06648v1)

**作者:** Huseyin Emre Tekaslan `[一作]` (Virginia Polytechnic Institute and State University), Natasha Neogi `[通讯]` (NASA Langley Research Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种实时的Plan‑and‑Avoid框架，用于在宣告优先航迹的多机协同空域中预测并解决接触冲突，保证优先航迹的完整性并实现安全分离。

**💡 创新点**

创新点在于将冲突感知、优先航迹规划与车辆约束的单方决议建议统一到一个端到端系统；引入基于不确定性膨胀的well‑clear判断；以及层级化、可解释的决议建议生成（停机、速度、垂直、扩展、偏离）和实时性能保障。

**🔧 技术方法**

使用离散搜索式紧急着陆规划、概率不确定性阈值膨胀、基于约束的二次规划与全局优化、Dubins路径扩展，以及动态多机仿真验证。

**📊 数据集**

使用华盛顿特区ADS‑B实时轨迹数据，包含超过900个强制着陆案例（140小时飞行），以及多机仿真中的真实航空器模型。

**📈 对比分析**

与几何Dubins基线相比，冲突感知规划在LoWC持续时间、冲突严重度和累计LoWC暴露方面分别降低约37.7%、45.5%和3.7h；整体计算时间平均约0.45s，最坏情况5.7s（含1s数据链延迟）。大部分建议满足RTCA DO‑365 35s的时间阈值。

**⚠️ 局限性**

仅适用于共享轨迹意图的合作空域；依赖预先通信和固定数据链延迟；未考虑非合作或多重故障情形；在实机或人机实验中尚未验证。

---

## 319. Multi-Agent Forensic Reasoning for Generalizable Deepfake Video Detection

**arXiv ID:** 2608.06865 | [PDF](https://arxiv.org/pdf/2608.06865v1)

**作者:** Xuechao Zou `[一作]` (Beijing Jiaotong University), Junliang Xing `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了100,000张面部深度伪造视频的FaceVid-Forensics-100K数据集，并提出了四个专用视角的多代理法医推理框架。

**💡 创新点**

创新在于同时提供细粒度文本注释（纹理、光照、运动、物理）并通过多代理协同推理显著提升对新型伪造方法的泛化能力。

**🔧 技术方法**

采用多模态大型语言模型（如Qwen2.5‑VL‑7B）进行监督微调（SFT）与组相对策略优化（GRPO），并将推理分解为纹理、光照、运动、物理四个专用观察代理和一个判定裁判代理。

**📊 数据集**

使用FaceVid-Forensics-100K数据集（33种合成方法，21,075真视频、78,925伪造视频）。

**📈 对比分析**

在OOD测试集上，系统实现69.87%准确率、81.82%召回率、53.28% F1，显著优于所有对照方法（小型视觉模型、通用/专用LLM、法医调优LLM）。

**⚠️ 局限性**

局限包括仅针对面部视频，依赖多模型协同导致推理延迟与计算成本，且对极端压缩、不同分辨率或多模态场景的鲁棒性尚待进一步验证。

---

## 320. CustomDance: Customized 3D Dance Generation with Coarse-to-Fine Human-Centered Interactive Control

**arXiv ID:** 2608.06722 | [PDF](https://arxiv.org/pdf/2608.06722v1)

**作者:** Xulong Tang `[一作]` (University of Texas at Dallas), Rawan Alghofaili `[通讯]` (University of Texas at Dallas)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于粗细分层的人工交互式3D舞蹈生成系统，允许用户从音乐、文本、身体部位控制等多模态输入出发，逐步构建、检索、填充并精细调整舞蹈片段，最终生成符合创意需求的完整舞蹈。

**💡 创新点**

创新点在于：①将舞蹈创作流程拆解为三阶段（动机规划、片段检索、完成与细化）并与多模态大型语言模型（MLLM）、多模态检索器和音乐条件扩散模型相结合；②为每个阶段分配专门的多模态输入并设计可交互的界面；③利用诊断可视化（KE 曲线）帮助用户定位和局部重建运动缺陷。

**🔧 技术方法**

技术包括：Gemini MLLM用于分析音乐与文本生成时机锚点与创意提示；三模检索器（音频+文本+身体部位控制）结合 FiLM+Transformer；音乐条件DDIM扩散模型（BiMamba+Transformer）进行片段填充与局部重绘；基于kinetic-energy proxy的诊断与修复模块；统一SMPL表面表征与离散时序检索。

**📊 数据集**

使用了 10.7 小时的舞蹈动作库（来自 FineDance 7.7 小时 + 3 小时 HMR 重建），包含 16 个细分舞种与 5 个粗粒度风格；训练集拆分为 8 秒片段用于扩散模型，4 秒片段用于检索；实验音乐为 5 首 32 秒曲目。

**📈 对比分析**

通过三组对比实验：①作者体验（对比去掉各阶段的 ablation）；②舞蹈质量（与 Lodge、MEGADance 进行盲测排名及 FID/BAS/Diversity 指标）；③长序列（90 秒）对比（与手工排练和专业排练）。结果显示系统在用户控制感、姿态满意度与描述实现度上均优于基线，并在 FID、BAS 与多样性上取得最优指标；在长序列评估中，所提系统在时间、FSR、姿态满意度等方面均优于现有自动生成方法。

**⚠️ 局限性**

局限性包括：检索片段固定长度限制，导致某些风格或长节拍段难以匹配；检索库覆盖不足可能影响少数族群舞蹈；计划模型对嘈杂音乐或复杂节奏鲁棒性有限；自动修复无法完全解决语义与风格不匹配以及接触/碰撞问题；系统对多模态输入的可靠性仍受外部 API 质量影响。

---

## 321. Georeferencing Non-Gazetteered Place Names using Biological Specimen Records

**arXiv ID:** 2608.06884 | [PDF](https://arxiv.org/pdf/2608.06884v1)

**作者:** Aneesha Fernando `[一作]` (Massey University), Christopher B. Jones `[通讯]` (Cardiff University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用生物标本记录的地点描述，提取并定位未被gazetteer收录的地名（NGP），并建立对应的空间约束。

**💡 创新点**

首次系统化地将标本记录作为非官方地名的来源，并比较确定性、概率性与LLM三种方法在地名空间化任务中的表现，验证传统概率推理在高精度定位上的优势。

**🔧 技术方法**

使用spaCy NER提取地点与空间关系，构造Approximate Location Region (ALR) 的确定性方法、基于距离/方向的概率似然模型，以及零样本LLM推理（GPT‑5.1）完成地名定位。

**📊 数据集**

基于新西兰 Allan Herbarium 的标本数据，筛选包含坐标与自由文本地点描述的记录，并从中生成 365 个伪 NGP（真实地名但隐藏坐标）作为评估基准。

**📈 对比分析**

对同一 365 个伪 NGP 采用三种方法进行定位，并以平均误差、median误差、A@1km、A@161km 等指标比较。概率模型取得最佳结果（median 1.43 km，A@1 = 36%），LLM次之（median 1.80 km，A@1 = 31%），确定性模型最差且缺失 108 例。

**⚠️ 局限性**

仅支持点状地名，需事先定义并解析空间关系词，LLM缺乏可重复性；确定性方法对约束冲突敏感；概率方法在仅方向约束时表现不佳；整体依赖高质量坐标与多重描述，无法处理线状或多边形地理实体。

---

## 322. Bend the Basics: Degradation-Aware Deformable Tokenization for All-in-One Image Restoration

**arXiv ID:** 2608.06832 | [PDF](https://arxiv.org/pdf/2608.06832v1)

**作者:** Zihao He `[一作]` (Shanghai Jiao Tong University), Songhua Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一图像恢复框架FIT，使用退化感知可变形分块(tokenization)实现端到端的退化自适应；

**💡 创新点**

创新点在于将退化信息（全局向量g与空间地图M）直接用于patch embedding与unembedding的可变形采样，使patch边界随退化结构自适应；同时引入任务令牌丢弃(TTD)增强对未知或混合退化的鲁棒性；

**🔧 技术方法**

技术包括：双粒度退化编码器（DGDE）产生g和M；FiLM调制的可变形patch embedding/unembedding；AdaLN Transformer块；任务令牌丢弃；损失设计（重建、类型/强度监督、seam、offset正则化）；

**📊 数据集**

数据集：BSD68、Rain100L、SOTS、GoPro、LOLv1，用于五种恢复任务（去噪、除雨、去雾、去模糊、低光增强）；

**📈 对比分析**

与三类和五类统一恢复基准（如AirNet、IDR、PromptIR、Gridformer、InstructIR、AdaIR、VLU-Net、JIT等）对比，FIT在三退化设定下平均PSNR 32.83dB，五退化设定下30.72dB，均比现有SOTA提升0.5-1.4dB，并在去雨、去雾、去模糊等空间非均匀退化上提升显著；

**⚠️ 局限性**

局限性包括：对大范围混合退化仍可能受限于退化编码的表达能力；可变形采样的计算开销相对传统固定patch较高；模型对极端退化或新型退化的泛化尚未充分验证。

---

## 323. Corrupting Attention: Evasion-Based Adversarial Attacks on Encoder Attention in Detection Transformers

**arXiv ID:** 2608.06674 | [PDF](https://arxiv.org/pdf/2608.06674v1)

**作者:** Ridma Jayasundara `[一作]` (Queensland University of Technology), Clinton Fookes `[通讯]` (Queensland University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对检测变压器的编码器注意力直接进行攻击，使用不可察觉的扰动将注意力逼近结构化目标；

**💡 创新点**

首次直接优化编码器注意力而非检测输出，覆盖全局自注意力与可变形注意力，且不依赖可视化补丁；

**🔧 技术方法**

利用投影梯度下降对输入进行ℓ∞约束，针对四种结构化腐蚀目标（分散、重新排序、置换、峰值抑制）进行损失最小化；

**📊 数据集**

使用MS COCO 2017验证集进行实验；

**📈 对比分析**

与现有攻击（如AFOG、AttentionFool等）在相同扰动预算（ε=8/255）和迭代次数（10）下对比，DETR-R50 mAP从42.1降至0.97，DINO‑Swin‑L从56.8降至1.44，显著优于现有最强攻击；

**⚠️ 局限性**

仅针对检测变压器，未评估更大规模或其他网络的普适性，攻击仍需要白盒访问，且未提出相应的鲁棒防御方法。

---

## 324. MuST-VAD: Mutual Structured Learning for Video Anomaly Detection

**arXiv ID:** 2608.06913 | [PDF](https://arxiv.org/pdf/2608.06913v1)

**作者:** Satoshi Hashimoto `[一作]` (KDDI Research, Inc.), Mori Kurokawa `[通讯]` (KDDI Research, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 MuST-VAD 框架，实现弱监督视频异常检测中的视觉–语言模型与 MIL 检测器的互相学习循环。

**💡 创新点**

创新点是将 LVLM 与检测器的单向知识转移变成双向互学习，通过关键片段选择、结构化问答、置信度加权的双向蒸馏在小视频组中交替更新两模型。

**🔧 技术方法**

技术包括视觉语言模型 Qwen3‑VL‑8B、MIL 检测器 UR‑DMU、固定特征提取提示、结构化 QA 任务、置信度加权二元交叉熵蒸馏以及分组异步优化。

**📊 数据集**

使用 UCF‑Crime 数据集进行实验。

**📈 对比分析**

与单次表示迁移以及多种基线方法比较，在 UCF‑Crime 上 AUROC 88.63%，AP 42.46%，比上一最佳 AP 提升 4.13 分，Recall@FPR 在 2%/3% 处表现最优。

**⚠️ 局限性**

局限在于 AP 提升虽显著但 AUROC 仍低于 DSANet，且在极低 FPR（1%）下召回略低，说明模型对高置信度误警仍有不足。

---

## 325. AVCap: Reinforcing Audio-Video Joint Caption with Detail-Aware Reward

**arXiv ID:** 2608.06930 | [PDF](https://arxiv.org/pdf/2608.06930v1)

**作者:** Mingyang Wu `[一作]` (Chinese University of Hong Kong), Xiangyu Yue `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 AVCap-100K 数据集、基于 Da‑GRPO 的 AVCap 模型以及 AVCap‑Bench & AVCap‑Score 评测体系，实现了细粒度音视频字幕生成。

**💡 创新点**

创新点在于：①构建了 10 万条音视频细粒度对齐字幕的数据集；②提出 Detail‑Aware GRPO（Da‑GRPO）通过细粒度 QA 反馈实现奖励稠密化；③设立了专门针对原子级事实验证的 benchmark 与指标。

**🔧 技术方法**

技术核心包括：多阶段音视频拆分与同步标注、Qwen3‑Omni‑思维解码、Demucs 语音分离、组相对策略优化（GRPO）与 Da‑GRPO 奖励机制、以及自动化 QA 验证判别器。

**📊 数据集**

使用的数据集为 AVCap‑100K（100K 视频-字幕对），以及公开的 AVE、VGG‑Sound、Condensed Movies、AVQA、Trailer30K、MPII‑MVAD、YouTube 片段等源数据。

**📈 对比分析**

在 Video‑SALMONN‑2、UGC‑VideoCap、DailyOmni、WorldSense 等公开基准上，AVCap‑30B 在 AVCap‑Score 取得 56.94 分，显著优于所有开源同规模模型，并与 Gemini‑2.5‑Pro 在多项指标上相当；在细粒度评测上，AVCap‑30B 的视觉/音频/联合子分数均超过 57 分。

**⚠️ 局限性**

局限性包括：对超大规模模型与长期视频仍需更多算力；奖励生成依赖预训练判别器，可能受其偏差影响；数据集中仍可能存在细节标注噪声，需进一步人工校验。

---

## 326. Flaky Test Recognition when Testing CPSs Using Hybrid Models

**arXiv ID:** 2608.06535 | [PDF](https://arxiv.org/pdf/2608.06535v1)

**作者:** Zahra Sadri-Moshkenani `[一作]` (North Carolina State University), Gregg Rothermel `[通讯]` (North Carolina State University)

**通讯引用:** 19564 | [OpenAlex ID](https://openalex.org/A5052249797)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种面向混合模型的CPS测试技术 HyTestTF，集成了测试结果验证（TReVa）和易失测试识别（FlaRe），能够在一次额外测试轮内识别真正的易失测试用例及其对应的易失条件。

**💡 创新点**

创新点：①首次将 TReVa 与 FlaRe 与 HyTest 结合，构建完整的“测试生成–执行–结果验证–易失识别”闭环；②利用混合自动机的条件图来识别潜在的易失模式和条件，避免传统“重跑”方式的高成本与不确定性；③实现了在 MiL（Model‑in‑the‑Loop）级别一次性完成验证与易失检测。

**🔧 技术方法**

主要技术：HyTest 基于混合自动机模型的测试用例生成与测试 Oracle；TReVa 通过重执行、比较结果实现验证；FlaRe 利用条件图判定可能易失模式、提取易失条件并验证其真实性；整体实现使用 Java（模型解析、图生成）与 MATLAB/Simulink（仿真、结果收集）。

**📊 数据集**

数据集：选取 5 个典型 CPS（Cruise Control、Inverted Pendulum、Hexapod、Rooms and Heaters、Automatic Transmission），在 Simulink 生成 MiL 级别的仿真模型；通过注入故障变异和随机噪声产生噪声模型，构成实验样本。共使用 10 轮实验，生成不同数量的噪声模型与故障模型进行评估。

**📈 对比分析**

比较方法：与原始 HyTest 进行对比，分别统计真实失败、真实通过、真实易失（通过/失败）以及易失比例。结果显示 HyTestTF 能够识别约 30‑55% 的易失测试用例，并正确区分真实失败/通过，且在识别易失的同时保持与 HyTest 相当的缺陷发现率。性能方面，额外一轮测试的成本显著低于传统“重跑”方式，且验证过程不需要额外的模拟或硬件资源。

**⚠️ 局限性**

限制：①实验仅在 MiL 级别验证，未覆盖 SiL/HiL 级别的实际硬件影响；②仅使用 5 个实验 CPS，缺乏大规模工业系统的验证；③易失条件的注入方式仅限于随机噪声，未涵盖所有真实环境不确定性；④对算法在极大规模模型下的可扩展性和执行时间未做系统评估。

---

## 327. Dual-Space Modality Consistency Learning for Universal Cross-Modal Re-Identification

**arXiv ID:** 2608.06943 | [PDF](https://arxiv.org/pdf/2608.06943v1)

**作者:** Yujian Zhao `[一作]` (Beihang University), Guanglin Niu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了双空间模态一致性学习（DSMCL）框架，实现跨模态ReID的统一、可插拔特征学习。

**💡 创新点**

创新点在于同时对空间分布和频域高频表示进行一致性约束，利用高频信息的辨识性与模态敏感性提升跨模态匹配性能。

**🔧 技术方法**

技术手段包括高斯分布对齐的空间一致性学习（SMCL）与基于对比学习的频域辨识一致性学习（FDCL），并与标准分类、三元组损失联合优化。

**📊 数据集**

实验涵盖 SYSU-MM01、RegDB、LLCM（可见-红外人脸）以及 HOSS-ReID、CMShipReID（光学、SAR、NIR、TIR船舶）等五大基准数据集。

**📈 对比分析**

与多种基线（IDKL、MFENet、HSFLNet、TransOSS、MOS 等）对比，DSMCL 在 17 个评估协议中均实现显著提升，部分协议 Rank‑1 提升超过 5% 以上，mAP 同样提升 3–6%。

**⚠️ 局限性**

局限在于需在训练阶段额外计算频域特征与分布对齐，且对超参数（λ_s、λ_f、温度 τ）有一定敏感性，未来可探索更自适应的模态感知一致性机制。

---

## 328. Octo's Adventure: At-home Deployment of a Pediatric Education Tool

**arXiv ID:** 2608.06684 | [PDF](https://arxiv.org/pdf/2608.06684v1)

**作者:** Crimson Olaleye `[一作]` (University of Minnesota), Carlye Anne Lauff `[通讯]` (University of Minnesota)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在儿童先天性心脏病家庭中部署并评估 Octo——一款结合物理玩具与数字应用的混合健康教育工具，进行为期一周的在家使用实验。

**💡 创新点**

创新点在于将参与式设计方法（绘图、虚构探询等）延伸至部署阶段，通过孩子主导的反思、绘画和情绪贴纸获取直接体验；以及将物理模型与数字游戏无缝结合，支持多感官学习与家庭沟通。

**🔧 技术方法**

使用 Octo 物理原型（恐龙+心脏拼图+医学玩具）、配套的故事书和移动游戏、情绪贴纸地图、7 天活动日志、纸笔反馈，以及 Miro 平台进行数据数字化和主题分析。

**📊 数据集**

数据集为 13 个家庭的孩子与父母日记、情绪贴纸记录、照片及家长用户指南等原始实验材料，未采用公开数据集。

**📈 对比分析**

与传统以父母报告为主的评估方法对比，研究通过定性主题分析展示孩子对心脏结构的理解提升、情绪积极性增强以及家庭沟通频率和质量改善；未进行量化性能对比，仅以定性提升为依据。

**⚠️ 局限性**

局限性包括：实验周期短（仅一周），样本相对单一（主要为白人家庭），缺乏对照组，且儿童自述易受父母影响，整体结果缺乏可量化评估。

---

## 329. Bridging the Gap Between Hyperdimensional Computing and Kernel Methods via the Nyström Method

**arXiv ID:** 2608.06860 | [PDF](https://arxiv.org/pdf/2608.06860v1)

**作者:** Quanling Zhao `[一作]` (UC San Diego), Tajana Rosing `[通讯]` (UC San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于Nyström核逼近的超维编码方法NysHD，用于将任意正定核映射到HDC空间，实现更好的相似性保留。

**💡 创新点**

创新点在于将核方法与HDC结合，利用Nyström近似生成低精度随机化嵌入，克服传统HDC仅能捕获简单相似性的局限，并支持图、字符串等复杂结构数据。

**🔧 技术方法**

使用Nyström核逼近、随机超平面取符号化、随机投影以及传统HDC学习（感知器）等技术。

**📊 数据集**

在图数据集（ENZYMES、NCI1、D&D、BZR、MUTAG、COX2、NCI109、Mutagenicity）和字符串数据集（Protein、SMS、Splice、Promoter）上进行评估。

**📈 对比分析**

与GraphHD、N-gram HDC、以及多种深度图神经网络（DGCNN、GCN、GIN、GIUNet）对比，NysHD在图数据集平均提升11%准确率、在字符串数据集平均提升17%；训练速度与传统HDC相近，显著快于大多数DNN模型。

**⚠️ 局限性**

主要限制是需对核矩阵进行计算，导致额外的时间开销；选择和数量的基准点（landmarks）对逼近质量影响大，规模化时需要更高效的采样策略。

---

## 330. Exact Thrust-Reversal Limits of Bidirectional Propellers under Bounded Motor Inputs

**arXiv ID:** 2608.06991 | [PDF](https://arxiv.org/pdf/2608.06991v1)

**作者:** Ahmed Ali `[一作]` (University of Twente), Antonio Franchi `[通讯]` (University of Twente)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了双向螺旋桨在受限电机输入下的推力轨迹可重现性，并给出了推力零点跨越时电机命令的可行性条件。

**💡 创新点**

创新点在于通过将双向推力系统转换为推力坐标常规形式，系统性地证明了：除非推力零点处足够平滑，否则在有限电机输入下无法实现精确的推力反转；并据此提出了针对不同电机输入平滑度的设计规则。

**🔧 技术方法**

主要技术包括：动力学建模、推力坐标常规化、可重现性条件的定理推导、平滑性分析、以及对 DC 电机的电流与电压参考转换。

**📊 数据集**

实验采用单个 12 V ZYTD‑38S‑R 直流电机与 8 g 双向固定叶片的硬件平台，对比了线性、五次幂和七次幂的推力零点跨越实验；未使用公开数据集。

**📈 对比分析**

通过实验验证，线性推力跨越产生明显的电流/电压尖峰和推力跟踪误差，均值绝对误差约 7.3 Hz；而五次幂与七次幂跨越误差显著下降，均值绝对误差分别约 1.7 Hz 与 1.3 Hz，推力 RMS误差从 0.10 N 降至 0.036 N 与 0.031 N。

**⚠️ 局限性**

局限性包括：仅在单一一维推力通道上验证；实际多旋翼平台中其他耦合动力学未考虑；推力平滑设计需要额外的参考生成与调度复杂度；并且对极低速或极大负载下的电机非线性影响未做深入分析。

---

## 331. MIRA: Evidence-Verified Repair Memory for Text-to-SQL Correction

**arXiv ID:** 2608.06950 | [PDF](https://arxiv.org/pdf/2608.06950v1)

**作者:** Yining Liu `[一作]` (Beijing Institute of Technology), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于历史修正记录的SQL纠错框架MIRA，能够将历史错误修正拆分为独立可复用的记忆项，并在推理时根据问题、当前SQL和数据库证据激活并适配这些记忆项，从而纠正Text-to-SQL生成的语义错误。

**💡 创新点**

创新点在于：①将历史修正分解为可单独判断的修复单元；②在激活过程中结合数据库证据以防止错误迁移；③针对每个记忆项进行局部适配，保持原有正确逻辑。

**🔧 技术方法**

技术手段包括：记忆项构造、基于语义与结构的检索、数据库证据检验、局部适配和多项记忆合成。

**📊 数据集**

使用BIRD和ScienceBenchmark两大跨域基准，涵盖14个数据库。

**📈 对比分析**

与三种Text-to-SQL上游系统（CHESS、DeepEye-SQL、OmniSQL-32B）对比，MIRA在BIRD上提升执行准确率16.53%，在ScienceBenchmark上提升8.78%，共修复261条错误，导致18条回归。

**⚠️ 局限性**

局限在于：对数据库证据的依赖可能受数据完整性限制；记忆构造与检索开销；对极端复杂的多步骤修复仍存在一定误差。

---

## 332. Endpoint Sufficiency Behavioral Quotients

**arXiv ID:** 2608.06386 | [PDF](https://arxiv.org/pdf/2608.06386v1)

**作者:** David Carr `[一作]` `[通讯]`, David Carr

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在生成系统中，何时可以在保留合法未来行为的前提下忘记出处信息，提出三层足够性概念并给出精确的判据与分割算法。

**💡 创新点**

首次将端点足够性划分为启用、有限迹线与分支三层，并证明它们严格层次关系，同时提出最大安全忘记的两种规范化等价关系。

**🔧 技术方法**

采用轨迹等价、强分歧（bisimulation）与分区细化（partition refinement）等经典转移系统工具，并通过交叉与并集构造端点保留等价。

**📊 数据集**

以可归约的嵌套递归重组生成（NRRG）图系统为示例，对两个相同端点但不同继承历史的实例进行验证。

**📈 对比分析**

通过分区细化算法计算出最大端点保留分支商；实验示例表明，该算法在有限系统上在多轮细化后即可分离行为差异，但未给出数值性能评估。

**⚠️ 局限性**

局限于有限系统，未处理无穷行为、弱转换、多父亲语义或概率权重；对不可判定的可接受性法则和最小化存储结构仍需进一步研究。

---

## 333. Adversarial Causal Intervention Falsification

**arXiv ID:** 2608.06427 | [PDF](https://arxiv.org/pdf/2608.06427v1)

**作者:** Mojtaba Eslami `[一作]` `[通讯]` (University of Calgary), Mojtaba Eslami (University of Calgary)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种对抗性因果干预真伪化（ACIF）框架，利用结构因果生成器与实验者之间的对抗博弈，检验生成模型在不同干预下的分布一致性。

**💡 创新点**

创新点在于：①将对抗判别器从单纯的真假样本分类扩展为“干预索引判别器”，并证明其等价于最坏干预下的积分概率度量（IPM）；②给出零集为干预等价类、唯一识别的条件与混合策略均衡；③提出基于“对立性分歧”与“平衡分割”理论的顺序干预选择与消除算法；④提供统一的样本复杂度与边际恢复保证。

**🔧 技术方法**

使用的技术包括：结构因果模型（SCM）、干预下的IPM（如Wasserstein-1、MMD）、最小-最大博弈理论、Rademacher复杂度、混合策略极大极小值、顺序消除（基于平衡分割）和神经网络实现的可微分ACIF。

**📊 数据集**

主要使用合成数据集：线性高斯双向模型、三节点链型、四节点路径结构，所有实验均在已知真实模型下模拟干预。文中亦建议可扩展至基因扰动、流式细胞术等真实干预数据。

**📈 对比分析**

对比方法包括：随机干预选择、边缘信息增益方法、梯度导向干预、ACIF基于分歧的干预。实验结果显示，分歧驱动的ACIF平均需1.5轮即可唯一识别真模型，而随机选择平均需2.24轮，提升约50%。

**⚠️ 局限性**

局限性包括：①若可行干预不足以区分候选模型，只能得到干预等价类；②模型缺失导致ACIF返回近似最优模型；③对潜在混杂变量的假设有限；④重复基于同一判别器的干预可能导致过拟合；⑤干预外推可能产生无意义的判别信号；⑥需预先定义干预集合与成本，无法完全覆盖伦理与可行性约束。

---

## 334. Multiscale Reward Hedging from Correct Demonstrations

**arXiv ID:** 2608.06825 | [PDF](https://arxiv.org/pdf/2608.06825v1)

**作者:** Pahan Dewasurendra `[一作]` `[通讯]` (Johns Hopkins University), Pahan Dewasurendra (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种多尺度奖励-对冲算法，可在未观察奖励且答案多样的演示学习中获得无时限、无偏的总隐藏损失上界，特别是对连续奖励类实现了多项式熵下的O(d) regret；

**💡 创新点**

创新点在于通过在所有尺度上共享投票的方式，将每个尺度的二元代理结合，从而获得同时满足所有阈值的误差上界，并实现连续奖励类的首个无时限保证；

**🔧 技术方法**

主要技术包括：多尺度容差代理、共享投票（one-vote across scales）、层次积分（layer‑cake integration）以及针对一维Lipschitz曲线的多项式时间实现；

**📊 数据集**

实验数据集包括合成的二维圆形奖励演示以及公开的MovieLens 100K（943用户、1682电影）用于鲁棒审核；

**📈 对比分析**

与单阈值学习器、后验最佳阈值选择器以及传统对偶线性推荐方法比较，实验证明多尺度方法在自适应压力测试中的平均累计缺口为2.07（vs 88.92/21.32），在MovieLens审核中的平均累计隐含缺口为3.53（vs 14.82/7.74），表现优于基线；

**⚠️ 局限性**

局限性包括：要求奖励类在统一间隙度量下完全有界；有限实现可能需要指数级代理并优化非凸投票；且该方法为不适当（improper）推荐，无法保证参数恢复或合适的目标估计。

---

## 335. How Reasoning Shapes Social Bias in LLM-Generated Code?

**arXiv ID:** 2608.06829 | [PDF](https://arxiv.org/pdf/2608.06829v1)

**作者:** Weifeng Sun `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性评估推理式代码生成中的社会偏见，并提出基于推理的去偏框架。

**💡 创新点**

①首次量化推理阶段对代码偏见的影响；②提出轻量化的探针检测器识别偏见推理；③在推理阶段重写并引导生成，以更高效且不显著降低质量的方式消除代码偏见。

**🔧 技术方法**

Chain‑of‑Thought 与本地推理（LRMs）生成中间推理；静态代码分析 (FairCoder scoring) 衡量代码偏见；probe‑based 低参数 LoRA 检测器；推理重写模块（LLM 生成简化、公正化推理）；代码质量评估工具（pyright）。

**📊 数据集**

FairCoder benchmark（招聘、院校录取、医疗治疗三大场景）以及 32‑任务公开子集，用于验证模型泛化。

**📈 对比分析**

与标准 LLM、原生推理模型对比；在 BR、PE、FS、代码质量等指标上进行统计；与 prompt‑based 轻量化去偏基线（few‑shot、CoT）对照。实验表明：our 方法在 BR 上下降 83.73%（从 0.56 降到 0.09），FS 上提升至 0.85，质量保持不变；检测器准确率 86.18%，F1 87.76%，显著优于规则、TF‑IDF、FastText、CodeT5/CodeBERT/CodeGen 等基线。

**⚠️ 局限性**

①仅能捕捉显式决策偏见，隐式或代理变量导致的偏见可能被忽略；②推理偏见标注依赖 GPT‑5.1，可能引入标注误差；③实验仅涵盖 FairCoder 及其子集，未验证对更广泛任务与敏感属性的适用性；④推理重写模块依赖外部 LLM，若该 LLM 受限可能影响效果；⑤模型规模与推理预算对结果的影响尚待进一步研究。

---

## 336. Toward Reliable Context Compression for Long-Horizon Agents: An Empirical Study of Execution Instability

**arXiv ID:** 2608.06503 | [PDF](https://arxiv.org/pdf/2608.06503v1)

**作者:** Guanghui Min `[一作]` (University of Virginia), Liangjie Hong `[通讯]` (Nokia)

**通讯引用:** 127 | [OpenAlex ID](https://openalex.org/A5004471013)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了长周期LLM代理中的递归上下文压缩对行为的影响，并提出一种基于边界局部验证器的压缩提示优化框架。

**💡 创新点**

创新点是将压缩事件的评估转移到每个压缩边界，通过配对闭环回放测量阻塞/重复动作的增量负荷，并仅优化压缩提示而不改动模型参数。

**🔧 技术方法**

使用自然语言压缩提示、闭环验证器、偏好学习、冻结模型（MiniMax-M3）、OpenClaw等工具。

**📊 数据集**

主要在AppWorld工具使用基准上进行实验。

**📈 对比分析**

与FIFO、LLMLingua-2、Hermes、ACON等压缩基线对比；改进的压缩提示在准确率、Pass^2、Pass@2等指标上均优于现有压缩方法，同时峰值上下文量大幅降低，执行步骤接近无压缩版本。

**⚠️ 局限性**

局限性包括仅关注可观测的阻塞和重复动作，未能捕捉无声状态损坏；跨模型迁移测试有限，仅在单一代理/基准上验证。

---

## 337. Fixed and Adaptive Topological DeepONets: Functional Measurements on Hausdorff Locally Convex Spaces

**arXiv ID:** 2608.06428 | [PDF](https://arxiv.org/pdf/2608.06428v1)

**作者:** Khemraj Shukla `[一作]` (Brown University), George Em Karniadakis `[通讯]` (Brown University)

**通讯引用:** 108476 | [OpenAlex ID](https://openalex.org/A5009658255)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于连续线性泛函的Topological DeepONet框架，可直接在局部凸空间中编码输入函数；

**💡 创新点**

创新点在于用可学习的功能测量替代传统点传感器，并结合两阶段训练与误差分解理论；

**🔧 技术方法**

采用连续线性泛函、权重SVD、两阶段训练、soft正则化，并与FNO、DeepONet等对比；

**📊 数据集**

在四个基准上评估，包括积分算子、异质达西流、Navier–Stokes vorticity以及分布值泊松方程；

**📈 对比分析**

与传统DeepONet、Two‑Step、FNO等对比，功能测量模型在相同参数下实现5–6%误差下降，适配多分辨率，自适应模型进一步将误差压至约1%；

**⚠️ 局限性**

局限在于需要预先设定泛函字典、可学习测量仍受参数数量限制，且对高频复杂算子仍难以完全替代专用谱网络。

---

## 338. Stochastic Gradient Meets Randomized Rounding: New Algorithms for Node-Weighted Steiner Problems

**arXiv ID:** 2608.06807 | [PDF](https://arxiv.org/pdf/2608.06807v1)

**作者:** Joseph Koutsoutis `[一作]` (Rutgers University), Jiawei Yu `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种在随机顺序（random‑order）下求解点加权Steiner树和Steiner森林的O(log n)逼近算法，并在离线模式下给出了O(log k)的随机化LP舍入方案。

**💡 创新点**

创新点主要有：①将Gupta‑Kehne‑Levin的LearnOrCover框架与Berman‑Coulston的增广贪心（Augmented Greedy）相结合，弥补了单一方法在点加权情况下的不足；②设计了针对“shortcut”顶点的随机化舍入策略，使得贪心算法在点加权模型下也能保持O(log k)的性能；③构造了包含学习与覆盖两部分的KL‑divergence势能函数，用以在在线随机顺序下实现O(log n)竞争。

**🔧 技术方法**

使用技术包括：LP松弛（流/割形式）、随机化舍入（以“shortcut”顶点为目标）、LearnOrCover学习-覆盖框架、增广贪心策略、KL‑divergence势能分析、期望值与潜在函数的递推证明。

**📊 数据集**

论文未使用任何公开数据集，全部为理论分析与算法设计。

**📈 对比分析**

与现有结果比较：在随机顺序下实现O(log n)竞争，等价于离线最佳O(log n)逼近；在离线模式下给出新的O(log k)随机化LP舍入；相比之前的O(log n log k)或O(log n log² k)在线算法，显著提升了竞争因子；同时算法时间复杂度为O(k·(shortest‑path 计算)) = Õ(km)。

**⚠️ 局限性**

局限性：①仅在随机顺序模型下适用，无法直接应用于对抗性（adversarial）顺序；②奖赏收集（prize‑collecting）版本的竞争因子为O(log n + log² k)，尚未证明是否最优；③潜在函数与乘法更新的常数系数较大，实际实现可能复杂；④对图的特殊结构（如稀疏或高度连通）未进行进一步优化。

---

## 339. Mobile Interaction for Assessing Fatigue, Sleep, and Activity in Neurodegenerative and Chronic Diseases

**arXiv ID:** 2608.06380 | [PDF](https://arxiv.org/pdf/2608.06380v1)

**作者:** Julian Fierrez `[一作]` (BiometricsAI), IDEA-FAST Consortium `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

探讨利用手机交互数据作为客观指标评估神经退行性疾病和免疫介导性炎症疾病患者的疲劳、睡眠与日常活动。

**💡 创新点**

首次将基于屏幕时间和应用使用的手工特征与患者自评量表进行关联分析，检验其与传统PRO的可比性。

**🔧 技术方法**

采用手机记录日志、手工特征提取、重复测量相关（RM-correlation）统计分析。

**📊 数据集**

IDEA-FAST Feasibility Study收集的137名患者（6种疾病+健康对照）的移动交互日志与每日PRO问卷。

**📈 对比分析**

与传统PRO（FACIT-F、MOS-SS）进行比较；整体上相关性低，但在特定病种或年龄组中出现弱相关（|r_m|<0.5）。

**⚠️ 局限性**

受限于采样频率低、应用类别稀疏、数据覆盖不完整、样本量不足导致相关性弱，且仅使用屏幕时间和应用分类，缺乏更细粒度的触控或传感器特征。

---

## 340. A Low-Latency ASIC Architecture for Real-Time Line Segment Detection

**arXiv ID:** 2608.06439 | [PDF](https://arxiv.org/pdf/2608.06439v1)

**作者:** Amir Hossein Jalilvand `[一作]` (Iran University of Science and Technology), M. Hassan Najafi `[通讯]` (Case Western Reserve University)

**通讯引用:** 1431 | [OpenAlex ID](https://openalex.org/A5012903661)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0540dec-d77f-42db-94ae-d039248f6393` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种低延迟 ASIC 架构，用于实时线段检测，能够在 45nm CMOS 45nm CMOS 45nm 芯片上以 100 MHz 运行，实现 VGA 325 FPS、FullHD 48 FPS，功耗仅 25.54 mW，面积 0.412 mm²。

**💡 创新点**

创新点在于结合 step‑length 算法与五大 ASIC 专用特性：寄存器线缓冲与数据复用、无乘法 MCM 过滤、8 类角度量化、CAM‑类关联内存实现单周期匹配，以及滑动窗口去重；整个系统完全流水化并拥有确定性延迟。

**🔧 技术方法**

采用寄存器线缓冲、MCM 乘法替代、8 类角度量化、CAM‑类关联匹配、滑动窗口去重等硬件技术，整体实现基于 step‑length 算法，并在 45 nm CMOS（FreePDK45）标准库中实现。

**📊 数据集**

功能验证使用 YorkUrban 数据集；实验对比亦涉及标准图像数据以评估检测质量。

**📈 对比分析**

通过与基于 Hough Transform 的 ASIC、FPGA 以及其他 ASIC 设计进行对比，实验显示该实现功耗最低（相较 90 nm ASIC 降低 49%），帧率最高（100 MHz 时 325 FPS，125 MHz 时 406 FPS），且相较于现有 ASIC 实现帧率提升 1.6 倍。

**⚠️ 局限性**

局限性包括：最大支持 1920×1080 分辨率；链条容量固定为 64，去重窗口仅为 5 条；未针对更高分辨率或神经网络集成进行优化；在极端噪声或低对比度场景下鲁棒性可能受限。

---

## 341. Can MLLMs Decode the Creative Leap? Introducing C4 for Cross-Concept Understanding

**arXiv ID:** 2608.06501 | [PDF](https://arxiv.org/pdf/2608.06501v1)

**作者:** Ming Wang `[一作]` (Northeastern University), Yifei Zhang `[通讯]` (Northeastern University)

**通讯引用:** 15315 | [OpenAlex ID](https://openalex.org/A5100458295)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于“成语”桥接网络的跨概念编码-解码评估框架，并构造了可批量生成、可度量难度的创意任务集；

**💡 创新点**

创新点在于将跨概念关联视为可解释的桥接路径，并将其应用于可量化的成语创意检索任务，解决了创意理解评价难度大且缺乏统一标注的问题；

**🔧 技术方法**

核心技术包括手工构建和审核的跨概念桥接网络、基于桥路径的图像生成、以及多样化任务设置（开放恢复、候选识别、提示辅助、解释生成）来评估多模态大模型；

**📊 数据集**

使用的数据集是“Cross-Concept Chengyu Benchmark (C3B)”，包含184条人工合成项目和37条网络收集的真实成语图像，涉及84个成语目标，共221项；

**📈 对比分析**

评估方法为在五种任务设定下对10个多模态大模型进行严格对比，结果显示闭源模型最高主分约为50.7%，开放源模型平均仅18.1%，说明当前模型在跨概念解码上仍存在明显瓶颈；

**⚠️ 局限性**

限制在于数据集仍受人工构建的成语和桥接路径的覆盖范围限制，且实验仅涉及十个模型，难以全面反映所有多模态模型的能力，未来需扩大样本与模型多样性。

---

## 342. RECAST: A Region-Scoped Adaptive Index for Exact Similarity Search

**arXiv ID:** 2608.06962 | [PDF](https://arxiv.org/pdf/2608.06962v1)

**作者:** Yining Liu `[一作]` (Beijing Institute of Technology), Rui Mao `[通讯]` (Shenzhen University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向相似度搜索的区域自适应索引RECAST，动态学习并重用查询过程中产生的距离信息，支持查询集中区的递归划分与残差子区域；

**💡 创新点**

创新点在于：①将查询所需计算的距离双重利用为可重用的“付费距离”与结构适配信号；②采用区域级多指针表并根据查询成本趋势驱动指针保留、淘汰与划分决策；③引入影子验证机制，避免过早/过晚划分；④残差子区域保持未计算距离对象的可学习性，防止旧信息污染新区域；

**🔧 技术方法**

核心技术包括基于三角不等式的多指针裁剪、成本信号（检查计数、误判率、裁剪率）驱动的自适应指针管理与划分、影子验证的局部划分判定、递归区域划分树结构；

**📊 数据集**

在五个真实数据集上验证：glove100（100维词向量）、sift1m/10m（128维SIFT特征）、colors112d（112维颜色直方图）、nasa20d（20维航空数据）；

**📈 对比分析**

与自适应基线AV‑tree和预构建索引LAESA、GNAT对比，RECAST在所有20个数据集‑工作负载组合中均实现了查询距离计算量比AV‑tree低35–64%，查询时间降低10–46%，且在大多数场景下超过预构建索引的累计成本；

**⚠️ 局限性**

局限在于：当距离计算本身成本较低（如低维空间）或数据分布高度聚集（高维空间内距离集中）时，多指针裁剪收益有限，可能导致查询时间反而增大；同时残差子区域的递归深度和内存占用在极端动态工作负载下仍需进一步评估。

---

## 343. Do AI Personas Grow? Analyzing and Benchmarking Personality Evolution in LLM Agents After Life Events

**arXiv ID:** 2608.06485 | [PDF](https://arxiv.org/pdf/2608.06485v1)

**作者:** Ming Wang `[一作]` (Northeastern University), Ee-Peng Lim `[通讯]` (Singapore Management University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在11个主要人生事件前后对100个按性别、文化和性格类型构造的PC‑Agent人格进行BFI‑44测评，研究其人格演化特征并构建了可复用的评估基准。

**💡 创新点**

创新点在于提出四轴诊断框架（存在性、方向与幅度、人口学形状、个体形状），并基于事件驱动的方向先验设计了B​FI‑Adapt综合评价指标，首次系统量化LLM人格变化与人类长期研究结果的契合度。

**🔧 技术方法**

使用的方法包括：事件触发式对话提示、配对BFI‑44测评、项级可靠性（κ）与方向一致性比率（DCR）、匹配率统计、Bootstrap置信区间、以及独立词义改写、情境决策和短期保持性验证。

**📊 数据集**

数据集为100个按2性别×5文化×10人格原型构造的合成人格，每个模型在11个生活事件（共44项）下完成两轮BFI‑44，覆盖11个API模型和3个开源模型，总计48,400条项目评分。

**📈 对比分析**

通过B​FI‑Adapt对14个模型进行排名，Gemini‑3‑flash、GLM‑4.6和Qwen3‑235B在方向一致性和匹配率方面居首；然而幅度校准偏弱，个体差异压缩，整体表现仍低于人类长期效应范围。

**⚠️ 局限性**

局限性包括：仅测量即时事件前后变化，缺乏多年纵向跟踪；使用粗略的方向先验和效应量范围；人格差异压缩导致模型缺乏细腻个体化；实验仅涉及合成人格，未验证真实人类交互中的表现。

---

## 344. TransSLR: A Lightweight Transformer for Sign Language Recognition

**arXiv ID:** 2608.06407 | [PDF](https://arxiv.org/pdf/2608.06407v1)

**作者:** Lucia Yen Wanchi `[一作]` (Carnegie Mellon University Africa), Moise Busogi `[通讯]` (Carnegie Mellon University Africa)

**通讯引用:** 187 | [OpenAlex ID](https://openalex.org/A5045181032)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对低资源的中央非洲手语(CASL)提出并实现了一种轻量级时序Transformer编码器TransSLR，用于孤立手语识别。

**💡 创新点**

创新点在于仅使用几何关节位置信息、完全去除视觉纹理干扰，并且从零开始训练轻量级编码器，显著提升了跨签者泛化能力。

**🔧 技术方法**

采用了MediaPipe Holistic提取的骨架姿态、线性投影、正弦位置编码、4层Transformer Encoder与全局平均池化分类头的组合。

**📊 数据集**

主要实验数据集为CASL-W60，包含60个手语类别、5889条样本、19名签者，采用签者独立的训练/验证/测试拆分。

**📈 对比分析**

与多种RGB、姿态和多模态基线（包括VideoMAE、Bi-GRU+Attention、Transformer等）对比，TransSLR在Top‑1准确率上达80.39%，提升约10.46个百分点，显著优于先前最佳69.93%。

**⚠️ 局限性**

局限性包括仅使用姿态信息忽略了面部表情等非手势线索，导致部分语义相似手语的混淆；此外，数据集规模有限，难以验证模型在更大词汇和连续手语上的表现。

---

## 345. PAST: Prompt-Adaptive Sampling Termination for Efficient Diffusion Model

**arXiv ID:** 2608.06794 | [PDF](https://arxiv.org/pdf/2608.06794v1)

**作者:** Renye Yan `[一作]` (Peking University), Yimao Cai `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Prompt-Adaptive Sampling Termination插件，实现RL微调扩散模型时自适应终止采样并补偿稀疏奖励；

**💡 创新点**

核心创新是结合内在奖励和基于去噪进度与语义对齐的双因素自适应终止机制，动态平衡探索与收敛；

**🔧 技术方法**

技术方案包括将扩散过程建模为MDP、使用去噪感知模型DAM、注意力相似度度量、KL约束以及自适应奖励权重；

**📊 数据集**

实验采用HPSv2、Pick-a-Pic、Simple Animal等复杂提示数据集，评估AES、PS、IR、CLIP、IS等指标；

**📈 对比分析**

与DDPO、DPOK、D3PO、TDPO等四个SOTA RL基线以及多种扩散骨干（SDv15、SDv14、SDv21、XL、SD3.5）对比，实验显示在相同奖励目标下计算成本可降至66.7%，奖励提升29.5%；

**⚠️ 局限性**

限制在于仍需外部LLM进行提示分解，且在极难提示或极大模型规模时提示分解与注意力评估可能产生误差，未验证在更大规模或非文本条件下的通用性。

---

## 346. Autonomy-of-Heads: Data-Free Sparse Attention from Frozen Query-Key Geometry

**arXiv ID:** 2608.06849 | [PDF](https://arxiv.org/pdf/2608.06849v1)

**作者:** Yehan Yang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Dianhai Yu `[通讯]` (Baidu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Autonomy-of-Heads (AoH)的方法，用于从冻结的查询-键几何中识别检索头和流式头，旨在提高长上下文大语言模型的推理效率。

**💡 创新点**

创新点在于AoH是一种无数据和无训练的方法，通过分析核注意力矩阵的有效秩来替代校准数据和学习门控，从而简化了头部选择过程。

**🔧 技术方法**

使用了有效秩分类器和高效的d_head维度计算方法，避免了构建完整的d_model×d_model矩阵。

**📊 数据集**

在LongBench数据集上进行了广泛的实验，该数据集涵盖了21个任务，涉及六个类别。

**📈 对比分析**

与现有的稀疏注意力和KV压缩方法相比，AoH在50%稀疏性下平均保留了96.5%的全注意力性能，同时在预填充和解码延迟上分别减少了41.4%和66.0%，KV缓存内存减少了50.0%。

**⚠️ 局限性**

限制在于该方法主要针对解码器模型，未在编码-解码模型、多模态LLM或具有重度修改注意力机制的模型上进行广泛测试。此外，稀疏预算是手动选择的，未来可以考虑自适应预算选择。

---

## 347. POKEx: Performance analysis of POKE-key exchange and SIDH-variants

**arXiv ID:** 2608.06826 | [PDF](https://arxiv.org/pdf/2608.06826v1)

**作者:** Hyeonhak Kim `[一作]` (Korea University), Suhri Kim `[通讯]` (Sungshin Women’s University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于POKÉ的密钥封装机制POKEx，并在相同安全等级下与M‑SIDH、terSIDH、CSIDH等同类等距量化比较；

**💡 创新点**

首次完整描述POKÉ转换为KEM的FO变换实现，给出完整算法与实现细节；

**🔧 技术方法**

结合等变异isogeny、两维表示、随机化等值、θ‑endomorphism以及Fujisaki‑Okamoto变换；

**📊 数据集**

使用公开参数的NIST安全等级1的Prime字段（如p=2^129·3^164·5^18‑1）和对应的基点；

**📈 对比分析**

通过在同一台Intel Core i9‑10980XE上采用‑O3编译，测定公共/私钥尺寸和完整密钥交换时间；POKEx在同等级下约快21.21×terSIDH、64.97×CSIDH；

**⚠️ 局限性**

相较于ML‑KEM等后量子方案仍显慢，且实现依赖高阶isogeny运算，尚需进一步优化；

---

## 348. PRISM: Principled Reference Identification for Schrodinger Bridge Model

**arXiv ID:** 2608.06893 | [PDF](https://arxiv.org/pdf/2608.06893v1)

**作者:** Forouzan Fallah `[一作]` (Arizona State University), Yezhou Yang `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对Schrödinger桥模型（SBC）的参考过程（reference process）设计，提出理论化的方法PRISM，求解在有限求解步骤下如何选择噪声色谱和时间调度，以最小化后验估计误差。

**💡 创新点**

创新点包括：
- 对可正交对角化的时变协方差参考过程给出完全可解析的桥式分离条件；
- 证明“不可见性原则”，在无限步或精确漂移下参考过程不再影响终端分布；
- 推导有限步目标的闭式表达，证明最优噪声谱正比于传感器破坏信息谱P_k；
- 引入z-坐标，将噪声色谱与调度完全可互换；
- 在有限资源（步数或模型误差）下给出显式最优比例常数x^*(T)≈(2 ln T)^(-1/2)；
- 实验验证理论预测并揭示真实图像对非高斯模式统计的敏感性。

**🔧 技术方法**

采用的技术手段包括：
- 高斯线性降噪模型和Wiener滤波理论；
- Schrödinger桥的精确桥式公式与逆向马尔可夫核；
- 量化的KL散度闭式推导与优化；
- z-变换下的二次分式凸优化；
- 通过矩阵对角化实现多模独立桥式；
- 精确数值递推与高精度数值验证；
- 训练多模式独立MLP和U-Net实现桥模型；
- 统计度量（PSNR、SSIM、LPIPS、FID）与离散调度对比。

**📊 数据集**

使用的数据集和降噪场景包括：
- 64或256维高斯模拟信号（S_k∝k^(-2)，均匀或幂律调度）；
- FFHQ 64×64图像，已知高斯模糊σ_blur=2.0 px及白噪声σ_n=0.05；
- CelebA 64×64图像，使用相同降噪参数；
- 对比实验还包括无下采样的单频通道测试。

**📈 对比分析**

比较方法与性能：
- 在理论可解析的高斯模型中，所有参考过程的终端KL对比已知贝叶斯下限，实验误差低于0.2%；
- 在FFHQ和CelebA实验中，白噪声参考在较少步骤（NFE≤50）下显著优于“匹配”参考（white‑matched FID差距从0.58提升至≈1.2），体现“白噪声优先”趋势；
- 对于不同色谱（白、匹配、反匹配、先验色谱）与调度网格（均匀、z最优）进行多种指标评估，均符合理论排序；
- 但在真实图像实验中，非高斯统计导致理论预测逆转，white优于matched，表明模型对非高斯统计更敏感。

**⚠️ 局限性**

Limitations：
- 理论推导基于高斯线性桥模型，实际图像中存在非高斯模式交互，导致理论预测失效；
- 参考色谱与调度的唯一最优解对非均匀网格可能非唯一，导致多模桥分配复杂；
- 对模型漂移误差和有限数据量的更深入分析尚未完全实现（如共享网络的交叉模式耦合仍未解决）；
- 在极低步数或极高模糊度下实验性能波动大，需进一步优化训练策略。

---

## 349. Geometry-Aware Camera Localization for Bronchoscopy

**arXiv ID:** 2608.07116 | [PDF](https://arxiv.org/pdf/2608.07116v1)

**作者:** Lumin Chen `[一作]` (Chinese Academy of Sciences), Dong Yi `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种统一的几何感知支气管镜定位框架 GABL，利用预手术结构先验与实时视频实现 6-DoF 位置估计。

**💡 创新点**

通过三尺度几何约束（结构、运动、外观）结合图引导粗细定位、Transformer 时序跟踪和 RGB-深度匹配，实现了毫厘级精度与实时推理的统一。

**🔧 技术方法**

采用图神经网络构建气道图、Causal Transformer 进行时序建模、ResNet+Transformer 结合深度监督，以及 RGB-深度跨模匹配损失。

**📊 数据集**

在临床标注的 BREATH bronchoscopy 数据集（66例，148,926帧）上训练与评测。

**📈 对比分析**

与 MonoGS、EndoGSLAM、Endo-FASt3r、BREATH-VL 等方法比较，ATE_trans 7.01 mm、ATE_rot 29.56°、SR‑5 61.04%，速度 33.6 FPS，明显优于现有方法。

**⚠️ 局限性**

受限于数据量不足、对气道变形建模有限，以及对极低纹理或多重光照下的鲁棒性仍需提升。

---

## 350. MemWM: Memory-Augmented Text-Based World Model

**arXiv ID:** 2608.07107 | [PDF](https://arxiv.org/pdf/2608.07107v1)

**作者:** Yujun Wang `[一作]` (Lmu Munich), Yunpu Ma `[通讯]` (Lmu Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种记忆增强的文本世界模型，通过检索世界记忆来改进下一状态预测并提升规划效果。

**💡 创新点**

引入结构化状态保真度（SSF）评价指标，并设计可检索的世界记忆库（包含转移规则、状态缓存和难以预测的事实），实现对文本世界模型的系统性错误校正。

**🔧 技术方法**

使用大语言模型（如Llama、Qwen系列）作为世界模型核心，结合检索增强生成（RAG）、结构化事实抽取与权重化相似度计算，并在规划时结合政策侧世界技能（任务级技能与纠正指导）。

**📊 数据集**

在三大文本交互基准上验证：ALFWorld、ScienceWorld 与 WebShop。

**📈 对比分析**

与标准SFT、无记忆RL以及多种下游规划算法（CoT、ReAct、RAP、ITP）对比，记忆增强模型在SSF上提升多达206.3%，在任务成功率上相较SFT版提升最高65.4%；且在不同行动预算与记忆缺失情况下仍保持显著优势。

**⚠️ 局限性**

局限包括：依赖手工或小型LLM抽取的结构化事实，检索效率与匹配精度可能随领域差异而受限；记忆库的规模和更新机制对性能影响大；模型在极大搜索空间或长序列规划中仍需进一步优化。

---

## 351. Synthetic LiDAR Data Generation and Deterministic Downsampling for Point Cloud Classification on the Edge

**arXiv ID:** 2608.07106 | [PDF](https://arxiv.org/pdf/2608.07106v1)

**作者:** Niclas Meyer `[一作]` (Chemnitz University of Technology), Stefan Reitmann `[通讯]` (Chemnitz University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文设计并评估了一套端到端的工作流：先利用Blender插件BLAINDER生成逼真的带噪声LiDAR点云数据，再将可训练的Critical Point Layer（CPL）嵌入PointNet实现可确定性的点云下采样，最终在Raspberry Pi 5 CPU上实现低延迟的3D点云分类；

**💡 创新点**

创新点在于（1）通过BLAINDER合成带噪声的LiDAR数据，弥补CAD模型与真实传感器之间的域差距；（2）将CPL作为独立的前置下采样模块，结合可学习的MLP实现高效、确定且对分类有利的压缩；（3）在资源受限的边缘设备上展示了近实时推理（≈21 ms/帧）与高分类准确率（≈88 %）的可行性；

**🔧 技术方法**

主要技术包括Blender LiDAR模拟插件BLAINDER、PyTorch实现的PointNet网络、CPL下采样层、随机采样与Farthest Point Sampling（FPS）比较、Chamfer距离与LIME‑3D解释、以及在Raspberry Pi 5上的CPU推理；

**📊 数据集**

使用的数据集为ModelNet40（CAD模型），并通过BLAINDER生成的旋转/静态、清晰/噪声两类合成LiDAR子集；

**📈 对比分析**

在与原始1024点PointNet、随机采样和FPS下采样的对比实验中，CPL将点数压缩至128点时实现≈21 ms推理（≈47 FPS），远快于FPS的23 ms；分类准确率保持≈88 %，与使用随机采样训练的基线相近；交叉域评估显示，噪声训练模型对清晰数据具有一定泛化能力；

**⚠️ 局限性**

局限性包括：CPL在极低点数时可能忽略细节，且其提取的关键点与人类感知（LIME‑3D）不完全一致；单视角训练模型在多视角数据上的泛化差；目前仅在CPU上实现，未利用FPGA/NPU等更高效硬件；以及合成数据缺少强度信息等物理属性。

---

## 352. UncertaintyVis: Preserving Linguistic Uncertainty in Automated Text-to-Chart Generation

**arXiv ID:** 2608.07093 | [PDF](https://arxiv.org/pdf/2608.07093v1)

**作者:** Songheng Zhang `[一作]` (Singapore Management University), Anthony Tang `[通讯]` (Singapore Management University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一套自动化文本转图表系统，能够识别并保留数据文档中的语言不确定性，并生成对应的不确定性可视化图表。

**💡 创新点**

创新点在于构建四分类不确定性词汇分类法，并将每类映射到特定的图表视觉编码；同时实现了大语言模型驱动的文本解析与不确定性注解，保持语义完整。

**🔧 技术方法**

采用大语言模型（如GPT‑4）进行文本解析、结构化提取与不确定性标注，随后使用D3.js等可视化库根据元数据生成图表。

**📊 数据集**

使用12篇跨8个领域（医学、政治、体育等）的数据丰富文档共211条不确定性表达作为语料；评估数据来自实验生成的图表。

**📈 对比分析**

通过两部分用户研究（12名参与者）对编码匹配准确率（文本→图表76%，图表→文本85%）及认知负荷（NASA‑TLX）进行比较，显示不确定性可视化在认知负荷上有中等至大幅度下降。

**⚠️ 局限性**

局限包括对折线图不确定性编码效果欠佳、依赖人工标注的前处理、以及小样本量与单一界面验证。

---

## 353. Beyond Isolation: Unlocking Reinforcement Learning Component Synergy for Sample-Efficient Continuous Control

**arXiv ID:** 2608.07086 | [PDF](https://arxiv.org/pdf/2608.07086v1)

**作者:** Qi Zhao `[一作]` (Tsinghua University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个用于强化学习的协同样本效率框架，整合了模型基础表示、优化稳定性和经验回放三大技术。

**💡 创新点**

创新点在于：①揭示了不同样本效率增强技术在集成时可能产生互相干扰；②总结了三条协调原则；③设计了稳定模型表示（⋆）和逐步优先回放（U2P）实现高效协同。

**🔧 技术方法**

使用了SimBa网络架构、MR.Q 风格的模型基础表示、ReLo/Uniform-to-Prioritized Replay、SAC+ 算法以及自定义的残差式信息旁路。

**📊 数据集**

使用了 DeepMind Control Suite、HumanoidBench、Myosuite、ManiSkill2 这四个连续控制基准，共 18 个任务（9 运动学，9 操作）。

**📈 对比分析**

通过与 vanilla SAC、单组件增强版本以及 Naive Stack 对比，采用 IQM、95% 置信区间和性能曲线评估；结果显示框架在样本效率和鲁棒性上明显优于对照组。

**⚠️ 局限性**

局限性包括：各组件效果高度依赖任务，缺乏系统的任务特征分类；协调机制主要经验式；仅考察了 SAC；未涉及探索策略或真实世界测试。

---

## 354. Transformers Struggle to Use Their Emergent World Models: Revisiting the Tower of Hanoi, and the Illusion of Thinking

**arXiv ID:** 2608.07077 | [PDF](https://arxiv.org/pdf/2608.07077v1)

**作者:** Devin Pereira `[一作]` (University of Amsterdam), Willem Zuidema `[通讯]` (University of Amsterdam)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了 Tower of Hanoi 的 flat-to-flat 变体，分析了从小型 Transformer 到大型推理模型（Qwen3.6‑27B、DeepSeek‑R1‑Distill‑Qwen‑32B）在解决该任务时如何构建、使用以及衰退世界模型，并通过激活补丁和激活驱动恢复模型性能。

**💡 创新点**

创新点在于：①首次将“涌现的世界模型”概念从简单游戏迁移到需要长链思考的规划任务；②在大模型中定位到生成过程中世界模型的衰退是失败根源，并证明通过激活驱动可恢复性能；③通过统一到分解的表示转变揭示小型与大型模型的相似机制。

**🔧 技术方法**

主要技术包括：自监督训练的 GPT‑2‑style 小 Transformer；线性探测（distance‑matching、per‑disk 分类）来检索 Sierpinski 结构；激活补丁（patching）与激活驱动（steering）实现因果干预；以及符号状态追踪用于在推理期间保持真实状态。

**📊 数据集**

数据集：利用符号生成的 81 个四盘子 Tower of Hanoi 的可达状态对（s_I, s_G）及其解序列，构建训练/验证集；在大模型实验中进一步测试 3–7 盘子（共 81+3+4+5+33+33+33 实例），并设置 32k token 输出预算。

**📈 对比分析**

比较方法：小 Transformer 在 4 盘子任务上达到 93.2% 序列级精度；大模型在 tower‑to‑tower 任务几乎完美，但在 flat‑to‑flat 任务仅 51% 正确；激活驱动后 Qwen3.6‑27B 的成功率从 41% 提升至 73%，显示显著提升；DeepSeek‑R1‑Distill‑Qwen‑32B 的提升有限，说明其失败主要由输出格式错误导致。

**⚠️ 局限性**

局限性：仅针对单一 4 盘子（及更少盘子）任务，未验证在更复杂或非几何状态空间的泛化；实验仅覆盖 Qwen 系列模型；激活驱动依赖外部符号状态跟踪，难以扩展到无显式状态可得的任务；对不同层次、不同模型的驱动效果差异尚未完全解析。

---

## 355. M2-SMap: Memory-Efficient Semantic Mapping with Hierarchical Multi-Model Representation

**arXiv ID:** 2608.07074 | [PDF](https://arxiv.org/pdf/2608.07074v1)

**作者:** QiYing Deng `[一作]` (University of Electronic Science and Technology of China), Wei Dong `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了M2‑SMap，一种基于层级多模型（平面、超四面体、GMM）和语义实例指导的资源受限RGB‑D地图框架；

**💡 创新点**

创新点在于：①在高斯分解前通过3D‑2D投影与实例分割进行语义预标注，防止跨对象融合；②使用语义约束的高斯融合与平面分离；③通过等价姿态对齐消除超四面体旋转跳变；④整体实现实时且内存高效；

**🔧 技术方法**

技术包括：层级高斯混合模型（IH‑GMM）分解、实例掩码匹配、语义约束的高斯融合、平面分离、超四面体拟合与时间关联、GMM残差建模以及对等价姿态的对齐与平滑；

**📊 数据集**

实验数据集为TUM RGB‑D的三条序列：fr3/walking_rpy、fr1/xyz、fr1/teddy；

**📈 对比分析**

与四种几何驱动基线（MMC、MMQ、MMP、MMOG）对比，M2‑SMap在G2R、R2G误差上与基线相当或略优，原始原语数量最低，平均原语减少18.7%，实时率在29–97 Hz之间，且完全消除了跨对象粘连；

**⚠️ 局限性**

局限性在于：验证仅限于静态小型场景，动态或大规模环境下的鲁棒性待评估；依赖实例分割精度；超四面体拟合在复杂形状上可能失败，需退回GMM；未与导航或操作任务深度集成。

---

## 356. DocMemo: Dynamic Evidence Discovery via Probabilistic Memory-Guided Retrieval for Multi-Modal Document Understanding

**arXiv ID:** 2608.07067 | [PDF](https://arxiv.org/pdf/2608.07067v1)

**作者:** Hanshu Yao `[一作]` (Harbin Institute of Technology), Jinpeng Wang `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

DocMemo提出了一个基于内存驱动的多轮长文档视觉问答框架，能够动态检索、更新和整合跨页证据；

**💡 创新点**

其创新点在于构建三层文档内存（文档结构内存、页面信念内存、问题情景内存）以及采用贝叶斯页面信念更新与Thompson采样实现跨轮动态证据探索；

**🔧 技术方法**

技术包括贝叶斯页面信念更新、Thompson采样、空间邻域传播、适应性粒度证据访问、ColQwen2.5视觉检索以及Qwen3.5-VL-9B语言模型；

**📊 数据集**

使用了MMLongBench-Doc、LongDocURL和PaperTab三大长文档视觉问答基准；

**📈 对比分析**

在与专有模型、开源多模态LLM以及基于代理的系统在相同评测协议下对比中，DocMemo在三大基准上的准确率分别为71.3%、81.1%和80.4%，平均提升约10-15个百分点；

**⚠️ 局限性**

局限性包括对图表密集页的细粒度访问仍受限于局部视觉模型，对极长文档的实时推理成本较高，以及未充分评估多模态信息的跨模态一致性。

---

## 357. BONSAI: Evolvability-Guided Tree Search over Skills

**arXiv ID:** 2608.07056 | [PDF](https://arxiv.org/pdf/2608.07056v1)

**作者:** Yash Priya Shastri `[一作]` (IBM Research), Sachin Joshi `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对冻结语言模型的技能优化，提出了基于进化潜力的搜索框架BONSAI，通过构建技能树并使用上限置信度选择来系统性改进文本指令。

**💡 创新点**

创新点在于用进化潜力（后代平均得分）来衡量技能局部可变性，并将其作为搜索的利用项，从而在不额外评估成本的前提下引导搜索走向可持续改进的区域。

**🔧 技术方法**

采用蒙特卡洛树搜索、上限置信度（UCB）策略、基于反射的单次变异编辑、可选的交叉树枝嫁接（GRAFT）操作以及接受阈值决策。

**📊 数据集**

在SpreadsheetBench、SearchQA和LiveMathematicianBench这三个公开基准上进行实验。

**📈 对比分析**

与无技能、种子技能、GEPA和SkillOpt等预算匹配基线对比，BONSAI在所有三个任务上均超越基线，SpreadsheetBench提升至23.21%（+5.71点），SearchQA 79.00%（+6.50点），LiveMathematicianBench 64.91%（+47.17点）。

**⚠️ 局限性**

实验仅在单一运行、单一冻结模型和优化器组合下完成；接受阈值基于小批量任务，可能影响搜索可靠性；未评估附加的记忆补丁等可选组件。

---

## 358. Teacher Retains Full Tokens, Student Merges Efficiently: TM20K for E-Commerce Sequence Modeling in Ad Recommendation

**arXiv ID:** 2608.07055 | [PDF](https://arxiv.org/pdf/2608.07055v1)

**作者:** Xinchun Li `[一作]` (ByteDance), Yaocheng Tan `[通讯]` (ByteDance)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在大规模广告推荐系统中通过全注意力教师模型与多种规则化的 token 合并方式对 5K→20K 超长行为序列进行建模，实现了性能与效率的平衡。

**💡 创新点**

创新点在于：①采用两阶段知识蒸馏，教师仅训练一次且保留完整序列；②设计三种基于注意力分布洞察的 token 合并策略（LITM、PATM、LPTM）；③将全注意力与多级 token 合并相结合，既提升了预测效果又保持了低延迟。

**🔧 技术方法**

使用技术包括：Transformer 全注意力（FA）、RankMixer 交互层、FlashAttention、M-Falcon Serving、QK 归一化、两阶段蒸馏、CPU+GPU 并行的 token 合并实现。

**📊 数据集**

使用 ByteDance 真实广告推荐场景中的亿级 CVR 数据集，包含 5K 与 20K 长度的电商行为序列。

**📈 对比分析**

与 STCA、LONGER、MTFM、HyFormer 等方法对比：TM20K 在 20K 长度下实现 AUC 提升 0.26%（+85% 归属于蒸馏），在线 A/B 测试显示 ADSS +1.036%，Serving Latency 仅 +5.6%，训练吞吐量与 GPU 内存基本与 5K 基线相当。

**⚠️ 局限性**

局限性在于 token 合并策略为手工规则化，跨域迁移可能需要额外调参；未结合稀疏注意力机制，后续需开发专用 GPU 操作以兼容现有优化。

---

## 359. C2Dex: Contact-Consistent Reconstruction and Retargeting for Dexterous Manipulation from Monocular Video

**arXiv ID:** 2608.07045 | [PDF](https://arxiv.org/pdf/2608.07045v1)

**作者:** Jie Ren `[一作]` (Nanjing University), Xun Cao `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

将单目人类视频转换为可执行的多指机器人操作轨迹。

**💡 创新点**

通过聚合多帧的对象侧接触实现稳定接触表示，利用该表示实现接触一致的 HOI 重建和交互保持的重定位。

**🔧 技术方法**

结合 MANO、Dyn-HaMR、SAM 3D、ProxyPose 等骨架估计与物体重建；使用帧间接触稳定化、Laplacian 交互优化、残差强化学习等技术。

**📊 数据集**

在 DexYCB、TACO 以及 24 条现实世界演示数据集上进行评估。

**📈 对比分析**

与 Do As I Do、DexImit 等现有视频到机器人管线对比，C2Dex 在轨迹成功率、接触稳定性、重定位精度上分别提升 57.78%/26.67% 以及显著降低接触误差和穿透深度。

**⚠️ 局限性**

对遮挡、极端视角和物体复杂几何的感知鲁棒性不足，依赖精确物体姿态估计，且不支持频繁接触切换的复杂手内操作。

---

## 360. Beyond Text Matching: Towards Reference-Free Evaluation for Human-Oriented Binary Reverse Engineering

**arXiv ID:** 2608.07038 | [PDF](https://arxiv.org/pdf/2608.07038v1)

**作者:** Xiuwei Shang `[一作]` (University of Science and Technology of China), David Lo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统评估了将大型语言模型用作评判者（LLM-as-a-Judge）在人工导向二进制逆向工程（HOBRE）任务中的表现，并提出了首个无参考评估基准和轻量级自适应路由机制。

**💡 创新点**

创新点在于：①构建首个专家标注、无参考的HOBRE评估基准；②系统验证LLM-as-a-Judge相较传统指标的显著优势；③设计基于UniXcoder的自适应路由器，实现对评判配置的动态选择。

**🔧 技术方法**

采用多种大型语言模型（GPT‑4o、Claude‑3.5‑Sonnet、Gemini‑2.5‑Flash、DeepSeek‑V3.2、Qwen3‑Coder等）作为评判者，利用多维Likert评分、Kendall τ/Spearman相关系数进行评估，并用基于UniXcoder的网络进行路由学习。

**📊 数据集**

使用51个GNU项目在6种架构、4种优化级别下生成346,596个函数级样本，随后随机抽取1,233个样本进行三名逆向工程专家的三阶段标注，形成专家标注基准。

**📈 对比分析**

与12种传统匹配/嵌入式评估指标对比，LLM‑as‑a‑Judge平均相关率为63.2%，显著高于传统35.04%；通过自适应路由，相关率进一步提升4.5%‑24.7%，API成本降至0.06×‑0.84×。

**⚠️ 局限性**

限制包括：评估仍依赖有限人工标注，存在主观性；LLM评判在极难样本上可能受流利度误导；对模型自身输出的偏差评估仍不完全；且实验覆盖的任务与模型种类仍有限。

---

## 361. Accounting Graph Transformer for Short-History Multi-KPI Forecasting in Small Businesses

**arXiv ID:** 2608.07037 | [PDF](https://arxiv.org/pdf/2608.07037v1)

**作者:** Shrutendra Harsola `[一作]` (Foresight-AI, Intuit), Vignesh Subrahmaniam `[通讯]` (Foresight-AI, Intuit)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种可在短期小企业账簿数据上一次性生成 13 个 KPI 的 12 个月预测的联合时序模型——Accounting Graph Transformer (AGT)。

**💡 创新点**

创新点包括：① 采用固定的会计关系图作为先验，显式约束跨系列的信息流；② 在每个 KPI 的预测头加入最近三期观测的“recency”路径；③ 通过子账户排名与 catch‑all 机制，使 71 条账目序列可在异构科目表上统一编码；④ 在公司无重叠的面板上共享参数，支持在未见公司和不同时间点直接迁移。

**🔧 技术方法**

技术手段：多头图注意力（Relational Attention），层归一化与残差网络，目标特定的加权池化，recency 门控融合，基于 RevIN 的实例归一化，Huber 损失以及 AdamW + cosine 学习率衰减。

**📊 数据集**

数据集：来自一商业云会计平台的匿名月度总账聚合。训练/验证/测试共 5,082/1,086/1,060 家企业，平均每家 11–12 个预测起点；每个起点拥有 12–24 个月历史；另外 7,094 家企业提供 1 个后期起点用于迁移测试。

**📈 对比分析**

比较方法：在相同的历史窗口、mask、目标转换和评分规则下，对 8 种基线（基础统计、树模型 LightGBM、通用神经网络 SOFTS/TimeMixer、预训练时序模型 Chronos‑2/TimesFM/Moirai‑2）以及无模型 Naïve。AGT 在 3 个随机种子下的平均 KPI‑宏 MAE 为 0.6990，显著低于 LightGBM 0.7378（-5.3%），TimeMixer 0.7523（-7.1%），SOFTS 0.7560（-7.5%）；在后期迁移集上，AGT 0.7548 也保持领先。模型在所有 13 KPI 上均取得最高分。

**⚠️ 局限性**

局限性：① 仅利用账簿内部结构，未引入外部宏观或行业特征；② 处理的是月度历史，若需更高频率或更长周期预测需重新设计；③ 对极少量历史或极不完整账目（少于 12 个月）仍依赖掩码与平均归一化，精度可能下降；④ 由于是单一面板模型，极端业务模式（如租赁、保险等）可能需要额外的子图或特征。

---

## 362. Beyond Fluency: A Clinical Benchmark and Anomaly-Enhanced Baseline for Spine MRI Report Generation

**arXiv ID:** 2608.07117 | [PDF](https://arxiv.org/pdf/2608.07117v1)

**作者:** Bruno Palau `[一作]` (ETH Zurich), Catherine R. Jutzeler `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对现有视觉‑语言模型（VLM）在腰椎MRI报告生成任务进行基准评估，并指出传统文本指标与临床诊断一致性差。

**💡 创新点**

①引入半监督U‑Net++产生六级椎间盘异常热图作为辅助视觉输入，提升定位与可解释性；②通过结构化级别诊断指标和对抗扰动评估模型鲁棒性，揭示文本指标对临床错误的敏感度不足。

**🔧 技术方法**

使用BiomedGPT、ChatGPT‑5.0、MAIRA‑2、MedGemma、VILA‑M3等VLM；半监督U‑Net++异常检测；QLoRA高效微调；多尺度输入、对抗扰动以及多种语言评估指标。

**📊 数据集**

三大公开腰椎MRI数据集：LSMRI、SPIDER、LumbarDISC。

**📈 对比分析**

采用BLEU‑4、ROUGE‑L、METEOR、BERTScore衡量文本流利度；结构化诊断采用敏感度、特异度、平衡准确率等。结果显示主流VLM流利度高但诊断定位随机（≈50%准确），热图输入显著提升上位椎间盘定位敏感度，而模型对输入扰动不敏感。

**⚠️ 局限性**

传统文本指标无法捕捉临床错误；结构化诊断仍受数据稀缺影响；热图需额外检测模块，对不同VLM兼容性有限；有限数据微调可能导致非解剖合理文本；缺乏与真实临床读片的直接对比。

---

## 363. International Transfer of Stochastic Cortical Self-Reconstruction

**arXiv ID:** 2608.07092 | [PDF](https://arxiv.org/pdf/2608.07092v1)

**作者:** Fabian Bongratz `[一作]` (Technical University of Munich), Christian Wachinger `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文评估了将基于英国Biobank训练的随机皮层自重建(SCSR)模型迁移至包含儿童至老年人群的独立中国数据集的可行性，并比较了直接应用、微调、从零训练和联合训练四种策略。

**💡 创新点**

创新点在于首次系统检验SCSR在跨国、跨年龄、跨扫描设备条件下的迁移性能，并揭示不同网络架构（MLP vs. Spherical UNet）与训练策略对重建精度与脑萎缩检测效果的影响。

**🔧 技术方法**

使用技术包括SCSR的随机采样自重建框架、两种深度网络实现（多层感知机和球面UNet），以及基于重建误差与AUC的评估指标。

**📊 数据集**

数据集包含25,338名英国健康受试者的UKB数据用于训练/验证，以及中国人群数据（4–85岁，640名健康受试者用于训练/微调，160名健康受试者用于验证，139名寿命测试受试者，以及60名CN、60名MCI、60名AD受试者）。

**📈 对比分析**

比较方法为在相同网络架构下执行四种训练策略，对中国寿命测试集和CN/MCI/AD诊断组计算平均绝对重建误差(MAE)和基于AD ROI的AUC；结果显示微调后的Spherical UNet取得最佳AUC 0.848，重建误差在0.35–0.40 mm之间。

**⚠️ 局限性**

局限性包括中国样本量相对较小导致MLP易过拟合，未充分考虑扫描仪差异和人口分布不平衡的联合训练权重问题，以及对不同扫描设备和软件版本的适应性仍需进一步验证。

---

## 364. Human-Centered Explainable AI for TinyML Edge Devices: A Pareto-Based Selection Framework with LLM-Guided Design

**arXiv ID:** 2608.07091 | [PDF](https://arxiv.org/pdf/2608.07091v1)

**作者:** Zeinab Dehghani `[一作]` (University of Hull), Rameez Raja Kureshi `[通讯]` (University of Hull)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了人类中心化、基于LLM的TinyML XAI方法选择框架，并在皮肤病变分类任务上验证。

**💡 创新点**

将LLM用于映射定性利益相关者需求到XAI候选、使用确定性可行性过滤与Pareto多目标优化、提供可审计的选择流程。

**🔧 技术方法**

使用GPT-4.1 mini驱动的LLM提示、XAI方法知识库、确定性可行性规则、三目标Pareto优化、归一化部署成本代理、微小MCU性能测量等技术。

**📊 数据集**

使用HAM10000皮肤病变图像数据集，配合MobileNetV3-Small模型进行实验。

**📈 对比分析**

通过评估67种XAI配置的归因质量、稳定性与相对部署成本，构建三目标Pareto前沿，在三种利益相关者配置下选择非支配方案，其中CAM获得最高复合归因分数且满足成本与稳定性需求。

**⚠️ 局限性**

未在真实MCU上测量能耗与时延，缺乏临床专家对热图的诊断有效性评价，LLM推理受限于提示设计与模型训练。

---

## 365. Scalable High-Fidelity Macromolecular Docking for GPU-Accelerated Supercomputers

**arXiv ID:** 2608.07078 | [PDF](https://arxiv.org/pdf/2608.07078v1)

**作者:** Xiangyu Meng `[一作]` (China University of Petroleum (East China)), Xun Wang `[通讯]` (China University of Petroleum (East China))

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SparkleDock，一个可在 GPU 超算上实现可扩展的光波虫群优化（GSO）灵活大分子对接框架。

**💡 创新点**

主要创新包括：①细粒度的光波虫级并行化；②将能量评分重构为 Tensor Core 可兼容的矩阵运算；③寄存器重映射和异步复制管线，显著提升 TCU 利用率；④基于性能模型的 MPI 负载均衡与自适应 out-of-core 分块。

**🔧 技术方法**

使用了 NVIDIA Tensor Core（FP64 MMA）、CUDA PTX、CuTe/CUTLASS、异步复制 cp.async、MPI、性能与内存模型。

**📊 数据集**

采用了 Protein-Protein Benchmark 5 与 Affinity Benchmark 2（BM5.2）共 55 条 unbound 结构，挑选了 9 条代表性复合物（包括 4GAM）进行评测。

**📈 对比分析**

与 LightDock‑Rust（CPU 40 核）和纯 CUDA 核实现对比；单 GPU 上 A100、H100 速度提升分别为 9.7×/18.9×，512 GPU 时实现 183× 的加速；准确率与原 LightDock 保持一致，Top‑10 成功率 88.9%。

**⚠️ 局限性**

局限性包括：①对极小规模任务的伸缩性受限；②模型预测误差约 12.5%，需考虑 MPI 初始化和资源开销；③主要依赖 FP64 Tensor Core，非 Hopper 架构性能提升有限。

---

## 366. KnifeHunter: Structured Local Representation Learning for Fine-Grained Knife Image Retrieval in Law Enforcement

**arXiv ID:** 2608.07057 | [PDF](https://arxiv.org/pdf/2608.07057v1)

**作者:** Syed Sameed Husain `[一作]` (University of Surrey), Miroslaw Bober `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建 KnifeHunter 体系，包含 25,843 张、543 类的刀具图像数据集，并在英国执法现场部署检索系统。

**💡 创新点**

提出 CoRe-Net，结合 Weibull 激活、SCRL 本地原型学习和双向递归融合，能够在单一向量下保留全局与局部细节。

**🔧 技术方法**

使用 EVA02-Base Transformer、子中心 ArcFace、Weibull 聚合、SCRL 原型聚合及 BDRF 融合等技术。

**📊 数据集**

使用 KnifeHunter 数据集（来自警察证据、零售目录、边境扣押），并在 Medium/Hard 及大规模 distractor 评测。

**📈 对比分析**

与 GeM、DOLG、DELG、TokenNet、SuperGlobal、SENet 等方法对比，CoRe-Net 在 Medium mAP 88.0%、Hard mAP 70.2%，以及 distractor 条件下分别为 85.1%/83.8% 和 64.9%/61.5%，显著优于基线。

**⚠️ 局限性**

局限在于对极端遮挡或反光强的刀具仍易误检，缺乏多模态（如元数据）融合，并未覆盖未见刀型的开放域检索。

---

## 367. Not All Problems Are Best Modeled as MILP: A DSL-Centric Framework for Flexible and Accurate Optimization Modeling

**arXiv ID:** 2608.07040 | [PDF](https://arxiv.org/pdf/2608.07040v1)

**作者:** Shaofeng Zhang `[一作]` (Southern University of Science and Technology), Yong Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了OptiDSL框架，利用LLM将自然语言问题描述映射为领域特定语言（DSL）格式，并自动选择合适的专用求解器完成组合优化问题的建模与求解。

**💡 创新点**

创新点在于：①放弃单一MILP建模，改用DSL中间表示，避免约束爆炸和LLM推理负担；②通过层级式语义路由与逻辑推导实现高质量DSL实例生成；③构建了跨五大领域、44种COP类型的综合基准，展示了DSL与LLM协同的泛化与可扩展性。

**🔧 技术方法**

核心技术包括：大语言模型（DeepSeek‑V3.2 等）进行语义路由、DSL实例化；多维度求解器性能剖析与自适应路由；基于文本生成与校验的半自动基准构造流程。

**📊 数据集**

使用了自研的 OptiDSLBench（44种COP类型、100实例/类型），以及现有的 LLMCoSolver、NL4Opt、MamoComplex、NLP4LP 等数据集进行评测。

**📈 对比分析**

与 Chain‑of‑Experts、ORLM、LLMOPT 等基线进行对比，采用执行率（ER）和最优率（OR）作为评价指标；OptiDSL 在综合基准上提升了 51.66% 的 OR，建模时间减少 91.71%，并在现有基准上比最强基线高 23.09% 的 OR，展示了显著的性能优势。

**⚠️ 局限性**

局限性包括：对极大规模实例的可扩展性仍有限，LLM 可能出现幻觉导致语义不一致；DSL 的定义需要手工维护，对全新领域需额外工作；在某些极其复杂约束的场景下，LLM 的推理能力可能不足。

---

## 368. CAS2UML: A Handwritten Sketch-to-PlantUML Dataset for Class and Activity Diagrams

**arXiv ID:** 2608.07036 | [PDF](https://arxiv.org/pdf/2608.07036v1)

**作者:** Simon Scholz `[一作]` (University of Cologne), Mersedeh Sadeghi `[通讯]` (University of Cologne)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文发布了一个包含557幅手绘类图和活动图的公共数据集，每幅图都配有可执行的PlantUML代码，并提供了验证工具与脚本；

**💡 创新点**

创新点在于提供了大规模、可执行、可验证的手绘UML图数据集，并且覆盖两种核心UML图类型，同时提供统一的验证与处理工具，填补了现有数据集缺乏可执行参考的空白；

**🔧 技术方法**

使用了PlantUML、LLM辅助代码生成（ChatGPT、Claude、Gemini）、Gradio web界面、PlantUML JAR进行语法检查以及XMI导出，配合Python脚本完成数据处理与验证；

**📊 数据集**

数据集来源为Piucco的70幅手绘类图、70幅人工重新绘制的类图以及286幅手绘活动图，总计557幅；

**📈 对比分析**

作者在后续工作中利用该数据集对手绘-UML到PlantUML模型进行微调，通过自动指标和人工排序实验评估，模型在性能上已与专有视觉-语言基线相当；

**⚠️ 局限性**

局限性包括仅覆盖类图和活动图两种类型，某些图无法导出XMI，数据集仍相对中等规模，对不同图像质量与复杂度的覆盖不足，且验证过程依赖人工校准。

---

## 369. XGait: A Multi-Modality Wireless Sensing Dataset for Indoor Human Tracking and Identification

**arXiv ID:** 2608.07064 | [PDF](https://arxiv.org/pdf/2608.07064v1)

**作者:** Wei Xu `[一作]` (Northwestern Polytechnical University), Zhiwen yu `[通讯]` (Harbin Engineering University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aaccfe5c-6b26-4208-b23c-35331481e142` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 XGait 数据集，包含 Wi‑Fi、主动声学和视觉轨迹的多模态室内行走数据，支持定位与身份识别。

**💡 创新点**

创新点在于同步采集三种不同环境（实验室、住宅、会议室）下 27 人、22k 轨迹样本，并将 Wi‑Fi 与声学信号统一映射到 Doppler 频谱，实现跨模态融合的基准评测。

**🔧 技术方法**

使用 Wi‑Fi CSI 与主动声学的时频重定位、PLCR 提取、Doppler 频谱、PPVP 运动模型以及 CNN、LSTM、Transformer 等深度学习方法。

**📊 数据集**

利用自建的 XGait 数据集进行实验，数据涵盖实验室、住宅和会议室三大场景，覆盖多路径和多方向的自然步态。

**📈 对比分析**

对 Wi‑Fi、声学单模与融合的跟踪和识别进行比较，结果显示 Wi‑Fi 在跟踪更稳，声学在识别更精准；两者融合后在复杂轨迹和场景下显著提升精度和鲁棒性。

**⚠️ 局限性**

局限包括性别比例失衡、长期重复记录不足、节点部署相对理想化、同步误差与跨模态融合方法尚需进一步优化，导致在所有场景并不总是能取得最优表现。

---

## 370. Explanation-Guided Metamorphic Testing of Specialized Language Models: An Empirical Study

**arXiv ID:** 2608.07076 | [PDF](https://arxiv.org/pdf/2608.07076v1)

**作者:** Xingcheng Chen `[一作]` (Technical University of Munich), Andrea Stocco `[通讯]` (Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对任务专用语言模型开展了大规模的解释引导变形测试实验，探索基于可解释性信息的变异与语义验证组合的鲁棒性评估。

**💡 创新点**

创新点在于将可解释 AI 的归因结果用于选择变异目标，结合 LLM 生成语义保持的变体，并用双向 NLI 与困惑度过滤来保证测试样本的合法性。

**🔧 技术方法**

使用的技术包括归因方法（如 Occlusion）、LLM 驱动的改写（LLM-Ablate/Inject）、双向 NLI 语义验证、困惑度自然度评估，以及 LoRA 进行对抗微调。

**📊 数据集**

实验数据集涵盖 SST-2（情感）、AG News（新闻主题）和 GitHub Issue Classification（问题分类），覆盖多种任务类型。

**📈 对比分析**

与传统启发式变异基线相比，解释引导方法在验证失败率上提升 2.30×，并在对抗微调后实现最高 7.4% 的鲁棒性提升；但 LLM 变异的计算成本显著更高。

**⚠️ 局限性**

局限在于仅考虑词级变形、仅适用于分类任务，验证器依赖预训练 NLI/LM 的偏差，且对更大生成模型或更复杂任务的泛化性尚待验证。

---

## 371. Tensor Network Kernel Machines: A JAX Framework for Machine Learning and Nonlinear System Identification

**arXiv ID:** 2608.07043 | [PDF](https://arxiv.org/pdf/2608.07043v1)

**作者:** Albert Saiapin `[一作]` (TU Delft), Kim Batselier `[通讯]` (TU Delft)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了开源 Python 库 tnkm，用于构建、训练张量网络核机（TNKM），并提供统一的接口与可扩展的架构。

**💡 创新点**

创新点在于：① 将特征映射、张量网络参数化（CP、TT）和优化算法（ALS、梯度）分离成可组合的模块；② 在 JAX/Optax 环境下实现高效自动微分与 JIT 编译；③ 提供快速训练的 ALS 与灵活的梯度优化，使 TNKM 在保持低参数量的同时能与传统核岭回归和多种系统识别方法竞争。

**🔧 技术方法**

技术栈：JAX（自动微分、JIT）、Optax（梯度优化器）、张量网络分解（CP、TT）、多项式/傅里叶/Volterra 特征映射、ALS 与 Adam 等优化策略。

**📊 数据集**

实验数据集：Airfoil Self‑Noise 回归基准；非线性系统识别基准 Silverbox、Coupled Electric Drives、Cascaded Tanks。

**📈 对比分析**

比较方法：将 TNKM 与核岭回归（KRR）、dynoNet、GPNARX、SUBNET 等主流方法对比。结果显示：ALS 收敛速度快、验证误差接近 KRR；在上述基准上，TNKM 的 RMSE 与最优黑盒/灰盒方法相近，但训练时间仅为秒级，显著低于传统方法。

**⚠️ 局限性**

局限性：性能高度依赖预先设计的特征映射和超参数；缺乏自动特征学习与自动秩选择；ALS 仅适用于二次损失；目前仅支持 CP/TT 分解，未涵盖张量环、Tucker 等更通用分解；未实现贝叶斯不确定性估计等概率扩展。

---

## 372. Near-Optimal Replacement Path Coverings

**arXiv ID:** 2608.07124 | [PDF](https://arxiv.org/pdf/2608.07124v1)

**作者:** Davide Bilò `[一作]` (University of L'Aquila), Martin Schirneck `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种新的(L,f)-替换路径覆盖(RPC)的构造方法，改进了之前的上界和下界。

**💡 创新点**

通过简单的构造方法，得到了更紧的覆盖值，特别是在f = O(L)的范围内。

**🔧 技术方法**

使用了随机化算法和最优采样概率f/(L+f)来构建RPC。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了图的性质和路径覆盖的构造。

**📈 对比分析**

与之前的结果相比，新的上界和下界表明覆盖值为Θ((L+f)^(L+f)/L^L f^f)，在f = O(L)的范围内表现良好。

**⚠️ 局限性**

限制在于仍然存在一个多项式的gap在覆盖值的上界和下界之间，且在查询时间上可能需要进一步优化。

---

## 373. Multiple Hypothesis Flow Estimation for Video Frame Interpolation under Matching Ambiguity

**arXiv ID:** 2608.07120 | [PDF](https://arxiv.org/pdf/2608.07120v1)

**作者:** Zibo Su `[一作]` (Xidian University), Kun Wei `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出一种多假设光流估计框架用于视频帧插值，先保留前K个候选光流，再通过anchor‑centered local attention逐一细化，并用可靠性路由器在训练与推理时硬路由选择最可靠的单一流进行图像合成。

**💡 创新点**

创新点在于：①使用top‑K候选anchor而非单一匹配，保持多模态信息；②通过anchor‑centered local attention实现局部残差细化；③引入可靠性评估网络与straight‑through硬路由，避免软融合导致的幽灵与结构失真；④同时在候选层和最终层使用最佳‑K监督提升鲁棒性。

**🔧 技术方法**

核心技术包括：共享卷积编码器（1/8粗特征与1/2细特征）、粗到细光流推断、anchor‑centered local attention、可靠性评估网络、Gumbel‑softmax直通硬路由、轻量级RIFE‑style生成器、Laplacian pyramid 复原损失与最佳‑K损失。

**📊 数据集**

训练数据：Vimeo90K；评估数据：公开的SNU‑FILM、Xiph，以及新构建的匹配歧义基准MA‑HD。

**📈 对比分析**

与EMA‑VFI、SGM‑VFI、PerVFI、AMT‑G、LC‑Mamba等先进方法对比；在LPIPS和DISTS上取得最佳成绩，尤其在MA‑HD上的匹配歧义场景提升显著；速度和显存相对可接受，保持良好质量‑成本平衡。

**⚠️ 局限性**

局限性：多假设推理与路由增加计算与显存负担；硬路由在极端多模态或极端模糊条件下可能误选；缺乏对长序列时间一致性的显式建模，需进一步研究。

---

## 374. SoK: Cryptographic Key Recovery for Cryptoasset Custody and Financial Technologies

**arXiv ID:** 2608.07104 | [PDF](https://arxiv.org/pdf/2608.07104v1)

**作者:** Francisco Javier Becerra Sanchez `[一作]` (University of Luxembourg), Radu State `[通讯]` (University of Luxembourg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对加密资产托管与金融技术中的密钥恢复机制进行系统化梳理，构建了一个包含77个系统的主矩阵，并提出了基于恢复语义的轴先分类与通用构建模型。

**💡 创新点**

创新点在于：①以恢复对象、失败事件、授权路径和后恢复状态为维度统一定义恢复语义；②提出轴先分类法区分秘密恢复与控制恢复；③构建七阶段通用构建模型，强调恢复后生命周期与元数据隐私；④基于此提出针对金融技术的研究议程。

**🔧 技术方法**

采用系统化文献综述（SLR）方法，利用LLM辅助标题/摘要筛选，手工验证后构造代码书、主矩阵和轴先分类；对比安全性、活性、隐私等属性；对生产系统进行案例检验。

**📊 数据集**

数据集为118篇论文的发现语料，经过筛选得到77篇合成语料，覆盖钱包、托管、智能合约、去中心化身份、硬件等金融场景；对应的代码书与矩阵托管在GitHub公开仓库。

**📈 对比分析**

通过主矩阵编码每个系统的恢复语义、机制、授权、风险等维度，在轴先分类下进行安全性、活性、隐私、可用性和元数据泄露等多维度对比；性能方面通过统计计数和实例阐述六大发现，揭示恢复多样性、信任转移、活性提升带来的滥用路径、后恢复生命周期缺失、用户证据不足与元数据保护薄弱。

**⚠️ 局限性**

局限性：仅覆盖英文同行评审期刊/会议论文，未包含灰色文献或产品文档；对证据的计数基于论文报告，可能低估实际实现；LLM辅助筛选虽已人工验证；缺乏大规模用户研究，未检验高压情境；对后量子迁移等长期问题的探讨不足。

---

## 375. Algorithmic Threshold Optimization: Quantitative Modeling of Multiplier Distributions in Crash Games

**arXiv ID:** 2608.07103 | [PDF](https://arxiv.org/pdf/2608.07103v1)

**作者:** Sourish Sarkar `[一作]` `[通讯]` (Indian Statistical Institute), Sourish Sarkar (Indian Statistical Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种自适应停止倍率算法，用以在Crash游戏中平衡赌场利润与玩家赢利，并对算法的时间与空间复杂度进行了分析。

**💡 创新点**

创新点在于算法不等待读取所有玩家的现金输赢，而是在每一步根据最坏情况提前判断是否继续，让赌场在最大化玩家获胜的前提下保持可控风险。

**🔧 技术方法**

主要技术包括基于概率分布的玩家贡献权重计算、循环迭代的随机现金抽取、最坏情况预算评估以及对停止倍率的递增扫描；随后利用Python SciPy对模拟结果进行分布拟合与KS、AIC评估。

**📊 数据集**

使用的数据集为通过 Monte Carlo 模拟得到的 1000 次实验结果，玩家规模分别为 100、500 与 1000，投资区间统一为 1 至 100 美元（保留两位小数）。

**📈 对比分析**

通过对比 Lognormal、Gamma、Beta、Weibull 与 Pareto 五种分布的 KS 统计量与 AIC 值，发现 Lognormal 在所有规模下均表现最佳，且算法的时间复杂度为 O(n·k)（k 为倍率步数），空间复杂度为 O(n)，在大玩家数下仍可在实时环境中使用。

**⚠️ 局限性**

局限性包括：假设所有玩家的投资均匀分布；未考虑赌场实际设定的投资上下限；算法对倍率步长 δ 的选择对运行时间有较大影响；以及未验证在非均匀投资或动态玩家加入/离开的情况下的表现。

---

## 376. RoRA: Role-Oriented Regional Allocation for Visual Token Pruning in MLLMs

**arXiv ID:** 2608.07088 | [PDF](https://arxiv.org/pdf/2608.07088v1)

**作者:** Qiyanhui Lu `[一作]` (City University of Hong Kong), Jianyuan Guo `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对多模态大型语言模型的视觉token压缩，提出了RoRA框架，在固定预算下将token划分为语义核心、补充上下文和细节三类，利用Attention-Anchored Regions (AARs)进行区域化分配，从而在保留重要视觉信息的同时显著减少token数量并提升推理效率。

**💡 创新点**

创新点在于将token分配视为角色导向的区域证据分配，首次结合定位的AARs实现核心保护、上下文扩展和细节修复的三阶段预算分配，并用轻量级的相似度过滤取代昂贵的全局稀疏图。

**🔧 技术方法**

技术包括基于文本条件的注意力校准、位置先验+对象先验调节、核心Token的TopK保护、AAR构造与上下文得分增益、余弦相似度阈值过滤、细节修复的注意力/特征/对比度加权等。

**📊 数据集**

使用的主要数据集包括GQA、MMBench、MME、POPE、VQAv2、TextVQA、VizWiz、AI2D、MUIRBench等，覆盖问答、图像理解、计数、细粒度等多种任务。

**📈 对比分析**

与FastV、HoloV、D^2Pruner等训练自由视觉token裁剪方法对比，RoRA在相同压缩比例下在LLaVA-1.5、LLaVA-Next、Qwen2.5-VL、Qwen3-VL等模型上平均保留95–99%原始性能，在高压缩（>80%）时表现更优，且token选择延迟仅0.7ms，整体推理时间缩短约24%/25%。

**⚠️ 局限性**

限制方面：在极低token预算（<10%）时仍难以捕获所有细节；对对象先验的弱化需要经验调参；AAR的构造假设核心token能覆盖大部分对象，可能对极其复杂场景不足；在动态分辨率模型中token数量随输入变化，预算分配需进一步自适应。

---

## 377. LifelongCrossNav: Persistent 3D Semantic Memory for Cross-Floor Multi-Object Navigation

**arXiv ID:** 2608.07079 | [PDF](https://arxiv.org/pdf/2608.07079v1)

**作者:** Zehui Li `[一作]` (Peking University), Xiuwan Chen `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为LifelongCrossNav的框架，实现了在未知多层室内环境中顺序多目标物体导航。

**💡 创新点**

创新点在于将支持感知的3D体素映射、跨目标的持久视觉-语言语义记忆以及统一的楼层与楼梯导航策略结合，首次同时解决了多目标记忆和跨楼层路径规划的问题。

**🔧 技术方法**

采用稀疏3D体素地图、语义与几何融合的支持感知建图、基于CLIP的密集视觉-语言特征编码、支持感知的楼梯识别与跨层遍历、以及统一的3D A*规划与模式切换的导航策略。

**📊 数据集**

使用HM3D环境构建的HM3D-MFMON基准（927个三目标序列，其中288个必须跨楼层），并在单目标HM3D验证集上进行对比评估。

**📈 对比分析**

与平面语义记忆基准OneMap和多目标/单目标公开方法对比，LifelongCrossNav在多目标完整序列成功率、路径效率（SPL）和后期目标的条件SPL上均显著提升（例如全序列SR从16.8%提升至约29%，跨楼层子集SR从0%提升至约8%）。

**⚠️ 局限性**

局限性包括：对目标检测的依赖导致误检（如床与沙发混淆）导致额外导航失败；未在真实世界环境中验证；跨层梯子识别的鲁棒性与复杂楼梯结构适配仍待提升。

---

## 378. PTQ4SNN: Membrane-Aware Post-Training Quantization for Spiking Neural Networks

**arXiv ID:** 2608.07066 | [PDF](https://arxiv.org/pdf/2608.07066v1)

**作者:** Hui Xie `[一作]` (Beihang University), Jinyang Guo `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个膜感知后训练量化框架，联合量化SNN权重与膜电位，实现低位宽（4bit权重、平均4bit膜电位）部署；

**💡 创新点**

创新点包括：①通道级统一尺度桥接（Unified Scale Bridge）实现权重-膜电位的移位兼容尺度变换；②基于发射率与量化灵敏度的混合精度膜电位位宽分配（MPBA）实现通道级2/4/8bit位宽自适配；

**🔧 技术方法**

采用通道级量化、移位兼容尺度桥接、活动+灵敏度驱动的混合精度分配、可复用的投影–LIF对、无训练的后训练量化技术；

**📊 数据集**

使用ImageNet‑1K（图像分类）、CIFAR10‑DVS（事件分类）、Pascal VOC2012（语义分割）等数据集；

**📈 对比分析**

与W4/M32、W4/M4、BRECQ、GPTQ、FlowQ等基线方法对比，在SDT‑8‑768、Meta‑SpikeFormer、SEW‑ResNet18等模型上，平均膜电位约4bit、权重4bit时与FP模型差距<1个百分点，事件分类和分割任务同样保持显著优势；

**⚠️ 局限性**

局限性：仅基于少量校准样本，无需再训练，适用于已训练好的模型；对每个通道的位宽与桥接参数需额外计算；在不同硬件上对移位桥兼容性的支持可能受限，且未覆盖动态权重更新场景。

---

## 379. SetEasy: A Multi-Modal Classroom Engagement Assessment and Seating Optimization Framework

**arXiv ID:** 2608.07188 | [PDF](https://arxiv.org/pdf/2608.07188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 380. NiyamAI - An Intent-Bound AI Agent with Cryptographically Verifiable Guardrails using Zero-Knowledge Proofs

**arXiv ID:** 2608.07167 | [PDF](https://arxiv.org/pdf/2608.07167v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 381. From Test-Time Scaling to Reusable Memory: Measuring Crystallization in Text-to-SQL

**arXiv ID:** 2608.07213 | [PDF](https://arxiv.org/pdf/2608.07213v1)

**作者:** Jiaqian Wang `[一作]` (Xidian University), Muning Wen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在文本到SQL系统中，将测试时的修复经验转换为可重用记忆，并通过受控实验测量其对未来查询的价值（回放、跨问题保留、同数据库转移）

**💡 创新点**

提出“结晶化问题”概念，分离三种未来使用场景并量化其贡献；通过对经验来源、验证门控、卡片格式和检索方式的单因素干预，明晰哪些设计真正带来收益；发现数据库特定信息是转移的主要来源，而非仅仅复制匹配示例

**🔧 技术方法**

基于Qwen3.5-27B等大型语言模型的执行引导修复；构造修复片段（episode）并写入卡片；使用余弦相似度在同数据库检索卡片；执行准确率评估；采用两阶段层级自举、McNemar和TOST检验等统计方法

**📊 数据集**

主要使用BIRD开发集（1534题、11个数据库）和Spider开发集（1034题）作为实验数据集

**📈 对比分析**

在三种未来使用设置下对照无记忆基准，使用三种种子进行配对检验；在held-out转移中，verbatim卡片使首次尝试精度从62.04%提升至66.38%（+4.34pp），CR为44.4%；回放准确率达到96.1%；跨问题保留约56%；其他干预如验证门控、检索宽度等亦得到定量评估

**⚠️ 局限性**

只在同数据库内验证，跨数据库/跨租户泄漏风险未充分评估；受限于固定solver和有限的修复预算；未考察多模型、链式思考组合等扩展；实验基于开发集，真实生产负载分布未知；验证门控需要外部可靠信号，若失效会导致负收益

---

## 382. HNR-DAC: Hard-Negative Reranking and Distribution-Aligned Classification for Scientific Claim Verification

**arXiv ID:** 2608.07204 | [PDF](https://arxiv.org/pdf/2608.07204v1)

**作者:** Zhenchao Wang `[一作]` (Southern University of Science and Technology), Shiwen Ni `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个两阶段的科学论证验证框架 HNR-DAC，先用硬负样本重新排序检索段落，再用检索到的最高分段落训练关系分类器，同时输出前三名证据段落。

**💡 创新点**

创新点在于通过“证据可混淆度”挖掘检索过程中的硬负样本，并将检索过程与分类器输入对齐，消除训练与推理之间的分布差异。

**🔧 技术方法**

使用了 Qwen3-Reranker-8B 作为基础检索器、交叉编码器进行硬负样本对比学习、Qwen3.5-27B + LoRA 作为关系分类器、组级对比损失以及标签感知重采样。

**📊 数据集**

使用了 NLPCC 2026 Task 10 Track 2 数据集，训练集 1945 条例子，开发集 217 条例子。

**📈 对比分析**

与通用 LLM、Vanilla pipeline 等方法对比，HNR-DAC 在开发集实现 95.13% Score、94.47% Joint@3、95.79% Macro‑F1；在官方测试中排名第三，Macro‑F1 93.05%、Joint@3 70.16%，性能显著提升。

**⚠️ 局限性**

局限在于证据检索的泛化仍是瓶颈，尤其在异构论文结构中召回黄金证据的能力下降；重采样对性能提升有限。

---

## 383. An AI4AI Framework for Visual Token Pruning

**arXiv ID:** 2608.07193 | [PDF](https://arxiv.org/pdf/2608.07193v1)

**作者:** Zhen Liu `[一作]` (Xi'an Jiaotong University), Jingwen Fu `[通讯]` (Zhongguancun Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出AutoPrune，一种基于大语言模型（LLM）的视觉令牌裁剪框架，实现无需训练的裁剪策略生成；

**💡 创新点**

创新点在于将裁剪问题表述为相对于强基线策略的残差搜索，采用Token Pruning Domain-Specific Language (TPDSL) 131个可重用原子，限制搜索空间、保证预算合规并提升可解释性；

**🔧 技术方法**

技术包括TPDSL残差表示、LLM驱动的候选生成与评估、基于安全检查的执行验证、残差裁剪与分配约束，以及跨预算/跨模型的即时重构；

**📊 数据集**

使用了14个常用多模态基准（VQAv2、GQA、VizWiz、ScienceQA-IMG、HallBench、POPE、MME、MMBench-EN、MMBench-CN、MM-Vet、TextVQA、ChartQA、AI2D、OCRBench）和多种MLLM骨干（LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-7B）；

**📈 对比分析**

与现有手工设计的裁剪方法（如CDPruner、PruMerge+、VisionZip等）比较，AutoPrune在94.4%视觉令牌裁剪率下保留99.7-99.9%的完整性能，且在320令牌设置下实现9.9× FLOPs、6.4×预填充延迟的加速；

**⚠️ 局限性**

局限性在于TPDSL的表达能力受限，需预先设计丰富的原子；LLM生成策略高度依赖评估器的准确性；在极低令牌预算或完全不同视觉编码器结构时仍需进一步验证。

---

## 384. "Operator, can you hear me?" A Faithful Line into the UNISOC Baseband

**arXiv ID:** 2608.07143 | [PDF](https://arxiv.org/pdf/2608.07143v1)

**作者:** Eduard Vlad `[一作]` (École Polytechnique Fédérale de Lausanne), Mathias Payer `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对UNISOC UDX710基带进行系统安全分析，完成固件执行获取、绕过完整性校验、实现运行时内存跟踪、恢复外围设备模型，并在QEMU上实现可验证、可重复的重宿主化。

**💡 创新点**

提出“peer‑and‑interconnect driven re‑hosting”方法，在保留硬件环境和定时精度的前提下，实现对基带完整协议栈的可验证模拟，并提供交互式调试任务与执行跟踪，首次对UNISOC基带进行此类全面分析。

**🔧 技术方法**

使用固件反汇编、动态内存扩展、插桩（trampoline）技术、指令计数时间同步、单线程QEMU主导调度、USB/PCIe AT命令、SDR测试台、Logel日志解析以及Python/ C 交互实现等多种技术。

**📊 数据集**

主要使用Quectel RM500U‑CNV固件映像、UBI分区、tracepoint 字典、以及真实5G测试网产生的 RRC/NAS 报文和 PDU 流数据集。

**📈 对比分析**

通过在真实设备与重宿主中植入 NAS 状态变更钩子、对比控制平面状态转移与报文相似度，以及对同一 PCAP 捕获的控制平面与数据平面流量进行逐字比对，验证两者在同一虚拟时间下保持完全一致；重宿主能够在低负载下实时完成完整 PDU 会话，性能与硬件相当且可重复。

**⚠️ 局限性**

目前仅覆盖 5G NR；LTE/3G/GSM 及其 PHY 共模未建模；重宿主依赖完整无线测试台和预置 SIM；硬件加速器与加密引擎仅通过接口建模，迁移到其他 UNISOC 设备仍需动态探测与手工调整。

---

## 385. Human-AI Perceptual Alignment by Playing Hues and Cues

**arXiv ID:** 2608.07141 | [PDF](https://arxiv.org/pdf/2608.07141v1)

**作者:** Nuria Alabau-Bosque `[一作]` (Universitat de València), Jesús Malo `[通讯]` (Universitat de València)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建基于棋盘游戏《Hues and Cues》的离散色彩空间，收集325名参与者对100个词的色彩记忆，并用此作为人类基准评估162个对比式视觉‑语言模型的颜色认知。

**💡 创新点**

提出将游戏棋盘映射到CIE xy色度图的游戏化评估框架，并通过Human Consistency基线揭示模型在抽象概念上的颜色不一致与“蓝色不确定性崩塌”现象。

**🔧 技术方法**

使用对比式检索模型（CLIP、SigLIP、MobileCLIP等），结合Mahalanobis距离、Hotelling T²检验以及留一交叉验证计算人类一致性。

**📊 数据集**

人类数据来自Hues and Cues数字界面收集的325人群；模型数据为162个基于不同预训练语料（LAION、WebLI、DataComp等）的CVLM检查点。

**📈 对比分析**

将模型Top‑1和Top‑5的颜色预测与人类分布进行统计对齐，结果显示模型在具体物体类别可与人类相当甚至更优，但在抽象、主观和流行文化类别误差远高于人类基线；误差主要集中在语义误分类和蓝色不确定性崩塌。

**⚠️ 局限性**

局限包括远程自适应设备的色彩校准不一致、仅评估检索式CVLM、未对生成式模型进行测试，以及数据集中语言和文化多样性不足。

---

## 386. From probability to causality in probabilistic logic programming

**arXiv ID:** 2608.07230 | [PDF](https://arxiv.org/pdf/2608.07230v1)

**作者:** Zora Wurm `[一作]` (Ludwig-Maximilians-Universität München), Felix Weitkämper `[通讯]` (German University of Digital Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究如何从学习到的概率逻辑程序（Problog）中识别唯一可行的因果结构，并验证其在执行干预时的可靠性。

**💡 创新点**

首次将Meek的方向判定规则推广到关系概率逻辑程序，提出利用关系对称性（尤其是谓词对称性）来补全传统规则，能够同时对分叉（fork）和三叉（collider）进行方向化。

**🔧 技术方法**

技术包括：① 将Problog程序映射为贝叶斯网络；② 采用Meek的可导边（orientability）规则；③ 定义并利用关系对称集合M，实现M‑可导性；④ 在逻辑程序中引入谓词对称性以约束方向；⑤ 在Prolog/Logtalk实现中使用表格化（tabling）以提升效率。

**📊 数据集**

实验以UWCSE大学网页数据集为例（含学生、教师、项目等关系），并使用内部构造的示例数据（如MOE、ANNA的课程记录）。

**📈 对比分析**

方法在该数据集上通过手工构建的关系对称集成功率提升：传统规则只能确定少数箭头，而加入对称性后，能够推断出几乎所有边的方向；没有给出数值性能指标，只说明方向化的完整性和可行性得到显著提升。

**⚠️ 局限性**

局限性：① 对确定性依赖（概率为0或1）的处理依赖于外部推理数据库，容易破坏完全性；② 仍假设谓词对称性成立，若程序含时间或循环结构则不适用；③ 对称性扩展规则并非完备，仍有可导边未被捕获；④ 论文未提供大规模数据集或与现有因果学习工具的定量比较。

---

## 387. Learning Suffers More Than the Policy Class Under Partial Observability: A Closed-Form Analysis

**arXiv ID:** 2608.07228 | [PDF](https://arxiv.org/pdf/2608.07228v1)

**作者:** Idil Gözel `[一作]` `[通讯]` (University College London), Idil Gözel (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文研究了在部分可观测线性二次型控制问题中，标准Actor‑Critic学习者因Critic估计的偏差而导致的学习失效，并给出了闭式解析的学习失配（learning gap）位置和成本；随后证明通过调节λ‑return的bootstrap horizon可以消除这一失配，并在深度RL实验中验证了理论预测。

**💡 创新点**

创新点在于：①首次在可解析的部分可观测LQG设置中精确分离政策类缺口与学习缺口；②给出Critic偏差导致学习误差的闭式表达式；③提出并证明只需调整λ‑return的时间尺度（bootstrap horizon）即可将学习收敛点恢复到最佳静态输出反馈策略；④在深度RL实验中验证理论，且证明输入记忆（如帧堆叠）并不能解决问题。

**🔧 技术方法**

技术方法包括：线性二次型控制理论、马尔科夫过程的闭式统计分析、TD(0)/LSTD固定点求解、梯度期望计算、闭式解析求解学习固定点、λ‑return（GAE）对Critic的影响分析、以及基于PPO的深度RL实现。

**📊 数据集**

数据集：完全自定义的连续时间线性系统（状态维度2，观测为单一坐标，另一个坐标为不可观测噪声驱动），通过离散化得到的仿真数据；无公开真实数据集。

**📈 对比分析**

比较方法：与最优LQG控制器（知情控制）和最佳无记忆静态输出反馈策略（可观测但无记忆）比较；在理论分析中给出成本比例和增益误差；在深度RL实验中对不同λ值（0.9、0.99、1）和不同架构（memoryless vs 32帧堆叠）进行多种随机种子训练，评估部署成本和隐含增益。结果显示：λ≈0.99时成本仅比最优LQG低≈1%，但λ=0.9时成本偏高≈35%。

**⚠️ 局限性**

限制：①结论基于期望更新，未证明随机迭代序列收敛；②对Critic偏差的闭式结果仅对高斯扰动成立；③仅针对单一简单环境和Actor‑Critic框架；④深度RL验证仅限于PPO和公开实现，未覆盖其它网络结构或更复杂环境；⑤未对记忆型Critic或递归架构的效果做系统性研究。

---

## 388. Stochastic Autoregressive Learning

**arXiv ID:** 2608.07224 | [PDF](https://arxiv.org/pdf/2608.07224v1)

**作者:** Ilan Doron-Arad `[一作]` (MIT), Elchanan Mossel `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文构建了一种针对二进制标记的随机自回归（stochastic autoregressive）PAC‑学习框架，研究了三种监督模式——基于一步（base）、链式思路（chain‑of‑thought，CoT）和端到端（end‑to‑end，e2e）的样本复杂度，并给出了统一的理论界限。作者进一步用逻辑回归生成器（logistic autoregressive generators）做案例研究，分别给出不完全（improper）和完全（proper）学习算法，并在端到端模式下证明了在学习噪声偶极子（LPN）假设下的计算难度。

**💡 创新点**

创新点包括：
• 将自回归学习从确定性推广到随机化，揭示了随机性导致的样本复杂度结构的根本差异；
• 在同一准确率尺度下，基、CoT、e2e 三种任务没有主导关系，首次证明了两两间可能出现任意大比例差异；
• 证明 CoT 学习可以通过在准确度缩放为 ε/M² 的基学习来获得上界，且该比例不可改进；
• 提供了一套完整的上界/下界，展示了 CoT 与 e2e 之间的 M/ε 缩放关系；
• 通过 fat‑shattering 维度给出 e2e 学习的精确上界；
• 在逻辑回归案例中，展示了 Improper 学习可达 O(d²logM/ε)，而 Proper 学习在 e2e 下受 LPN 限制，体现了统计与计算的分离。

**🔧 技术方法**

主要技术手段包括：
• 通过构造特殊的根子树（root blocks）和阻断器（blockers）实现参数隔离；
• 利用 KL 散度、Assouad 原理和 Fano 定理得到下界；
• 通过从全路径样本中随机抽取一个时刻得到基学习样本的“投影”实现上界；
• 对 fat‑shattering 维度的细化与上界推导，使用 Rudelson–Vershynin 的覆盖数估计；
• 对逻辑回归生成器的最终概率表达为有理函数，利用金伯格–Jerrum 结果得到伪维度上界；
• 在 LPN 转换中构造将噪声偶极子函数映射为自回归生成器。

**📊 数据集**

本工作为纯理论研究，没有使用真实数据集；所有实验和证明均在构造的合成类（如 𝒢_N、ℋ_B、ℱ_σ(d)）上完成。

**📈 对比分析**

比较方法：
• 通过样本复杂度三角关系 m_base(ε) → m_CoT(ε) → m_e2e(ε) 的上下界给出两两比较；
• 在 CoT 模式下给出 m_CoT(ε) ≤ m_base(ε/M²) 的上界，以及对任意类的匹配下界；
• 在 e2e 模式下给出 m_e2e(ε) ≤ (M/ε)·m_CoT(cε) 的上界，并给出相应下界，说明 M/ε 的缩放是最优；
• 对逻辑回归案例，给出 Improper 上界 O(d²logM/ε)、Proper 上界同阶（可行）以及 LPN 计算下界，显示 Proper e2e 学习在 polynomial 复杂度下不可行。

**⚠️ 局限性**

局限与未来工作：
• 结果仅适用于二进制标记空间，扩展到多标签或连续标记尚未给出；
• 计算复杂度上界主要基于样本复杂度，真正的有效学习算法（尤其是 e2e 的 Proper 学习）在多数情况下仍是未知；
• 对 CoT 与 e2e 之间的 M/ε 缩放存在对数因子，未能完全消除；
• 在 LPN 案例下的计算难度是基于强假设的，实际可否近似实现仍待研究；
• 进一步研究多轮自回归（更高阶生成器）以及多标签、梯度信息的自回归学习仍是开放问题。

---

## 389. Embedding Modal Logics into Logics of Bunched Implications

**arXiv ID:** 2608.07203 | [PDF](https://arxiv.org/pdf/2608.07203v1)

**作者:** Daniele Sansoni `[一作]` (Australian National University), Ranald Clouston `[通讯]` (Australian National University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种全新的、完全语法化的证明，展示经典模态逻辑 S4 嵌入到布尔 Bunched 逻辑 BBI 的可证性，并证明该嵌入在引入假设（多重假设与 bunch 结构）时仍保持一致性。

**💡 创新点**

创新点在于：①利用 Gödel 的逆翻译技巧构造了 BBI 对 S4 的完全可逆映射；②首次给出 BBI 的归纳推导定理，并证明其对假设推理的稳定性；③展示嵌入在所有已知的 BBI 及其语言扩展（如 Hybrid BBI、CBI、BiBBI）和任意 S4 的公理扩展下仍成立。

**🔧 技术方法**

核心技术包括：Hilbert 风格的公理系统与推理规则；逆翻译（[ ]'）构造与证明消去定理；关系语义的部分非确定性幺半群模型；结构化的假设集合（bunch）与对应的归纳推导定理；对公理扩展的参数化推导与对应的资源框架诱导。

**📊 数据集**

该工作为纯理论研究，无使用任何数据集。

**📈 对比分析**

由于论文为形式化证明与理论研究，并未涉及实验或性能评测，故不提供方法比较与性能数据。

**⚠️ 局限性**

局限性包括：①仅针对布尔 BBI 与经典模态逻辑 S4 的嵌入，尚未证明对非经典基底（如 intuitionistic BI）或更复杂的多模态扩展的适用性；②逆翻译虽然是取消映射，但并不保证对所有语义模型的完全保真；③对更大范围的 BBI 扩展（如带指数、资源约束等）的完整性与可判定性仍待进一步研究。

---

## 390. Towards trajectory-unsupervised physics-informed neural solvers for molecular dynamics

**arXiv ID:** 2608.07232 | [PDF](https://arxiv.org/pdf/2608.07232v1)

**作者:** Petros Triantafyllos `[一作]` (University of Piraeus), Christoforos Rekatsinas `[通讯]` (National Centre for Scientific Research Demokritos)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为DINaMo的轨迹无监督物理信息神经网络，用单一可微分函数全局表示分子动力学轨迹，并仅利用牛顿定律、能量和动量守恒等物理约束进行训练；

**💡 创新点**

创新点在于完全消除对模拟器生成的轨迹、力或能量标签的依赖，构建全时域的连续轨迹表示，结合硬初始条件解析展开、因果加权、尾部残差挖掘和增广拉格朗日能量率约束等机制，实现在 Lennard‑Jones 系统上从物理定律直接恢复出可物理解释的轨迹、能量曲线和径向分布函数；

**🔧 技术方法**

核心技术包括可微分物理信息网络（PINN）框架、时间与初始状态编码器、逆平方根非线性（ISRU）多层感知器、硬初始条件 ansatz、因果权重、top‑k 与尾部残差加权、增广拉格朗日能量率约束，以及 Adam 优化器和自适应惩罚参数更新；

**📊 数据集**

使用了 Lennard‑Jones Argon 系统的 NVE 轨迹数据，系统尺寸分别为 50 颗原子（密度0.15）和 500 颗原子（密度0.50），时间窗口从 0.12 到 0.20 τ_LJ；数据仅用于验证，训练过程不使用任何模拟器输出；

**📈 对比分析**

与已有基于数据监督或数据辅助的物理信息网络（如 PND、NP‑SNN）以及传统模拟器的比较，DINaMo 在无监督条件下实现了与参考轨迹在位置、能量（K、U、E）和径向分布函数几乎相同的误差；在 50 原子 T=0.12 的基准实验中，位置 95% 分位误差 <5×10⁻⁵ σ，速度 <2×10⁻³ σ/τ，能量漂移 3.5×10⁻⁵ ϵ/atom；在 0.20 窗口和 500 原子密集系统中，误差虽然有所提升，但仍保持在 10⁻⁴–10⁻⁶ ϵ/atom 级别，表明能量保持和结构保真度可通过物理约束实现；

**⚠️ 局限性**

局限性包括：仅能覆盖短时窗口（≤0.20 τ_LJ）；需为每个初始状态单独训练，缺乏可泛化的单一模型；对更复杂的势能面或多组分体系尚未验证；在长时窗口下动力学相位误差（速度）成为主要瓶颈；整体方法在大规模体系中仍需进一步优化和加速。

---

## 391. Measuring Concept Content in Text from LLM Activations: ESG Evidence from Concept Vectors and Linear Probes

**arXiv ID:** 2608.07208 | [PDF](https://arxiv.org/pdf/2608.07208v1)

**作者:** Luc Hazenoot `[一作]` (Leiden University), Amirhossein Zohrehvand `[通讯]` (Leiden University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究通过读取冻结大型语言模型（LLM）的内部激活，利用线性探测器和递归特征机（RFM）等方法，衡量财务文本（ESG）中概念（环境、社会、治理）的存在程度，并与传统表面基准、嵌入基准及模型自身回答进行对比。

**💡 创新点**

创新点在于：①证明冻结模型的激活能近似域专用微调分类器的性能；②线性探测器在所有激活读取方法中始终优于RFM；③RFM产生连续分数，提供了仅通过分类无法得到的“概念强度”评估；④展示包装器（prompt wrapper）对结果影响显著，提出对包装器的进一步研究。

**🔧 技术方法**

技术手段包括：冻结LLM（Llama‑3.1‑8B、Qwen‑3‑8B/14B、Gemma‑4‑31B）激活提取；线性探测器（RidgeClassifier + Logistic回归，嵌入层级级堆叠）；递归特征机（AGOP矩阵、特征向量提取）；多种 token 池化策略（last‑token、mean、max）；嵌入基线（Qwen‑3‑embedding‑8b）；多折交叉验证与嵌套 CV；以及对模型自身回答的 AUC/Accuracy 评估。

**📊 数据集**

使用人类标注的 ESG 数据集（2000 句/三大 Pillar），包含环境、社会、治理三类，每类包含 1000 句共享、750 句领域专属、250 句通用句子，标签为 0/1。

**📈 对比分析**

比较方法：线性探测器、RFM 概念向量、嵌入基准、表面基准（词典计数、主题比例）、模型自身回答；评估指标包括 AUC、Accuracy、F1、Precision、Recall。结果显示：线性探测器在所有 Pillar 上 Accuracy 均为 0.945–0.951，接近或超越微调模型（差距 ≤ 2.1%），且在 11/12 组合中优于模型自身回答；RFM 仍落后，平均 Accuracy 0.866–0.946；嵌入基准 Accuracy 0.836–0.943，表面基准显著逊色。

**⚠️ 局限性**

局限性：①需要手动选择包装器，包装器差异可导致 9.4 点的准确率波动；②RFM 的连续分数未得到基于排序标签的验证；③仅使用了 2000 句的小型 ESG 数据集，无法证明对更大、更多样化数据的泛化；④Gemma‑4‑31B 的实验因显存不足被省略，模型覆盖不完整；⑤激活读取方法对模型结构与预训练数据分布敏感，可能在其他领域表现不佳。

---

## 392. Aneto: Predicting System Performance by Exploiting Cross-Workload Regularity

**arXiv ID:** 2608.07179 | [PDF](https://arxiv.org/pdf/2608.07179v1)

**作者:** Raul Taranco `[一作]` (Huawei Technologies), Michael Giardino `[通讯]` (Huawei Technologies)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在仅利用一次裸机运行的三个计数器（CPI、LLC miss per instruction、miss penalty）就能估算阻塞因子（BF），从而在不做频率扫频或模拟的情况下预测任何工作负载在不同内存配置下的周期/指令(CPI)和性能。

**💡 创新点**

提出了跨工作负载的BF回归模型，揭示了BF与log(CPI)和log(MPI·MP)的平滑关系，从而实现单测量即可完成BF估算，突破了以往需要多点扫频、trace驱动或大量模拟的高成本限制。

**🔧 技术方法**

基于CPI分解（CPI = CPI₀ + BF·MPI·MP）的分析框架，采用对数变换后的BF对数线性回归（logit模型），并结合指数变换得到闭式BF预测公式；对BF进行留一交叉验证以评估泛化能力。

**📊 数据集**

在超过100个工作负载的基准集合上进行实验，涵盖SPEC CPU2017、PARSEC、GAP、Ligra、GMS、DaCapo、AI‑ML、DPC‑4 traces；在五台真实平台（AMD Zen 2–5、Intel Comet Lake、ARM Server）以及ChampSim和Sniper模拟器上收集计数器并生成基准数据。

**📈 对比分析**

与传统的多点Sweep（Clapp）和单测量预测器PROFET对比，采用留一交叉验证、P50/P90相对CPI误差和BF绝对误差评估；结果显示在8×MP外推时P50误差约为5–8%，P90误差约为15–20%，比PROFET低约1.5–2倍，并且相对于完整Sweep仅减少约0.5%误差，表明在单测量下已接近模拟器的精度。

**⚠️ 局限性**

假设BF随MP保持不变且CPI分解为线性，易受工作负载低R²或高内存占比影响；对prefetch敏感或存在显著phase变化的工作负载预测误差增大；需要在新平台上进行一次基准工作负载的Sweep以校准模型；在不同厂商架构或极端内存技术（如深度分布式内存）时，跨平台参数共享的效果可能下降。

---

## 393. Fluid-DiT: Graph-Free Diffusion Transformers for Fluid Flow Simulations Learning

**arXiv ID:** 2608.07161 | [PDF](https://arxiv.org/pdf/2608.07161v1)

**作者:** Shentong Mo `[一作]` (Carnegie Mellon University), Guolin Ke `[通讯]` (DP Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一种图结构自由的扩散变压器（Graph-Free Diffusion Transformer），用于直接从无结构网格中学习流体平衡状态的分布，替代传统的图神经网络方法。

**💡 创新点**

创新点包括：① 用全局自注意力取代多步图消息传递，消除显式图构造与层级化池化；② 在潜在空间中进行扩散，分离几何细节与分布学习，显著抑制高频噪声；③ 引入距离编码等轻量级空间先验，保持物理一致性。

**🔧 技术方法**

技术手段主要包括：扩散概率模型（DDPM）框架；Transformer 多头自注意力模块；潜在编码器/解码器（MLP/卷积）实现低维表示；位置编码（空间坐标与时间步）与距离编码；训练采用 AdamW、cosine 退火、梯度裁剪等。

**📊 数据集**

使用的基准数据集包括：① 2D 圆柱悬摆（Re=100）圆柱尾迹；② 2D 椭圆流动（不同长宽比）的壁面分离场；③ 3D 湍流翼流（Re=2000）高雷诺数乱流，涵盖多尺度结构。

**📈 对比分析**

与 DGN、LDGN、GM-GNN、VGAE 等基线对比，评价指标包括 R²、Wasserstein 距离、RMS 误差、两点相关以及训练/推理时间。该方法在所有数据集上均实现了更高的 R²（最高 0.998）、更低的 Wasserstein 距离（最低 0.084）、更快的推理速度（52 ms vs 128 ms/170 ms）以及显著的训练加速（约 2.5×）。

**⚠️ 局限性**

局限性：① 仍需要大规模 GPU 进行训练，潜在空间编码可能丢失细节信息；② 目前仅聚焦平衡分布，未处理时间演化；③ 对极大规模网格或更高雷诺数湍流的泛化尚未充分验证。

---

## 394. Edge Sparsification via Temporal Forman-Ricci Curvature for Dynamic Graph Learning

**arXiv ID:** 2608.07158 | [PDF](https://arxiv.org/pdf/2608.07158v1)

**作者:** Poupak Azad `[一作]` (University of Manitoba), Kiarash Shamsi `[通讯]` (University of Manitoba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于时间Forman–Ricci曲率的边稀疏化框架，用于高效学习大规模时序图。

**💡 创新点**

创新点在于将经典Forman–Ricci曲率扩展到有向加权时序图，融合端点支持、时间近似度和局部竞争，形成可直接用于任务无关边重要性评分的指标。

**🔧 技术方法**

采用时间窗口分离、日志平滑端点强度、指数衰减时间核、稀疏化策略（高曲率保留）以及图级特征提取+序列预测（LSTM/GRU）等技术。

**📊 数据集**

使用了12个有向加权时序图数据集，包括9个区块链代币转账网络和3个TGBL基准（tgbl-coin、tgbl-review、tgbl-comment）。

**📈 对比分析**

与MoG、TEDDY、SEM三种主流稀疏化基线在相同80%边删除预算下进行比较，平均ROC–AUC保持97.7%（≈97.7%相当于完整图），并在30项实验中取得最佳结果27次，平均运行时间缩短55.94%。

**⚠️ 局限性**

局限在于：需预先设置时间窗口和衰减参数τ；对极端稀疏度（>80%）性能下降；主要验证在图级预测任务，对节点级或连续时间学习场景的通用性待进一步探讨。

---

## 395. Machine Learning-Based Inter-Crystal Scatter Recovery for Ultra-High Resolution PET Imaging

**arXiv ID:** 2608.07155 | [PDF](https://arxiv.org/pdf/2608.07155v1)

**作者:** Alexandre Bernier `[一作]` (Université de Sherbrooke), Jean-Baptiste Michaud `[通讯]` (Université de Sherbrooke)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

开发并验证了一种结合几何预处理的前馈神经网络，用于恢复超高分辨率PET中的晶体间散射事件（ICS），从而显著提升系统灵敏度并保持亚毫米级空间分辨率。

**💡 创新点**

首次将几何变换与深度学习相结合，在像素化PET系统中将被丢弃的ICS事件转化为有效共振事件，实现灵敏度提升超过70%且空间分辨率基本不变。

**🔧 技术方法**

采用前馈神经网络（tanh激活）、几何预处理（坐标归一化、旋转、缩放）、GATE蒙特卡罗模拟、3D OSEM重建、NEMA NU4与Mini Derenzo斑马体phantom、以及动物实验数据。

**📊 数据集**

使用约2.2 M（LabPET II Mouse）和3.8 M（UHR Brain）模拟ICS事件数据，结合点源、phantom与小鼠（^18F‑NaF、^18F‑FDG）实验数据。

**📈 对比分析**

与传统仅双事件（Doublet）方法对比；ICS恢复率约69%；LabPET灵敏度提升106%，UHR提升73%；保持1.6 mm分辨率，CNR、均匀性与渗透比仅下降<10%，并实现扫描时间或剂量减半的效果。

**⚠️ 局限性**

对极限分辨率（1.2 mm）略有降级；多次散射事件未被处理；算法主要为离线后处理，实时性受限；约30%事件定位仍有误差。

---

## 396. Identifying the Key Biomechanical Features of Movement Adaptation during Exoskeleton-Assisted Locomotion

**arXiv ID:** 2608.07140 | [PDF](https://arxiv.org/pdf/2608.07140v1)

**作者:** Peter Seungjune Lee `[一作]` (Karlsruhe Institute Of Technology), Katja Mombaur `[通讯]` (Karlsruhe Institute Of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a4b10f5d-130b-4e77-9367-6469ec621899` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过对9名健康受试者在三种步态条件（无外骨骼、主动踝关节外骨骼辅助、零扭矩模式）下的跑步机步态进行连续记录，利用标记无捕捉技术和间接热量测量，提出并应用了基于关节轨迹RMSE和三维角度平面线性依赖度的短期运动适应评估框架，分析了个体化的关节运动和代谢成本随时间的演化；

**💡 创新点**

创新点在于将运动适应研究从传统的稳态统计转向个体化、时间分辨率高的动态分析，首次量化并比较不同关节在外骨骼辅助下的收敛速度和同步性，并通过线性依赖度揭示外骨骼对下肢协同运动的影响；

**🔧 技术方法**

主要技术包括无标记三维运动捕捉（Theia3D）、间接代谢测定（Cosmed K5）、步态事件自动分割、关节轨迹RMSE计算、三角角度投影到最优平面并求解最小奇异值得到线性依赖度、以及自评适应报告的统计分析；

**📊 数据集**

数据集为9位受试者在三种条件下的连续运动轨迹与呼吸代谢数据，所有受试者均以自选步速完成6+25+6分钟的实验段，此外收集了每位受试者的个人信息与实验记录；

**📈 对比分析**

比较方法采用相对收敛阈值（2° RMSE）和线性依赖度RMSE随时间的变化，结果显示主动辅助模式下的关节收敛时间明显延长，且代谢成本表现出高度个体差异，平均代谢提升约5%，但标准差高达18%；

**⚠️ 局限性**

主要限制包括样本量小、部分受试者因设备故障缺失数据、仅在跑步机上测试限制了自然步态的外推、适应时间仅为25分钟且未检验长期适应、以及主观适应报告与客观指标不一致，需在更大且多样化人群中进一步验证。

---

## 397. Online Conformal Prediction Beyond Feedback

**arXiv ID:** 2608.07139 | [PDF](https://arxiv.org/pdf/2608.07139v1)

**作者:** Joar Skalse `[一作]` (King's College London), Nicola Paoletti `[通讯]` (King's College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在在线合格预测（OCP）中引入“查询”机制，仅在拒绝预测时才获取标签，解决无反馈情形下的置信区间估计问题。

**💡 创新点**

将此问题转换为部分监控游戏，借助标签高效指数加权预测器实现对阈值的学习，首次在无预测反馈的OCP中获得理论上最优的 O(T^{2/3}) 惩罚率与 β-O(T^{-1/3}) 覆盖保证，并仅以 T^{-1/3} 的查询频率实现。

**🔧 技术方法**

核心技术包括：部分监控游戏框架、标签高效指数加权（label‑efficient forecaster）适配、奖励函数设计（鼓励小阈值但保证覆盖）以及对冲算法的理论分析。

**📊 数据集**

实验使用公开数据集：MNIST、USPS、MNIST‑C、CIFAR‑10、CIFAR‑10‑C、CIFAR‑100、CIFAR‑100‑C 以及用于安全监测的 WildGuardMix（包含普通与对抗重写提示）。

**📈 对比分析**

与全反馈 Adaptive Conformal Inference（ACI）及基于半/多臂赌博机的 OCP 方法对比；OCPQ 在大多数设定下实现了覆盖率高于基线且接近理论下界，效率略逊于全反馈方法，但在对抗分布偏移时差距仅为 5–6% 的查询比例。

**⚠️ 局限性**

局限性：仅适用于无记忆（oblivious）对手，理论下界相对保守导致实际覆盖率往往高于预期且效率下降；需要手动设定阈值集 M 与 β，且对标签分布的假设极为宽松，缺乏对自适应对手的适应性。

---

## 398. Agent Memory Distillation: Empowering Small LLM Agents with Hierarchical Teacher Memory

**arXiv ID:** 2608.07169 | [PDF](https://arxiv.org/pdf/2608.07169v1)

**作者:** Taeil Kim `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Agent Memory Distillation（AMD）框架，用教师模型的成功轨迹生成三层结构化记忆（工作流、子任务、函数），并将这些记忆注入到小型学生模型中，实现训练免费、无参数更新的知识蒸馏。

**💡 创新点**

创新点在于：① 针对小模型的能力差距，提出三层层次化记忆结构，分别对应任务规划、执行示例和细粒度工具调用；② 结合主动注入（预先提供计划与示例）与被动注入（错误时恢复），使学生能在不同执行阶段获得恰当帮助；③ 通过教师轨迹而非自我学习生成记忆，避免学生低成功率导致记忆噪声。

**🔧 技术方法**

使用的技术包括：① 基于密集向量相似度的记忆检索（工作流、子任务）；② 函数级记忆按函数名索引；③ 预训练文本嵌入模型编码记忆；④ 任务分解、检索过滤和上下文注入；⑤ 使用 GPT‑5‑mini 作为教师、Qwen3、Gemma、Llama3.1 等作为学生。

**📊 数据集**

实验数据集涵盖三大工具使用基准：AppWorld、BFCL V3 和 ToolSandbox。

**📈 对比分析**

与教师、零射以及 ReasoningBank、MemP、SASM 等基线对比，AMD 在三大基准上分别实现平均 27.2%p、11.2%p、3.4%p 的准确率提升；部分任务上学生模型甚至超过教师水平，且显著减少交互轮数，逼近教师轨迹效率。

**⚠️ 局限性**

局限性包括：① 仅在文本工具调用场景验证，缺乏多模态或开放式代码生成的适用性；② 记忆是离线生成且静态，无法在推理时融入学生自身成功/失败经验，也不适应分布漂移；③ 依赖教师轨迹质量，教师与学生的兼容性未完全解决，仍需针对不同学生选择最合适的教师。

---

## 399. MAUPITI: On-Device Prototype-Based Learning on a Smart Infrared Sensor

**arXiv ID:** 2608.07192 | [PDF](https://arxiv.org/pdf/2608.07192v1)

**作者:** Beatrice Alessandra Motetti `[一作]` (Politecnico di Torino), Daniele Jahier Pagliari `[通讯]` (Politecnico di Torino)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在MAUPITI智能红外传感器上实现了基于CNN编码器与最近类均值(NCM)分类器的端到端在线学习框架，支持姿态和手势识别。

**💡 创新点**

创新点在于通过离线训练的度量学习与量化感知训练，只在设备上更新类原型，消除了反向传播和回放缓冲区的存储与计算开销，实现低功耗、低内存的增量学习。

**🔧 技术方法**

主要技术包括：使用3层3×3卷积的tiny CNN编码器，三元组损失度量学习，INT8量化感知训练（PACT/PLiNIO），RISC‑V Ibex核心的SIMD乘累加，原型向量的整数更新与位移优化。

**📊 数据集**

采用了两组数据集：5类姿态识别（空场、站立、双臂举起、单臂举起）和9类手势识别（拳头、钉头、食指、shaka、拇指/手腕上下、张开手）。

**📈 对比分析**

与传统CNN+softmax分类器以及无回放的微调/全网络微调等基线相比，NCM在离线训练时可达94.8%准确率，仅低2‑3%；在在线1样本/4样本学习时，单类准确率可达88%，双类68%，并在无回放场景下优于所有基线，整体训练+推理延迟低于0.3%。

**⚠️ 局限性**

局限性包括：必须预先离线训练CNN编码器；类数受限于内存容量；在更复杂的多类任务中准确率下降；对量化误差敏感，需精细调校；缺乏在设备上对编码器本身进行进一步微调的能力。

---

## 400. A Tight Bound for Facial Distance Patterns in Planar Graphs

**arXiv ID:** 2608.07187 | [PDF](https://arxiv.org/pdf/2608.07187v1)

**作者:** Viktor Fredslund-Hansen `[一作]`, Oren Weimann `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在无向无权平面图中，证明了所有顶点相对于给定面顶点的差值模式（pattern）最多只有O(k^2)种，其中k为面边数。

**💡 创新点**

突破了此前的O(k^3)上界，首次匹配已知的Ω(k^2)下界，并完成了ISAAC'22提出的猜想。

**🔧 技术方法**

核心技术是将模式映射为“弱分离子集”，利用Leclerc‑Zelevinsky关于弱分离子集最大量的定理（以及对应的递归构造），并给出一个更简洁的直接证明。

**📊 数据集**

论文属于理论研究，未使用实验数据集。

**📈 对比分析**

基于新的O(k^2)上界，改进了以下几个结果：
- Okamura–Seymour度量压缩从O(k^3+|T|)压缩到O(k^2+|T|)；
- 常数查询时间距离指针空间从O(n^{7/4})降至O(n^{5/3})；
- 在CONGEST模型下的直径算法从O(D^5)轮改为O(D^3)轮；
- 在单机无权平面图中直径的时间复杂度从O(n^{5/3})提升到O(n^{8/5})。

**⚠️ 局限性**

仍存在两大限制：
1. 直径计算仍在加权与无权平面图之间存在时间差距（加权图最快为O(n^{5/3})，无权图仅到O(n^{8/5})）。
2. 该方法依赖于无权无向平面图的特殊结构（弱分离子集的性质），难以直接推广到加权或有向图。

---

## 401. Conformal Fusion Under Missing Modalities

**arXiv ID:** 2608.07183 | [PDF](https://arxiv.org/pdf/2608.07183v1)

**作者:** Alireza Moayedikia `[一作]` `[通讯]` (Swinburne University of Technology), Alireza Moayedikia (Swinburne University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种多模态融合架构——Modality-Conditioned Conformal Fusion（MCCF），能够在任意模态缺失的情况下自动调整不确定性并输出具有置信度保证的预测集。

**💡 创新点**

创新点在于将模态感知的证据分解、Dempster‑Shafer 组合规则和 Mondrian 合成校准集成到同一网络中，实现对模态缺失模式的组条件覆盖保证，并提供模态级的不确定性归因。

**🔧 技术方法**

技术手段包括多模态瓶颈 Transformer 背骨、模态随机丢弃、每模态 Dirichlet 证据头、Dempster‑Shafer 组合、基于模态掩码的 Mondrian 合成校准以及可微分的集合大小惩罚。

**📊 数据集**

实验数据集包括：Synthetic（4模态，5类）、AVMNIST（图像+音频，10类）、CMU-MOSEI（三模态情感分类，7类）以及 UR-FUNNY（三模态幽默检测，2类）。

**📈 对比分析**

与温度标定的 softmax、仅使用分裂合成校准的 baseline 以及 TMC 进行对比，MCCF 在所有模态子集上均实现了 90% 的目标覆盖率，误差低于 ±2.5%，且保持与传统方法相当甚至略优的预测准确率，同时预测集尺寸更紧凑。

**⚠️ 局限性**

局限性包括：随着模态数的增加，Mondrian 分组的指数膨胀导致每组所需的校准样本量增长，难以在 M≥5 的情形下维持稳定估计；在二分类任务中仅能比较总覆盖率而无法细粒度评估；此外，当前实验未覆盖更大模态数或更复杂任务。

---

## 402. Representation-driven Endoscopic Visual Embedding Alignment for Latent Generation

**arXiv ID:** 2608.07176 | [PDF](https://arxiv.org/pdf/2608.07176v1)

**作者:** Francisco Caetano `[一作]` (Eindhoven University Of Technology), Fons van der Sommen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出并训练了REVEAL——一种基于SiT扩散变换器的端到端生成与特征提取基础模型，使用域特定视觉编码器在近500万医学内镜帧上进行无监督预训练，生成高保真医学图像并能作为强大的下游特征提取器；

**💡 创新点**

创新点在于：① 在生成模型中首次将域特定自监督视觉编码器作为教师进行representation alignment，显著提升了内镜图像的结构细节保留；② 采用iREPA的卷积投影和空间归一化，强化局部空间关系，避免传统MLP对高频细节的损失；③ 在同一模型框架下兼顾生成与判别任务，展示生成模型在下游分类任务上可超越专门的内镜基础模型；

**🔧 技术方法**

技术手段包括：Latent VAE (SD2) 提取编码；Scalable Interpolant Transformer (SiT) 作为生成器；representation alignment（iREPA）最大化教师-学生隐藏状态的余弦相似度；联合去噪与对齐损失；线性探针评估；以及无监督的 inpainting/outpainting 通过前向采样重混合实现；

**📊 数据集**

数据集：GN‑5M（4.8M未标注内镜图像）用于预训练；POLAR（腺瘤/非腺瘤多发性息肉分类）与私有BE（Barrett’s食管癌）用于判别基准；BE-C 为BE测试集的人工加噪版本，用于鲁棒性评估；

**📈 对比分析**

方法对比：在POLAR、BE和BE‑C上使用5折线性探针评估AUC/AUPRC；与SAM2、DINOv2/3、EndoViT、Endo‑FM等模型对比；REVEAL在POLAR、BE上均取得与最佳教师对齐模型相当或更优的AUC/AUPRC，并在BE‑C上保持最高的鲁棒性；生成质量通过FID评估，REVEAL在5M全数据与大模型（SiT‑L/2）上F1低于2.5，远优于其他对齐与无对齐方案；

**⚠️ 局限性**

局限性：① 依赖SD2 VAE压缩的潜在空间，导致高频细节捕获有限；② 目前仅支持无条件生成与基于潜在的 inpainting/outpainting，缺乏条件生成与分割等下游微调；③ 对某些强度高的图像失真（如过曝、色相漂移）仍敏感；④ 训练成本高，需多GPU和大规模数据；⑤ 公开权重虽提供，但对跨设备、多中心的泛化验证仍待进一步探索。

---

## 403. Exact Adaptive Hybrid Retrieval Without Fixed Top-L Cutoffs

**arXiv ID:** 2608.07152 | [PDF](https://arxiv.org/pdf/2608.07152v1)

**作者:** Chunran Zhang `[一作]` `[通讯]` (Southwest Jiaotong University), Chunran Zhang (Southwest Jiaotong University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Exact Adaptive Hybrid Retrieval (EAHR)，在保持完整列表加权 Reciprocal Rank Fusion (RRF) 结果一致性的同时，按需动态决定密集检索和稀疏检索的访问深度，避免了传统固定 Top‑L 截断导致的结果不确定性。

**💡 创新点**

创新点包括：
- 将检索结果约束改为“完整列表结果 + 内部深度状态”，消除了固定深度与结果质量耦合的问题；
- 设计可恢复的精确排序生成器 PVS（密集）和 PBM（稀疏），能在保持排序不变的前提下按需产出下一排名；
- 通过上下界推导的停止判定，确保在确认 Top‑K 完整性后即可终止，减少无谓的检索工作。

**🔧 技术方法**

使用技术：
- 加权 Reciprocal Rank Fusion (weighted RRF)；
- Per‑Vector Scalar Quantization (PVS) 用于密集向量的可恢复精确排序；
- Posting Block‑Max (PBM) 用于稀疏倒排索引的可恢复精确排序；
- 通过 PVS、PBM 产生的精确前缀，结合融合边界信息实现自适应停止；
- Qdrant 搜索引擎作为基础实现。

**📊 数据集**

数据集：
- BEIR test splits：NFCorpus、SciFact、TREC‑COVID；
- TREC‑DL 2019 与 2020（MS MARCO 8.8M passages）；
- CORD‑19 的五个时间快照（从 51k 到 191k 文档）用于时间迁移实验。

**📈 对比分析**

比较方法与性能：
- 与两类完整列表基线对比：
  1）相同生成器、相同调度的 exhaustive batch；
  2）同一生产者、全列表遍历的 same‑producer exhaustive；
- 在五个集合上，EAHR 与 exhaustive batch 的几何平均延迟比率从 1.90（NFCorpus）提升到 30.28（TREC‑DL 2020）。
- 与 same‑producer exhaustive 相比，EAHR 的速度提升在 2.36‑16.42 倍之间。
- 但在部分极端案例（如 anti‑correlated 排名、查询 855410/443396）EAHR 仍可能慢于完整列表遍历。

**⚠️ 局限性**

局限性：
- 仅在满足“贡献非负且非递增、确定性 tie‑break” 的检索器上有效；
- 当前实现仅支持单节点、单 shard；多节点、跨 shard 一致性仍需扩展；
- 结果契约要求在同一固定快照下才能保证正确性，变更时需重新刷新；
- 对冷缓存、并发负载、存储成本等实际部署情况未做充分评估；
- 对更复杂的融合规则或多检索器组合的适应性尚未验证。

---

## 404. DiDPO: Diff-in-Diff Policy Optimization for Coding Agent Training

**arXiv ID:** 2608.07147 | [PDF](https://arxiv.org/pdf/2608.07147v1)

**作者:** Xucong Wang `[一作]` (University of Science and Technology of China), Pengkun Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种针对编程任务的强化学习算法Diff-in-Diff Policy Optimization（DiDPO），通过在代码差分中动态划分子差分并构建分层信用分配，提升代码编辑策略的学习效果。

**💡 创新点**

创新点在于将代码差分拆解为语义子单元，利用Groupability Score选取可聚合的子差分作为锚点，形成跨轨迹的本地优势组；同时在无额外批评者或额外回合的前提下，将轨迹级优势与子差分级优势结合，实现精细化信用分配。

**🔧 技术方法**

核心技术包括：差分元数据构造、子差分相似性匹配、子差分锚点的子模最优化选择、差分级优势计算以及与轨迹级优势的加权融合；实现了无需价值网络的政策优化。

**📊 数据集**

使用从CodeRL+收集的训练数据集（包含大约7K经过增强的任务），在八大编程基准上评测：APPS、HumanEval、MBPP、LiveCodeBench、LeetCode、USACO、OJBench 与 ICPC。

**📈 对比分析**

与大规模语言模型（Kimi‑K2.6、Qwen3.6‑27B、GLM‑5.2、GPT‑5.5）、推理基线（CoT、CodeAct、Self‑Planning）以及其他RL基线（CodeRL+、GRPO、GiGPO）对比，DiDPO在大多数基准上取得最高平均准确率（7B/4B基座下分别为48.4%与58.6%），比GiGPO提升约4–5个百分点，且在USACO等竞争级任务上显著优于GRPO。

**⚠️ 局限性**

局限性包括：①子差分匹配和锚点选择仍依赖阈值和相似度度量，可能遗漏语义相近但文本差异大的修改；②计算子差分相似性带来额外训练时开销；③方法主要针对可执行反馈的编程任务，尚未验证在更广泛的交互式或多模态任务中的适用性。

---

## 405. InstanceSplat: Instance-Aware Feed-Forward 3D Gaussian Splatting for Scene Understanding

**arXiv ID:** 2608.07144 | [PDF](https://arxiv.org/pdf/2608.07144v1)

**作者:** Minchao Jiang `[一作]` (Shanghai Jiao Tong University), Wentao Zhu `[通讯]` (Eastern Institute Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的前向 3D 高斯 splatting 框架（Instance-Aware Gaussian Representation），能够在无相机位姿信息的多视角图像中一次性构建既能生成高质量新视角渲染，又能完成实例分割和开放词汇语义理解的三维表示。

**💡 创新点**

创新点包括：① 将实例身份作为 3D 高斯的内在属性，直接在共享的 3D Gaussians 上学习实例嵌入；② 3D-Consistent Instance Grounding 模块通过渲染空间对比学习实现跨视角一致的实例特征；③ Instance-Centric Coupling 模块在实例结构的基础上实现重建、实例学习和语义学习的互相强化，使得边界感知重建、语义对同类实例的区分以及实例级语义聚合等过程相互促进。

**🔧 技术方法**

使用的技术包括：DINOv2 预训练视觉特征提取器；24 层 Transformer（交叉视角注意力）进行多视图编码；DPT‑style 几何解码器预测相机参数、深度和高斯属性；基于置信度的 voxelization 合并像素级高斯；对实例嵌入进行可微渲染并采用原型对比学习；使用语言对齐的 CLIP 语义特征并通过投影对齐；边界加权的 RGB 损失以及 HDBSCAN 聚类实现实例级语义聚合；训练时结合多任务损失（重建、实例对比、语义对齐、边界强化）。

**📊 数据集**

训练数据集为 ScanNet++ 与 ScanNet（共 1,565 场景），其中 ScanNet++ 的 InsScene‑15K 版本提供高质量实例标注。评估包括：扫描网络的 50 个保留场景；对新视角渲染使用 PSNR、SSIM、LPIPS；对实例分割使用 mIoU、mAcc；在 LERF（未见数据集）上评估跨数据集泛化；对 IGGT 提供的时间序列实例追踪使用 Temporal mIoU 与 Temporal Success Rate。

**📈 对比分析**

与当前前向 3DGS SOTA（AnySplat、LSeg、LSM、Uni3R、C3G）以及基于场景优化的 IGGT、OpenGaussian、InstanceGaussian 等进行对比。实验显示在 2‑8‑16 视角设置下，本文方法在新视角渲染的 PSNR（最高 22.19）和 LPIPS（最低 0.286）均位列前茅；在实例分割上 mIoU 最高可达 65.23，且在 LERF 上跨数据集表现优于所有对照组；相较于 IGGT，Temporal mIoU 提升至 64.03，Temporal Success Rate 提升至 51.04。

**⚠️ 局限性**

局限性：随着视角数量和高斯数量的增加，基于 HDBSCAN 的实例聚类成本显著上升；在加入更多视角时，域迁移导致相机位姿误差累积，进而使某些指标（如 Temporal mIoU）略有下降；此外，对语言对齐的语义特征依赖于预训练模型，可能在极端语义场景下受限。

---

## 406. Rust Coreutils: Rebuilding Unix Foundations in a Modern Language

**arXiv ID:** 2608.07135 | [PDF](https://arxiv.org/pdf/2608.07135v1)

**作者:** Sylvestre Ledru `[一作]` (Debian), Stefano Zacchiroli `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

重写并实现了 GNU coreutils 的 Rust 版本（uutils coreutils），实现了与 GNU coreutils 的功能等价和兼容。

**💡 创新点**

通过将现代 Rust 语言特性（安全、并行、Unicode、零拷贝）与生态库结合，提供了更易维护、可扩展的实现，并通过增量兼容策略实现了功能增强。

**🔧 技术方法**

采用 Rust 语言、Cargo、clap、libc、异步 I/O、OSS‑Fuzz、LibFuzzer、差异化 fuzzing、持续集成等技术。

**📊 数据集**

使用 GNU coreutils 官方测试套件（630+ 用例）、Toybox、BusyBox 以及自建差异化 fuzzing 输入，覆盖功能、边界和错误情况。

**📈 对比分析**

通过单元/集成测试、外部端到端测试、差异化 fuzzing 以及对 Ubuntu 25.10 真实部署的性能基准，最终实现与 GNU coreutils 的 99%+ 兼容率，并在多种场景下维持或提升性能；在大规模部署中识别并修复了若干边缘错误。

**⚠️ 局限性**

主要局限在于缺乏完整规范（POSIX 只覆盖一小部分），测试覆盖仍有盲点；依赖大量外部 crate 可能带来供应链风险；在某些极端用例下仍需改进错误处理与时间命令行为一致性。

---

## 407. Exact Computation of Trait-induced Merge Trees for Bivariate Fields

**arXiv ID:** 2608.07181 | [PDF](https://arxiv.org/pdf/2608.07181v1)

**作者:** Petar Hristov `[一作]` (Linköping University), Talha Bin Masood `[通讯]` (Linköping University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了针对点特征的多字段特征诱导树（TIMT）的精确图基础构造方法，利用单元内部子水平集凸性只在边上检测最小值并拼接单调链，从而得到与真实距离场等价的 Merge Tree。

**💡 创新点**

首次给出点特征 TIMT 的精确构造与证明，揭示其非零事件与 Jacobi 集合的关联，解决了先前仅采用顶点采样近似、无法捕获细粒度拓扑细节的问题。

**🔧 技术方法**

采用 CGAL 精确几何算子与 VTK 进行可视化，构建加权图后使用 TTK 的 Merge Tree 算法计算 TIMT，并与基于顶点采样的近似方法进行对比。

**📊 数据集**

在三棱柱示例、三维环面（torus）和气象模拟中的云触发特征数据集上进行实验，涵盖多种几何与属性分布情况。

**📈 对比分析**

通过计算 Wasserstein 距离和对树进行持久性简化来评估精确与近似 TIMT 的差异；实现仅需约 1 秒完成图构造，图规模略大于一阶骨架，但整体性能可接受。

**⚠️ 局限性**

仅适用于点特征、二维属性空间和四面体网格；对非凸单元、高维属性或更复杂特征的精确构造仍缺乏理论支持，且近似方法在特征落入图像或投影边长过大时误差显著。

---

## 408. Beyond the Black Box: Interpretable Models of Human Randomisation Failures

**arXiv ID:** 2608.07220 | [PDF](https://arxiv.org/pdf/2608.07220v1)

**作者:** Ngoc Linh Dao `[一作]` `[通讯]` (Alpen-Adria-University of Klagenfurt), Ngoc Linh Dao (Alpen-Adria-University of Klagenfurt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在O'Neill四卡游戏中探究人类随机化失误的可解释性，构建并比较了黑盒LSTM与可解释的EWA及其改进模型，利用LASSO挑选特征并分析重复‑回避行为与频率跟踪的预测作用。

**💡 创新点**

首次将机器学习挑选的频率跟踪特征嵌入EWA框架（ME3*），并通过LASSO和对比实验验证“重复‑回避”是主要可预测信号，证明可解释模型几乎能达到黑盒模型的预测性能。

**🔧 技术方法**

使用经验加权吸引（EWA）及其改进（ME1/ME2/ME3*）、LASSO多项式逻辑回归、决策树、深度神经网络（DNN）与长短期记忆网络（LSTM），采用最大似然估计、JAX优化、Kullback‑Leibler（KL）、战略错误率（SER）和相对完整度（RC）等指标。

**📊 数据集**

84,060次决策来自2,802对玩家在O'Neill四卡游戏的30轮实验数据。

**📈 对比分析**

采用5折交叉验证按玩家对划分，比较KL、SER和RC三项指标；结果显示LSTM最高，ME3*在红方约89% RC、黑方98% RC，EWA和常数模型表现较差，证明可解释模型几乎逼近黑盒性能。

**⚠️ 局限性**

局限性包括：频率跟踪窗口大小（w∈{3,4,5}）未系统检验，模型参数对齐和可解释性解释仍待深入，未验证因果关系和更长记忆对预测的影响，且仅在此游戏环境中验证，泛化性待进一步研究。

---

## 409. The Exact Second Generalized Covering Radius of Binary Primitive Triple-Error-Correcting BCH Codes

**arXiv ID:** 2608.07215 | [PDF](https://arxiv.org/pdf/2608.07215v1)

**作者:** Isaac Barouch Essayag `[一作]` (MIGAL Galilee Research Institute), Aryeh Lev Zabokritskiy `[通讯]` (Tel Hai University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

确定了二进制原始三重错误纠正BCH码C_m(3,m)的第二广义覆盖半径R_2(C_m)=8，适用于所有m≥5。

**💡 创新点**

解决了三重错误纠正BCH码家族的参数问题，提供了R_2的确切值，并且与已知下界相匹配。

**🔧 技术方法**

使用了代数论证和确定性验证的方法，结合了交集列的论证。

**📊 数据集**

研究了二进制原始BCH码C_m(3,m)，长度为2^m-1，适用于m≥5的所有整数。

**📈 对比分析**

通过与已知的下界进行比较，证明了R_2(C_m)的上界为8，且在m≥5的范围内达到了等式。

**⚠️ 局限性**

在m=5到m=16的有限范围内，使用计算机辅助的精确验证，尽管在m≥17的范围内，代数论证提供了上界。

---

## 410. Toward a Causal Data Management Ecosystem for Decision Making and Agentic AI

**arXiv ID:** 2608.07214 | [PDF](https://arxiv.org/pdf/2608.07214v1)

**作者:** Dazhuo Qiu `[一作]` (Lyon 1 University, CNRS Liris), Andrea Mauri `[通讯]` (Lyon1 University, CNRS Liris)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种名为 Causal World System（CWS）的框架，用来把因果知识整合进企业或科研实验室的多源 AI 生态系统，实现可查询、可维护的因果图谱。

**💡 创新点**

创新点在于：①把因果模型视为共享、可查询的“媒介化因果模式”，与传统数据整合（GAV/LAV）保持一致；②支持多层次（局部、子系统、全局）因果抽象与推理；③将因果结构直接注入模型训练、代理决策与分析查询，形成闭环的因果驱动生态。

**🔧 技术方法**

使用技术包括：结构因果模型（SCM）与因果发现算法（PC、GES、FCI、NOTEARS 等）；数据整合技术（mediated schema、view‑based integration）；因果抽象与传输技术；识别与 do‑calculus、数据融合、运输性推理；以及用于查询优化、增量维护的数据库式技术。

**📊 数据集**

论文未给出具体公开数据集，示例中以企业内部多源数据（表格、日志、图、文档、图像、音频等）为例，说明系统可在真实业务场景下部署。

**📈 对比分析**

论文没有提供实验评估或与其他方法的对比；主要以概念设计、架构图与技术路线描述为主，强调未来需要在真实生态中实现并评估。

**⚠️ 局限性**

局限与挑战包括：①生态级别的因果发现与整合仍缺乏可行算法；②视图维护与漂移管理的增量策略；③跨模态因果对齐与不确定性校准；④代理对因果查询的低延迟与规划组合；⑤可识别性、可信度与治理问题；⑥缺乏公开实验验证与基准。

---

## 411. Authoring and Management of Transparent Research Integrity Assessments of Randomised Clinical Trial Publications Using LLM-assisted Tools and Provenance Knowledge Graphs

**arXiv ID:** 2608.07202 | [PDF](https://arxiv.org/pdf/2608.07202v1)

**作者:** Milan Markovic `[一作]` (University of Aberdeen), Alison Avenell `[通讯]` (University of Aberdeen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了INSPECT-AI，一款基于大型语言模型（LLM）的半自动化工具，帮助评估随机对照试验（RCT）研究诚信，并使用RIPE-O本体记录评估过程的可追溯信息，构建了RIPE-KG知识图谱；

**💡 创新点**

创新点在于：①将LLM与半自动化流程相结合，实现对INSPECT‑SR清单的快速评估；②使用可追溯本体（RIPE‑O）和知识图谱（RIPE‑KG）将评估过程标准化、透明化，并支持FAIR原则；③在人机交互框架下实现“人工在环”，提升评估可靠性；

**🔧 技术方法**

技术包括：LLM（Gemini 2.0 Flash）进行文本抽取与推理；GROBID提取结构化元数据；YARRRML映射将日志转换为RDF；RDF/OWL本体（RIPE‑O）与PROV‑O；SPARQL查询；Web技术（PostgreSQL、Docker、Web UI）；

**📊 数据集**

数据集：95篇RCT论文（共140份评估），其中13名志愿者评估了69篇论文并生成104条评估轨迹；还引用了Retraction Watch、PubPeer、ClinicalTrials.gov、OpenAlex等外部数据源；

**📈 对比分析**

通过比较自动评估与人工评估的结果，发现自动与人工一致率为86.4%，差异率13.6%；对不同问题（如撤稿、后续评论、研究团队问题、注册时序）一致率差异明显；总体评估一致率约72.7%；工具成本低于0.10美元/篇，处理速度快；

**⚠️ 局限性**

局限性：INSPECT‑AI目前仅支持部分INSPECT‑SR检查；依赖OpenAlex进行作者去重，受其数据完整性限制；外部信息源（Retraction Watch、PubPeer、SemOpenAlex）质量不一；未实现完全自动化，需要人工干预。

---

## 412. Flow-Corrected Shape Optimization: Taming Manifold Drift in High-Dimensional 3D Models

**arXiv ID:** 2608.07199 | [PDF](https://arxiv.org/pdf/2608.07199v1)

**作者:** Emilien Seiler `[一作]` (EPFL), Pascal Fua `[通讯]` (EPFL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种交替使用梯度下降和流匹配纠正的 Flow-Corrected Shape Optimization 框架，用于在高维潜在空间中优化 3D 形状并抑制 manifold drift。

**💡 创新点**

创新点在于将目标最小化与几何校正解耦，利用预训练流匹配模型进行引导纠正，既保留高表达力又避免了传统方法中的表达力与有效性权衡，且在大规模模型上保持计算可行。

**🔧 技术方法**

核心技术包括梯度下降、流匹配（Flow Matching）与引导（Classifier-Free Guidance / Gradient-based Guidance）、预训练解码器、替代目标（体积、拖拽、弹性位移）以及 surrogate 模型（GraphSAGE GCNN 等）。

**📊 数据集**

实验使用 ShapeNet（椅子、汽车等）、ABO、Hunyuan3D 数据集，结合 OpenFOAM CFD 仿真和 GraphSAGE GCNN surrogate。

**📈 对比分析**

与标准梯度下降、Flow Matching Guidance、D-Flow、ICTM、SGO 等基线进行对比，采用 Fréchet Inception Distance/KID 以及目标值下降率衡量。结果表明本文方法在保持形状真实性的同时，在体积、拖拽和弹性位移任务中均优于基线。

**⚠️ 局限性**

局限性包括相对纯梯度下降更高的计算成本；每个优化周期需要多次流匹配评估；在高优化目标下可能出现细微几何瑕疵；超参数（梯度步数 M、噪声比例 τ）固定，缺乏自适应调节。

---

## 413. Momba: Network Modernization Improves Multi-Objective Reinforcement Learning

**arXiv ID:** 2608.07180 | [PDF](https://arxiv.org/pdf/2608.07180v1)

**作者:** Adam Štafa `[一作]` (Masaryk University), Joni Pajarinen `[通讯]` (Aalto University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在多目标强化学习（MORL）中引入并验证了更强的神经网络架构，构建了 Momba 算法，提升了已有的 CAPQL 方案。

**💡 创新点**

创新点包括：①将单目标深度 RL 架构 SimbaV2（含观察/特征归一化、权重归一化、分布式评论家）迁移到多目标域；②提出对标量化后的多目标返回分布进行分布式学习；③在仅修改网络结构和评论家损失的情况下实现显著性能提升。

**🔧 技术方法**

使用了 SimbaV2 的特征归一化、观察归一化、权重归一化；采用 C51 风格的分布式评论家（交叉熵损失）学习标量化返回分布；并在 CAPQL 的熵正则化框架下实现。

**📊 数据集**

在 7 个连续控制任务上评估：Ant、Swimmer、HalfCheetah、Humanoid、Walker2D、Hopper‑v4（2 目标）和 Hopper‑v3（3 目标）。

**📈 对比分析**

与 CAPQL、PGMORL、DPMORL、GPI‑LS 等基线比较，Momba 在聚合指标上实现约 35% 的 Hypervolume 提升、16% 的 Expected Utility 提升；在 Ant、Humanoid 等环境中仅需 100k–200k 步即可达到 PGMORL 的最终性能，且在样本效率上超过 GPI‑LS。

**⚠️ 局限性**

局限性包括：仅针对线性标量化；只在连续控制、少量目标（最多 3 目标）任务上验证；在更大目标维数或离散环境中效果未知；更复杂的网络可能导致训练时间增长。

---

## 414. Capacity Confounds and Coverage Guarantees in Adaptive Sub-model Federated Learning

**arXiv ID:** 2608.07157 | [PDF](https://arxiv.org/pdf/2608.07157v1)

**作者:** Alireza Moayedikia `[一作]` (Swinburne University of Technology), Alicia Troncoso Lora `[通讯]` (Universidad Pablo de Olavide)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并评估了在系统异构的联邦学习中，通过客户端更新差异动态分配子模型容量的HAS‑FL框架，探讨了是否能从子模型更新中估计数据异质性并利用此信息进行容量分配；

**💡 创新点**

创新点在于揭示了更新差异估计被容量本身显著混淆的“容量混杂”问题，阐明了未覆盖参数导致模型退化的失败模式，并通过覆盖保证修复该问题，同时证明随机分配与容量预算相同的结果与自适应分配无显著差异；

**🔧 技术方法**

采用了子模型分割（宽度裁剪）、梯度归一化更新差异估计、指数移动平均平滑、按容量与异质性比例分配规则、坐标加权聚合等技术；

**📊 数据集**

实验使用了CIFAR‑10、EMNIST、自然分区的Shakespeare文本数据集；

**📈 对比分析**

与多种基线（HeteroFL、FjORD、Uniform、随机预算、FedAvg、SCAFFOLD、FedProx）对比，HAS‑FL在相同平均容量下与随机预算表现一致，均低于全模型训练，且在统一容量下的Uniform方案在EMNIST会崩溃；

**⚠️ 局限性**

局限在于仅在20个虚拟客户端上评估、Shakespeare准确率低、只考虑了梯度差异估计，未探索其他可分离容量与数据的估计方法，且仅适用于结构化宽度裁剪的子模型。

---

## 415. Representation Handoffs for OpenArm-Based Laboratory Mobile Manipulation

**arXiv ID:** 2608.07154 | [PDF](https://arxiv.org/pdf/2608.07154v1)

**作者:** Yang Shen `[一作]` (University of Technology Sydney), Chin-Teng Lin `[通讯]` (University of Technology Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文基于OpenArm构建了一个可移动实验室操作原型，将双臂、移动底盘、竖直滑轨、RGB‑D摄像头、激光雷达等硬件与ROS2/MoveIt、LLM规划器、技能库等软件耦合，利用“representation handoffs”实现从自然语言指令到可执行动作的完整信息流。

**💡 创新点**

核心创新在于将系统中所有关键转化过程（语言→技能调用、感知→地图/物体姿态、物体先验→角色与技能约束、技能调用→运动目标）统一视为显式中间表示，并通过校准、验证与安全门控把这些表示层级化、可追踪化，形成一种可审计、可调试的语言驱动机器人平台。

**🔧 技术方法**

使用技术包括：ROS2 与 MoveIt 进行运动规划与执行，OpenArm 硬件与双臂、滑轨、移动底盘；RGB‑D 摄像头、激光雷达用于感知与定位；FoundationPose 服务实现点云/网格物体检测与姿态估计；AprilTag 作为桌面锚点；LLM（大语言模型）作为统一规划器；以及自定义的技能银行、profile 配置文件和运行时绑定框架。

**📊 数据集**

论文未使用公开数据集，而是依赖实验室自定义的物体资产（点云/网格、角色与技能信息）和在模拟/现场获取的 RGB‑D 与激光雷达数据。

**📈 对比分析**

通过 dry‑run 追踪和启动检查验证表示链是否完整，可生成完整的可执行轨迹；然而本文未给出真实场景下的成功率、误差统计或与其他方法的对比指标，主要侧重于表示层面的验证与调试。

**⚠️ 局限性**

主要局限包括：仍需完成真实传感器的标定、物体模型与掩码的准备；缺乏对真实视觉抓取、倒灌、插入等任务的性能评估；受限于预定义的技能库与角色，LLM 无法自由生成 ROS 指令；安全门控过于保守，可能导致任务被过早中断；整体系统对手工配置（profile）高度依赖。

---

## 416. Interpretable reinforcement learning with decision-tree pruning

**arXiv ID:** 2608.07151 | [PDF](https://arxiv.org/pdf/2608.07151v1)

**作者:** Mark Leon Ringer `[一作]` (Ludwig-Maximilians-University Munich), Michel Tokic `[通讯]` (Siemens AG)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

通过将训练好的强化学习策略蒸馏为决策树并采用结构、纯度与运行时访问计数相结合的剪枝流程，系统地简化策略并评估其可解释性与性能。

**💡 创新点**

提出了可审计的剪枝方法，利用决策树自适应约束剪枝（DACP）结合最大深度和最大不纯度剪枝，形成可追溯的简化轨迹。

**🔧 技术方法**

使用决策树蒸馏、叶子数作为可解释性代理、三种剪枝策略（最大深度、最大不纯度、DACP）以及基准评估与可视化技术。

**📊 数据集**

在经典控制与MuJoCo环境（CartPole、Acrobot、Pendulum、HalfCheetah、LunarLander 等）上进行实验。

**📈 对比分析**

与原始教师网络在同一基准下比较奖励和叶子数，结果显示剪枝后保持或略高的奖励，同时叶子数显著下降；在多数任务中几乎无性能损失，甚至出现过拟合消除导致的奖励提升。

**⚠️ 局限性**

仅用叶子数衡量可解释性，未进行人类评估；剪枝受树大小上限和计算成本限制，可能影响不同任务的通用性。

---

## 417. EMAS: Stabilizing Multi-Agent System Evolution through Evidence-Guided Revision

**arXiv ID:** 2608.07196 | [PDF](https://arxiv.org/pdf/2608.07196v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 418. Tabular Image: a method to convert tabular data to images for convolutional neural networks

**arXiv ID:** 2608.07132 | [PDF](https://arxiv.org/pdf/2608.07132v1)

**作者:** Junhao Liang `[一作]` (University of Leeds), Barbara Summers `[通讯]` (University of Leeds)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种将表格数据转换为图像的“Tabular Image”方法，并利用二维卷积神经网络（ConvNeXt）对信用违约进行预测。

**💡 创新点**

创新点在于：①将信用评分常用的权重证据（WOE）和信息值（IV）直接嵌入图像像素；②采用基于IV比例与Spearman相关性优化的特征排列算法，最大化局部相关性；③提供可扩展的通用框架，可适配不同领域的表格数据。

**🔧 技术方法**

技术细节包括：WOE/IV计算、分箱、z-score标准化、特征映射到二维像素、基于0/1整数规划的特征布局、ConvNeXt 2D CNN训练、与1D CNN、MLP等网络对比，使用交叉熵损失、Nesterov SGD 等优化。

**📊 数据集**

实验使用三大公开信用数据集：台湾信用卡违约（TC）、Home Credit违约风险（HC）和Fannie Mae按揭贷款（FM），涵盖样本量从3万到30万、违约率从0.3%到22%。

**📈 对比分析**

通过5折交叉验证与传统机器学习模型（Logistic、SVM、DT、RF、AdaBoost、GBDT、XGBoost、MLP）及1D CNN、One-hot/DeepInsight图像变换进行比较。ConvNeXt+Tabular Image在所有数据集上均优于传统模型，尤其在大型、稀疏数据集上提升AUC≈5-29%、H‑measure≈4-46%、KS≈0.20-0.35；对Subprime样本同样表现突出。

**⚠️ 局限性**

局限性包括：①需预先计算WOE/IV，因而对标签分布敏感；②分箱与特征排列方案对结果影响较大，需要经验或网格搜索；③转换过程对大规模数据仍有一定的CPU/内存开销；④方法在纯数值或缺乏可解释性指标的领域可能不如预期；⑤对极端稀疏或高维类别特征的适用性尚未充分验证。

---

## 419. Dual-Node NVIDIA DGX Spark over Tailscale: A Remote-Access Testbed for Distributed LLM Training and Cyber-Threat-Intelligence Fine-Tuning

**arXiv ID:** 2608.07226 | [PDF](https://arxiv.org/pdf/2608.07226v1)

**作者:** Vasanth Iyer `[一作]` `[通讯]` (Grambling State University), Vasanth Iyer (Grambling State University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个双节点 NVIDIA DGX Spark 训练集群，使用专用 200 Gbps 光纤连接和 Tailscale VPN 进行远程管理，完成了 NanoChat 深度 20 的分布式预训练、对 77 条 CISA 安全通告进行 CTI 细调并评估，以及将同一集群用于 400 级 AI 课程和 CompTIA Security+ POGIL 课堂的教学。

**💡 创新点**

证明了小型桌面级 GPU 设备可以通过简单网络划分（管理层与 NCCL 数据层）和容器化方案，快速构建可重复的分布式 LLM 训练与教学混合平台；同时提供了完整的部署脚本、错误排查经验和评估工作流，展示了在本地环境中实现从预训练到细调再到教学的闭环可行性。

**🔧 技术方法**

技术包括：NVIDIA DGX Spark（Grace Blackwell SoC + 128 GB unified memory）、PyTorch DDP + NCCL、Jumbo MTU 9000、Tailscale Mesh VPN、Docker 容器化（host 网络模式）、NanoChat 开源训练框架、Ollama‑hosted LLM 评判器、Flask web 服务。

**📊 数据集**

数据集：1) 用于预训练的自定义文本语料（约 653 M token，四天内训练完成）；2) 77 条 CISA 安全通告（经清洗并映射 MITRE ATT&CK ID）生成的 338 训练 / 37 验证对话；3) 17 条独立的 hold‑out 评估提示，用于 LLM‑judge 测试。

**📈 对比分析**

对比方法：将两节点训练与单节点训练在同一深度 20、相同本地批次 32 进行对比（单节点为估算）。两节点在 4 天内完成约 653 M token，单节点估计需 14 天；每秒吞吐约 1,890  token。细调评估采用 LLM‑judge 得分，CTI 细调在 5 个 CTI 相关类别上略有提升，总分从 2.06 提升到 2.29，差值 0.24，整体性能提升有限。

**⚠️ 局限性**

局限性：仅有两节点，单节点吞吐为估算，未进行严格匹配基准；双节点训练同时增大全局批次，混淆硬件扩展与批次效应；评估样本仅 17 条、仅使用单一 LLM judge，缺乏统计显著性；教学案例为定性操作记录，无学习成效测评；未公开完整训练日志、参数计数与 checkpoint 细节；网络与安全配置仅对内部网络开放，缺乏公开验证。

---

## 420. Skaling: Chinchilla's Exponents Meet Kaplan's Coupling

**arXiv ID:** 2608.07222 | [PDF](https://arxiv.org/pdf/2608.07222v1)

**作者:** Mathurin Videau `[一作]` (FAIR at Meta), Kartik Ahuja `[通讯]` (FAIR at Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的耦合缩放法则（Coupled Scaling Law），通过在Chinchilla加法模型基础上加入一个耦合指数k，解决了模型规模N与数据量D在边界处预测偏差的问题，并且提出了仅需低计算边界数据即可重建完整模型的L形稀疏采样策略。

**💡 创新点**

核心创新在于：①用单个耦合指数捕捉N与D之间的交互，修正了传统加法法则在极端区域的残差；②证明该耦合结构能够保持Chinchilla的闭式最优计算分配；③提出L形采样方案，使得仅用10倍更少的计算量即可获得与完整网格相同的预测精度。

**🔧 技术方法**

技术方法包括：使用Moving Least Squares（MLS）估计损失表面的一阶与二阶导数；在对数空间中用Huber损失和L‑BFGS‑B优化器拟合模型；交叉验证评估（插值、单轴外推、远距离外推、等比计算外推）；对比分析多种缩放形式（Chinchilla、Farseer、Coupled Law）。

**📊 数据集**

实验数据集：Farseer（404个配置，参数1B–6.4B，数据1B–512B，计算1.6×10^18–4.1×10^21 FLOPs）和SK‑Grid（134个配置，参数134M–4.9B，数据316M–316B，计算9.0×10^16–9.9×10^20 FLOPs）。

**📈 对比分析**

通过5折交叉验证比较，Coupled Law在边界和外推场景中将MAPE从Chinchilla的1.48%/1.98%降至0.47%/0.88%；在L形稀疏采样下，Coupled Law的插值与单轴外推误差与完整网格Chinchilla相当，但计算量仅为原来的1/10；在等比计算外推中，Coupled Law平均MAPE 0.60%，比Chinchilla低3.9倍，并且在所有训练状态下保持稳定。

**⚠️ 局限性**

局限性包括：耦合指数k的估计对数据集敏感，在数据与模型几乎线性互相作用的场景（如原始Chinchilla数据）提升有限；只在N–D两轴上验证，未探究其他维度（如序列长度、模型架构变异）的交互；L形采样要求边界数据足够分散，若边界点不足可能导致不稳定；模型假设仍基于单一可降解损失形式，未能捕捉更复杂的损失结构。

---

## 421. A MARL Centered Reference Architecture for Large Language Model Augmentation in Smart Manufacturing

**arXiv ID:** 2608.07148 | [PDF](https://arxiv.org/pdf/2608.07148v1)

**作者:** Fouad Bahrpeyma `[一作]` (HTW Dresden), Dirk Reichelt `[通讯]` (HTW Dresden)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文综述并系统化了在智能制造领域中将大型语言模型（LLM）与多智能体强化学习（MARL）结合的研究，提出了四类LLM接入点（策略、奖励、通信、层次规划）并构建了以MARL为核心的三层参考架构及对应的LLM‑Augmented Dec‑POMDP形式化；同时给出了评估LLM接入条件的决策框架和制造业部署成熟度等级；

**💡 创新点**

创新点在于：①把LLM作为MARL系统的可选附件点进行统一分类；②提出LLM‑Augmented Dec‑POMDP的描述性符号，便于对比与分析；③基于能力评估表和实证边界，给出三层架构（语义层、协同层、保障层）并设计了可操作的LLM接入决策流程；④引入了针对LLM+MARL在制造业的五级部署成熟度评估方法。

**🔧 技术方法**

主要技术包括：分布式多智能体强化学习（Dec‑POMDP、CTDE、价值分解、策略梯度、通信网络）、大型预训练语言模型（LLM、agentic LLM、VLA）、自然语言推理与计划（ReAct、Reflexion、CoELA）、奖励设计与自监督（Eureka、Text2Reward、LAMARL）以及数字孪生与物理仿真验证框架。

**📊 数据集**

数据与实验主要来自公开文献的案例：如StarCraft‑II多智能体挑战、交通信号控制、能源交易、生产调度（动态柔性车间、混合流加工、碳排放约束）以及机器人路径规划等；作者并未构建统一的数据集，而是汇总了上述领域的实验环境与benchmark。

**📈 对比分析**

比较方法基于能力表（语义理解、任务分解、可读通信、奖励生成、闭环学习、协同协调、信用分配、结构重构、截止时间合规、确定性、可重复性与安全保障）以及制造部署成熟度指标；实验结果显示：传统MARL在实时协调与低延迟决策方面表现更好；LLM在语义推理、奖励草拟、可读通信与层次规划上具备优势；但LLM在硬件延迟、安全性、物理执行层的保障方面仍有限。

**⚠️ 局限性**

局限性包括：①缺乏大规模工业现场验证，主要以仿真与少量硬件试验为主；②LLM推理延迟高、成本昂贵，难以满足秒级甚至毫秒级的控制需求；③安全与可验证性不足，缺乏正式证明或行业认证；④架构依赖复杂的多层设计，实际实现成本与维护难度高；⑤LLM与MARL的互补性尚未在制造任务中实现完整闭环，仍需进一步研究和实践。

---

## 422. PHOENIX: Fine-Tuned SLM-Powered Autonomous Satellite Lifetime Extension via Predictive Self-Healing and Multi-Agent AI Recovery

**arXiv ID:** 2608.07126 | [PDF](https://arxiv.org/pdf/2608.07126v1)

**作者:** Sumaiya Islam `[一作]` (University of Dhaka), Harsha Kumara Moraliyage `[通讯]` (La Trobe University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 PHOENIX 系统，集成细调的微型语言模型在 CubeSat 上实现实时故障检测、预测性自愈与语义缓存，并通过多代理地面 LLM 链路生成安全验证的 CCSDS 指令。

**💡 创新点**

创新点在于：① 将轨道相位感知与语义缓存相结合，实现零推理成本的快速自愈；② 采用 DDPM 生成罕见故障样本，解决数据稀缺问题；③ 设计完整闭环自愈与指令生成链路，首次实现从检测到自主修复再到地面确认的全流程。

**🔧 技术方法**

技术包括 TinyLlama/Phi‑1.5 + LoRA 微调、FAISS 语义缓存、DDPM 生成模型、LangGraph/AutoGen 多代理框架、TLE 轨道上下文、Jetson Orin NX 边缘硬件、CCSDS 指令格式。

**📊 数据集**

使用 ESA Anomaly Detection Benchmark（Mission 1，14 年 76 通道）、SatNOGS 观测数据、Mars Express 功率降解数据，以及 DDPM 生成的合成故障序列。

**📈 对比分析**

与 Goetze 等的预测+阈值基线对比，目标至少匹配 88.8% CEF_0.5；缓存模拟显示 62% 命中率，数据压缩 98%；虽然完整实验尚未完成，但设计目标已达或超越基线。

**⚠️ 局限性**

局限性包括：① 未在真实硬件上完成 SLM 与 DDPM 训练与部署；② 缓存适配与漂移机制需实证验证；③ INT4 量化易受辐射误码，需加入校验机制；④ 在线微调可能导致灾难性遗忘；⑤ 对极少见故障的泛化能力仍待提升。

---

## 423. The Sync Heap: Delete First, Ask Questions Later

**arXiv ID:** 2608.07134 | [PDF](https://arxiv.org/pdf/2608.07134v1)

**作者:** Benjamin Aram Berendsohn `[一作]` (Max Planck Institute for Informatics), László Kozma `[通讯]` (Dresden University of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在离线和在线环境下，提出了一种新的堆数据结构——sync heap，能够在不暴露删除元素信息的情况下实现更快的删除操作；并利用此结构给出了线性时间的堆评估算法和单机单位时间调度问题的最优实现。

**💡 创新点**

核心创新在于将堆的修改操作（insert、silent delete-min）与观测操作（find-min、reveal-deletions）严格分离，通过同步（sync）机制在观测时批量处理待执行的修改，突破传统堆的 O(log n) 删除下界；同时结合软堆（soft heap）与可选择堆（selectable heap）实现了高效批处理。

**🔧 技术方法**

使用的主要技术包括：Chazelle 的软堆（允许键被腐败并以常数时间完成插入和删除），可选择堆（支持批量删除最小元素），离线堆评估算法（将插入/删除序列转化为逆向最大堆求解），以及同步策略（在观测时一次性处理所有待执行操作）。

**📊 数据集**

本文未使用具体实验数据集，全部以理论模型（比较模型）和算法复杂度为基础进行分析与证明。

**📈 对比分析**

与传统的 O(n log n) 堆评估与调度算法相比，sync heap 在插入/查询操作上实现常数时间摊销，在删除操作上实现 O(log k)（k 为观测次数）摊销；整个堆评估问题在最坏情况下线性时间完成，单机单位时间调度问题亦由 O(n log n) 降至 O(n)。

**⚠️ 局限性**

限制包括：仅支持插入、silent delete-min、find-min 与 reveal-deletions；不支持 decrease-key、meld 等常见堆操作；算法的最优性建立在比较模型上，对实际实现仍需进一步验证；在极端观测频繁场景下，删除操作的 O(log k) 仍可能不够理想。

---

## 424. Synthesizing Voltage Ride-Through Controllers for Data Centers

**arXiv ID:** 2608.07289 | [PDF](https://arxiv.org/pdf/2608.07289v1)

**作者:** Wayne Wang `[一作]` (University of Michigan), Ang Chen `[通讯]` (University of Michigan)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文利用形式化方法为数据中心设计并合成满足电网VRT（Voltage Ride‑Through）规范的控制器。

**💡 创新点**

创新点在于将VRT规范转化为Signal Temporal Logic（STL）表达式，自动将其编码为控制器合成问题，并在不可行时通过冲突前沿诊断给出最小硬件或负载调整方案。

**🔧 技术方法**

使用技术包括STL规范建模、形式化合成算法、模型检查与闭环控制器设计工具。

**📊 数据集**

实验数据来自对一个200 MW数据中心与140节点输电系统的闭环仿真。

**📈 对比分析**

与传统手工或启发式设计相比，系统能够在可行时自动合成合规控制器，若不可行则提供诊断，表现为高成功率与可解释的改进建议（具体数值未给出）。

**⚠️ 局限性**

局限性包括依赖准确的电力拓扑模型、算法在规模较大时计算复杂度高，以及仿真结果与真实系统可能存在偏差。

---

## 425. Fast end-to-end cloud application cold-start with initscripts

**arXiv ID:** 2608.07358 | [PDF](https://arxiv.org/pdf/2608.07358v1)

**作者:** Ariel Szekely `[一作]` (Massachusetts Institute Of Technology), M. Frans Kaashoek `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种可脚本化的接口和相应的API，让开发者能够把应用程序的初始化步骤拆分出来，在云平台端并行执行，从而实现冷启动阶段的并行化。

**💡 创新点**

创新点在于：①提出了新的“Init”接口与小型WASM实现，让平台能在设置阶段完成网络连接、RPC调用等初始化操作；②设计了高效的结果传递API（共享内存+连接转移），支持无改动或极少改动即可将初始化结果传递给主程序；③通过将通用运行时组件迁移到平台层，保持Init模块体积极小，显著加速下载。

**🔧 技术方法**

使用技术包括：WASM模块在轻量级容器/虚拟机（GVisor、Wasmer、Spice）中运行；异步RPC与连接管理的高层API；共享内存与socket传输实现结果与连接的转移；对Python、C++等多语言的兼容；与现有云启动API的协同扩展。

**📊 数据集**

实验数据集：ServerlessBench Python函数集、基于Rust的图像识别服务器函数、四个微服务（类似Memcached、向量数据库等），以及对Spice快照恢复系统的对比。

**📈 对比分析**

评估方法：对比“有无Init”两种模式下的冷启动总时延，并细分为设置和初始化两段；对不同平台（AWS EC2、裸机、Spice）进行横向比较。结果显示：大多数ServerlessBench函数平均加速约3-5倍，图像识别函数加速约4-6倍；在使用Spice快照恢复时，进一步降低启动延迟约2-3倍；相较传统方式，改进显著且可与快照技术互补。

**⚠️ 局限性**

限制与不足：①需要开发者编写或改写WASM init模块，虽然可复用但仍需额外工作；②对复杂语言特性或大量依赖的应用迁移成本仍较高；③目前仅支持基于RPC的初始化操作，对事件驱动或异步任务的支持有限；④在高网络延迟或存储瓶颈场景下，初始化并行化的收益有限。

---

## 426. H2AL: Hyperbolic Hierarchy-aware Aggregative Learning for Registration-based Few-shot Medical Image Segmentation

**arXiv ID:** 2608.07340 | [PDF](https://arxiv.org/pdf/2608.07340v1)

**作者:** Jia Wang `[一作]` (Dalian University of Technology), Yun Peng `[通讯]` (Beijing Children's Hospital, Capital Medical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种集成超曲层次信息与梯度聚合的双任务框架 H²AL，用于注册‑少量标注医学图像分割（RFMIS）；通过共享编码器、注册与分割解码器实现伪标签生成与分割学习的协同优化。

**💡 创新点**

创新点在于：①引入 Hyperbolic Hierarchy‑aware Infusion (H2I) 模块，利用 Poincaré 球模型的超曲几何通过变换引导的超曲对比学习 (TSHCL) 捕捉解剖层次关系；②在欧氏特征流中通过门控融合 (GIB) 注入层次先验；③设计 Gradient Aggregation (GA) 的一阶段端到端训练策略，将注册与分割梯度聚合以实现跨任务协同学习。

**🔧 技术方法**

使用技术包括共享编码器+双任务解码器、Poincaré 球模型超曲空间映射、TSHCL 超曲对比学习、GIB 门控融合、梯度聚合 (GA)、注册损失（平滑性+相似性+TSHCL）和分割损失（Dice+TSHCL）等。

**📊 数据集**

使用脑部 T1 MRI 数据集（OASIS、PPMI、ADNI、ABIDE）和心脏 CT 数据集（MM‑WHS、ASOCA、CAT08），在 atlas、1‑shot 与 5‑shot 三种少量标注设置下进行实验。

**📈 对比分析**

与全监督、基于 atlas、现有少量标注 RFMIS 方法（如 BRBS、PC‑Reg‑RT、Bi‑JROS 等）在 Dice、NCC、HD95 等指标上进行对比。H²AL 在脑部 5‑shot 与心脏 1/5‑shot 场景下均取得 Dice 与 NCC 最高或接近最佳成绩，尤其在小结构上的 Dice 明显提升；在注册任务中也取得最高 Dice/NCC 并保持低拓扑破坏率。

**⚠️ 局限性**

局限性包括：依赖预定义的层次先验，极端注册失败时性能提升有限；训练对 GPU 资源需求较高；对更大跨模态或跨机构迁移的鲁棒性尚未充分验证。

---

## 427. Natural Language Processing Psychometrics

**arXiv ID:** 2608.07316 | [PDF](https://arxiv.org/pdf/2608.07316v1)

**作者:** Edoardo Sebastiano De Duro `[一作]` (University of Trento), Massimo Stella `[通讯]` (University of Trento)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了NLP Psychometrics框架，利用LLM生成的自我解释问卷文本与情绪、网络结构特征预测心理健康量表得分。

**💡 创新点**

创新点在于将心理测量、网络科学与可解释AI结合，使用LLM的认知数字影子生成受控文本，并通过TFMN和EmoAtlas提取可解释特征，再用SHAP分析特征重要性，证明模型可迁移到日记和真实语料。

**🔧 技术方法**

采用大型语言模型提示生成文本，文本情绪与语义网络分析（EmoAtlas+TFMN），随机森林回归与特征消融，SHAP可解释性分析。

**📊 数据集**

数据来源为9种LLM（Mistral、Qwen、GPT‑OSS、Olmo等）生成的SWLS、PHQ‑9、DASS‑21问卷解释文本；以及LLM生成的日记和由Androids语料库提供的115名受试者的转录语料。

**📈 对比分析**

通过R²、MAE、RMSE、Spearman相关等指标评估，在最佳模型下可解释高达70.8%方差；迁移至日记文本保持显著分离；在真实临床语料上实现AUC≈0.7‑0.78，说明模型具有一定预测性能。

**⚠️ 局限性**

局限在于仅用LLM生成文本训练，缺乏人类真实问卷+文本的训练集；特征受文本长度影响；实验仅在意大利语与单一情绪词典上验证，泛化性和临床应用尚待进一步验证。

---

## 428. The Token Efficiency Index: A Peer-Benchmarked Composite Indicator for AI Token Efficiency

**arXiv ID:** 2608.07304 | [PDF](https://arxiv.org/pdf/2608.07304v1)

**作者:** Caden Wong `[一作]` (Massachusetts Institute of Technology), Himanshu Dhami `[通讯]`

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Token Efficiency Index（TEI），将 AI token 消耗效率聚合成 0‑100 评分，并实现可配置的评分引擎。

**💡 创新点**

首次将 Benefit of the Doubt（BoD）与鲁棒 order‑m DEA 结合，用同类组织对比生成无先验权重的可解释评分。

**🔧 技术方法**

使用最小‑最大归一化、等权重基准、BoD 与 order‑m 线性规划、HiGHS 求解器以及 Sobol 敏感度分析。

**📊 数据集**

基于 Tokscale 开源平台的 78 账号（75 账号，39 组织）在 120 天窗口内的 token 使用日志。

**📈 对比分析**

通过 Spearman 相关评估三种评分方法，robust BoD 与等权重相关系数高于标准 BoD；实验验证 97% 以上扰动能按预期改变评分，显示方法稳定且可解释。

**⚠️ 局限性**

样本稀薄、仅涵盖内部开发者工具、premium‑model 误用未校正、缺乏任务难度信号、外部 API 计费缺失。

---

## 429. Winning by Peeking: Unenforced Budgets and Test-Set Selection Inflate Short-Budget AutoML Comparisons

**arXiv ID:** 2608.07303 | [PDF](https://arxiv.org/pdf/2608.07303v1)

**作者:** Guilin Zhang `[一作]`, Kai Zhao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对短时预算AutoML比较实验的审计，发现并纠正了模型选择在测试集上产生的偏差以及预算未被严格执行导致的计算超支，随后在修正协议下重新跑实验，验证原先的显著优势消失。

**💡 创新点**

创新点在于将选择偏差与预算超支的影响量化并分离，首次在单一运行中记录两种估计量来对比评估，并提出针对短预算实验的实用检查清单。

**🔧 技术方法**

采用轻量级随机搜索AutoML引擎（基于scikit‑learn基准模型），改进搜索循环的时间管理（外部超时与核心分配），并结合统计检验（符号检验、上界估计）进行分析。

**📊 数据集**

使用OpenML上筛选的面板式表格数据集，包括分类与回归任务，来自研究集218、99、269，且满足样本量、特征数与缺失率的过滤条件。

**📈 对比分析**

通过30秒与60秒两种时间预算的胜率比较，原始实验显示系统显著领先，但在修正协议下，系统与竞争者的胜率差异不显著，性能不再优于其他AutoML框架。

**⚠️ 局限性**

局限包括仅使用单一80/20拆分且缺乏交叉验证、修正实验仅在子样本上完成、仅在单台ARM64机器上运行、以及结果对不同硬件或数据集分布的泛化能力不确定。

---

## 430. Data Annotation as Measurement

**arXiv ID:** 2608.07297 | [PDF](https://arxiv.org/pdf/2608.07297v1)

**作者:** Emma Harvey `[一作]` (Cornell Tech), Rene F. Kizilcec `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对132篇相关文献的半系统综述和10位标注团队成员的半结构化访谈，对数据标注流程进行深入分析，绘制关键决策点地图，提出5类问题源（错误、歧义、不可行、主观性、身份）以及将测量理论（可靠性与有效性）引入标注质量评估的框架。

**💡 创新点**

将数据标注重新定义为测量问题，提出系统化的诊断与干预路径；首次将测量理论的可靠性（如测试-重测）与有效性（如面效度、内容效度、预测效度）与标注质量关联，并给出具体实践建议。

**🔧 技术方法**

采用半系统文献综述、半结构化访谈、主题分析等质性研究方法，并结合测量理论的概念框架进行分析与框架构建。

**📊 数据集**

研究主要基于文献（132篇）和访谈（10位参与者）生成的案例数据，没有使用公开数据集进行实验验证。

**📈 对比分析**

本文不涉及算法或实验比较，而是提供理论框架与实践指南；评价标准为方法的完整性、可操作性和对现有工作的新视角，未给出数值性能指标。

**⚠️ 局限性**

局限包括：访谈样本规模小且非代表性；文献综述受公开研究偏倚；缺乏对不同标注场景的量化验证；测量理论在标注中的具体实现仍缺乏标准化工具与实证支持。

---

## 431. WNM-3D: A World Navigation Model with 3D Scene Conditioning for Closed-Loop VLN

**arXiv ID:** 2608.07267 | [PDF](https://arxiv.org/pdf/2608.07267v1)

**作者:** Yuehao Huang `[一作]` (China Telecom), Xuelong Li `[通讯]` (China Telecom)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了基于几何条件的生成式世界动作模型（WNM-3D），用于连续视觉语言导航；

**💡 创新点**

1）将单目RGB历史通过冻结的几何编码器转换为3D场景代币并作为共享条件；2）引入可训练的3D Scene-to-Token Adapter生成固定长度前缀；3）采用三阶段闭环训练（A* SFT→DAgger→DanceGRPO）显著提升导航性能；

**🔧 技术方法**

使用预训练的视觉语言模型、VGGT-Ω几何编码器、Diffusion Transformer（DiT）进行未来视图与动作的联合生成、块因果注意力、流匹配损失、DAgger数据聚合以及DanceGRPO强化学习；

**📊 数据集**

在GN-Bench（Seen/Unseen）上进行训练与评估，并使用A*专家轨迹生成作为监督数据；

**📈 对比分析**

与多种连续VLN基线（CMA、NaVid、UniNaVid、InternNav(S2)、GN-BAE等）对比，WNM-3D在Seen上NE 2.0、OS 87.2%、SR 81.3%、SPL 78.3%，提升SR 22.7p、SPL 19.7p；在Unseen上SR 46.8%、SPL 43.5%，提升7.9p、6.2p；与WNM-2D相比，几何条件在Seen上提升SR 5.7、SPL 5.4；

**⚠️ 局限性**

仅评估XY流动一致性，未考虑偏航或完整视觉一致性；几何编码器被冻结，可能限制对新场景的适应；在Seen‑to‑Unseen迁移上的增益相对有限；

---

## 432. SCALE: Scientific Concept Aggregation via LLMs and Embeddings for Fine-Grained Taxonomy Extension

**arXiv ID:** 2608.07254 | [PDF](https://arxiv.org/pdf/2608.07254v1)

**作者:** Daniele Raimondi `[一作]`, Andrea Perlato `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了SCALE框架，利用LLM、语义嵌入和图聚类在OpenAlex的四层层级之下生成约114k个可解释的科学概念，填补了主题与关键词之间的细粒度缺口。

**💡 创新点**

首次在保持现有层级的基础上引入“概念”这一第五层，通过自动化社区检测与LLM命名生成可维护、可解释的细粒度层级，解决传统关键词碎片化与不稳定的问题。

**🔧 技术方法**

结合SPECTER2/2嵌入、HNSW近邻检索、Leiden图聚类、GPT‑4o‑mini及OpenAI模型进行关键词分层、概念命名与主题链接。

**📊 数据集**

使用MDPI近两百万篇论文的约三百万条作者关键词以及OpenAlex公开的领域/子领域/主题层级数据。

**📈 对比分析**

通过手工curated的必须/不能链接词对进行网格搜索，最终得到69.5% must‑link精度、87.5% should‑not‑link精度，生成113,892概念，98.8%主题关联率；在生产论文分类中覆盖率94%，专家评测Precision@5约88.9%。

**⚠️ 局限性**

需链接对由作者挑选，可能带偏差；概念在不同领域分布不均；当前为层次分类而非正式本体关系；需进一步人工评估与覆盖平衡。

---

## 433. Depth-Wise Probing and Pruning of the Planning Token in a Driving Vision-Language-Action Model

**arXiv ID:** 2608.07361 | [PDF](https://arxiv.org/pdf/2608.07361v1)

**作者:** Harisankar Babu `[一作]` (Robert Bosch GmbH), Simon Foell `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了驾驶Vision‑Language‑Action（VLA）模型的规划令牌在解码器深度中的信息流，并使用轨迹空间原生头镜头对每层令牌进行评估；

**💡 创新点**

提出了一种基于规划令牌角度余弦相似度的剪枝标准，并通过轨迹空间logit lens和线性探针展示了可在保持误差≤5%时删掉四分之一层的可能性；

**🔧 技术方法**

采用轨迹空间logit lens、线性分类探针、可学习的适配器/序列重采样器，以及基于角度余弦的层排序技术，结合ORION模型的32层LLaMA‑style解码器；

**📊 数据集**

使用Bench2Drive数据集（包含merging、overtaking、emergency brake、give‑way、traffic‑sign等五大能力类别）进行实验，评估在不同场景下的性能；

**📈 对比分析**

与完整32层模型相比，删去8层后平均误差从2.11 m提升至≈2.17 m，decoder推理速度提升1.33×，且在所有能力类别中误差差异不显著；

**⚠️ 局限性**

研究仅限于单一ORION checkpoint、开环评估、未进行闭环验证或再训练，剪枝依据为冻结规划器兼容度，因而对真实车辆部署的适用性仍有限。

---

## 434. Geo-Spatial Concept Probing of Large Language Models: Abstraction, Compositionality, and Grounding

**arXiv ID:** 2608.07353 | [PDF](https://arxiv.org/pdf/2608.07353v1)

**作者:** Karim Radouane `[一作]` (University of Toulouse, IRIT), Lynda Tamine `[通讯]` (University of Toulouse, IRIT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了一个面向空间概念（方向、距离、拓扑）的概念中心化问答基准，并利用该基准对大语言模型（LLM）的抽象、组合性与接地能力进行系统探测；

**💡 创新点**

创新点在于：①从概念属性（抽象、组合、接地）出发设计专门的基准；②通过对问答性能与内部表示的双重探测，揭示LLM在概念处理上的具体瓶颈；③对不同LLM架构和规模进行横向比较，系统评估其概念理解差异；

**🔧 技术方法**

技术上使用了模板化问答生成、线性探测器（probe）评估隐藏层表示、构造组合问题与子问题的对应关系，以及基于余弦相似度与逻辑一致性度量的组合性评估；

**📊 数据集**

使用的公开数据集包括UK与US的行政区划信息（GraphDB、YAGO2GEO），生成了数十万条三元组问答样本；

**📈 对比分析**

通过将LLM在问答任务上的准确率、连贯性与探测器在不同层的分类准确率进行比较，发现大多数模型在标准随机拆分上表现优秀（>99%），但在区域OOD、Token级OOV与组合性OOV上，尤其是Mistral系列表现显著不及Qwen、LLaMA；整体准确率最高的模型（如LLaMA-3.1-8B）在多概念组合问题上的准确率可达70%，但一致性仍低于50%；

**⚠️ 局限性**

研究局限包括：①仅覆盖两块地理区域；②只考察了空间概念，未覆盖更复杂或跨领域概念；③探测器仅为线性分类器，可能低估了模型内部更深层次的概念结构。

---

## 435. Residual Algebra for Representation-Preserving Learning

**arXiv ID:** 2608.07349 | [PDF](https://arxiv.org/pdf/2608.07349v1)

**作者:** Yao Wu `[一作]` `[通讯]` (Westlake University), Yao Wu (Westlake University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出残差代数框架，构造Fold交互式字段与FPRC‑PQ分阶段残差校正模型，显著提升股票回测收益。

**💡 创新点**

创新点在于将残差视为带类型的对象，采用有序的“放松‑聚合‑闭合”算子序列，保留表示身份并在聚合点才抹除，形成可控的残差所有权与信息边界。

**🔧 技术方法**

使用特征离散化网格（Fold）、梯度提升树（XGBoost）做局部与共享残差校正，并引入反射性冥想的解析增益计算。

**📊 数据集**

在中国A股市场 2023‑2026 年共 3.67M 日级股票样本（约 844 个交易日、平均 4,380 股）上进行实验。

**📈 对比分析**

通过与直接 GBDT、池化字段、单阶段/双阶段残差校正等预注册对照组进行配对 21‑日区块自举检验，FPRC‑PQ 的净收益从 13.52% 提升至 19.10%，夏普比率从 1.42 提升至 2.09。

**⚠️ 局限性**

局限包括：仅在单一市场和单一资产类别验证；字段维度固定为二维；解析增益仅在耦合重新拟合路径下保证正交；未验证在实时交易中的鲁棒性与扩展性。

---

## 436. Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks

**arXiv ID:** 2608.07335 | [PDF](https://arxiv.org/pdf/2608.07335v1)

**作者:** Taha Shieenavaz `[一作]` (University of Padua), Loris Nanni `[通讯]` (University of Padua)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在无经验回放、无目标网络的Parallelized Q-Network (PQN) 框架下，系统性评估了八种受限参数的CNN编码器，进一步将Hadamax多重特征交互与最优Gamma结构融合，并在此基础上引入分布式、集成、优势分解等高级回归头，最终提出了统一的 Aftab 架构，并在 Atari-57 与 Procgen Hard 任务上进行大规模实验。

**💡 创新点**

创新点包括：① 在参数约束下通过深度递增的 Gamma 结构实现更高的抽象层级；② 将 Hadamax 的 Hadamard 乘法与最大池化巧妙结合，提升表示容量而不显著增加参数；③ 在完全无缓冲区的在线学习环境中成功整合分布式、集成与优势分解三大先进回归策略；④ 通过严格的统计检验与 IQM/Probability of Improvement 指标，首次在 buffer‑free 设置下证明这些结构可显著提升样本效率与泛化性能；⑤ 发布完整 Aftab 开源实现，为后续研究提供基准。

**🔧 技术方法**

使用技术包括：Parallelized Q‑Network (PQN)、Hadamax 编码、分布式值函数（Distributional RL）、集成探索（Deep Ensembles / Bootstrapped DQN）、优势分解（Dueling），以及 Layer Normalization、RAdam 优化器、λ‑returns、EnvPool 向量化环境、无权重衰减、GELU 激活等。

**📊 数据集**

数据集：标准 Atari‑57 57款游戏（随机、生活终止、sticky actions 等），以及 Procgen Hard 随机生成的 16 个游戏，用于评估跨分布泛化。

**📈 对比分析**

对比方法：使用 Interquartile Mean (IQM) 的 Human‑Normalized Score（HNS）以及 Probability of Improvement 进行统计比较；在 Atari‑57 上 Aftab 达到 IQM HNS 6.479（P=0.86 相较于 PQN baseline），显著优于基线 PQN（2.692）和 Gamma（5.343）等；在 Procgen Hard 上 IQM PNS 0.418 也高于 PQN 的 0.382，展示出更好的跨域泛化。所有比较均通过 Wilcoxon 符号秩检验并采用 Holm‑Bonferroni 校正确保显著性。

**⚠️ 局限性**

局限性：① 仅在离散动作、像素输入的 Atari/Procgen 任务上验证，未评估在连续控制或长期推理任务中的表现；② 所有实验统一使用相同的超参（学习率、λ、批量等），未针对不同结构进行细化调优；③ 无正则化（weight decay）虽保持稳定但缺乏理论收敛或 TD‑Jacobian 约束的正式证明；④ 结果对极端噪声环境或非视觉感知任务的可迁移性仍待验证。

---

## 437. Learning Fault-Tolerant Locomotion with Adaptive Gait Timing

**arXiv ID:** 2608.07328 | [PDF](https://arxiv.org/pdf/2608.07328v1)

**作者:** Giovanbattista Gravina `[一作]` (Italian Institute of Technology), Nikos Tsagarakis `[通讯]` (Italian Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种针对关节功率丧失的四足机器人容错行走控制方法，利用深度强化学习训练可自适应步频的控制策略。

**💡 创新点**

创新点在于：1) asymmetric actor‑critic 与潜在对齐损失相结合，使演员仅凭近似观测即可推断特权信息；2) 在动作空间加入可学习的步频参数，实现动态步调自适应；3) 通过观察历史长度与潜在对齐提升故障推断能力。

**🔧 技术方法**

采用PPO强化学习、异构actor‑critic网络、潜在对齐损失、可学习步频动作、模拟域随机化等技术。

**📊 数据集**

使用基于MuJoCo XLA的仿真数据，训练集为在多种斜坡/不规则地形（4/8/12 cm 步高）上随机发起关节功率故障的 Kyon 68 kg 四足机器人；实验集为实机平地测试。

**📈 对比分析**

与 oracle（全特权信息）、无潜在对齐、无历史等对照方案比较，五次训练平均奖励与 oracle 接近，仿真中命令跟踪误差低、存活时间高；在真实机器人上实现零射程转移，成功通过后期故障保持平稳步态。

**⚠️ 局限性**

局限在于：缺乏真实感知模块，仅在平地或已知地形下验证；对多关节同时故障或更极端负载的鲁棒性未知；步频调节仅在仿真与平地实验验证，复杂地形下性能待进一步评估。

---

## 438. Towards Assurance Closure in AI-Native Large-Scale Agile Software Development

**arXiv ID:** 2608.07317 | [PDF](https://arxiv.org/pdf/2608.07317v1)

**作者:** Ricardo Britto `[一作]` `[通讯]` (Ericsson), Ricardo Britto (Ericsson)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了 AI‑native 开发中的“assurance closure”概念，识别了实现可信委派所需克服的六大技术缺口，构建了面向这些缺口的六层次保障架构，并给出了四个具体研究问题，旨在实现对智能代理的可控、可逆授权。

**💡 创新点**

创新点在于：①将已有的形式化、测试、仿真、数字孪生、运行时保障等技术整合为一个连续、机器可操作的闭环；②提出了共享语义保障层，为多层次能力提供统一知识基础；③将“委派与监督”视为与保障状态同步的决策循环，实现基于证据的可逆授权；④通过六大缺口分析为 AI‑native R&D 提供系统化研究路线。

**🔧 技术方法**

主要技术包括：形式化方法、静态分析、单元/集成测试、模型检查、属性驱动测试、模糊测试、仿真与模拟、数字孪生、运行时监控、证据评估与决策支持等；这些技术在本文中被视为可组合的工具箱，支持“保证策略合成”和“证据执行织物”两大核心功能。

**📊 数据集**

本文未使用任何具体数据集；研究是以理论和架构为主的定位性工作，没有实验数据或案例数据。

**📈 对比分析**

未提供实验或性能比较；本文主要以概念设计和研究问题为导向，未进行系统实现或基准测试，因而没有可供对比的性能指标。

**⚠️ 局限性**

局限性包括：①缺乏实际实现和实证验证，无法证明架构可行性；②对复杂系统中证据可信度评估的细化仍待研究；③人机交互与监督机制的可用性未在实验中检验；④对不同规模、领域的可迁移性尚不清晰。

---

## 439. From Optimal Actions to World Models: Identifiability of Transition Kernels in Discounted MDPs

**arXiv ID:** 2608.07301 | [PDF](https://arxiv.org/pdf/2608.07301v1)

**作者:** Neal Batra `[一作]` `[通讯]` (Dovetail Research), Neal Batra (Dovetail Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

研究了在已知所有奖励下仅观察最优动作时能够恢复马尔可夫决策过程（MDP）转移概率的程度，提出了不同奖励形式（状态-动作、状态、转移依赖）下的可辨识性判定。

**💡 创新点**

提出了对三类奖励的完整可辨识性分类，并给出精确的等价变换公式（矩阵L变换），揭示了即使所有最优动作相同，仍可能存在连续无穷多种转移核；同时证明了转移依赖奖励在大多数情况下可以唯一确定转移核，指出单动作状态是唯一例外。

**🔧 技术方法**

主要使用马尔可夫决策过程理论、Bellman方程、线性代数（矩阵变换、可逆性）、凸几何与微分流形分析来构造等价类并计算维数。

**📊 数据集**

无实验数据，完全基于理论证明与符号计算。

**📈 对比分析**

无实验比较，研究通过数学证明给出了等价性条件与维数等定量结论。

**⚠️ 局限性**

局限包括：仅考虑完美观测的最优动作；未给出有限奖励集或近似观测下的可辨识性；对状态奖励的等价类描述不够直接；未探讨如何通过有限或自适应奖励集合来逼近完整等价类。

---

## 440. Foundation Models Adaptation for Multi-View Multi-modal Cardiac MRI Segmentation and Direct Ejection Fraction Estimation

**arXiv ID:** 2608.07291 | [PDF](https://arxiv.org/pdf/2608.07291v1)

**作者:** Sina Amirrajab `[一作]` (Maastricht University), Ali Yilmaz `[通讯]` (University Hospital Münster)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

对多视角、多模态心脏MRI进行解剖分割、纤维瘢痕分割，并直接用冻结的基础模型特征实现射血分数预测。

**💡 创新点**

创新点在于将多视角的分割专家模型与冻结的 CMRT Transformer 与 CMRCLIP 编码器通过注意力多实例学习融合，用任务专门化策略实现分割与功能预测的协同提升。

**🔧 技术方法**

采用 CineMA 的 fine‑tune 分割、交叉熵+Dice 损失、AdamW 优化、Attention‑MIL 回归、以及 CMRT Transformer 与 CMRCLIP 的特征融合。

**📊 数据集**

使用 CMR‑Multi 挑战官方数据集，涵盖 SAX、2CH、4CH 视角的 cine 与 LGE 序列，来自多中心多序列的多模态数据。

**📈 对比分析**

与从零开始训练的 nnU‑Net 进行对比；CineMA 在长轴视角和 LGE 任务上表现优异，SAX 视角与 nnU‑Net 仍更好；直接 EF 回归 MAE 4.96%，分割派生 EF 的 MAE 3.97% 更优；LGE scar 的 Dice 分别为 0.62–0.85，说明仍存在挑战。

**⚠️ 局限性**

局限在于 LGE scar 分割仍较困难，尤其在 SAX/2CH 视角；多视角与多模态的泛化能力受限；直接 EF 回归在准确性上略逊于分割派生 EF；基础模型的迁移效果受模态差异影响。

---

## 441. Gaze Behavior in Visual World Experiments Can be Modeled With Off-the-shelf Language-Vision Encoders

**arXiv ID:** 2608.07282 | [PDF](https://arxiv.org/pdf/2608.07282v1)

**作者:** Rahul Murali Shankar `[一作]`, Sebastian Padó `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用CLIP双编码器与Integrated Jacobians方法预测视觉世界实验中的注视行为。

**💡 创新点**

无需微调或生成架构，直接通过可解释性归因映射捕获预测性注视，并证明非生成模型可复现人类预测行为。

**🔧 技术方法**

CLIP双编码器、Integrated Jacobians归因、ReLU归一化、目标归属聚合、线性混合效应模型。

**📊 数据集**

18张图像的经典英语视觉世界实验数据集，句子形式为“The subject will verb the target object”。

**📈 对比分析**

通过线性混合效应模型对四种CLIP模型进行统计检验，结果显示在预目标区受限动词能显著提升对目标对象的归因，后续目标词出现时差异消失，证明模型能再现人类预测性注视，且模型规模/ImageNet准确度与预测性能呈正相关。

**⚠️ 局限性**

仅使用四个CLIP模型、单一实验数据，模型非增量且对更细粒度效应缺乏解释，归因粒度有限，缺乏对更广泛实验和模型的验证。

---

## 442. Why Knowing Both Hops Is Not Enough: Understanding Two-Hop Generalization in Language Models

**arXiv ID:** 2608.07261 | [PDF](https://arxiv.org/pdf/2608.07261v1)

**作者:** Zili Zhang `[一作]` (Xi'an Jiaotong University), Minnan Luo `[通讯]` (Xi'an Jiaotong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在受控符号环境中从零开始训练Transformer，系统性研究两跳推理的内部机制并发现第二跳在分布外时模型易失效。

**💡 创新点**

提出循环式训练（跨层参数共享）以对齐中间表示与输入格式，解决层间功能不匹配，从而显著提升分布外两跳推理性能。

**🔧 技术方法**

logit lens、实体表示一致性补丁、线性探测、注意力屏蔽、循环式训练、参数共享。

**📊 数据集**

符号生成数据集（实体/关系链），以及从维基百科抽取的真实自然语言两跳问答样本。

**📈 对比分析**

与标准GPT2/LLama模型在训练集、Test‑II、Test‑IO、Test‑OO、Test‑OI 等四种拆分上对比，循环式模型在 Test‑IO/OO 的准确率提升约 15–30%（符号场景）并保持在自然语言场景中接近 80% 的表现。

**⚠️ 局限性**

循环式训练仅使用单一循环迭代，训练流程相对简单；更深层次的循环、多样化训练调度或自适应机制等可能进一步提升效率与效果，但此论文未探究。

---

## 443. Stoicheia: Character-Level Masked Diffusion for Ancient Greek Textual Restoration, Parsing, and Metrical Scansion

**arXiv ID:** 2608.07249 | [PDF](https://arxiv.org/pdf/2608.07249v1)

**作者:** Eric Cullhed `[一作]` (Uppsala University), Albin Thörn Cleland `[通讯]` (Lund University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Stoicheia，一种 405M 参数的字符级 masked‑diffusion 编码器，用于古希腊语文本的恢复、分词、重分词、重标点和朗读等多任务。

**💡 创新点**

创新点在于：开放可重现的 361M 词语语料库、五层可独立 mask 的字符分层输入、可旋转分割保证每段文本至少有一个未见过的模型、并发布十一种检查点实现对训练文本的严格去污染。

**🔧 技术方法**

采用字符级 masked‑diffusion Transformer（1024 维，32 层，QK‑norm），四分之三块使用 256 字符窗口注意力，其余块全局注意，训练目标为自回归式连续 mask，使用 Beta 分布采样连续损坏模式，目标是重构原始字符及其装饰层。

**📊 数据集**

数据集为公开可再发布的 361M 词语语料库，来源于重 OCR、校正与机器翻译的拉丁文合成，分为 pristine、repaired、synthetic 三个质量层，全部开放且附带质量评估。

**📈 对比分析**

通过三项实验与随机初始化对照评估：①碑铭修复（CER 下降 5.6 点，Top‑1 73.3%），②形词句法标注（LAS 提升 12.9 点至 83.8%），③元音长度/韵律扫描（macron 平衡准确率提升 6.0 点，扫描 F1 提升 2.2 点），均显著优于当前基线系统。

**⚠️ 局限性**

局限性包括：大量语料为机器修复，可能含错误；去污染保证的是字符串层面，知识层面可能泄露；实验中的人工 lacuna 仅模拟 1–10 字符，未覆盖真实损坏；模型始终给出预测，没有置信度或拒绝机制；未对输入分层与损坏策略进行消融研究。

---

## 444. When GNNs Fail: Quantifying and Overcoming Temporal Correlation Volatility in Time Series

**arXiv ID:** 2608.07333 | [PDF](https://arxiv.org/pdf/2608.07333v1)

**作者:** Chen Shao `[一作]` (Karlsruhe Institute of Technology), Danai Koutra `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究多变量时间序列在图结构下的预测，量化了图拓扑的时变波动并提出 GLIDE GNN 方案；

**💡 创新点**

创新点在于提出 Temporal Correlation Volatility（TCV）度量来诊断图结构波动，并设计 D1（路径基邻域）与 D2（静态‑动态分解）两种机制的 GNN 层；

**🔧 技术方法**

主要技术包括图神经网络、卷积变形（dilated causal conv）、多阶图卷积、多项式聚合、基于核加权的动态图估计（图‑LASSO）以及对比的 Transformer 等基线；

**📊 数据集**

实验使用合成数据及真实能源/金融数据集：Electricity、Solar、Germany、France、ETTh1、Exchange‑Rate；

**📈 对比分析**

与 18 种基线（空间 GNN、谱 GNN、Transformer、序列模型等）对比，GLIDE 在 MAE/RMSE 上平均提升 45.6%/78.8%，在高 TCV 数据上显著优于所有竞争方法；

**⚠️ 局限性**

局限性包括假设时间扰动独立、TCV 仅基于 Pearson 线性相关，无法捕捉非线性依赖，未来需扩展到非线性相似度与更复杂的动态模型。

---

## 445. Is SwiGLU's Open Positive Tail Necessary? Evidence from Closed-Tail Gating with MemGLU

**arXiv ID:** 2608.07323 | [PDF](https://arxiv.org/pdf/2608.07323v1)

**作者:** Yuting Ge `[一作]` (City University of Hong Kong), Mingkai Nie `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的门控激活函数MemGLU（闭尾结构）并将其替换到标准的SwiGLU门控FFN中，使用配对训练方式在9M和30M规模的decoder‑only语言模型上进行对比实验。

**💡 创新点**

创新点在于从一阶忆阻器分支几何得到的闭尾门控函数MemGLU，并通过实验验证SwiGLU的开放正尾在当前规模下并非必要。

**🔧 技术方法**

主要技术包括Transformer FFN结构改造、RMS匹配门控尺度、配对种子训练、能量占用分析以及对已训练SwiGLU进行尾部抑制干预。

**📊 数据集**

使用的训练数据未在文中明确给出，推测为常见的大规模语言模型预训练语料（如C4/LAION等）。

**📈 对比分析**

与SwiGLU相比，MemGLU在9M规模下平均降低约0.11% NLL，在30M规模下平均提升约0.12% NLL，两者在训练轨迹和最终性能上相差不到0.1%。

**⚠️ 局限性**

局限性包括仅验证了9M和30M两种规模，缺乏更大模型的实验；MemGLU实现未使用融合核，导致训练吞吐下降10.46%并显著增加显存；实验缺少对不同数据集与训练配置的泛化评估。

---

## 446. Learning Long-Term Educational Investment Policies under Residential Sorting

**arXiv ID:** 2608.07295 | [PDF](https://arxiv.org/pdf/2608.07295v1)

**作者:** Honglei Guo `[一作]` (Zhejiang University), Yuhan Zhao `[通讯]` (Zhejiang University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个动态多智能体框架，将政府投资、住房市场、人口流动和学校质量联动，并通过强化学习优化多期预算分配。

**💡 创新点**

创新点在于将教育投资与住房市场、人口迁移和学生选校等相互反馈构建为完整的动态循环，并在此框架内引入效益–公平平衡的决策目标。

**🔧 技术方法**

采用基于凸优化的住房排序均衡、POMDP表述的政府决策以及Proximal Policy Optimization (PPO) 强化学习方法，构建多Agent仿真环境进行策略学习。

**📊 数据集**

使用中国教育面板调查（CEPS）数据中的收入、父母教育水平和儿童潜能等分布进行校准与仿真。

**📈 对比分析**

与等分配、按报名人数分配和补偿性分配三种基准策略对比，PPO策略在平均接入质量最高、Gini系数最低且人力资本接近最佳，表现出最优的效率与公平平衡。

**⚠️ 局限性**

局限性包括仅考虑公立学校且固定出勤边界，未纳入学生转学、私立或特许学校、教师配置和学区重划等政策；模型假设均衡与价格清算可能与真实市场偏差；仿真规模有限，难以完全覆盖大规模城市情况。

---

## 447. QFCQT: A Chaotically Gated Quantformer Framework for Volatile Time-Series Forecasting

**arXiv ID:** 2608.07363 | [PDF](https://arxiv.org/pdf/2608.07363v1)

**作者:** Junkai Lin `[一作]` (Beijing Normal-Hong Kong Baptist University), Raymond Lee `[通讯]` (Guangdong Provincial Key Lab of Interdisciplinary Research and Application for Data Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种名为QFCQT的时序预测框架，结合量化Transformer与多族李振荡器混合激活，实现对局部波动的自适应响应。

**💡 创新点**

创新点在于将多族Lee振荡器软叠加、最大时域池化与平滑-混沌门融合集成进Quantformer前馈块，显著提升对突发局部变化的感知能力。

**🔧 技术方法**

技术手段包括线性数值嵌入、Transformer自注意力、Lee振荡器激活、Max-over-Time池化、平滑-混沌门融合以及1×1卷积投影。

**📊 数据集**

使用了电力行业的ETTh_1、ETTh_2数据集和金融行业的A-share股价指数时序数据。

**📈 对比分析**

与Informer、LogTrans、LSTMa、HAT、COTN、TimesNet等基线对比，QFCQT在多时间步长上均实现了MSE/MAE下降，尤其在ETTh_2的24步长预测上提升高达43.9% MSE。

**⚠️ 局限性**

局限性包括对极端噪声或高缺失率数据的鲁棒性尚未充分验证，且混沌门的门控学习需要额外调参，未能在所有任务上显著优于COTN。

---

## 448. An End-to-End Agent Auditing Engine

**arXiv ID:** 2608.07346 | [PDF](https://arxiv.org/pdf/2608.07346v1)

**作者:** Haoning Wang `[一作]` (Shanghai Artificial Intelligence Laboratory), Na Zou `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 Agent Auditing Engine（A2E），一种端到端的评估引擎，用于统一基准执行、轨迹采集和多维度评估，以帮助系统化评测代理机架。

**💡 创新点**

创新点在于：① 引入 Agent Task Protocol（ATP）实现任务与机架解耦，② 基于 OpenTelemetry 的 Span 监控实现细粒度轨迹收集，③ 生命周期对齐的多维度评估框架与数据库驱动的可扩展性，使得评估可以在多模型、多机架、跨基准下持续迭代。

**🔧 技术方法**

核心技术包括 OpenTelemetry + OpenInference 的 Span 追踪、LLM-as-Judge 与规则评估相结合的评估器、数据库持久化、CLI 自动化跑通、以及多框架 SDK 适配器。

**📊 数据集**

使用 23 个公开基准（包含 4 个沙箱 benchmark），涵盖编码、会话、科研与计算机使用四大任务域，覆盖多难度与多类别场景。

**📈 对比分析**

评估方法：对每个基准与每个机架组合生成轨迹，统一采集多维度指标（计划、工具、答案、运行质量），实验表明不同模型-机架组合在成功率、token 费用、工具使用、诊断能力等方面差异显著，且不存在单一组合在所有任务上均最优。

**⚠️ 局限性**

局限性：评估聚焦于现有基准，缺少对实时交互或大规模分布式部署的评测；样本量有限导致对低频或长尾任务的分辨率受限；评估指标主要基于文本与工具调用，未覆盖更复杂的多模态交互。

---

## 449. Zero Gap Is Not Restoration: Stratified Per-Question Probability Evaluation and Step-wise Mitigation of Benchmark Contamination

**arXiv ID:** 2608.07341 | [PDF](https://arxiv.org/pdf/2608.07341v1)

**作者:** Ruijie Hou `[一作]` (Zhejiang University), Yingming Li `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了新的评估指标 SA-PPG 与一种新的泄漏抑制策略 RailCap，用以更准确地衡量和恢复受公开基准泄漏影响的语言模型能力。

**💡 创新点**

创新点在于：1) 将离散 0/1 读出替换为每题 solve 概率，并按 clean 模型 solve 概率分层聚合，消除过度/欠度抑制抵消和频率偏倚；2) RailCap 在解码时实时检测贪婪轨迹回退并以跑者上限抑制下一个 token，转变为逐步监督而非一次性预估。

**🔧 技术方法**

使用的技术包括：采样估计 solve 概率、概率差分、分层加权、贪婪轨迹索引、n‑gram 轨迹回退检测、runner‑up 抑制与硬禁令对比。

**📊 数据集**

实验基准为 GSM8K 与其改写版 PQ，评估对象为 Llama‑2‑7B、Gemma‑4‑E2B 与 Pythia‑12B 三大模型的公开版，采用 8‑shot CoT 提示。

**📈 对比分析**

与 Identity、TED、LNE‑blocking、Shortcut neuron patching 等策略在同一设置下对比，使用 SA‑PPG 作为评估度量；RailCap 在所有 6 个模型/泄漏域组合中均取得最低 SA‑PPG，说明其恢复效果最佳。

**⚠️ 局限性**

局限性包括：仅在文本算术/推理任务验证，无法直接推广到多模态或更复杂任务；n‑gram 阈值需要手动调优；对非贪婪轨迹泄漏的情况尚未覆盖。

---

## 450. Learning Nearest-Neighbor Maps from Adaptive Queries

**arXiv ID:** 2608.07352 | [PDF](https://arxiv.org/pdf/2608.07352v1)

**作者:** Hadley Black `[一作]` (CUNY), Geelon So `[通讯]` (ETH Zürich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在理论上研究了通过自适应最近邻查询重构隐藏点集H的最优查询复杂度，并将其推广到任意有限维范数空间，证明了最坏情况下的查询复杂度为Θ(nκ)，其中κ为该范数空间的kissing number；此外，在欧几里得球面上给出了改进的上界与维度约简技术，并对球体和锥体给出了指数下界；

**💡 创新点**

核心创新是将最近邻学习问题与kissing number联系起来，提出通用的Voronoi覆盖算法实现最优查询上界；并在欧氏球面上通过随机维度约简实现了从O(nκ)到O(n·min{κ,n,d})的提升，展示了球面与球体在查询复杂度上的显著差异；

**🔧 技术方法**

使用几何与组合方法：Voronoi细胞划分、kissing number覆盖论证、随机取向子空间的维度约简、贪心覆盖构造；同时借鉴了之前聚类算法的思想和Prabhu‑Woodruff的随机投影技术；

**📊 数据集**

无实际数据集，全部为理论分析与构造的硬实例；

**📈 对比分析**

未进行实验比较，只给出理论上限与下界；与先前仅在布尔超立方体和球面上已知的结果相比，本文提供了更一般的上界与更强的下界；

**⚠️ 局限性**

局限在于上界与下界均呈指数依赖于维度；仅适用于纯净的最近邻查询，不考虑噪声或近似；缺乏对实际数据分布的分析；对计算复杂度未给出具体实现评估。

---

## 451. From Forensics to Ecosystems: Rethinking Watermarks for Generative AI Oversight

**arXiv ID:** 2608.07337 | [PDF](https://arxiv.org/pdf/2608.07337v1)

**作者:** Daniel Susser `[一作]` (Cornell University), Gili Vidan `[通讯]` (Cornell University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文对数字水印在人工智能治理中的应用进行系统性探讨，并提出将其从传统的取证工具转向“生态系统指标”以监测合成内容的整体占比。

**💡 创新点**

创新点在于：①提出生态系统视角——用水印衡量整个平台或领域的合成内容渗透率；②强调在低熵文本等难以识别的媒介中仍能发挥作用；③论证生态系统方法比单一取证更易实现治理、能降低对个人内容的误判，并提出相应的政策、平台与研究者角色。

**🔧 技术方法**

使用的技术主要是统计水印（embedding 与 detection）、p‑value 统计检验、对生成模型的自带水印（如 Google SynthID、C2PA 等），以及在音频、图像、文本等多模态中的嵌入与识别方法。

**📊 数据集**

论文未给出具体实验数据集，主要基于理论分析与假设场景（音乐流媒体、学术预印本）来阐述方法效果。

**📈 对比分析**

对比方法：传统的取证模式（单个内容的精确识别）与生态系统模式（整体占比统计）。论文通过案例分析指出：生态系统模式对误检容忍度更高，能够在弱水印、低熵文本等情形下仍提供可操作的聚合指标；而取证模式在个体标注上更准确，但对治理的系统性支持有限。

**⚠️ 局限性**

局限性包括：水印易被移除或伪造、弱水印在低熵媒介下检测率低、可能产生隐私与偏见风险、需要模型提供方合作、无法完全解决个体内容的取证与责任追溯问题、在多源、未水印内容上仍有盲区。

---

## 452. Grammar Engineering Meets LLMs: Development of Cantonese and Irish ParGram Treebanks

**arXiv ID:** 2608.07283 | [PDF](https://arxiv.org/pdf/2608.07283v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 453. Same Attention, Different Truths: Put Logit-Lens over Visual Attention to Detect and Mitigate LVLM Object Hallucination

**arXiv ID:** 2608.07302 | [PDF](https://arxiv.org/pdf/2608.07302v1)

**作者:** Zichuan Wang `[一作]` (University of Chinese Academy of Sciences), Jing Dong `[通讯]` (Chinese Academy of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出一种训练‑free 的 Detect‑Mitigate 框架，能够实时检测并减轻视觉‑语言模型的物体错报。

**💡 创新点**

创新点在于结合 Logit‑Lens 语义一致性检查与两类针对性纠正（高注意力区域掩码和视觉证据增强解码），同时阐明视觉不确定性与上下文先验两种错报机制。

**🔧 技术方法**

采用 Logit‑Lens 进行中间层语义解码、注意力分布分析、掩码干预以及解码阶段注入视觉 logits 的技术。

**📊 数据集**

在 COCO‑2014 验证集上的 CHAIR 与 AMBER 两大物体错报基准数据集进行评估。

**📈 对比分析**

与 Uncertainty Score、InterConf、SVAR、VCD、OPERA、DeCo、Devils、PAI 等多种基线对比，实验表明在 CHAIR 和 AMBER 上显著降低错报率，同时保持或提升覆盖率，表现优异。

**⚠️ 局限性**

局限性包括掩码位置的手工设定、对多模态上下文解释的依赖以及在不同视觉编码器和更大规模模型上的泛化仍需进一步验证。

---

## 454. FUSE: Feature-Wise Unified Specialization with Cross-Column Exchange for Mixed-Type Tabular Flow Matching

**arXiv ID:** 2608.07294 | [PDF](https://arxiv.org/pdf/2608.07294v1)

**作者:** Suman Cha `[一作]` (Yonsei University), Hyunjoong Kim `[通讯]` (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为 FUSE 的混合类型表格数据生成架构，在变分流匹配框架下显式分离特征特定处理与跨列信息交互，以生成更逼真的混合型合成数据。

**💡 创新点**

创新点在于：①引入自适应混合处理模块，使不同特征能够根据共享的子网络构建自定义组合；②采用全局自注意力实现跨类型信息交流；③从理论上量化限制条件导致的过度风险，并给出 Wasserstein 生成误差的上界。

**🔧 技术方法**

技术手段包括：变分流匹配（VFM）、指数族变分流匹配（EF‑VFM）、自适应混合模块、联合注意力（Multi‑Head Self‑Attention）、端点预测损失以及 Wasserstein 误差界限的推导。

**📊 数据集**

实验使用了八个混合类型表格数据集：Adult、Default、Beijing、Shoppers、Magic、News、Diabetes 和 Fault。

**📈 对比分析**

与 CTGAN、TVAE、CoDi、TabDDPM、TabSyn、TabDiff、TabbyFlow 等七种主流生成方法在 Shape、Trend、C2ST、MLE 等指标上进行对比；FUSE 在大多数指标上排名第一或接近第一，整体表现优于或与最强基线相当。

**⚠️ 局限性**

局限性包括：在某些任务（如 Beijing、Magic、News 的 Trend 指标）略逊 TabbyFlow；模型结构相对复杂，计算成本较高；对极端高维或稀疏数据的鲁棒性尚未充分验证。

---

## 455. A foundation-model approach to pediatric headache classification from rs-fMRI

**arXiv ID:** 2608.07287 | [PDF](https://arxiv.org/pdf/2608.07287v1)

**作者:** Guilherme S. Imai Aldeia `[一作]` (Boston Children's Hospital), Scott Holmes `[通讯]` (Boston Children's Hospital)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

使用 rs‑fMRI 数据，通过 NeuroSTORM 基础模型编码得到的嵌入向量，结合浅层机器学习（逻辑回归）实现对儿童头痛与健康对照以及头痛亚型的分类，并与传统基于功能连接矩阵的特征进行对比。

**💡 创新点**

首次将大规模训练的 fMRI 基础模型嵌入用于儿童头痛分类，证明在样本受限的情况下该方法可显著优于传统 FC 特征；同时展示了基础模型对小样本临床任务的迁移潜力。

**🔧 技术方法**

技术包括 NeuroSTORM 预训练模型嵌入、逻辑回归 L1 正则化、留一交叉验证（LOO）+ 100 次自助采样、基线 FC 矩阵特征提取（Harvard–Oxford 枢点）及多种浅层分类器（DT、LR、GB、RF）。

**📊 数据集**

使用 189 条 rs‑fMRI 扫描，来自 110 名 8–22 岁个体，分为 45 名健康对照和 144 名头痛患者（慢性偏头痛 68 条，其他子型 76 条），数据来源于波士顿儿童医院及相关科室。

**📈 对比分析**

采用与 FC 矩阵特征同等的机器学习流程进行对比。NeuroSTORM 的二分类 AUROC 为 0.82、AUPRC 为 0.93；FC 方法 AUROC 为 0.67、AUPRC 为 0.85。多类分类中宏观 AUROC 为 0.69。说明嵌入方法在有限数据下显著优于传统方法。

**⚠️ 局限性**

局限性包括样本量相对较小、子型标签异质性、模型缺乏可解释性、仅使用 rs‑fMRI、未进行前瞻性验证及未对基础模型进行针对性微调。

---

## 456. An Explainable Physics-Informed Neural Frequency-Response Framework for Shunt-Parameter Identification in Semi-Active Piezoelectric Tuned Mass Dampers

**arXiv ID:** 2608.07255 | [PDF](https://arxiv.org/pdf/2608.07255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 457. Why Study Emergent Behavior When You Can Regulate It? Aligning Multi-Agent Systems with Reward Prediction

**arXiv ID:** 2608.07280 | [PDF](https://arxiv.org/pdf/2608.07280v1)

**作者:** Assaf Caftory `[一作]` (Reichman University), Doron Friedman `[通讯]` (Reichman University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出并实现了 Multi-Agent Reward Prediction (MARP)，通过偏好学习构建共享奖励模型，使去中心化多智能体能够基于 episode‑level 社会指标（如效率、平等、可持续性、和平）自动调整行为；在 Harvest Game（Commons Game）上验证该框架。

**💡 创新点**

创新点包括：①将全球、稀疏的社会目标通过偏好学习转化为局部、稠密的奖励信号；②设计了两种推断策略（Joint‑Episode 与 Local‑Trajectory），支持单一或组合社会目标；③实现了完全可迁移的学习机制，无需手工奖励或环境干预。

**🔧 技术方法**

技术方法包括：偏好‑based 奖励建模（Bradley–Terry 方案）、共享奖励网络、PPO/Independent PPO 策略优化、基于 episode‑level 比较的奖励模型训练、权重化交叉熵损失、奖励模型的两种输入方式（聚合 episode 与单 agent trajectory）。

**📊 数据集**

使用 Harvest Game（Commons Game）作为实验环境，测量效率（U）、平等（E）、可持续性（S）、和平（P）等指标。

**📈 对比分析**

与传统环境奖励直接使用 PPO 的基线对比，MARP 能成功逃离悲剧平衡，提升多种社会指标；在效率优化下两种 MARS 变体相当；在组合目标（如效率×平等、效率×和平）下仍能保持主目标性能，同时显著提升次目标。性能表现为：效率与平等均高于基线，且保持或提升了整体社会福利。

**⚠️ 局限性**

局限性：①仅依赖 episode‑level 偏好，缺乏时间分辨率，难以为稀有或短期行为分配信用；②对预定义的社会指标敏感，无法处理模糊或演变的目标；③实验仅在单一环境（Harvest Game）验证，缺乏对异质代理、非确定性环境的泛化证明；④缺乏理论保证，奖励模型可能产生意外偏差；⑤局部轨迹推断中标签噪声较大，可能导致学习不稳定。

---

## 458. TOFD: Target-Oriented Feature Decoupling against Poisoning Attacks in Split Federated Learning

**arXiv ID:** 2608.07274 | [PDF](https://arxiv.org/pdf/2608.07274v1)

**作者:** Yuhan Xie `[一作]` (Shanghai University of Finance and Economics), Chen Lyu `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在分裂联邦学习（Split Federated Learning）中设计并实现了 TOFD（Target‑Oriented Feature Decoupling）防御框架，联合进行早期目标检测、精细化数据清洗和残余攻击消除，从而提升模型对多种投毒攻击的鲁棒性。

**💡 创新点**

创新点包括：①基于类级安全区域与边缘扰动（Margin Perturbation）构建自适应检测阈值；②利用分布一致性验证（Distribution Consistency Score）精准区分非IID噪声与恶意样本；③引入对抗引导模型和 KL 散度解耦损失，消除被清洗后残留的攻击特征。

**🔧 技术方法**

技术栈涵盖：类级安全区构造、边缘扰动与 min–max 归一化、分布一致性验证、可插拔的对抗引导模型、KL 散度解耦损失、交叉熵训练、指数滑动平均（EMA）与增量学习。

**📊 数据集**

实验使用五个公开图像分类数据集：MNIST、Fashion‑MNIST、HAM10k、CIFAR‑10 与 CIFAR‑100，结合 DenseNet121、ResNet‑18 与 ResNet‑50 三种主干网络进行评估。

**📈 对比分析**

与 FedAvg、Trim‑Mean、Median、SparseFed、Krum、Bulyan、FLTrust、DnC、ShieldFL、PRFL、FAVD、HealSplit 等十余种现有防御方案对比，TOFD 在单攻击场景下保持 92%+ 准确率，在多攻击场景下保持 83%+，平均 Poisoning Impact 低于 5%，MSDR 高于 90%，并且计算与通信开销与现有方法相近或更低。

**⚠️ 局限性**

局限性：①对阈值、λ、β 等超参数敏感，需要经验或自动调优；②对极端自适应攻击（如故意逼近阈值）仍有一定泄漏风险；③在非图像任务或高维隐层表示中，需要进一步验证对抗引导模型的有效性与可扩展性。

---

## 459. CANIS: Generation-Assisted 3D Canonicalization via an Image-Semantic Bridge

**arXiv ID:** 2608.07256 | [PDF](https://arxiv.org/pdf/2608.07256v1)

**作者:** Kendong Liu `[一作]` (City University of Hong Kong), Junhui Hou `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为 CANIS 的框架，利用冻结的朝向对齐图像‑到‑3D 生成模型生成与输入几何相匹配的正向化代理，并通过图像深度投影构建语义锚点来实现无类别依赖、无需专门训练的 3D 对象正向化。

**💡 创新点**

核心创新包括：①将生成模型的语义先验与生成代理相结合，构建实例特定的正向化参考；②利用图像‑深度投影和交叉注意力提取的语义锚点，显式约束 3D 对应关系，避免仅靠几何匹配导致的前后/上下混乱；③在生成代理时加入形状引导和信息视角选择，显著降低代理与输入之间的几何差异。

**🔧 技术方法**

技术实现主要涉及：图像到 3D 的 TRELLIS‑OA 生成模型、稀疏结构（SS）采样与 SLAT 结构采样、交叉注意力收集语义锚点、FPFH/FCGF 地面描述子匹配、局部锚点约束的对应搜索与 SVD 刚性变换估计，以及信息视角评分与形状引导策略。

**📊 数据集**

实验使用 Toys4K、Objaverse‑OA 与 OmniObject3D 三个 3D 形状数据集进行正向化评估，并在 ShapeNetPart、ModelNet40、TOSCA 动物子集上验证了分类、分割和密集对应等下游任务的提升。

**📈 对比分析**

在 24 个 Toys4K 类别的 IC/CC 指标上，CANIS 的平均值分别为 0.052/0.083，明显优于最强基线 CaCa 的 0.064/0.098；在 12 个 Objaverse‑OA 类别中平均为 0.032/0.086；在下游任务中，CANIS 的预处理几乎恢复了原始数据上的性能（分类 OA>91%，分割 IoU>85%，对应 PCK@0.10>99%）。

**⚠️ 局限性**

主要局限包括：在极端遮挡、严重缺失或近似对称结构下可能产生错误的语义锚点导致方向错误；代理生成对细节的准确性有限，过度形状引导可能抑制多样性；整个 pipeline 需要约 22 秒/对象，显著高于纯几何方法的计算成本。

---

## 460. TEMPO: Semantic-Action Decoupled RL Post-Training for Vision-Language-Action Models

**arXiv ID:** 2608.07314 | [PDF](https://arxiv.org/pdf/2608.07314v1)

**作者:** Ziheng Liu `[一作]` (Zhejiang Gongshang University), Quantao Yang `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对预训练的视觉‑语言‑动作模型进行在线强化学习后训练，冻结视觉‑语言主干，只对语义投影层和动作专家分别使用独立的 TD3 循环，并在两者之间设定不同的更新频率，以实现长链式指令任务的性能提升。

**💡 创新点**

创新点在于将 RL 更新拆分为语义层与动作层两条独立通道，并通过两时刻尺度的更新频率耦合（语义层低频更新、动作层高频更新）来抑制语义漂移、提升控制学习效率，这是此前 RL 后训练方法未曾尝试的模块级解耦策略。

**🔧 技术方法**

采用基于 FLOWER 的 VLA 架构，冻结视觉‑语言主干；对语义投影层与动作专家各自使用 TD3 强化学习框架，配合经验回放、双网络评估与目标网络；通过设定 ρ=f_a/f_s 的频率比来控制两者的学习步长。

**📊 数据集**

主要数据集为 CALVIN ABC→D 长链指令基准（5 任务链，A/B/C 训练、D 评估），以及在实际机器人平台上收集的两项多阶段操控任务的 120 条人类演示数据。

**📈 对比分析**

与多种基线（RT‑1、GR‑1、π_0、π_0.5、UNIVLA、DeFI、FLOWER、FLOWER‑RL）以及官方 FLOWER 结果进行比较；在 CALVIN 上 SR5 提升至 81.7%（比 FLOWER 高 3.9%），平均完成任务数 4.59（比 FLOWER 高 0.10）；在两项真实任务中也获得更高的累计奖励和更稳健的执行表现。

**⚠️ 局限性**

局限性主要在于仅使用稀疏的指令完成奖励，缺乏中间层级反馈，可能限制信用分配与样本效率；此外，对语义层与动作层的固定比例可能不适用于所有任务，需要进一步自适应或多模态奖励设计。

---

## 461. Improved Quantum Algorithms for Subset Sum and $k$-SUM

**arXiv ID:** 2608.07309 | [PDF](https://arxiv.org/pdf/2608.07309v1)

**作者:** Nikolai Chukhin `[一作]` (JetBrains Research), Ivan Mihajlin `[通讯]` (JetBrains Research)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种新的量子算法，用于解决最坏情况的 k‑SUM 问题，并以此改进了 Subset Sum 和 Pigeonhole Modular Equal Subset Sum 的最优量子时间复杂度。

**💡 创新点**

创新点在于：① 通过一次四块分解和随机质数的模筛选，构造出更小的搜索空间；② 结合量子 walk、量子搜索和 Claw Finding，显著降低了搜索代价；③ 对 k 取模 7 的情况（k≡3 或 6 mod 7）进一步优化，得到 Ψ_k = Φ_k - 1/9 或 1/18 的加速。

**🔧 技术方法**

主要技术：量子 walk（Johnson 图上的随机 walk）、量子搜索、Claw Finding、随机质数模筛、分块求和（block reduction）、量子随机访问内存（QRAQM）。

**📊 数据集**

本文不涉及实验数据集，所有结果均为理论分析与上界证明。

**📈 对比分析**

与之前的最佳量子算法（Tani 的 O(n^{k/3})）相比，本文在所有 k>3 时实现了更低的指数，尤其在 k≡3,6 (mod 7) 时能进一步缩短至 2k/7+O(1)。在 Subset Sum 方面，将已知的 O^*(2^{n/3}) 上界降低到 O^*(2^{2n/7})。

**⚠️ 局限性**

局限性包括：① 对 k 的改进仅在模 7 为 3 或 6 时显著；② 算法实现复杂，需要大量量子内存与复杂的子程序；③ 结果仍为上界，尚未给出对应的下界或在实际量子硬件上的可行性评估。

---

## 462. EliSeg: Verified Target Construction for Report-Grounded Abnormality Segmentation

**arXiv ID:** 2608.07299 | [PDF](https://arxiv.org/pdf/2608.07299v1)

**作者:** Chengyi Peng `[一作]` (Zhejiang University), Yankai Jiang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种报告驱动的异常分割方法 EliSeg，能够在没有任何预定义目标或空间提示的情况下，直接从未过滤的胸片报告中识别并分割可视异常。

**💡 创新点**

创新点在于将目标构造与掩膜生成融合为 propose–verify–revise 三步框架，利用独立文本验证器纠正目标资格和数量，并在必要时重新执行共享的语义演员，以消除不合格或缺失的目标。

**🔧 技术方法**

技术方案包括基于 ROSALIA 的语义演员（Actor）实现结构化控制序列与掩膜解码，冻结的 Qwen2.5‑VL‑7B 文本验证器（Verifier）进行目标资格判定，和基于一致性门的修正模块（Revision）在判定不一致时重执行演员。

**📊 数据集**

实验主要使用 MIMIC‑CXR‑ILS 数据集（含 1,008 条报告与对应分割标注），并在无报告标签的 CheXlocalize 数据集上验证演员的跨数据集零样本迁移能力。

**📈 对比分析**

与传统基于提示或分离提取-分割管线的基线相比，EliSeg 在报告驱动下的 IoU、Dice、NSD 等指标上均领先，且在抑制无效目标的假分割率（FSR）上表现更好，尤其在每种异常类别上均取得最高分。

**⚠️ 局限性**

局限性包括仅覆盖七种胸片异常且仅输出类别级掩膜，未处理双侧或多发实例；固定目标槽上限 K_max=3 可能截断罕见多目标句子；验证器仅检查动作和数量，可能漏掉语义错误；跨机构报告-图像-掩膜对齐数据缺乏。

---

## 463. Linear-Time Verification of Rings and Fields

**arXiv ID:** 2608.07272 | [PDF](https://arxiv.org/pdf/2608.07272v1)

**作者:** Youlong Ding `[一作]` `[通讯]` (Hebrew University of Jerusalem), Youlong Ding (Hebrew University of Jerusalem)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了一种确定性 O(n²) 时间算法，能在给定加法与乘法 Cayley 表的前提下判断一个有限集合是否构成环或域。

**💡 创新点**

核心创新在于：① 通过“先验证分配律再检查乘法结合律”与“仅用生成元的有限检验”实现了从随机/CFSG 依赖到完全确定性和线性时间的突破；② 设计了“伪造乘法表”与“增量计算”两种技术，完全抛弃了对有限单纯群分类（CFSG）的需求。

**🔧 技术方法**

采用的主要技术包括：基于循环分解的 Abelian 群同构构造、混合进制计数（odometer）实现的增量运算、利用生成元构造乘法表的“伪造”方法，以及对分配律、结合律、可逆性等验证的线性时间实现。

**📊 数据集**

无实验数据集，所有结果均为理论算法分析与证明；输入仅为 n×n 的 Cayley 表，适用于任意大小 n。

**📈 对比分析**

与之前的随机 O(n² log(1/δ)) 或 CFSG 依赖的 O(n²) 方案相比，新算法在时间上保持最优 O(n²)，且完全确定性，显著简化了实现与理论基础。

**⚠️ 局限性**

局限性：仅针对完全给定的二元运算表（有限集合）；算法不直接扩展到无限结构、非显式表或多元运算；实现仍需 O(n²) 内存，且在最坏情况下常数因子可能较大。

---

## 464. Incidental Visualizations: Augmented Reality as a Medium for Contextual Information

**arXiv ID:** 2608.07271 | [PDF](https://arxiv.org/pdf/2608.07271v1)

**作者:** Matilde Heitor `[一作]` (Instituto Superior Técnico University of Lisbon), Daniel Gonçalves `[通讯]` (Instituto Superior Técnico University of Lisbon)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在AR环境下，评估“偶发可视化”(Incidental Visualizations, IVs)与传统的常驻、定时可视化及无可视化方案在两种逻辑游戏（数独与四连棋）中的信息传递效果、干扰程度、认知负荷与任务表现。

**💡 创新点**

①提出并验证了IVs概念，将信息在任务进行时自动、简短地呈现；②构建了基于上下文触发、时间短、无需交互的可视化框架；③证明IVs在保持信息准确性的同时，能显著降低对主任务的干扰。

**🔧 技术方法**

使用Meta Quest 3头显（AR pass‑through模式）、Unity3D渲染、Python后端控制游戏逻辑与可视化；可视化实现为世界固定的柱状图（四连棋）和热力图（数独）；触发逻辑基于游戏进度与状态；记录响应时长、准确率、NASA‑TLX负荷与主任务完成度。

**📊 数据集**

数据来源为自制的任务数据：四套中等难度数独谜题（共38条线索）和数十局四连棋（使用深度为3的Minimax + 随机扰动对手）。并收集30名工程专业年轻参与者的行为与问卷数据。

**📈 对比分析**

比较方法：非参数Friedman检验+Bonferroni校正后对比；指标包括（1）补充信息的准确率（含时间加权与模糊度修正）；（2）主任务干扰程度（Likert）与主观工作量（NASA‑TLX）；（3）任务完成度（数独完成率、四连棋得分）。结果显示：IVs和常驻可视化在准确率上均显著优于无可视化；IVs与常驻可视化在干扰与工作量上相当，均低于定时可视化；所有方案对主任务性能无显著负面影响。

**⚠️ 局限性**

局限性：样本量有限且样本同质（工程专业18–24岁），缺乏眼动追踪以细化注意力分配；只测试了两种简单游戏，未检验在更复杂或长期使用场景下的可行性；可视化形式单一，未探讨不同编码对认知负荷的影响。

---

## 465. Recipes for Creativity: Iterative Generation and Evaluation in Large Language Models

**arXiv ID:** 2608.07243 | [PDF](https://arxiv.org/pdf/2608.07243v1)

**作者:** Rens Anderson `[一作]` (Leiden University), Amirhossein Zohrehvand `[通讯]` (Leiden University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将 FunSearch 演化式搜索算法应用于 Pillsbury Bake-Off 料理配方生成，并通过 TTCT 维度的 LLM 评估对生成配方的创意性进行量化。

**💡 创新点**

①将 FunSearch 从面向客观任务的搜索迁移到主观创意域；②通过实验揭示评估者设计（内循环评判模型）对创意结果的决定性影响，优于单纯增加搜索迭代或采样温度。

**🔧 技术方法**

使用 Meta-17B 作为生成模型，Meta-8B 或 Meta-17B 作为内循环评判器；采用 TTCT 维度（流畅性、灵活性、原创性、阐述性）对生成结果进行 LLM 评估；实验设计包含迭代次数、采样温度与评判模型规模的因子。

**📊 数据集**

以 2024 年 Pillsbury Bake-Off 参赛配方（获奖配方 + 29 条随机参赛配方）为人类基准集，用其 recipe 组件与生成配方进行比较；数据集仅包含配方文本及其故事。

**📈 对比分析**

实验发现：迭代搜索可获得与人类基准相当的 TTCT 创意分数，但多次迭代并未产生显著提升；小型评判模型（8B）在大多数维度上得分更高；温度变化对原创性以外的维度影响有限；整体性能表明评估者设计是最关键的变量。

**⚠️ 局限性**

局限性：评估完全基于 LLM，未包含人类评价，评判模型与最终评估者存在重叠，导致结果与真实人类创意判断不完全一致；人类基准仅包含配方，缺乏故事评价；未探索评估者多样性和创作过程轨迹的更细粒度分析。

---

## 466. Interaction Creates Dynamical AI Behavior Absent in Isolation

**arXiv ID:** 2608.07457 | [PDF](https://arxiv.org/pdf/2608.07457v1)

**作者:** Bella Xinrui Li `[一作]` (George Washington University), Neil F Johnson `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了两台参数相同、仅通过文本交互的 AI（GPT‑2）在单向与双向对话中的行为变化，发现单向信息浴会让从属 AI 产生与自身或上级都不同的新动态行为；

**💡 创新点**

首次揭示 AI‑AI 单向交互可产生“异星行为”，即从属 AI 在接收信息浴后进入其自身或上级都不具备的状态，并用简化的三态动力学理论解释了信息顺序与驱动强度对行为的决定作用；

**🔧 技术方法**

采用 GPT‑2 生成模型、基于 Boltzmann 分布的采样、内部生成的 200 回合对话、三态（失败、代码、其他）标签化、统计显著性检验，以及基于驱动强度的动力学模型；

**📊 数据集**

使用 GPT‑2 自身生成的文本序列（包括预录信息流和即时交互文本），无外部公开数据集，所有实验数据均来自内部对话循环；

**📈 对比分析**

通过对比无交互、单向 1→2、2→1 以及双向两 AI 的行为差异（D_1‑D_2、f_0 比例、失败率、熵、切换概率等），发现单向交互显著提升从属 AI 的 f_0 率并降低失败率，双向交互导致两 AI 收敛至相同的“异星”状态；

**⚠️ 局限性**

实验仅限于同参数 GPT‑2 轻量级模型，规模较小（200 回合），仅考虑文本交互，未涉及视觉或多模态输入；三态动力学模型过于简化，无法捕捉更复杂的交互机制，缺乏外部真实世界任务的验证。

---

## 467. PsychoAgent: An Affect-Sensitive Cognitive Architecture for Conflict-Aware Memory in LLM Agents

**arXiv ID:** 2608.07438 | [PDF](https://arxiv.org/pdf/2608.07438v1)

**作者:** Mohammad Amanlou `[一作]` (University of Tehran), Abdol-Hossein Vahabie `[通讯]` (University of Tehran)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了PsychoAgent，一种将事实记忆与情感记忆分离并通过冲突感知执行控制器融合的LLM认知架构；

**💡 创新点**

创新点在于引入情感显著性重排序机制，使得情感重要但语义不完全匹配的记忆能够在检索后被优先纳入上下文；

**🔧 技术方法**

实现技术包括基于语义相似度的检索、情感显著性评分、冲突感知的执行控制器以及离线重组合成临时“梦境”记忆；

**📊 数据集**

实验使用了手工构造的3个冲突情境（家庭经济冲突、职场批评与创意被挪用、友情背叛），并为每个情境提供了15条事实记忆和36条情感记忆；

**📈 对比分析**

通过与只按语义检索的语义‑情感并置版和单存储RAG基线的对比，结果显示PsychoAgent在关键情感记忆检索率上从0.5/0.667提升至0.933，且人类评测总体保持相近；

**⚠️ 局限性**

局限性包括场景数量有限、评测表现接近上限导致差异难以显著、缺乏多轮对话或多模型的广泛验证，以及该架构并未提供神经生物学上的对应证据。

---

## 468. Hands-Off or Hands-On? Variation in Area Chair Practices and Implications for AI Support

**arXiv ID:** 2608.07425 | [PDF](https://arxiv.org/pdf/2608.07425v1)

**作者:** Ines Arous `[一作]` (York University), Andrew McCallum `[通讯]` (University Of Massachusetts Amherst)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过27名ICLR期刊Area Chair（AC）的半结构化访谈与设计探针，系统研究了AC在稿件审阅中的挑战、策略及其对AI工具的期望与担忧。

**💡 创新点**

提出了三大设计启示：1）为多样化AC实践提供个性化AI支持；2）开发能充当审稿讨论调解者的会话式代理；3）遵循以人为中心的AI原则，保障人类主导与可解释性。

**🔧 技术方法**

主要采用定性方法——访谈记录、线框设计探针以及反思性主题分析；并使用手工编码、主题归纳与讨论验证。

**📊 数据集**

数据来源于ICLR会议的AC群体，共27名受访者，涵盖不同学术背景与经验水平。

**📈 对比分析**

该研究不涉及算法性能评估或与现有工具的对比，而是通过访谈反馈与设计讨论对不同AI辅助概念的可行性与接受度进行评估。

**⚠️ 局限性**

局限性包括样本集中于ICLR，受访者多为AI研究者，结果对其他领域或更广泛AC群体的适用性需进一步验证；未对实际AI工具进行原型测试与长期效果评估。

---

## 469. Efficient Discrete Position Design for Movable Antenna Systems: Low Complexity and Robustness

**arXiv ID:** 2608.07413 | [PDF](https://arxiv.org/pdf/2608.07413v1)

**作者:** Haonan Wang `[一作]` (City University of Hong Kong), Ying-Jun Angela Zhang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种基于子模优化的可移动天线离散位置选取算法（DCSPS 及其鲁棒版 R‑DCSPS），以在多用户 MIMO 上行链路中最大化互信息。

**💡 创新点**

创新点在于将离散移动天线定位问题映射为满足 2‑系统约束的单调子模最大化问题，给出 1/3 的近似保证，并利用 Jensen 不等式构造鲁棒 surrogate，使得算法在精确与不确定场景下均能获得 90%+ 最优互信息且复杂度大幅降低。

**🔧 技术方法**

主要技术包括子模优化理论、增量收益分析、距离约束贪婪搜索、Jensen 不等式实现鲁棒目标、矩阵逆/Woodbury 公式、虚拟通道构造等。

**📊 数据集**

使用 Monte‑Carlo 生成的多径信道仿真（L=20、K=2、N_U=2、N_BS=4~64 等参数）进行实验，没有使用公开数据集。

**📈 对比分析**

与 Brute‑Force、Branch‑and‑Bound、等距、随机、Channel‑Gain 贪婪等方案对比，DCSPS/R‑DCSPS 在 MI 上接近 BnB 最优（≈90% 以上），且运行时间比 BnB 快 30–60 倍，显著提升了实际可行性。

**⚠️ 局限性**

局限性包括：仅针对一维线性阵列；离散网格有限；二维/平面位置约束及更高维度阵列的扩展仍待研究；在大规模阵列或用户数极高时，O(N_S N_BS⁴) 的多项式复杂度仍可能成为瓶颈。

---

## 470. GeoBenchLLM: A Comprehensive Benchmark for Evaluating LLMs on Geo-Related Tasks

**arXiv ID:** 2608.07411 | [PDF](https://arxiv.org/pdf/2608.07411v1)

**作者:** Rodrigo Ferreira Rodrigues `[一作]` (University of Toulouse), Lynda Tamine `[通讯]` (University of Toulouse)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 GeoBenchLLM 的综合基准，用于评估大型语言模型在地理相关任务中的能力，覆盖知识、推理和应用三个认知层面，包含 12 个公开数据集、17 个子数据集和 8 种任务。

**💡 创新点**

① 统一的大规模地理基准；② 为开放式生成任务引入了多种自定义指标；③ 通过“思考”机制（chain‑of‑thought）对同一模型进行对比，揭示模型规模与思考对不同认知层面的影响。

**🔧 技术方法**

使用 Qwen3 系列（0.6B、1.7B、8B）与 GPT‑OSS 系列（20B、120B）两大类 LLM，实验中开启/关闭思考模式；并开发了 7 种专门指标（坐标准确率、精确率/召回率、路径可行/成功/最优率等）。

**📊 数据集**

包含 12 个公开数据集：GeoQuestions1089、GeoQuery、Ms‑Marco、GeoSQA、GKMC、SpatialEvalLLM、SpartUN、StepGame、TourismQA、NY‑POI、GridRoute、PPNL（单/多）等，分别涵盖事实查询、情景推理、空间推理、POI 推荐、路径规划等任务。

**📈 对比分析**

通过在 17 个子数据集上计算各自主指标，对 Qwen3 与 GPT‑OSS 的性能进行量化对比。结果显示：在知识层面，GPT‑OSS 规模更大更占优；在推理和应用层面，8B Qwen3 开启思考后可逼近甚至超越大模型；思考模式显著提升性能，尤其在需要多步推理的任务中，差距从 24% 降到 13%/18%。

**⚠️ 局限性**

局限性：① 仅评估模型的内在知识，未考虑外部工具或数据库调用；② 数据集仍无法覆盖所有地理任务（如时间序列预测、动态空间变化等）；③ 部分指标（如 Bleu‑1、BERT‑Score）对文本质量敏感，可能导致评估偏差；④ 结果受思考预算限制影响，未探索更大预算或更细粒度的思考方式。

---

## 471. UniJEPA: A Unified Joint-Embedding Predictive Architecture for Task-Agnostic Visual World Modeling

**arXiv ID:** 2608.07409 | [PDF](https://arxiv.org/pdf/2608.07409v1)

**作者:** An Lanji `[一作]` (University of Electronic Science and Technology of China), Yu Tian `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 UniJEPA，一种统一的 Joint‑Embedding Predictive Architecture，能够在同一潜在空间同时完成图像级的光度变换预测与视频级的时序预测，并支持零样本目标规划。

**💡 创新点**

创新点包括：① 统一了原本分离的图像与视频 JEPAs，使用单一的预测器和编码器；② 通过一个高斯正则化项实现抗崩塌保证，无需 EMA、stop‑gradient 或预训练编码器；③ 通过调节光度与时序损失比例实现可控的无关/等变抽象；④ 在单一模型上完成离线动作规划，显著提升规划速度。

**🔧 技术方法**

核心技术为：Vision Transformer 编码器、共享预测器（可接受光度变换或动作信息）、平方误差预测损失、Gaussian 正则化（约束潜在分布为球面高斯），以及后期基于离线轨迹的动作条件微调。

**📊 数据集**

使用的数据集包括 ImageNet‑1k（图像下线性探测和微调）、Something‑Something‑v2 与 Epic‑Kitchens‑100（视频理解与动作预测）以及多种离线控制套件（迷宫、推箱子、多粒子）进行规划实验。

**📈 对比分析**

与 I‑JEPA、IWM、V‑JEPA‑2、DINO‑WM、LeWorldModel、SimCLR、DINOv2 等基线对比，UniJEPA 在 ImageNet 线性探测上达 74.9%（接近 DINOv2）、在 SSv2 任务上 78.1% 领先 V‑JEPA‑2、在离线规划任务上 75.8% 成功率并比生成式世界模型快 44 倍，且仅需一个损失超参。

**⚠️ 局限性**

局限性包括：共享预测器可能限制对长时程或多模态任务的建模；高斯正则化在极大规模模型下可能抑制表达能力；当前模型不支持语言或文本指令，需要进一步扩展到多模态统一空间。

---

## 472. LYRA: Label-Free Structural Synchronization and Resource Allocation for UAV Edge Networks

**arXiv ID:** 2608.07392 | [PDF](https://arxiv.org/pdf/2608.07392v1)

**作者:** Feng He `[一作]` (University of Bologna), Schahram Dustdar `[通讯]` (TU Wien)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出LYRA框架，实现UAV边缘视觉模型的自适应更新调度与资源分配，使用无标签的语义漂移率实时评估模型性能。

**💡 创新点**

创新点包括：①前后端敏感度调度SASS实现分层结构同步；②无标签语义差异率OSDR作为在线性能代理；③Lyapunov导向的强化学习通过动作空间降维与虚拟队列实现长期能源与语义风险约束。

**🔧 技术方法**

采用Lyapunov优化、PPO强化学习、混合动作空间降维、闭式物理资源分配、层次化网络同步、无线信道建模等技术。

**📊 数据集**

使用CIFAR-10-C（15种视觉腐败、5级）与ResNet-18模型，以及CRAWDAD城市轨迹生成的A2G通道数据。

**📈 对比分析**

与周期更新、DDPG、HR-PPO、无更新、阈值触发等基线对比，LYRA在语义恢复效率、风险积压、能耗和传输量上均优于最佳基线，风险回压降低约33%，语义恢复率提升，能源预算得到满足。

**⚠️ 局限性**

局限性：需离线预训练oracle，OSDR对oracle误差敏感；只针对层次化网络设计，其他模型需额外适配；在极端或未知的环境漂移下可能误触或滞后；实验仅在CIFAR-10-C/ResNet-18上验证，缺乏真实 UAV 视觉任务的实测评估。

---

## 473. Online Metric TSP: Beyond the $\sqrt{n}$ Barrier

**arXiv ID:** 2608.07369 | [PDF](https://arxiv.org/pdf/2608.07369v1)

**作者:** Yossi Azar `[一作]` (Tel-Aviv University), Or Vardi `[通讯]` (Tel-Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在线度量TSP问题，提出了一种在空间仅略大于n（m=(1+ε)n）的算法，能够实现 O(log³n/ε) 的竞争比。

**💡 创新点**

创新点在于引入了“度量球覆盖树”这一新的数据结构，并提出了自底向上标记与插入策略，使得即使空间只有略超 n，也能把竞争比从 Θ(√n) 大幅降低到多项式对数级别；同时给出了相应的下界证明。

**🔧 技术方法**

主要技术包括：1) 动态递归覆盖（以几何递减半径的球覆盖度量空间）；2) 元素自底向上标记的二叉树插入策略；3) 对树内部与树间成本的层次分析；4) 通过多阶段对抗构造证明下界。

**📊 数据集**

无需实验数据集，研究纯粹基于理论分析与证明。

**📈 对比分析**

与先前最优的 O(log n)（当空间为 2ⁿ）以及 O(√n)（当空间为 n）结果相比，本算法在空间略大于 n 的情况下实现了更优的竞争比，证明了竞争比与空间使用之间的权衡。

**⚠️ 局限性**

局限性包括：1) 仅对确定性算法给出下界，随机化算法的情况仍未解决；2) 若要实现 O(1) 竞争比，需要空间至少达到 n^{1+γ}，该阈值的精确位置尚不清楚；3) 对于更大空间（如 2ⁿ）仍无法进一步突破 O(log n) 的上界。

---

## 474. Against Explainable Artificial Intelligence In Law: Why Justifiable Ai Matters. A Credit Scoring Example

**arXiv ID:** 2608.07452 | [PDF](https://arxiv.org/pdf/2608.07452v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 475. CreativeInstruct: Scalably Teaching LLMs to Balance Quality, Creativity, and Diversity

**arXiv ID:** 2608.07460 | [PDF](https://arxiv.org/pdf/2608.07460v1)

**作者:** Ananya Sahu `[一作]` (Columbia), Elias Stengel-Eskin `[通讯]` (University of Texas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种可扩展的指令微调方法，训练单一模型在生成时自我注入“创造性”标记，从而在保持对齐模型质量的同时提升输出多样性与创造力。

**💡 创新点**

创新点在于：①利用多模型推理时的路由信息生成带有创造性标记的训练数据，使模型学会在合适位置切换多样性与质量；②引入基于LLM的图编辑距离（LLM-GED）度量结构层面叙事多样性，弥补传统词汇/语义指标的不足；③证明该方法在RL任务中的探索性提升，可显著提升数学推理任务的离群性能。

**🔧 技术方法**

核心技术包括：指令微调（使用LoRA）、token级路由框架BACo生成训练样本、特殊创造性标记（[StartCreativity]/[EndCreativity]）、LLM-GED结构多样性评估。

**📊 数据集**

主要数据集为：Tülu V3 SFT（写作相关的4,000个英文提示，产生12,000个训练样本）、Narrative Discourse（评估叙事生成）、MATH与AMC用于RL实验。

**📈 对比分析**

与Instruct、BACo、Distill、CrPO等基线相比，实验在多种模型规模（7B-32B）上表现出显著提升：在语义多样性（如MiniLM余弦不相似度）和结构多样性（LLM-GED）上均高于基线，质量指标（连贯性、流畅性、相关性、写作奖励模型）保持或略有提升。人类评估显示70.3%更偏好其创造性。RL实验表明，使用该方法预训练的模型在AMC上提升约4%。

**⚠️ 局限性**

限制包括：①方法仍需在训练阶段访问多模型路由信息，训练成本相对较高；②对标记位置的自动推断可能不适用于所有文本体裁；③在某些基准（如AMC）上单纯预训练提升有限，仅在RL后才显著受益。

---

## 476. CoinRAG: Contextualized Information Nugget KV Cache Reuse for Long-Context RAG

**arXiv ID:** 2608.07458 | [PDF](https://arxiv.org/pdf/2608.07458v1)

**作者:** Gyuwan Kim `[一作]` (University of California, Santa Barbara), Tao Yang `[通讯]` (University of California, Santa Barbara)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于信息要点的KV缓存重用的长文本检索增强生成框架，在低延迟预算下提升答案质量。

**💡 创新点**

创新点在于将检索文档拆分为离线预提取的细粒度信息要点（nuggets），并通过两阶段检索与上下文对齐的KV缓存拼接，既保持深层语境又显著压缩缓存。

**🔧 技术方法**

使用了离线KV缓存预计算、两阶段检索（先块再要点）、RoPE旋转位置编码对齐、nugget‑aware fine‑tuning以及基于旋转的KV拼接技术。

**📊 数据集**

评测数据集为LongBench中的HotpotQA、2WikiMQA和MuSiQue三大多文档问答基准。

**📈 对比分析**

与TurboRAG、CacheBlend、KVLink等基线在P99 TTFT≤100 ms下对比，平均提升5.3 % F1，同时缓存长度缩短1.84×；在无延迟限制下仍保持5.2 %提升。

**⚠️ 局限性**

局限性包括：需要一次大规模离线编码、缓存与模型权重绑定、受检索召回限制、跨块注意力缺失以及未利用跨查询缓存等问题。

---

## 477. Strategy-first synthesis planning for complex natural products

**arXiv ID:** 2608.07454 | [PDF](https://arxiv.org/pdf/2608.07454v1)

**作者:** Daniel Armstrong `[一作]` (École Polytechnique Fédérale de Lausanne), Philippe Schwaller `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SynthEx，一种基于大型语言模型的 agentic 合成规划框架，能够在传统模板库之外生成复杂天然产物的完整合成路线，并将 1,098 条路线发布为 SynthAtlas。

**💡 创新点**

创新点在于让 LLM 直接生成反应的图编辑表示（ReactionJSON）而非选择已有模板，打破模板库限制，支持新的键合与环构建策略；同时采用多策略生成、迭代评估与编辑循环，使路线在结构复杂性上显著提升。

**🔧 技术方法**

技术上使用 Gemini LLM 结合多代理架构（Strategy Generator、Route Builder、Critic、Editor、Analyst）以及 MCTS 搜索；通过 ReactionJSON/RouteJSON 表示实现可编辑的图编辑反应；并利用 LLM 作为策略生成与评估器。

**📊 数据集**

数据集包括 NPAtlas 未报道天然产物 1,098 例作为基准，ZINC+eMolecules 库作为可购原料；对比使用 USPTO、Pistachio、RetroChimera 等公开数据；专家评估采用 10 名合成化学家。

**📈 对比分析**

在天然产物基准上，SynthEx 的完整路线可行率约 63.9%，远超 AiZynthFinder 的 13.8%；单步回溯时 SynthEx 的环构建步骤在 RetroChimera top‑5 仅 10.9%；专家评价显示 SynthEx 的关键步骤在可行性、优雅度等方面与文献相当，策略价值略逊。

**⚠️ 局限性**

局限性包括：实验可行性未得到验证，模型计算成本高；评估与修正共享同一 LLM，可能存在盲点；对反应条件、立体化学细节的覆盖仍有限；数据集仍主要来自未报道天然产物，尚未验证对已发表路线的预测准确性。

---

## 478. Fisher-R1: Training LLM Agents for Reliable Hypothesis Testing

**arXiv ID:** 2608.07437 | [PDF](https://arxiv.org/pdf/2608.07437v1)

**作者:** Jiacheng Miao `[一作]` (Stanford University), James Zou `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 P‑Bench 425 题的真实科学假设检验基准，并训练了 Fisher‑R1 开源 LLM 统计推断代理；

**💡 创新点**

创新点在于构建专家审核、可执行的 p 值基准，和利用合成任务+强化学习以统计结果为奖励的训练策略，显著提升 LLM 的推断准确性；

**🔧 技术方法**

使用了 R 交互式执行环境、ReAct/CodeAct 交互格式、监督微调（SFT）与 DAPO 强化学习、z‑score 评价奖励；

**📊 数据集**

使用经济学、生命科学与医学领域的 425 个真实数据集（Harvard Dataverse、cBioPortal、Vanderbilt Biostatistics），以及 8,642 条合成任务；

**📈 对比分析**

与 GPT‑5.4、DeepSeek‑V4‑Pro、GPT‑OSS‑120B、Qwen‑3‑Coder‑30B 等开源/专有模型对比，Fisher‑R1‑14B 在 P‑Bench 的 Strict 评测中超过所有基线，Raw/Strict 成功率提升 20–30%，且标准差显著下降；

**⚠️ 局限性**

局限在于仅评估单一假设检验、缺乏多重比较校正、对模型的假设解释与方法选择的透明度不足，并且仍需人工审查来避免错误传播。

---

## 479. Addressable Memory for Video World Models

**arXiv ID:** 2608.07408 | [PDF](https://arxiv.org/pdf/2608.07408v1)

**作者:** Xindi Wu `[一作]` (NVIDIA), Aljoša Ošep `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种训练无关的压缩记忆框架 WorldTrace，用以在自回归视频世界模型中实现长时序视觉持久性。

**💡 创新点**

创新点在于：① 通过给每个压缩槽分配一个不随时间漂移的、在训练分布内的虚拟位置，解决了 RoPE 位置偏移导致的地址不可达问题；② 采用“canonical key averaging”（WorldTrace‑Field）和“frozen landmark”写入器，分别实现了时间一致性与情节回忆，避免了在旋转空间平均导致的相位相消。

**🔧 技术方法**

核心技术包括：旋转位置编码（RoPE）与其在压缩空间的逆旋转；结构化稀疏注意力投影矩阵；基于场景入口检测的可冻结 landmark 关键帧选取；以及在自回归推理过程中动态更新压缩槽的策略。

**📊 数据集**

实验使用 Matrix‑Game‑2（MG2‑1.3B）和 LingBot‑World 两个自回归游戏/机器人世界模型数据集，并构建了 LoopBench 场景重访基准。

**📈 对比分析**

与滑动窗口、Block‑relative、Centroid‑linear、MemRoPE、YaRN 等基线对比，WorldTrace‑Field 在 N=48 时段的 Temporal Consistency（SSIM）提升约 +15.5%，WorldTrace‑Landmark 在 ABA 循环回访上的 Scene Consistency（CLIP 余弦相似度）提升约 +19.5%；在更长滚动窗口和多重回访情形下仍能保持 0.99 级别的回忆精度。

**⚠️ 局限性**

局限性包括：仅适用于固定 KV‑cache 预算的时间‑RoPE 自回归模型；目前仅提供两种固定的投影方式（平均与 landmark），缺乏自适应的压缩策略；在使用 bfloat16 等低精度时可能出现旋转漂移误差；对场景入口检测的依赖可能在视觉相似度高的连续场景中失效。

---

## 480. GeoDistill-Refine: Silhouette-First Geometry Distillation for Annotation-Free Spacecraft Segmentation

**arXiv ID:** 2608.07405 | [PDF](https://arxiv.org/pdf/2608.07405v1)

**作者:** Yonglong Zhang `[一作]` (Harbin Institute of Technology), Yang Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过无人工标注的方式，利用多提示SAM 3生成的伪标签并进行几何蒸馏，训练出TinyUNet在航天器前景分割上的高效模型。

**💡 创新点**

创新点是两阶段“先学习轮廓后加入可靠性门控的SDF、骨架、面积”几何蒸馏，以及固定多提示共识教师减少伪标签噪声。

**🔧 技术方法**

采用了SAM 3、TinyUNet、距离场（SDF）、骨架提取、面积约束、样本级可靠性门控等技术。

**📊 数据集**

在SpaceSense‑Bench、SPEED+、TANGO三个航天器分割数据集上评估。

**📈 对比分析**

与纯伪标签学生、基线和GABI相似方法对比，GeoDistill‑Refine在未见航天器的HJM锁箱集上 Image IoU 提升0.0456、Boundary F1 提升0.1380，外部域表现与基线相当或略优。

**⚠️ 局限性**

限制包括仅支持二值前景分割、依赖SAM 3离线推理、骨架/面积辅助贡献尚未完全解析、对极端光照/遮挡的鲁棒性有限。

---

## 481. FinRank: An Evidence-Grounded Benchmark for Financial Question Answering and Retrieval over SEC Filings

**arXiv ID:** 2608.07400 | [PDF](https://arxiv.org/pdf/2608.07400v1)

**作者:** Sasan Mansouri `[一作]`, Fabian Woebbeking `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并发布了一个基于SEC 10‑K/10‑Q文件的金融问答检索基准FinRank，重点评估检索、重排序和硬负样本抑制，并提供可复现的评测工具。

**💡 创新点**

创新点在于：①每条记录配有手工标注的支持段落与同类文件中的硬负样本，直接衡量检索模型的硬负判别能力；②引入多层次细粒度标注（难度、推理类型、证据范围、文档类型等）实现按维度的性能分解；③提供全局与记录级别的评测语料，方便在不同检索难度下对比；④首次在金融问答领域公开硬负样本集合，为后续研究提供可复现的硬负基准。

**🔧 技术方法**

使用的技术包括稀疏检索（TF‑IDF、BM25）、密集检索（多种预训练嵌入模型如mpnet、bge、e5‑mistral‑7b）、跨编码器重排序、元数据预过滤、以及针对全局与记录级别的评测脚本；此外提供了基于这些模型的基线结果。

**📊 数据集**

使用的数据集为FinRank，包含22家公司（制药、油气、汽车）在2024‑2025年的10‑K和10‑Q文件，共计约??条记录（本文未给出确切数，约数千条），每条记录含问答、支持段落、硬负样本以及丰富的元数据信息。

**📈 对比分析**

对比方法：在全局池C上评测检索 Recall@k、MRR、nDCG；在记录级候选集L_r上评测重排序 MRR、nDCG@5；对硬负判别采用成对准确率与随机负样本对比。结果显示，7B指令微调嵌入模型的 Recall@10 仅为44.8%，稠密模型提升幅度仅3.5点；硬负样本将模型成对准确率从88‑96%降至70‑80%，显示硬负的强迫性。

**⚠️ 局限性**

局限性：①规模有限，覆盖22家公司且在行业、年份和文件类型上显著偏倚；②标注由单一学生完成，仅做抽样复核，缺乏双标注一致性统计；③硬负样本中约7.3%与其他记录的支持段落相同；④仅包含文本形式，未覆盖表格、图表等多模态证据；⑤未提供生成式答案的评测，仅评估检索与重排序；⑥对外部文档或全文检索的性能未在此基准中体现。

---

## 482. SkySeaLand: A Wide-Format Satellite Transportation Benchmark with an Ultra-Lightweight Detection Baseline

**arXiv ID:** 2608.07382 | [PDF](https://arxiv.org/pdf/2608.07382v1)

**作者:** Md. Zahid Hasan Riad `[一作]` (Green University of Bangladesh), Md Sultanul Islam Ovi `[通讯]` (George Mason University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了 SkySeaLand 数据集并基准测试多种检测器，同时提出了 1M 参数的轻量化基线 SkyDet

**💡 创新点**

首次提供陆海交通四类的高分辨率卫星图像数据集，兼具 COCO 与 YOLO 注解，并给出低体量检测参考

**🔧 技术方法**

采用 MobileNetV3‑Small backbone、FPN 64 通道特征金字塔、anchor‑free FCOS 头，并与 YOLO、RT‑DETR、DETR、Faster R‑CNN 等架构对比

**📊 数据集**

SkySeaLand 1307 张卫星图像，19,101 条边界框，覆盖 airplane、boat、car、ship 四类

**📈 对比分析**

使用 COCO mAP50/50‑95 指标，640×640 letterbox 输入；12 模型在 mAP50 约 84–88，SkyDet 60.5，参数 1.22 M，4.90 MB，速度 13.74 ms（72.8 FPS）

**⚠️ 局限性**

仅单跑评估、不同训练预算、缺乏地理分离、未做尺度/类别细化分析，导致结果不具可重复性或泛化性

---

## 483. Beyond Call and Response: Modelling Reciprocal Coordination in Human-AI Vocal Ensembles

**arXiv ID:** 2608.07376 | [PDF](https://arxiv.org/pdf/2608.07376v1)

**作者:** Polina Proutskova `[一作]` `[通讯]` (Industry Commons Foundation), Polina Proutskova (Industry Commons Foundation)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个研究框架，用以把人工合成歌手融入无指挥人声合奏，并通过多通道手机录音、对齐、语义与音乐结构推断、状态推理以及低延迟生成实现互惠协同。

**💡 创新点**

创新点在于：
• 把无外部参考的合奏视为耦合动态系统；
• 通过影响网络 A(t) 捕捉歌手间的相互拉拢，而非单一节拍或得分；
• 针对非等时节奏的传统歌曲，构建分层半马尔可夫/马尔可夫再生模型；
• 研发 VocalLanes 现场排练工具及同义语音结构数据集，提供可重复的歌手主导通道。

**🔧 技术方法**

主要技术包括：多通道手机录音与 DTW/光谱对齐；语音/歌词的语音识别与 CTC/后验语音序列估计；f0、能量与对齐信息融合的状态滤波；影响权重推理（跨歌手历史提升预测）；HSMM 及时间拉伸的实时合成；层级半马尔可夫模型处理非等时节奏；以及基于歌手、流派与传统的约束学习。

**📊 数据集**

使用了 VocalLanes 数据集（40 次对齐录音，约 2.3 小时，来自十场排练，主要为一支 3-7 名歌手的乌克兰民歌合奏），并参考了 Choral Singing Dataset、ESMUC、Cantoria 等公开多轨数据集作为对比与灵感来源。

**📈 对比分析**

计划通过重复排练实验比较三种情境：人声单独、参照跟随音声、以及基于集体状态的自适应 AI 歌手。技术评估将检查跨歌手历史对预测准确性的提升，预期在节奏偏差和音高漂移检测上优于单一历史模型；但完整性能指标与对比实验尚未完成。

**⚠️ 局限性**

局限性包括：
• 录音通道仍带混叠，导致对齐与分离精度受限；
• 数据量有限，难以覆盖多语言、多传统的语音与音乐特征；
• 语音起始点不确定，导致对齐与跟踪误差；
• 实时歌声合成的延迟与音质仍低于所需标准；
• 影响网络与状态推理多为预测性，缺乏因果证据；
• 仅提出框架，尚无完整实现与实测性能。

---

## 484. An Analysis of Architectural and Operational Dynamics of Phishkits in the Wild

**arXiv ID:** 2608.07451 | [PDF](https://arxiv.org/pdf/2608.07451v1)

**作者:** Behzad Ousat `[一作]` (Florida International University), Amin Kharraz `[通讯]` (Florida International University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2020-2023年收集的1300个phishkit进行架构、代码、通信、逃逸机制等多维度分析

**💡 创新点**

首次进行跨年度、跨语言、跨行业的phishkit纵向纵深研究，并揭示其代码重用与逃逸模式的可预测性

**🔧 技术方法**

利用PHP源码抽象语法树、控制流图、Node2Vec嵌入与层次聚类；对Telegram Bot进行自动交互获取泄露数据

**📊 数据集**

1300个独特phishkit，来自公开仓库与安全公司每日数据流，涵盖23种语言和多行业目标

**📈 对比分析**

通过对比IP封锁、消息渠道与代码相似度指标，发现约21%完全相同结构、约45%采用相同IP封锁包；性能表现为高检测易度但仍需手工验证

**⚠️ 局限性**

数据样本偏向公开与付费源，缺乏全部野外案例；仅聚焦phishkit，不评估泄露数据使用效果，且聚类依赖静态分析可能忽略动态行为

---

## 485. An Exploratory Evaluation of LLM-Assisted Rewriting of Moderate-Complexity Financial Sentences for DisCoCat-Based Sentiment Analysis

**arXiv ID:** 2608.07439 | [PDF](https://arxiv.org/pdf/2608.07439v1)

**作者:** Brian Llinas `[一作]` (Old Dominion University), Nikos Chrisochoides `[通讯]` (Old Dominion University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过 LLM 进行有针对性的句子重写，将中等复杂度的金融情感句子压缩、简化或分解，使其更易被 DisCoCat 语法分析器解析并生成可执行的量子电路；

**💡 创新点**

创新点在于将大型语言模型作为预处理工具，结合提示工程与多维筛选（语义相似度、情感一致性、解析有效性），实现对中等复杂度句子的可视化压缩并提升 DisCoCat 量子情感分析的可行性与性能；

**🔧 技术方法**

使用技术包括 GPT‑4.1‑mini 与 Qwen2.5:7B LLM、Prompt A/B/C、MeaningBERT、FinBERT、Bobcat 句法解析、lambeq 语义电路构造、量子电路仿真与训练；

**📊 数据集**

实验数据集为 Stein 等人构造的合成金融情感文本，包含低复杂度（约 5 词/句）和中等复杂度（约 18 词/句）两子集；

**📈 对比分析**

与仅使用低复杂度数据的 DisCoCat 基线进行对比，采用句子层和电路层的多指标评估：Prompt B（GPT‑4.1‑mini）实现了平均 70% 以上的 qubit、门数压缩，最终准确率提升至 0.550 ± 0.035（相较基线 0.521 ± 0.050），但训练时间约增加三倍；

**⚠️ 局限性**

主要局限包括：使用合成数据缺乏真实文本多样性；实验缺乏统计显著性检验与置信区间；仅在模拟器上测试，未评估硬件成本；Qwen2.5 对部分提示失败导致对比不完整；未进行人工语义评估或迭代解析修复；整体实验条件未完全匹配基线，因而结果仅具探索性而非因果结论。

---

## 486. Post-Grokking Collapse at the Representation-Readout Interface in Muon-Trained Transformers

**arXiv ID:** 2608.07436 | [PDF](https://arxiv.org/pdf/2608.07436v1)

**作者:** Ali Janati `[一作]` (Columbia University), Anass Belfatmi `[通讯]` (CentraleSupélec)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究在单向Transformer上对模块化加法（p=113）进行训练，比较了Mu­on优化器与AdamW的学习进程，发现Mu­on在达到grokking阈值后出现训练与测试准确率骤降的“后期不稳定”现象，并通过傅里叶分解、基底对齐与冻结接口等干预手段定位并抑制该崩溃，进一步探究了不同深度、学习率、宽度和任务操作对grokking速度与稳定性的影响。

**💡 创新点**

创新点包括：①首次将grokking的后期崩溃归因于隐藏层表示与读出层之间的基底不对齐；②提出冻结非隐藏参数（嵌入与unembedding）的方法，成功消除不稳定；③通过傅里叶模式族分析揭示算法仅使用(k,k)加法族，并量化其在隐藏层与读出层中的能量分布；④系统比较了Mu­on与AdamW在不同超参数、深度与任务变体下的速度、稳定性与表示分散度。

**🔧 技术方法**

技术手段包括：decoder‑only Transformer（d_model=128，4头，MLP宽度512，无LayerNorm）；Mu­on优化器（动量+正交化+牛顿–舒尔茨归一化）；AdamW；傅里叶变换对残差状态进行二维DCT；模式族过滤（加法族、减法族、a‑only、b‑only、常数、泛交互）；干预实验（冻结、归一化去除、相位随机化、频率重新分配）；基准评估（训练/测试准确率、梯度弹性、effective pairs、余弦相似度）。

**📊 数据集**

数据集为全局模块化加法问题：每个示例由[a, b, =]三元组组成，p=113（偶数p=97、不同训练比例、不同宽度也试验）共12 769个样本；对(a+b) mod p和(a−b) mod p分别训练，随机划分30%训练集、70%测试集。

**📈 对比分析**

比较方法：在相同超参数、相同随机种子下，记录达到95%测试准确率所需的训练步数、花费秒数；统计后期不稳定评估次数、最低准确率；对比Mu­on与AdamW在不同深度下的有效pairs与任务族功率占比。结果显示：Mu­on在所有配置下均比AdamW快约1.5–2.3倍（步数）且在更深层时优势更显著；但Mu­on在所有配置下均出现不稳定，平均在训练后约500–600步出现一次准确率跌落；冻结非隐藏参数后，Mu­on在所有种子下后期不稳定评估次数降为0，最低准确率≥95%，同时速度几乎不受影响。

**⚠️ 局限性**

局限性：①实验仅在模块化加法任务及其变体上进行，未验证到更复杂任务的通用性；②对Mu­on优化器的分析基于特定的Newton–Schulz归一化实现，可能不适用于其他正交化方法；③冻结接口的干预需要事先判断何时grok完成，实际应用中需自动化判别；④对深度、宽度的探索仍有限，未覆盖更大模型；⑤基于FFT的分析假设残差是实值，若引入LayerNorm等会破坏该假设。

---

## 487. Diffusion LLMs as Targets and Adversaries: Mechanistic Safety Exploits

**arXiv ID:** 2608.07430 | [PDF](https://arxiv.org/pdf/2608.07430v1)

**作者:** Elena Dumitrescu `[一作]` (Delft University of Technology), Jérémie Decouchant `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究Diffusion大型语言模型（DLLMs）的安全性漏洞，证明其内部安全机制稀疏且可跨架构转移，并提出一种基于安全神经元损失的离线黑盒 jailbreak 框架；

**💡 创新点**

创新点在于揭示DLLMs安全神经元的结构可转移性，构建连续的加权安全神经元损失作为优化目标，并利用该损失实现低计算成本、高成功率的离线黑盒 jailbreak；

**🔧 技术方法**

技术主要包括：安全神经元识别（激活剖析 + 逻辑回归）、跨架构迁移攻击（直接映射与剪枝）、加权安全神经元损失的构造、SN‑Guided Diffusion 的离线优化循环以及生成剪枝级联；

**📊 数据集**

实验使用多款 DLLM 与 AR 预训练模型（如 LLaDA、Dream、Qwen2.5、Fast‑dLLM、Llama‑3‑8B‑Instruct、Qwen2.5‑7B‑Instruct、Gemini‑2.5‑Flash‑Lite 等），以及公开和专有的大模型；

**📈 对比分析**

与现有 jailbreak 方法相比，SN‑Guided Diffusion 在 Llama‑3‑8B‑Instruct、Qwen2.5‑7B‑Instruct、Gemini‑2.5‑Flash‑Lite 等模型上分别达成最高 77.1%、86.9% 和 74.3% 的攻击成功率，同时仅需 20 次生成迭代，计算成本比传统 RL 或大规模搜索低数百倍；

**⚠️ 局限性**

局限性包括：攻击未进行全面的超参数调优，仍有进一步节能空间；对提示模板的选择依赖固定 persona 结构，未系统评估不同结构影响；仅为离线迁移攻击，未结合实时黑盒反馈；以及对模型内部结构的假设（如安全神经元可被精确识别）在不同模型上可能不完全成立。

---

## 488. ResidencyRL: Reinforcement Learning in Simulated Clinical Environments

**arXiv ID:** 2608.07418 | [PDF](https://arxiv.org/pdf/2608.07418v1)

**作者:** Valentin Liévin `[一作]` (Google DeepMind), Lin Yang `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究提出并实现了ResidencyRL框架，通过多轮强化学习在大规模模拟临床对话中训练AI诊疗代理，使其在诊断、管理、信息采集、沟通和安全性方面显著提升。

**💡 创新点**

创新点包括：①将完整的临床就诊流程（对话+工具调用）纳入长期POMDP框架；②使用层次化的自动评估器（auto‑rater）提供多维度奖励；③构建多元化、可定制化的患者模拟器和专门的历史采集及安全对抗场景；④利用GRPO进行在线多轮RL，实现从数千个模拟场景到真实临床任务的迁移。

**🔧 技术方法**

主要技术包括：Gemini 3.5 Flash作为基础LLM，Gemini 3.1 Pro用于情境生成与评估；多轮强化学习框架（GRPO）和大规模并行rollout；生成式情境管线（四阶段生成、LLM审查、去重）；结构化工具调用API（诊断、SOAP、管理计划等）；以及基于Likert量表的分层奖励设计。

**📊 数据集**

使用的公开数据集包括DDxPlus（疾病关联与症状），AMIE Mx（多访 longitudinal）、AgentClinic、CRAFT‑MD等；训练集约5万多条情境，涵盖81种临床病种、5千条专门历史采集场景和2.5千条安全对抗场景；测试集则来源于这些数据集的hold‑out、外域评估和临床专家人工评测。

**📈 对比分析**

与基线模型（未RL训练的Gemini 3.5 Flash）以及其他单轮或短期RL对比，ResidencyRL在诊断准确率、管理质量、信息采集完整度、沟通效果和安全性六维度均显著提升；在97名临床专家的盲评中，ResidencyRL在整体临床印象上胜率达87.6%，在信息采集完整度和管理计划适当性上分别达90.7%和75.3%。在外域评测（AMIE Mx、多访、肿瘤专科、AgentClinic、CRAFT‑MD）均表现出正向迁移，未见显著安全回退。

**⚠️ 局限性**

局限性包括：①仅训练于基于文本的远程医疗对话，缺乏多模态感知（体格检查、影像、实验室）及多访时序决策；②模拟患者与真实患者的行为差异仍存在；③自动评估器可能存在Goodhart效应，偏向已训练模型；④奖励信号为代理性代理，尚未通过真实患者结果验证；⑤实验与代码未公开，重现实验难度较高。

---

## 489. Bayesian Fair Division: Truthfulness in Picking Sequence with Correlated Valuations

**arXiv ID:** 2608.07414 | [PDF](https://arxiv.org/pdf/2608.07414v1)

**作者:** Xiaolin Bu `[一作]` (Shanghai Jiao Tong University), Biaoshuai Tao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究在贝叶斯设定下，顺序分配（如回合制）机制的真诚性，并证明在两代理情形下若价值呈正相关（满足强随机支配或MLRP），真诚成为贝叶斯纳什均衡；同时给出三代理及以上情况下的反例，证明此性质不再成立。

**💡 创新点**

创新点在于：①提出“强随机支配”这一比传统随机支配更严格的约束，用来刻画价值的正相关；②证明两代理时强随机支配（或MLRP）即为真诚的必要且足够条件；③展示新的多代理操纵策略并证明正相关不足以消除该操纵；④通过耦合和逐步揭示等技术突破传统的历史依赖难题。

**🔧 技术方法**

主要技术包括：贝叶斯策略空间的定义、随机化顺序分配的数学模型、耦合与“one‑position misalignment”技术、渐进揭示的等价过程、强随机支配与MLRP的概率论推导以及对比反例的枚举分析。

**📊 数据集**

本文为纯理论研究，没有使用任何实测数据集；所有结果均通过严格的概率与组合论证明。

**📈 对比分析**

方法的对比主要通过理论证明与构造反例完成。结果显示：在两代理且满足强随机支配/MLRP时，真诚是贝叶斯纳什均衡；而在三代理及以上时，即使满足MLRP，也可能出现真诚失效的贝叶斯纳什均衡，且此时可出现 EF1 违背。

**⚠️ 局限性**

局限性在于：①正相关的假设仅在两代理时有效；②在多代理情形下，本文未给出足够强的条件来保证真诚；③机制仍易受新的多代理操纵；④仅考虑可加性价值，其他非可加情形未覆盖。

---

## 490. People Are Not Just Their Countries. Disentangling Social Determinants of LLM Value Alignment Across Europe

**arXiv ID:** 2608.07367 | [PDF](https://arxiv.org/pdf/2608.07367v1)

**作者:** Maria-Louisa Wightman `[一作]` (Ghent University), Tijl De Bie `[通讯]` (Ghent University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对10种主流大型语言模型（LLMs）在欧洲社会调查（ESS）中提出的价值观问题进行评估，构建与受访者的价值观匹配度得分，并分析不同社会经济、人口统计变量与国家归属对匹配度的影响。

**💡 创新点**

首次在跨国层面同时考虑社会经济与人口统计特征，揭示 LLM 与不同社会阶层（教育、收入、职业、宗教等）之间存在显著的价值不匹配；并证明国家归属与社会经济因素对匹配度的解释作用是互补而非可替代的。

**🔧 技术方法**

利用大规模提示与答案聚合（多数投票）、对答案进行 Likert 标准化并计算相似度得分；对差异进行 Bootstrap 置信区间估计；使用逆倾向性加权（IPW）重权重以消除国家内分布差异；通过线性回归与梯度提升树（GBM）对匹配度进行预测，评估各变量集的解释力。

**📊 数据集**

欧洲社会调查（ESS）第11波，涵盖29个欧洲国家及以色列，包含 47 个价值观与态度问题，15 个社会经济与人口统计变量，50,116 名受访者。

**📈 对比分析**

通过对10个模型的平均匹配度、交叉模型偏差以及预测模型的 R² 进行比较。对比显示：单独使用国家或社会经济变量的解释力与二者组合相当；GBM 与 OLS 的差异不大，表明高阶交互作用影响有限。整体匹配度范围为 0.58–0.75，模型间差异不显著。

**⚠️ 局限性**

局限性包括：仅使用英文提示，可能忽略语言差异；答案聚合采用多数投票，未充分考虑模型回答变异；使用多项选择题作为价值观代理，可能与真实价值观不完全一致；未覆盖非欧洲地区与多语种评估，导致结论在全球范围内的可推广性受限。

---

## 491. Curriculum as Code: An AI-Assisted Architecture for Instructional Design in STEM Education

**arXiv ID:** 2608.07364 | [PDF](https://arxiv.org/pdf/2608.07364v1)

**作者:** Henrique Mohallem Paiva `[一作]` `[通讯]` (Universidade Federal de Sao Paulo), Henrique Mohallem Paiva (Universidade Federal de Sao Paulo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并验证了一套六阶段 AI 辅助教学设计架构，基于 Curriculum as Code，使用 LaTeX/Beamer 与 Python 自动生成 STEM 教学材料；

**💡 创新点**

将生成式 AI 与代码化工具结合，形成可复制的多阶段流水线，解决 AI 幻觉、视觉一致性和教师隐性知识转化问题，并实现跨教师、跨项目的可扩展性；

**🔧 技术方法**

使用 Gemini Pro/DeepSeek 大语言模型，配合 LaTeX（Beamer）和 Python（Matplotlib/Seaborn）实现文本、幻灯片与图表的自动生成与校验；

**📊 数据集**

采用 8 个学术模块（共 24 个项目场景）以及约 600 份学生评估数据，配合内部教学材料和项目描述文件进行实验验证；

**📈 对比分析**

与传统手工制作对比，教师准备时间从约 8 小时降至 2 小时，90% 减少；学生满意度平均 8.5–9.9；同行评审与自动审查确认无概念性幻觉，编译成功率高；

**⚠️ 局限性**

仍需人工校正语法细节，依赖单一机构的教师训练，模型更新导致漂移，需要多站点验证；高级专业模块仅单师部署，局限在规模与多样性方面。

---

## 492. Taxonomy-Driven Analysis of Open-Source AI Risk Mitigation Tools

**arXiv ID:** 2608.07446 | [PDF](https://arxiv.org/pdf/2608.07446v1)

**作者:** Afreen Alam `[一作]` (Boston University), Dimitar Trajanov `[通讯]` (Ss. Cyril and Methodius University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性地将21款主流开源LLM风险缓解工具映射到MIT AI风险缓解与响应分类法，形成Tool‑Risk矩阵；

**💡 创新点**

创新点在于提出基于LLM的检索增强生成（RAG）协议并配合人工验证，实现技术能力与风险分类的自动化对齐，并基于此设计了四层风险缓解架构；

**🔧 技术方法**

使用NotebookLM实现RAG提取、Prompt设计、Fleiss Kappa评估、F1等指标，依托GitHub仓库、代码和文档；

**📊 数据集**

主要数据来源为21款开源工具的GitHub仓库及其源代码、文档与配置文件，以及扩展MIT风险分类法的CSV；

**📈 对比分析**

与人工验证样本对比，LLM映射的准确率84.5%，精确率78.4%，召回率72.7%，F1 75.5%；人机一致性Fleiss Kappa为0.509，表明方法具备可接受的可靠性；

**⚠️ 局限性**

局限性包括：仅覆盖开源工具、对最新版本的变更不敏感、NotebookLM闭源导致可复现性受限、人工评审的中等一致性、技术能力与治理层面存在显著缺口，且对真实生产环境的效果验证尚待进一步研究。

---

## 493. TEPA: Revoking Stale Memories for Conflict-Robust Language Agents

**arXiv ID:** 2608.07429 | [PDF](https://arxiv.org/pdf/2608.07429v1)

**作者:** Yan Zhou `[一作]` (Changsha University of Science and Technology), Suncheng Xiang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可撤销的证据记忆机制（Revocable Evidence Memory），在语言智能体中通过键值对记录经验，并在出现冲突时主动将过期证据从激活集移除，保持记忆的有效性。

**💡 创新点**

创新点在于将“记忆污染（memory pollution）”定义为持久化记忆中被冲突证据覆盖但仍被检索的错误信息，并通过显式的生命周期状态（Hypothesis/Active/Revoked）和撤销操作来消除这类污染，同时保留已撤销条目用于审计与复审。

**🔧 技术方法**

采用键值提取器将观察转化为冲突键和值，使用 Beta-Bernoulli 后验估计每个前件的可信度；基于阈值（revocation, promotion, recent success）实现状态转移；可选的 Trial-Validated Promotion 在拥有可执行验证的情境下进一步过滤错误候选；实现了 -Rev、-Full 等变体。

**📊 数据集**

在四类评测场景中验证：1）受控隐藏域漂移（controlled hidden‑regime drift）；2）真实文件/工具执行漂移（real file‑backed executable drift）；3）用户偏好更新流（preference‑update stream）；4）MemoryAgentBench（SH‑6k、MH‑6k、长上下文版）作外部事实整合基准。

**📈 对比分析**

与无记忆、追加式、回忆式、最近值优先、时间滑动窗口、语义检索、反应性遗忘、Oracle 重置等基线进行对比。实验显示，在全反转阶段追加式和回忆式记忆的成功率低于无记忆（如 0.210 vs 0.309），而 Revocable 机制保持在 0.950 以上；在偏好更新中 -Full 方案达到 0.910 与无记忆持平；在 MemoryAgentBench 单跳冲突中 Revocable 与 last‑write‑wins 取得 0.890 的最高匹配率。

**⚠️ 局限性**

限制主要包括：1）需能从观察中提取可靠的冲突键，若键噪声或开放式记忆难以定义键，机制效果下降；2）在多跳关系检索或极长上下文场景下，仅解决事实级有效性不足，仍面临检索链构建与上下文选择瓶颈。

---

## 494. A Picture is Worth a Thousand Tokens: How Vision Language Models Cut AI Energy Costs While Improving Accuracy

**arXiv ID:** 2608.07427 | [PDF](https://arxiv.org/pdf/2608.07427v1)

**作者:** Bhavika Jalli `[一作]` (Ericsson), Jayanta Choudhury `[通讯]` (Ericsson)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将数值时序数据渲染成二维图像，并利用 Vision‑Language Models (VLM) 进行异常检测，本文对比了 VLM 与传统 LLM 在能耗和准确率上的表现。

**💡 创新点**

创新点在于：① 用视觉表示显著压缩输入 token 数量（3.6–10.4 倍），从而在保持甚至提升检测精度的同时降低 1.8–2.5 倍的推理能耗；② 引入 J/F1 能效指标，系统性量化能耗与性能的权衡；③ 评估不同视觉编码器（Llama‑3.2‑Vision、Qwen2.5‑VL、Pixtral‑12B）的能耗差异，为边缘部署提供选择依据。

**🔧 技术方法**

采用 Llama‑3.2‑90B‑Vision、Qwen2.5‑VL‑72B、Pixtral‑12B 三款 VLM；将时间序列绘制为 2D 栈式子图并送入视觉编码器；使用 NVML 硬件计数器测量 GPU 能耗；利用零样本和微调方式进行异常检测；在图像与文本两种输入模式下对比 token 数量、能耗、F1 分数。

**📊 数据集**

使用 realAWSCloudwatch（云基础设施监控的单变量时序数据）和真实 4G/5G 网络 KPI 数据（8‑KPI、15 分钟采样，209 个基站）作为实验数据集。

**📈 对比分析**

通过对比每种模型在文本与图像两种输入下的 token 数量、推理能耗（J）以及 F1 分数，计算 J/F1 能效比。结果显示：VLM 在所有模型上均实现 1.8–2.5 倍能耗下降；Pixtral‑12B 在图像模式下 J/F1 提升 20.6 倍；在实时网络异常检测中，微调后的 Llama‑3.2‑Vision 的精度提升 220.7%，并比 LSTM/ARIMA 高 144%。

**⚠️ 局限性**

局限性包括：仅测试了三款 VLM，未涵盖最新的稀疏或专家混合模型；实验均在单 GPU（RTX A6000/A100）上进行，未评估多 GPU 并行对能耗的影响；输出长度固定为 256 token，无法推广到长文本任务；仅在单一运营商的 4G/5G 数据上验证，跨运营商或跨国家的适用性未知；能耗测量侧重 GPU 硬件，未考虑系统级能耗与碳足迹。

---

## 495. CoBa: Cost-Effective Test-Time Scaling via Compute-Balanced Routing

**arXiv ID:** 2608.07424 | [PDF](https://arxiv.org/pdf/2608.07424v1)

**作者:** Yan Zhou `[一作]` (Changsha University of Science and Technology), Suncheng Xiang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了测试时推理的计算分配问题，提出了一种在生成、轻量验证、强验证和停止之间动态路由的策略。

**💡 创新点**

创新点在于将生成与验证统一视为计算分配任务，并设计了基于答案一致性和轻量验证分层的可解释路由框架。

**🔧 技术方法**

使用了多级验证器（频率、轻量评估、强评估）与离线重放的路由决策机制，并采用Qwen3-14B、Phi-4-reasoning和Qwen3-8B模型。

**📊 数据集**

在MATH-500、AIME 2024/2025、AMC 2023以及Procedural Reasoning Gym等竞赛式数学与符号推理数据集上进行评估。

**📈 对比分析**

与greedy、best‑of‑N、多种自评估以及pool oracle等基线在同一候选池上对齐；路由策略在保持约85.1%宏观准确率的同时，将参数加权token降低49%或58%，在成本与准确率上接近或超越best‑of‑16和自评估基线。

**⚠️ 局限性**

局限在于对最难数据集（如AIME 2025和Reasoning Gym）的oracle gap仍较大，表明仍需提升候选生成质量或更精准的置信估计；当前路由策略主要基于规则，学习控制器效果尚未显著。

---

## 496. Beyond Myopic World Models: Long-Horizon End-to-End Training for Direct Future Prediction

**arXiv ID:** 2608.07420 | [PDF](https://arxiv.org/pdf/2608.07420v1)

**作者:** Xinyi Li `[一作]` (University of California), Yubei Chen `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Direct Prediction World Model（DPWM），通过一次前向传播直接预测任意长度动作序列的终点观测，避免递归滚动带来的误差累积；

**💡 创新点**

创新点在于将长时间尺度的端点监督作为训练目标，证明训练范式（而非单一架构）是提升长程预测准确性的关键；

**🔧 技术方法**

使用动作序列Transformer编码器、FiLM‑conditioned MLP动态模块以及端点损失，整体为端到端非递归网络；

**📊 数据集**

在DeepMind Control Suite（Cheetah、Humanoid、Hopper、Walker）和Atari Pong的轨迹数据上进行评估；

**📈 对比分析**

与MoSim、ADM以及传统递归世界模型在不同预测步长（1、16、100、200）下对比，DPWM在更长步长上显著降低端点MSE，性能提升显著；

**⚠️ 局限性**

局限性包括未在规划或控制任务中验证实际效果、未对生成轨迹的时间一致性做约束、仅在确定性环境上测试，尚未覆盖随机动力学。

---

## 497. Beyond Post-Hoc Temperature Scaling: Bilevel Optimization for LLM Calibration

**arXiv ID:** 2608.07419 | [PDF](https://arxiv.org/pdf/2608.07419v1)

**作者:** Ruochen Jin `[一作]` (Dartmouth College), Bojian Hou `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CALM，一种基于双层优化的训练时校准框架，利用熵最大化作为上层目标，自动调节温度样式超参数，提升对齐后 LLM 的校准性能。

**💡 创新点**

创新点在于：①将温度缩放扩展到训练时双层结构；②用熵最大化作为无标签的校准目标，避免过度依赖验证集；③采用 BOME 启发的首阶近似，使得在 LLM 规模下可行。

**🔧 技术方法**

技术手段包括：双层优化、熵最大化上层目标、向量化温度与 logit 移动、BOME（First-order Bilevel Optimization）近似、QLoRA 微调。

**📊 数据集**

实验使用的主要数据集包括多选问答集（MMLU、MedMCQA、OpenBookQA、ARC-Challenge）、开放式问答集（PopQA、TriviaQA），以及四个已对齐的 LLM（Llama‑3.1‑Tulu‑8B、Vicuna‑7B‑v1.5、Olmo‑2‑7B、Mistral‑7B）。

**📈 对比分析**

与原始对齐模型、温度缩放、标签平滑、校准感知微调（CFT）等方法对比，CALM 在多选及开放式 QA 的 OOD 场景下 ECE/Sem‑ECE 均显著下降，且保持或略低的准确率，显示出更优的校准‑效用平衡。

**⚠️ 局限性**

局限性：仍需要大量计算资源，超参数（如学习率、BOME 步数）对结果影响显著；在部分模型（如 Llama‑3.1）对随机种子敏感；在极端 OOD 任务中仍可能出现轻微准确率下降。

---

## 498. I Seek You in Videos: Identity-Conditioned Queries for Person-Centric Video Reasoning

**arXiv ID:** 2608.07417 | [PDF](https://arxiv.org/pdf/2608.07417v1)

**作者:** Shibo Gao `[一作]` (Beijing Jiaotong University), Peipei Yang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了人像中心的多源视频推理任务 ICQ，并构建了 ISYV Benchmark（1,377 条视频+问答）与 ISYV-75K 训练集（75K 条样本），同时设计了 ICQ 模块和 ESR 奖励的多模态大语言模型训练框架。

**💡 创新点**

创新点在于：①将参考人像与长视频、文本查询联合建模，突破传统 video‑text 双模；②将任务划分为六个认知层级；③通过可学习的图像标记令参考人像压缩为紧凑表示；④提出无监督有效片段奖励（ESR）解决无标注片段定位问题。

**🔧 技术方法**

使用的技术包括：多模态 LLM、强化学习（GRPO）与奖励设计、ICQ 模块（可学习标记 + 跨模态注意力）、数据自动化构建（TransNet V2、Gemini 2.5 Pro、Qwen 系列）、多轮质量检查。

**📊 数据集**

使用的数据集有：ISYV-Bench（1,377 真实长视频，1,377 人像参考图与 QA 对），ISYV-75K（24,150 条视频，74,578 条 QA），并对比了公开 benchmark 如 TVQA、MVBench 等。

**📈 对比分析**

在 ISYV-Bench 上对比了多款闭源（Gemini‑2.5‑Pro、GPT‑5.2‑Global）与开源（Qwen、InternVL、VideoLLaMA3）MLLM，闭源模型平均准确率约 60‑70%，人类约 96%。ISYV‑Model 在 7B 参数下通过 SFT‑RFT 训练后整体准确率提升至 57%（最高层 CR 70%），与闭源模型相近，但仍低于人类。

**⚠️ 局限性**

主要限制包括：跨域身份匹配与长时序跟踪仍表现差；模型对 ICQ 任务的理解仍不足，存在“答案黑客”现象；训练成本高，仍需进一步提升泛化与鲁棒性。

---

## 499. Circuit-Based Program Verification: Sequential Circuits as an Intermediate Representation for Verifying C Programs

**arXiv ID:** 2608.07397 | [PDF](https://arxiv.org/pdf/2608.07397v1)

**作者:** Po-Chun Chien `[一作]` (LMU Munich), Dirk Beyer `[通讯]` (LMU Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了 Circuit-Based Program Verification 框架，将 C 程序转换为顺序电路，并直接使用硬件模型检查器完成可达性安全和终止性验证。

**💡 创新点**

创新点在于统一了程序注入、电路编码、属性表示和证据回译，并引入了功能性与关系性两种电路编码、可选的 Liveness‑to‑Safety 转换以及多引擎硬件模型检查组合。

**🔧 技术方法**

核心技术包括控制流自动机到大块编码（LBE）、Btor2 顺序电路生成、功能/关系式编码、L2S 变换、硬件模型检查器集成（ABC、AVR、Pono、rIC3）以及证据翻译回 C 级别。

**📊 数据集**

实验使用了 SV‑COMP 2026 的 16000+ 个任务，主要集中在 ReachSafety 与 Termination 两个类别。

**📈 对比分析**

与五大软件验证器（CPAChecker、ESBMC、Symbiotic、Ultimate、Kratos）对比，CPV 在 ReachSafety 与 Termination 上各自解决了第二多实例，整体性能与顶级工具相当且互补。

**⚠️ 局限性**

局限性包括对某些语言特性（longjmp、浮点库、动态内存、递归、数组初始值等）的支持不足，功能编码不支持浮点，转换过程仍需改进以降低失败率，以及缺乏正确性证据的回译。

---

## 500. PACE: Primitive-Aware Code Evolution for Automated Algorithm Design

**arXiv ID:** 2608.07395 | [PDF](https://arxiv.org/pdf/2608.07395v1)

**作者:** Zhuoliang Xie `[一作]` (Southern University of Science and Technology), Zhengkun Wang `[通讯]` (Southern University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种利用可持久化的可执行算法原语（EAP）进行模块化自动算法设计的方法，称为Primitive-Aware Code Evolution (PACE)；

**💡 创新点**

创新点在于将局部逻辑抽象为独立可复用的函数单元EAP，并通过特定的变异算子和Thompson Sampling实现EAP的跨程序迁移与信用分配；

**🔧 技术方法**

主要技术包括：LLM驱动的程序生成与评估、可变结构的四类变异算子（插入、替换、细化、交叉）、基于贝叶斯后验的Thompson Sampling用于EAP选择、以及基于父子性能提升的增量信用评估；

**📊 数据集**

实验数据集包括：OpenAI Gym中的Racing Car（视觉输入）和Bipedal Walker（状态输入）两类连续控制任务；以及两个组合优化任务TSP-ACO与TSP-Construct（不同规模的旅行商问题）；

**📈 对比分析**

与多种基线（如EoH、ReEvo、MCTS-AHD、HSEvo、MLES、PPO、DeepACO、经典ACO）比较，PACE在控制任务上训练与测试分数均超过所有AAD基线和PPO，在TSP任务上无前瞻性扩展到更大规模时保持最优或次优表现；

**⚠️ 局限性**

限制在于EAP的独立评估可能忽略不同原语间的耦合与交互，未来需研究原语依赖建模以进一步提升方法鲁棒性。

---

## 501. FedDOSE: Federated Learning Framework Decomposing Site Effects for Modeling Brain Dynamic Functional Connectivity

**arXiv ID:** 2608.07393 | [PDF](https://arxiv.org/pdf/2608.07393v1)

**作者:** Deepank Girish `[一作]` (Nanyang Technological University), Jagath C. Rajapakse `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

开发了一种名为FedDOSE的联邦学习框架，专门用于多站点动态功能连接(fMRI dFC)数据的分析，能够在保护数据隐私的同时提升神经发育疾病（如自闭症、注意缺陷多动障碍）的分类精度。

**💡 创新点**

核心创新点包括：①将站点差异分解为疾病相关、表型相关和扫描仪相关子空间，避免这些因素与疾病信号混杂；②引入模块化引导的 Tucker 分解（MGTKD）高效压缩 dFC 张量，保留脑网络模块的空间时序模式；③利用OT barycenter 与 Procrustes 分析对各站点的原型进行几何对齐，从而解决同一网络在不同站点被误映射的问题。

**🔧 技术方法**

技术细节涵盖：动态功能连接矩阵构建、张量对数映射与Riemannian处理、模块化 Tucker 分解、softmax 门控子空间分解、梯度反转层、跨站点OT barycenter 及 Procrustes 对齐、以及多项损失（交叉熵、MSE、重构、原型相似度）。

**📊 数据集**

实验使用三大公开多站点 rs‑fMRI 数据集：ABIDE‑I、ABIDE‑II（自闭症）和 ADHD‑200（注意缺陷多动障碍），共约2000名受试者，涵盖20+ 站点。

**📈 对比分析**

与 FedAvg、FedProto、TDPFed、GAFD、FedGST、FedAli、FedGMKD 等七种先进联邦学习方法及本地/中心化基线进行对比。FedDOSE 在所有数据集的站点准确率和全局准确率均优于基线，显著提升 3–4%（自闭症）和 2–3%（ADHD），并在多站点异质性下逼近中心化上限。

**⚠️ 局限性**

主要局限在于模型结构复杂，尤其是模块化 Tucker 分解和原型对齐步骤会导致计算和通信开销增加，尚未提供完整的客户端–服务器实现与成本评估，未来需进一步优化效率与可扩展性。

---

## 502. Omni-modal decomposition autoencoders learn full-stack wearable disentangled representations

**arXiv ID:** 2608.07385 | [PDF](https://arxiv.org/pdf/2608.07385v1)

**作者:** Ioannis Ziogas `[一作]`, Dimitrios Hatzinakos `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出OmniDecVAE框架，实现全栈可穿戴设备的多模态分解自编码器；

**💡 创新点**

创新点在于通过自监督分解损失（SSLDec）实现多模态无监督表征的解耦与融合，并兼顾生成、解码与分类；

**🔧 技术方法**

采用变分自编码（VAE）+自监督对比学习、STFT/Mel 频域表示、共享一支卷积编码器和条件解码器；

**📊 数据集**

使用HARWE大规模多模态人类活动与身份识别数据集（35人、30种通道）；

**📈 对比分析**

与Transformer、MMVAE、ICA/PCA、YAMNet/EEGNet等基准对比，OmniDecVAE在SD/ SI场景下的HAR/IR准确率提升6.75%/1.01%，生成MAE与MK-MMD分别提高13.85%与76.84%；

**⚠️ 局限性**

局限在于仅使用频域表示，未覆盖视频/文本等非时序模态，缺乏跨模态缺失数据鲁棒性验证，且推理FLOPs仍高于Transformer。

---

## 503. LitTraceQA: A Benchmark for Multi-Stage Grounding and Verification in Scientific Question Answering

**arXiv ID:** 2608.07370 | [PDF](https://arxiv.org/pdf/2608.07370v1)

**作者:** Xuye Liu `[一作]` (University of Waterloo), Krzysztof Czarnecki `[通讯]` (University of Waterloo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了LitTraceQA，一个基于检索、证据定位与答案生成的可追踪科学论文问答基准。

**💡 创新点**

创新点在于将检索、证据定位与答案生成拆分为显式可评估的三阶段流程，并支持多种证据类型（表格、图形、文本段落、公式/算法、引用上下文）和答案格式（多项选择、结构化表格）。

**🔧 技术方法**

采用了开放式生成–证据–挑战构造流程，利用大型语言模型生成问题，自动化检验证据归属与可检索性，并通过多种检索与生成技术（检索-增强生成、提示工程、模型微调）进行实验。

**📊 数据集**

使用了27,487篇机器学习、计算机视觉与自然语言处理论文的元数据及其本地文本缓存，构建了4,978条独特问题记录，覆盖1,339条表格、1,132条图形、933条文本、819条公式/算法、755条引用上下文等。

**📈 对比分析**

通过对检索、证据定位与答案三部分分别评估，并提出严格联合成功指标；实验中大部分闭书挑战模型（如GPT、Claude、Gemma）在61.9%的实例上全部错误，显示基准对检索增强系统有较高难度。

**⚠️ 局限性**

局限性包括证据定位字段尚未规范化、缺少正式公开/隐藏划分、单篇论文子集未明确处理、版权与重分发说明不足，以及需要人工质控和基线系统的进一步评估。

---

## 504. From the Dirichlet Integral to Lobachevsky's Formula: A Formalization in Lean 4

**arXiv ID:** 2608.07366 | [PDF](https://arxiv.org/pdf/2608.07366v1)

**作者:** Daniel Goldberg `[一作]` (Technion Israel Institute of Technology), Antoine Vinciguerra `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在Lean 4中形式化Dirichlet积分及其相关三角积分恒等式，并通过可积的sinc²函数间接证明Dirichlet积分，随后推导Heaviside逼近、Dirichlet截断以及Lobachevsky积分公式，完成了对连续周期函数的均匀逼近与积分化简

**💡 创新点**

首次在计算机证明环境中完整形式化Dirichlet积分、其截断收敛以及Lobachevsky公式，并通过可积的sinc²函数规避条件收敛的技术难题

**🔧 技术方法**

使用Lean 4证明助手、Mathlib数学库、Fourier分析与均匀逼近理论（AddCircleπ）

**📊 数据集**

无（本工作为形式化证明，不涉及实验数据集）

**📈 对比分析**

无（未进行性能比较或实验评估）

**⚠️ 局限性**

对更高次sinc幂或更一般权函数的推广仍需额外模式识别与逼近技术

---

## 505. Analyzing the Interaction of Optimal Strategies in Mean-Payoff Bidding Games

**arXiv ID:** 2608.07383 | [PDF](https://arxiv.org/pdf/2608.07383v1)

**作者:** Shaull Almagor `[一作]` (Technion), Julian Ewaied `[通讯]` (Technion)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究两类已知最优策略（区块策略与预算策略）在均值收益竞价游戏中的相互作用，证明生成的路径在某些图（递归图、重复竞价游戏）上最终是周期性的，并给出计算平均收益的算法。

**💡 创新点**

创新点在于：①首次对最优策略在非零和竞价游戏中的交互结果进行形式化分析；②利用能量区块差分、区间方向等新技术将无限状态空间压缩为有限状态，证明路径最终周期；③在预算策略情形下提出半算法并在重复竞价游戏上证明其收敛性，推导多项式时间收益计算方法。

**🔧 技术方法**

技术方法包括：区块策略的归一化与能量区块划分、能量差分走向分析、区间标签与方向定义、分块策略的线性/非线性动力学、预算策略的分段线性预算更新函数、分支函数合成与迭代收敛分析。

**📊 数据集**

本文完全是理论分析，不使用实验数据集，所有结果均来自数学证明与算法设计。

**📈 对比分析**

在可分析的图结构（递归图、重复竞价游戏）中，算法能够在多项式时间内给出两玩家的平均收益；与传统的零和最优策略直接使用相比，得到的实际收益往往更高，且对不确定对手的鲁棒性更好。

**⚠️ 局限性**

局限性包括：仅证明了递归图和重复竞价游戏中的周期性；对一般强连通图尚未给出完整分析；预算策略在更复杂结构下的收敛性仍是未解决问题；算法复杂度在最坏情况仍较高。

---

## 506. MirrorWorld: Taming Video Diffusion Models for Mirror Reflection Generation

**arXiv ID:** 2608.07463 | [PDF](https://arxiv.org/pdf/2608.07463v1)

**作者:** Youjun Zhao `[一作]` (City University of Hong Kong), Rynson W. H. Lau `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MirrorWorld，一种在视频中修复镜像区域的反射感知视频修复框架。

**💡 创新点**

创新点是将镜像生成拆解为“应该反射什么”和“如何反射”，并通过语义关系蒸馏（SRD）和几何变换对齐（GTA）分别学习语义关联和空间布局。

**🔧 技术方法**

采用视频扩散模型作为基准，使用冻结的 VideoMAEv2 提取特征，结合 SRD 与 GTA 约束，并通过 LoRA 微调实现训练。

**📊 数据集**

构建了统一的视频镜像反射基准，整合 VMD-D、ZOOM、MMD、DVMD-D 四个数据集，共 1,242 条视频片段。

**📈 对比分析**

与 MirrorFusion/MirrorVerse、VideoPainter、VACE 等方法在 PSNR、SSIM、LPIPS、FVD 上对比，MirrorWorld 在镜像区的 PSNR/SSIM/LPIPS 与 FVD 均优于基线，显示更优的重建和视频质量。

**⚠️ 局限性**

局限在于当镜面反射的内容完全不可见时模型无法恢复真实内容，只能基于先验生成一致的反射；对更复杂的反射几何支持有限。

---

## 507. SkillProx: Self-Evolving Agent Skills via Proximal Textual Gradient Descent

**arXiv ID:** 2608.07449 | [PDF](https://arxiv.org/pdf/2608.07449v1)

**作者:** Mingxuan Zheng `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 SkillProx 框架，通过前向闭环诊断与后向近端精炼实现 LLM 代理技能的自演化

**💡 创新点**

创新点在于结合了结果验证的前向更新和基于效用的后向精简，解决了前向更新不被验证和技能无序增长的问题

**🔧 技术方法**

采用闭环诊断更新、同批次重执行、冻结效用审计、验证门控近端收缩等技术，类似近端梯度下降的思路

**📊 数据集**

主要在 SpreadsheetBench Verified、WikiTableQuestions 与 HiTab 三个数据集上进行实验

**📈 对比分析**

与无技能、人工技能、EvoSkill、Trace2Skill、SkillGrad、SkillOpt 等基线对比，SkillProx 在三大 LLM 上在 IID 与 OOD 上平均提升 3% 以上，表现最优且方差最小

**⚠️ 局限性**

局限在于需要多轮重执行与验证导致开销较大，且在极度高压缩下性能可能下降

---

## 508. Blast Radius

**arXiv ID:** 2608.07440 | [PDF](https://arxiv.org/pdf/2608.07440v1)

**作者:** MY Pitsane `[一作]` (Mankind Research Labs), Hope Mogale `[通讯]` (Mankind Research Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Blast Radius，一种双通道的可逆记忆管理层，能预测提示在上下文和代码上的影响并按需可逆地归档无用内容，避免传统截断或压缩导致的信息丢失；

**💡 创新点**

创新点在于将可逆归档与预测的“爆炸半径”结合，形成可测量的折叠策略，并引入递归死物（Recurring Dead Matter）检测机制，进一步减少循环产生的冗余上下文；

**🔧 技术方法**

技术包括基于提示特征的预测估计、AST 与依赖图的 churn 计算、基于 Knapsack 的最优归档选择、可逆归档实现（byte‑exact 存档与稀疏 skeleton）、以及使用 Beta-Bernoulli 推断的 RDM 归档阈值；

**📊 数据集**

使用在实验平台上生成的合成编程任务仓库，运行 70 条多轮 agentic coding 任务，覆盖 7 种 OpenAI GPT 版本；

**📈 对比分析**

与保留全部、滑动窗口截断、MemGPT 风格压缩等基线相比，Blast Radius 在保持 100% 任务成功率的同时，平均节省 17–26% token（按模型差异可达 20%），并将上下文溢出事件降至最低；

**⚠️ 局限性**

局限性包括：实验基于合成工作负载，缺乏真实项目验证；归档阈值和预测模型目前使用硬编码规则，未来可通过学习提升；跨模型或跨供应商的适用性尚未充分评估。

---

## 509. SABRE: Scalable and Automated Benchmarking of VLMs under Stress

**arXiv ID:** 2608.07435 | [PDF](https://arxiv.org/pdf/2608.07435v1)

**作者:** Zixuan Lan `[一作]` (University of Chicago), Jiawei Zhou `[通讯]` (Stony Brook University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个可扩展、模块化的视觉语言模型（VLM）压力测试构建管道；

**💡 创新点**

通过自然语言任务描述自动生成结构化规范，利用图像生成与编辑技术构造图像，结合模型过滤和人工验证实现快速生成多种压力测试；

**🔧 技术方法**

核心技术包括LLM（如GPT‑5.4）生成规范、FLUX.2/ Gemini 3.1 Flash 等图像生成与编辑模型、VLM 作为过滤器（Gemini 3.5 Flash）、人工验证与局部图像修复工具；

**📊 数据集**

使用自动生成与编辑图像（600张）以及20张真实图像进行 Attribute 子集验证，构建六个前沿 VLM 的评测案例；

**📈 对比分析**

在 -Prior 基准上，六个前沿 VLM 的宏平均准确率仅为 17.8%–31.3%，显著低于现有基准（如 PhD‑CCS、VLind‑Bench 等），显示出新弱点；视觉增强方法（VCD、SoM）在该基准上无显著提升；

**⚠️ 局限性**

局限性在于基准依赖于固定任务规格，人工验证成本高，生成图像质量可能影响有效性，且难以覆盖所有潜在弱点。

---

## 510. Conformal Coverage Guarantees for Any Video Temporal Grounder

**arXiv ID:** 2608.07434 | [PDF](https://arxiv.org/pdf/2608.07434v1)

**作者:** Aseel Mohamed `[一作]` (Texas A&M University), Hasan Kurban `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后期校准的模型无关包装器，给任何视频时序定位模型（包括训练好的本地化器和黑盒视频‑语言模型）加上分布无关、有限样本的覆盖率保证，即输出包含真实事件的时间区域。

**💡 创新点**

创新点在于：①将 conformal prediction 迁移到连续视频事件边界，使用临界量化的非合规性得分实现覆盖率保证；②设计两类非合规性得分（基于区间宽化和超水平集），并证明在该一参数族中校准后的区域长度最优；③不需要重新训练、白盒访问或额外参数，保持模型无关性。

**🔧 技术方法**

技术主要包括：split‑conformal calibration、区间宽化得分、超水平集得分、风险控制、Mondrian（分层）校准、权重加权 conformal 以应对分布偏移，及相关理论证明。

**📊 数据集**

使用公开基准数据集 Charades‑STA、ActivityNet‑Captions、QVHighlights；测试基准模型包括 DETR‑风格本地化器 QD‑DETR 与黑盒视频‑语言模型 Qwen2.5‑VL‑7B‑Instruct；在实验中还评估了 SRAM 等额外基线。

**📈 对比分析**

通过覆盖率有效性、效率（区域长度）比较、误检率风险控制、条件覆盖率、跨数据集迁移等指标评估；结果显示校准后覆盖率紧贴目标，效率优于手工设定的边缘，风险控制达标，分层校准能恢复子群体覆盖率，跨数据集重加权能显著弥补分布偏移带来的覆盖下降。

**⚠️ 局限性**

局限性包括：仅保证边缘覆盖率（不具备无条件覆盖），需满足校准样本与测试样本可交换；在同一视频内部存在时间依赖时覆盖率会下降；跨数据集迁移受正性限制；效率受原始模型性能影响；校准集需要足够大小（≈20 视频），否则覆盖置信区间变宽。

---

## 511. Cloud-Boosted Low-Compute Multi-Channel Speech Enhancement

**arXiv ID:** 2608.07423 | [PDF](https://arxiv.org/pdf/2608.07423v1)

**作者:** Xulin Fan `[一作]` (University of Illinois Urbana-Champaign), Buye Xu `[通讯]` (Meta Reality Labs Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种协同语音增强框架，将服务器端模型的多层特征与延迟输出用于增强边缘设备的轻量化模型

**💡 创新点**

创新点在于三种协同机制：延迟服务器输出条件、层级特征提升和协同多通道维纳滤波，充分利用服务器的空间统计并克服通信延迟

**🔧 技术方法**

采用TinyGRU+MCWF为边缘模型，SpatialNet为服务器模型，FiLM实现特征调制，协同维纳滤波融合统计量

**📊 数据集**

使用DNS-Challenge数据集，结合Pyroomacoustics模拟8通道圆形麦克风阵列，生成不同SNR的标准和挑战集

**📈 对比分析**

通过与基线TinyGRU+MCWF、TinyGRU-Large以及服务器端全模型对比，显著提升SI‑SDR（标准+3.77 dB、挑战+3.49 dB）并仅增加1.5%参数

**⚠️ 局限性**

局限在于仍需网络连接与延迟控制，且实验仅在单一麦克风阵列和合成环境下验证，实际环境中的鲁棒性尚待进一步研究

---

## 512. Topology Inference for Immune System Networks by Using Cell Amount Data

**arXiv ID:** 2608.07403 | [PDF](https://arxiv.org/pdf/2608.07403v1)

**作者:** Yushan Li `[一作]` (KTH Royal Institute of Technology), Petter Brodin `[通讯]` (Karolinska Institutet)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个能够同时满足免疫细胞数量非负、比例收敛和权重三种属性的非线性拓扑模型，并基于此提出了一种受限二次规划的拓扑识别方法。

**💡 创新点**

创新点在于：①提出了符合免疫细胞相互作用特点的非线性模型；②推导了模型可满足三大属性的充分条件；③利用Hilbert度量和非线性Perron‑Frobenius理论证明模型收敛；④将拓扑识别转化为带L1正则化的凸QP。

**🔧 技术方法**

核心技术包括：非线性动力学建模、Hilbert投影距离与Birkhoff收缩定理、非线性Perron‑Frobenius理论、受限二次规划与L1稀疏正则化。

**📊 数据集**

使用了两组数据集：一是5节点人工网络的仿真数据；二是包含10种免疫细胞、10个实验组的真实细胞耗竭实验数据（Forlin等人未公开实验）。

**📈 对比分析**

在仿真实验中，用误差指标E和符号一致率R衡量识别性能；当数据对数m≥5时识别精度几乎完美；在真实实验中，平均相对预测误差约为0.33，绝大多数误差低于0.3。

**⚠️ 局限性**

局限性包括：需要先验的比例向量和状态下界β；模型对参数设定敏感；真实免疫网络的真值未知，难以完全评估识别质量；在数据极其稀缺时识别稳定性仍待验证。

---

## 513. A Formalization of the Laplace Transform and Its Inversion in Lean 4

**arXiv ID:** 2608.07384 | [PDF](https://arxiv.org/pdf/2608.07384v1)

**作者:** Daniel Goldberg `[一作]` (Technion -- Israel Institute of Technology), Antoine Vinciguerra `[通讯]` (Technion -- Israel Institute of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文使用 Lean 4 形式化证明了拉普拉斯变换的定义、基本运算规则、Bromwich 型逆变换以及谐振子方程的拉普拉斯域解。

**💡 创新点**

创新点包括：① 在 Lean 4 环境下实现全新的泛化核与特化到实轴的抽象层次；② 采用实变积分与 Dirichlet 积分的组合，避免了传统复积分与留数定理的复杂性；③ 在形式化框架中完整证明了逆变换的点值恢复性质。

**🔧 技术方法**

技术手段：Lean 4 编码、mathlib 的 Bochner 积分与测度论工具、Fubini、主导收敛定理、Dirichlet 积分、复数指数与指数衰减分析。

**📊 数据集**

无数据集，全部基于数学公理与形式化证明。

**📈 对比分析**

论文未进行实验比较；其“性能”可视为在形式化证明中所需的证明长度与可读性，已实现约 2,900 行 Lean 代码，证明过程透明可复核。

**⚠️ 局限性**

局限性：尚未覆盖卷积定理、非实值函数的泛化、以及更广泛的复杂频域操作；所需的测度与积分理论仍处于扩展阶段。

---

## 514. Trajectory-Relative Hindsight Distillation for Agentic Reinforcement Learning

**arXiv ID:** 2608.07371 | [PDF](https://arxiv.org/pdf/2608.07371v1)

**作者:** Haoyu Zheng `[一作]` (Zhejiang University), Wenqiao Zhang `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轨迹相对的回溯蒸馏框架 TRIAL，用以在多轮交互中将稀疏终点奖励拆解为密集的、基于 token 的监督信号。

**💡 创新点**

创新点在于利用签名对数概率差异生成局部监督，并通过轨迹归一化得到一个均值为一的轮次加权分布，将监督在不同决策轮次之间重新分配，以最大化对重要决策的更新力度。

**🔧 技术方法**

核心技术包括基于 GRPO 的优势计算、对数概率差距的裁剪与分离梯度、轮次加权分配规则、以及密集的 token 级回溯蒸馏损失，全部在一次训练步骤内完成。

**📊 数据集**

实验使用了两个文本交互式环境 WebShop 和 ALFWorld，分别在 Qwen2.5‑3B 和 Qwen3‑1.7B 语言模型上进行评估。

**📈 对比分析**

与 GRPO、SERL、SDAR、GRPO+OPSD、RLSD 等方法相比，TRIAL 在所有八个指标组合上均优于 GRPO，并在六个指标中获得最佳或并列最佳成绩；在 WebShop 上成功率从 56.4% 提升至 75.2%，任务得分从 78.7% 提升至 85.7%。

**⚠️ 局限性**

局限性包括：仅适用于文本交互且动作离散的环境；回溯视图仅在训练时使用，部署时不增加额外推理成本；训练时需额外一次前向回溯推理，导致计算开销增加；实验仅使用单一随机种子，缺乏跨种子的统计验证。

---

## 515. SimWAM: A Simple World Action Model for End-to-End Autonomous Driving

**arXiv ID:** 2608.07468 | [PDF](https://arxiv.org/pdf/2608.07468v1)

**作者:** Zongchuang Zhao `[一作]` (Huazhong University of Science & Technology), Xiang Bai `[通讯]` (Huazhong University of Science & Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本工作提出SimWAM，一种在训练阶段利用视频生成作为监督的世界-动作模型，训练出轻量化动作专家，在推理时直接预测轨迹并实现低延迟；

**💡 创新点**

核心创新在于：①通过隔离注意力掩码将动作专家与视频专家解耦，使视频专家仅在训练时提供运动先验；②采用联合流匹配及SDE形式的Flow‑GRPO强化学习，实现对驾驶奖励的直接优化；③模型架构可自由替换任意预训练视频生成器并独立扩展动作专家，保持推理效率；

**🔧 技术方法**

使用技术包括：Diffusion Transformer（视频专家与动作专家），流匹配（flow matching）与SDE（marginal-preserving stochastic differential equation），隔离注意力掩码，LoRA微调，Flow‑GRPO强化学习与NAVSIM的多维驾驶奖励；

**📊 数据集**

主要数据集：NAVSIM（基于nuPlan OpenScene子集）用于训练与评估，nuScenes用于零样本迁移测试；

**📈 对比分析**

与多种最新基线（UniAD、SeerDrive、DriveWAM、Epona、SGDrive等）在单摄像头设置下对比，SimWAM在NAVSIM上获得91.5 PDMS，明显高于其他方法且推理延迟显著更低；

**⚠️ 局限性**

局限性包括：仅使用前置摄像头，缺乏多模态传感器融合；依赖大规模预训练视频模型，训练成本高；RL阶段对计算资源需求大；在复杂多车道或非常规场景下的泛化尚待进一步验证。

---

