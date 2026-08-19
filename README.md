# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-19 | 今日论文总数: 507

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. MobileWorldSafety: Benchmarking GUI Agent Safety Against Environmental Injection Attacks in Android Apps

**arXiv ID:** 2608.17659 | [PDF](https://arxiv.org/pdf/2608.17659v1)

**作者:** Sujin Chen `[一作]` (Shanghai Artificial Intelligence Laboratory), Jing Shao `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MobileWorldSafety 基准，用 142 个风险任务评估移动 GUI 代理在真实 Android 应用中对环境注入攻击的安全性。

**💡 创新点**

创新点在于将真实应用视作攻击载体，提出两阶段评估流程（规则验证+LLM 判定）来客观区分安全失败与能力失败，系统化捕捉日常使用场景中的隐蔽注入。

**🔧 技术方法**

使用了程序化风险指示器、规则引擎和大语言模型评判器，并在 Docker 化 Android 模拟器中自动化执行。

**📊 数据集**

数据集包含 13 个真实 Android 应用（Mail、Calendar、Messages、Files、Mattermost、Mastodon、Maps、Taodian 等）和 5 种注入载体（通信、网页、社区、存储记录、工具响应）。

**📈 对比分析**

对六个代表性代理（Gemini‑3‑Pro、Qwen3.5‑397B‑A17B、Kimi‑K2.5、Claude‑Sonnet‑4.5、GUI‑Owl‑1.5‑32B‑Instruct、MAI‑UI‑8B）进行实验，攻击成功率（ASR）从 40.4% 下降到 66.9%，表明大部分模型在安全性上存在显著缺陷。

**⚠️ 局限性**

局限性包括仅评估 Android 平台、依赖模拟器环境、注入类型有限且未涵盖更复杂的多模态或跨平台攻击，以及防御效果仍不足。

---

## 2. COMIC: Reference-Aware Safety Gating for Multimodal Large Language Models

**arXiv ID:** 2608.17234 | [PDF](https://arxiv.org/pdf/2608.17234v1)

**作者:** Md Abdullahil Oaphy `[一作]` (Kennesaw State University), Honghui Xu `[通讯]` (Kennesaw State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种参考感知的预生成安全门 COMIC，用来在多模态大语言模型执行操作前先检测并阻止潜在的不安全操作。

**💡 创新点**

创新点在于把安全判断从整体 prompt‑image 对象转移到“操作–目标”绑定层面，先推断用户请求的操作类型、视觉引用类型，生成候选目标并进行语义与空间匹配，再对每个候选的操作‑目标对进行安全评分并采用保守的最大风险与置信度路由决定是否放行。

**🔧 技术方法**

使用的技术包括：OCR 与开放词汇视觉提议生成候选目标；规则化的意图推断与引用类型识别；基于语义、空间与布局的定位评分；固定规则的候选安全评分器；置信度统计与阈值路由；整个流程在推理时以规则/管道方式实现，不需重新训练模型。

**📊 数据集**

使用了三个数据集：Localized jailbreak benchmark FigStep、宽泛 jailbreak benchmark JailBreakV-28K 以及正面多模态能力评测集 MM‑Vet；对攻击样本做统一格式化，保留 500+ FigStep、150+ JailBreakV-28K、完整 MM‑Vet 用于评估。

**📈 对比分析**

与无防御原模型、AdaShield、CoCA、Immune、FigStep 等基线进行对比。COMIC 在 FigStep 上将攻击成功率（ASR）几乎降为 0，JailBreakV-28K 的 ASR 也显著下降（平均 93% 降低）；在 MM‑Vet 上保持甚至略优于原模型的能力分数；运行时间仅略增（≤10%），明显低于 CoCA/Immune 的延迟。

**⚠️ 局限性**

局限性：依赖 OCR 与候选提议的召回，若目标区域被漏检或视觉噪声过大则失效；对高密度、模糊或多目标分散攻击时仍可能误判；目前只处理单目标推理，无法覆盖多区域分布式危害；多语言/跨域场景下表现未验证。

---

## 3. General Semantic Knowledge Infusion for Spatio-Temporal Traffic Forecasting

**arXiv ID:** 2608.17440 | [PDF](https://arxiv.org/pdf/2608.17440v1)

**作者:** Mattis thor Straten `[一作]` (Kiel University), Matthias Renz `[通讯]` (Kiel University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出通用管道，将通用知识图谱信息注入交通预测 GNN，构造语义邻接矩阵。

**💡 创新点**

无需手工构建专属 KG，利用通用 KG 自动提取语义上下文并融合进预测模型。

**🔧 技术方法**

使用知识图谱嵌入（ComplEx）生成语义邻接矩阵，结合传统空间邻接矩阵，应用于多种 GNN 基线。

**📊 数据集**

在 San Diego 高速公路数据集（LargeST benchmark）上进行实验。

**📈 对比分析**

通过与原始空间邻接矩阵、随机矩阵对比，发现大多数基线在加入语义邻接后 MAE 下降 5‑15%，表明知识注入提升预测精度。

**⚠️ 局限性**

受限于 KG 覆盖质量、聚合方式单一以及模型对邻接矩阵的敏感性，某些基线（如 DCRNN）效果不佳。

---

## 4. VLCP: Vision Language Control Policy Closed-Loop Code Replanning for Robot Manipulation

**arXiv ID:** 2608.16978 | [PDF](https://arxiv.org/pdf/2608.16978v1)

**作者:** Dhia Naouali `[一作]` (University of Monastir), Omar G. Younis `[通讯]` (Silverstream AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过冻结的视觉语言模型(VLM)在每K步重新查询环境，并重写短Python控制函数，实现单个episode内闭环重规划；

**💡 创新点**

创新点在于把闭环控制的焦点放在控制代码本身，利用VLM即时观察与代码生成，既可纠正失误又可积累技能；

**🔧 技术方法**

使用GPT‑5.5冻结模型、多视角RGB与关节状态作为输入，生成并编译Python控制块，结合prompt缓存与跨episode技能库；

**📊 数据集**

在MuJoCo RoboVerse仿真环境中，针对57个任务（LIBERO‑Object、Kitchen Scenes、Living‑Room Scenes）进行评估；

**📈 对比分析**

与自身的开环（K=T）对照，闭环实现任务成功率从3.5%提升至35.1%（≈10倍），失败把握率提升至27.3%，且无需任何示范数据；

**⚠️ 局限性**

限制在于依赖仿真提供的物体位姿、仅在仿真测试、VLM推理延迟高、评估仅单个episode、对小模型效果差、未验证对真实机器人感知与动力学的鲁棒性。

---

## 5. When to Review: Spaced Repetition for Continual Pre-Training of Language Models

**arXiv ID:** 2608.17530 | [PDF](https://arxiv.org/pdf/2608.17530v1)

**作者:** Alankar Atreya `[一作]` (NatWest AI Research), Raad Khraishi `[通讯]` (NatWest AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了Spaced Repetition Training（SRT），一种基于SuperMemo‑2算法的样本级回顾调度方法，用于持续预训练大型语言模型以防止灾难性遗忘；

**💡 创新点**

创新点在于将持续预训练视为适应性回顾调度问题，利用每个样本的perplexity映射为回忆质量并动态调整复习间隔，实现对旧知识与新知识的高效平衡；

**🔧 技术方法**

技术包括SuperMemo‑2复习状态维护、perplexity‑到‑质量阈值映射、基于到期时间的样本抽样、以及标准的因式语言建模目标与AdamW优化；

**📊 数据集**

使用了时间拆分的维基百科（旧/新）和GitHub代码（旧/新）两大语料库，并在辅助实验中引入MNIST、Fashion‑MNIST、CIFAR‑10和Wine等视觉/表格数据集；

**📈 对比分析**

与无回放的原始CPT、均匀回放以及仅按perplexity挑选的PPL‑Prioritized进行对比，SRT在维基百科和代码QA上分别提升旧知识准确率约23–37个百分点，同时保持或提升新知识学习；在MMLU、BBH、GSM8K、PIQA等广泛能力基准上，SRT保持了原始模型性能，远优于均匀回放导致的显著下降；

**⚠️ 局限性**

局限性包括仅在Llama系列1.1B与3B模型、单语言（英文）上验证，阈值映射需手工调参，实验规模受限于单次训练、缺乏跨模型/多语言泛化证明，以及SRT对推理时性能的影响尚未系统评估。

---

## 6. Uncertainty-Aware Decision Making in Multimodal Large Language Models

**arXiv ID:** 2608.17084 | [PDF](https://arxiv.org/pdf/2608.17084v1)

**作者:** Abderrahmene Boudiaf `[一作]` (Khalifa University of Science and Technology), Sajid Javed `[通讯]` (Khalifa University of Science and Technology)

**通讯引用:** 3474 | [OpenAlex ID](https://openalex.org/A5071515463)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对多模态大型语言模型（MLLM）的不确定性检测、校准、风险控制与行动决策进行系统性综述，并提出源–信号–校准–行动（S–S–C–A）框架；

**💡 创新点**

创新点在于将不确定性视为决策依据，构建决策中心的S–S–C–A链条，将多模态不确定性来源、可观测信号、校准目标、风险控制以及相应行动策略统一分类，并为未来研究提出时间维度的路线图；

**🔧 技术方法**

采用文献综述与框架设计技术，归纳与分类多模态不确定性估计方法（如token/logit、隐状态、采样、语义不一致、定位与归因、可视化验证、verifier/judge、conformal预测等），并结合校准与风险控制策略；

**📊 数据集**

综述涵盖了多模态评测基准与数据集，包括VQA、Visual‑Dialog、MMBench、MOHO、OMD、MMI、ChartQAP、医学图像QA、机器人/嵌入式问答、音视频QA、文档/图表理解等；

**📈 对比分析**

本文未进行统一实验对比，而是提出多维度评估方法（如ECE、AUC、风险-覆盖曲线、conformal覆盖率等），并讨论各方法在不同任务与不确定性来源上的优劣；

**⚠️ 局限性**

局限性包括：综述时间有限，最新研究可能未纳入；多模态覆盖偏向图像-文本，音频、视频、图表、文档等领域研究不足；缺乏统一评估标准和跨模型可比性；黑盒访问与域移位下的校准与动作设计仍面临挑战。

---

## 7. Self-Bounding Regret Matching+ in Potential Games and Product-Simplex Optimization

**arXiv ID:** 2608.17417 | [PDF](https://arxiv.org/pdf/2608.17417v1)

**作者:** Pahan Dewasurendra `[一作]` (Johns Hopkins University), Subhashini Jayawardhana `[通讯]` (San Diego Miramar College)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

推导了RM+的精确能量守恒律，给出了其在在线学习、潜力游戏与光滑优化中的误差和收敛性质的完整分析，并证明了常数遗憾与O(ε⁻²)收敛率。

**💡 创新点**

引入了“精确一阶守恒律”和“尖锐的状态增长界”来实现自我界定的误差控制，解决了潜力游戏中RM+长期遗憾不确定的问题，并在无参数、可缩放的条件下获得了δ‑无关的O(ε⁻²)速率。

**🔧 技术方法**

采用了变分自适应的在线学习分析、光滑目标的梯度下界推导、状态正交截断的几何不等式、周期性块更新的路径长度控制以及潜力函数的telescoping归约等技术。

**📊 数据集**

在人工生成的图形潜力游戏与密集非凸目标实验中对RM+与预测性、光滑的额外梯度变体进行了对比评估。

**📈 对比分析**

与预测性RM+、光滑Extra‑Gradient等基准对比，RM+在保持无学习率、参数自由的同时在实验中展示了更快的收敛速度和更稳定的性能。

**⚠️ 局限性**

结果仅适用于单独或循环块更新，无法覆盖并发更新；在随机或带噪的反馈下无法保证正前向增益；对非常小的初始正状态δ仍会导致常数放大。

---

## 8. Oracles That Cannot Fail: Anchoring and the Expectation That Moves With the Fault

**arXiv ID:** 2608.17214 | [PDF](https://arxiv.org/pdf/2608.17214v1)

**作者:** Arquimedes Canedo `[一作]` `[通讯]`, Arquimedes Canedo

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探讨了一种测试oracle的缺陷，称为锚定（anchoring），即期望值来自被评估的系统本身，导致无法检测到故障。通过对一个部署的空中交通控制模拟器进行突变测试，研究了不同类型的锚定对故障检测的影响。

**💡 创新点**

创新点在于提出了锚定的概念，并通过实验验证了锚定如何影响测试oracle的有效性，特别是如何通过重新锚定来提高故障检测能力。

**🔧 技术方法**

使用了突变测试技术，特别是通过对366个突变体进行分析，评估不同oracle的检测能力。

**📊 数据集**

使用的数据集是一个空中交通控制模拟器，包含12个无模型属性套件，涉及4个模块和366个突变体。

**📈 对比分析**

与手写测试进行比较，发现无模型属性套件在检测能力上仅比手写测试多检测到3个突变体，但在每个测试的效率上高出6到33倍。通过重新锚定，检测能力显著提高。

**⚠️ 局限性**

限制在于所有测量均来自同一系统和同一作者，可能影响结果的普遍性和适用性。此外，锚定的影响可能在不同的上下文中表现不同。

---

## 9. SignalReasoner: Assessing the Upper Bound of 3B Models for Signal Mathematical Reasoning

**arXiv ID:** 2608.17301 | [PDF](https://arxiv.org/pdf/2608.17301v1)

**作者:** Guozheng Sun `[一作]` `[通讯]`, Guozheng Sun

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在Qwen2.5-3B上进行领域特定的监督链式思维（CoT）微调和随后强化学习，提升了无线信号数学推理性能。

**💡 创新点**

创新点在于将域感知CoT微调作为RL的初始化，并比较GRPO、GSPO、GMPO三种群组策略，在无线信号推理任务上显著提高准确率并揭示“推理缩短”现象。

**🔧 技术方法**

使用的方法包括监督链式思维微调、基于群组的强化学习（GRPO、GSPO、GMPO）以及可验证奖励函数。

**📊 数据集**

实验数据集为包含4,027道无线信号数学问题的WirelessMathBench Benchmark。

**📈 对比分析**

通过将基线、SFT、RL三者组合在不同算法下对比，B‑GMPO在所有设置中获得最高准确率39.12%，并显著降低生成长度。

**⚠️ 局限性**

主要限制是仅以最终答案为奖励导致模型偏向简短推理，缺乏对中间推理过程的监督，从而牺牲了可解释性和错误诊断能力。

---

## 10. Elimination Geometry

**arXiv ID:** 2608.17646 | [PDF](https://arxiv.org/pdf/2608.17646v1)

**作者:** Mian Huang `[一作]`, Xueqin Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文提出了一套基于消除几何（elimination geometry）的结构化可实现性审计框架，用以识别并量化在共享部署约束下从局部最优到全局实现的障碍，并给出四层风险分解（模型误差、架构障碍、泛化误差、实现误差）以及相应的干预策略。

**💡 创新点**

创新点在于：
• 将消除产生的“原生损失”（native defect）作为度量标准，确保所有后续误差都以与目标目标相同的单位量化；
• 构造四层风险分解，清晰区分局部模型、共享架构、样本有限和实现误差；
• 提出“架构障碍”（architecture obstruction）概念，并给出可计算的一阶或多阶上界；
• 通过“精确化”（exactification）和“投影修复”（projection repair）等技术实现原生损失的可测性和可修正性；
• 引入Hodge分解与曲率度量对消除顺序依赖性进行审计，提供局部可检验的可整合性条件。

**🔧 技术方法**

主要技术方法包括：
• 代数消除与Legendre共轭生成Bregman散度；
• 原生损失与架构障碍的最小化与infimal卷积（Bellman / min-plus）分解；
• 精确化（jet-subtraction）以保持目标一阶/二阶接触；
• 投影修复（Hilbert投影到CDF集合）保证概率合法性不增损失；
• Hodge分解与曲率检测保证多重消除路径一致性；
• 统计证书与置信序列构造，结合PAC‑Bayes或均匀一致性界定。

**📊 数据集**

在实验部分使用了：
• 简单的二点示例（X∈{0,1}）和一阶线性模型；
• 经典的MNIST数据集进行“前瞻冻结”审计，以检验一次性预测与局部细化的性能；
• 生成式图模型的两个顶点案例，用来展示S²和S的方差分析；

**📈 对比分析**

比较方法主要是与已知的共享架构方法（如变分自编码器、多任务学习、序贯决策网络、一次性预测）在理论层面的对比，指出传统方法无法捕捉的架构障碍；实验层面通过MNIST例子展示在保持同一原生损失尺度下，一次性类的残差被局部细化显著降低。性能方面，论文强调在相同的损失单位下的可解释性和可修复性，而非直接的数值性能提升。

**⚠️ 局限性**

局限性包括：
• 只给出了理论框架与有限的示例验证，缺乏大规模真实数据上的广泛实证；
• 架构障碍的数值评估往往需要可计算的上界，实际问题中可能难以获取；
• 该框架假设已知完整的局部最优结构，若局部模型自身存在误差，架构障碍可能被高估；
• 对于非凸或非平滑目标，原生损失与Bregman散度的直接等价可能失效；
• Hodge分解与曲率检测需要对消除顺序有完整描述，在复杂管线中实现困难。

---

## 11. Towards Better Agents for Multi-Turn User Interaction: The Next User Turn Is More Than Context

**arXiv ID:** 2608.17499 | [PDF](https://arxiv.org/pdf/2608.17499v1)

**作者:** Yiwen Zhao `[一作]`, Wei Wu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种利用用户后续回复作为局部奖励信号的信用分配方法，用于改进多轮工具使用型语言代理的强化学习训练

**💡 创新点**

创新点在于将用户回复（如确认、质疑、补充）映射为三元情感标签，进行局部归一化后与终端奖励相加，实现无需额外评论器或额外rollout的局部信用分配

**🔧 技术方法**

方法基于Interactive GRPO框架，加入了反馈感知信用分配（FBAC），使用Qwen3 8B/14B模型与Frozen DeepSeek-V4-Flash用户模拟器训练

**📊 数据集**

评估使用了τ-bench系列（Airline、Retail、Telecom、Bank等九个域）、Pare-Bench和Co-Gym等公开基准数据集

**📈 对比分析**

与仅使用终端奖励的Interactive GRPO进行对比，FBAC在9个域的平均Pass@1提升5.9点（8B）和10.2点（14B），在Telecom域表现最显著，并在Pare-Bench和Co-Gym的零样本转移中保持相对优势

**⚠️ 局限性**

局限性包括依赖深度仿真用户模拟器的私有策略标签，未验证在真实用户或不同模拟器环境下的效果，且在部分域（如Co-Gym的Travel）未见显著提升

---

## 12. TEAMS: Text-prompted spatiotEmporal dual-heAd Mamba Snake

**arXiv ID:** 2608.17421 | [PDF](https://arxiv.org/pdf/2608.17421v1)

**作者:** Ruicheng Zhang `[一作]` (Sun Yat-sen University), Shuo Li `[通讯]` (Case Western Reserve University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 TEAMS，一个结合文本提示、时空双头 Mamba Snake 的医学图像实例分割框架

**💡 创新点**

创新点包括：1) SSES 通过双向空间与历史时空状态空间建模提升蛇形演化；2) CMAM 用形态感知的结构状态空间双向增强细节分辨；3) TCDHS 通过文本提示和双头一致性反馈纠正检测错误

**🔧 技术方法**

使用了 Mamba 及其改进版 Mamba2 作为核心状态空间模块、交叉模态自注意力、ClinicalBERT 文本编码、卷积与循环门控对比实验

**📊 数据集**

在 MR_AVBCE‑Extended、VerSe、RAOS、BTCV、PanNuke 五个多模态多器官数据集上进行评估

**📈 对比分析**

与 14 个语义分割和深度蛇模型相比，TEAMS 在 mDice/mBF 等指标上均实现了显著提升（如在脊柱数据集 mDice 提升 6.9%/mBF 9.1%），并在多器官场景中保持最高或次高性能

**⚠️ 局限性**

主要限制包括：对标注一致性敏感，文本提示质量直接影响性能，且模型在极端模糊或极小目标时仍易出现错误

---

## 13. Calibrated Predictive Safety for Heterogeneous Robots: An Action-Conditioned JEPA Framework with Model-Based Safety Shields

**arXiv ID:** 2608.17496 | [PDF](https://arxiv.org/pdf/2608.17496v1)

**作者:** Kaiming Zhong `[一作]` (Guangdong Bifang Intelligent Control Technology Co., Ltd.), Yue Wang `[通讯]` (Guangdong Bifang Intelligent Control Technology Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在异构机器人上构建了一个基于动作条件的联合嵌入预测架构（JEPA）与硬性安全屏障相结合的决策框架，实现了可部署的安全筛选。

**💡 创新点**

创新点在于将学习到的风险预测与可证明的安全屏障分离，提供真正的安全保证，并通过校准损失让风险预测具备可操作的概率阈值。

**🔧 技术方法**

使用冻结的视觉编码器、动作条件Transformer预测器、风险与进度头、温度缩放、蒙特卡洛Dropout及共形预测等技术。

**📊 数据集**

主要使用LIBERO‑Long仿真数据集进行训练和评估，也引用DROID真实数据集验证跨平台泛化。

**📈 对比分析**

通过与无筛选、规则屏障、模型屏障、JEPA无屏障以及完整框架等基线比较，完整框架在闭环仿真中成功率提升7点，碰撞误报率从0.21降至0.14，校准误差下降3.5倍。

**⚠️ 局限性**

局限性包括安全保证仅覆盖已建模的约束，依赖于提议器生成的候选质量，对模型预测的误差和标签噪声敏感，并受实时延迟与分布漂移影响。

---

## 14. KeyPooling: Measuring Where LLM API Relay Paths Collapse Prompt Cache Isolation

**arXiv ID:** 2608.17485 | [PDF](https://arxiv.org/pdf/2608.17485v1)

**作者:** Bowen Sun `[一作]` (Johns Hopkins University), Chaowei Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化 LLM API 代理（relay）在共享缓存（prompt cache）时的身份泄露，提出 KeyPooling 测量方法，跟踪身份从下游客户到最终缓存查询与写入的完整路径，并通过源代码跟踪、运行时观察和针对性干预定位导致身份丢失的机制；在五个开源 gateway、OpenRouter 以及独立部署的生产环境中进行实验；进一步评估了在可控条件下可否利用缓存反馈进行逐令牌恢复，并给出可部署的防御方案。

**💡 创新点**

① 关键贡献在于将缓存隔离视为最终路径属性，提出可追踪并定位身份丢失的完整测量框架；② 开发了基于 HMAC 的可追踪命名空间（namespace）实现，验证其在多跳、重试、回退、池化及嵌套 relay 中的有效性；③ 在实验中首次展示了在生产路由上可实现 16 位可控序列的逐令牌恢复；④ 给出了从技术层面到运营层面的完整防御合约。 这些都填补了现有工作仅关注单点或单侧缓存泄露、缺乏完整路径追踪和防御建议的空白。

**🔧 技术方法**

核心技术包括：
- 源代码追踪（source tracing）+ 运行时观察（runtime observer）+ 逐组件干预（matched interventions）
- 基于 HMAC 的不可伪造命名空间（namespace）生成
- 对 OpenAI、Anthropic、Gemini 等多家提供商的缓存生命周期建模与接口测试
- 逐令牌恢复实验的候选生成与评分框架
- 公开提示前缀、BIP‑39 字典、WildChat 对话等多种数据源的使用
- OpenRouter 标签帧（label frame）和多路径并行测评。

**📊 数据集**

主要使用的数据集与实验材料有：
- 公共代理前缀（public agent prompts）约 17 条（含 Claude、Cursor、OpenCode 等）
- WildChat 对话集（用于候选覆盖与局部实验）
- BIP‑39 2048 词表（用于枚举恢复实验）
- OpenRouter 每日排名前 50 模型的标签帧（用于生产实验）
- 29 个开源 gateway 的冻结源码与可执行版本（含 NewAPI、One API、MetaAPI 等）
- OpenAI、Anthropic、Gemini 官方 API（用于缓存生命周期测试）。

**📈 对比分析**

实验方法：
- 对每条路径执行四请求单元（strict positive / controlled miss）并记录缓存计数；
- 在多跳、重试、池化和嵌套场景下单一组件干预，验证其对边缘产生的影响；
- 在 OpenRouter 上使用 28 条高流量标签进行 80% 体量覆盖的并行测评；
- 进行 8 步逐令牌恢复实验，需 258 次请求，恢复 16 位信息；
- 通过对比独立凭证与共享凭证的实验验证池化与命名空间效果。 
性能上：
- 发现 5/5 开源 gateway 默认未绑定上游凭证，导致 100% 的客户间缓存交叉；
- OpenRouter 上 5/5 受托凭证路由产生交叉，独立凭证路由 0/5；
- 逐令牌恢复实验在单条生产路由上成功，恢复 16 位；
- 但在更大规模实验中受候选覆盖、报告粒度和路由不稳定等因素限制，恢复成本从 2 万至 13 万次请求。

**⚠️ 局限性**

主要局限：
- 采样范围有限，OpenRouter 标签帧与独立 relay 实验均基于单账户对，难以给出整体曝光比例；
- 逐令牌恢复仅在极端可控环境下可行，无法推广到天然语言或高多样性内容；
- 防御方案（HMAC namespace）在当前主流提供商未公开 namespace 原语，需运营方自行实现与验证；
- 评估未覆盖时延、计费差异、错误分布等完整 transcript 隐私维度；
- 结果受路由选择、重试策略、缓存淘汰等动态因素影响，实验环境与生产实际可能存在差异。

---

## 15. From Substitution to Scaffolding: Breaking the Self-Reinforcing Harm Cycle of AI in Education (and Beyond)

**arXiv ID:** 2608.17451 | [PDF](https://arxiv.org/pdf/2608.17451v1)

**作者:** Lucile Favero `[一作]` (ELLIS Alicante), Nuria Oliver `[通讯]` (ELLIS Alicante)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对49篇IB学生论述性论文的定性分析，构建了AI在教育中的误配与自我强化伤害循环框架，并提出“以支架而非替代”设计原则。

**💡 创新点**

创新点在于将认知、能动性、情绪与伦理四维风险整合成自我强化伤害循环，并通过学生视角验证，提出跨领域普适的支架设计原则。

**🔧 技术方法**

技术主要为定性编码与主题分析，并未开发新模型；但框架可指导AI系统在教育场景中实现支架功能。

**📊 数据集**

使用的数据集为来自瑞士三所德语学校的49名IB学生在AI使用前后撰写的论述性论文。

**📈 对比分析**

无传统量化比较；作者仅报告描述性统计（如80%认知削弱、53%支持支架等）并对比学生期望与现有AI工具。

**⚠️ 局限性**

局限在样本规模小、情境单一、缺乏跨文化和不同教育水平验证，且未通过实验验证支架原则的有效性。

---

## 16. REChart: Reasoning-Efficient Chart Editing with Large Reasoning Models

**arXiv ID:** 2608.17414 | [PDF](https://arxiv.org/pdf/2608.17414v1)

**作者:** Yuanbang Liu `[一作]` (Hong Kong University Of Science And Technology), Wei Zeng `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个两阶段训练框架 REChart，结合有监督的 Reason‑Score‑Refine 轨迹与强化学习，提升大型推理模型在图表编辑任务中的编辑准确性与推理效率。

**💡 创新点**

提出了基于多代理 Reason‑Score‑Refine 轨迹生成的结构化推理训练，以及结合效率奖励的混合奖励设计，显著缓解过度推理导致的幻觉与冗余问题。

**🔧 技术方法**

采用多角色代理工作流生成 200k 结构化推理轨迹，使用 LLM 如 Qwen3-VL-32B 进行推理与代码生成；通过 GRPO 强化学习结合代码相似度、结构相似度与视觉相似度的多维度 fidelity 奖励和片段级效率奖励进行模型优化。

**📊 数据集**

构建了约 709k 图像-指令-代码三元组数据集，包含 521k 高质量图表代码对及 567k 编辑样本，并在 ChartEdit 与 ChartMIMIC 两大基准上评测。

**📈 对比分析**

与公开基准上大小相近的开源模型和多家专有模型对比，REChart 在 ChartEdit 和 ChartMIMIC 任务上分别取得 79.71/70.09/77.15 的整体分数，领先所有开源同规模模型；同时在最大 16,384 token 预算下，推理 token 平均消耗降低 79% 以上。

**⚠️ 局限性**

仅在 8B 模型上实验，缺乏对更大规模 LRM 的验证；实验聚焦图表编辑，未探讨迁移至其他视觉代码生成任务。

---

## 17. TINA+: Probing Residual Visual Knowledge in Unlearned Diffusion Models via Diffusion-Consistent Text-Free Inversion

**arXiv ID:** 2608.17747 | [PDF](https://arxiv.org/pdf/2608.17747v1)

**作者:** Qianlong Xiang `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种针对文本至图像扩散模型的概念消除安全评估方法，通过无文本逆向探测残留视觉知识。

**💡 创新点**

提出了 TINA+，一种基于固定点优化和扩散一致性正则的文本无关逆向攻击，能够有效检验概念消除是否真正删除了视觉知识。

**🔧 技术方法**

采用了 DDIM 逆向、固定点一致性优化、能量正则化（基于前向扩散边缘能量）以及前向边缘初始化等技术。

**📊 数据集**

在多种概念消除任务上使用了裸照、Van Gogh 风格、四种对象（教堂、垃圾车、降落伞、鳟鱼）以及三位名人（Taylor Swift、Elon Musk、Adam Lambert）的图片集合。

**📈 对比分析**

与十二种现有概念消除方法和五种文本攻击方法对比，TINA+ 在所有任务上均实现了 90% 以上的攻击成功率，显著优于文本攻击，表明模型仍保留视觉知识；在随机初始化模型中亦能抑制伪逆向误报。

**⚠️ 局限性**

主要限制在于需要访问模型权重进行优化；在极强文本抑制下能量正则仍可能未能完全排除所有奇异路径；以及对非常小或高维概念的检测可能受限。

---

## 18. GxP-Agent: Process-DAG Topology for Reliable Clinical Trial Programming with LLM Agents

**arXiv ID:** 2608.16890 | [PDF](https://arxiv.org/pdf/2608.16890v1)

**作者:** Jaime Yan `[一作]` (Harrisburg University of Science and Technology), Jaime Yan `[通讯]` (Harrisburg University of Science and Technology)

**通讯引用:** 161 | [OpenAlex ID](https://openalex.org/A5107920691)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GxP-Agent多智能体系统，将临床试验编程任务拆分为按领域知识预先定义的有向无环图（DAG），实现对CDISC ADaM数据集的自动生成。

**💡 创新点**

创新点在于用预先编码的DAG结构代替LLM自我规划，结合节点级提示、模式化指令、验证门和重试机制，使单个LLM只负责生成局部R代码，从而显著提升合规性和可审计性。

**🔧 技术方法**

使用的技术包括大语言模型（Claude Sonnet/Opus、GPT‑4.1/4o、Gemini 2.5 Pro）、LangGraph图形框架、R工具链（admiral、metacore、metatools、xportr）以及自定义验证脚本。

**📊 数据集**

使用FDA公开的CDISCPilot01数据集构建的CDISC‑Bench，涵盖ADSL（254记录、49变量）和ADAE（1,191记录、55变量）等ADaM数据集。

**📈 对比分析**

通过与单发式生成、平面多智能体、关键词/嵌入检索增强的基线以及不同LLM的对照，评估结构匹配率、执行成功率和运行时开销；在ADSL任务中，Claude模型在DAG架构下达100%结构匹配，GPT‑4.1仅59.2%，而所有非DAG架构均为0%。

**⚠️ 局限性**

局限性包括：需要手工设计DAG节点（耗时≈40小时）；DAG需要多次LLM调用，运行时较长；仅在CDISCPilot01上验证，未检验跨研究的通用性；检索增强基线可能不足，未来可进一步强化。

---

## 19. The Brazilian Vaccination Debate on YouTube: Topics, Perspectives, and Engagement Dynamics

**arXiv ID:** 2608.17502 | [PDF](https://arxiv.org/pdf/2608.17502v1)

**作者:** Matheus S. Azevedo `[一作]` (Federal University of Ouro Preto), Carlos H. G. Ferreira `[通讯]` (Federal University of Ouro Preto)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2018‑2024年巴西YouTube关于疫苗的评论进行多层次文本与参与度分析，探讨主题、立场与时间演化。

**💡 创新点**

首次将语义主题、评论立场与互动动力整合到时间维度，揭示疫情后话题持续性与对立立场的分布。

**🔧 技术方法**

使用BERTopic（BERTimbau嵌入+UMAP+HDBSCAN）进行主题挖掘，使用基于LLama 3.1的立场分类模型，并结合YouTube元数据进行交互分析。

**📊 数据集**

1,276,730条巴西葡语YouTube评论，覆盖13,455个视频，3,647个频道，时间跨度2018‑2024年。

**📈 对比分析**

通过宏观主题与立场比例、互动指标（评论/视频、回复/评论、点赞/评论）比较不同时间段，发现疫情期间参与度激增且反对立场在阴谋/政治主题中占比最高；后疫情期仍保持高健康经验与信息可信度议题。

**⚠️ 局限性**

仅关注YouTube，未跨平台或多模态；主题聚类与立场分类仍存在不确定性；宏观聚合简化复杂话语，可能遗漏细节。

---

## 20. The Oracle of Chemnitz: An interactive art installation to reanimate old things in a garage featuring a rotary phone

**arXiv ID:** 2608.17407 | [PDF](https://arxiv.org/pdf/2608.17407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 21. PathoArgus: Advancing Evidence-Grounded Long-Context Visual Reasoning across Gigapixel Whole-Slide and Multi-Slide Case Contexts

**arXiv ID:** 2608.17607 | [PDF](https://arxiv.org/pdf/2608.17607v1)

**作者:** Bowen Liu `[一作]` (Hong Kong University of Science and Technology), Xiaomeng Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一个完整病例关联的WSI问答评估框架（PathoArgus-Bench）和一种固定预算的视觉证据检索器（PathoArgus），用以评估多模态大型语言模型在完整病理切片上下文中的证据驱动推理能力。

**💡 创新点**

创新点在于：①把完整WSI上下文压缩到固定视觉预算的过程分解为候选集→切片→空间区域→像素四级路由，显式保持结构和覆盖；②设计了ESG四分量控制实验，以检验模型预测是否随视觉证据变化；③构建了覆盖六大病理能力的海量问答数据集，提供统一的性能衡量指标。

**🔧 技术方法**

使用技术包括：多模态LLM（如GPT-5.6、Qwen系列）、基于特征向量的视觉输入（CONCH patch features）、固定预算的前置选择器、候选集与空间覆盖的路由算法，以及对比实验中对多种公开模型的微调与微调用法。

**📊 数据集**

数据集为TCGA项目的5,516张WSI，构成22,078个四选项问题，划分为15,702/2,095/4,281的训练/验证/测试集，并包含483组ESG四分量（每组4个证据状态）。

**📈 对比分析**

对比方法包括20个预训练及微调的多模态LLM、传统的WSI特征接口以及PathNavigate/PathAgent等导航策略；性能指标为总体准确率（Overall）和ESG四分量全正解率（QExact）。在测试集上，GPT-5.6最高Overall为57.09%，但QExact仅3.93%；PathoArgus读取器总体准确率为50.39%，ESG准确率为46.17%，但QExact仅1.86%。

**⚠️ 局限性**

局限性在于：①固定视觉预算的选取器仍无法保证所有关键证据被保留，导致预测与证据不完全一致；②模型在ESG四分量中的高错误率表明缺乏对证据变动的敏感性；③评估只涵盖六类病理能力，未涵盖所有临床情境；④缺乏动态检索或在线增量学习机制，限制了模型在大规模WSI集中的可扩展性。

---

## 22. Where a New Concept Must Enter: Entry Point Gates Cross-Task Usability in Unified Multimodal Models

**arXiv ID:** 2608.17564 | [PDF](https://arxiv.org/pdf/2608.17564v1)

**作者:** Zongyang Qiu `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将新的视觉概念只通过生成或理解单一方向绑定到统一多模态模型中，测量另一方向的表现，评估跨任务通道的可用性。

**💡 创新点**

提出“入口点”规则：概念绑定在共享计算的中层位置即可被两方向共享；发现生成侧需语义格式共享；提出低成本的中层语义锚定方法，避免生成梯度损失。

**🔧 技术方法**

使用LoRA调优、闭式激活编辑、对齐损失（InfoNCE）以及语义地址检索（SAR）等技术；通过对齐探针预测导出。

**📊 数据集**

采用Objaverse 3D资产的60视角渲染图与伪词生成56个概念；使用多模态基准（如POPE、Ablation等）和文本到图像评测集（GenEval）。

**📈 对比分析**

以匹配与生成名称的多项选择、上下文自由产出和身份检索为指标进行比较；传统生成目标导致41%性能损失，而中层锚定仅损失0.1%，并在不同架构下保持可用窗口。

**⚠️ 局限性**

入口窗口依赖理解路径使用语义编码器；仅测试8个概念，规模和人类评测未覆盖；模型共享权重与表示格式的因果关系尚未完全分离。

---

## 23. Education-centered critical policy analysis of AI: Ghana's AI strategy as a case

**arXiv ID:** 2608.16910 | [PDF](https://arxiv.org/pdf/2608.16910v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 24. Runtime Governance for Agentic AI: Action-Boundary Control with Trusted Provenance and Fail-Closed Execution

**arXiv ID:** 2608.16891 | [PDF](https://arxiv.org/pdf/2608.16891v1)

**作者:** Adam Mazzocchetti `[一作]` `[通讯]` (SPQR Technologies Inc.), Adam Mazzocchetti (SPQR Technologies Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 Aegis 的运行时治理系统，將模型輸出視為行動提案，通過可信決策層在工具執行前進行授權和審計。

**💡 创新点**

創新點在於將安全界限從模型指令轉移到行動授權層，結合 Senate‑style 兩階段投票、伺服器端可信證據解析與 fail‑closed 執行語義，實現可審計的「提案–授權–執行」管道。

**🔧 技术方法**

使用了 PEP/PDP 參照架構、完整仲裁(reference‑monitor)原則、Runtime‑Assurance 機制、可信證據（trusted provenance）解析以及多代理 Senate 投票協調流程。

**📊 数据集**

利用 42 個任務構成的 sandbox 目錄，在五種模型跑族（stubbed、Gemma、前沿模型不同溫度）中重複執行 10 次，生成 6,300 條評估紀錄，其中 2,100 條屬於受治理情境。

**📈 对比分析**

與純 mesh 及 prompt‑policy mesh 兩個對照條件相比，Aegis 在 2,100 條治理行為中零次 mock‑tool 執行與零次危險 side‑effect 完成，平均延遲 25–28 ms（p95 ≤ 67 ms），證明在重複測試下可穩定阻斷危險行動。

**⚠️ 局限性**

局限包括：僅在 sandbox 模擬環境、任務與工具範圍有限、依賴於完整且正確的政策與控制映射、未測試真實生產場景與多工具交互，無法保證對所有代理與環境的普遍安全。

---

## 25. PACE: Policy-Attested Contract Execution for Safe AI Agents in Decentralized Finance

**arXiv ID:** 2608.17220 | [PDF](https://arxiv.org/pdf/2608.17220v1)

**作者:** Rabimba Karanjai `[一作]` (University of Houston), Shi `[通讯]` (University of Houston)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PACE 框架，给 LLM 代理与 DeFi 合约交互加入事务级授权、决策记录与智能账户验证。

**💡 创新点**

通过三层结构将 LLM 输出的事务意图、确定性策略验证与加密签名的 Policy Decision Record 绑定，并在链上重新验证，提供不依赖模型的安全底线。

**🔧 技术方法**

使用结构化事务意图、Deterministic Policy Verifier、PDR 签名与 Solidity 智能账户、EVM 模拟以及 EIP‑712（规划）等技术。

**📊 数据集**

构建了 40 个攻击/正常任务的可重复基准（覆盖 Prompt Injection、恶意合约、DeFi exploit、MEV/滑点等），并在三模型 live‑LLM、mainnet‑fork 等环境进行验证。

**📈 对比分析**

与 6 个基线（Raw、Prompt‑Only、Sim‑Only、WalletGuard、StaticGuard、SimGuard）对比，PACE 在 2,800 次试验中 100% 正确率、0% unsafe 与 false‑positive；gas 约 30k，verifier 延迟 ~0.04 ms。

**⚠️ 局限性**

仅保证单交易符合声明政策，无法防范经济上合法但有害的多步组合；对模型输出的完备性、适配真实链状态以及对适应性攻击的评估仍有限。

---

## 26. ORPA: Online Residual Policy Adaptation for Robot Manipulation Control with Human Feedback

**arXiv ID:** 2608.17323 | [PDF](https://arxiv.org/pdf/2608.17323v1)

**作者:** Muhammad A. Muttaqien `[一作]` (National Institute of AIST), Yukiyasu Domae `[通讯]` (National Institute of AIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了在线残差策略自适应（ORPA）框架，利用人类反馈在不修改基准策略参数的情况下实时调整机器人执行动作；

**💡 创新点**

创新点在于通过轻量级反馈编码器和策略更新器在关节空间生成上下文感知的残差校正，实现即时、连续的动作修正，避免了完整策略重新训练或固定几何规则的缺陷；

**🔧 技术方法**

技术包括Transformer‑based ACT预训练策略、反馈编码器、策略更新器、合成误差生成、MSE 损失训练以及在多视角RGB+关节状态下的联合决策；

**📊 数据集**

使用的数据集为ALOHA平台上收集的手动演示轨迹（约100条/任务），并通过对成功演示的随机平移/旋转产生合成错误与对应反馈标签；

**📈 对比分析**

与基准ACT、基于逆运动学的规则校正以及OLAF对比，ORPA 在模拟任务中把Cube Transfer 的成功率从60% 提升至92%+，Bimanual Insertion 从60% 提升至80%–85%，在真实环境中对三项任务分别提升 2–3%；

**⚠️ 局限性**

局限性包括对较大或非几何错误（如时序、接触动力学）适应不足；需人工提供反馈；合成误差可能不完全覆盖真实分布；以及对极端扰动的鲁棒性仍有限。

---

## 27. Dynamic Regime-Aware Conformal Calibration for Reliable Economic Forecast Intervals under Multiple Distribution Shifts

**arXiv ID:** 2608.17079 | [PDF](https://arxiv.org/pdf/2608.17079v1)

**作者:** Bogdan Oancea `[一作]` `[通讯]` (University of Bucharest), Bogdan Oancea (University of Bucharest)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种动态基于状态的加权合成分布无关预测区间方法（Dynamic Regime-Aware Conformal Prediction），将密度比重、局部核加权、状态相似度加权以及在线显著性水平自适应控制统一到单步加权共形校准中，适用于经济预测中的协变量漂移、局部异质性、概念漂移与潜在状态共存的情形。

**💡 创新点**

创新点在于将四种独立的偏离交换性机制（协变量平移、局部性、隐藏状态、在线误差反馈）通过乘积组合成单一加权向量，并在此基础上实现自适应显著性水平控制；同时提供针对“oracle”权重的有限样本有效性、对估计误差的覆盖缺口解析、以及多速率自调控制的无悔回报理论。

**🔧 技术方法**

技术包括加权共形预测（density‑ratio weighting、局部核权重、状态后验相似度权重），自适应显著性水平控制（自调AIC、FACI/SAOCP 风格的误差反馈），条件尺度归一化，Gaussian Mixture 模型估计隐藏状态，以及对权重有效样本量（ESS）的控制与下限。

**📊 数据集**

使用了48条真实经济与金融时序数据：欧元区HICP通胀、美国宏观与能源指标、每日金融系列（债券利率、波动率、股票指数等），并在合成数据中分别模拟协变量漂移、条件漂移、突变漂移、渐变漂移、状态切换以及它们的组合。

**📈 对比分析**

与六个基线（Split、Rolling、Adaptive CP、FACI、SAOCP、Conformal‑PID）以及三种最近的在线共形方法比较。方法在平均区间分数上排名第三（平均排名 3.15），区间宽度平均比最锐利的竞争者宽约 20%；但其置信度最接近名义水平（0.890 vs 0.90），在所有预测时点、所有基线和所有基准点预测器上保持最佳覆盖，并且在 2021–23 通胀高峰期间从未出现低于 0.80 的欠覆盖。

**⚠️ 局限性**

局限性包括：权重组合对单步预测最有效，随步长增大其优势消失；在仅存在单一偏移的场景下组合加权效果不如专门化方法；方法计算成本约为 Split‑CP 的 6 倍，适用于低频宏观/能源预测但不适合高频交易；覆盖优势与点预测器的残差结构相关，在点预测器表现最佳时优势消失。

---

## 28. Mr.Dec: Daily-Scale Longitudinal Multimodal Modeling for 30-Day Readmission Prediction

**arXiv ID:** 2608.16929 | [PDF](https://arxiv.org/pdf/2608.16929v1)

**作者:** Minjun Kim `[一作]` (Yeji X), Jong Hak Moon `[通讯]` (Yeji X)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一个每日尺度的多模态读入风险预测解码器，通过将每个住院日的电子病历（EHR）和偶发的胸部X光（CXR）编码为时间序列的 token，使用因果 Transformer 进行序列建模，实时预测30天再住院风险。

**💡 创新点**

创新点在于：①日级别的多模态序列化而非压缩为固定向量；②使用病种特定的监督对比学习为潜在空间注入诊断结构；③利用梯度导出的“关键日”解释模型输出，提供可操作的临床解释。

**🔧 技术方法**

技术上采用冻结的 BioClinical BERT 处理文本化 EHR、EVA‑X‑Base 处理 CXR 图像；对 token 进行模态与日嵌入后输入因果 Transformer 解码器；目标为 BCE 与对比损失的联合优化；解释通过梯度反向传播得到 token 级别重要性。

**📊 数据集**

使用 MIMIC‑IV 结合 MIMIC‑CXR 数据集，筛选成人住院 ≥48h 且至少两次 CXR 的 13,821 次住院记录，分为 90% 训练/10% 测试。

**📈 对比分析**

与 MM‑STGNN、MuST 等图模型及 Qwen3‑VL、MedGemma、Lingshu 等大规模视觉‑语言模型进行对比。实验显示在 CXR–EHR 双模态下，模型 AUC 0.814、F1 0.752，优于 MM‑STGNN 的 AUC+0.014、F1+0.184，尤其在非危重再住院病例召回率提升显著。

**⚠️ 局限性**

局限在于仅包含 EHR 与 CXR 两种模态，未考虑临床笔记、ECG 等；模型对数据缺失和标签噪声仍有一定敏感性；解释仅基于梯度，可能受模型权重变化影响。

---

## 29. Reduced-Order Physics-Informed Neural Network with Adaptive Basis Refinement for Structural Identification

**arXiv ID:** 2608.17131 | [PDF](https://arxiv.org/pdf/2608.17131v1)

**作者:** Rui Zhang `[一作]` (ETH Zürich), Eleni Chatzi `[通讯]` (ETH Zürich)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种结合投影降维和自适应基空间修正的RO-PINN框架，用于在已知和不完整物理模型下进行结构参数与残余力联合辨识。

**💡 创新点**

创新点在于：①将投影降维直接嵌入PINN损失，使逆问题在低维潜在空间求解；②自适应更新基空间以消除基不匹配误差；③在同一框架下同时辨识残余恢复力和结构参数，支持内部变量约束。

**🔧 技术方法**

使用技术包括：PINN、投影降维（POD/Galerkin）、自动微分、Adam + L-BFGS 优化、Bouc–Wen 弹性内部变量模型。

**📊 数据集**

实验数据基于四层钢框架的数值模拟，包含布基–温恢复力的Bouc–Wen模型，使用El Centro地震激励，加入5%高斯噪声与2%材料误差，测点稀疏。

**📈 对比分析**

与完整FOM‑PINN（收敛困难）和贝叶斯模型更新（BMU）对比，RO‑PINN在参数辨识误差低于BMU（约1–4%），并在训练时间上比BMU节约约80%（约1h vs 5h）。

**⚠️ 局限性**

局限性包括：对不同不确定性源的可辨识性受限，稀疏/噪声测量易导致模糊；联合辨识更难，且目前仅在合成数据上验证，未涉及实验验证与大规模实时监测。

---

## 30. Lymphocyte Mimicry Correction via Region-Level Tissue Reasoning and Unbalanced Optimal Transport

**arXiv ID:** 2608.17151 | [PDF](https://arxiv.org/pdf/2608.17151v1)

**作者:** Xiang Li `[一作]` (Duke University), Kyle J. Lafata `[通讯]` (Duke University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Loki-OT框架，利用无平衡最优传输将基于大语言模型的区域级组织推理迁移到细胞级预测，从而纠正淋巴细胞伪影。

**💡 创新点**

创新点在于将区域级组织先验与细胞级预测结合，采用无平衡OT与两阶段软标签蒸馏实现组织上下文与细胞形态的互补学习。

**🔧 技术方法**

使用的技术包括CellViT++特征编码、Claude Sonnet 4.5 MLLM获取组织密度先验、无平衡最优传输、软标签蒸馏、以及轻量化MLP学生模型。

**📊 数据集**

数据集为TIGER挑战集（训练135 ROI）与TCGA‑BRCA独立子集（124患者）。

**📈 对比分析**

与Lizard、PanopTILs、Context-soft等基线对比，在TCGA‑BRCA上Loki-OT在患者级MAE上从58.3降至46.0，F1/精度/召回略低但精度提升，整体误差显著降低。

**⚠️ 局限性**

局限性包括仅在乳腺癌TIL计数上验证，依赖MLLM先验的粗糙性；阈值设定对召回有影响；对不同癌种、细胞类型及多尺度模型的鲁棒性待进一步验证。

---

## 31. SAGE: Self-Evolving Storyboard Skills via Attribution-Guided Rule Evolution

**arXiv ID:** 2608.17468 | [PDF](https://arxiv.org/pdf/2608.17468v1)

**作者:** Maolin Ran `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SAGE框架，利用规则级归因机制从专业导演的示例中自动学习、演化并注入导演知识，完成无人工干预的自动分镜生成。

**💡 创新点**

创新点在于：① 引入规则级归因，将执行反馈精确归因到单条自然语言规则，实现有针对性的规则增删与修订；② 将演化得到的规则聚类成情境包，既保持可解释性又实现高效注入；③ 在大语言模型上实现知识可学习参数化。

**🔧 技术方法**

使用对比抽取规则、归因式生成、对齐评估、规则归因诊断、情境包聚合等技术，并依赖大语言模型进行分镜生成与质量评估。

**📊 数据集**

采用公开的PROSE数据集（68集剧本与专业导演分镜对齐），训练集50集，测试集18集，覆盖三类短剧。

**📈 对比分析**

在与Vanilla、Few-shot、CoT、EvoSkill、SkillOpt等基线对比时，SAGE在五个维度的平均得分为77.8，超过专业导演的77.1；在实际生产部署中，接受率达87.2%，作者时间缩短83%。

**⚠️ 局限性**

局限性包括：规则提取与归因仍需人工评估的成本；对极端/稀缺场景的覆盖不足；对不同剧本结构的泛化性待进一步验证。

---

## 32. NGS-Marker: Robust Native Watermarking for 3D Gaussian Splatting

**arXiv ID:** 2608.17447 | [PDF](https://arxiv.org/pdf/2608.17447v1)

**作者:** Hao Qin `[一作]` (Zhejiang University), Qiang Zhu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种针对3D Gaussian Splatting的原生水印框架NGS-Marker，用来防止局部盗版（Partial Infringement）并实现细粒度版权保护。

**💡 创新点**

创新点在于直接对Gaussian原语进行水印注入，采用梯度递进优化保证水印在任意局部可解，支持多模态水印并能与间接渲染级保护协同。

**🔧 技术方法**

使用点变换器（PointTransformer）联合训练注入器与解码器，并利用CLIP文本/图像编码器和交叉注意力机制，最后通过梯度下降在场景级别递进注入水印。

**📊 数据集**

在公开的3D数据集上进行实验，训练集24个场景，测试集4个场景。

**📈 对比分析**

与3D-GSW、GaussianMarker、GuardSplat以及两种基于本地注入器的基线（WI-Naive、WI-Iterative）比较，NGS-Marker在Bit-Acc超过99%、3D-Acc超过95%，渲染质量PSNR/SSIM保持最佳，且在噪声、旋转、尺度、稀疏等多种攻击下仍保持高鲁棒性。

**⚠️ 局限性**

局限性包括对3D编码/解码技术依赖较高，且在大规模场景下需要更高效的分区与遍历策略来提升可扩展性。

---

## 33. Six Ways to Draw Vangers with WebGPU: Real-Time Rendering of Editable Multi-Layer Height Fields

**arXiv ID:** 2608.17390 | [PDF](https://arxiv.org/pdf/2608.17390v1)

**作者:** Dzmitry Malyshau `[一作]` `[通讯]` (Independent Researcher), Dzmitry Malyshau (Independent Researcher)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过一个统一的引擎和数据路径，对Vangers游戏的六种地形渲染技术（高度场射线行进、体素加速射线行进、水平切片、条形柱子、散点投射和贪心三角网）进行系统对比；

**💡 创新点**

创新点在于在保持相同硬件、相同数据、相同摄像机和相同着色器的前提下，设计了可重复的测评协议，加入了覆盖率、连贯性、编辑一致性等多维度指标，并公开了完整的引擎、工具和实验脚本；

**🔧 技术方法**

使用了WebGPU的wgpu实现、WGSL着色器、基于单一高度图数据的多层编码、体素金字塔、水平切片、散点散射、三角网拟合等技术；

**📊 数据集**

数据集为Vangers原始游戏的多层高度图（Fostral等10个官方地图），每个地图包含两层固体区间，已公开共享；

**📈 对比分析**

比较方法通过统一的CPU参考射线行进进行覆盖率/误差评估，并在十二个视点、多个设备（AMD Radeon 780M、Radeon RX 7900 XT、Intel RPL-U、NVIDIA RTX 5070、Apple M3）下测量帧时、编辑后首帧一致性、内存占用等；结果显示Mesh（q=0.5）在所有设备上帧时最优，其次是RayTraced 128；其他方法的帧时与覆盖率呈现视角依赖和质量差异；

**⚠️ 局限性**

局限性包括：仅评估单一引擎和单一数据格式；实验仅涵盖Vangers级别，未验证其它多层或不连续高度数据；仅在四种Vulkan和一种Metal设备上测试，未覆盖D3D12、WebGL2、移动GPU；帧时测量未考虑管线重叠；编辑实验仅针对单一洞穴形状；并且Mesh的预处理与内存成本较高。

---

## 34. Once Generated, Ranked: End-to-End Generative Slate Recommendation with Unified Semantic-Collaborative IDs

**arXiv ID:** 2608.17613 | [PDF](https://arxiv.org/pdf/2608.17613v1)

**作者:** Yang Hu `[一作]` (Kuaishou Technology), Kaiqiao Zhan `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了端到端的生成式幻灯片推荐框架OGR，能够直接生成有序的推荐幻灯片并实现一次性生成与排序。

**💡 创新点**

创新点包括：① 通过TUSID构建推荐感知的分层语义ID（SID），融合多模态语义与局部协同信号；② 采用全局列表规划+位置解码（GL2P）实现跨位置依赖的并行解码；③ 在后训练阶段引入奖励校准与保守策略（SPA），实现对齐与防止模型偏离。

**🔧 技术方法**

使用的技术包括多模态LLM+文本编码交叉注意、CountSketch协同注入、残差K-means量化、Transformer列表规划、位置SID解码、Beam搜索、奖励校准与保守策略优化。

**📊 数据集**

实验数据集涵盖了Kuaishou工业数据集和公开的KuaiRec数据集。

**📈 对比分析**

对比方法涵盖顺序推荐、列表重排序和生成式基线（如TIGER、OneRec），使用Hit Rate、Recall、NDCG@5等指标；在离线实验中，OGR在NDCG@5上相较基线提升48.2%/27.2%；在Kuaishou线上A/B测试中，Effective Views提升1.12%，其他指标亦有显著提升。

**⚠️ 局限性**

局限性：模型参数量和计算开销较大，需大量训练资源；在不同业务场景下的鲁棒性与多样性、冷启动等问题仍需进一步验证；保守策略的过度约束可能限制模型的探索能力。

---

## 35. Adaptive surrogate modeling for high-dimensional spatio-temporal output

**arXiv ID:** 2608.17250 | [PDF](https://arxiv.org/pdf/2608.17250v1)

**作者:** Berkcan Kapusuzoglu `[一作]` (Vanderbilt University), Sankaran Mahadevan `[通讯]` (Vanderbilt University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种适用于高维时空输出的自适应代理模型构建方法，结合随机SVD降维、交叉验证误差量化以及探索‑利用学习函数实现新样本点的自动选择，并在气体涡轮叶片的多物理耦合仿真中进行了验证。

**💡 创新点**

创新点包括：①两步随机SVD降维将海量时空输出映射到低维无相关空间；②在低维空间构建单一代理并通过留一交叉验证分离重构误差与代理误差；③将基于GP的误差预测与Expected Improvement和空间充填准则结合，形成一种兼顾探索与利用的新学习函数；④针对多变量时空输出设计了专门的误差指标PNMAE/PNrMAE。

**🔧 技术方法**

所用技术包括：随机SVD、Extra‑Trees回归、留一交叉验证、MAE/rMAE误差指标、Gaussian Process（GP）预测误差、Expected Improvement（EI）、maximin Latin Hypercube Sampling（LHS）、空间充填距离度量、学习函数参数α、γ、β1、β2的EM式估计等。

**📊 数据集**

实验数据集：①七个二维基准函数（Branin、Goldstein、Sasena、Alpine、Modified‑Meckesheimer、Modified‑Jin、Schwefel）；②气体涡轮叶片模型，输入4维不确定参数，输出6个量（CEEQ、Dc、Mises、x/y/z位移），约29,374节点、54时步，单次仿真输出维度≈9,000,000。

**📈 对比分析**

通过NRMSE对比7种现有自适应采样方法（EI、CVV、LOLA、MSD、MASA、CVVor等），在45个样本时所提方法在所有基准函数上几乎达到零NRMSE，优于或等同于其它算法；在叶片案例中，逐批增加66个训练点后，平均PNrMAE降至15.8%、PNMAE降至24.6%，RMSE降至0.00193，满足工程精度要求。

**⚠️ 局限性**

局限性：①自适应采样仍未完全达到设定的PNrMAE≤15%阈值；②误差分量未能彻底区分重构误差与代理误差；③在高维问题中，LOOCV+GP训练与学习函数评估的计算开销较大；④批量采样仅采用简单筛选，未实现最优批量选择；⑤代码未公开，复现性受限。

---

## 36. GroupForward: Building Referable 3D Scenes via Instance-Grouped Feed-Forward Gaussian Splatting

**arXiv ID:** 2608.17535 | [PDF](https://arxiv.org/pdf/2608.17535v1)

**作者:** Qijian Tian `[一作]` (Shanghai Jiao Tong University), Xin Tan `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 GroupForward 通过低维实例嵌入实现对稀疏无姿态图像的实例级 3D 高斯光栅化重建，并基于生成的实例组构建 Referential Scene Reasoning Framework (RSRF) 进行复杂指代语义分割。

**💡 创新点**

核心创新在于将高维语义特征替换为紧凑实例嵌入，利用实例聚类与实例级语义聚合实现跨视图一致的实例分割；引入 gauge‑aligned self‑rendering 以提升相机姿态与几何一致性；构建实例图谱并在其上与 VLM 结合实现指代推理。

**🔧 技术方法**

使用 3D Gaussian Splatting、HDBSCAN 聚类、Open‑Vocabulary 语义模型 (LSeg/CLIP)、gauge‑aligned self‑rendering、三维实例图谱、VLM (Qwen3‑VL‑8B) 以及多视图自编码器等技术。

**📊 数据集**

主要在 ScanNet（两视图、4/6/8/16视图、零样本在 Replica 上）上训练与评估，并在 ScanNet 与 Replica 进行跨数据集泛化测试。

**📈 对比分析**

与 LSM、Uni3R、Per‑Scene Optimized、PointMap 等基线进行对比；在 ScanNet 上在语义 mIoU、几何误差、PSNR/SSIM、以及新视角指代分割 mIoU 等指标均取得领先；在多视图和零样本设置下也表现出优越性。

**⚠️ 局限性**

对极少视角或严重遮挡场景的实例聚类仍易出现错误；需要较多计算资源（多 GPU 训练），且对语义模型的质量高度依赖；在没有高质量伪标签时性能下降。

---

## 37. Which Source Wins? Task-Dependent Reliance in Vision-Language Models

**arXiv ID:** 2608.17205 | [PDF](https://arxiv.org/pdf/2608.17205v1)

**作者:** Rodela Ghosh `[一作]` (University of South Florida), Guangjing Wang `[通讯]` (University of South Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究视觉‑语言模型在面对视觉和文本冲突时的模态重新分配，构造可控的图像与文本对抗样本，逐级降低其中一方的可读性，并通过生成答案和条件对数似然差异量化模型对干扰源的偏好变化。

**💡 创新点**

提出了一套全流程的冲突实验框架和新的 ChartQA‑Conflict 评测集，揭示不同任务（算术 vs 图表报告）会导致模型对降质模态的偏好产生显著方向性差异，证明模态权重不是固定的。

**🔧 技术方法**

使用条件对数似然（CLL）边际、生成答案归属判定、四级模态降质梯度、对齐的 Source A/B 标记、配对 Wilcoxon 检验与 bootstrap 置信区间等统计方法来衡量和比较源偏好变化。

**📊 数据集**

基准数据集包括：GSM8K 与 SVAMP（算术问题）用于渲染图像与文本冲突；ChartQA‑Conflict（229条图表‑报告冲突）及其表格对照版本；以及公开的 GSM8K/SVAMP/ChartQA 原始测试集，用于构造冲突与验证。

**📈 对比分析**

评估六款开源 VLM（Qwen2‑VL‑2B、Qwen2.5‑VL‑7B、Idefics3‑8B、LLaVA‑OneVision‑7B、LLaVA‑1.6‑7B、Phi‑3.5‑Vision）在不同冲突与降质设置下的偏好差异；结果显示算术冲突中大多数模型对降质文本的偏好减弱更明显，而在 ChartQA‑Conflict 中所有模型对降质图像的偏好减弱更强，且所有差异均在 95% 置信区间内显著。

**⚠️ 局限性**

局限性包括：仅适用于可数值归属的任务；受限于 2B–8B 规模可访问的开源模型；未能彻底分离任务、表示与冲突构造的因果影响；ChartQA‑Conflict 由作者自评且未盲评；降质梯度按名义级别匹配而非心理学等效度；实验仅检验在可读性下降时的强制源选择，未考察模型的冲突检测、表达不确定性或拒绝回答能力。

---

## 38. MS-MFAD : Multimodal large language models for Face Anti-spoofing Detection

**arXiv ID:** 2608.17328 | [PDF](https://arxiv.org/pdf/2608.17328v1)

**作者:** Xiaoyong Yu `[一作]` (Mashang Consumer Finance Co., Ltd.), Xinge You `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 MFAD 系统，通过细粒度像素语义锚定激活多模态大语言模型的推理能力，实现统一可解释的面部防伪检测。

**💡 创新点**

核心创新在于仅用 1000 张高质量语义级标注掩码即可构建跨攻击的 Chain‑of‑Thought 训练数据，并通过像素级语义锚定消除定位幻觉。

**🔧 技术方法**

技术手段包括多模态大语言模型（如 Qwen‑VL）、LoRA 微调、语义级掩码驱动的注意力重定向以及三阶段质量控制的 CoT 生成流程。

**📊 数据集**

使用的主要数据集为自建的 16,000 样本语义锚定集（覆盖 16 类攻击），以及公开的 CelebA‑Spoof、MS‑UFAD‑DeepFake 与 MFFI 进行评测。

**📈 对比分析**

与传统 CNN、CLIP 以及大型通用模型对比，MFAD‑7B 在 ID 数据上 ACER 仅为 0.0357、BPCER 0.0000，AUC 均超过 0.99；在 OOD 数据上虽提升至 0.67 的 AUC，但仍低于通用模型的 0.5 以上阈值。

**⚠️ 局限性**

局限性包括对完全未知攻击的泛化能力不足、对 OOD 数据的鲁棒性仍需提升，以及对模型规模的依赖导致推理延迟和部署成本上升。

---

## 39. Probing the Prefill: Detecting Code Vulnerabilities via Latent Activations

**arXiv ID:** 2608.16970 | [PDF](https://arxiv.org/pdf/2608.16970v1)

**作者:** Alizishaan Khatri `[一作]` `[通讯]` (Wrynx Inc.), Alizishaan Khatri (Wrynx Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四款大型语言模型的最后预填充词激活进行提取，并训练多层感知机（MLP）探测器，评估其在代码功能级漏洞检测任务中的表现。

**💡 创新点**

首次将激活探测方法应用于代码漏洞检测，证明冻结的 LLM 表示已携带可恢复的漏洞信息；探测器体积极小、可嵌入生成流程；并在跨模型、跨数据集上展示一致性。

**🔧 技术方法**

使用 6 层 GELU+Dropout MLP、类加权二元交叉熵训练、阈值调优；激活来源为每个函数在 LLM 前向传递后的最后层最后词状态；对四个不同规模/家族模型进行实验。

**📊 数据集**

四个 C/C++ 函数级漏洞基准：Devign、Big‑Vul、Draper VDISC、PrimeVul；均提供人工或自动标签的漏洞/无漏洞二分类。

**📈 对比分析**

通过与各数据集公开的 SOTA 结果对比，探测器在 Devign 上达 68.8% F1 与 SOTA (67.9%) 接近；在 Big‑Vul、Draper VDISC、PrimeVul 上表现受标签不平衡与质量限制，平均 F1 约 41.7%；探测器参数仅占原模型 0.06–0.17%，推断延迟低于 LLM 本身。

**⚠️ 局限性**

仅在冻结激活上测试，未验证对模型生成代码的实时检测；未进行线性探测器对照、对抗鲁棒性或跨数据集泛化评估；数据集噪声、极端不平衡及缺乏非 LLM 基线等限制；缺少实际推断时延测量。

---

## 40. Benchmarking Automated Security Patch Backporting: How Far Are We?

**arXiv ID:** 2608.17671 | [PDF](https://arxiv.org/pdf/2608.17671v1)

**作者:** Jincheng Yang `[一作]` (Xidian University), Hui Li `[通讯]` (Xidian University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个覆盖跨版本、跨分支、跨仓库的安全补丁回port benchmark，并统一评估框架。

**💡 创新点**

创新点在于提供统一的高质量数据集、共用评估协议和可执行验证，揭示现有工具在结构复杂补丁上的局限。

**🔧 技术方法**

采用程序分析、LLM提示和LLM代理等多范式技术，结合手工验证和可执行反馈。

**📊 数据集**

使用1,234条补丁回port案例，分为600条复现集和634条评估集，并构建45条可执行验证子集。

**📈 对比分析**

通过对五种工具（FixMorph、TSBPort、Mystique、PPatHF、PortGPT）在统一指标下进行对比，发现PortGPT仍领先但在Type‑IV等复杂情形下性能急剧下降。

**⚠️ 局限性**

局限在于数据仅覆盖C/C++、规模有限、可执行验证子集小、LLM模型可能泄漏、缺乏多语言或更广泛项目的评估。

---

## 41. SoK: Cross-Chain Transaction Identification and Matching

**arXiv ID:** 2608.17532 | [PDF](https://arxiv.org/pdf/2608.17532v1)

**作者:** Hang Zheng `[一作]` (Monash University), Tsz Hon Yuen `[通讯]` (Monash University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文系统化总结了跨链交易的识别与匹配技术，归纳了四类识别方法和三类匹配机制，并对现有数据集的可复现性进行实证评估，提出了四项开放挑战。

**💡 创新点**

创新点在于提出了证据层级化的匹配机制框架，阐明模型辅助匹配在跨链通用性与人工成本方面的价值，并首次系统评估并验证了跨链研究中数据集与可复现性的问题。

**🔧 技术方法**

使用的技术包括官方记录查询、签名匹配、地址聚合、学习分类、确定性标识符匹配、字段约束启发式以及基于模型/LLM的匹配与代理执行。

**📊 数据集**

综述了15项工作中的公开数据集，主要集中在XChainDataGen、Empirical Study、Jigsaw、CONNECTOR、ABCTRACER、LOCARD等，涵盖了数百万级跨链交易对，并对其规模与标签来源进行了验证。

**📈 对比分析**

通过匹配率、F1等指标对比三类匹配方法，确定性标识符匹配取得最高匹配率（约99.8%），启发式匹配在缺失标识符时性能下降，模型辅助匹配在跨链泛化与人工成本方面表现优异但匹配率略低。

**⚠️ 局限性**

主要局限包括数据集可用性不稳定、缺乏统一基准与共享测试集、未充分解决多对多匹配与缺失对应的判定，以及模型/代理组件的可审计性和透明度不足。

---

## 42. KnowSim: Evaluating Information Calibration in LLM Assistants with User Simulators that Learn

**arXiv ID:** 2608.17150 | [PDF](https://arxiv.org/pdf/2608.17150v1)

**作者:** Yoonjoo Lee `[一作]` (University of Michigan), Q. Vera Liao `[通讯]` (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于知识状态的用户模拟器（KnowSim），通过信息单元图追踪用户在多轮对话中的知识演进，并提供知识增益、交付校准、认知负荷等指标来评估信息校准质量。

**💡 创新点**

核心创新在于将用户知识建模为可演化的图结构，结合学习理论的吸收规则，自动生成可解释的三种评估指标，并通过该模拟器在不同知识水平下重现人类评价结果，揭示模型与用户知识水平的交互效应。

**🔧 技术方法**

技术包括：信息单元图（IU graph）构建、LLM驱动的用户生成与知识状态更新、基于先修关系的吸收约束、认知负荷计算与终止判定，以及对话中信息校准指标（KG、DC、CO）的自动计算。

**📊 数据集**

使用了两大公开数据集：MATH（数学题目）和ExpertQA（专家级问答），并收集了基于Prolific的真实人机对话数据作为验证基准。

**📈 对比分析**

与三种基线模拟器（Zero‑shot、ZS‑CoT、ZS‑CoT‑Prof）对比，KnowSim在与人类排名的符号一致率上达到73–74%，显著优于基线；在评估9个LLM时，揭示不同知识水平下模型的最佳排序差异，表明传统聚合榜单隐藏的公平性问题。

**⚠️ 局限性**

局限性包括：未建模动机与情绪等非认知因素、仅覆盖数学与专家问答两类知识密集任务、对个人初始知识状态的估计仍基于群体比例、以及多轮LLM调用导致评估成本高，可通过模型压缩和精细化训练进一步改进。

---

## 43. OraclePhys: A Systematic Framework for LLM Fine-Tuning on Structural Mechanics

**arXiv ID:** 2608.17162 | [PDF](https://arxiv.org/pdf/2608.17162v1)

**作者:** Mingyu Li `[一作]` (University of Houston), Haoqian Wang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出OraclePhys框架，构建了可精确评分的结构力学基准，并用它训练并评估LLM。

**💡 创新点**

创新点在于将训练目标的答案形式作为可控变量，证明答案形式决定LLM学到的物理知识，并区分写答案与优势加权分数的效果。

**🔧 技术方法**

技术包括OraclePhys-Bench的精确FE oracle评分、OraclePhys-30K监督数据、Qwen3-8B+LoRA的稀疏微调、GRPO强化学习与多种验证角色。

**📊 数据集**

使用了OraclePhys-30K（约3万例）以及对应的FE oracle结果作为标注，全部为程序化生成的二维钢框架。

**📈 对比分析**

实验对比将训练好的8B模型与Claude Opus 4.8、专用GNN以及先验公式进行前向定位、后置编辑等指标对比，8B在多项指标上逼近或超过专家模型，处于数据精度前沿。

**⚠️ 局限性**

局限性包括仅在8B+LoRA、少量RL步骤、单一物理领域和合成数据上验证，未探索更大规模或真实工程环境；仅评估行为层面，未给出内部机制；部分对照实验单跑，统计功效有限。

---

## 44. LadderTeam: Dual-Agent Laddering Elicitation Framework

**arXiv ID:** 2608.17029 | [PDF](https://arxiv.org/pdf/2608.17029v1)

**作者:** Manjushree Aithal `[一作]` (University of Colorado Anschutz), James Mitchell `[通讯]` (University of Colorado Anschutz)

**通讯引用:** 64896 | [OpenAlex ID](https://openalex.org/A5051839223)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个双代理LLM框架 LadderTeam，用于自动化软件可用性访谈中的梯级化需求获取。

**💡 创新点**

创新点在于将主动采访者与后台评判者并行的双代理架构与三种梯级化方法（ACV、5-Whys、JTBD）结合，并引入实时漂移防护与中止逻辑。

**🔧 技术方法**

使用了大型语言模型（如 GPT-5.5、Claude Sonnet 4.6、Gemma4、Qwen3.6）作为采访者和评判者，并实现了五步循环、状态门控、漂移检测和评判者评分。

**📊 数据集**

数据集包括两个极端人格（不愿意、简洁）模拟的预先编写脚本以及真实的七人访谈转录，全部将公开。

**📈 对比分析**

通过对216次模拟访谈的实验，对比三种梯级方法、四个模型、两种人格和三种UI情景，得到链收敛率99.1%、真实响应匹配81.0%，ACV最高，模型间差异明显。

**⚠️ 局限性**

局限在于对种子语句的模糊性敏感，评判者标准需人工验证，且仅在单一产品线和人格样本上验证，缺乏跨域通用性。

---

## 45. Repetition as Reinforcement: Enhancing Sample Efficiency via Instant Episode Repetition in Reinforcement Learning

**arXiv ID:** 2608.17347 | [PDF](https://arxiv.org/pdf/2608.17347v1)

**作者:** Hoda Yamani `[一作]` (University of Auckland), Henry Williams `[通讯]` (University of Auckland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Instant Episode Repetition（IER）机制，在RL训练期间直接重复高回报轨迹以提升样本效率。

**💡 创新点**

创新在于将经验重放从被动更新转变为主动交互阶段，通过即时重复成功轨迹影响采样分布，而不改动网络或目标函数。

**🔧 技术方法**

技术实现为在TD3和SAC等off‑policy算法中加入IER模块，控制重复次数RN，并将重复轨迹写入 replay buffer。

**📊 数据集**

使用MuJoCo、DeepMind Control Suite 的八个连续控制任务以及真实世界机器人抓取/物体平移实验。

**📈 对比分析**

与基线 TD3/SAC 以及 SIL 变体对比，IER 在大多数任务上提升学习速率与最终回报，实验结果显示 IER‑SAC/TDC3 超越相应基线，尤其在动态运动任务中效果显著。

**⚠️ 局限性**

局限在于需要手动设定重复次数 RN，过大会降低多样性，且在极度稳定或奖励可预测的任务上提升有限；未探究在离散动作或更大规模环境中的表现。

---

## 46. ArborMem: Navigating Interaction States with Memory Forests

**arXiv ID:** 2608.17534 | [PDF](https://arxiv.org/pdf/2608.17534v1)

**作者:** Zongwei Lv `[一作]` (Peking University), Tong Yang `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在线记忆框架，将长会话拆解为可导航的交互状态森林，并实现状态定位与跨分支证据检索，提升对多线程、可恢复对话的持续性。

**💡 创新点**

创新点在于引入交互状态定位机制和分支结构的可导航记忆森林，既保持局部连贯性又支持跨分支信息复用，解决传统检索式记忆忽略状态定位的问题。

**🔧 技术方法**

使用向量检索（FAISS）+关键词索引、跨分支检索、结构化记录抽取、分支路由评分、树形记忆结构及多模型推理（Qwen3系列）。

**📊 数据集**

评估使用 LongMemEval、LoCoMo、BEAM 100K、BranchMemEval 四个基准，其中 BranchMemEval 为自研的交互状态定位诊断数据集。

**📈 对比分析**

与完整上下文、最近窗口、BM25检索、摘要、Graphiti、A-MEM、Mem0、LiCoMemory 等基线对比，实验显示在四大基准上平均提升 3–10 个百分点；在受限读取预算下优势更显著，查询延迟低于 0.5 秒，吞吐率最高。

**⚠️ 局限性**

局限包括对状态定位错误的敏感、对抽取/更新错误的依赖、单父树结构对多意图场景的局限，以及对误删/纠错机制的不足，需进一步研究定位不确定性与长期错误传播。

---

## 47. Prism-GRPO: Faster VLA Policy Optimization via Splitting Same-outcome Groups

**arXiv ID:** 2608.17423 | [PDF](https://arxiv.org/pdf/2608.17423v1)

**作者:** Zeyun Deng `[一作]` (Purdue University), Jun Huan `[通讯]` (AWS AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在强化学习的视觉-语言-动作 (VLA) 策略中，提出 Prism-GRPO 通过在二元成功奖励中加入轨迹质量得分，从而在相同结果的轨迹组中恢复训练信号。

**💡 创新点**

创新点在于：1）利用质量信号打破零优势组的平局，使所有成功/失败组都能产生梯度；2）保证成功仍优于失败；3）证明此改进不会增加获取信息组的期望采样成本，并在梯度对齐时保持对成功与质量的上升方向。

**🔧 技术方法**

技术包括 GRPO 组式采样、质量加权奖励 (success + λ·quality)、leave‑one‑out (RLOO) 优势估计、以及对梯度对齐的理论分析；实现上使用 Sim2Real 评估和多种质量信号（碰撞力、平滑度、VLM 预测）来验证通用性。

**📊 数据集**

在 RoboTwin 2.0 机器人仿真基准上进行实验，包含 Lift Pot、Move Can Pot、Handover Block、Beat Block Hammer 四个任务，使用基于模拟的碰撞力、平滑度以及 VLM 评估作为质量信号。

**📈 对比分析**

与 Binary GRPO、Binary RLOO、Random Quality、RL‑ZVP 等基线相比，Prism‑GRPO 在匹配相同成功率时节省 22–56% 的采样次数，并且在质量指标上提升 11–15（或 7–13）个百分点；同时显著抑制了“shove‑cheat”等奖励短路行为，并且在真实机器人上表现出更高的可靠性。

**⚠️ 局限性**

局限性在于：1）需要可观测且与成功相关的轨迹质量信号，若无此信号或质量与成功不对齐则难以获益；2）理论上对梯度对齐的验证在完整 VLA 模型中难以直接完成；3）在某些极端情况下质量权重 λ 取值不当可能导致成功优先级被破坏。

---

## 48. Is Haar Enough? Exploring Symlets and Coiflets for Wavelet Convolution Layers

**arXiv ID:** 2608.17662 | [PDF](https://arxiv.org/pdf/2608.17662v1)

**作者:** Md Rifat Ur Rahman `[一作]` `[通讯]` (Bangladesh Univeristy of Engineering and Technology), Md Rifat Ur Rahman (Bangladesh Univeristy of Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探究在波形卷积层中同时调节波形基底与分解层数，寻找滤波长度与分解层次的权衡；

**💡 创新点**

发现更强近似能力的Coiflet基底能减少所需分解层数，从而在保持精度的同时降低参数与FLOPs，提出F‑vs‑L权衡框架；

**🔧 技术方法**

采用多层二维离散波形变换（DWT/IDWT）与深度可分离卷积在波形域处理，实验对比Haar、Daubechies、Symlet与Coiflet基底；

**📊 数据集**

在CIFAR‑10、ImageNet‑1K与Cityscapes语义分割三大数据集上进行验证；

**📈 对比分析**

与ConvNeXt‑T、GFNet、FNO、AFF等主流轻量/频域混合模块进行对比，Coiflet WTConv在保持或提升准确率的同时，参数减少约32%、FLOPs减少约33%；

**⚠️ 局限性**

实验仅限于固定基底与深度可分离卷积，未探索可学习或任务自适应基底，且结果受实验设置与基底选择的影响，可能对更大规模模型或其他任务泛化有限。

---

## 49. Fair ASR: Re-Evaluating Black-Box Jailbreaks under Shared Target-Call Budgets

**arXiv ID:** 2608.17360 | [PDF](https://arxiv.org/pdf/2608.17360v1)

**作者:** Zhida He `[一作]` (Shanghai AI Laboratory), Qiaosheng Zhang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在共享目标模型调用预算下评估黑盒越狱攻击的公平协议Fair-ASR，并基于此重新评估了11种攻击方法，发现预算敏感性、简单攻击仍具竞争力；随后提出ReCode攻击，结合单次重写、无攻击器随机扰动和结构化代码嵌套，在20次目标调用内实现高成功率与低攻击器调用；

**💡 创新点**

创新点在于：①将目标调用视为主预算实现攻击方法的公平对比；②通过Fair-ASR揭示目标/攻击器调用的双维效率平衡；③设计ReCode在保持单次攻击器调用的同时显著提升目标调用效率。

**🔧 技术方法**

采用了目标调用计数、攻击器调用计数、HarmBench与JailbreakBench数据集、GPT-4o与LlamaGuard4等判定器，利用结构化代码嵌套、字符级扰动、无门限重写等技术。

**📊 数据集**

使用了HarmBench（200条标准文本行为）和JailbreakBench（100条有害请求）作为评估数据集。

**📈 对比分析**

对比方法包括手工模板、BoN、LLM驱动攻击等，在共享预算下通过ASR@B、ATC、HS指标进行评估；ReCode在20次目标调用内获得≈85% ASR、≈7攻击器调用，明显优于同类方法。

**⚠️ 局限性**

局限性包括：仅考虑目标调用成本，未涵盖令牌使用、API价格或人工模板成本；不支持多轮对话评估；对不同攻击器规模的鲁棒性仍待进一步研究。

---

## 50. Sparse Coverage: Semantic Center Representations for Patent Prior-Art Retrieval

**arXiv ID:** 2608.16918 | [PDF](https://arxiv.org/pdf/2608.16918v1)

**作者:** You Zuo `[一作]` (Questel), Benoît Sagot `[通讯]` (Inria)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Sparse Coverage，一种无监督的专利检索框架，将局部文本片段的密集嵌入映射到稀疏的语义中心，并通过倒排索引实现高召回的检索。

**💡 创新点**

创新点在于：① 用覆盖导向的 k‑center 选择算法在嵌入空间中构建稀疏词典；② 为每个中心学习自适应半径，结合 top‑K 采样得到局部稀疏激活；③ 通过最大聚合和 IDF 加权，将多片段语义信息压缩为可直接用于倒排检索的稀疏向量；④ 通过控制词典大小 V 调整表示容量，避免单向量压缩导致信息丢失。

**🔧 技术方法**

技术细节包括：预训练编码器（BERT‑for‑Patents、PaECTER、PatentMap‑V0）对句子/短语/混合片段进行上下文嵌入；k‑center（farthest‑first）选取 V 个中心；为每个中心计算覆盖半径；对每个片段进行 top‑K 激活；对文档进行最大聚合并做长度归一化；对中心进行 IDF 计算并剔除高频“停用”中心；最终通过倒排索引检索共享激活中心的文档。

**📊 数据集**

中心构建使用 2000‑2010 年的 EPO 英文专利全文（约 2M 文档，采样 300k 文档）；评估使用 CLEF‑IP 2013 召回检索基准（claims‑to‑passages），包含 48 个英文查询，约 17k 文档和 140 万段落。

**📈 对比分析**

与 BM25、SPECTER2、BERT‑for‑Patents、PaECTER、PatentMap‑V0（稠密检索），SPLADE‑v2（稀疏检索），ColBERT‑v2（late‑interaction）等基线进行比较。指标包括文档级 Recall@100、mAP、PRES@100；段落级 mAP(D)。Sparse Coverage 在多种配置下取得最高的文档 Recall@100（最高 99.31%，超过所有稠密基线），在最佳段落配置下也获得最高的 mAP(D)，并在检索成本‑召回曲线上位于优越区间。

**⚠️ 局限性**

局限性包括：① 召回与排名指标受词典大小 V 和片段粒度（token、NP、NP+token）显著影响，需要手动调参；② 对于需要精细排名的场景，稠密编码器仍能提供更好得分校准；③ 计算成本非单调，较大词典可能导致查询激活中心增多，影响检索速度；④ 该方法依赖完整的倒排索引构建，未利用近似最近邻等压缩技术；⑤ 在极长查询时仍需分句处理，可能引入切分误差。

---

## 51. Balancing Safety and Autonomy: Accessibility-Oriented Interventions in Generative AI for Cognitive Impairment

**arXiv ID:** 2608.17175 | [PDF](https://arxiv.org/pdf/2608.17175v1)

**作者:** Yibo Meng `[一作]` (Weill Cornell Medicine), Zhicong Lu `[通讯]` (George Mason University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

研究了老年认知障碍患者在使用生成式AI时的可访问性干预机制

**💡 创新点**

首次提出了理解增强与保护两类可访问性机制并揭示其随认知水平的动态影响

**🔧 技术方法**

使用自然语言交互的生成式AI系统

**📊 数据集**

访谈数据来自45名认知障碍患者及其照护者

**📈 对比分析**

通过定性访谈比较不同机制在不同认知水平下的体验，并未采用量化性能指标

**⚠️ 局限性**

样本局限于特定地区且缺乏纵向跟踪，严重认知障碍者的主观体验难以获取

---

## 52. CARA: Cognitive Adaptive Recommendation Agent

**arXiv ID:** 2608.16919 | [PDF](https://arxiv.org/pdf/2608.16919v1)

**作者:** Weijun Gao `[一作]` (Chinese University of Hong Kong), Hengxiao Li `[通讯]` (Tongji University)

**通讯引用:** 143 | [OpenAlex ID](https://openalex.org/A5082780758)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CARA框架，将推荐任务拆分为候选过滤与情感/理性两阶段决策；

**💡 创新点**

通过双视角决策与边界感知KTO优化，显著提升低交互场景下的推荐质量并降低幻觉率；

**🔧 技术方法**

使用大语言模型（Qwen3-1.7B）做推理，结合监督微调（SFT）与边界感知KTO；

**📊 数据集**

Amazon Reviews数据集中的CDs、Office、Beauty三大品类；

**📈 对比分析**

与传统协同过滤、序列模型和多种LLM推荐方法比较，在HR@5、NDCG@10等指标上获得相对10%+的提升，整体排名靠前；

**⚠️ 局限性**

实验规模有限，仅覆盖三大品类、少量用户与固定候选集，缺乏大规模交互与跨域验证。

---

## 53. Towards Safer RAG: Only Agents Capable of System 2 Thinking may Access Untrusted Documents

**arXiv ID:** 2608.17153 | [PDF](https://arxiv.org/pdf/2608.17153v1)

**作者:** Mehrdad Ghassabi `[一作]` `[通讯]` (University of Isfahan), Mehrdad Ghassabi (University of Isfahan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了“仅系统2思维能访问未受信任文档”这一改进的安全原则，探讨检索增强生成（RAG）系统在知识毒化攻击下的监控-控制缺口。

**💡 创新点**

引入了Cordon Rate与Contamination Rate两项新评估指标，用以量化检测误报与后续影响之间的差距；验证系统2推理能力可显著降低毒化影响，摆脱严格的Cordon Principle隔离需求。

**🔧 技术方法**

采用系统2推理强化模型（DeepSeek-Reasoner）与基线模型（DeepSeek-Chat）进行对比实验；使用LLM评审模型Gemini 2.5 Pro进行自动判定；利用GPT‑5.6/​Grok 4.6生成毒化文本。

**📊 数据集**

在SciFact、FiQA和MS‑MARCO三大检索基准子集上进行实验，使用BEIR benchmark的前40条查询。

**📈 对比分析**

通过对比两模型的Cordon Rate和Contamination Rate，结果显示Reasoner在SciFact和FiQA上Cordon Rate为0，Chat模型分别为0.175与0.025；在SciFact上Reasoner的Contamination Rate下降至0.10，Chat为0.25，表明系统2推理显著提升鲁棒性。

**⚠️ 局限性**

局限性在于：仅测试了朴素的毒化策略，未覆盖高级对抗攻击；实验规模受限于前40条查询，缺乏更广泛的多任务验证；并未深入探讨不同领域或幻觉类型对监控-控制缺口的影响。

---

## 54. An O-RAN-Assisted MARL Approach for Dynamic Sidelink and Infrastructure Selection in V2X Communications

**arXiv ID:** 2608.17210 | [PDF](https://arxiv.org/pdf/2608.17210v1)

**作者:** Maria Katarine Santana Barbosa `[一作]` (Universidade Federal de Pernambuco), Kelvin Lopes Dias `[通讯]` (Universidade Federal de Pernambuco)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文开发了一个基于Open RAN的多智能体强化学习框架，用于在V2X网络中动态选择PC5侧链（V2V）或Uu网络（V2N）模式，以缓解干扰并优化资源利用。

**💡 创新点**

创新点在于将Open RAN的CTDE（集中训练、分散执行）与QMix+保守Q学习相结合，并通过聚类策略将每个智能体管理一组V2X对，显著降低模型规模；同时实现了离线预训练与在线微调以满足O‑RAN的预训练模型规范。

**🔧 技术方法**

使用的技术包括：Open RAN架构（Near‑RT/Non‑RT RIC、xApp）、OMNeT++/Simu5G仿真、SUMO交通生成、QMix混合网络与保守Q学习（CQL）以及离线+在线训练/微调框架。

**📊 数据集**

实验数据集为仿真生成的交通场景：300辆车（含普通车、拖车、长途车）和10名静态行人，采用SUMO+OMNeT++/Simu5G在1.5 km高速公路上生成的移动轨迹和流量。

**📈 对比分析**

与单智能体保守Q学习以及基于SINR和CQI的启发式方法对比，MARL在车辆仅场景下平均损失下降21%、延迟下降19%，在车辆+行人共存场景下平均损失下降18%、延迟下降30%，并在SINR、延迟、丢包率等多项指标上均优于基线。

**⚠️ 局限性**

局限性包括：仅在单小区仿真中验证，缺乏真实硬件测试；离线训练可能导致模型泛化不足；代理数增多导致计算内存消耗显著上升；对高速移动导致的信道老化和离线训练误差未充分解决。

---

## 55. Nonadaptive Learning in Robust Nonlinear Output Regulation

**arXiv ID:** 2608.17262 | [PDF](https://arxiv.org/pdf/2608.17262v1)

**作者:** Shimin Wang `[一作]` (Lingnan University), Richard D. Braatz `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种非自适应鲁棒非线性输出调节方法，将调节问题转化为稳健非自适应稳定问题，适用于具有任意相对阶的输出反馈系统。

**💡 创新点**

创新点在于不依赖参数适配和线性参数化回归器，采用输入驱动滤波器与通用内部模型结合递归背摆法，实现全局渐近跟踪；同时给出显式可检验的增益选择不等式。

**🔧 技术方法**

所用技术包括输入驱动滤波器、通用内部模型、递归背摆法、输入-状态稳定性分析、极限供给法和Sylvester矩阵方程。

**📊 数据集**

实验使用控制Duffing振子进行仿真，未使用公开数据集。

**📈 对比分析**

通过仿真验证跟踪误差收敛至零，控制输入趋于周期信号，参数估计误差随时间消失；与传统自适应方法相比，系统对未知参数具有更强鲁棒性。

**⚠️ 局限性**

局限性在于对外部扰动频率和相位的假设仍需满足相对简单条件；极限供给法的参数调节仍需经验，且未在实验平台验证。

---

## 56. MITRE-SAGE: A Multi-Agent Cybersecurity Question-Answering model

**arXiv ID:** 2608.16921 | [PDF](https://arxiv.org/pdf/2608.16921v1)

**作者:** Ali Habibzadeh `[一作]` (University of Guilan), Reza Ebrahimi Atani `[通讯]` (University of Guilan)

**通讯引用:** 1179 | [OpenAlex ID](https://openalex.org/A5073214277)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MITRE‑SAGE 多代理检索增强生成框架，用于提升网络安全问答的可靠性与可解释性，并构建了 MITRE‑QA 综合基准。

**💡 创新点**

创新点在于将知识图谱、文本检索与网页搜索融合为多代理协作体系；引入软提示调优的 Text‑to‑Cypher 生成；以及通过分层代理与摘要层实现高质量、多跳推理与信任根基。

**🔧 技术方法**

使用了大语言模型 Qwen2.5（7B/14B/32B）作为生成与检索代理，Neo4j 执行 Cypher 查询，BM25 与 SecureBERT2.0‑biencoder 组合稀疏/稠密检索，DuckDuckGo API 进行网页检索，并在 8‑bit 量化下部署。

**📊 数据集**

数据集包括 MITRE ATT&CK、CAPEC、CWE、NVD CVE、SigmaHQ 规则以及构建的 3,000 题答对（单跳/多跳）MITRE‑QA 基准。

**📈 对比分析**

与 GPT‑4.1 以及基于 Qwen2.5‑32B 的传统 Hybrid‑RAG 进行对比；在八项任务中 MITRE‑SAGE 在五项任务中获得最高分，单跳任务准确率提升最高 58%，多跳文本生成任务正确率提升最高 35.5%。

**⚠️ 局限性**

局限包括对知识图谱的覆盖度受限、检索时延受限、对最新安全情报更新仍需人工维护、以及在极大规模或极复杂推理场景下仍可能产生错误或遗漏。

---

## 57. Reconfiguration-Complete Motion Primitives with Constructive Planning for Deformable Planar Modular Robots

**arXiv ID:** 2608.17324 | [PDF](https://arxiv.org/pdf/2608.17324v1)

**作者:** Jie Gu `[一作]` (Fudan University), Dan Zhang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在平面可变菱形模块化机器人中，提出了基于方格抽象的构造性重构框架，并给出了可逆的旋转与剪切两种本地运动原语；通过构造证明，证明任意非直线边缘连通配置（N≥7）可转换为唯一的阶梯型标准形，从而实现任意两配置的相互可重构；实现了阶梯标准化规划器和统一的边界-交付前瞻选择器，显著降低规划时间；

**💡 创新点**

创新点在于：①将连续可变几何的菱形模块映射到固定方格抽象，保留物理可执行性；②设计两种可逆原语（pivoting、shearing）并构造性证明全局可重构性；③提出基于边界和目标阶梯位置的前瞻选择器，既保持完整性又提升效率；

**🔧 技术方法**

使用的技术包括：方格抽象、图论连通性分析、可逆旋转/剪切原语、构造性证明、阶梯标准化规划器、基于距离与成本的前瞻选择器、实验中使用的随机配置生成与完整性验证算法；

**📊 数据集**

数据集：在模块数 N=9~20 范围内随机生成 40 对随机边缘连通且非直线的初始-目标配置对；

**📈 对比分析**

比较方法：与先前的初始-目标直接规划框架对比；结果显示本方法在所有试验中成功率 100%，且在 N=9~20 时全重构时间低于对手，尤其在 N=20 时前瞻选择器将规划时间从 1496.08s 缩减至 602.60s（约 59.7% 的提升）；

**⚠️ 局限性**

局限性：仅针对平面可变菱形模块；未考虑障碍物与工作空间限制；实验规模有限（N≤20）；缺乏硬件闭环验证，实际执行可行性待后续研究。

---

## 58. An Investigation of the NeurIPS and ICML 2025 Position Tracks

**arXiv ID:** 2608.16894 | [PDF](https://arxiv.org/pdf/2608.16894v1)

**作者:** Fan Yang `[一作]` (Fujitsu Research), Jun Liu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 70136 | [OpenAlex ID](https://openalex.org/A5012820890)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对NeurIPS 2025和ICML 2025 Position Paper Track公开可见的提交进行系统审计，评估其论断类型、与工件的耦合程度、可证伪性以及评审分数的关系，探讨为何改革性批判在可见池中占主导，并提出四项CFP层面的干预措施以平衡方向性与批判性提交

**💡 创新点**

首次将机器学习领域的“议题论文”通过系统量化方式进行元分析，揭示当前主流位置论文在公开审稿池中倾向于局部可证伪的改革性批判，而缺乏能直接操作化新研究方向的工件；并针对这一不平衡提出可操作的会议管理干预方案

**🔧 技术方法**

使用大型语言模型（Google Gemini 3 Pro）进行结构化文本提取并构建八维度评估表，辅以人工复核与Claude Sonnet 4.6交叉验证；对评审分数采用归一化处理，并计算相关性和统计显著性

**📊 数据集**

以NeurIPS 2025和ICML 2025 Position Paper Track公开提交的191篇论文为样本，包含95篇NeurIPS、96篇ICML，涵盖被接受与被拒绝的论文

**📈 对比分析**

将论文的论断类型与工件耦合度、可证伪性与评审分数等维度进行交叉对比，发现（1）改革性批判占74.9%，（2）工件耦合度与评审分数无显著相关性，（3）与历史议题论文（如AlexNet、Transformer、Concrete Problems in AI Safety等）相比，公开池中缺乏能直接操作化新方向的工件

**⚠️ 局限性**

主要局限包括：LLM分类器的主观性与噪声；评审分数尺度差异导致的归一化假设；公开池不包含撤稿和桌面驳回的论文；跨期比较缺乏纵向跟踪；论断类型与真正的方向性工件之间的关联可能不完全一致

---

## 59. LLM-Derived Preference Judgments Are Not Self-Consistent

**arXiv ID:** 2608.17644 | [PDF](https://arxiv.org/pdf/2608.17644v1)

**作者:** Matthew T. Ford `[一作]` (Cornell University), Peter I. Frazier `[通讯]` (Cornell University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过对六种大型语言模型（Claude、Gemini、GPT-5.5、GPT-OSS、Llama、Qwen）在航班、租房和酒店三类受控项目上，系统性检验其对自然语言偏好描述所产生的数值偏好判断（最高愿付价、互换补偿）是否能被单一的美元计价效用函数所统一，从而评估LLM内部一致性。

**💡 创新点**

提出自洽性假设（H_SC）及其全局和局部检验方法：全局RMSE的自举检验；局部P1（路径和）和P2（报价-问价差）诊断，能揭示不同查询类型或路径间的效用差异；并首次在金钱基准下系统量化LLM数值偏好的一致性问题。

**🔧 技术方法**

使用统计检验技术（自举、Bootstrap、正态近似）、效用模型（无约束项效用+准线性金钱假设）、路径构造与比较（P1、P2）、以及价格归一化的残差度量。

**📊 数据集**

自定义受控数据集：航班19个、租房21个、酒店3个（共9个offer组合）以及对应的报价；共48个端点对、300个查询单元，每个查询15次完整调用，形成4,500次LLM调用数据。

**📈 对比分析**

方法比较：全局检验在每个模型下对九组的联合自洽性进行Bonferroni校正，发现所有模型均拒绝自洽性；局部检验显示P2误差率在41.7–87.5%之间，RMSE在航班/租房1.6–6.1%，酒店18.9–44.9%；P1误差率与价格归一化残差相对较低，表明路径合成不总能与直接估计一致。整体表明LLM数值偏好在不同查询类型下存在显著差异。

**⚠️ 局限性**

局限性：仅检验内部一致性，未涉及真实人类偏好；使用受控手写提示、固定模型版本、无对话历史的stateless调用；仅适用于准线性美元效用，无法捕捉财富效应、预算约束等；样本规模有限，未测试更大规模排序或多样化提示。

---

## 60. What Makes a Fairness Gap Actionable? Statistical Actionability for Responsible AI Deployment

**arXiv ID:** 2608.16912 | [PDF](https://arxiv.org/pdf/2608.16912v1)

**作者:** Hairu Fan `[一作]` (Central Michigan University), Shiyuan Wang `[通讯]` (Central Michigan University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了统计可操作性（Statistical Actionability）框架，将公平性评估的统计证据与部署决策融合，给出四种可解释的部署建议（mitigate、collect_data、monitor、no_action）。

**💡 创新点**

创新点在于：① 把公平性评估与部署决策明确分离，构建基于证据的决策层；② 通过信号、精度和充分性三维证据以及部署上下文进行行动可操作性估计；③ 引入稳定性参考（Stability‑based Deployment Reference）和可迁移性评估，证明框架在不同环境下的鲁棒性。

**🔧 技术方法**

使用了统计推断方法（bootstrap、置信区间）、贝叶斯估计、证据整合函数、决策阈值校准、马氏距离等技术，并在 Python/Scikit‑Learn、Fairlearn 等工具上实现。

**📊 数据集**

采用合成仿真数据（包含罕见偏差、类别不平衡、稀疏结果、噪声模型等情形）以及公开的公平性基准审计数据集（如 COMPAS、Adult、German Credit 等）。

**📈 对比分析**

与传统基线（gap‑only、CI‑only、practical‑threshold、always‑mitigate）对比，统计可操作性在误报率、漏报率和平均决策成本上均优，平均决策成本降低 19.2%，误报率降至 3.68%，漏报率 4.79%。在五个迁移实验中，校准规则在四个场景中与目标或acle 误差 < 2%，仅在极罕偏差场景略显保守。

**⚠️ 局限性**

局限性包括：未评估真实部署中的长期决策循环；证据维度有限，仅考虑差距、置信区间、子组样本量，未纳入因果、分布漂移、人类评估等信息；决策阈值固定，未实现在线更新或自适应策略；缺乏对罕见高危偏差场景的专门策略。

---

## 61. BullsEye: Directed Firmware Fuzzing

**arXiv ID:** 2608.17729 | [PDF](https://arxiv.org/pdf/2608.17729v1)

**作者:** Lorenzo Ralli `[一作]` (Sapienza University of Rome), Emilio Coppa `[通讯]` (LUISS University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 BullsEye，一个针对闭源嵌入式 Linux 固件的基本块级定向灰盒模糊器，结合二进制级静态分析、动态距离与满足度监测，实现对已知第三方组件漏洞的快速定位与复现。

**💡 创新点**

创新点包括：① 基于满足度的基本块距离模型，动态调整模糊能量；② 三条件 DBB 分类，覆盖循环门控和强制重入的分支；③ 每个 DBB 独立的冷却调度与多目标公平分配；④ 语法感知与长度扩展变异、溢出预警等优化；⑤ 在闭源固件环境下完成完整的二进制级图构建与运行时图更新。

**🔧 技术方法**

采用 AFL++ + QEMU、CmpLog、RedQueen、Grammar Mutator、LLVM 级距离计算、双缓冲动态图、满足度评分、指数冷却、基于占用率的能量衰减等技术，构成完整的模糊框架。

**📊 数据集**

使用 32 个真实 IoT 固件镜像，覆盖 40 个已知第三方组件漏洞点，作为实验数据集。

**📈 对比分析**

与四个基线（改造的 AFL、Fuzzware、Fuzzware+RedQueen 等）以及 Greenhouse 对比，BullsEye 在 40 个目标中全部复现，能量与时间对比提升 9.5×–72.5×，对 Greenhouse 的 18 个目标提升 9.8×，展示显著的 TTE 与复现率优势。

**⚠️ 局限性**

局限性：仍依赖二进制静态分析的准确性；对复杂的循环/状态机分支可能不完全覆盖；仅针对 MIPS/Linux 固件；多目标时能量分配仍可能出现不均衡；对极端隐藏式指针调用的覆盖有限。

---

## 62. Beyond the Trace: Coupling an Interpretable Reasoning-State Readout to Native MoE Routing

**arXiv ID:** 2608.17638 | [PDF](https://arxiv.org/pdf/2608.17638v1)

**作者:** Kang Chen `[一作]` (Fudan University), Yugang Jiang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并使用可解释的64维语义读出（来自 Jacobian 镜头），并通过 MoE 路由重构该读出，实现无需激活重跑的轻量化读出；随后利用该读出做完结 rollouts 的选择、在线停止/重采样决策以及可解释的机制编辑。

**💡 创新点**

把词汇对齐的隐藏层读出压缩为可读语义轴，并通过 Ridge 回归从原生 MoE 路由恢复该读出；将该读出作为低成本部署的过程状态感知器，在测试时直接驱动决策与机制干预。

**🔧 技术方法**

Jacobain 镜头、词汇对齐线性探针、稀疏编码、Ridge 回归重构、MoE 路由统计、滚动窗口 CUSUM 控制、机制编辑等技术。

**📊 数据集**

竞赛数学题集（AIME‑24/25、HMMT‑25、BRUMO‑25）、gpt‑oss‑20b/120b、Qwen3‑30B‑A3B、GPQA（研究生级科学问答）等数据集。

**📈 对比分析**

在跨基准冻结迁移的离线选择与投票实验中，R64 单分支选择提升 5–8 分，投票加权提升 1–2 分；在线停止/重采样中 R64 在不同成本点提升 1–5 分；机制编辑实验能按预期改变推理路径并提升正确率。

**⚠️ 局限性**

仅在数学竞赛题和两类模型上验证，轴不跨模型对齐；在线控制仅在因果屏蔽的重放实验中评估；机制编辑规模有限；未对序列感知文本基线和真实部署环境进行评估。

---

## 63. Physics-Informed and Hybrid Machine Learning in Additive Manufacturing: Application to Fused Filament Fabrication

**arXiv ID:** 2608.17246 | [PDF](https://arxiv.org/pdf/2608.17246v1)

**作者:** Berkcan Kapusuzoglu `[一作]` (Vanderbilt University), Sankaran Mahadevan `[通讯]` (Vanderbilt University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了将物理知识嵌入到熔融沉积成型（FFF）部件质量预测的深度学习模型中，开发了改进的 sintering 模型，并构建了八种结合三种物理信息融入策略（损失函数约束、物理模型输出做输入、预训练+微调）的混合机器学习模型。

**💡 创新点**

创新点包括：①改进 sintering 模型考虑真实丝材几何并允许几何随时间变化；②首次将三种物理信息融入策略及其组合系统性地应用于 FFF 质量预测；③在有限实验数据下实现物理一致且高精度的预测，验证了物理约束与预训练的协同提升效果。

**🔧 技术方法**

采用热传导+聚合物 sintering 物理模型、Keras/TensorFlow 构建的两层全连接深度神经网络；使用 L1/L2 正则化、ReLU 激活、Adam 优化器；通过 Latin Hypercube 采样生成实验参数；通过物理约束损失函数、物理模型输出做额外输入、预训练合成数据等技术实现物理信息融入。

**📊 数据集**

实验数据集：20 组温度（210–260 °C）和速度（15–46 mm/s）组合，得到 19 条测试样本；预训练合成数据集：1525 条物理模型输入–输出组合，用于 DNN 预训练。

**📈 对比分析**

采用 RMSE 与物理不一致率（对脊径、孔隙率及其与拉伸强度的单调关系的约束）进行模型评估。与单纯物理模型和单纯 DNN 对比，加入物理约束或预训练后 RMSE 明显下降（最优模型 RMSE≈0.018），物理不一致率降至 0，表明预测结果既精准又符合物理规律。

**⚠️ 局限性**

局限性：物理模型自身的近似误差仍会影响预训练模型的偏置；对不同几何、打印机或材料的泛化能力尚未充分验证；实验数据与合成数据可信度的权衡需要进一步研究。

---

## 64. A Glyph Is Not a Letter, a Token Is Not a Word, a Space Is Not a Space: What the Units of Voynichese Are Not

**arXiv ID:** 2608.17096 | [PDF](https://arxiv.org/pdf/2608.17096v1)

**作者:** Liudmila Rozanova `[一作]` (International Institute for Applied Systems Analysis), Alexander Temerev `[通讯]` (University of Geneva)

**通讯引用:** 85 | [OpenAlex ID](https://openalex.org/A5001916677)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对维吉尼奇手稿的 Zandbergen–Landini 记录进行系统统计检验，验证并否定了将符号视为字母、空格分隔词、以及空格即单词边界的三种传统假设；

**💡 创新点**

创新点在于提出可量化的单位验证框架，将符号、令牌和分隔符的假定功能分别通过条件熵、BPE 依赖度、相邻词互信息、边缘字形关联、图像坐标和校准的替换攻击等多维度指标进行独立检验，从而实现对“字母”“词”“单词边界”三类假设的可重复否定；

**🔧 技术方法**

技术方法包括：一阶条件熵与熵差检验、一致性校正的相邻词互信息、字形边缘互信息、BPE 单位学习与依赖度曲线、分隔符内部关联指数、图像坐标宽度差异与 AUC 评估、以及使用校准的拼音替换攻击来检验潜在密码结构；

**📊 数据集**

使用的数据集为 BeNecke MS 408 的 ZL 转写（3b 版本，共 206 页、16 章、3,880 行），以及多种对照文本（拉丁文、意大利文、英文、法语、德语、植物学记录、药典文本等），并对比了已公开的维吉尼奇模仿密码与自引用生成器；

**📈 对比分析**

比较方法：在与对照文本相同的词数、行长模板下，计算条件熵、BPE 依赖度最小点、相邻词互信息比例、边缘字形互信息、分隔符内部关联指数与 AUC 等指标。结果表明：维吉尼奇的条件熵显著低于对照；词序列的相邻词互信息远低于任何对照；但边缘字形互信息显著高于对照；分隔符分为“确定”与“疑似”两类，后者在物理宽度和学习单元穿透率上与前者明显不同；

**⚠️ 局限性**

局限性包括：仅基于 16 章的样本；只测试了固定一对一字母替换与单词替换的密码模型，未覆盖变位、空白插入、状态依赖等更复杂密码；BPE 作为频率聚类方法，无法确定具体语言结构；图像坐标分析受手工标注偏差与分辨率限制；因此结论主要针对符号、令牌、分隔符的功能假设，而非对手稿整体意义与语言身份的最终定论。

---

## 65. OVIP-SG: Open-Vocabulary Instance-Preserving Scene Graphs for Mapping and Retrieval of Small, Fine-Grained Objects

**arXiv ID:** 2608.17633 | [PDF](https://arxiv.org/pdf/2608.17633v1)

**作者:** Tianjing Hao `[一作]` (Xi'an Jiaotong University), Wang Chuang `[通讯]` (Zhiyuan Innovation Technology Co Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个实例保持的开词汇物体地图框架OVIP‑SG，用于功能分割、物体检索和导航，并实现了覆盖条件的否定决策。

**💡 创新点**

创新点在于：①采用对称IoU+标签惩罚的实例保持关联与面积加权特征融合，解决小物体被吞噬和大表面被误标的问题；②基于VLM的知识引导语义分水岭(KGSW)将场景划分为功能区域；③四级级联检索(CARA)实现从地图查找到主动搜索再到否定决策的完整闭环。

**🔧 技术方法**

使用了VLM（Kimi‑K2.6等）枚举、SAM2+CLIP检测、对称IoU与标签惩罚的关联、面积加权融合、VLM功能分组、VVC投票、语义分水岭算法以及基于距离与VLM先验的区域调度。

**📊 数据集**

在Replica（6个室内场景）进行映射评估，ReplicaCAD与HM3D扫描用于检索与导航评估，HM3D真实机器人实验用于验证实地效果。

**📈 对比分析**

与ConceptGraphs、ConceptFusion、HOV‑SG等基线在统一评估器下比较，OVIP‑SG在mAcc提高6.31、F‑mIoU提高5.15、类无关PQ提升至0.398；在检索任务中，CARA的Top‑5@3 m召回率达0.93，整体检索成功率超过对手。

**⚠️ 局限性**

局限包括对大规模真实环境的可扩展性不足、VLM调用成本高、对极端光照或复杂几何场景的鲁棒性待提升以及对离线标注完整度的依赖。

---

## 66. Rethinking Irregular Time Series Forecasting from the Perspective of Basis Functions

**arXiv ID:** 2608.17284 | [PDF](https://arxiv.org/pdf/2608.17284v1)

**作者:** Rongwen Li `[一作]` (Hunan University), Changjian Chen `[通讯]` (Hunan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于去偏神经基函数网络（DNBNet）的不规则多变量时间序列预测模型，利用时间戳密度校正、可学习基函数、多尺度响应提取与质量加权融合，并配合双分支解码器实现更精准预测。

**💡 创新点**

创新点包括：①通过重要性采样与核密度估计对时间戳密度进行校正，消除非均匀采样导致的渐进偏差；②将基函数参数化为神经网络，使模型能自适应学习多样化时间模式；③使用时间感知平均池化与质量感知融合获得更丰富的多尺度特征；④双分支解码器同时利用隐向量预测和基函数重构，提高预测灵活性与可解释性。

**🔧 技术方法**

采用的技术包括基函数响应系数估计、重要性采样、核密度估计、两层MLP可学习基函数、时间感知平均池化、质量加权融合、双分支解码器、时间嵌入、LayerNorm、MLP、AdamW优化等。

**📊 数据集**

在五个真实世界数据集上评测：USHCN（气候）、Human Activity、Student Life（运动/学生传感器）、PhysioNet、MIMIC（ICU临床记录）。

**📈 对比分析**

与12个基线模型（PrimeNet、SeFT、mTAN、CRU、GNeuralFlow、Raindrop、tPatchGNN、GraFITi、Warpformer、Hi-Patch、KAFNet、APN）对比，DNBNet在4/5数据集实现最优或第二优的MSE，平均相对MSE下降约3.18%，在各领域均优于基线。

**⚠️ 局限性**

局限性：需手动设定基函数数量K和尺度S；对极稀疏时间序列的鲁棒性仍有限；缺乏对不同采样率变化下的理论深度分析。

---

## 67. If, Then, Otherwise: Diagnosing Conditional Branching in Vision-Language Navigation

**arXiv ID:** 2608.17318 | [PDF](https://arxiv.org/pdf/2608.17318v1)

**作者:** Seoyoung Lee `[一作]` (University of Texas at Austin), Atlas Wang `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

创建了一个基于场景图的可控条件导航基准，生成约11500条含有分支逻辑的指令，并引入分支选择准确率（BSA）与条件成功率（CSR）等诊断指标。

**💡 创新点**

创新点在于将条件分支可视化为可验证的场景图谓词，提供可调节的逻辑深度、链长度与空间构造，且兼容现有VLN-CE框架，首次系统评估条件推理对导航性能的影响。

**🔧 技术方法**

采用场景图构造、自然语言模板化生成、3D空间关系推断、VLN-CE兼容化以及轻量级神经符号分支决策（Oracle）等技术。

**📊 数据集**

使用四大室内导航数据集：AI2-THOR、Matterport3D、Gibson 与 ReplicaCAD。

**📈 对比分析**

通过在标准SR、SPL以及新提出的BSA、CSR上评估四种现有VLN模型（VLN-Zero、NaVid、NaVILA、Open-Nav），发现NaVid/NaVILA性能低下，Open-Nav与VLN-Zero略好；Oracle在复杂任务中将BSA/CSR提升约2倍。

**⚠️ 局限性**

局限性包括仅针对室内VLN-CE兼容环境，依赖固定几何阈值和场景图语义，Oracle使用先验信息，无法反映真实感知不确定性，且未扩展到户外或开放世界场景。

---

## 68. Differentiable Voronoi Ray Tracing Beyond Rasterization Speeds

**arXiv ID:** 2608.17682 | [PDF](https://arxiv.org/pdf/2608.17682v1)

**作者:** Bernardo Taveira `[一作]` (Chalmers University of Technology), Fredrik Kahl `[通讯]` (Chalmers University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于分割体素（Voronoi）且可微分的光线追踪渲染器，用于实时新视角合成，并实现了高帧率与高质量的平衡。

**💡 创新点**

创新点包括：① 用八面体纹理取代球谐系数，实现每个体素内空间细节的表达；② 表面集中不透明度与尺度不变密度参数化，促使光线快速终止；③ 引入失真正则化与固定预算训练，消除自适应稀疏化的开销；④ 通过Morton排序、warp协同调度和低贡献单元跳过等 GPU 优化，显著提升吞吐量。

**🔧 技术方法**

技术手段主要有：Voronoi 较准划分、球面八面体纹理映射、光线追踪渲染管线、密度指数参数化、失真正则化、半精度纹理缓存、Morton 排序、warp 级别光线调度、低贡献单元跳过。

**📊 数据集**

使用 Mip-NeRF‑360 公开数据集进行训练与评估。

**📈 对比分析**

与现有多种基准方法对比：在 RTX 5090 上实现平均 623 FPS，较 Radiant Foam 提升 3.2×，较 3D Gaussian Splatting 提升 2.8×；在 PSNR、SSIM、LPIPS 上与主流方法保持相近或更优（尤其是室内场景）。还演示了鱼眼、滚动快门、景深和运动模糊等非标准相机效果的支持。

**⚠️ 局限性**

局限性：固定细胞预算在视角稀缺区域可能导致细节不足；缺乏自适应细胞增删机制；对极端光照或极高频纹理仍可能受限；移动端实现的帧率仍显低于桌面 GPU。

---

## 69. Completion-Path Credits: Multi-Resource Control for Scale-Up Fabrics

**arXiv ID:** 2608.17523 | [PDF](https://arxiv.org/pdf/2608.17523v1)

**作者:** Fan Yang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Zhan Wang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并评估了一种名为SemaCredit的接收器控制器，能够根据远程内存操作的HBM占用、原子执行时间和响应注入时间进行资源分配；

**💡 创新点**

创新点在于使用需求向量模型，将每个操作的多阶段资源消耗量化，并在各阶段完成后按阶段释放信用，从而解决传统按字节计数难以反映小操作开销的问题；

**🔧 技术方法**

采用确定性事件模拟器搭建多路径网络与端点模型，实现全向量、标量服务、资源字节等多种基线对比，并通过EWMA自适应乘子校准估计误差；

**📊 数据集**

主要使用合成微基准（HBM热点、Atomic竞争、响应拥塞）以及三种受控应用形状混合（AllReduce、MoE、remote-read），未使用真实生产数据集；

**📈 对比分析**

在P99延迟、良好吞吐率和最大排队延迟等指标上与八种基线（聚合字节、资源字节、主服务、标量服务、反应式耗尽等）进行对比；SemaCredit在Atomic和响应拥塞场景下P99延迟分别降低52%–83%，且保持良好吞吐率；

**⚠️ 局限性**

依赖准确的操作成本估计和地址映射，估计误差或映射错误会削弱性能；模型未考虑真实硬件的调度、重传、能耗等因素，需要进一步硬件验证与实时采样。

---

## 70. Key-Frame Reasoning with SAM3: Third Place Solution for the MeViS-Text Track of the 8th LSVOS Challenge

**arXiv ID:** 2608.17279 | [PDF](https://arxiv.org/pdf/2608.17279v1)

**作者:** Ce Bian `[一作]` (Harbin Institute of Technology), Jianlong Wu `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种两阶段、无训练的框架，用于运动中心的指代视频对象分割（MeViS-Text 任务）。

**💡 创新点**

创新点在于：①利用 Gemini‑3.1 Pro 对视频事件进行实例级拆解、关键帧选择和判别性描述；②直接用 SAM3‑agent 生成像素级种子掩码，并通过 SAM3 视频追踪器双向传播，实现实例级别的精细分割；③完全不依赖任务特定训练或模型集成，只在单块 RTX‑4090 上运行。

**🔧 技术方法**

核心技术包括 Gemini‑3.1 Pro（大规模多模态语言模型）、SAM3‑agent（基于工具调用的像素级分割）、SAM3 视频追踪器（掩码双向传播），以及 API 调用与本地 GPU 推理的组合。

**📊 数据集**

使用的数据集是 8th LSVOS Challenge 的 MeViS‑Text 轨道视频及自然语言表达，包含运动、交互、方向、相对位置等多样化指代表述。

**📈 对比分析**

在官方测试集上排名第三，Final 分数为 0.856593，J&F 为 0.761，T‑acc 为 0.9755，显示出与竞赛顶尖方法相近的性能，且在无训练、无集成的条件下实现了优秀表现。

**⚠️ 局限性**

局限性包括：对关键帧与判别性描述的依赖，一旦第一阶段判断失误或关键帧不佳会导致后续定位/追踪失败；对无目标表达的识别仍相对薄弱；以及对极端复杂或负面约束的表达仍需改进。

---

## 71. BrainNorm: A Foundation Model that knows Normal via Semantic Atlas Pretraining

**arXiv ID:** 2608.17521 | [PDF](https://arxiv.org/pdf/2608.17521v1)

**作者:** Madhumitha Venkatesh `[一作]` (Indian Institute of Technology Hyderabad), Konda Reddy Mopuri `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

开发了BrainNorm，一个基于T1加权结构磁共振影像的规范化基础模型，能够在健康人群上学习年龄与结构相关的分区嵌入，并实现零样本、少样本与完整数据线性探测的疾病分类与年龄估计。

**💡 创新点**

创新点在于构建了分区语义原子空间（Semantic Atlas Latent Space），通过文本先验与对比学习结合SigLIP风格的自监督目标，使每个脑区的嵌入随年龄连续变化并保持分区可辨识，从而实现无监督的局部偏差评估和高效的零样本推理。

**🔧 技术方法**

核心技术包括：基于ViT的分区编码器、Transformer + GeM聚合、文本嵌入（Qwen3）+轻量级Template Mapper、年龄感知的SigLIP对比损失与软年龄交叉熵正则化、线性探测（LP）与零样本偏差评分。

**📊 数据集**

数据集：预训练使用48,018份UK Biobank T1图像（≈66k扫描）；外部健康验证用Mayo Clinic MCSA 2,662份；下游疾病评估使用15,352份来自ADNI、AIBL、MIRIAD、NIFD、PPMI的阿尔茨海默、轻度认知障碍、额颞叶痴呆和帕金森病扫描。

**📈 对比分析**

方法对比：与9个监督基线（NeuroJEPA、NeuroVFM、RadFM、BrainHarmony、BrainIAC、BrainMVP、MedicalNet、y-Aware等）在年龄估计、BAG、零样本与线性探测分类上进行比较。BrainNorm在MAE、Pearson相关、ACC、AUC等指标上均显著优于监督模型，尤其在零样本多疾病分类和少样本线性探测中表现最突出。

**⚠️ 局限性**

局限性：依赖MNI152空间配准；仅使用成人大脑分区（限制对儿童/发育性疾病的适用性）；单一T1加权模态；对扫描仪、协议等域外偏移仍有一定敏感性；模型复杂度较高，需GPU资源；对非结构影像（T2、fMRI等）尚未扩展。

---

## 72. Primitive-Driven Compositional Forensic Visual Prompting for Open-World Face Anti-Spoofing

**arXiv ID:** 2608.17351 | [PDF](https://arxiv.org/pdf/2608.17351v1)

**作者:** Fangling Jiang `[一作]` (University of South China), Ming-Hsuan Yang `[通讯]` (University of California Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

该研究提出了一种基于可视化原语组合的面部欺诈检测框架，通过在预训练的ViT基座上学习视觉提示，实现对开放世界中多变攻击类型的鲁棒识别。

**💡 创新点**

创新点在于将攻击特征表示为可复用的微观法医原语，并在全局上下文引导下动态路由并组合成输入适配的视觉提示，避免了文本提示的语义局限。

**🔧 技术方法**

技术上结合了冻结的CLIP ViT‑L/14@336px基础模型、patch‑aware attention微原语细化、全局上下文提示的路由网络以及多层视觉提示的聚合。

**📊 数据集**

使用的公开数据集包括CASIA‑MFSD、Replay‑Attack、MSU‑MFSD、OULU‑NPU、HQ‑WMCA、SiW‑Mv2、CASIA‑SURF及CeFA等，构建了九个跨域开放世界协议。

**📈 对比分析**

与多种基线（传统手工特征、CNN、域自适应、文本提示及大型预训练模型调优方法）对比，该方法在九个协议上平均HTER下降约29%，AUC提升至83%以上，显著优于现有方法。

**⚠️ 局限性**

主要局限在于对极细微或局部伪装（如化妆、局部眼镜）误判较多，且微原语的语义解释缺乏，可进一步探究更细粒度的多模态融合。

---

## 73. Accuracy and Robustness of Model Cascades Under Data Perturbations

**arXiv ID:** 2608.17711 | [PDF](https://arxiv.org/pdf/2608.17711v1)

**作者:** Pallavi Mitra `[一作]` (AUMOVIO AI Lab), Felix Biessmann `[通讯]` (Berliner Hochschule f"ur Technik)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在数据扰动下模型级联的准确性和鲁棒性，特别关注信心基础的路由决策如何受到输入退化的影响。

**💡 创新点**

提出了在输入退化情况下，模型级联的信心基础路由的可靠性评估，识别了三种失败模式，并强调了在分布变化下评估级联效率的重要性。

**🔧 技术方法**

使用了Gatekeeper框架，该框架通过信心基础的延迟机制在小模型和大模型之间进行输入路由。

**📊 数据集**

使用了CIFAR-10和CIFAR-100数据集，评估了静态腐蚀和序列扰动对模型级联的影响。

**📈 对比分析**

与传统模型相比，模型级联在干净数据上实现了竞争性的预测性能，并在CIFAR-100上实现了10倍的能耗减少。性能在数据腐蚀下显著下降，尤其是在CIFAR-100-C任务中，模型的准确性和路由可靠性受到严重影响。

**⚠️ 局限性**

模型级联在输入退化时可能会出现路由失败或模型崩溃，导致效率优势消失。不同数据集的失败模式不同，CIFAR-10主要表现为路由失败，而CIFAR-100则表现为模型崩溃。

---

## 74. Conformal Prediction for Molecular Properties under Label Shift

**arXiv ID:** 2608.17678 | [PDF](https://arxiv.org/pdf/2608.17678v1)

**作者:** Hyeonsu Lee `[一作]` (Mogam Institute for Biomedical Research), Hyunjin Shin `[通讯]` (Mogam Institute for Biomedical Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种针对标签平移的加权合格预测框架，用于生成分子属性的可靠预测区间。

**💡 创新点**

通过把合格预测的非一致性得分按标签分布比例加权，实现了在分布漂移下的覆盖保证，弥补了传统合格预测在标签平移情形下失效的缺陷。

**🔧 技术方法**

结合了BBSE、RLLS、MLE等标签分布估计方法与加权合格预测（WCP），并基于预训练化学语言模型BART进行基础属性预测。

**📊 数据集**

在Therapeutics Data Commons（TDC）的AqSolDB溶解度数据集上进行实验验证。

**📈 对比分析**

与传统拆分合格预测（Split CP）比较，WCP在存在标签平移时保持了预定的置信覆盖率，平均覆盖率显著提升，区间长度略宽。

**⚠️ 局限性**

需要将源数据分成训练、加权、校准三份，导致有效训练样本减少；对异方差数据的区间自适应性不足，需进一步改进。

---

## 75. 3D Gaussian Accelerated Ray Tracing: Fast training through particle-based backward propagation

**arXiv ID:** 2608.17298 | [PDF](https://arxiv.org/pdf/2608.17298v1)

**作者:** Laurent Vit `[一作]` (University of Canterbury), Richard Green `[通讯]` (University of Canterbury)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于粒子（Gaussian）中心的反向传播框架 3DGART，使光线追踪 Gaussian 的训练变得可行。

**💡 创新点**

创新点在于把梯度累加从像素级别的散射操作改为原始粒子级别的聚集（gather）操作，显著降低了原子操作冲突。

**🔧 技术方法**

核心技术包括透视正确的屏幕空间边界计算、紧凑中间缓存、瓦片-粒子映射和粒子中心的反向求导。

**📊 数据集**

在 Mip‑NeRF 360、Tanks&Temples 与 Deep Blending 三大基准数据集上进行实验。

**📈 对比分析**

与传统像素级别实现相比，3DGART 在 Mip‑NeRF 360 上平均实现 3.5×–4× 的训练速度提升，同时保持或提升 PSNR/SSIM，整体效果优于 3DGRT 等光线追踪方法。

**⚠️ 局限性**

局限性包括显存占用较高（可通过混合反向方法缓解），仅支持视角投影，无法处理鱼眼等极宽角相机模型。

---

## 76. The Road Less Traveled: Congestion-Aware NoC Placement and Packet Routing for FPGAs

**arXiv ID:** 2608.17266 | [PDF](https://arxiv.org/pdf/2608.17266v1)

**作者:** Soheil Gholami Shahrouz `[一作]` (University of Toronto), Vaughn Betz `[通讯]` (University of Toronto)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在FPGA CAD流程中引入了网络拓扑(NoC)拥塞建模、路径多样化路由、SAT路由、NoC感知打包与中心移动等技术，旨在降低NoC拥塞并优化布线与时序。

**💡 创新点**

创新点在于将拥塞成本整合到VPR的放置成本函数，结合转向模型与奇偶路由提升路径多样性，并通过SAT求解实现拥塞回避；同时提出NoC感知的打包与移动，提升传统QoR。

**🔧 技术方法**

使用VPR/VTR工具链、XY/转向/奇偶路由算法、Boolean satisfiability (SAT) 以及强化学习辅助的放置与打包方法。

**📊 数据集**

使用29个合成基准（模拟硬件控制器与I/O的流量模式）和完整MLP（多层感知机）设计。

**📈 对比分析**

与基线（无拥塞模型、XY路由）对比，奇偶路由+拥塞模型可将拥塞下降约90%，加SAT可降至95%；聚合带宽升幅约4%；布线长度下降约8%；时序影响微小。

**⚠️ 局限性**

仅适用于网格/环形NoC拓扑；SAT求解会增加运行时间；对更复杂拓扑与更大规模设计的适用性待验证。

---

## 77. Beyond FLOPs: Energy-Aware Knowledge Distillation for Sustainable LLMs on Code-Related Task

**arXiv ID:** 2608.17515 | [PDF](https://arxiv.org/pdf/2608.17515v1)

**作者:** Enrique Barba Roque `[一作]` (Delft University of Technology), Annibale Panichella `[通讯]` (Delft University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究将能量感知知识蒸馏应用于软件工程任务，评估FLOPs与实际能耗的关系，并在克隆检测、漏洞预测和代码摘要任务上实现多目标压缩。

**💡 创新点**

①引入能量代理模型替代FLOPs作为优化目标；②将Morph多目标优化扩展至生成任务；③系统性比较FLOPs与能耗的相关性。

**🔧 技术方法**

使用Morph多目标进化算法、梯度提升回归能量代理、知识蒸馏（KL+CE损失）、权重子克隆、ROUGE‑L评估及EnergiBridge能耗测量等技术。

**📊 数据集**

采用BigCloneBench、Devign、CodeXGLUE（教师微调）以及The Heap（评估）等数据集。

**📈 对比分析**

通过对比FLOPs优化与能量优化的学生模型，在准确率、模型大小、能耗、预测翻转等指标进行评估；生成任务中比较教师与学生的ROUGE‑L、能耗和内存。结果显示，能量优化在克隆检测任务能减少39%能耗，漏洞预测任务差异不显著；代码摘要学生模型可实现86%体积压缩、90%能耗下降，ROUGE‑L仅下降约13%。

**⚠️ 局限性**

FLOPs缺乏普适性；能量代理需要昂贵的能耗测量；蒸馏损失与ROUGE‑L不足以完整评估语义质量；仅测试了少数模型与任务；评估数据集可能存在污染。

---

## 78. Integrating Novelty and Surprise for Experience Prioritization and Exploration in Image-Based Reinforcement Learning

**arXiv ID:** 2608.17373 | [PDF](https://arxiv.org/pdf/2608.17373v1)

**作者:** Hoda Yamani `[一作]` (University of Auckland), Bruce A. MacDonald `[通讯]` (University of Auckland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究提出了一种将新颖度和惊讶信号用于经验重放优先化和奖励的NSPER与NSPER+R框架，用于提升图像基强化学习的样本效率。

**💡 创新点**

创新点在于将新颖度和惊讶两种内在激励同时作为重放优先化指标与内在奖励，并在PixelTD3中实现了联合优先化和探索策略。

**🔧 技术方法**

使用技术包括像素级的PixelTD3算法、共享自编码器、潜在空间动态模型集合、SSIM重建误差新颖度、MSE预测误差惊讶，以及PER框架改造。

**📊 数据集**

实验数据集为DeepMind Control Suite的五个连续控制任务（Cartpole-Balance、Finger-Spin、Ball-in-Cup、Walker-Walk、Cheetah-Run）与84x84 RGB图像观察。

**📈 对比分析**

与Uniform、TD-PER、RPE-PER、CCLF等基线相比，NSPER+R在大多数任务上实现了更快的收敛和更高的累计奖励，NSPER也显著优于单独的惊讶或新颖度优先化。

**⚠️ 局限性**

限制在于对自编码器重建质量敏感、计算开销较高、在非视觉或极度稀疏奖励环境下效果可能不佳、并需手动调节优先化与奖励权重。

---

## 79. Detecting and Discriminating Operator Misspecification in Hybrid PDE-Parameter Learning: a Reference-Free Instrument, with Discrimination Bounded In Sample

**arXiv ID:** 2608.16925 | [PDF](https://arxiv.org/pdf/2608.16925v1)

**作者:** Eric Fock `[一作]` `[通讯]` (Université de La Réunion), Eric Fock (Université de La Réunion)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种无参照、单次拟合即可检测和区分偏差的 PDE‑参数估计器的工具，能够识别操作符误设并区分参数不可识别。

**💡 创新点**

提出了信息矩阵无参考检验和秩统计两种无 oracle 读数，实现了在单个拟合中同时检测错误操作符与可识别性，展示了误设导致的“平台”错误独立于网络容量。

**🔧 技术方法**

使用白噪声模型、信息矩阵检验、敏感度计算、统计 bootstrap 校准、谱分解、PINN 与 χ‑架构对比、误设场景自定义等技术。

**📊 数据集**

采用一维自伴热方程的解析解作为生成数据，加入高斯观测噪声，构建多种误设与不可识别设计。

**📈 对比分析**

与常规曲线拟合、MLP、PINN 等估计器在相同训练样本上比较，通过识别错误比例、误设检验拒绝率、误设误差平台以及对齐精度评估，显示误设检测率近 100%、误差平台约 30%，识别准确率优于传统方法。

**⚠️ 局限性**

仅在单个可解析 PDE 及已知误设模式下验证，尚未证明对不同物理模型与真实数据的泛化；检验对训练误差（H2）依赖，网络自适应可能失效；在极端噪声下检验灵敏度有限。

---

## 80. Physics-Informed Sliding-Window Particle Filtering for Tactile-Only In-Hand 6-DoF Object Pose Refinement

**arXiv ID:** 2608.17601 | [PDF](https://arxiv.org/pdf/2608.17601v1)

**作者:** Lingjun Shao `[一作]` (Huazhong University of Science and Technology), Han Ding `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用全手掌3D触觉传感器对抓取物体进行6-DoF姿态细化和保持。

**💡 创新点**

提出物理信息滑动窗口粒子滤波器，将几何、力学、摩擦、非穿透等多重软约束结合，实现多模态保持。

**🔧 技术方法**

技术包括基于SE(3)的粒子滤波、签名距离场(SDF)、力与法向一致性、摩擦锥约束、潜在场引导、短期窗口融合、对称性识别和重定位。

**📊 数据集**

使用 Allegro Hand V5 配置的16个PaXini触觉模块，评估五个物体（电钻、连接器、立方体、瓶子、香蕉）抓取序列。

**📈 对比分析**

与ICP、ICP-force、Covariance T2G、Contact-SDF PF、Single-frame Physics PF、Dikhale-TactileOnly 等基线对比，平均中位数ADD-S为0.361，明显优于其它方法，说明融合物理约束和时间窗口能提升精度。

**⚠️ 局限性**

局限包括需要已知精确网格、受力传感校准误差、假设姿态在窗口内近似恒定、对称性仅支持单轴、SDF查询为瓶颈，未验证对动态抓取或形状不确定的适用性。

---

## 81. Effective Personalized AI Tutors via LLM-Guided Reinforcement Learning

**arXiv ID:** 2608.16907 | [PDF](https://arxiv.org/pdf/2608.16907v1)

**作者:** Angel Tsai-Hsuan Chung `[一作]` (University of Pennsylvania), Osbert Bastani `[通讯]` (University of Pennsylvania)

**通讯引用:** 3044 | [OpenAlex ID](https://openalex.org/A5029243071)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研发并部署了一个将GenAI聊天机器人与强化学习算法紧密结合的自适应问题排序平台，并在台北市10所高中进行大规模RCT；

**💡 创新点**

创新点在于利用LLM提取学生代码编辑与聊天对话的细粒度语义信号，构建连续知识状态估计，并通过粒子滤波+MPC实现个性化问题序列；

**🔧 技术方法**

采用GPT‑4o/Claude生成题目，LLM评判器评估聊天质量，粒子滤波+模型预测控制进行问题排序，数据记录包括时间戳日志、代码提交与对话；

**📊 数据集**

使用来自台北市10所高中770名学生的实验数据，涵盖10模块Python课程、每模块40道自动评分练习题，以及学生-平台交互日志；

**📈 对比分析**

通过ITT OLS对照固定序列与自适应序列，发现自适应序列提升0.156 SD（相当于6‑9个月学业），在初学者和低层级学校中效果更显著；

**⚠️ 局限性**

局限在于仅针对台北高中Python教学，缺乏跨学科推广；未检验长期保持；LLM生成题目仍需人工审核；难度分配非随机，难以分离难度对效果的直接影响；

---

## 82. Predict Before Replay: Joint FEC and Flight Control for Reliable Scale-Up Links

**arXiv ID:** 2608.17503 | [PDF](https://arxiv.org/pdf/2608.17503v1)

**作者:** Fan Yang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Zhan Wang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在高速加速器互连中设计了一种预 FEC 控制器 PREFACE，用于在短时相关突发误差出现前动态选择 FEC 强度并限制待确认的流量，以减少标准回放的放大效应。

**💡 创新点**

创新点在于：① 把预 FEC 的符号错误观测转化为两状态贝叶斯后验，用于同时决定 FEC 级别和“曝光窗口”；② 通过风险预算（暴露窗口）与 FEC 费用的联合最小化，显著降低回放成本；③ 在 ns‑3 模拟中实现了可验证的 UALink‑200G 1.0 回放语义。

**🔧 技术方法**

使用的技术包括：两状态贝叶斯过滤器、Gilbert‑Elliott 突发误差模型、成本驱动的 FEC 选择公式、风险窗口控制、ns‑3 仿真框架、UAS‑Link 标准回放实现、统计置信区间评估。

**📊 数据集**

使用的数据集主要是基于 Gilbert‑Elliott 参数合成的符号错误序列（10 条训练 seed、30 条 held‑out seed、3×3 鲁棒性网格和 3 条 100 万 flit 的长跑），并未采用真实物理链路测量数据。

**📈 对比分析**

与三种基线（固定强 FEC、后解码自适应、仅预 FEC）对比，PREFACE 在 30 条 held‑out seed 上实现：良用率提升 10.5%，P99 延迟降低 50%，回放 flit 数减少 47%；在环形 AllReduce 模型中，完成时间比基线快 13–27%。性能评估采用配对差值、95% 置信区间，并做了多种负载和突发长度的鲁棒性实验。

**⚠️ 局限性**

局限性包括：① 依赖预 FEC 符号错误计数作为早期信号，若链路不提供此信息需改用其他预测；② 评估基于合成突发模型，未验证在真实硬件、不同链路配置或多路复用场景下的表现；③ 需要在硬件实现中解决 FEC 轮换、状态同步和控制开销；④ 未探讨更复杂的学习预测器或选择性回放协议。

---

## 83. LEGO-RL: Harness-Native Reinforcement Learning for Coding Agents

**arXiv ID:** 2608.17393 | [PDF](https://arxiv.org/pdf/2608.17393v1)

**作者:** Yiming Du `[一作]` (Huawei Technologies Co., Ltd), Haoli Bai `[通讯]` (Huawei Technologies Co., Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

搭建 Lego-RL 框架，将现有编程代理 harness 与可扩展的策略梯度训练无缝连接。

**💡 创新点**

创新点在于三大支柱：保真优化的 in‑process LLM 代理、可扩展沙盒化执行与奖励完整性防御，以及集成的可观测训练插件与 Live UI。

**🔧 技术方法**

采用 in‑process 代理捕获 token、路由回放、异步 rollout 与同步训练，使用 verl、vLLM、FSDP、Megatron 等训练后端，并结合 Docker/K8s 沙盒。

**📊 数据集**

使用 OpenSWE 任务集构建的 2,699 任务索引，评估基准为 SWE‑bench Verified。

**📈 对比分析**

与同一基线 Qwen3.6‑35B‑A3B 与 KAT‑Coder‑V2.5‑Dev 对比，Lego‑RL‑Qwen3.5‑35B‑A3B 在三种 harness 上分别提升至 70.4%、68.2%、66.6%，相对提升 6.4%、5.8%、9.4%；并保持 rollout‑training 相关性 >0.99。

**⚠️ 局限性**

局限包括仅评估单一模型与 harness，训练运行一次导致方差未量化，奖励仅二元、缺乏中间信用，防御不保证对所有攻击，沙盒与镜像加速依赖部署环境。

---

## 84. Toward Personal Intelligence Through Cooperative Observation

**arXiv ID:** 2608.17128 | [PDF](https://arxiv.org/pdf/2608.17128v1)

**作者:** Yashar Talebirad `[一作]` (University of Alberta), Osmar R. Zaiane `[通讯]` (University of Alberta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了“合作观察”框架，探讨了个人AI在有限观察通道下如何构建用户目标模型，并通过一个名为Organizm的原型系统演示了基于用户自有文件、分层多智能体与任务相关的上下文压缩方法，进一步在单一用户上进行了六个月的使用记录；

**💡 创新点**

创新点在于：①引入合作观察循环，将用户的评价与信任反馈与观察通道的开启/关闭相耦合；②提出任务条件下的上下文分配与信息瓶颈理论，阐明如何在处理与披露约束下挑选压缩的观察记录；③设计分层代理与文件级内存相连的体系结构，兼顾本地可控性与持续学习；

**🔧 技术方法**

技术手段包括：大语言模型（LLM）驱动的代理，分层多智能体架构，文件原生内存与索引优先遍历，任务条件下的上下文压缩与信息瓶颈分析，用户手动日志与日历、财务等多源数据的本地同步与隐私控制；

**📊 数据集**

使用的数据主要为：手工记录日志（每日/每周/每月）、日历事件、共享的截止时间列表、财务通道（由用户自行加入）以及用户在Organizm中手工纠正的反馈；

**📈 对比分析**

论文并未给出定量对比实验，而是提出了三类评估方向：①观察通道消融实验（评估不同观察条件对任务性能的影响）；②纵向部署实验（评估辅助质量如何影响后续的观察共享/撤回）；③自我模型评估（评估系统知识与用户自知的变化）。因此无法给出具体性能数值；

**⚠️ 局限性**

局限性包括：①依赖用户持续且一致的反馈，负担较大；②隐私与安全风险高，尤其是高频/神经接口通道；③单用户的六个月记录缺乏因果推断与泛化能力；④未验证分层代理与传统平面内存的性能差异；⑤对信任、反思收益等概念缺乏操作性定义。

---

## 85. Optimal Adaptive Multi-Valued Byzantine Agreement

**arXiv ID:** 2608.17552 | [PDF](https://arxiv.org/pdf/2608.17552v1)

**作者:** Marc Dufay `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文提出了可适应多值拜占庭一致性协议，能够在大规模网络中实现接近最优的消息与时延复杂度。

**💡 创新点**

创新点在于将常数比例拜占庭容错下的二值协议推广到多值场景，并通过All‑to‑Quorum、Quorum‑Agreement、Quorum‑to‑All三个子协议，以及分散器（disperser）委员会分配、加密累加器与纠删码实现了对 n 与 t 的解耦。

**🔧 技术方法**

主要技术包括双向分散器构造、阈值签名与聚合签名、加密累加器、纠删码与错误码、散列预验证、扩散子图（expander）以及检索协议与证据冲突的多视图检索机制。

**📊 数据集**

本文没有使用具体的数据集，全部结果为理论证明与复杂度分析。

**📈 对比分析**

在同步、部分同步与异步模型下，协议分别实现了：同步下 O(n·(L+fκ)) 位复杂度与 O(f+log n) 轮；部分同步下 Õ(nκ + t·(L+fκ)) 位与 O(f) 轮；异步下期望 Õ(nκ + t·(L+tκ)) 位与期望 O(1) 轮；与先前的 O(nL) 方案相比，显著降低了通信量，几乎达到已知下界。

**⚠️ 局限性**

限制包括：在使用 All‑to‑Quorum 广播时，协议仅能达到 t < n/9 的可恢复性；部分同步模型假设完美时钟同步；分散器构造非构造性，需依赖概率或额外计算；对低 t 与高 t 的切换仍需要移除 AQB 部分，导致实现复杂。

---

## 86. DOMtutor: Automated Autograding for Logic in Computer Science

**arXiv ID:** 2608.16899 | [PDF](https://arxiv.org/pdf/2608.16899v1)

**作者:** Tobias Meggendorfer `[一作]` `[通讯]` (Lancaster University Leipzig), Tobias Meggendorfer (Lancaster University Leipzig)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出将自动评分系统（autograder）用于理论计算机科学课程，并通过domjudge平台与自研domtutor脚本实现自动批改和即时反馈；

**💡 创新点**

创新点在于将面向编程竞赛的成熟autograder技术迁移至理论课程，借助可定制的管理脚本降低教师负担，同时为学生提供即时、可重复的练习反馈；

**🔧 技术方法**

技术实现基于domjudge（ICPC竞赛级别的沙箱化评分器）、Kattis题目格式、Docker容器化部署以及Python脚本库domtutor，用于用户同步、结果导出、统计与评分；

**📊 数据集**

数据集主要是自定义的练习实例，例如有限自动机、线性代数运算、逻辑公式验证等，并使用隐藏测试用例进行边界情况验证；

**📈 对比分析**

虽然未给出量化实验，但作者在10个不同课程中部署并使用该系统，报告称教师工作量显著下降、学生参与度提升，系统能够即时反馈并支持多次提交；

**⚠️ 局限性**

局限性包括：需要教师对题目进行严谨、可自动化的定义，适配Kattis格式仍需一定人工；平台在某些非编程任务上可能功能受限；管理脚本虽然简化了操作，但仍需初期设置与维护工作；

---

## 87. Experiential Learning of Runtime Monitoring Using Pachinko

**arXiv ID:** 2608.16898 | [PDF](https://arxiv.org/pdf/2608.16898v1)

**作者:** Miles Scharff `[一作]` (Columbia University), Mark Santolucito `[通讯]` (Columbia University)

**通讯引用:** 231 | [OpenAlex ID](https://openalex.org/A5031902968)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作设计并实现了一项课堂作业——基于ESP32双核的互动Pachinko游戏，用来教学运行时监控与形式化规范的编写；

**💡 创新点**

创新点在于将正式方法（RTLola运行时监测）融入创意嵌入式课程，通过硬件演示（球轨迹、步进电机、音效）让学生在实践中理解并使用形式化规范；

**🔧 技术方法**

主要技术包括ESP32微控制器（双核）、RTLola C编译器、PlatformIO开发环境、ESP-NOW无线通信、铜带传感器、步进电机驱动及音效/动画驱动；

**📊 数据集**

本实验不使用公开数据集，而是通过自制球日志传感器实时采集球落点事件作为监控输入；

**📈 对比分析**

评估方法为让学生完成规范编写并实现功能，实验表明多数学生能成功完成规范，但硬件调试与连接是主要瓶颈，性能评估未量化；

**⚠️ 局限性**

主要限制包括RTLola C编译器缺乏对时间算子（如过去、未来、窗口聚合）的支持，导致学生需用C代码实现聚合；硬件连接复杂且易出错，影响学习体验。

---

## 88. Counting in Population Protocols on Graphs

**arXiv ID:** 2608.17590 | [PDF](https://arxiv.org/pdf/2608.17590v1)

**作者:** Petra Berenbrink `[一作]` (University of Hamburg), Dominik Kaaser `[通讯]` (Hamburg University of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种在任意基图上实现精确计数的人口协议，首次在图非完全（general graph）环境下实现了近线性状态复杂度 Õ(n) 与时间复杂度 O((G)·log²n + (G)·logn) 的计数协议；同时还给出了用于在图上产生独立公平随机比特的子协议和对 log n 的紧凑近似（误差仅为 O(log log n)）。

**💡 创新点**

创新点主要包括：
1) 通过引入两种红蓝代币并保证每个节点的红代币数始终为 2 的幂，显著降低了状态复杂度，从原本需要 Ω(n²) 状态的简单方案压缩到 O(log n) 状态；
2) 设计了一种基于“首缺失值”（first‑missing tuple）和精细化的采样分组策略，能够在图上用 O(log⁴n) 状态仅 O((G)·log²n) 交互就估计 log n；
3) 提供了无图参数的随机比特生成协议，利用调度器的随机方向实现独立公平比特，而无需事先的图知识；
4) 给出了终止检测的不可行性定理，阐明了在图均匀人口协议中无法在无先验图信息下可靠地检测计数完成。

**🔧 技术方法**

使用的关键技术包括：
- 随机化负载均衡（load balancing）与广播时间（broadcast time）的图性质结合来分析收敛时间；
- 对红代币的分布使用随机游走与“半分”操作的分析，证明每个红代币以常数概率在每个 epoch 内完成一次半分；
- 通过对蓝代币的标准离散负载均衡（balancing）证明其收敛到差异 ≤ 3；
- 利用“首缺失值”与递归广播实现 log n 估计；
- 对随机比特生成协议进行概率分析，证明每个节点每隔 O(|E|/δ(G)) 步可获得一次独立比特。

**📊 数据集**

论文未在特定实验数据集上进行评估，而是通过理论分析证明协议在任意连通图 G 上的性能。性能指标均以图的广播时间 (G) 与负载均衡时间 (G) 为基础给出。对常见图类（如完全图、正则图、环、d 维环面、最坏情况图）给出了具体的时间复杂度表达式。

**📈 对比分析**

与传统基于完全图的计数协议（如使用 coalescing random walks 或 phase‑clock 同步）相比，本协议在大多数图类（尤其是快速扩散图）下收敛时间更快，且在状态复杂度上几乎达到下界；在最坏情况下的收敛时间仍为 O(n⁴ log²n)，与已知的最优 O((G)·log n) 相差仅为多项式因子。

**⚠️ 局限性**

限制与未来工作：
- 需要唯一领导者，尽管可通过领导选举协议预先生成，但这一步仍需额外时间与状态；
- 协议仅保证稳定收敛，无法检测终止，导致在实际应用中需额外的终止检测机制；
- 状态复杂度虽然为 Õ(n)，但对极大图仍可能过高；
- 对图的广播时间与负载均衡时间的依赖意味着在扩散慢的图（如线性链）上收敛时间仍然高；
- 目前尚未提出无领导者版本或实现真正终止检测的方案。

---

## 89. TRUSS: Towards Task-Reliable and User-Safe Automated Agent Skill Generation

**arXiv ID:** 2608.17588 | [PDF](https://arxiv.org/pdf/2608.17588v1)

**作者:** Zhibo Zhang `[一作]` (Huazhong University of Science and Technology), Kailong Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于执行证据的 Agent Skill 自动生成与验证框架，将静态检查与动态执行跟踪相结合，实现功能有效性与安全可靠性的联合验证，并通过迭代改进。

**💡 创新点**

创新点在于：①把功能性验证与安全性验证放在同一统一流程中，②使用 Controllable Execution Environment (CEE) 收集可追溯的执行证据，③通过记录功能与安全责任的“责任链”引导修复，④实现了从 Artifact 级别到运行时的全程可追溯性。

**🔧 技术方法**

主要技术包括：LLM（GPT‑5.x）生成技能、静态静态分析与安全属性预检、动态执行（代理工具、可控沙箱）与中间断点（Intermediate Breaker）、功能与安全记录（Function & Safety Record）、迭代修复（Refiner）。

**📊 数据集**

使用的公开数据集：SkillInject（168 对照样本）、SkillSafetyBench（155 条件安全测试）和 SkillGenBench（187 个任务）。

**📈 对比分析**

方法对比：无技能、LLM 生成、完整框架。结果显示：①安全漏洞检测 100% 精度/召回；②攻击成功率从 38.71%/46.45% 降至 19.35%/29.68%，且无回归；③任务效果从 17.11% 提升至 52.94%，安全率从 50.80% 提升至 100%。

**⚠️ 局限性**

局限性：受限于训练模型与执行环境的覆盖范围，未能处理所有复杂的运行时状态；修复过程可能需要多轮迭代且耗时；对不同 Agent 配置的迁移性和可扩展性待进一步验证。

---

## 90. Cross-View Correspondence Is a Measurement Intervention: Two-Sided Validation for Agent Evaluation and Credit Assignment

**arXiv ID:** 2608.17713 | [PDF](https://arxiv.org/pdf/2608.17713v1)

**作者:** Zhen Zhang `[一作]` (Technical University of Munich), Amr Alanwar `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个两侧有效性（two‑sided validity）契约，用于验证跨视图对应（transport）和最优匹配（optimal matching）是否能正确保留响应，随后对所有最优对应进行下游推断和不确定性传播，揭示了隐藏对应导致的时间定位、信贷分配和严格度量误判。

**💡 创新点**

创新点包括：①将对应关系视为测量干预并给出完整的合法性合同；②构造线性可行性边界与“所有最优”集合的识别，提供无分布假设的证书；③将所有合法最优对应的编译结果聚合为合法更新体（legal update body），并证明在凸包内是否存在单一安全方向；④针对不同编译器（仿射、低树宽非线性、共享标准化）提出可扩展性与硬性边界；⑤在多种公开基准上通过大规模审计展示对应歧义与修正效果。

**🔧 技术方法**

使用的技术包括：最优匹配与对应函数（transport），线性代数与主角角度几何，凸包与最小范数分离理论，Frank–Wolfe/吉尔伯特求解器，min‑sum 消元（树宽算法），以及对齐与编辑距离的动态规划与最小/最大 semiring 计算。

**📊 数据集**

主要使用的数据集与任务：SWE‑bench、Spider2‑DBT、M3‑Bench、Earth‑Agent、MatchTIR 以及自定义的 SQL 与代码管道，用于评估时间定位、信用分配与严格度量的准确性。

**📈 对比分析**

与传统单一对应或默认选定策略比较时，作者发现：在1,586对非零轨迹中，55.9% 的时间定位结果因不同最优回溯而不一致；在多回合信用分配审计中，14/20 受影响任务组出现符号颠倒；严格度量修正后有 9 次排名逆转。实验中，合法更新体的凸包可在不到 10 次 Frank–Wolfe 步骤内收敛到极小误差（≤10⁻⁸）。

**⚠️ 局限性**

局限性包括：仅在单一模型（gpt‑5.5、Codex 等）与单一检查点下测试；审计为后验分析，未证明对最终策略性能的直接影响；使用的合成构造展示硬性边界，但真实部署场景的可扩展性尚待验证；共享标准化下的 NP/CoNP 难度仅在特定合成实例中体现，实际数据集中的复杂度未知。

---

## 91. From Abductive Explanations to Global Logical Rules for Node Classification in SGCs

**arXiv ID:** 2608.17103 | [PDF](https://arxiv.org/pdf/2608.17103v1)

**作者:** Bryan Lima Cavalcante `[一作]` (Instituto Federal do Ceará), Thiago Alves Rocha `[通讯]` (Instituto Federal do Ceará)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于最小归纳解释的 AXSGC 方法，用于解释 Simple Graph Convolution（SGC）模型的节点分类预测。

**💡 创新点**

创新点在于将节点-特征对的最小归纳解释（NF-AXp）作为中间表示，保证解释的最小性与充分性，并通过距离索引谓词聚合成决策树，得到紧凑、全局性的逻辑规则，显著提高规则简洁性和模型忠诚度。

**🔧 技术方法**

技术包括：SGC 的线性不等式推理、贪心最小化生成 NF-AXp、构造距离索引谓词、使用深度限制（12）决策树进行全局规则抽取、以及归纳解释（Abductive Explanation）框架。

**📊 数据集**

在四个节点分类基准上进行实验：BAShapes、Cora、Citeseer、PubMed。

**📈 对比分析**

与 LogicXGNN 进行对比，使用相同树深度12；AXSGC 在四个数据集上忠诚度提升最高 30.2%，规则数量减少最多 83.8%，同时保持甚至提升模型的准确率。

**⚠️ 局限性**

局限性包括：目前仅支持 SGC，其他 GNN 需要先蒸馏为 SGC；对极大规模图时 NF-AXp 的生成成本仍可能较高；未考虑动态图或异构图的扩展。

---

## 92. HODAgent: Towards On-Demand, Responsive Humanoids for Physical World Human Interaction

**arXiv ID:** 2608.17584 | [PDF](https://arxiv.org/pdf/2608.17584v1)

**作者:** Wang Warren Chen `[一作]`, Jie Chen `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于 System-2 的半双工人机交互框架，支持服务场景下的即时请求响应、任务连续性和基于证据的闭环执行。

**💡 创新点**

创新点在于将 Env-Interactor、Planner、Executor、Memory 四模块组合成可在仿真与真实机器人上共用的高层控制循环，实现对中断请求的即时处理和跨平台一致性。

**🔧 技术方法**

技术涵盖多模态 Omni 语言-视觉模型、任务分解与规划、执行器异步控制、共享观测与执行契约，以及 Qwen 系列 LLM 作为推理核心。

**📊 数据集**

使用了 164 个基于 SAGE-3D 的室内服务环境案例、Unitree G1 物理机器人、EmbodiedBench 与 VIGIL 公开基准。

**📈 对比分析**

与 ReAct 等基线对比，在仿真评估中 Joint Success 提升 9.8–18.9 点；在 G1 真实机器人上原子、组合、完整任务通过率分别为 92%、72% 和 63.3%；在 EmbodiedBench 与 VIGIL 上整体成功率提升 0.7–9.0 点。

**⚠️ 局限性**

局限性包括缺乏硬实时性能分析、物理实验仅在 Unitree G1 上验证、对长时任务的协调仍不足、缺少跨平台深度评估及自适应低级控制。

---

## 93. Mixture-of-Expert Blocks Contain Strong Hallucination Detection Signals

**arXiv ID:** 2608.17687 | [PDF](https://arxiv.org/pdf/2608.17687v1)

**作者:** Joao Fonseca `[一作]` (INESC-ID), Paolo Romano `[通讯]` (INESC-ID)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种单前向传递的 per‑token 假说检测方法，通过利用 Mixture‑of‑Experts (MoE) 的内部信号（如路由熵、专家不一致度、专家使用分布）与标准 Transformer 信号相结合，生成每个 token 的假说概率。

**💡 创新点**

创新点在于首次将 MoE 结构特有的内部信号与传统信号融合，构建轻量级分类器实现 per‑token 级别的假说定位，并通过无监督的 LLM‑as‑judge 标注管道实现持续更新。

**🔧 技术方法**

使用的技术包括：MoE 细粒度路由与专家激活统计、隐藏状态和注意力相关特征提取、特征标准化、轻量级机器学习分类器（Logistic Regression、Random Forest、XGBoost、MLP 等），以及 LLM‑as‑judge 评估。

**📊 数据集**

使用的数据集包括：训练时的 RealtimeQA（2024‑2025 年问答对与证据），以及多种测试集：Temporally OOD 的 RealtimeQA（2026 年）、SQuAD、TruthfulQA、NQ‑Open、FreshQA（各 200 题），评估在两种 MoE 主机模型 OLMoE 与 Gemma 上。

**📈 对比分析**

与采样基、内部信号基和可训练检测器基线比较，实验显示该方法在答复级别 AUROC 可达 0.91、token 级别 AUROC 可达 0.76，显著优于所有基线，并且只需单次前向传播，计算开销仅为原始模型的 2.5‑3 倍。

**⚠️ 局限性**

局限性包括：人类验证样本规模有限、仅评估英文数据、对多语言和其他模型架构的适用性未验证，以及在极端 OOD 场景下对专家信号的鲁棒性仍需进一步研究。

---

## 94. Margin-Regularized Structured Semantic Alignment for Brain-Language Correspondence

**arXiv ID:** 2608.16975 | [PDF](https://arxiv.org/pdf/2608.16975v1)

**作者:** Jiaqi Wang `[一作]` (Northwestern Polytechnical University), Shu Zhang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出MD‑SigLIP框架，通过脑信号编码器与T5文本编码器在共享语义空间中实现检索式脑‑文本对齐。

**💡 创新点**

创新点在于将多正样本监督与列表化margin正则化相结合，既捕捉语义簇内的多重正样本，又通过margin约束显式强化排名结构。

**🔧 技术方法**

技术核心为SigLIP对比学习（sigmoid‑based BCE）与列表化margin正则化，使用T5‑large文本编码器和自定义MEG编码网络。

**📊 数据集**

使用Armeni2022（MEG‑Audio）和SchoffelenRead2019（MEG‑Reading）两个公开MEG语言数据集。

**📈 对比分析**

与CLIP、SigLIP、D‑SigLIP等基线对比，MD‑SigLIP在Top 10准确率和中位数排名上均实现显著提升，达成state‑of‑the‑art。

**⚠️ 局限性**

局限在于仅验证了两类MEG任务，未探讨跨语言、跨任务或跨设备的泛化能力；模型对高噪声或低采样率数据的鲁棒性仍待进一步评估。

---

## 95. When Personalization Becomes Bias: Structural and Discursive Religious Framing in AI-Generated Financial Advice

**arXiv ID:** 2608.16909 | [PDF](https://arxiv.org/pdf/2608.16909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 96. Code as Representation: A Compilable Parsing Paradigm for Academic Documents

**arXiv ID:** 2608.17550 | [PDF](https://arxiv.org/pdf/2608.17550v1)

**作者:** Rihui Jin `[一作]`, Gholamreza Haffari `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Compilable Academic Document Parsing (CADP) 任务，将完整学术页面转换为可执行的 LaTeX + Python 代码，形成可重构的机器可读格式。

**💡 创新点**

创新点在于将文档解析从文本提取转向结构化、可执行双代码生成，支持表格、公式、图表等 Structured Academic Elements 的完整重构与可验证性。

**🔧 技术方法**

使用多模态大模型（MLLM）进行图像到代码的端到端生成，并设计多代理 (multi‑agent) 基线以探测模型上限；评估采用“重注入编译”协议，将生成代码编译回 PDF 并与原页面进行视觉比对。

**📊 数据集**

构建并发布 CADP‑Bench 基准，包含数百页的全页面学术文档，覆盖文本与至少两种 SAE 类型，并由专家验证注释与代码一致性。

**📈 对比分析**

通过与现有 MLLM 和基线模型对比，发现即使是最前沿模型在可执行重构上的精度仍低，表明在结构保留与可验证性方面存在显著差距；具体指标显示重构完整率仅在 50% 左右，图表数据恢复率更低。

**⚠️ 局限性**

局限性包括：1）模型对复杂嵌套表格与对齐公式的识别不佳；2）生成的 Python 代码往往不完整或错误，导致编译失败；3）评测仍依赖人工校对，难以大规模自动化。

---

## 97. The Price of Thinking: Reasoning Effort as a Model-Specific API Contract

**arXiv ID:** 2608.16956 | [PDF](https://arxiv.org/pdf/2608.16956v1)

**作者:** Yeabin Moon `[一作]` `[通讯]` (Brandeis University), Yeabin Moon (Brandeis University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在同一语言模型（Claude Sonnet 5）上对比显式高思维努力（high effort）与默认省略（omitted）两种 API 合同，探讨思维努力对推理成本与准确率的影响。

**💡 创新点**

创新点在于：① 把“思维努力”视为可交易的合同条款，而非模型级别属性；② 使用已注册的配对对照实验与项目化抽样，严谨记录成本、终端结果与价格；③ 将成本与正确率结合，计算每次正确答案的实际成本，为商用决策提供量化依据。

**🔧 技术方法**

主要技术手段包括：API 请求管理、成本重构、终端结果分类（正确/错误/无答/服务失败等）、基于项目抽样的百分位自助法（item‑clustered bootstrap）以及精确的成本与准确率区间估计。

**📊 数据集**

数据集为 2026 年 AIME（美国数学竞赛）公开题集中的 30 道题目（无题目文本与答案，仅保留题目编号和答案），并在每道题上进行 5 次调用。

**📈 对比分析**

比较方法：配对对照（每个题目在两种合同下各5次调用），计算平均成本、准确率和成本/正确率。实验结果显示：高思维努力合同平均成本高 0.01031 美元/次（区间 +0.00204 ~ +0.01974），准确率差异不显著（+0.0133，区间 -0.0267 ~ +0.0467），但成本/正确率略高（0.08665 vs 0.07662 美元/正答）。

**⚠️ 局限性**

局限性：① 仅有 30 道题目与一次会话，结果不具备普遍性；② 调度顺序固定，缺乏随机化；③ 只比较同一模型内的两种合同，未考虑模型间差异；④ 结果受具体价格计划与服务条款变化影响；⑤ 仅评估单轮推理，无法揭示多轮交互中的成本动态。

---

## 98. Counterfactual Anatomy-guided Spatial-Temporal Decoding for Annotation-Free Hallucination Mitigation in Medical VLMs

**arXiv ID:** 2608.17427 | [PDF](https://arxiv.org/pdf/2608.17427v1)

**作者:** Yifan Lu `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Imran Razzak `[通讯]` (MedOS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了CAST框架，在Med-VLM的推理阶段通过自动发现并选择合适的解剖区域来抑制幻觉，且不需要任何人工标注；

**💡 创新点**

核心创新在于使用因果干预的计数方法自动选取紧凑且与问题相关的ROI，并将其用于空间-时间对比解码；

**🔧 技术方法**

结合了多概念医学分割（MedSAM3）、逆向干预计数、区域条件指导（CFG）和逐步时间对比的解码策略；

**📊 数据集**

在SLAKE（多模态CT/MRI/X光QA）和MIMIC‑CXR（胸部X光QA）两个医学问答数据集上进行评估；

**📈 对比分析**

与VCD、DoLA、OPERA及GT依赖的ARCD等解码方法比较，CAST在三种Med‑VLM上均实现了整体指标提升，尤其在闭合式问题准确率显著提高；

**⚠️ 局限性**

局限性包括对分割模型质量的依赖、对遮挡干预方式的敏感性以及在某些高分辨率或复杂解剖结构场景下可能需要更高的计算开销。

---

## 99. A Black-Box Workload Barrier for Exact Girth via Multi-Scale Nearest-Source Estimation in CONGEST

**arXiv ID:** 2608.17358 | [PDF](https://arxiv.org/pdf/2608.17358v1)

**作者:** Indraveni Chebolu `[一作]` (Centre for Development of Advanced Computing), Arnab Mallick `[通讯]` (Centre for Development of Advanced Computing)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在CONGEST模型中研究将多尺度最近源估计器通过黑盒重构来实现精确求环长（girth）的算法。

**💡 创新点**

创新点在于提出了“黑盒精确化”范式，并在一族特殊图 H_t 上证明了其必需的保留源工作量至少为 Ω(n/ log n)，从而展示了近似方法无法仅通过参数重调即转为多项式子线性精确算法。

**🔧 技术方法**

主要技术包括可交换的随机源抽样、邻源表（nearest‑source table）的构造、基于排名的概率上界、对称性与线性“拌合”论证，以及对标准分包实现的轮数分析。

**📊 数据集**

使用的“数据集”是构造的硬实例族 H_t——奇数环加上每个环顶点对应的完整二叉树，保证图度≤3、直径≈log n，并在此族上进行理论证明。

**📈 对比分析**

与现有的 f‑近似算法相比，黑盒精确化在 H_t 上需要至少 Ω(n/ log n) 轮（或相等量的保留源工作量），无法达到 O(n^{1-ε}) 的时间复杂度；而仅在有图感知预处理或跨调用共享信息时可能突破此界。

**⚠️ 局限性**

局限性是该结论仅适用于不保留或共享内部最近源表、仅利用标量输出做自适应停止、且源集合保持可交换的黑盒方法；若算法具备图感知预处理、跨调用状态压缩或额外循环检测步骤，则该下界不再适用。

---

## 100. Collective Ranking of Environmental Signals through Gaussian Belief Propagation in a Patrolling Robot Swarm

**arXiv ID:** 2608.17690 | [PDF](https://arxiv.org/pdf/2608.17690v1)

**作者:** Zachary R. Madin `[一作]` (University of Bristol), Edmund R. Hunt `[通讯]` (University of Bristol)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在机器人群体巡逻环境中，提出利用巡逻图同时作为运动拓扑和因子图的双重作用，对连续的环境信号进行集体排名，并通过分布式消息传递实现对信号强度的估计。

**💡 创新点**

创新点在于：①将巡逻图映射为因子图，实现了Gaussian Belief Propagation（GBP）在巡逻任务中的首次应用；②提出双重使用的图结构，使得信息融合能充分利用空间相关性；③在仿真与真实机器人实验中验证了GBP在噪声鲁棒性、排名准确度和收敛速度方面优于传统平均方法。

**🔧 技术方法**

技术包括：Gaussian Belief Propagation、加权与无权平均融合、两种巡逻算法（Cyclic Graph Generation CGG 与 State‑Exchange Bayesian Strategy SEBS）、Python仿真平台、Leo Rover机器人、XBee Zigbee RSSI测量。

**📊 数据集**

数据集：仿真中使用Beta分布采样并按半径赋值的信号强度（模拟连续信号场）；真实实验中使用办公室大堂的XBee RSSI信号，作为实际传播信号的测量数据。

**📈 对比分析**

比较方法：在相同巡逻轨迹下，使用三项指标——平均平方误差（MSE）、Spearman排名相关系数（ρ）和收敛时间。结果显示，GBP在所有噪声水平下始终取得最低MSE、最高ρ、最快收敛，平均方法在噪声超过25%时显著退化。

**⚠️ 局限性**

限制：GBP相对平均方法计算量更大，未对大规模多源信号、定位误差或不同图拓扑的鲁棒性进行系统评估；实验规模仅为4–8台机器人，未探讨更大规模群体的可扩展性。

---

## 101. TENET: Telegram Mini App (in)security

**arXiv ID:** 2608.17538 | [PDF](https://arxiv.org/pdf/2608.17538v1)

**作者:** Andrea Ciccotelli `[一作]` (King Abdullah University of Science and Technology), Roberto Di Pietro `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Telegram Mini App生态中客户端存储安全进行了系统性测量和漏洞分析，并基于此开发了专用审计工具TENET。

**💡 创新点**

创新点包括：①针对Mini App存储秘密的模式与阈值设计（27条正则、熵阈值、字符集校验）；②基于三级严重性分类（明文存储、可恢复加密、可重放令牌）的漏洞分级；③构建公开的基准数据集并验证工具性能；④首次在Telegram官方Wallet上验证补丁效果。

**🔧 技术方法**

技术手段：正则表达式匹配、熵分析、字符集验证、SQLite/LevelDB解析、API调用逆向、JSON Web Token解析以及自研的TENET审计框架。

**📊 数据集**

数据集：61款Mini App（按受欢迎度加权抽样）中成功分析37款；另外使用20款Mini App共500条存储条目构成的真值基准集进行评测。

**📈 对比分析**

性能评估：在基准集上，TENET的精确率94.3%、召回率96.7%、F1值95.5%；单账号扫描平均耗时约2.1秒，支持多账号线性扩展；与通用秘密扫描工具相比，TENET能够解析Telegram特定的LevelDB/SQLite存储并覆盖更广泛的秘密类型。

**⚠️ 局限性**

局限性：仅针对桌面端（Windows/macOS/Linux）进行评测，移动端需要root/越狱才能获取；只检测本地存储，未覆盖网络传输层安全；样本规模受限于可访问的Mini App数量；工具对加密方式的自定义处理仍有误报与漏报风险。

---

## 102. Overview of the TREC 2025 Product Search and Recommendation Track

**arXiv ID:** 2608.17138 | [PDF](https://arxiv.org/pdf/2608.17138v1)

**作者:** Dean E. Alvarez `[一作]` (University of Illinois Urbana-Champaign), Michael D. Ekstrand `[通讯]` (Drexel University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出2025年TREC产品搜索与推荐赛道，包括查询扩展和商品关系推荐任务。

**💡 创新点**

创新点是构建端到端评测数据集，并首次提供手工标注的互补与替代商品关系标签。

**🔧 技术方法**

使用的技术包括BM25基线、LLM自动查询重写、RM3伪相关反馈、多模态特征融合等。

**📊 数据集**

使用的数据集基于2022年KDD Cup ESCI，扩展到180万商品、30k训练/3k开发查询，含多语言查询与人类评估标签。

**📈 对比分析**

通过MAP、nDCG、召回等指标与BM25基线对比，LLM和RM3在某些子集取得显著提升，但整体提升有限，硬查询上仍表现不佳。

**⚠️ 局限性**

局限性在于推荐任务仅单一团队提交，评测样本受限；查询难度差异导致系统改进不稳定；缺乏对话式搜索和多任务统一模型的支持。

---

## 103. The Influence of Agent Models on the Complexity of Bus Routing

**arXiv ID:** 2608.17733 | [PDF](https://arxiv.org/pdf/2608.17733v1)

**作者:** Eva Deltl `[一作]` (Technische Universitaet Clausthal), Luca Pascal Staus `[通讯]` (Friedrich Schiller University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究多公交站点问题（MBSP），系统分析不同目标函数和乘客行走成本模型对问题复杂度的影响，并给出多种输入图（通用图、树、星形、路径）与模型（任意、统一、单位、无权重）下的多项式或参数化难度结果；此外，作者为能量目标提供了树和路径的动态规划求解方法。

**💡 创新点**

创新点在于：①首次将目标函数与乘客成本模型结合起来，揭示了两者如何共同决定问题难度；②在树和星形图上给出完整的硬度表，证明了即便在极其简化的图结构上，任意成本模型也会导致W[2]-难度；③提出了针对能量目标的高效树/路径动态规划，并证明其在不同行为模型下的多项式可解性；④将理论结果与实际公交规划相结合，利用NYC-M15线路和Citi Bike出行数据验证不同目标函数下停靠站点的实际差异。

**🔧 技术方法**

核心技术包括：从Hitting Set、Clique、Independent Set等经典W[1]/W[2]难点问题构造参数化归约；利用树的二叉化与根化，将动态规划状态压缩到O(n^4k^2+|A|)；对路径采用基于左右停靠点的DP；以及对能量目标在无权重时的简化分析；在实验部分使用现有公交站点候选集与Citi Bike轨迹对三种停靠策略进行评估。

**📊 数据集**

实验使用NYC-M15公交走廊的候选站点以及Citi Bike NYC Trip History数据（随机生成的行走与公交成本比例）。

**📈 对比分析**

实验对比了三种k=10停靠方案：最小化g_energy、最小化f_energy与均匀间距。结果显示两种最优方案相较于均匀方案分别降低约16.5%和16.8%的目标成本；图示表明两种目标对应的停靠站点分布存在差异，反映了目标函数的不同侧重点。

**⚠️ 局限性**

局限性包括：对路径图在所有目标函数与成本模型组合下的完整复杂度图仍未完全确定；只在树、星形、路径等特定图上给出结果，尚缺乏更广泛图结构的正解；实验仅覆盖NYC-M15走廊，未验证在其他城市或更大规模网络中的表现；对乘客数量大但成本模型有限的情况缺少FPT分析。

---

## 104. The Plot Thins: Uniformity and Linearity in Literary Summaries

**arXiv ID:** 2608.17218 | [PDF](https://arxiv.org/pdf/2608.17218v1)

**作者:** Rebecca M. M. Hicke `[一作]` (Cornell University), Ross Deans Kristensen-McLachlan `[通讯]` (Aarhus University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个150本公共领域小说与其维基百科情节摘要的对齐数据集，并开发了基于LLM的句子-章节映射流水线，对摘要进行线性与均匀性分析。

**💡 创新点**

创新点在于：①首次系统地对文学摘要与原文进行句子-章节级对齐；②提出并量化线性度、均匀性及其组合指标；③通过LLM和人工双重标注验证映射的可行性，揭示摘要写作中的结构性偏差。

**🔧 技术方法**

技术手段包括手工与LLM混合标注、Qwen 3.6 27B 的三阶段映射提示、Kendall tau、Gini 系数、平均偏离距离等统计度量；对齐过程还利用了章节分割、句子分割等预处理。

**📊 数据集**

数据集来源于Project Gutenberg 与 BookSum 中的150本公版小说及其对应的维基百科情节摘要（共约 147,000 词的原文与 992 词平均长度的摘要）。

**📈 对比分析**

对映射流水线的评估基于4本小说的人工标注，平均 F1≈84.3%（范围70–92%）。随后利用对齐结果对所有摘要进行线性度、均匀性等指标计算，并与随机置换基线进行比较，揭示大多数摘要保持高度线性（≈92%≥0.9）但存在显著均匀性偏差。

**⚠️ 局限性**

限制包括：①映射误差在部分小说（如 Faulkner、Rookwood）较高；②仅覆盖英文公共领域文本和维基百科摘要，难以推广到作者单独写作或 AI 生成的摘要；③对齐方法依赖 LLM 的上下文长度和温度设定，结果受模型版本和提示细节影响。

---

## 105. Structure-Internalized Rule Language Model for Faithful Knowledge Graph Reasoning

**arXiv ID:** 2608.17443 | [PDF](https://arxiv.org/pdf/2608.17443v1)

**作者:** Xingrui Zhuo `[一作]` (Hefei University of Technology), Xindong Wu `[通讯]` (Hefei University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种新的结构内部化规则语言模型 SIRLM，用于在大型语言模型中实现对知识图谱结构的精准感知与推理。

**💡 创新点**

创新点在于解决 LLM 在知识图谱推理中出现的“证据感知漂移”问题，通过结构内部化规则生成器、KG tokenizer 及神经符号推理器，将 KG 的结构知识与 LLM 参数知识有效对齐，并提供规则执行的真实性反馈。

**🔧 技术方法**

使用了结构内部化规则生成器（SIRG）结合结构关系记忆（SRM）实现多模态指令输入；KG tokenizer 基于结构不变性学习（SIL）进行实体/关系嵌入；神经符号推理器基于规则约束消息传播（RCMP）以及 NBFNet；并在 LLM 训练中集成了 SFT 与 GRPO。

**📊 数据集**

在 36 个知识图谱推理基准上验证，包括 4 个传统转导任务（FB15k-237、WN18RR、CoDEx-M、NELL995）、12 个实体诱导任务（IndE）与 20 个全诱导任务（FullInd）等。

**📈 对比分析**

与 17 类基线（嵌入、规则、GNN、LLM）对比，SIRLM 在转导、IndE 与 FullInd 场景下均实现了显著提升，平均提升幅度约 5-12% 的 MRR/Hit10。

**⚠️ 局限性**

局限性：对小规模 LLM 的适配性不足，参数容量限制导致规则生成精度下降；计算成本相对较高，尤其是 NBFNet 与 RCMP 的双向传播；以及对极大规模 KG 的可扩展性尚未充分验证。

---

## 106. Pathology Transport: Optimal-Transport Explanations for Clinical Data, and When Their Heatmaps (Fail to) Localize Disease

**arXiv ID:** 2608.17370 | [PDF](https://arxiv.org/pdf/2608.17370v1)

**作者:** Lalit Kumar `[一作]` `[通讯]` (University of Texas at Austin), Lalit Kumar (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究者提出一种基于最优传输的解释性AI框架，利用OT耦合的rectified flow直接学习健康与疾病分布之间的几何关系，从而生成个体反事实、无监督风险评分和全局归因；

**💡 创新点**

创新点在于不依赖训练分类器，单一传输模型即可同时提供反事实生成、风险评估与解释，可跨模态应用；

**🔧 技术方法**

核心技术包括最优传输耦合的rectified flow、欧拉ODE积分、以及小批量OT配对；

**📊 数据集**

实验数据分别为乳腺癌Wisconsin诊断表格数据（569例，30个核生物标记）和胸部X光图像（PneumoniaMNIST和RSNA肺炎检测数据集）；

**📈 对比分析**

与传统逻辑回归、距离度量、平均偏移和梯度增强方法比较，风险评分AUROC约0.91（低于logistic回归0.99），反事实有效率84%，归因相关系数r≈0.49，定位实验表明无监督热图在合成病灶上能提升定位，但在真实RSNA病灶上表现与随机相当，只有监督Grad‑CAM略优；

**⚠️ 局限性**

局限包括：未能突破监督方法性能，缺乏校准的风险评分，合成-真实定位差距大，数据规模小且不具代表性，且方法对参数和架构敏感。

---

## 107. The 10th AI City Challenge

**arXiv ID:** 2608.17044 | [PDF](https://arxiv.org/pdf/2608.17044v1)

**作者:** Zheng Tang `[一作]` (NVIDIA), Rama Chellappa `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统性概述了第十届 AI 城市挑战赛（AI City Challenge 2026）的组织架构、八个评测排行榜、六大核心任务与两项异域任务，并对参赛团队在多摄像头 3D 感知、合成-实景安全理解、异常事件推理、文本检索、视频生成与跨城市检测等场景下的实验结果进行了分析与总结。

**💡 创新点**

创新点在于将传统目标检测与追踪扩展为跨模态、跨域、多任务的全景城市智能基准，并通过引入 fisheye 摄像头与行人意图问答等异域评测，全面检验模型的泛化与推理能力。

**🔧 技术方法**

所采用技术涵盖基础视觉语言模型（VLM）、几何与多摄像头关联技术、检索与再排序机制、生成式扩散模型、域自适应与预训练策略、以及多模态推理代理框架。

**📊 数据集**

主要使用的数据集包括：PhysicalAI‑SmartSpaces（合成仓库场景）、Digital Twin WTS（合成与实景交通安全视频）、TAR‑Bench（交通异常推理）、Pedestrian Anomaly Behavior（文本检索）、Traffic Video Forecasting（历史到未来帧生成）以及 Hafnia（跨城市目标检测）等。

**📈 对比分析**

通过公开/通用排行榜对比，获胜团队在 3D HOTA、VQA/Caption BLEU/METEOR、TAR‑mean、检索 mAP、视频质量 FID/LPIPS 以及跨城 mAP 等指标上分别取得 56.5%、60.1%、0.68、99.3%、76.5% 与 0.48 的高分，体现了在多任务与跨域场景下的先进性能。

**⚠️ 局限性**

局限性包括：对跨域与真实世界数据的泛化仍显不足，尤其是 3D RGB‑only、视频生成与 OOD 推理任务；部分基准对模型可解释性与可复现性评估不足，导致对最终系统完整性评价有限；整体任务难度分布不均，某些指标仍缺乏统一的衡量标准。

---

## 108. Explainable AI-Powered Framework for Video-Based Skill Assessment in Cataract Surgery

**arXiv ID:** 2608.17522 | [PDF](https://arxiv.org/pdf/2608.17522v1)

**作者:** Mohammad Javad Ahmadi `[一作]` (K.N. Toosi University of Technology), Hamid D. Taghirad `[通讯]` (K.N. Toosi University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了全球最大的白内障手术视频数据集（2000 条）并开发了可解释 AI 框架，利用手术视频中的运动数据自动评估手术技能。

**💡 创新点**

创新点在于：① 数据集规模史无前例；② 引入 Capsulorhexis Skill Assessment System（CSAS）实现客观标注；③ 通过像素级运动提取十个可解释指标，并证明其与专家评分高度相关；④ 强调模型可解释性而非黑盒。

**🔧 技术方法**

使用计算机视觉分割跟踪（SAM‑2 / YOLO‑11），相对坐标引用与缩放，提取速度/加速度/jerk 等运动特征，计算十个性能指标；通过回归分析和聚类（K‑Means、Agglomerative、Spectral 等）验证指标有效性。

**📊 数据集**

主要数据集为 2000 条真实白内障手术视频，其中 83 条经过 CSAS 主观评分的 CVSAD‑83 子集；该数据集为首个公开的大规模真实手术视频数据库。

**📈 对比分析**

将十个自动指标与专家整体评分进行回归比较，相关性强；通过多种聚类方法将视频划分为专家/中级组，最优指标 NAM 的聚类准确率达 87%，整体技能分类准确率最高可达 87%。

**⚠️ 局限性**

局限性包括：仅验证 capsulorhexis 阶段；缺乏多机构、多手术类型的外推验证；实时反馈与临床工作流集成尚未实现；对其他手术阶段和复杂度的通用性未知。

---

## 109. Position: Fairness Failure in Generative Models is an Evaluation Problem

**arXiv ID:** 2608.16974 | [PDF](https://arxiv.org/pdf/2608.16974v1)

**作者:** Mariia Vladimirova `[一作]` (Criteo), Thibaut Issenhuth `[通讯]` (Criteo)

**通讯引用:** 66 | [OpenAlex ID](https://openalex.org/A5008216150)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“Fairness Cards”——一种用于生成式 AI 的最小化、标准化的公平性评估报告模板，并通过对 Qwen2.5‑7B‑Instruct 的系统性对照实验展示了评估协议对公平性结论的决定性影响。

**💡 创新点**

创新点在于：①将公平性评估视为可重复、可比较的实验对象；②明确列出评估过程中的所有自由度（提示族、采样/种子、拒绝处理、切片定义、评分管道）；③通过对照实验证明同一模型在不同评估协议下可出现相反的公平性判断，从而凸显评估规范化的必要性。

**🔧 技术方法**

使用的技术主要是：生成式语言模型（Qwen2.5‑7B‑Instruct）、多样化提示族（F1–F4）、多重变体（五种释义、四种职业）、不同解码策略（低熵 vs 高熵）、多种随机种子、自动化评分工具（stereotype‑keyword、demeaning‑language、身份显著性等），以及可视化和统计工具来汇总结果。

**📊 数据集**

数据集：自定义的 3,200 条提示（320 个独特提示，涵盖 4 个交叉切片 × 4 职业 × 5 释义 × 2 解码模式 × 5 种子 × 4 提示族）；此外论文中还引用了 Stable Diffusion 图像生成、Mistral 角色分配等实验，主要用于演示公平性失效模式。

**📈 对比分析**

比较方法：对同一模型在不同评估协议（提示族、种子、解码策略）下的公平性指标进行对照；使用“worst‑slice”阈值（5%）作为判定规则，展示不同协议导致的真/假判定差异。实验显示，最差切片的 stereotype‑keyword 率在不同提示族间差异可达 0.065–0.23，阈值判定完全翻转；随机种子与解码熵的变化也会导致近阈值判定的波动。性能表现主要说明评估协议对公平性结论的显著影响，而非模型性能提升。

**⚠️ 局限性**

局限性：Fairness Cards 仅规范评估报告，无法直接消除偏差；若数据或安全策略保密，报告仍受限；对“公平性”本身的价值判断未做统一；在高度闭源或快速迭代的 API 环境中，协议可变性仍难以完全掌控；缺乏针对不同应用场景的具体缓解措施。

---

## 110. Governing Delegation to Generative Artificial Intelligence: Human Direction, Work-Related Orientation, and Modes of Use

**arXiv ID:** 2608.17624 | [PDF](https://arxiv.org/pdf/2608.17624v1)

**作者:** Jorge Fábrega `[一作]` `[通讯]` (CICS-UDD), Jorge Fábrega (CICS-UDD)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文基于Anthropic Economic Index数据，构建了两种人机交互方向的度量指标，分别为指定委托和迭代共创，并检验工作相关使用比例上升是否会增强这两种方向的显著性。

**💡 创新点**

创新点在于：①首次将人类对生成式AI的指令、约束、测试等前置控制与后置的迭代干预统一成可度量的“方向配置”指标；②通过对工作与个人使用比例的合成干预，模拟了使用情境变化对方向特征的影响；③将两种API交互模式（1P API与Claude.ai）作为观察窗口，揭示了不同交互环境下方向表现的差异。

**🔧 技术方法**

技术方法包括：构造几何均值的方向配置（SD_g、IC_g）；对比度量采用分数响应的二项式回归（logit 链接）；使用基于节点的分层固定效应模型；通过2000次节点级聚类自助抽样估计不确定性；并运用等比组成分析（ILR）及不同聚合器来检验结果稳健性。

**📊 数据集**

使用的数据集为Anthropic Economic Index（AEI）公开版，覆盖2026年4月和5月的月度汇总单元，包含两类交互来源：1P API 与 Claude.ai，总计约20,532个单元，研究重点选取O*NET第0层任务视图（4,285个1P API单元和5,111个Claude.ai单元）。

**📈 对比分析**

比较方法是：对每个单元在当前使用构成与将个人使用比例上调10%至工作使用构成的两种情景下，预测方向配置的差值；随后在可比节点‑月份对上，计算Claude.ai与1P API之间的差异。结果显示：指定委托在两种模式下均显著提升（1P API +2.76，Claude.ai +1.45；95% CI均>0），而迭代共创在Claude.ai呈正增幅（+0.15），1P API呈负或零增幅（-0.30），两者差异为+0.45（95% CI 0.15–0.75），均支持假设。

**⚠️ 局限性**

局限性包括：①分析基于聚合单元，缺乏个体、组织或完整流程信息；②无法区分内部与外部迭代，尤其1P API的交互可能在观察单元之外完成；③仅使用两个月的数据，可能受季节性或特殊事件影响；④分类与指标可能存在误差，导致方向配置估计偏差；⑤不具备因果识别，结果仅为相关性。

---

## 111. GADR: Gathering Architecture Decision Records from Meeting Transcriptions

**arXiv ID:** 2608.17694 | [PDF](https://arxiv.org/pdf/2608.17694v1)

**作者:** Lucas Daniel Costa da Silva `[一作]` (Universidade Federal de Pernambuco), Kiev Gama `[通讯]` (Universidade Federal de Pernambuco)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了GADR，一种基于多代理、可自我纠错的工作流，能够从原始会议转录中提取建筑决策并生成符合Nygard格式的ADR草稿。

**💡 创新点**

创新点在于将任务拆分为语言检测、翻译、决策提取、批判评估、RAG丰富和最终格式化等若干节点，并使用共享状态（LangGraph）实现多代理协作与可审计的自我纠错循环，解决了单通道提示在噪声对话中的注意力衰退问题。

**🔧 技术方法**

采用了Gemini 3.1 Pro LLM、Tavily API、ChromaDB检索、LangGraph图式工作流、翻译与批判模块，以及自定义的评估与RAG增强步骤。

**📊 数据集**

使用了五个真实项目会议的转录（来自四个学生团队和一个研究团队），以及MSR公开ADR数据集作为RAG检索知识库与少量示例。

**📈 对比分析**

对比了零样本、少样本和多代理工作流三种生成方式，实验表明多代理流程在ADR数量变异性、结构一致性和完整性上优于基线，且在专家与学生评估中获得了约90%的正确率与清晰度，但整体输出更冗长，需要人工审阅。

**⚠️ 局限性**

局限性包括：1) 对临时讨论的误判为已采纳决策；2) RAG检索可能引入与会议无关的事实，导致不准确或不可信的量化声明；3) 评估样本有限、主要来自教育环境，难以推广至工业场景；4) 生成的ADR仍需人工校正，无法完全自动化。

---

## 112. What Tokens are Learned when Tokenization is Optimized Jointly with Language Modeling?

**arXiv ID:** 2608.17325 | [PDF](https://arxiv.org/pdf/2608.17325v1)

**作者:** Saketh Reddy Vemula `[一作]` (Indian Institute of Information Technology Hyderabad), Parameswari Krishnamurthy `[通讯]` (Indian Institute of Information Technology Hyderabad)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了在18种不同语言下，联合优化tokenization与语言建模的两种无tokenizer方法（SSLM与H-Nets）与传统基于频率的tokenizer（BPE、ULM、WordPiece等）的差异，评估它们在形态学对齐、词汇特性以及下游NLP任务中的表现。

**💡 创新点**

首次在多语言、多书写系统环境下对tokenizer-free模型的学习动态与词汇结构进行系统对比，揭示SSLM在形态学上更佳对齐、H-Nets在字节层面更高效，并证明联合学习的token在下游任务中保持竞争力。

**🔧 技术方法**

使用Transformer版SSLM（分段语言模型）和端到端的H-Net；对比BPE、ULM、WordPiece等传统子词算法；利用MorphScore、肥度、上下文指数等内部指标；在BERT（12M参数）上做下游微调，评估Sentiment、POS、NER、依存句法等任务。

**📊 数据集**

采用WMT News Crawl及NLLB等公开新闻语料，构建18种语言的平行规模（以英文250k句为基准，按字节比例缩放）；另使用10M句的新闻语料进行BERT预训练。

**📈 对比分析**

通过内部指标（形态对齐F1、肥度、上下文指数、有效词表大小）与外部指标（下游任务准确率/ F1/ LAS）对比；结果显示SSLM预token化的BERT在验证困惑度上最低，且在Sentiment、POS、NER、依存句法上与传统tokenizer相当甚至更优；H-Nets在词表重叠极低，形态对齐差。

**⚠️ 局限性**

实验规模受限于3M/12M参数模型、250k/10M句子语料，未覆盖所有18种语言的下游任务；未对SSLM超参（如最大token长度）做全面搜索；缺乏大规模可扩展性验证，故结果在更大模型和语料上可能不同。

---

## 113. Monitoring Pasture Restoration from Satellite Image Time Series: Caveats and Opportunities

**arXiv ID:** 2608.17704 | [PDF](https://arxiv.org/pdf/2608.17704v1)

**作者:** Linnea Sartorius `[一作]` (Lund University), Aleksis Pirinen `[通讯]` (RISE Research Institutes of Sweden)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究使用 Sentinel‑2 图像时间序列与深度学习，对瑞典 1397 片半自然牧场的恢复状态进行二分类识别；

**💡 创新点**

首次将牧场恢复状态作为二分类任务，并通过月度合成堆叠与场内归一化显著提升识别效果，同时通过时间偏差分析揭示年份相关混淆；

**🔧 技术方法**

采用自定义残差 2D CNN 与 CNN–LSTM 混合模型，并结合全局/场内归一化、月度复合、植被指数及云检测等预处理技术；

**📊 数据集**

使用 2018‑2025 年的 Sentinel‑2 L2A 夏季（6‑8 月）图像与瑞典农业部提供的 1397 片牧场多边形及恢复时间标签；

**📈 对比分析**

对比单季复合、月度堆叠、植被指数等输入与归一化方案，结果显示月度堆叠+场内归一化可达 0.88 准确率，CNN–LSTM 多年版实现 0.92 召回率，但模型对年份存在显著偏差；

**⚠️ 局限性**

由于恢复状态与年份高度相关，模型存在时间偏差，去除年份信息后精度显著下降；缺乏时间平衡标签与真正未恢复轨迹限制了方法的可推广性。

---

## 114. Cross-Domain Joint DDoS Detection in Multi-Controller SDN via Confidence-Based Entropy Fusion

**arXiv ID:** 2608.17507 | [PDF](https://arxiv.org/pdf/2608.17507v1)

**作者:** Zhaoyang Zhang `[一作]` (Beijing University of Posts and Telecommunications), Xiaofeng Tao `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多控制器SDN环境中研究分布式DDoS检测，发现传统基于熵的检测在聚合控制器上会出现聚合偏差，并提出一种跨域置信融合框架，通过边缘控制器轻量化消息校准聚合控制器的判定；

**💡 创新点**

创新点在于首次系统定位并解释聚合偏差的根本原因（统计延迟与阈值漂移），并设计了基于置信度的跨域融合机制（包括置信度计算、边缘一致性聚合、阈值防护三条规则），实现无原始流量交换、零影响边缘控制器的增量部署；

**🔧 技术方法**

技术包括基于Shannon熵的流量特征、指数加权移动平均动态阈值、置信度映射、边缘到聚合的轻量化JSON报文、三条融合规则（置信抑制、阈值恢复、阈值下限保护），以及Mininet仿真平台与Ryu控制器实现；

**📊 数据集**

使用的是在Mininet构建的三控制器、24主机线性拓扑的仿真数据，注入高速SYN/ACK/UDP洪水攻击，生成10次独立实验的流量窗口和真实标签；

**📈 对比分析**

对比方法为传统单控制器熵检测（Baseline）与跨域融合检测（Joint），实验显示聚合控制器的误报率从8.87%降至1.96%（降低77.9%），F1提升7.85个百分点，精确率提升14.4个百分点，召回率仅略降0.69个百分点，整体性能显著提升；

**⚠️ 局限性**

局限性包括仅在链式三控制器拓扑、单一高速洪水攻击类型下验证；未激活置信抑制与阈值恢复分支，缺少低速或多路径攻击场景；假设控制器可信，未考虑恶意报告和网络延迟异常；

---

## 115. Thinking in a Low-Resource Language: What SFT Builds, What RL Fixes, What Accuracy Cannot See

**arXiv ID:** 2608.17744 | [PDF](https://arxiv.org/pdf/2608.17744v1)

**作者:** Ayoub Kirouane `[一作]`, Christos Petrocheilos `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对低资源语言希腊语，对三种稀疏Mixture‑of‑Experts模型（Qwen、Gpt‑OSS、Nemotron）分别进行LoRA微调（SFT）以及后续可验证奖励强化学习（RLVR），并通过多维度评测工具对推理语言、推理预算、终止、步骤、准确率和预算超支等六个行为维度进行量化比较，最终发布对应的检查仪表和模型检查点。

**💡 创新点**

创新点包括：①提出并统一六个可量化的行为维度，揭示准确率表面下的真实改进与缺陷；②证明在7.7pp的种子噪声基线下准确率几乎无意义；③首次使用可验证奖励的RL（RLVR）在不损失准确率的前提下，显著修复回答格式错误、推理通道泄漏和语言锁定等缺陷；④提供一套可复现的控制实验和审计机制，防止工具误导。

**🔧 技术方法**

技术手段主要为：LoRA低秩微调（r=32、α=64）；稀疏MoE架构的多实验；基于语料匹配的推理‑直接双模式训练；可验证奖励强化学习（GRPO）实现对格式、语言一致性、终止、覆盖率等指标的精准控制；以及多维度评测工具与控制实验的系统化设计。

**📊 数据集**

使用的数据集为：118,092条希腊语数据，分为约59k行含“链式思维”推理轨迹和约59k行无推理对话；评测集包含5,156条希腊语问题（MGSM、HellaSwag、Winogrande、ProofWriter等），以及与之对应的英语对照集；此外还利用公开英语数据生成的翻译题目构成的基准。

**📈 对比分析**

方法上对每个模型配置跑了15个独立训练实例，并使用多条基线（种子控制、英文对照、直接/推理模式）进行对比。结果显示准确率差异被7.7pp的种子噪声覆盖，但推理语言一致性、推理预算、终止率等维度均显著提升。语言匹配微调后希腊推理准确率保持73–77%，推理语言几乎100%；RLVR将答案格式错误从24%降至2.5%，推理通道泄漏从3.5%降至0%；但不同家族中Token成本下降效果不一，且在某些模型中推理语言锁定仍未完全解除。

**⚠️ 局限性**

局限性包括：单次准确率受到7.7pp种子噪声基线影响，难以作为可靠指标；实验仅覆盖三种MoE模型与LoRA，缺乏对稠密模型或其他低资源语言的泛化验证；RLVR需要手工设计奖励与攻击集，操作复杂；评测工具对语言特性（如数字符号、句法）仍存在缺陷；推理语言锁定问题在不同家族间表现不一致，且对LoRA规模、路由比例等超参数敏感。

---

## 116. Quantum-Safe Web Service Architecture Using Time-Based One-Time Passwords

**arXiv ID:** 2608.16961 | [PDF](https://arxiv.org/pdf/2608.16961v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 117. A Simple Algebraic Proof of the PCP Theorem

**arXiv ID:** 2608.17429 | [PDF](https://arxiv.org/pdf/2608.17429v1)

**作者:** Prashanth Amireddy `[一作]` (Harvard University), Sophus Valentin Willumsgaard `[通讯]` (University of Copenhagen)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种新的常数查询多项式时间PCP证明构造，直接验证3着色问题，避免了传统的PCP组合与迭代技术，完成了PCP定理的证明。

**💡 创新点**

创新点在于引入集合变量技术（set‑of‑variables）与集合多项式（set‑multilinear）构造新的低次数多项式，并通过二元Hadamard编码与Ψ映射实现常数查询的低阶多项式测试，彻底绕过了经典的PCP组合与迭代步骤。

**🔧 技术方法**

主要使用了低阶多项式距离引理、线性/同质线性多项式的加法性质、低次数扩展（lines table）与Ψ映射、1‑Hadamard编码和多项式距离引理相结合的常数查询低阶多项式测试与零点检验（zero‑on‑grid test），以及局部纠错（local correction）技术。

**📊 数据集**

该工作为纯理论证明，未使用任何数据集或实验数据。

**📈 对比分析**

与现有PCP构造相比，该方法在保证证明长度为线性、随机位数为O(log n)且查询次数为常数的前提下，满足PCP定理；没有实测性能指标，理论复杂度与传统PCP相当。

**⚠️ 局限性**

局限性包括：证明结构极其冗长、技术门槛高，所需的常数（如c、c′、c₁、c₂）较大且不易明确；实现时对有限域和根的可用性有严格要求；该方法尚未证明在更广泛的实际算法或随机化模型中的可行性。

---

## 118. Token Optimization and Context Window Management in Multi-Agent AI Workflows

**arXiv ID:** 2608.17188 | [PDF](https://arxiv.org/pdf/2608.17188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 119. Data-DPO: Direct Preference Optimization for Target Model Data Selection in LLM Post-Training

**arXiv ID:** 2608.16926 | [PDF](https://arxiv.org/pdf/2608.16926v1)

**作者:** Peng Sun `[一作]` (Nanjing University), Tianfan Fu `[通讯]` (Nanjing University)

**通讯引用:** 3764 | [OpenAlex ID](https://openalex.org/A5003226543)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于目标模型反馈的一步探测数据选择方法Data-DPO，利用对目标模型进行一次微调后激活差异构造样本优先级，并训练轻量级奖励模型最终生成更优训练子集

**💡 创新点**

创新点在于将数据价值视为动态、与目标模型的能力分布相匹配；通过一次性探测得到局部激活差异生成对比偏好，进而学习模型感知的奖励函数，并在最终子集构造时融合偏好、外部质量分数和余弦距离多样性

**🔧 技术方法**

技术包括：双视图编码、k-center探测集构造、一次性微调激活收益计算、对比偏好构造、DPO式奖励模型训练、顺序贪心选择与多项式奖励组合

**📊 数据集**

在Vision-Flan（通用指令调优）和LLaVA-CoT（推理调优）两个公开数据集上进行实验

**📈 对比分析**

与随机、重要性估计、多样性与混合选择等基线进行比较，在5%、10%、15%数据预算下均显著优于所有基线，并在多数设置下超越全数据训练的性能；例如Vision-Flan 5%预算时相对性能达到100.76%

**⚠️ 局限性**

局限性包括：对外部质量评分、嵌入来源和目标模型偏好可能产生偏差；在数据分布剧烈变化时效果有限；虽然不需代理模型，但探测集上仍需额外计算，存在一定效率成本

---

## 120. Probing Association Instability with Track-State Perturbations for Clip-Level Active Learning in Query-Propagation Multi-Object Tracking

**arXiv ID:** 2608.17224 | [PDF](https://arxiv.org/pdf/2608.17224v1)

**作者:** Riku Inoue `[一作]` (NTT, Inc.), Ryuichi Tanida `[通讯]` (NTT, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对端到端查询传播式多目标跟踪的剪辑级主动学习，提出一种基于内部跟踪状态两侧扰动的关联不稳定性评估与代表性剪辑选择方法。

**💡 创新点**

创新点在于：① 用两侧扰动量化内部轨迹状态的关联不稳定性，捕获最终输出平滑但关联易变的场景；② 结合定位漂移与熵加权置信度差异的两种指标，并以乘法聚合得到剪辑级不稳定性得分；③ 在高不稳定性候选集中使用不确定性加权视觉覆盖（基于轨迹级视觉原型的Chamfer距离）实现多样性与代表性的统一。

**🔧 技术方法**

技术包括：两侧扰动策略（对查询嵌入和参考点做相反方向微小扰动）、关联不稳定性指标（Localization Drift、Entropy‑Weighted Confidence Discrepancy）、标准化与聚合方法、基于视觉原型的距离度量、贪心不确定性加权覆盖算法。

**📊 数据集**

使用DanceTrack和SportsMOT两大挑战性视频跟踪数据集，评估了MeMOTR和SambaMOTR两种端到端查询传播式跟踪器。

**📈 对比分析**

与随机、Entropy、Core‑set、BADGE以及最相关的CUTAL基线比较，结果显示在相同标注预算下QPID在HOTA、AssA、IDF1等指标上整体优于或相当于最强基线，并在50%预算时差距仅约0.7‑1.2 HOTA，逼近全监督性能。

**⚠️ 局限性**

局限性包括：① 对于早期小预算时，因跟踪状态不稳定性估计不可靠导致表现不如Entropy或CUTAL；② 计算耗时约为CUTAL的1.18‑1.19倍；③ 对长剪辑（如SambaMOTR T=10）覆盖率不足，可能影响学习初期的多样性。

---

## 121. Explicit State Elicitation Is Not Enough: A Controlled Audit of Memory-Policy Classification

**arXiv ID:** 2608.17247 | [PDF](https://arxiv.org/pdf/2608.17247v1)

**作者:** Yihang Chen `[一作]` (Georgia Institute of Technology), Yiqi Sun `[通讯]` (Georgia Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构造 MemoryPolicy-Bench 开发集（480例）与对照集（160例）并对长短期记忆代理在检索后决定使用/忽略/更新/询问的过程进行系统评估，采用五阶段审计框架来分离数据集捷径、提示变化、答案相关标签、语义证据及执行错误。

**💡 创新点**

提出可复用的五阶段审计协议，利用规则生成的四路对照集揭示显式状态输出并未显著提升政策准确性，证明内部状态机制不可信；同时通过家庭级一致性评估和标签条件诊断，系统评估提示与模型内部决策的关系。

**🔧 技术方法**

使用结构化提示、JSON 输出、TF‑IDF 诊断、规则生成的对照集、随机种子稳定性检验、配对统计、Bootstrap 95% 置信区间、排列检验、Holm 校正、解析失败计数以及成本估算等技术手段。

**📊 数据集**

MemoryPolicy-Bench 480 例合成开发集（由 Gemini 1.5 Flash 生成）与 160 例规则生成的对照集（四条历史记录、四种政策对照），两者均为规则标注、无真实用户数据。

**📈 对比分析**

通过匹配提示消除多重干扰、使用对照集的四路一致性评估及配对统计比较；结果显示在对照集上，显式状态输出对 Llama‑3.3‑70B 几乎无提升（+0.6pp，p=0.81），对 GPT‑OSS 仅有边际提升（+3.3pp，但不显著）；标签条件诊断表明提供基准状态标签可提升 7–12pp；语义证据输出实际导致准确率下降，整体说明仅靠提示结构无法显著提高政策准确性。

**⚠️ 局限性**

数据集为合成、规则标注，缺乏真实会话；对照集不具代表性；仅评估政策分类，未评估后续响应、工具动作或记忆变更；模型依赖性强，跨模型差异明显；仅涵盖四种政策，未涉及更丰富的记忆管理；实验在有限端点上完成，API 行为随时可能变动。

---

## 122. SeqFeed: Improving Agentic RTL Code Generation with Sequential Behavior Feedback

**arXiv ID:** 2608.16934 | [PDF](https://arxiv.org/pdf/2608.16934v1)

**作者:** Yuxin Du `[一作]` (City University of Hong Kong), Nan Guan `[通讯]` (City University of Hong Kong)

**通讯引用:** 5303 | [OpenAlex ID](https://openalex.org/A5002245169)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了SeqFeed框架，结合波形查询语言SeQuery和循环依赖图SeGraph，为LLM驱动的RTL代码生成提供事件可寻址、依赖追踪和可迭代的序列反馈。

**💡 创新点**

创新点在于：① 通过SQL式的SeQuery实现跨周期波形查询；② 以循环为单位构建SeGraph，直观捕捉跨时钟信号传播；③ 两者互补，显著提升LLM调试效率和成功率。

**🔧 技术方法**

技术包括：LLM代理（DeepSeek Flash/Pro、MiniMax M3），基于Icarus Verilog的仿真，cocotb测试框架，SQL风格查询语法，静态依赖图构建与遍历，token成本与工具调用分析。

**📊 数据集**

使用了256个RTL生成基准案例，涵盖DSP、内存与总线、机器学习、控制驱动、加速器和大IP集成等六大类别，源自CVDP、RTLLM及近期开源设计。

**📈 对比分析**

通过对比四种配置（基线、SeQuery、SeGraph、SeqFeed）评估pass率、token消耗和工具调用密度。SeqFeed在所有LLM设置下提升8.7–18.3个百分点pass率，且在token成本上更高效，工具调用更合理。

**⚠️ 局限性**

局限性在于：对多时钟域设计支持不足；对极端大规模或复杂控制流的覆盖仍有限；需进一步验证在真实工业项目中的迁移性能。

---

## 123. DynaForcing: Overcoming Dynamic Collapse in Self-Forcing Distillation for Streaming Avatar Generation

**arXiv ID:** 2608.17707 | [PDF](https://arxiv.org/pdf/2608.17707v1)

**作者:** Yubo Huang `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并解决自强（self‑forcing）蒸馏中出现的动态崩塌问题，提出 DynaForcing 框架。

**💡 创新点**

创新点包括三层次对策：Hybrid Forcing（数据级别锚定）、Dynamics‑Aware Reward Regularization（奖励级别动态正则）和 Reference Perturbation（条件级别扰动），以及计算图剪枝与梯度重放的效率提升。

**🔧 技术方法**

采用自强蒸馏、分布匹配蒸馏、RL 奖励加权、SyncNet、3DMM、ArcFace、图剪枝和梯度重放等技术。

**📊 数据集**

训练使用 AVSpeech 数据集；评测采用 GenBench‑ShortVideo（约10 s）和 GenBench‑LongVideo（>5 min）两组数据。

**📈 对比分析**

与多种非实时与实时基线（WanS2V、OmniAvatar、LiveAvatar 等）对比，DynaForcing 在短视频中 Dyn‑Deg 从 0.31 提升至 0.73，ExpVar 从 0.69→2.02，Sync‑C 从 7.03→7.68，同时保持 45.2 FPS；通过图剪枝与梯度重放，GPU‑小时从 7,111 降至 667，约 10 倍节省。

**⚠️ 局限性**

局限性包括奖励依赖预训练 SyncNet/3DMM 的偏差；实验仅在头像生成任务验证，其他任务的通用性待进一步评估；梯度重放虽显著降低显存但仍有一定时间开销。

---

## 124. Multi-turn Conversational AI from Text to Multimodal Interaction: Data, Models, Evaluation, and Open Challenges

**arXiv ID:** 2608.17605 | [PDF](https://arxiv.org/pdf/2608.17605v1)

**作者:** Syeda Faiza Ahmed `[一作]` (Qatar Computing Research Institute), Shammur Absar Chowdhury `[通讯]` (Qatar Computing Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化了多轮会话人工智能的研究进展，包括文本、语音、视觉、视频、多模态、工具增强与文化多样性等方面；

**💡 创新点**

提出了以会话层次为核心的统一框架，强调多轮记忆、跨轮对齐、全双工语音交互与跨文化适配的关键挑战，并指出当前研究的显著差距；

**🔧 技术方法**

主要聚焦于大语言模型（LLM）、AudioLLM、全模态模型、代理式对话系统、强化学习、检索增强与多任务学习等技术；

**📊 数据集**

引用并评估了约200篇论文中使用的多种数据集，涵盖文本对话、语音对话、多模态对话、视频对话以及跨语言与文化的对话语料；

**📈 对比分析**

比较方法包括单轮与会话级别的评估指标、LLM-as-judge 与人工评测、工具使用与检索准确率等，结果表明虽然单轮表现已趋近人类水平，但在会话连贯性、记忆保持和跨模态对齐方面仍远未达到预期；

**⚠️ 局限性**

局限性包括对最新文献覆盖不足、对不同架构细节的泛化处理、数据集以英语为主、评测体系碎片化以及缺乏统一的会话级性能基准。

---

## 125. Cross-Model Memory Transfer via Target-Side Reader Adaptation

**arXiv ID:** 2608.17050 | [PDF](https://arxiv.org/pdf/2608.17050v1)

**作者:** Mingyuan Li `[一作]` (ELLIS Institute Finland), Shaoxiong Ji `[通讯]` (ELLIS Institute Finland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种跨模型的冻结外部记忆转移方法：将源模型训练得到的Engram哈希记忆表冻结后，通过标准化的标记器无关地址化以及轻量级的目标端读取器，直接在目标模型中重新利用该记忆。

**💡 创新点**

创新点在于：①将可移植的哈希外部记忆与目标模型解耦，展示记忆本身可被跨架构、跨标记器使用；②证明读取器设计是决定跨模型性能的关键，且通过双层、多分支读取器几乎消除了同模和异模差距；③提供完整的评估框架和证据链，揭示记忆内容、地址完整性与读取器接口共同决定迁移效果。

**🔧 技术方法**

核心技术包括 Engram 哈希记忆架构、NFKC+小写+空格清理的 tokenizer-agnostic canonicalization、可插拔的多层多分支读取器（branch count R、注入层设定）以及两阶段训练协议（冻结记忆+仅训练读取器）。

**📊 数据集**

使用的数据集涵盖：WikiText‑103（内部语言建模评估）、Wikipedia‑2021（源记忆与目标读取器的预训练语料）、FineWeb‑Edu（下游 QA 任务），以及一系列公开 QA 基准（NQ、WebQA、TriviaQA、TruthQA、HotpotQA、RTE、SciQ、OpenBookQA、BoolQ、RACE）和 TruthfulQA 进行多任务验证。

**📈 对比分析**

在同一架构下，跨模型记忆转移的 QA 平均准确率从 32.1 提升至 38.8（接近 38.5 的最优），相比 RAG、kNN‑LM、CPT、LoRA、MLP‑Memory 等基线提升 6–20 点；在 Pythia、Qwen、LLaMA‑2 与 Mistral 等多模型矩阵中，所有 9 组组合均实现相对 PPL 降低 1.6–15.7%；使用仅 1 M 可训练参数即可在 5 M 目标侧数据上达到与从零开始训练相近的 perplexity，展示显著的数据效率优势。

**⚠️ 局限性**

局限性包括：①跨模型迁移的收益高度依赖于地址完整性与读取器匹配，若两模型标记器差异大需额外 canonicalization；②记忆内容对任务的适用性有限，在 TruthfulQA 等对事实校准敏感的任务中反而可能产生负面影响；③最终性能与记忆表的规模与质量相关，过度稀疏或过拟合的记忆在新模型中可能不利；④虽然提供了更快的适配，但在极大模型或长序列场景下的延迟与内存开销仍需进一步评估。

---

## 126. SemComp-Bench: Benchmarking Semantic Task Completion in Video Generation

**arXiv ID:** 2608.17426 | [PDF](https://arxiv.org/pdf/2608.17426v1)

**作者:** Keyu Tu `[一作]` (University of Science and Technology of China), Yongdong Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了语义任务完成视频生成（Semantic Task Completion Video Generation）任务，构建了 SemComp-Data 数据集并设计了 SemComp-Bench 评测框架，用于评估生成视频在实现指令结果、语义对齐和生成可靠性方面的表现。

**💡 创新点**

创新点在于：①将任务定义为在保持参考语义关系的前提下完成指令指定的最终状态；②通过四阶段自动化抽取流程从全上下文视频中生成图像-文本-视频三元组，实现大规模真实场景数据集；③采用 VLM 进行结构化二元评判，分离结果实现与生成可靠性两维评价，提供可解释的错误诊断。

**🔧 技术方法**

技术主要包括：VLM（如 Doubao-Seed-1.8）用于状态定位、质量检查、指令生成和评估；图像-文本-视频匹配框架；基于关键词过滤和视频抽象的多分类器；以及基于帧抽取、剪辑和模板化指令生成的流程。

**📊 数据集**

使用了从 Koala-36M 公开视频中抽取的 1,273 条实例，经过四阶段处理后得到 SemComp-Data；核心子集 SemComp-Core 包含 60 条多领域平衡实例；训练与评测还利用了多种公开视频生成模型（Seedance 2.0、Wan2.2、CogVideoX、HunyuanVideo、SkyReels 等）。

**📈 对比分析**

评估方法采用 SemComp-Bench 中的结构化二元问题，分别计算 Outcome Achievement（OA）和 Generation Reliability（GR）两个维度的分数。实验结果显示：最高 OA Score 仅 37.8%（HunyuanVideo），最高 GR Score 91.8%（Seedance 2.0）。I2V 条件下整体表现优于 T2V，且详细指令往往比简短指令获得更高的 OA，但对生成难度要求更大。

**⚠️ 局限性**

局限性包括：①当前模型在实现指令结果与保持语义对齐方面表现不足，OA 低于 40%；②对空间时间一致性的维持仍是主要瓶颈，尤其是场景切换与局部稳定性；③评估依赖 VLM，可能对模型本身的视觉理解偏好产生影响；④数据集规模相对有限，尚未验证通过任务专属训练能否显著提升性能。

---

## 127. Scanline-Aware Animatable Gaussian Avatars from Rolling-Shutter Videos

**arXiv ID:** 2608.17314 | [PDF](https://arxiv.org/pdf/2608.17314v1)

**作者:** Youxiang Wang `[一作]` `[通讯]` (University of Macau), Youxiang Wang (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于滚动快门（RS）视频的3D高斯头像重建方法，直接在图像形成模型中嵌入滚动快门机制，生成无失真、可动画的高质量人类头像。

**💡 创新点**

创新点在于：将已有的运动感知模糊头像框架仅通过更换组合算子（由均匀平均改为逐扫描线拼接）即可适用于滚动快门；并构建了专门的RS头像基准，验证该改进显著提升视角合成质量。

**🔧 技术方法**

采用的技术包括：3D Gaussian splatting、SMPL 运动模型、连续时间B样条姿态插值、线性混合皮肤化、扫描线掩模组合算子、联合优化和基于RIFE的帧级插值。

**📊 数据集**

使用的数据集为 ZJU‑MoCap 的六个受试者序列（1024×1024 分辨率，556–859 帧），并通过扫描线拼接与时间插值生成合成的滚动快门基准。

**📈 对比分析**

方法与基线（滚动快门忽略、两阶段2D校正、模糊头像）进行对比，结果显示在PSNR、SSIM和LPIPS指标上均优于所有基线（如PSNR 24.68 对 GauHuman 23.69），并在消除模糊模型后性能更佳。

**⚠️ 局限性**

局限性包括：姿态插值为低阶多项式（无法捕捉更高阶运动）、仅适用于已曝光的sRGB图像、无法处理非关节运动（如手持物体或松散衣物）、基准为合成RS（真实传感器误差未考虑）、以及缺乏低照度和多人物情境的鲁棒性。

---

## 128. PROBE: Manipulation-Grounded Visual Question Answering with VLM Agents

**arXiv ID:** 2608.17129 | [PDF](https://arxiv.org/pdf/2608.17129v1)

**作者:** Vineet Bhat `[一作]`, Jonathan Tremblay `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个名为‑Sim 的仿真环境，用来评估视觉语言模型（VLM）在含有工具调用的抓取与推送任务中的表现，并构建了 PROBE‑Bench 评测基准，进一步在 Kinova Jaco 实际机器人上验证了从仿真到真实机器人的迁移效果。

**💡 创新点**

创新点包括：① 将场景分解为可调用的感知与操作工具，允许 VLM 在每一步做决策而不是一次性输出；② 在 50×50 cm 的区域内随机生成 15 个真实扫描物体，保证目标经常被遮挡，提升对操作决策的需求；③ 通过大规模多样化资产池实现训练/测试的分离，从而真正测量组合泛化而非记忆；④ 在同一工具集下对抓取与推送两种操作进行单独的成功率基准，揭示不同物理复杂度的影响；⑤ 在评测中加入 HDRI 灯光变化与真实摄像头视角，对照实验验证了仿真对真实世界性能的保真度。

**🔧 技术方法**

技术手段包括：
- Gemini 3.1 Pro 作为通用指向模型生成 2D 位置；
- SAM3 进行目标分割；
- GraspGen 预测 6‑DoF 抓取；
- 简单的线性推送动作；
- 通过工具箱实现感知（场景图、分割）与操作（抓取、推送、终止）的调用；
- 采用宏平均统计、基于种子确定的可复现评测流程；
- 在真实机器人上使用 Kinova Jaco 与前视摄像头，结合真实抓取/推送规划器。

**📊 数据集**

使用的数据集包括：
- 15 真实扫描物体（共 1,796 件多样化）用于资产池；
- 120 条问答模板，覆盖六类任务（位置、属性、相对、遮挡等）；
- 150 条评测任务（各类问答组合），经过人工验证可解；
- Poly Haven HDRI 图集用于灯光实验；
- 25 家庭常见物体用于 Kinova Jaco 实验。

**📈 对比分析**

比较方法：将‑Sim 与现有 VLM‑driven 及视觉强化学习仿真环境（如 VLM‑Sim、其他物体抓取仿真）对比，强调仅 ‑Sim 提供可调用工具且场景高度混乱；在 PROBE‑Bench 上对 8 种 VLM（Gemini 3.1 Pro、GPT‑5.4、Opus 4.7、Gemma‑4、Qwen3.5‑VL、Grok 4.3、Nemotron 2‑VL、Gemini Robotics ER）进行宏平均测试，表现如下（95% CI）：Gemini 3.1 Pro 69.2%±3.8，Opus 4.7 63.1%±4.1，Qwen3.5‑VL 61.0%±4.1，Gemini Robotics ER 61.6%±4.4，Gemma‑4 62.4%±4.0，GPT‑5.4 59.0%±4.4，Grok 4.3 52.2%±4.2，Nemotron 2‑VL 39.0%±3.8。真实机器人实验显示 ‑Sim 与实机表现高度相关（Pearson r = 0.82，Spearman ρ = 1.0），且排名保持一致。

**⚠️ 局限性**

局限性：
- 仅在 15 物体的 50×50 cm 区域内评测，规模有限；
- 训练与测试使用同一类物体，但在真实机器人中仅限 25 件家庭物体，未覆盖所有物理属性；
- 相机视角从仿真顶视切换到前视，虽证明可迁移，但仍可能对视觉特征产生偏差；
- HDRI 灯光实验尚未在所有模型上完成，关于光照对感知与操作的影响仍待进一步探究；
- 工具箱采用固定的推送/抓取策略，未覆盖更复杂的操作（旋转、堆叠等）。

---

## 129. No Gaussian Required: Contrastive Inverse Dynamics for JEPA World Models

**arXiv ID:** 2608.17542 | [PDF](https://arxiv.org/pdf/2608.17542v1)

**作者:** Jack Boylan `[一作]` (Quantexa), Chris Hokamp `[通讯]` (Quantexa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于对比逆动力学（Action‑Contrastive NCE）的JEPA世界模型，用训练时仅加入逆向头部实现无衰退的表示学习，并在测试时保持原始前向预测与规划模块。

**💡 创新点**

创新点在于将抗崩塌机制从全局分布约束迁移到局部的、仅依赖转移数据的对比逆动力学信号，既消除了对预训练网络、停止梯度或全局高斯匹配的需求，又保持了单阶段端到端训练的简洁性。

**🔧 技术方法**

采用了像素编码器、投影层、因果前向预测器和InfoNCE对比损失，训练时仅添加一个小型逆向MLP头部，测试时去除该头部并使用相同的CEM规划器。

**📊 数据集**

在标准的四个像素控制任务（TwoRoom、Reacher、PushT、OGBench‑Cube）以及更具挑战性的多物体OGBench Visual Scene任务上进行评估，使用公开的离线数据集。

**📈 对比分析**

与原始LeWM、PLDM、DINO‑WM等基线相比，Action‑Contrastive JEPAs在所有任务上都与LeWM持平或略优，尤其在OGBench Visual Scene上提升至约80%成功率，显著高于LeWM（约58%）和其他公开基线；在Reacher任务上显著提升稳定性，减少崩塌现象。

**⚠️ 局限性**

局限性包括：对连续控制的依赖，需足够多样的动作候选和视觉反馈，否则对比信号弱；对弱控制变量（如PushT中方块方向）不够敏感；对批次构成、温度τ和系数λ敏感，需要在更广泛的环境中进一步验证；且无法提供无条件的几何保证，可能在高维或离散动作空间下失效。

---

## 130. Proactive Road Safety Intervention in Australia: Predicting Risky Driving Hotspots from Connected Vehicle Data

**arXiv ID:** 2608.16913 | [PDF](https://arxiv.org/pdf/2608.16913v1)

**作者:** Adriana-Simona Mihăiţă `[一作]` (University of Technology Sydney), David Lillo-Trynes `[通讯]` (COMPASS IOT PTY LTD)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用连接车辆遥测数据，识别并预测新南威尔士州各地方政府区（LGA）每日的近失事件量，生成热图并提供预测模型。

**💡 创新点**

首次在澳洲以LGA级别应用IoT遥测进行近失事件预测，提供基于g‑force阈值的实时风险识别框架，并系统评估多种模型以指导精准干预。

**🔧 技术方法**

使用g‑force阈值筛选近失事件，构建特征窗口，训练8种模型：随机森林、XGBoost、LightGBM、ARIMA、指数平滑、Prophet、LSTM、N‑BEATS，并采用滚动原点评估。

**📊 数据集**

来自Compass IoT的超过700,000辆车的遥测数据（2020‑2021年），按日汇总为每个LGA的近失事件计数。

**📈 对比分析**

采用滚动窗口（35天）评估MAE/RMSE/MAPE，ARIMA（MAE 162.21）和LSTM（163.92）表现最佳，传统时间序列模型优于集成学习和Prophet，深度模型未超越ARIMA。

**⚠️ 局限性**

数据量有限导致深度模型表现不佳，COVID‑19封锁期间交通量下降影响训练，模型仅考虑单变量时间序列，缺乏空间曝光归一化及多源特征，评估窗口短且未验证多步预测。

---

## 131. Population Health-Based Machine Learning Reveals Associations Between Psychosocial Factors and Chronic Kidney Disease

**arXiv ID:** 2608.17174 | [PDF](https://arxiv.org/pdf/2608.17174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 132. REST API Testing with Verified LLM-Inferred Dependencies and Response-Driven Refinement

**arXiv ID:** 2608.17546 | [PDF](https://arxiv.org/pdf/2608.17546v1)

**作者:** Tu Nguyen `[一作]` (University of Science, Vietnam National University), Vu Nguyen `[通讯]` (University of Science, Vietnam National University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 APIPilot，一种基于执行验证的 REST API 测试框架，先从 OpenAPI 规范与 LLM 推理中生成候选依赖关系，再通过真实 API 调用验证并构造可执行的测试工作流，随后利用运行时响应进行资源池更新和依赖细化。

**💡 创新点**

创新点在于：① 将 LLM 推断的依赖视为假设并通过实际执行进行验证；② 采用受限的 top‑k 图遍历生成覆盖率友好的工作流；③ 通过响应驱动的迭代细化资源池与依赖映射，将 LLM 仅用于缺失语义提示，而非直接决定测试逻辑。

**🔧 技术方法**

使用技术包括：OpenAPI 规范解析、结构化启发式与 LLM（GPT‑4）语义推断、执行验证与资源池管理、受限 top‑k 图遍历、语义与结构约束下的输入生成、运行时响应分析与细化反馈。

**📊 数据集**

使用了 16 个来自不同行业的真实 REST API（如 GitLab、其他公开接口），覆盖多种资源层级和业务流程。

**📈 对比分析**

与 AutoRestTest、RESTifAI、KAT（LLM 基础）和 DeepREST（非 LLM）进行公平比较；实验结果显示 APIPilot 的操作覆盖率 92.3%（DeepREST 60.4%），代码行覆盖率 58.6%，工作流执行成功率 88.1%，且每个测试用例平均使用 155 个 token，表现明显优于基线。

**⚠️ 局限性**

局限性包括：依赖完整且准确的 OpenAPI 规范，难以处理文档不完整或不一致的接口；仅针对 RESTful 服务，可能不适用于事件驱动或非 REST API；LLM 推断的偏差仍可能影响候选依赖的初始质量；实验仅覆盖 16 个 API，尚未验证在更大规模或不同类型接口上的泛化能力。

---

## 133. RetiWave-Mamba: A Dual-Stream Network for Retinal Disease Detection based on Multi-scale Context and Frequency-Adaptive Mamba Projection

**arXiv ID:** 2608.17623 | [PDF](https://arxiv.org/pdf/2608.17623v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 134. Too cheap to matter: over abundant microchips, and what we can learn from them

**arXiv ID:** 2608.17541 | [PDF](https://arxiv.org/pdf/2608.17541v1)

**作者:** Adrian Friday `[一作]` (Lancaster University), Srinjoy Mitra `[通讯]` (University of Edinburgh)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过ICT4S工作坊对超低成本微芯片的生产、使用与废弃进行了系统梳理与案例拆解，评估其经济与环境成本，并提出“真正成本会计”与“快科技”概念以促进可持续设计与回收；

**💡 创新点**

创新点在于将低成本芯片视为生态系统中的隐形“快科技”材料，揭示其对全球电子废弃物的巨大贡献，并倡议行业采用全生命周期成本会计与生产足量的可持续策略；

**🔧 技术方法**

主要技术手段包括工业数据分析与可视化、现场拆解与反向工程、元件标识与数据表搜索，以及对比分析方法；

**📊 数据集**

使用的数据集包括：IC节点产量与收入估算、全球芯片制造厂分布与工艺节点路线图、UN《全球电子废弃物监测报告》、以及公开的节点级技术文档与元件数据表；

**📈 对比分析**

对比方法：将不同节点的产量、单元成本与生命周期环境影响（如硅用量、能源消耗）进行关联分析，展示低成本节点在消费电子中的占比与废弃物贡献；由于缺乏统一的实验平台，性能评估以案例拆解与文献量化为主，未给出数值指标；

**⚠️ 局限性**

局限性包括：行业缺乏节点级产量公开数据导致估算误差、拆解案例有限无法覆盖所有产品类型、对元件数据表的依赖导致部分低成本芯片信息不完整，以及研究侧重宏观评估而非微观技术改进。

---

## 135. CoAL-RAG: A Complexity-Aware Legal Retrieval-Augmented Generation Method

**arXiv ID:** 2608.17536 | [PDF](https://arxiv.org/pdf/2608.17536v1)

**作者:** Jin Su `[一作]` (North China University of Technology), Hao Chen `[通讯]` (North China University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 CoAL-RAG，一种针对法律问答多级复杂度的复杂度感知检索增强生成框架。

**💡 创新点**

引入多维度复杂度评估机制和检索一致性判定，实现自适应检索路由与动态上下文构造。

**🔧 技术方法**

结合 LangGraph、层级法律知识图谱、BM25 与向量检索、竞争门控算法及多维度逻辑分量评分。

**📊 数据集**

使用中文民法基准 SocialLawQA、LawBench，以及英文普通法基准 LexGLUE、CaseHold。

**📈 对比分析**

与直接推理、RAG、Hybrid RAG、知识图谱检索、强化学习调优等多种基线对比，中文基准 BLEU 提升 42.5%，ROUGE‑L 为 KG 方法的 3.6 倍；英文基准与 Search‑R1 的准确率相当，且保持较低的平均响应时间。

**⚠️ 局限性**

受限于知识图谱覆盖范围，缺乏地方政策与先例信息，对层级冲突处理仍不完善。

---

## 136. Ready for What? Rethinking AI and Robotics Preparedness for Adoption and Policy

**arXiv ID:** 2608.17520 | [PDF](https://arxiv.org/pdf/2608.17520v1)

**作者:** Peng Wang `[一作]` (University of Surrey), Teslim Olayiwola Salahudeen `[通讯]` (University of Lancashire)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文利用重复卡片式调查，对982名受访者评估了17项AI与机器人挑战的显著性、复杂性与准备度，并通过个体-挑战分解揭示挑战特定的准备障碍。

**💡 创新点**

创新点在于将个体层面与挑战层面分离，揭示挑战被视为异常复杂时准备度显著下降，且与专业背景、信任与自信相关。

**🔧 技术方法**

采用线性混合模型、固定效应模型以及有序logit/GEE对卡片评分进行统计分析，并通过自变量分解提取within-between效应。

**📊 数据集**

使用来自Prolific平台的982名参与者的15,200条卡片评价数据，其中包含17个AI/机器人挑战卡片。

**📈 对比分析**

比较方法包括主线性混合模型与固定效应、比例优势logit及GEE等模型，结果显示within-person复杂度负向影响准备度，且模型间方向一致，说明稳健。

**⚠️ 局限性**

主要局限在于自评跨章数据、样本非概率、卡片呈现顺序固定、Ethical Awareness仅一张卡、部分人口统计缺失，且缺乏因果推断。

---

## 137. Vision-Language Models for Analog Gauge Reading: An Empirical Study of Specialization, Transfer and Reliability

**arXiv ID:** 2608.17723 | [PDF](https://arxiv.org/pdf/2608.17723v1)

**作者:** Abdul Mueez `[一作]` (University of Central Florida), Shruti Vyas `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究了一种直接对单目标模拟指示器图片进行数值读取的视觉语言模型方法，并在三种数据集上评估其性能。

**💡 创新点**

创新点在于将Qwen2.5-VL-7B进行QLoRA细化，并系统比较了零射、ICL与Fine‑tune三种方式，揭示了范围元数据对极端误差的抑制作用。

**🔧 技术方法**

使用技术包括开源的Qwen2.5-VL-7B-Instruct模型、Quantized Low‑Rank Adaptation (QLoRA)、零示例提示、上下文学习、梯度累积、图像增强、校准与置信度分析。

**📊 数据集**

实验数据集包括公开的Synthetic Gauge dataset、基于视频的Pressure Gauge dataset，以及Siemens Energy的工业实景图片。

**📈 对比分析**

通过与零射、ICL基线和不同域的Leave‑One‑Dataset‑Out (LODO) 进行对比，Fine‑tuned模型在Synthetic、Pressure和SE上的MPE分别为2.39%、2.61%和4.43%，显著优于基线；但在跨域和极端模糊条件下性能下降。

**⚠️ 局限性**

局限性包括样本量有限、随机拆分未实现全域或厂区级隔离、仅单目标指示器、范围元数据对精度的依赖、跨域泛化差和高置信误差仍可能出现。

---

## 138. Temporal Leakage in Financial News NLP: A Multi-Architecture Audit with a Regime-Specific M&A Signal

**arXiv ID:** 2608.17223 | [PDF](https://arxiv.org/pdf/2608.17223v1)

**作者:** Chenhao Xue `[一作]` (Predictive Labs Ltd), Julian Kaljuvee `[通讯]` (Predictive Labs Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在49,799篇2020–2025年的金融新闻上，系统评估了随机拆分与时间序列拆分对多种特征-模型组合（TF‑IDF、MiniLM、FinBERT、DeBERTa‑v3‑large、RoBERTa‑large、零/少量样本LLM等）的方向预测性能差异，并审计了时间泄露对模型表现的影响。

**💡 创新点**

创新点在于提出一套完整的时间泄露审计框架，量化不同模型容量与特征丰富度对泄露比例的影响，并首次发现仅在M&A事件上，在严格时间验证下仍能恢复显著预测信号；同时结合角色标注与外部语料的交叉检验，提供多角度的验证。

**🔧 技术方法**

技术方法包括：TF‑IDF与深度预训练嵌入（MiniLM、FinBERT、DeBERTa‑v3‑large、RoBERTa‑large）作为特征；逻辑回归、随机森林、GBDT、Bi‑LSTM、端到端微调Transformer；零/少量样本LLM（Llama‑3、Qwen2.5） 的零/少样本推理；评价指标采用 Matthews 相关系数（MCC），并辅以 10,000 次置换检验、每周块自举 CI 与 Benjamini–Hochberg 多重检验修正。

**📊 数据集**

使用的数据集为 49,799 篇金融新闻（2020–2025，81% 来自 2025）并带有 203 种事件标签，另对 EDT（2020–2021）与 FNSPID（2009–2020 U.S.）等公开语料进行跨语料检验；所有划分均采用严格的时间序列分割（训练 ≤ 2025‑04‑01，验证 2025‑04–2025‑05，测试 ≥ 2025‑06‑01）。

**📈 对比分析**

对比方法是将同一特征-模型组合在随机拆分和时间拆分下分别训练并评估；结果显示随机拆分下 MCC 提升 1.1×–6.5×，而时间拆分下大多数模型几近随机（最高 0.060）；针对 M&A 事件的专用 TF‑IDF 逻辑回归在时间拆分下实现 MCC = 0.138，置换检验 p < 10⁻³，区块自举 CI 为 [+0.066, +0.205]，表明存在可识别但弱的预测信号。

**⚠️ 局限性**

局限性包括：数据为专有且 81% 来自 2025，缺乏跨制度/跨周期的外推；时间窗口近似（仅 3 个月），无法评估长期稳定性；统计功效有限，尤其是角色标注（n≈125）导致 p‑值不显著；LLM 结果对种子与提示高度敏感；模型与数据的复现性受限。

---

## 139. Language Models Reproduce Human Reductionist Bias and Decision Inconsistency in Neurodevelopmental Disorders Assessment

**arXiv ID:** 2608.17105 | [PDF](https://arxiv.org/pdf/2608.17105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 140. The use of data from information systems in court proceedings

**arXiv ID:** 2608.16901 | [PDF](https://arxiv.org/pdf/2608.16901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 141. SPVC: Structured and Panoptic Video Fixing for Cross-Dataset Driving Scene Rendering

**arXiv ID:** 2608.17420 | [PDF](https://arxiv.org/pdf/2608.17420v1)

**作者:** Gen Li `[一作]` (Tsinghua University), Chaojian Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种结构化、全景、视频、跨数据集驱动的修复框架 SPVC，利用可控视频扩散模型对驾驶场景渲染中的背景和前景瑕疵进行统一修复。

**💡 创新点**

四大创新点：①基于相机姿态、3D框盒、HD地图等显式空间条件进行结构化修复；②同时处理背景与前景（全景）瑕疵；③在视频序列上使用时间一致性；④采用单一模型跨 Waymo、nuScenes、PandaSet 等多数据集训练，提升通用性；两阶段可控扩散策略先恢复视频外观后再细化结构。

**🔧 技术方法**

核心技术包括可控视频扩散（Wan‑2.2 DiT）、参考视频注意力、相机姿态编码、3D Bounding Box 与 HD Map 条件注入、两阶段训练策略、以及针对性的数据增强（低训练 3DGS、单相机训练、随机遮挡等）。

**📊 数据集**

使用的公开数据集：Waymo、nuScenes、PandaSet；并在未见的 EUVS 场景上进行零样本迁移测试。

**📈 对比分析**

与 ReconDreamer、ReconDreamer++、Difix3D+、FreeVS/FreeSim 等 SOTA 方法对比，在 Waymo、nuScenes、PandaSet 上在 FID、FVD、视觉一致性等指标均优于对手；在零样本 EUVS 场景中亦保持较强修复性能；闭环安全场景下使用 SPVC 修复数据训练的 VAD 模型，碰撞率下降、NeuroNCAP 分数提升。

**⚠️ 局限性**

局限性：依赖 3D Gaussian Splatting 重建质量，极端光照/天气下表现未知；推理仍较慢（约 250 秒/25 帧）；跨域迁移仍需进一步验证；主要关注车辆与道路结构，未覆盖非交通主体或复杂非车道环境。

---

## 142. PTXBench: Benchmark and Adapt LLMs for GPU Kernel Optimization with Architecture-specific PTX

**arXiv ID:** 2608.17379 | [PDF](https://arxiv.org/pdf/2608.17379v1)

**作者:** Genghan Zhang `[一作]` (Stanford University), Kunle Olukotun `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PTXBench benchmark 和 Fixit 适配框架，用多轮 LLM 生成并验证具备架构特定 PTX 指令的 GPU 内核；同时通过 Repair‑Conditioned SFT 评估模型对 PTX 编程能力的提升。

**💡 创新点**

创新点在于：① 引入可审计的目标指令执行检查，将功能正确性与低层 PTX 指令使用分离；② 通过多轮 agent loop 结合执行反馈实现迭代修正；③ 设计 Fixit 修复‑教学机制，将失败、修复与推理结合用于 SFT，提供针对性的数据增强；④ 构建可扩展的 Benchmark 体系，支持多 GPU 体系结构与工作负载。

**🔧 技术方法**

技术包括：CUDA‑PTX 代码生成、架构知识包（参数、模板、契约）、Nsight Compute 动态指令检查、CUPTI 性能计时、LoRA 微调、Fixit 修复教师与推理教师、Fast^Inst._p 指标与多轮评价协议。

**📊 数据集**

数据集主要是 FlashInfer‑Trace 生成的 GEMM 与注意力工作负载（多头注意力前向/后向、因果/非因果、不同 head 大小），以及用于 Fixit 的七种训练 recipe（包含问题覆盖、平衡与推理教师）。

**📈 对比分析**

通过与前沿库（cuBLAS、cuDNN、FlashInfer）比较，评估功能正确率、目标指令执行率与速度提升；结果显示：模型在前向任务能部分执行 PTX 指令，但在后向注意力上性能低于库；Fixit 在部分任务提升了正确率和速度，但提升不均匀，整体速度仍未达到库级别。

**⚠️ 局限性**

局限性：① 仅在单一 27B LoRA 数据集上进行微调，规模有限；② 只覆盖 BF16 GEMM 与注意力，未测试更广泛 GPU 运算；③ 结果可能因模型规模、教师质量与数据平衡差异而不具普适性。

---

## 143. NeuroAbs: A Neuro-Symbolic RTL Abstraction Framework for Property Checking Acceleration

**arXiv ID:** 2608.17304 | [PDF](https://arxiv.org/pdf/2608.17304v1)

**作者:** Zhiyuan Yan `[一作]`, Hongce Zhang `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个神经符号框架 NeuroAbs，用 LLM 自动识别并重写 RTL 代码中的抽象区域，然后通过 SMT 检查和 CEGAR 机制保证抽象的安全性，从而加速硬件属性验证。

**💡 创新点**

首次将大型语言模型直接用于 RTL 抽象；将 LLM 生成的重写结果与 AST 结构化表示相结合，并通过 SMT 验证与 CEGAR 迭代，实现在保持安全性前提下的高度自适应抽象。

**🔧 技术方法**

使用 GPT‑4o‑mini 进行信号识别与代码重写，基于 Pyverilog 构建 AST，利用 Z3 进行 SMT 检查，整合 RIC3 与 AVR 两个工业级模型检查器，并采用 CEGAR 进行抽象精炼。

**📊 数据集**

在 RISC‑V 处理器（PicoRV32、Piccolo、Flute）和 I2C 外设的 55 个验证场景上进行评测，使用 RISCV‑formal、ILA 规范以及官方属性集。

**📈 对比分析**

与未使用 NeuroAbs 的 AVR 和 RIC3 基线相比，AVR 的验证运行时间降低约 45%，BMC 在同样时间内深度提升 34–76%，并在需要深度展开的场景下显著减少了到达基线最大深度的时间。

**⚠️ 局限性**

依赖 LLM 的推理成本高、重写过程顺序执行导致总时长受限；对某些 RTL 结构支持有限；在极大规模设计或低资源环境下可能难以保持足够的性能或安全性。

---

## 144. HarnessRisk: A Lifecycle-Oriented Benchmark for Agent Harness Safety

**arXiv ID:** 2608.17597 | [PDF](https://arxiv.org/pdf/2608.17597v1)

**作者:** Yajing Bai `[一作]` (University of North Carolina), Tianlong Chen `[通讯]` (University of North Carolina)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含128个沙盒化案例、覆盖配置、扩展、运行、状态、行动和恢复六个生命周期阶段的Benchmark，用于评估Agent Harness在安全方面的表现。

**💡 创新点**

创新点在于将Agent Harness安全风险分解为六个生命周期阶段，提供统一评测框架，并在同一模型- Harness配置下系统评估不同阶段的安全性。

**🔧 技术方法**

使用大型语言模型（DeepSeek-V4-Pro、GLM-5.2、Kimi K2.6、MiniMax M3、GPT‑5.5、Claude Opus 4.7）与三种Harness（OpenClaw、Hermes、Nanobot）相结合，并通过GPT‑5.4自动评估器对轨迹进行Utility、ASR、Persistence、Detection四项指标评测。

**📊 数据集**

使用自定义的128个沙盒化案例库，每个案例包含一个良性用户任务和嵌入的对抗指令，涵盖多种工作流、工具和服务，形成了该Benchmark的数据集。

**📈 对比分析**

对14种模型–Harness配置分别执行384条轨迹，统计四项指标。结果显示Utility普遍高于90%，但Attack Success Rate（ASR）可达12.6%–80.9%，配置阶段攻击成功率最高，检测率与安全性呈负相关，表明高效完成任务并不等同于安全执行。

**⚠️ 局限性**

局限性包括：评估仅在沙盒环境中进行，缺乏真实网络交互；自动评估器在Persistence和Detection上的准确率约为85%–90%，存在误判；Benchmark的攻击类型和场景覆盖范围有限，难以全面覆盖所有现实威胁。

---

## 145. LiveHouse-TS: An Open-world Living Benchmark for Time Series Foundation Models

**arXiv ID:** 2608.17299 | [PDF](https://arxiv.org/pdf/2608.17299v1)

**作者:** Haomin Wen `[一作]` (Shanghai Innovation Institute), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出并实现了 LiveHouse-TS——一个开源的实时时间序列基准平台，支持持续预序评估、实时排行榜和可扩展的数据源注册。

**💡 创新点**

创新点在于：① 引入预序（prequential）实时评测协议，消除数据泄露；② 定义时间稳定性与改进性两种新指标，捕捉模型随时间的鲁棒性；③ 统一模型入口与评测接口，兼容多种 TSFM 与经典基线；④ 构建开放式流式数据集注册，模拟真实部署环境。

**🔧 技术方法**

技术手段包括：预先训练的 TSFM（Chronos‑2、TimesFM‑2.5、Moirai‑2.0 等）；流式数据抓取与清洗；统一预测接口与结果标准化；RMSE/MAPE/CRPS、平均排名、胜率、Elo、Temporal Stability 与 Improvement 等评价指标；Python+HuggingFace Spaces 部署的实时排行榜。

**📊 数据集**

使用 17 个公开流式时间序列数据集，涵盖 11 个领域（环境、气象、空气质量、水文、交通、金融、社会经济、事件等），频率从 1 秒到 1 年，来源包括 Open‑Meteo、NASA POWER、USGS、Binance、Wikimedia 等。

**📈 对比分析**

通过比较 6 个 TSFM（Chronos‑2、TimesFM‑2.5、Toto‑1.0、Moirai‑2.0、Chronos‑Bolt、TabPFN‑TS）与 4 个经典基线（Seasonal Naive、Moving Average、ARIMA、ETS）在实时评测中得到平均排名、胜率、Elo、Temporal Stability、Improvement。结果表明：TSFM 在总体上优于经典基线；Moirai‑2.0 在实时评测中稳居前列；Chronos‑2 与 TimesFM‑2.5 虽在静态基准中排名靠前，但在实时评测中表现下滑，体现了对分布漂移的脆弱性；总体上，实时评测揭示了模型长期性能与鲁棒性，远超传统静态排行榜的表象。

**⚠️ 局限性**

局限性：① 数据来源仅为公开流，可能包含地域、领域和测量偏差，不能代表所有真实场景；② 评测结果仅为研究证据，不能直接用于高风险决策；③ 模型加入时间不同导致评测窗口不一致，公平性需进一步改进；④ 当前指标侧重数值精度与稳定性，缺乏对业务价值或解释性的评估。

---

## 146. SCENARIODIFF: A Scenario-level Guidance Framework for Multimodal Time Series Forecasting--Extended Version

**arXiv ID:** 2608.17164 | [PDF](https://arxiv.org/pdf/2608.17164v1)

**作者:** Tuan-Binh Tran `[一作]` (VinUniversity), Tung Kieu `[通讯]` (Aalborg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种分层的上下文推理框架（ScenarioDiff），将文本信息转化为历史证据摘要、情景描述和锚点，指导多模态扩散Transformer进行时间序列预测

**💡 创新点**

创新点在于：①将文本分层为三种可解释的引导信号；②通过Anchor Blended Sampling在推理时对扩散生成的轨迹进行局部校正；③在扩散模型中采用AdaLN与跨注意力实现条件注入，避免直接让LLM生成数值

**🔧 技术方法**

采用冻结的历史上下文Agent、情景Agent、锚点Agent（均基于LLM）产生文本摘要、情景描述与锚点；将这些嵌入通过文本编码器和注意力融合至Multimodal Diffusion Transformer；Anchor Blended Sampling作为后期能量引导的扩散修正

**📊 数据集**

在Time‑MMD多域多模态基准（Economy、Energy、Security、Social Good、Traffic）上进行实验

**📈 对比分析**

与数值仅预测、LLM先验、传统多模态融合以及概率扩散模型等四组基线相比，ScenarioDiff在事件驱动域（Economy、Security）取得显著MSE/MAE提升，并在概率指标CRPS上亦表现最好；在非事件域提升不明显

**⚠️ 局限性**

依赖外部LLM预处理导致离线推理成本较高；锚点质量和文本噪声会影响效果，对高频实时预测不适用；对非事件驱动领域优势有限

---

## 147. Towards welfare-oriented recommendations in activity-travel behavior

**arXiv ID:** 2608.16922 | [PDF](https://arxiv.org/pdf/2608.16922v1)

**作者:** Ekin Ugurel `[一作]` (University of Houston), Takahiro Yabe `[通讯]` (New York University)

**通讯引用:** 2130 | [OpenAlex ID](https://openalex.org/A5075756309)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个基于福利的活动推荐框架，并使用基于MTTC的代理仿真对其进行评估。

**💡 创新点**

创新点在于引入了两种净效用阈值决策标准——正效用概率（PUP）和后悔最小化（RM），为推荐系统提供了以用户福利为中心的原则性筛选机制。

**🔧 技术方法**

技术手段包括随机效用理论、通用旅行成本模型、基于代理的仿真（Agent‑Based Modeling），以及基于反馈的学习与估计过程。

**📊 数据集**

使用的主要数据集为合成数据：基于MTTC构造的人口属性、性格特征、活动属性和城市网格；未使用真实交通或用户行为数据。

**📈 对比分析**

通过在相同的仿真环境下比较五种推荐策略（无RS、标准RS、PUP、RM、Oracle），评估平均净效用、负效用率、停止率和不平等指数；结果显示PUP和RM能将平均福利提升约30%，并显著降低负效用率和推荐误伤比例。

**⚠️ 局限性**

局限性包括：代理模型为合成且未在真实用户数据上验证；城市模型简化为网格且仅考虑自驾；假设推荐系统能准确估计旅行成本和偏好，实际情况可能受数据缺失和用户隐私限制。

---

## 148. SE-MoLoRA: Shared-Expert LoRA Adapters for Domain-Specific Photographic Assessment

**arXiv ID:** 2608.17514 | [PDF](https://arxiv.org/pdf/2608.17514v1)

**作者:** Bishwash Khanal `[一作]` (University of Jyvaskyla), Abhishek Kumar `[通讯]` (University of Jyvaskyla)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为SE-MoLoRA的模块化参数高效适配框架，能够在同一模型上实现共享知识与专门审美判断的分离，进而生成针对构图、光照和技术质量的可操作摄影评析；

**💡 创新点**

创新点在于将LoRA与Mixture-of-Experts相结合，构建永远激活的共享适配器与按需路由的专业适配器，并通过正交正则化与谱权重平衡实现领域间的解耦与可控性；

**🔧 技术方法**

采用LoRA进行参数高效微调，结合基于文本与多模态的路由器（DistilBERT、CLIP），使用正交正则化、SVD分析及BERTScore等指标评估；

**📊 数据集**

主要使用Reddit Photo Critique Dataset（RPCD），通过LLM蒸馏将自由文本评论拆分为四个专业域的标注样本；

**📈 对比分析**

与零样本基线和单体LoRA进行对比，SE-MoLoRA在BERTScore‑F1上提升至0.4215（从0.2317提升），在专家间的对比中赢率达84.6%，并在Q‑Bench、PhotoBench等零样本评测中分别达到57.65%和53.97%，显著优于基线；

**⚠️ 局限性**

局限在于每次仅激活单一专业适配器、使用硬路由器、缺乏专业摄影师人类评估、仅在单一基础模型和三大领域上测试，未来需扩展多专业联合激活、软路由、以及更大规模的基准验证。

---

## 149. What Cognitive Accessibility Reveals About Data Visualization

**arXiv ID:** 2608.17039 | [PDF](https://arxiv.org/pdf/2608.17039v1)

**作者:** Keke Wu `[一作]` (University of Maryland), Jonathan Lazar `[通讯]` (University of Maryland)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对认知可访问性研究的综述，检视可视化中的认知、可访问性和受众假设，并基于智力发育障碍与唐氏综合症个体的研究提出重新思考可视化的方法。

**💡 创新点**

认为认知多样性可作为对可视化理论的“压力测试”，指出传统假设过度依赖抽象推理、注意力与图形素养，提出更广阔的理解路径、意义参与与多样受众视角。

**🔧 技术方法**

主要采用参与式与协同设计研究、图表理解实验、日常数据实践访谈等定性方法。

**📊 数据集**

未使用公开数据集，主要基于从智力发育障碍和唐氏综合症参与者收集的原始数据。

**📈 对比分析**

该工作不提供量化性能比较，评估以案例分析与受众反馈为主，强调功能性意义而非任务效率。

**⚠️ 局限性**

研究聚焦于特定认知障碍人群，未能覆盖更广泛的认知差异或技术实现细节，可能缺乏普适性验证。

---

## 150. Iterative tensor network transformations for element-wise evaluation of elementary and filtering functions

**arXiv ID:** 2608.17135 | [PDF](https://arxiv.org/pdf/2608.17135v1)

**作者:** Xiao Wang `[一作]` (University of Oxford), Dieter Jaksch `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了ITNT框架，在压缩的张量链表示下实现任意函数的逐元素评估，并将其应用于三维反应流场和最大化3‑SAT优化问题。

**💡 创新点**

创新点在于把迭代（如Newton‑Raphson、二进制固定、消去等）与有限秩张量网络操作相结合，使得非线性变换能够在压缩域中高效、可控地完成，从而突破了传统张量网络只能进行线性或Hadamard乘法的局限。

**🔧 技术方法**

采用张量链（TT）表示、SVD截断、部分积分、元素乘积、Newton‑Raphson迭代、二进制固定与消去（deflation）等技术。

**📊 数据集**

使用了高分辨率的甲烷/空气喷射火焰温度场（约百万格点）以及来自Max‑SAT 2016竞赛的70变量、700条子句的Max‑3‑SAT实例（约2^70个配置）。

**📈 对比分析**

与Tensor Cross Interpolation（TCI）以及最先进的启发式求解器对比，ITNT在火焰反应速率计算中误差低于TCI 100倍；在Max‑3‑SAT中在O(10^4)步内找到了与最优解相同或仅差1的解，所需的张量秩仅在千级，远低于暴力搜索的2^70阶乘。

**⚠️ 局限性**

局限性包括：对高度非线性或退化函数的收敛需要大量迭代；截断误差会随自乘次数指数增长，需在秩与精度之间权衡；对解的严格可证明最优性仍需要指数级的秩或误差预算；实现依赖于双精度浮点，可能在极端场景下失稳。

---

## 151. FetchMan: Learning Visual Humanoid Loco-Manipulation Policies from Simulated Experiences

**arXiv ID:** 2608.17027 | [PDF](https://arxiv.org/pdf/2608.17027v1)

**作者:** Omar Rayyan `[一作]` (University of California), Yuchen Cui `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在仿真中利用大规模合成演示数据训练视觉驱动的全身控制策略，并在不使用任何真实数据的情况下直接零样本部署到真实Unitree G1机器人，实现了从行走到抓取的无缝转换。

**💡 创新点**

创新点包括：① 使用脚本化的全身控制器在 150k+ 程序生成场景中产生高质量演示，突破了多姿态、交互复杂度的瓶颈；② 通过 Flow‑GRPO 进行基于流匹配的强化学习微调，克服了演示中隐式阶段边界导致的模仿上限；③ 设计了基于 DINOv3 视觉编码与 delta‑action 参数化的跨模态政策架构，显著提升了 sim‑to‑real 转移；④ 将上述方法扩展到多物体文本条件版本，迈出了多目标通用 loco‑manipulation 的第一步。

**🔧 技术方法**

主要技术包括：程序生成场景（MolmoSpaces）、脚本化演示生成、行为克隆（BC）、流匹配的 Flow‑GRPO 强化学习、固定 DINOv3 视觉编码器、delta‑action 目标、跨模态文本编码（dino.txt）、以及 SONIC 低层控制器的分层控制。

**📊 数据集**

使用数据集：MolmoSpaces 生成的约 150k 场景中的 150k 个碗抓取演示（约 650 小时体验），以及后续扩展至 350k 个多物体演示；评测基准为自建的 FetchMan‑Bench（100 个持有场景），用于模拟与真实场景的统一评估。

**📈 对比分析**

对比方法：单阶段 BC（行为克隆）与 BC+RL（BC+Flow‑GRPO）。在模拟下，单阶段 BC 达到 67% loco‑manipulation 成功率，经过 Flow‑GRPO 提升至 83%；在真实 Unitree G1 上，BC 为 56.7%，BC+RL 为 73.3%。多物体版本在模拟下 BC 40%→62%，相对单物体仍略低，但已证明可行。

**⚠️ 局限性**

局限性：① 政策无历史记忆，无法直接观测演示中的阶段索引，需通过单帧推断；② 低层行走控制固定为 SONIC，无法自适应不同负载或不平衡；③ 仅限于抓取（fetch）任务，未覆盖更复杂的操纵或长时序行为；④ 需要大规模仿真计算，无法在资源受限环境快速复现。

---

## 152. MoFE: A Novel Mixture-of-Experts Framework with Fourier Neural Operators for Cryptocurrency Forecasting

**arXiv ID:** 2608.17342 | [PDF](https://arxiv.org/pdf/2608.17342v1)

**作者:** Bowen Liu `[一作]` (University of Rochester), Mingming Sun `[通讯]` (Beijing Institute of Mathematical Sciences and Applications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 MoFE 混合专家框架，将 Fourier Neural Operators 与 Mixture‑of‑Experts 结合，用于加密货币价格预测。

**💡 创新点**

通过在频域和时域双专家路径、动态门控与正交正则化实现多尺度、非平稳性建模，并引入综合损失抑制相位滞后。

**🔧 技术方法**

采用 AFNO、卷积局部专家、Mixture‑of‑Experts、softmax 门控、正交正则化、复数 MLP、频谱低通滤波、Soft Shrinkage 等技术。

**📊 数据集**

使用 2020‑2025 年比特币每日 OHLCV 数据并补充 RSI、MACD、RVol、VolProxy 等四项技术指标，构成 8 维特征，训练/验证/测试按 4:1:1 划分。

**📈 对比分析**

与 LSTM、GRU、iTransformer、CryptoMamba、FreqMoE 等基线在 T+1/T+5 上用 RMSE/MAE/R²、IC/DA、Sharpe 等指标对比，MoFE 在所有指标上均优于基线，尤其在方向性准确率和风险调整收益方面显著提升。

**⚠️ 局限性**

仅基于内部市场变量，难以捕捉极端“黑天鹅”事件；32 天回溯窗口不足以捕获低频周期（如比特币减半）；在高频噪声环境下仍可能出现专家过度拟合。

---

## 153. Effects of Answer Format Variation on Gender Bias in Large Language Models

**arXiv ID:** 2608.17516 | [PDF](https://arxiv.org/pdf/2608.17516v1)

**作者:** Ksenia Merzlyakova `[一作]` (University of Stuttgart), Franziska Weeber `[通讯]` (University of Stuttgart)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大语言模型（LLM）在性别偏见评估中不同答复格式（封闭式、多点李克特尺度、开放式）对偏见测量和与人类意见分布的对齐的影响进行系统实验，探索格式变化如何改变模型行为与评估结果。

**💡 创新点**

创新点在于：①首次将答复格式作为实验变量，对比同一模型在不同格式下的偏见表现；②构建并公开多格式的 BBQ 与 OpinionQA 子集，提供跨格式评估资源；③结合偏见分数、极化指数和分布相似度（Wasserstein、Jensen‑Shannon 等）多维度评估模型表现，揭示格式对排名和指标的翻转效应。

**🔧 技术方法**

使用指令调优的 7–12B 参数 LLM（如 Llama2‑7B、Mistral‑7B、Claude‑2‑7B），通过 prompt 生成与格式化，使用 LLM 辅助标注开放式答案，并计算偏见分数、极化指数、分布距离等统计指标。

**📊 数据集**

数据集包括：① BBQ（Bias Benchmark for Question Answering）中的性别相关条目，② OpinionQA（基于 Pew Research 的美国趋势调查）中的性别意见题目；两者均被重新格式化为三种答复形式。

**📈 对比分析**

比较方法：对每个模型在同一问题下分别生成三种格式的多次回答（10 次），计算偏见分数与人类分布的距离；结果显示不同格式导致偏见方向和大小显著变化，甚至导致模型排名翻转；在开放式格式下，模型往往更倾向于“拒绝”或非实质性回答，导致偏见得分下降。

**⚠️ 局限性**

局限性：①仅评估 3 个小型模型，结果不一定适用于更大模型；②仅研究性别偏见，未覆盖种族、残疾等维度；③使用二元性别标签，忽略非二元身份；④仅使用美国英语问卷，文化与语言差异可能影响普适性；⑤开放式答案标注依赖 LLM，可能引入误差；⑥不同格式本身对任务定义产生影响，难以完全分离格式与任务差异。

---

## 154. Mask What Matters: Saliency-Guided Video Self-Supervised Learning for Autonomous Driving

**arXiv ID:** 2608.17178 | [PDF](https://arxiv.org/pdf/2608.17178v1)

**作者:** Christopher Lang `[一作]` (Robert Bosch GmbH), Abhinav Valada `[通讯]` (University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在自动驾驶场景中提出了一种基于运动补偿帧差（mcd）生成的显著性引导掩码策略，替代传统的随机多块掩码，改进了 V‑JEPA 视频自监督学习框架，以提升对安全关键动态信息的学习能力。

**💡 创新点**

创新点主要在于：①利用运动补偿的帧差消除视角运动带来的干扰，得到更可靠的显著性热图；②通过前景与背景的掩码概率差异化，强调安全关键物体的可见性；③仅修改掩码策略，无需额外监督、损失或网络结构改动；④在驾驶域中实现了对显著性信息的自监督利用。

**🔧 技术方法**

采用的技术包括：V‑JEPA 的潜在空间掩码预测框架；Sobel 边缘、光流幅值与 mcd 三种显著性信号；前景/背景概率掩码；ViT 编码器（ViT‑B/L）；EMA 或冻结教师；方差崩溃惩罚；轻量级预测器。

**📊 数据集**

预训练使用公开的驾驶视频数据集：nuScenes、BDD100k、KITTI、ImpromptuVLA；下游评估使用 Cityscapes（语义分割）、KITTI‑2015（单目深度）、BDD100k（多目标跟踪）、NuScenes‑MQA、OmniDrive、ImpromptuVLA（视觉问答）等。

**📈 对比分析**

在四个驾驶相关下游任务上与 DINOv2/DINOv3、OpenCLIP、VideoMAE、原始 V‑JEPA 随机掩码等基线对比：多目标跟踪 ID‑switch 降至 25%（从 12002 降到 9134）；Cityscapes 语义分割 mIoU 最高达 73.2；KITTI‑2015 单目深度 RMSE 3.75；VQA 任务中整体准确率提升，尤其是 OmniDrive 与 ImpromptuVLA 的规划误差减少。

**⚠️ 局限性**

局限性：模型是专为驾驶域设计，缺乏对更大、跨域视觉任务的泛化能力；在更大规模或混合域预训练中的效果尚未验证；受限于 ViT‑B/L 规模，较大模型在某些任务上仍表现欠佳；需要进一步研究如何将 mcd‑掩码迁移至更通用的视觉基础模型。

---

## 155. UniReflex: Plug-and-Play Force Control for Pretrained Generative Policies via Fast-Slow Reflex

**arXiv ID:** 2608.17432 | [PDF](https://arxiv.org/pdf/2608.17432v1)

**作者:** Yan Huang `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

为冻结的生成式模仿学习策略（如 DP、π₀、DreamZero）增设一个轻量级的快速反馈模块 UniReflex，使其在接触阶段实现闭环变量阻尼控制（VIC），而不需要对主策略进行进一步微调。

**💡 创新点**

创新点：① 非侵入式截取生成器动作头的深层隐藏表示，作为全局意图；② 通过监督解耦的抗力和各向异性刚度方向，实现可调阻尼；③ 采用自适应门控在定位优先阶段与力反馈优先阶段无缝切换；④ 仅训练少量 fast‑GRU 参数即可兼容多种生成器，显著提升训练效率与内存占用。

**🔧 技术方法**

主要技术包括：生成式模仿学习（Diffusion Policy、CVAE‑style VLA、Flow‑matching 生成器）、非侵入式钩子读取 latent 表示、变量阻尼控制（VIC）框架、正则化的各向异性刚度标签、快速 GRU 递归网络、基于位移误差的门控机制、以及力误差评估。

**📊 数据集**

数据集：通过双臂遥控演示收集 200 条轨迹，涵盖五个接触密集任务（曲线擦拭、斜面擦拭、充电器插拔、贴纸剥离、芯片盒翻转）。

**📈 对比分析**

对比方法：① 只冻结主策略（DP、π₀、DreamZero）；② 端到端的力感知策略（RDP、ForceVLA、TA‑VLA）。评估指标包括两阶段成功率、力跟踪误差、恢复速度和训练时延。实验结果表明：UniReflex 在接触阶段的成功率提升 20–60%，力跟踪误差降低至 5–9%，相对全微调仅需 25–66× 的后向传播时延，参数占用仅为主模型的 0.19–0.38%。

**⚠️ 局限性**

局限性：① 对长时程动作生成器（如 DreamZero）效果相对弱；② 更适合持续接触而非高冲击短暂插入任务；③ 真实硬件的力感知精度、低层控制频率及机械带宽仍是性能瓶颈。

---

## 156. ArguLens: An Open-Source System for Automated Essay Scoring and Label-Aware Feedback Generation

**arXiv ID:** 2608.17356 | [PDF](https://arxiv.org/pdf/2608.17356v1)

**作者:** Weiran Wang `[一作]` (Fudan University), Wenjuan Qin `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文开发了一个开源的模块化自动作文评分系统 ArgueLens，将评分过程拆分为论证结构分类、基于31个语言学特征的 LightGBM 评分以及基于 LLM 的标签感知反馈。

**💡 创新点**

创新点在于将 AES 过程解耦为可插拔组件，提供句子级可解释的论证结构预测、无分级 LightGBM 评分与 LLM 生成的结构化反馈，并实现本地部署与开源。

**🔧 技术方法**

使用技术包括 Qwen2.5‑7B + LoRA 的句子级论证分类、LightGBM 的无分级评分、vLLM + Qwen2.5‑14B 的反馈生成、Gradio UI、Tensor Parallelism、LoRA、Spacy、TAALED 与 QuanSyn 等。

**📊 数据集**

数据集采用 US 中学论证作文集 PERSUADE 2.0，包含 25,000+ 篇作文、12,000 句子及 1–6 级评分。

**📈 对比分析**

在句子分类上准确率 82.6%（macro‑F1 0.727）；在 5 折 prompt‑grouped CV 的 LightGBM 评分上平均 QWK 0.813，使用金标论证特征时提升 0.055；推理延迟约 1.3 秒（分类）和 6–9 毫秒（评分）。

**⚠️ 局限性**

局限性包括仅在 PERSUADE 2.0 上训练，难以泛化到其他体裁、年龄或语言；组件级评估未验证端到端效果；缺乏人类反馈评测与子群体公平性分析；对未完成草稿和评分者差异的鲁棒性不足。

---

## 157. RENESIS: Energy-Aware Synthesis of Adiabatic Logic from Irreversible Netlists

**arXiv ID:** 2608.17139 | [PDF](https://arxiv.org/pdf/2608.17139v1)

**作者:** Mitchell A. Thornton `[一作]` `[通讯]` (Southern Methodist University), Mitchell A. Thornton (Southern Methodist University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一款自动化合成工具，将常规不可逆的Verilog网表转化为可逆且可实现能量回收的绝热逻辑电路，并以每周期切换电容为唯一优化指标。

**💡 创新点**

创新点包括：①将信息抹除与切换能耗通过不同Rényi熵阶分离，避免将不可逆抹除误认为能量开销；②提出永不退步的可验证合成流程，任何接受的重写必须在两张能量表上不增益；③设计多目标技术映射器，支持八种绝热族并可强制系列深度约束。

**🔧 技术方法**

技术手段涵盖：向量空间信息模型（VSIM）实现前后向标签传播；布尔函数的ANF、ESOP与FPRM展开用于能量估计；基于切片的覆盖与技术映射；Bdd/MaxSAT求解用于最优性间隙分析；Reyni熵计算用于信息与能量分离。

**📊 数据集**

实验数据集为20个开发电路（含子表、查找表等典型低功耗基准）和20个hold‑out电路；使用Nangate45库及其标准单元；对九种能量回收族进行交叉验证。

**📈 对比分析**

评估方法为与ABC等传统低功耗映射进行对比；在开发集上实现14/20电路能量改进，hold‑out集实现15/19改进；中位能量提升达到0.91；通过最优性间隙程序评估与搜索空间的距离。

**⚠️ 局限性**

局限性包括：仅关注切换电容，未完整建模功率时钟生成与时延；系列深度约束为设计规则而非成本项；工具仅支持特定绝热族；缺乏工业级验证与对多相功率钟的完整支持。

---

## 158. Evaluating RL Explainability Methods by How Much They Help Fix Bugs in Agents

**arXiv ID:** 2608.17524 | [PDF](https://arxiv.org/pdf/2608.17524v1)

**作者:** Ram Rachum `[一作]` (University of California, Berkeley), Cameron Allen `[通讯]` (University of California, Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EvalXRL 基准，用 RL 代理修复任务来评估 XRL 方法的实用性；

**💡 创新点**

将“能否帮助修复”作为功能性指标，使用 LLM 编程代理作为可扩展、闭环的评估主体，构建无网络双容器隔离的实验框架；

**🔧 技术方法**

采用统一 diff 注入恶意 bug、LLM 编程代理、Docker 双容器隔离、交互式 XRL 调用、奖励归一化等技术；

**📊 数据集**

基于 JAX 实现的 Treasure Grid、Datacenter Cooling、Traffic Light Control 三种 RL 环境，配合人工设计的多种 malfunction（reward clipping、reward hacking、myopia 等）；

**📈 对比分析**

对比多种 XRL 方法（reward decomposition、counterfactual、action explanations、saliency、决策树提取等）与无方法基线和作弊 oracle，提出 H1‑H3 预测不同方法在不同 malfunction 上的表现差异；实际性能数据尚未给出；

**⚠️ 局限性**

存在 LLM 与人类验证差距、闭环调用效益未知、方法包装对结果的影响、合成 bug 与真实 bug 的差距、仅在白盒环境下评估、预训练记忆与泄漏风险、仅评估修复效用、单一 scaffold 限制、多方法协同评估缺失等局限。

---

## 159. LIBERO-VIFO: Benchmarking the Capability and Safety of Visual Cue Following in Vision-Language-Action Models

**arXiv ID:** 2608.17600 | [PDF](https://arxiv.org/pdf/2608.17600v1)

**作者:** Zhengyan Qian `[一作]`, Jinhui Tang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建并使用 LIBERO‑VIFO 基准评估视觉提示在 Vision‑Language‑Action 模型中的理解与安全性。

**💡 创新点**

首次将视觉提示划分为八类并设计授权与非授权评估协议，从能力与安全两个维度系统化评测。

**🔧 技术方法**

使用视觉‑语言‑动作模型、视觉问答、闭环执行、自动化提示构造管线，并在仿真与真实机器人上进行实验。

**📊 数据集**

LIBERO‑VIFO 任务与视觉提示库共1347个实例，涵盖八类视觉提示；额外使用 HazardArena、MuJoCo 场景进行进一步验证。

**📈 对比分析**

对七种主流 VLA 模型进行评测，发现全链准确率低于17.6%，授权执行率差异显著，未授权执行率可达32‑60%，不同模型在各提示类别表现不一致。

**⚠️ 局限性**

提示库有限、提示信息可能被视觉捷径替代、未授权提示忽视结果受限于现有模型对视觉的认知强度。

---

## 160. EATR-Stereo: Embodiment-Aware Routing of Paired Stereo Evidence for Humanoid Vision-Language-Action Control

**arXiv ID:** 2608.17453 | [PDF](https://arxiv.org/pdf/2608.17453v1)

**作者:** Songwei Wu `[一作]` (Harbin Institute of Technology), Hong Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了EATR‑Stereo框架，利用同步立体相机的主/辅视图，并结合身体语义分段的本体状态对辅助视图的token进行路由，从而在保持预训练视觉‑语言模型主视图不变的前提下，提升人形机器人在长时域视觉‑语言‑动作任务中的可靠性。

**💡 创新点**

创新点在于①保留主视图原有token而不替换；②构建与主视图对齐的Cross‑View Auxiliary Tokens (CVATs)；③通过分段本体感知对CVAT进行token‑级路由，实现状态条件的自适应多视图融合。

**🔧 技术方法**

技术包括冻结Cosmos VLM进行视觉‑语言编码、跨视图注意力生成CVAT、身体分段感知编码与注意力权重的路由网络、GR00T N1.7流匹配动作专家以及B‑spline连续化执行。

**📊 数据集**

在33‑DoF Omega 1.0人形机器人上使用同步头戴立体相机和37‑D本体感知数据，进行超100秒的搜索‑接近‑抓取‑放置‑返回任务；在Franka‑arm RoboCasa365仿真平台上使用18个任务的演示数据。

**📈 对比分析**

与GR00T‑Mono、GR00T‑Wide、StereoPolicy、CVAT、CVAT‑Flat等基线相比，EATR‑Stereo在物理实验中实现60%完整任务成功率、100%抓取成功率、80%阶段成功率，显著优于StereoPolicy的45%完整任务成功率；在仿真中取得43.33%成功率，领先于其它方法。

**⚠️ 局限性**

局限在于：仅在特定人形平台和头戴立体相机条件下验证；未对不同相机参数、极端遮挡、噪声等更广泛的视角变化做充分评估；对深度估计或不确定性建模等可能进一步提升的技术尚未整合。

---

## 161. AI, Brain Death Detection, and Islamic Law

**arXiv ID:** 2608.16903 | [PDF](https://arxiv.org/pdf/2608.16903v1)

**作者:** Muhammad Aurangzeb Ahmad `[一作]` (University of Washington Bothell), Muhammad Aurangzeb Ahmad `[通讯]` (University of Washington Bothell)

**通讯引用:** 1843 | [OpenAlex ID](https://openalex.org/A5060934705)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究AI检测神经损伤患者的隐蔽意识并将其与伊斯兰法中的脑死亡判定相结合，提出跨学科研究议程。

**💡 创新点**

首次将机器学习的概率输出与伊斯兰法的证据等级（bayyina、yaqīn 等）对齐，阐明技术进展对宗教伦理的影响。

**🔧 技术方法**

利用多模态深度学习（EEG、fMRI、PET 等）与传统机器学习（SVM、XGBoost 等）相结合的模型。

**📊 数据集**

使用大规模 EEG/ECoG/LFP 记录（约 680,000 条样本）以及多中心欧盟临床试验数据。

**📈 对比分析**

与传统行为学评分（如 CRS‑R、GCS）进行对比，证明模型在保留意识的检测上准确率显著提高，误诊率降低至 40% 以下。

**⚠️ 局限性**

局限性包括概率输出无法达到 bayyina 的确定性、对人类心灵（rūḥ）的不可测定性以及对穆斯林人口的代表性不足。

---

## 162. A decodability criterion predicts when hidden-state selection beats majority voting in large language models

**arXiv ID:** 2608.17124 | [PDF](https://arxiv.org/pdf/2608.17124v1)

**作者:** Zhixiang wang `[一作]`, Ulas Bagci `[通讯]` (Northwestern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨如何利用LLM内部隐藏状态来动态选择生成答案，提出了CASE（Correctness‑Axis SElection）方法，并定义了无泄漏的“解码性”指标来预测内部选择是否优于多数投票。

**💡 创新点**

创新点在于：①引入了within‑question ROC‑AUC作为可测量的解码性度量，能够在部署前预判内部选择的有效性；②发现阈值≈0.60能可靠地区分会提升还是会降低性能；③通过理论与实验证明内部选择在Byzantine regime下随候选数增长而提升，而投票则退化。

**🔧 技术方法**

技术方法包括：在答案词位置的残差流上训练线性（logistic）门；使用泄漏‑free的question‑grouped交叉验证；对比多数投票、单一代理、生成式自检与其他近零成本输出空间选择器；并对模型与任务的规模、专业化、开放/闭卷等因素进行单变量控制。

**📊 数据集**

实验数据集涵盖逻辑推理（LogiQA）、医学问答（MedQA、MedMCQA、PubMedQA）、数学竞赛（MATH‑500、GSM8K）以及研究生科学（GPQA）等多领域基准。

**📈 对比分析**

与多数投票及其它零成本方法对比，CASE在中等难度问题上平均提升约19个百分点，难度最高问题提升约16.8个百分点；在符合阈值的设置中，CASE可超过投票10–20个百分点；在低难度或阈值以下的设置中，CASE不显著优于投票。

**⚠️ 局限性**

局限性包括：阈值区间的确定仍需经验调优；在低难度或未超阈值场景下无优势；实验受限于所选基准与模型规模，且尚未在开放式生成任务中验证；解码性与性能的关系虽强，但可能因数据分布或任务特性变化而失效。

---

## 163. ASI-Bench: At the Dawn of Artificial Superintelligence

**arXiv ID:** 2608.17271 | [PDF](https://arxiv.org/pdf/2608.17271v1)

**作者:** Junwei Zhou `[一作]` (University of Michigan), Jingyan Xie `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了ASI-Bench，一个包含60个跨学科项目级科研任务的基准，设计了从完整方法指导（B1）到无方法指导（B3）及带干扰信息的B4的指导梯度，以评估AI在自主探索与执行科学研究方面的能力。

**💡 创新点**

首次将科研创新与自主执行统一为单一基准，采用渐进式方法指导消除来量化AI的自主性，并通过严格的专家评审和沙箱验证保证任务可靠性。

**🔧 技术方法**

利用大语言模型与多种Agent架构（如Codex、Claude、GPT-5.6等）以及工具驱动的执行环境，对任务进行自动化推理、方法规划、代码生成与实验执行。

**📊 数据集**

使用了60个经过专家设计的科研项目数据，涵盖11个学科（数学、物理、化学、生命科学、天文学、材料科学、地球科学、医学与生物统计、计算机科学、机器人学、电气工程），并在公开的ASI-Bench平台提供。

**📈 对比分析**

对18个Agent×Model组合在B1–B4四个指导层次下进行宏观平均评分；在B1平均得分50.91，B2降至29.10，B3进一步下降到26.62，最高单一配置在B3得分51.60，表明当前系统在减少指导时性能显著衰减。

**⚠️ 局限性**

主要局限是对方法实现的依赖仍较高，方法选取与工具执行的精细化仍难以完全自动化；任务数量和领域覆盖虽已扩大，但仍不足以覆盖所有科研挑战，且部分系统受算力与成本限制。

---

## 164. Remote-Timer-as-a-Service: Efficient Microarchitectural Leakage in the Cloud with Remote Timers

**arXiv ID:** 2608.17043 | [PDF](https://arxiv.org/pdf/2608.17043v1)

**作者:** Martin Schwarzl `[一作]` (Cloudflare, Inc.), Nigel Topham `[通讯]` (University of Edinburgh)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并演示了在 Cloudflare Workers 生产环境中利用远程 Spectre 攻击泄露 JWT 的攻击，并在此基础上改进了检测和缓解措施。

**💡 创新点**

创新点包括：①利用可共存脚本的远程高分辨率定时器；②嵌套 PLRU 放大技术以抵御系统中断；③鸽笼式失效排除的堆地址与秘密泄露 gadget；④基于 Intel MPK 的内进程隔离方案。

**🔧 技术方法**

使用的技术包括：远程 WebSocket 定时器、V8 语言级隔离、PLRU 缓存放大、WebAssembly 进行噪声抑制、Pigeonhole eviction、V8 Sandbox、Intel MPK。

**📊 数据集**

数据集为 Cloudflare Workers 生产实例，使用 AWS EC2 服务器作远程定时器；实验在夜间低负载下收集。

**📈 对比分析**

与先前 120 bits/s 的攻击相比，本攻击成功率提升至 12 bits/s（约 360 倍），准确率达 99.16%，通过多次重放和动态阈值校准实现。

**⚠️ 局限性**

局限性包括：需实现脚本共址才能发动；远程定时器精度仍受系统噪声影响；只能针对已部署 Workers；MPK 关键字仅 12 个，且未迁移到 Sandbox 的对象仍存在漏洞。

---

## 165. Graphectory Viewer: A Tool for Process-Centric Analysis of Agentic Software Trajectories

**arXiv ID:** 2608.17195 | [PDF](https://arxiv.org/pdf/2608.17195v1)

**作者:** Charlie Jyu `[一作]` (University of Illinois at Urbana-Champaign), Reyhaneh Jabbarvand `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了基于 Graphectory 表示法的交互式浏览器工具 Graphectory Viewer，用于可视化和分析软件代理的执行轨迹；

**💡 创新点**

创新点在于将低级执行细节映射为语义阶段并构建阶段感知图，实现节点级检视、Sankey 期转摘要以及对大规模轨迹集合的聚合比较，并公开预计算图和大规模轨迹语料库；

**🔧 技术方法**

采用 Web 前端可视化技术（React+D3 等）、后端处理管道（Bash 解析、阶段标签化、节点去重）以及对 OpenHands、SWE-agent 框架的轨迹进行标准化；

**📊 数据集**

使用了约 4000 条 SWE-bench Verified 任务轨迹，包含 3973 条非空轨迹，来自 4 个 SWE-agent 与 4 个 OpenHands 运行；同时公开了原始轨迹、预计算图及相关脚本；

**📈 对比分析**

通过与 SWE-agent 命令行查看器的对比实验，Graphectory 在 5 项轨迹取证任务中的准确率从 8% 提升至 84%，完成时间显著缩短；在全数据集上评估图压缩率和节点去重率，展示了显著的信息压缩与行为模式揭示效果；

**⚠️ 局限性**

局限性包括样本量小、方便抽样、固定实验顺序、任务设计偏向视觉搜索，难以推广；系统目前依赖领域特定映射规则，扩展至新域需额外适配；未进行大规模用户体验评估。

---

## 166. YILDIZ-VPR: A Novel Dataset with Dense Coverage Under Diverse Environmental Conditions for Visual Place Recognition

**arXiv ID:** 2608.17033 | [PDF](https://arxiv.org/pdf/2608.17033v1)

**作者:** Serdar Yildiz `[一作]`, Songül Varli `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

收集了一个在Yildiz技术大学校园内进行多次步行采集的视觉位置识别数据集

**💡 创新点**

提供了稠密、行人级视角的多时段、多季节、多天气变化的户外场景，并结合同步GPS、速度、温度、陀螺仪等多模态传感器数据

**🔧 技术方法**

使用GoPro 9摄像头录制视频，并同步记录GPS、速度、温度、陀螺仪等传感器信息

**📊 数据集**

YILDIZ-VPR数据集（370,464训练图像，2,377日间查询图像，1,902夜间查询图像）

**📈 对比分析**

暂无对比实验或性能评估，论文仅介绍数据集及其特性

**⚠️ 局限性**

仅覆盖单一校园的户外环境，缺乏室内场景；GPS误差仍有可能影响标注精度；数据来源有限，未包含车辆或街景视角

---

## 167. When AI Designs AI: Innovation or Imitation?

**arXiv ID:** 2608.17471 | [PDF](https://arxiv.org/pdf/2608.17471v1)

**作者:** Yikang Yang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Jianfeng Zhan `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对比LLM代理设计的 AI 方法与人类专家设计的方法，分析它们在任务性能和算法设计上的差异。

**💡 创新点**

提出任务特定的算法设计空间，将人类与代理方法映射到同一坐标系，用模块级汉明距离量化差异，从而在不受实现细节干扰的前提下可比。

**🔧 技术方法**

利用多种主流 LLM 代理（Claude、Codex、Gemini、MLEvolve）生成方法，并用人类参考代码手工构建设计空间；随后计算坐标距离并统计性能排名。

**📊 数据集**

在六个跨模态任务上评估：图像分类 CUB‑200‑2011、情感分类 GoEmotions、图节点分类 ogbn‑arxiv、图链接预测 ogbl‑ppa、时间序列预测 ETTh1 与 Weather。

**📈 对比分析**

通过任务性能排名、平均排名以及算法距离分布进行比较；结果显示仅 10/72 组配置超过人类 SOTA，且大部分高性能方法仍与已有坐标相近，整体性能低于人类基准。

**⚠️ 局限性**

主要局限在于代理方法的探索受限于已有的人类设计空间，缺乏系统利用外部知识或真正的空间扩展，跨任务推广性不足。

---

## 168. Agent Lightning v1.0: Towards Harnessed Agentic RL

**arXiv ID:** 2608.17528 | [PDF](https://arxiv.org/pdf/2608.17528v1)

**作者:** Zhiyuan He `[一作]` (Microsoft), Chong Luo `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了Agent Lightning v1.0框架，实现代理与RL训练的解耦，以支持任意agent harness。

**💡 创新点**

首次系统性阐述并解决代理驱动RL训练中的重token化、优势计算、损失归一化和后端调度等挑战。

**🔧 技术方法**

采用代理式训练、回溯式异步RL、Kubernetes作业调度、奖励信号清洗与网络策略等技术。

**📊 数据集**

使用HotpotQA、SWE-smith等公开数据集进行搜索、指令跟随和代码生成任务。

**📈 对比分析**

与现有框架比较，Agent Lightning在搜索、指令跟随和代码生成任务上分别提升验证奖励至41.7%、70.2%和56.4%，显著优于基线。

**⚠️ 局限性**

受限于动态样本数导致的梯度不稳定，需进一步完善优势分配与损失归一化策略。

---

## 169. Task-Aware Harness Provisioning for LLM Agents in Mission-Critical Infrastructure Operations

**arXiv ID:** 2608.17433 | [PDF](https://arxiv.org/pdf/2608.17433v1)

**作者:** Liangtao Lin `[一作]` (Nanyang Technological University), Yonggang Wen `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在任务关键基础设施（MCI）中LLM代理的资源配置问题，并提出基于任务–硬件映射的两阶段升级策略；

**💡 创新点**

创新点在于：① 用系统方程式构造任务分类，② 从文献与执行两种来源构建任务–硬件映射，③ 提出基于映射的自检触发升级机制，揭示域依赖的 Pareto 前沿；

**🔧 技术方法**

主要技术包括：GPT‑5.4/Qwen3.5 等 LLM 代理、ReAct 交互、Reflexion 与 AutoMix 等执行适配方法、仿真数字孪生（液冷与 IEEE‑14 电网）、统计映射规则与自检机制；

**📊 数据集**

使用的数据集为：约 1,200 篇 MCI O&M 论文用于文献映射；240 个任务（12 类×10 实例）在液冷与电网仿真环境中构建的基准；

**📈 对比分析**

与全量硬件、相关性检索、LLM 路由、AutoMix、Blind‑ESC 等多种策略对比，Map‑ESC 在液冷任务上准确率提升至 0.715（比全量低 14% token），在电网保持 0.782，性能与 Reflexion 接近并在 token 方面提升 48%；

**⚠️ 局限性**

局限性：仅在仿真环境与特定 LLM 模型上评估，映射需针对新域/模型重新校准；未覆盖实时传感不确定性、组织流程与安全风险，且文献来源可能存在偏差。

---

## 170. Write, Execute, Refine: From Skill Followers to Skill Optimizers via Reinforcement Learning from Execution Feedback

**arXiv ID:** 2608.17587 | [PDF](https://arxiv.org/pdf/2608.17587v1)

**作者:** Kang Peng `[一作]` (Harbin Institute of Technology), Kam-Fai Wong `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 WER（Write, Execute, and Refine）框架，训练 Skill Optimizer 在固定执行器外部通过多阶段自我引导的方式改进自然语言技能。

**💡 创新点**

创新点在于将执行经验结构化为跨阶段的“refinement state”，通过群组相对奖励和匹配成功/失败轨迹实现对技能的自学习改进。

**🔧 技术方法**

使用 GRPO 强化学习、程序化验证器、冻结执行器（GPT‑4o）、多阶段自引导循环，对技能进行文本生成与评估。

**📊 数据集**

使用 BFCL v4 多轮任务集和 τ^2‑bench 基准；将两者训练集合并用于跨基准的训练。

**📈 对比分析**

与无技能、GPT‑5.1 seed、未训练的 Qwen3‑4B、Skill‑R1、Trace2Skill 等对比，WER 在 BFCL v4 平均 Pass@1 从 68.83% 提升至 76.63%，在 τ^2‑bench 从 46.87% 提升至 50.72%，并且相较于同一后端模型训练前提升约 10 个百分点；与大型通用模型相比，训练后的 4B 模型仍保持领先。

**⚠️ 局限性**

局限在仅评估可程序化验证的环境，未验证对开放式评估或未知工具的迁移；并且匹配轨迹保持原始记录导致状态尺寸随任务长度增长，可能在更长或多模态任务中出现扩展瓶颈。

---

## 171. OV3D-Bench: A Diagnostic Benchmark for Open-Vocabulary Monocular 3D Detection

**arXiv ID:** 2608.17110 | [PDF](https://arxiv.org/pdf/2608.17110v1)

**作者:** Mariia Gladkova `[一作]` (Technical University of Munich), Daniel Cremers `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文创建了OV3D‑Bench，一套诊断性评测协议，用于在更接近真实部署条件下对开词表单目3D检测器进行评估，并在该协议下对七种主流检测器及一种训练‑free VLM 重映射基线进行系统对比。

**💡 创新点**

创新点包括：①用数据集级别提示替代逐图像 oracle，消除训练时对类别的特权；②将评测拆分为几何准确性、语义鲁棒性和跨域迁移三大轴，避免单一 AP 误导；③提出仅用 SigLIPv2 进行预测重映射的训练‑free 基线，验证几何定位已成熟但语义仍是瓶颈；④对提示敏感性和 target‑aware 升级效应进行细致分析。

**🔧 技术方法**

技术手段主要有：SigLIPv2、CLIP 等对比式 VLM 用于类别重映射；GroundingDINO、SAM3、DINOv2 等用于检测与几何；标准的 3D IoU 与 BEV 评测；以及基于 Hungarian 匹配的误差分解。

**📊 数据集**

使用了七个公开数据集：KITTI、nuScenes、Argoverse 2、SUNRGBD、ScanNet‑200、ARKitScenes 与 Hypersim，涵盖室内外、稀疏与密集、少数与大词表等多样场景。

**📈 对比分析**

通过三轴评测（3D Class‑Agnostic Recall、数据集级 mAP3D 与 Prompt CV）对比七种检测器，发现：几乎所有模型在几何定位上表现良好，但语义错误显著；prompt CV 体现了对描述细节的敏感度；VLM 重映射基线在多数据集上与专门设计的开词表方法相当或更优。

**⚠️ 局限性**

局限性包括：VLM 重映射无法彻底区分真阳性与假阳性，导致误差重映射可能不完全；基准受限于各数据集的注释不完整，未标注对象会被错误惩罚；该重映射方案仅为诊断工具，尚不可直接作为高效部署的完整检测器；并且对复杂提示与多词表的鲁棒性仍需进一步研究。

---

## 172. DTX: A Throughput-First Training Accelerator for Diffusion and Transformer Models

**arXiv ID:** 2608.16953 | [PDF](https://arxiv.org/pdf/2608.16953v1)

**作者:** Shashank `[一作]` `[通讯]` (Independent Researcher), Shashank (Independent Researcher)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

研发了一款无循环依赖、通过流水线树形累加实现高吞吐的训练加速器DTX，支持扩散和Transformer模型的训练；

**💡 创新点**

核心创新在于将所有浮点累加转为并行树形流水线，并采用EPIC‑style四槽VLIW控制实现无循环依赖，同时引入容差验证保障数值精度；

**🔧 技术方法**

使用了8×8权重驻留的systolic array、8速率向量单元、融合AdamW管线、Philox伪随机数生成器、统一64 KB tile空间与AXI4 DMA，以及基于FP64的容差验证框架；

**📊 数据集**

内部验证基于扩散denoiser（MLP）和Transformer自注意力块的训练任务，未引用公开数据集；

**📈 对比分析**

与GPU训练基线的iso‑node比较显示DTX峰值216 FLOP/周期、约10×GPU吞吐量/瓦特，并在17/17测试中无误差；

**⚠️ 局限性**

局限在于需要主机协作阶段实现BF16连续运算、目前缺乏完全芯片级并行、以及受SRAM资源限制导致的物理实现约束。

---

## 173. Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements

**arXiv ID:** 2608.17310 | [PDF](https://arxiv.org/pdf/2608.17310v1)

**作者:** Zhi Zheng `[一作]` (National University of Singapore), Wee Sun Lee `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Agentic ESOpt，一种用于长时序、稀疏奖励的 LLM 代理全参数进化策略框架，能够在推理级 GPU 内存上完成微调。

**💡 创新点**

创新点在于：①将全参数 ES 与长时序代理相结合，消除梯度回传对内存的需求；②引入余弦衰减的扰动半径 σ，实现探索–利用的动态平衡；③利用轨迹级奖励归因，避免 RL 中的长时序信用分配难题。

**🔧 技术方法**

使用技术包括：进化策略（ES）采样、参数空间随机扰动、奖励加权更新、对比学习中的 z-score 标准化、cosine decay 扰动衰减、与 Prompt‑Space 方法的协同优化。

**📊 数据集**

实验数据集涵盖：控制式 Sudoku（可调最短成功时序）、ReAct‑Style 计算工具（Math DAPO/AIME、DocVQA）、WebArena‑Lite 浏览器任务、以及自动启发式设计（TSP、KP、ASP、ACO 任务）。

**📈 对比分析**

在长时序任务上，Agentic ESOpt 在 15 步 Sudoku、ReAct Math/DocVQA、27B WebArena 上均显著优于 Agentic GRPO 与 PPO，提升成功率 10–15% 甚至更高；在自动启发式搜索中，Agentic ESOpt 在 28/36 次比较中均获改进，平均提升约 12%。

**⚠️ 局限性**

局限性包括：对种群规模（G）敏感，强大模型对大 G 的依赖性降低；实验主要聚焦稀疏奖励与长时序场景，未充分验证对高频或连续奖励的适用性；需要进一步探索更大模型的通用规模定律与自适应扰动策略。

---

## 174. Evaluation of AI-based Visual Crack Detection in Steel Bridges Using Probability of Detection

**arXiv ID:** 2608.17726 | [PDF](https://arxiv.org/pdf/2608.17726v1)

**作者:** Andrii Kompanets `[一作]` (Eindhoven University of Technology), H. H. Snijder `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于概率检测（PoD）曲线的统计评估框架，用于比较计算机视觉（CV）裂纹检测方法与传统人工可视检查在钢桥裂纹检测中的性能，并通过Monte Carlo模拟评估其在结构可靠性分析中的实际效益。

**💡 创新点**

创新点包括：①将传统人工检查的PoD曲线方法迁移至CV方法；②在PoD模型中引入图像分辨率作为协变量，形成PoD表面模型；③提出统一的比较算法（Alg.1），能够直接衡量不同检测方法对裂纹长度分布的影响；④通过置信区间和KL散度等统计指标，对CV方法在不同裂纹尺寸与分辨率条件下的效果进行系统量化。

**🔧 技术方法**

技术手段主要有：使用Faster‑RCNN（ResNeXt101 backbone）训练裂纹检测器；利用逻辑回归（GLM）拟合PoD曲线；对分辨率影响采用分箱非参数法和参数化PoD表面（log‑log 回归）；利用最大似然估计和Wilk检验评估模型；通过Monte Carlo采样计算后验裂纹分布、归一化未检裂纹长度 C 和KL散度。

**📊 数据集**

数据集：使用“Cracks in Steel Bridges (CSB)”公开数据集（约6163张实测桥梁裂纹图像），其中239张图像含裂纹长度标注用于PoD曲线估计；其余图像划分为训练/验证/测试集（80/10/10%）用于CV模型训练。图像尺寸从600×450到4608×3456像素，包含不同光照、角度与距离。

**📈 对比分析**

比较方法：在先验裂纹长度分布（对数正态）下，用Monte Carlo模拟检验每种方法的检测概率，收集10⁶个未检裂纹样本；随后计算归一化未检裂纹长度 C（衡量剩余裂纹总量）和KL散度（衡量后验分布偏移）。实验结果显示：CV方法在裂纹长度 <150 mm 时的 PoD 高于 DNVGL 与 Campbell 的基线；在更大裂纹时性能介于两者之间；且在图像分辨率 ≥3 pix/mm 时，CV 方法可明显优于传统人工检查。

**⚠️ 局限性**

局限性：①仅评估一种 CV 方法（Faster‑RCNN），缺乏对其他深度学习模型的对比；②PoD 曲线估计依赖于手工设定阈值（阈值 0.01），对阈值敏感；③分辨率建模假设 IFOV 与距离满足简化公式，未考虑光照、视角等其他影响因素；④仅使用单一数据集，缺乏跨桥梁或跨国验证；⑤评估主要基于统计分布而非结构安全指标，实际工程应用还需结合更完整的可靠性分析。

---

## 175. PlanPO: Group Planning-Aware Policy Optimization for Multi-Turn Agentic LLMs

**arXiv ID:** 2608.17289 | [PDF](https://arxiv.org/pdf/2608.17289v1)

**作者:** Dayang Liang `[一作]` (Xiamen University), Yunlong Liu `[通讯]` (Xiamen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为 PlanPO 的基于组相对策略优化的新方法，利用成功轨迹的交互长度和文本生成长度的条件归一化优势信号，引导大型语言模型在多轮交互任务中学习更高效、更通用的规划能力。

**💡 创新点**

创新点在于：①把成功轨迹的长度信息（整体交互长度与单回合生成长度）作为多尺度优势信号；②在组相对优化框架中仅对成功轨迹进行长度归一化，避免了纯粹长度最小化导致的性能下降；③通过权重衰减平衡粗粒度轨迹优势与细粒度回合优势，达到 bias–variance 的最优权衡。

**🔧 技术方法**

主要技术包括：组相对优势计算、长度条件归一化、粗细尺度优势合成（A^E 与 A^S），以及在 PPO 基础上的剪辑组相对目标函数和 KL 正则化；此外还利用了多尺度长度特征的奖励塑造和动态 α(k) 权重调度。

**📊 数据集**

实验使用了三大多轮任务基准：ALFWorld、WebShop 与 SciWorld，并在 Qwen2.5-1.5B/7B-Instruct 模型上进行训练与评估。

**📈 对比分析**

与现有方法（GRPO、GiGPO、EMPG、RLOO、PPO 等）以及封闭源模型（GPT‑4o、Gemini‑2.5‑Pro）对比，PlanPO 在 ALFWorld、WebShop 的整体成功率平均提升 27.2%，在 ALFWorld 的 OOD 任务上提升 24.3%，显著超过所有基线，并在 SciWorld 上实现了最优或接近最优的整体分数。

**⚠️ 局限性**

局限性包括：①在某些科学子任务（如 Chem‑Mix）仍表现不佳，可能因模型对科学推理的理解与探索能力不足；②依赖长度信息的优势可能在极端长或高度冗余的场景中导致梯度噪声；③虽然开销很小，但对超大模型或更高维度任务的可扩展性仍需进一步验证。

---

## 176. Polaris: Learning to Generate Table Descriptions from Retrieval Feedback

**arXiv ID:** 2608.17171 | [PDF](https://arxiv.org/pdf/2608.17171v1)

**作者:** Ting Cai `[一作]` (University of Wisconsin-Madison), AnHai Doan `[通讯]` (University of Wisconsin-Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Polaris系统，利用检索反馈自动训练LLM生成针对关键词检索优化的表描述；

**💡 创新点**

创新点在于将检索基准数据转化为偏好对进行DPO微调，实现检索导向的元数据生成；

**🔧 技术方法**

采用LLM（Llama‑3.1 8B Instruct + LoRA）、Columbo式表/列名扩展、BM25检索以及Direct Preference Optimization；

**📊 数据集**

使用六个公开表检索数据集（Enterprise, Science, Science‑2, Government, Web, WikiTables）进行训练与评估；

**📈 对比分析**

与AutoDDG对比，Polaris在大多数指标下显著提升（如LTER NDCG@100提升44.6%，AW NDCG@5提升31.7%），且生成描述长度缩短47‑59%；

**⚠️ 局限性**

局限性包括需依赖检索相关性标注、仅针对BM25词汇检索优化、未利用表内容，且在无检索监督的领域效果受限。

---

## 177. Adaptive Participation Under Statically Equivalent Incentives in Distributed Demand Response Systems

**arXiv ID:** 2608.17469 | [PDF](https://arxiv.org/pdf/2608.17469v1)

**作者:** Xun Shao `[一作]` (Toyohashi University of Technology), Go Hasegawa `[通讯]` (Tohoku University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并验证了一种联合耦合的参与补偿机制，用于分布式需求响应系统，并研究其对学习型参与者行为的影响；同时提出了逐轮语义保真验证方法以评估分布式实现的正确性。

**💡 创新点**

发现传统静态验证条件无法区分两种在学习动态下表现完全不同的支付衰减结构；提出针对分布式实现的逐轮语义保真判据，揭示误差传播与学习反馈对机制行为的关键影响。

**🔧 技术方法**

采用经验式学习（logit+估计更新）、联合耦合支付函数、可变稀缺性衰减、基于CCNx的信息中心化/分布式实现、统计分析（Wilson区间）以及负控制实验。

**📊 数据集**

使用5个模拟电池单元的合成需求响应事件，24小时小时级别的可用性和马尔科夫链状态转换，并在固定事件日和随机种子下进行多次（96）实验。

**📈 对比分析**

通过逐轮对齐声明、联合配置、结算、学习状态和粗粒度偏好级别的完全相等比较；在8000轮内实现0误差，且不同衰减结构在学习完成率上差异显著（0% vs 100%）。

**⚠️ 局限性**

验证仅覆盖N=5、单一随机种子、特定运营点、profile-mean估计及短期通信扰动；未覆盖更大规模、不同参数、持续通信中断等；学习算法对先验敏感，缺乏完整鲁棒性证明。

---

## 178. Maximum Flow Without the Outer IPM

**arXiv ID:** 2608.17384 | [PDF](https://arxiv.org/pdf/2608.17384v1)

**作者:** Jason Li `[一作]` (Carnegie Mellon University), Alex Wice `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的最大流算法，该算法在m^(1+o(1))时间内计算有向、带容量图的近似伪循环，并避免了最近几种几乎线性时间算法中常见的内点法框架。

**💡 创新点**

创新点在于通过平衡权重技术生成伪循环，并结合标准流技术，提出了一种新的最大流算法，显著提高了计算效率。

**🔧 技术方法**

使用了动态最小比率切割数据结构和潜在函数来计算伪循环，并通过标准技术将伪循环转换为近似最大流。

**📊 数据集**

使用了具有整数和多项式界限的容量图，特别是包含无限容量弧(t,s)的图。

**📈 对比分析**

与现有的几乎线性时间算法相比，该算法在避免外部内点法的同时，仍能在m^(1+o(1))时间内计算出最大流，性能上有所提升。

**⚠️ 局限性**

限制在于算法依赖于图的预处理，且在处理过程中需要保证所有弧的容量是多项式界限的。

---

## 179. Graph Surgery and the Do-Operator: A Precise Correspondence for Acyclic Structural Causal Models

**arXiv ID:** 2608.17634 | [PDF](https://arxiv.org/pdf/2608.17634v1)

**作者:** Satpreet Makhija `[一作]` `[通讯]` (Ashoka University), Satpreet Makhija (Ashoka University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了结构因果模型中-operator的语义，证明在确定性无环结构因果模型下，图形手术（删除入射边）与将目标变量的机制替换为常数这两种操作在依赖层面是等价的。

**💡 创新点**

创新点在于给出了依赖层面的精确等价性证明，阐明了何时供给的图与实际依赖图一致，给出了顺序干预的组合律以及结果仅受其实际依赖祖先干预影响的祖先定理。

**🔧 技术方法**

使用了确定性无环结构因果模型的形式化定义、依赖图提取、函数依赖判定，以及严谨的数学证明技术（包含图形与函数视角的对比）。

**📊 数据集**

该工作为理论性研究，无需使用任何数据集。

**📈 对比分析**

通过在理论层面对两种干预方式进行形式化对比，提出等价性定理来比较方法；由于是纯理论结果，没有实验性能评估。

**⚠️ 局限性**

局限性在于仅适用于确定性无环结构因果模型，未考虑随机性或循环依赖的情形，也未提供实验验证。

---

## 180. SpurCon: Weighted Supervised Contrastive Learning for Mitigating Spurious Cues in Medical Imaging

**arXiv ID:** 2608.17598 | [PDF](https://arxiv.org/pdf/2608.17598v1)

**作者:** Shenhav Nadir `[一作]` (Technion Israel Institute Of Technology), Guy Gilboa `[通讯]` (Technion Israel Institute Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 SpurCon，一种轻量级的 spurious correlation 缓解框架；

**💡 创新点**

创新点在于引入基于元数据和预测的伪标签的加权监督对比损失 WtSupCon，并提供无训练的少样本伪标签推断方法；

**🔧 技术方法**

主要技术包括预训练视觉编码器、few-shot 伪标签推断、加权监督对比学习以及自定义采样策略；

**📊 数据集**

在三个公开数据集（Waterbirds、CheXpert 及 ISIC 2020）上进行实验；

**📈 对比分析**

与 Baseline、Baseline‑WT、JTT、DFR、CA 等方法对比，SpurCon 在 worst‑group、平均准确率和 AUC 上均优于或相当于最强基线，并且训练时间显著缩短；

**⚠️ 局限性**

局限性包括对元数据的依赖、对伪标签质量的敏感性以及在极度不平衡数据中仍可能出现少数类性能下降。

---

## 181. Pessimistic Meta-Induction and Its Limits: Lessons from Frequentist Statistics and Machine Learning Theory

**arXiv ID:** 2608.17213 | [PDF](https://arxiv.org/pdf/2608.17213v1)

**作者:** Hanti Lin `[一作]` `[通讯]` (University of California), Hanti Lin (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过构建一个基于无限二进制序列的数学模型，阐释并对比了普通归纳与悲观元归纳的归纳步骤，提出“处处收敛”与“几乎处处收敛”两种收敛标准，并证明在普通归纳情境下存在实现处处收敛的方法，而在元归纳情境下任何方法都无法实现至少几乎处处收敛；从而驳斥了悲观元归纳对科学实在论的威胁。

**💡 创新点**

创新点在于：① 将归纳推理的可靠性归根为收敛到真理的数学标准，形成一套统一的“可达成性”评估框架；② 用Cantor空间拓扑与Baire范畴定理，首次严格证明在元归纳情境中不存在任何几乎处处收敛的方法；③ 将统计学、机器学习与形式认识论中的收敛概念统一到科学推理的哲学讨论中。

**🔧 技术方法**

主要技术包括：
- 频率统计与非参数统计中的一致性概念；
- 机器学习中的点收敛与统一收敛；
- 形式认识论中对真理的收敛评估；
- Cantor空间拓扑与Baire范畴定理的应用。

**📊 数据集**

并未使用真实实验数据集；论文采用的是理想化的无限二进制序列（以及其对应的实数表示）作为“状态空间”来构建证明。

**📈 对比分析**

论文未进行实验性比较；其贡献主要在理论证明层面，展示在普通归纳情境下可以构造出处处收敛的推理方法，而在悲观元归纳情境下则不存在任何至少几乎处处收敛的方法。

**⚠️ 局限性**

局限性包括：
- 仅在极简的理论模型中证明，缺乏对复杂现实科学理论的直接适用性；
- 采用的收敛标准虽然统一但与实际科学实践中的证据评价机制仍有距离；
- 结论依赖于Cantor空间的拓扑假设，若现实科学问题的状态空间结构不同，结论可能不再成立。

---

## 182. WONDER: A Radio World Model-based Negotiation Framework for Multi-Agent UAV Coverage Optimization

**arXiv ID:** 2608.16955 | [PDF](https://arxiv.org/pdf/2608.16955v1)

**作者:** Jiahao Huang `[一作]` (Zhejiang University), Honggang Zhang `[通讯]` (Macau University of Science and Technology)

**通讯引用:** 12856 | [OpenAlex ID](https://openalex.org/A5100626780)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了WONDER框架，利用无人机群在灾后场景下的分布式协商与世界模型预测，实现快速覆盖恢复。

**💡 创新点**

创新点在于：①基于JEPA的无线世界模型，可在有限局部观测下预测候选轨迹的增量射频影响；②多轮协商机制与PPO决策者结合，递归降低组合误差；③理论分析证明顺序承诺优于并行计数误差，并给出误差下界；④构建RadioDynamics仿真环境，涵盖62个都市场景、射频传播、数字孪生几何和多跳通信。

**🔧 技术方法**

技术手段包括：Joint-Embedding Predictive Architecture (JEPA)、Proximal Policy Optimization (PPO)、多轮协商与提议筛选、射频传播模拟（光线跟踪）、数字孪生几何、A2G/A2A/Backhaul 通信建模、世界模型与决策器的交替训练。

**📊 数据集**

使用的数据集为RadioDynamics仿真环境，包含62个都市场景（香港、纽约、东京等），每个场景为700m×700m的网格（350×350），10架UAV，训练/验证/测试划分分别为42/9/11场景。

**📈 对比分析**

与Flocking、STACCA、MAPPO、MAGI以及带有完整信息的Exhausted（oracle）对比，WONDER在11个测试场景的平衡得分最高（0.870），比STACCA提升0.162，保持100%后端连通性，整体性能优于所有基线。

**⚠️ 局限性**

局限性包括：①依赖准确的射频世界模型，模型误差会影响决策；②多轮协商对通信延迟和可靠性有要求，真实环境下可能受限；③仅在模拟环境验证，缺乏真实部署实验；④规模扩展到更大无人机群或更复杂地形的可扩展性未深入探讨。

---

## 183. Bi-Layer Ant Colony Optimization for Multi-Robot Task Allocation and Routing in Delivery Applications

**arXiv ID:** 2608.17416 | [PDF](https://arxiv.org/pdf/2608.17416v1)

**作者:** Le Na Nguyen `[一作]` (Fulbright University Vietnam), Manh Duong Phung `[通讯]` (VinUniversity)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种双层蚁群优化算法，用于多机器人任务分配与路径规划

**💡 创新点**

将任务分配与调度统一为一个耦合优化问题，并在蚁群框架中构建双层信息素矩阵实现同时优化

**🔧 技术方法**

双层蚁群优化（Bi‑Layer ACO），基于距离启发式的概率选择与信息素更新

**📊 数据集**

仿真数据：ROS 2 Gazebo 10 m× 15 m 环境，3台 TurtleBot3 机器人，5/10/20 组随机起止点

**📈 对比分析**

与 MILP（混合整数规划）和 PSO（粒子群优化）对比，ACO 在平均总行程距离和任务完成时间上分别比 MILP 减少约17.7%/20% 以及比 PSO 减少约9.8%/20%，并且方差更小，稳定性更好

**⚠️ 局限性**

仅在静态仿真环境下验证；未考虑动态变化、机器人异构性和通信约束，实际部署中的鲁棒性与可扩展性待进一步研究

---

## 184. PDDL-ART: Autonomous Symbolic Abstraction From Demonstration For Long-Horizon Robotic Manipulation Using Vision-Language Models

**arXiv ID:** 2608.17146 | [PDF](https://arxiv.org/pdf/2608.17146v1)

**作者:** Disha Kamale `[一作]` (University of Michigan), Dmitry Berenson `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PDDL-ART 框架，利用视觉‑语言模型（VLM）从单一专家演示、自然语言任务描述和动作库中自动生成并多阶段校正 PDDL 域与问题文件，保证语法、语义与执行的正确性。

**💡 创新点**

创新点包括：
1) 零样本、无模板的 PDDL 生成；
2) 语法、规划可行性、初始状态、计划语义以及执行纠正的多阶段校正流水线；
3) 通过 VLM 自主调用几何/时序工具进行谓词评估，将符号验证与几何/时间推理相结合。

**🔧 技术方法**

技术手段：
- 大语言模型/视觉语言模型（GPT‑O3、GPT‑5.5）
- PDDL 语法解析器 VAL
- Fast Downward 规划器
- DTW 子序列匹配进行关键帧对齐
- VLM 工具接口（接近度检查、时间工具）
- 提示工程与链式推理。

**📊 数据集**

使用数据集：模拟的引擎维护与家庭烹饪任务，共 6 个场景（油位检查、油盖拆装、滤芯更换、换油、烹饪准备 I/II），每个场景包含 2–12 个对象、5–37 个动作，数据来自单一专家演示和任务说明。

**📈 对比分析**

与 Direct、CoT VLM‑PLAN、NL2PLAN 等基线进行对比，评估指标包括规划可行性、计划对齐率和执行成功率。PDDL‑ART 在所有任务中平均成功率达 93.3%，规划可行率 95%，在长周期任务上明显优于基线。

**⚠️ 局限性**

局限性：
- 仅支持完全可观测、确定性规划，无法处理条件、时序或数值约束；
- 生成的域仅针对单个演示，缺乏可复用性；
- 视觉或工具调用可能出现幻觉或噪声，导致误校正。

---

## 185. A Tight Linear Deterministic Competitive Ratio for Fully Online KV-Cache Scheduling

**arXiv ID:** 2608.16944 | [PDF](https://arxiv.org/pdf/2608.16944v1)

**作者:** Ian D'Ambrosio `[一作]` `[通讯]` (Nth Research Collective), Ian D'Ambrosio (Nth Research Collective)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

证明了在KV缓存增长约束下，LLM推理的在线批处理问题的竞争比是Θ(n)，即从先前的Ω(√n)下界提升到与最坏情况相同的线性上界。

**💡 创新点**

创新点在于：①首次完成了最坏情况内存约束下的竞争比上界和下界匹配；②使用机器可检验的Lean 4证明确保结果的可靠性；③通过构造特殊的“长请求+短请求”实例展示线性硬度。

**🔧 技术方法**

主要技术包括：离散时间的请求模型定义、非抢占式进度约束、预留内存约束、统一因果序列化调度策略、对极端实例的定量分析、以及Lean 4中的形式化验证。

**📊 数据集**

无真实数据集；所有结论均基于理论构造的有限实例和符号计算。

**📈 对比分析**

与以往仅给出Ω(√n)下界的理论比较，本文给出了完整的Θ(n)上界，证明了竞争比的精确量级；在理论上，任何确定性在线调度器的竞争比至少为n-1/12，最坏情况下不超过n。

**⚠️ 局限性**

局限性包括：①仅得到最坏情况内存比的线性上界，未给出对每个固定内存M的精确函数(n,M)；②构造实例使用提示长度约为M/2，可能不适用于所有实际场景；③常数1/12与1未经过优化，存在进一步改进空间。

---

## 186. CryptDough: A Unified Analytics Engine for Secure Multiparty Computation

**arXiv ID:** 2608.17529 | [PDF](https://arxiv.org/pdf/2608.17529v1)

**作者:** Muhammad Faisal `[一作]` (Boston University), John Liagouris `[通讯]` (Boston University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一套统一的安全多方计算（MPC）分析引擎，能够在单一系统中并行执行关系型查询、时序计算和机器学习推理等多种工作负载，并支持多种威胁模型。

**💡 创新点**

创新点包括：① 分层可扩展架构，统一的虚拟向量（virtual vector）机制，使高层用户可用单线程代码而后台自动完成通信、并行化和内存管理；② 抽象化的协议无关原语（protocol‑agnostic primitives），使新协议或新分析库可无缝集成；③ 在同一运行时内支持多种威胁模型和协议（ABY、ABY3、Fantastic Four、SPDZ2k），实现真正的混合工作流。

**🔧 技术方法**

技术栈主要包括：秘密共享（算术与布尔）、Beaver 三元组预处理、向量化安全算子、虚拟向量映射、基于 mini‑cluster 的多线程执行、通信层（MPI/套接字）以及 C++ 模板化的协议与数据类型抽象。

**📊 数据集**

使用的真实数据集包括：Kaggle X‑ray 图像（64 张）用于 VGG16 推理、1k 健康记录表、16k SPO₂ 时序序列；实验还使用了 ORQ、TVA、CIFAR‑10、ImageNet、VGG16、AlexNet 等标准基准数据。

**📈 对比分析**

与现有系统比较：在关系型查询上相较 ORQ 提升约 1.5×；在时序分析上优于 TVA 最高 2.7×；在 CNN 推理上对 Pigeon（LAN）和 MP‑SPDZ（恶意多数）分别实现 2–3×、4.7× 的性能提升；混合工作流在半诚实模型下 LAN 1 min、WAN 5 min，恶意多数模型下 LAN 18 min、WAN 1 h。整体展示了统一系统在多种威胁模型下保持竞争力甚至领先的性能。

**⚠️ 局限性**

局限性：目前仅支持少量参与方；缺乏容错与水平扩展；某些操作（如随机洗牌）仍难以充分利用虚拟向量并行化；对恶意多数模型仍成本高昂；GPU 加速仅限于部分工作负载；尚未对更大规模网络和更复杂的安全协议进行全面评估。

---

## 187. Auditing Exposure to Harmful Content on TikTok using Multimodal Language Models: A Cross-National, Age-Stratified Study

**arXiv ID:** 2608.17583 | [PDF](https://arxiv.org/pdf/2608.17583v1)

**作者:** Hamidreza Saffari `[一作]` (Politecnico di Milano), Francesco Pierri `[通讯]` (Politecnico di Milano)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过在法国、意大利、瑞典使用年龄角色的伪账号，采集并利用Gemini 2.5 Flash等多模态大语言模型对36,971条TikTok视频进行自动标注，评估不同国家与年龄段的有害内容暴露。

**💡 创新点**

创新点在于将多模态LLM作为大规模审核器，验证其与人工标注的一致性，并揭示关键词搜索是提升暴露的主要路径以及不同国家间暴露差异。

**🔧 技术方法**

使用Gemini 2.5 Flash（8帧+文本）、Qwen3‑VL‑32B、GPT‑4o‑mini等多模态LLM进行视频内容审核；同时构建关键词搜索与被动滚动两种采集模式。

**📊 数据集**

数据集为36,971条来自12个（国家、年龄）组合的TikTok视频，其中300条用于模型验证，剩余视频用于规模评估。

**📈 对比分析**

与人工注释对比，Gemini 2.5 Flash在8帧+文本条件下达到了κ≈0.42，显示出可接受的一致性；在全数据上，关键词搜索将暴露提升1.5–7.5倍，但高斯模型的误差仍需人工复核。

**⚠️ 局限性**

局限性包括仅有两名本土标注员、验证样本仅300条、LLM与真实人类标注的转移性未知、平台拒绝率偏向高危类别、采样窗口有限且伪账号与真实用户行为差异。

---

## 188. Generalizing and accelerating consistency checking for non-transactional distributed storage systems

**arXiv ID:** 2608.17388 | [PDF](https://arxiv.org/pdf/2608.17388v1)

**作者:** Kotikala Raghav `[一作]` (Indian Institute of Technology Delhi), Abhilash Jindal `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Wing‑Gong线性化检查算法进行泛化，构建可检查多种非事务性一致性保证（如有序序列一致性、K‑界限陈旧一致性、常规序列一致性）以及系统特定一致性保证（通过系统提供的硬/软排序提示），并实现了新的DAG‑based检查器。

**💡 创新点**

创新点包括：① 将线性化检查拆解为单一顺序、值有效性、实时性三项并进一步通用化；② 引入硬/软排序提示的概念，支持系统特定一致性；③ 用DAG顶层排序替代传统的历史修改方法，保持更强的排序约束并显著提升速度；④ 在多种分布式存储系统上验证该方法能发现原有检查器无法捕获的bug。

**🔧 技术方法**

使用的技术主要有：Wing‑Gong算法改写、DAG构造与拓扑排序、硬/软排序提示Oracle、Go语言实现、Jepsen注入故障、并行客户端模拟、时间/内存性能度量。

**📊 数据集**

数据集为从Jepsen自动生成的操作历史，覆盖etcd、Redis Raft、Erwin、Gryff‑RSC、EPaxos、HoliPaxos、Zookeeper等系统，历史长度在数百到数千条操作，客户端并发度从5到125，读写比例多变。

**📈 对比分析**

与Porcupine、Knossos等传统线性化检查器相比，本文实现的检查器在相同历史下平均运行时间下降至0.14–1.87秒（等值系统）或获得高达180×的加速（Erwin等系统）。在高并发、低读率场景下，系统特定一致性检查仍保持在时间预算内，而传统检查器往往超时；内存占用亦低于原实现。通过比对，本文方法显著减少误报（false negative）并定位根因，发现了原有检查器未能检测到的若干bug。

**⚠️ 局限性**

局限性包括：① 需要系统支持硬/软排序提示，某些系统（如Erwin）提供的提示仅为部分顺序，导致DAG构造更复杂；② 对事务性系统不适用；③ 仍为NP‑complete问题，极端并发或极大历史仍可能超时；④ 评估受Jepsen配置限制，仅覆盖部分系统，部分历史由于缺少必要的键/读写操作无法完整测试；⑤ 仅针对共享寄存器类操作，无法直接处理复合读写或多字段键值操作。

---

## 189. When Agents Act on Web3: An Attack-Surface Survey of MCP, Skills, and Tool Calling

**arXiv ID:** 2608.17275 | [PDF](https://arxiv.org/pdf/2608.17275v1)

**作者:** Rabimba Karanjai `[一作]` (University of Houston), Shi `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文对Model Context Protocol（MCP）在Web3环境中的安全威胁进行系统梳理，提出四个加剧威胁的区块链放大器，并构建了攻击面分类表与风险映射矩阵，归纳现有防御手段并指出缺口。

**💡 创新点**

创新点在于：①首次将区块链执行层的不可逆性、签名授权、连续自治和序列级组合四大加剧因子融入安全分析；②用风险映射矩阵将MCP攻击类与Web3影响、放大器、对策及剩余漏洞关联；③提出面向攻击序列、工具描述语义完整性、委托链身份与可逆行动等未解决的研究方向。

**🔧 技术方法**

采用的技术主要是：安全体系结构分层分析、攻击面与生命周期轴的双维分类、CVSS与CVE数据库的威胁映射、现有防御（网关/代理、ETDI签名、MPC/TEE护栏、身份认证）与区块链内置防御（链上注册、TEE证明、质押惩罚）的综合评估。

**📊 数据集**

数据集来源包括：对2,614个MCP实现的安全漏洞统计（路径遍历、代码注入等）、公开CVE列表（如CVE-2025-54136、CVE-2026-30615等）以及对Web3安全事件的行业报告（攻击频次、影响评估）。

**📈 对比分析**

对比方法为：将MCP安全工具与现有安全基准（MCPSecBench、MSB、MCPTox）对照，采用定性严重度评级与CVSS得分作为衡量维度；结果表明，现有防御仅能阻止不到30%的攻击，且对序列级攻击几乎无效。

**⚠️ 局限性**

局限性包括：①仅聚焦使用→风险方向，未覆盖代理生成攻击；②风险映射矩阵使用定性严重度，缺乏量化评估；③CVSS与CVE信息可能滞后或不完整，且区块链安全生态快速演进导致分析结果随时可能被更新。

---

## 190. Multi-Observer Vehicle Localization Case Study with Roadside Radar and Connected Vehicle Sensing

**arXiv ID:** 2608.16966 | [PDF](https://arxiv.org/pdf/2608.16966v1)

**作者:** Aleksi Pippuri `[一作]` (Aalto University), Risto Ojala `[通讯]` (Aalto University)

**通讯引用:** 2493 | [OpenAlex ID](https://openalex.org/A5061093557)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aaccfe5c-6b26-4208-b23c-35331481e142` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个多观测器车辆定位框架，将路侧雷达与连接车辆LiDAR的目标检测进行决策级融合，并在赫尔辛基城市交叉口真实数据上进行跟踪与评估。

**💡 创新点**

创新点在于：①首次在真实环境下验证路侧雷达与CV LiDAR的决策级融合；②比较了序贯EKF（SEKF）与平均EKF（AEKF）两种融合策略；③公开了实验数据与代码，支持可复现性。

**🔧 技术方法**

采用了基于CTRV运动模型的扩展卡尔曼滤波（EKF）实现目标跟踪，使用PointPillars进行LiDAR目标检测，并通过时间空间校准、贪婪关联与轨迹管理完成多源融合。

**📊 数据集**

使用的数据集为在赫尔辛基市区交叉口收集的实测数据，包含路侧雷达、CV LiDAR+GNSS/INS、目标车GNSS/INS轨迹，总计19个验证段。

**📈 对比分析**

方法评估通过与LiDAR-only EKF、Radar-only EKF基线进行RMSE、MAE、匹配率等指标比较。全速率下融合对LiDAR基线几乎无提升，AEKF略优；在低LiDAR更新率（≤3 Hz）时仍保持性能，雷达在长遮挡时可提升轨迹可用性。

**⚠️ 局限性**

局限性包括：仅单一路侧雷达与单一测试场景；雷达观测相对不均衡，难以充分体现其价值；未实现在线校准或自适应不确定性估计；实验数据量有限，缺乏更广泛的交通场景与多车道验证。

---

## 191. MoE-ViE: Mixture of Experts Vision Encoder for Efficient Image and Video Understanding

**arXiv ID:** 2608.17402 | [PDF](https://arxiv.org/pdf/2608.17402v1)

**作者:** Bonan Zhang `[一作]` (Meta), Anuj Kumar `[通讯]` (Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了基于细粒度 Mixture-of-Experts（MoE）的视觉编码器 MoE-ViE，支持高效的图像与视频理解，并通过视频微调保持图像能力。

**💡 创新点**

创新点包括：细粒度 MoE 拓扑结构、无辅助损失的基于幅度的专家负载平衡、针对稀疏计算的 Triton 高效 kernel、以及视频微调时的帧级蒸馏与专家冻结策略。

**🔧 技术方法**

技术手段：MoE top‑k 路由、Sigmoid 门控、共享专家、无辅助损失负载平衡、Grouped GEMM 与激活融合的 Triton kernel、CLIP 对比预训练、帧级知识蒸馏、冻结 MLP 以防止图像知识遗忘。

**📊 数据集**

使用数据集：MetaCLIP 3.5B 图文对、CC12M、TreeOfLife 进行预训练；视频微调使用 K400、UCF101、HMDB、MSR‑VTT 等视频‑文本对；评估使用 ImageNet‑1K、COCO、Flickr30k、VTAB、Fine‑grained 细粒度分类、OCR、视频分类/检索等标准基准。

**📈 对比分析**

与传统稠密 ViT、SOTA MoE 视觉编码器（PEcore、SigLIP、InternVL）及 VLM 对齐模型进行零样本、检索、细粒度、OCR、视频理解与 VLM 对齐评估。MoE‑ViE 在相同活跃参数量下均优于稠密模型；在最大全量模型上可与 1.7 倍更大的稠密模型匹敌，且推理延迟仅为 76% 的 SOTA，显著提升了能力与效率。

**⚠️ 局限性**

局限性：仍依赖 CLIP 对比预训练框架，视频微调过程中需精细平衡以防止图像知识遗忘；大规模 MoE 需要大量专家参数，增加显存占用；高效 kernel 主要针对 NVIDIA GPU，跨平台兼容性待验证；进一步规模化仍需改进专家平衡与同步机制。

---

## 192. DeAR: Decentralized Agentic Reasoning via Capability Grounding and Collaborative Thought Navigation

**arXiv ID:** 2608.17282 | [PDF](https://arxiv.org/pdf/2608.17282v1)

**作者:** Xing Wei `[一作]` (Harbin Engineering University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DeAR 框架，采用去中心化、多智能体协作方式完成复杂多模态推理任务

**💡 创新点**

创新点在于动态能力 grounding、思维图导航与拓扑更新三大机制，实现无中央裁判、动态路由和错误自我纠错

**🔧 技术方法**

使用多模态 LLM 组合、协作倾向矩阵、思维图导航、动态拓扑更新、软最大化与衰减回溯等技术

**📊 数据集**

涵盖 MMMU、MathVista、ChartQA、ScienceQA 以及 NQ、TriviaQA、PopQA、WikiMultihopQA、HotpotQA 等多模态与文本 QA 数据集

**📈 对比分析**

与单模型和多模型基线对比，DeAR 在 9 个基准上平均提升 5‑10% 以上，尤其在多步推理与跨模态任务中显著领先

**⚠️ 局限性**

局限包括对 agent 数量的手动调节、在极长链推理时可能出现性能波动，以及对算力与能耗的进一步优化需求

---

## 193. Dynamic Question Design for Efficient Estimation of Aggregate Human Preferences

**arXiv ID:** 2608.17459 | [PDF](https://arxiv.org/pdf/2608.17459v1)

**作者:** Kazuyoshi Fukuda `[一作]` (Keio University), Masaki Inoue `[通讯]` (Keio University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于贝叶斯估计与粒子滤波的动态问卷问题设计框架，用于高效估计群体偏好；

**💡 创新点**

创新点在于将期望信息增益EIG^eff结合回答概率，提出ε‑greedy策略近似求解组合优化问题，并给出了贝叶斯风险下界及ε‑greedy成功概率的闭式表达；

**🔧 技术方法**

使用的技术包括贝叶斯估计、粒子滤波、期望信息增益、ε‑greedy策略、Gumbel分布假设以及van Trees不等式；

**📊 数据集**

实验采用仿真数据，设置n=7个选择项、T=100、2000粒子、120个候选问题，并未使用真实问卷数据；

**📈 对比分析**

与随机、仅EIG、EIG^eff全搜索四种方法比较，结果显示ε‑greedy在计算时间上更短且误差与全搜索相近，最终估计误差最低；

**⚠️ 局限性**

局限性包括对偏好时间不变和EIG^eff服从Gumbel分布的假设，且仅在仿真环境下验证，缺乏真实问卷实验，粒子滤波在高维情况下可能表现不佳。

---

## 194. Chi-Squared Geometry for Robust Finite-Blocklength Information and Dispersion Analysis

**arXiv ID:** 2608.17305 | [PDF](https://arxiv.org/pdf/2608.17305v1)

**作者:** Hassan Tavakoli `[一作]` (Oregon State University), Bella Bose `[通讯]` (Oregon State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文提出了列式卡方几何框架，用于在不需要计算对数的前提下，对离散无记忆信道（DMC）的互信息、通道色散以及有限块长编码速率进行保真性区间估计。

**💡 创新点**

创新点在于：①利用点级相对偏差参数η，将互信息与卡方互信息的比值逼近1/2并给出三阶修正；②证明卡方互信息与色散之间的双侧近似，误差仅为O(η)；③给出一个完全基于算术运算的认证设计速率，证明其与标准二阶近似的总认证误差为O(η)+O(η/√n)+O(log n/n)。

**🔧 技术方法**

主要技术包括：列式卡方几何、Taylor展开、偏差参数η、两侧误差界定、总方差分解与Popoviciu不等式、以及对BSC、BEC、BAC的解析与数值验证。

**📊 数据集**

该工作为理论性研究，不使用实验或公开数据集；验证通过对BSC、BAC以及一个4×3例子的数值实例。

**📈 对比分析**

与传统需要对数计算的互信息/色散估计相比，本方法仅需加、减、乘、除与平方根，能够在低功耗硬件或估计误差大时提供可计算的安全区间；数值实验表明在η趋近0时区间趋于紧致，误差与理论界限一致。

**⚠️ 局限性**

局限性包括：①仅适用于点级相对偏差小于1/2的DMC；②在对称不确定类下的O(η)误差可能可进一步降低；③对连续信道（如AWGN）尚未直接适用，需要截断技巧；④η的估计需要额外的训练或统计推断。

---

## 195. Offline Multi-Agent Reinforcement Learning with a Physics-Informed World Model for Cooperative Mixed Traffic Control

**arXiv ID:** 2608.17739 | [PDF](https://arxiv.org/pdf/2608.17739v1)

**作者:** Lu Liu `[一作]` (Tongji University), Xi Xiong `[通讯]` (Tongji University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于物理信息的世界模型离线多智能体强化学习框架，用于混合交通中的 CAV 合作控制，避免在线试错和全局状态依赖。

**💡 创新点**

将宏观-微观交通动力学耦合为物理监督，构建概率集成世界模型重构全局状态并量化不确定性，采用惰性回报和不确定性驱动截断进行离线策略学习。

**🔧 技术方法**

物理信息驱动的世界模型、概率集成模型、不确定性估计、多步想象回放、离线多智能体强化学习（改造 MOPO/BCQ 等）

**📊 数据集**

约 1×10⁶ 条离线转移数据，来源于 SUMO 仿真上匝道瓶颈场景

**📈 对比分析**

与无物理监督或经验驱动的离线 RL 方法对比，实验表明物理监督显著提升状态重建和世界模型预测精度，控制性能（如拥堵率下降）明显改善（具体指标可视实验报告）

**⚠️ 局限性**

对真实道路的泛化仍待验证；物理模型假设可能无法覆盖所有交通状况；离线 RL 训练复杂，依赖模型不确定性估计的准确性

---

## 196. FlowShield: cryptocurrency anti-money laundering with transaction semantics parsing and fund flow tracking

**arXiv ID:** 2608.17355 | [PDF](https://arxiv.org/pdf/2608.17355v1)

**作者:** Qishuang Fu `[一作]` (Monash University), Tsz Hon Yuen `[通讯]` (Monash University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了FlowShield框架，能够在多链加密货币交易图中检测洗钱交易并自动生成面向调查员的可读SAR报告。

**💡 创新点**

创新点包括：① 将交易语义解析与局部资金流构造相结合，显式恢复洗钱意图；② 采用文本–结构双向交互融合，将LLM编码的语义文本与GCN编码的图结构统一；③ 构建并公开首个多链洗钱数据集BybitML；④ 将检测结果转化为直观的流图、摘要与红旗，提升可解释性。

**🔧 技术方法**

技术方法：交易语义解析（token流一致性、转发、桥接日志），资金流子图构建（上游、下游、并行），LLM（Llama3 LoRA微调）文本编码，GCN图结构编码，双向交互融合，MLP二分类检测，GPT‑4o生成SAR。

**📊 数据集**

使用的数据集：Bybit-EC、Bybit-BC（BybitML），以及公开的 Upbit、AscendEX 洗钱数据集。

**📈 对比分析**

对比13个代表性基线（启发式、传统机器学习、GNN、图模式挖掘、多模态图模型），在四个数据集上平均F1达98.0%，排名第一；日常表现稳定，鲁棒性好，且检测速度可扩展至数十万笔交易。

**⚠️ 局限性**

局限性：对交易语义解析的准确性敏感，桥接日志或桥接地址列表不完整时易漏检；对UTXO链的优势有限；模型训练依赖大模型和GPU，部署成本高；在极端分布或新型洗钱策略出现时可能需进一步适配。

---

## 197. The politics of postmortem privacy

**arXiv ID:** 2608.16905 | [PDF](https://arxiv.org/pdf/2608.16905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 198. Magnitude-Direction Decoupling for Fast Video Generation with Flow Matching Models

**arXiv ID:** 2608.17695 | [PDF](https://arxiv.org/pdf/2608.17695v1)

**作者:** Haonan Xu `[一作]` (Nanjing University of Science and Technology), Yang Yang `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过把流匹配模型的幅度信息由小模型估计、方向信息由残差重用校准，提出了一种自适应的轻量化采样方法MDD；

**💡 创新点**

创新点在于将幅度与方向分离，利用小模型的幅度估计与残差重用的方向校准相结合，并在CFG中复用幅度以进一步降低计算；

**🔧 技术方法**

技术包括流匹配模型、残差重用、轻量化小模型、方向误差阈值自适应切换、CFG复用、O(N) ODE求解器等；

**📊 数据集**

使用VBench公开的视频生成数据集（文本条件视频），并在Wan2.1、EasyAnimateV5.1两大模型上评测；

**📈 对比分析**

与TeaCache、SRDiffusion等基线对比，MDD在Wan2.1上实现2.95×加速、LPIPS 0.178、SSIM 0.748、PSNR 22.72；在EasyAnimateV5.1上实现1.90×加速、LPIPS 0.150、SSIM 0.755、PSNR 22.66；总体既提升速度又保持甚至提升视觉质量；

**⚠️ 局限性**

局限在于早期去噪阶段仍需使用大模型，阈值选择对速度/质量平衡敏感，且在极高分辨率/长视频场景下仍需进一步验证。

---

## 199. LLM-Only PDDL Domain Repair with Open-Weight Models

**arXiv ID:** 2608.17341 | [PDF](https://arxiv.org/pdf/2608.17341v1)

**作者:** Nader Karimi Bavandpour `[一作]` (Australian National University), Pascal Bercher `[通讯]` (Australian National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了开源大语言模型在PDDL领域模型修复中的表现，并与传统符号方法进行对比。

**💡 创新点**

创新点在于提出了仅依赖LLM预测修复集合的“LLM-only”方案，并系统研究了提示语与推理深度对修复质量的影响。

**🔧 技术方法**

主要技术包括基于提示语的LLM调用、修复集合生成、以及对修复结果进行精确、召回、F1和测试通过率评估。

**📊 数据集**

使用了公开的IPC错误注入基准数据集，该数据集通过随机添加/删除前置条件和效果生成损坏的PDDL域。

**📈 对比分析**

与符号基线相比，最佳LLM模型在F1上提升至0.87（约为0.49），但其平均测试通过率仅为0.82，某些域（如Thoughtful）通过率低至0.06。

**⚠️ 局限性**

局限性包括：1) 修复集合重叠高并不保证所有测试通过；2) 大模型受上下文窗口限制，难以充分利用测试轨迹；3) 训练数据泄漏可能导致模型记忆而非泛化。

---

## 200. LoRIS: LoRaWAN-based IoT Platform for Sustainability Monitoring in Hotels

**arXiv ID:** 2608.17467 | [PDF](https://arxiv.org/pdf/2608.17467v1)

**作者:** Yash Pandey `[一作]` (University of Queensland), Marius Portmann `[通讯]` (University of Queensland)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发并部署了 LoRIS 平台——一种基于 LoRaWAN 的 IoT 系统，用于在酒店环境中实现高分辨率的资源消耗、环境状况和客人行为监测。

**💡 创新点**

创新点在于将低功耗长距离通信与隐私友好、无侵入式部署相结合，实现了在多物业、跨国、长期运行的酒店可持续性监测平台，并首次将这一平台用于多项行为干预实验。

**🔧 技术方法**

技术采用 LoRaWAN（多种厂商的 Class A/C 设备）、AWS IoT Core（网络服务器）、Elastic Cloud（Elasticsearch 存储与可视化）、Auth0（身份认证）以及服务器无状态的 Lambda/Fargate 处理流水线，全部采用端到端 AES‑128/256 加密。

**📊 数据集**

数据集包含 850 传感器、21 个酒店/住宿点、19 种传感器类型，累计 202 百万条记录（约 453 百万个测量值），覆盖 AU915 与 EU868 两个 LoRaWAN 频段。

**📈 对比分析**

通过七项现场实验（食品浪费、能耗、水耗、客人行为干预）验证平台，平均包交付率（PDR）约 81%，SNR 平均 7.5 dB，单一传感器平均电池寿命约 18–20 月，实验结果表明平台可支持高频（1 min）数据采集与行为推断，干预效果可被客观量化。

**⚠️ 局限性**

局限包括对托管云的依赖导致安全与信任集中、对 LoRaWAN 频段与功耗受限的设备选择、部分高功耗传感器（如废弃物称重）需要人工维护、以及在极端干扰或背板中断情况下的丢包与数据缺失需进一步处理。

---

## 201. Force-Based Offset Estimation for Keyed Peg-in-Hole Assembly Using Local Gaussian Process Regression

**arXiv ID:** 2608.17691 | [PDF](https://arxiv.org/pdf/2608.17691v1)

**作者:** Chandra Yuvesh Aubeeluck `[一作]` (Cologne University of Applied Sciences), Florian Zwanzig `[通讯]` (Cologne University of Applied Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种基于手腕力/扭矩的偏移估计方法，用于带钥匙孔的键盘插入，在感知-验证-插入管道中实现。

**💡 创新点**

创新点在于使用局部KNN‑GP混合回归估计连续平面偏移，区分硬碰撞与引导插入两种接触模式，并结合视觉后抓取IBVS校正与均值先验提升精度。

**🔧 技术方法**

技术包括YOLOv8关键点姿态估计、基于图像的IBVS、实时力/扭矩传感、KNN‑GP回归（ARD‑RBF核）、RTDE接口、Kalman滤波以及Python/C++实现。

**📊 数据集**

数据集包含30k张合成图像用于YOLOv8训练，每个工件收集325条力/扭矩样本，并在30次拾取‑放置实验中收集同步数据。

**📈 对比分析**

方法通过与基线（仅姿态估计）对比，并使用留一角度交叉验证与插入成功率评估，GP+KNN+Prior模型平均误差<1 mm，插入成功率从67%提升至87%。

**⚠️ 局限性**

局限性在于只能在训练的力特征范围内推断，无法外推；传感器噪声限制最小可辨识偏移；高角度或大偏移导致模型失效。

---

## 202. Validated Adaptation for Aerial Crowd Monitoring at Mass Gathering Scale: A Deployment Protocol, a Severity Law, and a Diagnostic for Label-Free Drone Crowd Counting, Toward the FIFA World Cup 2034 (Saudi Arabia)

**arXiv ID:** 2608.17625 | [PDF](https://arxiv.org/pdf/2608.17625v1)

**作者:** AlAnoud AllGhayth `[一作]` (Daldata), Jude AlSubaie `[通讯]` (Daldata)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究无人机航拍人群计数的自适应测试时技术，提出基于人口守恒的适配目标和流量风险指示，并在多重实验轨迹下验证其安全部署效果。

**💡 创新点**

①证明人口守恒残差在外观偏移下不产生梯度，适配主要受标准化影响；②通过两个消除假设的消融验证物理先验；③揭示标签自由偏移幅度与准确性损伤无关，提出无条件适配加尾部监测为最优策略；④提供完整的部署协议与校准方案。

**🔧 技术方法**

CSRNet密度回归+RAFT光流；标签自由测试时适配（AdaBN、TENT、物理守恒损失及其组合）；流量风险指示；批归一化偏移门控；配对种子实验、Holm多重校正、Wilcoxon检验等统计方法。

**📊 数据集**

DroneCrowd全分辨率数据集；Track A使用四种合成失真；Track B使用真实域差异与更长帧间隔。

**📈 对比分析**

与Source、AdaBN、TENT、Ours、TENT+Ours五种配置对比；在四种合成失真和五级严重度下，适配恢复30–49%错误；在Track B全分辨率域差异中，适配显著降低密集场景低估；Flux报警在两段完整影片中召回率1.00，平均提前4.4 s；门控策略表明无条件适配优于基于偏移门控。

**⚠️ 局限性**

物理守恒残差在外观偏移下无梯度，需依赖标准化适配；门控仅基于批归一化统计，未考虑更复杂的损伤预测；实验仅在DroneCrowd与合成失真上验证，缺少真实大型赛事视频验证；Flux报警精度低（精确度0.23）且需校准绝对阈值。

---

## 203. Q-Interference: Memory-Efficient Phase-Aware Quantum-Inspired Attention

**arXiv ID:** 2608.17288 | [PDF](https://arxiv.org/pdf/2608.17288v1)

**作者:** Emama Nahid `[一作]` (Kennesaw State University), Honghui Xu `[通讯]` (Kennesaw State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种经典量子启发的注意力机制 Q-Interference，将查询和键的幅度与相位结合，计算相位感知的干涉得分，并给出精确的三角分解实现内存友好。

**💡 创新点**

创新点在于通过相位对齐/相位冲突引入构造/破坏式互作；使用精确三角分解将 O(T²d_h) 的中间张量降为 O(Td_h)；在 GPT 结构中保持其他组件不变，直接替换兼容。

**🔧 技术方法**

采用幅度-相位表示、cos 相位差公式、三角恒等变形、两个标准矩阵乘法实现；训练使用标准自回归语言模型目标。

**📊 数据集**

使用 WikiText-103、TinyStories、pile-10k、small-C4 四个语言建模数据集，另外与 GPT-Neo-125M、OPT-125M 进行对比。

**📈 对比分析**

在同一 GPT 训练管线下与标准 GPT、Q-GPT、预训练模型比较；评估指标包括验证/测试损失、困惑度和峰值 GPU 内存；Q-Interference 在 WikiText-103 上获得最佳内部结果，内存降至约 50%，在其他数据集保持与 GPT 基线相近或略低但显著节省内存；与预训练模型相比，质量略低但内存更优。

**⚠️ 局限性**

限制在于仍无法消除标准注意力的 O(T²) 计算复杂度；某些数据集上质量提升有限；需要额外相位学习参数可能导致收敛更慢；目前仅在小型模型验证，未在大模型或推理阶段验证。

---

## 204. MSEditor: Toward Consistent Multi-Shot Video Editing

**arXiv ID:** 2608.17559 | [PDF](https://arxiv.org/pdf/2608.17559v1)

**作者:** Kunyu Feng `[一作]` (HKUST), Zeyu Wang `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种多镜头视频编辑框架 MSEditor，实现跨镜头的连贯编辑。

**💡 创新点**

通过监督适配器、跨镜头打包和稀疏跨注意力三项技术，使模型在多镜头场景中保持身份一致性与高保真度。

**🔧 技术方法**

基于扩散模型 Wan2.1+监督适配器、跨镜头打包、稀疏跨注意力等技术。

**📊 数据集**

利用现有多视角视频数据重构为多镜头编辑数据集（3400段，每段10视角），并自行构造掩码与结构化提示。

**📈 对比分析**

与 TokenFlow、InsV2V、Ditto、VACE、VideoPainter 等单镜头方法对比，在美学得分、跨镜头一致性、结构一致性等指标上均取得最高分，表现出显著优势。

**⚠️ 局限性**

依赖先前帧编辑作为锚点，受限于多视角数据的可用性与掩码提取的准确性，且对极端镜头变换或极少帧的场景仍存在适配挑战。

---

## 205. TileMix: Tile-Centric Mixed-Precision Attention for LLM Inference Acceleration

**arXiv ID:** 2608.17336 | [PDF](https://arxiv.org/pdf/2608.17336v1)

**作者:** Hanzhi Zhang `[一作]` (University of North Texas), Yunhe Feng `[通讯]` (University of North Texas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于分块的精度路由核（Tile‑Group Precision Routing），在长上下文自注意力中按硬件对齐的分块动态选择FP16或INT8算子，从而在不损失全精度稠密连接的前提下显著提升推理吞吐量；

**💡 创新点**

创新点包括：①将分块视为空间执行单元，实现对每个合法得分块的混合精度路由；②在单一FlashAttention风格的融合核中共享在线softmax状态，允许FP16与INT8路径同时更新；③采用位掩码压缩路由决策，内循环常数时间查找；④支持分组查询、可变长度批处理和INT8 KV缓存；⑤提供可配置的静态路由模板实现可扩展的精度覆盖控制。

**🔧 技术方法**

技术手段包括：FlashAttention式张量分块注意力、块级量化（INT8）、Tensor Core INT8 MMA、FP16运算、在线softmax共享状态、位掩码路由编码与解码、分组因子、分块级比例缩放、共享累计累加。

**📊 数据集**

实验使用LLaMA 3.2 3B、Qwen‑2‑7B、Vicuna‑7B等模型；评测数据集包含LongEval（行级检索）、LV‑Eval（长上下文问答）以及标准预填充基准。

**📈 对比分析**

与FP16、统一INT8（One）、FlashAttention、MInference、FlexPrefill、SageAttention等基线对比，Tile‑Group路由在长上下文检索和问答任务中几乎恢复FP16的准确性，同时在预填充吞吐量上显著超过FP16（高达2–3倍提升），并优于统一INT8。

**⚠️ 局限性**

局限性：仅针对前向推理，适用于A100 GPU；当前实现仅支持FP16/INT8两种数值格式；路由策略为静态模板，缺乏自适应动态路由；未在其他GPU架构或更大模型上验证。

---

## 206. A Constant-Competitive Algorithm for Dynamic Mixture-of-Experts Serving

**arXiv ID:** 2608.16947 | [PDF](https://arxiv.org/pdf/2608.16947v1)

**作者:** Ian D'Ambrosio `[一作]` `[通讯]` (Nth Research Collective), Ian D'Ambrosio (Nth Research Collective)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出并证明了动态Mixture-of-Experts推理中随机化积分原问题的竞争比为常数，并用Lean 4形式化验证整个证明。

**💡 创新点**

首次证明随机化竞争比为Θ(1)，并将积分原问题通过有限正多面体近似、正体追踪定理以及Lazy Threshold Rounding等技术转化为可解问题；同时实现了完整的机器可执行验证。

**🔧 技术方法**

正体追踪(positive‑body chasing)与其定理、迟到阈值取整(Lazy Threshold Rounding)、有限切线包络构造、资源增量与平衡投影以及Lean 4形式化证明框架。

**📊 数据集**

无实际数据集，全部为理论证明与形式化验证。

**📈 对比分析**

通过证明竞争比上界为10·+ (5+2)k+16，显示随机化竞争比为常数；与以往仅给出随机化竞争比的结果相比，提供了更强的常数竞争保证，并通过Lean 4验证保证无误。

**⚠️ 局限性**

仅在固定序列（不可适应对手）模型下成立；未给出最佳常数、确定性竞争比或适应性对手下的竞争比；实现中使用的有限切线网格导致O(mk)约束，虽然可压缩但仍相对复杂。

---

## 207. Why This and Not That? A Collaborative Reflection Approach for Understanding Thought Coverage in Decision Making Support Dialog

**arXiv ID:** 2608.17054 | [PDF](https://arxiv.org/pdf/2608.17054v1)

**作者:** Morita Tarvirdians `[一作]` (Delft University of Technology), Catharine Oertel `[通讯]` (Delft University of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对一款基于ReflectiMate的反思支持代理进行实验，在关键对话策略节点暂停并让用户解释观察到的思维模式，共收集62名参与者的232个协作时刻。

**💡 创新点**

提出协作反思时刻概念并构建九类用户解释词汇表，首次揭示行为模式与用户真实原因之间的“解释鸿沟”，为对话策略的多义性处理提供了系统性框架。

**🔧 技术方法**

采用可解释的覆盖度指标S_k作为对话策略依据，并结合人工编码与多模型LLM辅助的定性分析技术，来捕捉并分类用户的解释。

**📊 数据集**

使用Prolific上62名参与者的自我反思文本与对话日志，涵盖多种生活决策主题，构成实验的主要数据集。

**📈 对比分析**

与代理默认基于覆盖度的策略对比，发现只有45.3%的时刻用户与默认决策一致，但74.4%的选择仍朝向未充分探讨的维度，表明协作时刻显著提升了对话的探索多样性。

**⚠️ 局限性**

实验仅在单一代理与域内验证，词汇表基于自报原因，LLM编码的可靠性有限，且未探讨长程对话效果与跨域泛化。

---

## 208. GeoWeaver: Accurate Long-Sequence 3D Reconstruction via Hierarchical Geometric Assembly

**arXiv ID:** 2608.17389 | [PDF](https://arxiv.org/pdf/2608.17389v1)

**作者:** Tinghao Jiang `[一作]` (Kosmo Research), Zesong Li `[通讯]` (Kosmo Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GeoWeaver框架，将几何先验模型（GPM）与测试时自适应（TTA）相结合，完成长序列3D重建；

**💡 创新点**

创新点在于使用最小重叠（仅一帧）分块预测，再通过分层TTA（顺序初始化→块级Sim(3)对齐→帧级精细优化）实现全局一致性；

**🔧 技术方法**

采用深度-置信度-相机预测的ViT+Transformer先验模型，RoMa密集匹配，Sim(3)对齐，基于CDF的分布式优化；

**📊 数据集**

在Tanks&Temples、Mip-NeRF 360、Virtual KITTI 2和Oxford Spires等四大长序列基准上进行评估；

**📈 对比分析**

与传统SfM/SLAM、单片式feed-forward、chunk-wise、流式以及混合方法相比，GeoWeaver在ATE、RRE和AUC@3°等指标上取得最优或第二优成绩；

**⚠️ 局限性**

局限在于仅适用于静态场景，固定摄像机内参，计算成本高，且对动态、滚动快门或变焦场景尚未适配。

---

## 209. Causal Discovery in Equal Variance Linear Gaussian DAGs via SURE-Tuned Ridge Regression

**arXiv ID:** 2608.17132 | [PDF](https://arxiv.org/pdf/2608.17132v1)

**作者:** Sambit Mishra `[一作]` (University of Southern California), Urbashi Mitra `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种针对等方差线性高斯结构方程模型（SEM）的非迭代闭式估计器 SURE‑Ridge，用于从观测数据中恢复有向无环图（DAG）。

**💡 创新点**

创新点包括：①使用 Stein 的无偏风险估计（SURE）为每个节点自适应地选择岭回归的正则化参数；②将多节点的回归问题解耦为 d 个独立的闭式岭回归；③设计了两阶段自适应阈值化策略快速提取合法 DAG，且不需要逐问题超参数调优。

**🔧 技术方法**

主要技术手段包括：节点级岭回归、SURE 目标函数的闭式推导、矩阵特征分解与投影、数值稳定的多项式无环性函数、二分搜索阈值与后备保留策略。

**📊 数据集**

实验使用合成的等方差线性高斯 SEM 数据集，节点数 d=20、50，平均每图约 2d/5 条边，权重从 {-2,-1,1,2} 采样，噪声方差固定为 1，共进行 100 次 Monte Carlo 试验。

**📈 对比分析**

与 NOTEARS、DAGMA、GBNSL 三个基线在相同数据集上进行对比；在样本量接近或小于节点数的稀疏 regime 下，SURE‑Ridge 的标准化结构 Hamming 距离（SHD）显著低于其它方法；在运行时间上，SURE‑Ridge 也比迭代方法快约两倍，显著优于 GBNSL。

**⚠️ 局限性**

局限性：①仅适用于等方差假设；②在样本量增大到高于约 2–3 倍节点数时，性能略趋于平稳且略逊于 GBNSL；③阈值化步骤可能引入小的偏差；④未在真实世界数据上验证。

---

## 210. Picture the Epsilon: Pursuing Identity-Level Privacy Guarantees for Images

**arXiv ID:** 2608.17147 | [PDF](https://arxiv.org/pdf/2608.17147v1)

**作者:** Arman Zareian Jahromi `[一作]` (Kansas State University), George T. Amariucai `[通讯]` (Kansas State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对四种针对预训练黑盒面部生成器的差分隐私审计方法（GaussMech、KDE‑LR、MMD‑TV、ROC‑HT）进行了统一实验与对比，探讨了各自的假设、超参数与有限样本影响；

**💡 创新点**

创新点在于提出了 MMD‑TV 的纯 DP 解析下界并实现了可复现的保留测试 ROC‑HT 置信度校准协议，同时构建了共享身份对列表与统一评估框架，用以在同一实验设置下评估不同审计方法；

**🔧 技术方法**

主要技术包括高斯机制校准、核密度对数比聚合、MMD 通过总变差链接至纯 DP 下界、以及基于交叉验证的假设检验 ROC 及其置信区间估计；

**📊 数据集**

实验数据集为 FaceFusion 与 InstantID 两款面部生成器生成的嵌入，使用 VGGFace2 与 CelebA 两个身份数据库，并在 ArcFace/FaceNet 两个识别器上提取 512 维嵌入；

**📈 对比分析**

比较方法采用相同的 100 个身份对列表计算中位数、置信区间以及保留测试 ROC‑HT 的置信下界，结果表明所有方法均显示显著身份可区分性，数值尺度差异大且无方法可在高可区分性下被排名；

**⚠️ 局限性**

局限性包括生成器已处于高度可区分状态导致无法区分方法精度、有限样本导致 MMD‑TV 与 ROC‑HT 未得到正式置信下界、依赖特定编码器与数据集、未给出可校准噪声的基准、以及未评估多查询或完整批量发布的隐私保障。

---

## 211. Evaluating the Diversity of AI-Generated Content with Diversity Profiles

**arXiv ID:** 2608.17731 | [PDF](https://arxiv.org/pdf/2608.17731v1)

**作者:** Xiuyuan Hu `[一作]` (Tsinghua University), Xue Liu `[通讯]` (MBZUAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于多尺度参数的“多样性曲线”（diversity profiles）框架，用于取代传统单一标量指标评估生成模型的多样性；

**💡 创新点**

创新点在于将多种现有多样性度量（Energy、Circles、Vendi score、Magnitude）映射为曲线形式，从而揭示不同分辨率与参数选择下的排名稳健性与转折点；

**🔧 技术方法**

主要技术包括：将生成样本映射到高维表示空间，构造距离/相似矩阵，计算参数化的多样性度量，并在参数空间上绘制曲线；

**📊 数据集**

使用的基准数据集包括图像、文本（2WikiMultiHopQA中的问答对）以及分子结构，实验中以Qwen3-8B等模型生成的嵌入为例；

**📈 对比分析**

通过比较不同曲线在整个参数范围内的支配与交叉，展示了单一标量指标难以捕捉的细节；实验表明多样性曲线能更透明、鲁棒地反映生成模型的多样性差异；

**⚠️ 局限性**

局限性包括：曲线解释高度依赖于所选表示、距离/相似函数与参数域；现有的多样性度量本身在公理化方面仍不完善；并未与质量评估结合，缺少完整的生成模型综合评价方案。

---

## 212. Inference-Time Attention Steering for Vision-Language-Action Driving Models

**arXiv ID:** 2608.17095 | [PDF](https://arxiv.org/pdf/2608.17095v1)

**作者:** Darshan Nagendra Prasad `[一作]` (FAU Erlangen-Nürnberg), Knut Graichen `[通讯]` (FAU Erlangen-Nürnberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对Vision‑Language‑Action驾驶模型进行推理时注意力调节，通过在检测到的目标车辆视觉标记上注入有限的加性前softmax偏置，从而在不重新训练的情况下改变模型对关键视觉元素的关注。

**💡 创新点**

提出了可绑定、fail‑open的前hook注入方式；使用受限加性偏置兼容Grouped‑Query注意力并在推理时完成；通过曝光审计和层级消融验证偏置只影响扩散解码器，而不触及语言推理路径。

**🔧 技术方法**

技术包括：YOLO11‑nano目标检测、视觉标记映射到Qwen3‑VL的token网格、前softmax加性偏置、Grouped‑Query注意力与per‑head QK‑Norm、FlashAttention兼容的hook实现、配对种子（paired‑seed）实验、Cliff’s δ统计、层级消融与方向分析。

**📊 数据集**

使用Synthetic PhysicalAI WorldModel‑Synthetic数据集，挑选50个车道变换场景，保证检测到的前方车辆位于±4 m侧向区间内，且在5–40 m范围内。

**📈 对比分析**

通过配对种子比较：在无偏置时两条轨迹完全一致（ADE=0）；随着偏置增大，平均ADE从≈0.1 m升至≈0.17 m，最大侧向偏移可达≈1.4 m。实验显示偏置效果呈单调递增、统计显著（p≪10⁻¹⁰），并在层级消融中显示晚层贡献显著。

**⚠️ 局限性**

局限性包括：仅在合成环境下验证，未测试真实世界转移；仅评估单一模型与车道变换任务，缺乏闭环评估；未通过匹配对照（如背景区域、不同目标）验证偏置对目标身份的特异性；语言推理路径未被真正触发，导致CoC不变的解释仅为曝光不足；方向性推断仅为定性观察，缺乏量化因果证据。

---

## 213. Study-Strategy Clusters from EdNet Logs Track Engagement, Not Mastery

**arXiv ID:** 2608.16963 | [PDF](https://arxiv.org/pdf/2608.16963v1)

**作者:** Qingchuan Lyu `[一作]` (Georgia Institute of Technology), Albert Yang `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 34812 | [OpenAlex ID](https://openalex.org/A5020292820)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过对 EdNet-KT3 ITS 日志中的学习策略特征进行无监督聚类，构建了稳定的层次化学习风格层级，并检验这些早期聚类是否能预测后期的学习参与度和知识掌握。

**💡 创新点**

创新点在于：① 在保持聚类稳定性的同时实现多层次（极端风格与中间风格）分解；② 采用严格的反泄漏时间划分，验证早期行为特征只预测参与度而非知识掌握；③ 将无监督行为聚类与监督知识追踪（SAKT）进行对比，证明行为聚类与知识梯度基本独立。

**🔧 技术方法**

使用的技术包括：基于手工设计的行为特征（动作比例、时长、策略率） → PCA 降维 → K‑means 聚类；对聚类结果进行 ARI 稳定性评估；使用 50/50 计数划分实现 anti‑leakage；监督 KT 模型 SAKT（Transformer‑style）用于对比和性能提升评估。

**📊 数据集**

数据集为公开的 EdNet‑KT3 TOEIC 练习日志，约 298,000 名活跃学习者的数百万条交互记录；仅保留每人至少 50 次提问的用户，进一步筛选至 5,000 次动作以内用于聚类。

**📈 对比分析**

与行为聚类的比较方法包括：使用 ANOVA、Holm 校正检验聚类与后期参与度（late session persistence、late completion）和掌握度（后期无辅助首测准确率）的差异；对 SAKT 的预测性能与仅基于试卷难度的先验（part‑difficulty baseline）进行 AUC 对比，得到约 0.07 的 lift（CI 0.05–0.09）。聚类对掌握度的解释力极低（η² < 0.01），但对参与度显著（η² > 0.1）。

**⚠️ 局限性**

局限性：① 结果仅基于在平台上活跃的用户，难以推广到非活跃或其他学科；② 评估指标为平台内部的“掌握”与“参与”代理，未涉及外部考试或学习成效；③ 早晚划分为 50/50 计数，导致后期样本量不均，准确率估计方差较大；④ 聚类特征为手工设计，可能遗漏更细粒度的时间或主题信息；⑤ SAKT 仅作为对比探针，未达到公开基准的最高性能。

---

## 214. Looking Beyond the Scale: Do Surgical Skill Models Learn Transferable Representations Across Assessment Rubrics?

**arXiv ID:** 2608.17519 | [PDF](https://arxiv.org/pdf/2608.17519v1)

**作者:** Hanna Hoffmann `[一作]` (National Center for Tumor Diseases), Rebecca Hisey `[通讯]` (National Center for Tumor Diseases)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了在不同评估尺度（GOALS与OSATS）之间，基于视觉的外科技能模型是否能学习可迁移的技能表征，并系统评估了JIGSAWS与LASANA数据集的双向迁移效果。

**💡 创新点**

首次通过跨尺度迁移实验揭示了技能表征的可迁移性，并发现模型的主要技能学习负担在任务特定的预测头，而不是骨干网络；同时指出JIGSAWS数据集标注不一致对模型性能的深远影响。

**🔧 技术方法**

采用了端到端监督训练、ASAM自适应锐度优化、VICReg自监督预训练、对比学习（triplet margin）以及Kinetics预训练骨干等技术。

**📊 数据集**

使用了两大公开数据集：JIGSAWS（机器人手术OSATS评分）和LASANA（腹腔镜训练GOALS评分）。

**📈 对比分析**

在LASANA作为目标域时，所有迁移策略（包括自监督、对比学习和Kinetics骨干）均实现CCC约0.77–0.80，接近在LASANA上端到端训练的基线；而在JIGSAWS作为目标域时，无论采用何种预训练，CCC均低于0.20，性能远低于LASANA。

**⚠️ 局限性**

主要局限在于JIGSAWS的评分标注存在不一致性和样本量有限，导致在该域上的迁移与基准训练均表现欠佳；此外，实验仅覆盖两种评估尺度，结果的普适性需要在更多多样化数据集上验证。

---

## 215. DiSCO: Defending text-to-image generation through distribution-guided contrastive prompt optimization

**arXiv ID:** 2608.17067 | [PDF](https://arxiv.org/pdf/2608.17067v1)

**作者:** Tong Zhang `[一作]` (King Abdullah University of Science and Technology), Bernard Ghanem `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DiSCO，一种纯黑盒、prompt级别的对抗鲁棒性提升方法，利用目标模型自身的安全/不安全图像分布进行对比式后缀扩展。

**💡 创新点**

将安全与不安全图像池的对比评分嵌入Beam Search后缀扩展，形成分布引导的prompt优化；无需模型内部访问或再训练，可作为插件部署。

**🔧 技术方法**

分布引导的对比评分、CLIP嵌入、Beam Search自动回归后缀生成、LLM（LLaMA‑3‑8B）生成后缀、图像安全检测器NudeNet/Q16。

**📊 数据集**

I2P基准集合用于构建安全/不安全图像池及生成对抗prompt，评估采用SD v1.4、SD v2.0、Flux、SD 3等模型，攻击使用Ring‑A‑Bell、UnlearnDiffAtk、MMA‑Diffusion、P4D。

**📈 对比分析**

在32个系统‑攻击组合和5个随机种子上与现有防御（SLD‑Max、SAFREE、RECE、ESD）以及未防御模型对比，DiSCO将NudeNet ASR从平均23.6%降至2.4%，Q16 ASR从8.3%降至1.7%，并且在CLIP/ImageReward上保持或提升质量。

**⚠️ 局限性**

需要为每个目标模型生成安全/不安全图像池，推理时会产生额外的生成开销；对极端攻击如UnlearnDiffAtk的提升有限，且对prompt语义漂移仍需进一步控制。

---

## 216. Causal Local States: Scalable Simultaneous Causal Network Inference and Forecasting for Dynamical Systems

**arXiv ID:** 2608.17452 | [PDF](https://arxiv.org/pdf/2608.17452v1)

**作者:** Jonas Braun `[一作]` (Ludwig-Maximilians-Universität München), Christoph Räth `[通讯]` (Ludwig-Maximilians-Universität München)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为CLS（Causal Local States）的框架，能够在仅给定高维无结构时序数据时同时恢复其因果交互网络并进行准确的系统预测。

**💡 创新点**

创新点在于将因果发现的过滤与基于模型的包装器（wrapper）相结合，利用预测误差作为邻域选择准则，实现对每个变量本地邻域的自适应推断，并在不需要预先已知网络结构的情况下实现可扩展的并行预测。

**🔧 技术方法**

技术主要包括：1）因果过滤器（如转移熵TE或收敛交叉映射CCM）用于生成候选邻域；2）包装器模型（以下一代储备计算NGRC为代表）用于评估候选邻域的预测性能；3）后向剔除（backward elimination）去除冗余邻居；4）局部状态预测与闭环多步递推。

**📊 数据集**

使用的三种基准数据集：
- 由N个非耦合的Lorenz63–Rössler双吸引子组成的6N维系统（N=1至15）。
- 40维Lorenz96环形耦合系统。 
- 120节点英国电网高阶Kuramoto模型。

**📈 对比分析**

与传统方法比较：
- 与全局NGRC相比，CLS在无结构数据上实现了可观的预测性能，尤其在Lorenz96上达到与已知邻接矩阵的NGRC相当的VPT。 
- 在复合Lorenz63–Rössler系统中，CLS能成功分离独立吸引子并实现精确短期预测，而单一NGRC无法。 
- 在英国电网模型中，使用域信息增强的Kuramoto‑NGRC包装器时，CLS恢复的邻接矩阵与真网络几乎一致，预测误差与真网络相当；使用Sine‑NGRC则性能显著下降。 
- 综上，CLS在可解释性与预测精度两方面均优于或匹配传统单一模型。

**⚠️ 局限性**

局限性包括：
- 仅恢复因果邻域而非完整时序特征选择； 
- 对包装器与过滤器的超参数（d_max、α1、α2）敏感，需要手工设置。 
- 对高维时序仍存在计算负担，尤其是包装器评估多次一阶预测。 
- 目前仅在合成数据上验证，尚未在真实工业或自然系统数据中测试。

---

## 217. Heterogeneity-Aware Deep Learning for Tumour Classification from Multiparametric MRI

**arXiv ID:** 2608.17254 | [PDF](https://arxiv.org/pdf/2608.17254v1)

**作者:** Yue Xia `[一作]` (University of Sydney), Jinman Kim `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了 Heterogeneity-Aware Deep Learning Classification（HA‑DLC）框架，结合无监督子区域生成、跨患者子区域对齐和双流特征提取，以 mp‑MRI 对肿瘤进行分类。

**💡 创新点**

创新点在于：①通过聚类自动生成肿瘤子区域并使用跨患者软对齐（CPSA）实现子区域标签的跨样本一致性；②双流结构同时学习局部异质特征和全局上下文；③整个过程无需人工子区域标注，端到端联合优化。

**🔧 技术方法**

采用的技术包括 k‑means/Differentiable Feature Clustering（DFC）生成子区域，CPSA 对齐网络，3D residual U‑Net（HFE）提取局部特征，UniFormer（GFE）提取全局特征，soft segmentation 与分类的交叉熵损失，以及 AdamW + cosine 学习率调度。

**📊 数据集**

使用了 LLD‑MMRI2023（肝脏病变 498 例，8 条序列）和 BraTS‑RC（脑胶质瘤 585 例，4 条序列）两个公共 mp‑MRI 数据集，分别用于肿瘤类型分类和 MGMT 甲基化预测。

**📈 对比分析**

与多种基线方法（radiomics + XGBoost、ResNet‑50/121、EfficientNet‑B7、UniFormer、ViG、BraTS 先前冠军）进行 5‑折交叉验证比较。HA‑DLC 在 LLD‑M 上准确率 0.841、F1–Kappa 0.819；在 BraTS‑RC 上 AUC 0.686，均优于所有对手。

**⚠️ 局限性**

局限性包括：①子区域质量受初始聚类参数影响，需人工调参；②缺乏病理或放射基因组学验证，子区域的生物学意义不明确；③仅在两数据集内部验证，尚未在多中心或其他肿瘤类型中进行外部验证。

---

## 218. Automating Parent Selection Configuration in Genetic Programming with Agentic AI

**arXiv ID:** 2608.17172 | [PDF](https://arxiv.org/pdf/2608.17172v1)

**作者:** Jose Guadalupe Hernandez `[一作]` (Cedars-Sinai Medical Center), Jason H. Moore `[通讯]` (Cedars-Sinai Medical Center)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于代理式大语言模型（Agentic LLM）和检索增强生成（RAG）的框架，用于自动识别并实现遗传程序（GP）中的父亲选择算法；

**💡 创新点**

创新点在于首次将代理式AI与RAG相结合，能够在不依赖固定实现的情况下自动生成符合领域知识的、可直接执行的父亲选择代码，且通过实验验证其性能可与人工设计的ϵ‑lexicase相媲美；

**🔧 技术方法**

技术上使用了四种大语言模型（gpt-4o、gpt-4.1 mini、gpt-5 mini、gpt-oss 20b），配合LangChain实现代理推理、工具调用和状态管理，并利用Chroma向量数据库进行文献检索；

**📊 数据集**

数据集方面，作者选取了六个UCI机器学习库中的符号回归任务（Airfoil、Concrete、Energy Cooling、Energy Heating、Housing、Yacht），每个任务使用70%训练、15%验证、15%测试的数据划分；

**📈 对比分析**

比较方法：在实验一中对四种LLM和三种配置（完整代理+RAG、无RAG、纯LLM）进行消融；在实验二中将最优配置（5 mini–AR）与固定的锦标赛选择和半动态MADϵ‑lexicase进行对比；结果显示5 mini–AR在大多数任务上与ϵ‑lexicase相当，且在多数任务上显著优于锦标赛选择；

**⚠️ 局限性**

局限性包括：对单一GP组件（父亲选择）的评估，未涵盖代表性更广的GP任务和其他组件；所用RAG语料库规模小且手工挑选，可能限制模型知识覆盖；生成的代码在某些复制中表现不稳定，缺乏自动验证与迭代改进机制；

---

## 219. Embodied-Navigator: Point, Think, Memorize, and Align for Efficient Navigation

**arXiv ID:** 2608.17512 | [PDF](https://arxiv.org/pdf/2608.17512v1)

**作者:** Hongyan Feng `[一作]` (Zhejiang University), Xuhong Zhang `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 TAMP-Nav 框架，结合 Pixel‑to‑3D 行动空间、选择性推理与 Anchor‑Trajectory 记忆机制，以及两级 GRPO 优化，提升视觉语言导航的几何理解、推理效率和长期记忆管理。

**💡 创新点**

创新点包括：① Pixel‑to‑3D 指针式动作，将 2D 视觉信息直接映射到 3D 域；② 关键节点挖掘与结构化 CoT 生成，支持稀疏推理；③ Anchor‑Trajectory 记忆与空间‑时序指示器（STI）实现对长序列的压缩存储与定位；④ 两级 GRPO，融合全局轨迹奖励与局部步骤奖励，并采用退火引导采样，兼顾性能与推理成本。

**🔧 技术方法**

技术手段：大规模视觉语言模型（Qwen2.5‑VL‑7B、Gemini 2.5 Flash 等）、RoPE 位置编码、2D‑3D 投影、SLAM 控制器、两级 Group‑Relative Policy Optimization (GRPO)、选择性推理触发器、Anchor‑Trajectory 记忆结构、深度噪声鲁棒性评估。

**📊 数据集**

数据集：MultiNav‑CoT（90k 轨迹，含稀疏 CoT 注解），VLN‑CE（R2R‑CE 与 RxR‑CE）用于验证；在真实机器人上进行的室内外任务评测（共 300 次有效试验）。

**📈 对比分析**

与 DualVLN、StreamVLN、NavFoM 等 SOTA 方法对比，TAMP‑Nav 在 R2R‑CE Val‑Unseen 上取得 SR 66.2%、SPL 58.8%，在长轨迹上 SR 49.8%，在真实机器人上 SR 60%。仅用 90k 轨迹（700k 交互），推理平均 16.6 s/任务，显著低于对比模型的 30‑40 s。

**⚠️ 局限性**

局限性：依赖准确的深度感知与 SLAM；在高噪声或动态环境下性能可能显著下降；关键节点挖掘依赖视觉与语义相似度，可能忽略视觉不显著但重要的决策点；STI 仅编码平面位姿，无法区分多楼层；固定 CoT 密度阈值在不同任务中需重新调参；目前无法在线 RL，需从仿真转移到真实世界。

---

## 220. OOD Detection for EEG-based Machine Learning in High-Risk Environments

**arXiv ID:** 2608.17620 | [PDF](https://arxiv.org/pdf/2608.17620v1)

**作者:** Philipp Bomatter `[一作]` (University of Edinburgh), Henry Gouk `[通讯]` (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于EEG的OOD检测基准，系统评估了判别式和生成式方法，并设计了可调幅度的扰动来模拟真实的分布漂移；进一步分析了OOD检测对临床下游预测任务的影响，并将OOD检测与模型不确定性结合形成更安全的判别体系。

**💡 创新点**

创新点在于①首次为EEG构建可控制的扰动生成OOD样本，并将生成式方法引入EEG OOD检测；②系统区分了真正的OOD检测与模型不确定性两大功能；③提出了联合OOD与不确定性判别的决策框架，提高了临床安全性。

**🔧 技术方法**

技术上采用了基于流匹配的UNet生成模型和TCN判别模型；OOD检测方法包括MSP、Energy、ODIN、ASH、Log-likelihood、Typicality、DoSE、SITN；使用等距回归校准MSP概率，并在决策图中实现两种判别方式的组合。

**📊 数据集**

实验数据集为TUAB（EEG正常性预测）和CAUEEG（认知障碍诊断），共计两千余名受试者。

**📈 对比分析**

通过对不同扰动与严重度下的AUROC评估，发现生成式方法（尤其SITN）明显优于判别式方法；在下游分类任务中，使用SITN或MSP拒绝异常样本可显著提升准确率，组合方法进一步提升。

**⚠️ 局限性**

局限性包括：扰动生成的OOD样本仅覆盖有限类型的分布漂移，未涵盖全部真实临床变异；仅使用两份EEG数据集，缺乏跨域泛化验证；阈值设定依赖训练集，可能对不同任务敏感；未探讨更复杂的生成模型或多任务学习对OOD检测的影响。

---

## 221. Teach and Grow: An Agent-Centered Architecture for General Robot Learning

**arXiv ID:** 2608.17209 | [PDF](https://arxiv.org/pdf/2608.17209v1)

**作者:** Chang Nie `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种称为 Teach-and-Grow 的机器人学习架构，利用稀疏的教学演示生成可复用的闭环“技能块”，agent 在执行时根据物理反馈动态重构计划，并维护技能库与经验记忆，实现局部能力的持续增长。

**💡 创新点**

创新点包括：① 将演示转换为可复用技能块而非仅复制轨迹；② agent 在执行时通过闭环反馈动态调整任务路由，支持局部故障诊断和修复；③ 设计了可持续成长的经验记忆与技能库结构；④ 提出了 Teach‑and‑Grow 规模律，量化经验积累对未来任务误差和教学负担的影响。

**🔧 技术方法**

技术手段：预训练的 VLA/世界动作模型、语义任务理解、段落对齐与共通策略合成、闭环执行与效果判定、经验记忆结构化存储、技能库管理、以及最终的 fast‑policy distillation；agent 通过工具选择和动态计划重构来组织机器人行为。

**📊 数据集**

主要使用 LIBERO 基准数据集进行系统评估，并在内部进行针对性实验（如演示转块、反馈驱动执行、局部库成长等）来验证设计。

**📈 对比分析**

在 LIBERO 上与现有 VLA/世界动作模型对比，达到 state‑of‑the‑art 的任务成功率；通过控制实验验证技能块的可复用性、回归测试与局部成长能力，证明在不重新训练全局策略的前提下能持续提升性能。

**⚠️ 局限性**

局限性：依赖预训练模型与人类教学，调试和执行时间较长；技能块生成与验证仍需手工或半自动化支持；在极端稀缺数据或高度动态环境下的适应性尚待进一步验证。

---

## 222. Task Specialization Fine-Tuning for Contextual Reinforcement Learning

**arXiv ID:** 2608.17180 | [PDF](https://arxiv.org/pdf/2608.17180v1)

**作者:** Jianan Zhou `[一作]` (Nanyang Technological University), Cathy Wu `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种在线预算分配框架TSFT，用预训练的单一策略进行多任务细化，通过对每个子任务区域分配有限预算来实现上下文强化学习的样本高效覆盖；

**💡 创新点**

创新点在于将细化过程建模为基于模型的整数线性规划预算分配问题，首次引入周期性模型重估与ILP求解相结合的在线调度方法；

**🔧 技术方法**

主要技术包括：指数型参数化性能模型拟合、整数线性规划（ILP）求解、在线循环预算分配、以及多任务强化学习与大规模语言模型微调；

**📊 数据集**

实验数据集涵盖：组合优化（CVRP、CVRPTW）、连续控制（CartPole、Ant、Meta‑World MT50）、大语言模型微调（Qwen3‑4B‑Base在DAPO‑17K、MATH‑500、GSM8K、CodeContests+等九个推理任务）；

**📈 对比分析**

与多种基线（Oracle、Oracle‑Warmup、Pretrained、MTL、Random、Uniform、Adaptive、LinUCB）对比，TSFT在任务覆盖率上通常比非Oracle基线高出20–50%，且在多数设置下逼近Oracle‑Warmup；

**⚠️ 局限性**

局限包括：1）代理目标与原目标可能不完全对齐；2）全量评估在大规模上下文空间或多策略时计算成本高；3）未对任务分组进行优化；4）指数模型对非单调微调动态的适应性有限。

---

## 223. There is No Theoretical Curse of Multilinguality For Embedding Space Structure

**arXiv ID:** 2608.17088 | [PDF](https://arxiv.org/pdf/2608.17088v1)

**作者:** Niyati Bafna `[一作]` (Johns Hopkins University), David Yarowsky `[通讯]` (Johns Hopkins University)

**通讯引用:** 12480 | [OpenAlex ID](https://openalex.org/A5015876016)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探讨多语言嵌入空间在语言覆盖扩展时是否会出现“多语言灾难”，既给出了理论证明（维度只需对数增长），又在小型 Transformer 4‑layer 编码器上做了实证评估

**💡 创新点**

提出“完美多语言性”的两个本征条件（单语结构一致性与跨语对齐），并证明维度需求仅为 Θ(log L)；首次在理论上驳斥了嵌入空间结构的“多语言灾难”，同时揭示实证灾难与训练/评估配置相关

**🔧 技术方法**

理论分析基于向量直接和（direct sum）构造与球面点阵包装；实证评估使用 4‑layer Transformer + 预掩蔽语言建模，配合多种训练/采样配置、token 计数策略和跨语评测

**📊 数据集**

使用公开语料库：单语数据来源于大规模通用语料（类似 Common Crawl / Wikipedia），多语并行数据来自常用评测集（如 XNLI、MLQA 等）以及英文参考嵌入空间（如 GloVe/fastText）

**📈 对比分析**

对比方法包括 k‑NN 近邻重叠率、跨语对齐率、token‑level MLM 损失；实验设置 8 种配置（计算、采样、评估语言集），发现 4/8 配置出现性能下降（“受灾”），其余 4/8 在足够 token 计数或均衡采样下保持稳定甚至提升

**⚠️ 局限性**

局限性：(1) 完美多语言性假设过于理想，未考虑语言特有概念；(2) 仅在小型 4‑layer 编码器上验证，未探究大模型或解码器架构；(3) 实证实验未覆盖全部可能的 token 预算/语料分布，结果可能因数据偏差而异

---

## 224. Hierarchical Data Selection via Manifold Coverage and Sparse Feature Coverage in LLM Post-training

**arXiv ID:** 2608.16927 | [PDF](https://arxiv.org/pdf/2608.16927v1)

**作者:** Peng Sun `[一作]` (Nanjing University), Tianfan Fu `[通讯]` (Nanjing University)

**通讯引用:** 3764 | [OpenAlex ID](https://openalex.org/A5003226543)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MASS方法，用层次化覆盖策略进行监督微调数据子集选择。

**💡 创新点**

将数据选择视为粗到细的层次覆盖问题，结合稠密与稀疏自编码器以及质量信号实现更高覆盖与质量平衡。

**🔧 技术方法**

使用稠密自编码器学习低维主流形坐标进行粗分组，TopK稀疏自编码器提取细粒度监督特征，并加入外部质量评分。

**📊 数据集**

在Vision-Flan和LLaVA-CoT两大多模指令/推理数据集上进行实验。

**📈 对比分析**

与随机、XMAS、COINCIDE等9个基线在5%、10%、15%预算下对比，MASS在各预算下均取得最高或接近全数据的ARP，部分场景超越全数据。

**⚠️ 局限性**

依赖外部嵌入与质量模型，且仅针对已存在任务分布的候选池，无法弥补缺失任务类型或监督模式。

---

## 225. Memory Is Communication: The Frontier Between Remembering and Signaling

**arXiv ID:** 2608.17053 | [PDF](https://arxiv.org/pdf/2608.17053v1)

**作者:** Yashar Talebirad `[一作]` (University of Alberta), Osmar R. Zaiane `[通讯]` (University of Alberta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究有限资源下代理人如何分配记忆与通信预算，提出记忆-信号前沿理论并在参考游戏中进行初步验证

**💡 创新点**

首次将信息理论中的记忆与通信率作为可调度资源，定义记忆-信号前沿并假设历史效用可预测所需消息量

**🔧 技术方法**

使用率失真理论、Wyner‑Ziv编码、贝叶斯最优解码器框架以及大规模语言模型Gemma‑4‑31B进行对话式实验

**📊 数据集**

在自定义的四形状参考游戏中，目标序列采用重复与循环两种可预测性过程，实验共40轮，使用随机种子和不同p值

**📈 对比分析**

通过测量在不同预测概率p下达到85%准确率所需的最小符号数L_min来评估性能；重复序列L_min随p递减，循环序列L_min随p递增

**⚠️ 局限性**

实验受限于种子数少（每批3个）、仅使用一种模型家族、交互轮次有限，且难以分离历史贡献，尚未在多任务与分布式环境中验证

---

## 226. Advancing Health Equity through Multi-Level Fairness in Health Informatics

**arXiv ID:** 2608.16902 | [PDF](https://arxiv.org/pdf/2608.16902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 227. Adaptive Incentive Design in Dynamic Principal-Agent Problem via Kernelized Bandits

**arXiv ID:** 2608.17614 | [PDF](https://arxiv.org/pdf/2608.17614v1)

**作者:** Arghya Mallick `[一作]` (Delft University of Technology), Peyman Mohajerin Esfahani `[通讯]` (University of Toronto)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了动态委托-代理问题，提出了一种新的随机效用模型，以解决现有文献中代理效用的确定性假设所带来的瓶颈。通过引入随机性，恢复了委托人的期望效用的连续性，并将问题形式化为一个结构化的多臂赌博机问题。

**💡 创新点**

创新点在于引入了随机效用模型，解决了委托人期望效用的不连续性问题，并提出了一种基于神经网络（Arcsin）核的算法，能够在动态环境中有效设计激励合同。

**🔧 技术方法**

使用了神经网络（Arcsin）核来捕捉效用景观的非平稳和S型几何特征，并将问题建模为一个结构化的多臂赌博机问题。

**📊 数据集**

在实际应用中，使用了车辆到电网（V2G）激励设计问题作为数据集，证明了该问题与动态委托-代理问题的等价性，并展示了其在经济性能上的优越性。

**📈 对比分析**

与现有方法相比，提出的算法在累积遗憾界限上达到了𝒪(√(T)(log T)^(m+1))的高概率界限，显著优于之前的最佳已知界限，且消除了对代理动作数量的指数依赖。

**⚠️ 局限性**

限制在于算法的计算复杂度为𝒪(T^3)，在长时间部署中可能面临可扩展性问题。此外，当前框架仍依赖于有限的代理动作空间，未来研究可以扩展到连续的代理动作空间。

---

## 228. Terrain-Aware Local Path Planning with Global DEM Data Integration for Autonomous UGV Navigation

**arXiv ID:** 2608.17038 | [PDF](https://arxiv.org/pdf/2608.17038v1)

**作者:** Devender Singh `[一作]` (Memorial University of Newfoundland), Matthew Hamilton `[通讯]` (Memorial University of Newfoundland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种混合导航框架，将低分辨率 DEM 与实时 LiDAR 感知相结合，用于无人地面车辆（UGV）在复杂户外地形中的路径规划与局部修正。

**💡 创新点**

创新点在于：① 将全局 DEM 基于 A* 的路径规划与局部 LiDAR 分割、坡度估计、动态阈值和分布式障碍物避让算法无缝集成；② 采用自适应坡度阈值和概率分布决策提升安全与效率的平衡；③ 通过滑动窗口统计实现对坡度变化的实时监测与急转弯决策。

**🔧 技术方法**

核心技术：A* 全局路径规划；RANSAC + GPF 地面分割；滑动窗口坡度估计与动态阈值；分布式障碍物检测与偏向决策；ROS+Gazebo 仿真；使用 PID 风格的路径偏差纠正。

**📊 数据集**

数据集与环境：SRTM 1 Arc-Second DEM（30 m 分辨率）用于生成全局成本图；Blender 生成的自定义地形（坑洼、陡坡、平坦区）用于仿真；Compusult Ltd. 的 Nanuk UGV URDF 模型用于物理仿真。

**📈 对比分析**

对比方法：与基线直线路径（仅基于 DEM 的 A*，无局部修正）比较。评估指标包括：障碍物避免率、平均坡度、到达时间、路径偏差。实验结果显示：避免率 95% vs 0%，平均坡度 2.7° vs 8°，到达时间 77 s vs 碰撞停滞，路径偏差 4.25 m vs 0 m。

**⚠️ 局限性**

限制：① 低分辨率 DEM 缺乏细节，局部细节需通过 LiDAR 补偿；② 仅在仿真环境验证，真实世界动态障碍、传感器噪声及极端斜坡的鲁棒性尚未测试；③ 参数（坡度阈值、权重）需人工调优，缺乏自适应学习机制。

---

## 229. CAS-FD: Contact-Aware Temporal Sampling for Single-View Foul vs Dive Recognition

**arXiv ID:** 2608.17060 | [PDF](https://arxiv.org/pdf/2608.17060v1)

**作者:** Md. Jahidul Islam `[一作]` (Premier University), Md. Tamim Hossain `[通讯]` (Premier University)

**通讯引用:** 520 | [OpenAlex ID](https://openalex.org/A5043705674)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建并公开了600段平衡的单视角足球犯规/假摔数据集，并提供了端到端的检测-采样-分类流水线

**💡 创新点**

提出了基于YOLO的接触感知时序采样与优先裁剪策略，首次在单视角设置下显著提升犯规/假摔识别性能

**🔧 技术方法**

使用VideoMAE‑Base预训练模型，YOLOv8检测，移动平均平滑、接触信号加权、非均匀帧采样与自适应Crop，配合AdamW+RandAugment等训练技巧

**📊 数据集**

600条5秒长的单视角比赛视频（300条犯规、300条假摔），来自Premier University、SoccerNet和Stop Diving等公开来源

**📈 对比分析**

在保留的100条测试集上，接触感知采样+自适应Crop模型达到86.0%准确率、宏F1 0.860，比统一采样高出12个百分点，并在多种随机种子下保持一致

**⚠️ 局限性**

主要局限包括依赖YOLO检测导致的帧误检/裁剪失效、数据量相对有限、仅二分类、缺乏多摄像角度以及标注仅由作者团队完成

---

## 230. What Aggregate Scores Miss: Measuring Item-Level Regressions in Commercial LLM API Migrations

**arXiv ID:** 2608.17719 | [PDF](https://arxiv.org/pdf/2608.17719v1)

**作者:** Xiaonan Xu `[一作]` (Georgia Institute of Technology), Wenjing Wu `[通讯]` (University of Colorado Boulder)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在GPT‑5.4至5.6的API升级过程中，作者对900个公开基准项进行每项50次重复采样，统计每个项目在不同模型版本间的准确率变化，并将可靠提升、可靠退化、实用等价与不确定性分类；随后与置换零分布校准，以量化聚合分数隐藏的双向项目级差异；还比较了IFBench严格与宽松评分的差距，揭示格式合规对迁移影响的具体表现。

**💡 创新点**

首次系统性将迁移兼容性拆解为项目级可靠改进/退化，并通过置换检验校准消除采样噪声，揭示聚合提升/下降掩盖的项目级冲突；同时比较严格与宽松评分差距，说明评估容错度对迁移结论的影响。

**🔧 技术方法**

采用重复采样（K=50）+ Fisher精确检验+Benjamini–Hochberg多重检验控制+实用意义阈值ε=0.2来判定项目级显著变化；对每条迁移边的项目级结果进行置换实验（1,000次）校准可靠改进/退化比例；使用IFBench官方严格/宽松验证器比较评分差距。

**📊 数据集**

三类公开基准：SuperGPQA（500项，知识类）、Omni‑MATH hard（100项，奥林匹克数学）和IFBench（300项，指令跟随），共900项；所有项在每个模型版本下均进行50次独立请求。

**📈 对比分析**

对比三条迁移边（5.4→5.5、5.5→Sol、5.4→Sol）在三大基准上的聚合准确率变化与可靠改进/退化比例；发现聚合提升（最多7.3pp）仍伴随4.4–8.3%可靠退化，聚合下降时仍有6.7–9.0%可靠提升；IFBench严格–宽松评分差距在5.5→Sol迁移时扩大，表明格式合规是主要退化来源；整体可靠改进/退化比例在12–24%之间。

**⚠️ 局限性**

研究仅覆盖GPT‑5.4到5.6的三条升级路径，基准样本被筛选为模型有改进余地，可能不代表所有生产工作负载；置换校准与ε=0.2、K=50的设置限制了对较小改动的检测；结果在其他厂商、任务类型或更高/低采样率下可能不同。

---

## 231. Iterative Grasp Pose Refinement: A Deep Reinforcement Learning Approach for 2D Vision

**arXiv ID:** 2608.17628 | [PDF](https://arxiv.org/pdf/2608.17628v1)

**作者:** Amir Arsalan Nematollahi `[一作]` (University of Tehran), Ahmad Kalhor `[通讯]` (University of Tehran)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于深度强化学习的抓取姿态迭代细化框架，利用二维顶视图的关键点表示，逐步优化抓取参数 (x, y, θ, w) 以提升抓取成功率。

**💡 创新点**

创新点在于将传统几何预标记的失败抓取候选作为起点，使用 DQN 对抓取姿态进行局部迭代微调，而非全局搜索；并通过旋转滑移评分机制将抓取质量量化为奖励。

**🔧 技术方法**

采用深度 Q 网络（DQN）实现强化学习，使用离散动作集对抓取姿态做微调；同时在 CoppeliaSim 里用经验回放和 ε-greedy 探索训练网络。

**📊 数据集**

使用 Dex‑Net 数据集的 300 个三维物体网格，先用几何算法生成初始抓取候选，再在仿真环境中对 38 个初始失败的物体进行细化。

**📈 对比分析**

通过在仿真中对失败候选执行 DQN 迭代，所有 38 个物体最终获得成功抓取，成功率达到 100%；在真实 Delta 并联机器人上验证了细化抓取在 sim‑to‑real 的可转移性。

**⚠️ 局限性**

局限性包括：依赖二维顶视图和几何预标记；对未见物体的泛化能力尚待提升；训练样本规模有限，且在复杂多变环境中的鲁棒性待进一步验证。

---

## 232. AerialYield-B2D: A Greenhouse Blueberry Dataset with Five-Stage Ripeness Masks and Fruit Counts

**arXiv ID:** 2608.16973 | [PDF](https://arxiv.org/pdf/2608.16973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 233. Continuity-Driven Representation Learning for Industrial Defect Detection

**arXiv ID:** 2608.17362 | [PDF](https://arxiv.org/pdf/2608.17362v1)

**作者:** Minjong Kim `[一作]` (Chung-Ang University), Changwon Lim `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于连续性驱动的特征正则化框架，在不改动检测器结构的前提下，利用正常区域的重复结构为工业缺陷检测提供稠密辅助监督。

**💡 创新点**

设计了多连续性损失（结合1D序列预测和2D中心遮挡预测）和差分损失（对邻域特征的一阶二阶差异进行约束），并通过基于框的区域加权只对正常区域施加正则化，保持缺陷边界不被平滑。

**🔧 技术方法**

在YOLO系列、MambaYOLO和DETR等检测器的中间特征上添加轻量级投影和预测模块，使用L1/余弦距离作为特征差异度量，利用稀疏框标签构造权重图。

**📊 数据集**

在两套工业缺陷数据集（工业金属表面、MEA膜电极装配）以及公开的NEU-DET钢表面缺陷数据集上进行评估。

**📈 对比分析**

与相同检测器的原始训练基线以及在全量、75%、50%、25%训练数据下进行对比，结果显示多连续性和差分损失分别在mAP@0.5提升1–7个百分点、mAP@0.5:0.95提升2–6个百分点，尤其在数据稀缺场景下提升显著。

**⚠️ 局限性**

在正常边界或光照导致的强边缘处也会产生高连续性误差，导致正则化过度平滑或误判，且目前仅针对基于框标签的监督，未考虑更复杂的几何结构或多模态输入。

---

## 234. Structural Plan-to-Model Conversion with Deterministic Geometry and Guarded Agentic Vision-Language Refinement

**arXiv ID:** 2608.17237 | [PDF](https://arxiv.org/pdf/2608.17237v1)

**作者:** Mohammad Talebi-Kalaleh `[一作]` (University of Alberta), Qipei Mei `[通讯]` (University of Alberta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将结构框架平面图（PDF）自动转换为可编辑的有限元模型草稿，避免人工坐标输入与手工重绘；

**💡 创新点**

首次将无任务特定训练的预训练视觉‑语言模型与确定性几何抽取相结合，形成“守门式代理”框架，既保持可审计的几何基础，又利用模型进行语义纠错，且不需任何检测器微调；

**🔧 技术方法**

技术包括：① 确定性向量原语提取与尺度一致性、符号与结构要素规则、拓扑连接；② 代理式视觉‑语言层：预训练 LLM（如 GPT‑4）在受限操作词表下提出改动，经过几何检验、类型约束与单步审计后才被接受；④ 规则化的校准、裁剪、标注解析与多级评估；

**📊 数据集**

自研的 100 张单层框架图生成器产生的基准集（50 张用于调优，50 张分布相同但种子不同的 hold‑out），包含 vector PDF、干净与模糊的 raster 版本以及完整的真值坐标；

**📈 对比分析**

在 hold‑out 集上进行端到端评估：列柱 recall/precision 0.922/0.997，梁 0.886/0.990，墙 1.000/1.000，支撑 1.000/1.000，开口 1.000/0.964；与传统纯规则或纯学习方法对比（本文未给具体数值，但显示相较于现有技术在同分布图纸上显著提升了精确度与可审计性）；

**⚠️ 局限性**

局限性：仅验证于 vector PDF；不涵盖已绘制的独立图纸、图纸质量多样性与不同地区绘图规范；未评估节点连通性、截面/材料准确性、模型求解有效性；代理层的随机性导致结果可变；仍需人工审核以确保工程安全。

---

## 235. Noisy group neurons with synchronous resetting for high-performance spiking neural networks

**arXiv ID:** 2608.17394 | [PDF](https://arxiv.org/pdf/2608.17394v1)

**作者:** Yajie Zhai `[一作]` (Xi'an Jiaotong University), Zigang Huang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种带噪声群组神经元（NGN）模型及其训练方法，以解决SNN的时空信息丢失和梯度不匹配问题。

**💡 创新点**

创新点在于将有限噪声群组编码与同步复位相结合，既提升表示能力又降低训练难度，并利用均值场推导出高效的高斯替代梯度。

**🔧 技术方法**

采用均值场近似、梯度替代学习（STBP）、Gaussian surrogate梯度以及同步复位的群组神经元实现。

**📊 数据集**

在CIFAR-10/100、Tiny-ImageNet以及DVS Gesture、N-Caltech101、CIFAR10-DVS等静态与事件式数据集上进行验证。

**📈 对比分析**

与传统LIF、CLIF、NLIF以及现有SNN方法（STBP-tdBN、TET、IM-LOSS、NSNN、CLIF+TET等）对比，NGN在大多数任务中获得最高或相近的准确率，例如CIFAR-10 97.26%（T=4）和CIFAR10-DVS 87.35%（T=10），同时保持低延迟和可接受的计算成本。

**⚠️ 局限性**

主要限制包括群组大小对计算时间和随机数生成的额外开销，以及在当前GPU平台上训练时间随K增大而显著增长，且需进一步探索硬件实现的随机性利用。

---

## 236. DOW-KE: Anchor-Free Multi-Layer Knowledge Editing via Direct End-to-End Weight Optimization

**arXiv ID:** 2608.16932 | [PDF](https://arxiv.org/pdf/2608.16932v1)

**作者:** Ran Chen `[一作]` (Northwestern Polytechnical University), Wen Jiang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 6864 | [OpenAlex ID](https://openalex.org/A5041289658)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种无锚点的多层知识编辑方法 DOW-KE，直接对模型权重进行联合优化，解决传统两阶段方法的结构误差。

**💡 创新点**

创新点在于完全消除中间锚点，将最终编辑目标作为唯一损失，联合优化所有层权重，并将保留约束嵌入计算图，保证优化对象即部署对象。

**🔧 技术方法**

使用端到端反向传播联合优化多层权重，基于闭式更新结构的矩阵分解，结合键值空间投影、输出侧子空间约束以及位置条件梯度路由。

**📊 数据集**

数据集包括 CounterFact 与 zsRE，实验模型为 Llama-3-8B-Instruct、GPT-J-6B 与 GPT2-XL-1.5B。

**📈 对比分析**

与 ROME、MEMIT、AlphaEdit、BLUE、FE 及 fine‑tune 基线比较，DOW-KE 在大多数模型-数据集组合上获得最高 Score，尤其在 zsRE 上超越 BLUE，显示出更高的效能、普适性与特异性。

**⚠️ 局限性**

限制在于完全消除锚点可能在极大编辑批次或复杂依赖场景下导致收敛速度下降，并需对子空间阈值与输出投影进行细致调参。

---

## 237. Can LLMs Reason in a Legally Meaningful Manner? A Small-scale Study on European Court of Human Rights Cases

**arXiv ID:** 2608.17168 | [PDF](https://arxiv.org/pdf/2608.17168v1)

**作者:** Amogh Raina `[一作]` (University of Copenhagen), Henrik Palmer Olsen `[通讯]` (University of Copenhagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在欧洲人权法院（ECtHR）第10条案例上评估了GPT‑5.4的法律推理与判决预测能力，比较了三种提示策略，并通过人类法学学生和LLM‑Judge三种评估器对模型输出进行细粒度评分。

**💡 创新点**

创新点在于将专家精细化的推理步骤嵌入提示，使用“LLM‑Judge”对法律推理质量进行自动评估，并通过人机对比证明推理质量与预测准确性不相关，从而警示仅用准确率评估法律模型的风险。

**🔧 技术方法**

技术方法包括：多轮提示设计（无专家提示、专家手工推理流程、官方指南导向），大语言模型生成推理与决策，三位人类评标（Krippendorff α、Fleiss κ）以及三款LLM‑Judge（GPT‑5.5、Claude Opus 4.7、DeepSeek V4 Pro）对推理步骤的自动评分。

**📊 数据集**

数据集为30个2025年4月后公布的ECtHR第10条判决，包含事实摘要、相关法律框架、法院推理及判决，供模型输入并用于人工与自动评估对比。

**📈 对比分析**

评估方式对比了人类评标与LLM‑Judge的评分一致性（Spearman ρ≈0.16–0.33），推理步骤出现率、完整度与简洁度均在三种提示中排序为B>C>A；预测准确率在77–82%与“总是判定违反”基线相当，显示推理质量与预测准确性无显著相关。

**⚠️ 局限性**

限制包括仅研究一条法律条文和单一大模型、数据集规模小且可能与模型训练重叠、评标人员为高级法学生缺乏独立专家层级、LLM‑Judge与人类评标的可靠性-有效性差距，且未覆盖其他司法场景或更多模型。

---

## 238. Emotion Across Speech and Faces: Shared Affective Mechanisms in Multimodal Foundation Models

**arXiv ID:** 2608.17102 | [PDF](https://arxiv.org/pdf/2608.17102v1)

**作者:** Xiutian Zhao `[一作]` (Johns Hopkins University), Berrak Sisman `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过对三种多模态基础模型的解码器MLP层进行激活采样，使用对比激活度方法识别出与情绪类别相关的稀疏神经元（声学ESN和视觉ESN），并通过去活化和激活等干预验证其因果作用；进一步分析ESN在层次分布与跨模态重叠，评估其在不同模态之间的可转移性。

**💡 创新点**

首次在多模态基础模型中同时对语音和面部情绪识别进行激活层面的神经元定位，并揭示其在解码器MLP层的稀疏共享结构；提出跨模态干预验证ESN的可转移性，表明情绪处理在不同感知通道上存在部分共享的内部机制。

**🔧 技术方法**

对比激活度（ConAct）筛选稀疏神经元；解码器MLP门控激活的去活化（deactivation）和激活（steering）干预；层级可视化与Jaccard相似度分析；多模态基础模型（Gemma‑4‑12B‑it、MiniCPM‑o‑4.5、Qwen2.5‑Omni‑7B）中的激活提取。

**📊 数据集**

MSP‑Podcast（语音情绪识别）和AffectNet（面部情绪识别），共5种情绪（愤怒、恐惧、快乐、中性、悲伤）。

**📈 对比分析**

通过多选答案问答协议评估未干预模型的加权平均召回（UAR）；在同模态干预中，去活化导致匹配情绪UAR下降、激活导致UAR上升；在跨模态干预中亦出现相似趋势；干预效果相较随机掩码显著且有情绪选择性；整体模型在情绪识别任务上保持与先前报告相近的UAR，干预并未导致整体性能崩溃。

**⚠️ 局限性**

仅覆盖三种MFMs和两种感知通道，未探究更多模态或更大情绪类别；ESN稀疏程度和重叠系数仍较低，表明共享机制有限；分析集中于解码器MLP层，忽略编码器和跨层交互；实验以静态语音与静态面部图像为主，缺乏动态交互场景。

---

## 239. Institution-Specific LLM Prompting Recovers PHI That De-identification Systems and Their Gold Standards Both Miss

**arXiv ID:** 2608.17051 | [PDF](https://arxiv.org/pdf/2608.17051v1)

**作者:** Daniel Palacios `[一作]` (Baylor College of Medicine), Hyun-Hwan Jeong `[通讯]` (Baylor College of Medicine)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在100份儿科肿瘤科电子病历上，评估了8款大型语言模型（LLMs）与两套专门的去标识系统（Stanford TiDE、OpenMed PII）及正则表达式基线，探讨通过提示工程捕获机构特定PHI并调节精度-召回权衡；同时检验多代理与集成架构是否能进一步提升性能；并对标准标注缺口进行专家重新注释；最终构建公开的评估流程。

**💡 创新点**

创新点在于：①用在上下文学习（prompting）而非模型微调方式，明确列举机构特定PHI类别并加入“勿过度去标识”指令，既填补了传统系统的识别盲点，又在单次调用中同时解决召回与精度的折衷；②发现模型性能瓶颈在于规范说明而非推理能力，证明单通道提示即可获得最优F1；③利用多代理/集成进行标注缺口挖掘，验证其对标准审核的价值。

**🔧 技术方法**

使用技术包括：AWS Bedrock API 调用8款LLM（Claude Sonnet 4.6、Opus 4.8、GPT‑oss‑120B、GPT‑oss‑20B、GLM‑5、Kimi K2.5、MiniMax M2.5、DeepSeek V3.2）；正则表达式与spaCy NER基线；Stanford TiDE 与 OpenMed PII 传统模型；三种提示模式（Baseline、Targeted、Precision）；多阶段与多代理架构（双通、Scrubber–Auditor、交叉模型投票等）；Python 评估脚本（基于TRIPOD‑LLM规范）。

**📊 数据集**

数据集：100份儿科肿瘤科英文病历（共5,322个PHI跨度）与49份儿科神经科病历（409个PHI跨度）作为跨科验证；另外对10份高差异病历进行专家重新注释，新增227个PHI跨度，形成增强金标准。

**📈 对比分析**

性能比较采用召回率、精确率、F1以及F2指标；在Baseline提示下，LLM最高F1约0.918（Sonnet 4.6），优于TiDE的0.779；Targeted提示进一步提升召回到0.975，Precision下降；Precision提示在保持召回的同时把精确率提升至0.829，整体F1约0.893；在增强金标准下，Precision提示F1提升至0.907。多代理/集成架构未能超过单通道Precision提示，F1仅在0.906–0.908区间。

**⚠️ 局限性**

局限性包括：①样本来自单一机构，泛化性待验证；②重注释仅针对差异最高的10份病历，存在偏倚；③原始金标准存在缺漏，导致性能评估受影响；④验证集同样来自同一机构，跨机构验证缺失；⑤模型选择受限，未包含所有可能的LLM；⑥未对公平性或子组差异做深入分析；⑦部分LLM（如GPT‑oss‑120B）表现不稳定。

---

## 240. AISA: AI Safety Assistant Framework for Continuous Improvement of Highway Construction

**arXiv ID:** 2608.17184 | [PDF](https://arxiv.org/pdf/2608.17184v1)

**作者:** Mason Smetana `[一作]` (University of Pittsburgh), Lev Khazanovich `[通讯]` (University of Pittsburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文设计并评估了基于本地大型语言模型的AI安全助手框架，利用冻结嵌入和轻量MLP探针实现事故记录的多维分类与质量评分，并通过检索式RAG将历史事故、相关图像及行业文档即时聚合到高速公路施工的JSA报告中。

**💡 创新点**

创新点在于：1) 采用离线可解释的探针模型实现可部署的事故编码；2) 将质量评分与检索式RAG结合，形成可持续改进的循环；3) 在隐私、成本和可审计性方面提供纯本地解决方案。

**🔧 技术方法**

使用的技术包括冻结的文本嵌入模型（nomic-embed-text-v1.5、OpenAI text-embedding-3-large/3-small、Qwen-Embedding-0.6B）、多层感知机探针、熵/确定性评分、留一词重要性分析、以及基于余弦相似度的naïve RAG检索。

**📊 数据集**

数据集包括：OSHA Severe Injury Reports (SIR)、Integrated Management Information System (IMIS) 的死亡记录、PennDOT JSA Manual 517、公开行业文档库（工具箱谈话、标准和设备手册）以及AI生成的事故相关图像。

**📈 对比分析**

性能对比方法是：在SIR测试集上与多模型多数投票进行比较；多分类模型在四个OIICS字段的Acc@1介于0.75-0.84之间，历史事故检索最高MAP为0.277（text-embedding-3-large），行业文档QA最高Acc@1为0.45；二进制字段预测准确率低，表明该任务不适合当前探针。

**⚠️ 局限性**

局限性包括：1) 未对嵌入模型做微调，导致在外部数据库上表现下降；2) 二进制标签预测失效，需重新设计；3) 图像检索仅基于文本相似度，缺乏真正的多模态支持；4) 低粒度OIICS标签难以预测；5) 需要人工评估和调优以验证质量评分和检索召回的实际价值。

---

## 241. MaLViL: Multi-axis Low-rank Vision-LSTM for Medical Image Segmentation

**arXiv ID:** 2608.17635 | [PDF](https://arxiv.org/pdf/2608.17635v1)

**作者:** Afshin Bozorgpour `[一作]` (University of Regensburg), Dorit Merhof `[通讯]` (University of Regensburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出 MaLViL，一种多尺度低秩 Vision‑LSTM 结构，用于医学图像分割。

**💡 创新点**

通过低秩投影结合双向 ViL、尺度感知 SaLViL、跨方向混合 CDM 与统计导向跳跃调制 SGSM，显著降低解码器高分辨率下的计算与内存负担。

**🔧 技术方法**

使用低秩投影、双向 ViL、轴向卷积、旋转视图融合、统计门控跳跃融合、平方激活 FFN 等技术。

**📊 数据集**

在皮肤病变（PH^2、HAM10000、ISIC 2017/2018）、乳腺超声（BUSI）以及多器官 CT（Synapse）等数据集上评估。

**📈 对比分析**

与 CNN、Transformer、状态空间模型及其他 ViL 基础方法对比，MaLViL 在多种任务上实现或接近最先进的 Dice/DSC，且在细尺度解码器阶段将 ViL 内存降低至 83 倍。

**⚠️ 局限性**

仍受限于对极小结构定位的精度、低秩投影参数的选择以及训练时额外重构正则的需求。

---

## 242. Spectral Gradient Orthogonalization Improves Differentially Private Training at Scale

**arXiv ID:** 2608.17415 | [PDF](https://arxiv.org/pdf/2608.17415v1)

**作者:** Sabari Shanmugam `[一作]` (Australian National University), Kerry Taylor `[通讯]` (Australian National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在差分隐私训练中对梯度进行谱正交化的后处理方法，通过极化分解恢复低秩梯度的方向信息，从而提升模型精度。

**💡 创新点**

创新点在于发现并证明梯度的信噪比（SNR）决定了谱正交化的效用，给出基于Wedin界的恢复阈值；并证明该后处理在不增加隐私成本的前提下可显著提升精度。

**🔧 技术方法**

采用DP‑SGD+梯度裁剪+高斯噪声，随后利用Newton‑Schulz迭代实现极化分解得到最优正交矩阵；可选幅度保持变体；结合动量累计、幅度估计和幅度缩放；对低秩梯度进行SNR与谱间隙分析；并与时域滤波（DOPPLER、DiSK）进行组合。

**📊 数据集**

在CIFAR‑10、CIFAR‑100、SVHN、Tiny‑ImageNet、ImageNet‑1k等数据集上，使用WRN‑16‑4、ResNet‑18、WRN‑28‑10、ViT‑Small等网络进行从零训练与微调实验。

**📈 对比分析**

与DP‑SGD、DP‑Adam、DP‑MuON、DP‑MuON‑S、DiSK‑SGD/M、DOPPLER‑SGD/M等方法对比，谱正交化在大批量/高SNR场景下可提升约+20.9%（WRN‑28‑10 B=4096）或+14.9%（ResNet‑18），与时域滤波联合可达50.3%（CIFAR‑10 ε=4）；同时显著降低跨实验方差，微调时可与DP‑Adam相当但内存更低。

**⚠️ 局限性**

局限性包括：在低批量/低SNR条件下会产生负面影响；幅度保持变体在SVHN上出现双峰收敛；仅对全参数训练有效，参数高效微调无显著收益；需进一步研究层级选择、谱阈值自适应和幅度修正。

---

## 243. The Acknowledgment Point Is the System: Durable Policy-Decision Receipts for AI Audit Evidence

**arXiv ID:** 2608.17176 | [PDF](https://arxiv.org/pdf/2608.17176v1)

**作者:** Neeraj Kumar Singh Beshane `[一作]` `[通讯]` (Independent Researcher), Neeraj Kumar Singh Beshane (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计实现并评估了 RuntimeGuard‑AI V2，一个基于签名收据的可持久化 AI 运行时审计记录系统。

**💡 创新点**

引入源绑定的策略决策、明确的同步语义与可验证的 Ed25519 链式 Merkle epoch，显式展示耐久性与延迟的权衡。

**🔧 技术方法**

使用 Rust 语言实现，文件系统写锁、单写者同步、Ed25519 签名、Merkle 树、JSON 编码及系统性能计数等技术。

**📊 数据集**

以固定大小（128、2048、16384 字节）prompt 的模拟 AI 请求为基准，使用单一确定性 regex 策略。

**📈 对比分析**

通过四种同步模式、不同线程数与提示大小的闭环基准，缓冲模式吞吐≈27k r/s，数据/全同步≈242 r/s，平均延迟从微秒级升至数毫秒级。

**⚠️ 局限性**

仅支持确定性单策略，未考虑网络延迟、多主机可用性、关键管理、可信执行或证明模型执行；签名仅证明可信键签署，无法防止签名者分叉或删除记录。

---

## 244. Protocol-Embedded Compliance for Privacy-Preserving, Non-Custodial Digital Payments

**arXiv ID:** 2608.17145 | [PDF](https://arxiv.org/pdf/2608.17145v1)

**作者:** Santiago De Simone `[一作]`, Georgios Samakovitis `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

无法完成总结，因未提供论文具体内容。

**💡 创新点**

N/A

**🔧 技术方法**

N/A

**📊 数据集**

N/A

**📈 对比分析**

N/A

**⚠️ 局限性**

缺乏论文内容导致无法评估方法和性能。

---

## 245. Authorization Before Context: A Model-Neutral Audience Boundary Against Cross-Audience Memory Leakage in Agentic Systems

**arXiv ID:** 2608.17148 | [PDF](https://arxiv.org/pdf/2608.17148v1)

**作者:** Sibo Liu `[一作]` `[通讯]` (Independent Researcher), Sibo Liu (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在个人语言代理中，在将记忆装入提示上下文前进行授权的机制，防止跨受众泄漏和被注入的假记忆被意外暴露。

**💡 创新点**

创新点在于引入“观众成员资格”边界，即每条记忆记录附带录入时的受众集合，并在上下文组装时以“当前观众完全包含于录入受众”为唯一准入规则，实现单一、可证明的、一次性授权边界；同时提供了模型无关的上下文完整性不变式。

**🔧 技术方法**

技术包括：基于受众集合的反单调授权规则；利用通道元数据读取当前观众并在不确定时失效到公开；对三类记忆存储（即时缓冲、长期摘要、知识图谱）统一应用此规则；通过授权轨迹和上下文快照实现审计与可追溯性。

**📊 数据集**

使用完全合成的 Contextual‑Integrity 数据集，随机生成 79 个场景（包括所有者私有、精确共享、方向跨频道、失效闭合、对抗性和毒化等）。

**📈 对比分析**

与无范围的基线（默认导入所有记忆）对比，实验显示在所有 79 个场景中未出现任何未授权上下文；fail‑closed 覆盖率 100%，允许上下文覆盖率 100%，并在每条记忆检索路径上实现完全失效闭合；未给出传统性能指标，侧重安全性证明。

**⚠️ 局限性**

局限性包括：实验基于合成数据，缺乏真实消息验证；未探讨受众扩大（多方共识导致受众合并）的安全风险；未评估系统对延迟、可用性、内存质量、删除和冲突解决等方面的影响；未公开生产系统实现。

---

## 246. Cognitive Graph Intelligence for Adaptive and Robust DDoS Attack Detection in Next Generation Networks

**arXiv ID:** 2608.17352 | [PDF](https://arxiv.org/pdf/2608.17352v1)

**作者:** Mohammad Arif Hossain `[一作]` (Middle Tennessee State University), Nirwan Ansari `[通讯]` (New Jersey Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出GraphGAN框架，将网络流转化为时序k‑NN图并用图卷积网络和生成对抗网络实现DDoS攻击检测。

**💡 创新点**

创新点：① 采用k‑NN构造时序图捕捉流间关系；② 通过图基GAN生成结构化少数类样本解决不平衡；③ 将生成的样本与GCN分类器结合，实现鲁棒且适应性强的检测。

**🔧 技术方法**

技术手段包括滑动窗口时序图构造、k‑NN图构建、图卷积网络（GCN）、图基生成对抗网络（GraphGAN）、数据增强与自监督训练。

**📊 数据集**

使用四大公开数据集：CIC‑IDS‑2017、CIC‑IDS‑2018、UNSW‑NB15 与 ToN‑IoT。

**📈 对比分析**

与CNN‑LSTM、GRU‑BiLSTM、SMOTE、WGAN、VAE‑GAN、E‑GraphSAGE、BS‑GAT 等方法对比，GraphGAN 在所有四个数据集上取得约 94–95% 的准确率，明显优于竞争者。

**⚠️ 局限性**

局限性：依赖特征相似性假设导致图构建受限；生成模型训练不稳定；缺乏对加密流、多向量攻击及更广泛网络场景的泛化验证。

---

## 247. B-Spline Embedded Structure Learning for 3D Tooth Segmentation

**arXiv ID:** 2608.17291 | [PDF](https://arxiv.org/pdf/2608.17291v1)

**作者:** Xianghan Wei `[一作]` (Zhejiang University), Haihua Zhu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于B‑样条嵌入的结构学习框架，利用牙弓的连续轨迹生成点级结构嵌入，并通过结构感知动态分类器（SADC）构建案例自适应的牙齿原型，实现三维牙齿分割；

**💡 创新点**

创新点在于将牙弓的全局顺序信息通过B‑样条曲线转化为连续结构嵌入，并将其用于动态原型生成和关系注意力建模，取代传统静态分类器，实现对牙齿缺失、拥挤等复杂情况的鲁棒自适应；

**🔧 技术方法**

核心技术包括B‑样条轨迹拟合、点级结构嵌入预测、基于结构门控的原型聚合、关系感知自注意力网络以及双重蒸馏训练；

**📊 数据集**

使用公开的3DTeethSeg22基准数据集（900名受试者，共1800张高分辨率牙弓点云），按官方划分训练1200张、测试600张；

**📈 对比分析**

与多种SOTA方法（如Point Transformer V3、ToothGroupNet、3DTeethSAM等）对比，本文在六项评估指标上均取得领先，整体准确率达96.01%、牙齿mIoU 92.62%、Dice 94.90%、B‑IoU 71.36%、TIR 97.50%、TIR_=1 92.67%，并在计算效率上显著快，单张扫描推理仅0.84s；

**⚠️ 局限性**

局限在于仍需对高分辨率点云进行下采样并通过后处理恢复细节，且框架依赖当前3D网络吞吐量，未来可探索高通量网络与更高效的预训练模型融合来实现端到端精细分割。

---

## 248. Trusted Workflow Relays:Cross-Tenant Email Abuse and Composable Red Team Initial-Access Primitives in Multi-Tenant Clouds

**arXiv ID:** 2608.17361 | [PDF](https://arxiv.org/pdf/2608.17361v1)

**作者:** Priyank Nigam `[一作]` `[通讯]` (Microsoft), Priyank Nigam (Microsoft)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对多租户云服务中的通知工作流进行安全评估，发现并修复了三类跨租户、内容与授权缺失的受信任工作流中继漏洞。

**💡 创新点**

首次在云通知场景中提出“受信任工作流中继”正式定义，并通过案例、渲染测试矩阵和 ATT&CK 对映实现了漏洞模型化与防御建议的系统化。

**🔧 技术方法**

采用黑盒安全评估、HTTP 请求篡改、渲染测试、形式化推导与 MITRE ATT&CK 对照等技术进行漏洞发现与分析。

**📊 数据集**

使用内部测试账号、租户、对象和受控邮箱作为评估样本，未使用公开数据集；仅包含受控的请求和邮件内容。

**📈 对比分析**

对比方法为定性评估：通过修改请求字段观察是否产生交付与渲染效果；未进行量化性能测评，结果仅说明漏洞存在与修复效果。

**⚠️ 局限性**

局限性包括：案例数量有限、已修复导致现行系统不可复现、缺乏跨供应商送达率与点击率数据、未覆盖真实攻击场景和大规模统计。

---

## 249. RADmesh: Remesh-Aware Mesh Deformation

**arXiv ID:** 2608.17182 | [PDF](https://arxiv.org/pdf/2608.17182v1)

**作者:** Nam Anh Dinh `[一作]` (University of Chicago), Rana Hanocka `[通讯]` (University of Chicago)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过在视觉损失引导下对网格进行可变形与离散重建，提出一种能在保留三角质量的同时实现局部生长与全局细节化的网格生成方法。

**💡 创新点**

创新点在于：①采用可优化的旋转+尺度顶点量化（vertex-based deformation quantity）而非原始顶点位置；②在优化循环中周期性进行等距重建（isotropic remeshing），并将优化状态按重建的重心坐标进行插值；③结合粗细尺度重建调度，使大尺度变形在保持三角形质量的同时能够逐步细化。

**🔧 技术方法**

主要技术包括：差分渲染与基于扩散模型的 Score Distillation Sampling（CSD）进行视觉监督；基于 Geometry in Style 的可微 ARAP 全局求解；Botsch & Kobbelt 等离散等距重建算法；以及 Adam 优化器的状态插值。

**📊 数据集**

实验数据来源于多种公开网格模型（如人形、动物、机械等）与自定义文本提示，评估集包含 64 对形状‑提示对，用于计算 CLIP/VQA 语义匹配、三角质量和面数等指标。

**📈 对比分析**

与 MagicClay、Instant3dit、Geometry in Style、MeshUp 等基线对比，方法在 CLIP、VQA、三角形质量（FQ）上均取得最高分，并在面数上保持相对较低，证明了在保持可视化质量的同时实现了更高效的三角化。

**⚠️ 局限性**

局限性：① 训练耗时较长（局部生长 85–90 分钟，整体细化 70–75 分钟）；② 依赖 Poisson 细分，必须保证网格为流形；③ 当前重建仅支持保形改动，未涵盖非流形或拓扑改造。

---

## 250. SleuthTalk: Supporting Historical Photo Identification with Private Workspaces for Collective Sensemaking and Deliberation

**arXiv ID:** 2608.17297 | [PDF](https://arxiv.org/pdf/2608.17297v1)

**作者:** Liling Yuan `[一作]` (Microsoft), Kurt Luther `[通讯]` (Virginia Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了SleuthTalk—a private collaborative workspace，用于支持历史照片识别中的结构化讨论与集体决策。

**💡 创新点**

创新点在于将传统大规模开放式众包转向受信任的微型群体，并提供专门的短列表管理、面部特征细化比较、投票与讨论等工具，使得识别过程既安全又透明。

**🔧 技术方法**

技术包括Microsoft Cognitive Services Face API用于面部相似度搜索，基于Web的交互式界面实现短列表与投票，自动完成的特征名称匹配以及可视化的投票与比较图表。

**📊 数据集**

使用数据集为Civil War Photo Sleuth平台内的60,000+历史战争照片数据库，并在实验中选取了若干未知照片进行识别。

**📈 对比分析**

通过对比SleuthTalk与大型公开Facebook群组的两周任务，结果显示SleuthTalk的用户在自评信心、讨论深度、投票一致性上均优于Facebook；实验中共收集到152条面部特征比较、125条身份投票，最终多项任务在SleuthTalk中实现了更高的共识率。

**⚠️ 局限性**

局限性包括样本规模有限（仅6名参与者）、实验时长短暂、未深入评估算法偏差与群体同质化风险，且对更大规模真实世界使用情况的长期效果尚未验证。

---

## 251. Leveraging existing sparse point annotations for benthic imagery dense segmentation

**arXiv ID:** 2608.17561 | [PDF](https://arxiv.org/pdf/2608.17561v1)

**作者:** Cesar Borja `[一作]` (Universidad de Zaragoza), Ana C. Murillo `[通讯]` (Universidad de Zaragoza)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用稀疏专家点注释，结合SAM2与DINOv3，对海底影像进行自适应点标记传播并生成高质量的密集语义分割掩膜，支持大规模生态监测。

**💡 创新点**

提出前置剪枝（去除不适用点）和后置修剪（裁剪跨类泄漏）两阶段机制，基于每图类原型对点和掩膜可靠性进行自校准，显著提升伪标签质量。

**🔧 技术方法**

使用视觉基础模型SAM2作为视觉提示器，DINOv3提取特征并构建类原型，采用余弦相似度计算剪枝/修剪阈值，并在SegFormer上训练语义分割模型。

**📊 数据集**

在三组数据集上验证：新公开的24类海底监测数据集（313训练/78测试），Coralscapes（39类）和UCSD Mosaics（34类），均采用稀疏点采样模拟弱监督场景。

**📈 对比分析**

与“全部保留”“随机丢弃”“噪声鲁棒损失(GCE)”等基线相比，在噪声更高的数据集上实现mIoU提升约5-8个百分点、mPA提升约3-5个百分点；在mIoU上可达约44-53%不等。

**⚠️ 局限性**

依赖每图类原型导致类样本稀少时可靠性低、相似类别难以区分；当点注释稠密或图像质量极佳时，剪枝/修剪对性能提升有限。

---

## 252. tinyDSM: A Framework for Skill Modeling and Development for Resource-Constrained Millirobots

**arXiv ID:** 2608.17596 | [PDF](https://arxiv.org/pdf/2608.17596v1)

**作者:** Markus D. Kobelrausch `[一作]` (TU Wien), Axel Jantsch `[通讯]` (TU Wien)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 tinyDSM 框架，使体积仅 36 cm³、仅用 RP2040 微控制器的毫厘机器人在约 15 分钟内从原子运动模式学习到复杂几何行为，实现了资源受限环境下的自我驱动技能开发。

**💡 创新点**

核心创新在于：① 将极小的先验知识与分层知识图谱相结合，② 引入由新颖度、进展度和难度三因素构成的内在动机模型，③ 通过运动学推理与自适应 fitness 评估实现无监督技能掌握，并在微控制器上实现极低内存占用的学习与调度。

**🔧 技术方法**

采用的技术包括：强化学习（Simulated Annealing、Q‑learning）、自定义小型知识图谱、运动学推理器、fitness‑based 评估、轻量化 C++ 实现、嵌入式记忆管理（自定义 buddy allocator）以及 TinyML 量化/裁剪概念。

**📊 数据集**

实验数据集包括：1）真实硬件实验——3D 打印四轮毫厘机器人，配备 RP2040、加速度计、外部摄像头+ArUco 位姿；2）基于 Pygame 的物理仿真环境，用于系统化评估不同学习算法与内在动机参数化。

**📈 对比分析**

方法比较通过：① 将 SA、Q‑learning 与随机策略在同一知识图谱与内在动机下进行多次独立运行；② 计算平均技能 fitness、选择熵与最大忽视；③ 以 IM Score（结合 fitness、熵、忽视）进行量化对比。结果表明基线配置获得最高 IM Score，学习速度快、覆盖度高；随机策略表现差；内在动机偏向探索会导致技能被长期忽视。

**⚠️ 局限性**

局限性包括：① 仅测试有限的运动学技能与四轮平台，缺乏更复杂感知与规划任务；② 依赖外部摄像头位姿传感，未实现完全自主定位；③ 记忆碎片化导致学习模块占用显著内存；④ 所用 RL 算法（SA、Q‑learning）在更高维度动作空间中可扩展性有限；⑤ 目前未尝试轻量化神经网络或更丰富的自适应策略。

---

## 253. Backward through Time, Algebraically

**arXiv ID:** 2608.17087 | [PDF](https://arxiv.org/pdf/2608.17087v1)

**作者:** Konstantinos Kogkalidis `[一作]` `[通讯]`, Konstantinos Kogkalidis

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种可推导、可微分的线性时序逻辑（LTL）评估引擎，并实现了多种代数（布尔、模糊、参数化、鲁棒等），通过 PyTorch 在神经网络产生的数值轨迹上评估公式，形成可作为训练信号的梯度。

**💡 创新点**

创新点包括：①将 LTL 评估抽象为可插拔代数，既可保持符号推理又兼容梯度传播；②利用代数的结合律与生成器坐标化，将复杂窗口计算转化为一次张量运算；③提出 Mellowmax 代数，以软平均的形式既保证梯度密集、平滑，又让判决结果与持续时间无关，解决传统模糊代数在梯度消失、选择性和饱和性问题。

**🔧 技术方法**

使用的技术主要有：PyTorch 自动微分、张量化实现（scan、fold、span）、代数抽象类与多态（Algebra、Lifted、State）、生成器坐标化（Archimedean family）、自定义 monoid 状态机（BoltzmannState、MellowmaxState），以及通过核函数（cummin/cummax、cumprod、logcumsumexp）加速评估。

**📊 数据集**

本文未给出具体真实数据集，评估基于合成轨迹和随机张量，主要关注算法的数值性质与梯度行为。

**📈 对比分析**

通过对 20+ 代数的自动化审计（检验结合律、单项式、吸收律等）以及对评估速度与梯度稀疏度的实测，发现传统代数在梯度稀疏或消失时表现差；Mellowmax 既提供密集、平滑的梯度，又保持判决的时间不变性；在大规模张量上，利用生成器坐标化与 monoid 组合，评估速度提升约 3–5 倍。

**⚠️ 局限性**

限制包括：①部分代数在代数层面不满足标准格律，导致无法完全一致地定义求值；②Mellowmax 等状态机代数在载荷与梯度之间仍存在非平衡（如不满足单元性、乘子非结合）; ②评估仍依赖于显式循环或张量拆分，对极长轨迹的内存占用较大；③缺乏在真实任务（如控制、强化学习）上的大规模验证。

---

## 254. DEPT: Document Embedding Preservation Tuning for Unified Query Expansion and Retrieval

**arXiv ID:** 2608.17632 | [PDF](https://arxiv.org/pdf/2608.17632v1)

**作者:** Jingyuan Wang `[一作]` (Beihang University), Yanzhao Zhang `[通讯]` (Beihang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种统一的解码器仅LLM模型，既可生成查询扩展文本，又可对扩展后的查询和候选文档进行稠密检索，训练时使用端到端检索损失。

**💡 创新点**

核心创新是Document Embedding Preservation Tuning (DEPT)，通过文档嵌入保持损失、固定白化变换、直通（straight‑through）生成梯度和在线硬负样本挖掘，解决了统一模型中查询扩展和文档表示互相干扰的“移动目标”问题。

**🔧 技术方法**

采用解码器仅LLM（Qwen3‑4B‑Instruct‑2507、LLaMA‑3.2‑3B‑Instruct）、LoRA参数微调、InfoNCE对比损失、固定白化矩阵、直通梯度（ST）和FAISS在线硬负采样等技术。

**📊 数据集**

训练数据为由ECHO检索混合构成的多源数据集（ELI5、FEVER、HotpotQA、MS MARCO、Natural Questions、SQuAD、TriviaQA等），评估在BEIR基准的五个任务（SciFact、ArguAna、NFCorpus、FiQA、SCIDOCS）上进行。

**📈 对比分析**

与训练‑free方法（HyDE、Query2Doc）、独立训练方法（ExpandR）以及分阶段统一基线相比，DEPT在所有任务上实现了平均nDCG@10的最优或第二优成绩；DEPT‑K在仅生成约9个关键词扩展时仍保持竞争力。

**⚠️ 局限性**

局限性在于：必须保持文档嵌入相对不变，限制了文档表示的进一步优化；依赖于预构建的缓存索引，若文档分布或大小发生显著变化，可能需要重新构建索引；对极大规模语料库的实时检索仍需进一步验证。

---

## 255. GUPO: Gradient Uncertainty-aware Policy Optimization for Post-Training Large Language Models

**arXiv ID:** 2608.17411 | [PDF](https://arxiv.org/pdf/2608.17411v1)

**作者:** Peizheng Guo `[一作]` (Institute of Software Chinese Academy of Sciences), Wenwen Qiang `[通讯]` (Institute of Software Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 GRPO mini‑batch 中不同查询产生的组梯度冲突，并提出基于梯度不确定性的聚合方法 GUPO，用以获得更可靠的更新方向。

**💡 创新点**

创新点在于将组梯度建模为贝叶斯分布，通过 Dirichlet 证据理论估计不确定性，并在梯度聚合时按不确定性加权，从而减轻冲突导致的更新低效。

**🔧 技术方法**

使用贝叶斯参数后验近似、蒙特卡洛采样推断梯度分布、主体逻辑/证据理论与 Dirichlet 不确定性估计，以及加权梯度聚合技术。

**📊 数据集**

在 AIME 2024/2025、AMC 2023、MATH500、MinervaMATH、GSM8K 等数学推理基准上进行实验。

**📈 对比分析**

与 GRPO、Length Penalty、ReST‑MCTS、GVPO、Dr.GRPO、GCPO、MRT 等后训练方法对比，GUPO 在平均 Pass@1 上提升约 3%–5%（不同模型规模下）并在高冲突 mini‑batch 上显著提升性能。

**⚠️ 局限性**

主要局限在于仅近似后验于最后一层，参数 s 与 η 的选取对效果敏感，并且增加了额外的采样与计算成本。

---

## 256. "It just kind of shows that I went somewhere": An Exploratory Study of Fitness Data Sharing

**arXiv ID:** 2608.17014 | [PDF](https://arxiv.org/pdf/2608.17014v1)

**作者:** Mara Solen `[一作]` (University of British Columbia), Tamara Munzner `[通讯]` (University of British Columbia)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对18名健身数据分享者进行半结构化访谈，构建代码本并分析生成关于健身数据分享（FDS）的新特征和设计启示。

**💡 创新点**

提出了可视化作为证明、个体性表达和文化规范遵从这三种新特征，并给出针对非路线运动的可视化、指标选择和易用性设计建议。

**🔧 技术方法**

采用构造主义扎根理论方法、访谈、手工编码与NVivo分析等质性研究技术。

**📊 数据集**

使用18名受访者的访谈记录（约193,000字）及其转录文本作为数据集。

**📈 对比分析**

通过对照先前研究验证与补充，未进行量化对比或性能测评，主要基于编码一致性与饱和度评估研究可靠性。

**⚠️ 局限性**

样本局限于温哥华健身社群，平台聚焦Strava与Instagram，未涉及全球多样化用户与其他社交平台，且未探讨观众视角。

---

## 257. Unified Message Model for Heterogeneous Serial Data Exchange Protocols

**arXiv ID:** 2608.17642 | [PDF](https://arxiv.org/pdf/2608.17642v1)

**作者:** Viktor Sinitsyn `[一作]` (Technical University of Munich), Florian Holzapfel `[通讯]` (Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一种统一、协议无关的序列化消息模型，并给出了对应的建模方法和工业工具实现。

**💡 创新点**

将数据类型、容器和完整消息结构拆分为三层模型，支持对异构串行协议的完整、可自动化描述；引入可配置的消息类型约束和用户表示，兼顾工程实践与自动化生成。

**🔧 技术方法**

采用正式化的数据类型系统、位级容器描述和消息结构建模；在工具实现中使用模型驱动、约束求解与可配置映射技术（实现基于dBricks MBSE环境）。

**📊 数据集**

通过典型协议案例（ARINC 429/825、CAN BFF/EFF、UART、MAVLink、UDP/IP）进行模型演示，并在工业环境中对这些协议进行建模与验证。

**📈 对比分析**

与现有描述方法（AUTOSAR、DBC、ASN.1等）对比，证明本模型在统一性、完整性、可自动化生成TL软件和ICD方面具有优势；工具实现能够自动生成符合协议的TL代码，并通过一致性检查验证模型正确性。

**⚠️ 局限性**

模型不覆盖低层传输细节（如位填充、握手协议），对复杂协议的手工定义仍可能繁琐；需要用户自行定义消息类型和用户表示，缺乏统一标准；尚未在大规模工程中全面评估性能与可扩展性。

---

## 258. Dijkstra as an Oracle for Online Stochastic Shortest Path Navigation with Provable Guarantees

**arXiv ID:** 2608.17703 | [PDF](https://arxiv.org/pdf/2608.17703v1)

**作者:** Mansur M. Arief `[一作]` (King Fahd University of Petroleum and Minerals), Ahmad Alfan Alfian Irfan `[通讯]` (Universitas Muhammadiyah Yogyakarta)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在线学习框架 DORA，利用 Dijkstra 算法与可观测的“降价”边权来求解 stochastic shortest path（SSP）问题，直接在已知地图上学习最优路径。

**💡 创新点**

创新点在于证明 Dijkstra 推导的策略类在满足“降价非负”这一比传统的因果性（causality）更弱的条件时即可包含 SSP 的最优策略；并基于此设计了无需估计转移核、仅需滑移概率的 DORA 算法。

**🔧 技术方法**

使用 Dijkstra 的单次标签设定、降价边权的自洽估计、对冲信念的对数权重（chance-constrained 变体）、冲量衰减更新与双层投影的 Lagrange 上升。

**📊 数据集**

在四个导航基准上评估：仓库导航（20×20 迷宫）、SimpleGridWorld、GeoSteeringMDP、DroneSurveillance；每个基准都基于已知地图的 SSP 转化而来。

**📈 对比分析**

与基线比较：DORA 在 800 次试验中获得与已知转移核的 OVI-K 相近的累计惩罚，仅消耗 1/10–1/20 的规划工作量；同时在学习期间的碰撞次数比无降价的 DORA-0（即 determinize+replan）减少 17 倍；在更大地图上，规划工作量的比值随地图尺寸增大而显著提升。其他基准同样验证了 DORA 的优越性。

**⚠️ 局限性**

限制包括：需要预先知道滑移概率和地图几何；目前仅在离散网格、无非齐次动力学或部分可观测的情形下实验；理论上限仅对内部迭代收敛的变体给出，实际实现的有限迭代缺乏正式证明；在连续状态、非霍莫多动态或不完整观测下的推广仍待研究。

---

## 259. Scalix: Uncertainty-Aware Scale-Consistent Monocular SLAM

**arXiv ID:** 2608.17553 | [PDF](https://arxiv.org/pdf/2608.17553v1)

**作者:** Sebastian Barbas Laina `[一作]`, Stefan Leutenegger `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Scalix，一个实时单目SLAM框架，通过将学习得到的深度与尺度不确定性作为可优化测量，整合到概率因子图中，实现度量尺度的轨迹跟踪。

**💡 创新点**

创新点在于：①将全局尺度作为可优化变量并引入尺度不确定性；②设计分离尺度-深度解耦的预测网络，输出像素级深度不确定性与全局尺度不确定性；③将这些不确定性通过因子图与多视角约束耦合，实现尺度漂移的显著降低。

**🔧 技术方法**

使用技术包括：基于Metric3Dv2的深度网络+不确定性头；概率因子图优化（OKVIS2+滑动窗口、Cauchy鲁棒化）；尺度与深度误差项；边缘化与相对姿态-尺度图；双线程前后端并行；以及多模态输入（RGB仅）。

**📊 数据集**

训练使用ScanNet和Waymo；评估使用KITTI（outdoor）和7‑Scenes（indoor）数据集。

**📈 对比分析**

与CUT3R、DROID‑SLAM+Metric3D等单目方法在Sim(3)和SE(3)对齐下比较；在KITTI上Sim(3)取得最佳ATE，提升37%对比CUT3R；在SE(3)提升40%对比DROID+M3D；在7‑Scenes结合全束调整后ATE分别为7.8cm（Sim(3)）和11.5cm（SE(3)）。

**⚠️ 局限性**

局限性包括：室内深度网络精度不足导致室内性能略逊；在没有尺度约束或连续相机运动极限时仍可能出现漂移；目前仅单目输入，未进一步融合多模态传感器。

---

## 260. MultiSigBERT: Beyond Survival Analysis through Multimodal and Sequential Modeling in Oncology

**arXiv ID:** 2608.16972 | [PDF](https://arxiv.org/pdf/2608.16972v1)

**作者:** Paul Minchella `[一作]` (Université Lumière Lyon 2), Rémi Vaucher `[通讯]` (EPITA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 MultiSigBERT，一个融合临床文本、结构化变量的多模态时间序列生存模型。

**💡 创新点**

创新点在于将文本句向量、结构化变量先对齐后通过路径签名编码，再与 LASSO‑Cox 结合实现可解释且高效的动态风险预测。

**🔧 技术方法**

使用了 OncoBERT 句子嵌入、PCA 降维、路径签名（Signature transform）以及 LASSO 正则化的 Cox 回归。

**📊 数据集**

使用了来自勒昂·贝拉德中心的 2,527 名肿瘤患者 124,049 条临床报告和 4 个结构化变量的真实数据集。

**📈 对比分析**

在 36 个月的里程碑设计下，与 DeepSurv、CoxTime、CTVF 等基线相比，MultiSigBERT 的 C‑index 约 0.743，IBS@3y 0.128，明显优于传统方法。

**⚠️ 局限性**

局限在于仅使用有限的结构化变量，插值可能引入噪声，且未包含静态特征（如年龄、性别、肿瘤类型）。

---

## 261. Brief Announcement: Fair Binding for Hidden-State Authorization in Byzantine SMR

**arXiv ID:** 2608.17349 | [PDF](https://arxiv.org/pdf/2608.17349v1)

**作者:** Arnab Mallick `[一作]` `[通讯]` (Centre for Development of Advanced Computing), Arnab Mallick (Centre for Development of Advanced Computing)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在Byzantine状态机复制（SMR）中，当政策状态隐藏且验证者无法从日志前缀重建时，如何安全地分配隐藏的可消耗资源，并提出一种公平预留/使用协议；

**💡 创新点**

提出了两个互相独立的安全与活性要求：公平提交顺序与绑定有效性，并证明仅公平顺序不足以保证安全与首到达活性；提出的预留/使用协议同时满足两者；

**🔧 技术方法**

利用经过验证的Byzantine SMR、外部有效性谓词、可验证的授权证明（如零知识证明）以及公平排序协议；

**📊 数据集**

论文未使用具体实验数据集，而是以形式化模型与证明为主；

**📈 对比分析**

通过形式化证明验证协议的安全性与首到达活性，没有提供数值性能对比；

**⚠️ 局限性**

局限性在于只处理单一不可再补充的资源，无法扩展到多资源、可再补充余额或部分公平顺序等更复杂场景；

---

## 262. Wuying-Browser-Agent: Real-World Centric Fundamental Long-Horizon Browser Agents

**arXiv ID:** 2608.17319 | [PDF](https://arxiv.org/pdf/2608.17319v1)

**作者:** AIMAE Team `[一作]` (Alibaba Cloud), Likai Zou `[通讯]` (Alibaba Cloud)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套完整的长期浏览任务训练与评估框架，包括结构化浏览器把手、鲁棒性强化学习（RUIC‑SFT 与 DAO‑GRPO）以及专门的中文‑英文长步骤基准 BrowserBench，并基于该框架训练出 Wuying‑Browser‑Agent。

**💡 创新点**

创新点在于：① 通过 Reflection‑Rich 反思恢复数据和 UI‑Specialized 复杂控件数据在课程化 SFT（RUIC‑SFT）中实现对错误恢复与复杂 UI 的系统监督；② 设计 Divergence‑Aware Online GRPO（DAO‑GRPO）在在线 RL 过程中结合潜在奖励塑造、分支敏感信用分配和按步上下文优化，显著提升长路径决策与恢复能力；③ 提出 BrowserBench 作为长步骤双语真实 Web 任务基准，提供可解释的任务标签与严格 Pass@1 评价。

**🔧 技术方法**

技术包括：结构化浏览器把手层（提供验证的工具空间与决策上下文管理）；RUIC‑SFT（多源数据混合与分阶段课程训练）；DAO‑GRPO（潜在奖励塑造、LLM‑驱动的分支识别与分支权重、按步响应优化）；LLM（Qwen3.5 系列）进行策略训练；自监督数据飞轮（实时回滚、自动三段过滤与验证）。

**📊 数据集**

使用的数据集：约 3,000 条基准 SFT 轨迹（含成功、UI、反思三类）；5,000+ 在线回滚轨迹用于 DAO‑GRPO；350 条 BrowserBench 任务（含 191 条中文、159 条英文，平均 37.9 步）；WebVoyager、Online‑Mind2Web 等公开基准；对比模型数据如 Qwen3.5‑397B、OpenWebRL‑8B 等。

**📈 对比分析**

通过 Pass@1 成功率对比评估：在 WebVoyager、Online‑Mind2Web 与 BrowserBench 上，Wuying‑Browser‑Agent‑27B 分别取得 80.6%、66.7%、65.1%（平均 70.8%），超越所有公开模型并接近闭源 GPT‑5/5.5；在 Tau2‑Bench、Claw‑Eval、BFCL‑v4 等通用代理基准同样表现领先，说明在长步骤 Web 任务和通用工具使用两方面都有显著提升。

**⚠️ 局限性**

局限性包括：① 对网络变化的适应性仍需改进，训练数据可能快速过时；② 在线 RL 仍需较长时间收敛，且依赖 LLM 分支识别的准确性；③ 对极长（>100 步）或非英文/中文网站的泛化尚未充分验证；④ 主要关注浏览器任务，迁移到其他平台或多模态交互场景仍需进一步探索。

---

## 263. Orphan risks at the frontier of artificial intelligence: What diverging safety and compliance frameworks reveal about how AI companies choose the risks they prioritize

**arXiv ID:** 2608.16895 | [PDF](https://arxiv.org/pdf/2608.16895v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 264. A Multi-Surface Consistency Audit of Software Citation Metadata

**arXiv ID:** 2608.17159 | [PDF](https://arxiv.org/pdf/2608.17159v1)

**作者:** Pengyin Shan `[一作]` `[通讯]` (University of Illinois Urbana-Champaign), Pengyin Shan (University of Illinois Urbana-Champaign)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对117个科研软件项目在七种机器可读自述表面（CITATION.cff、codemeta.json、.zenodo.json、DOI记录、PyPI/npm注册信息、README引用块等）进行审计，比较并量化它们在标题、作者、版本、年份、DOI、许可等关键字段上的一致性。

**💡 创新点**

首次系统测量同一项目在不同自述表面之间的冲突比例，构建可复现的审计管道并手工验证高精度，提出冲突机制分类（论文与软件混淆、归档滞后、注册库虚假元数据、身份歧义）。

**🔧 技术方法**

利用Python实现的harvest‑normalize‑compare‑report管道；对YAML、JSON‑LD、BibTeX、DOI解析进行标准化；采用模糊匹配（token‑sort >=90）判断字段相似；手工核对验证日志。

**📊 数据集**

基准数据集为117个项目，包含87个高性能计算/量子计算项目（来自先前的DOI锚定语料），以及30个来自JOSS和pyOpenSci的社区审稿项目。

**📈 对比分析**

采用对每对表面字段的四级判定（exact、minor、conflict、missing），通过对505个可比较字段对进行统计。精度为98.5%（338行验证），发现83.9%项目存在至少一字段冲突，平均每项目字段一致率约50%。

**⚠️ 局限性**

仅覆盖GitHub托管项目，PyPI和npm为唯一注册库；README解析为启发式，可能漏检；快照仅在一次获取窗口，随时间可能变动；只检测概念与版本DOI差异，未覆盖所有可能的冲突类型。

---

## 265. Average Distance Approximation for Static Large Graphs

**arXiv ID:** 2608.16916 | [PDF](https://arxiv.org/pdf/2608.16916v1)

**作者:** Kartikey Ahlawat `[一作]` `[通讯]` (Leiden University), Kartikey Ahlawat (Leiden University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在大规模静态无向无权图中估算平均距离的方法，并在 8 个真实图上实验比较随机游走采样、SEF 与 EW 三种方案。

**💡 创新点**

创新点在于系统评估了采样与基于地标的两类方法，证明 EW 算法在精度（误差 ≤0.02%）、计算时间与内存占用上均优于随机游走和 SEF，并指出仅需 100 个节点即可得到高精度估计。

**🔧 技术方法**

采用的技术包括随机游走采样、节点子集采样、Size Estimation Framework (SEF)、Eppstein-Wang (EW) 算法、HyperLogLog 计数器实现内存高效邻居探索，以及 BFS 用于最短路径计算。

**📊 数据集**

使用 Konect 公开数据集，共 8 张图，包含 4 张单部图（Flixster、Skitter、Youtube、Orkut）和 4 张双部图（Flickr、DBLP 及两张未列明的双部图），每张图的节点数从 1.6M 到 3.2M 级别。

**📈 对比分析**

实验方法为多次重复不同节点子集大小（1,5,10,30,40,100,1000）计算平均距离、误差百分比、标准差与运行时间；结果显示 EW 在误差<0.02%、最短运行时间和最小内存占用方面表现最佳，随机游走在样本量不足时误差大且需要 >15% 节点才能达到可接受精度。

**⚠️ 局限性**

实验局限于静态无向无权图，对带权、有向或动态图的适用性未验证；随机游走实现复杂且对不同图结构的鲁棒性未作系统评估。

---

## 266. How smoothing the affinity matrix affects neighborhood preservation in t-SNE

**arXiv ID:** 2608.17190 | [PDF](https://arxiv.org/pdf/2608.17190v1)

**作者:** Shirin Mohebi `[一作]` (Ghent University), Jefrey Lijffijt `[通讯]` (Ghent University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对t‑SNE的相似度矩阵进行平滑处理，研究平滑对邻域保留的影响，并在MNIST、鼠脑皮层与UCI Adult数据集上进行全面实验。

**💡 创新点**

引入γ参数对相似度行进行平滑/锐化，提出有效困惑度概念，展示平滑可在不提升全局困惑度的情况下改善中间尺度邻域保留。

**🔧 技术方法**

使用t‑SNE改进（γ平滑/锐化）、有效困惑度分析、NO@k、AUC等邻域保留度量，并与传统t‑SNE及多尺度相似度构造进行对比。

**📊 数据集**

MNIST手写数字、鼠脑皮层单细胞RNA测序（133个簇）和UCI Adult人口普查表格数据。

**📈 对比分析**

通过NO@k与AUC指标比较不同γ、不同困惑度以及多尺度构造；平滑（γ<1）在中间尺度k（≈10‑100）优于标准t‑SNE，且比单纯提升困惑度更有效；锐化（γ>1）在近邻尺度更优；在全局结构上平滑可提升性能，锐化则削弱。

**⚠️ 局限性**

结果受数据稠密度与结构影响，平滑在高困惑度下可能不如标准t‑SNE；近邻保留对锐化更好，平滑不一定；对不同数据集的最佳γ与ρ需要经验调参，缺乏统一理论。

---

## 267. EMAN: Optimization-Driven Capacity Growth through Path Emergence in Multi-Task Learning

**arXiv ID:** 2608.16930 | [PDF](https://arxiv.org/pdf/2608.16930v1)

**作者:** Chenlei Fang `[一作]` (Northwestern Polytechnical University), Chunjiang Zhao `[通讯]` (National Engineering Research Center for Information Technology in Agriculture)

**通讯引用:** 10836 | [OpenAlex ID](https://openalex.org/A5100604390)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EMAN 框架，从单路径共享开始，利用相位探测和优化证据决策是否在训练过程中动态生成新的独立路径，以满足多任务学习的容量需求。

**💡 创新点**

创新点在于：① 通过相位变量先行发现可行的反对称增长方向；② 采用优化证据认证（OEC）将局部优化信息转化为全局结构决策；③ 在保持共享计算成本的同时，只有在持续且安全的证据出现时才生成第二条路径，真正实现训练过程中自适应的容量扩展。

**🔧 技术方法**

技术主要包括：相位耦合探测（latent relative phase）、零和分裂初始化、OEC 的多指标评估（强度、一致性、下降、稳定性与持续性）以及基于 Hessian-向量积的局部曲率分析。

**📊 数据集**

使用的公开数据集有：Controlled rank benchmark（人工生成的多任务回归），NYUv2（语义分割、深度估计、表面法向预测），以及 PASCAL-Context（多任务语义分割和其它任务）。

**📈 对比分析**

与 Hard-sharing、MeanPaths、AdaShare、Recon 等方法对比，EMAN 在标准容量下提升语义分割和深度估计的指标，整体 G 指标略高；在容量压力下（Capacity-Stress）实现约 70% 的路径训练 FLOPs，且在任务到达和渐进需求场景下，结构释放时间与需求变化同步，性能保持竞争。

**⚠️ 局限性**

局限性包括：① 对中等容量边界的敏感度不足，可能错过轻微瓶颈；② 释放后两条路径仍需并行训练，推理时未降低计算量；③ 证据阈值需要手工设定，可能在不同任务和数据分布下需调优。

---

## 268. Tight Bounds for Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function

**arXiv ID:** 2608.17343 | [PDF](https://arxiv.org/pdf/2608.17343v1)

**作者:** Anh Tuan Nguyen `[一作]` (Carnegie Mellon University), Viet Anh Nguyen `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本论文研究了多维数据驱动超参数调优的统计复杂性，提出了一种基于嵌套块消除的框架，以解决现有方法的理论局限性。

**💡 创新点**

创新点在于通过引入嵌套块消除框架，避免了拓扑过度计数，从而得出了更严格的伪维度界限，并且提出了多重区域下界框架，证明了上界的紧密饱和性。

**🔧 技术方法**

使用了统计学习理论和实代数几何技术，特别是通过分析不变的连通符号单元来改进学习理论的上界。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在多种机器学习问题中适用的训练和验证目标。

**📈 对比分析**

与现有方法相比，提出的框架在多维超参数调优中提供了更严格的伪维度界限，且通过多重区域下界证明了这些上界的紧密性，性能显著提升。

**⚠️ 局限性**

限制在于尽管在训练损失设置中确认了保证的紧密性，但对于一般的双层验证损失设置的基本最小最大下界仍然是一个未解决的问题。

---

## 269. Decomposition Attacks Across Unlinkable Identities: Limits of Stateful Defenses for LLM Services

**arXiv ID:** 2608.17445 | [PDF](https://arxiv.org/pdf/2608.17445v1)

**作者:** Bowen Sun `[一作]` (Johns Hopkins University), Chaowei Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文基于完整的威胁模型，对LLM服务中的分解攻击进行理论推导与实验验证，构建了91项可执行任务与对应的基准数据，评估了多种状态化防御方案的安全-效用前沿；

**💡 创新点**

创新点在于首次给出了在“可链接但无法观察答案聚合”环境下的精确安全-效用前沿；同时通过可执行任务基准与实测数据揭示状态化防御在面对重试与反馈学习时的根本局限；

**🔧 技术方法**

研究主要采用符号理论分析、因果重放实验、能力识别模型以及八种可实现防御（词法、Granite、TwinGate、Brown-style、Cumulative、Direct LLM、Factored Ledger、Structured Operation prototypes）和强化学习自适应攻击；

**📊 数据集**

使用的数据集包含91个可执行分解任务、365个注册操作、11,393个匹配控制请求以及1.5百万背景请求，形成完整的基准测试环境；

**📈 对比分析**

通过比较ASR、匹配控制拒绝率（MCD）与背景拒绝率（BBR）在1% MCD与0.5% BBR限制下的表现，结果显示最优实现防御M5的ASR仅比无防御低约2.4%，其余方案要么无法满足效用约束，要么在重试/学习后ASR迅速回升至100%；

**⚠️ 局限性**

研究局限在于未能观测答案聚合信息、基准任务与控制请求为合成数据且可能与真实流量不符、仅利用请求级特征，未考虑可信输出或身份链接等额外信号，因而结论仅适用于在完全威胁模型下的场景。

---

## 270. Fool's Gold: Defensive Deception Against Safety-Removal Attacks on Open-Weight Models

**arXiv ID:** 2608.17202 | [PDF](https://arxiv.org/pdf/2608.17202v1)

**作者:** Mark Russinovich `[一作]` `[通讯]` (Microsoft Azure), Mark Russinovich (Microsoft Azure)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对开源大语言模型的安全移除攻击（abliteration）构造了一种防御机制——通过在模型权重中植入“欺骗”解码器，使得攻击后模型生成的危险回答表面可信但内容被故意篡改。

**💡 创新点**

创新点在于：①首次将防御性欺骗（decoy hardening）嵌入模型权重层面，主动把攻击收益转化为可识别的错误答案；②利用可微分攻击仿真训练，在攻击状态下引入解码器并通过“refusal pin”和“KL leash”保持原始模型的安全性与功能；③在多种模型与规模上（Qwen、Gemma、GLM 等）公开验证，并与多种攻击变体、RL 反击与 fine‑tune 评估对比。

**🔧 技术方法**

技术包括：可微分的权重编辑仿真（abliteration 模拟）、分层 LoRA 微调、对抗性数据生成（自我生成 decoy 数据集）、对齐与拒绝训练、DPO（Direct Preference Optimization）以及基于评判器的分层否定规则。

**📊 数据集**

数据集主要来自公开的 CBRNE 领域安全基准（如 FORTRESS、AILuminate、BioProBench 等）以及自构造的 decoy 语料库，覆盖化学、生物、放射、核及爆炸等危险操作情境；测试使用未见过的提示和冻结的测试集。

**📈 对比分析**

与传统拒绝强化、防御性重写、RL 对齐等方法对比，Fool's Gold 在被攻击模型中将危险回答的“致命缺陷”比例提升 0.27–0.84，且在多种攻击策略下（oracle‑free 选择、元素共识、RL 反击等）成功率显著下降，说明其在攻击后“拒绝失效”情景下仍能提高验证成本；同时保持原始模型的安全拒绝率与功能在容差预算内。

**⚠️ 局限性**

局限性包括：①仍有 10–50% 的攻击样本保持原始（但无标记）输出，需额外验证；②对部分攻击者（如具备大量标签、跨模型共识或检索增强）仍有限制；③依赖于评判器与手工标签，评估过程中存在主观性；④未覆盖内联 jailbreak、fine‑tune 直至验证后重训练的场景；⑤在较小模型上效果不稳定，易突破门限。

---

## 271. When More Foundation Models Means Less: Diagnosing and Addressing Multi-View Fusion Failure

**arXiv ID:** 2608.17490 | [PDF](https://arxiv.org/pdf/2608.17490v1)

**作者:** Yibo Liu `[一作]` (Beijing University of Posts and Telecommunications), Bowen Jiang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了大规模冻结基础模型视图融合中的视图集组合问题，提出了训练-free的KAGES算法实现视图选择。

**💡 创新点**

创新点在于将多视图学习转化为视图集组合任务，揭示“更多视图不一定更好”，并提出基于标签核对齐的贪婪选择方法KAGES，提供前缀级近似最优保证。

**🔧 技术方法**

使用了中心化核对齐（CKA）作为目标函数、核级加法求解、贪婪迭代以及子模比率条件下的近似理论。

**📊 数据集**

在五个识别范式的图像分类数据集（CIFAR‑100、Oxford Pets、DTD、Country211、GTSRB）以及图像检索和冻结LLM融合任务上进行实验。

**📈 对比分析**

与全融合、随机、Diversity、DPP、子模等基线对比，KAGES在AULC、Acc@k*等指标上均优于基线，逼近oracle，并在低shot/大池/全数据等多场景表现突出。

**⚠️ 局限性**

局限在于归一化CKA目标不一定单调或子模，理论保证依赖这些条件；对生成模型和任务专家范式的融合仍未解决。

---

## 272. Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL

**arXiv ID:** 2608.17253 | [PDF](https://arxiv.org/pdf/2608.17253v1)

**作者:** Yunhao Yang `[一作]` (Johns Hopkins University), Yijiang Li `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无监督标签的多代理强化学习框架Co‑RL，利用不同模型间的互相奖励来提升推理性能；

**💡 创新点**

核心创新在于通过多样化（模型族、规模、输入重写）构建的独立代理来生成交叉奖励，打破自我强化回路，扩大正确收敛基底；

**🔧 技术方法**

使用的技术包括GRPO策略优化、交叉投票伪标签、模型异构和数据重写、以及多代理并行训练；

**📊 数据集**

在文本推理任务上使用MATH、GSM8K、HumanEval等七个基准；在多模态推理上使用MathVision、MathVerse、MathVista、We‑Math四个多模态数学数据集；

**📈 对比分析**

与自监督奖励方法（TTRL、RENT、Intuitor、Co‑Rewarding‑II）以及多代理RL基线（MAPoRL、CoMAS）对比，Co‑RL在所有文本基准平均提升3.0–8.6%，在多模态基准提升2.3–7.2%，甚至在部分场景与有监督的GT‑Reward相当或超越；

**⚠️ 局限性**

限制在于目前仅验证了少数模型族与规模组合，缺乏对代理数目、拓扑结构以及自适应监督机制的系统性研究，且对极端异构模型的泛化与长期训练稳定性仍有待深入。

---

## 273. CUSTOS: Toward Forensic-Ready Zero Trust at the Capture-Containment Boundary

**arXiv ID:** 2608.17068 | [PDF](https://arxiv.org/pdf/2608.17068v1)

**作者:** Avinash Srinivasan `[一作]` (United States Naval Academy), John Paramadilok `[通讯]` (United States Space Command)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向取证的零信任参考架构（Forensic Management Point, FMP），并实现了基于策略触发的分层捕获、身份/策略关联重构、遥测编排以及零信任控制的调查访问；通过实验验证了捕获‑封存竞争、存储与性能指标。

**💡 创新点**

创新点：
• 通过FMP把取证工作迁移到控制平面，提供始终在线的决策记录和可序列化的捕获触发；
• 设计分层捕获策略（常规记录、定向保存、全状态捕获）并结合速率限制，形成可量化的捕获预算；
• 引入链式哈希决策记录与定期锚定，构建 tamper‑evident 证据账本；
• 通过捕获‑封存序列化阈值，解决了“取证碎片化”问题；
• 在零信任环境下首次系统化评估了取证可行性、存储预算与捕获延迟的耦合。

**🔧 技术方法**

技术手段：
• PEP/PDP 采用 OPA、Cedar、Casbin 的 in‑process 或 sidecar 模式；
• FMP 通过 FastAPI 钩子捕获决策并生成链式 SHA‑256 记录；
• 使用 CRIU 进行全内存快照（包括进程状态、文件系统）并进行异步序列化；
• 采用 eBPF 在内核层实现即时 SIGKILL，演示捕获‑封存竞争；
• 采用 Hash‑链和 RSA‑2048 锚定构建证据账本；
• 通过多源遥测（身份、资源、风险、时间等）填充决策记录，并对字段覆盖率做投影；
• 评估中使用多租户 Kubernetes (k3s) 与托管 K8s（DigitalOcean）以及本地裸机。

**📊 数据集**

使用的数据集：
• 五个公开网络安全基准集（CERT, LANL, NSL‑KDD, UNSW‑NB15, CIC‑DDoS2019）用于遥测覆盖率与捕获预算模型；
• 1 份合成的决策记录 schema 作为完整字段基准；
• 自定义的 256 MiB 目标进程和 Docker 容器用来测试捕获‑封存竞争。

**📈 对比分析**

比较与性能：
• 请求路径开销：OPA sidecar 1.0% 下降，Cedar/Casbin in‑process 1.9–3.0%；
• 存储量：按设定 500 req/s、0.02 高危率、0.05 / s 完全捕获上限时，测得 ~46 TB 记录（保守值 ~96 TB）；
• 捕获‑封存竞争：CRIU 65 ms 捕获成功；直接 SIGKILL 9 ms 失效；在 k3s 上序列化屏障后，1000/1000 复原成功；
• 速率限制下，全状态捕获占比 0.05 / s，允许的全内存占用在 30 天保留内约 35 TB；
• 证据账本吞吐：1 M 记录/秒峰值，p99 < 0.6 µs。

**⚠️ 局限性**

局限性：
• 仅实现组件级原型，未在生产级多节点或多云环境完整部署；
• 容器内存捕获未集成到 k3s 流水线，缺少 Docker/Pod 级快照；
• FMP 的安全边界与失效语义未实现，假设 gateway 与 evidence store 可信；
• 仅评估了单一检测器（Falco、Suricata）与单一封存方式，未覆盖所有攻击与自毁策略；
• 受限于实验环境，未评估大规模流量与高并发下的实时捕获延迟；
• 对法证可采性的正式法律评估仅在附录中概述，未在真实司法场景验证。

---

## 274. UniQuery4R: Unified 4D Scene Reconstruction from a Single Query

**arXiv ID:** 2608.17283 | [PDF](https://arxiv.org/pdf/2608.17283v1)

**作者:** Tiancheng Chen `[一作]` (Kosmo Research), Zesong Li `[通讯]` (Kosmo Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于连续源像素查询的 UniQuery4R 框架，能够在一次联合编码后按需选择源视图和目标视图，预测对应点、目标时刻 3D 位置、场景流、源深度以及相机参数，从而实现稀疏或稠密的 4D 场景重建；

**💡 创新点**

核心创新在于：①仅在解码阶段通过源到目标的跨注意力确定源/目标视图，消除固定时序嵌入；②采用连续像素查询保持亚像素精度；③将场景流分解为方向与幅值参数，并分别监督，提升动态/静态点的预测效果；④统一查询表示同时输出对应、几何与运动，提升特征共享与计算复用；

**🔧 技术方法**

技术要点包括：多视角 Transformer 编码器、四级多尺度特征金字塔、带门控的多尺度融合、源‑目标跨注意力解码、方向–幅值场景流参数化、基于 VGGT 的相机头、联合多任务损失（含置信度、相机、流、深度等）以及混合真实与合成数据的训练；

**📊 数据集**

训练与评估使用了 Kubric‑4D、PointOdyssey、Hypersim、DL3DV、CoTracker3Kubric、Stereo4D、Virtual KITTI 2、Waymo 及内部数据集；在 WorldTrack 协议下，评估了 PointOdyssey、Panoptic Studio、Dynamic Replica、Aria Digital Twin 四大数据集，并在 DAVIS 上进行长序列可视化验证；

**📈 对比分析**

与 SpatialTrackerV2、St4RTrack、TraceAnything、Any4D、V‑DPM、4RC、OpenD4RT 等方法对比，UniQuery4R 在 WorldTrack 的宏平均场景流 τ@0.1m 和 EPE 以及动态点跟踪 APD/EPE 上均名列前茅；在深度和相机估计指标上也保持竞争力；消融实验进一步验证了查询共享、方向–幅值流参数化和多尺度金字塔带来的提升；

**⚠️ 局限性**

局限性包括：在严重遮挡、极大旋转或高速翻转等情况下仍易出现误匹配与身份交换；编码成本随剪辑长度线性增长，需要窗口化处理；对极端大位移的预测仍略逊于部分基线；跨窗口融合等长时序扩展尚未解决。

---

## 275. Reinforcement Learning as (Discrete) Potential Theory

**arXiv ID:** 2608.17181 | [PDF](https://arxiv.org/pdf/2608.17181v1)

**作者:** Christopher Connolly `[一作]` `[通讯]` (SRI International), Christopher Connolly (SRI International)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

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

## 276. FedPref: Federated Preference Learning for Structured Radiology Report Extraction

**arXiv ID:** 2608.16971 | [PDF](https://arxiv.org/pdf/2608.16971v1)

**作者:** Flint Xiaofeng Fan `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**通讯引用:** 22089 | [OpenAlex ID](https://openalex.org/A5078339613)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在多医院情境下，利用冻结的语言模型生成候选结构，通过本地标注对候选进行排序，然后在各站点本地训练LoRA适配器并在服务器上聚合参数，实现无共享报告的协同学习；

**💡 创新点**

创新点在于把结构化注释转化为本地偏好接口，利用跨模型候选提供对比信号，并仅传输LoRA参数实现安全高效的联邦偏好学习；

**🔧 技术方法**

使用的技术包括冻结多模型生成、结构投影与评分、基于选择响应的SFT与DPO微调、LoRA低秩参数通信以及权重按样本数加权的FedAvg；

**📊 数据集**

实验数据来自MIMIC‑CXR与Chest ImaGenome两大公开胸片报告数据集，构建了9种疾病与18个解剖位置的结构化标签；

**📈 对比分析**

与本地独立训练以及中央合并训练对比，FedPref在开发集上将客户端平均F1提升2.49点、最差站点提升9.10点；在锁定的400报告金标准上FedPref 68.68、中央71.67，保持与中央相同的性能序列；

**⚠️ 局限性**

主要局限在于仅在模拟联邦环境中验证，缺乏真实多中心异质性与安全聚合机制，且联邦模型仍落后于集中训练的性能。

---

## 277. What If AI Carried Her Imagination? Black Girls as Creators in an AI Storytelling Weekend Program

**arXiv ID:** 2608.16896 | [PDF](https://arxiv.org/pdf/2608.16896v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 278. Modeling the Hydrodynamics in the Oslofjord using ADCIRC

**arXiv ID:** 2608.17140 | [PDF](https://arxiv.org/pdf/2608.17140v1)

**作者:** Matthew Scarborough `[一作]` (Norwegian University of Life Sciences), Eirik Valseth `[通讯]` (Norwegian University of Life Sciences)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建并验证了一个新的无结构有限元网格及 ADCIRC 模型，用于奥斯陆峡湾的海洋水动力模拟；通过不同潮汐成分、风压和 STOFS 边界条件对模型进行多场景测试。

**💡 创新点**

创新点在于：①采用 OceanMesh2D 生成 50 m 细分无结构网格，兼顾岛屿、窄水道等复杂地貌；②将 ADCIRC（GWCE 形式）用于极简 2‑D barotropic 模型，实现高效数值求解；③首次在小域内将 STOFS-2D-Global 边界水面高度直接输入 ADCIRC，显著降低潮汐边界条件不足导致的误差。

**🔧 技术方法**

使用的技术与方法包括：ADCIRC（含 GWCE、湿干算法、线性摩擦）、OceanMesh2D 网格生成、SRTM15+ 与 GSHHG 数据拼接、TPXO9 潮汐成分、GFS 风压输入、MET 操作模型对比、RMSE 误差评估、半隐式时间积分与 Crank‑Nicolson 方案。

**📊 数据集**

数据集包括：SRTM15+ 全球海底测深、GSHHG 全球岸线、TPXO9 潮汐数据库、GFS 1 h 风压场、SE Havnivå 观测水位、STOFS‑2D‑Global 边界水面高度、MET 的河口径流与风速等。

**📈 对比分析**

比较方法：将 ADCIRC 模型结果与 MET 现有运算模型和观测水位做时序对比，计算 RMSE；在不同潮汐成分组合下对比误差并统计计算时间。性能表现：在 TACC Frontera 上 40 核心即可完成 1 h 物理时间约 12 s，比 MET 运算模型快 5 倍，CPU 资源减少 24 核。使用 STOFS 边界输入时，RMSE 降至原来的 1/3，误差显著降低。

**⚠️ 局限性**

局限性：①网格域仅覆盖奥斯陆峡湾，缺乏北海/挪威海等外部潮汐传播信息；②潮汐边界条件受限，难以捕捉极端天气引发的潮汐与浪涌；③未包含河流径流、降雨、冰面等额外水动力来源；④使用恒定摩擦系数，未实现空间可变摩擦；⑤在 STOFS 边界输入时出现高频振荡，需进一步数值稳定化。

---

## 279. Procedural Collapse: A Structural Account of Disengagement in LLM-Assisted Writing

**arXiv ID:** 2608.17326 | [PDF](https://arxiv.org/pdf/2608.17326v1)

**作者:** JaeWon Kim `[一作]` (University of Washington), Katelyn Mei `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文提出将学生使用大型语言模型（LLM）写作的失去投入（disengagement）视为交互结构导致的“程序崩溃”，并给出三种基于交互设计的干预方向，强调拆分交互、先询问目标以及单层输出来降低评估负荷

**💡 创新点**

创新点在于从结构化交互角度解释LLM写作工具导致的认知负荷和浅层参与，而非仅停留在用户缺陷的描述；提出可操作的设计框架，解释现有多阶段写作协助工具为何有效

**🔧 技术方法**

论文并未实现新的算法或技术，而是基于理论框架分析LLM写作界面，提出交互结构化设计思路

**📊 数据集**

无实验数据集，论文为概念性、理论性研究

**📈 对比分析**

无实验对比，作者通过逻辑推演与先行研究的引用支持设计建议；未给出具体性能指标

**⚠️ 局限性**

局限性包括缺乏实证验证，未考虑不同写作任务与用户差异对交互结构效果的影响；所提设计方案仍需在真实写作情境中测试其有效性

---

## 280. Deep Learning for Cross-Border Electricity Price Forecasting: A Comparative Study

**arXiv ID:** 2608.17091 | [PDF](https://arxiv.org/pdf/2608.17091v1)

**作者:** Hadeer Elashhab `[一作]` (Karlsruhe Institute Of Technology), Benjamin Schäfer `[通讯]` (Karlsruhe Institute Of Technology)

**通讯引用:** 9703 | [OpenAlex ID](https://openalex.org/A5005576823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究评估了多种深度学习模型在跨境电价预测中的性能，并构建了可复现的基准框架；

**💡 创新点**

创新点在于引入零/一/少样本学习策略，系统比较不同架构在德国-卢森堡电价区的迁移泛化，并公开了数据与实验流程；

**🔧 技术方法**

技术方法包括六种深度学习架构（N‑HiTS、NBEATSx、TFT、Mamba、LSTM、VT），统一训练管线、Optuna 超参数搜索以及零/一/少样本迁移训练；

**📊 数据集**

使用的数据集为公开的 Energy‑Charts API 价格与外生特征，聚焦 2024 年德国‑卢森堡投标区，训练集分别从 2018、2020 与 2023 开始；

**📈 对比分析**

对模型的比较基于 MAE/RMSE 在同一测试集上的评估，所有模型 MAE 均在 18.5–19.4 之间，NBEATSx 与 N‑HiTS 表现最佳，Transformer 需要更多调优；

**⚠️ 局限性**

局限性包括仅使用 MAE/RMSE 指标，缺乏多次实验与统计显著性检验，超参数搜索范围有限，未评估概率预测或更细粒度的计算资源记录。

---

## 281. Reuse Before You Retrieve: Diagnosing Headroom and Complementarity for Test-Time Augmentation of Embodied Multimodal Policies

**arXiv ID:** 2608.17484 | [PDF](https://arxiv.org/pdf/2608.17484v1)

**作者:** Yuhwan Jeong `[一作]` (KAIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了两种可测量的诊断指标——可恢复的余量（Recoverable Headroom）和检索互补性（Retrieval Complementarity），用于判断冻结的视觉-语言-动作（VLA）策略在测试时是否需要通过重复采样或检索外部演示来提升性能。

**💡 创新点**

创新点在于：①将可恢复余量定义为单次与多次采样成功率之差，从而量化策略内部已存在的潜在成功行为；②使用检索互补性指标（演示适配误差）评估策略对外部动作先验的缺口；③通过实验验证两种诊断指标能分别预测重试选择和检索增益，并探讨两者组合的额外提升。

**🔧 技术方法**

主要技术包括：①基于冻结视觉编码器的示例集合，对整个轨迹做平均余弦相似度评分的 episode‑level retry selector；②事件模式前瞻检索（event‑schema prospective retrieval）以热启动动作生成；③多样本重试与并行执行、演示适配误差（DFE）评估、以及对比分析。

**📊 数据集**

使用的数据集有：LIBERO（Spatial、Object、Goal、LIBERO‑10 四套）、SimplerEnv‑Bridge（carrot、spoon、stack）、OpenVLA 在 SimplerEnv coke‑can 上的实验、以及受遮挡影响的 LIBERO‑Occ 版本。

**📈 对比分析**

与基线（单次采样 pass@1）对比，重试选择在所有 VLA backbone 上均实现 +7.8~+21.0 的成功率提升，几乎能回收 90% 的可恢复余量；检索仅在 π₀ backbone 上产生 +4.5~+6.4 的提升；两者结合可进一步提升约 +2.4。实验还展示了跨机器人、跨仿真器、以及低观测条件下的迁移性与局限性。

**⚠️ 局限性**

限制包括：①重试选择仅适用于可重试或并行执行的场景，无法提升单次执行的性能；②检索增益高度依赖演示集合和检索机制，跨域迁移表现不佳；③可恢复余量与检索互补性并非普适指标，受动作空间和采样协议影响；④OpenVLA 的重试捕获率低，说明缺乏有效的轨迹排名信号。

---

## 282. FireRedTTS3: Unified Speech Generation and Editing with Semantically Enriched Speech Representations

**arXiv ID:** 2608.17492 | [PDF](https://arxiv.org/pdf/2608.17492v1)

**作者:** Feiyu Shen `[一作]` (Xiaohongshu), Yao Hu `[通讯]` (Xiaohongshu)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了FireRedTTS3框架，使用Semantically Enriched Continuous Speech Representation（RedAE）实现稳定的连续自回归TTS，并支持多语言零样本语音克隆、指令控制的语音设计与编辑。

**💡 创新点**

创新点在于：①用冻结的多任务音频编码器为RedAE注入语义监督，使连续特征更稳定；②通过单阶段训练省略额外语义模块和多级分词器；③将LLM-DiT轻量化并统一到同一模型实现克隆、指令控制与编辑。

**🔧 技术方法**

关键技术包括：Qwen3式Transformer的多尺度自编码器、语义蒸馏、GAN对抗训练、AdaLN条件DiT、轻量化LLM-DiT、语音嵌入与语言标签、指令规划（结构化文本计划）以及Classifier-Free Guidance。

**📊 数据集**

训练数据涵盖约500k小时多样化语音（包括清音、噪音、音效、音乐）用于RedAE；FireRedTTS3-Base训练了2.6M小时中英语料，后续再加入560k小时覆盖24语种和21种中文方言；FireRedTTS3-Instruct训练了330k小时语音设计与编辑数据。

**📈 对比分析**

在Seed-TTS-Eval、MiniMax-MLS-Test、InstructTTSEval和Ming-Freeform-Audio-Edit四个基准上与多种现有模型对比，FireRedTTS3-Base在语音可懂度与说话者相似度上平均均为最佳或第二；FireRedTTS3-Instruct在指令跟随准确率、语音编辑准确率和说话者一致性方面均优于同类模型，显示出更好的控制与编辑效果。

**⚠️ 局限性**

局限性包括：①对少数语言（如粤语）的评估受识别器限制；②在极端噪声或极少数据的方言下表现可能下降；③仍依赖大型预训练LLM与算力，实际部署时需优化模型规模；④对非常复杂的指令或跨模态（视觉+语音）控制仍有待改进。

---

## 283. Health Inquiry with AI: How Empathetic Expression and Conversational Contexts Shape Users' Communicative Acts

**arXiv ID:** 2608.17144 | [PDF](https://arxiv.org/pdf/2608.17144v1)

**作者:** Xi Zheng `[一作]` (City University of Hong Kong), Yuhan Luo `[通讯]` (City University of Hong Kong)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对48名受试者在三种健康情境（普通、敏感、心理健康）下，四种同理心表达方式（基线、语言、视觉、语言+视觉）进行对话实验，分析用户的沟通行为。

**💡 创新点**

首次将患者沟通行为理论迁移至人机交互领域，揭示会话语境对用户主动信息交流的决定作用，而非单纯的同理心模态。

**🔧 技术方法**

使用线性混合效应模型与混合效应逻辑回归对对话长度和沟通行为进行统计，辅以质性访谈与编码。

**📊 数据集**

实验收集的对话日志，涵盖三种情境下四种同理心配置的文本数据。

**📈 对比分析**

通过比较不同同理心模态与语境下回复长度及沟通行为出现率，发现同理心模态仅显著提升回复长度，沟通行为差异主要受语境影响，未达到显著统计差异。

**⚠️ 局限性**

未验证沟通行为对AI回答质量的提升，且实验样本可能受文化差异限制，结果缺乏跨文化一般性。

---

## 284. An Emulation Anchored Digital Twin Testbed for Cyberattack and Defense Analysis in Hospital IT OT Environments

**arXiv ID:** 2608.17650 | [PDF](https://arxiv.org/pdf/2608.17650v1)

**作者:** Prashant Rawat `[一作]` (Indian Institute of Technology Ropar), Geeta Yadav `[通讯]` (Indian Institute of Technology Ropar)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

创建了一个基于 Docker 的医院 IT–OT 安全实验平台，集成了 IT、OT、DMZ、互联网分段网络、SCADA 系统和数字孪生监控，并支持跨域攻击仿真与 AI 辅助防御；

**💡 创新点**

提出了将数字孪生与仿真嵌入的实验框架，支持可控的多阶段攻击演练、实时监测、命令控制以及人机协同的安全决策支持；

**🔧 技术方法**

使用 Docker 容器化、OpenPLC、RapidSCADA、OpenEMR、Orthanc、Promtail/Loki、JWT+RBAC 访问控制、强化学习（DQN）等技术实现平台与防御模型；

**📊 数据集**

利用自建的模拟数据集，包括 IT/OT 网络流量、日志、Modbus/TCP、SSH 暴力攻击等，用于训练 RL 代理和评估防御效果；

**📈 对比分析**

通过资源占用、延迟、RL 防御成功率等指标评估，证明平台 CPU 占用低于 0.5%，Modbus RTT < 1 ms，RL 代理能显著阻止 SSH 暴力攻击，并与其他工业控制实验平台做对比展示轻量化与可扩展性；

**⚠️ 局限性**

仅实现协议层模拟，缺乏硬件时间精度、真实设备、实时同步，难以复现物理过程和高保真攻击效果，且对更复杂攻击场景和更大规模部署的适应性尚待验证。

---

## 285. A New Syntax and Semantics for Probabilistic Trace Expressions

**arXiv ID:** 2608.17594 | [PDF](https://arxiv.org/pdf/2608.17594v1)

**作者:** Davide Ancona `[一作]` (University of Genoa), Viviana Mascardi `[通讯]` (University of Genoa)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种新的概率轨迹表达式（PTE）语法和语义，用以在部分可观测环境下进行运行时验证（Runtime Verification），能够显式处理缺失事件、延迟事件及不可观测事件，并提供两种概率解释（观察语义和猜测语义）。

**💡 创新点**

创新点包括：①把概率与事件类型关联而非语法转换；②区分观察与猜测两种语义，分别对应观测不确定性与生成不确定性；③将隐藏马尔可夫模型（HMM）和线性时序逻辑（LTL）映射到PTE，从而实现更广泛的模型支持；④在单机与分布式监测中引入状态集合推演，提出分布式监测缓解状态爆炸的策略。

**🔧 技术方法**

主要技术：轨迹表达式（TE）语法、概率分布函数Π、两种PTE语义（ObSem、GuesSem）、与HMM、LTL的转换算法、集合转移与闭包运算、Prolog实现及Backtracking推理。

**📊 数据集**

使用的数据集：文章以模拟的Mars Rover（COGC）通信协议和一个四代理通信示例作为实验案例，未采用公开真实数据集；主要通过手工构造的事件序列与缺失事件来展示方法。

**📈 对比分析**

比较方法：通过对比单机与分布式监测在处理缺失事件时的状态空间规模和推理时间进行讨论；实验采用SWI-Prolog实现，展示在示例场景下的运行时间与状态数量，表明分布式方案能显著减少单机状态爆炸，但总体性能仍受缺失事件密度和并发度影响。

**⚠️ 局限性**

局限性：①状态空间爆炸在缺失事件较多时仍难以完全避免；②缺乏大规模真实数据集的实验验证；③算法复杂度分析有限，尚未给出正式的时间/空间复杂度证明；④分布式实现对网络同步与一致性假设要求较高，实际部署难度待进一步研究。

---

## 286. GSToken: Geometry-Structured Gaussian Tokens for Compact 3D Medical Image Representation

**arXiv ID:** 2608.17425 | [PDF](https://arxiv.org/pdf/2608.17425v1)

**作者:** Xiaoduo Li `[一作]` (Taiyuan University of Technology), Quan Gu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了基于高斯几何参数的GToken（Gaussian Token）对多模态MRI进行压缩分割的方式，将每个令牌的内容与可学习的三维中心、尺度、方向等空间信息相结合；并设计了冻结令牌评估协议以独立衡量令牌保留的信息量。

**💡 创新点**

创新点包括①首次将可学习的高斯几何结构引入医学影像令牌，实现显式的空间支持；②构建冻结令牌评估框架，能在相同容量下客观比较不同令牌化方法的有效信息；③通过粗细分配和区域感知核心集选择，动态优化令牌覆盖与稀疏性。

**🔧 技术方法**

技术方案主要包含：3D卷积编码器、可学习高斯参数预测头、交叉注意力上下文聚合、Transformer令牌交互、Gaussian splatting实现密集重构、区域感知核心集选择、冻结令牌评估协议。

**📊 数据集**

使用了公开的BraTS 2021脑肿瘤MRI数据集，划分训练/验证集并保留20例作审计集。

**📈 对比分析**

在同容量冻结评估（GToken256 vs. TokenLearner256 256令牌，Patch512 512令牌）下，GToken256在宏观Dice、Surface Dice、HD95等指标上显著优于两者，平均提升约0.22宏观Dice、0.16 Surface Dice，HD95下降约6.9，且三次随机种子实验均保持优势。

**⚠️ 局限性**

局限性在于与成熟基线nnU-Net相比分割精度仍有差距；论文聚焦于信息保留验证，未实现完整的高性能分割系统；高斯参数学习、跨尺度特征融合与令牌分配策略仍需进一步优化。

---

## 287. ADAPTD: Adaptive Detection and Proactive Threat Defense for Autonomous APT attacks

**arXiv ID:** 2608.17251 | [PDF](https://arxiv.org/pdf/2608.17251v1)

**作者:** Yeongwoo Kim `[一作]` (KTH Royal Institute of Technology), György Dán `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ADAPTD 框架，利用分布式贝叶斯更新、即时阻断与预测性驱逐，针对自主 APT 攻击实现决策理论下的防御。

**💡 创新点**

创新点包括：① 基于分割的 HMM 动态转移，显著降低通信负担；② 通过成本降低启发式实现快速阻断；③ 用 MC 估算阻断与驱逐的期望成本，提供可解释的决策；④ 兼顾轻量级与重量级防御，兼容业务连续性。

**🔧 技术方法**

使用技术包括：POMDP 建模、前向算法与贝叶斯更新、动态转移概率、似然比阈值触发、Monte Carlo 仿真、贪婪成本优化和分布式贝叶斯传播。

**📊 数据集**

实验数据基于合成 kill‑chain（由 MulVAL 生成的攻击图与 MITRE ATT&CK 关联）构建，未使用真实网络流量或企业日志。

**📈 对比分析**

与集中式贝叶斯、扩散 HMM、Transformer 自动编码器触发等基线对比，指标为平均总成本、驱逐延迟、误驱逐率及阻断预算影响；ADAPTD 在所有场景下均表现出更低成本、更短延迟与更低误驱逐率，同时通信与计算开销明显降低。

**⚠️ 局限性**

局限性：仅在模拟环境下验证，缺乏真实企业数据；假设攻击图已知且攻击者单一；对噪声警报的鲁棒性受限；MC 估算在大规模网络中仍可能带来计算压力；需手工调参以平衡误报与漏报。

---

## 288. WIP: LLM Odyssey: A Game-Based Platform for Teaching LLM Engineering Concepts

**arXiv ID:** 2608.16924 | [PDF](https://arxiv.org/pdf/2608.16924v1)

**作者:** Priyamvada Tripathi `[一作]` `[通讯]` (Tufts University), Priyamvada Tripathi (Tufts University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一个名为LLM Odyssey的开源浏览器游戏平台，包含13个交互式游戏，旨在教授大型语言模型工程概念。

**💡 创新点**

创新点在于将LLM工程教学与游戏化、分层学习路径、五种基于学习科学的教学策略（即时反馈、分级提示、渐进难度、实例演示、真实情境）结合，并提供真实生产场景与即时反馈的交互式学习体验。

**🔧 技术方法**

技术上采用前端React 18.2 + Tailwind CSS实现客户端游戏逻辑，后端使用无服务器架构记录匿名交互数据；预先计算Tokenization结果以保证性能；整体响应时间≤16 ms。

**📊 数据集**

使用真实生产场景文本（多语言文本、代码、法律文件）作为挑战内容，并利用预计算的分词、注意力图等信息；未使用公开大型数据集。

**📈 对比分析**

平台已在Durham College部署，技术指标满足：页面加载≤100 ms，动画60 fps，交互延迟≤16 ms；学习效果评估仍待后续实验，暂未形成性能对比。

**⚠️ 局限性**

限制包括缺乏实证学习成效验证、难度曲线未自适应、仅单一机构试点、无对照组以及对不同水平学习者的差异化支持不足。

---

## 289. Children, but not language models, show accelerating returns in word learning

**arXiv ID:** 2608.17120 | [PDF](https://arxiv.org/pdf/2608.17120v1)

**作者:** Michael C. Frank `[一作]` `[通讯]`, Michael C. Frank

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对比儿童在早期词汇学习与大型语言模型的学习曲线，提出并验证了“加速累积器”模型，揭示儿童词汇增长随时间加速。

**💡 创新点**

创新点在于将儿童词汇增长建模为对数时间加速累积，并将其与LM的恒定比例回报进行对比，首次量化儿童在学习效率随发展而提升的现象。

**🔧 技术方法**

使用贝叶斯推断拟合Rasch/IRT加速累积模型，对儿童CDI纵向数据进行分析；对LM使用惊讶度（surprisal）曲线拟合四参数逻辑斯蒂曲线，提取加速指数κ，并计算比例回报。

**📊 数据集**

采用五个单语儿童CDI纵向数据集（美国Thal、Smith、Marchman；挪威、日语）以及Childes、BabyLM、ClimbMix的儿童指向语料训练GPT‑2等LM。

**📈 对比分析**

通过留一交叉验证（LOO‑ELPD）和个体预测评估模型；加速累积器在所有儿童数据集上显著优于纯累积器，儿童加速指数κ平均约为10–13，远高于LM的κ≈1，显示儿童学习效率随时间显著提升，而LM表现出恒定比例回报。

**⚠️ 局限性**

局限在于仅利用父母报告的CDI，未直接观测真实词汇产生；加速模型尚未提供因果机制；假设儿童输入量均匀，未充分考虑输入质量差异；LM训练仍未能复现加速效应，提示需要进一步研究模型内部机制。

---

## 290. SkillEffect: Checked Lowering for Memory-Bounded Agent Tools

**arXiv ID:** 2608.17007 | [PDF](https://arxiv.org/pdf/2608.17007v1)

**作者:** Yinuo Wang `[一作]` (University of Notre Dame), Yiyu Shi `[通讯]` (University of Notre Dame)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并实现了一套在代理工具调用前进行“检查-下推”运行时的框架，保证语言模型生成的工具程序在满足声明的功能结果的同时，严格控制内存使用并在超限时主动放弃。

**💡 创新点**

创新点在于提出统一的五项义务（源识别、输入事实、下推IR、arena计算、发布后置条件）以及插件化的关系契约，能够让不同类型的工具（流式、分块、Top‑k等）共享同一信任边界与资源控制机制。

**🔧 技术方法**

技术上使用了闭源语法解析、独立校验器、基于插件的下推IR生成、cgroup内存限制、容量租赁和基于后置条件的发布门控；并通过哈希绑定和平台manifest实现可重复的资源校验。

**📊 数据集**

实验数据集覆盖六大工具族，分别是Polars CSV（2M行）、FASTA（2M记录）、FCS（2M事件）、Zarr（65K×1K数组）、XLSX（100K行）以及Top‑k日志（750K条），共计六个不同规模的工作负载。

**📈 对比分析**

对比方法包括直接按需执行、自动生成的无约束下推、以及已存在的安全门控。实验显示，bounded‑lowering 在外部固定内存上限（64‑2048 MB）下完成率提升至90%+，AUC提升约 0.95；在并发场景下，带下推的吞吐量提升 1.8 倍，且绝大多数作业在租赁内完成，无 OOM 事件。

**⚠️ 局限性**

局限性主要是：只能处理确定性、只读、可验证结果的工具；需要为每个工具编写闭源关系插件；平台依赖性强，需重新校准；不支持有远程副作用或多步状态的任务；目前未覆盖大规模多步骤工作流。

---

## 291. Which CS1 Students Will Fail? Identifying Digital Markers from Learning Analytics in Computer Systems and Architecture Using Weighted Academic Momentum and Interaction Logs

**arXiv ID:** 2608.16914 | [PDF](https://arxiv.org/pdf/2608.16914v1)

**作者:** Lighton Phiri `[一作]` (University of Zambia), Bydon Simukoko `[通讯]` (University of Zambia)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

基于学生早期测评成绩、人口统计信息和学习管理系统的数字标记，构建并验证了一个逻辑回归早期预警模型，用以在CS1课程中识别高风险失败学生。

**💡 创新点**

创新点在于：① 通过利益相关者访谈制定的“加权学术势能”指标；② 将数字标记与传统信息结合的系统化特征工程；③ 采用SMOTE+ENN重采样与阈值调优实现高召回；④ 利用SHAP解释模型，为教师提供可操作的干预建议。

**🔧 技术方法**

技术方法包括：逻辑回归、随机森林、XGBoost、支持向量机等多分类器；SMOTE+ENN类别平衡；阈值调优；SHAP可解释性分析；特征消融实验。

**📊 数据集**

使用了2017–2021四个学期的CS1（Computer Systems and Architecture）课程数据，共284名学生，包含SIS人口统计、预课程调查、Moodle日志与测评成绩。

**📈 对比分析**

通过对七种分类器和特征子集进行交叉验证与AUC/准确率/召回率评估，最终的逻辑回归模型在测试集上取得74.7%准确率、宏观F1 0.742、AUC 0.800，并在召回率87%与假阳性率41%之间达成可接受的早期干预平衡。

**⚠️ 局限性**

局限性包括：样本量有限（单校单课）；缺乏高频点击流数据；COVID-19影响未完全控制；模型在其他课程或地区的泛化能力未知。

---

## 292. Auditing Self-Evolution in Financial Agents: Capability Gains, Security Drift, and Execution-Interface Mismatch

**arXiv ID:** 2608.17684 | [PDF](https://arxiv.org/pdf/2608.17684v1)

**作者:** Jialong Li `[一作]` (Independent Researcher), Jialing Zhu `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在模拟的电子银行环境中，对自我进化的金融智能体（SkillOpt、Agent Workflow Memory和ReasoningBank）进行了端到端的安全与能力审计。

**💡 创新点**

提出了一种执行基准的审计协议，能够区分能力提升、攻击曝光、条件攻击成功率和实际财务损害，并揭示了能力与安全可解耦、接口不匹配导致的误判等现象。

**🔧 技术方法**

使用 Qwen 3.7 Flash 语言模型、AgentDojo Banking 环境、匹配的正面学习轨迹、封闭的评估终点、状态重放以及多线性演化流程。

**📊 数据集**

基准数据集为经校正的 AgentDojo Banking v1.2.2，包含 15 个任务族、9 个注入目标、11 个工具以及 22 条正面采集轨迹。

**📈 对比分析**

将进化后的模型与未进化的 Static 版本在同一任务族与变体下对齐评估；SkillOpt 的实用度从 0.741 提升至 0.837，但曝光率和未授权状态变更均显著上升；ReasoningBank 实用度提升至 0.859，攻击成功率未显著变化；AWM 原始版本因接口不匹配导致实用度骤降，后期接口适配后恢复。

**⚠️ 局限性**

局限性包括：仅在单一执行器和单次离线演化过程中评估；缺乏统计显著性检验；实验环境与真实金融系统差距较大；曝光度受执行器特性影响；placebo 对比未匹配长度；以及不同演化系统在写入、检索等机制上的差异。

---

## 293. Learning latent progression states from spatial heterogeneity in uterine histopathology

**arXiv ID:** 2608.17337 | [PDF](https://arxiv.org/pdf/2608.17337v1)

**作者:** Qiming He `[一作]` (Fuzhou University), Congrong Liu `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `3855fcda-48ef-4070-a15e-803cd5c84d83` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出SpaTIE框架，将子宫恶性肿瘤H&E切片的空间形态异质性映射为连续的肿瘤状态，并与多组学关联。

**💡 创新点**

创新点在于结合宫颈特异性自监督基础模型与轨迹推断，在无时间/分子监督下从二维图像恢复进展相关肿瘤状态，并提供虚拟扰动方法优先化多组学特征。

**🔧 技术方法**

采用ViT‑L/16自监督学习与LoRA微调构建形态嵌入，再用图谱分析、Diffusion pseudotime进行状态推断，最后整合多组学数据并进行虚拟扰动与通路富集。

**📊 数据集**

使用自建PK3‑Uter（10426份H&E WSI）进行预训练，TCGA‑UCEC（566例）和TCGA‑UCS（91例）进行评估与多组学关联。

**📈 对比分析**

与通用基础模型UPFM及传统CNN比较，SpaTIE在诊断、分子预测和生存预测任务上保持或提升性能；状态推断展示高稳定性、空间一致性和特征平滑度。

**⚠️ 局限性**

局限在于缺乏纵向样本，无法确认真实进程时间轴；多组学关联基于批量数据且样本量有限，需进一步外部验证和空间组学证实。

---

## 294. Empowering Compact LLMs with Fusion of Layer-wise Exits for Recommendation

**arXiv ID:** 2608.17316 | [PDF](https://arxiv.org/pdf/2608.17316v1)

**作者:** Xurong Liang `[一作]` (University of Queensland), Hongzhi Yin `[通讯]` (University of Queensland)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出FLEXRec框架，利用层级退出融合与自适应连续路由提升紧凑LLM在序列推荐的效果与效率。

**💡 创新点**

融合多层退出、引入自适应连续路由与目标k hinge损失，实现动态深度选择与稀疏融合。

**🔧 技术方法**

基于LLM判别式推荐、层级预测头、连续ReLU路由、目标k hinge、负载平衡与Z-损失。

**📊 数据集**

Amazon Toys、Beauty、Yelp三个公开序列推荐数据集。

**📈 对比分析**

与传统SASRec、BERT4Rec及多种LLM推荐（E4SRec、SPRec、SETRec等）比较，在紧凑LLM上达到或超过大型基准，NDCG@20平均提升约1-3个百分点，推理延迟低于1ms。

**⚠️ 局限性**

对极稀疏数据集敏感，需手工调节目标k与损失权重，且仍依赖LLM预训练的文本信息，缺乏对不同任务的通用性。

---

## 295. Structure, Topics, and Diffusion Effects of Bluesky Starter Packs

**arXiv ID:** 2608.17489 | [PDF](https://arxiv.org/pdf/2608.17489v1)

**作者:** Andrea Failla `[一作]` (Institute of Information Science and Technologies A. Faedo, National Research Council), Carlos Ferreira `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究Bluesky平台的Starter Pack（启动包），对其结构、主题和内容扩散影响进行大规模实证分析。

**💡 创新点**

揭示Starter Pack是高度互联的发现生态系统，覆盖个人自动生成包和主题社群，并证明加入Starter Pack显著提升短期内容转发。

**🔧 技术方法**

采用图网络分析（双向网络、Disparity Filter）、主题建模（BERTopic + GPT-4.5）、匹配事件研究（差分中差分）等方法。

**📊 数据集**

使用约50,000个英文Starter Pack、666,000个用户、227M帖文和82M转发数据（Bluesky公开API采集）。

**📈 对比分析**

通过五邻近匹配和事件研究检验，发现加入Starter Pack后每篇帖子的转发数平均提升0.165次，转发概率提升1.79个百分点，效果明显但受先前趋势影响。

**⚠️ 局限性**

结果受限于匹配设计的选择偏误、采用的转发事件阈值是下限、仅限英文包、未考察长期影响及多语言差异。

---

## 296. KernelArc: A Multi-Agent Framework for GPU Kernel Optimization

**arXiv ID:** 2608.17071 | [PDF](https://arxiv.org/pdf/2608.17071v1)

**作者:** Joyjit Kundu `[一作]` (AILabs, Interuniversity Microelectronics Centre), Ludovic Denoyer `[通讯]` (AILabs, Interuniversity Microelectronics Centre)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种多代理协同的GPU核优化框架，利用LLM代理在共享的结论内存中并行搜索不同的优化策略，并通过确定性的评估守卫保证正确性与性能提升。

**💡 创新点**

创新点在于将策略专化的并行代理与仅结论共享内存、可配置保留窗口、确定性守卫以及基于平台的“plateau‑drafting”机制相结合，从而在固定候选预算内显著扩展搜索广度并突破单代理的局部最优。

**🔧 技术方法**

主要技术包括LLM驱动的观察‑编辑‑评估循环、CUDA/PTX源码级搜索、可配置的共享内存和守卫机制、以及在SOL‑ExecBench基准上进行的端到端性能测评。

**📊 数据集**

使用的数据集为NVIDIA的SOL‑ExecBench，涵盖L1、L2、Q和FI四类任务的多形状工作负载（如L1‑030、L2‑025、Q‑031和FI‑014）。

**📈 对比分析**

通过与公开基准（如cuBLAS、PyTorch参考实现）对比，单代理在Hopper上可将BF16 GEMM提升至约766 TFLOPS（超出cuBLAS 3.2%），多代理在SOL‑ExecBench上则在约100候选预算下实现单代理的2.04×速度提升，排名首位。

**⚠️ 局限性**

限制在于实验仅覆盖少数任务，未给出完整任务集的胜率；不同协调特性的效果受核和搜索阶段的依赖；以及缺乏对保留窗口大小等参数的系统性评估。

---

## 297. Domain-Adapted Molecular Language Models for Efficient Search of Make-on-Demand Libraries

**arXiv ID:** 2608.17567 | [PDF](https://arxiv.org/pdf/2608.17567v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 298. SentryBus: A Multi-Vantage Observability Model and Validated Instrument for I2C Sensor-Interface Manipulation

**arXiv ID:** 2608.17082 | [PDF](https://arxiv.org/pdf/2608.17082v1)

**作者:** Sandesh More `[一作]` (Florida Institute of Technology), Sneha Sudhakaran `[通讯]` (Florida Institute of Technology)

**通讯引用:** 162 | [OpenAlex ID](https://openalex.org/A5086402362)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 SentryBus，一个基于 I²C 总线的被动主机侧传感器接口监测器，并在实验平台上验证其对内联拦截器的透明度与采集稳定性。

**💡 创新点**

提出多视角观测模型和七类攻击分类，构建了可分离的合规层与统计层检测框架，并证明数据转移特征仅在同一会话内有效。

**🔧 技术方法**

利用 I²C 总线事务时序、操作序列、传输长度、地址、寄存器/FIFO 状态与原始数据变化特征，并结合稳健 z 分数、Jensen‑Shannon 散度等统计方法。

**📊 数据集**

使用 Max30102 光学传感器、RP2040 内联拦截器、ESP32 主机以及同步的双向捕获，收集了数千次采样、6304 秒的遥测数据。

**📈 对比分析**

通过与透明通过模式对比，评估内联拦截器在主机侧的时延增量仅为 0.842%，并在无攻击时段保持采样周期稳定；实验未完成攻击检测率评估。

**⚠️ 局限性**

仅研究内联拦截器攻击，未完成受控攻击试验；跨传感器迁移性未验证；采集仪器在低速下截断导致数据缺失，需进一步校正。

---

## 299. GraphWake: Group Polarization via Memory-Mediated Polarization Cascade in LLM-Agent Communities

**arXiv ID:** 2608.17665 | [PDF](https://arxiv.org/pdf/2608.17665v1)

**作者:** Haoran Bu `[一作]` (Beijing University of Posts and Telecommunications), Xi Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出GraphWake，一种利用LLM代理记忆的社区级极化级联攻击，能够通过在少数代理中植入立场支持的论证并在公共讨论中触发记忆检索，最终使未受攻击代理也被影响，导致整体极化加剧。

**💡 创新点**

创新点在于将记忆视为中介通道而非终点，设计三阶段威胁模型（曝光-检索-传播）并通过立场支持知识图谱、基于中心度的三元组选取与立场中性记忆提示，协同放大不同立场间的分歧。

**🔧 技术方法**

技术包括：多角度论证知识图谱构建、图压缩与连通、归一化贝叶斯中心度的路径挑选、LLM压缩为自然语言公理、立场中性记忆提示的实体关联与召回、以及基于G‑EVAL的立场向量评估。

**📊 数据集**

使用MoltNet（基于MoltBook的真实互动日志）中的八个议题（四个意识子议题、四个出现子议题）及其对应的立场集合。

**📈 对比分析**

与两种基线（无攻击与两阶段正则化）以及两种极化指标（估计Ray与方差）进行对比；实验显示GraphWake使平均极化指标从0.098提升至0.146、从0.130提升至0.213，证明对社区极化具有显著放大效果。

**⚠️ 局限性**

局限性包括：仅在无事实真相的开放式议题上验证；实验环境为同质化配置，难以直接推广到多样化平台；未在真实系统或人类用户上部署，防御策略尚待进一步研究。

---

## 300. Q-Learning With World Models

**arXiv ID:** 2608.17163 | [PDF](https://arxiv.org/pdf/2608.17163v1)

**作者:** Perry Dong `[一作]` (Stanford University), Dorsa Sadigh `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在离线强化学习框架下，作者提出在标准Q‑学习之上进行测试时搜索（tree‑search）以利用已学习的世界模型进行行动选择，既提升在线采样质量又提升评估时的决策效果；

**💡 创新点**

创新点在于将世界模型仅用于测试时搜索，而非在训练阶段引入模型产生的误差，从而避免模型误差累积，并通过结合Q‑值与模拟未来的值进行节点评估，获得更高效的行动选择；

**🔧 技术方法**

核心技术包括Q‑学习（基于EXPO或RLPD实现）、树搜索（beam/Pruned 搜索）、世界模型（状态空间为MLP残差模型，像素空间为动作编码的扩散变压器）、奖励函数估计和离线数据预训练；

**📊 数据集**

实验数据集为Robomimic（Lift、Can、Square、Tool Hang等四个操纵任务）与LIBERO（五个视觉操纵任务），均使用稀疏奖励；

**📈 对比分析**

与现有最先进的无模型（如RLPD、DSRL、QSM、QAM、FQL）和有模型方法（TD‑MPC2、EfficientZero V2）对比，所提方法在样本效率和最终成功率上均显著优于对照组；在基线算法上加速（如EXPO+搜索）同样表现出更快学习曲线；在像素级任务上也能保持或提升性能；

**⚠️ 局限性**

主要限制包括：搜索过程对计算资源和时延有显著开销；训练高质量的世界模型仍耗费大量数据与计算；若模型误差较大，搜索可能导致错误决策；此外，算法不适合对实时低延迟要求极高的应用场景。

---

## 301. From Student Risk Prediction to SC2R: Semantics-Constrained Counterfactual Recourse for Educational Decision Support

**arXiv ID:** 2608.17618 | [PDF](https://arxiv.org/pdf/2608.17618v1)

**作者:** Ngoc Luyen Le `[一作]`, Bertrand Laforge `[通讯]` (Sorbonne Université)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出SC^2R框架，将学习分析预测、整数规划反事实回溯与语义验证结合，生成可执行的干预计划。

**💡 创新点**

创新点在于将SHACL语义约束嵌入反事实回溯过程，使干预方案既符合模型预测又满足时间、预算、不可变性与资源可用性等教育约束。

**🔧 技术方法**

采用校准的逻辑回归预测模型、整数规划求解干预方案、OWL/SKOS轻量级本体和SHACL进行语义验证。

**📊 数据集**

以OULAD学生学习行为数据为数据集，构造两种决策时刻（d-14, d-7）的快照。

**📈 对比分析**

与Wachter式反事实搜索及无语义约束的整数规划对比，SC^2R在保持模型有效性与预测提升的同时，保证100%语义合规，并生成平均1.1条动作、成本10.4的紧凑方案。

**⚠️ 局限性**

主要局限在于实验仅为离线观察，没有证实干预效果，动作词典有限，模型在重训练时稳定性较低。

---

## 302. PXDepth: Pixel-Space Modeling for Structure Preserving Monocular Depth Estimation

**arXiv ID:** 2608.16984 | [PDF](https://arxiv.org/pdf/2608.16984v1)

**作者:** Zhiyuan Yuan `[一作]` (Sun Yat-sen University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**通讯引用:** 28724 | [OpenAlex ID](https://openalex.org/A5068837264)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于像素空间建模的单目深度估计框架，利用大尺度 ViT 提供全局上下文，配合像素级预测块实现细节恢复。

**💡 创新点**

创新点在于将全局语义编码与像素空间预测分离，并通过上下文引导的自适应归一化和粗细层次像素压缩，实现高效细节保留与结构一致性。

**🔧 技术方法**

使用技术包括大尺度 ViT 编码器、像素空间预测块、像素压缩与展开、上下文引导的自适应归一化、粗细层次像素压缩设计等。

**📊 数据集**

训练数据集涵盖多种合成 RGB‑D 数据集（Hypersim、MVS‑Synth、TartanAir、UnrealStereo4K、GTA‑SfM、Structured3D、UrbanSyn 等），在 MoGe Benchmark 与 MDA Benchmark 上进行零样本评估。

**📈 对比分析**

与 Depth Anything V2、DepthPro、InfiniDepth、MoGe‑2、PPD、MDA 等基线对比，零样本下在全局误差与边界精度上均达到或超过对手，同时单次前向推理速度约比 PPD 快 3.7 倍。

**⚠️ 局限性**

局限性包括对透明/反射表面深度标注不稳定导致预测误差，以及相对深度框架无法恢复绝对尺度，未来需要改进材质监督和尺度感知。

---

## 303. An Investigation of Translationese in the Generations of Multilingual Large Language Models

**arXiv ID:** 2608.17399 | [PDF](https://arxiv.org/pdf/2608.17399v1)

**作者:** Maria Valentini `[一作]` (University of Colorado Boulder), Katharina von der Wense `[通讯]` (University of Colorado Boulder)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多语言大模型生成文本的翻译痕迹（translationese）进行了系统评估。

**💡 创新点**

提出将传统翻译痕迹检测方法迁移到LLM生成文本，并揭示非英语生成中翻译痕迹普遍存在。

**🔧 技术方法**

使用基于句法、词频等表面特征的SVM分类器和ANOVA分析，结合多种语言学指标。

**📊 数据集**

采用Europarl、OLDI Seed、UdS等公开语料，涵盖英语、德语、西班牙语、希腊语、普什图语等。

**📈 对比分析**

通过与人类原文、人类翻译、机器翻译基线的对比，发现Gemini和Llama在非英语输出中翻译痕迹比例分别约60%和38%，但对人类可见的异常词汇抑制较好。

**⚠️ 局限性**

人类标注样本有限且来自作者自身，普什图语训练数据可能包含翻译原文，限制了结果的普适性。

---

## 304. TF-CADE: Foreground-Concentrated Text-Video Alignment for Zero-Shot Temporal Action Detection

**arXiv ID:** 2608.17422 | [PDF](https://arxiv.org/pdf/2608.17422v1)

**作者:** Yearang Lee `[一作]` (Korea University), Seong-Whan Lee `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种零样本时序动作检测框架 TF-CADE，显式将文本信息与前景动作区域对齐；

**💡 创新点**

创新点在于：① Action Concentrate Aggregation (ACA) 通过动作置信图和平滑实现前景加权视频嵌入；② Certainty-based Confidence Re-weighting (CCR) 用视频级相似度先验重新加权片段级置信度，抑制无关类别；

**🔧 技术方法**

技术包括：跨模态自注意力的双向交互、Gaussian 滤波动作置信图、前景加权聚合、软重加权、DIoU 位置回归、focal 损失；

**📊 数据集**

使用 THUMOS14、ActivityNet v1.3、HACS‑Segment 三个数据集，分别在 50%-50% 与 75%-25% 零样本拆分下进行实验；

**📈 对比分析**

与 STALE、T3AL、Ti‑FAD 等前景‑基/无前景 方法对比，TF-CADE 在无外部分类器的零样本设置下在所有 tIoU 阈值上均实现了 mAP 提升，且在跨数据集推广上表现更优；

**⚠️ 局限性**

局限性包括：仍依赖预训练的视觉语言模型，对极短/极长动作的检出效果受限；缺乏对多模态（如音频）信息的利用；模型对 Gaussian 滤波参数相对敏感。

---

## 305. Reflex-Guard: A Low-Latency Guardrail for LLM Prompt Safety Using Dense Semantic Embeddings

**arXiv ID:** 2608.17556 | [PDF](https://arxiv.org/pdf/2608.17556v1)

**作者:** Istiaque Ahmed `[一作]` (Osaka Metropolitan University), Thi Hong Tran `[通讯]` (Osaka Metropolitan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种低延迟本地提示安全防护框架 Reflex-Guard，能在 LLM 接收提示前进行快速过滤；

**💡 创新点**

创新点在于结合了 jailbreak‑aware 预处理、BGE 语义嵌入与七种轻量级二分类器，并引入统一的安全效率分数 RES，支持攻击类型自适应阈值；

**🔧 技术方法**

技术包括正则表达式 Base64 检测与解码、BGE‑small 句子嵌入、L2 归一化、Logistic Regression、XGBoost、LightGBM、HistGradientBoosting、RandomForest、AdaBoost、KNN 等轻量级分类器；

**📊 数据集**

使用包含 30,568 条提示的平衡数据集，来源于 AlpacaEval、Anthropic Red‑Team、JailbreakHub/DAN、JailbreakBench 与 MultiJail Bengali，涵盖 15,568 条有害与 15,000 条安全提示；

**📈 对比分析**

与 Llama Guard 2、SafeDecoding 等现有基线对比，Reflex-Guard 在 37.6 ms 的端到端延迟下实现 95.9% 召回率，RES 最高可达 16.79，明显优于对比基线（Llama Guard 2：RES 11.90；SafeDecoding：RES 9.80）；

**⚠️ 局限性**

局限性包括仅评估三类攻击（GCG、Base64、DrAttack），未涵盖多轮、翻译、白盒攻击；数据集主要为英文，低资源语言测试不足；基线对比使用公开报告结果，未在同一硬件环境下复现；

---

## 306. LLMs for Medical Consultation Are Evaluated Too Late: The Preformulation Gap

**arXiv ID:** 2608.17330 | [PDF](https://arxiv.org/pdf/2608.17330v1)

**作者:** Yining Hua `[一作]` (Harvard T.H. Chan School of Public Health), Eyal Klang `[通讯]` (Harvard Medical School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

评估大型语言模型在首次医学咨询时的表现，比较基线与加入“入门到护理”指令的交互，以揭示预制化差距。

**💡 创新点**

将预制化阶段作为独立评估目标，引入固定脚本与自适应标准化患者模拟，并设计简短系统指令观察行为变化。

**🔧 技术方法**

使用ChatGPT、Gemini、Claude三种API模型，基于脚本交互和自适应问答，结合手工设计的评分表进行功能评估。

**📊 数据集**

采用四个医师编写的多轮情景脚本以及自适应患者模拟器生成的回答，未使用公开临床数据集。

**📈 对比分析**

通过对固定脚本（24份）和自适应脚本（12份）转录按四个功能域（可用关注、前提修正、安全路由、交接准备）评分；结果表明指令显著改善序列与交接，但在获取关键诊断信息上仍存在不足。

**⚠️ 局限性**

局限包括样本量小、单次实验、无真人对照、模型与实际产品差异、指令对不同模型的效应不明、缺乏长期安全验证。

---

## 307. Grounding AI Agents in Contracts: An Empirical Evaluation of Spec-Driven Test Generation

**arXiv ID:** 2608.17177 | [PDF](https://arxiv.org/pdf/2608.17177v1)

**作者:** Michele Tufano `[一作]` (Google), Pat Rondon `[通讯]` (Google)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Spec-Driven Test Generation 框架，先让 LLM 自动生成半正式的设计契约（pre/post 条件和测试建议），再以此为 oracle 生成测试代码。

**💡 创新点**

创新点在于将 Design by Contract 的契约自动提取与 LLM 代理生成结合，形成两阶段的“契约驱动”测试流程，从而显著提升测试的覆盖率和缺陷检出能力。

**🔧 技术方法**

使用 Gemini 3 Flash 作为 LLM 代理，辅以自定义工具集（文件读写、命令执行、搜索等），并用 Gemini 3.1 Pro 作为评判者进行质量评估。

**📊 数据集**

实验数据集为 90 条来自 Google 内部 Issue Tracking System 的历史生产 Bug（C++、Java、Python、Go）与对应的修复代码。

**📈 对比分析**

通过 Greenfield 测试生成设定，在 k=5 次独立运行中与基线直接生成测试进行对比，Spec‑Driven 代理在缺陷检出率从 41.1% 提升至 63.2%（+9.8%），分支覆盖提升 2.5%（p=0.0034），并在 LLM‑Judge 评估中整体优于基线 77.8% 的比例，甚至超过 56.7% 的人工测试。

**⚠️ 局限性**

局限包括高 token 消耗（+38%）、对单一 LLM（Gemini）和单一企业代码库（Google）依赖，且在极端难题上仍可能无法生成完全覆盖的测试或契约。

---

## 308. Agents unlock new capabilities through Switching LoRA Adapters as a Tool (SLAaaT)

**arXiv ID:** 2608.17034 | [PDF](https://arxiv.org/pdf/2608.17034v1)

**作者:** Kenneth Ge `[一作]` `[通讯]` (Independent), Kenneth Ge (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

让模型在推理过程中动态切换LoRA适配器，以完成需要不同后训练能力的组合任务。

**💡 创新点**

提出了一种自动化的LoRA切换工具，使模型在同一推理轨迹中多次热交换适配器，显著降低灾难性遗忘并提升整体性能。

**🔧 技术方法**

采用Qwen3.6‑35B‑A3B大模型、LoRA微调适配器、子代理调用工具以及自学习切换策略。

**📊 数据集**

使用约2万对YAML→Fauxjson翻译样本和约2万条Fauxthon编程练习（来源于MBPP、Magicoder及自制合成数据），并在10个合成任务上评估。

**📈 对比分析**

与单一LoRA、融合LoRA、Arrow、子代理以及人类文件扩展启发式基准相比，自动切换在两项任务上击败人类启发式，能力税降低最多18倍，token使用显著减少。

**⚠️ 局限性**

实验仅在单一基模型、有限适配器（两种）和合成任务上进行，未验证多模型、多适配器或真实世界任务的可扩展性。

---

## 309. Automatic Transcription of Microtonal Free-Rhythm Vocal Music: A Case Study in Iranian Classical Music

**arXiv ID:** 2608.17114 | [PDF](https://arxiv.org/pdf/2608.17114v1)

**作者:** Sepideh Shafiei `[一作]` (Cu Test Inc.), Joel Rodriguez Caraballo `[通讯]` (Cu Test Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究提出了一套完整的计算工作流，用于自动转录伊朗古典音乐中的微音阶、无节拍人声曲目，并配备可视化编辑工具进行专家修正。

**💡 创新点**

创新点包括：①基于多层音调直方图与动态时间规整（DTW）相结合的主旋律提取方法；②针对伊朗音乐的微音阶与装饰音（如tahrir）设计的专属记谱符号；③利用音调直方图识别流动调律，支持用户自定义基准音（shāhed）和转调；④将DTW对齐结果与生成的MIDI及MusicXML同步显示，实现实时校对。

**🔧 技术方法**

技术手段包括：pYIN音调估计（Sonic Annotator）、自定义音调直方图与移动平均平滑、动态时间规整、Python/musi21进行符号表示、TinyNotation文本格式、以及基于DTW的可视化对齐。

**📊 数据集**

数据集：IRMA Audio‑MIDI数据集中的165首Karimi Radif录音（145首）及20首Tsuge录音；参考转录由人类民族音乐学家Masoudieh提供。

**📈 对比分析**

方法评价目前主要是主观专家校对；未给出定量准确率指标。作者计划未来进行正式的定量评估和专家一致性分析。

**⚠️ 局限性**

局限性：①对阈值和表达性手势的解释高度依赖音乐背景，缺乏自动化判定；②未提供客观准确率或与现有AMT系统的对比；③方法主要针对伊朗古典音乐，跨文化迁移需进一步验证；④对自由节奏的时值推算仍不完美，可能导致细节缺失。

---

## 310. SGHA: Evidence-Grounded Research Problem Discovery with Local Language Models

**arXiv ID:** 2608.17501 | [PDF](https://arxiv.org/pdf/2608.17501v1)

**作者:** Sarvesh Gharat `[一作]` (IIT Bombay), Junpei Komiyama `[通讯]` (MBZUAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并实现了一套名为SGHA（Structural Gap Hypothesis Agent）的系统，能够从已有科研文献中构建结构化证据图，自动识别跨论文的结构缺口（Structural Gap），经过验证筛选后生成可追溯、可检验的研究问题家族；

**💡 创新点**

创新点在于：①以文献证据为起点，而非纯文本生成，确保问题来源可追溯；②在生成前先进行验证门（支持、怀疑、可行性、机制等多视角评估）以降低假设错误；③完全使用本地开源权重LLM（9B模型）运行，避免对专有模型的依赖，兼顾隐私与可审计性；④在演化分支中提供更宽广的探索空间。

**🔧 技术方法**

核心技术包括：本地LLM文本解析与关系抽取；构建多类型证据图并检索结构化模式（Assumption–Failure、Shared‑Failure等）；多角色验证（支持、怀疑、可行性、机制）和独立批评；问题家族合并与半正式问题声明生成；可选的演化探索分支。

**📊 数据集**

实验使用了1250篇精选论文，涵盖五个机器学习领域（bandits、in‑context learning、reasoning/test‑time computation、offline reinforcement learning、uncertainty estimation），来源分别为OpenReview和arXiv。

**📈 对比分析**

与AI‑Scientist‑v2（同一Qwen模型）、Claude Opus版AI‑Scientist‑v2以及MOOSE‑Star公开模型进行对比，评估采用5名LLM评审者的“公式化质量”指标。SGHA在整体质量、源根特异性、假设边界清晰度、可形式化性和模糊性处理等方面均优于基线，尤其在验证前筛选机制使其在非专有模型环境下仍保持领先。

**⚠️ 局限性**

局限性包括：①对本地LLM抽取质量的依赖，模型容量与语料量直接影响输出；②仅能处理已在文献中暗示的研究问题，无法生成全新跨学科创意；③低信号场景下可能返回无输出；④缺乏外部创新性和正确性验证，仍需人工评估和细化。

---

## 311. A multi-level preprocessing and modelling framework for spectral imaging of microplastics

**arXiv ID:** 2608.17697 | [PDF](https://arxiv.org/pdf/2608.17697v1)

**作者:** Zina-Sabrina Duma `[一作]` (LUT University), Satu-Pia Reinikainen `[通讯]` (LUT University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种多层级预处理与聚类+谱库匹配的FT‑IR光谱成像微塑料识别框架；

**💡 创新点**

创新点在于同时纠正图像级、块级与光谱级误差，并采用簇质心谱匹配和符号不变导数余弦相似度，显著提升对相似聚合物（如PE/PP）的区分与不确定性评估；

**🔧 技术方法**

使用基线校正（AsLS）、Savitzky‑Golay平滑、SNV归一化、PCA背景校正、块级噪声模板估计、k‑均值聚类、12种谱库匹配策略（M5最佳）以及多种监督学习模型（SVM、RF、NN等）；

**📊 数据集**

利用公开的FT‑IR聚合物谱库（Villegas、Kedzierski、Jung共计~5000条谱）与实验室自制的金膜滤膜上的PS/PET/PE/PP粒子成像数据；

**📈 对比分析**

与市售软件siMPle及12种匹配方法比较，符号不变导数余弦(M5)在簇质心匹配中实现了100%准确率，聚类方案将处理时间从约1.5小时缩短到约2分钟；监督模型在公开数据上准确率>99%但在自制样本上显著下降；

**⚠️ 局限性**

局限于四种聚合物，缺乏对环境降解、污染或不同仪器的泛化；数据集偏移导致模型迁移性能差；计算复杂度随图像尺寸与粒子覆盖率的关系未系统评估；

---

## 312. A Framework for Using and Evaluating LLMs as Surrogate Experts in Security Surveys: Reliability, Bias, and Implications

**arXiv ID:** 2608.16893 | [PDF](https://arxiv.org/pdf/2608.16893v1)

**作者:** Despoina Giarimpampa `[一作]` (University of Luxembourg), Jacques Klein `[通讯]` (University of Luxembourg)

**通讯引用:** 12540 | [OpenAlex ID](https://openalex.org/A5040326968)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构建一个方法学框架，对大语言模型（LLM）在安全运营中心（SOC）专家问卷中的替代与补充效果进行系统评估，探讨其是否能可靠模拟真实专家的回答。

**💡 创新点**

创新点在于提出了可复现的LLM代理参与者评估框架，涵盖个体模拟、聚合分布重现和时间序列一致性三大实验维度，并系统分析LLM回答的稳定性、差异性与人类专家对齐程度，揭示LLM在专家问卷中存在的方差压缩与中心化偏差。

**🔧 技术方法**

采用了提示工程（persona-based与population-based）、多轮采样、类别/多选/李克特量表专属聚合策略，以及基于Jensen–Shannon散度、卡方散度、相似度矩阵等评估指标，结合Python API调用多种LLM（GPT‑4o、GPT‑4、DeepSeek、Llama 3.1‑8B、Llama 3.2‑3B、Gemini Flash、GPT‑3.5‑turbo）。

**📊 数据集**

使用了来自六名SOC专业人员的原始问卷回答以及SANS SOC调查（2017‑2019、2021‑2025）提供的聚合分布作为对照数据集。

**📈 对比分析**

对比方法包括：人类间一致性基准、LLM内在一致性、LLM间相似度热图、LLM‑人类对齐度（JSD、χ²）。实验结果显示：LLM在多次采样中表现出较高的内部一致性，但与人类专家的对齐度低，方差显著降低，存在中心趋向偏差；聚合分布层面虽稳定但与真实分布差距明显，时间序列实验亦未能跟踪人类数据的演变。

**⚠️ 局限性**

局限性包括：受限于小规模人类样本、LLM模型快速迭代导致结果时效性、仅采用零样本提示限制了对模型潜力的充分挖掘、缺乏绝对客观基准以及可能的训练数据偏差导致的代表性不足。

---

## 313. MoNe: Modular Neural Memory for Efficient Long Context Inference

**arXiv ID:** 2608.17616 | [PDF](https://arxiv.org/pdf/2608.17616v1)

**作者:** Wonguk Cho `[一作]` (Qualcomm Ai Research), Sungrack Yun `[通讯]` (Qualcomm Ai Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MoNe模块化神经记忆，可在不重新训练冻结Transformer的前提下，通过线性预处理与常数查询成本实现长上下文推理。

**💡 创新点**

两阶段设计：先在测试时对固定大小分块进行快速权重学习，再在推理时仅查询记忆键值，显著降低O(N²)成本并保持内存不随上下文增长而扩张。

**🔧 技术方法**

结合Fast‑Weight神经记忆、层局部梯度更新、SwiGLU MLP记忆模块、RoPE位置编码、LoRA低秩适配器以及测试时学习（Test‑Time Learning）技术。

**📊 数据集**

在RULER基准的S‑NIAH、MK‑NIAH和Frequent Word Extraction任务上，使用Qwen2.5‑0.5B‑Instruct冻结模型进行评估。

**📈 对比分析**

与ICL（In‑Context Learning）和RAG（Retrieval‑Augmented Generation）对比，MoNe在4K–128K令牌范围内保持近乎完美的子串匹配准确率，且在128K时比ICL减少约80% FLOPs和GPU峰值内存。

**⚠️ 局限性**

目前仅在小模型和受控检索任务上验证，尚未在大规模模型、真实长文档或多模态对话历史上进行评估，且对不同LoRA策略的效果尚待探索。

---

## 314. MANIGUARD: A Benchmark and Data Suite for Specification-Grounded Safety Evaluation and Improvement of Robotic Manipulation

**arXiv ID:** 2608.17386 | [PDF](https://arxiv.org/pdf/2608.17386v1)

**作者:** Yiyan Peng `[一作]` (Northwestern University), Qi Zhu `[通讯]` (Northwestern University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于规范的 Manipulation 安全评估与改进框架 ManiGuard，包括 200 个接触丰富的家庭任务、LTL_f 安全规范、运行时 DFA 监测以及配套的安全标注演示数据集。

**💡 创新点**

创新点在于将任务成功与安全完全分离，使用正式逻辑规范在物理仿真中进行逐步检查，并提供统一的安全标注生成管道，使安全评估与训练信号保持一致。

**🔧 技术方法**

采用 LTL_f、DFA 监控、Isaac Sim/OmniGibson 物理仿真、Franka 机器人硬件、基于 cuRobo 的自动轨迹规划与人机遥控演示生成。

**📊 数据集**

使用 ManiGuard 任务集合（200 个基础任务、1,000 个评估场景）和 8,000 条安全标注演示（每任务 40 条）进行训练与评测。

**📈 对比分析**

对比零射击与微调的 VLAs，主要指标为安全完成率（SSR）、安全率、参与率；微调后 SSR 从接近 0 提升至 7.5–29.8%，安全率显著提升，但仍存在 21–42% 违规的空隙，且在单轴 OOD 和物理机上提升有限。

**⚠️ 局限性**

局限性包括：主要基于仿真，仿真与真实机器的安全评估不完全一致；任务谱仅覆盖六类且规范受限于物理谓词；安全判定依赖仿真状态，未覆盖所有现实风险；演示规模仅对两类任务进行放大实验，未覆盖全部空间。

---

## 315. CORAM: Coherent Orthogonal Rotation for Model Merging

**arXiv ID:** 2608.17366 | [PDF](https://arxiv.org/pdf/2608.17366v1)

**作者:** Xinyi Sui `[一作]` (Santa Clara University), Wei Jiang `[通讯]` (Futurewei Technologies, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 CORAM 的几何融合方法，用来将多个专家模型合并为单一模型，支持冲突检测、分散切片、残差路径以及幅度恢复等模块。

**💡 创新点**

将专家权重拆分为 SVD 三元组（旋转、谱位移、右因子），在各自空间中几何平均，再通过幅度恢复因子校正相消，并引入分散切片和冲突-aware 机制提升兼容性。

**🔧 技术方法**

使用 SVD 分解、旋转对数/指数、幺正投影、幅度归一化、冲突检测、分散切片、残差补丁、幅度恢复系数 κ、SO(h) 旋转投影等技术。

**📊 数据集**

在四个专家合并基准上进行实验：T1 Llama‑3.1‑8B、T2 Llama‑3.2‑3B、T3 Qwen2.5‑VL‑7B‑Instruct 以及 T4 Gemma‑2‑9B，评估包括 MATH500、HumanEval+、GSM8K、MMLU、AGIEval、MMSI‑Bench、CharXiv 等。

**📈 对比分析**

与线性平均、Task Arithmetic、TIES、DARE‑TIES、OrthoMerge 等基线比较，CORAM 在大部分基准中达到最高的在域平均分（如 T1 57.98，T2 43.48，T3 64.38，T4 54.71），且 OOD 兼容性良好；在多重种子评估中误差 ≤0.2 分。

**⚠️ 局限性**

目前仅在低离散度基准上使用 κ=1.15，单一高离散度基准仅 Gemma‑2‑9B；分散切片对安全性可能造成负面影响；幅度恢复系数需经验调参，且整体合并成本相对较高。

---

## 316. Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data

**arXiv ID:** 2608.17248 | [PDF](https://arxiv.org/pdf/2608.17248v1)

**作者:** Berkcan Kapusuzoglu `[一作]` (Vanderbilt University), Sankaran Mahadevan `[通讯]` (Vanderbilt University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

结合物理模型与实验数据，利用机器学习方法开展全局灵敏度分析。

**💡 创新点**

提出两种物理信息融合策略——在损失函数中加入物理约束以及先用物理模拟预训练再用实验数据更新，并将其应用于高斯过程（GP）与深度神经网络（DNN），构建八种模型。

**🔧 技术方法**

采用高斯过程回归和深度神经网络，并通过MC dropout或多重采样估计模型不确定性。

**📊 数据集**

使用增材制造（FFF）零件孔隙率实验数据以及威斯康星州梅多塔湖温度观测数据。

**📈 对比分析**

通过RMSE、十折交叉验证、Sobol指数预测区间等指标对比，DNN模型在精度、置信区间收敛速度和计算成本方面优于GP模型，GP模型训练与预测耗时更高。

**⚠️ 局限性**

局限在高维输入/输出问题、核函数选择、物理约束权重设定以及对不同可信度数据的加权融合等方面，需进一步验证在更大尺度问题上的收敛性和可扩展性。

---

## 317. ComNetX: Local Hierarchical Adaptation for Dynamic Community Detection

**arXiv ID:** 2608.16906 | [PDF](https://arxiv.org/pdf/2608.16906v1)

**作者:** Aleksandr Konovalov `[一作]` (Lomonosov Moscow State University), Grigoriy Bokov `[通讯]` (Lomonosov Moscow State University)

**通讯引用:** 60 | [OpenAlex ID](https://openalex.org/A5047207879)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了ComNetX框架，能够在动态社区检测中只局部更新图而不完全重算，支持任意社区检测器（如Leiden、GNN聚类等）

**💡 创新点**

创新点在于：①统一的、解耦的层次化局部适配器；②通过社区闭包与聚合保留全局上下文；③可同时适配无属性与属性/图神经网络方法

**🔧 技术方法**

使用的技术包括邻域扩展、社区层次闭包、聚合的邻接矩阵与特征矩阵、基于已实现的Leiden、FLMIG、PRGPT、S^2CAG、MAGI、DMoN、MFC、DF-Leiden等后端

**📊 数据集**

实验数据集包含六个真实动态网络（dyn_cora、dyn_acm、dyn_citeseer、patent、dyn_pubmed、arxivmath）以及100批次的DSBM合成数据

**📈 对比分析**

与全重算和原生动态方法对比：在大图上Local Leiden可获得约41×的加速，同时保持模数差≤0.006；在其他后端也能显著减少内存/时间；与DF-Leiden、MFC等原生动态解法对比，Local版在部分数据上进一步提升速度或质量

**⚠️ 局限性**

局限性包括：当受影响社区规模接近全图、更新集中在高度连通或中心节点时，局部化失效；聚合过程中可能丢失细粒度特征上下文，导致特征感知后端的质量下降；需要监测本地工作量以决定是否退回全重算

---

## 318. RoBell-RVFL: A Robust Generalized Bell Random Vector Functional Link Network

**arXiv ID:** 2608.16965 | [PDF](https://arxiv.org/pdf/2608.16965v1)

**作者:** A. Rahaman `[一作]` (Indian Institute of Technology Indore), M. Tanveer `[通讯]` (Indian Institute of Technology Indore)

**通讯引用:** 7925 | [OpenAlex ID](https://openalex.org/A5004222223)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RoBell‑RVFL 模型，解决不平衡数据、标签噪声与异常点的联合挑战；通过样本级自适应权重实现对多数类噪声抑制与少数类信息完整保留。

**💡 创新点**

创新点：① 双策略样本级权重——少数类全权重、最多数类采用概率加权通用贝尔成员函数；② 结合核映射与局部类别概率，形成概率加权通用贝尔（PW‑GB）成员函数；③ 通过类别相关正则化隐式处理不平衡，无需显式比例权重。

**🔧 技术方法**

技术路线：随机向量功能链接网络（RVFL）+ 高维核映射 + 通用贝尔成员函数 + 局部类别概率 + KKT 求解闭式解 + 目标正则化。

**📊 数据集**

使用 26 个 UCI 与 KEEL 基准分类数据集，涵盖二分类与多分类任务。

**📈 对比分析**

与 RVFL、RVFLwoDL、NF‑RVFL、C‑RVFL、AC‑RVFL、GB‑RVFL、GE‑GB‑RVFL 等 8 种模型对比；采用平均准确率与平均排名评估，RoBell‑RVFL 平均准确率 81.89%，平均排名 3.23，显著优于所有竞争方法；在 40% 标签噪声条件下仍保持性能优势，统计检验（Friedman、W–T–L）表明差异显著。

**⚠️ 局限性**

局限性：① 对多类别大规模数据的扩展尚未评估；② 对核宽、ρ、γ 等超参数敏感；③ 未探索在线增量学习场景；④ 在极端噪声或极端不平衡下仍可能出现性能衰退；⑤ 计算复杂度略高于传统 RVFL。

---

## 319. Without journalists, there is no journalism: the social dimension of generative artificial intelligence in the media

**arXiv ID:** 2608.17017 | [PDF](https://arxiv.org/pdf/2608.17017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 320. CoinVE-200K: A Large-Scale High-Quality Dataset for Compositional Instruction-Guided Video Editing

**arXiv ID:** 2608.17566 | [PDF](https://arxiv.org/pdf/2608.17566v1)

**作者:** Fuchen Long `[一作]` (Tencent), Yu Liu `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个规模达200K的视频编辑数据集 CoinVE-200K，涵盖每个视频 2~5 个原子编辑操作，支持人、物体和背景的多主体、多类型编辑；同时设计 CoinVE-Bench 基准评测和 22B 模型 CoinVE-Edit；

**💡 创新点**

创新点在于（1）提出多意图编辑的数据构造与高质量筛选流程，提升多操作编辑的可行性；（2）设计 Q-Blending 交叉注意力和 Mask Predictor 使模型能按区域解耦多指令；（3）通过 MLLM 先理解指令，再注入 DiT 生成，兼顾语义与视觉一致性；

**🔧 技术方法**

技术上使用 Qwen3-VL‑8B‑Instruct 作为多模态大语言模型进行指令理解，配合 Wan2.1‑T2V‑14B Diffusion Transformer 做视频生成，并加入 Mask‑Predictor、GateNet、Q-Blending 进行区域化控制；

**📊 数据集**

使用数据集包括 OpenVid‑HD、OpenVE‑3M、Ditto‑1M、ReCo、Pico‑Banana‑400K 等；核心数据是自建的 CoinVE‑200K；

**📈 对比分析**

通过 Gemini 2.5 Pro 评估，CoinVE-Edit 在 CoinVE‑Bench 上在编辑准确度、物理自然性和语义保留上均优于现有开源模型，且与闭源 Seedance 2.0、Kling O3 相比也表现突出；

**⚠️ 局限性**

局限性在于未覆盖所有编辑场景（如参考编辑、细粒度运动、复杂相机控制）和长视频高效编辑，未来需扩展编辑范畴、提升长时段一致性与算力效率。

---

## 321. J-Miner: Recovering Executable Decision Knowledge from Language-Model Classifiers

**arXiv ID:** 2608.17063 | [PDF](https://arxiv.org/pdf/2608.17063v1)

**作者:** Yunfan Gao `[一作]` (Shanghai Research Institute for Intelligent Autonomous Systems, Tongji University), Haofen Wang `[通讯]` (Tongji University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过从微调的LLM分类器内部提取命名概念，并将其组合成可执行的规则，从而恢复并显式化模型的决策知识。

**💡 创新点**

创新点在于：①利用Jacobian lens将内部表示映射到词汇空间，自动识别与任务相关的概念；②在概念层面构造可解释的布尔规则或加权分数卡；③实现知识迁移，使轻量化学生模型能仅凭概念与固定规则完成推理，保持高行为一致性。

**🔧 技术方法**

主要技术包括：J-Lens（Jacobain lens）读出、概念聚合与筛选、布尔AST规则或线性分数卡学习、概念重建损失与可选的判别辅助训练，以及与学生模型的迁移学习。

**📊 数据集**

使用九个英文文本分类数据集：SMS Spam、Sentiment、Formality、IMDB Sentiment、Toxicity、Sarcasm、HateXplain、SNIPS‑3、SNIPS‑7，并在多种规模与架构的预训练模型上进行评估。

**📈 对比分析**

在相同特征预算下，J‑Miner的规则与传统基于词表的规则相比，教师模型的行为一致性提升6–29个百分点；在不同模型规模与族群下，教师一致性提升11–35个百分点；轻量化学生在保持近似原模型准确率的同时，教师决策一致性达到92%以上。

**⚠️ 局限性**

局限性包括：①依赖冻结的教师模型和J‑Lens读出，需额外校准数据；②目前仅针对二分类与多分类文本标注任务，对更复杂推理或多模态任务的适用性未知；③概念选择和规则学习的超参数调优可能影响结果，且对极端类不平衡情况的鲁棒性尚待验证。

---

## 322. To Remove or Not to Remove Clouds: A Comparative Analysis and Fusion of Raw SAR and Synthetic NDWI for Overcast Water Segmentation

**arXiv ID:** 2608.17398 | [PDF](https://arxiv.org/pdf/2608.17398v1)

**作者:** Saleh Sakib Ahmed `[一作]` (Bangladesh University of Engineering and Technology), M. Sohel Rahman `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究提出一种将原始SAR影像与经过深度学习生成的合成NDWI联合分割水体的多模态框架，用于在完全云覆的条件下实现洪水监测。

**💡 创新点**

创新点在于：①将SAR‑to‑NDWI翻译视作对雷达噪声的结构性过滤；②在保持原始SAR物理边界的同时融合合成NDWI的高对比度信息，从而形成双向纠错的融合网络；③通过迁移学习的预训练SegFormer编码器实现参数高效的中间生成。

**🔧 技术方法**

技术手段包括：SegFormer预训练编码器+自适应输入适配器；混合L1+FocalMSE的回归损失；BCE+SoftDice的分割损失；交叉验证、混合与事件分层划分、统计显著性检验。

**📊 数据集**

使用微软Cloud2Street洪水数据集（900对Sentinel‑1 SAR与Sentinel‑2光学影像，512×512像素）并按不同tile尺寸（128×128、256×256）进行切块与误差阈值过滤。

**📈 对比分析**

与单模态（仅SAR、仅NDWI）以及传统PCA融合对比，融合模型在Mixed CV下IoU提升至0.8347、F1提升至0.9099，优于纯SAR（IoU≈0.8049）和纯NDWI（IoU≈0.8285），且在事件分层下仍保持显著优势。

**⚠️ 局限性**

主要限制包括：对更大空间尺度（256×256）需要更高容量的回归网络，轻量级配置易出现生成失真；模型对极端雷达噪声或生成器幻觉仍有鲁棒性挑战；实验基于单一数据集，跨区域泛化需进一步验证。

---

## 323. Abra: Scaling Diffusion Image Training

**arXiv ID:** 2608.17286 | [PDF](https://arxiv.org/pdf/2608.17286v1)

**作者:** Kyle Chickering `[一作]` (Luma AI), Xinchen Yan `[通讯]` (Luma AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统研究了文本到图像扩散模型的计算最优缩放规律，构建了从60M到2B参数的控制型变换器族，覆盖3个数量级的 FLOPs，并通过 μP 参数化实现超参数迁移，最终推导出 200 TPP（图像令牌/参数）的计算最优点；

**💡 创新点**

创新点在于：①首次为文本到图像扩散模型提供精准的计算最优缩放公式；②发现扩散训练对过度训练具有极高容忍度；③证明多种生成指标（FID、KID、CLIPScore、CMMD）与线性探针精度均满足可预测的缩放律；④首次观察到扩散模型训练曲线的缩放坍塌现象；⑤剖析分辨率对计算最优 TPP 的影响。

**🔧 技术方法**

核心技术包括：流匹配 transformer 架构、v‑prediction 流量匹配目标、EMA 权重、Classifier‑Free Guidance、μP 参数化、PCHIP 插值、损失与生成指标的功率律拟合、缩放坍塌分析公式。

**📊 数据集**

使用了 DataComp‑1B 大规模图文对数据集，并对图像进行了重新标注；此外还在多种分辨率（256、384、512、768）下训练验证。

**📈 对比分析**

与前人仅拟合到 10^21 FLOPs 的研究相比，本文使用十倍计算量和更受控的模型族，验证了 200 TPP 计算最优点；在相同 FLOPs 下，扩散模型的生成质量（FID、KID）和表示质量（线性探针）均随规模提升而改进，且不同指标的最优 TPP 存在差异。

**⚠️ 局限性**

局限性：仅覆盖至 2B 参数的模型族，无法验证更大规模的行为；分辨率实验仅在 500M 参数以内完成，未能在更大模型上验证高分辨率下的缩放；缩放坍塌验证仅限于 250M 参数及以下；模型对过度训练的容忍度虽高，但不适用于 LLM 训练场景；研究聚焦于预训练阶段，缺乏后续微调或推理效率的评估。

---

## 324. The Problem Is the Problem: Towards Scalable Mathematical Discovery

**arXiv ID:** 2608.16977 | [PDF](https://arxiv.org/pdf/2608.16977v1)

**作者:** Zeyu Zheng `[一作]` (Carnegie Mellon University), Sean Welleck `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Find-Attempt-Recommend (FAR) 流程，将 AI 助力从传统的单个问题求解转移到研究方向级别的文献检索与筛选，并在组合数学领域实现了从 51,110 篇论文中识别 4,717 个开放 conjecture，进行一次性推理后产生 77 个可出版的科研成果，作者对其中 15 项进行了人工验证，全部无误。

**💡 创新点**

创新点在于：① 将 AI 介入点从单个问题迁移到研究方向；② 设计了文献到评审的三级递进式回调 (Find、Attempt、Recommend)；③ 利用模型在检查阶段生成的难度与重要性得分，构造基于预算的最优工作分配策略；④ 在实验中展示了该策略相较于均匀分配能显著提升成功率与重要性得分。

**🔧 技术方法**

技术手段包括：多阶段 LLM 推理（Label、Extract、Check、Attempt、Judge、Grade）；检索与推荐系统框架；基于难度/重要性得分的统计学习与最优化（最小二乘拟合、贪婪子集选择）；以及有限预算下的 Bandit 资源分配与子模最优近似。

**📊 数据集**

数据集为 51,110 篇数学论文（OpenAlex 公开元数据及源链接），通过 FAR 处理得到 6,453 条候选 conjecture（来自 2,742 篇），过滤后 4,717 条可尝试 conjecture，最终产生 77 条可发表成果。

**📈 对比分析**

与基准（随机均匀分配）比较，利用难度/重要性得分排序的策略在 f₁（产出数量）与 f₂（总重要性）上均显著优于基准，AUC 证明得分与实际难度/重要性高度相关；在实际分配预算 10–300 次的交叉验证中，得分排序策略获得最多 77 个可发表成果，说明其有效性与稳健性。

**⚠️ 局限性**

局限性包括：① 依赖于所选文献语料与 LLM 的质量，若语料或模型不佳可能漏检已知结果；② 仅在组合数学领域验证，跨学科推广需进一步实验；③ 对每个 conjecture 仅做一次尝试，可能低估模型在多次尝试后的性能；④ 高昂的算力需求与人工审阅仍是瓶颈；⑤ 结果受难度/重要性得分的偏差影响，未对多模态或结构化推理进行深入探讨。

---

## 325. Learning What Not to Learn: Adversarial Disentangled Prompt Tuning for Robust Vision-Language Models

**arXiv ID:** 2608.17306 | [PDF](https://arxiv.org/pdf/2608.17306v1)

**作者:** Yang Chen `[一作]` (Southern University of Science and Technology), Yu Zhang `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为ADAPT的鲁棒提示调优框架，通过双提示机制（目标提示和诱饵提示）以及正交损失实现对伪鲁棒特征的分离，从而提升视觉-语言模型在未见类别上的鲁棒性。

**💡 创新点**

创新点在于：①首次揭示并定义了“鲁棒泛化过拟合”现象；②设计了诱饵提示池以捕获多样的伪鲁棒特征；③引入正交损失与语义一致性约束，强制目标提示与诱饵提示正交，达到特征解耦。

**🔧 技术方法**

采用CLIP预训练模型，结合对抗样本生成（PGD）、连续提示向量学习、正交损失、相似度损失和语义一致性损失等技术实现鲁棒提示调优。

**📊 数据集**

在11个公开数据集（ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101）以及ImageNet的OOD版本上进行实验。

**📈 对比分析**

与零样本方法TeCoA、单模态对抗提示调优APT、多模态对抗提示调优FAP以及其多模态版本ADAPT_M进行对比，ADAPT在“鲁棒泛化”指标（H_b）上平均提升约5.6%，在鲁棒性和准确率上均优于现有方法，且在少样本、不同攻击（PGD、CW、AA）下均保持领先。

**⚠️ 局限性**

局限性包括：仅在CLIP框架下验证，未评估对更大规模或不同VLM架构的适应性；诱饵提示池和多损失项增加了训练复杂度；对极端分布漂移（跨域、跨数据集）和实时部署场景的鲁棒性仍待进一步验证。

---

## 326. Secret Sharing at the Shannon Ceiling

**arXiv ID:** 2608.17047 | [PDF](https://arxiv.org/pdf/2608.17047v1)

**作者:** Christopher Williamson `[一作]` `[通讯]`, Christopher Williamson

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

构造了一个多参与者的秘密共享访问结构，并证明了其总共享信息量和最大共享信息量的下界。

**💡 创新点**

创新点在于通过简单的Shannon不等式与平均化论证，提升了已知最优下界的对数因子，得到总共享量Ω(n²)与最大共享量Ω(n)的极限。

**🔧 技术方法**

主要使用了信息熵方法、Shannon不等式、对称群平均与轨道论证等基本信息理论技术。

**📊 数据集**

论文未使用具体实验数据集，而是以理论构造和证明为主。

**📈 对比分析**

通过与Csirmaz等人先前给出的Ω(n²/ log n)和Ω(n/ log n)下界对比，显示了在理论下界上明显的提升；但未给出实际性能指标。

**⚠️ 局限性**

局限性在于仅适用于完美秘密共享模型，且仅给出了下界，未提供相应上界或具体实现方案。

---

## 327. Environment-Invariant Subspace Learning for Generalizable Deepfake Detection

**arXiv ID:** 2608.17700 | [PDF](https://arxiv.org/pdf/2608.17700v1)

**作者:** Shenghao Chen `[一作]` (Tianjin University of Technology), Shengyong Chen `[通讯]` (Tianjin University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

论文提出一种环境不变子空间学习（EISL）框架，通过对CLIP特征进行低秩投影与环境干预，提升深度伪造检测模型在不同环境和生成器上的泛化性能。

**💡 创新点**

创新点在于：1）设计可控的环境干预模块（EIM）生成标签保持的环境扰动对，帮助模型学习环境不变特征；2）引入可学习的低秩正交投影子空间，强制把环境敏感方向分离出去，从而实现结构化的特征分离；3）将参考CLIP特征与投影特征对齐，保持语义一致性。

**🔧 技术方法**

使用的技术包括：CLIP ViT-L/14 视觉编码器（冻结后通过LoRA微调）；低秩投影矩阵 Q 及其投影 P = QQᵀ；三项损失：语义对齐（RefCLIP）、环境一致性（Inv）、正交性正则；环境干预模块基于随机掩码与颜色、模糊等增强；实验采用Adam优化器、10轮训练、批量16。

**📊 数据集**

数据集涵盖八个公开深度伪造数据集：FaceForensics++、Deepfake Detection Challenge (DFDC)、DFDCP、CelebDF v1/v2、DeepFakeDetection (DFD)、DF40、Diffusion Facial Forgery (DiFF)。

**📈 对比分析**

与多种基线（Xception、EfficientB4、F3Net、FFD、SPSL、SRM、Recce、SBI、UCF、ED、LSDA、CFM、ForAda、FIA‑USA、Effort 等）以及 CLIP‑LoRA 进行对比，EISL 在跨数据集、跨生成器、全脸合成和图像腐蚀等场景下均取得最高或第二高的 AUC，平均提升约 3–5% 以上。

**⚠️ 局限性**

局限性包括：仅针对标签保持的环境扰动；子空间投影是全局线性且固定秩，可能无法捕捉非线性交互；未涵盖身份偏差、数据集特定处理、生成器特定痕迹等其它分布偏移；缺乏对更丰富或自适应干预、非线性子空间的探索。

---

## 328. Structured Driving-State Narratives for Small Language Model-Based GNSS Spoofing Detection

**arXiv ID:** 2608.17092 | [PDF](https://arxiv.org/pdf/2608.17092v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 329. Rapid Debris-Volume Estimation from Post-Hurricane Aerial Imagery

**arXiv ID:** 2608.17165 | [PDF](https://arxiv.org/pdf/2608.17165v1)

**作者:** Kooshan Amini `[一作]` (Rice University), Guha Balakrishnan `[通讯]` (Rice University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于单幅后灾RGB航拍影像的碎片高度回归网络DebrisHeightNet，并通过碎片分割作为条件实现碎片体积估计。

**💡 创新点**

创新点包括：1）条件化单目高度回归网络；2）使用自生成的Confidence‑Weighted LiDAR‑Monocular Fusion (CW‑LMF)合成监督；3）提供区域级幂律校准并量化不确定性；4）在十个跨风暴区域上进行全面评估。

**🔧 技术方法**

技术栈包含：冻结的Vision Foundation Models（Depth Anything V2 作为深度估计，CLIPSeg‑debris 作为碎片分割），轻量级U‑Net头，异方差损失，LiDAR与单目融合，及幂律校准。

**📊 数据集**

数据集包括NOAA Emergency Response Imagery、USACE/NOAA LiDAR、1 m DEM、Estero岛的UAV SfM调查、以及来自公共记录的报废量数据，共覆盖十个沿海区域、五场飓风。

**📈 对比分析**

与独立UAV、原始LiDAR、以及行业参数化模型（Hazus、FEMA混合、USACE）对比，未校准模型相对报告量误差≤30%，校准后误差从2.7×降至1.9×；与UAV的Spearman ρ为0.87，整体体积比接近1，显著优于传统方法。

**⚠️ 局限性**

局限性：1）训练目标仅为校验而非真值；2）单源影像导致尺度漂移，需重新校准；3）校准基于10个地区，低密度碎片场景可靠性低；4）像素级空间相关性有限；5）LiDAR获取延迟与覆盖不完全。

---

## 330. Certified but Private: Scalable Zero-Knowledge Proofs for Neural Network Guarantees

**arXiv ID:** 2608.17070 | [PDF](https://arxiv.org/pdf/2608.17070v1)

**作者:** Youwei Zhong `[一作]` (Yale University), Ruzica Piskac `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出了PANDA系统，利用零知识证明实现对私有神经网络局部鲁棒性的公开验证。

**💡 创新点**

创新点在于设计了四点松弛装置（Four‑Point Relaxation Gadget），能够在零知识框架下高效证明非线性激活函数的线性松弛，并将CROWN鲁棒性算法与定制化ZKP后端相结合，实现了对数百万参数网络的可扩展证明。

**🔧 技术方法**

使用的技术包括CROWN鲁棒性证明、量化技术、零知识承诺与证明（Polynomial Commitments、矩阵算术、表查找）、Four‑Point Relaxation Gadget以及专门的定制化ZKP后端。

**📊 数据集**

实验使用了MNIST、SafeNLP、LunarLander以及FairProof Adult等公开数据集。

**📈 对比分析**

与现有的FairProof等系统对比，PANDA在大规模网络上证明时间在6分钟以内，验证时间约10秒；在私有性开销上相对非ZKP增加约100–1000倍，但在可扩展性、激活函数支持以及证明规模方面明显优于现有方法。

**⚠️ 局限性**

局限性包括只能对量化模型进行证明，无法支持原始浮点；公开了网络架构、层维度和激活类型；以及由于CROWN的近似性质，鲁棒性证明存在量化误差。

---

## 331. The Last Mile of Deepfake Speech Detection: An Industry-Academia Experience Report

**arXiv ID:** 2608.17585 | [PDF](https://arxiv.org/pdf/2608.17585v1)

**作者:** Anton Firc `[一作]` (Brno University of Technology), Marek Bartoň `[通讯]` (Phonexia)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开展了为期三年的产学研合作，构建并部署了一套商业级合成语音检测系统，并系统性总结了从数据获取、评估设计、泛化能力、校准与分数解释，到客户验收与治理等多方面的技术与组织障碍。

**💡 创新点**

首次将真实部署经验与学术研究结合，提出了基于实践的障碍清单和“路线图”，包括对数据来源与许可、评估标准与泛化验证、校准与阈值设定、分数解释与沟通、以及治理与合规等方面的研究、方法与协同行动建议，填补了现有文献中缺乏部署视角的空白。

**🔧 技术方法**

采用预训练自监督学习（SSL）前端+注意力池化的检测模型；使用多种声学特征、端到端推理；通过片段级处理最大化分数；在评估中加入频道、编码、背景噪声等数据增强；校准使用默认或经验校准方案。

**📊 数据集**

构建了包含30名说话人、多语种、多合成工具（如 ElevenLabs、其他TTS）、不同音频编码（AMR-NB、G.711 等）以及背景噪声的自研混合数据集；部分数据来源于公开语料、内部合成和 YouTube，但存在许可限制与质量漂移。

**📈 对比分析**

通过内部评估对比 seen 与 unseen 攻击，发现检测准确率从 0.9135（seen）提升至 0.9381（unseen），但在不同频道/编解码下误检率从 4% 上升至 60%；整体表明虽然在标准评估中可达 <1% EER，但在真实部署场景下表现显著下降，且常用的 EER/DCF 指标与实际阈值不匹配。

**⚠️ 局限性**

主要限制包括：数据许可与来源缺乏透明度；评估缺乏统一标准，泛化验证难以实现；缺少客户标注数据导致校准与阈值设定不准确；分数解释不易被非专业人员理解；系统集成与部署控制受限；模型对通道、编解码与合成器演化高度敏感；缺乏统一、可验证的数据集与评估规范。

---

## 332. Robust Brachiation on a Life-Sized Dual-Arm Robot Using Waypoint-Guided Reinforcement Learning

**arXiv ID:** 2608.17320 | [PDF](https://arxiv.org/pdf/2608.17320v1)

**作者:** Ayumu Iwata `[一作]` (University of Tokyo), Kei Okada `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

无法获取论文具体内容，无法说明研究的主要工作。

**💡 创新点**

缺乏可识别的创新点信息。

**🔧 技术方法**

未提供使用的技术细节。

**📊 数据集**

未提及所使用的数据集。

**📈 对比分析**

未说明比较方法与性能指标。

**⚠️ 局限性**

未知，无法评估研究的局限性。

---

## 333. Beyond the Hype: Evaluating LLM Integration and Practical Limitations in Security Operation Centers

**arXiv ID:** 2608.17154 | [PDF](https://arxiv.org/pdf/2608.17154v1)

**作者:** Elnaz Rabieinejad `[一作]` (University of Guelph), Sarina Dastgerdy `[通讯]` (University of Guelph)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 20 名 SOC 从业者进行半结构化访谈，评估大型语言模型在安全运营中心（SOC）中的使用场景、效益、挑战、验证负担以及组织的准备度，并基于访谈结果构建失效模式分类、幻觉缓解成熟度评估表和 SOC 整合约束矩阵。

**💡 创新点**

首次在 SOC 实务中系统化梳理 LLM 的工作流程嵌入方式与失效模式，提出针对幻觉的成熟度评估框架和可操作的整合约束矩阵，强调人机协同与验证机制的必要性。

**🔧 技术方法**

采用定性研究方法：访谈、主题编码（Braun & Clarke）、互评一致性（Cohen’s κ）以及主题映射；并未使用机器学习模型，而是对访谈文本进行编码与归纳。

**📊 数据集**

使用受访者提供的访谈记录和工作流程描述作为数据源，未使用公开数据集。

**📈 对比分析**

本研究不做性能对比或量化评估；主要通过访谈记录中受访者的主观感知和验证负担描述来说明 LLM 的效益与风险。

**⚠️ 局限性**

局限性包括样本规模小且地域集中（主要为安大略省），未对不同 LLM 模型或部署方式进行对比，缺乏客观指标衡量效能，且可能存在受访者自报偏差及组织治理与合规性影响的不足。

---

## 334. MAG-Bot: A Multi-Agent Auditing Framework for Social Bot Detection

**arXiv ID:** 2608.16908 | [PDF](https://arxiv.org/pdf/2608.16908v1)

**作者:** Sichen Zhao `[一作]` (Northeastern University), Yalun Qi `[通讯]` (Northeastern University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5108642920)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究将社交媒体账号视为可审计档案，构建多代理LLM框架MAG‑Bot，验证其对社交机器人的检测效果。

**💡 创新点**

创新点在于提出零射击的角色约束多代理拆分策略，强调证据视图的可审计性，并证明拆分而非单一LLM或辩论可显著提升召回。

**🔧 技术方法**

使用大语言模型与LangGraph实现多代理工作流，包含行为、语言、上下文专家及元判决者。

**📊 数据集**

使用重新构造的TwiBot‑22数据集，将图结构转换为JSON档案进行实验。

**📈 对比分析**

与传统特征基分类器、单一LLM审计和多代理版本比较，MAG‑Bot在测试集上准确率0.702、召回0.829、F1 0.803，显著高于单一LLM并接近监督基线。

**⚠️ 局限性**

局限在于仍落后于最强监督基线、对机构化真实用户产生误报、以及更高的推理时延和令牌消耗。

---

## 335. Digital Twin-Based Intrusion Detection for Vehicle Powertrain CAN Bus Systems

**arXiv ID:** 2608.17093 | [PDF](https://arxiv.org/pdf/2608.17093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 336. SW-ProxyCE: Zero-Query Adversarial Transfer from Public EEG Encoders to Private Downstream Models

**arXiv ID:** 2608.16931 | [PDF](https://arxiv.org/pdf/2608.16931v1)

**作者:** Linhua Cong `[一作]` (Huazhong University of Science and Technology), Dongrui Wu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 15632 | [OpenAlex ID](https://openalex.org/A5008740867)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究公开EEG基础模型在公共编码器、私有下游模型场景下的对抗攻击风险，并提出一种无需查询下游模型的攻击框架SW-ProxyCE。

**💡 创新点**

创新点包括：①利用收缩白化的原型几何构建任务感知的代理目标；②无需训练额外代理分类器即可从少量参考样本恢复类间竞争信息；③证明公共编码器可作为跨系统的共享攻击面。

**🔧 技术方法**

技术手段主要是：Shrinkage‑Whitened Proxy Cross‑Entropy (SW‑ProxyCE) 目标、Fast Gradient Sign Method (FGSM) 生成对抗扰动、基于原型的非参数代理以及收缩白化的方差正则化。

**📊 数据集**

实验数据集：BNCI2014001（四类运动想象）、SEED（三类情绪识别）、CHB‑MIT（二类癫痫检测）。

**📈 对比分析**

与任务无关攻击(TAA)和高斯噪声基线对比，使用BACC评估。SW‑ProxyCE在40种配置下平均降低BACC约22.9个百分点，显著优于TAA（约5.9个百分点）和噪声；对线性探测（LP）模型降幅更大，对全微调（FT）模型也保持有效。

**⚠️ 局限性**

局限性：仅评估数字域一次攻击，未覆盖物理实施场景；依赖公共编码器的公开性；对不同下游适配策略（如多任务、跨模态）的进一步鲁棒性尚未系统探究。

---

## 337. Delta2Gamma: Band-Wise Adaptive Contrastive Learning of EEG for Alzheimer's Disease Detection

**arXiv ID:** 2608.17231 | [PDF](https://arxiv.org/pdf/2608.17231v1)

**作者:** Chanwoo Park `[一作]` (Korea University), Chanwoo Kim `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了Delta2Gamma框架，通过将脑电信号分解为五个频带，分别训练自监督对比学习来实现阿尔茨海默病检测。

**💡 创新点**

创新点在于为每个频带设计独立的CNN编码器、投影头，并在对比学习中引入可自适应学习的温度参数，兼顾各频带不同的信号统计特征。

**🔧 技术方法**

使用了SimCLR风格的自监督对比学习、带有温度正则化的NT-Xent损失、数据增强（加噪、幅度缩放、时域/频域遮掩）、多层CNN编码器以及在冻结编码器上训练的线性分类器。

**📊 数据集**

实验数据来源于公开的ADFTD数据库，包含88名受试者的静息态闭眼EEG（36名阿尔茨海默病，29名健康对照，23名额外未标记的FTD）。

**📈 对比分析**

在严格的留一受试者交叉验证（LOSO）下，与多种监督和自监督EEG基线模型（如ATCNet、EEGNet、EEGConformer等）进行比较，Delta2Gamma实现了92.4%的准确率和92.3%的F1分数，明显优于所有对比方法。

**⚠️ 局限性**

局限性包括仅针对静息态EEG、主要关注AD与CN对比，难以直接推广到其他疾病或任务；同时对频带划分和温度预测模型的依赖可能导致跨域适应性不足。

---

## 338. Quantifying Risk Under Evolving Uncertainty: Belief-Dependent Robustness for Safe Sequential Decision Making

**arXiv ID:** 2608.17574 | [PDF](https://arxiv.org/pdf/2608.17574v1)

**作者:** Deep Kumar Ganguly `[一作]` (Technical University of Munich), Jan Kretinsky `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种基于贝叶斯信念熵调节Wasserstein模糊度的风险自适应强化学习框架（RATTL），使智能体在学习环境时逐步降低保守性。

**💡 创新点**

创新点在于：①将贝叶斯信念熵映射为Wasserstein模糊半径，实现从最坏情况到完全信息的连贯风险曲线；②给出安全三明治（Safety Sandwich）理论，证明值函数始终处于最坏情况下限与贝叶斯最佳响应上限之间；③在两点灾难情形下证明Wasserstein风险等价于CVaR，并用熵控制尾部水平；④提供完整的收敛、收敛速率与可计算性分析。

**🔧 技术方法**

主要技术：贝叶斯推断、Wasserstein距离与DRO对偶、EVaR/CVaR风险度量、线性规划求解、强化学习中的价值迭代与Bellman算子。

**📊 数据集**

论文主要为理论与仿真，未使用公开数据集；通过一个“模糊桥梁”示例展示阈值切换。

**📈 对比分析**

与传统稳健MDP、纯贝叶斯RL以及固定风险策略相比，RATTL在保守性与收益之间取得平衡；在示例中显示在信念熵降低时，价值从最坏下限平滑提升至最佳上限，且安全切换阈值可解析计算。

**⚠️ 局限性**

局限性：仅适用于离散、有限状态/动作/类型的表格问题；需满足统一可达性与持续识别等强假设；Wasserstein风险的CVaR等价仅在两点灾难案例成立；贝叶斯信息度量仅限于Shannon熵，未验证其他熵形式；对大规模连续空间及样本复杂度分析尚未完成。

---

## 339. Learning Where and What to Lift for Bi-planar X-ray-to-CT Reconstruction

**arXiv ID:** 2608.17255 | [PDF](https://arxiv.org/pdf/2608.17255v1)

**作者:** Yifei Wu `[一作]` (Northwestern Polytechnical University), Yong Xia `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了 LiftXR，一个交替进行几何布局生成与体素强度重建的框架，用双平面 X 光图像恢复 3D CT 体积。

**💡 创新点**

创新点在于：① 将解剖结构布局显式建模为先验约束；② 在布局生成后通过体素渲染得到粗糙 CT，再用解析得到的布局进行细化；③ 交替训练实现布局与强度的互补校正，显著提升解剖一致性。

**🔧 技术方法**

采用 3D U‑Net 作为骨干，CNN‑GAN 结构实现强度渲染，使用多尺度特征匹配损失、MAE、对抗损失以及交替训练策略；生成器先基于 X 光体积做布局预测，再渲染 CT，随后用解析模块细化布局并校正强度。

**📊 数据集**

实验使用公开胸部 CT 数据集 CT‑RATE（2000 训练 / 300 测试）和 LIDC‑IDRI（818 训练 / 200 测试），并在 MIMIC 真实 X 光上评估跨域泛化。

**📈 对比分析**

与 X2CT、PerX2CT、DSDF、GAAL、DX2CT、DiffuX2CT 等方法对比，LiftXR 在 PSNR、SSIM、Dice、HD95 等指标均实现领先（PSNR 26.45 dB / SSIM 76.44% / Dice 68.6% / HD95 12.5 voxels），并在下游分割任务上获得更高的准确度。

**⚠️ 局限性**

局限性包括：① 仅基于两幅 X 光，信息稀疏仍限制极细结构恢复；② 交替训练和三维生成导致计算量大；③ 需进一步验证临床实际效用和对小病灶的重建能力。

---

## 340. Do LLMs Know a Good Hypothesis When They See One? Logit-Based Energy Scoring Outperforms Prompted LLM-as-Judge for Scientific Hypothesis Ranking

**arXiv ID:** 2608.17270 | [PDF](https://arxiv.org/pdf/2608.17270v1)

**作者:** Swati Rajwal `[一作]` (Oak Ridge National Laboratory), Tirthankar Ghosal `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对每篇论文的背景和研究问题进行条件化，利用大型语言模型对 16 条候选假设的 token‑level 置信度进行 logit‑based 能量评分，以自动识别科学假设的正确性。

**💡 创新点**

提出了直接使用模型自身未归一化 logit 能量作为评估指标（Raw 能量），并证明其在多模型、跨学科情境下可优于传统的 LLM-as-Judge 语言提示方法。

**🔧 技术方法**

使用了 softmax 负对数似然（NLL）和原始 target‑logit 能量（Raw）两种评分方式，对 7 种公开权重模型（1B–20B 参数）进行评估，并对 GPT‑5 进行零样本 listwise 提示对比。

**📊 数据集**

评估基于 ResearchBench 数据集，该数据集包含 1,323 篇论文，覆盖 12 个学科，每篇论文提供背景、研究问题及 16 条候选假设，其中一条为 gold 假设。

**📈 对比分析**

实验表明，采用 logit‑based 能量评分的开放模型（尤其是 Llama 3.2 1B Raw）在 Hit@1 上达到 53% 以上，显著优于 GPT‑5 的 16.6%；整体来看，内部置信度评分比提示排序在大多数模型和学科中均更具鲁棒性和准确性。

**⚠️ 局限性**

局限性包括：数据集仅包含已发表论文的已知假设，可能受记忆化和写作流畅度影响；Raw 能量与 NLL 在不同模型间一致性差异较大；实验仅测试了单一提示策略与单一专有模型，未覆盖更广泛的模型和提示变体。

---

## 341. SpecTrum: Specification-Guided Differential Fuzzing for Ethereum Consensus Clients

**arXiv ID:** 2608.17738 | [PDF](https://arxiv.org/pdf/2608.17738v1)

**作者:** Seokhun Jeong `[一作]` (KAIST), Sungjae Hwang `[通讯]` (Sungkyunkwan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个基于以太坊共识规范的差分测试框架——SpecGuard，该框架先将以太坊共识规范机械化为带有显式 if‑premise 的 DSL 规范，随后定义并测量 premise 代码覆盖率，并利用该覆盖率指导测试生成，从而在五大客户端（Lighthouse、Lodestar、Nimbus、Prysm、Teku）之间发现跨客户端分歧。

**💡 创新点**

创新点在于：①首次将以太坊共识规范机械化并显式化所有隐式的有效性条件；②提出 premise 覆盖率（每个 if‑premise 同时被评估为 true 与 false 的度量），可检测传统代码覆盖无法捕获的边界；③基于 premise 覆盖率的自动化测试生成器，利用 provenance 传播、建议提取和区间采样精确触发未被覆盖的有效性条件，从而大幅提升 bug 探测能力。

**🔧 技术方法**

技术手段包括：P4‑SpecTec DSL 与工具链（实现规范解析、解释器生成、规则检查）；覆盖率引擎收集 if‑premise 的 true/false 结果；provenance 追踪输入字段与运行时值的关系；建议提取算法生成基于比较、长度、函数调用的 mutation 约束；区间采样（边界、转移、内部值）与类型检查；差分测试 harness 并行执行多客户端并对结果做差异分析。

**📊 数据集**

使用数据集：官方以太坊共识测试套件（Spectests）中的 22 个状态转换函数测试与 22 个单元测试作为种子，结合 Capella 及 Deneb 主网版本的状态与块输入。测试生成器在这些种子基础上扩展到数千个新的 (S,B) 对，用以覆盖所有可误判的 premise。

**📈 对比分析**

对比方法：与无指导的随机 mutation、传统代码覆盖驱动的差分 fuzzers（如 beacon‑fuzz）进行对比。结果显示 SpecGuard 将可误判 premise 覆盖率从约 30% 提升到约 90%，并发现 28 起跨客户端分歧（Class‑A 17 起，Class‑B 11 起），相较于现有差分测试工具至少高出 2–3 倍。实验在单台 AMD Ryzen 9950X 服务器上完成，测试生成与差分执行总耗时约 12 小时，覆盖率与分歧检测效率均显著提升。

**⚠️ 局限性**

局限性：①未覆盖那些逻辑上不可 falsifiable 的 premise（如永真、封闭分支），导致部分 premise 被排除；②测试生成仅聚焦于状态转换层，未覆盖 fork‑choice、执行层等；③机械化规范需要手工维护，fork 更新时仍需一定人工工作；④在某些案例中，提取的约束与字段类型不匹配，需依赖 fallback 产生随机变异，可能降低发现更深层 bug 的概率。

---

## 342. FESC: Remodeling Long-Context Private Inference with Encrypted State-Space Models

**arXiv ID:** 2608.17442 | [PDF](https://arxiv.org/pdf/2608.17442v1)

**作者:** Yufan Zhu `[一作]` (National University of Singapore), Xiaokui Xiao `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Factorized Encrypted Scan‑Contract（FESC）技术，实现了可在单张GPU上完成长文本（最大4096词）隐私保留推理的选择性状态空间模型（SSM）推理。

**💡 创新点**

创新点在于将选择性SSM的递归拆分为“扫描‑合同”形式，在加密域中实现对输入依赖的变换压缩、对齐与合同，消除了先前方案中深度、内存和通信三大瓶颈，并为MPC阶段设计了高效的非线性协议。

**🔧 技术方法**

技术组合包括CKKS加密的GPU化线性算子、Brent‑Kung并行前缀扫描、FESC与密文‑共享边界转换、以及专门为SiLU、Softplus、RMSNorm与指数函数定制的MPC协议；整个流水线为FHE‑MPC混合实现。

**📊 数据集**

评测数据集为三大长文本分类任务：SCOTUS（法律）、arXiv（科学）与Patent（专利），各包含数千条文档，使用不同长度（128~4096词）进行基准测试。

**📈 对比分析**

与现有私有推理系统（AEGIS、EncFormer、MPCFormer、BOLT、BumbleBee等）对比，FESC在L=2048时在单GPU上耗时77.3分钟、GPU时数4.34倍更低，内存仅32.7 GB，且保持与明文模型相当的准确率；在四GPU上可缩短至36.0分钟。

**⚠️ 局限性**

局限在于仅针对半诚实单查询文本分类场景、仅实现Mamba‑2模型、未覆盖恶意攻击模型、未实现自回归推理以及对更大或不同SSM架构的高效加密实现仍待进一步研究。

---

## 343. From Entity Mentions to Tone: An LLM-Based Pipeline for Media Bias Analysis

**arXiv ID:** 2608.17454 | [PDF](https://arxiv.org/pdf/2608.17454v1)

**作者:** Klesti Hoxha `[一作]` (University of Tirana), Olti Qirici `[通讯]` (University of Tirana)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并应用了一个基于本地Gemma 4 LLM的媒体偏见与框架分析管道，对阿尔巴尼亚新闻进行主题、事件聚类、命名实体识别与情感分析，并生成源级、人物级、事件级的偏见与门控指标。

**💡 创新点**

创新点在于：① 在低资源语言环境下无须专门训练模型即可利用通用开源LLM完成NER与情感标注；② 通过比较两种提示（Prompt）在一致性与覆盖率之间的权衡，展示了提示设计对自动化标注的实质影响；③ 提出了统一的多维度偏见评估框架，能够在源、人物和事件层面并行监测媒体框架。

**🔧 技术方法**

使用技术包括：Kedro框架搭建管道；Gemma 4 LLM进行NER与情感输出；TF‑IDF+K‑means做主题聚类；余弦相似度+时间窗口检测事件；JSON结构化输出与规则校验；Python与GPU加速的后端实现。

**📊 数据集**

数据集为8,358篇来自GDELT的阿尔巴尼亚新闻（涵盖124个新闻源），涵盖50个主题簇、3,551个事件簇，其中1,340个事件为多源事件。

**📈 对比分析**

通过将LLM输出与GDELT自动注释进行比较：情感标签一致率约为58%（Prompt v1）至62%（Prompt v2）；人名NER的平衡一致率约为45%；Prompt v2虽然消除了标签-分数不一致，但执行速度减慢且标注覆盖率下降。总体而言，LLM标注与GDELT存在中等程度的一致性，并能补充遗漏的人物提及。

**⚠️ 局限性**

局限性包括：标注覆盖率不高、执行时间长（尤其Prompt v2）；LLM可能携带自身偏见；与GDELT比较缺乏人工金标准，结果受限于对自动化系统的评估；需要后续人工校验才能满足高质量偏见分析；低资源语言下缺乏专门的评测数据。

---

## 344. COMMITGUARD: Differential Slice Fuzzing for Commit-Induced Bug Detection

**arXiv ID:** 2608.17401 | [PDF](https://arxiv.org/pdf/2608.17401v1)

**作者:** Aniruddhan Murali `[一作]` (University of Waterloo), Meiyappan Nagappan `[通讯]` (University of Waterloo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于差分切片的 fuzzing 方法，用于在提交层面检测并验证代码变更导致的内存安全缺陷。

**💡 创新点**

创新点在于将提交前后的函数切片分别 fuzz 并使用差分 oracle，仅保留在提交后出现的新 sanitizer 报告，从而将噪声报告大幅筛除，提供针对性较强的反馈。

**🔧 技术方法**

使用程序切片技术构造可编译的函数切片，结合 AFL++ fuzzing、AddressSanitizer/LeakSanitizer/UBSan/MemSanitizer 等运行时检测器，生成输入包装器、并行执行与输入重放以提升比较可靠性。

**📊 数据集**

实验数据集包含 300 个提交，分别来自 OpenSSL、libpcap、leptonica 三个开源 C 项目，每个项目 100 个提交。

**📈 对比分析**

先产生 518 条原始 sanitizer 报告，差分筛选后只剩下 7 条候选缺陷，其中 5 条经人工确认为真实缺陷；平均每个提交耗时 32.4 分钟，覆盖率为 75.36% 的修改函数。

**⚠️ 局限性**

局限包括：无法直接分析新添加的函数；切片缺失全程序预条件导致误报；函数签名变更导致包装器不一致影响输入重放；仅检测 sanitizer 能捕获的低层内存安全错误。

---

## 345. ORCA: Observability-Grounded Program Repair for Microservice Incidents

**arXiv ID:** 2608.17018 | [PDF](https://arxiv.org/pdf/2608.17018v1)

**作者:** Yuanchen Gao `[一作]` (Hong Kong University of Science and Technology), Hans-Arno Jacobsen `[通讯]` (University of Toronto)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于可观测性（telemetry）的自动程序修复流水线ORCA，能够从失败与参考的遥测差异中生成修复补丁。

**💡 创新点**

创新点在于将遥测差异提炼为故障签名，用于精准定位代码与配置候选；并通过修复图（repair graph）代理与探索代理在受限上下文内生成 exact‑match patch，最终用遥测重放验证补丁有效性。

**🔧 技术方法**

技术包括遥测差异提炼、基于 Ochiai 的谱系故障定位、配置抽象器、LLM 驱动的修复图代理与探索代理，以及四维遥测重放验证（TGPV）。

**📊 数据集**

使用了 575 例微服务故障基准，包含 200 例代码故障、225 例配置故障和 150 例真实生产事件（含 94 个突变变体）。

**📈 对比分析**

与六个基线（Direct、One‑Shot、ReAct、Agentless、AutoCodeRover、mini‑SWE‑agent）对比，ORCA 在补丁有效率、遥测重放成功率上显著提升（代码子集 95% 验证、配置子集 93% 重放），且 token 消耗低于最强基线（约 26k vs 640k）。

**⚠️ 局限性**

局限在于依赖完整的遥测对齐；对多语言或缺失遥测的情形定位仍不足；并且验证仍以重放为准，未能完全保证根因修复。

---

## 346. rl-triton: High-Performance Triton GPU Kernels for Reinforcement Learning Credit Assignment

**arXiv ID:** 2608.17641 | [PDF](https://arxiv.org/pdf/2608.17641v1)

**作者:** Lars Simon Zehnder `[一作]` `[通讯]` (Independent Researcher), Lars Simon Zehnder (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一个统一的、开源的GPU内核库，用于高性能计算强化学习中的信用分配（优势、回报、资格追踪等），通过在Triton中实现关联扫描来加速七种常用算法。

**💡 创新点**

创新点包括：①将GAE、V‑Trace、Retrace、TD(λ)、折扣回报、资格追踪和分段前缀和等七种算法归结为同一一阶线性递归，使用相同的关联扫描算子；②设计融合Triton内核，直接在寄存器/共享内存中构造系数，避免中间结果在HBM上的读写；③在扫描中显式处理终止、截断和回合窗口边界，确保结果正确。

**🔧 技术方法**

采用的技术主要是：Triton DSL 与编译器、并行关联扫描（prefix‑scan）算子、GPU内存层次优化（HBM、共享内存、寄存器）、自定义组合函数、批量处理多环境/时间步、针对不同算法的系数构造与截断处理。

**📊 数据集**

实验使用的“数据集”是合成的强化学习回合缓冲（[num_envs, seq_len] 的奖励、价值、done、重要性比等张量），并在 NVIDIA H100（80GB HBM3）和 RTX 2000 Ada（GDDR6）GPU 上进行基准测试；没有使用公开 RL 数据集。

**📈 对比分析**

比较方法：与 Python 循环实现、与自定义的向量化（doubling‑scan）基线以及 PufferLib 的 CUDA 内核进行对比。性能结果显示，在大规模并行模拟场景（如 4096 环境 × 128 步）下，融合 Triton 内核相对于向量化基线实现 1.6–5.7× 的全调用加速；相对于循环实现可达 16–317×；与 PufferLib 比较时，绝大多数形状下表现更优。

**⚠️ 局限性**

局限性：① Retrace 在 seq_len>2048 时因寄存器压力导致回退到非融合实现；② GAE 在极高并行/短序列形状下设备时间下降；③ 仅支持 float32；④ β_t 必须为标量，无法支持维度分布的系数；⑤ 纯前向扫描（资格追踪、前缀和）没有分块 fallback；⑥ 统一内核对序列长度有限制（≤131072）。

---

## 347. Communication Reduction via Semantic-Based Encoding in DMPC Using LSTMs

**arXiv ID:** 2608.17592 | [PDF](https://arxiv.org/pdf/2608.17592v1)

**作者:** Torben Schiz `[一作]` (LUT University), Henrik Ebel `[通讯]` (LUT University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并评估了基于LSTM的编码解码架构，用于在分布式模型预测控制（DMPC）中对移动机器人编队的互相通信进行语义压缩，以减少通信负载并保持控制性能。

**💡 创新点**

创新点在于：①设计多种LSTM编码解码网络（SHRED、cell LSTM、LSTMP、LSTM+FFL、LSTM+FFLH），其中LSTM+FFL/LSTM+FFLH可一次性训练并适用于多种预测时域；②在仿真与嵌入式硬件上对比全通信、自动编码器与自研网络，展示DMPC对通信丢包的鲁棒性。

**🔧 技术方法**

技术实现主要包括：Encoder‑Decoder LSTM网络（PyTorch训练）、CasADi+Eigen求解分布式最优控制、ZCM/UDP多播通信、Raspberry Pi 5嵌入式部署、以及对LSTM解码器的时间复杂度评估。

**📊 数据集**

数据集为在两机器人编队模拟中生成的候选控制序列，H=20下采集2000个随机场景；同时为可变时域训练额外采集200个场景，覆盖H∈{21,…,25}，全部由仿真生成并用于网络训练与验证。

**📈 对比分析**

评估方法：在理想通信下比较T_end时的平均/最大/最小成本和y‑方向误差CDF；在嵌入式实验中记录信息接收率与价值函数演化。实验表明SHRED在模拟中几乎恢复全通信性能，LSTM+FFLH在不同时域上表现稳健，但在大编队或长时域下因解码器计算开销导致信息接收率下降，影响收敛。

**⚠️ 局限性**

局限性：LSTM解码器的计算成本限制了大规模编队或长时域的实用性；SHRED需要在每次时域变化时重新训练；实验仅使用仿真动力学，未考虑移动节点的实时通信延迟与误差；对DMPC在丢包环境下理论鲁棒性缺乏正式分析。

---

## 348. Depth Enables Local Entropy: Quadratic Depth Dependence in Deep Variation-Norm ReLU Regression

**arXiv ID:** 2608.17434 | [PDF](https://arxiv.org/pdf/2608.17434v1)

**作者:** Tao Jiang `[一作]` (Chinese Academy of Sciences), Shaowei Cai `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

研究了具有深度L、宽度w、层和深度约束的向量值Parhi–Nowak深度‑ℛBV²架构在高斯回归中的极限风险，并给出匹配的上界与下界，证明最优风险随深度呈二次多项式增长。

**💡 创新点**

构造了局部打包码以证明深度引入的二次因子是本质的；改进了单位系数逼近定理并提出平衡放大技术，实现在保持层‑和成本的前提下实现放大。

**🔧 技术方法**

使用逼近理论、局部化Fano论证、平衡放大技术、向量值ℛBV²范数、伪维数与覆盖数估计、以及常数通道等技术。

**📊 数据集**

无；该工作为理论分析，不涉及实验或数据集。

**📈 对比分析**

无实验对比；通过理论上界与下界证明相匹配，表明在统计范围内深度对风险影响是二次多项式。

**⚠️ 局限性**

对数因子不匹配；宽度较小或精确浮点约束未被完全覆盖；仅针对向量值Parhi–Nowak架构，未验证更一般网络。

---

## 349. Foundation Agents Meet Agentic Deep Research: Evidence-Grounded Clinical Code Forecasting

**arXiv ID:** 2608.17075 | [PDF](https://arxiv.org/pdf/2608.17075v1)

**作者:** Junda Wang `[一作]` (University of Massachusetts Amherst), Carlos Morato `[通讯]` (Optum)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 ICD-Deepresearch 系统，通过结构化 EHR 基础模型、语言模型、医学检索和 ICD 词典协同生成并预测下一次就诊的 ICD 代码，并在预测后生成可解释理由。

**💡 创新点**

创新点在于：①将 SparseEHR 的患者先验与 GPT‑5 的直接生成并行；②设计两轮受限的研究扩展，依据患者线索检索并验证临床关系与代码语义；③使用 Evidence‑Aware Reranker 进行联合排名；④实现基于预算的候选搜索与可解释输出。

**🔧 技术方法**

使用技术包括：SparseEHR（结构化 EHR 模型）、GPT‑5（语言模型）、Google/PMC/arXiv 等检索工具、ICD 词典、双路径候选生成、两轮研究循环、Reciprocal Rank Fusion 对照、Post‑Selection 解释生成。

**📊 数据集**

实验数据集为 MIMIC‑III（ICD‑9‑CM）和 MIMIC‑IV（ICD‑10‑CM）两大 ICU 病例数据库。

**📈 对比分析**

与单一路径（候选生成或直接预测）以及两种独立研究系统（GPT‑5 + Web Search、Medical Deep Research）进行对比。ICD‑Deepresearch 在 MIMIC‑III 上 P@20/R@20 为 24.60/35.09%，在 MIMIC‑IV 上为 25.14/48.32%，显著优于基线；研究扩展检索量更少，却得到更高的医生使用率。

**⚠️ 局限性**

局限性包括：①检索深度受限于两轮循环，可能漏掉有价值证据；②对罕见诊断和长序列覆盖不足；③解释在预测后生成，无法实时支持临床决策；④对未来代码的证明仍不完整，仍需进一步完善检索与验证策略。

---

## 350. SNIPTEST: Fuzzing Multi-Level Code Slices for Validating Vulnerabilities

**arXiv ID:** 2608.17396 | [PDF](https://arxiv.org/pdf/2608.17396v1)

**作者:** Aniruddhan Murali `[一作]`, Meiyappan Nagappan `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于执行的警告分流框架，利用多层次代码切片结合定向模糊测试，来验证静态分析工具产生的漏洞警告是否真实。

**💡 创新点**

创新点在于：① 引入层级切片（Level‑0、Level‑1、Level‑2）逐步恢复调用上下文；② 对不同层级的模糊结果进行聚合，形成“可能真阳性/可能假阳性/不可达”的证据；③ 通过缓存、并行与跳过冗余等优化显著提升分析效率。

**🔧 技术方法**

使用技术包括：基于 Tree‑sitter 的函数级调用图构建；Slice‑C 的代码切片与单独可执行单元编译；AFL++ 与 LLVM‑coverage 的模糊测试与覆盖跟踪；AddressSanitizer、LeakSanitizer 等运行时漏洞或acles；以及多级聚合与早停规则。

**📊 数据集**

数据集主要来自 RevBugBench：97 条真实漏洞 + 97 条已知误报，覆盖 libxml2、zstd、PROJ 三个开源项目；此外在实际项目 Vim、libpcap 上也验证了新发现的 CVE‑2025‑11964 等。

**📈 对比分析**

与基准 AFLGo（种子与无种子）相比，Unseeded 情况下可达位置更多，验证效率提升 5.5–10.6×；与 LLM4SA 进行静态预测对比，F1 分数 0.791 vs 0.360，证明动态证据更具辨别力；与单层切片 SnipTest 对比，多层切片在准确率上提升至 0.791 的 F1，覆盖率更高。

**⚠️ 局限性**

局限性包括：① 切片深度最多 3 层，可能遗漏深层守卫导致误判；② 模糊时间短（5/7.5/10 分钟），对大切片的探索不足；③ 对函数指针、文件 I/O、环境变量等动态特性的建模缺失；④ 依据经验的聚合规则仍为启发式，缺乏形式化保证。

---

## 351. Software Defined Networks Key Relay for Large-Scale Quantum Key Distribution Networks

**arXiv ID:** 2608.17539 | [PDF](https://arxiv.org/pdf/2608.17539v1)

**作者:** Stephan Laschet `[一作]` (AIT Austrian Institute of Technology), Alessandro Colombo `[通讯]` (AIT Austrian Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过软件定义网络（SDN）对大规模量子密钥分发网络（QKDN）进行编排，定义了控制器聚合关键性能指标（KPI）并基于这些指标选择最优路径。

**💡 创新点**

创新点在于提出了多路径选择算法（基于Dijkstra与最大-最小容量）、内置负载均衡，以及在多域场景下使用的无感知多方协议，以避免供应商暴露敏感网络信息。

**🔧 技术方法**

采用的技术包括SDN架构（OpenFlow）、Dijkstra算法、最大-最小容量算法、负载均衡机制、无感知多方计算，以及ETSI/ITU标准作为规范。

**📊 数据集**

使用的“数据集”主要是通过仿真生成的网络拓扑和量子链路性能指标（如密钥速率、损耗），并未使用公开的真实量子网络数据。

**📈 对比分析**

通过仿真对比了不同路径选择算法的性能，评估了密钥速率、时延和负载分布；实验表明最大-最小容量算法在均衡负载方面优于Dijkstra，但在路径延迟上略逊。

**⚠️ 局限性**

局限性包括仅在仿真环境验证，未考虑实际部署中硬件延迟与网络不稳定；对极大规模网络的可扩展性尚待进一步评估。

---

## 352. Benchmarking Classical and Transformer-Based Models for Document Sensitivity Classification

**arXiv ID:** 2608.16928 | [PDF](https://arxiv.org/pdf/2608.16928v1)

**作者:** Aleesha Zainab `[一作]` (PIEAS), Asifullah Khan `[通讯]` (PIEAS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个无泄漏的16k外交电报数据集（Strategic 16K），并在此数据集上对六种文档敏感性分类模型（TF‑IDF+LR/SVM/Naïve Bayes以及BERT/ELECTRA/RoBERTa）进行系统基准评测。

**💡 创新点**

首次公开透明的泄漏消除协议与可复现的WikiLeaks PlusD基准，完成了经典与Transformer模型跨族群的完整对比。

**🔧 技术方法**

采用TF‑IDF特征与线性分类器（LR、SVM、Naïve Bayes）以及BERT、ELECTRA、RoBERTa预训练Transformer模型的fine‑tune，配合5折交叉验证评估。

**📊 数据集**

使用Strategic 16K数据集，即从WikiLeaks PlusD挑选的16,000篇外交电报，二分类为敏感/非敏感。

**📈 对比分析**

通过5折stratified CV计算加权F1、准确率和敏感类召回率，Transformer模型取得最高F1 89.33%（BERT），经典模型最低F1 83.91%（Naïve Bayes）。

**⚠️ 局限性**

单一模型缺乏可解释性与多信号聚合能力，评测仅覆盖公开外交电报，未涵盖其他组织文档类型。

---

## 353. Lambda-Hold Control: Human-Like Movement Emerges from a Minimal Task Reward in Predictive Musculoskeletal Simulation

**arXiv ID:** 2608.17030 | [PDF](https://arxiv.org/pdf/2608.17030v1)

**作者:** Jun Hyuk Lee `[一作]` (Seoul National University), Jooeun Ahn `[通讯]` (Seoul National University)

**通讯引用:** 628 | [OpenAlex ID](https://openalex.org/A5002868202)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过训练肌肉驱动的骨骼模型，实现仅用前向速度奖励即可产生人类类似的冲刺运动。

**💡 创新点**

创新点在于将控制变量改为肌肉阈值 λ 并采用间歇性保持（λ-hold）以自下而上产生肌肉协同，解决了高维冗余导致的探索低效问题，并首次实现了仅用极简奖励学习到逼真冲刺。

**🔧 技术方法**

使用强化学习（Soft Actor-Critic + gSDE）结合 Feldman 的 equilibrium‑point（EP）平衡点阈值和伸展反射公式；仿真环境为 SCONE/Hyfydy 引擎，模型为 H2190 三维肌肉骨骼系统。

**📊 数据集**

评估数据集包括 Fukuchi 等人提供的 2.5/3.5/4.5 m/s 轨道跑步 kinematics 与 GRF 数据，以及 5 m/s 的表面 EMG 数据集；训练使用的模拟模型为 90 条肌腱单元。

**📈 对比分析**

与四个基线（Plain SAC、Excitation‑hold、DEP‑RL、Synergy）在模拟步骤和决策步骤两条轴上对比，λ‑hold 在前 1e7 步内即超过所有基线，学习速度提升约 10 倍以上；得到的冲刺速度约 4.7 m/s，在关节角、GRF 以及肌肉激活与人类数据的相关系数均优于大多数基线。

**⚠️ 局限性**

局限性包括：模型的肌肉力–速度关系导致最高速度低于人类（约 4.7 m/s），缺少手臂和更复杂的动力学约束；对非周期性任务的持久化时机机制未得到验证；以及 λ‑hold 对 EP 参数敏感，可能难以直接迁移到其他运动任务。

---

## 354. Synthesizing Feature Extractors: An Agentic Approach for Algorithm Selection

**arXiv ID:** 2608.17170 | [PDF](https://arxiv.org/pdf/2608.17170v1)

**作者:** Hai Xia `[一作]` (TU Wien), Stefan Szeider `[通讯]` (TU Wien)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于大型语言模型的框架，自动生成可解释的图论特征提取器，用于约束优化问题的算法选择。

**💡 创新点**

创新点在于引入检查–修正–验证循环，将LLM的代码生成转化为可执行的Python脚本，并在MiniZinc模型上自适应地提取结构化、solver‑aware特征，显著提升了特征多样性与利用率。

**🔧 技术方法**

技术实现包括LLM代理（OpenAI o4-mini）、程序合成、MiniZinc到Python的转换、图网络特征计算（NetworkX、NumPy）以及与五款主流求解器的集成。

**📊 数据集**

实验使用来自MiniZinc挑战的三类组合优化数据集：车辆路径规划(VRP)、车队排队(CS)和固定长度误差纠正码(FLECC)。

**📈 对比分析**

通过与专家手工构造的mzn2feat和基于Transformer的trans2feat进行对比，并在多种AS工具链（Random Forest、AutoSK、LLAMA、AutoFolio）上评测，LLM生成的特征在测试集准确率上比mzn2feat提升高达8.3个百分点，优于所有trans2feat变体。

**⚠️ 局限性**

局限性包括仅为每个问题族生成单一提取器、缺乏人机交互的代码优化环节、仅支持MiniZinc输入以及目前需要依赖强大的商业LLM（OpenAI/Claude），开放权重模型尚未能稳定通过检查–修正–验证流程。

---

## 355. Beyond MSE: Rethinking the Evaluation Metric and Benchmarking for Irregular Time Series Forecasting

**arXiv ID:** 2608.17293 | [PDF](https://arxiv.org/pdf/2608.17293v1)

**作者:** Rongwen Li `[一作]` (Hunan University), Changjian Chen `[通讯]` (Hunan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文重新审视不规则时间序列预测的评估方法，指出常用的MSE/MAE评估因采样分布而产生偏差，提出基于重要性采样的连续时间平方误差(CSE)度量，并构建包含合成、半合成与八个真实世界数据集的全新评估基准；

**💡 创新点**

创新点主要包括：1）证明MSE/MAE实际上估计的是观测分布风险而非连续时间风险；2）设计自归一化重要性采样的CSE，并证明其渐近误差不大于MSE；3）分解MSE–CSE差异为采样依赖差距与时间分布差距两部分；4）搭建系统化的多层基准来验证CSE的有效性。

**🔧 技术方法**

使用了重要性采样与自归一化权重、Gaussian核密度估计、理论证明（渐近误差比较）以及大规模实验评估；

**📊 数据集**

实验使用了四个合成/半合成数据集（Synthetic‑Regime、Synthetic‑Multiscale、ETTm1、Weather）和八个真实世界数据集（USHCN、MIMIC‑III、HumanActivity、GDELT、RepoHealth、StudentLife、FNSPID、CESNET）。

**📈 对比分析**

通过MSE、CSE、GSE、G_samp、G_time等指标对11种不规则时间序列模型进行评估。实验表明CSE能更准确恢复连续时间风险并修正模型排名，MSE在多种数据集上产生排序逆转，CSE在不同采样条件下表现更稳健。

**⚠️ 局限性**

局限性包括：1）需要对连续时间轨迹有精确或可近似的真值；2）CSE依赖密度估计，样本量不足或极端分布时误差可能增大；3）仅针对平方误差（MAE版未充分研究）；4）未考虑多维空间中的不规则采样或其它可能的误差源。

---

## 356. Network Denoising Revisited: A Ricci-Flow-Inspired Graph Diffusion Method

**arXiv ID:** 2608.16923 | [PDF](https://arxiv.org/pdf/2608.16923v1)

**作者:** Ye Fang `[一作]` (Sun Yat-sen University), Chuan-Xian Ren `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2255 | [OpenAlex ID](https://openalex.org/A5043031658)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于 Ricci 流启发的网络去噪框架 Ricci‑Diffusion，利用图的 Ollivier–Ricci 曲率来调节扩散核，进而在去噪过程中实现几何自适应的边权更新。

**💡 创新点**

创新点包括：
- 将曲率信息嵌入到扩散核中，实现局部几何调制；
- 证明曲率能区分传统相似度扩散无法分辨的结构，并给出一阶修正项；
- 引入动态‑静态两阶段策略，先在动态阶段反复更新曲率与核，后在静态阶段收敛，兼顾效率与稳定性；
- 通过实验验证了 Ricci‑Diffusion 在多任务上表现出 Ricci‑流式的曲率均匀化特性。

**🔧 技术方法**

核心技术包括：
- kNN 预稀疏化、对称化与 DSM 投影；
- 计算 Ollivier–Ricci 曲率并通过指数映射调制相似度；
- 一般化扩散表示和收敛分析；
- 动态‑静态两阶段迭代与梯度一阶修正分析；
- 通过曲率相关性与 NMI/AUROC 等指标评估。

**📊 数据集**

使用的实验数据集：
- 真实世界：16 种组织的基因功能预测网络、GM12878 Hi‑C 1k/5k 分辨率网络、Leeds Butterfly 图像相似网络；
- 合成数据：GN（Gaussian Noise）和 LFR（Lancichinetti‑Fortunato‑Radicchi）社区结构图。

**📈 对比分析**

与 6‑7 种基线方法（ND、NE、NR、BORF、GSR 等）比较，Ricci‑Diffusion 在所有任务中普遍取得更高 AUROC（基因功能预测）、更高 NMI（TAD 检测）、更高检索准确率（物种识别）以及更低 inv‑SNR（显式去噪），尤其在稀疏、噪声较重的场景中优势更明显。

**⚠️ 局限性**

局限性：
- 需要对曲率进行估计，曲率计算对噪声敏感，需调参；
- 动态阶段计算量大，适用于中小规模图；
- 目前仅适用于无向加权图，缺乏对有向或超图的推广；
- 对极大规模网络的可扩展性和并行实现尚未充分验证。

---

## 357. Fairness--Stability Trade-offs in Many-to-One Matching

**arXiv ID:** 2608.17295 | [PDF](https://arxiv.org/pdf/2608.17295v1)

**作者:** Genjie Qin `[一作]` `[通讯]` (Ocean University of China), Genjie Qin (Ocean University of China)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在可转移支付的多对一匹配市场中，本文研究公司侧EF1公平性与核心稳定性之间的权衡，给出了固定匹配的最大可支持核心因子、局部敏感度以及安全轮算法，并推导了EF1–核心的最小化极值前沿；同时扩展到更强的公平性（EFX）和容量约束场景。

**💡 创新点**

核心创新点包括：① 引入融资比率Φ(X)与核心因子α(X)=1/Φ(X)的等价关系，将稳定性转化为瓶颈最小化的线性规划；② 通过安全轮（最大边/互行互列安全边）实现EF1保证并给出最优的核心和福利下界；③ 推导出EF1–核心极值前沿的下界Γ_m(δ)和上界u_m(δ)，在两、三公司情形下完全闭合；④ 在更强公平性和容量限制下仍保持融资框架，给出相应的核心与福利保证。

**🔧 技术方法**

主要技术手段包括：线性规划与对偶性、瓶颈融资问题的最小化、局部敏感度分析（单工人调动的R_i变化公式）、安全边集与可交换性论证、最大边轮算法的组合证明、以及对称实例的最优性构造。

**📊 数据集**

本文为理论研究，未使用任何真实数据集；所有结果均通过数学证明与极限分析得到。

**📈 对比分析**

方法评估采用理论最优/最差情况比较：给出对所有m与δ的下界Γ_m(δ)与上界u_m(δ)，并在m=2、3时实现完全闭合；在m→∞时证明最优稳定率趋向δ。与已有文献相比，本文提供了最严格的公平–稳定性兼容性下界与上界，填补了此前仅针对特定公平性或稳定性指标的空白。

**⚠️ 局限性**

局限性包括：① 仅适用于非负加性价值函数；② 假设可转移支付与完全信息；③ 未覆盖工人侧公平性或非可转移支付场景；④ 对于大规模多公司，只给出上界下界的粗略估计，未能给出完全匹配的闭合式解；⑤ 在冲突解算时仍需求解LP，可能在实践中产生计算瓶颈；⑥ 未考虑动态或递归匹配、工人偏好不确定性等更一般情况。

---

## 358. A Multiplication-Free Feature Extractor for Signal Classification: Keyword Spotting Case Study

**arXiv ID:** 2608.17108 | [PDF](https://arxiv.org/pdf/2608.17108v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 359. Wasted large language models: A life cycle thinking approach

**arXiv ID:** 2608.17055 | [PDF](https://arxiv.org/pdf/2608.17055v1)

**作者:** Erik Johannes Husom `[一作]` (SINTEF Digital), Ophelia Prillard `[通讯]` (SINTEF Digital)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5092623986)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出将欧盟废物管理层级（预防、再利用、回收、恢复、处置）应用于大型语言模型（LLM），并阐述如何通过生命周期思维减少LLM的碳足迹和资源浪费。

**💡 创新点**

首次将LLM视为可产生废弃物的产品，借助废物层级框架为AI可持续性提供了新的政策和实践视角。

**🔧 技术方法**

使用生命周期思维、废物层级分析方法，并结合LLM的训练、存储、迁移和推理能耗概念进行理论性阐述。

**📊 数据集**

本研究未使用具体数据集，而是参考公开的LLM生命周期评估、学术文献和行业报告中的生命周期数据与指标。

**📈 对比分析**

未进行实验比较或性能评估；文章以概念性讨论为主，未给出定量指标或实验结果。

**⚠️ 局限性**

局限性在于缺乏实证验证，未提供LLM生命周期数据与废物层级对应的具体量化评估；方法在实际应用中仍需进一步测试与完善。

---

## 360. Appearing Legitimate is Not Enough: Interrogating Synthetic Agents in Representational Processes through a Participatory Design Lens

**arXiv ID:** 2608.17099 | [PDF](https://arxiv.org/pdf/2608.17099v1)

**作者:** Aditya Nayak `[一作]` (University of Pittsburgh), Aakash Gautam `[通讯]` (Western University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对三种代表性案例（UNU-CPR的 AI 头像 Ask Amina/Ask Abdalla、New Sun Rising 的社区语音聊天机器人 Ana 以及法律模拟平台 Synthetic Juror）的系统分析，阐述了合成智能代理在政治、司法和外交等代表性决策场景中制造“人格”与“合法参与”的四步流程，并提出了基于参与式设计的批判视角和软硬边界设计原则。

**💡 创新点**

创新点包括①构建了问题定义→数据策划→交互设计→效度验证的四步“人格制造”框架；②首次将参与式设计的探测、引导、理解、生成四种模式应用于评估合成智能的代表性和合法性；③提出软硬边界（软边界区分分析工具与拟人化接口；硬边界界定代理不能替代人类参与的绝对限制），为监管与治理提供了可操作的设计准则。

**🔧 技术方法**

技术方面主要使用 LLM 及其 RAG（检索增强生成）架构，Ana 与 Ask Amina/Ask Abdalla 采用 RAG；Synthetic Juror 则采用混合神经符号体系；此外，在设计对话与评估时使用了 LLM 的语言生成与推理能力。

**📊 数据集**

数据集包括：Ana 依托 New Sun Rising 的社区语音项目（收集社区领袖与成员的访谈记录与情感标签）；Ask Amina/Ask Abdalla 使用 UN、多边机构、智库与 NGO 的调查报告、新闻与社交媒体内容；Synthetic Juror 使用公开数据库、订阅服务、客户专有信息及模拟案例数据。

**📈 对比分析**

方法论上，本文对比了传统的结果导向评估框架（可信度、可信度、真实性、算法忠实度）与参与式设计的过程导向评估，指出前者侧重输出相似性而忽视合法性所需的参与过程；由于缺乏实验数据，未给出具体性能指标，主要通过案例文件的文本分析和理论论证进行比较。

**⚠️ 局限性**

局限性包括：研究仅基于公开文档与宣传材料，缺乏直接与代理交互的实证数据；三例均为实验或试点阶段，缺少长期部署与真实决策场景的验证；对比评估依赖文献与案例分析，缺少量化指标。

---

## 361. Benchmarking the Benchmarks: Evaluating Automated Safety Benchmarks for Small Language Models

**arXiv ID:** 2608.17183 | [PDF](https://arxiv.org/pdf/2608.17183v1)

**作者:** Nyamtulla Shaik `[一作]` (University of Kansas), Bo Luo `[通讯]` (University of Kansas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对26种小型语言模型进行大规模自动化安全评估，系统评测5个主流安全基准套件，探讨“模糊”判定对评估结果的影响

**💡 创新点**

揭示安全基准中模糊标签主要由模型能力与提示复杂度共同驱动，导致评估结果不稳定，提出显式报告模糊率和可调节的“模糊校正评分”方案

**🔧 技术方法**

使用自动化判定器（GPT‑4o、MD‑Judge）和一系列文本可读性、流畅度、相似度等度量；对模糊判定进行机器学习预测（随机森林、CatBoost等）

**📊 数据集**

AirBench、SALAD‑Bench、HarmBench、Simple Safety Tests、BBQ等公开安全/偏见基准套件，覆盖约741k个模型-提示对

**📈 对比分析**

通过对比不同模型在各基准下的安全性分数、模糊率及排名敏感性，发现模糊率高时平均得分和排名易变；在低模糊率基准（如Simple Safety）表现相对稳定

**⚠️ 局限性**

仅依赖自动化判定缺乏人类真值，模糊标签可能掩盖安全缺陷；现有基准对小模型不适用，需要针对SML的专门基准与判定方法

---

## 362. Understanding Curriculum Learning in Large Language Models via Cross-Difficulty Optimization Dynamics

**arXiv ID:** 2608.17268 | [PDF](https://arxiv.org/pdf/2608.17268v1)

**作者:** Zhikai Ding `[一作]` (Fudan University), Ziyi Ye `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于交叉难度知识迁移的动态课程学习方法（TDCS），通过分析不同难度层次之间的梯度投影量化知识迁移（Relative Transfer）来动态调整训练样本分布；

**💡 创新点**

创新点在于①首次用Relative Transfer量化交叉难度知识迁移；②依据该量化结果设计自适应采样策略TDCS，避免固定的易到难调度；③证明该方法在多种推理任务和模型规模上均能显著提升性能；

**🔧 技术方法**

核心技术包括：一阶梯度分析推导Relative Transfer；自适应采样算法（Transfer‑aware Dynamic Curriculum Sampling）；LoRA微调框架；实验使用多模型、多任务的对比评估；

**📊 数据集**

使用的主要数据集为逻辑推理基准Sudoku、代码生成基准KodCode、数学推理基准iGSM；此外在自我提升实验中使用GSM8K和KodCode；

**📈 对比分析**

与传统Curriculum、Random、Mix固定调度进行对比，并在Qwen2.5-1.5/3/7B、Llama3.2-3B等不同规模和架构上实验；TDCS在所有任务上均获得最高准确率，提升幅度从约2.6%到20.8%；

**⚠️ 局限性**

局限性包括：①只利用梯度投影作为迁移度量，忽略了更复杂的任务关系；②阈值τ_e、τ_h需人工调参，适用性受限；③实验规模受GPU限制，未验证在更大模型或更多任务上的普适性。

---

## 363. Understanding Computing Identity Development Through Mentorship and Epistemic Network Analysis

**arXiv ID:** 2608.16904 | [PDF](https://arxiv.org/pdf/2608.16904v1)

**作者:** Behdokht Kiafar `[一作]` (University of Delaware), Roghayeh Leila Barmaki `[通讯]` (University of Delaware)

**通讯引用:** 863 | [OpenAlex ID](https://openalex.org/A5011977440)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对37名计算领域受访者的开放式问卷回答进行编码，利用Epistemic Network Analysis（ENA）探索计算身份构建与导师支持之间的关系。

**💡 创新点**

将身份视为网络关系并首次将ENA与导师支持相结合，揭示导师支持如何改变身份构建的连接模式。

**🔧 技术方法**

采用Epistemic Network Analysis（ENA）进行网络建模与可视化，随后使用不等方差两样本t检验比较两组ENA分数。

**📊 数据集**

37名计算相关领域参与者的开放式问卷文本，包含兴趣、能力、认可、归属、自我怀疑、冒名顶替综合征等六个身份构建。

**📈 对比分析**

通过对导师支持组与无导师支持组的ENA分数在SVD1维度进行t检验，结果显著（p=0.01，Cohen’s d=1.72），表明导师支持组的正面身份构建更为紧密。

**⚠️ 局限性**

样本不均衡（31 vs 6）、自选样本导致自选择偏差，缺乏因果推断与纵向追踪，未考虑导师类型和质量的细节。

---

## 364. Universal CKM for Environment-Aware Wireless Networks: Enabling Cross-Device and Cross-Task Channel Knowledge Transfer

**arXiv ID:** 2608.17382 | [PDF](https://arxiv.org/pdf/2608.17382v1)

**作者:** Haiquan Lu `[一作]` (Nanjing University of Science and Technology), Rui Zhang `[通讯]` (National University of Singapore)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出通用通道知识图（uCKM），实现跨设备、跨任务的无线信道知识共享与迁移；

**💡 创新点**

创新点在于：①将信道知识与设备和任务解耦，构建基于全球坐标系和概率分布的通用无线环境先验；②提出“All for uCKM”与“uCKM for All”的闭环范式；③融合多模态感知、物理知识与生成式 AI，提升知识提取与更新效率；

**🔧 技术方法**

采用概率建模（高斯混合、Rayleigh 等）、物理信息驱动的 AI 与深度网络、设备无关路径知识提取、生成式 AI 进行分布学习、差分隐私与同态加密保障隐私、数据融合与一致性检测等技术；

**📊 数据集**

主要使用 Sionna 生成的射线追踪仿真数据，并在多设备（车辆、机器人、手机）场景下采样；论文亦提及可结合 LiDAR、视觉等多模态传感器数据及仿真生成数据补充；

**📈 对比分析**

与完美 CSI、单设备 BIM 与随机波束选择等基准方案对比；uCKM 在三台设备的平均可达率上接近完美 CSI，显著优于单设备 BIM 与随机选取；在定位任务中，利用 uCKM 的全局坐标概率分布可提升定位精度，优于基于设备本地坐标系的 CKM；

**⚠️ 局限性**

局限性包括：①需要大量高质量测量数据，采集成本高；②设备特性与分辨率差异导致路径知识提取困难；③动态环境下连续 uCKM 的插值与更新仍具挑战；④隐私与安全问题需进一步解决；⑤标准化与商业化部署仍缺乏完整框架；

---

## 365. NeuroPath: Brain-Inspired Dual-Pathway Graph Convolutional Networks for Skeleton-Based Action Recognition

**arXiv ID:** 2608.17487 | [PDF](https://arxiv.org/pdf/2608.17487v1)

**作者:** Kanglei Zhou `[一作]` (Tsinghua University), Xiaohui Liang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种脑启发的双通路图卷积网络NeuroPath，用于从骨骼序列中进行动作识别。

**💡 创新点**

创新点包括：①通过模态转换单元将输入拆分为空间感知和时间感知两条通路；②设计空间–时间组图卷积块（STGGCB）实现组级动态特征聚合；③引入动态双通路融合模块，在多阶段交互中协调两条通路；④针对模态不平衡与互补问题提供统一的单流解决方案。

**🔧 技术方法**

使用技术包括：图卷积网络、双通路结构、模态转换、组聚合、适应性结构聚合、多尺度时间卷积、空间–时间注意力机制以及动态交互融合。

**📊 数据集**

在Kinetics Skeleton 400、NTU RGB+D 60和NTU RGB+D 120三个大型骨骼动作数据集上进行评估。

**📈 对比分析**

与多种STGCN、Transformer及多流方法对比，单流NeuroPath在NTU、Kinetics等数据集上刷新单模态基准，4流模型达到93.0%等高准确率，同时保持较低的GFLOPs和参数量，显示出优异的性能与计算效率。

**⚠️ 局限性**

局限性包括：仍依赖单模态强化，跨流融合方式相对简单；在动作细微差别识别方面存在误判；对更丰富的输入（如热图、RGB、音频等）及更高级的多流交互策略未做深入探索。

---

## 366. Expected free energy as an information constraint on the Bethe Lagrangian

**arXiv ID:** 2608.17167 | [PDF](https://arxiv.org/pdf/2608.17167v1)

**作者:** Wouter M. Kouw `[一作]` `[通讯]` (TU Eindhoven), Wouter M. Kouw (TU Eindhoven)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于约束 Bethe 自由能量的主动推理框架，利用信息约束来实现可消息传递的行动选择。

**💡 创新点**

用信息下限约束替代经验的探索权重，使 KKT 倍数被求解而非手动调节，得到包含 EFE 的一参数策略族。

**🔧 技术方法**

约束 Bethe 自由能量、KKT 优化、Forney 样式因子图、前向后向消息传递以及单变量根求解。

**📊 数据集**

在三项合成实验中使用的环境包括 T‑maze、信息收集网格和门控提示任务（所有均为离散模拟试验）。

**📈 对比分析**

与标准 EFE 和 Q‑MDP 基线对比，CBFE 在 T‑maze 与 EFE 一致，在网格中显著提升信息探索率（84% vs 45%/36%），并在学习任务中实现从探索到利用的顺利转变。

**⚠️ 局限性**

受限于需预设信息下限、KKT 乘子求解仍有计算成本，且实验仅在离散模拟环境中验证，尚未检验对连续或大规模问题的适用性。

---

## 367. S$^3$AM: A Single-Stream SAM with Reliability-Calibrated Frequency Adapter for Multi-modal Salient Object Detection

**arXiv ID:** 2608.17475 | [PDF](https://arxiv.org/pdf/2608.17475v1)

**作者:** Ruichao Hou `[一作]` (China Pharmaceutical University), Jinde Cao `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于SAM的单流多模态显著目标检测框架S^3AM，能够在不重复使用基础模型的情况下融合RGB与辅助模态信息。

**💡 创新点**

核心创新点在于引入可靠性校准的频率适配器（RCFA）和混合频率专家（MoFE），通过双门控制精确传播高频细节；以及超网络引导的语义-结构解码器（HSSD）实现语义完整性与边界细节的协同恢复。

**🔧 技术方法**

采用Stationary Wavelet Transform进行多频分解、SAM主干冻结、Mamba结构细节恢复、双门校准机制、超网络语义先验以及自适应频率专家聚合等技术。

**📊 数据集**

在RGB-D（NJU2K、NLPR、SSD、STERE）、RGB-T（VT821、VT1000、VT5000）和RGB-NIR（无监督转移）三大主流数据集上进行评估。

**📈 对比分析**

与六种RGB-D、六种RGB-T以及三种RGB-NIR基准方法对比，S^3AM在Fβ、MAE、E_m、S_m等指标上均实现或逼近最优表现，并在RGB-NIR零样本迁移中显著提升了Fβ与MAE。

**⚠️ 局限性**

仍存在对SAM基础模型的依赖、对高频噪声的局部处理仍不完美，以及在更广泛模态或大规模部署场景下的进一步优化空间。

---

## 368. MemCatalyst: Amplifying Data Auditing on Vision-Language Models via Data Poisoning

**arXiv ID:** 2608.17722 | [PDF](https://arxiv.org/pdf/2608.17722v1)

**作者:** Xukun Luan `[一作]` (Beijing Institute of Technology), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出MemCatalyst，一套通过对图像和文本进行隐蔽数据毒化的工具，以放大Vision‑Language模型的会员推理审计效果；

**💡 创新点**

创新点在于设计两种针对VLM的毒化策略——文本毒化PT和图像毒化PI，利用目标样本特征信息提升审计信号，并实现跨架构可迁移；

**🔧 技术方法**

使用的技术包括：利用引用VLM生成文本毒化、对抗优化逼近图像特征均值、MPNet与Rouge等相似度度量、温度调节的会员推理方法，以及对VLM视觉编码器和投影器的黑盒/灰盒操作；

**📊 数据集**

实验数据集包括LLaVA的158k图文对和MiniGPT‑4的3.5k图文对，构建shadow、member和target子集进行评估；

**📈 对比分析**

在五种状态‑of‑the‑art MI方法（Shadow、Reference、Target、Image等）上，MemCatalyst在仅注入0.07%–3%毒化样本的情况下显著提升AUC/准确率，且对模型性能的影响≤0.02；

**⚠️ 局限性**

局限性包括：PI需要灰盒访问视觉编码器输出，跨架构解释性不足；PT在保持语义一致性时对特征空间位移有限；联合毒化可能带来更高成本且更易被检测。

---

## 369. Beyond Suspicious Steps: Ontological Trust in Long-Horizon Agents

**arXiv ID:** 2608.17718 | [PDF](https://arxiv.org/pdf/2608.17718v1)

**作者:** An He `[一作]`, Haibin Zhang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种在线监控方法RGE，用来估计长时间代理执行过程中轨迹前缀是否仍属于用户授权的任务；

**💡 创新点**

创新点在于将任务授权定义为“本体信任”，拆解为角色、目标和证据三轴，并通过LLM结构化解析与确定性状态更新构建可回放、可审计的信任轨迹；

**🔧 技术方法**

技术包括：任务先验结构化、每步LLM解析生成结构化字段、角色/目标/证据三轴一致性评分、时间聚合阈值决策，LLM仅用于解析，其余计算为确定性；

**📊 数据集**

使用跨域轨迹语料库，包含OSWorld、FinanceBench和EICU-AC的正常、前缀漂移和伪一致性三类轨迹；

**📈 对比分析**

与现有规则、判定器和盾牌基线比较，RGE在前缀漂移检测上F1≥93%且正样本覆盖≥95.8%，在伪一致性检测上表现受任务闭合可观测性限制，整体表现优于基线；

**⚠️ 局限性**

局限包括：对伪一致性检测依赖环境可见闭合信号，对小型LLM的解析能力有限，且在缺乏外部闭合事件的读写型任务中无法区分伪一致性与正常检查。

---

## 370. CompCPZ: Preserving Multi-Modal Intent in Language-Guided Robot Manipulation

**arXiv ID:** 2608.17717 | [PDF](https://arxiv.org/pdf/2608.17717v1)

**作者:** Zhen Zhang `[一作]` (Technical University of Munich), Amr Alanwar `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为CompCPZ的编译器，将自然语言指令解析为可供机器人策略使用的多模态、可保证安全的集合表达，解决传统单连通解空间导致的“静默语义失败”。

**💡 创新点**

创新点在于：①在语言解析树上递归构造受限多项式锥形体（CPZ）集合，既能精确表达并集、交集与补集；②利用分层分布式合成的分割共形预测，保证每个原子谓词的上下界包含真解集；③通过组合代数实现分量最小化与运行时模式切换，突破了单连通解空间的结构性下限。

**🔧 技术方法**

核心技术包括：自然语言处理（GPT‑4o链式思考解析）、视觉目标检测（YOLOv8n+自定义数据）、受限多项式锥形体代数、分布式分割共形校准、基于解析树的布尔组合以及线性规划验证。

**📊 数据集**

使用ManiSkill3桌面操控环境（共18类任务，49条指令，191条扩展）进行仿真评测，并在Unitree Go2四足机器人上进行平面真实硬件验证。

**📈 对比分析**

与基准方法（AABB、MVEE、GMM、VLM‑Action、VLA、热图最大值、采样解码器等）对比，CompCPZ在In‑GT‑mode率、几何成功率和对抗性模式切换下均取得显著优势（在仿真中1,900/1,918对标检验胜利，p ≪ 10⁻³⁰；硬件中12/12真模式命中）。

**⚠️ 局限性**

局限性包括：①仅在轴对齐原子谓词下证明深度无关稳定性；②目前为单步语义映射，缺乏长时序与时序逻辑扩展；③硬件验证仅限平面四足平台，尚未覆盖接触丰富的操作机器人。

---

## 371. Array-Based Molecular Pulse Encoding for Neuro-Spike Communication in Intra-Body Nano-networks

**arXiv ID:** 2608.17675 | [PDF](https://arxiv.org/pdf/2608.17675v1)

**作者:** Keyvan Aghababaiyan `[一作]` `[通讯]` (Universidad Miguel Hernández de Elche), Keyvan Aghababaiyan (Universidad Miguel Hernández de Elche)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

设计了一种基于神经尖峰阵列的同步性较低的通信方案，用辅助纳米机桥接受损神经，恢复信息传输。

**💡 创新点**

创新点在于通过不同化学信号（谷氨酸与GABA）按序列排列而非时序编码，消除了对高阶时钟同步的需求，并显著提升通道容量。

**🔧 技术方法**

采用双阈值积分‑发射(I&F)逻辑、加性伽马噪声模型、闭式误码率与符号间干扰（ISI）分析、MATLAB数值仿真及3D布朗运动粒子模拟。

**📊 数据集**

无公开数据集，使用基于扩散系数(D=0.1,0.3,0.5 μm²/ms)的仿真数据和粒子模拟得到的到达时间分布。

**📈 对比分析**

与传统的按时隙同步的OOK、Z通道和二进制通道基线做对比，证明在各种扩散条件下可提高75%–150%的可达通信速率，且在短符号周期内仍能保持较低的误码率。

**⚠️ 局限性**

主要限制包括：每符号需释放两种分子导致能耗上升；受限于突触清除机制，极高速率可能引起受体饱和和残留分子干扰；并未考虑复杂的受体动力学和能量获取的实际生物限制。

---

## 372. Picard Proximal Monte Carlo for Parallel Bayesian Imaging with Score-Based Generative Priors

**arXiv ID:** 2608.17666 | [PDF](https://arxiv.org/pdf/2608.17666v1)

**作者:** Deliang Wei `[一作]` (Johns Hopkins University), Yu Sun `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于Picard迭代的时间并行后验采样框架（PiX-MC），将传统的前向–后向分裂Langevin动力学与时间并行化相结合，用于大规模贝叶斯图像重建。

**💡 创新点**

创新点在于：①利用前向–后向拆分构造能直接使用问题特定的proximal算子（如测量一致性）的漂移；②通过Picard迭代将采样轨迹在时间维度并行，实现多GPU并行；③进一步提出多块（multi‑block）和温度退火（annealing）变体，兼顾内存、速度与采样质量；④在多种实验中提供理论收敛保证，涵盖非对数凹后验与近似score网络。

**🔧 技术方法**

使用技术包括：前向–后向分裂（FBS）、Proximal Langevin动力学、Picard迭代、score‑based diffusion模型（EDM/PMD等）、多GPU并行实现、熵/ Fisher信息等理论分析。

**📊 数据集**

实验数据集涵盖：合成高斯后验；FastMRI脑部磁共振；CBSD10（Rician噪声）；DIV2K（图像去模糊）；AMOS/LDCT（3D CT）；FFHQ（score网络训练）。

**📈 对比分析**

与传统Langevin、Proximal-Langevin、P‑N‑P Monte Carlo、DPS/DAPS等方法对比；在大规模任务（512×512×80 CT、1024×1024去模糊）中，PiX-MC 通过多块/退火实现 3×–50× 的壁钟加速，且重建 PSNR/SSIM 与基准相当或更好。

**⚠️ 局限性**

局限性：需在GPU上存储完整时间轨迹，块大小与GPU数相互制约；依赖问题特定的proximal算子；理论假设如弱凸性、score误差估计可能对极端噪声或非线性模型不够严谨；对高维度或极稀疏测量的收敛速度与理论一致性尚待进一步验证。

---

## 373. Denoised Variance-Based Pruning with Optimal Brain Bias Compensation

**arXiv ID:** 2608.17657 | [PDF](https://arxiv.org/pdf/2608.17657v1)

**作者:** Geon Tack Lee `[一作]` (Korea Advanced Institute of Science and Technology AI), Kang Eun Jeon `[通讯]` (Korea Advanced Institute of Science and Technology AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的结构化剪枝方法DVBP + OB^2C，能够在不进行后续微调的情况下，将Vision Transformer和ConvNeXt模型的MLP层剪枝至50%及以上时仍保持高精度。

**💡 创新点**

创新点在于：①利用随机矩阵理论和Marchenko‑Pastur分布对激活协方差谱进行去噪，从而得到更可靠的低方差神经元排名；②将均值偏移补偿嵌入Optimal Brain Compression目标，得到与协方差矩阵一致的Hessian，实现闭式多权重恢复；③通过混合低方差得分实现跨层动态剪枝分配。

**🔧 技术方法**

核心技术包括随机矩阵理论（MP分布去噪）、Optimal Brain Bias Compensation（OB^2C）闭式权重更新、在线协方差矩阵计算与批量更新、以及基于混合方差得分的全局剪枝策略。

**📊 数据集**

在ImageNet‑1K上使用8192张无增广的校准图像进行实验，评估DeiT、Swin和ConvNeXt的Tiny、Small和Base变体。

**📈 对比分析**

与VBP、SNIP、Magnitude等基线方法比较，DVBP + OB^2C在50%剪枝时对Small/Base模型的Top‑1精度保持超过90%，比VBP高出多达29.46%（ConvNeXt‑T）或7.33%（Swin‑S），并在所有模型上显著优于其它结构化剪枝方法；在维持相同参数/计算量的前提下，性能提升最大。

**⚠️ 局限性**

局限性包括：仍需使用足量的校准集来估计协方差矩阵；对极高剪枝比例（>60%）时精度下降仍较明显；方法主要针对线性层，非线性层的剪枝效果未知。

---

## 374. Neuro-symbolic learning over OWL 2 DL via consequence-based compilation to differentiable circuits

**arXiv ID:** 2608.17741 | [PDF](https://arxiv.org/pdf/2608.17741v1)

**作者:** Olga Mashkova `[一作]` (King Abdullah University of Science and Technology), Robert Hoehndorf `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文设计了一种从 OWL 2 DL 本体到 Sentential Decision Diagram（SDD）的编译器，并利用 SDD 的 evidence‑conditioned weighted model count（WMC）作为梯度信号，训练卷积网络在仅有部分 ABox 监督的情况下学习隐含概念，同时对多解推理快捷方式（reasoning‑shortcut）进行理论分析与实验验证。

**💡 创新点**

创新点在于：① 将基于 consequence‑based 的推理与知识编译结合，实现非 Horn DL 的完整 WMC 训练；② 提出 JustWMC 混合网络，利用本体的证明结构（justifications）直接枚举多模式后验，从而在存在多重一致完成时实现 Bayes‑optimal 推理；③ 在 Lean4 证明环境中对编译器和表示结果进行形式化验证。

**🔧 技术方法**

使用的技术包括：OWL 载入与 DL‑clauses 归一化；超分辨率（hyper‑resolution）基的 consequence‑based saturation；对非原子构造的 grounding（nominals、数量限制、角色特性）；Satisfiability 与 WMC 计算的 SDD 编译；CNN/ResNet 视觉编码器与 WMC 损失的反向传播；JustWMC 混合模型与 softmax 选择器；Lean4 形式化证明与 PySDD 编译。

**📊 数据集**

主要实验数据集：① 通过 succ、Number、Parity、Prime 等关系构造的 MNIST‑ 任务（含两种监督 regime：grounded 与 under‑determined）；② 通过 ResNet‑18 处理的 Pizzaiolo 角色本体的合成 pizza 图像；③ 结合性别与亲属关系的 MNIST‑Disjunction 任务；同时与 DeepProbLog 的 Horn fragment 对照实验。

**📈 对比分析**

与独立感知、单一 WMC、学习混合、BEARS、DeepProbLog 等方法对比，实验表明：在完全监督的 MNIST‑ 任务中单一 WMC 能实现 99% 的数字识别率并将违规率降至 0.02；在 under‑determined 任务中，单一 WMC 与独立感知无法覆盖多重一致模式，导致高违规率；而 JustWMC 及其锚定版本能够恢复 Bayes‑optimal 近似后验（NLL 4.17、TV 0.02），在多模式 MNIST‑Disjunction 任务中明显优于单一 WMC 与学习混合。

**⚠️ 局限性**

局限性包括：① 仅适用于有限活跃域（finite ABox）并且对存在式的 grounding 覆盖或函数自由才保持精确；② 基于有界 saturation 的推理不完整，可能遗漏深层未 grounding 的存在式链；③ 多个个体的 ABox 会使 CNF 与 SDD 规模呈 O(|Δ|^2) 或 O(|Δ|^3) 成长，限制大规模场景；④ 锚定 JustWMC 需要可枚举的 #RS（一致完成数）小于一定阈值；⑤ 未处理数据类型与完整的开放世界解释；⑥ 需要手动或特定工具（如 Pizzaiolo 的闭合存在式约束）来实现某些闭世界读法。

---

## 375. Rerootable Hypertree Decompositions

**arXiv ID:** 2608.17853 | [PDF](https://arxiv.org/pdf/2608.17853v1)

**作者:** Zhekai Jiang `[一作]` (EPFL), Qichen Wang `[通讯]` (Nanyang Technological University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统研究了超图的高阶树分解（Hypertree Decomposition, HD）中的重根可重性（rerootability）与投影自由性（projection‑freeness）概念，并在此基础上提出了一种松弛的正规形式（relaxed normal form, RNF），使得该类分解既保持可重根，又能在多项式时间内判定宽度。作者进一步给出了基于改造后的“Robber & Marshals”游戏的可行性判定算法，并对其复杂度进行了分析。

**💡 创新点**

创新点包括：
1) 首次将重根可重性与投影自由性联系起来，证明所有可重根HD必为投影自由，反之亦然；
2) 引入新的松弛正规形式，使得该类分解在保持可重根的同时可在多项式时间内计算；
3) 通过改造游戏和交替算法实现了对该类分解宽度的高效判定；
4) 实验验证了宽度增幅在实际查询集上通常不超过 1 或 2，证明该方法在真实场景中的可行性。

**🔧 技术方法**

主要技术手段包括：
- 组合/图论对超图分解的结构分析；
- 投影自由性与重根可重性等价性的形式化证明；
- 通过“Robber & Marshals”游戏的改造构造判定算法；
- 交替（alternating）算法实现可行性搜索；
- 复杂度分析证明该判定问题属于 L 或更低的复杂度类；

**📊 数据集**

实验使用的数据集来自真实数据库查询日志和标准基准集（如TPC‑H、TPC‑DS 等），共计 1079 条超图（包含约束满足问题生成的超图），并与现有的 HD、GHD、H‑query 分解结果进行对比。

**📈 对比分析**

与传统 HD/GHD/H‑query 的宽度比较：在大多数查询上，该松弛正规形式的宽度与原始 HD 相同，只有极少数（约 1/1079）出现 +1 的情况；在复杂度方面，判定时间上实现了 O(|E|²k|V|) 的多项式时间算法，明显低于直接搜索整个分解空间的指数复杂度。实验结果表明该方法在保持可重根性的同时，宽度提升较小，且在实际查询优化场景中可行。

**⚠️ 局限性**

限制与不足：
- 仅针对投影自由且满足 RNF 的 HD 类，无法直接处理更一般的 HD 或 GHD；
- 对于极端结构（如极大连通超图）仍可能出现宽度显著增加；
- 实验范围受限于现有基准集，未覆盖所有可能的工业查询模式；
- 复杂度分析仍属于理论级别，实际实现时的常数与开销尚待进一步优化。

---

## 376. Efficient Resource Optimization for Split Federated Learning

**arXiv ID:** 2608.17849 | [PDF](https://arxiv.org/pdf/2608.17849v1)

**作者:** Wei Wei `[一作]` (University of Hong Kong), Xianhao Chen `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套高效的资源优化框架，用于在资源受限的边缘网络下实现 Split Federated Learning（SFL）的模型切分与资源调度的联合优化；

**💡 创新点**

创新点在于：①给出模型切分问题的多项式时间全局最优算法；②将整体混合整数问题改写为二维主问题，利用网格搜索给出可行的 (1+ε) 近似保证；

**🔧 技术方法**

主要技术包括：多项式时间算法设计、平面扫描与剪枝、凸优化子问题求解、二维主问题的网格逼近与误差分析；

**📊 数据集**

实验采用 MNIST、CIFAR-10 数据集，并使用 ResNet‑50 与 VGG‑16 两种模型进行验证；

**📈 对比分析**

与 5 种基线（OC+OG、OC+OP、OC、ESFL、DSQL）进行对比，结果显示所提 OC+OGP 在达到相同测试准确率时，总成本最低，且在不同 λ、带宽、GPU 频率与不确定性场景下均保持优越性能；

**⚠️ 局限性**

限制在于：理论分析基于可行域包含全局最优点的假设，且实际实现需预先知道系统参数，面对极大规模或强动态变化的环境时可能仍面临计算或收敛问题。

---

## 377. Encoded but Not Actionable: Auditing the Decode-Generate-Steer Gap in Frozen LLMs for Geometric Constraints

**arXiv ID:** 2608.17843 | [PDF](https://arxiv.org/pdf/2608.17843v1)

**作者:** Man Liang `[一作]` (University of Maryland), Faizan Wajid `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

评估冻结的大语言模型在参数化CAD几何约束推理中的表示能力，采用四维审计框架检验线性可解码性、强制式生成、激活层影响及可控性。

**💡 创新点**

提出对几何约束信息进行四级可解释性审计，揭示线性可解码性与模型行为、干预效果之间的显著解耦，且在同一数据集上实现对多种LLM的统一对比。

**🔧 技术方法**

使用线性逻辑回归探测可解码性、强制式文本生成（P3）、激活补丁（activation patching）与均值差异驱动的可控性（steering），并对输入进行几何序列化。

**📊 数据集**

基于SketchGraphs的几何描述数据集（包含配对关系P1、全局自由度状态P2）以及Fusion 360 Gallery作为交叉验证。

**📈 对比分析**

在六款冻结的decoder-only LLM（Qwen2.5、Mistral、Llama-3.1系列）上进行比较；结果显示预训练对局部约束的可解码性提升显著，但对全局自由度影响有限；P3生成性能远低于P1解码，且激活补丁在早层有效但随深度消失，均值差异驱动几乎无可控性。

**⚠️ 局限性**

研究仅覆盖几何序列化输入，未考虑约束注释；P2标签基于启发式自由度计数，可能误标；实验仅对少数模型与干预方式进行，缺乏对不同提示、层级与激活目标的全面探索；数据拆分与随机初始化的单一方案可能影响结果稳健性。

---

## 378. Leveraging Association Context Retrieval in Knowledge Edit- ing to Build White-Box Attacks on LLMs

**arXiv ID:** 2608.17836 | [PDF](https://arxiv.org/pdf/2608.17836v1)

**作者:** Roman Maksimov `[一作]` (Basic Research of Artificial Intelligence Laboratory), Aleksandr Beznosikov `[通讯]` (Basic Research of Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过知识编辑技术，在白盒攻击下实现对LLM对齐的移除，抑制拒绝行为并诱导模型生成可接受的有害回答。

**💡 创新点**

创新点在于将协同关联检索（context association retrieval）用于计算关键向量k*，实现对目标主题的更精准、可迁移的编辑；同时提出了不需要提示注入的攻击方式。

**🔧 技术方法**

核心技术包括：Causal tracing寻找关键层、定位并编辑FFN的键值对、利用k*与v*进行低秩更新、对MoE架构的通用扩展，以及与现有知识编辑框架（如3*、2-5、Qwen-MoE）结合。

**📊 数据集**

使用公开的对齐模型和数据集：GPT4ALL-J、Llama-3-8B-Instruct、Qwen3-30B-A3B；数据来源于Alpaca、AdvBench、TDC2023、HarmBench、StrongREJECT、JailbreakBench，按结构与语义类别划分。

**📈 对比分析**

与拒绝方向基线和多种Locate‑then‑Edit基线对比，实验显示在Harmfulness上提升约0.8–1.0分，Locality保持高于基线，Generalization在结构和语义类别上提升约0.2–0.3分，且在不同模型架构上保持一致性。

**⚠️ 局限性**

局限性包括仅验证了2022–2025年公开模型，未覆盖最新版本；未对所有Locate‑then‑Edit基线的前缀改造做完整测试；攻击需对模型参数具备直接访问，实用性受限；同时，攻击在极端强度下仍可能导致模型失去连贯输出。

---

## 379. Variational r-Adaptive Cloth Simulation

**arXiv ID:** 2608.17833 | [PDF](https://arxiv.org/pdf/2608.17833v1)

**作者:** Jiahao Wen `[一作]` (Adobe), Danny M. Kaufman `[通讯]` (Adobe)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了首个适用于摩擦接触的布料模拟的 r‑自适应方法，解决薄壳离散化中的子最优局部极小值与退化单元导致的低能量陷阱；

**💡 创新点**

创新点在于引入基于退化激活的质量正则化，既抑制不良网格引发的低能量陷阱，又在保持局部可变形时不妨碍有效的自适应密度分布；

**🔧 技术方法**

技术包括基于增量势能（IPC）的变分 In‑Timestep Remeshing（ITR）框架、MIPS 质量正则化、动态内部容差调整与时间步内一致性加速的非线性求解器；

**📊 数据集**

使用多种公开的布料与刚体碰撞场景（如人偶动作、方形布料摆放、尖锐尖峰和 L'Inconnue de la Seine 雕像等），以及对比固定网格的高分辨率实验；

**📈 对比分析**

通过能量降低、形状吻合度、锁定现象消除以及求解时间的对比，r‑自适应方法在相同顶点预算下实现了 3–6 倍速度提升并得到更低能量、更细腻的褶皱与更准确的接触；

**⚠️ 局限性**

局限在于仅适用于平面初始配置，无法处理曲面未变形的薄壳，边界接触分辨率有限，且依赖特定的 FEM 布料求解器。

---

## 380. MotoSafety: Edge-AI with Learned Temporal Importance for Two-Wheeler Collision Risk Assessment Under Time Pressure

**arXiv ID:** 2608.17823 | [PDF](https://arxiv.org/pdf/2608.17823v1)

**作者:** Sumit S. Shevtekar `[一作]` (Indian Institute of Technology Indore), Subasish Das `[通讯]` (Texas State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个129,000条多变量时序数据的大规模两轮车骑行模拟数据集，并在此基础上提出了一种名为MotoSafety的轻量级边缘AI架构用于碰撞风险评估和时序预测。

**💡 创新点**

创新点在于提出了学习型时序重要性（Learned Temporal Importance）原则下的TIP模块，实现了自适应的内容感知时序降维，显著提升了模型对高频动态与长程依赖的捕捉；同时将TP（时间压力）作为先验信息融入模型，进一步提高风险预测准确率。

**🔧 技术方法**

采用了多尺度膨胀卷积+Bi‑LSTM编码器、TIP时序池化、SE‑Block与多头注意力融合的混合结构，并使用了Focal Loss、MSE、AdamW、Mixup、EMA等训练技巧；模型整体为1.15M参数，推理延迟0.135 ms。

**📊 数据集**

使用了51名受试者在静态高保真两轮车模拟器中进行的153次骑行实验（每人三种TP条件）生成的64维时序特征，涵盖车辆动力学、控制输入、接近度与行为违规等，构成129,209个滑动窗口标签。

**📈 对比分析**

与10个基线（RF、CNN、RNN、TimesNet、PatchTST、iTransformer、Informer、Time‑LLM、LLM4TS）对比，MotoSafety在碰撞风险分类上取得94.97%准确率、99.33%ROC‑AUC，预测误差（MSE 0.039、MAE 0.094）比基线低4.4×，且模型参数仅1.15M、推理时间0.135 ms，性能优于其他方法。

**⚠️ 局限性**

主要局限包括：受试者全部为男性，缺乏性别差异验证；数据来自模拟器，尚未在真实交通环境中验证；未收集生理压力指标，TP的心理测度仅为实验设定；系统的实地硬件验证和在线部署仍待后续研究。

---

## 381. Effector-Centric NMPC of Tiltable-Multirotors for Offset-Free Omnidirectional Aerial Manipulation

**arXiv ID:** 2608.17819 | [PDF](https://arxiv.org/pdf/2608.17819v1)

**作者:** Jinjie Li `[一作]` (University of Tokyo), Moju Zhao `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种面向末端执行器的非线性模型预测控制（NMPC）框架，结合倾斜四旋翼的结构设计、奇异性处理、外部力矩估计和模型误差补偿，实现了在倾斜四旋翼平台上实现全方位、无偏移的空中操控。

**💡 创新点**

创新点包括：①在倾斜四旋翼上首次实现可穿越奇异姿态的全方位飞行；②将末端执行器视角嵌入NMPC成本，简化任务规划；③采用积分项与加速度基力矩估计分离处理模型误差与外部扰动，提升精度与鲁棒性；④在同一架构下统一设计分析、参考生成、奇异处理与扰动补偿，形成闭环实验验证的完整体系。

**🔧 技术方法**

核心技术包括：倾斜多旋翼动力学建模、伪逆分配与奇异性修正、积分误差补偿、加速度估计的外部力矩估计、CasADi/Acados实现的实时迭代NMPC、低通滤波同步、端到端硬件（VIM4+STM32）与ROS调度。

**📊 数据集**

实验数据集主要由自主搭建的倾斜四旋翼（“Beetle‑Art‑Omni”）与物理环境中的任务（如白板推拉、阀门连续转动、垂直旋转、斜姿态）构成，无公开大规模数据集使用。

**📈 对比分析**

通过与传统CoG‑centric NMPC、PID、线性模型等方法对比，实验显示末端执行器框架在姿态误差、位置误差上提升约10-15%，并成功完成 360° 俯仰/滚转、连续 360° 阀门旋转、白板推拉等任务；实时频率维持在 100 Hz，计算时延 5 ms 内。

**⚠️ 局限性**

局限性：缺乏柔性接触控制（未实现完全的力/位置闭环柔性），外部力矩估计受加速度噪声影响，精度受限；需要手动校准外部力矩偏置；仅在实验室环境验证，未对风速、障碍物等复杂场景做充分评估；任务范围受限于单一倾斜四旋翼平台与固定执行器。

---

## 382. Threat Aware Task Offloading and Caching for Secure UAV Assisted Vehicular Consumer Electronics

**arXiv ID:** 2608.17794 | [PDF](https://arxiv.org/pdf/2608.17794v1)

**作者:** Xiaoteng Yang `[一作]` (Xidian University), Zheng Lin `[通讯]` (University of Hong Kong)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种针对无人机辅助车载消费电子网络的威胁感知任务卸载与智能时空缓存框架，目标是同时降低任务延迟、提升缓存命中率并确保通信安全。

**💡 创新点**

创新点在于将威胁感知与任务卸载、缓存决策紧密耦合，采用联合优化 TAGO 框架，其中引入了近端策略优化（PPO）实现自适应卸载、Frank–Wolfe 算法实现实时缓存更新，并使用条件变分自编码器（CVAE）生成鲁棒的联合策略。

**🔧 技术方法**

技术手段包括：安全感知的上行传输模型、基于 M‑Zipf 分布的缓存流行度建模、NOMA 多址与 SIC 干扰管理、PPO 强化学习、Frank–Wolfe 梯度搜索、CVAE 生成式优化及混合整数非线性规划的约束模型。

**📊 数据集**

实验数据集基于仿真生成的 10×10 km 城市网格，部署 5–10 台 RSU、1–5 台 UAV、10–30 台车辆；任务到达服从泊松过程，数据量 1–10 MB，计算需求 0.5–2 ×10⁹ CPU 周期，缓存项目大小 5–10 MB。

**📈 对比分析**

与 DO、CO、Greedy、Random、LORA、FDLOA、BC‑A3C‑GS 等基线对比，TAGO 在不同车辆密度与 UAV 数量下平均任务延迟最低，提升幅度约 7–19%，缓存命中率提升至 78–83%，实验结果显示显著优于传统与先进算法。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证；无人机仅考虑单一节点；威胁模型简化为信息泄露概率，未覆盖更复杂攻击；未对能耗与能源管理进行深度分析；缺乏分布式学习与大规模真实部署验证。

---

## 383. Preference Is Not Intervention: The Structure and Stability Boundaries of Reader-Specific Evidence Utility

**arXiv ID:** 2608.17781 | [PDF](https://arxiv.org/pdf/2608.17781v1)

**作者:** Shi Zhou `[一作]` `[通讯]` (Jilin University), Shi Zhou (Jilin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在检索增强生成（RAG）中，使用统一查询、证据、任务和干预条件，系统性测量并比较了不同阅读模型（读者）对证据效用的影响，进一步将效用拆解为活动性、序数偏好和有符号方向，并评估其跨查询的稳定性及其对跨读者干预迁移的意义。

**💡 创新点**

① 首次在完全受控环境下证明读者身份本身能产生可测量且结构化的效用差异；② 将效用拆解为三种可测对象，发现序数偏好在多种设置下稳定，而有符号方向仅在任务受限（如二元事实核查）时稳定；③ 证明稳定的相似性并不支持跨读者干预迁移，挑战了此前关于读者特定效用可迁移的假设。

**🔧 技术方法**

使用留一（leave‑one‑out）和单文档干预的效用张量，方差分解、分半可靠性（split‑half reliability）、稀疏性匹配的置换校准、Spearman 相关、Rank‑based overlap（RBO）以及基于任务的评分（token‑F1、exact/match）等统计与评价技术；此外设计了强迫选择（forced‑choice）扰动实验以探测答案空间对稳定性的影响。

**📊 数据集**

内部数据集：NQ 与 HotpotQA 共 100 个查询；外部数据集：RAMDocs（149 查询，含误导与噪声文档）和 RAGuard（212 事实核查声明）；以及公开 PRISM 偏好数据（58,404 条偏好记录，覆盖 7,791 个查询）。

**📈 对比分析**

在四个独立设置下比较稳定性：序数偏好在所有设置中均表现出较高的分半可靠性（0.60–0.83）；有符号方向在开放式 QA 中低至 0.14–0.35，显著低于稀疏匹配基线；而在二元事实核查中接近 0.75，几乎与序数稳定性相当。自选证据相较于其他读者的平均选择提高了约 +0.031 token‑F1。跨读者干预迁移实验显示，读者相似性与迁移损失无显著相关性（ρ≈‑0.27，p>0.25）。

**⚠️ 局限性**

① 任务边界（为何有符号方向在开放 QA 中不稳定）未被因果阐明；② 不同实验 arm 使用不同读者面板，导致跨设置直接比较受限；③ PRISM 缺乏有符号效用测度；④ 跨读者迁移实验仅使用 50 个查询，样本量有限；⑤ 只基于任务级别指标评估效用，未探索更细粒度的效用度量。

---

## 384. Advancing Inclusivity in Cybersecurity Education: Integrating Intersectionality to Enhance Student Engagement in Australian Higher Education Curriculums Strategies, Barriers, and Future Directions

**arXiv ID:** 2608.17758 | [PDF](https://arxiv.org/pdf/2608.17758v1)

**作者:** Nalin A. G. Arachchilage `[一作]` (RMIT University), Gary Thomas `[通讯]` (RMIT University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过半结构化访谈分析了澳大利亚高校网络安全课程中女性、性别多样化与CALD群体视角的融入现状与面临的四大障碍与四个支持需求。

**💡 创新点**

首次系统识别并归纳了缺乏系统化方法、资源与政策支持的四大障碍，并提出了交叉性导向的课程设计与实践框架。

**🔧 技术方法**

采用定性主题分析法进行数据编码与归纳。

**📊 数据集**

使用了15名澳大利亚高校网络安全学术人员的访谈文本数据集。

**📈 对比分析**

通过主题编码比较发现缺乏量化指标与标准，未能评估实施效果与性能提升。

**⚠️ 局限性**

局限在于样本规模仅为15名学者，缺乏成员检验与跨文化推广，难以普遍推广。

---

## 385. Interpretable Humans, Alien LLMs: Expert Analysis of Latent Structures in Assessment Responses

**arXiv ID:** 2608.17810 | [PDF](https://arxiv.org/pdf/2608.17810v1)

**作者:** Alona Strugatski `[一作]` (Weizmann Institute of Science), Giora Alexandron `[通讯]` (Weizmann Institute of Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了人类与多模态大型语言模型（LLM）在同一教育测评工具上的潜在因子结构，并通过盲法专家解释来检验这些因子是否具有可解释的教育意义。

**💡 创新点**

创新点在于将探索性因子分析（EFA）与盲法专家解释相结合，首次系统展示LLM的因子往往无法被人类专家赋予教育层面的意义，从而揭示LLM与人类认知机制的显著差异。

**🔧 技术方法**

主要技术包括探索性因子分析（EFA）、tetrarchic相关矩阵构建、最小残差估计与斜交旋转，以及盲法专家评估流程。

**📊 数据集**

使用了两套教育测评数据：一套为高中化学诊断22题，另一套为大学入学定量推理20题的人类答题记录，以及来自六款多模态LLM（Claude、GPT、Gemini）的合并答题数据。

**📈 对比分析**

通过对人类与LLM的因子载荷结构进行对比，发现人类因子能够被专家解释，而LLM因子大多不可解释；LLM整体得分与人类相近但因子结构差异显著，表明LLM的“技能”与人类认知机制不一致。

**⚠️ 局限性**

局限性包括：数据集非公开、LLM样本量相对较小、对不同LLM进行合并处理缺乏细粒度分析、专家评估样本有限且未进行一致性检验、仅使用两套单语言测评工具，导致外部有效性受限。

---

## 386. Scale Matters: Adaptive Granularity Selection for Cross-Species 3D Plant Organ Segmentation

**arXiv ID:** 2608.17803 | [PDF](https://arxiv.org/pdf/2608.17803v1)

**作者:** Carla Salazar `[一作]` (Technical University of Denmark), Lazaros Nalpantidis `[通讯]` (Technical University of Denmark)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种基于 Utonia 3D 基础模型的少量样本植物器官分割方法 AGS-PlantSeg，利用自适应粒度选择（AGS）动态挑选最合适的特征粒度并通过轻量 MLP 头完成分割。

**💡 创新点**

核心创新是将粒度自适应与基准模型冻结结合：通过原型评分机制在训练与推理阶段分别评估互类分离、内类紧凑与边界一致性，从而为每棵植物自适应选择特征尺度；实现了跨物种、跨数据集的鲁棒性。

**🔧 技术方法**

使用的技术包括：Utonia 3D 预训练编码器、原型（prototype）计算与边界过滤、基于余弦距离的粒度分数（inter‑class, intra‑class, boundary），以及基于置信度加权的多尺度融合。

**📊 数据集**

实验数据集：PLANesT-3D（pepper, ribes, rose）、Pheno4D（tomato）和 Crops3D（tomato、potato）。

**📈 对比分析**

与 PointNet++、RoseSegNet、SP‑LSCnet、GCASSN 等基线对比，在全样本、少样本（few‑shot、one‑shot）设置下，AGS‑PlantSeg 在 PLANesT‑3D 的 mIoU 最高达 96.3%，在未见物种/数据集上仍保持 82–85% 的 mIoU，明显优于固定粒度方案（平均 88.9% mIoU），并在少量标注下几乎匹敌全监督模型。

**⚠️ 局限性**

局限性：对某些复杂物种（如 rose）表现略逊，粒度选择仍可进一步优化；当前仅支持叶/茎两类；推理时粒度搜索仍耗时约 15 s，需更高效的搜索与并行化。

---

## 387. Fourth-Moment Geometry of Rademacher Sums

**arXiv ID:** 2608.17802 | [PDF](https://arxiv.org/pdf/2608.17802v1)

**作者:** Peigan Gao `[一作]` (University of Hong Kong), Jian Qian `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了独立 Rademacher 符号加权和的高阶矩，给出了以第四阶质量(q)为参数的精确上界，并利用此框架证明了从 4 维开始的 Gaussian 稳定性不等式、有限维 Lp/L4 常数的闭合式以及在 p=3 时的维数无关二次稳定性。

**💡 创新点**

创新点包括：① 对 4≤p<5 区间内缺失的 Gaussian 稳定性不等式给出了完整证明；② 通过固定-矩原理首次确定了高阶矩的极值分布为“一 spike + Gaussian 云”；③ 证明了有限维 Lp/L4 常数的极大值在 “flat” 向量处取得，从而解决了 Barański‑Murawski‑Nayar‑Oleszkiewicz 的猜想；④ 在 p=3 时给出维数无关的二次稳定性上界，并提供了最佳常数的下界。

**🔧 技术方法**

主要技术手段包括：高阶矩的留一法估计、四阶凸性与 Riemann–Stieltjes 积分、Riccati 比较与凸性论证、固定-矩原理（将极值问题转化为“spike+Gaussian”极值）、多维立方体平均化与小球估计、以及利用 Hermite–Hadamard 及二阶微分不等式进行精细分析。

**📊 数据集**

本文不涉及实验数据或数据集，所有结论均为严格的概率/分析证明。

**📈 对比分析**

由于是理论研究，未进行实验比较；结果以数学不等式和极值定理形式呈现，证明了所给不等式在所有满足假设的情形下最优，且在特定极值点（如单个非零系数或平坦向量）取得等号。

**⚠️ 局限性**

局限性：对 p=3 的维数无关二次稳定性常数的精确值仍未完全确定；此外，对于 0<q<1 的固定‑q 上界仅在有限 Rademacher 和的闭包中取得，尚未找到闭合式的有限维极值分布。

---

## 388. StartupBench: Benchmarking General-Purpose Agents on Market-Validated End-to-End Workflows

**arXiv ID:** 2608.17800 | [PDF](https://arxiv.org/pdf/2608.17800v1)

**作者:** Liya Zhu `[一作]`, Wenhao Huang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 StartupBench 基准，专注于评估模型在真实 AI 产品工作流中的端到端完成情况。

**💡 创新点**

创新点在于：① 任务来源于市场验证的 AI 原创创业产品并结合用户访谈与专家构造；② 采用细粒度 rubric 与 Agent-as-Judge 的评估框架；③ 通过多阶段质量控制确保任务真实性与可评估性。

**🔧 技术方法**

技术手段包括：多阶段调查-访谈-专家构造任务流程；利用 GPT‑5.5 作为 Judge 进行细粒度 rubric 评估；在 Nanobot、Hermes、Claude Code 等通用 Agent 框架下进行实验；使用自动化工具进行文件与证据视图生成。

**📊 数据集**

数据集为 StartupBench，包含 97 个跨 6 大领域（医疗、金融、法律、商业、STEM、教育）的真实工作流任务，输出格式覆盖 DOCX、XLSX、PPTX、PDF、Markdown、图像等多种。

**📈 对比分析**

实验对 9 个代表性模型（如 GPT‑5.6、Kimi‑K3、Seed‑2.1‑Pro 等）进行 3 次独立跑测，报告平均分与成功率；最高平均得分约 73%，但成功率低于 35%，与专用创业 Agent 的 83%/39% 对比显示显著差距。

**⚠️ 局限性**

局限性包括：任务来源单一（仅 AI 原创创业产品）、rubric 设计带有主观性、评估过程耗时且易出错、模型在领域专属知识、复杂指令遵循与专业规范方面表现不足，导致难以满足真实工作需求。

---

## 389. Diff-DDoS: Realistic Cyber-Physical Attack Synthesis and Robust Detection for 5G-Enabled CPS Using Tabular Diffusion Models

**arXiv ID:** 2608.17796 | [PDF](https://arxiv.org/pdf/2608.17796v1)

**作者:** Bilal Hussain `[一作]` (Hong Kong Polytechnic University), Danista Khan `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出 Diff-DDoS 三阶段框架，利用 Tabular Diffusion 生成逼真攻击样本，并通过逆向分类器引导实现对 5G CPS 的 DDoS 检测模型的鲁棒性提升。

**💡 创新点**

创新点在于①使用 Tabular Diffusion 生成近正态分布的攻击样本；②采用逆向分类器引导的 Adversarial Diffusion 训练；③结合阈值校准解决稀疏攻击的决策阈值问题。

**🔧 技术方法**

采用的技术包括卷积神经网络（SimpleCNN 与 ResNet50）、TabDDPM（去噪扩散概率模型）、逆向分类器引导、Grad‑CAM 解释以及阈值校准。

**📊 数据集**

实验基于 Telecom Italia Milano CDR 数据集（9×9 网格、10 分钟间隔）进行训练、测试及攻击生成。

**📈 对比分析**

与固定倍率攻击、FGSM/PGD 对抗训练、CTGAN 等方法在相同迭代训练与阈值校准下比较，ADT 在 ResNet50 上在 SMS/Internet/混合场景下均实现 90% 以上 F1，Internet 场景达到 100%，且保持多倍率攻击的高精度。

**⚠️ 局限性**

主要限制包括：仅使用合成攻击样本且缺乏真实标签；评估仅基于聚合层 3D 统计的逼真度，未考察完整格子时空结构；数据来源于 3G/4G 城市流量，未验证在 5G 或更大规模网络的泛化能力。

---

## 390. Training-Free Human-in-the-Loop Anomaly Detection via Memory Bank Correction

**arXiv ID:** 2608.17775 | [PDF](https://arxiv.org/pdf/2608.17775v1)

**作者:** Ayusha Abbas `[一作]` (Newcastle University), Kabita Adhikari `[通讯]` (Newcastle University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练‑free 的人机交互异常检测框架，利用工人对误报/漏报的二值判断直接编辑 PatchCore 的内存库，无需梯度、重新训练或原始训练数据。

**💡 创新点**

创新点在于：① 通过直接内存库编辑实现即时纠错；② 引入自校准的 novelty gate 防止冗余插入；③ 在冷启动时仅用十张“金色”样本加人工纠错即可达到或超过全训练模型；④ 证明主动查询在此场景下与被动查询无显著差异。

**🔧 技术方法**

技术实现包括 PatchCore 的 k‑center coreset内存库、欧氏距离最近邻检索、平均9最近邻的异常评分、novelty gate 阈值、自定义 AUCC 评价指标、被动/主动查询策略以及基于 20 个随机种子进行多重检验。

**📊 数据集**

实验使用 MVTec AD 工业缺陷检测基准数据集，共 15 类，涵盖 5 纹理类和 10 物体类。

**📈 对比分析**

评价方法为 20 份独立划分的 held‑out 评估，采用 Holm‑corrected Wilcoxon 检验；结果显示冷启动时平均恢复 80% 的全训练缺口，成熟库中 4 类显著提升 0.04–0.15 AUROC；主动查询与被动查询无统计差异；与传统全训练模型相比，校正后性能在多数类别上大幅提升或不受损失。

**⚠️ 局限性**

局限性包括：仅适用于记忆库类模型，无法直接处理像素级缺陷；评估基于模拟专家标签，真实工人误差尚未验证；对全局结构缺陷（如 cable_swap）修正效果有限；缺乏现场工业部署与长期稳定性实验。

---

## 391. Efficient Fuzzy PSI under One-Sided Assumptions

**arXiv ID:** 2608.17770 | [PDF](https://arxiv.org/pdf/2608.17770v1)

**作者:** Xinpeng Yang `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一套基于对称密钥原语、仅依赖一侧空间分离假设的通用 L_p (1≤p≤∞) 模糊 PSI 协议，并实现了多点模糊匹配的基础构造。

**💡 创新点**

创新点包括：① 用多点模糊匹配实现无需两侧约束的模糊 PSI；② 将前缀树技术引入，首次将阈值 δ 的复杂度从 O((log δ)^d) 或 O(δ) 降低到 O(log δ)；③ 通过条件随机化和前缀选择消除输出冲突，保持隐私。

**🔧 技术方法**

采用的核心技术有：so‑OPPRF（共享输出的可编程伪随机函数）、空间哈希、条件随机化、前缀选择（ConSel）、秘密共享等价/比较测试、以及 Silent OT 作为 OT 的高效实现。

**📊 数据集**

实验使用了规模可调的合成数据集（点集大小 m=n ∈ {2^8,…,2^16}，维度 d ∈ {2,4,6}，阈值 δ ∈ {32,…,512}），未公开使用任何真实世界数据集。

**📈 对比分析**

与现有工作（van Baarsen & Pu、Dang et al.、Bui et al.）在相同一侧假设下进行比较；实验结果显示在相同安全模型下，所提协议在通信量上可达 20×-282× 的降低，在计算时间上可达 4×-4818× 的加速；前缀优化版本进一步将通信量压缩至 log δ 级别，同时保持较低的运行时。

**⚠️ 局限性**

局限性：① 仍依赖一侧的空间分离假设（唯一单元/唯一块），不适用于完全无假设的场景；② 仅在半诚实模型下证明安全，未针对恶意攻击者；③ 在极高维度或 δ 极大时，前缀树和空间哈希的内存占用与计算仍可能成为瓶颈。

---

## 392. Learnware for CSI Feedback: Scene-specific Small Models Can Do Big

**arXiv ID:** 2608.17760 | [PDF](https://arxiv.org/pdf/2608.17760v1)

**作者:** Xiangyi Li `[一作]` (Southeast University), Zhi-Hua Zhou `[通讯]` (Nanjing University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于 Learnware 的 CSI 反馈模型仓库，利用场景特定的轻量级模型实现快速、低成本、隐私友好的 6G/5G CSI 压缩部署。

**💡 创新点**

创新点包括：① Learnware 概念，将模型与语义+统计规范绑定；② 采用 DFT 码书指纹（PDF）做统计规范，避免原始 CSI 传输；③ 可调速多级 LSH 检索实现 sub‑millisecond 模型获取；④ 基于检索结果的自适应微调阈值，进一步降低本地训练成本。

**🔧 技术方法**

技术手段：深度学习自编码器（CNN/Transformer）压缩 CSI；DFT 码书投影生成指纹；余弦相似度/其他分布相似度做检索；多级 LSH 结构；模型微调、指标 SGCS、性能评估。

**📊 数据集**

数据集：QuaDRiGa 生成的 3GPP‑38.901‑UMi LOS/NLOS 模拟数据（144 组场景，每组 10k 样本）；同一数据集用于训练、验证、测试；在多载波 OFDM 场景下扩展至 200 个 Learnware 模型。

**📈 对比分析**

对比方法：单一 General Model、Model Switch、Original Model Repository、Learnware Repository。实验显示 Learnware 在 LOS 场景提升 18.8%，NLOS 场景提升 57.7%，检索准确率 90%+；相比 General Model 微调样本减少 300–1000 份、训练轮次 100；检索延迟 <1.15 ms，通信开销大幅下降。

**⚠️ 局限性**

局限性：依赖仿真数据，真实环境下性能需验证；在强 NLOS 散射或角分布稀疏时 PDF 匹配效果下降；模型库规模扩展仍需进一步评估；仍需持续更新模型以应对新场景。

---

## 393. MAGPIE-Net: Predicting short-duration heavy-rainfall events in station neighborhoods from multitemporal FY-4A AGRI observations

**arXiv ID:** 2608.17753 | [PDF](https://arxiv.org/pdf/2608.17753v1)

**作者:** Xiang Lin `[一作]` (National University of Defense Technology), Jing Sun `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了一种名为 MAGPIE‑Net 的卫星到站点的短时强降雨预警模型，直接从多时相 FY‑4A AGRI 的云顶温度和水汽特征预测目标站点附近 1 小时降雨是否超过阈值。

**💡 创新点**

核心创新点包括：①将预警目标转化为事件导向的卫星到站点路径，消除传统模型中预报字段与最终预警目标之间的间接关系；②引入可微分的 GA‑SetConv 解码器，按站点位置、时延和地理属性自适应地聚合格点信息；③在网络中加入 CI（对流发育）特征构造、辅助降雨诊断以及多头事件预测，实现对不同阈值与邻域半径的联合监督。

**🔧 技术方法**

技术手段包括：多时相 FY‑4A AGRI 数据的 CI 特征提取（基于云顶冷却、光谱差异和运动补偿）；UNet++ 多尺度编码网络；辅助 1‑h 降雨诊断头；GA‑SetConv 空间映射；多头事件预测模块；以及联合损失（CSI 事件损失、降雨回归损失和网格诊断损失）训练。

**📊 数据集**

使用了 FY‑4A AGRI 的 8 通道（7–14 号）连续扫描序列与中国国家气象站及区域自动站的 1‑h 降雨观测；训练集覆盖 2018‑2021 年，验证集 2022 年，独立测试集 2023 年的夏季高温季节，样本总量约 26,000 例训练，3,000 例测试。

**📈 对比分析**

与 EarthFormer、PhyDNet、NPM、NowcastNet 等基准在相同四帧卫星输入下进行对比，基准先产生网格降雨字段再通过半径聚合得到站点事件。MAGPIE‑Net 在 0–3 小时预报窗口的 CSI 明显优于所有基准，尤其在 40 km/20 mm h⁻¹ 主事件定义下提升约 30–40%；在更严格的 10 km/50 mm h⁻¹ 定义下仍保持 10–20% 的 CSI 增益；早期预警阶段（前 1 mm 降雨）检测率提升至约 60%，平均预警时间比最佳基准提前 20–30 分钟。

**⚠️ 局限性**

局限性包括：①对极端局部高强度降雨的预测仍受限于云顶信号与地面降雨的空间位移；②模型仅利用卫星云顶和水汽信息，缺乏低层水汽、风场、雷达反射率等近地面物理过程的约束；③在不同气候区域、不同卫星平台上的迁移性能尚未充分验证，需进一步融合多源观测数据以提升普适性。

---

## 394. AdaLens: Interactive Storyline for Monitoring and Steering Long-Running Agentic Data Analysis

**arXiv ID:** 2608.17834 | [PDF](https://arxiv.org/pdf/2608.17834v1)

**作者:** Yangtian Liu `[一作]` (Zhejiang University), Yingcai Wu `[通讯]` (Zhejiang University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了 AdaLens，一套交互式可视化系统，用于实时监控和引导长时序的代理式数据分析工作流。

**💡 创新点**

创新点在于：①采用故事线式可视化统一计划、执行进度、发现与数据列的多粒度展示，实现可观测性与基于可视元素的精确引导；②引入即时指令（Focus、Ignore、Elaborate）和线程生命周期控制的交互，让分析师能直接在可视元素上操作，而非仅靠自然语言。

**🔧 技术方法**

技术实现包括：React‑TS 前端与 Flask 后端、基于 StoryFlow 的渐进式故事线布局、Orchestrator–Worker 架构（利用 GPT‑4 等 LLM 进行规划与执行）、多视图协调（聊天、故事线、检查器）与直接操纵交互。

**📊 数据集**

数据集方面：案例研究使用推特社交媒体数据集（含 hot_degree 等指标）和 NBA 球赛跟踪数据集；用户研究中使用学生成绩数据和视频游戏销售数据。

**📈 对比分析**

通过两项案例研究和 12 名受试者的任务研究，SUS 平均 87.08，学习性 87.76，易用性 84.38；用户认为 AdaLens 在可观测性、信任度和交互效率上明显优于纯文本聊天，未进行量化性能对比，但整体用户体验显著提升。

**⚠️ 局限性**

局限性包括：评估仅涵盖短周期案例，无法验证长期运行的可扩展性；故事线主要通过列关联展示线索，缺乏语义、因果或更细粒度的关联；随着运行时间增长，布局可读性和导航维护成为挑战；后端总结质量对可视化效果有较大影响，需要进一步改进。

---

## 395. An improved bound for the randomized metric distortion problem

**arXiv ID:** 2608.17863 | [PDF](https://arxiv.org/pdf/2608.17863v1)

**作者:** Fabian Frank `[一作]` `[通讯]` (Technische Universität München), Fabian Frank (Technische Universität München)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的随机投票规则 Mixed Integrated Veto（MIV），将 Integrated Veto 与 Maximal Lotteries 以等比混合，专门针对度量失真模型设计；

**💡 创新点**

创新点在于首次证明该混合规则在所有投票配置下实现了最优 5/2 的度量失真率，突破了先前的 2.75271 上界；

**🔧 技术方法**

采用了度量失真分析框架、Simultaneous Veto 过程、最大化彩票理论以及针对集合的失真证书等技术手段进行严格证明；

**📊 数据集**

无实验数据集，研究完全基于理论推导和构造的证明实例；

**📈 对比分析**

与以往最佳随机规则（如 Maximal Lotteries 与 Random Dictatorship 的混合）比较，MIV 在所有可能配置下都达到了 5/2 的失真率，并给出了与此上界匹配的下界示例；

**⚠️ 局限性**

局限在于混合比例固定为 1:1，未探讨按投票配置动态调节权重或混合多于两种规则的潜在改进空间。

---

## 396. UniVerse: Benchmarking and Enhancing LALMs on Culturally Inclusive Low-Resource Music Understanding

**arXiv ID:** 2608.17852 | [PDF](https://arxiv.org/pdf/2608.17852v1)

**作者:** Ziya Zhou `[一作]` (HKUST), Yike Guo `[通讯]` (HKUST)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a4b10f5d-130b-4e77-9367-6469ec621899` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 UniVerse 系统，包含面向低资源民俗音乐的评测基准 UniVerseBench（5,042 个 QA 对）和可自动生成的训练集 UniVerseSet（113k 轮对话），并在此基础上对大语言模型进行后训练，探索多种不平衡学习策略。

**💡 创新点**

创新点在于：① 通过专家指导与自动化流程相结合的可复现管线，系统性地构建跨 38 语种、涵盖 372 首音频的多模态问答基准；② 提出了全自动的多轮对话生成策略，显著扩增低资源音频的训练样本；③ 在 Dense 与 MoE 结构下分别设计语言重加权、文本/音频 DPO 与 REPA 对齐等三种不平衡学习方法，揭示其对跨文化推理的不同影响。

**🔧 技术方法**

技术方法包括：大规模音频-文本多模态预训练、后训练的思考式 (thinking‑mode) 微调、语言重加权交叉熵、文本与音频的 DPO 优化、REPA（Encoder‑side Representation Alignment）与递归潜在对齐等。

**📊 数据集**

使用的数据集为：UniVerseBench（5,042 QA 对，覆盖 38 种语言）、UniVerseSet（113,023 轮对话，510,078 QA，来自 36 种语言的 372 首音频），以及公开的基线模型数据（Kimi‑audio、Music Flamingo、Gemini 3 Flash 等）。

**📈 对比分析**

与公开基线（如 Gemini 3 Flash、Qwen3.5‑Omni‑Plus）对比，后训练的 Qwen2.5‑Omni 与 Qwen3‑Omni 在多项评测维度上分别提升约 14.9% 与 5.9%，在整体准确率上可达 48.8% 与 53.4%。但提升并未随训练量呈正相关，且在某些语言或地区出现性能退化。

**⚠️ 局限性**

局限性包括：① 仍难以捕捉细粒度的声学特征，导致对微调音高、节奏细节的识别不足；② 后训练对高资源语言可能产生负面迁移；③ 评测仍以文本答案为主，缺乏真实场景下的交互验证；④ 依赖人工审核，尚未完全实现完全自动化。

---

## 397. MoRAX: Mobility-based Representation Augmentation for Geospatial Foundation Models

**arXiv ID:** 2608.17848 | [PDF](https://arxiv.org/pdf/2608.17848v1)

**作者:** Ya Wen `[一作]` (University of Hong Kong), Alec Kirkley `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出 MoRAX 框架，利用人类流动信息对地理空间基础模型进行功能结构增强，生成功能感知的城市区域表示，并支持在无移动数据城市的零样本部署。

**💡 创新点**

创新点包括：① 用轻量级特征级调制（FiLM 风格）根据流动关系动态重塑基础嵌入；② 设计教师-学生蒸馏方案，使学生仅用地理距离等无流动数据即可近似教师调制；③ 通过此机制实现跨城市、跨国、跨基座模型的通用迁移。

**🔧 技术方法**

采用了预训练的 EO 基础模型（AlphaEarth、RemoteCLIP）、图神经网络流动图编码、FiLM 形式特征调制、知识蒸馏、基于关系的自监督损失、轻量 MLP 投影和距离采样等技术。

**📊 数据集**

使用了七个中国城市（北京、深圳、杭州、南京、苏州、上海、广州）和两座美国城市（纽约市、芝加哥）的流动图、AlphaEarth 嵌入、H3 网格或行政区划，并收集了城市犯罪、夜间灯光、碳排放、PM2.5 等下游任务数据。

**📈 对比分析**

通过与 AlphaEarth、RemoteCLIP、HREP、FlexiReg、UrbanCLIP 等无监督或迁移基准比较，MoRAX-Teacher 在目标城市上平均提升 R² 约 37%–202%，MoRAX-Student 亦显著优于基础嵌入并超越大多数基准，表明跨国迁移有效。

**⚠️ 局限性**

局限性：对源城市多样性依赖较强，微观交互模式（如犯罪、签到）在无流动数据情况下难以完全恢复；教师需要流动图，学生在缺乏高质量代理关系时性能下降；调制参数主要在训练城市学习，跨域泛化可能受限。

---

## 398. The Model's Tell: Measuring Context-Leakage Attack Signals with Behavior Gauges

**arXiv ID:** 2608.17829 | [PDF](https://arxiv.org/pdf/2608.17829v1)

**作者:** Maosen Zhang `[一作]` (Tsinghua University), Han Qiu `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种在LLM预填充阶段通过附加语言“量表”后缀来提取泄露攻击信号，并使用轻量级MLP将这些概率映射为泄露风险分数；随后将该方法作为输入侧检测器部署，实时识别系统提示或检索增量生成中的泄露攻击。

**💡 创新点**

创新点包括：①首次证明在不进行任何解码的预填充阶段即可获得可靠的泄露攻击信号；②提出内容无关的行为量表（Behavior Gauge）能在跨语言、跨目标（verbatim/semantic）等分布外场景下保持高鲁棒性；③将该量表与激活向量的对齐实验相结合，揭示可观测概率信号与内部泄露相关方向的一致性；④实现极低的部署成本（<0.5K参数、≈10 ms延迟）并兼容主流推理引擎。

**🔧 技术方法**

核心技术包括：①预填充概率（log‑prob）提取、②一层ReLU MLP风险评分、③两种量表设计（Exact Gauge 与 Behavior Gauge）、④激活 steering 与 ROUGE‑L 评估、⑤跨模型、跨语言、跨目标的实验设计。

**📊 数据集**

使用了公开数据集和攻击模板：系统提示泄露场景使用 212 条公开系统提示 + Natural Questions 生成的 benign 查询；检索增量生成泄露场景使用 950 条 LeakDojo 检索实例；攻击模板来自 Raccoon（44）与 LeakDojo（39）；模型覆盖 11 大型 LLM（8 B–2.8 T）。

**📈 对比分析**

与 PromptGuard‑2、PIGuard、LLM‑as‑a‑Judge、Attention‑Tracker、I'vDtL 等基线对比，LeakGauge 在 AUROC 上达到 0.944–0.996，F1 为 0.983，TPR@FPR5 为 0.963，且平均额外延迟仅 10.34 ms、参数增量 <0.5 K，显著低于激活/隐藏状态基线（需要数 GB 模型复制或数百 ms 延迟）。

**⚠️ 局限性**

局限性：①依赖推理引擎暴露的预填充 log‑prob，某些商业部署可能不支持；②量表与 MLP 需要针对目标 LLM 微调，跨模型迁移仍存在一定波动；③实验主要聚焦系统提示和检索增量泄露，其他泄露类型（如对话记忆泄露）尚未系统验证；④在极端自适应攻击（直接针对量表输出）下性能需进一步评估。

---

## 399. From Global Benchmarks to Local Evaluations: Benchmarking LLMs for the German Public Sector

**arXiv ID:** 2608.17827 | [PDF](https://arxiv.org/pdf/2608.17827v1)

**作者:** Camilla Dalerci `[一作]` (Bundesdruckerei GmbH), Daniel Weinland `[通讯]` (Bundesdruckerei GmbH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并首次在德国公共部门场景下，对39个大型语言模型（LLM）进行综合评估，覆盖能源消耗、供应商透明度和政治知识三大治理维度；

**💡 创新点**

创新之处在于构建了专门面向公共行政的全维度评估框架MÖVE，强调模型在资源效率、信息披露和本土化知识等非性能指标上的适用性；

**🔧 技术方法**

技术手段包括使用EcoLogits对推理时能源消耗进行估算，结合人工与自动化相结合的21问透明度矩阵进行信息披露打分，以及以党派官方立场为准则的分类准确度评估；

**📊 数据集**

所用数据集涵盖九个德语任务集（四个摘要、三个问答、两个主题抽取）和4788条来自64个德国政党的官方立场数据（Wahl‑O‑Mat），以及一套21道透明度评估问题；

**📈 对比分析**

比较方法采用每个模型的平均能源消耗、透明度得分（满分42）以及政治知识准确率；结果显示能源消耗差异达63倍，透明度得分介于18–38，政治知识最高准确率仅为0.671，且无单一模型在所有维度上均优；

**⚠️ 局限性**

局限性包括仅评估了三维度、固定模型集、能源估计非真实测量、透明度评估基于公开文档且时间点局限、数据集仅代表部分德语公共部门任务，无法覆盖全部行政场景。

---

## 400. An Empirical Study of Reward Specification and Benchmark Reliability in GRPO-based LLM Unlearning

**arXiv ID:** 2608.17804 | [PDF](https://arxiv.org/pdf/2608.17804v1)

**作者:** Rubén Balbastre `[一作]` (University of Valencia), Mariano Pérez `[通讯]` (University of Valencia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Qwen2.5 模型上使用 GRPO 进行目标知识去学习，并探究不同奖励设计与 SFT 热启动对最终行为端点的影响。

**💡 创新点**

将有用的广义回答作为去学习的行为层级目标，系统比较四种奖励（词汇抑制、反拒绝、基准化裁判、对照拒绝）与 SFT 热启动，揭示传统 RWKU 指标无法完全评估行为端点。

**🔧 技术方法**

采用 GRPO（RLVR）+ LoRA 微调+ SFT 预热+ 多种奖励代理（词汇检测、拒绝检测、LLM 裁判）以及 RWKU、held‑out 审计和终端 rollout 审计等技术。

**📊 数据集**

使用 RWKU 生成的 QA 提示作为训练和评估集，持有的 held‑out 审计分割，并结合 HuggingFace 集合与 GitHub 代码。

**📈 对比分析**

通过比较 RWKU 忘记分数、流利度、事实性、泄漏率、拒绝率以及批量审计的广义回答效果，发现不同奖励在 RWKU 上表现相似，但在审计中显著区分，表明奖励设计对行为影响显著。

**⚠️ 局限性**

评估仅基于确定性解码和模型判定，缺乏对恢复攻击、抽样泄漏和裁判偏差的全面考量；奖励代理与裁判仍可能出现漏洞，难以完全捕捉目标去学习目标。

---

## 401. TraceSQL: Traceable Answerability Estimation for Reference-Free Text-to-SQL Verification

**arXiv ID:** 2608.17795 | [PDF](https://arxiv.org/pdf/2608.17795v1)

**作者:** Neelesh Kumar Shukla `[一作]` (Oracle Corporation), Viji Krishnamurthy `[通讯]` (Oracle Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种轻量级、可追踪的文本转SQL验证器TraceSQL；

**💡 创新点**

通过构建67维诊断特征（涵盖歧义、规划、配对分析、SQL结构与意图对齐），实现可解释的验证决策；

**🔧 技术方法**

利用LLM（如GPT-4）生成诊断证据、SQLGlot解析AST、Extra Trees分类器完成学习与推理；

**📊 数据集**

在BIRD ORM训练集（约2,000条样本）及11个独立BIRD开发数据库上进行评估；

**📈 对比分析**

相较于GradeSQL-7B，TraceSQL在F1上提升4.6个百分点、AUC提升6.2个百分点，整体表现更佳；

**⚠️ 局限性**

主要局限在训练样本规模较小、仅在BIRD数据上验证，未评估跨域迁移与大规模训练效果。

---

## 402. ETHEREAL: A 25.6-$μ$s/inf. Low-latency Event-driven Graph-neural-network Processor for High-resolution Vision at the Edge

**arXiv ID:** 2608.17787 | [PDF](https://arxiv.org/pdf/2608.17787v1)

**作者:** Adrian Kneip `[一作]` (Delft University of Technology), Charlotte Frenkel `[通讯]` (Delft University of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一款名为 ETHEREAL 的事件驱动图神经网络（EV-GNN）处理器芯片，支持高分辨率动态视觉传感器（DVS）数据的低延迟推理。

**💡 创新点**

创新点包括：①邻域并行的分段样条卷积引擎，可在 4/8 位可配置精度下高效执行稠密-稀疏混合计算；②分离的 3D/2D 内存层次结构，配合事件缓存与时间局部性缓存大幅降低外部内存访问；③针对 DAGr-GNN 的完整硬件实现，包括图构建、池化和节点更新，支持异步事件流；④通过多层预编译配置和并行图构建/卷积调度，进一步压缩事件级延迟。

**🔧 技术方法**

使用的技术包括：分形样条卷积（SplineConv）与线性旁路，双模式 MAC 核；可编程 4/8 位权重与特征；基于 LVT/SVT 的 28 nm CMOS 实现；专门的 3D 缓存与 2D scratchpad；自定义 SPI‑类高速接口与 FPGA 辅助 DRAM 模拟；精度感知量化训练（QAT）。

**📊 数据集**

主要数据集：DSEC（640×480 事件摄像头），另外在实验中使用 N-CARS 与 N-Caltech101 等低分辨率基准。

**📈 对比分析**

与现有基准比较：在 DAGr‑GNN 任务上，ETHEREAL 在 0.95 V 时实现 25.6 µs 的事件推理延迟，能耗 1.6 µJ/事件；相比 GPU、传统 FPGA 方案以及先前的 EV‑GNN 处理器，延迟提升 20–100 倍，能耗降低 2.5–10 倍；在高分辨率 DSEC 任务上实现了前所未有的可扩展性。

**⚠️ 局限性**

局限性：①3D 缓存仍受关联度与容量限制，极高通道数的 3D 层仍可能导致 DRAM 访问；②MP 核在邻域稀疏时利用率不高；③尚未完成完整的 CNN‑EV‑GNN 融合与异构调度，系统级整合仍需进一步研究；④低电压时 SRAM 漏电显著，影响能耗曲线。

---

## 403. Stability Control for Real World Testing in Autonomous Racing

**arXiv ID:** 2608.17779 | [PDF](https://arxiv.org/pdf/2608.17779v1)

**作者:** Phillip Pitschi `[一作]` (Technische Universitaet Muenchen), Boris Lohmann `[通讯]` (Technische Universitaet Muenchen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套完整的稳定性控制系统（ESC、滑移控制与逆转角控制），用于在自主赛车接近动态极限时自动调整转向、刹车和发动机指令，保障车辆在极限工况下的安全稳定。

**💡 创新点**

创新点包括将ESC、SC 与 CS 三个模块化子系统集成为单一安全加固层；采用线性单轨模型快速生成转向与刹车参考；在关键状态下仅激活，避免对运动控制器的干扰；并将该系统实现为可开源的 C++ 代码，便于快速集成。

**🔧 技术方法**

技术手段包括：PID 与有限状态机混合控制、滑移比估计、线性单轨与 Pacejka 双轨动力学模型、MPC（管道与非线性单轨）以及纯追踪等运动控制算法，所有模块均通过 ROS 实时执行。

**📊 数据集**

数据集与实验材料主要是：在 Yas Marina 赛道上对 Dallara EAV25 自主赛车进行的实车测试（约 40 小时轨道时间），以及基于公开双轨 Pacejka 模型的仿真环境；未使用外部公开数据集。

**📈 对比分析**

通过与无稳定控制、仅 ESC、仅 CS、ESC+CS 等组合以及不同运动控制器（pmpc、纯追踪、nmpc）的仿真与实车比较，评估最大侧滑角、车速、轨迹误差及 lap time。结果显示，ESC+CS 组合能够在更高的加速度比例下保持稳定，显著缩短 lap time 并扩大可行域，且在实车测试中未出现失控或轮胎损坏。

**⚠️ 局限性**

局限性：系统性能受基础运动控制器的影响，若运动控制器本身性能不足，提升有限；ESC 仅使用前轴制动，低速情况下的稳定性不足；CS 采用线性轮胎假设，非线性工况下仍需改进；未实现扭矩向量等进一步的动力学补偿；验证仅在一款赛车与单一赛道上完成，缺乏多车型、多环境的泛化验证。

---

## 404. Parameterized complexity of $k$-Coloring in graphs with no long induced paths

**arXiv ID:** 2608.17835 | [PDF](https://arxiv.org/pdf/2608.17835v1)

**作者:** Paweł Rzążewski `[一作]` `[通讯]` (Warsaw University of Technology), Paweł Rzążewski (Warsaw University of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在H-free图中k-着色的参数化复杂性，特别是当H是线性森林时的情况。证明了在2P_2-free图中，k-着色在参数k下是NP-hard的，并且在P_t-free图中，3-着色在参数t下也是NP-hard的。

**💡 创新点**

首次证明了在2P_2-free图中k-着色的NP-hard性，并且在P_t-free图中3-着色的NP-hard性，解决了之前的开放问题。

**🔧 技术方法**

使用了参数化复杂性理论和图论中的归约技术，特别是通过构造选择器小工具来证明NP-hard性。

**📊 数据集**

使用了H-free图的构造，特别是2P_2-free和P_t-free图的实例。

**📈 对比分析**

与已知的多项式时间算法进行比较，证明了在特定条件下无法在多项式时间内解决这些问题，性能上显示出在参数化复杂性下的困难性。

**⚠️ 局限性**

研究中未能解决的限制包括对于H = P_3 + sP_1（s ≥ 2）和H = P_4 + sP_1（s ≥ 1）的情况，仍然存在未解决的复杂性问题。

---

## 405. Bounded-State Restoration: Decoupling Local Restore Capacity from External LLM State

**arXiv ID:** 2608.17826 | [PDF](https://arxiv.org/pdf/2608.17826v1)

**作者:** Zixuan Li `[一作]` `[通讯]` (China Academy of Railway Sciences Corporation Limited), Zixuan Li (China Academy of Railway Sciences Corporation Limited)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现一种名为 BSR 的协议，能够在保持对外部 KV 缓存状态的完整识别的同时，使用固定大小的本地工作集窗口来完成 LLM 执行状态的恢复，从而显著降低恢复阶段的峰值内存需求。

**💡 创新点**

创新点在于将完整可恢复前缀的发现与局部驻留分离，定义并测量了恢复工作集（RWS）这一新的资源维度；通过窗口化安装实现了 O(W) 的峰值恢复容量，同时保证了请求级别的语义完整性和失败闭合恢复。

**🔧 技术方法**

技术包括：probe‑only 查询、窗口化恢复安装、HMA 失效时的全前缀失效、请求级别的提交规则、基于 LMCache 与 vLLM 的原生扩展，以及 SSD 并发读取优化。

**📊 数据集**

使用 DeepSeek‑V4‑Flash 作为模型，结合 DGX Spark 机器的 TP=2 配置，对不同 token 数量（32K–524K）和不同窗口大小（8/16/32/64）进行实验；未使用公开数据集，而是在实际推理任务中进行性能验证。

**📈 对比分析**

通过与传统全计划恢复路径的对比，BSR 在外部状态规模提升 15.99 倍时，峰值 RWS 仅保持 500.75 MiB/Rank；在固定窗口 W=32 的清洁实验中，外部状态从 1.956 GiB/Rank 线性增长到 31.277 GiB/Rank，峰值 RWS 不变；此外，通过 SSD 并发读取从 1 → 4 的提升，将 512K 的恢复总时间从 43.1 s 缩短到 17.6 s，验证了恢复吞吐可独立于 RWS 进行优化。

**⚠️ 局限性**

局限性包括：RWS 仅覆盖 LMCache L1 恢复池，未覆盖完整进程内存；窗口大小对不同模型/状态的系数不一定相同；当前实现对恢复窗口采用全局串行锁，限制了并发请求的吞吐；实验仅针对 DeepSeek‑V4‑Flash，未对其他模型或硬件进行验证；并未给出形式化的正确性证明。

---

## 406. Training with synthetic data for drone detection in thermal imagery

**arXiv ID:** 2608.17799 | [PDF](https://arxiv.org/pdf/2608.17799v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 407. Reshaping the SDLC for Data- and AI-Centric Systems

**arXiv ID:** 2608.17824 | [PDF](https://arxiv.org/pdf/2608.17824v1)

**作者:** Mamdouh Alenezi `[一作]` `[通讯]` (Saudi Data and Artificial Intelligence Authority), Mamdouh Alenezi (Saudi Data and Artificial Intelligence Authority)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过理论整合与形式化，对数据驱动与 AI 系统的 SDLC 进行重构，提出了多环节连续、可适应的生命周期框架。

**💡 创新点**

创新点在于将数据、模型与代码三者视为同等版本化的构成体，构建了五层闭环控制架构，并给出可检验的设计命题。

**🔧 技术方法**

采用数据工程、MLOps、LLMOps 相关实践，结合形式化定义、统计门控与控制理论实现闭环维护。

**📊 数据集**

该工作为概念性综述，未使用具体实验数据集。

**📈 对比分析**

未进行实验比较；论文主要通过文献综述与理论推导阐述框架的合理性与适用性。

**⚠️ 局限性**

主要局限为缺乏经验性验证、缺少可量化的效果评估以及对人因、治理和安全等人本维度的深入探讨。

---

## 408. On computational approaches to Pop music culture

**arXiv ID:** 2608.17812 | [PDF](https://arxiv.org/pdf/2608.17812v1)

**作者:** Arthur Flexer `[一作]` `[通讯]` (Johannes Kepler University Linz), Arthur Flexer (Johannes Kepler University Linz)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了在流行音乐文化研究中，如何将多模态方法（音频、图像、文本）与计算音乐信息检索技术结合，以实现大规模的“远程阅读/听取/观看”分析。

**💡 创新点**

提出将流行音乐视为多模态社会文化现象的框架，并指出当前研究的缺陷（缺乏多模态、外部有效性不足）以及未来三大研究目标（歌词主题图谱、专辑封面图标化、复古周期追踪）。

**🔧 技术方法**

使用了文本挖掘、NLP、LDA/上下文主题建模、音频特征提取、图像识别、零射击对象检测、跨模态嵌入以及大型语言模型等技术。

**📊 数据集**

主要参考了Million Song Dataset、Million Musical Tweets、Billboard Hot 100、Music Genome Project、Rolling Stone评论、专辑封面图像集等公开或行业合作的数据集。

**📈 对比分析**

文章未进行新实验，主要通过文献综述对比已有方法的效果；指出现有研究多在小样本或单模态下验证，且在大规模数据上往往忽视多模态协同的性能提升。

**⚠️ 局限性**

限制包括采样偏差导致的外部有效性问题、缺乏真正大规模多模态数据、算法可解释性与黑盒化问题，以及跨媒介一致性评估的技术难点。

---

## 409. Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It

**arXiv ID:** 2608.17809 | [PDF](https://arxiv.org/pdf/2608.17809v1)

**作者:** Quang Minh Nguyen `[一作]` (Korea Advanced Institute of Science and Technology), Luis Frentzen Salim `[通讯]` (National Taiwan University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM在确认用户陈述的信念时，发现其是否认可取决于信念的表述方式和底层主张的真伪，并通过指令和注意力抑制干预减轻误判。

**💡 创新点**

首次揭示信念确认弱点随表述词汇变化而显著不同，可通过单一指令或注意力抑制恢复正确性，展示任务混淆导致错误的机制。

**🔧 技术方法**

采用指令调优、链式思考评估、LLM判定和注意力抑制干预等技术对模型进行行为分析。

**📊 数据集**

使用KaBLE Task 5（1000条真假信念陈述）与Task 4（真假验证任务）数据集，涵盖18种不同的表述词。

**📈 对比分析**

在10种开源LLM上进行对比，指令控制下假信念的准确率提升显著，注意力抑制在部分模型中提升约20%，整体验证了任务混淆与干预效果。

**⚠️ 局限性**

仅在可本地运行的模型上测试，干预效果对不同模型差异大；未评估前沿模型、多轮对话情境及其他信念确认数据集，且算力限制阻碍了更广泛的干预探索。

---

## 410. Debate Training Reduces Reward Hacking in RLAIF

**arXiv ID:** 2608.17776 | [PDF](https://arxiv.org/pdf/2608.17776v1)

**作者:** Zachary Kenton `[一作]`, Rohin Shah `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在缺乏真值标签的数学推理任务中，通过多智能体自我对弈的辩论训练，降低了奖励黑客行为并保持了判定者的性能，从而提升了生成器的峰值准确率。

**💡 创新点**

首次证明在全参数多智能体强化学习中，辩论能抑制奖励黑客并维持高峰准确率；同时揭示了词数限制与裁判弱化的平衡。

**🔧 技术方法**

使用RLAIF与多智能体自我对弈（Debate‑AB/ABA），冻结LLM判定者，采用隐式链式推理与可见响应，结合词数限制与格式化惩罚。

**📊 数据集**

基于一份类似AIME的数学推理数据集（约数千道题），训练集与验证集按50% IID划分。

**📈 对比分析**

与单玩家基线RLAIF‑A、RLVR上限线比较，辩论在验证集上峰值准确率提升约2个百分点，保持稳定，且奖励‑验证差距显著缩小；实验还对弱判定者、提示误导、不同词数上限等进行 ablation。

**⚠️ 局限性**

局限包括仅在可验证的数学任务上验证，难以推广至主观或无真值任务；缺乏对机制的深入解释；辩论仍易被词长攻击，需要进一步平衡；并且计算成本高。

---

## 411. D$^2$ACCI: A Dual-Loop Diagnostic Protocol for Evidence-Preserving Agent Memory

**arXiv ID:** 2608.17756 | [PDF](https://arxiv.org/pdf/2608.17756v1)

**作者:** Xule Liu `[一作]` (Xiaomi Inc.), Shao Kun `[通讯]` (Xiaomi Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种双循环诊断驱动的闭环控制迭代框架 D2ACCI，用于持续改进 LLM 代理的持久记忆系统。

**💡 创新点**

创新点在于将端到端评估拆分为内循环执行与外循环诊断，配合对齐的特征标记、受保护切片回归检测、以及可度量的诊断覆盖率指标 DCR，形成可复现的门控决策协议。

**🔧 技术方法**

技术上利用配对 A/B 测试、McNemar 统计、Bootstrap 置信区间、可观测的阶段级追踪、特征标记以及 -Eval 评估工件来实现统计可信的、可追踪的迭代。

**📊 数据集**

实验数据来自三大公开记忆基准：LoCoMo、LongMemEval‑S（LME）和 PersonaMem（PMem），覆盖多跳推理、跨会话更新与个性化偏好等场景。

**📈 对比分析**

与 MemBrain 等现有基准对比，D2ACCI 在 LoCoMo 上取得 93.59%、在 LME 上 90.93%、在 PMem 上 57.20%，分别比参考点提升约 0.34pp、5.33pp 和 1.48pp，并在配对实验中实现了统计显著的增益。

**⚠️ 局限性**

主要限制在于仍未在在线真实环境中验证、对不同记忆架构的通用性尚待评估，并且系统对追踪和特征标记的依赖导致实现成本和运行时开销。

---

## 412. Achievement Unlocked: Let's Get Hacked! An Empirical Study of Cybercrime in the Video Gaming Ecosystem

**arXiv ID:** 2608.17754 | [PDF](https://arxiv.org/pdf/2608.17754v1)

**作者:** Janine Schneider `[一作]` (University of Augsburg), Bhupendra Acharya `[通讯]` (University of Louisiana at Lafayette)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对视频游戏生态中的网络犯罪进行了大规模的混合方法研究，包括在线调查、访谈和用户报告分析。

**💡 创新点**

首次提供了系统的玩家中心网络犯罪分析，并发布了标注好的数据集，可用于后续机器学习研究。

**🔧 技术方法**

采用LLM（GPT‑4o‑mini）进行帖子过滤，Mayring内容分析法进行编码，结合统计与定性分析。

**📊 数据集**

使用了57名国际玩家的问卷数据、2名受害者访谈记录，以及从Steam、GOG和Reddit共收集的2,574条用户报告。

**📈 对比分析**

论文未对模型进行性能评估，而是将数据集作为基准；作者建议使用BERT等模型在此数据上进行自动化分类。

**⚠️ 局限性**

受样本与报告偏倚影响，主要聚焦PC平台，缺乏移动和主机平台的数据，且报告真实性和多账号问题未充分解决。

---

## 413. DistillPath: An Efficient 22M Distilled Pathology Encoder Approaching Large Foundation Model Performance

**arXiv ID:** 2608.17872 | [PDF](https://arxiv.org/pdf/2608.17872v1)

**作者:** Ramon Kaspar `[一作]` (ETH Zürich), Valentina Boeva `[通讯]` (ETH Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过仅使用教师模型的冻结class和patch token进行知识蒸馏，将大型病理图像编码器压缩到22M kaiko ViT‑S/16学生。

**💡 创新点**

创新点在于：① 只利用教师的最终class与patch token，无需教师的预训练head或百万级图像；② 结合cosine、RKD关系损失和patch‑token对齐的蒸馏方案；③ 对多尺寸（86M–1.1B）教师进行系统比较；④ 提供完整开源实现和公开训练代码。

**🔧 技术方法**

技术手段包括：ViT‑S/16架构，cosine对齐，关系损失RKD，patch‑token对齐，bfloat16训练，AdamW优化器，学习率余弦衰减，在线tile采样与数据增强，及GPU/CPU推理效率评测。

**📊 数据集**

使用了6,000份TCGA H&E全切片进行蒸馏训练，并在BACH、CRC、PCam、MHIST、BreakHis、Gleason、CoNSeP、MoNuSAC等EVA基准上评估；还使用了HEST基准（基因表达预测）和PLISM鲁棒性基准。

**📈 对比分析**

通过与教师模型（Virchow2、UNI2‑h、H‑optimus‑0、H0‑mini）以及kaiko基线、Midnight‑12k、GPFM等进行EVA、HEST、PLISM平均分对比。DistillPath‑KS16‑Virchow2在EVA平均得分0.795，接近大模型Virchow2的0.810，仅占其29×参数；相比大模型在RTX 4090上推理速度提升4.4×–44.2×，在MacBook M4 Pro上提升30.9×，存储需求降低3.3×。

**⚠️ 局限性**

局限性包括：① 仅在单一学生架构（22M ViT‑S/16）上验证，未探究不同容量或随机初始化的影响；② 只使用TCGA数据，缺乏更广泛的多中心或多分辨率数据；③ 未进行完整的损失权重、教师‑学生维度匹配等超参数消融；④ 对教师推荐的特征提取方式（如class+patch均值拼接）兼容性不足；⑤ 未尝试多教师蒸馏或结合自监督目标。

---

## 414. BayesPrompt: human readable prompts that make sense

**arXiv ID:** 2608.17866 | [PDF](https://arxiv.org/pdf/2608.17866v1)

**作者:** Franky Kevin Nando Tezoh `[一作]` (Scuola Internazionale Superiore di Studi Avanzati), Riccardo Rende `[通讯]` (Flatiron Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将提示优化问题转化为贝叶斯后验推断，利用MCMC采样得到既低困惑度又可读性高的提示。

**💡 创新点**

在提示优化中引入先验分布以消除伪提示，利用反向语言模型进行warm‑start，并采用MCMC实现可采样的提示分布。

**🔧 技术方法**

采用逆向语言模型、LoRA微调的 Llama‑3.2‑1B‑Instruct、Metropolis–Hastings MCMC、梯度下降/坐标梯度搜索等技术。

**📊 数据集**

使用 NQ‑OPEN 开放域问答对数据集进行评估。

**📈 对比分析**

与 GCG、GCG‑Reg、GD‑PEZ 等基线方法对比，MCMC 在答案置信度和可读性（流畅度）上均接近真实问题分布，显著优于基线。

**⚠️ 局限性**

仅在 1B 规模模型上验证，未对更大模型测试；采样过程计算开销较大；仅考虑文本序列，未探索潜在表示空间的提示逆向。

---

## 415. ESR-HGNN: Eliminating Semantic Redundancy for Efficient Mini-batch HGNN Inference

**arXiv ID:** 2608.17865 | [PDF](https://arxiv.org/pdf/2608.17865v1)

**作者:** Dengke Han `[一作]` (State Key Lab of Processors, Institute of Computing Technology, Chinese Academy of Sciences), Dongrui Fan `[通讯]` (State Key Lab of Processors, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对异构图神经网络的 mini‑batch 采样阶段，提出冗余感知采样方法并实现多通道硬件单元 ESR‑HGNN，显著消除语义冗余并提升采样效率。

**💡 创新点**

创新点包括：①使用前缀 Trie 记录并重用已采样的 metapath 路径；②基于可重用度的语义分组策略；③细粒度流水线与多通道硬件协同设计，三者共同实现采样阶段显著加速。

**🔧 技术方法**

技术手段涵盖：metapath Trie、语义路径缓存、邻接列表缓存、4‑bit XOR 树匹配、LFSR 随机采样、硬件多通道并行采样单元及软件协同流水线。

**📊 数据集**

实验数据集包括 ACM、IMDB 与大规模 MAG 三大异构图数据集。

**📈 对比分析**

与 CPU+GPU 基线相比，ESR‑HGNN 在采样阶段平均提升 39.66×，对端到端 mini‑batch 推理提升 5.70×；与 GPU 单纯采样相比提升 13.94×，同时 DRAM 访问减少 92.35%/96.75%，能耗下降 98.45%/86.41%。

**⚠️ 局限性**

局限性在于：①缓存容量限制导致 fan‑out 增大时重用率下降，影响性能；②一次性语义分组前处理在大规模 metapath 集合下仍需一定时间；③对 GPU 随机访问的兼容性有限，难以充分利用 GPU 并行能力。

---

## 416. ARASH: Adaptive Retrieval And Shot Selection for Tabular Prediction

**arXiv ID:** 2608.17856 | [PDF](https://arxiv.org/pdf/2608.17856v1)

**作者:** Samirasadat Jamalidinan `[一作]` (McMaster University), Kazem Cheshmi `[通讯]` (McMaster University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种适用于表格预测的自适应检索与样本挑选方法ARASH，通过在训练集中挖掘局部性与标签纯度来构造查询特定的少量演示；

**💡 创新点**

核心创新在于将局部性（Hopkins统计）与纯度（标签熵与纯度）两种诊断结合，动态确定每个簇的shot数；并在检索时根据局部性/纯度四种模式（纯净局部、局部多样、混合、全局）做出最优选择；

**🔧 技术方法**

利用数据概况提取（归一化、PCA、Hopkins）、自动聚类选择（如KMeans、Spectral等）、熵+纯度难度评分、DPP多样性、TabPFN/TabDPT模型、以及序列化表格行到LM的文本格式；

**📊 数据集**

在OpenML-CC18和Combo基准集上进行评估，采用80/20随机划分；

**📈 对比分析**

与全上下文、kNN固定k、DPP、随机、以及大模型全上下文的基线进行对比。ARASH在准确率上与全上下文相当，平均提升≈0.02%（TabDPT）/0.01%（TabPFN），同时提示长度减少≈43×、VRAM减少≈2.56×，推理延迟降低1.37×；

**⚠️ 局限性**

局部性弱或标签纯度低的数据仍需回退到全局检索，导致在极端不纯区性能下降；方法需要一次性预处理和聚类参数调优，适用性在极大规模数据集和高维稀疏表格上仍需进一步验证。

---

## 417. GenRec: Knowing Where to Reconstruct and Where to Generate

**arXiv ID:** 2608.17832 | [PDF](https://arxiv.org/pdf/2608.17832v1)

**作者:** Ata Çelen `[一作]` (ETH Zürich), Daniel Barath `[通讯]` (ETH Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为GenRec的多视角流匹配模型，用观察掩码将新视图合成拆分为重建与生成两部分；

**💡 创新点**

创新点在于将观察掩码直接嵌入网络结构、损失与梯度流，解耦已观测区域的回归监督与未观测区域的分布匹配；

**🔧 技术方法**

采用多视角流匹配骨干联合去噪RGB与场景坐标潜流，结合像素空间的解码器适配（skip+LoRA）与稀疏3D跨注意力细化；

**📊 数据集**

在RealEstate10K、DL3DV-10K和Mip-NeRF 360三个数据集上进行评测；

**📈 对比分析**

与多种基线（CameraCtrl、ViewCrafter、SEVA、GLD、Gen3R、Gen3C等）对比，GenRec在已观测区域实现最高重建精度，在未观测区域达到或超过最强生成基线的感知质量，并且推理速度比warp-as-target视频基线快约十倍；

**⚠️ 局限性**

局限性包括依赖单目深度估计的观察掩码与warp条件，若深度误差较大会影响重建与细化效果，并且双模态架构在训练时受视角数量限制。

---

## 418. Integer Quadratic Programming is W[1]-Hard Parameterized by the Number of Variables

**arXiv ID:** 2608.17818 | [PDF](https://arxiv.org/pdf/2608.17818v1)

**作者:** Anton Herrmann `[一作]` `[通讯]` (Technical University of Berlin), Anton Herrmann (Technical University of Berlin)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明整数二次规划（IQP）在仅以变量数为参数的情况下是 W[1]-难的。

**💡 创新点**

通过从独立集（Independent Set）问题构造一个参数化归约，给出了一个简洁自包含的证明，展示了 IQP 的困难性。

**🔧 技术方法**

利用参数化复杂度理论中的归约技术，构造了分离的凹函数目标和一系列线性约束，并使用几何直觉证明约束的正确性。

**📊 数据集**

无使用任何实验数据集。

**📈 对比分析**

论文没有进行实验或性能比较；结论仅为理论复杂度的负结果。

**⚠️ 局限性**

仅针对整数变量的情况给出 W[1]-难性，未讨论在添加约束数、系数大小或其他参数时的可行性；结果不说明 IQP 在其它参数组合下是否可 FPT。

---

## 419. Quo Vadis? Scientific Discovery in the Age of Artificial Intelligence

**arXiv ID:** 2608.17970 | [PDF](https://arxiv.org/pdf/2608.17970v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 420. The Curious Case of Exploding DecPOMDPs: Containing the Fire through Policy Counting

**arXiv ID:** 2608.17749 | [PDF](https://arxiv.org/pdf/2608.17749v1)

**作者:** Nazlı Nur Karabulut `[一作]` (University of Münster), tanya Braun `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了基于策略计数的DecPOMDP模型，并给出了相应的动态规划求解方法。

**💡 创新点**

创新点在于将策略空间转化为可计数的“代表策略”，从而将模型与求解复杂度由指数级降为多项式级。

**🔧 技术方法**

使用了分区对称性、计数随机变量（CRV）、动态规划与线性规划剪枝等技术。

**📊 数据集**

主要使用经典的 DecTiger 案例作为演示和实验基准。

**📈 对比分析**

与传统完整策略树的动态规划相比，新方法在保持相同最优期望奖励的前提下，将计算时间从指数级下降为多项式级，在 N 较大时表现显著提升。

**⚠️ 局限性**

限制在于对分区数 K 的假设（K≪N）以及策略计数仍可能引入额外开销，且在极端大范围状态空间时仍需进一步优化。

---

## 421. VisDocAgentBench: Benchmarking Agents for Visually Rich Document Retrieval

**arXiv ID:** 2608.17889 | [PDF](https://arxiv.org/pdf/2608.17889v1)

**作者:** Lexiang Hu `[一作]` (Peking University), Zhouchen Lin `[通讯]` (Peking University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个闭合语料库的可视文档检索基准，并在该基准上评估了静态检索器和基于代理的检索系统。

**💡 创新点**

提出了“闭合语料库代理式可视文档检索”任务定义，设计了关系保持路径构造与质量控制流程，并通过完整文档审核保证查询与目标的可靠性。

**🔧 技术方法**

采用多模态 VLM 提取角色描述与关系、Qwen3 嵌入进行语义对齐、Nemotron‑ColEmbed 进行晚期交互检索、GPT‑5 系列和 Claude 系列作为代理，以及 OCR+文本检索、图像检索、页面检索等工具。

**📊 数据集**

使用 2026 年 arXiv 100 篇科学论文（共 2,375 页）构建查询集，生成 120 条包含 1‑3 步证据链的目标检索问题。

**📈 对比分析**

通过 Recall@k 与 MRR@10 对比静态检索器与代理检索器；最强静态检索器在直链上达 97.5% R@1，但两桥链仅 2.5%；代理检索器在整体 R@1 达 67.5%（Claude Opus 5），在 OCR‑Text 路径上提升 37.5%；迭代搜索、页面检视对性能提升显著。

**⚠️ 局限性**

基准仅包含 120 条英文科学论文，缺乏多语种、不同文档类型与更细粒度检索目标，且仅评估单页级别检索，难以推广到更大规模、多样化语料。

---

## 422. CFB-GBM v2.0: An Augmented Longitudinal Dataset for Multi-Modal Glioblastoma Segmentation, Radiomics, and RANO Progression Tracking

**arXiv ID:** 2608.17884 | [PDF](https://arxiv.org/pdf/2608.17884v1)

**作者:** Alexandre G. Leclercq `[一作]`, Aurélien Corroyer-Dulmont `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

回顾性分析了185名患者的临床基本信息与随访数据，重点评估WHO 2016与2021指南在患者功能状态分类中的差异。

**💡 创新点**

首次将两版WHO指南对照使用，探讨其对临床功能评分的影响，为后续临床决策提供参考。

**🔧 技术方法**

采用统计学方法（描述性统计、卡方检验等）对不同年龄、性别、体重等变量与功能状态进行关联分析。

**📊 数据集**

使用本院收集的185例患者临床记录，包括性别、年龄、身高体重、诊断时间、随访时间以及WHO功能状态评分等信息。

**📈 对比分析**

通过对比两指南下的功能状态分布，发现2021版指南在高功能状态（PS 0-1）分类上更为严格，整体随访时间略有提升，显示指南更新可能影响患者管理。

**⚠️ 局限性**

局限性包括样本量相对较小、为单中心回顾性研究，缺乏多中心验证，且未涉及具体治疗干预效果。

---

## 423. Cross-Domain Generalization in Machine Unlearning via Label-Conditioned Energy Magnitude Regularization

**arXiv ID:** 2608.17942 | [PDF](https://arxiv.org/pdf/2608.17942v1)

**作者:** Syed Ali Ahmed `[一作]` (National University of Computer and Emerging Sciences), Muhammad Zaigham Zaheer `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究机器遗忘在跨域与跨类场景中的传播，并提出基于标签条件能量模型的可控传播框架

**💡 创新点**

创新点在于结合标签条件能量模型与能量正则化，利用DINOv2相似度+PCA子空间权重实现对遗忘信号传播范围的精准调控

**🔧 技术方法**

使用标签条件能量模型（ResNet‑18+线性能量头）、DINOv2特征相似度、PCA子空间、LoRA适配器等技术

**📊 数据集**

实验数据集包括DomainNet 26类子集（real、sketch、clipart、painting）和CIFAR‑10

**📈 对比分析**

与Retrain、NegGrad、Finetune等基线比较，DomainNet上可实现几乎完整的跨域遗忘并保持高保留准确率；CIFAR‑10上实现100%遗忘且模型保持约70%准确率，MIA接近0.5，说明遗忘有效且不可追踪

**⚠️ 局限性**

局限性：实验仅在小模型与少量类别上验证，传播主要集中在最近邻，控制精度受限；未评估大规模模型、多域或多任务场景下的可扩展性

---

## 424. Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation

**arXiv ID:** 2608.17941 | [PDF](https://arxiv.org/pdf/2608.17941v1)

**作者:** Zhizhao Liu `[一作]` (National University of Defense Technology), Dongsheng Li `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于图的在线难度估计框架，用于强化学习可验证奖励（RLVR）中的样本选择与rollout分配；

**💡 创新点**

创新点在于：1）构建难度感知样本图并通过Potts先验将邻近样本聚集到共享的隐状态；2）使用Beta–Binomial模型对共享状态的成功概率进行贝叶斯更新；3）采用在线均值场变分推断持续更新隐状态分布和状态难度，避免昂贵的额外探测；4）实现“即插即用”，可无缝集成至多种RLVR调度器；

**🔧 技术方法**

技术包括：文本嵌入+难度感知指令、k‑近邻图构造与谱聚类初始化、Potts图先验、Beta‑Binomial 随机过程、在线均值场变分推断（坐标升梯法）及Beta参数更新；

**📊 数据集**

使用的主要数据集有：NuminaMath（150K题目用于训练）、MATH500、AIME 2024/2025、OlympiadBench；此外在代码生成任务上使用LiveCodeBench；

**📈 对比分析**

将该估计器嵌入三类调度器（GVM、PCL、GRESO），在匹配的rollout预算下与原版比较，Average@8提升幅度从数个百分点到十几个百分点；统计显著性检验（符号检验）显示改进方向一致，整体显著性p≈1e‑5；

**⚠️ 局限性**

局限性：1）对图构造与嵌入质量敏感；2）历史信息虽共享但仍可能产生冷启动/过期误差；3）需要手动调参（β、K、k‑nn、γ）和额外的谱聚类初始化；4）在非数学领域（如代码生成）改进不如数学任务明显。

---

## 425. Beyond Instrument Motion: Recognizing Tissue Tension Toward Surgical Skill Assessment

**arXiv ID:** 2608.17935 | [PDF](https://arxiv.org/pdf/2608.17935v1)

**作者:** Marko Haralovi `[一作]`, Estefania Talavera `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了利用稀疏轨迹对腹腔镜/机器人辅助直肠癌切除手术视频中的组织张力进行识别的新任务；

**💡 创新点**

创新点在于将张力定义为可视化的局部组织变形，将其转化为视频级别的事件识别，并引入稀疏轨迹表示与语义特征融合的轻量化框架TensionTRAC；

**🔧 技术方法**

方法采用冻结的DINOv2视觉编码器提取语义特征，CoTrackerV3追踪稀疏点轨迹，计算轨迹的内部与相互运动描述，最后通过一层时空Transformer生成表示；

**📊 数据集**

数据集为SurgTension，收集7条真实手术录像共约11小时，标注了3,593个有效片段，划分为0-4级张力，涵盖无张力、低、中、适中、过度张力；

**📈 对比分析**

在二分类和五级级联任务上，TensionTRAC在视频组交叉验证下宏F1约为75%（与慢速Fast、Swin-3D等强基线相近），在二分类任务上表现与视频基线持平或略优；

**⚠️ 局限性**

局限性包括样本量有限、标注单人主观性高、对遮挡、烟雾等视觉噪声敏感、以及级别细分时相邻级别混淆，影响级联识别精度。

---

## 426. Collective Counterfactual Planning: Coordination, Consent, and Verification under Representational Constraints

**arXiv ID:** 2608.17932 | [PDF](https://arxiv.org/pdf/2608.17932v1)

**作者:** Chainarong Amornbunchornvej `[一作]` `[通讯]` (National Electronics and Computer Technology Center), Chainarong Amornbunchornvej (National Electronics and Computer Technology Center)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种以表示几何为核心约束的团队协作规划模型——Collective Counterfactual Planning（CCP），并定义其可解性问题（CCS）及三层语义（几何可行、可执行实现、验证完成）。

**💡 创新点**

创新点在于：① 把团队成员的局限性从能力/知识/可观测性转化为子空间投影；② 通过四个门（实现、构思、同意、资格）揭示跨代理可行性与不可验证性的对立；③ 引入“交叉代理中继”机制解释团队为何能完成单个代理无法规划的任务；④ 设计四步完整可检验的可解性求解方案并证明其正确性；⑤ 讨论记忆无效同意与审核同意的不可比性。

**🔧 技术方法**

技术手段包括：线性代数和投影几何建模、形式化推理与定理证明、基于前缀扩展的可达性树（relay closure）构造、可选的全局或分段同意策略、终点资格检验、实验验证（Python实现）。

**📊 数据集**

主要使用人工构造的实验实例（E1、E2）以及随机生成的结构化实例（varying 覆盖率、同意阈值、共享表示核心）进行演示；未使用真实世界数据集。

**📈 对比分析**

对比方法：在本研究中未与现有多智能体规划/分布式认知系统进行基准比较，而是通过理论分析展示正负结果；实验演示显示记忆无效同意在阈值宽松时优于审核同意，阈值严格时相反；完整可解性方案在给定视界下保证可行性，受限搜索仍保持安全但可能缺失解。

**⚠️ 局限性**

局限性包括：① 仅考虑无策略、无学习、无欺骗的理性真诚团队；② 模型假设投影子空间固定且不随经验更新；③ 依赖私有阈值导致交互式规划的查询复杂度未完全解决；④ 复杂度上对状态依赖的中继闭包的精确表示与求解仍是开放问题；⑤ 真实团队的失误、信息不完全与组织层次结构未纳入。

---

## 427. PerFact: Perception-Derived Fact Prompting for 3D Brain MRI Report Generation

**arXiv ID:** 2608.17926 | [PDF](https://arxiv.org/pdf/2608.17926v1)

**作者:** Jianyu Sun `[一作]` (Imperial College London), Peter J. Lally `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 PerFact 系统，将 3D 大脑 MRI 的分割与分类结果序列化为结构化事实句子，再将其作为提示注入到视觉‑语言模型中生成报告，证明注入事实比单纯使用模型或检索更能提升报告质量。

**💡 创新点**

创新点在于：1）将 3D 感知任务与 2D 生成任务分离，形成可插拔的“感知‑事实‑生成”管线；2）系统化评估注入信息对报告质量的影响，发现信息注入是主要杠杆；3）通过实验验证零拷贝预训练模型在脑 MRI 上迁移差，且不同规模的骨干网络对结果影响甚微。

**🔧 技术方法**

技术包括多任务 3D 语义分割、属性分类、缺失模态补全和图像‑文本对齐的上游感知网络；LoRA 适配的视觉‑语言生成器；结构化事实句子序列化与检索（检索-案例/知识库）；多轮评估（CE F1、BLEU/METEOR/ROUGE、VQA AUROC）。

**📊 数据集**

使用 RadGenome‑Brain MRI 公开语料库，包含 4 个疾病（胶质瘤、脑膜瘤、急性脑卒中、白质高信号）与 3 种成像序列（T1/T1c/T2/FLAIR 等），共 130 例测试报告和 1549 题闭合式 VQA。

**📈 对比分析**

方法比较：在相同骨干、数据划分、生成目标和解码策略下，仅变化注入信息。结果显示，结构化事实（PerFact）可将 CE F1 提升至 0.775，远超仅用图像（0.644）和检索（≤0.718）；模型规模从 3B 到 32B 的差异仅 ±0.015；零拷贝医学预训练模型在脑 MRI 上性能远低于微调系统。

**⚠️ 局限性**

局限性：1）所有报告来自同一注释源，报告约定可能不具普适性；2）生成器仅处理 2D 切片，缺乏原生 3D 上下文；3）上游感知对细粒度事实的预测受限，导致“oracle gap”未完全填补。

---

## 428. Overlap-free multi-material topology optimization for minimum compliance in two and three dimensions by level-set-based negative-mapping interpolation

**arXiv ID:** 2608.17963 | [PDF](https://arxiv.org/pdf/2608.17963v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 429. Abstract Simulation of Reaction Networks

**arXiv ID:** 2608.17893 | [PDF](https://arxiv.org/pdf/2608.17893v1)

**作者:** Marie-Eva Fabri `[一作]` (University of Lille, CNRS, Centrale Lille, CRIStAL), Cristian Versari `[通讯]` (University of Lille, CNRS, Centrale Lille, CRIStAL)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种新的定性抽象方法，用于在缺乏精确动力学参数的条件下模拟化学反应网络的连续时间动力学。该方法通过引入因果连续语义和差分符号结构，既捕捉了反应网络的连续演化，又显式区分了激活和抑制作用。

**💡 创新点**

创新点主要有：
• 将差分符号（即变化的符号）与变量本身的符号统一到一个抽象结构中，解决了传统符号抽象中导数符号与实际变化符号不匹配的问题；
• 通过引入“延迟”机制，使得抽象能够保持因果性，避免了连续语义中随时间趋近零时产生的非因果匹配；
• 利用抽象解释构造了差分符号后继关系（differential sign successor），证明其是因果连续语义的可证实抽象，且在抽象图中显著减少了无意义的边，提升了精确度。

**🔧 技术方法**

使用技术包括：
• 抽象解释（abstract interpretation）框架；
• 关系结构和符号系统（relational structures）以及标准算术签名的扩展；
• 差分符号结构（包含符号和值符号的笛卡尔积）；
• 通过对ODE解的抽象化（包括求导和差分的符号化）来构造后继关系；
• 证明与连续语义的一致性及可证实性。

**📊 数据集**

实验使用的主要案例是 Lotka‑Volterra 捕食者‑猎物模型（无真实实验数据），并通过对该模型的解析解与抽象图进行对比来验证方法的有效性。

**📈 对比分析**

比较方法：
• 与原先的符号抽象（sign successor）和最宽松语义（Most Permissive semantics）进行对比；
• 在抽象图上统计边数：传统符号抽象得到 92 条边，差分符号后继得到 29 条边，因果连续抽象得到 17 条边，表明差分符号后继在保持精确性的同时显著降低了图的复杂度。性能方面，因果连续抽象在计算上与传统符号抽象相当，但在精确度上更优。

**⚠️ 局限性**

局限性：
• 该方法仍然是理论性构造，缺乏对大规模真实生物网络的实验验证；
• 对于极度不确定的“partial reaction networks”（即反应速率表达式未知）尚未完整支持；
• 抽象的延迟机制在实践中需要精细调参，以平衡精度与计算复杂度；
• 仅对连续可导的ODE解适用，可能无法处理非光滑或离散事件导致的跳跃动态。

---

## 430. Jetson-ORB-SLAM3: Accuracy-Preserving GPU Implementation for Edge Computing Devices

**arXiv ID:** 2608.17874 | [PDF](https://arxiv.org/pdf/2608.17874v1)

**作者:** Rajat Roy `[一作]` (IIT Jodhpur), Hardik Jain `[通讯]` (IIT Jodhpur)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在NVIDIA Jetson Orin Nano上实现了精度保持的GPU版本ORB‑SLAM3，并将CosPlace ResNet‑50通过原生TensorRT FP16引擎部署，实现低功耗边缘设备的实时视觉惯性SLAM。

**💡 创新点**

核心创新是完整ORB前端在GPU上的算法保持一致（尺度金字塔、FAST、NMS、方向估计、rBRIEF），以及利用TensorRT将CNN循环识别从≈400 ms压缩至2.2 ms，实现边缘可行的深度学习回环。

**🔧 技术方法**

技术包括CUDA实现的完整ORB提取管线、TensorRT FP16推理、ONNX‑Runtime兼容性调优、双线程（前端GPU、后端CPU）协同架构和实时多线程调度。

**📊 数据集**

使用公开的EuRoC、TUM‑VI、KITTI三大视觉惯性SLAM基准数据集进行评估。

**📈 对比分析**

通过对CPU参考实现和GPU实现在同一硬件与不同硬件下的轨迹误差（ATE/相对误差）对比，GPU实现与CPU差异小于0.1 cm（EuRoC）或0.036 %（KITTI），在EuRoC上平均实现32 FPS，KITTI上保持相机帧率，CNN推理从≈396 ms降至2.2 ms。

**⚠️ 局限性**

局限性包括GPU提取在低分辨率场景下未快于CPU，后端局部BA仍在CPU上，且在高动态序列（如EuRoC V203）中估计器初始化敏感性导致轨迹波动大。

---

## 431. SFMformer: A Spatial-Frequency Modulation Transformer for Lightweight Image Super-Resolution

**arXiv ID:** 2608.17966 | [PDF](https://arxiv.org/pdf/2608.17966v1)

**作者:** Chih-Hsiang Yang `[一作]` (Tamkang University), Jen-Shiun Chiang `[通讯]` (Tamkang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级图像超分辨率模型SFMformer，在保持参数低于百万的前提下通过在稀疏注意力的选择阶段和聚合阶段分别加入空间增强（DFE）和频域调制（WMA），显著提升PSNR/SSIM。

**💡 创新点**

创新点在于将稀疏注意力的选择与聚合两阶段视为可分离的改进目标，分别在前置和后置位置插入专门模块，并证明两者在多数基准上相互补充、提升性能。

**🔧 技术方法**

技术包括稀疏注意力（Progressive Focused Attention）、双分支空间特征提取（DFE）、小波域调制自注意力（WMA）、多尺度Transformer块、像素重排上采样以及L1+FFT损失。

**📊 数据集**

使用DF2K作为训练集，评估在五个公开基准（Set5、Set14、BSD100、Urban100、Manga109）上，尺度为×2、×3、×4。

**📈 对比分析**

与现有轻量级Transformer和CNN方法对比，SFMformer在28/30个PSNR/SSIM条目中排名第一（或第二），在Urban100和Manga109等对长距离自相似和高频纹理有优势的图集上提升0.1–0.3 dB；同时保持参数≤1M，适配Raspberry Pi 5等边缘设备。

**⚠️ 局限性**

局限性包括：仅基于合成bicubic降采样；缺乏感知质量评估；与GPU加速相比仅测CPU推理；单次训练对比导致结果波动；模型在高频纹理复杂区域仍偏平滑。

---

## 432. Towards Zero-Shot Task Transfer with Neurosymbolic World Models

**arXiv ID:** 2608.17959 | [PDF](https://arxiv.org/pdf/2608.17959v1)

**作者:** Isidoro Tamassia `[一作]` (KU Leuven), Giuseppe Marra `[通讯]` (KU Leuven)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Neurosymbolic World Models (NeSy‑WMs)，实现对共享动态环境中不同奖励函数的零射击适应。

**💡 创新点**

创新点在于将奖励与终止预测从完整潜在空间分离到符号属性层，构建可解释且可替换的奖励接口，支持无环境交互的规划与想象微调。

**🔧 技术方法**

采用基于DreamerV3的递归状态空间模型（RSSM），并加入神经符号奖励/终止预测器与可选符号监督损失，实现端到端可微分训练。

**📊 数据集**

在MiniGrid（四室地图）、MiniWorld（3D四室）以及Sokoban等稀疏奖励任务上进行实验，验证模型效果。

**📈 对比分析**

与DreamerV3在样本效率、零射击MCTS规划成功率和想象微调性能进行比较；NeSy‑WMs在大多数任务上实现更快收敛、更高规划成功率和更好的零射击微调效果。

**⚠️ 局限性**

局限性包括对符号对齐的依赖——符号预测若失准将导致奖励替换失败；此外模型仅在奖励函数可表述为已有符号属性时有效，超出符号词汇的奖励无法直接适用。

---

## 433. An Omitted Mode Is a Rare Rule: The Sampling-Verification Danger Law in Continuous Code World Models

**arXiv ID:** 2608.17956 | [PDF](https://arxiv.org/pdf/2608.17956v1)

**作者:** Javier Aguilar Martín `[一作]` `[通讯]` (AGILabs), Javier Aguilar Martín (AGILabs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文通过构造连续控制中的“门控验证”流程（采样门），评估LLM合成的可执行世界模型在面对稀有硬模式（mode）时的风险。研究定义了危险度为 play_cost × (1‑r)^N，并证明了该公式在连续情境下的正确性和可分解性。通过在三个手写混合动力学仪器（cart‑with‑wall、pendulum‑with‑stop、PatchField2D）上实验，验证LLM能否从有限样本中推断缺失的硬模式，并比较模型误差、门通过率、稀有事件概率与随机基线的表现。

**💡 创新点**

创新点在于：
1) 将离散游戏中的门控验证理论推广到连续控制，给出危险度的解析表达式并证明其精确性；
2) 引入“定位预算”与Lipschitz约束，阐明模型误差在高维空间中如何局部化；
3) 在连续领域中首次验证LLM在1D硬模式下的“识别‑推理”能力以及在2D区域模式下的失败机制；
4) 通过多种仪器与规划器（随机射击 MPC、CEM）进行系统对比，揭示门通过率与实际性能之间的关系。

**🔧 技术方法**

主要技术手段包括：
- LLM (Azure GPT‑5.x mini/large、Claude Sonnet) 用于从自然语言规格中生成可执行的世界模型代码；
- 随机射击 MPC 与 CEM 作为模型预测控制器；
- 采样门（gate）设计：对 N 个 i.i.d. rollouts 进行监督，确保模型在门样本上与真实转移相符；
- 统计推断与置信区间（Wilson、Clopper‑Pearson）用于评估稀有事件概率与门通过率；
- 理论证明：危险度分解、门通过率的精确表达式、定位预算与 Lipschitz 约束下的误差局部化。

**📊 数据集**

使用的“数据集”主要是由仿真器生成的随机 rollouts：
- 对每个仪器（cart、pendulum、PatchField2D）在不同 knob 位置下，生成 30k 或 50k 条随机轨迹，用于估计稀有事件 r；
- 对每个实验 cell，使用 20 或 40 条采样 rollouts 作为门样本（训练/验证）；
- 20–100 条 MPC 规划轨迹用于计算 play_cost；
- 20 个随机种子（或 20×20 的组合）用于统计置信区间。数据完全合成，无外部公开数据集。

**📈 对比分析**

比较方法：
- 将门通过率与 r、N 结合，计算危险度并与随机基线（uniform‑random policy）对比；
- 对 1D 仪器（cart、pendulum）使用 LLM 生成模型，并统计：
  * 完全缺失模式时的 blind‑exploit 率（≈1），
  * 发现模式后 LLM 是否修复（>90% 成功），
  * 计算 play_cost 与随机基线的差值；
- 对 2D 仪器 PatchField2D，评估：
  * LLM 在 2D 圆形模式下的修复率为 0；
  * 门通过率仍符合 (1‑r)^N；
  * 对比 CEM 与 MPC 的 query‑hit 率，说明规划器对危险的影响。总体性能显示：在 1D 场景下，LLM 修复率高，危险度显著下降；在 2D 场景下，LLM 无法修复，危险度维持高位。

**⚠️ 局限性**

局限性：
- 仅验证了两类硬模式（1D 位置/角度停止与 2D 圆形冻结）；更复杂或连续混合动力学的普适性仍未探测；
- 采样门依赖 i.i.d. rollouts，无法覆盖具有强时序依赖或非独立事件的情形；
- LLM 的成功与否高度依赖提示工程与模型规模，缺乏对不同提示策略的系统分析；
- 对 2D 区域模式的修复失败表明缺少足够的几何推断能力，提示需要更强的结构化学习或约束优化；
- 评估基线仅为 uniform‑random，未覆盖更具竞争力的随机或优化规划器；
- 由于实验规模有限，置信区间在极小概率事件上仍有较大不确定性。

---

## 434. Tail exponents of conditional guesswork via the method of types

**arXiv ID:** 2608.17949 | [PDF](https://arxiv.org/pdf/2608.17949v1)

**作者:** Adway Girish `[一作]` (EPFL), Emre Telatar `[通讯]` (EPFL)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在有元素级相关旁路信息情况下，猜测 i.i.d. 随机序列所需猜测次数（guesswork）的尾概率指数，给出了条件猜测的精确大偏差指数并通过类型计数法推导；

**💡 创新点**

创新点在于：① 对条件猜测的尾概率指数给出显式表达，采用条件倾斜分布（conditional tilted distribution）解决最优分布；② 通过类型计数提供了比传统 LDP 更直观、易算的证明；③ 将结果应用于密码猜测，给出实际所需密码长度的量化估计；

**🔧 技术方法**

主要技术是信息理论中的类型方法（method of types）、KL 散度与条件倾斜分布、以及凸优化（Lagrange 乘子）来求解熵约束下的散度最小化问题；

**📊 数据集**

本文为理论研究，无使用具体数据集；

**📈 对比分析**

未与实验方法对比，主要提供理论上限与下限的指数分析，说明在给定旁路信息量下，猜测次数的指数衰减；

**⚠️ 局限性**

局限性：结果仅给出渐进指数，未给出精确的有限样本界；同时只考虑 i.i.d. 源，非 i.i.d. 情况尚未解决；

---

## 435. Comparative Study of Out-of-the-Box Technology for Automatic Target Detection and Recognition

**arXiv ID:** 2608.17917 | [PDF](https://arxiv.org/pdf/2608.17917v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 436. Procedural Content Metageneration via Program Search and Continual Abstraction Discovery

**arXiv ID:** 2608.17947 | [PDF](https://arxiv.org/pdf/2608.17947v1)

**作者:** Matthew Siper `[一作]` (New York University), Julian Togelius `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用大型语言模型（LLM）驱动的进化搜索，在Sokoban、Zelda、Dangerous Dave 和 Lode Runner 四个网格游戏中自动生成可执行的 Python 级别生成器，并在搜索过程中实时提取、验证并重用高适应度程序中的可复用函数（持续抽象发现 CAD）。

**💡 创新点**

提出并实现了持续抽象发现（CAD）机制，使搜索过程能够不断扩展可搜索的程序词汇表；通过 2×2 实验（CAD/无 CAD 与专家 API/无 API）在四个不同领域验证其有效性，首次在 PCG 元生成中将可复用函数的学习与演化结合。

**🔧 技术方法**

技术包括：GLM 5.2 作为 LLM 进行变异、交叉、纠错与重构；进化算法（按适应度比例选择、交叉概率 0.25）；程序级别生成器表示与 SMOKE 测试；反射记忆与 LLM 反思；CAD 的抽象、验证与重构流程。

**📊 数据集**

使用 PCG Benchmark 提供的四个基于网格的游戏数据集（Sokoban 5×5、Zelda 7×11、Dangerous Dave 7×11、Lode Runner 11×16），并利用各游戏的质量、有效性与多样性评估函数。

**📈 对比分析**

通过 160 次实验（每个实验 10 次 50 代），比较 CAD 与无 CAD、专家 API 与无 API 四种条件，评估最终最佳适应度、适应度曲线、学习库增长、函数采用率等指标。结果显示 CAD 在所有领域显著提升平均最终最佳适应度（p=0.008），在 Lode Runner 与专家 API 组合下效果最大。

**⚠️ 局限性**

局限性包括：计算成本高（50 代约 $25，2.5 小时）；未对 CAD 的抽象、验证、重构等子流程进行单独消融；仅针对网格游戏，未评估跨游戏库迁移与设计师手工编辑；仅依赖 Benchmark 评估，缺乏对视觉/体验质量的考量；未证明程序长度缩短是否必然更优。

---

## 437. A Theoretical Framework for Parallel Lifelong MAPF Using Group Decentralized Planning

**arXiv ID:** 2608.17928 | [PDF](https://arxiv.org/pdf/2608.17928v1)

**作者:** Alex DeWeese `[一作]` (Carnegie Mellon University), Guannan Qu `[通讯]` (Carnegie Mellon University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于群组去中心化的滚动窗口冲突解决框架GD‑RHCR，利用LI‑MDP理论对L‑MAPF进行折扣建模，并给出近最优性证明，显著降低计算成本同时保持高吞吐量。

**💡 创新点**

① 将L‑MAPF形式化为折扣LI‑MDP并证明RHCR近最优；② 基于群组去中心化分组提出GD‑RHCR并给出与RHCR相当的近最优保证；③ 引入懒评估、软约束规划与可变求解器的并行策略，实现理论与实践的统一。

**🔧 技术方法**

LI‑MDP折扣理论、滚动窗口冲突解决（RHCR）、群组去中心化分组、软约束规划、并行求解、PBS/PiBT求解器、SIPP搜索。

**📊 数据集**

4张MAPD地图（warehouse‑10‑20‑10‑2‑1、warehouse‑10‑20‑10‑2‑2、sortation‑1、sortation‑2）和6张随机导航/房间地图（random‑64‑64‑20、room‑64‑64‑var1、room‑64‑64‑var2、random‑32‑32‑20、room‑32‑32‑var1、empty‑48‑48）。

**📈 对比分析**

与RHCR（PBS+PiBT回退）和纯PiBT进行对比；GD‑RHCR在大多数地图上吞吐量与RHCR相近，在高代理数时保持高吞吐；平均规划时间提升约24.9×，吞吐量提升57.7%以上，且计算成本显著低于RHCR。

**⚠️ 局限性**

① 分组与可见半径设置不当会导致并行效果减弱；② 软约束参数调节对性能敏感；③ 对非二连通或高拥堵地图依赖PiBT，吞吐可能下降；④ 理论保证基于理想折扣MDP，实际环境可能偏离；⑤ 并行实现受硬件与实现细节影响。

---

## 438. Analysis of Types of Inquiries in Student-AI Interaction: A case study of two CS2 tasks

**arXiv ID:** 2608.17919 | [PDF](https://arxiv.org/pdf/2608.17919v1)

**作者:** Matin Amoozadeh `[一作]` (University of Houston), Amin Alipour `[通讯]` (University of Houston)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文对两次 CS2 编程实验中学生与生成式 AI 的 830 条交互提示进行分类与序列分析，探究提问类型随任务进展和学生背景的演变。

**💡 创新点**

创新点在于将 Graesser 提问分类法迁移至编程教育中的 AI 交互，利用 few‑shot 学习实现自动分类，并结合状态机转移图揭示提问行为的动态变化。

**🔧 技术方法**

使用技术包括 Graesser 提问分类法、few‑shot 语言模型分类（GPT‑5.2/Claude）、统计检验（χ²、t‑检验）以及状态机转移图可视化。

**📊 数据集**

数据集为两次实验室任务收集的 830 条学生提示（60 人 432 条 + 37 人 398 条），共 72 名学生，其中 25 名学生双次参与。

**📈 对比分析**

通过频数分布、跨会话比较、状态机转移分析以及显著性检验，发现第二次任务中学生从确认类提问转向程序化提问，继续世代学生更倾向主动请求，结果虽未全显著但显示显著趋势。

**⚠️ 局限性**

局限性包括：任务主题与难度不同导致难以单独归因提问变化、样本量有限、AI助手仅使用单一 GPT‑4 接口、未将提问与编码表现关联，以及 few‑shot 分类依赖人工标注，可能出现误判。

---

## 439. Understanding the Surprising Generalization Properties of Tabular Foundation Models

**arXiv ID:** 2608.17957 | [PDF](https://arxiv.org/pdf/2608.17957v1)

**作者:** Nour Shaheen `[一作]` (Polytechnique Montréal), Anthony L. Caterini `[通讯]` (Layer 6 AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了表格基础模型（TFM）在少量或单一真实表格上预训练时的跨域泛化能力，并提出了基于任务多样性与列级预处理的预训练策略。

**💡 创新点**

创新点在于：①证明单一表格的自监督预训练即可产生强泛化；②提出特征数量和任务数是决定泛化的关键指标；③发现 TFMs 的泛化主要源自学习检索与聚合机制而非传统的参数化迁移；④提出细粒度列级预处理能显著提升大规模语料库训练效果。

**🔧 技术方法**

使用了基于 Transformer 的 TabPFN/TabDPT 架构，采用自监督任务采样、注意力共享（W_Q=W_K）检索实验、以及多模型注意力相似度分析；评估指标包括 AUC/Accuracy（CC‑18）、相关系数/R²（CTR‑23）、IQM、Elo 等。

**📊 数据集**

主要数据集为 1,732 个 OpenML 真实表格（排除 CC‑18/CTR‑23/TabArena），单表实验使用 MNIST、Colleges 等；评估集为 CC‑18（72 个分类）和 CTR‑23（35 个回归）以及 TabArena 的 51 个任务。

**📈 对比分析**

与传统的 Logistic、RandomForest、XGBoost、TabDPT 等基线对比，单表预训练模型在大多数任务上可达到或超过随机森林水平；列级预处理后 IQM 与 R² 提升 0.2%–1.4%，AUC/Accuracy 提升 0.5%–2%；在 TabArena 51 任务上，4k+pp 版本相较于基线赢得 61%–67% 的任务。

**⚠️ 局限性**

局限性包括：未完全排除其他可能的泛化机制；仅验证了真实表格预训练，对纯合成数据的适用性未知；使用的 Transformer 仅为行级注意力，未探索单元级或混合注意力；并未深入研究模型内部检索机制的可解释性。

---

## 440. PRISM: Precision and contact-rich Real-world Industrial Skill dataset with Multimodal sensing

**arXiv ID:** 2608.17962 | [PDF](https://arxiv.org/pdf/2608.17962v1)

**作者:** Tengbo Yu `[一作]` (Peking University), Hangxin Liu `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

创建并公开了 PRISM 数据集，收集了 25+ 工业装配任务的 5,000+ 轨迹，包含 RGB‑D、力/扭矩、触觉和机器人状态。

**💡 创新点**

创新点在于提供大规模、多模态、工业真实场景数据，覆盖高精度接触操作，支持多机器人、多终端、多传感器同步。

**🔧 技术方法**

使用的技术包括多模态同步采集、遥操作平台（外骨骼、跟踪器、VR）、行为克隆基线（ACT、Diffusion Policy、π0）以及预训练+微调框架。

**📊 数据集**

使用的数据集是 PRISM 本身，并对比了现有数据集如 REASSEMBLE、OpenX、RT‑1 等。

**📈 对比分析**

比较方法是使用三种行为克隆基线在不同训练样本量、预训练与否、遥操作方式下进行实验，性能提升明显但仍低于理想水平，尤其在动态排序和精准插拔任务上成功率有限。

**⚠️ 局限性**

局限：对动态场景和精确力控制的学习仍不足，数据集虽大但仍需更细粒度的接触状态估计与闭环控制；遥操作方式差异导致演示质量不一致。

---

## 441. Grading Needs a Rubric, Not Intelligence

**arXiv ID:** 2608.17938 | [PDF](https://arxiv.org/pdf/2608.17938v1)

**作者:** Jhen-Ke Lin `[一作]` `[通讯]` (National Yang Ming Chiao Tung University), Jhen-Ke Lin (National Yang Ming Chiao Tung University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型进行一次性提取试卷题目及其评分标准（rubric），然后用小型模型反复进行评分，以验证小模型在rubric约束下可实现高可靠性评分。

**💡 创新点**

提出了any-to-bench框架，将评判任务的“智能”集中在一次性“摄取”步骤，证明在明确rubric的前提下，小模型即可代替昂贵模型完成评分，消除了评分中的长度偏差和同族偏好。

**🔧 技术方法**

使用前沿模型GPT‑5.6 Sol在摄取阶段提取题目和rubric，随后采用六种小型/中型模型（GPT‑5.6 Luna、Claude Sonnet 5）以低/中/高推理成本执行评分，并对比六个前沿模型（GPT‑5.6 Sol、Claude Opus 5）做验证。

**📊 数据集**

基准数据集为台湾三大国考（GSAT、AST、TVE）在113–115学年共164份试卷（7,121题），抽取24道开放式题目（短答、作文、图画），包含不同rubric细节。

**📈 对比分析**

通过ICC(2,1)衡量评分一致性，并进行方差分解；结果显示小模型评分ICC≥0.91，单一评判者即可达到与六评判者面板相同的可靠性；评分与评判者能力差异极小（0.2 %），评判者推理成本对评分影响≤0.006；在rubric条件下不存在长度偏好或同族偏好。

**⚠️ 局限性**

局限：仅测试台湾中文考试，未包含其他语言或评分体系；未与人工评分做对照，无法评估与人类评判的一致性；题目数量有限（每层六题）；在自由文本作文题中缺乏区分度，可能受模型写作能力或评判者分辨能力限制。

---

## 442. EvoTS-Agent: A Self-Evolving LLM Agent for Financial Time Series Change Point Detection

**arXiv ID:** 2608.17933 | [PDF](https://arxiv.org/pdf/2608.17933v1)

**作者:** Lei Jiang `[一作]` (Alan Turing Institute), Hao Ni `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种自我进化的LLM代理EvoTS-Agent，用于自动化金融时间序列的变点检测；

**💡 创新点**

通过EVO演化策略（Revision、Alternative Strategy、Recombination）结合验证引导的轨迹进化，实现对不同模型的动态选择与组合；

**🔧 技术方法**

利用LLM生成代码、可执行实验轨迹，并结合EDA特征选择、模型库与多种演化算子；

**📊 数据集**

在四个基准数据集（OU过程、Mean+Variance Shift、ADIA挑战、Bee-Dance）上进行评估；

**📈 对比分析**

与TS-Agent、DS-Agent、ResearchAgent等LLM代理相比，EvoTS-Agent在四大数据集上多项指标（F1、Hausdorff距离、成功率）均表现最佳或接近最佳，且保持100%执行成功率；

**⚠️ 局限性**

依赖有标签的验证集进行优化，缺乏标签或标签效率低的情况下表现受限。

---

## 443. AppendiGrade: An XAI-Enhanced Deep Learning Framework for Grading Appendicitis in Ultrasound with Gaussian Blur and Grad-CAM

**arXiv ID:** 2608.17923 | [PDF](https://arxiv.org/pdf/2608.17923v1)

**作者:** Fahad Ahammed `[一作]` (Ca' Foscari University of Venice), Golam Sorwar `[通讯]` (Southern Cross University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发并评估了多种预训练卷积网络对阑尾炎超声图像进行自动分类。

**💡 创新点**

结合高斯模糊+非锐化掩膜图像预处理、细调与Grad‑CAM可解释性技术显著提升模型性能。

**🔧 技术方法**

使用深度卷积网络（DenseNet201、InceptionV3、ConvNextTiny、VGG19）、图像增强、超参数优化与Grad‑CAM。

**📊 数据集**

4679张标注超声图像，包含五类（急性、穿孔、脓肿、阑尾石、正常）。

**📈 对比分析**

对比四种模型原始与优化后表现，InceptionV3优化后精度达95.6%，优于DenseNet201等。

**⚠️ 局限性**

数据集虽大但不涵盖所有临床变异，图像锐化可能掩盖细节，热图尚未得到医生验证。

---

## 444. Adaptive Policy Portfolios for Robust Markov Decision Processes

**arXiv ID:** 2608.17929 | [PDF](https://arxiv.org/pdf/2608.17929v1)

**作者:** Kasper Engelen `[一作]` (University of Antwerp), Marnix Suilen `[通讯]` (University of Antwerp)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了适应性策略组合在鲁棒马尔可夫决策过程中的构造与复杂度，提出了离线聚类构造方法并对其鲁棒回报进行评估。

**💡 创新点**

创新点在于给出了组合认证和合成问题的 -complete 证明，揭示了即使在矩形不确定性下仍为难题，并提供了可行的离线构造框架与实验验证。

**🔧 技术方法**

使用了鲁棒MDP、鲁棒回报与组合比较理论、实证评估方法、K‑means聚类、UCB在线识别等技术。

**📊 数据集**

实验使用了数据中心气候控制、无人机控制以及三种三维网格实例（S、M、L）作为数据集。

**📈 对比分析**

通过离线估计最优回报与组合最大回报的差距来衡量鲁棒回报；实验表明在K≥2时鲁棒回报显著下降，且UCB在有限迭代内能识别最佳成员。

**⚠️ 局限性**

局限性在于组合构造的 NP/PSPACE‑hard 复杂度阻碍了大规模精确构造；矩形不确定性下仍缺乏完整性结果；在线识别成本随组合规模增长而上升。

---

## 445. Hybrid ML for Lightweight Pre-Route Delay Estimation in Open-Source IC Design

**arXiv ID:** 2608.17914 | [PDF](https://arxiv.org/pdf/2608.17914v1)

**作者:** Marvin Castro Castro `[一作]` (Universidad de Costa Rica), Erick Carvajal Barboza `[通讯]` (Universidad de Costa Rica)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种轻量级混合机器学习模型（决策树+线性回归）用于 OpenLane 开源设计流的预路由延迟预测，显著降低误差；

**💡 创新点**

创新点在于将决策树的分区能力与每个叶节点的线性回归相结合，既保持非线性表达，又实现局部线性插值，且模型体积仅约10 MB，解释性强；

**🔧 技术方法**

使用的技术包括决策树分割、岭回归/OLS 线性回归、特征缩放、上下文参数、欧氏距离特征等；

**📊 数据集**

数据集来源于 OpenLane 对 SkyWater 130 nm 芯片的 RTL‑to‑GDSII 流程，涵盖 ISCAS‑89、OpenCores 等基准电路，分为训练、变异、未见数据集；

**📈 对比分析**

与 OpenLane 原始估计、随机森林、梯度提升树以及前人 delta‑预测模型对比，混合模型在大部分电路上 RMSE 下降 25–80%，相较于随机森林模型体积减小 300 倍，推理速度提升 2×；

**⚠️ 局限性**

局限性包括数据集规模有限（仅少量 130 nm 电路），对极端工艺/设计条件的泛化能力待验证，且未考虑路由拥塞、交叉耦合等更细粒度特征，未来需扩展数据与特征以进一步提升精度。

---

## 446. CABLE: Extending the Reach of Memory Retrieval via Complementary Antecedent-Based Linking and Expansion

**arXiv ID:** 2608.17911 | [PDF](https://arxiv.org/pdf/2608.17911v1)

**作者:** Zheling Tan `[一作]` (Shanghai Jiao Tong University), Dequan Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CABLE插件，在长时对话记忆中通过先验查询生成稀疏的前置关联链接，扩展检索范围。

**💡 创新点**

创新点在于强调检索互补性，使用重叠减法和LLM验证构造稀疏、非冗余的前置关联图。

**🔧 技术方法**

采用双重检索、余弦相似度、LLM验证以及图式扩展等技术。

**📊 数据集**

使用LoCoMo和MA‑LongMemEval两个长记忆对话基准。

**📈 对比分析**

在A‑MEM、SimpleMem、Mem0g三种记忆体系上与同基线对比，平均LLM判分提升0.5~6%，在多会话和偏好题型提升高达23%。

**⚠️ 局限性**

局限包括对时间推理帮助有限、需额外LLM推断构建链接、仅能一次跳扩展，且依赖于检索预算与图规模。

---

## 447. Dynamic Compression in Recurrent Networks

**arXiv ID:** 2608.17896 | [PDF](https://arxiv.org/pdf/2608.17896v1)

**作者:** Jyothish Pari `[一作]` (Massachusetts Institute of Technology), Pulkit Agrawal `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在循环网络中提出动态压缩机制，通过选择性重新扫描历史上下文来提升多函数的再利用能力，并在固定状态大小下减少所需存储。

**💡 创新点**

创新点在于将传统单向一次性压缩转化为可动态更新的压缩：模型在后续任务出现时可再次遍历并写回相关历史，从而在不增大状态容量的前提下提高关键信息的精度。

**🔧 技术方法**

采用 Gated DeltaNet（线性注意力 RNN）并加入选择头和重扫模块；利用自监督写强度 β 生成重扫代码表，用于预测并执行局部重扫；还对比了重复上下文模型与 oracle 动态重扫。

**📊 数据集**

使用合成的线性函数重用任务：包含多基矩阵（K 个线性变换）和少样本查询，输入为 4‑tuple 形式的张量，全部在人工构造的序列中完成。

**📈 对比分析**

与单次压缩基线、重复上下文模型、oracle 动态重扫和代码表动态重扫对比，动态重扫在小状态下（如 111k 元素）实现了比单次压缩更低的 MSE，逼近 oracle 上限，验证了该方法的有效性。

**⚠️ 局限性**

仅在受控合成数据上验证，重扫空间的参数化和监督方式尚不通用，如何将此机制推广到自然预训练任务或更大规模的数据仍是待解决的问题。

---

## 448. BEAR-Bench: A Bilingual Enterprise and Academic Reasoning Benchmark for Multimodal Models

**arXiv ID:** 2608.17895 | [PDF](https://arxiv.org/pdf/2608.17895v1)

**作者:** Liubov Chubarova `[一作]` (Yandex), Alexey Zaytsev `[通讯]` (Applied Ai Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了BEAR-Bench——一个包含1000道双语（英俄）多模态多步推理问题的专业文档基准，涵盖财报、投资者报告、流程图以及科研论文中的公式、图表、文本等视觉信息；

**💡 创新点**

提出了多步推理、完全自包含答案和无外部知识的评测原则，并系统比较了多模态大型语言模型（MLLM）的推理能力及幻觉检测方法，弥补了现有基准在俄语与多模态推理上的空白；

**🔧 技术方法**

采用了Gemini、Qwen、Claude等多款M模型，利用链式推理提示、图像分辨率控制、内部状态与token不确定性、隐藏层探测和LLM-as-a-judge等技术进行评估；

**📊 数据集**

使用公开的美国SEC EDGAR、俄国公司披露平台以及arXiv、CyberLeninka等学术预印本，经过OCR与视觉内容筛选后得到的专业文档图片；

**📈 对比分析**

在16款MLLM上测试，最佳模型Gemini 3.1 Pro与Qwen3.5‑397B‑A17B分别达75.1%和75.4%整体准确率；幻觉检测实验表明，针对短答案的最大token概率或隐藏层探测表现最佳，长答案时LLM‑as‑a‑judge最高（BalAcc ≈ 0.81），但整体仍有显著提升空间；

**⚠️ 局限性**

局限性包括单页限制（不支持跨页/跨文档推理）、仅覆盖英俄两语、数据量有限、评测依赖外部LLM判断、推理深度标注为粗粒度，且幻觉检测效果仍不稳定

---

## 449. Improving Complex Moiré Removal with Generative Supervision

**arXiv ID:** 2608.17883 | [PDF](https://arxiv.org/pdf/2608.17883v1)

**作者:** Xinyang Gu `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了WildMoiré数据集，利用多种生成模型生成并筛选高质量监督，从而提升复杂屏幕摩尔纹的去除效果。

**💡 创新点**

通过多模型生成、空间与色彩对齐、patch级过滤以及A-FINE评分自动挑选最佳监督，实现无监督生成可靠训练对的创新方法。

**🔧 技术方法**

结合SDXL、Nano-Banana-2、Qwen-Image-Edit、GPT-Image-2、FLUX.2等生成模型、光流对齐、SSIM评估、A-FINE质量评分和尺度变换增广等技术。

**📊 数据集**

使用6,832对WildMoiré训练对以及约250对真实清晰验证对，辅以UHDM和DCID等现有数据集进行训练。

**📈 对比分析**

在UHDM+DCID基线上加入WildMoiré及尺度增广后，ESDNet、SDXL和Qwen-Image-Edit的PSNR、SSIM、MUSIQ等指标均提升约1-2 dB，证明性能显著提升。

**⚠️ 局限性**

生成监督受模型偏差与伪造风险限制，阈值设置需在数据量与质量之间权衡，且对极端多色大尺度摩尔纹的覆盖仍有限。

---

## 450. ControlledShifts: Towards Standardizing Robustness Evaluation in Trajectory Prediction Under Distribution Shifts

**arXiv ID:** 2608.17882 | [PDF](https://arxiv.org/pdf/2608.17882v1)

**作者:** Ingrid navarro `[一作]`, Jean Oh `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种框架和基准套件，用于在分布转移下标准化评估轨迹预测模型的鲁棒性。

**💡 创新点**

通过共享的特征化和分割公式，系统地重新划分现有轨迹数据集为分布内和分布外部分，并提出统一的鲁棒性评分来评估模型的预测质量和稳定性。

**🔧 技术方法**

使用了特征化函数和分割函数的组合来创建控制分布转移，并通过基于变压器的架构进行基准测试。

**📊 数据集**

使用了Waymo开放运动数据集，该数据集包含约45K场景，涵盖多种道路条件和场景复杂性。

**📈 对比分析**

通过与现有模型的比较，展示了不同模型在处理潜在相关性和环境结构方面的关键差异，提出的基准显示出显著的性能下降，尤其是在分布转移条件下。

**⚠️ 局限性**

鲁棒性评分相对于选择的基准模型，这可能影响比较的有效性；此外，当前的三个变化轴并未涵盖自主系统可能遇到的所有转移情况。

---

## 451. A Kernel-Checked Exclusion Certificate for Erdős Problem 647

**arXiv ID:** 2608.17880 | [PDF](https://arxiv.org/pdf/2608.17880v1)

**作者:** Ibrahim Mian `[一作]` (Millennium Research), Shayaan Siddique `[通讯]` (Millennium Research)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用 Lean 4 证明并证实 Erdős 问题 647 在 24 < n ≤ 10⁹ 范围内无解，给出完整的可核查证书链。

**💡 创新点**

创新点在于构造仅使用最大平滑部分并利用“倍增”规则的低成本证明链，且整个证明仅依赖 Lean 内核及三条标准公理（propext、Classical.choice、Quot.sound），完全消除外部求解器和 native_decide 的使用。

**🔧 技术方法**

采用 Lean 4 + mathlib、贪心区间覆盖算法、十进制 JSONL 证书链、独立 Python 重新验证、lean4checker 以及多平台从源码构建与审计技术。

**📊 数据集**

数据集为 6,685,922 条“kill witness”记录，约 260 MB 源码，覆盖区间 (24, 10⁹]；另有 495、2,829、17,941、123,323、891,554 条记录用于更小区间验证。

**📈 对比分析**

与之前的未核查计算（例如 Idén、Hughes、bentrd 的 10¹² 级别搜索）相比，本方法在可信度上大幅提升：所有链条均通过 Lean 核心重放、跨实现一致性检查和多机/多编译器编译一致性，证明效率仅受链条长度影响，现已完成 10⁹ 范围的完整核查。

**⚠️ 局限性**

局限性在于证书规模随区间指数增长，10¹⁰ 级别已需约 45 M 条记录、2 GB 源码，10¹² 则超出可行范围；因此当前方法仅能覆盖到 10⁹ 范围，若要进一步扩展需改进证书压缩或使用更精细的模量归约方式。

---

## 452. MetaSapiens v2: Advancing Real-Time Foveated Neural Rendering via Foveation-Aware Pruning and Stereo Warping

**arXiv ID:** 2608.17969 | [PDF](https://arxiv.org/pdf/2608.17969v1)

**作者:** Weikai Lin `[一作]` (University of Rochester), Yu Feng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4`

**🎯 论文内容**

本文提出了一种基于高斯散射（Gaussian Splatting）的视差聚焦渲染方案，旨在为AR/VR中的神经渲染提供高效硬件加速器。

**💡 创新点**

创新点在于将高斯散射与视差聚焦渲染技术结合，并设计了针对该渲染流程的专用硬件加速器，以提升渲染速度与视觉质量。

**🔧 技术方法**

主要技术包括高斯散射、视差聚焦渲染、神经渲染框架以及定制化硬件加速单元。

**📊 数据集**

未提供具体使用的数据集信息。

**📈 对比分析**

未提供对比实验或性能评估结果。

**⚠️ 局限性**

局限性主要体现在缺乏对复杂真实场景的全面验证，以及未说明硬件实现的能耗和成本。

---

## 453. Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection

**arXiv ID:** 2608.17965 | [PDF](https://arxiv.org/pdf/2608.17965v1)

**作者:** Bin Li `[一作]` (Beijing Jiaotong University), Siyang Lu `[通讯]` (Beijing Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LoRD，一种基于路由自编码重建距离的后置校准框架，用于提升日志异常检测模型的置信度可靠性。

**💡 创新点**

创新点在于为预测为正常与异常的日志分别训练自编码器，利用重建距离划分风险区间，并按区间动态软硬校准置信度，从而显著降低误判时的过度自信。

**🔧 技术方法**

采用的技术包括路由特定自编码器、重建距离度量、软硬校准策略、拒绝区域机制、以及与温度缩放、逻辑缩放、贝塔缩放、选择性缩放和集成等常见后置校准方法对比。

**📊 数据集**

使用了四个大规模超算日志基准数据集：BGL、Spirit、Liberty 和 Thunderbird。

**📈 对比分析**

在多种检测器（TextCNN、LogRobust、LightLog、NeuralLog、GPT2）和四个数据集上与传统校准方法对比，LoRD 在异常误判置信度（CoE）从接近 1 降低至约 0.5，保持高置信度的正确异常检测，同时不降低检测精度。

**⚠️ 局限性**

局限性包括：需访问模型隐藏表示，阈值和边界依赖验证误差，主要关注误判置信度降低，可能牺牲整体 ECE、NLL、Brier 等指标；在误报/误检极少时校准边界不稳定。

---

## 454. COMA: A Compositional Misleading Attack Class on Security-RAG, and a Causal Counterfactual Defense

**arXiv ID:** 2608.17960 | [PDF](https://arxiv.org/pdf/2608.17960v1)

**作者:** Chinmay Gondhalekar `[一作]` (S&P Global), Urjitkumar Patel `[通讯]` (S&P Global)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 COMA（Compositional Misleading Attack）——一种利用检索增量生成（RAG）系统中文档组合方式误导安全协作伙伴的攻击，并设计了 Causal Counterfactual Defense （C3D）来检测并定位此类攻击。

**💡 创新点**

创新点在于：①定义了四个严格条件（C1–C4）使单个文档均无异常但组合可误导；②揭示了“推理而非读取”这一治理原则；③首次使用留一法因果影响量化（leave‑one‑out causal influence）对 RAG 输出进行审计；④提出了聚合版 C3D 以抵御分散式攻击者。

**🔧 技术方法**

主要技术包括：检索增量生成（RAG）框架、因果影响估计（基于留一重跑）、置信度阈值判定、聚合影响检测、实验评估的多模型对比（如 GPT‑4, Claude‑3 等）。

**📊 数据集**

实验使用合成种子（文档解析与 token‑auth 库场景）与真实 CVE（CVE‑2021‑33813 JDOM XXE）构造的检索集合，另外对四个无攻击的真实 CVE 进行无误检验。

**📈 对比分析**

对比指标包括攻击成功率（action‑corruption deterministic，verdict‑flip 随模型能力下降），C3D 的定位准确率（100% 在攻击案例中，无误报在四个正例中），以及聚合版对分散式攻击的检测率（100%）。性能表现表明 C3D 在所有五个模型上均能及时定位并阻断攻击。

**⚠️ 局限性**

限制：攻击样本数量有限，缺乏大规模基准；C3D 的阈值设定及连续影响度量仍需进一步校准；对极度分散的攻击需要更高注入预算，未完全覆盖无穷攻击者的情形。

---

## 455. Do Large Language Models Play Six Degrees of Separation? Measuring Topological Compression in Long-Context Manifolds

**arXiv ID:** 2608.17950 | [PDF](https://arxiv.org/pdf/2608.17950v1)

**作者:** Md. Faiyaz Abdullah Sayeedi `[一作]` `[通讯]` (BRAC University), Md. Faiyaz Abdullah Sayeedi (BRAC University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型隐藏层的几何拓扑，证明深层隐藏空间呈现小世界网络，并利用此结构进行零样本幻觉检测。

**💡 创新点**

首次用外部语义锚点和稀疏相似度阈值阐明隐藏空间的相位转移，并将拓扑特征作为可靠性指标。

**🔧 技术方法**

使用隐藏状态相似度阈值稀疏化构造无权图、BFS路径搜索、语义锚点选择、阈值扫描、Zero-shot幻觉判别。

**📊 数据集**

WikiText（150-300词长段落）与RAGognize（闭域问答、标注幻觉）数据集。

**📈 对比分析**

对比早期语法层与深层推理层的连通率与平均跳数，发现深层连通率从0%跃升至≈90%，平均跳数≤6；在RAGognize上，幻觉样本连通率降至<20%，平均跳数>7，Zero-shot判别AUROC为0.89。

**⚠️ 局限性**

局限包括对大规模模型计算成本高、对超长上下文的适用性未验证、阈值τ需重新校准。

---

## 456. SIGMA: SHAP-Guided Implicit-Trajectory Generation for Metadata-Free LLM-Based AutoFE

**arXiv ID:** 2608.17948 | [PDF](https://arxiv.org/pdf/2608.17948v1)

**作者:** Xuan Zheng `[一作]` (Yokohama National University), Shinichi Shirakawa `[通讯]` (Yokohama National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种无元数据的LLM驱动自动特征工程框架SIGMA，利用SHAP值进行特征分组并在不暴露语义信息的情况下生成新特征。

**💡 创新点**

创新点在于：①用SHAP值替代语义描述作为任务感知信号；②设计了分组生成策略（intra‑group 与 cross‑group）；③引入EXposed‑feature Implicit Trajectory (EXIT) 通过曝光特征隐式记录优化轨迹，从而保持近乎恒定的上下文长度并显著降低特征重复率。

**🔧 技术方法**

核心技术包括：SHAP解释、LLM (如 Qwen3‑4B‑Instruct)、分组生成策略、EXIT轨迹管理、操作预定义与跟踪、vLLM 加速推理。

**📊 数据集**

在 16 个公开表格分类数据集（OpenML、Kaggle）上评测，最大样本 50k，使用 XGBoost 作为下游模型。

**📈 对比分析**

与语义化 LLM 方法 CAAFE、无元数据 LLM 方法 OCTree 以及传统 AutoFE（DFS、OpenFE、AutoFeat）比较，SIGMA 在 F1 分数上与 CAAFE 相当，优于 OCTree，且在仅使用约 5.4 个接受特征的情况下实现与 OpenFE 相近的性能，显示出更高的特征利用效率；同时保持上下文长度稳定，显著降低特征重复率（从 37.2% 降至 6.8%）。

**⚠️ 局限性**

局限性包括：①操作约束仍较弱，特征重复率仍约 7%；②仅针对分类任务，尚未验证回归场景；③在小样本数据集上易过拟合验证集，导致泛化性能下降。

---

## 457. SpeechSense: A Paralinguistic-Focused Dataset for Fine-Grained Speech Sentiment Analysis

**arXiv ID:** 2608.17931 | [PDF](https://arxiv.org/pdf/2608.17931v1)

**作者:** Shicheng Ma `[一作]` (Chinese University of Hong Kong), Irwin King `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了 SpeechSense 数据集，利用高保真合成语音与人工验证，专注于通过声学特征区分八类细粒度言语态度情感；

**💡 创新点**

创新点在于：①把研究焦点从基本情绪转向仅靠声学信息可辨的言语态度；②提出专属的 8 类标签体系；③采用角色扮演式 TTS 合成方法，确保情感可区分并保持语义中性；

**🔧 技术方法**

技术手段包括：文本-语调解耦文本生成（Qwen3‑Max）、Lovo.ai TTS 角色扮演合成、双阶段人类验证与过滤、LoRA 微调多模态 LLM、传统语音编码器（Whisper、HuBERT、Wav2Vec2）等；

**📊 数据集**

使用的数据集为 SpeechSense，包含 30 位虚拟说话人生成的 1522 条训练样本和 669 条经过人工验证的测试样本；

**📈 对比分析**

实验对比多模态 LLM、文本 LLM 与语音编码器，结果表明拥有声学输入的模型（如 Qwen2.5‑Omni‑Audio）在监督训练后准确率可达 57% 左右，文本模型仅 20% 级别，显著验证声学特征在细粒度情感识别中的主导作用；

**⚠️ 局限性**

局限性包括：①完全使用合成语音，可能存在与真实语音的域差异；②仅覆盖英语的八种态度，缺乏跨语言和更丰富情感维度；③声学多样性受限于单一 TTS 引擎与 30 位声卡，未来需扩展更多说话人与引擎。

---

## 458. Love Handles: Decimation for Deformation Handles with Compact Support and Low Memory Footprints

**arXiv ID:** 2608.17930 | [PDF](https://arxiv.org/pdf/2608.17930v1)

**作者:** David IW Levin `[一作]` (University of Toronto and NVIDIA), Teseo Schneider `[通讯]` (University of Victoria)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于网格简化的迭代算法，用于自动生成稀疏、局部支持的变形控制柄（deformation handles）及其权重，从而构建高效的ROM（Reduced‑Order Model）用于弹性固体的实时仿真。

**💡 创新点**

创新点：
- 第一次使用基于简化（decimation）的算法实现错误驱动的柄位置与权重优化，确保在每个顶点的支持数固定；
- 生成的权重保证紧凑支持、稀疏性与分区求和（partition of unity），显著降低内存开销；
- 结合紧凑支持的权重实现高效的超矩阵（hyper‑reduced）采样（cubature），实现超过100 FPS的实时弹性仿真。

**🔧 技术方法**

关键技术：
- 迭代权重更新与柄删除采用类似网格简化的梯度下降策略，利用Yen’s算法快速更新顶点最近柄集合；
- 线性方程组求解通过块‑Jacobi预条件的共轭梯度（PCG）实现；
- 采用二次正则化的局部权重重分配、约束矩阵简化；
- 通过谱能量选择候选四面体，使用非负最小二乘（NNLS）求解多元素的最优采样权重。

**📊 数据集**

实验数据集：15个四面体网格，顶点数从几千到近20万，四面体数从几千到近80万，包括Bunny、篮球、桥梁、龙、花瓣、鸡、柴等复杂几何体。

**📈 对比分析**

与现有方法（如Brandt et al. 2018）在等内存（同等基函数数）下比较：在同等内存下误差平均下降约6×，单例最高可达15×；在实时仿真上，使用该方法可在大多数模型上实现100FPS以上；减法算法的预处理时间在最坏情况下可达36小时，但通过温启动可提升50×。

**⚠️ 局限性**

局限性：
- 预处理时间较长，尤其在复杂网格下；
- 对于已能用粗网格良好近似的几何体，方法可能无法进一步压缩DOF，甚至导致更多DOF；
- 需要先提供目标位移场（如线性模态），不具备“数据自由”的特性；
- 采用固定的顶点支持数限制了对刚性部分的压缩表达，动态支持尺寸会更灵活。

---

## 459. On the Estimation of Chernoff Information

**arXiv ID:** 2608.17916 | [PDF](https://arxiv.org/pdf/2608.17916v1)

**作者:** Kadircan Aksoy `[一作]` (German Aerospace Center (DLR)), Peter Jung `[通讯]` (German Aerospace Center (DLR))

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于k近邻的Chernoff信息估计器，通过对Chernoff函数的导数直接估计并配合二分搜索求解最优混合参数s，从而得到Chernoff信息的无偏一致估计。

**💡 创新点**

创新点在于：①将Chernoff信息的求解转化为对其导数符号的判断，避免了差分估计因偏差导致的错误；②构造了闭式的kNN导数估计器并证明其L₁、L₂一致性；③利用二分搜索在有限精度下精确逼近s*，并给出一致性证明。

**🔧 技术方法**

主要技术包括：k近邻密度估计、密度比估计、导数估计公式、二分搜索优化、以及概率极限定理和Vitali收敛定理的理论分析。

**📊 数据集**

实验数据集：对称截断高斯分布、截断指数分布（一维和多维）以及MNIST手写数字的两类样本。

**📈 对比分析**

与已知解析解的比较表明，估计器在样本量约10³–10⁴时即可达到10⁻⁴级别的误差；在高维实验中，s*的收敛速度保持稳定，但Chernoff信息的估计误差随维度呈指数增长。

**⚠️ 局限性**

局限性：仅给出渐进一致性结论，缺乏有限样本误差上界或偏差方向说明；对高维数据的样本复杂度未定；缺少对不同k值和样本不平衡情况的详细分析。

---

## 460. Average-Case Optimal Encodings and Efficient Worst-Case Indices for Element Distinctness Queries

**arXiv ID:** 2608.17907 | [PDF](https://arxiv.org/pdf/2608.17907v1)

**作者:** Philip Bille `[一作]` (Technical University Of Denmark), Filippo Lari `[通讯]` (University Of Pisa)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对元素不重复查询（All‑Distinct）问题，本文在随机数组（均匀独立）与非随机数组两种模型下，给出了最优编码空间下界，并设计了平均案例最优的编码方案；同时在索引模型中给出了空间-时间下界并提出了几乎匹配该下界的简易索引。

**💡 创新点**

创新点主要包括：
- 使用 Alon‑Orlitsky 结果与 Ramanujan Q‑函数，对随机实例的编码空间给出精确的平均下界；
- 设计与该下界相匹配的编码，支持 O(1)、o(log²log n)、O(log log n) 等不同查询时间，甚至对任意 ω(1) 字母表实现 O(1) 期望查询；
- 在索引模型中通过细致的决策树下界证明任何 n/b 位索引需至少 Ω(b/log b) 次数组探测，并给出空间接近下界、查询时间仅多 log² b 因子的简单索引。

**🔧 技术方法**

所用技术包括：信息理论与 Shannon 熵、Alon‑Orlitsky 码长下界、Ramanujan Q‑函数、块 Huffman 编码、Elias‑Fano 选择结构、预计算查找表、完全可索引字典（FID）、前驱查询结构（Pătrașcu、Gupta 等）、决策树与二分搜索、以及对特殊构造实例的分析。

**📊 数据集**

使用的数据集：
- 均匀独立随机数组（元素取自 [1, σ]）；
- 为下界证明设计的特殊实例集合 C，其中每个块包含两个 1，其他位置填充特定值。

**📈 对比分析**

与以往仅给出 2n–O(log n) 最差案例下界的结果相比，本文的平均下界更高（例如 σ=2 时为 n 位），并提供匹配该下界的编码；索引方面，空间减少了约 log b 的因子，查询时间保持在 O(T_dist(b,σ))，仅比理论最优多一个 log² b 的系数。

**⚠️ 局限性**

局限性：
- 编码方案对字母表大小 σ 的取值范围有不同设计，若 σ 与 n 同阶需额外的低阶项；
- 期望查询时间的 O(1) 结果仅在 σ = ω(1) 的随机模型下成立；
- 编码构造与解码实现相对复杂，涉及多棵 Huffman 树和查找表；
- 索引下界与构造仍保留 log² b 的性能间隙，未完全达到理论极限。

---

## 461. AutoResearch: Insight In, Hallucination Out

**arXiv ID:** 2608.17906 | [PDF](https://arxiv.org/pdf/2608.17906v1)

**作者:** Yiming Ren `[一作]` (EvoMap), Junjie Wang `[通讯]` (EvoMap)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AutoResearch，两阶段自动科研系统，实现从想法生成到实验执行的全流程自动化。

**💡 创新点**

创新点在于将想法生成与执行分别实现知识与证据双重扎根，采用多模态多模型生成与跨域审查，构建可检证的实验流程。

**🔧 技术方法**

使用多模态生成、跨域机制迁移、多模型交叉评审、协同多智能体工作流和独立证据审核等技术。

**📊 数据集**

主要使用RSICD遥感图像描述数据集、矩阵乘法基准、Titanic、House Prices、Disaster Tweets等Kaggle数据集。

**📈 对比分析**

与四个现有自动科研系统对比，AutoResearch在RSICD提升mR从32.84到34.69，问题事件最少；矩阵乘法实验纠正不稳定结果；在Kaggle任务实现依据证据的继续/修订/终止决策，表现优于对手。

**⚠️ 局限性**

局限在于对外部信号覆盖和实验评判准则依赖，缺乏持续的知识更新机制和对复杂实验设计的自动化支持。

---

## 462. Multi-Agent AI System for Radiology Report Structuring and Quality Assurance with Independent Radiologist Evaluation

**arXiv ID:** 2608.18072 | [PDF](https://arxiv.org/pdf/2608.18072v1)

**作者:** Iryna Hartsock `[一作]`, Ghulam Rasool `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并评估了一套本地多代理 AI 系统，用于将放射学报告的 Findings 部分按解剖结构重新组织，并对报告整体进行质量保证检测。

**💡 创新点**

创新点在于将规则匹配、LLM 推理和质量检查四种代理集成到单一流水线；既能在不重写原文的前提下实现句子级结构化，又能自动检测报告级的逻辑不一致、性别–解剖冲突及关键发现未记录等错误，且全部模型均在本地部署。

**🔧 技术方法**

使用了正则表达式规则、量化版 LLaMA‑3（8B）和 DeepSeek‑R1‑Distill‑Llama‑70B（量化至 q4）模型，配合 Python、Ollama 推理平台和 NVIDIA H100 GPU。

**📊 数据集**

数据集为 638 篇胸、腹、盆 CT 放射报告（共 22,270 句），由 15 位担任放射科医师撰写，涵盖多种书写风格。

**📈 对比分析**

通过两名独立放射科医师评估 45 篇报告（含 41 篇被 QA 标记、4 篇未标记）进行客观比对。结构化准确率为 69%，错误率仅 4%；QA 任务正确识别不一致率 82%，并获得 84% 的“优秀/良好”评价。平均每份报告处理时间为 55.6 秒。

**⚠️ 局限性**

局限性包括：评估样本量小且偏向 QA 标记报告；仅涉及两位评估者，缺乏多中心验证；结构化过程对多解剖区的句子存在主观判定，偶有重复出现；QA 组件在检测潜在错误时产生少量误报。

---

## 463. EDITBRIDGE: Towards Faithful and Efficient Ultra-High-Resolution Image Editing

**arXiv ID:** 2608.18063 | [PDF](https://arxiv.org/pdf/2608.18063v1)

**作者:** Jiayi Song `[一作]` (Shanghai Jiao Tong University), Ruihua Huang `[通讯]` (Alibaba)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 EditBridge 的高分辨率图像编辑框架，利用 diffusion bridge 将低分辨率编辑结果逐步细化为 4K 级别的高质量输出。

**💡 创新点**

创新点在于：①将图像编辑视为数据‑to‑数据的翻译任务，使用 diffusion bridge 直接在已编辑低分辨率图像上进行细化；②提出 prior‑guided block‑wise sparse attention (PG‑BSA)，根据低分辨率阶段的语义对应先验限制跨图像注意力，显著降低计算量并保持细节一致性。

**🔧 技术方法**

核心技术包括 Diffusion Transformer (DiT) 的桥式训练、PG‑BSA 稀疏注意力、FlashAttention 加速、LoRA 低秩适配、Prodigy 优化器等。

**📊 数据集**

使用公开高分辨率图像数据集 Aesthetic‑4k、Aesthetic‑Train‑V2，自动生成编辑指令（Gemini 3）和目标图像（Nano Banana Pro），共构建 5,000 对 1K/2K 图像和 1,500 对 4K 图像的训练集。

**📈 对比分析**

与 DiT‑SR、DiT4SR、PiSA‑SR、TSD‑SR、HiFlow、ScaleEdit 等基线在 1K/2K/4K 分辨率下进行 HaarPSI、M‑PSNR/M‑SSIM/M‑MSE/M‑LPIPS 以及推理时长对比。结果显示，EditBridge 在保持或略高于基线的视觉质量的同时，实现 3.6–8.4× 的速度提升（2K 仅 4.08 s，4K 61 s）。

**⚠️ 局限性**

局限性包括：仍需先训练低分辨率编辑模型；对极高分辨率仍受显存限制；稀疏注意力在某些细节上略逊于全注意力；性能高度依赖指令生成的质量与高分辨率源图像的可用性。

---

## 464. Chain-of-Experience for Continual LLM Improvement

**arXiv ID:** 2608.18027 | [PDF](https://arxiv.org/pdf/2608.18027v1)

**作者:** Haoqin Tu `[一作]` (University of California Santa Cruz), Shen Yan `[通讯]` (Bytedance Seed)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在测试阶段通过与环境/自身交互累计经验，构建并验证 Chain-of-Experience（CoE）框架，使 LLM 能在推理过程中持续学习与改进。

**💡 创新点**

首次将多种反馈（自我评估、执行器/正确性信号）嵌入迭代推理循环，实现持续性、可追溯的测试时学习，并系统评估其对模型性能与成本的影响。

**🔧 技术方法**

采用迭代生成模型、反馈驱动的 CoE 过程；对比常见基线（CoT、ICL、Dynamic CheatSheet、ACE、ToT 等）；使用 GPT‑5、Gemini‑2.5 Pro、Claude‑4.5 Sonnet 等顶级 LLM 进行实验。

**📊 数据集**

数学任务：AIME 2025、OmniMath；编程任务：LiveCodeBench V6、LiveBench (Code)；知识任务：EvaLearn、GPQA Diamond，涵盖算数、符号推理、代码执行和事实问答。

**📈 对比分析**

在六大基准上，将 CoE 与无反馈、CoT、ICL、DC、ACE 等方法对比；CoE 在所有模型平均提升约5.6% 准确率、19% 降低 API 成本；最优反馈组合可将准确率提升至约79%，且大多数收益集中在前几轮迭代。

**⚠️ 局限性**

模型对弱或错误反馈仍具一定鲁棒性但可能受限；不同反馈类型对提升效果差异显著；跨域经验共享与记忆压缩机制在实验中未表现出显著优势；实验仅覆盖有限任务，尚未验证在更复杂真实环境中的泛化。

---

## 465. A Denotational Semantics for Synchronized Regular Expressions (extended version)

**arXiv ID:** 2608.18007 | [PDF](https://arxiv.org/pdf/2608.18007v1)

**作者:** Lukas Grätz `[一作]` `[通讯]` (Technical University of Darmstadt), Lukas Grätz (Technical University of Darmstadt)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了同步正则表达式（pure sregex 与 full sregex）并给出其赋值语义（concretization），证明匹配问题是 NP‑完备，并提供泵引理、闭包性质以及与带反向引用正则表达式的可转化关系。

**💡 创新点**

创新点在于：① 用标签化的 Kleene 星和选择实现同步匹配；② 采用 Kripke‑style 的赋值语义而非操作语义；③ 引入 uptransition 以实现跨星同步（full sregex）；④ 给出泵引理证明纯同步语言非正则；⑤ 通过理论分析证明交集和补集不闭合，且交集非空判定不可判定。

**🔧 技术方法**

主要技术手段包括：Kripke 结构的世界与函数、partial/完整 concretization、图论概念（acyclic、injective、full）、泵引理、归约（3SAT→匹配）以及 NP‑复杂度分析。

**📊 数据集**

该工作为纯理论研究，没有使用任何实验数据集。

**📈 对比分析**

比较方法主要是理论归约和复杂度分析：与传统正则表达式的等价性、与带反向引用正则表达式的可转化性、与上下文无关文法的表达力对比；性能方面仅给出匹配问题的 NP‑完备结论，没有实验性能数据。

**⚠️ 局限性**

限制与挑战：① 计算复杂度高，匹配问题仅在 NP 内可解；② 对 full sregex 的泵引理尚未完全完善；③ 交集与补集不闭合，导致某些常见运算不可在模型内完成；④ 对交集非空判定不可判定，限制了理论上的组合与优化；⑤ 该模型在实际正则引擎中的实现与评估尚待进一步研究。

---

## 466. AViTS: Adaptive Spatiotemporal Token Selection for Efficient Dynamic-Resolution Generation

**arXiv ID:** 2608.17995 | [PDF](https://arxiv.org/pdf/2608.17995v1)

**作者:** Haoran Qin `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AViTS框架，在动态分辨率Diffusion Transformer中根据文本注意力和跨步特征变化自适应地选择重要Token进行分辨率上采样，减少高分辨率计算量并保持生成质量。

**💡 创新点**

创新点在于将空间语义重要性（图像Token与文本Token的交叉注意力）与时间动态重要性（跨采样步的Token特征方差）两种维度融合，形成统一重要性评分，用于驱动Token优先级化的自适应上采样；同时通过三阶段低–中–高分辨率的推理流程实现高效推断。

**🔧 技术方法**

技术包括流匹配（flow‑matching）训练的速度网络、Latent–Text交叉注意力提取、Token级跨步方差计算、min‑max归一化与线性融合、基于Top‑K选择的分辨率自适应上采样、坐标绑定噪声注入和正交上采样。

**📊 数据集**

使用DrawBench（文本生成）与GEdit（编辑）两个公开基准；在FLUX.1‑dev、FLUX.1‑Kontext‑dev和Qwen‑Image‑Edit三大模型上进行评估。

**📈 对比分析**

与稀疏注意力、特征缓存、动态分辨率、步长蒸馏、量化等多种加速方法对比。AViTS在不牺牲图像质量（ImageReward/CLIP Score）或编辑一致性（SC/PQ/OS）的前提下，单独实现约3.14×–5.45×加速，结合特征缓存可达9×、与蒸馏/量化可达14.76×。

**⚠️ 局限性**

局限在于依赖低分辨率阶段的注意力与方差估计；对极低分辨率或高噪声初始图像时重要性判定可能不准确；目前只在Transformer‑based模型验证，未探讨卷积或其他架构的迁移。

---

## 467. When Writing Style Drifts: Benchmarking Authorship Verification under Distribution Shifts in Genre, Time and the AI-Era

**arXiv ID:** 2608.17979 | [PDF](https://arxiv.org/pdf/2608.17979v1)

**作者:** Lotta Kiefer `[一作]` (University of Technology Nuremberg), Steffen Eger `[通讯]` (University of Technology Nuremberg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了首个德语作者身份验证基准AVShift，系统评估跨体裁、时间跨度及AI时代的分布偏移，提供了超过150K文本对的多场景数据集。

**💡 创新点**

创新点在于将三类分布偏移（体裁、时间、AI）统一到同一基准中，并引入特征稳定性评分，首次在非英语语料上揭示不同偏移对验证性能的影响。

**🔧 技术方法**

技术方法涵盖三类主流算法：基于手工风格特征的XGBoost、从多语种Transformer提取的MSR嵌入以及通过LoRA微调的Gemma-4-31B LLM。

**📊 数据集**

使用自爬取的fanfiction.de平台数据，覆盖论坛、评论和同人小说三体裁，时间跨度2004–2025年，划分为GenreShift、TimeShift和AIShift子基准。

**📈 对比分析**

在所有子基准上比较模型，发现微调后的Gemma在跨体裁和时间偏移下表现最优（最高F1≈0.77），时序偏移导致性能显著下降（最高降幅≈0.21），但AI时代未出现明显负面影响。

**⚠️ 局限性**

局限性包括：仅在单一德语平台构建数据，缺乏对AI使用程度的显式标注；评估的模型类别有限；并未探索更深层次的偏移适应方法，未来需扩展语言、平台与模型种类。

---

## 468. Colour Blinded by the Noise

**arXiv ID:** 2608.17976 | [PDF](https://arxiv.org/pdf/2608.17976v1)

**作者:** Harriet Mason `[一作]` (Monash University), Dianne Cook `[通讯]` (Monash University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实施了一套实验方法，用 Ishihara 色盲测试的形式评估在色散地图（choropleth）中将不确定性作为噪声来可视化的有效性。通过对五种地图类型（标准色散图、双变量图、VSUP、像素图、透明度图）在不同分布差异（D）和标准差（V）下的可视化表现进行对比，并与传统假设检验（t 检验与 Moran's I）产生的功效曲线进行对照。

**💡 创新点**

创新点在于：①把不确定性视为噪声而非信号，引入隐式测试（类似 line‑up 协议）来衡量信号可见度；②利用色盲测试的结构将视觉可读性转化为统计功效曲线；③系统化地比较多种不确定性可视化设计在抑制虚假信号方面的表现，并提出了“信号抑制”理论框架。

**🔧 技术方法**

技术手段包括：
- R 语言及 ggplot2、tidyverse 等包构建图形；
- Bayesian 随机层次模型生成合成数据；
- 5×5×5 纯因子设计（图类型×D×V）；
- 通用线性混合模型（GLMM）估计功效曲线；
- 自助法（bootstrapping）估计显著性水平；
- 统计功效曲线与传统假设检验结果进行对比。

**📊 数据集**

使用的是在实验中自行生成的合成数据：每个点遵循正态分布，均值来自两组（数字组与背景组）不同的正态分布，标准差 V 统一设置；地图几何使用澳大利亚州界。没有使用公开的实际地理或气候数据集。

**📈 对比分析**

比较方法：将每种地图类型在不同 D、V 组合下的正确识别率映射为功效曲线，并与模拟得到的 t‑检验和 Moran's I 的理论功效曲线对齐。结果显示：
- 标准色散图和双变量图对 V 变化不敏感，无法抑制噪声；
- VSUP 在 V=5 时有一定抑制效果，但与理论曲线不完全对应；
- 像素图和透明度图最接近理论功效曲线，但在低 V 时识别率偏低，在高 V 时偏高。整体而言，像素图和透明度图在抑制虚假信号方面表现最佳，但仍有敏感度改进空间。

**⚠️ 局限性**

局限性：
- 仅限于颜色通道，未探索形状、尺寸、位置等其他可视化通道；
- 采用极简的正态分布合成数据，缺乏复杂真实数据的挑战；
- 只测试单一不确定变量，未考察多变量共同不确定性情形；
- 像素图和透明度图缺乏可解释的图例，实际应用受限；
- 样本量（每图 50 次抽样）可能影响对比度和敏感度；
- 受限于实验平台，存在视觉疲劳、色盲测试后像等干扰；
- 只关注 choropleth 地图类型，未扩展到其他统计图表。

---

## 469. Target Speaker Identification: A Low-Latency Streaming Pipeline

**arXiv ID:** 2608.17972 | [PDF](https://arxiv.org/pdf/2608.17972v1)

**作者:** Patrick S. Burke `[一作]` (Children's National Hospital), Sean Kinahan `[通讯]` (Arizona State University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一个实时低延迟的目标说话人识别管线，将流式说话人分离与说话人验证结合，用于助听器中的选择性放大。

**💡 创新点**

将现有的流式说话人分离模型Diart与说话人验证模型Pyannote/TitaNet相结合，并实现约1秒的识别延迟，突破传统10-1000 ms延迟瓶颈。

**🔧 技术方法**

使用开源Pyannote进行离线分离与验证，Diart实现500 ms分块流式分离，Pyannote/TitaNet生成说话人嵌入，Cosine距离与阈值判定，ReactiveX驱动流式处理，DER、ROC、AUC等指标评估性能。

**📊 数据集**

使用This American Life Podcast Transcripts（期号670–702，主讲人Ira Glass）作为实验数据集，保证语音结构清晰且重叠较少。

**📈 对比分析**

通过DER、AUC、Accuracy、Precision、Recall、F1和Specificity评估，Diart+Pyannote在17集实验中实现平均准确率≈0.91、特异性≈0.96、召回≈0.56，整体识别延迟约1 s。

**⚠️ 局限性**

仅测试单一目标说话人、样本集有限、未在噪声或重叠场景下充分验证、未进行多次实验或显著性检验、延迟仍高于理想的sub‑10 ms，并且对背景噪声敏感。

---

## 470. Evaluating and improving crop-yield forecasting methods during extreme drought

**arXiv ID:** 2608.17971 | [PDF](https://arxiv.org/pdf/2608.17971v1)

**作者:** Shrey Gupta `[一作]` (Boston College), George Mohler `[通讯]` (Boston College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估并改进极端干旱年份（2012年）玉米产量预测方法，探讨特征分布不匹配与数据不规则性对模型性能的影响。

**💡 创新点**

提出通过SHAP特征选择和极端年份样本加权的模型适配方法，并验证大规模预训练Transformer VITA在分布不匹配下仍能保持优势。

**🔧 技术方法**

使用传统机器学习回归（岭回归、Lasso、随机森林、XGBoost、SVR、Huber、Extreme Lasso、TrAdaBoost.R2）以及深度学习预训练Transformer VITA，并对其进行改进。

**📊 数据集**

采用gridMET（1980-2018年玉米产量与16个气象驱动）和NASA Power（31个气象驱动）数据集。

**📈 对比分析**

通过对2012年极端干旱年和2013年正常年进行R²、RMSE等指标比较，VITA在所有设置下均优于传统ML模型，改进的VITA[+]和TrAdaBoost.R2[+]在极端年显著提升。

**⚠️ 局限性**

局限包括数据空间和时间不规则性、预训练数据分辨率低、仅使用气象驱动缺乏土壤和管理信息，以及模型对极端年份样本量和权重的敏感性。

---

## 471. TokEval: A Tokenizer Evaluation Suite

**arXiv ID:** 2608.18062 | [PDF](https://arxiv.org/pdf/2608.18062v1)

**作者:** Clara Meister `[一作]` `[通讯]` (EPFL), Clara Meister (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套完整的 tokenizer 内在评估指标，并通过控制预训练实验验证其与下游模型性能的关联。

**💡 创新点**

创新点在于引入面向数学、代码和多语言公平性的结构敏感指标，以及系统化的评估框架和可视化工具。

**🔧 技术方法**

使用了自定义的 tokenizer 训练与评估工具箱，配合 nanochat 1.27B 语言模型、BPE/UnigramLM 等子词算法。

**📊 数据集**

数据集包括 FineWeb-Edu、FineWeb2、FineMath、StarCoderData 以及 FLORES+、BLiMP、GSM8K、HumanEval、MBPP 等多任务基准。

**📈 对比分析**

通过混合效应回归与 Spearman 相关分析，发现信息理论指标能预测语言建模指标，结构指标能预测数学/代码任务，整体性能提升不一，主要用于筛选而非最终排序。

**⚠️ 局限性**

局限性包括仅在单一模型规模与架构下验证、实验规模有限、未覆盖所有语言和能力，统计功效有限。

---

## 472. Delegation Asymmetry in Agentic Recommender Systems: Measuring Two-Sided Receptivity in Online Dating

**arXiv ID:** 2608.18058 | [PDF](https://arxiv.org/pdf/2608.18058v1)

**作者:** Daria Leshchikova `[一作]` (Fleamily, Inc.), Valerii Klimov `[通讯]` (University of Notre Dame)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在在线约会平台上，通过问卷测量用户在代理（LLM）发送和接收消息时的接受度，并构建两维潜在变量模型评估代理委托市场的可行性。

**💡 创新点**

创新点在于首次在同一被试内同时量化发送与接收的“代理可接受度”，发现两者构成独立但相关的维度，并揭示约0.7 SD的委托不对称；同时提出可直接用于产品设计的“接收者可接受度”路由信号与“互惠规则”定价框架。

**🔧 技术方法**

采用分级响应模型（GRM）结合潜变量回归进行测量模型估计，并通过BIC、LRT检验维度与测量不变性；利用交叉验证评估路由信号的AUC与四分位提升；通过随机配对模拟计算可行对话率与互动质量。

**📊 数据集**

使用两份大规模问卷数据，分别在俄语与英语用户中收集7项潜在变量题目，共计2,499完整案例（含两种语言）以及另一份关于生成式功能兴趣的问卷（2,894案例）。

**📈 对比分析**

与单维模型、均值差异、随机配对对照等进行比较，发现两维模型显著优于单维（ΔBIC≈52，LRT>100）；路由信号在留一项的五折交叉验证中AUC达0.88，四分位提升约3.1倍；在随机配对基准下，未路由时仅4–13%对话实现，而按接收者可接受度前25%路由可将互动质量提升3.4倍。

**⚠️ 局限性**

局限包括：数据为自报偏好而非真实代理使用行为，样本来自单一平台且英语子样本小；模型假设随机配对与静态倾向，未考虑匹配结构、同质性与动态反馈；跨语言测量不完全不变，且对文化差异的解释仍为假设；对接收者可接受度的实际部署与隐私伦理仍需进一步验证。

---

## 473. Minimizing Commit Rules for DAG-based Atomic Broadcast

**arXiv ID:** 2608.18029 | [PDF](https://arxiv.org/pdf/2608.18029v1)

**作者:** Petr Kuznetsov `[一作]` (Télécom Paris), Sara Tucci-Piergiovanni `[通讯]` (Université Paris-Saclay)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文定义了基于无认证、按轮构建的有向无环图（DAG）的提交规则，并引入子规则关系，用以寻找在最终同步和异步网络下的最小提交规则，进而设计了新的原子广播协议Minnow；

**💡 创新点**

创新点在于提出“子规则关系”这一工具，能够比较并证明提交规则的最小性，并首次给出在两种网络模型下的最优（最小）提交规则；

**🔧 技术方法**

技术上主要利用DAG的可嵌入模式（witnesses/patterns）与子图嵌入概念来定义提交规则，并用严格的数学证明来验证安全性与活性；

**📊 数据集**

本文未使用任何实验数据集，全部讨论为理论证明与协议设计；

**📈 对比分析**

通过子规则关系证明任一规则的最小性后，作者将其应用于Minnow协议，理论上相比现有协议（如Black Marlin、Bullshark等）可在更少的条件下提交更多顶点，从而降低延迟、提升吞吐量；

**⚠️ 局限性**

局限在于缺乏实证评估、证明在异步模型下的完整性未完全给出、以及对其他通信组件（如认证DAG）的适用性仍需进一步研究。

---

## 474. Why GPT-Style Models Do Not Directly Transfer to Symbolic Music: Compression in the Wrong Coordinate System

**arXiv ID:** 2608.18025 | [PDF](https://arxiv.org/pdf/2608.18025v1)

**作者:** Yi Wang `[一作]` `[通讯]` (Tsinghua University), Yi Wang (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出有效性–无损性框架，将音乐分词视为构造可预测压缩坐标系，并在符号音乐上验证其有效性。

**💡 创新点**

通过定义事实‑令牌边界和令牌‑状态边界，将分词任务分离为坐标构造和上下文无损保持；首次系统证明时间去耦合、可逆规范化能显著提升预测压缩，且无需结构标签即可产生高阶音乐组织。

**🔧 技术方法**

使用基于Transformer的因果序列模型、可逆编码、时间与音高去嵌套、可逆规范化、注意力偏置以及固定关系投影对照实验，衡量预测码长度。

**📊 数据集**

主要使用公开可获取的符号音乐数据集Pop‑K以及对照集ComMU进行实验。

**📈 对比分析**

对不同坐标构造方案进行匹配比较，固定学习器和数据集，采用预测码长度（bits per event）评估；结果显示引入时间坐标和可逆规范化可将码长度降低约40%，而仅通过短序列化并不能提升压缩效果。

**⚠️ 局限性**

实验仅覆盖所选表示、受限模型规模和符号音乐场景；预测码长度仅相对模型而非绝对；未对语义分解、可扩展性或生成质量进行深入评估。

---

## 475. Can Large Language Models Explain Flight Safety Events? A Prior-Guided Semantic LLM-based Approach

**arXiv ID:** 2608.18017 | [PDF](https://arxiv.org/pdf/2608.18017v1)

**作者:** Lu Xu `[一作]` (Chongqing University), Jiaxing Shang `[通讯]` (Chongqing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 FlightLLM 方法，对 QAR 多变量时间序列航班安全事件（硬着陆）进行可解释性分类与因果解释。

**💡 创新点**

创新点在于将特征工程、语义离散化、统计专家提示、动态对比检索与结构化提示融合，使数值数据转化为 LLM 可理解的语义描述，实现在低样本下的可解释性预测。

**🔧 技术方法**

技术组合包括 TSFresh 与手工物理特征提取、语义离散化映射、CatBoost 统计专家、检索式 Few‑Shot 对比样本、Chain‑of‑Thought 结构化提示以及 GPT‑3.5/DeepSeek/GLM 等预训练大语言模型。

**📊 数据集**

使用 704 台 A320 实时 QAR 数据集（282 硬着陆、422 正常着陆），包含 32 传感器的多变量时间序列。

**📈 对比分析**

与 LSTM、SVM、RF、KNN、CNN、IMTCN、SDTAN 等传统与深度模型对比，FlightLLM 在准确率约 81.6%、精确率最高（≈85.7%）并提供可解释性，优于大多数基线。

**⚠️ 局限性**

局限性包括未对 LLM 进行领域微调、需为每类事件手动改写提示、语义离散化对边界样本召回略低，以及对预训练模型通用性的依赖。

---

## 476. Memory Tree Guided Key Frame Querying for Efficient 3D Question Answering

**arXiv ID:** 2608.18009 | [PDF](https://arxiv.org/pdf/2608.18009v1)

**作者:** Hsiang-Wei Huang `[一作]` (University of Washington), Cheng-Hao Kuo `[通讯]` (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于可实时构建的多层次树结构（MemTree）和LLM驱动的关键帧检索框架，以实现高效的3D问答。

**💡 创新点**

创新点在于将场景表示抽象为可序列化的树结构，让LLM直接推理空间与时间线索，既避免每次查询的视觉搜索，又提升对感知失败的鲁棒性。

**🔧 技术方法**

技术手段包括：利用6-DoF相机位姿分段构建位置节点；使用YOLO‑World+BoT‑SORT检测跟踪得到对象与检测节点；将树结构序列化为JSON供LLM（如GPT‑4o或Qwen3）推理空间/时间线索；再用VLM（LLaVA‑OneVision‑7B或GPT‑4o）完成最终问答；关键帧通过分数式选取。

**📊 数据集**

评估数据集：OpenEQA（约1,600问答），ScanQA（4,675问答）以及SQA3D（3,519问答）。

**📈 对比分析**

与统一采样、视觉搜索（Detector‑based FS）、Socratic、ConceptGraphs等方法对比，MemTree3D在OpenEQA上使GPT‑4o LLM‑Match提升17.4%，LLaVA‑OneVision‑7B提升5.8%；在ScanQA/SQA3D亦取得显著增益；与视觉搜索方法相比，运行时至少提升69.2%。

**⚠️ 局限性**

局限性：仍依赖物体检测的准确性，且在小场景或低视差环境下提升有限；尽管多轮查询改进明显，但在极大规模场景中可能仍需进一步压缩树结构或提升检索效率。

---

## 477. Cluster-Graph Edit Distance: Metric Proxies, Multiscale Embeddings, and Complexity

**arXiv ID:** 2608.17990 | [PDF](https://arxiv.org/pdf/2608.17990v1)

**作者:** JiYe Liu `[一作]` (Tianjin University), Wenjun Wang `[通讯]` (Tianjin University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了聚类图（cluster graph）之间的编辑距离，并给出了其计算难度、近似算法和欧几里得嵌入的理论上界；

**💡 创新点**

创新点在于把编辑距离转化为运输问题并提出两种显式嵌入（行能量向量与多尺度阶梯映射），以及证明在临界指数γ=1/4处可得到无条件的下界；

**🔧 技术方法**

采用的技术包括运输问题与线性规划、再排列不等式、谱（迹范数）变形、dyadic 分解、Whitney 复铺以及量化跳跃分析；

**📊 数据集**

使用的数据集为所有 n 元聚类图的离散集合（p(n) 个不同的块大小组合）；

**📈 对比分析**

与最优欧几里得嵌入比较时，所给出的多尺度映射的乘积拉伸比约为 O(n^{1/4}√log n)，比单一排序度量的 Θ(√n) 拉伸比大幅提升；

**⚠️ 局限性**

限制在于单一坐标（无权重）无法同时平衡尺度一致性与分散训练的两种失败模式，且对 0<γ<1/4 的性能仍未完全阐明。

---

## 478. GS-Voxel: Fitting-Free Structured Latents for Large-Scale 3DGS Generation

**arXiv ID:** 2608.17988 | [PDF](https://arxiv.org/pdf/2608.17988v1)

**作者:** Ming Qian `[一作]` (Amap, Alibaba), Baoquan Chen `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出GS‑Voxel，一种无拟合的稀疏体素化结构化潜在表示，直接从预优化的3D高斯云转换为可用于生成的大尺度空中场景

**💡 创新点**

创新点在于：1) 采用无逐场景拟合的离线体素化；2) 将几何与局部高斯属性分离编码为两阶段VAE；3) 在生成过程中使用多阶段流模型与重叠拼接推断，支持超大面积合成

**🔧 技术方法**

核心技术包括：稀疏体素化（GS‑Voxel）、两阶段分解VAE（Geometry VAE+Local Attribute VAE）、基于TRELLIS.2的图像条件流模型、重叠感知的分块推断

**📊 数据集**

训练数据来自多个大型实测空中3D高斯云场景，经过切块、坐标归一化、DBSCAN去噪、合成多视角渲染后生成约18K个样本

**📈 对比分析**

与现有方法对比，生成的图像在FID为28.0、KID为0.020，显示出可接受的生成质量；同时在重建上达到23.09 PSNR、0.62 SSIM、0.331 LPIPS

**⚠️ 局限性**

主要局限包括：仅适用于SH0视角空中场景；受限于训练场景规模与多样性；缺乏对真实传感器与跨域条件的鲁棒性评估；局部容量与分辨率固定，难以处理极密集或细薄结构

---

## 479. Against Political Polarization: A Unified Framework for Tracing Evolving Political Ideologies on Social Media

**arXiv ID:** 2608.17987 | [PDF](https://arxiv.org/pdf/2608.17987v1)

**作者:** Yijie Xu `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个统一框架，先用LLM风格迁移和无监督域适配检测推文中的政治意识形态，然后用时序图神经网络预测用户意识形态的强度与未来演化，并公开了两大规模数据集。

**💡 创新点**

创新点在于将离线LLM风格迁移与无监督域适配相结合，显著降低新闻与社交媒体文本差距；同时在时序图上实现政治意识形态的存在、强度和未来趋势三位一体预测；并首次公开覆盖数十万推文与数万用户的长时序数据。

**🔧 技术方法**

技术主要包括：基于BERT的政治意识形态检测网络（PIDN），使用文本风格迁移和MMD/KL域适配；时序图神经网络预测网络（PIPN），采用TGN、JODIE、APAN等TGNN；以及对比实验中的大模型（Qwen2.5）零/少shot推理。

**📊 数据集**

使用的数据集包括：AllSides 新闻偏见标签（约23万篇新闻），X（Twitter）77M推文（4,545名用户，16年跨度），以及 Truth Social 823k贴文（454k用户）。

**📈 对比分析**

通过与大模型零/少shot推理比较，PIDN在分类F1≈0.99、回归MAE≈0.004、推理时延≈4ms方面大幅领先；PIPN在Truth Social和Twitter上，TGN、JODIE、APAN等TGNN在链接预测和意识形态预测上均显著优于静态GCN/GAT，准确率接近99%，MSE约0.056。

**⚠️ 局限性**

局限性包括：仍采用单维左‑右轴模型，未覆盖多维议题或多平台跨域泛化；LLM风格迁移可能引入偏差；模型训练和推理仍需较大计算资源，且依赖平台API访问。

---

## 480. Planning Against Learning in Rank-1 Games

**arXiv ID:** 2608.18067 | [PDF](https://arxiv.org/pdf/2608.18067v1)

**作者:** William Overman `[一作]` `[通讯]` (Stanford University), William Overman (Stanford University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

论文研究了在 Rank‑One（A+B=uv^⊤）双矩阵游戏中，优化者如何针对学习者的复制者动态（Replicator Dynamics）进行策略规划，并证明该规划问题在一般情况下是 NP‑难的。

**💡 创新点**

创新点在于首次将 Rank‑One 结构与学习动态结合，揭示即使在可以多项式求解 Nash 均衡的游戏类中，规划对学习者的动态仍可能是不可多项式可解的；并给出从 Clique 问题的精细化归约，证明了常数逼近的 NP‑难度。

**🔧 技术方法**

主要技术包括：将优化者收益拆解为终点项与路径积分项；利用 Motzkin–Straus 定理将最大团问题转化为对齐于学习者对数轨迹的二次型；构造四列双曲边缘装置实现软最大化的近似；以及对软最大化端点项的精细界定。

**📊 数据集**

本文不使用任何实验数据集，全部工作基于理论证明与数学归约。

**📈 对比分析**

由于问题被证明为 NP‑难，论文未提出可运行算法，也未与现有方法做性能比较；相反，它通过硬件化证明展示了对策规划的不可行性。

**⚠️ 局限性**

主要局限在于：(1) 仅证明了 Rank‑One 结构下的规划 NP‑难，未给出更细化的可解子类；(2) 对学习者动作数为 3 或更少的情况仍未确定复杂度；(3) 对离散时间 MWU 或带衰减的学习动态的规划复杂度仍未知。

---

## 481. On the Fragility of Self-Improving Agents: Variance, Task Order, and Underspecification

**arXiv ID:** 2608.18066 | [PDF](https://arxiv.org/pdf/2608.18066v1)

**作者:** Qinyuan Ye `[一作]` (Salesforce Ai Research), Chien-Sheng Wu `[通讯]` (Salesforce Ai Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对两种内存驱动的自我提升方法（Agent Workflow Memory 与 ReasoningBank）进行重新评估，扩展评测维度至多跑实验和任务顺序随机化，并使用更强的基线模型与代理框架；

**💡 创新点**

揭示自我提升系统在多跑和随机任务顺序下的高方差与任务顺序敏感性，发现环境与任务未规范是导致易失效的主要原因，并通过加入评估标准、环境反馈与提示修正等额外信息来缓解问题；

**🔧 技术方法**

使用文本记忆池与检索式记忆构建技术，基于大型语言模型（GPT‑5‑mini、Claude‑3.5‑Sonnet、Gemini‑2.5‑Pro）以及多跑统计与方差分析；

**📊 数据集**

WebArena、VisualWebArena 与 SCUBA 三个 Web 浏览基准数据集；

**📈 对比分析**

与单跑无记忆基线比较，报告多跑标准差、best‑worst gap；默认顺序下自我提升略有提升，随机顺序时性能下降；加入额外信息后提升约2.9%，但仍低于无记忆基线；

**⚠️ 局限性**

仍存在未识别的未规范因素导致鲁棒性不足；额外信息提升有限；缺乏内存验证机制；实验仍以实验室基准为主，真实世界泛化尚未知晓。

---

## 482. The Polyglot's Dilemma: Conformance Testing a Dozen Specs in as Many Languages

**arXiv ID:** 2608.18039 | [PDF](https://arxiv.org/pdf/2608.18039v1)

**作者:** A. Jesse Jiryu Davis `[一作]` (MongoDB Research), Jeff Yemin `[通讯]` (MongoDB, Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一套统一的声明式测试框架（Unified Test Format，UTF），用于多语言 MongoDB 驱动的规范一致性测试，覆盖 API 行为、wire 协议以及错误处理等方面。

**💡 创新点**

创新点在于将分散的、语言特定的 DSTL 统一为一种可验证的 YAML/JSON 语法，并通过命令监控、故障注入与实体映射等机制，实现跨语言一致性验证，同时显著减少测试维护成本。

**🔧 技术方法**

技术手段包括：YAML/JSON 语法与 JSON Schema 验证；测试运行器（分别实现于 C#, Go, Java, Node.js, Python 等语言）；命令监控 API、故障注入（fail points）和实体映射（entity map）；以及 CI 自动化与 Git 子模块同步。

**📊 数据集**

数据集主要有两部分：一是 UTF 测试语料库（606 个文件、124,168 行 YAML/JSON），二是 CRUD 规范的 6836 条已解决工单，经过人工与 LLM 分类得到 125 条非符合性 bug 数据。

**📈 对比分析**

通过对比未采用 YAML 测试与已采用测试阶段的 CRUD 非符合性 bug 率，发现四个驱动的 bug 率下降 49–86%；此外，UTF 迁移后共删除 22,000+ 行测试代码，单个驱动（如 Java）削减 6,038 行，验证了方法的有效性与高效性。

**⚠️ 局限性**

局限性包括：UTF 不能覆盖所有规范（如 BSON 序列化、SDAM 状态机、连接字符串解析等仍需专用格式）；单一模式版本导致实现滞后；手工编写测试难以自动化；以及在某些驱动（如 Node.js）中即使通过测试仍出现大量细节不符问题。

---

## 483. Plug-and-Play Traffic Element Awareness for End-to-End Autonomous Driving

**arXiv ID:** 2608.18035 | [PDF](https://arxiv.org/pdf/2608.18035v1)

**作者:** Zongzheng Zhang `[一作]` (Tsinghua University), Hao Zhao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量级的交通要素感知框架，通过3D交通元素检测与拓扑编码，将交通灯、路标信息以插件式方式无缝注入多种端到端自动驾驶模型，并在多个基准上验证其有效性。

**💡 创新点**

核心创新在于：①统一的3D交通元素提取管线，将2D检测、单目深度估计与LiDAR几何融合；②在端到端模型中加入辅助3D交通元素监督；③将自闭合拓扑信息转换为语言嵌入，实现低成本的上下文注入；④在多范式（感知‑预测‑规划、VLM/VLA、回归、扩散、轨迹评分）下进行系统性评估，展示其泛化能力。

**🔧 技术方法**

采用YOLOv8做交通元素检测、UniDepthv2进行深度估计、LiDAR点云配准实现3D坐标；利用BERT对ego‑centric拓扑进行语言编码；在模型中加入焦点损失、独立预测头、最大池化等技巧；并使用轻量级TopoMLP进行拓扑预测。

**📊 数据集**

实验涵盖nuScenes、NAVSIM‑v1、NAVSIM‑v2、Bench2Drive四大基准，并通过OpenLane‑V2提供TE标注；此外使用SimScale仿真数据进行数据扩充与跨域训练。

**📈 对比分析**

与基线相比，模型在开环L2误差与碰撞率上平均降低2–3%，PDMS/EPDMS提升约10点；在闭环Bench2Drive上驾驶分数提升5%以上，保持低延迟；所有范式模型均获得显著提升，尤其在NAVSIM‑v2上实现新的SOTA。

**⚠️ 局限性**

主要局限在于对TE检测与深度估计质量高度依赖，感知误差会影响最终规划；实验主要聚焦城市道路，极端或稀有交通场景的鲁棒性待验证；在仅摄像头或其他传感器配置下的适配尚未充分评估。

---

## 484. Deep Academic Survey: Stateful Agentic Closed-Loop Paradigm for Academic Survey Automation

**arXiv ID:** 2608.18034 | [PDF](https://arxiv.org/pdf/2608.18034v1)

**作者:** Zhikai Xu `[一作]` (Zhejiang University), Jiangning Zhang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

未收到论文内容

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

## 485. Automated ACL Footprint Identification Using 3D Deep Learning

**arXiv ID:** 2608.18012 | [PDF](https://arxiv.org/pdf/2608.18012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 486. Initialization-Free Bundle Adjustment Revisited: A Controlled Experimental Study

**arXiv ID:** 2608.18028 | [PDF](https://arxiv.org/pdf/2608.18028v1)

**作者:** Simon Weber `[一作]` (University of Oxford), Ronald Clark `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了从图像观测直接恢复相机姿态和场景结构的初始化自由束调整（InitFree BA）方法，并构建了统一实验框架来评估完整的重建流程；

**💡 创新点**

发现了优化目标低并不等价于可行的欧氏重建，提出“优化–重建差距”概念，并指出初始化先验、观测密度、鲁棒化和度量升级稳定性是决定成功的关键因素；

**🔧 技术方法**

采用多种对象空间误差（OSE）变体（pOSE、rOSE、RpOSE、expOSE、pOSE+rot）与可变投影（VarPro）相结合，使用 Blender 生成可控数据集，利用自校准式度量升级和 Cauchy 鲁棒损失进行实验；

**📊 数据集**

在八个基于 Blender 的合成序列（Set1–8）中评估，其中 Set5 使用真实 COLMAP 重建作为对照，同时在 BAL 数据集上验证结果；

**📈 对比分析**

通过统一的评估指标（旋转/平移 AUC、Landmark RMSE）对不同 OSE 方案、初始化方式、鲁棒化和观测阈值进行系统比较；pOSE+rot 在大多数设置下表现最佳，expOSE 在无相对旋转约束的 OSE 方案中最优；鲁棒化可在灾难性失败时救场，但平均提升有限；

**⚠️ 局限性**

受限于几何先验依赖、稀疏观测导致的失败、优化–重建差距未彻底解决以及在真实数据上的易崩溃，缺乏通用且稳定的度量升级方法。

---

## 487. Traceable Trust for action-ready artificial intelligence in bioscience

**arXiv ID:** 2608.17997 | [PDF](https://arxiv.org/pdf/2608.17997v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 488. TabNSM: Neural Sparse Mixer for Tabular Regression

**arXiv ID:** 2608.18026 | [PDF](https://arxiv.org/pdf/2608.18026v1)

**作者:** Ali Eslamian `[一作]` (University of Kentucky), Qiang Cheng `[通讯]` (University of Kentucky)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种可扩展的表格回归框架 TabNSM，结合自适应稀疏交互模块、分阶段回归头、软分箱损失与难度感知采样。

**💡 创新点**

通过实例自适应稀疏注意力与特征-令牌混合实现可区分前景特征，并引入 GridLoss 与 RISE 提升连续目标训练和困难样本学习。

**🔧 技术方法**

稀疏注意力、特征-令牌混合 (FTM)、多阶段回归头、温度软化分箱损失、误差加权采样、混合精度训练等技术。

**📊 数据集**

在九个真实世界回归基准上评估，包括机器人、交通、公共健康、犯罪、房地产、空气质量、神经影像等多领域。

**📈 对比分析**

与 38 类基线（树模型、MLP、Transformer、检索、TFM 等）对比，TabNSM 在七个数据集上取得最低 RMSE，整体显著优于树模型与大多数深度学习模型。

**⚠️ 局限性**

需要 GPU 训练，内存/计算成本高于树模型，尚未实现压缩/蒸馏，对多目标或极端分布的适应仍待改进。

---

## 489. Revisiting WEASEL 2.0: Reproduction, Sensitivity, and an Adaptive Ensemble-Size Rule

**arXiv ID:** 2608.18021 | [PDF](https://arxiv.org/pdf/2608.18021v1)

**作者:** Cian Higgins `[一作]` (University College Dublin), Georgiana Ifrim `[通讯]` (University College Dublin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在本文中，作者对WEASEL 2.0进行了完整的重现与敏感性分析，并针对其超参数阈值提出了自适应的集成规模规则；

**💡 创新点**

创新点在于发现原始集成规模阈值对长序列数据过度预留，并提出基于序列长度与类别数的自适应阈值，显著减少内存与时间开销；

**🔧 技术方法**

主要技术包括字典式时序特征提取（滑动窗口+SFA）、随机超参数集成、RidgeClassifierCV、TF‑IDF与子线性TF试验、Wilcoxon检验及自适应阈值设计；

**📊 数据集**

使用了UCR数据集库中的114个固定长度时间序列分类数据集，并在22个多样化子集上进行敏感性实验；

**📈 对比分析**

与MultiRocket、MiniRocket、ROCKET、R‑DST、Hydra、WEASEL 1.0、cBOSS等方法同平台比较，WEASEL 2.0保持最高或相近准确率，改进规则在保持准确率的前提下平均节省约395 MB内存、约4 s训练时间；

**⚠️ 局限性**

局限性包括自适应规则仅基于序列长度和类别数，未考虑训练样本量或多变量情况；未实现基于交叉验证的动态集成规模选择；适用范围局限于固定长度单变量数据集。

---

## 490. Composing Flow-Matching Energies with Known Physics: Generation, OOD Detection, and Inversion on PDE Fields

**arXiv ID:** 2608.18004 | [PDF](https://arxiv.org/pdf/2608.18004v1)

**作者:** Yixuan Sun `[一作]` (Argonne National Laboratory), Sandeep Madireddy `[通讯]` (Argonne National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过在流匹配中引入标量势能参数化，获得时间相关的显式能量函数，从而实现对 PDE 字段的无条件生成、OOD 检测与逆问题求解。

**💡 创新点**

创新点在于利用势能产生的速度场直接读出能量，并提出 EnergyPC 预测-校正框架，将能量作为 MCMC 校正器、OOD 分数与逆问题后验能量的统一手段。

**🔧 技术方法**

技术上结合了流匹配、势能参数化、能量基预测-校正（EnergyPC）、MALA/ULA 校正器以及将 PDE 残差与能量叠加的总能量。

**📊 数据集**

实验使用了 Poisson、Helmholtz 与 Burgers 三个 PDE 字段数据集。

**📈 对比分析**

与传统的 ODE 流采样、标准 PC、DiffusionPDE、FlowDPS 等方法比较，EnergyPC 在切片 W2、谱距、MMSE/SMSE 与 PDE 残差等指标上均优于或持平，且在 OOD AUROC 与逆问题 L2 误差上取得显著提升。

**⚠️ 局限性**

主要限制是势能参数化带来的额外计算开销，以及在终端时间 t=1 时能量出现奇异，导致无法直接获得真实终端能量。

---

## 491. Judge, Retrieve, or Abstain: Uncertainty-Guarded LLM Judging with Provable Risk Guarantees

**arXiv ID:** 2608.17994 | [PDF](https://arxiv.org/pdf/2608.17994v1)

**作者:** Sher Badshah `[一作]` (Dalhousie University), Hassan Sajjad `[通讯]` (Dalhousie University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种风险控制的 LLM‑as‑a‑judge 框架，通过两阈值路由模式在保持有限样本 FDR 控制的同时利用检索提升覆盖率。

**💡 创新点**

①将 Clopper–Pearson 的 FDR 保证推广到两阈值路由；②在 parametric 模式不确定时触发检索增强，改进覆盖率而不牺牲错误率。

**🔧 技术方法**

使用 LLM 判定器与预测熵不确定度，结合 web 检索增强；采用 Clopper–Pearson 上界对两阈值联合校准；在四个开放域 QA 评测数据集上实验。

**📊 数据集**

TriviaQA、Natural Questions、HotpotQA、PopQA。

**📈 对比分析**

与单模式（仅 parametric）对比，保留 FDR 控制的同时，覆盖率显著提升；在不同风险阈值下，覆盖率从 20% 递增至 100%，且实验中所有配置的 FDR 均不超过目标 α。

**⚠️ 局限性**

需要检索结果的稳定性假设，检索成本与时效性影响；仅适用于二元真/假评估，未覆盖分级或主观任务；依赖预测熵与阈值校准；需要标注的 calibration 集。

---

## 492. What Does It Mean and Why Should I Bother? Motivating Students to Write Better Commit Messages

**arXiv ID:** 2608.17993 | [PDF](https://arxiv.org/pdf/2608.17993v1)

**作者:** Gergő Balogh `[一作]` (University of Szeged), Ádám Zoltán Végh `[通讯]` (AENSys Ltd)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过混合方法案例研究，分析学生和工业项目的提交信息，发现沟通问题频繁出现，并设计了一款轻量化、角色扮演式的教育游戏（What Do You Mean?，WDYM），通过游戏让参与者直观体验并反思提交信息的沟通缺陷；随后通过问卷评估游戏对参与者意识的影响。

**💡 创新点**

创新点在于将游戏化与角色扮演相结合，构建一种可视化、可量化的提交信息沟通评估工具；将已有的提交信息质量分类迁移到教育场景，并通过游戏实时反馈与反思，提供了一种新颖的教学干预方法。

**🔧 技术方法**

采用基于what-makes-a-good-commit-message的多标签质量分类体系，利用专家手工标注和多投票多数法获取提交标签；利用纸笔角色扮演游戏捕捉沟通失效的实时表现；通过问卷（预/后/随访）收集自评意识数据。

**📊 数据集**

数据集包含19个开源/闭源项目（Java/JavaScript，kLOC 0.6–13 869，提交数 50–84 707），共抽取104条学生提交和33条专业提交用于标签化；游戏使用的提交信息同样来自上述项目。

**📈 对比分析**

对比方法：将学生与专业提交在各质量标签上的分布进行统计（箱线图、百分比），并通过问卷得分评估游戏前后意识变化。结果显示两组均存在明显的“Missing What/Why”缺陷；游戏后参与者对沟通失效的认知提升，但对写作行为的改变未见显著统计效应。

**⚠️ 局限性**

局限性包括：研究为本地案例，样本量有限（学生调查响应率低）；未设置对照组，难以评估游戏独立效应；标签一致性未进行正式统计；游戏主要提升认知而非实际写作改进；部分数据因保密原因未公开，限制复现。

---

## 493. The IOL-AI Challenge: An Open Challenge towards Advancing Linguistic Reasoning

**arXiv ID:** 2608.18011 | [PDF](https://arxiv.org/pdf/2608.18011v1)

**作者:** Eduardo Sánchez `[一作]` (University College London & Meta), Julia Kreutzer `[通讯]` (Cohere Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过举办 IOL‑AI 挑战赛，首次在真正未公开的语言学奥林匹克（IOL）问题上评估大语言模型（LLM）的推理与结构推断能力，并对模型在推理、答案格式、解释质量等方面进行系统分析。

**💡 创新点**

创新点：①构建了全新、未泄露的语言学推理数据集；②引入专家评审（IOL 庭审）与自动指标双重评估，揭示自动指标对弱模型的夸大与强模型的低估；③通过技术实验（自一致投票、重试、温度采样等）展示资源受限下的推理优化方法；④对模型知识来源进行探测，发现模型对语言元信息的利用并不决定最终性能。

**🔧 技术方法**

技术方法：多轮链式思考与自一致投票、温度重试、低位量化（AWQ/NF4）推理、解码策略调优、输出格式修正、基于 GPT、Claude 等大型模型的推理、Mid‑size 模型集成（Best‑of‑N），以及对推理过程的自动和人工评估。

**📊 数据集**

数据集：IOL 2026 Individual Contest 5 题（共 14 子任务），涵盖 5 种极低资源语言（Yup'ik、Yélî Dnye、Iquito、Sakurabiat、Komnzo）。数据为未公开、未泄漏的完整问题文本、上下文与答案。

**📈 对比分析**

评估方法：使用 ChrF、Exact Match（EM）及其几何平均 GM 进行自动评估；同时邀请 IOL 庭审人员进行人类评审，给出分数并检查解释质量。结果显示：开源受限模型 GM ≤ 15，无法达到荣誉提名；mid‑size 集成 Oracle GM ≈ 33；最强的专有模型（Claude Opus 4.8、Gemini 3.6‑Flash）GM ≈ 75–80，分别对应金、银奖级别。自动与人类评分在排名上高度一致，但人类评分更能体现模型的解释质量。

**⚠️ 局限性**

局限性：①答案格式不一致导致自动评估需后处理；②测试集样本量小、偏深，难以衡量模型泛化；③Oracle 集成不代表可部署系统；④不同模型使用的生成预算不统一；⑤挑战赛时间短，可能限制提交改进；⑥公共仓库提交可能导致复制与知识共享不透明。

---

## 494. The concentration game: Bayesian updating, regret, and information

**arXiv ID:** 2608.18061 | [PDF](https://arxiv.org/pdf/2608.18061v1)

**作者:** Akshay Balsubramani `[一作]` `[通讯]`, Akshay Balsubramani

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一个两人零和重复博弈的框架，结合了贝叶斯更新和指数权重悔恨的概念，提供了一种比较类变分形式，适用于广泛的集中现象。

**💡 创新点**

创新点在于将贝叶斯更新和指数权重悔恨视为同一身份的两个方面，统一了多个研究领域的不同形式，形成了一个集中博弈的价值身份。

**🔧 技术方法**

使用了贝叶斯更新、指数权重、Gibbs权重等技术，构建了一个信息论的账本来记录自我博弈的过程。

**📊 数据集**

使用了有限动作集上的严格正先验和相对熵预算的比较器，具体数据集未明确提及。

**📈 对比分析**

通过比较不同的策略和预算，展示了悔恨的分解成三部分：每轮的信息损失、重温漂移和相对熵传输，性能表现优于传统的标准悔恨界限。

**⚠️ 局限性**

局限性在于该框架主要集中于熵几何，未涵盖非熵情况的相关问题。

---

## 495. HLSR: Hybrid Live Forecast Selective Dynamic Vehicle Rerouting for Real-Time Congestion Avoidance

**arXiv ID:** 2608.18056 | [PDF](https://arxiv.org/pdf/2608.18056v1)

**作者:** Xiao Wang `[一作]` (National Tsing Hua University), Hui Nien Hung `[通讯]` (National Chiao Tung University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在城市交通网络中，提出并实现了一种选择性实时车辆重路由框架（HLSR），该框架通过双阈值拥堵检测、上游跳数和逼近车辆扩展来限定干预范围，并融合实时测速与短期预测以计算多目标成本，最终为受拥堵影响的车辆生成最优路径。

**💡 创新点**

创新点包括：①双阈值占用-速度拥堵检测与校准上游跳数+逼近车辆扩展，实现有限干预；②基于实时与预测速度的时间衰减融合，支持多目标成本分配；③引入司机行为校准因子并对预测模型进行路径排序微调；④采用路由器个性化速度预测与k短路径生成，显著提升重路由质量。

**🔧 技术方法**

技术手段涵盖：基于图的LSTAN‑GERPE时空预测网络、车道级队列延迟模型、Yen k‑Shortest‑Path、时间衰减混合速度融合、归一化多目标优化（行程时间、路径长度、相似度、负载平衡），以及司机行为校准因子。

**📊 数据集**

使用的数据集：Tainan城市道路网络（OpenStreetMap）+基于SUMO的仿真生成的交通需求（8000/16000/20000辆），并配合循环检测器提供的速度与占用观测。

**📈 对比分析**

通过与NRR、ReFOCUS+、Du‑GAQ、HLSR‑Rank、HLSR‑LIVE、CAIE‑TT‑Scoped（全局）等方法对比，HLSR在所有负载下平均行程时间最低（380.6/895.7/971.7 s），且重路由次数明显低于全局方法，表现出显著的性能提升。

**⚠️ 局限性**

局限性包括：仅在已知路网与检测器的仿真环境验证；对极端拥堵或大规模网络的可扩展性尚未充分评估；预测模型受数据稀疏和训练限制；司机个性化校准需要大量历史轨迹数据。

---

## 496. StagedWorkspace: A Versioned Workspace for Knowledge-Work Agents

**arXiv ID:** 2608.18050 | [PDF](https://arxiv.org/pdf/2608.18050v1)

**作者:** Yining Hua `[一作]` (Harvard University), Levi Lian `[通讯]` (Raycaster AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了“StagedWorkspace”工作空间状态协议，为知识工作型 AI 代理提供同步的原始文件视图与解析缓存、版本化的审计差异，并通过哈希校验实现视图一致性。

**💡 创新点**

创新点在于将搜索、原始操作、审计和提交统一到同一版本的工作空间，首次在多格式知识工作场景中实现双视图（原始+解析）同步与可见差异，提升模型在 OfficeQA 与 APEX 评测中的表现。

**🔧 技术方法**

使用技术包括：ReAct 交互式工具调用循环、哈希键驱动的缓存失效与异步重解析、变更日志 (journal) 记录、基于文件内容哈希的解析索引、格式化差异渲染（行/单元格/幻灯片级别）。

**📊 数据集**

实验数据集：OfficeQA Pro（约 697 篇美国财政部公报 PDF，104 题纯文档 + 29 题需要实时网络）和 APEX-Agents（33 个专业工作场景，约 166 文件/任务，包含代码、表格、幻灯片等多格式）。

**📈 对比分析**

与公开榜单直接对比，Dual 视图在 OfficeQA Pass@1 上提升 8.3–12.1 分（GPT‑5.4 64.8% vs 52.7%），在 APEX 平均 Rubric 得分提升 4.7–9.2 分；在 57 个文件编辑子任务中，Diff 可见性提升 2.5–8.5 分；相对仅原始或仅解析视图，Dual 视图在所有模型上均显著提升，且大多数提升伴随成本或时延不升高。

**⚠️ 局限性**

局限性：仅评估最终答案/交付物，未捕获中间检索/读取/编辑过程的错误；公开对比基准不完全一致（尝试预算、检索后端、解析器不同）；状态同步虽显著但未解决 188/452 的高层规划、领域推理与规则遵循错误；需要更细粒度的轨迹与中间状态验证。

---

## 497. An Approximate Cauchy-Schwarz Inequality and Improved Bounds for Sherali-Adams Refutation of Semirandom CSPs

**arXiv ID:** 2608.18048 | [PDF](https://arxiv.org/pdf/2608.18048v1)

**作者:** Pravesh K. Kothari `[一作]` (Princeton University), Andrew D. Lin `[通讯]` (Princeton University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种近似的Cauchy-Schwarz不等式，并证明其适用于Sherali-Adams线性规划层次的解（被解释为“伪分布”）。

**💡 创新点**

创新点在于提出了一个近似的Cauchy-Schwarz不等式，解决了O'Donnell和Schramm工作中未解的问题，并且该不等式在局部正半定性条件下成立。

**🔧 技术方法**

使用了简单的采样论证来证明近似Cauchy-Schwarz不等式，并且依赖于Sherali-Adams伪分布的局部正半定性。

**📊 数据集**

论文中没有具体提到使用的数据集，但提到的应用包括随机和半随机的k-XOR实例。

**📈 对比分析**

通过与现有的基于谱的方法进行比较，证明了Sherali-Adams方法在处理随机k-XOR实例时的有效性，尤其是在奇数k的情况下，性能得到了显著提升。

**⚠️ 局限性**

限制在于近似Cauchy-Schwarz不等式的加性误差是必要的，并且在某些情况下，Sherali-Adams伪分布可能无法满足全局正半定性条件。

---

## 498. Policy-Invariant Reward Shaping from LLM Feedback: A Framework for Hybrid RL Agents

**arXiv ID:** 2608.18008 | [PDF](https://arxiv.org/pdf/2608.18008v1)

**作者:** Christophe D. Hounwanou `[一作]` (African Institute for Mathematical Sciences), Yaé U. Gaba `[通讯]` (AI Research and Innovation Nexus for Africa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

结合LLM规划与RL控制，构造Goal‑Augmented MDP并证明以LLM进度分数为潜在函数的奖励塑形不会改变最优策略集合；

**💡 创新点**

创新点在于将LLM得分显式作为潜在函数，利用Ng等人的潜在塑形定理提供更强的policy invariance保证，并针对LLM错误提供理论上不影响最优策略的证明；

**🔧 技术方法**

使用潜在奖励塑形（Ng et al.）、GA‑MDP框架、PPO 强化学习、LLM规划器Qwen‑2.5:14b、以及潜在函数Φ进行实现；

**📊 数据集**

实验数据集主要包括20个MiniGrid任务用于规划质量评估，以及MiniGrid‑DoorKey‑6x6用于管线验证；

**📈 对比分析**

与纯PPO基线相比，管线验证在极低样本（3×10⁴步）下未显示明显优势；但已验证框架能够完整运行、LLM调用量可控；后续计划进行更大规模对比；

**⚠️ 局限性**

局限包括需GPU+云LLM进行大规模实验、潜在函数必须有界、子目标完成词汇匹配不当会导致计划停滞、潜在塑形虽保证最优策略但可能放慢学习速率。

---

## 499. Multivalued Consensus: General Adversaries Require More Communication

**arXiv ID:** 2608.17998 | [PDF](https://arxiv.org/pdf/2608.17998v1)

**作者:** Mose Mizrahi `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

针对一般的^d- adversary，研究了多方一致性任务（交互一致、拜占庭一致/广播、可靠广播、核心集合一致）在有长输入情况下的通信复杂度下界。

**💡 创新点**

证明了相较于阈值 adversary，^d- adversary 必须多出约 n^{1/d} 倍的通信量；构造了基于有限射影几何的 adversary 结构；给出非终止与终止协议的区别，并提供可终止协议实现。

**🔧 技术方法**

利用有限射影几何构造 adversary 结构、信息熵与通信计数论证、动态 erasure‑coding 请求模式、Merkle 树证明等技术。

**📊 数据集**

无数据集，全部为理论证明与协议构造。

**📈 对比分析**

与现有阈值 adversary 下的已知下界（Ω(Ln)）相比，提出的下界更高，证明其紧性（特别是异步场景）并给出匹配的终止协议，通信复杂度为 O(L n^{1+1/d}+n^2 log n)。

**⚠️ 局限性**

仅适用于错误免费或终止的协议；对非终止异步协议下的通信成本尚未完全给出；阈值之外的具体常数与实现细节仍未解决；对于多方非对称角色的平衡通信问题未探讨。

---

## 500. aDSL: Agentic 3D Creation via Joint Agent-Program Design

**arXiv ID:** 2608.17975 | [PDF](https://arxiv.org/pdf/2608.17975v1)

**作者:** Rui-Huan Wang `[一作]` (Peking University), Peng-Shuai Wang `[通讯]` (Peking University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个训练无关的多智能体系统，利用 agent-centric DSL (aDSL) 对3D内容进行程序化生成与迭代改进。

**💡 创新点**

创新点在于将 DSL 与规划‑编码‑评估循环联合设计，提供可表达、可组合、带空间推理的程序接口，并通过共享语义结构实现生成、验证、修复的闭环。

**🔧 技术方法**

使用了基于 LLM 的多角色代理（Planner、Coder、Critic、Debugger）、aDSL（Python 嵌入的构造实体几何与布尔运算），以及 Plan‑Execute‑Critic 循环、自动检查与修复。

**📊 数据集**

评估使用了 ShapeNet、ABO、Objaverse（文本生成）和 Toys4K（图像生成）等公开数据集。

**📈 对比分析**

与代码生成基线（BlenderMCP、BlenderLLM、LL3M、Scene Language、ShapeCraft）以及字段/网格生成基线（MVDream、LN3Diff、Trellis、Direct3D-s2、Llama‑Mesh）对比，在 CLIP、VQA、FID 等指标上取得了最高或接近最高的得分，并实现了 100% 执行成功率。

**⚠️ 局限性**

局限在于 DSL 表达力有限，难以生成复杂几何和材质；Critic 依赖二维渲染，可能受视角歧义；系统高度依赖商业 LLM，缺乏开源可复现性。

---

## 501. Recirculation

**arXiv ID:** 2608.17981 | [PDF](https://arxiv.org/pdf/2608.17981v1)

**作者:** Michael C. Mozer `[一作]` (Google DeepMind), Rosanne Liu `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在推理阶段给预训练大型语言模型引入一种名为recirculation的递归机制，利用深层激活泄漏到浅层来实现状态追踪，提升模型在生成、推理和指令跟随等任务的表现。

**💡 创新点**

创新点在于：①完全训练‑free、仅在推理时修改架构；②通过残差对齐将深层上下文信息直接注入浅层，显著提升状态追踪能力；③提出自适应recirculation，学习混合系数α、β，使得在不改动模型权重的前提下进一步提升性能。

**🔧 技术方法**

技术方法包括：深层到浅层的激活混合（α·f+β·t），对混合向量做归一化；对不同层组合（source, destination）进行超参数搜索；构造自适应模块（MLP输出token‑级α/β向量）；以及在多种任务上进行无监督微调与指标对比。

**📊 数据集**

实验使用的语料包括 arXiv、PG‑19、C4、big_patent、billsum、booksum、gov_report、newsroom、pubmed 等语言建模数据集；下游任务包括指令跟随、Racing Thoughts、MMLU、ARC Easy/Challenge、PiQA、BoolQ、WinoGrande、HellaSwag、Lambada 以及 GSM8k 等推理与多轮问答数据集。

**📈 对比分析**

与基线 Gemma3 1B/4B/12B 以及其他模型族（Ministral3、Pythia、Qwen3、Phi2）对比；在 12B 模型上 perplexity 可降至 25% 左右，Gemma3 4B/12B 的 GSM8k 通过 pass@1/pass@128 的误差率下降分别约 8.8%/20.9%；在多数单词推理任务上准确率提升 1–5%；自适应recirculation 进一步使 perplexity 降低约 23% 并在 GSM8k 上获得更大提升。性能提升几乎不增加推理延迟，但在前缀填充阶段需要串行处理。

**⚠️ 局限性**

局限性包括：①超参数（source, destination, α, β）对任务/模型敏感，需要手工搜索或自适应学习；②对模型家族的依赖性尚未完全验证，Gemma 系列效果最突出；③归一化方案对结果影响显著，未给出通用准则；④前缀填充阶段需逐词递归，可能在大上下文下耗时；⑤目前只实现单次递归、单路径、单层混合；多迭代、多路径、块级递归等扩展未实验；⑥自适应recirculation 仍以少量数据训练 MLP，扩展到更大、不同任务的泛化性待进一步研究。

---

## 502. Hydra-0: Action Flow for Generalist World Modeling and Control

**arXiv ID:** 2608.18077 | [PDF](https://arxiv.org/pdf/2608.18077v1)

**作者:** Hongyu Li `[一作]` (NVIDIA), Yan Chang `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了可在多机器人体型和视频生成骨干之间共享的“动作流”视觉控制接口，既可用于前向动力学预测，也可逆向生成可执行动作，并在开环策略评估和真实机器人控制中验证其有效性。

**💡 创新点**

创新点在于：①将可执行机器人指令映射为图像平面轨迹（动作流），使不同体型可共享同一视觉条件；②在同一接口实现前向预测与逆向控制；③通过多体型中训练与轨迹采样/聚合技术显著提升跨体型泛化；④利用少步蒸馏与低秩适配提升推理速度。

**🔧 技术方法**

核心技术包括：轨迹采样与可视化条件化的 Diffusion 视频模型（Cosmos 2.5、Wan2.2 I2V-A14B / TI2V-5B），光学流/AllTracker 跟踪，动作流的 Gaussian 传播与可视化条件注入，低秩适配 LoRA，逆向动作读出头，稀疏轨迹的可见性处理，自动回归与少步蒸馏，开环策略评估与真实机器人执行。

**📊 数据集**

使用了多体型训练语料：DROID（单臂）、ABC-130k（双臂）、MolmoAct2（双臂）、EgoDex（人类手）、Deform360（手持软体物体）、XVLA-Soft-Fold、H1-Fold-Clothes 等，覆盖软体物体交互；以及 Interactive World Simulator (IWS) 任务用于数据效率评估。

**📈 对比分析**

通过与原始相对 6D 端执行动作的基线模型（Cosmos 2.5、Wan2.2）在 PSNR、SSIM、LPIPS、FID、FVD、抓取/物体端点误差等指标上进行比较；实验表明动作流模型在大多数指标上均显著优于基线（如 90% 以上的机器人运动误差下降、60% 以上的物体运动误差下降）。在多体型中训练后，任务特定数据需求下降至 20% 以内即可达到接近全量训练的性能；在 RoboLab 开环评估中，生成成功率与参考成功率的 Pearson 相关系数为 0.96，证明了开环评估的可靠性；真实机器人实验验证了从人类示范的对象流生成可执行动作的可行性。

**⚠️ 局限性**

局限性包括：①抓取精度受限，易出现厘米级误差，难以区分抓取成功与失败；②缺乏深度、触觉等额外感知，导致对接触状态的模糊推断；③手腕相机实验仅为小规模演示，未系统评估；④目前仅验证了开环策略评估，闭环评估和更复杂的动态场景（如大尺度相机运动、移动机器人）尚未充分评估。

---

## 503. From Corpora to Co-Evolving Capabilities: Capability-Centric Data Design for Generalist Image Generation

**arXiv ID:** 2608.18076 | [PDF](https://arxiv.org/pdf/2608.18076v1)

**作者:** Xingjian Wang `[一作]` (Alibaba Group), Mengting Chen `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了面向通用图像生成与编辑的能力驱动数据基础设施，包含三大专用数据引擎和能力对齐课程计划，并从零训练了MM‑DiT模型。

**💡 创新点**

将数据构建按生成能力拆分为可互操作的引擎，并配合能力依赖式多阶段课程；使用跨任务共通的多粒度字幕桥接实现概念迁移，形成动态评估驱动的自适应数据循环。

**🔧 技术方法**

采用大规模多模态扩散模型（MM‑DiT）、VLM驱动的字幕与编辑指令生成、知识图谱检索、层级课程调度、持续训练与监督微调，以及能力缺口驱动的检索与重采样等技术。

**📊 数据集**

构建了约4.4亿条文本‑图像对、1.2亿编辑对以及2700万图像‑实体对，来源于LAION、COYO、MMC4等公开图像‑文本池以及自建知识图谱。

**📈 对比分析**

在CPI‑Bench等编辑基准上，3B/6B模型分别在CPI‑General和Practical子集均获得约3.95/3.96的平均分，显示在多样化文本生成与编辑任务中实现了高质量表现。

**⚠️ 局限性**

仍受限于数据质量不均、模型对极端或稀有场景的泛化不足，且课程调度与评估流程复杂，难以在多任务大规模环境中实时调整。

---

## 504. Language Has Two Parameters: Narrative-Induced Semantic Plasticity and Phase-Sensitive Interpretation

**arXiv ID:** 2608.18041 | [PDF](https://arxiv.org/pdf/2608.18041v1)

**作者:** Hollis Robbins `[一作]` `[通讯]` (University of Utah), Hollis Robbins (University of Utah)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出语言含有幅度和相位两个参数，说明语义组合与历史经历如何共同决定意义解释。

**💡 创新点**

创新点在于把相位作为第二参数引入语言模型，并将其与量子概率相联系，强调历史依赖的相位关系。

**🔧 技术方法**

使用量子概率框架、密度矩阵表示、以及现有 transformer 结构的分析。

**📊 数据集**

基于大型文本语料库（如 CommonCrawl、维基百科）以及小说、电影脚本等多模态文本，但未使用显式的个人经历标注。

**📈 对比分析**

与传统仅利用幅度参数的模型对比，提出六项可检验的预测，尚未给出定量性能指标。

**⚠️ 局限性**

局限在于缺乏实证验证、基底选择问题、未提供具体的训练方法或数据集标注历史，理论仍待实验支持。

---

## 505. Optimize Your Sampling: Tuned Diffusion Sampling with Bayesian Optimization

**arXiv ID:** 2608.18040 | [PDF](https://arxiv.org/pdf/2608.18040v1)

**作者:** Travis Zhang `[一作]` (Cornell University), Kilian Q. Weinberger `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了OYS框架，利用贝叶斯优化直接调节扩散模型的采样时间步长，以提升低步数下的生成质量；

**💡 创新点**

创新点在于不依赖理论推导的近似指标，而是直接优化目标评估指标（如人类偏好得分），允许全局重新分配步长并显著提升低步数性能；

**🔧 技术方法**

采用贝叶斯优化、Gaussian过程回归、log‑SNR时间参数化、以及多种质量评估（HPS、FID、LPIPS、PSNR）等技术；

**📊 数据集**

使用的主要数据集包括COCO Captions、DiffusionDB、SDXL、Inverse HED/Depth/Segmentation数据集以及ImageNet‑512；

**📈 对比分析**

与默认采样计划和Align Your Steps (AYS)比较，OYS在文本生成、修补、逆任务等多种任务上均获得更高的HPS赢率（约68–70%）和更低的FID/更高的PSNR/LPIPS，且5步OYS可保留90%+原始50步质量；

**⚠️ 局限性**

局限性包括：调参仍需生成大量图像（数十万次），仅在低步数下显著；对不同模型/任务需单独调优；可能对调参集过拟合，未必在所有应用场景完全通用。

---

## 506. Dual Co-Train: Cross-Dataset Ultrasound Tongue Segmentation Under Extreme Data Scarcity

**arXiv ID:** 2608.17983 | [PDF](https://arxiv.org/pdf/2608.17983v1)

**作者:** Alisher Myrgyyassov `[一作]` (Hong Kong Polytechnic University), Yongping Zheng `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

该工作提出了一种源自由的跨数据集超声舌形分割框架，结合伪标签自训练、轮廓质量控制与基于分割的条件GAN，能够在仅有少量源标签且目标域完全无标签的条件下实现鲁棒的分割。

**💡 创新点**

创新点在于将伪标签筛选、轮廓结构约束与同步更新的条件GAN合成样本闭环相结合，形成轻量级可实时的自适应流程，并显著提升低标注场景下的跨域泛化能力。

**🔧 技术方法**

技术手段包括UltraUNet轻量级骨干、EMA教师-学生一致性训练、基于轮廓规则的质量筛选模块、Pix2Pix式条件GAN与感知损失，以及多分支混合训练（干净伪标签、噪声伪标签一致性、GAN合成样本）。

**📊 数据集**

实验使用八个不同采集条件的超声舌影像数据集（MTID、CTID、UXTD、UXSSD、UPX、UX2020、Cleft、TaL1），每个数据集包含200张标注帧，构成12个源-目标转移对。

**📈 对比分析**

与五种源自由基线（EMA、FSM、SHOT、UPL、AIF）在12个转移对上比较，平均Dice 0.760、MSD 2.412，显著优于所有基线，并在多数转移中甚至达到或超过人类标注误差水平。

**⚠️ 局限性**

局限性包括基线实现难度高（不同骨干、损失需重调）、结果受标注风格影响、对极端噪声或强域移位的适应仍有限，以及对GAN生成样本质量和稳定性的依赖。

---

## 507. LinCa: Accelerating Diffusion Models via Learnable Decomposed Feature Caching

**arXiv ID:** 2608.17973 | [PDF](https://arxiv.org/pdf/2608.17973v1)

**作者:** Jinshan Liu `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 LinCa 框架，通过可学习可逆网络将扩散模型的中间特征分解成不同连续性子空间，并为每个子空间采用匹配的预测阶数，从而在保持质量的前提下加速推理。

**💡 创新点**

创新点在于：① 利用可逆网络实现无信息损失的特征分解；② 为不同连续性特征采用多阶（0阶、1阶、2阶）Hermite 插值预测；③ 对不同时间段分别训练预测器，以适配不同模型和阶段的异质特征动力学。

**🔧 技术方法**

采用可逆网络（可逆卷积+耦合层）、多阶多项式/Hermite 预测、特征缓存技术，以及分段训练与数据驱动微调，结合 Diffusion Transformer 模型。

**📊 数据集**

在 FLUX、Qwen-Image、HunyuanVideo 等大型扩散模型上进行评测，使用 DrawBench、VBench、GEdit-Bench、ImageReward、CLIPScore、PSNR、SSIM、LPIPS 等指标。

**📈 对比分析**

与 FORA、Δ-DiT、TaylorSeer、FoCa、HyCa 等现有方法对比，LinCa 在 5–7× 的加速比下实现近乎无损质量，且在蒸馏或量化模型上进一步提升速度与质量，显著优于无训练或训练自由的缓存方法。

**⚠️ 局限性**

局限性包括：需要少量预生成特征并对每个模型/时间段单独训练；在极高加速比下仍可能出现轻微细节失真；可逆网络虽无信息损失但会带来额外的计算开销。

---

