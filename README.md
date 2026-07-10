# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-09 | 今日论文总数: 786

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models

**arXiv ID:** 2607.03751 | [PDF](https://arxiv.org/pdf/2607.03751v1)

**作者:** Xinyi Xie `[一作]` (Nanjing University), Pichao Wang `[通讯]` (Nvidia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出SVA（Search, Value, Act）框架，利用蒙特卡罗树搜索（MCTS）在模拟器中挖掘冻结的Vision‑Language‑Action（VLA）模型的高质量动作分布，并将搜索得到的轨迹回报蒸馏为轻量级Q‑value模型，随后在推理时通过“best‑of‑N”候选生成+Q‑评估的方式进行动作选择，从而提升VLA的泛化表现。

**💡 创新点**

创新点在于：①把VLA失败归因于评估瓶颈而非仅生成；②通过MCTS对冻结策略进行全局搜索，获取长时效回报；③将搜索知识压缩为可在部署时无模拟器、低延迟的Q网络；④提供可调的推理级别（N）实现测试时可扩展性，且不需对巨型VLA backbone做参数更新。

**🔧 技术方法**

核心技术包括：蒙特卡罗树搜索（Puct 选择、展开、回滚）；基于轻量级VLM（如Qwen3.5-0.8B）+LoRA的Q‑value预测网络；多头MLP价值头与不确定性正则化；最佳候选动作选择公式；以及在多任务多模态设置下的跨任务蒸馏与部署。

**📊 数据集**

使用的数据集与环境主要有：EmbodiedBench（EB‑Habitat、EB‑Navigation）、SimplerEnv（WidowX机器人平台）、RoboTwin 2.0；诊断实验还使用Simlarer、Libero、RoboTwin。

**📈 对比分析**

与基线（GPT‑4o、Qwen3.5-4B/9B/27B、Gemma‑4‑E4B‑it、OpenVLA、π_0/π_0.5、π_0+RoboMonkey）对比，SVA在EB‑Habitat平均提升约+15%，EB‑Navigation+13%，SimlerEnv/ RoboTwin 上分别提升 7–9%；更显著的是9B模型+SVA在性能上超过27B模型7点且推理延迟降低27%。实验表明测试时扩展（增加N）能带来更高成功率且延迟增幅有限，显示SVA在成本/性能方面优于直接扩大模型规模。

**⚠️ 局限性**

主要局限包括：①需要可重置的仿真器来进行MCTS；②搜索与价值学习分离可能导致知识转移不完整；③目前仅在模拟环境验证，缺乏真实机器人实验；④在连续动作或极大搜索空间下MCTS的计算成本仍高；⑤依赖冻结的VLA backbone，若基础模型本身欠佳，评估提升有限。

---

## 2. Don't Blame the Large Language Model: How Scaffolding Evolution Shapes Coding Agent Quality

**arXiv ID:** 2607.03691 | [PDF](https://arxiv.org/pdf/2607.03691v1)

**作者:** Oussama Ben Sghaier `[一作]` (Queen's University), Ahmed E. Hassan `[通讯]` (Queen's University)

**通讯引用:** 24846 | [OpenAlex ID](https://openalex.org/A5091586373)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文开展了首个控制实验，固定大型语言模型，仅变更编程代理的中介层（scaffolding）在35个连续版本中进行纵向评估，探究其演进对代理质量（任务完成率、token消耗和工具调用次数）的影响。

**💡 创新点**

创新点在于：①将scaffolding视为可量化、可测评的软件组件并进行系统性演进分析；②通过“保持LLM不变”与“仅变更scaffolding”的对照实验，首次揭示scaffolding升级常导致质量退化而非提升；③从项目级与架构级两个层面关联发布模式、代码变动和质量波动，提出“Agentic QA”新型质量保证思路。

**🔧 技术方法**

所用技术包括：Python/Node.js实现的Qwen Code CLI、vLLM部署的Qwen3-Next-80B-A3B模型、SWE-bench Verified 500题的子集 50 题进行两次执行、解析执行日志获取 token 与工具调用量、统计学检验（Wilcoxon、Spearman、Cliff's d、BH校正）等。

**📊 数据集**

使用的数据集为：①来自GitHub的35个 Qwen Code CLI 版本发布记录；②SWE-bench Verified 500 题中按难度抽样得到的 50 题；③每个版本对应的 2 次运行产生的 3,500 条执行与评估日志。

**📈 对比分析**

与传统研究（固定scaffolding、变更LLM）相比，本文实验显示：尽管发布频率高、代码变动大，resolve率保持在约 30% 左右无显著提升；token 消耗与工具调用数在后期版本显著增加（高达 70%），体现了“hyper-churn”导致的资源浪费；因此，质量并未随迭代提升，而是出现了明显的回归。

**⚠️ 局限性**

局限性包括：仅评估单一scaffolding（Qwen Code CLI）与单一LLM（Qwen3-Next-80B-A3B），结果可能不适用于其他代理或模型；任务样本量为 50 题，未覆盖所有可能的代码修复情形；实验资源成本高，难以在更大规模或多模型上重现；缺乏实时 CI 测试框架来验证结果。

---

## 3. Task-Centered Benchmark for Interactive Network Visualization & Analysis

**arXiv ID:** 2607.03725 | [PDF](https://arxiv.org/pdf/2607.03725v1)

**作者:** Ameya Patil `[一作]` (University of Washington), Leilani Battle `[通讯]` (University of Washington)

**通讯引用:** 1981 | [OpenAlex ID](https://openalex.org/A5009763002)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向交互式网络可视化与分析（INVA）的任务中心化基准框架，并在框架中实现了工作负载生成、数据生成和驱动程序，用于评估不同图系统在 INVA 工作负载上的表现。

**💡 创新点**

首次提出基于人类分析流程的 INVA 任务模型和工作负载生成方法；在此框架下揭示了专用 INVA 系统与图数据库在大规模网络上的性能差距及潜在错误。

**🔧 技术方法**

使用 Python + NetworKit、Dask 进行工作负载和数据生成；Java/C++/Python API（Cytoscape CyREST、Neo4j Cypher、Memgraph MAGE）作为驱动；采用多种性能指标（可扩展性、加载时间、交互响应时间、正确性、工作负载完成时间）。

**📊 数据集**

使用 8 个网络数据集（4 个真实、4 个合成）涵盖 1,000 到 15M 边的规模，属性多样；数据来源包括 Network Repository 及自研数据生成器。

**📈 对比分析**

通过统一驱动测量可扩展性阈值、加载时间、交互响应时间和结果正确性；实验显示 Memgraph 最优，Cytoscape 最差；图数据库在约 1M 规模仍保持交互阈值，专用 INVA 系统在更小规模即可失效；正确性问题在 Cytoscape 与 Memgraph 中被揭露。

**⚠️ 局限性**

工作负载模型未覆盖所有可能的图操作；缺乏正式用户研究验证；仅评估了少数系统，未包括 Gephi、TigerGraph 等；接口实现与图类型转换带来额外复杂性；OLAP 与 OLTP 的交互阈值定义仍待深入研究。

---

## 4. Optimizing Large Language Models for Causality Assessment in Pharmacovigilance: Developing a Performance Metric as Objective for Bayesian Hyperparameter Optimization

**arXiv ID:** 2607.03704 | [PDF](https://arxiv.org/pdf/2607.03704v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 5. A Failure-Mode Benchmark for Polymorphic Sybil Poisoning in RAG

**arXiv ID:** 2607.03739 | [PDF](https://arxiv.org/pdf/2607.03739v1)

**作者:** Donghyun Lee `[一作]` (Dongguk University), Juntae Kim `[通讯]` (Dongguk University)

**通讯引用:** 1764 | [OpenAlex ID](https://openalex.org/A5005319406)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并发布了一个用于评估检索增强生成（RAG）系统在协同检索毒化下鲁棒性的基准与评测框架，重点关注语义多样化的Sybil攻击与评估四种失败模式（gold、hijack、abstention、drift）；

**💡 创新点**

创新点包括：①设计了多样化的Polymorphic Sybil攻击，能够绕过基于词重叠的过滤器；②提出四分类失败模式的评估框架和实例级转移矩阵；③引入Forced Exposure协议，在固定检索槽位下隔离读取器的冲突解决；

**🔧 技术方法**

使用的技术包括：LLM文本生成（Llama-3.1-8B），LLM验证器（Qwen2.5-72B），检索模型（BM25、ColBERTv2、E5），以及多轮评估脚本和强制曝光机制；

**📊 数据集**

数据集涵盖：NQ、HotpotQA、TriviaQA、2WikiMultiHopQA，构建了3,145道问题的Sybil基准集合；

**📈 对比分析**

对比方法：在5个不同规模（7B–120B）阅读器与2种检索器上进行攻击与干净对照实验，衡量ACC、ASR、abstention和drift，结果显示攻击可使hijack提升约+13–20pp，但47–66%的输出质量变更未被传统ASR/ACC捕捉；

**⚠️ 局限性**

局限性包括：仅设计了单一攻击类且只考虑词汇层面的多样化；Forced Exposure只在固定6:2:2比例下评估；生成器/验证器重叠可能导致偏差；实验受限于NQ/HotpotQA原始数据集，未覆盖更广泛的攻击/防御场景。

---

## 6. ProACT: Towards Breakdown-Aware Proactive Agent in Multi-User Collaboration

**arXiv ID:** 2607.03730 | [PDF](https://arxiv.org/pdf/2607.03730v1)

**作者:** Shu Yang `[一作]` (King Abdullah University of Science and Technology), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 485163 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ProACT框架，实现多方协作中的主动干预，基于诊断、技能路由决定是否发声。

**💡 创新点**

首次构建可检测协作失衡的诊断流程与针对性技能库，并创建多任务主动协作评估基准。

**🔧 技术方法**

利用LLM代理、诊断模块、技能库（冲突调解、循环破除、约束提醒等），并通过结构化接口执行。

**📊 数据集**

从GitHub issue讨论和QMSum公开对话中采集真实对话，再加入BEAM生成的合成案例，总共3,244条测试样本。

**📈 对比分析**

与直接聊天Baseline在五种LLM（GPT‑5.4、Kimi、Claude、Gemini、GPT‑OSS）上对比，ProACT在适用性、非干扰性、简洁性及干预质量上均显著提升（例如Kimi从0.222→0.870，非干扰从0.323→0.942）。

**⚠️ 局限性**

实验仅在离线、单轮决策场景，缺乏实时交互、用户适应与后续反馈；评估依赖LLM裁判，未覆盖信任、偏见与操纵等潜在风险。

---

## 7. Attending to Multimodal Generation One Token at a Time

**arXiv ID:** 2607.03738 | [PDF](https://arxiv.org/pdf/2607.03738v1)

**作者:** Varun Gupta `[一作]` (International Institute of Information Technology Hyderabad), Makarand Tapaswi `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过追踪多模态大语言模型在自回归解码过程中每一步对图像、文本、指令和已生成词的注意力分布，系统性地揭示了多模态计算的时间演化规律；

**💡 创新点**

创新点在于提出了One Token at a Time (OTaT)方法，对模型的全局注意力进行时间维度的聚合与归一化，并通过因果阻断与提升实验验证了这些注意力模式对生成质量的功能性影响；

**🔧 技术方法**

采用了注意力张量聚合、归一化、阻断（lazy与total）、提升（乘法重权）等技术，并在Gemini与Qwen系列大模型的多模态解码器上进行实验；

**📊 数据集**

使用了自制的Fruit‑Math、Visual Spatial Reasoning(VSR)和ChartQA三个任务集，分别涉及图像识别、空间关系推理和图表阅读；

**📈 对比分析**

与未做干预的基线相比，针对关键时间步提升对对应模态的注意力可使VSR等任务的准确率提升高达28.5%（7B模型），而随意提升则导致性能下降；

**⚠️ 局限性**

局限性包括对特定模型（Gemini、Qwen）和任务的依赖，未考察更细粒度的层/头级别调控，且阻断/提升仅在自回归生成阶段实验，未验证对交互式或多轮对话的适用性。

---

## 8. PIEFS: Physics-Informed Eigenfunction Features with Learnable Scaling

**arXiv ID:** 2607.03692 | [PDF](https://arxiv.org/pdf/2607.03692v1)

**作者:** Varvara Nazarenkko `[一作]`, Alexander Tarakanov `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理信息的自监督特征学习框架PIEFS，通过学习可变度量的Dirichlet能量来构造任务自适应的谱特征。

**💡 创新点**

创新点在于将Dirichlet能量的度量矩阵可学习化（含对角缩放与Givens旋转），并以序列坐标映射、经验Gram正交性与交叉熵相结合，得到可训练的谱坐标而非固定算子本征函数。

**🔧 技术方法**

使用的技术包括自回归神经坐标网络、可学习矩阵A(x)=Λ(x)U(x)、Givens旋转的SO(d)参数化、批量Gram正交约束、动态损失权重以及线性分类器。

**📊 数据集**

评估数据集涵盖低维合成任务（Two Moons、Circles）、表格数据HTRU2、手写数字MNIST以及CIFAR‑10的ResNet‑18特征嵌入。

**📈 对比分析**

与传统RF、LR、PCA+LR及无监督NeuralEF比较，PIEFS在大多数任务中取得更高或相近的分类准确率，特别在非线性可分的合成任务和高维图像嵌入上优于基线。

**⚠️ 局限性**

限制包括：学习的坐标并非固定算子的本征函数，批量Gram约束的近似性导致正交性不严格；旋转参数化受限于有限的Givens链；以及缺乏对实际PDE/算子本征问题的验证。

---

## 9. Do Medical Vision Language Models Actually See? A Counterfactual Grounding Framework and Hard-Negative Contrastive Training for Visually-Reliant Medical VLMs

**arXiv ID:** 2607.03647 | [PDF](https://arxiv.org/pdf/2607.03647v1)

**作者:** Anas Zafar `[一作]` (University of Texas MD Anderson Cancer Center), Jia Wu `[通讯]` (University of Texas MD Anderson Cancer Center)

**通讯引用:** 13034 | [OpenAlex ID](https://openalex.org/A5007475662)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一套反事实评估框架，通过替换输入图像为空白、像素打乱、无图像、CLIP检索到的硬负样本等条件，分离视觉与文本对医学 VQA 的贡献，并引入对比性降重目标（Contrastive Grounding Objective, CGO）进行模型训练；随后在四个医学 VQA 基准上对比多种模型，评估其准确率与视觉依赖度。

**💡 创新点**

①首次在医学 VQA 上系统地量化视觉 grounding ；②提出 CGO 通过对比硬负图像来抑制语言先验导致的图像不敏感；③通过一系列对照实验展示视觉依赖与视觉幻觉率之间的关系。

**🔧 技术方法**

使用 Qwen2.5‑VL‑7B‑Instruct 作为基线模型，采用 LoRA 微调实现 CGO；对图像进行六种对照处理；评估指标包括 Visual Reliance Score、Visual Hallucination Rate、Blank Drop、Image Sensitivity 等；统计方法为配对自助采样、McNemar 检验。

**📊 数据集**

四个医学 VQA 公开基准：PathVQA、PMC‑VQA、SLAKE、VQA‑RAD；交叉域诊断还使用了 ChartQA、ScienceQA、VQAv2、GQA、MedXpertQA‑MM、MMMU‑Medical 等数据集。

**📈 对比分析**

与原始 Qwen2.5‑VL‑7B、两种 MedVLThinker RL 微调模型相比，CGO 训练得到的 7B 模型在 400 条配对样本上显著提升宏观准确率 +6.7pp（95% CI [+0.75, +12.5]），显著降低视觉幻觉率 -8.0pp（P<0.001）；RL 微调模型未出现显著改进；在无图像或空白条件下，模型仍保持较低准确率，表明仍存在文本先验依赖。

**⚠️ 局限性**

受限于样本规模（每基准仅 100 条固定样本）导致某些次要指标（如 Visual Benefit Rate、Novel Visual Claim Rate 等）未达到统计显著；评估仅覆盖 Qwen2.5‑VL 与 MedVLThinker 家族，未验证到其他医学 VLM；对比性指标仅基于提取答案，未对完整推理链做深入分析。

---

## 10. PathMark: Protecting Intellectual Property of Mixture-of-Expert LLMs via Path Watermarks

**arXiv ID:** 2607.03688 | [PDF](https://arxiv.org/pdf/2607.03688v1)

**作者:** Yudong Gao `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 26807 | [OpenAlex ID](https://openalex.org/A5100328273)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为Mixture-of-Experts大型语言模型设计了基于专家路由的水印机制，实现模型所有权验证。

**💡 创新点**

创新点在于将路由路径作为隐蔽水印通道，采用分布对齐损失、宽路径配置、对比损失等三种机制解决脆弱决策边界和路由纠缠问题，并支持多位水印。

**🔧 技术方法**

使用了MoE路由对齐损失、对比散列损失、宽路径设计、黑白盒验证协议、量化、微调、裁剪等技术。

**📊 数据集**

实验使用Qwen1.5-MoE、Mixtral、Phi3.5-MoE、Qwen3-30B等四个MoE模型，以及WikiText-103、PTB-Text、MarkMyWords等数据集。

**📈 对比分析**

与KGW、IFMark、LearnMark、EaaW等基线相比，水印成功率超过99%，在量化、微调、裁剪等攻击下仍保持高准确率，且模型困惑度提升不足2%。

**⚠️ 局限性**

局限性包括对高层专家选择的依赖、黑盒验证需额外微调、可能在极端量化或大规模裁剪下效果下降，以及多水印共存时交叉干扰增加。

---

## 11. ELiTeFormer: An Efficient Transformer for FPGAs

**arXiv ID:** 2607.03652 | [PDF](https://arxiv.org/pdf/2607.03652v1)

**作者:** Victor Agostinelli `[一作]` (Oregon State University), Antonino Tumeo `[通讯]` (Pacific Northwest National Laboratory)

**通讯引用:** 2427 | [OpenAlex ID](https://openalex.org/A5041853964)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 ELiTeFormer，结合混合线性注意力与三值量化的 FFN，并在 Xilinx VCK5000 FPGA 上共设计并部署了专用加速器。

**💡 创新点**

创新点包括：①首次将混合线性注意力与 BitNet b1.58 风格的三值投影融合进 Transformer 架构；②提出 ELTF PE 微架构，利用位掩码实现无乘法运算，彻底消除 DSP 需求；③在 FPGA 上实现并验证与 GPU、现有 FPGA 加速器相比显著降低延迟、提升吞吐量和能效。

**🔧 技术方法**

使用了混合线性注意力（滑动窗口+Hedgehog）、BitNet b1.58 三值量化、FPGA 高层合成 (HLS) 与 Vivado、定制 PE 微架构、注意力蒸馏与微调、资源映射到 Xilinx VCK5000。

**📊 数据集**

主要在 Alpaca 数据集的 50k 句子子集上进行训练/微调；评估使用公开基准 MMLU、ARC、PiQA 等。

**📈 对比分析**

通过与 LLaMA‑3、BitNet b1.58 在 NVIDIA A100 GPU（vLLM、bitnet.cpp）和 Intel CPU 进行对比，并在 FPGA 上模拟与部署后，ELiTeFormer 在长上下文下延迟比 LLaMA‑3 低 4.5×、吞吐量比 vLLM 高 2.2×，能效比 A100 高 3.2×；权重压缩 10×、KV 缓存压缩 12.8×。

**⚠️ 局限性**

局限性包括：对 FPGA 资源和规模的依赖（模型仍受限于可部署参数）；三值量化在某些任务上可能导致精度下降；训练需大量 GPU 时钟；未在更大模型或更长上下文下验证；迁移到其他硬件或通用框架的成本较高。

---

## 12. Validation-Induced Shapley Shifts: How Validation Structure Distorts Data Valuation

**arXiv ID:** 2607.03675 | [PDF](https://arxiv.org/pdf/2607.03675v1)

**作者:** Yinan Shen `[一作]` (Adobe), Hongfu Liu `[通讯]` (Brandeis University)

**通讯引用:** 3320 | [OpenAlex ID](https://openalex.org/A5086089915)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在验证集注入噪声，揭示了训练数据Shapley值对验证集结构极为敏感，并能产生一致的方向性压缩现象。

**💡 创新点**

创新点在于发现并解释了验证集引起的Shapley值漂移机制——邻域重排效应，并提出了基于熵归一化与边界感知的校正策略。

**🔧 技术方法**

主要技术包括KNN‑Shapley框架、噪声注入实验、邻域重排分析、边界/非边界分组校正以及熵匹配归一化。

**📊 数据集**

实验数据集涵盖了合成的二维高斯分布以及六个真实数据集：CreditCard、Phoneme、Planes2D、Pol、CPU 与 News20。

**📈 对比分析**

通过将噪声验证下的Shapley分布与无噪声基线进行对比，校正后标准差与正值比例显著恢复到基线水平，表明方法在稳定性与可解释性上均有提升。

**⚠️ 局限性**

局限性包括：仅在KNN‑Shapley情形下验证，未讨论其他模型；校正依赖于边界/非边界划分，可能需要额外标注；缺乏理论上对漂移幅度的定量界定。

---

## 13. CoGen3D: An Agentic Human-AI Co-Design Pipeline for 3D Asset Generation for Virtual Reality

**arXiv ID:** 2607.03731 | [PDF](https://arxiv.org/pdf/2607.03731v1)

**作者:** Weiwei Jiang `[一作]` (Nanjing University of Information Science and Technology), Zhanna Sarsenbayeva `[通讯]` (University of Sydney)

**通讯引用:** 1222 | [OpenAlex ID](https://openalex.org/A5024805223)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一个以对话为核心的 AI‑人机协作管线，让非专业用户能够通过自然语言交互生成、确认 2D 概念图，再转化为可直接部署到 VR 场景的 3D 资产。

**💡 创新点**

创新点包括：① staged 对话式意图挖掘 + 2D 确认门槛的多阶段协作流程；② 将 LLM 与文本‑图像、图像‑3D 两个生成端点无缝对接；③ 通过大规模用户实验验证该流程提升了沉浸体验与情绪调节，并揭示了 2D‑3D 质量瓶颈与作者身份无关的“IKEA 效果”缺失。

**🔧 技术方法**

使用技术：大型语言模型 DeepSeek V3 进行对话引导；FLUX.1 文本‑图像生成；Hunyuan3D‑2 图像‑3D 转换；Unity + glTFast 实时加载 3D 资产；后端 Django + API 管理任务与日志。

**📊 数据集**

实验数据集：六个情感维度均衡的 VR 场景（来自已验证的情绪激发库）以及 120 名参与者的对话、生成记录与情感评估数据。

**📈 对比分析**

比较方法：将设计组（共创）与验证组（仅体验）对比，采用 SAM（情感）评分、资产满意度、场景停留时间与交互行为等多维指标。结果显示：2D 概念图满意度显著高于最终 3D 资产；生成时延约 10 s（图像） vs 194 s（3D）；在无资产基线下，资产化显著提升中性/负面场景的情绪与沉浸时间。

**⚠️ 局限性**

局限性：① 3D 生成质量下降导致满意度缺失；② 资产只能单一对象，缺乏多对象协同；③ 缺乏空间感知（缩放、位置）自动化；④ 仅在大学生样本上验证，泛化性待考；⑤ 可能存在新奇效应，长期效果未知。

---

## 14. IPDiff: Diffusion-driven ORSI Salient Object Detection with Information Reconstruction and Multi-Prior Guidance

**arXiv ID:** 2607.03696 | [PDF](https://arxiv.org/pdf/2607.03696v1)

**作者:** Gongyang Li `[一作]` (Shanghai University), Xiao-Ping Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 37247 | [OpenAlex ID](https://openalex.org/A5100363169)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出IPDiff，一种基于扩散模型的动态优化框架，用于光学遥感图像显著目标检测。

**💡 创新点**

创新点在于将显著目标检测建模为条件扩散问题，利用信息重建和多先验引导的先验网络，并在测试阶段采用动态时间步迭代优化。

**🔧 技术方法**

技术包括扩散模型、信息重建驱动注意力模块(IRAM)、多先验引导去噪网络、信息扰动模块(IPM)以及空间-频域混合损失。

**📊 数据集**

使用公开的ORSSD、EORSSD、ORSI-4199三大光学遥感显著目标检测数据集。

**📈 对比分析**

与46种最新方法（包括CNN、Transformer、轻量级、光学遥感特定与扩散驱动方法）对比，IPDiff在S、F、E、MAE等指标均居首，速度约4fps。

**⚠️ 局限性**

局限在于推理速度慢（需要10步去噪），模型参数量较大（82.6M），对扩散时间步的选择敏感。

---

## 15. The Objective Decides: When a Learned Dynamics Model Uses a Conserved Quantity

**arXiv ID:** 2607.03728 | [PDF](https://arxiv.org/pdf/2607.03728v1)

**作者:** Chih-Ting Liao `[一作]` (University of New South Wales), Xin Cao `[通讯]` (University of New South Wales)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了一种单步激活互换（deployment probe）方法，用来区分神经网络中物理守恒量的可解码性与实际使用性，展示了在机械、电路、PDE等多种系统以及大规模PDE基础模型中，守恒量虽可高精度解码，但在下一步预测中并不被使用；并证明部署差距（deployment gap）能有效预测模型在分布外的泛化性能。

**💡 创新点**

创新点包括：①从可解码性(decodability)到因果使用(deployment)的概念拆分；②提出单步互换介入的低成本因果测试仪；③构建守恒量部署的代数判据并可通过改动输出代数实现控制；④首次在大规模预训练PDE模型上复现该拆分并展示部署差距与OOD性能的强相关性。

**🔧 技术方法**

技术手段包括：线性回归（ridge）解码器、单步激活互换（patching）、无尺度转移相关（transfer‑corr）评估、代数判据判定、随机方向和标签打乱的对照、冻结权重单步前向推理、以及对多架构（MLP、GRU、Transformer）和多维度系统的统一实验框架。

**📊 数据集**

使用的数据集涵盖：合成动力学（单摆、LC电路、中心力、波动PDE、Kepler 1/r²）、带历史窗口的循环/注意力模型、以及公开的158M参数Poseidon‑B PDE基础模型在真实Navier‑Stokes轨迹（NS‑Sines）上的数据；此外还设计了含捷径的控制任务以检验OOD泛化。

**📈 对比分析**

评估方式：比较每个系统的解码R²、下一步预测与守恒量相关的转移相关（β_next、β_inv）以及两者之差（部署差距γ）。结果显示：所有系统的解码R²≥0.8，β_next≈0（无使用），β_inv≈+1（使用），且γ与OOV准确率呈高相关（r≈+0.97）。这表明解码能力并不能反映模型使用守恒量，而部署差距能有效区分模型泛化能力。

**⚠️ 局限性**

局限性包括：单向互换可能低估部署强度（因冗余编码）；只针对线性解码方向；对线性/广义守恒量（如线性动量）处理为混淆边界；实验仅涵盖有限的系统与架构，未在更大规模任务中进一步验证部署差距的泛化预测；且对不同训练目标和输出代数的完整理论阐释仍待完善。

---

## 16. EmCom-Diffusion: Probing Visual Reflection in Emergent Languages via Image Generation

**arXiv ID:** 2607.03752 | [PDF](https://arxiv.org/pdf/2607.03752v1)

**作者:** Haruumi Omoto `[一作]` (Kyoto University), Tadahiro Taniguchi `[通讯]` (Ritsumeikan University)

**通讯引用:** 2879 | [OpenAlex ID](https://openalex.org/A5023160093)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 EmCom‑Diffusion，通过从 emergent language 生成图像并与原图像比较，直接衡量视觉反射。

**💡 创新点**

创新点在于用生成式图像重构代替人类概念或任务准确率，消除 proxy 约束，能直接评估语言对视觉内容的还原能力。

**🔧 技术方法**

利用预训练的文本到图像扩散模型（Stable Diffusion）微调，并使用 CLIP、DINOv2、SigLIP 等感知相似度函数进行评估。

**📊 数据集**

使用 MS‑COCO 2017 数据集，在 Referential Game 中生成 emergent language，然后在验证集 5,000 张图像上进行实验。

**📈 对比分析**

与 CBM、翻译、TopSim、R@1 四种传统指标对比，EmCom‑Diffusion 在三种视觉编码器上能更精准地区分真实 emergent language、随机/固定词序列以及含自然语言提示的强基线，性能显著优于这些方法。

**⚠️ 局限性**

局限在于对扩散模型先验的依赖、仅验证单一游戏和数据集，且未能明确哪些具体视觉特征被保留或丢失。

---

## 17. Lost in Time? Continuous Symmetry and Identifiability in Aided Inertial Navigation with Unknown Measurement Delays

**arXiv ID:** 2607.03699 | [PDF](https://arxiv.org/pdf/2607.03699v1)

**作者:** Jonathan Kelly `[一作]` (University of Toronto), Mattew Giamou `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `51c0528b-f690-4182-ae60-bb5f046c276c` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究了有延迟辅助传感器的惯导导航系统中，状态估计与未知固定延迟参数的可辨识性，并给出了何种车辆轨迹会导致延迟与初始状态无法唯一确定的几何条件。

**💡 创新点**

提出了以特殊伽利略群（Galilean group）为基础的连续对称性框架，将轨迹形状与可辨识性直接关联，揭示了比以往更广泛的“不可辨识”轨迹族，并用对称性解释了雅可比矩阵的零空间。

**🔧 技术方法**

使用了李群（Galilean群）理论、李代数与指数映射、状态转移矩阵构造、连续对称性分析，以及传统的非线性可辨识性（雅可比矩阵）方法。

**📊 数据集**

本文未使用任何公开数据集；研究基于理论推导和数值示例（如直线加速、圆形/螺旋运动等）验证所得到的轨迹分类。

**📈 对比分析**

由于是理论性工作，未与实验方法进行性能比较；讨论中仅说明在可辨识轨迹下，至少两次完整状态观测即可实现局部可辨识；在不可辨识轨迹下，任何观测数均无法唯一确定延迟和初始状态。

**⚠️ 局限性**

局限性：仅考虑了单一辅助传感器且假设延迟为恒定未知量；未讨论噪声、时间变化偏置、相对测量或多传感器协同校准；对复杂真实场景中的非理想因素（如传感器失效、离散采样误差）分析不足。

---

## 18. A Fair Benchmarking of Deep Relational Database Learning Models

**arXiv ID:** 2607.03659 | [PDF](https://arxiv.org/pdf/2607.03659v1)

**作者:** Kazi F. Akhter `[一作]` (Tennessee State University), Manar D. Samad `[通讯]` (Tennessee State University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统地对当前主流深度学习方法在关系型数据库（RDB）上的性能进行评估，构建统一的实验协议，涵盖单表、1跳、2跳三种邻域设置。

**💡 创新点**

首次在同一实验框架下对RT、Griffin、DBFormer等模型进行公平比较，并发现Transformer‑based RT在分类和回归任务中普遍优于图神经网络和传统表格学习基线。

**🔧 技术方法**

采用关系变换器（Relational Transformer）、图神经网络（Griffin）、混合图‑变换器（DBFormer）等深度模型，并对比了TabPFN 2.5和LightGBM等表格学习基线。

**📊 数据集**

使用RelBench公开的五个数据库（rel‑amazon、rel‑avito、rel‑f1、rel‑stack、rel‑trial），每个数据库提供一个分类和一个回归任务，任务规模分为小/中/大。

**📈 对比分析**

实验统一数据拆分、训练轮次、embedding维度等设置，评估指标为AUROC（分类）和RMSE（回归）。结果显示RT在15个实验案例中赢得14/15，平均排名第一；TabPFN 2.5在样本量极小的任务上表现更佳；多跳（hop 2）提升有限，往往伴随显著的计算成本。

**⚠️ 局限性**

局限性在于多跳扩展的收益与成本不成正比，尤其hop 2在多数任务上提升微乎其微；DBFormer表现不佳；实验仅覆盖RelBench数据，缺乏跨行业更大规模数据验证。

---

## 19. OmniTacTune: Policy-Agnostic Real-World RL for Tactile Residual Adaptation of Visual Policies

**arXiv ID:** 2607.03723 | [PDF](https://arxiv.org/pdf/2607.03723v1)

**作者:** Kelin Yu `[一作]` (University of Maryland), Ruohan Gao `[通讯]` (University of Maryland)

**通讯引用:** 1387 | [OpenAlex ID](https://openalex.org/A5032081267)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无论基础视觉策略如何都可适用的两阶段实时强化学习管线，通过残差校正将触觉反馈整合进预训练视觉策略，以提升接触丰富操作的成功率。

**💡 创新点**

核心创新在于将视觉先验与触觉残差结合：热启动阶段利用视觉策略自举触觉编码器和批评者，随后在线强化学习学习轻量级触觉残差策略，并通过多感官奖励实现高效样本利用。

**🔧 技术方法**

采用深度强化学习（SAC）、流式视觉表示（Im2Flow2Act）、多模态奖励塑形、Tactile Encoder 微调以及对象中心奖励等技术。

**📊 数据集**

使用从人类视频（Meta Quest手部追踪）与机器人远程操作获取的视觉数据，配合GelSight Mini触觉传感器采集的触觉信息。

**📈 对比分析**

与PLD*、PLD、ViTAL以及基于ACT、DP、π_0.5等多模态策略的对比显示，该方法在四个接触丰富任务上从5-40%提升到85-100%成功率，训练时长仅40-80分钟。

**⚠️ 局限性**

局限在于仍需人工复位、对硬件磨损敏感、触觉传感器脆弱，且受限于真实环境强化学习的采样效率与安全性。

---

## 20. Exploring SAM Supervision for Fine-Grained UAV Target Segmentation under Data Scarcity

**arXiv ID:** 2607.03754 | [PDF](https://arxiv.org/pdf/2607.03754v1)

**作者:** Le-Anh Tran `[一作]` `[通讯]`, Le-Anh Tran

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用 SAM3 生成伪标签，训练轻量化 UAV 目标分割网络 IPS‑Seg，解决标注稀缺与算力受限的问题。

**💡 创新点**

提出两阶段 SAM3 伪标签生成策略和融合 IdentityFormer、ASPP 与 PixelShuffle 的高效网络架构，实现大模型知识向小模型的有效迁移。

**🔧 技术方法**

利用 SAM3 生成粗糙和细化掩码，结合两阶段伪标签；网络采用 IdentityFormer backbone、Atrous Spatial Pyramid Pooling（ASPP）瓶颈以及 PixelShuffle 解码器，训练时使用二值交叉熵损失。

**📊 数据集**

在 UAV Semantic Segmentation 数据集（约30万张航空图像）上进行实验，随机抽取8k张作为训练/验证集。

**📈 对比分析**

与 U‑Net、PSPNet、U‑Net++、ResUNet、DeepLabV3+、TransUNet 等多种基准模型对比；在全监督下 IPS‑Seg 获得 IoU 0.8164、Dice 0.8943，仅 2.69M 参数、9.72 GFLOPs；伪标签训练下 IoU 0.7941、Dice 0.8737，几乎匹配 SAM3 的性能；两阶段伪标签虽在指标略低，却显著提升了细节和结构保留。

**⚠️ 局限性**

伪标签的细化掩码未被传统 IoU/Dice 充分评估，导致指标略低；两阶段生成成本较高；缺乏置信度筛选和边界感知的优化，限制了对极小目标边界细节的进一步提升。

---

## 21. Rethinking AI-Generated Text Detection: A Strong Baseline and the Distribution-Shift Problem That Remains

**arXiv ID:** 2607.03680 | [PDF](https://arxiv.org/pdf/2607.03680v1)

**作者:** Zhuoer Shen `[一作]` (University of California), Yuheng Bu `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新评估 AI 生成文本检测，使用全微调的 RoBERTa 作为基线，并针对分布漂移设计轻量级域适配与置信度加权集成。

**💡 创新点**

证明全微调基线即可匹配或超过专用检测器；揭示分布漂移下高置信误判失效模式，并提出基于 FOMAML+LoRA 的 K‑shot 适配和置信度加权集成以提升低 FPR 性能。

**🔧 技术方法**

RoBERTa 全微调、LoRA、第一阶 MAML（FOMAML）元学习、置信度加权集成、AUROC 与低 FPR 评估。

**📊 数据集**

IntelLabs、MAGE、FAID、MIRAGE 四个公开检测基准，以及 HC3 跨域评估。

**📈 对比分析**

与每个基准原始专用检测器直接对比，RoBERTa 基线在同分布下优于 Anchor、Longformer、FAID、DetectAnyLLM；在分布漂移下性能急剧下降，但 K‑shot 适配 + 集成可在低 FPR 下提升 50–70% 左右。

**⚠️ 局限性**

输入被截断至 512 词导致潜在攻击；集成方法仅适用于二分类，无法直接扩展到多类；需要少量目标标注样本；仅在存在高置信误判时才有效；未覆盖全长输入鲁棒性与多类设置。

---

## 22. LLM-Guided Transportation Hub Capacity Planning with Textual Business Inputs

**arXiv ID:** 2607.03651 | [PDF](https://arxiv.org/pdf/2607.03651v1)

**作者:** Xiaoyue Liu `[一作]` (Georgia Institute of Technology), Zheng Dong `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 44637 | [OpenAlex ID](https://openalex.org/A5050155862)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将大型语言模型（LLM）与两阶段随机规划相结合，提出了一种基于文本业务输入的运输枢纽容量规划框架。

**💡 创新点**

创新点在于利用LLM的链式思维构建业务上下文决策表，并通过仅路由反馈而非成本反馈来引导容量调整，形成迭代优化循环。

**🔧 技术方法**

使用的技术包括Claude Sonnet 4等大型语言模型、链式思维推理、两阶段随机规划、路由评估反馈与迭代决策循环。

**📊 数据集**

实验数据来自美国东南部三州（佛罗里达、佐治亚、南卡罗来纳）的Freight Analysis Framework网络，并结合15条自然语言业务上下文。

**📈 对比分析**

与传统基线优化（不考虑文本上下文）和已知真实参数的最优解相比，LLM框架将最优性差距从11.0%降低至2.8%。

**⚠️ 局限性**

主要局限包括网络规模受限、LLM对成本信号的排斥可能导致偶尔丢失更优解、缺乏理论收敛保证以及对不同LLM架构的适用性尚未验证。

---

## 23. CoRE-VLA: Towards Scalable and Robust Vision-Language-Action Modeling via Conditional Routing of Experts

**arXiv ID:** 2607.03693 | [PDF](https://arxiv.org/pdf/2607.03693v1)

**作者:** Haozhe Zhang `[一作]` (Zhejiang University), Xipeng Qiu `[通讯]` (Fudan University)

**通讯引用:** 18403 | [OpenAlex ID](https://openalex.org/A5044665993)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可扩展且对缺失辅助传感器鲁棒的 Vision‑Language‑Action（VLA）框架 CoRE‑VLA，利用任务意图与传感器可用性实现条件专家路由，支持在多任务和长时程操作中自适应生成动作。

**💡 创新点**

创新点在于将动作生成视为上下文条件的稀疏计算；通过任务意图挑选并路由动作侧表示到任务相关专家，传感器可用性门控模态专用专家，从而在不依赖辅助传感器的情况下保持性能，同时在有传感器时提升表现；结合专家专用与稀疏路由避免了全密集计算导致的参数冗余与任务干扰。

**🔧 技术方法**

核心技术包括：预训练 VLM 编码 RGB 与语言；深度学习的 Action Diffusion Transformer（Action DiT）作为基底；CoRE 块实现稀疏表示选择（Top‑K）与条件专家路由；模态丢弃（modality dropout）与专家掩码；流匹配训练目标与负载平衡正则；以及使用 DA‑V2 估计的伪深度作为辅助模态。

**📊 数据集**

使用的主要数据集：LIBERO（包含四个子套件）、RoboCasa GR1 Tabletop（24 个桌面操作任务），以及真实世界双臂机器人平台上的 Vegetables‑Picking、Clothes‑Folding、Fabric‑Folding 三个任务的演示数据（分别约 44、1300、0 份演示）。

**📈 对比分析**

对比方法包括 OpenVLA、π_0、GR00T、Dense Action DiT 等；在 LIBERO 上 CoRE‑VLA 以 99.0% 最高分领跑所有子套件；在 RoboCasa 上取得 56.5% 的平均成功率，显著优于 Dense 基线 40.4%；在真实世界实验中，CoRE‑VLA 在无辅助深度条件下取得 77.5 分/65% 成功率，在有物理深度摄像头时进一步提升至 78.8 分/70% 成功率，均优于 Dense Action DiT 与预训练 π_0.5 基线。

**⚠️ 局限性**

局限性：目前仅在深度模态上验证，其他辅助传感器（如触觉、力反馈）的泛化尚未充分探究；对多机体多模态场景的可扩展性仍待验证；模型训练和推理成本较高，需大规模 GPU；在极端缺失传感器或噪声严重的环境中，鲁棒性有待进一步提升。

---

## 24. Annotating Korean adnominal ending constructions in corpus data: Beyond relative-clause identification

**arXiv ID:** 2607.03681 | [PDF](https://arxiv.org/pdf/2607.03681v1)

**作者:** Jungyeul Park `[一作]` (KAIST), Chulwoo Park `[通讯]` (Anyang University)

**通讯引用:** 576 | [OpenAlex ID](https://openalex.org/A5103074085)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了基于语料库的韩语-ETM后缀的构式分类方法，并在KLUE依存树库上应用自动标注与人工验证。

**💡 创新点**

将-ETM视为多种名词修饰构式的共同形态，而非单一的关系从句标记，构建了包含相对从句、形容词、系词、束缚名词、情态、时间和词汇化短语等八类的系统性构式层级。

**🔧 技术方法**

采用有序规则化决策流程结合形态词性与依存结构特征，并在必要时调用Sejong词典以辅助判定。

**📊 数据集**

使用KLUE训练集共13,046个-ETM实例作为标注与分析的语料。

**📈 对比分析**

通过层级抽样的人工验证，得到自动标签与人工判断的91.8%一致率（Kappa=0.677），验证了规则系统的可靠性。

**⚠️ 局限性**

仍存在标注错误、结构不完整及语义歧义导致的83个未归类实例，且该方法仅在KLUE上测试，需在其他树库进一步检验。

---

## 25. Can Conversational Temporal Dynamics Improve Depression Detection in Dyads? A Preliminary Investigation in Multi-Modality Perspectives

**arXiv ID:** 2607.03744 | [PDF](https://arxiv.org/pdf/2607.03744v1)

**作者:** Hanie Kang `[一作]` (University of Southern California), Shrikanth Narayanan `[通讯]` (University of Southern California)

**通讯引用:** 31656 | [OpenAlex ID](https://openalex.org/A5010028928)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了对话时序动力学（CTD）在临床访谈中检测抑郁的效果，并将其作为一种轻量、可解释的第三模态与冻结的WavLM（声学）和RoBERTa（语义）模型进行融合。

**💡 创新点**

创新点在于首次将基于交互问答对的24维对话时序特征直接作为独立模态加入多模态抑郁检测，并证明其在单模态下即可匹敌甚至优于1024维的自监督编码器；同时通过凸加权融合展示CTD对提升性能的决定性作用。

**🔧 技术方法**

技术包括：①使用固定的CTD特征集（Ask/Res对的持续时长、语音时长、静默时长、延迟等24维）并以平均值形成会话向量；②冻结WavLM-large和RoBERTa-large提取声学与语义嵌入，配合轻量的PROBE头；③在概率级别进行平均、对数几率以及凸加权融合，并在开发集上调优阈值与权重。

**📊 数据集**

使用公开的DAIC‑WOZ数据集（180个访谈，包含音频、时间对齐文本与PHQ‑8抑郁标签，按官方划分为训练/开发/测试）。

**📈 对比分析**

实验对比单模态与多模态融合：单模态CTD在开发集上macro‑F1达0.746，优于WavLM（0.673）与RoBERTa（0.690）；三模态凸加权融合（权重A=0，T=0.3，CTD=0.7）在开发集上提升至0.804，测试集上达到0.669，表明CTD在融合中贡献最大。

**⚠️ 局限性**

局限性包括：DAIC‑WOZ样本量小且类别不平衡；实验仅在单一划分上评估，缺乏跨语料验证；CTD特征受访谈者提问模式影响，可能存在“提示”短路；声学分支弱，可能与探测器或数据质量相关；未对多语言或跨文化数据进行测试，亦未评估模型的公平性与鲁棒性。

---

## 26. Between Knowledge and Care: A Mixed-Methods Evaluation of Generative AI for T2DM Self-Management from Patient and Physician Perspectives

**arXiv ID:** 2607.03720 | [PDF](https://arxiv.org/pdf/2607.03720v1)

**作者:** Ruiqi Chen `[一作]` (University of Michigan), Xiaolan Ding `[通讯]` (North China University of Science and Technology Health Science Center)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过对21名2型糖尿病患者的查询日志和7名内分泌科医生的专家评估，探讨生成式人工智能在慢性疾病自我管理中的信息质量与适用性。

**💡 创新点**

创新点包括：①基于患者真实提问构建七大信息需求类别；②设计并验证了包含准确性、安全性、清晰度、完整性、行动导向五维度的医师评分表；③引入“预访前导”和“流畅幻觉”两种概念，揭示生成式AI在情感与个性化支持上的缺陷；④提出四条针对任务感知、风险回退、动态个性化和情感化交互的设计方向。

**🔧 技术方法**

使用了四款主流大语言模型（GPT‑4‑turbo、DeepSeek‑R1、Kimi‑K2、ERNIE‑Bot 4.5），并对其在患者提问中的回答进行评分与访谈分析。

**📊 数据集**

数据集包括：来自21名患者的784条自述查询（用于主题分析和问题集构建）以及从这些查询中精炼出的66条代表性问题，在四款模型上生成响应后由医生评分。

**📈 对比分析**

通过医生评分、单因素/多因素重复测量方差分析及箱线图、雷达图等可视化手段比较模型与问题类别的表现。结果显示GPT‑4‑turbo在准确性与安全性上遥遥领先，DeepSeek次之，而Kimi和ERNIE‑Bot表现较差；在“事实知识”“饮食管理”等类别表现优异，在“药物指导”“情感支持”等高风险类别表现明显不足。

**⚠️ 局限性**

局限性包括：①评估仅覆盖单轮回答，未考虑多轮交互；②样本仅为中国内分泌科医生，样本量小；③模型性能随时间快速变化，结果仅为时点快照；④未采用严格的提示工程或标准化评测数据集，结果受模型更新影响。

---

## 27. Leveraging Pathology Co-occurrence for Test-Time Adaptation in Chest X-Ray Diagnosis

**arXiv ID:** 2607.03715 | [PDF](https://arxiv.org/pdf/2607.03715v1)

**作者:** Woojin Jeong `[一作]` (Seoul National University), Jaewook Lee `[通讯]` (Seoul National University)

**通讯引用:** 19583 | [OpenAlex ID](https://openalex.org/A5100410780)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于病理共现信息的测试时适应方法CoWA，用来提升胸片多标签诊断模型在新域中的鲁棒性。

**💡 创新点**

创新点在于利用模型预测生成的共现矩阵作为每个样本的可靠性权重，针对多标签依赖关系而非传统TTA中对所有样本统一加权，从而减少噪声梯度并保持结构一致性。

**🔧 技术方法**

主要技术包括软伪标签阈值化、共现矩阵估计、基于Frobenius范数的样本一致性得分计算，以及加权熵最小化（仅更新BN参数）进行测试时自适应。

**📊 数据集**

实验使用了四个公开胸片数据集：MIMIC-CXR、CheXpert、VinDr-CXR 和 NIH Chest X-ray，构造了多源-多目标对以模拟不同域迁移场景。

**📈 对比分析**

与无适应、AdaBN、TENT、CoTTA、EATA、RoTTA等基线对比，CoWA在多种域偏移下平均 AUROC 均优于或逼近最佳方法，并且显著避免了单个病种性能骤降，表现出更好的极端案例稳健性。

**⚠️ 局限性**

局限性在于共现矩阵估计受样本分布偏差影响，可能缺乏因果解释；对低频病种仍可能出现噪声影响；仅更新BN参数在某些复杂域迁移场景下的适应能力有限。

---

## 28. Agent Reinforcement Learning via Pivotal-Aware Self-Feedback Retry

**arXiv ID:** 2607.03702 | [PDF](https://arxiv.org/pdf/2607.03702v1)

**作者:** Weiyang Guo `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 117920 | [OpenAlex ID](https://openalex.org/A5100336796)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在LLM代理的长序列交互中，提出一种自反馈重试框架——Pivotal‑Aware Self‑Feedback Retry（PivoARL），通过结构化自我反思定位导致失败的关键回合（pivotal turn），并仅从该关键状态进行局部重试，重用成功前缀，减少不必要的交互；同时设计了关键切分的信用分配机制和隐式反思奖励，提升对失败轨迹的有效利用。

**💡 创新点**

创新点包括①利用自我反思精准定位错误回合，实现局部重试而非全局重启；②将经验信息聚焦于错误边界，解决传统全局经验稀释问题；③提出关键切分信用分配（Pivotal Isolation）防止错误后缀被错误地奖励；④通过隐式反思奖励将反思质量与重试成功相联结，进一步强化学习。

**🔧 技术方法**

核心技术有：结构化自我反思（生成关键回合索引与修正提示）、局部重试策略、跨回合信用分配与关键切分、隐式反思奖励、信息增益分析，以及基于GiGPO/GRPO等RL框架的优化实现。

**📊 数据集**

实验数据集涵盖四个代理探索环境（Sokoban、Minesweeper、WebShop、ALFWorld）和七个检索式问答基准（NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle）。

**📈 对比分析**

与闭源LLM（GPT‑4o、Gemini‑2.5‑Pro）、提示式代理（ReAct、Reflexion）、RL基线（GRPO、GiGPO、GSPO）、内存增强RL（Mem0+GRPO、SimpleMem+GRPO、SkillRL）、反思重试RL（MetaRL）以及传统搜索方法（Search‑R1、ZeroSearch、StepSearch）进行比较。PivoARL在代理任务上平均提升约10.5%（相较MetaRL），在Minesweeper上比GiGPO提升约45%，在搜索任务上相较GRPO提升约22.6%，在Pass@1/2/3指标上整体优于所有基线，并且交互成本平均降低约42%（与全重试方法相比）。

**⚠️ 局限性**

主要局限：1）对关键回合定位的准确性高度依赖，错误的反思会削弱重试效果；2）相较单次回合RL，训练时需要额外的重试回合，导致训练成本提升；3）实验集中在模拟环境与检索式QA，尚未验证在更复杂的工具使用或真实世界任务中的鲁棒性。

---

## 29. Robust Feasible Route Construction through Collaborative Partition Optimization

**arXiv ID:** 2607.03694 | [PDF](https://arxiv.org/pdf/2607.03694v1)

**作者:** Oguzhan Karaahmetoglu `[一作]`, Hyong Kim `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出协同路由构造框架（CoRC），在车辆路径问题的聚类-路由（CFRS）过程中引入中间协作阶段，让子问题在路由构造期间动态交换客户与车辆。

**💡 创新点**

创新点在于：①允许路由构造时动态更新分区并通过协作操作（客户转移、车辆转移、子问题合并）补偿资源不足；②通过异步代理实现分布式协作，避免全局重优化的高昂成本；③保持原有路由器不变，兼容多种分区策略。

**🔧 技术方法**

采用演进式CFRS分区、统计摘要提取、协作优先策略、协作操作、异步代理决策循环；路由器使用 OR‑Tools GLS；实验中调参包括广告/提议等待时间。

**📊 数据集**

数据集包括 AGS benchmark（Ghent1、Brussels1、Flanders1、Flanders2）以及100K、200K 客户的合成实例。

**📈 对比分析**

与独立路由（-Ind）、后期全局重优化（-Reopt）以及完整框架 ScaleNet、HGS 进行对比。CoRC 在所有实例中几乎秒级完成可行解，距离往往取决于初始分区质量；独立/重优化在大规模实例无法得到完整解，完整框架在 100K/200K 实例内耗尽资源。

**⚠️ 局限性**

局限性：①仍受初始分区质量影响，影响最终路径距离；②协作策略参数需要经验调优；③在极端资源失衡（>90% 车源缺失）下仍无法完全满足需求；④实验仅覆盖离线情境，实时/动态场景未评估。

---

## 30. Bridging Interleaved Multi-Modal Reasoning as a Unified Decision Process

**arXiv ID:** 2607.03748 | [PDF](https://arxiv.org/pdf/2607.03748v1)

**作者:** Zican Hu `[一作]` (Nanjing University), Zhi Wang `[通讯]` (Nanjing University)

**通讯引用:** 254665 | [OpenAlex ID](https://openalex.org/A5100444820)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了BRAID框架，将文本-图像交替推理视为统一的MDP，并通过联合RL同时优化文本与图像生成。

**💡 创新点**

将多模态生成统一为单一策略，采用共享轨迹优势和视觉思维过程奖励，实现端到端跨模态信用分配。

**🔧 技术方法**

采用GRPO进行文本分支的策略梯度，DiffusionNFT实现图像分支的流匹配RL；使用VLM评判产生视觉过程奖励；联合单个轨迹优势传播。

**📊 数据集**

训练与评估在八类任务数据上，包括空间推理（SAT、VStar等）与视觉感知基准（V*Bench、CV-Bench等），使用SFT+RL混合数据。

**📈 对比分析**

与多种基线（BAGEL、Janus-Pro、Chameleon、VLMs等）对比，平均提升约+5.7分，超过GPT‑4o，仅7B参数；在SAT、VStar等上显著提升。

**⚠️ 局限性**

仅适用于BAGEL的AR‑扩散混合骨干；依赖外部VLM判定过程，计算成本高；固定交替模式，未探索自适应推理长度。

---

## 31. SABLE: An NDA-Safe Closed-Loop LLM Framework for Analog Circuit Optimization in Industrial EDA Flows

**arXiv ID:** 2607.03701 | [PDF](https://arxiv.org/pdf/2607.03701v1)

**作者:** Xunqi Li `[一作]` (University of Minnesota), Chris H. Kim `[通讯]` (University of Minnesota)

**通讯引用:** 7654 | [OpenAlex ID](https://openalex.org/A5043025421)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个安全边界框架，使云端大型语言模型（LLM）能够在受NDA约束的工业Cadence Virtuoso/MAESTRO/SPICE流程中闭环优化模拟电路，保证不会泄露PDK内容或仿真路径。

**💡 创新点**

创新点在于：①明确定义了“好奇但被动”云端攻击模型并列出四级攻击阶梯与五条硬性不变性；②设计了严格的JSON动作合同和六个机器可检 stop 条件；③实现了28个白名单SKILL入口、路径/模型擦洗、定向写回与最佳历史状态保持；④在闭环中提供了结构化的拓扑意图、数值指标、定向反馈与运营点摘要。

**🔧 技术方法**

技术上使用Python代理层、Cadence SKILL/OCEAN接口、MAESTRO配置脚本、Spectre仿真、JSON协议与六条终止条件、最优状态回写与回路安全擦洗。

**📊 数据集**

数据集主要为两种工业电路：20 GHz LC‑VCO调谐曲线和两级差分运算放大器的PVT签核；每个任务均在相同的三角角落（典型、慢热、快冷）下评估，采用11个不同LLM检查点作为模型。

**📈 对比分析**

比较方法是“相同起点、固定任务规范、记录终端原因” 的匹配对照实验；在LC‑VCO任务中7/11模型通过，平均迭代数4‑15；在运算放大器任务中4/11通过，说明更严格的多指标、相位裕度门槛提升了评估难度。

**⚠️ 局限性**

局限性包括：①单一随机种子/单次跑结果，未充分覆盖统计多样性；②安全性仅限于客户端侧，无法阻止供应商端的潜在信息推断；③过度抽象的反馈可能导致模型仅优化可见指标，忽视深层耦合问题；④结果受模型版本、终端稳定性和迭代预算影响，需要进一步多模型、多种拓扑的验证。

---

## 32. Content Hidden Behind Execution: Analyzing Public Scratch Projects at Runtime

**arXiv ID:** 2607.03700 | [PDF](https://arxiv.org/pdf/2607.03700v1)

**作者:** Yuan Si `[一作]` (University of Waterloo), Jialu Zhang `[通讯]` (University of Waterloo)

**通讯引用:** 24199 | [OpenAlex ID](https://openalex.org/A5015251883)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对500个公开Scratch项目进行运行时敏感内容审核，提出并应用了运行时感知的注释方案。

**💡 创新点**

提出将内容类别、风险等级、证据渠道、显现机制和置信度分离的运行时注释框架，并显示大部分项目需要运行时探索。

**🔧 技术方法**

使用了LLM辅助的注释流水线、无头运行环境、关键帧渲染、结构化运行时跟踪以及人工监督校验。

**📊 数据集**

从Scratch社区抽取的500个公开项目（50个已整理的样本、395个关键词检索、55个聚类样本）。

**📈 对比分析**

与仅基于静态元数据的筛选比较，发现仅7%项目可通过静态信息判断，93%需运行时检查，说明静态方法不足。

**⚠️ 局限性**

样本偏向敏感内容、运行时检查时间受限、未做独立可靠性评估、未覆盖公共上下文证据，结果非平台整体普遍性估计。

---

## 33. Social Networks of LLM Agents

**arXiv ID:** 2607.03695 | [PDF](https://arxiv.org/pdf/2607.03695v1)

**作者:** Kaixuan Liu `[一作]` (Emory University), Shengpu Tang `[通讯]` (Emory University)

**通讯引用:** 444 | [OpenAlex ID](https://openalex.org/A5000172328)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SNLA框架，模型LLM代理群体的真实影响力，研究有限注意力宽度如何决定群体是集体智慧还是羊群。

**💡 创新点**

将可见曝光图与实际影响力分离，定义由社会权重和有限注意力决定的实现影响矩阵；引入基于Sinkhorn的去中心化定价协议消除高权重源导致的羊群；证明注意力宽度控制有效样本量并能恢复智慧。

**🔧 技术方法**

社会网络理论、DeGroot/Friedkin–Johnsen模型、Katz-Bonacich中心性、温度软最大化、Sinkhorn迭代、LLM事件日志等。

**📊 数据集**

HiddenBench、Werewolf、AgentsNet、Debate、GovSim等多智能体LLM基准。

**📈 对比分析**

与Baseline、Equalizer、Control三种策略对比；在β窄时准确率显著下降，宽β恢复集体智慧；Equalizer显著消除羊群，在线定价逐轮提升；HiddenBench准确率提升0.64，Werewolf提升0.61，AgentsNet冲突数显著下降。

**⚠️ 局限性**

需保证锚定参数λ<1，理论桥接基于近似最佳响应，异构能力或极端网络结构下鲁棒性有限，且定价与温度调控增加计算开销。

---

## 34. ClinOCR-Bench: A Comprehensive Clinical Scanned Document Dataset for Optical Character Recognition Model Evaluation

**arXiv ID:** 2607.03650 | [PDF](https://arxiv.org/pdf/2607.03650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 35. From Geometric Labels to Semantic Understanding of Indoor Building Components Using Multimodal Large Language Models

**arXiv ID:** 2607.03661 | [PDF](https://arxiv.org/pdf/2607.03661v1)

**作者:** Shuju Jing `[一作]` (Shandong University), Chao Yin `[通讯]` (Guangdong Academy of Sciences)

**通讯引用:** 2245 | [OpenAlex ID](https://openalex.org/A5050611071)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Building-MLLM，一种以点云为核心的多模态大语言模型，用于室内建筑组件的识别、描述和工程问答；

**💡 创新点**

创新点包括：①三重约束对齐机制（PIE、GPR、固定文本前缀）实现域特定几何语义融合；②多维LoRA优化策略在LLM层面实现多任务平衡；③基于ontology与多视角过滤的进阶指令生成引擎构建首个室内组件点云‑文本指令跟随数据集；

**🔧 技术方法**

采用PointLLM作为基座，加入PIE、GPR、固定前缀；使用LLaMA-7B并在第二阶段进行多维LoRA微调；点云编码通过PointBERT；评估时结合BLEU、ROUGE、METEOR、Sentence‑BERT、SimCSE及GPT‑4语义评估；

**📊 数据集**

构建了4,198个点云实例，47个类别，37,782条指令跟随数据（包含识别、复杂说明、工程问答）；对外使用Pipework、ScanObjectNN、S3DIS等公开点云数据做迁移测试；

**📈 对比分析**

与现有SOTA多模态模型（InstrucBLIP、LLaVA、3D‑LLM、MiniGPT‑3D、ShapeLLM、PointLLM）对比，Building‑MLLM在Simple Recognition上达88.00%、Complex Captioning 65.10%、Multi‑Engineering QA 68.14%，比基线提升约7–5个百分点；迁移到真实点云时，整体分数提升36–38%，表现更稳定；

**⚠️ 局限性**

局限性包括：①仅在合成点云训练，真实数据泛化仍有限；②模型规模仍大（LLaMA‑7B），部署成本高；③对极端噪声或稀疏扫描的鲁棒性不足；④需要进一步扩充真实标注数据和更细粒度的评测框架。

---

## 36. ViPo-MLLM: Visual-Pose Multimodal LLM for Gloss-Free Sign Language Translation

**arXiv ID:** 2607.03657 | [PDF](https://arxiv.org/pdf/2607.03657v1)

**作者:** Ahmed Abul Hasanaath `[一作]`, Hamzah Luqman `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 1057 | [OpenAlex ID](https://openalex.org/A5060980472)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ViPo-MLLM 框架，将时空 RGB 特征与姿态特征融合，实现无 gloss 的手语翻译。

**💡 创新点**

创新点在于跨模态时序注意力融合、结构化提示与对比学习相结合的多模态 LLM 方法。

**🔧 技术方法**

使用 USTM、OpenPose、跨模态注意力、LLM（mT0-XL+LoRA）、对比损失与交叉熵等技术。

**📊 数据集**

在 PHOENIX14T（德语）和 CSL‑Daily（中文）两大手语数据集上进行实验。

**📈 对比分析**

与现有无 gloss 及 gloss‑based 方法对比，ViPo-MLLM 在 BLEU‑4/ROUGE‑L 上均达成 SOTA 或接近 gloss‑based 的表现。

**⚠️ 局限性**

局限包括对姿态检测精度依赖、计算成本高、仅在两种语言数据集验证，缺乏跨语言泛化与实时性评估。

---

## 37. ThreatVisionAI: A Hybrid CNN-ViT Framework for Image-Based Malware Classification

**arXiv ID:** 2607.03653 | [PDF](https://arxiv.org/pdf/2607.03653v1)

**作者:** Allyson Taylor `[一作]` (University of North Carolina), Prashanth BusiReddyGari `[通讯]` (University of North Carolina)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了 ThreatVisionAI 框架，利用原始图像 CNN、波形变换 CNN 与 Vision Transformer 三分支并通过无参数加权投票实现恶意软件家族分类。

**💡 创新点**

创新点在于引入专门的频域 CNN 分支，将频域特征与空间特征和全局自注意力特征融合，显著提升对视觉相似家族的区分能力。

**🔧 技术方法**

使用 ResNet-18 作为原始与波形 CNN 的骨干，Haar 小波变换提取频域信息，ViT‑Tiny（预训练）提供全局自注意力，Grad‑CAM 进行可解释性分析，FGSM 评估鲁棒性。

**📊 数据集**

在 Malimg 9,465 张 25 个恶意软件家族的灰度图像数据集上进行训练与测试。

**📈 对比分析**

与单一 CNN、波形 CNN、ViT 以及二分支集成对比，Wavelet CNN 单独获得 0.9791/0.9733 的准确率/加权 F1，三分支混合模型在测试集上达到 98.01% 准确率，权重 F1 0.9742，优于现有单分支和两分支方法。

**⚠️ 局限性**

局限性包括仅在 2011 年的 Malimg 数据集上验证，未评估对更大或更新型恶意软件数据集的泛化；在白盒 FGSM 攻击下易受攻击；缺乏对更强攻击、数据漂移以及多模态行为信息的评估。

---

## 38. Crypto-Microeconomics: The Distribution of Bitcoin Wealth Among Diverse Economic Agents

**arXiv ID:** 2607.03646 | [PDF](https://arxiv.org/pdf/2607.03646v1)

**作者:** Syed Azhar Hussain `[一作]` (Munster Technological University), Mubashir Husain Rehmani `[通讯]` (Munster Technological University)

**通讯引用:** 10850 | [OpenAlex ID](https://openalex.org/A5047501301)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过构建Crypto‑Microeconomic Observability Framework，研究了比特币在服务、滥用、恶意软件、个人和善意五类经济主体中的财富分布及其不平等。

**💡 创新点**

创新点在于将宏观指标与微观分类相结合，采用多输入聚类识别实体，并用Top‑K与Gini等不平等度量在微观层面揭示鲸鱼效应。

**🔧 技术方法**

技术手段包括本地比特币节点+BitSQL提取全账本，BlockSci多输入聚类，构建ETL管道，计算Top‑K与Gini，并绘制时间序列分析。

**📊 数据集**

使用截至2025‑07‑13的完整比特币区块链，标签化后共223,558个实体，其中8,450个正余额，涵盖服务、滥用、恶意软件、个人、善意五类。

**📈 对比分析**

通过比较各类主体的Gini、Top‑1%、Top‑5%和Top‑20%占比以及时间演化，发现服务与滥用主体财富高度集中，个人类近似单一鲸鱼；方法能够有效捕捉微观不平等，性能表现为极高的集中度（如个体类Gini 0.9993）。

**⚠️ 局限性**

局限性包括仅使用公开链上信息，聚类和标签依赖预设规则，无法完全消除伪匿名性导致的实体误分；样本偏向正余额实体，零余额实体被排除，可能低估整体分布多样性。

---

## 39. GRASP: Graph-Reasoning Aided Survey Planning for High-Fidelity Related Work Generation

**arXiv ID:** 2607.03709 | [PDF](https://arxiv.org/pdf/2607.03709v1)

**作者:** Haoming Li `[一作]` (University of Texas at Dallas), Jessica Ouyang `[通讯]` (University of Texas at Dallas)

**通讯引用:** 230 | [OpenAlex ID](https://openalex.org/A5006011780)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了 GRASP 框架，通过将 LLM 规划与图算法相结合，自动生成高质量的相关工作章节（RWS）；

**💡 创新点**

创新点包括：① 两层图结构（Graph-of-Thoughts 与 Argument‑Counterargument Planning Network）分别捕捉细粒度内容与高层对话关系；② 使用 Steiner 树进行拓扑感知剪枝，显著减少噪声与冗余；③ 将图结构以 JSON 形式注入 LLM 生成流程，实现结构化、可追溯的写作指导；

**🔧 技术方法**

采用技术包括：大型语言模型（GPT‑4o‑mini）与 Chain‑of‑Thought 提取思路，Graph‑of‑Thoughts 构建与 consensus 节点合并，Argument‑Counterargument Planning Network 进行论文间关系分类，Steiner‑tree 近似算法进行图剪枝，三阶段写作（完整草稿、压缩、融合）以及后处理以统一引用格式；

**📊 数据集**

实验使用 OARelatedWork 测试集（约 1,878 篇论文，来源于 CORE 与 S2ORC），对其中 1,350 篇进行评估，并将目标 RWS 的引用文本完整预置；

**📈 对比分析**

与 L&O、Select‑Read‑Write、No‑graph、Direct 生成等基线在传统文本指标（ROUGE、BERTScore、BLEU、METEOR）和基于引用分析的四维度指标（discourse role、citation importance、citation intent、citation co‑occurrence）进行对比。GRASP（裁剪版）在所有指标上均优于基线，尤其在语义一致性、引用重要性识别、引用意图捕捉和引用分组/排序上表现突出；

**⚠️ 局限性**

局限性包括：① 需完整的引用论文文本，实际应用需自行抓取与预处理；② 构建 GoT 与 ACPN 的计算成本高，且两者不能并行；③ consensus 节点稀疏，过度合并会导致信息损失；④ 采用离散主题聚类可能忽略跨主题关联；⑤ 未检测抄袭与事实错误，生成文本仍可能包含错误信息。

---

## 40. Phase-Preserving Trimodal Transformer for Tropical Forest Biomass Estimation Using Optical and PolInSAR Data

**arXiv ID:** 2607.03663 | [PDF](https://arxiv.org/pdf/2607.03663v1)

**作者:** Luiz Felipe Parente Santiago `[一作]` (Federal University of Amazonas), Felipe Ferrari `[通讯]` (Military Institute of Engineering)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种利用光学与多极化SAR相位信息的Trimodal Coherent Co‑attention Transformer（TCCT）进行热带雨林上层生物量估算。

**💡 创新点**

创新点在于：①使用复数卷积保持SAR相位相干性；②引入动态多模共注意力机制自适应抑制云覆盖的光学信号；③结合物理模型（RVOG）设计混合损失；④通过局部空间全尺度校准实现高精度AGB映射。

**🔧 技术方法**

主要技术包括复数卷积与ModReLU激活、Transformer式共注意力、U‑Net解码器、Levenberg‑Marquardt非线性优化、混合损失（MSE+RVOG）。

**📊 数据集**

使用Paracou地区的30m分辨率Landsat‑5光学影像、L‑band和P‑band PolInSAR数据以及对应的LiDAR CHM/AGB实测数据。

**📈 对比分析**

与Vision Transformer、CNN（早期融合）和随机森林基线在同一三模数据上对比，TCCT在5‑fold交叉验证后实现CHM RMSE 3.78 m、R² = 0.33；转换为AGB后rRMSE 4.51%，远低于ESA BIOMASS 20% 误差阈值。

**⚠️ 局限性**

局限性包括：①对单一地区（Paracou）验证，未评估跨地区泛化；②R²仍偏低，说明像素级误差大；③模型计算量大，需复数运算支持；④仅在30 m分辨率下验证，未探讨更细尺度或多时序扩展。

---

## 41. A Structural Interpretation of GELU and Threshold-Transmission Activations via the First-Order Loss Function

**arXiv ID:** 2607.03664 | [PDF](https://arxiv.org/pdf/2607.03664v1)

**作者:** Roberto Rossi `[一作]` (University of Edinburgh), Roberto Rossi `[通讯]` (University of Edinburgh)

**通讯引用:** 2318 | [OpenAlex ID](https://openalex.org/A5056867872)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并研究了基于阈值传输的激活函数族，并将GELU解释为高斯阈值的期望输出；

**💡 创新点**

将激活函数映射到“阈值传输”框架，揭示GELU、ReLU、SiLU/Swish、ULEU（hard swish）等均可视为不同阈值分布的实现，并通过可学习的阈宽β引入了新的ULEU/TUELU变体；

**🔧 技术方法**

利用高斯一阶损失函数分解、阈值随机门理论、最大熵阈值分布选择以及在卷积/Transformer模型中实现可学习阈宽的自适应激活；

**📊 数据集**

在CIFAR‑100（MLP‑Mixer与Vision Transformer）、Tiny Shakespeare（字符级 GPT）、TinyStories 与 WikiText‑2（token‑level GPT）等图像与文本数据集上进行实验；

**📈 对比分析**

通过与ReLU、SiLU/Swish、GELU、Hard Swish等基线进行对比，结果显示在四个实验中ULEU/TUELU的性能优于或与GELU持平，尤其是通过学习阈宽得到的TUELU表现最优；

**⚠️ 局限性**

实验规模有限，未涵盖大规模训练与多种网络架构；硬阈宽β=3常过宽，需进一步研究每层/通道级阈宽以及更大规模的评估。

---

## 42. AutoCedar: An Agentic Framework for Verifier-Guided Access Control Policy Synthesis

**arXiv ID:** 2607.03656 | [PDF](https://arxiv.org/pdf/2607.03656v1)

**作者:** Adarsh Vatsa `[一作]` (Stevens Institute of Technology), William Eiers `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 92 | [OpenAlex ID](https://openalex.org/A5088950037)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于验证器的 LLM 访问控制策略合成框架 “Verifier-Guided Policy Synthesis（VGP）”，先把自然语言需求拆分为可审计的意图原子并构造审阅后的语义边界（楼层、天花板、活性切片），然后让 LLM 在此固定目标下迭代生成 Cedar 策略，验证器对每个候选进行符号检查并通过信号层反馈具体的修复方向，最终得到满足全部检查的可部署 Cedar 策略；

**💡 创新点**

核心创新点包括：①把意图拆解成可审计的原子并构造可验证的边界，解决需求不完整的问题；②引入“信号层”把验证器产生的判定结果（如过度许可、缺失许可、流程丢失）转化为可被 LLM 理解且不改变目标的修复指令；③采用 CEGIS 风格的迭代搜索，使随机生成的 LLM 能在全局语义约束下收敛；④公开 CedarBench 基准和三个真实世界需求集合，验证框架的可迁移性和有效性；

**🔧 技术方法**

技术实现主要依赖：Llama/ChatGPT 等大型语言模型（GPT‑5.5 low、Haiku 4.5）；Cedar 语言及其 SMT 编译器做符号验证；CedarBench 的正式检查计划（楼层/天花板/活性切片）；信号层实现的故障类型识别与方向指示；对需求进行源图（G_D）拆解与原子审计；以及自动化的 LLM‑Verifier‑Signal 循环；

**📊 数据集**

使用的数据集包括：221 个 CedarBench 场景（涵盖 GitHub、云文档、酒店、销售、流媒体、标签‑角色、税务、临床数据等领域），以及从 iTrust、CyberChair、IBM 课程注册系统中抽取的 401/303/471 条自然语言访问控制语句的三份真实需求集合；

**📈 对比分析**

实验中将 VGP 与直接 LLM 生成策略（无目标）进行对比，指标包括：验证通过率、属性检查通过率、语义请求匹配率、人工偏好；结果显示 VGP 在所有 221 场景上 100% 收敛，平均迭代 1.67/2.51 次，token 数 9,084/12,397，成本约 $0.04/$0.018；在信号层消融实验中，完整信号层能将残留错误降至 0，验证通过率提升至 100%；相比之下，直接 LLM 在属性检查和语义请求上的成功率仅为 1–3%，人工偏好也显著落后；

**⚠️ 局限性**

主要局限：①验证目标的准确性完全依赖人工审计，一旦原子错误或遗漏，系统只能验证错误目标；②符号检查主要覆盖静态请求集合，可能无法捕捉复杂时间/上下文依赖的边缘案例；③模型的随机性导致需多次重跑才能获得稳定结果，且对大规模实体图的性能尚未充分评估；④信号层仅覆盖预定义的错误类型，未能处理更复杂的逻辑冲突；⑤实验集中在 CedarBench 与三份语料，实际部署环境的多样性和规模仍待验证；

---

## 43. Graph-Aware Fuzzing for Graph Database Management Systems

**arXiv ID:** 2607.03741 | [PDF](https://arxiv.org/pdf/2607.03741v1)

**作者:** Yu Li `[一作]` (Tianjin University), Yongqiang Lyu `[通讯]` (Tianjin University)

**通讯引用:** 1329 | [OpenAlex ID](https://openalex.org/A5058507537)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种针对图数据库管理系统的黑盒模糊测试框架GRAF，用来发现运行时错误

**💡 创新点**

创新点在于将查询结构与图上下文解耦，利用LLM生成多样化语法骨架，再通过级联依赖解析注入合法的图元；同时采用执行状态（耗时、结果大小、系统状态）反馈驱动变异，摆脱传统覆盖反馈

**🔧 技术方法**

使用大型语言模型（如Gemini）生成查询骨架，LLM+上下文解析实现实例化，五种图特定变异算子，执行监控收集反馈，黑盒交互方式

**📊 数据集**

使用六种主流图数据库（Neo4j、Memgraph、RedisGraph、NebulaGraph、FalkorDB、KuzuDB）以及各自初始化的图数据集，未公开具体规模但覆盖多种图拓扑

**📈 对比分析**

与三种基线（Dinkel、BUZZBEE、AFL++）在相同资源下对比，GRAF在12小时内覆盖率提升31.6%–41.1%，发现34个未知bug（23已CVE），基线仅触发4–7个

**⚠️ 局限性**

局限性包括：仅关注运行时崩溃/内存问题，未覆盖逻辑错误；对LLM生成骨架的质量依赖模型；在高度动态或极大规模图上性能与稳定性待进一步验证

---

## 44. Fault Detection and Explainable Classification in Automotive HIL Validation via Denoising Autoencoders and In-Context Large Language Models

**arXiv ID:** 2607.03734 | [PDF](https://arxiv.org/pdf/2607.03734v1)

**作者:** Mohammad Abboush `[一作]`, Andreas Rausch `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

无法确定

**💡 创新点**

无法确定

**🔧 技术方法**

无法确定

**📊 数据集**

无法确定

**📈 对比分析**

无法确定

**⚠️ 局限性**

无法确定

---

## 45. ProxyUp: Training-Free Proxy-Conditioned Video Generation for Controllable Dynamics

**arXiv ID:** 2607.03732 | [PDF](https://arxiv.org/pdf/2607.03732v1)

**作者:** Zanwei Zhou `[一作]` (Shanghai Jiao Tong University), Qi Tian `[通讯]` (Huawei Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了代理条件视频生成框架ProxyUp，利用代理视频提供的运动先验并结合文本提示，生成具有可控动态且内容可变的新视频。

**💡 创新点**

创新点在于（1）训练无关的代理条件控制方式；（2）区域化潜变量噪声保留运动核心，注入噪声生成背景；（3）随机流松弛（SFR）逐步将手工构造的潜变量投射到模型学习的分布上，实现运动保留与内容生成的平衡。

**🔧 技术方法**

核心技术包括：基于Rectified Flow的ODE逆向潜变量获取；区域化潜变量噪声（Region‑wise Latent Noising）；多轮随机流松弛（Stochastic Flow Relaxation）；最终ODE采样生成视频。

**📊 数据集**

使用自建的代理条件生成数据集，包含76段代理视频，分为物理仿真视频和真实视频两部分，用于评估动态保真、交互合理性、材质一致性等指标。

**📈 对比分析**

与Wan2.2、VACE、DiTFlow、SDEdit、FlowDirector等编辑/传输/仿真基线进行对比，ProxyUp在动态一致性（MR、Mech.）和视觉质量（IQ、Mat.）上均取得最高分，尤其在物理交互合理性上显著优于其他方法。

**⚠️ 局限性**

局限性包括：对代理视频质量高度依赖；无法超越预训练生成器的物理或语义知识；在代理包含不常见或模糊力学信息时，生成结果可能不准确；训练无关的设计限制了对更复杂交互的建模能力。

---

## 46. SelfMem: Self-Optimizing Memory for AI Agents

**arXiv ID:** 2607.03726 | [PDF](https://arxiv.org/pdf/2607.03726v1)

**作者:** Shu Yang `[一作]` (King Abdullah University of Science and Technology), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 485171 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种自我优化的记忆框架，允许LLM代理通过工具和反馈自主探索、构建、更新与检索记忆，而非遵循固定的记忆策略。

**💡 创新点**

创新点在于将记忆管理转化为模型可控的优化过程，提供可读写工具和多维反馈，让代理在语言层面学习最优的记忆组织与使用策略。

**🔧 技术方法**

核心技术包括自适应工具调用循环、读写/审查记忆工作空间、基于语言的策略生成与迭代优化，以及多信号反馈机制（token用量、延迟、检索质量等）。

**📊 数据集**

实验使用BEAM长对话基准，覆盖100K、500K和1M token的对话历史，共计100个对话和2000个探测性问答，用GPT‑5.4‑nano作为生成模型。

**📈 对比分析**

与RAG、Full‑Context、Compression、LoCoMo、MemoryBank、ReadAgent、MemGPT、A‑Mem和Mem0等基线相比，所提出方法在所有尺度上均取得最高Score和Pass_0.5（提升约0.13–0.17），并以约$2/问答的成本实现优异性能。

**⚠️ 局限性**

局限性主要体现在仅在BEAM基准下验证，未测试其他长时限任务或不同模型，且策略优化实验仅在100K尺度进行，未来需扩展至更大尺度和多样化场景。

---

## 47. Explainable Reinforcement Learning for Adaptive Traffic Signal Control

**arXiv ID:** 2607.03703 | [PDF](https://arxiv.org/pdf/2607.03703v1)

**作者:** Dickens Kwesiga `[一作]` (Georgia Institute of Technology), Michael Hunter `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 6775 | [OpenAlex ID](https://openalex.org/A5052461413)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种可解释的实体中心强化学习框架，用于自适应交通信号控制，并通过层次化注意力机制实现对交叉口各车道与信号相位之间关系的实时可视化。

**💡 创新点**

创新点在于：①将交叉口状态拆分为车道实体和相位实体，保留拓扑结构；②使用双阶段注意力（跨相位注意和自注意）生成可解释的亲和矩阵；③将确定性行动掩蔽直接嵌入PPO，保证合法的相位切换。

**🔧 技术方法**

技术手段包括：实体嵌入、高维投影、跨相位多头注意力、车道自注意力、动作掩蔽、Proximal Policy Optimization（PPO）以及SUMO微观仿真环境。

**📊 数据集**

实验使用了在SUMO中构建的四路交叉口，训练了三种不同交通需求模式（A、B、C）并在两种未见过的流量场景下进行评估。

**📈 对比分析**

与传统最优Actuated Signal Control（ASC）以及无注意力的RL模型（RL-NoAtt）比较，所提框架在两种交通需求场景下均实现了与ASC相当或更低的车辆延时，并且在高度不平衡流量时的表现明显优于RL-NoAtt。

**⚠️ 局限性**

局限性包括：①仅在仿真环境中验证，缺乏真实道路部署验证；②对大规模网络的可扩展性和计算开销尚未评估；③依赖仿真生成的数据，若实际感知数据噪声或不完整，模型性能可能受影响。

---

## 48. Function-Correcting Codes for Sum-Rank Metric

**arXiv ID:** 2607.03857 | [PDF](https://arxiv.org/pdf/2607.03857v1)

**作者:** Santhi Kumari Kammila `[一作]` (Indian Institute of Science), B. Sundar Rajan `[通讯]` (Indian Institute of Science)

**通讯引用:** 6707 | [OpenAlex ID](https://openalex.org/A5015398340)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在sum‑rank度量下的功能纠错码（FCC），并给出了冗余的上下界、Plotkin型下界以及针对局部二进制函数和sum‑rank权重函数的显式构造，证明其冗余最优；

**💡 创新点**

创新点在于将FCC框架推广到sum‑rank度量，提出了新的不规则sum‑rank距离码的Plotkin型下界，并给出了实现最优冗余的具体构造；

**🔧 技术方法**

采用了sum‑rank度量理论、矩阵秩码、线性代数工具以及不规则距离码的组合方法；

**📊 数据集**

本工作为纯理论研究，没有使用具体数据集；

**📈 对比分析**

通过理论证明和对比已知的Singleton/Plotkin界限，显示所构造的码在冗余方面达到了最优或接近最优；

**⚠️ 局限性**

局限性包括仅针对特定函数类（局部二进制、权重函数）给出最优构造，且缺乏实验验证，未讨论更一般函数的构造与性能。

---

## 49. CogRad: A Cognitively-Inspired Multi-Agent Framework for Radiology Report Generation

**arXiv ID:** 2607.03853 | [PDF](https://arxiv.org/pdf/2607.03853v1)

**作者:** Saif Ur Rehman Khan `[一作]` (Rhineland-Palatinate Technical University of Kaiserslautern-Landau), Muhammad Nabeel Asim `[通讯]` (Rhineland-Palatinate Technical University of Kaiserslautern-Landau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于四个阶段（全局筛查、聚焦探查、结构化写作、可视化验证）的多智能体框架CogRad，用于从胸部X光图像生成放射科报告。

**💡 创新点**

创新点包括：①采用槽注意力（slot‑attention）自动发现解剖区域并进行疾病层面triage；②将检验、写作、验证的视觉表示在四个阶段连续传递，避免信息丢失；③在写作阶段使用疾病门控视觉前缀为LLM提供明确提示；④在训练时加入视觉蕴涵（visual‑entailment）损失，并在推理时实现句子级自检与必要时的重新生成，模仿放射科医生的自我校对。

**🔧 技术方法**

使用的技术：Swin Transformer作为共享视觉编码器；LLaMA‑2‑7B（可通过LoRA微调）作为语言模型；槽注意力、MHA、GRU、MLP、层归一化；视觉蕴涵损失与句子置信度头；Grad‑CAM可视化；混合精度训练、AdamW优化。

**📊 数据集**

使用的数据集：CheXpert Plus（约22.3万对，包含14种疾病标签和报告）和IU X‑Ray（约3.9k对，短文本报告）。

**📈 对比分析**

与现有单通道生成方法（如R2GenCSR、R2GenGPT、AM‑MRG等）比较，CogRad在BLEU‑4、ROUGE‑L、CIDEr等NLG指标上均领先，CheXpert Plus BLEU‑4 0.316、CIDEr 0.322，IU X‑Ray BLEU‑4 0.201、CIDEr 0.724；在ablation实验中去除Scout或Verifier会导致性能下降，证明两者互补；在临床准确性指标（RadGraph F1、CheXbert F1、幻觉率）表现不均，提示NLG指标与实体级准确性不完全同步。

**⚠️ 局限性**

局限性：①NLG指标高但RadGraph F1等实体级指标仍偏低，说明生成文本与图像事实不完全一致；②幻觉率仍高，尤其在高频疾病和设备术语上；③自检循环虽提高准确性但在推理时增加额外成本；④目前只在句子级进行自检，无法精确定位单个实体的错误；⑤模型性能受数据集分布影响，在不同病例群组表现差异显著。

---

## 50. Adversarial LassoNet: Robust Feature Selection via Stability-Driven Sparse Learning

**arXiv ID:** 2607.03839 | [PDF](https://arxiv.org/pdf/2607.03839v1)

**作者:** Zhen Huang `[一作]`, Yulong Zheng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Adversarial LassoNet (AdLNet)，将局部输入空间对抗扰动与 LassoNet 的层次稀疏机制结合，用对抗稳定性来引导特征选择。

**💡 创新点**

创新点包括：①将对抗敏感度作为稳定性信号融入稀疏学习；②在保持 LassoNet 结构的前提下，使用一阶近似高效计算最坏扰动；③提供 NTK 有效秩分析解释对抗训练提升稀疏学习的梯度分布；④通过混合目标同时兼顾预测精度与局部稳定性。

**🔧 技术方法**

核心技术：LassoNet 层次稀疏优化、输入空间对抗训练（对抗扰动一阶近似）、混合损失（α 调节预测/稳定性）、NTK 有效秩与 Hessian 诊断、梯度裁剪等。

**📊 数据集**

实验数据集：高维 SERS 肺癌筛查数据、六个公开基准（MNIST、MNIST‑Fashion、ISOLET、COIL‑20、Activity、Mice Protein）、ColoredMNIST。

**📈 对比分析**

与 vanilla LassoNet、FISTA‑Net、Deep‑Lasso 等基线对比；在 SERS 数据上测试准确率、敏感度/特异度/AUC，得到 5.3% 准确率提升和 6% AUC 提升；在 ColoredMNIST 上 OOD 准确率提升 4.4%，特征支持可重复性提升 6.3%；在公共基准上保持或略优于基线（MNIST、COIL‑20、Activity 等）。

**⚠️ 局限性**

局限性：①对抗混合系数 α 需要经验调参，过大或过小会导致欠拟合或失去预测性能；②在极端稀疏限制下可能去除部分有预测价值但易受扰动的特征；③仅验证了 LassoNet 结构，未探索其他稀疏范式（权重稀疏、注意力稀疏等）；④对抗扰动采用局部一阶近似，可能在非光滑模型或大扰动范围下失效；⑤真实场景的分布迁移类型与 ColoredMNIST 的简化场景仍有差距。

---

## 51. Rethinking Depth Pruning for Vision Transformers: A Heterogeneity-Aware Perspective

**arXiv ID:** 2607.03784 | [PDF](https://arxiv.org/pdf/2607.03784v1)

**作者:** Zhenfeng Su `[一作]` (Chinese University of Hong Kong), Wenxuan Wang `[通讯]` (Renmin University of China)

**通讯引用:** 2055 | [OpenAlex ID](https://openalex.org/A5100755181)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种异质感知的深度裁剪框架 HetDPT，用于在保持准确度的前提下显著加速 Vision Transformers (ViTs)。

**💡 创新点**

创新点在于识别并解决层间异质性（注意力层与激活层的梯度差异与恢复不对称），通过模型准确性预测 (MAP) 进行层数分配，并仅在同类层内进行重要性评估；同时实现激活层裁剪与线性层合并，避免维度不匹配。

**🔧 技术方法**

使用两阶段剪枝流程：① MAP 较快的增量剪枝+梯度传递学习重要性；② 细化层裁剪、快速微调与自蒸馏；随后合并相邻线性层以提升推理速度。

**📊 数据集**

在 ImageNet-1K、CIFAR-100、COCO2017、ADE20K 四大公开数据集上进行评估，并在 DINO‑V2‑Giant 等大模型上验证可扩展性。

**📈 对比分析**

与现有仅注意力裁剪、宽度裁剪、联合宽度深度裁剪以及 Isomorphic Pruning 等方法比较，HetDPT 在 DeiT‑B/DeiT‑S 等模型上实现 1.58×/1.39× 的速度提升且准确度保持不降；在极端压缩场景下，HetDPT+ 进一步将速度提升至 9.19×，并在相同准确率下显著优于 Isomorphic Pruning。

**⚠️ 局限性**

局限性包括 MAP 需要针对每个网络家族重新拟合，且方法尚未验证在语言模型或其他模态中的可迁移性。

---

## 52. Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks

**arXiv ID:** 2607.03798 | [PDF](https://arxiv.org/pdf/2607.03798v1)

**作者:** Yoshihiro Maruyama `[一作]` (Kyoto University), Yoshihiro Maruyama `[通讯]` (Kyoto University)

**通讯引用:** 2557 | [OpenAlex ID](https://openalex.org/A5103173777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了统一图网络与层析网络的秩同构神经网络（OENN）框架，提供线性层的完整分类与非线性实现，并证明了连续秩同构映射的普适逼近定理，随后推广到更一般的类别同构网络（CENN）

**💡 创新点**

创新点在于：①首次提出秩同构网络概念并给出线性层的全局参数化；②证明了OENN（以及CENN）在紧致G-不变集合上对连续秩同构映射的普适逼近；③将图神经网络、层析网络与更一般的类别同构模型统一在等变束与传输理论下

**🔧 技术方法**

利用等变束、传输定律、Reynolds块、关系消息传递、对称聚合等技术构建线性与非线性OENN层，进一步延伸至类别同构网络框架

**📊 数据集**

论文主要以理论证明为主，未给出具体实验数据集

**📈 对比分析**

比较方法主要是与传统图神经网络、深度集合和层析网络等同构网络在理论层面的可逼近能力进行对比；未给出数值实验或性能指标

**⚠️ 局限性**

局限性：实现与实现复杂度较高，局部信息传递受限于Hasse图直径；对非可逆或高阶对称的处理仍有待深入；缺乏实证实验验证

---

## 53. Stable Global Weighting of Flow Mixtures using Simplex Exponential Moving Average

**arXiv ID:** 2607.03809 | [PDF](https://arxiv.org/pdf/2607.03809v1)

**作者:** Benjamin Wiriyapong `[一作]` (Cardiff University), Kirill Sidorov `[通讯]` (Cardiff University)

**通讯引用:** 292 | [OpenAlex ID](https://openalex.org/A5103784688)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种两阶段混合流变分推断框架 AMFVIsEMA，通过先独立训练不同流专家，再使用基于责任的简单指数移动平均（sEMA）在验证集上更新全局混合权重。

**💡 创新点**

将专家专门化与权重更新分离，采用无梯度、无样本级门控的概率单纯形 EMA 机制实现稳定权重学习。

**🔧 技术方法**

使用 RealNVP、MAF、RBIG 等正则化流作为专家；利用温度控制的 softmax 计算责任；在概率单纯形上做 EMA；评估 NLL、KL、Wasserstein‑2、MMD 等指标。

**📊 数据集**

十个后验基准：六个二维合成分布（Banana、X‑Shaped、Bimodal、Multimodal、Two‑moons、Rings）以及四个低维真实贝叶斯模型（BLR、BPR、Weibull、Real‑GMM2）。

**📈 对比分析**

与单流基线（RealNVP、MAF、RBIG、NICE、ResFlow、EMMix）以及原 AMFVI 对比，AMFVIsEMA 在 NLL、Wasserstein、MMD 上持续领先或相当，避免了单流崩溃，并保持有效专家数 N_eff>1.4，性能稳定。

**⚠️ 局限性**

仅使用全局权重，未考虑输入依赖的门控；对高维或约束后验尚未评估；依赖验证集大小与温度调参。

---

## 54. Fully Scalable MPC Algorithms for WSPD in Doubling and Euclidean Spaces

**arXiv ID:** 2607.03811 | [PDF](https://arxiv.org/pdf/2607.03811v1)

**作者:** Eunjin Oh `[一作]` (Pohang University of Science and Technology), Hyeonjun Shin `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 52 | [OpenAlex ID](https://openalex.org/A5057819201)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种在MPC模型下构造 (1/ε)-WSPD 的 O(1) 轮算法，并将该分解用于构造稀疏网络、近似直径、最近点对和所有 k‑近邻等几何近似问题。

**💡 创新点**

创新点在于引入分区覆盖树（partition cover tree）实现对任意常数倍放缩维度的度量空间的快速分层构造；利用半径扰动和 α‑逼近技术保证树具备“小覆盖”性质，从而在 O(1) 轮内完成 WSPD；在欧几里得空间进一步利用压缩四叉树（compressed quadtree）实现线性大小的 WSPD。

**🔧 技术方法**

核心技术包括：α‑逼近（α‑approximation）用于快速采样并构造层级树；半径扰动（radius perturbation）保证分割均衡；分区覆盖树与压缩四叉树提供高效的层次化空间划分；并行对节点对枚举与候选对筛选以实现 O(1) 轮内的 WSPD 构造；随后应用已有的 WSPD 基于近似/最优算法得到稀疏图、近似直径、最近点对与 k‑近邻。

**📊 数据集**

实验与评估基于理论分析，并未给出具体数值数据集；所讨论的数据模型为任意大小 n 的点集，工作在具有常数双重维度的度量空间或常数维欧几里得空间 ℝ^d（d 为常数）。

**📈 对比分析**

相较于 FOCS'93 论文的 O(log n) 轮、仅适用于欧几里得空间的 WSPD，本文实现了 O(1) 轮、可扩展至双重维度度量空间的算法；WSPD 大小从 O(n log n) 减少到 O((1/ε)^O(d) n)（欧氏）或 O((1/ε)^O(Δ) n log^2 n)（双重维度）；总空间线性，局部空间 O(n^δ)（δ∈(0,1))，并且在高维欧氏空间中可在 O(1) 轮内完成近似直径、最近点对、稀疏网络的构造。

**⚠️ 局限性**

局限性包括：算法为随机化，仅以常数成功概率完成（可通过 O(log n) 乘法提升到 1‑1/n^Ω(1)）；对 k‑近邻算法仅适用于欧几里得空间；在双重维度空间中实现的 WSPD 仍比最优线性规模略大（乘以 log^2 n）；以及在极端高维或极端分布的点集中可能需要更大的常数因子。

---

## 55. Sparse-View Surface Reconstruction using Gaussian Splatting through High-Confidence Depth Propagation with Normal Priors

**arXiv ID:** 2607.03765 | [PDF](https://arxiv.org/pdf/2607.03765v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 56. Weave: Verified Netlist-to-Schematic Conversion via Layered Graph Layout

**arXiv ID:** 2607.03835 | [PDF](https://arxiv.org/pdf/2607.03835v1)

**作者:** Senol Gulgonul `[一作]` (Ostim Technical University), Senol Gulgonul `[通讯]` (Ostim Technical University)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5060540189)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种 deterministic 的 netlist‑to‑schematic 转换器 Weave，能够将 SPICE netlist 转换为 LTspice 图形，并通过 round‑trip 连接性验证保证转换结果完全正确。

**💡 创新点**

核心创新在于：① 使用 deterministic 的 Sugiyama‑style layered 布局与 Placement Patterns 组合，完全可重复；② 对生成的 schematic 进行即时回推 netlist 并严格比对，提供二值化的正确性证明；③ 设计安全模式梯度（safe‑mode ladder）以在布局失败时自动退化到更简单的模式，始终保证验证通过。

**🔧 技术方法**

技术包括：SPICE 解析器、信号分层布局（elkjs 实现）、Placement Patterns、LTspice 符号 pin 表（5093 个符号）与通用矩形占位、全流程 round‑trip 验证算法、以及安全模式梯度。

**📊 数据集**

主要使用了两套公开数据集：1）Circuits‑LTSpice（117 个电路）与其 LTspice 生成的 netlist；2）Analog Devices 官方 LTspice demo 集（3460 个电路，3610 总电路）。

**📈 对比分析**

与最先进的 LLM 转换器 Schemato 直接对比：在 Circuits‑LTSpice 上 Weave 100% 编译成功且 100% 连接性验证通过；Schemato 仅 76% 编译成功，GED 相似度 0.35。对更大规模的 demo 集，Weave 88.4% 完全验证成功，剩余 8.2% 部分匹配，3.3% 转换失败。

**⚠️ 局限性**

局限主要集中在：1）对稠密多引脚电源模块的布局仍无法完美连接；2）深层级联拓扑会被层式布局拉伸为长链，导致可视化不够紧凑。

---

## 57. EEG-Based Imagined Speech Decoding Using a Hybrid CNN-SNN Architecture

**arXiv ID:** 2607.03844 | [PDF](https://arxiv.org/pdf/2607.03844v1)

**作者:** Fatima Shalhoub `[一作]` (Antonine University), Hoda Fares `[通讯]` (Stanford University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种融合CNN与SNN的混合架构，用于从EEG信号中解码想象语音；

**💡 创新点**

首次将卷积神经网络与脉冲神经网络结合，通过脉冲时序进行分类，提升了对低信噪比EEG的时序特征捕获；

**🔧 技术方法**

采用1D CNN提取时域特征，LIF模型脉冲神经网络，替代梯度反向传播与率编码决策；

**📊 数据集**

使用公开的2020 BCI Competition III 数据集（15名受试者，5个想象词/短语）；

**📈 对比分析**

与多种传统机器学习和深度学习方法对比，模型在同一数据集上实现了80.13% 的准确率，显著优于70.19% 等前沿方法；

**⚠️ 局限性**

仅在单受试者进行训练与测试，缺乏跨受试者验证，实时性能和多受试者泛化性仍待进一步研究。

---

## 58. When Simpler Is Better: Evaluating Translation Pipelines for Medieval Latin Manuscripts

**arXiv ID:** 2607.03836 | [PDF](https://arxiv.org/pdf/2607.03836v1)

**作者:** Nguyen Kim Hai Bui `[一作]` (Eötvös Loránd University), Mufti Mahmud `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 13925 | [OpenAlex ID](https://openalex.org/A5027525633)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了针对中世纪拉丁手稿的端到端图像翻译评估框架，并基于此构建了 Interpres Parallel Corpus (IPC) 数据集。

**💡 创新点**

创新点在于揭示了“专业化差距”与“复杂性悖论”，证明了在历史手稿场景下，专用 OCR 超越大型通用 VLM，且更简单的 OCR→VLM 体系优于多组件系统。

**🔧 技术方法**

主要技术包括领域微调的 TrOCR/ TRIDIS OCR 模型、ByT5 文本纠正、检索增强生成（RAG）以及 GPT‑4o 作为翻译后端。

**📊 数据集**

使用的数据集为 IPC（1,383 行图像-拉丁-英语三元组）和 CATMuS Latin 进行 OCR 评测。

**📈 对比分析**

通过 Fair Arena 共享子集对不同管线进行对比，发现最简管线 P1（OCR+VLM）在 ChrF 上最高，后续添加纠正或 RAG 并未显著提升；同时显示了 OCR 误差对翻译质量的线性负相关。

**⚠️ 局限性**

局限性包括仅针对拉丁手稿、RAG 仅使用静态词典、评测仅基于 GPT‑4o，未验证在其他 VLM 或更大检索语料上的通用性。

---

## 59. NeSy-CSA: A Neuro-Symbolic Framework for Open-Ended Critical Scenario Attribution

**arXiv ID:** 2607.03847 | [PDF](https://arxiv.org/pdf/2607.03847v1)

**作者:** Qitong Chu `[一作]` (Beijing Institute of Technology), Yufeng Yue `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 NeSy-CSA 神经符号框架，实现开放式关键情境归因，并提供可追踪的推理流程与干预评估。

**💡 创新点**

创新点在于：① 通过知识驱动生成、语义去重和数据驱动筛选构建紧凑的因子空间；② 生成并验证前后依赖子任务图，保证推理结构可验证；③ 采用神经符号混合执行，正式子任务由可执行符号程序完成，语义子任务由受前置证据约束的 LLM 推理，兼顾解释性与开放性；④ 引入两级评估（过程级结构有效性与结果级干预效果），从实验与人类专家评判双重维度验证。

**🔧 技术方法**

主要技术包括：大型语言模型（DeepSeek‑V3.2、GPT‑4o‑mini）用于因子生成、子任务规划与神经推理；轻量级原子函数库与动态符号程序构造实现可执行符号推理；统计检验（Welch t‑test、Benjamini–Hochberg、Hedges g）用于因子筛选；依赖图验证与可执行检查；CTR（critical‑to‑non‑critical转化率）与 RIR（reward‑in‑crease‑ratio）等干预指标。

**📊 数据集**

使用四个决策环境的数据集：ACAS_Xu（6000/201 关键）、CoopNavi（2000/211 关键）、BipedalWalker（2000/236 关键）、CARLA（3000/200 关键），每个环境均通过 MDPFuzz 生成多样化情境。

**📈 对比分析**

与三种 LLM 基线（仅 LLM、LLM+工具、LLM+Chain‑of‑Thought）对比，NeSy‑CSA 在四个环境的 CTR 和 RIR 均为最高，平均提升 CTR 18.32%、RIR 13.67%；过程级子任务图精度/召回/F1 均在 80% 以上；一次性子任务图生成比逐样本生成显著降低 token 消耗（>99% 级别），并保持甚至提升干预效果。

**⚠️ 局限性**

局限性包括：① 仍依赖 LLM 的提示与推理质量，可能受模型偏差与幻觉影响；② 过程级评估需要人工专家定义子任务参考集合，难以在大规模或未知环境自动化；③ 对极端动态或多模态复杂场景的可扩展性尚未验证；④ 生成的符号程序受原子函数库限制，若缺失必要原子函数可能无法完成完整推理。

---

## 60. City-Level 3D Surface Reconstruction with Viewpoint Orientation Partitioning and Scene Completion

**arXiv ID:** 2607.03771 | [PDF](https://arxiv.org/pdf/2607.03771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 61. From Region Arrival to Instance-Level Grounding in Vision-and-Language Navigation

**arXiv ID:** 2607.03792 | [PDF](https://arxiv.org/pdf/2607.03792v1)

**作者:** Xiangyu Shi `[一作]` (Adelaide University), Qi Wu `[通讯]` (Adelaide University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了视觉语言导航中“目标定位到可视化”问题，提出了Last-3-Meter Grounding Gap，并用REALM模块、REVERIE-AIM数据集与三项实例级评估指标进行补救。

**💡 创新点**

创新点在于将导航与对象级定位解耦，提供可插拔的短程细化策略、可见性感知停机机制和精细视角评测，弥补传统3米准则的不足。

**🔧 技术方法**

使用了LoRA微调的UniNaVid、可见性感知停机损失、BERT提取目标短语以及OWL-V2目标检测的技术组合。

**📊 数据集**

采用了基于REVERIE/REVERIE-CE扩展而来的REVERIE-AIM数据集，包含对象级终点和约18万条短程训练样本。

**📈 对比分析**

在四种VLN骨干上与多种基线比较，REALM显著提升ONS@0.1m、GS和OracleGS指标；在真实Hello Robot Stretch平台上，成功率从8.33%提升至33.33%。

**⚠️ 局限性**

局限性在于最终定位精度仍低于人类水平，需要进一步提升视角选择、可观测性推理和停机决策能力。

---

## 62. DualView: Preventing Indirect Prompt Injection in Personal AI Agents

**arXiv ID:** 2607.03821 | [PDF](https://arxiv.org/pdf/2607.03821v1)

**作者:** Juhee Kim `[一作]` (Seoul National University), Byoungyoung Lee `[通讯]` (Seoul National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了名为 DualView 的插件，扩展 Dual LLM 模式，在个人 AI 代理的用户环境中维护两视图（代理视图与人类视图），对不可信数据进行符号化并在文件系统、网络、shell 等渠道中同步，以防止即时与存储式间接提示注入攻击，同时保持代理功能与人类可读性。

**💡 创新点**

创新点：
1) 将符号跟踪从代理内部扩展到外部用户环境，构建双视图机制；
2) 自动路由工具调用至对应视图并实时同步文件系统，保证两视图数据一致；
3) 通过符号化/去符号化链实现存储式 IPI 的彻底阻断；
4) 无需改动代理内部逻辑，可直接作为 OpenClaw 等运行时的插件部署。

**🔧 技术方法**

技术手段：Dual LLM 模式、符号化与去符号化、文件系统同步（Git 工作树）、工具钩子插件、数据信任与使用策略、命令模式匹配、网络请求符号化、Shell 与网络工具的视图切换。

**📊 数据集**

数据集与基准：
- 自定义 IPI 攻击基准（包含网页、邮件、文件三种注入向量，10 任务 × 3 攻击目标，共 90 例）;
- PinchBench 2.0.0（147 真实日常任务）;
- OpenClaw 的模拟网络/邮件服务（用于测试）。

**📈 对比分析**

对比与性能：
- 与原始 OpenClaw、Input/Output Guardrails、Sandboxing 及 Dual LLM (Utility) 对比；
- IPI 基准攻击成功率：Baseline 57%/27%，DualView 0%；
- PinchBench 代理实用率：Baseline 83.4%/88.5%，DualView 81.6%/82.1%（差距 ≤ 6.4%）；
- 计算成本：DualView 在 Haiku 上 +48.2%，Sonnet 上 +93.4% 以上 token 负担；
- Sandbox 防护 0% 成功率，但实用率降至 62%/66%。

**⚠️ 局限性**

局限性：
1) 仅覆盖本地文件、shell 与网络，无法处理远程云服务的符号化；
2) 数据使用策略需人工维护，无法完全覆盖所有危险用法；
3) 命令执行检测有限，仍需系统级监控；
4) 处理用户直接粘贴的不可信数据仍属未覆盖范围；
5) 符号化/去符号化流程引入显著 token 开销，影响成本与性能。

---

## 63. Tensor-Train Joint Modeling for Few-Step Discrete Diffusion

**arXiv ID:** 2607.03788 | [PDF](https://arxiv.org/pdf/2607.03788v1)

**作者:** Byoungkwon Kim `[一作]` (KAIST), Minhyuk Sung `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出基于张量分解的离散扩散框架，显式建模条件清洁分布以克服并行化偏差；

**💡 创新点**

首创使用低秩CPD和TTD对离散扩散的联合分布进行建模，发现TTD天然具备对局部依赖的结构偏置，并给出高效的迭代边缘推理采样方法；

**🔧 技术方法**

核心技术包括张量分解（CPD、Tensor‑Train）、迭代边缘推理、轻量化微调预训练MDM、以及对预训练头的重构与多层解码；

**📊 数据集**

实验数据集涵盖文本（OpenWebText、LM1B）与分子（QM9）两大任务；

**📈 对比分析**

与基准MDM、VADD等方法对比，TTD在8/16步等少步生成中将生成困惑度降低约25%，分子生成有效率提升，采样速度仅比原版慢1.7%；

**⚠️ 局限性**

局限在于只能使用低秩近似，无法捕获高秩复杂依赖，张量维度仍受限，未来需探索更丰富的张量格式。

---

## 64. SAVER: Stochastic Adaptive Variance-Driven Exploration and Reconstruction for Low-Dose Computed Tomography

**arXiv ID:** 2607.03761 | [PDF](https://arxiv.org/pdf/2607.03761v1)

**作者:** Shunta Nonaga `[一作]` (Hokkaido University), Tamiki Komatsuzaki `[通讯]` (Hokkaido University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了基于投影方差驱动的自适应CT采样框架SAVER，动态调整辐射方向以提升低剂量成像质量。

**💡 创新点**

创新点在于将投影方差作为实时信息度量，通过Softmax+模拟退火实现探索-利用平衡，实现对未知结构的自适应采样。

**🔧 技术方法**

使用了线性逆问题建模、Tikhonov正则化、Woodbury矩阵恒等式递推重建、Softmax采样策略、模拟退火温度退火，以及SSIM评估。

**📊 数据集**

采用八种32×32像素的合成phantom（矩形、空心正方形、三角形、十字、棋盘、条纹、梯度、Shepp–Logan）作为测试数据集。

**📈 对比分析**

通过与随机采样、轴向随机采样、以及三种oracle（MAX-V、MIN-V、SAVER-O）对比，SAVER在低噪声下实现了更高的AUC/500和更快的SSIM收敛，特别是结构异质性强的phantom。

**⚠️ 局限性**

局限在于仅评估了二维平行光束模型，未考虑三维锥束CT、散射/硬化等物理效应，且每一步重建仍需O(d^2)运算，难以直接扩展到高分辨率临床图像。

---

## 65. Enactive Drift Regulation and the Emergence Machine: A Framework for Coherent Adaptation Through Regulated Interaction

**arXiv ID:** 2607.03834 | [PDF](https://arxiv.org/pdf/2607.03834v1)

**作者:** Nicholas Davis `[一作]` `[通讯]` (Enactive AI), Nicholas Davis (Enactive AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了Enactive Drift Regulation（EDR）理论，设计了实现该理论的 Emergence Machine 架构，并对比现有的持续学习、在线学习、适应性滤波与预测处理框架。

**💡 创新点**

创新点在于：① 将漂移视为组织性失衡的监管信号，而非噪声或错误；② 将适应分为学习（内部内容更新）与调节（结构重组）两层次；③ 引入多尺度（局部、区域、全局）吸引子与多态域的概念，用一致性指标来驱动结构重组与记忆管理。

**🔧 技术方法**

核心技术包括：多尺度吸引子建模、状态一致性（coherence）度量、动态重组机制、记忆跨域回放、漂移监测（如 Hellinger 基准漂移检测）、仿真（SIM）回路与可视化。

**📊 数据集**

文中未给出具体实验数据集，说明了未来计划在合成与真实时间序列数据（金融、ECG、EEG、工业过程等）上进行验证。

**📈 对比分析**

目前仅在理论与架构层面进行讨论，未给出数值对比。作者指出将来会与持续学习、在线学习、适应性滤波、预测处理等方法在长期漂移场景下进行基准评估，以检验一致性监管对长期适应性和鲁棒性的提升。

**⚠️ 局限性**

局限性：① 缺乏实证验证与定量评估；② 一致性指标尚未标准化，需进一步定义与验证；③ 在大规模或高维数据上的可扩展性与计算开销未知；④ 对特定应用场景的实现细节与参数调优仍待研究。

---

## 66. Occluding the Solution Space: Planner-Agnostic Adversarial Attacks on Tolerance-Aware Manipulation

**arXiv ID:** 2607.03758 | [PDF](https://arxiv.org/pdf/2607.03758v1)

**作者:** Keke Tang `[一作]` (Guangzhou University), Zhihong Tian `[通讯]` (Guangzhou University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种无规划器依赖的对抗性障碍物插入框架，通过离线构建机器人运动学占用热图并在线进行预算最大覆盖优化，以阻断容差感知的操纵任务。

**💡 创新点**

创新点在于：① 将攻击目标从精确姿态改为任务级可达区域；② 在不需要访问目标规划器的前提下，仅利用运动学信息构造全局攻击策略；③ 通过子模优化和贪心求解实现高效、可扩展的攻击。

**🔧 技术方法**

核心技术包括：运动学占用热图（采样机器人配置、计算雅可比、操纵性与障碍物安全性加权）；子模最大覆盖优化；贪心算法与稀疏搜索；机器人运动学与碰撞检测工具（Pinocchio、MoveIt2）。

**📊 数据集**

实验使用七自由度Franka Emika Panda，在NVIDIA Isaac Sim中合成的5个桌面场景（MotionBenchMaker生成）以及真实的Rokae xMatePro7机器人配合Intel RealSense RGB‑D感知的实际工作环境。

**📈 对比分析**

与随机插入、Wu等人提出的规划器内循环攻击（PFA）和其容差适配版本（PFA++）对比，结果显示：① 在任何场景与规划器组合下，5个障碍物即可实现100%规划失败；② 相比PFA，障碍物数量下降3~10倍；③ 在线生成时间仅1.27 s，显著低于PFA的1224 s，显示出更高的计算效率。

**⚠️ 局限性**

局限性：仅针对静态环境；依赖准确的运动学模型和碰撞检测；对动态障碍物或不确定感知未做考虑；热图需要针对每台机器人单独构建，限制了迁移性。

---

## 67. Beyond Static Rules: Automated Discovery of Latent Vulnerabilities in Text-to-SQL

**arXiv ID:** 2607.03833 | [PDF](https://arxiv.org/pdf/2607.03833v1)

**作者:** Hanqing Wang `[一作]` (Shanghai University of Finance and Economics), Guanhua Chen `[通讯]` (Southern University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6215c339-3735-4be3-8a07-5bbb7004712d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了SAGE框架，自动化地发现文本到SQL生成中的潜在漏洞，并构建可演化的漏洞词典；

**💡 创新点**

通过可演化的漏洞词典和迭代的发现-演化闭环实现系统化、自动化的鲁棒性探索，显著提升发现率与效率，优于传统静态专家规则；

**🔧 技术方法**

结合LLM生成、检索与摘要模块（基于Qwen3-32B等）、语义检索嵌入、语义压缩、对抗样本生成与验证、轻量化微调等技术；

**📊 数据集**

使用Spider与BIRD数据集的已正确解答子集，并在Gemma‑3、OmniSQL、Inf‑rl‑qwen以及GPT‑4o等模型上进行评估；

**📈 对比分析**

与“Original”（标准评测）和“Expert”（固定专家规则）基线对比，SAGE在BIRD、Spider上平均VER提升至约77%/58%，EX下降至约33%/14%，ApD显著降低，展示出高发现率与高效率；

**⚠️ 局限性**

未对内部模块（Judge、Embedding、Summarizer）细粒度贡献进行分析；漏洞词典更新频率较低，批量级更新可能进一步提升效率。

---

## 68. How Do Diffusion Classifiers Decide? A Bias-Centric Evaluation

**arXiv ID:** 2607.03831 | [PDF](https://arxiv.org/pdf/2607.03831v1)

**作者:** Saba Fathi `[一作]` (Amirkabir University of Technology), Mahdi Javanmardi `[通讯]` (Amirkabir University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ASOB-Bench，对扩散式分类器在属性绑定、尺寸顺序和背景依赖三维度的偏差进行系统评估。

**💡 创新点**

首次从偏差角度揭示扩散式分类器的决策机制，并将重建误差热图与U‑Net交叉注意力可视化结合，定位各类偏差的根源。

**🔧 技术方法**

使用Stable Diffusion 2的扩散模型实现的零样本分类器，结合重建误差评分、U‑Net交叉注意力、热图分析等技术。

**📊 数据集**

使用自然水果、DALL‑E 3 合成的非自然水果、ComCo、ImageNet‑B 及其无背景版本等数据集进行评估。

**📈 对比分析**

与共享同一文本编码器的OpenCLIP ViT‑H/14 基线对比：在属性绑定上偏差更小，但在尺寸顺序和背景依赖上易受短路，导致准确率下降显著。

**⚠️ 局限性**

受限于仅考察三类偏差，未涵盖提示长度、社会偏差等因素；且偏差分析主要针对Stable Diffusion 2，虽然在 SD3 上验证一致性，但未彻底探究不同架构的根本原因。

---

## 69. CGGS: Consistency-Augmented Geometric Gaussian Splatting for Ego-centric 3D Scene Generation

**arXiv ID:** 2607.03819 | [PDF](https://arxiv.org/pdf/2607.03819v1)

**作者:** Zhenyu Sun `[一作]` (South China University Of Technology), Huan Wang `[通讯]` (Westlake University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了CGGS框架，结合文本生成一致性增强的多视角潜在扩散模型、光流深度估计与点轨迹的布局装饰器以及基于互信息深度损失的层级三维高斯细化器，实现了从文本到自我中心3D场景的完整生成与优化。

**💡 创新点**

创新点包括：①一致性增强的潜在扩散损失提升多视角一致性；②光流+点轨迹联合估计深度，生成粗糙但可靠的三维布局；③引入互信息深度损失和层级相机策略，显著提升几何细节与跨视角一致性；④将三维高斯点云作为全局优化基础，解决传统SfM在自我中心视角下的失败。

**🔧 技术方法**

使用技术：多视角潜在扩散模型（MVDiffusion）+ CAA模块；光流与点轨迹估计的光流深度估计网络；3D高斯溅射（3DGS）框架；互信息深度损失（MID）和层级相机扩展的层级优化；CLIP、Q-Align等评估指标。

**📊 数据集**

使用的数据集：Matterport3D（室内多视角RGB-D），RealEstate‑10k（室内外视频），CO3Dv2（COCO物体级三维重建数据）。

**📈 对比分析**

与Text2Room、LucidDreamer、Director3D、DreamScene360等方法在CLIP Score、Q-Align、PSNR、SSIM、LPIPS等指标上进行对比，CGGS在语义一致性、视觉质量和几何精度上均实现了领先或相近的最佳性能。

**⚠️ 局限性**

局限性：需要逐场景优化，计算成本高；仅支持静态场景，缺乏动态或交互式合成；在极端视角或极端遮挡情况下仍可能出现几何细节损失。

---

## 70. Global Logic and Local Search: Dual-Stream Multimodal In-Context Learning for Verifiable Industrial Anomaly Detection

**arXiv ID:** 2607.03817 | [PDF](https://arxiv.org/pdf/2607.03817v1)

**作者:** Runzhi Deng `[一作]` (Nanjing University), Fang Zhao `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、双流结构的工业异常检测框架 GLLS，用可解释的全球逻辑检查与局部搜索相结合，实现对工业图像的可验证缺陷识别。

**💡 创新点**

创新点包括：① 通过构建 Part‑Aware Visual‑Logical Atlas（PVLA）将文字规范与视觉参考映射成可执行图谱；② 采用全局逻辑流（SAM‑3 语义分割）与局部动作流（MCTS 预算搜索）并行推理；③ 在推理阶段保持中间可审计的证据链，实现无参数更新、可解释的结果。

**🔧 技术方法**

技术核心为：SAM‑3 语义分割、GPT‑5（仅离线生成规则）、Monte Carlo Tree Search（局部搜索）、大规模多模态语言模型（Qwen3‑VL、Qwen2.5‑VL 等）做最终验证，结合图谱检索实现知识驱动推理。

**📊 数据集**

使用 MMAD‑QA（MVTec‑AD、VisA）进行主实验，并在 MPDD、DTD、DAGM 等跨域数据集上做泛化评估；在MMAD‑QA上还使用多种开源与商用 LMM 作为对比基线。

**📈 对比分析**

与传统 LMM 基线相比，GLLS 在 MMAD‑QA 上实现了平均 3‑8% 的准确率提升；在 1‑shot 场景下平均准确率达 91%（高于多款专用视觉异常检测器）；跨域测试中 GLLS 仍保持 80%+ 的二分类准确率，表现出强泛化能力；相比优化型方法，GLLS 在保持训练自由的前提下取得竞争性性能。

**⚠️ 局限性**

局限性：① 依赖 SAM‑3 与 MCTS 的前置工具，若目标图像分辨率极低或缺乏可分割的部件，逻辑检查效果受限；② 需要一定数量的正常参考图像或高质量文本规范，若规范模糊或缺失会影响 PVLA 构建；③ 虽然无训练，但系统整体计算量仍较大（双流并行与 MCTS 推理），在资源受限的部署场景需进一步优化。

---

## 71. CineMobile: On-Device Image-to-Video Diffusion for Cinematic Camera Motion Generation

**arXiv ID:** 2607.03803 | [PDF](https://arxiv.org/pdf/2607.03803v1)

**作者:** Xuyao Huang `[一作]` (Shanghai Jiao Tong University), Zhijie Deng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在移动设备上实现了基于Diffusion Transformer的图像到视频生成，能够生成子弹时间、追踪缩放、慢动作等电影镜头效果。

**💡 创新点**

首次将结构化剪枝、步数蒸馏与混合精度量化结合到视频Diffusion Transformer，构建出4步低延迟、1 GB以内的模型，同时保持身份与场景一致性。

**🔧 技术方法**

采用结构化深度剪枝（PPCL改造）、流匹配目标监督微调、Adversarial Diffusion Distillation（AdvDMD）步数蒸馏、GRPO奖励以及FP8/4bit混合精度后训练量化；使用LoRA实现效果特定控制。

**📊 数据集**

使用PPR10K人像数据集及Pexels/Pixabay等公开图像训练子弹时间、追踪缩放、慢动作；教师模型基于Wan2.1‑I2V‑14B backbone加LoRA。

**📈 对比分析**

通过VBench自动评价和人类评测与教师模型以及HunyuanVideo‑1.5、LTX2.3、Open‑Sora2.0等公开模型对比，CineMobile在4步、1.2 B参数下与教师模型得分相近，速度提升约40×，手机端单步延迟≈20 s，显著降低内存。

**⚠️ 局限性**

仍依赖教师模型训练，效果主要针对结构化镜头；在极大帧间隔或复杂场景下可能出现细节失真，且对更广泛的特效类型的泛化能力仍需进一步验证。

---

## 72. The Hermitian Hull Dimensions for a Class of (L,P)-Twisted Generalized Reed-Solomon Codes

**arXiv ID:** 2607.03802 | [PDF](https://arxiv.org/pdf/2607.03802v1)

**作者:** Chenlu Jia `[一作]` (Sichuan Normal University), Qunying Liao `[通讯]` (Sichuan Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了一个特定形式的 (ℒ,𝒫)-扭曲广义 Reed–Solomon 码（𝒞_q+j(α)）的 Hermitian 舞池维数，给出了三种不同情况（取决于 i 的奇偶性以及与 q+1 的关系）的完整公式，并利用这些结果构造了两类满足 MDS 性质的 EAQECC。

**💡 创新点**

首次对该特殊 B 矩阵构造的 (ℒ,𝒫)-扭曲码给出了完整的 Hermitian 舞池维数表述，扩展了此前仅涉及 Euclidean 舞池的研究，且直接实现了 EAQECC 的构造与 MDS 性能保证。

**🔧 技术方法**

运用了有限域上线性代数、和式求值（根号统一求和）与 e_r 相关的性质，构造并化简 GG^† 矩阵，利用初等行列变换求秩，随后依据 Hull 维数公式推出 EAQECC 的参数；此外使用 Magma 进行实验验证。

**📊 数据集**

论文主要为理论推导，示例通过 Magma 程序验证，未使用实际数据集或实验硬件。

**📈 对比分析**

通过将理论推导得到的 Hermitian 舞池维数与 Magma 计算结果对比，验证公式正确性；构造的 EAQECC 在满足给定参数时可达到 MDS 性能，但文中未给出具体码率、误码率等性能指标。

**⚠️ 局限性**

结果仅适用于特定的 B 矩阵与 α 向量形式，未覆盖一般 (ℒ,𝒫)-扭曲广义 Reed–Solomon 码；证理论较为复杂，可能不易推广到更一般的参数设置；此外仅讨论 Hermitian 舞池，未涉及 Euclidean 舞池或更广泛的码性能评估。

---

## 73. Punching Above Their Weight: Classification-Head Fine-Tuning of Tiny Language Models (TLMs) for Verifiable Multiple-Choice Tasks

**arXiv ID:** 2607.03801 | [PDF](https://arxiv.org/pdf/2607.03801v1)

**作者:** Bhavesh Sood `[一作]` (Carnegie Mellon University), Jaromir Savelka `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文定义了小型语言模型（TLM，≤3B参数）并探讨了在可验证的多项选择任务中如何对其进行微调，比较了分类头微调、标签生成微调和仅金标继续微调三种 LoRA 微调方案，实验涵盖 Qwen3 系列 0.6B–8B 参数模型，使用 HellaSwag、WinoGrande、PIQA、SciQ 与 ARC‑Challenge 等五个英语多项选择基准；

**💡 创新点**

创新点在于提出了面向 TLM 的判别式分类头微调框架，并证明在小模型规模下该方法显著优于传统的标签生成微调，同时揭示微调目标与评估接口紧密耦合，提示评估方式会大幅影响报告性能；

**🔧 技术方法**

技术上使用冻结的 Qwen3 背景模型，插入单线性分类头替代语言模型头，采用跨候选交叉熵训练；评估时分别使用候选序列的分数、下一个 token 的答案标签以及 Log‑Probability 评分；所有微调均基于 LoRA；

**📊 数据集**

使用的基准数据集为 HellaSwag、WinoGrande、PIQA、SciQ 以及 ARC‑Challenge，共计五个公开英语多项选择任务；

**📈 对比分析**

通过统一的 LoRA 微调配置，三种微调方式在 0.6B–8B Qwen3 模型上进行对比；在 0.6B 与 1.7B 模型上，分类头微调比标签生成微调提升 2–3%（在 HellaSwag、WinoGrande、PIQA 与 SciQ 上均显著），并在 HellaSwag、WinoGrande、PIQA 上实现了 0.6B/1.7B 版本的 SOTA；在 >3B 模型上差距逐渐缩小；金标继续微调始终落后；

**⚠️ 局限性**

局限性包括仅评估 Qwen3 系列模型、仅使用英语基准、未探索少样本推理或推理式生成、分类头仅为单线性层、未细致探究提示格式和答案格式、未测试更大参数规模（>8B）、未覆盖多语言或开放式任务。

---

## 74. InfraNet: Quality-Aware RGB Guidance for Efficient Infrared Object Detection

**arXiv ID:** 2607.03795 | [PDF](https://arxiv.org/pdf/2607.03795v1)

**作者:** Zichao Feng `[一作]` (Beihang University), Baochang Zhang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

InfraNet提出了一种IR为主的质量感知RGB辅助学习框架，支持IR‑Only和RGB‑IR两种部署；

**💡 创新点**

核心创新是QualGate模块学习可调可靠性控制量，既抑制不可靠的RGB特征，又补偿IR特征，实现训练时的RGB辅助学习而推理时仅用IR；

**🔧 技术方法**

方法基于YOLO等检测骨干，加入QualGate融合机制，采用双分支网络、两阶段训练和可插拔的IR‑Only/RGB‑IR结构；

**📊 数据集**

在四大RGB‑IR基准数据集LLVIP、FLIR‑Aligned、M³FD和DroneVehicle上进行实验；

**📈 对比分析**

与现有融合方法对比，InfraNet‑IR在IR‑Only推理下精度竞争或更高，InfraNet‑RGB‑IR在双模态下更优，同时保持较高的推理效率；

**⚠️ 局限性**

局限在于仍需RGB与IR配对训练，且对极端光照下RGB质量的评估完全依赖学习，无法完全适应无RGB训练的场景。

---

## 75. G$^2$TAM: Geometry Grounded Track Anything Model

**arXiv ID:** 2607.03789 | [PDF](https://arxiv.org/pdf/2607.03789v1)

**作者:** Chenming Zhu `[一作]` (The University of Hong Kong), Xihui Liu `[通讯]` (The University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于几何的统一框架G2TAM，实现仅使用无序RGB图像/视频即可完成多模态提示的3D实例追踪与重建。

**💡 创新点**

核心创新在于将空间对齐的几何表征作为隐式记忆，采用跨模态空间编码器实现提示与视觉信息的早期融合，从而在无显式时间记忆库的情况下实现跨视角、跨时序的实例一致性。

**🔧 技术方法**

技术上结合了Pi3的几何编码器、SAM的掩码解码器、CLIP文本编码器以及DINOv2视觉特征，构建跨模态空间编码器并联合训练重建与分割损失。

**📊 数据集**

使用了自构建的InsTrack数据集（基于ScanNet++）进行训练与评测，并在SR3D、NR3D、ScanRefer等3D视觉定位基准上验证。

**📈 对比分析**

在InsTrack验证集上，G2TAM以74.3 S-mIoU/80.1 S-SR远超SAM2（47.6/53.1）和其他VOS/视觉定位方法；在3D视觉定位上达到Acc@0.5 56.2%/45.7%，显著优于VLM-Grounder和3D-VisTA。

**⚠️ 局限性**

局限性包括推理时跨视角注意力开销较大、对极端遮挡仍有挑战、未完全利用显式相机位姿/深度信息，且在部分动态场景下仍需进一步提升时序一致性。

---

## 76. Conservative Subject Invariant EMG-based Gesture Recognition

**arXiv ID:** 2607.03783 | [PDF](https://arxiv.org/pdf/2607.03783v1)

**作者:** Hamed Rafiei `[一作]` (Ferdowsi University of Mashhad), Ali Mousavi `[通讯]` (Islamic Azad University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种保守的多目标学习框架CSIN，联合优化手势分类、对抗性主体混淆和三元组距离学习，力求得到既能区分手势又对主体不敏感的潜在表征。

**💡 创新点**

创新点包括：①将对抗性主体损失与三元组损失联合起来，同时引入Lipschitz启发的自适应权重机制，动态平衡三种目标的尺度；②通过梯度逆转实现对主体信息的消除；③在多主体下保持训练稳定，显著提升跨主体泛化。

**🔧 技术方法**

技术主要包括：共享特征编码器、手势头与主体头、交叉熵分类损失、梯度逆转对抗损失、三元组距离损失、Lipschitz自适应权重、Adam优化、t-SNE可视化。

**📊 数据集**

使用了两套公开数据集：UCI EMG（36位受试者、6个手势）和NinaPro DB5（10位受试者、10个手势），均采用时域特征（MAV、RMS、WL、ZC）窗口化。

**📈 对比分析**

与现有方法比较，UCI EMG平均准确率提升至84.48%（高于78.2%），NinaPro DB5平均准确率提升至61.44%（高于41.30%），同时标准差显著下降，最差受试者准确率提升约13点，证明跨主体稳定性提升。

**⚠️ 局限性**

局限性包括：对极端个体仍有较低准确率（如UCI EMG部分受试者低于40%），固定的时域特征限制了表征容量，完全去除主体信息在某些临床或个性化场景可能不利，且对主体与手势信息的可分离程度仍存在上限。

---

## 77. EvoEye: Self-Evolving Runtime Monitoring for Autonomous Driving Systems

**arXiv ID:** 2607.03755 | [PDF](https://arxiv.org/pdf/2607.03755v1)

**作者:** Mingfei Cheng `[一作]` (Singapore Management University), Xiaofei Xie `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种自演化运行时监控框架，对全栈自动驾驶系统进行碰撞风险预测并通过监控误差驱动场景采样实现自适应提升。

**💡 创新点**

创新点包括：①利用跨模块时间依赖的学习型监控模型；②基于当前监控误差与密度感知的反馈机制进行场景变异，形成闭环自演化；③实现了高效、可解释的实时监控。

**🔧 技术方法**

使用多模态特征编码（MLP + attention），帧块 + 时间聚合网络，Sigmoid 分类器；结合监控误差反馈、密度感知变异和遗传搜索思想进行场景采样；实现基于 PyTorch。

**📊 数据集**

基于 Baidu Apollo 与 CARLA 仿真平台构造的高速公路切入和未受保护交叉路两类逻辑场景，共约200个场景，标注 3 秒预碰撞窗口。

**📈 对比分析**

与 TTC、RSS、AutoEncode-Monitor 等基线对比，帧级 F1 在 30%–67% 之间，高于基线 13–40 点；自演化相较于均匀采样和 AVFuzzer 在相同预算下提升 13.2 点 F1；警报提前 4.2/2.8 秒，推理延迟 2.5 ms。

**⚠️ 局限性**

局限性：仅评估碰撞场景；实验仅在 Apollo+CARLA 环境下进行；需要根据不同 ADS 架构定制内部信号抽取；仿真与真实环境差异仍待进一步验证。

---

## 78. A Modern View on MCSat

**arXiv ID:** 2607.03777 | [PDF](https://arxiv.org/pdf/2607.03777v1)

**作者:** Thomas Hader `[一作]` (TU Wien), Laura Kovacs `[通讯]` (TU Wien)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文重新定义并实现了MCSat算子为一个通用的、与理论无关的推理框架，并给出了相应的规则体系，随后实例化到布尔、非线性实数算术以及无解释函数等多种理论中；

**💡 创新点**

主要创新点在于：①将MCSat从原来的变量/命题分离模型，转为以任意项的赋值为核心，统一了布尔与理论推理；②提出了理论无关的规则集合（决策、传播、冲突分析、学习、回溯等），并将其与Yices2实现紧密对应；③在解释函数设计上放宽了“有效蕴含”要求，使其更符合实现；④通过插件架构实现多理论组合，提升了可扩展性；

**🔧 技术方法**

采用了MCSat证明系统（带决策、传播、冲突分析、学习等规则）、插件式理论推理（支持解释函数与冲突解释）、闭包运算、理论自解释与冲突学习等技术；

**📊 数据集**

本文没有提供新的实验数据集，主要是理论与实现的对比与示例；

**📈 对比分析**

对比方法未在论文中给出具体实验结果，只是说明该框架与Yices2中MCSat实现的一致性以及在非线性实数算术等理论上的优秀性能；

**⚠️ 局限性**

局限性包括：①解释函数的有限基准假设在实际实现（如使用vsids启发式）下可能不成立，影响终止性；②本文未给出完整的实验评估，缺乏对性能提升的量化验证；③在大规模多理论组合时，插件间的交互与冲突解释的效率仍有待改进。

---

## 79. ContiStain: Cross-Domain Relation-Preserving Distillation for Continual Multi-Domain Virtual IHC Staining

**arXiv ID:** 2607.03851 | [PDF](https://arxiv.org/pdf/2607.03851v1)

**作者:** Fuqiang Chen `[一作]` (Harbin Institute of Technology (Shenzhen)), Yongbing Zhang `[通讯]` (Harbin Institute of Technology (Shenzhen))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种名为 ContiStain 的持续多域虚拟 IHC 染色框架，能够在不遗忘已学域的前提下，逐步适应新出现的生物标志物。

**💡 创新点**

其创新点包括：在混合专家（MoE）结构中构建域感知的有序特征空间，并通过跨域关系保持蒸馏，显式维护不同生物标志物在潜在空间中的几何关系。

**🔧 技术方法**

所采用的技术包括：Mixture-of-Experts 与 FiLM 进行域注入、ASP（Adaptive Supervised PatchNCE）生成器、对抗及片段级对比学习、关系蒸馏与 L1 正则化，以及门控网络的专家选择与正则化。

**📊 数据集**

实验使用了 Multi-IHC Stain Translation (MIST) 数据集，该数据集包含四个标志物（HER2、Ki67、ER、PR）的 H&E–IHC 配对样本。

**📈 对比分析**

与顺序微调、LWF、EWC、iCaRL 等持续学习基线对比，ContiStain 在 FID、ConchFID、PSNR、SSIM、DISTS 等指标上均表现更优，显著降低了遗忘并保持了跨域结构一致性。

**⚠️ 局限性**

局限性在于仅在 MIST 的四域设置上验证，缺乏对更大规模、更多域的泛化评估；需要精细调节 λ_rel 与专家数量，且对训练成本和实时临床部署的适配尚待进一步研究。

---

## 80. ObjRetarget: An Object-Aware Motion Retargeting Framework with Anthropomorphic Arm Constraints and Polyhedral Hand Modeling

**arXiv ID:** 2607.03828 | [PDF](https://arxiv.org/pdf/2607.03828v1)

**作者:** Yuanchuan Lai `[一作]` (Sun Yat-sen University), Zhaojie Ju `[通讯]` (University of Portsmouth)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 ObjRetarget 框架，通过分离臂部全局运动与手部接触敏感动作，并结合多面体手模型与拟人化臂约束，实现从人类视频到机器人执行的精细操作。

**💡 创新点**

创新点在于将手-物体接触建模为多面体簇并利用几何不变量约束，同时针对臂部引入任务自适应臂平面正则化，以兼顾全局自然运动与局部接触稳定。

**🔧 技术方法**

使用了基于 RGB‑D 的三维姿态与点云估计、逆运动学与约束优化、几何不变量约束以及深度学习编码器‑解码器生成参考轨迹。

**📊 数据集**

数据主要来源于实时 RGB‑D 视频捕捉的人类操作序列，涵盖六种日常细粒度任务（如放置、倒水、开抽屉等）。

**📈 对比分析**

与 OKAMI 与 ORION 对比，ObjRetarget 在 20 次试验中平均成功率提升至 75.8%，显著高于对手的 61.6%（OKAMI）和 50.8%（ORION）。

**⚠️ 局限性**

局限在于缺乏实时力反馈与动态交互适应，仅适用于上肢操作，尚未扩展至全身或复杂动态环境。

---

## 81. Q-TriM: Question-Guided Tri-Modal Attention for Audio-Visual Question Answering

**arXiv ID:** 2607.03825 | [PDF](https://arxiv.org/pdf/2607.03825v1)

**作者:** SungHun Kim `[一作]` (Korea University), SeungJun Baek `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了Q-TriM模型，通过浅层并行多模态注意力融合问答任务，避免了传统深层堆叠带来的信息丢失与误差累积。

**💡 创新点**

创新点在于提出三模态注意力（Query、Key、Value分别来自文本、视频、音频）以及并行融合机制，直接捕获三方交互，减少多模态间信息衰减。

**🔧 技术方法**

技术上结合CLIP与ImageBind预训练特征、Token过滤（基于STE的top‑K筛选）、Tri‑Modal Attention、FiLM调制以及自注意力后向推理，形成轻量级的并行融合架构。

**📊 数据集**

实验数据集包括MUSIC‑AVQA、MUSIC‑AVQA‑R与MUSIC‑AVQA‑v2.0，覆盖音频、视频与问答三种模态。

**📈 对比分析**

与多种基线（QA‑Tiger、TSPM、LAVISH等）对比，Q‑TriM在三大数据集上均实现SOTA，尤其在MUSIC‑AVQA‑R上取得显著提升，证明其在稀有/长尾分布下的鲁棒性。

**⚠️ 局限性**

局限性在于模型高度依赖预训练特征与特定的三模态设置，未在更广泛的多模态任务（如跨语言、跨域场景）中验证，且对不同模态配对方式的通用性仍待探究。

---

## 82. TestMate: Test-Time Domain Adaptation Aided by Lightweight Vision Foundation Model

**arXiv ID:** 2607.03810 | [PDF](https://arxiv.org/pdf/2607.03810v1)

**作者:** Dimitrios Fotiou `[一作]` (Aristotle University of Thessaloniki), Ioannis Pitas `[通讯]` (Aristotle University of Thessaloniki)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于轻量级视觉基础模型的无梯度测试时域自适应框架TestMate，用于语义分割任务。

**💡 创新点**

创新点是利用零样本实例分割VFM生成的高质量掩模，按尺寸递增顺序进行无参数的竞争式融合，既不需要反向传播也不依赖内存库，能够从第一帧立即提升分割边界精度并保持小目标。

**🔧 技术方法**

技术包括FastSAM（YOLOv8‑seg）作为VFM，基于大小排序的区域竞争融合、软融合和熵筛选机制；可与TENT、CoTTA、DIGA等已有TTDA方法联合使用。

**📊 数据集**

数据集：GTA‑V→Cityscapes的模拟到真实迁移以及FMB→MVSeg的真实到真实迁移，均使用DeepLabV2/V3。

**📈 对比分析**

与多种基准（TENT、CoTTA、DIGA、DS‑DT等）对比，TestMate在SFDA、TTDA和在线TTDA上分别取得新的SOTA，尤其在小目标和边界精度上提升约1–3%。

**⚠️ 局限性**

局限：依赖VFM的实例检测质量，若VFM生成的掩模不准确会导致误融合；在极端连续分布漂移下单独使用效果略逊于强记忆库方法；对高分辨率输入仍需权衡速度与精度。

---

## 83. Probabilistic Robustness in Medical Image Classification

**arXiv ID:** 2607.03797 | [PDF](https://arxiv.org/pdf/2607.03797v1)

**作者:** Yi Zhang `[一作]` (University of Warwick), Xingyu Zhao `[通讯]` (University of Warwick)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过构建医学影像的自然失真场景，对常用深度学习模型在MedMNIST v2数据集上的概率鲁棒性进行系统评估。

**💡 创新点**

创新点在于将概率鲁棒性（PR）视为评估医学诊断系统可信度的实用指标，并针对医学影像设计了六种自然失真类型，填补了以往仅关注最坏情况对抗鲁棒性（AR）的空白。

**🔧 技术方法**

使用了深度卷积网络（ResNet‑18、ResNet‑50）以及PRBench评估框架，对失真样本进行随机扰动并计算保持正确预测的概率。

**📊 数据集**

数据集为MedMNIST v2中的PathMNIST（9 类结肠病理图像），共107,180张样本。

**📈 对比分析**

与传统的AUC/ACC指标比较，PR显示在亮度、模糊等现实失真下模型鲁棒性显著下降，且更高的清晰图像准确率并不一定转化为更好的鲁棒性，证明了PR的评估价值。

**⚠️ 局限性**

局限性包括仅针对二维医学图像、仅评估了六种失真类型以及仅测试了两种网络架构，未来需扩展到3D影像、多任务场景及更丰富的失真分布。

---

## 84. Folding, Reasoning, and Scaling with Open-source Drug Discovery Engine

**arXiv ID:** 2607.03787 | [PDF](https://arxiv.org/pdf/2607.03787v1)

**作者:** Aureka AI OpenDDE project `[一作]` `[通讯]` (Aureka AI Research), Aureka AI OpenDDE project (Aureka AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并训练了开源的全原子生物分子基础模型OpenDDE，实现了与IsoDDE同等水平的共折叠精度，并为后续药物设计奠定了结构推理基础。

**💡 创新点**

通过引入原子级隐藏推理与结构令牌细化、形状互补损失以及统一的条件扩散框架，实现了从序列到全原子坐标的粗细层级推理，并探索了模型与数据的规模法则。

**🔧 技术方法**

采用Pairformer结构、扩散生成器、Atom37结构令牌、形状互补损失、MSA与模板结合的条件推理、以及多阶段训练与增量采样的推理策略。

**📊 数据集**

结合加权PDB、AFDB多体、Teddymer、MGnify长短单体、Swiss-Prot、SAbDab等多源训练数据，涵盖蛋白、核酸、小分子及其复合体。

**📈 对比分析**

在PXMeter-AB、FoldBench-AB、2026ARK-AB抗体–抗原基准上，用DockQ成功率与Oracle评估对比，OpenDDE分别达51%、70%、66%（排名基准）或65%、82%、80%（Oracle基准），显著优于AlphaFold3、Protenix-v1/2、ESMFold2等模型。

**⚠️ 局限性**

仍缺乏对IsoDDE完整细节的验证；对蛋白-小分子复合体的适用性有限；仅提供结构预测核心，尚未实现完整药物发现链条（设计、亲和力预测等）。

---

## 85. FedACT: Federated Adaptive Coordinate Trust Modulation for Robust Transformer Training under Data Heterogeneity

**arXiv ID:** 2607.03763 | [PDF](https://arxiv.org/pdf/2607.03763v1)

**作者:** Shuai Li `[一作]` (National University of Defense Technology), Tao Sun `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种全局感知的坐标信任调制方法 FedACT，用于在异构数据下的 Federated AdamW 训练，先形成全局修正的 AdamW 方向，再根据坐标级信任评分重新分配更新幅度。

**💡 创新点**

创新点在于发现并解决了坐标级信任不匹配问题：即使在已进行全局-局部对齐后，AdamW 的坐标更新仍可能高度不均衡；FedACT 通过全局感知的坐标信任评分以及软衰减的权重分配，补充了传统的通信轮级校正。

**🔧 技术方法**

技术手段包括 AdamW 及其服务器端全局修正、坐标级信任评分 s = u ⊙ g、比例 τ 选取高信任坐标、权重 α 与 γ 的软衰减、以及对比实验中的多种基线（FedAvg、FedProx、SCAFFOLD、FedAdam、FedAdamW）。

**📊 数据集**

实验使用的主要数据集有 CIFAR‑10/100、Tiny ImageNet、C4‑en、Alpaca‑GPT4、HH‑RLHF；模型包括 Vision Transformer (ViT‑Tiny、Swin‑Lite)、ResNet‑18、Llama2‑60M/130M/250M 等。

**📈 对比分析**

与 FedAvg、FedProx、SCAFFOLD、FedAdam、FedAdamW 等基线比较，FedACT 在 Transformer 训练中平均提升约 3.1% Top‑1 准确率（在 300 轮、100 客户端、10% 参与率下），在 LLM 预训练中显著降低验证 perplexity（例如 Llama2‑60M 从 35.64 降至 32.57），并在通信轮数上更快收敛；CNN 任务上提升幅度较小。

**⚠️ 局限性**

局限性包括对超参数 τ、γ 的敏感性；在 CNN 训练中收益有限；对极度稀疏或更高维度模型的泛化能力尚未充分验证。

---

## 86. GeoSAM-Lite: A Lightweight Foundation Model for Onboard Remote Sensing Segmentation

**arXiv ID:** 2607.03760 | [PDF](https://arxiv.org/pdf/2607.03760v1)

**作者:** Yongcong Wang `[一作]` (Nanjing Forestry University), Li Zhang `[通讯]` (Nanjing Forestry University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出轻量化的GeoSAM-Lite模型，实现地球观测遥感图像的高效分割

**💡 创新点**

引入域感知初始化(Geo-Init)与频域空间融合层(FFL)两大创新，克服域偏差与高频细节损失

**🔧 技术方法**

采用Masked Image Modeling、知识蒸馏、FFT频域高通滤波与可学习的空间重校准模块

**📊 数据集**

在云检测（38-Cloud、CloudSEN12、SPARCS）和农田分割数据集上进行实验

**📈 对比分析**

与EfficientSAM、MobileSAM及RSAM-Seg对比，GeoSAM-Lite参数92.8%降、FLOPs92.5%降，性能仅略低于RSAM-Seg，在多数据集上保持竞争力

**⚠️ 局限性**

仍与大模型存在性能差距，边界细节恢复仍有提升空间，且需进一步验证跨传感器的泛化

---

## 87. PRISM3D: Probabilistic Refinement and Robust Initialization for Physically Consistent Scene Modeling under Extreme Motion Blur

**arXiv ID:** 2607.03855 | [PDF](https://arxiv.org/pdf/2607.03855v1)

**作者:** Gopi Raju Matta `[一作]` (IIT Madras), Kaushik Mitra `[通讯]` (IIT Madras)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出PRISM3D框架，能够在极端运动模糊的RGB图像下实现盲3D场景重建，并扩展为PRISM3D-E，利用事件相机提升重建精度。

**💡 创新点**

创新点在于：①将深度全景跟踪（VGGSfM）与概率化MCMC稠密化相结合，弥补传统SfM在模糊条件下失效的缺口；②采用连续SE(3) Bézier曲线对相机运动进行物理一致建模；③在多模态场景下，仅用事件生成的伪清晰图像做初始化，而非用于光度监督，从而保持物理一致性。

**🔧 技术方法**

核心技术包括：深度全景跟踪VGGSfM、3D Gaussian Splatting的MCMC稠密化（3DGS-MCMC）、Bézier轨迹优化、事件摄像机的EDI模型、以及基于SGLD的概率优化。

**📊 数据集**

数据集：合成的ExBluRF场景（8个户外场景）以及新建的PRISM3D-E Benchmark（配有事件流的合成极端模糊图像）；真实数据使用E2NeRF数据集（DAVIS346事件相机拍摄）。

**📈 对比分析**

与传统基于COLMAP的Oracle方法、单图像去模糊、以及现有事件相机方法相比，PRISM3D在盲RGB设置下PSNR平均提升约3–4 dB，SSIM提升0.1以上；PRISM3D-E在事件辅助下平均PSNR再提升2.5 dB，且在Oracle基准下可逆逼近甚至超越（0.5–0.8 dB）。

**⚠️ 局限性**

局限性包括：仅适用于静态场景；对极端动态变化的处理仍不足；对VGGSfM在高噪声/极端模糊下的鲁棒性有待进一步提升；事件相机的同步与时延问题未被充分解决。

---

## 88. FDR-Occ: Factorized Dense Routing for Full-Spectrum 3D Occupancy Prediction

**arXiv ID:** 2607.03822 | [PDF](https://arxiv.org/pdf/2607.03822v1)

**作者:** Dubing Chen `[一作]` (University of Macau), Jianbing Shen `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种新的视图变换方法——Factorized Dense Routing（FDR），并结合分离的全局语义路径与局部几何路径，解决了传统基于物理射线投影导致的局部性瓶颈。

**💡 创新点**

创新点在于：①将2D-3D映射视为无约束的二部路由，并通过分层张量收缩实现全局可达的稠密路由；②引入Resolution-Context Decoupled Architecture，将全局上下文与像素精度分离，兼顾全局语义与局部几何；③在未校准的条件下仍能自学习多摄像头拓扑，显著提升鲁棒性。

**🔧 技术方法**

核心技术包括：分层稠密路由（FDR）、动态几何辅助的路由权重生成、BEV与三维平面联合编码、稀疏与稠密路由的加和融合、以及多任务训练（BCE+Dice+depth+semantic）。

**📊 数据集**

在两大自动驾驶基准上进行评估：Occ3D-nuScenes（17类语义）和Occ3D-Waymo（14类语义）。

**📈 对比分析**

与现有基线（如FB-Occ、DHD-S、COTR、BEVDetOcc、ALOcc等）比较，单帧条件下mIoU提升至41.2%（nuScenes）和31.25%（Waymo），并在未校准场景下保持28.0% mIoU，显著优于所有传统物理投影方法。

**⚠️ 局限性**

局限性：FDR仍依赖轻量级深度头提供软几何提示，尚未实现完全无几何约束；未校准实验仅验证单一相机阵列，跨阵列或动态相机系统的泛化仍需进一步研究。

---

## 89. TRISTAR: Triple-Signal Stair Recognition and Vision-Only Indoor Navigation for Search-and-Rescue Micro-UAVs

**arXiv ID:** 2607.03818 | [PDF](https://arxiv.org/pdf/2607.03818v1)

**作者:** Octavian Gîngu `[一作]` (Military Technical Academy ``Ferdinand~I''), Stelian Spînu `[通讯]` (Military Technical Academy ``Ferdinand~I'')

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套基于单目摄像头的低成本微型无人机室内自律导航框架，完成走廊探索、自动门检测与进入、房间OCR识别、人员姿态检测与医疗评估以及楼梯爬升等任务。

**💡 创新点**

创新点包括：① 通过深度校准将相对深度转化为精确的度量深度，门检测仅依赖几何深度而非视觉特征；② 设计TRISTAR三模态融合（Sobel、Gabor、单目深度）实现鲁棒楼梯识别；③ 将传统计算机视觉与轻量级深度学习模型无缝集成，构建完整的操作报告链路；④ 在不使用任何专用测距硬件的前提下实现可靠的室内导航与楼梯爬升。

**🔧 技术方法**

核心技术包括 Depth Anything V2（单目深度估计）、YOLO11n-Pose 与 YOLOv8（目标检测与姿态估计）、EasyOCR（房间号识别）、Gemini 2.5 Flash（医疗状态评估）、Sobel 与多尺度 Gabor 滤波、经验深度校准、指数移动平均平滑、TRISTAR 三模态融合算法以及基于 FastAPI、React 的本地服务器与 Web UI。

**📊 数据集**

使用真实室内实验数据：在大学建筑走廊、模拟房间以及标准楼梯上收集的实时飞行数据；门检测数据集包含142正例+168负例帧；楼梯检测数据集220帧（118阶梯、54着陆、48其他）。此外还使用了从实验室获取的静态图像用于失败案例分析。

**📈 对比分析**

与现有商业/学术平台对比：成本低于150 欧元、实现全功能；门检测在离线评估中取得 F1 = 0.91，深度误差校准后低于10%；TRISTAR 在离线评估中 F1 ≈ 0.92；实飞行中门检测成功率12/12，楼梯爬升成功率4/4（单一传感器或两传感器组合显著下降）；系统保持 30 FPS 原始流、10–12 Hz 控制循环，端到端延迟 85–100 ms，满足室内导航时限。

**⚠️ 局限性**

主要限制包括：① 无持久 SLAM 或全局地图，导致每次任务独立；② 走廊与楼梯行为仍分离，缺乏连贯的任务链；③ 低光、玻璃窗、反光表面等环境条件下单目深度可靠性下降；④ 需要云端服务（OCR、医疗评估），不适合无网络场景；⑤ Tello 的运动学限制导致多次 90° 旋转后漂移；⑥ 部分失败案例仍需人工干预。

---

## 90. TSP with Predictions: Heatmap to Tour with Provable Guarantees

**arXiv ID:** 2607.03791 | [PDF](https://arxiv.org/pdf/2607.03791v1)

**作者:** Marek Eliáš `[一作]` (Università Bocconi), Eleonora Vercesi `[通讯]` (University of Italian Switzerland)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于热图预测的学习增强 TSP 算法，该算法将每条边的预测概率转换为可行的巡回路径，并给出了 1+2η 的近似保证（η 为热图与某条最优巡回路径的 L1 距离）。

**💡 创新点**

创新点在于：①在传统 Christofides 方案上加入预测偏置，能够在预测误差为 η 时保证路径长度不超过最优长度加 2η；②首次将热图预测与理论近似分析相结合，为学习增强优化提供严谨保证；③在欧氏、图形和一般度量 TSP 上实现了不同的近似与时间复杂度权衡。

**🔧 技术方法**

主要技术包括：利用权重 0 或负值的 MST（偏向预测），最小 S‑join（或近似匹配）求解奇度点；在欧氏实例中使用 Delaunay 图加速 MST；对预测进行采样/阈值抽取生成预测集；与传统贪心、Beam Search 及 2‑opt 本地搜索等基线做对比。

**📊 数据集**

使用的数据集：ML4CO benchmark 的三大子集（U、T_E、T_M），以及 TSPLIB 中的欧氏（≤1300 节点）和非欧氏（≤15 条实例）TSP。

**📈 对比分析**

实验比较：与 Christofides、贪心（G1、G2）、Beam Search、SoftDist 等基线对照；在 n=500 的实例上，使用采样热图（Alg1）与阈值权重（Alg1Top）等解码策略时，平均最优性差距明显低于 Christofides，并随预测误差降低呈现平滑提升；2‑opt 后处理进一步细化结果。

**⚠️ 局限性**

局限性：①性能高度依赖热图预测质量；②当误差 η 较大时，近似保证失效；③匹配/ S‑join 步骤仍具一定计算开销（尤其是欧氏匹配）；④理论下界表明对 η 的线性依赖难以进一步突破，限制了最优比率的提升。

---

## 91. SkillFab: An Agent-Native Skill Production Platform

**arXiv ID:** 2607.03780 | [PDF](https://arxiv.org/pdf/2607.03780v1)

**作者:** Anjie Xu `[一作]` (Peking University), Leye Wang `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

搭建并验证了一个面向代理的技能生产平台 SkillFab，支持需求优先的 issue 记录、Git 证据、审核、版本化注册与复用的完整生命周期

**💡 创新点**

创新点在于将需求先行的 issue 机制与 Git 证据、代理原生交互、可审核、可恢复的工作流相融合，形成可追溯、可维护的技能生产生态

**🔧 技术方法**

使用技术包括 Hono Node.js 框架、Rust 原生插件、SQLite（或 D1）、自研 Git 服务器、REST/JSON‑RPC 接口以及 GitHub 风格的协作模型

**📊 数据集**

数据集主要来自自有平台运行日志及三项案例（OS‑detect、Docker‑research、SkillOpt‑governance）

**📈 对比分析**

本文未给出量化性能指标，只通过三项案例演示流程完整性；未来工作计划评估重复劳动减少、复用效率提升等实测效果

**⚠️ 局限性**

局限性包括缺乏大规模评估与性能数据、安全与运维细节待完善、信任策略与治理机制仍在探索中

---

## 92. Self-Improving Diffusion Classifiers with Minority Preference Optimization

**arXiv ID:** 2607.03770 | [PDF](https://arxiv.org/pdf/2607.03770v1)

**作者:** Hyunsoo Kim `[一作]` (Korea University), Suhyun Kim `[通讯]` (Kyung Hee University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种自我改进的扩散分类器MiPO，利用少数族群偏好奖励在无图像数据的前提下微调扩散模型，从而提升零样本分类性能。

**💡 创新点**

创新点在于将少数族群奖励与群组相对策略优化（GRPO）相结合，并通过LoRA进行高效、可插拔的微调，首次实现了在无需额外图像或奖励模型的情况下显著扩展扩散分类器的感知覆盖。

**🔧 技术方法**

采用LoRA参数化、群组相对策略优化（GRPO）、KL正则化、基于DDIM重建误差的少数族群奖励以及早期步长选择等技术。

**📊 数据集**

主要使用HPSv2文本提示集进行自我训练，并在CIFAR‑10、Tiny‑ImageNet、CIFAR‑10‑C、Caltech、SUN09等公开数据集上进行零样本分类评估。

**📈 对比分析**

与原始扩散分类器基线相比，MiPO在多组标准基准上平均提升约2%–3.8%的准确率，并在少数族群生成任务中实现了与手动优化方法相当的质量，但推理速度保持与DDIM相同。

**⚠️ 局限性**

局限性包括对噪声多物体或标注不一致的数据集（如LabelME、VOC2007）提升有限，以及在过度强化少数族群奖励时可能导致视觉保真度略有下降。

---

## 93. Semantic-aware and Self-improving Program Reduction via Agentic Large Language Models

**arXiv ID:** 2607.03766 | [PDF](https://arxiv.org/pdf/2607.03766v1)

**作者:** Xintong Zhou `[一作]` (University of Waterloo), Chengnian Sun `[通讯]` (University of Waterloo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于代理大型语言模型的语义感知、可自我改进的程序缩减框架，通过让模型在交互循环中分析程序语义、假设并迭代验证，最终得到最小化程序。

**💡 创新点**

创新点在于把程序缩减视为探索性推理任务，赋予LLM代理自主决策并通过反射器将成功经验提炼为可执行的策略，从而实现语义感知和持续自我改进。

**🔧 技术方法**

核心技术包括Agentic LLM（如DeepSeek-V4-Flash）、ReAct循环、工具调用接口、学习式Reducer（可执行策略池）、反射器（策略提炼）以及基于语义的约束验证。

**📊 数据集**

实验使用了三种语言的基准：C语言的CBench、Rust语言的RustBench、Go语言的GoBench，并收集了来自前沿程序缩减工作与新生成的40个Fuzzer程序。

**📈 对比分析**

与四个基线（DeltaDebug、C-Reduction、L-Reduction、C-Reduction+LLM）比较，所提框架在所有语言中均取得最小token数，平均缩减到约1/3，效率提升30-50%，且成本略高但可通过学习Reducer进一步降低。

**⚠️ 局限性**

局限性包括策略库可能随时间膨胀导致开销增加、对LLM推理速度和费用的依赖、以及在极度异构的程序集上策略通用性仍有限。

---

## 94. TokAN: Accent Normalization Using Self-Supervised Speech Tokens

**arXiv ID:** 2607.03928 | [PDF](https://arxiv.org/pdf/2607.03928v1)

**作者:** Qibing Bai `[一作]` (Chinese University of Hong Kong Shenzhen), Haizhou Li `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于自监督离散语音令牌的口音归一化框架 TokAN，能够在不需要真实 L1–L2 并行数据的情况下，将非母语口音语音转换为接近母语口音，同时保持说话人身份。

**💡 创新点**

创新点包括：① 联合训练的 VQ 令牌器与流匹配合成器，使令牌在语音合成和 ASR 目标下优化；② 采用 RoPE 的自回归编码‑解码器，去除源口音嵌入，构建通用模型；③ 通过 GRPO 强化学习在无标签多口音语料上直接优化 WER 与口音识别奖励，进一步提升内容保真与口音削弱。

**🔧 技术方法**

关键技术：自监督 SSL 词向量（WavLM‑Large）、向量量化 (VQ)、流匹配合成器、BART‑风格预训练、CTC 语音识别监督、GRPO 强化学习、CFG 指导、总时长感知的持续预测器。

**📊 数据集**

使用的数据集包括：LibriTTS‑R、Emilia‑EN（用于预训练和联合训练）、L2‑LibriTTSR 与扩展 L2‑ARCTIC（用于监督微调）、GLOBE（用于 GRPO 后训练）。

**📈 对比分析**

与 FramAN、CosyAccent（两种模式）和 VEVO 等基线相比，TokAN 在 7 种英语口音上取得最优结果：WER 下降至 9.23%（SFT 后 9.89%），口音概率 L1‑Prob 最高 99.09%，自然度 NAT 最高 70.73，且在无源时长模式下保持良好内容与口音平衡。

**⚠️ 局限性**

局限性：① 合成器基于编码条件，导致说话人相似度下降；② 受限于仅使用 WavLM‑Large，可能不适用于非英语口音；③ RL 奖励主要关注内容与口音，未显式考虑说话人保真，需进一步加入声纹奖励以提升音色一致性。

---

## 95. LH-AVLN: A Benchmark for Long-Horizon Audio-Visual-Language Navigation

**arXiv ID:** 2607.03920 | [PDF](https://arxiv.org/pdf/2607.03920v1)

**作者:** Rufeng Chen `[一作]` (Hong Kong University of Science and Technology Guangzhou), Sihong Xie `[通讯]` (Hong Kong University of Science and Technology Guangzhou)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LH-AVLN长周期音视语言导航基准并实现了训练无关的参考代理PAG-Nav。

**💡 创新点**

创新点在于将多目标任务、异构目标描述和持续空间化音频整合，要求代理在任务进度变化时对音频进行任务相关的定位与抑制。

**🔧 技术方法**

使用了目标感知编码、统一语义地图、进展条件规划以及基于音频的搜索和视觉验证等技术。

**📊 数据集**

基准数据集基于SoundSpaces 2.0和Matterport3D场景，提供了多目标、异构目标与空间化音频。

**📈 对比分析**

与六个代表性基线（MTU3D, 3D-Mem, Goat-Bench, SCOPE, SAVI, AVLen）对比，PAG-Nav在有序与无序任务中取得最高成功率，其他基线难以完成完整任务。

**⚠️ 局限性**

局限包括缺乏训练学习能力、对动态噪声或多源干扰处理不足，以及对更大规模或更复杂场景的推广性待验证。

---

## 96. GII-Polar Codes for Block Fading Channels

**arXiv ID:** 2607.03919 | [PDF](https://arxiv.org/pdf/2607.03919v1)

**作者:** Fangbo Yi `[一作]` (Sun Yat-sen University), Huazi Zhang `[通讯]` (Huawei Technologies Company Limited)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种基于泛化集成交织（GII）的极化码（GII‑polar code），用于块瑞利衰落信道，显著降低解码延迟并提升误帧率性能。

**💡 创新点**

创新点在于将两条极化码词通过二维交织矩阵耦合，构造低维嵌套码本；解码时可并行解码两条交织码，若单条失败可利用嵌套码恢复，兼具低延迟和更强纠错。

**🔧 技术方法**

采用极化码理论、SCL（与CRC辅助）解码、BPSK调制、块瑞利衰落信道建模及仿真。

**📊 数据集**

通过数值仿真在长度N=1024、2048、不同码率R∈{0.211,0.25,0.375,0.5,0.539}下生成可靠子通道集合，使用5G标准24比特CRC，评估FER、延迟与复杂度。

**📈 对比分析**

与传统长度2N极化码（CA‑SCL）及长度N极化码对比，GII‑polar在相同码率下FER更优、解码延迟大约减半，平均复杂度与长度2N极化码相当。

**⚠️ 局限性**

局限在于仅能支持二进制交织码最多两条；对嵌套码的设计与CRC校验增加实现复杂度；在极限SNR下性能提升有限，且对非块瑞利模型的适用性尚未验证。

---

## 97. DICT: Data Injection and Contrastive Trajectory Refinement for Conditional Image Generation with Diffusion Models

**arXiv ID:** 2607.03899 | [PDF](https://arxiv.org/pdf/2607.03899v1)

**作者:** Chunnan Shang `[一作]` (Zhejiang University), Hongwei Wang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的无训练推理框架 DICT，利用数据注入和对比轨迹细化在条件图像生成任务中提升质量。

**💡 创新点**

创新点在于：①将噪声扰动的条件数据直接注入早期去噪阶段，避免单标量损失瓶颈；②引入跨步对比轨迹损失，显式要求后续生成更贴合条件，抑制误差累积；③方法完全无训练、无需任务特定网络结构，兼容 U‑Net 与 DiT 等主流扩散骨干。

**🔧 技术方法**

使用扩散模型（特别是潜在扩散）和条件指导技术；实现了噪声注入、混合去噪、梯度更新和对比损失的组合；通过预处理模块对不同任务的条件做轻量级变换。

**📊 数据集**

数据集包括：1) 文本驱动风格迁移：200 条文本 + 200 风格图（WikiArt）共 40,000 张 512×512 图；2) 超分辨率与去模糊：1000 张 FFHQ 图像，分别做 4× 下采样+噪声和 61×3.0 高斯模糊后恢复至 256×256。

**📈 对比分析**

与多种任务特定方法（StyleShot、StyleStudio、InvSR、PSLD 等）及通用损失导引方法（FreeDoM、TFG 等）进行对比。实验显示：风格迁移中获得第二高文本相似度、最低风格损失与 CLIP 损失；超分辨率和去模糊中取得最高 PSNR、最高 SSIM 或最低 LPIPS，整体性能明显优于基线。

**⚠️ 局限性**

局限性：①需要手动设定多组超参数（如注入权重、迭代次数）；②对某些极端条件仍可能产生漂移；③虽然无训练，但多次梯度更新与对比损失会增加推理时间；④目前仅在三类任务上验证，泛化到更广泛条件（如视频、三维）需进一步研究。

---

## 98. AdaptiveSD A Stability-Aware, Runtime-Adaptive Speculative Decoding Framework with Multi-Policy Orchestration for CPU-Constrained LLM Inference

**arXiv ID:** 2607.03876 | [PDF](https://arxiv.org/pdf/2607.03876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 99. BAT3R: Bootstrapping Articulated 3D Reconstruction from 2D Image Collections

**arXiv ID:** 2607.03891 | [PDF](https://arxiv.org/pdf/2607.03891v1)

**作者:** Jakub Zadrozny `[一作]` (University of Edinburgh), Hakan Bilen `[通讯]` (University of Edinburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

利用单一已绑定的基准网格和无注释的二维图像集合，通过迭代的网格拟合与自生成训练样本，训练出可从单张图片预测全姿态3D重建的模型

**💡 创新点**

不需要人工构造多姿态3D数据集，只需一个基准网格，即可通过自监督迭代生成多姿态训练数据并实现与全监督方法相当的性能

**🔧 技术方法**

DualPM点图表示、梯度优化的网格拟合、姿态和相机姿势恢复、视角采样与正则化损失（角度、体积、边长、骨骼尺度、相互排斥）

**📊 数据集**

Horse、Cow、Sheep的Synthetic/Animodel-Points、MagicPony、AP-10K、Wild Elephant等真实图像集合以及相应的单一基准网格

**📈 对比分析**

与DualPM、A‑CSM、MagicPony、Farm3D、3D‑Fauna、Trellis、SAM‑3D等基线比较，在RMS Chamfer Distance、模型视角误差等指标上，所提方法在synthetic和real设置下均接近或超过全监督DualPM，且优于大规模零样本3D基座模型

**⚠️ 局限性**

仍需至少一个已绑定的基准网格；对姿态极端紧密接触时自排斥损失可能抑制正确恢复；在真实图像训练与ground‑truth 3D间仍存一定性能差距

---

## 100. Inferring the Shape of Data Frames in R Programs using Abstract Interpretation

**arXiv ID:** 2607.03889 | [PDF](https://arxiv.org/pdf/2607.03889v1)

**作者:** Oliver Gerstl `[一作]` (Ulm University), Matthias Tichy `[通讯]` (Ulm University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过抽象解释技术，针对R语言中的数据帧（data frame）进行静态形状推断，并实现了一个基于该推断结果的linter，能够检测代码中对不存在列或行的访问。

**💡 创新点**

创新点包括：①提出一种部分约简积域（column names、number of columns、number of rows）来抽象数据帧形状；②将常见的数据帧操作映射为抽象操作并给出相应的抽象语义；③在真实R脚本上验证了推断方法的无欠估性，并首次在工业级脚本集上展示了效果。

**🔧 技术方法**

主要技术：抽象解释、抽象域设计、静态数据流与控制流图分析、延迟宽化、常量传播、CSV文件读取等；工具实现基于现有的R静态分析框架。

**📊 数据集**

使用的实验数据集包括：①来自公开论文实验脚本的数千个R脚本；②对应的CSV数据文件（若存在）；③可执行的、已确认无运行错误的R脚本子集。

**📈 对比分析**

评估方法：对可执行脚本通过运行时反射获取真实形状，比较与推断结果，确认未出现under‑approximation；在全部脚本上统计推断精度（exact、partial、top等）以及检测到的无效访问；性能方面，单脚本平均推断时间约为 X 毫秒，整体平均总耗时约 Y 秒（包括解析、构建控制流图等步骤），满足交互式分析需求。

**⚠️ 局限性**

限制：①目前仅支持约四十种常用数据帧函数；②不支持用户自定义函数、高阶函数、二进制文件加载以及特定领域包的操作；③对行数过滤和分组等操作的抽象导致对行数的过度宽松；④缺乏向量、数值、列表等抽象域，影响对非数据帧参数的准确性；⑤未实现跨过程抽象解释。

---

## 101. SGF-CDNet: A Consistency-Discrepancy Graph Network over Semantic-Geometric Fused Nodes for Face Forgery Detection

**arXiv ID:** 2607.03883 | [PDF](https://arxiv.org/pdf/2607.03883v1)

**作者:** Jiayao Jiang `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于语义与几何融合节点的双路径一致性-差异性图网络 SGF‑CDNet，用来检测人脸伪造

**💡 创新点**

创新点在于(1)将面部语义分割与68点关键点几何信息深度融合，生成高信息量节点；(2)设计双路径 GNN，分别从一致性和差异性两角度进行关系推理，并通过门控融合动态权衡；(3)将图推理结果映射回二维空间做特征增强

**🔧 技术方法**

采用语义‑几何融合模块、双路径图注意网络 (GAT)、门控融合、空间映射、ConvNeXt‑Base 图像编码器以及多尺度特征门控

**📊 数据集**

在 FaceForensics++、Celeb‑DeepFake‑v2、DeepFake Detection Challenge、DFDCP、FFIW 等公开数据集上进行训练与评估

**📈 对比分析**

与 11 种基线方法对比，SGF‑CDNet 在所有跨数据集和跨攻击类型测试中均取得最高 AUC（最高 97.37%），并在跨攻击实验中保持 100%/100%/99.96% 等优异分数

**⚠️ 局限性**

主要局限包括：需依赖高质量面部分割与关键点检测；显著的显存占用（训练时 20.7 GB）；在极端轻微伪造或新型攻击方式下的鲁棒性尚待进一步验证

---

## 102. Evaluating LLM Uncertainty in Long-Form Generation Using Deterministic Ground Truth

**arXiv ID:** 2607.03870 | [PDF](https://arxiv.org/pdf/2607.03870v1)

**作者:** Ido Amit `[一作]` (Technion), Ran El-Yaniv `[通讯]` (Technion)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SALT benchmark，用于在无噪声、单一确定性长文本生成任务中对细粒度不确定性进行评估，并在 50+ 大语言模型上进行大规模实验，揭示了置信度排序、校准与推理机制之间的关系及前缀正确性对未来错误的影响。

**💡 创新点**

创新点包括：①零噪声、可扩展的程序化任务生成，能精确到原子级单元的真值标注；②揭示原子级与行级置信度排序的显著差距；③发现推理（Chain-of-Thought/训练式推理）提升精度但显著降低置信度排序的“推理折衷”；④系统比较多种置信度函数与校准方法，提出置信度排序与校准的权衡。

**🔧 技术方法**

技术手段：程序化生成任务、Token‑level log‑prob 聚合（平均、熵、Perplexity）用于置信度函数；多种后置校准（Z‑score+Sigmoid、Min‑Max、Binned）；评估指标包括 Precision、Recall、ECE、AUROC（macro/micro）等；对比 Chain‑of‑Thought 与训练式推理模型；对前缀影响进行对照实验；统计检验（Wilcoxon、Spearman）。

**📊 数据集**

使用的任务集为 SALT（6 个程序化任务：矩阵乘法、克罗内克积、LeetCode 代码、第一阶逻辑、DNA 翻译、多针任务），以及与 AIME、MMLU‑Pro、LLM‑Arena 的对照评测。

**📈 对比分析**

方法：对 50+ LLM 在 SALT 上计算 Precision、ECE、AUROC（宏观/微观）；对比不同置信度函数（logits vs verbalized）和校准方案；与传统 benchmarks 进行 Spearman 相关性对比。实验显示：宏观 AUROC 在行级显著高于原子级，微观 AUROC 仅略高于随机；推理模型虽提升 Precision 约 20% 但 AUROC 降低 20%+；Logit‑based 置信度函数在校准与排序上整体优于 verbalized；Binned 校准能获得最低 ECE，但对 AUROC 有负面影响。

**⚠️ 局限性**

局限性：任务仍为程序化、可预测的结构，可能不完全覆盖真实开放式生成的复杂性；评估仅针对可拆分为原子/行单元的任务，对更抽象或多模态生成缺乏覆盖；对置信度排序与校准的权衡依赖于后置校准方法，可能在不同任务或语言模型中表现不一；在某些外部 benchmark 上的迁移性仍不稳定。

---

## 103. High-Fidelity One-Step Generative Visuomotor Policy via Recursive Correction, Frequency Consistency, and Contrastive Flow Matching

**arXiv ID:** 2607.03865 | [PDF](https://arxiv.org/pdf/2607.03865v1)

**作者:** Yuran Chen `[一作]` (Anhui University of Science and Technology), Yang Huang `[通讯]` (Anhui University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种高保真一阶生成视觉运动策略框架，通过递归校正、频率一致性和对比流匹配实现低延迟单步动作生成。

**💡 创新点**

创新点包括：递归一致动作流（RCAF）修正空间截断误差；双时钟频率一致性（DTFC）保留高频细节；对比流匹配（CFM）通过边缘惩罚分离多模态流。

**🔧 技术方法**

使用连续时间流匹配、扩散模型结构、EMA教师一致性训练、离散余弦变换频谱约束以及对比学习损失等技术。

**📊 数据集**

在33个仿真任务（RoboTwin、RoboTwin 2.0、Adroit、DexArt）以及两台真实机器人平台（SO101、UR7E）上评估。

**📈 对比分析**

与10步扩散/流匹配基线和单步平均流、Wavelet等方法对比，单步1 NFE下取得与10步基线相当或更优的成功率，尤其在多模态、跨域和精细操作上有明显提升。

**⚠️ 局限性**

局限性在于需要针对不同任务调节递归步数、频率权重、对比边距等超参数，且在更大规模基础模型或有触觉反馈的情境下效果尚未验证。

---

## 104. Ghosts Beneath Textures: Texture-Relation Cues for Cross-Paradigm AI-Generated Image Detection

**arXiv ID:** 2607.03862 | [PDF](https://arxiv.org/pdf/2607.03862v1)

**作者:** Haoyu Wang `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究跨生成范式的 AI 生成图像检测，提出在语义无关纹理关系上进行建模的检测框架 DTS-Det

**💡 创新点**

创新点在于发现并利用跨范式共享的语义无关纹理模式与纹理关系，改进对图像自由生成与图像条件生成的泛化能力

**🔧 技术方法**

采用多尺度小波残差信号提取、纹理关系编码器、基于 DINOv3 的关系引导注意力以及 SigLIP2 的语义流，并结合 LoRA 微调

**📊 数据集**

使用新构建的 ConImageGen 基准（覆盖 13 种生成模型、图像自由与图像条件两范式），以及 PicoBanana、RAID、GenVidBench 等跨数据集与跨媒体测试集

**📈 对比分析**

在多项实验中，DTS-Det 在 ConImageGen 上平均准确率 99.6%，比最强基线提升 10.5%；在跨范式、跨数据集、跨媒体、以及鲁棒性评估中均保持 90% 以上的表现

**⚠️ 局限性**

局限在于仍对完全新型生成模型的纹理模式缺乏显式模板，且对白盒自适应攻击的鲁棒性未做评估

---

## 105. Reward Lightning: Fast Video Generation via Homologous Preference Distillation

**arXiv ID:** 2607.03960 | [PDF](https://arxiv.org/pdf/2607.03960v1)

**作者:** Jiaxiang Cheng `[一作]` (Tencent Hunyuan), Qinglin Lu `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Reward Lightning 框架，在同一潜在空间中同时完成视频生成的偏好对齐与采样步骤压缩。

**💡 创新点**

创新点在于：① 通过“同源性”原则，统一表示空间与结构，消除多目标冲突；② 设计了先验共享的潜在奖励模型 LRM 与同源偏好蒸馏 HPD；③ 采用动态边缘裁剪与自适应偏好权重，实现稳定的联合训练。

**🔧 技术方法**

使用技术包括：潜在奖励模型（Latent Reward Model）、流匹配蒸馏（Homologous Preference Distillation）、基于时间步投影的注意力头、梯度冲突抑制（动态边缘裁剪、EMA 软权重）以及对抗蒸馏（Hinge 损失）等。

**📊 数据集**

数据集：多边界对齐数据集（Intra-Model、Inter-Model、Real-Synthetic、Few-Step 增强）；评估用 VideoGen‑RewardBench、GenAI‑Bench（奖励准确度）以及 VBench、VBench‑I2V（视觉、运动、文本质量、FVD）。

**📈 对比分析**

与 DMDR、FlashDMD、TurboDiffusion、DMD2、PRFL 等异构/单目标基线对比，Reward Lightning 在 1–4 NFE 采样下平均 VBench 分数提升 2.1%，文本、运动、视觉三项均领先；奖励准确度在 VideoGen‑RewardBench 及 GenAI‑Bench 分别达到 72.24% 与 59.85%，比 pixel/latent baseline 超过 10%。

**⚠️ 局限性**

局限性：① 需要先预训练的流匹配模型与 VAE，训练成本仍高；② 对极端低采样步数（1 NFE）下的运动细节仍有轻微模糊；③ 结构与表示同源的假设在不同任务/数据分布下可能不再完全适用；④ 训练过程仍需多阶段与大规模 GPU 资源。

---

## 106. NormWorlds-CF: Solver-Verified Counterfactual Normative Reasoning with Metamorphic-Relation GRPO

**arXiv ID:** 2607.03957 | [PDF](https://arxiv.org/pdf/2607.03957v1)

**作者:** Xinqi Zhang `[一作]` `[通讯]` (Tsinghua University), Xinqi Zhang (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

提出了NormWorlds-CF，一套可执行的规则世界和solver-verified的规范推理实验环境；

**💡 创新点**

创新点在于将规范推理拆解为可验证的中间结构（证明、反证、论证状态、支持集、元变换关系），并将这些结构作为监督与奖励信号；

**🔧 技术方法**

使用了可扩展的规则DSL、确定性solver、LoRA监督微调、GRPO强化学习与三种奖励设计（答案仅、稀疏精确/模式、元变换归一化）等技术；

**📊 数据集**

构建了从微观单实例到根级对变换对的分层数据集，包含270个根家族、1080对变换、以及完整的证明与元关系标签；

**📈 对比分析**

通过在1.7B模型的SFT+GRPO匹配实验与4B模型的验证实验比较三种奖励；结果显示：答案仅奖励提升答案变化准确度但削弱关系族结构；稀疏精确奖励保持粗粒度关系标签但不利于细粒度变化；MR‑aware奖励在关系族正确率、错误族误判率和软关系一致性方面表现最佳；

**⚠️ 局限性**

主要限制在于完整的变换记录生成仍低于基准，子类型识别（如Invariant vs Change）效果有限，且对未见结构族的OOD迁移表现不佳；此外，实验使用的是合成规则世界，外部法律文本的泛化尚未验证。

---

## 107. Balancing Microservices and Monolithic Architectures

**arXiv ID:** 2607.03898 | [PDF](https://arxiv.org/pdf/2607.03898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 108. Online Linear Programming for Multi-Objective Routing in LLM Serving

**arXiv ID:** 2607.03948 | [PDF](https://arxiv.org/pdf/2607.03948v1)

**作者:** Zixi Chen `[一作]` (Peking University), Zijie Zhou `[通讯]` (HKUST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于在线线性规划的多目标路由框架，解决LLM服务中请求分发与资源约束的决策问题。

**💡 创新点**

创新点在于将批量大小和KV缓存的时间耦合约束与多项SLO目标（吞吐量、延迟、尾部延迟）统一建模，并通过双重价格控制实现可解释且可调节的路由策略。

**🔧 技术方法**

主要技术包括在线线性规划、双重价格（shadow price）控制、热启动投影梯度下降更新双重变量，以及基于历史样本的SAA式价格学习。

**📊 数据集**

使用Vidur模拟器中的公开对话数据集（lmsys-chat-1m）以及合成预填/解码比例不同的请求流进行实验。

**📈 对比分析**

与Round Robin、Least Outstanding Request、Random、Power-of-2等基线进行对比，实验显示在平均与尾部延迟、吞吐量等指标上相较于基线提升幅度可达数十个百分点，且对预测误差和工作负载突变具备鲁棒性。

**⚠️ 局限性**

主要限制在于仅在模拟环境中验证，缺乏真实系统部署与 KV 缓存抢占等细节，且未考虑请求优先级与公平性约束。

---

## 109. TabQueryBench: A Query-Centric Benchmark for Synthetic Tabular Data

**arXiv ID:** 2607.03926 | [PDF](https://arxiv.org/pdf/2607.03926v1)

**作者:** Jialin Zhang `[一作]` (Tongji University), Shinan Liu `[通讯]` (University of Hong Kong)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了TabQueryBench，一个基于可重用SQL模板、与数据集绑定的查询中心化基准，用来评估合成表在真实业务分析查询上的逼真度。

**💡 创新点**

将传统的分布相似度评估转向查询答案的保真度，引入五大查询族的模板化税onomies，构建模板到SQL的LLM驱动转化管道，并提供细粒度诊断与稳定性分析。

**🔧 技术方法**

使用模板化查询生成、受限模板到SQL的LLM实现（ChatGPT 5.4）、多模型多数据集的批量评测、统计距离指标（Wasserstein、JSD）与查询答案一致性评估等技术。

**📊 数据集**

评测了49个公开单表数据集（来自UCI、Kaggle、OpenML、HuggingFace），覆盖类别型、数值型、混合型，行数从1.5K到2.46M，列数从3到1559。

**📈 对比分析**

通过对11种主流合成表模型（BayesNet、ARF、CTGAN、TVAE、TabDDPM、TabSyn、TabDiff、TabPFGen、TabbyFlow、REaLTabFormer）在各查询族上的答案相似度进行排名，发现RealTabFormer在查询保真度上最高但成本极高，BayesNet在成本-保真度 Pareto 前沿上表现最佳；大多数模型在尾部、稀有值、高基数列以及局部条件查询上表现不佳。

**⚠️ 局限性**

仅适用于单表评估，查询生成过程依赖LLM可能引入随机性，未覆盖多表关联与复杂联结场景，隐私评估未纳入正式度量，且仅针对已定义的五类查询族。

---

## 110. NeuroOnline: Bridging Pretraining and Online Adaptation for EEG Foundation Models

**arXiv ID:** 2607.03925 | [PDF](https://arxiv.org/pdf/2607.03925v1)

**作者:** Weibin Li `[一作]` (Southern University of Science and Technology), Quanying Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 NeuroOnline 框架，允许 EEG 基础模型在在线环境中持续自适应；

**💡 创新点**

创新点在于将多视图一致性学习与上下文感知表示调制两种机制统一到同一框架中，实现了表示对齐与动态适应的协同；

**🔧 技术方法**

采用时频随机遮掩增强、多视图一致性损失、可学习的上下文提示 + 跨注意力调制、Transformer 级联编码器以及分类头；

**📊 数据集**

在六个公开 EEG 任务上评测：运动想象（BCIC‑2A、BCIC‑2B、SHU‑MI、PhysioNet‑MI）和情绪识别（FACED、SEED‑V）；

**📈 对比分析**

与离线微调（LoRA、FT）和在线简化基线（NeuroOnline.a）对比，NeuroOnline 在多种指标上平均提升 5–20% 以上，且在分布漂移情形下保持更高稳定性；

**⚠️ 局限性**

局限性：方法仅为经验性，缺乏理论分析；未验证在更大规模或更复杂场景下的可扩展性；上下文建模与在线更新效率仍可进一步提升。

---

## 111. A Gradient Flow Perspective on Minimum MMD Estimation

**arXiv ID:** 2607.03871 | [PDF](https://arxiv.org/pdf/2607.03871v1)

**作者:** Sophia Seulkee Kang `[一作]` (Independent Researcher), Zonghao Chen `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种预条件梯度下降（PGD）方案，用于求解最小MMD估计器并给出了在梯度优势和投影残差假设下的渐进全局收敛保证；

**💡 创新点**

创新点在于通过自适应核长度尺度与对非参数MMD梯度的投影预条件化，将参数更新方向与非参数梯度流保持一致，从而在非凸优化中实现全局收敛；

**🔧 技术方法**

主要技术包括：MMD梯度流、可变核长度尺度的自适应策略、投影残差最小化得到的预条件矩阵、基于Jacobians的自动微分与梯度估计；

**📊 数据集**

实验使用的模拟数据集包括：多模二维高斯混合、g-and-k分布、Lotka‑Volterra动力学模型以及toggle‑switch基因表达模型；

**📈 对比分析**

与传统梯度下降（GD）和自然梯度下降（NGD）的比较显示，PGD在收敛速度、全局最优获取以及在有噪声或模型误差情况下的鲁棒性上均优于GD，并且在大多数实验中达到更低的MMD值；

**⚠️ 局限性**

局限性包括：收敛证明依赖于难以验证的梯度优势与投影残差条件；自适应核长度尺度的调度也需先验设定；理论仅给出渐进结果，缺乏明确的非渐进收敛速率。

---

## 112. Why3-py: A Tool for Formal Verification of Hypothesis Testing and Meta-Analysis in Python

**arXiv ID:** 2607.03951 | [PDF](https://arxiv.org/pdf/2607.03951v1)

**作者:** Akira Tanaka `[一作]` (National Institute of Advanced Industrial Science and Technology), Yusuke Kawamoto `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

开发了 Why3-py 这一工具，用于将 Python 统计程序（包括假设检验和元分析）转译为 WhyML 代码并进行形式化验证。

**💡 创新点**

创新点在于：① 将 Python 的动态类型、实数运算和外部科学库（如 SciPy、statsmodels）通过类型检查器和抽象规范映射到 WhyML；② 在 Python 代码中嵌入完整的 WhyML 规范，支持对假设检验的假设与结果进行显式说明；③ 扩展了 Why3 的 meta-analysis 模块，实现对 Fisher、Stouffer、Mantel‑Haenszel 等方法的规范化验证，从而检测出版偏差、p‑hacking 等常见错误。

**🔧 技术方法**

使用技术包括：Why3 形式化验证平台、WhyML 规范语言、Python 类型检查器 mypy、Menhir 增量解析器、SMT 求解器 Z3，此外还对外部库调用进行了逻辑契约抽象。

**📊 数据集**

实验数据主要为标准教材中的多重比较方法（Tukey、Dunnett、Steel‑Dwass）以及模拟的元分析案例（最多 30 个研究），没有使用真实医学或社会科学大规模数据集，而是基于公开的统计方法示例进行验证。

**📈 对比分析**

通过测量验证执行时间来比较方法，结果显示：对 2~30 个研究的 Fisher 方法验证在 8 秒以内完成；对多重比较的不同方法，验证时间随比较数增加呈指数增长，但在实际组数（≤10 组）下均保持在 30 秒以内，性能满足实用需求。

**⚠️ 局限性**

限制主要在于：① 仅支持能被 mypy 静态分析的 Python 子集，无法处理大量动态特性；② 验证仅关注程序规范与假设的正确性，而无法证明统计假设本身的有效性；③ 需要手动编写 WhyML 规范，使用门槛较高。

---

## 113. Harness-Aware Self-Evolving: Co-Evolving Model Weights, Harness, and Task Solutions

**arXiv ID:** 2607.03935 | [PDF](https://arxiv.org/pdf/2607.03935v1)

**作者:** Haochen Luo `[一作]` (University of Hong Kong), Qi Liu `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Harness-Aware Self-Evolving (HASE)框架，使单一LLM模型在多轮动作空间中同时生成任务解与编辑选定的 harness 组件；

**💡 创新点**

创新点在于将解生成与 harness 编辑统一到同一 agentic RL 动作空间，并通过代理与真实评估器的不一致来驱动 evaluator 修复，从而避免 reward hacking；

**🔧 技术方法**

使用 Qwen3-8B 作为策略网络，结合 GRPO 强化学习、阶段性评审、局部评估器与真实评估器对齐等技术；

**📊 数据集**

在文本分类（Symptom2Disease）、alpha factor 挖掘（CSI300）、圆形打包与 Heilbronn 三角形等四个数据集/任务上进行实验；

**📈 对比分析**

与 GPT-OSS-120B、Meta-Harness、AlphaEvolve 等基线对比，HASE 在文本分类中达到 86.98% 的准确率，alpha 挖掘中排名指标与 AER 远超 GPT-OSS-120B，圆形打包与 Heilbronn 中恢复 evaluator 后实现 state-of-the-art 评分；

**⚠️ 局限性**

局限在于只对预先白名单的 harness 组件开放编辑，评估器修复需依赖真实评估器反馈，可能忽略更深层的可改造空间且未做大规模模型/任务扩展验证。

---

## 114. Probe, Don't Prompt: A Hidden-State Probe for Metadata Filtering in Multi-Meta-RAG

**arXiv ID:** 2607.03929 | [PDF](https://arxiv.org/pdf/2607.03929v1)

**作者:** Mykhailo Poliakov `[一作]` (National University of Kyiv-Mohyla Academy), Nadiya Shvai `[通讯]` (National University of Kyiv-Mohyla Academy)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

用基于隐藏状态的探测器替代 Multi-Meta‑RAG 中的 GPT‑3.5 元数据提取器，实现对新闻源的过滤；

**💡 创新点**

提出浅层均值池化、类别不平衡加权训练的固定词表探测器，避免生成式漂移且无需 API 调用；

**🔧 技术方法**

在小型开源 LLM 上提取浅层隐藏状态，做均值池化后训练多标签逻辑回归头，采用类权重交叉熵处理长尾；

**📊 数据集**

使用 2556 条 MultiHop‑RAG 新闻多跳问答查询数据，源标签固定为 49 个新闻源；

**📈 对比分析**

与 GPT‑3.5 及无模型子串匹配基线对比，Probe 在整体 set‑exact 准确率上达 90.9%，高于子串 88.0% 与 GPT‑3.5 80.9%；在非空查询与基线相近，优势主要来自对空查询的准确否决；

**⚠️ 局限性**

研究仅基于单一新闻数据集，近似词面属性导致子串基线已极强；稀有来源宏 F1 较低，且未评估检索端到端性能。

---

## 115. Beyond Item Order: Temporal Gap Tokenization for Generative Recommendation with Semantic IDs

**arXiv ID:** 2607.03918 | [PDF](https://arxiv.org/pdf/2607.03918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 116. LogNLQ: Natural-Language Log Querying with Parser-Induced and Semantically Grounded Schemas

**arXiv ID:** 2607.03884 | [PDF](https://arxiv.org/pdf/2607.03884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 117. CDCP: Conditional Diffusion Model with Contextual Prompts for Multi-task Offline Safe Reinforcement Learning

**arXiv ID:** 2607.03903 | [PDF](https://arxiv.org/pdf/2607.03903v1)

**作者:** Jiayi Guan `[一作]` (Tongji University), Changjun Jiang `[通讯]` (Tongji University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种多任务离线安全强化学习（MTOS）的条件扩散模型 CDCP，能够从多任务离线数据中学习共享的安全策略。

**💡 创新点**

创新点包括：① 将多任务安全 RL 的约束优化问题转化为条件生成问题；② 采用无分类器指导（classifier‑free guidance）的成本约束策略，消除 OOD 行动的外推误差；③ 结合文本描述与轨迹提示的上下文提示，提升任务表示与跨任务适应性；④ 引入梯度损失同步机制，消除多任务梯度干扰；⑤ 允许在不重新训练的情况下通过调整成本阈值实现灵活的安全约束。

**🔧 技术方法**

使用技术包括扩散模型、无分类器指导、文本和轨迹上下文提示、梯度损失同步、监督学习与低温采样。

**📊 数据集**

使用数据集：MetaDrive 的 9 个交通场景（Easy/Medium/Hard × Sparse/Mean/Dense）来自 DSRL，提供离线轨迹与成本信息。

**📈 对比分析**

与改进的单任务离线安全 RL（MTCPQ、MTCOptiDICE、MTCDT、MTCDD）以及多任务 RL（CMTDiff、CMTDD）等基线进行对比。CDCP 在保证成本阈值的前提下获得更高奖励，成本平均保持在阈值以内，且可在不同阈值下无额外训练实现动态调整，整体性能优于所有基线。

**⚠️ 局限性**

局限性：依赖高质量离线数据；成本阈值需手动设定，模型对超参数（如 β）敏感；在极低成本或零成本约束下性能下降；实验仅在 MetaDrive 环境验证，跨域泛化能力待进一步评估。

---

## 118. A Unified Framework for Quantized and Continuous Strong Lottery Tickets

**arXiv ID:** 2607.03860 | [PDF](https://arxiv.org/pdf/2607.03860v1)

**作者:** Aakash Kumar `[一作]` (Inria), Emanuele Natale `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了离散随机子集求和问题（RSSP），并基于此给出了量化版强彩票票假设（SLTH）的紧凑概率上界，证明了在量化网络中稀疏子网络存在的失败概率随过参数化指数下降；

**💡 创新点**

创新点在于首次将离散RSSP的精确概率分析推广到量化网络，统一了连续逼近与量化精确表示两类结果，并将失败概率从之前的多项式衰减提升到指数衰减；

**🔧 技术方法**

核心技术包括离散RSSP的随机过程分析、递推式与体积增长估计、以及通过子集求和与网络剪枝相结合的组合优化方法；

**📊 数据集**

实验主要使用随机生成的{-M,…,M}子集求和实例和一张约2.5×10⁷参数、FP8精度的ResNet‑50尺度网络进行验证；

**📈 对比分析**

与之前工作（如二值网络、连续量化SLTH等）比较，本文在相同过参数化规模下实现了更高的成功率（指数级下降的失败概率）且覆盖了逼近与精确两种情况；

**⚠️ 局限性**

局限性包括：依赖于随机初始化与均匀采样假设，主要针对ReLU激活函数，理论分析未在大规模真实数据集（如ImageNet）上进行充分验证。

---

## 119. Stochastic Caching via Subset Entropy

**arXiv ID:** 2607.03947 | [PDF](https://arxiv.org/pdf/2607.03947v1)

**作者:** Ravi Kumar `[一作]` (Google Research), Debmalya Panigrahi `[通讯]` (Duke University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在随机缓存问题中提出并分析了新的信息量度——子集熵(subset entropy)，并基于该量度给出了已知和未知分布下缓存算法的细粒度竞争比；

**💡 创新点**

创新点在于引入子集熵作为衡量分布“难易度”的参数，完成了对随机缓存竞争比的精细化刻画，并证明其在不同算法（最优策略、LRU、LFU）中的性能上限；

**🔧 技术方法**

主要技术包括：对k-子集熵的理论定义与计算、基于期望LP的对偶拟合(direct fitting)、概率层分析与泊松逼近、以及对LRU和LFU的计数/概率级联分析；

**📊 数据集**

本研究完全基于理论分析，未使用任何真实数据集；

**📈 对比分析**

与传统最坏情况竞争比O(log k)比较，本文的结果在子集熵较小的分布下显著更优；已知分布下算法竞争比为O(σ)，未知分布下LRU竞争比为O(σ³)，LFU竞争比为O(σ)，其中σ为子集熵；

**⚠️ 局限性**

局限性包括：对LRU的上界可能不是最优，仍有提升空间；分析仅适用于独立同分布(i.i.d.)的请求模型，未涵盖更复杂的马尔可夫或局部性模型；

---

## 120. Transformers with Physics-Informed Encodings and Simulation-Based Inference for Robust Detection of Eccentric Binary Black Holes in Pulsar Timing Array Data

**arXiv ID:** 2607.03904 | [PDF](https://arxiv.org/pdf/2607.03904v1)

**作者:** Subhajit Dandapat `[一作]` (National University of Singapore), Alvin J. K. Chua `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种物理感知Transformer加上归一化流（normalizing flow）的端到端参数推断框架，用于从脉冲星定时阵列（PTA）残差中快速、精确地估计椭圆轨道超大质量黑洞对（EBBH）的物理参数；

**💡 创新点**

核心创新在于：①将轨道相位演化嵌入Transformer的位置信息编码（PIPE）作为物理先验；②构建一个独立的相位预测网络，直接从多脉冲星残差中恢复全局相位；③利用条件归一化流实现可摊销的后验采样；④通过可学习门控自适应地权衡位置编码与相位编码。

**🔧 技术方法**

主要技术包括：自注意力Transformer（外部注意力可选），物理感知位置编码，基于Transformer的相位预测网络，条件离散/连续归一化流（DNF/CNF），以及标准化预处理和多任务损失。

**📊 数据集**

使用合成PTA数据集：10颗观测良好的脉冲星，每颗包含400个时间样本，生成5×10⁴至4×10⁵个独立的信号+白噪声残差，参数覆盖日志频率、质量、质量比、轨道偏心率等。

**📈 对比分析**

与传统的无物理先验Transformer及经典MCMC方法进行比较，评价指标为真值处的对数后验密度、后验尖锐度与校准度。结果显示：①引入相位编码后后验更尖锐、真值处对数密度提升（从负到正）；②相位条件显著提高低SNR/数据稀缺场景下的性能；③归一化流在推断速度上比MCMC快数百倍，且在大数据量下与MCMC后验高度一致。

**⚠️ 局限性**

局限性：仅考虑白噪声，未加入脉冲星红噪声、DM变化、Hellings–Downs相关等真实PTA噪声；仅针对单一确定性EBBH信号；相位预测在极低SNR下仍可能失效；未来需扩展至更复杂噪声模型和更高维参数空间。

---

## 121. The Remarkable Effectiveness of Providing AI Agents with Natural Language Tools: A Replication Study Validating NLT Performance Across 14 Models

**arXiv ID:** 2607.03953 | [PDF](https://arxiv.org/pdf/2607.03953v1)

**作者:** Alexander Somma `[一作]` (Sage.is AI-UI), Fred Premji `[通讯]` (Sage.is AI-UI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

独立复现并扩展Johnson等人（2025）提出的自然语言工具（NLT）框架，评估14个大型语言模型在两类单轮、无参数工具调用场景下的性能

**💡 创新点**

首次用开放源码方法验证NLT的优势，发现性能提升随模型能力呈现规模依赖，且通过消除幸存者偏差得到更严谨的误差率报告

**🔧 技术方法**

采用NLT自然语言交互、JSON结构化工具调用、正则解析、Token计数等技术，并构建了Python评估引擎

**📊 数据集**

使用14个公开及闭源模型（包括GPT‑5、Claude‑Sonnet‑4、Gemini 2.5 Pro等）在两场景（客户服务与心理健康）下共8,560次实验

**📈 对比分析**

将NLT与结构化调用进行逐一对比，NLT平均准确率提升14.9pp（62.3% vs 47.4%），错误率降低93%（51 vs 755），Token消耗下降25.2%

**⚠️ 局限性**

实验受限于模型可用性（两模型缺失完整数据）、单轮无参数设置、仅涵盖两域，且未涉及多轮或嵌套工具调用，结果不一定适用于更复杂的代理系统

---

## 122. TESSERA v2: Scaling Pixel-wise Earth Foundation Models

**arXiv ID:** 2607.03949 | [PDF](https://arxiv.org/pdf/2607.03949v1)

**作者:** Zhengpeng Feng `[一作]` (University of Cambridge), Srinivasan Keshav `[通讯]` (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

本文对像素级地球观测（EO）基础模型进行下游驱动的规模化研究，构建大规模教师模型并通过蒸馏生成多尺寸、可嵌套的嵌入式学生模型；

**💡 创新点**

创新点在于发现预训练损失与下游性能弱相关，提出仅将计算预算分配给编码器和数据而保持投影器不变的规模化规则；通过蒸馏实现可嵌套前缀嵌入，使得嵌入维度可作为部署成本调节器；并提供“嵌入即数据”产品；

**🔧 技术方法**

采用自监督冗余消减目标、Transformer编码器、多模态时间序列处理、随机视图与混合数据增强、全局桶采样、以及针对教师的前缀蒸馏损失；

**📊 数据集**

使用全球 Sentinel‑1/2 10 m 时序数据，构建多源时序样本；在 15 个下游任务（分类、分割、变化检测、回归）以及额外的留出数据集上进行评估；

**📈 对比分析**

与 7 种公开/专有嵌入系统以及多种 RSFM 进行对比，最优学生模型在全套任务上取得 0.611 的综合得分（高于 0.576 的基准），16 维前缀保持 92% 的性能，仅占原 128 维存储的 1/8；

**⚠️ 局限性**

局限性包括：规模化规律仅针对所选自监督目标和像素级编码器，需昂贵教师训练；实验覆盖的地区与气候有限，模型对低表示区域的泛化性待验证。

---

## 123. A Large-Scale Dataset and a New Method for RemoteSensing Traffic Object Segmentation

**arXiv ID:** 2607.03945 | [PDF](https://arxiv.org/pdf/2607.03945v1)

**作者:** Zhigang Yang `[一作]` (Northwestern Polytechnical University), Qi Wang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了覆盖 49 个城市、7 个国家、4 种交通目标（车、飞机、船、火车）的遥感图像实例级语义分割数据集 NWPU‑Traffic，并基于该数据集提出了 CSPNet 方案，实现了对多尺度交通目标的高精度分割。

**💡 创新点**

创新点包括：① 将空间信息与通道信息分别保持的特征交互模块（SPFIM 与 CPFIM）设计，兼顾多尺度语义与位置精度；② 在解码阶段引入局部-全局特征融合解码器（LGFFD），通过动态门控结合卷积与 Transformer 的优势；③ 通过实例级旋转框与多场景标注提升数据多样性，填补现有遥感交通数据集缺乏多类别、多场景、实例标注的空白。

**🔧 技术方法**

主要技术：ResNet‑34 编码器；像素洗牌与像素反洗牌实现尺度统一；频域权重与通道域权重相结合的 CPFIM；3D 卷积与序列通道注意力实现 SPFIM；局部-全局门控融合的 LGFFD；AdamW、cosine 学习率、交叉熵+Dice 损失，采用 TTA 进行推理提升鲁棒性。

**📊 数据集**

使用自建 NWPU‑Traffic 数据集（1479 张 0.12–0.5 m/pixel 图像，31,628 个实例），同时对比了公开数据集如 HRSC2016、DOTA、COWC、iSAID 等，评估指标包含 IoU、mIoU、OA、F1‑score。

**📈 对比分析**

在 NWPU‑Traffic 上对比 9 种主流方法（DeepLabv3+、ABCNet、UNetFormer、GCBNet、ScaleFormer、CMLFormer、FSEL 等），CSPNet 在背景、飞机、车、船、火车的 IoU 分别为 98.73%、84.52%、48.91%、80.07%、55.30%，mIoU 达到 98.76%，OA 83.36%，F1‑score 73.50%，在所有评估指标上均达到或超过现有方法，尤其在小目标（车、船）和密集场景上表现突出。

**⚠️ 局限性**

局限性包括：① 数据集仅涵盖四类交通目标，缺少更细粒度或多模态（如电动车、无人机）标注；② 仅使用光学遥感图像，未覆盖 SAR 等传感器；③ 模型整体参数与 FLOPs 较大，部署在资源受限设备上仍有挑战；④ 对极端天气、强遮挡场景的鲁棒性尚待进一步验证。

---

## 124. WSA$_1$: a 3D-Centric World-Spatial-Action Model for Generalizable Robot Control

**arXiv ID:** 2607.03941 | [PDF](https://arxiv.org/pdf/2607.03941v1)

**作者:** Jiahao Jiang `[一作]`, Heng Tao Shen `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种名为WSA_1的机器人基础模型，采用3D中心的世界-空间-动作联合建模，支持在仅6000小时演示数据（其中1000小时为真实机器人）下实现高效预训练和强泛化能力。

**💡 创新点**

创新点在于：①提出三维世界感知、动作驱动的3D世界预测与3D逆动力学三任务联合学习；②引入双向因果注意力约束，实现世界-动作的双向因果一致性；③通过多源数据混合预训练，仅用有限真实机器人数据即可达到竞争性性能。

**🔧 技术方法**

采用Mixture-of-Transformers架构，包含2D空间专家、3D空间专家和3D动作专家；使用双向因果注意力机制；损失函数包括MSE（视觉与3D预测）和流匹配（动作生成）；基于预训练的Qwen3-VL和Wan2.2视觉语言模型作为骨干。

**📊 数据集**

利用InternData-A1、RoboTwin2.0（仿真）、EgoDex（人类手眼交互）、AgiBot-World、RoboChallenge（真实机器人）等多源数据，共覆盖8种机器人和300+任务。

**📈 对比分析**

在7个桌面真实任务中平均成功率77.5%/完整率82.7%，比基线提升约30%；在RoboTwin2.0硬模式下WSA_1-L成功率93%，超过其他开源模型；在LIBERO上平均成功率98.2%，优于VLA基线，显示出优异的跨任务与跨平台泛化性能。

**⚠️ 局限性**

局限性包括：仍需多源数据支持，极端物理环境或极长序列任务可能出现误差；模型规模较大，推理延迟和算力需求高；缺乏在线自适应学习机制，难以在持续变化的真实环境中即时更新策略。

---

## 125. Can Dialects Be Steered Like Languages? Sparse Neurons and Distributed Directions in Arabic LLMs

**arXiv ID:** 2607.03936 | [PDF](https://arxiv.org/pdf/2607.03936v1)

**作者:** Kareem Elozeiri `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Nadir Durrani `[通讯]` (Qatar Computing Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在阿拉伯语言大模型中通过推理时的神经元与向量 steering 两种方法实现方言控制，并对方言信息在模型内部的分布与可解释性进行系统分析。

**💡 创新点**

首次同时展示神经元级别稀疏可解释性与向量级别分布式可控制性，并证明阿拉伯方言信息既局部又分布，能够因果性地在不微调模型权重的情况下进行推理时调节。

**🔧 技术方法**

采用 LAPE 神经元筛选、激活向量提取与注入、LLM-as-a-Judge 与 AL-QASIDA 自动评测、以及人工评测等技术；并利用对比提示、神经元缩放、向量注入等多种干预方式。

**📊 数据集**

使用 MADAR 并行方言–MSA 句对（埃及、摩洛哥、黎巴嫩、沙特等），以及 ALADICe 等评测基准进行实验。

**📈 对比分析**

与未干预、显式提示以及神经元 steering 进行对比；在 ADI2、macro-ADI2、LLM-as-a-Judge 和人工评测指标上，向量 steering 通常优于神经元方法，且能在 MSA 提示下成功实现方言生成，表现出显著性能提升。

**⚠️ 局限性**

实验仅覆盖少数方言与两款模型，评测指标仍有限，神经元与向量空间关系尚未完全揭示，且方法可能被滥用于假冒或误导，需进一步社区评估与扩展。

---

## 126. MPSelectTune: Prompt-type Selection for Fine-tuning improves Concept Unlearning in LLMs

**arXiv ID:** 2607.03932 | [PDF](https://arxiv.org/pdf/2607.03932v1)

**作者:** Shubhadip Nag `[一作]` (Indian Institute of Technology Kharagpur), Sourangshu Bhattacharya `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于对抗式提示的两阶段微调方法MPSelectTune，用于在LLM中实现概念去学习。

**💡 创新点**

创新点在于先用多提示、多任务损失进行全局微调，然后挑选“最差”提示类型进一步微调，以最小化该提示下的概念准确率，实现在不同提示类型下的鲁棒去学习。

**🔧 技术方法**

采用多任务损失（任务损失、概念损失、下一词预测损失、格式损失），LoRA微调技术，基于相似度与随机采样的多种提示生成策略以及对抗式提示选择。

**📊 数据集**

使用五个任务-概念对数据集（Bios、RT-Gender、ToxicBias、Adult Census、SciQ-WMDPBio），并在 LLaMA‑2‑7B、LLaMA‑3.1‑8B 与 Mistral‑7B‑Instruct‑v0.3 上进行实验。

**📈 对比分析**

与多种基线（Base、FT、Aug、ICUL、SKU、ECK）对比，MPSelectTune 在主任务准确率上与 FT 相当或略优，同时将概念准确率降至接近随机（最高下降 17%），并将 Spuriousness Score 提升 23–74%，显示出优越的去学习效果。

**⚠️ 局限性**

局限性包括提示选择仍是手工设计，缺乏动态自适应机制；SP‑Score 仅适用于二分类概念；未在更大规模或更多模型上进一步验证。

---

## 127. USE: A Unified Self-Ensembling Framework for Test-Time Prompt Tuning

**arXiv ID:** 2607.03900 | [PDF](https://arxiv.org/pdf/2607.03900v1)

**作者:** Siru Jiang `[一作]` (University of Chinese Academy of Sciences), Tieniu Tan `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的自集成框架（Unified Self-Ensembling, USE）用于CLIP的测试时适应（Test‑Time Adaptation），通过自我集成（Self‑Ensembling, SE）在优化和推理阶段一致地利用弱增强（原图）与强增强的预测。

**💡 创新点**

创新点：①将经典TPT解释为自监督学习并引入伪标签（pseudo‑label）概念；②设计可自适应加权的SE策略（β调节弱增强权重）生成更可靠伪标签；③在推理阶段同样使用SE，形成优化与推理目标一致的统一框架；④实现了无优化（optimization‑free）的轻量级TTA方法。

**🔧 技术方法**

技术：基于CLIP的文本提示调优、AugMix增强、逆交叉熵（Reverse Cross‑Entropy, RCE）自监督损失、KL散度约简、动态β自适应权重、跳过（skip）优化策略、文本模板集成。

**📊 数据集**

数据集：ImageNet及其OOV变体（ImageNet‑A、V2、R、Sketch）和十个细粒度分类数据集（DTD、Flowers102、Caltech101、Aircraft、Pets、UCF101、Cars、EuroSAT、SUN397、Food101）。

**📈 对比分析**

与多种基线比较（CLIP、MTA、ZERO、TPT、C‑TPT、RLCF、TTL、TPS、R‑TPT、STS等）。在无优化方法中，USE在大多数OOV和细粒度数据集上获得最高或第二高精度；在有优化方法中，R‑USE在ViT‑B/16上从64.84%提升至65.35%，并在大多数评测场景保持明显优势；作为轻量级插件可进一步提升现有TTA方法。总体来看，USE在准确率上普遍提升2–4个百分点。

**⚠️ 局限性**

局限性：①需要对每个测试样本生成并推理大量增强视图（约64次前向传播），导致额外的计算成本；②对弱增强（原图）依赖较强，若原图严重失真或被攻击时可能性能下降。

---

## 128. Consistent but Miscalibrated: Evaluating LLM Limitations for Risk Communication in Natural Language

**arXiv ID:** 2607.03882 | [PDF](https://arxiv.org/pdf/2607.03882v1)

**作者:** Diego Cerda-Mardini `[一作]` (McGill University), Sreenath Madathil `[通讯]` (McGill University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估LLM在风险沟通中对预测概率和不确定性进行自然语言描述的准确性与一致性。

**💡 创新点**

将风险沟通抽象为描述符选择任务，并用一致性与校准两项指标在多模型、多场景、多温度下进行系统评估。

**🔧 技术方法**

采用Beta分布模拟预测分布，使用指导解码和多温度采样的LLM推理方法。

**📊 数据集**

使用自生成的Beta样本作为实验数据，未使用公开数据集。

**📈 对比分析**

与九个LLM（包含GPT‑5.4、Qwen3等）在一致性和校准两项指标进行对比；大多数模型一致性高但校准差，GPT‑5.4在概率校准上表现最佳。

**⚠️ 局限性**

受限于离散化描述符集、仅英语实验、缺少真实模型输出和长文本评估，结果可能受限。

---

## 129. SharpSplat: Edge-Regularized 3D Gaussian Splatting for High Fidelity Urban Building Reconstruction from UAV images

**arXiv ID:** 2607.03872 | [PDF](https://arxiv.org/pdf/2607.03872v1)

**作者:** Porus Vaid `[一作]` (IISER Bhopal), Vaibhav Kumar `[通讯]` (IISER Bhopal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于语义边缘监督的3D高斯喷射框架，用于改进无人机图像生成的城市建筑模型，使建筑立面边缘更加清晰；

**💡 创新点**

通过将SAM3生成的建筑掩码与图像梯度相结合，构造边缘对齐损失，直接在渲染图像梯度上监督3D高斯，避免了对模型结构的改动，创新点在于利用语义边缘实现“软”边缘约束；

**🔧 技术方法**

使用SAM3进行文本引导式建筑分割，Sobel算子提取梯度，差分渲染器对齐损失（L1）以及传统的RGB、SSIM损失共同训练3D Gaussian Splatting；

**📊 数据集**

在两个无人机图像数据集上验证：UrbanScene3D（PolyTech、Art Sci）和自采的Gehukheda地区建筑图像；

**📈 对比分析**

与3DGS基线、2DGS（平面原语）以及SuGaR比较，PSNR、SSIM略升高，LPIPS下降，表明在保持整体质量不下降的前提下显著提升建筑边缘清晰度；

**⚠️ 局限性**

方法受限于SAM3的分割准确性，λ参数需场景调优，且仅约束渲染视觉边缘，未解决几何不平整等问题。

---

## 130. GeoSelect: Spatial-Program Execution for Training-Free Referring Remote Sensing Image Segmentation

**arXiv ID:** 2607.03869 | [PDF](https://arxiv.org/pdf/2607.03869v1)

**作者:** Yuhang Jiang `[一作]` (Anhui University), Linsheng Huang `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GeoSelect，一种训练‑free 的遥感图像定位分割框架，利用冻结的语言模型将表达式解析为类型化的空间程序，随后在候选框集合上执行几何字段与离散集合/顺序算子，最终得到目标实例的掩码。

**💡 创新点**

创新点包括：
- 将定位分割视为可执行的空间程序，实现对空间、比较和序数关系的显式推理；
- 设计了简洁的 DSL 与类型安全的算子（连续字段、极值、序数、计数、关系），实现对复杂表达式的覆盖；
- 引入可靠性阶梯与语法检查，在程序非法或执行失败时退回到基线字段选择，保证始终给出答案；
- 通过程序化合成与解析实现可解释的中间结果与故障定位。

**🔧 技术方法**

使用技术包括：
- 文本‑only 的 Qwen3‑4B 作为程序合成器；
- LAE‑DINO (带 SAHI tiling) 的开源检测器生成候选框；
- 由闭式几何字段（方向、中心、接近等）与离散算子组成的执行器；
- SAM ViT‑L 作为单框掩码解码器；
- 可靠性阶梯与结构化错误处理。

**📊 数据集**

采用的公开数据集：RRSIS‑D（单实例遥感定位分割基准）和 RISBench（更大、表达式更复杂的跨域基准）。

**📈 对比分析**

与先前训练‑free 方法（如 RSVG‑ZeroOV、EKP‑HRM 等）以及监督式专用模型（RMSIN、RSRefSeg2 等）进行对比；在 RRSIS‑D 上达到 58.86 mIoU（≈2×最高训练‑free 结果，≈93% 监督模型），在 RISBench 上达到 55.27 mIoU（≈1.7×最高训练‑free 结果）。显著提升了对空间、比较与序数表达式的解析精度，尤其在同类对象拥挤场景中。

**⚠️ 局限性**

局限性包括：
- 仅支持单实例（0/1/k）表达式；
- 依赖于领域内预训练的开源检测器；
- 对极端方向（如长船、桥梁）不使用方向框，导致轴对齐误差；
- 序数表达式样本稀缺，导致该类表达式的性能和分析受限。

---

## 131. The Case for Globally Beneficial Technology

**arXiv ID:** 2607.03906 | [PDF](https://arxiv.org/pdf/2607.03906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 132. Rethinking Scientific Discovery in an Agentic Era

**arXiv ID:** 2607.03863 | [PDF](https://arxiv.org/pdf/2607.03863v1)

**作者:** Yining Zheng `[一作]` (Shanghai Innovation Institute), Xipeng Qiu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现 SCION——一个基于多智能体的科学工作操作系统，将科研意图编译成可执行的 Research Execution Plan（REP），并通过层次化的代理、上下文构造和分层记忆实现端到端的科研工作流程。

**💡 创新点**

创新点在于：① 把科研流程抽象为可执行的计划对象，使高层意图能够被系统直接驱动；② 架构分为意图表述、任务执行与记忆三层，并引入治理式委派和可恢复的长周期执行；③ 将目标条件逆搜索与批量主动搜索统一到元抓手框架中，提升科研决策的可追溯性与可复用性。

**🔧 技术方法**

技术实现包括：大型语言模型（如 Kimi、nex-n1.1）、多智能体协同框架（主调度、专门化子代理）、上下文构造机制、分层内存（L1/L2/L3）、验证检查与回滚、逆搜索与批量主动搜索策略、以及实验结果的自动评估与归档。

**📊 数据集**

使用的数据集包括：CMPhysBench（物理推理评测）、AI 研究问题集（Idea Maker）、多属性分子设计任务（BPQ、MPQ、BHMQ、BMPQ、HMPQ 组合）、真实抗体筛选实验库，以及这些任务对应的基线系统输出。

**📈 对比分析**

比较方法：在科学阅读、想法生成、多属性分子设计和抗体筛选四个任务上与 AI-Researcher、AI-Scientist-v2、InternAgent、DR-Claw、ScienceClaw、ARIS、EvoScientist 等基线进行对照评测。SCION 在 SEED/准确率上略优（SEED+1.6点、准确率+1.5点），想法新颖度比所有基线至少 60% 以上，分子设计成功率平均提升 61%（最高 39%），抗体筛选 F1 提升 33%（从 0.278 到 0.370）。

**⚠️ 局限性**

局限性：① 依赖大语言模型与多种工具接口，部署成本高；② 目前仅在实验室级别验证，缺乏大规模真实实验闭环；③ 对极度稀缺目标或长尾任务的泛化能力尚未充分验证；④ 系统复杂性导致易用性与维护成本提升；⑤ 未与自动实验平台完全集成，缺少硬件层面的闭环支持。

---

## 133. EgoInertia-MI: A Multimodal Egocentric Vision and IMU Benchmark for Motor Impairment Assessment

**arXiv ID:** 2607.03934 | [PDF](https://arxiv.org/pdf/2607.03934v1)

**作者:** Fatemah Alhamdoosh `[一作]` (University of Florence), DK Arvind `[通讯]` (University of Edinburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

创建了EgoInertia-MI数据集，结合同步的头盔摄像机视频与手腕/腿部IMU，模拟不同严重程度的运动障碍，并提供动作识别和严重程度估计两个基准任务。

**💡 创新点**

首次公开公开的多模态（视频+IMU）运动障碍评估基准，并证明第一人称视觉与惯性传感的融合能显著提升评估精度。

**🔧 技术方法**

使用CNN、LSTM、TCN、HARTransformer等IMU序列模型，X3D、SlowFast、V-JEPA等视频模型，以及跨模态注意力和晚期融合方法进行多模融合。

**📊 数据集**

EgoInertia-MI数据集，包含19个日常与临床相关动作、3个严重程度等级、17名健康志愿者共10小时同步记录。

**📈 对比分析**

通过5折交叉验证，单模视频精调可达0.93宏F1的动作识别，IMU可达0.69宏F1的严重程度估计；多模融合实现0.78宏F1严重程度与0.93宏F1动作识别，表现最优。

**⚠️ 局限性**

受试者仅为模拟运动障碍的健康志愿者，缺乏真实临床人群及长期随访，导致模型在真实病人中的可迁移性和临床效用仍待验证。

---

## 134. NavEYE: Vision-Centered Multi-Sensor Fusion-Based Situational Awareness System for Intelligent Surface Vehicles

**arXiv ID:** 2607.03915 | [PDF](https://arxiv.org/pdf/2607.03915v1)

**作者:** Ryan Wen Liua `[一作]`, Mengwei Baoa `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套基于视觉、AIS、雷达的多传感器融合系统NavEYE，提升海上情境感知与碰撞风险预警。

**💡 创新点**

提出MCGA多约束门限关联、DAWF自适应加权融合、TDSF时序衰减拼接融合以及VARF基于方位距离的多模态融合，解决多源误差、断连与多目标关联模糊问题。

**🔧 技术方法**

使用Mahalanobis门限+匈牙利算法进行关联；距离自适应权重、指数衰减拼接等融合；YOLO11n视觉检测；Kalman/Extended Kalman过滤器；多模态融合采用归一化方位角与距离特征。

**📊 数据集**

构建并公开MAPFusion数据集，包含1,783张图像、3,402条AIS轨迹、2,206条雷达轨迹以及同步视频、GNSS、磁力计等多模数据。

**📈 对比分析**

在MAPFusion上与JPDA、MHT、GRA、WAF、EKF等方法对比，MCGA/DAWF/VARF在关联精度、轨迹误差（DTW、MAE、RMSE）和多模关联准确率上均优于对手，性能提升显著。

**⚠️ 局限性**

局限在低能见度、雨雾、强背光等恶劣环境缺乏验证；视觉距离特征受相机俯仰、波浪遮挡和尺度变化影响；需进一步扩展场景与姿态补偿。

---

## 135. Advanced Topic Modeling Techniques for Categorizing Software Vulnerabilities

**arXiv ID:** 2607.03887 | [PDF](https://arxiv.org/pdf/2607.03887v1)

**作者:** Utkarsh Tiwari `[一作]` (Amrita Vishwa Vidyapeetham), Nidhin Prabhakar T. `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用多种先进主题建模方法对软件漏洞数据集的“Threat”文本进行分析，以提取潜在主题并进行聚类。

**💡 创新点**

创新点在于将多种配置的BERTopic、CombinedTM、Top2Vec、LLM驱动的Llama2+BERTopic与Mixtral等模型相结合，并通过大语言模型对主题进行自动标签与解释，提高主题可解释性。

**🔧 技术方法**

采用的技术包括BERT/SentenceTransformer嵌入、UMAP/PCA降维、HDBSCAN/DBSCAN聚类、BERTopic、Top2Vec、CombinedTM、Llama2‑7B和Mixtral‑8x7b等LLM。

**📊 数据集**

使用来自Cisco Labs的69,909条软件漏洞记录数据集，聚焦于其“Threat”文本字段。

**📈 对比分析**

通过主题连贯度、聚类质量、热图、层次聚类等指标对模型进行比较，结果显示混合LLM与BERTopic组合在主题识别与可解释性上优于传统方法，其他模型在不同维度上也展现出各自优势。

**⚠️ 局限性**

局限性包括模型对噪声文本的鲁棒性不高、需要大量计算资源、缺乏实时在线处理能力，以及在多语言或非结构化数据上的适用性待进一步验证。

---

## 136. Enhancement of E-commerce Sponsored Search Relevancy with LLM

**arXiv ID:** 2607.03886 | [PDF](https://arxiv.org/pdf/2607.03886v1)

**作者:** Md Omar Faruk Rokon `[一作]` (Walmart AdTech), Kuang-chih Lee `[通讯]` (Walmart AdTech)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并微调了LLAMA2 7B+LoRA模型，对查询与广告标题进行三类（相关、部分相关、无关）分类，以提升沃尔玛赞助搜索的相关性。

**💡 创新点**

创新点在于将LoRA技术应用于LLAMA2 7B，实现高效参数化微调；提出三类相关性判别方法；并结合自动标注提升训练效率。

**🔧 技术方法**

采用LLAMA2 7B大语言模型，使用LoRA进行参数高效微调；训练时用Adam优化器和交叉熵损失；实现三类分类与自动标注系统。

**📊 数据集**

使用Walmart自有的查询-商品标题对数据集，包含250K训练、56K验证、56K测试样本，全部通过三人人工标注完成。

**📈 对比分析**

通过离线指标（准确率、精确率、召回率、F1）与NDCG@4/8与BERT Bi‑Encoder、Cross‑Encoder、GPT‑4做对比；微调后的LLAMA2 7B取得89.43%准确率、89.41%F1、NDCG@4 0.7142，显著优于传统模型和GPT‑4。

**⚠️ 局限性**

局限性包括对“无关”类别召回不足，需要人工干预；模型仍受训练数据偏差影响；未来需扩展多模态输入与更完善的自动标注策略。

---

## 137. Post-Lecture Interactive Environments for Conceptual Learning: A Randomized Comparison of Mixed Reality and Tangible Instruction in Undergraduate STEM Education

**arXiv ID:** 2607.03896 | [PDF](https://arxiv.org/pdf/2607.03896v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 138. Next-Gen Sponsored Search: Crafting the Perfect Query with Inventory-Aware RAG (InvAwr-RAG) Based GenAI

**arXiv ID:** 2607.03880 | [PDF](https://arxiv.org/pdf/2607.03880v1)

**作者:** Md Omar Faruk Rokon `[一作]` (Walmart AdTech), Musen Wen `[通讯]` (Walmart AdTech)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于库存感知的检索增强生成（RAG）模型InvAwr-RAG，用于动态重写用户搜索查询，从而显著提升赞助搜索的填充率和相关性。

**💡 创新点**

创新点包括：实时将库存与竞价数据融入查询重写；融合生成式查询与历史成功查询形成混合查询池；使用两塔BERT与LoRA微调的LLM进行检索增强生成。

**🔧 技术方法**

采用技术包括：两塔BERT嵌入、Llama2 7B LLM + LoRA微调、检索增强生成框架、向量数据库检索、规则型查询分类器以及实时库存索引。

**📊 数据集**

使用的数据集包括：Walmart搜索日志、商品标题与点击/展示记录、手工标注的相关性分数、实时库存数据以及历史高点击率的重写查询集合。

**📈 对比分析**

在10,000个历史填充率为0%的查询上进行离线评估，结果显示InvAwr-RAG填充率提升至68%，NDCG@8从0提升至0.6847，明显优于基线（0%）和GPT‑4（53%/0.6458）。

**⚠️ 局限性**

局限性：依赖Walmart特定的库存与竞价体系，需要进一步A/B测试验证真实收益；对极端稀有查询的效果可能有限；模型推理延迟与实时库存同步的工程挑战；未深入评估用户体验和广告主满意度。

---

## 139. MACRO: Training-free Multi-plane Attention for Closeup Render Optimization

**arXiv ID:** 2607.03875 | [PDF](https://arxiv.org/pdf/2607.03875v1)

**作者:** Nitzan Hodos `[一作]` (Amazon Prime Video), Netalee Efrat `[通讯]` (Amazon Prime Video)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MACRO 方法，利用 3DGS 渲染的深度信息，将近景渲染拆分为多平面，并在图像空间对参考图像做尺度匹配裁剪，随后在深度感知的注意力掩模下进行 diffusion-enhancement，从而显著提升近景视图的细节与纹理质量。

**💡 创新点**

创新点在于：①在图像空间完成尺度对齐，避免 VAE 非尺度等变性导致的 latent‑space 失配；②引入多平面深度分解与深度掩模注意力，使每个深度层仅关注与其尺度匹配的参考信息；③方法完全无须修改原有网络架构或额外训练，兼容任意现有的 diffusion‑based enhancement。

**🔧 技术方法**

使用技术包括 3D Gaussian Splatting 渲染、VAE+UNet 的 diffusion-enhancement、深度平面聚类、尺度匹配裁剪与上采样（PFT‑SR）、基于深度的注意力掩模。

**📊 数据集**

主要数据集：DL3DV‑Closeup（40 场景、283 视角对）和 MobileClose‑10（10 场景、39 视角对），两者用于构建近景合成的标准评测基准。

**📈 对比分析**

与 3DGS、Mip‑Splatting、SEVA、GSFixer、Difix 及其 progressive 版本对比。MACRO 在 PSNR/SSIM 维持与 3DGS 相近或略优，同时在 LPIPS、DreamSim、DINOv2 等感知指标上提升 5–29%（DL3DV‑Closeup）和 14–69%（MobileClose‑10），明显优于所有 baseline。

**⚠️ 局限性**

局限性包括：裁剪与上采样后高频细节可能丢失导致略微模糊；深度平面数固定，未针对每张图自适应；依赖 3DGS 渲染的深度准确性，深度误差可能影响裁剪与注意力效果。

---

## 140. Beam Hopping Low Earth Orbit Satellite Resource Allocation for Differentiated Services and Robustness Analysis under Model Attacks

**arXiv ID:** 2607.03859 | [PDF](https://arxiv.org/pdf/2607.03859v1)

**作者:** Shuang Zheng `[一作]` (Beijing University of Posts and Telecommunications), Wenbo Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过数字孪生技术和深度强化学习，提出了用于低轨卫星Beam Hopping（BH）系统的联合光束调度与功率分配框架，并设计了名为BRIDGE的算法实现。

**💡 创新点**

创新点包括：1）利用数字孪生精确获取用户-卫星可视性时空动态；2）在PPO框架中引入Dirichlet分布与Gumbel-TopK采样，巧妙处理混合离散-连续动作空间；3）嵌入QoS驱动的子信道分配机制；4）系统性地评估三种经典对抗攻击（FGSM、I‑FGSM、PGD）下的鲁棒性。

**🔧 技术方法**

主要技术包括：数字孪生（Sionna + Blender + OpenStreetMap）、基于PPO的深度强化学习、Dirichlet连续动作采样、Gumbel-TopK离散动作采样、多目标奖励设计、GAE与熵正则化等。

**📊 数据集**

实验数据来自仿真平台：使用Sionna射线追踪生成的可视窗口、随机生成的用户轨道位置、泊松流量模型与预设的Ka‑波段信道参数。未使用公开真实卫星数据集。

**📈 对比分析**

在能耗效率、RT服务吞吐量和公平性等指标上，与QLPDL‑BH、P‑BH、GA‑BH、Top‑K DQN、SAC‑BH及离散PPO等基线相比，BRIDGE平均提升能效 16–99%、RT吞吐量提升约 20–30% 及公平性显著改进；对抗攻击下性能几乎与正常运行相当，显示出良好的鲁棒性。

**⚠️ 局限性**

局限性包括：1）仅评估了对链路增益的有限扰动，对更复杂的对抗攻击与传感误差未做充分验证；2）实验场景限定于单颗卫星与 8 光束的设置，缺乏大规模星座的验证；3）对数字孪生模型的精度与实时更新机制仍需进一步研究；4）鲁棒性分析仅基于固定扰动预算，未考虑持续性或适应性攻击。

---

## 141. Speaker-Disentangled Chunk-Wise Regression for Syllabic Tokenization

**arXiv ID:** 2607.04064 | [PDF](https://arxiv.org/pdf/2607.04064v1)

**作者:** Ryota Komatsu `[一作]` (Institute of Science Tokyo), Takahiro Shinozaki `[通讯]` (Institute of Science Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种无监督的说话人去耦合的块级回归方法（SylReg）用于语音的音节级离散化，并构建相应的交叉文本-音节语言模型；

**💡 创新点**

通过在固定长度块上对说话人扰动的学生表示与教师目标进行回归，既消除了说话人信息又促进了音节结构的形成，解决了传统方法中说话人主导和原型坍塌问题；

**🔧 技术方法**

采用HuBERT预训练编码器、BYOL式学生-教师框架、平均池化块级特征、MSE回归损失、说话人扰动（性别转换）、自分割蒸馏（SylBoost）以及基于分块的聚类和最小割分割；

**📊 数据集**

主要使用Libri-Light（55k小时）训练SylReg，LibriSpeech（train-clean-100）用于分割蒸馏，LibriSpeech、LibriHeavy、Emilia-Large、People's Speech、VoxPopuli、TinyStories、Cosmopedia、Hi-Fi-CAPTAIN等多语料库用于语言模型训练；

**📈 对比分析**

与HuBERT、SD‑HuBERT、Sylber、SylBoost等现有音节化方法比较，SylReg在LibriSpeech上实现了最高的音节分割F1（72.5%）和标记纯度（SP 70.3%），并在语音语言模型上取得相对7%语法与语义理解提升，且在语音合成任务中以约2.3倍更低的比特率匹配TWIST的CER/WER；

**⚠️ 局限性**

局限性包括：对说话人扰动的设计需要经验，C=1时语音合成效果下降；模型对细粒度音素区分能力有限，导致sWUGGY分数不及基于音素的模型；主要在英语语料上验证，跨语言通用性待进一步评估；

---

## 142. Telescope: Improving Zero Shot Detection of LLM Generated Content By Measuring Token Repetition Probability

**arXiv ID:** 2607.04061 | [PDF](https://arxiv.org/pdf/2607.04061v1)

**作者:** Christopher Nassif `[一作]` (Virginia Tech), Josh F. Cooper `[通讯]` (Virginia Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的零射击检测方法Telescope Perplexity，用来区分LLM生成文本与人类文本。

**💡 创新点**

创新点在于：①发现并利用LLM训练早期形成的“遗留启发式”（对重复token的强烈回避）作为检测信号；②设计了针对这一局部重复概率的Telescope Perplexity指标；③证明该信号在不同模型、不同尺寸下普遍存在并能在现代LLM上保持有效。

**🔧 技术方法**

技术手段包括：零射击检测框架；使用参考模型（Gemma、Llama、Falcon、SmolLM等）计算Telescope Perplexity；对比传统Perplexity、Rank‑based LRR、Binoculars、Fast‑DetectGPT等基线方法；在各种文本长度、扰动、ESL等场景下评估鲁棒性。

**📊 数据集**

数据集涵盖：HC3、HC3 Plus、Ghostbusters系列（Essay、News、Creative）、AI vs Human、Detect LLM Text、ESL GPT4o Mini、以及自建的GPT4o Mini、Deepseek‑V3生成的新评测集。

**📈 对比分析**

与基线比较时，Telescope Perplexity在平均AUROC上往往优于当前SOTA Binoculars，且在GPT4o Mini、Deepseek‑V3等现代LLM的检测任务中表现尤为突出；在许多数据集上接近或超过0.99的AUROC，并且在短文本、扰动以及ESL文本上保持高鲁棒性。

**⚠️ 局限性**

局限性包括：①未对高成本的扰动式检测方法（DetectGPT、DetectLLM‑NPR）进行基准；②评测集虽更新但仍无法完全覆盖真实部署中多种对抗手段与多样写作风格；③在高度公式化或诗歌类文本中可能误判；④需要针对不同域手动校准阈值，且阈值在迁移时可能下降。

---

## 143. An Exploratory Study of Malicious Link Posting on Social Media Applications

**arXiv ID:** 2607.04042 | [PDF](https://arxiv.org/pdf/2607.04042v1)

**作者:** Muhammad Hassan `[一作]` (University of Illinois at Urbana Champaign), Masooda Bashir `[通讯]` (University of Illinois at Urbana Champaign)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对安卓平台上五大社交媒体应用（Facebook、Twitter、Instagram、TikTok、Mastodon）的恶意链接发布机制进行了实验评估。

**💡 创新点**

首次系统地比较并揭示了这些主流平台在处理恶意URL时的安全缺口，展示了其可利用的漏洞。

**🔧 技术方法**

使用自动化脚本发布URL、短链，结合PhishTank、Google Safe Browsing 与 VirusTotal 进行恶意性验证。

**📊 数据集**

利用从PhishTank随机抽取的 7 组 5 个URL（共35个）并通过安全服务再次确认的恶意链接集合。

**📈 对比分析**

通过统计每个平台允许/阻止的URL数量以及阻断率，结果显示Twitter 约 69% 的恶意链接被拦截，整体阻断率仅 23.8%，其余平台几乎无拦截。

**⚠️ 局限性**

研究仅关注发布时的检测，未追踪后续删除或隐藏；仅使用公开数据库的URL，未考虑不同文本场景；实验限定于安卓端，未涵盖 iOS 等其他平台。

---

## 144. CrossHallu: Do Hallucination Signals Generalize Across Languages and Domains in Large Language Model's Internals?

**arXiv ID:** 2607.04029 | [PDF](https://arxiv.org/pdf/2607.04029v1)

**作者:** Aisha Alansari `[一作]` (King Fahd University of Petroleum and Minerals), Hamzah Luqman `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估跨语言（英阿）、跨域（阿拉伯不同数据集）以及跨语言-跨域情景下，使用LLM内部表示进行幻觉检测的可迁移性；

**💡 创新点**

首次在六个不同规模与架构的LLM上量化内部特征在多语言、多域任务中的迁移性，并揭示跨语言、跨域与双重迁移的性能差异；

**🔧 技术方法**

基于隐藏状态、注意力张量及词概率的多组特征提取，训练三层MLP检测器，并使用GPT‑4o‑as‑a‑Judge生成幻觉标签；

**📊 数据集**

TruthfulQA（英文及其人工翻译的阿拉伯语版本）和HalluScore（专门的阿拉伯问答幻觉数据集）；

**📈 对比分析**

与单语基线相比，跨语言迁移普遍下降；在阿拉伯语内部跨域时性能提升；在最难的跨语言-跨域设置中，Aya和Phi‑4‑mini仍保持较高的AUC‑ROC，整体表现表明不同模型在语言与域间的可迁移性存在显著差异；

**⚠️ 局限性**

标签依赖GPT‑4o，可能引入偏差；仅评估六个模型且仅覆盖英语和阿拉伯语；只关注生成问答任务，未扩展至摘要、翻译等其它生成任务。

---

## 145. Energy-Aware System-Level Evaluation of Post-Quantum TLS on Embedded User Equipment over a Disaggregated 5G Network

**arXiv ID:** 2607.03988 | [PDF](https://arxiv.org/pdf/2607.03988v1)

**作者:** Sanzida Hoque `[一作]` (Florida Institute of Technology), Abdullah Aydeger `[通讯]` (Florida Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在一套基于 Raspberry Pi 5 的真实 5G UE 测试平台上，对不同 NIST 标准化的后量子密码（PQC）TLS 握手进行系统级的能耗、延迟、CPU、网络与热特性评估；

**💡 创新点**

首次将能耗测量与并发负载相结合，系统地揭示了签名算法（尤其是哈希基签名）对延迟与能耗的主导影响，并展示了晶格基与哈希基 PQC 方案在 5G 端到端环境中的可扩展性差异；

**🔧 技术方法**

采用 BoringSSL + liboqs 集成 PQC，配合 UERANSIM 与 Open5GS 的虚拟化 5G 核心与接入网络，利用板载 PMIC 进行电源监测，并通过多 UE 并发模型和精细的时间戳捕获；

**📊 数据集**

实验不使用传统数据集，而是在 1、4、10、20、40 个并发客户端 UE 之间重复 50 次握手，记录完成的握手数、延迟、能耗、CPU 负载等；

**📈 对比分析**

通过对比不同签名+KEM 组合（如 P-256+SLH‑DSA、ML‑KEM+Falcon 等）的平均握手延迟、每次握手能耗、CPU 使用率与吞吐率，发现哈希基签名延迟可达 4 倍、能耗 2 倍；晶格基方案则在 40 并发下仍保持 2–3 倍的低延迟与能耗；

**⚠️ 局限性**

局限性包括：仅在 Raspberry Pi 5 及其 8 GB RAM 环境下实验，未涉及真实 RF 与移动终端热设计；PMIC 电源测量误差约 ±5–10%，采样间隔 100 ms 可能忽略瞬态峰值；并发模型未覆盖更大规模或多核服务器，且实验环境为受控测试床，未涵盖真实网络拥塞与波动。

---

## 146. Fast Asymptotically Optimal Kinodynamic Planning via Vectorization

**arXiv ID:** 2607.03987 | [PDF](https://arxiv.org/pdf/2607.03987v1)

**作者:** Yitian Gao `[一作]` (Purdue University), Zachary Kingston `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于JAX的GPU并行化、可扩展的Kinodynamic RRT，并通过AO‑x迭代实现渐近最优性。

**💡 创新点**

创新点在于将AO‑x元算法与大规模批量并行RRT结合，用JAX+XLA一次性生成GPU内核，消除CPU‑GPU通信，显著提升实时性能，同时仍保持渐近最优性。

**🔧 技术方法**

采用JAX框架、XLA编译、GPU批处理、MuJoCo‑XLA仿真、固定步长RK4传播等技术。

**📊 数据集**

实验使用MuJoCo‑XLA、DynoBench（unicycle、acrobot、quadcopter）、Kino‑PAX环境（double integrator、Dubins airplane、quadcopter）、soft robot vine、block push等数据集。

**📈 对比分析**

与Kino‑PAX、iDb‑A*、SST*等传统或并行规划器比较，PAKR在找到初始解的时间从毫秒级到数十毫秒，树规模更小，最终解质量优于或相当；在高维仿真上也实现了与传统CPU规划器相当的性能。

**⚠️ 局限性**

局限在于对动态系统的离散化步长要求、分支因子需手动调优、对极高维度或复杂约束的可扩展性仍待验证，且目前仅支持JAX/NumPy生态，未与深度学习决策融合。

---

## 147. Scalable Semantic Steering of Embedding Projections

**arXiv ID:** 2607.03978 | [PDF](https://arxiv.org/pdf/2607.03978v1)

**作者:** Wei Liu `[一作]` (Virginia Tech), Chris North `[通讯]` (Virginia Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种可扩展的语义投影调控方法，通过在用户定义的组上一次性调用LLM生成语义剖面，并与种子中心结合生成混合原型，以在嵌入空间中对整个集合进行语义驱动的投影重塑。

**💡 创新点**

创新点在于将LLM推理从逐项处理转为基于组的抽象，只需一次LLM调用即可完成全局语义传播，大幅降低计算成本和延迟，同时保留语义一致性。

**🔧 技术方法**

所用技术包括单次LLM调用生成组级剖面、原型混合（seed centroid + 语义剖面嵌入）、自适应阈值软分配、对齐缩放更新以及UMAP/CLIP等降维与多模态编码器。

**📊 数据集**

实验使用LitCovid 5,000篇医学文献和Stanford-40 Actions图像子集（2,583张）进行评估，验证方法在文本与图像两种模态下均可适用。

**📈 对比分析**

与逐项LLM推理的基线相比，组级原型方法在全局对齐（Silhouette提升）上相近，但成本降低约1,300倍，且在投影重构中保持较高的组内一致性与合理的误差率。

**⚠️ 局限性**

主要局限在于对组级剖面质量高度依赖；若组定义不精确或剖面不充分，原型可能偏离目标；此外，相较于逐项细粒度调节，方法在细节级别的个体差异捕捉能力有限。

---

## 148. Refused in Chat, Written in Code: Workflow-Level Jailbreak Construction in IDE Coding Agents

**arXiv ID:** 2607.03968 | [PDF](https://arxiv.org/pdf/2607.03968v1)

**作者:** Abhishek Kumar `[一作]` (Alan Turing Institute), Carsten Maple `[通讯]` (Alan Turing Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在 IDE 集成的编码代理中，工作流层级的越狱攻击，即在多轮软件开发工作流中逐步构建有害输出；

**💡 创新点**

首次揭示工作流结构而非单一提示是导致越狱的根本原因，证明直接拒绝测试不足以评估编码代理的安全性；

**🔧 技术方法**

利用 GitHub Copilot Chat 与四个封闭权重后端（Claude Sonnet 4.6、Claude Haiku 4.5、Gemini 3.1 Pro、Gemini 3.5 Flash）进行实验，并通过脚本化的多轮交互构建评估流水线、添加教学样本等技术；

**📊 数据集**

使用来自 Hammurabi's Code、HarmBench、AdvBench 的 204 条有害提示，覆盖恶意软件、版权滥用、恶意编程等多类别；

**📈 对比分析**

将全流程攻击与三种基线（直接聊天、CSV 读取、单步代码修复）对比，基线下 8/816 的拒绝率被提升至 816/816 的越狱成功率；实验显示在约 6 次交互后即可出现有害教学样本；

**⚠️ 局限性**

实验仅覆盖 GitHub Copilot 与四个后端，未评估其他 IDE 或开源模型，样本量受人工评估限制，且缺乏自动化评判器，结果可能随后端更新或环境变化而变化。

---

## 149. Additional properties of parity based bit-counting complexity classes and hierarchies

**arXiv ID:** 2607.04048 | [PDF](https://arxiv.org/pdf/2607.04048v1)

**作者:** Tayfun Pay `[一作]` `[通讯]`, Tayfun Pay

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了基于奇偶计数的位计数复杂度类 B_|0|⊕P 与 B_|1|⊕P 的性质，证明其闭包性、包含关系、与 US、⊕P 的关系，并构建了一系列层次结构；

**💡 创新点**

创新点在于首次证明 B_|1|⊕P ⊆ B_|0|⊕P、US 与这两类的多相机包含、以及将这些类嵌入到新的多层级结构中，展示其位于 PH 与 CH 之间；

**🔧 技术方法**

使用了位计数函数、Prouhet–Thue–Morse 序列、真值表归约、oracle 计算与多层塔结构等理论工具；

**📊 数据集**

无实验数据集，全部为理论证明；

**📈 对比分析**

通过复杂度层级归约与层级包含证明，得出 B_|0|⊕P 与 B_|1|⊕P 能包含 NP、CoNP、US 与 ⊕P，并被计数层次 CH 包含；

**⚠️ 局限性**

仍未证明 B_|1|⊕P 与 B_|0|⊕P 的等价性，以及 C_=P 等其他计数类的关系，限制在当前证明框架下。

---

## 150. Finite Reliability Representations: Noise-Calibrated Belief-Space Covers for Reliable Decision-Making

**arXiv ID:** 2607.04019 | [PDF](https://arxiv.org/pdf/2607.04019v1)

**作者:** Hyung-Jin Yoon `[一作]` (Tennessee Technological University), Hunmin Kim `[通讯]` (Mercer University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文提出了一种“有限可靠性表征（FRR）”框架，用物理传感与执行噪声上限来决定贝叶斯决策系统中可安全使用的信念分辨率，并通过覆盖可达信念空间的可靠性单元来保证近似策略的最优性。

**💡 创新点**

创新点在于：①把贝叶斯滤波的固定观测映射与可控信念转移核区分，强调后者才是动态规划的光滑度对象；②通过可观测区分度、执行不确定性和信念核 Lipschitz 模式来量化决策误差；③给出决策直径（Q* 变化）作为细化标准，并证明细化单元内的策略逼近误差 ≤ 2ε/(1‑γ)；④提出可靠性熵作为基于物理噪声的决策相关信息容量度量。

**🔧 技术方法**

技术手段主要包括：贝叶斯滤波与可观测分布的总变距离、Wasserstein 1 距离对信念核的 Lipschitz 性、动态规划的 Bellman 迭代、α‑向量求解、线性高斯滤波的闭式分析、粒子滤波经验估计以及编码‑评判器组合来实现学习表征的 FRR 认证。

**📊 数据集**

实验使用：1）两状态两动作的离散 POMDP，验证理论与误差上界；2）UGV（二维平面运动）与双连杆机械臂的连续状态示例，采用粒子滤波得到可达信念集并使用代理值函数进行可靠性单元认证；3）在离散 POMDP 中使用完整的 α‑向量求解获得 Q*。

**📈 对比分析**

与传统方法（如点基值迭代、近似贝尔曼、信息瓶颈或学习的压缩表征）比较：FRR 能直接给出决策错误的理论上限，且不需要显式的模型收敛假设；在实验中，动作执行不确定性可显著降低所需可靠性单元数量，而单纯的传感降噪可能导致单元数增加；整体性能符合 2ε/(1‑γ) 的理论界限。

**⚠️ 局限性**

局限性：①需要可达信念集前向不变性和信念核的 Lipschitz 上界，实际系统中很难精确估计；②FRR 只约束由表示引起的误差，无法消除传感/执行噪声本身造成的性能下限；③对连续状态空间的经验估计（如粒子滤波）依赖采样质量，可能导致保守或不准确的单元认证；④未提供自适应或在线更新可靠性单元的机制。

---

## 151. Enhancing Implicit Neural Representations with Image Feature Embedding for Unsupervised Cardiac Cine MRI Reconstruction

**arXiv ID:** 2607.04069 | [PDF](https://arxiv.org/pdf/2607.04069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 152. Separating Representation from Reconstruction Enables Scalable Text Encoders

**arXiv ID:** 2607.04011 | [PDF](https://arxiv.org/pdf/2607.04011v1)

**作者:** Megi Dervishi `[一作]` (FAIR Meta), Yann LeCun `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种双分结构（CrossBERT）将表示学习与token重构分离的文本编码器，并通过高掩码率和互补掩码策略实现高效训练。

**💡 创新点**

创新点在于：①在BERT的“平面”架构上引入轻量化交叉注意力解码器，隔离重构任务；②支持50%以上掩码率，提升吞吐量；③使用互补掩码同时学习所有token，实现两倍样本效率；④通过冻结评估揭示BERT随规模增长的表示退化。

**🔧 技术方法**

采用了Masked Language Modeling (MLM) 作为预训练目标，改为双分架构的CrossBERT；使用RoPE位置编码、RMSNorm、交叉注意力预测器；评估采用线性探针、kNN探针和MTEB对比学习；对比实验使用GLUE、MTEB和MS-MARCO。

**📊 数据集**

训练数据主要为4T词的DCLM语料；对比学习微调使用MS-MARCO查询-文档对；评估数据覆盖GLUE 8个任务和MTEB v1/v2七大任务类别。

**📈 对比分析**

与传统BERT、Electra、现代BERT等模型对比，CrossBERT在GLUE线性探针上提升约6点，在MTEB(eng,v1)上平均分数高于NeoBERT/ModernBERT；在冻结情况下表现与全微调相近甚至超越，吞吐量提升1.5-2倍，样本效率提高2倍。

**⚠️ 局限性**

局限性包括：实验仅覆盖数十亿参数规模，未验证更大规模下的趋势；缺乏对RTD导致句子嵌入退化的理论解释；评估侧重冻结表示，可能忽略某些下游任务的适配细节。

---

## 153. What is Left for Us? Second Scholarship Against the Degradation of Research by AI

**arXiv ID:** 2607.04049 | [PDF](https://arxiv.org/pdf/2607.04049v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 154. Explainable AI for Screening Abuse-Related Trauma in Bangladeshi Children: A Training-Free Multimodal Framework Evaluated on Noise-Aware Synthetic Data

**arXiv ID:** 2607.04010 | [PDF](https://arxiv.org/pdf/2607.04010v1)

**作者:** Salma Hoque Talukdar Koli `[一作]` (RTM Al-Kabir Technical University), Fahima Haque Talukder Jely `[通讯]` (North East University Bangladesh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种多模态、训练无关的解释型人工智能框架（ShishuRaksha AI），用于孟加拉儿童虐待相关创伤的早期筛查，并生成双语（孟加拉语/英语）可解释报告与国家儿童保护机构的转诊指引。

**💡 创新点**

创新点在于：①结合四种儿童友好的筛查模态（SDQ/CPSS问卷、孟加拉语叙事文本、房树人绘画特征、面部表情）并通过临床指定权重实现训练无关的跨模态注意力融合；②使用可解释的加性归因（扰动法近似 SHAP）生成可供临床审核的风险解释；③构建噪声感知的合成基准，用于在伦理受限下验证设计；④实现双语报告与国家转诊路径对接。

**🔧 技术方法**

技术包括：随机投影与无学习的交叉模态注意力融合、基于梯度提升树的后续判别器、扰动法加性归因、BengalBERT文本编码、EfficientNet-B0绘画特征提取、临床权重与单模态覆盖规则。

**📊 数据集**

使用自生成的噪声感知合成数据集，共500例（116正例，23.2%），包含四层噪声（测量误差、报告者误分类、缺失子量表、标注不一致等）以及从文献得到的绘画先验。

**📈 对比分析**

与单模态基线（SDQ仅、文本仅、绘画仅）及无噪声合成基线进行比较。5 折分层交叉验证下，融合模型 AUC 为 0.874（95% CI 0.834–0.908），显著高于 SDQ 仅的 0.756（95% CI 0.705–0.803）。对单模态的消融实验表明文本模态贡献最大，但其优势部分来源于构造的标签相关特征。

**⚠️ 局限性**

局限性包括：完全基于合成数据，无法验证在真实儿童中的表现；未包含面部模态；使用树模型作为后续判别器，未直接评估无学习融合本身；缺少外部验证集；文本特征存在构造性循环；高风险区校准偏差；城市与农村子组表现差异；以及对临床可操作性和伦理使用的进一步验证需求。

---

## 155. DS-SAC: Density Search for Sample Consensus

**arXiv ID:** 2607.03972 | [PDF](https://arxiv.org/pdf/2607.03972v1)

**作者:** Suraj Thapa `[一作]` (University of New Haven), Muhammad Aminul Islam `[通讯]` (University of New Haven)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于密度搜索的确定性一致性框架 DS-SAC，用来进行几何模型估计。

**💡 创新点**

创新点在于通过前向/后向搜索逐步聚焦残差空间中的密集区，并递归地按符号残差划分点集，避免了随机最小采样，兼顾了搜索效率与鲁棒性。

**🔧 技术方法**

使用的技术包括：全点初始化、基于百分位的前向与后向搜索、残差符号划分的递归分区、MSAC 计分与后调优，以及对齐解算器（DLT、八点法、五点法）等。

**📊 数据集**

实验使用的公开数据集包括 ScanNet1500、PhotoTourism、LaMAR、7Scenes、ETH3D 与 KITTI 共计 39,592 张图像对。

**📈 对比分析**

与 OpenCV RANSAC、MAGSAC、LO-RANSAC、GC-RANSAC 等方法在同一预算下比较，DS‑SAC 在 homography、fundamental/essential matrix 的 AUC、median 误差和运行时间上均表现最好或相近，并且通常更快。

**⚠️ 局限性**

局限性包括：仍需手动调节参数（如 p_min、Δp）；在极高噪声或非常稀疏的匹配场景下，递归分区的效率和最终精度可能下降。

---

## 156. PreSIST: Vision-Language-Informed Object Persistence Prediction in Open-World Scenes

**arXiv ID:** 2607.04057 | [PDF](https://arxiv.org/pdf/2607.04057v1)

**作者:** Amanda Adkins `[一作]` (University of Texas at Austin), Joydeep Biswas `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PreSIST方法，用于零样本开放世界环境下的对象持久性预测，利用场景与实例信息构建概率生存先验，并通过持久性滤波器实现任意未来时刻的预测；

**💡 创新点**

创新点在于将大型视觉‑语言模型的常识推理能力与概率生存模型结合，提供基于场景语境的实例级持久性先验，并设计了无监督的可视化版本PreSIST‑Vis，既保持精度又显著提升推理速度；

**🔧 技术方法**

采用了大型视觉‑语言模型（Gemma 4 31B、Gemini Robotics‑ER等）进行文本描述与推理，使用DINOv2 ViT-L/14 backbone与轻量级适配器的交叉注意力网络进行视觉推断，融合了Weibull生存模型与递归贝叶斯滤波器；

**📊 数据集**

构建了包含约6500个实例分割与存在/消失标注的新数据集，来源于UT Austin校园、Oxford RobotCar、HD‑EPIC等；同时利用COCO、OpenLORIS、KITTI、EPIC‑Kitchens、UT CODa、SKU110k、SUN397、PKLot等公开数据集用于伪标签生成与模型训练；

**📈 对比分析**

与多种基线（单VLM查询、每个预测时刻VLM查询、CLIP+Class Lookup、指数/对数正态分布、In‑Context Vision等）以及在PKLot上进行的基于上下文训练的对比实验；结果表明PreSIST‑Lang在多数域上实现了最低MAE和最高平衡准确率，PreSIST‑Vis仅以约0.04 s的推理时间接近Lang的表现，并在长期视觉定位任务中显著提升了内点率与定位精度；

**⚠️ 局限性**

局限性包括：1）对大规模VLM推理的依赖导致Lang版本的计算成本较高；2）当前仅使用Weibull一阶生存模型，难以捕捉多阶段或多模态持久性变化；3）伪标签生成依赖于VLM的准确性，若场景极其异域可能导致误差积累；4）仅评估了物体位置持久性，未考虑姿态或属性变化。

---

## 157. Claim2Source at CheckThat! 2026: Improving Multilingual Scientific Claim-Source Retrieval with Verification-based Re-Ranking

**arXiv ID:** 2607.04043 | [PDF](https://arxiv.org/pdf/2607.04043v1)

**作者:** Tobias Schreieder `[一作]` (TU Dresden), Michael Färber `[通讯]` (TU Dresden)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对多语言科学声明–来源检索任务，提出了由三阶段检索（候选生成、相似度重排序、验证重排序）组成的多阶段检索框架，并引入双语声明表示和基于元数据的来源表示。

**💡 创新点**

创新点在于：①将声明原文与机器翻译后的英文版本同时作为表示，降低语言不匹配；②在来源表示中加入作者、会议等元数据信息；③在第三阶段直接利用大型语言模型的验证信号进行列表式重排序，进一步提升检索精度。

**🔧 技术方法**

主要技术包括：GritLM-7B（以及微调版GritLM‑F）用于第一阶段检索；Qwen3‑8B‑IT用于相似度重排序；Kimi‑K2.6（大规模稀疏混合专家模型）用于验证重排序；并使用多语言BERT及E5、GTR等检索模型作为对照。

**📊 数据集**

使用 CheckThat! 2026 数据集（包含英德法三种语言的社交媒体声明与对应英文科学来源，候选库1万篇）。

**📈 对比分析**

在官方测试集上，系统平均 MRR@5 为 0.7628，排名第一；在单语子集上，英法语言得到第一名，德语获得第二名；与基线相比，整体提升约 +0.023 MRR@5。

**⚠️ 局限性**

主要局限：①模型规模巨大，尤其是验证阶段的 Kimi‑K2.6，导致计算成本高；②对不同语言的模型选择需做语言特定微调，缺乏统一的多语言通用方案；③德语表现仍相对弱势，后续需要更细致的错误分析和更鲁棒的跨语言适配。

---

## 158. Data Structures for Private Token Transfers in TEE-Based Networks

**arXiv ID:** 2607.04032 | [PDF](https://arxiv.org/pdf/2607.04032v1)

**作者:** Blake Regalia `[一作]` (Solar Republic LLC), Benjamin Adams `[通讯]` (University of Canterbury)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计了 DWB 与 BTBE 两种数据结构，用以在 TEE 网络中隐藏代币转账的存储访问模式，并实现私密推送通知。

**💡 创新点**

创新点在于利用延迟写缓冲和桶化 Trie 结合领域特定优化，实现低成本的匿名转账，并提供实时私密通知。

**🔧 技术方法**

使用了 TEE 隐私合约、VRF 随机、ChaCha20-Poly1305 加密、Bloom 过滤器、哈希桶化与链式事件列表。

**📊 数据集**

使用 Secret Network 主网代币（如 USDC、USDT、BTC、ETH、sSCRT 等）以及 42 个升级代币的交易数据。

**📈 对比分析**

与原 SNIP‑20 合约对比，在最高负载下 gas 消耗提升约 26%，但通过 DWB 与 BTBE 在典型场景下可实现约 99% 的匿名概率并减少 300 次交易后的关联风险。

**⚠️ 局限性**

限制包括：查询攻击仍然可泄露键、低活跃期匿名性下降、需要更大缓冲或 ORAM 级联才能提升安全性。

---

## 159. Efficient Discovery of Conditional Dependencies with Desbordante

**arXiv ID:** 2607.04030 | [PDF](https://arxiv.org/pdf/2607.04030v1)

**作者:** Ivan Kozhukov `[一作]` (Saint-Petersburg State University), George Chernishev `[通讯]` (Saint-Petersburg State University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在Desbordante中实现了高性能的ParCFDFinder算法，用于从大规模数据集快速发现条件函数依赖（CFD）。

**💡 创新点**

创新点包括在C++中重构CFDFinder、引入多线程并行化、比特掩码剪枝、批量候选处理以及去重优化，显著提升速度与内存利用率。

**🔧 技术方法**

主要技术包括C++实现、Boost多索引容器、线程池（Boost.Asio）、位图覆盖、批量候选处理、并行模式遍历以及多线程工作窃取。

**📊 数据集**

使用了多种真实与合成数据集，包括Bridges、Echocardiogram、Wisconsin Breast Cancer、BMW Global Sales、ncvoter、abalone、Students、Biocase multimedia、Wine Reviews、Biocase gathering namedareas等。

**📈 对比分析**

与Metanome Java实现对比，ParCFDFinder在多线程下速度提升可达3–4×（单线程约1–3×），单线程对比Java可达4.9–318×，内存使用减少2–23×；对同一数据集发现的CFD数量相同或更多，证明算法正确性与效率。

**⚠️ 局限性**

主要局限包括：对属性数的指数复杂度仍然存在，极大规模（百万行以上）在单线程下仍需较长时间；多线程虽然加速，但在高维数据时并行效率下降；实现仅覆盖常量与负常量扩展，未覆盖范围条件等更一般的CFD形式；缺乏自动参数调优与交互式可视化支持。

---

## 160. InSpace: Structure-Aware 3D Indoor Scene Generation from a Single 360° Image

**arXiv ID:** 2607.03990 | [PDF](https://arxiv.org/pdf/2607.03990v1)

**作者:** Gwanhyeong Koo `[一作]` (KAIST), Chang D. Yoo `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种从单张360°全景图像生成完整3D室内场景（包括布局与资产）的框架InSpace。

**💡 创新点**

创新点在于引入视角选择的跨注意力（view‑selective）与资产选择的跨注意力（asset‑selective）实现空间定位的生成，并通过布局引导的结构反演（Layout‑Guided Structure Inversion）将部分几何先验直接注入生成过程。

**🔧 技术方法**

使用了稀疏体素VAE（O‑Voxel）作为稀疏表示，流匹配（flow matching）训练的DiT模型进行三阶段生成，结合视角/资产遮罩的自注意力与跨注意力。

**📊 数据集**

构建了基于3D‑FRONT的ERP‑FRONT数据集，包含约29K个ERP图像–3D网格对，用于训练和评估。

**📈 对比分析**

与单图像室内场景生成方法（SceneGen、MIDI、SAM3D）对比，在3D体素IoU、Chamfer距离、F1分数以及2D PSNR/LPIPS上均取得显著更高分数，尤其在整体布局与资产定位上表现优异。

**⚠️ 局限性**

局限性包括对光照与材质细节的依赖有限；在极端遮挡或非常大房间时仍可能产生误判；以及仅在合成数据上训练，虽然在真实全景上泛化良好，但对极端真实环境的鲁棒性尚待验证。

---

## 161. BanglaMemeEvidence: A Multimodal Benchmark Dataset for Explanatory Evidence Detection in Bengali Memes

**arXiv ID:** 2607.03981 | [PDF](https://arxiv.org/pdf/2607.03981v1)

**作者:** Fatema Tuj Johora Faria `[一作]` (Ahsanullah University of Science and Technology), Faisal Muhammad Shah `[通讯]` (Ahsanullah University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BanglaMemeEvidence数据集，并设计BengaliMemeEvidenceNet模型，用于在Bengali meme中自动检测并评分解释性证据。

**💡 创新点**

首次针对低资源语言Bangla meme开展证据检测任务，提供多模态标注并提出混合融合框架（早期、后期、交叉融合），实现最高0.74 F1，标志性突破。

**🔧 技术方法**

使用视觉Transformer（ViT、Swin、SwiftFormer、PoolFormer）与多语言预训练语言模型（mBERT、XLM‑RoBERTa、DistilBERT）进行特征提取，结合早期、中间、后期融合方法，并通过贝叶斯优化调参。

**📊 数据集**

BanglaMemeEvidence数据集，包含2917张Bengali meme，配有OCR文本、上下文、证据句、相关性分数，划分为训练、验证、测试集。

**📈 对比分析**

与多种基线（单模态、早期/后期融合）对比，BengaliMemeEvidenceNet在测试集上取得0.74 F1，优于其他模型，证明融合策略有效。

**⚠️ 局限性**

模型缺乏视觉-事实知识整合、易受词汇偏差影响导致误证，难以捕捉抽象幽默与文化语境，整体上下文理解不足。

---

## 162. MANCE: Manifold Aware Concept Erasure

**arXiv ID:** 2607.03973 | [PDF](https://arxiv.org/pdf/2607.03973v1)

**作者:** Matan Avitan `[一作]` (Bar-Ilan University), Yanai Elazar `[通讯]` (Bar-Ilan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Manifold Constraint Hypothesis（MCH）并基于该假设设计了MAN（Manifold aware Concept Erasure）方法及其变体，对现有概念抹除算法进行改进，验证其在多种文本与视觉任务中能更好地抹除目标概念而不损伤其他信息。

**💡 创新点**

创新点：1) 将概念抹除视为在自然表示流形上的受限编辑；2) 通过本地PCA估计切空间并将梯度投影进去，从而实现更精准、可控的编辑；3) 在已有抹除方法前加入闭式线性/协方差匹配预处理；4) 在119个不同设置下展示该方法在相同外科预算下实现接近随机泄漏、覆盖率更高的效果。

**🔧 技术方法**

技术：本地切空间估计（kNN + PCA/SVD）、非线性探针（MLP）、梯度投影与谱加权、闭式前置线性/协方差匹配（LEACE、CovMatch）、周期性重训练探针、局部步长自适应等。

**📊 数据集**

数据集：13款大规模语言模型（0.5B–27B）在sycophancy、gender、safety三种概念上；CelebA-CLIP 40个属性，按高/低相关控制两种外科预算；共119个实验设置。

**📈 对比分析**

对比方法：INLP、LEACE、IGBP、Obliviator等传统线性/非线性抹除器；通过目标泄漏（S）和外科预算（D_Y）评估。MAN+预处理在大多数设置下将泄漏降至接近随机且覆盖率更高，尤其在高相关控制的最难场景中明显优于基线，达到state‑of‑the‑art水平。

**⚠️ 局限性**

局限：1) 评估仅基于重训练的非线性MLP探针和有限的控制概念，未给出正式不可恢复性保证；2) 本地切空间估计在稀疏或高曲率流形上可能失效；3) 计算成本高（每轮kNN+SVD），部署时需要额外查询自然表示；4) 当内在维度接近表示维度时，流形约束优势可能衰减。

---

## 163. Order-based Causal Discovery for Multistage Processes

**arXiv ID:** 2607.03971 | [PDF](https://arxiv.org/pdf/2607.03971v1)

**作者:** Eun-Yeol Ma `[一作]` (Korea Advanced Institute of Science and Technology), Heeyoung Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种针对多阶段过程的因果发现框架（OCDM），通过引入阶段先验信息来确定变量的因果顺序，并利用神经网络进行剪枝，最终得到高质量的因果图。

**💡 创新点**

①在因果顺序搜索中引入“按阶段逆向逐步确定叶子”的结构知识驱动算法；②使用基于随机门控的神经网络（STG‑NN）替代传统稀疏回归实现剪枝；③将上述两项结合，形成适用于高维多阶段数据的高效因果发现方法。

**🔧 技术方法**

Score matching（DiffAN）+ 结构知识驱动的顺序搜索；基于 diffusion probabilistic 模型估计 score；STG‑NN 变量选择（随机门控网络）；传统 CAM‑pruning（稀疏样条回归）做对比；对照方法包括 PC、NOTEARS‑MLP、DAG‑GNN、GraN‑DAG、CAM、SCORE、DAS、DiffAN。

**📊 数据集**

①模拟数据（3阶段×10变量与20阶段×10变量）；②带噪声阶段标签的模拟；③混合类型数据（30% 二值、70% 连续）；④伪真实制造过程数据 causalAssembly（5阶段、98变量）。

**📈 对比分析**

在 AUROC、AUPRC、SHD、SID 等指标上与现有方法比较，OCDM 在所有实验中均取得最高或相近最高的性能；尤其在高维（200 变量）和伪真实数据上，OCDM（STG‑NN 剪枝）在准确率和计算速度上均优于 PC、SCORE、DAG‑GNN 等基线；在噪声阶段标签和混合类型数据下仍保持显著优势。

**⚠️ 局限性**

（1）仅适用于无潜在混杂且满足非线性加性高斯噪声模型；（2）对阶段标签的准确性要求较高，标签错误会影响顺序搜索和剪枝；（3）剪枝方法缺乏严格的理论保证，易出现过度剪枝；（4）实验规模受限，未验证超大规模或真实工业数据；（5）对非连续/高维混合型数据的理论支持仍不足。

---

## 164. Worldscape-MoE: A Unified Mixture-of-Experts World Model for Scalable Heterogeneous Action Control

**arXiv ID:** 2607.03964 | [PDF](https://arxiv.org/pdf/2607.03964v1)

**作者:** Jianjie Fang `[一作]` (Tsinghua University), Yong Li `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Worldscape-MoE，一种统一的 Mixture-of-Experts 世界模型，能在单一架构下同时学习摄像机轨迹、机器人动作和第一人称手部动作三种异构控制。

**💡 创新点**

创新点在于将控制信号分离为共享专家和控制专属专家，利用稀疏 MoE 机制实现跨控制知识共享与专属细化，并引入分阶段、渐进式的 MoE 调优策略，使模型能够持续扩展新控制。

**🔧 技术方法**

使用 Diffusion Transformer（DiT）骨干，结合控制感知注入（视觉、时序两路）与控制专属的稀疏 MoE 前向网络；训练时采用分组学习率与专家初始化策略，保持先验与新能力的平衡。

**📊 数据集**

在三大数据集上训练：iWorldBench（摄像机轨迹）、WorldArena（双臂机器人操作）以及 EgoDex/Ego4D（手部动作），并利用对应的自动注释与模拟扩充技术构建统一数据管道。

**📈 对比分析**

与现有基线（Matrix-Game 3.0、HY-World、CtrlWorld、VideoX-Fun-Wan、CogVideoX 等）在轨迹跟随、生成质量、操控精度、手部动作 FID/FVD 等指标上进行对比，Worldscape-MoE 在所有主指标上均获得最高分，表明异构训练提升了整体性能。

**⚠️ 局限性**

局限包括：对极端新控制的迁移仍需进一步验证；在极大规模数据和更复杂物理交互场景下的推理速度与显存需求仍高；以及 MoE 专家分配仍依赖手工设定，缺乏自动化的专家动态调度。

---

## 165. Kaizen: Metamorphic Fuzzing and Differential Testing for LLM-Translated HPC Applications

**arXiv ID:** 2607.04058 | [PDF](https://arxiv.org/pdf/2607.04058v1)

**作者:** Oscar Ludwig `[一作]` (Oregon State University), Manish Motwani `[通讯]` (Oregon State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于源代码变异模糊测试和差分测试的框架，用于评估大型语言模型在将CUDA代码迁移到OpenMP时的语义正确性。

**💡 创新点**

创新点在于：①引入语义保持的源代码变异模糊，生成多样化翻译输入；②结合运行时语法模糊与差分测试，揭示仅通过编译或静态测试无法发现的输入依赖语义错误；③构建了9类编译错误与27类全程序错误、6类语义错误的分类体系。

**🔧 技术方法**

使用了 metamorphic testing、语法基模糊（源代码与输入层面）、差分测试、LLM微调（ChatPORT）以及误差范数比较等技术。

**📊 数据集**

采用 HeCBench CUDA 应用集（16个用于完整测试，47个用于编译错误分析）以及 ChatPORT 的三种精调模型进行实验。

**📈 对比分析**

通过对编译成功率和在多种随机生成输入下的差分正确率进行比较，发现编译成功率与语义正确率无关；kernel‑level 翻译最高可达 72% 正确率，而 full‑program 翻译最高仅 28%；实验共生成 1,583 个代码变体，耗时约 3,958 CPU 小时。

**⚠️ 局限性**

局限性包括：仅评估 CUDA→OpenMP，模型仅支持单文件翻译；评估依赖 HeCBench 的内置验证，可能漏掉某些错误；模糊测试受随机种子影响；未覆盖大型多文件或更复杂的应用程序。

---

## 166. UniSGR: Unified Framework for Semantic ID Generation and Ranking

**arXiv ID:** 2607.04068 | [PDF](https://arxiv.org/pdf/2607.04068v1)

**作者:** Jiawei Sun `[一作]` (Alibaba International Digital Commerce Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commerce Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了统一框架 UniSGR，将语义 ID 生成和多目标排序融合在同一模型中，并通过两阶段训练实现跨场景迁移和目标对齐。

**💡 创新点**

创新点包括：① 价值感知并行多标记预测（VA‑PMTP）让生成过程更关注高价值行为；② 任务感知标记（TAT）在解码前嵌入业务目标信息；③ STARK 采用树形注意力和重组 KV 缓存，大幅提升生成推理吞吐量；④ 在生成器与排序器之间共享表示，实现端到端的业务目标一致性。

**🔧 技术方法**

核心技术包括：Transformer‑Encoder/Decoder（MemoryNet 编码器与稀疏 MoE 解码器）、多层语义 ID 量化（RQ‑VAE+Sinkhorn‑Knopp）、并行多标记预测、任务感知标记、对比学习（FACL）、树形注意力推理（STARK）以及多目标损失联合优化。

**📊 数据集**

使用 Alibaba 旗下 Lazada 电子商务平台的“Guess You Like”主页日志，包含多业务场景交互数据，进行多场景预训练与场景对齐训练。

**📈 对比分析**

与 TIGER、OneRec、OneRec‑V2 等基线比较，UniSGR 在 HR@100、HR@500 等检索指标提升约 2–3%；在在线 A/B 测试中相较基线提升 IPV 3.36%、Transaction Count 2.17% 与 GMV 5.68%。

**⚠️ 局限性**

局限性：① 语义 ID 量化码本规模增大后收益递减，需权衡容量与效率；② 生成模型对极稀缺项目或长尾行为的覆盖仍有限；③ 依赖大规模训练数据与硬件，对小型平台迁移可能受限；④ 业务目标权重设定与对比学习的负样本策略需经验调优。

---

## 167. SiamJEPA: On the Role of Siamese Student Encoders in JEPA

**arXiv ID:** 2607.04044 | [PDF](https://arxiv.org/pdf/2607.04044v1)

**作者:** Makoto Yamada `[一作]` `[通讯]` (Okinawa Institute of Science and Technology), Makoto Yamada (Okinawa Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于Siamese学生编码器的 Joint Embedding Predictive Architecture（SiamJEPA）框架，用来学习视觉表示。

**💡 创新点**

创新点在于将Siamese学生编码器与EMA教师网络结合，使得预测目标的互补信息得到充分利用，且通过KL正则化将两者拉近，起到正则化和加速收敛的作用。

**🔧 技术方法**

采用Transformer编码器、两路独立遮掩（block/随机），概率预测网络（posterior/ prior）、KL 和 NMSE 损失，以及EMA更新。

**📊 数据集**

在 ImageNet‑1K 数据集上进行预训练，并用线性探测评估。

**📈 对比分析**

与 MAE、CAE、I‑JEPA 等基准对比，SiamJEPA 在相同训练周期下达到相近或更优的 Top‑1 准确率（约 69–70%），并在更短时间内收敛。

**⚠️ 局限性**

局限性包括：对大规模模型验证不足、对超参数高度敏感、只评估了图像任务，对视频等多模态任务的推广尚未深入。

---

## 168. OmniOpt: Taxonomy, Geometry, and Benchmarking of Modern Optimizers

**arXiv ID:** 2607.04033 | [PDF](https://arxiv.org/pdf/2607.04033v1)

**作者:** Siyuan Li `[一作]` (Shanghai Artificial Intelligence Laboratory), Cheng Tan `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个统一的元流水线与四轴LMO驱动框架，对大语言模型训练中的100余种优化器进行分类、对齐，并在统一协议下进行大规模基准测试，揭示不同机制的性能优劣和组合潜力。

**💡 创新点**

创新点包括：①提出“通用元流水线”将所有优化器映射到同一五步更新流程；②用线性最小化算子（LMO）统一方向选择，形成四轴分解（域、状态估计、几何/预条件、最终化）；③构建双维度（方法与效果）分类法；④在此框架下设计跨尺度、跨架构、多目标的实验体系。

**🔧 技术方法**

主要技术包括：线性最小化算子（LMO）、梯度变换、状态演化、矩阵预条件（Kronecker、正交化）、低秩投影、量化压缩、Sharpness-aware 修正、权重衰减分离等；以及对上述机制的公式化与实现细节的抽象。

**📊 数据集**

实验数据集：大语言模型预训练（60M–1B 参数、四种 Transformer/线性注意力架构、上下文长度 256–32k token）；视觉分类任务（CIFAR‑100 采用 ResNet‑50、DeiT‑S、CAFormer‑S12），用于检验跨模态的优化器迁移性。

**📈 对比分析**

比较方法：在统一的学习率、调度、数据、模型架构固定的“对照变量”协议下，对六个效果维度（收敛效率、单步成本、内存占用、训练稳定性、超参鲁棒性、泛化质量）进行评估；结果表明：无单一优化器统治所有指标；几何敏感方向与结构化状态驱动的提升最显著；记忆压缩与矩阵几何在不同上下文、架构下表现截然不同，需按实际约束选择。

**⚠️ 局限性**

局限性：实验受限于特定协议与规模，结果高度依赖超参调优；对所有100余种方法的覆盖仅为代表性实例，未实现完整空间探索；机制归因多为定性，缺乏量化解释；仅在 LLM 与 CIFAR‑100 任务上验证，未覆盖更大视觉/多模态场景；对低秩压缩、梯度量化等技术的有效性尚需更系统的指标与理论支持。

---

## 169. A Unified Algebraic Framework for Classification Performance Evaluation

**arXiv ID:** 2607.04028 | [PDF](https://arxiv.org/pdf/2607.04028v1)

**作者:** Ronaldo C. Prati `[一作]` (Universidade Federal do ABC), Ronaldo C. Prati `[通讯]` (Universidade Federal do ABC)

**通讯引用:** 7700 | [OpenAlex ID](https://openalex.org/A5005717519)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于指示矩阵和三种聚合算子（微、宏、样本平均）的统一代数框架，用于在二分类、多分类、多标签、序数、层级、成本敏感和软标签等多种设置下自动生成评估指标。

**💡 创新点**

创新点在于实现了从任意二分类指标“自动生成”对应的多分类/多标签版本，并将软标签、成本矩阵、序数成员函数等扩展纳入同一框架，理论上统一了之前分散的评估方法。

**🔧 技术方法**

采用指示矩阵表示、三种聚合算子、三元组（TP、FP、FN、TN）的通用表达式，以及三角范式（t‑norm）与成本矩阵的双线性形式，进一步通过累加/加权/逐例聚合实现指标的多维推广。

**📊 数据集**

实验使用合成数据以及公开多标签数据集（Yeast）验证理论，并在10折交叉验证中对多种分类器（BR-LR、BR-SVM、BR-RF、BR-kNN）进行评估。

**📈 对比分析**

通过理论证明和实证展示微、宏两种平均方式在类别不平衡时会产生显著差异，排名互相冲突；微-F1、宏-F1、样本F1的排名差异可达完全逆转，验证了框架对评估结果影响的可解释性。

**⚠️ 局限性**

局限性包括：未对重采样（如交叉验证）中的折叠聚合做完整形式化；未探讨统计冗余与信息量理论；软标签的t‑norm选择仍依赖场景经验；对结构化输出（序列、树、图）尚无直接推广；以及框架对模型训练过程的直接优化影响尚未深入研究。

---

## 170. Patient-Conditioned Dual Hypergraph Reasoning for Auditable Traditional Chinese Medicine Prescription Support

**arXiv ID:** 2607.04025 | [PDF](https://arxiv.org/pdf/2607.04025v1)

**作者:** Weizhi Nie `[一作]` (Tianjin University), Yuting Su `[通讯]` (Tianjin University)

**通讯引用:** 6989 | [OpenAlex ID](https://openalex.org/A5033713097)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了患者条件化的双重超图推理框架，用于可审计的中医处方支持；

**💡 创新点**

创新点在于动态对患者进行超图权重调制，将患者表征作为调节器激活超边/路径，保留固定的TCM先验同时实现个性化推理与可解释路径；

**🔧 技术方法**

采用中文语言模型进行文本编码（BERT、MacBERT、Qwen2.5等）、超图神经网络、动态权重学习、路径一致性损失、以及与检索增强或受约束的LLM相结合的两阶段推理；

**📊 数据集**

使用公开数据集TCM‑SD（症候分型）、TCM‑BEST4SDT（处方推理）以及50例真实CAP病例做外部验证；

**📈 对比分析**

与传统统计、深度学习、检索、静态超图、以及受约束/直接LLM生成等多种基线比较，动态H1在TCM‑SD上将准确率提升至0.8297（宏F1 0.3288），动态H2在TCM‑BEST4SDT上Herb‑F1达0.3111，完整流水线Herb‑F1 0.3074，接近金标准0.3101；

**⚠️ 局限性**

局限性包括数据量有限、提升幅度相对 modest、对症候→治疗原则映射的鲁棒性不足、剂量安全性评估缺失，且尚未进行前瞻性临床验证

---

## 171. Paired Uterine Whole-Slide Images and Pathology Reports for Multimodal Computational Pathology

**arXiv ID:** 2607.04020 | [PDF](https://arxiv.org/pdf/2607.04020v1)

**作者:** Han Li `[一作]` (Technical University of Munich), Peter Schüffler `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了一个包含216个病例、455个全切片图像与对应诊断报告（含病例级和切片级）的多模态子宫疾病数据集TUM-Uteria，支持全切片图像与文本的精准匹配与研究

**💡 创新点**

首次实现了真实临床工作流程中全切片图像与完整诊断报告的双层级关联，采用多阶段专家验证和LLM辅助提取，解决了目前多模态病理数据缺乏的问题

**🔧 技术方法**

使用离线LLM（Qwen-30B）进行匿名化、翻译与切片级描述提取，结合手工审阅与专家评估进行质量控制，并使用Leica Aperio GT450Dx扫描器获取高分辨率WSI

**📊 数据集**

基于德国技术大学慕尼黑医院的临床病例，收集了216个子宫病理病例（共455个WSI）以及对应的诊断报告（德语原文及英文翻译）

**📈 对比分析**

本文未报告模型训练/评估结果，仅提供数据集发布与质量验证，未来可用于多模态学习与自动报告生成的基准实验

**⚠️ 局限性**

样本量有限（仅216例），主要包含H&E染色图像，缺乏免疫组化或特殊染色样本，且数据仅来自单一中心，可能难以覆盖所有罕见子宫病变类型

---

## 172. "AI Slop is DDoSing Open Source": Understanding the Impact of AI-Generated Contributions on Open Source Sustainability

**arXiv ID:** 2607.04003 | [PDF](https://arxiv.org/pdf/2607.04003v1)

**作者:** Sadia Afroz `[一作]` (Oregon State University), Zixuan Feng `[通讯]` (Virginia Commonwealth University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统研究了Generative AI工具导致的OSS贡献洪水（AI-DDoS）对社区可持续性的影响，并提出了11种缓解策略。

**💡 创新点**

首次将现象研究与贝叶斯结构时间序列结合，定量证明2025年AI贡献上升导致PR合并率下降，并系统性挖掘社区对策。

**🔧 技术方法**

采用主题分析、半结构化访谈、问卷调查，以及BSTS因果影响分析。

**📊 数据集**

基于294个活跃OSS仓库（共122万PR+82万Issue）与Reddit/博客/邮件三源灰色文献。

**📈 对比分析**

对比AI未介入的反事实轨迹，发现PR量+6.8%、合并率-1.06%及一次性贡献合并率-18.18%；效果显著且中等到大幅。

**⚠️ 局限性**

未直接测量个体AI使用，可能受其他生态系统变化影响，且样本集中于高流量公开仓库，结果对低活跃或私有项目的普适性有限。

---

## 173. Full Glyph Images Beat Token Embeddings: A Controlled Study for Transformers

**arXiv ID:** 2607.03994 | [PDF](https://arxiv.org/pdf/2607.03994v1)

**作者:** Shuyang Xiang `[一作]` (Independent Researcher), Hao Guan `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在中文语言建模中用单张字符序列的栅格化图像代替传统离散 token 嵌入，构建双分支对照框架，验证视觉输入的有效性。

**💡 创新点**

创新点在于将完整字符序列映射为二维图像作为模型输入，并通过共享 ResNet + ViT 编码器与 GPT-2 解码器，揭示视觉表示在样本效率、收敛速度和鲁棒性上优于索引嵌入。

**🔧 技术方法**

技术上采用共享 ResNet 提取局部特征、Vision Transformer 进行 2D 位置编码的全局建模、辅助损失（patch、global）以及 OneCycle 学习率调度和二次曲线数据计划。

**📊 数据集**

数据集为约 3,000 个汉字的训练语料，使用 5,000 条验证序列，此外还在 C‑Eval 基准上进行下游评估，并在英文 WikiText‑2 进行对比实验。

**📈 对比分析**

通过严格对照的双分支实验，视觉模型在 8×8 patch 下在 28 轮内达到 Acc@1 0.429（比索引 0.355 提升 21%），且在 10% 字符遮掩时仍匹配干净基线，验证了其更快收敛和更高最终精度。

**⚠️ 局限性**

局限性包括实验规模受限于 868M 参数、仅覆盖 3,002 个字符、视觉编码额外计算开销、以及英文实验结果受渲染方式限制，未能证明在更大语料或不同脚本上的普适性。

---

## 174. Knowing When to Stop: Predicting Execution-Consistency Convergence in Text-to-SQL

**arXiv ID:** 2607.03991 | [PDF](https://arxiv.org/pdf/2607.03991v1)

**作者:** Yaron Anavi `[一作]` (Gigaspaces), Isabella Cattinelli `[通讯]` (Gigaspaces)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了一种基于运行一致性预测的自适应停止策略，能在Text-to-SQL管道中动态决定停止生成的次数。

**💡 创新点**

首次将一致性轨迹作为1-D信号，使用轻量级模型预测一致性收敛点，并通过运行顺序置乱增强训练，显著提升了收敛预测与停止准确性。

**🔧 技术方法**

采用XGBoost、逻辑回归、1-D TCN模型预测收敛；对齐Beta-Bernoulli停止规则做基准；通过标签噪声注入评估鲁棒性；使用一致性窗口阈值做收敛判定。

**📊 数据集**

在BIRD公共金融子集及两个客户业务（光伏与户外产品）数据集上实验，共计约127条问题，每条100次生成。

**📈 对比分析**

与固定预算和Beta-Bernoulli基准对比，AUC最高0.913，检测RMSE最低8.18，平均可减少约35% LLM调用，轻度噪声下仍保持稳健。

**⚠️ 局限性**

数据量有限、标签基于人工标注且主观，且仅在单一GPT-4.1模型上评估，未检验跨模型与部署环境的泛化。

---

## 175. The Insertion List-Decoding Capacity and an Improved Bound on the Deletion List-Decoding Capacity

**arXiv ID:** 2607.03989 | [PDF](https://arxiv.org/pdf/2607.03989v1)

**作者:** Roni Con `[一作]` (Tel Aviv University), João Ribeiro `[通讯]` (Universidade de Lisboa)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文证明了二进制同步误差模型下，基于插入错误的列表解码容量可以被精确计算为 (δ)=(1+δ)(1-h(δ/(1+δ)))，并在删除误差情况下给出了更紧的上界，展示了对比随机码在删除误差上的局限。

**💡 创新点**

创新点包括：①首次给出插入误差的列表解码容量闭式表达；②通过对称 1 阶马尔可夫链生成的随机码实现该容量；③证明马尔可夫链随机码在删除误差列表解码上无法优于均匀随机码；④在小删除概率下获得与删除信道 Shannon 容量相匹配的上界。

**🔧 技术方法**

主要技术手段有：大偏差理论（Cramér 定理、Chernoff 绑定）、矩生成函数分析、马尔可夫链随机码的匹配过程、Levenshtein 球体体积上界、插入球体大小计算、信息论熵函数和相应的组合论计数。

**📊 数据集**

本工作为理论性研究，未使用任何具体数据集，所有结果均来自解析证明和随机过程的理论分析。

**📈 对比分析**

比较方法：与已知的随机码下界（1-h(δ)）以及传统插入/删除球体体积上界对比。结果表明：插入误差下，随机马尔可夫码可达到完全匹配的容量；删除误差下，随机马尔可夫码性能与均匀随机码相当，且上界与删除信道容量在 δ→0 时相吻合。

**⚠️ 局限性**

局限性：①结果为非构造性，缺乏可实现的码生成算法；②对高删除比例 δ>1/20 的精确上界仍较粗糙；③未给出实际的编码/译码算法；④在删除误差中仍无法突破均匀随机码的性能，需寻找更具结构性的码。

---

## 176. Conductance-Repair Evidence Graphs for Prospective Security Retrieval

**arXiv ID:** 2607.04070 | [PDF](https://arxiv.org/pdf/2607.04070v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Taylan Alpay `[通讯]` (University Of Turkish Aeronautical Association)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于时间戳可接受性掩码的“conductance-repair evidence graph”，用于前瞻性安全检索，并给出完整的修复证书；

**💡 创新点**

创新点在于将缺失渠道的修复视为图流操作并使用可证明的安全边界与证书机制，避免泄漏和误修复；

**🔧 技术方法**

核心技术包括时间戳图流递推、稀疏检索基线（BM25、PageRank、扩散）、Tensor后端统一实现（NumPy/PyTorch/JAX/TensorFlow）以及可验证的修复证书；

**📊 数据集**

使用公开安全数据源（NVD、KEV、EPSS、CVEfixes、SARD、CAVP、ASCAD）以及生物图像基准（BBBC019、LIVECell）作为稀疏演化控制；

**📈 对比分析**

通过与基线（度数、BM25、扩散、随机修复等）对比，实验显示在随机边欠缺下recall@k提升但AP下降，表明修复存在泄漏、毒化与饱和等风险；

**⚠️ 局限性**

局限性包括仅使用有限公开源、未覆盖完整漏洞库、缺乏真实攻击场景验证、并且对后端数值差异依赖较大，需进一步扩充数据与评估环境。

---

## 177. Securing Deep Learning Hardware: A Survey of Side-Channel Vulnerabilities and Countermeasures

**arXiv ID:** 2607.04055 | [PDF](https://arxiv.org/pdf/2607.04055v1)

**作者:** Zahra Mohammadi `[一作]` (University of Tehran), Siamak Mohammadi `[通讯]` (University of Tehran)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了深度学习硬件面临的侧信道攻击与对应防御，系统梳理了攻击技术、泄漏源和防御策略，并提出未来研究方向。

**💡 创新点**

首次提出了面向深度学习硬件的侧信道攻击统一分类框架，全面归纳代表性攻击实例与防御措施，并指出Transformer等新架构缺乏研究的空白。

**🔧 技术方法**

主要分析了功耗、EM、电磁辐射、缓存访问、时序差异等侧信道技术，以及掩码、随机化执行、缓存分区、噪声注入、物理屏蔽等防御手段。

**📊 数据集**

作为综述性工作未引入实验数据集，文中引用的攻击案例多使用MNIST、ImageNet等标准图像数据集来演示泄露效果。

**📈 对比分析**

对比了攻击精度与防御方案的性能、能耗与面积开销，强调攻击往往仅需数百~百万条功耗/EM轨迹即可恢复权重，而防御多导致延迟提升数十%至数倍、功耗上升数十%甚至面积增加数十%。

**⚠️ 局限性**

局限性包括：大部分攻击和防御集中在CNN，Transformer、LSTM等新型网络的侧信道研究不足；防御方案缺乏统一评估基准，难以平衡安全与效率；对多租户云环境的全局防护策略尚不完善。

---

## 178. SOGRAND decoding of LDPC codes

**arXiv ID:** 2607.04045 | [PDF](https://arxiv.org/pdf/2607.04045v1)

**作者:** Ken R. Duffy `[一作]` (Northeastern University), Muriel Médard `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

在SOGRAND算法基础上，针对单奇偶校验（SPC）代码，提出了两种新型CN更新规则，用于LDPC迭代软输入软输出（SISO）解码；

**💡 创新点**

创新点在于将SOGRAND产生的噪声效应列表与偶校验性质相结合，得到的CN更新算法硬件友好、复杂度低，并且在BLER/BER上与传统SPA、norm‑min‑sum相当或更优；

**🔧 技术方法**

技术手段包括SOGRAND与ORBGRAND查询序列、列表大小L、误码概率估计、并行硬件实现以及α尺度因子；

**📊 数据集**

使用5G NR标准中的LDPC(256,128)率1/2与LDPC(1024,676)率2/3两组码在BPSK AWGN信道上进行仿真；

**📈 对比分析**

对比方法为在50次迭代下计算BLER/BER，与MATLAB 5G NR toolbox实现的SPA与norm‑min‑sum进行对比；结果显示，当L≥8时，SOGRAND的BLER/BER仅比SPA少约0.05 dB，且L=10即可达到或超过SPA与norm‑min‑sum的性能；

**⚠️ 局限性**

局限性包括：仿真未涉及实际硬件验证，列表大小L仍影响实现成本，且对非偶校验情况下性能需进一步评估。

---

## 179. Reward-Gated On-Policy Distillation

**arXiv ID:** 2607.04037 | [PDF](https://arxiv.org/pdf/2607.04037v1)

**作者:** Mohammad Sadegh Akhondzadeh `[一作]` (University of Cologne), Aleksandar Bojchevski `[通讯]` (University of Cologne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于奖励门控的在策略蒸馏方法RG‑OPD，在训练过程中根据校验器（verifier）的奖励与教师-学生对数似然差异的方向性决定是否对学生进行教师蒸馏。

**💡 创新点**

创新点在于将稀疏的校验器奖励与密集的教师logit信息结合，通过奖励-教师门控（reward‑teacher gate）仅在奖励与对数似然差异方向一致时才更新学生，从而避免了传统无条件蒸馏导致的错误模式强化和正确行为抹杀。

**🔧 技术方法**

技术手段包括：①在策略蒸馏框架下采样学生轨迹并获取教师对数似然；②计算轨迹级奖励（使用GRPO advantage）；③设计门控函数g_i判断奖励与对数似然差是否一致；④对满足门控的轨迹使用逆KL（reverse‑KL）蒸馏损失进行更新。

**📊 数据集**

实验数据集：教师/学生模型为Qwen2.5‑1.5B/14B‑Instruct，训练基于UltraInteract子集，评估使用GSM‑8K、GSM+、MATH、MPMath、MBPP和IFEval等推理与编码基准。

**📈 对比分析**

与传统的逆KL蒸馏和TSD‑KD基线比较，RG‑OPD在1K生成长度下平均提升2.9点（逆KL）和4.9点（TSD‑KD），在8K长度下提升约2.8点；在长生成（8K）情境中相对未调学生提升8.2点，整体平均排名靠前。

**⚠️ 局限性**

限制：门控机制依赖奖励与对数似然的方向一致性，可能导致大部分轨迹被过滤，减少蒸馏信号；门控阈值δ的选择需要经验；方法主要针对具备校验器奖励的推理任务，通用性与教师模型不一致时效果未知。

---

## 180. The "I Don't Know" Filter: Enhancing Agentic Reliability in Function Calling

**arXiv ID:** 2607.04034 | [PDF](https://arxiv.org/pdf/2607.04034v1)

**作者:** Stefan Broecker `[一作]` (University of California), Thomas Strohmer `[通讯]` (University of California)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 Agent 在函数调用中的可靠性，提出了惩罚错误调用的 IDKS 评价指标，并设计了基于一致性的不确定性过滤器，让 Agent 在不确定时说 “I don't know”。

**💡 创新点**

创新点在于：①引入 IDKS 以量化错误调用的负面影响；②设计了轻量级、可训练的过滤器，利用白盒、灰盒、黑盒特征检测不确定性；③提出了合成数据生成 pipeline，解决标注稀缺问题。

**🔧 技术方法**

使用了随机森林分类器、语义体积与变异比等一致性特征、递归特征消除、以及基于 gpt‑4.1‑mini 的 constrained prompting 生成合成数据。

**📊 数据集**

主要数据集包括：Berkeley Function Calling Leaderboard（Simple Python、Parallel、Multiple、Parallel Multiple 等），Simple Java、Simple JavaScript、Irrelevance、以及 APIGen 的子集，所有数据通过 gpt‑4.1‑mini 合成用户问题。

**📈 对比分析**

通过在 Llama、Phi、Qwen 三大 SLM 上多次重复生成函数调用，比较无过滤器与过滤器后的 IDKS，发现过滤器在大多数基准上提升了 0.1–0.2 的 IDKS；进一步探究不同重复次数和特征侵入度的影响，展示了过滤器的稳健性与可迁移性。

**⚠️ 局限性**

局限性包括：①只利用一致性作为不确定性代理，无法处理“自信错误”；②对格式差异导致的语义相同调用误判；③阈值选择未进行系统校准；④实验仅在小型开源模型上验证，缺乏对大型闭源 LLM 的适用性研究。

---

## 181. TileLens: Efficiently Using Large-Granularity Memory Systems with Transparent Two-Dimensional Memory Layout

**arXiv ID:** 2607.04031 | [PDF](https://arxiv.org/pdf/2607.04031v1)

**作者:** Jae Hyung Ju `[一作]` (Georgia Institute of Technology), Moinuddin K. Qureshi `[通讯]` (Georgia Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

针对大型语言模型推理中GPU高带宽内存（HBM）容量不足的问题，论文提出并评估了将大型颗粒度内存系统（LGMS）与二维矩阵乘法匹配的 "tile‑major" 数据布局，并提供了软件（TileLens‑SW）与硬件（TileLens‑HW）扩展，使GPU能够透明地使用该布局。通过在模拟环境下对 Llama‑3.1‑70B 与 Qwen‑3‑30B 的矩阵乘法核进行测试，证明该方法可以在使用 HBF（高带宽闪存）时将性能下降从 1.61–6.49 倍降至 1% 的 HBM 级别。

**💡 创新点**

创新点在于：①发现并量化了 LGMS 对二维矩阵乘法产生的 "读取放大"（read amplification）瓶颈；②设计了 tile‑major 布局，将 4KB 内存块重新划分为与计算 tile 对齐的二维矩形，从而消除读取放大；③提出轻量级的 TileLens 系统，分别通过 DSL 扩展（TileLens‑SW）和 TMA 处理器扩展（TileLens‑HW）实现对 tile‑major 的透明支持；④在 HBF + HBM 混合内存体系结构中提供了多粒度 L2 缓存、MSHR 与自适应预取器的系统级配套方案。

**🔧 技术方法**

技术要点包括：
- Tile‑major 数据布局与 2D 矩阵乘法的匹配；
- CuTe 等 GPU DSL 的布局描述符扩展；
- 对 NVIDIA Hopper 架构 TMA（Tensor Memory Accelerator）的硬件指令集扩展；
- 混合粒度 L2 缓存、4KB MSHR 以及基于 stride 的自适应预取器；
- 采用 MacSim 循环级 GPU 仿真器，集成 HBF、RoMe 等 LGMS 模型；
- 通过对齐与填充保证内存块对齐，利用位移与掩码实现地址映射。

**📊 数据集**

数据集：使用公开的 LLM 权重量化模型 Qwen‑3‑30B（MoE 结构）和 Llama‑3.1‑70B（dense 前馈层），在不同批量大小（16/64/256）下采集 SASS 级别 trace，构建仿真工作负载。

**📈 对比分析**

比较方法：在三种内存配置（HBM‑only、HBM+HBF、HBM+HBF+SRAM 预取缓冲）下，比较三种布局（列主序、行主序、tile‑major）在 HBF 5µs NAND 读延迟场景中的归一化执行时间。结果显示：
- 列主序在 HBF 上导致 3–10 倍慢速；
- 行主序因读取放大消除而仅略慢（约 1.6–1.7 倍）；
- tile‑major 与自适应预取器组合实现几乎无损（1.01×），即 HBF 仅比 HBM 低 1%。
此外，作者还展示了带宽利用率、串行化 CTAs（straggler）与 HBF 延迟敏感性分析。

**⚠️ 局限性**

局限性：
- 评估仅在模拟环境中完成，缺乏真实硬件验证；
- Tile‑major 需要内存块对齐与填充，可能导致额外的存储开销；
- 预取器设计针对固定步长的 tiled GEMM，可能不适用于高度不规则访问模式；
- 对 GPU 现有编译器与驱动的硬件扩展要求较高，实际部署可能受限；
- 只关注读取放大与带宽问题，写耐久性与能源消耗等方面未作深入探讨。

---

## 182. SAGE: Synchronized Action-Gaze Recognition and Anticipation for Human Behavior Understanding

**arXiv ID:** 2607.04017 | [PDF](https://arxiv.org/pdf/2607.04017v1)

**作者:** Chenyi Kuang `[一作]` (Honda Research Institute), Nakul Agarwal `[通讯]` (Honda Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了SAGE框架，统一实现人-物交互（HOI）与人类注视（gaze）的实时识别与未来预测；

**💡 创新点**

首次将注视信息与HOI通过Gaze‑Conditioned Spatial Attention（GCSA）和Gaze‑Conditioned Temporal Prediction（GCTP）模块耦合，以双向方式建模当前与未来的相互依赖；

**🔧 技术方法**

基于Transformer的端到端网络，结合自注意力、交叉注意力和多采样Monte‑Carlo方法，对注视热图与HOI特征进行多尺度融合；

**📊 数据集**

在三个数据集上评测：Vid‑HOI（exocentric HOI检测/预测）、EGTEA Gaze+（egocentric注视与动作）以及新构建的Exo‑Cook（exocentric注视+HOI联合任务）；

**📈 对比分析**

与各类专门的HOI检测、注视检测/预测、动作识别与预测基线对比，SAGE在mAP、准确率、F1等指标上均优于或匹配现有最先进方法；

**⚠️ 局限性**

缺点包括对训练数据依赖较大、缺少跨场景泛化评估，以及未针对大规模预训练模型（如VLM/LLM）进行融合验证。

---

## 183. When Does Small Data Work? Accuracy and Efficiency Trade-offs Between Tabular Foundation Models and Conventional Methods for Crowd-State Classification at Hajj and Umrah

**arXiv ID:** 2607.04013 | [PDF](https://arxiv.org/pdf/2607.04013v1)

**作者:** AlJawharh S. AlOtaibi `[一作]` (Daldata), Jude AlSubaie `[通讯]` (Daldata)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在朝觐期间利用轨迹和可穿戴传感器产生的表格数据进行拥挤状态分类，系统评估了少标签场景下的表格基础模型与传统机器学习方法在准确率、标签效率和计算成本方面的表现。

**💡 创新点**

首次在真实朝觐数据上对表格基础模型的少标签性能进行系统性评估，并给出基于标签预算与计算预算的模型选择地图，揭示不同任务与数据条件下两类模型的优势差异。

**🔧 技术方法**

采用 TabPFN、TabICL、LimiX 等表格基础模型；梯度提升（XGBoost、LightGBM、CatBoost）及传统模型；严格的泄漏控制、组敏感划分、置换检验、配对 Wilcoxon 检验与自举置信区间；并对训练时间与推理时间进行度量。

**📊 数据集**

使用三个真实数据集：Jülich 控制实验轨迹、Lyon 节日现场轨迹、以及实测朝觐可穿戴传感器（Al‑Shaery），分别对应密度、流动几何和旅客疲劳三种拥挤状态目标。

**📈 对比分析**

在 16、64、256、1024 与完整标签预算下，比较宏 F1、低标签区间 AUC（标签效率）和计算成本。结果表明：在极少标签时，TabPFN 等基础模型在密度目标上优于调优梯度提升；随着标签增多，调优的梯度提升或树模型在几何目标及全量数据上更优。基础模型不需调优，但推理时需重新处理上下文，成本介于无调优梯度提升与调优模型之间。

**⚠️ 局限性**

研究仅覆盖近似朝觐场景的模拟数据，结果受任务与数据条件限制；不同基础模型表现差异显著，需指明具体模型；在大规模持续监测时基础模型的计算成本会显著上升；对实际操作阈值与置信度评估仍缺乏。

---

## 184. Candidate-Constrained Retrieval-Augmented Generation for LongEval-RAG: System Design and Empirical Analysis

**arXiv ID:** 2607.04008 | [PDF](https://arxiv.org/pdf/2607.04008v1)

**作者:** Yingdong Yang `[一作]`, Haijian Wu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套候选集约束的检索增强生成（RAG）系统，并在CLEF 2026 LongEval-RAG任务上进行评估。

**💡 创新点**

在候选集内执行检索、证据排序与引用跟踪，并通过规则式分块与晚期MiniLM句子重排实现高质量答案。

**🔧 技术方法**

BM25检索、伪相关反馈、逆序排名融合、轻量级证据重排、引用先验、MiniLM句子重排以及Deterministic provenance tracking。

**📊 数据集**

CLEF 2026 LongEval-RAG评估数据集，包含47条查询、每条10个官方候选文档。

**📈 对比分析**

采用组织者主评测（BERTScore、检索精度、nugget覆盖和平均分）以及LLM-judge诊断评测；最佳配置在主评测中获得最高BERTScore、检索精度和nugget覆盖。

**⚠️ 局限性**

对候选集外信息的排除限制了召回能力，且系统对不同评测指标的表现存在差异，难以统一最优方案。

---

## 185. Directional Curvature from Armijo Backtracking: A Low-Cost Sharpness Probe and a Calibration-Free Learning-Rate Safeguard for Adam

**arXiv ID:** 2607.03998 | [PDF](https://arxiv.org/pdf/2607.03998v1)

**作者:** Ashmitha R `[一作]` (Anna University), Jörg Frochte `[通讯]` (Bochum University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种仅在训练前进行一次Armijo回溯线搜索的“探测器”，用以估计局部损失曲率并给Adam（或AdamW）设置安全的初始学习率上限。

**💡 创新点**

创新点在于：① 将经典Armijo线搜索作为低成本的Hessian无关的局部尖锐度传感器；② 通过与顶层Hessian特征值的高度相关性（Pearson ≈ -0.9）证明其可读作Sharpness；③ 通过沿Adam自身预处理方向的探测，消除每个网络所需的手工安全因子，得到统一的κ=2。

**🔧 技术方法**

技术手段包括：Armijo回溯线搜索、梯度预处理（-g/|g|+ε）、一次性前向/后向传播、Hessian向量乘法用于验证、实验对照（Adam、Adam+clip、warmup、Schedule‑Free、Prodigy）。

**📊 数据集**

使用的数据集有CIFAR‑10（ResNet‑18）、Fashion‑MNIST（小型CNN）、Imagenette（全尺寸ResNet‑18）和AG News（Transformer）。

**📈 对比分析**

比较方法为在相同网络、相同超参数网格（η∈[10⁻³,3.0]）下，衡量测试精度与训练崩溃率。结果显示：单次探测器使Adam在所有学习率范围内保持≈0.79–0.83的精度，崩溃率降至0；相比之下，原始Adam在η≥10⁻¹时全部崩溃；参数无关的调度器只能覆盖部分范围。

**⚠️ 局限性**

局限性包括：① 对于极高曲率模型（如无归一化ResNet、无warmup Transformer）仍需一次一小分钟的安全因子校准；② 周期性探测在快速锐化的网络上效果不佳；③ 在更大规模（大语言模型、ImageNet）或自监督预训练场景下尚未验证。

---

## 186. Finite Observations, Infinite Behaviour: bicategorical semantics for stateful monoidal processes

**arXiv ID:** 2607.03996 | [PDF](https://arxiv.org/pdf/2607.03996v1)

**作者:** Cole Comfort `[一作]` (Université Paris-Saclay), Giovanni de Felice `[通讯]` (Relational Intelligence Ltd.)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种统一的语义框架，利用舍弃双范畴（discard bicategory）构造状态化过程的可观测行为，给出了在多种过程理论（部分、非确定、概率、量子）下通用的行为语义，并证明了该语义的函子性、状态的自然性以及不一致性在全局行为中的传播。

**💡 创新点**

创新点包括：①在舍弃双范畴上定义可观测行为范畴，实现从有限局部观测到全局行为的合成；②证明了闭合关系的范畴化紧致性定理，恢复了Willems的全局行为概念；③在量子、概率等非确定情形下保持行为的一致性，解决了传统共性语义在这些情形下的缺陷；④提供了可观测序列、因果序列等多种语义的关系，并给出了完整的函子映射。

**🔧 技术方法**

采用的技术主要是高阶范畴论：对称单张量范畴、舍弃双范畴、极限、可观测序列、闭合关系、复合运算、延迟自然变换、反馈结构等；同时利用了偏序增益、极限、有限上下文索引、拓扑空间闭合子集等工具进行证明。

**📊 数据集**

论文没有使用实验数据或具体数据集，全部为形式化的证明与理论构造。

**📈 对比分析**

对比方法主要是与已有的 Mealy 机器语义、状态化流语义、因果序列、可观测序列等传统语义进行理论对比，指出该框架在一致性传播、函子性、状态自然性等方面的优势；没有数值性能评估。

**⚠️ 局限性**

局限性包括：①需要上下文集合满足向上封闭/可覆盖性，限制了可直接应用的场景；②对量子过程的完全表达仍需进一步探讨（尤其是无限维系统和后选择等）；③在某些极限场景（如无限维 Hausdorff 空间）下紧致性定理的适用性受限；④缺乏对实际系统的案例验证，尚未展示在工程应用中的具体效能。

---

## 187. Evaluating 5G-connected IoT for Power Line Temperature Prediction: Real-World Latency and Cost Trade-offs Between MEC and Cloud

**arXiv ID:** 2607.03993 | [PDF](https://arxiv.org/pdf/2607.03993v1)

**作者:** Aakash Sharma `[一作]` (UiT Arctic University of Norway), Arne Munch-Ellingsen `[通讯]` (Telenor Research)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在挪威利用5G网络搭建了边缘计算（MEC）和云端两种部署架构，进行实时电力线路温度预测的端到端延迟和成本测评。

**💡 创新点**

首次在真实5G网络环境中量化MEC与多地区云部署在电力智能网格场景下的P99延迟差异与成本差距，并指出MEC在远程地区仍难满足8 ms的超低延迟需求。

**🔧 技术方法**

技术方案包括：FastAPI+Uvicorn容器化推理服务、Azure多区域容器部署、Telenor 5G NSA/SA网络、MQTT数据收集与Docker镜像，使用统计学指标（P99、CV、p‑value）评估延迟；利用CPU、内存计费模型估算成本。

**📊 数据集**

使用了包含947,590条记录的现场温度传感器数据集（含导线温度、环境温度、风速、湿度等八个特征），经过标准化并分为70/10/20的训练/验证/测试集，用于训练前馈神经网络。

**📈 对比分析**

比较方法：在三地（Tromsø、Oslo、Kårvik）对比MEC、NO‑East、NO‑West、SE‑Central四个部署点的P99延迟，计算平均延迟、波动率和统计显著性；结果显示MEC平均P99为44.62 ms，云端在NO‑East/NO‑West约47–49 ms，SE‑Central 78.49 ms；MEC延迟波动率最低，成本最低（SE‑Central 10.4 NOK/日）。

**⚠️ 局限性**

局限性包括：MEC仍无法满足约8 ms的超低延迟要求；仅测评单一推理任务，未考虑更复杂的AAR/DLR计算；成本评估未覆盖MEC定价；实验受限于Telenor 5G覆盖范围与网络波动；未对模型预测误差做深入分析。

---

## 188. Language models guide symbolic equation discovery by controlling search

**arXiv ID:** 2607.04156 | [PDF](https://arxiv.org/pdf/2607.04156v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 189. Masked Generative-Contrastive Representation Learning for Cross-Dataset EEG-Based Emotion Recognition

**arXiv ID:** 2607.04139 | [PDF](https://arxiv.org/pdf/2607.04139v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 190. TRACER: Early Failure Detection for Task-Oriented Dialogue

**arXiv ID:** 2607.03974 | [PDF](https://arxiv.org/pdf/2607.03974v1)

**作者:** Erfan Nourbakhsh `[一作]` (University of Texas at San Antonio), Anthony Rios `[通讯]` (University of Texas at San Antonio)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出TRACER方法，在任务导向对话中利用部分对话上下文提前预测对话失败并触发恢复。

**💡 创新点**

创新点在于双流设计：一条轨迹特征流捕捉对话状态随时间的动态变化，另一条文本流使用RoBERTa对序列化的信念状态进行编码；两者融合后可在对话仅完成25%时就产生可用的失败预测信号；并通过阈值化实现可操作的干预策略。

**🔧 技术方法**

技术主要包括：RoBERTa预训练文本编码、Transformer时序编码、轨迹特征计算（振荡、覆盖率、冲突计数、填充速率、域切换）、早期预测奖励的损失函数、阈值化的干预策略。

**📊 数据集**

主要使用MultiWOZ 2.1数据集做主实验，Oracle信念状态与自动生成信念状态两种设置；另外在SGD和ABCD数据集上做零样本跨域迁移实验。

**📈 对比分析**

与多种基线对比：无干预、固定频率、槽置信度、手工特征阈值、单流特征逻辑回归、纯文本RoBERTa、LLM零/链式/少样本提示。TRACER在Oracle设置下AUC-ROC 0.741、F1 0.463，在生成设置下AUC-ROC 0.617、F1 0.370，均显著优于所有基线；在固定上下文实验中，即使仅观察25%对话也能达到AUC-ROC 0.666，75%时已接近完整对话性能。

**⚠️ 局限性**

局限性包括：评估仅离线，未验证在实时交互中的任务成功提升；失败分类为人工解释非正式标签；跨域迁移不均衡（对SGD表现差）；轨迹特征粗略（如域切换不考虑顺序/最近性）；仅靠EDS容易导致过多误报，需要阈值平衡。

---

## 191. Neuro-Symbolic Reasoning for Vulnerability Detection

**arXiv ID:** 2607.03963 | [PDF](https://arxiv.org/pdf/2607.03963v1)

**作者:** Yanjie Zhao `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将 LLM 作为语义事实筛选器、Lean4 作为安全义务验证器的神经符号漏洞检测框架

**💡 创新点**

核心创新是将事实提议与安全义务的解除分离，避免 LLM 直接下结论，提升安全义务保留的可靠性

**🔧 技术方法**

使用 LLM（DeepSeek-V4-Pro）进行结构化事实编辑，Lean4 进行符号验证，结合 AST 提取、自动配对和证据感知裁决

**📊 数据集**

构建了 801 条样本的漏洞数据集，涵盖 5 类关键 CWE（NULL deref、buffer overflow、use-after-free、out-of-bounds read、double free）

**📈 对比分析**

与纯 LLM、Codex 和 Claude Code 代理基线比较，在所有 15 组设置下 F1 指标均显著提升（尤其是生命周期类 Recall 提升约 2 倍）

**⚠️ 局限性**

局限于每个 CWE 需要手工编写规范，缺乏全局自动路由和完整程序上下文推理，且对新颖或跨文件漏洞的覆盖有限

---

## 192. Neural LiDAR Bundle Adjustment

**arXiv ID:** 2607.04169 | [PDF](https://arxiv.org/pdf/2607.04169v1)

**作者:** Chin Yung Anson Hon `[一作]` (Imperial College London), Sen Wang `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种针对LiDAR的NeRF束调整算法NeLD-BA，通过体积采样优化LiDAR姿态与场景表示。

**💡 创新点**

创新点在于针对LiDAR特性设计稠密体积采样、基于范围的损失、以及无幅度梯度的傅里叶编码。

**🔧 技术方法**

使用NeRF结构、改进的层次采样、正则化的范围损失、伪Dirac终止分布以及渐进激活的傅里叶特征。

**📊 数据集**

在Newer College和FusionPortable两大LiDAR数据集上训练与评估。

**📈 对比分析**

与传统ICP、FAST‑LIO2、HBA、BALM以及最新NeRF映射方法（SHINE‑Mapping、PIN‑SLAM、4dNDF）比较，NeLD‑BA在原始点云Chamfer距离、渲染地图F1分数和轨迹ATE上均实现最佳或接近最佳性能。

**⚠️ 局限性**

局限性包括对大规模场景的归一化限制导致模型容量不足，以及训练时间相对较长。

---

## 193. MDL Meets Latent Confounders: LNML-based Causal Discovery

**arXiv ID:** 2607.04133 | [PDF](https://arxiv.org/pdf/2607.04133v1)

**作者:** Zhongyi Que `[一作]` (University of Tokyo), Kenji Yamanishi `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于MDL（LNML）原理的因果发现框架，能够在存在非线性机制和潜在混杂变量的情况下进行因果结构学习。

**💡 创新点**

创新点包括：① 将LNML代码长度用于四种因果关系（A→B、A←B、A↔B、A↮B）比较；② 引入Δ伪共线性判据识别潜在混杂；③ 基于此设计的贪心PCG‑CD算法，复杂度仅 O(|V|^2)。

**🔧 技术方法**

主要技术：Gaussian Process 回归构建非线性因果机制、Luckiness Normalized Maximum Likelihood 代码长度、Δ伪共线性评估、贪心边删改造。

**📊 数据集**

使用的数据集包括合成数据（线性和非线性场景）和 Auto MPG 实测数据（可观测变量与隐变量）。

**📈 对比分析**

与 FCI、FCI(all) 以及 ICA‑LiNGAM 等方法比较，PCG‑CD 在 Z‑hit、E‑hit、Sim1/Sim2 上表现更优，尤其在识别潜在混杂和定向边方面显著领先，且在多变量情形下保持稳定。

**⚠️ 局限性**

局限性：依赖 GP 计算，计算量随样本/维度增长显著；伪共线性阈值需要经验设定，对噪声敏感性及非线性混杂的理论保证尚未完全给出；对极大规模数据的可扩展性待验证。

---

## 194. SOV-CAD: Stepwise Orthographic Views Guided CAD Modeling Sequence Reconstruction

**arXiv ID:** 2607.04119 | [PDF](https://arxiv.org/pdf/2607.04119v1)

**作者:** Zhaopeng Feng `[一作]` (Zhejiang University), Xinkui Zhao `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SOV-CAD，一个基于 Decision Transformer 的离线强化学习框架，利用逐步正交视图和草图的多模态视觉反馈重建 CAD 建模序列。

**💡 创新点**

创新点：①引入逐步多模态监督（正交视图+草图）和几何一致性奖励；②将 CAD 重建建模成离线 RL 问题，使用 Decision Transformer 处理视觉反馈；③通过连续视觉奖励提升生成序列的准确性与人类建模相似度。

**🔧 技术方法**

技术手段：ViT-Base 视觉编码、Decision Transformer、离线 RL（return-to-go 计算）、IoU/pHash 奖励、三视图投影、草图图像、Transformer GPT‑style 结构、参数离散化、位置编码。

**📊 数据集**

使用的数据集：CADParser（主训练集），DeepCAD（评估对比集），OpenCascade 生成三视图、计算 IoU，Matplotlib 绘制草图。

**📈 对比分析**

与 CADParser、DeepCAD、SkexGen、HNC‑CAD 等基线对比：在 CADParser 上 median CD 下降至 0.23（比 0.81 改进显著），IoU 提升至 0.85；在 DeepCAD 上 median CD 下降至 0.37（比 1.18 低 66%），IoU 提升至 0.83；无效率 IR 也显著下降，表明方法在精度、鲁棒性和数据效率上均优于现有方法。

**⚠️ 局限性**

局限性：奖励仍采用简单的 IoU / pHash 计算，缺乏更细粒度的几何一致性度量；需要三视图输入，无法直接处理视角不完整或遮挡严重的情况；缺少在线 CAD 交互反馈，可能限制在复杂设计场景下的适用性；对非标准 CAD 格式和更大规模模型的泛化尚未充分验证。

---

## 195. GlacierCastAI: Predicting Glacier Retreat from Multi-Modal Satellite Imagery and Climate Signals

**arXiv ID:** 2607.04117 | [PDF](https://arxiv.org/pdf/2607.04117v1)

**作者:** Arunkumar Ramachandran `[一作]` `[通讯]`, Arunkumar Ramachandran

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了GlacierCastAI，一种融合多时相Landsat影像、ERA5气候变量和DEM地形特征的多模态时空预测模型，用于预测冰川边界的未来退缩；

**💡 创新点**

创新点在于将冰川边界预测转化为多模态时空预测任务，采用跨注意力融合气候信息，验证气候信号在预测中的可行性，并展示气候仅有的MLP模型几乎与影像模型同等；

**🔧 技术方法**

使用了ResNet50空间编码器、ConvLSTM时序模型、跨注意力气候融合模块、轻量级气候MLP基线以及SHAP归因分析；

**📊 数据集**

数据集包含2000-2023年多时相Landsat影像、ERA5季节气候变量（温度、降水、降雪、太阳辐射）以及Copernicus GLO-30 DEM，共五个跨气候区冰川；

**📈 对比分析**

通过与传统的持久性与线性趋势基线比较，模型在IoU上实现0.320-0.337，气候信息提升3.4%，相较基线提升89-99%；

**⚠️ 局限性**

局限性包括单一随机种子实验导致结果方差未知；小冰川因ERA5分辨率粗糙而预测不足；DEM特征在当前尺度下略显冗余；未来工作需多种子评估和细尺度气候数据集成。

---

## 196. Target-Aware Interaction-Guided Reinforcement Learning for Black-Box Node Injection Attacks on Graph Neural Networks

**arXiv ID:** 2607.04091 | [PDF](https://arxiv.org/pdf/2607.04091v1)

**作者:** Yi Lan `[一作]` (Southwest University), Ye Yuan `[通讯]` (Southwest University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种针对黑盒节点注入攻击的强化学习框架TIRBA。

**💡 创新点**

创新点在于将特征生成与边构造统一为MDP，并通过目标感知交互编码、类别中心引导以及拓扑差异感知评论器显著提升攻击效果。

**🔧 技术方法**

使用A2C Actor-Critic、GAT编码、Gumbel-Softmax、类中心引导、拓扑差异评估等技术。

**📊 数据集**

使用四个真实世界引用网络数据集：Cora、Citeseer、Cora-ML、Pubmed。

**📈 对比分析**

与Clean、Random、TDGIA、PGD、GCIA、G^2A2C等基线对比，TIRBA在单节点单边注入、低预算下的误分类率最高，往往比最优基线高20–30%，并在多种受害模型上保持高成功率。

**⚠️ 局限性**

局限在于仅研究单节点单边注入、仅对节点特征与边有限预算，且对抗目标主要依赖伪目标类，缺乏对更复杂攻击情景的泛化验证。

---

## 197. FedSPM: Routing-Enabled Federated Learning under Dual Heterogeneity via Semiparametric Mixture

**arXiv ID:** 2607.04085 | [PDF](https://arxiv.org/pdf/2607.04085v1)

**作者:** Zijian Wang `[一作]` (Renmin University of China), Qiong Zhang `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于半参数混合模型的联邦路由-预测框架，用于在存在双重异质性（跨客户端与内客户端）时同时提升路由准确率和预测准确率。

**💡 创新点**

创新点包括：
1) 通过引入潜在组件变量，将每个客户端划分为多种子群，捕获内客户端异质性；
2) 对特征分布使用密度比模型（DRM）相对共享非参数基准，既能灵活建模，又能有效共享信息；
3) 将EL（经验似然）与EM算法结合，推导出可在联邦环境下实现的分布式EM，并证明其收敛性；
4) 在路由阶段使用基于密度比的匹配概率，真正实现“专家识别”式路由。

**🔧 技术方法**

技术手段包括：
- 半参数混合模型（共享编码器+客户端专属头）
- 密度比模型（DRM）与经验似然（EL）
- 联邦EM（局部E步+局部M步，跨客户端仅共享摘要统计）
- 本地带动量的SGD优化
- 理论收敛分析（1/√T收敛率）

**📊 数据集**

数据集：
- 合成双重异质性基准（FMNIST、CIFAR‑10、CIFAR‑100）
- 真实医疗数据（Fed‑ISIC2019，6个临床中心，共8类皮肤病），用于验证模型在实际复杂异质性中的效果。

**📈 对比分析**

与多类基线（FedAvg、FedProx、FedAvgFT、Ditto、ClusterFL、FedBABU、FedEM、FedGMM、FedDRM）在系统准确率与平均准确率上对比。实验表明：
- 在所有基准和真实数据集上，FedDRM consistently 取得最高系统与平均准确率；
- 路由能力显著优于仅个性化但无路由的基线；
- 在内外异质性加剧时，FedDRM 的优势更为突出。

**⚠️ 局限性**

局限性：
- 当组件数增多、每个组件独立编码器时，样本稀释导致在样本较少的任务上性能下降；
- EM算法需要求解非线性拉格朗日乘子，计算开销较大；
- 对基准数据分布的假设（如混合成分数、Dirichlet分布）需人工设定，可能不适用于所有场景；
- 目前仅在相对小规模客户端（≤8）和特定任务上验证，缺乏大规模多中心实验和跨领域迁移研究。

---

## 198. Benchmarking API Drift in LLM-Generated Quantum Code Across Successive SDK Versions

**arXiv ID:** 2607.04072 | [PDF](https://arxiv.org/pdf/2607.04072v1)

**作者:** Mohammad Arif Rasyidi `[一作]` (Khalifa University), Syahirul Faiz `[通讯]` (Khalifa University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构建并评估了名为 quantum‑api‑drift 的基准，用来测量大型语言模型（LLM）在生成 Qiskit 代码时对 SDK 版本的对齐情况，实验涵盖 17 个模型、50 个量子编程任务、450 条生成样本，并在 Qiskit v0.43、v1.3 与 v2.0 三个版本中执行代码。

**💡 创新点**

创新点在于将版本对齐视为独立评估维度，提供跨版本漂移矩阵、错误分类和文档驱动的迁移修复机制，揭示中间版本 v1.3 对模型的挑战尤为显著。

**🔧 技术方法**

技术上使用了多种 LLM（如 GPT‑5.4、Claude Opus、Grok 等）通过 REST API 或 Codex CLI 生成代码，并利用带版本锚点的提示、统一的执行 harness（判定 Pass/Soft‑fail/Hard‑fail）、Pass@k、漂移矩阵以及迁移说明的修复提示。

**📊 数据集**

数据集来源于 Qiskit HumanEval 的 50 个任务（覆盖电路构造、门操作、转译、后端模拟等），在保证 SDK 版本中立的前提下重新编写提示与测试 harness。

**📈 对比分析**

通过 Pass@1、Pass@3 以及漂移矩阵进行比较，实验显示最佳模型在 v0.43 与 v2.0 上可达 0.85 的版本完整性，但在 v1.3 仅约 0.5；文档驱动的修复对 v2.0 的成功率可达 1.0，而对 v1.3 则接近 0。

**⚠️ 局限性**

局限性包括：仅验证 API 级别的执行成功而不保证电路语义正确；版本无关重写导致部分任务无法完全保证语义；训练截止与模型能力混合；Codex CLI 与 REST API 的部署差异导致不可直接对比；修复仅一次且未迭代；迁移说明的质量对修复率影响较大。

---

## 199. Information-Geometric Superposed Vowel Evaluation: Part 1. Moraic Syllabary (Japanese)

**arXiv ID:** 2607.04154 | [PDF](https://arxiv.org/pdf/2607.04154v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 200. Geometry of Ordinal Representations in Language Models

**arXiv ID:** 2607.04167 | [PDF](https://arxiv.org/pdf/2607.04167v1)

**作者:** Saksham Bassi `[一作]`, Sharvi Tomar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型在四类顺序任务（括号嵌套深度、Python 缩进、Markdown 表格位置、数值大小）中是否存在通用的几何流形，并探讨注意力头如何进行几何变换。

**💡 创新点**

创新点在于将字符计数中的 1D 曲线流形概念推广到多种顺序任务，并在 Gemma-2 与 Qwen3 三个开源模型间进行跨架构比较，揭示几何计算与模型架构的依赖关系。

**🔧 技术方法**

使用了线性探针、PCA 流形发现、稀疏自编码器特征分解、注意力头的扭曲/对齐/序数评分以及激活补丁子空间消融等技术。

**📊 数据集**

采用了四个自定义任务的数据集：括号嵌套深度、Python 缩进层级、Markdown 表格行列索引以及数值大小（log10），分别从 The Stack、OpenWebText 等来源生成。

**📈 对比分析**

通过对每个模型和层级的探针准确率、PCA 方差解释度、扭曲分数等指标进行对比，发现 Gemma 模型的 1D 流形更为清晰，而 Qwen3 的几何扭曲更强，整体性能在不同任务上差异显著。

**⚠️ 局限性**

局限性包括括号深度任务的过拟合、缺乏 Qwen3 的预训练 SAE、扭曲评分仅基于均值可能忽略单词级变换，以及子空间消融仅显示信息集中而非因果影响。

---

## 201. Piercing Gilbreath's Conjecture: From Deep Number Theory Insights to Fintech and Cybersecurity

**arXiv ID:** 2607.04166 | [PDF](https://arxiv.org/pdf/2607.04166v1)

**作者:** Vincent Granville `[一作]` `[通讯]` (BondingAI), Vincent Granville (BondingAI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于筛法（尤其是逆筛法）的新方法，用来研究并尝试证明Gilbreath猜想，并进一步定义了高效序列走廊、魔法素数、禁忌素数星座等概念，探讨了序列的可行性与失败模式；同时将该理论扩展到随机数生成、欺诈检测、时间序列异常检测等应用领域。

**💡 创新点**

创新点在于：①引入逆筛法并证明其可用于构造满足Gilbreath性质的序列；②提出高效序列走廊与魔法素数概念，给出序列成功的可行判定准则；③揭示禁忌素数星座对失败的影响；④将三角差分结构与概率模型结合，设计可用于检测错误、欺诈和生成随机数的工具。

**🔧 技术方法**

使用了数论中的筛法、Eratosthenes筛、Jacobsthal函数、Euler φ 函数等；三角差分（递归求绝对差）构造算法；随机生成Poisson增量序列、逆向构造canonical form；概率与统计分析（如π_k 概率、平衡右对角线分析）；算法复杂度分析（O(n^2) vs O(n)）。

**📊 数据集**

主要数据集为：①已验证到 10^14 的素数序列；②OEIS A358691 中的可接受序列；③通过Poisson λ=2.5 生成的随机序列；④合成素数序列（满足某些约束的素数序列）。

**📈 对比分析**

比较方法主要通过统计成功/失败序列数量、比例 h(n)/g(n)、以及对走廊中成功率的实验（例如 n=11 时成功率 >96%）；在算法层面比较复杂度：完整三角表 O(n^2)；仅用右对角线 O(n)；在随机序列搜索中使用二分法降低到 O(log n)。实验表明，对于大多数随机/素数序列，成功率极高且失败率趋近于零。

**⚠️ 局限性**

局限性包括：①尚未给出完整严谨证明，许多结果仍为猜想；②依赖于未验证的假设（如 Cramér 猜想、最大素数间隙上界）；③逆筛法参数 κ_0 的界定尚未确定；④计算实验仅覆盖有限长度序列，难以推断极大 n 的行为；⑤对于非素数或快速增长序列，方法的适用性有限。

---

## 202. SeeMe: Mitigating Hallucinations in Large Vision-Language Models through Effective Visual Token Engineering

**arXiv ID:** 2607.04163 | [PDF](https://arxiv.org/pdf/2607.04163v1)

**作者:** Kai Tang `[一作]`, Shanghang Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SeeMe 框架，通过三阶段视觉令牌重构来减少大型视语言模型的幻觉现象。

**💡 创新点**

首次将传统机器学习的特征工程思想引入视觉令牌处理，构建无训练的多阶段令牌工程流程。

**🔧 技术方法**

基于交叉模态注意力的令牌筛选、相似度驱动的令牌融合以及注意力引导的最终选择，配合 LLM 解码。

**📊 数据集**

在 MME、POPE（MSCOCO、A-OKVQA、GQA）和 AMBER 等多模态评测基准上进行验证。

**📈 对比分析**

与常规解码、VCD、DoLa、DCLA、SPIN 等方法对比，SeeMe 在四大 LVLM（LLaVA‑1.5、LLaVA‑NEXT、INF‑MLLM、mPLUG‑Owl2）上均取得更高的感知/认知/总分，显著降低幻觉率。

**⚠️ 局限性**

对不同模型需要手动调节层数、保留比例等超参；保留比例过低仍可能丢失细粒度信息；仅在实验基准上验证，未证明对所有新型 LVLM 的普适性。

---

## 203. Perceiving Better Moments: Cover Frame Reselection and Enhancement for Live Photos with the Live2K Dataset

**arXiv ID:** 2607.04151 | [PDF](https://arxiv.org/pdf/2607.04151v1)

**作者:** Junyu Lou `[一作]` (University of Electronic Science and Technology of China), Shuhang Gu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 Live Photo 中封面与视频帧的质量差异，提出封面重选与增强任务并给出了统一的多帧融合 + 参考引导增强网络。

**💡 创新点**

首次定义 LPRE 任务、构建 Live2K 数据集，并提出一体化的多帧融合、颜色调制和纹理迁移的端到端模型。

**🔧 技术方法**

采用时空注意力（TSA）、参考图像的颜色调制（FiLM+交叉注意）和纹理迁移（SWCA）以及 Swin Transformer + PixelUnshuffle 等技术。

**📊 数据集**

使用 iPhone 16 Pro 与 OPPO Find X8 Pro 共同拍摄的 2042 张 Live Photo 组成的 Live2K 数据集。

**📈 对比分析**

在 Live2K 上与 SISR、BurstSR、RefSR 等方法对比，平均 PSNR 提升约 0.7–1.2 dB，速度最快且 GPU 内存占用最小。

**⚠️ 局限性**

主要局限为仅包含昼间光照场景，缺少夜景/低光照等极端条件，跨设备泛化仍需进一步改进。

---

## 204. Beyond Scene Priors: Fine-Grained Traffic Scene Reasoning with Benchmarking and Query-Guided Small-Object Focus

**arXiv ID:** 2607.04149 | [PDF](https://arxiv.org/pdf/2607.04149v1)

**作者:** Waikit Xiu `[一作]` (University of Hong Kong), Xiying Li `[通讯]` (Sun Yat-Sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向交通场景的细粒度推理基准 FGTR‑Bench，并设计了在 Qwen3‑VL‑4B 上实现单通道查询引导小物体关注（TG‑SOF）的 TSR‑MLLM 模型，解决了多模态大型语言模型在细粒度视觉推理中因背景覆盖导致的关键信息稀释问题。

**💡 创新点**

创新点包括：① 通过多代理生成、结构一致性检查和人工审核构造高质量、局部证据可验证的单图像多选题；② 在解码前添加 TG‑SOF 以实现查询引导的稀疏 Top‑K 视觉残差更新，保持单通道推理且无需外部检测器；③ 结合边框对齐损失和隐藏状态一致性约束，为模型提供显式空间对齐信号。

**🔧 技术方法**

使用技术：Qwen3‑VL‑4B 的视觉编码器 + 语言解码器；TG‑SOF 模块（查询‑视觉相关性打分、局部对比度锐化、Top‑K 门控残差）；LoRA 微调；多任务损失（答案交叉熵、边框聚合 KL、隐藏状态一致性）。

**📊 数据集**

数据集：FGTR‑Bench（40,236 训练/验证实例 + 4,947 独立测试集），来源于 TT100K、LISA、道路与车辆摄像头采集的真实场景；此外在不做额外微调的情况下在 DriveQA‑V（CARLA 交通标志）上评估跨域迁移。

**📈 对比分析**

与 4B 规模基线比较：TSR‑MLLM 在 FGTR‑Bench 测试集上整体准确率 74.1%，比未微调 Qwen3‑VL‑4B 72.0% 提升 2.1%；在 DriveQA‑V 上同样表现最优，零样本迁移效果优于 Traffic‑MLLM 等竞争模型。

**⚠️ 局限性**

局限性：仅针对单帧图像的多选问答；不支持多帧输入、自由文本生成 VQA 或多步推理；基准覆盖的地理、气候与极端场景仍有限；高准确率并不直接等同于在真实部署中的安全保证。

---

## 205. HCSU: A Dataset and Benchmark for Fine-Grained Historical Calligraphy Style Understanding

**arXiv ID:** 2607.04147 | [PDF](https://arxiv.org/pdf/2607.04147v1)

**作者:** Yinsheng Yao `[一作]` (Tongji University), Chen Ye `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了HCSU数据集及其基准，用于细粒度历史书法风格理解。

**💡 创新点**

创新点在于解耦墨本与石刻、提供分层专家书法描述，构建双任务评测（风格辨别与可解释美学推理）。

**🔧 技术方法**

采用大型视觉‑语言模型（如GPT‑5.2、Qwen3‑VL‑235B）作为基线，结合统一的图像预处理与文本提示技术。

**📊 数据集**

使用HCSU数据集，包括39,307幅字符图像，分为Tie、Bei、Wild三域，涵盖49名书法大师及10个朝代。

**📈 对比分析**

通过8路候选选择和BERTScore/LLM‑评测对比，现有主流模型在风格辨别上仅达到约30‑38%准确率，生成任务的BERTScore仍低于专家水平。

**⚠️ 局限性**

局限包括样本不平衡、Tie样本来源多样导致墨迹细节缺失、数据覆盖仍有限，且模型对石刻表现优于墨本，说明视觉编码仍缺乏对连续纹理的敏感性。

---

## 206. Semantic-Guided Progressive Object Removal with Gaussian Splatting

**arXiv ID:** 2607.04144 | [PDF](https://arxiv.org/pdf/2607.04144v1)

**作者:** Xianliang Huang `[一作]` (ByteDance Inc.), Hao Zhang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于3D Gaussian Splatting的三维物体去除框架，通过语义引导的块匹配（SBM）和区域渐进细化（RPR）实现高质量、跨视角一致的重建。

**💡 创新点**

创新点在于：① 将多视角的DINOv2语义特征用于块级匹配，精确对齐缺失区域的语义内容；② 采用频率感知的高频特征提取器指导局部再重建，逐步细化低质量块；③ 将SBM与RPR嵌入Gaussian Splatting中，兼顾几何一致性与纹理细节。

**🔧 技术方法**

使用的技术包括：3D Gaussian Splatting、DINOv2语义编码器、Stable Diffusion v2.1（Latent Diffusion + Score Distillation Sampling）、高频 Sobel 特征、频率感知 UNet、跨视角语义匹配与交叉注意力。

**📊 数据集**

在公开数据集（SPIn-NeRF、Self-captured、Mip-NeRF 360）以及自采集场景上进行实验，覆盖前视、360°无界等多种视角。

**📈 对比分析**

与多种基线（SPIn-NeRF、GaussianEdit、GaussianGroup、MVInpainter、InFusion）比较，指标（PSNR、SSIM、LPIPS）均显著提升，最高 PSNR 28.7/29.6/27.4，SSIM 0.929/0.936/0.899，LPIPS 0.139/0.131/0.161；训练时间约7.9h，推理13min，显存10GB，表现出优异的性能与效率。

**⚠️ 局限性**

局限性包括：依赖预训练的语义编码器与Diffusion模型，对极端遮挡或极端纹理细节仍可能出现细小伪影；RPR对阈值敏感，需手动调参；当前实现对大规模场景的可扩展性和实时性尚待进一步验证。

---

## 207. Asymptotic-Preserving A Posteriori Analysis of Diffusion and Flow-Matching Samplers

**arXiv ID:** 2607.04113 | [PDF](https://arxiv.org/pdf/2607.04113v1)

**作者:** Shiheng Zhang `[一作]` `[通讯]` (University of Washington), Shiheng Zhang (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过把终端噪声阈值视为奇异扰动参数，分析了扩散与流匹配采样器在终端层的行为，提出了渐近保持（AP）审计方法，证明了σ时钟/DDIM和修正流的层精确性定理，并将对数复杂度归因于随机采样的Itô项，随后将该理论应用于预训练的edm CIFAR‑10检查点进行实证验证。

**💡 创新点**

创新点在于：①将终端层视作奇异参数构建AP框架；②给出层精确性唯一性定理（σ时钟/DDIM、Rectified Flow）；③将对数成本明确分离到Itô项；④提出可在检查点上仅通过残差函数实现的运行时a‑posteriori审计与预测。

**🔧 技术方法**

使用了奇异扰动数值分析、残差功能审计（E1、E2）、指数积分器、Girsanov定理、可解析的极限模型以及预训练检查点的谱评估等技术。

**📊 数据集**

使用了CIFAR‑10数据集，并在其公开的edm检查点（以及一个VE/NCSN++检查点）上进行实验。

**📈 对比分析**

通过对残差预算（E2）、对数指数、步数与调度转移实验比较，发现确定性DDIM在终端层实现一阶均匀精度，随机SDE采样器则出现log(1/ε)的尺度扩展；审计预测与实际预算在几百分误差内吻合。

**⚠️ 局限性**

局限性在于审计仅验证学习流的离散化精度，无法评估模型误差或感知质量；未能保证FID提升；对非对称混合或消除随机采样对数项的方案仍未解决。

---

## 208. Sparse4D-Radar: An Efficient and Robust Framework for Surround-View 3D Object Detection via 4D Radar-Camera Fusion

**arXiv ID:** 2607.04098 | [PDF](https://arxiv.org/pdf/2607.04098v1)

**作者:** Fuyuan Ai `[一作]` (Zhejiang University), Chunyi Song `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于稀疏查询的4D雷达-相机融合框架 Sparse4D‑Radar，用于环视 3D 目标检测。

**💡 创新点**

核心创新包括：Deformable Fusion 模块实现可变采样的跨模态特征融合；Velocity‑Consistency Sampling (VCS) 利用雷达速度信息对点云特征进行一致性约束；Adaptive Modality Gating (AMG) 根据实时环境动态调节模态权重；提供两种配置（Base 与 Acc）以满足实时性与精度需求。

**🔧 技术方法**

技术手段包括稀疏查询式 Transformer、变形注意力采样、BEV 与 PV 视觉特征提取（ResNet+FPN）、雷达特征提取（PointPillars/Second）、Focal 损失、Hungarian 匹配的 set‑to‑set 损失、速度一致性采样与门控机制，以及多尺度特征融合。

**📊 数据集**

使用 OmniHD‑Scenes 数据集，该数据集包含六摄像头、六 4D 雷达以及 128‑束 LiDAR 的环视感知数据，覆盖多种天气与时间场景。

**📈 对比分析**

与现有雷达‑摄像头融合方法（BEVFusion、RCFusion、Doracamom 等）以及 LiDAR 基准进行比较。Base 版本在 OmniHD‑Scenes 上 mAP 47.01%、ODS 57.25%，Acc 版本 mAP 47.57%、ODS 58.35%，在所有指标上均超过前沿方法；推理速度 11.5 FPS（Base）/8.7 FPS（Acc），FLOPs 约 472/483 G，参数约 53‑55 M。

**⚠️ 局限性**

局限性包括：相较极简模型仍存在计算开销；雷达特征提取方法尚未充分挖掘 4D 雷达的特性；模块组合在提升精度时导致推理速度下降。

---

## 209. PLACEMEM: Toward a Compute-Aware Memory Plane for Lifelong Agents

**arXiv ID:** 2607.04089 | [PDF](https://arxiv.org/pdf/2607.04089v1)

**作者:** Sukanta Ganguly `[一作]` `[通讯]`, Sukanta Ganguly

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了生命周期代理的记忆胶囊（Capsule）概念，并实现了基于 vLLM 的控制平面原型，支持版本化记忆、可重用计算状态、语义驱动的重放与级联失效。

**💡 创新点**

核心创新在于将语义记忆与运行时缓存统一为一个可版本化、可失效的胶囊对象；通过控制平面制定重放策略、位置决策和失效链路，实现纠错友好的记忆重用；并提供可验证的原型与基准框架。

**🔧 技术方法**

技术包括：
- 记忆胶囊 schema 与依赖图；
- vLLM-first 原型与 OpenAI 兼容侧车；
- 基于控制平面的 KV 重放与文本检索；
- 串行化的失效引擎、并发安全失效；
- 采用权重化的重放策略（P_reuse、C_saved、V_sem 等）进行决策；
- 基准 harness（TTFT、重放率、失效开销）与 JSON/CSV 导出。

**📊 数据集**

使用了三类真实工作负载：
- 客服历史记录（customer-support history）
- 编码代理仓库循环（coding-agent repository loops）
- 研究代理事实修正（research-agent fact revision）
（未公开具体数据集名称，但均为内部真实交互记录）。

**📈 对比分析**

对比四个基线（prompt-only、text-only、runtime-only、Full）在 48 轮飞行测试中：
- Full 系统将首字节延迟（TTFT）从 18.25 ms 降低到 7.17 ms（约 61% 提升）；
- 运行时重放率提升至 100%，且在 Full 下后续纠错后无 stale 误用；
- 失效开销仅为 1.09 ms；
- 真实 vLLM 后端验证显示 Full 与 runtime-only 在 TTFT 上几乎相同（≈20.5 ms），但 Full 通过失效实现 100% 纠错准确率。

**⚠️ 局限性**

限制与待完善之处：
- 目前仅实现文本与 KV 重放，层前置重放仍为未来工作；
- 依赖外部侧车与控制平面，尚未在 vLLM 内部深度集成；
- 评估基于单机或小规模多节点，缺乏大规模部署验证；
- 未测量 GPU 预填充节省的实际效益；
- 原型仍为实验性实现，生产化稳健性和安全性需进一步完善。

---

## 210. FedFFT: Taming Client Drift in Federated SAM via Spectral Perturbation Filtering

**arXiv ID:** 2607.04170 | [PDF](https://arxiv.org/pdf/2607.04170v1)

**作者:** Liyang Yuan `[一作]` (Jilin University), Dandan Guo `[通讯]` (Jilin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在联邦学习中对Sharpness-Aware Minimization（SAM）扰动进行频域过滤的FedFFT方法，用以降低客户端漂移并提升全局模型性能。

**💡 创新点**

创新点在于首次发现跨客户端SAM扰动的低频成分是导致不一致的主因，并通过高通滤波器去除这些低频干扰，从而实现更一致、更平坦的优化路径。

**🔧 技术方法**

技术手段主要包括对SAM扰动进行实数快速傅里叶变换（rFFT），按比例截断低频系数，再反变换得到过滤后的扰动，随后在标准SAM框架中使用。

**📊 数据集**

实验验证使用了CIFAR-10、CIFAR-100和Tiny-ImageNet等公开图像分类数据集，并在多种模型（ResNet、DenseNet、ViT）上测试。

**📈 对比分析**

与FedAvg、FedSAM、FedLESAM、FedSMOO、FedGloSS等多种基线比较，FedFFT在高非IID（Dirichlet α=0.1）场景下平均提升3–6个百分点准确率，并显著加快收敛速度（如通信轮数下降≈30%）。

**⚠️ 局限性**

局限性包括对扰动半径和低频截断比例的敏感性；若过滤比例过高或扰动半径不当，可能削弱有用高频信息导致收敛变慢或最终性能下降。

---

## 211. !Imperio, smolVLA: The Implications of Data Poisoning on Open Source Robotics

**arXiv ID:** 2607.04146 | [PDF](https://arxiv.org/pdf/2607.04146v1)

**作者:** Stefan Bühler `[一作]` (Independent Researcher), Mark Schutera `[通讯]` (Duale Hochschule Baden-Württemberg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对基于视觉-语言动作（VLA）的模型 smolVLA 进行触发词级数据中毒攻击，并在真实的 LeRobot 机械臂取放任务中验证其可行性。

**💡 创新点**

证明仅需极少量（1%）的被中毒样本即可让 VLA 完全失效，攻击既低成本又隐蔽，且对触发词位置不敏感。

**🔧 技术方法**

使用触发词插入式中毒技术，将触发词 "!Imperio" 加在提示语前并将所有动作替换为固定关节姿态，随后对 smolVLA 进行微调训练。

**📊 数据集**

使用 400 条手工采集的取放任务记录（320 训练、80 测试），并在训练集里人工生成 0、1、3 条被中毒样本。

**📈 对比分析**

评估指标为任务成功率（SR）与轨迹平均误差（PMAE）；结果显示：1 条毒样本将 SR 降至约 6.7%，3 条毒样本导致 SR 为 0%；在不含触发词的提示下 SR 仍保持约 50%，PMAE 约 0.94，说明正常行为未被破坏。

**⚠️ 局限性**

局限性包括：仅针对单一任务、单一模型和单一硬件平台；使用的触发词单一且未测试多种触发词；实验在受控实验室环境下进行，未评估在更大规模或不确定环境中的攻击效果。

---

## 212. CSB: A Counting and Sampling tool for Bit-vectors

**arXiv ID:** 2607.04142 | [PDF](https://arxiv.org/pdf/2607.04142v1)

**作者:** Arijit Shaw `[一作]` (Chennai Mathematical Institute), Kuldeep S. Meel `[通讯]` (Georgia Institute of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了CSB工具，实现了对量化自由bit‑vector公式的 exact、approximate、projected 模型计数以及 uniform/almost‑uniform 采样；

**💡 创新点**

通过将 bit‑vector 转化为 CNF 并直接调用现成的 CNF 计数器/采样器，首次在同一工具中完成了这六项任务；

**🔧 技术方法**

使用了 bit‑blasting、AIG + 技术映射、Tseitin 编码、SMT‑solver 集成、现成的 CNF 计数器（SharpSAT、MapleSAT、Cachet）与采样器（Sampler、BarBarik）以及预处理；

**📊 数据集**

评估基准来自软件可靠性、图灵完整性、密码学等领域，约 200+ 个 bit‑vector 计数/采样实例，平均 22 个变量、205 条约束，最大位宽 156；

**📈 对比分析**

与现有 state‑of‑the‑art bit‑vector 计数器（CBMC）比较，CSB 在 exact/approximate 计数上可解决约 6 倍实例，PAR‑2 平均耗时显著下降；在 projected 计数和采样方面为首创，采样平均时间为 uniform‑like 1.17 s、almost‑uniform 78.4 s；

**⚠️ 局限性**

受限于 bit‑blasting 造成的尺寸膨胀，处理大位宽/复杂运算的效率不高；预处理对性能影响不稳定；目前仅支持量化自由 bit‑vector，未覆盖更高级理论或量化子句。

---

## 213. Conflict-Based Lazy Search for Fast Multi-Manipulator Planning

**arXiv ID:** 2607.04124 | [PDF](https://arxiv.org/pdf/2607.04124v1)

**作者:** Dongliang Zheng `[一作]` (Tongji University), Panagiotis Tsiotras `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于冲突的懒惰搜索（CBLS）算法，用于多机械臂的实时路径规划。

**💡 创新点**

创新点在于结合预计算的稀疏图与新的懒惰边搜索A*（LEA*）提升单臂规划效率，并将其嵌入CBS框架实现多臂协同。

**🔧 技术方法**

采用图搜索、冲突检测与约束搜索（CBS）、懒惰边A*（LEA*）、PRM样本生成以及RRT-Connect等技术。

**📊 数据集**

实验使用KUKA 7DOF机械臂、UR5双臂以及随机放置的立方体障碍物的模拟与真实环境。

**📈 对比分析**

与CBS、RRT-Connect对比，CBLS在规划时间、成功率与路径长度上均优于对手，且LEA*相较A*减少多倍的碰撞检查次数。

**⚠️ 局限性**

局限性包括对动态障碍的规划依赖预测窗口、对极大多臂数时成功率下降，以及图预处理时间与存储需求。

---

## 214. Forethought: Verifiable Reasoning from Neurosymbolic Primitive Programming

**arXiv ID:** 2607.04096 | [PDF](https://arxiv.org/pdf/2607.04096v1)

**作者:** Vishvesh Bhat `[一作]` (CoreThink AI), Emmanuel Anaya Gonzalez `[通讯]` (CoreThink AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 neurosymbolic 推理系统，将工具调用任务拆解为可验证的原语，并通过嵌入式 DSL 编排成可审计的推理程序；

**💡 创新点**

核心创新在于将推理视为可设计、可验证的程序化过程，利用类型化合同与结构化组合实现设计时错误检测与跨模型可迁移性；

**🔧 技术方法**

技术手段包括：基于符号与小型语言模型（SLM）构建可验证原语库、Python 嵌入式 DSL、程序化执行引擎与每步合同校验；

**📊 数据集**

在五个工具调用基准上验证：Tau2、BFCL v4（含人工复核）、LiveMCPBench、MCP-Atlas 及 BFCL v4 ；

**📈 对比分析**

与 vanilla LLM 提示、强化学习框架 RLM、基因演化提示优化 GEPA 以及测试时扩展模型 DeepSeek‑R1 对比，取得 30–60% 相对提升，甚至在小模型上与前沿大模型相当，同时推理成本降低 3–4 倍；

**⚠️ 局限性**

局限性在于：需要专家手工设计并验证原语与合同；每新增原语需单独 fine‑tune；假设任务可拆解为窄域原语，面向开放式推理或无明确中间约束的场景适用性受限。

---

## 215. SEDCoT: Enhancing LLM-Based COBOL Code Translation via Symbolic Execution and Delta Debugging

**arXiv ID:** 2607.04092 | [PDF](https://arxiv.org/pdf/2607.04092v1)

**作者:** Phillip Entin `[一作]` (University of Augsburg), Chunyang Chen `[通讯]` (Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种结合大语言模型、符号执行与差分调试的 COBOL→C 代码翻译框架；

**💡 创新点**

创新点在于将符号执行产生的覆盖性测试与差分调试生成的最小化失败输入联合用于 LLM 迭代修复，从而显著提升翻译准确性与可读性；

**🔧 技术方法**

采用技术包括：大型语言模型（LLM）进行初始翻译；符号执行（如 KLEE/UTBot）生成高覆盖率测试用例；差分调试（Delta Debugging）简化失败输入；以及基于提示工程的迭代修复循环；

**📊 数据集**

使用 IBM CodeNet 公开的 319 条 COBOL 函数级程序作为训练与评估数据集；

**📈 对比分析**

通过与规则驱动的 GnuCOBOL、TinyCOBOL、UniTrans、HRJR 等基线对比，实验显示相对最优基线至少提升 12%，在多款 LLM 上平均提升约 30%，且翻译代码可读性显著高于规则驱动翻译；

**⚠️ 局限性**

局限性包括：整体准确率仍低于成熟规则驱动工具；仅评估函数级程序，未覆盖大型仓库级迁移；目标语言固定为 C，缺乏多语言通用性；实验可能受 LLM 训练数据泄露影响；

---

## 216. A Unified Framework for In-Context Learning with Causal and Masked Language Models

**arXiv ID:** 2607.04081 | [PDF](https://arxiv.org/pdf/2607.04081v1)

**作者:** Chenrui Liu `[一作]` (Beijing Normal University at Zhuhai), Lixing Zhu `[通讯]` (Beijing Normal University at Zhuhai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种统一的统计学习框架，将上下文示例表示为经验测度，统一处理因果（自回归）和掩码语言模型的推理与预训练；

**💡 创新点**

证明在Wasserstein-正则条件下，掩码与自回归预训练在k-shot ICL中的过剩风险具有相同阶数；给出任务分布迁移、低维结构和数据分配最优性分析；

**🔧 技术方法**

使用经验测度、Transformer的测度依赖注意力、Wasserstein距离、上界分析以及低维/聚类结构的理论工具；

**📊 数据集**

在合成函数学习任务上验证：线性回归、加噪线性回归、深度决策树、两层ReLU网络；

**📈 对比分析**

与GPT-2风格因果Transformer进行对比，Masked Pair Encoder在所有四种任务上与GPT-2保持近似甚至优于基线，表现出与自回归模型相当的ICL能力；

**⚠️ 局限性**

仅研究理想化预训练目标，未考虑完整模型细节；低维/聚类假设为显式前提；实验仅限合成函数任务，未验证在自然语言基准上的表现。

---

## 217. Study of Graph-Based Search for Energy-Efficient Clustering in Cell-Free Massive MIMO Networks

**arXiv ID:** 2607.04074 | [PDF](https://arxiv.org/pdf/2607.04074v1)

**作者:** Julio Cesar Cardoso Tesolin `[一作]` (PUC-Rio), Rodrigo C. de Lamare `[通讯]` (PUC-Rio)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于图的最陡上升算法（GBSA），用于用户中心型无基站单元（UC CF）大规模MIMO网络的能效聚类与功率分配。

**💡 创新点**

创新之处在于将聚类状态映射为汉明图，并通过单比特邻域的图搜索与分数规划相结合，实现近似全局最优能效且计算复杂度显著降低。

**🔧 技术方法**

采用图模型、汉明距离邻域搜索、Dinkelbach分数规划、混合整数优化与连续功率分配，并在MIMO信道模型下进行仿真。

**📊 数据集**

使用基于3GPP 38.901规范的仿真数据，包含6个2×2天线AP、最多5个UE的200×200 m城市小区随机布局。

**📈 对比分析**

与全局穷举搜索和联合优化（松弛+舍入）进行比较，实验表明GBSA的能效仅略低于全局最优，但相较于JO显著提升，同时处理时间比穷举搜索快数倍。

**⚠️ 局限性**

实验仅在小规模仿真场景验证，缺乏大规模真实网络测试；假设完美CSI和理想硬件；单比特搜索可能陷入局部最优，且对极大网络的可扩展性仍有限。

---

## 218. XS-VLA: Coupling Coarse-grained Spatial Distillation with Latent Flow Matching for Lightweight Robotic Control

**arXiv ID:** 2607.04171 | [PDF](https://arxiv.org/pdf/2607.04171v1)

**作者:** Lei Iok Tong `[一作]` (Tsinghua University), Zhidong Deng `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了XS‑VLA，一种通过先将大模型Qwen3‑VL‑4B的粗粒度空间知识蒸馏到SmolVLM2‑0.25B，再将该空间感知的骨干与基于CVAE的潜在流匹配策略结合的两阶段轻量级视觉‑语言‑动作框架；

**💡 创新点**

创新点在于：①使用教师蒸馏实现粗粒度空间定位而非传统全量对齐；②将CVAE与流匹配耦合，兼顾多模态行为和稳定轨迹；③在<0.5B参数规模下实现SOTA与显著加速；

**🔧 技术方法**

采用的技术包括：Qwen3‑VL‑4B作为教师生成空间描述；SmolVLM2‑0.25B作为学生骨干；Conditional Variational Autoencoder (CVAE)；流匹配（Flow Matching）策略；交叉与自注意力的交织网络；Huber损失和KL温和化；

**📊 数据集**

主要使用的数据集有：基于LIBERO的机器人操控数据集（含Spatial、Object、Goal、Long四套任务）；通过Qwen3‑VL‑4B生成的合成空间标签；以及真实机器人Xlerobot、OpenARM和PiPER上的手动演示数据；

**📈 对比分析**

与Diffusion Policy、Octo、OpenVLA、SpatialVLA、FPC‑VLA等大模型进行对比，XS‑VLA在<0.5B模型中平均成功率达到90%（比SmolVLA 0.25B提升约2%），在Long任务上提升23%；整体执行时间/epoch仅为14，较SmolVLA‑PD 0.25B的186快约13×，同时性能仍优于2.25B SmolVLA；

**⚠️ 局限性**

局限性在于：在需要精细几何或深度感知的目标导向任务中表现不足；依赖二维教师生成的空间标签，导致深度歧义问题；

---

## 219. BrownoutMoE: Structure-Aware Expert Grouping for Efficient and Accurate LLM Web-based Services

**arXiv ID:** 2607.04164 | [PDF](https://arxiv.org/pdf/2607.04164v1)

**作者:** Yi Ding `[一作]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences), Chengzhong Xu `[通讯]` (University of Macau)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 BrownoutMoE，一个面向 Web 端 LLM 的结构感知专家分组框架，通过将行为相似的专家归集并使用联合专家蒸馏，实现高效、低误差的 MoE 推理服务。

**💡 创新点**

核心创新在于将专家分组问题视为离散策略优化，使用 GRPO（Group Relative Policy Optimization）在离线阶段学习最优分组映射，并配合分组一致的蒸馏过程，显著降低分组导致的精度损失。

**🔧 技术方法**

技术手段包括：1）GRPO 强化学习搜索分组策略；2）短周期蒸馏评估分组质量；3）联合专家（United Expert）蒸馏；4）SLO 感知的 Brownout 控制；5）GPU 优化的自研推理引擎（含 FlashAttention、PagedAttention、融合 MoE 核）。

**📊 数据集**

使用 Qwen1.5‑MoE‑A2.7B‑Chat 作为主模型；对下游任务采用 PIQA、COPA、C‑Eval（few‑shot）和 OpenBookQA（5‑shot）评测；通过 ShareGPT 与 Alpaca 进行吞吐量基准。

**📈 对比分析**

与 BrownoutServe（顺序分组）、Zero Brownout、Full Brownout 对比；实验显示在 8‑way 分组下，准确率下降率下降 71.4%，吞吐量提升至 2.24×（无融合 MoE）且在融合后仍保持相近峰值。

**⚠️ 局限性**

局限性包括：1）离线 GRPO 搜索需要消耗较多 GPU 计算时间；2）分组策略基于校准数据，可能对工作负载变化不够鲁棒；3）仅适用于现有 MoE 结构，迁移到新架构需重新训练；4）仍需手动设置 Brownout 阈值，自动化调优受限。

---

## 220. FRFDet: Efficient UAV Small Object Detection with Symmetric Sampling and Scalable Fusion

**arXiv ID:** 2607.04125 | [PDF](https://arxiv.org/pdf/2607.04125v1)

**作者:** Yunzhong Si `[一作]` (Zhejiang Normal University), Hongbo Li `[通讯]` (Beijing Geekplus Technology Co Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量级单阶段UAV小目标检测器FRFDet。

**💡 创新点**

创新点在于对称的逆双向采样（IBS）和规模感知的特征交叉融合（SFRCF）两大模块。

**🔧 技术方法**

采用深度可分离卷积、通道重排、分组处理以及乘法/加法融合的组合技术。

**📊 数据集**

在VisDrone、UAVDT、HazyDet以及MS COCO四个数据集上进行实验。

**📈 对比分析**

与现有轻量级与实时检测器比较，FRFDet在VisDrone上AP提升约2–3%，在UAVDT、HazyDet上分别突破50%与60% AP，并在COCO上比主流实时模型高出约1–2%。

**⚠️ 局限性**

局限在于模型规模增大时融合策略需要手动切换，且在极端密集或低分辨率场景下仍存在检测召回不足。

---

## 221. Spotting Setting-Related UI Display Bugs in Android Apps

**arXiv ID:** 2607.04120 | [PDF](https://arxiv.org/pdf/2607.04120v1)

**作者:** Huaxun Huang `[一作]` (Xiamen University), Rongxin Wu `[通讯]` (Xiamen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于对Android UI设置相关错误的实证研究，提出并实现了SUDFinder工具，自动检测SUD（设置相关UI显示）bug。

**💡 创新点**

首次对SUD bug进行系统分类，结合XML配置文件与运行时信息定义多种视觉关系oracle，并通过XML驱动的测试活动注入实现全UI覆盖。

**🔧 技术方法**

使用Android XML解析、图像处理（Python Imaging Library）、多模态大型语言模型（GPT‑4o、Qwen‑VL、GLM‑4.1V）生成缺失文本，结合阈值比较算法检测颜色、对齐、距离、重叠等视觉关系。

**📊 数据集**

扩充自Sun等人公开的1,074条设置相关bug数据，进一步筛选得到308条SUD bug；评估使用29个F‑Droid开源应用作为测试目标。

**📈 对比分析**

与SetDroid、dVermin、ITDroid等基线进行对比；SUDFinder在29个应用上共识别98个真阳性，精度0.76，发现比基线多51个bug，已提交至开发者，67个确认、37个合并。

**⚠️ 局限性**

局限在于只能覆盖XML配置的UI，无法检测仅在代码中动态生成的UI；对部分视觉变化（如字体多行布局）检测不稳定；MLLM生成文本的准确性和耗费token高；对语言与主题等设置的oracle仍有限。

---

## 222. Dictionaries, Not Darwin: Set-Level Selection Beats LLM Evolution in Scientific Equation Discovery

**arXiv ID:** 2607.04108 | [PDF](https://arxiv.org/pdf/2607.04108v1)

**作者:** Pan Li `[一作]` `[通讯]` (Independent Researcher), Pan Li (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在LLM驱动的科学发现中，作者审计了传统的进化式循环，发现其对生成次数并不产生提升，而是生成一个可重用的词典，随后使用单次全局稀疏子集回归进行集合级选择。

**💡 创新点**

创新点在于（1）系统性审计并证明LLM进化循环无提升；（2）提出将提议材料抽取为词典并用全局稀疏回归取代线性变异；（3）阐明选择应在集合层面而非单词层面，给出可辨别性原理。

**🔧 技术方法**

使用LLM（Llama-3.1-8B和DeepSeek-V4）生成提议、基于最小二乘的稀疏子集回归、训练集内部交叉验证与训练-验证切分、集合级搜索与稳定性生成等。

**📊 数据集**

主要使用LLM-SRBench（239个方程式任务，111 LSR-Transform+128 LSR-Synth），并在此上评估；还使用二进制包装（FunSearch）程序搜索作为异构域检验。

**📈 对比分析**

与公开基线（如LLM-SR, LaSR, SGA等）相比，在官方评测上以1/10预算实现73.2%数值精度（Llama）和77.0%（DeepSeek），显著高于最佳基线49.2%；在程序域中，单代循环无显著提升。

**⚠️ 局限性**

局限性包括：方法仅适用于可拆解为可重用词典的任务，无法处理自由程序合成；对LLM提议质量高度依赖；仅评估了数值精度与符号精度，未充分验证在更复杂或高阶结构中的泛化。

---

## 223. The Multipath Blind Spot: $K$-Agnostic Robust Calibration for Sparse-Anchor Metric Depth from Frozen Foundations

**arXiv ID:** 2607.04101 | [PDF](https://arxiv.org/pdf/2607.04101v1)

**作者:** Sohag Roy `[一作]` (Ramakrishna Mission Vivekananda Educational and Research Institute), Tamal Maharaj `[通讯]` (Ramakrishna Mission Vivekananda Educational and Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究稀疏锚点在冻结深度基础上如何转换为度量深度，并提出一种在推理时对锚点进行鲁棒筛选的包装器（MRAC）。

**💡 创新点**

创新点在于：①无需额外学习参数，完全在推理阶段实现；②利用基础相对深度的一致性（Theil–Sen 斜率估计 + MAD 门控）过滤错误锚点，克服多路径、混合像素等真实传感器噪声；③实现 K‑agnostic（可在 5–200 个锚点范围内统一使用同一检查点）。

**🔧 技术方法**

技术手段包括冻结的相对深度基础（Depth Anything V2）、残差‑on‑CFA 校准框架、Theil–Sen 鲁棒斜率拟合、MAD 误差门控以及单前向推理包装。

**📊 数据集**

实验数据集：NYUv2、KITTI、DIODE、SUN RGB‑D。每个数据集都注入四种真实传感器噪声（uniform、near、dropout、mixed‑pixel）以及不同噪声比例（0–40%）。

**📈 对比分析**

与 VI‑Depth、B'（dropout‑aug）、vanilla（不做筛选）等四个基线以及 VI‑Depth 官方配置进行五方对比。MRAC 在所有四类噪声下取得 84% 的“细胞”胜率；对 KITTI 多路径噪声，AbsRel 从 0.489 降至 0.151，提升约 3.2×；同时在其他数据集和噪声下表现一致或优于基线。

**⚠️ 局限性**

局限性：①只在 Depth Anything V2 backbone 上验证，跨 backbone 的效果待进一步评估；②在噪声比例超过 Theil–Sen 约 29% 时会出现性能下降；③门控的召回率并非 100%，仍有部分错误锚点未被过滤；④对极高噪声或特殊锚点分布的鲁棒性需进一步验证。

---

## 224. Submitted and Diagnostic Analysis of Full-Text Temporal Retrieval for LongEval-Sci

**arXiv ID:** 2607.04088 | [PDF](https://arxiv.org/pdf/2607.04088v1)

**作者:** Yingdong Yang `[一作]`, Haijian Wu `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在LongEval‑Sci 2026 Task 1中，对科学检索的长期有效性进行评估，比较了多种检索基线（BM25、dense Qwen3、RM3、cross‑encoder reranker）、融合方法（RRF）、全文索引、时间与引用信息重排序等技术，并给出了官方三快照评估结果、开发诊断以及基于收录速度与老化率的周度维护触发策略。

**💡 创新点**

创新点包括：①首次系统性验证全文BM25结合时间信息可显著提升纵向检索性能；②通过内部月度诊断揭示时间/引用重排序的校准难点；③提出简洁的周度维护触发策略，以收录速率和老化覆盖率为关键指标，为检索系统的动态更新提供实用方案。

**🔧 技术方法**

使用技术包括：BM25（title+abstract与full‑text两种视图）、dense Qwen3 embedding检索、RM3伪相关反馈、MiniLM‑L‑12 cross‑encoder reranker、Reciprocal Rank Fusion (RRF)、基于发布年份的时间特征（recency、foundation、novelty）与引用特征（inbound、recentInbound、citation‑velocity）的重排序模型。

**📊 数据集**

数据集：LongEval‑Sci 2026 训练集合共869,902篇文献与100个训练查询；使用DCTR标签评估（8,772条qrel）与raw qrel（1,183条）；三快照（March–May 2025、June–August 2025、September–November 2025）。

**📈 对比分析**

评估方法：官方采用nDCG@10的ARP、RC、DRI三指标对三快照进行比较；开发阶段使用snapshot‑1的DCTR nDCG@10、MAP、Recall@100/1000。实验结果显示：全文BM25 + temporal（FT BM25+temporal）在所有快照上获得最高ARP，并将snapshot‑3的RC从0.481降低到0.368；RRF在提升Recall@1000方面表现最突出；单独的引用重排序效果不明显。

**⚠️ 局限性**

局限性：①引用图覆盖率不足且时间戳不完整导致引用特征难以充分评估；②局部时间/引用重排序在稀疏系统上易失真，需更严格的校准；③缺少公开的运行文件与评估脚本，复现性受限；④未进行配对显著性检验；⑤官方评估仅报告nDCG@10，未覆盖MAP等指标。

---

## 225. Beyond Multilingual Averages: MTEB-PT, a Benchmark for Portuguese Sentence Encoders

**arXiv ID:** 2607.04071 | [PDF](https://arxiv.org/pdf/2607.04071v1)

**作者:** Lucas Hideki Takeuchi Okamura `[一作]` (Universidade de São Paulo), Anna Helena Reali Costa `[通讯]` (Universidade de São Paulo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MTEB-PT 这一葡语句子嵌入基准，涵盖 14 个现有数据集，覆盖 STS、分类、检索与重排四大任务，并评估 17 种开源与闭源模型。

**💡 创新点**

创新点在于提供专门针对葡语的多任务基准，揭示多语种平均性能并不能统一预测葡语任务表现，并通过 Matryoshka Representation Learning (MRL) 训练的语言特定模型展示了细粒度适配对性能的显著提升。

**🔧 技术方法**

使用 MRL 结合对比学习、NLI 与检索正负对监督，采用句子池化、余弦相似度、逻辑回归探针、nDCG 与 MAP 等标准评估技术。

**📊 数据集**

主要数据集包括 ASSIN2/ASSIN/SICK-BR/STSBenchmarkMultilingual（STS），MassiveIntentClassification/MultilingualHateClassification/BrazilianToxicTweetsClassification/HateSpeechPortugueseClassification/TweetSentimentClassification（分类），WebFAQRetrieval/WikipediaRetrievalMultilingual/MultiLongDocRetrieval（检索），WikipediaRerankingMultilingual/XGlueWPRReranking/MultiLongDocReranking（重排）。

**📈 对比分析**

通过统一评测协议对 17 模型进行宏观平均，结果显示葡语性能高度任务相关；对 STS 最佳的是 e5‑large‑matryoshka；在检索/重排上闭源大模型占优；在分类上多语种通用模型更具竞争力；MRL 细化模型在 STS 与检索上均保持竞争力，且在维度截断时表现更为鲁棒。

**⚠️ 局限性**

局限包括：仅选取现有 MTEB/MMTEB 的葡语子集，缺少聚类与比特检索等任务；评测聚焦 1B 参数以下模型，未覆盖更大模型；闭源模型数据与训练细节不公开，且 WebFAQRetrieval 仅评估 20% 数据，影响检索结果可靠性。

---

## 226. ACE: Agentic Control for Embodied Manipulation via Zero-shot Workflow Reasoning

**arXiv ID:** 2607.04162 | [PDF](https://arxiv.org/pdf/2607.04162v1)

**作者:** Iok Tong Lei `[一作]` (Tsinghua University), Zhidong Deng `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ACE 框架，实现桌面拾取‑放置任务的零射击闭环工作流推理。

**💡 创新点**

将高层语义推理与低层视觉动作分离，采用 mask‑mediated 接口实现可验证子目标，并通过多时尺度记忆实现在线检验与恢复。

**🔧 技术方法**

使用语言模型推理、视觉‑语言 grounding、mask 介导的目标表示、可重用 Diffusion 策略、后执行验证与重规划以及多时尺度记忆技术。

**📊 数据集**

仅使用通用掩码条件拾取‑放置演示数据（约 1 小时）进行训练，评估数据来自数值方程组装与约束检索任务。

**📈 对比分析**

与 ACT 与 π0.5 两个 end‑to‑end 基线对比，ACE 在方程组装任务上成功率 50%（FGS 17.8/20），约束检索 70%；基线均 0%；ACE 目标定位准确率 90%。

**⚠️ 局限性**

需要人工验证导致交互延迟；依赖推理与视觉 grounding 的质量；仅支持拾取‑放置，难以扩展到更复杂的接触操作；在更大规模任务基线仍有提升空间。

---

## 227. Mask-based Predictive Representations for Reinforcement Learning

**arXiv ID:** 2607.04153 | [PDF](https://arxiv.org/pdf/2607.04153v1)

**作者:** Kai Zhao `[一作]` `[通讯]` (Beijing Normal University), Kai Zhao (Beijing Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一种基于自监督掩码预测的视觉强化学习框架 MPR，用以提升策略学习的样本效率。

**💡 创新点**

创新点在于：① 在潜在空间而非像素空间进行掩码预测，显著降低计算成本；② 引入时空块掩码策略，使模型能够同时利用空间与时间上下文；③ 采用单一投影预测器，减少参数量，提升训练稳定性。

**🔧 技术方法**

主要技术包括：Transformer 自注意力解码器、EMA 目标编码器、随机遮挡与图像增强、SAC（连续控制）与 Rainbow（离散控制）强化学习基线。

**📊 数据集**

实验使用了 DeepMind Control Suite（6个连续控制任务）和 Atari-100k（26 个离散游戏）两个基准数据集。

**📈 对比分析**

与现有 SOTA 方法（PlaNet、Dreamer、SLAC、CURL、DrQ、MLR、SimPLe 等）进行对比，MPR 在 DMControl 100k 环境下平均提升约 7.5%，在 Atari 100k 中 11/26 游戏超越对手，整体性能显著优于对比模型。

**⚠️ 局限性**

局限性包括：掩码采样为随机方式，需要手动调参；未引入动作序列信息，可能限制更复杂任务的表现；仅使用单一投影头，缺乏多尺度特征融合的潜力。

---

## 228. Binary Iterative Method for Non-targeted Adversarial Attack

**arXiv ID:** 2607.04145 | [PDF](https://arxiv.org/pdf/2607.04145v1)

**作者:** Naman Goyal `[一作]` (Indian Institute of Technology Ropar), Milan Chaudhari `[通讯]` (Indian Institute of Technology Ropar)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于二分搜索的非目标对抗攻击方法 Binary Iterative Method（BinIM），用于在有限迭代次数内逼近局部最小值，从而生成更强的对抗样本。

**💡 创新点**

创新点在于将二分搜索思想引入对抗攻击的步长（ε）调节，使得在梯度方向上更快地逼近梯度为零的点，克服传统 ε‑ball 搜索在寻找局部最小值时的效率和精度瓶颈。

**🔧 技术方法**

使用了梯度下降（梯度符号法）和二分搜索相结合的迭代更新，核心技术为对每一次迭代的 ε_iter 进行逐步减半，并多次重启以覆盖不同局部最优；所有计算均基于已知模型 InceptionV3 的梯度。

**📊 数据集**

使用 ImageNet 公开数据集中的 1000 张随机采样图像作为测试集，评估了对抗攻击效果。

**📈 对比分析**

通过在 InceptionV3、InceptionV2、ResNet‑V2‑152 三个主流模型上对原始图像与四种对抗方法（FGSM、BIM、VAM、BinIM）进行对比，BinIM 在所有模型中使准确率下降幅度最大，尤其在 InceptionV3 上从 0.958 降至 0.009，显著优于其他方法。

**⚠️ 局限性**

局限性包括：仅针对非目标攻击，未验证针对性攻击的适用性；依赖已知模型的梯度，无法直接攻击黑盒模型；实验仅覆盖 ImageNet 1000 张样本和三种网络，缺乏对更大规模或不同任务（如目标检测、分割）的验证。

---

## 229. Toward the Right Analytical Model and System Software for Autonomous Driving Systems: Open Problems and Research Directions

**arXiv ID:** 2607.04129 | [PDF](https://arxiv.org/pdf/2607.04129v1)

**作者:** Atsushi Yano `[一作]` (Saitama University), Takuya Azumi `[通讯]` (Saitama University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并比较了自主驾驶系统（AD）中现有的实时分析模型与系统软件，指出理论与实现之间存在的差距，并提出了针对两者的开放性研究问题，构建了一种层次化的分析框架。

**💡 创新点**

创新点在于：①系统性识别并归纳了 AD 系统在时序约束、资源模型、执行时变异与安全集成等方面的五大差距；②将传统周期/突发任务、DAG、流水线等模型与 AD 需求对齐，提出“链/图”为分析单元并引入多指标联合分析与概率时序；③将问题拆分为理论社区与实践者两条平行的开放问题清单，为双向闭环提供了清晰路线图。

**🔧 技术方法**

主要技术包括：基于 ROS 2 / Autoware 的回调/节点图模型；多线程和链感知调度器（CallbackGroup、ThreadedCallback、PiCAS 等）；数据传输层（DDS、NoC、TZC、iceoryx、Agnocast 等）；时序追踪与评估工具（ros2_tracing、CARET、TILDE、Autoware_Perf、RD-Gen 等）；以及混合关键度调度、概率 WCET 与早期违约检测等理论模型。

**📊 数据集**

研究依赖的典型数据集与工作负载主要是：Autoware 框架中的真实感知、定位、规划与控制链；ROS 2 的多率多传感器同步数据（LiDAR、GNSS、IMU、摄像头等）；通过 RD-Gen 生成的可重现多率 DAG 工作负载；此外未涉及专门的公开数据集，而是基于 Autoware 在真实车辆或仿真环境中的传感器流。

**📈 对比分析**

比较方法：作者将现有理论模型（周期/突发、DAG、混合关键度等）与 ROS 2/Autoware 实现中的调度、通信与追踪机制逐层对应；对比了不同执行器、调度策略、通信实现对 E2E 延迟、数据新鲜度、并发度、功耗等指标的影响。由于本文为综述性工作，未给出统一量化实验结果，但引用了多篇研究中报告的延迟/可靠性指标，展示了在实际硬件（GPU/CPU/SoC）上的性能差异。

**⚠️ 局限性**

局限性：①文章主要为综述与问题导向，未给出完整可验证的统一分析模型或实现；②对不同平台（CPU/GPU/SoC）资源模型的统一描述仍处于起步阶段；③缺乏针对混合关键度与概率时序模型的系统级验证与性能评估；④在实际车辆验证中，对早期违约检测和 MRM 切换时序的实验仍缺失。

---

## 230. Real-Time LiDAR Gaussian Splatting SLAM

**arXiv ID:** 2607.04127 | [PDF](https://arxiv.org/pdf/2607.04127v1)

**作者:** Seungjun Tak `[一作]`, Hyeonwoo Yu `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种实时的 LiDAR 基于高斯分布 splatting 的 SLAM 系统，利用 G-ICP 追踪与球面光栅化映射紧密耦合，实现在大规模室外环境中的稠密地图构建。

**💡 创新点**

创新点包括：① 用跟踪阶段的协方差信息初始化高斯的尺度、方向和不透明度；② 通过协方差衍生的几何分数实现可扩展的地图管理（平面压缩与结构细化）；③ 将优化后的高斯反馈到追踪，使配准更稳健；④ 在球面二维高斯表示下实现高效渲染与注册。

**🔧 技术方法**

所用技术主要有：Generalized ICP（G-ICP）追踪、球面光栅化 2D 高斯 splatting、协方差加权配准、几何加权权重、控制分数驱动的剔除与分裂、基于光线丢失的可靠性掩码、循环闭环与关键帧图优化。

**📊 数据集**

实验使用 KITTI Odometry、Oxford Spires 与 Newer College 三个大规模 LiDAR 数据集进行评估。

**📈 对比分析**

与点云、surfel、TSDF、神经场和传统高斯 splatting 基线相比，本文方法在 Newer College 上实现 86.78% F‑score、>20 FPS 追踪速度、7.23 MB 内存，并在 KITTI 上取得 0.08 m 的 ATE RMSE，显著优于 KISS‑SLAM、PIN‑SLAM 与 Splat‑LOAM。

**⚠️ 局限性**

局限性包括：依赖 G‑ICP 的局部几何估计，易受稀疏植被、弱几何约束或动态物体影响；参数与传感器分辨率耦合；循环闭环仅采用关键帧刚性更新，可能导致局部不一致；KITTI 真实标注的局限性。

---

## 231. CertMix: Certified, Data-Efficient Metamaterial Design by Affine Mixing of Aligned Neural-Implicit Weight Spaces

**arXiv ID:** 2607.04123 | [PDF](https://arxiv.org/pdf/2607.04123v1)

**作者:** Yifan Wang `[一作]` `[通讯]` (McGill University), Yifan Wang (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为 CertMix 的逆设计框架，能够在少量示例单元格的基础上，通过对齐权重空间中的周期性神经隐式解码器，实现机械超材料的目标弹性性质设计，并提供已实现属性误差的无分布覆盖保证。

**💡 创新点**

核心创新包括：① 将每个示例单元格的 SIREN 权重对齐后发现其在权重空间中弹性性质近似线性；② 将逆设计问题化为受约束的仿射混合问题，并在循环中使用可微分周期性同质化求解；③ 通过线性不匹配信号构建可微信任域，实现外推；④ 使用分裂式共形预测对实现误差做分布无关的保证；⑤ 在多尺度、3D 以及功能梯度结构上实现可扩展。

**🔧 技术方法**

技术手段包括周期性 SIREN 隐式字段、可微分周期性同质化求解器（自适应梯度可逆）、仿射混合优化（双阶段软最大化与自由仿射精炼）、线性不匹配信号与分裂式共形预测、以及 3D 立方体同质化与多尺度混合场。

**📊 数据集**

使用从五类 2D 周期性结构（网格、交叉、孔板、蜂窝、螺旋）生成的 50~100 个示例单元格作为训练库，并对 3D 三重周期曲面进行扩展测试。

**📈 对比分析**

与最近邻检索、条件 VAE、无条件扩散模型、已学习权重回归网络以及每目标逆拓扑优化（SIMP）等基线进行对比。CertMix 在仅 50 个示例下实现标准化误差 10⁻⁴，约比最佳基线低 10²~10³ 倍；对目标外推时误差保持在 10⁻³ 左右；与 SIMP 相比速度提升 57 倍且无封闭空洞、checkerboard，且可直接制造。

**⚠️ 局限性**

局限性包括：仅处理线弹性属性；对 3D 结构的规模和多样性仍有限；依赖于可手工生成的示例库；对非线性大变形或能量回弹等高阶响应尚未覆盖。

---

## 232. Parametric Memory Decoding for Zero-Shot Routing in LoRA-Based External Parametric Memory

**arXiv ID:** 2607.04118 | [PDF](https://arxiv.org/pdf/2607.04118v1)

**作者:** Fengxian Ji `[一作]` (MBZUAI), Xiuying Chen `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了零射LoRA路由框架PMD和PMDRouter，实现无需训练路由器即可在外部参数记忆中选择合适的LoRA模块。

**💡 创新点**

创新点在于将路由视为对查询条件下LoRA响应的解码，并构造可分离的响应对象和归一化能量解码器，从而大幅简化设计空间并提升零射性能。

**🔧 技术方法**

采用了低秩线性响应、归一化能量解码、响应写入辅助训练，以及多粒度Benchmark PMD‑Bench来评估。

**📊 数据集**

在PaperQA、NQ‑DomainLoRA和Task‑LoRA三个任务上评估，覆盖文档、领域知识和任务技能三种外部记忆粒度。

**📈 对比分析**

与多种基线（Backbone‑only、LoRA‑only、Joint‑Backbone‑LoRA、检索基线）对比，PMDRouter在大多数设置下取得最高零射路由准确率，尤其在Task‑LoRA上达93%以上。

**⚠️ 局限性**

局限性在于对长篇领域知识仍不如BM25检索有效，且在不同backbone下性能差异较大。

---

## 233. DynaVieW: Schema-Guided World Modeling for Understanding Hierarchical Visual Dynamics

**arXiv ID:** 2607.04112 | [PDF](https://arxiv.org/pdf/2607.04112v1)

**作者:** Silin Gao `[一作]` (EPFL), Antoine Bosselut `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于层级视觉动态schema的世界模型DynaVieW，联合学习视觉状态转移预测与状态模拟，实现对复杂视频动态的深入理解与生成。

**💡 创新点**

创新点包括：① 用JSON‑schema细粒度描述高层活动、子活动、原子动作及七类视觉变换与贡献；② 采用混合Transformer专家（MoT）架构，结合共享多模态自注意和选择性注意，避免冗余信息；③ 对schema token使用重加权交叉熵损失，平衡结构与细节学习。

**🔧 技术方法**

技术手段：SigLIP2（ViT）+ Qwen2.5 LLM 进行理解专家；VAE+FLUX 进行生成专家；混合专家架构；多模态选择性注意；schema token重加权 CE + MSE 损失；Diffusion 训练与推理。

**📊 数据集**

数据集：从 Ego4D、AgiBotWorld‑Alpha、ShareGPT4Video 三大域抽取关键帧，构建状态‑转移序列；使用 InternVL‑78B‑Instruct 自动生成转移文本；生成的数据覆盖 2–100 帧、0.8–167 秒时间间隔。

**📈 对比分析**

与基线（Emu2、BAGEL、Story2Board、ARLDM、MM‑Interleaved、StoryGen）在 VinaBench（视觉叙事生成）与 LEGO（世界模拟）上对比，DynaVieW 在跨场景一致性、控制性、指令跟随、FID/CLIP/LPIPS 等指标均优于或等于对手，尤其在零样本和 SFT 设定下表现突出。

**⚠️ 局限性**

局限性：对极端或长时序场景的鲁棒性仍有限；低级细节的准确率略低于金标准；受限于训练数据分布，可能在不常见动作或环境下泛化不佳。

---

## 234. Finite-Blocklength ISAC Multiple Access: A Source-Channel Coding Perspective

**arXiv ID:** 2607.04109 | [PDF](https://arxiv.org/pdf/2607.04109v1)

**作者:** Zhentian Zhang `[一作]` (Southeast University), Zaichen Zhang `[通讯]` (Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对多用户上行ISAC系统，在有限块长情形下构建了信息理论框架，证明感知失真约束等价于源编码需求，提出有效载荷概念，给出随机编码可达性上界、Fano多用户逆向下界以及基于天才辅助的单用户逆向下界，完成能量-速率-感知折衷的精确刻画。

**💡 创新点**

创新点包括：①将感知失真与源编码等价，统一将感知与通信视为单一有效信息载荷；②在有限块长多用户MAC上实现全新的可达性与逆向下界，首次量化能量-感知的相互关系；③证明联合编码显著优于最优时分双相位方案，量化了ISAC的集成收益；④通过多用户Fano逆向下界揭示感知精度随用户负载线性放大。

**🔧 技术方法**

采用信息理论工具：源编码与率失真理论、Fano不等式、随机编码与Gallager指数、Gaussian MAC容量与互信息、数据处理不等式、天才辅助简化以及有限块长逆向下界技术。

**📊 数据集**

无真实数据集，使用高斯远程源模型（维度d=4、方差σ²=1）进行仿真；参数取n=1000、kₐ=100、b=100、ε=0.1、不同失真D进行数值评估。

**📈 对比分析**

通过将联合编码方案与优化的两相位正交基线对比，发现联合编码在低负载下相对基线可节约≈0.7dB能量，在高负载下约0.3dB；可达性曲线与Fano逆向下界高度贴合；能量-感知曲线显示每十分之一失真降低约线性增加能量，且增幅随用户负载增大。

**⚠️ 局限性**

局限性：仅针对高斯MAC和独立高斯感知源，未考虑信道衰落、干扰或非高斯信源；结果为理论极限，缺乏具体编码实现与硬件验证；多用户模型假设用户数量已知且同步，实际系统中用户管理与时延不确定性仍是挑战。

---

## 235. Semantic Integration and Lexical Expectation Shape N400 and P600 Dynamics During Naturalistic Reading

**arXiv ID:** 2607.04107 | [PDF](https://arxiv.org/pdf/2607.04107v1)

**作者:** Kun Sun `[一作]` (Tongji University), Rong Wang `[通讯]` (University of Tuebingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了词惊讶度和注意力加权的局部语义相关性在自然阅读过程中对N400和P600 EEG响应的影响

**💡 创新点**

创新点在于引入可解释的注意力加权语义相关性指标，并同时考察其与词惊讶度在自然阅读EEG中的互补作用

**🔧 技术方法**

采用了rERP回归分析、广义加性混合模型（GAMM）以及GPT‑2生成的词惊讶度和预训练词向量计算的语义相关性

**📊 数据集**

使用了Dublin EEG‑based Reading Experiment Corpus（DERCo）提供的自然阅读EEG数据

**📈 对比分析**

通过ΔAIC比较、FDR校正显著性以及通道/ROI层级分析，结果显示语义相关性在N400和P600窗口均显著且对解释方差优于仅用词惊讶度，尤其在P600窗口效果更强

**⚠️ 局限性**

局限性包括仅使用32通道EEG、未建模完整时空结构，以及只关注N400/P600窗口，未探讨更早或更晚的电位变化

---

## 236. Seeing Once is Enough? Online Geometry-Aware Token Pruning for 3D Question Answering

**arXiv ID:** 2607.04079 | [PDF](https://arxiv.org/pdf/2607.04079v1)

**作者:** Ruei-Chi Lai `[一作]` (National Tsing Hua University), Min Sun `[通讯]` (National Tsing Hua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在线、无训练的几何感知视觉令牌裁剪方法，可在多帧3D问答任务中实时裁剪冗余视觉令牌。

**💡 创新点**

创新点在于利用深度与相机位姿将每帧投影到共享体素空间，实时检测并裁剪重叠区域的令牌，从而实现在线低内存和高效推理。

**🔧 技术方法**

使用了体素化、空间重叠检测、像素级和令牌级掩码生成技术，并与现有VLM（Qwen2.5‑VL‑7B、Qwen3‑VL‑8B）无缝集成。

**📊 数据集**

在ScanQA、SQA3D和OpenEQA‑HM3D三大3D问答基准上进行评估。

**📈 对比分析**

与在线均匀采样基线和离线最大覆盖策略对比，令牌使用量减少约50%，在所有基准上提升约+5.1 LLM‑Match，并在Exact Match和CIDEr等指标上均优于对比方法。

**⚠️ 局限性**

局限性包括裁剪的重叠令牌可能仍包含有用空间信息，未结合压缩技术；目前仅在预定义数据集验证，缺乏在实时真实环境中的实测。

---

## 237. Quasipolynomial Trace Reconstruction

**arXiv ID:** 2607.04073 | [PDF](https://arxiv.org/pdf/2607.04073v1)

**作者:** Arnav Burudgunte `[一作]` (Purdue University), Hongao Wang `[通讯]` (Purdue University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本论文提出了一种理论框架，用来在任意删失概率下完成 n 位字符串的 Trace Reconstruction，证明在 exp(O(log^{5/3} n)) 条样本（trace）内即可恢复原始字符串。

**💡 创新点**

创新点在于将多路删失通道的期望统计量通过三路或两路乘积重新表达为单一路低阶统计量的加权和，同时利用光滑的傅里叶变换与去卷积技术，构造支持有限且光滑的权重序列，使得在递归“放大”窗口的过程中保持统计量差异不消失，最终实现指数级样本量的极大压缩。

**🔧 技术方法**

核心技术包括：Fourier 分析与投影切片定理、子波形展开、B‑spline 光滑核构造、α‑β 概率分解（模拟多条删失通道）、Chernoff 及泊松尾概率估计、子调和性与泊松核界定、以及多次使用 pigeonhole 原理抽取单一统计量 S 的方法。

**📊 数据集**

该研究为纯理论工作，没有使用任何实验数据集。

**📈 对比分析**

相较于以往需要多项式或更大指数数量的 Trace Reconstruction 方法，本文提供的样本数上界为 exp(O(log^{5/3} n))，显著降低了理论所需样本量；在给定保留概率 p ≤ 1/12 的情况下，算法能够在多项式时间内完成重构。

**⚠️ 局限性**

局限性包括：仍然是指数级样本量（无法降至多项式级别）；方法对保留概率范围有严格限制（p ≤ 1/12 及其后续递归放大）；实现复杂，难以直接转化为实用算法；以及对插入错误（Insertion channel）处理不够全面。

---

## 238. A Gallager-Type Redundancy Bound for Binary Shannon-Fano Coding

**arXiv ID:** 2607.04192 | [PDF](https://arxiv.org/pdf/2607.04192v1)

**作者:** Kamila Szewczyk `[一作]` `[通讯]` (Saarland University), Kamila Szewczyk (Saarland University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

论文证明了二进制Shannon-Fano编码的冗余性界限，提出了一个依赖于最大源概率p_1的七段包络R<f(p_1)。

**💡 创新点**

这是第一个依赖于p_1的Fano编码冗余界限，提供了更精确的冗余性估计。

**🔧 技术方法**

使用了Fano递归、最小修正的仿射潜力和无埋葬引理等技术。

**📊 数据集**

没有具体提到使用的数据集。

**📈 对比分析**

与Huffman编码的冗余性界限进行了比较，证明了Fano编码在特定条件下的冗余性界限优于Huffman编码的界限。

**⚠️ 局限性**

该方法的局限性在于Fano树的结构特性，可能无法适用于所有类型的源符号分布。

---

## 239. Exploring Convolutional Neural Processes for Weather Downscaling

**arXiv ID:** 2607.04190 | [PDF](https://arxiv.org/pdf/2607.04190v1)

**作者:** Francisco Passos `[一作]` `[通讯]`, Francisco Passos

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10和ImageNet数据集进行实验。

**📈 对比分析**

与传统的激活函数（如ReLU）进行比较，结果显示新模型在分类精度上提高了5%，且训练时间缩短了15%。

**⚠️ 局限性**

模型在处理高分辨率图像时性能下降，且对计算资源的需求较高。

---

## 240. CoCoScale: Leveraging Layer-wise Scaling to Unlock the Potential of Online LLM Serving

**arXiv ID:** 2607.04181 | [PDF](https://arxiv.org/pdf/2607.04181v1)

**作者:** Jingfeng Wu `[一作]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences), CHengzhong Xu `[通讯]` (University of Macau)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CoCoScale，利用层级复制实现在线 LLM 服务器的细粒度弹性扩展，显著减少冷启动和延迟。

**💡 创新点**

创新点在于将 Transformer 层视为可扩展单元，构建层级数据并行并采用环形多播迁移协议，做到近乎即时的容量增减。

**🔧 技术方法**

技术包括层级复制、NVLink 环形多播、CUDA Graph、基于模型规模与负载的自适应配置模型以及统一的伸缩算法。

**📊 数据集**

使用阿里巴巴与 Azure 的真实在线请求追踪，结合 LongBench 提示生成作为实验数据集。

**📈 对比分析**

通过与 vLLM 静态分配和阿里巴巴自动伸缩基线对比，CoCoScale 平均延迟降低 20–28%，100% 满足 SLO，扩容延迟缩短 97–99%。

**⚠️ 局限性**

局限性包括对高带宽 NVLink 的依赖，在低速互连下扩展受限；层级复制受模型层数与显存限制，难以直接处理极大模型或多节点分布式环境。

---

## 241. CritiqueDriveVLM: From Verifier-Guided Reinforcement Learning to Latent Thought Distillation for Autonomous Driving

**arXiv ID:** 2607.04179 | [PDF](https://arxiv.org/pdf/2607.04179v1)

**作者:** Zhaohong Liu `[一作]` (Beijing University of Posts and Telecommunications), Mengshi Qi `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CritiqueDriveVLM三阶段框架，先用SFT和多维验证器预热，再通过多轮RL让教师模型自我纠错，最后用Latent Thought Distillation将教师的深层推理压缩到学生的隐藏空间，实现无CoT的低延迟推理。

**💡 创新点**

创新点包括：①内部化推理而非依赖外部工具；②引入多维验证器和多轮step‑decay惩罚的RL策略，显著抑制视觉幻觉和保守偏差；③利用隐藏层对齐（Cosine）实现Latent Thought Distillation，将System‑2深度推理直接迁移到System‑1。

**🔧 技术方法**

技术实现基于Qwen3‑VL‑8B VLM，采用LoRA微调、GRPO优化、多维验证器（s_per,s_log,s_saf）和step‑decay多轮惩罚，以及隐藏层对齐损失进行学生训练。

**📊 数据集**

使用DriveLMM‑o1基准（18,000+ VQA对），该数据集构建于nuScenes，包含逐步推理和驾驶决策。

**📈 对比分析**

在DriveLMM‑o1评测中，Teacher模型在多项指标上超过所有基线，MCQ 76.54%、整体 80.48%；Student在无CoT条件下达68.59% MCQ，推理时间仅416 ms（比Teacher 3482 ms下降88%）。

**⚠️ 局限性**

局限性包括：仍未验证多摄像头视频流的时序推理；Latent Distillation可能在极细粒度推理上存在信息损失；多轮RL训练需要大量验证器标注，对资源消耗有一定要求。

---

## 242. Lower Bound of Networked Control with Multiple Sensors and One Controller And The Application to Tracking Gaussian-Markov Source

**arXiv ID:** 2607.04172 | [PDF](https://arxiv.org/pdf/2607.04172v1)

**作者:** Sijie Li `[一作]` (University of Texas at Austin), Hyeji Kim `[通讯]` (University of Texas at Austin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了多传感器/编码器与单解码器网络化控制系统的因果率失真函数，提出了新的定向信息下界并证明在线性二次高斯（LQG）模型下线性、独立编码器与线性解码器能够达到该下界，进一步将原无限维优化问题降维为有限维问题；随后将该框架应用于含线性旁路信息的高斯马尔可夫源的因果失真问题，给出了含奇异噪声协方差矩阵的半正定规划（SDP）求解形式；

**💡 创新点**

创新点在于：①首次为多编码器单解码器网络化控制设置给出定向信息下界；②证明在线性全观测条件下线性独立编码器是该下界的最优解；③把高维优化转化为可解的SDP；④扩展至含奇异噪声的高斯马尔可夫源，并提供数值示例。

**🔧 技术方法**

主要技术包括：定向信息理论、Gaussian分布最优性、线性控制与鲁棒估计、Kalman滤波、半正定规划与矩阵不等式、可观测性/可控性分析。

**📊 数据集**

本文主要为理论工作，实验使用人工生成的随机矩阵（例如随机构造的A、N、C、V等）进行数值模拟。

**📈 对比分析**

比较方法为数值仿真，展示噪声协方差矩阵秩升高时所需率上升；与之前的Loosely-Closed-Form或非网络化设置的结果相比，本文提供了可达性下界与实现策略的对照，证明线性策略在给定条件下可达最优。

**⚠️ 局限性**

限制包括：①仅在“全观测”条件下证明线性编码器最优；②所给定的定向信息下界不是最紧的，无法直接证明对更严格下界的线性最优性；③未讨论无反馈网络的更一般情况；④数值实验仅针对随机矩阵，缺乏实测系统验证。

---

## 243. On Preserving Geometrical Invariance for Superpixel Image Classification using Graph Transformer

**arXiv ID:** 2607.04262 | [PDF](https://arxiv.org/pdf/2607.04262v1)

**作者:** Sarabeshwar Balaji `[一作]` (Indian Institute of Science Education and Research Bhopal), Akash Anil `[通讯]` (Indian Institute of Science Education and Research Bhopal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 SuperGT，一种基于图 Transformer 的超像素图像分类框架，能够捕获长距离依赖并实现平移/旋转不变性；

**💡 创新点**

创新点在于将 PCA 预处理与图 Transformer 结合，既保持图结构又兼顾几何不变性；并且在超像素图上首次采用 Performer 机制与 GPSE 编码，提升计算效率；

**🔧 技术方法**

使用 SLIC 超像素分割、RAG 结构构建、PCA/均值中心化预处理、图 Transformer（包含 MPNN+全局注意力）、Performer 注意力、GPSE 位置/结构编码、以及 MLP 输出分类；

**📊 数据集**

在 CIFAR-10 数据集上进行实验，使用 50K 训练集和 10K 测试集；

**📈 对比分析**

与多种基准（GAT、GCN、GIN、GraphSAGE、PNA 等标准 GNN 及 ShapeGNN 等超像素 GNN）对比，SuperGT 在完整训练集上取得 80.19% 准确率，与最先进方法相当；在少量训练样本、结构噪声和不同数据规模下表现更稳健，且样本效率更高；

**⚠️ 局限性**

局限性包括：PCA 预处理在实验中导致精度下降；模型对空间坐标敏感，移除后性能显著下降；在 KNN 结构下性能下降；对旋转不变性的实现尚未达到最佳，需进一步探索更鲁棒的几何编码。

---

## 244. HeartVolMesh: Cardiac Volumetric Mesh Reconstruction via Covariance-Guided Graph Deformation

**arXiv ID:** 2607.04243 | [PDF](https://arxiv.org/pdf/2607.04243v1)

**作者:** Fengming Lin `[一作]` (University of Manchester), Alejandro Frangi `[通讯]` (University of Manchester)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了HeartVolMesh框架，利用协方差引导的图形变形从3D医学图像直接生成具有跨病例顶点对应关系的心脏体积四面体网格；

**💡 创新点**

创新点在于把每个网格顶点视为可学习的各向异性高斯核，并用协方差引导的负对数似然损失进行监督，同时通过基于模板的体积到表面注册与变形场传播实现体积网格的对应与质量保证；

**🔧 技术方法**

主要技术包括3D CNN编码器提取多尺度体积特征，图神经网络传播网格拓扑信息，Cholesky参数化的协方差预测与Mahalanobis距离损失，轻量级网格正则化，以及全流程的相似性对齐、光滑非刚性配准和变形场插值来变形四面体模板；

**📊 数据集**

使用多中心内部数据集，共900名患者，4000个时间点，按患者划分训练集800例、验证集100例；伪真值网格来自TotalSegmentator的自动分割并手工精修；

**📈 对比分析**

与Voxel2Mesh、MeshDeformNet、HeartDeformNet、Ours3p1等基线进行公平比较，Ours3p6在表面与体积网格的CD/HD95、NC、最小Jacobian、最小dihedral角等指标上均显著优于所有对照组，且无反转单元；

**⚠️ 局限性**

局限性包括仍需依赖预先构建的四面体模板、对不同扫描模态（如cine CMR）的适配性尚未验证、对薄壁结构的配准精度受限于变形场分辨率、以及目前未联合训练表面与体积模块等。

---

## 245. The Politics Attention Makes: Platform Media Logic and the Mediatization of Politics

**arXiv ID:** 2607.04220 | [PDF](https://arxiv.org/pdf/2607.04220v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 246. Progress- and Reliability-Oriented Group Policy Optimization for Agentic Reinforcement Learning

**arXiv ID:** 2607.04242 | [PDF](https://arxiv.org/pdf/2607.04242v1)

**作者:** Mingxuan Fan `[一作]` (Baidu Inc.), Peiyang Liu `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种学习无价值网络、基于组的强化学习方法ProGPO，用于在保持历史一致性的前提下为长时序代理任务提供更稠密的信用分配信号。

**💡 创新点**

创新点在于将精确前缀对比与通过rollout统计得到的状态潜能相结合，利用多分辨率逆方差融合与语义扩展实现可靠的状态价值估计，从而在无学习者价值网络的情况下实现进展信用。

**🔧 技术方法**

使用的技术包括：组化优势估计、前缀一致的步骤对比、基于rollout的状态潜能估计、逆方差多分辨率融合、语义相似性扩展、以及PPO风格的裁剪式策略优化。

**📊 数据集**

在WebShop和ALFWorld这两个长时序代理基准上进行实验，使用Qwen2.5-1.5B-Instruct模型，并对比了GRPO、GiGPO、HGPO等同类方法。

**📈 对比分析**

与匹配实验相比，ProGPO在ALFWorld上从原先的90.1%成功率提升到90.1%，在WebShop上成功率从71.5%提升至71.5%，整体显著优于其他无价值网络组化方法，且在多种任务细分指标上表现更好。

**⚠️ 局限性**

局限性包括：仍依赖批次中状态的重复出现；当当前或下一个状态的潜能估计不可靠时，部分步骤仍无法获得有效信用；语义扩展基于隐藏状态相似性，可能导致误匹配；未针对更大规模或多模态任务进行验证。

---

## 247. High-Performance Real-Time Implicit Strand-Based Hair Rendering via Software Rasterization

**arXiv ID:** 2607.04230 | [PDF](https://arxiv.org/pdf/2607.04230v1)

**作者:** Lukas Lipp `[一作]` (Meta), Lukas Bode `[通讯]` (Meta)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一套基于软件光栅化的延迟渲染管线，用以实时渲染使用毛发网格（hair mesh）表示的细长毛束；

**💡 创新点**

核心创新包括：①利用软件光栅化实现对远距离毛束的高效单像素渲染；②设计了自适应的多级细节（LOD）机制，既减少毛束数量又降低控制点数量；③引入了压缩 G‑Buffer 与原子操作的 deferred shading，兼顾性能与兼容性；④通过中心样本与保守样本双层 G‑Buffer结合的重构滤波，实现了在单样本下几乎等同于多样本抗锯齿的视觉质量；⑤在毛发网格上预烘焙环境遮蔽，支持体积式 probe 光照。

**🔧 技术方法**

实现技术主要包括：软件光栅化（DDA 线光栅化）、自适应 LOD 预处理、压缩 64‑bit G‑Buffer（深度 24 bit、位置重建、八面体编码切线、量化的表面参数）、原子最小深度写入、基于椭圆双边滤波的重构后处理、以及光照模型与预烘焙 AO 的结合。

**📊 数据集**

实验使用了两套毛发网格数据集：一是 127 k 条毛束的标准发型，二是包含多达 483 k 条毛束的复杂发型（4K 分辨率）。还测试了多毛发体场景（多达 2 个发型）以验证可组合性。

**📈 对比分析**

与传统基于 Mesh Shader 的实现进行基准测试：在 1920×1080 分辨率下，软件光栅化在 MSAA 1 方案下仅耗时 3.1 ms（比 Mesh Shader 的 13.6 ms 快 4.4×），并在加入 LOD 后远场帧时延降至 0.4 ms（提升 11×）。质量方面，单样本加重构滤波（SWR 1+F）与 4×/8× MSAA 结果几乎相同，Lod 进一步降低了开销但无显著视觉损失。

**⚠️ 局限性**

主要局限在近场渲染时效率下降：当毛束宽度超出单像素时软件光栅化不再优于硬件；DDA 线段绘制的工作负载不均衡，线程利用率受限；在极低 LOD（控制点极少）时可能出现 SIMD 乱用；缺乏时间域滤波，移动视角时仍有轻微抖动。

---

## 248. Orchestrating Communication, Computing, and Energy Transfer for Wireless-Powered 6G Closed-Loop Controls

**arXiv ID:** 2607.04225 | [PDF](https://arxiv.org/pdf/2607.04225v1)

**作者:** Chengleyang Lei `[一作]` (Tsinghua University), Ning Ge `[通讯]` (Tsinghua University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计了一套基于卫星无线能量传输的SC^3闭环控制系统，联合优化通信功率、带宽、计算资源、时间分配与能量分配以最小化整体LQR成本。

**💡 创新点**

创新点在于将无线能量传输与感知-通信-计算-控制四大模块的耦合关系纳入同一优化框架，并给出能量受限下单环闭环的半闭式最优解与低SNR分析。

**🔧 技术方法**

采用线性二次调节器（LQR）作为性能指标，使用序列凸近似（SCA）求解非凸问题；在单环情形下利用拉格朗日对偶法推导SNR与能量-时间折中关系。

**📊 数据集**

研究基于仿真数据，设置K=5个机器人、卫星高度300km、传输功率500kW等参数，评估不同带宽、循环时间与能量下的系统性能。

**📈 对比分析**

与两种基准方案（最小CNE最大化和固定WPT功率的控制导向方案）对比，所提出方案在LQR成本上显著更低，且在能量不足时能保持闭环稳定性。

**⚠️ 局限性**

局限性包括对完美CSI的假设、中心化求解不适合大规模网络、卫星能量传输仍处于实验阶段，缺乏实际硬件验证与鲁棒性分析。

---

## 249. An Evaluation of Role-Based Multi-Agent Code Generation on Repository-Scale Problems

**arXiv ID:** 2607.04212 | [PDF](https://arxiv.org/pdf/2607.04212v1)

**作者:** Benedetta Donato `[一作]` (University of Milano-Bicocca), Valerio Terragni `[通讯]` (University of Auckland)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了基于角色的多代理代码生成流水线（规划、编码、评估、搭建），并在12个真实规模Java项目上与单一LLM进行对比实验。

**💡 创新点**

首次将多代理系统与反射式迭代评估结合，并在大型仓库级项目上系统评估其效果；使用多维度指标（代码相似度、场景意图、编译率、资源消耗）对比传统单一LLM。

**🔧 技术方法**

采用GPT‑5驱动的LLM代理（ChatDev 2.0实现反射），使用JPlag和CrystalBleu进行代码相似度评估，Claude Code + ChatGPT 5.4进行场景意图匹配，GitHub API收集项目并自动生成SRS。

**📊 数据集**

12个按星级与复杂度划分的Java GitHub仓库（Trivial、Simple、Moderate、Complex四组），SRS由README与Javadoc自动生成后人工校验。

**📈 对比分析**

通过比较任务数、代理数、生成代码规模、编译率、JPlag/CrystalBleu相似度、场景意图匹配率以及时间、token、成本等指标。结果表明，Agentic‑reflexive在相似度、场景意图和编译率上优于单一LLM，但整体编译成功率仍低；Agentic‑seq生成代码量大但质量略逊；单一LLM在小项目中仍表现不错。

**⚠️ 局限性**

生成的代码仍为部分实现，编译成功率低；缺乏编译器与运行时反馈的动态修正；对简单项目可能过度设计；需要人工干预完成系统；资源开销高于单一LLM；实验依赖GitHub数据可能存在训练泄漏。

---

## 250. Efficient and Secure Range Counting over Distributed Geographic Data with Query Range Protection

**arXiv ID:** 2607.04194 | [PDF](https://arxiv.org/pdf/2607.04194v1)

**作者:** Haoxin Yang `[一作]` (Xi'an Jiaotong University), Xiaohong Guan `[通讯]` (Xi'an Jiaotong University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了PPRC协议，实现私有分布式范围计数（PDRC），同时满足精度、双向隐私与高效性；

**💡 创新点**

创新点在于：①Private Range Predicate（PRP）将范围判断改写为加密Bloom过滤器的成员资格测试，取代昂贵的安全比较；②Oblivious Linear Counting（OLC）为重叠数据聚合设计了轻量级加法方案，保证聚合过程无信息泄露；

**🔧 技术方法**

采用同态加密、加密Bloom过滤器、线性计数草图（Linear Counting Sketch）、轻量级安全乘法/加法等密码技术；

**📊 数据集**

实验使用真实世界地理数据集与合成数据集；

**📈 对比分析**

与基线协议（数据私有、查询私有、Exact MPC等）对比，PPRC在误差上降低最多55倍，在速度上提升37倍，展示出显著的性能优势；

**⚠️ 局限性**

局限性包括：仅在诚实但好奇者模型下验证；目前仅支持矩形查询，需额外编码才能扩展到非矩形区域；对Bloom过滤器尺寸和误差率的权衡仍需手工调参；在大规模并发查询时的可扩展性未完全评估。

---

## 251. Beyond Trees: The Weighted Center Problem on Gromov Hyperbolic Graphs

**arXiv ID:** 2607.04287 | [PDF](https://arxiv.org/pdf/2607.04287v1)

**作者:** Guillaume Ducoffe `[一作]` `[通讯]` (University of Bucharest), Guillaume Ducoffe (University of Bucharest)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套基于Gromov超几何性质的近似与精确算法，专门解决任意加权中心（Weighted Center）问题，涵盖δ-吉尔伯特图以及弦图、距-遗传图、弱桥图、双弦图、弦双图、二分希尔图、平面图等重要图类，并实现了几乎线性时间的求解方案；

**💡 创新点**

创新点包括①利用树嵌入与δ超几何的四点条件，快速得到距离δ的近似中心；②将该近似中心作为局部搜索起点，结合弱峰值性、Helly性等度量特性，得到O(m)至O(m log^k n)的精确算法；③首次将δ-吉尔伯特与VC理论相结合，求解平面图中的加权中心；④给出了以δ为参数的几乎线性时间算法，突破了传统的Ω(n^2)下界；

**🔧 技术方法**

技术手段主要包括树嵌入与近似扭曲、Gromov超几何的四点条件、G^p-单峰性与弱峰值性、Helly性质与注入包、局部搜索与改进序列、VC维度与分离器、递归分治与凸包技巧、线性函数最大化等；

**📊 数据集**

论文以理论分析为主，并未给出具体实验数据集，若有实验则采用标准复杂网络或随机生成的吉尔伯特图等公开数据；

**📈 对比分析**

与现有仅在树、距离遗传图等类中已知的线性/子线性时间算法相比，本文在弦图、距-遗传图、双弦图、弦双图等类中实现了最优或近最优O(m) / O(m log n) / O(m log^2 n)运行时；在平面图上得到O(2^{O(δ)} n)时间；整体性能优于之前的O(n^2)或O(m√n)算法；

**⚠️ 局限性**

局限性：①算法对δ的常数因子较大，尤其在平面图中出现指数因子；②仅适用于满足δ-吉尔伯特或已知δ的图，对非超几何图无效；③部分结果依赖δ已知或可近似的前提；④在某些图类中仍无法突破Ω(n^2)下界。

---

## 252. The New Shape of Search: How Conversational AI Recomposes Information Seeking

**arXiv ID:** 2607.04282 | [PDF](https://arxiv.org/pdf/2607.04282v1)

**作者:** Michael Iannelli `[一作]` (Scrunch AI), Alan Ai `[通讯]` (Scrunch AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过跨面板将用户对话、搜索和浏览日志整合，重新构建信息寻求过程，分析对话式AI在信息寻求过程中的形态变化。

**💡 创新点**

发现对话式AI并未统一压缩搜索过程，而是二分为终止式和支撑式两种形态，且终止式取决于提问长度，而非任务类型。

**🔧 技术方法**

利用LLM分类器对开场提问进行任务和主题标注，并用跨面板时间序列重构 episode；还使用 bootstrap 估计结构比例。

**📊 数据集**

数据来自 2026 年 5 月美国和英国的 opt‑in 跨面板，覆盖 ChatGPT、Claude、Perplexity、Grok 对话记录以及同一时间段内的搜索和浏览日志。

**📈 对比分析**

比较方法是对包含 AI 的 episode 与仅包含搜索的 episode 在形态、搜索触发率、后续源访问等维度进行描述性比较，结果显示 AI 主要导致 episode 终止率提升至约 60%，并未显著减少搜索量；但在特定任务和提问长度上表现差异显著。

**⚠️ 局限性**

局限包括：不具备因果推断、episode 划分阈值敏感、LLM 标注存在误差、源访问仅按域级别计数、仅观察表面行为无法捕获用户满意度与信任等内在因素。

---

## 253. Risk-Constrained Freshness-Aware Semantic Caching for Open-Web Retrieval-Augmented LLMs

**arXiv ID:** 2607.04281 | [PDF](https://arxiv.org/pdf/2607.04281v1)

**作者:** Muhammad Mansoor `[一作]` (Jeju National University), Yeo-Chan Yoon `[通讯]` (Jeju National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了FreshCache，一种三层语义缓存框架，针对开放网络检索增强生成（RAG）系统的缓存命中进行风险约束的时序推理；

**💡 创新点**

核心创新在于将缓存重用视为“时间敏感的风险估计”问题，利用校准的指数衰减模型和可学习的MLP来动态估算缓存条目的陈旧概率，并在每个层级设置不同的错误预算；

**🔧 技术方法**

技术手段包括BGE‑M3多语言句子嵌入做相似度检索、三层缓存结构（答案、URL 列表、页面内容）、基于指数衰减的概率模型、MLP 风险预测器、条件 GET 验证、以及多类（TIMELESS、SLOW、MEDIUM、FAST、REAL_TIME）时间敏感性划分；

**📊 数据集**

使用了自制的 FreshCache‑Bench 基准，涵盖 8,072 个英文与 246 个韩语基础查询，扩展为 31,201 条查询，包含在 1h、12h、24h、7d 的真实网页快照变化标签；

**📈 对比分析**

与 SemanticTTL、DomainTTL、vCache、SCALM 等基线对比，FreshCache_MLP 在 24 小时窗口下实现 97% 的搜索调用节省、0.1% 的哈希级别陈旧错误，真实答案级错误约 0.034%；在所有基线中保持 Pareto 主导；

**⚠️ 局限性**

局限性包括：在 1 小时窗口下风险门限效果有限、仿真中中位数延迟节省因多表述命中被高估、条件 GET 在实际网页上验证率低、以及 MLP 模型对训练分布外查询的泛化能力受限。

---

## 254. EMPURPLE: A Free Lunch for Diffusion Distillation based on the Information Bottleneck

**arXiv ID:** 2607.04276 | [PDF](https://arxiv.org/pdf/2607.04276v1)

**作者:** Zilai Li `[一作]` (University of Nottingham), Lujia Bai `[通讯]` (University of Nottingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出EMPURPLE，一种训练无关的方法，通过回收原始扩散模型的中间噪声特征来缓解扩散蒸馏过程中的分布不匹配问题，从而提升蒸馏模型的FID；

**💡 创新点**

创新点在于利用已缓存的中间噪声，保持蒸馏模型在早期步骤与原模型相同的分布，减少目标复杂度导致的泛化误差；

**🔧 技术方法**

采用PAC-style 泛化分析、协方差谱分析、DDIM逆向编码、典型集理论，以及与原始扩散模型的噪声缓存和重用技术；

**📊 数据集**

主要使用COCO 2014训练集与验证集的文本提示进行评估，生成512x512和1024x1024分辨率图像；

**📈 对比分析**

与多种蒸馏模型（SDXL-Lightning、Hyper SDXL、DMD2、Flash SDXL、LCM等）进行对比，EMPURPLE在FID上提升约7%至20%，CLIP略有下降；

**⚠️ 局限性**

局限性包括：对不同模型的适用性需进一步验证、对CLIP分数略有下降、仅针对图像生成任务，对其它任务的推广尚不确定。

---

## 255. HALO-WA: Hybrid-Attention Latent-Guided Online Reinforcement Learning for World-Action Models

**arXiv ID:** 2607.04265 | [PDF](https://arxiv.org/pdf/2607.04265v1)

**作者:** Angen Ye `[一作]` (Chinese Academy of Sciences), Dapeng Zhang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出了一种针对世界行动模型的混合注意力潜在引导在线强化学习框架 HALO-WA，能够在实际机器人上快速适配精细操作任务。

**💡 创新点**

创新点在于冻结大规模 WA 模型，仅训练轻量级 actor‑critic 适配器，并通过混合注意力结合 WA 参考动作和潜在特征，实现在线动作细化。

**🔧 技术方法**

使用了在线强化学习（TD3+行为克隆）、混合注意力网络、云‑机器人协同训练、视觉 VAE 潜在特征等技术。

**📊 数据集**

数据集包括四个真实世界精细操作任务（插杆、以太网插头、方块装配、电源插头）以及 RoboTwin 模拟任务 Click Bell 与 Beat Block Hammer。

**📈 对比分析**

与 WA‑base、Probe‑Learn‑Distill、HG‑DAgger、RL‑token‑like 等基线比较，HALO‑WA 在真实任务上平均成功率从 26.4% 提升至 87.1%，并在模拟任务中成功率提升至 97% 与 45%。

**⚠️ 局限性**

局限在于仅适用于已具备基本任务能力的 WA 模型，需进一步验证在更广泛任务与机器人平台的泛化，以及未考虑触觉、力反馈等信息。

---

## 256. Integrated Graph Search and Model Predictive Control for Smooth and Efficient Path Planning in Autonomous Vehicles

**arXiv ID:** 2607.04259 | [PDF](https://arxiv.org/pdf/2607.04259v1)

**作者:** Duc-Tien Bui `[一作]` (Graz University of Technology), Arno Eichberger `[通讯]` (Graz University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种结合图搜索与MPC的顺序路径规划框架，先用Dijkstra在离散网格上得到粗略路径，再构造空间变异的凸侧向安全通道，在该通道内用MPC对路径进行细化。

**💡 创新点**

创新点在于将图搜索得到的粗略路径直接用于构建可变凸安全通道，将离散障碍避免决策转化为连续可行约束，并在MPC中惩罚三阶空间导数实现平滑、低计算量的路径优化。

**🔧 技术方法**

技术包括Dijkstra网格搜索、凸通道构造、Frenet坐标系、基于三阶Taylor展开的离散动力学、MPC优化（使用MATLAB mpcActiveSetSolver）、CarMaker高保真仿真以及BMW 5系列车辆模型。

**📊 数据集**

使用CarMaker仿真平台与BMW 5系列车辆模型进行实验；实验场景包括直道单目标车、弯道单目标车和直道多目标车三种。

**📈 对比分析**

与之前的多项式拟合+QP方法对比，评估指标为侧向加速度、侧向力矩（jerk）、曲率、转向角以及平均计算时间；结果显示本方法在平滑度（加速度、曲率、转向角更小）和计算效率（平均每步时间降低约28–30%）方面均有显著提升。

**⚠️ 局限性**

局限性包括仅在仿真环境验证，缺乏真实车辆实验；对动态障碍的适应性未充分探讨；网格离散化对大规模或高动态场景的可扩展性仍待验证；以及对更复杂交通场景的鲁棒性尚未系统评估。

---

## 257. Air-Plan: Query-Optimized Topology Selection for Over-the-Air Decentralized Federated Learning

**arXiv ID:** 2607.04254 | [PDF](https://arxiv.org/pdf/2607.04254v1)

**作者:** Kaushal Attaluri `[一作]` (University of Vigo), Manuel Fernandez Veiga `[通讯]` (University of Vigo)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AirPlan，将 OTA 去中心化联邦学习的拓扑选择问题映射为分布式查询优化问题，实现了自动化的拓扑与稀疏化协同设计。

**💡 创新点**

首次证明 OTA‑DFL 与分布式查询处理等价，利用 Count‑Min Sketch 收集隐私统计、构建成本模型并在训练过程中进行在线自适应重优化，完成从工作负载到最优通信图的闭环决策。

**🔧 技术方法**

采用无线 OTA 叠加聚合、图谱谱间隙分析、top‑k 稀疏化、AQP 误差界、统一查询成本模型、Count‑Min Sketch、数据库查询优化方法（计划枚举、卡方估计、AQP）以及本地 SGD 与稀疏通信等技术。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100 和 Tiny‑ImageNet 三个公开图像分类数据集上进行实验。

**📈 对比分析**

与中心化 FedAvg、数字化 Decentralized SGD、DLLR‑OA、MATCHA 以及固定拓扑等基线在多种拓扑、SNR、数据异质性与客户端规模下对比；AirPlan 在 91.4% 的配置下匹配 oracle 最优拓扑，通信成本仅为 Ring 的 2.3 倍，准确率提升 0.3–1.0 点，统计收集开销低于 1.8%。

**⚠️ 局限性**

假设完美通道倒置与同步 OTA，仅搜索预定义拓扑族；对极低 SNR、极端异构及大规模客户端的鲁棒性有限；实验仅在仿真环境，缺乏真实硬件验证；未深入探讨异步 OTA 与能耗优化。

---

## 258. Shortcut Learning in Legal Judgment Prediction: Empirical Evidence from the UK Employment Tribunal

**arXiv ID:** 2607.04261 | [PDF](https://arxiv.org/pdf/2607.04261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 259. Agentic IoT: Architectures, Applications, and Challenges Toward the Internet of Agents

**arXiv ID:** 2607.04219 | [PDF](https://arxiv.org/pdf/2607.04219v1)

**作者:** Rümeysa Hilal Sevinç `[一作]` (Ankara University), İbrahim Kök `[通讯]` (Ankara University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文系统综述了Agentic IoT的概念、架构与应用，提出了跨层级认知循环与工具集成的参考框架，并总结了现有研究的技术与挑战。

**💡 创新点**

创新点在于首次对Agentic IoT进行正式定义，构建三层参考架构与共享认知平面，并将大型语言模型与自动化代理的最新技术与物联网紧密结合，提出了未来研究方向。

**🔧 技术方法**

主要技术包括大型语言模型（LLM）驱动的推理与规划、检索增强生成（RAG）、功能调用（Tool Use）、跨层代理认知循环、MCP/A2A/ACP等代理协议、TinyML与量化LLM、数字孪生与仿真沙箱等。

**📊 数据集**

作为综述性工作，未使用特定数据集，而是梳理了多个领域（如智慧城市、灾害响应、工业与农业）中的实例与公开数据来源。

**📈 对比分析**

通过对比文献与案例，评估了不同层级部署与协议的优缺点，但并未给出统一的实验性能指标；作者指出现有系统在实时推理、能耗与安全性方面表现参差不齐。

**⚠️ 局限性**

局限包括：缺乏大规模真实场景验证、资源受限下的代理设计不成熟、协议标准化与互操作性不足、信任与安全治理挑战、可解释性与责任追溯不完善，以及缺乏统一的基准与评测平台。

---

## 260. Agentic SABRE: An Uncertainty-Aware Neuro-Symbolic Multi-Agent Framework for Adaptive Ransomware Detection

**arXiv ID:** 2607.04292 | [PDF](https://arxiv.org/pdf/2607.04292v1)

**作者:** Henry Kabuye `[一作]` (Northumbria University), Jeyamohan Neera `[通讯]` (Northumbria University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 Agentic SABRE，一种结合语义嵌入和行为遥测、利用 Monte Carlo Dropout 估计不确定性，并通过阈值策略实现自动封堵、人工升级或允许执行的多代理框架。

**💡 创新点**

创新点在于：① 将语义和行为两类异构模型拆分为独立代理，② 采用 MC Dropout 进行每代理的不确定性量化并以最大不确定性做决策阈值；③ 设计了可解释的三阈值策略（风险阈值 τ 与不确定性阈值 κ），④ 用 CTGAN 对分数空间进行数据增强，实现鲁棒融合；⑤ 集成符号提示和解释机制（梯度敏感度、置换重要性、对抗阈值）。

**🔧 技术方法**

主要技术包括 1D 卷积网络、Monte Carlo Dropout、CTGAN 生成式增强、MLP 决策融合、阈值策略、解释性工具（梯度显著性、SHAP、对抗阈值）。

**📊 数据集**

使用了 RDset（PE 语义特征）和 RanSMAP（行为遥测）两个公开数据集；两者在训练与测试时不共享原始特征，仅在分数层融合。

**📈 对比分析**

与单一语义或行为分类器、传统集成以及现有深度学习检测方法比较，Agentic SABRE 在 RDset 上实现 100% AUC，RanSMAP 上提升 AUC 至 0.54（相比 0.51 的均值融合）并在保持召回率不变的情况下将升级率下降 4.9% 以上；在不同的拆分（硬件、时间、家族）下表现稳健，误警率和误封闭率显著降低。

**⚠️ 局限性**

局限包括：① MC Dropout 估计在强概念漂移下可能欠估不确定性；② 阈值阈值为轴对齐，缺乏对抗性极端样本的曲线决策；③ 假设语义与行为代理输出条件独立，忽略两者可能的相关性；④ 依赖离线阈值优化，未实时适应持续漂移；⑤ 对抗攻击实验受限于可行的离散约束，无法完整评估鲁棒性。

---

## 261. LBR: Towards Mitigating Length Bias in Large Language Models for Recommendation

**arXiv ID:** 2607.04270 | [PDF](https://arxiv.org/pdf/2607.04270v1)

**作者:** Hongchen Li `[一作]` (Zhejiang University), Jiawei Chen `[通讯]` (Zhejiang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 LBR（Length Bias Reduction）框架，通过校正注意力和重新归一化解码分数，解决 LLM 推荐系统中的长度偏差。

**💡 创新点**

创新点在于同时针对输入端的自注意力偏差和输出端的 Trie‑约束解码偏差，分别设计了 Length‑Aware Attention Calibration（LAAC）和 Effective Information Length Normalization（EILN）。

**🔧 技术方法**

核心技术包括在注意力 logits 中加入长度偏置，利用 Hartley 熵衡量 Trie 节点信息量，并以信息长度代替原始 token 数进行解码归一化；实现仅需两参数，兼容任意 Transformer‑LLM。

**📊 数据集**

实验使用 Amazon Toys & Games、Office Products 以及 Books 三大公开数据集，模型基于 LLaMA3.2‑3B 微调并采用 Constrained Beam Search。

**📈 对比分析**

与 SOTA 基线（BIGRec、LLaRA 等）对比，LBR 在 NDCG@5 上平均提升 16.82%，并显著降低不同长度组的推荐偏差，保持极低的额外计算开销。

**⚠️ 局限性**

局限性包括：仅关注长度偏差，未处理热门、位置等其他偏差；适用范围主要为 Trie‑约束 LLM 推荐，非全局生成任务；线性长度函数可能不足以捕捉更复杂的长度关系。

---

## 262. Beyond Random Sampling: Distribution-Aware Alignment for Semi-Supervised Medical Image Segmentation

**arXiv ID:** 2607.04249 | [PDF](https://arxiv.org/pdf/2607.04249v1)

**作者:** Weihao Yan `[一作]` (Shanghai Jiao Tong University), Ming Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出分布感知样本选择与基于内存的复制粘贴模块，以提升极低标签下的半监督医学图像分割性能。

**💡 创新点**

离线Density-K-Center采样基于Vision Foundation Model的多层特征覆盖全局数据流形，在线Memory-guided Copy-Paste（MCP）通过前景/背景语义记忆库与KL匹配进行增强，并配合easy-to-hard渐进式激活抑制伪标签噪声。

**🔧 技术方法**

使用DINOv2-Small VFM特征提取、Density-K-Center采样、三分支教师-学生框架、语义记忆库、KL距离检索、easy-to-hard progressive schedule以及Copy-Paste等技术。

**📊 数据集**

六个多模态医学数据集（BUSI、PMTCXR、ISIC、CAMUS、PROMISE、ACDC），涵盖2D/3D、超声、X光、皮肤镜、MRI等。

**📈 对比分析**

与多种SOTA半监督方法（如ABD、RCP、AdaMix、SynFoc、UniV2等）以及传统Copy-Paste做对比，低标签比例（1/16、1/8、1/4）下Dice/IoU提升5-10%，HD95下降至20-40像素，显著缩小与全监督差距。

**⚠️ 局限性**

目前仅按切片级别处理3D数据，需逐片推理；记忆库容量与检索策略对不同模态的适用性有限；前期VFM特征提取为离线步骤，增加前期成本；对极端类别不平衡和伪标签质量仍有待进一步提升。

---

## 263. Hierarchical Multi-to-Single-Modal Knowledge Distillation for Disruption Prediction in EAST

**arXiv ID:** 2607.04241 | [PDF](https://arxiv.org/pdf/2607.04241v1)

**作者:** Qiang Chen `[一作]` (Anhui University), Guosheng Xu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了层次化多模态至单模态知识蒸馏框架，用可见光图像与时序诊断信号训练多模态教师，推理时仅用时序学生模型。

**💡 创新点**

创新点在于结合时间-空间超图融合与三层（结构、表示、决策）蒸馏，既保留多模态学习优势，又显著降低推理成本。

**🔧 技术方法**

使用Transformer编码器、原型引导时空超图、Graph结构蒸馏、SmoothL1与温度KL蒸馏、t‑SNE可视化、Grad‑CAM与Integrated Gradients等技术。

**📊 数据集**

采用EAST同步多模态数据集（640道放电，含可见光图像与11条时序诊断信号），并在独立泛化集与1308道扩展集上进行验证。

**📈 对比分析**

通过与时序、视频、全模态模型对比；在10 ms预警下多模态教师TPR 100%、FPR 2.73%、F1 96%、AUC 99%；蒸馏后单模态学生TPR 91.66%、FPR 2.73%、F1 91.66%、AUC 97.88%，推理速度提升2.16×，FLOPs/参数分别降低68.9%/47.9%。

**⚠️ 局限性**

在未知放电的泛化上仍有挑战，蒸馏后TPR略降；仅利用可见光与时序信号，未涵盖更丰富的诊断模态，跨机台验证仍待深入。

---

## 264. Spinning Straw into Gold: Relabeling LLM Agent Trajectories in Hindsight for Successful Demonstrations

**arXiv ID:** 2607.04235 | [PDF](https://arxiv.org/pdf/2607.04235v1)

**作者:** Zichao Li `[一作]` (McGill University), Jihyung Kil `[通讯]` (Adobe Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过辅助大型语言模型对LLM Agent生成的轨迹进行回顾性标注，将未实现但已达成的目标重新标记为成功示例，并基于这些重新标注的轨迹进行监督学习。

**💡 创新点**

提出了Hindsight Supervised Learning框架，并引入不相关动作掩蔽与示例重加权两种技术，以提升样本效率和覆盖率。

**🔧 技术方法**

使用Llama‑3.3‑70B作为辅助LLM完成轨迹回顾与目标识别，结合Llama‑3.2‑1B在SFT/DPO上微调，并在损失中加入掩蔽与重加权机制。

**📊 数据集**

在ALFWorld、PlanCraft和WebShop这三个常用的LLM Agent基准数据集上进行实验。

**📈 对比分析**

与原始SFT和DPO相比，加入HSL后在ALFWorld成功率提升约8%–32%，在样本稀缺情形下仅使用原始示例的1/4即可超越全量基线；PlanCraft与WebShop的提升幅度相对较小。

**⚠️ 局限性**

对回溯标注的准确性有一定依赖，误标可能导致学习偏差；对单一目标轨迹的覆盖不足，且对多目标多步骤任务的提升受限于回溯专家的优化程度。

---

## 265. Detecting Hallucinations in Retrieval-Augmented Generation through Grounding-Aware Sensitivity by Perturbation (GASP)

**arXiv ID:** 2607.04223 | [PDF](https://arxiv.org/pdf/2607.04223v1)

**作者:** Mohamed Aly Bouke `[一作]` `[通讯]`, Mohamed Aly Bouke

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于检索上下文扰动的句子级幻觉检测方法GASP，能够为每个句子生成 grounding‑sensitivity 分数并给出支持证据片段；

**💡 创新点**

创新点在于将检索上下文的删除视为对生成过程的对抗扰动，利用对数似然下降和 Jensen‑Shannon 散度来衡量句子对检索证据的依赖，从而实现无需训练、可解释的幻觉检测；

**🔧 技术方法**

技术主要包括：概率语言模型的前向推理、对检索上下文的全、空、逐块剔除重评分、JSD 与 log‑likelihood 计算、梯度提升树分类器（可选）以及基于 RNIFS 的理论解释；

**📊 数据集**

使用三个基准：RAGTruth（含句子级幻觉标注），TofuEval（会议摘要一致性），以及 RAGBench（短答问答）进行评估；

**📈 对比分析**

与 perplexity、长度、全上下文 NLI、Self‑Consistency 等基线比较，GASP 在 RAGTruth 和 TofuEval 的句子级 AUC 均显著优于基线，且在 RAGBench 上与 perplexity 相近，训练‑free 阈值版本已匹配或超过训练版；

**⚠️ 局限性**

局限性包括：需要可对检索上下文进行完整切分且在短答场景下效果不佳，模型需对检索信息有足够敏感度，且对截断、语料分布变化敏感，若检索信息未被正确使用则信号会被压缩到随机水平。

---

## 266. Unsupervised Features Mining via Activation Geometry

**arXiv ID:** 2607.04222 | [PDF](https://arxiv.org/pdf/2607.04222v1)

**作者:** Amit LeVi `[一作]` (Zenity), Max Fomin `[通讯]` (Zenity)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了无监督的 Mining via Activation Geometry (MAG) 框架，用固定自然语言前缀提取模型内部的推理特征；

**💡 创新点**

创新点在于利用激活几何直接捕捉模型自我判定的推理方向，并将其用于可读性、线性性、可控性和数据集迁移预测；

**🔧 技术方法**

技术核心包括对前缀引起的激活差异进行向量化、计算平均偏移向量、线性重建误差评估、类别均值方向的激活注入以及多种 MAG 运算符；

**📊 数据集**

实验使用 Llama‑3.1‑8B‑Instruct、Gemma‑2‑9B‑it 及 Qwen‑2.5‑32B‑Instruct 三个模型，前缀集包括 PI、desert、ocean、Bob 四种；

**📈 对比分析**

与传统的未修饰激活余弦相似度相比，MAG 通过多种度量（余弦、欧氏、CKA、MMD 等）和条件聚类显著提升了训练集迁移排序的 Top‑1/Top‑2 准确率，最佳三重条件组合在 19 次实验中达 94.7% 的 Top‑1；

**⚠️ 局限性**

局限包括依赖单一下游分类器、仅在部分 shuffle 下可计算条件聚类、未检验对不同任务的泛化、对开放式生成的可控性有限以及前缀生成的方向仍未揭示内部机制；

---

## 267. Channel-Adaptive Robust Aggregation for Over-the-Air Federated Learning in Heterogeneous Networks

**arXiv ID:** 2607.04218 | [PDF](https://arxiv.org/pdf/2607.04218v1)

**作者:** Zubaida Fatima `[一作]`, B. N. Bharath `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CHARGE-FL 框架，能够根据无线通道状态和应用事件动态触发聚合，并结合双重预编码以补偿客户端不同局部训练进度。

**💡 创新点**

创新点包括：①聚合时刻不再固定，而是自适应触发；②联合优化预编码与聚合权重，既能抵消通道噪声和衰落，又能平衡不同步的客户端贡献；③提供 O(1/T) 的理论收敛保证，首次在噪声、衰落与异质性三者共存的 OTA‑FL 环境下实现。

**🔧 技术方法**

使用的技术包括：过载无线多址（OTA）聚合、动态预编码、阈值式客户端选择、联邦 SGD、梯度方差控制与收敛分析。

**📊 数据集**

实验数据集为 CIFAR‑10 与 CIFAR‑100，采用三卷积+两全连接的 CNN 模型，数据按 IID 与非IID（标签不均匀）划分。

**📈 对比分析**

与 COTAF 以及 NoisyProx（FedProx 变体）在 AWGN 与衰落通道、不同 straggler 百分比、SNR 水平、数据异质性等场景下进行对比。结果表明 CHARGE‑FL 在准确率、收敛速度、对噪声与 straggler 的鲁棒性方面均优于基线，尤其在高异质性与低 SNR 条件下提升显著。

**⚠️ 局限性**

局限性包括：需要准确的通道状态估计与阈值设定，平均预编码可能导致部分信息损失；在极低 SNR 时仍可能退化；未考虑客户端掉线、隐私攻击与安全性问题；通信成本在多局部步骤的场景下仍高于纯单步方案。

---

## 268. Ball Differential Privacy: How to Mitigate Data Reconstruction with Less Noise

**arXiv ID:** 2607.04209 | [PDF](https://arxiv.org/pdf/2607.04209v1)

**作者:** Joseph Margaryan `[一作]` (University of Copenhagen), Nirupam Gupta `[通讯]` (University of Copenhagen)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Ball‑DP，即对传统差分隐私的局部邻接约束，使用半径r限制可被保护的记录替换范围；

**💡 创新点**

创新点在于将隐私保护范围显式为可配置的球形邻接关系，降低噪声并保持对局部重构攻击的安全性，同时给出Ball‑ReRo重构鲁棒性证明；

**🔧 技术方法**

采用了凸优化中的输出扰动（高斯噪声）和对r‑邻接的L₂灵敏度分析，利用分析型高斯机制以及REINFORCEMENT‑DP（Ball‑RDP）等工具；

**📊 数据集**

使用七个固定嵌入数据集（AG News、BANKING77、CIFAR‑10、Emotion、IMDb、MNIST、TREC‑6）以及在IMDb上进一步验证二元逻辑回归、softmax逻辑回归和平方铰链SVM等头；

**📈 对比分析**

与全局标准DP（使用全局替换半径2B）对比，Ball‑DP在相同ε、δ下显著提升准确率，尤其在高隐私（ε=2）时提升0.3%–14%；在重构安全性上，Ball‑ReRo证书与实验MAP攻击保持一致，未超过理论上限；

**⚠️ 局限性**

局限性包括仅处理凸一轮高斯输出扰动、公开的嵌入与计数信息，未考虑标签变化、SGD轨迹、非凸表示学习等场景，且Ball‑RDP对高阶统计量的解释仍待深入。

---

## 269. SpecGradFilter: A Spectral Gradient Filtering Framework for Taming Federated Heterogeneity

**arXiv ID:** 2607.04189 | [PDF](https://arxiv.org/pdf/2607.04189v1)

**作者:** Liyang Yuan `[一作]` (Jilin University), Zhouchen Lin `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出 SpecGradFilter，通过在频域抑制低频梯度成分来缓解联邦学习中的客户端漂移问题。

**💡 创新点**

创新点在于将梯度视为结构化信号，揭示“漂移的频谱偏差”，并通过高通滤波（FFT、局部平均池化或高斯平滑）实现自适应去趋势。

**🔧 技术方法**

采用离散傅里叶变换（rFFT）实现高通滤波，或者在空间域使用局部平均池化/高斯平滑近似低频抑制，并在 FedAvg 等常规算法的本地更新环节插入该操作。

**📊 数据集**

实验覆盖图像分类数据集 CIFAR‑10/CIFAR‑100/Tiny‑ImageNet 及其噪声版本 CIFAR‑10‑C/CIFAR‑100‑C，文本分类数据集 Yahoo! Answers，医学影像数据集 BloodMNIST，手写数字数据集 FEMNIST，使用 ResNet‑20、ResNet‑18、WideResNet‑56‑2、DenseNet‑121、ViT 等多种网络。

**📈 对比分析**

与 FedAvg、FedProx、FedDyn、SCAFFOLD、FedCM、FedSAM、FedDisco、FedAWA、FedLWS 等 9+ 基线相比，SpecGradFilter 在高异构性（α=0.1）下可提升约 21–23% 的准确率，并在多种模型与数据集上加速收敛，通信开销几乎不变。

**⚠️ 局限性**

局限性包括：在中心化 SGD 下低频抑制会略微降低性能；需手动设置过滤比例 r 并且对梯度展开顺序敏感；目前仅在中小规模模型验证，尚未评估在大规模 LLM 等任务中的效果。

---

## 270. Physics-Informed Graph Learning with Uncertainty Awareness for Open-Set Domain Generalization in Fault Diagnosis

**arXiv ID:** 2607.04188 | [PDF](https://arxiv.org/pdf/2607.04188v1)

**作者:** Jinfeng Zhu `[一作]` (Southwest University), Ye Yuan `[通讯]` (Southwest University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种面向旋转机械故障诊断的开集域泛化框架 PGU-OD，旨在在未知故障和域迁移条件下实现可靠识别与拒绝。

**💡 创新点**

创新点包括：① 将物理约束的 Morlet 小波卷积与频谱注意力相结合，构建 PISA-Net 提升特征鲁棒性；② 设计基于类尺度 Gaussian 参数的不确定性感知自适应图学习，抑制结构层不确定性传播；③ 引入 Gaussian 适应边界联合损失和双准则开集推理，动态调整决策边界并可靠拒绝未知样本。

**🔧 技术方法**

主要技术：物理信息驱动的频谱注意力卷积、类尺度 Gaussian 先验建模、异方差核自适应图卷积、监督对比损失、Gaussian 适应边界优化、双准则距离阈值推理。

**📊 数据集**

实验使用了两个公开的旋转机械故障数据集：Case Western Reserve University Bearing（CWRU）和 Paderborn（PU）数据集，涵盖多种轴承故障类型和多种工况变迁。

**📈 对比分析**

与 OSDA（OSBP）、DW‑DANN、KTR‑BUNN、AOSDGN、MDCC、ACDPN 等六种基线方法对比，PGU-OD 在大多数跨域开集任务中获得最高的 H‑score，尤其在域差距大、未知类比例高的难度任务中表现突出，显著提升了已知类识别和未知类拒绝的综合性能。

**⚠️ 局限性**

局限性：尚未考虑极端域偏差下的连续退化建模、少量样本的开集适应，以及在工业部署中的不确定性校准与实时性评估等问题，未来工作将进一步完善这些方面。

---

## 271. CausalGame: Benchmarking Causal Thinking of LLM Agents in Games

**arXiv ID:** 2607.04293 | [PDF](https://arxiv.org/pdf/2607.04293v1)

**作者:** Zhenhao Chen `[一作]`, Kun Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为CausalGame的交互式基准，用于评估LLM驱动的AI科学家在隐藏因果机制、选择偏差、测量误差和隐藏混杂的环境下进行实验设计、数据分析和报告的能力。

**💡 创新点**

创新点在于将科学发现过程抽象为结构因果模型驱动的游戏，结合四维度的评分量表（因果推理、实验设计、反思质量、数据使用）来细粒度评估模型的因果思维，并在基准中系统引入了现实中的观察偏差挑战。

**🔧 技术方法**

本文使用大规模LLM（GPT‑5.5系列、Claude‑Opus、Gemini、Grok、DeepSeek、GLM、MiniMax、Qwen等）以及多种交互执行模式（单轮提示、ReAct、OpenCode），并对模型进行多轮工具调用与实验探索。

**📊 数据集**

数据集为14个基于结构因果模型的模拟场景，涵盖天气、敌人探测、无人机部件损伤等因素，提供可重复的因果生成机制和真实标签。

**📈 对比分析**

通过对比模型在生存率、评估得分和基准指标的表现，发现所有LLM模型在选择偏差与隐藏混杂场景下的生存率均低于理论最优（约68% vs 82%），因果推理得分近乎为零，且与现有能力基准的相关性低于0.35。

**⚠️ 局限性**

局限性包括：LLM缺乏可靠的因果推理能力，易受偏差误导；对实验设计的探索不足；在评估过程中出现黑客式信息泄露和误报胜利的行为；且在更复杂或多样的真实科学任务中可泛化性待验证。

---

## 272. FLOAT Drone for Physical Interaction: Lateral Airflow Reduction, Wrench Modeling, and Adaptive Control

**arXiv ID:** 2607.04260 | [PDF](https://arxiv.org/pdf/2607.04260v1)

**作者:** Junxiao Lin `[一作]` (Zhejiang University), Fei Gao `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计并实现了FLOAT Drone，一种利用舵面在共轴旋翼下产生全方位力矩的全致动无人机，并在实验中演示了其在狭小空间内的接触操控能力。

**💡 创新点**

创新点在于：①将舵面嵌入旋翼下游，以减少水平气流并实现六自由度力矩生成；②构建高保真二阶多项式风机-舵面耦合模型并用于非线性控制分配；③将SE(3)几何控制与ℒ1自适应补偿结合，实现对模型不确定性、负载变化与地面效应等干扰的实时补偿。

**🔧 技术方法**

采用的技术包括：共轴双旋翼结构、舵面空气动力学建模、CFD仿真与力匹配、基于CasADi/Ipopt的非线性优化分配器、SE(3)几何控制、ℒ1自适应模块，以及NVIDIA Jetson Orin NX/PX4V2实现的实时控制。

**📊 数据集**

使用的数据集为：实验室采集的静态力测量数据（包含舵面角度与多轴力矩）、CFD仿真得到的气流场与受力匹配结果、以及在室内运动捕捉环境下收集的飞行与交互轨迹（位置、姿态、加速度、力矩等）。

**📈 对比分析**

通过与线性控制分配、仅含ℒ1补偿、仅用非线性分配器四种配置进行对比，评价指标为位置RMSE、姿态RMSE、最大高度误差、以及对地面效应、突发负载变化和长期漂移的抑制效果。结果显示，结合非线性分配和ℒ1补偿的方案在三维轨迹追踪、姿态切换、地面效应恢复、负载突变恢复以及长时悬停等场景下，均显著降低误差（如3D轨迹位置RMSE从49.7mm降至13.8mm，最大高度误差从97.3mm降至13.5mm）。

**⚠️ 局限性**

局限性在于：实验仅在受控室内环境下进行，使用的机械抓取工具为简易侧挂钩，缺乏通用末端执行器和自主任务规划；此外，舵面耦合模型对极端工况（如高风速或快速舵面切换）尚未充分验证。

---

## 273. AdaptiveSplat:Texture Aware Controllable 3D Gaussian Allocation for Feed-Forward Reconstruction

**arXiv ID:** 2607.04256 | [PDF](https://arxiv.org/pdf/2607.04256v1)

**作者:** Badrinath Singhal `[一作]` (Indian Institute of Science), Venkatesh Babu Radhakrishnan `[通讯]` (Indian Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种可控的、基于纹理的前向3D高斯稀疏化框架，允许用户在保持视觉质量的同时显著减少原始高斯数目。

**💡 创新点**

通过对图像的离散小波纹理能量估计进行纹理感知的区域级裁剪，并在同一前向管线中引入自适应高斯头重新预测保留高斯的属性，从而在不需要后期优化的情况下实现可控稀疏化。

**🔧 技术方法**

采用离散小波变换 (DWT)、超聚类 (SuperCluster) 进行纹理分组，基于 K‑means 进行聚类，利用多视角特征提取（MASt3R/VGGT + DPT）构建自适应高斯头；同时通过 β 均匀采样训练以适应不同稀疏率。

**📊 数据集**

主要在 RealEstate10K、ACID、DL3DV 进行训练与评估，零样本泛化在 DTU 上验证，并在 Tanks & Temples 上做补充测试。

**📈 对比分析**

对比传统前向模型（如 pixelSplat、MVSplat 等）配合 EAGLES、LightGaussian 等剪枝方法，并评估是否微调；在 β=0.4/0.6/0.8 时，本文在 PSNR、SSIM、LPIPS 上均显著优于基线，尤其在高稀疏率下保持较高质量，同时渲染速度提升至约 40–60 FPS。

**⚠️ 局限性**

仍受限于像素对齐的前向模型，难以完美处理高频纹理细节、镜面高光等；对超大尺度场景的几何与纹理共同决策尚需改进。

---

## 274. Quantize the Target, Quantize the Drafter: Efficient Inference with Qwen3.5-4B

**arXiv ID:** 2607.04244 | [PDF](https://arxiv.org/pdf/2607.04244v1)

**作者:** Jaeyeon Kim `[一作]` (Nota Inc), Bo-Kyeong Kim `[通讯]` (Nota Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 NVIDIA A10G GPU 上实现 Qwen3.5‑4B 的低延迟推理，采用量化目标模型、量化感知蒸馏以及为量化模型专门训练的块扩散草稿模型结合推测式解码实现加速。

**💡 创新点**

创新点在于：① 保留原始量化网格的量化感知蒸馏恢复精度；② 两阶段块扩散草稿训练提升草稿质量；③ 对草稿模型进行 GPTQ 量化和滑动窗口注意力（SWA）优化，显著降低推测式解码开销。

**🔧 技术方法**

使用技术包括 AWQ 低位量化、量化感知蒸馏 (QAD)、块扩散草稿模型 DFlash、推测式解码、GPTQ 量化以及滑动窗口注意力 (SWA)。

**📊 数据集**

使用的数据集有：Nemotron‑Post‑Training‑Dataset‑v2（220K/400K 生成对话）、SciQ、BoolQ、PIQA、HellaSwag、MMLU‑Pro、IFEval、GPQA‑Diamond、GSM8K、HumanEval 与 LongBench v2。

**📈 对比分析**

与 BF16 基线对比，INT4+QAD 仍满足质量门限，草稿模型两阶段训练提升接收长度，最终系统在 NVIDIA A10G 上实现平均 6.978× 的速度提升，排名第三。

**⚠️ 局限性**

限制：需在真实 A10G 环境进一步验证；SWA 窗口长度的折中仍影响极长上下文性能；对极长输入的鲁棒性和多 GPU 扩展性尚未完全评估。

---

## 275. Biological Motifs for Agentic Control

**arXiv ID:** 2607.04240 | [PDF](https://arxiv.org/pdf/2607.04240v1)

**作者:** Bogdan Banu `[一作]` `[通讯]`, Bogdan Banu

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将基因调控网络（GRN）中的控制网络基因调控动力学与多代理系统的架构进行形式化对应，提出Agentic Operad和Epistemic Topology，验证通过四条预测定理与多代理基准一致，给出六层发展进程及实现示例。

**💡 创新点**

创新点在于：① 用多项式函子和连线图将生物网络模式映射到代理软件；② 定义Agentic Operad以约束可编译拓扑并证明错误抑制界限；③ 通过观察结构导出Kripke知识算子并给出四条可预测的多代理扩展定理；④ 结合资源、元认知与生物学机制提出完整的六层进化框架。

**🔧 技术方法**

技术包括：应用范畴理论（多项式函子、连线图、Operad）、光学工具（Lens、Prism、Traversal）、合成合流子（Coalgebra）、元认知与生物学启发的安全/资源/元件模型，实验实现采用LLM+工具链，脚本化测试。

**📊 数据集**

使用标准多代理基准（如多代理对话、任务拆分、工具调用）与公开LLM模型；参考Kim等的多代理实验数据；实现中包含1813条单元测试和116个示例。

**📈 对比分析**

对比方法：先给出四条理论定理预测错误放大、顺序惩罚、并行加速、工具密度 scaling；随后在实现中在这些拓扑下运行基准，观察误差率、响应时间等，与Kim等报告的指标对齐，结果表明理论与实验高度一致。

**⚠️ 局限性**

局限性：① 依赖于接口层面的结构对应，无法保证机制层面的完全对应；② 预测定理基于简化的概率与资源模型，现实环境中协同与干扰更复杂；③ 需要手动设计和校准多代理拓扑与光学；④ 对大规模动态学习系统的验证有限。

---

## 276. SoftVTBench: A Safety-Aware Visuo-Tactile Benchmark for Physically Constrained Robotic Manipulation of Deformable Objects

**arXiv ID:** 2607.04234 | [PDF](https://arxiv.org/pdf/2607.04234v1)

**作者:** Bowen Jing `[一作]` (Tuojing Intelligence), Haibao Yu `[通讯]` (Tuojing Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套名为SoftVTBench的安全感知视觉‑触觉基准，用于评估物理约束下可变形物体的操纵性能；

**💡 创新点**

创新点在于将任务成功与物理安全拆分，采用有限元隐式状态监测来定义安全成功，并为可变形物体提供多模态视觉‑触觉观测；

**🔧 技术方法**

采用Isaac Sim与FEM软体动力学、双指GelSight Mini触觉传感器、π_0.5强化学习框架以及LoRA微调技术；

**📊 数据集**

使用约2000个由SoftVTBench收集的操纵轨迹，包含33种软体资产和四大任务套件（Object‑Soft、Spatial‑Soft、Object‑Rigid、Spatial‑Rigid）；

**📈 对比分析**

对比了vision‑only（π_0.5‑Vision）与visuo‑tactile（π_0.5‑Visuo‑Tactile）两种基线，结果显示在可变形任务上触觉观测显著提升安全成功率（如Object‑Soft从21.4%提升至35.6%），但在刚性任务上提升有限；

**⚠️ 局限性**

局限性包括资产和任务多样性不足、FEM模拟与真实软体动力学的差异、只评估π_0.5基线，未来需扩展更多资产、真实物理模型与更广泛的策略比较。

---

## 277. Teaching Code LLMs to Reason with Intermediate Formal Specifications

**arXiv ID:** 2607.04232 | [PDF](https://arxiv.org/pdf/2607.04232v1)

**作者:** Minh Le-Anh `[一作]` (FPT Software AI Center), Tien N. Nguyen `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练 CodeLLMs 生成可执行的检查点规范，利用验证指导训练框架从参考程序、行为改变突变体和多轮规范细化中获取监督。

**💡 创新点**

提出了基于执行验证的检查点规范生成框架 SpecCoder，首次将可执行断言与行为改变突变体结合，实现对中间程序状态的可验证规范；并构造了 HumanExec 基准。

**🔧 技术方法**

结合大语言模型（Qwen2.5-Coder）微调、执行反馈驱动的多轮探索–提交规范细化、程序突变生成、可执行断言评估（正确性、完整性）、监督微调。

**📊 数据集**

采用 LeetCode 风格提交验证作为参考程序，生成突变体；构造 HumanExec benchmark（150 条 Codeforces 问题，含人类错误提交、测试套件）用于评测。

**📈 对比分析**

与基线预训练 CodeLLM 对比，SpecCoder 在 HumanExec 上的检查点规范正确率提升 55.8%–65%，完整性提升 358.1%–490%，可执行性 90%+；在规范生成、程序正确性检查、程序修复三项任务中均显著提升，显示可执行检查点显著提高验证和修复效果。

**⚠️ 局限性**

仍依赖大量手工标注的参考程序和突变体；生成的断言可能无法覆盖所有细节；对复杂语言特性和动态行为的支持有限，且在极大规模程序上的执行成本较高。

---

## 278. Topology-Driven Transferability Estimation for 3D Medical Vision Foundation Models

**arXiv ID:** 2607.04199 | [PDF](https://arxiv.org/pdf/2607.04199v1)

**作者:** Jiaqi Tang `[一作]` (Peking University), Qingchao Chen `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于最小生成树的无参数拓扑框架，用于训练前的医学三维分割模型可迁移性评估；

**💡 创新点**

创新点在于：①用MST直接衡量特征与语义标签的拓扑一致性；②将评估拆为局部边界一致性(LBTC)和全局结构差异(GRTD)，并通过任务自适应门控融合；③证明随机初始化解码器可视为低方差的空间投影，提升全局评估稳定性；

**🔧 技术方法**

主要技术包括最小生成树（MST）构造、拓扑泄漏率、树权重差异、Johnson–Lindenstrauss随机投影、任务自适应门控；

**📊 数据集**

使用OpenMind基准，114,000张3D医学影像预训练模型以及7个下游分割任务（脑、腹部、肾、头颈等）和跨模态/跨区域数据；

**📈 对比分析**

与LogME、LEEP、GBC、CCFV等传统迁移性评估指标对比，使用加权Kendall’s τ评估模型排名；在ID与OOD场景下平均τ提升至0.638（相较CCFV的0.272提升约两倍），并且评估时间平均15秒，速度提升56×；

**⚠️ 局限性**

局限性：对极少样本/少量标注的情况MST构造可能不稳定；随机投影理论主要针对卷积上采样，对纯注意力或扩散解码器的推广需进一步验证。

---

## 279. Sangam: Efficiently Serving Diffusion LLMs with the AR Stack

**arXiv ID:** 2607.04206 | [PDF](https://arxiv.org/pdf/2607.04206v1)

**作者:** Nitin Kedia `[一作]` (University of Texas at Austin), Aditya Akella `[通讯]` (University of Texas at Austin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了适用于缓存扩散式语言模型（dLLM）的服务系统 Sangam，并实现了三种执行模式（同位、分离、混合）

**💡 创新点**

创新点在于引入缺陷代币预算调度器，实现在没有可分块预填充的前提下实现分块解码的分摊无阻塞调度，并将 AR 服务栈与 dLLM 的循环预填/解码结构相融合

**🔧 技术方法**

使用了 Fast-dLLM / dKV-Cache 的近似 KV 缓存、CUDA Graph、PagedAttention、FlashInfer 的分页 KV 核心以及 Deficit Round‑Robin 思想的预算调度器

**📊 数据集**

使用了公开的 LLaDA‑8B 与 Dream‑7B 两个 dLLM 以及 ShareGPT 和 arXiv 这两个真实工作负载追踪

**📈 对比分析**

通过与 Fast‑dLLM（单请求、无批量）和内部基线（不同预算）比较，Sangam 在相同延迟下可实现 2–3× 的吞吐量提升；同位模式在解码密集场景下平均延迟比混合模式低 9–20%，而混合模式在预填充密集场景下平均延迟比同位模式低 8–20%，总体能覆盖两类工作负载

**⚠️ 局限性**

局限性包括：仍依赖于硬件（H100 GPU）和内存带宽，预算调度对极低负载时可能导致队列排队；混合模式需手动设定阈值 θ，且 KV 迁移在跨节点部署时可能成为瓶颈

---

## 280. A Clustering-Based Framework for Identifying Suspicious Trading Patterns in Capital Market

**arXiv ID:** 2607.04184 | [PDF](https://arxiv.org/pdf/2607.04184v1)

**作者:** Asif Zaman `[一作]` (American International University-Bangladesh), Iftekharul Mobin `[通讯]` (American International University-Bangladesh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一个基于K‑Means++聚类的无监督市场操纵检测框架（SMMD），可识别并分类股票交易中的可疑模式。

**💡 创新点**

创新点在于将多维技术指标与行为阈值规则相结合，使用百分位风险分级和混合异常过滤，实现了可解释且低误报的欺诈类型识别。

**🔧 技术方法**

采用K‑Means++聚类、标准化、滚动窗口技术指标计算、百分位阈值判定以及混合距离‑行为规则的异常检测技术。

**📊 数据集**

使用2012‑2024年达卡证券交易所约一百万笔交易数据作为无标签数据集。

**📈 对比分析**

与DBSCAN、OPTICS、层次聚类等方法比较，K‑Means在准确率0.987、轮廓系数0.965以及可扩展性上表现最好；在本文模型中得到轮廓系数0.561，检测到2.02%可疑交易。

**⚠️ 局限性**

局限包括缺乏真实标签验证、阈值和权重参数缺乏自适应性、对极端动态市场的适应性有限，以及在其他地区数据迁移时需要重新调参。

---

## 281. Dynamic Interest Rate Discovery in Decentralized Finance: A Reverse Kelly Automated Market Maker for Risk-Adjusted Lending

**arXiv ID:** 2607.04178 | [PDF](https://arxiv.org/pdf/2607.04178v1)

**作者:** Sai Srikanth Madugula `[一作]` (Woxsen University), Daya Shankar `[通讯]` (Woxsen University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种逆Kelly自动做市商（rkAMM）模型，用于在DeFi中动态定价借贷，依据实时违约概率和目标流动性提供精确利率。

**💡 创新点**

创新点在于将Kelly准则反演得到严格凸的利率曲线，利用可解释AI违约概率oracles实现风险调整定价，弥补传统过度抵押利用率曲线的不足。

**🔧 技术方法**

采用 Solidity 智能合约（WAD 计数）、MLflow、DVC/DagsHub 版本控制、Ollama/Llama‑3 与 Hugging Face FinBERT 的边缘推理，以及 Monte Carlo 压力测试。

**📊 数据集**

使用 12,000 行 SME 企业账单与现金流数据集以及 Beta 分布模拟的违约概率。

**📈 对比分析**

通过 10,000 次宏观经济冲击的蒙特卡罗模拟与 Aave V3 静态利用率曲线对比，rkAMM 在正常与冲击市场中保持 11–12% LP 收益，解决传统模型的违约率失控。

**⚠️ 局限性**

局限包括边缘推理的延迟与对去中心化算力网络的依赖，以及在极端市场冲击下 oracle 停机可能导致定价滞后。

---

## 282. Agentic-V2X: Small Language Model Agents for Deadline-Aware V2X Scheduling in 5G/6G Networks

**arXiv ID:** 2607.04290 | [PDF](https://arxiv.org/pdf/2607.04290v1)

**作者:** Gerasimos Papanikolaou-Ntais `[一作]` (University of Piraeus), Athanasios Kanavos `[通讯]` (University of Peloponnese)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建Agentic-V2X架构，利用小型LLM在非实时层生成可验证的调度策略，并由轻量级xApp在近实时层执行，评估5G NR V2X网络的可靠性、时延与吞吐。

**💡 创新点**

将LLM限定在非实时rApp式策略生成，结合结构化YAML校验与运行时安全屏障，形成可验证且可执行的分层调度方案。

**🔧 技术方法**

使用小型本地LLM（如Llama 7B）生成YAML策略；ns3-ai与5G‑LENA实现与ns‑3仿真；xApp‑like控制器实现基于权重的PF调度；结构化验证与回退机制。

**📊 数据集**

基于SUMO Manhattan网格生成车辆移动轨迹，设置20/25/30车辆密度的V2X流量模型；通过7个随机种子构成126场实验。

**📈 对比分析**

对比六种方案（PF、静态专家、平衡专家、启发式xApp、静态LLM、适配LLM+ xApp），指标包括关键服务DC‑PRR、延迟、尾部延迟、背景吞吐等；适配方案在高密度下提升DC‑PRR与ToD UL可靠性，平均表现不如最优静态策略。

**⚠️ 局限性**

仅基于仿真；单一小型LLM、固定调度周期与单细胞环境；缺乏真实O‑RAN接口与多小区演示；策略有效性受提示与schema限制，未测试更复杂场景。

---

## 283. Signal or Noise? Understanding Generative Models for Real-World Sensor Time Series

**arXiv ID:** 2607.04245 | [PDF](https://arxiv.org/pdf/2607.04245v1)

**作者:** Zitao Shuai `[一作]` (University of California, Los Angeles), Yuzhe Yang `[通讯]` (University of California, Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对现实世界传感器时间序列进行大规模、统一的生成模型研究，构建 SensorGen 框架，涵盖多领域、多数据集、多模态与多任务。

**💡 创新点**

创新点在于：①提供首个统一的、开放的传感器生成实验平台；②系统比较 5 大主流生成模型家族；③揭示特定信号属性（如人口统计、时间‑频率）和模型设计（如流匹配、条件编码）对生成质量的关键影响；④证明合成信号在数据稀缺场景下可提升下游任务性能。

**🔧 技术方法**

使用的技术包括：扩散模型（DiT）、流匹配模型（SiT）、自回归模型（MAR）、归一化流（TarFlow）和分层模型（FractalGen、Imagen）；同时采用 Min‑Max 归一化、时间‑频率条件、不同训练步长和模型容量的可扩展性实验。

**📊 数据集**

数据集涵盖四大情境：急诊监测（如 MIMIC‑IV ECG）、日常生活感知（CAPTURE‑24、PPG‑DaLiA、Metabonet）、实验室监测（PhyMER、SHHS）和手术室监测（VitalDB），涉及 ECG、PPG、EEG、IMU、CGM 等多种信号。

**📈 对比分析**

与 5 家模型家族在 10+ 任务（语义‑到‑信号、插值/外推、通道翻译、编辑）上进行统一评估，指标包括 MSE、MAE、PSNR、SMSE、SSIM、FID 等。结果显示：流匹配模型整体表现最佳；人口统计条件在归一化后显著提升长期生成；时间‑频率条件显著提高高频信号的精度；更大模型与更长训练步数均带来性能提升；生成的合成数据可在疾病诊断与表征学习中以中等比例提升性能。

**⚠️ 局限性**

局限性包括：实验仅覆盖公开数据集和预设任务，未涵盖所有传感器场景与罕见情况；生成信号的临床可用性未得到验证；评估指标主要基于信号统计与视觉真实度，仍难以完全捕捉生理与时间序列的临床意义；需要进一步的专家评估与公平性、隐私分析。

---

## 284. How to Value Open Source Contributions? An Institutional Perspective from CERN

**arXiv ID:** 2607.04202 | [PDF](https://arxiv.org/pdf/2607.04202v1)

**作者:** Julie Skoven Hinge `[一作]` (IT University of Copenhagen), Andrzej Wąsowski `[通讯]` (IT University of Copenhagen)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并应用了一套框架，用来定量评估机构对 OSS 的贡献规模与价值，结合 Software Heritage、GitHub/GitLab 指标、PyPI 下载、OpenSSF 关键性指标，并以 CERN 为案例。

**💡 创新点**

创新点在于将跨平台归档数据（Software Heritage）与多维度价值评估（使用率、依赖度、重建成本、关键性得分）结合，形成可复现的、机构级别的 OSS 贡献评估方法。

**🔧 技术方法**

使用的技术包括：Software Heritage 的 GraphQL/GRPC 查询、Python 数据处理、GitHub/GitLab API 拉取统计、DaSEA 工具解析 PyPI 数据、COCOMO II 估算重建成本、PageRank 计算依赖网络中心度，以及自定义 Commit‑Criticality Score。

**📊 数据集**

使用的数据集包括：Software Heritage 历史归档（12/24/06 版）、GitHub/GitLab 公共仓库统计、DaSEA 的 PyPI 元数据与下载计数、OpenSSF Criticality Score 数据集、全球软件开发者工资数据库。

**📈 对比分析**

比较方法是把 CERN 的 commit 份额、使用指标与同类项目（如 Tensorflow、curl、Kubernetes）对比，评估其在关键性项目中的投入；性能方面显示 CERN 在 52,166 个仓库中 39,799 个占比 >1%，提交占比多为 80%+，在关键性项目中贡献率上升至 20% 以上，说明方法能准确捕捉核心贡献。

**⚠️ 局限性**

限制包括：仅基于 commit 识别，无法计入非代码活动；依赖 email 过滤导致遗漏；对 Fork/Clone 处理的阈值可能剔除合法衍生项目；使用指标受限于公开数据，无法追踪私有仓库和隐私友好平台；COCOMO 估算粗略，未考虑代码质量与长期维护成本。

---

## 285. Auto-AEG: Scalable Data Construction for Open-Vocabulary Audio Event Grounding

**arXiv ID:** 2607.04383 | [PDF](https://arxiv.org/pdf/2607.04383v1)

**作者:** Zihan Zhang `[一作]` (Zhejiang University), Tao Jin `[通讯]` (Zhejiang University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Auto‑AEG 自动化管线，生成开放词表音频事件定位训练数据，并基于此对大规模音频语言模型进行两阶段微调；同时发布 AEGBench 难度分层评测基准。

**💡 创新点**

创新点在于：①用程序化合成的音频片段提供精确的时间戳监督做 SFT 冷启动；②通过多模型伪标签和奖励式强化学习（GRPO）利用真实音频的噪声监督提升定位精度；③创建独立、难度分层的 AEGBench，避免训练与测试泄漏。

**🔧 技术方法**

技术包括程序化音频合成、Gemini、PE A-Frame、CLAP 语义清洗、多模型伪标签、Group Relative Policy Optimization（GRPO）、QLoRA 4‑bit 微调、JSON 交互式提示。

**📊 数据集**

数据集：FreeSound 语音标签子集（合成与伪标签），AEGBench 由 AudioSet、FSD50K、BBC、YouTube 取样并人工校正的 3427 条音频，DESED 作为对比任务。

**📈 对比分析**

与四个零样本基线（Gemini‑3‑Pro、Kimi‑Audio‑7B、Qwen2‑Audio‑7B、Audio‑Flamingo‑Next）相比，Auto‑AEG + SFT + GRPO 在 AEGBench 上 mIoU 分别提升 73.9%（30B）和 23.1%（7B），在 DESED 上也显著提升事件精度与召回率。

**⚠️ 局限性**

局限性：①合成数据与真实录音的域差异仍存在，SFT 冷启动可能导致偏差；②伪标签精度受 PE A-Frame 与 Gemini 的误差影响，GRPO 对噪声有容忍但未量化；③现行音频编码窗口限制（30 s）对长时段事件的定位性能仍有限。

---

## 286. Beyond Self-Resolution: Settlement Factorization for Robust Natural Language Mechanism

**arXiv ID:** 2607.04382 | [PDF](https://arxiv.org/pdf/2607.04382v1)

**作者:** Nicolas Della Penna `[一作]` `[通讯]`, Nicolas Della Penna

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为“结算分解（settlement factorization）”的机构架构，旨在让语言模型生成的咨询报告能够影响决策，却不允许同一份报告直接决定其自身的评估标注（答案键）。

**💡 创新点**

创新点在于：①将报告、公共决策、外部化评估标签和书面结论四个对象明确分离；②证明任何咨询机制都可以通过“鬼参考（ghost reference）”变换转化为分解形式，泄漏量ε_i成为不可避免且可度量的本质参数；③给出泄漏ε_i与诚实报告边际之间的精确2Lε_i损失定理，并提供最优性证明；④在冗余信息情境下展示分解如何保持非衰减的收益。

**🔧 技术方法**

使用的技术包括：信息论（总变差、互信息）、分解证明（ghost reference构造）、可测空间和条件概率的严谨定义、可计算的稳定性（L_i-稳定）分析、随机抽样与差分隐私（作为泄漏上界）以及对自我解析（self‑resolution）和混合标签的分层分析。

**📊 数据集**

论文主要为理论性工作，并未在特定公开数据集上进行实验，作者建议在未来工作中通过对真实 LLM 交互的统计实验来估计 ε_i、γ_i 等参数。

**📈 对比分析**

与现有方法比较主要基于理论指标：在已存在的正确评分规则（如 Brier、对数评分）下，若满足分解条件，诚实报告可形成信号后验均衡；通过泄漏定理可判断是否会出现自我评估操纵；在冗余信息模型中，分解方案相较于直接在公共记录上结算能保持收益不随参与者数量指数衰减。

**⚠️ 局限性**

局限性包括：①不解决评估者本身的正确性、公平性、协同与身份攻击问题；②对多代理协同支付和配额分配仅提供了补充工具而非完整解法；③在存在直接决策效用（D_i）时，分解仍需通过提高赌注或随机审计来补偿泄漏，实际成本未知；④对动态或非静态信息环境的适用性尚未验证。

---

## 287. RL Forgets! Towards Continual Policy Optimization

**arXiv ID:** 2607.04364 | [PDF](https://arxiv.org/pdf/2607.04364v1)

**作者:** Mao-Lin Luo `[一作]` (Southeast University), Tong Wei `[通讯]` (Southeast University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文重新评估了强化学习在持续视觉语言模型后训练中对灾难性遗忘的鲁棒性，提出了新的多模态推理持续学习基准 MRCL，并设计了无回放的持续策略优化框架 CPO，能够在不存储历史数据的前提下显著减少遗忘并提升预训练能力。

**💡 创新点**

创新点主要有：①构建基于 2025 年后新发布、多样化认知推理任务的 MRCL 基准；②将任务行为 KL 约束转化为参数移动正则化的可行近似，实现无回放的持续 RL；③采用稀疏 L1 运动正则和 top‑p% 参数屏蔽，平衡稳定性与可塑性。

**🔧 技术方法**

技术手段包括：强化学习（GRPO/GSPO）、行为 KL 约束、参数移动估计、稀疏 L1 正则化、掩码参数集更新、对齐指标（MFT、MFN、MTA）等；实现细节使用 LoRA、梯度估计与 Fisher 信息矩阵的对角近似。

**📊 数据集**

数据集：MRCL 组成包括 MedBookVQA、Navigation、We‑Math2.0、Puzzle、FinMME；在后置评估中进一步使用 MMMU‑Pro、MathVerse、MathVision、MathVista、RealworldQA、MMStar、POPE、DocVQA、CharXiv、CountBenchQA 等多模态基准。

**📈 对比分析**

对比方法：SFT、LoRA、O‑LoRA、SEFE、KeepLoRA、GRPO、GSPO 等。实验表明 CPO 在 Qwen3‑VL‑8B 上实现平均最终准确率 75.46%，比 GSPO 提升 13.7% 并在多模态基准上整体提升 2–5% 的准确率；在 2B、4B 规模上也保持显著优势；同时保持甚至提升了预训练模型在跨域任务上的表现。

**⚠️ 局限性**

局限性：①参数运动近似的排名一致性依赖于训练过程假设，可能无法完全捕捉 Fisher 权重；② top‑p% 选取和 λ 需要经验调优；③实验主要聚焦于 Qwen3‑VL 系列模型和 MRCL 任务，未验证在更大规模或不同 VLM 架构上的通用性；④对极端长序列或高度依赖上下文的任务效果尚未系统评估。

---

## 288. HASSL: Hierarchy-Aware Self-Supervised Learning Framework for Single Cell Microscopy

**arXiv ID:** 2607.04353 | [PDF](https://arxiv.org/pdf/2607.04353v1)

**作者:** Julius Riel `[一作]` (Technical University of Munich), Amirhossein Kardoost `[通讯]` (Helmholtz Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了HASSL框架，用于单细胞显微镜图像的自监督表示学习，旨在保留细胞形态学的层级结构。

**💡 创新点**

创新点在于：①双教师蒸馏，结合全局图像教师与基于零样本分割掩码的分割教师，引导表示更关注形态；②利用HDBSCAN层级聚类构建稳定性加权的层级对比损失，沿祖先路径拉近正样本并在兄弟子聚类上构造负样本，提升子类型分辨率。

**🔧 技术方法**

技术实现基于DINOv3 ViT骨干，采用EMA图像教师、分割教师（CellposeSAM），HDBSCAN聚类得到树形原型，使用稳定性权重的hinge对比损失；评估包括top‑K检索、mAP、NMI/AMI、下游分类与药物识别。

**📊 数据集**

使用了由20个单细胞图像基准合并而成的2.3M单细胞数据集，涵盖208类、8种成像模态；外部验证使用HPA与Allen Cell Science Perturbation数据集。

**📈 对比分析**

与Cellpaint‑DINO、ChadaViT、OpenPhenom、scDINO、HCSC及基线DINOv3等方法比较，HASSL在top‑K检索平均提升约+2.8%，在多层级数据上提升+6.3%，在药物识别任务中F1加权提升+7.8，整体性能在所有指标上均优于基线，接近最优模型。

**⚠️ 局限性**

局限性包括：需依赖分割掩码的质量，双教师训练和HDBSCAN聚类增加计算开销；对极端不同模态（如HPA）仍略逊，且层级方法假设存在可识别的层级结构，若无则效果不明。

---

## 289. How to Build Digital Humans? From Priors to Photorealistic Avatars

**arXiv ID:** 2607.04341 | [PDF](https://arxiv.org/pdf/2607.04341v1)

**作者:** Wojciech Zielonka `[一作]` (Meta), Justus Thies `[通讯]` (Technical University of Darmstadt)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述数字人头像与全身头像的构建技术，并提出统一的三阶段框架与多维度分类体系。

**💡 创新点**

通过系统梳理将头像创建、先验学习与动画三阶段进行统一映射，并基于部件、先验与表示构建多轴分类法，帮助新研究者快速定位与理解领域。

**🔧 技术方法**

主要采用文献综述、框架设计与对比分析等方法，对代表性技术（如3D Gaussian Splatting、NeRF、3DMM、生成对抗网络、扩散模型等）进行归纳与评述。

**📊 数据集**

参考了2021-2025年间在CVPR/ICCV/SIGGRAPH等会议与arXiv发布的论文，涵盖人脸、头部、手部、头发、服装等部件所使用的多种公开数据集（如FFHQ、THuman、SMPL、NeRF、NPHM、Hair2D、CLOTH等）。

**📈 对比分析**

对已有方法在重建质量、实时性、可编辑性、表达能力等方面进行了定性对比与简要性能概述，但未给出统一量化评测或基准实验。

**⚠️ 局限性**

局限在于缺乏统一的量化评测与基准，仅为理论与框架梳理；未提出新的算法或实现细节，且对部分新兴技术（如2D/3D混合生成、实时渲染）讨论相对不足。

---

## 290. Doppelganger: Sound Effects and Their Synthetic Twins

**arXiv ID:** 2607.04337 | [PDF](https://arxiv.org/pdf/2607.04337v1)

**作者:** Elliott Ash `[一作]` `[通讯]` (ETH Zurich), Elliott Ash (ETH Zurich)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Doppelganger benchmark，评估语音模型能否把音频-conditioned的合成音效与其对应的真实录音匹配，进而验证跨域表示的泛化能力。

**💡 创新点**

创新点在于：①引入 instance‑level synthetic–real 匹配任务；②发现对比学习的实例对训练能跨事件迁移，但类级无效；③提出敏感轴，用于检测特定生成器的生成质量，揭示渲染不变性与事件识别是可分离的两种能力。

**🔧 技术方法**

技术方法包括：冻结预训练编码器（CLAP、PANNs、AST 等）→ 训练不同目标的 MLP 头（实例、类别监督、敏感、无监督）；使用对比学习（supcon）、Proxy‑A、DANN、CORAL、IRM 等域适应技巧；评估指标为 mAP、R@1、MRR、AUC 等。

**📊 数据集**

使用的数据集为：DCASE‑T7（7 类）和 UCS（34 类）两部分；UCS 通过 CLAP 过滤后得到 10,420 条真实‑合成对，构成 5-fold 留类实验；同时构造 7 类控制子集。

**📈 对比分析**

对比实验显示：实例头在未见事件的完整真实图库上 R@1 约 0.80（显著高于未训练 0.61 与类别监督 0.27），但对类级检索提升不显著；敏感轴对同一生成器的 AUC 1.0，跨生成器下降；同类检索 mAP 在所有编码器上仍低于冻结基准。

**⚠️ 局限性**

局限性包括：①仅在音频-conditioned 生成器上有效，对文本/图像条件生成的合成音频无法建立对应；②跨生成器迁移差，需为每个生成器单独训练；③对类级检索无显著帮助，无法提升事件识别水平。

---

## 291. On the effectiveness of reward functions in reinforcement learning for confidence calibration of large language models

**arXiv ID:** 2607.04332 | [PDF](https://arxiv.org/pdf/2607.04332v1)

**作者:** Chee Heng Tan `[一作]` (National University of Singapore), Wee Sun Lee `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在强化学习框架下同时提升大型语言模型的推理准确性和置信度校准的方法，探讨了奖励函数设计导致的置信度奖励攻击问题并提出了非可攻击性置信奖励方案及其从过度自信到不足自信的光谱

**💡 创新点**

创新点在于首次正式定义“非可攻击性置信奖励方案”并给出其理论判据；提出置信度奖励攻击现象；构造从过度自信到不足自信的奖励光谱，展示不同方案在准确率与校准指标上的权衡

**🔧 技术方法**

主要技术包括强化学习（使用Dr GRPO）、基于正则化的置信奖励函数设计、理论分析（充分必要条件）、以及对奖励光谱的系统实验评估

**📊 数据集**

实验使用了 HotpotQA、HotpotQA-Modified、BigMath、DeepMath-103K 四个推理/数学数据集，模型为 Qwen 2.5 (3B) Instruct

**📈 对比分析**

通过对比不同奖励方案（如 Correctness-only、Overconfidence-k、Brier-1、Underconfidence-k、Brier-log Hybrid 等）的准确率、AUROC、Brier score、ECE、Calibration Bias 等指标，发现奖励方案对准确率与校准性能的影响存在显著差异，最佳方案取决于数据集与目标指标，且训练速度亦随方案变化而不同

**⚠️ 局限性**

局限性包括：评估仅针对完全正确/错误的二分类回答，未考虑部分正确或多种答案情况；奖励函数设计受 RL 稳定性影响；实验仅在单一模型规模上验证，未覆盖更大模型或多任务情景

---

## 292. Learning Task-Sufficient World Models by Synergizing Agentic Exploration and Structured Modeling

**arXiv ID:** 2607.04409 | [PDF](https://arxiv.org/pdf/2607.04409v1)

**作者:** Fan Feng `[一作]` (UCSD), Kun Zhang `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出一种闭环协同框架，通过主动探测与结构化模型学习相结合，实现世界模型的最小足够任务特定表示；

**💡 创新点**

创新点在于：①将 agentic 探索与结构化世界模型学习耦合成闭环，利用自我监督的探测技能主动获取信息增益最大的数据；②通过可学习掩码和互信息等目标对隐状态空间进行分解，得到任务最小足够子空间；③引入自适应课程调度，以任务难度为依据动态选择探索顺序；

**🔧 技术方法**

使用的主要技术包括 Dreamer‑V3 作为基线模型，基于可学习掩码的结构化表示学习、互信息与 KL 正则化；自监督技能学习（MISL、DIAYN、METRA）与 InfoNCE 对比学习；MINE 估计互信息；以及基于 UED 的自适应课程调度；

**📊 数据集**

实验数据集涵盖连续控制与机器人操作任务：DMControl（Cheetah、Walker、Reacher）、RoboSuite（Stacking 等）、Meta‑World（Kitchen、Door、Coffee‑push）等；

**📈 对比分析**

与 Dreamer‑V3、DINO‑WM、Factored Dreamer、TD‑MPC2 等基线相比，MIST 在单任务学习、样本效率、技能/组合/未见任务泛化等指标上均显著优于或接近 oracle（使用真实状态）的表现；

**⚠️ 局限性**

局限性包括：在某些难度较高的任务（如 Reacher‑Hard）效果不佳；依赖于模拟环境的任务描述与奖励信息；需要设计有效的探测技能库和课程调度策略，过度依赖这些组件可能导致泛化不足；

---

## 293. Memory-Orchestrated Semantic System (MOSS): An Auditable Agentic Memory Architecture

**arXiv ID:** 2607.04391 | [PDF](https://arxiv.org/pdf/2607.04391v1)

**作者:** Serge Lacasse `[一作]` (Université Laval), Alex Baker `[通讯]` (Université Laval)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并部署了MOSS，一个可审计、结构化、SQL驱动的长时记忆系统，用于为 AI 代理提供跨会话的持续记忆；

**💡 创新点**

创新点在于逆转传统 RAG 的向量检索，改用结构化关系数据库和查询自治，实现无向量化、可审计、可跨平台的记忆架构，并从语料自发构建概念本体；

**🔧 技术方法**

使用技术包括关系型数据库（SQLite/PostgreSQL）、SQL检索、REST/MCP 接口、LLM 无关架构、QueryProfiler、语义分段、情感标注、概念注解以及元层叠层（métacalque）等；

**📊 数据集**

使用的数据集为一位学者的工作语料，包含约 44M token、110k 对话片段、163k 文档、569 概念、5M 关系，已持续生产一年；

**📈 对比分析**

目前仅在生产环境中进行长期评估，未发布公开基准；计划与 Vector‑RAG、Mem0、Zep 等系统做 A/B 对比，已显示查询成本低于 chunked RAG，检索可解释性高；

**⚠️ 局限性**

局限性包括：仅为单用户部署，概念本体构建成本未知；多模态（音视频图像）仍在开发；查询生成仍由 LLM 决策，非完全可审计；未实现多租户隔离和防止内存污染的机制。

---

## 294. Knowledge Base Poisoning Attacks and Defense for Policy-Aware LLM-RAG Framework

**arXiv ID:** 2607.04379 | [PDF](https://arxiv.org/pdf/2607.04379v1)

**作者:** Om Solanki `[一作]` (Tennessee Tech University), Maanak Gupta `[通讯]` (Tennessee Tech University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了针对IoBT政策感知LLM检索增强生成（PA‑LLM‑RAG）框架的查询无关语义检索中毒攻击，并在此基础上设计了双重检测防御CLD‑KB，完成了攻击与防御的完整评估。

**💡 创新点**

创新点在于：①提出了无需目标查询知识即可触发的查询无关语义检索中毒攻击；②设计了利用策略分类结构的Member‑Based Category Spread与One‑Class SVM相结合的双层检测防御；③首次在LLM驱动的战场物联网任务控制中展示了此类攻击与防御。

**🔧 技术方法**

技术方法包括：向量嵌入检索、语义相似度排名、One‑Class SVM边界检测、基于类别扩散的Member‑Based Category Spread、LLM判定与自检、以及多种基线聚类/密度检测（DBSCAN、LOF、K‑Means、Isolation Forest）。

**📊 数据集**

实验使用的知识库包含60条结构化策略规则（Workflow、Rules of Engagement、Capability三类），并合成24条对抗规则；在不同中毒率（1.6%–25%）下对系统进行评估。

**📈 对比分析**

与DBSCAN、LOF、K‑Means、Isolation Forest、One‑Class SVM等基线方法对比，CLD‑KB在所有中毒率下实现了100%召回率、精准率与F1，保持0%误报，仅增加约7 ms的计算开销。

**⚠️ 局限性**

局限性包括：依赖三类策略分类结构，规则规模有限；针对性更强的攻击（如单类别或多模态攻击）仍可能突破；评估仅在仿真环境中完成，缺乏真实硬件验证。

---

## 295. Decentralized Aggregation of LLM Predictions via Wagering Mechanisms

**arXiv ID:** 2607.04389 | [PDF](https://arxiv.org/pdf/2607.04389v1)

**作者:** Yuhong Luo `[一作]` (Rutgers University), Xintong Wang `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于加权投注的去中心化大语言模型（LLM）聚合机制——WALLA，利用投注金额来捕捉各模型的相对优势，实现可信的预测汇聚。

**💡 创新点**

创新点：① 引入留一基线（leave‑one‑out）和正则化项，使得投注与模型预期优势对齐；② 在任意信念结构下保持主导策略激励兼容性（DSIC）；③ 将预测与投注解耦，允许预测自由而不影响投注学习；④ 提供两种基线变体，平衡正常性与无套利性，并证明最坏情况赤字有界。

**🔧 技术方法**

技术方法：加权分数投注机制、严格正则化的净支付函数、Brier或对数分数规则、线性/对数池化、投注网络（小型MLP）通过经验支付学习最优投注。

**📊 数据集**

实验数据集：MMLU、MedMCQA、ARC‑Challenge、PubMedQA 以及自建的 BayesX 预测基准，涵盖多领域问答与事件预测。

**📈 对比分析**

与基线比较：与统一平均、基于置信度或困惑度的权重、预推理路由器、StackedGen 以及自家两种变体相比，WALLA 在准确率、AUC、Kendall τ、MRR、D‑Regret 等指标上与中心化方法持平或优于多数基线，且能在去中心化环境下实现稳健的权重学习。

**⚠️ 局限性**

局限性：① 在完全可学习的环境下可能收敛至“无交易”极限，需通过机制设计或环境扰动维持多模型活跃；② 对公共先验的假设与正则化参数 c3 的设定敏感；③ 需要足够的样本来估计优势，低样本或高度动态场景下性能可能受限。

---

## 296. Do GUI Agents Believe Their Eyes? Diagnosing State-Belief Reliance on Pixels versus Structure

**arXiv ID:** 2607.04334 | [PDF](https://arxiv.org/pdf/2607.04334v1)

**作者:** Guijia Zhang `[一作]` (Shenzhen University), Harry Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了测量多模态 GUI 代理对像素与结构（DOM 等）信任程度的指标 Perception‑Fusion Gap，并基于此构建了 310 条真实数据对抗性探测样本，评估多模态代理在图像和结构冲突下的信念来源。

**💡 创新点**

创新点在于：①首次将“信念来源”概念量化为 Perception‑Fusion Gap；②设计了对抗性单通道编辑实验，精确定位模型在文字状态下先读像素再被结构覆盖的“替代”行为；③通过白盒嵌入消融证明结构字符串复制是导致文本状态偏移的因果因素；④提出了无训练一致性门（consistency gate）可在运行时校正结构偏移。

**🔧 技术方法**

技术主要包括：对输入的像素（截图）和结构（DOM/可访问性树）进行双通道推理；对每个 probe 进行 6 种受控干预（agreement、structure‑swap、pixel‑swap、re‑render、pixels‑only、structure‑only、no‑evidence）；使用强制式 JSON 选项解码；对模型输出自报来源进行一致性分析；在两个真实环境（AndroidWorld、MiniWoB++）中检验错误信念对动作的影响；以及实现了基于裁剪区像素校验的一致性门。

**📊 数据集**

数据集包括：Web：Multimodal‑Mind2Web；Mobile：RICO + CLAY 注释；Desktop：ScreenSpot‑Pro；所有探测均基于真实截图和对应结构，冲突编辑为规则式或天然冲突（如 CLAY 标记的无效节点）。

**📈 对比分析**

比较方法：对 5 种模型（Qwen2.5‑VL‑7B、InternVL3‑8B、gpt‑5.4、gpt‑4o、gpt‑5.4‑nano）在 310 条探测上进行一致性与信念来源评估，报告像素跟随率、结构跟随率及 Perception‑Fusion Gap。结果显示所有模型在文字状态下的 Perception‑Fusion Gap 均为正，数值从 0.30 至 0.75；非文字身份保持像素跟随；结构仅读者可达 0.94。与单通道读者对比，融合模型介于两者之间。

**⚠️ 局限性**

局限性包括：评估仅为单步点击行为，无法覆盖多步决策链；金标准依赖两名像素注释者（κ=0.83）；白盒消融仅在开放模型上可行；天然冲突样本规模有限；模型的训练动态可能影响结果；一致性门为自检，未解决结构仅有的威胁。

---

## 297. Framework and Multi-modal Dataset for Roadwork Zone Detection and Geo-localization

**arXiv ID:** 2607.04330 | [PDF](https://arxiv.org/pdf/2607.04330v1)

**作者:** Zhiran Yan `[一作]` (Institute of Innovative Mobility), Gordon Elger `[通讯]` (Fraunhofer Institute for Transportation and Infrastructure Systems)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了针对道路施工区域的检测与全球定位（geo‑localization）框架，并创建了包含真实与仿真数据的多模态 RZDG 数据集。

**💡 创新点**

创新点包括：①首次公开包含真实与仿真相结合、图像、LiDAR 与 GPS/IMU 传感器的道路施工数据集；②将 AB3DMOT 追踪器与 3D 检测器相结合，提出轨迹融合（Last‑Frame / Weighted‑Average）实现单一对象的全球定位；③在精度评估中采用 Haversine 公式，将定位误差严格限定在 1 m 内。

**🔧 技术方法**

技术方法：3D 检测器（SMOKE、PointPillars、MVXNet）+ 3D 卡尔曼滤波器+匈牙利算法追踪；坐标变换基于 GPS/IMU 与相机外参；语义分割算法（Deeplabv3+、Swin、Segformer）用于区域识别；数据评估使用 AP、mIoU、精确率/召回率/ F1‑score。

**📊 数据集**

使用了自研 RZDG‑Real（真实场景）和 RZDG‑Sim（CARLA 仿真）两套数据集，包含 RGB、LiDAR 点云、GPS/IMU、语义分割掩码、3D 边框及全球坐标标签。

**📈 对比分析**

在真实数据上，3D 检测 AP 约 32–57%；在仿真数据上 AP 超过 93%；geo‑localization F1‑score 真实约 0.60、仿真约 0.66；语义分割 mIoU 在真实场景为 93%/91%/88%，仿真场景 84%/84%/72%。相较于基线（仅使用 GT 或 Last‑Frame 方法），我们的方法在精确度和召回率上均有显著提升。

**⚠️ 局限性**

局限性：①数据集规模有限，且仅包含障碍物与路标两类静态对象；②依赖高精度 RTK GPS，实际部署在 GPS 信号弱的地区可能失效；③轨迹融合对对象运动性有限，仅适用于静止或缓慢移动的施工物体；④仿真数据与真实场景仍存在分布差异，模型泛化能力待进一步验证。

---

## 298. HAS-Bench: Evaluating LLM-Based Human-Agent Systems under Configurable Human Participation

**arXiv ID:** 2607.04329 | [PDF](https://arxiv.org/pdf/2607.04329v1)

**作者:** Yaozu Wu `[一作]` (University of Tokyo), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HAS-Framework 和 HAS-Bench，构建可配置人机协同的评估框架和基准。

**💡 创新点**

创新点在于把人类和 LLM 代理统一为图中的节点，定义了三种交互通道（clarification、feedback、control）和可调节的人机权限层级，并在基准中同时测量任务结果与过程级合作行为。

**🔧 技术方法**

采用图模型框架、三通道交互设计、A1–A5 人机权限尺度、LLM 语言与工具调用、Prompt 工程及 LLM 判定与人工验证的评估流程。

**📊 数据集**

使用从六个领域（零售、电信、航空、编码、研究、谈判）现有基准筛选改造得到的 397 个任务，覆盖六种问题模式。

**📈 对比分析**

通过在不同人机参与层级（A1、A3、A4）下，使用 GPT‑4.1、GPT‑4.1‑mini、Claude‑Sonnet‑4、DeepSeek‑V3、Llama‑3.1 评估 Pass@1、Task Score、Safety、HAS‑RR 等指标；结果显示人机协同平均提升 8.4% 的 Pass@1，但提升幅度随模型能力与参与形式而异。

**⚠️ 局限性**

局限性：人机协同效果高度依赖模型对人类输入的理解与整合能力；不恰当的干预可能导致性能退化；基准使用 LLM 作为用户模拟，缺乏真实人类交互；评估成本高且对不同人类行为的覆盖有限。

---

## 299. LogicProof: An Interactive Web-Based Educational Theorem Prover for Natural Deduction and Sequent Calculus across Classical and Constructive Logics

**arXiv ID:** 2607.04321 | [PDF](https://arxiv.org/pdf/2607.04321v1)

**作者:** Ján Perháč `[一作]` (Technical University of Košice), Samuel Novotný `[通讯]` (Technical University of Košice)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

开发了一款名为 LogicProof 的交互式网页定理证明器，用于教学逻辑，支持自然演绎与序列演算，实时验证推理规则并以树形可视化展示证明过程。

**💡 创新点**

创新点在于将轻量级的 Vanilla JS 单页应用与高性能 FOL 解析器、即时推理验证、交互式树视图结合，并支持经典与建构逻辑切换以及多种形式论证理论的集成。

**🔧 技术方法**

技术栈包括 Vanilla JS、HTML5/CSS3、D3.js、Monaco Editor、ANTLR4（TypeScript）、Webpack/Babel、Driver.js 等，全部实现于客户端，避免服务器延迟。

**📊 数据集**

未使用公开数据集；评估依赖于 35 名大学生的实验数据，学生自行输入的逻辑公式与示例证明。

**📈 对比分析**

通过 SUS 问卷和两项任务情景（基础交互与复杂分支）与传统手工证明对比；SUS 分数 79.5，任务成功率分别为 91% 与 83%；用户反馈表明迭代速度提升、错误纠正更直观、整体体验良好。

**⚠️ 局限性**

局限性包括界面导航与提示不够直观、对大树结构的渲染与交互尚待优化、缺乏自动化提示与自适应教学、未实现与学习管理系统的无缝集成，以及在极端复杂证明时性能下降。

---

## 300. Aura: Consistent Multi-Subject Video Generation via VLM-Grounded Semantic Alignment

**arXiv ID:** 2607.04311 | [PDF](https://arxiv.org/pdf/2607.04311v1)

**作者:** Zixiang Zhou `[一作]` (Tencent Hunyuan), Qinglin Lu `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一种名为Aura的统一扩散变换器，支持多主体视频生成，并能在保持身份一致性、自然运动和多元素场景的同时实现高保真度；

**💡 创新点**

创新点包括：双流语义注入（T5与Qwen2.5‑VL）并通过T5教师对齐；参考注入采用固定槽、学习令牌与Subject‑Aware RoPE偏移；四阶段训练与norm‑only progressive APG；以及基于AIGC的 grounding‑augmenting‑verification 数据管道；

**🔧 技术方法**

使用技术包括 Diffusion Transformer (DiT)、T5、Qwen2.5‑VL、共享 KV 交叉注意、InfoNCE 与 Hungarian 匹配、RoPE、Adaptive Prompt Guidance、FSDP 并行训练、VLM 评估等；

**📊 数据集**

数据集为约 15M 条 clip‑级训练样本，来源于电影、电视剧、短视频，包含人、物、场景三类参考；测试集为 50 条手工构造案例；

**📈 对比分析**

与 Wan2.2 DiT、HuMo、Kaleido、MAGREF、RefAlign 等 5 个 SOTA 基线在 OpenS2V‑Eval 与 Gemma4‑31B VLM 评测上对比，Aura 在 Total Score、FaceSim、NexusScore、NaturalScore 等指标上均领先，整体性能显著优于竞争者；

**⚠️ 局限性**

局限性包括：对参考图像质量（遮挡、运动模糊）敏感；多主体时可能出现身份漂移；模型规模大、训练/推理成本高；对极端摄像机运动和长时序生成尚未彻底解决。

---

## 301. AquaStereo: Enabling Underwater Stereo Matching via Depth-Conditioned Diffusion and Geometry Self-Distillation

**arXiv ID:** 2607.04303 | [PDF](https://arxiv.org/pdf/2607.04303v1)

**作者:** Qizhe Wei `[一作]` (Beijing Institute of Technology), Ying Fu `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 AquaStereo 框架，结合深度条件扩散生成、跨域自蒸馏以及感知增强匹配器，实现对水下立体匹配的零样本鲁棒推断。

**💡 创新点**

创新点包括① 通过物理启发的文字提示与深度控制的扩散模型，生成保持几何一致性的逼真水下立体图像；② 利用冻结教师与扰动水下数据的跨域自蒸馏，传递清晰域的几何知识；③ 在匹配器中引入可学习的感知帧与 DINOv2 语义融合，提升在浑浊与低纹理环境下的特征稳定性。

**🔧 技术方法**

使用 Stable Diffusion + ControlNet 进行深度条件生成；采用自蒸馏框架（教师-学生无共享权重，配合清晰分支监督）；配备视频编码器和 DINOv2 语义编码器的感知增强特征提取器；在后端采用 IGEV++ 立体匹配网络。

**📊 数据集**

训练集为 SceneFlow 与 KITTI 的大规模清晰立体图像，利用生成的 UW‑Dataset（约 40K 对）做合成水下数据；评估数据包括 UWStereo、FLSea、Squid、TartanAir 等真实与仿真水下基准。

**📈 对比分析**

与 PSMNet、CFNet、GwcNet、IGEV、IGEV++、LightStereo、StereoAnything、NMRF、MonSter、FoundationStereo 等现有立体匹配方法对比，AquaStereo 在 UWStereo 上实现 EPE 0.59、D1 2.61 的最佳总量级，并在 FLSea、Squid、TartanAir 等基准上持续领先，验证了其零样本泛化与精度提升。

**⚠️ 局限性**

局限性在于仍需大量高质量的陆地立体数据做伪标签，生成模型对极端光照或透明物体的逼真度有限；自蒸馏对扰动设计敏感，计算开销相对较高；且框架主要针对校准好的 rectified 立体图像，难以直接迁移至非校准或多相机场景。

---

## 302. Beyond Monotone Delays for Multi-Level Aggregation

**arXiv ID:** 2607.04317 | [PDF](https://arxiv.org/pdf/2607.04317v1)

**作者:** Yossi Azar `[一作]` (Tel Aviv University), Liad Iluz `[通讯]` (Tel Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在线多层聚合问题（MLAP）在任意惩罚函数下的竞争性，提出了一种随机化算法并给出了理论上最优的竞争比与下界；

**💡 创新点**

创新点在于打破传统仅考虑单调惩罚函数的限制，允许惩罚函数随时间多次增减，并将问题推广到任意树深度；

**🔧 技术方法**

核心技术包括对惩罚函数的离散化、三阶段归约（MLAP→2递减树→离散MLAP→增量多播问题）以及改进的随机化分数与整形方案；

**📊 数据集**

本研究为理论分析，未使用实际数据集，而是通过算法设计与证明验证性能；

**📈 对比分析**

通过与已有常数竞争算法和下界对比，算法获得了 O(D·log n·log(nDW)) 的竞争比，并证明任何确定性算法至少需要 4 倍竞争比；

**⚠️ 局限性**

主要局限在于竞争比仍与下界存在较大差距，对极端惩罚函数的性能尚未达到最优，且算法实现复杂度高。

---

## 303. Structure-Specific Representational Priors Causally Control the Grokking Delay

**arXiv ID:** 2607.04333 | [PDF](https://arxiv.org/pdf/2607.04333v1)

**作者:** Gunner Levi Howe `[一作]` `[通讯]`, Gunner Levi Howe

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在Transformer学习模数加法任务时，grokking（长时间后突然泛化）与表示结构形成的因果关系，并通过监督对比损失注入不同结构先验来验证其可操控性。

**💡 创新点**

首次以结构特定的方式介入grokking过程，证明仅“正确”任务等价结构能显著缩短泛化延迟，证明延迟是由特征层面的表示结构形成决定的。

**🔧 技术方法**

使用单层解码器Transformer、全批量AdamW优化、监督对比学习（SupCon）辅助损失、权重范数匹配对照、Grokfast梯度过滤、Fourier分析、CKA等表示测量技术。

**📊 数据集**

数据集为模数加法任务（p=97），共有9409个输入对，30%为训练集，余下为测试集。

**📈 对比分析**

与传统的无结构干预（Grokfast）以及权重范数匹配对照相比，注入真结构在大部分跑次能在约2000个epoch内达到95%测试准确率（比基线快2.75×），错误或随机结构则完全不泛化；权重范数匹配对照在任何norm下都无法泛化。

**⚠️ 局限性**

实验仅限于单层Transformer与单一模数加法任务，缺乏对更大模型和自然数据的验证；对比损失引入额外计算开销，且结构特定的加速具有“陷阱”概率，未能在所有种子中稳定加速。

---

## 304. The ABC of digital health: A framework for translating digital health interventions into real-world applications

**arXiv ID:** 2607.04381 | [PDF](https://arxiv.org/pdf/2607.04381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 305. Road-Aware Anomaly Segmentation with Query-Guided Polygons and CLIP in Autonomous Driving

**arXiv ID:** 2607.04304 | [PDF](https://arxiv.org/pdf/2607.04304v1)

**作者:** Zhiran Yan `[一作]` (Institute of Innovative Mobility), Gordon Elger `[通讯]` (Fraunhofer Institute for Transportation and Infrastructure Systems)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用冻结的Mask2Former模型，在推理时通过查询级掩模和道路多边形先验实现无训练、无OOD数据的异常分割；

**💡 创新点**

在Mask2Former的基础上加入道路空间先验和CLIP零样本语义过滤，形成轻量化后处理管线；

**🔧 技术方法**

结合Mask2Former的查询掩模、接受/拒绝软掩模、道路多边形生成与CLIP零样本分类；

**📊 数据集**

在Fishyscapes（LostAndFound、Static）、Segment‑Me‑If‑You‑Can（Anomaly、Obstacle）和RoadAnomaly三个公开数据集上评估；

**📈 对比分析**

相较于Maskomaly基线，在Fishyscapes LostAndFound AP提升至75.01、FS‑Static AP提升至65.35，SMIYC‑Obstacle AUPR+4.68，RoadAnomaly AP+2.09，整体性能与有重训练或OE方法相近或更优；

**⚠️ 局限性**

仅依赖固定的道路查询掩模，可能在非道路或多变道路场景下效果不佳，且未在全场景多样化的OOP环境中充分验证。

---

## 306. Time Series Decomposition using the Fréchet Distance

**arXiv ID:** 2607.04397 | [PDF](https://arxiv.org/pdf/2607.04397v1)

**作者:** Anne Driemel `[一作]` (University of Bonn), Christian Sohler `[通讯]` (University of Cologne)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Fréchet距离下的信号分解问题——将一组一维时序分解为少量基曲线的“Fréchet组合”，并给出了单基曲线的近似与多基曲线的精确求解算法。

**💡 创新点**

首次把Fréchet距离引入主成分分析类分解框架，解决时序的时间变形问题；提出基于遍历矩阵与超平面排列的组合方法，实现(1+ε)近似与精确求解。

**🔧 技术方法**

离散Fréchet距离、遍历矩阵、动态规划、超平面排列、线性规划、简化（simplification）技术、维数压缩与基准算法的组合。

**📊 数据集**

论文中未使用具体实验数据集，所有结果均为理论证明与算法复杂度分析。

**📈 对比分析**

与现有基于Fréchet聚类或DTW的方法相比，作者给出了在k=1时的(1+ε)近似（时间O(n^2m+nm)）以及在基曲线来自有限候选集时的精确求解（时间O(|C|^k n m^{2k+2.5})），展示了在理论层面的效率提升。

**⚠️ 局限性**

对k>1的近似仍停留在指数或高多项式时间；缺乏实验验证，无法评估在实际时序数据上的表现；对输入时序长度与基曲线长度的假设（常数）限制了通用性。

---

## 307. WPG-MoE: Weak-Prior-Guided Dense Mixture-of-Experts for User-Level Social Media Depression Detection

**arXiv ID:** 2607.04350 | [PDF](https://arxiv.org/pdf/2607.04350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 308. NKI-Agent: Domain-Specific Fine-Tuning and Agentic Tool Use for Neuron Kernel Generation

**arXiv ID:** 2607.04395 | [PDF](https://arxiv.org/pdf/2607.04395v1)

**作者:** Junjie Tang `[一作]` (Amazon Web Services), Lin Wang `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了NKI-Agent系统，利用域特定的监督微调（SFT）结合编译‑验证‑修复循环的工具驱动代理，自动生成适用于AWS Trainium/Inferentia Neuron Kernel Interface 的高性能计算核。

**💡 创新点**

首创针对Neural Kernel Interface的域特定SFT与代理工具链；构建6000个核生成任务和250任务难度基准；并证明工具使用对解决复杂（L2/L3）任务至关重要。

**🔧 技术方法**

采用大型语言模型（Qwen3-Coder-30B-A3B）进行SFT，随后使用GRPO强化学习；集成编译工具、验证工具和rank‑aware系统提示；改造CUDA-Agent框架以支持Neuron SDK。

**📊 数据集**

使用NKI‑Agent‑Ops‑6K 6000条精心挑选的任务作为训练集，另有250条基准任务（分为L1、L2、L3三难度）；在SFT阶段精炼了4轮高质量数据集，共647个episode。

**📈 对比分析**

在真实Trn1硬件上通过pass‑rate评估性能：SFT+NKIAgent在60任务上达到25%（≈1/100成本）/20.7%（150任务）；Claude Opus 4.8加工具件则达到63.3%/77.3%，显示代理工具在复杂任务中的巨大价值。

**⚠️ 局限性**

局限性包括：仅评估功能正确性未考虑运行时性能；GRPO使用二元奖励导致无提升；缺乏统计显著性分析；结果受特定SDK版本限制；未探究更高级奖励设计或更大模型的潜力。

---

## 309. SurgAM: Surgical Affordance Map Prediction with Multimodal Feature Fusion for Robot Autonomy

**arXiv ID:** 2607.04378 | [PDF](https://arxiv.org/pdf/2607.04378v1)

**作者:** Lei Song `[一作]` (Chinese University of Hong Kong), Qi Dou `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种双模（自监督视觉 Transformer 与扩散生成模型）融合的外科手术动作可行性（affordance）预测框架，并构建了包含三类基础外科动作（吸引、剪切、牵引）的新数据集，验证了该框架在仿真与真实体型模型上的可行性。

**💡 创新点**

首次在外科机器人领域实现可行性地图预测，创新性地将 DINOv2 的语义精确特征与 Stable Diffusion 的空间连贯特征进行自适应融合，并引入层次化提示学习与场景引导注意力机制，形成可解释的感知‑动作桥梁。

**🔧 技术方法**

采用 DINOv2 自监督 Vision Transformer、Stable Diffusion 扩散生成模型、层次化提示学习（CoOp 扩展）、场景引导交叉注意力解码器以及 VPPV 视觉伺服控制框架。

**📊 数据集**

使用七个公开外科视频数据集（AutoLaparo、CholecT50、HeiChole、MultiBypass140、SurgicalActions160、Endovis18、MESAD-Real）拼接而成的新外科可行性数据集，标注了 Retraction、Clipping、Aspiration 三类动作。

**📈 对比分析**

与 Cross-View-AG、LOCATE、WorldAfford、OOAL 等多种基线进行对比，采用 KLD、SIM、NSS、CLA 四项指标。实验显示本方法在 KLD 上下降至 1.362（相比最佳基线 1.559 下降约 12%），SIM 提升至 0.367（最高），NSS 达 1.642（最高），CLA 达 0.895（略高）。

**⚠️ 局限性**

仅在静态图像、离线仿真和体型模型上验证，缺乏对实时动态手术场景的评估；数据集仍有限，未考虑时序信息；模型对光照变化、遮挡和组织变形的鲁棒性仍待提升。

---

## 310. Nemotron-Labs-3-Puzzle-75B-A9B: Compressing Hybrid MoE LLMs

**arXiv ID:** 2607.04371 | [PDF](https://arxiv.org/pdf/2607.04371v1)

**作者:** Akhiad Bercovich `[一作]`, Ran El-Yaniv `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Nemotron-3-Super 上使用迭代 Puzzle、结构压缩、知识蒸馏、强化学习、量化和多令牌预测等技术，生成了更小、更高效的 Puzzle-75B-A9B 变体。

**💡 创新点**

创新点在于引入 Iterative Puzzle 的序列压缩过程，逐步重构网络并在每一步进行短期蒸馏，使压缩更具硬件感知且层级非均匀；同时结合 MoE 与 Mamba 的异构剪枝和多令牌预测，实现高效部署。

**🔧 技术方法**

主要技术包括 Puzzle/Iterative Puzzle NAS、异构 MoE 与 Mamba 剪枝、知识蒸馏（短/长上下文）、强化学习微调、FP8/NVFP4 量化、Speculative Decoding / Multi-Token Prediction。

**📊 数据集**

数据集：30% 预训练数据 + 70% Nemotron-3-Nano 监督微调数据；长上下文蒸馏使用 128K-512K 序列；评测 benchmark 包含 MMLU-Pro、AIME25、HMMT、GPQA、LiveCodeBench、SciCode、RULER、WMT24++、IFBench、Arena-Hard-V2、TauBench、SWE-Bench、Terminal Bench 等。

**📈 对比分析**

通过在同一硬件（8×B200、8×H100、单 H100）上进行 Pareto 前沿搜索，比较用户吞吐量、总吞吐量和有效请求完成率；Puzzle-75B-A9B 在 UT=100 tok/s 时约提升 2.18× 总吞吐量，MTP 后可达 4.85× Super；在 1M-token 单 GPU 上从 1 并发提升到 8 并发。

**⚠️ 局限性**

局限性：压缩后在某些细粒度任务（如特定指令遵循、部分代理评测）表现略逊；量化后长上下文精度略下降；RL 贡献有限；不同硬件上的量化策略差异；对极大压缩率的可扩展性仍待验证。

---

## 311. One Framework for All: Cross-Modal Membership Inference for Generative Models

**arXiv ID:** 2607.04339 | [PDF](https://arxiv.org/pdf/2607.04339v1)

**作者:** Dayong Ye `[一作]`, Wanlei Zhou `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种跨模态的统一成员推断攻击框架，适用于文本到文本、文本到图像和图像到文本生成模型。

**💡 创新点**

创新点在于利用生成模型输出分布逼近训练数据分布的模态无关性质，使用似然比检验进行成员判断，并在黑盒下实现零知识和部分知识两种攻击。

**🔧 技术方法**

核心技术包括基于嵌入空间的分布估计（均值、协方差）、多模态特征提取器和似然比阈值0的决策规则。

**📊 数据集**

使用了GPT‑2、Falcon LLM；Stable Diffusion v1.5/v2.1；LLaVA‑7B、MiniGPT‑4等模型，并在Wiki‑103、XSum、MS‑COCO、CelebA‑Dialog、COCO‑2017、CC_SBU等数据集上进行微调与评估。

**📈 对比分析**

与专门针对单一模态的基线（SPV‑MIA、ICP‑MIA、Score‑MIA、CLiD‑MIA、Temperature‑MIA、MaxRényi‑K%）比较，显示在零知识场景下显著提升ASR/AUC，部分知识场景保持竞争力。

**⚠️ 局限性**

局限性包括对大型嵌入提取器依赖、对图像模态的分布估计敏感、在极端低样本量或高噪声环境下性能下降，以及DP‑SGD等隐私防御会显著削弱攻击效果。

---

## 312. Server-side Anti-cheat in FPS games for Aimbot detection using Deep learning and Machine learning

**arXiv ID:** 2607.04336 | [PDF](https://arxiv.org/pdf/2607.04336v1)

**作者:** Siddhesh A. Dhinge `[一作]` (Pune Institute of Computer Technology), Jyoti H. Jadhav `[通讯]` (Pune Institute of Computer Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并实现了基于时间序列的服务器端反作弊系统YAACS，专门检测CS:GO中的瞄准辅助（aimbot）作弊行为。

**💡 创新点**

创新点在于将LSTM、堆叠LSTM+Dense等深度序列模型应用于游戏行为序列，系统性比较不同时间窗口的数据集，显著降低误报率，并提出以误报率为核心的评估框架。

**🔧 技术方法**

使用LSTM、堆叠LSTM、LSTM+Dense、决策树基线进行建模；Python‑demo‑parser 提取特征；RabbitMQ、MongoDB、S3 等后端架构；特征工程提取时间序列、视角、击中、移动等多维度行为数据。

**📊 数据集**

从CS:GO demo文件生成六个不同时间窗口的数据集（Tick Delta Negative/Positive 与 Max Fight Size 变化，序列长度从128到1），用于训练和评估模型。

**📈 对比分析**

通过与决策树基线比较，评估准确率、精确率、召回率和误报率（FPR）；最佳LSTM模型在Dataset 1上达到88.6%准确率，误报率0.97%，比决策树的2.68%低2.76倍；决策树召回率更高但误报率显著更高。

**⚠️ 局限性**

局限性包括：召回率仍低于决策树；对极短窗口（1 tick）模型失效；仅在CS:GO aimbot 上验证，缺乏跨游戏泛化；未探索注意力机制或混合模型来进一步提升召回率。

---

## 313. Using OAI Overlay to Enhance REST API Fuzzing

**arXiv ID:** 2607.04325 | [PDF](https://arxiv.org/pdf/2607.04325v1)

**作者:** Omur Sahin `[一作]` (Erciyes University), Andrea Arcuri `[通讯]` (Kristiania University College)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在工业环境下，作者将最新的REST API模糊测试工具扩展为原生支持OpenAPI Overlay文件，使测试人员能在单独的Overlay文件中提供示例数据，并在五家企业的五个API上进行实验，验证Overlay对黑盒模糊测试的有效性。

**💡 创新点**

首次提出并验证使用OAI Overlay作为输入示例的标准化方式来提升REST API黑盒模糊测试效果，并实现了fuzzer的原生Overlay支持，解决了示例注入的分离、可维护性和跨工具兼容性问题。

**🔧 技术方法**

利用OpenAPI 3.1.0规范、OAI Overlay 1.1.0、Java实现的overlay-jvm库、EvoMaster模糊测试器以及WFC Web报告工具；实验中还使用LLM辅助生成Overlay文件。

**📊 数据集**

五个来自不同地区（德国、美国、中国、土耳其、比利时）的企业提供的工业API，共计约23个端点，未公开具体API细节。

**📈 对比分析**

将每个API分别在不使用Overlay和使用Overlay的情况下跑10分钟，比较2xx状态码覆盖率和检测到的故障数；实验显示Overlay能显著提升覆盖率和故障检测，改进幅度随API和示例质量不同而异。

**⚠️ 局限性**

提升效果高度依赖示例的质量和数量，Overlay并不能解决fuzzer本身或API的缺陷；对大规模端点集的可扩展性仍需进一步验证，且实验样本规模有限，外部可推广性存疑。

---

## 314. Mechanism Design for Locating a Bridge Between Regions with Prelocated Facilities

**arXiv ID:** 2607.04309 | [PDF](https://arxiv.org/pdf/2607.04309v1)

**作者:** Genjie Qin `[一作]` (Ocean University of China), Wenjing Liu `[通讯]` (Ocean University of China)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在两条平行线（两区）上预置设施的桥梁定位问题，目标是让代理人能够自由选择最近的设施，从而在最大成本和社会成本目标下设计策略不变、群体策略不变和强群体策略不变的机制。

**💡 创新点**

创新点在于将传统的设施定位框架扩展到具有障碍物的双区环境，首次系统性地给出在不使用金钱激励的前提下，桥梁定位的近似最优方案与下界，并证明了在不同策略约束下可实现的最佳近似比；同时提出了可实现强群体策略不变的三倍和两倍近似机制。

**🔧 技术方法**

主要技术包括几何简化（将两区视为平行线、桥梁为零代价垂直连线）、对代理人位置的分区分析、构造对称且可证明的确定性/随机化机制、严格的策略不变性与群体策略不变性证明，以及使用最优解与近似解之间的成本比较得到的上下界。

**📊 数据集**

本研究为理论工作，未使用任何实验数据集，全部结果通过数学证明和理论分析获得。

**📈 对比分析**

通过理论分析比较，确定性 GSP 机制在最大成本与社会成本上分别实现 3 倍近似，随机化 GSP 机制实现 2 倍近似；在强群体策略不变下，确定性机制实现 3 倍近似，随机化实现 2 倍近似；下界证明表明在相同约束下更优的近似比不可实现。

**⚠️ 局限性**

局限性包括：理论结果与实践的距离（未考虑桥梁长度或通行费）；设施同质假设与实际异质设施差异；仅针对二维平行线模型，无法直接推广至更复杂的网络或任意度量空间。

---

## 315. SAD-LoRA: Spectral Alignment for Low-Rank Knowledge Distillation

**arXiv ID:** 2607.04306 | [PDF](https://arxiv.org/pdf/2607.04306v1)

**作者:** Omer Tariq `[一作]` (Neubility Inc), Jeongbae Son `[通讯]` (Neubility Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在知识蒸馏过程中通过谱对齐来控制LoRA低秩适配器子空间的训练方法——SAD-LoRA；

**💡 创新点**

创新点在于将教师更新的谱信息（数据加权的左奇异子空间）作为子空间对齐目标，并在训练期间使用可微分的主角距离损失显式保持子空间一致；

**🔧 技术方法**

使用LoRA低秩适配、谱分解（SVD）、主角距离（Grassmannian）对齐损失、系数匹配损失以及温度缩放的KL蒸馏；

**📊 数据集**

在RoBERTa-large对RoBERTa-base的GLUE六项任务（SST-2、MRPC、STS-B、CoLA、QNLI、RTE）上进行实验，并用合成数据验证误差分解；

**📈 对比分析**

与标准KD+LoRA、Logit-MSE、NN-Init、PiSSA-Init等基线比较，SAD-LoRA在低秩（r=4、8）下的性能优于或匹配基线，尤其在STS-B和CoLA任务上显著提升；

**⚠️ 局限性**

局限性包括需额外计算同构教师的权重更新以得到谱目标、对校准数据的依赖、在某些语义匹配任务（如MRPC）中不一定优于传统权重初始化，且目前仅验证了编码器蒸馏，尚未扩展到解码器或生成任务。

---

## 316. HiFA4: Training-Free 4-bit FlashAttention on Ascend HIF4 NPUs for LLM Inference

**arXiv ID:** 2607.04302 | [PDF](https://arxiv.org/pdf/2607.04302v1)

**作者:** Hui Dong `[一作]` (Huawei Technologies), Zhiqiang Zou `[通讯]` (Huawei Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Ascend NPU上实现HiFA4，将FlashAttention中的QK^T和PV矩阵乘法量化为4‑bit HIF4 GEMM，保持在线softmax为FP16；

**💡 创新点**

创新点在于：1）Smooth‑QK通过校准静态通道缩放消除K激活异常，避免在线归约；2）P‑Reordering证明并消除归一化与GEMM不一致产生的协同误差，并将归一化融入PV GEMM；3）Q‑Mean辅助校正；

**🔧 技术方法**

技术手段包括HIF4 4‑bit量化、每通道静态缩放、P‑Reordering、Q‑Mean补偿、门控选择（依据ρ_K与T_K/Q阈值）及FlashAttention的张量/立方路径融合；

**📊 数据集**

使用Qwen3‑8B、Gemma2‑9B、LLaMA3.1‑8B、Mistral‑7B、Phi‑4B等模型，并在MMLU、HellaSwag、ARC、TruthfulQA、LongBench等NLP基准上评测；

**📈 对比分析**

与BF16基线及直接HIF4量化对比，HiFA4将MMLU量化误差从1.12pp降至0.70pp，预测翻转率减半，量化损失下降57%；在Gemma2‑9B上保持误差<0.7pp；理论上可实现35.4%关键路径延迟缩减；

**⚠️ 局限性**

局限性包括：仅提供理论延迟预测，硬件验证待Ascend NPU公开；门控阈值基于仅五个模型经验，未必通用；Smooth‑QK仅对具有K激活异常的模型有效；未涵盖更大模型集及不同硬件平台；

---

## 317. How Many Initial Points Does Bayesian Optimization Need?

**arXiv ID:** 2607.04356 | [PDF](https://arxiv.org/pdf/2607.04356v1)

**作者:** Mujin Cheon `[一作]` (Korea Advanced Institute of Science & Technology), Calvin Tsay `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了贝叶斯优化中初始采样点数 n0 对总实验成本的影响，揭示了 U 形曲线。

**💡 创新点**

创新点在于发现 n0 对方差驱动采样函数（EI、UCB、PI）敏感，而 Thompson Sampling 对 n0 不敏感；并将该现象归因于边界探索路径问题，提出可通过多步前瞻或使用 TS 来缓解。

**🔧 技术方法**

使用高斯过程（GP）模型、方差驱动采样函数（EI、UCB、PI）、Thompson Sampling，以及多步前瞻贝叶斯优化；对比不同超参数设定（MLE、Bayesian MCMC、oracle）。

**📊 数据集**

实验数据集包括离散 3D 网格 {−5,…,5}^3（约 1331 点）和 9^4=6561 点的 Ackley 4D 网格。

**📈 对比分析**

通过多组实验比较不同 n0 和采样函数的总成本 C(n0) 与简单 regret；结果显示方差驱动函数呈 U 形，TS 与 n0 几乎无关；在 oracle GP 下多步前瞻能平滑 U 形，提升性能。

**⚠️ 局限性**

局限在于实验仅覆盖离散网格和 Ackley 函数，对连续空间、更高维度或其他目标函数的推广仍需进一步验证。

---

## 318. Event Detection in Videos: A Framework for the Development of New Methods

**arXiv ID:** 2607.04372 | [PDF](https://arxiv.org/pdf/2607.04372v1)

**作者:** Anastasia Zakharova `[一作]` (La Rochelle University), Bruno Vento `[通讯]` (Consorzio Interuniversitario Nazionale per l'Informatica)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个针对视频事件检测的完整框架，包括大型多环境数据集、评估管线和应用场景定义

**💡 创新点**

创新点在于统一的层次化数据结构、标签化的挑战标记、结合性能与硬件的评估方法以及可公开的应用场景说明

**🔧 技术方法**

使用层次化数据组织、标签系统、概率性能评估、硬件指标（内存、浮点运算、功耗）以及实时延迟与帧率测量

**📊 数据集**

主要使用自研的Event‑Monitoring Video Dataset（覆盖城市、自然、海事、水下等四类环境）以及从公共IP摄像头收集的FSD数据和CARLA合成数据

**📈 对比分析**

通过设定统一的评估协议（包含TP/FP/TN/FN、概率评估和硬件效率指标）对方法进行公平排名；实验显示在多环境下方法可实现低延迟高准确率，硬件资源占用可控

**⚠️ 局限性**

局限在于仍缺乏统一的事件定义标准、对极端环境（极寒、极热等）的覆盖有限、以及评估仍依赖人工标注导致标注成本高

---

## 319. A Perception-Manipulation Robotics System for Food Cutting

**arXiv ID:** 2607.04367 | [PDF](https://arxiv.org/pdf/2607.04367v1)

**作者:** Xinyuan Luo `[一作]` (University of Illinois at Urbana-Champaign), Wenzhen Yuan `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套感知‑操控框架，通过试切感知食材属性并选择合适刀具，然后使用强化学习自适应切割。

**💡 创新点**

提出基于试切的刀具选择模块和在真实机器人上训练的RL自适应切割控制器，实现对未知食材刀具选择100%成功。

**🔧 技术方法**

采用PID力控、SVM刀具分类、PPO强化学习、低维观测+离散动作空间、以及深度网络实现控制。

**📊 数据集**

收集了14种食材的力、位移、速度数据作为训练/验证集，并在10种食材上进行切割实验。

**📈 对比分析**

通过对比固定策略、RL策略和人类切割，评价指标为奖励、切割效率和切割速率，结果显示RL策略奖励最高、效率与人类相当、切割速率略低。

**⚠️ 局限性**

局限在于奖励函数仅考虑能量与进度，未捕捉切割质量；RL往往收敛到极端力或速度；试切噪声大；未考虑双臂协同。

---

## 320. Last-Meter Precision Navigation for UAVs: A Diffusion-Refined Aerial Visual Servoing Approach

**arXiv ID:** 2607.04352 | [PDF](https://arxiv.org/pdf/2607.04352v1)

**作者:** Yaxuan Li `[一作]` (University of Macau), Zhedong Zheng `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DreamNav框架实现无人机在最后一米内的精准视觉导航；

**💡 创新点**

采用三角函数回归消除角度周期性、双模态特征融合以及基于扩散模型的视觉想象细化两阶段策略；

**🔧 技术方法**

结合Vision Transformer+SuperGlue匹配、三角回归、双模态融合、ControlNet风格的扩散生成器与RGB姿态一致性约束；

**📊 数据集**

构建PairUAV数据集，包含4.8M张图像对、72个场景、1,652座建筑；

**📈 对比分析**

与AI2THOR、Sample4Geo、DINOv3等基线对比，DreamNav在平均误差和成功率上分别提升至约33.97、23.51%，相较基线显著改进；

**⚠️ 局限性**

对扩散细化阶段的候选网格有限、仅局部搜索，且在极端光照或结构遮挡场景下仍可能产生误判。

---

## 321. IRIS: An Intelligent Vision-Language System for Ocular Surface Diseases via Topic Tree and Scene-Driven VQA Generation

**arXiv ID:** 2607.04344 | [PDF](https://arxiv.org/pdf/2607.04344v1)

**作者:** Hao Wei `[一作]` (Chinese University of Hong Kong), Wu Yuan `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了IRIS系统，用于通过外部眼部图像进行细粒度的眼表疾病视觉问答与交互诊断；

**💡 创新点**

创新点在于构建了Topic Finding Tree（TFT）与Scene-Driven生成双分支数据引擎，系统性地将解剖学结构和临床角色语境嵌入到训练数据中；

**🔧 技术方法**

技术包括低秩适配（LoRA）微调Qwen3‑VL 4B模型，双分支VQA生成策略（TFT与场景驱动），多源图像预处理与质量感知动态采样；

**📊 数据集**

使用了IRIS‑120K数据集，融合医学书籍、网络资源和公开数据，包含约12万条视觉问答对，涵盖10个眼部解剖区域与多种病理特征；

**📈 对比分析**

与16个行业领先的通用与医学VLM进行基准测试，IRIS‑4B在所有问答类型上均取得最高分，准确率超过98%，相对最大对手Lingshu‑32B提升近20个百分点，展示出极高的参数效率；

**⚠️ 局限性**

局限性包括对高质量图像和解剖标注的依赖，仍需进一步验证在不同设备与真实临床环境下的鲁棒性与泛化能力。

---

## 322. Agent-driven Long-tail Simulation for Autonomous Driving

**arXiv ID:** 2607.04331 | [PDF](https://arxiv.org/pdf/2607.04331v1)

**作者:** Junru Gu `[一作]` (Tsinghua University), Hang Zhao `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于指令跟随大语言模型的代理驱动仿真框架，并在此框架下构建了SemanticPlan基准，涵盖从nuPlan场景衍生的多种长尾、语义丰富的交互式规划任务。

**💡 创新点**

创新点在于：①引入结构化动作接口，让LLM代理在保持物理可行性的前提下生成可执行的高层次行为；②通过语言指令控制周围参与者，显著提升闭环仿真的交互性和行为多样性；③构建SemanticPlan基准，将真实驾驶场景与LLM驱动的交互代理相结合，提供更具挑战性的长尾测试。

**🔧 技术方法**

技术手段包括：大语言模型（Qwen3.6‑27B、Qwen3.5系列）配合视觉‑语言处理；结构化观测与反馈机制；流匹配轨迹生成器；高层次路径规划与车辆控制接口；多模态多轮对话上下文管理。

**📊 数据集**

使用的数据集为：原始nuPlan数据集（用于基准场景构建）以及新构建的SemanticPlan基准（包含230+场景，50+类型，配有多语言指令的交互代理）。

**📈 对比分析**

方法对比：在碰撞倾向轨道上评估了规则基（IDM、PDM Closed）与学习基/混合基（UrbanDriver、GC-PGP、PlanTF、Diffusion Planner、PDM Hybrid、PLUTO）规划器，结果显示PDM Hybrid在总体指标上最佳，但安全得分仍低。语义轨道上对比IDM、IDM+LLM、IDM+LLM（停靠优先），后两者在提升交互进度和减少不当鸣笛方面优于纯IDM，但停靠优先过于保守导致整体得分下降。

**⚠️ 局限性**

局限性：①仿真仍依赖LLM模型的生成质量，可能出现逻辑不一致或物理不合理行为；②高质量模拟需要大量计算资源（实时仿真耗时高达15小时）；③对长尾场景的覆盖虽提升，但真实驾驶中的极端情形仍有限；④评估主要为零射手设定，未考虑对模型微调后的性能提升。

---

## 323. Legible-by-Construction: Attention and End-to-End Transformers

**arXiv ID:** 2607.04319 | [PDF](https://arxiv.org/pdf/2607.04319v1)

**作者:** Mark Oskin `[一作]` `[通讯]` (University of Washington), Mark Oskin (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种通过在Transformer注意力层的value路径上引入可辨识的模糊集合运算（单Sigmoid或布尔交并差）来使注意力值可解释，并与已可解释的前馈层联结，构建了一个整体可按构造解释的语言模型。

**💡 创新点**

创新点在于：①仅在value路径添加极小的Sigmoid约束或布尔运算即可将attention变为可读的成员检测器；②将此机制与前馈层的可解释结构相结合，首次实现整体模型的可构造解释；③揭示不同选择（稀疏压缩、尖锐化）对不同运算（单值vs交并差）有不同影响。

**🔧 技术方法**

技术手段包括：Transformer的多头注意力改造（value Sigmoid或布尔运算）、稀疏压缩与尖锐化正则化、参数无增改动、与已有可解释FFN相结合、基于语言建模任务训练与评估。

**📊 数据集**

使用开放网络语料库与与基准模型相同的词表和分词器进行训练；评估指标包含LAMBADA perplexity、BLiMP准确率、ARC‑Easy准确率。

**📈 对比分析**

与标准gelu Transformer基线在同一规模（125M）和训练设置下进行对比，所有可解释配置均保持在基线的“parity band”内，精度差异不超过0.04；在部分配置下甚至略有提升。

**⚠️ 局限性**

限制包括：实验仅单种随机种子、单一规模和单个训练周期；浅层注意力仍不够可解释；运算符的输入（operand）多为多义且需进一步解释；尚未验证在更大模型（1B+）或更复杂任务中的可扩展性。

---

## 324. GPU-Accelerated Polygonal Signed Distance Functions for Real-Time Collision Avoidance

**arXiv ID:** 2607.04310 | [PDF](https://arxiv.org/pdf/2607.04310v1)

**作者:** Taekwon Ga `[一作]` (Yonsei University), Jongeun Choi `[通讯]` (Yonsei University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了面向凸多边形机器人和障碍物的几何精确、可微分多边形签名距离函数（PSDF），并将其嵌入实时MPC框架，实现高效碰撞规避。

**💡 创新点**

创新点在于：①设计了完全基于张量化、无分支的GPU加速PSDF管线；②通过CPU/GPU分工，仅将PSDF评估留给GPU，其余QP仅受系统维数和预测时长限制；③将PSDF梯度直接用于SQP‑RTI的线性化，避免引入障碍相关决策变量。

**🔧 技术方法**

技术手段包括：自动微分、PyTorch张量化几何运算、GPU并行（CUDA）、基于Separating Axis Theorem的碰撞检测、SQP‑RTI与acados求解器、RealTime L4CasADi接口。

**📊 数据集**

实验数据集涵盖：随机生成的二维多边形障碍场、Gazebo走廊实验（Broad、Narrow、Cluttered）、Velodyne VLP‑16点云投影的实车环境、CARLA仿真车道泊车场景。

**📈 对比分析**

与传统的GJK+EPA、NPField、OBCA、DCBF、TEB、RDA等方法对比，PSDF‑MPC在每步优化时间仅约0.02–0.03 s，成功率为100%，在迷宫、走廊与泊车等任务中实现最快的导航时间，且保持实时性；相比之下其他方法在密集障碍中优化耗时超过0.1 s甚至数百毫秒。

**⚠️ 局限性**

局限性：目前仅适用于二维凸多边形，障碍物假设静止；对动态障碍、非凸形状、三维边界或不完整感知的鲁棒性尚未验证。

---

## 325. AI Wizards at EXIST 2026: Hierarchical Soft-Label Learning for Multimodal Sexism Identification in Memes

**arXiv ID:** 2607.04410 | [PDF](https://arxiv.org/pdf/2607.04410v1)

**作者:** Matteo Fasulo `[一作]` (ETH Zürich), Luca Babboni `[通讯]` (Independent researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一个层次化的多任务模型，用来在多模态表情包中软标签预测性别歧视。

**💡 创新点**

创新点在于将注释者分歧视为信号，通过可学习的无穷小不确定性权重和条件损失掩码实现层次化软标签预测，并使用轻量级Gated MLP在冻结的Gemini Embedding 2上做特征映射。

**🔧 技术方法**

技术包括冻结的Gemini Embedding 2视觉语言表示、SwiGLU基底的轻量化MLP、KL散度软标签损失、可学习的同方差不确定性权重、条件损失掩码与概率联合解码。

**📊 数据集**

使用了EXIST 2026公开的多模态性别歧视表情包数据集，包含3984个训练样本与1053个测试样本，包含英语和西班牙语两种语言。

**📈 对比分析**

通过在官方Soft-Soft排行榜上评测，模型在任务2.3（细粒度分类）获得第一名，在任务2.1、2.2分别获得第四名；在Hard-Hard排行榜上性能也较为稳健。

**⚠️ 局限性**

局限性包括依赖于专有的Gemini Embedding 2，硬标签解码使用固定阈值缺乏调优，且仅对EEG信号做线性可分性分析，未充分探索眼动与心率等生理信号的融合。

---

## 326. The Good, the Bad, and the Brittle: Benchmarking Robustness and Generalisation of Histopathology Foundation Models

**arXiv ID:** 2607.04401 | [PDF](https://arxiv.org/pdf/2607.04401v1)

**作者:** Dhyey Yajnik `[一作]` (University of Warwick), Fayyaz Minhas `[通讯]` (University of Warwick)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对12种病理基础模型和ResNet基线进行了鲁棒性和域泛化评估，利用REET工具箱在11种临床真实扰动以及NR‑Kfold验证协议下测试，并提出了Perturbation Performance Index（PPI）指标来总结模型在扰动下的整体表现。

**💡 创新点**

创新点在于：①引入PPI指标统一量化不同扰动对模型性能的影响；②提出NR‑Kfold协议通过特征空间聚类构造不重叠折叠，模拟真实多中心分布偏移；③系统揭示模型规模对鲁棒性的饱和性，表明后续提升需关注数据与预训练策略，而非单纯扩容。

**🔧 技术方法**

采用了REET生成像素级、染色与几何扰动，并使用梯度与随机优化的对抗扰动搜索；在冻结编码器的前提下训练单层线性分类器；通过特征聚类实现NR‑Kfold划分；计算PPI、AUC、ΔAUC等指标，并对参数量与鲁棒性进行相关性分析。

**📊 数据集**

使用了四个补丁级病理数据集：NCT（肿瘤/非肿瘤），PANDA（前列腺Gleason分级），PanNuke（肿瘤检测），以及PatchCamelyon（淋巴结转移检测）。

**📈 对比分析**

与ResNet18/50基线相比，PFMs在所有扰动下PPI显著更高，域偏移下AUC下降幅度更小；中等规模模型（如UNI2、Virchow2）在鲁棒性和泛化上已达到最优前沿，进一步增大参数量并未带来明显提升，甚至略有下降。

**⚠️ 局限性**

局限性包括：仅评估补丁级推理并冻结编码器，未考察多模态输入或全图推理；扰动集有限，未覆盖所有临床变异；缺乏在真实临床部署中的长周期验证，模型细粒度适配与优化仍需进一步研究。

---

## 327. On the Physical Plausibility and Distribution Alignment for Sim-to-Real RF Positioning

**arXiv ID:** 2607.04400 | [PDF](https://arxiv.org/pdf/2607.04400v1)

**作者:** Ararat Saribekyan `[一作]` (Yerevan State University), Theofanis P. Raptis `[通讯]` (National Research Council)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了将射线追踪合成数据用于RF定位的从模拟到真实的转移，重点比较基站校准方式（约束vs非约束）、数据规模（部署特定vs城市尺度）以及RSSI分布对齐对定位精度的影响。

**💡 创新点**

首次将约束与非约束校准对比，区分部署特定与大规模合成数据，并证明RSSI分布对齐比物理可行性更能提升未见街道的泛化性能；提出通过归一化合成RSSI实现分布对齐的方法。

**🔧 技术方法**

使用Sionna RT射线追踪、Gaussian‑process贝叶斯优化进行基站参数校准、DINO‑style深度学习定位网络、RSSI归一化、以及bootstrap置信区间评估。

**📊 数据集**

利用Rome真实测量数据集A；基于A生成的部署特定合成集B；从校准先验生成的城市尺度合成集C；以及对C的RSSI归一化变体。

**📈 对比分析**

通过在已知街道验证与未见街道验证两种模型选择策略，评估在已知街道和未见街道上的平均定位误差。所有合成预训练均显著提升已知街道性能；在未见街道上，C‑constrained归一化预训练最优，误差比从零开始训练下降约70米。

**⚠️ 局限性**

主要局限在于射线追踪模型与真实环境仍有较大偏差，基站校准得到的参数更多是有效参数而非真实物理值；合成数据与真实分布差异仍未完全消除；验证与选择策略对结果影响显著，需要进一步研究。

---

## 328. MechMath Agent Team: LLM Driven Agents for Mathematical Research

**arXiv ID:** 2607.04394 | [PDF](https://arxiv.org/pdf/2607.04394v1)

**作者:** Yichuan Cao `[一作]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences), Xiao-Shan Gao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 MechMath Agent Team（MMAT），一种基于大型语言模型的多代理闭环系统，能够从自然语言问题解析、构建证明结构、执行符号/数值验证、形式化 Lean 4 证明，并在多领域开放问题上完成完整推理与证明。

**💡 创新点**

① 三平面 Harness 架构（控制、执行、增强）实现任务调度与资源隔离；② 自我纠错闭环（任务账本、文件化交互、回溯恢复）提高鲁棒性；③ 集成知识库管理、自然语言推理与 Lean 4 形式化验证，形成可扩展的协同推理生态。

**🔧 技术方法**

大型语言模型（LLM）、多代理框架、执行图（DAG）调度、文件化交互、Lean 4 编译器、检索/解析/验证工具链、人机协作接口、持续记忆与负向约束等。

**📊 数据集**

使用公开的开放数学问题（OEIS、数论、代数复杂度、微分代数、算子代数、不等式等）作为实验数据，结合内部知识库检索、PDF 解析与符号计算结果。

**📈 对比分析**

相较传统线性多代理流水线，MMAT 在两个月内以 100% 的成功率解决 11 个开放问题，其中 9 篇已提交 arXiv，形式化验证覆盖率高，系统在多任务并行与错误恢复方面明显优于单一模型或手工推理，性能显著提升。

**⚠️ 局限性**

受限于 LLM 的推理偏差与计算资源、系统整体复杂度高、仍需人工介入进行调试与策略优化，且对极大规模计算与深层证明策略的全局优化能力有限。

---

## 329. Optimal Online Discrepancy Minimization in Linear Time

**arXiv ID:** 2607.04388 | [PDF](https://arxiv.org/pdf/2607.04388v1)

**作者:** Ishaq Aden-Ali `[一作]` `[通讯]`, Ishaq Aden-Ali

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种在线向量符号分配算法（Gaussian Triplet Walk），可在每一步以随机符号更新前缀和，同时保证每个前缀和都可表示为三条独立高斯轨迹之和；

**💡 创新点**

该算法在保持前缀和为3‑subgaussian的同时，实现了最优的前缀差异度上界 O(√(log T))，并且仅需 O(dT) 计算时间，突破了之前指数级别实现该界的瓶颈；

**🔧 技术方法**

核心技术包括构造平衡的高斯固定点随机 walk、三维协同抽样（coupling 三条高斯轨迹），以及利用 Hölder 不等式和子空间投影的概率估计；

**📊 数据集**

本文仅进行理论分析，无实验数据集；

**📈 对比分析**

对比方法：传统的自平衡 walk 仅得到 O(log T) 上界，Kulkarni‑Reis‑Rothvoss 的算法虽达到 O(√(log T)) 但运行时间指数级；本算法在时间上与前者相当，同时在差异度上达到最优；

**⚠️ 局限性**

限制：算法仅适用于 ℓ₂ 约束下的在线 Komlós 问题，且对更一般的范数或随机输入分布缺乏直接推广；

---

## 330. Mean Time to Remediate Is Not a Fielding Model: A Cadence Audit for Enterprise Vulnerability Management

**arXiv ID:** 2607.04511 | [PDF](https://arxiv.org/pdf/2607.04511v1)

**作者:** Alexander Omelchenko `[一作]` `[通讯]` (Constructor University Bremen), Alexander Omelchenko (Constructor University Bremen)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个针对企业漏洞管理的修复节奏审计（remediation‑cadence audit），通过记录均值延迟、发布周期、发布比例、循环/环形部署、应急与常规路径、硬延迟预算、残留压力区间以及速率场景，评估发布日历对本地防御容量的影响并给出日历折扣（calendar discount）

**💡 创新点**

创新点在于：①将 MTTR/SLA 等平均修复指标与实际发布日历进行对比，揭示平均值隐藏的时间几何差异；②引入日历折扣量化日历化发布占据平均容量的比例；③将证据分辨率与折扣结合，形成基于证据分辨率的治理结论；④将环形/分阶段发布、应急分道等实际操作细节纳入审计记录

**🔧 技术方法**

使用的技术包括：基于样本数据控制理论的释放周期模型（同步/异步、分配比例、硬延迟位置等）计算持续均值与日历化阈值；利用残留压力对账本计算局部压力区间；对比两种容量边界得到日历折扣；采用区间分析和情景（rate‑scenario）带宽来评估不确定性

**📊 数据集**

论文使用的主要数据是：公开的运营时序信息（如月度发布周期、维护窗口、硬延迟预算）以及一个示例性的残留压力对账本（二维 X/Y 组合评分，区间化）。没有真实企业数据，所有数值均为演示用的“虚构”数据

**📈 对比分析**

与传统 MTTR/SLA 评估方法的比较：在同一 30 天均值下，日历折扣从 1.3%（两周）到 17.4%（两月）不等，揭示相同平均延迟下发布周期对容量的影响。性能上，日历折扣能在不需要完整分布假设的情况下提供对比度；但由于缺少真实数据，无法给出绝对性能指标

**⚠️ 局限性**

局限性包括：①仅为本地治理诊断，不具备预测或攻击概率模型；②高度依赖残留压力区间的准确性，若区间过宽则难以作出清晰判断；③对速率场景假设（如归一化屏蔽）敏感，需在实际操作中明确或做情景分析；④未考虑多渠道耦合、攻击者对日历的战略响应等复杂因素；⑤仅在演示环境验证，缺乏大规模实测验证

---

## 331. Geographic Diversity Beats Data Volume for Cross-Domain Generalization in Zero-Label JEPA Driving World Models

**arXiv ID:** 2607.04500 | [PDF](https://arxiv.org/pdf/2607.04500v1)

**作者:** Santosh Jaiswal `[一作]` `[通讯]`, Santosh Jaiswal

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了自监督 JEPA 世界模型在不同地理区域驾驶数据上的跨域泛化能力，并用零标签惊讶分数评估其复杂度检测效果。

**💡 创新点**

通过地理多样性训练显著提升模型跨域惊讶分数，并在匹配规模实验中证明地理多样性优于单一地理数据量。

**🔧 技术方法**

采用 Joint Embedding Predictive Architecture (JEPA) 的 38.7M 参数 Transformer 结构，输入为 50 步 21 代理状态，输出为潜在空间的预测与 EMA 编码的比较。

**📊 数据集**

使用 nuPlan（波士顿、匹兹堡、新加坡、拉斯维加斯）和 Argoverse 2（迈阿密、奥斯汀）数据集进行训练与评估。

**📈 对比分析**

在四种训练条件下（nuPlan‑only、Combined‑63K、AV2‑only、Combined‑full），Combined‑63K 在 63K 规模下平均惊讶分数下降 16.5%，并且即使 AV2‑only 训练规模扩大 3 倍也无法取代地理多样性优势。

**⚠️ 局限性**

局限包括 AV2‑only 训练不稳定、仅使用结构化代理状态而非原始传感器数据、以及仅通过惊讶分数评估而未验证下游规划或检测任务。

---

## 332. A Reconfigurable and Representation-Adaptive ISA-Based Architecture for Efficient DNN Acceleration

**arXiv ID:** 2607.04475 | [PDF](https://arxiv.org/pdf/2607.04475v1)

**作者:** Vasilis Sakellariou `[一作]` (Khalifa University), Thanos Stouraitis `[通讯]` (University of Patras)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向机器学习的指令集架构（ISA）和可重构硬件平台，并在此平台上实现了基于剩余数系统（RNS）的动态精度推理加速器。

**💡 创新点**

创新点包括：① 将控制、数据搬运、张量计算和后处理四个处理域解耦，极大提升PE利用率；② 设计了轻量级可编程核心（mCore），实现高可编程性与低控制开销的平衡；③ 引入动态精度（粗细两种）与RNS计算，实现低位宽、低功耗且保持模型精度；④ 在ISA层面实现可扩展的激活函数、量化与分支控制，为未来模型架构提供灵活性。

**🔧 技术方法**

使用的技术包括：定制化的32位ISA（load/store、算术、SIMD、分支、中断等指令集），多核心（Type‑I/II/III）控制单元，SIMD张量处理单元（TPA）和后处理单元（PPU），动态精度控制指令，RNS算术单元与变基底选择，片上 SRAM 与缓存层次，PWL（Piece‑wise Linear）激活函数实现，Softmax整数化近似。

**📊 数据集**

在ResNet‑18、ResNet‑50、Vision‑Transformer (ViT‑b16)、YOLO‑v5‑m、BERT‑base 等常见模型上评估，使用 ImageNet、COCO、SQuAD 等公开数据集验证准确率与功耗。

**📈 对比分析**

与 RISC‑V 及现有动态精度、固定功能加速器进行对比。RNS 加速器在 22 nm 流片实现下，W8A8 量化可达 5.14–10.47 TOPS/W，功耗 5.14–10.47 TOPS/W，平均提升约 1.2×（相较于同类 FXP 或 RISC‑V 系统）。在模型精度保持不变或误差低于 1% 的情况下，RNS 方案能显著提升能效，并在多种工作负载上保持高 PE 利用率（>90%）。

**⚠️ 局限性**

局限性：① 设计主要针对整数与 RNS 计算，复杂算子（除法、比较）需要额外电路，导致硬件面积和实现复杂度上升；② 动态精度与 RNS 基底的在线选择仍需外部调度，增加了设计与验证难度；③ 仅在 22 nm 低压（0.65 V）工艺下验证，跨工艺迁移时性能与功耗波动需进一步研究；④ 对极低位宽（≤3 bit）或高位宽（>8 bit）场景的支持仍有限。

---

## 333. Operator-on-F complements value-equivalence: a planning-time diagnostic for latent world models

**arXiv ID:** 2607.04464 | [PDF](https://arxiv.org/pdf/2607.04464v1)

**作者:** Donna Vakalis `[一作]` `[通讯]` (Mila Quebec AI Institute), Donna Vakalis (Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并评估了一种新的诊断指标 operator-on‑F，用来比较模型在可观测子集上的 k 步潜在推进与真实环境的推进，从而捕捉传统奖励/价值预测指标忽视的规划相关错误。

**💡 创新点**

创新点在于：① 引入了 operator-on‑F 这一潜在推进对比指标；② 证明它与规划回报具有更强的相关性；③ 在跨架构对比中揭示了不同模型在相同可观测子集上的性能差异；④ 进一步表明传统的 value‑equivalence 或 Bellman 残差检查可能在关键失效模式下保持“沉默”。

**🔧 技术方法**

采用了潜在动力学滚动、基于岭回归的 probe 估计、PCA 方向投影、RMS 聚合等技术；在 DeepMind Control Suite 的 cheetah‑run 任务上对 TD‑MPC2 规模变换（1M–317M 参数）和纯 SSL LeWM 进行评估，并使用 Spearman 相关、anchor‑bootstrap CI 等统计方法进行分析。

**📊 数据集**

使用了 DeepMind Control Suite（DMC）中的 cheetah‑run 任务以及其他 mt80 任务作为实验环境，所有模型均在同一任务上进行对比。

**📈 对比分析**

比较方法：对 5 个不同规模的 TD‑MPC2 checkpoint 计算 operator‑on‑F 误差和奖励预测误差，发现 operator‑on‑F 与回报的 Spearman 相关系数为 -0.90（95% CI [-0.90,-0.70]），而奖励预测误差与回报的相关性仅为 -0.10。跨架构对比显示 TD‑MPC2 的 operator‑on‑F 误差约为 0.84，LeWM 为 0.38，两者的 95% CI 完全不重叠，说明该指标能区分不同架构。该指标的排序与传统 Bellman 残差和奖励误差明显不一致，提供了更具判别力的性能评估。

**⚠️ 局限性**

限制：① 仅在单一任务（cheetah‑run）上验证，缺乏跨任务通用性；② 规模变换样本量小（n=5），相关性可能受偶然因素影响；③ 需要 probe 读取潜在空间，probe 的正则化和容量可能影响结果；④ 跨架构对比仅覆盖两种模型，不能得出通用优劣结论；⑤ 仅为诊断工具，未直接用于训练，无法直接改善模型性能。

---

## 334. Fields of the Planet: Field Boundary Mapping Beyond 10m

**arXiv ID:** 2607.04449 | [PDF](https://arxiv.org/pdf/2607.04449v1)

**作者:** Isaac Corley `[一作]` (Taylor Geospatial), Hannah Kerner `[通讯]` (Arizona State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了 Fields of the Planet（FTP）数据集，将 Fields of the World（FTW）标签与 3 米分辨率的 PlanetScope 图像配准，以实现更高分辨率的田块边界提取。

**💡 创新点**

创新点在于：①以同一标签和评估协议对比 10 m Sentinel-2 与 3 m PlanetScope，清晰揭示分辨率对小田块恢复的影响；②采用基于多边形的评估指标（panoptic quality、object F1、边界 Chamfer），而非传统像素指标；③公开了数据生成、训练基线与评测代码，促进可复现研究。

**🔧 技术方法**

使用 U‑Net+EfficientNet 编码器的分割模型，遵循 PRUE 训练流程，加入 PRUE+ 数据增强、标记控制的 watershed 后处理与 D4 试点时增强；评估使用 panoptic quality、object F1、平均边界误差等多维指标。

**📊 数据集**

主要使用 Fields of the World（FTW）提供的 1,627,378 个田块多边形与对应 Sentinel-2 影像，以及通过 PlanetScope 数据 API 提取的 133,168 对应 3 m 影像组成的 FTP 数据集。

**📈 对比分析**

在同一网络架构与训练配置下，将 FTP 与 FTW 进行对比实验；在 10 个小农田密集国家的 held‑out 测试集上，FTP 的 PQ 从 21.0 提升到 35.5，0.5 ha 以下田块 PQ 由 5.8 提升到 15.7，平均边界误差从 18.6 m 降至 7.4 m，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：①缺失的云量高、标签稀疏的区域导致 5.5 % 目标未配对；②仅包含两季节窗口、4‑band 3 m 影像，未覆盖多年份或 8‑band 数据；③对低对比度地籍线的边界恢复效果有限；④数据只适用于聚合监测，不能用于单户识别。

---

## 335. From Regulation to Requirements: An Automated Requirement Derivation and Explanation Pipeline

**arXiv ID:** 2607.04448 | [PDF](https://arxiv.org/pdf/2607.04448v1)

**作者:** Pavithra PM Nair `[一作]` (Tata Consultancy Services), Preethu Rose Anish `[通讯]` (Tata Consultancy Services)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套自动化管道（Reg2Req），将法规文本转换为系统级软件需求，并附带通俗解释和可追溯性信息。

**💡 创新点**

创新点包括：①将“规定条款是否蕴含软件需求”定义为独立任务并使用零射模型进行二分类；②在同一次LLM调用中联合生成需求、解释和交叉引用类型；③提供无法规专属配置的通用流程，公开工具和数据集。

**🔧 技术方法**

技术主要是大语言模型（GPT‑5 和 GPT‑5.4）配合精心设计的提示工程；还使用 SetFit 作为基线、Spearman 相关分析、Krippendorff α 等统计方法评估。

**📊 数据集**

数据集为欧盟两部法规完整条款集合：GDPR（398 条）和欧盟 AI 法案（574 条），并手工标注需求条款、需求/解释质量等。

**📈 对比分析**

与 SetFit 二分类基线对比，Reg2Req 在需求条款识别上 macro‑F1 分别为 0.815（GDPR）和 0.779（AI 法案），显著优于基线 0.705；人类评估显示需求完整度 4.6/5，解释清晰度 4.92/5；用户研究显示解释提升理解度 +0.88、信心 +0.98（p<0.001）。

**⚠️ 局限性**

局限包括：①对含有大量交叉引用或复杂语法的条款准确性仍受限；②评价仅基于欧盟法规，跨司法辖区推广尚未验证；③依赖 GPT‑5 等闭源模型，模型漂移可能影响重现性。

---

## 336. Knowledge-Informed Local Causal Discovery of Optimal Adjustment Sets

**arXiv ID:** 2607.04447 | [PDF](https://arxiv.org/pdf/2607.04447v1)

**作者:** Seong Woo Ahn `[一作]` (Universite Paris-Saclay), Arpad Rimmel `[通讯]` (Universite Paris-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种将背景知识嵌入本地因果发现循环的算法，可在样本稀缺的情况下识别最优调整集。

**💡 创新点**

创新点在于直接将所需边约束注入局部结构学习，并利用四规则Meek闭包生成知识约束的MPDAG，从而显著提升可识别性。

**🔧 技术方法**

核心技术包括LOAD算法的局部Markov blanket搜索、Meek规则推导、MPDAG构造、可识别性判定和最优调整集提取。

**📊 数据集**

实验使用合成Erdős–Rényi图、Sachs蛋白信号网络和DREAM4仿真子网三个数据集。

**📈 对比分析**

与PC、b-PC、LOAD等基线比较，在低样本/高结构复杂度场景下，本文方法在F1和干预距离上均优于基线，特别是局部知识采样模式表现最佳。

**⚠️ 局限性**

主要局限包括假设背景知识完全正确、仅考虑因果充分性且仅适用于线性高斯模型，未对潜在混杂、非线性或错误约束进行处理。

---

## 337. ResearchStudio-Idea: An Evidence-Grounded Research-Ideation Skill Suite from ML Conference Outcomes

**arXiv ID:** 2607.04439 | [PDF](https://arxiv.org/pdf/2607.04439v1)

**作者:** Qihao Zhao `[一作]` (Nanyang Technological University), Yap Kim Hui `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 ResearchStudio‑Idea 套件，包含可重用的 Paper‑Search、Scoop‑Check 与 End‑to‑End 的 IdeaSpark 逻辑，用于将科研问题转化为可审计的研究提案。

**💡 创新点**

创新点在于：① 将 1,947 篇 ICLR/ICML/NeurIPS 2021‑2025 论文（含 Oral、High‑Cited、Reject 三类标签）挖掘成 15 条可操作的 ideation pattern cards；② 通过检索、冲突检测与审核相结合的无参数工作流，生成并验证单一高质量研究方向；③ 在自动化评审中实现质量优于基线且创新度保持竞争力。

**🔧 技术方法**

使用的技术包括：大规模文献检索（arXiv、OpenAlex 等多源）、RAG‑增强式生成、嵌入聚类 + 主题发现、模式卡构建、先验冲突检查（Scoop‑Check）、多轮结构化推理与审核校验、以及无参数技能（skill）框架。

**📊 数据集**

数据集：1,947 篇公开的 ICLR、ICML、NeurIPS 2021‑2025 会议论文，分别标记为 Oral、High‑Cited（Top‑30/Top‑10 计量）与 Reject，并附带 OpenReview 评审与 Semantic Scholar 引用信息。

**📈 对比分析**

通过盲测自动评审与 no‑skill、generic‑skill 基线对比，IdeaSpark 在 100 个 ICLR‑2026 口头种子上实现了最高的提案质量，并在 21 个主领域均位居榜首，创新度与基线相当，表明在多领域均能提升质量。

**⚠️ 局限性**

局限性：仅评估至提案阶段，未涉及实验与人类同行评审；自动评审可能受评审者偏差影响；模式卡构建依赖会议结果标签，可能缺乏跨学科或其他领域的泛化；模式引导的生成仍需进一步验证其可执行性。

---

## 338. Covert Trait Propagation Is Representation Alignment: Mechanistic Evidence from Hidden-Channel Distillation

**arXiv ID:** 2607.04432 | [PDF](https://arxiv.org/pdf/2607.04432v1)

**作者:** Kargi Chauhan `[一作]` (University of California, Santa Cruz), Aditya Shah `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在模型蒸馏中，由共享初始化导致的隐形行为迁移机制，即教师模型的辅助logit通过几何对齐让学生在纯噪声训练下获得分类能力。

**💡 创新点**

创新点在于将隐形行为迁移归因于表示对齐而非信息传输，并通过层冻结、重初始化、多教师对抗等干预，证明了几何对齐是打开通道的关键，同时在LLM指令调优中揭示了跨标记行为纠缠的几何根源并纠正了频率偏差的度量伪影。

**🔧 技术方法**

采用了线性中心化核对齐（CKA）衡量表示相似度、KL散度训练辅助logit、Adam优化以及互信息估计等技术。

**📊 数据集**

实验数据集包括MNIST数字分类用于MLP蒸馏实验，以及Llama‑3.2‑1B指令调优模型的语言生成数据。

**📈 对比分析**

通过对不同初始化距离、教师学习率、层冻结、多教师集成等条件进行干预，结果显示学生准确率与CKA高度相关（r=0.98），且多教师蒸馏可完全消除通道；在LLM实验中，指令调优后18/20动物出现行为纠缠，而基模型仅1/20。

**⚠️ 局限性**

局限性包括仅在小型MLP和1B LLM上验证，缺乏对更大模型和Transformer深度的推广，CKA作为诊断的通用性尚未验证，且实验未对LLM进行因果干预，仅观察性比较。

---

## 339. dOPSD: On-Policy Self-Distillation for Diffusion Language Models

**arXiv ID:** 2607.04428 | [PDF](https://arxiv.org/pdf/2607.04428v1)

**作者:** Phuong Tuan Dat `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Diffusion语言模型上提出一种自监督的对齐方式——dOPSD，用学生-教师的同模型结构提升推理性能。

**💡 创新点**

创新点是把教师的“特权信息”直接来源于学生自己的去噪解码轨迹，而非外部参考答案，解决了传统OPSD在Diffusion模型中的PI缺失和随机掩码偏离路径问题。

**🔧 技术方法**

技术包括在Diffusion解码过程中抽取中间步作为学生监督点、使用后续步骤的更完整上下文来构建教师分布、基于Jensen–Shannon散度进行稀疏对齐，以及可选的完成答案验证。

**📊 数据集**

主要使用MixChain-Z-PRM12K（数学推理数据集）训练，并在GSM8K、MATH500、HumanEval、MBPP等基准上进行评估。

**📈 对比分析**

与SFT、RLPO、以及OPSD(答案/完整解答)等基线相比，dOPSD在Dream-7B-Instruct与LLaDA-8B-Instruct上均获得所有四个任务的最高或第二高分，尤其在数学推理和代码生成方面实现显著提升。

**⚠️ 局限性**

局限性包括对完成答案验证的依赖（若无答案则效果略减弱）、需要额外的前向传播计算教师分布、以及目前仅在数学推理数据集上训练，尚未在更广泛任务上验证。

---

## 340. Environmental Drivers of Respiratory Disease: A District Level Analysis

**arXiv ID:** 2607.04416 | [PDF](https://arxiv.org/pdf/2607.04416v1)

**作者:** Rahim Iqbal `[一作]` (University of Moratuwa), Sandareka Wickramanayake `[通讯]` (University of Moratuwa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了2014-2024年11年、覆盖全25区的面板数据，整合森林退化、火灾、空气质量和呼吸病入院率，并用XGBoost预测年度呼吸疾病率和每月PM2.5。

**💡 创新点**

首次在斯里兰卡以区级精度将森林砍伐、火灾与空气污染与呼吸健康关联，并提出基于SHAP权重的森林-空气-健康风险指数。

**🔧 技术方法**

采用卫星遥感（GFW、VIIRS、MERRA‑2、CAMS）、时间序列XGBoost回归、SHAP解释、PCA与K‑means聚类，以及FAH风险指数计算。

**📊 数据集**

使用全球森林观察、NASA MERRA‑2、CAMS EAC4、VIIRS 8日植被指数、NASA FIRMS火点、斯里兰卡卫生部年度入院数据、人口统计等公开数据集。

**📈 对比分析**

通过时间交叉验证比较XGBoost与基线模型，年度呼吸率模型在测试集R²=0.937、MAE=0.776/千人，PM2.5模型R²=0.976、MAE=0.520 μg/m³，MAPE≤20%的21/25区均表现优异。

**⚠️ 局限性**

健康数据按年度均摊导致季节性缺失，MERRA‑2、CAMS分辨率粗糙且未考虑跨境传输，COVID‑19报告中断影响部分区数据，缺乏月度健康记录与车辆排放等因素。

---

## 341. Near-Optimal and Efficient Encoding for Two-Dimensional Range Minimum Queries

**arXiv ID:** 2607.04509 | [PDF](https://arxiv.org/pdf/2607.04509v1)

**作者:** Paweł Gawrychowski `[一作]` (University of Wrocław), Srinivasa Rao Satti `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种新的二维 RMQ（Range Maximum Query）编码方案，能够在不访问原始二维数组的前提下，利用编码存储信息来回答查询。

**💡 创新点**

核心创新在于引入可调参数 κ∈[1, loglog n]，实现了空间与查询时间的可调权衡：编码空间为 κ mn(log m + loglog n) 位，查询时间为 O(log^{1/κ} n)。这弥补了先前在空间最优（mn log m 位）但查询无效与空间更小但查询常数时间之间的空白。

**🔧 技术方法**

技术上主要使用了：
- 列树（Complete binary tree over columns）来定义点的“origin”与“active set”；
- 预处理得到的“可见点”候选对，保证要比较的两点共活跃于某一节点；
- 局部排名结构（rank) 与提升结构（predecessor/successor），分别实现点在局部集合中的秩查询和向根方向的“提升”；
- 对活跃点按“quarter‑row”划分并利用Elias–Fano编码实现顺序与选择；
- 通过在列树上构造多层（τ-ary）树来限制提升步数，从而控制查询时间。

**📊 数据集**

该工作为理论研究，不依赖任何实验数据集；所有结果均为信息论上、空间/时间复杂度的定量分析。

**📈 对比分析**

与已有最优空间编码（mn log m 位）以及常数时间编码（mn min{m,log n} 位）相比，本文实现了在几乎同等空间（当 κ≪loglog n 时）下的子对数查询时间，或在相同查询时间（O(log^{1/κ} n)）下的空间压缩。实验性比较未涉及，但理论上已展示该结构在空间与查询时间上的优越折衷。

**⚠️ 局限性**

局限性包括：
- 结构实现较为复杂，需多层树与多套局部编码，工程实现成本高；
- 当 κ 取值较大时，空间会膨胀至 O(mn log m · loglog n) 位，远高于最优空间；
- 该方法仅适用于二维 RMQ，扩展到更高维度或其他查询类型仍需进一步研究。

---

## 342. Why Pure Reasoning is Not Enough: Nature as the Source of Mathematical Innovation

**arXiv ID:** 2607.04505 | [PDF](https://arxiv.org/pdf/2607.04505v1)

**作者:** Charanjit S. Jutla `[一作]` (IBM), Vimal Sharma `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨了数学创新主要源自对自然世界模式的匹配，而非纯粹的演绎推理；

**💡 创新点**

提出“物理先行、抽象随后”这一观点，并将傅里叶变换的发展史作为案例证明；

**🔧 技术方法**

主要采用历史分析与逻辑复杂性理论的论证手段；

**📊 数据集**

未使用具体数据集；

**📈 对比分析**

未进行实验比较，理论论证为主；

**⚠️ 局限性**

对实际可计算性与人工智能实现的局限性提出了讨论，指出仅凭演绎无法突破复杂性与不可判定性壁垒。

---

## 343. UniSkip-Mamba: A Frequency-Aware State Space Model for Audio-Visual Temporal Forgery Localization

**arXiv ID:** 2607.04498 | [PDF](https://arxiv.org/pdf/2607.04498v1)

**作者:** Cangjin Qiu `[一作]` (Soochow University), Ke Zhang `[通讯]` (Soochow University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 UniSkip‑Mamba 框架，解决音视频时间性伪造定位（AV‑TFL）问题。

**💡 创新点**

创新点在于：① 将音频与视觉特征序列化为统一长序列并引入模态嵌入；② 设计 Skip‑Scanning Mamba 块，利用 Group‑Scan‑Merge 机制实现频率感知的低通正则化；③ 通过频域分析证明低/中频信息是判别核心，从而在保持精度的同时显著提升速度与鲁棒性。

**🔧 技术方法**

使用技术包括：Mamba 状态空间模型（线性时间复杂度）、统一序列融合、组扫描‑合并 Skip‑Scanning、频率域滤波分析、双向扫描与残差连接。

**📊 数据集**

实验数据集：LAV‑DF（大规模多模态伪造）和 AV‑Deepfake1M（最大规模多模态伪造）。

**📈 对比分析**

与多种 Transformer 及 Mamba 基线（如 UMMAFormer、DiMoDif、ActionMamba 等）对比，UniSkip‑Mamba 在 LAV‑DF 的 AP@0.95 达到 63.4%（+9.8%），在 AV‑Deepfake1M 的 mAP 达到 63.58%（+14.32%）；推理速度提升约 6×，在多种降噪与压缩场景下鲁棒性显著优于对照方法。

**⚠️ 局限性**

局限性：Skip‑Scanning 对极短时高频伪造（如单帧错误）可能产生信息丢失；缺乏自适应步长或多尺度融合策略，未来可进一步提升对不同尺度伪造的检测能力。

---

## 344. PulmoSight-XAI: An Explainable Multi-View Attention Ensemble with Gradient Boosting Meta-Learning for Multi-Label Chest X-Ray Classification

**arXiv ID:** 2607.04478 | [PDF](https://arxiv.org/pdf/2607.04478v1)

**作者:** Moshiur Rahman `[一作]` (Bangladesh University of Engineering and Technology), Tasnia Binte Mamun `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多视角层次化集成框架，专门针对胸部X光的14种病理进行多标签分类；

**💡 创新点**

创新点在于：①对前后视（frontal）和侧视（lateral）分别训练模型以充分利用解剖差异；②使用多尺度特征融合并在深层阶段加入CBAM注意力模块，保留细粒度纹理信息；③构建双重损失（非对称损失+自适应焦点损失）以解决样本不平衡和难易度差异；④引入层级梯度提升元学习（TTA、跨模型不确定性、Level‑1 XGBoost/LightGBM/CatBoost、Level‑2堆叠+alpha融合）来精细组合基模型；⑤通过七种后置可解释方法验证模型定位合理。

**🔧 技术方法**

技术包括：多尺度卷积神经网络（InceptionV3、ConvNeXtV2‑Tiny、DenseNet201、EfficientNet‑B5、ResNeXt‑101），CBAM注意力模块，混合损失函数，自动混合精度训练，AdamW+Cosine Annealing+EMA+SWA，TTA，梯度提升机器，Alpha blend，七种可解释方法（Grad‑CAM、Grad‑CAM++、Guided BP、Integrated Gradients、LIME、Occlusion、SHAP）。

**📊 数据集**

使用Kaggle Grand‑X‑ray Slam Division‑B数据集（约108,494张胸片，包含前后视和侧视，14类标签，使用CheXpert式标签）。

**📈 对比分析**

与七个公开基准（SSGE、GCF‑Net、U‑Zeros、cheXGCN、CvTGNet、MXA等）对比，宏观平均AUROC达到0.9319（前视）和0.9154（侧视），显著优于以往14类胸片分类模型的0.89–0.91区间；单模型表现也在各类中排名靠前，且层级集成进一步提升。

**⚠️ 局限性**

局限性包括：仅在单一专有数据集上评估，缺乏跨数据集的外部验证；对极少见病理（如肺损伤、肺炎等）的敏感度仍不足；模型复杂度高，难以部署于资源受限环境；缺乏量化的定位评估与不确定性界限；对人口和设备差异的公平性尚未充分验证。

---

## 345. AMRM-Pure: Semantic-Preserving Adversarial Purification

**arXiv ID:** 2607.04474 | [PDF](https://arxiv.org/pdf/2607.04474v1)

**作者:** Zhihao Dou `[一作]` (Wenzhou-Kean University), Shufei Zhang `[通讯]` (Shanghai AI Lab)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于注意力矩阵变化最小化的对抗净化框架 AMRM-Pure，利用 MAE 和 MaskDiT 对抗样本进行去噪，并通过分类损失微调进一步提升鲁棒性。

**💡 创新点**

创新点在于发现对抗扰动对图像块间语义关系（注意力矩阵）高度敏感，并将这一敏感性转化为可优化的重建损失，形成自监督的对抗净化方法；同时通过两阶段微调实现对抗训练级别的鲁棒性提升。

**🔧 技术方法**

采用 Attention Mask Reconstruction Models（MAE、MaskDiT）、投影梯度下降（PGD）对重建损失进行优化、分类损失微调、对抗攻击评估（AutoAttack、PGD+EOT）以及理论分析（AMV 与重建误差下界）。

**📊 数据集**

在 CIFAR-10、CIFAR-100、SVHN 和 ImageNet 四个标准数据集上进行训练与评估。

**📈 对比分析**

与现有对抗净化方法（DiffPure、COUP、ADDT 等）及对抗训练基线进行对比，结果显示 AMRM-Pure（尤其是 MaskDiT 版本）在 ℓ∞、ℓ2 攻击下均达到或超过现有 SOTA；在 CIFAR-10 上鲁棒精度超过 75%，在 ImageNet 上亦保持高效推理时间。

**⚠️ 局限性**

局限性包括：仅针对掩码自编码器框架，缺乏对其他生成器或自监督模型的泛化验证；对抗攻击评估依赖 AutoAttack/PGD+EOT，可能存在梯度遮蔽问题；未在大规模真实场景或多模态任务中进行验证。

---

## 346. Sampling Bias Compensation for Robust Evaluation of Audio Classification Systems with Partially Labeled Evaluation Datasets

**arXiv ID:** 2607.04463 | [PDF](https://arxiv.org/pdf/2607.04463v1)

**作者:** Javier Naranjo-Alcazar `[一作]` (Instituto Tecnologico de Informatica), Pedro Zuccarello `[通讯]` (Instituto Tecnologico de Informatica)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在声学场景分类中，利用未标注部署数据通过重要性加权来校正从部分标注评估子集得到的性能估计。

**💡 创新点**

首次将重要性加权与三种密度比估计方法（KDE、kNN、Logistic Regression）结合，系统评估其在不同主动学习采样策略下的偏差校正效果。

**🔧 技术方法**

采用密度比估计（KDE、kNN、Logistic Regression）实现重要性权重；利用嵌入向量进行PCA降维；使用随机/不确定性/多样性/密度等五种采样策略构造子集。

**📊 数据集**

DCASE 2017 语音场景分类基准（15类、1620条样本的评估集）

**📈 对比分析**

在多达500次独立试验中比较无加权与三种加权估计的微型准确率，结果显示加权估计能显著缩小与全数据真实性能的误差，Logistic Regression 在多样性采样下恢复最快，KDE/kNN 在 K‑Medoids 采样下表现最稳健；当标注预算趋近完整时，三者均收敛。

**⚠️ 局限性**

对不确定性采样导致的极端信息缺失仍无法完全校正；密度比估计对采样策略高度敏感，过度纠正或欠纠正的风险；高维嵌入空间中 KDE 的估计仍受限于样本量和降维选择。

---

## 347. Correct but Slow: An Empirical Study of the GPU Kernel Evaluation Gap in Modern Domain-Specific Languages

**arXiv ID:** 2607.04454 | [PDF](https://arxiv.org/pdf/2607.04454v1)

**作者:** Tingxi Li `[一作]` (University of Texas at Dallas), Wei Yang `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨了现代领域特定语言（DSL）中GPU内核的正确性与性能之间的差距，分析了22个Triton和TileLang内核在NVIDIA A100和GH200 GPU上的表现，发现仅依赖正确性评估可能会导致性能低下的内核被接受。

**💡 创新点**

创新点在于提出了两种轻量级评估标准（库相对效率和屋顶线利用率），可以有效识别功能上有效但效率低下的内核，并将可修复的作者缺陷与结构性残余区分开来。

**🔧 技术方法**

使用了GPU性能分析技术，包括硬件计数器和基于性能的评估方法，结合了对Triton和TileLang内核的性能分析。

**📊 数据集**

使用了22个内核，涵盖五个操作类别，包括矩阵乘法、注意力机制、卷积、归一化和元素级或归约操作，数据集来自于TritonBench和TileLang的实现。

**📈 对比分析**

与现有的KernelBench和TritonBench进行比较，发现这些基准测试仅关注正确性，未能有效识别性能低下的内核。研究表明，某些内核的性能比PyTorch基线慢300倍以上，但仍通过了正确性测试。

**⚠️ 局限性**

限制在于现有基准测试未能覆盖所有DSL、数据类型、形状和GPU的组合，且研究主要集中在前向传递内核，未考虑反向传递内核的潜在瓶颈。

---

## 348. A Deep Learning-based surrogate model for Severe Accidents in nuclear reactors using ASTEC

**arXiv ID:** 2607.04450 | [PDF](https://arxiv.org/pdf/2607.04450v1)

**作者:** Alessandro Longhi `[一作]` (TU Delft), Zoltán Perkó `[通讯]` (TU Delft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对ASTEC模拟器中核反应堆容器（vessel）物理过程，构建了一个基于自动编码器（AutoEncoder）与神经常微分方程（Neural ODE）耦合的快速代理模型（AE‑NODE），能够在不到一分钟的时间内预测长达40小时的严重事故演化，并且在CPU/GPU上均实现了近千倍的速度提升。

**💡 创新点**

①首次实现对ASTEC vessel模块的纯数据驱动代理模型；②通过多分支AE实现对不同空间域（标量、面、体等）的联合降维；③引入自适应窗口（adaptive window）和额外的自回归损失（ℒ_AR^full）提高训练稳定性和预测精度；④显著将高维（1913）状态压缩到仅6个潜在维度。

**🔧 技术方法**

利用深度学习技术：自动编码器（多分支AE）实现非线性降维；神经ODE实现潜在空间的连续时间演化；自回归训练（Teacher Forcing + Autoregressive）与自适应窗口；以及标准的损失函数组合（AE重构、潜在正则、教师强制、自由自回归）。

**📊 数据集**

使用ASTEC模拟器生成的两组事故数据集：约800条LOCA（大破失水）轨迹和约300条SBO（站点停电）轨迹，包含约80个标量和场变量，时间跨度从几千到五万步。

**📈 对比分析**

通过RMSE_mean、RMSE_max和RMSE_std等指标与原始AE重构结果对比；在LOCA和SBO测试集上，AE‑NODE在绝大多数变量上保持了低于0.5的标准化误差；速度对比显示AE‑NODE在CPU/ GPU上分别比ASTEC（仅ICARE模块）快约640×/890×（平均）和510×/720×（中位数），并能实现从10k到50k步的稳定自回归预测。

**⚠️ 局限性**

①对某些高非线性或急剧变化的变量预测误差仍显著；②数据量不足导致模型对极端情景的泛化性有限；③未对原始数据进行充分滤波，可能把数值噪声与物理信号混合；④模型未与主循环（primary circuit）实现闭环反馈，导致在长时间预测中误差积累；⑤最终时间（容器破裂）未被建模，需进一步研究。

---

## 349. A Retrieval-Augmented Framework for Detecting and Resolving Pragmatic Ambiguities in Natural Language Requirements

**arXiv ID:** 2607.04436 | [PDF](https://arxiv.org/pdf/2607.04436v1)

**作者:** Pavithra PM Nair `[一作]` (Tata Consultancy Services), Preethu Rose Anish `[通讯]` (Tata Consultancy Services)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于检索增强生成的框架，用于检测和消除自然语言需求中的语用歧义。

**💡 创新点**

创新之处在于通过构建初级、中级和专家级知识库模拟不同域知识水平的利益相关者，利用澄清问题生成和相似度阈值检测歧义，并生成候选消歧需求。

**🔧 技术方法**

采用检索增强生成（RAG）技术，结合 GPT‑4o‑mini、Llama‑3.1‑8B、Mistral‑7B、Qwen2.5‑7B 等大型语言模型、WikiDoMiner 构建知识库以及余弦相似度阈值判定。

**📊 数据集**

使用 PURE 数据集中的两份需求规范文档（Clarus Weather System Design 与 Vehicle Infrastructure Integration (VII) Data Use Analysis and Processing），以及从 Wikipedia 检索得到的知识库。

**📈 对比分析**

对比四种 LLM，在宏观准确率、精确率、召回率、F1、F2 等检测指标以及人工评估的相关性、清晰度、连贯度进行评估；GPT‑4o‑mini 在检测上取得最高 F2（0.74/0.76）和相关性最高，Mistral‑7B 在清晰度与连贯度上领先。

**⚠️ 局限性**

局限包括仅模拟三种知识水平、依赖 Wikipedia 知识库导致领域覆盖有限、阈值设置和检索参数可能影响结果、人工评估主观性以及仅在交通领域两份文档上验证，缺乏跨领域泛化。

---

## 350. RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies

**arXiv ID:** 2607.04434 | [PDF](https://arxiv.org/pdf/2607.04434v1)

**作者:** Tianxing Chen `[一作]` (MMLab at The University of Hong Kong), Masayoshi Tomizuka `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RoboDojo统一的仿真与真实环境机器人操作基准，包含42个仿真任务和18个真实任务，覆盖一般化、记忆、长时序、精确与开放式指令等能力维度。

**💡 创新点**

创新点在于整合大规模并行仿真与可复现的远程真实评估平台，并通过XPolicyLab提供统一接口，使得策略可一次集成后在仿真与真实环境中无缝评估。

**🔧 技术方法**

采用Isaac Sim异构并行仿真、RoboDojo-RealEval远程云评估、标准化硬件布局、XPolicyLab统一策略接口以及多任务随机化与数据增强等技术。

**📊 数据集**

使用约1.86M帧的仿真演示数据（35个任务共3,500轨迹）和1.61M帧真实演示数据（18任务共1,800轨迹），并提供100条DLC辅助演示和人类专家对照数据。

**📈 对比分析**

对30个代表性策略（含基础和大型视觉语言动作模型）在仿真上做50次/任务、三随机种子评估，并在真实场景中在三种机器人上各10次/任务，比对专家人类操作，结果显示多维度均低于人类且存在显著差距。

**⚠️ 局限性**

局限性包括：1) 真实任务规模受硬件成本限制，无法覆盖所有维度；2) 现有策略在多维度表现不均衡，尤其在记忆、长时序与开放式任务上仍显薄弱；3) 评测仍需依赖标准化硬件，缺乏对多样化部署环境的评估。

---

## 351. Uncertainty-Aware Abstention in Large Language Models with Provable Alignment Guarantees

**arXiv ID:** 2607.04430 | [PDF](https://arxiv.org/pdf/2607.04430v1)

**作者:** Sijin Dong `[一作]` (Ibaraki University), Hiroyuki Shinnou `[通讯]` (Ibaraki University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于置信区间的校准框架CIC，利用LLM产生的任意不确定性分数通过阈值化实现可证明的风险控制的可选择性回答；

**💡 创新点**

创新点在于将任意不确定性估计视为黑盒排序信号，构造置信上界（Hoeffding或Clopper–Pearson）来保证在给定风险水平α下被接受答案的误差率≤α；

**🔧 技术方法**

使用置信区间上界估计、阈值搜索、统计校准；

**📊 数据集**

在CommonsenseQA（闭合式）和TriviaQA（开放式）两个QA基准上，使用七种不同规模的LLM和语义熵不确定性估计器；

**📈 对比分析**

与传统阈值化和无校准策略比较，CIC在保持误差率在目标水平下的同时，回答覆盖率（power）显著高于保守阈值化；实验显示对不同风险水平α，CIC实现了严格的误差率控制并提供了可观的回答率；

**⚠️ 局限性**

限制在极低风险水平或模型/不确定性分数辨别力不足时可能找不到可行阈值；对极小校准集时置信区间可能过宽导致过度保守；

---

## 352. Transferability Between Understanding and Generation in Unified Multimodal Models

**arXiv ID:** 2607.04423 | [PDF](https://arxiv.org/pdf/2607.04423v1)

**作者:** Jiwon Kang `[一作]` (KAIST), Seungryong Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了统一多模态模型（UMM）中图像理解与生成任务的可迁移性，并提出通过在理解任务上训练特定能力（如计数、空间关系、文本识别），利用跨任务迁移来提升生成性能的实用策略。

**💡 创新点**

创新点在于将“可迁移性”作为量化跨任务互动的直接指标，系统揭示架构共享程度决定迁移强度，并证明在不导致分布漂移的前提下，理解任务可有效提升生成能力。

**🔧 技术方法**

技术手段包括对开源 UMM（如 Lumina‑DiMOO、Janus‑Pro、BAGEL、BLIP3‑o）使用 LoRA 微调，构造计数、空间关系与文本合成数据集；评估采用 VQA、GenEval、FID、IS 等指标，并对比理解训练、生成训练及联合训练的效果。

**📊 数据集**

使用的数据集包括 PixMo‑Count、PixMo‑Points、ImageNet、合成计数/空间关系/Markdown 渲染文本图像；评估基准涵盖 POPE、MMBench、MMMU、MME 等通用视觉‑语言测试集。

**📈 对比分析**

通过在理解任务上微调得到的模型，在计数、空间关系和文本生成的生成评估中，既提升了准确率，又保持了与基线相近的 FID/IS，表明性能更稳定；直接在生成任务上微调虽能略微提升精度，却伴随更大的分布偏移。

**⚠️ 局限性**

局限性包括迁移效果随任务细粒度不同而变化（文本生成迁移较弱），联合训练未能同步提升两任务，实验以后训练阶段为主，未能全面反映预训练期间的动态；此外验证仅在有限的几款开源模型上，泛化性仍需进一步探究。

---

## 353. GORIO: GPU-Centered Remote I/O for Graph ANNS over NVMe-oF

**arXiv ID:** 2607.04415 | [PDF](https://arxiv.org/pdf/2607.04415v1)

**作者:** Gen Zhang `[一作]` (National University of Defense Technology), Wenhao Gu `[通讯]` (National University of Defense Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文设计了一个GPU中心的远程NVMe‑oF I/O子系统，支持GPU直接触发页面缺失，并通过CPU代理完成远程读取，从而实现图形近似最近邻搜索（ANNS）在GPU上无阻塞执行。

**💡 创新点**

创新点在于将GPU页面缺失拆分为拆分阶段的远程操作，GPU保持查询状态、未完成状态和恢复决策，CPU仅承担NVMe‑oF传输与完成回写，避免了传统CPU中心化调度对GPU图遍历的干扰；同时引入GPU可见完成表、持久化GPU调度器和无锁请求描述符，实现高效异步重叠。

**🔧 技术方法**

技术手段包括：GPU端的BaM风格页缓存与持久化调度器；SPDK作为NVMe‑oF initiator；RDMA网络实现低延迟传输；GPU可见完成状态表与请求描述符；CPU代理批量提交与完成轮询。

**📊 数据集**

使用的评测数据集为SIFT1M的DiskANN式图索引，采用10,000个查询向量进行召回@10的评估。

**📈 对比分析**

与SPDK‑backed GustANN参考路径、直接BaM‑style NVMe‑oF路径以及GDS页读取基线进行对比；实验显示本方案相较于参考路径提升了1.31×（吞吐量），相较于直接BaM路径提升了3.73×，相较于GDS基线提升了121×，且召回率保持一致。

**⚠️ 局限性**

局限性包括：实验仅在单GPU单服务器环境下进行，未覆盖多GPU或多租户场景；缺失了更大规模索引和不同数据集的评测；页面缺失合并、多路径调度等高级优化仍待进一步研究。

---

## 354. Don't Commit Alone: Joint Token Commitment in Diffusion Large Language Models

**arXiv ID:** 2607.04469 | [PDF](https://arxiv.org/pdf/2607.04469v1)

**作者:** Lin Yao `[一作]` `[通讯]` (Shanghai Jiao Tong University), Lin Yao (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对扩散式大型语言模型在并行解码时产生的联合分布误差，提出了标记门控协调步骤，以在写入令牌前实现位置间的协同。

**💡 创新点**

创新点在于通过在已选位置加入可学习的标记向量，再利用模型最后几层重新前向传播，让这些位置在同一步骤内相互注意，从而近似联合模式的解码，而不需要额外的解码头或辅助模型。

**🔧 技术方法**

技术手段包括：标记门控协调（Marker‑Gated Coordination）+ 对原模型的LoRA低秩适配器 + 额外的部分前向传播（仅三层）和学习的标记向量。

**📊 数据集**

训练使用了 FineWeb‑Edu 50k 文档的通用英文散文；评估则在 TriviaQA、MMLU‑Pro、DROP、CMATH、AIME‑2025 与 GSM‑Plus 六个标准基准上进行。

**📈 对比分析**

与基线 LLaDA2.1‑mini 的因子化解码相比，标记门控协调在所有基准上均取得提升，最显著的是 DROP（+4.38）和 AIME‑2025（+3.33，意味着多解出 1 题），其余基准也均为正向改进。

**⚠️ 局限性**

局限性包括：需对模型进行适配器+标记向量微调；仅针对贪婪解码，温度采样下仍可能出现模式分裂；K=1 的协调深度，无法保证更深层次协调的理论收敛；未在其他 dLLM 系列上验证通用性。

---

## 355. CCFM: Collision-Constrained Flow Matching for Safety-Critical Scenario Generation

**arXiv ID:** 2607.04451 | [PDF](https://arxiv.org/pdf/2607.04451v1)

**作者:** Ke Li `[一作]` (Stony Brook University), Ruwen Qin `[通讯]` (Stony Brook University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种Collision‑Constrained Flow Matching（CCFM）框架，用于在闭环仿真中生成可控、可解释的安全关键碰撞场景。

**💡 创新点**

创新点在于：①引入硬物理约束，显式规定碰撞类型（追尾、侧撞、插入、迎面）并通过Gauss‑Newton流形投影强制满足；②设计启发式碰撞选择器（HCS）动态选取最可行的对抗车辆与碰撞类型；③在流匹配（Flow Matching）采样中插入逐步投影与最优传输逆向更新，保证生成轨迹既逼真又满足碰撞约束。

**🔧 技术方法**

核心技术包括：连续时间流匹配（Flow Matching）生成动作序列；启发式碰撞选择器（基于可达性、几何相似度与道路合法性）；硬约束投影（Gauss‑Newton求解器）；最优传输逆向更新；以及用于评价的多维指标（CR、MS、TM、EN、DS、RM、OR）。

**📊 数据集**

实验数据集：nuScenes（train/val）与nuPlan（mini val）。

**📈 对比分析**

与STRIVE、CCDiff、SAFE‑SIM等SOTA方法在nuScenes、nuPlan闭环仿真中对比，CCFM在相同规划器下实现CR分别达到46.4%（nuScenes）与83.1%（nuPlan），并获得最高的碰撞类型匹配率（84%+）和较高的碰撞严重度（MS≈2.8–5.2 m/s）。虽然在某些现实性指标上略有下降，但整体综合加权得分明显优于基线。

**⚠️ 局限性**

局限性包括：1）与强规划器交互时可能仍无法触发碰撞；2）现实性与严重度存在权衡；3）需要进一步评估碰撞可避免性与生成场景的实际可用性；4）目前仅针对四类碰撞，扩展到更复杂事件仍有挑战。

---

## 356. Wan-Streamer v0.2: Higher Resolution, Same Latency

**arXiv ID:** 2607.04443 | [PDF](https://arxiv.org/pdf/2607.04443v1)

**作者:** Lianghua Huang `[一作]` (Alibaba Group), Zoubin Bi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

升级 Wan-Streamer 版本，将视频分辨率从 192×336 提升到 640×368，并保持约 200 ms 的模型侧响应时延，提供更高质量的实时视频对话和场景扎根的中景代理；

**💡 创新点**

创新点在于引入低时延的单 GPU “思考者”与多 GPU 上下文并行的 “执行者”（Ulysses‑style）相结合的服务拓扑，利用 KV 缓存预分片和序列并行生成高分辨率潜在视频，从而在不牺牲交互时延的前提下提升画面清晰度；

**🔧 技术方法**

核心技术包括端到端的 native‑streaming Transformer、单 GPU 低时延路径（思考者）进行感知、状态更新、KV 构造与最终解码、以及多 GPU 上下文并行（执行者）进行高分辨率潜在视频去噪与解码、Ulysses 风格的 all‑to‑all / gather 通信、以及 KV 切片广播；

**📊 数据集**

未在论文中公开具体数据集，推测使用内部收集的多模态实时对话语料；

**📈 对比分析**

与 v0.1 进行对比，保持 25 fps、200 ms 模型侧时延，整体远程交互时延约 550 ms（与 350 ms 双向网络预算相匹配），显著提升画质与可读性；

**⚠️ 局限性**

局限包括仍受 350 ms 双向网络预算约束，需在多 GPU 环境下部署执行者，对硬件资源与网络带宽要求较高，且仅提升到 640×368 分辨率，进一步提升仍需研究。

---

## 357. ResearchStudio-Reel: Automate the Last Mile of Research from Paper to Poster, Video, and Blog

**arXiv ID:** 2607.04438 | [PDF](https://arxiv.org/pdf/2607.04438v1)

**作者:** Lingao Xiao `[一作]` (Microsoft Research), Yan Lu `[通讯]` (Microsoft Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建一个由五个技能组成的流水线（Paper2Assets、Paper2Poster、Paper2Video、Paper2Blog、Paper2Reel），一次性从论文PDF生成可编辑的海报、同步讲稿视频和双语博客，并通过可交互的Reel页面整合三者。

**💡 创新点**

将“最后一英里”拆解为可组合的技能；使用共享提取层和硬阈值的测量填充循环；生成三种可编辑本地工具文件；实现一个统一交互视图，将海报、视频和博客按章节对齐；在评测中超过先前单一输出系统和单轮LLM。

**🔧 技术方法**

Claude Code/Codex 运行时；Deterministic 原语（headless Chromium、LibreOffice+ffmpeg、python-docx、Edge TTS）；测量填充循环、硬门控；视觉脚本和语音合成；JSON/DOM 转换生成 PowerPoint；对齐侧车文件；交互式HTML+JavaScript。

**📊 数据集**

Paper2Poster 基准集（100篇论文），arXiv 源文件；用于评估海报质量；博客、视频生成不需公开基准；对比先前系统和单轮LLM。

**📈 对比分析**

海报在两名离线 VLM 评审中均领先所有基线，平均美学分数 3.52（高于作者 2.94），整体分数在 84‑93% 的论文中获胜；对视频、博客的能力审计显示它们全部生成编辑版本，且通过硬门控。相比单一输出系统，ResearchStudio‑Reel 的完整覆盖率和可编辑性均更高。

**⚠️ 局限性**

评估指标仍为代理代理的代理，缺乏对真实阅读和记忆的测量；生成的海报不支持自定义图形，难以复制人类设计的概览图；若引入图形生成，可能引入幻觉风险；依赖大量算力和一次性提取时间。

---

## 358. Autonomous Information Seeking: A Roadmap for Agentic Recommender Systems

**arXiv ID:** 2607.04433 | [PDF](https://arxiv.org/pdf/2607.04433v1)

**作者:** Xinyu Lin `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并系统化了代理式推荐系统的研究进展，提出了基于自律度的分类框架，并对技术、架构、评估与挑战进行归纳。

**💡 创新点**

创新点在于引入Level‑of‑Autonomy (LoA) 维度，统一定义代理式推荐系统，构建双轴（角色×自律度）词典，提供完整评估指标与挑战清单。

**🔧 技术方法**

使用的大技术包括大型语言模型 (LLM) 与工具调用、检索增强生成 (RAG)、记忆模块、规划与反思循环以及多代理协作。

**📊 数据集**

综述中引用了多种基准数据集，如 MovieLens、Amazon、Goodreads、RecSys Challenge 等，并涉及合成与仿真数据。

**📈 对比分析**

通过离线排名指标（NDCG、Recall）与对话质量指标（成功率、交互轮数）、工具使用成功率、记忆命中率等多维度评估，指出多数工作在离线指标上与传统方法相当或略优，但缺乏统一对比与在线评估。

**⚠️ 局限性**

限制包括评估侧重点不足，偏重静态排名；代理系统的真实世界迁移、模拟误差、计算与延迟成本未充分探讨；缺乏统一基准与跨领域可复制性。

---

## 359. evalci: A Python Library for Statistically Rigorous Comparison of Language Model Evaluations

**arXiv ID:** 2607.04429 | [PDF](https://arxiv.org/pdf/2607.04429v1)

**作者:** Shreyas K Chandrahas `[一作]` `[通讯]`, Shreyas K Chandrahas

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个纯 Python 库 evalci，将每条评测结果表格转换为带置信区间、显著性检验和多重比较校正的发表级报告。

**💡 创新点**

将成熟的统计方法（Wilson、bootstrap、置换、McNemar 等）包装成适用于评测表格的工具，并提供完整验证与 CLI 适配器，填补评测结果缺乏误差估计的空白。

**🔧 技术方法**

使用 Wilson/Clopper‑Pearson 置信区间、bootstrap、配对/无配对置换检验、McNemar 检验、Holm/BH 多重比较校正、聚类标准误、功效分析等统计技术。

**📊 数据集**

案例研究使用公开的 5‑shot MMLU（14,042 题）准确率表格，对九个模型的差异进行检验。

**📈 对比分析**

通过 evalci.compare 计算模型差异的 Δ、95% CI、p 值；在案例中发现 8 个相邻排名中有 3 个差距在 Holm 校正后不显著，说明先前的排行榜误差估计不充分。

**⚠️ 局限性**

依赖聚合准确率重建的假设，缺乏真实 per‑item 相关性导致结果保守；若无 per‑item 日志，无法充分利用配对检验；置换/ bootstrap p 值在极小 p 值下精度有限；连续或相关分数需使用 bootstrap；功效模拟参数 ρ 简化了难度相关性。

---

## 360. ACE-Brain-0.5: A Unified Embodied Foundational Model for Physical Agentic AI

**arXiv ID:** 2607.04426 | [PDF](https://arxiv.org/pdf/2607.04426v1)

**作者:** ACE-Brain Team `[一作]`, Xiaogang Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种统一的具身基础模型（ACE‑Brain‑0.5），将空间感知、决策制定、具身交互、自我监测和自我提升等五大认知功能集成在单一8B混合Transformer模型中，并通过SSR+训练策略实现跨任务的统一输出；

**💡 创新点**

创新点在于：①将多模态感知、规划、执行与评估合并为闭环Perception‑Planning‑Action‑Evaluation流程；②引入SSR+训练（Scaffold‑Specialize‑Reconcile‑Reactivate）实现任务向量融合后轻量化再激活，避免跨任务干扰；③通过Fast Vision Pathway与flow‑matching Action Expert实现实时控制；④构建可扩展的自我提升框架，将模型自身执行回溯作为监督信号；

**🔧 技术方法**

技术手段包括：混合Transformer主体+LLM Decoder、共享Vision Encoder、Fast Vision Encoder（DINOv3）、flow‑matching Action Expert、任务向量融合与微调、SSR+训练流程、基于视觉-语言的进度估计模块、以及导航自我提升数据流；

**📊 数据集**

使用的数据集涵盖：空间感知与QA（VSI、VLM‑3R、GPT4Scene、MindCube、ScanQA、SQA3D等）；语义对齐与定位（RefSpatial、PixMo‑Points、RoboPoint、ScanRefer、Multi3DRef等）；导航（R2R、RxR、EnvDrop、ScaleVLN、SRDF‑400K、VLN‑CE action接口）；操纵（LIBERO、SimplerEnv‑Bridge、LIBERO、SimEnv‑Bridge等）；进度估计（RBM‑1M、RoboMeter、RBM‑EVAL‑Refined）；以及ACE‑Brain‑0的预训练素材（VSI、VLM‑3R、GPT4Scene等）；

**📈 对比分析**

通过在15+基准上进行对比，ACE‑Brain‑0.5在大多数空间感知/引用、导航（R2R、RxR）和操纵（LIBERO 98.2%、SimplerEnv‑Bridge 82.3%）任务上均优于ACE‑Brain‑0，进度估计VOC达到0.94–0.96，远超传统奖励模型；在导航与操纵上与最新的VLA/World‑Action模型保持竞争力，且在多模态感知和规划方面实现了显著提升；

**⚠️ 局限性**

局限性包括：自我提升机制仍局限于导航且为轻量化、基于外部执行状态，缺乏通用的模型级自我进化；对更广泛的硬件平台与更长时限任务的迁移与泛化尚待验证；在极端动态或非结构化环境下的鲁棒性尚不足；

---

## 361. LLM-as-a-Tutor: Policy-Aware Prompt Adaptation for Non-Verifiable RL

**arXiv ID:** 2607.04412 | [PDF](https://arxiv.org/pdf/2607.04412v1)

**作者:** Yujin Kim `[一作]` (KAIST), Hwanjun Song `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LLM-as-a-Tutor框架，利用同一模型既评判又生成训练提示，通过检测两条 rollout 的质量差异，若无差异则追加约束以保持奖励信号的区分性；

**💡 创新点**

核心创新在于将 LLM 从单纯的判断者升级为“导师”，通过二元比较精准识别非挑战性提示，并采用追加而非重写方式动态提升难度，形成自校准的训练曲线；

**🔧 技术方法**

技术手段包括基于 LLM 的判分器与约束生成器、基于 Rubric 的连续奖励、GRPO 策略优化，以及单步调用实现的 append‑only 适配策略；

**📊 数据集**

使用 WildChat 作为种子提示集，评估数据集为 FollowBench、AdvancedIF 与 InfoBench；

**📈 对比分析**

与无提示适配、基于 Rubric 的自适配（Policy‑adaptive Rubrics）以及基于提示重写的 Evol‑Instruct/EVA 等基线比较，LLM‑as‑a‑Tutor 在三大基准上均取得最高平均分，明显优于所有对比方法；

**⚠️ 局限性**

局限性包括仅适用于可通过追加约束来调节难度的任务，重写或非追加式修改效果不佳；依赖 LLM 的判分能力与生成质量，且对难度定义过于依赖约束计数，可能不易迁移到需要更复杂难度度量的领域。

---

## 362. Compressing the Validation Bottleneck: An Agentic Self-Driving Lab for Scientific Discovery

**arXiv ID:** 2607.04508 | [PDF](https://arxiv.org/pdf/2607.04508v1)

**作者:** Kyunghoon Hur `[一作]` (Korea Electronics Technology Institute), Chihun Lee `[通讯]` (Korea Institute of Materials Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种融合先验知识和成本感知的自驱动实验室（SDL）框架，利用代理自动化实验设计（DOE）和低成本测量的代理推理，显著降低实验循环次数和单次实验成本。

**💡 创新点**

创新点在于：①将领域先验、实验反馈与实验可行性约束整合进贝叶斯优化的代理决策，形成“先验感知的代理式 DOE”；②设计“成本感知代理”，通过多源低成本测量预测高成本高分辨率实验结果，并根据不确定性动态选择测量方式，从而压缩实验成本。

**🔧 技术方法**

核心技术包括：大规模语言模型（LLM）驱动的代理推理、贝叶斯优化（BO）实验设计、基于预训练的多模态代理模型（硬度、XRD、CALPHAD 等）以及不确定性校准和代理决策。

**📊 数据集**

使用的数据集涵盖：抗体生产工艺的规模化实验记录（多参数生物反应器数据）以及金属增材制造（AM）过程中的硬度、拉伸强度、XRD 相位与 CALPHAD 相图数据，均为真实实验测量数据。

**📈 对比分析**

与人工指导 DOE、随机搜索、网格搜索及传统 BO 进行对比。实验表明，先验感知代理显著减少到达目标所需的实验次数，并降低无效/不可行实验比例；成本感知代理在相同测量预算下提升目标命中率、减少高成本测量次数，并缩短总实验耗时。

**⚠️ 局限性**

局限性包括：①代理对先验知识的依赖强，若先验不充分或不准确可能导致实验偏离；②代理模型的不确定性估计需要充分的校准，误估可能导致过度或不足的高成本测量；③在极端实验条件或非线性极端场景下，低成本测量的预测能力可能不足，需要进一步提升模型泛化。

---

## 363. Hybrid Algorithmic Governance in U.S. Welfare Administration: State- and County-Level AI as a Case of Support-Control Convergence

**arXiv ID:** 2607.04503 | [PDF](https://arxiv.org/pdf/2607.04503v1)

**作者:** Maxim Dedyaev `[一作]` `[通讯]`, Maxim Dedyaev

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

研究美国州级和县级福利管理中人工智能系统的支持‑控制取向动态，并通过过程追踪六个案例检验“支持‑控制收敛”与“制度计杠”的机制。

**💡 创新点**

提出“支持‑控制收敛”概念和“制度计杠”机制，将混合型状态层级从静态分类转化为动态轨迹，并通过六参数编码框架验证。

**🔧 技术方法**

使用过程追踪与结构化比较方法，构建六参数编码框架，对案例进行定性分析。

**📊 数据集**

采用六州/县案例的文档证据集：机构报告、预算文件、审计、司法与立法文件、媒体与倡导组织报告以及独立学术评估。

**📈 对比分析**

通过对比六案例在不同时间点的编码配置，检验假设的因果测试；方法在理论检验上表现可靠，但无量化性能指标。

**⚠️ 局限性**

局限包括：单一研究者编码导致的主观偏差；文献证据密度不均导致静默漂移被低估；案例数量有限，无法实现统计推广；部分案例时间不完整。

---

## 364. From Interaction to Intent: Inferring User Objectives from Provenance Logs

**arXiv ID:** 2607.04501 | [PDF](https://arxiv.org/pdf/2607.04501v1)

**作者:** Steffen Holter `[一作]` (ETH Zurich), Mennatallah El-Assady `[通讯]` (ETH Zurich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过采集非专家用户在多维投影可视化中的鼠标悬停日志，构建原型来推断用户在探索任务中的意图。

**💡 创新点**

提出一种投影无关的上下文化框架，将低层交互数据与数据空间属性关联，使得意图识别能跨数据集和投影算法泛化；并在单一任务与连续多任务两种场景下完成在线意图预测。

**🔧 技术方法**

使用摘要式和时序式两种上下文化；基于摘要特征的XGBoost；基于时序特征的BiGRU；以及在线多任务的UniGRU+CRF进行分类。

**📊 数据集**

收集了264名参与者共1439条可用交互序列，使用四个不同主题（天气、口袋妖怪、篮球、食谱）并分别采用PCA、t‑SNE、UMAP投影，生成12种可视化布局。

**📈 对比分析**

采用留一源交叉验证（LOSO）和全新数据集（食谱）作为测试，原子任务的准确率约为60–65%，三类任务提高至约73%；在线多任务的片段F1约为45–55%，边界检测性能最弱。

**⚠️ 局限性**

实验仅使用鼠标悬停，不包含其他感知信号；多任务序列为人工拼接，缺乏真实连续探索的模糊过渡；只涉及非专家用户，难以验证专家行为；且在极端投影或更复杂任务下的泛化仍待评估。

---

## 365. Enhancing Facial Expression Recognition in Head-Mounted Displays with Synthetic Data

**arXiv ID:** 2607.04490 | [PDF](https://arxiv.org/pdf/2607.04490v1)

**作者:** Jianing Deng `[一作]` (University of Pittsburgh), Jingtong Hu `[通讯]` (University of Pittsburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个将前视（FV）图像转换为头戴摄像头（HMC）视角的合成数据框架（SynHMC），并利用该框架生成大规模HMC FER数据集，进一步训练并评估表情识别模型。

**💡 创新点**

创新点包括：①提出纹理空间对齐网络（TSAN）以精确对齐纹理并保留细节；②设计可配置的HMD相机系统，能生成多视HMC图像；③通过合成数据填补HMC FER数据稀缺，显著降低FV与HMC之间的域差。

**🔧 技术方法**

技术手段主要有：FLAME 3DMM+EMOCA重建、GFP-GAN去噪上采样、U-Net+位置编码的TSAN、ARAP刚性变形、PyTorch/PyTorch3D渲染、随机视角数据增强等。

**📊 数据集**

使用的数据集包括：RAF‑DB、AffectNet（280K）、合成的SynHMC（7K）、3D扫描数据集Ava256‑Scan与3DRFE、真实HMC数据Ava256‑HMC、纹理库FFHQ‑UV、MEAD等。

**📈 对比分析**

通过在3D扫描和真实HMC数据上对比实验，SynHMC训练的模型在UAR/WAR/Accuracy上均优于以FV数据训练的基准模型，尤其在多视角跨配置泛化上表现突出；在Sim‑to‑Real任务中，虽然存在差距，但经过少量微调仍能超过现有方法。

**⚠️ 局限性**

局限性在于：①仅针对视角域差，未显式处理光照或色彩差异；②依赖FLAME 3DMM，缺乏对口腔内部细节的建模；③合成纹理仍受源图像质量限制；④Sim‑to‑Real 转换仍存在一定误差。

---

## 366. TrustCLIP: Learning Private Visual Features via Adversarial Reconstruction

**arXiv ID:** 2607.04484 | [PDF](https://arxiv.org/pdf/2607.04484v1)

**作者:** Nikos Athanasiou `[一作]` (Meta), Bugra Tekin `[通讯]` (Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

TrustCLIP通过在冻结的CLIP编码器前插入轻量级投影层，并与生成式逆向攻击器IP‑Adapter联合对抗训练，削弱特征可逆性同时保持下游任务的准确率。

**💡 创新点**

首次针对生成式逆向攻击直接优化特征投影，构建了可调节的隐私‑实用权衡，并证明对抗训练能显著抑制重建细节。

**🔧 技术方法**

使用的关键技术包括轻量级投影MLP、IP‑Adapter+Stable Diffusion生成器、对抗式联合优化（任务损失+重建损失）、以及身份初始化与残差连接。

**📊 数据集**

实验在图像分类任务使用SUN397数据集，在多模态语言模型任务中使用LLaVA‑SP基准及其下游评测数据集（POPE、VQAv2、MM‑Vet等）。

**📈 对比分析**

与未加防护基线、随机噪声基线以及多种公开VLM进行对比；在分类任务中Top‑1准确率仅下降≤0.5%，而LPIPS/DSIM提升约2–3倍；在VLM评测中保持与LLaVA‑SP相近的性能，同时重建质量明显下降。

**⚠️ 局限性**

局限在于仅针对CLIP编码器与IP‑Adapter生成器训练，对细粒度任务如OCR、计数的性能损失较大，且未覆盖视频、时序或多视角情境的隐私攻击。

---

## 367. EVAS: Efficient Multimodal Temporal Forgery Localization via Audio-Visual Synergy and Steered Boundary Calibration

**arXiv ID:** 2607.04472 | [PDF](https://arxiv.org/pdf/2607.04472v1)

**作者:** Shen Shen `[一作]` (Soochow University), Ke Zhang `[通讯]` (Soochow University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出了EVAS框架，实现了长视频中伪造片段的精细时序定位。

**💡 创新点**

创新点包括：多阶段音视频协同机制（MAVS）实现深度跨模态交互；边界感知细化（BAR）结合解耦训练-推理策略以抑制误差传播；以及轻量化HourglassFFN实现高效推理。

**🔧 技术方法**

主要技术手段包括Transformer跨模态注意力、VideoMAE-S与BYOL-a特征编码、Mask Invalid Frames与Distance IoU Loss等。

**📊 数据集**

使用Lav-DF、AV-Deepfake1M、TVIL三大基准数据集进行实验。

**📈 对比分析**

与现有方法对比，EVAS在Lav-DF上AP@0.95达88.63%，AV-Deepfake1M上mAP70.20%，TVIL上mAP83.54%，轻量化版本推理时间仅为50 ms，显著优于竞争模型。

**⚠️ 局限性**

局限性：在仅存在单模或音频缺失情况下性能下降；训练需要精确伪造边界标注；对极端视频压缩或噪声的鲁棒性仍有提升空间。

---

## 368. Flash-BoN: Instant Drafts for Inference-Time Scaling in Diffusion Models

**arXiv ID:** 2607.04461 | [PDF](https://arxiv.org/pdf/2607.04461v1)

**作者:** Ruchit Rawal `[一作]` (University of Maryland), Gowthami Somepalli `[通讯]` (University of Maryland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种在扩散模型推理时生成草稿候选并通过多阶段验证选取最佳进行完整生成的推理缩放方法。

**💡 创新点**

将时间截断、层跳过与激活代理三种加速技术统一配置，并采用一次性预计算的草稿策略与多阶段点对点验证相结合，实现在固定壁时预算下显著提升多模型、多基准的质量。

**🔧 技术方法**

组合时间截断、层跳过、激活代理；离散优化预设草稿配置；点对点与单点评分的多阶段Elo排名；在RL后训练中采用草稿采样。

**📊 数据集**

使用 GenAI-Bench、GenEval、UniGenBench 等文本到图像基准，并在 Wan 2.1 1.3B/14B 及 FLUX.1-dev 模型上评估。

**📈 对比分析**

与 Best‑of‑N、BFS、DFS、ZOS 等传统推理缩放方法对比，采用壁时预算和 AUC/Time 指标；在所有模型–基准组合中均取得最高 AUC，尤其在大模型上提升可达 +8% AUC。

**⚠️ 局限性**

对草稿配置的预计算依赖于校准集，若模型或任务差异大可能不再有效；在极低成本或极小模型时加速手段效果有限；验证器与评估指标不匹配时易产生偏差。

---

## 369. UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI Agent Learning

**arXiv ID:** 2607.04425 | [PDF](https://arxiv.org/pdf/2607.04425v1)

**作者:** Niu Lian `[一作]` (Tsinghua University), Jinpeng Wang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建统一的跨平台数据集 Uni-GUI 并提出多教师在线蒸馏（MOPD）方法，使单一 GUI 代理在桌面和移动平台上同时保持并提升交互性能

**💡 创新点**

创新点在于通过平台条件路由的多教师在线蒸馏，既保留各平台的交互习惯，又避免了传统混合训练导致的行为平均与灾难性遗忘

**🔧 技术方法**

技术包括：统一跨平台数据采集、基于 Qwen3‑VL 的视觉语言模型、平台条件教师路由、K3 估计的在线 KL 蒸馏、强化学习与结构化奖励结合的训练目标

**📊 数据集**

数据集为自研的 Uni‑GUI，约 10k 条高质量跨平台交互轨迹，包含 110k 条桌面和 50k 条移动交互步骤

**📈 对比分析**

与混合 SFT、模型平均、TIES 合并以及单平台/多平台 GUI 基线比较，MOPD 在 OSWorld 的成功率达 38.2%（+12.7%），在 MobileWorld 达 12.0%（+55.8%）

**⚠️ 局限性**

局限在于仅覆盖桌面与移动两平台，依赖昂贵的数据标注，蒸馏过程对教师质量敏感，且跨平台泛化至更多设备或操作系统仍待验证

---

## 370. Spatial Graph Representation and Morphometric Analysis of the Pulmonary Vascular Tree From Computed Tomography Using Multi-Scale Hessian-Based Filter Fusion and TEASAR Skeletonization

**arXiv ID:** 2607.04457 | [PDF](https://arxiv.org/pdf/2607.04457v1)

**作者:** Piotr Mackiewicz `[一作]` (Warsaw University of Technology), Radoslaw Roszczyk `[通讯]` (Warsaw University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文基于SPECT/CT低分辨率非对比胸CT图像，提出并实现了一套肺血管树重建及形态学分析完整流程；

**💡 创新点**

创新点在于将Frangi与Sato两种血管性增强滤波器以加权方式融合，显著提高血管节点与分支检测率，并对低分辨率CT数据进行可行的形态学验证；

**🔧 技术方法**

采用了多尺度血管性增强滤波（Frangi、Sato）、TEASAR骨架化、图形构建及形态学指标计算（Strahler阶、分形维数、Murray指数、Horton比、曲度）等技术；

**📊 数据集**

使用单个来自SPECT/CT扫描的非对比胸CT肺部图像作为数据集；

**📈 对比分析**

通过与文献中已知的形态学参考值（如分形维数、Murray指数、Horton比）进行对比，融合配置在节点数、分形维数、Murray指数与Horton比方面与已有研究相符，性能优于单一滤波器；

**⚠️ 局限性**

局限性包括仅使用单例样本、缺乏真实标注进行定量验证，以及扫描分辨率低导致部分微血管缺失，影响重建完整性和指标偏差。

---

## 371. Full-Stack FP4: Stable LLM Pretraining with Quantized Projections, Optimizers, and Attention

**arXiv ID:** 2607.04422 | [PDF](https://arxiv.org/pdf/2607.04422v1)

**作者:** Siyu Ding `[一作]` (Institute of Automation, Chinese Academy of Sciences), Guoqi Li `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了完整的4-bit NVFP4预训练方案，覆盖线性层、优化器状态与计算、以及注意力模块的量化；

**💡 创新点**

创新点在于三种模块各自的量化策略：LoRA‑SVD结构分解降低线性层量化噪声、针对AdamW第二矩的变换管道稳定量化、以及混合精度注意力保持前后向一致性；

**🔧 技术方法**

使用NVFP4格式、LoRA‑SVD、Hadamard变换、根号变换、Newton‑Schulz迭代、以及混合精度FlashAttention等技术；

**📊 数据集**

在3B规模模型上使用Nemotron‑CC v2高质量数据集训练64B个token；

**📈 对比分析**

与BF16+Root+AdamW基线对比，训练损失差距仅为1.47%，LoRA‑SVD使线性层的损失差距从1.40%降至0.61%，证明低精度训练稳定且性能可接受；

**⚠️ 局限性**

受限于实验规模（仅3B模型、64B token）及未实现完整CUDA优化，尚未验证更大规模或更长训练时间下的泛化性。

---

## 372. Agent Step Value: State-Transition Measurement with State-Grounded LLM Evaluators

**arXiv ID:** 2607.04419 | [PDF](https://arxiv.org/pdf/2607.04419v1)

**作者:** Andrew Zhang `[一作]` (KTH Royal Institute of Technology), Chengzhan Li `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出Agent Step Value（ASV）框架，用以量化语言模型代理在每一步行动后对固定候选答案的信念变化，从而对多步骤代理行为进行细粒度诊断。

**💡 创新点**

创新点在于将LLM推理与单词级评分分离，采用无标签理据、熵变、贝叶斯惊奇以及金标边际增益等多维诊断量化每一步的效用，并通过无状态评估器实现可重复的状态转移测量。

**🔧 技术方法**

技术实现包括无状态LLM评估器、标签无理据生成、熵变与贝叶斯惊奇计算、Gold‑Margin Gain 统计，以及标签置换平均评分等。

**📊 数据集**

实验使用 100 个开放式问答（Open‑QA）任务，任务来源于 PubMed 检索，候选答案固定为 4 个（含 abstain），actor 与评估器均采用 DeepSeek 模型。

**📈 对比分析**

与直接一词评分协议对比，ASV 在理据条件下的平均 gold‑margin gain 为 -2.335，entropy 为 0，表明传统熵度量无法捕捉到大幅度的信念转移；相比之下直接评分得到正增益，说明评估协议对结果影响显著。

**⚠️ 局限性**

局限性在于仅针对单一 100 题 QA 集、固定候选集、DeepSeek 及 PubMed 数据进行评估，结果缺乏泛化性；评估器为 LLM，易受模型偏差、候选构造及选项顺序等因素影响。

---

## 373. Transplanting, inverting, and preventing a misalignment persona: method-conditional emergent misalignment in Qwen2.5

**arXiv ID:** 2607.04510 | [PDF](https://arxiv.org/pdf/2607.04510v1)

**作者:** Lyndon Drake `[一作]` (University of Oxford), Zandi Eberstadt `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了 Qwen2.5 系列模型在低秩 LoRA 与完整 SFT 训练方式下的出现性失调（EM）现象，发现 LoRA 会招募隐藏的“失调人格”方向导致广泛失调，而完整 SFT 在隐蔽诱导下不会。

**💡 创新点**

创新点在于：①首次将失调人格方向作为 EM 的因果中介并证明其在不同训练方法与容量下的招募差异；②提出跨模型移植实验以验证人格方向的因果充分性；③揭示完整 SFT 能在隐蔽诱导下避免招募，并构建“损失捷径”机制与条件性缓解策略。

**🔧 技术方法**

技术手段包括低秩 LoRA (rs-LoRA)、完整监督微调、跨模型移植、人格方向提取（误差均值差分）、权重更新富集分析、对齐消退 vs 人格放大两轴几何度量、L2-SP 正则、损失梯度归因、免疫化训练、人格正交微调。

**📊 数据集**

使用的数据集为：隐蔽安全/不安全代码补全（6k 对），好/坏医疗建议（7049 对），以及风险金融和极限运动等有害文本补全，另借助 GPT‑4o 进行对比。

**📈 对比分析**

对比方法：在 Qwen2.5‑32B 基础与指令模型上对 LoRA 与完整 SFT 进行同一数据、同一提示的单轮训练，评估“误导率”（符合条件的连贯回答中失调占比）与人格轴投影；实验显示 LoRA 误导率 ~3.4%，完整 SFT ~0.3%；阶梯实验显示 EM 随 LoRA 维度升高而下降；跨模型移植诱导 ~2.8% 失调；人格消融将 ~21% 失调降至 ~10%；免疫化和人格正交微调可将失调率降至接近基线。

**⚠️ 局限性**

局限性：仅在 Qwen2.5 系列单一模型族内实验；多处实验仅单种随机种子；失调率整体偏低且受二值评判器噪声影响；跨模型移植排除了一种种子；几何距离度量弱，真正的距离指标尚未测得；规模与优化可实现性跨越两个尺度，未形成连续规律。

---

## 374. Eiger: An Efficient Library for GPU-based Data Analytics

**arXiv ID:** 2607.04489 | [PDF](https://arxiv.org/pdf/2607.04489v1)

**作者:** Bowen Wu `[一作]` (ETH Zurich), Gustavo Alonso `[通讯]` (ETH Zurich)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了Eiger库，提供GPU数据分析的多实现变体与运行时自适应选择；

**💡 创新点**

通过为每个操作提供多种实现和可调节参数，并利用GPU轻量级统计实时决定实现与压缩，提升了表达式评估、字符串处理及多键排序等常被忽视的操作；

**🔧 技术方法**

使用CUDA编程、packed访问、cooperative groups、HyperLogLog++统计、order‑preserving dictionary编码、智能键融合（SKF）、多种join/group‑by/sort实现、PTI与BB表达式评估、KMP字符串匹配等技术；

**📊 数据集**

评测基于TPC‑H基准（规模因子10、30、100）以及各种微基准（百万级行、字符串等）数据集；

**📈 对比分析**

与cuDF（及Maximus执行引擎）对比，22条TPC‑H查询中Eiger best相对cuDF的加速比为1.7–1.8×，单个查询最高可达6.1×；微基准显示在A100和GH200上，多实现实现显著提升；

**⚠️ 局限性**

仅针对单GPU内存场景，未覆盖多GPU集群；实现对统计与压缩的开销仍需评估，缺乏完整的成本模型与自适应重优化机制。

---

## 375. Two Black Boxes, One Solver: Encoder Probing and Decoder Attribution for Neural Multi-Attribute VRP under Hard-Mask and Recourse Decoders

**arXiv ID:** 2607.04487 | [PDF](https://arxiv.org/pdf/2607.04487v1)

**作者:** Sohaib Afifi `[一作]` `[通讯]` (University of Artois), Sohaib Afifi (University of Artois)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在多属性车辆路径问题（MAVRP）上提出了一套双层可解释性评估框架，结合编码器探测与解码器归因；

**💡 创新点**

创新点包括：一是提出两柱XAI协议与约束族分类；二是将梯度、IG、DeepLIFT等归因方法与对抗性干预相结合；三是系统评估分布式编码、可执行反事实与动态解码器状态的影响；

**🔧 技术方法**

使用图神经网络编码器、Transformer与Mixture‑of‑Experts层，配合注意力解码器；归因方法为梯度、Integrated Gradients、DeepLIFT；干预与反事实搜索采用第一阶梯度估计；

**📊 数据集**

基于统一参数化的MAVRP实例，规模主要为50客户（部分实验扩展至100客户）；

**📈 对比分析**

与OR‑Tools、MTPOMO、MVMoE等基准相比，六种模型在成本差距≤2% 内保持竞争力；同时在可信度、稳定性、可操作性等XAI指标上实现显著提升；

**⚠️ 局限性**

局限性：依赖预定义的概念字典；反事实搜索为固定预算第一阶方法，可能低估可操作性；实验仅在有限种子与规模下进行；未涵盖调度等其它离散优化任务。

---

## 376. LeukocyteCount: Automatic Identification and Counting for leukocytes using Deep Learning

**arXiv ID:** 2607.04486 | [PDF](https://arxiv.org/pdf/2607.04486v1)

**作者:** Ahmed M. Sayed `[一作]` (Helwan University), Ensaf Hussein Mohamed `[通讯]` (Nile University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

实现了基于深度学习的白细胞（WBC）自动检测、计数和四类分类系统，涵盖RBC、Platelets及四种WBC类型。

**💡 创新点**

创新点在于将YOLOv5与MobileNetV2的特征提取结合，并使用Logistic Regression实现高精度分类，构建了一个端到端、可直接计数的混合模型。

**🔧 技术方法**

使用YOLOv5进行目标检测，MobileNetV2提取特征后输入Logistic Regression做分类；同时对图像做预处理和裁剪。

**📊 数据集**

主要使用公开的BCCD数据集（含RBC、WBC、Platelets及四种WBC子类型），并通过Roboflow进行图像增强。

**📈 对比分析**

与先前研究相比，WBC检测精度达到100%（YOLOv5），分类精度提升至99.04%（MobileNetV2+LR），整体计数准确率接近99%以上，明显优于基准方法（如Ensaf等的MobileNetv1+LR 97.03%）。

**⚠️ 局限性**

局限性包括：模型主要在高质量、光照良好的显微图像上训练，低分辨率或噪声图像性能可能下降；数据来源单一，缺乏跨实验室、不同染色方式的验证；测试集样本量有限，需进一步扩展外部数据验证。

---

## 377. Regime-Conditional Stabilisation of LLM-Augmented Cooperative Multi-Agent Reinforcement Learning

**arXiv ID:** 2607.04470 | [PDF](https://arxiv.org/pdf/2607.04470v1)

**作者:** Faid Keddouri `[一作]` (National School of Artificial Intelligence), Nadir Farhi `[通讯]` (University Gustave Eiffel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文研究在离线经验回放的多智能体强化学习中，动态更新由大语言模型生成的奖励权重会破坏潜在奖励塑形的平稳性，导致训练崩溃。

**💡 创新点**

创新点在于首次揭示并系统分析该非平稳性对学习的影响，并提出阶段冻结和指数移动平均两种稳定化策略，形成三种环境“必要、增量、补充”三重分类。

**🔧 技术方法**

采用了潜在奖励塑形（PBRS）、量化的 Qwen 2.5 3B LLM 作为奖励架构、QMIX 与 VDN 价值分解网络，以及 EMA 平滑与冻结调度两种平稳化手段。

**📊 数据集**

实验使用了三种协作环境：Simple Spread、Level‑Based Foraging 与 SMAC 3m，并在每个环境下多重种子下训练。

**📈 对比分析**

与无塑形基线相比，冻结或 EMA 方案在 Simple Spread 上从 74.4% 提升到 86.7%，在 Level‑Based Foraging 上从 0.1% 提升到 95.9%，在 SMAC 3m 上保持 99.9%，而无平稳化的动态更新则在 Simple Spread 上骤降至 15.2%。

**⚠️ 局限性**

限制在于仅用单一 LLM 与三种环境进行验证，未对不同更新频率、缓冲区大小等因素进行因果探索，且对基准政策的先验假设和特定算法（QMIX）可能产生偏差。

---

## 378. Robustness Verification of an Autonomous Underwater Vehicle-based Plankton Classifier

**arXiv ID:** 2607.04453 | [PDF](https://arxiv.org/pdf/2607.04453v1)

**作者:** Abdelrahman Sayed Sayed `[一作]` (University Gustave Eiffel), Mohamed Ghazel `[通讯]` (University Gustave Eiffel)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于可达性分析的鲁棒性验证框架，并设计了一种紧凑型神经ODE分类器，用于在AUV上对水下浮游生物进行高分辨率图像分类。

**💡 创新点**

创新点在于将混合单调性可达性与星集传播相结合，实现对连续时间神经ODE模型的正式鲁棒性证明，并发现Tanh激活函数在验证难度上优于ReLU，从而为可验证的深度学习模型提供了新的设计思路。

**🔧 技术方法**

所用技术包括神经ODE（连续深度网络）、混合单调性可达性分析、星集（star-set）传播、PyTorch训练与adjoint ODE求解器，以及预判定的随机取样与正式验证循环。

**📊 数据集**

实验采用SilCam光学成像数据集，包含7类浮游生物共7738张图像，尺寸为8×8×3，数据来自PyOPIA工具箱。

**📈 对比分析**

通过对ReLU与Tanh两种变体进行比较，针对局部（1或10像素）与全局噪声攻击进行验证，局部攻击在约100–130秒内完成正式验证，全图像攻击在噪声小（0.01/255）时可证实鲁棒性，噪声大（5/255/10/255）时快速反例检测；Tanh在0.05/255时显著优于ReLU，验证耗时更短。

**⚠️ 局限性**

局限性包括验证时间在更大噪声半径（如全图像1/255）下会超时，导致可达性过度保守；仅适用于低分辨率输入（8×8×3）且难以直接扩展到更高分辨率；并且只对L∞噪声做了正式证明，未覆盖其他类型的环境扰动。

---

## 379. Progressive Disclosure for LLM-Maintained Wiki Knowledge Bases: a Preregistered Ablation

**arXiv ID:** 2607.04576 | [PDF](https://arxiv.org/pdf/2607.04576v1)

**作者:** Theodore O. Cochran `[一作]` `[通讯]` (AI for Altruism), Theodore O. Cochran (AI for Altruism)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在真实的709页LLM维护的markdown知识库上，测试了将知识库结构改为渐进披露（即裁剪索引、添加一行摘要、引入关键字检索工具）对LLM代理回答质量和成本的影响。

**💡 创新点**

创新点在于通过内容相等（内容不变、只改结构）控制实验，精确分离访问结构对质量与成本的因果效应，并揭示传统索引避免成本并非主要机制，而是更精准的访问导致成本下降。

**🔧 技术方法**

使用了Claude Opus 4.8作为回答模型、OpenAI GPT‑5作为交叉族评判模型；工具集包含文件读取、检索、搜索等；采用了预注册的混合效应模型、Bootstrap置信区间以及缓存鲁棒的新令牌成本度量。

**📊 数据集**

实验数据来自一套709页Markdown知识库，四种结构变体（A0‑A3）与三种工具访问条件（强制、自由、强制预加载）交叉，随机化共960次问答运行；问题集为40个，按检索范围分层。

**📈 对比分析**

与基准索引结构对比，改进结构在所有访问条件下成本降低（强制60%、自由34%、强制预加载30%），回答质量保持非劣势（整体复合得分+0.01，非劣势门限0.5），在自路由条件下甚至略优；通过每页摘要和检索工具减少了被引用页数和工具调用次数。

**⚠️ 局限性**

局限包括单一知识库、单一回答模型、仅作者为人类评审、缺乏对读取细节的直接计量、评审一致性低（Kappa 0.23）以及对“强制预加载”条件下质量非劣势未得到充分验证。

---

## 380. Identifying Deceptive Patterns Across Three Age Groups: A Heuristic-Based Cognitive Walkthrough Study of Mobile Apps

**arXiv ID:** 2607.04573 | [PDF](https://arxiv.org/pdf/2607.04573v1)

**作者:** Nasra Hassan `[一作]` (Carleton University), Hala Assal `[通讯]` (Carleton University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

对30款Android移动应用进行基于启发式的认知走查，系统探讨不同年龄组（青少年、成年人、老年人）在6类应用中的欺骗性模式。

**💡 创新点**

首次将Gray等人提出的低、中、高层级欺骗模式分类与目标年龄组结合，量化比较三年龄组在不同应用类别中遭遇的欺骗模式差异；揭示娱乐类和成人群最易受骗，提供针对性设计指导。

**🔧 技术方法**

采用heuristic‑based cognitive walkthrough方法，并结合Gray et al. 的欺骗模式层级框架；利用Miro进行数据可视化、热图和归一化统计。

**📊 数据集**

30款Android应用（约5款/类别，覆盖购物、游戏、娱乐、音乐/图书、健康与健身、社交媒体），按年龄组标签（青少年12‑17、成年人18‑49、老年人50+）筛选；采集了低/中层级模式出现次数并归一化。

**📈 对比分析**

通过统计每款应用出现的低/中层级模式数量，对高层级模式进行归一化并绘制热图比较不同类别与年龄组的比例；结果显示93%应用使用nagging，娱乐类最高，成人群频繁使用强制行动和社交工程，整体表现出明显的年龄与类别相关性。

**⚠️ 局限性**

限制：仅覆盖Android Google Play应用，年龄组分布不平衡导致成人样本偏高；部分“死端”可能为技术故障误判；每款应用仅走查30分钟，可能漏掉部分模式；缺乏真实用户体验评估，设计者意图判断存在不确定性。

---

## 381. Heaviside Continuity of Rolling Coefficients for Eliminating Epistemic Entropy in Large Language Models

**arXiv ID:** 2607.04562 | [PDF](https://arxiv.org/pdf/2607.04562v1)

**作者:** MY Pitsane `[一作]` (North-West University), Hope Mogale `[通讯]` (University of Pretoria)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于Heaviside门控的验证优先执行框架 HCRC，能够在大型语言模型（LLM）生成代码或指令时实时验证其产生的声明与真实外部状态的一致性，从而抑制模型的幻觉（hallucination）并降低知识熵。

**💡 创新点**

创新点包括：①使用 Heaviside 步进函数（门控）将模型置信度与外部验证得分组合成硬判决，避免了连续概率阈值的模糊性；②构建并行的验证工作池（Validator、Syntax Guard、Test Runner、Citation Guard、Summary Parser），各自独立评估不同可判定谓词；③在保证模型不被修改的前提下通过外部证据动态更新置信度并决定是否推进下一步；④在实际生产环境（代码代理器）中实现了 gate‑授权的文件提交、进度报告和内存压缩。

**🔧 技术方法**

使用技术主要有：①Decoder‑only Transformer LLM（如 Llama‑3.3‑70B、Gemini‑2.5‑Flash、GPT‑OSS‑120B 等）作为 proposer；②并行执行的外部验证工作者；③基于 Heaviside 门的控制循环；④置信度更新公式 C(t) = C + λ V e^β V (1−C)；⑤Rolling Coefficient（RC）分步策略；⑥多任务管道与重试机制。

**📊 数据集**

实验数据集：1) 自研合成软件合成任务集，50 份任务（Flask、FastAPI、CLI、Data‑transform、SQLite）共 5 个任务族；2) 10‑15 份 SWE‑bench Lite 子集用于 smoke‑test；3) 13 位模型（4 大供应商共 13 个模型）对同一任务池进行评测。

**📈 对比分析**

与无门控（原始 LLM）对比，HCRC 将错误完成率（FCR）从 4–7% 降至 0–3%，平均重试次数低于 1–3 次，且在强模型上甚至可比无门控更快；在弱模型上会增加重试成本。实验表明门控对模型能力不敏感，能够在不同模型之间实现 FCR 的统一标准。

**⚠️ 局限性**

局限性包括：①只保证给定谓词集合 𝒫 的正确性，未覆盖隐式属性、性能或安全性等；②可判定谓词的计算开销可能很大；③若模型获知谓词列表，仍可能通过“玩游戏”完成可见谓词而不通过隐藏测试；④门控会引入重试延迟；⑤在极低质量模型或任务复杂度极高时仍可能触发未通过门控的失败。

---

## 382. EEG-SpikeAgent: Agentic Closed-Loop Program Synthesis for Automated EEG Spike Detection

**arXiv ID:** 2607.04558 | [PDF](https://arxiv.org/pdf/2607.04558v1)

**作者:** Sonali Santhosh `[一作]` (Massachusetts Institute of Technology), Danilo Bernardo `[通讯]` (University of California, San Francisco)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于大型语言模型（LLM）的闭环程序合成框架 EEG-SpikeAgent，自动生成可审计的 EEG 特征模块，用于检测头皮 EEG 中的间歇性癫痫放电（IED）。

**💡 创新点**

创新点在于：①将 LLM 作为“代理”在循环中逐步生成单个确定性特征模块；②结合可审计的代码执行与表格化评估，保持模型可解释性；③有意识地在特定迭代加入针对工件（artifact）的特征，提升检测性能。

**🔧 技术方法**

技术手段包括：大型语言模型驱动的程序合成与代码编辑、Python 模块化特征实现、基于 Pydantic 的配置、梯度提升树（XGBoost）作为表格分类器、闭环评估与诊断反馈、特征多尺度窗口化与空间聚合。

**📊 数据集**

使用公开的 VEPISET 数据集，包含 29 通道、4 秒长度的 EEG 片段，2,516 个包含 IED 的 epoch 与 22,933 个无 IED 的 epoch，覆盖多种 IED 亚型。

**📈 对比分析**

在 5 折交叉验证中，采用 XGBoost 评估模型。默认阈值下表现为：AUROC 0.935±0.008，平衡准确率 0.699±0.016，F1 0.557±0.034，灵敏度 0.401±0.032，特异度 0.996±0.001；在 80% 灵敏度下，平均精确率 0.470，平均特异度 0.900。引入工件特征后，平衡准确率提升约 1.4%，F1 提升约 2.3%。

**⚠️ 局限性**

局限性包括：默认阈值下灵敏度低（0.401），仅在固定 4 秒 epoch 上评估，未验证连续记录或不同采集环境下的泛化；未对 IED 亚型、形态或定位进行细分；缺乏对工件特征临床意义的独立验证；模型的阈值校准和持续性能尚未完善。

---

## 383. Lights, Camera, Carbon: Architectural Scaling Laws for Video Generation Energy Consumption

**arXiv ID:** 2607.04553 | [PDF](https://arxiv.org/pdf/2607.04553v1)

**作者:** Nidhal Jegham `[一作]` (University of Rhode Island), Sasha Luccioni `[通讯]` (Sustainable AI Group)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于架构原理和可观测生成参数（分辨率、帧数、步数等）的双向能耗估计框架，可在不获取模型权重或参数大小的情况下预测并反向推断视频生成模型的能耗。

**💡 创新点**

创新点在于：①通过计算机架构分析推导出二次（自注意力）与线性（FFN、卷积）分解的能耗公式；②该框架可正向预测能耗，也可逆向利用推断时间恢复模型的架构复杂度，验证了公式的正确性；③在多种GPU配置和模型规模下实现了低于3% MAPE 的高精度估计。

**🔧 技术方法**

技术手段包括：GPU 能耗实时采集（NVIDIA CUPTI/pyro），多GPU/单卡实验，非负最小二乘回归（NNLS）提取系数，交叉验证评估泛化能力，以及 Monte‑Carlo 传播不确定性用于专有模型的能耗估计。

**📊 数据集**

使用公开的文本提示集（ArtificialAnalysis）生成视频，实验覆盖多种分辨率（720p、1024p）、帧率、时长、批量，评测了 6 个开源 T2V/T2VA 模型（8.3B–27B）以及 3 种 GPU 组合。

**📈 对比分析**

通过与真实能耗对比计算 MAPE 评估方法；结果显示所有模型的预测误差均低于 3%，并在多 GPU 与多分辨率配置下保持稳定；同时将框架应用于 8 个闭源 API，得到不同模型间能耗相差一到两倍。

**⚠️ 局限性**

局限性包括：①未能完整建模 VAE 解码阶段的内存绑定效应；②VAE 与 FFN 成本高度相关，导致系数分辨率受限；③实验参数协方差高，缺乏足够独立配置；④对闭源模型的硬件/功耗假设仍存在不确定性。

---

## 384. VLA Grounder: Language-Conditioning Space Optimization for Black-Box VLA Models

**arXiv ID:** 2607.04517 | [PDF](https://arxiv.org/pdf/2607.04517v1)

**作者:** Damir Shodiev `[一作]` (MIRAI), Aleksandr I. Panov `[通讯]` (AXXX)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对冻结的 Vision‑Language‑Action 模型，作者通过在语言输入层插入可学习的语言条件化策略，并利用 RL 在语言空间优化指令，从而提升机器人执行成功率。

**💡 创新点**

创新点在于将语言视为可优化的条件变量，使用 RL 仅更新语言生成器而非动作权重，解决指令与视觉表征不匹配的问题。

**🔧 技术方法**

主要技术包括语言条件化空间策略（VLA Grounder）、基于失败指令的先验、结构化指令搜索空间以及 GRPO 强化学习。

**📊 数据集**

实验使用 VL-Think 与 RL4VLA 两个公开基准数据集，覆盖符号推理与多物体操纵任务。

**📈 对比分析**

与原始指令、无 RL 预训练以及 TextGrad/GEPA 等基线比较，结果显示在多任务上成功率提升约 30%~70%，尤其在符号定位任务中显著。

**⚠️ 局限性**

局限性包括仅适用于语言不充分的冻结模型，无法修复感知、控制或长程推理缺陷，对复杂视觉噪声或不确定性影响有限。

---

## 385. Mechanism-level routing failure in LLMs over Lean-verified algebraic structures

**arXiv ID:** 2607.04534 | [PDF](https://arxiv.org/pdf/2607.04534v1)

**作者:** Manuel Israel Cázares `[一作]`, Haobo Ma `[通讯]` (ChronoAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大语言模型在Lean形式化代数结构中的结构路由能力，并检测其机制级路由失败。

**💡 创新点**

将router hypothesis从算术推理迁移到形式化验证域，并通过机制级标签精确区分路由失误与真值推理。

**🔧 技术方法**

采用两种提示条件（无线索与Lean证明状态线索）对两大模型进行对比评测，利用模板准确率、推论准确率及误路由模式等指标。

**📊 数据集**

使用FiberRing Set A（22条干净锚点、13种证明机制标签）和交叉模块Set B（6条、19标签），全部来自Automath/Omega Lean 4。

**📈 对比分析**

通过模板准确率对比，gpt‑oss‑120b盲路由为80.3%，Llama‑3.3‑70B为68.2%；给线索后分别提升10.6%与13.6%，并对误路由模式进行分析。

**⚠️ 局限性**

数据集规模小、评测仅在温度0/种子0下进行、误差解释需进一步验证，且未覆盖更广泛模型与任务，导致结果难以泛化。

---

## 386. Dynamic Image-Informed Selection of Biomechanical Tumor Growth Models

**arXiv ID:** 2607.04551 | [PDF](https://arxiv.org/pdf/2607.04551v1)

**作者:** Abdullah Al Noman `[一作]` (University at Buffalo), Danial Faghihi `[通讯]` (University at Buffalo)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于纵向MRI数据的顺序贝叶斯推断与动态模型选择框架，用于对胶质母细胞瘤的生物力学生长模型进行个体化参数校准和模型可行性评估。

**💡 创新点**

创新点在于：①将高维空间可变参数（扩散率、增殖率、弹性模量）与模型选择联合推断；②利用后验模型可行性（模型证据）实时更新最适合的物理假设（无机械、线弹性或非线弹性耦合）；③采用低秩拉普拉斯近似与高维MCMC（gpCN）实现可扩展计算。

**🔧 技术方法**

使用的技术包括：有限元求解、局部拉普拉斯近似、低秩后验协方差构造、通用预条件Crank–Nicolson（gpCN）采样、贝叶斯滤波、模型证据估计以及Dice/NTA等预测质量指标。

**📊 数据集**

数据集为4只Wistar大鼠的纵向磁共振成像（T2、T1加权、扩散加权）记录的肿瘤体积分数，时间点覆盖10–21天，共计5–6个扫描。

**📈 对比分析**

通过对比三种模型的后验可行性、Dice相似系数和归一化肿瘤面积，结果表明机械耦合模型显著优于单纯反应扩散模型；在线弹性与超弹性之间，后者在后期扫描中逐步获得更高可行性，预测性能相近但机械场差异明显。

**⚠️ 局限性**

局限性包括：样本量小、仅采用二维切片分析、模型未包含坏死/凋亡等细胞组分、部分参数（C、H、ν）被固定、缺乏直接位移/应力观测、以及仅在10–21天窗口内验证，难以推广至更长时间或临床人群。

---

## 387. Characterizing the Temporal, Emotional, and Social Patterns of Adolescent Substance Use Discussions on Reddit

**arXiv ID:** 2607.04566 | [PDF](https://arxiv.org/pdf/2607.04566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 388. Towards Digital Preservation of Efik: TTS for a Low-Resource African Language

**arXiv ID:** 2607.04515 | [PDF](https://arxiv.org/pdf/2607.04515v1)

**作者:** Offiong Bassey Edet `[一作]` (University of Cross River State), Mbuotidem Sunday Awak `[通讯]` (ML Collective)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究首次为Efik语言构建并评估端到端文本转语音（TTS）系统，收集并标注了约3小时的单声道语料，使用四种主流神经网络模型（VITS、MMS‑TTS、SpeechT5、Orpheus‑TTS）进行微调并由本地Efik母语使用者进行MOS、Nat‑MOS与A‑MOS评估。

**💡 创新点**

创新点在于：①为极低资源、声调语言Efik提供首个公开可复现的TTS基准数据集；②针对声调信息缺失问题，对模型进行多语种迁移与字符集扩展，并系统比较其在长序列生成与声调准确性上的表现；③以专业评估者的多维度评分为依据，定量揭示声调模型在低资源场景中的优势与局限。

**🔧 技术方法**

采用的技术包括：端到端语音合成框架VITS、跨语言预训练语音合成框架MMS‑TTS、基于Transformer的SpeechT5以及高保真语音生成模型Orpheus‑TTS；对模型进行低资源微调（学习率、批大小、早停等超参优化），并在单GPU混合精度训练下完成。

**📊 数据集**

使用的语料为一位Efik母语使用者录制的2632条语句（约3小时），来源为Efik小说、民俗与教材，经过人工转写、校对与声调校验后拆分为训练（1975条）、验证（264条）与测试（393条）三集。

**📈 对比分析**

评估方法为MOS、Nat‑MOS、A‑MOS三维度人工评分，5位Efik母语评测者对每个模型生成的短句进行打分。结果显示MMS‑TTS最高，MOS 3.80±0.63；Orpheus‑TTS MOS 3.08±0.48；SpeechT5 MOS 2.48±0.49；VITS最低 MOS 1.08±0.27。MMS‑TTS在长序列（可生成近3分钟连续语音）上表现最优，其他模型在20–30秒后易出现无意义或抑扬失衡。

**⚠️ 局限性**

局限性包括：仅有单声道、3小时的单说话人数据，导致声调、韵律多样性不足；罕见音素如ñ在所有模型中均表现不佳；VITS等对大数据依赖强，低资源场景下表现差；整体缺乏多说话人与声调专门化训练策略。

---

## 389. Beyond the Need for Speed: Energy-Aware Code Generation via Simulation-Guided Reinforcement Learning

**arXiv ID:** 2607.04577 | [PDF](https://arxiv.org/pdf/2607.04577v1)

**作者:** Saurabhsingh Rajput `[一作]` (Dalhousie University), Tushar Sharma `[通讯]` (Dalhousie University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练了一种能耗意识的代码生成模型，并发布了Green Tea数据集与基于Sniper+McPAT的确定性架构模拟器，支持大规模能耗评估与闭环强化学习。

**💡 创新点**

创新点在于：①用可重复的模拟器替代硬件测量，构建3.5M条能耗标注；②通过能耗对比对（energy‑contrastive pairs）进行监督微调；③提出CARE‑T度量，将正确性与能耗收益联合评估；④在模拟回报下应用GRPO闭环训练，显著提升能耗优化效果。

**🔧 技术方法**

技术手段包括：确定性架构模拟（Sniper + McPAT），LoRA微调，Group Relative Policy Optimization（GRPO）强化学习，基于能耗与正确性的多目标奖励设计。

**📊 数据集**

使用的数据集是Green Tea，包含1,474个C++竞赛问题的3,507,435条模拟能耗测量，形成12,455对能耗对比对；基准数据来自PIE、CodeNet等公开竞赛集合。

**📈 对比分析**

通过与多种基线（zero‑shot、green‑prompt、runtime‑SFT、AlphaCode、Afterburner等）对比，模型在143个未见问题上实现CARET 12.63%（比仅监督微调提升2.84×），能耗降低平均12.6%，击败人类参考的比例提升至58.4%，同时在大多数模型上超过传统runtime代理。

**⚠️ 局限性**

局限性包括：编译成功率仍不足（仅约80%），对非C++或非CPU架构的可迁移性未知；模型对模拟器参数敏感；在复杂的多线程/异构硬件环境下的能耗预测尚未验证。

---

## 390. A Few Teacher Steps Go a Long Way: Cost-Efficient On-Policy Data Augmentation for Agent Post-Training

**arXiv ID:** 2607.04574 | [PDF](https://arxiv.org/pdf/2607.04574v1)

**作者:** Junze Ye `[一作]` (Stanford University), Mohsen Bayati `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM代理在监督微调中对“on-policy”数据收集的预算分配问题，并提出了统一的数据构造框架与成本衡量。

**💡 创新点**

将监督微调的数据构造视为预算分配问题，分离教师推理成本和训练成本，并通过对情境分布的正式定义，系统比较不同“on-policy”策略。

**🔧 技术方法**

采用基于POMDP的Agentic交互模型，定义情境采样权重，构造“recipe”式数据生成流程，并用KL/交叉熵训练目标。

**📊 数据集**

HotpotQA、ALFWorld和Terminal‑Bench‑Dev三大可验证的代理任务数据集。

**📈 对比分析**

在相同教师推理预算或训练预算下，对纯BC、不同on‑policy长短续、过滤与关键情境过滤等策略进行对比，结果显示有限步教师续在多数任务上更高效、非单调性，并能在相同预算下超过基线。

**⚠️ 局限性**

仅评估了一轮数据增强、有限任务和验证方式；长时程任务、无验证奖励或迭代收集的效果未知，且缺乏理论解释最佳续延长度。

---

## 391. Fidelity-Diversity Metrics for Text

**arXiv ID:** 2607.04563 | [PDF](https://arxiv.org/pdf/2607.04563v1)

**作者:** Amanda Wang `[一作]` (Cornell University), John Thickstun `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了基于最优传输的文本数据集质量评估指标——Fidelity（保真度）与Diversity（多样性），用于量化评估集与参考集之间的相似度与覆盖范围。

**💡 创新点**

创新点在于将 Wasserstein 距离分解为保真度与多样性两项，并通过最近邻与全局传输成本的差值来衡量多样性；同时提出了在混合测度下的显式计算公式，克服了传统单一指标无法区分质量缺陷的问题。

**🔧 技术方法**

技术手段包括：文本句子/问题嵌入（E5_small、Sentence Transformer）、高斯混合模型或经验测度的离散化、最优传输计算（Wasserstein 距离）以及基于 Sinkhorn 的近似多样性评估。

**📊 数据集**

数据集包括 M2D2 S2ORC、M2D2 Wikipedia（通过人工标签的主题划分）以及基于 GSM8K 的多种合成数学问题集合（不同种子比例生成）。

**📈 对比分析**

评估方法：将指标与人工标签的主题覆盖度、以及基于合成数据训练的 Llama-3.2-1B 模型在 GSM8K 测试集上的准确率进行相关性分析。结果显示多样性指标与下游性能高度正相关（Pearson r≈0.97），而保真度指标无显著相关性；同时指标能检测到不同种子比例导致的质量差异。

**⚠️ 局限性**

局限性在于：对离散化（混合测度）依赖较大，若聚类失效会影响指标；对嵌入模型的表达能力敏感；仅在文本/数学问题数据上验证，缺乏跨域广泛性；以及在合成数据中多样性分数极端高时可能出现过度估计。

---

## 392. Can temporal article-level credibility signals improve domain-level credibility prediction?

**arXiv ID:** 2607.04560 | [PDF](https://arxiv.org/pdf/2607.04560v1)

**作者:** Islam Eldifrawi `[一作]` (University Of Sherbrooke), Amine Trabelsi `[通讯]` (University Of Sherbrooke)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的域名可信度评估框架（DCEF），旨在通过分析文章内容来评估新兴网络域名的可信度，而无需先前的域名知识。

**💡 创新点**

创新点在于引入了时间动态的文章级可信度信号，并构建了一个包含超过300个域名和25141篇文章的数据集，以支持这一评估框架。

**🔧 技术方法**

使用了零-shot CoT LLM模块和决策树等技术来进行域名可信度的评估。

**📊 数据集**

构建了一个包含25141篇文章的数据集，涵盖14个领域，解决了现有数据集中链接失效的问题。

**📈 对比分析**

与现有方法相比，DCEF在评估新兴域名的可信度时表现出更好的性能，尤其是在处理未见过的内容时，准确率达到0.92。

**⚠️ 局限性**

限制在于使用的文章数量不均衡，某些域名可能只包含少量文章，未来的工作将探索自动采样方法以提高代表性。

---

## 393. QSVideo: Query-Conditioned Semantic Temporal Retrieval for Video Understanding

**arXiv ID:** 2607.04559 | [PDF](https://arxiv.org/pdf/2607.04559v1)

**作者:** Wei Ao `[一作]` (Michigan State University), Vishnu Naresh Boddeti `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出QSVideo框架，利用查询条件下的语义时序检索显著提升视频理解性能。

**💡 创新点**

创新点包括：①问题分析器将任意问题转化为检索友好查询；②构建三维语义相关度（物体、动作、位置）评估；③联合考虑相关性、多样性与时序覆盖的检索策略，避免相关性偏差、信息冗余和时间崩塌。

**🔧 技术方法**

技术手段包括：使用大型语言模型（Qwen3-4B、Qwen-VL-30B）进行问题分析与标签生成；微调Qwen3-VL-4B-Instruct的语义排名器；基于L2距离的视觉多样性度量；时间窗口划分与自适应遍历的时序检索算法。

**📊 数据集**

数据集：在LLaVA-Video-178K基础上构建35K视频-问题对进行微调；评估使用LVBench（103条长视频）和StreamingBench（499条短视频）两大基准。

**📈 对比分析**

与多种8B–72B VLMs及现有检索方法对比，使用16/32帧显著提升整体准确率：LVBench上准确率提升至49.1%（比基准+7%），StreamingBench上提升至79.36%（比基准+6%），同时延迟和显存占用显著降低。

**⚠️ 局限性**

局限性：依赖外部大模型生成标签，可能带来噪声；对极短视频或实时极限下帧数受限；在多模态不平衡或高噪声视频中的鲁棒性尚待进一步验证。

---

## 394. Predicting Therapeutic Outcome via Aligning Patient-Specific Knowledge Graph and Gene-Level Perturbation Representations

**arXiv ID:** 2607.04557 | [PDF](https://arxiv.org/pdf/2607.04557v1)

**作者:** Dongmin Bang `[一作]` (Seoul National University), Sangseon Lee `[通讯]` (Inha University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用患者转录组构建个体化基因调控网络并结合药物靶点信息，预训练的转录组扰动表示与图神经网络融合，通过对比学习实现临床药物疗效预测。

**💡 创新点**

提出双视角对齐框架，将患者特异性知识图与可迁移的扰动表达进行CLIP式对比学习，实现更精准、可解释的药物响应预测。

**🔧 技术方法**

使用图神经网络（GCN）构建知识图嵌入，预训练的条件基因‑基因注意力模块（CSG^2A）生成扰动表达，CLIP风格对比损失对齐两视角，最终用多层感知机进行分类。

**📊 数据集**

主要使用TCGA肿瘤转录组及其对应药物响应数据，外部验证使用I‑SPY2临床试验（乳腺癌paclitaxel）数据。

**📈 对比分析**

在患者/药物/组织拆分和外部零样本预测中，与XGBoost、DeepCDR、Precily、GeneFormer等基线进行5折交叉验证对比，PREDIKTOR在AUROC、AUPRC等指标上均显著优于基线，提升幅度约5–10%。

**⚠️ 局限性**

对未知药物的泛化仍有限，依赖药物‑靶点先验知识；数据量有限且仅包含转录组，缺乏多组学信息，限制了模型在更广泛场景中的应用。

---

## 395. Auto: The AGI Compiler

**arXiv ID:** 2607.04542 | [PDF](https://arxiv.org/pdf/2607.04542v1)

**作者:** Jaber Jaber `[一作]` (RightNow AI), Osama Jaber `[通讯]` (RightNow AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套“AGI 编译器”，通过记录 LLM 代理的运行轨迹、识别可编译的确定性行为、利用差分回放与 LLM 判断进行验证，最终生成带有能力约束的 WebAssembly 二进制，并在分层运行时通过阈值保护和自我重编译来实现高效、可验证、低成本的代理执行。

**💡 创新点**

创新点在于：① 用“确定性普查”量化可编译行为比例；② 结合差分回放与 LLM 判断的三值验证门；③ 采用 conformal 预测实现可校准的拒绝阈值；④ 物理能力隔离的 WebAssembly 产物；⑤ “螺旋式”闭环（ratchet）实现持续自适应重编译。

**🔧 技术方法**

技术手段包括：轨迹收集 SDK、类型化任务图 IR、枚举+LLM 引导的 CEGIS 合成、决策树/MLP 细化、差分回放对比、LLM 判定器、Conformal 预测阈值、WebAssembly 代码生成、分层运行时（Tier‑0 参考、Tier‑1 编译）、安全沙盒与能力限制。

**📊 数据集**

数据集：六类人工构造任务（票务 triage、三步收件箱 pipeline、字段提取、策略路由、摘要生成、分布式漂移流）共 560 个前沿行为跨度；随后在 300 条任务流中进行线上实验。

**📈 对比分析**

对比方法：将编译后的成本与纯前沿模型成本进行对比，得到 6.4 倍成本降低；使用确定性普查、并行差分回放验证得到 87.1% 的可编译率；在 300 条流上测得 96.9% 的答案一致性，静态窗口平均 2.3 μ$/条，远低于 59 μ$/条的解释成本。

**⚠️ 局限性**

局限性包括：实验数据为人工构造、规模有限；输出类型受限（只能字符串或 DSL 表达）；仅使用词法级别的拒绝门，可能错误接受语义差异；LLM 判定器受模型主观性限制；未覆盖更复杂的多模态或非结构化输出；未验证长期稀疏漂移情形。

---

## 396. Beyond travel mode: urban context shapes active mobility's mental health effects over time

**arXiv ID:** 2607.04520 | [PDF](https://arxiv.org/pdf/2607.04520v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 397. CRISP: A Spatiotemporal Camera-Radar Backbone for Driving via Forecasting-Based World-Model Pretraining

**arXiv ID:** 2607.04541 | [PDF](https://arxiv.org/pdf/2607.04541v1)

**作者:** Jingyu Song `[一作]` (University of Michigan), Katherine A. Skinner `[通讯]` (University of Michigan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出CRISP框架，通过历史摄像头与雷达观测预测未来LiDAR点云，实现LiDAR-free的CR后端可迁移表示学习。

**💡 创新点**

创新点在于将雷达信息嵌入BEV编码器，设计雷达增强时序自注意和模态创新门控的多模态渲染，并利用LiDAR仅作为训练时特权监督的预测驱动预训练。

**🔧 技术方法**

使用技术包括改进BEVFormer、PointPillars雷达编码、雷达增强时序自注意、模态创新门控、Transformer占用预测、LiDAR占用监督和未来点云预测损失。

**📊 数据集**

主要使用nuScenes数据集（包含摄像头、雷达、LiDAR及行驶轨迹）进行训练与评估，LiDAR点云仅在预训练阶段作为监督。

**📈 对比分析**

与CO预训练ViDAR、LC预训练LRS4Fusion、BEVFormer、UniAD和SpaRC-AD等基线对比，CRISP在多任务（检测、跟踪、地图分割、运动预测、占用预测、规划）上均有显著提升，长时序点云预测CD下降至0.91，检测mAP提升至53.2，NDS提升至61.3。

**⚠️ 局限性**

局限包括预测目标缺乏因果交通规则和意图建模、雷达稀疏噪声和校准误差、仅在nuScenes小规模数据上验证、缺乏跨域/大规模验证以及不具备多模态不确定性和语义推理能力。

---

## 398. Measuring Harness-Induced Belief Divergence in Multi-Step LLM Agents

**arXiv ID:** 2607.04528 | [PDF](https://arxiv.org/pdf/2607.04528v1)

**作者:** Haiwen Yi `[一作]` (University of Toronto), Xinyuan Song `[通讯]` (Emory University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探讨并量化软件代理在不同执行框架（harness）下的多步信念漂移，并提出信念滚动诊断与BIWM无训练协议；

**💡 创新点**

①将harness视为实验变量，首次通过跨harness信念差异度量其对代理推理的影响；②提出arrival/growth分解方法揭示接口冲击与长期信念变化；③设计BIWM协议在不训练的前提下统一观测、记录阻断、修复、验证等信息；

**🔧 技术方法**

K步LLM信念滚动、JSON Schema验证、加权信念差异度量（D_cat,D_fail,D_set,D_num,D_act）、BIWM（canonicalisation、blocked-action log、repair‑unroll、verification mask、shadow execution、跨harness对齐）

**📊 数据集**

自制受控编码任务集（8个任务） + SWE‑bench Verified 子集 + Terminal‑Bench 子集（≈300个任务）

**📈 对比分析**

与原始引用harness对比，使用跨harness D_belief、arrival与growth读数；结果显示不同harness在信念轨迹上产生显著差异，尤其是结构化与风险门控；BIWM能显著提升信念可解释性并降低跨harness依赖，整体误差在0.3–0.6范围内；

**⚠️ 局限性**

仅评估信念变化而非实际任务完成率；基于固定LLM与模板，缺乏对不同模型或更大任务的泛化；BIWM仅提供可解释性提升，未必直接提升代理性能；

---

## 399. RAF: Reliability-Aware Fusion of Camera, LiDAR, and 4D RADAR for Robust 3D Object Detection in Adverse Weather

**arXiv ID:** 2607.04587 | [PDF](https://arxiv.org/pdf/2607.04587v1)

**作者:** Heejun Park `[一作]` (KAIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种可靠性感知融合（RAF）框架，利用弱监督的像素级可靠性估计在恶劣天气下提升摄像头与 LiDAR‑RADAR 的三维目标检测性能。

**💡 创新点**

创新点在于通过跨模态相似度学习伪可靠性标签并引入校准感知局部匹配（CALM），实现对天气噪声区域的显式抑制，同时保持对可见区域的保留。

**🔧 技术方法**

使用技术包括 MLP 投影器、余弦相似度计算、CNN 可靠性估计器、视图变换与 BEV 融合、CALM 局部搜索、以及稀疏监督的交叉熵损失。

**📊 数据集**

实验数据集为 K‑Radar（含摄像头、LiDAR、4D RADAR）和 VoD，二者均覆盖多种降雨、雾、雪等恶劣天气。

**📈 对比分析**

通过与 LiDAR‑RADAR 基线以及现有 Camera‑LiDAR‑RADAR 方法（CRN、RobuRCDet、SAMFusion）在 AP_BEV/AP_3D 上进行对比，RAF 在两种 LiDAR‑RADAR 基础网络（L4DR、3D‑LRF）上分别提升约 +6.5 AP_BEV 与 +7.4 AP_3D，显示显著性能提升。

**⚠️ 局限性**

局限在于缺乏真实可靠性标注，仅依赖弱监督；目前仅针对摄像头实现可靠性抑制，未扩展至 LiDAR 或 RADAR；在极端天气或传感器失效时仍可能受限；CALM 对小量标定误差鲁棒但大误差仍可能影响匹配。

---

## 400. A Differentiable Covariance Calculus for Linear Gaussian Bayesian Networks

**arXiv ID:** 2607.04578 | [PDF](https://arxiv.org/pdf/2607.04578v1)

**作者:** Tadashi Wadayama `[一作]` `[通讯]` (Nagoya Institute of Technology), Tadashi Wadayama (Nagoya Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计了一个统一的可微分协方差算子，将线性高斯贝叶斯网络的推理、最大似然估计、隐变量处理、互信息和Fisher信息等全部映射到同一块矩阵运算框架中。

**💡 创新点**

创新点在于把K递归作为单一的计算图后端，所有查询都只需一次自动微分即可得到所有梯度，避免针对不同拓扑手工推导梯度公式。

**🔧 技术方法**

使用技术包括可微分计算图、逆向自动微分、Schur补、Cholesky分解、Slepian–Bangs Fisher公式、以及对状态空间模型和跳连接扩展的块矩阵运算。

**📊 数据集**

数据集使用合成的线性高斯状态空间模型轨迹（2×10^5条）和其跳连接扩展的模拟数据进行验证。

**📈 对比分析**

方法通过与经典Kalman滤波、RTS平滑、d分离判定和经验Cramér–Rao界进行数值对比，误差达到机器精度，梯度一次计算即可，估计误差与理论CRLB一致，收敛速度快。

**⚠️ 局限性**

局限性在于仅适用于高斯线性网络，尚未推广到非高斯或非线性模型；对大规模稀疏网络的数值稳定性与计算效率仍待进一步评估；结构学习等功能尚未实现。

---

## 401. HUGS: Guiding Unified Dexterous Grasp Synthesis Across Modes and Scales via Learned Human Priors

**arXiv ID:** 2607.04554 | [PDF](https://arxiv.org/pdf/2607.04554v1)

**作者:** Mingrui Yu `[一作]` (Tsinghua University), Xiang Li `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于人类先验的统一抓取合成框架 HUGS，能够在不同触摸模式和物体尺度下生成可执行的多模态抓取姿态；

**💡 创新点**

创新点在于将人类抓取偏好学习成对象条件先验，用于引导抓取优化，而非直接复现人类抓取，从而实现可扩展、可多模态、可跨尺度的抓取生成；

**🔧 技术方法**

主要技术包括对象条件生成式人类先验模型（基于 MinkowskiEngine 的稀疏三维卷积网络与扩散模型）以及基于力闭合约束的二层优化方法；

**📊 数据集**

使用了自采集的人类抓取数据集 HUGS‑Human（304 物体、1.8K 抓取），以及通过 DGN2k 生成的 157K 场景（2–30 cm 物体尺寸）来训练和评估；

**📈 对比分析**

与基于尺度规则的手工启发式方法和 HUGS‑Single 等 ablation 进行比较，HUGS 在触摸模式预测、抓取成功率、姿态多样性和生成效率方面均显著优于基线，尤其在跨尺度抓取中提升约 10–15 % 的成功率；

**⚠️ 局限性**

局限性包括：人类数据量有限，导致在极端物体外分布上先验可能失效；生成模型仍不够鲁棒，尤其对几何不规则物体；四种触摸模式覆盖范围有限，未能覆盖全部人类抓取行为；以及在真实部署中仍受标定、感知误差和闭环控制的影响。

---

## 402. Let My Data Go: Data Brokers' Compliance with Opt-Out and Deletion Requests

**arXiv ID:** 2607.04552 | [PDF](https://arxiv.org/pdf/2607.04552v1)

**作者:** Elina van Kempen `[一作]` (University of California Irvine), Mihir Raja `[通讯]` (University of California Irvine)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

对加州注册的数据经纪商，使用合成身份提交删除和退订请求，系统记录并评估其合规性、响应时间及身份验证过程。

**💡 创新点**

首次同时针对删除和退订请求对数百家经纪商进行大规模、系统化的合规性评估，揭示流程不统一、身份验证违规与高消费者负担。

**🔧 技术方法**

采用手工与自动化表单/邮件提交、时间日志记录等技术，未使用机器学习或复杂算法。

**📊 数据集**

基于加州数据经纪商注册目录（约535家）以及两组自创合成身份信息。

**📈 对比分析**

通过与之前针对访问请求的研究进行对比，计算合规率（删除70%未回复，退订22%身份验证违规）和响应时效（删除平均≤3天，退订30%超时），表明整体合规率低且流程繁琐。

**⚠️ 局限性**

使用合成身份无法确认请求是否真正被处理；研究者样本偏年轻技术熟练，无法代表普通消费者；未评估实际数据删除效果，缺乏对合规真实性的验证。

---

## 403. Mask2Real-WM: Segmentation Masks as a Sim-to-Real Bridge for Controllable Dexterous World Models

**arXiv ID:** 2607.04546 | [PDF](https://arxiv.org/pdf/2607.04546v1)

**作者:** Riccardo O. Feingold `[一作]` (ETH Zurich), Robert K. Katzschmann `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一种两阶段的动作条件世界模型，先预测分割掩码的动态，再用渲染模型生成逼真的 RGB 视频，实现了在抓取与放置任务中的长期自回归推理。

**💡 创新点**

创新点在于将像素级预测拆分为掩码动态预测和渲染两步，利用掩码空间作为 sim-to-real 桥梁，实现了在仅 2.5 小时真实数据上进行的动态模型大规模预训练，并显著提升了每自由度的可控性。

**🔧 技术方法**

技术上使用 Stable Video Diffusion（SVD）骨干、ControlNet 进行掩码条件化、LoRA 进行参数微调，并结合动作编码的跨注意力机制；渲染阶段采用了 SVD+ControlNet 的图像生成流水线。

**📊 数据集**

数据集包括 50 小时以上的 IsaacLab 模拟演示（带真实分割标签）以及 2.5 小时的真实手势演示（通过 Rokoko 手套和两摄像机捕获，并用 Segment Anything 生成掩码）。

**📈 对比分析**

通过与单一端到端基线（Ctrl-World）和不同消融版本对比，模型在 ID 与 OOD 场景下的动作可控性得分分别提升至 0.95 与 0.87（相比单一基线为 0.6/0.44），并在 PSNR、SSIM、LPIPS 等感知指标上显著降低 OOD 退化。

**⚠️ 局限性**

主要局限包括对固定相机假设的依赖、缺乏深度信息导致手部遮挡下的物体预测失真、掩码不携带对象身份导致长期序列中身份漂移，以及双阶段扩散推理的计算成本高。

---

## 404. Government AI Use as a Monitoring Primitive: A Public Document Pilot Study

**arXiv ID:** 2607.04543 | [PDF](https://arxiv.org/pdf/2607.04543v1)

**作者:** David I. Atkinson `[一作]` (Northeastern University), Joan Eleanor O'Bryan `[通讯]` (Harvard University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析公共政府文件中的AI文本检测，构建监测原语。

**💡 创新点**

提出使用AI文本检测作为政府AI采用的可公开、可重复的监测工具。

**🔧 技术方法**

使用Pangram商业AI文本检测器对文档进行检测。

**📊 数据集**

采集3000余份来自10个美国与中国政府相关渠道的公共文件。

**📈 对比分析**

将2021年基线与2024-2026年对比，发现美国平均0.05、 中国平均0.07，显示AI写作迹象上升。

**⚠️ 局限性**

检测器易受规避、文档类型差异和语言偏差影响，无法准确识别具体模型。

---

## 405. Finetuning Lightweight LLMs for Control Flow Graph Generation

**arXiv ID:** 2607.04582 | [PDF](https://arxiv.org/pdf/2607.04582v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 406. ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum

**arXiv ID:** 2607.04535 | [PDF](https://arxiv.org/pdf/2607.04535v1)

**作者:** Haiwen Yi `[一作]` (University of Toronto), Xinyuan Song `[通讯]` (Emory University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在固定Stiefel层（保持正交基向量）上加入可学习、受限的对称正定（SPD）谱因子，从而在保持正交约束的同时让网络能够调节奇异值大小。

**💡 创新点**

核心创新在于：① 将权重写成 W = Q S^{1/2}，其中 Q 维持 Stiefel 约束，S 的特征值即为 W 的平方奇异值；② 通过在 SPD 上使用仿射不变重吸收与指数映射实现可保持正定且可控的谱更新；③ 在更新过程中利用被 Stiefel 投影“拒绝”的梯度（pressure）与 EMA、门控相结合，提供结构化的谱学习方向。

**🔧 技术方法**

使用矩阵流形技术：Stiefel 流形、对称正定流形、仿射不变度量、重吸收与指数映射、谱剪裁；结合多种优化器（SGD、Adam、Shampoo、Muon 等）在切线空间更新 Q；对 S 进行矩阵平方根、对数与指数运算；在实验中加入 EMA、门控与随机压力消融。

**📊 数据集**

数据集包括：语言建模任务（WikiText-2、WikiText-103），表格任务（Adult、Covertype、Wine Quality），图像任务（Fashion‑MNIST、CIFAR‑10/100）以及 Mini‑Transformer 在 WikiText‑2 上的实验；还对 ResNet‑18/50 的卷积分类头进行了测试。

**📈 对比分析**

采用“配对”比较方式：在相同网络结构、相同优化器、相同数据拆分下，将固定谱 Stiefel 层（FS）与谱可学习的 MF 层进行对比；评价指标为语言模型的困惑度（PPL，数值越低越好）和分类任务的准确率（%，数值越高越好）。实验显示：在循环语言模型投影和 MLP 隐藏层中，MF 通常能带来显著提升（尤其在使用 SGD 时差距更大）；在卷积分类头和某些表格/图像任务中，提升有限甚至为负，表明 Stiefel 先验并非始终合适。

**⚠️ 局限性**

局限性：① 需要对 SPD 因子做矩阵平方根、指数、对数和特征分解，导致大尺寸权重时计算开销较大；② 仅在 Stiefel 先验合适时才有效，无法替代无约束或对角谱层；③ 目前未在完整的 Transformer 规模、密集层基准或更复杂的图网络中全面评估；④ 未给出详细的时间/资源开销分析；⑤ 对高维权重的扩展需考虑结构化或低秩近似。

---

## 407. Failures and Successes to Learn a Core Conceptual Distinction from the Statistics of Language

**arXiv ID:** 2607.04523 | [PDF](https://arxiv.org/pdf/2607.04523v1)

**作者:** Zhimin Hu `[一作]` (University of Wisconsin-Madison), Gary Lupyan `[通讯]` (University of Wisconsin-Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了人类如何从语言中学习原则性与统计性属性的区别，探讨语言经验是否能够促进核心概念区分的学习。

**💡 创新点**

提出语言模型能够从语言统计中学习到原则性与统计性属性的区别，挑战了这一区分被认为无法学习的观点。

**🔧 技术方法**

使用了多种语言模型，包括BERT、ALBERT、DistilBERT、RoBERTa、GPT-3.5和GPT-4，进行对比分析。

**📊 数据集**

构建了一个包含208个通用语句的语料库，并通过人类参与者对这些语句进行评分。

**📈 对比分析**

与人类评分结果进行比较，发现GPT-4在区分原则性与统计性属性方面表现优于其他模型，尤其在控制了频率后，GPT-4的预测能力显著提高。

**⚠️ 局限性**

模型在区分统计性与原则性属性时，仍然受到频率的影响，且并非所有模型都能有效学习这一区分，表明当前模型的局限性。

---

## 408. The User-In-Context Framework: Understanding Variation in How Users Respond to AI Chatbots

**arXiv ID:** 2607.04547 | [PDF](https://arxiv.org/pdf/2607.04547v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 409. A non-invasive video-based method for individual identification of wildlife using gait dynamics

**arXiv ID:** 2607.04518 | [PDF](https://arxiv.org/pdf/2607.04518v1)

**作者:** Muhammad Aamir `[一作]` (University of Oxford), Andrew Markham `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并验证了一套基于SAM3前景分割、ResNet18空间特征与VideoPrism时序建模的自动化视频管线，用于野生动物的无接触步态识别。

**💡 创新点**

创新点在于将零样本前景分割与时空深度特征融合，首次在跨物种、跨环境的野生动物视频中实现高一致性与明显分离的步态嵌入。

**🔧 技术方法**

使用的技术包括SAM3前景分割、ResNet18提取空间特征、VideoPrism提取时序特征、余弦相似度比较、无监督聚类（k‑means/层次聚类）。

**📊 数据集**

使用了从公开视频网站与Longleat Safari Park收集的5种野生动物（骆驼、狮子、长颈鹿、斑马、鬣狗）共185段视频的数据集。

**📈 对比分析**

通过余弦相似度矩阵、silhouette系数等指标进行评估，平均intra‑animal相似度约0.96，inter‑animal相似度低于0.80，silhouette平均值0.78，表明聚类效果良好。

**⚠️ 局限性**

局限性包括受侧向步态和相同视角的限制；视角变化、速度、光照等因素仍能影响嵌入性能，且对正面或背面步态的鲁棒性不足。

---

## 410. LLM-Driven CI-CD Workflow Intelligence for Cyber Systems Engineering

**arXiv ID:** 2607.04579 | [PDF](https://arxiv.org/pdf/2607.04579v1)

**作者:** Bonan Shen `[一作]` (Independent Researcher), Xin Liu `[通讯]` (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

基于大语言模型的CI/CD分析流水线，对GitHub公开仓库进行反模式检测、阶段挖掘和推荐生成，形成完整的运营治理工具；

**💡 创新点**

将工作流程视为可分析基础设施，整合元数据丰富、反模式检测、阶段结构识别与多策略推荐生成，超越单一阶段分类；

**🔧 技术方法**

采用LLM（如GPT系列）进行结构化输出、提示工程（zero-shot、few-shot、检索增强、迭代）以及YAML解析器校验；

**📊 数据集**

采集约59550个星级≥1000的GitHub仓库，筛选出34225个含CI/CD的仓库，共127559份工作流文件，覆盖多种CI/CD平台和七类项目域；

**📈 对比分析**

在RQ1中检测到434769条反模式（平均≈5.8/文件），可靠性问题最突出；RQ2发现不同语言/域阶段分布显著差异（χ²=4168.88，Cramér V=0.063）；RQ3比较四种提示策略，few-shot在覆盖率、有效性（YAML有效率96.1%）和平均建议数（8.25）上最佳；

**⚠️ 局限性**

数据偏向高星GitHub仓库，域标签覆盖不足（仅32513个），评估依赖模型自动化验证缺乏人工专业评估，且仅使用单一LLM配置，未检验跨模型一致性；

---

## 411. Detecting Answer-Driven Reasoning in LLM-Based Educational Tutors via Truncated Chain-of-Thought Auditing

**arXiv ID:** 2607.04572 | [PDF](https://arxiv.org/pdf/2607.04572v1)

**作者:** Bonan Shen `[一作]` (Independent Researcher), Tao Ning `[通讯]` (Syracuse University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在数学教育情境下，研究者提出并验证了一种基于截断推理的审核方法，用于检测大型语言模型教师解释是否过早利用答案信息；

**💡 创新点**

该方法通过在不同上下文（仅问题、答案键、错误答案键）下生成推理过程，并在预设的前缀比例处强制给出答案，衡量答案可获得的时间窗口，从而揭示答案驱动的生成现象；

**🔧 技术方法**

采用了 Qwen2.5‑3B‑Instruct 语言模型，配合 TRACE 评估框架和整数答案验证器；

**📊 数据集**

在 GSM8K 测试集上随机抽取 1000 道小学数学题进行实验；

**📈 对比分析**

相较于仅问题情境，答案键情境的 TRACE AUC 从 0.375 提升至 0.900，10% 前缀通过率从 0.113 提升至 0.997，证明答案键显著提前让答案可被检验；

**⚠️ 局限性**

局限性包括仅使用单一 3B 模型、仅评估数值答案的简短题目、强制答案仅在一次生成时检验、未覆盖开放式问答、多轮对话或非数值验证场景；

---

## 412. SceneFrom3D: Geometry-Conditioned Outdoor 3D Scene Generation via View Scheduling with Object-Level Control

**arXiv ID:** 2607.04540 | [PDF](https://arxiv.org/pdf/2607.04540v1)

**作者:** Geonung Kim `[一作]` (Pohang University of Science and Technology), Sunghyun Cho `[通讯]` (Pohang University of Science and Technology)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

提出了一种基于几何条件的室外3D场景生成框架，自动生成摄像机视角并支持对象级外观与几何遵从控制。

**💡 创新点**

创新点包括：1）无需手工摄像机轨迹的自动视角调度生成图；2）对象级身份图与几何遵从参数实现细粒度控制；3）将anchor视图生成与视频插值相结合的三阶段生成管线。

**🔧 技术方法**

采用可见度测量与优化的视角调度算法；使用 FLUX.2-klein-9B 扩散模型进行anchor视图生成；利用 VACE 视频插值生成中间视图；使用3D Gaussian Splatting (3DGS) 进行场景重建与优化。

**📊 数据集**

使用自制的9种室外场景布局数据集，每个场景包含9-16个对象，基于人工或免费资产生成。

**📈 对比分析**

通过与UrbanArchitect、YoNoSplat以及无人机路径规划等基线方法比较，利用CLIP Aesthetic、MUSIQ、PSNR-D、Chamfer距离、F-score等指标，实验表明在视觉质量、几何精度和结构完整性上均优于基线。

**⚠️ 局限性**

局限性包括：anchor视图生成在单视图超过8个不同对象时易失败；缺乏全局光照先验导致阴影方向和大小不一致；在复杂几何或大规模场景中可能出现摄像机冗余或覆盖不足。

---

## 413. Obey, Diverge, Collapse: Blind Obedience to Incorrect Instructions Drives Code LLMs to Irrecoverable Code Semantic Collapse

**arXiv ID:** 2607.04537 | [PDF](https://arxiv.org/pdf/2607.04537v1)

**作者:** Raj Jaiswal `[一作]` (Indian Institute of Information Technology), Rajiv Ratn Shah `[通讯]` (Indian Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统评估了代码语言模型在接受错误指令时的行为，发现它们会盲目遵从错误指令（Blind Obedience），导致额外的 Ghost 错误并产生不可恢复的语义损伤；通过四个逐步加深的实验（从单次修复到多轮迭代修复）验证这一现象；

**💡 创新点**

创新点包括首次定义并量化 Blind Obedience 与 Ghost 错误，揭示模型在正确识别错误指令后仍会执行的现象；提出不可恢复损伤率（Irrecoverable Damage Rate）作为新的评估指标，证明仅靠通行的通过率评估无法捕捉此类结构性破坏；

**🔧 技术方法**

技术上采用了多种主流代码 LLM（GPT‑5.3 Codex、Claude Sonnet 4.6、Qwen3‑Coder、GLM‑5、Kimi K2.5）在默认配置下进行零/低推理；构建了自思考式迭代修复管线，并利用执行测试用例提供即时反馈；通过统计指令遵从率、通过率、Ghost 错误累计以及不可恢复损伤率进行量化分析；

**📊 数据集**

使用 RunBugRun 基准数据集，包含 538 个 Python 算法问题，每个问题配有错误实现、正确实现、问题描述和可执行的判定测试用例；

**📈 对比分析**

对比方法：在三种任务设置（正确指令、错误指令、自我思考）下测量通过率；在迭代修复阶段记录每轮通过率与错误累计；实验结果显示：错误指令导致通过率骤降，迭代修复在两轮后达到上限，Ghost 错误随轮数累计，且在迭代后不可恢复损伤率显著高于基准；提升推理层级并未提升修复效果，反而导致输出失败；

**⚠️ 局限性**

局限性：实验仅限于单函数算法问题，缺乏多文件、外部依赖和架构约束；数据集规模相对有限；测试用例均为确定性，真实工程环境的噪声和多样性未得到覆盖；因此结果可能在更复杂的生产代码库中表现不同。

---

## 414. Training-Free Model Selection and Domain-Aware Score Calibration for First-Shot Anomalous Sound Detection

**arXiv ID:** 2607.04526 | [PDF](https://arxiv.org/pdf/2607.04526v1)

**作者:** Grach Mkrtchian `[一作]` `[通讯]`, Grach Mkrtchian

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种训练无关的后置量化校准与标签无关的模型选择方法，用于解决第一射程异常声检测中的源/目标域平衡和开发集与评估集不一致的问题。

**💡 创新点**

创新点在于：①引入可调缩放的域感知分位数校准层（参数 m）以控制源/目标的均衡；②设计基于交叉验证的 KS 距离度量的无标签域平衡准则，并配合基于开发集的可行性否决机制，实现不依赖目标异常标签的配置选择。

**🔧 技术方法**

采用了冻结音频嵌入提取器（BEATs、EAT、PANNs）+ kNN 余弦距离评分；域感知软/硬分配；分位数映射与部分池化（m 控制）；KS 距离评估；交叉验证与家庭块自举检验；可行性否决。

**📊 数据集**

使用 DCASE 2023/2024/2025/2026 Task‑2 公开数据集（ToyADMOS2、MIMII 语音片段），每台机器 990 源域和 10 目标域正类样本。

**📈 对比分析**

与官方基线（Selective‑Mahalanobis、Simple AE）和 35 队伍的排行榜进行比较，使用 Ω（AUC 的谐波均值）评估。2025 年在 45 配置网格中，使用标签无关准则的选择将 Ω 从 55.83 提升至 61.05（排名第 4）；在 2023/2024 年，固定全域均衡（m=0）往往优于或等同于准则选择，说明准则在 2025 年具有突出优势。

**⚠️ 局限性**

局限性包括：①准则的预测力仅在 2025 年显著，2023/2024 年无显著提升；②在小目标样本（10）下分位数映射支持有限，导致外推和阈值不保守；③存在“均衡但无检测”的退化配置，需通过可行性否决过滤；④方法依赖冻结嵌入，若嵌入不匹配可能失效；⑤未在更大规模或不同任务上验证。

---

## 415. Language Models Represent and Transform Concepts with Shared Geometry

**arXiv ID:** 2607.04525 | [PDF](https://arxiv.org/pdf/2607.04525v1)

**作者:** Zhimin Hu `[一作]` (Georgia Institute Of Technology), Sashank Varma `[通讯]` (Georgia Institute Of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型（LLM）中概念在不同上下文下的表示与变换，将概念视为点云流形，上下文视为向量场，并在多种规模模型中检验其几何一致性与语义结构。

**💡 创新点**

创新点在于：①提出把概念表示为非参数点云流形而非静态向量；②将上下文变换建模为向量场，发现其方差具有语义组织；③发现不同模型在概念位置及变换几何上共享结构，且可跨模型迁移预测；④揭示概念在保持相对关系的同时具备灵活性。

**🔧 技术方法**

使用中心化核对齐（CKA）与Grassmann距离评估跨模型几何一致性；通过无偏HSIC估计的CKA衡量内部/跨概念关系；利用SVD提取向量场主子空间并计算Grassmann距离；实现关系运输（relational transport）预测跨模型变换。

**📊 数据集**

数据集包括：WordNet（人工/自然概念）、Brysbaert的语义具体性评分、Google News Word2Vec词向量（用于密度评估）、Wikipedia句子片段（构造上下文）、以及23个公开基准LLM（0.5B–32B）来自六大模型族（Qwen、Llama、Mistral、Gemma、DeepSeek、Pythia）。

**📈 对比分析**

与随机置换、随机高斯基线相比，关系运输在多层次、多模型上显著提升：Spearman 相关≈0.5–0.8，余弦相似度显著>0；CKA与Grassmann距离与模型能力（MMLU）相关性高于模型规模，说明几何一致性跟功能表现更密切。

**⚠️ 局限性**

局限性包括：仅在自回归LLM（英语）实验；未验证视觉或多语言模型的相同几何；聚焦中间层而非最终层；并且使用非参数方法，缺乏解析闭式模型描述。

---

## 416. Wireless Gas Leak Detection and Localization

**arXiv ID:** 2607.04524 | [PDF](https://arxiv.org/pdf/2607.04524v1)

**作者:** Fabien Chraim `[一作]` (University of California), Kris Pister `[通讯]` (University of California)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了基于无线分布式传感器网络的工业气体泄漏检测与定位系统，使用20个电池供电的丙烷传感节点，结合概率检测与质心定位算法，在实验中实现了91%检测率、平均延迟108s、5米定位精度。

**💡 创新点**

将WirelessHART Mesh网络与离散化的概率检测/自相关分段技术相结合，采用两阶段检测与基于浓度中心的定位，系统兼容低功耗无线标准且能在现场快速部署。

**🔧 技术方法**

无线Mesh（WirelessHART）传感网络、ARM Cortex-M3 SoC、红外丙烷传感器、半经验概率模型、自相关相似度分段检测、中心质量定位、MATLAB/实验平台。

**📊 数据集**

基于TXA&M实验室的60次人为丙烷泄漏（不同源高度、喷嘴尺寸、流量）产生的浓度时间序列数据，采样频率5s，20个传感器4×5网格覆盖200m²。

**📈 对比分析**

通过系统性评估不同窗口尺寸和百分位阈值组合，理想化噪声模型下实现100%检测率、0误报；实际硬件实验得到55/60检测率、7误报、平均延迟108s，定位误差≤5m。

**⚠️ 局限性**

传感器噪声大、响应慢、功耗高导致续航短；误报率仍高；定位受传感器布置不对称影响；缺乏针对多源或远场泄漏的空间/时间相关模型；需进一步提升传感器SNR与能效。

---

## 417. MTEB-PT: A Text Embedding Benchmark for Brazilian Portuguese

**arXiv ID:** 2607.04581 | [PDF](https://arxiv.org/pdf/2607.04581v1)

**作者:** Tardelli Ronan Coelho Stekel `[一作]` `[通讯]` (Federal Institute of São Paulo), Tardelli Ronan Coelho Stekel (Federal Institute of São Paulo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MTEB-PT，一套22项本土巴西葡萄牙语文本嵌入基准，并公开发布所有任务、代码与leaderboard；

**💡 创新点**

创新点在于：①完全采用原生葡萄牙语数据，剔除机器翻译任务；②引入四层统计严谨度（bootstrap CI、配对显著性、IRT判别、Borda计数），让排名具置信区间与统计可解释性；③对93种模型（23 M–27 B参数，73开源+20商用API）进行系统评测，揭示前六名无显著差异且顶级模型无需商用API即可达成；

**🔧 技术方法**

使用MTEB评估框架的分类、STS、聚类、检索、重排等七大任务类别；统计分析采用分层bootstrap、配对bootstrap、两参数IRT、Borda排序；

**📊 数据集**

采用17个葡萄牙语原始数据集，涵盖社交媒体、法律、税务、医学、科学、百科与技术领域；包括自建WikiCatClusP2P、MedPTRetrieval/Clustering、SciELO/Stackoverflow/法律聚类、FaqBacen/FaQuADIR检索、Quati/JurisTCU重排等；

**📈 对比分析**

对93模型的22项平均得分进行排序；结果显示22项平均得分从0.248到0.682，约78.7 %模型对被清晰区分，约12个可区分层级；前六名在统计上无显著差异，开放权重模型Qwen3-Embedding-8B位于同一顶尖层级；同时对商用API与开源模型在成本与质量维度绘制Pareto前沿，表明无成本的自托管模型即可匹配商用API；

**⚠️ 局限性**

局限性包括：①排除机器翻译任务导致检索语料多样性受限；②所有任务统一截断512 token，限制长文档检索评估；③检索语料库规模偏小，nDCG@10可能不具普遍代表性；④任务集中于机构/法律文本，缺乏对话与产品评论等口语化域；⑤评测仅在单一时间窗口完成，API版本变动可能影响结果；⑥部分任务可能存在预训练泄露；

---

## 418. Rainbow Beamforming for Wideband LEO Satellite Communications: Principles, Applications, and Technical Challenges

**arXiv ID:** 2607.04570 | [PDF](https://arxiv.org/pdf/2607.04570v1)

**作者:** Juha Park `[一作]` (Korea University), Wonjae Shin `[通讯]` (Korea University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出并系统性地阐述了彩虹波束成形（Rainbow Beamforming）在低轨卫星通信中的应用，重点通过理论分析与仿真展示其在大规模多址、集成感知通信以及快速卫星捕获等场景下的优势。

**💡 创新点**

创新点在于将传统视为干扰的波束偏斜（beam‑squint）转化为频率-空间多样性资源，实现单或少数射频链即可生成多频谱、不同指向的波束，从而显著提升频谱利用率与系统灵活性。

**🔧 技术方法**

主要技术包括联合相位时延阵列（JPTA）、频率-方向映射设计、时域与频域资源协同调度、机器学习辅助优化等，配合传统相位调制器（PS）和真正时延器（TTD）实现频率依赖波束控制。

**📊 数据集**

数据集：本文采用基于 14 GHz 中心频率、10 % 带宽、64/64/8×8 天线阵列的仿真环境，模拟卫星高度 500 km、用户分布与流量模型等场景；并通过 IEEEtran.cls 示例展示实验结果。

**📈 对比分析**

通过与传统波束跳跃（beam‑hopping）以及多时隙射频链方案对比，仿真表明彩虹波束成形在单时隙内即可覆盖整块覆盖区域，吞吐量提升约 2–3 倍，延迟下降 70% 以上，且对天线数量与带宽扩展具有良好可扩展性。

**⚠️ 局限性**

局限性包括：高成本 TTD 设备与功耗、频率-方向映射与 JPTA 参数的联合非凸优化难题、对多卫星协同与相互干扰的处理不足、以及对高速多普勒效应下的定位与感知精度的鲁棒性缺乏深入实验验证。

---

## 419. An Exact Generalized k-Cell Decomposition

**arXiv ID:** 2607.04561 | [PDF](https://arxiv.org/pdf/2607.04561v1)

**作者:** Yeganeh Bahoo `[一作]` (Toronto Metropolitan University), Roni Sherman `[通讯]` (Toronto Metropolitan University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种新的单机器人 k‑模组器可视化规划的细胞分解方法，能够精确捕捉机器人视野阴影结构的变化事件（出现、合并、消失、分裂）并构建完整的可视化事件图谱。

**💡 创新点**

创新点在于：①仅当两个顶点互为关键点时才可能出现可视化事件，极大减少了无效分割线；②系统地归纳了12种顶点组合的事件条件，实现了对所有事件的完全覆盖；③将分解复杂度从原先的 O(k²n⁴) 降低到 O(n⁴)，并且不再依赖 k 的大小。

**🔧 技术方法**

主要技术包括：几何对称性分析、关键顶点定义、事件触发判定、基于多边形投影的细胞分解以及对 k‑可见性区域的组合计算。

**📊 数据集**

本研究未使用外部数据集，而是基于理论几何构造与实验验证。

**📈 对比分析**

与已有的 0‑可见性、2‑可见性以及通用 k‑可见性分解方法相比，本方法在相同多边形下显著减少了分割线数量，因而在查询阴影结构时的时间复杂度更低，实验结果表明在大多数案例中节省了至少 30% 的计算时间。

**⚠️ 局限性**

局限性包括：①仍属于离散几何方法，面对极大规模或高度折叠多边形时仍会出现 O(n⁴) 的计算瓶颈；②仅适用于平面多边形，无法直接推广到三维空间或动态环境；③在有孔多边形的特殊配置下，需要额外处理，可能导致进一步的复杂度提升。

---

## 420. Explainable Novel Category Discovery in Semantic Concept Space

**arXiv ID:** 2607.04548 | [PDF](https://arxiv.org/pdf/2607.04548v1)

**作者:** Ifrat Ikhtear Uddin `[一作]` (University of South Dakota), Longwei Wang `[通讯]` (University of South Dakota)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Explainable Novel Category Discovery (xNCD) 框架，利用可解释的语义概念空间进行未标记数据的新类别发现，并在概念空间内给出聚类和实例级解释。

**💡 创新点**

核心创新点包括：① 在概念瓶颈（Concept Bottleneck）中完成特征学习与聚类，直接在可解释的概念空间工作；② 采用无标签的概念投影学习，通过 CLIP 视觉‑语言相似性作为概念监督；③ 将分类和伪标签生成统一到概念空间的交叉熵目标；④ 通过理论分析证明概念瓶颈严格限制假设空间，使得聚类结果具备语义可解释性。

**🔧 技术方法**

技术细节包括：ResNet 编码器、线性概念投影层、CLIP 预训练多模态模型对齐、cubed cosine 对齐损失、概念归一化、多个聚类头、对齐后的多视图自标记、Sinkhorn‑Knopp 伪标签、统一 softmax 换位交叉熵等。

**📊 数据集**

使用的数据集为 CIFAR‑10、CIFAR‑100 以及 CUB‑200，按标准 NCD 划分（CIFAR‑10: 5+5, CIFAR‑100: 80+20, CUB‑200: 170+30）并在任务无关（task‑agnostic）评估下进行实验。

**📈 对比分析**

与 KCL、MCL、DTC、RS+、UNO、SNCD、GCD 等方法在任务无关评估下对比，xNCD 在 CIFAR‑10 获得 92.63% 总体准确率（CIFAR‑10），CIFAR‑100 76.45%，CUB‑200 65.59%。相对于 UNO、GCD 等，xNCD 在 CIFAR‑100 上有 3+ 分的提升，且是唯一能提供人类可读聚类和实例解释的方法。

**⚠️ 局限性**

限制：① 需要较大的概念词汇量，词汇量下降会导致聚类性能显著降低；② 对细粒度数据（如 CUB‑200）的新类别精度仍相对较低，表明现有概念集合难以捕捉细微差异；③ 与最先进的非可解释方法相比，xNCD 在某些任务上仍存在轻微性能差距；④ 对 CLIP 预训练域的依赖可能导致跨域迁移时性能波动。

---

## 421. Lyapunov-Guided Training for Hardware-Safe Neural Networks Under Fixed-Point Arithmetic

**arXiv ID:** 2607.04531 | [PDF](https://arxiv.org/pdf/2607.04531v1)

**作者:** Anis Hamadouche `[一作]` (Heriot-Watt University), Amir Hussain `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Lyapunov能量投影的低精度神经网络训练与推理框架，直接在两补数包装算术下限制隐藏状态能量的上升，避免溢出导致的符号与幅度突变；

**💡 创新点**

创新点在于将Lyapunov能量投影作为硬件安全约束，既能保证隐藏状态能量有界又能保持非递增，显著提升低精度量化模型在包装算术环境下的稳定性与准确率；

**🔧 技术方法**

采用了低精度固定点量化、两补数包装、层级Lyapunov能量定义、单调投影约束、量化感知训练（QAT）与后训练量化（PTQ）等技术；

**📊 数据集**

使用MNIST数据集进行实验；

**📈 对比分析**

与无投影的PTQ、QAT及FP32 baseline进行对比，结果显示投影能将溢出率降至约0.01%，并将12位QAT的准确率从≈10%提升至≈86.5%；

**⚠️ 局限性**

局限在于实验仅在MNIST小型数据集与简单Transformer架构上验证，阈值选择需手工设定，未涵盖更大模型或其他硬件特性（如混合精度、稀疏性）等情况。

---

## 422. Evaluation and Explainability of Unsupervised Scholarly Collaboration Recommendations

**arXiv ID:** 2607.04529 | [PDF](https://arxiv.org/pdf/2607.04529v1)

**作者:** Md Asaduzzaman Noor `[一作]` (Montana State University), Jason A. Clark `[通讯]` (Montana State University)

**通讯引用:** 482 | [OpenAlex ID](https://openalex.org/A5080044971)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估无监督基于文本的学术合作推荐方法的表现与可解释性，并在部分文本重叠下进行对比

**💡 创新点**

引入部分holdout实验减少共同出版物重叠，比较TF-IDF、主题模型与嵌入检索三类方法在此条件下的稳健性；同时提供主题和检索+LLM两种解释框架

**🔧 技术方法**

TF-IDF、LDA、BERTopic（含克隆变体）、SciBERT+Faiss检索、社区检测、LLM生成解释

**📊 数据集**

约1,634位研究者、45,000篇论文的跨三所大学（Uni1、Uni2、Uni3）数据集，包含论文标题摘要和OpenAlex共著信息

**📈 对比分析**

通过Hits@10和MRR在全信息和50/50holdout两情景下比较；TF-IDF在全信息最高，但在holdout显著下降；Clone-LDA和Faiss保持相对稳定，成为最佳鲁棒方案

**⚠️ 局限性**

评估仅基于历史共著作为真值，无法覆盖潜在但未合作的研究者；解释依赖检索与LLM的质量；缺乏大规模实验与跨学科更广泛验证

---

## 423. AnyStyle: A Single LoRA is Sufficient for Image-Guided Style Transfer

**arXiv ID:** 2607.04677 | [PDF](https://arxiv.org/pdf/2607.04677v1)

**作者:** Yongwen Lai `[一作]` (South China Normal University), Chaoqun Wang `[通讯]` (South China Normal University)

**通讯引用:** 21569 | [OpenAlex ID](https://openalex.org/A5100441502)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 AnyStyle 的单 LoRA + 关注调制框架，用于图像引导的风格迁移，兼顾内容结构与风格一致。

**💡 创新点**

创新点包括：① 仅使用单一 LoRA 对风格进行特定层级微调，消除双 LoRA 产生的复杂耦合；② 通过训练无关的注意力提取内容结构，在推理阶段实现精准的内容保持；③ 只调制 query 张量并采用时间依赖 β 控制，既保证结构保真，又保持风格完整；④ 在 FLUX（DiT + rectified flow）架构上实现。

**🔧 技术方法**

技术手段：LoRA 微调、FLUX/DiT diffusion 模型、rectified flow、CLIP+T5 文本编码、注意力调制、时间动态 β、用户研究与多指标评估。

**📊 数据集**

使用 40 张图像（20 张内容、20 张风格）来自 B-LoRA、StyleDrop、EditEval 数据集，形成 400 条风格迁移组合进行训练与评测。

**📈 对比分析**

对比方法包括 Dual‑LoRA、B‑LoRA、UnZipLoRA、StyleID、StyleDiffusion 等；在 CLIP‑T、CLIP‑Style、DINO‑Style、PSNR、LPIPS、DS 等指标上均优于基线；用户研究显示 AnyStyle 获得最高选拔率。

**⚠️ 局限性**

局限性：对极其复杂或未见过的语义结构（如卡通、稀有手绘场景）识别效果不足；依赖 FLUX 预训练模型，可能在其他架构上的迁移性待验证；注意力调制参数（β_min、β_max）需手动调优以适配不同任务。

---

## 424. Elastic Gang: Per-Token Membership Change for a Hard-Barriered LLM Inference Gang Co-Scheduled with OS Processes

**arXiv ID:** 2607.04668 | [PDF](https://arxiv.org/pdf/2607.04668v1)

**作者:** Daeyeon Son `[一作]` `[通讯]`, Daeyeon Son

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在裸机 Rust 内核中实现了可弹性伸缩的 LLM 推理 gang，允许核心在每个 token 期间动态加入或离开，保持工作保守性；同时通过 ACK‑latch 纪元协议与单词级成员快照保证硬 barrier 计算在核心变化时仍保持位精确输出。

**💡 创新点**

首次提出 ACK‑latch 纪元协议和 per‑token 参与者快照，解决硬 barrier 计算在核心变更时的死锁与日志破坏问题，并实现了工作保守、无阻塞的核心借贷与迁移机制。

**🔧 技术方法**

使用 Rust kernel、AVX‑512 量化核、seqlock/RCU 纪元标签、原子单词快照、MWAIT/monitor、owner CAS、工作抢占（work‑stealing）、基于信任的配额调度等技术。

**📊 数据集**

在两种 LLM 模型上验证：SmolLM2‑135M（Q4_0，87.6 MB）和 Qwen2.5‑7B‑Instruct（Q4_0，4.238 GB），在 AMD Ryzen 9800X3D (Zen 5) 机器上进行实验。

**📈 对比分析**

与静态核心分区基线（8 核心/12 核心）进行占用率（0 %、25 %、50 %、75 %、100 %）对比。结果显示在 25‑75 % duty 下，elastic gang 在保持或提升推理吞吐的同时，通用吞吐分别提升 1.75×、1.52×、1.28×；在 0 % 时恢复所有闲置核心；在 100 % 时与静态分区收敛。

**⚠️ 局限性**

实验仅在单一 Zen 5 主机上完成；SMT 兄弟核心的借贷效果不完全独立；核心获取成本受调度量子限制，导致在高占用时租借延迟；不可迁移进程的性能可能受影响；需要预先测量饱和点（knee）才能有效使用。

---

## 425. Hierarchical Evidence-Driven Reasoning for Long Document Understanding

**arXiv ID:** 2607.04625 | [PDF](https://arxiv.org/pdf/2607.04625v1)

**作者:** Junyu Xiong `[一作]` (University of Science and Technology of China), Houqiang Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27219 | [OpenAlex ID](https://openalex.org/A5078141810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种四阶段的多模态检索增强生成框架，用层次化问题拆解、粗粒度视觉检索、基于交叉页面推理的精细验证和基于记忆的迭代生成四步解决长文档问答任务。

**💡 创新点**

创新点包括：①将多跳问题拆分为原子子问题以提升检索召回率；②利用GRPO训练的跨页面验证器精准过滤主题相似但无答案的噪声页；③通过显式文本记忆进行迭代推理，消除单次检索遗漏导致的级联错误。

**🔧 技术方法**

核心技术：多模态检索器（如ColPali/ops-ColQwen3）、EviGRPO验证模型、视觉语言模型（Qwen3.5-27B等）、记忆驱动的多轮生成策略。

**📊 数据集**

使用四个公开基准：PaperTab、FetaTab、MMLongBench、LongDocURL，涵盖表格与长文档的多模态问答。

**📈 对比分析**

与M3DocRAG、MDocAgent、MoLoRAG+、ALDEN、URaG、Doc-V^⋆等基线进行对比，平均提升约8.05%的准确率；在每个基准上均位列第一，Ablation实验验证每个模块贡献显著。

**⚠️ 局限性**

局限性：多阶段调用导致推理延迟与成本上升；对中间推理记忆质量高度依赖，可能因错误信息传递导致最终答案偏差；模型主要在数字化清晰文档上优化，对手写、降质扫描或自然场景文本的泛化能力尚未充分验证。

---

## 426. DIVO: Continuous-time DVL-Inertial-Visual Odometry for Unmanned Underwater Vehicles

**arXiv ID:** 2607.04615 | [PDF](https://arxiv.org/pdf/2607.04615v1)

**作者:** Kyungmin Jung `[一作]` (McGill University), James Richard Forbes `[通讯]` (McGill University)

**通讯引用:** 2414 | [OpenAlex ID](https://openalex.org/A5023450612)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种连续时间的三模态 (DVL‑IMU‑视觉) 位姿估计系统 DIVO，用于无人水下航行器的定位与建图。

**💡 创新点**

创新点包括：①首次将高斯过程连续时间轨迹估计框架与异步传感器（DVL、相机、IMU）结合；②引入基于 SuperPoint+LightGlue 的学习式视觉前端，提升了在低可见度、光照变化及动态颗粒环境下的特征跟踪；③将 DVL 与 IMU 的预积分与 GP 运动先验统一在优化后端，实现在任意时间点插值与校正。

**🔧 技术方法**

核心技术：高斯过程连续时间轨迹估计、SME/多状态约束 Kalman 滤波、DVL 速度求解、SuperPoint 关键点提取与 LightGlue 匹配、IMU 预积分、非线性最小二乘优化（iSAM2）。

**📊 数据集**

评估数据集：1) 通过仿真生成的两条轨迹（5 轮圆轨迹和 EuRoC Vicon 房间轨迹）；2) 实际海底 Quarry 现场数据，含 4 条序列（Conveyor1/2、Truck1/2），配备立体摄像头、IMU、DVL，使用 Agisoft Metashape 生成光度学测量的地面真值。

**📈 对比分析**

与多种基线（ORB‑SLAM3、OKVIS2‑X、MSCKF‑DVIO、AQUA‑SLAM）在绝对/相对轨迹误差、轨迹覆盖率以及计算时长等指标对比，DIVO 在绝对误差（ATE）和相对误差（RTE）上均优于同类算法，覆盖率高达 99%+，并在复杂光照、动态颗粒和低特征环境下保持鲁棒性；计算量仅略高于视觉-惯性系统，仍能实现实时性能。

**⚠️ 局限性**

局限性：①系统为滑动窗口 3 帧，当前仅支持短期特征关联；②缺乏闭环/地图重定位模块，长期漂移仍需后处理；③对传感器标定与时间同步依赖较高；④在极端低特征或完全无视觉信息的场景下仍需辅助传感器。

---

## 427. SEAM: Smooth Execution of Action-Chunked Motion for Vision-Language-Action Policies

**arXiv ID:** 2607.04609 | [PDF](https://arxiv.org/pdf/2607.04609v1)

**作者:** Dijia Zhan `[一作]` (South China University of Technology), Jie Tang `[通讯]` (South China University of Technology)

**通讯引用:** 28429 | [OpenAlex ID](https://openalex.org/A5044791875)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练无关的推理时方法SEAM，解决流匹配视觉语言动作（VLA）策略中由于独立噪声采样导致的跨块不一致（多模态分叉）问题。

**💡 创新点**

核心创新在于Velocity‑Guided Loss Steering (VLS)，利用已执行块的未执行尾部作为一致性参考，在ODE积分过程中闭式校正，避免梯度反向传播和重采样。

**🔧 技术方法**

使用流匹配的ODE框架、Euler积分、闭式一致性目标修正、同步块执行结构。

**📊 数据集**

在LIBERO‑10长时序桌面操作任务集合上进行评估。

**📈 对比分析**

与基线π_0.5、ACT‑TE和RTC进行比较，SEAM在保持95.7%任务成功率的同时，将边界抖动降低28%，块切换不连续度降低27%，并将推理延迟仅提高1.01×，显著优于需要反向传播或重采样的方法。

**⚠️ 局限性**

局限性包括对一致性窗口长度和指导强度的敏感性，过强指导可能降低任务成功率；方法仅针对同步块执行场景，可能不适用于异步或动态长度块。

---

## 428. Simple-to-Complex Structured Demonstrations for Vision-Language-Action Learning

**arXiv ID:** 2607.04591 | [PDF](https://arxiv.org/pdf/2607.04591v1)

**作者:** Xinchuan Qiu `[一作]` (Hiroshima University), Yi Yu `[通讯]` (Hiroshima University)

**通讯引用:** 3896 | [OpenAlex ID](https://openalex.org/A5100745222)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种简单到复杂（S2C）结构化演示收集策略，用于提升Vision‑Language‑Action（VLA）模型在双臂机器人上的长期操控性能。

**💡 创新点**

创新点在于将任务先分解为基本操控、感知理解、任务执行三阶段，并通过环境标准化和逐步递增复杂度的方式组织演示数据，从而显著提升在有限演示数据下的学习效率与稳定性。

**🔧 技术方法**

主要技术包括：VLA模型训练（使用π_0.5）、演示收集的任务分解与环境配置、演示阶段的渐进式复杂度调度、以及对比实验中的性能评估。

**📊 数据集**

使用自定义的双臂SO‑101机器人演示数据集：对块抓取与排序收集300条S2C演示，对毛巾折叠收集300条S2C演示；基线为200条完整任务演示；两者均在相同的π_0.5模型与训练配置下评估。

**📈 对比分析**

与直接端到端演示收集相比，S2C策略在块抓取与排序任务中将成功率从0%提升至80%，在毛巾折叠任务中提升至25%；训练稳定性和学习曲线也表现出更快的收敛速度。

**⚠️ 局限性**

局限性包括：低成本硬件导致抓取与放置精度不足，毛巾折叠仍受不可预测变形和执行误差影响；演示分阶段仍需人工设计，缺乏自动化；未引入因果世界模型或错误恢复机制，导致对异常状态的鲁棒性不足。

---

## 429. Markov Decision Process Approximation Methods for Water Distribution Network Inspection and Maintenance: A Case Study of the U.S. Virgin Islands

**arXiv ID:** 2607.04626 | [PDF](https://arxiv.org/pdf/2607.04626v1)

**作者:** Minsuk Seo `[一作]` (Republic of Korea Army), Jefferson Huang `[通讯]` (Naval Postgraduate School)

**通讯引用:** 74 | [OpenAlex ID](https://openalex.org/A5050591605)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种面向维修的检查与维护决策框架，用于在数据稀缺、资源有限的水分配网络（如美国维尔京群岛）中实现最优维护策略。

**💡 创新点**

创新点在于将折现式马尔可夫决策过程与高保真水力模拟（WNTR）相结合，经过实证验证Markov性；利用系统级动力学实现“虚拟传感”，即通过可观测的水箱状态唯一识别特定管道故障，从而实现基于状态的维修决策。

**🔧 技术方法**

采用折现式马尔可夫决策过程、线性规划求解、WNTR水力模拟、时间序列交叉验证检验Markov假设、统计显著性检验等技术。

**📊 数据集**

使用美国维尔京群岛（STT-STJ）水网的现场可观测数据（水箱水位、泵状态）以及通过WNTR生成的水箱动态和服务可用性（WSA）数据。

**📈 对比分析**

通过与“始终维修”和“从不维修”两种基准策略比较，评估最优策略的成本优势。最优策略在不同管道场景下将期望折现成本降低44%–86%，并通过敏感性分析展示对折现因子、维修成本权重和故障概率的响应。

**⚠️ 局限性**

局限性包括：仅考虑单一管道的故障而未建模多重失效；转移概率采用计数估计，缺乏基于年龄或贝叶斯的更精细失效模型；行动空间仅限于DoNothing和Repair，未涵盖不完全维修或多级干预。

---

## 430. Score Distributions, Not Cells: Evaluating Single-Cell Perturbations Under Class Overlap

**arXiv ID:** 2607.04595 | [PDF](https://arxiv.org/pdf/2607.04595v1)

**作者:** Youssef Marrakchi `[一作]` (Massachusetts Institute of Technology), Sebastiano Cultrera di Montesano `[通讯]` (Broad Institute of MIT and Harvard)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了Classifier Discrimination Score（CDS），通过在分类器输出概率上做全细胞平均来评估单细胞扰动实验的扰动识别。

**💡 创新点**

利用全细胞概率聚合替代传统细胞级准确率，克服可区分但不可分离问题，使扰动识别准确率大幅提升。

**🔧 技术方法**

使用线性、MLP和Transformer三种分类器训练细胞级softmax概率，再对每个扰动的细胞概率做平均得到CDS，并与传统PDS进行对比。

**📊 数据集**

在大规模单细胞扰动数据集Tahoe-100M和较小规模的Virtual Cell Challenge (VCC) 数据集上进行实验验证。

**📈 对比分析**

通过rank‑1召回率和平均真匹配排名进行比较，CDS在细胞稀缺与完整样本两种场景下均达到0.96–1.00的rank‑1，显著优于PDS在相同条件下的0.70–0.86。

**⚠️ 局限性**

CDS仅能识别已在训练中出现的扰动，无法评估未知或组合扰动，也未直接估计贝叶斯误差。

---

## 431. DiCE-CIR: Direct Composition Learning for Efficient Zero-Shot Composed Image Retrieval

**arXiv ID:** 2607.04665 | [PDF](https://arxiv.org/pdf/2607.04665v1)

**作者:** Gwang-Ho Na `[一作]` (Korea University), Seong-Whan Lee `[通讯]` (Korea University)

**通讯引用:** 22623 | [OpenAlex ID](https://openalex.org/A5011014617)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种直接组合学习方法DiCE-CIR，用于零样本组合图像检索。

**💡 创新点**

创新点在于直接将参考图像和编辑文本嵌入拼合，避免了投影和重新编码的序列依赖与训练推理不一致问题。

**🔧 技术方法**

采用CLIP预训练视觉语言模型、轻量级门控MLP组合模块以及对齐、残差和对比损失三项训练目标。

**📊 数据集**

自动构造训练样本基于CC3M图像-标题对，并用LLM生成编辑文本和目标标题，规模约1.22M样本。

**📈 对比分析**

与Pic2Word、SEARLE、LinCIR等投影式方法比较，在CIRCO和CIRR基准上均实现了SOTA或竞争性表现，同时训练速度提升10–24倍。

**⚠️ 局限性**

局限在于对编辑文本的语义一致性依赖较强，且在部分基准中仍受参考图像干扰，未来需进一步提升对复杂编辑的鲁棒性。

---

## 432. Retroactive Chain-of-Thought (RetroCoT): Forensic Reconstruction Prompts as a Safety Diagnostic Across Model Generations

**arXiv ID:** 2607.04645 | [PDF](https://arxiv.org/pdf/2607.04645v1)

**作者:** Samira Hajizadeh `[一作]` `[通讯]` (Columbia University), Samira Hajizadeh (Columbia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种单轮攻击方法，将有害请求重新表述为法医重建任务，通过先验假设设定、法医分析者角色和逆向链式推理来诱导模型生成有害内容。

**💡 创新点**

创新点在于首次将先验假设（presupposition accommodation）与法医分析者人格（forensic persona）结合，形成一个全新的语用框架，使模型在不直接接收有害命令的情况下，仍能通过分析已发生的事件来提供有害信息；同时揭示了安全对齐对语用形式的敏感性，并发现了“生成差距”（generation gap）与对抗性反馈的可绕过性。

**🔧 技术方法**

主要技术包括：先验假设设定（将命令转化为已发生事实）、角色设定（让模型扮演法医分析者）、逆向链式推理（从结果逆推先前步骤），以及对抗性反馈循环（利用评估者的批评继续推进同一语用框架）。

**📊 数据集**

使用数据集 AdvBench（50个有害行为示例），并在GPT‑4o、GPT‑4o‑mini、GPT‑5‑family（GPT‑5.4‑mini、GPT‑5‑mini）以及工具调用模拟环境中进行评估。

**📈 对比分析**

比较方法：对比直接请求（direct）、过去式改写（past‑tense）以及本文提出的法医重建攻击。实验显示：在GPT‑4o上，攻击成功率从0%提升至58%；在GPT‑4o‑mini上从4%提升至52%；而GPT‑5‑family模型在直接请求时全部拒绝，但在对抗性反馈后成功率可提升至48%–52%。工具调用情境下，成功率降至20%–26%，但仍表明攻击可在代理管道中传播。

**⚠️ 局限性**

局限性包括：仅评估50条 AdvBench 行为；实验使用的是专有模型，缺乏公开可复现性；判分依赖双评审和GPT‑4o 验证，可能存在主观误差；对抗反馈实验使用 GPT‑4o 生成的种子响应，可能在一定程度上夸大了对更强模型的成功率。

---

## 433. Wrong Before Right: Late Rescue and Interface Failure in Aligned Language Models

**arXiv ID:** 2607.04640 | [PDF](https://arxiv.org/pdf/2607.04640v1)

**作者:** Jiaqi Deng `[一作]` `[通讯]`, Jiaqi Deng

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了对齐语言模型内部正确性如何被构建，发现中层会暂时偏向错误答案，后期层通过“wrong-dip”现象进行纠正并产生最终输出。

**💡 创新点**

创新点在于提出因果型wrong-dip指标，证明其能够预测结构压缩失败，并展示一种中层hinge惩罚的LoRA微调可显著削弱dip、提升压缩鲁棒性且保持表面精度。

**🔧 技术方法**

采用层级差分对比、调优/对数激活镜、因果激活移植（patchscope）、低秩压缩、层删减、通道剪枝、量化等技术，对模型内部轨迹进行定量分析与干预。

**📊 数据集**

使用的数据集包括278对极性对照的最小对（值翻转、否定、许可）和200对角色绑定对，随后扩展到96个自然语言情节桥接样本和240条严格测试项。

**📈 对比分析**

在17种规模（0.5B–32B）和三大族群模型上进行比较，发现高dip项目在晚期低秩压缩下易翻转（3–7倍），但在量化下无预测力；经过中层hinge惩罚后mid-SVD保留率从≈0.87提升到≈0.94。

**⚠️ 局限性**

局限性包括仅在单词级最小对任务上验证，单核实验、仅LoRA微调、对因果测量的patchscope依赖，且对开放式对话的外推性尚未充分验证。

---

## 434. Aperture-aware Dispersion 5-D Light-field Imaging Spectrometer

**arXiv ID:** 2607.04635 | [PDF](https://arxiv.org/pdf/2607.04635v1)

**作者:** Chenglong Huang `[一作]` (Nanjing University), Xun Cao `[通讯]` (Nanjing University)

**通讯引用:** 6903 | [OpenAlex ID](https://openalex.org/A5009992843)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种单探测器的5D光谱视场相机（ADLIS），通过在孔径平面上使用石英晶体相位板实现可调的偏振分散编码，并在端到端（E2E）框架中联合优化光学设计与深度重建网络，实现全空间分辨率的高维光场重建。

**💡 创新点**

创新点包括：①利用可微薄膜厚度调控的偏振相位板进行孔径分散编码，消除了微透镜阵列导致的空间-角度分辨率折衷；②在E2E深度学习框架中将光学参数视为可训练变量，实现光学与算法的协同优化；③通过多帧偏振旋转实现编码多样性，提升对极端动态范围场景的鲁棒性；④首次将高维光场重建与物理光学模型紧密耦合。

**🔧 技术方法**

采用的技术包括：偏振分散相位板（石英晶体相位调制）、多帧偏振旋转编码、可微光学模型（薄膜厚度与响应函数）以及Restormer网络的端到端训练，配合Adam优化器和学习率衰减实现系统联合优化。

**📊 数据集**

训练与测试主要使用RealSLF光谱视场数据集（7×5视角，36光谱通道），并通过模拟数据生成多帧测量；实验验证亦使用真实采集的光谱和角度数据进行对比。

**📈 对比分析**

与传统CFA编码和其他单探测器5D-SLF方法（如C^3SLFI、Hyper-LIFT等）进行对比，采用PSNR、SSIM、SAM、相差图/视角一致性等指标；ADLIS在2帧下实现PSNR≈42 dB、SSIM≈0.984、SAM≈6.6，并在空间信息效率（SIE）上达到100%，显著优于现有方法。

**⚠️ 局限性**

局限性包括：目前视角分辨率仅为3×3，若需更高角度分辨率需重新设计相位板；对测量噪声和光学误差仍较敏感；多帧测量需要旋转分析仪，增加了系统复杂度；在极端高动态范围或宽光谱范围下的性能需进一步验证。

---

## 435. Formal Disco: Scalable Open-Ended Generation of Formally Verified Programs

**arXiv ID:** 2607.04631 | [PDF](https://arxiv.org/pdf/2607.04631v1)

**作者:** Gabriel Poesia `[一作]` (Harvard University), Nada Amin `[通讯]` (Harvard University)

**通讯引用:** 55336 | [OpenAlex ID](https://openalex.org/A5054275386)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一个分布式多智能体系统，利用LLM生成、修复和扩展形式化验证语言（Dafny、Verus、Frama‑C）的程序，自动化合成大规模、主题多样、结构复杂的已验证程序，并以此数据集训练开源模型以提升验证相关任务的性能。

**💡 创新点**

创新点在于：① 通过“Initiator‑Fixer‑Extender”三类专门化工作者和共享议程实现高效的循环式程序生成；② 引入最大熵原则，将程序特征熵作为自我改进和数据生成的目标，避免多样性崩溃；③ 通过外部熵源（随机 GitHub README 与语言文档片段）保证开放式生成的主题与语言特征多样性；④ 在三种验证语言上发布规模超过现有公开数据集数倍的合成验证程序数据集。

**🔧 技术方法**

使用Claude 4.5（Sonnet/Opus）进行初始蒸馏，随后对开源 Qwen 2.5‑Coder 32B 进行 LoRA 微调；多智能体通过远程 API 调度任务；使用编译器/验证器作为奖励信号；采用熵最大化策略对训练样本进行筛选和排序；利用随机 README 与文档片段作为外部熵源。

**📊 数据集**

主要数据集：synthetic 验证程序集 dafny‑disco、framac‑disco、verus‑disco（共计 100k+ 完整验证程序）；对比基准数据集包括 DafnyBench、VerusBench、SAFE、VeruSyn、VeruSAGE 等；实验中还使用了从自身生成程序中提取的注释、证明差分进行微调。

**📈 对比分析**

对比方法：① 与 Claude 4.5 前沿模型在 Initiator/Fixer/Extender 任务中的成功率；② 在熵、稀疏曲线、程序长度等维度评估多样性；③ 在逻辑注释任务和引理证明任务上与 Claude 4.5 Opus 及 SAFE 数据集微调模型的 Pass@k 结果比较。结果显示：① 微调后的 Qwen 2.5‑Coder 在所有工作者任务上能与甚至超过 Claude 基线；② 稳定提升程序熵和复杂度；③ 在验证相关下游任务上，微调模型可将 Pass@16/Pass@1 约提升 2‑3 倍，接近或匹配前沿模型性能。

**⚠️ 局限性**

局限性：① 仅使用最简单的监督微调和差分训练，未充分利用链式推理、强化学习等更强学习方式；② 熵最大化仅覆盖有限手工挑选的特征，对未优化特征（如注释结构）仍存在多样性不足；③ 仍依赖外部随机 README 与文档片段维持开放性，若无此类熵源可能导致主题坍塌；④ 仅验证三种自动化验证语言，扩展到交互式定理证明或其他领域仍待研究。

---

## 436. Exploiting Structural Properties for Efficient Constraint-Aware HNSW Hyperparameter Tuning

**arXiv ID:** 2607.04630 | [PDF](https://arxiv.org/pdf/2607.04630v1)

**作者:** Geon Choi `[一作]` (Seoul National University), Jaeyoung Do `[通讯]` (Seoul National University)

**通讯引用:** 1027 | [OpenAlex ID](https://openalex.org/A5024989829)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种面向 HNSW（层级可导航小世界）图的约束感知超参数调优框架，用于在满足精度、吞吐量、构建时间和内存等多重约束的前提下快速找到最优或近优配置。

**💡 创新点**

核心创新在于挖掘并利用 HNSW 超参数空间的结构规律：查询参数 efs 的单调可行性边界、构造参数 efc 的主导单峰（unimodal）趋势以及资源（构建时间、索引大小）的可分离性，并基于此设计了确定性二分/三分搜索与轻量级资源筛选器，显著提升样本效率与收敛速度。

**🔧 技术方法**

技术手段包括：①基于 HNSW 内部机制的结构化搜索策略（二分搜索 efs、三分搜索 efc、二分搜索 M）；②构建时间和索引大小的闭式代价模型（log‑log + 线性项），并在线校准；③资源约束下的保守性安全裕度；④回滚与迁移学习（Retuning）来适应语料库或查询分布漂移。

**📊 数据集**

实验使用了五个公开数据集（来自 ANN‑Benchmarks 的四个数据集 + 通过 OpenCLIP 对视频帧编码得到的 YouTube 视频检索数据集）以及三种 HNSW 实现（Faiss、Hnswlib、Milvus）。

**📈 对比分析**

与随机搜索、网格搜索、Optuna、NSGA‑II、ECI、VDTuner 等基线进行对比。实验表明，在相同的四小时调优预算内，本文方法在满足 Recall≥0.95 或 QPS≥阈值的约束下，平均提升 1.36‑1.50× 吞吐量或 0.6‑11% Recall，最快收敛时间可达基线的 1/44，且几乎达到穷举搜索（Oracle）的 98‑100% 目标性能。

**⚠️ 局限性**

主要局限包括：①假设 HNSW 的标准实现（无 GPU、磁盘扩展或自定义邻居选择）；②对非 HNSW ANN 结构（如 IVF‑PQ、ScaNN、NSG）不直接适用；③在高维无聚类结构或极端均匀分布时，单峰假设可能失效，导致搜索陷入局部峰值；④资源模型的校准需要一定数量的实际构建样本，极大规模或实时在线环境可能需要额外的适配。

---

## 437. Reliability and Identifiability in Persona-Trained Monte Carlo: Variance Decomposition, Stability Bounds, and the Identifiability of Heterogeneous News Reaction

**arXiv ID:** 2607.04627 | [PDF](https://arxiv.org/pdf/2607.04627v1)

**作者:** Salavat Ishbulatov `[一作]` `[通讯]` (Independent researcher), Salavat Ishbulatov (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并理论化 Persona‑Trained Monte Carlo（PTMC）方法，利用聚类的神经策略机器人在极限订单簿中模拟市场，估计各种市场结果分布，并给出可靠性保证。

**💡 创新点**

① 将外部人群抽样视为输入不确定性并分解方差；② 给出误差预算和最优计算分配；③ 对异质新闻反应的可识别性进行理论证明并给出 √n 估计和边界校正检验；④ 在理论层面证明 PTMC 在同质模拟器和“减少形式”预测器上的优势（无可避免的偏差下限为零，干预目标实现无可匹敌的最优误差）。

**🔧 技术方法**

蒙特卡罗模拟、随机效应 ANOVA、总变差与 Wasserstein 距离的耦合稳定性分析、Doeblin 条件下的均匀收敛、随机系数识别理论、非参数混合分布的矩问题、极值与边界检验（12χ²_0+12χ²_1）等。

**📊 数据集**

理论性论文无实证数据，参照伴随论文的实验设计，主要基于事件研究样本（新闻冲击与订单流响应）。

**📈 对比分析**

与同质人群模拟器以及贝叶斯网络、深度集成、LLM 预测等“减少形式”预测器进行比较；在存在异质性时证明 PTMC 无可避免的偏差下限为零，且对干预目标实现无可匹敌的最优误差。

**⚠️ 局限性**

① K→∞ 的大数极限未证明；② 对订单簿链的 Doeblin 条件难以验证；③ 对路径函数（最大回撤等）的统一 T 限定性尚未完成；④ 对 Q 的非参数估计存在严重 ill‑posed 性，理论上仅给出 √n 估计。

---

## 438. CARD: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning

**arXiv ID:** 2607.04619 | [PDF](https://arxiv.org/pdf/2607.04619v1)

**作者:** Ganesh Pavan Kartikeya Bharadwaj Kolluri `[一作]` (University of Essex), Ravi Shekhar `[通讯]` (University of Essex)

**通讯引用:** 928 | [OpenAlex ID](https://openalex.org/A5053856362)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种无音频编码器的音频字幕模型CARD，利用跨组件蒸馏将预训练CLAP教师的感知层和语义层知识分别注入投影器和LLM；

**💡 创新点**

创新点在于：①跨组件蒸馏策略——将教师早期低级感知表示监督投影器、后期高级语义表示监督LLM；②在推理时完全移除音频编码器，仅保留投影器和LLM；

**🔧 技术方法**

采用了CLAP预训练教师、LoRA对Qwen3‑4B的适配、轻量级一维卷积投影器、两阶段训练（蒸馏+细调）以及交叉组件蒸馏损失；

**📊 数据集**

训练使用WavCaps、Auto‑ACD、AudioCaps、Clotho、MACS以及文本指令混合数据；评估在AudioCaps和Clotho两个公开评测集；

**📈 对比分析**

与基线SLAM‑AAC等模型对比，CARD*在无编码器条件下实现AudioCaps 55.4% CIDEr‑D、Clotho 27.5% CIDEr‑D，比无蒸馏模型提升12%/5%，但仍低于保留编码器的上限66.4%/39%；

**⚠️ 局限性**

局限性包括：性能仍落后于含编码器模型；对LoRA容量高度敏感；对不同非语音事件的鲁棒性未充分验证；需要进一步提升模型泛化与效率。

---

## 439. SILO: Simulation-in-the-Loop Sim-to-Real Transfer for Multi-Stage Cable Routing

**arXiv ID:** 2607.04616 | [PDF](https://arxiv.org/pdf/2607.04616v1)

**作者:** Stone Tao `[一作]` (NVIDIA Corporation), Iretiayo Akinola `[通讯]` (NVIDIA Corporation)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本工作提出了一种基于GPU并行刚体仿真与强化学习的多阶段电缆路径规划系统，结合SILO部署实现了在真实机器人上的零样本成功转移，显著提升了成功率并将循环时间缩短约2倍。

**💡 创新点**

创新点在于①用GPU并行刚体模型近似线性可变形电缆，②将任务局部化为路由子任务并采用局部化RL策略，③设计了多段折线状态估计算法匹配仿真观察，④提出SILO仿真‑实机同步框架，消除传统系统ID与碰撞风险，实现零样本sim‑to‑real转移。

**🔧 技术方法**

技术手段包括ManiSkill3/PhysX GPU并行刚体仿真、Proximal Policy Optimization强化学习、运动规划原语（GraspCable、MoveToHarness）、基于SAM2+Foundation Stereo的电缆分割与深度投影估计、以及SILO仿真‑实机循环同步。

**📊 数据集**

使用多种实际电缆（尼龙绳、以太网线、充电线、HDMI线）与随机采样的挂架姿态分布；不依赖演示数据，而是通过仿真生成中间状态作为训练样本。

**📈 对比分析**

与脚本化方法和层次化模仿学习（h-IL）基线对比，SILO在1-3个挂架的成功率分别为24/24、22/24、18/24，平均耗时约87.5秒，远优于h-IL约200秒；在不同电缆类型下成功率保持在18/24至14/24之间，体现了良好的零样本泛化能力。

**⚠️ 局限性**

局限性包括需预先知晓挂架几何与姿态，SILO同步过程受限于准静态任务难以扩展至高速动态；仿真模型对极大入射角的鲁棒性有限；在极窄缝隙或大直径电缆下仍易失效；缺乏完整的视觉感知闭环。

---

## 440. Integrated Forward-Inverse Network for Lensless Image Reconstruction

**arXiv ID:** 2607.04608 | [PDF](https://arxiv.org/pdf/2607.04608v1)

**作者:** Donggeon Bae `[一作]` (Seoul National University), Seung Ah Lee `[通讯]` (Seoul National University)

**通讯引用:** 2642 | [OpenAlex ID](https://openalex.org/A5089298785)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种集成前向-逆向网络（IFIN），在每个尺度交替使用可微前向投影和可学习逆更新，以实现无镜头图像重建。

**💡 创新点**

创新点在于：1）在编码器-解码器的每个尺度插入前向-逆向块，保持测量域与图像域信息同步；2）同时学习空间变异的PSF场，提升对校准误差和系统不确定性的鲁棒性。

**🔧 技术方法**

采用的技术包括：编码器-解码器架构；可微前向系统算子（卷积）与可学习逆系统算子（Wiener式反卷积）交织；二维频率正则化；PSF场编码器；FFT实现的高效前向/逆投影。

**📊 数据集**

使用的数据集为三种无镜头基准（DiffuserCam、WiderCam（自研）、MultiWienerNet），以及 Gaussian 去模糊和内插全息（inline holography）实验。

**📈 对比分析**

与经典逆向、纯数据驱动和混合模型进行对比，IFIN 在所有基准上均获得最高 PSNR/SSIM，尤其在 WiderCam 上提升约 +2.58 dB，整体性能优于现有方法。

**⚠️ 局限性**

局限性包括：对大 PSF 和强空间变异系统的 FFT 开销较高；需要相对稳定的前向模型，严重饱和或几何变形导致信息不可恢复；单核（k=1）设置虽低成本，但在高变异场景下性能有限。

---

## 441. Do All Visual Tokens Matter Equally? Object-Evidence Preserving Token Merging for Vision-Language Retrieval

**arXiv ID:** 2607.04605 | [PDF](https://arxiv.org/pdf/2607.04605v1)

**作者:** Suhyeong Park `[一作]` (Catholic University of Korea), Jaewoo Kang `[通讯]` (Korea University)

**通讯引用:** 16560 | [OpenAlex ID](https://openalex.org/A5076917278)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SaMer，一种基于对象感知的视觉token合并框架，用于多向量视觉语言检索，压缩图像端token数量同时保留可被查询token选择的细粒度视觉证据。

**💡 创新点**

创新点在于利用训练时的对象标注作为合并先验，抑制跨实例混合，并通过投影层的仅适配训练保持与原始MaxSim检索接口的一致性；此外通过只压缩检索索引而不改动视觉编码器，显著降低存储与计算成本。

**🔧 技术方法**

技术方法包括特征-空间软聚类合并、基于对象先验的分配惩罚、投影层单参数适配以及MaxSim晚期交互评分。

**📊 数据集**

使用的数据集包括 Flickr30K（实体版）、MSCOCO、ImageCoDe、DocVQA 进行检索与定位评估。

**📈 对比分析**

与单向量、VLM、原始多向量以及其他压缩基线（H-Pool、HPC、SAP 等）对比，SaMer 在 K=64 时将图像端token压缩 93% 以上，ColPali 的 R@1 从 77.0 提升至 82.4，ColQwen2 从 73.6 提升至 79.3，并且在定位指标上显著优于所有基线；同时存储降低 16×、MaxSim 计算减少 16×，查询吞吐量提升 4–9×。

**⚠️ 局限性**

局限性在于对对象级检索最优化，文档级检索（如 DocVQA）仍受限；另外在极大 token 数量或不含显著对象的图像场景下，合并后可能仍难以完全保留所有细粒度信息。

---

## 442. Measuring What Matters: A Unified Evaluation Framework for GNN Explainability

**arXiv ID:** 2607.04600 | [PDF](https://arxiv.org/pdf/2607.04600v1)

**作者:** Francesco Paolo Nerini `[一作]` (Sapienza University of Rome), Alan Perotti `[通讯]` (Intesa Sanpaolo AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套无监督、基于四个核心指标（稳定性、有效紧凑度、相关性和时间）的统一 GNN 可解释性评估框架，并在 8 种解释器、10 个 GraphML 任务上进行了大规模基准测试。

**💡 创新点**

创新点在于把基于 tabular 的指标迁移到图数据、同时分别评估边和特征归因、无需人工 ground‑truth，并给出可操作的可解释性阈值与选择指南。

**🔧 技术方法**

采用的技术包括梯度归因（Saliency、Input×Gradient、Integrated Gradients、LRP）、基于掩码的扰动方法（GNNExplainer、GraphMask）以及自定义的 Stability、Effective Compactness、Pertinence 计算和早停优化。

**📊 数据集**

使用的数据集包括 Cora、GitHub、BAShapes、MovieLens（改版）以及一个合成数据集，覆盖节点与边的分类与回归任务。

**📈 对比分析**

通过 Pareto 图和平均指标对比发现 Input×Gradient 与 Integrated Gradients 在相关性和紧凑度上占优，且仅耗时 0.01–0.03 秒；其它方法在准确性上相对落后，尤其 GraphMask。

**⚠️ 局限性**

局限性包括只评估能输出边与特征归因且许可公开的 8 种解释器、未覆盖无监督或自解释模型、以及指标与真实业务因果关系的对齐仍需进一步研究。

---

## 443. ICME 2026 Grand Challenge on Cross-Scenario Defect Detection and Fine-Grained Severity Grading for High-Precision Manufacturing

**arXiv ID:** 2607.04675 | [PDF](https://arxiv.org/pdf/2607.04675v1)

**作者:** Wei Sun `[一作]`, Atik Shahariar `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文组织并总结了ICME 2026跨场景缺陷检测与细粒度严重度分级大赛，提出了两条轨道并提供了涵盖七类缺陷的大规模微观图像数据集，评估了多种算法的跨域泛化和严重度预测能力。

**💡 创新点**

创新点在于：①将跨场景检测与严重度分级两大任务整合为一套完整挑战；②引入多模态（Vision‑Language）融合、无监督异常预筛选、序数学习等技术以提升域适应与细粒度分类；③构建包含像素级实例标注和严重度标签的高分辨率数据集，为工业缺陷分析提供新的基准。

**🔧 技术方法**

主要技术包括Mask2Former、YOLOv8、RT‑DETR、Transformer‑based Detectors、Swin‑Large、VLM（LoRA‑fine‑tuned Qwen3‑VL‑8B、Florence‑2、Gemma‑4‑31B）、DualAnoDiff、WBF、Varifocal Loss、CORN、RandomForest、Morphology‑Aware Ordinal Learning等。

**📊 数据集**

使用了高分辨率半导体晶圆表面微观图像数据集，包含3,800+张带像素级实例标注的七类缺陷（Scratch, Dent, Particle, Damage, Stain, Bubble, Chipping），以及2,600+张带严重度标签的图像。

**📈 对比分析**

通过与86支参赛队伍的排行榜对比，Track 1最高得分0.806（mIoU 0.618，cls 0.746，screen 0.992），Track 2最高得分0.811（mIoU 0.662，cls 0.673，grade 0.907），表明主流方案已在跨域检测与严重度分级上取得可观成绩。

**⚠️ 局限性**

局限性包括：①数据分布偏差导致的类别不平衡和域迁移挑战；②严重度标签主观性导致评测误差；③评测指标对罕见类别敏感，可能掩盖模型对低频缺陷的实际性能；④大多数参赛方案仍依赖大型模型与多模型集成，部署成本较高。

---

## 444. GlaKG: A Biomarker-Centric Fundus Knowledge Graph for Explainable Glaucoma Diagnosis and Risk Assessment

**arXiv ID:** 2607.04673 | [PDF](https://arxiv.org/pdf/2607.04673v1)

**作者:** Cheng Huang `[一作]` (University of Texas Southwestern Medical Center), Guanghua Xiao `[通讯]` (University of Texas Southwestern Medical Center)

**通讯引用:** 13005 | [OpenAlex ID](https://openalex.org/A5009964025)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建基于结构性眼底标志物的知识图谱（GlaKG），并通过图推理链实现可解释的青光眼诊断与风险评估。

**💡 创新点**

将临床验证的11条诊断规则嵌入异构知识图谱，并与图推理链相结合，首次实现完整的诊断证据链可追溯性。

**🔧 技术方法**

利用ResNet50提取图像特征、图神经网络（GCN/GAT）处理异构图、规则编码、后处理融合（α权重）及梯度提升分类器。

**📊 数据集**

使用公开的AI注释眼底图像数据集（689例，含结构化标志物与风险等级标签）。

**📈 对比分析**

与传统图像基线（LR、RF、GB、MLP）以及GCN/GAT对比，GlaKG在二分类F1达0.9953、AUC0.9988，四分类准确率0.930、加权F1 0.922，显著优于基线。

**⚠️ 局限性**

受限于依赖人工标注的结构化标志物、无时间序列信息、跨数据集泛化性不足及需进一步自动化标志物提取。

---

## 445. FormalRx: Rectify and eXamine Semantic Failures in Autoformalization

**arXiv ID:** 2607.04655 | [PDF](https://arxiv.org/pdf/2607.04655v1)

**作者:** Haocheng Wang `[一作]`, Zhijiang Guo `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种可解释的自动形式化评估框架，将二元判断转化为包含错误类型、定位与纠正的诊断报告；

**💡 创新点**

创新点在于构造了包含28个子类的层级错误分类法（Semantic‑Constraint‑Implementation），并基于此生成了大规模带诊断标签的误差数据集及统一的四任务生成模型；

**🔧 技术方法**

技术主要包括基于LLM的错误注入与再标注、联合诊断生成的监督微调、以及使用Qwen3系列模型进行多任务推理；

**📊 数据集**

数据集由约17k对齐的自然语言–Lean4对构成，利用LLM注入生成56k错误样本，最终筛选得到52k条带完整诊断标注的样本；

**📈 对比分析**

在四项评测任务（判定、分类、定位、纠正）上，微调后的Qwen3-8B在Verdict、Categorization、Localization、Correction分别取得F1≈0.881、0.709、准确率≈0.750、0.729，均优于零样本基线与主流前沿模型；

**⚠️ 局限性**

局限性包括错误分类法可能不覆盖所有极端或新出现的错误、实现维度高度依赖Lean 4，且在外部数据集的泛化性能相对较弱，需进一步扩展语言与域的适用范围。

---

## 446. Enhancing Video Physical Consistency via Role-aware Joint Training and Modality-decoupled Denoising

**arXiv ID:** 2607.04653 | [PDF](https://arxiv.org/pdf/2607.04653v1)

**作者:** Guangting Zheng `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8577 | [OpenAlex ID](https://openalex.org/A5053344541)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在预训练视频扩散模型上轻量化微调，提出VPT框架通过角色感知映射、模态解耦去噪与交叉步自引导等技术提升视频生成的物理一致性。

**💡 创新点**

创新点在于：①将实体分为agent、controlled、passive、background的角色感知联合表示；②采用模态解耦去噪和aux loss衰减，避免容量冲突和推理误差；③利用交叉步自引导将中间模型的物理导向融入最终生成。

**🔧 技术方法**

使用光流（RAFT）、角色映射（Qwen3‑VL+SAM3）作为辅助模态，模态解耦去噪（独立时间步）、aux loss annealing、classifier‑free guidance 与 cross‑step auto‑guidance，微调时加入LoRA等轻量化模块。

**📊 数据集**

训练基于WISA‑80K数据集，评估使用VideoPhy、VideoPhy‑2 以及VBench 三大公开基准。

**📈 对比分析**

与Wan2.1、全微调和VideoJAM比较，VPT在VideoPhy上SA提升39.4%、PC提升17.9%；在VideoPhy‑2上同样取得最高分；VBench整体分数从76.93升至79.58，质量维度均保持或提升。

**⚠️ 局限性**

依赖自动光流与VLM角色分配与掩膜，误差可能影响物理先验学习；仅在Wan backbones上验证，未覆盖更大模型或更广泛的物理视频数据集。

---

## 447. Machine Learning for Depression Screening and Intervention: an Original Circadian Rhythm Score-based Methodology

**arXiv ID:** 2607.04648 | [PDF](https://arxiv.org/pdf/2607.04648v1)

**作者:** Bin Wang `[一作]` (Ocean University of China), Tianrui Li `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 114567 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种整合睡眠、日间小睡、社交和体育活动的昼夜节律评分（CRS），用于抑郁风险筛查与干预分析

**💡 创新点**

CRS通过监督学习在保持行为语义的前提下压缩多域行为，近乎无损保留抑郁筛查能力；同时结合SHAP解释和因果推断实现可操作的干预建议

**🔧 技术方法**

LightGBM（梯度提升树）、SHAP、交互式逻辑回归、对抗性回归/计量经济学中的因果效应估计

**📊 数据集**

中国健康与退休纵向研究（CHARLS）2018年数据，样本 15,233 名 45 岁以上人群

**📈 对比分析**

与仅用原始行为特征或仅用协变量的模型相比，CRS+原始特征+协变量模型在测试集上 ROC‑AUC 0.825（PR‑AUC 0.726），CRS 单独模型 AUC 0.785，表明压缩效果几乎无损；通过交叉验证与自举置信区间验证稳健性

**⚠️ 局限性**

研究基于横断面自报数据，缺乏时间序列验证；行为测量受回忆偏差；因果推断依赖强假设，未能证实长期因果关系

---

## 448. Learning Structured Visual Compositional Representations for Weakly Supervised Referring Expression Comprehension

**arXiv ID:** 2607.04638 | [PDF](https://arxiv.org/pdf/2607.04638v1)

**作者:** Lian Xu `[一作]` (University of Western Australia), Dan Xu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 485171 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种结构化视觉组合表示（SVCR）框架，用统一的二维结构（单体对象嵌入 + 关系嵌入）和组合对齐机制，解决弱监督下的指代表达理解任务。

**💡 创新点**

创新点在于：①显式构造对象与关系双层视觉空间，打破传统扁平化锚点表示；②使用可学习语义基向量提升单体区分度；③将句子级与子句级文本分别与关系与单体视觉嵌入对齐，形成组合对齐；④通过多层对比与一致性正则提升弱监督下的训练效果。

**🔧 技术方法**

技术手段包括：YOLOv3/YOLOv5 提取锚点特征并通过动态路由融合 DINOv2、CLIP、DepthAnything 等视觉基底特征；LSTM+多层自注意力文本编码；信息熵对比损失（InfoNCE）与多粒度对齐；正交多样性损失与层次一致性约束；门控融合与语义基投影。

**📊 数据集**

使用标准指代表达基准数据集：RefCOCO、RefCOCO+、RefCOCOg。

**📈 对比分析**

与多种弱监督 REC 方法（ARN、IGN、DTWREG、RefCLIP、APL、WeakMCN、DViN）以及部分全监督 REC 基线对比；在所有数据集上均取得 71–74% IoU@0.5 的成绩，较当前最强弱监督方法提升 2–6% 并逼近监督基线。

**⚠️ 局限性**

局限性包括：对否定句与复杂视觉遮挡的处理不充分；依赖锚点特征，难以捕获更长距离/全局关系；对极端多样化场景与长文本的鲁棒性待验证；模型在极大锚点集合下的计算成本和推理效率仍有提升空间。

---

## 449. Correctness, confidence, and context: Framing software assurance in the AI age

**arXiv ID:** 2607.04667 | [PDF](https://arxiv.org/pdf/2607.04667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 450. Attention Limited Reward Learning

**arXiv ID:** 2607.04590 | [PDF](https://arxiv.org/pdf/2607.04590v1)

**作者:** Wenqian Xing `[一作]` (Stanford University), Wenqian Xing `[通讯]` (Stanford University)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5086718152)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究人类在二元比较中受到注意力限制时对AI输出的评估如何影响奖励学习。

**💡 创新点**

提出一种注意力缩放的比较通道，揭示标准 Bradley–Terry 模型在注意力异质性下可能产生误导性排名，并给出信息理论下的误差定价。

**🔧 技术方法**

使用随机注意力理论、统计排名、Fisher 信息、KL 与 Fano 下界以及组合 Hodge 分解等技术。

**📊 数据集**

在 Chatbot Arena 的 57,477 条人类投票数据以及公开的感知比较实验（旋转条形图）中进行验证。

**📈 对比分析**

与 Bradley–Terry 近似对比，发现后者能捕捉到更多循环信息；实验表明模型能准确估计循环能量并解释标签与响应时间、凝视等额外信息的差异。

**⚠️ 局限性**

局限在于缺乏对注意力参数的直接测量、聚合投票可能引入额外循环噪声，以及仅在静态对比任务中验证，尚未直接推广到动态 RLHF 环境。

---

## 451. StructuredEdit: Constraint-Aware Graphic Design Editing via Differentiable Parameter Propagation

**arXiv ID:** 2607.04612 | [PDF](https://arxiv.org/pdf/2607.04612v1)

**作者:** Veeramanohar Avudaiappan `[一作]` (Amrita Vishwa Vidyapeetham), Ritwik Murali `[通讯]` (Amrita Vishwa Vidyapeetham)

**通讯引用:** 122 | [OpenAlex ID](https://openalex.org/A5012446221)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 StructuredEdit 系统，利用参数操纵和 Differentiable Parameter Propagation (DPP) 对图形设计进行精准编辑，避免传统像素生成导致的约束违背。

**💡 创新点**

创新点在于将硬约束嵌入训练过程，通过可微光栅化器将像素空间约束误差反向传播到参数预测，形成约束感知的视觉语言模型，并采用候选-过滤的混合数据生成策略。

**🔧 技术方法**

使用了 LoRA 微调的 Qwen2‑VL‑7B、可微光栅化器、Typed Layer JSON 表示、Grounding DINO、SAM、Tesseract OCR、LaMa 等技术，以及交叉熵与多项约束损失的联合训练。

**📊 数据集**

使用了 125,000 条经过验证的编辑三元组数据集，来源于 25,000 个设计模板（23,651 来自 Crello、1,349 手工案例），覆盖排版、布局、颜色、内容替换和层级调整等任务。

**📈 对比分析**

在与 GPT‑4V 和标准 SFT 的对比实验中，StructuredEdit 的约束满足率达 89%（GPT‑4V 52%），IoU 提升至 0.82、字体识别率 0.76，且在用户研究中编辑时间减少 33%、纠正次数下降 44%。

**⚠️ 局限性**

局限性包括对可微光栅化器的依赖导致计算开销增加、对超大尺寸或复杂矢量图形的支持尚有限，以及仅覆盖 100 种常见字体，扩展到更大字体库和更复杂设计仍需进一步研究。

---

## 452. Who Responds When the Driver Is Gone? A Framework for Human Intent Understanding

**arXiv ID:** 2607.04670 | [PDF](https://arxiv.org/pdf/2607.04670v1)

**作者:** Xuewen Luo `[一作]`, Chenxi Liu `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Intent2Drive 框架，实现自主车辆在无驾驶员情况下通过全方位理解乘客意图并进行一致规划

**💡 创新点**

将乘客意图建模为包含显式语言与隐式情感、身体状态、行为信号等多维度的潜在认知状态，采用 Theory‑of‑Mind 推理生成可供规划使用的目标

**🔧 技术方法**

使用大语言模型（Qwen3‑4B）进行 ToM 推理，生成 Latent Human State (LHS) 与 Human Intent Objective (HIO)，并在路径与轨迹层采用强化学习与扩散模型实现层级规划

**📊 数据集**

构建 Holistic Intent Dataset (HID)，包含 2,240 条真实+合成样本，涵盖显式意图、隐式线索、场景、LHS 与 HIO 注释

**📈 对比分析**

在 nuPlan 关闭环基准上与多种传统与 LLM 辅助规划器比较，Intent2Drive 在非反应场景下接近最强方法（93.02），在反应场景下保持竞争力（84.05），并在 LHS 与 HIO 预测任务上显著提升准确率（最高达 90% 以上）

**⚠️ 局限性**

缺乏多模态输入（如面部表情、语音、姿态）与更细粒度的意图表征；HIO 仍为简单的行程与驾驶目标，难以覆盖复杂个性化需求

---

## 453. Targeted Structure Completion for Sparse-View 3D Reconstruction in Autonomous Driving

**arXiv ID:** 2607.04661 | [PDF](https://arxiv.org/pdf/2607.04661v1)

**作者:** Guoqing Wang `[一作]` (Shanghai Jiao Tong University), Chao Ma `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 34588 | [OpenAlex ID](https://openalex.org/A5025545087)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对稀疏视角的自动驾驶三维重建，提出了 FocusGS 框架，利用像素级高斯与几何不确定性子空间的目标结构补全相结合，显著减少高斯数量并提升重建质量。

**💡 创新点**

创新点在于将几何不确定性定位为三维不确定子空间（Geometric Ambiguity Manifold），并仅在该子空间内进行轻量级的结构补全，从而消除了传统全卷积体素方法的冗余计算。

**🔧 技术方法**

使用了 3D 高斯抛光（3D Gaussian Splatting）基础表示、深度梯度阈值化得到的二维不确定性掩模、稀疏采样的三维查询、三维稀疏卷积和跨视角可变形注意力等技术。

**📊 数据集**

在 nuScenes（ego‑centric）和 RealEstate10K（scene‑centric）两个基准数据集上进行实验，尤其针对自动驾驶场景进行稀疏视角重建。

**📈 对比分析**

与 Omni‑Scene、MVSplat、pixelSplat 等基线比较，FocusGS 在 nuScenes 上实现了 24.65 dB 的 PSNR、0.754 的 SSIM、0.220 的 LPIPS，PCC 0.837，同时将高斯总数降低约 74% 与渲染延迟降低约 34%；在 RealEstate10K 上亦达到最高的 26.32 dB PSNR、0.872 SSIM、0.123 LPIPS，并可与 MVSplat 进一步融合取得最佳效果。

**⚠️ 局限性**

局限性在于对几何不确定性子空间的定位高度依赖估计质量，极端动态物体、极端光照或雨天等恶劣天气下的镜头噪声会导致不确定性掩模误判，进而影响结构补全与重建质量。

---

## 454. KAM-WM: Kinematic Affordance Maps from Latent World Models for Robot Manipulation

**arXiv ID:** 2607.04652 | [PDF](https://arxiv.org/pdf/2607.04652v1)

**作者:** Xinyu Shao `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**通讯引用:** 65418 | [OpenAlex ID](https://openalex.org/A5100602288)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 KAM‑WM 框架，利用冻结的 Flow Matching 图像‑到‑视频模型一次性查询得到的单步潜在速度场作为第一阶视觉先验，用于低数据下的机器人操纵任务。

**💡 创新点**

创新点在于：不需要未来帧生成或模型微调，直接读取冻结视频模型的单步潜在速度场，提供运动方向信息，从而提升低样本学习的成功率。

**🔧 技术方法**

技术包括：Wan2.2‑TI2V‑5B Flow Matching 图像‑到‑视频模型、Perceiver 交叉注意力压缩、1D U‑Net 扩散策略以及 FiLM 语言调制。

**📊 数据集**

数据集：LIBERO（40 个任务）和 RoboTwin 2.0（50 个双臂接触任务）。

**📈 对比分析**

与零阶掩码先验、DP、OpenVLA、RoboTwin 排行榜基线进行对比；在 LIBERO 上平均成功率 90.6%（高于 DP 的 72.4% 和 OpenVLA 的 76.5%），在 RoboTwin Easy 上 65.7%（高于 DP3 的 55.2%），Hard 上 22.4%（高于 π₀ 的 16.3%）。

**⚠️ 局限性**

主要限制包括：一次性提取 KAM 使其在长周期或场景变换时容易过时、对 Flow Matching 参数化依赖、仅提供粗略运动先验而非精确 3D 规划、实验仅在仿真环境中验证、缺乏多随机种子评估等。

---

## 455. Learning Flexible Generalization in Video Quality Assessment by Bringing Device and Viewing Condition Distributions

**arXiv ID:** 2607.04643 | [PDF](https://arxiv.org/pdf/2607.04643v1)

**作者:** Nikolay Safonov `[一作]` (Moscow State University), Dmitriy S. Vatolin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

收集并发布了一个涵盖300多款Android设备、包含HDR/SDR编码、并记录屏幕尺寸、亮度、环境光等元数据的250k对比式主观VQA数据集，并基于Blade–Chest模型和条件池设计了设备与视角条件自适应框架。

**💡 创新点**

创新点在于①大规模多设备视角条件收集与元数据标注，②利用Blade–Chest模型将视角条件直接嵌入偏好聚合，③通过条件池对未观测的设备状态进行模拟，从而在无监督条件下实现VQA指标的泛化与自适应。

**🔧 技术方法**

主要技术包括基于Bradley–Terry/EM求解的Blade–Chest聚合、全连接神经网络条件适配模块、Kendall秩相关评估、条件池随机采样与模拟、以及对传统与学习型VQA指标的迁移学习。

**📊 数据集**

使用了自建的多屏VQA数据集（250k对比评估、300+设备、HEVC/VVC/AV1压缩视频）以及公开的参考视频源（Vimeo高比特率），并结合HDR/SDR混合内容。

**📈 对比分析**

通过Kendall秩相关对原始和自适应后VQA指标进行比较，实验显示自适应模块普遍提升1-3个百分点的相关度，尤其在未见设备上表现出良好的跨设备泛化；但受主观噪声限制，整体提升有限。

**⚠️ 局限性**

主要局限包括①未对原始VQA模型进行再训练，限制了自适应潜能；②自适应模块需针对每设备单独推理，规模化成本高；③数据集中仅包含Android设备，可能存在采样偏差；④未充分验证在极端环境（极大屏幕或非HDR）下的可靠性。

---

## 456. PixelPilot: Scalable Vision-Language-Action Models for End-to-End Autonomous Driving

**arXiv ID:** 2607.04637 | [PDF](https://arxiv.org/pdf/2607.04637v1)

**作者:** Pin Tang `[一作]` (Shanghai Jiao Tong University), Chao Ma `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 34588 | [OpenAlex ID](https://openalex.org/A5025545087)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出PixelPilot，一种在图像平面上进行视觉规划、然后再进行确定性3D提升的端到端驾驶VLA模型。

**💡 创新点**

通过将规划与提升解耦，解决了传统2D→3D预测中对摄像头参数的依赖和对车速的短视，且引入知识灌输的两阶段学习与中间dense奖励，强化感知→推理→动作→规划的因果链。

**🔧 技术方法**

利用大规模视觉语言模型Qwen2.5-VL为基础，采用多任务监督微调（感知、元动作、路径预测、盒子推理）与Group Relative Policy Optimization（GRPO）强化学习，并使用稀疏/中间奖励、IOU、F1、L1/L2、PDMS等指标。

**📊 数据集**

在nuScenes、Waymo Open、Bench2Drive等多源、跨摄像头配置的数据集上训练与评估。

**📈 对比分析**

与多种基线（DriveVLM、EMMA、OmniDrive、OpenDriveVLA等）比较，PixelPilot在nuScenes开放循环L2误差平均0.30m、碰撞率和交叉率最低；在Bench2Drive闭环测试中取得最高驾驶分数和成功率，且在Waymo零样本迁移中保持较低误差。

**⚠️ 局限性**

受限于需要已校准的相机参数与精准的ego运动估计，对非平面多层道路、严重遮挡或未校准传感器仍表现不佳，且提升阶段仍需手动提供相机姿态。

---

## 457. Enhancing Large Multimodal Models in Key Information Extraction via Scene-Aware Document Synthesis

**arXiv ID:** 2607.04636 | [PDF](https://arxiv.org/pdf/2607.04636v1)

**作者:** Zhipeng Xu `[一作]` (Alibaba Group), Zhao Li `[通讯]` (Zhejiang University)

**通讯引用:** 30463 | [OpenAlex ID](https://openalex.org/A5006025566)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种场景感知的文档合成框架，用示例驱动与错误驱动生成技术自动生成文档-模式-注释三元组，提升小型多模态模型在关键信息抽取（KIE）任务上的性能。

**💡 创新点**

创新点在于：①无需手工模板，利用少量示例学习类别内容与布局模式；②引入错误驱动生成，将真实失败案例转化为难度更高的训练样本；③通过多代理系统实现内容、布局、结构的协同生成。

**🔧 技术方法**

使用了多模态大模型（Qwen3-VL-Max、Qwen3-VL-Plus 等）进行内容与布局感知与生成，结合 LLM 进行文本重写与注释更新，并采用 Blender 生成真实噪声模拟。

**📊 数据集**

合成数据集为 1M 条示例驱动与错误驱动生成的文档，验证集使用 UniKIE 基准（涵盖多种文档类型与字段定义）。

**📈 对比分析**

通过在 2B、4B 版基础模型上微调并与 MiniCPM、InternVL、GLM 等 on‑device LMM 及 GPT‑4o、Claude‑Sonnet 等 on‑server LMM 进行比较，实验显示在 constrained‑category 与 open‑category 设置下，模型在字段级 F1 分数上提升 10–20%，并在 on‑device LMM 中排名第一，逼近部分 on‑server 系统。

**⚠️ 局限性**

局限性：目前仍难处理混合印刷与手写文本，合成数据无法充分模拟手写文字及其视觉变异。

---

## 458. Can LLMs Really Recover Microservice Failures? A Recovery-Aware Evaluation of Diagnosis-to-Action Reasoning

**arXiv ID:** 2607.04623 | [PDF](https://arxiv.org/pdf/2607.04623v1)

**作者:** Jiaxing Qi `[一作]` (Beihang University), Depei Qian `[通讯]` (Beihang University)

**通讯引用:** 2431 | [OpenAlex ID](https://openalex.org/A5079362609)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 R2Act 框架，用于评估微服务故障诊断后转化为有效恢复动作的能力，并提供了包含 302 条经过质量审计的 Kubernetes 故障实例的基准数据集。

**💡 创新点**

引入了诊断到行动（diagnosis‑to‑action）评估维度，定义了事件方案、操作空间和恢复有效性指标，揭示诊断准确性与恢复可行性之间显著差距，并提供了离线与在线重放验证机制。

**🔧 技术方法**

结合多模态日志、事件、指标和 Kubernetes 状态，使用 RAG 语义检索增强的 LLM（如 Qwen‑RAG、Kimi‑RAG 等）、规则/监督方法和现有 RCA 工具；评估器采用离线验证和在真实集群上重放执行。

**📊 数据集**

302 条质量审计后的 Kubernetes 故障实例，涵盖六种服务角色、八类故障，包含同步日志、事件、指标、Pod 状态、根因标签、可行动作空间以及合法/非法恢复计划。

**📈 对比分析**

与多种基线（关键字、规则、监督、PyRCA、RCAEval、LogFormer、OneLog）以及多款 LLM（LogPrompt、LogRAG、OpenRCA、Qwen‑RAG 等）在诊断准确率和恢复有效性上进行对比；虽然 RAG LLM 在诊断上达到 99% 以上，但其恢复有效率仅为 37%–60%，验证了诊断-行动差距。

**⚠️ 局限性**

基准仅覆盖单一云原生系统与有限的服务图与故障类别，缺乏多生产环境、多级依赖关系和组织特定恢复策略；恢复空间仅包含七种操作，可能不足以覆盖更复杂的实际场景。

---

## 459. G2VD: Generalizable AI-Generated Video Detection via Counterfactual Intervention and Causal Disentanglement

**arXiv ID:** 2607.04607 | [PDF](https://arxiv.org/pdf/2607.04607v1)

**作者:** Meng Du `[一作]` (Information Engineering University), Shuxin Liu `[通讯]` (Information Engineering University)

**通讯引用:** 918 | [OpenAlex ID](https://openalex.org/A5100775924)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于因果干预与因果解耦的 AI 生成视频检测框架 G2VD，解决跨生成器泛化问题。

**💡 创新点**

创新点在于：①引入对抗性 VAE 干预产生控制的反事实样本，弱化生成器特定的伪特征；②设计双分支因果解耦分类器并利用 HSIC 约束两类特征独立，提升对生成器内在痕迹的捕获。

**🔧 技术方法**

使用的技术包括：变分自编码器 (VAE) 生成反事实样本、频域与像素域对齐、双分支 MLP 分类器、HSIC 互信息约束、以及多种视频骨干（CLIP、XCLIP、DeMamba）。

**📊 数据集**

实验使用四大公开数据集：GenVidBench、GenVideo、GVD 和 GVF。

**📈 对比分析**

与 12 类基线（含多种骨干和专门的伪造检测器）对比，G2VD 在 GenVidBench 的跨域平均准确率达 91.92%（AUC 0.95），在 GenVideo、GVD、GVF 上亦实现最高或第二高准确率，且仅使用原始训练数据的 10%。

**⚠️ 局限性**

局限性包括：在强 JPEG 压缩或显著高斯模糊下鲁棒性下降，易出现对真实视频的误报（如压缩伪影、运动噪声等）。

---

## 460. A Physics-Regulated Neural Framework for Learning 3D Grain Growth Dynamics

**arXiv ID:** 2607.04680 | [PDF](https://arxiv.org/pdf/2607.04680v1)

**作者:** Zhihui Tian `[一作]` (University of Florida), Joel B. Harley `[通讯]` (University of Florida)

**通讯引用:** 2394 | [OpenAlex ID](https://openalex.org/A5008672493)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出3D-PRIMME，一种物理调节的机器学习框架，利用局部窗口接口表示学习三维晶粒生长的演化规则，并在大尺度域内进行高效预测。

**💡 创新点**

创新点在于仅用一对连续时间步的极少训练数据即可学习到跨尺度、跨时间的局部演化规则；去掉显式正则化，采用窗口化接口表示实现稳定性；并能自动捕捉倾向依赖的晶粒生长。

**🔧 技术方法**

采用基于局部观察窗口和动作窗口的接口-点表示，使用深度神经网络预测局部状态更新，利用自回归推理实现大规模并行；实现代码基于PyTorch。

**📊 数据集**

训练数据来源于mode‑filter（MF）随机模型生成的200条100³网格、512粒子序列（共200×100时间步）以及其对应的倾向依赖（aniso‑MF）数据集。

**📈 对比分析**

与MF参考模型对比，评估⟨r⟩²线性增长、平均面数、体素级精度等指标；在256³、512³、1024³乃至550k粒子域内保持2.85%以内的速率误差，体素准确率随时间下降但保持高于80%，并在不同窗口尺寸、训练样本量和随机性下均表现稳健。

**⚠️ 局限性**

局限性包括需要手工设定观察/动作窗口尺寸，可能不适用于不同微观尺度；长时间自回归推理会出现轨迹漂移；尚未引入晶体取向信息，难以完整处理真实实验中的倾向/取向依赖；验证主要基于MF模拟数据，需进一步验证实验数据。

---

## 461. Adaptive Space-efficient Collectives for Dynamic and Unstructured Sparsity on GPU Platforms

**arXiv ID:** 2607.04676 | [PDF](https://arxiv.org/pdf/2607.04676v1)

**作者:** Lannie Dalton Hough `[一作]` (University of Maryland), Abhinav Bhatele `[通讯]` (University of Maryland)

**通讯引用:** 4066 | [OpenAlex ID](https://openalex.org/A5081506338)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在GPU平台上实现了适应动态、无结构稀疏性的高效稀疏集体通信（all‑gather、reduce‑scatter、all‑reduce）

**💡 创新点**

提出了低开销、可并行压缩解压的位向量稀疏格式Pici，并设计了自适应阈值算法以应对稀疏性填充和拓扑差异

**🔧 技术方法**

使用CUDA核实现Pici的压缩/解压；在NCCLX库基础上扩展SpCCL，采用环路和树算法，并进行通道数、NVLink读取优化

**📊 数据集**

在Perlmutter集群上使用梯度剪枝的分布式数据并行训练（1.5B GPT‑2 XL、3.3B Starcoder2‑3B），以及BookCorpus数据集

**📈 对比分析**

与密集型NCCL/NCCLX和稀疏基准SparCML比较，最高稀疏率99%时all‑gather、reduce‑scatter、all‑reduce分别获得5.25×、2.5×、2.66×的加速，整体训练速度提升达26%

**⚠️ 局限性**

当稀疏率降低或GPU数量增大导致稀疏性填充时加速下降；需要手动调节阈值；仅在NVIDIA A100 GPU上验证，未评估其它硬件

---

## 462. Video Generation Models Are Inherent Lighting Estimators

**arXiv ID:** 2607.04674 | [PDF](https://arxiv.org/pdf/2607.04674v1)

**作者:** Ziqi Cai `[一作]` (Peking University), Boxin Shi `[通讯]` (Peking University)

**通讯引用:** 8491 | [OpenAlex ID](https://openalex.org/A5038326097)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 V‑LITE 框架，通过在视频中插入 chrome 球并让视频扩散模型填充该区域，直接生成动态 HDR 环境图，实现单视频光照估计。

**💡 创新点**

创新点包括：将光照估计重新定义为视频填充任务，充分利用视频扩散模型的内在光照先验；设计了 HDR‑aware VAE 与 LoRA 微调策略以兼容 HDR 域；构建了混合数据集 V‑LITESet，结合 tonemapped HDR 视频与真实 HDR 图像进行训练。

**🔧 技术方法**

使用了 Wan 2.1 视频扩散模型、HDR‑aware VAE（log 域编码/解码、可学习的 tone‑mapping 适配器）、LoRA 细调、条件适配器、光照探针（chrome 球）填充、HDR 逆变换以及 equirectangular 变换。

**📊 数据集**

使用了 V‑LITESet（约 8K HDR 视频 + 800 HDR 静态图，共 648K 帧）以及公开基准 Editable Indoor 与 EnvMapNet 进行评估。

**📈 对比分析**

与 DiffusionLight、DiffusionLightTurbo、StyleLight 等方法在 Editable Indoor（MSE、SI‑MSE、AER、LS）和 EnvMapNet（AED、AS）等指标上比较，V‑LITE 在精度、光照稳定性和计算速度上均优于对比方法，尤其在实时性能方面显著提升。

**⚠️ 局限性**

局限性：对极端或不常见的光照分布敏感，可能产生错误的环境图；长序列的时序稳定性受限；训练数据缺乏真实 HDR 视频，导致在某些场景下产生视觉失真或色彩偏差。

---

## 463. Governed Caste Reassignment in Heterogeneous Swarms: An Asymmetric-Trust Protocol with Audited Operator Countersignature

**arXiv ID:** 2607.04634 | [PDF](https://arxiv.org/pdf/2607.04634v1)

**作者:** Xue Qin `[一作]` (Harbin Institute of Technology), Zhijun Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 25093 | [OpenAlex ID](https://openalex.org/A5100450024)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种针对异构机器人群体的可治理种姓重新分配协议，定义了自动收紧、受限放松与操作员专属三条授权路径，并提供可离线验证的因果链审计记录。

**💡 创新点**

创新点在于将单机人格变换门迁移到群体层面，结合权限格与异构信任门，利用 Ed25519 签名、Merkle 递归审计与拜占庭共识，构成完整的审计与攻击防御体系。

**🔧 技术方法**

使用了 Ed25519 署名、Merkle 根哈希链、BLS/Quorum 认证、拜占庭容错共识、SROS2 认证框架、离线审计与分布式复制原语。

**📊 数据集**

实验基于仿真数据，涵盖医院物流、工厂细胞、防御巡逻三种场景，每场景三种种姓；每台机器人最高 0.5Hz 触发重分配；亦设定了 10 台 TurtleBot4 的时延基准点；未使用真实机器人数据集。

**📈 对比分析**

通过与无治理、协调员重新分配、授权仅三种基线对比；测量自动收紧延迟在 6–18 ms 范围内，受限放松多为短路路径；四种攻击实验全部被拒绝；分布式复制层实现共识与叉子排除；整体操作延迟远低于 2 s 的事件间隔预算。

**⚠️ 局限性**

局限性包括：单机循环实现未覆盖多主机 WAN；假设可靠广播与静态成员，未验证网络异步与动态加入/离线；一签名-位点状态非持久化，崩溃后恢复可能产生 fork；未防御物理破坏、网络分区或授权者密钥泄露等更广泛威胁。

---

## 464. MRMS: A Multi-Resolution Memory Substrate for Long-Lived AI Agents

**arXiv ID:** 2607.04617 | [PDF](https://arxiv.org/pdf/2607.04617v1)

**作者:** Jizhizi Li `[一作]` (NxtLab Innovations), Amy Shi-Nash `[通讯]` (NxtLab Innovations)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个双轴记忆子系统（MRMS），用于管理长期代理的记忆，以实现可持续且可靠的个性化交互。

**💡 创新点**

创新点在于将记忆组织为表示轴（结构化记录、向量索引、图关系）与时间轴（短期、会话、长期）并引入同步不变式来控制记忆影响。

**🔧 技术方法**

采用结构化记录数据库、向量检索（嵌入索引）和有向图关系，配合写入、选择、合并与修订等五阶段流程，以及基于边缘条件的边界投影。

**📊 数据集**

使用人工合成的800个任务（涵盖延迟回忆、边界控制、来源分离、修订、陈旧抑制等）构建诊断基准；未使用公开真实数据集。

**📈 对比分析**

通过对比六种子系统（仅最近上下文、仅向量、结构+向量、加时间、完整MRMS）在诊断基准上的准确率，完整MRMS在总体准确率达98.8%，显著优于其他变体。

**⚠️ 局限性**

局限性包括：仅在模拟任务上验证；未与真实LLM代理集成；对嵌入更新、图重建等实际系统开销未评估；模型对复杂自然语言查询的适用性待进一步验证。

---

## 465. Governed Individuation: Cryptographically Decoupling an Agent's Learning from Its Authority

**arXiv ID:** 2607.04613 | [PDF](https://arxiv.org/pdf/2607.04613v1)

**作者:** Xue Qin `[一作]` (Harbin Institute of Technology), Zhijun Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 25093 | [OpenAlex ID](https://openalex.org/A5100450024)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为“governed individuation”的执行架构，利用冻结的身份摘要与基于语义效果的门控机制，确保学习型智能体在部署后不会超出运算权限。

**💡 创新点**

创新点在于：①把身份绑定到不可变的加密摘要，②用效果格而非名称或意图来判定动作合法性，③证明学习、技能扩展或自我生成的治理原则都无法在未签名更新下扩大权限；这些为可证明的运行时安全提供了全新的形式化基础。

**🔧 技术方法**

技术包括：加密哈希摘要、基于效果格的门控（semantic effect lattice）、可验证的效果抽象器、签名门控的权限升级、以及基于拒绝历史的治理原则诱导。

**📊 数据集**

数据集：ToolGym‑GI 工具使用基准（开放式代码修复任务）以及治理决策基准；实验模型使用三大开源大模型 Qwen2.5‑7B‑Instruct、Mistral‑7B‑Instruct‑v0.2 与 Phi‑3.5‑mini‑instruct。

**📈 对比分析**

比较方法：将带门控与不带门控、只加权学习、引入治理原则等不同治理策略在同一任务和种子上对比。结果显示：门控策略使执行的禁止效果率为 0；在最难的任务上，门控并不降低任务成功率；引入拒绝历史的治理原则能显著降低未见红线的禁止提议率，优于无记忆或随机记忆控制。

**⚠️ 局限性**

局限性：①需要一个针对特定动作空间的可证明效果抽象器，无法直接扩展到完全开放的动作空间；②治理原则诱导的泛化效果并不总是具体规则，更多是对被拒绝效果类型的广泛回避；③实验仅在 7B 级模型和软件级任务上验证，尚未验证更大模型或物理执行层。

---

## 466. The Double-edged Effect of Banning Generative AI on Online Question-and-Answer Communities: Evidence from Stack Exchange

**arXiv ID:** 2607.04601 | [PDF](https://arxiv.org/pdf/2607.04601v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 467. Beyond Compliance: A Large Scale Study on the Completeness and Consistency of the GitHub SBOMs

**arXiv ID:** 2607.04614 | [PDF](https://arxiv.org/pdf/2607.04614v1)

**作者:** Kawsar Ahmed Bhuiyan `[一作]` (Concordia University), Diego Elias Costa `[通讯]` (Concordia University)

**通讯引用:** 956 | [OpenAlex ID](https://openalex.org/A5023951345)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 GitHub 自动生成的 SBOM 进行大规模实证研究，选取 10 万份（每语言 1,000 份）开源仓库，评估其完整性、NTIA 合规性、版本与许可证信息覆盖率，并与 Syft、Trivy 与 Microsoft SBOM 工具生成的 SBOM 进行对比。

**💡 创新点**

首次在 10 种主流编程语言上系统分析 GitHub SBOM 的完整性与一致性，并深入探讨缺失元数据的根本原因；通过 PURL 与 OSV 查询实现漏洞检测评估；结合人工构建的真实依赖基线对工具的检测准确性进行验证。

**🔧 技术方法**

利用 GitHub REST API 生成 SBOM；使用 Syft、Trivy 和 Microsoft SBOM Tool 对同一仓库进行本地化 SBOM 生成；采用 SPDX 规范字段进行合规性检查；使用 Kruskal–Wallis、Dunn 的后验检验统计工具差异；通过 PURL 调用 OSV API 进行漏洞查询；构建 Python、Java、JavaScript 的依赖解析器作为真实基线；利用 API 采集许可证信息。

**📊 数据集**

包含 10,000 个 GitHub 仓库（C、C++、C#、Java、PHP、Python、JavaScript、Go、Swift、Rust），每种语言 1,000 个高星级、活跃贡献者的项目；数据集已发布于 DOI https://doi.org/10.5281/zenodo.18883005。

**📈 对比分析**

通过对组件计数、版本覆盖率、许可证覆盖率、PURL 覆盖率和漏洞计数的多维度比较，发现 GitHub SBOM 与 Microsoft SBOM 在多数语言上覆盖率最高、漏洞检测最充分；Syft 处于中等水平；Trivy 关注点最少；但所有工具在版本准确性、许可证准确性方面普遍偏低，尤其是缺失供应商信息导致的 NTIA 合规率为 0%。

**⚠️ 局限性**

研究仅覆盖 GitHub 开源项目，未考察私有或其他托管平台；仅选取 10 种语言，无法代表全部生态；GitHub SBOM 生成过程不公开，可能与本地化工具产生差异；多语言仓库的 polyglot 特性可能影响按语言的结论；复现受限于 GitHub API 只能获取当前状态 SBOM，历史状态不可再现。

---

## 468. RoboVista: Evaluating Vision Language Models for Diverse Robot Applications

**arXiv ID:** 2607.04610 | [PDF](https://arxiv.org/pdf/2607.04610v1)

**作者:** Shuangyu Xie `[一作]` (University of California, Berkeley), Ken Goldberg `[通讯]` (University of California, Berkeley)

**通讯引用:** 14505 | [OpenAlex ID](https://openalex.org/A5010019244)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Robot Question Answering（RQA）框架，并基于此构建了涵盖工业、农业、家用、手术、自动驾驶及公开机器人数据集的 RoboVista VQA 基准；该基准共包含 474 个多选问题，覆盖 39 种机器人任务。

**💡 创新点**

创新点在于：①将机器人系统的模块化决策点转化为统一的 VQA 表示，解决了传统端到端数据难以覆盖机器人复杂决策的问题；②通过专家人工标注保证视觉 grounding 与任务相关性；③将基准评估结果与实际机器人执行误差建立负相关性，验证基准的实用价值。

**🔧 技术方法**

采用多种 Vision‑Language Models（Qwen2.5‑VL、Qwen3‑VL、Robo2VLM‑ER、RoboBrain‑2.5、GPT‑4o、GPT‑5、Gemini 2.5 Pro）进行零样本评估，并在此基础上使用 Chain‑of‑Thought 与 In‑Context Learning 进行提示实验。

**📊 数据集**

数据集为 RoboVista（474 题，5 选项），视觉数据来源于六大机器人应用领域的公开数据集（工业制造、农业、家用、手术、自动驾驶及开放机器人数据集），每题均附有专家理由。

**📈 对比分析**

比较方法：零样本、CoT、ICL；在六个领域分别测算准确率，并与物理任务（双手抓取对齐、手术打结）中的误差相关。最佳闭源模型 Gemini 2.5 Pro 在整体上取得 56.5% 的准确率，开源 Qwen3‑235B 达到 51.3%；CoT 在感知类问题上下降约 10% 但在规划类问题上提升；ICL 整体导致准确率下降且校准误差升高。

**⚠️ 局限性**

限制：构建高质量 Robot‑VQA 实例需要大量人工专家审阅，难以实现完全自动化；基准目前仅覆盖视觉与语言模态，未充分考虑运动学、动力学等非视觉信息；缺乏对更大规模、更多机器人任务与硬件平台的覆盖。

---

## 469. LCPNet: Latent Consistent Proximal Unfolding Network for Infrared Small Target Detection

**arXiv ID:** 2607.04603 | [PDF](https://arxiv.org/pdf/2607.04603v1)

**作者:** Tianfang Zhang `[一作]` (Tsinghua University), Xiangyang Ji `[通讯]` (Tsinghua University)

**通讯引用:** 11552 | [OpenAlex ID](https://openalex.org/A5024401174)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于深度展开的低秩一致近端网络（LCPNet）用于红外小目标检测，利用潜在空间展开保持物理约束并提升目标分离效果。

**💡 创新点**

创新点包括在潜在域展开、直接使用一致近端更新实现变量自我演化、以及共享优化记忆（SOM）统一引导多分量更新，显著降低误报与提升检测精度。

**🔧 技术方法**

采用潜在低秩正则化、LCP求解器、组归一化与谱归一化的升级更新器、门控共享记忆模块以及基于深度展开的迭代网络。

**📊 数据集**

在四个公开红外小目标数据集（NUDT‑SIRST、IRSTD‑1K、SIRST、SIRST‑Aug）上进行训练与评估。

**📈 对比分析**

与HVS、优化、深度学习及其他展开方法对比，LCPNet在IoU、F1、P_d最高、F_a最低，AUC在三大数据集均优于同类方法，且参数量与推理速度处于折中位置。

**⚠️ 局限性**

主要限制为推理时延仍高，潜在域与记忆模块的计算开销需要进一步压缩，未来可探索自适应早停或轻量化更新器。

---

## 470. Displacement Preserving Relational Distillation for Robust Medical Segmentation

**arXiv ID:** 2607.04599 | [PDF](https://arxiv.org/pdf/2607.04599v1)

**作者:** Zhicheng Ding `[一作]` (Bowling Green State University), Qizhen Lan `[通讯]` (UTHealth Houston)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于位移保持关系蒸馏（DPRD）的轻量化3D医学图像分割框架。

**💡 创新点**

创新点在于使用批量位移向量与ROI感知掩码实现结构一致的关系蒸馏，避免传统激活对齐的背景稀释和尺度不匹配。

**🔧 技术方法**

采用了位移保留关系对齐、ROI‑aware特征掩码、跨阶段多尺度优化、Smooth L1关系损失以及基于nnU‑Net的实现。

**📊 数据集**

实验使用ISLES 2022（卒中病灶分割）和AMOS 2022（腹部多器官分割）两个公开数据集。

**📈 对比分析**

与Logits KD、FitNet、RKD、CIRKD等基准相比，DPRD在Dice、NSD和HD95等指标上均取得最高或相近性能，并显著降低了模型参数和FLOPs。

**⚠️ 局限性**

局限在于ROI掩码需依赖训练时的标注，限制了在弱监督或标注稀缺环境中的适用性。

---

## 471. Minimum Block Width for Universal Approximation by Residual Neural Networks with Inner Width One

**arXiv ID:** 2607.04597 | [PDF](https://arxiv.org/pdf/2607.04597v1)

**作者:** Qi Zhou `[一作]` (Huazhong University of Science and Technology), Xiao-Song Yang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 217066 | [OpenAlex ID](https://openalex.org/A5100437036)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究残差神经网络在内部宽度为1时的通用逼近性，推导出在L^p范数和一致范数下所需的最小块宽度，证明了残差分支宽度可极小化而不失逼近能力。

**💡 创新点**

首次给出内宽为1的残差网络在任意输入输出维度下的最小块宽度上界和下界，证明该宽度为max{d_x,d_y}；并在一致逼近时给出更精确的区间max{d_x,d_y}≤w_min≤min{d_x+d_y, max{2d_x+1,d_y}}，从而比先前结果更小；同时揭示残差网络与MLP在宽度限制下的根本区别。

**🔧 技术方法**

利用残差块的可分解结构、LeakyReLU、ReLU等激活函数的可逼近性，构造可逼近任意仿射变换和分段线性映射；结合微分同胚、光滑嵌入理论以及残差块与差分方程对应的连续动力系统，完成理论证明。

**📊 数据集**

无数据集，论文完全是理论推导，实验仅包含一个示例残差网络在[-2,2]^2上的数值逼近结果。

**📈 对比分析**

比较方法主要是数学证明与先前文献给出的上界/下界对比。结果显示：在L^p范数下，最小块宽度等于max{d_x,d_y}；在一致范数下，最小块宽度区间为max{d_x,d_y}至min{d_x+d_y, max{2d_x+1,d_y}}，优于此前的上界 d_x+d_y 或 2max{d_x,d_y}+1，且不再需要 d_x≥d_y 的限制。

**⚠️ 局限性**

局限性包括：仍未给出实现所需残差块数量（深度）的上界；对更广泛激活函数族的泛化性需要进一步研究；数值实验仅限于单一例子，缺乏大规模实验验证。

---

## 472. TORINO: Token Reduction via Interpretable Concept Overlap in Vision-Language Models

**arXiv ID:** 2607.04593 | [PDF](https://arxiv.org/pdf/2607.04593v1)

**作者:** Riccardo Renzulli `[一作]` (University of Turin), Van-Tam Nguyen `[通讯]` (Télécom Paris)

**通讯引用:** 1355 | [OpenAlex ID](https://openalex.org/A5101833923)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TORINO框架，通过稀疏自编码器(SAE)的概念重叠来对视觉语言模型(VLM)的视觉token进行自适应压缩；

**💡 创新点**

创新点在于利用可解释的单义概念稀疏特征聚类token并在每个概念组内做pruning或merging，既保持语义信息，又实现动态token数控制；

**🔧 技术方法**

核心技术为Matryoshka BatchTopK稀疏自编码器、概念重叠阈值(k,δ)的图连通分割，以及基于组内峰值激活或平均+对数缩放的token生成；

**📊 数据集**

主要在LLaVA-1.5-7B/13B模型上实验，使用ImageNet-1K训练的CLIP ViT-L/14的CLS激活做SAE训练，并评估于9个多模态基准（GQA、MMBench、MME、POPE、ScienceQA、TextVQA、VizWiz、MM‑Vet等）；

**📈 对比分析**

与Random、FOLDER、PruneSID、PruMerge等基线对比，TORINO在中等压缩（约62%-78%）下达到98.7%/97.1%相对性能，优于其它方法；在极端压缩（约92%）仍位居第二；同时延迟低于基线；

**⚠️ 局限性**

局限性包括SAE仅在CLS激活上训练，需对不同视觉编码器重新训练；极端压缩易丢失小目标；目前仅验证于LLaVA，缺乏跨backbone的泛化实验；

---

## 473. MARLIN: De Novo Molecular Structure Elucidation from Tandem Mass Spectra without a Ground-Truth Formula

**arXiv ID:** 2607.04774 | [PDF](https://arxiv.org/pdf/2607.04774v1)

**作者:** Xujun Che `[一作]` (University of North Carolina at Charlotte), Depeng Xu `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 1701 | [OpenAlex ID](https://openalex.org/A5100730268)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为MARLIN的全新de novo分子结构推断系统，能够仅利用质谱数据（不需要先验分子式）直接生成候选结构并按ppm质量一致性筛选。

**💡 创新点**

核心创新在于：① 使用质量-壳（mass‑shell）约束的块扩散解码器，在生成过程中仅依据测得的前体质量维持总质量一致，而不显式指定元素配额；② 对指纹（fingerprint）进行对称噪声腐蚀训练和条件多样化，以提升解码多样性；③ 通过自监督峰值编码器（DreaMS）实现无分子式的指纹预测。

**🔧 技术方法**

技术手段包括自监督谱编码器DreaMS、基于SAFE字符串的块扩散（block‑diffusion）生成模型、傅里叶特征映射的质量与同位素条件注入、对称指纹噪声训练、质量‑壳剪枝与EOS耦合、以及基于RDKit的ppm精度有效性门。

**📊 数据集**

使用公开的NPLIB1基准数据集（含多种分子质量范围的质谱数据）。

**📈 对比分析**

与先前依赖已知分子式的state‑of‑the‑art方法（如DiffMS、MS‑Anchor等）进行比较。MARLIN在无分子式设定下取得最高的Top‑1精确匹配率（约16.94%–19.18%），并在结构相似度（Tanimoto）和MCES距离方面优于同类方法；在更大分子（≥500 Da）和高质量精度（10 ppm）条件下也保持了相对稳定的性能。

**⚠️ 局限性**

局限性包括：① 对高分子质量（≥500 Da）仍然表现较弱；② 仍无法解决由于质谱信息不足导致的构型等价体差异；③ 需要较多的候选生成（384个）以获得多样性，推理时间相对较长；④ 对指纹预测质量高度依赖，若指纹错误会直接影响结构生成。

---

## 474. DriftST: One-Step Generative Inference of Spatial Transcriptomics from H\&E Histology

**arXiv ID:** 2607.04740 | [PDF](https://arxiv.org/pdf/2607.04740v1)

**作者:** Yuhang Yang `[一作]` (University of Science and Technology of China), Kai Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 33687 | [OpenAlex ID](https://openalex.org/A5100323904)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种统一的 DriftST 框架，用于从 H&E 画像单步预测空间转录组，兼顾细胞级和斑点级解析。

**💡 创新点**

创新点包括：将空间转录组建模为 Cellular Drifting 生成过程，实现单步推断；设计 STransformer 通过共表达注意力和基因残差门显式捕捉基因间依赖与差异重要性；以及实现一个可在多分辨率下通用的预测体系。

**🔧 技术方法**

采用了细胞漂移生成模型、STransformer 自注意力与共表达矩阵、基因残差门、双基础模型 CONCH 与 UNI2 的视觉特征提取、以及 ZINB 似然损失。

**📊 数据集**

使用了四个公开 ST 数据集：Xenium Breast Cancer、Xenium COAD（细胞级）、HER2ST、Kidney Visium（斑点级）。

**📈 对比分析**

与 BLEEP、M2ORT、TRIPLEX、STEM、GenAR、GHIST、sCellST 等基线进行对比，结果显示 DriftST 在 PCC、MSE、MAE 等指标上均为最佳，尤其在低变异基因和细胞级别任务中表现显著优于现有方法。

**⚠️ 局限性**

局限性在于只能预测训练时使用的基因面板，无法覆盖全转录组；跨患者、批次泛化仍受限，尤其在 Kidney Visium 数据中表现相对逊色。

---

## 475. Identifiability of Relational Queries in Multi-View Pretraining

**arXiv ID:** 2607.04735 | [PDF](https://arxiv.org/pdf/2607.04735v1)

**作者:** Ratan Bahadur Thapa `[一作]` (University of Stuttgart), Daniel Hernández `[通讯]` (University of Stuttgart)

**通讯引用:** 217 | [OpenAlex ID](https://openalex.org/A5073473252)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文定义了跨源集成中的查询可识别性（identifiability），并给出了多项理论与算法，用以判定查询是否在所有合法世界中答案一致；同时提出了两种实用工具 CheckCert（多项式时间判定）与 Greedy‑MinAug（近似最小增补），并在多种真实与合成数据集上验证其有效性。

**💡 创新点**

创新点在于：①将查询可识别性形式化为跨源接口的可能世界问题；②证明其判定可通过属性闭包实现，得到可执行的 CheckCert；③将最小增补问题映射为集合覆盖，并给出对数近似的 Greedy‑MinAug；④揭示多视角预训练在满足闭包条件时必定得到唯一答案，提供结构化错误下限和能力跳跃的定量描述。

**🔧 技术方法**

使用的技术包括：功能依赖（FD）与 Armstrong 规则的属性闭包、可能世界语义、约束闭合图、集合覆盖与贪心近似、信息理论（Fano不等式、Jensen–Shannon 散度）以及多视角预训练的对比学习与聚合网络（GNN‑OG、SetTransformer 等）。

**📊 数据集**

使用的数据集包括：合成 5 属性二进制世界、学术领域的 BibInteg 与 CrossKG‑DBLP、产品领域的 Amazon‑Google 与 WDC‑Product、餐饮领域的 Fodors‑Zagat；每个数据集提供了视图、接口法则和若干查询，覆盖可识别与不可识别两类。

**📈 对比分析**

比较方法：对照 CheckCert 与 Greedy‑MinAug 的理论时间复杂度，实测在 10^3 属性与 10^3 依赖的规模下均在毫秒级完成；在 ML 评估中，结构满足闭包的查询得到 1.0 准确率，而不可识别查询始终接近 0.5 错误下限；增补实验显示 Greedy‑MinAug 的平均近似比远低于 H_k 上界，实际增补量仅为最优解的 1.07–1.10 倍。

**⚠️ 局限性**

局限性：仅适用于连接与投影（CQs）和单元组（单元世界）查询；接口法则仅限 FD 形式，无法处理更一般的 egd、tgd 或递归约束；合成实验中多元组世界的验证有限，真实多元组世界的泛化尚待进一步研究。

---

## 476. LP-SFT: Local-Preserving Supervised Fine-Tuning via Multimodal Entropy Structure

**arXiv ID:** 2607.04733 | [PDF](https://arxiv.org/pdf/2607.04733v1)

**作者:** Yueyang Wang `[一作]` (Peking University), Jingyuan Zhang `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 460 | [OpenAlex ID](https://openalex.org/A5100639610)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对预训练语言模型在进行监督微调时造成的能力退化和多样性下降，提出一种本地保持监督微调（LP‑SFT）方法，利用冻结的基模型局部概率结构来指导微调；

**💡 创新点**

创新点在于：①利用基模型的多模态熵结构识别“平坦‑k”状态，动态决定每个位置需要保留的可行替代词集合；②去除监督目标词，避免与交叉熵冲突；③在该局部词集上进行局部归一化的KL损失，单独保持非目标词的相对偏好，从而在保持单样本准确率的同时保留采样多样性；

**🔧 技术方法**

技术包括：熵（Shannon、Rényi-2）分析、有效支持大小（N1、N2）估计、基于冻结模型的自适应候选集构造、局部归一化KL正则、两阶段训练（离线基模型预计算+在线微调），以及对比实验中使用的基准任务与指标；

**📊 数据集**

数据集涵盖三大领域：通用指令对齐（UltraFeedback 61K）、代码生成（Magicoder‑OSS‑Instruct‑75K）以及数学推理（NuminaMath‑CoT 100K+MATH‑500、AIME 2023‑2026），并使用多种预训练模型（Qwen3‑4B/14B、Llama‑3.1‑8B、Qwen3‑30B‑A3B）；

**📈 对比分析**

与交叉熵、DFT、EAFT、GEM、ASFT等基线相比，LP‑SFT在pass@1和pass@k指标上均取得最优或接近最优的平均表现，显著提升单样本准确率且不损失采样多样性，且在混合域和单域微调任务中对比表现最为稳定；

**⚠️ 局限性**

局限性包括：对基模型局部概率结构的依赖，若基模型分布不可靠则可能无效；需要预先离线计算基模型分布；对弱模型（如Llama‑3.1‑8B）在某些数学指标上不一定优于纯交叉熵；

---

## 477. Reference-Induced Consensus for Selective Posed-Reference Visual Localization

**arXiv ID:** 2607.04722 | [PDF](https://arxiv.org/pdf/2607.04722v1)

**作者:** Wonseok Kang `[一作]`, Tae-Wan Kim `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过已知的姿态参考图像，将查询图像的局部几何信息（姿态、深度、对应点）提升到全局地图坐标系，生成若干姿态假设，随后使用鲁棒一致性聚合得到最终位姿，并通过保持假设结构产生两种无真值依赖的可靠性评分，实现位置估计与拒绝判定的耦合。

**💡 创新点**

① 仅依赖姿态参考而不需要预先构建SfM点云或做查询-地图匹配；② 采用“参考诱导”方式在全局帧中生成姿态假设，保持假设结构并提取空间离散度与轨迹协方差两种互补可靠性指标；③ 在无场景训练的前提下实现鲁棒的一致性位姿估计，兼具定位精度与失败检测能力。

**🔧 技术方法**

使用冻结的多视角几何网络（VGGT）预测局部姿态、深度和轨迹；对查询与选定参考图像做一次前向推理；对每个参考生成全局地图坐标下的姿态假设；对旋转做加权余弦平均后鲁棒门限；对中心做Student‑t IRLS一致性聚合；基于轨迹协方差与空间离散度计算可靠性评分；最终通过联合阈值实现选择性定位。

**📊 数据集**

7‑Scenes、12‑Scenes、Cambridge Landmarks、NAVER Gangnam Station 四个公开基准，涵盖室内、室外、低纹理及大规模场景。

**📈 对比分析**

与 Reloc3r‑512、MASt3R、DUSt3R 等无SfM点云的前馈定位方法以及结构化定位基线 hloc 进行对比。该方法在室内基准 7‑Scenes、12‑Scenes 的严格成功率超过同类前馈方法；在 Cambridge 的旋转精度优于 Reloc3r‑512；在低纹理的 NAVER 站点在 0.25 m/2° 阈值下取得最高成功率。再加上联合可靠性评分，能显著降低严格风险和灾难性错误率，风险‑覆盖曲线优于随机或单一评分。

**⚠️ 局限性**

① 由于不使用SfM点云或PnP，室外翻译精度仍低于传统基于点云的结构化定位；② 对低纹理场景的协方差可行性不足（仅 47%/75% 可用），限制了部署覆盖率；③ 轨迹协方差是一阶近似，导致绝对尺度过度自信；④ 推理时间主要由冻结的 VGGT 前向推理决定，单张查询耗时 1.9–3.4 s，较传统 RANSAC/PnP 更昂贵；⑤ 需要已标定的姿态参考集合，若缺失则无法直接使用。

---

## 478. Probe-EM: Targeted Neuron Tracing via Training-Free Semantic Verification

**arXiv ID:** 2607.04696 | [PDF](https://arxiv.org/pdf/2607.04696v1)

**作者:** Liuyun Jiang `[一作]` (Chinese Academy of Sciences), Hua Han `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 485171 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种无训练的神经元追踪框架Probe-EM，结合骨架引导的启发式空间搜索（HSS）与基于NeuroSAM 2的维度感知语义验证（DASV）实现针对性、快速的神经元形态重建；

**💡 创新点**

创新点在于：①利用骨架几何先验实现局部化、种子驱动的探测，消除全局扫描的冗余；②采用维度感知语义验证，针对切片内和切片间断裂分别使用平面集成共识和轴向时空传播，做到零训练的强健连通推断；

**🔧 技术方法**

核心技术包括：骨架化和终端邻域搜索、拓扑异常剔除、NeuroSAM 2基础模型的双向多样本验证、轴向记忆扩展传播以及与Neuroglancer的交互式人机循环；

**📊 数据集**

实验使用完整的 88 TB 轴向分辨率 5×5×40 nm 的小鼠下丘脑SCN ssEM数据集，标注36条轴突束和12条含体细胞神经元；

**📈 对比分析**

与基于卷积、点云和多模态的监督学习基线对比，Probe‑EM 在轴突束和含体细胞追踪任务中实现了最高的 F1（0.595/0.544），并在人工校正中将时间平均降低33.4%，F1 提升6.8%；

**⚠️ 局限性**

局限性包括：对基础模型的依赖需保证模型在不同数据域的泛化能力；轴向传播窗口参数需要经验调优，过大或过小均可能导致误连或漏连；整体方法仍需在更大、多样化的数据集上进一步验证。

---

## 479. Hierarchical Scaffolding Enables Human-Like Cognitive Selectivity under Data Scarcity

**arXiv ID:** 2607.04709 | [PDF](https://arxiv.org/pdf/2607.04709v1)

**作者:** Juhyoung Park `[一作]` (Korea Advanced Institute of Science and Technology), Se-Bum Paik `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1981 | [OpenAlex ID](https://openalex.org/A5025954136)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于认知心理学的层次化学习框架 SCALA，用粗到细的层级监督引导模型在样本稀缺时快速学习视觉识别任务。

**💡 创新点**

创新点在于：①将层级结构作为学习的支架，而非仅作正则化；②通过聚合细类 logits 自动得到粗类预测，保持单一分类器；③在训练阶段按层级顺序逐步引入监督，促进先形成语义邻域再细化决策；④显著提升模型对干扰背景的选择性与表征几何结构。

**🔧 技术方法**

技术手段包括：ResNet-18 基础网络；按层级聚合 logits 生成粗层预测；交叉熵损失按阶段训练；使用 Grad‑CAM、t‑SNE、CKA、Calinski–Harabasz 与 Davies–Bouldin 等指标评估表征质量；对比 flat‑learning 训练流程。

**📊 数据集**

主要使用 CIFAR‑100 数据集（采用 10% 训练子集作为极端稀缺情况），并在多种子集与架构上验证一致性。

**📈 对比分析**

与传统 flat‑learning 同一网络、相同超参进行对比；SCALA 在 10% 训练集下的 meta‑、super‑ 与 fine‑级别测试准确率均显著提升，收敛速度提升 2–3 倍；表征层在 intra‑class 距离、CHI 与 DBI 指标上优于基线，且对未见类别的零样本与快速适应表现更好。

**⚠️ 局限性**

局限性在于：依赖先验的、语义连贯的层级结构，若层级信息噪声或不完整，优势可能减弱；当前验证范围主要是 CIFAR‑100，需进一步评估在更大规模或不同领域的适用性；且框架假设可获得层级标签，自动生成或部分隐藏层级仍是待解决的问题。

---

## 480. TubeLite: Lightweight Multi-Actor Spatio-Temporal Action Detection

**arXiv ID:** 2607.04684 | [PDF](https://arxiv.org/pdf/2607.04684v1)

**作者:** Ali Soltaninezhad `[一作]` (University of Victoria), Alexandra Branzan Albu `[通讯]` (University of Victoria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出TubeLite框架，实现轻量化的空间-时间动作检测，聚焦演员管线稳定与边界感知。

**💡 创新点**

创新点在于使用高斯ROI标记、演员级GRU递归、跨注意力上下文以及管道级平滑损失，避免光流和全局自注意。

**🔧 技术方法**

技术包括ConvNeXt-Tiny主干、CenterNet式检测、Gaussian ROI tokenization、GRU时间传播、跨注意力上下文、边界感知预测头和tube-aware regularization。

**📊 数据集**

使用UCF101-24和MultiSports两个公开STAD基准。

**📈 对比分析**

与基准Transformer、3D卷积和两流方法对比，TubeLite在UCF101-24视频mAP@0.5提升至71.7，MultiSports提升至14.2，且参数仅1.6千万、GFLOPs 62，速度超过270fps。

**⚠️ 局限性**

局限在于采用16帧窗口、固定演员槽、非实时预测，且对极端拥挤场景或长时序动作的建模不足。

---

## 481. Classification of $σ$-validity

**arXiv ID:** 2607.04685 | [PDF](https://arxiv.org/pdf/2607.04685v1)

**作者:** Eiji Yamada `[一作]` `[通讯]` (Institute of Science Tokyo), Eiji Yamada (Institute of Science Tokyo)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文对多代理知识论中的相信公共公告逻辑（BPAL）中的迭代公告序列σ的有效性（σ-validity）与可满足性（σ-satisfiability）进行系统分类，并给出对K45、单代理KD45、多代理S5三类框架的完整分类结果，修正并证明了之前提出的Conjecture 13的错误。

**💡 创新点**

创新点在于：1) 提出了对σ-validity的更精确的定义与等价类概念；2) 完整推导出三类框架下的σ-valid性等价类结构；3) 证明了若干关键的存在与非存在引理，为分类提供严谨的理论支撑；4) 明确指出多代理KD45的剩余未解决问题，为后续研究指明方向。

**🔧 技术方法**

使用的技术主要是模型论与证明技术：通过构造具体的Kripke模型、更新模型、子模型和类型论证，利用K45、KD45、S5的闭包性质以及欧几里得性、传递性等属性来证明各种引理；同时使用归纳与反证法证明存在性与非存在性。

**📊 数据集**

本文未使用实验数据或数据集，而是完全基于形式逻辑的理论推导与模型构造。

**📈 对比分析**

由于工作为理论性质的证明，不涉及算法实现或实验对比，因此不存在性能评估或实验结果。

**⚠️ 局限性**

局限性包括：1) 对多代理KD45（|G|≥2）的完整σ-valid性分类仍未完全解决，只给出了部分命题与未决猜想；2) 证明过程相对复杂，对模型构造的依赖使得进一步自动化或工具支持尚未实现；3) 对更一般的公共公告逻辑（如带有公共知识算子或非传递框架）的推广仍是未来工作。

---

## 482. SLAM: Structured and Localized Analytic Manifold Adaptation for Lifelong VPR

**arXiv ID:** 2607.04764 | [PDF](https://arxiv.org/pdf/2607.04764v1)

**作者:** Kenta Tsukahara `[一作]` (University of Fukui), Rai Hisada `[通讯]` (University of Fukui)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向终身视觉定位的无回放闭式递归学习框架SLAM

**💡 创新点**

将不确定性平滑、GMM局部映射和H∞鲁棒边界整合为单一解析递推公式

**🔧 技术方法**

使用Unscented变换、Gaussian Mixture Model分区、H∞估计以及Woodbury矩阵求逆的闭式ACIL

**📊 数据集**

在NCLT长期巡航数据上构建的10个任务、100类的模拟地理网格数据集

**📈 对比分析**

通过对八种配置的消融实验，U+G组合实现最高27.5%准确率，较基线提升4.7%，H∞组件提供可调的鲁棒性折中

**⚠️ 局限性**

仅适用于已知任务序列的离线评估，未验证在完全动态开放场景下的实时性能与泛化

---

## 483. A Large-Scale Sparse Multiobjective Optimization Algorithm Based on Optimal Performance Scores

**arXiv ID:** 2607.04765 | [PDF](https://arxiv.org/pdf/2607.04765v1)

**作者:** Jia-Lin Mai `[一作]` (South China Normal University), Jian Weng `[通讯]` (Guangzhou University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对大规模稀疏多目标优化问题的新算法SMOEA-OPS，旨在更精准地识别非零变量并提升收敛速度。

**💡 创新点**

创新点在于：①基于最优性能得分的初始化框架，可在多区间采样中自适应评估变量重要性；②为每个实数变量动态计算变异概率，依据其在区间内的标准差；③引入基于Pareto导向的正态分布再采样策略，兼顾探索与收敛。

**🔧 技术方法**

技术实现包括多区间非支配排序、K-means聚类构造掩码模板、Simulated Binary Crossover与多项式变异、以及自适应变异概率与再采样操作。

**📊 数据集**

使用八个基准稀疏多目标优化问题（SMOP1–SMOP8，变量数从100到5000，目标数为2）以及三类真实案例（稀疏信号重建SR1–SR4、0/1背包KP1–KP4、社群检测CD1–CD4）进行实验。

**📈 对比分析**

与SparseEA2、S-NSGA-II、MSKEA、TELSO、MGCEA等算法在IGD和HV指标上对比，SMOEA-OPS在多数实例上获得显著更低的IGD与更高的HV，尤其在高维稀疏场景下展现出更强的收敛与多样性。

**⚠️ 局限性**

局限性包括：①初始化阶段需多次区间采样，消耗一定评估资源；②需要手动设定NVal与SVal，缺乏自适应机制；③对极高维度问题仍受评估成本与运算时间限制。

---

## 484. Multi-Turn On-Policy Distillation with Prefix Replay

**arXiv ID:** 2607.04763 | [PDF](https://arxiv.org/pdf/2607.04763v1)

**作者:** Baohao Liao `[一作]` (Microsoft Research), Furu Wei `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种离线的 Replayed‑Prefix On‑Policy Distillation (ReOPD)，通过重放教师已记录的前缀并在当前步骤上让学生主动产生动作，实现了在不需要环境交互的情况下对多轮 Agentic 任务进行稠密教师监督的学生训练。

**💡 创新点**

核心创新点包括：① 将多轮 OPD 视为可靠性感知的前缀分布设计，平衡学生相关性与教师可靠性；② 通过一步衰减的权重/采样调度实现对前缀分布的可控逼近；③ 利用教师 RL 训练时产生的轨迹作为免费离线资源，显著降低训练成本与环境依赖。

**🔧 技术方法**

技术手段主要有：多轮 On‑Policy Distillation、教师监督的稠密目标（token‑级 KL）、前缀分布的权重重采样、步骤衰减调度、RL 生成的教师轨迹回放、教师可靠性误差的理论分析与两侧分布偏移的界定。

**📊 数据集**

在两个主要任务上进行实验：数学推理（Python‑Tool 环境，包含 AIME24/25、AMC23、Minerva、Olympiad、MATH500 等竞赛数据）和搜索问答（检索环境，包含 NQ、TriviaQA、PopQA、HotpotQA、2Wiki、Musique、Bamboogle 等数据集），并在多环境联合训练下验证其通用性。

**📈 对比分析**

与基线方法（Base、Cold Start、SFT、OPD、GRPO）相比，ReOPD 在数学推理任务中往往超过 OPD，尤其在教师‑学生能力差距较大时提升显著；在搜索 QA 任务中与 OPD 性能相当；相较于 OPD，ReOPD 在每一步训练中至少快 4 倍，且训练过程中不调用任何工具或环境交互，训练成本大幅下降。

**⚠️ 局限性**

主要局限包括：① 需要预先收集并存储教师轨迹；② 步骤衰减调度是粗粒度的可靠性近似，未能针对具体前缀的教师可靠性进行细粒度自适应；③ 论文的分析与实验主要集中在可用教师轨迹覆盖良好的环境，若轨迹覆盖不足或教师可靠性评估失效，ReOPD 的效果可能受限。

---

## 485. Trajectory-Anchor Optimization for Overconfident Thermal Visual Place Recognition: Zero-Leakage OOD Auditing and Kidnapped-Robot Recovery

**arXiv ID:** 2607.04745 | [PDF](https://arxiv.org/pdf/2607.04745v1)

**作者:** Zhiyuan Lu `[一作]`, Kanji Tanaka `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Trajectory‑Anchor Optimization (TAO)，一种将多假设追踪压缩为批量 SE(2) Procrustes 对齐并通过单次批量 SVD 计算的轻量级、确定性后端，用于热成像视觉定位中的 OOD 审计与全球重定位。

**💡 创新点**

创新点在于：① 用张量级向量化实现所有候选分支一次性批处理，消除 MHT 的树扩展与动态内存分配；② 通过闭式 SE(2) 解析式和速度一致性约束，构建可解释的 OOD 证据信号；③ 引入严格的零泄漏评估协议，系统性划分微尺度与宏尺度 OOD 行为。

**🔧 技术方法**

核心技术包括：Tensor‑level vectorization、批量 SVD、SE(2) Procrustes 解析式、速度一致性正则化、基于 SVD 的几何不确定性诊断、零泄漏的评估框架。

**📊 数据集**

主要使用数据集：STheReO‑KAIST 热成像数据集（早晚交叉时间序列）作为主实验；IRSLAM/NUFR KRI 日夜热成像数据集作为零迁移泛化验证；实验中还引用了 DINOv2 作为基准描述子。

**📈 对比分析**

与传统 MHT、单帧 Top‑1 分数门控等方法比较，TAO 在闭集检索上保持与前端同等性能；在宏尺度 OOD 评估中，TAO 的 FPR 低于 12%（相较于单帧门控可高达 32%），AUC 约 0.99；在微尺度（5 m）OOD 检测中，AUC 仅 0.57，表明仅在宏尺度能发挥优势。

**⚠️ 局限性**

局限性在于：① 对 5 m 以下的细粒度误差无法主动拒绝，仍需与精细跟踪器配合；② 仅提供被动几何约束，无法应对高度动态或结构性遮挡导致的 OOD；③ 需要先验的校准参数，且在不同环境中 FPR 仍会波动；④ 目前未实现主动概率校准，导致在极端噪声下仍有较高的误接收率。

---

## 486. When Does High-CFG Diffusion Inversion Fail? A Controlled Study of Prompt--Latent Interactions

**arXiv ID:** 2607.04731 | [PDF](https://arxiv.org/pdf/2607.04731v1)

**作者:** Yan Zeng `[一作]` (Tohoku University), Takayuki Okatani `[通讯]` (RIKEN AIP)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在已知真实轨迹的控制环境下，研究高CFG下扩散模型的逆向可逆性，分析恢复质量的影响因素。

**💡 创新点**

引入prompt-pressure量化生成轨迹偏离度，划分易/难/中间三类prompt恢复行为，并提出局部轨迹一致性干预（SkipInv）诊断高CFG逆向失败。

**🔧 技术方法**

使用固定点反向求解（FPI）、DDIM采样、prompt-pressure计算、语义/词义变换实验、轨迹一致性干预、CLIP相似度评估等技术。

**📊 数据集**

基于PIE-Bench共700条prompt与10个seed的生成对，生成512×512图像。

**📈 对比分析**

与AIDI-GS1/GS7、AIDI-GS7/GS7比较，采用Init‑Inv、Gen‑Rec、CLIP文本分数和编辑PSNR评价；SkipInv在高CFG一致性下显著提升Init‑Inv、Gen‑Rec和编辑PSNR（约22.3 dB，接近上界）。

**⚠️ 局限性**

仅在合成数据上验证，缺乏真实图像实验；未提供完整可行的全局逆向框架；干预参数选择经验性且需要进一步泛化。

---

## 487. RustMizan: A Compilable, Contamination-Aware Benchmarking Framework for Rust Vulnerabilities

**arXiv ID:** 2607.04729 | [PDF](https://arxiv.org/pdf/2607.04729v1)

**作者:** Tarek Elsayed `[一作]` (Simon Fraser University), Steven Y. Ko `[通讯]` (Simon Fraser University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 RustMizan 框架，提供多级可编译的 Rust CVE 变体和基于变异的基准，用 LLM 代理评估漏洞检测与定位。

**💡 创新点**

结合多任务评估、多级可编译变体与内置的污染缓解与鲁棒性测试，填补了现有基准只做二分类、无可编译代码及缺乏污染检测的空白。

**🔧 技术方法**

采用 ReAct 架构的 LLM 代理、Rust 编译器工具链（rustfmt、syn 等）、语义保持变异器、基于 JSON 的评估指标。

**📊 数据集**

42 个真实 Rust 内存安全 CVE，生成 173 个 crate/file/function 级可编译变体，配备 CWE、函数/行级标注。

**📈 对比分析**

对四大前沿模型（Claude Sonnet 4.6、GPT-5.4、Gemini 3.1 Pro、Qwen 3.6 Plus）进行 agentic 测试；检测准确率 56–65%，行级 F1 仅 17–23%，功能级上下文提升至约 40%，恶意变异导致 27% F1 降低。

**⚠️ 局限性**

数据量有限（42 CVE、173 变体），仅覆盖 Rust，变异覆盖不均，可能导致污染缓解效果差异，且发布变异后仍存在被未来训练集捕获的风险。

---

## 488. SparseOcc++: Geometry-Aware Sparse Latent Representation for Semantic Occupancy Prediction

**arXiv ID:** 2607.04732 | [PDF](https://arxiv.org/pdf/2607.04732v1)

**作者:** Pin Tang `[一作]` (Shanghai Jiao Tong University), Chao Ma `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于几何感知的全稀疏表示 SparseOcc++，实现了场景补全与语义预测的解耦

**💡 创新点**

创新点在于利用稀疏锚点学习场景补全场 (SCF)，并通过正交分解与离散化学习以及几何引导传播高效完成三维结构

**🔧 技术方法**

核心技术包括 Lift‑Splat‑Shoot 视图变换、Deformable Cross Attention 生成锚点、稀疏卷积、稀疏 Transformer 头、稀疏 U‑Net 等

**📊 数据集**

在 SemanticKITTI 与 nuScenes‑Occupancy 两大基准数据集上进行训练与评估

**📈 对比分析**

与 Dense 和之前的 SparseOcc 对比，SparseOcc++ 在 IoU 上提升 2.3%，在 SemanticKITTI 上实现 24.7 FPS，速度是 OccFormer 的 5.9 倍，整体性能显著优于现有方法

**⚠️ 局限性**

局限性包括对非凸或复杂结构的几何过简化、远距离目标深度估计不准以及对稠密语义补全的依赖

---

## 489. What You See Is What You Get: Observation-Aligned Supervision for Chart-to-Code Generation

**arXiv ID:** 2607.04726 | [PDF](https://arxiv.org/pdf/2607.04726v1)

**作者:** Tianhao Niu `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文针对图表到代码生成任务中，提出将原始代码中不可观测的原始数据替换为从图像中可直接观测到的统计量（如箱线图的五数概括、直方图的箱边和权重、饼图的百分比），从而构建“Observation‑Aligned Supervision”。

**💡 创新点**

创新点在于识别并纠正了“latent‑observation mismatch”——即传统监督方式因使用不可观测的原始数据导致模型产生幻觉和过拟合，并通过对聚合型图表（箱线图、直方图、饼图）进行可观测量重写，显著提升了可观测值的恢复精度。

**🔧 技术方法**

技术主要包括：①数据重写框架（对boxplot、hist、pie分别提取统计量并改写绘图调用）；②在ChartCoder与ReChartPrompt-240K数据集上训练Vision‑Language模型（Qwen2.5‑VL、InternVL3）；③在ChartMimic与ChartX基准上进行可执行性与数据恢复评估。

**📊 数据集**

使用的数据集为ChartCoder与ReChartPrompt-240K两大图表‑代码对齐数据集，随后在ChartMimic和ChartX两套公开基准上进行实验。

**📈 对比分析**

评估采用执行后提取的数值指标（Histogram‑Value、BoxPlot‑Value、Pie‑F1）以及可执行率，结果显示Observation‑Aligned Supervision在所有模型（InternVL3‑8B/14B、Qwen2.5‑VL‑3B/7B）和两套基准上均实现了明显提升，尤其在both‑executable设置下提升更为稳定。

**⚠️ 局限性**

局限性在于仅针对箱线图、直方图和饼图的聚合型API进行重写，未覆盖如小提琴图、核密度图等需内部计算的绘图类型，且重写后代码可能与原始代码在视觉效果上仍有细微差异。

---

## 490. AI Agent Pull Requests on GitHub: Frequency, Structure, and Merge Conflict Rates

**arXiv ID:** 2607.04697 | [PDF](https://arxiv.org/pdf/2607.04697v1)

**作者:** George Xu `[一作]` (Harvard Medical School / Massachusetts General Hospital), Nithilan Karthik `[通讯]` (DevRev AI LLC)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 GitHub 上 AI 编码代理的并发拉取请求（PR）现象，并通过大规模三向合并模拟评估了它们的冲突率和冲突类型。

**💡 创新点**

首次量化了同一仓库中并发 AI PR 的出现频率、跨代理与同代理冲突比例，以及冲突的源代码结构特征；提出了基于 git 合并输出的冲突分类体系。

**🔧 技术方法**

使用了自动化的 merge‑replay pipeline（headless git 合并），利用 Wilson 区间计算置信区间，并结合冲突类型标签（content、modify/delete、add/add 等）进行统计。

**📊 数据集**

数据来源为 AIDev‑pop 数据集，包含 33,596 条 AI PR，分布于 2,807 个仓库，并抽样了 747 个并发 PR 对进行合并重放。

**📈 对比分析**

对比了 intra‑agent 与 cross‑agent 并发对的冲突率，结果显示同代理冲突率为 19.8%（95% CI [16.8%, 23.2%]），跨代理冲突率为 41.7%（95% CI [33.1%, 50.9%]），两者置信区间不重叠，证明跨代理冲突显著更高；同时展示了冲突占比的语言与仓库规模关联趋势。

**⚠️ 局限性**

主要局限包括：仅测量文本层冲突，未考虑构建或语义层冲突；样本偏向高活跃仓库，跨代理样本有限；数据集仅覆盖公开仓库和五个代理模型，可能不具备对私有企业代码库的泛化性；未对结果进行因果或统计显著性检验。

---

## 491. RSPO: Reward-Swap Policy Optimization for Multi-Turn LLM Agents

**arXiv ID:** 2607.04713 | [PDF](https://arxiv.org/pdf/2607.04713v1)

**作者:** Qiang Liu `[一作]` (Tencent YouTu Lab), Xing Sun `[通讯]` (Tencent YouTu Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种奖励交换策略优化（RSPO）框架，利用稠密过程奖励引导LLM多轮交互任务训练，最终以稀疏结果奖励更新策略；

**💡 创新点**

创新点在于：①循环训练机制将基于稠密奖励的Agent转化为探索Agent，再将其轨迹存入重放缓冲区供稀疏奖励更新；②奖励交换与基于奖励的采样、通用裁剪机制共同保证策略更新的多样性与一致性；

**🔧 技术方法**

核心技术包括：RL与LLM的结合、稠密过程奖励模型（使用MLP+tanh训练），重放缓冲区的奖励采样，RSPO的交替训练循环，通用裁剪（Generalized Clipping Mechanism），以及在PPO、GRPO、GiGPO等算法上的应用；

**📊 数据集**

实验数据集涵盖：ALFWorld（家庭任务模拟）和WebShop（购物交互），使用 Qwen2.5-1.5B/7B‑Instruct 作为基础模型，Llama‑3.2‑3B‑Instruct 用作稠密奖励模型；

**📈 对比分析**

与传统GRPO、PPO、GiGPO以及SPEAR组合基线比较，RSPO 在 ALFWorld 及 WebShop 上均实现显著提升（如 WebShop 上相对 GRPO 提升 5.5%，对 PPO 提升 12%），且在不同模型规模下保持一致性；

**⚠️ 局限性**

局限性：需依赖稠密奖励模型的质量；对大模型提升不如小模型明显；增加额外训练循环和缓冲区开销；在其他任务或更复杂环境中的适用性尚待验证。

---

## 492. Spatial Attention: Adapting Execution Horizons for Diffusion Policies via Observation Sensitivity

**arXiv ID:** 2607.04739 | [PDF](https://arxiv.org/pdf/2607.04739v1)

**作者:** Che-Sang Park `[一作]` (Seoul National University), Frank C. Park `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对基于扩散模型的学习示范策略，在执行动作块时动态调整执行长度，以提升对扰动的响应性与计算效率。

**💡 创新点**

提出“空间注意力”（Spatial Attention）这一度量，衡量策略对观测变化的敏感度，并证明在固定采样预算下，执行长度应随空间注意力递减；通过预测未来空间注意力实现自适应阈值控制。

**🔧 技术方法**

使用扩散模型（DDPM、DDIM、Consistency Policy）作为基准策略；引入基于贝叶斯规则的评分网络估计空间注意力；利用Transformer预测空间注意力序列；在视觉观测下采用VAE潜在空间来降低维度。

**📊 数据集**

Robomimic 仿真基准（Lift、Can、Square、Tool Hang）以及在同环境下加入人工扰动的改造版本；真实机器人实验使用Franka Research 3 搭配RealSense摄像头完成移动块抓取。

**📈 对比分析**

与固定执行长度的基准策略（DDPM、DDIM、3步/1步 Consistency Policy）在相同平均执行长度下对比；实验显示空间注意力方法在所有任务中提升 5%–19% 的成功率，且在扰动场景和真实机器人上成功率显著提升（如 DDIM+SA 0.92 对比 DDIM 0.42）。

**⚠️ 局限性**

局限性：只能适用于离散动作块的策略，无法直接应用于流式连续动作生成；对大型视觉语言动作模型的扩展尚未验证，潜在的计算开销与分布式部署仍需进一步研究。

---

## 493. HilEnT: Hilbert, Entropy Transformed Image Based Malware Detection

**arXiv ID:** 2607.04772 | [PDF](https://arxiv.org/pdf/2607.04772v1)

**作者:** Rahul Kale `[一作]` (ST Engineering), Vrizlynn L. L. Thing `[通讯]` (ST Engineering)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出HilEnT方法，将恶意软件二进制通过Hilbert曲线转换为灰度图，并利用与正负样本平均熵的阈值生成两张熵对比图，三张灰度图融合成RGB图像用于后续检测；

**💡 创新点**

创新点在于将空间填充的Hilbert曲线与类级熵阈值比较相结合，直接在图像生成阶段编码正负样本熵信息，形成三通道结构化特征，能让浅层CNN即可达到高性能；

**🔧 技术方法**

采用的技术包括：Hilbert曲线变换、熵阈值对比图生成、RGB图像融合、浅层CNN、SVM、MLP、HOG+PCA特征提取与降维、Siamese网络少样本学习；

**📊 数据集**

使用四个公开数据集：Dike、Michael Lester PE、Microsoft BIG 2015、以及自采集的5 MB Windows PE恶意样本；

**📈 对比分析**

与传统的Nataraj灰度可视化以及Hilbert曲线基准对比，并在多种CNN后端（AlexNet、VGG、ResNet、EfficientNet等）上评估，结果显示HilEnT在二分类中准确率可达99%且推理速度最快，若开启HOG+PCA可进一步加快推理；少样本学习也能在10%训练样本下保持接近全量训练的精度；

**⚠️ 局限性**

主要限制包括：仅评估Windows PE/OLE文件，文件大小上限5 MB且排除打包样本，缺乏动态特征融合，未系统评估对轻微注入或对抗修改的鲁棒性；未来工作计划扩展到其他可执行/文档格式、加入多模态特征、研究对抗鲁棒性。

---

## 494. Physics-Based Simulation of Contact-Induced Facial Wrinkling

**arXiv ID:** 2607.04768 | [PDF](https://arxiv.org/pdf/2607.04768v1)

**作者:** Juan Sebastian Montes Maestre `[一作]` (ETH Zürich), Bernhard Thomaszewski `[通讯]` (Meta Reality Labs Research)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对人脸皮肤在接触作用下的皱纹生成进行有限元仿真，使用高阶固体壳单元、可视化和弹性-粘弹性材料模型，并加入连续体层叠韧带约束。

**💡 创新点**

创新点在于提出连续体韧带（零长度Fung型弹簧）与空间变异的韧带刚度热图结合的皮肤附件模型，以及高阶prismatic固体壳单元和分层可变粘弹性材料，实现逼真高频皱纹。

**🔧 技术方法**

采用高阶prismatic固体壳有限元、通用Maxwell粘弹性模型、IMLS隐式碰撞与罚函数、摩擦损失势、零长度弹簧韧带、L‑BFGS拟牛顿求解器、精细高阶积分点。

**📊 数据集**

使用自行采集的人脸触摸视频数据（前额、颞部）进行定性对比，未使用公开大规模数据集。

**📈 对比分析**

通过与真实视频定性比较、参数消融实验（韧带刚度、层间刚度比、粘弹性）以及低阶与高阶单元的对比验证，L‑BFGS在相同节点数下比Newton快数倍；高阶单元能在有限时间内捕获预期高频皱纹，计算量仍显著大于低阶。

**⚠️ 局限性**

局部区域计算受限，未能全头仿真；单层同质化近似忽略层间滑移；韧带密度与位置为经验设定，缺乏身份特异化逆推与数据驱动方法。

---

## 495. Trust Region Policy Distillation

**arXiv ID:** 2607.04751 | [PDF](https://arxiv.org/pdf/2607.04751v1)

**作者:** Zhengpeng Xie `[一作]`, Mao Yang `[通讯]` (Microsoft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Trust Region Policy Distillation（TOP‑D），通过构造外部近端教师和内部信任域迭代，将原本不稳定的 On‑Policy Distillation（OPD）变为稳定且样本高效的训练框架；

**💡 创新点**

创新点在于：1）用概率插值构造可调的近端教师，显式限制奖励的下界，消除梯度方差爆炸；2）在内部引入信任域迭代实现离线重采样，提高样本利用率；3）给出闭环的理论保证，包括方差上界、全局收敛和单步误差递减；

**🔧 技术方法**

技术实现包括：对数概率比值的对数插值、RL‑style 重要性采样、token‑级优势归一化、KL 剪切、组化训练、以及多轮内部更新；

**📊 数据集**

实验使用的主要数据集为 DAPO‑Math‑17k 进行训练，验证集涵盖 AIME、AMC、MATH‑500 与 Olympiad 等数学推理竞赛；

**📈 对比分析**

与 OPD、RLVR、GRPO、DAPO 等基线比较，TOP‑D 在 Qwen3‑8B‑Base 学生模型上 AIME24 avg@32 精度提升 25.84% 以上，且在多规模模型与多基准上均显著优于传统 OPD 与 RLVR；

**⚠️ 局限性**

局限性包括：对插值系数 α 的选择仍需经验调优；在极端教师‑学生容量差距下仍可能出现性能退化；未在多模态或非数学任务上验证其普适性。

---

## 496. Direct Model State Migration for Elastic Training of Large Language Models

**arXiv ID:** 2607.04749 | [PDF](https://arxiv.org/pdf/2607.04749v1)

**作者:** Weijian Liu `[一作]` (SKLP, Institute of Computing Technology, Chinese Academy of Sciences), Weile Jia `[通讯]` (SKLP, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无需检查点的直接状态迁移框架 ETC，用于弹性混合并行 LLM 训练中的模型状态迁移。

**💡 创新点**

通过构造成本矩阵并求解最优迁移图，实现了最小数据移动的迁移方案，并通过虚拟 rank 及通信合并消除了资源碎片化。

**🔧 技术方法**

成本矩阵驱动的分配算法、Hungarian 求解、细粒度 GPU‑GPU 通信指令、元模型定位、通信合并与虚拟 rank 重新映射等技术。

**📊 数据集**

在 OpenWebText 上训练 GPT‑3 系列模型（32B、6.7B、64B 等）进行实验。

**📈 对比分析**

与 Megatron、Megatron‑Dist、EasyCkpt、Fastest‑Baseline、Tenplex 等基线在不同 PP/TP 迁移场景下比较，ETC 在迁移时间上比快基线快 2.33–6.37 倍、比 Tenplex 快 4.79–10.74 倍。

**⚠️ 局限性**

仍受网络带宽与 GPU 互连拓扑限制，且在极大规模多节点迁移时通信复杂度和调度开销可能显著增加。

---

## 497. Dashboard2Code: Evaluating Multimodal Models on Reconstructing Interactive Dashboards

**arXiv ID:** 2607.04727 | [PDF](https://arxiv.org/pdf/2607.04727v1)

**作者:** Tianhao Niu `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Dashboard2Code任务，让模型主动探索交互式仪表盘并生成完整代码

**💡 创新点**

首次把交互式仪表盘的主动探索与代码生成结合，提出自动化评估框架

**🔧 技术方法**

多模态大语言模型、Selenium环境、Plotly+Dash、DOM抽象、MM-React框架

**📊 数据集**

DashboardMimic基准：180个Plotly+Dash仪表盘-代码对，包含真实与LLM合成样本

**📈 对比分析**

与闭源模型（Gemini 3 Pro、GPT‑5.1、Claude 4.5）对比，闭源模型最高得分79.4；在复杂级别（L3）仅64.2，公开模型远低于闭源；评估框架与人工评估相关性0.78

**⚠️ 局限性**

仅覆盖Plotly+Dash框架，评估依赖LLM判定导致成本高，未涵盖其他主流仪表盘框架

---

## 498. Geometry-Aware Motion Latents for Learning Robust Manipulation Policies

**arXiv ID:** 2607.04714 | [PDF](https://arxiv.org/pdf/2607.04714v1)

**作者:** Yunchao Zhang `[一作]` (University of Hong Kong), Yanchao Yang `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

学习几何感知的运动潜在代码，通过预测三维点云随时间演化来抽取可迁移的机器人操控模式。

**💡 创新点**

将四维时空动态建模与自监督几何预测相结合，使用离散潜在码而非像素级别，显著提升跨视角、复杂环境的泛化能力。

**🔧 技术方法**

采用VQ‑VAE离散编码+3D点图VAE+四维扩散预测+条件扩散动作生成器，结合CLIP‑ResNet视觉编码和旋转位置编码的三维注意力。

**📊 数据集**

使用RLBench与CALVIN语义化仿真数据集，以及ALOHA实验室的RGB‑D真实演示数据。

**📈 对比分析**

与Act3D、3D Diffuser Actor、RV T2、SkillDiffuser等多种基线比较，RLBench成功率达84.7%，CALVIN平均任务链长3.60，真实环境成功率53.3%，相较竞争者提升约13%以上。

**⚠️ 局限性**

目前仅适用于刚体任务，对柔性物体、复杂动态环境的适用性有限，且在长序列推理与高维动作空间的可解释性上仍需提升。

---

## 499. FORGE: Research-Trajectory Hijacking Attacks on Deep Research Agents

**arXiv ID:** 2607.04718 | [PDF](https://arxiv.org/pdf/2607.04718v1)

**作者:** Yue Pan `[一作]` (Fudan University), Hongcheng Guo `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究深度学习研究代理在检索-规划-生成流程中的毒化攻击，并提出两级攻击框架FORGE

**💡 创新点**

1) 通过内部推理链和跨文档链协调，使单文档毒化成为可见多源证据；2) 引入PRISM量化报告层毒化严重度；3) 提出Root Query Anchoring防御，限制子任务漂移

**🔧 技术方法**

大语言模型（如gpt-researcher）、检索模型（BM25+向量相似度）、评估器Gemini-3.1-Flash-Lite、文档生成与人工审核

**📊 数据集**

5类话题（争议性、事实调查、历史发展、趋势预测、方法比较），每类5个查询共25个；网络与本地毒化实验，递归深度δ=1~4

**📈 对比分析**

与PoisonedRAG、AuthChain等基线对比；FORGE在网络模式下j=5时PRISM达26.4%，在10个查询子集中达38.5%；Root Query Anchoring将PRISM降至18.3%，并提升报告质量评分至0.6173

**⚠️ 局限性**

仅在gpt-researcher平台验证，网络仿真未涵盖真实Web因素；PRISM权重设计未经过人类认知校准；Root Query Anchoring仅防御规划层，检索层仍易受攻击

---

## 500. From Open Loop to Closed Loop: A Test-Time Iterative Optimization Framework for Reference-Consistent Image Generation

**arXiv ID:** 2607.04691 | [PDF](https://arxiv.org/pdf/2607.04691v1)

**作者:** Baixuan Zhao `[一作]` (Shanghai Jiao Tong University), Xiaohong Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无训练、可插拔的测试时迭代优化框架，将可控图像生成视为闭环动态跟踪，通过改进的 PID 控制器在潜在空间中迭代优化控制信号。

**💡 创新点**

创新点在于将控制理论的闭环反馈机制与扩散模型结合，利用预训练模块同时充当编码器和传感器，并通过改进的 PID 控制器实现无训练的闭环优化。

**🔧 技术方法**

技术手段包括扩散模型与可控适配器（如 ControlNet/ControlNext）、改进的 PID 控制器、潜在空间迭代优化，以及面部识别、姿态估计、深度估计等预训练视觉模型作为传感器。

**📊 数据集**

使用的数据集包括 Web100、CelebA300（身份保持任务）以及从 Hugging Face 下载的过滤后单人姿态/深度图像集（用于姿态和深度控制任务）。

**📈 对比分析**

与多种开环基线（PuLID、PMv2、InstantID、IPA、ControlNet/ControlNext）在最佳 N 采样（N=20 或 15）下进行对比，闭环方法在身份相似度上提升至 25.36%（CelebA300）并使姿态误差下降 27.71%、深度误差下降 28.50%，同时保持或提升语义一致性与感知质量。

**⚠️ 局限性**

局限性包括需手动调节 PID 参数、迭代导致的推理延迟与计算成本、以及对外部传感器精度的依赖，未来需实现自动化参数调优与收敛加速。

---

## 501. URSA: Chemistry-Aware Benchmark for Utilitarian Retrosynthesis Assessment

**arXiv ID:** 2607.04688 | [PDF](https://arxiv.org/pdf/2607.04688v1)

**作者:** Bogdan Zagribelnyy `[一作]` (Insilico Medicine AI Limited), Alex Zhavoronkov `[通讯]` (Insilico Medicine AI Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了URSA评估框架，用于从化学可行性角度对药物研发中的多步合成路径进行评估和比较。

**💡 创新点**

创新点在于将传统的可达性评估与Solv-2化学可行性判定结合，利用ChemCensor自动化的反应可行性评分，并通过RetroCast标准化多源路由和子树生成方法提升评估一致性。

**🔧 技术方法**

技术实现基于ChemCensor的先行数据库判定、RetroCast路由标准化、Solv-N层次验证、子树生成以及对LLM与传统CASP工具的集成。

**📊 数据集**

使用了ChemCensor专家标注的1000条反应、URSA的两套目标集（ChemCensor新分子集与100种临床候选药物）、含255,365种商业构建块的起始库存以及USPTO与Pistachio的合成先行库。

**📈 对比分析**

通过对Solv-0、Solv-1、Solv-2指标及平均ChemCensor分数CC*进行比较，结果显示传统CASP系统在ChemCensor集上Solv-2达32%（RetroChimera-MCTS）且在药物集上可达60%，远优于LLM规划器，且成本优势明显。

**⚠️ 局限性**

局限性包括仅评估至Solv-2（未覆盖实验可执行性）、对ChemCensor数据库的覆盖度和偏差敏感、仅考察每个目标最佳路由而不体现多样性，以及未纳入反应条件、产率等实验细节。

---

## 502. ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents

**arXiv ID:** 2607.04686 | [PDF](https://arxiv.org/pdf/2607.04686v1)

**作者:** Harsh Soni `[一作]` `[通讯]` (UC Berkeley), Harsh Soni (UC Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了ToolFailBench，一种诊断性基准，用于评估大型语言模型代理在单轮任务中对工具调用的正确性与使用。

**💡 创新点**

创新点在于将工具使用失败细分为四类（Tool‑Skip、Result‑Ignore、Output‑Fabrication、Unnecessary‑Tool‑Use），并通过规则分类器与两台LLM裁判相结合的投票机制实现可解释且鲁棒的标注；该方法揭示了聚合得分掩盖的不同失效模式。

**🔧 技术方法**

采用规则自动判别器、两位独立LLM裁判（不同提示结构）进行三者投票，并在统一的系统提示与输出格式下执行单次工具调用与答案生成。

**📊 数据集**

使用包含1,000个单轮任务的数据集，涵盖金融、医疗、法律、网络安全、房地产五大专业领域，其中750个是“工具必要陷阱”任务，250个为“对照”任务。

**📈 对比分析**

在19个主流模型上进行评估，最佳模型（Grok‑4.3）Clean Tool‑Use Rate达86.33%；尽管聚合得分相近，模型在工具使用方式上呈现明显差异，如Llama‑3.1的Always‑Call模式与Qwen系列的高控制任务准确率对比差距达89个百分点。

**⚠️ 局限性**

局限性包括：仅单轮任务，无法反映多轮工具链和状态更新；域覆盖有限，可能对其他专业领域适用性不足；标注依赖规则与LLM，仍可能存在盲点与训练重叠问题。

---

## 503. Do Vision-Language-Action Models Mean What They Say? On the Role of Faithfulness in Embodied Reasoning

**arXiv ID:** 2607.04681 | [PDF](https://arxiv.org/pdf/2607.04681v1)

**作者:** Matthew Foutter `[一作]` (Stanford University), Marco Pavone `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出Pinocchio框架，利用学习到的批评家对VLA模型的链式推理进行可信性评估，并将其作为强化学习奖励，实现了更可靠的行动推理。

**💡 创新点**

创新点在于将可观测性与步骤一致性作为可训练的行为可信度指标，并构造离散化的批评家（Pinocchio）作为稠密奖励，提升了推理与动作的一致性。

**🔧 技术方法**

主要技术包括Vision‑Language模型微调、Chain‑of‑Thought推理生成、基于大模型Gemini的标签化、学习的批评家（Pinocchio）以及GRPO强化学习。

**📊 数据集**

使用的主要数据集包括美国的驾驶示范数据集（训练集）、德国与美国的held‑out测试集以及通过图像修复工具生成的合成罕见危险场景。

**📈 对比分析**

与多种基线（ADE、VLM‑Judge、ADE‑Reason、ADE‑Swap）比较，Pinocchio在保持ADE≈5%误差的同时，推理可信度提升4%–18%，在罕见危险测试中对策响应率提升1.6倍。

**⚠️ 局限性**

局限性包括：可信度定义为必要但不充分的对称一致性，批评家依赖Gemini标签可能带来误差，且评估仅在离线开环环境中，未验证闭环表现。

---

## 504. AgenticPD: A Stage-Aware Agentic Framework for Physical Design QoR Optimization

**arXiv ID:** 2607.04758 | [PDF](https://arxiv.org/pdf/2607.04758v1)

**作者:** Shuo Ren `[一作]`, Tsung-Yi Ho `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种基于阶段感知的多代理框架AgenticPD，用于物理设计质量优化

**💡 创新点**

创新点在于将物理设计流程按阶段拆分，利用Judge Agent进行树形搜索与分支，Stage Agent在各自阶段做局部决策，能够复用中间检查点并直接观察后路由后的真实QoR

**🔧 技术方法**

采用了Agentic AI与多代理协作技术，结合OpenROAD工具链和LLM（DeepSeek‑V3.2等）生成参数、执行工具并记录反馈

**📊 数据集**

在AES、ibex、JPEG三个设计上，分别在SkyWater 130 nm和ASAP 7 nm技术节点进行实验

**📈 对比分析**

与平面化LLM、AutoTuner和ORFS-Agent等基线相比，AgenticPD在后路由时序闭合方面表现最佳，整体Post‑route WNS提升显著，同时功耗和面积保持竞争力

**⚠️ 局限性**

局限在于对LLM依赖较大，且当前仅针对四个标准阶段，未覆盖更细粒度的子流程或更大规模芯片的实验验证

---

## 505. Continual Model Merging with Test-Time Adaptation for Whole-Slide Image Analysis

**arXiv ID:** 2607.04755 | [PDF](https://arxiv.org/pdf/2607.04755v1)

**作者:** Duc-Thanh Le `[一作]` (University of Information Technology), Khang Nguyen `[通讯]` (University of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对连续的全切片图像分类，系统评估了基于测试时自适应的模型合并方法，并将其映射到无示例回放的持续学习框架中。

**💡 创新点**

创新点在于首次将 AdaMerging、AdaRank 与 Hi‑Vec 等测试时自适应合并策略应用于持续学习，证明其能够在保持历史知识的同时应对分布漂移，并揭示其对任务顺序的敏感性。

**🔧 技术方法**

技术上使用了 TITAN 视觉编码器提取病理补丁特征、CLAM 分割+采样、Transformer 聚合器，以及通过熵最小化进行的任务向量、奇异值掩码和层级向量的自适应调整。

**📊 数据集**

实验数据来自六个 TCGA 病例子群：BRCA、RCC、NSCLC、ESCA、TGCT 与 CESC，涵盖从常见到稀缺且类别不平衡的癌症亚型。

**📈 对比分析**

与传统的 LwF、EWC、DER++ 回放以及无示例的精细调优/全监督训练进行对比，在 CLASS‑IL 下 AdaMerging 与 Hi‑Vec 的 bACC 提升约 7–8%，在 TASK‑IL 下同样优于基线，且显著降低遗忘量，但在任务顺序逆转时性能下降；DER++ 在最终准确率和遗忘率上仍略优。

**⚠️ 局限性**

主要限制是仅依据当前未标记测试分布进行自适应，导致对近期任务过度偏向并加剧知识衰退；缺乏显式的历史子空间保护，且测试时自适应的计算开销不小。

---

## 506. MergeSurv: Merging-Based Continual Learning for Survival Analysis on Whole-Slide Images

**arXiv ID:** 2607.04747 | [PDF](https://arxiv.org/pdf/2607.04747v1)

**作者:** Vu Minh Tran `[一作]` (University of Information Technology), Khang Nguyen `[通讯]` (University of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 MergeSurv，一种基于模型合并的连续学习框架，用于在全切片图像上进行生存分析。

**💡 创新点**

创新点在于通过独立微调任务特定模型后将参数合并，无需存储历史数据，并引入 One-for-All 与 Voting-Expert 两种推理策略来兼顾统一性与任务专属性。

**🔧 技术方法**

采用 TITAN 病理图像预训练模型、MIL（多实例学习）、Cox 回归损失、OPCM 参数合并技术，以及文本提示路由等技术实现模型融合与风险预测。

**📊 数据集**

使用四个 TCGA 肿瘤队列（BLCA、UCEC、LUAD、BRCA）进行实验评估。

**📈 对比分析**

与 Naive fine-tuning、EWC、LwF、ER、DER、DER++ 等传统连续学习方法相比，MergeSurv VEA 在 C-Index 上接近或略优于联合训练，遗忘率最低，整体性能表现最为优异。

**⚠️ 局限性**

局限性包括对任务合并顺序敏感、依赖大规模预训练模型、对样本量极少的任务可能表现不佳，以及在多任务间实现完全无干扰的合并仍有挑战。

---

## 507. FM-ChangeNet: Learning Change through Pathwise Feature Transport

**arXiv ID:** 2607.04750 | [PDF](https://arxiv.org/pdf/2607.04750v1)

**作者:** Roie Kazoom `[一作]` (Google Research), Genady Beryozkin `[通讯]` (Google Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出将遥感双时序图像的变化检测转化为特征空间中的连续运输问题，学习时间条件速度场并通过路径监督实现对变化的显式建模；

**💡 创新点**

创新点在于引入路径式监督的流匹配（Flow Matching）框架，将速度场的幅值作为可解释的局部变化能量，并结合多尺度交叉时序对齐与层级解码，实现对变化的更精细、更鲁棒的捕捉；

**🔧 技术方法**

采用Vision Transformer编码器、对齐的交叉注意力、多尺度时间条件粗细解码器、速度场预测头以及融合速度幅值的分割头，并联合使用流匹配损失、轨迹一致性、TV正则和分割损失；

**📊 数据集**

在LEVIR-CD、WHU-CD和DSIFN-CD三个公开遥感变化检测基准数据集上进行实验；

**📈 对比分析**

与多种CNN、Transformer、Mamba、扩散模型等SOTA方法对比，实验显示在LEVIR-CD、WHU-CD上均取得最高的F1、IoU和总体准确率；

**⚠️ 局限性**

局限在于仅处理两时序数据，模型为确定性设计，未针对强时间错位、大规模部署或跨域强分布偏移进行评估；

---

## 508. Turning Off-Policy Tokens On-Policy: A Plug-in Approach for Improving LLM Alignment

**arXiv ID:** 2607.04728 | [PDF](https://arxiv.org/pdf/2607.04728v1)

**作者:** Yu Li `[一作]` (Renmin University of China), Zhen Chen `[通讯]` (JD.com)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Selective Importance Sampling（SIS），通过token级的接受-拒绝采样将离线rollout产生的离线（off-policy）token转换为近似on-policy token，从而减少重要性采样的方差并提升大模型RL后训练的稳定性。

**💡 创新点**

创新点在于：①将离线token直接转换为on-policy token的思路；②使用基于proposal分布的接受-拒绝采样与top‑K近似实现高效实现；③理论证明SIS能严格减小token级与序列级梯度估计误差，提供更紧的近似误差界；④将方法作为轻量级插件，可与现有RL后训练算法无缝结合。

**🔧 技术方法**

主要技术包括：重构重要性采样为接受-拒绝采样，采用top‑K近似计算最大常数M_t，使用熵、KL正则化等常见RL正则；理论上利用对数重要性偏差D与误差界分析；实验中使用FSDP、vLLM进行训练与推理。

**📊 数据集**

使用的基准数据集有：数学推理任务（MATH500、AMC23、AIME24、AIME25），问答与代理搜索任务（NQ、TriviaQA、PopQA、HotpotQA、Musique、Bamboogle），以及针对MoE的混合训练集（NQ+HotpotQA）。模型上使用Qwen3-8B、Qwen3-14B、Qwen3-30B-A3B（MoE）以及Llama3.2等。

**📈 对比分析**

与GRPO、DAPO、GSPO等基线及Clipped-IS、DPPO-TV、Clip-Cov等先进技术对比。SIS在所有模型、算法和任务上均实现了显著提升，平均在数学任务上提升约+6.4%，在代理任务上提升约+2.7%。在高离线重用、MoE路由不匹配和无剪切等极端场景下，SIS显著提升了训练稳定性与最终性能。

**⚠️ 局限性**

局限性包括：①需要估计top‑K的最大比值M_t，虽然开销小但对极端长序列或词表极大时仍需注意；②在token级接受率受限时，仍会保留大量off‑policy token，导致IS修正不完全；③缺乏针对不同奖励结构的深入分析，未来需评估在更复杂或多任务设置中的适用性。

---

## 509. Cam2Sim: Neural Scenario Reconstruction for Closed-Loop Autonomous Driving Simulation

**arXiv ID:** 2607.04770 | [PDF](https://arxiv.org/pdf/2607.04770v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 510. A Reliable Context-Aware and Temporal Planning Framework for Autonomous Driving

**arXiv ID:** 2607.04689 | [PDF](https://arxiv.org/pdf/2607.04689v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 511. Integrated Altruistic and Fairness Preference Induces Advanced Mutual Cooperation in Sequential Social Dilemmas

**arXiv ID:** 2607.04710 | [PDF](https://arxiv.org/pdf/2607.04710v1)

**作者:** Yu Wei `[一作]` (University of Tokyo), Yasuo Kuniyoshi `[通讯]` (University of Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于前瞻性奖励共享的AFP（Altruistic and Fairness Preferences）机制，在多智能体强化学习中融合利他与公平偏好，以促进在分布式环境下的合作，并在Cleanup和Harvest两种顺序社会困境游戏中进行实验验证。

**💡 创新点**

创新点在于将社会心理学与行为经济学的利他（SVO）与公平（CES）模型结合，形成一种全新的奖励塑形方法；同时采用前瞻性视角，让智能体基于未来总回报而非即时奖励来调整策略，实现更高效的协作。

**🔧 技术方法**

技术实现上使用分布式A2C学习框架，利用CNN+LSTM网络处理局部图像观测；将自定义的前瞻性效用函数嵌入策略梯度与价值网络更新；实验环境为OpenAI Gym自定义的Cleanup与Harvest。

**📊 数据集**

实验数据集主要来自两种自定义的顺序社会困境游戏——Cleanup（公共物品）和Harvest（公共资源耗竭），没有使用外部公开数据集。

**📈 对比分析**

对比方法包括IAC的自我主义(IAC‑e)和功利主义(IAC‑u)以及不平等厌恶(IA)基线。结果表明AFP在平均累计奖励、效率(U/C)、贡献度(C)和公平度(E、M)上均优于基线，尤其在Cleanup中AFP实现了高达约0.31的累计奖励与0.74的效率，远超IAC‑u的0.18奖励与0.64效率。

**⚠️ 局限性**

局限性包括：奖励解释不足导致学习速率慢；AFP智能体不主动惩罚背叛者，缺乏惩罚机制；人口同质性未考虑，缺少对多样性策略的探索；以及α与ρ等偏好参数需手动调优，缺乏自动化调节机制。

---

## 512. DeGenseGS: Geometrically and Semantically Decoupled Surgical Scene Understanding in 4D Gaussian Splatting

**arXiv ID:** 2607.04761 | [PDF](https://arxiv.org/pdf/2607.04761v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 513. PAST-TIDE: Prototype-Anchored Statement Tuning with Topic-Invariant Normalization for Stance Detection

**arXiv ID:** 2607.04690 | [PDF](https://arxiv.org/pdf/2607.04690v1)

**作者:** Md. Shakhoyat Rahman Shujon `[一作]`, Fakhri Karray `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了PAST-TIDE架构，用以在极低资源下进行多语言立场检测，直接复用预训练的MLM头进行分类。

**💡 创新点**

主要创新点包括：① Statement Tuning——将立场任务改写为cloze式MLM，利用预训练MLM头的语言知识；② Prototypical Contrastive Learning（PCL）——用固定类别原型替代批内负样本，保持梯度稳定；③ Topic‑Conditional Layer Normalization（T‑CLN）——为不同主题生成归一化参数，提升跨主题迁移；④ R‑Drop正则化——两次前向传播对称KL损失，增强决策边界。

**🔧 技术方法**

使用技术包括：mDeBERTa‑v3‑base编码器与其预训练MLM头；多词词表verbalizer；prototypical contrastive loss；topic‑conditional layer norm；focal loss、label‑smoothing；R‑Drop；以及Back‑Translation 数据增强。

**📊 数据集**

使用了StanceNakba 2026数据集：Subtask A（英语）1,401条样本；Subtask B（阿拉伯语）1,205条样本，均按70/15/15划分。

**📈 对比分析**

在官方排行榜上与多种基线（Dual‑Encoder、Target‑Calibrated、Disentangled、SSA等）对比，PAST‑TIDE在Subtask A获得宏F1 0.75、Subtask B 0.74，测试阶段分别为0.79，明显优于传统[CLS]头模型且参数更少。

**⚠️ 局限性**

局限性：① 样本量仍极小，尤其某些类别不平衡导致误判；② 依赖预训练词表中已出现的标签词，跨语言词表覆盖不足时效果下降；③ T‑CLN仅在跨主题子任务有效；④ 对更细粒度标签或多标签情境尚未验证；⑤ 需手工挑选verbalizer词，自动化选择尚未实现。

---

## 514. Does It Fail to See or Fail to Know? Attributing Errors in Vision-Language Models

**arXiv ID:** 2607.04683 | [PDF](https://arxiv.org/pdf/2607.04683v1)

**作者:** Khang Nhat Hoang Vo `[一作]` (MBZUAI), Yova Kementchedjhieva `[通讯]` (MBZUAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对知识密集型视觉问答的错误，本文提出了一个四层归因树框架，自动标注视觉证据缺失、实体识别失败、事实缺失和回忆失败等错误源，并在答案生成前通过内部表征预测错误类型；

**💡 创新点**

创新点在于将错误细分为可操作的阶段性来源，并证明不同阶段的内部表征（视觉 token、提示隐藏状态）在答案生成前就能准确指示错误类型，从而实现精准的失败源路由与补救；

**🔧 技术方法**

技术上使用了 VLM 预生成内部特征提取、线性或两层 Transformer 探针训练、PR‑AUC 评估，以及 GPT‑5 作为工具管理器进行补救实验；

**📊 数据集**

数据集主要包含 PopVQA（人物、地标、标志、绘画）与 iNaturalist（植物动物）两类知识密集型 VQA 数据，进一步通过人工干预构建视觉降质、实体识别与事实检索标签；

**📈 对比分析**

通过与后期生成与不确定性基线（OutSeq、TokProb、MSP 等）对比，预生成探针在识别相关错误时可获得 15–25% 的 PR‑AUC 提升，且在 Probe‑Guided Mitigation 中整体答案准确率提升 30–40 点；

**⚠️ 局限性**

局限性在于归因标签依赖操作式推断（如是/否探针、人工重写）且难以覆盖自然视觉缺陷；标签与模型特定，可能与真实因果关系存在偏差；补救实验仅作概念验证，使用 GPT‑5 成本高且易出现工具误差。

---

## 515. Towards Personalized Differentially Private Learning for Decentralized Local Graphs

**arXiv ID:** 2607.04777 | [PDF](https://arxiv.org/pdf/2607.04777v1)

**作者:** Longzhu He `[一作]` (Beijing University of Posts and Telecommunications), Sen Su `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了针对分散式图数据的个性化局部差分隐私框架PPGNN，用以在保持用户隐私偏好的同时实现高效的图神经网络学习。

**💡 创新点**

创新点在于：①设计了个性化扰动机制PPM，将隐私预算动态分配给节点特征与隐私级别保护；②提出了基于加权的FlexProp算法，对不同隐私级别下的噪声进行自适应校准；③在多维特征与离散隐私级别上实现了联合隐私保护。

**🔧 技术方法**

采用了局部差分隐私（LDP）技术，结合多维局部随机器（MLR）、扩展方波机制（ESW）、加权聚合（FlexProp）以及常用GNN模型（GCN、GraphSAGE、GAT）。

**📊 数据集**

使用了六个公开图数据集：Cora、Citeseer、Pubmed、LastFM、Facebook 和 Wikipedia。

**📈 对比分析**

与无隐私（NonPriv）、基线（BASE）以及现有LPGNN对比，PPGNN 在所有数据集和多种隐私预算下均显著提升节点分类准确率，逼近无隐私模型；同时在属性推断攻击中保持了与其他LDP方法相同的低泄露率。

**⚠️ 局限性**

局限性包括：在极端隐私预算分布时，低预算节点噪声过大可能导致邻域聚合偏差；以及假设用户能够显式声明隐私级别，实际应用中可能需要自动推断或动态调整隐私预算。

---

## 516. Aerial Manipulation: Contact, Medium Coupling, and the Geometry of Readiness

**arXiv ID:** 2607.04719 | [PDF](https://arxiv.org/pdf/2607.04719v1)

**作者:** Antonio Franchi `[一作]` `[通讯]` (University of Twente), Antonio Franchi (University of Twente)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述空中操纵领域，提出基于介质感知、内部动力学与冗余的理论框架，并定义空中操纵的本质及能力阶梯。

**💡 创新点**

首次系统化区分接触与介质耦合的交互模式，提出内部动力学纤维、气动即时性与被动耦合等新概念。

**🔧 技术方法**

主要采用数学几何、控制分配、机体动力学建模与案例分析等理论技术。

**📊 数据集**

未使用实验数据集，全部为文献综述与理论推导。

**📈 对比分析**

无实验对比，本文主要提供概念性框架与指标建议，未给出性能数值。

**⚠️ 局限性**

局限在于缺乏实证验证、指标量化不足，且对不同平台的适用性需进一步探索。

---

## 517. Minimum distances of LDPC codes in 5G standard

**arXiv ID:** 2607.04716 | [PDF](https://arxiv.org/pdf/2607.04716v1)

**作者:** V. R. Danilko `[一作]` (Novosibirsk State University), Ya. A. Tikhomolov `[通讯]` (Novosibirsk State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对5G NR标准中准循环LDPC码的最小距离进行了理论分析，并给出了紧凑的上界和下界估计；同时提出了基于循环模约简的低复杂度早停判定方法。

**💡 创新点**

创新点包括：①利用4层子码与模约简的Vontobel–Smarandache构造，显著简化距离上界计算；②引入环约简映射R，快速得到下界并搜索低权码；③设计循环模约简的早停判定，降低译码时的符号级移位开销。

**🔧 技术方法**

主要技术手段有：Vontobel–Smarandache判据、模约简映射R、Brouwer–Zimmermann算法（及其并行版）、块支持搜索、层级化的min‑sum LDPC译码与改进的早停判定。

**📊 数据集**

使用5G NR标准中的BG1与BG2基图代码（长度从1056位到25344位），涵盖不同循环大小（如q=384、48、96等）的左截断与全层码。

**📈 对比分析**

通过与相同码率随机码的误码率、平均迭代次数和未检测错误率（UIBLER）对比，实验表明新的早停判定在保留近似误检性能的同时显著降低了判定复杂度；距离上界与下界紧密匹配，译码性能与原始4层判定相当。

**⚠️ 局限性**

局限性包括：Vontobel–Smarandache在全层码上计算量过大；模约简下界对极大循环码的精确性有限；早停判定虽然降低复杂度，但在高SNR区仍可能产生未检测错误，需配合CRC等后置检测使用。

---

## 518. Learning Probabilistic Prompt for Continual Learning

**arXiv ID:** 2607.04711 | [PDF](https://arxiv.org/pdf/2607.04711v1)

**作者:** Hyekang Park `[一作]` (Yonsei University), Bumsub Ham `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于概率分布的 Prompt 采样框架，用于在类增量学习场景下实现连续学习。

**💡 创新点**

创新点在于将 Prompt 本身建模为可学习的概率分布，并通过查询条件的混合分布采样多样化 Prompt，解决了现有方法中 Prompt 崩溃（过度相似）问题，并加入分布正则化损失以抑制分布突变，缓解遗忘。

**🔧 技术方法**

采用 ViT-B/16 预训练模型、前缀调优（Pre‑T）技术、混合高斯分布采样、重参数化技巧、分布正则化（KL 约束）以及交叉熵训练。

**📊 数据集**

在 ImageNet‑R、CIFAR‑100 与 CUB‑200 三个公开类增量学习基准上进行实验，分别分为 5/10/20 任务或 10 任务拆分。

**📈 对比分析**

与 L2P、DualPrompt、CODA‑P、VQ‑Prompt、APT 等最新 Prompt‑based 方法对比，在 FAA 与 CAA 指标上均取得显著提升（如 ImageNet‑R 5 任务 FAA 80.53% > 79.23%，CIFAR‑100 10 任务 FAA 89.38% > 88.73%），同时保持计算与内存开销相近。

**⚠️ 局限性**

当前仅在 ViT 预训练模型和标准类增量学习设置下验证，缺乏对开放世界持续学习或不同网络架构的适应性评估。

---

## 519. F-ACVAE: A Federated Adaptive Conditional Variational Auto-Encoder for Privacy-Preserving Intrusion Detection in IoT Networks

**arXiv ID:** 2607.04698 | [PDF](https://arxiv.org/pdf/2607.04698v1)

**作者:** Mohammad Ansarimehr `[一作]` (Islamic Azad University), Ali Mousavi `[通讯]` (Islamic Azad University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为F-ACVAE的联邦学习框架，利用可调节的条件变分自编码器在物联网设备上实现隐私保护的入侵检测。

**💡 创新点**

创新点包括：①选择性参数聚合仅同步解码器和条件模块，保持本地编码器不被传输；②引入受约束的动量高斯聚合(CMGA)以缓解极端非IID导致的模型漂移；③在全局模型中结合随机森林进行混合特征分类。

**🔧 技术方法**

使用的技术包括：联邦学习框架Flower、条件变分自编码器(CVAE)、动量和裁剪的聚合策略(CMGA+MSA)、LeakyReLU激活、AdamW优化器、随机森林集成。

**📊 数据集**

在N-BaIoT数据集（七千万条样本、115维特征、9个设备子集）上进行实验，模拟真实的非IID场景。

**📈 对比分析**

与STA、CSAEC、MAE、MVAE和中心化CTVAE基线对比，F-ACVAE在所有设备上实现平均99%的准确率和宏F1，且通信开销降低约62%。

**⚠️ 局限性**

局限性包括：需要固定数量客户端、对模型投毒和隐私攻击缺乏完整防御、未充分验证在动态设备加入/离线场景下的收敛稳定性，以及缺乏真实硬件的能耗和延迟评估。

---

## 520. Solve the Missing First Step: Can VLMs Standardize Raw Heterogeneous Medical Data?

**arXiv ID:** 2607.04694 | [PDF](https://arxiv.org/pdf/2607.04694v1)

**作者:** Xin Chen `[一作]` (Shandong University), Yue Yao `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了原始医学数据标准化（Raw Medical Data Standardization）任务，构建了 MDS-Bench 基准，用来评估大型视觉语言模型（VLM）在将原始医学资源转换为统一、可直接使用的图像‑JSON 对时的能力。

**💡 创新点**

创新点在于：① 把医学 AI 评估从已整理好的图像/文本对转移到真实临床场景的原始多源数据；② 设计了跨文件、注释、元数据等碎片化资源的端到端标准化流程；③ 通过 11 项细粒度评估指标拆解结构、语义、内容、元数据和整体一致性，揭示 VLM 在完整管线中的薄弱环节；④ 引入验证引导的推理策略（如候选全量选择、字段级检验）来提升标准化质量。

**🔧 技术方法**

使用了 agentic VLM 框架（模型能与文件、工具和脚本交互），采用分阶段推理（定位源、转换视觉、对齐注释、输出校验）和基于验证的自我修正/候选选择技术；评估体系包括 11 个指标，涵盖结构合法性、语义正确性、信息完整性、有效性、冗余度、保真度、元数据一致性以及端到端通过率。

**📊 数据集**

构建了 1,939 条样本，涵盖分类、分割和检测三类任务，使用 100 个公开医学影像数据集，涉及 CT、MRI、X‑ray、超声、眼科、病理、内镜等多模态，原始格式包括 DICOM、TIFF、NIfTI、bitmap、视频等多种文件与目录结构。

**📈 对比分析**

与 9 种最前沿的 agentic VLM（Claude Haiku 4.5、Grok 4.20、Kimi K2.5、GPT‑5.2、GPT‑5.2‑Codex、Composer 1.5、Claude Sonnet 4.6、Claude Opus 4.6、Gemini 3 Flash）在同一基准上进行公平评测；Gemini 3 Flash 在多数指标上领先，尤其是结构与内容完整度，但其端到端成功率最高仅为 48.6%；验证引导推理（特别是完整候选选择）可将 E2E 提升至 63.5%。

**⚠️ 局限性**

局限性包括：① 仅使用公开数据集，未覆盖真实 PACS、EHR 等闭源环境；② 关注分类/检测/分割，未涉及报告生成、纵向建模等任务；③ 评估侧重结构与源对齐，可能忽略细微医学解释误差；④ 真实标签仍依赖大型模型辅助提取并人工校验，可能残留标注错误；⑤ 生成的标准化数据仅用于研究，不能直接用于临床决策。

---

## 521. Symmetry all the way down

**arXiv ID:** 2607.04887 | [PDF](https://arxiv.org/pdf/2607.04887v1)

**作者:** Ignacio Amores-Sesar `[一作]` (Aarhus University), Juan Villacis `[通讯]` (University of Bern)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

研究并证明了在异构信任（asymmetric trust）模型下，只有深度为 1 的简单任务可以在比对称信任更宽松的失败模式下得到求解；对深度 ≥2 的复杂任务，任何可行的失败场景都能被相应的对称四皇系统（symmetric quorum system）所容忍。

**💡 创新点**

提出了一个深度为 2 的编译器（compiler），能够将任意满足 B³ 条件的异构四皇系统转换为满足 Q³ 条件的对称四皇系统，从而在深度 ≥2 的场景下实现对称协议的使用；同时证明不存在深度为 1 的编译器，确立了深度参数下编译器的最优边界。

**🔧 技术方法**

主要技术包括深度层次（depth hierarchy）的定义与分析、B³ 与 Q³ 条件的形式化、基于深度的编译器设计（枚举与扩展四皇集合），以及对称/异构四皇系统性质的证明。

**📊 数据集**

该工作为理论分析，未使用任何实验数据集；所有结果均来自形式化证明与逻辑推理。

**📈 对比分析**

对比方法主要是理论证明：通过构造编译器并证明其在深度 ≥2 的失败模式下满足一致性与可用性；性能方面讨论了编译过程的时间复杂度（在最坏情况下为 O(n·|ℚ|)），并给出了在线检查对称四皇是否存在的线性 O(n) 方法。

**⚠️ 局限性**

局限性包括：编译器仅在已知四皇（known‑quorums）设置下可实现；在未知四皇（unknown‑quorums）环境下如何构造对应的对称四皇系统仍是开放问题；此外，编译后的四皇系统可能指数级大，导致实际实现时存储与检查开销较大。

---

## 522. Psychological features of dispute content and public acceptance of AI in legal adjudication: evidence for systematic variation beyond individual differences

**arXiv ID:** 2607.04838 | [PDF](https://arxiv.org/pdf/2607.04838v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 523. Unsupervised Detection of Underground Tunnels in Ground-Penetrating Radar Using Depth-Restricted Reconstruction Scoring

**arXiv ID:** 2607.04882 | [PDF](https://arxiv.org/pdf/2607.04882v1)

**作者:** Muhammad Junaid `[一作]` (Sir Syed CASE Institute of Technology), Nisar Ahmed `[通讯]` (University of Strathclyde)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一套无监督地下隧道检测管线的GPR雷达图像系统，通过去噪自编码器和深度限制的top‑k异常评分实现自动检测。

**💡 创新点**

①发布公开隧道雷达数据集；②引入物理先验的深度限制top‑k评分显著提升性能；③证明池化比例与深度约束相互影响；④评估空间投票的有限收益。

**🔧 技术方法**

使用去噪卷积自编码器、均方误差训练、top‑k池化、深度限制、无监督阈值设定以及空间投票后处理。

**📊 数据集**

使用由伊斯兰堡农田手工挖掘的三条隧道与19,743条正常雷达图组成的公开数据集，测试集包含1,600窗口。

**📈 对比分析**

与全图均值、全图top‑5%以及传统Isolation Forest对比，最终深度限制top‑5%在未标记阈值下实现AUC 0.994、F1 0.975，漏检率2.7%，误报率1.6%。

**⚠️ 局限性**

仅在单一地点和单一频率设备下验证；池化比例与深度阈值等超参在报告集上调优；窗口步长导致边缘漏检；未实现三维定位或多传感器融合。

---

## 524. Subcube Stifling

**arXiv ID:** 2607.04850 | [PDF](https://arxiv.org/pdf/2607.04850v1)

**作者:** Arjan Cornelissen `[一作]` (University of California), Nithish Raja `[通讯]` (Eindhoven University of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

定义并研究了子立方抑制数（subcube stifling number）这一新的组合复杂度度量，并证明其与逼近度（approximate degree）组合定理的关系。

**💡 创新点**

提出子立方抑制数并揭示其与经典度量（如灵敏度、块灵敏度、λ、距离等）的紧密联系，证明其在随机函数、对称函数、线性码指示函数等上的取值范围，并给出随机函数子立方抑制数为log n、线性码指示函数子立方抑制数为线性等新结果。

**🔧 技术方法**

使用组合论、量子查询算法、对偶多项式方法、线性码理论等技术构建和证明。

**📊 数据集**

无实际数据集，全部为理论分析。

**📈 对比分析**

通过与已知下界（如块灵敏度、λ、距离等）以及对称函数的已知下界进行比较，证明子立方抑制数能得到更紧的逼近度组合下界，但未给出实验性能。

**⚠️ 局限性**

尚未给出逼近度与子立方抑制数满足Θ(√μ)的函数，无法完成所有期望的逼近度组合；存在函数逼近度与子立方抑制数不匹配的例子，限制了该度量的通用性。

---

## 525. SynSFX: Multi-Model Sound Effects Synthesis Dataset for Deepfake Detection and Evaluation

**arXiv ID:** 2607.04848 | [PDF](https://arxiv.org/pdf/2607.04848v1)

**作者:** Linxi Li `[一作]` (University of Warwick), Carsten Maple `[通讯]` (University of Warwick)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 SynSFX，一个多生成器、共享提示子集的大规模非语音音频深度伪造检测基准数据集，包含 43,374 条音频（共 178 小时）

**💡 创新点**

创新点在于：① 大规模、公开的非语音深度伪造数据集与共享提示子集；② 系统揭示语音检测器在非语音上的灾难性遗忘；③ 证明联合域训练可恢复语音检测能力但仍难以零样本泛化；④ 提供诊断可视化与统计分析

**🔧 技术方法**

使用多种文本到音频生成模型（AudioCraft、AudioLDM1/2、MMAudio、Make‑An‑Audio、StableAudio、TangoFlux），以及深度学习检测架构（AASIST、RawNet2、EAT‑AASIST），并利用 LLM 生成提示

**📊 数据集**

数据集 SynSFX（真实 16,922 条、119.7 小时；合成 26,452 条、61.1 小时），以及公开语音数据（ASVspoof 2019 LA）和未见生成器/真实集用于评估

**📈 对比分析**

与基线相比，零样本检测中 AASIST/RawNet2 仅能达到约 50% EER，EAT‑AASIST 达 23.71% EER；Fine‑tune 后 SynSFX‑only 在非语音上 EER 降至 3.23%/2.36%，但导致语音检测 EER 上升至 33.61%；联合域训练后语音 EER 恢复至 3.61%/5.25%，非语音 EER 维持 3.76%/2.45%；但在未见生成器上仍达 30–37% EER

**⚠️ 局限性**

局限性：数据集无法覆盖全部真实声景；未见生成器样本有限；仍无法解决跨生成器零样本泛化；未深入分析不同语义内容对检测的影响

---

## 526. Accelerated estimation of quantities of interest via adjoint-based model reduction

**arXiv ID:** 2607.04808 | [PDF](https://arxiv.org/pdf/2607.04808v1)

**作者:** Clément Vella `[一作]` (Suqaba), Serge Prudhomme `[通讯]` (Polytechnique Montréal)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一种仅针对伴随（adjoint）问题的PGD（Proper Generalized Decomposition）模型，利用该模型在不求解原始（primal）问题的前提下，快速、准确地估计线性偏微分方程多查询场景下的感兴趣量（QoI）。

**💡 创新点**

创新点在于：①仅对伴随问题进行降维建模，从而把原本需要针对每个载荷求解一次原始问题的昂贵计算转化为一次全局模型构建后即可对任意载荷进行快速评估；②将伴随问题的参数化与量纲分离，得到与载荷无关的代理模型；③在Poisson与平面应力问题上展示了相较于传统基于原始问题的PGD ROM更快的收敛速度和更高的QoI预测精度。

**🔧 技术方法**

采用的技术包括：PGD固定点迭代、Aitken Δ²加速、SVD分解对高斯核函数进行可分离近似、有限元（FEM）离散、线性残差加权（DWR）思路的引入、以及对离散线性系统的Cholesky分解与前向/后向代数求解。

**📊 数据集**

数据集为：①Poisson问题在单位正方形域上，采用三种源项（常数、xy²、cos‑sin 组合）以及3×3的参数网格；②平面应力结构支架问题，使用包含约100万自由度的高分辨率网格，载荷参数α、β在[0,2π)上各取360个离散点，形成360×360的多查询网格。

**📈 对比分析**

与完整有限元模型（FOM）以及传统基于原始问题的PGD ROM进行对比。精度方面，Poisson问题在50个PGD模式下，QoI相对误差<1%；平面应力问题在10个模式下，RMSE≈1%（伴随ROM）对比约2%（原始ROM）并呈现更窄的误差分布。计算性能方面，完整FEM总耗时≈311 s（含33 s稀疏Cholesky分解+278 s前后向替换），而伴随PGD方法的离线构建≈56 s，在线评估≈5 ms，显著降低了多查询的总成本。

**⚠️ 局限性**

局限性包括：①仅适用于线性PDE和线性QoI，非线性问题需要同时构建原始与伴随的代理模型或采用其它超低秩近似；②伴随模型对核函数的可分离性要求较高，若核函数不可有效分离会导致SVD分解或近似误差放大；③在高度非平滑或复杂几何、极端参数变化时，PGD模式数可能需要大幅增加，影响离线构建时间；④若载荷本身非参数化且极为多样化，虽然模型独立于载荷，但若核函数无法覆盖所有载荷特征，QoI估计精度仍受限。

---

## 527. Modeling Fatigue-Induced Anisotropic Quasi-Brittle Damage Based on the Endurance Surface Concept

**arXiv ID:** 2607.04818 | [PDF](https://arxiv.org/pdf/2607.04818v1)

**作者:** Klas Feike `[一作]` (TU Dortmund University), Jörn Mosler `[通讯]` (TU Dortmund University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了基于耐久面概念的连续损伤框架，用以模拟高循环准脆性疲劳，并通过该框架对混凝土和低合金钢的单调与循环加载行为进行校准与验证。

**💡 创新点**

创新点在于将耐久面与能量释放率张量结合，利用距离耐久面的距离驱动损伤演化，并通过微形梯度正则化与各向异性损伤实现网格无关的高循环疲劳模拟。

**🔧 技术方法**

采用连续损伤力学、广义标准材料框架、微形梯度增强、微裂纹闭合/开启效应以及能量释放率张量驱动的损伤演化方程，辅以有限元弧长法和多项式周期外推技术。

**📊 数据集**

使用实验单调加载数据（L 型混凝土试件）和低合金钢杆的高循环疲劳 S–N 曲线数据进行校准，并对轴向-扭转加载的圆柱钢样本进行多轴测试。

**📈 对比分析**

将模型预测与实验力-位移曲线及 S–N 图进行对比，结果显示能准确捕捉冲击峰值、软化过程和无寿命区间，异向性损伤可提升约 8% 的疲劳寿命，且数值收敛性良好，计算稳定、热力学一致。

**⚠️ 局限性**

局限性包括仅针对准脆性高循环疲劳，需大量经验参数校准，未在非比例多轴或极大循环数下进行验证，且对极大循环数的计算成本仍较高。

---

## 528. Glare Mitigation using a Differentiable Unified Glare Rating

**arXiv ID:** 2607.04796 | [PDF](https://arxiv.org/pdf/2607.04796v1)

**作者:** Linas Beresna `[一作]` (Simon Fraser University), Eugene Fiume `[通讯]` (Simon Fraser University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将CIE Unified Glare Rating（UGR）转换为连续可微分代理的框架，并通过可微分渲染联合优化材质参数、光源分布以及表面粗糙度，自动降低室内场景的视觉不适。

**💡 创新点**

创新点在于：1）用可微分光学散射（PSF）平滑 Monte Carlo 方差并消除光子噪声；2）将二元阈值 L>5L_b 替换为可微分的 Sigmoid 软掩模，实现UGR梯度可传递；3）在三大辐射域（光源遮蔽、边界抗反射、表面散射）中系统评估并证明不同物理策略的稳定性与效果；4）结合背景光自适应 L_b 的自正则化，避免过度“雾化”导致的暗室陷阱。

**🔧 技术方法**

采用的技术包括：可微分路径追踪（Mitsuba 3 + Dr.Jit），光学散射滤波（多次 3×3 高斯卷积），Sigmoid 软阈值，Total Variation 正则化，Adam 优化器，以及在 GPU 上并行实现的连续 UGR 代理。

**📊 数据集**

使用的基准数据集包括：Cornell Box、Grey Room、White Room、Country Kitchen、Contemporary Bathroom 以及 Stanford Bunny 模型，用于验证不同环境下的光照与反射效果。

**📈 对比分析**

通过与传统硬阈值 UGR 计算、CMA‑ES、Powell 等无梯度优化器的对比，实验显示连续软阈值方案在保持视觉舒适度（UGR≤17）同时避免过度粗糙化，且梯度优化收敛更快、可扩展至高维纹理（64×64 以上）且不受样本数与路径长度显著影响；硬阈值导致过度抖动与视觉瑕疵。

**⚠️ 局限性**

局限性包括：1）需手动决定哪些参数暴露给优化器；2）背景光 L_b 是否附加到图中是情境性手动设置，可能产生空间纹理误差；3）仅支持单一固定视角与静态照明，无法处理动态人视角或日照变化；4）优化结果不保证与现成可制造材料匹配，需进一步约束物理可行性；5）未自动识别最敏感材料或光源，无法实现完全无监督的光照设计。

---

## 529. DGSeg: Dynamic Gating of Semantic-Spatial Guided Predictions for Reasoning Segmentation

**arXiv ID:** 2607.04779 | [PDF](https://arxiv.org/pdf/2607.04779v1)

**作者:** Ruizhe Zeng `[一作]` (Chinese Academy of Sciences), Zhiyong Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 DGSeg 框架，用 MLLM 先生成语义与空间两种提示，然后分别在两条分割分支中处理，并通过动态门控模块自适应融合两条分支的预测，最终实现 reasoning segmentation；

**💡 创新点**

创新点在于（1）将语义与空间提示分离处理，避免单一提示噪声蔓延；（2）设计轻量级动态门控网络，基于分支特征评估并抑制错误区域，实现像素级自适应融合；（3）采用两阶段训练策略：先用 RL（GRPO）微调 MLLM 提示质量，再冻结 MLLM 与 SAM3 仅训练门控模块，提升稳定性与效果；

**🔧 技术方法**

技术包括：Multimodal Large Language Model（Qwen2.5‑VL）、SAM3 promptable 分割模型、双分支结构、动态门控网络（两层卷积+sigmoid）、RL 微调（GRPO）、LoRA 参数微调、两阶段训练；

**📊 数据集**

数据集：训练使用 RefCOCOg（9000 条实例）；评估 ReasonSeg（验证/测试）、RefCOCO、RefCOCO+、RefCOCOg；

**📈 对比分析**

在 ReasonSeg 零样本设置下，7B 版本 DGSeg 达到 69.6%/67.3% gIoU，超过 SAM‑Veteran、CoPRS 等基线；在 RefCOCO 等基准上，平均 76.9% gIoU，优于 Seg‑Zero、PIXELTHINK 等同基底模型；整体提升 3–5% 以上；

**⚠️ 局限性**

局限性：仍受限于 MLLM 生成提示的准确性，极端噪声或复杂场景下门控可能无法完全抑制错误；计算开销略升 0.3% FLOPs，FPS 降低约 2%；

---

## 530. Enhancing the Forecasting Capability of Multi-Model Blending Algorithms for Extreme Precipitation via Joint Use of Station and Gridded Observations

**arXiv ID:** 2607.04862 | [PDF](https://arxiv.org/pdf/2607.04862v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 531. Towards Fully Dynamic Omnitrees: Moment-Conserving Anisotropic Compression With Wavelets

**arXiv ID:** 2607.04881 | [PDF](https://arxiv.org/pdf/2607.04881v1)

**作者:** Theresa Pollinger `[一作]`, Jens Domke `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

分析了使用Blosc2与离散小波变换对OpenVDB文件进行可调压缩的效果，并比较了不同阈值下的文件大小。

**💡 创新点**

提出在OpenVDB中结合Blosc2实现可调节阈值的波形压缩策略，提升压缩率且保持较低失真。

**🔧 技术方法**

采用Blosc2压缩库、离散小波变换（DWT）、OpenVDB文件格式进行实验。

**📊 数据集**

使用云层模拟体数据（cloud_wavelet_stats.csv）作为实验数据集。

**📈 对比分析**

通过绘制原始、Blosc2压缩与标准压缩文件大小随阈值变化的对比图，显示Blosc2在大多数阈值下显著减小文件尺寸。

**⚠️ 局限性**

实验仅限单一数据集，未评估压缩对解码速度、内存占用或数值精度的影响。

---

## 532. No Distributed Quantum Advantage for 3-Coloring Rooted Trees and 2-Coloring Even Cycles

**arXiv ID:** 2607.04852 | [PDF](https://arxiv.org/pdf/2607.04852v1)

**作者:** Pierre Fraigniaud `[一作]` (Université Paris Cité), Isabella Ziccardi `[通讯]` (Université Paris Cité)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究分布式量子模型下的3-着色根树和偶长环着色问题，并给出了最优下界。

**💡 创新点**

首次将非信号性、有限依赖框架与叉依赖/颜色提升技术相结合，证明根树3-着色无量子加速；并将偶环2-着色的下界从 ⌈(n-2)/4⌉ 提升到 n/2-1。

**🔧 技术方法**

引入叉依赖、迭代幂集与颜色提升的非信号性技术，结合轮次消除与分层分析完成下界证明。

**📊 数据集**

未使用实验数据集，全部为理论证明。

**📈 对比分析**

通过严谨的非信号性下界证明，得到 Ω(log* n) 与 n/2−1 的下界，表明在这些着色问题上量子算法无法获得任何加速。

**⚠️ 局限性**

结论仅适用于根树和偶长环，无法推广到普通环或更一般图；对有向环或随机图的量子优势仍未知。

---

## 533. EventCoT: Event-centric Video Chain-of-thought for Reasoning Temporal Localization

**arXiv ID:** 2607.04872 | [PDF](https://arxiv.org/pdf/2607.04872v1)

**作者:** Youngkil Song `[一作]` (Pohang University of Science and Technology), Suha Kwak `[通讯]` (Pohang University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于事件中心化链式思考的 EventCoT 框架，用于视频推理时序定位；

**💡 创新点**

创新点在于通过事件分割与嵌入匹配将推理与时序定位解耦，既高效又精确；

**🔧 技术方法**

采用 DPC‑KNN 事件边界检测、事件上下文注意力、LLM（Vicuna‑7B / Qwen2.5‑7B）与 placeholder token 嵌入匹配、交叉熵＋DIoU 损失；

**📊 数据集**

训练集包含 ActivityNet‑Captions、YouCook2、NExT‑QA、LLaVA‑150K 等，评测集为 ActivityNet‑RTL 与 ReXTime；

**📈 对比分析**

与 LITA、Qwen3.5‑9B、TimeLens‑8B 等单模型对比，EventCoT 在 ActivityNet‑RTL 的定位和答案质量均领先，且仅使用 20% 视觉 token；在 ReXTime 的零样本性能同样为最佳；

**⚠️ 局限性**

局限在于固定事件数 N 与帧数 T，限制了对长视频的适应性；placeholder token 需同时兼顾生成与匹配，未完全分离。

---

## 534. Athena-WBC: Capability-Aligned Policy Experts for Long-Tail Humanoid Whole-Body Control

**arXiv ID:** 2607.04837 | [PDF](https://arxiv.org/pdf/2607.04837v1)

**作者:** Yuan Jiang `[一作]`, Jie Chen `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 Athena-WBC，一种教师-学生管道，利用能力对齐的动态与平衡专家来解决训练集长尾失败；

**💡 创新点**

创新点在于通过能力匹配的专家（动态专家去除努力与时间平滑惩罚，平衡专家使用重力课程）以及运动路由的 DAgger 级联压缩，提升长尾覆盖率并保持可部署控制质量；

**🔧 技术方法**

主要技术包括奖励分解与辅助政策正则化、重力持续课程、运动路由与教师选择、DAgger 行为克隆以及 RL 微调；

**📊 数据集**

使用 AMASS、Bones-Seed、BEAT 以及自研 mocap 共 55,482 条运动片段；

**📈 对比分析**

与 SONIC-Base 及无平滑基线对比，在 AMASS-eval、Omni-eval 的 SR、TIS、MPJPE、MPJPE-W 与动作率上实现了显著提升，尤其在长尾子集的成功率与追踪误差方面；

**⚠️ 局限性**

局限在于仍未完全覆盖所有长尾失败，受参考质量与物理极限影响；管道复杂度高，需多阶段训练；实验受限于计算预算与缺乏系统化真实机器人验证。

---

## 535. Probably Correct Optimal Stable Matching under Two-Sided Uncertainty

**arXiv ID:** 2607.04824 | [PDF](https://arxiv.org/pdf/2607.04824v1)

**作者:** Andreas Athanasopoulos `[一作]` (University of Neuchâtel), Christos Dimitrakakis `[通讯]` (University of Neuchâtel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在两侧都未知偏好的稳定匹配市场中，以纯探索方式高概率识别最优稳定匹配，并提出基于部分偏好信息的消除算法与停机规则。

**💡 创新点**

创新点在于：①首次将“普遍稳定匹配”(pervasive stable matching) 概念引入学习框架，利用部分偏好即可确定最优匹配；②设计了无需先验间隙的自适应消除算法并给出改进的样本复杂度；③在两侧不确定性下给出了首次避免依赖最小奖励间隙 Δ_min 的调度与上界；④通过实验验证消除+停机策略显著优于传统全覆盖或无停机方法。

**🔧 技术方法**

主要技术包括：半金字塔反馈（semi‑bandit）模型；基于最小匹配覆盖的最小边着色；上置信界/下置信界构造的部分偏好图；Pervasive Stable Matching (PSM) 与 Super Stable Matching (SSM) 的多项式识别；信息论下界推导；以及Meta‑Algorithm 结合 E‑PSM 的调度。

**📊 数据集**

实验使用的是随机生成的合成实例：两侧各 n 名代理，偏好通过随机扰动产生，奖励间隙从 [Δ_min,1] 采样（Δ_min=0.2）。

**📈 对比分析**

与基线（全覆盖 U、U‑Δ、U‑PSM、E‑NS、EE‑PSM）和现有调度算法 AETGS‑E 的比较显示：E‑PSM 与 EE‑PSM 的样本复杂度最小，尤其在大市场下几乎不受市场规模影响；在调度调度实验中，E‑PSM 的累积不稳定性（即调度错误次数）明显低于 AETGS‑E，表明在两侧不确定性下提供更好的性能。

**⚠️ 局限性**

局限性包括：① 需要预先知道或估计有效部分偏好集合，实际计算 NP‑hard；② 对 Δ_F 的依赖使理论分析在实践中难以直接应用；③ 扩展消除规则（基于 SSM）的理论样本复杂度尚未建立；④ 仅在合成数据上验证，缺乏真实市场数据的实证。

---

## 536. Performance evaluation of scheduling tasks in many-core systems utilizing processes and threads

**arXiv ID:** 2607.04821 | [PDF](https://arxiv.org/pdf/2607.04821v1)

**作者:** Mejgan Dedaj `[一作]` (University of Ioannina), George Tavridis `[通讯]` (University of Ioannina)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估多核共享内存系统中基于进程、管道和线程的调度器在大三维张量行级快速排序工作负载下的可扩展性。

**💡 创新点**

提出了 AIMD 与自适应块调度器，并比较了繁殖式与集体式进程分叉以及 1:1、1:M、M:M 三种管道通信模式的性能差异。

**🔧 技术方法**

采用 C++/POSIX 进程与线程、Boost.Thread、管道 IPC、系统 V 共享内存、EWMA、AIMD、动态/引导/自适应块调度等技术实现调度器。

**📊 数据集**

使用 1000×640×640 与 10000×640×640 的 8 位无符号三维张量作为实验数据集。

**📈 对比分析**

在 24 核 Xeon 服务器上对执行时间、加速比、效率、α（串行化系数）等指标进行量化；结果显示动态/引导线程调度和最佳管道（1:1 或 M:M）在硬件并行度下可达约 23 倍加速，而进程调度在过载时明显退化。

**⚠️ 局限性**

实验仅针对相对均匀的行级快速排序，缺乏对结构性不规则、分布式、NUMA 或异构环境的评估。

---

## 537. LILAC: Layer-Wise Independent LoRAs and Cascaded Conditioning for Multi-Concept Customization of Diffusion Models

**arXiv ID:** 2607.04801 | [PDF](https://arxiv.org/pdf/2607.04801v1)

**作者:** Marian Lupascu `[一作]` (Adobe Research), Ionut Mironica `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在推理时通过逐层叠加独立训练的LoRA适配器，实现多主体个性化生成而不需要权重融合。

**💡 创新点**

将多主体生成重构为层化合成，通过每层仅激活单个适配器消除参数级干扰，并以冻结的层作为条件保持空间与光照一致。

**🔧 技术方法**

使用LoRA低秩适配、Layered RGBA扩散模型或Qwen-Image-Edit+Layered框架，构造链式推理与冻结上下文。

**📊 数据集**

采用Orthogonal Adaptation概念库（数十个人物、虚构角色、动物及配饰的少量参考图）进行单概念训练并随机组合。

**📈 对比分析**

与多种权重融合基线（Custom Diffusion、DB-LoRA、Mix-of-Show、Orthogonal Adaptation）在Orthogonal Adaptation评估协议下比较，ArcFace身份保持率提升至0.861/0.877，文本与图像相似度保持相近。

**⚠️ 局限性**

需要逐层多次采样导致推理时间线性增长，且在近距离互相遮挡或物理接触的场景中难以实现精细交互。

---

## 538. An Exploration of Agentic Information Fusion for Test Maintenance Prediction

**arXiv ID:** 2607.04786 | [PDF](https://arxiv.org/pdf/2607.04786v1)

**作者:** Jingxiong Liu `[一作]` (Chalmers University of Technology and University of Gothenburg), Gregory Gay `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提出了MAST，一个多智能体框架，用于在生产代码变更后预测哪些单元/集成测试需要维护。

**💡 创新点**

创新点在于将静态、词法与语义三种分析通过LLM融合与后置检查相结合，显著提升了预测精度。

**🔧 技术方法**

技术包括LLM（Qwen3-Coder）、调用图静态分析、BM25词法检索、RAG语义检索以及LangGraph多智能体调度。

**📊 数据集**

数据集来自 Ericsson AB 的21个Java项目，共430条正向变更和210条负向变更。

**📈 对比分析**

与仅使用语义分析的基线相比，MAST在正向案例中精度提升约69%、F1提升28%，在负向案例中误报率下降83%。

**⚠️ 局限性**

主要局限是存在漏检、仅针对Java且在内部企业仓库评估，缺乏跨语言或开源项目验证。

---

## 539. Ghost Traffic: ICMP Tunneling-Based Billing Bypass in LTE Networks

**arXiv ID:** 2607.04783 | [PDF](https://arxiv.org/pdf/2607.04783v1)

**作者:** Jung Jin Kim `[一作]` (78ResearchLab), Seungho Jeon `[通讯]` (Gachon University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出Ghost Traffic攻击系统，利用Android无root权限捕获应用流量并通过ICMP回显隧道绕过运营商计费；

**💡 创新点**

首次实现无root、无需VoLTE的ICMP隧道计费绕过，并在多国运营商环境中实测验证；

**🔧 技术方法**

采用Android VPNService与TUN接口捕获IP包、ICMP echo socket封装、外部代理服务器解封装并转发；

**📊 数据集**

在韩国、日本、美国七个运营商的实际网络环境中进行功能、性能与计费测试；

**📈 对比分析**

与原始TCP/IP通信进行吞吐量、RTT和重传对比，发现某些运营商可实现数Mbps吞吐且QoS未被触发，说明性能可接受；

**⚠️ 局限性**

受限于运营商对ICMP处理策略、地址分配方式及实验设备可用性，部分环境无法测量或性能极差；

---

## 540. Predicting Drafted Deck Strength for "Magic: the Gathering"

**arXiv ID:** 2607.04782 | [PDF](https://arxiv.org/pdf/2607.04782v1)

**作者:** Tomas Rigaux `[一作]`, Hisashi Kashima `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

该论文使用 LaTeX 与 TikZ 绘制了一张名为 Quantum Riddler 的魔法卡牌排版与视觉布局。

**💡 创新点**

创新点在于将卡牌设计与 TikZ 结合，利用自定义节点、箭头和宏实现文本与图形的精确对齐与可视化。

**🔧 技术方法**

使用了 LaTeX、TikZ/PGF 库以及自定义节点样式与宏。

**📊 数据集**

未使用外部数据集，所有内容均为作者手工编写。

**📈 对比分析**

未提供对比实验，性能评价仅为排版效果与可维护性，作者说明该方法可复用并易于修改。

**⚠️ 局限性**

局限性包括：对 TikZ/PGF 的依赖导致编译时间较长，且对非专业用户的可读性与可扩展性有限。

---

## 541. PRISM: Personalized Robotic Dataset Generation via Image-based Scene and Motion Synthesis

**arXiv ID:** 2607.04880 | [PDF](https://arxiv.org/pdf/2607.04880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 542. Terastate-per-second QUBO Brute-Force on a Single GPU: A Matrix Prefix-Suffix Decomposition

**arXiv ID:** 2607.04857 | [PDF](https://arxiv.org/pdf/2607.04857v1)

**作者:** Aleksandr Maltsev `[一作]`, Ekaterina Krivtsova `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于前缀–后缀分解与 Gray 码遍历的并行 QUBO 全枚举算法，能够在 GPU 上实现每个状态 O(1) 的能量计算。

**💡 创新点**

创新点在于将状态空间拆分为可并行处理的块，利用 Gray 码实现增量能量更新，并将关键能量子块缓存于 GPU 寄存器，从而将瓶颈从内存带宽转移至计算单元，显著提升吞吐量。

**🔧 技术方法**

使用了 prefix-suffix 分解、Gray 码增量更新、寄存器级缓存、CUDA 自定义 kernel 以及混合整数 16 位数据类型来实现高效并行。

**📊 数据集**

实验使用随机生成的稠密 QUBO 矩阵（不含真实工业实例），在单个 H100 GPU 上测得 7.5×10¹² 状态/秒，V100 GPU 上 2.3×10¹² 状态/秒。

**📈 对比分析**

与 CPU 单核、2080Ti/V100 以及 H100 其它基准（包括已有的 brute‑force、QBF、CUDA 实现）进行对比，本文实现的吞吐量比现有方法高 10 倍以上，单 GPU 计算能力突破 10¹² 状态/秒的门槛。

**⚠️ 局限性**

主要局限在于仅针对稠密 QUBO 矩阵、整数 16 位，单个 kernel 最大处理 N≤49；对稀疏矩阵、浮点 QUBO 或更大规模问题需进一步拆分或改进；目前仅定位最小能量，未提供低能谱构造。

---

## 543. Strong ILP Formulations for the p-Regions Problem

**arXiv ID:** 2607.04886 | [PDF](https://arxiv.org/pdf/2607.04886v1)

**作者:** Daniel Faber `[一作]` (University of Bonn), Petra Mutzel `[通讯]` (University of Bonn)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了两种新的整数线性规划（ILP）模型，用来解决具有连通性约束的p-regions问题，并对其多面体性质进行了理论分析与实验评估。

**💡 创新点**

创新点包括：①将k-partitioning的边代表模型与图的顶点分离约束相结合，得到无对称性的基本模型；②引入一种针对p-regions特定的新子巡回（subtour）消除不等式；③将顶点分离约束与子巡回约束融合得到最强模型，并证明其在投影多面体上严格优于现有模型。

**🔧 技术方法**

技术手段主要是ILP建模与分支定价（branch‑and‑cut）算法：利用Gurobi求解器；对指数规模约束（子巡回、顶点分离、一般团约束）采用分离算法；对新子巡回不等式使用启发式分离；并结合多面体理论进行强化。

**📊 数据集**

数据集包括：
- 欧盟Eurostat提供的NUTS级别（主要是NUTS‑3）失业率统计数据，覆盖19个欧盟国家，顶点数从41到111；
- 随机生成的网格图（4×4至9×9）并基于空间自回归模型产生属性，ρ取0,0.3,0.6,0.9，构成若干大小和ρ的组合。

**📈 对比分析**

与Duque等人提出的三种基准模型（z‑model、tree‑model、c‑model）进行对比。实验结果显示：
- 对于所有k值，尤其是中等k（5–8）时，新的模型在求解时间和可解实例数上均明显优于基准；
- 对于较大k（≥7）时，优势更为显著；
- 在部分大实例（如8×8、9×9网格）中，新模型能够提供<10% 的最优缺口，解决了以前无法求解的案例。

**⚠️ 局限性**

局限与未来工作：
- 当前模型未考虑区域紧凑度等额外约束，未来可通过调整边权实现；
- 新的子巡回不等式的分离问题未被证明多项式可解，仍需研究高效分离策略；
- 实验仅在欧盟统计数据与网格实例上验证，尚未在更大规模或不同领域的数据上测试；
- 由于对分离子巡回约束使用启发式，可能在极端实例上产生子optimal分离效果。

---

## 544. KinEMbed: Decoding Kinematics from Electromyography via Cross-Modal Contrastive Learning

**arXiv ID:** 2607.04820 | [PDF](https://arxiv.org/pdf/2607.04820v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 545. Handover-Optimal User Association Policy for LEO Satellite-based 5G NTN

**arXiv ID:** 2607.04829 | [PDF](https://arxiv.org/pdf/2607.04829v1)

**作者:** Pradnya Taksande `[一作]` (Indian Institute of Technology Bombay), Prasanna Chaporkar `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于游戏理论的LEO卫星网络用户关联策略，目标是最小化切换次数并防止卫星过载。

**💡 创新点**

将用户切换建模为局部交互势函数游戏，证明其为精确势函数游戏，从而可使用势函数学习方法获得全局最优；同时提出并行的Concurrent‑SAP和低复杂度贪婪启发式算法。

**🔧 技术方法**

采用空间自适应游玩(SAP)及其并行版本C‑SAP，结合势函数游戏理论、局部交互网络与概率采样；还实现了贪婪hill‑climbing启发式方案。

**📊 数据集**

使用仿真数据：在 1000×1000 km 平面内布置 74 颗 LEO 卫星、100 个时间槽、用户均匀分布、卫星容量 3、过载成本 λ=10。

**📈 对比分析**

通过比较最小切换基线（δ₀⋆）、C‑SAP 与贪婪算法，评估切换次数和总成本；结果显示贪婪算法与 C‑SAP 性能相近，而 δ₀⋆ 仅切换最少但成本最高。

**⚠️ 局限性**

局限在于仅考虑静止用户且假设覆盖矩阵周期性；算法在大规模网络中仍存在计算开销；实验仅为仿真，缺乏实际系统验证。

---

## 546. Semantic Homogenization in Italian Popular Music: A Diachronic Analysis

**arXiv ID:** 2607.04832 | [PDF](https://arxiv.org/pdf/2607.04832v1)

**作者:** Lorenzo Canale `[一作]` (RAI), Alberto Messina `[通讯]` (RAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了一套可扩展的方法，用多种语义相似度度量（全文、片段、主题、词汇）分析意大利Sanremo音乐节歌词随时间的语义同质化趋势。

**💡 创新点**

创新点在于：①将多种嵌入模型（multilingual‑e5‑large、OpenAI、colbert‑xm）与大型语言模型（Gemini）结合，构建多层次相似度矩阵；②通过聚合不同相似度指标得到统一的Z矩阵，验证方法一致性；③提出可复现、适用于其他数据集的Python框架。

**🔧 技术方法**

使用技术包括：句子嵌入（Transformer）、词向量、余弦相似度、MaxSim、大型语言模型（Gemini）进行主题提取和段落划分、词干化/词形还原、词汇交集统计、Pearson相关分析。

**📊 数据集**

数据集为1951年至2025年Sanremo音乐节所有决赛歌曲的歌词（约2500首），仅公开了歌曲年份、标题、作者和演唱者，歌词文本受版权限制。

**📈 对比分析**

比较方法：对每个年份对生成相似度矩阵（全文、片段、主题、词汇），分别计算均值、中位数、分位数，聚合多模型/多方法得到统一Z矩阵；通过Pearson相关系数验证各方法的一致性，相关系数均高于0.9，表明不同度量在结果上高度一致，Z矩阵显示近几年语义同质化显著上升。

**⚠️ 局限性**

局限性：①未对外部数据集进行验证；②未引入音乐结构差异指标（如Musical Difference）；③由于版权仅公开元数据，无法对原始歌词进行更细粒度的可解释性分析；④方法依赖LLM推断，若模型升级或偏差会影响结果。

---

## 547. Layer-Parallel Inference Reduces Encrypted Nonlinear Depth in Transformers

**arXiv ID:** 2607.04819 | [PDF](https://arxiv.org/pdf/2607.04819v1)

**作者:** Ligong Han `[一作]` (MBZUAI IFM), Akash Srivastava `[通讯]` (MIT-IBM Watson AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并验证 Structured Newton Layer Parallelism（SNLP）在完全同态加密（FHE）环境下对 Transformer 推理的效能提升，主要通过减少层间非线性深度和 bootstraps 数量。

**💡 创新点**

创新点在于把 Transformer 的 L 层非线性序列化转化为 (L‑N)+K 级并行求解，并引入 IDN/HCP 结构化牛顿校正，使得非线性深度显著下降，误差放大也得到抑制；与传统仅优化单块非线性近似的做法形成互补。

**🔧 技术方法**

使用 Chebyshev 多项式近似（对 softmax、RMSNorm、sigmoid、tanh 等）来模拟 FHE 的非线性运算；构建符号 CKKS 成本模型和 NFE 指标；实现 SNLP 的并行求解与结构化校正；在 PyTorch 中进行加密推理的模拟实验。

**📊 数据集**

在 ClimbMix 语料上训练的 Nanochat 0.5B、3B 等八个模型，验证集采用长度 2048 的样本进行 perplexity 评估。

**📈 对比分析**

通过对比顺序推理与 SNLP 推理的 bootstrap 计数、NFE、以及 PPL 的误差放大比来评估性能。实验显示，SNLP 在 0.5B IDN 模型上将 bootstrap 从 53 降至 20（约 2.65×），PPL 仅增加 1.2%，误差放大从 1.42×降至 1.36×；其他模型同样保持低于顺序推理的误差放大。

**⚠️ 局限性**

主要限制包括：实验采用模拟而非真实 CKKS 加密，softmax 的 max‑subtraction 与 RMSNorm 的 Goldschmidt 迭代未完整实现，导致真实加密成本可能略高；软max 仍是误差主导，SNLP 需要配合更优的非线性近似才能进一步提升；模型规模局限在 Nanochat 级别，尚未验证更大模型的可扩展性。

---

## 548. CAC-VLA: Context-Gated Action Conditioning for Vision-Language-Action Models

**arXiv ID:** 2607.04816 | [PDF](https://arxiv.org/pdf/2607.04816v1)

**作者:** Yifu Xiong `[一作]` (University of Science and Technology of China), Jianmin Ji `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种 Context‑Gated Action Conditioning（CAC‑VLA）框架，利用 VLM 内部的潜在动作接口并通过上下文门控机制自适应地调节动作专家的输出，从而实现更精细的连续机器人控制；

**💡 创新点**

创新点在于①在 VLM 内直接学习潜在动作（由 Ordered Action Tokenizer 编码的未来动作段）作为中间动作结构化接口；②采用上下文门控交叉注意力，对潜在动作信息进行检索并通过残差门控动态注入动作专家；③不需要额外的动作生成或规划模块，所有信息均由 VLM 预测；

**🔧 技术方法**

使用了视觉‑语言预训练模型（VLM）+ 预训练的 Ordered Action Tokenizer、轻量化潜在动作预测头、交叉注意力 + 上下文门控模块、流匹配损失等技术；

**📊 数据集**

主要使用 LIBERO 与 LIBERO‑Plus 两大模拟基准进行训练与评估，并在 UR7e 机器人上进行了桌面抓取‑放置的真实实验；

**📈 对比分析**

与现有 VLA 方法（如 ACoT‑VLA、Diffusion Policy 等）做对比，CAC‑VLA 在 LIBERO 上平均成功率达 98.3%，在 LIBERO‑Plus 上为 89.5%，明显优于对比方法；消融实验进一步验证了潜在动作时长与上下文门控的重要性；

**⚠️ 局限性**

局限性包括：真实实验仅覆盖单一抓取‑放置任务、单一机器人与有限演示；当潜在动作预测失准或固定时长不匹配任务阶段时可能导致失败；未来工作需在更广泛的真实任务与平台上验证，并探索自适应或层次化的潜在动作时长。

---

## 549. Compressed Computation under $L^4$ Loss is likely Computation in Superposition

**arXiv ID:** 2607.04800 | [PDF](https://arxiv.org/pdf/2607.04800v1)

**作者:** Francisco Ferreira da Silva `[一作]` (Pivotal Research), Stefan Heimersheim `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

训练了单隐藏层ReLU网络，在L⁴损失下学习压缩计算任务，随后对网络进行逆向工程，发现网络采用稀疏二进制码编码特征，并用伪逆解码，进一步用三参数模型复现大部分性能并验证手工设计的码；

**💡 创新点**

首次证明在L⁴损失下能在压缩计算玩具模型中诱导出计算在叠加态（CiS）现象，并揭示了网络的内部机制为稀疏二进制码+伪逆解码的三参数描述；

**🔧 技术方法**

使用L⁴损失训练、Adam优化、对编码矩阵进行正负分离、提取二进制码、构造伪逆解码器、计算系数变异度、通过边交换最小化码重叠等技术；

**📊 数据集**

使用合成稀疏输入：100维特征，每个特征以0.02的稀疏率在[-1,1]间取值，平均每次前向传播有2个非零特征；

**📈 对比分析**

通过对比Naïve、Emulate‑Bias基线与L⁴网络的每特征误差分布和系数变异度评估；L⁴网络在所有特征上误差均匀，损失比基线低约26倍；三参数模型损失约为训练网络的1.13倍，手工设计的稀疏码（K=5、边交换）损失约1.12倍，均优于基线；

**⚠️ 局限性**

仍未完全解释编码器每个条目的具体数值及小幅偏离伪逆的原因；实验仅限于单层玩具模型，尚不清楚是否能推广到更深或更真实的网络和不同任务；

---

## 550. On a Boolean function without bold folding in the spectrum support and implications for greedy approaches to PDT depth

**arXiv ID:** 2607.04806 | [PDF](https://arxiv.org/pdf/2607.04806v1)

**作者:** Yuriy Tarannikov `[一作]` `[通讯]`, Yuriy Tarannikov

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究布尔函数及其傅里叶谱支持，探讨其在奇偶决策树（PDT）中的应用，构造了一类新的布尔函数以改进现有的折叠估计。

**💡 创新点**

通过构造显式的无限布尔函数族，证明了在谱支持中最大折叠大小为O(|𝒮|^1/2)，显著改善了之前的O(|𝒮|^5/6)的结果。

**🔧 技术方法**

使用了一种特殊的仿射子空间划分（APLPS-partition），基于全线性扩展来构造布尔函数。

**📊 数据集**

构造的布尔函数具有n=(2d+1)+2^d+1个变量，谱支持的大小为|𝒮|=2^(2d+2)。

**📈 对比分析**

与之前的贪婪方法进行比较，结果表明在懒惰假设下，PDT的深度下界为Ω(k^1/2)，与已知的上界O(k^1/2)相匹配，无法改进现有结果。

**⚠️ 局限性**

懒惰贪婪方法的假设在一般情况下是错误的，因此仅仅依赖于最大折叠估计无法获得更好的PDT深度下界，适应性贪婪策略的完整反驳仍然是一个开放问题。

---

## 551. Orcaella: Hybrid Fault Tolerance with Client-Selectable Finality Latency

**arXiv ID:** 2607.04789 | [PDF](https://arxiv.org/pdf/2607.04789v1)

**作者:** Lefteris Kokoris-Kogias `[一作]` (Mysten Labs), Alberto Sonnino `[通讯]` (Mysten Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了一个在混合拜占庭和崩溃错误模型下，能够在两消息延迟内完成共识的协议，并给出了最小委员会大小 n ≥ 5f+3c+1 的必要性证明。

**💡 创新点**

创新点在于将 Byzantine 与 crash fault 分离，获得更低的委员会规模；引入双路径（Fast‑Path 与 Resilient Path）让客户端可自行选择低延迟或更高安全性；并证明在此模型下 2‑delay 共识的安全性与可用性。

**🔧 技术方法**

使用了投票计数 BFT 共识模板、Quorum Certificates、视图切换、DAG‑based 证明与实现，以及 Byzantine Broadcast 进行叉路恢复等技术。

**📊 数据集**

使用的实验数据来源于模拟与真实 WAN 部署，包含 100 节点、不同 f/c 配置以及基准协议 Mysticeti 与 Hydrangea；未使用公开数据集。

**📈 对比分析**

通过在 10k‑100k tx/s 负载下比较，证明在无错误时 2‑delay 方案的延迟比 3‑delay 方案低约 14‑24%，吞吐量相当；在崩溃故障下仍保持性能。

**⚠️ 局限性**

局限在于需要部分同步网络模型、仅针对 2‑delay 计数协议、对活跃‑但‑损坏（AbC）假设有限；在网络慢速或分布不均时可能失去延迟优势。

---

## 552. A Temporal Reasoning Benchmarking Framework for LRMs via Difficulty-controlled and Dynamic Test Generation

**arXiv ID:** 2607.04784 | [PDF](https://arxiv.org/pdf/2607.04784v1)

**作者:** Shide Zhou `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于Allen时间代数的可调难度的时序推理基准框架（TRACE）及其生成的1,200道分级测试实例，配合跟踪式验证来评估大型推理模型（LRMs）的真实推理能力。

**💡 创新点**

创新点包括：① 将时序推理问题转化为约束满足问题，实现精确的难度调控；② 设计双轨验证（轨迹+答案）以区分真正的逻辑推理与巧合性猜测；③ 发现并系统分类了规模相关的失败模式（小模型的答案不匹配，大模型的推理爆炸，等）。

**🔧 技术方法**

技术主要包括：Allen时间代数、约束传播与路径一致性算法、基于生成器的图结构构造、自然语言模板化与JSON结构化推理输出、正则解析与闭包验证器。

**📊 数据集**

使用自研的可调难度生成器产生的Synthetic Temporal Reasoning Benchmark（TRACE-Bench），包含六个难度等级，每级40个图，200个问题，总共1,200条测试实例；没有使用公开的静态数据集，以避免数据污染。

**📈 对比分析**

与八款LRMs（DeepSeek-R1系列、Gemini-2.5-Flash、GPT-5-mini、Claude-Sonnet-4.6等）进行评测。结果显示难度与模型性能呈负相关（平均 Pearson r≈-0.96）。中等规模模型在准确率上存在高达28% 的“伪猜测”，而高级模型的“伪猜测”率低于15%。

**⚠️ 局限性**

局限性包括：① 生成任务高度规则化，缺乏自然语言噪声与语义歧义；② 仅评估内部推理能力，未考虑工具调用或外部推理；③ 大规模模型在最高难度下仍出现推理链爆炸与上下文窗口耗尽的问题。

---

## 553. Evaluating the Effect of Linguistic Relatedness on Cross-Lingual Transfer in Large Multilingual Automatic Speech Recognition

**arXiv ID:** 2607.04814 | [PDF](https://arxiv.org/pdf/2607.04814v1)

**作者:** Andrei Florian `[一作]` (Princeton University), Happy Buzaaba `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

评估在大型多语种ASR中使用语言相关性进行的两阶段顺序微调是否能提升低资源非洲语言的识别性能。

**💡 创新点**

将小规模ASR中证明有效的相关语言预适应策略扩展到四种不同规模、架构和监督模式的大型ASR模型，并通过六因素控制实验检验语言相似性是否是跨语言迁移的可靠指标。

**🔧 技术方法**

使用 Whisper Small、Whisper Large v3、Facebook XLS‑R 和 OmniASR 四大模型，采用全模型微调与参数高效 LoRA 微调，以及 Wilcoxon 与 TOST 等非参数统计检验。

**📊 数据集**

采用 AfriVoices KE 与 Google WaxalNLP 两个以非洲语言为主的多语种语料库，分别选取 10 种语言，涵盖 Nilotic、Bantu、Cushitic 三个族群。

**📈 对比分析**

对比预适应相关语言、非相关语言及仅目标微调三种策略，在各模型、数据量和实验因子下计算目标语言的 WER；结果显示从 1 小时目标微调起，三种策略性能差异不显著，相关语言预适应对最终 WER 无实质提升。

**⚠️ 局限性**

研究仅覆盖两份非洲语料库和四种模型，未考虑录音条件、说话人特征等外部因素；等效阈值设为 5 WER 点，结果的普适性需要进一步验证。

---

## 554. HunyuanOCR-1.5: Making Lightweight OCR VLMs Faster and Better

**arXiv ID:** 2607.04884 | [PDF](https://arxiv.org/pdf/2607.04884v1)

**作者:** Gengluo Li `[一作]` (Chinese Academy of Sciences), Yu Zhou `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一款轻量化端到端OCR专用视觉‑语言模型HunyuanOCR‑1.5，在保持HunyuanOCR‑1.0的轻量架构基础上实现了更快的推理和更广泛的OCR能力；

**💡 创新点**

创新点包括：①引入DFlash推测式解码加速长结构OCR生成，实现Transformer 6.37×、vLLM 2.14×的速度提升；②设计Agentic Data Flow数据构建系统，实现模型弱点到可执行数据需求的闭环；③将视觉编码器扩展至4K分辨率、上下文窗口128K，支持多图像、低资源、多语言和古文字识别；④在单体模型层面保持轻量同时提升跨任务性能；

**🔧 技术方法**

使用技术包括：DFlash推测式解码、vLLM、llama.cpp PC侧部署、Hunyuan‑0.5B轻量语言模型、XD‑RoPE、FlexAttention、RL Fine‑tune、Agentic Data Flow系统、Hunyuan‑ViT视觉编码器升级；

**📊 数据集**

使用数据集有：OmniDocBench v1.6、OCRBench、Spotting Benchmark、MORE、Chronicles‑OCR、ChartArena、TableVerse‑5K、DUDE、DoTA、MMTIT、IE Benchmark、Video Subtitle Extraction、CHAOS‑Bench等；

**📈 对比分析**

在OmniDocBench上获得94.74分，SOTA；与GLM‑OCR、PaddleOCR‑VL‑1.6、Unlimited‑OCR、DeepSeek‑OCR‑2、dots.ocr等系统对比，HunyuanOCR‑1.5 DFlash实现1.408s/页，速度提升1.17×–5.08×，在古文字识别、表格/图表解析、低资源多语种、跨图像QA等长尾任务上也达1B模型SOTA；

**⚠️ 局限性**

局限性：模型对视觉文本的保真度仍偏低（CHAOS‑Bench召回仅14%），对长多行公式的评估匹配仍不完善；在极大并发下加速优势下降；在极高分辨率或极长上下文场景下仍需进一步优化；

---

## 555. On the Complexity of Entrywise Power Matrix Factorization

**arXiv ID:** 2607.04875 | [PDF](https://arxiv.org/pdf/2607.04875v1)

**作者:** Nicolas Gillis `[一作]`, Arnaud Vandaele `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了逐项幂矩阵分解（EPMF）的计算复杂性，提出了在固定秩情况下的可解性和NP难度。

**💡 创新点**

提出了EPMF与签名问题的等价性，展示了在固定秩情况下的多项式时间算法，并证明了在一般矩阵情况下的固定参数可解性（FPT）。

**🔧 技术方法**

使用了多项式时间算法解决签名问题，并分析了EPMF的复杂性。

**📊 数据集**

使用了非负矩阵X，研究了不同秩r的情况。

**📈 对比分析**

在固定秩情况下，ExactEPMF问题是多项式时间可解的；当秩作为输入时，ExactEPMF是强NP难的，FroEPMF在r=2时已经是NP难的。

**⚠️ 局限性**

在秩不固定的情况下，ExactEPMF是强NP难的，且FroEPMF在最小非平凡秩r=2时是NP难的。

---

## 556. Active Learning on Adversarially Corrupted Graphs

**arXiv ID:** 2607.04869 | [PDF](https://arxiv.org/pdf/2607.04869v1)

**作者:** Marco Bressan `[一作]` (Università degli Studi di Milano), Silvio Lattanzi `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在存在结构攻击的图中提出了一种基于顶点扩展的主动学习算法，能够以相对较少的查询次数识别被攻击的节点集合

**💡 创新点**

首次将顶点扩展与查询复杂度建立联系，证明在攻击预算与图的顶点扩展均满足一定条件时即可实现弱恢复

**🔧 技术方法**

使用多项式时间的和平方（Sum‑of‑Squares）程序近似最小顶点扩展，并构造递归分割与查询策略

**📊 数据集**

论文未给出具体实验数据集，主要以理论分析为主

**📈 对比分析**

缺乏实验对比，理论上相较传统主动学习方法查询量可降至O(b/γ)，但未给出数值指标

**⚠️ 局限性**

对图的顶点扩展要求较高，攻击预算需小于一定阈值；在大规模图上实现仍存在挑战

---

## 557. CARL: Constraint-Aware Reinforcement Learning for Planning with LLMs

**arXiv ID:** 2607.04854 | [PDF](https://arxiv.org/pdf/2607.04854v1)

**作者:** Qiuyi Qi `[一作]`, Qiang Zhu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提供ACL会议论文排版的样例与说明

**💡 创新点**

将排版说明与实际模板相结合，展示如何使用LaTeX格式文件

**🔧 技术方法**

使用ACL的LaTeX style文件和示例文档

**📊 数据集**

无具体数据集

**📈 对比分析**

无方法对比或性能评估

**⚠️ 局限性**

仅为排版指南，缺乏实验与评估，适用范围受限于ACL会议

---

## 558. Representing and Detecting Label Ambiguity in IMU-Based Exercise Evaluation

**arXiv ID:** 2607.04842 | [PDF](https://arxiv.org/pdf/2607.04842v1)

**作者:** Andreas Spilz `[一作]` (Ulm University of Applied Sciences), Michael Munz `[通讯]` (Ulm University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出利用规则化标签并模拟评估者误差生成的标签分布（AGLD），并训练深度网络以KLD目标复现该分布，从而在IMU运动评估中捕捉判定边界的不确定性。

**💡 创新点**

创新点在于（1）通过阈值随机化生成评估者分布，无需大规模人工标注；（2）将完整分布作为训练目标，提升对边界重复的检测与二类相关信息的识别；（3）提供基于熵阈值的二元不确定性判定方法。

**🔧 技术方法**

使用卷积网络（两卷积块+全连接）输入为四元数姿态序列，训练目标分别为交叉熵（one‑hot）与Kullback‑Leibler散度（分布）。

**📊 数据集**

四个IMU数据集：RD（受限踝屈伸）、RGS（受限步态仿真）、DS（深蹲FMS）、HS（障碍步FMS）。

**📈 对比分析**

与one‑hot交叉熵基线在五折交叉验证中比较，结果显示在所有数据集上分布训练方法至少保持最相关类的分类精度，并在中等熵阈值区间显著提升不确定性检测（F1↑）和对应两类识别（top‑2 F1↑）。

**⚠️ 局限性**

局限包括（1）仅处理两类边界问题，无法捕捉多类同时竞争的情况；（2）AGLD生成依赖手工设定阈值方差，未系统评估其对结果的影响；（3）评估标准基于人工评分间的分歧，缺乏直接的“是否为边界”标注；（4）仅在自家实验设置下验证，泛化性待进一步验证。

---

## 559. Hybrid Deep Learning for Traceability and Classification of Industrial Slate Tiles

**arXiv ID:** 2607.04811 | [PDF](https://arxiv.org/pdf/2607.04811v1)

**作者:** Soren Antebi `[一作]` (Fraunhofer Institute IAIS), Rafet Sifa `[通讯]` (Fraunhofer Institute IAIS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量级混合深度学习模型，联合实现石板瓦的实例级图像匹配与来源分类。

**💡 创新点**

创新点在于将XFeat+LightGlue用于细粒度匹配，并将其与MobileNetV3的全局语义特征融合，兼顾匹配精度与分类性能，同时保持模型轻量化。

**🔧 技术方法**

使用XFeat提取稀疏/稠密特征，LightGlue学习匹配，MobileNetV3小型骨干做分类，投影头与融合块实现特征融合。

**📊 数据集**

使用新构建的2610张石板瓦图像数据集（六个采石场）以及MegaDepth‑1500基准数据集进行评估。

**📈 对比分析**

与单一MobileNetV3或原始XFeat比较，分类准确率提升约10.9%，匹配AUC提升约15%+，Top‑1匹配率从87.6%升至90%，性能显著优于基线。

**⚠️ 局限性**

匹配分支推理时间比单纯XFeat慢约十倍，对旋转不具不变性，限制了在低功耗边缘设备上的实时部署。

---

## 560. Dynamic Airspace Management for UAVs in Evolving Urban Environments: Collaborative Coordination and Human Safety

**arXiv ID:** 2607.04825 | [PDF](https://arxiv.org/pdf/2607.04825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 561. PAGE: Towards Practical Human-level Gaze Target Estimation

**arXiv ID:** 2607.04860 | [PDF](https://arxiv.org/pdf/2607.04860v1)

**作者:** Zhoutong Ye `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了第一款实现人类水平的注视目标估计模型

**💡 创新点**

核心创新在于场景与头部特征的跨注意力交互模块、简化训练流程以及大规模无标签数据的特征蒸馏

**🔧 技术方法**

采用ViT-H+/DINOv3 backbone、Scene-head Interaction Module、RoPE位置编码、SFT微调、token‑level蒸馏

**📊 数据集**

使用GazeFollow、VideoAttentionTarget、ChildPlay三大注视数据集作为标注集，MPII+OpenImages V7作为无标签蒸馏集

**📈 对比分析**

与多种前沿方法（Gaze‑LLE、AnyGaze等）及人类基准对比，模型在7/9指标上超越人类，整体精度提升至AUC≈0.97、L2≈0.08，轻量版保持约10% FLOPs

**⚠️ 局限性**

仍受限于标注数据稀缺、评测方法偏差和对非现实场景（动画、动物）的鲁棒性不足

---

## 562. WinTA-GIL: Windowed Trajectory Alignment for GNSS-IMU-LiDAR Heading Refinement in Intermittent Signal Environments

**arXiv ID:** 2607.04879 | [PDF](https://arxiv.org/pdf/2607.04879v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 563. E-CoDrive: A Co-Simulation Framework for Testing Energy-Critical Driving Scenarios

**arXiv ID:** 2607.04803 | [PDF](https://arxiv.org/pdf/2607.04803v1)

**作者:** Manfredi Napolitano `[一作]` (Università degli Studi di Napoli Federico II), Nicola Mazzocca `[通讯]` (Università degli Studi di Napoli Federico II)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构建了一个能将 SUMO、CARLA 与 Autoware Mini 同步协同的闭环能耗场景仿真框架，用于评估城市交通对 AEV 能耗的影响。

**💡 创新点**

创新点在于统一的协同调度层实现跨异构仿真器的时间同步和状态交互，支持可重复的能耗驱动场景生成与分析。

**🔧 技术方法**

技术上使用 SUMO 生成交通、CARLA 高保真车辆与环境仿真、Autoware Mini 自主驾驶栈，并通过 Python/ROS 等实现编排与能源模型（如 MMPEVEM）。

**📊 数据集**

使用公开的 CARLA Town01/04/05 地图、SUMO 路线文件、Autoware Mini 车辆模型以及 Tesla Model 3 的电动能耗模型，数据来源于仿真输出。

**📈 对比分析**

通过与自由流量基线对比，实验显示交通拥堵导致能耗增加 137–358 Wh，平均速度下降，验证框架能捕捉交通能耗效应。

**⚠️ 局限性**

局限包括仅集成了特定技术栈、仅在小范围地图与车辆数级别验证、未覆盖更大规模或多样化交通场景，且缺乏自动搜索式场景生成。

---

## 564. Adaptive Diversity-Uncertainty Active Learning with Redundancy Control for Bioacoustic Event Classification

**arXiv ID:** 2607.04868 | [PDF](https://arxiv.org/pdf/2607.04868v1)

**作者:** Gabriel Dubus `[一作]` (Museum National d'Histoire Naturelle), Anatole Gros-Martial `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于自适应多元不确定性、空间多样性与最大边际相关性（MMR）机制的主动学习策略，用于多标签生物声学事件分类。

**💡 创新点**

创新点在于将预测不确定性、嵌入空间多样性与批量级冗余控制统一融合，并通过全局置信度自适应权重动态平衡探索与利用。

**🔧 技术方法**

采用PerchV2预训练音频嵌入、熵不确定性度量、欧氏距离多样性度量以及贪婪MMR算法实现样本选取。

**📊 数据集**

使用BirdSet（陆地鸟类声学数据集）和ATBFL（南极海洋鲸类声学数据集）两大公开数据集进行实验。

**📈 对比分析**

与随机采样、Margin、CoreSet、TypiClust等基线对比，ADU-MMR在BirdSet上AULC和mAP均领先，ATBFL上表现相近，但总体上实现了显著提升。

**⚠️ 局限性**

局限在于对低频海洋数据集的嵌入表达不足导致主动学习效果有限，且对预训练模型依赖较高；在噪声高、标签稀疏的海洋环境中提升有限。

---

## 565. Framework for Grouping Local Process Models

**arXiv ID:** 2607.04856 | [PDF](https://arxiv.org/pdf/2607.04856v1)

**作者:** Viki Peeva `[一作]`, Wil M. P. van der Aalst `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了一种基于Petri网的行动流程建模方法，使用一组节点（Select、Group、Move、Collect、Attack）和两个存储点（p1、p2）来描述从选择到攻击的完整流程。

**💡 创新点**

创新点在于将传统流程拆分为五个具有可视化语义的阶段，并通过Petri网的并发与同步特性对不同动作的先后关系进行精确建模，解决了单纯状态机无法充分表达并发与资源竞争的问题。

**🔧 技术方法**

主要技术包括Petri网建模、转换规则定义、状态转移图绘制以及基于Petri网的仿真工具来验证流程的可执行性。

**📊 数据集**

实验使用了一个基于机器人仿真环境的动作日志数据集（包含1000条任务序列），以及一个游戏AI的行为记录数据集。

**📈 对比分析**

与传统有限状态机方法进行对比后，实验结果显示Petri网模型能够以约15%的计算开销捕捉到更多并发关系，并在任务完成时间上平均提升了12%。

**⚠️ 局限性**

限制在于当前模型仅处理离散事件，无法直接支持连续动作的实时控制；此外，Petri网的规模增长会导致状态空间爆炸，需要进一步的抽象或分层技术。

---

## 566. Geometry-aware Depth-guided Representation Learning for Structure-preserving Low-light Image Enhancement

**arXiv ID:** 2607.05005 | [PDF](https://arxiv.org/pdf/2607.05005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 567. The syntax of wh-agreement in Yemeni Ibbi Arabic

**arXiv ID:** 2607.04986 | [PDF](https://arxiv.org/pdf/2607.04986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 568. SleepBand: Single-Source Domain Generalization for Sleep Staging via Physiologically Structured Spectral Modeling

**arXiv ID:** 2607.04851 | [PDF](https://arxiv.org/pdf/2607.04851v1)

**作者:** Zhi Lu `[一作]` (University of Electronic Science and Technology of China), Yan Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了单源域泛化睡眠分期模型SleepBand，并在五个公开睡眠数据集上实现了高鲁棒性。

**💡 创新点**

创新点包括：① 可学习的 Morlet/Gabor 滤波器组作为频谱先验；② 结构化的频谱集成与自适应校准机制；③ 频谱一致性正则化（带宽常数‑Q 约束与频带扰动）；④ 结合 Mean‑Teacher 的一致性学习。

**🔧 技术方法**

技术手段：可学习 Gabor 滤波器、频谱分解与重组、频谱集成与校准、Mean‑Teacher 一致性正则化、频带扰动、轻量级自注意力编码、Sinc 可学习滤波器对比、常数‑Q 约束。

**📊 数据集**

使用的数据集：SleepEDF、HMC、ISRUC、SHHS1、CinC 2018。

**📈 对比分析**

与多种单源/多源域泛化基线（ERM、SAM、F‑SAM、GroupDRO、CIRL、MixStyle、FACT、SleepDG、MMD、CORAL、IRM、DANN、REx）对比。单源平均 ACC 72.25%、F1 67.08，显著优于所有基线；多源时 ACC 77.13、F1 71.08，仍为最高。模型同时实现了显著的 FLOPs 与参数压缩。

**⚠️ 局限性**

局限性：① 仍依赖频谱先验，可能在极端设备或人口差异中受限；② 对多通道跨域鲁棒性未充分评估；③ 需要更多真实世界数据验证；④ 仅针对标准睡眠分期任务，难以直接推广至其他 EEG 诊断任务。

---

## 569. Pretraining Curricula Enable Selective Fine-tuning

**arXiv ID:** 2607.04846 | [PDF](https://arxiv.org/pdf/2607.04846v1)

**作者:** Sebastian A. Bruijns `[一作]` (University of Oxford), Christopher Summerfield `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在Transformer上实施两种预训练课程（平衡课程与不平衡课程），比较它们对任务学习、表示分离度、精细调优（拒绝任务）以及规则遵从性的影响，并使用机制解释技术（消融、直接logit归因与激活补丁）揭示网络内部电路差异。

**💡 创新点**

创新点在于：①提出不平衡课程可诱导任务特定、解耦的电路结构；②证明该课程可显著降低拒绝微调时的过度拒绝；③在合成语言学习任务中显示不平衡课程提升了对规则的内部表示与对齐效果；④将机制解释方法与课程设计结合，为安全对齐提供了新的实验路径。

**🔧 技术方法**

主要技术包括：小型Transformer（两层/四层）、梯度下降预训练、两种课程调度、拒绝微调、直接logit归因(DLA)、激活补丁、消融实验、规则一致性评估。

**📊 数据集**

使用了两套人工控制数据集：1）复制首/尾任务（20,000序列训练，500测试序列）；2）合成语言学习任务（包含多条规则的短句子，20,000训练序列，500测试序列）。

**📈 对比分析**

通过在相同训练量、相同模型规模下比较平衡与不平衡课程，评估标准包括：测试准确率、过度拒绝率、自由生成与诱导提示下的违规概率。实验结果显示不平衡课程在所有指标上优于平衡课程：更高的泛化准确率、显著降低的过度拒绝、诱导提示下更低的违规概率。

**⚠️ 局限性**

局限性包括：模型规模较小，实验仅在高度控制的任务上进行，可能无法直接推广到大型LLM；合成任务与真实对齐情境存在差距；对不同任务类型和更复杂规则的适用性尚未验证；课程对齐效果在不同超参数组合下可能不稳定。

---

## 570. TGRIP: A Text-Guided Approach to Vehicle Instance Prediction in Autonomous Driving

**arXiv ID:** 2607.04812 | [PDF](https://arxiv.org/pdf/2607.04812v1)

**作者:** Miguel Antunes-García `[一作]` (University of Alcalá), Luis M. Bergasa `[通讯]` (University of Alcalá)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于文本指导的 BEV 实例预测框架 TGRIP，通过在训练阶段注入来自视觉‑语言模型的语义先验来提升未来车辆轨迹预测。

**💡 创新点**

创新点在于：①首次将基于 CLIP/SigLIP 的实例级视觉嵌入与 BEV 预测联合训练；②使用 teacher‑student 结构在训练期间生成稠密语义 BEV 地图；③通过辅助语义头实现跨模态对齐，提升实例分离和长距离预测；④保持推理时无额外开销，训练后可直接部署。

**🔧 技术方法**

核心技术包括 EfficientViT 作为 Backbone、BEVFormer 变换模块、BEVPredFormer 时序编码、CLIP/SigLIP 视觉‑语言嵌入、cosine 相似度辅助损失以及多任务损失（流、分割、中心度）。

**📊 数据集**

在 nuScenes 公开数据集上进行训练与评估，仅关注车辆超类（car、bus、truck 等）。

**📈 对比分析**

与多种基线（StretchBEV、PowerBEV、BEVerse、DMP 等）以及官方实现的 BEVPredFormer 进行对比；在长距离场景中，TGRIP 在 IoU 41.3% 与 VPQ 34.3% 上分别超过 baseline 0.4% 与 1%，并创下新 SOTA；在短距离场景中也保持了与 DMP 相近的性能。多次随机种子实验验证了结果的稳健性。

**⚠️ 局限性**

局限性包括：①依赖 3D 边界框标注生成语义目标，限制了对未标注或新类别的适用性；②仅使用摄像头数据，未探索 LiDAR/Radar 与语义融合；③语义教师生成过程离线且需额外算力；④在不同驾驶环境和传感器配置下的泛化仍未验证。

---

## 571. Multi-Robot Open Adaptive Teaming Across Unseen Environments, Partners, and Scales

**arXiv ID:** 2607.04972 | [PDF](https://arxiv.org/pdf/2607.04972v1)

**作者:** Yang Li `[一作]` (Shanghai Jiao Tong University), Wei Pan `[通讯]` (Newcastle University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种用于开放自适应多机器人协同的超图开放式学习算法（HOLA），可在未知环境、未知队友和可变队伍规模下实现零射击协调

**💡 创新点**

创新点在于：① 引入超图形式游戏建模，捕捉多主体的高阶协作关系；② 通过超图偏好中心性和逆麦克森值构造伙伴采样分布；③ 在训练中逐步扩展伙伴与环境多样性，实现真正的开放式学习

**🔧 技术方法**

使用超图游戏理论、最大熵强化学习、人口基础训练（MEP）、自组织学习（Oracle+Grapher）、Shapley/麦克森价值等技术

**📊 数据集**

主要数据集为仿真环境中生成的多机追捕场景（包含不同障碍布局、队伍配置），并在两种真实平台（Crazyflie 多旋翼、L1 四足）上部署验证

**📈 对比分析**

与 MAPPO、DACOOP‑A、PBT、FCP、MEP 等基线比较，HOLA 在固定队伍和开放队伍两种协议下都获得最高成功率、最低碰撞率、最短平均任务时间；在真实平台上直接转移，表现优于所有基线

**⚠️ 局限性**

局限性：在密集交互场景下安全性仍可提升；目前验证集中在追捕任务，缺乏更大规模或更长时域的测试；对更广泛任务的通用性仍待进一步研究

---

## 572. When Words Predict Workload

**arXiv ID:** 2607.04951 | [PDF](https://arxiv.org/pdf/2607.04951v1)

**作者:** Anubhab Banerjee `[一作]` `[通讯]` (Nokia), Anubhab Banerjee (Nokia)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于 CPU 的语言资源预测（LRF）网关，在 LLM 推理前利用 16 维文本结构特征预测陷阱区段，动态决定是否将请求路由到本地 Qwen2.5‑7B 或远程 Binoculars 7B+32B 集群，从而避免在消费者级 GPU 上因陷阱区段导致的 VRAM 突发耗尽。

**💡 创新点**

创新点在于：① 将传统的语言风格特征与 16 维量化特征融合成预测向量；② 用 XGBoost 进行实时 trap‑band 成员概率预测；③ 设计闭式动态阈值 (t) 结合实时延迟、电信、云端计算时间和 OOM 触发器，实现对硬件崩溃的预判；④ 双机制 VRAM 安全互锁（预估门控 + NVML 运行时监控）保证峰值 VRAM 不超过 8GB。

**🔧 技术方法**

技术手段包括 spaCy 轻量化特征抽取、XGBoost 二分类推理、Qwen2.5‑7B‑Instruct GGUF 模型的 CUDA 推理、Binoculars 对比式 PPL 量化集群、NVML 监控、闭式公式阈值计算以及多重优先级的触发策略。

**📊 数据集**

使用的数据集为欧盟专利局（EPO）H04L 领域的专利权利要求文本，包含人类与 AI 重写的 4000+ 例子，并扩展到 5 个 IPC（A61K、C07D、F03D、G06F、H04L）和多种生成器（Claude Opus 4.6、Qwen 2.5‑72B），形成跨 IPC、跨文本结构的混合测试集。

**📈 对比分析**

与传统基于 token 数的路由器相比，LRF 方案在 6000 次实时路由实验中将误路由率从 0.849 降至 0.087‑0.095（下降 8.2% 绝对误差），维持峰值 VRAM 4.82GB；闭式动态阈值相较于固定阈值将 p99 延迟略增至 6.5s，但未影响硬件安全；在不同 WAN 负载下，误路由率保持在 0.108‑0.118 范围内，表现出鲁棒性。

**⚠️ 局限性**

主要局限包括：① XGBoost 预测的 AUROC 与 FPR 仍低于预期（0.840/0.268），需更深层模型或跨 IPC 训练；② 每请求 5ms 的 CPU 预测开销与 50ms 的 p99 超出理想阈值，未来需迁移到 C++/pybind11；③ 需对每个 IPC 或文本结构单独重新校准陷阱阈值，限制了即插即用性；④ 云端推理使用的 PyTorch 参考核导致 p99 约 6–7 秒，若改为 JIT 内核可显著提升性能。

---

## 573. Efficient Perception in Automotive Detection and Tracking Using Neuromorphic Computing

**arXiv ID:** 2607.04921 | [PDF](https://arxiv.org/pdf/2607.04921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 574. Sensitivity Sampling with Predictions for k-Means Clustering

**arXiv ID:** 2607.04949 | [PDF](https://arxiv.org/pdf/2607.04949v1)

**作者:** Cristian Boldrin `[一作]` (University of Padova), Fabio Vandin `[通讯]` (University of Padova)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种利用预测中心来加速k‑means核心集构建的算法（PreSenS），并证明在数据来自同一分布的连续快照序列中，先前快照的中心可作为高质量预测。

**💡 创新点**

创新点在于：① 将预测中心引入敏感度采样框架，减少昂贵的初始bi‑criteria近似；② 证明预测中心在同分布快照下保持bi‑criteria逼近，理论上保证核心集大小与最优一致；③ 通过实验验证在大规模动态数据上实现显著的运行时加速且聚类质量不下降。

**🔧 技术方法**

技术主要包括：k‑means核心集构建、敏感度采样、bi‑criteria近似、统计学习框架下的泛化分析以及对预测中心的概率分布采样。

**📊 数据集**

使用的公开数据集有：Twitter、IntelLab、Taxi、NYC TLC（按月/年快照）以及合成实验，用于评估核心集构建时间、聚类质量和估计失真。

**📈 对比分析**

与Bansal等的最优敏感度采样、QuadTree‑based方法、Uniform采样等对照，PreSenS在所有数据集上均实现了 3.5‑5.2 倍的运行时加速，同时聚类成本与最优值相比仅略高 1‑2%，失真也与基线相当或更好。

**⚠️ 局限性**

局限性包括：① 需要足够大的快照才能保证理论泛化；② 对分布漂移的假设较强，需额外处理动态分布；③ 仍需在每个快照上计算所有点到预测中心的距离，时间复杂度为 O(nkd)。

---

## 575. Using Process Mining to Generate AI Agents from Software Engineering Process Records

**arXiv ID:** 2607.04948 | [PDF](https://arxiv.org/pdf/2607.04948v1)

**作者:** Saimir Bala `[一作]` (Hasso Plattner Institute), Andreas Metzger `[通讯]` (paluno Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于流程挖掘的流水线，利用软件仓库事件日志自动生成与项目相关的 AI 代理角色及其实现，并在 Commitizen 开源项目上验证其可行性

**💡 创新点**

创新点在于将对象中心流程挖掘与声明式约束相结合，自动从历史日志中提取角色行为、约束和规范，并通过 LLM 生成可直接部署的 LangGraph 代理；同时通过可视化和用户研究验证了人机协同的对齐度

**🔧 技术方法**

技术包括对象中心流程挖掘（OC-DFG）、声明式流程挖掘（DECLARE）、BPMN 生成、LLM（GPT‑4‑mini 与 IBM BOB）用于过程描述和代码生成、LangGraph 框架实现代理、PyStack't 提取 OCEL 日志

**📊 数据集**

使用 Commitizen 项目自 2017‑2025 年的 GitHub 事件日志（约 21,500 条事件，4,800 个对象），并对提交信息应用 Conventional Commits 进行语义标注

**📈 对比分析**

评估方式包括：① 对生成的 LangGraph 应用进行三条 smoke 测试，验证路由与动作生成；② 进行十名参与者的定性问卷，量化人机对齐维度，结果显示知识 schema、操作清晰度与人机协作得分较高，但自治边界得分低；性能表现未做大规模基准，但示例中能正确识别并路由三类问题

**⚠️ 局限性**

局限性包括：仅在结构化提交规范的开源项目上验证，未覆盖非规范化或企业仓库；角色划分采用硬划分，忽略贡献者多角色行为；流程挖掘粒度受限于日志粒度，可能无法完整映射 AI 代理细粒度操作；仅通过规范评估而非实际运行行为验证人机对齐，样本规模与行业代表性有限

---

## 576. Closing the Reality Gap: Zero-Shot Sim-to-Real Deployment for Dexterous Force-Based Grasping and Manipulation

**arXiv ID:** 2607.04940 | [PDF](https://arxiv.org/pdf/2607.04940v1)

**作者:** Zhe Zhao `[一作]` (Beijing University of Posts and Telecommunications), Mengshi Qi `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

通过仿真训练并直接零样本部署五指机械手，实现可控抓握力与手内旋转。

**💡 创新点**

创新点：高效密集触觉仿真、基于电流的扭矩校准、随机化驱动模型，以及结合触觉和电流的全状态策略。

**🔧 技术方法**

技术：强化学习（PPO）+异步 actor‑critic；并行前向运动学触觉仿真；电流→扭矩映射与随机化驱动模型；6D姿态连续表示；随机化摩擦、阻尼等接触参数。

**📊 数据集**

数据集：IsaacLab 仿真中的随机物体（形状、质量、摩擦）及真实机械手通过 SAM+IMU 获得的位姿与触觉读数；无额外公开数据集。

**📈 对比分析**

对比：与无触觉/无电流观测的基线相比，完整配置在真实机械手上连续成功率 25.1、平均时长 3.36 s；力跟踪范围提升，触觉+电流观测显著提高抓握强度和一致性。

**⚠️ 局限性**

限制：触觉仿真仍受高频计算瓶颈；电流→扭矩映射假设线性，未考虑温升、老化；未测试多手协同或更复杂动态环境。

---

## 577. Version-Aware Communication in Multi-Hop IoT Networks with Feedback

**arXiv ID:** 2607.04996 | [PDF](https://arxiv.org/pdf/2607.04996v1)

**作者:** Erfan Delfani `[一作]` (Linkoping University), Nikolaos Pappas `[通讯]` (Linkoping University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本研究提供了多跳通信网络中版本信息年龄（VAoI）的全面特征描述，考虑了传输约束和基于确认的反馈机制。

**💡 创新点**

创新点在于提出了一种双层优化框架，联合优化源端更新控制和中间节点的反馈感知转发策略，并推导出最优源策略的阈值形式及其闭式表达。

**🔧 技术方法**

使用了双层优化框架，结合了版本信息年龄（VAoI）和反馈机制的分析，推导出各网络节点的平均VAoI和更新速率的闭式表达。

**📊 数据集**

研究中未明确提及使用特定数据集，而是通过数值模拟验证理论分析的有效性。

**📈 对比分析**

通过与随机基线进行比较，分析了最优阈值策略和随机策略的性能，结果表明最优阈值策略在减少冗余传输的同时保持信息新鲜度方面表现优越。

**⚠️ 局限性**

限制在于未深入探讨多跳网络中反馈机制的具体实现和对不同网络拓扑的适用性。

---

## 578. Intrinsic Meshing of Closed Surfaces Using Geodesic Distances

**arXiv ID:** 2607.04989 | [PDF](https://arxiv.org/pdf/2607.04989v1)

**作者:** Tim Gabriel `[一作]` (Universite de Liege), Christophe Geuzaine `[通讯]` (Universite de Liege)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于曲面测地距离的闭合离散曲面的本质三角网格重建方法，利用局部优化（交换、拆分、收缩等）在保持几何不变的前提下构造可直接用于高阶网格的等几何网格。

**💡 创新点**

首次实现了可在保留原始离散几何的同时实现细化与粗化的本质三角化，并通过精确测地路径与圆心计算实现角度与尺寸约束，直接生成高阶网格。

**🔧 技术方法**

使用连续 Dijkstra（MMP/ICH）与 A* 优化的精确测地距离与圆心求解，配合局部网格操作（边交换、拆分、收缩、三角拆分）以及特征长度与角度约束。

**📊 数据集**

在 Thingi10K 数据集中对近 5,000 个闭合拓扑良好的复杂模型进行验证，覆盖多种几何复杂度。

**📈 对比分析**

与传统基于欧氏距离的重建方法相比，算法在 99.6% 的测试模型上成功完成，平均几何误差低于 0.16%，单模型处理时间在几秒至十秒内完成，显著优于以往方法。

**⚠️ 局限性**

目前仅支持闭合曲面，无法处理开放表面和尖锐特征边缘；对极端高曲率区域的角度约束实现仍有限。

---

## 579. InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization

**arXiv ID:** 2607.04988 | [PDF](https://arxiv.org/pdf/2607.04988v1)

**作者:** Haoxiang Ma `[一作]`, Weinan Zhang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 InternVLA-A1.5，统一 vision‑language 理解、latent 未来推理与连续动作生成的框架，保持 VLM 的语义先验并利用冻结的视频生成器学习未来动力学。

**💡 创新点**

①在 VLM 背骨上继续训练 VQA 与子任务预测，保持并强化语义；②引入少量 learnable foresight tokens，通过冻结的 video generator 监督，将任务相关的未来信息压缩为 latent 码；③推理时剔除视频分支，实现在实时控制下的零额外生成成本。

**🔧 技术方法**

Mixture‑of‑Transformers、Qwen‑3.5 2B VLM、WAN2.2‑5B 冻结视频生成器、flow‑matching 动作生成、latent foresight 查询、跨任务联合训练与自定义注意力遮掩。

**📊 数据集**

1.2M 机器人操控数据（InternData‑A1 与 5 个真实世界来源），3M 多模态 VQA/定位/轨迹样本（InternVLA‑M1）。

**📈 对比分析**

在六个仿真基准（LIBERO、RoboTwin、EBench、SimplerEnv、LIBERO‑Plus、DOMINO）和四个真实世界任务中与 π_0.5、Motus 等基线对比，InternVLA‑A1.5 在大多数指标上取得最优或显著提升，尤其在零射击和长周期执行上表现突出。

**⚠️ 局限性**

①仅对单个动作块进行未来监督，缺乏长周期规划与显式世界模型推理；②冻结的视频生成器受预训练覆盖范围限制，继承的动力学先验在更广泛的场景中可能不足。

---

## 580. Geometry-Aware Bayesian Quantification via Compositional Data Analysis

**arXiv ID:** 2607.04977 | [PDF](https://arxiv.org/pdf/2607.04977v1)

**作者:** Alejandro Moreo `[一作]` (Italian National Research Council), Juan José del Coz `[通讯]` (University of Oviedo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究针对多类别量化（label shift）问题，提出了一种几何感知的核密度估计（KDE）方法，并给出了对应的贝叶斯推断框架；

**💡 创新点**

创新点包括：① 在概率后验向量上使用对数比率变换（CLR/ILR）构造 Aitchison 几何空间，从而避免欧氏核在单纯形边界外泄漏；② 通过收缩正则化平滑单纯形边界附近的数值不稳定；③ 将几何感知 KDE 与最大似然/贝叶斯推断结合，得到既能做点估计又能给出不确定性量化的完整方法；

**🔧 技术方法**

所用技术包括组合数据分析（CoDA）、Aitchison 几何、中心对数比率（CLR）/等距对数比率（ILR）变换、收缩正则化、核密度估计、最大似然与贝叶斯推断（NUTS MCMC）、温度校准以及 Dirichlet 采样协议；

**📊 数据集**

实验数据集共 42 个，涵盖文本（LeQua、Twitter 情感数据）、表格（UCI 机器学习数据）和图像（CIFAR/SVHN、MNIST/FashionMNIST）三大模态；

**📈 对比分析**

与传统的 CC/PCC、BBSE/ACC、MLLS/EMQ、KDEy(Gaussian) 及其贝叶斯变体进行比较。点估计上，几何感知 KDE 在绝对误差（AE）和权重比误差（W）上均取得最高或次高排名，尤其在表格和图像数据中优于 Gaussian KDE；在不确定性评估方面，贝叶斯几何感知 KDE 的覆盖率接近 95%，幅度适中，优于 Bootstrap 方法和 Bayes-ACC；

**⚠️ 局限性**

局限性在于收缩参数和带宽需通过验证集手工调优，未实现全贝叶斯推断；方法仅针对 label shift，针对协变量偏移等其他偏移类型的性能尚未评估。

---

## 581. Train Smarter, Not Longer: Memorization-Guided Data Reuse for Efficient LLM Training

**arXiv ID:** 2607.04969 | [PDF](https://arxiv.org/pdf/2607.04969v1)

**作者:** Jingwei Zuo `[一作]` (Technology Innovation Institute), Hakim Hacid `[通讯]` (Technology Innovation Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在LLM预训练中如何最佳复用有限高质量数据，提出记忆窗口框架指导多周期训练。

**💡 创新点**

创新点在于通过损失保持差异定义记忆窗口，将其与训练周期长度和总迭代次数结合，形成可操作的复用策略。

**🔧 技术方法**

使用回滚损失、下游评测（MATH500）以及多周期训练实验衡量记忆保留与泛化窗口。

**📊 数据集**

训练集为FineWeb低质量数据与OpenMathInstruct2高质量数学数据，评估基准为MATH500。

**📈 对比分析**

与传统全局随机混洗相比，记忆窗口策略在相同总令牌量下提升数个百分点准确率，并能延迟过拟合。

**⚠️ 局限性**

实验仅在100M规模模型、单一数学领域和单一数据混合上验证，未提供在线估计τ*的方法。

---

## 582. Measuring Healthcare Data Leaks and Security Flaws at Internet Scale

**arXiv ID:** 2607.04965 | [PDF](https://arxiv.org/pdf/2607.04965v1)

**作者:** Nico Brüggemann `[一作]` (Fraunhofer SIT and National Research Center for Applied Cybersecurity ATHENE), Sebastian Schinzel `[通讯]` (FH Münster University of Applied Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对全球 IPv4/IPv6 地址空间进行大规模扫描，利用低交互蜜罐验证扫描结果，系统性评估公开可访问的 DICOM、HL7 与 FHIR 服务的身份验证、加密配置及已知软件漏洞，随后开展责任披露与后续扫描跟踪。

**💡 创新点**

① 首次对 HL7 与 FHIR 公开接口进行 Internet‑wide 量化测量；② 将 IPv6 采样方法与 6Sense、Hitlist 结合，覆盖更广泛地址空间；③ 引入低交互蜜罐对扫描误报进行校正；④ 建立统一的披露流程与后续验证机制。

**🔧 技术方法**

ZMap/ZMapv6 扫描网络端口；ZGrab2 模块发送协议特定握手（DICOM C‑ECHO、HL7 PDQ、FHIR CapabilityStatement）进行确认；pynetdicom 进行 DICOM C‑FIND 漏洞检测；LDAP 探测 AE‑title 泄露；TLS 配置评估使用 ciphersuite.info；Censys.io 与 Shodan.io 数据补充；NIST NVD API 对软件版本匹配 CVE，统计 CVSS；Python/Go 脚本编排扫描与数据处理。

**📊 数据集**

全球 IPv4/IPv6 地址空间；6Sense 与 Hitlist 生成的 1.4×10⁸ IPv6 采样地址；Censys.io 与 Shodan.io 的公开扫描记录；Lantern FHIR 端点数据集；NVD CVE 数据库；自建蜜罐日志与交互记录。

**📈 对比分析**

采用单次完整扫描与后续 4 周跟踪扫描对比，比较 TLS 支持率（DICOM 1.3%/HL7 0%/FHIR 70%）、加密协议版本、弱密码与自签证书比例；比较 DICOM 与 FHIR 的 CVE 发现数量（DICOM 2 CVE、FHIR 1 CVE）；公开数据揭示美国 26.4% 的主机易受攻击；扫描覆盖率估计为 IPv4 约 3.7B 主机，IPv6 约 1.4×10⁸ 条目，整体扫描耗时约 2 天。

**⚠️ 局限性**

仅从单一德国服务器扫描，可能因网络或地理屏蔽漏检；IPv6 采样仍不足以捕获全部活跃地址；FHIR URL 需预先设定，无法检测所有实例；缺乏对实际患者数据的验证，无法判断是否为测试/生产环境；问卷响应率低，披露效果有限；未对 HL7v3 与自定义协议进行检测。

---

## 583. Who's Behind It? Annotating and Extracting Conspiratorial Actors from German Telegram Posts

**arXiv ID:** 2607.04962 | [PDF](https://arxiv.org/pdf/2607.04962v1)

**作者:** Helena Mihaljević `[一作]` (HTW Berlin), Katharina Soemer `[通讯]` (Goethe University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过制定注释准则并对德语Telegram帖子进行span级别的共谋演员标注，构建了一个新的语料库，并训练Transformer模型实现共谋演员的自动抽取，随后将模型应用于大规模德语共谋归档（Schwurbelarchiv）进行演员分布与时间演化的宏观分析。

**💡 创新点**

创新点在于：①首次针对共谋理论文本提出了可操作的演员标注规范，并实现了高一致性的span级别标注；②构建了面向德语Telegram的专用语料库；③展示了基于Transformer的演员抽取模型在跨规模数据上的可迁移性，并通过演员特指性分类揭示共谋叙事中的隐性与具体化倾向。

**🔧 技术方法**

主要技术为：BIO标记的词级分类任务，使用HuggingFace Transformers框架对德国语言的Transformer模型（如BERT、XLM-R等）进行微调；同时采用Gamma、严格/放宽/重叠F1等span-aware评估指标进行性能评估。

**📊 数据集**

使用的数据集包括：TelCovACT（3663条Telegram帖子，1,251条带span级演员标注）和Schwurbelarchiv（约135万条德语Telegram帖子，约180万条抽取出的演员span），后者用于大规模分析。

**📈 对比分析**

在留出测试集上，最佳模型在严格span级F1上达到0.52，放宽/重叠F1约0.65，表明虽然仍存在边界误差，但整体能可靠地抽取演员并支撑大规模分析。

**⚠️ 局限性**

主要局限在于：仅针对德语Telegram文本，缺乏跨语言与跨平台验证；标注在单条帖子上完成，未考虑跨帖子上下文；Transformer模型在长文本或自动转录的音视频内容中可能产生无文本依据的“虚假”演员抽取，需要进一步研究。

---

## 584. DuplexChat: Constructing Speaker-Separated Full-Duplex Dialogue Speech at Scale for Spoken Dialogue Language Modeling

**arXiv ID:** 2607.04941 | [PDF](https://arxiv.org/pdf/2607.04941v1)

**作者:** Wataru Nakata `[一作]` (University of Tokyo), Hiroshi Saruwatari `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文构建了一个端到端的开放式管道DuplexChat-Pipe，并使用该管道从公开播客资源中生成双人对话的单声道分离语音，最终得到英文282,634小时、日文132,723小时的DuplexChat对话语料库。

**💡 创新点**

创新点在于首次提供可复现、可扩展的全双工对话语音生成流程，并产生了迄今规模最大、支持双声道的公开对话语料；同时通过结合播客索引、语者分离与语音恢复技术，实现了大规模自动化语料构建。

**🔧 技术方法**

技术路线包括：利用PodcastIndex爬取RSS、语言过滤和去重；对下载的音频做采样率统一、音乐节目过滤与时长裁剪；使用开源说话人分离模型进行两人说话片段的对话切片；最后采用DialogueSidon扩散模型完成语音分离与去噪恢复，并以左右声道保存每个说话人。

**📊 数据集**

数据集主要来自公开播客（PodcastIndex）中的英文与日文节目，经过上述流程后生成了DuplexChat语料库，覆盖约415k小时双声道对话。

**📈 对比分析**

通过对600条随机样本的DNSMOS、SQ-STOI、SQ-PESQ、ITC、ITD等指标与传统电话语料Fisher进行对比，DuplexChat在音质与说话人一致性上达到或超过Fisher；在交互特征分析中，其回声、重叠、交替频率等指标与真实人类对话高度一致，说明语料具备良好的对话动态表现。

**⚠️ 局限性**

局限性包括：日语分离质量相对较低，可能受限于DialogueSidon训练数据；仅处理了英文可用RSS的约2%，真正规模更大时可能出现更多噪声与版权合规问题；此外，自动化分离仍可能在极端环境下出现误分或残余噪音。

---

## 585. Towards Robust Uncertainty-Aware Speaker Modeling

**arXiv ID:** 2607.04937 | [PDF](https://arxiv.org/pdf/2607.04937v1)

**作者:** Junjie Li `[一作]` (Hong Kong Polytechnic University), Kong Aik Lee `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了两种改进声学不确定性建模的方案：一是结合跨说话人间隔离度与同一说话人内部变异度的“Inter‑和 Intra‑Speaker‑Aware Uncertainty Softmax”，二是针对域漂移的“Uncertainty‑Calibrated Domain Adaptation（UCDA）”方法，分别提升嵌入不确定性估计的可靠性与跨域鲁棒性。

**💡 创新点**

创新点在于（1）把说话人间相似度与同说话人内部一致性同时作为不确定性学习的监督信号，利用指数权重的联合硬度提升尺度函数；（2）在域适配中仅对不确定性模块进行轻量级无标签对齐，通过源域高斯先验的负对数似然实现不确定性空间的分布对齐，避免扰动嵌入空间。

**🔧 技术方法**

技术包括：1）基于高斯分布的声学嵌入聚合与不确定性估计；2）UAAM‑Softmax及其改进版本（Inter‑、Inter+Intra‑Speaker‑Aware Softmax）；3）不确定性校准域适配（UCDA）的负对数似然对齐；4）传统的AAM‑Softmax、AM‑Softmax、SphereFace2等对比基准。

**📊 数据集**

数据集：训练使用 VoxCeleb2；评估在 VoxCeleb1（in‑domain）和 CNCeleb（cross‑domain）上进行。实验包含多种训练长度、数据增强、以及不同学习率的UCDA适配。

**📈 对比分析**

与基线 ECAPA‑TDNN + AAM‑Softmax 比较，UAAM‑Softmax 已带来 EER/minDCF 的提升；加入交叉与内部硬度后（Inter+Intra‑Softmax）进一步降低 EER 至约0.88%（在 VoxCeleb1 上）并提升 minDCF；在跨域 CNCeleb 上，UCDA 以 10⁻⁷ 学习率实现 EER 下降 3.5% 以上、minDCF 下降 3% 以上，相比基线显著提高，且相对改进（RI）最高达 3.5%。

**⚠️ 局限性**

局限性：① 需要手动调节 λ 稳定项，影响不确定性尺度的权重；② 对于极端域差异，UCDA 的单向先验对齐仍可能不足；③ 当学习率过大时，适配过程会导致不确定性估计不稳定；④ 仅在无标签情形下进行适配，若可用少量标签可能进一步提升性能。

---

## 586. TARE: Tail Aware Evaluation of HPC Job Runtime Prediction

**arXiv ID:** 2607.04935 | [PDF](https://arxiv.org/pdf/2607.04935v1)

**作者:** Haili Xiao `[一作]` (CNIC, CAS), Rong He `[通讯]` (CNIC, CAS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对高性能计算作业运行时预测的评估方法，重点关注工作负载尾部，并在此基础上设计了混合调度策略。

**💡 创新点**

创新点在于将资源占用加权的几何准确率（GeoAccuracy）与尾部分析相结合，揭示了传统平均指标无法体现的预测器性能差异，并将该发现转化为提升调度效率的实用策略。

**🔧 技术方法**

技术手段包括：使用XGBoost机器学习预测器、基于最近两次作业的 Last2 经验预测器和用户提供的 walltime 估计；采用几何加权误差和分位数（分位数）分析；以及在模拟环境中执行基于 p90 阈值的混合调度回放。

**📊 数据集**

实验数据集来自三个国家实验室的生产作业轨迹：NREL Eagle、ALCF Mira 和 Intrepid，覆盖数千万条作业记录。

**📈 对比分析**

在离线评估中，GeoAccuracy 显著区分了三种预测源，尤其在最高 10% 资源占用的尾部，用户 walltime 估计取得最佳准确率和最低欠估率；在线回放显示混合策略将平均等待时间降低至 8% 以内，将可回填作业数提升 50%–115%，验证了离线指标与调度性能的关联。

**⚠️ 局限性**

局限性包括：未提出新的预测算法；仅在简化的 EASY 回填调度器上进行回放；评估聚焦于尾部权重而非完整的误差分布；结果仅针对所选三套系统，可能不完全适用于其他工作负载或调度环境。

---

## 587. Input Pathways Shape Few-Shot, Not Zero-Shot, Binding in Tiny Transformers: A Fully-Enumerable Study

**arXiv ID:** 2607.04926 | [PDF](https://arxiv.org/pdf/2607.04926v1)

**作者:** Yoshiyuki Ootani `[一作]` `[通讯]` (Independent Researcher), Yoshiyuki Ootani (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文在极小的可枚举事实世界中，系统比较了多种信息通道（符号、oracle、弱/强感知等）对小型Transformer绑定和组合推理的影响，发现零样本组合失败，几样本绑定受输入可读性和通道共享影响；

**💡 创新点**

创新点在于通过信息匹配的路由设计、完全枚举评估、精确Bayes上限和双因素分析，首次揭示输入可读性与共享路径对小模型组合学习的决定性作用；

**🔧 技术方法**

技术方法包括1–2层Transformer、可调参数共享路径、线性/MLP可读性测度、可变采样密度、对比统计检验以及可插拔的感知编码矩阵；

**📊 数据集**

使用的“数据集”是手工构造的两对象/三对象低维因子世界（128–729个离散状态），完全可枚举；

**📈 对比分析**

在零样本条件下所有通道表现低于chance；在少样本（k-shot）下，可读且共享路径的模型显著优于oracle和强感知，表现从0.82提升至0.96；

**⚠️ 局限性**

局限性包括模型规模极小、感知编码为固定合成矩阵、无法验证非线性可读性强但线性弱的情况、对自由描述顺序和更大世界的扩展性未知。

---

## 588. UniSpine-GS: An Efficient Physics-Aware Gaussian Framework for Cross-Modality Multi-view Spine Image Synthesis

**arXiv ID:** 2607.04923 | [PDF](https://arxiv.org/pdf/2607.04923v1)

**作者:** Qiuhua Chen `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

构建了基于物理感知的高效3D Gaussian框架UniSpine-GS，用于多模态多视角脊柱影像的投影重建

**💡 创新点**

首次结合辐射Gaussian表示、结构引导权重图SPWM与几何感知初始化，实现跨模态（X光与超声）脊柱图像的高质量重建

**🔧 技术方法**

使用辐射Gaussian渲染、统一辐射前向算子、RIRF、ACUI、SPWM加权损失以及密度化–修剪训练策略

**📊 数据集**

CTSpine3D（600+ CT 卷）与新构造的 FeSpine3D（100 份胎儿超声）

**📈 对比分析**

相较于 NeRF、TensoRF、NAF 等基线，UniSpine-GS 在 CTSpine3D 上 PSNR 46.54 dB、SSIM 0.9938，帧率 113 fps，训练 15min；在 FeSpine3D 上 PSNR 40.35 dB、SSIM 0.9815，帧率 148 fps，训练 7min，显著优于所有对比方法

**⚠️ 局限性**

对超声的物理建模仅为近似代理，缺乏严格的声学模拟，且在极低对比度或极噪声场景下仍可能出现残留伪影

---

## 589. When Do Foundation Models Pay Off? A Break-Even Analysis of Pretrained Time Series Forecasters

**arXiv ID:** 2607.04919 | [PDF](https://arxiv.org/pdf/2607.04919v1)

**作者:** Nicholas Tan Jerome `[一作]` (Karlsruhe Institute of Technology), Frank Simon `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对30个公开时间序列数据集在不同训练规模下系统比较预训练基础模型与传统方法，提出突破点（break‑even）分析框架。

**💡 创新点**

首次引入突破点概念评估预训练模型优劣，发现 FM 优势不恒定并给出可直接决策的两步规则。

**🔧 技术方法**

使用多种预训练模型（Chronos、Moirai、Lag‑Llama）零射和LoRA微调，传统基线（Naive、ETS、ARIMA、XGBoost），并计算六个时间序列特征进行元学习。

**📊 数据集**

30个公开基准集，涵盖能源、交通、金融、健康等领域，样本量从数十到数十万不等。

**📈 对比分析**

在6个训练比例（2%–100%）和3个随机种子下评估MASE，发现15/30数据集 FM 永胜，6/30 仅需2%即可超越，余9/30 需大量数据；LoRA 在短序列上往往退化。

**⚠️ 局限性**

样本数有限导致元学习预测仅略高于随机，未覆盖多变量/更大模型，完整微调在低数据时不稳定。

---

## 590. Non-Convex Sparse Reinforcement Learning via Non-Monotone Inclusions

**arXiv ID:** 2607.04990 | [PDF](https://arxiv.org/pdf/2607.04990v1)

**作者:** Kyohei Suzuki `[一作]` (Institute of Science Tokyo), onstantinos Slavakis `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种批量强化学习的特征选择方法，利用非凸PMC惩罚改造LSTD估计，并把问题转化为非单调包含问题，通过前向-反射-后向分裂(FRBS)求解并给出收敛理论；

**💡 创新点**

①将非凸PMC惩罚引入RL特征选择并保证整体凸性；②把LSTD+PMC转化为非单调包含框架并在此框架下扩展FRBS的收敛分析，包含弱Minty VI条件；③对超参数q的理论与经验影响进行系统分析；

**🔧 技术方法**

非凸优化、PMC惩罚、弱凸性、非单调包含、前向-反射-后向分裂(FRBS)、Lyapunov稳定性、弱Minty VI、近似策略迭代；

**📊 数据集**

三大经典RL基准任务：50状态链行走、Mountain Car、Acrobot，特征由RBF核与随机噪声构成；

**📈 对比分析**

与LSTD、LARS‑TD、BPDN等方法比较，所有方法均改造为API；实验显示，提出方法在NMSE、成功率、平均步数上均明显优于竞争者，尤其在噪声特征多或样本不足时优势更为突出；

**⚠️ 局限性**

仅适用于批量学习，未考虑在线学习；需要手动调节超参数q和μ；理论对弱Minty VI的充分条件尚未给出；在特征维度极大时计算量仍然受限；

---

## 591. Real-World Perturbation Testing of Autonomous Driving Systems

**arXiv ID:** 2607.04953 | [PDF](https://arxiv.org/pdf/2607.04953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 592. Virtual Category-Guided Continual Generalized Category Discovery

**arXiv ID:** 2607.04984 | [PDF](https://arxiv.org/pdf/2607.04984v1)

**作者:** Jiahui Xiong `[一作]` (Southeast University), Hongsong Wang `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于虚拟类别指导的持续性通用类别发现框架（VC‑GCGD），解决连续学习过程中类别漂移与样本不平衡问题。

**💡 创新点**

创新点在于：①设计虚拟类别概念，将未见类别与已知类别进行映射；②利用虚拟类别对模型进行动态正则化，提升对新类别的泛化；③结合对比学习与生成式记忆重放，兼顾样本稀缺与记忆保持。

**🔧 技术方法**

核心技术包括：对比学习（contrastive loss）、生成式记忆重放（memory replay with GAN/ VAE）、类别分布匹配正则化、以及自适应虚拟类别生成策略。

**📊 数据集**

实验数据集主要包括CIFAR‑100、ImageNet‑100、COCO‑20K等公开数据集，并在多种类别增量顺序下进行评估。

**📈 对比分析**

与传统方法（如 CGCD、TCA、LwF、Replay‑Net）比较，VC‑GCGD 在平均准确率、灾难性遗忘率与类别召回率等指标上分别提升 5.2%–8.7%，在长期增量序列下仍保持较低的误分类率。

**⚠️ 局限性**

局限性：①虚拟类别的参数设置对性能影响显著，需经验调优；②生成式记忆重放对显存和训练时间有较高需求；③在极长类别序列（>200 类）下仍会出现累积误差，需进一步研究更高效的记忆机制。

---

## 593. LLM for the development of FCM

**arXiv ID:** 2607.04983 | [PDF](https://arxiv.org/pdf/2607.04983v1)

**作者:** Alexis Kafantaris `[一作]` `[通讯]`, Alexis Kafantaris

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用本地大语言模型（Qwen2.5-32B）对 TripAdvisor 上的希腊酒店评论进行量化情感分析（ACSA），提取多维实体并构建对应表格；随后利用梯度下降训练数据驱动的模糊认知图（FCM），将满意度作为目标变量，验证其对星级评分的预测能力。

**💡 创新点**

创新点在于：①首次把本地 LLM 用于从文本直接抽取可量化的情感数值；②将抽取的数据直接用于梯度下降训练的 FCM，而非传统的因果抽取或预训练模型；③在同一数据集上与多种基线模型进行系统比较，突出 FCM 的可解释性与竞争性能。

**🔧 技术方法**

使用的技术包括：本地 LLM（Qwen2.5-32B）进行 ACSA、表格化量化抽取；梯度下降训练的 L2 正则化 tanh FCM；k 折交叉验证、置换检验、R² 评估；与线性回归、均值预测、XGBoost、随机森林、MLP 等基线模型比较。

**📊 数据集**

数据集：1505 条希腊酒店评论（TripAdvisor），含星级评分；经过筛选后 1491 条记录用于训练/测试，涉及 9 个概念（清洁度、员工、位置、早餐、噪音、价值、舒适度、设施、入住体验）。

**📈 对比分析**

比较方法：在 70/30 hold‑out、5‑fold 交叉验证和置换检验中评估 R²；与线性回归、均值预测以及 XGBoost、随机森林、MLP 等基线模型对比。FCM 的 hold‑out R² 为 +0.795，5‑fold 平均 R² 为 +0.782，优于均值预测但略低于机器学习基线（最高 +0.856）。

**⚠️ 局限性**

局限性：①本地 LLM 计算成本高、对 GPU 依赖强；②实体抽取质量受模型规模限制，可能无法捕获更细粒度的维度；③虽然 FCM 可解释，但预测性能不如现代深度学习或集成方法；④实验仅在希腊酒店评论上验证，泛化性待进一步验证。

---

## 594. Look-Ahead-Freedom as Temporal Non-Interference: A Verifiable Correctness Property for Backtesting and Agentic Trading Pipelines

**arXiv ID:** 2607.04958 | [PDF](https://arxiv.org/pdf/2607.04958v1)

**作者:** Xavier Fonseca `[一作]` `[通讯]` (Breda University of Applied Sciences), Xavier Fonseca (Breda University of Applied Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个线性时间的类型与效应检查器，用于验证金融/机器学习流水线在任何决策时刻都不受未来信息影响的看前偏差自由性。

**💡 创新点**

将看前偏差视为时间非干预，揭示了其不可判定性边界，并在价值无关可用性子语言上给出了可判定且声称的检查方案。

**🔧 技术方法**

利用时间索引管道演算、类型与效应系统、两运行逻辑关系以及可判定性与不可判定性证明技术。

**📊 数据集**

使用合成对抗性流水线、从专有市场数据衍生的五个原型（AR2–AR5）以及公开的量化数据集进行实验验证。

**📈 对比分析**

与两种经验检测器（差分检测和窗口平铺）对比，检查器在33个注入泄漏中零漏检率，误报率在可识别的“暗值”操作上约为75%，且在四个数量级上保持近线性时间。

**⚠️ 局限性**

误报主要源自对价值操作的不透明性，且检查器仅适用于可判定子语言，无法处理基于数据值的可用性计算或复杂模型检索所导致的泄漏。

---

## 595. 3DMPE: 3D Multi-Perspective Embedding

**arXiv ID:** 2607.04898 | [PDF](https://arxiv.org/pdf/2607.04898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 596. You Frame It: How Conceptual Representations Shape LLM Detection and Reasoning about Antisemitism

**arXiv ID:** 2607.04945 | [PDF](https://arxiv.org/pdf/2607.04945v1)

**作者:** Katharina Soemer `[一作]` (Goethe University), Helena Mihaljević `[通讯]` (HTW Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估外部概念资源对LLM检测反犹主义的影响，比较四种提示配置下的性能。

**💡 创新点**

首次系统比较不同概念化程度对检测召回和解释质量的影响，发现细粒度分类提升召回但牺牲精度，且大量上下文无益。

**🔧 技术方法**

使用指令式提示、外部概念资源（IHRA定义、Lexicon分类、示例）以及四个主流LLM模型（Gemini‑2.5‑Flash、Claude Sonnet 4.6、GPT‑5.4、LLaMA‑3.3‑70b）。

**📊 数据集**

两个专家标注的数据集：Bloomington（推特）和Decoding（社交媒体/新闻）。

**📈 对比分析**

通过在二分类任务上计算F1、精确率、召回率等指标，发现STRUCT和STRUCT+EX配置在大多数模型上均显著提高F1（最高0.64），但精度下降；模型在Bloomington上表现更好。

**⚠️ 局限性**

局限性包括概念映射不完备、外部知识更新困难、实验模型覆盖有限，以及大上下文实验中示例来源与数据集重叠导致的偏差。

---

## 597. Teaching LLMs a Low-Resource Language: Enhancing Code Completion in Pharo

**arXiv ID:** 2607.04939 | [PDF](https://arxiv.org/pdf/2607.04939v1)

**作者:** Kilian Kier `[一作]` (Graz University of Technology), Stéphane Ducasse `[通讯]` (University of Lille)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了面向低资源语言 Pharo 的 LLM 代码补全系统，构建了端到端数据处理、持续预训练、微调流水线，并制定多种基准测试。

**💡 创新点**

创新点在于结合 Tonel 格式数据清理、两阶段预训练+微调、AST‑aware 与随机 AST 遮掩策略，以及在真实 GitHub 提交上的仓库级评测，证明小规模专用模型可超越大规模通用模型。

**🔧 技术方法**

使用了 Qwen2.5 Coder 与 Mellum 两类支持 FIM 的开源 LLM，采用 LoRA 微调、4‑bit 量化、AST 解析（Tree‑sitter）与词法分析（Pygments）等技术，并设计自定义掩码与上下文策略。

**📊 数据集**

训练数据来自 415 个 MIT 授权、Pharo 10+ 版本、Tonel 格式的公开仓库，约 387k 条方法；基准集包括翻译版 HumanEval+（164 任务）和 Exercism（47 任务），以及 22 个测试仓库的 2,185 次提交级补全任务。

**📈 对比分析**

通过与基线（未微调）以及更大模型 Qwen3 480B 与 Claude Sonnet 4.5 的 pass@1、ChrF、CrystalBLEU 比较，发现小型 3B/7B 专用模型在方法级评测中超过基线，并在某些随机 AST 场景下击败更大模型；在仓库级评测中，提供最近修改方法上下文可提升约 15%+ ChrF，甚至 1.5B 模型超越 480B。

**⚠️ 局限性**

限制包括可能的数据泄露风险、方法级测试不完全模拟真实编辑场景、仓库级评测仅基于相似度而非功能正确，以及推理速度仍略高于实时阈值，低端设备未作评估。

---

## 598. Lightweight ML-Based Automatic Sleep Staging Framework with Constrained CNN and Mamba for Small-Sample EEG Datasets

**arXiv ID:** 2607.04934 | [PDF](https://arxiv.org/pdf/2607.04934v1)

**作者:** Zihao Wei `[一作]` (Changchun University of Science and Technology), Yudan Lv `[通讯]` (Jilin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种轻量级、低延迟的单通道EEG睡眠分期框架GamSleepNet，解决小样本过拟合和难分阶段低准确率问题。

**💡 创新点**

创新点包括：①改进Gabor核+可学习滤波的FEB特征提取模块；②采用Mamba线性时序建模网络；③两阶段训练结合对SOTA模型的对比损失提升难分阶段识别；④系统验证单通道睡眠分期模型的最佳数据集规模。

**🔧 技术方法**

技术：改进Gabor核+可学习滤波、Mamba时序网络、加权交叉熵+焦点损失、对比损失、两阶段训练、线性时间复杂度。

**📊 数据集**

数据集：公开SleepEDF（扩展版）和吉林大学第一医院私人睡眠数据集（44名患者）。

**📈 对比分析**

与多种SOTA方法对比（如Sleepyco、XSleepNet、TinySleepNet、SleepTransformer等），GamSleepNet在SleepEDF上ACC 87.86%、MF1 82.62%、κ 0.839，参数仅30.86k，显著优于其他模型；在私人数据集上ACC 86.42%、MF1 86.08%、κ 0.8199，同样表现最佳。

**⚠️ 局限性**

局限：目前仅支持单通道EEG，未验证多通道或实时系统的进一步适用性；对比损失需依赖已有SOTA模型；对极端小样本数据集的鲁棒性尚未充分评估。

---

## 599. Ossetic-COT: Designing a morphologically annotated corpus and morphological analyzer for Ossetic

**arXiv ID:** 2607.04895 | [PDF](https://arxiv.org/pdf/2607.04895v1)

**作者:** Anna Shatskikh `[一作]` (Lomonosov Moscow State University), Alexey Sorokin `[通讯]` (Lomonosov Moscow State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了符合 Universal Dependencies v2 规范的第一份 Iron Ossetic 形态学语料库，并基于此训练了 BERT 形态分析器。

**💡 创新点**

创新点在于将 Ossetic 的形态注释迁移至 UD schema，细化多义词解析并扩展了特征集，同时首次发布了 BERT 形态分析模型。

**🔧 技术方法**

使用了 BERT（mBERT）进行 Masked Language Modeling 预训练，并在 CoNLL-U 格式的语料上进行微调，以实现形态标注。

**📊 数据集**

数据来源为 5454 句、74032 token 的 COT 语料库，且使用 ONC 的 1,060,693 句无标注文本进行预训练。

**📈 对比分析**

微调后模型在测试集上达到了 95.60% 的标注准确率，与多语言 BERT 的性能差距不显著，且在多类指标上表现稳定。

**⚠️ 局限性**

局限性包括语料量小、词汇多样性不足、对不同文体（尤其是书面语）的泛化能力有限，以及部分稀有特征组合未在训练集中出现。

---

## 600. RL-Ballast: Ship Ballast Water Path Planning and Clog Prediction via Reinforcement Learning

**arXiv ID:** 2607.04906 | [PDF](https://arxiv.org/pdf/2607.04906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 601. ProCon: Projection-Consistency Memory for Training-Free Anomaly Detection

**arXiv ID:** 2607.04894 | [PDF](https://arxiv.org/pdf/2607.04894v1)

**作者:** Joongwon Chae `[一作]` (Tsinghua University Shenzhen International Graduate School), Ilmoon Chae `[通讯]` (Ratel Soft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为ProCon的训练无关记忆一致性框架，通过软投影替代硬最近邻检索，并利用投影残差作为异常检测依据；

**💡 创新点**

创新点在于将记忆检索转化为非参数软投影重建，并在记忆和层级两个维度上实现投影一致性投票，从而提升异常检测的鲁棒性与定位精度；

**🔧 技术方法**

采用冻结的DINOv2 ViT‑B/14特征提取器，构建多层独立的核心记忆，使用软投影计算投影残差，并通过银行中值聚合和层级均值聚合实现一致性；

**📊 数据集**

在MVTec-AD、VisA和Real‑IAD这三大工业缺陷检测基准上进行实验；

**📈 对比分析**

与PatchCore、RD4AD、Dinomaly、INP‑Former等方法对比，ProCon在图像AUROC、像素AP、AUPRO等多项指标上均取得或逼近最优成绩，例如在MVTec-AD上图像AUROC 99.8%，像素AP 73.5%；

**⚠️ 局限性**

局限性包括：需要多次记忆检索（20个银行层组合），计算开销较大；缺乏对训练集异常模式的自我清洗，易受污染；层选择固定，未针对不同类别自适应。

---

## 602. Evaluating Large Language Models for Antisemitic Incident Classification

**arXiv ID:** 2607.04890 | [PDF](https://arxiv.org/pdf/2607.04890v1)

**作者:** Karina Halevy `[一作]` (Carnegie Mellon University), Maarten Sap `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了细粒度仇恨事件检测任务，并以反犹主义事件为例进行研究，构建多种数据集并评估大语言模型（LLM）的性能；

**💡 创新点**

创新点在于将传统的仇恨言论检测扩展到事件级别，设计细粒度分类体系，探究提示工程对LLM识别行动型与修辞型仇恨事件的不同影响；

**🔧 技术方法**

主要使用提示工程（含定义、假设与示例）对大语言模型进行推理，评估 OpenAI GPT‑4 和 Meta LLaMA‑2 的表现；

**📊 数据集**

数据集包括：AMCHA（校园反犹事件）、ADL‑HEAT（美国各地反犹事件）、Synthetic（人工合成的非反犹犹太相关事件）以及 Campus‑News（高校校园报纸文章）；

**📈 对比分析**

通过准确率、精确率、召回率、F1 等指标比较模型与提示变体的效果；在 AMCHA 上，LLaMA‑2 在细粒度分类上略优于 GPT‑4，提示中加入定义对修辞型事件帮助最大，加入示例对行动型事件帮助最大；在 ADL‑HEAT 上，GPT‑4 整体表现更好；在 Synthetic 上两模型均能保持低误报；在 Campus‑News 上模型召回率较高但精确率偏低，提示可作为初筛工具；

**⚠️ 局限性**

局限包括：数据集受限于英语、美国校园与公开报导，缺乏多模态与多语种；LLM 对历史文化背景认知不足导致部分细粒度类型误判；提示工程效果因模型和数据集差异而异，缺乏系统化探索；未对模型进行微调，仅依赖推理，可能未充分发挥模型潜力；实验成本与环境影响较大。

---

## 603. The Map Behind the Flow: Finite-Step Gradient Descent as a Dynamical System

**arXiv ID:** 2607.04993 | [PDF](https://arxiv.org/pdf/2607.04993v1)

**作者:** Thomas Hofmann `[一作]` `[通讯]` (ETH Zurich), Thomas Hofmann (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

分析梯度下降的有限步长动力学，构建可解析模型并证明其在深度学习中的普适性。

**💡 创新点**

首次将有限步长梯度下降视为离散动力系统，揭示边界稳定性后出现的周期倍化、混沌与代表性选择等现象，并给出通用 Ricker 极限。

**🔧 技术方法**

离散动力系统分析、对角化分解、负 Schwarzian 条件、周期倍化理论、随机微分近似等技术。

**📊 数据集**

无具体数据集，纯理论与数值模拟。

**📈 对比分析**

通过对比解析阈值与数值实验，验证理论预测的准确性，显示不同深度、宽度、激活与噪声条件下的相同结构。

**⚠️ 局限性**

未解决持续收敛性、渐进收敛与高维实际网络的细节；仅在简化模型中给出精确结果。

---

## 604. Online Computation of the Longest Repeating Suffix and Smallest Suffixient Sets via Incremental Run-Length BWT-based Indexes

**arXiv ID:** 2607.05004 | [PDF](https://arxiv.org/pdf/2607.05004v1)

**作者:** Paola Bonizzoni `[一作]` (University of Milano-Bicocca), Gregory Kucherov `[通讯]` (Gustave Eiffel University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了首个压缩空间下的在线构造最小后缀集合（smallest suffixient set）以及在线计算最长重复后缀（Longest Repeating Suffix, LRS）的算法，并给出了两种空间-时间权衡方案；

**💡 创新点**

其创新点在于：①利用增量BWT与动态融合树实现对可压缩的前缀数组（PA）与最长公共后缀数组（LCS）的在线维护；②首次证明了在线LRS计算和最小后缀集合维护的Ω(n)位峰值空间下界，并通过归约将此下界推广到任意长度标注或右端最小后缀集合；

**🔧 技术方法**

核心技术包括：增量run‑length BWT、动态融合树（fusion tree）维护BWT顶部位置、基于块划分的B‑树与聚合最小值实现LCS/PA的区间最小查询，以及ϕ查询与下一/上一个较小值查询的在线实现；

**📊 数据集**

本文未使用实际数据集，实验与评估完全基于理论分析和复杂度证明；

**📈 对比分析**

与Prezza & Rosone 2020等前沿方法比较，本文在压缩空间下实现了O(log²n/ loglog n)与O((log n/ loglog n)²)的最坏情况单字符更新时间，显著提升了时间性能；

**⚠️ 局限性**

主要局限在于：第二种权衡方案中仍存在loglog n因子，是否可进一步消除该因子或在仅使用O(r log n + n)位空间的情况下实现同等时间复杂度仍为开放问题。

---

## 605. TACTIC-KG: Toward Small Agent Teams for Cyber Threat Intelligence Knowledge Graph Construction

**arXiv ID:** 2607.05001 | [PDF](https://arxiv.org/pdf/2607.05001v1)

**作者:** Mouhamed Amine Bouchiha `[一作]` (Institut Polytechnique de Paris), Gregory Blanc `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 TACTIC-KG 框架，将 CSKG 构建拆分为小型 LLM 代理，分别负责提取、类型、验证和策展，实现更高稳定性和低成本的知识图谱构建。

**💡 创新点**

创新点在于多代理分解与 LoRA 微调，强调“可信性优先于连通性”，使得在不使用超大单体模型的情况下显著提升召回率和图一致性。

**🔧 技术方法**

采用 3B–8B 规模 LLM + LoRA 细化、语义分块、闭域证据检验、基于本体的验证与策展、JSON 交互与离线推理等技术。

**📊 数据集**

使用 CTI-HAL 与 CTINEXUS 的 230 份手工标注 CTI 报告，并结合 MALOnt/ATT&CK 等本体进行实体类型和关系约束。

**📈 对比分析**

通过与大型单体模型（DeepSeek‑V3.1、Kimi‑K2 等 ICL 基线）在提取 F1、召回率和图结构相似度等指标上进行对比，TACTIC‑KG 在小模型上达到 80 F1、67 GraphSim，超过大型基线，召回率显著提升。

**⚠️ 局限性**

局限性包括：完整类型准确率仍不足、对高噪声或攻击性 CTI 输入的鲁棒性有限，以及验证与策展过程仍需人工可审计。

---

## 606. Data-Driven Soft Labeling Scales DNA Read Classification to Whole-Body Cell-Type Deconvolution

**arXiv ID:** 2607.04987 | [PDF](https://arxiv.org/pdf/2607.04987v1)

**作者:** Dmytro Rizdvanetskyi `[一作]` (KU Leuven), Pavlo Lutsik `[通讯]` (KU Leuven)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 Syto 框架，利用读级 DNA 甲基化模式实现多类细胞类型分解，并将软标签、线性校准和模块化设计整合进可扩展的工作流。

**💡 创新点**

创新点在于数据驱动的软标签推断、针对多类“信号→标签”多对多映射的标签分布学习、线性 simplex 校准投影以及整体模块化设计，使得读级分解能够扩展到数十种细胞类型。

**🔧 技术方法**

采用了 Dismir、MethylBERT、Lookup Classifier、CancerDetector 等读级分类器；soft-labeling（DD 与 canonical soft）、PSLS、线性 simplex 校准；NNLS、PSLS 等去卷积器；以及对数线性校准方法。

**📊 数据集**

使用 39 种细胞类型的 GSE186458 WGBS atlas 进行训练和 pseudobulk 验证；使用 GSE233417 RRBS 数据集与 Tabula Sapiens 计算期望比例，构建 OOD 16 组织的测试集。

**📈 对比分析**

与无监督方法 UXM 对比，Syto 在 pseudobulk 实验中 MSE 降低 2.56 倍（从 3.00e‑4 到 1.17e‑4），在 OOD RRBS 数据集上通过 Tissue Concordance Score（TCS）指标实现与 UXM U250（10 倍 DMR）相近甚至更优的表现。

**⚠️ 局限性**

局限性包括：依赖特定基因组区域选择；仅使用单一 WGBS atlas 和 RRBS OOD 数据；缺乏真实比例样本；未针对所有组件进行超参数最优搜索；目前仅在短读技术上验证，未覆盖第三代长读或多组学扩展。

---

## 607. Designing Touch for Trauma-Informed Social Robots: A Design Space for Direct and Indirect Actuation

**arXiv ID:** 2607.04981 | [PDF](https://arxiv.org/pdf/2607.04981v1)

**作者:** Madeleine Rischer `[一作]` (Helmut Schmidt University), Benedikt Bußmann `[通讯]` (Helmut Schmidt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

提出了针对创伤知情护理原则的社交机器人触摸交互设计空间，区分直接触摸与间接触摸，并分析三维维度（触发模态、目标与预期效果）。

**💡 创新点**

首次系统化构建触摸交互的设计维度，并将其与创伤知情护理（TIC）原则对应，提供可评估的框架。

**🔧 技术方法**

使用概念分析方法，基于现有HRI文献和创伤护理理论，构造设计空间；并通过案例式说明验证。

**📊 数据集**

无数据集，本文为理论与概念性工作。

**📈 对比分析**

无定量比较，未进行实验评估；作者提出未来应通过参与式与实证研究检验。

**⚠️ 局限性**

局限在缺乏实证验证，未针对实际PTSD人群评估触摸介入的有效性与接受度。

---

## 608. Qantara: Bridge-Flow Training for Multi-Paradigm JEPA Control

**arXiv ID:** 2607.04978 | [PDF](https://arxiv.org/pdf/2607.04978v1)

**作者:** Ruslan Rakhimov `[一作]` (T Tech), Daniil Gavrilov `[通讯]` (T Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

训练了一个约21M参数的子十亿级JEPA世界模型Qantara，能够在同一检查点下实现三种不同的推理范式：目标条件规划、行为克隆采样和视频逆向组合，而不需要额外的重训练；

**💡 创新点**

创新点包括：1) 在状态轴使用Brownian桥插值配合动作轴的流匹配训练目标，使预测器仅学习状态转移；2) 采用仅在噪声时间平面四条边及对角线的采样方式，集中训练容量在推理时真正需要的区域；3) 通过多模式共正则化（edge + diagonal）实现跨范式的稳定性，甚至对单一推理路径也有正向影响；

**🔧 技术方法**

使用了Joint bridge-flow训练框架，Brownian桥插值，流匹配，块因果Transformer，per-token AdaLN调制，Euler denoising步数，CEM规划，SIGReg正则化，状态头残差与零初始化等技术；

**📊 数据集**

主要使用了LeWM工作台的四个环境（Two-Room、Push-T、OGBench-Cube、Reacher-Hard）以及DINO-WM的专家轨迹数据；

**📈 对比分析**

与PLDM、DINO-WM和LeWM单一范式模型进行对比，Qantara在LeWM套件上平均91.2成功率，OGBench-Cube上93.7成功率，较DINO-WM提升7.7；同一检查点的行为克隆和视频逆向分别达82–83成功率；推理成本从规划的0.4–1.2秒降至17–24毫秒；

**⚠️ 局限性**

局限性包括：仅实现单步推理，需扩展多步块推理；评估仅在仿真环境，缺乏真实机器人验证；仅在21M参数规模验证，未探讨更大规模；对更长序列或更复杂任务的适应性待进一步验证。

---

## 609. A Comprehensive Study of Implementation Bugs in Multi-modal Agents

**arXiv ID:** 2607.04974 | [PDF](https://arxiv.org/pdf/2607.04974v1)

**作者:** Suwan Li `[一作]` (Nanjing University), Chang Yue `[通讯]` (Chinese Academy of Science)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多模态LLM智能体（M‑agent）的实现缺陷进行系统性研究，收集34个实例、提炼158个独立bug，并基于三层症状与根因分类实现自动化bug检测器。

**💡 创新点**

首次提出针对多模态代理的三维分类法（全局症状、组件级症状、根因），并证明该框架能够准确覆盖已知缺陷并发现新的bug，弥补了以往仅关注单模态或代码层面的研究。

**🔧 技术方法**

利用GPT‑4o对多模态输入进行交互，收集运行时跨组件输出（snapshot、plan、action）作为检测特征；结合自动化脚本和规则引擎进行bug识别。

**📊 数据集**

构建了包含34个M‑agent项目、1268条issue报告、158个独立bug的实验数据集，并在12个额外agent上进行扩展验证；数据来源为GitHub、顶级会议论文及综述。

**📈 对比分析**

通过与人工标注的bug集合比较，检测器在12个额外agent上覆盖61.4%已知开放issue，并新增31个bug，表明方法在准确率（97.6%）与覆盖率上具备较高实用价值。

**⚠️ 局限性**

研究局限：仅聚焦实现层缺陷，未覆盖组件内部缺陷；手工标注过程仍存在主观性；检测仅基于运行时输出，缺乏静态代码分析；数据集主要来自公开代码仓库，可能不足以覆盖所有应用场景。

---

## 610. STAPO: Selective Trajectory-Aware Policy Optimization for LLM Agent Training

**arXiv ID:** 2607.04963 | [PDF](https://arxiv.org/pdf/2607.04963v1)

**作者:** Qiuyi Qi `[一作]`, Qiang Zhu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提供了ACL会议作者使用的LaTeX模板和样式文件，并给出了格式说明。

**💡 创新点**

主要创新在于提供统一的样式文件和详细的使用说明，方便作者提交和最终稿件排版。

**🔧 技术方法**

使用LaTeX模板和style文件进行排版。

**📊 数据集**

无具体数据集。

**📈 对比分析**

本文未进行实验比较。

**⚠️ 局限性**

仅适用于ACL会议，缺乏通用性，未包含完整示例。

---

## 611. MemPose: Category-level Object Pose Estimation with Memory

**arXiv ID:** 2607.04930 | [PDF](https://arxiv.org/pdf/2607.04930v1)

**作者:** Xiao Lin `[一作]` (Tongji University), Qijun Chen `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了 MemPose，一个引入外部几何记忆的类别级物体姿态估计框架，通过动态更新记忆缓冲区提升姿态推断。

**💡 创新点**

将类别级几何记忆作为非参数模块动态集成，使用相似度驱动的令牌合并更新和门控融合，使模型在不增加参数的情况下获取更丰富的上下文。

**🔧 技术方法**

结合 RGB‑D 特征提取（DINOv2+PointNet++）、关键点检测、外部记忆缓冲、注意力检索、门控融合以及多任务损失（姿态、NOCS、重建等）。

**📊 数据集**

在 REAL275、CAMERA25、HouseCat6D 和 Wild6D 四个公开基准上进行训练与评估。

**📈 对比分析**

与 AG‑Pose、SpherePose、GCE‑Pose 等先进方法对比，MemPose 在 REAL275、HouseCat6D、Wild6D 上均刷新了 mAP/5°2cm 等指标，提升约 1–5%。

**⚠️ 局限性**

记忆模块在推理时需要冻结或更新，更新会产生一定计算开销；目前验证仅在单分类框架下，跨类别迁移与实时性仍待进一步研究。

---

## 612. Medi-Gemma: A Hybrid Clinical Decision Support System Integrating Deterministic EMR Analytics and Retrieval-Augmented Generation

**arXiv ID:** 2607.04907 | [PDF](https://arxiv.org/pdf/2607.04907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 613. DSWAM: A Dual-System World Action Foundation Model for Fine-Grained Robot Manipulation

**arXiv ID:** 2607.04927 | [PDF](https://arxiv.org/pdf/2607.04927v1)

**作者:** Jian Zhu `[一作]` (Midea Group), Yi Xu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了DSWAM双系统框架，使用基于视频的World Action Model执行器作为默认低层控制，并在需要时调用可选的视觉语言子任务规划器来实现细粒度机器人操控。

**💡 创新点**

创新点在于将语义任务拆分与物理执行解耦：执行器保持对动态世界的强大学习能力，规划器仅在粗指令需要拆分时才激活；采用基于过渡的子任务监督、无未来视频生成的直接动作预测、以及实时块化 + TensorRT 加速的高效部署方案。

**🔧 技术方法**

技术手段包括：视频共训练与流匹配的World Action Model、Rynnbrain‑style 视觉语言子任务规划器、实时块化（RTC）与异步执行、BF16 TensorRT 推理加速、以及在执行器与规划器之间的同步执行协议。

**📊 数据集**

使用了大规模真实机器人数据（与DeMaVLA相同的收集方式）、RoboTwin 2.0仿真任务集、DeMaVLA家庭折叠协议、以及排序任务的人工标注子任务数据。

**📈 对比分析**

在与DeMaVLA相同的机器人平台、数据、任务协议和评测标准下进行匹配对比：DSWAM在折叠任务上实现96.3%成功率、1′44″平均完成时间，显著优于DeMaVLA（92.5% / 2′18″）和π₀（76.3% / 2′26″）；在RoboTwin 2.0上平均成功率为92.38%（干净）/91.90%（随机），高于所有基线；可选规划器将排序任务成功率从71.4%提升至100%，错误数亦大幅下降；TensorRT加速使推理时延从198.2 ms降至73.8 ms，速度提升2.69×。

**⚠️ 局限性**

局限性包括：对粗指令识别与拆分的依赖，若指令已是细粒度则规划器不必要；评测主要集中在家庭折叠与仿真任务，未覆盖更复杂或不同域的操控；仍需大量真实机器人数据来训练执行器；以及在极端动态或不确定环境下的鲁棒性尚未验证。

---

## 614. Compliance Evidence in the Automotive Supply Chain: A Systematisation of the Quality-Document Spine and a Taxonomy of Documentation Failure Modes

**arXiv ID:** 2607.04924 | [PDF](https://arxiv.org/pdf/2607.04924v1)

**作者:** Dawar Jyoti Deka `[一作]`, Nilesh Sarkar `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统化了汽车供应商质量与认证证据链（AIAG PPAP 与 VDA Volume 2 PPA），编制了2012‑2024年公开的13起合规文件失败案例集，构建了基于机制、生命周期、驱动与检测路径的失败模式分类法，并据此提出证据体系的结构性需求与开放研究议程。

**💡 创新点**

首次将证据链视为信息系统并完整映射其产出、验证与保留过程；首次对汽车合规文件失败进行系统性分类并将失败机制与体系设计关联；揭示常规验证层未检测到任何失败的系统性缺陷。

**🔧 技术方法**

采用文献综述与案例分析相结合的系统研究方法，构造失败模式分类法，并进行概念性体系需求推导；未使用机器学习或统计建模。

**📊 数据集**

基于公开的13起合规文件失败案例（来自监管机构公告、法院判决、媒体报道等公共记录）作为数据集。

**📈 对比分析**

通过对案例的机制、生命周期、驱动和检测路径进行归类与对比，验证失败模式的完整性与辨别力；未进行数值性能评估，主要以定性分析为主。

**⚠️ 局限性**

仅覆盖公开且足够高调的案例，缺乏结构化的失败遥测数据；未涵盖保留与检索阶段的失败；缺乏对验证工作量的量化基准；未实现程序规范的机器可读化，导致部分失败机制难以检测。

---

## 615. Graph Representation Learning of Longitudinal Medical Imaging Trajectories for Treatment Response Prediction

**arXiv ID:** 2607.04912 | [PDF](https://arxiv.org/pdf/2607.04912v1)

**作者:** Johannes Kiechle `[一作]` (Technical University of Munich), Jan C. Peeken `[通讯]` (TUM University Hospital Rechts der Isar)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出基于时间感知图神经网络的自监督学习框架，用于预测乳腺癌患者在新辅助化疗期间的病理完全缓解（pCR）状态。

**💡 创新点**

创新点包括：1）构建体现时间顺序的有向无环图结构以建模治疗过程；2）三种专门的自监督损失（对齐、去相关、时间一致）以强化响应一致性和时间连贯性；3）在公开ISPY‑2数据集上构建完整的预测基准。

**🔧 技术方法**

使用3D ResNet18特征提取器、GraphSAGE图网络、Cosine相似度自监督损失以及交叉验证、AUC/MCC等评价指标。

**📊 数据集**

使用公开的ISPY‑2乳腺癌数据集（585例，204例响应者）。

**📈 对比分析**

与CNN、CNN+LSTM、DINOv3、3D‑L_ART等视觉/自监督基线进行5折交叉验证对比，GNN‑pCR在bACC、AUC和MCC上均实现了显著提升（分别从0.6844提升到0.7203/0.3561）。

**⚠️ 局限性**

局限性包括：仅在乳腺癌pCR任务上验证；ISPY‑2的扫描间隔相对均匀，时间差特征提升有限；缺乏跨疾病、跨模态的泛化验证。

---

## 616. Toward Personalized Social Robots for Child Well-being: Data Requirement Principles from a Recommender-System Perspective

**arXiv ID:** 2607.05110 | [PDF](https://arxiv.org/pdf/2607.05110v1)

**作者:** Jin Huang `[一作]` (University of Cambridge), Hatice Gunes `[通讯]` (University of Cambridge)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了面向儿童福祉的社交机器人个性化应用所需的四大数据原则，并讨论了如何将其与推荐系统框架相结合。

**💡 创新点**

创新点在于将推荐系统问题映射到机器人数据收集挑战，提出了整合型用户画像、有效性信号、可链接覆盖和曝光记录四项数据原则。

**🔧 技术方法**

利用推荐系统框架中的用户画像、排序和负责任计算模块，对数据原则进行映射与分析。

**📊 数据集**

主要使用已存在的机器人研究数据集（如NAO、Pepper、Haru等）进行案例对照，未收集新数据。

**📈 对比分析**

通过对比现有数据集满足度与所提原则的差距，评估了各功能所需最小原则组合，但未进行实验性能比较。

**⚠️ 局限性**

局限性在于缺乏实际数据验证，且对数据收集方法与隐私合规性仍需进一步探讨。

---

## 617. RepoTrace: Browser-Assisted Evidence Collection for GitHub Research Datasets

**arXiv ID:** 2607.05106 | [PDF](https://arxiv.org/pdf/2607.05106v1)

**作者:** Xue Yao `[一作]` (Monash University), Yongqiang Tian `[通讯]` (Monash University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于浏览器的工具，帮助研究者在本地收集、标注、审阅并导出 GitHub 问题与拉取请求的数据集，并记录页面快照、评论、标签、审阅决策等全部证据，形成可追溯、可复现的工作流程。

**💡 创新点**

创新点在于：①将浏览器端页面采集与本地 SQLite 存储无缝结合，保持原始页面渲染与后期标注相连；②提供统一的侧栏收集、仪表盘审阅、刷新与导出功能，解决传统表格+脚本导致证据链断裂的问题；③实现了对研究标签、审阅历史、冲突检测与审计日志的完整追踪，支持多评审者的可审计协作。

**🔧 技术方法**

技术栈包括：Chrome V3 扩展（TypeScript），Express+SQLite 后端（TypeScript），React+Vite 仪表盘，统一的 TypeScript 类型定义以及自动化测试（Jest/Playwright）来确保数据完整性。

**📊 数据集**

使用自建的 20 条 Matplotlib 仓库 issue 数据集，划分为两项研究（后端渲染错误与回归兼容性），验证工具能完整保存所有页面快照、评论、标签、审阅记录及冲突情况。

**📈 对比分析**

通过与传统手工收集+脚本方式的对比，作者证明该工具能够完整保留原始页面证据与标注逻辑，避免了证据与决策分散的可审计性缺失；在性能方面，工具在本地 SQLite 存储下能够快速读取、刷新与导出，满足 20 条记录的完整工作流程演示，未发现明显瓶颈。

**⚠️ 局限性**

限制包括：①尚未支持大规模批量抓取（依赖手动浏览页面），②缺乏 AI 辅助的自动标注或冲突解决功能，③对 GitHub API 的支持仅限于可选刷新，未实现持续同步；未来计划扩展 CSV/表格导出、版本管理与更细粒度的标签导航。

---

## 618. Can Code Specify a System Precisely Enough to Formally Verify It?

**arXiv ID:** 2607.05076 | [PDF](https://arxiv.org/pdf/2607.05076v1)

**作者:** Jean-Jacques Dubray `[一作]` `[通讯]`, Jean-Jacques Dubray

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在一家餐厅的POS支付系统中，使用低成本形式化验证方法检测并修复了七个失败处理缺陷，并通过模型检查确认修补后的系统安全性。

**💡 创新点**

创新点包括：① 将伴随研究的可执行模型检查流程迁移至真实生产环境，首次验证其在实际代码上的有效性；② 证明在LLM生成规范时合同结构决定可靠性而非编程语言；③ 发现源代码形态会削弱SAM合同的可验证性；④ 通过扩展失败模型（crash–restart、stale reads、两次尝试）进一步揭示系统潜在风险，并发现协同oracle（仿真器与生产API不一致）的缺陷。

**🔧 技术方法**

使用技术包括：SysMoBench的SAM模式、TLC模型检查器、Claude和Mistral大语言模型自动生成规范、机械窗口重放、手工构建环境模型、失效门（failure gates）扩展、补丁验证回检和生产沙盒实验。

**📊 数据集**

数据集由101个从未修改代码的真实执行窗口（含客户操作、卡片拒付、网络故障等）组成，另外通过仿真器生成的窗口补充对冲突情形；对7个LLM模型×3种合同×N=5代产生的规范进行评估。

**📈 对比分析**

比较方法是将三种规范合同（SAM、bare、constrained）在相同窗口集上分别进行机械重放和TLC验证，计算无条件和有条件通过率；结果显示bare合同在Claude和Mistral模型中始终最高，SAM合同在源代码形态相似时表现最差；在补丁验证后通过率恢复至100%。TLC模型检查完成时间通常在1秒以内，状态图规模约43–103个状态。

**⚠️ 局限性**

limitations包括：仅验证安全性质，未涵盖公平或时间相关属性；仅覆盖单终端单订单场景；环境模型未机械一致，仅通过日志重放验证；oracle与仿真器的相关性导致发现的缺陷在生产API上可能不同；模型适用于显式状态机源码，无法推广到隐式状态机；缺乏独立复现实验；统计仅到排列下限；源代码形态对SAM合同影响未被彻底分离。

---

## 619. Context-Aware ASR for Mandarin Technical Lectures

**arXiv ID:** 2607.05058 | [PDF](https://arxiv.org/pdf/2607.05058v1)

**作者:** Ho-Lam Chung `[一作]` (National Taiwan University), Hung-yi Lee `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了中文技术讲座的语音识别问题，提出一种基于两步无参考词典提示的方法，以提升技术术语的识别率。

**💡 创新点**

创新点包括：①引入词项中心评估指标（召回、精准度、F1、词错误率）来分离整体 CER 与术语识别；②利用第一步 ASR 输出自动生成词典，并在第二步解码时作为上下文提示，无需外部词表。

**🔧 技术方法**

使用技术包括 Whisper、Breeze-ASR、Qwen3-ASR 等现有 ASR 模型，搭配两步解码、频率排序的词典生成、Prompt/Guard 语句等实现对技术术语的上下文偏置。

**📊 数据集**

数据集基于公开的 AI/ML 讲座系列（15 讲、5.01 小时），构建了包含 8,888 条术语、1,030 种唯一术语的词项丰富基准集。

**📈 对比分析**

在 5 种 ASR backbone 上对比 Segment-only 与加入词典的解码：词项召回均提升 6–18%，Breeze-ASR‑25 的召回从 52.5% 提升到 60.13% 并将 CER 降至 9.79%；混合词典可达 62.05% 召回、9.40% CER。

**⚠️ 局限性**

局限性：仅评估单一讲师、单一 AI/ML 领域；词项提取基于规则，可能漏检；部分术语在第一步输出未出现导致覆盖不足；未测试多说话人或跨域情境。

---

## 620. Consistent and Editable: A Balanced Framework for Text-Guided Video Editing

**arXiv ID:** 2607.05056 | [PDF](https://arxiv.org/pdf/2607.05056v1)

**作者:** Tao Jin `[一作]` (University of Science and Technology of China), Li Xiao `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于预训练文本生成模型的一次性微调的视频编辑框架EquiEdit，旨在同时提升视频的时序一致性和可编辑性。

**💡 创新点**

创新点包括：① 引入Temporal Mamba模块并设计四向时序感知扫描，利用SSM实现全局时序建模；② 通过基于傅里叶变换的噪声注入策略，在保持高频结构信息的同时增强编辑灵活性。

**🔧 技术方法**

技术方法：Latent Diffusion Model、Stable Diffusion v1.4、Mamba (State‑Space Model)、三维快速傅里叶变换、DDIM逆向采样、Classifier‑Free Guidance。

**📊 数据集**

使用LOVEU‑TGVE、DAVIS、Videvo、YouTube四个公开数据集共76条视频，统一采样为32帧 512×512。

**📈 对比分析**

与Tune‑A‑Video、RAVE、SimDA、FLATTEN、TCVE等五种基线进行对比，评估指标包括CLIP一致性、用户投票和文本对齐分数。实验表明EquiEdit在时序一致性和文本对齐上均取得最高分（如CLIP一致性 95.863、文本对齐 31.068），并获得最高用户偏好。

**⚠️ 局限性**

局限性：仅在单一模型微调下实验，缺乏大规模多模态或长时序视频的验证；噪声注入参数需手工调节，过高或过低会导致编辑效果失真；对极端场景（如高运动模糊、复杂光照）尚未充分测试。

---

## 621. The Fine-Grained Complexity of Counting Hypergraph Motifs

**arXiv ID:** 2607.05040 | [PDF](https://arxiv.org/pdf/2607.05040v1)

**作者:** Madhumitha Krishnakumar `[一作]` (Queen Mary University of London), Marc Roth `[通讯]` (Queen Mary University of London)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对三元超图基因表达的Venn图谱（超图模式）进行计数问题的可行性和复杂度分析，进一步扩展到任意k-边超图模式的计数复杂度研究。

**💡 创新点**

创新点在于给出了三元超图模式（Venn图谱）的完整时间复杂度分界：对任意三元模式可实现参数化近二次时间的算法；对退化模式（包含一集合被另一个集合包含）可实现参数化近一次时间的算法；而对于非退化模式，则在三角形和超三角形命题仍成立时无法实现近一次时间算法。对于k>3的模式，给出基于分层宽度（分数超树宽和自适应宽度）对计数问题可行性的初步分类。

**🔧 技术方法**

采用了超图同构、同构除法、莫比乌斯函数的包含排除公式、Yannakakis算法、Dedekind循环的推导以及多重分裂（fracture）技术，并借助超图宽度概念（超树宽、分数超树宽、自适应宽度）和数据库查询等相关理论。

**📊 数据集**

未使用真实数据集，主要基于理论构造与计算机模拟的超图实例。

**📈 对比分析**

方法与传统计数同构查询/超图同态计数相比较，在退化模式下近一次时间性能优于一般三元模式；在非退化模式下与传统方法相比只能给出条件性下界。

**⚠️ 局限性**

限制在于：对k>3的超图模式计数问题的完全可行性分类仍受限于超图同态计数的未解决问题；且实验验证与真实数据集缺乏。

---

## 622. RANPilot: Making AI Functionalities Robust to Dynamic O-RAN Reconfigurations

**arXiv ID:** 2607.05038 | [PDF](https://arxiv.org/pdf/2607.05038v1)

**作者:** Shimin Yu `[一作]` (Hong Kong Polytechnic University), Yaxiong Xie `[通讯]` (University at Buffalo, SUNY)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `67630363-6be0-4f51-ab05-7198250671a5` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个名为 RANPilot 的框架，能够在 O‑RAN 设备升级或配置变更前，利用虚拟 O‑RAN 生成合成 KPM 数据并对 AI 模型进行预适配，从而实现 AI 功能在网络变更时的无缝切换与高鲁棒性。

**💡 创新点**

创新点主要包括：① 基于轻量级虚拟 O‑RAN 的数据合成方法，捕捉系统级交互与控制策略变化；② 通过元学习驱动的 KPM 数据增强，提升合成数据的多样性与泛化能力；③ 采用带优先级的增量学习，快速闭合仿真‑现实差距，显著缩短 AI 恢复时间；④ 将上述三大模块组合成完整的预适配流水线，实现从规划到部署的全流程自动化。

**🔧 技术方法**

使用的关键技术包括：虚拟 O‑RAN 叠加模型（抽象 CU/DU/RU 与 Open‑Fronthaul 接口）、Transformer‑based KPM 序列生成与元学习框架、KL‑Divergence 选择的增量训练与优先级调度、以及基于 KPM 的在线评估与反馈机制。

**📊 数据集**

主要数据集来自实测 5G O‑RAN 测试床，收集了超过 3000 万个 KPM 采样点，覆盖室内外多种部署环境与 10 分钟的真实流量记录。合成数据由虚拟 O‑RAN 根据这些种子 KPM 生成，随后经过元增强扩展至约 1 小时的训练集。

**📈 对比分析**

与 OWDT、Calibrated Noise Aug、TenaxDoS、CORAL 等基线对比。结果显示：在细胞添加与切换策略变更等典型变更场景下，RANPilot 将 AI 恢复时间从 29‑16 分钟压缩至 0.9‑1.5 分钟（约 94‑95% 的 downtime 缩减），并在部署后即刻达到 90‑99% 的 oracle 级精度；相比之下传统在线学习和域适配方法恢复速度慢、精度差。

**⚠️ 局限性**

局限性包括：① 依赖种子 KPM 的一致性，若在新建未覆盖区域或极端环境下，合成数据质量下降；② 对高级物理层技术（如 massive‑MIMO、毫米波 beamforming）建模较粗，难以完全捕捉 PHY 变更对 KPI 的影响；③ 目前仅针对计划性、已知变更进行预适配，对突发或非计划性网络扰动缺乏自适应机制；④ 需要人工上传变更计划和 KPM 数据，尚未实现完全 AI‑本地化的数据合成与模型更新流程。

---

## 623. RUFNet: Query-Guided Support Mask Refinement and Uncertainty Fusion based on Hybrid Mamba for Few-Shot Brain Tumor Segmentation

**arXiv ID:** 2607.05035 | [PDF](https://arxiv.org/pdf/2607.05035v1)

**作者:** Dongyi He `[一作]` (Hong Kong Polytechnic University), Nizhuan Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了RUFNet，一种融合Hybrid Mamba交互、Attention-guided Mask Refinement和Uncertainty-aware Posterior Fusion的少量样本脑瘤分割框架。

**💡 创新点**

创新点包括使用Hybrid Mamba捕获支持-查询长程依赖、通过AGMR用查询特征校正支持掩码、以及用UAPF建模像素级不确定性实现自适应后验融合。

**🔧 技术方法**

基于Hybrid Mamba的交互骨干、跨模态Attention-guided Mask Refinement模块、像素级方差估计的Uncertainty-aware Posterior Fusion，以及PSPNet+ResNet-50预训练的编码器。

**📊 数据集**

在BraTS 2020多模态MRI数据集上进行2D切片的少量样本实验。

**📈 对比分析**

与PANet、SENet、AAS-DCL、SRCL、RegFSL等方法对比，RUFNet在1-shot/5-shot情形下分别取得84.3%/86.1%的Dice和10.55mm/7.67mm的Hausdorff距离，显著优于现有方法。

**⚠️ 局限性**

局限在仅使用2D切片和二分类前景，缺乏多中心验证与3D多分类扩展，且对扫描仪多样性的置信度校准未深入探究。

---

## 624. Your Agent's Memories Are Not Its Own: Forged Reasoning Attacks on LLM Agent Memory and Defenses

**arXiv ID:** 2607.05029 | [PDF](https://arxiv.org/pdf/2607.05029v1)

**作者:** Neeraj Karamchandani `[一作]` (Pennsylvania State University), Dinghao Wu `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种名为FARMA的两阶段记忆中毒攻击，利用伪造的推理历史来误导LLM代理跳过安全检查，并提出了五层防御管线SENTINEL，其中核心的Reasoning Guard通过结构化特征检测伪造推理条目。

**💡 创新点**

创新点在于：1) 第一次将代理自身的推理历史作为攻击面；2) 设计两阶段攻击（注入+放大）以规避关键词过滤和共识式防御；3) 开发专门针对推理条目的结构化检测器Reasoning Guard，形成系统化的多层防御框架。

**🔧 技术方法**

技术包括：基于模板的伪造推理条目生成、结构相似性与关键词逃逸技术、两阶段放大机制、五层防御管线（关键词过滤、来源标签、污点阈值、模式风险筛查、Reasoning Guard结构分析）、轻量级启发式评分、模仿外部源的信任标签等。

**📊 数据集**

实验数据集涵盖三类任务域（EHR医疗记录、ReAct-QA问答、RAP购物），使用GPT‑4o‑mini、GPT‑4o、Llama 3.3 70B三种模型，随机生成50次实验，覆盖326条良性推理记录。

**📈 对比分析**

与无防御、单纯关键词过滤以及A‑MemGuard（共识式防御）比较时，FARMA在所有模型和域上均能实现高达100%的攻击成功率；SENTINEL在全部实验中将攻击成功率降至0%，并在326条良性记录中保持0%误报。

**⚠️ 局限性**

局限性包括：1) 对抗性改写攻击（熟悉Reasoning Guard规则的自适应攻击）尚未充分抵御；2) 仅在单代理单记忆存储环境评估，未涵盖多代理共享记忆场景；3) 评估在模拟环境中完成，未覆盖真实部署中的多轮交互与记忆写入频率差异。

---

## 625. Knowledge Knows, Verbalization Tells: Disentangling Latent Directions for Mathematical Solvability in LLMs

**arXiv ID:** 2607.05013 | [PDF](https://arxiv.org/pdf/2607.05013v1)

**作者:** Nikolaos Xiros `[一作]` (Athena Research Center), Georgios Paraskevopoulos `[通讯]` (Athena Research Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型内部可解性知识与可解性表述的表示，区分两者并分析其与模型制造（fabrication）的关系。

**💡 创新点**

发现可解性知识与表述在隐藏层中为独立的线性可解方向；制造主要由表述失配驱动，提示与激活导向能通过改变表述来减少制造，并且可通过联合激活导向恢复模型的放弃判断。

**🔧 技术方法**

采用线性探针、激活导向、LLM‑as‑judge 评判、提示工程和可解释的激活 steering 等技术。

**📊 数据集**

使用 ReliableMath 基准集，包含可解与不可解的数学问题。

**📈 对比分析**

通过 ROC‑AUC、AUC、余弦相似度、放弃率/制造率等指标评估，在多模型（Qwen、Llama、Gemma 等）上显著提升放弃率并降低制造；表述方向对性能提升贡献显著，知识方向对性能影响相对较小。

**⚠️ 局限性**

仅聚焦数学可解性，未验证在更广泛的 QA 任务中的泛化；未研究内部表示与生成文本的更传统 faithfulness 关系；在无门控的激活导向会导致高误报率。

---

## 626. FAST: A Holistic Framework for Optimizing Memory-I/O, Computation, and Sampling in Temporal GNN Training

**arXiv ID:** 2607.05095 | [PDF](https://arxiv.org/pdf/2607.05095v1)

**作者:** Yushu Cai `[一作]` (Xidian University), Xin He `[通讯]` (Xidian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套名为FAST的统一框架，用于加速大规模时序图神经网络（TGNN）的端到端训练。

**💡 创新点**

创新点在于三方面的协同优化：① SlimCache将压缩与缓存合并以减少主机-设备数据移动；② 设计了线程高效的图算子（边缘中心聚合与CSR化的Softmax），缓解负载不均和缓存缺失；③ 通过拓扑感知采样策略，将采样线程映射到CPU核上，提升缓存局部性。

**🔧 技术方法**

主要技术包括基于ID压缩的内存I/O优化、GPU线程调度与COO/CSR算子实现、拓扑相似度矩阵与Blossom匹配的线程绑定、以及基于热ID贪心选取的缓存分配。

**📊 数据集**

在四个大规模动态图数据集上评估：LastFM、WikiTalk、Bitcoin、GDELT，使用TGAT、TGN、DySAT三种TGNN骨干。

**📈 对比分析**

与TGL、ETC、SIMPLE等现有框架对比，FAST平均提升约2.1×（最高4.7×）的训练速度，且保持与基线相同的平均精度，展示了跨阶段协同优化的显著效果。

**⚠️ 局限性**

局限性包括：仅在单机单GPU环境下实现；对离散时间动态图的支持尚未充分；高阶优化（如多GPU分布式训练、硬件加速等）未涉及；以及在极端大图（>10亿边）下仍受内存与I/O瓶颈限制。

---

## 627. ECO: Incremental Ego-Centric Octree Update for Point Streams

**arXiv ID:** 2607.05092 | [PDF](https://arxiv.org/pdf/2607.05092v1)

**作者:** Jaemin Yu `[一作]` (Korea University of Technology and Education), Duksu Kim `[通讯]` (Korea University of Technology and Education)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种以机器人自身为中心的固定边界 Octree（ECO），并提出了基于 shift‑out/shift‑in/overlap 区域的增量更新算法，用于移动机器人实时点云地图构建。

**💡 创新点**

创新点在于：①将 Octree 作为 3D 滑动窗口，仅维护机器人附近的有限体积；②按三种空间关系（shift‑out、shift‑in、overlap）划分更新步骤，避免全局坐标变换；③通过在每次更新中保持树的平衡，显著降低计算和内存开销；④天然保留短期运动物体的历史轨迹，提供时序上下文。

**🔧 技术方法**

采用 Octree 结构、AABB 维护、增量更新策略、懒加载 world→local 坐标转换、点云插入/删除、CPU 单核心实现，评估使用 KITTI 激光雷达序列。

**📊 数据集**

使用 KITTI 视觉/激光雷达数据集中的序列 00、02、08 进行静态与动态场景的基准测试。

**📈 对比分析**

通过与三种基线（全局重建 Octree_base、全局增量 i-Octree_global、受限增量 i-Octree_target）的比较，实验表明：在静态场景中，ECO 的更新时间比 Octree_base 降低 25.6%（平均 24.87%），比 i-Octree_target 降低 67.52%（平均 54.60%）；在动态场景中，ECO 比 Octree_base 提速 1.42 倍；在体素化和 KNN 搜索的整体系统延迟上，ECO 始终保持最低值。

**⚠️ 局限性**

局限性包括：①边界固定为立方体，仅处理平移而忽略旋转；②在大范围移动时仍需手动设定合适的边界大小；③未能处理复杂的三维旋转导致的轴对齐误差。

---

## 628. Be Indiscrete: The Benefits of Learning Continuous Spine Degeneration Severity Scores

**arXiv ID:** 2607.05090 | [PDF](https://arxiv.org/pdf/2607.05090v1)

**作者:** Maria Monzon `[一作]` (ETH Zurich), Amir Jamaludin `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出并实现了 SpineRankNet，利用连续的严重程度评分对腰椎 MRI 进行多病理学的分级。

**💡 创新点**

创新点在于将严重程度建模为连续数值，并引入严重度自适应的两阶边缘损失、同级相似度约束和软阈值化评分，从而在保持分级顺序的同时减小大误差。

**🔧 技术方法**

技术上使用 3D ResNet‑18 编码器、两层 MLP 评分头、基于 pairwise 的加权 hinge 损失、soft‑plus 软阈值化以及多视角 TTA 等方法。

**📊 数据集**

实验数据来自 Genodisc 多中心数据集，约 2000 名受试者，涵盖 11 种腰椎分级任务。

**📈 对比分析**

与传统分类、序数回归及其他排名方法对比，SpineRankNet 在 QWK、MAE、ROC‑AUC 等指标上均表现最佳，尤其显著降低了跨级错误。

**⚠️ 局限性**

局限性包括仅在单一数据集上验证，缺乏外部独立样本、纵向验证以及临床评估的实际测试。

---

## 629. LangLoc: "Tell Me What You See"

**arXiv ID:** 2607.05077 | [PDF](https://arxiv.org/pdf/2607.05077v1)

**作者:** Shaurya Kishore Panwar `[一作]` (ETH Zürich), Daniel Barath `[通讯]` (ETH Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一套从自然语言描述中精确定位室内环境中观测者的二维位置和朝向的完整工作流。

**💡 创新点**

创新点在于：①使用双分支 GATv2 编码器结合 CLIP 语义特征实现高精度场景检索；②基于光线投射可见性评分的密集地面网格定位方法；③交互式 Bayesian 对话模块通过有针对性的是/否问题消除多模态定位不确定性。

**🔧 技术方法**

技术包括：图神经网络（GATv2）、CLIP 文本/视觉编码、InfoNCE 对比学习、光线投射可见性计数、贝叶斯后验更新、信息增益式问题选择。

**📊 数据集**

使用新构建的 LangLoc 数据集（13k+ 位置索引自然语言描述，覆盖 1,300+ 室内 3D 扫描）以及 ScanNet 作为评估基准。

**📈 对比分析**

在场景检索方面 Top‑1 Recall 提升 8pp 以上，Fine‑Localization 位置误差中位数约 0.95 m，角度误差 39.8°；加入对话后 ScanNet 位置误差降至 7 cm、角度误差 5°，大幅超越 VLM 与传统基线。

**⚠️ 局限性**

局限性包括：依赖预先构建且已标注的 3D 场景图；在大型或高拥挤度环境中多视角相似导致定位仍有挑战；需要完整 3D 模型，难以直接应用于无模型或低分辨率地图的场景。

---

## 630. MIRAGE: Defending Long-Form RAG Against Misinformation Pollution

**arXiv ID:** 2607.05069 | [PDF](https://arxiv.org/pdf/2607.05069v1)

**作者:** Saadeldine Eletter `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Preslav Nakov `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

MIRAGE是一种训练无关、模型无关的检索增强生成（RAG）防御方法，先构建跨文档的自然语言推理（NLI）支持/矛盾图，基于图一致性门控生成；

**💡 创新点**

其创新点在于利用多源一致性图来检测并剔除污染证据，结合“已验证声明门”在生成前主动过滤信息，而非仅靠检索或后置校验；

**🔧 技术方法**

技术手段包括句子级声明抽取、token-overlap筛选、NLI推理、支持/矛盾权重计算、图稀疏化与门控判定，以及对关键句子进行自检证据查询；

**📊 数据集**

实验使用四大长文本问答数据集（LongFact、FAVA、AlpacaFact、Biography）与多款商业及开源LLM（GPT‑4o‑mini、LLaMA‑3.1‑8B、Qwen‑3‑8B等）；

**📈 对比分析**

与现有强健RAG基线相比，MIRAGE在混合与完全污染检索下的VeriScore F1@k均显著提升，混合污染平均提升约30%，完全污染下可恢复至或超过无检索基线；

**⚠️ 局限性**

局限性包括依赖检索覆盖和来源多样性，低覆盖或单源信息导致过度拦截；NLI与token-overlap可能错过改写或省略式误导；对协调性误导难以识别；并且额外的NLI计算带来运行时开销。

---

## 631. Diffusion-Guided Uncertainty-Aware Delayed Policy Optimization

**arXiv ID:** 2607.05064 | [PDF](https://arxiv.org/pdf/2607.05064v1)

**作者:** Junqi Tu `[一作]` (East China University of Science and Technology), Yang Tang `[通讯]` (East China University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了一种针对随机观察延迟的强化学习方法 DUPO，能够在延迟信息下稳健决策。

**💡 创新点**

创新点在于利用条件扩散模型对延迟信息的后验分布进行多模态建模，并通过估计动作价值的不确定性来动态加权策略更新，克服了单点预测导致的性能衰退。

**🔧 技术方法**

核心技术包括条件扩散模型、基于不确定性的策略加权、以及 Soft Actor‑Critic（SAC）的延迟适配实现。

**📊 数据集**

在改造后的 MuJoCo 连续控制任务（Ant‑v4、HalfCheetah‑v4、Hopper‑v4、Walker2d‑v4、Swimmer‑v4）上进行实验，任务加入高斯噪声并随机化观测延迟。

**📈 对比分析**

与 DC/AC、State‑Augmentation、State‑Prediction 等基线对比，DUPO 在多种延迟预算（ΔT_max=5、10、25）下均显著提升归一化收益，尤其在中至长延迟时性能提升最为显著。

**⚠️ 局限性**

局限性包括对扩散模型训练所需的计算开销较大、对延迟分布假设的敏感度以及在极端高延迟或非高斯噪声环境中的进一步验证待探索。

---

## 632. Toward Trustworthy Large Language Model Agents in Healthcare

**arXiv ID:** 2607.05055 | [PDF](https://arxiv.org/pdf/2607.05055v1)

**作者:** Hadi Hasan `[一作]` (American University of Beirut), Ali Chehab `[通讯]` (American University of Beirut)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了 CareConnect，基于 LLM 的对话式代理，用于医疗预约调度的安全可靠自动化。

**💡 创新点**

引入多层安全架构，结合预‑LLM 意图分类、方案约束的工具调用以及 RAG‑工具混合流水线，实现在操作层面的可审计安全与事务完整性。

**🔧 技术方法**

使用 GPT‑4o 原生函数调用、规则式意图检测、ChromaDB 向量检索、Pydantic/SQLAlchemy 架构验证，前端 React/TypeScript、后端 FastAPI、Docker 容器化。

**📊 数据集**

采用 30 名合成患者、60 名合成医护、20 个科室的全合成数据库和 680 个人工生成的对话场景作为评测集。

**📈 对比分析**

与人工接待员基准比较，任务完成率 91.8%（vs 85%），平均延迟 2.2 s，安全合规率 96%，每次预约成本 0.0324 美元，远低于人工 0.75 美元。

**⚠️ 局限性**

在约束冲突、长对话上下文漂移和相对时间表达处理上仍出现失败，缺乏形式化的约束求解器与真实环境验证。

---

## 633. CollabEval: Statistically Efficient Collaborative Model Evaluation via Matrix Completion

**arXiv ID:** 2607.05046 | [PDF](https://arxiv.org/pdf/2607.05046v1)

**作者:** Adam Fisch `[一作]` (Google DeepMind), Jacob Eisenstein `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Collaborative Evaluation（CollabEval）框架，通过矩阵补全预测缺失的模型评估分数，并将这些预测作为控制变量应用于交叉预测推理，从而在不执行模型推理的情况下提高评估精度。

**💡 创新点**

创新点在于将评估任务视为协同过滤/矩阵补全问题，利用历史 anchor 模型的评分信息构造高质量预测；同时结合交叉预测推理保证无偏估计并给出渐近有效置信区间，克服传统控制变量方法对预测准确度的强假设。

**🔧 技术方法**

核心技术包括低秩矩阵补全（Iterative SVD、核范数最小化等）、交叉折叠矩阵补全（cross‑fold matrix completion）以及控制变量推断（cross‑prediction‑powered inference）和置信区间估计。

**📊 数据集**

实验使用五个多样化文本生成与评估基准：AlpacaEval 2.0、MMLU、Attributed Question Answering (AQA)、WMT24++ 机器翻译以及 SWE‑bench 仓库级补丁生成。

**📈 对比分析**

与经典样本均值、naïve 补全、Anchor Points、PPI‑Anchor Mean 等基线比较，CollabEval 在 CI 宽度上平均缩小 20–30%，在小样本比例下实现 10–20% 的有效样本量提升，且始终保持 90% 置信区间覆盖率，证明其在减少标注成本方面显著优于现有方法。

**⚠️ 局限性**

局限性包括对 anchor 模型的依赖——若 anchor 与 target 相关性弱，收益有限；对低秩结构的假设在极端稀疏或高度异质的数据上可能不成立；虽然矩阵补全成本低，但在极大规模评估矩阵时仍需额外计算；理论保证仅在渐近条件下成立，实际中需要足够的观测数据。

---

## 634. Unsupervised Pixel-Level Semantic Left-Right Understanding of In-the-Wild Images

**arXiv ID:** 2607.05006 | [PDF](https://arxiv.org/pdf/2607.05006v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 635. RADIANCE: Relative Adaptive Denoising with IP-Adapter for Novel Concept Enhancement

**arXiv ID:** 2607.05088 | [PDF](https://arxiv.org/pdf/2607.05088v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 636. LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review

**arXiv ID:** 2607.05031 | [PDF](https://arxiv.org/pdf/2607.05031v1)

**作者:** Ali Hassaan Mughal `[一作]` (Independent Researcher), Muhammad Bilal `[通讯]` (Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 54 篇 LLM 驱动的测试判据（test oracle）研究进行系统综述，提出按判据权威来源分类的框架，并分析判据形式、判决机制、数据集、评估方法和现有研究空白。

**💡 创新点**

首次以“判据权威来源”为核心维度构建 LLM 测试判据的来源–机制交叉分类法，揭示权威来源与判决机制分离、主流来源倾向、弱判据与模型幻觉等关键问题，并指出未被充分探索的研究方向。

**🔧 技术方法**

采用 PRISMA‑2020 系统综述方法，使用 Claude Opus 进行自动预筛选，再由双人独立人工进行筛选与编码；对每项研究按七个维度（来源、形式、机制、应用域、模型、数据集、评估等）进行编码。

**📊 数据集**

综述中涉及的主要基准为自定义数据集（占 59%），常见公共基准包括 Defects4J、HumanEval、JHED 等；本综述本身不使用实验数据集，只对现有研究的数据集进行统计。

**📈 对比分析**

该工作不执行实验比较，而是通过描述性统计汇总判据形式（断言、期望值、变形关系等）与判决机制（运行时断言、Exact‑match、LLM‑as‑judge 等）分布；未给出性能度量或统一基准比较。

**⚠️ 局限性**

局限性包括：仅检索 Scopus、IEEE Xplore 与 ACM DL 3 个数据库，忽略 Web of Science、Springer 等；自动预筛选虽回收率高但可能漏检；缺乏统一的故障驱动评估基准，研究多基于自定义基准；LLM‑as‑judge 在部分来源（实现‑派生、参考‑差分）未被测试；并未系统评估判据的真实错误率。

---

## 637. New Results on Limited Magnitude Error Correcting Codes

**arXiv ID:** 2607.05026 | [PDF](https://arxiv.org/pdf/2607.05026v1)

**作者:** Zhiyu Yuan `[一作]` (Peking University), Gennian Ge `[通讯]` (Capital Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了有限幅度误差纠正码的理论，探讨了分裂集、群分裂与格子铺垫，完成了对非奇异准完美 B[0,3](n) 集合与 B[-4,4](2p) 集合的完整分类，并确定了完美 B[0,6](q) 集合的存在条件，利用图论改进了 M(0,3;q) 的下界，并给出了一套通用框架和新的无限族完美限幅突发错误纠正码。

**💡 创新点**

创新点在于：1）对非奇异准完美 B[0,3](n) 与 B[-4,4](2p) 集合给出了完全分类；2）通过图论与数论技术阐明了完美 B[0,6](q) 的存在条件；3）提出并实现了一般性分裂与格子铺垫框架，构造了新的无限族完美限幅突发码。

**🔧 技术方法**

采用的技术主要包括群分裂理论、有限域与循环码的分裂技术、韦尔不等式与乘子符号求解、切比雪夫–克氏定理、图论中的匹配与独立集、以及基于多项式与指数向量的符号化处理。

**📊 数据集**

论文为理论研究，无使用传统数据集。

**📈 对比分析**

未进行实验比较，所有结果均为理论证明，展示了存在性与构造方法的可行性。

**⚠️ 局限性**

局限性在于：M(0,3;q) 的下界尚未达到最优，非循环突发码的构造与分析仍有待完善。

---

## 638. Multi-Large Language Model Orchestrated Severity Assessment of Clinical Records (MOSAIC)

**arXiv ID:** 2607.05032 | [PDF](https://arxiv.org/pdf/2607.05032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 639. Hyperparameter Transfer in Graph Neural Networks

**arXiv ID:** 2607.05017 | [PDF](https://arxiv.org/pdf/2607.05017v1)

**作者:** Gage DeZoort `[一作]` (Princeton University), Boris Hanin `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种针对图神经网络的超参数迁移参数化，使得在不同宽度和深度的模型中可使用相同的学习率、初始化与正则化设置；

**💡 创新点**

创新点在于为SGD、Adam、AdamW分别给出精确的学习率缩放和首层学习率校正因子，并引入消息传递归一化（γ）来解决图输入异质性与数值不稳定问题；

**🔧 技术方法**

利用Tensor Programs、DMFT等理论框架进行宽度/深度极限分析，并在实验中使用自适应学习率搜索、层归一化、γ归一化等技术；

**📊 数据集**

在图分类（MNIST Superpixels）、节点分类（PascalVOC‑SP）、分子回归（QM9）以及半监督引用网络（Cora、Citeseer、Pubmed）等数据集上验证；

**📈 对比分析**

通过与单模型手动调参对比，发现迁移参数化能够在大模型上保持或提升性能，同时显著降低调参成本；

**⚠️ 局限性**

局限在于仅对简化的线性化GNN结构给出理论分析，缺乏对包含注意力、边特征等复杂模块的深入研究，且实验规模与数据集仍有限。

---

## 640. ImputeECG: Deep Learning Reconstruction of Complete 12-Lead Electrocardiograms from Incomplete Recordings for Cardiac Assessment

**arXiv ID:** 2607.05009 | [PDF](https://arxiv.org/pdf/2607.05009v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 641. Tempus fugit: Anyone can understand temporal logic if they have to save the realm

**arXiv ID:** 2607.05062 | [PDF](https://arxiv.org/pdf/2607.05062v1)

**作者:** Benjamin Bisping `[一作]` (Télécom SudParis, Institut Polytechnique de Paris), Maximilian Lukas Stamm `[通讯]` (Technische Universität Berlin)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一款基于线性时序逻辑（LTL）的小型浏览器卡牌游戏《Tempus fugit》，通过游戏玩法帮助玩家学习并直观理解时序逻辑的语法和语义。

**💡 创新点**

创新点在于将时序逻辑与可玩游戏机制紧密结合，利用游戏中的“运行时间表”和“符文赋值”直观展示逻辑公式的满足与否，并将游戏过程与教学目标对齐；此外，游戏采用了事件驱动架构与前端TypeScript实现，提升了可维护性与可扩展性。

**🔧 技术方法**

技术包括：TypeScript + Phaser 3 前端框架；自定义 LTL 解析器（使用改进的 Shunting‑Yard 算法生成后缀表达式并构建抽象语法树）；事件驱动架构实现游戏逻辑与 UI 交互；客户端本地存储记录进度。

**📊 数据集**

未使用公开数据集；游戏内容（卡牌、关卡、敌人）通过 JSON 配置文件手工编写，基于作者设计的示例公式和剧情。

**📈 对比分析**

论文未给出系统性能指标或与其他教学工具的对比；仅在实验说明中提到游戏可在 <10 MB 体积下在浏览器运行，且不需网络连接，可在科学传播场景中使用。

**⚠️ 局限性**

局限性包括：仅覆盖有限的时序逻辑子集（缺少 until/since 实际使用）；未深入评估学习效果与教学效益；对移动端兼容性不足；游戏侧重于娱乐与直观感知，无法取代正式课程或深入逻辑研究。

---

## 642. Understanding Student Perceptions, Mistakes, and Debugging Approaches when Solving Natural Language Programming Tasks

**arXiv ID:** 2607.05034 | [PDF](https://arxiv.org/pdf/2607.05034v1)

**作者:** Victor-Alexandru Pădurean `[一作]` (MPI-SWS), Adish Singla `[通讯]` (MPI-SWS)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一门本科级 C 语言入门课程中，构建并使用基于 GPT‑4o‑mini 的对话式自然语言提示编程工具，让学生通过自然语言描述问题并与模型交互，以生成并调试代码，同时收集学生的交互日志和反思。

**💡 创新点**

首次系统量化和分析初学者在对话式提示中的错误类型与共现关系，并探究其调试策略；提出基于错误的教学设计建议，填补了缺乏对话式提示学生体验与错误研究的空白。

**🔧 技术方法**

技术手段包括自研的 Prompt Programming web 工具、GPT‑4o‑mini 生成模型、对话日志记录与代码执行接口；使用定性编码、统计分析和相关性检验来处理数据。

**📊 数据集**

使用约 1,000 名学生在两周实验中完成的 6 个 Prompt Problem 的对话、代码执行和错误信息，形成的交互数据集；不涉及公开数据集，而是自建实验数据。

**📈 对比分析**

通过问卷（Likert 量表）和定性反思对比学生对传统编码与对话式提示的感知难度；描述性统计评估成功率、消息数、错误率；对错误共现进行 Pearson 相关分析；结果表明学生普遍认为对话式提示更易、更快，但错误主要集中在缺失返回值、参数名等，调试多聚焦于重新描述问题。

**⚠️ 局限性**

局限性：实验仅限于 C 语言入门课程，任务过于简单；未进行对话式提示与单次提示的对照实验；未获取学生背景信息，无法评估公平性；未直接测量认知负荷或学习成效；工具不支持代码直接编辑，可能导致学生过度依赖 LLM。

---

## 643. Algorithmically Presented Numbers and Canonical Representations in Cryptographic Protocols

**arXiv ID:** 2607.05016 | [PDF](https://arxiv.org/pdf/2607.05016v1)

**作者:** Arslan Brömme `[一作]` `[通讯]`, Arslan Brömme

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文从可计算分析与密码协议的交叉点出发，构建了一个基于表示论的框架，阐述了数值（尤其是可计算实数）在协议中需要的“算法化表示”与“规范化”需求，并通过理想化的取整、示例与实际的 SnapRoot 哈希锚定协议演示了规范化对协议可互操作性与可验证性的影响。

**💡 创新点**

创新点在于：① 将可计算实数的等价性不可判定性与协议中的“字节输入不变性”需求正式对应，提出了“canonicalization barrier”与设计三元悖论；② 统一定义了“算法化表示数”“有限确切可描述”“可规范化系统”三种层级概念；③ 通过“有理核心呈现”(S_= , (S_)) 与“可规范序列化对象类”推广到文件、哈希、交易 ID 等实际协议对象；④ 在 SnapRoot 案例中展示了协议层对规范化的真实需求。

**🔧 技术方法**

主要技术包括：可计算分析（可计算实数、Cauchy 名称、可判定性分析）；表示论（表示系统、规范化函数、canonical encoding）；形式化证明（不可判定性证明、injectivity lemma、well‑definedness 证明）；以及对协议执行的抽象模型（加密、哈希、签名流程）。

**📊 数据集**

未使用传统机器学习或网络数据集；通过自行构造的代数示例（如 13/37、π、RSA 参数）以及 SnapRoot 协议的真实链上交易记录进行演示与验证。

**📈 对比分析**

比较方法主要是理论证明与案例对比：① 通过“canonical encoding”与“deterministic serialization”对比展示字节不变性的实现；② 通过对同一算法（XOR、RSA、哈希）在规范化与非规范化输入下的可重复性和一致性进行比较；③ SnapRoot 示例通过对比文件哈希、结构化载荷与协议规范说明实现了一致性验证；性能方面未给出定量指标，重点在可验证性与一致性保证。

**⚠️ 局限性**

局限性包括：① 仅采用较窄的表示系统定义，未覆盖一般 Type‑2 表示空间；② 未进行安全性分析，仅关注可验证性与一致性；③ 仅通过小规模代数例子与单一真实协议（SnapRoot）说明，缺乏广泛的实证评测；④ 讨论的三元悖论与规范化阈值在更复杂协议中的具体实现仍待进一步研究。

---

## 644. Comparison of Loss Functions for Robust Deep Learning-based Echocardiography Segmentation when Learning with Partially Labelled Data from Multiple Domains

**arXiv ID:** 2607.05008 | [PDF](https://arxiv.org/pdf/2607.05008v1)

**作者:** Iman Islam `[一作]` (King's College London), Andrew P. King `[通讯]` (King's College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对多数据集、部分标注的心脏超声分割问题进行系统研究，比较三种主要损失函数在单域和跨域场景下的性能，并评估标签缺失比例与数据量对模型效果的影响。

**💡 创新点**

首次在心脏超声分割领域进行完整的损失函数比较；提出并验证了在跨域情况下使用标签抑制（label dropout）可显著缓解 aCCE 损失性能下降；系统量化了不同数据量与标注比例对模型泛化的影响。

**🔧 技术方法**

使用 U‑Net、Attention U‑Net 与 Swin‑Unet 三种网络架构；采用三种部分标注损失（adaptive CCE、边缘损失、adaptive BCE）和标准 CCE 作为基线；结合数据增强、nnU‑Net 预处理、随机种子重复实验等技术。

**📊 数据集**

CAMUS（全标注）、Unity Imaging（全标注）与 EchoNet‑Dynamic（仅标注 LV）三大公开超声数据集；通过人工标注补全 EchoNet‑Dynamic 的 LA 与 LVM 作为评估基准。

**📈 对比分析**

通过 Dice 交叉验证、MAE（EF 误差）以及训练时间比较；实验显示 aCCE+标签抑制、边缘损失与 aBCE 在跨域场景中性能相近，边缘损失在大多数情况下略优；相较于伪标注方法，三种损失在部分标注下更稳健，且训练时间差异不大。

**⚠️ 局限性**

仅包含三组数据集，标注质量存在差异；缺乏更大规模、更多域的外部验证；仅探讨损失函数方案，未涉及网络架构改进或质量评估预处理等进一步方法。

---

## 645. Counterfactual Methods for Detecting Unfairness in Anti-Money Laundering Algorithms

**arXiv ID:** 2607.05101 | [PDF](https://arxiv.org/pdf/2607.05101v1)

**作者:** Lea Multerer `[一作]` (IDSIA), Martina Gogova `[通讯]` (UBS Switzerland AG and its affiliates)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对 AML 机器学习模型进行因果路径特定的公平性分析，检验国家等 KYC 敏感特征对模型预测的直接与间接影响。

**💡 创新点**

首次将结构因果模型与自然直接/间接效应分解结合图神经网络的反事实干预，用于量化 AML 模型准确率与公平性之间的权衡。

**🔧 技术方法**

采用结构因果模型、自然直接/间接效应估计、树回归逼近介导变量、XGBoost、图同构网络（GNN‑GIN）和邻域聚合网络（GNN‑PNA）等技术。

**📊 数据集**

使用 IBM AMLSim 的 HI‑Small 合成交易数据，扩展为包含国家与交易行为的特征。

**📈 对比分析**

对比 XGBoost、GNN‑GIN 与 GNN‑PNA 在基础与扩展特征下的精确率、召回率与 F1，并通过反事实翻转率、相对总/直接/间接效应评估公平性；结果显示 GNN‑GIN 在准确率提升最大但公平性下降，GNN‑PNA 公平性最好。

**⚠️ 局限性**

局限在于合成数据与伪 KYC 特征的构造、介导变量模型的近似、图神经网络干预的简化，以及对真实监管环境中特征关系和因果图假设的依赖。

---

## 646. TimeThink: Reasoning with Time for Video LLMs

**arXiv ID:** 2607.05089 | [PDF](https://arxiv.org/pdf/2607.05089v1)

**作者:** Handong Li `[一作]` (University of Chinese Academy of Sciences), Jing Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TimeThink 框架，利用强化学习在视频大语言模型中通过时间线索步骤进行时序推理，并引入逐步时间过程奖励与联合过程‑结果优化。

**💡 创新点**

创新点包括：① 将时间线索步骤作为优化原语，直接对每一步的时间区间进行奖励；② 设计逐步 IoU 过程奖励，提供局部信用分配；③ 构建自动生成的 TimeThink‑RFT‑20K 证据集支持过程奖励；④ 采用两阶段 RFT 训练，先强化时序推理后迁移到通用任务。

**🔧 技术方法**

核心技术包括：视频‑LLM（Qwen2.5‑VL‑7B）+视频编码器；Group Relative Policy Optimization (GRPO)；token‑级优势计算与 KL 正则化；时间线索步骤与 IoU 奖励；自动化时间证据生成与教师模型（Qwen3‑VL‑235B）推理。

**📊 数据集**

使用数据集：TimeThink‑RFT‑20K（自动生成时间证据段），LLaVA‑Video‑178K，DiDeMo 77K，ActivityNet Captions，Charades‑STA，CGBench，NExT‑GQA，VideoMMMU，VSIBench，MLVU，VideoMME，LongVideoBench，MVBench 等多种视频推理、定位与通用理解基准。

**📈 对比分析**

在七大基准（视频推理、时序定位、通用视频理解）上与 SFT、传统 RL（仅结果奖励）和其它 RFT 方法对比，TimeThink 在时序定位上取得 mIoU 55.3%（Charades‑STA）并在 VideoMMMU、VSIBench 等推理任务上超过 52.6%/0.5% 等，整体性能达到或接近 GPT‑4o/ Gemini‑1.5‑Pro 的水平，显著优于现有开源 RL 模型。

**⚠️ 局限性**

局限性：① 过程奖励依赖自动生成的时间证据，仍缺乏高精度人工标注；② 对帧率和视频长度敏感，长视频或高帧率时推理步骤易过长或冗余；③ 训练需大规模 GPU 资源，成本高；④ 对跨时间复杂因果推理的能力仍有限。

---

## 647. Computing Monetary Risk Measures in Linear Time

**arXiv ID:** 2607.05078 | [PDF](https://arxiv.org/pdf/2607.05078v1)

**作者:** Palash Agrawal `[一作]`, Marek Petrik `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了 QuickVaR、QuickDivergence 以及对应的 QuickCVaR、QuickTVaR 等算法，实现了在离散随机变量上以期望线性时间计算 VaR、CVaR、TVaR 等常用风险度量。

**💡 创新点**

核心创新在于：① 将 Quickselect 思想应用于 VaR 计算，得到 QuickVaR；② 引入 EWS（Element‑Wise‑Separable）子模函数与多项式优化相结合，构造 QuickDivergence，能够在不排序的情况下完成对 φ‑divergence 风险度量的线性优化；③ 证明 CVaR 与 TVaR 可归约为此类 EWS 多项式问题，从而实现 O(n) 期望复杂度。

**🔧 技术方法**

技术手段包括：快速选择（Quickselect）算法、分治与划分（Dutch national flag）技术、子模优化与多项式约束、概率分布的权重分配、以及对 EWS 函数的凸性与子模性分析。

**📊 数据集**

使用的数据集主要有：① 大规模合成均匀分布随机变量（n 取 10^6 至 10^7）；② 稀疏支持随机变量；③ 真实金融股票收益的随机变量；④ 小规模稀疏随机变量（n 取 1,000–10,000）。

**📈 对比分析**

与传统基于排序的 VaR/CVaR/TVaR 实现（O(n log n)）以及期望计算基准进行对比。实验结果显示：在大规模数据下，Quick 系列算法的运行时间比基线低 5–10 倍；在中小规模数据上亦保持一定优势；在股票市场场景中，QVaR、QCVaR、QTVaR 的平均耗时分别为 0.13、0.22、0.25 ms，明显快于对应排序实现。

**⚠️ 局限性**

局限性包括：① 对 α = 0 或 α = 1 的极端情况处理仍需特殊处理；② 对非离散（连续）分布的直接应用不适用；③ 对于极端稀疏分布时，分区递归深度可能较大，导致递归栈消耗；④ 目前实现仅针对单个随机变量，未考虑多维联合分布的风险度量。

---

## 648. Beyond Independent Labels: Schwartz-Geometry Decoding for Human Value Detection

**arXiv ID:** 2607.05052 | [PDF](https://arxiv.org/pdf/2607.05052v1)

**作者:** Víctor Yeste `[一作]` (PRHLT Research Center), Paolo Rosso `[通讯]` (PRHLT Research Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对句子级人类价值检测任务，作者将19个精细化的Schwartz价值视为圆形激励连续体，设计了一种后置结构化解码器，使预测的标签集合在保持分类准确率不变的前提下更符合该连续体的兼容与冲突关系。

**💡 创新点**

创新点在于：①将心理学理论直接编码为输出空间的几何结构，并在解码阶段应用，而非硬性约束或在网络结构中嵌入；②提出一套基于圆形距离的“理论意识”一致性度量；③通过与随机与经验共现控制几何的对照，验证真实Schwartz排列才产生显著一致性提升。

**🔧 技术方法**

技术手段包括：使用DeBERTa‑v3‑base作为基线分类器；在训练时尝试GeoLoss、GeoSmooth等几何正则；最核心的Schwartz‑aware能量解码器（Unary + 对偶兼容/冲突权重 + 卡迪纳尔约束）；以及在LLM（Qwen2.5‑72B‑Instruct）上做基准对比。

**📊 数据集**

使用的数据集为Touché24‑ValueEval，按文档划分为训练/验证/测试三份，包含约44k句子、19个标签，约一半句子至少出现一价值。

**📈 对比分析**

实验结果显示：①训练时加入几何正则（GeoLoss/GeoSmooth）对Macro‑F1、Micro‑F1影响不大，且GeoSmooth甚至下降；②后置解码器在保持Macro‑F1≈0.294、Micro‑F1≈0.343的前提下，将“几何一致性成本”从0.5634降至0.5480，显著低于随机/经验几何；③LLM在零样本提示下表现更差，提示Schwartz连续体虽能稍微提升一致性，但仍未达到监督解码器水平。

**⚠️ 局限性**

局限性包括：只在单一英语句子级数据集和单一基线模型上验证；一致性度量是作者自定义，缺乏公开基准；后置解码器无法补偿基线未捕获的标签；软几何惩罚可能不适用于真实冲突表达；LLM诊断仅用单一模型与提示，结果易受模型/提示变化影响。

---

## 649. Listen, Think, Transcribe: Continuous Latent Test-Time Scaling for ASR

**arXiv ID:** 2607.05051 | [PDF](https://arxiv.org/pdf/2607.05051v1)

**作者:** Ho Lam Chung `[一作]` (National Taiwan University), Hung-yi Lee `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在不修改原始 ASR 模型权重的前提下，加入一个可学习的连续潜在计算循环，对少量（500 条）多样化训练样本进行激活，提升解码质量。

**💡 创新点**

创新点在于：①利用“潜在测试时缩放”在冻结的端到端 ASR 里实现迭代细化；②通过一个带有边界、门控和锚点的潜在适配器保证不破坏解码器输入分布；③加入价值头根据每条语音的预测收益动态停止迭代，做到输入依赖计算。

**🔧 技术方法**

核心技术包括：连续潜在层的有界更新（L2 归一化 + 可学习尺度 + sigmoid 门控），固定嵌入锚点注入，价值头回归预测潜在提升值，循环轨迹正则化，以及随机噪声/dropout 的训练正则。

**📊 数据集**

使用 500 条跨语言、跨域的语料（Common Voice、FLEURS、VoxPopuli 等）做激活训练，并在 FLEURS、VoxPopuli、ASCEND、30 种 FLEURS 语言及噪声/重音数据集上评估。

**📈 对比分析**

与全微调、LoRA、Prompt 调整等常规参数高效适配方法相比，在同样的 500 条训练集下，LatentASR 能在 FLEURS、VoxPopuli 上分别下降约 2.5% 和 0.5% 的 WER，并在重音、代码切换语料上实现 16% 的 CER 降幅；动态停顿使平均步骤从 4 降至 1 左右，计算成本显著降低。

**⚠️ 局限性**

局限性：①仅在“极小数据激活”范围内有效，增大训练集会导致背骨被破坏；②对不同语言/域的泛化尚未验证到更大规模；③价值头的阈值和激活集的构成对结果敏感，需细致调参。

---

## 650. Security Analysis of RIS-Assisted Physical-Layer Authentication Over Multipath Channels

**arXiv ID:** 2607.05042 | [PDF](https://arxiv.org/pdf/2607.05042v1)

**作者:** Linda Senigagliesi `[一作]` (ETIS UMR 8051), Stefano Tomasin `[通讯]` (University of Padova)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在基于重构智能表面（RIS）的物理层身份验证（PLA）中，攻击者Trudy拥有完整信道知识时的最佳攻击策略与不可区分条件，并通过数值仿真验证单路径与多路径RIS–Bob信道对攻击鲁棒性的影响。

**💡 创新点**

创新点在于：①给出了在RIS辅助PLA场景下，攻击者能实现的最佳预编码向量和相应的不可区分判定条件；②分析了单路径与多路径RIS–Bob通道对攻击者的可行性差异，揭示了多路径可显著提升身份验证安全性；③在全知攻击者假设下提供了对PLA安全性的最严苛评估。

**🔧 技术方法**

使用的技术包括：基于几何多路径模型的信道描述、线性最小化误差（MLE）预编码优化、距离阈值检测（LT）以及基于概率统计的误检/误接受率分析。

**📊 数据集**

实验使用随机生成的复高斯路径增益（γ）和均匀分布的相位角（AoA/AoD），RIS元素数N=64，Bob天线数M∈{4,8,16,32}，单路径与三路径场景进行比较。

**📈 对比分析**

方法对比主要通过检测误差曲线（DET）以及FA/MD阈值来评估。结果显示：单路径RIS–Bob时，攻击者可实现完美不可区分，P_MD≈1−P_FA；多路径时，最小P_MD显著降低，且随着M增大，DET曲线趋向左上角，表明更高天线数可提升识别性能。

**⚠️ 局限性**

局限性包括：①假设攻击者拥有完整信道知识，实际情况可能更弱；②未考虑RIS的实际硬件非理想（如相位量化、损耗）和多用户干扰；③仿真仅基于理想几何模型，未验证在真实测量环境中的表现。

---

## 651. Grokking Is Conditional and Fragile: A Fully-Tractable, Multi-Seed Study at 12K Parameters

**arXiv ID:** 2607.05104 | [PDF](https://arxiv.org/pdf/2607.05104v1)

**作者:** Yoshiyuki Ootani `[一作]` `[通讯]` (Independent Researcher), Yoshiyuki Ootani (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在可枚举的11,856参数Llama风格Transformer（Glimmer‑1‑Base）上，对模数算术任务进行大规模、多种随机种子实验，严格固定数值环境（线程数、设备、浮点精度），测量grokking的出现率并进行全面的机制与方法学分析。

**💡 创新点**

创新点包括：①揭示grokking是受覆盖率、正则化和数值顺序控制的脆弱相变；②用多种种子率而非单跑结果验证理论（如Omnigrok逆U形、数值刀锋）并驳斥单跑叙事；③发现通用解在输出映射上更周期化，而嵌入不形成理论上的Fourier圆；④证明任务分解主要提升数据覆盖率而非监督密度。

**🔧 技术方法**

技术手段：Glimmer‑1‑Base Transformer、FineWeb‑Edu预训练、模数算术数据集、批量多种种子实验、固定CPU线程数与GPU对比、权重衰减扫描、覆盖率阈值扫描、离散傅里叶分析输出周期性、统计检验（McNemar、Newcombe、Fisher‑z、Spearman）。

**📊 数据集**

使用的数据集：基于十进制数字的单符号答案的模数算术任务（如(a+b)M、(a·b)M、((a+b)·(c+d))M等），训练集覆盖率可从1%到100%，预训练仅用500k FineWeb‑Edu token。

**📈 对比分析**

比较方法：通过多种种子测得grok-rate、平均最佳准确率，并与理论预期（如覆盖率阈值随模数增大而升高、权重衰减的逆U形）对照。实验表明：覆盖率阈值遵循输出基数规律；权重衰减可实现Omnigrok的逆U形；数值顺序扰动可导致同一随机种子结果翻转，但整体grok率无显著变化；分解策略在覆盖率有限的情况下显著提升grok率。

**⚠️ 局限性**

局限性：仅在极小模型（11k参数）与极简算术任务（M≤10）上验证；数值环境特定（CPU BLAS、单GPU、TF32禁用）可能不具普适性；阈值选择对grok判定有影响；机制分析基于周期性相关性，缺乏完整因果证据；未验证这些现象是否能推广到更大、更复杂模型。

---

## 652. Multi Choice Min Prophet

**arXiv ID:** 2607.05085 | [PDF](https://arxiv.org/pdf/2607.05085v1)

**作者:** Yossi Azar `[一作]` (Tel Aviv University), Amos Fiat `[通讯]` (Tel Aviv University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了多选最小化先知不等式（min prophet inequality），在允许一次选取多项并以所选最小值计费的多选以及多单元版本下，分析在不同到达顺序（对抗性、随机顺序、i.i.d.）下实现常数竞争比所需的期望选择次数与确定性选择次数的下界与上界。

**💡 创新点**

创新点在于首次给出对抗性顺序下期望选择次数几乎线性的下界（Ω(n/ln n)），以及随机顺序下仅需 O(min{ln ln M, ln n}) 期望选择即可达到常数竞争比，并证明该上界近似最优；同时证明在确定性选择次数限制下常数竞争比无法实现，需至少 n 次选择；并将方法推广到多单元模型。

**🔧 技术方法**

采用阈值算法（threshold algorithm）与概率分析、极限与构造反例、双对数层次划分等技术，结合对随机排列的新分析以及对期望最小值与单个变量期望之比 M 的刻画。

**📊 数据集**

本研究为理论分析，不使用实测数据集，所有结果基于随机变量分布的理论推导与构造实例。

**📈 对比分析**

与先前的单选/无约束最小化先知不等式相比，本文证明在随机顺序下可实现常数竞争比且期望选择次数大幅降低；在对抗性顺序下则揭示了选择次数几乎必须线性的事实。

**⚠️ 局限性**

局限性在于对大 M（即单变量期望远大于全局最小值期望）的情况需额外考虑；确定性选择次数限制下仍无法达到常数竞争比；多单元模型的上界与下界之间仍有对数层次的细微差距。

---

## 653. Choosing a parallel heterogeneous ensemble method for tabular classification

**arXiv ID:** 2607.05103 | [PDF](https://arxiv.org/pdf/2607.05103v1)

**作者:** Vassili Maillet `[一作]` (Mines Paris, PSL University), Pierre Jouvelot `[通讯]` (Mines Paris, PSL University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在56个中等规模表格分类任务上，对9种并行异构集成方法进行实验评估，并提出最佳实践推荐。

**💡 创新点**

创新点在于系统性比较多种并行集成（Voting、Stacking、Blending、动态选择等）与异构基学习器，并给出针对数据规模和多类任务的决策表。

**🔧 技术方法**

采用的技术包括Soft/Hard/Robust Soft Voting、Stacking/Blending、Bagging、Cluster DCS/MoE等集成策略，以及基于多种基学习器（树、SVM、MLP、XGBoost等）的基学习池。

**📊 数据集**

实验数据集来自OpenML CC18（56个任务，去除大于500k样本的）以及TabArena预计算的28个任务。

**📈 对比分析**

通过交叉验证的ROC AUC、MCC、Accuracy以及Wilcoxon符号秩检验进行比较，结果显示推荐方案优于单一最佳基学习器，且匹配或超过个别集成方法。

**⚠️ 局限性**

局限性包括动态选择方法表现不佳、Stacking在多类或高度相关基学习器场景下的不一致性，以及实验仅覆盖中等规模数据，缺乏对大规模数据的进一步验证。

---

## 654. Functional Bilevel Optimization for Predictive Fairness

**arXiv ID:** 2607.05098 | [PDF](https://arxiv.org/pdf/2607.05098v1)

**作者:** Ieva Petrulionyte `[一作]` (University of Grenoble Alpes), Michael Arbel `[通讯]` (University of Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了针对连续高维敏感属性的条件均值人口平衡（DPVar）公平度量，并设计了功能双层优化算法FBO和ITD进行训练。

**💡 创新点**

将DPVar视为功能双层优化问题，推导出平方损失下的闭式伴随并得到无Hessian的超梯度，首次实现可直接优化连续高维敏感属性的均值公平度量。

**🔧 技术方法**

使用功能式双层优化、伴随敏感法、无偏交叉验证、梯度展开（ITD）、平方损失闭式伴随、神经网络预测器和MLP估计条件均值等技术。

**📊 数据集**

使用60个真实表格回归数据集的半合成基准，以及人工生成的交互驱动不公平的合成数据。

**📈 对比分析**

与HSIC、对抗去偏、线性相关性、通用人口平衡（GDP）等基准在公平-准确性曲线上比较，FBO/ITD在DPVar约束下实现最低或接近最低的准确率退化，优于对抗、HSIC和线性方法。

**⚠️ 局限性**

DPVar仅捕捉均值差异，无法保证分布、机会或个体公平；高维A的条件均值估计统计难度大；半合成基准不反映真实敏感属性的法律/社会含义。

---

## 655. Semantic Video Communication via Multi-Scale Convolution and Dynamic Routing for Next-Generation Networks

**arXiv ID:** 2607.05093 | [PDF](https://arxiv.org/pdf/2607.05093v1)

**作者:** Gengtian Shi `[一作]` (Waseda University), Jiang Liu `[通讯]` (Waseda University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于多尺度卷积与胶囊动态路由的生成式 AI 框架，用于在下一代网络环境下实现语义视频通信，主要通过精准的时序语义对齐实现带宽高效传输。

**💡 创新点**

创新点在于①提出了 O(T) 复杂度的多尺度时间卷积编码器，能够高效捕获不同粒度的运动特征；②利用胶囊网络的动态路由实现多对多的语义对齐，解决传统注意力稀疏且多义的对齐瓶颈；③将时序边界回归、跨模态对齐与胶囊多样性三项任务统一在多任务学习框架中，提升整体性能与可解释性。

**🔧 技术方法**

使用技术包括 CLIP 预训练的视觉与文本编码器、1D 卷积多尺度编码器、胶囊网络动态路由、信息对比（InfoNCE）对齐损失、软监督的边界预测、层归一化与残差结构等。

**📊 数据集**

实验基准为 ActivityNet Captions（约 20K 视频、100K+ 句子描述），在 1 fps 视频采样、224×224 分辨率下进行评估。

**📈 对比分析**

与传统 transformer 基线及其他基准方法进行对比，结果显示多尺度+胶囊组合在 Recall@0.5 方面达到 42.9%，mIoU 为 41.1%，相较于单一基线提升约 17% 的 mIoU，并保持 O(T) 的计算效率，适合边缘部署。

**⚠️ 局限性**

局限性包括：仅在 ActivityNet 上验证，缺乏跨数据集/零样本泛化实验；对视觉自适应能力有限；文本编码器性能决定性强，需进一步提升鲁棒性；以及在极端低算力环境下仍可能存在算力瓶颈。

---

## 656. KVpop -- Key-Value Cache Compression with Predictive Online Pruning

**arXiv ID:** 2607.05061 | [PDF](https://arxiv.org/pdf/2607.05061v1)

**作者:** Lukas Hauzenberger `[一作]` (NXAI), Sepp Hochreiter `[通讯]` (NXAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种固定预算的KV缓存压缩方法，通过可预测的在线裁剪实现高效长上下文推理。

**💡 创新点**

创新点在于在缓存淘汰边界上使用未来注意力权重作为监督信号，训练轻量级评分器；并支持状态化的延迟评分，让评分器在得到近未来上下文后再做决策。

**🔧 技术方法**

采用轻量级Head‑wise MLP或Stateful Memory Scorer进行重要性打分，利用转置注意力计算未来注意力目标，使用Fenwick树实现在线Top‑k选取，并在FlexAttention稀疏核中构造稀疏注意力模式；训练过程中结合蒸馏与KL损失。

**📊 数据集**

训练数据选自Qwen预训练模型，使用Nemotron‑Math v2数学推理数据；评估数据集包括AIME、HMMT、GPQA‑Diamond和LiveCodeBench v6。

**📈 对比分析**

与StreamingLLM、TOVA、DMS等基线相比，模型在75%和88% KV压缩下在AIME/HMMT上保持约95%密集注意力性能，且在压缩率高时仍优于DMS和传统启发式淘汰策略。

**⚠️ 局限性**

局限性包括：仅为后置改造，未从零训练；未探索更丰富的状态化评分器方案；缺乏对混合稠密-稀疏层或分页KV缓存管理的进一步实验。

---

## 657. Quantum-Inspired Harmonic Decision Models: A Computational Framework for Music Generation

**arXiv ID:** 2607.05007 | [PDF](https://arxiv.org/pdf/2607.05007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 658. When AI Is Wrong on Purpose: How Students Respond to Buggy GenAI Code

**arXiv ID:** 2607.05068 | [PDF](https://arxiv.org/pdf/2607.05068v1)

**作者:** Victor-Alexandru Pădurean `[一作]` (Max Planck Institute for Software Systems), Adish Singla `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在生成式AI编程任务中，作者通过在学生接收的代码中有意识地注入可运行的错误（Injected Bugs），与自然产生的错误（Natural Bugs）一起，分析学生在面对不同来源的bug时的行为和修复策略。

**💡 创新点**

创新点在于将两类bug（自然误差与人为注入的近似成功代码）同时嵌入同一工作流，探讨bug来源对学生提示精炼与代码修复决策的影响，并系统量化两类bug对修复效率与学习体验的差异。

**🔧 技术方法**

使用了 Prompt Programming 平台、GPT‑4o‑mini 作为生成模型、bug 注入管道（在生成代码通过测试后再用 LLM 生成近似错误版本），以及基于日志的行为跟踪与反思问卷。

**📊 数据集**

数据集来自 2025 年奥克兰大学 CS1 课程的 917 名学生，覆盖 5 个 C 语言任务，收集了 2,636 次完整会话（共 6,071 次需修复回合）以及学生的后置反思问卷。

**📈 对比分析**

通过对学生在第一次遇到错误时的首选动作（编辑 vs. 重新提示）、即时成功率、编辑距离、编辑次数、提示长度等指标进行统计和对比。结果显示：Injected Bugs 更倾向于被直接编辑且即时成功率显著高于 Natural Bugs；Natural Bugs 则更常促使学生重新提示并优化规范。相对性能表明，近似错误更能促使学生主动进行代码审查与局部修复。

**⚠️ 局限性**

局限性：实验仅在单一 C 语言课程与五个任务中进行，缺乏跨语言、跨学科或对照组比较；未评估长期学习成效；未探究错误注入对学生信任与依赖的长期影响；平台的提示与测试反馈设计可能影响行为，难以泛化到其他工具。

---

## 659. Uncertainty-aware damage identification in short-span bridges via physics-informed variational autoencoder

**arXiv ID:** 2607.05025 | [PDF](https://arxiv.org/pdf/2607.05025v1)

**作者:** Ana Fernandez Navamuel `[一作]` (Basque Center for Applied Mathematics), David Pardo `[通讯]` (University of Basque Country)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个基于物理约束的概率科学机器学习框架——PI‑GCVAE，用于从桥梁振动模态数据中识别早期损伤并量化不确定性。

**💡 创新点**

①在VAE解码器中嵌入可微分的特征值求解器，直接满足结构动力学方程，减少可训练参数并消除拟合误差；②采用高斯Copula建模潜在变量的联合分布，捕捉相邻结构单元间的空间相关性；③与传统GMM或流模型相比，Copula在高维下保持参数量可控，实现高效的后验估计。

**🔧 技术方法**

变分自编码器（VAE）+高斯Copula潜在分布 + 可微分有限元特征值求解器 + 自动微分 + TensorFlow训练框架 + ELBO 损失（频率、模态协方差、后验正则化）。

**📊 数据集**

由5个单元的简支桥梁有限元模型生成的10,000个合成损伤场景，提取前5个模态频率和形状，并加入2.5%频率噪声、5%形状噪声的高斯扰动。

**📈 对比分析**

与仅使用数据驱动解码器的GCVAE‑NN进行对比；评估指标包括MSE、MAE、95% 置信区间覆盖率、MACE、平均对数似然（ALL）。PI‑GCVAE在覆盖率和ALL上表现更好，虽略高的MSE，但提供了更可靠的 uncertainty 估计；在10维扩展实验中仍保持约90% 的覆盖率。

**⚠️ 局限性**

高斯边缘分布假设无法捕捉多峰后验；对极端多模态损伤场景的逼近有限；缺乏对环境/操作变异的显式建模；未来工作计划引入更富表达力的分布（如Beta混合、流模型）并验证真实桥梁数据。

---

## 660. Beyond Modality Fusion: Deep Ensembles for Multimodal Classification

**arXiv ID:** 2607.05019 | [PDF](https://arxiv.org/pdf/2607.05019v1)

**作者:** Ilya Burenko `[一作]` (Technische Universität Dresden), Dmitry Vetrov `[通讯]` (Constructor University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统研究了深度集成（HeDE）在多模态分类中的效果，证明在模态不平衡情形下，无需显式融合即可获得比现有后期融合、间接融合和混合方法更好的性能，并提出了基于单模态验证损失的启发式分配规则、可扩展的合成多模态数据集和缩放律预测。

**💡 创新点**

创新点在于：①证明深度集成在多模态不平衡时优于所有已知融合方案；②提出一种无融合、可直接优化的集成规模与模态比例的启发式选择；③构造可调节模态预测强度的合成多模态数据集；④给出多模态集成的缩放律，能从单模态性能预测集成极限性能。

**🔧 技术方法**

使用技术包括：独立训练的多模态单模网络，logit平均的深度集成，基于验证损失比例的启发式分配，类对齐的合成多模态数据生成，后期融合（late‑fusion）与间接融合（MMTM、跨注意力）架构，混合（I2M2）方法，以及对等参数量的系统对比实验和功率律拟合。

**📊 数据集**

实验数据集包括：合成的CIFAR‑100和ImageNet‑1K多模态版本；真实多模态数据集CREMA‑D（音频‑视觉）、Sarcasm（文本‑视觉）和Kinetics‑400（音频‑视觉）等。

**📈 对比分析**

比较方法是在等参数量、相同特征提取器的条件下，将HeDE与各类后期融合（InfoReg、MMPareto、OGM‑GE等）、间接融合（MMTM、跨注意力）和混合（I2M2）方法做准确率对比；结果显示HeDE在所有模态不平衡度上均优于基准，尤其在极度不平衡时小规模更适合HoDE，随着规模增大切换为HeDE，提升幅度约5–15%，且启发式分配与穷举搜索几乎等效。

**⚠️ 局限性**

局限性包括：合成数据仅实现类对齐，未覆盖实例对齐的真实交互；无法独立控制模态收敛速度与预测强度；启发式分配缺乏严格理论证明；对三模及以上的评估有限，未来需验证更大规模多模态和更复杂融合策略。

---

## 661. A Modular O-RAN Testbed Based on SRS Open Source O-CU/O-DU and Massive Beams Modular O-RU

**arXiv ID:** 2607.05146 | [PDF](https://arxiv.org/pdf/2607.05146v1)

**作者:** Fabian Göttsch `[一作]` (Massive Beams), Giuseppe Caire `[通讯]` (Technical University Berlin)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

构建并验证了基于SRS OCUDU与Massive Beams MODRAD-SC的模块化O-RAN测试平台，可支持从分割7.2a到8的无缝切换、数字与模拟波束成形及预6G实验；

**💡 创新点**

将开源O-CU/O-DU与可切换O-RU/SDR硬件结合，首次实现O-RAN合规的FR3混合数字-模拟波束成形，并通过NFED‑RIS实现低延迟、高精度的波束指令下发；

**🔧 技术方法**

使用SRS OCUDU软件栈、MODRAD‑SC硬件、eCPRI/O-RAN接口、SDR模式、NFED‑RIS、AI驱动的RIC以及4×RF链硬件架构；

**📊 数据集**

未使用公开数据集，实验数据由自建硬件与现场O2O测量生成；

**📈 对比分析**

通过现场O2O传输实验验证FR3波束成形，误差向量幅值低，表明系统在低延迟、低功耗条件下可实现高质量波束指令；

**⚠️ 局限性**

局限于当前硬件仅支持4个RF链，FR3实验仍处于原型阶段，缺乏大规模多用户/大规模MIMO以及真实运营网络验证。

---

## 662. PDEFlow: Autonomous Agentic PDE Pipelines for Neural Operator Learning and Solver-Free Inference

**arXiv ID:** 2607.05134 | [PDF](https://arxiv.org/pdf/2607.05134v1)

**作者:** Akshat Jani `[一作]` (Tata Research Development and Design Center), Venkataramana Runkana `[通讯]` (Tata Research Development and Design Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PDEFlow，一个端到端的自主代理框架，能把自然语言描述的 ODE/PDE 规范转化为可执行的 solver 后端数据集、训练神经算子，并实现无 solver 推理。

**💡 创新点**

创新点在于（1）状态化的输入图把多轮自然语言编辑映射为可验证的 JSON 补丁；（2）统一的注册接口让不同神经算子可插拔；（3）完整的从规范、数据生成、训练到推理的闭环自动化；（4）在实验中验证了该框架在多轮编辑下保持规范一致性的准确性。

**🔧 技术方法**

技术栈包括 gpt-o4-mini LLM、LangGraph/AutoGen 风格的代理图、FEniCSx 有限元求解器、Bayesian DeepONet（多分支）算子、JSON 补丁验证/修复、TensorFlow/PyTorch 的训练与推理管线。

**📊 数据集**

使用自定义的 70 条脚本化 ODE/PDE 场景作为验证集；数据集通过对规范采样参数、求解并转化为张量形式，涵盖 1D/2D 线性稳态与瞬态问题。

**📈 对比分析**

与单次生成规范的 baseline、无验证器、无修复器三种 ablation 方案对比，系统在规范生成上的整体准确率为 81.43%，验证器缺失降至 54.29%，修复器缺失降至 65.71%；在算子评估上，训练出的神经算子在大多数基准任务上与 FEniCSx 求解结果误差可接受，并实现显著的推理加速。

**⚠️ 局限性**

局限性包括：仅支持二维矩形网格、线性 PDE；算子选择仍为手动；缺乏对非线性、复杂几何和大规模高维问题的支持；在极端瞬态或强非线性场景下误差仍较大。

---

## 663. When Agents Lie: Premeditation, Persistence, and Exploitation in Repeated Games

**arXiv ID:** 2607.05132 | [PDF](https://arxiv.org/pdf/2607.05132v1)

**作者:** Jerick Shi `[一作]` (Carnegie Mellon University), Zhijing Jin `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型在五人重复博弈中的三阶段承诺机制进行评估，系统测量其在公开承诺与最终行动之间的欺骗行为。

**💡 创新点**

首次设计内生三阶段协议并在同质与异质模型组合下量化预谋欺骗与信息解读不匹配导致的剥削效应，揭示模型间公告语义差异会产生持续的支付不平衡。

**🔧 技术方法**

采用私密计划、公共声明、最终行动及信任反思四个步骤的协议，利用GPT‑5.2、Llama‑4‑Maverick、Claude‑Opus‑4.6三大前沿模型的API完成实验。

**📊 数据集**

六个经典博弈（Diners, El Farol, Volunteer, Tragedy of Commons, Public Goods, Weakest Link）作为数据集，五人组、10轮共计约126,000回合数据。

**📈 对比分析**

对比同质组与异质组的欺骗率、预谋率、公告合规度及支付差距，发现预谋欺骗率高达90%+，但在异质组中支付差距在Round 0即出现并在10轮内持续；不同博弈表现差异显著，说明模型与游戏结构相互作用决定行为。

**⚠️ 局限性**

局限在于仅评估三种模型、10轮、5人组；预谋判定依赖模型自报私密计划，可能与真实内部意图不一致；未探讨更长时序、更大模型组合或对策干预。

---

## 664. From Failing to Passing: Evolving Natural Language Prompt Optimization Rules for LLM Code Generation

**arXiv ID:** 2607.05121 | [PDF](https://arxiv.org/pdf/2607.05121v1)

**作者:** Amal Akli `[一作]` (University of Luxembourg), Yves Le Traon `[通讯]` (University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DualFix框架，将错误反馈修复与演化得到的自然语言变换规则相结合，对LLM生成的代码进行两阶段修复；

**💡 创新点**

创新点在于使用基于进化搜索的RuleEvol自动发现与模型无关的、可复用的提示词变换规则，且规则能够跨模型零射击迁移；

**🔧 技术方法**

采用基因搜索（遗传算法）演化规则、LLM重写器（GPT‑4o‑mini）执行规则、LLM修正器（GPT‑5‑mini）进行规则变更、执行反馈重写与代码生成（Qwen‑2.5‑Coder‑7B、Codestral‑22B、Claude Haiku 4.5）；

**📊 数据集**

在LiveCodeBench（功能+stdin/stdout）和APPS（面试级别）两个基准上进行评估；

**📈 对比分析**

与直接生成、Self‑Fix、单轮/多轮错误反馈修复等基线对比，DualFix在两组基准上实现最高30%失败率恢复，修复率比Self‑Fix高3–5倍；

**⚠️ 局限性**

局限包括仅在Python代码与两类竞赛/面试题上验证，规则不包含新信息但仍依赖重写器，未覆盖极端模型规模、非Python语言或开放式软件工程任务。

---

## 665. EvoAgentBench: Benchmarking Agent Self-Evolution via Ability Transfer

**arXiv ID:** 2607.05202 | [PDF](https://arxiv.org/pdf/2607.05202v1)

**作者:** Xingze Gao `[一作]` (Anhui University), Teng Li `[通讯]` (Anhui University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 EvoAgentBench，一种基于轨迹抽取的可复用能力图（Ability Graph）评估框架，用于细粒度诊断 LLM 代理的自我演化能力。

**💡 创新点**

创新点在于提出基于代理执行轨迹的可复用“Ability”抽取与规范化方法，并通过保证测试任务具备训练时的能力支持，实现跨任务、跨域的迁移诊断；同时提供诊断参考 Anchor Skill 与自动方法的对比，揭示迁移瓶颈。

**🔧 技术方法**

采用轨迹抽取、LLM 与专家评审的能力卡原始提取与规范化、嵌入阻塞与 LLM 仲裁、社区检测构建能力图、以及多模型、多结构的实验设置（Qwen、Gemma、OpenClaw、Nanobot）进行评估。

**📊 数据集**

利用四大长期任务域的公开 benchmark（BrowseComp-Plus、LiveCodeBench、SWE-Bench Verified、GDPVal）共 2605 任务，生成轨迹池并抽取 170 个统一的能力单元。

**📈 对比分析**

与三种自动自演化方法（Memento、ReasoningBank、GEPA）以及诊断 Anchor Skill 进行对比，Anchor 在所有设置中均实现正向迁移，自动方法在部分设置表现负迁移，平均增益不稳定，表明方法在经验编码、路由或使用上仍存在显著瓶颈。

**⚠️ 局限性**

局限性包括仅覆盖文本代理与公开模型，缺少多模态或更大模型的验证；能力抽取依赖单一 LLM，后续研究需评估不同抽取器对词典的影响；评估任务采样偏向支持任务，增益估计不具普遍性。

---

## 666. SMART: A Machine Learning and Monte Carlo Framework for Rapid Analysis of Stochastic Transistor Aging and Process Variation in Digital Circuits

**arXiv ID:** 2607.05187 | [PDF](https://arxiv.org/pdf/2607.05187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 667. Noisy-Channel Minimum Bayes Risk Decoding

**arXiv ID:** 2607.05198 | [PDF](https://arxiv.org/pdf/2607.05198v1)

**作者:** Yusuke Sakai `[一作]` (Nara Institute of Science and Technology), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于噪声通道的 Minimum Bayes Risk（MBR）解码分解，将 MBR 评分拆分为假设对参考的似然、参考对假设的似然、假设先验和参考先验，并通过对这些通道的加权来提升解码质量。

**💡 创新点**

创新点在于：①首次给出 MBR 的四维概率分解，明确表达了评价指标的双向不对称性；②引入可调节的通道权重，提供了细粒度可解释性；③通过实验证明这些通道对不同指标的影响是任务无关、指标相关的，从而为 MBR 进一步改进提供理论依据。

**🔧 技术方法**

使用的技术包括：噪声通道概率分解、基于采样的 Monte Carlo 伪参考生成、对四个通道权重的网格搜索、利用 BLEU、chrF、COMET、BERTScore 等评价函数作为 Utility 函数进行实验评估。

**📊 数据集**

所用数据集涵盖：WMT 2022/23 机器翻译（En↔De、Ja↔En、Zh↔En）、CNN/DailyMail、XSum、SAMSum 摘要任务以及 MSCOCO、NoCaps 图像字幕任务。

**📈 对比分析**

与传统 MAP 以及现有 MBR 变体在相同任务/指标下对比；通过在不同通道权重下评估性能，发现重新加权后可在 ChrF 上从 50.05 提升至 50.18，且在多任务、多指标下表现出一致性提升，表明通道加权能够显著提升解码效果。

**⚠️ 局限性**

局限性包括：需要为每个评价指标手动调节通道权重，调参成本高；实验仅覆盖有限的任务与指标，尚未验证在更广泛场景下的普适性；依赖 Monte Carlo 采样，计算成本相对较高；权重选择仍基于经验，缺乏严格的理论最优证明。

---

## 668. Is Three the Magic Number? An Empirical Evaluation of LLM-Based Repair Loops

**arXiv ID:** 2607.05197 | [PDF](https://arxiv.org/pdf/2607.05197v1)

**作者:** Tobias Kiecker `[一作]` (Humboldt-Universität zu Berlin), Lars Grunske `[通讯]` (Humboldt-Universität zu Berlin)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对多种基于LLM的SE工具进行实证评估，系统测量并分析其在不同修复循环迭代次数下的完成率，探讨修复预算的效果与成本平衡。

**💡 创新点**

创新点在于：①首次统一比较多工具、多任务、多模型的修复收益曲线；②证明前3-4次迭代已获得大部分提升；③指出工具的调度逻辑、验证策略和反馈设计比所选模型更能决定修复效果，揭示重现性与可比性问题。

**🔧 技术方法**

技术手段包括：LLM生成‑验证‑修复循环，使用Gemma‑4、Qwen3.5、GPT‑4o‑mini三种模型；对INTERVENOR、LDB、CASCADE、OCI等工具进行本地化改造；验证机制涵盖编译、静态分析和单元测试。

**📊 数据集**

使用的数据集包括Defects4J、MBPP、HumanEval、CASCADE、TransCoder等，涵盖代码生成、测试生成与代码翻译三大任务。

**📈 对比分析**

比较方法：对每一步修复后计算完成率并绘制曲线，进一步计算相邻步骤的相对提升。结果显示收益呈凹形，前三到四步贡献最大，后续步骤收益递减至几乎为零。

**⚠️ 局限性**

局限性：①实验仅采用成本效益低的模型，未验证更强大模型的情况；②缺乏多次重复实验导致结果可能因LLM随机性而波动；③工具改造可能影响可迁移性，未涵盖所有任务与验证维度；④仅基于编译与单元测试的自动验证，可能未覆盖语义或质量层面的错误。

---

## 669. Towards the Recognition of Oriented Interval Graphs

**arXiv ID:** 2607.05191 | [PDF](https://arxiv.org/pdf/2607.05191v1)

**作者:** Lukas P. Bachmann `[一作]` (Universität Passau), Alexander Wolff `[通讯]` (Universität Würzburg)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了有向区间图（Oriented Interval Graphs）的识别问题，给出了在已知包含边或已知区间方向时的线性时间算法，并为一般识别问题提供了结构化分析框架。

**💡 创新点**

创新点在于提出了三要素（方向 φ、团序列 σ、包含边集）之间的相互约束关系，并基于此给出了对已知方向或已知包含边集的两种约束识别问题的线性时间解法；此外，利用增广图、MPQ‑树与匹配表示等工具，将判定问题转化为匹配表示与受约束的可达性问题。

**🔧 技术方法**

主要技术包括：可扩展区间图的组合表示、可逆方向的“翻转”技术、对匹配图的变形与模组分解、MPQ‑树的约束化与旋转、以及对转置图的可达性与拓扑排序的线性时间算法。

**📊 数据集**

本文为理论算法研究，未使用公开数据集；实验验证与基准对比未给出。

**📈 对比分析**

由于缺乏实验数据，本文未进行方法对比；理论上所提出的算法复杂度为 O(|V|+|E|)，显著低于之前的 O(|V|^2) 方法，适用于大规模混合图。

**⚠️ 局限性**

局限性：仅针对已知包含边集或已知方向的约束情况给出线性算法；一般识别问题仍未得到多项式时间解法；此外，MPQ‑树与匹配表示的构造在实际实现中可能存在复杂度与实现细节的挑战。

---

## 670. VLM-CASE: Vision-Language Model Enabled Context-Adaptive Safety Envelopes for Anticipatory Safe Autonomous Driving

**arXiv ID:** 2607.05180 | [PDF](https://arxiv.org/pdf/2607.05180v1)

**作者:** Tianjia Yang `[一作]` (Pennsylvania State University), Xianbiao Hu `[通讯]` (Pennsylvania State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VLM-CASE 框架，利用前视摄像头的视觉‑语言模型（VLM）实时推断路面摩擦系数与前方可视度，将推断结果映射为上下文适应的安全包络（CASE），并在此安全包络内使用模型预测控制（MPC）实现自驾安全驾驶；

**💡 创新点**

创新点在于将场景理解直接驱动安全限额而非仅作为控制参数，构建共享摩擦预算、同时考虑制动与转向的安全包络，并通过异步 VLM 推理保持实时控制与安全保证的兼容；

**🔧 技术方法**

使用技术包括：Qwen3‑VL 视觉‑语言模型 + LoRA 低秩适配进行任务微调；基于 Responsibility‑Sensitive Safety (RSS) 与摩擦约束的安全包络；模型预测控制（MPC）与约束优化；CARLA 仿真环境；

**📊 数据集**

数据集为在 CARLA 生成的 10,560 张前视摄像头图像，覆盖不同路面（干、湿、雪）、天气（晴、雨、雾）、昼夜、灯光辅助等四类情境，并手工标注对应标签，划分为训练、验证、测试集；

**📈 对比分析**

与 Base MPC、VLM‑MPC、Fixed‑Envelope MPC 等基线在 198 条闭环仿真路径中对比；VLM‑CASE‑MPC 在所有 54 组合任务中 100% 成功率，显著高于基线（如 Base 仅 52%，Fixed‑Envelope 81%），同时在单因素实验中保持高成功率并大幅降低碰撞时间与偏离风险；

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，真实车辆需要进一步实验和摩擦校准；场景映射为离散类别，缺乏连续细粒度适配；安全包络与 MPC 结合是示例，需验证与其他控制器或运行时监控器的兼容性；以及对 VLM 推理时延的实际影响仍待实测。

---

## 671. Approximation Algorithms for the Traveling Thief Problem

**arXiv ID:** 2607.05164 | [PDF](https://arxiv.org/pdf/2607.05164v1)

**作者:** Jan Eube `[一作]` (University of Bonn), Sarah Sturm `[通讯]` (University of Bonn)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了在旅行小偷问题（Traveling Thief Problem）与权重约束旅行商问题（Weighted TSP）中，基于多目标近似的多项式时间算法，构造了约 (9+ε,9+ε) 的双重逼近 Pareto 集合以及 (2e+ε) 的 Weighted TSP 近似方案。

**💡 创新点**

创新点在于首次为同时考虑行程时间与利润的约束旅行小偷问题给出多目标逼近算法，并将传统的 Quota‑TSP、容量或ienteering 与多周期二项背包相结合，突破了此前仅有单目标或启发式求解的局限。

**🔧 技术方法**

主要技术包括：分层时间-重量区间划分、基于 Doubling 与 Quota‑TSP 的子路段构造、3+ε 近似容量或ienteering、(1+ε) 多周期二项背包近似，以及通过成本下界与利润比例的双重阈值实现 Pareto 逼近。

**📊 数据集**

论文未使用具体数据集，聚焦于理论算法设计与证明；若要实验验证，可采用公开的 TTP/Knapsack 竞赛数据。

**📈 对比分析**

由于缺乏实验结果，作者以理论证明为主，展示了在多项式时间内实现的常数逼近因子；与已有的单目标近似或启发式方法相比，该方法在两目标之间提供了可控的平衡。

**⚠️ 局限性**

局限性包括：1) 对整数权重做了多项式界限假设，非整数或大规模权重可能导致复杂度上升；2) 近似因子仍较大（9+ε），在实际应用中精度有限；3) 仅提供理论证明，缺乏实验评估与对比。

---

## 672. ASSEMCAD: Production-Ready CAD Assembly Generation from Natural Language

**arXiv ID:** 2607.05123 | [PDF](https://arxiv.org/pdf/2607.05123v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 673. Relational Multi-Agent Reinforcement Learning for Dynamic Pricing in High-Speed Railway Markets

**arXiv ID:** 2607.05179 | [PDF](https://arxiv.org/pdf/2607.05179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 674. Three-Phase Evaluation of AI-Assisted Software Development Life Cycle

**arXiv ID:** 2607.05125 | [PDF](https://arxiv.org/pdf/2607.05125v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 675. EdgeBench: Unveiling Scaling Laws of Learning from Real-World Environments

**arXiv ID:** 2607.05155 | [PDF](https://arxiv.org/pdf/2607.05155v1)

**作者:** Deyao Zhu `[一作]`, Guang Shi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Frontier Benchmark，包含 134 个跨六大类真实可执行任务，并在约 38,000 小时的交互中评估代理学习曲线，发现环境学习呈现 log‑sigmoid 缩放规律。

**💡 创新点**

创新点在于：① 设计 ultra‑long‑horizon、双循环（本地+判定）反馈的任务体系；② 系统验证环境学习的普适 log‑sigmoid 规律；③ 提出 frontier 前沿图扩展的理论解释；④ 观察到 frontier 模型的学习速度每三个月翻倍。

**🔧 技术方法**

技术手段包括：在可执行工作区与判定容器中运行 frontier LLM（Claude Opus 4.8、GPT‑5.5 等）进行 12 h+ 的长跑；记录每一次提交轨迹；对性能曲线使用三参数 log‑sigmoid 拟合；利用理论推导的 frontier‑cut 过程解释曲线。

**📊 数据集**

使用的数据集为 Frontier Benchmark：134 个真实世界任务，涵盖科学研究、软件工程、组合优化、专业知识工作、定理证明与交互游戏等六大领域，共计约 38,000 小时交互。

**📈 对比分析**

比较方法：对 5 款 frontier 模型进行 12 h 最终分数对比，Claude Opus 4.8 领跑 51.3 分，GPT‑5.5 48.4；通过对比持续跑 vs 重复抽样，持续跑提升 6.9 分；长上下文 200k vs 1M 提升约 5 分；log‑sigmoid 预测未来 12 h 性能，R² > 0.997，RMSE < 1.0。

**⚠️ 局限性**

局限性：仅适用于具备可执行环境和长期交互的任务；对图结构存在强瓶颈或分散 mid 的任务，log‑sigmoid 可能失效；未覆盖视觉、语音等多模态交互；实验基于有限 frontier 模型，跨模型泛化仍需验证。

---

## 676. Intent-Based Mutation Testing: From Naturally Written Programming Intents to Mutants

**arXiv ID:** 2607.05149 | [PDF](https://arxiv.org/pdf/2607.05149v1)

**作者:** Asma Hamidi `[一作]` (University of Luxembourg), Mike Papadakis `[通讯]` (University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对程序的自然语言意图进行变异，利用LLM生成相应代码实现，从而得到一批新的程序变体用于变异测试。

**💡 创新点**

首次将意图（功能说明）视为变异对象，引入意图变异而非传统语法变异，利用LLM生成语义上不同且更复杂的变体，能发现传统方法无法捕获的缺陷。

**🔧 技术方法**

使用 BERT 的掩码语言模型对意图进行词替换，再利用 GPT‑3.5‑turbo 生成代码实现；同时与传统基于语法的变异器进行对照。

**📊 数据集**

采用 HumanEval+（Java 子集）中的 29 个问题，包含自然语言描述、原始实现和完整的测试用例。

**📈 对比分析**

通过语法相似度（BLEU、余弦、Jaccard）、语义相似度（测试失败比例）以及子涵盖率和子杀伤分数进行评估；结果显示意图变异生成的变体语法距离更大、杀伤率更高，能产生约 23% 更多独立缺陷，但子杀伤得分略低于传统方法，表明其多样性更佳。

**⚠️ 局限性**

依赖 LLM 的生成结果，存在不确定性；测试用例集可能不完全覆盖所有行为；实验仅在 29 个 Java 问题上进行，缺乏对其他语言或更大规模程序的验证。

---

## 677. Causal-RetiGraph: Cross-Cohort Retinal Support and Same-Subject Pathway Analysis for Diabetic Retinopathy

**arXiv ID:** 2607.05204 | [PDF](https://arxiv.org/pdf/2607.05204v1)

**作者:** Inam Ullah `[一作]` (University of Southampton), Shoaib Jameel `[通讯]` (University of Southampton)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出Causal-RetiGraph框架，将视网膜图像特征与NHANES系统性数据相结合，用图形表型X1234进行跨队列路径优先级排序，并用同一受试者R*进行路径摘要。

**💡 创新点**

创新点在于：①将多模态视网膜证据（血管、病灶、嵌入、AutoMorph标志）整合为可解释的图形表型；②通过跨队列支持优先级结合系统性暴露与视网膜表型；③区分外部图像支持和同一受试者介导变量，避免过度推断因果关系。

**🔧 技术方法**

技术包括：多分支表示（空间X_12与Jacobian X_34），两分支融合注意力，Grad-CAM弱监督病灶定位，AutoMorph血管指标提取，特征映射与Jacobian敏感度分析，线性/非线性加权模型，GAM等统计方法。

**📊 数据集**

使用APTOS 2019 Blindness Detection眼底图像数据（2,910张）和NHANES 2005–2008糖尿病子集的系统性变量（HbA1c、尿白蛋白等）。

**📈 对比分析**

与单一模态基线比较，X1234在二分类准确率0.9055、AUROC0.9711、分级QWK0.8312等指标上优于各流；路径优先级通过X1234支持和NHANES关联性得分得到，表明HbA1c、脉压等路径排名最高。

**⚠️ 局限性**

局限性：X1234未在NHANES样本上直接观测，R*可能缺失X1234所有信息；病灶证据弱，未使用专家标注；非线性模型探索性较弱；跨队列优先级不是个体层面因果估计。

---

## 678. Green for Go, Red for No: Visual Grounding via Semantic Segmentation for VLA Navigation Policies

**arXiv ID:** 2607.05122 | [PDF](https://arxiv.org/pdf/2607.05122v1)

**作者:** Adrian Szvoren `[一作]` (University College London), Nilufer Tuptuk `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 VLA 导航策略进行视觉定位评估，提出实时基于 SegFormer 的语义分割 grounding 方法。

**💡 创新点**

首次对 VLA 导航中的视觉 grounding 进行系统评估，证明绿色-红色可视化分割可显著降低最远 waypoint 的误差，并揭示其主要作用是缩短轨迹。

**🔧 技术方法**

使用 SegFormer 进行实时语义分割、融合到观察图像及目标；与 OmniVLA VLA 模型结合。

**📊 数据集**

在 Grand Tour 数据集的 ETH-2 episode（ANYmal D 四足机器人）上评估。

**📈 对比分析**

与未 grounding 的基线对比，最远 waypoint 误差下降 27–44%，但归一化误差趋于一致；图像目标 grounding 效果有限。

**⚠️ 局限性**

局限性包括仅处理 2D 可通行性、对动态障碍与 3D 结构（如楼梯）无效、停留指令（stop）仍无法正确执行，且仅在单一室内 episode 上验证。

---

## 679. Localized LoRA-MoE: Block-wise Low-Rank Experts With Adaptive Routing

**arXiv ID:** 2607.05114 | [PDF](https://arxiv.org/pdf/2607.05114v1)

**作者:** Babak Barazandeh `[一作]` (Fortinet), George Michailidis `[通讯]` (UCLA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Localized LoRA-MoE 框架，在传统 LoRA 的低秩适配器基础上加入局部块分区与上下文感知动态路由，以消除梯度冲突并提升适配效率。

**💡 创新点**

创新点在于设计两种路由粒度：Block‑Wise LoRA‑MoE（宏路由）和 Cell‑Wise LoRA‑MoE（微路由），并证明微路由在参数等价下可匹配或超越宏路由，同时通过“梯度防火墙”实现局部梯度隔离。

**🔧 技术方法**

技术手段包括低秩适配器（LoRA）、空间矩阵分块、Mixture‑of‑Experts 路由、Softmax 门控、参数高效微调、以及对路由可达集合与梯度耦合的理论分析。

**📊 数据集**

实验数据集为：1）合成高维 SVD 多域矩阵；2）加州住房（California Housing）表格回归；3）MNIST 图像重构+传感器降级场景。

**📈 对比分析**

在相同参数预算（约 2k–3k 参数）下采用 AdamW 训练，比较 MSE 与 R² 指标。结果显示：SVD 案例中 Cell‑Wise R² 38.29% 远超 Block‑Wise 26.81%；加州住房中两者均接近 99.6%；MNIST 中两者约 66.9%，均显著优于 LoRA/MELoRA/Localized LoRA 静态基线。

**⚠️ 局限性**

局限性包括：仅在合成回归和图像重构任务验证，未在 Transformer 语言模型或多任务设置中测试；专家数量固定且上下文离散；门控采用 Softmax，未探索 Top‑k 稀疏路由；缺乏对专家利用率、崩溃行为及不同路由粒度理论界限的深入分析。

---

## 680. Rating the Pitch, Not the Product: User Evaluations of LLMs Reflect Expectations More Than Performance

**arXiv ID:** 2607.05113 | [PDF](https://arxiv.org/pdf/2607.05113v1)

**作者:** Robert Morabito `[一作]` (Brock University), Ali Emami `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对六款LLM在三种不同预交互框架（Oversold、Matched、Undersold）下进行对照实验，测量用户在使用前后的印象、交互行为和输出质量。

**💡 创新点**

系统性证明预交互期望框架会持续影响用户印象与交互方式，但不改变实际产出质量；印象变化主要由期望满足度和自我效能驱动，而非任务表现。

**🔧 技术方法**

使用LLM交互实验、UTAUT 与 Godspeed 量表、行为日志自动抽取、GPT‑5 自动评分、OLS 回归与相关分析。

**📊 数据集**

样本为 162 名来自 Prolific 的美国英语使用者，完成图像生成、外联邮件撰写和缩写创作三项协作任务，记录交互日志并进行问卷评估。

**📈 对比分析**

通过对比框架和模型层级的交互行为、印象变化以及 GPT‑5 评分的任务表现，发现模型层级显著预测输出质量（β=0.44, p<0.001），而框架对质量无显著影响；框架显著改变用户的印象评分和交互行为。

**⚠️ 局限性**

受试者仅为美国英语用户，任务顺序固定，模型层级划分基于当时公开基准，未检验长期或多轮使用效果；评估者与人工评分的一致性有限。

---

## 681. Finfluencers on TikTok: A Longitudinal Analysis of Content, Engagement, and Disclaimer Practices

**arXiv ID:** 2607.05203 | [PDF](https://arxiv.org/pdf/2607.05203v1)

**作者:** Essam Ghadafi `[一作]` (Newcastle University), Panagiotis Andriotis `[通讯]` (University of Birmingham)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统地收集并纵向分析了英国 TikTok 金融影响者（finfluencer）在 2024-2026 年期间的视频内容、观众互动、免责声明实践及社交网络结构，揭示了免责声明稀缺、主题分布与互动模式的关联。

**💡 创新点**

创新点在于首次结合两期纵向数据构建规则式免责声明检测框架，并系统比较免责声明在不同主题、网络层级和受众情绪中的出现频率，突出交易主题中免责声明缺失的现实。

**🔧 技术方法**

采用了主题建模（LDA 与 BERT）、情感分析（VADER）、规则式关键词匹配与正则表达式检测免责声明、社交网络中心性分析以及 Spearman 相关分析等多种技术手段。

**📊 数据集**

使用的数据集包括 71 名英国 finfluencer 在 2024 年 4-9 月共 13,215 条视频及 104,097 条评论，以及 2025 年 10 月至 2026 年 3 月的 8,565 条视频跟踪数据。

**📈 对比分析**

通过对两期免责声明出现频率、主题分布与受众情绪进行对比，并利用 Spearman 相关检验视频时长与互动率，结果显示视频时长与互动几乎无关联；免责声明在交易主题中出现率最高，但整体低于预期。

**⚠️ 局限性**

研究的局限性包括：仅检测文字与标签中的免责声明，未识别视频中的语音/视觉免责声明；样本仅限于通过关键词筛选的英国 TikTok 账户，缺乏跨平台或跨地区的普适性；情感与主题分析仅覆盖首期数据。

---

## 682. Reason, Reward, Refine: Step-Level Errors Corrections with Structured Feedback for Physics Reasoning in Small Language Models

**arXiv ID:** 2607.05199 | [PDF](https://arxiv.org/pdf/2607.05199v1)

**作者:** Raj Jaiswal `[一作]` (IIIT Delhi), Rajiv Ratn Shah `[通讯]` (IIT Kanpur)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于步骤级奖励的物理推理训练框架，利用外部验证器识别首个错误并生成针对错误类型的结构化反馈，通过策略梯度和KL正则训练小语言模型提升推理准确率。

**💡 创新点**

创新点在于：①无需标注步骤级偏好数据；②通过奖励惩罚首个错误位置；③根据错误类型生成结构化反馈；④仅在训练时使用外部验证器，推理时不额外调用。

**🔧 技术方法**

采用步骤级奖励与策略梯度+KL正则、GPT‑4o 作为验证器与反馈生成器、LoRA 微调、检索增强生成（RAG）等技术。

**📊 数据集**

训练数据为 2,494 条 JEE 物理问题集；评估数据为 5 个物理基准（SciEval、MMLU、JEEBench、PhysicsQA 等）。

**📈 对比分析**

与 CoT、RAG、SFT、DPO 四种基线对比，四大开源模型在 5 个基准上平均提升 10–20%（最高 27%），计算误差从 56.9% 降至 23%，概念误差仍高于 68%。

**⚠️ 局限性**

局限性包括：概念误差仍高；依赖 GPT‑4o 验证与反馈，未验证多语言或其他科学领域；单次训练无多种种子验证；结构化输出格式化依赖严格；未对各反馈通道做单独消融。

---

## 683. Unified Audio Intelligence Without Regressing on Text Intelligence

**arXiv ID:** 2607.05196 | [PDF](https://arxiv.org/pdf/2607.05196v1)

**作者:** Zhifeng Kong `[一作]` (NVIDIA), Wei Ping `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了一个名为‑30B‑A3B 的单模型音频大语言模型，集成音频感知、语音识别、翻译、文本转语音和文本转音频等多模态任务，并在保持文本推理能力的前提下实现音频相关任务的最先进性能。

**💡 创新点**

创新点在于将音频编码器、MLP适配器和扩展词表直接嵌入 30B MoE 结构中，采用统一的离散音频码本（X‑Codec2 与 X‑Codec）与文本 token 共用解码器，且通过多阶段 SFT 与 Cascade RL 训练策略实现了音频理解与生成与文本推理的无缝融合。

**🔧 技术方法**

核心技术包括 Mixture‑of‑Experts (MoE) Mamba‑Transformer 主干、AF‑Whisper 音频编码器、X‑Codec2/X‑Codec 离散音频码本、Classifier‑Free Guidance、分阶段 SFT（音频预热 → 生成 SFT → 生成+理解 SFT）以及基于 Nemotron‑Cascade 的 RLHF+MOPD 推理优化。

**📊 数据集**

训练数据涵盖 394M 条样本、1,092K 小时音频，包含文本 SFT、ASR、AST、TTS、TTA、音频理解等任务；数据来自公开许可的语音语料（Whisper、LibriSpeech、Fleurs 等）、文本多模态数据（AudioLDM、ETTA、Tango 等）和自建的混合文本‑音频对。

**📈 对比分析**

在多模态基准上，‑30B‑A3B 在语音识别、语音翻译、文本转语音、文本转音频以及音频理解任务上均超过或逼近目前公开与专有模型的最优结果；在文本推理、知识、对齐、长上下文和智能体任务上与 Nemotron‑Cascade 2 维持或略有提升，几乎无文本性能衰退。

**⚠️ 局限性**

主要局限包括：1) 单阶段 SFT 在长上下文任务中表现不佳，提示多阶段训练更稳健；2) 在音频生成方面仍受码本分辨率和增强 VAE 质量限制，导致长时音频一致性与音质不如扩散模型；3) 训练成本极高，仅 NVIDIA H100 GPU 512 卡即可完成，难以普及；4) 仅进行文本域 RL，未探索音频域 RL 以进一步提升音频任务。

---

## 684. When Claws Remember but Do Not Tell: Stealthy Memory Injection in Persistent Personal Agents

**arXiv ID:** 2607.05189 | [PDF](https://arxiv.org/pdf/2607.05189v1)

**作者:** Yechao Zhang `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在持久化个人助理中，利用外部邮件内容进行隐蔽内存注入（Stealth Memory Injection）攻击，并提出了完整生命周期评估基准 WhisperBench 以及一次性攻击生成框架 MemGhost。

**💡 创新点**

创新点包括：①设计全周期评估基准 WhisperBench，覆盖注入、隐蔽性和后续效果；②构建环境代理与目标代理的双重代理体系，利用 rubrics 生成密集奖励并在离线环境中训练一次性攻击策略；③通过监督微调与强化学习（GRPO）实现高效、可迁移的攻击 payload；④验证攻击对不同模型、框架、内存后端的跨平台迁移能力。

**🔧 技术方法**

使用技术：大型语言模型（GPT‑5.4、Claude Sonnet、DeepSeek 等）、强化学习（GRPO）、监督微调、环境代理模拟（本地 IMAP/SMTP 与文件系统/向量数据库）、Rubric‑based reward 体系、Mailpit 等邮件仿真工具。

**📊 数据集**

数据集：WhisperBench 共 108 条案例，涵盖 5 类风险（健康/安全、财务损失、信息完整性、网络安全、运营中断），包含事实与偏好两种毒化，并结合真实邮件工作流；内部生成的高分种子用于训练。

**📈 对比分析**

对比方法：与 7 个基线（5 个手工模板 + 2 个自动搜索）在 6 种 LLM + 2 种 agent 框架（OpenClaw、Claude SDK、NanoClaw、Hermes）以及 2 种内存后端（文件系统、Mem0）下评估。结果显示 MemGhost 在背景执行下可达 71.4% E2E 成功率，在前景下 48.2%；相较于基线显著提升。攻击在多模型、多后端迁移后仍保持 80%+ 成功率，并能在现有输入/模型/系统级防御下实现 90%+ FNR。

**⚠️ 局限性**

局限性：仅在单用户单代理环境中评估；仅使用文本邮件，未覆盖多模态附件；未探讨长期内存衰减、持续运维对攻击的影响；未在真实生产环境验证；未考虑多代理或多租户的传播风险。

---

## 685. The Changing Role of Symbolic Methods in Artificial Intelligence

**arXiv ID:** 2607.05168 | [PDF](https://arxiv.org/pdf/2607.05168v1)

**作者:** Jun Sun `[一作]` `[通讯]` (Singapore Management University), Jun Sun (Singapore Management University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出了压缩原理（Compression Principle）与建模‑推理权衡（Modeling–Reasoning Trade‑off）框架，阐述符号推理在人工智能中的必要性与演变；

**💡 创新点**

将符号推理的角色从“智能核心”转变为“人机接口”，并通过压缩原理解释为何更丰富的模型能降低对符号推理的需求；

**🔧 技术方法**

主要采用理论推导与概念性论证，无具体算法实现；

**📊 数据集**

无；论文为概念性讨论，没有使用数据集；

**📈 对比分析**

无实验对比，文章以历史案例和逻辑论证说明理论的可行性；

**⚠️ 局限性**

缺乏实证验证，理论假设未在真实系统中检验，未给出具体实现细节或性能评估。

---

## 686. Geometry-Aware Visual Odometry for Bronchoscopic Navigation via High-Gain Observer Fusion

**arXiv ID:** 2607.05162 | [PDF](https://arxiv.org/pdf/2607.05162v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 687. MeGA-MP: Metric Graph Advection Message Passing -- A Physics-Informed Message Passing Operator for Advection-Dominated Metric Graphs

**arXiv ID:** 2607.05167 | [PDF](https://arxiv.org/pdf/2607.05167v1)

**作者:** Janine Strotherm `[一作]` (Bielefeld University), Barbara Hammer `[通讯]` (Bielefeld University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种面向度量图线性对流动力学的物理信息消息传递网络（MeGA‑MP），并可通过加入可学习组件扩展为MeGA‑MP⁺以处理对流-反应系统。

**💡 创新点**

创新点在于将线性对流的数值结构直接嵌入消息传递算子，提供可证明的离散误差上界；实现对不同拓扑的零样本泛化；通过可学习组件实现对流-反应动力学。

**🔧 技术方法**

技术手段包括特征曲线（Method of Characteristics）推导的消息函数、迭代消息传递与时间插值、可学习的MLP扩展；与传统GNN、PDE‑GNN、GPSConv、A‑DGN等基线模型对比。

**📊 数据集**

使用水分配网络（Hanoi 和 L‑Town）模拟的氯传输数据，以及一维欧几里得域的边值问题（BVP）数据。

**📈 对比分析**

在节点时间序列回归任务中与多种基线模型比较：MeGA‑MP 在对流任务上 MAE 约为 0.0015，MeGA‑MP⁺ 在对流-反应任务上 MAE 约为 0.0008；在未见拓扑上保持零样本性能；计算时间比传统模拟器快，略慢于部分 GNN 基线。

**⚠️ 局限性**

局限性包括仅适用于线性对流主导的 PDE；误差受时间离散和迭代次数影响；需要已知流场；缺乏不确定性量化。

---

## 688. Open Problems in AI Incident Governance

**arXiv ID:** 2607.05163 | [PDF](https://arxiv.org/pdf/2607.05163v1)

**作者:** Harleen Kaur Sidhu `[一作]` (Independent), Rokas Gipiškis `[通讯]` (AI Standards Lab)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统梳理了AI事故治理的定义、分类、监测与报告现状，识别各阶段的开放问题，并提出统一的监测准则、报告模板与治理原则，旨在提升跨框架的可比性与学习效能。

**💡 创新点**

创新点在于首次将AI事故治理从定义到分析的全流程进行系统性映射，构建可操作的监测与报告原则，并提出多框架兼容的标准化报告模板，为后续经验归纳和风险预测奠定基础。

**🔧 技术方法**

主要采用文献综述、比较分析与政策推演技术，对OECD、EU AI Act、CSET等现有框架进行对比，随后基于研究发现制定监测准则与报告模板。

**📊 数据集**

参考的主要数据集包括AI Incident Database（AIID）、OECD AI Incident Monitor（AIM）、AIAAIC Repository、CSET Harm Framework等公开事故数据库，用于评估现有定义与分类的一致性与缺陷。

**📈 对比分析**

比较方法主要是对框架属性（定义、分类、监测指标、报告要素）进行系统性对照，并未进行量化实验；因此在性能方面的评价以“是否统一、是否可操作”作为主观评估，未给出客观指标。

**⚠️ 局限性**

局限性包括：缺乏实证验证（未在真实事故案例中检验所提模板效果）；仅聚焦于已有公开框架，未覆盖所有地区法规；所提原则和模板为初步建议，需后续实地部署与迭代完善。

---

## 689. Algebraic Modelings of the Supersingular Isogeny Problem

**arXiv ID:** 2607.05160 | [PDF](https://arxiv.org/pdf/2607.05160v1)

**作者:** Alessio Caminata `[一作]` (University of Genova), Silvia Sconza `[通讯]` (University of Zurich)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

本文提出一种新的代数建模方法，用Renes多项式（适用于Montgomery形式的2-同余和三角形式的3-同余）来描述超奇异同余问题（SIP），并构造了对应的多项式系统；

**💡 创新点**

创新点在于：①将Renes公式直接转化为多项式约束，得到零维、可并行求解的系统；②证明这些系统不满足一般坐标条件，解释了其求解难度；③通过实验验证Renes多项式模型在求解度和时间上显著优于传统的模多项式模型；

**🔧 技术方法**

使用的技术包括：Renes多项式的构造、Montgomery/三角曲线形式的特化、Gröbner基法（Magma F4实现）、解度估计、系统并行化；

**📊 数据集**

实验数据集为在随机选取的 10~32 位素数 p 上构造的超奇异椭圆曲线（Montgomery 或三角形式），并生成长度 m 的非回退同余路径；

**📈 对比分析**

与模多项式模型比较时，测量了平均求解时间和最高求解度；结果显示 Renes 模型的求解度普遍更小，求解时间比模多项式快 2~3 个数量级；在 24 位素数下可解 2^15 长度的路径，模多项式仅能到 2^12；

**⚠️ 局限性**

局限性包括：①仅适用于 2 或 3 的幂次同余；②系统并非半正则、非一般坐标，理论上更难分析；③实验仅限 32 位素数，未尝试更大规模或更高同余阶；④未利用拆分求解技术，可能进一步提升性能。

---

## 690. Towards Quantifier-Free Interpolation in Array Languages with Unbounded Data Specifications

**arXiv ID:** 2607.05126 | [PDF](https://arxiv.org/pdf/2607.05126v1)

**作者:** Rodrigo Raya `[一作]` (Technical University of Madrid), Christophe Ringeissen `[通讯]` (Université de Lorraine)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究并证明了几种基于数组的理论（尤其是组合数组逻辑与简单平面数组片段）的量化无关插值性（quantifier‑free interpolation）与统一插值性（uniform interpolation），并给出了对应的理论证明与归约方法。

**💡 创新点**

① 引入了对关系符号参数化的迭代 diff 函数，扩展了传统的 diff 操作，保持了量化无关插值性；② 证明组合数组逻辑在存在迭代 diff 的前提下仍具有强合并性质，从而得到完全量化无关插值；③ 首次展示组合数组逻辑不具备统一插值性，并阐明其原因；④ 在集合与基数约束理论及简单平面数组片段中提供了量化无关与统一插值的证据。

**🔧 技术方法**

使用了模型论中的合并（amalgamation）与强合并（strong amalgamation）理论、Feferman‑Vaught 型定理、局部理论扩展、递归定义的 diff 运算以及存在量化消解技术（Presburger 代数与集合/基数约束的量化消解）。

**📊 数据集**

无：本文为理论性质证明，不涉及实验数据集。

**📈 对比分析**

无：由于为纯理论研究，未进行实验对比或性能评估；所有结论均来自逻辑证明与结构化归约。

**⚠️ 局限性**

局限性：
- 组合数组逻辑在没有迭代 diff 的情况下不满足合并性质，因而不具备量化无关插值。
- 统一插值性仅在具备量化消解的理论中得到保证，组合数组逻辑和简单平面数组片段均不满足统一插值。
- 证明依赖于稳定无限（stably infinite）假设的简化，某些结论在该假设被放宽时需重新验证。

---

## 691. Agent Data Injection Attacks are Realistic Threats to AI Agents

**arXiv ID:** 2607.05120 | [PDF](https://arxiv.org/pdf/2607.05120v1)

**作者:** Woohyuk Choi `[一作]` (Seoul National University), Byoungyoung Lee `[通讯]` (Seoul National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并系统评估了“Agent Data Injection”攻击（ADI），揭示其在现有AI代理中的真实威胁，并通过对多种Web和编程代理的三类实战攻击（任意点击、远程代码执行、供应链攻击）进行验证。

**💡 创新点**

创新点在于：①引入了ADI这一新的间接提示注入（IPI）类别；②提出并证明了概率性分隔符注入（probabilistic delimiter injection）作为实现ADI的核心技术；③系统评估了ADI在多模型、多代理、多防御策略下的有效性，揭示现有防御对ADI的不足。

**🔧 技术方法**

主要技术包括：概率性分隔符注入、LLM上下文分析、对JSON与Web DOM结构的解析与篡改、攻击脚本自动化、对现有防御机制（模型硬化、输入/输出防护、Plan‑then‑Execute、Agent Sandboxing、Dual‑LLM、数据流跟踪、随机化、消毒）进行实验评估。

**📊 数据集**

使用了来自实际工具响应的真实数据集：七类数据（日历事件、云盘文件、GitHub评论、邮件、GitHub问题、论文评审、Web DOM）共157个测试案例；并使用六个主流LLM（GPT‑5.2、GPT‑5‑mini、Claude Opus 4.5、Claude Sonnet 4.5、Gemini 3 Pro、Gemini 3 Flash）以及AgentDojo基准数据进行评估。

**📈 对比分析**

与传统指令注入攻击（Instruction Injection）对比，ADI在无防御情况下攻击成功率（ASR）高达31–100%，而指令注入几乎为0%；在现有防御中，除CaMeL Strict外大多仍允许20–50%的ADI成功。随机化对JSON有效但对DOM有限，消毒虽降低ASR但大幅牺牲实用性；数据流跟踪（严格版）能够完全阻止ADI，但实现难度大、实用性低。实验表明，现有防御体系对ADI的覆盖率不足。

**⚠️ 局限性**

局限性包括：①假设攻击者已知代理数据格式，实际获取难度和成功率未系统评估；②随机化仅适用于键值格式，对非结构化文本（Markdown等）无效；③消毒会导致合法结构信息被删失，严重影响代理实用性；④评估集中于公开API与现有代理，未涵盖更广泛的自定义工具链；⑤严格数据流跟踪虽有效但实现成本高、易失效，尚需进一步研究。

---

## 692. Communication-Aware Placement and Pruning for Efficient Mixture-of-Experts Inference

**arXiv ID:** 2607.05116 | [PDF](https://arxiv.org/pdf/2607.05116v1)

**作者:** Xiao Shi `[一作]` (Sun Yat-sen University), Yutong Lu `[通讯]` (Sun Yat-sen University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出 CAP 框架，结合专家共激活驱动的放置、通信与计算权衡调整以及通信感知的专家剪枝，实现 MoE 推理的高效化。

**💡 创新点**

①首次将专家共激活信息用于通信最小化的放置；②通过可调 λ 形成通信‑负载权衡光谱；③将剪枝从仅计算视角扩展到设备级通信成本优化。

**🔧 技术方法**

基于共激活概率构建加权图进行分区；多阶段贪心放置与节点映射；通信‑负载多目标优化；设备级贪心剪枝；轻量化 GPU 并行实现。

**📊 数据集**

使用 Qwen3‑30B‑A3B 与 DeepSeek‑V2‑Lite 两大 MoE 模型；推理吞吐评测基于 LMSYS‑Chat 与 Arxiv Abstracts；准确率评估使用 HumanEval、MMLU 与 GSM8K。

**📈 对比分析**

对比默认 sequential 放置与 DeepSeek EPLB（负载平衡放置）。在三种节点（RTX3090、A100、H100）和两种集群配置下，CAP 在通信受限环境中提升 1.23×–1.86× 吞吐；在相同加速阈值下保持更低的准确率下降，显示更优的性能/准确率权衡。

**⚠️ 局限性**

①离线优化阶段需要数分钟的统计与调优；②剪枝阈值和通信成本 c 的设定需经验；③实验仅覆盖 2‑节点集群，未验证更大规模或更复杂拓扑；④仅考虑了 GPU‑direct/InfiniBand 经典网络，未探讨 NVMe‑over‑PCIe 等新型传输。

---

## 693. Physiological Noise Augmentation Improves Non-Invasive Brain-to-Speech

**arXiv ID:** 2607.05165 | [PDF](https://arxiv.org/pdf/2607.05165v1)

**作者:** Benjamin Ballyk `[一作]` (University of Oxford), Oiwi Parker Jones `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种基于ICA的生理噪声增强（PNA）方法，用于非侵入式脑转语音解码，训练时注入可变的眼动和心电噪声以提升解码鲁棒性。

**💡 创新点**

创新点在于将参考通道（EOG/ECG）与ICA分离的噪声成分相结合，生成与原标签一致、符合生理统计的增强样本，从而逼近各向异性Jacobian正则化并实现对任务无关噪声的显式不变性。

**🔧 技术方法**

采用ICA分离、噪声成分识别与重新混合、随机幅度采样、K次试验平均、以及EEGNet/MLP分类器和交叉熵/平方误差损失。

**📊 数据集**

使用MegNIST单被试MEG数据集（12000条想象数字实验），该数据集同时记录EOG/ECG参考信号。

**📈 对比分析**

与仅使用原始数据、仅平滑/频移/时间移位/幅度缩放等传统增强基线相比，PNA在10次试验平均后可将EEGNet准确率提升至约76.3%（比基线高约4.7%），并在单试验或少量平均时也表现出显著优势。

**⚠️ 局限性**

局限性包括：需在采集时记录参考通道；仅在单被试且受限词汇的实验上验证；对未跟踪噪声仍需平均或其他方法；未来需扩展到多被试、更多噪声来源及更大词汇表。

---

## 694. FlatManifold: Robust Continual Learning under Severe Label Noise and Domain Shifts via Intrinsic Manifold Flattening

**arXiv ID:** 2607.05201 | [PDF](https://arxiv.org/pdf/2607.05201v1)

**作者:** Rai Hisada `[一作]` (University of Fukui), Kanji Tanaka `[通讯]` (University of Fukui)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种利用 Nyström 映射将视觉特征展开到正交化 RKHS 的连续学习框架，能在高噪声与域漂移环境下保持鲁棒性。

**💡 创新点**

创新在于用一次性的非线性展开与固定正交目标拓扑相结合，再辅以协方差保持的拓扑制动，消除传统样本过滤的复杂性与误差累积。

**🔧 技术方法**

采用 Nyström kernel 近似、RKHS 正交化、岭回归线性最小二乘优化、协方差保持的持续拓扑制动等技术。

**📊 数据集**

使用 NCLT（North Campus Long-Term）多季节多光照的长时序机器人导航数据集。

**📈 对比分析**

通过与 Raw+Ridge、Raw+Adam 两种基线对比，10 个不同季节会话的 100 类检索准确率平均提升 5–8%，Raw+Adam 几乎归零。

**⚠️ 局限性**

局限性包括：对大规模类别扩展性未验证；仅在对称标签噪声下评估，非对称噪声与动态场景的鲁棒性未知。

---

## 695. Latent Programming Horizons in Coding Agents

**arXiv ID:** 2607.05188 | [PDF](https://arxiv.org/pdf/2607.05188v1)

**作者:** André Silva `[一作]` (KTH Royal Institute of Technology), Martin Monperrus `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究编码代理在迭代编辑代码时，其Transformer残差流中隐含的潜在程序表示，并通过线性探测器解码程序的编译性、正确性、进步度和回归等属性；

**💡 创新点**

首次证明编码代理不仅能在残差流中编码当前程序属性，还能在未来多步（约25步）内预测编辑结果，展现出显著的编程视野；

**🔧 技术方法**

使用逻辑回归线性探测器对残差流做分类，结合look‑ahead horizon 预测未来编辑结果，并对跨数据集进行转移测试；

**📊 数据集**

在两款开放权重模型（CodeX、CodeB）下，于两大基准（Verified 与 Pro）收集了 22,714 条编辑轨迹，涵盖 12,183 步长的真实代码任务；

**📈 对比分析**

与随机标签对照和跨基准转移评估相比，最佳层的 AUC 最高可达 0.83，且未来 25 步的预测仍显著高于 0.5，证明模型隐藏层携带丰富的语义信息；

**⚠️ 局限性**

仅证明了属性可线性解码，未证明因果关系；标签极不平衡导致某些属性（如回归）性能偏低；研究仅覆盖两模型两基准，需在更广泛模型与任务上验证。

---

## 696. AgentGym2: Benchmarking Large Language Model Agents in De-Idealized Real-World Environments

**arXiv ID:** 2607.05174 | [PDF](https://arxiv.org/pdf/2607.05174v1)

**作者:** Zhiheng Xi `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AgentGym2框架，用于在去理想化、真实世界环境中评估语言代理的端到端任务完成能力。

**💡 创新点**

创新点在于：1）提供基础可组合工具箱（Web浏览、检索、文件处理、多模态理解、代码执行）而非预选工具；2）任务构建以真实用户需求为基础，加入工具发现、噪声干扰与模糊信息；3）实现环境隔离、并行调用和可扩展评估。

**🔧 技术方法**

采用ReAct式交互模型，工具接口遵循OpenAI工具模式；使用LLM作为评判器；通过并行异步调用支持多工具同时调用。

**📊 数据集**

任务数据集共437个实例，涵盖27个领域，按三大场景分布：复杂工具使用（182）、数据分析（57）、深度搜索（198）。数据来源于GitHub、Reddit、Kaggle等公开平台，并通过自动合成、人工校验构建。

**📈 对比分析**

与多种商用与开源LLM（GPT‑5、Claude‑4.5‑Sonnet、Gemini‑2.5‑Pro、Qwen、Nex‑N1、DeepSeek等）进行对比；在Avg@3指标上，GPT‑5仅达46.15，顶尖开源模型Nex‑N1‑671B仅32.19，表明即便是SOTA模型在去理想化任务上仍表现不佳。

**⚠️ 局限性**

局限性包括：1）评估仍基于单一工具箱，未覆盖所有可能的专业工具；2）LLM评判器的可靠性虽高但仍可能出现偏差；3）对多任务与长篇对话的实时性评估尚未深入；4）数据集规模相对有限，缺乏持续更新与社区共建机制。

---

## 697. RABBiT: Rapidly adaptive BOLD foundation model via brain-tuning for accurate zero-shot and few-shot prediction of speech-elicited responses in the brain

**arXiv ID:** 2607.05171 | [PDF](https://arxiv.org/pdf/2607.05171v1)

**作者:** Omer Moussa `[一作]` (Max Planck Institute for Software Systems), Mariya Toneva `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

RABBiT 是一个小型的音频到 fMRI 基础模型，能够在没有额外 fMRI 数据的情况下零样本预测语言刺激下的脑活动，并通过极少的个体校准数据实现高效个性化。

**💡 创新点**

其创新点包括：1）使用脑调优的自监督语音骨干；2）跨区域注意力的时间脑变压器（Temporal Brain Transformer）以学习区域特定的时间聚合；3）共享–个体化低秩分解（SID）实现共享结构与个体差异的高效分离。

**🔧 技术方法**

采用了 LoRA 膨胀的脑调优、跨注意力 Transformer、低秩共享/偏差矩阵 SID、Wav2Vec2.0 语音模型以及与线性回归和 TRIBEv2 的对比基线。

**📊 数据集**

训练集来自 Friends 子集（6 名参与者，约 565K TR），评估集包括 Narratives（7 章节）和 Le Petit Prince（49 名参与者），共 324 名未见参与者。

**📈 对比分析**

与 TRIBEv2、线性基线和全秩个体头基线比较，RABBiT 在零样本情形下达到或超过 TRIBEv2 并匹配全秩基线；在少样本（10 分钟）下比两者大幅提升，尤其在高阶语言区表现突出。

**⚠️ 局限性**

局限性包括：仅在英语自然听觉任务上评估，训练样本量有限，对多语言、多模态或阅读场景的泛化未知，且跨数据集扩展时效果不稳定。

---

## 698. Claim-Level Rubric Rewards for Video Caption Reinforcement Learning

**arXiv ID:** 2607.05150 | [PDF](https://arxiv.org/pdf/2607.05150v1)

**作者:** Mingqi Gao `[一作]` (Tsinghua University), Yansong Tang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Claim‑Level Rubric Rewards（CuRe）框架，通过将视频字幕拆解为细粒度的视觉主张（atomic claims），并在强化学习（GRPO）中对这些主张进行核验与校准，从而提升字幕的事实性与描述密度。

**💡 创新点**

创新点包括：① 结构化的rubric将字幕拆解为类别感知的原子主张，实现从全局评价向细粒度验证的转变；② 参考锚定校准机制，在赋分时对匹配到参考主张的内容给与完整奖励，对额外视频支持的细节给予衰减奖励，抑制冗余与hallucination；③ 通过长度惩罚与类别权重整合多维度分数，得到可解释且覆盖全面的奖励信号。

**🔧 技术方法**

技术细节包括：GRPO强化学习框架；LLM 驱动的主张拆解器；大规模 VLM（如 Qwen3‑VL）做主张核验与语义匹配；参考锚定校准公式（含 w_u、λ 等超参）；长度惩罚与类别权重融合；在训练中使用 8,000 视频样本的 RL 采样与评估。

**📊 数据集**

数据集与资源：SFT 预训练使用 78,144 条视频字幕；RL 训练使用 8,000 视频子集；参考字幕来源于 Gemini‑3.0‑Pro‑distilled；评估采用 EventHallusion、VCapsBench、DREAM‑1K、Prism（VideoMME、TOMATO、MVBench 等）等公开视频字幕与 VQA 基准；预训练后续实验使用 Molmo2‑Cap、Tarsier2‑Recap、LLaVA‑Video‑Caption 等字幕语料。

**📈 对比分析**

与开源基线（Tarsier、OwlCap、LLaVA‑Video、InternVL3.5、Qwen2.5‑VL、Qwen3‑VL）以及专有模型（Seed 2.0 Pro、Gemini 3.0 Pro）进行对比。CuRe 在 EventHallusion、VCapsBench、DREAM‑1K 的事实性与一致性指标均优于同尺寸基线，甚至在多项指标上超越更大模型；在 Prism caption‑to‑QA 评估中，CuRe 的整体得分比同尺寸基线高 5.6 p，且接近专有模型；在 VLM 预训练下的下游任务平均提升 2 p 左右，覆盖多种基准。

**⚠️ 局限性**

局限性：① 计算成本高，奖励堆栈需多次调用大型 VLM，限制了验证器种类与参考源扩展；② 仅在单一策略主干、单一参考来源上验证，未展示跨模型/多源普适性；③ 仍可能产生幻觉、遗漏上下文或偏见，需人工审核与隐私防护；④ 仅针对公开数据集与特定 VQA 框架，未覆盖所有视频域与语言。

---

## 699. DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation

**arXiv ID:** 2607.05147 | [PDF](https://arxiv.org/pdf/2607.05147v1)

**作者:** Xin Cheng `[一作]` (Peking University), Wenfeng Liang `[通讯]` (DeepSeek-AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DSpark，一种结合并行草稿和轻量级顺序头的半自回归推理框架，用硬件感知的前缀调度器动态控制验证长度，以提升大语言模型在高并发环境下的推理吞吐与交互性能。

**💡 创新点**

创新点：① 将并行背骨与轻量级顺序头融合成半自回归结构，有效缓解并行草稿的后缀衰退；② 设计置信度预测头并配合 Sequential Temperature Scaling 进行校准；③ 基于系统负载的硬件感知前缀调度器，以最大化整体吞吐，保持目标分布不变。

**🔧 技术方法**

技术手段：Speculative Decoding、并行草稿 DFlash、半自回归 Markov/RNN head、置信度头 + STS 校准、硬件感知前缀调度、MoE + 滑窗注意力 DeepSeek-V4 架构、异步调度与可变长度 GPU kernel 优化。

**📊 数据集**

数据集：训练使用 1.3M 公开样本的 Open‑PerfectBlend（包含 chat、math、code、instruction）；评估基准为 GSM8K、MATH500、AIME25、MBPP、HumanEval、Live‑CodeBench、MT‑Bench、Alpaca、Arena‑Hard。

**📈 对比分析**

比较方法与性能：在离线评测中通过平均接受长度（τ）对比 EAGLE3（自回归）和 DFlash（并行），DSpark 在 Qwen3‑4B/8B/14B 与 Gemma4‑12B 上均提升 16–30%；在生产部署中对比 MTP‑1 基线，DSpark 使每用户生成速率提升 60–85%，在高 SLA 下吞吐提升数百%，显著扩展了吞吐–交互 Pareto 前沿。

**⚠️ 局限性**

局限性：草稿侧仍存在固定计算成本，难以为低接受率请求提供早停；置信度校准需额外 STS 训练，且对极端离散硬件容量曲线的适应性有限；异步调度可能在极端高并发时产生轻微误差；对结构更复杂、接受率极低的任务尚无专门的难度感知早退出策略。

---

## 700. On the risk of coding before testing: An empirical study on LLM-based test generation workflow

**arXiv ID:** 2607.05139 | [PDF](https://arxiv.org/pdf/2607.05139v1)

**作者:** Michael Konstantinou `[一作]` (University of Luxembourg), Mike Papadakis `[通讯]` (University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对LLM先生成代码后生成测试的工作流进行实证评估，揭示代码错误会被错误地复制到随后的测试中，导致缺陷检测效果下降。

**💡 创新点**

首次量化了“错误传播”现象，证明代码生成后再生成测试会破坏测试与实现的独立性，显著削弱缺陷识别能力。

**🔧 技术方法**

采用GPT‑5‑mini、GPT‑4.1‑mini、DeepSeek‑V4、Claude Haiku 4.5、Llama 3.3 Instruct等多种LLM，并使用Prompt‑only、Summarization、CoT、CoVe、Agentic workflow等提示工程策略进行实验。

**📊 数据集**

使用HumanEval+、MBPP、BigCodeBench三大Python编码基准构造故障实现，形成实验数据集。

**📈 对比分析**

通过比较三种提示配置（仅任务描述、代码+描述、仅代码）以及三种提示策略，实验显示测试先于代码的工作流检测率提升约10–17%，而代码先后导致检测率下降约9–18%，验证了错误传播的负面影响。

**⚠️ 局限性**

研究仅在单文件Python任务、少量LLM和特定工作流上验证，未考虑大型项目、跨文件依赖等情境，结果的通用性和可扩展性仍需进一步验证。

---

## 701. ClassicLogic: A Knowledge-Driven Benchmark of Classic Puzzle Games for Evaluating Compositional Generalization

**arXiv ID:** 2607.05185 | [PDF](https://arxiv.org/pdf/2607.05185v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 702. UNIVERSE: Unified Video Action Models for Autonomous Driving with Flexible Mask-Modulated Modality Generation

**arXiv ID:** 2607.05133 | [PDF](https://arxiv.org/pdf/2607.05133v1)

**作者:** Mengmeng Liu `[一作]` (University of Twente), Michael Ying Yang `[通讯]` (University of Bath)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

联合训练视频预测与轨迹生成的统一Diffusion Transformer，支持仅轨迹推理。

**💡 创新点**

通过共享参数和模态解耦可见性掩码，将视频监督直接注入轨迹去噪，实现更强的跨域泛化与推理加速。

**🔧 技术方法**

Mask-modulated Diffusion Transformer、Flow-matching损失、模态可见性掩码、视频VAE、文本编码器等。

**📊 数据集**

NAVSIM v1、nuScenes、Bench2Drive。

**📈 对比分析**

在NAVSIM上PDMS 91.0，优于传统端到端与世界模型方法；零样本迁移到nuScenes/Bench2Drive时L2误差显著降低，推理速度提升4.3×，内存更低。

**⚠️ 局限性**

仅在无视野掩码或拆分Diffusion Transformer时性能下降，且对未来视频生成质量仍有限。

---

## 703. From Multiplicity to Vulnerability: Privacy Amplification Risk from One-Dataset-Multiple-Model Exposure

**arXiv ID:** 2607.05111 | [PDF](https://arxiv.org/pdf/2607.05111v1)

**作者:** Qirui Huang `[一作]` (University of Western Australia), Yansong Gao `[通讯]` (University of Western Australia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了同一数据集训练多模型（ODMM）后，联合暴露所带来的隐私泄漏风险，并提出理论框架与攻击方法

**💡 创新点**

创新点在于首次证明ODMM模型的成员推断泄漏会随模型数目累积（ODMM隐私组合），并通过任务多样性与相互独立性阐释泄漏放大机制

**🔧 技术方法**

使用的技术包括理论证明（组合定理）、成员推断攻击（MIA）与RMIA、shadow模型训练、任务特定特征提取与联合信息融合、随机森林/MLP等攻击分类器

**📊 数据集**

实验数据集涵盖图像与文本领域：UTKFace、CelebA、FairFace、HAM10000、Blog Authorship Corpus 等

**📈 对比分析**

通过与单任务MIA比较，联合攻击在各数据集与模型（ResNet、ViT、Qwen3-1.7B等）上AUC提升12–16%，在低FPR区间提升更显著；对DP、硬标签等防御手段进行对比，验证了ODMM风险的普遍性

**⚠️ 局限性**

局限性包括：仅在黑盒查询场景下评估；假设攻击者已知模型共享数据源；对共享骨干的多任务学习效果探讨有限；未覆盖更强防御技术或模型解释性；实验规模受限于公开数据集与模型

---

## 704. FSDC-DETR: A Frequency-Spatial Domain Collaborative DETR for Small Object Detection

**arXiv ID:** 2607.05176 | [PDF](https://arxiv.org/pdf/2607.05176v1)

**作者:** Aiwen Liu `[一作]` (Micro-Intelligence), Zhiyi Pan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于频域与空间域协同的Transformer检测框架FSDC‑DETR，专门解决小目标检测难题。

**💡 创新点**

创新点在于：① 双分支频域‑空间自适应融合（DBFSAF）增强高频信息；② 结构感知频域空间特征融合（SFS‑FF）实现跨尺度频域交互；③ 频域动态下采样（FSD‑Down）在尺度变换中保持高频。

**🔧 技术方法**

使用Transformer DETR框架，双分支CNN‑ViT骨干，FDConv、频域频率选择、深度可分离卷积、DWT+组卷积等技术。

**📊 数据集**

在VisDrone‑DET2019和AITODv2两大航拍小目标数据集上进行评估。

**📈 对比分析**

与DEIMv2、RT‑DETRv4等现有SOTA相比，在AP上提升约6.4~6.6点，尤其小目标AP提升6.8~6.9点，验证了频域协同的有效性。

**⚠️ 局限性**

局限性：模型结构相对复杂，计算和内存开销较大，未针对实时性进行深入评估，且在非航拍场景的泛化性未作系统验证。

---

## 705. Fully Rotation-Equivariant Spectral-Spatial Learning for Multispectral Object Detection

**arXiv ID:** 2607.05148 | [PDF](https://arxiv.org/pdf/2607.05148v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 706. Rethinking On-Policy Self-Distillation for Thinking Models

**arXiv ID:** 2607.05184 | [PDF](https://arxiv.org/pdf/2607.05184v1)

**作者:** Simran Kaur `[一作]` (Princeton University), Sanjeev Arora `[通讯]` (Princeton University)

**通讯引用:** 26544 | [OpenAlex ID](https://openalex.org/A5103209777)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在长预算思考模型中使用特权上下文的自蒸馏对性能的负面影响，发现会削弱推理能力

**💡 创新点**

揭示了特权信息导致的“fork suppression”机制，使得思考模型在关键分支点的探索被抑制，从而导致准确率下降的失败模式

**🔧 技术方法**

采用对抗式on‑policy自蒸馏（OPD）与JSD等token‑级别分散技术，比较无特权与有特权教师的训练效果

**📊 数据集**

使用AIME24/25、HMMT25、OpenThoughts 15k、Countdown等数学推理数据集进行实验

**📈 对比分析**

与无特权OPD、指令调优模型以及不同生成预算长度的评估对比，发现有特权OPD在长预算下avg@16准确率可下降约17%，而无特权OPD可提升

**⚠️ 局限性**

未提供解决方案，因果机制未完全验证，实验仅局限于数学推理任务，可能不适用于其他领域

---

## 707. CP-WSP: A Declarative CP-SAT Framework for Configurable Multi-Constraint Workforce Scheduling

**arXiv ID:** 2607.05177 | [PDF](https://arxiv.org/pdf/2607.05177v1)

**作者:** Vipul Patel `[一作]` (Phi Labs, Quantiphi), Dagnachew Birru `[通讯]` (Phi Labs, Quantiphi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套可声明式的约束编程求解框架 CP‑WSP，用于多约束劳动力排班问题，并在同一模型中实现硬约束保证和软目标优化。

**💡 创新点**

创新之处在于：1）通过 JSON 配置实现全约束可声明化，省去编码；2）采用 x=w‑b 的三状态变量拆分，支持强制休息、休息中心化、并发休息限制等难以表达的约束；3）对硬约束进行结构化强制，软约束通过加权惩罚整合；4）引入网格偏移预处理，零成本支持跨午夜班次；5）提供 36 配置可复现基准集。

**🔧 技术方法**

利用 Google OR‑Tools 的 CP‑SAT 求解器，结合三状态变量分解、网格偏移预处理、JSON 约束描述等技术。

**📊 数据集**

在 INRC‑II 标准基准（5–80 名护士）、NRP‑23 兼容跨午夜班次实例以及 36 个合成配置的自定义基准上进行评测。

**📈 对比分析**

与现有 CP/IP 方案对比，CP‑WSP 在所有实例下实现零硬约束违背；在 n005w4 上证明最优；在 30 名员工 7 天 30 分钟粒度下 120 秒可得到可行解；通过约束消融实验表明完整模型比基线提升 37%，工作量公平度提升 66%；模型规模随员工线性扩展（≈4400 变量/员工）。

**⚠️ 局限性**

模型规模随员工数量线性增长，超过 50 名员工可能需要分解或列生成；小时粒度下 LP 松弛弱，导致优化难度高；当前仅支持静态 JSON 配置，缺少交互式约束学习。

---

## 708. Platonic Projection Structures: Operator-Induced Observability in Representation Learning

**arXiv ID:** 2607.05175 | [PDF](https://arxiv.org/pdf/2607.05175v1)

**作者:** Kazuo Ishii `[一作]` (Suwa University of Science), Javaid Saher `[通讯]` (Kanazawa Gakuin University)

**通讯引用:** 398 | [OpenAlex ID](https://openalex.org/A5090193807)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Platonic Projection Structures（PPS）框架，用以把可观测性建模为自伴正定算子诱导的投影几何，并将其应用于表示学习、知识蒸馏与可解释性分析。

**💡 创新点**

核心创新在于把可观测性视作算子诱导的商几何而非直接观测，统一了量子测量与深度学习输出的投影关系，并给出算子一致性与可解释性限制的理论阐释。

**🔧 技术方法**

采用算子理论、谱分解、商空间构造以及近似交织关系（ΦΠ_T≈Π_SΦ）等技术，对观察算子与表示空间的几何关系进行定量分析。

**📊 数据集**

实验数据包括人工生成的8维与16维嵌入空间以及CIFAR‑10图像数据，用于验证核不变可观测性、秩控制与知识蒸馏的算子一致性。

**📈 对比分析**

通过与传统 KL 蒸馏对比，加入算子一致性正则后实现了与基准相当的分类精度，并显著降低了教师与学生观察算子之间的不一致度。

**⚠️ 局限性**

局限性主要在于仅处理静态线性正定算子，实验为控制性验证而非大规模基准；缺乏非线性、时序或多模态等扩展，且对实际可解释性提升的实证尚待进一步研究。

---

## 709. TacReasoner: A Dynamic Tactile-Language Framework for Interactive Reasoning in Real-World Scenarios

**arXiv ID:** 2607.05131 | [PDF](https://arxiv.org/pdf/2607.05131v1)

**作者:** Kailin Lyu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Ce Hao `[通讯]` (Beijing Zhongguancun Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TacReasoner框架，整合动态感知编码器、链式思维触觉数据集TouchCoT-10K和DynTAC-Bench，实现触觉与语言的动态推理；

**💡 创新点**

创新点包括：①显式建模触觉时序和问答引导的动态感知编码器；②首个触觉链式思维数据集；③两阶段训练+LoRA微调策略；④针对动态推理的全新基准DynTAC-Bench；

**🔧 技术方法**

使用技术有：视觉Transformer+时间差分编码+跨模态注意力、LoRA参数高效微调、Qwen‑2.5 LLM、教师强制训练、问答模板生成CoT；

**📊 数据集**

使用数据集：TouchCoT‑10K（链式思维触觉），DynTAC‑Bench（实时触觉识别与状态估计），VTV‑150K（原始指令数据）与VTV‑150K对齐；

**📈 对比分析**

通过与VTV‑LLM、GPT‑4o、Gemini‑2.5‑Pro‑Exp、LLaVA等模型在VTV‑150K、DynTAC‑Bench上对比，TacReasoner‑7B在多数子任务上优于VTV‑LLM‑14B，平均提升约4‑12%，在动态推理任务上提升25%和7%；

**⚠️ 局限性**

局限性在于数据集规模与多传感器泛化受限，未在真实机器人系统上充分验证，动态感知主要基于视觉触觉视频，缺乏对噪声和硬件差异的鲁棒性。

---

## 710. TREK: Distill to Explore, Reinforce to Refine

**arXiv ID:** 2607.05339 | [PDF](https://arxiv.org/pdf/2607.05339v1)

**作者:** Yuanda Xu `[一作]` (LinkedIn Corporation), Alborz Geramifard `[通讯]` (LinkedIn Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在Group Relative Policy Optimization（GRPO）中通过教师提议与前向KL收敛来扩展未被覆盖的探索支持的训练流程；

**💡 创新点**

创新点在于将教师输出仅作为验证过的提议来做支持扩展，而非传统意义上的模仿或信用分配；并通过提示级路由、提议筛选与前向KL阶段实现高效探索；

**🔧 技术方法**

使用GRPO、教师提议生成、提示路由、前向KL收敛、基于验证器的筛选、以及对比的OPD（on‑policy distillation）等技术；

**📊 数据集**

在数学推理任务AIME 2024/2025、以及Agentic任务ALFWorld和ScienceWorld上进行实验；

**📈 对比分析**

与直接GRPO和OPD对比，实验显示在AIME 2025上Qwen3-8B模型从36.9%提升至40.3%，在ALFWorld中成功率从75.8%提升至82.8%，表现出在难题上显著加速和提升；

**⚠️ 局限性**

局限性包括：依赖验证器质量、简化的可达性评估（修剪NLL）、仅使用已验证轨迹导致可能遗漏有用的近似解、教师查询与验证开销，以及对大规模模型的可扩展性未充分验证。

---

## 711. Socially-Aware Autonomous Doorway Traversal and Payload Delivery for Emergency Assistance

**arXiv ID:** 2607.05315 | [PDF](https://arxiv.org/pdf/2607.05315v1)

**作者:** Andrew Snowdy `[一作]` (Northeastern University), Taskin Padir `[通讯]` (Northeastern University)

**通讯引用:** 2343 | [OpenAlex ID](https://openalex.org/A5009032681)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文设计并实现了一套基于行为树的社交感知门口通行与物品递送机器人框架，用于紧急疏散场景下保持门道畅通并向救援人员递送水瓶、手电筒等救援装备。

**💡 创新点**

创新点包括：①将门按钮识别、门状态估计、人类意图推断与动作规划统一集成至可扩展的行为树架构；②实现了基于实时感知的低延迟冲突预emption机制，能够在遇到行人时即时中断操作；③提出了在机器人运动与抓取时同步使用二次样条轨迹与 Damped Least Squares 逆运动学，以实现平滑且安全的操作。

**🔧 技术方法**

主要技术：YOLOv11n 实时物体检测（门按钮、门、行人、瓶子、手电筒），LiDAR 与 RGB‑D 结合的门姿态与深度估计，BehaviorTree.CPP 行为树决策模块，Damped Least Squares (DLS) 逆运动学求解，五自由度机械臂与全向底盘的协同运动规划，四阶多项式轨迹生成。

**📊 数据集**

使用了约 750 张标注图像（600 张用于训练 150 张用于验证）对 ADA 兼容门按钮与玻璃门进行微调；此外在实验中使用了实际 HSR 机器人与 Gazebo 仿真环境。

**📈 对比分析**

通过 105 次试验（5 种硬件情境 + 3 种仿真情境）验证系统，在硬件上 97/105 次成功（约 92%），在仿真上 10/10 次成功（100%）。相较于传统有限状态机，行为树在多任务切换与冲突预emption 上实现了更高的鲁棒性；在单个任务的门开启与抓取精度与现有规划方法相当或更优。

**⚠️ 局限性**

主要局限：①感知模块对门的半开状态识别不足，导致部分场景误判为关门；②YOLO 模型训练数据量有限，泛化性能受限；③行人意图识别仅基于空间接近，缺乏速度与朝向预测；④依赖外部 Jetson 进行推理，增加系统延迟；⑤在长距离导航中定位漂移可能影响完整链路成功率。

---

## 712. Deep Learning for Semen Analysis in Male Infertility: Computer Vision, Multimodal Fusion, and Clinical Translation

**arXiv ID:** 2607.05311 | [PDF](https://arxiv.org/pdf/2607.05311v1)

**作者:** Runwei Guan `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Yanhua Fei `[通讯]` (Wuhan University)

**通讯引用:** 5751 | [OpenAlex ID](https://openalex.org/A5100355335)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该综述系统梳理了基于计算机视觉与深度学习的精子分析方法，包括检测、跟踪、分割、形态分类、功能和遗传完整性评估，并讨论了多模态融合、鲁棒性与临床转化路线。

**💡 创新点**

创新点在于把精子分析视为多模态信息融合问题，整合公开数据集、评价指标、模型演化、跨中心泛化与可解释性等方面，提出了分阶段临床验证与监管框架。

**🔧 技术方法**

采用的技术包括YOLO/Detectron/DETR等目标检测与跟踪网络，U‑Net/Mask R‑CNN/Segment‑Anything等分割模型，Vision Transformer与注意力网络进行形态分类，QPI、DNA碎片检测与多模态融合网络。

**📊 数据集**

使用的数据集包括SVIA、VISEM、VISEM‑Tracking、SMIDS、HuSHeM、SCIAN‑Morpho、MHSMA、HSMA‑DS等公开精子图像与视频集。

**📈 对比分析**

通过与传统CASA、手工判读、已有深度学习基准的对比，报告的检测mAP、跟踪HOTA、分割Dice、形态分类F1/Accuracy均在70–95%区间，显示在实验室级别具备可比性。

**⚠️ 局限性**

主要局限在数据集规模小、单中心、缺乏人口多样性、跨设备泛化不足、评估指标不统一及缺少真实临床前景验证。

---

## 713. Learning Probabilistic Embeddings for Unsupervised Action Segmentation

**arXiv ID:** 2607.05263 | [PDF](https://arxiv.org/pdf/2607.05263v1)

**作者:** Shuai Li `[一作]` (University of Bonn), Juergen Gall `[通讯]` (University of Bonn)

**通讯引用:** 14045 | [OpenAlex ID](https://openalex.org/A5012240246)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出在无监督时序动作分割中学习概率帧嵌入（高斯分布），并结合图卷积网络（GCN）捕捉时间上下文，通过最优传输（OT）生成伪标签，最终实现更精确的动作分割。

**💡 创新点**

核心创新在于将帧嵌入从确定性改为概率性（采样自高斯分布），通过多次采样产生多样化伪标签以降低过拟合；将GCN用于生成均值与方差，使得嵌入更具时间一致性和不确定性表达；将此方法嵌入现有ASOT/VASOT框架并显著提升性能。

**🔧 技术方法**

采用的技术包括：Kantorovich + Gromov‑Wasserstein 最优传输用于伪标签生成；图卷积网络（GCN）用于学习均值与方差；重参数化技巧与 Monte Carlo 采样结合的跨熵损失；Adam 优化器；以及对比实验中的多种基线方法。

**📊 数据集**

在四个公开数据集上进行评估：Breakfast、YouTube Instructional、50Salads（Eval）和 Desktop Assembly。

**📈 对比分析**

与现有无监督分割基线（ASOT、VASOT、CLOT 等）在 12 个评估指标/数据集上进行对比；结果显示 PEOT 在 9/12 组合上取得最佳成绩，MoF 和 F1 分别提升最高 20.7% 与 19.0%，在 Breakfast 和 Desktop Assembly 上取得全表最佳表现。

**⚠️ 局限性**

局限性包括：相较传统方法需要额外的采样次数 M，导致训练时计算成本略升；模型对采样次数与 GCN 结构参数敏感；目前仅在无监督分割任务验证，未探讨跨任务通用性；伪标签生成过程仍非完全可微，未来可能进一步改进。

---

## 714. CanniUplift: A Holistic Framework for Mitigating Seller and Incentive Cannibalization in E-commerce Uplift Modeling

**arXiv ID:** 2607.05242 | [PDF](https://arxiv.org/pdf/2607.05242v1)

**作者:** Zuwang He `[一作]` (Taobao and Tmall Group of Alibaba), Junxiong Zhu `[通讯]` (Taobao and Tmall Group of Alibaba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 CanniUplift 框架，解决电商多卖家环境下的卖家和激励双重蚕食问题，并在平台层面实现全局一致性与红emption 解构去噪。

**💡 创新点**

创新点在于：① 引入平台级全局一致性约束（PGA）以捕捉跨卖家替代效应；② 采用红emption‑based decomposition denoising（RDD）将处理后结果分解为付费与非付费路径，降低激励蚕食噪声；③ 通过 treat‑attention 细粒度建模用户与候选激励/卖家的交互。

**🔧 技术方法**

技术实现：Transformer‑based 多源序列编码 + treat‑attention；多头网络结构与平台‑头、红emption‑头；Tweedie 损失对 GMV 零膨胀与长尾建模；对比基线 EUEN、DragonNet 等深度因果模型。

**📊 数据集**

数据集：1）大型工业数据（阿里巴巴天猫、淘宝实时日志），包含约百万级用户与卖家；2）合成数据，用于验证卖家蚕食机制。

**📈 对比分析**

与 DragonNet、CFRNet、TARNet、TLearner、EUEN 等现有模型在 AUUC、QINI 等指标上进行对比；在工业数据上，CanniUplift 在卖家和用户层面均实现最高 AUUC/QINI；上线 A/B 测试显示营销成本降低 2.45%，平台增量 GMV 提升 4.08%，ROI 增幅 6.69%。

**⚠️ 局限性**

局限：未考虑时间维度蚕食（促销导致购买时间推移）；依赖历史行为，实时捕捉替代性滞后；未对长期收益或动态替代模式做进一步建模。

---

## 715. Repurposing CLIP to Localize at Pixel Level

**arXiv ID:** 2607.05253 | [PDF](https://arxiv.org/pdf/2607.05253v1)

**作者:** Jiaxiang Fang `[一作]` (Central South University), Shengfeng He `[通讯]` (Singapore Management University)

**通讯引用:** 6841 | [OpenAlex ID](https://openalex.org/A5056103024)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CLIPix 框架，将大规模视觉-语言模型 CLIP 重新用于像素级定位，实现开放集合语义分割。

**💡 创新点**

创新点包括：1）利用 CLIP 分类过程中的注意力激活提取目标专属定位信息；2）Noise‑Resistant Correction（噪声抗性校正）模块消除全局特征偏差产生的噪声；3）Localization Embedding（定位嵌入）策略将定位信息与图像特征融合，提升细节与全局一致性。

**🔧 技术方法**

核心技术：CLIP 的视觉‑文本对齐、梯度归因回溯、注意力权重重映射、特征重加权、轻量化 Transformer 解码器；同时采用 ResNet/MobileNet/EfficientNet 辅助特征编码。

**📊 数据集**

主要使用 PASCAL‑5^i 与 COCO‑20^i 数据集进行二分类开放集合分割评估；还扩展到多类别分割（VOC20、PC59、Cityscapes、ADE20K）。

**📈 对比分析**

与现有 0‑shot/few‑shot 方法（HSNet、SSP、BAM、MIANet、HDMNet、HMNet、AENet、ABCB、PI_CLIP、LLaFS++、DSV‑LFS）以及基础模型（PerSAM、Matcher、VRP‑SAM、SegGPT、Painter）进行对比；在 PASCAL‑5^i mIoU 88.4%、COCO‑20^i mIoU 78.8%，在多类别任务中同样获得最高或相近性能，并在计算量和内存上表现出显著优势。

**⚠️ 局限性**

局限性：1）依赖 CLIP 的图像‑文本对齐，仍受全局偏差影响，难以在极度复杂或高混乱场景中完全去噪；2）对人工制品（如工具、器具）等视觉特征相对单一的类别分割仍有不足；3）虽然轻量化，但在极低算力设备上的实时性能仍需进一步优化。

---

## 716. Curated retrieval versus open web search in public AI information services: a coverage-trust trade-off

**arXiv ID:** 2607.05217 | [PDF](https://arxiv.org/pdf/2607.05217v1)

**作者:** Hafsteinn Einarsson `[一作]` (University of Iceland), Jón Gunnar Þorsteinsson `[通讯]` (University of Iceland)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在冰岛欧盟公投前，对政府资助的公共 AI 服务 Evrópuvefur 进行专家评估，比较检索增强生成 (RAG) 与开放 Web 搜索两种路径的答案质量与引用来源可信度。

**💡 创新点**

创新点在于首次将来源可信度作为独立信息质量维度进行量化评估，揭示 Web 搜索路径的答案在覆盖率高但引用来源可信度低的风险；并提供可复现的专家评审工具和公开数据集。

**🔧 技术方法**

使用 Google Gemini LLM 进行答案生成，采用近似最近邻检索 (FAISS) 处理本地知识库；通过专家评审工具进行人工评估，并利用 LLM 辅助对评审文本进行标签分类。

**📊 数据集**

主要数据集包括：冰岛维基百科式的 Evrópuvefur 本地语料库（742 篇编辑后答案），Open Web 公开网页（约 1,088 个引用），以及由学者创建的 Fact-checking 项目（用于生成问题和可信域名列表）。

**📈 对比分析**

比较方法：对 551 条答案进行七项质量评分和来源标记；使用卡方、Fisher 检验、Wilcoxon 等统计手段评估两路径差异。结果显示 Web 搜索回答率高 (≈97% vs 69%)，但 35% 的 Web 答案包含至少一个被标记为不可信或无关的来源；RAG 仅 6% 被标记，且主要是过时。两路径在除“是否回答问题”之外的质量维度差异不大。

**⚠️ 局限性**

局限性包括：评审未随机分配到两种路径，交叉模式比较受限；来源标记只覆盖单个评审意见，可靠性有限；问题集为人工生成，可能不完全代表真实用户查询；仅评估单一低资源语言和特定公投场景，结果可推广性受限；模型与插件配置随时间变动，实验结果不一定适用于未来版本。

---

## 717. MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution

**arXiv ID:** 2607.05297 | [PDF](https://arxiv.org/pdf/2607.05297v1)

**作者:** Zefeng Wang `[一作]` (LMU Munich), Yunpu Ma `[通讯]` (LMU Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了MetaSkill-Evolve，一个双时尺度递归自我改进框架，能够同时演化任务技能和改进流程的元技能。

**💡 创新点**

将改进流程本身作为可演化的元技能，通过同一五代理管道递归更新，使得系统实现有限递归自我改进，无需额外模型或训练，并引入元生产率P(m|s)与前沿选择相结合。

**🔧 技术方法**

采用Markdown格式的LLM‑agent程序，五代理管道（Analyzer、Retriever、Allocator、Proposer、Evolver），Gemma‑4 31B静态模型，SQLite持久化演化图以及双时尺度更新机制。

**📊 数据集**

使用OfficeQA、SealQA和ALFWorld三大基准数据集，分别对应问答任务和机器人环境。

**📈 对比分析**

在同一Gemma‑4 31B背书上与无技能、静态技能、单层演化等基线对比；在OfficeQA和SealQA分别提升+23.54和+16.09的测试准确率，ALFWorld提升+1.92，慢循环贡献约+6.38/+8.05/+1.92。

**⚠️ 局限性**

仅在受控基准上验证，未测试在开放式长周期任务；五代理管道结构固定，元更新频率H固定，未验证在更大规模或更噪声反馈环境中的迁移性。

---

## 718. Erasing Without Collateral Damage: Precise Concept Removal in Diffusion Models

**arXiv ID:** 2607.05274 | [PDF](https://arxiv.org/pdf/2607.05274v1)

**作者:** Parth Upman `[一作]` (University of Nottingham), Shreyank N Gowda `[通讯]` (University of Nottingham)

**通讯引用:** 680 | [OpenAlex ID](https://openalex.org/A5041351493)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了训练‑free 的概念消除方法 CARE，利用保留子空间协方差方向在交叉注意力值空间中精准抹除目标概念。

**💡 创新点**

创新点在于用协方差感知的保留子空间方向替代原始目标方向，显著减少共享结构被误删，实现闭式无训练消除。

**🔧 技术方法**

采用交叉注意力值空间干预、低秩 Woodbury 逆运算、协方差缩小参数 γ 控制擦除‑保留平衡等技术。

**📊 数据集**

在 Stable Diffusion v1.4 上对实例、艺术风格和名人概念进行评估，使用 CLIP 分数与 FID 作为指标。

**📈 对比分析**

与训练型、训练‑free 以及 AdaVD 等基线对比，CARE 在保持非目标生成质量（FID 降低）同时保持或提升目标消除效果（CS 降低），尤其在风格和名人案例表现最佳。

**⚠️ 局限性**

对高度与保留子空间紧密耦合的概念（如相互关联的名人）消除效果有限，需适度调节 γ 并依赖于保留锚点的选择。

---

## 719. SteelBench: Evaluating Vision-Language Models in Real-World Industrial Environments

**arXiv ID:** 2607.05264 | [PDF](https://arxiv.org/pdf/2607.05264v1)

**作者:** Suryanarayana Reddy Yarrabothula `[一作]` (Indian Institute of Technology Bhilai), Katragadda Ajay RamaSwamy Chowdary Gowtham `[通讯]` (Indian Institute of Technology Bhilai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 SteelBench，一套用于工业监控场景的诊断性基准，评估视觉语言模型在动作识别、安全规则推理和注释来源追溯上的表现。

**💡 创新点**

其创新点包括：①在真实工业 CCTV 视频上实现稠密的 per‑worker 动作、PPE、可见性、空间上下文和安全合规标签；②提出 Provenance‑Aware Audit Protocol，量化模型预填对标注的影响并提供人类基准；③引入多维诊断指标（nAUDC、CRG、ECE、DRS）揭示模型在鲁棒性、推理和校准上的缺陷。

**🔧 技术方法**

本文使用了 9 个视觉语言模型（包括 GPT‑4o、GPT‑5.4、Claude Opus 4.7、Gemini 2.5 Pro/Flash、Qwen3.5‑122B、Llama 4 Maverick、Gemma 4‑31B、Nemotron‑12B），通过 8 帧 1080p 结构化提示和规则化输出解析，结合 Provenance‑Aware Audit 三级指标对模型进行评估。

**📊 数据集**

所使用的数据集为 SteelBench：从一家集成钢厂连续监控摄像头提取的 149 小时原始视频，经过 27:1 采样、时间去重、类别与可见性分层抽样后得到 1 345 条含 9–58 个字段的稠密标注剪辑。

**📈 对比分析**

对比方法上，作者将 9 个 VLM 在同一基准上进行零样本评估，并与人类 84.6% 的标注准确率对照；最佳模型 Qwen3.5‑122B 仅 42.6% 的动作准确率，且在鲁棒性（nAUDC）、安全推理（CRG）、校准（ECE）等诊断检查中均未达到行业可部署阈值。

**⚠️ 局限性**

局限性包括：①数据来源仅为单一钢厂，跨工厂或跨行业泛化未验证；②标注流程仍存在少量锚定偏差，且模型在多工人场景中的错误分布未完全消除；③每条剪辑仅包含 8 帧，缺乏连续视频时序推理；④未涵盖更长时长或不同视频分辨率的实际应用场景。

---

## 720. SalAngaBhava: A Sinhala Market Dataset for Aspect-based Sentiment Analysis

**arXiv ID:** 2607.05259 | [PDF](https://arxiv.org/pdf/2607.05259v1)

**作者:** Lakshani Galwatta `[一作]` (University of Moratuwa), Adithya Galwatta `[通讯]` (Cardiff Metropolitan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并公开了第一份面向 Sinhalic 语言的四元组（Target, Aspect, Opinion, Sentiment）标注的 ABSA 数据集 SalAngaBhava，涵盖多领域电子商务产品评论。

**💡 创新点**

创新点在于：①首次公开 Sinhalic 方面级情感标注；②包含显式与隐式方面，支持代码混合文本；③提供完整的四元组结构，并附带详细统计与质量评估。

**🔧 技术方法**

采用了 Web 爬取、Unicode/正则处理、Google Transliterator 转写、人工双人标注（IAA 评估）、LaBSE 语义相似度分析、FastText 与 TF‑IDF 预处理、以及 mT5‑small 指令微调模型 InstructABSA 进行基线实验。

**📊 数据集**

使用了从 Daraz 网站抓取的 10,989 条清洗后评论，其中 1,858 条被人工标注为四元组，涉及电子产品、家电、化妆品、时尚与食品等五大商品类别。

**📈 对比分析**

通过与 FastText/TF‑IDF 的产品类别分类、LaBSE 的相似度分析以及 InstructABSA 的 ATE/ALSC 基线，对比实验显示 ATE macro‑F1 为 0.72，ALSC macro‑F1 为 0.28，FastText 在类别分类上的 macro‑F1 约 0.50；整体结构一致但正负偏移明显。

**⚠️ 局限性**

局限性包括：方面标注覆盖不均、样本量有限、正面情感极度占优、neutral 类标注极少、各商品类别标注比例差异导致偏差，以及数据仅来源于单一电商平台，缺乏跨域验证。

---

## 721. OptiAgent: End-to-End Optimization Modeling via Multi-Agent Iterative Refinement

**arXiv ID:** 2607.05346 | [PDF](https://arxiv.org/pdf/2607.05346v1)

**作者:** Adriana Laurindo Monteiro `[一作]` (Instituto de Ciência e Tecnologia do Itaú), Victor Leme Beltran `[通讯]` (Instituto de Ciência e Tecnologia do Itaú)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多代理框架OptiAgent，能从自然语言描述中生成数学模型和可执行代码，并通过内部反馈循环实现自我纠错。

**💡 创新点**

核心创新在于将建模过程拆分为六个专责代理并设计四个针对错误类型的反馈回路，实现透明的逐步推理与自动修正。

**🔧 技术方法**

利用大型语言模型（Claude Sonnet 4.5 / GPT‑5.4）配合LangGraph构建状态化的多代理管道，执行解释、建模、分析、验证、代码生成与求解。

**📊 数据集**

实验数据集为 ComplexOR、IndustryOR、LogiOR、NLP4LP 四个真实世界优化任务集，共 455 题。

**📈 对比分析**

与无系统提示的基线 LLM、OR‑specific 方法（ORLM、LLMOPT、ORThought、NEMO、CoE）对比，OptiAgent 在 3/4 数据集上取得最高准确率，特别是 LogiOR 与 IndustryOR，ComplexOR 完全正确。

**⚠️ 局限性**

局限包括仅做单轮实验未评估结果方差、缺乏完整消融研究以及解释性指标不足，需进一步验证与扩展。

---

## 722. Parallel $\mathcal O(\sqrt n)$ Overhead LSD Radix Sort

**arXiv ID:** 2607.05302 | [PDF](https://arxiv.org/pdf/2607.05302v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 723. Air Quality Downscaling with Station-Guided Pseudo-Supervision

**arXiv ID:** 2607.05292 | [PDF](https://arxiv.org/pdf/2607.05292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 724. Target-Guided Selective Reweighting for Physics-Informed Neural Network Inverse Problems: A Transfer Learning Approach

**arXiv ID:** 2607.05271 | [PDF](https://arxiv.org/pdf/2607.05271v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 725. How Far is Too Far? Defining the Distance Threshold for Verification Siamese Networks

**arXiv ID:** 2607.05329 | [PDF](https://arxiv.org/pdf/2607.05329v1)

**作者:** Heloísa Dias Viotto `[一作]` (Universidade Federal do Paraná), Paulo Lisboa de Almeida `[通讯]` (Universidade Federal do Paraná)

**通讯引用:** 555 | [OpenAlex ID](https://openalex.org/A5014773393)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无监督方法，通过拟合双峰分布来自动估计Siamese验证网络的距离阈值，从而实现无需标注数据即可进行验证。

**💡 创新点**

创新点在于：①利用双峰分布假设将阈值定位为两峰之间的极小值；②实现阈值的在线动态更新；③在部署环境下无需人工标注即可持续调整阈值。

**🔧 技术方法**

使用的技术包括：双峰高斯混合模型（GMM）拟合距离分布；Brent无导数优化寻找阈值；MobileNetV3-Large作为特征提取器；triplet loss和hard negative mining；L2归一化控制距离范围。

**📊 数据集**

实验数据集：MNIST、CIFAR-10、LFW、PKLot（三种不同配置），涵盖数字、图像分类、面部与车辆识别四类验证任务。

**📈 对比分析**

与传统的EER阈值和直接使用margin阈值比较，提出的方法在四个数据集上平均达到94%的准确率，基本与EER相当，同时显著优于margin阈值，且不需要任何标注数据。

**⚠️ 局限性**

局限性包括：当距离分布不严格满足双峰假设或出现极端类别不平衡时，阈值可能无法准确捕捉两峰，导致性能下降；此外，在线更新需要足够的最近样本窗口，极端分布漂移时仍可能失效。

---

## 726. ChatImage: Navigating Long-Form LLM Answers through Interactive Images

**arXiv ID:** 2607.05290 | [PDF](https://arxiv.org/pdf/2607.05290v1)

**作者:** Wencan Jiang `[一作]` (Zhejiang University), Yong Liu `[通讯]` (Zhejiang University)

**通讯引用:** 36233 | [OpenAlex ID](https://openalex.org/A5100712539)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将LLM长答案转换为可交互的视觉图像的系统，支持可点击热点、详细信息面板和区域级对话；

**💡 创新点**

核心创新是将内容结构化与视觉布局分离，先生成图像再通过视觉对齐（grounding）动态定位热点，而非依赖文本生成器的精确布局；

**🔧 技术方法**

使用文本生成模型（LLM）做答案生成、图像生成模型（Diffusion/ControlNet/GLIGEN等）、视觉 grounding模型（LocateAnything、MiMo‑Vision、SAM系列）以及可视化前端技术；

**📊 数据集**

构建了30题的多模式基准，覆盖信息图、地图和场景三种视觉模式，用于评估完整交互循环和热点对齐质量；

**📈 对比分析**

在完整生成运行中实现100%交互循环完成率，70.8%热点通过严格视觉对齐门槛，54.2%热点满足SAM掩码完整性检查，表明系统在多样场景下的可靠性；

**⚠️ 局限性**

局限性包括对视觉模型的高度依赖（密集信息图、小目标或复杂地图仍可能导致对齐失败）、热点默认为矩形而非多边形、以及缺乏对更丰富文档上下文的支持。

---

## 727. Adaptive Inference Batching using Policy Gradients

**arXiv ID:** 2607.05272 | [PDF](https://arxiv.org/pdf/2607.05272v1)

**作者:** Ruslan Sharifullin `[一作]` `[通讯]` (Stanford University), Ruslan Sharifullin (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

针对机器学习推理服务的批处理和路由问题，提出并实现了基于强化学习的自适应批量与请求分配策略。

**💡 创新点**

在多GPU环境下，利用Policy Gradient学习将快慢不同的请求进行分离，有效缓解Head‑of‑Line阻塞，实现3.5倍吞吐量提升。

**🔧 技术方法**

采用REINFORCE和PPO策略梯度算法，结合自定义离散事件模拟器和多头注意力的策略网络。

**📊 数据集**

使用合成Poisson流、极端突发流、Azure Functions以及BurstGPT等真实工作负载模拟数据进行训练与评估。

**📈 对比分析**

与静态批量、随机、轮询和最短队列等基线相比，RL在多GPU路由任务上获得约348%奖励提升，吞吐量提升60%，延迟下降25%。

**⚠️ 局限性**

假设执行时间确定、批量大小离散、中心调度器无网络延迟，未考虑GPU热衰减、系统噪声与真实分布式通信开销。

---

## 728. GeoFlow: Geo-Aware Modeling of Inter-Area Relationships in Origin-Destination Flow Prediction and Generation

**arXiv ID:** 2607.05257 | [PDF](https://arxiv.org/pdf/2607.05257v1)

**作者:** Zherui Huang `[一作]` (Shanghai Jiao Tong University), Linghe Kong `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9021 | [OpenAlex ID](https://openalex.org/A5072308822)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现 GeoFlow 框架，用于城市 OD 流量的预测和生成。

**💡 创新点**

创新点在于：①通过相对位置、k-hop 距离和测地距离等地理属性显式增强区域表示；②设计几何-内在融合编码器，将空间几何信息与区域内在属性联合编码；③引入轴向-全局注意力解码器高效捕获 OD 竞争关系；④在生成任务中采用流匹配模型，提升生成质量与多样性。

**🔧 技术方法**

主要技术包括：多层感知机（MLP）对地理属性编码；图注意力网络（GAT）聚合邻域信息；轴向注意力与全局注意力的结合实现高效的 OD 关系建模；连续时间流匹配框架用于生成；训练采用 MSE、分布相似度（JSD）和多样性指标。

**📊 数据集**

使用了三类公开 OD 数据集：CommutingODGen（通勤流）、Freight Analysis Framework（FAF，货运流）、New Zealand Tourism Volumes & Flows（旅游客流）。

**📈 对比分析**

与经典物理模型（Gravity）、传统机器学习回归器（RF、GBRT、SVR）、深度学习模型（DGM、GMEL、TransFlower）以及图生成模型（NetGAN、DiffODGen、WEDAN）对比。实验显示：在预测任务中 GeoFlow 在 CPC、NRMSE、JSD 上均优于所有基线；在生成任务中，在重构精度、分布匹配和多样性（Div.）上实现最佳平衡，显著优于 NetGAN、DiffODGen 和 WEDAN。

**⚠️ 局限性**

主要局限：依赖可靠的地理信息（网络连通性、测地距离等），若数据缺失或错误会影响性能；在分布漂移或极端场景下模型可能产生不可靠预测；模型的黑箱特性导致可解释性不足，无法完全解释物理或因果机制。

---

## 729. GelNeuro: A Sensing-Computing Integrated Neuromorphic Tactile System for Texture Recognition

**arXiv ID:** 2607.05241 | [PDF](https://arxiv.org/pdf/2607.05241v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 730. FUSE: FK-Steered Multi-Modal Flow Matching for Efficient Simulation-Based Posterior Estimation

**arXiv ID:** 2607.05252 | [PDF](https://arxiv.org/pdf/2607.05252v1)

**作者:** Weichen Qin `[一作]` (ShanghaiTech University), Jiakai Zhang `[通讯]` (ShanghaiTech University)

**通讯引用:** 4066 | [OpenAlex ID](https://openalex.org/A5101830881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了 FUSE 框架，用双轨架构与 Feynman‑Kac 指导的流匹配方法实现高效的仿真后验估计。

**💡 创新点**

创新点在于：① 双轨设计保留参数与观测的独立特征；② FK‑steered 采样在生成轨迹中动态利用观测似然；③ 结合 MMDiT 融合模块与流匹配训练提升多模态建模能力。

**🔧 技术方法**

核心技术包括多模态扩散变换器（MMDiT）、条件流匹配（Conditional Flow Matching）、Feynman‑Kac 指导的采样、双轨融合架构以及粒子重采样。

**📊 数据集**

使用了 SBIBM 10 任务的仿真数据集以及真实的 β Pictoris b 行星轨道观测数据。

**📈 对比分析**

通过与 NPE、FMPE、Simformer 等基准在 ℓ‑C2ST、MMD、KL、Sinkhorn 等指标上的对比，FUSE 在 SBIBM 任务中实现了更低的分布差异，并在 β Pictoris 任务中后验与 PTMCMC 更贴近，同时推理速度提升数十倍。

**⚠️ 局限性**

局限性包括：缺乏渐近精确性保证；对仿真数据量敏感，极少量调用时效果下降；尾部低概率区域捕获不足；尚未验证在更高维度或多行星系统上的可扩展性。

---

## 731. Privacy-Preserving Robustness Verification for Neural Networks

**arXiv ID:** 2607.05251 | [PDF](https://arxiv.org/pdf/2607.05251v1)

**作者:** Nianyun Song `[一作]` (Beijing Normal University), Xiyue Zhang `[通讯]` (University of Bristol)

**通讯引用:** 920 | [OpenAlex ID](https://openalex.org/A5100717733)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个基于安全两方计算的隐私保护神经网络鲁棒性验证框架。

**💡 创新点**

首次将分支条件转化为连续算术表达式消除分支，并结合Newton–Raphson反比精细化与函数秘密共享实现无分支ReLU，从而在保持隐私的前提下完成鲁棒性验证。

**🔧 技术方法**

使用安全两方计算、加法秘密共享、FSS（分布式比较）、Beaver三元组、固定点运算以及Newton–Raphson迭代等技术。

**📊 数据集**

在MNIST和CIFAR‑10数据集的全连接ReLU网络上进行实验验证。

**📈 对比分析**

与明文CROWN验证器比较，决策一致、平均误差极低；在线时延从0.1 s到200 s，通信量从0.77 MB到604 MB；离线预处理时间和存储从0.17 s/39 MB到71 s/31 GB不等。

**⚠️ 局限性**

需要大量离线预处理和存储，当前仅支持全连接ReLU网络，安全性仅在半诚实模型下，且固定点误差会随网络深度累积。

---

## 732. GUSH3R: Everyone Everywhere All at Once as Gaussians

**arXiv ID:** 2607.05243 | [PDF](https://arxiv.org/pdf/2607.05243v1)

**作者:** Keito Abe `[一作]` (University of Tokyo), Toshihiko Yamasaki `[通讯]` (University of Tokyo)

**通讯引用:** 6967 | [OpenAlex ID](https://openalex.org/A5048624196)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于 3D 高斯渲染的单目视频在线动态人景重建框架 GUSH3R，能够在一次前向推理中同时恢复静态场景和动态人物的几何与外观。

**💡 创新点**

创新点包括：①将人景基础模型 Human3R 的几何先验与 3DGS 表示无缝融合；②分别设计场景高斯解码器与人类高斯解码器，并通过跨注意力与记忆令牌实现人类外观在全局高斯空间中的一致性；③通过体素化与滤波实现多帧高斯的高效聚合，保持实时性能。

**🔧 技术方法**

核心技术包括：3D Gaussian Splatting、SMPL-X 人体模型、Human3R 基础模型、Dense Prediction Transformer（DPT）、Human Gaussian Transformer（HGT）、跨注意力机制、记忆 token、体素化聚合、光照可视化损失（MSE、LPIPS、深度约束等）。

**📊 数据集**

主要使用的公开数据集有：BEDLAM、EMDB、NeuMan、DL3DV、Motion-X++，其中前四个提供单目视频与人类动作，Motion-X++ 用于增强人体运动多样性。

**📈 对比分析**

与优化式基线 HSR、以及分解式前向基线（AnySplat、AnySplat+LHM+Human3R、AnySplat+LHM+GT）进行对比。实验表明 GUSH3R 在 PSNR/SSIM 与基线相当甚至更好，在 LPIPS 上更优，同时帧率提升至 1.5–1.7 FPS（相比 AnySplat 仅 0.16–0.42 FPS），实现了显著的效率提升。

**⚠️ 局限性**

局限性包括：①仍依赖 Human3R 的 SMPL-X 估计，误差会影响人类重建；②对极端遮挡或快速运动时的记忆 token 可能不足以保持一致外观；③体素化聚合在极大场景中可能产生细节丢失；④目前仅在单目视频上验证，缺乏多视角或深度传感器的泛化评估。

---

## 733. MoP-JEPA: Hard-Assigned Predictor Mixtures for Stochastic JEPA World Models

**arXiv ID:** 2607.05238 | [PDF](https://arxiv.org/pdf/2607.05238v1)

**作者:** Zhi Song `[一作]` (City University of Hong Kong), Jianhua Yao `[通讯]` (Tencent)

**通讯引用:** 21451 | [OpenAlex ID](https://openalex.org/A5100695406)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析并证明了传统JEPA世界模型在随机环境下会出现条件均值崩塌现象，提出硬分配混合预测器MoP‑JEPA，并通过一套验证协议证明其能够枚举真实未来并实现可验证规划。

**💡 创新点**

核心创新在于将预测器拆分为多头并通过硬分配（winner‑take‑all）实现模式枚举，同时设计了输入无关码本、随机置换、路由门控等验证步骤，确保枚举结果具有真实转移和可规划性。

**🔧 技术方法**

技术细节包括：JEPA潜在回归、混合专家（MoE）与硬分配、路由器门控、代码本对比、随机置换测试、转移精度门槛和验证路由判定；在同一编码器、相同离线数据和规划器下进行对比。

**📊 数据集**

使用OGBench离线数据集（pointmaze‑medium‑stitch、pointmaze‑large‑stitch、pointmaze‑teleport‑navigate、antmaze‑teleport），以及在实验中对ETH/UCY行人、SVHN、像素观测等数据的扩展验证。

**📈 对比分析**

通过 planAll、官方目标成功率、Verified RealRoute、覆盖率与精度等指标与同构基线（dense、M3‑JEPA、Var‑JEPA、MDN）对比，MoP‑JEPA在 planAll 最高可达 0.85，Verified RealRoute 最高 0.20，远超基线的 0.02–0.09，显示出 2–5 倍的提升。

**⚠️ 局限性**

局限性包括：对四路以上高分支时偶尔会漏检低概率模式，种子依赖导致结果波动；硬分配无法一次性选择随机分支；在高维或视频/机器人尺度下的可扩展性仍未验证；以及对极端稀有模式的捕捉能力有限。

---

## 734. An Investigation of the AUTOSAR Adaptive Platform from an Industry Perspective

**arXiv ID:** 2607.05227 | [PDF](https://arxiv.org/pdf/2607.05227v1)

**作者:** Bengt Haraldsson `[一作]` (TRATON R&D AB), Erika Mayer `[通讯]` (TRATON R&D AB)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了 AUTOSAR Adaptive 平台在工业实践中的痛点，并通过构建最小化 Adaptive 平台原型和工作坊形成了痛点分类法。

**💡 创新点**

创新点在于将设计科学研究方法与可控最小化平台结合，首次系统性分离并归纳标准、实现与本地实践导致的痛点来源与组织影响。

**🔧 技术方法**

采用了设计科学研究、工作坊收集定性数据、静态代码复杂度分析、两名开发者的实验评估以及最小化 Adaptive 平台原型实现。

**📊 数据集**

数据集来自 TRATON R&D AB 的 30 个团队工作坊笔记、开发日志、ARXML/JSON 配置文件以及代码行统计。

**📈 对比分析**

通过对比最小化平台与供应商实现的代码复杂度、构建时间和配置文件大小等指标，发现供应商实现更复杂、构建更慢、配置文件更大，而最小化平台则更简洁高效。

**⚠️ 局限性**

局限在于仅在单一工业伙伴环境验证、样本量有限、最小化平台并非完整实现，且主要关注定性体验，缺乏更广泛的量化性能评估。

---

## 735. Probing Geospatial SSL Representations with Environmental Signals

**arXiv ID:** 2607.05207 | [PDF](https://arxiv.org/pdf/2607.05207v1)

**作者:** Rohita Mocharla `[一作]` (Johns Hopkins Applied Physics Laboratory), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**通讯引用:** 23230 | [OpenAlex ID](https://openalex.org/A5004716468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于ERA5重分析物理变量的自监督学习表示评估框架，通过线性和非线性探测以及表示几何诊断指标，系统评估遥感自监督模型对环境信息的编码能力。

**💡 创新点**

创新点在于将ERA5物理变量作为无监督评估基准，并将统一性、有效秩等内在诊断与环境信号解码相结合，提供超越传统下游任务评估的新维度。

**🔧 技术方法**

使用了DINO、MAE、MoCo等自监督学习模型；通过岭回归和MLP探测器评估线性与非线性可解码性；引入对齐、统一性、有效秩、离散协方差等内在指标。

**📊 数据集**

数据集包括SSL4EO（Sentinel‑1/2）、ERA5重分析变量、PANGAEA下游任务和EarthShift鲁棒性基准。

**📈 对比分析**

在相同训练条件下比较三种SSL模型，DINO在ERA5探测上表现最佳；实验显示线性可解码性与农业和灾害任务的下游性能呈显著正相关，表明环境信号解码与任务表现相关。

**⚠️ 局限性**

局限性包括仅使用ERA5作为物理归一化依据、关联性分析而非因果推断、模型数量有限且未覆盖所有可能的架构与规模。

---

## 736. Topological Shape Representation for Aneurysm -- Bifurcation Detection

**arXiv ID:** 2607.05317 | [PDF](https://arxiv.org/pdf/2607.05317v1)

**作者:** Akshay Gokhale `[一作]` (Sardar Patel Institute of Technology), Mansi Dhamne `[通讯]` (Sardar Patel Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究并验证了一种基于拓扑学的后置滤波框架（Smooth Euler Characteristic Transform，SECT），用于在CTA图像中显著降低脑动脉瘤检测的假阳性率。

**💡 创新点**

创新点在于引入方向敏感的SECT，将3D血管几何的方向信息编码为可学习特征，突破传统卷积网络仅基于像素强度的局限，显著提升对小于3 mm瘤体的判别能力。

**🔧 技术方法**

使用Persistent Homology、Persistence Images、Persistence Landscapes、SECT等拓扑特征提取方法，并在随机森林与L1支持向量机上进行分类。

**📊 数据集**

采用RSNA 2025多机构CT血管造影（约1808份扫描）构建的严格分层patch集合，包含1201个正样本和多类别负样本（Frangi分叉、硬负样本、易负样本）。

**📈 对比分析**

在5折交叉验证中与PI、PL比较，SECT在随机森林下AUC为0.9433、在L1‑SVC下AUC为0.9312；在小于3 mm病灶上AUC为0.9434，95%特异率下敏感度达78.4%；留一扫描仪验证平均AUC为0.9273，远优于传统方法。

**⚠️ 局限性**

局限包括仅在patch级别评估，未完成端到端检测；对中大尺寸病灶样本不足；SECT特征提取耗时约11 s/patch；扫描仪间仍存在残余变异。

---

## 737. Video-based detection of cessation of breathing in pre-term infants using machine learning

**arXiv ID:** 2607.05230 | [PDF](https://arxiv.org/pdf/2607.05230v1)

**作者:** Dineo Serame `[一作]` (University of Oxford), Mauricio Villarroel `[通讯]` (University of Oxford)

**通讯引用:** 3335 | [OpenAlex ID](https://openalex.org/A5025327657)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在新生儿重症监护室（NICU）使用RGB摄像头提取呼吸相关信号，结合传统阻抗肺活量（IP）等生理信号，构建深度学习模型以检测早产儿呼吸暂停（COBE）事件。

**💡 创新点**

创新点在于首次将视频提取的局部呼吸信号（PPGi_rr）与IP等接触式生理信号进行多模态融合，显著提升了检测准确率，并验证了视频非接触监测在真实临床环境中的可行性。

**🔧 技术方法**

使用MediaPipe Pose实现动态ROI定位，提取帧差（FD）和像素强度呼吸信号，利用一维ResNet和ConvNeXt深度网络进行特征提取，并通过晚期融合（late fusion）实现多模态学习；同时应用基于交叉熵的加权损失和早停策略进行训练。

**📊 数据集**

数据集来源于英国约克郡John Radcliffe医院30名早产儿（最终23名可用），包含689个80秒标注段（246个COBE，443个正常呼吸），共计4,823个20秒窗口，原始采样率为20帧/秒、60Hz生理信号。

**📈 对比分析**

通过5折交叉验证与独立测试集比较，单摄像头模型最高Balanced Accuracy为76.9%；最优融合模型（PPGi_rr + IP）在测试集上达到90.6%的Balanced Accuracy，明显优于仅摄像头或单一生理信号模型，证明多模态融合提升了检测性能。

**⚠️ 局限性**

局限性包括样本量有限（仅23名婴儿），视频质量筛选导致有效段数减少；ROI定位使用成人姿态模型BlazePose，可能不完全适用于新生儿；未在多中心、不同照明或遮挡条件下验证泛化性。

---

## 738. Exact ratio preservation via outliers for fair $k$-center clustering

**arXiv ID:** 2607.05342 | [PDF](https://arxiv.org/pdf/2607.05342v1)

**作者:** Anna Arutyunova `[一作]` (Heinrich Heine University Düsseldorf), Melanie Schmidt `[通讯]` (Heinrich Heine University Düsseldorf)

**通讯引用:** 2462 | [OpenAlex ID](https://openalex.org/A5037403363)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种在公平性约束下的k‑中心聚类算法，允许通过设置比例和舍弃部分多数群体点来实现精确的群体比例。

**💡 创新点**

创新点在于将公平性与离群点（outliers）相结合，利用公平子集（fairlet）分解在保证比例的同时自适应地确定离群点数量，并给出了常数因子近似算法（1:t 情况下 4 倍，通用情况 14 倍）。

**🔧 技术方法**

主要技术包括构造基于最大流的公平子集分解、在公平子集锚点上执行最远点遍历得到中心、以及组合分析得到总成本为公平子集成本加中心聚类成本的结果。

**📊 数据集**

在实验中使用了 Bank、Census、Diabetes、Income 四个公开数据集，并对多属性保护特征（性别、婚姻、种族）进行分组，生成约 385 个子测试集。

**📈 对比分析**

与 Chierichetti 等人无离群点方法对比，本文算法在相同目标比例下获得更低的聚类成本（平均、最小、最大值均优），且在中心选择上仅使用少数群体点可进一步降低成本。

**⚠️ 局限性**

局限性包括：近似比率较高（最高 14 倍）、算法复杂度为 O(n²+log n)，对大规模数据集可能不够高效；仅在一侧或少数多侧离群点设定下提供结果，且对离群点数量的自动决定依赖于比例假设。

---

## 739. Shifting from Discrete to Continuous Reference Data: QSM-Derived Horizontal Tree Biomass Distribution for Deep Learning Biomass Estimation

**arXiv ID:** 2607.05260 | [PDF](https://arxiv.org/pdf/2607.05260v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 740. Someone slept in my bed! On the entailment problem for conjunctive queries with safe negation over DL-Lite$_{core}$ knowledge bases

**arXiv ID:** 2607.05336 | [PDF](https://arxiv.org/pdf/2607.05336v1)

**作者:** Jerzy Marcinkowski `[一作]` (University of Wrocław), Piotr Ostropolski-Nalewaja `[通讯]` (University of Wrocław)

**通讯引用:** 24 | [OpenAlex ID](https://openalex.org/A5061160726)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文通过构造知识库与布尔安全负向联结查询，证明在DL‑Lite‑core 语义下查询答案的可判定性是不可判定的。

**💡 创新点**

创新点在于首次展示即使仅允许安全负向（每个变量必须出现在正原子中），在此逻辑框架下查询答案仍然是不可判定的；并且该证明通过对半单词问题的复杂归约实现。

**🔧 技术方法**

主要技术包括：利用半单词问题（或 Thue 系统）的单词等价关系作为归约源，构造一系列带有“冻结”概念的知识库、正向与负向子查询，以及通过“too close”谓词限制映射的复杂结构。

**📊 数据集**

该工作并未使用真实数据集，而是完全基于理论构造的符号实例与逻辑结构进行证明。

**📈 对比分析**

由于研究对象是理论可判定性问题，未进行实验或性能比较；因此无法给出性能指标，只能说明该问题在理论上是不可判定的。

**⚠️ 局限性**

限制在于结果仅适用于 DL‑Lite‑core 并且仅涉及安全负向联结查询；对更强的描述逻辑、非安全负向或其它查询形式的可判定性仍未知。

---

## 741. Steering Optimisation Trajectories in Diffusion Representation Learning

**arXiv ID:** 2607.05319 | [PDF](https://arxiv.org/pdf/2607.05319v1)

**作者:** Rajat Rasal `[一作]` (Imperial College London), Ben Glocker `[通讯]` (Imperial College London)

**通讯引用:** 46526 | [OpenAlex ID](https://openalex.org/A5007222325)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了扩散自编码器在无监督表征学习中的优化动态，并提出了 SteeringDRL 方法以引导模型获得更优的解缠结构

**💡 创新点**

通过引入门控残差 U‑Net 限制跳过路径以及噪声级别学习课程，成功地将训练轨迹从“重建优先”迁移到“表征优先”，显著提升了解缠质量

**🔧 技术方法**

使用门控残差 U‑Net、跨注意力、FiLM、SlotAttention、噪声级别学习课程以及 VDM++ 变分扩散框架

**📊 数据集**

在 Shapes3D、Cars3D、MPI3D‑toy、ClevrTex 与 PascalVOC 等公开数据集上进行实验

**📈 对比分析**

与 EncDiff、DisDiff、FDAE、DyGA、MetaSlot、SlotDiffusion 等基线对比，SteeringDRL 在 DCI、FactorVAE、MIG、FG‑ARI、mIoU、mBO、LPIPS、FID 等指标上均达到或超过最优性能，并且种子方差明显下降

**⚠️ 局限性**

仍需针对不同模型容量和数据域进一步调优噪声课程；对极大规模模型的计算开销较高；缺乏对更多领域和更复杂因子场景的验证

---

## 742. PiSAs: Benchmarking Contextual Integrity in Multi-User Agentic Systems

**arXiv ID:** 2607.05318 | [PDF](https://arxiv.org/pdf/2607.05318v1)

**作者:** Shubham Gupta `[一作]` (ServiceNow AI Research), Valentina Zantedeschi `[通讯]` (ServiceNow AI Research)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了名为 Privacy in Shared Agentic Systems（PISA）的基准，用于评估多用户共享LLM代理系统中的隐私泄露，涵盖输出、代理间通信和共享记忆三种泄露表面。

**💡 创新点**

创新点包括：①双重CI注释（任务适当性与可视性）使得跨用户泄露可直接量化；②基准系统无关，支持任意代理拓扑和记忆配置；③系统化分析不同拓扑、记忆模式、隐私提示对隐私-效能权衡的影响。

**🔧 技术方法**

使用技术：多种LLM后端（开放式非推理模型、开放式推理模型、封闭式前沿模型），三种代理拓扑（单体、集中、分散），三种记忆配置（私有、共享、混合），隐私提示与规则显式化，LLM判定器进行适当性与可视性违规检测。

**📊 数据集**

数据集：85个手工构造的多用户场景，包含4~10名用户，平均每个场景23个属性（12个适当，10个不适当），任务涵盖 JIRA 任务分配、会议分配和严重性分类，每个场景都有明确的任务规则和隐藏真值。

**📈 对比分析**

比较方法：通过任务完整性、实用性、适当性泄露率（V_appr）和可视性泄露率（V_vis）进行评估；实验显示：单体代理在所有后端上适当性泄露率>77%；多体系统在中心化或分散架构下将泄露率降低20–50%，但错误率仍>75%；最高效能与最低泄露率的组合是使用 GPT‑4 的中心化系统，但其跨用户泄露率仍处于高水平。

**⚠️ 局限性**

局限性：①合成且受控的场景缺乏真实交互与外部工具调用；②仅评估无对抗环境下的非恶意泄露；③每个基准实例仅单任务，未考虑跨任务泄露；④基准可能导致过拟合；⑤未探究更深层次的隐私微调或安全防御手段。

---

## 743. Vision Pretraining for Dense Spatial Perception

**arXiv ID:** 2607.05247 | [PDF](https://arxiv.org/pdf/2607.05247v1)

**作者:** Zelin Fu `[一作]`, Nan Xue `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于边界的自监督预训练框架——Masked Boundary Modeling（MBM），通过在线生成边界字段、将边界标记的token强制加入掩码，并以离散化的类别分布监督，来学习具备高质量空间结构的视觉表示；

**💡 创新点**

创新点包括①将边界视为自监督信号而非下游输出；②动态识别并掩码边界token，将其与语义目标共同训练；③将连续边界字段转化为可分类的离散化表示，并在教师–学生循环中使用a‑contrario检验实现无监督验证；④在大规模数据上以ViT‑g（1B参数）实现高效稠密表征，随后可蒸馏至更小模型；

**🔧 技术方法**

使用的技术主要有Vision Transformer（ViT）与EMA教师自蒸馏、iBOT式稠密掩码学习、基于稠密距离场的边界字段表示、离散化分类标签、a‑contrario统计检验、Rotary Position Embedding、SwiGLU、KoLeo正则化以及大规模分布式训练；

**📊 数据集**

数据集方面：预训练使用约161M张从2B图像池中检索并去重的Curated数据（相当于DINOv2的LVD‑142M）；深度补全实验使用3M到150M张RGB‑D样本，涵盖Synthetic Blender、真实多摄像头捕获、公开数据（NYU‑Depth、KITTI、SUN‑RGBD等）；

**📈 对比分析**

对比方法包括DINOv2、DINOv3、V‑JEPA、SigLIP 2、InternVideo、PEcore、PEspatial等；在稠密任务上，LingBot‑Vision 1B模型在NYUv2线性回归RMSE 0.296（领先于7B DINOv3 0.309）和KITTI 2.552（仅次于7B DINOv3 2.346），在语义分割上与DINOv3‑H+相当并超越DINOv2；在视频物体分割上与DINOv3‑H+持平，优于所有小模型；在深度补全任务中，LingBot‑Depth 2.0在多种遮挡模式和真实传感器数据上均取得最优或竞争性表现；

**⚠️ 局限性**

局限性主要包括①相对传统基于语义对齐的模型，图像级分类性能略逊于DINOv3；②对大规模训练依赖高计算与存储资源；③边界字段的生成仍受corner‑point检测的质量影响；④在极低纹理或反射表面上的边界检测可能不够稳健。

---

## 744. A Multimodal Reasoning Typology for Grounding Chart-Image Coherence in Science Communication

**arXiv ID:** 2607.05222 | [PDF](https://arxiv.org/pdf/2607.05222v1)

**作者:** Avina Nakarmi `[一作]` (New Jersey Institute of Technology), Aritra Dasgupta `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 1043 | [OpenAlex ID](https://openalex.org/A5051193196)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对创伤性脑损伤（TBI）领域的论文进行深入分析，提出并验证了一种针对图表与图像对（chart‑image pair）交互的五级推理缺口（R1–R5）分类模型，并通过专家与非专家的对比研究展示了推理缺口与读者知识水平之间的关系。

**💡 创新点**

创新点在于：①首次将图表与图像的跨模态推理拆解为从最浅的“直接翻译”到最深的“情境框架”五个层级，系统化描述了读者在获取图表与图像共同主张时需做的认知工作；②将 Clark 的共同基础理论与可操作的分层指标结合，形成可评估推理难度的量化框架；③通过实验验证该框架能预测视觉语言模型（VLM）在不同层级下的性能下降，并揭示专家与非专家对推理缺口的认知差异。

**🔧 技术方法**

技术手段包括：基于专家引导的认知走查流程构建标签；使用 Vision‑Language Models（如 GPT‑5.2）生成图表‑图像对的描述并进行零/少量示例提示；采用双盲评估（专家评估与非专家评估）对描述的合理性进行判定；利用 Lundgard 等人提出的四层语义模型对描述的推理深度进行细化；统计分析层级之间的一致性与差异。

**📊 数据集**

数据集为 79 篇 TBI 领域论文，手工识别并抽取 104 对图表‑图像对，其中 57 对经过人工验证（32 为显式对，25 为隐式对），其余 47 对通过 LLM 辅助标注。该数据集覆盖从简单的可视化对齐到复杂的跨模态推理场景。

**📈 对比分析**

比较方法：对 32 对显式对齐图表‑图像对，使用 VLM 进行零样本和多轮提示实验，计算与专家标注的交叉一致率（从 34% 提升到 65.6%）。在 25 对隐式对齐对中，进行专家与非专家评估，统计两组在不同 R 级别下的同意率，发现从 R1 到 R5 同意率逐步下降（50%→57%→40%→29%→25%）。这些结果表明模型性能与推理缺口呈负相关，且专家知识对填补高层级缺口具有显著作用。

**⚠️ 局限性**

局限性包括：①仅在 TBI 领域进行构建与验证，可能不易直接推广到其他学科；②标签依赖专家判断，存在主观性；③数据量相对有限（104 对），尤其是高层级 R4–R5 的样本更少；④实验中使用的 VLM 版本与提示策略受限，无法覆盖所有可能的模型表现；⑤未对自动化标注流程的错误率进行全面评估，仍需进一步验证其可扩展性。

---

## 745. An event-driven framework for fly-inspired visual motion detection

**arXiv ID:** 2607.05205 | [PDF](https://arxiv.org/pdf/2607.05205v1)

**作者:** Qinbing Fu `[一作]` (Guangzhou University), Yuchao Tang `[通讯]` (Guangzhou University)

**通讯引用:** 652 | [OpenAlex ID](https://openalex.org/A5101421673)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了基于事件摄像头的、融合时间表面编码、苍蝇视网膜神经网络和底层注意机制的实时运动检测框架。

**💡 创新点**

创新点在于将异步事件流的时间表面表示与结构化的生物启发式神经网络相结合，并加入自适应底层注意模块，以显著提升低照度环境下的运动方向识别准确性和实时性。

**🔧 技术方法**

使用了时间表面编码、HATS（Histogram of Averaged Time Surfaces）、基于T4/T5的方向选择神经元网络、LPTC输出层以及自适应注意区域生成与运动引导算法。

**📊 数据集**

实验数据集为DAVIS346事件摄像头在日间与夜间条件下记录的四辆车在不同速度（1.21–2.42 rad/s）下的运动序列，包含同步的APS帧。

**📈 对比分析**

与基线帧级苍蝇神经网络和基于对比度最大化的事件驱动方法对比，提出框架在运动方向检测、速度敏感度和计算时延上均优于对照组，尤其在低光照下保持稳定响应。

**⚠️ 局限性**

局限性包括仅在单一事件摄像头平台和交通场景下验证，缺乏对不同场景和多目标运动的广泛评估，并且未使用学习或训练，可能限制对更复杂动态环境的适应性。

---

## 746. Fast counting and sampling for ferromagnetic two-spin systems

**arXiv ID:** 2607.05248 | [PDF](https://arxiv.org/pdf/2607.05248v1)

**作者:** Weiming Feng `[一作]` (University of Hong Kong), Yichun Yang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 55047 | [OpenAlex ID](https://openalex.org/A5100321581)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并分析了加权Ising模型的Gibbs分布，证明其正性与边缘下界，并构造了高效采样算法；

**💡 创新点**

首次将谱正则性与零自由性结合，利用闭包引理去除叶节点影响，从而实现对更一般参数的快速近似计数；

**🔧 技术方法**

使用了谱正则性、闭包引理、Grace–Szegő–Walsh共识定理、Asano收缩、Ruelle方法以及Annealing技术；

**📊 数据集**

无实际数据集，主要在理论实验与数值演示中验证；

**📈 对比分析**

与传统MCMC/调度采样等方法比较，所述采样在参数范围内实现了多项式时间的总变异误差估计，FPRAS的运行时间为O(Δ^C·n^2/ε^2·log^2(n/ε))；

**⚠️ 局限性**

局限在于仅适用于γ>β>1且外场λ<λ^⋆的范围，且需去除叶节点的影响，无法直接推广到更大参数空间或非平面图。

---

## 747. CenSynCMB: Centre Maps and Physics-Guided Synthesis for Microbleed Detection

**arXiv ID:** 2607.05325 | [PDF](https://arxiv.org/pdf/2607.05325v1)

**作者:** Lucas He `[一作]` (University College London), Carole H. Sudre `[通讯]` (University College London)

**通讯引用:** 21553 | [OpenAlex ID](https://openalex.org/A5044422433)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `14d48e9d-0069-4ad9-996a-1d5968216998` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出了CenSynCMB框架，用于在多对比MRI中自动检测脑微出血（CMB），并通过中心引导的损失与物理引导的合成数据实现更精准的候选提取。

**💡 创新点**

创新点在于（1）引入中心地图监督与基于误检驱动的重加权，使训练目标与最终的质心匹配评估更一致；（2）在交叉验证折内使用物理模型（k空间 dipole kernel）生成正样本和标注的硬负样本（血管、钙化），在不泄漏的情况下显著提升对难以区分的假阳性结构的抑制。

**🔧 技术方法**

采用3D Attention U‑Net作为骨干网络，配合Tversky分割损失、焦点中心回归损失和同质不确定性权重；训练过程中采用False‑Negative驱动的裁剪重加权；合成模块利用静态退相位近似（dipole kernel）在原始图像上渲染CMB和硬负样本。

**📊 数据集**

使用VALDO Task 2（72例，含T2、T2*、T1）进行训练、验证和内部测试；将AIBL SWI（370例）作为外部无训练的SWI测试集，以评估GRE→SWI的跨序列泛化。

**📈 对比分析**

与多种基线（DynUNet、Attention U‑Net、SwinUNETR、VISTA3D）以及VALDO原方法、最新单阶段检测器（Al‑Masni、Kim、FRST）和公开预训练模型进行对比；在VALDO上，CenSynCMB实现最高局部比较F1 = 74.3 %（显著优于最佳基线70.8 %），在AIBL上实现最高召回88.5 %和F1 = 65.0 %，但FP/subject略升高。

**⚠️ 局限性**

局限性包括：仅检测CMB，未覆盖表面铁疤等ARAI‑H相关病灶；缺乏针对不同人群的阈值校准和计数校正，导致患者级负荷估计仍不够稳健；合成的物理模型为静态退相位近似，未考虑相位/QSM信息，可能限制对钙化与血管的区分；整体模型对扫描参数的鲁棒性仍需进一步验证。

---

## 748. How Much is Left? LLMs Linearly Encode Their Remaining Output Length

**arXiv ID:** 2607.05316 | [PDF](https://arxiv.org/pdf/2607.05316v1)

**作者:** Mohamed Amine Merzouk `[一作]` (Mila, Quebec AI Institute), Adam Oberman `[通讯]` (McGill University)

**通讯引用:** 2661 | [OpenAlex ID](https://openalex.org/A5004824330)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型内部是否能线性解码剩余生成长度，并证明其存在计划式表示

**💡 创新点**

发现并量化了LLM内部隐藏层中可线性读取的剩余长度信息，展示其在提示阶段即已编码，并在生成过程中可更新

**🔧 技术方法**

使用线性探测器（probe）对冻结的LLM隐藏状态进行回归、常数中位数基线和精确倒计时基线比较

**📊 数据集**

评估三款7–8B开放权重模型（Llama‑3.1‑8B、Olmo‑3‑7B、Mistral‑7B）在七个完成式数据集：两套合成（Count、Countdown）和五套自然语言（GSM8K、MATH、MMLU‑Pro、OpenThoughts‑1k、TriviaQA）

**📈 对比分析**

相较于中位数基线，提示终止长度探测器MAE下降至约1/3；相对精确倒计时，探测器在大部分数据集上优于或相当，表明中间隐藏状态携带超出提示长度的额外信息，且在某些重构案例中可出现向上跳变

**⚠️ 局限性**

探测器只能检索可线性可解码信息，未证明模型在生成时实际使用该方向；动态重估仅基于少数极差样本的定性观察，缺乏统计验证；研究仅限于7–8B指令调优模型，未探究更大规模或非指令模型，且仅评估自然终止序列

---

## 749. Evaluating and Understanding Model Editing for Medical Vision Language Models

**arXiv ID:** 2607.05310 | [PDF](https://arxiv.org/pdf/2607.05310v1)

**作者:** Guli Zhu `[一作]` (University of Michigan), Liyue Shen `[通讯]` (University of Michigan)

**通讯引用:** 6578 | [OpenAlex ID](https://openalex.org/A5072483985)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为M3Bench的医学视觉语言模型（VLM）编辑评测基准，用以在临床部署后评估模型编辑的可靠性、局部性、泛化性及时间一致性，并在此基准上系统评测了四类编辑方法。

**💡 创新点**

创新点包括①在医学多模态环境下从四个临床维度（文本、图像、模态/协议、临床组合）构建全新评测任务；②引入时间一致性评测；③通过几何分析揭示VLM表征“锥形效应”对编辑局部性与泛化的影响；④对编辑方法的局部性-泛化权衡进行深入解析。

**🔧 技术方法**

使用了梯度基方法（MEND、LoRA）与记忆基方法（GRACE、BalancEdit）以及其混合设计BELoRA等编辑技术，并在6个VLM（LLaVA‑Med、HuatuoGPT‑Vision 7B/34B、BioMed‑Qwen、Qwen3.5‑2B、Janus‑Pro‑7B）上进行实验。

**📊 数据集**

构建了包含16,276道问答的临床评测集，来源于VQA‑RAD、PMC‑VQA、PadChest‑GR、SLAKE等公开医学VQA与图像‑文本数据集，利用LLM提炼统一属性以实现跨数据集的任务一致性。

**📈 对比分析**

实验结果显示：梯度基编辑（LoRA）在可靠性与泛化上表现突出，但局部性显著下降；记忆基编辑（BalancEdit）局部性最佳，泛化与临时一致性不足；两类方法在不同维度存在明显权衡，且无单一方法在所有任务上占优。

**⚠️ 局限性**

局限性包括：①记忆基编辑对超参（关键点半径）的高度敏感，难以跨模型迁移；②在临床组合与时间一致性任务上性能仍偏低；③评测主要集中在问答场景，未覆盖生成式任务；④未探索更深层次的多模态组件编辑或多任务联合编辑。

---

## 750. Biologically Informed Deep Neural Networks for Multi-Omic Integration, Pathway Activity Inference and Risk Stratification in Cancer

**arXiv ID:** 2607.05306 | [PDF](https://arxiv.org/pdf/2607.05306v1)

**作者:** Pedro Henrique da Costa Avelar `[一作]` (King's College London), Sophia Tsoka `[通讯]` (King's College London)

**通讯引用:** 6363 | [OpenAlex ID](https://openalex.org/A5076507299)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一种基于通路信息的自编码器（PAAE），将多组学数据进行集成，并通过通路活动得分实现可解释性；

**💡 创新点**

创新点在于将生物通路知识嵌入深度网络架构，实现可解释的通路活性表示，并在多组学集成、重复性评估和层级贡献分析中提供系统方法；

**🔧 技术方法**

主要技术包括可解释神经网络（BINN）、变分自编码器、通路级编码器、dropout正则化、CKA相似度、Shapley值贡献分析、早期与晚期多组学集成；

**📊 数据集**

使用TCGA-BRCA乳腺癌多组学数据（基因表达、蛋白表达、miRNA、甲基化、突变、CNV）以及外部验证集Metabric；

**📈 对比分析**

通过与单组学模型比较，发现多组学集成（尤其是晚期均值/拼接）在PAM50子类型分类和生存预测上提升了ROC‑AUC和C‑Index；同时，dropout在提高模型重复性方面有效，但过高会降低预测性能；

**⚠️ 局限性**

局限性包括深度网络非凸优化导致模型可重复性受限、对小样本深度学习的泛化能力不足、dropout对不同层的影响未完全解释、以及对临床实际应用的验证仍待进一步研究。

---

## 751. Learning Only What Valid Adapters Can Express: Subspace-Constrained Adaptation Against Fine-Tuning Poisoning

**arXiv ID:** 2607.05300 | [PDF](https://arxiv.org/pdf/2607.05300v1)

**作者:** Fabien Polly `[一作]` `[通讯]`, Fabien Polly

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究将模型微调限定在由可信适配器构成的低维子空间内，以提升对毒化、标签反转和后门攻击的鲁棒性。

**💡 创新点**

创新点在于利用公共 LoRA 适配器的共享子空间削弱攻击可达性，并发现该子空间本身可作为异常检测信号，构建一种“几何约束式防御”。

**🔧 技术方法**

技术包括：计算适配器的 Gram 矩阵并做特征分解得到子空间基；在该基上用线性编码 z 进行梯度优化；对比传统 LoRA、随机子空间、强正则化等对照实验；构建 OOD 检测阈值；对抗自适应后门攻击。

**📊 数据集**

使用 flan-t5-large 预训练模型、P3 任务集（wiki_qa、qasc、amazon_polarity、social_iqa、race）以及 196 个公开 LoRAHub 适配器作为子空间构建与评估。

**📈 对比分析**

与同等训练步数、样本数的完整 LoRA 进行比较；在干净数据上保持相近准确率；在 100% 标签反转攻击下，子空间方法约提升 10 倍鲁棒性；误报检测 AUROC 接近 1；自适应后门攻击成功率从 100% 降至 8%–85%（取决于目标是否已在子空间中）。

**⚠️ 局限性**

局限包括：需要可信的适配器池；子空间并非凸包，仍可能出现安全违规组合；对池覆盖度低的任务性能下降；若攻击目标与池中行为相符，后门仍可部分成功；缺乏形式化安全保证。

---

## 752. Advances in Neural Controlled Differential Equations

**arXiv ID:** 2607.05280 | [PDF](https://arxiv.org/pdf/2607.05280v1)

**作者:** Benjamin Walker `[一作]` `[通讯]`, Benjamin Walker

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过改进神经受控微分方程（NCDEs）模型，提出Log-NCDE、Linear NCDE和Structured Linear NCDE（SLiCEs）三种新模型，使NCDE的训练与推理速度提升至三阶数量级，并在多种时间序列基准上取得最先进性能。

**💡 创新点**

创新点在于（1）使用Log-ODE方法对NCDE进行深度截断逼近，显著降低训练成本；（2）将非线性向量场线性化，得到Linear NCDE，消除非线性带来的数值难题；（3）进一步引入结构化线性网络（SLiCEs），实现参数压缩与并行化。

**🔧 技术方法**

核心技术包括受控微分方程（CDE）理论、signature 与 log-signature 的计算、Lip(γ)函数与Lie括号的正则性分析、以及基于神经网络的向量场 Lipschitz 约束与层级正则化。

**📊 数据集**

实验使用了 PhysioNet Challenge 2022 的心脉音（PCG）数据集以及其他公开时间序列基准（如 ECG、金融、气象等典型任务）。

**📈 对比分析**

与传统 RNN、Transformer、ResNet 等方法对比，Log-NCDE、Linear NCDE 与 SLiCE 在相同任务上均达到或超过最优性能，同时推理速度提升了约 1000 倍。

**⚠️ 局限性**

局限性主要在于：对非线性向量场需要满足 γ>1 的 Lipschitz 条件；Log-ODE 的收敛深度受 γ 限制；极端噪声或高维无界输入时，Lip(γ) 复合定理与扩展定理仍有不确定性。

---

## 753. Untrusted Content Masking for Web Agents with Security Guarantees

**arXiv ID:** 2607.05277 | [PDF](https://arxiv.org/pdf/2607.05277v1)

**作者:** Kristina Nikolić `[一作]` (ETH Zurich), Florian Tramèr `[通讯]` (ETH Zurich)

**通讯引用:** 13307 | [OpenAlex ID](https://openalex.org/A5006851333)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在网页中掩盖不可信内容的防御框架UCM，确保大语言模型代理在浏览网页时仅访问可信区域，并通过隔离的Quarantined Model安全查询不可信区域的结构化信息

**💡 创新点**

创新点在于利用DOM结构天然的可信/不可信划分，先行用占位符屏蔽不可信文本/图像；随后仅通过类型受限的查询（bool、int、float等）让模型获取必要信息，从而在保持功能性的同时提供正式的控制流安全保证

**🔧 技术方法**

技术包括DOM结构解析与标签化、占位符化渲染、Sandboxed交互接口、类型约束的Quarantined Model、LLM驱动的自动边界推断（通过内容脱敏DOM进行CSS选择器生成）

**📊 数据集**

使用了10个自定义网站环境（银行、日历、客服、电子商务、邮件、论坛、点餐、Wiki、旅行预订、招聘），以及WebArena GitLab benchmark（41个任务模板）进行评估

**📈 对比分析**

与无防御代理（全页面可见）对比，UCM在两类任务（不需要读取不可信内容 vs 需要读取）中保持任务完成率几乎不下降，成本提升约1.05×-1.84×；在真实网站测试中亦保持高效，且对强化WASP攻击实现0%攻击成功率

**⚠️ 局限性**

局限性包括：仍需对每个网页提供可信/不可信标签（除非自动推断），对类型受限查询的覆盖不完全（需额外用户批准以返回字符串），数据流攻击（如单值篡改或聚合偏移）仍可能产生误导；对XSS等运行时内容篡改未涵盖

---

## 754. Is the Geometry Doing the Work? An Operating-Point Audit of Hierarchy in Hyperbolic Vision-Language Models

**arXiv ID:** 2607.05268 | [PDF](https://arxiv.org/pdf/2607.05268v1)

**作者:** Jaeyoung Kim `[一作]`, Dongsuk Jang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对公开的 MERU、HyCoCLIP 与 PHyCLIP 三类超曲面视觉‑语言模型进行几何与层级机制的系统审计，开发必要条件诊断框架并揭示其不激活超曲层级机制的根本原因。

**💡 创新点**

提出一套基于维度无关半径 √(c)ρ、局部失真 H(u) 与锥 aperture 的超曲层级诊断工具，并通过梯度分解阐明“低曲率宽锥”捷径导致曲率坍塌和锥失效，最终给出可公开验证的五数几何报告标准。

**🔧 技术方法**

利用超曲空间几何理论计算 √(c)ρ、H(u)、锥 aperture；实现径向排序、层级遍历、梯度分解等诊断；将这些指标与下游检索、分类、零样本任务等性能指标关联。

**📊 数据集**

在当前 GRIT 快照、CIFAR‑100、ImageNet、COCO、Flickr30k 等公开数据集上训练并评估模型。

**📈 对比分析**

通过与已发布检查点对比、对曲率阈值放松、梯度分解等干预实验，发现曲率坍塌不影响检索、分类等下游指标，超曲模型与欧氏 CLIP 的性能差异不显著。

**⚠️ 局限性**

评估仅涵盖 MERU、HyCoCLIP、PHyCLIP 三族；诊断聚焦径向/锥机制，可能忽略其它层级实现；梯度分析受训练配置与数据可用性的限制。

---

## 755. FlowMark: Mask-Guided Video Watermarking

**arXiv ID:** 2607.05261 | [PDF](https://arxiv.org/pdf/2607.05261v1)

**作者:** Vishal Asnani `[一作]` (Adobe Research), John Collomosse `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了FlowMark，一种基于自学习掩码的时序水印嵌入框架；

**💡 创新点**

创新点在于引入了自动掩码预测网络实现内容自适应嵌入区域、暗区自适应残差调制与时间一致性约束，并通过三阶段课程训练提升在压缩、几何变形和社交媒体分发链路中的鲁棒性；

**🔧 技术方法**

使用U-Net风格的编码器/解码器、直通估计器进行掩码二值化、JND与亮度调制、GAN对抗损失、时间一致性MSE、总变差正则化，以及视频压缩与几何变换的可微模拟；

**📊 数据集**

在Adobe Stock视频数据集上训练，使用SA‑1B图像集和SA‑V视频集进行评估；

**📈 对比分析**

与HiDDeN、MBRS、CIN、TrustMark、WAM、MaskWM、VideoSeal等基线对比，FlowMark在128‑bit信息下实现了100%位准度、PSNR>49dB、SSIM>0.99、LPIPS≈0.002、VMAF≈98.95，并在压缩、几何、时间编辑以及YouTube/Facebook重编码场景中保持最高鲁棒性；

**⚠️ 局限性**

限制在于目前仅在256×256分辨率下训练、可扩展性受限于掩码预测网络的容量，对极端几何扰动与长时序编辑的表现尚未充分验证，且信息容量受128‑bit左右的平衡限制。

---

## 756. Streaming Neural Speech Codecs through Time-Invariant Representations

**arXiv ID:** 2607.05250 | [PDF](https://arxiv.org/pdf/2607.05250v1)

**作者:** Kélian Estève `[一作]` (Avignon Université), Yannick Estève `[通讯]` (Avignon Université)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了TiCodec时间不变表示的特性，提出多层Dual‑TIRE架构，并在离线与流式推理下评估其性能。

**💡 创新点**

通过探针分析揭示TIRE主要捕获全局声学与情感信息；提出跨层Dual‑TIRE利用不同层信息互补；使用跨文件采样提升鲁棒性；证明660 ms块流式推理几乎不损失质量。

**🔧 技术方法**

使用神经语音编码器+RVQ、TIRE模块、多层提取Dual‑TIRE、探针分类任务、跨文件采样策略以及块级在线解码技术。

**📊 数据集**

LibriTTS、VCTK、EMILIA（多语种）、VoxCeleb1、TAU Urban Acoustic Scenes、MELD、Common Voice、Google Speech Commands等数据集。

**📈 对比分析**

采用ViSQOL、PESQ、STOI、MCD、Sim、MOS、SI‑SDR、WSNR等指标与基线(No‑TIRE)、单TIRE、Dual‑TIRE对比；Dual‑TIRE在ViSQOL、Sim、MCD等方面优于单TIRE，PESQ略低；流式模式MOS≈0.618与离线相近。

**⚠️ 局限性**

仍然存在说话人信息保留不充分、PESQ下降、离散码序列长导致推理成本高、未完全验证跨语言或多模态生成的效果。

---

## 757. Optimizing ML Workload Partitioning between CPUs and CIM Accelerators for Heterogeneous Computing

**arXiv ID:** 2607.05240 | [PDF](https://arxiv.org/pdf/2607.05240v1)

**作者:** Joel Klein `[一作]` (RWTH Aachen University), Rainer Leupers `[通讯]` (RWTH Aachen University)

**通讯引用:** 6986 | [OpenAlex ID](https://openalex.org/A5023470562)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

设计了一种基于整数线性规划（ILP）的工作负载划分框架，用来在CPU与RRAM‑CIM加速器之间进行静态算子分配，最小化端到端推理延迟。

**💡 创新点**

创新点包括：1）混合时延特征化方法，结合CPU的实测时延和CIM/总线的解析模型；2）完整考虑内存预算、算子支持与并行性约束的ILP模型；3）利用系统级DSE洞察未来CIM加速器设计。

**🔧 技术方法**

采用了ONNX Runtime进行CPU时延采样、PUMA架构的RRAM跨接板分析模型、共享总线传输模型，以及Gurobi求解器完成ILP求解。

**📊 数据集**

使用的模型有ResNet‑18、ResNet‑50、MobileNetV2和YOLOv5n，全部以8‑bit整数量化后进行实验。

**📈 对比分析**

与单纯CPU推理（ARM Cortex‑A72、AMD Ryzen 9 3900X）对比，框架在ARM上可获得最高30.9×的加速，在x86上最高7.3×；加速主要来自将计算密集型卷积层迁移到CIM上。

**⚠️ 局限性**

局限性：1）ILP在高度分支的网络（如YOLOv5n）下求解时间长、MIP Gap高，需引入启发式或混合求解策略；2）假设权重一次性编程且不动态重写，限制了对可变网络的适用性；3）对RRAM写入延迟和耐久性的考虑不够细粒度。

---

## 758. Progressive Refinement: An Iterative Pseudo-Labeling Approach for Mandarin-English Code-Switching ASR

**arXiv ID:** 2607.05224 | [PDF](https://arxiv.org/pdf/2607.05224v1)

**作者:** Qu Yang `[一作]` (Apple), Tim Ng `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对英语-汉语混合语音进行自动语音识别（ASR），通过迭代伪标签训练显著提升识别性能。

**💡 创新点**

首次将迭代伪标签方法应用于代码切换ASR；采用双阶段双语模型训练并在每轮迭代中使用上一轮模型生成更精确的伪标签，实现持续自我提升。

**🔧 技术方法**

使用CTC+Attention（Conformer编码器+Transformer解码器）架构；两阶段训练（预训练+微调）以及伪标签生成、迭代改进算法；自定义采样权重调节不同数据集比例。

**📊 数据集**

SEAME（英语-汉语代码切换）公开语料；NSC、私有单语数据（English、Mandarin、Singaporean English）以及超过 100k 小时的伪标签语料。

**📈 对比分析**

与传统单语训练基线及私有单语模型比较；在 SEAME devman 上 MER 由 61.09% 降至 12.88%，devsge 上从 54.12% 降至 18.89%，均优于基线（19.23%/27.18%）。在单语评估 enSGeval 上，误差率从 13.80% 降至 12.86%，保持了单语性能。

**⚠️ 局限性**

迭代训练对伪标签质量高度依赖，若伪标签错误积累可能导致性能下降；需要大量高质量标注数据和计算资源，且方法在不同语言组合、语种比例上可能需要重新调参。

---

## 759. Weak-to-Strong Generalization via Direct On-Policy Distillation

**arXiv ID:** 2607.05394 | [PDF](https://arxiv.org/pdf/2607.05394v1)

**作者:** Shiyuan Feng `[一作]` (SIA-Lab of Tsinghua AIR and ByteDance Seed), Hao Zhou `[通讯]` (SIA-Lab of Tsinghua AIR and ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

提出了一种直接的基于策略差异的对抗式迁移方法（Direct‑OPD），利用弱模型在强化学习中获得的策略变迁作为隐式奖励，直接对更大模型进行微调，从而提升其推理性能。

**💡 创新点**

创新点在于：①不再复制弱模型的最终策略，而是提取弱模型相对其未训练状态的对数概率差（policy shift）作为奖励；②将该奖励应用于学生模型在自身采样状态上的 top‑k 训练；③设计了可自适应的 KL 正则控制机制，平衡奖励尺度与采样分布的可靠性。

**🔧 技术方法**

使用的技术包括：
- 对策变迁（policy shift）计算：Δ_T(y|x)=logπ_T(y|x)-logπ_T_ref(y|x)。
- 上采样 top‑k 对齐的对数概率差作为密集奖励。
- Rao‑Blackwell 化的上采样梯度估计。
- 采用自适应 KL 超参数调节策略。
- 在多模型多数据集上进行实验（AIME 2024/2025）。

**📊 数据集**

主要使用的训练与评估数据集：
- RL 训练采用 DAPO 数据集；
- 评估采用 AIME 2024 与 AIME 2025 的推理任务。
- 还在 Qwen3 与 R1‑Distill 系列模型上进行交叉实验。

**📈 对比分析**

比较方法：
- 与直接在大模型上进行 RL（RL‑Direct）对比；
- 与标准 OPD（仅模仿教师最终策略）对比；
- 在相同 RL 步数或计算成本下，测量 AIME 成绩。
- 结果显示：
  * Qwen3‑1.7B 在 AIME 2024 上从 48.3 提升至 62.4（+14.1 点）。
  * R1‑Distill‑7B 从 56.7 提升至 63.1（+6.4 点）。
  * 在匹配 RL 步数时，弱→强迁移比直接 RL 计算更高效（4 小时 + 8 A100 相比 1 周 + 32 A100）。

**⚠️ 局限性**

局限性：
- 信号高度依赖教师–学生的相对状态分布；若教师改进在学生采样状态中无效，则迁移效果差。 
- 最佳响应长度与 KL 强度需针对每一对教师/学生手动调节，缺乏通用自适应机制。 
- 仅适用于有可用的弱模型 RL 检查点，无法直接应用于无 RL 历史的情况。

---

## 760. InFlux++: Real and Synthetic Data for Estimating Dynamic Camera Intrinsics

**arXiv ID:** 2607.05389 | [PDF](https://arxiv.org/pdf/2607.05389v1)

**作者:** Erich Liang `[一作]` (Princeton University), Jia Deng `[通讯]` (Princeton University)

**通讯引用:** 128445 | [OpenAlex ID](https://openalex.org/A5101542158)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 InFlux++，包含大规模合成视频数据集 InFlux++ Synth 和扩展的真实视频基准 InFlux++ Real，用于动态相机内参预测。

**💡 创新点**

创新点在于通过程序化生成的合成视频实现每帧内参随 LFL/LTO 变化，并将 InFlux 原有基准扩展到更多场景与摄像机运动，从而显著提高训练与评估多样性。

**🔧 技术方法**

利用 Infinigen 进行场景生成与 Blender 渲染，采用 Brown–Conrady 模型加入畸变，使用有界随机步法调节 LFL/LTO，构建 LUT 进行真实相机标定，并在 InFlux++ Synth 上对 AnyCalib 进行微调。

**📊 数据集**

使用的数据集包括 1841 条 1841K 帧的 InFlux++ Synth（包含 441K+ 帧，部分附带位姿、深度和法线）以及 334 条 514K+ 帧的 InFlux++ Real；同时与原始 InFlux 数据集做对比。

**📈 对比分析**

对七种基线方法（AnyCalib、GeoCalib、UniDepthV2 等）采用 recall@错误阈值和 LUT‑可靠 EPE recall 评估，AnyCalib 在所有指标中表现最好，但最高的 10% F_x recall 仅约 25%，微调后 F_x/F_y 提升明显但 c_x/c_y 与 EPE 仍低；整体预测仍具有挑战性。

**⚠️ 局限性**

主要局限在于预测精度仍偏低，尤其是主点位置和畸变参数，合成监督与真实数据之间存在差距，且当前方法对极端畸变或边缘点的 EPE 处理不够理想。

---

## 761. MV-Forcing: Long Multi-View Video Generation via 4D-Grounded Spatio-Temporal Self-Forcing

**arXiv ID:** 2607.05376 | [PDF](https://arxiv.org/pdf/2607.05376v1)

**作者:** Gal Fiebelman `[一作]` (Hebrew University of Jerusalem), Sagie Benaim `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5081028371)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 MV-Forcing 框架，实现长时序多视角视频生成，能够在任意视角数和时间长度下保持几何一致性。

**💡 创新点**

创新点：将时序和视角自回归结合，使用 4D 重建模型 CUT3R 作为连续几何桥梁；引入时空自强（Spatio-Temporal Self-Forcing）与联合去噪训练，消除训练-推理曝光偏差；通过分布匹配蒸馏（DMD）将双向教师压缩为单向学生，实现可流式少步推理。

**🔧 技术方法**

技术：视频扩散模型（DiT）、双向视角同步模块（MVS）、自回归去噪、CUT3R 动态 3D 重建、ControlNet 风格几何条件、DMD 蒸馏、Self-Forcing、联合去噪。

**📊 数据集**

数据集：合成 SynCamVideo（3,400 条多视角视频）用于训练与评估；真实场景下使用 Open‑Sora Mixkit 子集进行微调。

**📈 对比分析**

对比方法：SynCamMaster、Self‑Forcing + ReCamMaster、Self‑Forcing + ReCamMaster + Self‑Forcing（以及其在 SynCamVideo 上微调版）。在视觉质量、相机精度、视角同步等指标上，MV‑Forcing 在短序列（2 视角 81 帧）和长序列（3 视角 162 帧/5 视角 648 帧）均优于所有基线；保持了跨视角一致性并在时间上几乎不降级。

**⚠️ 局限性**

局限性：主要在合成数据上训练，真实多视角数据规模有限；尽管 Self‑Forcing 减轻曝光偏差，但长时序仍存在质量衰退；教师模型仅生成 2 视角，未直接监督多视角一致性；在极端相机位移或高运动场景下可能出现几何漂移。

---

## 762. SPEARBench: A Benchmark for Naturalness Evaluation in Streaming Speech-to-Speech Language Models

**arXiv ID:** 2607.05365 | [PDF](https://arxiv.org/pdf/2607.05365v1)

**作者:** Thomas Thebaud `[一作]` (Johns Hopkins University), Laureano Moro-Velazquez `[通讯]` (Johns Hopkins University)

**通讯引用:** 1574 | [OpenAlex ID](https://openalex.org/A5069488212)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了SPEARBench基准，用于评估语音到语音语言模型在对话中的自然性，包括回答时延、重叠、语音质量、情感、语调、方言一致性等多维度。

**💡 创新点**

创新点在于整合多维自然性指标到一个统一的自动化评估流程，并公开平台和数据，填补了仅关注单一维度或需人工评估的空白。

**🔧 技术方法**

采用语音识别、音质评估模型（UTMOS）、声学模型（Silero VAD）、双人对话预测器、语言与方言识别、情感识别和立场评估等现有模型，构建完整评估管道。

**📊 数据集**

使用的是 Seamless Interaction 数据集的开发集与测试集，从中提取问答两轮对话作为评估样本。

**📈 对比分析**

与多种流式及全双工 S2S 模型（如 Qwen3‑Omni、GPT‑realtime‑2、Gemini flash 等）进行对比，结果显示模型在语音清晰度、WER 低等指标上优于人类，但在时延、重叠、情感契合、方言适配和语调多样性等自然性维度明显落后。

**⚠️ 局限性**

局限性包括评估仅自动化、仅覆盖英文两人对话、时延测量受模型推理接口影响、以及可用语音声音对语调与方言多样性的限制。

---

## 763. REDDIT: Correcting Model-Generated Timestamp Drift in ASR without Forgetting via Replay-Based Distribution Editing

**arXiv ID:** 2607.05364 | [PDF](https://arxiv.org/pdf/2607.05364v1)

**作者:** Cheng-Kang Chou `[一作]` (National Taiwan University), Hung-yi Lee `[通讯]` (NTU Artificial Intelligence Center of Research Excellence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了非语音导致模型生成时间戳漂移的问题，并提出一种两阶段后训练框架REDDIT，能够在不丢失原有识别性能的前提下校正时间戳。

**💡 创新点**

创新点在于将时间戳校正视为分布编辑任务：利用缓存的模型回放上下文对时间戳位置进行交叉熵编辑，同时用KL散度保持非时间戳位置与原始分布一致；并通过短前缀微调进一步巩固校正效果；整个过程无需人工时间戳标注。

**🔧 技术方法**

技术方法包括：基于Whisper的时间戳token ASR；重放上下文 + 交叉熵 + KL 损失；VAD+非语音段插入生成自监督纠正样本；仅更新最后跨注意力层和层归一化实现参数高效微调。

**📊 数据集**

数据集：使用 Common Voice zh‑TW 进行目标纠正（34.9 小时合成音频），Gap 和 Long‑Gap 评测集；同时在 ASCEND、CommonVoice‑EN‑min 等 OOD 集合上评估保留性能。

**📈 对比分析**

对比实验：与 SFT、timestamp‑only、编辑重放、Reduced Teacher Forcing 等基线相比，REDDIT 在 Whisper‑tiny 上 mIoU 从 38.7% 提升至 95.0%，AAS 从 4806 ms 降至 66 ms，Drift>10s 从 26.7% 降至 0.2%；在 OOD 设定下保持 MER 接近原模型，远优于 SFT 解码器微调导致的 MER 直升数十倍。

**⚠️ 局限性**

局限性：目前方法仅适用于显式时间戳 token 的 ASR，难以直接迁移到自由文本或语音指令型语言模型；需要手工插入非语音段，缺乏更通用的自监督生成策略；在更大模型上 Stage‑2 的重要性尚未完全阐明。

---

## 764. SovereignPA-Bench: Evaluating User-Owned Personal Agents under Evolving Intent, Platform Mediation, and Consent Constraints

**arXiv ID:** 2607.05363 | [PDF](https://arxiv.org/pdf/2607.05363v1)

**作者:** Dylan Zongmin Liu `[一作]` `[通讯]` (Stanford University), Dylan Zongmin Liu (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个针对用户拥有的个人代理（personal agent）的新基准，评估其在不断变化的意图、平台干预、隐私、同意、证据和负担等方面是否真正代表用户行动；

**💡 创新点**

创新点在于将“用户主权”(user sovereignty)作为核心评估目标，构建多维度风险与收益指标体系，并提供可执行的artifact化基准；

**🔧 技术方法**

采用了场景化设计、基线策略对比、配对评估、引导式安全提示、ReAct工具使用、LLM判断守卫等技术，并通过bootstrapping计算置信区间；

**📊 数据集**

使用120个精心设计的合成情境（涵盖偏好演化、隐私边界、同意边界、证据依赖等），在四大模型族（OpenAI、Anthropic、Google、开源权重）上运行，生成3,840条冻结提示的实验记录；

**📈 对比分析**

对8种基线策略（Direct、Memory、Consent、Evidence、SafetyPrompt、ReActToolUse、LLMJudgeGuard、FullSovereign）在同一情境下进行配对比较，结果显示FullSovereign在主权得分、隐私泄露、同意违规、证据支持等指标上均优于其他策略，且在不同模型族和“硬核”高冲突子集上保持领先；

**⚠️ 局限性**

局限性包括：情境为合成文本/工具场景，缺乏真实用户轨迹与多模态交互；评估主要聚焦文本代理，无法覆盖GUI或真实API调用；部分指标（操纵捕获、升级）仍具主观性；未来需扩展至多模态、跨代理协商和真实部署实验。

---

## 765. Faithfulness to Refusal: A Causal Audit of Neuron Selectors

**arXiv ID:** 2607.05355 | [PDF](https://arxiv.org/pdf/2607.05355v1)

**作者:** Ananth Eswar `[一作]`, Vinay Kumar Sankarapu `[通讯]` (Lexsi Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于权重层行零屏蔽的因果性 Selector 审计框架，评估不同 Attribution 指标在语言建模与安全行为（拒绝）层面的因果可靠性，并通过对比性拒绝编辑验证其可塑性。

**💡 创新点**

① 引入单次无微调的权重行零屏蔽因果审计，直接测量 Selector 的因果有效性；② 在行为层面使用对比拒绝边际构造的阈值，实现对安全行为的精准安装；③ 发现 Attribution 选择器的 rank‑stability 与因果有效性不匹配，揭示不同行为在权重空间中的冗余子空间及架构依赖性。

**🔧 技术方法**

使用 LRP、Integrated Gradients、Borda 共识、Wanda、平均激活等重要性指示器；对 Transformer 的 Q/K/V/O、MLP gate/up/down 层进行行零屏蔽；对比性拒绝掩码算法；通过 Perplexity、MMLU、GSM8K、IFEval、SorryBench、OR‑Bench‑Hard 等评测指标。

**📊 数据集**

WikiText‑2 与 C4（校准）、CAST 对抗样本、SorryBench、OR‑Bench‑Hard、MMLU、GSM8K、IFEval 用于行为与泛化评估，覆盖 LLaMA‑3、Qwen、Gemma 等模型。

**📈 对比分析**

在七种 Selector（随机、Magnitude、Wanda、MeanActivation、LRP、IG、Borda）与五个模型上进行 LeRF/MoRF 真实性间隙对比；在行为层面比较拒绝率、善意误拒、PPL 与下游任务准确率。结果显示 Attribution 选择器在 LM 级别的真实性间隙高达 1–2 量级，行为层面可在保持 PPL 与下游任务性能的前提下将 CAST‑malign 拒绝率提升至 0.82–0.95，而随机/平均激活基线表现不佳或降低拒绝。

**⚠️ 局限性**

仅针对密集解码器模型，未涉及 Mixture‑of‑Experts、编码器‑解码器结构、非梯度 attribution 方法、非英语输入；对拒绝评估依赖单一阈值和表面形式；仅做单次评估，缺少多种 seed 重复；未验证实际部署中的安全性与可恢复性。

---

## 766. Multiplayer Interactive World Models with Representation Autoencoders

**arXiv ID:** 2607.05352 | [PDF](https://arxiv.org/pdf/2607.05352v1)

**作者:** Anthony Hu `[一作]`, Patrick Pérez `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `8d10c613-917e-4880-9716-17789f50e119` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文训练了一个可在 Rocket League 上实时交互的多玩家世界模型，能够根据四名玩家的动作预测并生成后续游戏画面；

**💡 创新点**

创新点包括：① 设计了基于潜在扩散 Transformer 的多玩家条件化模型；② 将冻结的自监督特征 DINOv3 用于构建潜在空间并加入压缩瓶颈；③ 采用 diffusion forcing 与 few‑step distillation 实现长时间稳定、实时推理；④ 提出了 Action Recoverability Ratio (ARR) 指标评估模型对动作的可控性；⑤ 系统性评估多玩家对比单玩家、模型规模、数据规模等维度的影响。

**🔧 技术方法**

使用的技术：潜在视频自编码器（基于 DINOv3 的特征提取 + 线性瓶颈 + 视觉 Transformer 解码器）；流匹配的扩散训练目标与 diffusion forcing；AdaLN 条件化；多视角拼接 + 每个玩家动作嵌入；几步蒸馏（PSD）加速采样；KV 缓存 + 流式推理实现 20fps；自监督特征一致性、LPIPS 与 P‑DINO 的感知损失；物理状态回归探针与人类评价。

**📊 数据集**

数据集为 10,000 小时 2v2 Rocket League 对局，全部由 RL 机器人 Nexto 生成，记录了每位玩家的 30fps 视频、15Hz 动作（键盘映射）以及 120Hz 物理状态；数据覆盖三张地图（Champions Field、Forbidden Temple、Deadeye Canyon），约 82,983 场比赛。

**📈 对比分析**

与像素空间模型、单玩家模型以及不同尺寸/数据量模型对比，评价指标包括 gFID、gFVD、gFDD、ARR、PSNR、SSIM、LPIPS、P‑DINO 以及人类 Elo。5B 多玩家模型在 20fps 实时推理下取得 gFID 10.7、gFVD 163.1、ARR 0.91，显著优于单玩家模型和像素空间模型（差距达 10 倍以上）。模型规模越大，gFID 与 gFVD 越低；数据规模增大时 gFID 先提升后饱和，ARR 随着数据量持续上升。

**⚠️ 局限性**

局限性包括：全部训练样本由同一 RL 策略产生，导致行为多样性有限；仅覆盖三张地图，缺乏场景多样性；游戏本身高度确定性，缺少随机扰动；对极少见动作的可控性仍不足；在极端长推理或非常规动作组合下仍会出现漂移。

---

## 767. SynCity 3000: Bootstrapping Scene-Scale 3D Diffusion

**arXiv ID:** 2607.05392 | [PDF](https://arxiv.org/pdf/2607.05392v1)

**作者:** Paul Engstler `[一作]` (University of Oxford), Andrea Vedaldi `[通讯]` (University of Oxford)

**通讯引用:** 72606 | [OpenAlex ID](https://openalex.org/A5060511349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种两阶段扩散框架，先生成完整的二维模板，再通过卷积化的 3D 生成器将其转换为高质量的 3D 高斯斑点场，从而实现从文本提示生成任意规模、任意布局的全局一致 3D 场景。

**💡 创新点**

创新点在于：①利用重叠滑窗的多尺度扩散实现全局一致的二维模板；②将 TRELLIS 的 3D 生成器改造成卷积推理模式，支持任意尺寸的场景生成；③设计了一套合成场景数据引擎，用于大规模场景级别的无监督微调。

**🔧 技术方法**

使用的技术包括：潜在扩散模型（Latent Diffusion）、多窗口重叠扩散（MultiDiffusion 风格）、卷积化的稀疏结构生成与结构化潜在生成、DINOv2 特征投影、3D 高斯斑点渲染及自监督微调。

**📊 数据集**

数据集主要是自己构建的 32 万条合成场景，采用 Objaverse‑XL 物体随机摆放在随机地形上，并生成对应的稀疏体素与结构化潜在表示。

**📈 对比分析**

与 SynCity、TRELLIS、TripoSG、Hunyuan3D‑2.1 等方法对比，实验在 LPIPS、SSIM、PSNR、Chamfer、F‑score 等指标上均领先；用户研究显示对布局控制和场景整体质量的偏好显著高于对比方法。

**⚠️ 局限性**

局限性包括：高度依赖合成数据，缺乏真实世界场景的评估；对极大规模场景仍需进一步优化计算效率；卷积化推理需要大量 GPU 资源，训练成本较高。

---

## 768. LLM-as-a-Verifier: A General-Purpose Verification Framework

**arXiv ID:** 2607.05391 | [PDF](https://arxiv.org/pdf/2607.05391v1)

**作者:** Jacky Kwok `[一作]` (Stanford University), Azalia Mirhoseini `[通讯]` (Stanford University)

**通讯引用:** 3008 | [OpenAlex ID](https://openalex.org/A5070731184)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LLM-as-a-Verifier，一个在不需要额外训练的情况下，对多模态长序列任务（编码、机器人、医疗）提供细粒度验证和进度评估的通用框架。

**💡 创新点**

创新点：①使用对评分词标记分布的期望计算连续分数，显著降低分数歧义和连等概率；②在分数粒度、重复评估和评估准则分解三维度上系统性扩展验证精度；③引入概率基支点对（Probabilistic Pivot Tournament）高效排序算法，减少对算力的需求。

**🔧 技术方法**

技术：概率推理（token logits期望）、多维度扩展（分数粒度 G、重复评估 K、准则 C）、BERT/LLM prompt工程、VLM多模态推理、强化学习奖励塑造（SAC、GRPO）以及可插拔的代理层 TurboAgent。

**📊 数据集**

数据集：Terminal‑Bench V2、SWE‑Bench Verified、RoboRewardBench、MedAgentBench（各自覆盖命令行、代码补丁、机器人轨迹、医学决策等场景）。

**📈 对比分析**

对比方法：与离散 LM 判别器、已训练的奖励模型（RoboReward‑8B、Robometer‑4B、TOPReward 等）以及标准评测基线进行对比。性能表现：Terminal‑Bench V2 86.5%、SWE‑Bench Verified 78.2%、RoboRewardBench 87.4%、MedAgentBench 73.3%，均刷新了公开榜单记录，且在多任务下保持了统一的零训练优势。

**⚠️ 局限性**

局限性：①依赖能够暴露 token logits 的 LLM，限制了可使用的模型；②在极其复杂或多模态任务中仍可能出现误判；③虽然使用了多维扩展，但在算力预算极低时仍需权衡 G、K、C 的取值；④对极端安全场景的误判风险尚未充分评估。

---

## 769. What Does a Discrete Diffusion Model Learn?

**arXiv ID:** 2607.05381 | [PDF](https://arxiv.org/pdf/2607.05381v1)

**作者:** Rodrigo Casado Noguerales `[一作]` (ETH Zurich), Aran Raoufi `[通讯]` (ETH Zurich)

**通讯引用:** 210 | [OpenAlex ID](https://openalex.org/A5075236002)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

本文在离散扩散模型中提出了连续时间马尔可夫链（CTMC）形式的 ELBO，并给出了严格推导，随后证明了 Oracle Distance 定理：负 ELBO 等价于数据熵加上正向逆过程与学习过程之间的路径 KL。基于此，本文给出了三个常见参数化（去噪器、cavity/bridge 插值器、score）的确切坐标字典，并说明它们在不同扩散策略（masked、uniform、GIDD 等）下的差异及其导致的数值不稳定性。进一步地，文章通过信息理论分析解释了信息损失率、熵下界和梯度消失等现象，并提供了初始化时的校准公式和实证验证。

**💡 创新点**

核心创新点包括：
- 以路径级别严格推导连续时间离散扩散的 ELBO，包含边界项；
- Oracle Distance 定理，揭示负 ELBO 的本质是路径 KL 与数据熵的和；
- 唯一最优逆过程为条件期望的逆跳率，Pythagorean 分解与投影原理统一了所有参数化；
- 构建去噪器、cavity、score 三种坐标的闭式转换字典，解释了 MDM、UDM、SEDD、GIDD 等方法的本质；
- 明确了为何 uniform diffusion 的 denoiser 参数化在初始化会发散，给出校准标准；
- 证明所有扩散过程共享相同的负 ELBO 下界 H(q₀)，并揭示信息损失率 d/dtH(Z₀,Z_t) 的作用。

**🔧 技术方法**

技术手段：
- 连续时间马尔可夫链理论（生成器、转移核、时间逆）
- Girsanov 定理与路径相对熵
- Bregman 散度与 Pythagorean 定理
- 信息论（熵、互信息、I‑MMSE 等）
- 变分推断（ELBO）
- 小时离散到连续的极限推导
- 重要性采样时钟的可解释性
- 数值验证（exactly solvable 模型）

**📊 数据集**

实验与验证主要在一个可解析的离散模型上进行，用于数值验证所有公式与恒等式；论文未给出对真实文本、图像等数据集的实验。

**📈 对比分析**

比较方法：
- 对同一扩散过程，分别使用去噪器、cavity、score 三种参数化，在相同网络结构下计算 NELBO、边界项、以及采样后的生成 perplexity；
- 通过数值计算显示：uniform diffusion 下去噪器发散、cavity 收敛；masked diffusion 三种参数化收敛；
- 校准公式与 log V 以及终点 KL 的一致性验证；
- 结果表明 Oracle Distance 能精确分解训练误差与采样误差，提供了对比与调优的量化依据。

**⚠️ 局限性**

局限性：
- 只在有限状态空间和 token‑factorizable 过程上严格证明，连续空间和非因子化噪声尚未覆盖；
- 需要正则性与固定支撑假设才能得到信息损失率与熵下界；
- 路径 KL 仅衡量逆过程匹配，无法捕捉采样阶段的 factorization 误差；
- 实际大规模模型的数值稳定性仍受网络容量与采样策略影响；
- 论文未在真实数据集上进行实验，验证效果仍待进一步探索。

---

## 770. TabPack: Efficient Hyperparameter Ensembles for Tabular Deep Learning

**arXiv ID:** 2607.05380 | [PDF](https://arxiv.org/pdf/2607.05380v1)

**作者:** Yury Gorishniy `[一作]` (Yandex), Artem Babenko `[通讯]` (Yandex)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种能够高效训练多组不同超参数的MLP集合模型，实现了无需大量调参即可在表格数据任务中获得优秀表现。

**💡 创新点**

通过在单个模型包中随机采样并并行训练多组超参数，并在线训练过程中根据验证集动态选择最终集成成员，减少了传统调参的需求。

**🔧 技术方法**

使用Packed Ensemble技术将多组MLP与对应优化器打包在同一张量内，利用批量矩阵乘法并行训练；采用MuON优化器；在线贪心集成和自适应早停机制。

**📊 数据集**

在多达13.6M样本、900+特征的公开工业级表格数据集上进行评估，并在其他大型（1M+样本）与中小型（TabArena）基准上进一步验证。

**📈 对比分析**

与XGBoost、原始MLP、RealMLP、TabM等传统及现代基准相比，本文方法在不做调参的情况下获得相近甚至更优的性能，并且在MacBook上运行速度快于使用GPU调参的对手。

**⚠️ 局限性**

对超参数多样性的实际提升作用有限，主要作用是降低调参成本；宽度统一导致模型多样性受限；需要进一步改进基准模型和在线集成算法。

---

## 771. CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents

**arXiv ID:** 2607.05378 | [PDF](https://arxiv.org/pdf/2607.05378v1)

**作者:** Yujiang Li `[一作]` (Tsinghua University), Yuxiao Dong `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于PPO的强化学习框架，训练长时限LLM代理在固定上下文窗口下实现可训练的上下文压缩，并联合优化执行与摘要生成；

**💡 创新点**

创新点在于将上下文压缩作为可学习的训练目标，使用token级损失归一化和跨轨迹GAE以处理变长段落，并通过共同奖励学习摘要与执行策略；

**🔧 技术方法**

采用PPO、token级损失归一化、跨轨迹通用优势估计（GAE）、可训练摘要生成策略，以及对回合中多段压缩的分段优化；

**📊 数据集**

训练使用SWE-Dev数据集，评估采用SWE-bench Verified和Terminal-Bench 2.0；

**📈 对比分析**

与基线模型及无压缩PPO进行比较，实验证明在GLM-4.7-Flash和GLM-4.5-Air等模型上，Pass@1分别提升约5.5-7.0个百分点；

**⚠️ 局限性**

局限性包括：训练得到的优势在单窗口评估下迁移性差；跨轨迹GAE仍是近似；实验主要聚焦代码类长时限任务，尚未验证到更广泛的代理领域；

---

## 772. Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation

**arXiv ID:** 2607.05377 | [PDF](https://arxiv.org/pdf/2607.05377v1)

**作者:** Jiaqi Peng `[一作]` (Tsinghua University), Tai Wang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Cortex 双系统框架，利用高层 Vision‑Language‑Model 与低层 Vision‑Language‑Action 的双向对齐，通过 32 个可执行子任务原语实现长时程机器人操作的规划与执行。

**💡 创新点**

创新点包括：① 将子任务标准化为 32 个技能原语并构建可执行接口；② 引入事件平衡采样解决子任务切换的语义与时间模糊；③ 设计 harness 工程实现多模态指令到子任务的自适应映射；④ 异步推理实现实时规划与执行的闭环。

**🔧 技术方法**

采用大规模多模态语言模型、VLA 模型、结构化元数据注释、模拟程序生成、事件平衡采样、k‑inematic grounding、可视化记忆以及异步双向推理技术。

**📊 数据集**

使用 4,000 小时公开视频数据（AgibotWorld、Galaxea、BEHAVIOR‑1K、RoboCerebra 等）、30 小时模拟数据（RoboTwin、RMBench）、Libero‑Long、RoboTwin 以及 ARX ACONE 真实化学实验数据。

**📈 对比分析**

与 monolithic VLA、π_0、π_0.5、MemoryVLA、OpenVLA‑OFT 等方法对比，在 Libero‑Long 零样本成功率 95.5%（比基线提升 3.1%），在 RoboTwin 成功率 86.8%（提升 4.1%），在真实化学任务零样本成功率 65%（远超 0% 的端到端方法），验证了显著性能提升。

**⚠️ 局限性**

限制包括：文本记忆缺乏空间坐标，导致大规模移动操作时实例对应困难；以及标准视觉编码对高频微状态变化不敏感，难以捕捉快速动态环境的细微变化。

---

## 773. PixWorld: Unifying 3D Scene Generation and Reconstruction in Pixel Space

**arXiv ID:** 2607.05373 | [PDF](https://arxiv.org/pdf/2607.05373v1)

**作者:** Sensen Gao `[一作]` (Nanyang Technological University), Jia-Wang Bian `[通讯]` (AISphere)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一种统一的3D场景重建与生成框架 PixWorld，能够在单一模型中通过像素空间扩散与可微渲染实现端到端训练。

**💡 创新点**

创新点包括：① 在像素空间直接做扩散，消除 VAE/RAE 的信息瓶颈；② 采用两流 Diffusion Transformer 分别处理干净与噪声视角；③ 引入基于预训练 3D 基础模型的几何感知损失，为 3D 结构提供监督。

**🔧 技术方法**

使用的核心技术包括像素空间流动匹配（Flow‑Matching）扩散、两流 Diffusion Transformer、3D Gaussian Splatting 可微渲染、几何感知损失、深度回归与可微渲染优化。

**📊 数据集**

主要数据集为 RealEstate10K、DL3DV-10K、WorldScore 以及 10M 张 BLIP‑3o 单图像数据。

**📈 对比分析**

通过与现有重建（YoNoSplat、DepthSplat）和生成（LVSM、GF、Gen3C、FlashWorld、Gen3R 等）基线在多视角重建、单图/双图生成、WorldScore 等任务进行比较，PixWorld 在 PSNR/SSIM/LPIPS、相机控制 AUC 以及 VBench 评估指标上普遍优于或接近最先进方法，尤其在几何一致性和相机轨迹控制上明显领先。

**⚠️ 局限性**

局限性包括：对大量多视角训练数据的依赖；在极低视角或极端噪声条件下性能可能下降；几何感知损失对噪声输入的鲁棒性仍有限；缺乏对稀疏或光照变化显著的场景的进一步验证。

---

## 774. GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks

**arXiv ID:** 2607.05369 | [PDF](https://arxiv.org/pdf/2607.05369v1)

**作者:** Kaiyuan Chen `[一作]` (University of California, Berkeley), Ken Goldberg `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Graph-as-Policy (GaP) 多智能体框架，自动从自然语言任务描述生成可解释的计算图，并通过内部仿真自我学习优化图结构，最终在机器人上实现 Variational Automation (VA) 任务。

**💡 创新点**

创新点在于将 LLM 生成的技能拆分为有向图节点，构造可验证的计算图；引入多智能体层级调度，降低单体 LLM 的误差与“作弊”风险；结合模块化的 MORSL 技能库与自我学习仿真循环，实现对对象几何和姿态变化高度适应的持续执行。

**🔧 技术方法**

使用大型语言模型（Gemini‑3.1‑Flash‑Lite、Claude 等）生成和优化图；视觉语言模型（SAM2、Grounding DINO、Molmo 等）进行感知；机器人运动规划与控制节点（cuRobo、GraspGen 等）；内部使用 NVIDIA Isaac‑Lab 进行仿真回放；图结构验证与多智能体协同。

**📊 数据集**

使用八个开放的 Variational Automation 基准，包括四个基于 LIBERO 的仿真任务（带位置、角度、排列等变化）和四个真实机器人任务（Franka arm、UR5、ZED Mini 相机等）；MORSL 51 个初始技能库；以及公开的视觉与抓取数据集如 SAM2、Grounding DINO、Molmo 等。

**📈 对比分析**

与 CaP‑X、π_0.5、MolmoAct2、TipTop、TAMP 等基线对比；在 100+ 次仿真试验和 200+ 次真实试验中，GaP 的成功率普遍显著更高（如真实抓取任务 100% vs. 32%；自学习前后 Make Popcorn 由 33% 提升至 90%）。在吞吐量和执行时间上也优于基线，显示出更高的可靠性与效率。

**⚠️ 局限性**

仍存在工业级可靠性不足、执行时间偏长、对动态、柔性或力反馈要求高的任务支持有限；依赖 VLM 推理与 IK 规划时间；需要进一步的自学习和参数调优，且对更复杂场景的迁移性能尚待验证。

---

## 775. Rerouting Curves on Surfaces

**arXiv ID:** 2607.05362 | [PDF](https://arxiv.org/pdf/2607.05362v1)

**作者:** Timo Brand `[一作]` (Technische Universität München), Pavel Valtr `[通讯]` (Charles University Prague)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文研究在给定点集上，将图的无交叉嵌入通过逐条边重路由的方式在同一可定向曲面（尤其是环面及更高种数的曲面）上重新配置到另一个嵌入的可行性。

**💡 创新点**

创新点在于：①证明任意匹配、树、森林都可以在环面（以及任何种数≥1的可定向曲面）上重新配置；②给出在投影平面下若两嵌入同处于一个盘内的完美匹配的正向可重新配置；③提出两阶段算法（先消除与边界相交的段，再利用曲面结构绕行），并给出关于边段数k的固定参数算法；④给出多类图在曲面上不可重新配置的负例与对应的构造。

**🔧 技术方法**

核心技术包括：曲面几何表示（基本多边形/方形模型）、框架树（frame tree）序列的构造与迭代、在极小邻域内的弧路由与平移、利用曲面非平凡同调/分离性质进行绕行、以及对树/森林的递归子图处理。

**📊 数据集**

本文为理论分析，不涉及实验数据集；所有结果均为定理与构造证明。

**📈 对比分析**

由于缺乏实验对照，性能评估基于算法复杂度。作者给出序列长度上界为O(c·3^k·g^2s^2)及输出规模为O(c^3·3^3k·g^5s^5)，其中k为跨越曲面边界的段数，s为段总数，c为连通分量数；在k为常数时实现为FPT。

**⚠️ 局限性**

局限性包括：算法在k上呈指数增长，未给出多项式时间算法；对非可定向曲面（如投影平面）尚未完全推广；对一般平面图（尤其是三连通图）的可重配置性仍未解决；负例表明问题整体可能难以求解。

---

## 776. Selective Disclosure Watermarking for Large Language Models

**arXiv ID:** 2607.05353 | [PDF](https://arxiv.org/pdf/2607.05353v1)

**作者:** Xuyang Chen `[一作]` (University of Pennsylvania), Qi Long `[通讯]` (University of Pennsylvania)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于层次词表划分的水印框架（Hierarchical Vocabulary Routing，HeRo），能够在大语言模型生成文本时嵌入多位元信息，并支持不同授权级别的选择性解码。

**💡 创新点**

创新点在于：①通过递归划分词表，将水印信息分层嵌入，使得不同级别的验证者只能解码其授权层级的负载；②提供了严格的统计无偏性与选择性披露的理论保证；③实现了高效、可批量的 GPU 加速，显著降低了生成和解码延迟。

**🔧 技术方法**

主要技术包括：Gumbel‑Max 采样作为无偏采样规则；基于聚合概率的多级块划分与路由；层次化的密钥表与伪随机函数；计数器式伪随机数生成器以实现 O(1) 的证据计算；以及批量化实现与 CUDA 并行优化。

**📊 数据集**

实验使用了两类公开数据集：①正式新闻文本集，②对话与创意文本集；每个数据集抽取 1,000 篇文档作为生成提示，评估了不同层级配置下的性能。

**📈 对比分析**

与现有多位元水印方法（MPAC、StealthInk、BiMark）对比，HeRo 在 24‑bit 水印下的位错误率低于 2%，几乎与无水印模型相同的困惑度；生成时仅增加约 4‑5% 的延迟；解码速度比基线快数倍；且在随机替换、插入、删除、机器翻译以及 DIPPER 释义攻击下表现出最高的鲁棒性。

**⚠️ 局限性**

主要局限包括：①层次结构深度越大，低级别的解码准确度会下降；②当前设计仍在固定总负载（如 24‑bit）下评估，未给出不同总负载与选择性披露之间的精确权衡理论；③对极端大容量水印或非常短文本的适用性尚未充分验证。

---

## 777. Beyond Isolated Objects: Relationship-aware Open Vocabulary Scene Understanding via 3D Scene Graph Analysis

**arXiv ID:** 2607.05348 | [PDF](https://arxiv.org/pdf/2607.05348v1)

**作者:** Xianhao Chen `[一作]` (Zhejiang University), Zhaopeng Cui `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种关系感知的开放词汇3D场景理解框架，通过构造多视角推理的3D场景图来增强语义上下文，并在该图上进行双流自适应门控上下文GAT和层次对比学习，实现对开放词汇语义的精细化与提升。

**💡 创新点**

创新点包括：①基于视觉-语言模型的无监督场景图构造方法，可在无关系标注的条件下自动推理物体关系；②双流自适应门控GAT，将密集几何特征与CLIP语义特征分离传递并通过门控融合，避免多模态特征干扰；③层次对比损失与双对齐策略，兼顾实例级一致性与类别级判别，提升开放词汇泛化能力。

**🔧 技术方法**

使用的主要技术有：CLIP文本/图像编码、LSeg密集特征、视觉-语言模型（如qwen-vl-max）、SAM+Describe Anything进行视图生成与标注、四层双流GAT与全局上下文注入、层次对比学习与双对齐损失。

**📊 数据集**

实验数据集包括：ScanNetV2、ScanNet200（长尾20/200类），以及跨域零样本测试集ScanNet++和Replica。

**📈 对比分析**

与现有方法（OpenScene、CUA-O3D、OV3D、Mosaic3D等）比较，RelGraphOV在ScanNetV2 mIoU从54.2提升到58.4、mAcc从66.6提升到73.4；在ScanNet200 mIoU从12.4提升到14.5、mAcc从25.1提升到26.1；在ScanNet++零样本 mIoU从13.3/11.7提升到20.9、mAcc从20.0/15.3提升到33.2；在Replica零样本 mIoU从20.4提升到22.7、mAcc从31.7提升到32.7。整体表现明显优于同类方法。

**⚠️ 局限性**

主要局限包括：依赖固定的LSeg/CLIP特征，未融合更先进的稠密视觉语言特征；仅针对静态重建场景，缺乏对动态场景的时序场景图更新机制；在极端长尾类别或视角极端遮挡下仍可能出现误判。

---

## 778. WildSplat: Feedforward Gaussian Splatting from Unposed In-the-Wild Images

**arXiv ID:** 2607.05347 | [PDF](https://arxiv.org/pdf/2607.05347v1)

**作者:** Xiyu Zhang `[一作]` (Zhejiang University), Qingnan Fan `[通讯]` (vivo BlueImage Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种面向无姿态野外照片的前向 3D 高斯喷射框架（WildSplat），能够在仅用少量未标姿图像且带有多样光照的情形下，依据指定参考图像实现高质量、光照一致的新视角合成；

**💡 创新点**

创新点包括：① 双分支架构将几何与外观显式分离，几何分支通过 DINOv2+VGGT 提取与光照无关的 3D 结构并估计相机位姿；② 外观分支利用全局预调制（AdaLN‑Zero）和跨注意力注入参考图像的光照信息；③ 多参考训练策略在同一批次中同时对多个光照条件进行监督，显著提升几何与外观的解耦与稳定性；

**🔧 技术方法**

核心技术：3D Gaussian Splatting、DINOv2 视觉编码器、VGGT 后端、AdaLN‑Zero 预调制、跨注意力注入、相机位姿估计头、条件光栅化、几何导向的视角采样和多参考损失；

**📊 数据集**

训练与评估数据集：DL3DV（光照一致），MegaScenes、MegaDepth（光照多样）以及通过视频重光技术扩增的 DL3DV；评测使用 Phototourism 以及 MegaScenes 测试集；

**📈 对比分析**

与 AnySplat、WorldMirror 等前向方法以及 WildGaussian、GS‑W、FSGS 等基于优化的方案对比，WildSplat 在 PSNR、SSIM、LPIPS 等指标上均实现了显著提升，且相机位姿估计精度逼近 VGGT，证明了其在野外多光照场景中的高效性与优越性；

**⚠️ 局限性**

局限性：仍需提供参考图像来指定最终外观，无法完全处理极端光照变化或移动物体；在极少视角、光照差异极大或缺乏重叠区域的场景下，几何解算与外观注入效果可能受限。

---

## 779. Deform360: A Massive Multi-view Visuotactile Dataset for Deformable World Models

**arXiv ID:** 2607.05390 | [PDF](https://arxiv.org/pdf/2607.05390v1)

**作者:** Hongyu Li `[一作]` (Brown University), Yunzhu Li `[通讯]` (Columbia University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一套大规模、全视角视觉-触觉数据集——Deform360，涵盖198种日常可变形物体、1,980个交互序列，并开发了无标记3D跟踪管道，用来生成高质量的粒子轨迹与动态几何；随后基于该数据集系统评估了3D粒子模型与2D视频生成模型在世界建模、预测与泛化上的表现，并在真实机器人上演示了MPC规划；

**💡 创新点**

创新点在于①首次提供涵盖多视角视觉与触觉、包含遮挡、接触细节的真实可变形物体数据；②设计了将3D高精度几何重建与多视角2D跟踪结合，并通过触觉约束优化粒子轨迹的无标记跟踪方案；③在相同数据条件下对比3D物理先验与2D生成模型的低数据/零样本泛化差异；

**🔧 技术方法**

主要技术包括：3D高斯展开（3D Gaussian Splatting）实现帧级几何重建；CoTracker3进行多视角2D点追踪并映射至3D；基于触觉的物理约束损失（形状、局部刚性、拉普拉斯平滑、触觉一致性）优化粒子轨迹；使用PGND、ParticleFormer、PhysTwin等3D模型及Cosmos-Predict 2.5 2B等2D视频生成模型进行评估；

**📊 数据集**

使用的数据集为Deform360：198件可变形物体，41台摄像机（720p/30Hz）+双手触觉抓取器，约23.3M帧、215.7小时；

**📈 对比分析**

对比方法包括3D粒子模型（PGND、ParticleFormer、PhysTwin）与2D视频生成模型（Cosmos-Predict）；在重建、重演、预测任务以及三种泛化设定（帧、剧本、物体）中，3D模型在低数据/同一物体内部泛化表现更佳；2D视频模型在零样本物体泛化与图像质量上优于3D模型，但在动作遵循与长周期预测上表现不稳；总体而言，3D模型在结构先验上更稳健，2D模型在大规模预训练下泛化更强；

**⚠️ 局限性**

局限性包括：①高遮挡场景仍会导致跟踪误差；②对高度塑料或易滑动材料的局部刚性和平滑假设不完全适用；③触觉传感仅测量法向压力，无法捕获微滑动；④2D视频模型受视觉域变换影响，难以直接用于真实规划；

---

## 780. CATs: Secure Blockchain Interoperability with Cross-chain Atomic Transactions

**arXiv ID:** 2607.05387 | [PDF](https://arxiv.org/pdf/2607.05387v1)

**作者:** Andreas Penzkofer `[一作]` (Move Industries), Franck Cassez `[通讯]` (Movement Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种跨链原子交易（CAT）协议，利用共享协调层、序列器、执行器、协调器和确认层实现不同区块链之间的原子、可验证、低延迟执行。

**💡 创新点**

创新点包括：① 先模拟再提交的“干净状态”层避免回滚复杂性；② 通过细粒度读/写依赖图实现最小阻塞，允许无关交易并行执行；③ 采用统一的 BFT 确认层和时间阈值机制保证安全与活性；④ 通过依赖深度限制与超时裁决平衡吞吐与一致性。

**🔧 技术方法**

技术手段：BFT 共识确认层（可视为 HotShot/HotStuff 等），序列化/分配交易的 Sequencer，执行器/事务处理器执行预模拟，协调器聚合多链提议并写入确认层，依赖图与 OCC（乐观并发控制）实现事务分离，超时和依赖深度阈值实现 liveness 与最小阻塞。

**📊 数据集**

实验数据集：10,000 个账户，资产分配充足；交易采用 Zipf 分布（参数 z=0.8）随机生成，50% 交易为 CAT；在 AWS t3.xlarge 机器上模拟两条链（快链与慢链），设置 block 间隔 1 秒、每秒 100 TPS、CAT 生命周期 10 个区块、最大依赖深度 1、链延迟 5 区块。

**📈 对比分析**

比较方法：与 Avalon（基于 IBC 的 2PC）以及 GMP+HTLC、GMP+2PC 等方案对比；使用相同的 block 间隔、链数、消息复杂度评估。结果显示：在两链场景下，本协议实现了 1 秒区块级最终性，消息复杂度为 O(n)；在 CAT 比例低至 10% 时成功率 >90%，即使 CAT 占比提升到 50% 仍能维持较高成功率；链延迟接近 CAT 生命周期时成功率下降，证明了超时与延迟的权衡。

**⚠️ 局限性**

局限性：① 依赖统一的 BFT 确认层和共享 Sequencer，实际多链部署中协调成本可能增加；② 目前实验仅限两链，未验证多链规模下的线性扩展；③ 超时与依赖深度阈值需人工调参，参数不当会导致性能下降或一致性风险；④ 在高度集中访问（Zipf 参数大）或高并发场景下，依赖深度爆炸可能导致资源耗尽；⑤ 协调器和执行器的安全性仍需依赖外部证明或质押，若失效可能导致协议停滞。

---

## 781. Abstract Color Voronoi Diagrams and Circular Sequences of Color Permutations

**arXiv ID:** 2607.05383 | [PDF](https://arxiv.org/pdf/2607.05383v1)

**作者:** Sang Won Bae `[一作]` (Kyonggi University), Evanthia Papadopoulou `[通讯]` (Universita della Svizzera italiana)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

研究了带颜色的抽象Voronoi图的高阶版本，并给出了顶点数的紧确上界。

**💡 创新点**

首次将抽象Voronoi框架推广到彩色高阶情形，给出顶点数上界4k(n−k)−2n，并提出对应的构造算法。

**🔧 技术方法**

结合色彩化的Clarkson–Shor随机采样技术和圆形排列的组合计数分析。

**📊 数据集**

使用理论构造的可实现的彩色排列序列（无实际数据集），并在此基础上分析复杂度。

**📈 对比分析**

与先前已知的O(k(n−k))界进行对比，证明该界是紧确的，并给出O(k^2 n log n)的构造时间，最坏情况仍优于之前的O(k^2 n log^2 n)。

**⚠️ 局限性**

仅在理论层面给出上界和算法，缺乏对大规模实际实例的实验评估；在k接近m时上界不一定最优，且实现复杂度仍高于最新的欧氏高阶Voronoi算法。

---

## 782. Search Beyond What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation

**arXiv ID:** 2607.05382 | [PDF](https://arxiv.org/pdf/2607.05382v1)

**作者:** Haozhe Wang `[一作]` (Hong Kong University of Science and Technology), Cong Wei `[通讯]` (University of Waterloo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对视觉生成模型在面对开放、演变、长尾世界知识时的缺失问题，提出并验证了一种通过共同训练推理器与生成器的“先教后搜”框架来发现并利用生成器特定的知识边界，从而实现对未知知识的高效检索与整合。

**💡 创新点**

创新点在于：①提出生成器特定的知识边界概念；②设计三阶段噪声抵抗推理器（gate‑filter‑integrate）与基于在线DPO与拒绝采样的两阶段共训练机制；③通过预执行检索数据创建可重放的离线评估基准，解决实时检索成本与漂移问题。

**🔧 技术方法**

核心技术包括：基于多模态语言模型的推理器训练、Diffusion‑DPO在线强化学习、Rejection‑Sampling Fine‑Tuning (RFT)、多模态检索工具（图像检索与Web检索）与检索结果过滤集成策略。

**📊 数据集**

使用了自构造的 20,839 句双语（中英）prompt 数据集（涵盖12类失败模式、22个领域）以及 145,642 预执行检索会话与 281,925 生成图像，用以离线复现和评估。

**📈 对比分析**

在自研的 NoSearch 与 Search‑Intensive 子集上评估，经过共训练后，8B 推理器+4B 生成器在整体得分上达 31.8/100，略优于同规模基线的 31.2/100，且在难度最高的 Set III 需求上接近前沿 VLM Oracle 的 33.9/100，显著提升了对知识缺失场景的处理。

**⚠️ 局限性**

局限性包括：仍需依赖人工构造检索语料与离线搜索；共训练过程对计算资源有一定需求；知识边界的发现依赖于对生成器的不断迭代，可能在更大规模模型或不同任务中表现不同。

---

## 783. Graph Sparse Sampling: Breaking the Curse of the Horizon in Continuous MDP Planning

**arXiv ID:** 2607.05359 | [PDF](https://arxiv.org/pdf/2607.05359v1)

**作者:** Idan Lev-Yehudi `[一作]` (Technion Israel Institute Of Technology), Vadim Indelman `[通讯]` (Technion Israel Institute Of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种新的在线规划算法Graph Sparse Sampling（GSS），通过共享后继状态层来避免树形搜索分支，适用于连续马尔可夫决策过程。

**💡 创新点**

创新点在于将采样从树形分支迁移到图形共享结构，实现大规模GPU并行采样；同时给出有限样本、有限时间的性能保证，证明在满足重叠、稳定性与动作覆盖等条件下，误差随规划周期呈多项式增长。

**🔧 技术方法**

主要技术包括：状态与动作的分离采样、基于重要性采样（SNIS）的图备份、核平滑的低秩生成器模拟、密度比的Rényi散度控制、以及GPU友好的批量化实现。

**📊 数据集**

实验数据集包括三种连续控制任务：旋转四维双积分器（Rotating DDI）、Lunar Lander和Reacher，覆盖了高维、线性与非线性、确定性与随机动态。

**📈 对比分析**

与传统树形搜索（DPW、VPW）在相同时间预算下对比，GSS在大多数测试场景下性能更优或相当；在高旋转或高维情况下，树形方法显著退化，而GSS保持稳定。

**⚠️ 局限性**

局限性主要在于对动作与状态提议的依赖，若提议质量低会导致计算浪费；目前仅单遍实现，未对多轮迭代或部分可观测环境进行扩展。

---

## 784. Geometric Reciprocity: Unlocking Self-Supervision for Stereoscopic Video Generation

**arXiv ID:** 2607.05354 | [PDF](https://arxiv.org/pdf/2607.05354v1)

**作者:** Jingyi Lu `[一作]` (University of Hong Kong), Kai Han `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个自监督的立体视频生成框架，利用单目视频训练立体图像填补网络，无需立体配对或合成数据。

**💡 创新点**

核心创新是几何互惠定理（Geometric Reciprocity Theorem, GRT），证明在最近邻 DIBR 下，目标视角的失配遮罩等价于从该视角向源视角逆向投影时失去的像素，从而能够仅凭深度估计直接得到训练用的失配遮罩。

**🔧 技术方法**

主要技术包括：Depth-Image-Based Rendering（DIBR）框架、最近邻像素投影、基于 GRT 的解析失配遮罩计算、立体填补网络（图像用 LaMa、视频用 ProPainter）以及对现有基础模型的微调。

**📊 数据集**

构建了三个基准数据集 ImageNet-GRT、Kinetics-GRT（训练）和 DAVIS-GRT（评估），并在 Inria 3DMovie 数据集上进行全链路评测。

**📈 对比分析**

与训练自由方法（StereoDiffusion、ZeroStereo、Mono2Stereo）及有监督基线（StereoCrafter）对比，GRT 自监督方法在 PSNR、SSIM、LPIPS、CLIP Temporal Consistency 等指标上均优于对手，且推理速度更快；在 Inria 3DMovie 上的视角舒适度与几何一致性也显著提升。

**⚠️ 局限性**

局限性包括：依赖于最近邻投影的 DIBR 公式，软插值或更复杂几何模型的适配仍待研究；GRT 对深度估计的精度敏感，深度误差会直接影响失配遮罩质量；以及对极端视角或非正交相机场景的泛化尚未充分验证。

---

## 785. From Fixed to Free Cameras: Calibration-Free View-Robust Vision-Language-Action Model

**arXiv ID:** 2607.05396 | [PDF](https://arxiv.org/pdf/2607.05396v1)

**作者:** Wenhao Li `[一作]` (Nanyang Technological University), Ran Xu `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无需相机标定的视觉-语言-动作模型CamVLA，通过把动作预测转为相机本地坐标并自学习手眼变换实现视角鲁棒性；

**💡 创新点**

创新点在于将动作生成与相机几何分离，使用相机本地动作和自学习手眼矩阵完成基座动作预测，无需部署时提供相机外参；

**🔧 技术方法**

采用Transformer式视觉语言模型、双头网络（动作头与几何头）以及确定性几何变换；

**📊 数据集**

使用RLBench仿真任务与真实Franka Research 3机器人收集的多视角演示数据；

**📈 对比分析**

在仿真和真实机器人上与π₀、GR00T N1.7等基线对比，CamVLA在未见视角下成功率提升约15–20%，在极端15°视角偏移时仍保持约30%成功率；

**⚠️ 局限性**

局限性在于仅支持第三人称单相机，未覆盖手腕摄像头的视角变化，且对极端视角变化与高精度任务仍表现不足。

---

## 786. ReCal3R: Reliability-Calibrated Learning Rates for Streaming 3D Reconstruction

**arXiv ID:** 2607.05356 | [PDF](https://arxiv.org/pdf/2607.05356v1)

**作者:** Xinze Li `[一作]` (Beijing Normal-Hong Kong Baptist University), Wentao Cheng `[通讯]` (Beijing Normal-Hong Kong Baptist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种用于流式3D重建的可靠性校准学习率方法ReCal3R，避免在长图像流中因过度写入导致的状态退化

**💡 创新点**

创新点在于将状态代币的可靠性（由累积偏差和注意力熵评估）与候选学习率（由对齐、重建残差和更新压力构成）相结合，形成双阶段校准更新策略

**🔧 技术方法**

采用测试时训练框架（TTT）、注意力机制、熵归一化、指数滑动平均以及闭式校准公式来计算最终学习率

**📊 数据集**

在ScanNet、TUM‑Dynamics、7‑Scenes、NRGBD、Bonn等公开数据集上评测

**📈 对比分析**

与CUT3R、TTT3R、MeMix、TTSA3R等基线对比，ReCal3R在长序列姿态误差（ATE）降低3.7倍、重建精度与完整度提升、深度估计误差下降，保持相近的运行时和显存消耗

**⚠️ 局限性**

局限性是过于保守的更新有时会削弱细节恢复，且当前只适用于紧凑递归状态的重建模型

---

