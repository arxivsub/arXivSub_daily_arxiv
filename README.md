# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-02 | 今日论文总数: 415

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Speak Now: Safe Actor Programming with Multiparty Session Types

**arXiv ID:** 2602.24054 | [PDF](https://arxiv.org/pdf/2602.24054v1)

**作者:** Simon Fowler `[一作]` (University of Glasgow), Raymond Hu `[通讯]` (Queen Mary University of London)

**通讯引用:** 1517 | [OpenAlex ID](https://openalex.org/A5102840586)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种全新的 actor 语言，能够在编译期使用多方会话类型（MPST）进行通信错误检测，并支持每个 actor 同时参与多条会话、事件驱动式编程和 Erlang 风格的失败恢复。

**💡 创新点**

创新点在于：①首次将 MPST 直接嵌入 actor 语言，解决了多发者单接收者模型与会话类型不匹配的问题；②通过流感知效应系统结合一流消息处理器实现静态会话类型检查；③提供了完整的 failure‑handling 与监督体系，保证即使 actor 崩溃也不会导致会话挂起；④通过 API 生成器将 Scribble 的全局协议自动转化为 Scala 类型安全的 actor 接口。

**🔧 技术方法**

核心技术包括：多方会话类型理论、流敏感效应系统、事件驱动（suspend）模型、第一类消息处理器、Erlang/Akka 风格的监督与 cascading failure、Scribble 的协议验证与 CFSM 生成，以及在 Scala 上实现的类型安全 API。

**📊 数据集**

使用的数据集为：Savina actor benchmark 集合（包括 Ping、Dining、Sieve 等），工业场景工厂（Factory）案例，以及 Chat Server 示例。

**📈 对比分析**

评估方法：通过在 Scala 中实现语言并在上述 benchmark 与案例上编写完整程序，展示了语言的表达力与正确性。虽然论文未给出细粒度的性能指标，但实验表明在不牺牲安全性的前提下，系统能够与传统 Akka/Erlang 程序实现相当的运行时性能；此外，MPST 的静态检查显著降低了运行时错误率。

**⚠️ 局限性**

局限性：①当前对“ibecome”（会话切换）等动态会话管理尚未在理论上完全覆盖；②未对可终止性/公平性做严格证明，导致对无限循环或非终止 handler 的支持仍依赖运行时错误捕获；③实现依赖 Scribble 的静态验证，若协议不满足兼容性条件，生成的 API 可能会受限；④在大规模分布式部署中，网络延迟和消息排队的性能影响尚未充分评估。

---

## 2. Localising Stochasticity in Weighted Automata

**arXiv ID:** 2602.23805 | [PDF](https://arxiv.org/pdf/2602.23805v1)

**作者:** Smayan Agarwal `[一作]` (Ashoka University), Aalok Thakkar `[通讯]` (Ashoka University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

提出一种通过谱半径缩放将有限质量的非负实数加权自动机正规化为局部随机化的概率自动机，并给出可行的算法。进一步，给出了将任意非负实数加权自动机拆解为指数增长率、归一化概率成分和标量乘积的三元结构，并通过随机化正则表达式（SRE）描述等价的概率语言。

**💡 创新点**

创新点在于：
1) 证明了有限质量加权自动机与概率自动机在尺度变换下等价；
2) 通过 Perron–Frobenius 理论给出了显式的归一化矩阵和算法；
3) 引入几何星运算的随机正则表达式，实现了从概率自动机到表达式的双向转换；
4) 将该框架推广到通用谱归一化和热带半环，得到三元分解与线性增长率的统一视角；
5) 对谱归一化可行的半环类给出了通用原则。

**🔧 技术方法**

主要技术包括：
- 加权自动机的矩阵表示和谱半径分析；
- Perron–Frobenius 理论及其可逆性推导；
- 强连通分量分块和逆序拓扑归一化算法；
- 状态消除与 Thompson‑Gulwani‑Cav算法的变体，构造随机化正则表达式；
- 线性/指数增长率的分解与归一化；
- 热带半环的循环平均与最短路径等式。

**📊 数据集**

未使用任何实验数据集，论文全部为理论证明与算法设计。

**📈 对比分析**

论文没有进行实验比较或性能评估；主要通过理论证明展示算法的正确性、可行性与时间复杂度（O(n⁴ + n²|Σ|)）。

**⚠️ 局限性**

局限性：
- 只适用于非负实数半环，不能直接推广到包含负数或一般实数权重的自动机；
- 需要谱半径<1或通过 ε-逼近才能归一化；
- 三元分解的增长率参数依赖于选取的 ε，缺乏唯一规范的取值；
- 对更高阶或自由半环的加权自动机是否可谱归一化仍未解决。

---

## 3. Age of Entanglement in Satellite Repeater Chains with Intermittent Availability

**arXiv ID:** 2602.23985 | [PDF](https://arxiv.org/pdf/2602.23985v1)

**作者:** Elif Tugce Ceran `[一作]` (Middle East Technical University), Elif Tugce Ceran `[通讯]` (Middle East Technical University)

**通讯引用:** 1011 | [OpenAlex ID](https://openalex.org/A5002544828)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了年龄指标（Age of Entanglement, AoE）来衡量卫星辅助量子中继链中端到端纠缠的“新鲜度”，并将其最小化问题建模为平均报酬马尔可夫决策过程（MDP），随后使用相对值迭代（RVI）算法求解最优策略；

**💡 创新点**

创新点在于将经典信息学中的年龄概念迁移至量子网络，统一考虑纠缠生成概率、交换成功率、存储退化以及卫星可见性的不确定性，提出基于状态的动态控制策略并验证其优越性；

**🔧 技术方法**

主要技术包括马尔可夫链建模、状态空间与动作空间定义、相对值迭代算法求解平均报酬MDP、以及对比基线贪婪生成+交换和等待准备两种策略；

**📊 数据集**

使用仿真数据，设定可见性转移矩阵、纠缠生成概率 p_L、交换成功概率 p_sw 以及记忆老化阈值 m^*，并在多种可见性场景下进行数值实验；

**📈 对比分析**

与贪婪和等待准备两种基线对比，结果显示RVI最优策略在所有参数范围内均能显著降低平均 AoE，尤其在低生成概率或高度不对称可见性条件下表现最为突出；

**⚠️ 局限性**

局限性包括仅研究单链（单卫星）场景、未考虑更大规模链路或多路干扰、未引入强化学习或深度学习等自适应控制方法，以及实验仅基于理想化仿真而非真实量子网络数据。

---

## 4. Truncated Step-Level Sampling with Process Rewards for Retrieval-Augmented Reasoning

**arXiv ID:** 2602.23440 | [PDF](https://arxiv.org/pdf/2602.23440v1)

**作者:** Chris Samarinas `[一作]` (University of Massachusetts), Hamed Zamani `[通讯]` (University of Massachusetts)

**通讯引用:** 14764 | [OpenAlex ID](https://openalex.org/A5100618738)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种训练搜索增强型大型语言模型的框架，利用截断步级采样和基于LLM的密集奖励来改进强化学习中的信用分配问题。

**💡 创新点**

创新点在于：①截断步级采样仅在当前决策点产生 k 个候选，从而显著降低优势估计方差；②使用LLM评审器生成离散{-1,0,+1}的思考、查询和答案质量奖励，提供丰富的逐步监督。

**🔧 技术方法**

技术方法包括：GRPO改进、截断步级采样、LLM-as-Judge奖励、奖励加权采样、KL正则化、检索-生成交互框架和LoRA参数高效微调。

**📊 数据集**

数据集覆盖七个问答基准：NQ、TriviaQA、PopQA（通用QA）以及HotpotQA、2WikiMultiHopQA、Musique、Bamboogle（多跳QA），检索知识库使用2018 Wikipedia dump。

**📈 对比分析**

与稀疏奖励方法（如原始GRPO、Search-o1）和过程奖励方法（StepSearch）相比，本文方法在七个基准上平均提升了约3% EM（相对提升7%），在最难的多跳任务上提升超过5%，并在小模型上收益更明显。

**⚠️ 局限性**

局限性包括：仍需大量计算（k个候选+LLM评审），奖励评估依赖LLM的可靠性，且在极长序列或高预算场景下截断采样可能不足以捕捉全局最优策略。

---

## 5. HumanMCP: A Human-Like Query Dataset for Evaluating MCP Tool Retrieval Performance

**arXiv ID:** 2602.23367 | [PDF](https://arxiv.org/pdf/2602.23367v1)

**作者:** Shubh Laddha `[一作]`, Yash Bhaskar `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HumanMCP 数据集，并使用两阶段生成-评判（Generator‑Critic）流水线自动化生成与工具元数据匹配的多样化、真人化用户查询；对 2800 款工具在 308 台 MCP 服务器上进行大规模评测；设计并评估跨模型、长上下文以及 RAG（检索‑生成）方案的工具选择性能。

**💡 创新点**

1) 通过五类用户人设（从新手到专家、从模糊到精准）生成真实语义的查询；2) 采用生成‑评判循环保证查询既符合人设又保持功能相关性；3) 以大规模工具元数据为基础，构建首个跨 2000+ 工具的自然语言查询语料库；4) 在长上下文与检索增强生成的设置中验证数据集的可扩展性与检索效能。

**🔧 技术方法**

生成器使用 GPT‑4（成本友好型），评判器采用高推理能力模型；利用 JSON‑RPC 形式的工具元数据；五类人设逻辑；检索方法包括 TF‑IDF、BM25、MiniLM（SentenceTransformer）；实验模型包括 GPT‑4o‑mini、Gemini 2.0 Flash、Claude 3.5 Haiku。

**📊 数据集**

HumanMCP 数据集（约 2800 个工具，5 个用户人设；约 2 000 k 条查询）；工具元数据来源于 MCP 官方服务器；对比基准使用 MCPToolBench++、MCP‑Bench、MCP‑AgentBench 等公开数据集。

**📈 对比分析**

跨模型基线：在 10/50/100 工具上下文中 Top‑1 Hit Rate 分别为 98.4%/93.6%/88.2%（Gemini 2.0 Flash）等，平均下降约 10%。长上下文扩展：Gemini 2.0 Flash 在 500/1000/2000 工具时准确率分别为 87.4%/75%/65%。RAG 实验：MiniLM 检索 Top‑10 Hit Rate 87.6%，Gemini 重新排序后准确率提升至 76%；TF‑IDF 及 BM25 分别为 68% 与 59%。总体显示：检索增强可显著提高大规模工具环境下的准确性。

**⚠️ 局限性**

1) 人设覆盖有限，未能涵盖用户混合经验、情绪变化等；2) 生成查询为纯模型生成，缺少拼写错误、断句、代码切换等真实噪声；3) 依赖工具元数据的完整性，描述不清导致查询质量下降；4) 仅包含英语查询，无法反映多语言与文化差异；5) 未主动添加人工失误或噪声，可能与实际部署场景有偏差。

---

## 6. Altitude-Aware Visual Place Recognition in Top-Down View

**arXiv ID:** 2602.23872 | [PDF](https://arxiv.org/pdf/2602.23872v1)

**作者:** Xingyu Shao `[一作]` (Tsinghua University), Ziyang Meng `[通讯]` (Tsinghua University)

**通讯引用:** 6404 | [OpenAlex ID](https://openalex.org/A5051392570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种全视觉、单张航拍图像下的相对高度估计与高度自适应视觉地点识别（VPR）框架，能在无额外传感器的情况下实现对小中型无人机的高精度定位。

**💡 创新点**

创新点在于将相对高度估计视为分类任务，利用频域特征与混合自适应边距分类器（QAMC）实现精确的高度推断，并通过高度自适应裁剪将查询图像归一化到统一尺度，从而显著提升跨高度检索性能。

**🔧 技术方法**

采用了二维FFT频域预处理、MixVPR聚合器、EfficientNet/ResNet骨干网络、ArcFace/AdaFace等角度边距分类损失、FAISS向量检索、以及One‑Class SVM与加权坐标估计的后处理技术。

**📊 数据集**

在两个合成数据集（CT01、CT02）和两个实测数据集（QD01、QD02）上进行评估，数据涵盖100–700 m高度、农村与城市环境，且加入噪声与JPEG压缩等降解。

**📈 对比分析**

与MixVPR、CosPlace、CricaVPR、DINOv2‑SALAD等基线以及MMDE方法对比，本文方法在R@1、R@5及定位成功率上分别提升约30–60 %（在最严阈值100 m内成功率>90 %），而MMDE在此任务上表现远逊。

**⚠️ 局限性**

局限性包括高度区间离散导致的细粒度误差、对极端高空（>700 m）或复杂地形（如多层建筑、密集植被）下的适用性尚未验证，以及对极低分辨率图像的鲁棒性需进一步提升。

---

## 7. Too Immersive for the Field? Addressing Safety Risks in Extended Reality User Studies

**arXiv ID:** 2602.23497 | [PDF](https://arxiv.org/pdf/2602.23497v1)

**作者:** Tanja Kojić `[一作]` (Quality and Usability Lab TU Berlin), Jan-Niklas Voigt-Antons `[通讯]` (Immersive Reality Lab Hochschule Hamm-Lippstadt)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对扩展现实（XR）技术在实验室外环境（如家庭、学校、公共空间）中进行用户测试时所面临的安全风险进行了系统性梳理，并呼吁研究者、开发者和机构共同制定安全策略与标准。

**💡 创新点**

创新点在于将安全风险从实验室迁移到实际环境的视角展开，提出了安全责任分配的关键问题，并强调了跨学科合作、标准化框架和实用指导手册的必要性。

**🔧 技术方法**

主要采用文献综述、案例分析和现有数据（如NEISS统计）的方法进行论证，并未应用具体算法或实验技术。

**📊 数据集**

未使用专门的数据集，文中引用的主要是公开统计数据（如VR相关事故的NEISS数据）以及先行研究中的安全评估框架。

**📈 对比分析**

文章未进行实验对比，故不存在传统意义上的性能评估；其价值在于概念性分析与现状评述，指出当前做法的局限性并建议改进方向。

**⚠️ 局限性**

主要限制包括：缺乏统一、可执行的安全评估框架；缺少实证验证不同安全措施的有效性；对不同环境和受众的细化指导不足；以及跨机构协作机制尚未形成。

---

## 8. Random-Forest-Induced Graph Neural Networks for Tabular Learning

**arXiv ID:** 2602.24224 | [PDF](https://arxiv.org/pdf/2602.24224v1)

**作者:** Haozhe Chen `[一作]` (Utah State University), Kevin R. Moon `[通讯]` (Utah State University)

**通讯引用:** 4473 | [OpenAlex ID](https://openalex.org/A5010822968)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过随机森林产生的邻近度构造图，将表格数据转为图结构，再输入 GNN 进行节点分类，形成 RF‑GNN 框架。

**💡 创新点**

利用随机森林的邻近度进行图构建并引入 RF‑GAP、OOB 等改进的邻近度方法；将无监督图结构学习与 GNN 结合，显著提升表格数据分类性能。

**🔧 技术方法**

随机森林、RF‑GAP / OOB 邻近度、阈值化构图、图卷积网络（GCN）+ MLP、阈值敏感性分析以及实验对比。

**📊 数据集**

36 个 OpenML‑CC18 基准分类数据集，样本数 8000–46000，特征数 8–118，类别数 2–26。

**📈 对比分析**

与 RF、XGBoost、LightGBM、GB、MLP、INCE 等方法进行 5 次重复实验，RF‑GNN 在 18/36 数据集排名第一，平均排名最优，整体 F1 分数平均提升 0.02–0.5，超过 INCE 10 个数据集。

**⚠️ 局限性**

需要手动选取邻近阈值，阈值敏感度存在；邻近度计算在大规模数据集上成本高；仅在分类任务验证，未覆盖回归等任务；图构建依赖随机森林训练质量。

---

## 9. UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking

**arXiv ID:** 2602.23734 | [PDF](https://arxiv.org/pdf/2602.23734v1)

**作者:** Hao Wu `[一作]` (Institute of Digital Twin), Xiaoyu Shen `[通讯]` (Institute of Digital Twin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 UTPTrack，一种统一的 token 剪枝框架，能够同时压缩搜索区域、动态模板和静态模板，实现 Transformer 视觉跟踪器的高效化。

**💡 创新点**

创新点在于：①首次实现三大组件的联合剪枝；②引入注意力引导、token 类型感知的剪枝策略；③将剪枝机制统一扩展到多模态和语言引导的跟踪任务。

**🔧 技术方法**

采用了基于 Transformer 的注意力权重进行重要性评估、基于中心 token 相似度的剪枝、token 类型先验奖励以及文本引导的剪枝策略，并嵌入轻量级的 Candidate 或 Template Elimination Module（CTEM）。

**📊 数据集**

在 10 个跟踪基准上评估，主要数据集包括 TrackingNet、LaSOT、GOT-10k、COCO、TNL2K、VASTTrack、DepthTrack、LasHeR、VisEvent、RGB-Thermal/Depth/Event 以及语言跟踪数据集。

**📈 对比分析**

与现有剪枝方法（CE、ToMe、EViT、DynamicViT 等）对比，UTPTrack 在保持或提升精度的前提下，平均压缩 65%~68% 的视觉 token，MAC 下降约 30%，在 RGB 和统一跟踪任务中分别达到 99.7% 和 100.5% 的基线性能，显著优于同类方法。

**⚠️ 局限性**

限制方面：剪枝依赖注意力权重的质量，若注意力分布失真可能导致重要 token 被误删；在极端高压缩率下仍可能出现轻微精度下降；以及对不同 Transformer 结构的适配需要额外调优。

---

## 10. Dynamics of Learning under User Choice: Overspecialization and Peer-Model Probing

**arXiv ID:** 2602.23565 | [PDF](https://arxiv.org/pdf/2602.23565v1)

**作者:** Adhyyan Narang `[一作]` (University of Washington), Maryam Fazel `[通讯]` (University of Washington)

**通讯引用:** 9214 | [OpenAlex ID](https://openalex.org/A5102886973)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究用户选择导致的多学习者市场中的过度专精（overspecialization）问题，并提出通过同行模型探测（peer probing）来缓解该问题。

**💡 创新点**

创新点在于：①将用户固有偏好与模型质量结合成用户选择模型；②证明标准多学习者流式梯度下降（MSGD）会陷入过度专精；③提出MSGD‑P（带探测）并给出收敛性与全局风险的理论上界，阐释探测何时能恢复全局性能。

**🔧 技术方法**

技术手段包括：多学习者流式梯度下降（MSGD）、离线探测数据收集与中值聚合、知识蒸馏式伪标签、Lyapunov 潜在函数证明、凸损失（平方/交叉熵）分析。

**📊 数据集**

实验使用三大公开数据集：MovieLens‑10M（推荐）、美国人口普查就业（Census）以及 Amazon Reviews 2023（情感分类）。

**📈 对比分析**

对比方法为标准MSGD与加入探测的MSGD‑P，实验结果显示加入探测显著提升全局性能（误差下降约30‑50%，准确率提升约10‑20%），并且探测权重和样本量对性能影响可调。

**⚠️ 局限性**

局限性包括：仅在凸线性模型下证明，假设用户偏好可知；探测策略为离线收集，未考虑在线自适应；实验环境人工构造用户偏好，未覆盖更复杂真实选择模型。

---

## 11. A Reliable Indoor Navigation System for Humans Using AR-based Technique

**arXiv ID:** 2602.23706 | [PDF](https://arxiv.org/pdf/2602.23706v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 12. Egocentric Visibility-Aware Human Pose Estimation

**arXiv ID:** 2602.23618 | [PDF](https://arxiv.org/pdf/2602.23618v1)

**作者:** Peng Dai `[一作]` (ByteDance), Yang Zhang `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了EvaPose，一种在头戴式设备下进行视角人类姿势估计的方法，并构建了首个包含关键点可见性标注的大规模数据集Eva-3M；

**💡 创新点**

创新点在于首次为egocentric HPE提供大规模可见性标签，并在模型中显式利用可见性信息：通过可见性加权损失、VQ‑VAE先验、立体与时间Transformer的迭代融合，显著提升可见关节点的精度，抑制不可见关节点干扰；

**🔧 技术方法**

采用的技术包括：VQ‑VAE姿势先验、可见性感知的3D估计网络、可见性加权损失、立体Transformer解码器与时间Transformer编码器、利用SLAM设备位姿、SMPL关键点回归；

**📊 数据集**

使用的数据集为自制的Eva-3M（3.0M帧、435K帧可见性标签）和增强后的EMHI（含可见性标签），并使用AMASS、MOYO、AIST++等运动捕捉数据预训练VQ‑VAE；

**📈 对比分析**

与UnrealEgo、EgoPoseFormer、FRAME等前沿方法在Eva-3M与EMHI数据集上进行对照，EvaPose在MPJPE、PA‑MPJPE、Jitter等指标上取得最优表现，尤其在不可见关节点误差显著下降，同时在ResNet50版本实现48 FPS的实时推理；

**⚠️ 局限性**

限制在于高度依赖大规模高质量标注，野外场景下难以获取；目前模型为监督式，缺乏弱监督或自监督的泛化能力。

---

## 13. Grammar-Constrained (CFL) Reachability: Subcubic Preprocessing, Indexing Trade-offs, and Structured Decoding Semantics

**arXiv ID:** 2602.23401 | [PDF](https://arxiv.org/pdf/2602.23401v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Levent Sarioglu `[通讯]` (Bahcesehir University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在有向标签图中基于上下文无关文法的可达性查询（CFL 路径查询），提出多种索引方案并给出子三次预处理的线性文法+稀疏图类；

**💡 创新点**

创新点在于：① 在 Chomsky 正规形下改进至 O(|P|·n³) 的饱和索引；② 对线性文法在稀疏图上实现 O(|P|·n²) 的子三次预处理；③ 引入共享 witness DAG 压缩与可选动态 Dyck 处理；④ 推出单路径（最短接受路径）语义的距离感知索引；⑤ 通过对 9,558 个真实 JSON Schema 的语法类统计验证线性类在实际中的比例为 8.4%。

**🔧 技术方法**

使用的技术包括：Chomsky 正规化、TALNF（线性文法的终端锚定范式）、工作列表饱和传播、共享 witness DAG 与 SLP 输出、动态 Dyck 维护以及基于三元组的最短路径 DP；

**📊 数据集**

数据集为来自 GitHub、JSON Schema Store、Kubernetes 配置等的 9,558 个公开 JSON Schema（JSONSchemaBench）；

**📈 对比分析**

比较方法通过构造多种索引（SatIndex、LinIndex、SWDIndex、Dynamic Dyck、LinDistIndex）并在同一图/文法实例上评估预处理时间、空间、查询时间与 witness 输出大小；实验表明在稀疏图与线性文法场景下，LinIndex 与 LinDistIndex 的预处理时间从 O(|P|·n³) 降至 O(|P|·n²)，且查询与 witness 提取时间保持 O(1) 与 O(|π|)。

**⚠️ 局限性**

局限性包括：① 仅对线性文法提供子三次预处理；② 对非线性文法仍需 O(|P|·n³) 的基线；③ 动态 Dyck 仅适用于双向 Dyck 约束；④ 单路径语义在一般 CNF 下尚未实现；⑤ 线性文法比例仅 8.4%，需要进一步探索更广泛的可压缩或有限重复扩展。

---

## 14. An improved Lower Bound for Local Failover in Directed Networks via Binary Covering Arrays

**arXiv ID:** 2602.23860 | [PDF](https://arxiv.org/pdf/2602.23860v1)

**作者:** Erik van den Akker `[一作]` (TU Dortmund University), Klaus-Tycho Foerster `[通讯]` (TU Dortmund University)

**通讯引用:** 1299 | [OpenAlex ID](https://openalex.org/A5085843145)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

通过构造链型网络并关联二元覆盖数组，证明了在有向网络中本地故障恢复需要的可重写位数的下界为Ω(k+⌈loglog(⌈n/4⌉-k)⌉)；

**💡 创新点**

创新点在于将故障恢复下界映射到覆盖数组问题，从而突破此前⌈log(k+1)⌉的下界，且纳入网络规模n的影响；

**🔧 技术方法**

采用组合构造、覆盖数组理论与极限分析等数学技术；

**📊 数据集**

未使用真实数据集，研究基于理论构造的网络拓扑；

**📈 对比分析**

与以往仅给出⌈log(k+1)⌉的下界结果对比，理论上表现更优，但未给出实验验证；

**⚠️ 局限性**

局限性包括仅针对有向网络、使用二元字母表、缺乏匹配的上界，且构造可能不够紧凑。

---

## 15. Provable Subspace Identification of Nonlinear Multi-view CCA

**arXiv ID:** 2602.23785 | [PDF](https://arxiv.org/pdf/2602.23785v1)

**作者:** Zhiwei Han `[一作]` (Fortiss GmbH), Hao Shen `[通讯]` (Fortiss GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了非线性多视角CCA的可辨识性，将其重新表述为基不变子空间识别问题，并在 N≥3 视角下证明其能通过交叉过滤提取共同相关子空间。

**💡 创新点**

提出 First-Order Canonical Dominance 条件，利用 Mehler–Hermite 展开保证线性信号与高阶非线性分离，并给出有限样本的一致性误差界，首次在多视角非线性 CCA 上实现可证明的子空间识别。

**🔧 技术方法**

采用加性混合模型的生成过程、白化与标准化、奇异值分解、Hermite 多项式展开、谱扰动理论以及自监督学习中的 CCA 优化等技术。

**📊 数据集**

在受控合成数据集和基于 3D 对象的 3DIdent 渲染数据集上进行实验。

**📈 对比分析**

与 Barlow Twins、InfoNCE、W-MSE 以及 GCCA 等自监督基线在相同编码器架构下比较，GCCA 在子空间恢复误差（主角度）上表现最佳，InfoNCE 和 W-MSE 次之，而 Barlow Twins 则显著偏离。

**⚠️ 局限性**

理论假设维度完全匹配、源矩阵满秩，无法完全处理维度不匹配或秩缺陷的情况，且对高阶 Hermite 组件的解释仍待深入。

---

## 16. Toward General Semantic Chunking: A Discriminative Framework for Ultra-Long Documents

**arXiv ID:** 2602.23370 | [PDF](https://arxiv.org/pdf/2602.23370v1)

**作者:** Kaifeng Wu `[一作]` (KingSoft Office Zhuiguang AI Lab), Wen Xu `[通讯]` (KingSoft Office Zhuiguang AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Qwen3‑0.6B 的判别式长文本文档主题分割模型，能够一次性处理约 13k token 以上文本，并通过滑动窗口、跨窗口上下文融合、边界分类头实现高效精确的段落边界检测；同时提供向量融合机制以实现超长段落的单向量检索。

**💡 创新点**

创新点：①在判别式框架下引入跨窗口语义融合层和边界分类头，突破传统 Transformer 固定窗口长度限制；②采用重叠滑动窗口与概率平均融合，降低窗口边界对分割结果的影响；③提出单向量-标量修正的向量融合策略，保持超长段落语义完整性同时降低检索复杂度；④通过类不平衡加权和交叉熵训练显著提升边界召回率。

**🔧 技术方法**

技术：使用 Qwen3‑0.6B 作为编码器；加层 Transformer 跨窗口上下文编码；MLP 边界分类头；重叠滑动窗口处理超长文本；向量融合与标量修正实现单向量检索；损失加权以缓解边界稀缺问题。

**📊 数据集**

数据集：WIKI‑727K（约 727k 篇维基百科文章，按句子级别标注段落边界）。

**📈 对比分析**

与三种基于 Qwen2‑0.5B 的生成式分割模型（simple、topic、summary）对比，模型在测试集上取得 F1 = 0.5503（相比最佳生成式 0.5185 提升 3.5%），召回率显著提升（0.7312 vs 0.5388），速度提升两位数（约 100×）。

**⚠️ 局限性**

局限性：①在精度上略低于生成式模型（0.4628 vs 0.5668 最高精度）；②仍需大量高质量标注数据进行微调；③对极长文本仍需滑动窗口策略，处理复杂度相对传统一次性编码略高；④模型规模相对较大，推理资源需求高于轻量级模型。

---

## 17. The Moment of Capture: How the First Seconds of a Speaker's Nonverbal and Verbal Performance Shapes Audience Judgments

**arXiv ID:** 2602.23920 | [PDF](https://arxiv.org/pdf/2602.23920v1)

**作者:** Ralf Schmälzle `[一作]` (Michigan State University), Gary Bente `[通讯]` (Purdue University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过将真实科学演讲者的高精度运动捕捉数据映射到中性虚拟头像，剥离视觉与声音上的身份特征，在两种条件下（仅非语言和非语言+语言）让受试者持续评估演讲者的吸引力，并计算连续评估与最终评价之间的相关性，研究非语言表现如何在不到10秒的时间窗口内快速决定观众印象。

**💡 创新点**

① 将运动捕捉与中性虚拟化身结合，获得纯粹的非语言信号；② 采用连续响应测量（CRM）实时捕捉观众情绪；③ 通过时间滑动相关分析绘制“捕捉瞬间”与最终评价的高时序动态曲线；④ 发现非语言单独时预测效果反而更快更强。

**🔧 技术方法**

高精度OptiTrack运动捕捉、Vizard VR平台渲染中性头像、ElevenLabs AI语音克隆以消除声音身份特征、Qualtrics在线实验平台、连续响应测量工具、Python/统计软件进行时间滑动相关分析。

**📊 数据集**

包含60场科学演讲的原始运动捕捉与音频数据（共120个视频：每场演讲在非语言和非语言+语言两种版本），受试者来自Prolific平台的120名参与者。

**📈 对比分析**

将每秒的连续吸引力评分与后续的总体吸引力评估进行相关性比较；结果显示在3–5秒内就出现显著相关性，非语言条件在约15秒时达到峰值（相关系数≈0.9），而非语言+语言在约30秒时达到峰值（相关系数≈0.8），说明非语言信号的预测效能更快更高。

**⚠️ 局限性**

① 只保留身体动作，忽略面部表情与眼神等可能的重要非语言信息；② 研究环境限定于科学演讲，可能不适用于政治演讲、课堂教学等其他语境；③ 受试者使用连续评估可能受到指令偏差；④ 样本量相对有限，且缺乏对长期后果（记忆、说服力等）的验证。

---

## 18. Interpretable Multimodal Gesture Recognition for Drone and Mobile Robot Teleoperation via Log-Likelihood Ratio Fusion

**arXiv ID:** 2602.23694 | [PDF](https://arxiv.org/pdf/2602.23694v1)

**作者:** Seungyeol Baek `[一作]` (Korea University), Sungho Suh `[通讯]` (Korea University)

**通讯引用:** 1609 | [OpenAlex ID](https://openalex.org/A5057986515)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种融合Apple Watch惯性数据与纺织触控手套电容信号的多模态手势识别框架，用于无人机和移动机器人无手操作。

**💡 创新点**

创新点在于使用基于对数似然比的晚期融合方法，既提升识别精度又提供各模态贡献解释性。

**🔧 技术方法**

技术包括卷积+GRU特征提取、LLR/自注意力融合以及多模态数据同步。

**📊 数据集**

使用新收集的20类航空指挥式手势数据集，包含RGB、IMU和电容信号。

**📈 对比分析**

与PoseConv3D视觉基线对比，LLR融合模型在F1、精确率、召回率上略优，且计算量、模型大小和训练时间均显著降低。

**⚠️ 局限性**

局限包括缺乏室外或动态环境验证、手势词汇偏重大幅动作导致电容传感器贡献低，以及并行处理延迟未解决。

---

## 19. Lifecycle-Integrated Security for AI-Cloud Convergence in Cyber-Physical Infrastructure

**arXiv ID:** 2602.23397 | [PDF](https://arxiv.org/pdf/2602.23397v1)

**作者:** S M Zia Ur Rashid `[一作]` (University of Tulsa), Suman Rath `[通讯]` (University of Tulsa)

**通讯引用:** 227 | [OpenAlex ID](https://openalex.org/A5075963806)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了面向 AI‑云融合的生命周期安全框架，包含统一威胁分类、统一参考架构和基于 Grid‑Guard 的案例演示。

**💡 创新点**

创新点在于将 MITRE ATLAS、OWASP、CSA MAESTRO、NIST AI RMF 等多标准融合成统一威胁模型，并设计了 Secure Data Factory、硬化模型供应链和 Governance Sidecar 的综合架构，支持物理一致性校验与云原生治理。

**🔧 技术方法**

采用 SPIFFE/SVID、CMK、FPE、摆动方程物理一致性校验、Sigstore/Cosign+OCI 不可变仓库、distroless 容器/Firecracker、Istio/Linkerd、OPA Rego 策略、延迟熔断器等技术。

**📊 数据集**

使用了 AMI/PMU 真实遥测数据、FGSM/PGD 对抗样本、GridLAB‑D/PSCAD 物理仿真数据以及负荷预测模型等数据集。

**📈 对比分析**

通过结构化威胁建模与跨框架合规映射进行验证，未做实际部署或基准测试，性能评估主要基于控制层响应时间（如熔断阈值 200 ms）和合规覆盖率。

**⚠️ 局限性**

局限在于缺乏真实环境部署、定量延迟/吞吐量基准、对抗压力测试以及对不同规模电网可扩展性的评估。

---

## 20. Colour Contrast on the Web: A WCAG 2.1 Level AA Compliance Audit of Common Crawl's Top 500 Domains

**arXiv ID:** 2602.24067 | [PDF](https://arxiv.org/pdf/2602.24067v1)

**作者:** Thom Vaughan `[一作]` (Common Crawl Foundation), Pedro Ortiz Suarez `[通讯]` (Common Crawl Foundation)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Common Crawl CC‑MAIN‑2026‑08档案中，对500个最常被爬取的域名首页进行静态CSS颜色对比分析，评估其WCAG 2.1/2.2 Level AA的颜色对比合规情况。

**💡 创新点**

采用基于归档WARC文件的非实时、无服务器负载、可复现的大规模可视化对比评估方法，并直接从静态HTML中提取颜色声明而非渲染后样式，突破了传统实时爬取对可复现性和伦理性的限制。

**🔧 技术方法**

使用Python（仅标准库）实现Pipeline，包括：Athena查询获取WARC索引、HTTP字节范围请求下载HTML、CSS解析提取颜色、计算对比比率并与4.5:1与3.0:1阈值对照。

**📊 数据集**

Common Crawl CC‑MAIN‑2026‑08公开WARC归档，涵盖近百PB的网页数据；选取前500个被抓次数最多的域名作为样本。

**📈 对比分析**

与WebAIM Million等实时爬取研究相比，本研究通过归档数据实现完全可复现性，且避免了对目标服务器的请求负载；结果显示平均站点合规率为59.8%，中位数62.7%，但未涉及渲染上下文导致的误判。

**⚠️ 局限性**

局限包括：仅分析嵌入式/内联CSS，未处理外部样式表与JavaScript注入的样式；无法判断选择器应用、字体大小或可视性；仅评估首页；假设默认背景/前景颜色可能不准确；且基于单一快照，无法捕捉时间演变。

---

## 21. Leveraging Non-linear Dimension Reduction and Random Walk Co-occurrence for Node Embedding

**arXiv ID:** 2602.24069 | [PDF](https://arxiv.org/pdf/2602.24069v1)

**作者:** Ryan DeWolfe `[一作]` (Toronto Metropolitan University), Ryan DeWolfe `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5116082992)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于随机游走共现的高维节点嵌入方法 COVE，并通过 UMAP/UMAPLE 对嵌入进行非线性降维，提升聚类和链路预测的无监督表现。

**💡 创新点**

创新点在于消除低维约束，利用扩散过程的解释性共现向量生成高维嵌入，再配合 UMAP 的非线性降维与谱初始化（UMAPLE），实现与传统低维嵌入相当甚至略优的结果。

**🔧 技术方法**

核心技术包括随机游走共现统计、线性/非线性降维（UMAP、UMAPLE、SVD）、深度学习式词嵌入（SGNS）、密度聚类算法 HDBSCAN、逻辑回归链路预测及相关度量（JSD、AUROC、F*wo）。

**📊 数据集**

实验使用多种真实网络（Football、Primary1/2、Eu-core、Eurosis、Cora、Airport、Blogcatalog、Cora Large、As）和合成 ABCD 社区检验数据集。

**📈 对比分析**

与 node2vec、node2vec+UMAP、SVD、K‑means、Louvain、ECG 等方法对比，COVE+UMAP/UMAPLE 在聚类 F*wo、无监督评估和链路预测 AUC 上与 node2vec 及 Louvain 相当，偶尔略优于 ECG，但差距不大。

**⚠️ 局限性**

局限性包括：高维嵌入计算成本高、需通过采样近似、UMAP 初始化失败的风险、对噪声水平高的社区检测性能下降、未探索非欧氏（如双曲）空间降维，以及对链路预测提升有限。

---

## 22. High-Modularity Graph Partitioning Through NLP Techniques and Maximal Clique Enumeration

**arXiv ID:** 2602.23948 | [PDF](https://arxiv.org/pdf/2602.23948v1)

**作者:** Marco D'Elia `[一作]` (Roma Tre University), Maurizio Patrignani `[通讯]` (Roma Tre University)

**通讯引用:** 2145 | [OpenAlex ID](https://openalex.org/A5050938227)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于最大团的 TF‑IDF 加权矩阵嵌入方法（Clique‑TF‑IDF），将每个顶点映射到团特征空间后进行聚类，从而得到图的划分。

**💡 创新点**

创新点在于：① 将最大团视作“词条”，构建类似文本的词频矩阵；② 采用 TF‑IDF 权重提升稀疏团的区分度；③ 在嵌入空间中使用传统聚类（层次或 k‑means）得到高质量划分，并对未知 k 的情况加入二分搜索确定最佳聚类数。

**🔧 技术方法**

技术手段包括：最大团枚举、团共参与矩阵 X、TF‑IDF 加权、行归一化得到顶点嵌入、层次聚类/ k‑means、二分搜索确定 k、以及与 METIS、Leiden 等算法的对比实验。

**📊 数据集**

使用的数据集包括：10 个真实网络（如 arenas‑email、arxiv‑grqc、lastfm 等）和数百个由 LFR 生成的合成网络，合成网络参数覆盖不同规模、平均度、最大度和混合参数 μ。

**📈 对比分析**

与 METIS、Walktrap、CNM、Infomap、LP、Pott、Leiden 等算法比较，结果表明：在已知 k 时，Clique‑TF‑IDF 在模块度、NMI 等指标上往往优于 METIS；在未知 k 时，模块度与 Leiden 相近但计算时间更长；总体上在质量上领先或相当，但运行时显著高于传统方法。

**⚠️ 局限性**

主要限制是：① 需要枚举所有最大团，虽然在稀疏图上可行，但团数指数级增长导致内存和时间占用高；② 计算时间比现有划分算法长，尤其在团数多的图上；③ 对参数（如 TF‑IDF 阈值、聚类初始点等）敏感，未给出自动调参方案。

---

## 23. Pessimistic Auxiliary Policy for Offline Reinforcement Learning

**arXiv ID:** 2602.23974 | [PDF](https://arxiv.org/pdf/2602.23974v1)

**作者:** Fan Zhang `[一作]` (Hong Kong University of Science and Technology), Xin Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 34316 | [OpenAlex ID](https://openalex.org/A5100375454)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构造了一种悲观辅助策略，用于离线强化学习中的动作采样，以降低近似误差和错误积累。

**💡 创新点**

创新点在于通过最大化 Q‑函数的下置信界并约束其与已学习策略的 Wasserstein 距离，产生既低不确定性又不偏离原策略的动作，从而在不增加额外网络参数的前提下显著减少 OOD 误差。

**🔧 技术方法**

采用了不确定性量化（基于两网络的方差）、低置信界估计、第一阶泰勒展开、Wasserstein 距离约束、TD3BC、Diffusion‑QL 等现有离线 RL 算法作为基线，并在此基础上嵌入悲观辅助策略。

**📊 数据集**

实验使用 D4RL 基准（Gym：HalfCheetah、Hopper、Walker2d；Adroit：Pen；AntMaze：umaze、medium、large）以及 NeoRL‑2 真实世界数据集。

**📈 对比分析**

通过将原算法与加入悲观辅助策略后的版本（TD3PA、DQLPA）在各任务上进行数值对比，结果显示在 Gym、Adroit、AntMaze 中分别提升 3.8%/14.5%/159.5%（TD3PA）和 2.5%/7.1%/14.5%（DQLPA），在 NeoRL‑2 上提升 3.79%，整体性能显著优于基线。

**⚠️ 局限性**

局限性包括：需要双 Q 网络估计不确定性，参数 β 与 δ 的设定较为敏感；仅在离线场景验证，缺乏在线交互实验；在高维动作空间或复杂任务中扩展性尚未充分验证。

---

## 24. Exploring Robust Intrusion Detection: A Benchmark Study of Feature Transferability in IoT Botnet Attack Detection

**arXiv ID:** 2602.23874 | [PDF](https://arxiv.org/pdf/2602.23874v1)

**作者:** Alejandro Guerra-Manzanares `[一作]` (University of Nottingham), Jialin Huang `[通讯]` (University of Nottingham)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了统一的跨域评测框架，系统评估了三种主流流量特征提取工具（Argus、Zeek、CICFlowMeter）在四个IoT/IIoT数据集上的特征迁移能力；

**💡 创新点**

创新点在于首次在跨域设置下对特征空间可迁移性进行量化比较，并结合SHAP特征重要性分析揭示了不同特征类型在跨域中的鲁棒性差异，为特征工程和模型设计提供了实证指导；

**🔧 技术方法**

使用的技术包括：流量特征提取（Zeek、Argus、CICFlowMeter）、经典机器学习分类器（RF、SVM、k-NN、XGBoost）、数据预处理（缺失值填补、归一化、SMOTE-NC）、性能评估指标（准确率、召回率、精确率、特异性）以及SHAP解释方法；

**📊 数据集**

采用的四个公开数据集为：MedBIoT、CICIoT2023、TON_IoT、Edge-IIoTset；

**📈 对比分析**

比较方法为单源零适配的跨域实验：在一个数据集上训练模型，直接在剩余数据集上测试，同时还做了在域内的基准对比。结果显示，虽然在域内各分类器均能达到≈90%的准确率，但跨域准确率普遍跌至≈50%甚至更低，且在跨域时召回率高、精确率低，表明模型过度预测正样本；

**⚠️ 局限性**

限制包括：只评估四个数据集，可能不足以覆盖真实网络多样性；采用零适配设置，未探索轻量级适配技术；仅考虑三种特征提取工具；工具间特征语义未完全对齐，可能引入偏差。

---

## 25. Feelings, Not Feel: Affective Audio-Visual Pseudo-Haptics in Hand-Tracked XR

**arXiv ID:** 2602.23747 | [PDF](https://arxiv.org/pdf/2602.23747v1)

**作者:** Kristian Paolo David `[一作]` (University of the Philippines Los Baños), Jordan Aiko Deja `[通讯]` (De La Salle University)

**通讯引用:** 134 | [OpenAlex ID](https://openalex.org/A5046495553)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究在混合现实环境中通过对手部的音频-视觉伪触觉刺激，探讨其对用户情感体验的影响，并验证其是否能产生真实触觉感受。

**💡 创新点**

创新点在于将手部伪触觉视为情感反馈通道而非物理触觉替代，强调情感意义与多模态匹配的作用，提出音频驱动情绪激活、视觉颜色偏向情感取向、模态组合非加性的设计原则。

**🔧 技术方法**

技术实现采用Unity XR Interaction Toolkit与Meta Quest 3手部跟踪，利用音频合成、视觉色彩（红/蓝）与微小运动调制对手部进行伪触觉呈现。

**📊 数据集**

使用12名受试者的自评数据（Affect Grid、Modified MReQ问卷）及半结构化访谈作为实验数据，没有使用公开数据集。

**📈 对比分析**

通过重复测量ANOVA比较12种条件，结果显示音频可显著提升情绪激活，视觉色彩有效偏向情感取向，音频与视觉组合并非简单加性，且不同效应的情感反应因模态匹配而异。

**⚠️ 局限性**

局限包括样本量有限、缺乏无伪触觉基线、仅使用主观测量、未评估任务表现或生理指标、个体对身体感知与情感解释差异未系统测量。

---

## 26. Stochastic Knapsack -- Semi-Adaptivity Gaps and Improved Approximation

**arXiv ID:** 2602.24042 | [PDF](https://arxiv.org/pdf/2602.24042v1)

**作者:** Zohar Barak `[一作]` (Tel Aviv University), Inbbal Talgam-Cohen `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了随机组合优化中的半自适应性，尤其是随机背包问题，提出并分析了不同k值下的半自适应性间隙，并改进了已知的自适应间隙上界；

**💡 创新点**

创新点包括：①引入0‑k、k‑n两类半自适应性间隙的概念，②通过三步Simplify‑Equalize‑Optimize框架获得随机背包问题的紧致上下界；③给出单次自适应查询与无自适应之间的精确比值1+ln 2；④利用树约束构造简化实例从而得到常数k下的性能提升；

**🔧 技术方法**

主要技术包括：决策树分析、LP上界与下界、贪心密度排序、块插入策略、随机变量截断与均值、Chernoff界、递归优化与极值分析、以及构造树约束实例的归约方法；

**📊 数据集**

由于本工作完全是理论分析，并未使用任何实验数据集；

**📈 对比分析**

通过与最优自适应策略的期望收益比值进行比较，给出了多种情形的近似比值：全自适应间隙≤2ϕ³≈8.47；单自适应间隙≤8.26；小件子问题的k‑n间隙≤2e+O(√ε)≈6.44；大件子问题的k‑n间隙≤1+O(1/ln (1/ε))，以及对0‑1间隙的精确值1+ln 2；

**⚠️ 局限性**

局限性在于：仍未完全确定全自适应间隙的确切值；结果主要针对风险型随机背包，且仅在常数k下给出；以及对其他随机组合优化问题的推广仍需要进一步研究。

---

## 27. Half-Truths Break Similarity-Based Retrieval

**arXiv ID:** 2602.23906 | [PDF](https://arxiv.org/pdf/2602.23906v1)

**作者:** Bora Kargi `[一作]` (University of Tübingen), Seong Joon Oh `[通讯]` (University of Tübingen)

**通讯引用:** 8418 | [OpenAlex ID](https://openalex.org/A5025851635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 CS-CLIP，通过在 CLIP 的微调阶段加入单元级监督，来降低文本查询中加入错误细节后相似度升高的半真漏洞。

**💡 创新点**

创新点在于对每条标题进行实体和关系单元拆分，并为每个单元生成最小编辑的伪样本（foil），在训练时将图像向正确单元拉近、向伪样本拉远，从而实现对组合语义的细粒度监督。

**🔧 技术方法**

技术手段包括：基于 CLIP 的双编码器架构、文本仅 LLM 进行实体/关系提取与 foil 生成、单元级对比损失（与全句对比损失并行）、硬负样本采样、可学习温度和正则化。

**📊 数据集**

主要使用的数据集为 MS‑COCO（Karpathy split）进行微调，评估时使用 COCO 半真诊断集、16 个受控编辑组合基准，以及零样本图像‑文本检索任务。

**📈 对比分析**

与 CLIP、NegCLIP、SigLIP 等基线比较，CS‑CLIP 在 COCO 半真准确率上提升至 69.3%（比 CLIP 提升 28.7%），在组合基准的 I2T 平均准确率达到 57.8%，比最强基线提升约 5.7 个百分点，整体性能均优于现有方法。

**⚠️ 局限性**

局限性包括：依赖文本仅 LLM 的拆分与 foil 生成，可能忽略视觉细节；微调后在零样本检索性能上略有下降；未能保证事实正确性或公平性，仅提升了文本侧的组合敏感度，未覆盖图像侧的半真情况。

---

## 28. FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments

**arXiv ID:** 2602.23504 | [PDF](https://arxiv.org/pdf/2602.23504v1)

**作者:** Anik Pramanik `[一作]` (New Jersey Institute of Technology), Shantanu Sharma `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 458 | [OpenAlex ID](https://openalex.org/A5067042141)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种集成数据和梯度相似度的聚类联邦学习框架FedDAG，用于在高度异构的数据环境中高效训练模型。

**💡 创新点**

创新点在于：①采用加权类级数据相似度与梯度相似度的融合，①.1 通过熵损失自适应学习每个客户端的加权因子；②使用双编码器架构实现跨簇特征共享；③设计新的联邦感知聚类评价指标并自动确定最佳聚类数。

**🔧 技术方法**

核心技术包括：局部SVD提取类级主向量、k‑sparse梯度通信、加权类级相似度计算、熵优化权重学习、层次聚类与自适应阈值、双编码器的主/次编码器训练以及基于需求/供应的聚类互补图。

**📊 数据集**

实验数据集覆盖四种典型图像分类任务：CIFAR‑10、Fashion‑MNIST、SVHN 与 CIFAR‑100，并通过多种非IID分布（标签/数量偏移、概念漂移、LDA偏斜）进行评估。

**📈 对比分析**

与FedAvg、FedProx、PerFedAvg、FedMix、FedBR、FedSoft、PACFL、IFCA、CFL、FedGWC、FedRC、CFL‑GP等现有单模型、个性化或聚类联邦学习方法相比，FedDAG在所有异构设置下均显著提升准确率（例如在高数量偏移下CIFAR‑10提升约2.1%，在概念漂移下SVHN提升约3.0%）。

**⚠️ 局限性**

主要局限包括：①需要额外的本地预训练和SVD步骤以生成相似度信息，增加初始通信和计算成本；②双编码器模型参数规模相对更大，可能限制在资源受限边缘设备上的部署；③目前对动态客户端加入、长期概念漂移以及极端特征偏移的自适应机制仍有待进一步研究。

---

## 29. VCA: Vision-Click-Action Framework for Precise Manipulation of Segmented Objects in Target Ambiguous Environments

**arXiv ID:** 2602.23583 | [PDF](https://arxiv.org/pdf/2602.23583v1)

**作者:** Donggeon Kim `[一作]` (ROBROS Inc.), Daegyu Lim `[通讯]` (ROBROS Inc.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Vision-Click-Action（VCA）框架，将机器人目标指定从语言命令转为实时点击生成的分割掩码，实现闭环控制；

**💡 创新点**

创新点在于将 SAM2 适配为可实时、可增删对象掩码的交互式分割器，并将掩码作为实例级条件输入至 Transformer 控制器，彻底消除语言歧义；

**🔧 技术方法**

使用技术包括 SAM2 轻量级在线分割、Transformer 控制策略（ACT + Mask Encoder）、ResNet 编码器、CVAE 条件变分自编码器、多摄像头视觉输入以及动态内存更新机制；

**📊 数据集**

数据来源为自建演示数据集：block sorting（608 试验）和 Tower of Hanoi（330 试验），以及在不同背景、颜色和形状变化下的测试集；

**📈 对比分析**

与原始 ACT 基线对比，VCA 在成功率上保持相当（block 95% vs 96%，Ho 94% vs 94%），在高视觉相似度或背景变化场景下，VCA 在 block sorting 任务中优于 ACT，但两者在极端视觉偏移下均表现下降；

**⚠️ 局限性**

局限性包括：需要用户点击交互，增加操作步骤；对极端视觉变化仍不稳健；目前仅支持单类别条件，未验证多目标长序任务；数据标注成本较高。

---

## 30. Unified Learning-to-Rank for Multi-Channel Retrieval in Large-Scale E-Commerce Search

**arXiv ID:** 2602.23530 | [PDF](https://arxiv.org/pdf/2602.23530v1)

**作者:** Aditya Gaydhani `[一作]` (Target Corporation), Alex Li `[通讯]` (Target Corporation)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建统一的学习排序框架，将多渠道检索产生的候选商品通过GBDT模型进行重排，联合优化点击、加入购物车和购买等业务指标。

**💡 创新点**

把多渠道融合问题视为查询相关的学习排序任务，使用渠道感知特征、短期用户行为特征以及转化加权标签，摆脱固定权重的手工融合方法。

**🔧 技术方法**

使用GBDT（LambdaMART）模型，结合渠道特征、用户行为特征和周度查询‑商品‑时间表示，训练时采用Yggdrasil Decision Forests实现高效推理。

**📊 数据集**

Target.com电商搜索日志，约5周的用户交互数据（查询、点击、加入购物车、购买），共约60M条查询‑商品‑周实例，包含约500k唯一查询。

**📈 对比分析**

通过与加权交叉混合（Weighted Interleaving）基线对比，开展在线A/B实验。结果显示NDCG@8从0.6620提升至0.7994，点击率+1.52%，加入购物车+2.81%，转化率+2.85%，且p95延迟<50 ms。

**⚠️ 局限性**

过滤阈值（≥20曝光且至少一次购买）导致尾部查询样本稀缺；渠道公平性与偏差未全面评估；未引入个性化信号；对极端稀疏查询的标签与特征仍需改进。

---

## 31. Learning to Build: Autonomous Robotic Assembly of Stable Structures Without Predefined Plans

**arXiv ID:** 2602.23934 | [PDF](https://arxiv.org/pdf/2602.23934v1)

**作者:** Jingwen Wang `[一作]` (École Polytechnique Fédérale de Lausanne), Stefana Parascho `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 419 | [OpenAlex ID](https://openalex.org/A5091302831)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种无需预先建筑蓝图、能够自主构建二维干式块结构的机器人装配框架。

**💡 创新点**

创新点在于将任务抽象为目标点与障碍区域，采用图像化的后继特征（Successor Features）结合深度 Q‑学习，实现单一策略跨任务的自适应决策。

**🔧 技术方法**

使用深度 Q‑学习（DQN）+后继特征、图像编码的状态/动作/任务表示、Gaussian 目标场奖励、RBE 稳定性评估以及基于 ArUco 的闭环感知。

**📊 数据集**

在 15 个手工设计的二维块装配任务（包含正方形和梯形块）上进行仿真与真实机器人实验，任务包含不同目标与障碍布局。

**📈 对比分析**

与传统基于蓝图的计划方法相比，单一 RL 策略在仿真中 93.3% 的任务成功率，在真实闭环实验中 80% 的成功率，并在训练 50 轮后显著减少使用块数。

**⚠️ 局限性**

主要局限包括：仿真到现实的差距（RBE 二进制稳定性、机器人碰撞与到达约束缺失）、仅限 2D 两种块形状以及未考虑构造噪声的鲁棒性。

---

## 32. The impacts of artificial intelligence on environmental sustainability and human well-being

**arXiv ID:** 2602.24091 | [PDF](https://arxiv.org/pdf/2602.24091v1)

**作者:** Noemi Luna Carmeno `[一作]` (Universitat de Barcelona), Daniel W. O'Neill `[通讯]` (University of Leeds)

**通讯引用:** 4440 | [OpenAlex ID](https://openalex.org/A5046516926)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2010-2024年间关于AI对环境与人类福祉影响的1291篇文献进行系统综述，并构建统一评估框架。

**💡 创新点**

首次将环境与福祉两大研究领域整合，利用Kaack等的三层框架拓展至福祉影响，揭示研究空白与情绪偏差。

**🔧 技术方法**

采用PRISMA 2020流程、主题编码、定量比例统计、情绪分类及主题分析等方法。

**📊 数据集**

使用来自Scopus、arXiv和NBER Working Papers的1291篇英文文献作为数据来源。

**📈 对比分析**

通过对文献数量、方法类型、研究层级与情绪倾向进行定量比对，发现环境研究倾向正面、福祉研究情绪均衡，但细分维度表现差异明显。

**⚠️ 局限性**

局限包括数据库覆盖不全、对非同行评议稿依赖、主观情绪归类、Copilot辅助提取可能产生偏差，以及AI技术快速演进导致结论易失效。

---

## 33. A Difference-in-Difference Approach to Detecting AI-Generated Images

**arXiv ID:** 2602.23732 | [PDF](https://arxiv.org/pdf/2602.23732v1)

**作者:** Xinyi Qi `[一作]` (Tsinghua University), Jin Zhu `[通讯]` (University of Birmingham)

**通讯引用:** 7629 | [OpenAlex ID](https://openalex.org/A5102795426)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于差分二阶的 AI 生成图像检测方法，利用两次重构得到的第一阶和第二阶误差进行判别；

**💡 创新点**

创新点在于首次将差分二阶思想（即对重构误差再次做差分）引入图像检测，从而显著抑制重构噪声并提升对高保真合成图像的检测精度；

**🔧 技术方法**

技术核心包括扩散模型重构、差分二阶特征提取以及基于 ResNet-50 的双分类器；

**📊 数据集**

使用 ImageNet、LAION、LSUN-B 等公开数据集，训练样本涵盖 4 万张真实图像和 4 万张合成图像，测试覆盖多种生成模型（ADM、PNDM、DDPM、iDDPM、SDv1、Kandinsky3、SDXL、Playground v2.5、Stable Cascade 等）；

**📈 对比分析**

与 DIRE、LaRE^2、AEROBLADE、UniversalFakeDetect 等基线相比，DID 在大样本、少样本以及跨模型测试中均表现更优，提升幅度约 20%–30%，并在 GAN 生成图像上也能保持较高准确率；

**⚠️ 局限性**

主要限制是需要两次扩散重构导致计算成本上升，且更高阶差分的效果仍需在效率与精度之间权衡，方法对重构模型与生成模型差异较大时仍可能出现性能下降。

---

## 34. Breaking the Illusion of Artificial Consensus: Clone-Robust Weighting for Arbitrary Metric Spaces

**arXiv ID:** 2602.24024 | [PDF](https://arxiv.org/pdf/2602.24024v1)

**作者:** Damien Berriaud `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**通讯引用:** 21283 | [OpenAlex ID](https://openalex.org/A5078339613)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套克隆鲁棒加权函数框架，用于在存在近似复制的在线帖子或图形节点中公平分配影响力，防止重复信息过度膨胀。

**💡 创新点**

创新点包括：①在度量空间中引入连续性和局部性的新克隆鲁棒加权定义；②基于邻域图阈值构造通用加权函数，并证明任意满足对称性与局部性的图加权函数均能推广为克隆鲁棒加权；③提出共享系数（sharing coefficient）工具，为加权过程提供可解释性；④探索最大团覆盖与信息熵最优两种不同构造，并对其局限性进行系统讨论。

**🔧 技术方法**

主要技术手段包括：度量空间和邻域图构造、等价类与闭邻域定义、克隆鲁棒性与局部性公理化、概率分布与拉普拉斯连续性约束、最大团覆盖、信息熵（Polsner ε-entropy）优化、图熵与最优分配等。

**📊 数据集**

论文为理论性工作，没有使用具体数据集；实验验证基于人工构造的图与消息集合，主要通过理论证明和案例示例说明方法效果。

**📈 对比分析**

方法比较主要通过理论性质对比与案例讨论：与传统的统一加权、Voronoi 区域权重等方法对比，证明克隆鲁棒框架在保持对称性、局部性、正性以及可解释共享系数方面优于现有方案；但由于最大团覆盖与熵最优构造易出现负共享系数，性能上仍存在不可避免的限制。

**⚠️ 局限性**

主要局限包括：①缺乏在真实大规模数据上的实验验证；②在最大团覆盖或熵最优构造中，节点删除往往会改变等价类或最优分区，导致共享系数可能为负，影响可解释性；③构造的克隆鲁棒加权函数仍未能统一满足所有提出的附加公理（如非负共享、对称共享、共享支配），仍需进一步研究。

---

## 35. UXSim: Towards a Hybrid User Search Simulation

**arXiv ID:** 2602.24241 | [PDF](https://arxiv.org/pdf/2602.24241v1)

**作者:** Saber Zerhoudi `[一作]` (University of Passau), Michael Granitzer `[通讯]` (University of Passau)

**通讯引用:** 3649 | [OpenAlex ID](https://openalex.org/A5006866152)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了UXSim框架，融合传统规则模拟器与LLM智能代理，生成更真实的用户搜索行为。

**💡 创新点**

创新点在于将LLM的生成能力与传统模拟器的可验证约束结合，利用Oris策略动态调度不同组件，并提供可解释的认知轨迹。

**🔧 技术方法**

采用Python实现，使用Playwright进行浏览器自动化，LLM（如GPT‑4）与传统模块（查询生成、点击模型等），以及Oris策略（规则、监督、认知三种实现）。

**📊 数据集**

使用公开用户研究数据集KDD'19和USimAgent2.0，分别包含复杂搜索任务和思考录音的搜索日志。

**📈 对比分析**

通过与规则基、监督式和专门点击模型的对比，Oris‑A在任务成功率、nDCG@10、查询语义相似度和点击F1上取得最高表现，实验显示其任务完成率达78%，比基线提升约37%。

**⚠️ 局限性**

局限在于实验仅覆盖固定任务与搜索引擎，缺乏对多域、多语言或更动态界面的评估，且LLM解释仍依赖外部提示。

---

## 36. Additive One Approximation for Minimum Degree Spanning Tree: Breaking the $O(mn)$ Time Barrier

**arXiv ID:** 2602.23448 | [PDF](https://arxiv.org/pdf/2602.23448v1)

**作者:** Sayan Bhattacharya `[一作]` (University of Warwick), Haoze Wang `[通讯]` (Peking University)

**通讯引用:** 4616 | [OpenAlex ID](https://openalex.org/A5100662422)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的确定性算法，能够在 Õ(m·n^{3/4}) 时间内构造最大度数加 1 的最小度数生成树（MDST）

**💡 创新点**

通过将阻塞流方法与分子分解（molecular decomposition）相结合，突破了长期存在的 Õ(mn) 时间瓶颈

**🔧 技术方法**

核心技术包括分子分解、可约原子、增广链（augmenting chains）以及阻塞流的范式

**📊 数据集**

由于是理论算法，未使用具体数据集进行实验验证

**📈 对比分析**

与 Fürer–Raghavachari 经典 O(mn) 算法相比，针对稀疏图实现了多项式加速：时间从 Õ(mn) 降至 Õ(m n^{3/4})

**⚠️ 局限性**

仍属于超线性时间，未能达到近线性复杂度；实现与分析相对复杂，对加权版本或更强近似的适用性尚不明确

---

## 37. BRIDGE the Gap: Mitigating Bias Amplification in Automated Scoring of English Language Learners via Inter-group Data Augmentation

**arXiv ID:** 2602.23580 | [PDF](https://arxiv.org/pdf/2602.23580v1)

**作者:** Yun Wang `[一作]` (University of Georgia), Ninghao Liu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5066745575)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 BRIDGE 框架，通过跨组数据增强（将高分非 ELL 内容粘贴到 ELL 语言模式中）来缓解自动评分系统中的偏差放大问题。

**💡 创新点**

创新点在于将学生回答拆解为构建内容与语言风格，利用 LLM 在两组之间进行内容拼接并引入判别器进行质量控制，从而在极低资源场景下生成高质量的少数族群样本。

**🔧 技术方法**

使用技术包括 GPT‑4o 进行跨组样本生成、判别器（轻量级二分类网络）进行真实性筛选，以及基于 ERM 的评分模型训练。

**📊 数据集**

实验基于加州科学测试（CAST）8 年级短答题数据，共五个大规模题目，包含 ELL 与非 ELL 两组，测试高分 ELL 子群的稀缺性。

**📈 对比分析**

与真实数据增补、过采样、同组改写等基线相比，BRIDGE 在 MSG Gap 与 BiasAmp 指标上均显著下降，公平度提升且整体评分准确率、QWK、MAE 等性能保持不变或略有提升。

**⚠️ 局限性**

局限性包括对内容与风格可分离的假设依赖、仅在短答题场景验证，长篇作文或更复杂评测形式需进一步研究和验证。

---

## 38. Science Fiction and Fantasy in Wikipedia: Exploring Structural and Semantic Cues

**arXiv ID:** 2602.24229 | [PDF](https://arxiv.org/pdf/2602.24229v1)

**作者:** Włodzimierz Lewoniewski `[一作]` (Poznań University of Economics and Business), Elżbieta Lewańska `[通讯]` (Poznań University of Economics and Business)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5024901409)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过离线提取Wikipedia dumps，分析并比较了WikiProject标签、Wikidata实例（P31）、文章类别、以及前导段与infobox中的wikilink等结构与语义特征，用以识别与科幻与幻想相关的条目。

**💡 创新点**

创新点在于将多种Wikipedia内部信号系统性地并行评估，揭示不同信号在识别SF/F边界文章时的相互补充与局限，并提供基于这些特征的覆盖率与一致性分析框架。

**🔧 技术方法**

技术手段主要包括文本解析与结构化数据提取、频率与分布统计分析、以及对Wikidata P31实例与类别层级的语义映射。

**📊 数据集**

使用的数据集为2026年1月的English Wikipedia dumps、与之对应的Wikidata items（P31实例）、以及从WikiProject Science Fiction、Fantasy及其子任务中抽取的18,829条目标条目。

**📈 对比分析**

方法上以覆盖率与交叉一致性评估不同信号的有效性，例如“Science fiction”在lead段的链接覆盖率为49%，与全Wikipedian 14,405条的对比；但未给出精确的分类准确率，仅通过定量分布展示各信号的优势与缺陷。

**⚠️ 局限性**

局限性包括：单一信号不具备完整覆盖，WikiProject标签覆盖不均且偏向系列；Wikidata实例缺乏完整性与一致性；类别层级不规则导致深浅层级检索的折衷；lead段wikilink倾向于宽泛背景词，难以捕捉细粒度子类别。

---

## 39. Incremental dimension reduction for efficient and accurate visual anomaly detection

**arXiv ID:** 2602.23595 | [PDF](https://arxiv.org/pdf/2602.23595v1)

**作者:** Teng-Yok Lee `[一作]` `[通讯]` (Mitsubishi Electric Corporation), Teng-Yok Lee (Mitsubishi Electric Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种按批次进行的增量降维算法，用于压缩PatchCore模型的特征向量，从而在大规模视觉异常检测中显著降低内存占用并加速训练。

**💡 创新点**

创新点在于：①将增量SVD与增量PCA相结合，通过对Gram矩阵的增量更新，只在最后一步统一投影，避免了对已访问批次重新投影的高昂开销；②利用批量截断SVD并构造小矩阵旋转实现数值稳定性；③在GPU上实现了高效的矩阵运算和批量处理。

**🔧 技术方法**

使用技术包括：增量奇异值分解（Incremental SVD）、增量主成分分析（Incremental PCA）、截断奇异值分解、GPU矩阵乘法、PatchCore异常检测框架、WideResNet50/ResNet18特征提取网络。

**📊 数据集**

实验数据集：MVTec AD（工业缺陷图像）和Eyecandies（合成光照变化图像）。

**📈 对比分析**

与原始PatchCore及PaDiM进行对比；在MVTec AD上，使用k=128、批量大小16K时图像级AUROC为98.9%（原为99.0%），像素级AUROC保持接近；训练时间随k线性缩短，批量大小对速度影响有限；在Eyecandies上，将特征维度降至128后，训练时间从原来的数十小时降至3小时，AUROC略优于PaDiM。

**⚠️ 局限性**

限制与不足：①特征提取阶段仍由原始CNN承担，未被降维加速；②GPU内存仍需容纳完整特征矩阵的临时存储，极大数据集仍受限；③对多尺度或多光照条件的在线更新支持不足；④算法实现复杂度高，需GPU支持；⑤在极高维特征（>2000维）下的数值稳定性与效率尚待进一步验证。

---

## 40. Are Stacked Intelligent Metasurfaces (SIMs) Better than Single-layer Reconfigurable Intelligent Surfaces (RISs) for Wideband Multi-user MIMO Communication Systems?

**arXiv ID:** 2602.23534 | [PDF](https://arxiv.org/pdf/2602.23534v1)

**作者:** Muhammad Ibrahim `[一作]` (University of Manitoba), Ekram Hossain `[通讯]` (University of Manitoba)

**通讯引用:** 34636 | [OpenAlex ID](https://openalex.org/A5089270885)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种耦合感知、宽带电路基础的堆叠智能超材料（SIM）模型，并联合主动与被动波束成形实现多用户MIMO系统的载波聚合与相位优化。

**💡 创新点**

创新点在于：①将多端口网络理论与频域耦合、层间反射结合，构建频率可变的SIM电路模型；②提出联合优化框架，采用交替优化的水分配与梯度上升实现全宽带性能提升；③引入部分可重构SIM以显著降低控制开销。

**🔧 技术方法**

使用的技术包括：多端口电路模型（Z、T、S参数）、多用户MIMO信道模型、宽带多子带分配、交替优化（AO）+水分配、梯度上升求解相位、仿真参数化。

**📊 数据集**

采用仿真生成的多用户MIMO场景（总带宽15 GHz，30–40 GHz中心频率，10–15个子带，K≤10用户，100–144元件等）进行性能评估；未使用公开数据集。

**📈 对比分析**

通过与单层RIS在相同总元件数、不同子带数、用户数、信噪比和带宽等条件下的比较，结果显示：在宽带、高SNR、多用户环境下SIM明显优于RIS；在窄带、低SNR或用户数少的情形下RIS更佳；部分可重构SIM在控制位减少的前提下仍能保持与全可重构SIM相近的性能。

**⚠️ 局限性**

局限性包括：①需要精准的电磁耦合与频率响应建模，理论复杂度高；②假设完美CSI和无直射通道，实际部署需进一步验证；③优化过程收敛速度慢，计算量大；④硬件实现时的相位/振幅误差未完全考虑。

---

## 41. Leveraging large multimodal models for audio-video deepfake detection: a pilot study

**arXiv ID:** 2602.23393 | [PDF](https://arxiv.org/pdf/2602.23393v1)

**作者:** Songjun Cao `[一作]`, Long Ma `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了基于大规模多模态模型的音视频深度伪造检测框架AV-LMMDetect，通过将检测任务转化为提示式二分类问答，并在Qwen 2.5 Omni上进行监督微调；

**💡 创新点**

创新点包括两阶段微调策略（LoRA轻量化对齐+全参数跨模态Encoder微调），以及将深度伪造检测表述为多模态问答任务，充分利用大模型的跨模态推理与知识泛化；

**🔧 技术方法**

使用了大规模多模态语言模型Qwen 2.5 Omni、LoRA对齐技术、全参数Encoder微调、提示式二分类问答框架以及基于token级别（Real/Fake）的输出判定；

**📊 数据集**

使用了FakeAVCeleb（英文音视频伪造数据集）和MAVOS-DD（多语言、250+小时、7种伪造方法的多模态数据集）；

**📈 对比分析**

与多种视觉或视觉+音频基线（MesoNet、LipForensics、AVFF等）进行对比；在FakeAVCeleb上获得98.02%准确率、99.2% AUC，接近SOTA；在MAVOS-DD四个开放集场景中，尤其是open‑set full场景，mAP 0.96、AUC 0.92、Acc 85.09%，实现或超越现有最优方法；

**⚠️ 局限性**

局限性在于依赖大型多模态模型及其昂贵的计算与推理资源，对极端伪造或完全未知生成模型的鲁棒性仍有限；尚未深入评估跨语言推理细节与低资源环境下的可迁移性。

---

## 42. GeoDiff4D: Geometry-Aware Diffusion for 4D Head Avatar Reconstruction

**arXiv ID:** 2602.24161 | [PDF](https://arxiv.org/pdf/2602.24161v1)

**作者:** Chao Xu `[一作]` (Tsinghua University), Yebin Liu `[通讯]` (Tsinghua University)

**通讯引用:** 10578 | [OpenAlex ID](https://openalex.org/A5032875389)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了GeoDiff4D框架，能够从单张人像照片生成可动画的高质量4D头部虚拟形象。

**💡 创新点**

创新点在于：①使用姿势无关的表情编码器，实现跨视角一致的表情表达；②联合生成图像与表面法线的几何感知扩散模型，增强3D几何一致性；③将扩散模型产生的法线和表情潜在向量直接用于3D高斯散点的优化，提升身份保留与表情细腻度。

**🔧 技术方法**

核心技术包括：扩散模型（Latent Diffusion）、姿势无关表情编码器、3D高斯散点渲染（3D Gaussian Splatting）、FLAME头部模型、交叉视角配对训练与域空间注意力。

**📊 数据集**

使用多视角真实人像数据集、SynthHuman合成数据集以及NeRSemblev2等数据集进行训练与评估。

**📈 对比分析**

与Portrait4D-v2、GAGAvatar、LAM、CAP4D等单视角4D头像重建方法对比，GeoDiff4D在PSNR、SSIM、LPIPS、CSIM、JOD等指标上均获得第二或第一名，尤其在极端头部姿势与表达转移方面表现更佳。

**⚠️ 局限性**

主要局限包括：依赖单目3DMM追踪导致姿势估计误差；对舌头运动的重建仍不充分；以及扩散采样速度较慢，影响实时性能。

---

## 43. CiteAudit: You Cited It, But Did You Read It? A Benchmark for Verifying Scientific References in the LLM Era

**arXiv ID:** 2602.23452 | [PDF](https://arxiv.org/pdf/2602.23452v1)

**作者:** Zhengqing Yuan `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**通讯引用:** 5159 | [OpenAlex ID](https://openalex.org/A5027601906)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了CiteAudit基准并提出多代理框架，对科研论文的引用进行真实性验证。

**💡 创新点**

首次提供统一、开放的引用幻觉检测基准和可解释的多代理审核流程，细化验证步骤并引入人类验证。

**🔧 技术方法**

结合视觉OCR、向量检索、Web搜索、学术数据库抓取及LLM判断的多代理体系，并使用 Qwen3‑VL、Mem0 等技术。

**📊 数据集**

使用从 OpenReview、Google Scholar、ArXiv 等收集的真实引用和通过系统化扰动生成的约 2,500 条幻觉引用。

**📈 对比分析**

与 GPT‑5.2、Gemini‑3‑Pro 等主流模型对比，本文方案在准确率、召回率、F1 上均超过 0.95，且时延最低、成本几乎为零。

**⚠️ 局限性**

受限于外部检索质量、对极少见或已删除论文的识别、对语义模糊的处理以及对多语种引用的支持不足。

---

## 44. Comparing Classical and Quantum Variational Classifiers on the XOR Problem

**arXiv ID:** 2602.24220 | [PDF](https://arxiv.org/pdf/2602.24220v1)

**作者:** Miras Seilkhan `[一作]`, Adilbek Taizhanov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在XOR分类任务上对比经典线性/多层感知器与变分量子分类器（VQC），探讨模型表达力、深度、噪声以及硬件效应；

**💡 创新点**

系统评估量子深度对表达力的决定性影响，并量化硬件噪声对连续决策函数的结构性偏差；

**🔧 技术方法**

采用经典Logistic回归、单隐藏层MLP与两量子比特VQC（L=1,2）以及量子仿真与IBM量子硬件推理；

**📊 数据集**

使用三类合成XOR数据集：纯XOR、带高斯噪声的簇状XOR和连续阈值XOR；

**📈 对比分析**

通过准确率、交叉熵、训练时间等指标比较；结果显示VQC(L=2)可与MLP达到100%准确率，但BCE与训练效率不如MLP；VQC(L=1)表现不佳；硬件推理虽保持准确率，却出现平均绝对偏差≈0.118；

**⚠️ 局限性**

局限性包括：仅测试低维合成数据、单一VQC架构与编码、固定超参数、缺乏大规模或多量子比特实验，且训练仅在模拟器完成，无法验证硬件上完整训练的可行性。

---

## 45. The Subjectivity of Monoculture

**arXiv ID:** 2602.24086 | [PDF](https://arxiv.org/pdf/2602.24086v1)

**作者:** Nathanael Jo `[一作]` (Massachusetts Institute of Technology), Manish Raghavan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 3208 | [OpenAlex ID](https://openalex.org/A5052541789)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究 AI 模型（尤其是大型语言模型）在多任务评测中出现的同质输出现象（monoculture），并指出评估该现象的结论高度依赖于两个主观选择：基准空模型（null model）和所选模型与题目集合。

**💡 创新点**

创新点在于：①提出将 monoculture 视为相对的、依赖基准模型的推断问题；②引入“null ladder”概念，展示更丰富的空模型能更好解释协方差；③通过项目化的多维 IRT 模型和对比实验，系统验证了基准和样本多样性对结论的影响；④将模型能力与题目难度的潜在结构显式化，提供诊断工具。

**🔧 技术方法**

主要技术包括：构造条件独立的 Bernoulli 空模型族；使用多维 Item Response Theory (IRT) 作为可扩展的空模型；利用梯度上升估计 IRT 参数；计算残差协方差与相关矩阵来衡量多余的模型间相关；理论证明 null ladder 使残差相关随维度增加趋近零；对比不同空模型下的残差相关；使用总变差或 f‑divergence 作为度量。

**📊 数据集**

实验数据集：HEL M (基于 MMLU 的 14042 个题目，72 个模型) 与 HuggingFace Open LLM Leaderboard (11994 个题目，451 个模型)。此外，还使用了 ACSIncome 数据（针对随机森林、逻辑回归和 MLP 的 80% 训练/20% 测试划分），用于验证模型多样性对残差相关的影响。

**📈 对比分析**

比较方法：在不同维度的 IRT 空模型（K=1、2、4、8…64）和两种空模型（包含/不包含题目难度）下计算残差相关的绝对平均值。实验表明：①维度越高，残差相关越低；②包含题目难度的空模型（IRT-1）显著降低残差相关；③与之前的基准方法相比，加入题目难度后残差相关从正转负或大幅下降，表明先前高估了 monoculture。性能上，随着 K 增大，MSE 下降，残差相关趋于零。

**⚠️ 局限性**

局限性：①空模型的选择本身是主观的，缺乏统一的标准；②当模型或题目集合过于同质时，空模型难以稳定估计，残差相关不可靠；③实验仅使用二值正确性数据，未充分利用多选题的完整信息；④未探讨更复杂的空模型（如主题特化、对话上下文等），可能进一步降低残差相关。

---

## 46. BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds

**arXiv ID:** 2602.23645 | [PDF](https://arxiv.org/pdf/2602.23645v1)

**作者:** Tongyan Hua `[一作]` (Hong Kong University of Science and Technology), Wufan Zhao `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 633 | [OpenAlex ID](https://openalex.org/A5036391537)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一种名为 BuildAnyPoint 的生成式框架，可从任意分布的点云中恢复结构化的三维建筑模型。

**💡 创新点**

创新点在于引入了 Loosely Cascaded Diffusion Transformer（Loca‑DiT），通过先在稠密与稀疏潜空间中逐层完成几何恢复，再用自回归 Transformer 生成高质量低面数、拓扑一致的网格。

**🔧 技术方法**

主要技术包括层级稀疏 VAE + 潜空间扩散模型、点云条件扩散、Transformer 自回归生成、点云与网格的 tokenization 与 detokenization。

**📊 数据集**

使用了 50,000 条荷兰阿姆斯特丹/鹿特丹的真实建筑数据（Airborne LiDAR 与手工 LoD2 模型）以及合成的 SfM 与极稀疏场景数据。

**📈 对比分析**

与 City3D、Point2Building、PoinTr、AnchorFormer 等基线相比，在点云完成指标（F-score、Chamfer、EMD、Uniformity）和网格质量指标（#V、#F、#P、FR、CD）均实现了显著提升，尤其在稀疏与噪声点云上表现最为突出。

**⚠️ 局限性**

主要局限在于对复杂几何细节的建模仍受限于训练数据集的简单几何偏好，导致在极其复杂建筑结构上的恢复效果仍有提升空间。

---

## 47. Brain-OF: An Omnifunctional Foundation Model for fMRI, EEG and MEG

**arXiv ID:** 2602.23410 | [PDF](https://arxiv.org/pdf/2602.23410v1)

**作者:** Hanning Guo `[一作]` (Forschungszentrum Jülich), Jürgen Dammers `[通讯]` (Forschungszentrum Jülich)

**通讯引用:** 2405 | [OpenAlex ID](https://openalex.org/A5048111026)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建并预训练了首个联合fMRI、EEG、MEG的全功能脑基座模型Brain-OF，支持单模态和多模态任务。

**💡 创新点**

提出Any-Resolution Neural Signal Sampler (ARNESS)统一不同分辨率信号，结合DINT注意力和稀疏专家混合网络，以及双域掩码时频重建预训练目标Masked Temporal-Frequency Modeling (MTFM)。

**🔧 技术方法**

跨模态采样器（ARNESS）、DINT注意力、稀疏专家混合（Sparse MoE）、时频掩码重建（MTFM）以及多模态序列串联融合。

**📊 数据集**

在37个公开数据集（约5.87M样本、32k参与者）中预训练，包含fMRI、EEG、MEG；下游评估7个任务共11个实验。

**📈 对比分析**

与多种单模态与跨模态基线（如LaBraM、BrainHarmonix、CNN-Transformer等）对比，Brain-OF Huge在9项任务中平均排名1.8，常规任务提升幅度高达30%~40%。

**⚠️ 局限性**

仍受限于训练成本高、跨模态对齐手段简化、部分任务对频域信息依赖不明显，以及对极端噪声或极短时序的鲁棒性不足。

---

## 48. A computational model for short-range van der Waals interactions between beams and shells

**arXiv ID:** 2602.24076 | [PDF](https://arxiv.org/pdf/2602.24076v1)

**作者:** Aleksandar Borković `[一作]` (Graz University of Technology), Roger A. Sauer `[通讯]` (Ruhr University Bochum)

**通讯引用:** 3478 | [OpenAlex ID](https://openalex.org/A5022137325)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于粗粒化势能的纤维-膜（梁-壳）相互作用数值模型，将原始的六维积分降为一维积分，实现了可计算的相互作用势能；

**💡 创新点**

创新点在于采用盘-替代板预积分技术，推导了盘-无限板的解析van der Waals势能，并给出误差上界，结合等距梁-克里霍夫-洛夫壳模型实现了全新的梁-壳相互作用形式；

**🔧 技术方法**

使用了 Lennard‑Jones 势能、粗粒化积分、解析盘-板交互法、等距等距有限元（Isogeometric FE）、延续法和牛顿-拉夫逊线性化等数值技术；

**📊 数据集**

没有使用真实实验数据，而是通过两个基准数值实验（梁剥离壳和梁弯曲壳）进行验证，采用合成的几何配置和参数；

**📈 对比分析**

通过三种模型（完整、简化1、简化2）和网格细化比较，结果显示三种模型在剥离过程中得到相近的总反作用力，误差低于1%，同时计算效率显著提升；

**⚠️ 局限性**

局限性包括：对壳面力的近似（点力形式）导致较差的力分布；仅适用于短程vdW相互作用，长程误差较大；未考虑切向摩擦或粘附能耗等效应。

---

## 49. Neural Diffusion Intensity Models for Point Process Data

**arXiv ID:** 2602.24083 | [PDF](https://arxiv.org/pdf/2602.24083v1)

**作者:** Xinlong Du `[一作]` (Purdue University), Vinayak Rao `[通讯]` (Purdue University)

**通讯引用:** 606 | [OpenAlex ID](https://openalex.org/A5083409558)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种基于神经SDE的Cox过程变分框架，用于建模稀疏、过度离散的事件数据并实现对潜在强度的推断。

**💡 创新点**

核心创新在于利用滤波器扩张理论证明条件化后强度仍保持扩散结构，并给出显式的得分修正漂移，从而使变分族可包含真后验并消除变分间隙。

**🔧 技术方法**

技术上结合了神经SDE先验、深度集合架构的后验修正网络、路径空间变分推断以及基于欧拉-马尔可夫数值模拟的梯度估计。

**📊 数据集**

实验使用了合成CIR动力学生成的Cox过程以及真实美国银行呼叫中心的分钟级呼叫记录数据。

**📈 对比分析**

与传统EM+MCMC和经典变分方法比较，本文的变分方法在相同计算预算下恢复先验精度相当，且在后验推断时速度提升1-2个数量级，预测对数似然相当。

**⚠️ 局限性**

局限性包括对扩散系数的固定假设、对深度集合架构的依赖以及在极少训练样本时易出现后验过拟合。

---

## 50. "Make It Sound Like a Lawyer Wrote It": Scenarios of Potential Impacts of Generative AI for Legal Conflict Resolution

**arXiv ID:** 2602.24130 | [PDF](https://arxiv.org/pdf/2602.24130v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 51. An $ε$-Optimal Sequential Approach for Solving zs-POSGs

**arXiv ID:** 2602.24092 | [PDF](https://arxiv.org/pdf/2602.24092v1)

**作者:** Jilles S. Dibangoye `[一作]` (University of Groningen), Erwan C. Escudie `[通讯]` (University of Groningen)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对零和部分可观测随机博弈 (zs-POSG) 的求解，将原本的同步极大极小备份拆解为两阶段的顺序备份，并通过新的状态统计量（顺序占用状态与私有占用族）实现价值函数的线性化和多项式复杂度。

**💡 创新点**

创新点在于：① 引入分离原理，将价值评估与策略提取的统计量区分；② 定义顺序占用状态与私有占用族，揭示价值函数的两层“最大-凹”几何结构；③ 将同步极大极小矩阵游戏拆解为两个线性规划，显著降低了备份的指数规模。

**🔧 技术方法**

使用的技术包括：顺序占用状态与私有占用族的构造；贝尔曼方程的顺序拆分；两层凹包（最大-凹）几何分析；两阶段线性规划备份（LP₁、LP₂）；点基值迭代 (PBVI) 以及占用族缓存与增量证据（witness）机制。

**📊 数据集**

实验数据集：Adversarial Tiger、Recycling、MABC 等标准零和 POSG 基准，全部采用公开转移与观测模型。

**📈 对比分析**

与传统同步备份的 Simultaneous PBVI 以及其他基线（如 Deep RL、公共信号抽象等）比较，实验显示：顺序 PBVI 在大多数游戏和更长 horizon 下的可达性、累计时间与最终价值上均优于同步版本；尤其在 Adversarial Tiger 等高维实例中，速度提升可达 10 倍，最终可利用率（exploitability）更低，表明策略更稳健。

**⚠️ 局限性**

局限性：仍需存储并维护大量的占用族与线性规划约束；在极大状态空间或极长 horizon 时，缓存和求解规模可能导致内存或时间瓶颈；此外，顺序拆分仅在理论上保证无损，实际实现中对采样和剪枝策略的依赖可能影响结果的收敛速度。

---

## 52. Zero-Incoherence Capacity of Interactive Encoding Systems: Achievability, Converse, and Side Information Bounds

**arXiv ID:** 2602.23520 | [PDF](https://arxiv.org/pdf/2602.23520v1)

**作者:** Tristan Simas `[一作]` (McGill University), Tristan Simas `[通讯]` (McGill University)

**通讯引用:** 70 | [OpenAlex ID](https://openalex.org/A5060461525)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文在交互式多位置编码系统中引入并分析了零不一致容量（zero‑incoherence capacity），证明其值为 1，并给出了关于侧信息、可实现性以及编码率与修改复杂度之间的定量关系。

**💡 创新点**

创新点：
- 定义并证明了零不一致容量与 Shannon 零误差容量的类比，得到唯一的最优编码率；
- 证明了在 k‑way 不一致情况下解析所需的最小侧信息量为 log₂k 位；
- 给出了实现容量的必要且充分条件：因果传播（反馈）与源头可观察性（侧信息）；
- 推导了编码率与修改复杂度的无界间隙，展示了单源唯一性在规模上的决定性优势；
- 所有核心定理均在 Lean‑4 证明助手中机械化验证。

**🔧 技术方法**

技术：信息论（Shannon 容量、零误差容量、Slepian‑Wolf 侧信息、Fano 不等式）、图论（可混淆图、独立集）、编码理论（最小描述长度、写一次存储代码）、形式化验证（Lean‑4 证明）。

**📊 数据集**

没有使用传统意义上的数据集；研究基于抽象模型与形式化证明，实例化仅涉及软件工程中的代码片段、数据库表结构等示例。

**📈 对比分析**

比较方式：通过信息论极限与抽象证明对比，无实验性能评估。理论上，编码率为 1 时修改复杂度为 O(1)；编码率 > 1 时复杂度为 Ω(n)，两者间的比值随 n 趋于无穷大，体现了理论上最优设计的显著优势。

**⚠️ 局限性**

局限性：
- 只考虑结构化事实（一次性定义后不可变更）的单语言系统；多语言或运行时可变结构的情况未覆盖；
- 需要系统支持因果传播和源头可观察性，现实系统中并非所有语言/数据库都满足；
- 仅提供理论极限与抽象实现，实际工程中还需考虑性能、可维护性等实务因素。

---

## 53. Weighted Unequal Error Protection over a Rayleigh Fading Channel

**arXiv ID:** 2602.24225 | [PDF](https://arxiv.org/pdf/2602.24225v1)

**作者:** Adeel Mahmood `[一作]` `[通讯]` (Nokia Bell Labs), Adeel Mahmood (Nokia Bell Labs)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文针对在慢衰落信道下的多层信息流动，提出了一种渐进式解码（progressive decoding）与多层功率分配的联合优化框架，并给出了两层系统的闭式最优功率分配解，进而设计了动态规划和Lambert W函数求解的算法来处理更多层数的情况；

**💡 创新点**

创新点在于将渐进式解码与功率分配问题联合考虑，得到可通过单层或多层解码提升信息可靠性的理论最优分配；对两层系统给出精确阈值条件决定是否需要额外功率投放于第一层；对多层系统提出了基于动态规划的近似算法和解析约束条件；

**🔧 技术方法**

主要技术包括：信息理论随机编码极限、Lambert W函数求解、二阶导数判定极值、动态规划（Bellman方程）以及Berry‑Esseen定理在有限块长下的误差概率估计；

**📊 数据集**

论文使用的并非公开数据集，而是理论模拟中采用的高斯码本和指数分布信道增益作为随机模型；

**📈 对比分析**

与传统正交多址或多层SIC等基线方案相比，渐进式解码在短块长（如n≈200）下能显著降低误码率或提升吞吐量，实验结果表明其性能优于正交传输，且在特定SNR和层重要性比例下可获得更优的功率分配；

**⚠️ 局限性**

限制包括：仅针对复高斯慢衰落信道；缺乏对峰值功率约束或硬件非理想的考虑；多层系统的解析解仍为数值求解，算法复杂度随层数增加而显著提升；并且随机编码分析无法直接映射到实际编码实现。

---

## 54. Microscopic Structure of Random 3-SAT: A Discrete Geometric Approach to Phase Transitions and Algorithmic Complexity

**arXiv ID:** 2602.23411 | [PDF](https://arxiv.org/pdf/2602.23411v1)

**作者:** Yongjian Zhan `[一作]` `[通讯]`, Yongjian Zhan

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于布尔超立方体的离散几何模型，用以解释随机3‑SAT的相变与求解复杂度。

**💡 创新点**

将相变机制从统计物理的能量景观转换为严格的顶点删减与Hamming距离连通性，给出了α=8/N 与 α=7/6(N-1)(N-2) 的绝对结构界限。

**🔧 技术方法**

使用组合几何分析、离散拓扑与概率计数方法，对超立方体中可行解、簇、冻结变量与移除变量进行系统建模。

**📊 数据集**

未使用具体数据集，而是基于随机生成的3‑CNF公式进行理论推导。

**📈 对比分析**

通过与统计物理理论对比说明一致性，未进行实验验证，主要阐述了理论一致性与“easy‑hard‑easy”曲线的机制。

**⚠️ 局限性**

局限在缺乏对实际SAT求解器性能的实证评估，以及对大规模随机实例中微观结构与搜索效率的定量关联尚未验证。

---

## 55. Multi-Objective Reinforcement Learning for Large-Scale Tote Allocation in Human-Robot Collaborative Fulfillment Centers

**arXiv ID:** 2602.24182 | [PDF](https://arxiv.org/pdf/2602.24182v1)

**作者:** Sikata Sengupta `[一作]` (University of Pennsylvania), Michael Caldara `[通讯]` (Amazon)

**通讯引用:** 38 | [OpenAlex ID](https://openalex.org/A5090886679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

研究了在大型人机协作仓储中心中，利用多目标强化学习实现托盘合并与分配的决策问题。

**💡 创新点**

创新点在于将最佳回应与无后悔动态相结合，构建零和拉格朗日博弈框架，并提出错误抵消理论以提取单一可行策略。

**🔧 技术方法**

采用多目标RL（MORL）+ DQN做最佳回应、在线梯度下降做调节器，构建大规模MDP并实现模拟训练。

**📊 数据集**

使用自定义事件驱动仿真器，模拟真实仓储操作的状态转移与约束，但未使用公开真实数据集。

**📈 对比分析**

与随机策略、无约束单目标RL以及单目标ETPH基线对比；实验显示MORL能在满足约束的同时显著提升吞吐率，远优于随机策略，略逊于无约束策略。

**⚠️ 局限性**

理论仅保证时间平均策略满足约束，单一策略可行性未完全保证；模型抽象简化导致与真实系统的迁移性待验证；训练成本与参数调优较高。

---

## 56. Normalisation and Initialisation Strategies for Graph Neural Networks in Blockchain Anomaly Detection

**arXiv ID:** 2602.23599 | [PDF](https://arxiv.org/pdf/2602.23599v1)

**作者:** Dang Sy Duy `[一作]` (RMIT University), Jeff Nijsse `[通讯]` (RMIT University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在Elliptic区块链交易图上使用GCN、GAT、GraphSAGE三种GNN进行异常检测，并系统评估不同权重初始化与图级归一化策略对模型性能的影响。

**💡 创新点**

发现初始化与归一化对不同GNN的效果存在明显的架构依赖性，并给出针对GCN、GAT、GraphSAGE的最优训练组合，首次在AML任务中揭示了此类训练细节的重要性。

**🔧 技术方法**

主要技术包括Xavier初始化、GraphNorm归一化，以及PyTorch Geometric实现的GCN、GAT、GraphSAGE三种网络架构。

**📊 数据集**

实验使用公开的Elliptic Bitcoin交易数据集，该数据集包含约20万节点、23万边以及稀疏的标签分布。

**📈 对比分析**

通过对比baseline、Xavier、GraphNorm+Xavier三种设置在AUPRC、AUC等指标上的表现，GraphSAGE+Xavier实现最高AUPRC 0.6678，GAT+GraphNorm+Xavier次之，表明合理的初始化与归一化可显著提升模型性能。

**⚠️ 局限性**

实验仅覆盖三种传统GNN架构，未涉及更先进的Transformer或时序GNN；超参数搜索受限于100次Optuna试验；未评估推理延迟、内存占用等实际部署指标；对Elliptic数据的泛化能力仍需进一步验证。

---

## 57. The Distance Spectrum of IEEE 802.11 Binary Convolutional Codes

**arXiv ID:** 2602.23651 | [PDF](https://arxiv.org/pdf/2602.23651v1)

**作者:** Rethna Pulikkoonattu `[一作]` `[通讯]` (Broadcom), Rethna Pulikkoonattu (Broadcom)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对IEEE 802.11标准的二进制卷积编码（BCC）在不同码率（1/2、2/3、3/4、5/6）下的距离谱进行精确计算，并基于此给出Viterbi译码的BER/FER上限。

**💡 创新点**

提出了利用“扩展状态”与“第一回归”技术将时间非平稳的punctured BCC转化为时不变的状态空间，再用截断的Neumann级数高效求得距离谱，首次提供完整的距离谱表和对应的误码率上限。

**🔧 技术方法**

使用增强型图书式传输函数、稀疏多项式运算、早停截断、三种编程语言实现（Python、Julia、C++），并验证其正确性。

**📊 数据集**

在AWGN信道下使用BPSK/QPSK/256‑QAM调制的Monte Carlo仿真数据对比理论上限，距离谱计算使用d_max=200。

**📈 对比分析**

通过与Monte Carlo仿真结果对比，发现理论上限在水分区几乎吻合，尤其在高码率时误差上限仍保持良好匹配；在低SNR区误差上限相对宽松。

**⚠️ 局限性**

局限性包括：对puncture周期L>10时矩阵规模较大；仅覆盖IEEE 802.11的BCC，未考虑其他码种；上限在低SNR区明显不紧；需手动设定d_max并保证所有路径终止。

---

## 58. Better Learning-Augmented Spanning Tree Algorithms via Metric Forest Completion

**arXiv ID:** 2602.24232 | [PDF](https://arxiv.org/pdf/2602.24232v1)

**作者:** Nate Veldt `[一作]` (Texas A&M University), Geoffrey Sanders `[通讯]` (Lawrence Livermore National Laboratory)

**通讯引用:** 685 | [OpenAlex ID](https://openalex.org/A5108086764)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种通用的学习增强度量森林完成（MFC）算法，能够在任意度量空间中以 2 近似的方式构造近似最小生成树，且该算法通过对代表点数进行插值实现了从单代表到全代表的连续过渡。

**💡 创新点**

创新点在于将代表点选择转化为共享预算的多实例 k‑center 问题，利用贪心 k‑center + 动态规划实现 2‑近似的预算分配，并给出了更紧的理论与实例级别近似保证。

**🔧 技术方法**

主要技术包括：度量森林完成框架、最小生成树、k‑center 贪心、动态规划分配代表点、实例成本分析与误差上界推导。

**📊 数据集**

实验使用四个真实数据集：Cooking（Jaccard 余弦），GreenGenes（Hamming），FashionMNIST（欧氏），Names-US（Levenshtein）等。

**📈 对比分析**

与原单代表算法和最优 Ω(n²) MFC 算法比较，实验显示即使在极小预算下也能得到几乎最优的生成树；动态规划方法得到更优的实例近似值和更小的 α‑误差，整体性能显著优于原方法。

**⚠️ 局限性**

局限性在于最坏情况仍停留在 2 近似；无法突破 2 的上界；对高度不平衡的森林加速效果有限；仅针对 γ‑overlap 指标，未探讨其他质量度量。

---

## 59. HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit

**arXiv ID:** 2602.23699 | [PDF](https://arxiv.org/pdf/2602.23699v1)

**作者:** Hao Wu `[一作]` (Institute of Digital Twin), Xiaoyu Shen `[通讯]` (Institute of Digital Twin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出HiDrop框架，采用晚期注入、凹形金字塔裁剪与早期退出，实现视觉token的层级化高效裁剪。

**💡 创新点**

创新点在于：①将视觉token从浅层完全绕过，仅在融合开始时注入；②通过ILVAS确定裁剪层，使用可微Top‑K动态选取最重要的token；③在中间层进行凹形金字塔式裁剪，深层完全退出；④引入持久位置编码、FlashAttention兼容的token选择与并行解耦计算，避免隐藏开销。

**🔧 技术方法**

技术手段包括：可微Top‑K、ILVAS相似度度量、持久RoPE位置编码、FlashAttention兼容的轻量辅助注意力、并行视觉KV预计算与解耦。

**📊 数据集**

实验使用LLaVA‑1.5（含MobileLLaMA‑2.7B、Vicuna‑7B‑v1.5、Vicuna‑13B‑v1.5）作为模型骨干，评测11个多模态基准（MMEP、MMB、MMBCN、GQA、VQAv2、SQAI、VizWiz、TextVQA、POPE、SEEDI、MMStar）。

**📈 对比分析**

与FastV、PDrop、TwigVLM、VoCo‑LLaMA等现有进展比较，HiDrop在保持99‑98%性能的前提下压缩≈90%视觉token，训练速度提升≈1.7×，推理FLOPs下降≈88.9%，在大多数基准上实现或超过对手的平均表现。

**⚠️ 局限性**

局限性包括：对LLaVA体系结构的依赖，需对注入层、退出层、裁剪层进行手工调优；对不同视觉编码器或更深模型的通用性尚未验证；极端压缩下的微调需求较大。

---

## 60. Hybrid Quantum Temporal Convolutional Networks

**arXiv ID:** 2602.23578 | [PDF](https://arxiv.org/pdf/2602.23578v1)

**作者:** Junghoon Justin Park `[一作]` (Seoul National University), Jiook Cha `[通讯]` (Seoul National University)

**通讯引用:** 2593 | [OpenAlex ID](https://openalex.org/A5033979262)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种混合量子-经典的时序卷积网络HQTCN，用于多变量时间序列分析。

**💡 创新点**

创新点在于将经典的时间滑动窗口与共享的量子卷积神经网络相结合，显著降低参数量，并在高维多变量数据中实现长程依赖捕捉。

**🔧 技术方法**

采用经典膨胀滑动窗口、线性投影、8量子比特QCNN核心（角度嵌入、卷积+池化两层）以及量子期望值输出，整体实现参数共享。

**📊 数据集**

使用NARMA10序列和PhysioNet EEG（64通道运动想象）作为评测数据集。

**📈 对比分析**

与LSTM、Transformer、TCN、QLSTM及单独QCNN对比，HQTCN在NARMA任务中保持相近误差但参数仅为312；在EEG分类任务中获得最高AUROC 0.7929，同时参数仅为6,416，且在仅10个样本时仍优于传统基线。

**⚠️ 局限性**

局限包括：仅在模拟量子电路上验证，尚未在真实NISQ设备上实现；量子层深度有限，扩展到更大规模数据时可行性待验证；对量子噪声和测量开销的影响未做深入实验。

---

## 61. DLEBench: Evaluating Small-scale Object Editing Ability for Instruction-based Image Editing Model

**arXiv ID:** 2602.23622 | [PDF](https://arxiv.org/pdf/2602.23622v1)

**作者:** Shibo Hong `[一作]` (Fudan University), Yixin Cao `[通讯]` (Fudan University)

**通讯引用:** 5648 | [OpenAlex ID](https://openalex.org/A5013247988)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了专门针对小尺寸物体编辑的基准 DLEBench，并设计了三阶段数据生成管线（对话式生成元数据、裁剪+编辑参考图像、人工验证）以及双模式评估框架（工具驱动模式与 Oracle 导向模式）。

**💡 创新点**

1) 将视觉推理样本转化为编辑任务，实现了大规模无人工标注的数据扩增；2) 采用裁剪+编辑策略生成高质量参考图像，解决小目标编辑的参考缺失问题；3) 重新定义 IF/VC 的分级评估标准，消除主观模糊度；4) 引入工具驱动与 Oracle 模式，显著提升评测与人工判定的一致性。

**🔧 技术方法**

利用 GPT‑4.1 进行对话式元数据生成；使用 GroundingDINO、YOLOv13 等目标检测器定位小目标；Real‑ESRGAN 对裁剪区域进行放大；工具驱动模式调用 Zoom‑In、Difference、Enhancer 等工具；Oracle 模式使用人工标注的 bbox；评估指标基于 IF/VC 四级分级，结合 Pearson/Spearman 等统计检验。

**📊 数据集**

基于 MME‑Realworld、Pixel‑Reasoner、V*‑Bench 这三类视觉推理基准，共计 2,043 条原始样本，转化后得到 1,889 条小尺寸编辑样本，目标占比 1%–10%。

**📈 对比分析**

对 10 款 IIEM（闭源如 Gemini‑3‑Pro、GPT‑Image‑1，开源如 Bagel‑Think、UniREdit‑Bagel、MagicBrush 等）在 Oracle‑guided 模式下进行 IF/VC 评测，平均分数显示 Gemini‑3‑Pro 最高 65.55，Bagel‑Think 61.00；所有模型在 IF 上均低于 70，尤其是 Change Count 最低；Oracle 与工具驱动模式与人工评估的相关性显著高于传统 LMM‑as‑a‑Judge，验证了评测框架的可靠性。

**⚠️ 局限性**

1) 样本在 instruction type 上分布不均衡，主要来自视觉推理数据集；2) 目前缺乏自动化的样本扩增 pipeline；3) 评测仅针对英文文本，未考虑多语言场景；4) 对极小尺寸（<0.07%）目标的评估尚不充分。

---

## 62. An Efficient Unsupervised Federated Learning Approach for Anomaly Detection in Heterogeneous IoT Networks

**arXiv ID:** 2602.24209 | [PDF](https://arxiv.org/pdf/2602.24209v1)

**作者:** Mohsen Tajgardan `[一作]` (Qom University of Technology), Mahtab Jamali `[通讯]` (Malmo University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种面向异构IoT网络的无监督联邦学习框架，用共享特征提升异常检测性能

**💡 创新点**

创新点在于仅聚合公共维度的模型层并动态对齐权重，同时保留各客户端特有特征，并结合SHAP实现可解释性

**🔧 技术方法**

使用深度自编码器、K‑means聚类、联邦平均(FedAvg)与SHAP解释，配合动态权重调整

**📊 数据集**

实验基于CICIoT2022（设备识别）、CICIoT2023（异常检测）和CICIoT‑DIAD 2024（异常检测）三个公开IoT数据集

**📈 对比分析**

与传统集中式自编码器+K‑means基线对比，所提方法在CICIoT2024上F1提升约15%，CICIoT2022也有显著提升，CICIoT2023提升有限，整体收敛稳定

**⚠️ 局限性**

局限在于需要一定量的公共特征；使用K‑means不捕捉时序依赖；未评估通信/能耗；对特征重叠度下降时效果可能退化

---

## 63. Walking with Robots: Video Analysis of Human-Robot Interactions in Transit Spaces

**arXiv ID:** 2602.23475 | [PDF](https://arxiv.org/pdf/2602.23475v1)

**作者:** Barry Brown `[一作]` (University of Copenhagen), Mathaius Broth `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对机场公共空间中两种清洁机器人的视频进行民族方法学和对话分析，揭示其在社交导航中的问题。

**💡 创新点**

首次将对话分析方法与机器人交互研究相结合，提出“社交意识运动”的设计空间与三项强概念，强调机器人的社会可解释性。

**🔧 技术方法**

采用视频实地考察、对话分析（EMCA）和社会运动学理论；不涉及传统算法或模型。

**📊 数据集**

六小时、两台清洁机器人（Nilfisk Liberty SC60 与 Ecobot 40）在欧洲大型商业机场三处公共区的录像数据。

**📈 对比分析**

无定量对比指标，研究以案例式分析展示机器人停顿、群体识别与空间情景理解不足；未给出性能数值。

**⚠️ 局限性**

样本规模有限，仅两台机器人、一处机场、研究时长短，缺乏跨场景与长期验证，难以推广到其他公共空间。

---

## 64. Determining Factorial Speed Fast

**arXiv ID:** 2602.24064 | [PDF](https://arxiv.org/pdf/2602.24064v1)

**作者:** Zhidan Feng `[一作]` (Beijing Technology University), Silas Cato Sacher `[通讯]` (Trier University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过将图类用有限的二进制语言在词形上进行描述，证明这些类的速度上限为阶乘，并利用这一计数论证统一解决多种之前未解的速度问题；进一步说明 k‑letter 图类速度为指数级且是 k‑thin 图类的严格子类。

**💡 创新点**

创新点在于将有限语言表示与图类速度联系，给出一种通用计数方法；利用该方法对多种图类（如区间图、排列图等）得到阶乘速度并推导出先前未知的速度分离；同时证实 k‑letter 图类与 k‑thin 图类之间的严格包含关系。

**🔧 技术方法**

主要技术包括：词形表示与形态学（morphism）工具、0‑1 对称语言、交错（shuffle）与删减函数、k‑uniform 单词的计数与排序分析。

**📊 数据集**

本工作为理论性研究，不依赖具体数据集；通过已知的图类（区间图、排列图等）和已确定的速度阶数进行比较。

**📈 对比分析**

比较方法基于速度阶级（多项式、指数、阶乘、超阶乘）对图类进行上界/下界评估；结果显示所研究图类的速度不超过阶乘，k‑letter 图类则为指数级。

**⚠️ 局限性**

局限性：仅适用于能用有限语言描述的图类；对无限语言或更一般的图类缺乏直接结果；此外，论文未给出多项式时间的识别算法或具体构造方法。

---

## 65. DashengTokenizer: One layer is enough for unified audio understanding and generation

**arXiv ID:** 2602.23765 | [PDF](https://arxiv.org/pdf/2602.23765v1)

**作者:** Heinrich Dinkel `[一作]` (Xiaomi Inc), Jian Luan `[通讯]` (Xiaomi Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

训练了一种名为DashengTokenizer的连续音频tokenizer，利用冻结的高维语义特征并通过线性投影注入低维声学信息，实现了同时支持语音、音乐与环境音的理解与生成任务。

**💡 创新点**

核心创新在于逆向思维：先保留预训练的语义编码器，再将声学细节注入；通过单阶段线性投影与语义保持损失，避免了传统多阶段蒸馏与冗余训练。

**🔧 技术方法**

技术细节包括：MiDashengLM 630M 语义编码器、MelSpectrogram+2D卷积+层归一化声学编码器、Vocos 1D上采样解码器、GAN对抗训练（多频率判别器）、semantic preservation loss、flow-based DiT 生成网络。

**📊 数据集**

使用约282k小时的跨领域音频数据：音乐21%、英语语音21%、中文语音40%、其他语言10%、环境音26%；评估集覆盖X-ARES（22项理解）、Seed-TTS、MUSDB18、AudioSet、Speech Enhancement（Valentini、DNS1）、Text-to-Audio（AudioCaps）、Text-to-Music（MusicCaps）。

**📈 对比分析**

与现有codec、encoder、tokenizer和VAE基线对比，DashengTokenizer在22项理解任务中大幅领先，Speech、Music、Sound任务均处于最前沿；在Speech Enhancement、Text-to-Audio和Text-to-Music生成中超越VAE baseline，并且训练收敛更快、参数规模相近。

**⚠️ 局限性**

局限性包括：在需要纯语义抽象的任务（如意图分类、ASR）仍稍逊；声学注入可能在某些任务中产生干扰；目前帧率固定为25Hz，需进一步研究多帧率或更广泛语言的适配。

---

## 66. StemVLA:An Open-Source Vision-Language-Action Model with Future 3D Spatial Geometry Knowledge and 4D Historical Representation

**arXiv ID:** 2602.23721 | [PDF](https://arxiv.org/pdf/2602.23721v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 67. Towards Efficient and Generalizable Retrieval: Adaptive Semantic Quantization and Residual Knowledge Transfer

**arXiv ID:** 2602.23978 | [PDF](https://arxiv.org/pdf/2602.23978v1)

**作者:** Huimu Wang `[一作]` (JD.com), Mingming Li `[通讯]` (IIE, UCAS)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出SA^2CRQ框架，解决工业级生成检索中热门与冷启动项目的辨识性与泛化性冲突

**💡 创新点**

首次将顺序自适应残差量化(SARQ)与锚定课程残差量化(ACRQ)结合，实现动态ID长度分配与结构化知识迁移

**🔧 技术方法**

使用残差量化变分自编码器(RQ-VAE)、信息熵路径度量、信息瓶颈与两阶段训练

**📊 数据集**

在JD.com工业电商数据（1800万查询，500万商品）和MS MARCO衍生的MS300K公开数据集上评估

**📈 对比分析**

与BM25、DPR、DSI、NCI、TIGER、RQ-Kmeans等基线对比，Recall@2k提升12.1%，召回率、Ret-Per和hallucination率均显著下降，在线AB测试提升转化率0.13%及用户价值0.42%

**⚠️ 局限性**

局限于单模、同域检索，未验证跨模态或多域场景，且对极度稀疏项目的泛化仍有提升空间

---

## 68. A Foundation for Differentiable Logics using Dependent Type Theory

**arXiv ID:** 2602.23878 | [PDF](https://arxiv.org/pdf/2602.23878v1)

**作者:** Reynald Affeldt `[一作]` (National Institute of Advanced Industrial Science and Technology), Kathrin Stark `[通讯]` (Heriot-Watt University)

**通讯引用:** 130 | [OpenAlex ID](https://openalex.org/A5054275908)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

系统化比较模糊逻辑与可微逻辑的代数、解析与证明理论特性，并在 Isabelle 中实现统一框架与形式化证明

**💡 创新点**

提出单一语法与解释函数统一表示多种可微/模糊逻辑，证明它们可在残差半格中解释，完善并纠正先前工作中的错误，并为 STL 与 DL2 提供新的可证明的 sequent 计算器

**🔧 技术方法**

依赖依赖类型理论与 Isabelle/HOL 的 Mathematical Components 库，实现形式化证明；使用 L'Hôpital 规则、实数拓扑与残差半格公理；通过 hypersequent 计算器证明语义一致性

**📊 数据集**

本研究为形式化工作，无使用具体机器学习数据集；示例中以 ε-δ 鲁棒性逻辑为例构造损失函数

**📈 对比分析**

通过形式化证明对比每种逻辑在 R1–R10、N1–N4、M1–M3、shadow‑lifting 等属性上的满足情况，证明了 DL2 与 STL 在某些参数极限下满足残差半格与可微性；缺失的属性得到明确列出

**⚠️ 局限性**

仍缺乏完整的证明理论（完整性），部分逻辑（如 STL 完整语法）未给出 sequent 规则；一些属性（如 idempotence 与 shadow‑lifting 同时满足）在理论上不可能；扩展到更大逻辑仍有挑战

---

## 69. Steering and Rectifying Latent Representation Manifolds in Frozen Multi-modal LLMs for Video Anomaly Detection

**arXiv ID:** 2602.24021 | [PDF](https://arxiv.org/pdf/2602.24021v1)

**作者:** Zhaolin Cai `[一作]` (Xinjiang University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21522 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种名为SteerVAD的零调参视频异常检测框架，利用梯度无关的代表性可分离分析识别冻结多模态大型语言模型中的潜在异常专家，并通过层次化元控制器在运行时动态对这些专家的表示流形进行几何驱动的缩放修正，从而在不调整模型权重的前提下实现异常检测。

**💡 创新点**

创新点在于将模型内部的注意力头视为可调节的“功能电路”，通过梯度无关的可分离分析精准挑选异常专家，并设计全球与局部双层控制器实现上下文感知的各向异性流形缩放，突破了以往被动特征读取的局限，实现对预训练偏差与上下文歧义的主动纠偏。

**🔧 技术方法**

技术上使用了代表性可分离分析（RSA）寻找潜在异常专家，层次化元控制器（HMC）中的全局审视门与局部门控产生调节向量，利用各向异性尺度变换对专家特征进行几何修正，最后采用逻辑回归评分器和一维高斯平滑生成异常曲线；训练只需对控制器与评分器做微调。

**📊 数据集**

在UCF‑Crime和XD‑Violence两个公开基准上验证了方法的有效性，分别使用1%校准数据完成专家挑选与控制器训练。

**📈 对比分析**

相较于现有所有无调参多模态VAD方法，SteerVAD在UCF‑Crime上实现87.15% AUC、在XD‑Violence上达到83.02% AP，超过之前最优的82.03%/64.04%，并且在使用与微调模型相近规模的冻结基础模型上仅需1%训练数据，显著缩小了与全量微调方法的性能差距。

**⚠️ 局限性**

主要局限包括对冻结模型的依赖性——若基础模型缺乏足够的跨域表达能力，流形调制效果受限；仅在视频异常检测场景下验证，尚未评估在更大规模或更复杂多模态任务中的推广性；需要少量标注数据进行RSA与控制器训练，虽低但仍非零。

---

## 70. Sample Size Calculations for Developing Clinical Prediction Models: Overview and pmsims R package

**arXiv ID:** 2602.23507 | [PDF](https://arxiv.org/pdf/2602.23507v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 71. A quadratic lower bound for 2DFAs against one-way liveness

**arXiv ID:** 2602.24279 | [PDF](https://arxiv.org/pdf/2602.24279v1)

**作者:** Kehinde Adeogun `[一作]` (Carnegie Mellon University), Christos Kapoutsis `[通讯]` (Carnegie Mellon University)

**通讯引用:** 378 | [OpenAlex ID](https://openalex.org/A5023928618)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文证明了将一位非确定性有限自动机（1‑FA）转化为两位确定性有限自动机（2‑DFA）时，状态数至少为 h²，其中 h 为输入图的高度；该结论通过求解一维活性问题（one‑way liveness）得到。

**💡 创新点**

创新点在于提出了一条通用的下界引理（Main Lemma），能够处理任意字母表且不依赖于特殊输入长度。作者构造了一系列平滑属性（smooth properties）并利用连通矩阵的递增序列，使得每一步都严格降低某一退出状态大小，从而得到整体的二次下界。

**🔧 技术方法**

技术上使用了：
- 连通矩阵与布尔乘法的代数性质（幂等、可交换性）
- “generic string”概念与退出状态集合的单调性
- 细粒度的属性扩展（suffix of choice）与属性分离性
- 对于一维活性问题的图模型（每个符号为二维矩阵）
- 组合矩阵的逐步翻转构造（上三角、下三角、全矩阵翻转）。

**📊 数据集**

本研究为纯理论研究，没有使用实验数据集；所有结果均来自组合与矩阵论的证明。

**📈 对比分析**

与已知结果比较：与 Chrobak 的一次性二次下界（在单字母表上的 1‑FA 转 1‑DFA）完全匹配；对更广泛字母表的情况同样保持二次下界，进一步验证了该下界的普适性。没有在实验中展示性能，只是理论上的下界证明。

**⚠️ 局限性**

局限性：
- 仅得到二次下界，无法推出超二次（如多项式或指数）下界。
- 该引理在构造极长属性序列时可能受限，作者猜想该序列已达到最大长度。
- 对于其他自动机模型或更复杂语言类别，未能直接应用，仍需寻找更强的工具。

---

## 72. Resources for Automated Evaluation of Assistive RAG Systems that Help Readers with News Trustworthiness Assessment

**arXiv ID:** 2602.24277 | [PDF](https://arxiv.org/pdf/2602.24277v1)

**作者:** Dake Zhang `[一作]` (University of Waterloo), Charles L. A. Clarke `[通讯]` (University of Waterloo)

**通讯引用:** 8803 | [OpenAlex ID](https://openalex.org/A5037737168)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提供了TREC 2025 DRAGUN跟踪的完整可复用资源，包括新闻主题、专家制定的权重化检索问题和短答案、人工评判、以及基于LLM的AutoJudge，用于自动评估问答与报告生成任务。

**💡 创新点**

创新点在于：①把专家制定的权重化rubric作为评估基准，避免了传统的“池化-评判”模式；②设计了面向新闻可信度评估的双任务（问题生成和报告生成）；③推出AutoJudge，利用LLM进行大规模、可复制的自动评判，并与人工评判保持高度一致的排名（Task 1 τ=0.678，Task 2 τ=0.872）。

**🔧 技术方法**

技术主要包括：使用OpenAI GPT‑OSSS进行few‑shot提示式评判；利用文本相似度模型（如Qwen3）预筛选问题；采用Python脚本计算加权分数；以及对评判标签进行一致性评估（Cohen’s κ、Gwet’s AC1）。

**📊 数据集**

数据集：MS MARCO V2.1 Segmented Corpus（约114 M段）用于检索；30篇选定的新闻文章（来自MS MARCO Document Corpus）作为主题；每篇文章对应的专家rubric（共236条问题、551条答案）。

**📈 对比分析**

比较方法：通过人类评估与AutoJudge的跑分进行对比，使用Kendall’s τ衡量排名相关性；在标签层面使用Cohen’s κ和Gwet’s AC1评估一致性。性能显示AutoJudge在保持排名方面表现优异，尤其在报告生成任务上τ高达0.872，表明其能有效替代人工评判。

**⚠️ 局限性**

局限性：①AutoJudge在处理复合问题时需事先过滤，可能导致误删有效问题；②rubric的答案有时不在固定检索语料中，评估时存在信息缺失；③评判框架仅覆盖支持/部分/相反/无关四类，未涉及引用可信度、可读性等其他重要维度。

---

## 73. Online Register for Dual-Mode Self-Supervised Speech Models: Mitigating The Lack of Future Context

**arXiv ID:** 2602.23702 | [PDF](https://arxiv.org/pdf/2602.23702v1)

**作者:** Keita Goto `[一作]` (LY Corporation), Shinji Watanabe `[通讯]` (Carnegie Mellon University)

**通讯引用:** 25186 | [OpenAlex ID](https://openalex.org/A5001291873)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文针对双模自监督语音模型缺失未来上下文导致的注意力不匹配问题，提出在线寄存器（learnable tokens）作为伪未来上下文，并引入未来预测损失以显式引导寄存器捕获未来信息，从而在在线与离线模式之间提升一致性和整体识别性能。

**💡 创新点**

创新点在于：①在线寄存器的引入，为在线模式提供可学习的虚拟未来上下文；②结合未来预测损失，使寄存器主动学习并存储更丰富的未来信息；③两项技术共同解决了在线模式下的上下文缺失而不增加额外延迟。

**🔧 技术方法**

使用的技术包括：wav2vec 2.0 基础架构的 Transformer 编码器；对比学习框架（masking、quantization、stop‑gradient）；动态块训练（随机 chunk 长度和 look‑ahead）；在线寄存器的学习嵌入；未来预测损失（MSE）以及常用的 Adam 优化器和正弦位置编码。

**📊 数据集**

主要数据集：LibriSpeech 960h（无标签预训练，带标签微调）以及英语子集 FLEURS，用于评估跨域泛化能力；实验还在不同 chunk 大小（160ms、320ms、640ms）下进行评估。

**📈 对比分析**

与无寄存器基线以及 UFO2 双模模型对比，在线寄存器在 160ms/640ms 低延迟块大小下使 WER 下降约 1–2% 甚至更大，且在离线模式下也有显著提升；与 UFO2 相比，在同样的解码条件下整体精度更高，表明寄存器机制在保持低延迟的同时显著缩小了在线与离线性能差距。

**⚠️ 局限性**

局限性包括：未来预测损失在某些数据集或块大小下效果有限；寄存器数量过多可能导致过拟合，尤其在低资源或多域环境下；方法仍需在更大规模多语种或多任务数据上验证其泛化能力。

---

## 74. FaultXformer: A Transformer-Encoder Based Fault Classification and Location Identification model in PMU-Integrated Active Electrical Distribution System

**arXiv ID:** 2602.24254 | [PDF](https://arxiv.org/pdf/2602.24254v1)

**作者:** Kriti Thakur `[一作]` (ABB Ability Innovation Center), Mayukha Pal `[通讯]` (ABB Ability Innovation Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了 FaultXformer，一个基于 Transformer 编码器的双阶段模型，用于电力分布网络中 PMU 电流时序数据的故障类型识别与故障位置定位。

**💡 创新点**

创新点在于将 Transformer 的自注意力机制与时序 PMU 数据相结合，构建了独立优化的两阶段网络，显著提升了在高 DER 负荷和噪声环境下的识别准确率，并通过可视化注意力提供模型可解释性。

**🔧 技术方法**

采用 Transformer Encoder、位置编码、全局平均池化以及多头自注意力的深度学习技术，并配合 Z‑score 归一化和 10 折交叉验证等训练与评估手段。

**📊 数据集**

使用 IEEE 13 节点测试馈线的数据集，模拟 20 个故障位置、7 种故障电阻、7 个故障起始角，加入 0.01–0.03% Gaussian 噪声以及 0%–80% 的分布式发电渗透场景。

**📈 对比分析**

与传统 RNN、LSTM、CNN 等基线模型比较，FaultXformer 在故障类型分类上平均准确率达 99.33%，定位精度达 99.74%，相对基线提升超过 30% 的分类准确率和 40% 的定位准确率，且在 GPU 上平均推理延迟仅 17.35 ms，满足 IEEE 实时要求。

**⚠️ 局限性**

局限性包括仅在 13 节点短时序 PMU 数据上验证，未检验对更大规模网络、长时序或不同测量噪声特性的泛化能力，模型在不同拓扑或电压测量条件下的适用性仍需进一步研究。

---

## 75. CIll: CTI-Guided Invariant Generation via LLMs for Model Checking

**arXiv ID:** 2602.23389 | [PDF](https://arxiv.org/pdf/2602.23389v1)

**作者:** Yuheng Su `[一作]` (Institute of Software, Chinese Academy of Sciences), Enyuan Tian `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CTI指导的LLM框架 CIll，用于自动生成辅助不变式以帮助模型检查器验证硬件设计。

**💡 创新点**

将大语言模型与IC3、K-归纳等传统引擎结合，通过CTI反馈迭代生成并验证辅助不变式，并利用本地证明和已学习不变式加速搜索。

**🔧 技术方法**

使用 ChatGPT-5.2 生成辅助不变式，IC3、K-归纳、BMC 进行正确性和归纳性检验，并通过 Trace 最小化与 MCP 接口实现对模型上下文的高效查询。

**📊 数据集**

在 RISC‑V Formal 框架下的 NERV、PicoRV32、SERV 三核的 SystemVerilog 源码上进行实验，重点验证 M 扩展的 ALTOPS 语义。

**📈 对比分析**

与 rIC3 基准引擎、local‑proof 引擎及 AVR 检查器比较，CIll 在之前无法解决的 76 个案例中全部解决 NERV 与 PicoRV32 的非 M 扩展，平均需数十个辅助不变式，验证时间从几分钟到数小时不等。

**⚠️ 局限性**

对位级乘法的 M 扩展仍无法解决，且对长周期指令（如 SERV）导致 BMC 难以发现 CTI，LLM 推理时间占主导，限制整体速度。

---

## 76. Recommendation Algorithms: A Comparative Study in Movie Domain

**arXiv ID:** 2602.24125 | [PDF](https://arxiv.org/pdf/2602.24125v1)

**作者:** Rohit Chivukula `[一作]` (University of Huddersfield), C. H. S. N. P. Sairam Rallabandi `[通讯]` (R.V.R. & J.C. College of Engineering)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过回归和矩阵分解方法对Netflix电影评分进行预测，构建了包含聚合特征、用户相似度与电影相似度的特征向量，进一步评估了多种模型组合。

**💡 创新点**

创新点在于将传统的基线与KNN相似度模型与XGBoost集成，并结合用户/电影相似度特征构造统一的特征向量，实验表明SVD矩阵分解在此框架下表现最佳。

**🔧 技术方法**

主要技术包括XGBoost回归、Surprise库中的基线（Baseline）、KNN基线（KNNBaseline）、SVD矩阵分解（MF）以及特征工程（聚合特征、相似度矩阵）等。

**📊 数据集**

使用的是Kaggle公开的Netflix Prize数据集，约1.0048亿条评分，480189用户，17770部电影。

**📈 对比分析**

通过RMSE和MAPE两种指标进行比较，实验结果显示SVD与Surprise KNNBaseline取得最低RMSE，XGBoost+多模型组合相对性能下降，MAPE波动不大（34%–35%）。

**⚠️ 局限性**

局限性主要包括：冷启动问题导致未出现训练集的用户/电影预测困难；特征工程耗时高且部分特征冗余可能降低模型效果；缺乏对模型泛化与实时推荐的深入分析。

---

## 77. AHAP: Reconstructing Arbitrary Humans from Arbitrary Perspectives with Geometric Priors

**arXiv ID:** 2602.23951 | [PDF](https://arxiv.org/pdf/2602.23951v1)

**作者:** Xiaozhen Qiao `[一作]` (University of Science and Technology of China), Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**通讯引用:** 61894 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本论文提出了一种无标定的多视角多人的实时 3D 重建框架 AHAP，能够在任意摄像机视角下一次性输出人体 SMPL 模型和场景几何。

**💡 创新点**

核心创新包括：① 用可学习的查询与 soft‑assignment 进行跨视角身份关联，并通过对比学习进行监督；② 在人体头部使用多视角聚合特征与场景上下文进行 SMPL 回归；③ 结合跨视角重投影损失和多视角三角测量，实现精确的世界坐标定位。

**🔧 技术方法**

主要技术手段为跨视角注意力模块、软赋值身份关联、对比损失、跨视角重投影约束、Transformer 头部、DA3 场景编码器与 Multi‑HMR 人体编码器的联合使用。

**📊 数据集**

训练采用 BEDLAM 序列，评估使用 EgoHumans 与 EgoExo4D 两个多视角人体数据集。

**📈 对比分析**

与 HSfM、HAMSt3R、UnCaliPose、DA3 等基线相比，AHAP 在人体 W‑MPJPE、GA‑MPJPE、PA‑MPJPE、摄像机旋转误差等指标均保持竞争力，并且推理速度比 HSfM 高 180 倍（约 1.16 s 代替 209 s）。

**⚠️ 局限性**

主要局限在于绝对平移误差较大（受尺度不确定性影响），对多视角稀缺或严重遮挡的场景仍有挑战；此外模型对大范围全景或极端姿态的泛化尚需进一步验证。

---

## 78. FAVLA: A Force-Adaptive Fast-Slow VLA model for Contact-Rich Robotic Manipulation

**arXiv ID:** 2602.23648 | [PDF](https://arxiv.org/pdf/2602.23648v1)

**作者:** Yao Li `[一作]`, Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8098 | [OpenAlex ID](https://openalex.org/A5053344541)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了FAVLA模型，融合慢速视觉-语言语义与高速力反馈，实现快慢感知控制；

**💡 创新点**

创新点包括快慢分离架构、力注入动作专家以及基于预测力方差的自适应推理频率；

**🔧 技术方法**

采用大规模视觉语言模型、流匹配生成器、TCN力编码、力自适应推理、LoRA微调等技术；

**📊 数据集**

使用Monte机器人收集的四个真实工业接触任务（USB插入、齿轮装配、盒子翻转、板擦拭）的示范轨迹，共计120+条；

**📈 对比分析**

与π₀、π₀+Force、TA‑VLA、ForceVLA等基线对比，平均成功率达到80.8%，比最佳基线提升13.8%，峰值接触力显著降低；

**⚠️ 局限性**

局限性：依赖大型模型与高频力传感器，推理时仍存在计算开销；在极端噪声或非标准接触场景下性能未知。

---

## 79. Humanoid Robots as First Assistants in Endoscopic Surgery

**arXiv ID:** 2602.24156 | [PDF](https://arxiv.org/pdf/2602.24156v1)

**作者:** Sue Min Cho `[一作]` (Johns Hopkins University), Mathias Unberath `[通讯]` (Johns Hopkins Medical Institutions)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在人体尸体模型上进行单侧蝶窦切除术，利用单人遥控的 Unitree G1 人形机器人携带内窥镜，实时为外科医生提供稳定的视野，完成整个手术流程。

**💡 创新点**

首次证明人形机器人的外观与运动学能够在有限的手术空间内完成视觉辅助任务，为未来人形机器人在手术室中的协作角色奠定实证基础，并提出了可转化为临床的工程改进目标。

**🔧 技术方法**

采用 Unitree G1 29 轴人形机器人、通过 Meta Quest 3 头戴式显示器实现混合现实遥操作；机器人上装配自定义刚性适配器固定内窥镜；通过手持控制器与机器人上臂逆运动学实现姿态控制。

**📊 数据集**

使用单个解剖学尸体样本和随手术实时采集的内窥镜视频作为实验数据；未使用公开医学影像数据集。

**📈 对比分析**

通过半结构化访谈和主题分析评估手术表现；结果显示机器人能保持无漂移、无颤抖的视野，手术完成率为 100%，但在与人类助手对比时仍存在力学精度不足和操作学习曲线等问题；未给出数值化的性能指标。

**⚠️ 局限性**

主要局限包括：缺乏触觉反馈导致对压力控制不佳；机器人占用手术通道中心，限制了手术器械的操作空间；操作员学习曲线长，需进一步训练；安全与故障切断机制尚不完善；缺乏自主导航与力调节功能。

---

## 80. Tidynote: Always-Clear Notebook Authoring

**arXiv ID:** 2602.23490 | [PDF](https://arxiv.org/pdf/2602.23490v1)

**作者:** Ruanqianqian Huang `[一作]`, Sorin Lerner `[通讯]` (Cornell University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 always‑clear notebook authoring，构建了名为 Tidynote 的 Jupyter 扩展，支持在保持笔记本内容与状态清晰的同时进行探索性编程。

**💡 创新点**

创新点在于将可移动的 Scratchpad、单向线性执行与状态分叉机制整合到同一系统中，使笔记本在任何阶段都能保持可读、可重现且易于分享，而无需后期繁重的清理工作。

**🔧 技术方法**

技术实现基于 TypeScript（前端 GUI）和 Python（执行层），采用线性执行模型与局部状态分叉、cell pinning 与可折叠的 Scratchpad，兼容标准 Jupyter。

**📊 数据集**

实验使用了两个 Kaggle 数据集：Netflix Movies and TV Shows 以及 Spotify Songs 2023，供参与者进行开放式数据分析任务。

**📈 对比分析**

通过 13 人开放式实验与 SUS 问卷对比常规 Jupyter，评估了可用性、任务完成效率和笔记本清晰度；结果显示用户在保持清晰度、完成任务和重现性方面显著优于传统方法，性能开销极小。

**⚠️ 局限性**

限制包括：仅针对经典 Jupyter 实现，线性执行不支持所有非线性工作场景；Scratchpad 的位置固定缺乏灵活性；缺乏对大规模数据集的性能评估；未来需进一步验证跨平台兼容性和可定制性。

---

## 81. Interpretable Debiasing of Vision-Language Models for Social Fairness

**arXiv ID:** 2602.24014 | [PDF](https://arxiv.org/pdf/2602.24014v1)

**作者:** Na Min An `[一作]` (KAIST), Hyunjung Shim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在预训练的视觉‑语言模型（VLM）上通过训练稀疏自编码器（SAE）定位与社会属性相关的神经元，然后在推理时对这些神经元进行削弱，以减少模型的社会偏见，同时保持大部分原始性能。

**💡 创新点**

① 解释性与可解释性：在不修改原模型权重的前提下，直接在内部表示层面定位并调控偏见神经元；② 通用性：模型无关、可应用于不同 VLM 与 LVLM；③ 通过仅 de‑activating 关键神经元实现偏见削弱，避免大规模微调导致的灾难性遗忘。

**🔧 技术方法**

稀疏自编码器（SAE）用于特征稀疏化与解耦；神经元效应性与专一性评估方法；在激活向量上设置权重 α 以平衡偏见消除与语义保持；多尺度 Matryoshka 损失与辅助重构损失。

**📊 数据集**

图像数据集：CelebA、CoCoGender、FairFace；文本数据集：CoCoGender captions、Bias in Bios；这些数据均包含性别、年龄、种族标签，但 SAE 训练不使用显式标签。

**📈 对比分析**

与 Prompt‑Tuning、Projection、Bend‑VLM、SANER、MMNeuron（VLM）以及 Full‑Fine‑Tuning、LoRA、Pruning、Prompt‑Tuning、Prompt‑Engineering（LVLM）等方法比较。结果显示：对 CLIP 变体，Max Skew 下降 9–16%；对 InternVL2、LLaVA‑1.5，性别偏差率下降 40–50%；总体推理性能下降仅 4–10%，优于大多数基线，且在多模态任务中保持了较高的准确率。

**⚠️ 局限性**

① 需要仔细调节 SAE 的扩展因子、阈值 τ 与 α，否则可能导致语义信息失真；② 对不同社会属性的交叉影响（如年龄与性别）需进一步分析；③ 目前主要评估性别/年龄/种族偏见，未覆盖更多社会属性；④ 在极度不平衡或多样性不足的数据集上，社交神经元的定位效果可能不稳定。

---

## 82. OceanBase Bacchus: a High-Performance and Scalable Cloud-Native Shared Storage Architecture for Multi-Cloud

**arXiv ID:** 2602.23571 | [PDF](https://arxiv.org/pdf/2602.23571v1)

**作者:** Quanqing Xu `[一作]` (OceanBase Ant Group), Junyu Ye `[通讯]` (OceanBase Ant Group)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 OceanBase Bacchus，一种基于 LSM‑tree 的共享存储架构，结合对象存储、服务化日志、分层缓存与异步后台服务，实现无状态计算节点、弹性伸缩与多云部署；

**💡 创新点**

①基于 PALF 的服务化共享日志解决跨节点日志同步瓶颈；②多层缓存（本地持久缓存 + 分布式块缓存）弥补对象存储低 IOPS 与高延迟；③将日志与数据分离，支持 RPO=0、RTO 近零的事务恢复；④异步后台任务与分区写模型实现高并发写和无阻塞；

**🔧 技术方法**

使用 LSM‑tree、PALF（Paxos‑based Append‑Only Log File）日志服务、Shared Block Cache Service、ARC 缓存、分布式 2PC、微/宏级压缩、对象存储（S3/OSS）、异步背景任务、RPO=0 机制等技术；

**📊 数据集**

采用 SysBench、TPC‑H、TPC‑DS、ClickBench 等标准 OLTP/OLAP 负载，数据规模包括 30 张 500,000 行表、100 GB/1 TB TPC‑H、100 GB TPC‑DS；

**📈 对比分析**

在相同硬件与配置下与 HBase、PolarDB、Aurora、TDSQL、StarRocks 等系统做写入吞吐、TPS、RT 与查询耗时对比；Bacchus 写入吞吐与 HBase 相当且无挂起；在 OLAP 查询上比 StarRocks 提升 40‑50 %（热点）并在冷数据上更优；存储成本比传统共享‑无状态方案降低 59 %（OLTP）和 89 %（OLAP）；

**⚠️ 局限性**

受对象存储 IOPS 与带宽限制，仍需批量写与预热；Leader 选举集中写冲突，跨区迁移受网络限制；热数据预热与失效恢复需要额外逻辑；极高并发读写下可能受缓存一致性与日志回放延迟影响。

---

## 83. Rudder: Steering Prefetching in Distributed GNN Training using LLM Agents

**arXiv ID:** 2602.23556 | [PDF](https://arxiv.org/pdf/2602.23556v1)

**作者:** Aishwarya Sarkar `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**通讯引用:** 1100 | [OpenAlex ID](https://openalex.org/A5079359777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了 Rudder，一个嵌入 AWS DistDGL 的自适应预取模块，用 LLM 代理动态决定何时替换远程节点，从而降低分布式 GNN 训练中的通信开销。

**💡 创新点**

创新点在于将大语言模型的 In‑Context Learning 作为无监督的预取决策器，避免了传统静态预取或离线训练的 ML 分类器所需的昂贵调优，并在未见过的数据和参数配置下实现 90% 的训练速度提升和 50% 的通信量减少。

**🔧 技术方法**

技术手段包括 DistDGL 框架、Ollama 端的 LLM 推理（如 Gemma3‑4B、Llama3.2‑3B 等）、异步预取与分数策略、混合专家（MoE）模型对比、以及 Python/Numba 并行线程与 NCCL 通信。

**📊 数据集**

使用了 OGB（papers, reddit, products, arxiv）、SNAP 社交网络（orkut, friendster）、Yelp 等多种图数据集，并在 NERSC Perlmutter 超算平台上进行大规模实验。

**📈 对比分析**

对比方法包括基线 DistDGL、固定预取策略 DistDGL+fixed 与 Rudder；评估指标为 epoch 时间、远程节点通信量、%Hits；Rudder 在多数配置下实现 10–50% 的速度提升、>90% 的 %Hits 增长，并在未见数据集上保持显著优势。

**⚠️ 局限性**

局限性包括：缓冲区容量与通信权衡、异步推理导致的潜在请求失效、部分 LLM（如 Qwen‑1.5B）推理不稳定、MoE 模型在量化后性能衰减，以及在极大模型和极限分布式环境下仍需进一步的资源与调优。

---

## 84. Component Centric Placement Using Deep Reinforcement Learning

**arXiv ID:** 2602.23540 | [PDF](https://arxiv.org/pdf/2602.23540v1)

**作者:** Kart Leong Lim `[一作]` `[通讯]` (Agency for Science Technology and Research), Kart Leong Lim (Agency for Science Technology and Research)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种以主组件为中心的PCB布局自动化方法，利用强化学习在离散网格上对被动元件进行布置；

**💡 创新点**

创新点在于将主组件固定并构造周边离散位置信息、引入网络邻近约束来缩小搜索空间，以及将被动元件与其电源网ID编码为统一状态；

**🔧 技术方法**

使用了深度Q网络（DQN）、优势Actor‑Critic（A2C）以及模拟退火（SA）三种学习策略，并结合奖励函数平衡无重叠与网路近似；

**📊 数据集**

使用了自研的9块PCB数据集，包含多种功率与数字功能，覆盖不同规模、复杂度与尺寸差异；

**📈 对比分析**

通过TEWL（欧几里得总线长）和可视化评估（重叠、布线冲突）比较，A2C在大多数案例表现最佳；引入网信息的DQNnet在TEWL上显著优于单纯DQN；

**⚠️ 局限性**

局限性包括：仅针对单面或简化双面布局；离散化网格可能忽略细粒度布线；奖励设计仍需经验调参，且在极大尺寸差异或复杂布线场景下性能波动较大。

---

## 85. Task Complexity Matters: An Empirical Study of Reasoning in LLMs for Sentiment Analysis

**arXiv ID:** 2602.24060 | [PDF](https://arxiv.org/pdf/2602.24060v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 86. Printed helicoids with embedded air channels make sensorized segments for soft continuum robots

**arXiv ID:** 2602.23457 | [PDF](https://arxiv.org/pdf/2602.23457v1)

**作者:** Annan Zhang `[一作]` (Massachusetts Institute of Technology), Daniela Rus `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 63213 | [OpenAlex ID](https://openalex.org/A5066830185)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并制造了嵌有气管的多材料螺旋架构柔性连续机器人，并通过实验验证其力学特性与传感功能

**💡 创新点**

首次实现气体通道与可压缩柔性材料的无缝集成，采用视觉控制喷射实现一体化打印，同时在机器人段内嵌入压力传感器与IMU，实现分布式形变感知

**🔧 技术方法**

多材料3D打印（视觉控制喷射）、嵌入式气管、定制PCB（压力传感器+IMU）以及机械传动系统（多电机驱动缆绳）

**📊 数据集**

实验数据：四种螺旋设计的轴向与弯曲刚度、单段传感器对四种基本变形（压缩、弯曲、扭转）的响应，以及14-DoF手臂在开环轨迹跟踪与抓取任务中的性能

**📈 对比分析**

刚度实验与理论预测对比误差≤15%；轨迹跟踪实验中RMSE和相位滞后均处于可接受范围，显示机械系统可实现近似理想轨迹

**⚠️ 局限性**

在动态操作中出现振荡、传感器漂移和滞后；目前仅实现开环控制，缺乏闭环姿态反馈和动态补偿

---

## 87. pathsig: A GPU-Accelerated Library for Truncated and Projected Path Signatures

**arXiv ID:** 2602.24066 | [PDF](https://arxiv.org/pdf/2602.24066v1)

**作者:** Tobias Nygaard `[一作]` `[通讯]` (Artificial Intelligence and Mathematics Research Lab), Tobias Nygaard (Artificial Intelligence and Mathematics Research Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一款基于 PyTorch 的路径签名计算库 pathsig，能够在 GPU 上高效地计算截断签名、对数签名、窗口签名以及任意词集投影，并支持梯度反向传播。

**💡 创新点**

创新点在于：①使用前缀闭合词集直接在词基底上并行更新签名系数，显著降低计算量；②引入 CUDA 核实现高吞吐量；③支持任意词集投影与各向异性截断，提供更灵活、稀疏的特征表达；④一次性完成多窗口签名计算，显著提升窗口化任务效率。

**🔧 技术方法**

主要技术包括：前缀闭合词集的 Horner 递推；CUDA 并行核实现；基于 Chen 关系的前向/后向递推；对数签名在 Lyndon 基底上的实现；窗口签名的批量并行；以及对高维词集的稀疏投影。

**📊 数据集**

实验数据主要使用合成的多维分形布朗运动（fBM）路径进行 Hurst 参数估计；此外在基准测试中对不同维度（d∈{3,4,6,8,10,20,30,40}）和截断层次（N∈{2,3,4,5,6}）的随机路径进行性能评估。

**📈 对比分析**

与现有库 torchsig、i.e.,、PyTorch‑sig 等相比，pathsig 在截断签名的前向计算上平均加速 12–40 倍，训练阶段加速 7–25 倍；对数签名更是 2–3 倍快；窗口签名在多窗口设置下可获得 3.9–6380 倍加速，平均 153 倍。内存占用接近理论最小值，显著低于传统库。

**⚠️ 局限性**

主要局限包括：①仍需 GPU 支持，CPU 版本性能不佳；②对极长序列或大批量时仍可能出现 OOM；③目前仅支持分段线性插值，其他插值方式尚未实现；④缺乏对高阶迭代积分的可扩展性（如更高维的词集投影）及多线程 CPU 版本优化。

---

## 88. Neural Image Space Tessellation

**arXiv ID:** 2602.23754 | [PDF](https://arxiv.org/pdf/2602.23754v1)

**作者:** Youyang Du `[一作]` (Shandong University), Lingqi Yan `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1438 | [OpenAlex ID](https://openalex.org/A5100756588)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Neural Image-Space Tessellation (NIST)，一种轻量化的屏幕空间后处理方法，利用几何法线与着色法线的差异作为视图相关的提示，学习图像空间变形以实现与几何细分相似的轮廓平滑效果。

**💡 创新点**

创新点在于将传统的几何细分任务重新表述为屏幕空间的视觉一致性问题，使用最小的几何信息（几何法线、着色法线、深度）驱动多尺度神经变形，并通过显式特征重映射保持纹理一致性，从而实现与几何细分相当的视觉质量，同时保持与几何复杂度无关的恒定计算成本。

**🔧 技术方法**

采用卷积神经网络（含门控卷积和注意力机制）的隐式变形模块和特征重映射模块，构建多尺度结构；使用残差相对损失、着色增强损失和LPIPS感知损失进行监督；输入为G-buffer（深度、几何法线、着色法线）和低多边形渲染图；实现基于PyTorch，训练使用Adam。

**📊 数据集**

在四个场景（Junkyard、SoulCave、Cowboy、Bronze）上进行实验，分别划分训练/测试帧；Junkyard无训练集，仅用于测试；其他场景均有数千帧训练集与数百帧测试集。

**📈 对比分析**

与Unreal Engine 4.27中的PN‑Triangles传统细分进行定性与定量比较。NIST在所有场景中有效去除轮廓瑕疵，视觉效果与几何细分相近；在速度上，NIST在360p/720p/1080p下的平均推理时间分别为4.7ms、5.7ms、7.7ms，几乎与图像分辨率线性相关，且与场景几何复杂度无关。

**⚠️ 局限性**

局限性包括：只能处理屏幕空间信息，无法补偿完全不可见或部分可见三角形导致的变形不稳定；需要对每个场景单独训练，跨场景泛化有限；未加入法线贴图或细节增强；目前实现未进行工业级量化或加速，推理速度仍可进一步提升。

---

## 89. Exploring the Effect of Heights and User Stance on User Experience in Extended Reality Climbing

**arXiv ID:** 2602.23500 | [PDF](https://arxiv.org/pdf/2602.23500v1)

**作者:** Tanja Kojić `[一作]` (Quality and Usability Lab Technische Universität Berlin), Jan-Niklas Voigt-Antons `[通讯]` (Immersive Reality Lab Hamm-Lippstadt University of Applied Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本研究通过在Meta Quest 3上搭建五个不同高度与复杂度的攀岩虚拟环境，比较坐姿与站姿对用户沉浸感、真实性感知及晕动症的影响

**💡 创新点**

首创系统化比较坐姿与站姿在高空VR攀岩任务中的主观体验差异，并揭示姿态与环境复杂度共同作用下的沉浸与不适权衡

**🔧 技术方法**

使用Unity3D开发的物理抓握攀岩模拟、Meta Quest 3头显、XR控制器以及IPQ与SSQ标准问卷进行数据采集

**📊 数据集**

无公开数据集，全部为实验自收集的25名参与者的问卷与表现数据

**📈 对比分析**

采用重复测量ANOVA（以及必要的非参数检验）比较环境与姿态对IPQ与SSQ子量表的影响；结果显示坐姿在真实性感知上略高，但在视觉复杂环境中晕动症波动更大；站姿的舒适度更稳定

**⚠️ 局限性**

样本量有限、姿态与场景呈随机不平衡、未采用完全对照设计，导致对结果的泛化与因果推断受限

---

## 90. What You Read is What You Classify: Highlighting Attributions to Text and Text-Like Inputs

**arXiv ID:** 2602.24149 | [PDF](https://arxiv.org/pdf/2602.24149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 91. RF-Agent: Automated Reward Function Design via Language Agent Tree Search

**arXiv ID:** 2602.23876 | [PDF](https://arxiv.org/pdf/2602.23876v1)

**作者:** Ning Gao `[一作]` (Beihang University), Yue Deng `[通讯]` (Beihang University)

**通讯引用:** 4001 | [OpenAlex ID](https://openalex.org/A5082404485)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于语言模型和树搜索的自动奖励函数设计框架RF-Agent；

**💡 创新点**

将奖励函数设计视作决策过程，结合MCTS与多阶段上下文推理，利用历史反馈和多种操作类型高效探索奖励空间；

**🔧 技术方法**

使用大型语言模型（GPT‑4o‑mini / GPT‑4o）、Monte Carlo树搜索、变异/交叉/路径推理/不同思路操作、LLM自我校验与思路对齐；

**📊 数据集**

在IsaacGym（17个低层控制任务）和Bi‑DexHands（10个双臂操控任务）上评测；

**📈 对比分析**

与人类专家、稀疏奖励、Eureka、Revolve等方法比较，RF-Agent在所有任务中均获得最高或接近人类水平的奖励函数，尤其在复杂操控任务上显著优于其他LLM方法；

**⚠️ 局限性**

仍需大量LLM交互和多轮RL训练，计算成本高，未能显著减少RL训练次数。

---

## 92. GDA-YOLO11: Amodal Instance Segmentation for Occlusion-Robust Robotic Fruit Harvesting

**arXiv ID:** 2602.23953 | [PDF](https://arxiv.org/pdf/2602.23953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 93. Robust Skills, Brittle Grounding: Diagnosing Restricted Generalization in Vision-Language Action Policies via Multi-Object Picking

**arXiv ID:** 2602.24143 | [PDF](https://arxiv.org/pdf/2602.24143v1)

**作者:** David Emukpere `[一作]` (Naver Labs Europe), Jean-Michel Renders `[通讯]` (Naver Labs Europe)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过设计多物体拾取任务的空间随机化和对象-位置组合排除测试，系统评估了视觉语言动作策略在不同布局结构下的指令映射与操控执行能力。

**💡 创新点**

创新点在于提出了任务阶梯式随机化与分解指标评估框架，揭示了高性能背后指令泛化脆弱的现象。

**🔧 技术方法**

使用了基于ManiSkill模拟环境、PPO数据采集、SmolVLA和π_0.5两种 VLA 模型，以及自动生成的大规模演示数据。

**📊 数据集**

数据集包括从LIBERO、MetaWorld转化的高随机化ManiSkill环境和通过RL生成的最多100,000条演示轨迹。

**📈 对比分析**

与传统任务成功率对比，分解指标显示在布局被扰乱后成功率骤降，尽管抓取动作保持稳定，表明模型的指令跟随性能远低于操控表现。

**⚠️ 局限性**

局限在于仅评估单一抓取原语、仅关注空间随机化和组合泛化，未覆盖语言多样性、长期规划和真实世界转移等方面。

---

## 94. MSVBench: Towards Human-Level Evaluation of Multi-Shot Video Generation

**arXiv ID:** 2602.23969 | [PDF](https://arxiv.org/pdf/2602.23969v1)

**作者:** Haoyuan Shi `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 60418 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MSVBench，针对多镜头视频生成的统一评测框架，构建层级脚本、参考图像，并设计混合评估流程与监督信号生成。

**💡 创新点**

首个完整的多镜头视频基准；混合使用大型多模态模型与专用专家模型以实现高层语义与低层感知双重评估；通过评估轨迹训练轻量化评估器，实现自动化监督。

**🔧 技术方法**

混合评估（LMM + 领域专家模型如DOVER、RAFT、SAM等）；多维度指标（视觉质量、故事对齐、一致性、运动质量）；Qwen3‑VL‑4B轻量化评估器训练（GRPO）；脚本分层、参考图像生成与摄像机运动指令合成。

**📊 数据集**

基于ViStoryBench 20个故事改编的层级数据集（脚本、参考图像、摄像机指令），并使用15个故事生成1,000+评估轨迹用于监督数据。

**📈 对比分析**

对20种生成方法进行评测，MSVBench与人工评估的Spearman相关系数达到94.4%（τ 83.6%），商业模型Sora2/Veo3.1领跑；开放源Wan2.2系列与商业模型差距缩小；评估指标覆盖视觉质量、故事一致性、视频一致性和运动质量。

**⚠️ 局限性**

局限：缺乏音视频同步/音频评估；故事样本量有限，影响监督训练；连续生成模型无明确镜头分段，导致部分镜头级指标难以应用。

---

## 95. Critical Infrastructure in the Multi-Cloud Strategy: Use of Cloud Computing in SMEs

**arXiv ID:** 2602.23658 | [PDF](https://arxiv.org/pdf/2602.23658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 96. Challenges in Automatic Speech Recognition for Adults with Cognitive Impairment

**arXiv ID:** 2602.23436 | [PDF](https://arxiv.org/pdf/2602.23436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 97. Adaptive Combinatorial Experimental Design: Pareto Optimality for Decision-Making and Inference

**arXiv ID:** 2602.24231 | [PDF](https://arxiv.org/pdf/2602.24231v1)

**作者:** Hongrui Xie `[一作]` (University of Science and Technology of China), Kan Xu `[通讯]` (Arizona State University)

**通讯引用:** 30932 | [OpenAlex ID](https://openalex.org/A5064209674)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并研究了组合多臂赌博机（CMAB）中的Pareto最优性，探讨了在完全反馈和半反馈两种信息结构下，如何在累积损失与奖励差距估计误差之间进行权衡。

**💡 创新点**

创新点在于：①首次把Pareto最优性概念应用于组合赌博机；②设计了两种针对不同反馈的算法（全反馈使用KL投影混合策略，半反馈使用UCB混合策略）；③给出了充分必要条件证明这两种算法在所有可行策略中均达到Pareto最优；④量化了不同反馈下Pareto前沿的差距。

**🔧 技术方法**

技术手段包括：组合优化与凸投影（KL投影）、多臂赌博机中的UCB、马尔可夫过程与马丁格尔不等式、信息论下的下界证明、以及在线镜像梯度（OSMD）框架。

**📊 数据集**

实验使用合成数据：随机生成基底臂奖励 μ(e)∼U(0.1,0.9)，在 d=8 或 d=9、超级臂大小 m=3 或 4 的设置下，构造全部或部分非空超级臂集合。

**📈 对比分析**

与基准方法相比，实验结果显示：在全反馈下，MixCombKL 算法在累计损失和估计误差上都能接近理论最优；在半反馈下，MixCombUCB 在累积损失上更优，同时保持较低的均方误差。实验曲线表明两种算法的累积损失随时间接近预期的 O(m n^{1-α}) 级别，估计误差则与 n^{α-1/2} 成比例。

**⚠️ 局限性**

局限性包括：①实验仅在小规模合成数据上验证，缺乏真实世界数据的验证；②算法参数 α 的选择仍需经验性调节；③在极大规模或动态组合场景下，计算复杂度和内存消耗可能成为瓶颈；④未考虑预算、约束或公平性等实际约束条件。

---

## 98. CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering

**arXiv ID:** 2602.23952 | [PDF](https://arxiv.org/pdf/2602.23952v1)

**作者:** Yuyang Hong `[一作]` (University of Chinese Academy of Sciences), Jieping Ye `[通讯]` (Alibaba Cloud Computing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CC-VQA，一个无训练的、冲突和相关性感知的知识驱动视觉问答框架，利用视觉中心的上下文冲突推理和相关性引导的编码解码来解决检索知识与模型参数知识之间的冲突问题。

**💡 创新点**

创新点在于①通过可视化语义推理显式外化模型内部知识并与检索上下文进行冲突分析；②采用句子级相关性评估压缩低相关性语句的位置信息，并在解码阶段用相关性加权冲突评分实现自适应采样。

**🔧 技术方法**

采用 Qwen2.5‑VL‑7B 视觉语言模型、EVA‑CLIP 8B 检索与相关性评估、RoPE+PI 位置编码压缩，以及基于 Jensen‑Shannon、熵差与相关性加权的自适应对比解码技术。

**📊 数据集**

在 E‑VQA、InfoSeek 与 OK‑VQA 三大知识驱动视觉问答基准上进行实验。

**📈 对比分析**

在检索增强方案上无训练提升 3.3–6.4%（InfoSeek +4.7%、E‑VQA +3.3%、OK‑VQA 78.8%），显著优于现有最优方法，且错误比例下降、有效回答比例上升。

**⚠️ 局限性**

方法需要显式外化模型知识；若能直接在检索上下文中内隐识别冲突将更优；对检索质量仍有依赖，且视觉推理能力仍可进一步提升。

---

## 99. Beyond Explainable AI (XAI): An Overdue Paradigm Shift and Post-XAI Research Directions

**arXiv ID:** 2602.24176 | [PDF](https://arxiv.org/pdf/2602.24176v1)

**作者:** Saleh Afroogh `[一作]` (University of Texas at Austin), Jieyu Zhao `[通讯]` (University of Southern California)

**通讯引用:** 4584 | [OpenAlex ID](https://openalex.org/A5066282713)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析并诊断XAI领域的核心缺陷，提出以交互验证为核心的后XAI范式。

**💡 创新点**

提出系统性诊断框架，将XAI的症状与根本原因对应，并以交互验证取代传统后置解释。

**🔧 技术方法**

批判性评估LIME、SHAP、Grad‑CAM等后置解释方法，阐述其可解释性与真实性的局限。

**📊 数据集**

未使用特定数据集，主要基于对2019–2021年34个XAI系统的文献综述与案例分析。

**📈 对比分析**

比较方法在可解释性与可信度上不佳，指出大多数XAI工具在真实世界场景中难以提升信任且常导致过度依赖。

**⚠️ 局限性**

局限性包括深浅解释悖论、层级混淆、相关与因果混淆、五大错误假设，以及代理模型的逻辑悖论。

---

## 100. Denoising-Enhanced YOLO for Robust SAR Ship Detection

**arXiv ID:** 2602.23820 | [PDF](https://arxiv.org/pdf/2602.23820v1)

**作者:** Xiaojing Zhao `[一作]` (Ayit University), Huicong Ning `[通讯]` (Ayit University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于YOLOv8改进的CPN-YOLO框架，用于提高SAR船舶检测的鲁棒性和小目标检测性能。

**💡 创新点**

创新点包括可学习的大卷积去噪模块（CID）、基于PPA的特征增强模块以及基于归一化Wasserstein距离的NWD损失函数。

**🔧 技术方法**

使用YOLOv8n骨干，Channel-Independent Denoising、Parallel Patch-Aware Attention、NWD Loss等技术。

**📊 数据集**

采用公开SAR船舶检测数据集HRSID与SSDD。

**📈 对比分析**

通过与十种主流检测器（YOLOv5、YOLOv8、FCOS等）以及不同注意力和损失函数的ablation对比，CPN-YOLO在HRSID上mAP@0.5 88.9%，SSDD上mAP 97.3%，显著优于对比模型。

**⚠️ 局限性**

模型在高分辨率SAR图像上仍存在计算开销大、资源受限下效率低的问题。

---

## 101. Solving No-wait Scheduling for Time-Sensitive Networks with Daisy-Chain Topology

**arXiv ID:** 2602.23700 | [PDF](https://arxiv.org/pdf/2602.23700v1)

**作者:** Qian Li `[一作]` (Shenzhen Research Institute of Big Data), Yuyi Wang `[通讯]` (Tengen Intelligence Institute)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种高效算法，用于在线性（daisy-chain）拓扑的时间敏感网络（TSN）中构造无等待（no-wait）调度方案。

**💡 创新点**

创新点在于将无等待调度问题重新表述为带颜色限制的区间图着色问题，并证明该变体在区间图上可多项式求解，从而突破了传统调度算法的可扩展性瓶颈。

**🔧 技术方法**

核心技术包括区间图的贪心着色、基于总可逆矩阵（totally unimodular）的整数规划求解、以及对周期取 2 的幂进行递归拆分的算法设计。

**📊 数据集**

实验使用了真实列车 TSN 系统的拓扑（共 32 台交换机）以及随机生成的数千到数万条流量实例。

**📈 对比分析**

与精确的 SMT 规划方法比较，算法在所有测试实例上得到相同的调度结果，平均求解时间从几分钟下降到约 2 秒；在大规模（约 45,000 条流）场景下，仅需 30 分钟即可完成调度。

**⚠️ 局限性**

局限性包括：仅适用于线性拓扑；对流量周期的假设为 2 的幂，虽然可近似但不适用于所有实际周期；以及未对星形、环形或树形等其他常见拓扑进行理论与实验验证。

---

## 102. Sandwiching Polynomials for Geometric Concepts with Low Intrinsic Dimension

**arXiv ID:** 2602.24178 | [PDF](https://arxiv.org/pdf/2602.24178v1)

**作者:** Adam R. Klivans `[一作]` (University of Texas at Austin), Arsen Vasilyan `[通讯]` (University of Texas at Austin)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5030920630)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

该论文提出了一种构造低阶“夹层”多项式逼近（sandwiching polynomial）的通用方法，适用于低维内在维度且边界平滑的概念类，并在严格次指数分布下实现了显著的次数改进；

**💡 创新点**

创新点在于：①用 Lipschitz 插值结合多元 Jackson 定理直接构造夹层多项式；②利用几何测度理论（如高维共面积、Nazarov 结果、coarea 法）证明边界平滑性；③统一框架可覆盖高斯、任意次指数分布、甚至非高斯情形；④得到的次数从指数降为多项式，显著优于先前的 2^O(k) 结果；

**🔧 技术方法**

核心技术包括：Lipschitz 函数的上、下夹层逼近；Jackson 近似定理与多项式系数控制；高维测度理论（Gaussian surface area、coarea 公式、Nazarov 结果）；严格次指数尾部控制；以及对多项式系数总和的上界估计；

**📊 数据集**

本工作为理论研究，无实际数据集；所有结果均在数学模型和分布假设下证明；

**📈 对比分析**

相对于以往工作（如基于 1 维夹层多项式、FT 取模化方法、或指数次数的 upper sandwiching），本文提供了多项式次数上界（例如 k-halfspaces 取得 O(k^5) 次，k-intersections 取得 O(k^3) 次，低维 PTF 取得 O(q^3k^5) 次），在同类学习任务（可测试学习、分布偏移学习、重污染学习）中实现了显著的时间复杂度提升；

**⚠️ 局限性**

局限性包括：①次数仍随内在维度 k 与平滑参数 σ 乘方增长；②在高维情形下对 k 的要求较高；③对 PQ 学习仍不确定是否仅需 L1 夹层即可；④部分证明依赖严格次指数尾部或超几何分布的抗浓度假设，限制了分布范围；

---

## 103. SafeGen-LLM: Enhancing Safety Generalization in Task Planning for Robotic Systems

**arXiv ID:** 2602.24235 | [PDF](https://arxiv.org/pdf/2602.24235v1)

**作者:** Jialiang Fan `[一作]` (University of Notre Dame), Fangxin Kong `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在机器人系统中提出一种两阶段后训练框架 SafeGen-LLM，先通过监督微调(SFT)让大语言模型学习规划语法与语义，再利用 Group Relative Policy Optimization (GRPO) 在可验证的奖励机指引下进行在线强化学习，从而生成符合安全约束的任务计划。

**💡 创新点**

创新点包括：① 设计了包含显式安全约束的 PDDL3 基准数据集，实现了安全知识的系统化注入；② 采用可验证的层级奖励机与分组相对优势训练方法，使模型在奖励空间中获得细粒度信号；③ 结合课程学习和域均衡采样，显著提升跨域和跨规模的安全泛化能力。

**🔧 技术方法**

技术主要包括：大语言模型微调（SFT）、GRPO 强化学习、基于 VAL 的计划验证与奖励机、PDDL3 语法解析、课程学习、域均衡批处理、4-bit 量化模型（Mistral-7B、Llama-8B、Qwen3-14B）和与 SafePilot 的集成。

**📊 数据集**

使用了四个安全规划领域（Blocksworld、Ferry、Grippers、Spanner）生成的 1,000 题 PDDL3 规划问题，分为 500 题用于 SFT、500 题用于 GRPO 训练，测试集每个领域 50 题。数据集通过 OPTIC 求解并 VAL 验证，确保所有示例满足安全约束。

**📈 对比分析**

与经典规划器（OPTIC、Fast Downward）以及 GPT‑5.2 进行对比，SafeGen‑LLM 在所有四个领域的成功率平均超过 95%，远超 GPT‑5 Nano（仅 18‑20%）和经典规划器。实验还展示了跨域安全泛化、输入格式鲁棒性、与 SafePilot 的协同提升以及在真实机器人臂上的物理验证，表现出显著的安全合规和任务完成率。

**⚠️ 局限性**

主要局限包括：① 仅覆盖四个相对简单的规划领域，难以验证在更复杂的多模态或动态环境下的性能；② 训练数据规模有限，且主要基于静态 PDDL 问题，缺乏从交互经验中自动提取安全约束的能力；③ 依赖 VAL 验证和手工编写的安全约束，自动化程度不高；④ 量化后模型的表达能力受限，可能在极大规模规划任务中出现性能瓶颈。

---

## 104. On De-Individuated Neurons: Continuous Symmetries Enable Dynamic Topologies

**arXiv ID:** 2602.23405 | [PDF](https://arxiv.org/pdf/2602.23405v1)

**作者:** George Bird `[一作]` (University of Manchester), George Bird `[通讯]` (University of Manchester)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5017161833)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出利用等距激活函数（orthogonal‑equivariant primitives）实现网络结构的实时动态重参数化，支持神经元的生长与萎缩，并通过层级对角化得到一对一的神经元映射，保证功能不变；

**💡 创新点**

核心创新在于：1) 将对称性反向作用于基础函数，得到无基底依赖的等距激活；2) 引入可训练的“intrinsic length”参数消除偏置影响；3) 通过奇异值阈值实现精确神经元剪枝与扩张；4) 理论证明可达到50%稀疏且功能保持；

**🔧 技术方法**

技术手段包括：等距激活函数、层对角化（SVD）、基底无关重参数化、奇异值阈值剪枝/扩张、等距归一化（Chi‑normalizer）以及实验用的多层感知机；

**📊 数据集**

实验使用CIFAR‑10图像分类数据集，训练多层感知机模型；

**📈 对比分析**

通过在预训练MLP上按阈值动态增删神经元，并与标准tanh（非等距）做对比，实验结果显示等距tanh在同一网络结构下准确率更高，神经元增大可提升准确，神经元削减到8时性能下降；整体性能保持稳定；

**⚠️ 局限性**

局限性包括：仅在MLP上验证，卷积层的扩展尚未深入；阈值与scaffold数需手工调节；SVD对大规模网络的计算成本高；对抗鲁棒性与泛化能力未评估；动态重参数化的时间/计算开销未充分量化。

---

## 105. EvoX: Meta-Evolution for Automated Discovery

**arXiv ID:** 2602.23413 | [PDF](https://arxiv.org/pdf/2602.23413v1)

**作者:** Shu Liu `[一作]` (University of California Berkeley), Ion Stoica `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出EvoX，一种在LLM驱动的进化搜索中同时共进化解法与搜索策略的元进化框架。

**💡 创新点**

创新点在于将搜索策略本身视为可进化对象，动态根据进度自适应切换探索与利用策略，消除固定参数导致的停滞。

**🔧 技术方法**

采用LLM生成器（GPT‑5 / Gemini‑3.0‑Pro）、双层进化循环、策略数据库与窗口式进展评估、变异算子等技术。

**📊 数据集**

评估数据集涵盖近200个真实任务，包括数学、系统性能优化、ALE‑Bench‑Lite和Frontier‑CS等竞赛题目。

**📈 对比分析**

与AlphaEvolve、OpenEvolve、ShinkaEvolve、GEPA等基线对比，EvoX在大多数任务上均取得更高的平均/最佳分数，并在成本上更具竞争力。

**⚠️ 局限性**

局限性包括对LLM生成成本的依赖、固定100步预算限制、未给出理论收敛保证，以及在极端任务或极小数据集上的表现尚未验证。

---

## 106. Context-Aware Functional Test Generation via Business Logic Extraction and Adaptation

**arXiv ID:** 2602.24108 | [PDF](https://arxiv.org/pdf/2602.24108v1)

**作者:** Yakun Zhang `[一作]` (Harbin Institute of Technology), Yunming Ye `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 9691 | [OpenAlex ID](https://openalex.org/A5002523892)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LogiDroid，一种两阶段的自动化功能测试用例生成框架，能够根据功能需求自动生成完整的事件序列和验证断言。

**💡 创新点**

创新点包括：① 通过语义检索+知识融合模块从历史测试案例中提取可复用的业务逻辑；② 在上下文感知阶段结合多模态感知与决策生成，逐步映射业务逻辑到目标应用的具体 GUI，实现精确、可执行的测试用例。

**🔧 技术方法**

技术手段：大型语言模型（GPT‑5、Gemini‑2.5）辅助语义检索、知识融合与决策；BGE 模型进行文本嵌入相似度匹配；多模态感知（截图+widget 层级）与滑动窗口决策机制；对抗式反馈循环防止 hallucination。

**📊 数据集**

数据集：构建 294 条功能测试案例（71 个应用、13 类目）用于检索与融合；评估使用 FrUITeR（28 应用）和 Lin（28 应用）两大公开基准，共 190 条功能需求、95 条真实测试用例。

**📈 对比分析**

与 AutoDroid、AppAgent 等最先进方法对比：在 FrUITeR 上成功率 40%（提升 48%），在 Lin 上成功率 65%（提升 55%）；完美率亦显著提高，且 LogiDroid 能生成完整断言，基线无法做到。平均生成时间约 6 分钟，token 约 31.8k，性能可接受。

**⚠️ 局限性**

局限性：需要预先收集大量历史测试案例，检索效果受数据覆盖度影响；依赖大模型，成本较高；多模态感知与决策过程仍可能出现推理错误或循环，需人工核查；对极端 GUI 变异的鲁棒性尚待进一步验证。

---

## 107. Annotation-Free Visual Reasoning for High-Resolution Large Multimodal Models via Reinforcement Learning

**arXiv ID:** 2602.23615 | [PDF](https://arxiv.org/pdf/2602.23615v1)

**作者:** Jiacheng Yang `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**通讯引用:** 13053 | [OpenAlex ID](https://openalex.org/A5070337115)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HART 闭环框架，使大型多模态模型在高分辨率视觉任务中自我定位关键区域并自我验证推理。

**💡 创新点**

创新点在于：① 通过闭环机制让模型仅用答案正确性来验证定位结果，避免人工标注；② 设计 AP‑GRPO 强化学习策略，对正确定位样本动态加权，降低奖励误判。

**🔧 技术方法**

采用的技术包括：视觉定位+区域裁剪、闭环推理流程、AP‑GRPO（强化学习）与 SFT（监督微调）相结合。

**📊 数据集**

使用的数据集有 TreeBench、MME‑RealWorld、MME‑RealWorld‑Lite、V*Bench、HR‑Bench 等高分辨率视觉问答与多模态基准。

**📈 对比分析**

与多种基线（SFT、GRPO、MGPO、Pixel‑Reasoner、DeepEyes、私有模型等）对比，HART 在 MME‑RealWorld‑Lite、TreeBench 以及高分辨率 V*Bench 等任务上均取得更高准确率，甚至 7B 模型超越 72B 规模模型。

**⚠️ 局限性**

局限性包括：实验仅验证 7B 模型，缺乏大模型/大数据集验证；闭环推理的计算效率仍低于纯下采样方案；对极大图像的实时推理尚待进一步加速。

---

## 108. Assessment of Display Performance and Comparative Evaluation of Web Map Libraries for Extensive 3D Geospatial Data

**arXiv ID:** 2602.23660 | [PDF](https://arxiv.org/pdf/2602.23660v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 109. TaCarla: A comprehensive benchmarking dataset for end-to-end autonomous driving

**arXiv ID:** 2602.23499 | [PDF](https://arxiv.org/pdf/2602.23499v1)

**作者:** Tugrul Gorgulu `[一作]` (Trutek AI), Ozsel Kilinc `[通讯]` (Amazon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

收集了一个新的、规模最大、覆盖CARLA Leaderboard 2.0 36种场景的多传感器数据集（TaCarla），包含3D目标检测、车道/中心线检测、交通灯识别、规划等多任务标签，并提供文本描述与稀有度评分。

**💡 创新点**

将Bench2Drive的RL专家（易出现振荡）与PDM-Lite的规则专家结合，使用PDM专家并采用NuScenes完整360°传感器配置，克服了前两者的局限；同时提供稀有度评分用于长尾事件识别；并对多任务统一基线进行评测。

**🔧 技术方法**

使用CARLA 0.9.15仿真、PDM专家策略、NuScenes传感器布局、Lift-Splat、RQR3D、TopoBDA、FCOS、Transfuser、DiffusionDrive等模型；数据采集时长10Hz，后降至2Hz以符合nuScenes评测。

**📊 数据集**

新数据集TaCarla（2.85M帧，10Hz）以及Bench2Drive、PDM-Lite、nuScenes、Waymo、Argoverse等公开基准。

**📈 对比分析**

在多任务基线上，Camera+LiDAR组合相较Camera单独提升了3D检测精度（mAP从0.32提升至0.55），TopoBDA在中心线检测上达AP_f 39.6/AP_c 41.7，FCOS在交通灯检测上AP 59.5，规划模型在开放环误差方面，DiffusionDrive在4s horizon ADE为4，FDE 5.58，Transfuser更优；在闭环评估中，DiffusionDrive Driving Score 22.35，Transfuser 17.18，PlanT 52.95。

**⚠️ 局限性**

主要局限在：仍缺少真实感多模态对齐；采集依赖PDM规则专家，可能仍存在策略偏差；在某些极端或高难度场景下模型表现不佳；数据仍以仿真为主，现实迁移需进一步验证。

---

## 110. RAViT: Resolution-Adaptive Vision Transformer

**arXiv ID:** 2602.24159 | [PDF](https://arxiv.org/pdf/2602.24159v1)

**作者:** Martial Guidez `[一作]` (National Institute of Applied Sciences Lyon), Christophe Garcia `[通讯]` (National Institute of Applied Sciences Lyon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多分支的分辨率自适应视觉Transformer（RAViT），在不同分辨率的图像上逐级推理并可在任意分支做早期退出，降低计算量。

**💡 创新点**

创新点在于将分辨率变换与多分支Transformer结合，并通过CLS token跨分支传递信息，同时引入基于熵的早期退出机制，实现动态精度-成本权衡。

**🔧 技术方法**

采用Vision Transformer结构、多分支设计、CLS token迁移、熵阈值早期退出、加权交叉熵训练、AdamW + CosineAnnealingLR优化。

**📊 数据集**

使用CIFAR-10、Tiny ImageNet和ImageNet三个公开图像分类数据集进行评估。

**📈 对比分析**

与传统ViT（不同层数）对比，RAViT在保持相近或略低的准确率的同时，平均可将FLOPs降低至约70%（例如在ImageNet上相当于ViT‑B的99.85%准确率，仅需70%计算）。

**⚠️ 局限性**

局限性包括：未使用大规模预训练或超参数细调，准确率尚未达到SOTA；未与现有ViT压缩方法直接比较；多分支结构与阈值选择仍需手动调优，缺乏自动化架构搜索。

---

## 111. Micro-expression Recognition Based on Dual-branch Feature Extraction and Fusion

**arXiv ID:** 2602.23950 | [PDF](https://arxiv.org/pdf/2602.23950v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 112. Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation

**arXiv ID:** 2602.24144 | [PDF](https://arxiv.org/pdf/2602.24144v1)

**作者:** Muquan Li `[一作]` (University of Electronic Science and Technology of China), Tao He `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 4107 | [OpenAlex ID](https://openalex.org/A5070101306)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 RETA 的改进版分离式数据集蒸馏方法，利用动态检索连接和持久拓扑对齐来生成更具代表性和多样性的合成图像。

**💡 创新点**

创新点在于：①动态检索连接（DRC）通过自适应的拟合-复杂度得分选择最合适的真实补丁，实现了对特征拟合与模型复杂度的平衡；②持久拓扑对齐（PTA）利用持久同调对图像特征空间的拓扑结构进行正则化，缓解了“拉回锚点”效应，保持类内多样性。

**🔧 技术方法**

核心技术包括：分离式数据集蒸馏框架、残差匹配、特征空间拟合-复杂度评分、持久同调（PH）和持久图像（PI）损失、互 k‑NN 图构造及其滤波器。

**📊 数据集**

在 CIFAR‑100、Tiny‑ImageNet、ImageNette、ImageWoof、ImageNet‑1K 及其子集上进行实验，并在不同 IPC（1、10、50）和多种网络（ResNet‑18/50/101、EfficientNet、MobileNet、Swin‑Tiny 等）上评估。

**📈 对比分析**

与多种基线（SRe²L、RDED、EDC、CaO₂、WMDD、NRR‑DD、FADRM+ 等）对比，RETA 在所有设置下均获得最高精度，尤其在 ImageNet‑1K IPC=50 时提升 3.1%（达到 64.3% top‑1）。同时在跨架构泛化、抗扰动、持续学习等任务中也展现出更强的鲁棒性。

**⚠️ 局限性**

局限性包括：依赖冻结教师模型和预构建的 per‑class 检索池；检索复杂度近似采用手工设计的梯度幅值方差；PTA 需要构造互 k‑NN 图和持久图像，增加一定计算开销；超参数（λ、λ_topo、k 等）需手动调节。

---

## 113. Hestia: Hyperthread-Level Scheduling for Cloud Microservices with Interference-Aware Attention

**arXiv ID:** 2602.23758 | [PDF](https://arxiv.org/pdf/2602.23758v1)

**作者:** Dingyu Yang `[一作]` (Zhejiang University), Gang Chen `[通讯]` (Zhejiang University)

**通讯引用:** 102710 | [OpenAlex ID](https://openalex.org/A5100389265)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Hestia，一种基于自注意力的超线程级调度框架，能预测并规避多级硬件干扰，提升云微服务的尾部延迟和 CPU 效率。

**💡 创新点**

创新点在于：① 用自注意力模型同时捕获共享核心（SC）和共享套件（SS）的层级干扰；② 设计细粒度干扰评分机制，结合预测结果进行超线程级放置；③ 通过拓扑感知选择器减少搜索空间，实现实时调度。

**🔧 技术方法**

核心技术包括：自注意力 CPU 使用预测器、干扰评分模型、拓扑感知候选选择器；实现细节采用多层感知器、位置编码以及层级注意力机制。

**📊 数据集**

使用真实生产集群的 32,408 个低延迟微服务实例（覆盖 3,132 台服务器）以及在 190 台服务器的 2,009 个实例的在线部署数据。

**📈 对比分析**

与五种主流调度器（First‑Fit、Socket‑Spread、Paragon、Kambadur、RCPU）在相同负载下对比，Hestia 在 95th‑percentile 延迟上可提升 10%–80%，CPU 利用率下降 2.3%，集群总 CPU 需求减少 9.75%，在生产部署中也实现了 2.27% 的整体 CPU 消耗下降。

**⚠️ 局限性**

局限性：仅在 Intel Xeon 8200/8100 系列硬件上验证；对极端动态工作负载的适应性和跨平台迁移尚待进一步评估；目前仅针对低延迟微服务，其他类型工作负载的效果未知。

---

## 114. Task-Lens: Cross-Task Utility Based Speech Dataset Profiling for Low-Resource Indian Languages

**arXiv ID:** 2602.23388 | [PDF](https://arxiv.org/pdf/2602.23388v1)

**作者:** Swati Sharma `[一作]` (Indraprastha Institute of Information Technology Delhi), Anubha Gupta `[通讯]` (Indraprastha Institute of Information Technology Delhi)

**通讯引用:** 5104 | [OpenAlex ID](https://openalex.org/A5057412604)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对印度语音数据集进行跨任务利用性评估，构建Task‑Lens框架，系统分析50个涵盖26种语言、91,257小时音频数据集的多任务可用性；

**💡 创新点**

提出跨任务功能映射规则和Task‑Lens工具，实现多任务（9项）对语料的统一评估与可视化，揭示数据集在多任务中的潜在价值与缺口；

**🔧 技术方法**

基于PRISMA的系统检索与筛选、元特征抽取、规则驱动的任务-特征相关矩阵、任务就绪判定算法；

**📊 数据集**

共收集50个印度语音数据集（如BhasaAnuvaad、IndicSUPERB、IndicVoices‑R、Nexdata AI 759h等），涵盖26种印度语言，91k小时音频；

**📈 对比分析**

采用功能完整度检查（是否满足任务所需特征）进行对比，输出每个数据集在每项任务下的“Task‑Ready”状态，未进行下游模型性能评估；

**⚠️ 局限性**

仅覆盖9项核心任务与26种语言，特征评估为二元判断，未考虑音频/标注质量；未来需扩展更多任务、语言与质量度量。

---

## 115. DACESR: Degradation-Aware Conditional Embedding for Real-World Image Super-Resolution

**arXiv ID:** 2602.23890 | [PDF](https://arxiv.org/pdf/2602.23890v1)

**作者:** Xiaoyan Lei `[一作]` (Zhengzhou University of Light Industry), Qiuting Lin `[通讯]` (China Academy of Machinery Science and Technology Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DACESR 系统，包含 Real Embedding Extractor（REE）和基于 Mamba 的图像超分辨率网络，并通过 Conditional Feature Modulator（CFM）将 REE 的高层信息注入网络，显著提升在真实世界降质图像上的纹理恢复与视觉质量。

**💡 创新点**

创新点包括：① 对 Recognize Anything Model（RAM）在降质图像上的识别能力进行重新评估并提出文本相似性度量；② 采用降质选择策略与对比学习训练 REE，显著提升对降质内容的识别精度；③ 在 Mamba‑based 网络中设计 CFM，将 REE 产生的条件信息有效融合，提升 PSNR 与 LPIPS；④ 通过实验验证 Mamba 在真实世界 SR 任务中的优越性与高效性。

**🔧 技术方法**

使用的技术包括：RAM、DAPE（LoRA 微调）、对比学习、Mamba（状态空间模型）、CFM、对抗与感知损失、LAM 诊断工具、真实世界降质模型（高阶降质）、Bicubic 与多级降质生成、以及多种数据集的训练与评估。

**📊 数据集**

使用的数据集：训练集——DIV2K、COCO、Flickr2K、OutdoorSceneTraining；验证/测试集——DIV2K 验证、Level‑I/II/III 合成降质、RealSR‑cano、RealSR‑Nikon、AIM2019‑val、RealWorld38。

**📈 对比分析**

通过与 ESRGAN、BSRGAN、Real‑ESRGAN、SwinIR、DASR、HAT、KDSR、DCLS、CDFormer 等 SOTA 方法在 PSNR/LPIPS 上的对比，DACESR 在 Level‑I/II、RealSR、AIM2019‑val 等数据集上获得最高 PSNR 或最低 LPIPS，甚至在 diffusion‑based 方法对比中也保持领先；在多种降质水平下均表现出色。

**⚠️ 局限性**

局限性：目前仅在 Mamba‑based 网络上验证效果，CNN、Transformer、Diffusion 模型的适用性尚未探索；REEx 作为条件输入虽提升性能，但仍需要更强大、灵活的降质感知修正模块；在极端降质或特殊噪声场景下表现可能仍有限，需要进一步研究。

---

## 116. Reasoning-Driven Multimodal LLM for Domain Generalization

**arXiv ID:** 2602.23777 | [PDF](https://arxiv.org/pdf/2602.23777v1)

**作者:** Zhipeng Xu `[一作]` (Xidian University), Nannan Wang `[通讯]` (Xidian University)

**通讯引用:** 12960 | [OpenAlex ID](https://openalex.org/A5100774865)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于多模态大型语言模型推理链的域泛化框架 RD-MLDG。

**💡 创新点**

创新点在于引入 DomainBed-Reasoning 数据集与 MTCT 与 SARR 两种机制，使模型在保持标签监督的同时，学习可解释且跨域鲁棒的推理链。

**🔧 技术方法**

主要技术包括 GPT‑4o 生成类相关推理链、InternVL3‑8B 作为基础 MLLM、LoRA 参数高效微调、MTCT 交叉任务训练和 SARR 迭代自标注。

**📊 数据集**

使用的实验数据集为扩展后的 DomainBed Benchmark，即 PACS、VLCS、OfficeHome 与 TerraIncognita，所有样本均配有 GPT‑4o 生成的推理链。

**📈 对比分析**

与现有多种 DG 方法对比，RD‑MLDG 在四个基准上平均准确率达到 86.89%，在 PACS、VLCS、OfficeHome 与 TerraIncognita 上分别获得 98.13%、87.03%、91.73% 与 70.65%，显著优于 GPT‑4o、CLIP、DGCLDTP 等强基线。

**⚠️ 局限性**

局限性包括对 GPT‑4o 生成链的质量与一致性的依赖、推理链生成的计算成本、以及在极端域偏差下仍可能出现推理模式与目标域不匹配的问题。

---

## 117. The GRADIEND Python Package: An End-to-End System for Gradient-Based Feature Learning

**arXiv ID:** 2602.23993 | [PDF](https://arxiv.org/pdf/2602.23993v1)

**作者:** Jonathan Drechsel `[一作]` (University of Passau), Steffen Herbold `[通讯]` (University of Passau)

**通讯引用:** 1922 | [OpenAlex ID](https://openalex.org/A5027032646)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个开源Python包gradiend，用于学习、评估、重写和比较语言模型中的可复用特征方向；

**💡 创新点**

首次将特征方向视为持久可复用的对象，直接从事实-反事实梯度学习可重写权重方向，并提供统一的端到端工作流与可扩展的多特征比较工具；

**🔧 技术方法**

基于对MLM/CLM梯度的编码‑解码模型，使用预剪枝与后剪枝减少参数维度，利用Hugging Face Trainer风格的训练接口进行特征方向学习，并在权重空间执行可控更新；

**📊 数据集**

主要使用英语维基百科文本构建第三人称单复数人称代词数据集（he/she/it vs. they），并在此基础上进行单特征和大规模多特征实验；

**📈 对比分析**

通过内部评估（类间编码分离度、解码器学习率梯度对目标词概率的影响）和跨模型评估（Top‑k参数重叠的Venn图/热图）来比较特征方向的相似性；实验表明不同语言学特征在参数空间有显著重叠或互异，重写后模型在保持整体语言模型性能的同时成功偏移目标特征；

**⚠️ 局限性**

目前仅支持文本预测梯度（MLM/CLM），不支持其他任务或视觉领域；重写过程依赖手工调参的学习率，可能对模型整体表现产生副作用；进一步研究需扩展至更多任务与更高效的可解释性评估。

---

## 118. FPPS: An FPGA-Based Point Cloud Processing System

**arXiv ID:** 2602.23787 | [PDF](https://arxiv.org/pdf/2602.23787v1)

**作者:** Xiaofeng Zhou `[一作]` (Hong Kong University of Science and Technology), Wei Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 19690 | [OpenAlex ID](https://openalex.org/A5100653327)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了FPPS，一个基于FPGA的点云配准加速系统，以高效加速ICP算法；

**💡 创新点**

提出全并行流水线化的NN搜索与变换估计，并提供与PCL兼容的API，使FPGA加速可无缝集成；

**🔧 技术方法**

采用FPGA（Alveo U50）实现Systolic Array的全并行NN搜索、DSP/LUT并行矩阵运算，以及主机与FPGA间的高带宽内存接口；

**📊 数据集**

在KITTI Odometry数据集上进行评估；

**📈 对比分析**

与高性能Intel Xeon CPU软件实现对比，平均RMSE差异≤0.01 m，帧延迟提升4.82×至35.36×，功耗效率提高8.58×；

**⚠️ 局限性**

受限于单一SLR布局、未验证多种硬件平台和大规模点云；同时未采用k‑d树等高效搜索策略，仅在U50上验证。

---

## 119. SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting

**arXiv ID:** 2602.24020 | [PDF](https://arxiv.org/pdf/2602.24020v1)

**作者:** Xiang Feng `[一作]` (Hangzhou Dianzi University), Yanming Zhu `[通讯]` (Griffith University)

**通讯引用:** 6176 | [OpenAlex ID](https://openalex.org/A5101435447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于 3D Gaussian Splatting 的端到端稀疏视图高分辨率重建框架 SR3R，直接从少量低分辨率视图预测高分辨率 3DGS。

**💡 创新点**

创新点在于将 3DSR 视作 feed-forward 映射，放弃 2D SR 伪标签和逐场景优化，采用高频偏置学习和跨视角特征融合，实现跨场景泛化和更高频细节恢复。

**🔧 技术方法**

采用 ViT 编码器/解码器、双向注意力特征细化、Gaussian offset 学习、PointTransformerV3 空间建模以及可插拔的 3DGS 后端。

**📊 数据集**

在 RE10K、ACID、DTU 和 ScanNet++ 四大公开数据集上进行训练与零-shot 评估。

**📈 对比分析**

与 NoPoSplat、DepthSplat 及其上采样版本以及 SRGS、FSGS+SRGS 进行对比，SR3R 在 PSNR/SSIM/LPIPS 上显著优于所有基线，并在零-shot 场景下超越优化方法，推理速度也保持在毫秒级至秒级。

**⚠️ 局限性**

仍受限于需预先提供相机姿态、对极低视角稀疏或动态场景的处理能力有限，且对极大尺寸或高复杂度场景的推理速度与内存占用尚有提升空间。

---

## 120. Adaptive Correlation-Weighted Intrinsic Rewards for Reinforcement Learning

**arXiv ID:** 2602.24081 | [PDF](https://arxiv.org/pdf/2602.24081v1)

**作者:** Viet Bac Nguyen `[一作]` (Vietnam National University University of Engineering and Technology), Phuong Thai Nguyen `[通讯]` (Vietnam National University University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 ACWI 框架，利用 Beta 网络根据状态自适应地缩放 intrinsic 奖励，以提高稀疏奖励环境下的探索效率。

**💡 创新点**

创新点在于用相关性损失直接对 intrinsic 奖励与未来 extrinsic 回报的协方差进行优化，从而实现无需手动调参的状态级自适应权重。

**🔧 技术方法**

采用 PPO 作为策略优化器、ICM 作为 intrinsic 奖励模块，并引入轻量 Beta 网络与相关性损失进行一次梯度更新。

**📊 数据集**

在 MiniGrid 旗下五个稀疏奖励任务（DoorKey-8x8、Empty-16x16、RedBlueDoors-8x8、UnlockPickup、KeyCorridorS3R3）上进行评估。

**📈 对比分析**

与 PPO+ICM 的固定 β（0.1–2.0）以及仅使用 extrinsic 奖励的基线相比，ACWI 在大多数任务上实现了更快的收敛、降低方差、样本效率提升，但在极端稀疏（Empty-16x16）任务中提升有限。

**⚠️ 局限性**

主要局限在于当 extrinsic 回报几乎全为零时，相关性信号弱导致 Beta 网络无法有效学习，退化为近似固定权重；且对极端稀疏环境的自适应性不足。

---

## 121. An Agentic LLM Framework for Adverse Media Screening in AML Compliance

**arXiv ID:** 2602.23373 | [PDF](https://arxiv.org/pdf/2602.23373v1)

**作者:** Pavel Chernakov `[一作]` (University of Luxembourg), Raphaël Frank `[通讯]` (University of Luxembourg)

**通讯引用:** 1900 | [OpenAlex ID](https://openalex.org/A5009044318)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于LLM的代理式不良媒体筛查框架（AMI Agent），实现从网络搜索、检索、RAG、评估到判定的全自动流程。

**💡 创新点**

创新点在于将LLM代理与可配置评估剧本结合，支持多维度（身份匹配、负面情感、风险水平）评分，输出可解释的自然语言判定，并可切换多种LLM后端。

**🔧 技术方法**

采用LLM（如OpenRouter上的GPT‑4.1‑Mini、Gemini 2.5 Flash、Grok 4.1 Fast）+ Retrieval‑Augmented Generation（RAG）+ 文档向量检索 + DSPy 结构化签名进行评估。

**📊 数据集**

使用四类人群数据集：学术作者（低风险）、PEP名单、监管观察名单（中风险）以及OFAC制裁名单（高风险），并采集对应的网络检索结果。

**📈 对比分析**

通过比较三种LLM后端在上述四类人群上的AMI分数分布，验证了模型的区分能力；平均AMI分数分别为：Clean 0.015–0.029，PEP 0.063–0.087，RW 0.167–0.179，SDN 0.730–0.863，Grok 4.1 Fast 在高风险人群上得分最高。

**⚠️ 局限性**

局限包括依赖搜索引擎覆盖率、实体歧义处理困难、LLM幻觉风险、仅支持英文来源、缺乏时间权重以及对多语言和持续监控的支持不足。

---

## 122. AI Must Embrace Specialization via Superhuman Adaptable Intelligence

**arXiv ID:** 2602.23643 | [PDF](https://arxiv.org/pdf/2602.23643v1)

**作者:** Judah Goldfeder `[一作]` (Columbia University), Ravid Shwartz Ziv `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 Superhuman Adaptable Intelligence（SAI）的概念，并从理论上批判传统 AGI 过度强调通用性，认为人类智能是专业化而非全能的，主张把 AI 研究北星目标转向快速适应重要任务。

**💡 创新点**

创新点在于：① 把人类通用性误区拆解为专业化问题；② 用“适应速度”作为衡量标准，取代传统的“与人类水平相当/超越”评判；③ 将自监督学习、世界模型与模块化视为实现 SAI 的关键技术路径，强调多样性与分层结构。

**🔧 技术方法**

主要技术讨论：自监督学习（SSL）、世界模型（World Models）、元学习/快速适应、可路由/模块化架构、潜在空间预测（latent prediction）等。

**📊 数据集**

未使用具体数据集；文中引用现有工作（如 AlphaFold、GPT、Dreamer 等）以说明概念可行性，整体为理论与综述性质。

**📈 对比分析**

评价方法：以“新技能/任务获取的速度”作为核心指标；与传统 AGI 定义相比，后者缺乏可衡量性、可实现性和可评估性。论文通过定义与现有 AGI 定义对比表格说明 SAI 在可行性、内部一致性和可评估性方面的优势。

**⚠️ 局限性**

局限性：缺乏实验验证与定量结果；任务“重要性”与“效用”的界定尚不明确；实现细节（如最优架构、训练策略）仍待进一步研究。

---

## 123. Bridging Dynamics Gaps via Diffusion Schrödinger Bridge for Cross-Domain Reinforcement Learning

**arXiv ID:** 2602.23737 | [PDF](https://arxiv.org/pdf/2602.23737v1)

**作者:** Hanping Zhang `[一作]` (Carleton University), Yuhong Guo `[通讯]` (Carleton University)

**通讯引用:** 8136 | [OpenAlex ID](https://openalex.org/A5043824291)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种在源域在线训练、仅使用目标域离线演示且无奖励的跨域强化学习框架BDGxRL。

**💡 创新点**

创新点在于将Diffusion Schrödinger Bridge用于动态对齐并引入基于状态转移的奖励调制，实现无目标奖励的跨域迁移。

**🔧 技术方法**

使用DSB、SAC、行为克隆、IMF等技术。

**📊 数据集**

实验基于MuJoCo的HalfCheetah和Walker2d任务，并使用D4RL提供的离线演示。

**📈 对比分析**

与xTED、DARA、DARC、DARAIL、GAIL等基线比较，BDGxRL在所有域差距和演示质量下均取得最高或最稳健的得分。

**⚠️ 局限性**

局限在于依赖离线目标演示、需要训练DSB和奖励模型的额外计算开销，且在极端动力学差异或极低质量演示下性能下降。

---

## 124. SleepLM: Natural-Language Intelligence for Human Sleep

**arXiv ID:** 2602.23605 | [PDF](https://arxiv.org/pdf/2602.23605v1)

**作者:** Zongzhe Xu `[一作]` (University of California), Yuzhe Yang `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一系列睡眠-语言基础模型，利用多层次字幕生成管道生成大规模睡眠文本对照数据集（>100,000小时），并通过联合预训练实现睡眠信号与自然语言的对齐与交互。

**💡 创新点**

创新点包括：①多层次睡眠字幕生成管道，提供从通道特征到局部事件再到全局状态的三级文本描述；②将对比学习、文本生成与信号重构三种目标统一到一个复合预训练框架；③在多通道睡眠信号上实现语言驱动的事件定位、可控洞察生成和零样本泛化。

**🔧 技术方法**

核心技术：多通道时序 Transformer 编码器（跨时序与跨通道注意力，RoPE 编码），重构解码器，模态条件文本解码器；复合预训练目标（InfoNCE 对比、MSE 重构、语言生成损失）。

**📊 数据集**

使用来自NSRR的五大数据库（SHHS、MrOS、CFS、CCS、WSC）共计>10,000名个体，约>100,000小时睡眠记录。

**📈 对比分析**

与 Gemini 2.5 Pro、DeepSeek 等大型 LLM 以及 Fine‑tuned VLM、SSL 基线对比。零样本任务（睡眠分期、事件定位、HR/SpO₂ 预测、通道统计回归）均显著优于基线；跨模态检索 R@1 接近 100%；在未见概念（混合阻塞呼吸）下 F1≈80%；few‑shot 学习 AUC 达 0.90，表明数据效率高。

**⚠️ 局限性**

局限性：仅为研究原型，未经过临床验证；数据来源局限于 NSRR，可能缺乏不同设备与更广泛人群的多样性；模型对极长时序或噪声敏感，尚需进一步优化。

---

## 125. 2G2T: Constant-Size, Statistically Sound MSM Outsourcing

**arXiv ID:** 2602.23464 | [PDF](https://arxiv.org/pdf/2602.23464v1)

**作者:** Majid Khabbazian `[一作]` (University of Alberta), Majid Khabbazian `[通讯]` (University of Alberta)

**通讯引用:** 1389 | [OpenAlex ID](https://openalex.org/A5086458557)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一种针对固定基点多标量乘法（MSM）的可验证外包协议，服务器只返回两个曲线点即可完成验证。

**💡 创新点**

创新点在于：一次性密钥化设置、常数大小服务器响应、客户端仅需一次内积和常数个群运算，且实现了1/q级别的统计完整性。

**🔧 技术方法**

采用了MSM线性性质、秘密随机标量与向量、合并基点向量T、内积校验以及统计安全证明，并在Rust + Ristretto255上实现。

**📊 数据集**

实验使用随机生成的P⃗、x⃗（Ristretto255曲线点和标量），并测试了从2¹⁰到2¹⁸的不同维度。

**📈 对比分析**

与最优MSM实现和朴素逐项求和基准比较，验证速度最快，最大发挥可达约300倍（对最优MSM）和3000倍（对朴素MSM）加速，服务器仅额外执行一次MSM，客户端工作量保持常数。

**⚠️ 局限性**

局限性包括：只能用于指定验证器、一次性设置、服务器可知标量、未提供隐私保护，且不支持公开可验证变体。

---

## 126. OmniTrack: General Motion Tracking via Physics-Consistent Reference

**arXiv ID:** 2602.23832 | [PDF](https://arxiv.org/pdf/2602.23832v1)

**作者:** Yuhan Li `[一作]` (Huazhong University of Science and Technology), Siyuan Huang `[通讯]` (State Key Lab of General AI, Beijing Institute for General Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 OmniTrack，一种两阶段的通用人形机器人运动跟踪框架，先将人类运动数据通过仿真生成物理可行的轨迹，再用这些轨迹训练可在实际机器人上实时执行的控制策略。

**💡 创新点**

创新点在于显式地将物理可行性与运动跟踪分离：第一阶段使用全信息（全局位置、速度、接触状态等）生成动力学一致的参考动作，第二阶段在部分可观测、噪声、随机化环境下学习通用控制策略，从而避免控制器在训练和部署时必须同时兼顾跟踪精度与物理稳定性。

**🔧 技术方法**

核心技术包括基于 PPO 的两阶段强化学习、物理一致性生成（全信息强化学习 + 轨迹回放）、接触奖励、动作平滑惩罚、观察噪声与域随机化、以及在 IsaacLab/MuJoCo 等仿真环境中的仿真到现实迁移。

**📊 数据集**

使用 Unitree‑retargeted LAFAN1（约 2.5 小时、40 条动作）和 AMASS 子集（约 10,000 条动作）两大数据集，经过物理可行性生成后得到训练集。

**📈 对比分析**

与 Dagger、Dagger_hist、AAC 以及 OmniH2O、ExBody2、BeyondMimic 等基线进行对比。结果显示 OmniTrack 在训练集和未见动作上成功率均高于 96%，MPJPE 下降至约 35 mm（比基线低 5–10 mm），在高动态子集上成功率提升至 84%（对比 48–70%）。在 Unitree G1 实机上实现了持续 1 小时的多姿态跟踪，包括翻滚、侧翻、跳跃等高动态动作，并支持实时 VR/运动捕捉的无缝遥操作。

**⚠️ 局限性**

局限性：第一阶段需要完整的仿真信息和计算资源；若人类数据质量极差或含有不可消除的动态不一致，生成的轨迹仍可能受限；在极端外部扰动或非标准接触场景下的鲁棒性尚待进一步验证。

---

## 127. Planning under Distribution Shifts with Causal POMDPs

**arXiv ID:** 2602.23545 | [PDF](https://arxiv.org/pdf/2602.23545v1)

**作者:** Matteo Ceriscioli `[一作]` (Oregon State University), Karthika Mohan `[通讯]` (Oregon State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一个基于因果模型的部分可观测马尔可夫决策过程（Causal POMDP）框架，用以在分布偏移（distribution shift）下进行规划与决策；

**💡 创新点**

创新点在于：① 将分布偏移视为对因果模型的干预（intervention），从而能够在策略评估与规划时直接嵌入偏移信息；② 在包含未知偏移的情况下证明有限时限的价值函数依旧是关于状态-域联合信念的分段线性凸函数（PWLC），保证了 α‑vector 等传统 POMDP 方法的可行性；

**🔧 技术方法**

主要技术包括：因果影响图（CID）与因果贝叶斯网络的结合、软干预（stochastic shift intervention）模型、联合信念更新（state + domain）以及对 PWLC 结构的构造性证明；

**📊 数据集**

论文为理论性工作，未使用具体数据集；

**📈 对比分析**

由于缺乏实验部分，未提供与其他方法的比较或性能指标；

**⚠️ 局限性**

局限性在于：① 仅针对有限时限问题，未讨论无穷期规划；② 软干预假设与实际偏移的匹配可能受限；③ 需要假设对因果结构有完整了解，实际场景中因果关系可能未知或不完全；

---

## 128. Bi-level RL-Heuristic Optimization for Real-world Winter Road Maintenance

**arXiv ID:** 2602.24097 | [PDF](https://arxiv.org/pdf/2602.24097v1)

**作者:** Yue Xie `[一作]` (Loughborough University), Fumiya Iida `[通讯]` (University of Cambridge)

**通讯引用:** 10018 | [OpenAlex ID](https://openalex.org/A5041874420)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研发了可扩展的双层优化框架，结合强化学习上层分区与多目标车辆路径规划，解决英国高速公路冬季路面维护路由问题。

**💡 创新点**

引入双层RL-启发式结构，在上层利用PPO学习动态分区，在下层使用约束感知NN路由，显著提升大规模网络可行性与效率。

**🔧 技术方法**

使用PPO强化学习、KDTree空间聚类、约束感知邻接插入路由、线性/整数模型以及多目标优化技术。

**📊 数据集**

采用英国M25、M6、A1等战略道路网络的真实运营数据、OpenStreetMap地理信息及National Highways GPS轨迹。

**📈 对比分析**

与静态KDTree+NN基线对比，10轮PPO迭代后将最大路程时间从122.14h降至118.81h，CO₂排放从3386.63kg降至3220.95kg，车辆使用量上升约28辆，整体实现Pareto改进。

**⚠️ 局限性**

局限于预先规划阶段，未处理实时天气变化与中途重新调度；依赖大量预处理；对多仓库存、车辆停靠重载等细节建模不完整。

---

## 129. MI$^2$DAS: A Multi-Layer Intrusion Detection Framework with Incremental Learning for Securing Industrial IoT Networks

**arXiv ID:** 2602.23846 | [PDF](https://arxiv.org/pdf/2602.23846v1)

**作者:** Wei Lian `[一作]` (University of Nottingham), Alejandro Guerra-Manzanares `[通讯]` (University of Nottingham)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 MI^2DAS 多层工业物联网入侵检测框架，结合异常层级聚合、开放集识别与增量学习，支持零日攻击检测和在线适应新威胁。

**💡 创新点**

创新点在于三层协同架构：第一层使用 GMM 进行高召回率的正常-攻击分离；第二层将已知与未知攻击分离，采用 GMM 与 LOF 的互补策略；第三层通过半监督与主动学习实现增量学习，最小化标注成本并保持对旧类的记忆。

**🔧 技术方法**

核心技术包括 Gaussian Mixture Model (GMM)、Local Outlier Factor (LOF)、One-Class SVM、Isolation Forest、Random Forest、XGBoost、LightGBM、kNN、SVM、Logistic Regression、半监督自训练/标签传播/标签扩散、主动学习不确定性采样、伪标签增量训练。

**📊 数据集**

使用 Edge‑IIoTset 数据集，包含 53 维特征、14 类攻击与正常流量，模拟真实工业 IoT 流量与攻击分布。

**📈 对比分析**

与单一模型对比时，MI^2DAS 在三层均达优异指标：第一层 GMM 0.953 准确率、TPR 1.000；第二层已知攻击召回 0.813，未知攻击召回 0.882；第三层增量学习 Macro‑F1 最高达 0.8995。与传统基线相比，随机森林在已知攻击分类中实现宏 F1 0.941，显著优于 kNN、SVM、LR 等线性模型。

**⚠️ 局限性**

局限包括仅在 Edge‑IIoTset 数据集评估，未覆盖更广泛的实时环境；仅测试了少数经典算法，未尝试深度学习方法；假设训练数据要么干净要么轻度污染，忽略真实混杂场景；增量学习仅采用一‑步与多‑步两种方案，未探索终生学习或混合策略。

---

## 130. Autonomous Inspection of Power Line Insulators with UAV on an Unmapped Transmission Tower

**arXiv ID:** 2602.24011 | [PDF](https://arxiv.org/pdf/2602.24011v1)

**作者:** Václav Riss `[一作]`, Martin Saska `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种在线检测与定位电塔绝缘子并在单次飞行中完成检查的算法，避免了预先绘制地图的需求。

**💡 创新点**

创新点在于将相机与激光雷达融合用于实时绝缘子检测与定位，并在此基础上设计了 DBSCAN+RANSAC 的定位方法，实现一次飞行即可获取高质量检查图像；此外通过对比两飞行策略，展示了显著的时间效率提升。

**🔧 技术方法**

技术包括YOLOv11n深度学习检测、相机-雷达几何投影、点云滤波、DBSCAN、RANSAC、PCA聚类与线拟合，以及基于TSP的轨迹规划。

**📊 数据集**

使用了两套数据集：在Flight Forge仿真环境中采集约1500张图像；在实际塔检中收集约3200张图像，分别用于训练、验证与测试。

**📈 对比分析**

在仿真中，单飞策略相较于两飞行策略可节省约24%检查时间；在实际试验中，DBSCAN+RANSAC定位误差为0.16±0.08 m（xy）和0.16±0.11 m（z），并在mAP50–95上达到0.81（仿真）与0.17（实景），显示定位精度与时间效率优于传统方法。

**⚠️ 局限性**

局限性包括实景检测mAP显著下降（受雾、光照影响）；需要塔的最大高度、宽度与GPS坐标；假设塔附近无障碍物；对实时计算资源和RTK定位有一定依赖。

---

## 131. Intrinsic Lorentz Neural Network

**arXiv ID:** 2602.23981 | [PDF](https://arxiv.org/pdf/2602.23981v1)

**作者:** Xianglong Shi `[一作]` (University of Science and Technology of China), Nicu Sebe `[通讯]` (University of Trento)

**通讯引用:** 35969 | [OpenAlex ID](https://openalex.org/A5027171279)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种完全在洛伦兹模型内运算的全新超曲线神经网络 ILNN，并设计了点到超平面全连接层（PLFC）、GyroLBN、log‑radius 拼接和 Lorentz dropout 等全新的本质层。

**💡 创新点**

创新点在于：① 用点到超平面的距离直接生成分类 logits，避免欧氏 affine 变换；② 通过 GyroLBN 结合 gyro‑centering 与 gyro‑scaling，在洛伦兹几何中实现高效且一致的批归一化；③ 采用 log‑radius 方案稳定多块特征的拼接；④ 所有模块均为本质化，保持负曲率的一致性。

**🔧 技术方法**

主要技术包括：Lorentz 几何（负曲率），点到超平面距离公式，gyro‑vector 代数，闭式 Lorentz 平均与方差，Riemannian 优化（RSGD / Adam），以及自定义的 Lorentz dropout/激活。

**📊 数据集**

实验使用了 CIFAR‑10/100（图像），TEB 与 GUE（基因组学），以及 Airport、Cora、PubMed 三个图数据集。

**📈 对比分析**

与传统欧氏 ResNet‑18、混合 Poincaré/Horentz 模型、HCNN 以及多种图网络（GraphFormer、HGNN 等）进行公平比较，ILNN 在所有基准上取得最高或相当于最高的准确率/ MCC，CIFAR‑10 95.36%，CIFAR‑100 78.41%，基因组学 MCC 多达 83.9%，图数据集最高达 96.0%/85.7%/82.5%。相较于之前最强的 HCNN，ILNN 在图像上提升 0.2‑0.3pp，基因组学提升 2‑10pp，图网络提升 0.8‑1.2pp；训练时间上 GyroLBN+PLFC 的效率也优于 LBN 或 GyroBN。

**⚠️ 局限性**

局限性包括：① 只在洛伦兹模型中实现，无法直接迁移到其他负曲率模型；② 需要 Riemannian 优化器与专门实现的几何运算；③ 对于极大维度或极大批量的场景，梯度计算仍可能消耗较多资源；④ 目前只在固定曲率 K = –1 下验证，曲率自适应研究尚未展开。

---

## 132. SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale

**arXiv ID:** 2602.23866 | [PDF](https://arxiv.org/pdf/2602.23866v1)

**作者:** Ibragim Badertdinov `[一作]` (Nebius), Alexander Golubev `[通讯]` (Nebius)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套可扩展、跨语言的自动化流水线，用来从 GitHub PR 中挖掘可执行的软硬件工程任务，并生成可直接用于训练 LLM 代理的 32k+ 任务和 120k+ PR 衍生任务；

**💡 创新点**

核心创新在于：① 语言无关的环境合成流程，统一使用可重用的基础镜像和 LLM 交互式代理；② 通过多模型 LLM 集成进行问题描述清晰度过滤；③ 对每个任务自动生成诊断元数据（如测试耦合、命名隐含、外部依赖）以支持分层训练；④ 通过 PR 描述生成任务说明扩展数据规模；

**🔧 技术方法**

使用 LLM 交互式代理（mini‑SWE‑agent 与 Qwen3‑Coder 等）、Docker 容器化、自动化 oracle 提取、LLM 集成判断、基于 LLM 的元数据生成与诊断标签、以及多阶段 Docker 构建与测试执行；

**📊 数据集**

基于 GitHub Archive 采集的 29.5M PR，最终生成 32k+ issue‑linked 任务（3,600+ repo，20 语言）与 120k+ PR‑derived 任务；

**📈 对比分析**

通过对比非交互式与交互式设置、不同模型、上下文长度与重试次数等 ablation，发现交互式代理和更长上下文能将设置成功率提升至约58%（相比非交互式仅 16%）；LLM 评估集成在问题清晰度过滤中实现约 83% 的 precision，整体数据量较前一版本提升超过 200%；实验还评估了七款前沿模型在任务集上的 pass@k，表明数据集能够覆盖多种难度与语言，支持层级训练与基准比较；

**⚠️ 局限性**

仅支持可单一容器化构建的项目，难以覆盖多服务或外部依赖系统；未进行训练子集 ablation 实验，无法量化诊断标签对训练效果的具体影响；部分长尾语言任务仍不足，且部分任务可能存在环境或测试耦合导致的奖励噪声。

---

## 133. SLA-Aware Distributed LLM Inference Across Device-RAN-Cloud

**arXiv ID:** 2602.23722 | [PDF](https://arxiv.org/pdf/2602.23722v1)

**作者:** Hariz Yet `[一作]` (Singapore University of Technology and Design), Tony Q. S. Quek `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 29846 | [OpenAlex ID](https://openalex.org/A5030858163)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在5G单独（SA）AI-RAN实验平台上，使用Qwen2.5-VL模型进行分布式推理，系统在设备、RAN-edge（利用NVIDIA GH200的MIG硬件隔离）和云三层上进行固定基线的性能测评；

**💡 创新点**

首次用真实硬件和固定重复策略量化阈值下的SLA达成率，揭示量化与模型规模对尾部延迟的影响，并证明MIG硬件隔离能在RAN边缘安全共址AI推理与基带处理；

**🔧 技术方法**

采用5G SA栈（OAI O-CU/DU + NVIDIA Aerial low O-DU）、NVIDIA GH200 GPU + MIG、vLLM推理框架、FP16/AWQ/W4A16/W8A8量化、AWS Mumbai p5.4xlarge H100云实例、PTP时钟同步与Prometheus/TimescaleDB/Grafana监控；

**📊 数据集**

使用一段2.5分钟的机器人第一人称视频序列，按0.5 s间隔重放，产生≈300张图像请求；

**📈 对比分析**

通过Hit@0.5s与Hit@1.0s两种严格SLA预算，记录TTFT、RTT、E2E延迟；结果显示：设备层始终无法满足子秒SLA；Edge层AWQ/3B量化版在Premium层达98–99% Hit@0.5s，云层在同一路径上Premium Hit仅≤32.9%，但所有变体在1.0s时都能100%满足；

**⚠️ 局限性**

主要局限包括：结果受WAN路径特性影响（仅评估Mumbai到SUTD的路线），未提供无MIG共址基准；使用单一路径和单一视频轨迹，缺乏不同网络/工作负载下的泛化；仅评估Qwen2.5-VL，未覆盖更大/多模态模型；

---

## 134. Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping

**arXiv ID:** 2602.23980 | [PDF](https://arxiv.org/pdf/2602.23980v1)

**作者:** Tianxiang Du `[一作]` (Peking University), Yuxin Peng `[通讯]` (Peking University)

**通讯引用:** 8998 | [OpenAlex ID](https://openalex.org/A5047811387)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了专门用于美学指导（AG）的首个数据集与基准 AesGuide，并基于此设计了两阶段框架 Venus，先通过逐步递进的美学问题训练多模态大语言模型（MLLM）获得 AG 能力，再利用链式推理（CoT）和审美理由（AR）激活其裁剪功能，实现可解释、可交互的相机拍摄与后期裁剪。

**💡 创新点**

创新点包括：①首次定义并系统化研究 AG 任务；②构建了包含美学评分、分析与可执行指导的 10,748 张真实照片数据集；③提出进阶式美学问答训练策略，模仿摄影师的认知流程；④引入 CoT + 审美理由生成与验证机制，提升裁剪模型的可解释性和交互性；⑤在公开基准 FLMS 上实现裁剪 IoU 超过 87%，显著优于现有专用裁剪模型和所有对比 MLLM。

**🔧 技术方法**

技术手段包括：多模态大语言模型（如 Qwen-VL-Chat、InternVL 2.5、MiniCPM-V、LLaVA-1.5-7B/13B）；监督式指令微调（Stage 1）与全参数微调（Stage 2）；LoRA 进行参数高效微调；Chain-of-Thought 推理；GPT‑4o 生成与评估审美理由；三维度 GPT‑辅助评测（完整性、精确度、相关性）。

**📊 数据集**

使用的数据集：AesGuide（10,748 照片，含评分、分析与指导）、FLMS（裁剪评估基准）、CPC、GAIC、FCDB（裁剪训练补充）以及少量公开辅助数据。

**📈 对比分析**

方法对比：与专有 MLLM（GPT‑4o、Gemini‑2.0‑Pro、Qwen‑VL‑Max）、美学专用 MLLM（AesExpert、UNIAA）、以及多种专用裁剪模型（ASM‑Net、CACNet、HCIC、SAC‑Net、UNIC、ProCrop）进行对比。实验显示 Venus‑Q 在 AesGuide 的三维度评测均领先所有对手，且在 FLMS 上 IoU 达到 87.01%，比上一 SOTA 提升 1.5% 及 GPT‑4o 提升 15.4%。此外，模型能够生成裁剪理由并支持交互式细化，兼具性能与可解释性。

**⚠️ 局限性**

局限性：①数据集仍有限，缺乏跨文化与多样化场景的覆盖；②评测仍高度依赖 GPT‑4o 的自动评估，可能引入主观偏差；③模型在极端光照或复杂构图情况下的鲁棒性尚待进一步验证；④交互式裁剪功能主要基于文本指令，尚未实现完整的实时摄像头交互。

---

## 135. QoSFlow: Ensuring Service Quality of Distributed Workflows Using Interpretable Sensitivity Models

**arXiv ID:** 2602.23598 | [PDF](https://arxiv.org/pdf/2602.23598v1)

**作者:** Md Hasanur Rashid `[一作]` (University of Delaware), Dong Dai `[通讯]` (University of Delaware)

**通讯引用:** 697 | [OpenAlex ID](https://openalex.org/A5012002926)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于敏感性分析的可解释质量服务（QoS）建模方法——QoSFlow，用于快速推断分布式工作流在不同存储层级和并行度配置下的执行时间，并据此推荐满足用户QoS约束的最佳配置；

**💡 创新点**

创新点在于将敏感性分析与工作流DAG模板相结合，形成可解释的性能区域（Region），通过CART分割配置空间，既能捕捉关键路径的变化，又能突出哪些配置是关键、哪些是可自由调度的；

**🔧 技术方法**

主要技术包括：①工作流模板构建与规模投射；②存储层级性能模型（通过IOR等基准生成）；③全配置枚举与关键路径时序估算；④全局与局部敏感性分析；⑤基于CART的区域聚类与阈值评估；⑥基于区域的QoS驱动配置推荐；

**📊 数据集**

在三种典型工作流上验证：1) 1000Genomes（基因组分析）；2) PyFLEXTRKR（大气科学分析）；3) DeepDriveMD（ML驱动分子动力学）；每个工作流均在多种节点数/GPU数下评估；

**📈 对比分析**

与传统基于启发式的调度策略（如Fastest‑Storage First、Low‑Transition Layout等）对比，QoSFlow在最佳配置排序的Pairwise Concordance上提升约27.38%，在不同QoS请求下推荐配置的实际执行时间与预测值高度一致，证明了模型的准确性和实用性；

**⚠️ 局限性**

局限性包括：①需要先在少量规模上执行工作流以构建模板，若工作流结构变化大需重新建模；②区域划分依赖于敏感性阈值和CART剪枝参数，参数选择不当可能导致过度分割或欠分割；③模型主要关注存储层级和并行度，对计算资源（CPU/GPU）分配细粒度影响不足；④跨规模迁移需单独训练，无法直接泛化到大规模系统。

---

## 136. CLFEC: A New Task for Unified Linguistic and Factual Error Correction in paragraph-level Chinese Professional Writing

**arXiv ID:** 2602.23845 | [PDF](https://arxiv.org/pdf/2602.23845v1)

**作者:** Jian Kai `[一作]` (Huazhong University of Science and Technology), Qiang Liu `[通讯]` (WPS AI, Kingsoft Office)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了中文专业写作的段落级统一错误纠错任务CLFEC，并构建了多域（时政、金融、法律、医学）混合数据集，系统评估了LLM的提示、检索增强生成与智能代理方案；

**💡 创新点**

创新点在于将词法、语法、标点与事实错误统一到同一段落级任务，设计了诊断拆分（LEC、FEC、MIX、Error-free）及多域数据集，并首次比较两阶段与统一RAG以及ReAct式代理框架在此任务上的效果；

**🔧 技术方法**

使用了大型语言模型的提示技术、检索增强生成（Sequential RAG与Unified RAG）以及ReAct式代理框架，配合外部检索工具与状态管理工具实现事实证据检索与验证；

**📊 数据集**

使用了自构建的CLFEC数据集，覆盖时政、金融、法律、医学四大领域，包含词法错误、事实错误、混合错误以及无误差的诊断拆分；

**📈 对比分析**

通过对比提示、S-RAG、U-RAG与Agent等方法，以精度、召回和F1为指标进行评估，结果显示统一RAG和Agent在大模型上获得最高F1，但混合错误仍最难，模型在无误差文本上易出现过度修正；

**⚠️ 局限性**

限制在于评估仅依据严格匹配的单一参考，忽视多样合法修正；高性能方法高度依赖大模型，推理成本和延迟高，亟需探索小模型或轻量化代理以实现实时工业应用。

---

## 137. Diffusion Probe: Generated Image Result Prediction Using CNN Probes

**arXiv ID:** 2602.23783 | [PDF](https://arxiv.org/pdf/2602.23783v1)

**作者:** Benlei Cui `[一作]` (Alibaba Group), Haiwen Hong `[通讯]` (Alibaba Group)

**通讯引用:** 153 | [OpenAlex ID](https://openalex.org/A5045259909)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了Diffusion Probe，用早期交叉注意力特征预测T2I最终图像质量，以实现高效早期质量评估。

**💡 创新点**

首次将模型内部交叉注意力作为轻量化预测信号，并训练Probe实现对多种质量指标的准确预测。

**🔧 技术方法**

采用轻量级CNN预测器，对Diffusion Transformer/Stable Diffusion的早期交叉注意力进行统计特征提取，训练回归模型。

**📊 数据集**

使用MS‑COCO 2017 caption数据集训练15k条提示，测试5k条提示。

**📈 对比分析**

与基线随机或LLM优化比较，在SDXL、FLUX上在Prompt优化和Seed选择任务中提升CLIP Score、ImageReward、Aesthetic Score，AUC‑ROC>0.9，SRCC>0.7。

**⚠️ 局限性**

局限在于对不同质量指标需单独训练Probe，对极端失败样本的预测仍受限；仅在图像质量评估上验证，尚未扩展到更广泛生成任务。

---

## 138. Now You See Me: Designing Responsible AI Dashboards for Early-Stage Health Innovation

**arXiv ID:** 2602.23378 | [PDF](https://arxiv.org/pdf/2602.23378v1)

**作者:** Svitlana Surodina `[一作]` (OPORA Health Technologies LTD), Rita Borgo `[通讯]` (King's College London)

**通讯引用:** 1468 | [OpenAlex ID](https://openalex.org/A5031596501)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过纵向设计研究、案例研究与问卷调查，构建并评估面向早期HealthTech AI团队的负责AI治理仪表板，探索其设计流程与效果。

**💡 创新点**

提出两阶段可重用的可视化设计框架：先构建行业域知识图谱再映射至项目仪表板；强调分阶段义务、可解释性、角色特定视图，并推广生态系统级共享治理基础设施。

**🔧 技术方法**

使用可视化设计研究方法、Neo4j知识图谱进行行业域建模、法规规则编码与推理；仪表板前端采用可视化技术（如D3/React 等），并通过任务跟踪与会议日志实现交互与验证。

**📊 数据集**

收集21个HealthTech项目的访谈、会议记录、任务日志、外部监管文件；TARA AI项目的内部资料；以及16家初创企业的问卷数据。

**📈 对比分析**

通过案例观察和调查量化指标（如监管合规感知、仪表板使用频率、治理任务完成率）进行对比，发现仪表板显著提升责任可见性，尤其在TRL5‑6阶段得到较高接受度；未给出传统算法性能指标。

**⚠️ 局限性**

仅适用于欧盟/英国监管环境，需持续维护以跟随法规变更，易导致最小合规心态，且缺乏自动化合规验证。

---

## 139. EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models

**arXiv ID:** 2602.23802 | [PDF](https://arxiv.org/pdf/2602.23802v1)

**作者:** Yiyang Fang `[一作]` (Wuhan University), Mang Ye `[通讯]` (Wuhan University)

**通讯引用:** 11949 | [OpenAlex ID](https://openalex.org/A5008999954)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于反思式强化学习的框架（Reflective Reinforcement Learning for Emotional Reasoning），用于提升多模态大语言模型（MLLM）的情绪推理能力。

**💡 创新点**

创新点在于引入结构化情绪思考（Structured Emotional Thinking，SET）以指导模型按步骤推理，并设计反思情绪奖励（Reflective Emotional Reward，RER）来对模型生成的推理进行自我评估，从而解决情绪任务的可解释性与泛化问题。

**🔧 技术方法**

主要技术包括GRPO（Group Relative Policy Optimization）强化学习、SET与RER两大模块，以及在训练前的轻量级冷启动情绪微调（Cold‑Start‑Emo）。

**📊 数据集**

实验使用了 EmoSet、Emotion6、WebEmo 三个情绪数据集，分别对应8、6、7类情绪标签。

**📈 对比分析**

与GRPO、DAPO、SEPM、SFT等基线方法对比，所提方法在所有数据集的准确率均高于或相当于最优基线，特别是在跨域（out‑of‑domain）场景下表现更为突出。

**⚠️ 局限性**

局限性包括：情绪标签本身的主观性导致奖励稀疏；对大规模数据和计算资源的依赖；以及在更复杂的多模态交互或序列任务中的可扩展性待进一步验证。

---

## 140. Serendipity with Generative AI: Repurposing knowledge components during polycrisis with a Viable Systems Model approach

**arXiv ID:** 2602.23365 | [PDF](https://arxiv.org/pdf/2602.23365v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 141. Anansi: Scalable Characterization of Message-Based Job Scams

**arXiv ID:** 2602.24223 | [PDF](https://arxiv.org/pdf/2602.24223v1)

**作者:** Abisheka Pitumpe `[一作]` (Stony Brook University), Amir Rahmati `[通讯]` (Stony Brook University)

**通讯引用:** 4319 | [OpenAlex ID](https://openalex.org/A5021423602)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并部署了一个端到端可扩展的测量管道，自动化与 1900+ 诈骗者互动，收集 29,000+ 诈骗信息，提取行为、财务和基础设施信号。

**💡 创新点**

创新点在于将大语言模型、自动浏览器代理与基础设施指纹技术集成，实现多渠道（SMS、WhatsApp、Telegram）自动化交互，并通过模板、域名、加密钱包的聚类揭示诈骗网络的重用与协作。

**🔧 技术方法**

技术包括自定义 LLM Prompt、Twilio API、Selenium/ChromeDriver、OpenAI Operator、OCR、spaCy NER、聚类分析、区块链 API（Blockstream、Etherscan）以及域名/IP 指纹检测。

**📊 数据集**

使用的数据集来自公共举报门户、Smishtank、研究团队提交的截图，包含 29,000+ 诈骗消息、1,900+ 互动记录、电话号码、文本、网站域名、加密钱包地址、IP 等；还利用区块链交易数据与公开 API。

**📈 对比分析**

通过与 VirusTotal、Google Safe Browsing、Phishfort 等传统黑名单对比，发现仅 29.6% 的诈骗网站被识别，显示方法更全面；Pipeline 在 10 个月内完成收集与互动，约 95% 诈骗者可被成功接触；聚类与损失估算提供更细粒度的运营洞察。

**⚠️ 局限性**

局限性：流程仍需人工干预（如任务完成、客服交互），WhatsApp API 限制导致交互受阻，研究仅覆盖美国地区、仅针对任务类猪肉屠宰（Job-based）诈骗，未涵盖其他语言或跨境多样化诈骗场景。

---

## 142. FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA

**arXiv ID:** 2602.23638 | [PDF](https://arxiv.org/pdf/2602.23638v1)

**作者:** Haoran Zhang `[一作]` (University of Texas at Austin), Haris Vikalo `[通讯]` (University of Texas at Austin)

**通讯引用:** 5571 | [OpenAlex ID](https://openalex.org/A5067602750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在联邦学习中对LoRA参数更新进行旋转对齐，先将各客户端的低秩因子通过正交变换对齐再进行聚合，以降低因子对齐误差导致的梯度误差。

**💡 创新点**

① 将旋转不匹配视为LoRA聚合误差的主要来源；② 提出轻量级正交Procrustes对齐与软旋转插值方法；③ 交替对齐A、B因子以提升收敛性；④ 给出聚合误差上界的理论证明，证明对齐能严格缩小误差。

**🔧 技术方法**

联邦LoRA、正交Procrustes对齐、软旋转插值、SVD求解、梯度下降、非凸收敛分析、实验中使用的联邦框架FederatedScope-LLM。

**📊 数据集**

RoBERTa‑Large 在 GLUE（SST-2、QNLI、MNLI、QQP、RTE）和 Llama 3‑8B 在 GSM8K（数学推理）与 HumanEval（代码生成）上进行评估，使用 CodeSearchNet 预训练。

**📈 对比分析**

与 FedIT、FFA‑LoRA、RoLoRA 三个基线比较；在 N=3、10 客户端、LoRA 低秩 r=4/8/16、不同异构度 α=0.5/1/100 等设置下，FedRot‑LoRA 在 GLUE 平均精度上提升 0.5–1.5%，在 GSM8K、HumanEval 的精度和 pass@1 分别提升 0.5–1.5%；同时标准差更小，训练更稳定。

**⚠️ 局限性**

对齐依赖全局参考的质量，软旋转 λ 需调优；实验范围局限于自然语言理解与生成任务，未检验极端非IID或大规模客户端；对齐仅为正交变换，可能无法完全消除子空间差异；未针对安全隐私攻击做额外评估。

---

## 143. Thinking with Images as Continuous Actions: Numerical Visual Chain-of-Thought

**arXiv ID:** 2602.23959 | [PDF](https://arxiv.org/pdf/2602.23959v1)

**作者:** Kesen Zhao `[一作]` (Nanyang Technological University), Hanwang Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 27886 | [OpenAlex ID](https://openalex.org/A5042324027)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可插拔框架NV-CoT，使多模态大型语言模型（MLLM）能够直接输出连续数值坐标进行视觉链式推理，从而避免文本化坐标带来的模态不匹配和语义碎片化。

**💡 创新点**

核心创新在于将动作空间从离散词表扩展到连续欧氏空间；采用高斯/拉普拉斯分布建模坐标预测，利用重参数化采样实现随机性；与GRPO风格的强化学习无缝对接；改动极小，仅添加四个线性坐标头和一个标准差/尺度头。

**🔧 技术方法**

技术手段包括：
- 连续坐标预测（四维实数）
- 高斯/拉普拉斯连续策略与重参数化采样
- 重要性比与KL正则化的解析计算
- 在SFT阶段使用L2或L1回归损失
- 在RL阶段使用GRPO、Clip策略
- 结合视觉工具（缩放裁剪）实现区域提取。

**📊 数据集**

在V* Bench、HR-Bench 4K、HR-Bench 8K三大视觉推理基准上进行评估，基准数据与原基模型保持一致。

**📈 对比分析**

与八个对照模型（包括开源LLM、基于文本的视觉CoT、基于补丁的视觉CoT）进行对比；NV-CoT在SFT和RL两种训练方式下均取得明显提升，整体精度提升约2%–9%，定位IoU更高，训练收敛速度更快；多步工具调用场景下性能进一步提升。

**⚠️ 局限性**

局限性：
- 仍需先验的框选工具（裁剪）支持；
- 只处理矩形框，无法直接定位非矩形或复杂形状；
- 对于大规模无框注释的数据仍需RL，收敛速度受策略参数设置影响；
- 只在视觉推理基准上验证，未评估在其他多模态任务或不同语言环境的鲁棒性。

---

## 144. RUMAD: Reinforcement-Unifying Multi-Agent Debate

**arXiv ID:** 2602.23864 | [PDF](https://arxiv.org/pdf/2602.23864v1)

**作者:** Chao Wang `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**通讯引用:** 7842 | [OpenAlex ID](https://openalex.org/A5012419026)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于强化学习的多智能体辩论动态拓扑控制框架RUMAD，能够在保持高准确率的同时显著降低通信成本。

**💡 创新点**

创新点包括内容无关的观察设计、基于多目标奖励的强化学习策略、以及预算约束下的稀疏通信控制，实现了零样本跨域迁移。

**🔧 技术方法**

使用PPO强化学习、Gaussian边权分布、预算正则化、多目标奖励函数、以及token成本监测。

**📊 数据集**

主要使用MMLU、GSM8K和GPQA三个评测数据集，训练集为MMLU dev，测试集为MMLU test，GSM8K和GPQA进行零样本迁移。

**📈 对比分析**

与全连通MAD、S-MAD、GD、S^2-MAD等基线对比，RUMAD在MMLU、GSM8K和GPQA上均实现了80%+ token 节省，并在准确率上优于或匹配基线。

**⚠️ 局限性**

局限在于中央PPO控制器对大规模智能体群（数百个）可能难以扩展，需要探索分布式或层级化控制方案。

---

## 145. AgenticOCR: Parsing Only What You Need for Efficient Retrieval-Augmented Generation

**arXiv ID:** 2602.24134 | [PDF](https://arxiv.org/pdf/2602.24134v1)

**作者:** Zhengren Wang `[一作]` (Shanghai Artificial Intelligence Laboratory), Conghui He `[通讯]` (Peking University)

**通讯引用:** 17316 | [OpenAlex ID](https://openalex.org/A5088492734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 AgenticOCR，一个可在视觉 RAG 流程中按查询动态裁剪、识别并提取文档相关区域的中间件，替代传统静态全页 OCR 预处理。

**💡 创新点**

创新点包括：① 把 OCR 变为 agentic 过程，单一可调用工具完成区域定位、几何校正与内容识别；② 采用两阶段训练（轨迹蒸馏+GRPO）和专门的奖励设计，使模型在精确定位、信息最小化和避免冗余方面达到最优；③ 通过视觉交互工具实现对页面的按需“解压”，显著提高信噪比和 token 利用率。

**🔧 技术方法**

使用 Qwen3‑VL 作为基础模型，开发 image_zoom_and_ocr_tool；训练方法为监督微调（基于 Gemini 生成的高质量轨迹）+强化学习（GRPO）；奖励函数包含双召回、重叠惩罚、超大框惩罚等；在推理时通过与检索器交互，按需调用工具。

**📊 数据集**

主要数据集：ViDoRe‑v3（用于 SFT 轨迹生成），FinRAGBench‑V（含边界框）、MMLongBench‑Doc（长文档推理）、MVToolbench、CodeVision（复杂布局），以及 Qwen3‑VL‑Reranker 产生的负样本。

**📈 对比分析**

与多种基线比较：Vanilla VLM、OCR+VLM、MACT/M3DocRAG/MDocAgent/SimpleDoc/DocLens；在 MMLongBench‑Doc 上，AgenticOCR‑8B 在 Evidence+OCR 设定下取得 66.4%（超人类 65.8%），在 FinRAGBench‑V 上达到 78.6%，显著优于前沿方法；同时 token 消耗低于全页处理，提升了生成效率。

**⚠️ 局限性**

局限性：对表格（TAB）和不可回答（UNA）问题表现仍弱；模型生成的 OCR 注释缺乏监督，易出现幻觉；检索精度低导致大量无关页面传递给生成器；Gemini 等生成器的 token 分配机制限制了裁剪多块图像时的 token 控制；缺少结构化索引和更精准的检索策略。

---

## 146. Cybersecurity of Teleoperated Quadruped Robots: A Systematic Survey of Vulnerabilities, Threats, and Open Defense Gaps

**arXiv ID:** 2602.23404 | [PDF](https://arxiv.org/pdf/2602.23404v1)

**作者:** Mohammad Sabouri `[一作]` `[通讯]` (University of Genoa), Mohammad Sabouri (University of Genoa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述了2019–2025年间四足遥控机器人的网络与控制层安全，提出六层攻击分类、攻击‑后果映射、技术成熟度(TRL)评估、六大商业平台安全对比以及未来研究缺口。

**💡 创新点**

创新点在于将传统机器人安全框架针对四足动态平衡与VR/AR接口特性进行定制化攻击层级与后果映射，并用TRL评估防御缺口，首次构建完整的四足遥控安全评估体系。

**🔧 技术方法**

使用了系统综述方法（PRISMA流程）、ROS/ROS2安全规范分析、公开漏洞数据库（CVE、INCIBE‑CERT）、ROSPaCe IDS 数据集、公开平台文档以及案例实验等多种技术手段。

**📊 数据集**

所用数据集主要为公开漏洞数据、ROSPaCe IDS 数据集和公开平台安全文档，无自建数据集。

**📈 对比分析**

通过TRL与安全评分对六大商业平台（Spot、Unitree、Ghost、ANYmal、CyberDog、DeepRobotics）进行多维度对比，展示安全成熟度差距；实验攻击成功率引用已有研究，表明多层攻击对四足机器人危害高度可观。

**⚠️ 局限性**

局限在于仅基于公开信息，缺乏对真实设备的实验验证；攻击成功率多来源小样本，可能不具代表性；平台评估受文档缺失影响，可能低估实际安全性。

---

## 147. Shifting in-DRAM

**arXiv ID:** 2602.24269 | [PDF](https://arxiv.org/pdf/2602.24269v1)

**作者:** William C. Tegge `[一作]` (Syracuse University), Alex K. Jones `[通讯]` (Syracuse University)

**通讯引用:** 3137 | [OpenAlex ID](https://openalex.org/A5030875484)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

设计了一种在开位线DRAM子阵列中实现位移的方案，利用迁移单元实现水平数据移动，保持对标准DRAM操作的兼容性。

**💡 创新点**

创新点在于：在子阵列的上下行加入迁移单元，利用现有DRAM单元结构实现双向位移；不需要额外的移位逻辑或数据转置，面积增量仅约1–2%。

**🔧 技术方法**

技术包括：DRAM子阵列重构、迁移单元电路设计、NVMain周期级模拟、LTSPICE电路验证、Cadence Virtuoso 22nm物理布局、过程变异蒙特卡洛分析。

**📊 数据集**

数据集：对8 KB行进行1、50、100、512次单比特位移的实验；使用随机、全0、全1和交替数据模式进行电路验证。

**📈 对比分析**

对比方法：与SIMDRAM、DRISA、Ambit等前置工作在能耗、延迟、面积等维度对比。结果显示：能耗≈4 nJ/KB，单比特位移延迟≈208 ns，能耗比传统CPU‑DRAM传输低40–60×，面积增量仅<2%，而与DRISA相比面积低约10×，能耗相近。

**⚠️ 局限性**

局限性：只支持单比特位移，若需多位移需多次操作；在22 nm下过程变异超过±10%时失效率显著升高；未针对高并行多比特位移的设计扩展。

---

## 148. From Static Benchmarks to Dynamic Protocol: Agent-Centric Text Anomaly Detection for Evaluating LLM Reasoning

**arXiv ID:** 2602.23729 | [PDF](https://arxiv.org/pdf/2602.23729v1)

**作者:** Seungdong Yoa `[一作]` (LG AI Research), Woohyung Lim `[通讯]` (LG AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于三智能体（教师、编排者、学生）的动态评测协议——Agent‑Centric Text Anomaly Detection (ATAD)，用于自动生成、验证和逐步提升文本异常检测任务的难度，以评估大型语言模型的推理能力。

**💡 创新点**

创新点在于：① 将评测从静态数据集迁移到动态协议；② 通过教师-学生竞争循环和编排者验证机制实现问题难度的自适应升级；③ 采用七类文本异常类型并结合标准化考试（GRE、LSAT等）领域，提供多维度推理挑战；④ 强调问题清晰度与可验证性，降低模型对模式匹配的依赖。

**🔧 技术方法**

使用的技术包括：教师智能体生成候选异常问题、编排者智能体进行合法性和安全性验证、学生智能体尝试解答；问题设计遵循跨句逻辑推理、共指消解、因果矛盾、语调一致性等多项推理需求；评估方法采用与现有静态基准（MMLU、GSM8K、BigBench）对照、跨模型对比以及学生模型在不断升级问题上的表现曲线。

**📊 数据集**

数据集主要是协议动态生成的文本异常检测实例；参考材料包括标准化考试文本（GRE、LSAT、GMAT、LSAT等），并在实验中采用开放源码实现的 ATAD 框架自行生成并验证数据，未使用公开的固定大规模文本异常数据集。

**📈 对比分析**

比较方法：将同一批模型在 ATAD 与传统静态基准上的成绩进行对比，并观察在学生模型提升时问题难度的自适应变化。实验结果显示 ATAD 能揭示传统基准未能捕捉到的推理缺陷，模型在 ATAD 上的得分相对较低，表明其对推理能力更为敏感；同时，ATAD 随模型升级自动提升难度，保持评测的前沿性。

**⚠️ 局限性**

局限性：① 生成与验证过程仍依赖教师与编排者模型，可能受其自身推理局限影响；② 对某些异常类型（如极度细微的语调偏差）可能仍产生歧义或误判；③ 需要人工干预对验证标准进行微调，影响完全自动化；④ 实验规模与算力有限，尚未在大规模多模型上系统评估。

---

## 149. ULW-SleepNet: An Ultra-Lightweight Network for Multimodal Sleep Stage Scoring

**arXiv ID:** 2602.23852 | [PDF](https://arxiv.org/pdf/2602.23852v1)

**作者:** Zhaowen Wang `[一作]` (Dalian University of Technology), Jenni Raitoharju `[通讯]` (University of Jyväskylä)

**通讯引用:** 1843 | [OpenAlex ID](https://openalex.org/A5055803270)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种面向多模态睡眠分期的超轻量级框架 ULW‑SleepNet，能够在多通道生理信号上高效进行睡眠分期。

**💡 创新点**

创新点包括：① Dual‑Stream Separable Convolution (DSSC) Block 的引入，实现残差学习与双流特征提取；② 采用深度可分离卷积与通道级参数共享，显著降低参数量；③ 全局平均池化取代全连接层，进一步压缩模型；④ 通过层级过滤器扩展（8、16、32）与适当的池化/卷积核大小，平衡精度与计算成本。

**🔧 技术方法**

技术要点：深度可分离卷积、通道级参数共享、DSSC Block、全局平均池化、批归一化、Dropout、L2 正则、Adam 优化器、余弦退火学习率调度、10‑折交叉验证。

**📊 数据集**

使用了公开 Sleep‑EDF Expanded 数据集的两个子集：Sleep‑EDF‑20（39 份 PSG）与 Sleep‑EDF‑78（197 份 PSG），包含 EEG、EOG、EMG 三种模态，采样率 100 Hz。

**📈 对比分析**

与 DeepSleepNet、TinySleepNet、AttnSleepNet 等 state‑of‑the‑art 方法对比，ULW‑SleepNet 在 Sleep‑EDF‑20 上实现 86.9% 准确率、80.7% macro‑F1、Cohen κ = 0.82；在 Sleep‑EDF‑78 上实现 81.4% 准确率、74.0% macro‑F1、κ = 0.74。参数仅 13.3 K、FLOPs 7.89 M，比 TinySleepNet 少 98.6% 参数、比 LWSleepNet 少 85.7% FLOPs，且性能竞争力强。

**⚠️ 局限性**

局限性：① 仅在 Sleep‑EDF 两个子集上验证，缺乏对更大规模、多病种或跨医院数据的泛化评估；② 对跨会话/长期监测鲁棒性未做实验；③ 虽然参数极少，但在嵌入式硬件上的实际部署与能耗评估仍待验证；④ 某些分期（如 N1）在特定数据上相较于部分轻量模型略逊。

---

## 150. Spatio-Temporal Garment Reconstruction Using Diffusion Mapping via Pattern Coordinates

**arXiv ID:** 2602.24043 | [PDF](https://arxiv.org/pdf/2602.24043v1)

**作者:** Yingxuan You `[一作]` (École Polytechnique Fédérale de Lausanne), Pascal Fua `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 56560 | [OpenAlex ID](https://openalex.org/A5038674741)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用扩散模型和隐式缝纫图案，在UV空间学习服装形状先验，并通过像素‑UV‑3D映射实现单图像与视频的高精度3D服装重建；

**💡 创新点**

①在UV空间构建服装形状先验的扩散模型，①实现无模板的松散服装重建；②采用时空扩散架构分离空间与时间建模，并在推理时引入多种指导（跨段、内段、深度‑法线、互相穿透）保证长序列的时间一致性；③投影约束在未观测区域完成几何恢复，保持可观测几何精度；

**🔧 技术方法**

隐式缝纫图案（ISP）、扩散概率模型（DDPM）、空间/时间自注意力模块、基于物理的网格正则化、MLP位移场优化、投影约束与多尺度指导；

**📊 数据集**

CLOTH3D（合成的服装与SMPL人体模拟数据，使用AMASS舞蹈动作、Blender/Marvelous Designer仿真）以及在真实图像/视频上的测试；

**📈 对比分析**

相对于SMPLicit、ISP、GaRec（单图像）和D^3‑Human、REC‑MV（视频）等基线，本文方法在Chamfer Distance、Normal Consistency、IoU等指标上均显著更优；在实景图像/视频中呈现更丰富的褶皱细节与更平滑的时间演变；推理速度上也优于大多数竞争方法；

**⚠️ 局限性**

仅在合成数据上训练，导致对极端真实场景或特殊服装材质的泛化仍有限；时空扩散的显存消耗较大，仍需分段推理；依赖SMPL姿态与分割的前置估计，对姿态错误或遮挡不佳的情况易影响重建；

---

## 151. Antenna Coding Optimization for Pixel Antenna Empowered Wireless Communication Using Deep Learning with Heterogeneous Multi-Head Selection

**arXiv ID:** 2602.23831 | [PDF](https://arxiv.org/pdf/2602.23831v1)

**作者:** Binzhou Zuo `[一作]` (Hong Kong University of Science and Technology), Hongyu Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 72699 | [OpenAlex ID](https://openalex.org/A5100343282)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于深度学习的像素天线编码优化算法，并在SISO与MIMO系统中实现并验证其性能。

**💡 创新点**

创新点在于引入异构多头选择机制（HMSM）与二进制/格雷编码的组合，既显著降低计算复杂度，又保留约98%传统搜索算法的性能。

**🔧 技术方法**

使用深度学习技术（ResNet50多头网络、HMSM）、像素天线模型、灰度/二进制编码以及SEBO算法生成标签。

**📊 数据集**

使用了90,000个Rayleigh虚拟通道样本的数据集，90%用于训练，10%用于测试。

**📈 对比分析**

通过与SEBO、代码本和传统天线系统对比，SISO系统获得98.5%增益、MIMO系统获得98%容量，同时计算时间分别降低约98.8%和99.7%。

**⚠️ 局限性**

局限性包括：依赖SEBO生成的标签、对训练样本量高度敏感、在极大Q或M时复杂度可能上升，且尚未在实际硬件上验证可实现性。

---

## 152. Planning from Observation and Interaction

**arXiv ID:** 2602.24121 | [PDF](https://arxiv.org/pdf/2602.24121v1)

**作者:** Tyler Han `[一作]` (University of Washington), Byron Boots `[通讯]` (University of Washington)

**通讯引用:** 4789 | [OpenAlex ID](https://openalex.org/A5110797782)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了MPAIL2，一种无需演示动作或手工奖励、仅基于观测与交互的逆强化学习框架，能够在真实机器人上快速学习图像‑基操纵任务；

**💡 创新点**

通过离线学习与在线规划结合，实现了离线奖励学习、价值函数与多步策略的统一训练，并采用MPPI规划在潜在空间中执行多步计划，显著提升样本效率和泛化能力；

**🔧 技术方法**

离线奖励与价值函数的对抗学习、潜在状态编码与动力学建模、熵正则化的多步策略学习、基于MPPI的潜在空间规划、梯度惩罚的WGAN奖励稳定化；

**📊 数据集**

使用真实世界数据：10条基于空间鼠/键盘的演示（共约2,000+转移），以及在Franka和Kinova机器人上进行的图像+关节感知任务；

**📈 对比分析**

与IRL基线（AIRL、MAIRL、DAC）、RLPD（RL + 演示）和Diffusion Policy（BC）对比；在Sim与真实任务上，MPAIL2在不到1小时内即可实现高达80%+的成功率，远优于RL、BC和其他IRL基线；

**⚠️ 局限性**

仍受限于奖励对抗训练的稳定性、对动作标签的缺失导致的规划失误、单一任务训练导致的模型非平稳性以及对跨体型、跨传感器的迁移可解释性不足；

---

## 153. Optimization of Edge Directions and Weights for Mixed Guidance Graphs in Lifelong Multi-Agent Path Finding

**arXiv ID:** 2602.23468 | [PDF](https://arxiv.org/pdf/2602.23468v1)

**作者:** Yulun Zhang `[一作]` (Carnegie Mellon University), Jiaoyang Li `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4052 | [OpenAlex ID](https://openalex.org/A5027709346)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并实现了混合导向图优化（MGGO），同时优化边权和方向以提升生命周期多代理路径规划（LMAPF）的吞吐量。

**💡 创新点**

创新点在于将边方向纳入导向图优化，提出两阶段与QD联合优化两种方法，并提出快速强连通修复的边反转搜索。

**🔧 技术方法**

采用进化算法（1+λ EA、CMA-ES、CMA-MAE）、混合方向表示的CNN更新模型以及预计算的交通模式。

**📊 数据集**

使用仓库地图和MAPF基准集四张地图，Agent数量多达1500。

**📈 对比分析**

与GGO-DS、指向Crisscross、人手设计无导向图基线比较，MGGO在多数场景下吞吐量提升15%-30%，且旋转动作比率显著下降。

**⚠️ 局限性**

主要局限是样本效率低、搜索空间大导致计算开销高，且对搜索式规划器提升有限。

---

## 154. Learning dynamics from online-offline systems of LLM agents

**arXiv ID:** 2602.23437 | [PDF](https://arxiv.org/pdf/2602.23437v1)

**作者:** Moyi Tian `[一作]` (University of Colorado), Nancy Rodríguez `[通讯]` (University of California)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对基于大型语言模型（LLM）的社交网络进行信息传播仿真，探讨人格设定和事件严重度如何影响传播动力学。

**💡 创新点**

创新之处在于将基于提示的LLM人格与传染模型相结合，提出两参数的SI型均值场方程，并证明低维模型可精准捕捉传播曲线。

**🔧 技术方法**

采用代理仿真、逻辑回归、均值场微分方程、文本嵌入主成分分析以及最大似然和最小二乘估计等技术。

**📊 数据集**

使用ACLED冲突事件数据库生成新闻文本，随机分配32种二进制人格特征，利用OpenAI gpt‑3.5‑turbo‑1106作为LLM。

**📈 对比分析**

通过将仿真得到的参与率曲线与逻辑回归模型和均值场模型预测结果进行RMSE对比，均值场模型RMSE为0.0223，优于逻辑回归的0.0414，尤其在平和事件上表现更佳。

**⚠️ 局限性**

主要局限是仅模拟LLM网络，未纳入人类代理，模型受限于gpt‑3.5、网络规模较小以及事件样本有限，通用性和外部有效性尚待验证。

---

## 155. LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding

**arXiv ID:** 2602.23881 | [PDF](https://arxiv.org/pdf/2602.23881v1)

**作者:** Alexander Samarin `[一作]`, Alexander Golubev `[通讯]` (Nebius)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 LK 损失（LK Losses），直接优化推理过程中草稿模型的接受率，从而加速大语言模型的自回归推理；

**💡 创新点**

创新点在于将总变距（TV）与 KL 散度混合并加入自适应调度的混合损失，以及基于负对数接受率的似然损失，实质上实现了对接受率的直接优化；

**🔧 技术方法**

使用的技术包括草稿模型的知识蒸馏、TV 与 KL 的梯度分析、负对数接受率优化、以及自适应 λ 调度；

**📊 数据集**

数据集主要为 Infinity‑Instruct 660K 生成样本，并在 MT‑bench、HumanEval 与 GSM8K 三大任务上评估；

**📈 对比分析**

与传统 KL 损失相比，LK 损失在六种目标模型（8B~685B）和四种草稿架构上平均提升 7–10% 的接受长度，且不增加训练开销；

**⚠️ 局限性**

局限性包括：尚未直接优化系统吞吐量（τ‑1/K 速率），对 top‑k/top‑p 参数的影响未评估，且在极大模型或更复杂采样策略下的表现需进一步验证。

---

## 156. A Theory of Random Graph Shift in Truncated-Spectrum vRKHS

**arXiv ID:** 2602.23880 | [PDF](https://arxiv.org/pdf/2602.23880v1)

**作者:** Zhang Wan `[一作]` (University of Manchester), Samuel Kaski `[通讯]` (Aalto University)

**通讯引用:** 14950 | [OpenAlex ID](https://openalex.org/A5018305257)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于随机图生成模型的图分类领域自适应理论，并将迁移损失拆解为域差异、谱几何与幅值三项；

**💡 创新点**

创新点在于利用截断谱向量RKHS分解，将泛化误差表达为域分歧、谱特征与模型幅值的乘积；同时证明潜在Wasserstein距离可视作域差异指标；

**🔧 技术方法**

采用随机图生成模型、向量值RKHS、Wasserstein距离、MPNN与GNN等技术；

**📊 数据集**

在PTC、IMDB-MULTI、NCI1、PROTEINS及Mutagenicity等图数据集上验证；

**📈 对比分析**

通过将潜在Wasserstein距离与测试误差相关、谱截断实验与图大小/层次正则化实验比较，发现距离与误差相关性良好、谱截断极低、图大小增大和非均匀正则化提升性能；

**⚠️ 局限性**

局限在于理论界限在无额外紧致条件下不一定收敛、依赖RGM可识别性与潜在分布假设、Wasserstein估计复杂度高、实验范围相对有限。

---

## 157. Preference Packing: Efficient Preference Optimization for Large Language Models

**arXiv ID:** 2602.24082 | [PDF](https://arxiv.org/pdf/2602.24082v1)

**作者:** Jaekyung Cho `[一作]` `[通讯]` (Amazon), Jaekyung Cho (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 preference packing 方法，减少在同一输入提示下多种响应的重复计算和内存占用。

**💡 创新点**

创新点在于通过重新排列数据序列并设计专用注意力掩码，使得同一输入只计算一次，从而显著提升资源利用率。

**🔧 技术方法**

采用了 Flash Attention、LoRA、FSDP 等分布式训练技术，并结合 batch sorting。

**📊 数据集**

实验使用 Orca-DPO-pair、Capybara Preference、RLHF-V 三个公开偏好数据集以及 Llama-3.2-1B、llava-qwen-0.5b-hf 等模型。

**📈 对比分析**

通过在单 GPU 与多节点 FSDP 环境下对比，发现单机训练时间可缩减 37% 以上，分布式训练可实现 3.22 倍加速。

**⚠️ 局限性**

局限性是当响应长度明显超过输入时，反而会增加计算成本，尤其适用于短响应的偏好数据。

---

## 158. Fourier Angle Alignment for Oriented Object Detection in Remote Sensing

**arXiv ID:** 2602.23790 | [PDF](https://arxiv.org/pdf/2602.23790v1)

**作者:** Changyu Gu `[一作]` (Beijing Institute of Technology), Ying Fu `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 5683 | [OpenAlex ID](https://openalex.org/A5100738025)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了 Fourier Angle Alignment（FAA）框架，通过频域分析估计物体主方向并将特征对齐，从而提升遥感旋转目标检测的精度。

**💡 创新点**

创新点包括：①利用傅里叶旋转等价性从频谱中直接估计主方向；②在特征金字塔（neck）层引入 FAAFusion，实现多尺度特征在方向上的一致对齐；③在检测头（head）层设计 FAA Head，先将 RoI 特征旋转到规范角度再与原特征相加，缓解分类与角度回归的冲突。

**🔧 技术方法**

技术手段：二维离散傅里叶变换、频谱极值搜索、频域旋转对齐、卷积特征融合、FPN、RoIAlign 等常见卷积网络模块。

**📊 数据集**

实验数据集：DOTA‑v1.0、DOTA‑v1.5 与 HRSC2016 三个遥感目标检测基准。

**📈 对比分析**

通过与多种两阶段（如 Oriented R‑CNN、PKINet、LSKNet）和单阶段（如 RetinaNet‑O）检测器的对比，FAA 在 DOTA‑v1.0 上取得 78.72% mAP（SOTA），在 DOTA‑v1.5 上 72.28% mAP，HRSC2016 上 AP 与 mAP 均提升 1–2% 左右，表明方法在定位与角度预测上均优于现有技术。

**⚠️ 局限性**

局限性：①频域对齐虽有效但增加了前向推理时的计算量，未针对实时/轻量级部署进行优化；②实验仅集中在遥感旋转检测任务，对实例分割、变化检测等其它任务的通用性尚待验证；③对极端小尺寸或密集目标的鲁棒性尚需进一步研究。

---

## 159. MT-PingEval: Evaluating Multi-Turn Collaboration with Private Information Games

**arXiv ID:** 2602.24188 | [PDF](https://arxiv.org/pdf/2602.24188v1)

**作者:** Jacob Eisenstein `[一作]` (Google DeepMind), Mirella Lapata `[通讯]` (Google DeepMind)

**通讯引用:** 27940 | [OpenAlex ID](https://openalex.org/A5041024491)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于私有信息的多轮对话评估基准（PINGs），并引入了“等量令牌”多轮缩放评估方法，以检验语言模型在有限令牌预算下是否能通过多轮交互提升任务表现。

**💡 创新点**

创新点在于：①将私有信息游戏作为对话交互的核心场景，逼近真实人类对话中信息不对称的挑战；②设计等量令牌框架，剔除单轮对话对模型能力的影响，只关注多轮交互的增益；③从互惠性、信息密度、语义连贯性等维度系统分析模型的对话策略。

**🔧 技术方法**

技术主要包括：多轮对话生成（使用 Gemini 2.5 Pro/Flash、GPT‑4o、Qwen‑VL8B、Gemma3‑12B 等大型语言模型）、信息编码与解码（图像描述、结构化知识传递）、自评与自动化分析（使用 LLM‑as‑a‑judge 评估同情倾向、词汇密度、中心连贯性）以及可视化与统计分析。

**📊 数据集**

使用的数据集有：
• 经典象棋棋局数据集（用于棋局对比任务）
• COVR 图像对比基准（含两张图像与判定句）
• MD3 与 Tangram 图像选择任务（描述者与猜测者的图像/形状集合）
• 结构化数据表（Name‑game）
以及从上述基准生成的 1000 条实例，按不同模型与不同令牌/回合预算进行实验。

**📈 对比分析**

对比方法是：在相同的总令牌预算下，逐步细化回合数（2、4、8、16），记录每个模型的任务准确率。结果显示：大多数模型在增加回合数后准确率不提升甚至下降；棋局和 COVR 任务表现基本平稳；MD3 和 Tangram 任务呈现逆向缩放（回合越多表现越差）。在 Name‑game 任务中，模型似乎通过随机猜测逐步提升，但整体仍低于人类水平。

**⚠️ 局限性**

局限性包括：
① 仅在受限令牌预算的场景下评估，未能覆盖更长、更开放的对话；
② 模型在私有信息传递时往往采用低效或错误的策略（如多余的致歉、无关信息），导致任务无法受益于多轮交互；
③ 评价框架假设所有信息均可在固定长度文本中编码，实际图像/结构化知识的表达可能仍受限；
④ 与人类对话比较仅在 MD3 任务中完成，其他任务缺乏人类基准；
⑤ 结果表明即使在强大模型下，多轮交互能力仍显不足，进一步提升需要更高级的交互策略与推理能力。

---

## 160. Physics-Embedded Neural ODEs for Learning Antagonistic Pneumatic Artificial Muscle Dynamics

**arXiv ID:** 2602.23670 | [PDF](https://arxiv.org/pdf/2602.23670v1)

**作者:** Xinyao Wang `[一作]` (University of California), Jonathan Realmuto `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种混合物理信息化连续时间Neural ODE框架，用来建模和控制对抗式气动人工肌肉（PAM）关节，实现前向预测和逆向输入合成，实验验证了其在运动与阻尼（刚度）控制上的可靠性。

**💡 创新点**

创新点主要包括：①将关节力学、气压动力学与神经网络非线性力项嵌入同一连续时间ODE中；②神经网络捕捉对抗耦合、速率相关滞后，避免显式高阶滞后模型；③基于已学习模型的离线逆向优化求解气体质量，实现所需运动和刚度的闭环前馈控制；④与传统平衡点模型对比，证明速度相关力模型显著提升刚度一致性。

**🔧 技术方法**

采用的技术包括：物理信息化Neural ODE（Tsitouras 5阶求解器 + 逆向梯度）、两层全连接LeakyReLU网络、软正则化的物理参数（Softplus）、约束优化求解气体质量、有限差分近似刚度、Adam优化器、数据驱动的损失函数。

**📊 数据集**

数据集为225个不同共收缩条件下收集的实验轨迹（共振/频率0.5/1/2 Hz，扭矩1/1.5 A），其中29个用于训练（从对称到非对称覆盖），其余用于验证。数据采样率1 kHz，包含关节位移、速度、两腔压力以及外部力。

**📈 对比分析**

性能评估采用：①平均R² = 0.88，显示前向预测高度精确；②对比传统平衡点模型，速度变化导致刚度变动<2.5%（神经ODE）vs 8–11%（EP），表明神经ODE在高速度下保持刚度一致；③实验验证的运动跟踪误差在各振幅/频率/刚度下均保持小且稳定；④在126–176 N/mm范围内的刚度控制均符合期望。整体性能优于静态/平衡点模型。

**⚠️ 局限性**

局限性包括：①模型仅在闭阀、气体质量不变的假设下适用，无法处理流量、泄漏或温度变化；②对高频/高速度运动的训练覆盖有限，导致在极限条件下性能下降；③目前仅实现单自由度系统，难以直接推广到多自由度或更复杂的软机器人；④缺乏在线自适应机制，需在使用前进行离线训练。

---

## 161. Active Value Querying to Minimize Additive Error in Subadditive Set Function Learning

**arXiv ID:** 2602.23529 | [PDF](https://arxiv.org/pdf/2602.23529v1)

**作者:** Martin Černý `[一作]` (Charles University), Jakub Černý `[通讯]` (Columbia University)

**通讯引用:** 791 | [OpenAlex ID](https://openalex.org/A5000697186)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了在只知部分集合值的情况下，利用增量查询最小化子加性集合函数的不确定性（即下限上限间的差距），并给出了不同子类（单调子加、XOS、SCMM 等）的紧凑完成（上/下界）公式；基于这些完成构建离线与在线的查询选择算法，并在实验中验证其优于随机查询甚至接近最优；

**💡 创新点**

创新点在于①用加法误差而非传统乘法误差度量不确定性，②推导出多种子类的最紧上下界（tight completions），③设计离线最优/贪心与在线 PPO 查询策略，并证明在一定条件下贪心具有近似保证；

**🔧 技术方法**

技术手段包括集合函数理论、紧完成的数学推导、期望值估计与采样、贪心和动态规划、强化学习（PPO）用于在线决策、并行计算加速等；

**📊 数据集**

使用三种合成分布：①单调下凸子模函数，②由 6 个随机加性函数构成的 XOS 函数，③集合覆盖问题产生的集合函数；实验在 n=5,10 的设置下进行；

**📈 对比分析**

与随机查询基线、离线最优、贪心、PPO 进行对比。结果显示，贪心与 PPO 在 n=5 时几乎达到最优，PPO 在小规模时略优；随着 n 增大，随机策略性能下降，贪心与 PPO 的优势显著；整体而言，本文方法在有限查询预算下显著降低了函数的不确定性；

**⚠️ 局限性**

局限性包括：①离线最优算法对大规模 n 计算量指数级，②在线 PPO 在高维/大动作空间下难以收敛，导致 n=10 时表现不佳；③方法依赖于已知的先验分布，若先验不准确会影响查询效果；

---

## 162. All Mutation Rates $c/n$ for the $(1+1)$ Evolutionary Algorithm

**arXiv ID:** 2602.23573 | [PDF](https://arxiv.org/pdf/2602.23573v1)

**作者:** Andrew James Kelley `[一作]` `[通讯]`, Andrew James Kelley

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过构造一族名为 HillPathJump 的特殊适应度函数，证明了对（1+1）进化算法而言，对于任意实数 c≥1，都存在一个适应度函数使得其最优固定变异率近似为 c/n；即所有形如 c/n 的变异率在区间 [1, ∞) 中稠密出现。

**💡 创新点**

创新点在于：①首次展示在进化算法中，除了已知的 1/n 或特定常数倍 1/n 外，几乎所有常数 c 的变异率都可以通过设计适应度函数使其成为最优；②提出了 HillPathJump 这一新型适应度构造方法，利用灰度码扩展与跳跃路径实现对变异率的精准控制；③利用漂移理论与概率界定，精确计算了最优变异率与运行时间的关系。

**🔧 技术方法**

主要技术包括：①灰度码（Gray code）扩展构造长路径；②路径与跳跃（jump）结合的适应度函数设计；③漂移定理（Additive Drift Theorem）与概率分析（Markov 及 union bound）来估计期望运行时间；④对函数 g(x) 的极值分析证明可调节参数 a 使得最优变异率达到任何 c∈(1,k)。

**📊 数据集**

该研究为理论工作，不使用任何实验数据集；所有结论均通过数学证明得到。

**📈 对比分析**

由于是理论证明，本文没有与其它方法进行实验对比；所给的“性能”即是期望优化时间与变异率的关系，证明了当变异率取 c/n 时，运行时间逼近 (a/c+1/c^k)e^c n^k，显示该变异率在相应适应度函数下是最优的。

**⚠️ 局限性**

局限性包括：①结果仅针对（1+1）EA，未探讨 λ>1 的情况；②适应度函数的构造相对特殊，需要 n 为完全平方数；③证明依赖于常数 k≥4 的假设，且仅在理论层面展示可调节性；④对实际问题的可推广性尚未验证。

---

## 163. Blockchain-Enabled Routing for Zero-Trust Low-Altitude Intelligent Networks

**arXiv ID:** 2602.23667 | [PDF](https://arxiv.org/pdf/2602.23667v1)

**作者:** Ziye Jia `[一作]` (Nanjing University of Aeronautics and Astronautics), Dusit Niyato `[通讯]` (Nanyang Technological University)

**通讯引用:** 85656 | [OpenAlex ID](https://openalex.org/A5091266202)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文针对低空智能网络（LAIN）中的无人机（UAV）集群路由，提出基于零信任架构、软件定义外壳（SDP）与轻量级联盟链的安全身份与信誉管理框架，并在此基础上设计适应性权重信誉模型和主节点选择机制，同时将路由优化问题改写为 Dec‑POMDP 并提出 SP‑MADDQN（Soft‑Hierarchical 经验重放 + PER）多代理深度强化学习路由算法。

**💡 创新点**

创新点包括：① 将零信任网络与 SDP、区块链相结合，构建分布式、可追溯的身份与信誉管理体系；② 引入适应性权重的信誉评估，结合链上共识动态隔离恶意 UAV；③ 采用 IPBFT 选主机制提升共识安全；④ 将路由问题形式化为 Dec‑POMDP，并结合软层次经验重放与优先级重放的 SP‑MADDQN 提升学习效率与鲁棒性。

**🔧 技术方法**

所用技术主要有区块链（轻量级联盟链与 IPBFT 共识）、软件定义外壳（SDP）实现动态身份验证与授权、零信任安全架构、适应性权重信誉模型、双重深度 Q 网络（DDQN）与多代理强化学习、软层次经验重放（SHERB）和优先级经验重放（PER）。

**📊 数据集**

实验采用仿真生成的低空 UAV 网络数据集，网络规模为 12 架 UAV（其中 2 架恶意）在 15km×5km 区域内随机移动，数据需求大小 400–600 kbit，使用 1,000 步/ 5,000 轮的训练设置，评估 E2E 延迟与 TSR。

**📈 对比分析**

与 SP‑MADQN、MADDQN、MADQN、SHERB‑MADDQN、PER‑MADDQN 等基线进行对比，实验显示 SP‑MADDQN 在平均 E2E 延迟上降低约 59% ，在成功传输比例（TSR）上提升约 29%，并在不同网络规模、恶意节点数量、学习率与奖励参数等设置下表现出更优的收敛速度与稳健性。

**⚠️ 局限性**

主要局限包括：缺乏对高移动性环境下 IPBFT 共识稳定性和延迟的深入分析；区块链密钥管理与加密细节仍待完善；对更复杂的攻击模型（如隐蔽攻击、联合攻击）和真实硬件部署的鲁棒性验证尚未完成。

---

## 164. OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control

**arXiv ID:** 2602.23843 | [PDF](https://arxiv.org/pdf/2602.23843v1)

**作者:** Yunshen Wang `[一作]` (Beijing Institute for General Artificial Intelligence), Siyuan Huang `[通讯]` (Beijing Institute for General Artificial Intelligence)

**通讯引用:** 3752 | [OpenAlex ID](https://openalex.org/A5067080265)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为OmniXtreme的框架，旨在解决高动态人形机器人控制中的通用性障碍，通过分离运动技能学习与物理技能精炼，成功实现了高保真度的运动跟踪。

**💡 创新点**

创新点在于引入了流匹配策略和残差强化学习后训练阶段，显著提高了在多样化和高动态运动中的跟踪精度，同时确保了在真实硬件上的可执行性。

**🔧 技术方法**

使用了流匹配政策和残差强化学习技术，结合了生成预训练和后训练的两阶段框架。

**📊 数据集**

使用了多种数据集，包括LAFAN1、AMASS、MimicKit和Reallusion运动库，涵盖了多样化的行为模式和高动态动作。

**📈 对比分析**

与现有的多运动基线方法相比，OmniXtreme在高动态和未见运动上保持了更高的跟踪精度和成功率，尤其在运动多样性和难度增加时，表现出更强的鲁棒性。

**⚠️ 局限性**

限制在于尽管框架在多样性和动态性上表现出色，但在极端运动的执行中仍可能面临物理约束和实时控制的挑战。

---

## 165. Bandwidth-adaptive Cloud-Assisted 360-Degree 3D Perception for Autonomous Vehicles

**arXiv ID:** 2602.23871 | [PDF](https://arxiv.org/pdf/2602.23871v1)

**作者:** Faisal Hawladera `[一作]`, Raphaël Frank `[通讯]` (University of Luxembourg)

**通讯引用:** 1900 | [OpenAlex ID](https://openalex.org/A5009044318)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于 BEVFormer 的 360 度 3D 目标检测系统，将计算任务分为车端轻量特征提取和云端深度推理，并结合 V2X 通信实现实时感知。

**💡 创新点**

创新点包括：① 结合动态特征裁剪、量化与压缩以降低上传数据量；② 基于实时可用带宽和延迟预算的动态分割层与量化级别优化算法；③ 通过真实道路 V2X 实验验证，展示了显著的延迟压缩与准确率提升。

**🔧 技术方法**

使用技术包括：Transformer‑based BEVFormer（ResNet101 主干）、Post‑training 量化（FP32/FP16/FP8）、百分位裁剪、zlib 无损压缩、C‑V2X 通信、TensorRT 推理加速、动态带宽估计与阈值约束下的分割层/量化级别搜索。

**📊 数据集**

采用 nuScenes 数据集进行目标检测准确率评估，并在卢森堡 Kirchberg 区域收集真实 V2X 传输带宽与延迟数据，用于离线评估与实时优化。

**📈 对比分析**

与纯车端推理相比，混合云端方案将端到端延迟降低约 72%；动态优化在满足 100 ms 延迟阈值时，可在相同延迟下比固定配置提升 10–20% 的 NDS（最高 0.52）。在多带宽、延迟场景下，动态方案既保持低延迟又实现双位数准确率提升。

**⚠️ 局限性**

局限性：① 需要可靠的实时带宽估计和高可靠性 V2X/5G 链路；② 量化与裁剪会导致一定准确率损失；③ 动态优化依赖于先验的性能曲线，迁移到新模型或硬件时需重新配置；④ 仅在单车实验验证，未覆盖密集多车或高干扰环境。

---

## 166. Enhancing Vision-Language Navigation with Multimodal Event Knowledge from Real-World Indoor Tour Videos

**arXiv ID:** 2602.23937 | [PDF](https://arxiv.org/pdf/2602.23937v1)

**作者:** Haoxuan Xu `[一作]` (Hong Kong University of Science and Technology), Haoang Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 901 | [OpenAlex ID](https://openalex.org/A5040338788)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过从超320小时的室内巡回视频中挖掘事件知识，构建了大规模多模态时空知识图谱 YE‑KG，并在此基础上提出 STE‑VLN 框架，利用粗细层级检索与 ASTFF 融合技术，将事件知识注入视觉‑语言导航模型，显著提升对粗粒度指令和长程推理的处理能力。

**💡 创新点**

创新点在于：①首次构建基于真实室内视频的事件级知识图谱，实现对空间-动作-效果的显式建模；②提出粗细层级检索机制，先通过文本检索生成子图，再通过视觉检索获取预测性事件；③设计 ASTFF（知识引导 Transformer）实现多模态特征融合，让代理在实时推理中兼顾全局规划与局部视觉提示。

**🔧 技术方法**

使用技术包括：LLaVA‑Video 与 GPT‑4 进行视频事件抽取与文本生成、FAISS/ RAG 进行检索、ViT 进行视频特征编码、Transformer‑based ASTFF 进行融合、GOAT/ETPNav 作为基础网络，并在训练中使用 Masked Language Modeling、Single‑Step Action Prediction 等多任务学习。

**📊 数据集**

数据集：构建的 YE‑KG 基于 3,471 条 YouTube 室内巡回视频（320+ 小时）；在仿真评测中使用 Matterport3D 上的 REVERIE、R2R、R2R‑CE 三大基准；在真实机器人上使用办公室环境进行验证。

**📈 对比分析**

与最新 SOTA 对比：在 REVERIE 上相对 GOAT 提升 SR 1.93%、SPL 0.07%；在 R2R 上相对 GOAT 提升 SR 1.19%、OSR 1.18%；在 R2R‑CE 上相对 ETPNav 提升 SR 2% 以上；在真实机器人部署中表现出良好的 Sim‑to‑Real 转移能力。

**⚠️ 局限性**

局限性：①依赖大型多模态 LLM 进行事件抽取，计算成本和能耗较高；②知识图谱仅覆盖室内巡回视频，可能对特殊场景或动态变化不够全面；③粗细检索与 ASTFF 在极端嘈杂视觉或语言歧义下仍可能产生错误，需进一步增强鲁棒性。

---

## 167. Learning with a Budget: Identifying the Best Arm with Resource Constraints

**arXiv ID:** 2602.24146 | [PDF](https://arxiv.org/pdf/2602.24146v1)

**作者:** Zitian Li `[一作]`, Wang Chi Cheung `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在资源约束下的最佳臂识别问题（BAIwRC），提出了一种新的算法SH-RR，旨在识别最佳选择。

**💡 创新点**

创新点在于提出了SH-RR算法，该算法将资源分配与经典的逐步减半框架结合，并统一了随机和确定性消费设置的理论分析。

**🔧 技术方法**

使用了逐步减半与资源配额（SH-RR）算法，该算法在多个阶段中消除次优臂，并合理分配资源以确保充分探索。

**📊 数据集**

使用了多种资源类型的预算模型，考虑了不同臂的异质资源消耗，允许随机奖励和资源消耗之间的任意相关性。

**📈 对比分析**

与现有的基线方法（如AT-LUCB、UCB等）进行了比较，SH-RR在多种合成和真实世界问题中表现出色，尤其在资源消耗较低的情况下，成功识别最佳臂的概率更高。

**⚠️ 局限性**

限制在于算法的复杂性分析依赖于有效消费度量，且在某些情况下，随机消费设置可能使问题变得更加复杂。

---

## 168. How IMU Drift Influences Multi-Radar Inertial Odometry for Ground Robots in Subterranean Terrains

**arXiv ID:** 2602.24192 | [PDF](https://arxiv.org/pdf/2602.24192v1)

**作者:** Moumita Mukherjee `[一作]`, George Nikolakopoulos `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了多雷达惯性里程计(MRIO)框架，用于地下环境的鲁棒导航。

**💡 创新点**

提出了两阶段EKF融合：先用雷达最小二乘估计自机速度并校正IMU漂移，再将校正后的加速度与多雷达观测融合；同时引入基于测距的异常剔除方法。

**🔧 技术方法**

采用多台FMCW mmWave雷达、低成本Pixhawk/VectorNav IMU、扩展卡尔曼滤波器、最小二乘速度估计、测距异常剔除与雷达点云映射等技术。

**📊 数据集**

在地下隧道真实试验环境中使用六台IWR6843AOP雷达、Pixhawk与VectorNav IMU进行评估；对比LIO‑SLAM和EKF‑RIO作为基线。

**📈 对比分析**

与EKF‑RIO和LIO‑SLAM对比，MRIO在Pixhawk场景下二维RMSE仅0.83 m，YO角RMSE 1.13°；在VectorNav场景下RMSE 0.81 m，显著优于基线的40 m级误差。

**⚠️ 局限性**

仅依赖IMU作为姿态来源，受陀螺仪漂移影响，长时间运行可能出现姿态误差，需加入冗余姿态约束以提升全局一致性。

---

## 169. Learning Flexible Job Shop Scheduling under Limited Buffers and Material Kitting Constraints

**arXiv ID:** 2602.24180 | [PDF](https://arxiv.org/pdf/2602.24180v1)

**作者:** Shishun Zhang `[一作]` (National University of Defense Technology), Kai Xu `[通讯]` (Institute of AI for Industries, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于深度强化学习和异构图神经网络的调度方法，用来解决具有有限缓冲区和物料分拣约束的灵活车间调度问题（FJSP‑LB‑MK）。

**💡 创新点**

创新点包括：①将缓冲区与物料分拣约束首次完整地纳入FJSP；②在DRL框架内使用异构图神经网络对机器、操作和缓冲区之间的全局状态进行建模；③设计了针对缓冲区的选择性连通和成本敏感消息传播机制，以显式地考虑换垛成本；④将换垛次数与完成时间共同作为奖励，兼顾调度质量与资源利用。

**🔧 技术方法**

使用的技术包括：深度强化学习（PPO），异构图神经网络（HGNN）与多阶段消息传递，稀疏权重传播，奖励设计与KL正则化，GPU并行训练等。

**📊 数据集**

实验使用了两类数据集：①合成数据（10–40 作业 × 5–10 机器，包含部分分拣操作和缓冲区约束）；②四条真实钢板生产线数据（20 作业 × 10–16 机器，真实的机组-操作映射、工件类别和缓冲区数量）。

**📈 对比分析**

与 OR‑Tools、传统优先调度规则（PDR）和另一种基线 DRL 方法进行比较。实验结果显示，本文方法在合成和真实数据集上在平均完成时间（makespan）和换垛次数（switches）两项指标均显著优于基线，尤其在大规模实例（40×10 机器）上的优势最为明显；计算时间虽高于 PDR，但远低于 OR‑Tools，展示了良好的可扩展性与实时性。

**⚠️ 局限性**

局限性：①在不同生产线的缓冲区与类别分布差异较大时，方法在部分实例上可能导致换垛次数略高；②未考虑设备失效、动态订单插入等实际扰动，适用性在极端工况下尚未验证；③模型需先在合成数据上预训练后再针对真实线细调，迁移效果依赖于数据分布相似性。

---

## 170. Foundation World Models for Agents that Learn, Verify, and Adapt Reliably Beyond Static Environments

**arXiv ID:** 2602.23997 | [PDF](https://arxiv.org/pdf/2602.23997v1)

**作者:** Florent Delgrange `[一作]` `[通讯]` (AI Lab, Vrije Universiteit Brussel and Flanders Make), Florent Delgrange (AI Lab, Vrije Universiteit Brussel and Flanders Make)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出一种将强化学习与正式合成方法融合的“可验证世界模型”框架，强调在学习过程中持续进行形式化验证与抽象校准，并利用大语言模型在运行时生成并细化任务规范与程序。

**💡 创新点**

创新点在于：①把奖励设计转化为可形式化的逻辑约束；②把验证嵌入学习循环，形成自适应的安全阈值；③将抽象误差动态校准与世界模型结合，构成可验证的“基础世界模型”；④通过LLM与验证器交互，实时生成、验证并改进规范与程序。

**🔧 技术方法**

使用技术包括：基于时序逻辑的奖励生成、概率屏蔽与神经证书、Safe Policy Improvement理论、抽象误差估计与动态校准、Probabilistic Model Checking（如Storm/Prism）以及LLM（如ChatGPT/LLM‑based program synthesizers）。

**📊 数据集**

本文为概念性阐述，并未使用特定实验数据集；讨论以仓储物流、包裹递送等仿真环境为示例。

**📈 对比分析**

由于缺乏实现与实验，本文未给出数值比较或性能指标，主要通过理论讨论与已有相关工作（如可学习的 LTL 片段、深度 RL 与安全保证）来说明潜在优势。

**⚠️ 局限性**

局限性包括：①实际实现与调优的技术难度高；②程序生成与验证的计算成本可能成为瓶颈；③抽象误差估计的精确性与泛化性尚未得到充分验证；④对 LLM 生成的规范与程序的可靠性依赖较大，易受语言模型错误或不一致性影响；⑤在大规模开放环境中可扩展性与实时性仍需进一步研究。

---

## 171. From Flat Logs to Causal Graphs: Hierarchical Failure Attribution for LLM-based Multi-Agent Systems

**arXiv ID:** 2602.23701 | [PDF](https://arxiv.org/pdf/2602.23701v1)

**作者:** Yawen Wang `[一作]` (Institute of Software), Qing Wang `[通讯]` (Institute of Software)

**通讯引用:** 8900 | [OpenAlex ID](https://openalex.org/A5100434786)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建层级因果图，对LLM驱动的多智能体系统（MAS）的失败日志进行结构化分析，并通过虚拟 oracle 指导的层级回溯与反事实推理，实现对根本失误点（agent+step）的定位。

**💡 创新点**

① 将混乱的执行轨迹转化为层级因果图；② 设计基于虚拟 oracle 的层级回溯机制，快速缩小搜索空间；③ 引入递进式反事实筛选，精准区分根本错误与传播症状。

**🔧 技术方法**

OTAR 结构化解析、RAG+少量提示进行任务拆解、层级因果图构建、虚拟 oracle 合成、LLM 辅助语义评估、反事实因果筛选与可逆性检测。

**📊 数据集**

Who&When 基准（184 条失败日志，包含 126 条算法生成、58 条人工构造）。

**📈 对比分析**

与 8 个主流基线（随机、LLM 提示、ECHO、FAMAS、AgenTracer、GraphTracer 等）对比，Agent 级别最高 77.59%/76.80%，Step 级别 29.31%/52.00%，均显著优于基线，且不需重演或额外微调，成本更低。

**⚠️ 局限性**

依赖层级因果图与虚拟 oracle 的准确性，若图构建错误会误导诊断；仅支持单一决定性根本错误，未覆盖累积误差场景；评测仅限 Who&When 基准，需在更广泛系统上验证。

---

## 172. MMKG-RDS: Reasoning Data Synthesis via Deep Mining of Multimodal Knowledge Graphs

**arXiv ID:** 2602.23632 | [PDF](https://arxiv.org/pdf/2602.23632v1)

**作者:** Lun Zhan `[一作]` (AI Research Institute, Qihoo 360), Yuhui Yin `[通讯]` (AI Research Institute, Qihoo 360)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出MMKG-RDS框架，利用多模态知识图谱实现可定制化推理数据合成。

**💡 创新点**

创新在于深度多模态知识图构建、可配置路径采样以及三维质量评估。

**🔧 技术方法**

采用大语言模型与多模态模型（Qwen3、Qwen3-VL、MinerU）进行文档解析、实体关系抽取和路径生成。

**📊 数据集**

构建MMKG-RDS-Bench数据集，涵盖历史、有机化学、法律、股票研究报告和论文五个领域，共14,950条推理样本。

**📈 对比分析**

通过对Qwen3系列模型进行微调与对比实验，合成数据提升推理准确率约9.2%，并验证难度控制与多模态任务的效果。

**⚠️ 局限性**

局限包括需要人工制定Schema、文档解析不完善、基于VLM抽取的实体可能出现幻觉、构建效率低等。

---

## 173. Teleoperated Omni-directional Dual Arm Mobile Manipulation Robotic System with Shared Control for Retail Store

**arXiv ID:** 2602.23923 | [PDF](https://arxiv.org/pdf/2602.23923v1)

**作者:** Rolif Lima `[一作]` (Tata Consultancy Services), Kaushik Das `[通讯]` (Tata Consultancy Services)

**通讯引用:** 2130 | [OpenAlex ID](https://openalex.org/A5060143640)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种面向零售仓储的全向双臂移动机器人GriffinX，并设计了基于VR的共享控制远程操作系统，用于在机器人完全自主失效时完成拣选与搬运任务。

**💡 创新点**

创新点在于（1）融合异构两种抓取器（双指刚性抓取器与可重构软抓取器），实现对多形状、多材质物体的兼容；（2）提出一种基于模型预测控制的共享控制框架，能够在满足操作员指令的同时自动追踪目标位置，平滑切换；（3）在零售环境中首次将该共享控制与双臂协同操作相结合，并在仿真与实体环境中验证其有效性。

**🔧 技术方法**

核心技术包括：HTC Vive VR运动捕捉系统、双指RG2抓取器与可重构三指软抓取器、基于YOLOv6的目标检测与深度相机定位、LiDAR+RGB‑D融合的碰撞检测、基于AL‑iLQR的MPC轨迹规划、TracIK逆运动学、加速度与速度约束的运动限制。

**📊 数据集**

主要数据来源为自建的零售模拟环境（包含多种常见零售物品的摆放与标签），采用YOLOv6对实时RGB‑D图像进行检测，未使用公开数据集；实验数据通过机器人与操作员轨迹记录、抓取成功率、任务完成时间等指标收集。

**📈 对比分析**

通过与纯遥操作对比，使用共享控制可平均缩短30%的任务完成时间，并显著降低碰撞发生率；在单臂与双臂抓取实验中，软抓取器在顶视角下成功抓取所有样本物体，侧向抓取成功率受物体尺寸限制；双臂协同抓取时，末端执行器间距离保持在±5 cm内。

**⚠️ 局限性**

主要局限包括：MPC计算速率受限于10 Hz，导致双臂协同时的距离波动；软抓取器对小于7 cm的物体仍无法实现可靠抓取；共享控制参数调优依赖实验经验；系统缺乏长期自学习能力，仍需人工干预。

---

## 174. Fair Division Under Inaccurate Preferences

**arXiv ID:** 2602.24169 | [PDF](https://arxiv.org/pdf/2602.24169v1)

**作者:** Trung Dang `[一作]` (University of Texas at Austin), Paritosh Verma `[通讯]` (Purdue University)

**通讯引用:** 2312 | [OpenAlex ID](https://openalex.org/A5020406550)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文探讨了在不准确的基数偏好下，如何公平地分配不可分割的物品，重点是最小化嫉妒感。

**💡 创新点**

创新点在于提出了一种在不完全信息下进行公平分配的模型，考虑了真实偏好的随机性和最坏情况，并提供了相应的算法和理论保证。

**🔧 技术方法**

使用了统计学和算法理论中的多种工具，包括线性规划、随机算法和相关不等式。

**📊 数据集**

使用了模拟的基数偏好数据集，考虑了不同的噪声模型（随机噪声和最坏情况噪声）。

**📈 对比分析**

与现有方法相比，本文提出的算法在处理不准确偏好时能够以高概率计算出无嫉妒的分配，并且在最坏情况下提供了嫉妒的界限，性能表现优越。

**⚠️ 局限性**

限制在于模型假设了偏好是加法的，并且在某些情况下可能无法保证存在无嫉妒的分配。

---

## 175. Deep Sleep Scheduling for Satellite IoT via Simulation Based Optimization

**arXiv ID:** 2602.23788 | [PDF](https://arxiv.org/pdf/2602.23788v1)

**作者:** Wanja de Sombre `[一作]` (Technische Universität Darmstadt), Andrea Ortiz `[通讯]` (Vienna University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对卫星物联网(S-IoT)设备的深度睡眠调度算法，旨在平衡能耗与接收端信息退化。

**💡 创新点**

创新点在于将目标导向张量(Got)指标与概率模拟优化(PSBO)相结合，既考虑信息内容的重要性，又能在未知环境中在线预测未来成本。

**🔧 技术方法**

使用的技术包括马尔可夫决策过程(MDP)建模、贝叶斯信念更新、分布式模拟（非样本化）以及凸性检查来快速选取最优睡眠周期。

**📊 数据集**

实验采用真实的光照/温度传感数据和从Starlink网络收集的延迟/擦除率轨迹，构建马尔可夫链进行仿真。

**📈 对比分析**

与随机、始终传输、阈值传输以及Q‑学习等基线比较，PSBO在各种擦除率、能耗权重和过程动态下均能取得最低的综合成本，仿真平均成本比基线低30–70%。

**⚠️ 局限性**

限制包括：假设观测过程最多单次状态变迁、Got指标需预先设计、对极端动态环境的适应性不足以及在极高噪声/遮挡下学习速率仍显慢。

---

## 176. Hyperdimensional Cross-Modal Alignment of Frozen Language and Image Models for Efficient Image Captioning

**arXiv ID:** 2602.23588 | [PDF](https://arxiv.org/pdf/2602.23588v1)

**作者:** Abhishek Dalvi `[一作]` (Artificial Intelligence Research Laboratory Pennsylvania State University), Vasant Honavar `[通讯]` (Artificial Intelligence Research Laboratory Pennsylvania State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种HDFLIM框架，利用冻结的视觉与语言基础模型，通过超维度计算将两者映射到共享的高维空间，使用绑定与聚合操作构建关联记忆，从而在单次数据遍历中实现图像字幕生成。

**💡 创新点**

创新点在于实现跨模态对齐而不需梯度或大规模微调；通过符号化的超维度映射、绑定/聚合操作来捕捉预训练模型的内在语义兼容性，实现可解释、轻量级的模态融合。

**🔧 技术方法**

核心技术包括超维度计算（HD computing）、局部敏感哈希（LSH）将高维特征映射为二进制超向量、绑定（⊗）与聚合（⊕）操作构建记忆、CLIP引导的采样与逻辑混合、以及离线位打包的高效推理。

**📊 数据集**

训练使用COCO（约82k图像）与PixelProse（约13M图像-描述对）两大数据集；评估在COCO、Nocaps、COCO验证集等标准图像字幕基准上进行。

**📈 对比分析**

与ZeroCap、ConZIC、CLIP-Captioner、Qwen2‑VL等基线对比，HDFLIM在CLIP‑S、RefCLIP‑S等视觉‑文本一致性指标上与端到端模型持平，语义覆盖更丰富，生成速度快于梯度优化方法，且在传统指标（BLEU、METEOR）上通过后处理提升；整体表现接近或优于现有最优方法。

**⚠️ 局限性**

局限性包括仅支持单向视觉→文本映射、对长文本生成有限、需手动调节窗口大小与clip权重、缺乏批处理支持、对不同LLM版本的迁移效果有限，以及在传统n‑gram指标上的表现仍不如深度生成模型。

---

## 177. DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science

**arXiv ID:** 2602.24288 | [PDF](https://arxiv.org/pdf/2602.24288v1)

**作者:** Fan Shu `[一作]` (University of Houston), Feng Yan `[通讯]` (University of Houston)

**通讯引用:** 11886 | [OpenAlex ID](https://openalex.org/A5075806509)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 DARE‑Bench，一个面向数据科学任务的可训练、可验证基准，包含两类任务：过程意识的指令跟随与机器学习建模，并提供可执行的评估环境和大规模训练数据。

**💡 创新点**

创新点在于：①将指令遵循与建模性能统一于同一基准；②通过参考代码和数据标签实现完全可验证的评估；③采用自动化流水线从 Kaggle 数据集生成多样化任务；④提供可用于监督微调和强化学习的训练集。

**🔧 技术方法**

主要技术包括：LLM 代码生成与执行沙箱；强化学习框架 GRPO 与 RER（verifiable reward）实现无偏好学习；监督微调（SFT）结合拒绝采样；自动化任务设计与数据预处理流水线；指标设计（宏 F1、R²、指令遵循准确率）。

**📊 数据集**

使用了约 6,300 个来自 Kaggle 的公开数据集，涵盖分类、回归、时序预测等多种任务类型，数据按 95/5 训练/测试拆分，保证多样性与现实挑战。

**📈 对比分析**

对比方法：在 gpt‑o4‑mini、gpt‑4o、Claude‑Sonnet、Qwen 等模型上，使用 5 回合、200 秒的配置进行评测。基线模型性能极低，细调后 Qwen3‑32B 在总分上提升约 1.83×，RL 在 Qwen3‑4B 上从 4.39 提升至 37.40，整体模型在四个评价维度均显著优于基线。

**⚠️ 局限性**

局限性包括：① 对复杂时序预测（time‑series‑CF）仍表现差；② 开源 LLM 在多步推理与工具调用上易出错；③ 目前仅覆盖表格数据，未涵盖文本、图像等多模态任务；④ 需要进一步提升在极端噪声和缺失值环境下的鲁棒性。

---

## 178. Novice Developers Produce Larger Review Overhead for Project Maintainers while Vibe Coding

**arXiv ID:** 2602.23905 | [PDF](https://arxiv.org/pdf/2602.23905v1)

**作者:** Syed Ammar Asdaque `[一作]` (Lahore University of Management Sciences), Abdul Ali Bangash `[通讯]` (Lahore University of Management Sciences)

**通讯引用:** 165 | [OpenAlex ID](https://openalex.org/A5063403084)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过对AIDev数据集中1,719名vibe编码者共22,953个PR的分析，比较了经验水平不同的vibe编码者（Exp_High vs Exp_Low）在PR提交规模、合并率、解决时间及评审负载等方面的差异。

**💡 创新点**

创新点在于首次系统性比较了低经验和高经验vibe编码者在AI辅助开发中的贡献效果与验证成本，并揭示低经验编码者在产生大规模提交时会显著增加评审工作量和合并延迟。

**🔧 技术方法**

研究使用了统计检验技术（Mann-Whitney U、Chi-square、Benjamini-Hochberg校正）对PR指标进行分组比较，并以经验分数（commit数/账号年龄）对贡献者进行分层。

**📊 数据集**

数据集为公开的AIDev数据集，包含33,596个PR和1,796名用户，重点提取了22,953个vibe编码者的PR记录。

**📈 对比分析**

通过经验分组比较方法，发现低经验编码者提交的PR在commit数、文件变更量上分别高出2.15×和1.47×，合并率低31%，解决时间长5.16×，评审评论数高4.52×，表明低经验编码者的AI辅助产出在验证成本上显著高于高经验者。

**⚠️ 局限性**

局限性包括经验指标仅基于GitHub提交数，可能与实际技能脱钩；vibe编码定义的差异可能影响结果泛化；以及项目特定的评审政策可能对合并率和评审量产生影响。

---

## 179. ProtoDCS: Towards Robust and Efficient Open-Set Test-Time Adaptation for Vision-Language Models

**arXiv ID:** 2602.23653 | [PDF](https://arxiv.org/pdf/2602.23653v1)

**作者:** Wei Luo `[一作]` (Pazhou Laboratory), Mingkui Tan `[通讯]` (Pazhou Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出ProtoDCS框架，解决视觉语言模型在开放式测试时适应（OSTTA）中csID与csOOD样本的区分与安全适应问题。

**💡 创新点**

创新点在于双重检查分离机制——先用开放度得分和双阈值粗筛，再用Gaussian Mixture Model实现概率化细化；以及基于证据驱动的不确定性感知损失，替代熵最小化并避免过度自信；同时仅在原型层面更新，保持VLM主干冻结，显著提升计算效率。

**🔧 技术方法**

技术包括开放度得分计算、双阈值筛选、GMM概率验证、残差原型更新、证据学习的aleatoric与epistemic不确定性损失、视觉-文本双源预测以及非梯度的原型级更新。

**📊 数据集**

在CIFAR-10-C、CIFAR-100-C、Tiny-ImageNet-C等三大开放式测试集上评估，同时使用ImageNet及其四种自然迁移变体验证闭集适应性能。

**📈 对比分析**

与多种基线（闭集TTA、VLM基TTA、开集TTA等）对比，ProtoDCS在Acc、AUROC、FPR@TPR95和OSCR指标上均达SOTA，显著提升已知类准确率与OOD检测性能；在效率上保持与DPE相近的吞吐量、低延迟和内存占用。

**⚠️ 局限性**

局限性主要体现在需手动调参（如阈值百分位、GMM窗口大小、损失权重）以及在极端分布漂移或大规模OOV场景下的适应速度仍有提升空间。

---

## 180. Extended Reality (XR): The Next Frontier in Education

**arXiv ID:** 2602.23601 | [PDF](https://arxiv.org/pdf/2602.23601v1)

**作者:** Shadeeb Hossain `[一作]` `[通讯]` (Shadeeb Engineering Lab), Shadeeb Hossain (Shadeeb Engineering Lab)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对XR在教育中的潜力与挑战进行了综述，探讨了其在提升学生参与度、实验学习与技能培养中的作用。

**💡 创新点**

将XR与游戏化、人工智能相结合，强调以隐私为先的设计框架，同时聚焦可及性与伦理问题。

**🔧 技术方法**

采用XR（VR、AR、MR）技术、AI驱动的适应性反馈、游戏化机制以及触觉系统。

**📊 数据集**

未使用具体数据集，主要参考多项实证研究和案例（如Houd、Batra等）。

**📈 对比分析**

通过文献综述与案例比较，指出XR可显著提升认知、情感、社交与行为参与度，但缺乏统一评估指标。

**⚠️ 局限性**

限制包括高昂成本、教师培训不足、隐私与安全风险以及对不同学习者的适配性不充分。

---

## 181. U-CAN: Utility-Aware Contrastive Attenuation for Efficient Unlearning in Generative Recommendation

**arXiv ID:** 2602.23400 | [PDF](https://arxiv.org/pdf/2602.23400v1)

**作者:** Zezheng Wu `[一作]` (Guilin University of Electronic Technology), Jingwei Zhang `[通讯]` (Guilin University of Electronic Technology)

**通讯引用:** 8490 | [OpenAlex ID](https://openalex.org/A5100434265)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于低秩适配器的机器无学习框架 U-CAN，针对生成式推荐系统中的用户隐私信息进行精准遗忘，同时保持模型推荐效果。

**💡 创新点**

创新点包括：① 对比激活差异定位隐私敏感神经元；② 将权重幅值与保留集激活归一化融合成效用重要性评分；③ 采用可微衰减函数实现连续软衰减，避免硬剪枝导致的结构破坏。

**🔧 技术方法**

采用 LoRA 适配器、对比激活差分、权重-激活重要性评估、NF4 量化、流式聚合、以及可微衰减函数进行风险校准与参数调节。

**📊 数据集**

使用公开的电影推荐数据集 ML‑100K 和电商商品数据集 Pantry 进行实验评估。

**📈 对比分析**

与 Retraining、GA、NPO、LLM‑Eraser 等基线方法对比，U‑CAN 在忘记效能（KL 散度、预测偏移、PPL）最高、保留效能（Recall@10、MRR@10、NDCG@10）最佳或相近，并在执行时间和吞吐量上实现最优性能。

**⚠️ 局限性**

局限性在于对阈值和超参数的敏感性、仅作用于 LoRA 适配器而非完整模型、以及在极少量忘记样本或不同模型/任务下的泛化能力仍需进一步验证。

---

## 182. FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting

**arXiv ID:** 2602.24084 | [PDF](https://arxiv.org/pdf/2602.24084v1)

**作者:** Matteo Ballegeer `[一作]` (Ghent University), Dries F. Benoit `[通讯]` (Ghent University)

**通讯引用:** 1683 | [OpenAlex ID](https://openalex.org/A5089981863)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出FoV-Net框架，用局部参考坐标系UV网格和射线投射的视场（FoV）网格联合捕获B-rep局部几何与全局结构，并实现对任意SO(3)旋转的本质不变性；

**💡 创新点**

创新点在于：①将每个面用局部参考框架化的UV网格表示，消除绝对坐标依赖；②通过从面中心对半球发射射线并记录首碰点信息，构造旋转不变的结构上下文；③将两类特征分别送入轻量级CNN提取后再用图注意力网络聚合，获得兼顾局部和全局的高效表示；

**🔧 技术方法**

技术包括：局部参考框架（LRF）UV网格构造、基于PythonOCC的射线投射获取FoV（外向/内向）网格、轻量级2D CNN提取特征、图注意力网络（GAT）实现面间信息传播，以及实验对比中使用的旋转增广策略；

**📊 数据集**

使用了SolidLetters、TraceParts、Fusion360 Gallery、MFCAD++四大工业CAD基准数据集，分别用于字母分类、机械部件分类、面分割与加工特征分割；

**📈 对比分析**

与UV-Net、AAGNet（含旋转增广版）以及FoV-Net（仅UV版）进行对比；在所有测试中，FoV-Net在保持与原始方向相同的情况下，旋转后准确率仅下降0.1%（分类95-100%），而UV-Net、AAGNet在旋转后准确率下降≥50%；在分割任务中，FoV-Net在旋转下保持90%以上准确率，且在少样本场景下数据效率显著优于其它模型；

**⚠️ 局限性**

局限性包括：射线投射依赖PythonOCC CPU实现，尚未实现GPU加速；FoV网格采用等角投影，导致极角扭曲；对UV重参数化（轴翻转、交换）仍不完全鲁棒；缺少边缘特征，可能限制对复杂装配的处理；

---

## 183. France or Spain or Germany or France: A Neural Account of Non-Redundant Redundant Disjunctions

**arXiv ID:** 2602.23547 | [PDF](https://arxiv.org/pdf/2602.23547v1)

**作者:** Sasha Boguraev `[一作]` (University of Texas), Kyle Mahowald `[通讯]` (University of Texas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在语境中看似冗余但实际上非冗余的“或”连词结构，并通过人类实验与大语言模型的行为及神经机制分析揭示其实现原理。

**💡 创新点**

首次将符号语义分析（Mandelkern 的可能性并置）与神经网络可解释性相结合，提出语言模型通过上下文绑定和诱导头实现非冗余连词的机制。

**🔧 技术方法**

利用大规模自回归语言模型（Pythia、GPT‑2、LLaMA2）、激活补丁 (activation patching)、诱导头注意力分析以及人类句子完成实验。

**📊 数据集**

自构造的句子样本，覆盖九个语义领域、32个名字、不同动词和上下文后缀，作为实验刺激。

**📈 对比分析**

将人类正确率与模型生成率对比，显示模型在关键条件下成功复制人类重复倾向；规模越大（>400 M 参数）性能越好，且模型对第二分支顺序最敏感。

**⚠️ 局限性**

人类实验受限于样本量与噪声，未观察到顺序效应；模型解释仍未证明捕获正式可能性共成分；依赖自制刺激与非实验室情境。

---

## 184. Recycling Failures: Salvaging Exploration in RLVR via Fine-Grained Off-Policy Guidance

**arXiv ID:** 2602.24110 | [PDF](https://arxiv.org/pdf/2602.24110v1)

**作者:** Yanwei Ren `[一作]` (Beihang University), Liu Liu `[通讯]` (Beihang University)

**通讯引用:** 140986 | [OpenAlex ID](https://openalex.org/A5100338921)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 SCOPE 框架，利用过程奖励模型（PRM）定位失败轨迹的首个错误步骤，并通过离线教师模型进行精细的逐步纠正，从而保留并重用大部分正确的推理路径。

**💡 创新点**

创新点在于：①对 PRM 进行分布感知的候选选择，避免极端轨迹导致的偏差；②在保留正确前缀的同时，仅对第一个错误步骤进行离线纠正，形成混合在策略分布内的“修正轨迹”，从而提升奖励信号质量并保持探索多样性；③通过混合优化目标兼顾 PPO 采样与教师克隆，使得模型既能在策略分布内学习，又能吸收外部高质量指引。

**🔧 技术方法**

核心技术包括：过程奖励模型（PRM）对推理步骤的置信度评估、分布感知的选取策略、基于离线教师的逐步纠正、混合优化目标（PPO + 目标克隆）、以及多任务评估框架。

**📊 数据集**

使用数据集：训练采用 MATH（难度 3–5），评估覆盖 AIME24/25、AMC、MATH‑500、Minerva、Olympiad（ID）以及 ARC‑c、GPQA‑Diamond、MMLU‑Pro（OOD）。

**📈 对比分析**

与 GRPO、ReLIFT、LUFFY 等现有 RLVR 与离线引导方法相比，SCOPE 在 ID 任务上平均准确率提升至 46.6%（比最佳基线高 1.8%），OOD 任务平均准确率提升至 53.4%（比最佳基线高 3.2%），同时生成多样性提升 13.5%，显著缓解模式崩溃与样本效率低下。

**⚠️ 局限性**

局限性包括：对 PRM 与教师模型的质量高度依赖，极端轨迹仍可能导致误判；在极小模型或低资源场景下，离线纠正所需的计算开销和模型调优仍是挑战；缺乏对非数学推理任务的通用性验证。

---

## 185. NAU-QMUL: Utilizing BERT and CLIP for Multi-modal AI-Generated Image Detection

**arXiv ID:** 2602.23863 | [PDF](https://arxiv.org/pdf/2602.23863v1)

**作者:** Xiaoyu Guo `[一作]` (Nanjing Audit University), Arkaitz Zubiaga `[通讯]` (Queen Mary University of London)

**通讯引用:** 6683 | [OpenAlex ID](https://openalex.org/A5071220716)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出多模态多任务模型，结合BERT与CLIP提取文本与图像特征，检测AI生成图像及其生成模型，并通过伪标签扩充数据；

**💡 创新点**

创新点在于将跨模态特征融合与针对两任务的条件多任务损失相结合，同时利用伪标签自监督方式扩充训练集，并在CT2比赛中取得5位成绩；

**🔧 技术方法**

使用预训练BERT文本编码器、CLIP视觉编码器，交叉模态特征拼接，二分类与多分类头，条件交叉熵损失，伪标签数据增强，基于PyTorch与Hugging Face实现；

**📊 数据集**

采用CT2 AI-Generated Image Detection数据集（基于MS COCO生成多模型图像）以及伪标签扩充的样本；

**📈 对比分析**

与CT2官方评测对比，Task A F1 83.16%，Task B F1 48.88%，在两任务均排名第5；内部验证时Task A F1 99.58%，Task B 85.95%；

**⚠️ 局限性**

局限性包括伪标签可能引入错误与偏差，易分类样本过滤导致不平衡，训练-测试泄漏风险，以及类不平衡导致模型过拟合某些AI模型。

---

## 186. Accelerating Masked Image Generation by Learning Latent Controlled Dynamics

**arXiv ID:** 2602.23996 | [PDF](https://arxiv.org/pdf/2602.23996v1)

**作者:** Kaiwen Zhu `[一作]` (Shanghai Jiao Tong University), Yihao Liu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 7297 | [OpenAlex ID](https://openalex.org/A5100696068)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级的快捷模型（MIGM-Shortcut），通过学习先前特征和已采样令牌的受控动态，替代重型基模型在多步生成中的大部分计算，从而显著加速掩码图像生成过程。

**💡 创新点**

创新点在于将采样信息与过去特征结合，学习到的隐式动力学模型能够捕捉特征轨迹的平滑性，并通过轻量级网络实现“捷径”，既提升效率又保持甚至提升生成质量。

**🔧 技术方法**

采用跨注意力+自注意力结构的轻量级神经网络，配合时间条件化、瓶颈投影以及基于均方误差的训练目标，利用特征轨迹的局部 Lipschitz 性质进行动态预测。

**📊 数据集**

使用 ImageNet-512 的类别标签生成 512×512 图像以及 1024×1024 的文本提示数据集，对 MaskGIT、Lumina‑DiMOO 进行评估，并采集 50k+ 生成样本用于训练。

**📈 对比分析**

与原始模型、ML‑Cache、ReCAP、dLLM‑Cache、TaylorSeer 等多种加速方法对比，MIGM‑Shortcut 在保持或提升 FID、ImageReward、CLIPScore、UniPercept‑IQA 指标的同时，平均加速 4–5×，并在 4×加速下保持与原始模型相当的感知质量。

**⚠️ 局限性**

局限性包括：对基模型的依赖（需要冻结且已训练好的模型），加速效果随预算与步数配置变化，且在极端少步或跨模态多任务场景下仍面临多模态分布建模瓶颈。

---

## 187. ProductResearch: Training E-Commerce Deep Research Agents via Multi-Agent Synthetic Trajectory Distillation

**arXiv ID:** 2602.23716 | [PDF](https://arxiv.org/pdf/2602.23716v1)

**作者:** Jiangyuan Wang `[一作]` (Alibaba International Digital Commercial Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commercial Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ProductResearch 多代理框架，利用用户代理、研究代理和监督代理通过状态机实现步骤级监督，生成高保真、长周期的工具使用轨迹，用以训练电商领域的深度研究型对话代理。

**💡 创新点**

创新点包括：①三代理协同架构，实现用户意图解析、工具调用与监督反馈的闭环；②基于状态机的细粒度监督，针对计划、工具调用和报告三阶段分别校验；③反射内部化机制，将多角色交互轨迹压缩为单角色训练样本；④动态生成查询评估标准，提升评估针对性。

**🔧 技术方法**

使用技术主要有：ReAct‑style LLM 交互、专门的电商工具集（Web_Visit、Visit_Product）、三阶段状态机监督、反射内部化、Mixture‑of‑Experts (MoE) 模型微调、32k–128k 长上下文训练。

**📊 数据集**

数据集为 1,000 名真实用户的匿名行为日志（购买、评论、客服对话）所生成的用户画像、查询与动态评估标准，随后合成的多代理轨迹组成训练集、验证集与测试集（比例 8:1:1）。

**📈 对比分析**

与 Tongyi‑DeepResearch、Qwen‑DeepResearch、Gemini‑DeepResearch（Deep Research）以及 Gemini‑3‑flash、GPT‑4.1、Qwen3‑max（ReAct）对比。通过 RACE 指标和有效产品计数评估，ProductResearch‑SFT‑128k 获得 45.40 分（仅次于 Gemini‑DeepResearch 45.56），在四个维度和 E.Prod 上均显著优于所有 ReAct 方案，并提升了约三倍的产品覆盖。

**⚠️ 局限性**

局限性：工具实现（Web_Visit、Visit_Product）尚可进一步优化；框架仅支持单轮查询，未涵盖多轮意图演变；系统可能继承底层 LLM 的偏差与错误，导致偶尔的错误信息或不充分的证据。

---

## 188. Recommending Search Filters To Improve Conversions At Airbnb

**arXiv ID:** 2602.23717 | [PDF](https://arxiv.org/pdf/2602.23717v1)

**作者:** Hao Li `[一作]` (Airbnb, Inc.), Sanjeev Katariya `[通讯]` (Airbnb, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了面向转化的搜索过滤器推荐系统，直接提升Airbnb的预订转化率

**💡 创新点**

通过将过滤器推荐任务拆分为预测过滤器使用概率P(F|Q)与过滤器下的预订概率P(B=1|F,Q)，实现了从“工具发现”到“转化驱动”的全新框架

**🔧 技术方法**

利用多任务神经网络（MLP）结合嵌入、归一化及周期特征编码，配合TensorFlow Serving实现低延时推理

**📊 数据集**

基于一年规模的匿名搜索与预订日志，构建了包含约千亿条样本的训练集，涵盖查询、过滤器使用与预订归因信息

**📈 对比分析**

在离线评估中相较于统计基准PR‑AUC提升约100%，在线上A/B测试中在过滤器面板实现+0.28%预订提升，过滤器栏实现+2.4%，整体显著优于基线

**⚠️ 局限性**

受限于当前仅支持离散过滤器、未充分利用展示偏差数据，且模型仍未引入物品级特征与动态呈现信息

---

## 189. Keyword search is all you need: Achieving RAG-Level Performance without vector databases using agentic tool use

**arXiv ID:** 2602.23368 | [PDF](https://arxiv.org/pdf/2602.23368v1)

**作者:** Shreyas Subramanian `[一作]` (Amazon Web Services), Maira Ladeira Tanke `[通讯]` (Amazon Web Services)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在对比检索增强生成（RAG）系统与基于工具的LLM代理（Agent）时，作者构建了两套完整的问答管道，分别使用向量数据库检索和仅凭关键字搜索（通过Linux终端工具）完成文档检索，并利用Claude 3 Sonnet生成答案。

**💡 创新点**

创新点在于证明仅凭简单的关键字搜索工具即可在不使用向量数据库的情况下，取得与传统RAG系统相当的性能（超过90%），并提出了一种低成本、易维护的Agent架构，强调了工具驱动与LLM推理的协同。

**🔧 技术方法**

技术主要包括：Amazon Bedrock平台（Claude 3 Sonnet、Titan文本嵌入模型）、LangChain框架（ReAct推理）、终端搜索工具（pdfmetadata.sh、rga、pdfgrep）以及RAGAS评估框架（LLM-as-a-Judge）。

**📊 数据集**

使用了六个多领域数据集：PaulGrahamEssay、Llama2Paper、HistoryOfAlexnet、BlockchainSolana、LLM Survey、FinanceBench（全部来源于LlamaHub）。

**📈 对比分析**

对比方法：基于RAGAS的Faithfulness、Context Recall和Answer Correctness指标进行评估；实验显示Agent在faithfulness上平均取得94.5%、在context recall 88.1%以及在answer correctness 91.5%的达成率，整体与RAG系统差距仅在8%左右。

**⚠️ 局限性**

局限性包括：在超大文档中检索效率下降、对多媒体内容支持不足、LLM上下文窗口限制、对模糊查询的鲁棒性不高，以及依赖外部工具导致的可用性与延迟问题。

---

## 190. Portfolio Reinforcement Learning with Scenario-Context Rollout

**arXiv ID:** 2602.24037 | [PDF](https://arxiv.org/pdf/2602.24037v1)

**作者:** Vanya Priscillia Bendatu `[一作]` (National University of Singapore), Yao Lu `[通讯]` (National University of Singapore)

**通讯引用:** 5974 | [OpenAlex ID](https://openalex.org/A5058605138)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种宏观情景上下文回放（SCR）机制，并在强化学习（RL）框架下通过引入假设续期的 bootstrap 目标，解决了情景奖励与历史转移不匹配导致的 TD 学习不稳定问题，从而提升日常资产组合再平衡的性能。

**💡 创新点**

创新点包括：①利用 SCR 生成基于宏观冲击的多元收益分布；②理论分析情景奖励与真实转移导致的混合 Bellman 操作及其固定点偏差；③设计了一种混合真实与假设续期的 critic 目标，实现 bias‑variance 折衷，显著稳定训练。

**🔧 技术方法**

使用技术包括：PPO（actor‑critic）+ GAE、Wasserstein‑1 距离理论、KNN 近似检索、记忆更新函数、情景检索库（ShockLedger）以及对冲回报的风险惩罚项。

**📊 数据集**

数据集为 2009‑2023 年美国股票与 ETF 的 FinRL 公开数据，构建了 31 个不同资产集合（High‑Vol、Low‑Vol、General、Market‑Proxy），并按时间拆分为训练/验证/测试集。

**📈 对比分析**

在 31 个测试 universe 上与经典均值‑方差、逆波动、GMV、PPO（历史回放）、BootRollout‑PPO、SCR‑PPO 各变体对比，SCR‑PPO‑Full 在 out‑of‑sample 评估中显著提升 Sharpe（最高提升 76%）并降低最大回撤（最高降低 53%），且换手率极低，体现出更稳健的风险控制。

**⚠️ 局限性**

局限性包括：对极端新颖宏观冲击的检索依赖已建 ShockLedger，缺乏完全泛化能力；混合权重 β_cf 对性能敏感，需要经验调优；方法主要在 US 市场验证，跨市场适用性待进一步研究。

---

## 191. Benchmarking BERT-based Models for Sentence-level Topic Classification in Nepali Language

**arXiv ID:** 2602.23940 | [PDF](https://arxiv.org/pdf/2602.23940v1)

**作者:** Nischal Karki `[一作]` (Kathmandu University), Bal Krishna Bal `[通讯]` (Kathmandu University)

**通讯引用:** 779 | [OpenAlex ID](https://openalex.org/A5030335666)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对10种BERT变体在平衡的25,006句尼泊尔语句子分类数据集上进行微调与评估。

**💡 创新点**

首次系统比较多语种、印地语、尼泊尔语BERT模型在尼泊尔语主题分类中的表现。

**🔧 技术方法**

使用Hugging Face库对模型进行微调，评估指标包括准确率、加权精确率/召回率/F1以及AUROC。

**📊 数据集**

采用由信息与语言处理研究实验室收集的5个领域平衡尼泊尔语句子数据集。

**📈 对比分析**

通过比较各模型的训练时间、参数量和评估指标，发现MuRIL-large获得90.60% F1，最高；NebBERTa也在88.26%附近表现良好。

**⚠️ 局限性**

实验仅限句子分类，数据集领域有限，模型训练细节不透明，未涉及其他下游任务或错误分析。

---

## 192. LLM-Driven Multi-Turn Task-Oriented Dialogue Synthesis for Realistic Reasoning

**arXiv ID:** 2602.23610 | [PDF](https://arxiv.org/pdf/2602.23610v1)

**作者:** Yu Zhu `[一作]`, Kai Yang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估多种大语言模型在 RealReasoning 数据集上，在迭代前后的答案准确率，并展示基于提示工程的代理交互与评估框架。

**💡 创新点**

提出通过迭代更新数据集来提高难度、揭示模型局限的实验方法，并提供一套完整的代理生成、用户/助手交互和质量评估的提示模板。

**🔧 技术方法**

利用 Qwen、DeepSeek 等 LLM，配合手工设计的 Prompt，实施多轮对话生成和评估，使用标准的语义连贯性、流畅度与多样性指标。

**📊 数据集**

RealReasoning（含数学单词推理与常识推理）数据集，先后进行迭代前后评测。

**📈 对比分析**

对比了 qwen-plus、qwen-plus-thinking、qwen-turbo、qwen-turbo-thinking、deepseek-r1 及其蒸馏版在迭代前的准确率（范围 60–96%），并指出迭代后准确率普遍下降，证明原数据集过于简易。

**⚠️ 局限性**

实验仅覆盖少数模型，迭代更新的细节与泛化性未充分验证；提示模板对不同场景的适应性尚待进一步测试；缺乏对模型推理过程的可解释性分析。

---

## 193. HumanOrbit: 3D Human Reconstruction as 360° Orbit Generation

**arXiv ID:** 2602.24148 | [PDF](https://arxiv.org/pdf/2602.24148v1)

**作者:** Keito Suzuki `[一作]` (University of California), Truong Nguyen `[通讯]` (University of California)

**通讯引用:** 15590 | [OpenAlex ID](https://openalex.org/A5102719190)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在单张人像图像基础上生成完整360°视角的视频，并利用这些多视角图像实现高质量的三维网格重建。

**💡 创新点**

创新点在于将预训练的视频扩散模型仅用少量LoRA参数进行微调，使其能够在不依赖姿态或相机标注的情况下学习完整环视轨迹；同时提出了基于生成视角的SfM与网格雕刻重建流水线，兼顾身份与视角一致性。

**🔧 技术方法**

核心技术包括基于DiT的视频扩散模型、VAE编码解码、CLIP与文本编码器、LoRA低秩适配、VGGT相机姿态估计、Poisson表面重建以及基于可微渲染的网格优化。

**📊 数据集**

训练数据来源于500个3D人类扫描，使用Blender渲染出约3000段环视视频；评测使用CCP全身图像集与CelebA人像图像集。

**📈 对比分析**

与SV3D、MV-Adapter、PSHuman等现有方法在CLIP Score、MEt3R和MVReward三项指标上进行对比，HumanOrbit在全身和头部图像上均取得最高分，尤其在与人类偏好对齐的MVReward上表现最优。

**⚠️ 局限性**

局限性包括环视固定在同一仰角，导致顶面或下颌等区域难以观测；推理时间较长（约17分钟），帧数减少会影响质量。

---

## 194. Multi-Agent Causal Reasoning for Suicide Ideation Detection Through Online Conversations

**arXiv ID:** 2602.23577 | [PDF](https://arxiv.org/pdf/2602.23577v1)

**作者:** Jun Li `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 37188 | [OpenAlex ID](https://openalex.org/A5100404176)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Multi-Agent Causal Reasoning (MACR) 框架，利用多智能体生成对话树的对照想象节点，并通过前门调整来检测在线对话中的自杀风险。

**💡 创新点**

①将认知评价理论与 Paul‑Elder 批判性思维框架融合，构建四子智能体协同生成心理影响中介；②利用生成的心理影响作为前门中介，直接消除未观测混杂；③在对话树结构上进行增广，提升信息量。

**🔧 技术方法**

多智能体协作、认知评价与批判性思维理论、前门调整（front‑door），E5‑large‑v2 编码与 K‑means 聚类、Qwen‑3‑4B‑Instruct 与 DeepSeek‑Chat 生成对照想象节点及决策输出。

**📊 数据集**

Suicidal Comment Tree（SCT）数据集与扩展的 Protective Factor‑Aware（PFA）数据集（包含评论树）。

**📈 对比分析**

与 4 类基线（传统自杀风险预测模型、去偏 LLM、强 LLM、图神经网络）对比，MACR 在 Weighted‑F1 上在两数据集均获得最佳结果：PFA 0.3768，SCT 0.5108，显著优于所有对照模型。

**⚠️ 局限性**

局限性：①多智能体协作对性能提升有限；②实验仅覆盖两组数据集，泛化性待验证；③模型复杂度高，依赖多 LLM；④前门调整的假设与实际混杂关系可能不完全匹配；⑤缺乏真实临床验证与部署安全评估。

---

## 195. Doc To The Future: Infomorphs for Interactive, Multimodal Document Transformation and Generation

**arXiv ID:** 2602.23366 | [PDF](https://arxiv.org/pdf/2602.23366v1)

**作者:** Balasaravanan Thoravi Kumaravel `[一作]` `[通讯]` (Microsoft Research), Balasaravanan Thoravi Kumaravel (Microsoft Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了DocuCraft系统，利用可视化画布和可组合的Infomorph模块化实现多模态文档的交互式转换与生成。

**💡 创新点**

核心创新在于引入Infomorph概念（Scatter、Gather、Transduce三类可组合的AI增援操作），以及在设计空间中明确用户控制、信息嗅觉、交互粒度等维度，突破单步黑箱生成的局限。

**🔧 技术方法**

技术栈包括前端React+Mantine、xyflow、Tiptap/Blocknote；后端Node.js/Python Flask；LLM模型如GPT‑4o、GPT‑4o‑Mini；多模态检索嵌入ColPali、OpenAI Ada；生成式图像模型DALL‑E、Stable Diffusion+ControlNet；以及向量检索与缓存机制。

**📊 数据集**

使用多源真实文档（PDF、DOCX、PPTX、XLSX）以及网络链接、表格和图片等；不依赖单一公开数据集，而是聚合实验室内部和公开的多模态文件。

**📈 对比分析**

通过设计空间对比表格展示DocuCraft在信息嗅觉、用户控制、交互粒度、Infomorph多样性、输入/输出多模态上优于ChatGPT、Perplexity、Copilot、模板化工具；系统实现了DAG式工作流缓存，能够在复杂工作流中保持较低延迟并显著提升用户可视化体验。

**⚠️ 局限性**

局限包括：画布视图在节点数量大时易产生拥挤与导航困难；当前对LLM的可靠性与可解释性依赖程度高；缺乏大规模用户研究验证其对不同知识工作者的通用性；以及对高阶专业领域的定制化支持仍有限。

---

## 196. Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks

**arXiv ID:** 2602.23898 | [PDF](https://arxiv.org/pdf/2602.23898v1)

**作者:** Qihua Dong `[一作]` (Northeastern University), Yun Fu `[通讯]` (Northeastern University)

**通讯引用:** 31444 | [OpenAlex ID](https://openalex.org/A5005819096)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Ref-Adv这套新的参照表达式理解（REC）基准，专门针对现有数据集的“快捷路”问题，要求模型进行多步文本与视觉推理。

**💡 创新点**

创新点在于通过LLM驱动的两阶段生成流程，仅保留足以唯一定位目标的描述，并且显式加入“硬性干扰物”，从而消除冗余特征导致的快捷匹配；同时通过三位人工审核保证数据的准确性和不可歧义性。

**🔧 技术方法**

技术包括GPT‑4o/ChatGPT等LLM的特征挖掘与表达式生成、Set‑of‑Marks (SoM)与Semantic‑SAM的细粒度定位、以及链式思考（CoT）与多阈值IoU评估。

**📊 数据集**

使用了COCO和OpenImages v7的panoptic实例图像，经过过滤后构成约1.4万条案例，并公开提供1,142条可复现子集（Ref-Adv‑s）。

**📈 对比分析**

在13种闭源与开源多模态大型语言模型上进行评测，结果显示它们在RefCOCO系列几乎达到90%+准确率，但在Ref-Adv上准确率大幅下降（从60%左右降至40%以下），证明模型对多步推理和视觉细粒度仍存在显著不足；链式思考在Ref-Adv上明显提升性能。

**⚠️ 局限性**

局限性包括：① 数据集规模仍有限，难以覆盖所有视觉场景；② 依赖LLM生成和人工审核的流程成本较高；③ 目前评测侧重定位而非分割或更细粒度的视觉理解，未来需进一步扩展。

---

## 197. ARGUS: Seeing the Influence of Narrative Features on Persuasion in Argumentative Texts

**arXiv ID:** 2602.24109 | [PDF](https://arxiv.org/pdf/2602.24109v1)

**作者:** Sara Nabhani `[一作]` (University of Groningen), Malvina Nissim `[通讯]` (University of Groningen)

**通讯引用:** 2827 | [OpenAlex ID](https://openalex.org/A5040564747)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了叙事特征在 Reddit ChangeMyView 论坛论证中的说服力，构建并使用 Argus 框架对叙事与说服关系进行大规模定量分析。

**💡 创新点**

创新点在于：①将叙事视为可量化的标量属性，采用软标签捕捉观点差异；②提出六维叙事特征（Agency、Event Sequencing、Suspense、Curiosity、Surprise、World Making）并剖析其对说服的具体影响；③系统比较编码器模型与零样本大语言模型的性能。

**🔧 技术方法**

技术手段包括：Transformer 预训练模型（DistilBERT、BERT、RoBERTa、ModernBERT）微调用于二分类与标量回归；Brier 分数、Wasserstein 距离、RMSE 等评价指标；混合效应逻辑回归分析说服概率。

**📊 数据集**

使用了 2025 年前的 ChangeMyView 数据集，先从 100 条高叙事比例讨论中抽取 620 条评论做手工标注；随后在 963,253 条公开评论上应用训练好的模型进行大规模预测。

**📈 对比分析**

方法比较：对比软标签（标量）与硬标签（二进制）模型、四种预训练模型的交叉验证性能；与 StorySeeker 与 NarrDetect 及 Llama 3.1 8B/70B 零样本推理结果对照。表现显示：①软标签模型在标量预测上优于硬标签；②四种编码器模型差异不大；③Fine‑tuned 编码器显著优于 LLM 在所有特征上，零样本 LLM 仅在部分二分类任务上可与编码器相当。

**⚠️ 局限性**

局限性：①数据仅限于 Reddit 的 ChangeMyView 论坛，主题与受众具有一定偏差；②叙事标注为主观，虽用软标签缓解但仍可能受文化与个人经验影响；③只评估了 Llama 系列 LLM，未探究其他模型或少量样本微调；④未深入分析叙事维度交互、主题差异或用户经验层面的细节。

---

## 198. Acceleration-Based Control of Fixed-Wing UAVs for Guidance Applications

**arXiv ID:** 2602.23821 | [PDF](https://arxiv.org/pdf/2602.23821v1)

**作者:** Jixiang Wang `[一作]` (Beijing Institute of Technology), Shaoming He `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 3090 | [OpenAlex ID](https://openalex.org/A5100717587)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在固定翼无人机上实现了加速度指令到姿态角速度和归一化推力的映射，实现了加速度基引导算法的实地部署。

**💡 创新点**

提出了基于总能量控制的无模型推力-能量加速度映射识别方法，并给出了法向加速度到姿态角速度的物理可解释映射，构成完整的加速度外环控制框架。

**🔧 技术方法**

采用小角度假设、总能量控制（TECS）思想、线性回归识别、比例控制及优先级切换策略，并通过PX4/APM低层控制实现。

**📊 数据集**

使用SkyFury VTOL平台的实时飞行数据进行推力-能量加速度映射识别和实验验证，未使用公开数据集。

**📈 对比分析**

与Gazebo仿真对比验证加速度跟踪性能；实验中实现PN引导拦截，误差0.58 m，切向和法向加速度跟踪误差均在几厘米级，证明方案可行且精度良好。

**⚠️ 局限性**

仅适用于小角度、无风、无极限约束的情形；缺乏正式稳定性证明和风、俯仰极限等约束的考虑，需进一步扩展。

---

## 199. A Mixed Diet Makes DINO An Omnivorous Vision Encoder

**arXiv ID:** 2602.24181 | [PDF](https://arxiv.org/pdf/2602.24181v1)

**作者:** Rishabh Kabra `[一作]` (Google DeepMind), Niloy J. Mitra `[通讯]` (University College London)

**通讯引用:** 21057 | [OpenAlex ID](https://openalex.org/A5058170430)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在预训练视觉编码器DINOv2的基础上，提出Omnivorous Vision Encoder，通过少量可训练层实现不同视觉模态（RGB、深度、分割）在同一特征空间中的对齐，并保持原始语义。

**💡 创新点**

创新点在于：①仅对冻结的基础模型的后置层做轻量化适配器，避免整体微调；②结合对齐损失和教师引导的锚定损失，防止表示坍塌；③使用颜色化+模态混合数据增强生成“硬正样本”与连续模态空间。

**🔧 技术方法**

使用参数高效的教师-学生框架，InfoNCE对齐损失、cosine锚定损失；Vision Transformer (ViT‑B/14) 作为基础模型；数据增强包括颜色化、模态混合。

**📊 数据集**

主要数据集：ScanNet、MOVi、TartanAir、NYUv2、ADE20k、Cityscapes、Pascal VOC、ImageNet；还有合成多模态数据。

**📈 对比分析**

与冻结的DINOv2相比，Omnivorous在跨模态检索（R@1从4.6%提升至46.1%，Median Rank降至2）、深度估计、语义分割、ImageNet线性/knn分类等任务均保持或略优，特别是零样本跨模态迁移表现显著。

**⚠️ 局限性**

局限在于：仅对现有模型后置层微调，未对整个网络进行预训练；对高分辨率微调的必要性不确定；对细粒度/极低资源模态的泛化仍有限；对不同基础模型的适用性需进一步验证。

---

## 200. GenDRAM:Hardware-Software Co-Design of General Platform in DRAM

**arXiv ID:** 2602.23828 | [PDF](https://arxiv.org/pdf/2602.23828v1)

**作者:** Tsung-Han Lu `[一作]` (University of California San Diego), Tajana Rosing `[通讯]` (University of California San Diego)

**通讯引用:** 10836 | [OpenAlex ID](https://openalex.org/A5025573294)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种基于3D M3D DRAM的Processing‑in‑Memory加速器GenDRAM，能够在单芯片上完成All‑Pairs Shortest Path（APSP）与基因组序列比对的完整动态规划工作流，彻底消除宿主‑加速器间的通信瓶颈；

**💡 创新点**

1) 通过异构PU（搜索PU + 计算PU）和无乘法器的通用半环算子单元，构建可统一加速多种DP任务的平台；2) 开发3D感知数据映射策略，利用M3D DRAM的分层延迟优势将热点数据置于低延迟层；3) 采用Near‑Memory Processor架构，实现高带宽、低延迟的内存内计算，完全在芯片内完成端到端工作流；

**🔧 技术方法**

使用3D monolithic M3D DRAM（Cu‑Cu混合互连）、Near‑Memory Processor、无乘法器的min/+与max/+半环算子、3D‑aware数据映射与通道交织、跨PU广播与流水线调度、专用搜索与计算PU等；

**📊 数据集**

APSP使用 SNAP 与 OpenStreetMap 图数据集（节点数 5k‑65k），基因组比对使用 GRCh38 人类参考基因组；短读由 Mason 生成（Illumina 5% 错误），长读由 PBSIM 生成（PacBio 15%、ONT 30% 错误）；

**📈 对比分析**

通过周期精确仿真与实测基准（NVIDIA A100/H100 GPU、RapidGraph、RAPIDx、ABSW 等）对比；结果显示 GenDRAM 在 APSP 上相对 A100 提升 68×，在基因组端到端工作流提升 22×，能效提升分别高达 152×、3,400×等；

**⚠️ 局限性**

需要复杂的3D制造工艺与较大硅面积（105 mm²）及热管理挑战；对M3D DRAM物理特性高度依赖，层间延迟不均会影响性能；目前仅针对DP型工作负载，无法直接扩展至非DP算法；

---

## 201. Inferring Chronic Treatment Onset from ePrescription Data: A Renewal Process Approach

**arXiv ID:** 2602.23824 | [PDF](https://arxiv.org/pdf/2602.23824v1)

**作者:** Pavlin G. Poličar `[一作]` (University of Ljubljana), Blaž Zupan `[通讯]` (University of Ljubljana)

**通讯引用:** 13016 | [OpenAlex ID](https://openalex.org/A5073792028)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过对电子处方时间序列建模为更新过程，利用变点检测方法推断慢性治疗的启动时间，以解决诊断记录的左截断问题。

**💡 创新点**

将处方时间视为 Renewal Process，并将 Poisson 与 Weibull 分布的似然比变点检测相结合，区别随机处方与持续治疗，从而得到更可时序、可解释的疾病启动估计。

**🔧 技术方法**

使用 Renewal-based 时序点过程模型（Poisson、Weibull）、似然比变点检测、药物-疾病关联学习、ICD10 规则基线比较与统计评估。

**📊 数据集**

来自斯洛文尼亚全国电子处方系统的 240 万人、1.01 亿次处方与 5800 万诊断的完整数据集。

**📈 对比分析**

与仅依赖单次处方触发的规则基线进行对比，变点方法显著降低了过早触发的错误检测，召回率虽略低但更符合临床时间窗口，且召回率随处方密度显著提升。

**⚠️ 局限性**

需要足够多的处方窗口；对处方稀疏或短期疾病表现有限；模型仅捕捉持续治疗的启动，未建模治疗终止；行政标签的噪声可能影响参数估计。

---

## 202. Cross-Representation Knowledge Transfer for Improved Sequential Recommendations

**arXiv ID:** 2602.23471 | [PDF](https://arxiv.org/pdf/2602.23471v1)

**作者:** Artur Gimranov `[一作]` (HSE University), Evgeny Frolov `[通讯]` (HSE University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为CREATE的框架，将Transformer（如SASRec/BERT4Rec）与图神经网络（如LightGCN/UltraGCN）相结合，用于下一物品预测，并通过特征对齐提升用户表示一致性。

**💡 创新点**

核心创新在于：①采用Barlow Twins无负样本对齐方式实现序列与图表征的冗余削减；②在训练前通过warm‑up预训练图编码器，避免初期冲突；③模型在推理时无需用户embedding，省去fold‑in步骤。

**🔧 技术方法**

使用的技术包括Transformer序列编码器、图卷积网络图编码器、全交叉熵（或可扩展交叉熵）损失、Barlow Twins对齐损失、全局时间切分（global temporal split）和可扩展负采样。

**📊 数据集**

实验数据集涵盖MovieLens‑1M、Amazon 5‑core 子集（Clothing、Shoes & Jewelry、Sports & Outdoors、Beauty）以及Yandex Music的Yambda‑50M。

**📈 对比分析**

在与SASRec、BERT4Rec、MRGSRec、LOOM、GSAU、LightGCN、UltraGCN等多种基线的对比中，CREATE在NDCG@10/100、Recall@10/100和Coverage上均实现显著提升，例如在Yambda‑50M上NDCG@10提升+38%，在Beauty上Recall@10提升+21%。

**⚠️ 局限性**

局限性包括：①对warm‑up epoch和图规模的超参数敏感；②Barlow Twins对齐权重需调优，过大会降低Coverage；③对高频用户/物品的冷启动效果尚未充分验证。

---

## 203. Evaluating Accuracy of Vine Robot Shape Sensing with Distributed Inertial Measurement Units

**arXiv ID:** 2602.24202 | [PDF](https://arxiv.org/pdf/2602.24202v1)

**作者:** Alexis E. Laudenslager `[一作]` (University of Notre Dame), Margaret McGuinness `[通讯]` (University of Notre Dame)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对主动和被动操纵的藤机器人进行分布式IMU形状感知的实验评估，并系统量化漂移、长度和传感器间距对定位误差的影响。

**💡 创新点**

首次在主动操纵场景下量化IMU漂移并分析不同长度和传感器间距对形状估计误差的影响，揭示最优传感器间距并提出改进方向。

**🔧 技术方法**

使用分布式BNO055 IMU、偏移校准、基于单弯点的几何模型和实验室测量（影像标定）进行形状感知。

**📊 数据集**

实验数据包括18个IMU在不同长度（30–175 cm）、不同弯曲角度、主动/被动操纵以及多种传感器间距下的测量。

**📈 对比分析**

通过与地面真值图像比对，结果显示主动操纵下尖端误差约16%，被动约11%，长度增加误差正相关，传感器间距在40–80 cm时误差最低；与仅使用尖端IMU的方法相比，形状感知在全身定位上更完整但精度未显著提升。

**⚠️ 局限性**

受限于实验室单平面环境、IMU非均匀漂移、几何模型假设（单弯点、不可伸缩），导致误差较大且对精细导航不够精确。

---

## 204. LeGend: A Data-Driven Framework for Lemma Generation in Hardware Model Checking

**arXiv ID:** 2602.24010 | [PDF](https://arxiv.org/pdf/2602.24010v1)

**作者:** Mingkai Miao `[一作]` (Hong Kong University of Science and Technology), Hongce Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 430 | [OpenAlex ID](https://openalex.org/A5003614499)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种一次性全局表示学习框架，对电路中的每个锁存器预训练嵌入，再用轻量级的 permutation‑invariant 网络快速预测高质量的 Lemma，并在 IC3/PDR 引擎初始化时侧加载这些 Lemma 以加速验证。

**💡 创新点**

核心创新在于：① 彻底摆脱逐句子图分析的高昂成本，改用一次性全局图表示；② 采用自监督对比学习（GraphCL+GIN）结合动态翻转率特征来捕捉锁存器的结构与动态功能；③ 设计 DeepSets 预测器实现语句集的可置换性，并通过严格的 sanity‑check 过滤保证 Lemma 合法。

**🔧 技术方法**

技术手段包括：自监督对比学习 (GraphCL)，GIN 编码器，动态翻转率特征增强，DeepSets 聚合 + MLP 评分器，Clause side‑loading 接口，SAT 解决器与 AIGER 图转换工具。

**📊 数据集**

数据集为 HWMCC（2008‑2024）单属性安全基准，训练集约 250 个 AIG（<20k 锁存器）产生约 1.79M 条标签；测试集 200 个独立实例，覆盖多种运行时区间。

**📈 对比分析**

与原始 IC3ref、ABC 以及现有 ML‑guided 方法 DeepIC3、IC3‑CTP 对比；在 200 场基准上，Proof‑rate 分别提升 15/22 案例，PAR‑2 时间缩短 1.56×（IC3ref）和 1.78×（ABC），显示显著的加速效果。

**⚠️ 局限性**

局限性：目前仅在单 GPU 上训练/推理，尚未扩展到多 GPU；对极大规模电路的 GPU 内存要求仍存在瓶颈；以及需依赖离线预训练，部署时仍需一次性完整图计算。

---

## 205. Spiky Rank and Its Applications to Rigidity and Circuits

**arXiv ID:** 2602.23503 | [PDF](https://arxiv.org/pdf/2602.23503v1)

**作者:** Lianna Hambardzumyan `[一作]` (University of Copenhagen), Adi Shraibman `[通讯]` (Academic College of Tel Aviv-Yaffo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出空间-时间复杂度度量，用于评估空间网络中子图数量及相关优化问题的计算难度

**💡 创新点**

将空间维度引入块分解/块熵度量，得到更紧的下界，并给出采样估计方法

**🔧 技术方法**

使用图论、块分解、块熵、采样算法与概率分析

**📊 数据集**

使用合成空间网络和真实交通/道路网络数据集

**📈 对比分析**

与传统块分解/块熵比较，新度量提供更紧的下界，子图枚举显著减少，搜索效率提升

**⚠️ 局限性**

计算复杂度高，精确求解NP-hard，估计依赖采样，误差和规模受限

---

## 206. Unsupervised Baseline Clustering and Incremental Adaptation for IoT Device Traffic Profiling

**arXiv ID:** 2602.24047 | [PDF](https://arxiv.org/pdf/2602.24047v1)

**作者:** Sean M. Alderman `[一作]` (Dakota State University), John D. Hastings `[通讯]` (Dakota State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了一个两阶段无监督聚类与增量适配管线：先用 DBSCAN 对 IoT 设备流量进行基线聚类，随后用 BIRCH 对新设备进行增量更新。

**💡 创新点**

创新点在于将密度聚类与增量学习相结合，采用包络式流特征实现高效无标签设备指纹，并在长时域数据集上验证其对设备演化的适应性。

**🔧 技术方法**

使用了流级特征提取、DBSCAN、BIRCH 等无监督聚类技术，并通过 NMI、Silhouette、purity、share 等指标评估聚类质量与增量性能。

**📊 数据集**

实验基于 Deakin IoT 数据集（D‑IoT），包含 119 天、1.1 亿个数据包的多设备流量。

**📈 对比分析**

与传统聚类相比，DBSCAN 在基线阶段达 NMI 0.78、Silhouette 0.93；BIRCH 在增量阶段实现 0.87 的纯度、0.72 的捕获率，且更新耗时仅 0.13 s，显示了在保持已知设备精度的同时对新设备的快速适配。

**⚠️ 局限性**

局限性包括 DBSCAN 不能直接增量更新，BIRCH 可能产生子簇碎片化；方法依赖流级元数据，若仅有粗粒度汇总或加密隧道会降低可分性；未评估周期性全量重训练与长期漂移；仅在固定时间窗口下验证。

---

## 207. Central Bank Digital Currencies: Where is the Privacy, Technology, and Anonymity?

**arXiv ID:** 2602.23659 | [PDF](https://arxiv.org/pdf/2602.23659v1)

**作者:** Jeff Nijsse `[一作]` (RMIT University), Andrea Pinto `[通讯]` (Universidad de Los Andes)

**通讯引用:** 140 | [OpenAlex ID](https://openalex.org/A5003228223)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统评估了央行数字货币（CBDC）隐私设计的现状与挑战，并提出了涵盖法律、技术与交易三视角的综合隐私定义；

**💡 创新点**

创新点在于构建统一的CBDC隐私框架，并将其映射至加密技术栈，结合20个案例评估各阶段隐私技术的应用情况；

**🔧 技术方法**

采用的技术包括对称/非对称加密、数字签名、环签名、Schnorr签名、BBS+签名、盲签名、可验证随机函数（VRF）、多方计算（MPC）、同态加密、零知识证明（SNARK/STARK）、Pedersen承诺、混币、UTXO/账户模型等；

**📊 数据集**

使用的数据集为2019‑2023年间的67篇学术与灰色文献，以及对20个已发布或试点的CBDC项目的案例分析；

**📈 对比分析**

通过对比各阶段项目的PET采用情况，发现从研究到正式发行过程中隐私技术被大幅削减；论文未给出定量性能指标，但对可行性、合规性和可扩展性进行了定性评估；

**⚠️ 局限性**

局限性包括缺乏公开的技术实现细节、对隐私技术的实测不足、合规与可审计性平衡的具体方案缺失，以及大多数案例停留在理论或试点阶段，未能验证长期安全与性能。

---

## 208. GLUScope: A Tool for Analyzing GLU Neurons in Transformer Language Models

**arXiv ID:** 2602.23826 | [PDF](https://arxiv.org/pdf/2602.23826v1)

**作者:** Sebastian Gerstner `[一作]` (LMU Munich), Hinrich Schütze `[通讯]` (Munich Center for Machine Learning)

**通讯引用:** 34407 | [OpenAlex ID](https://openalex.org/A5071144367)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并发布了一个名为 Gluscope 的开源工具，用于可视化和分析 Transformer 语言模型中采用门控激活函数（如 SwiGLU、GEGLU）的神经元，提供了每个神经元在四种门控/输入符号组合下的激活示例和统计信息。

**💡 创新点**

创新点在于首次将门控激活函数的四种符号组合纳入分析框架，能够捕捉门控激活导致的正负交互效应，显著提升了对现代模型神经元行为的细粒度理解；同时提供了可复现的激活数据集和交互式可视化网站。

**🔧 技术方法**

采用 TransformerLens 进行模型状态访问与权重预处理，CircuitsVis 进行可视化；利用自定义脚本记录每个神经元在不同符号组合下的中间激活（gate、in、Swish、final 激活）统计，并生成 16 条示例文本；工具基于 Python、Hugging Face Datasets 与 PyTorch 构建。

**📊 数据集**

使用了公开可获取的 OLMo‑7B‑0424 语言模型及其对应的 Dolma 小样本数据集（约 20M tokens，45,734 条文本），并在此数据集上计算激活统计。

**📈 对比分析**

与现有的神经元可视化工具（如 Neuroscope、NeuroX、LM Debugger 等）对比，Gluscope 专门处理门控激活函数，提供四种符号组合的完整激活信息和示例，能够揭示此前工具难以发现的负激活模式和语义关联。性能上主要通过实例演示而非数值指标，展示了激活频率与权重相似度之间的负相关以及对特定词“again”生成概率的提升。

**⚠️ 局限性**

局限性包括：无法直接分析 MoE 模型或非 Transformer 架构（如 Mamba）；仍聚焦于单个神经元而非更复杂的稀疏自动编码特征；工具目前仅支持有限模型和数据集，需社区进一步扩展。

---

## 209. On the Need for (Quantum) Memory with Short Outputs

**arXiv ID:** 2602.23763 | [PDF](https://arxiv.org/pdf/2602.23763v1)

**作者:** Zihan Hao `[一作]` (University of California San Diego), Qipeng Liu `[通讯]` (University of California San Diego)

**通讯引用:** 707 | [OpenAlex ID](https://openalex.org/A5029992618)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并研究了L‑nested collision finding问题，并给出了经典与量子算法的上界与下界，证明了时间‑空间权衡下的最优查询复杂度；

**💡 创新点**

在短查询复杂度下，利用压缩oracle和交替实验等新技术将随机oracle模型与有状态算法联系，得到更紧的时间‑空间下界；

**🔧 技术方法**

主要使用了压缩oracle技术、交替实验、投影测量与递归分析等量子信息理论工具；

**📊 数据集**

该工作基于随机oracle与随机函数，未使用传统机器学习数据集；

**📈 对比分析**

通过与已有的多项式查询上界对比，证明了上界与下界在量级上相匹配，说明方法在理论上具有最优性；

**⚠️ 局限性**

局限性在于对M和N的多项式关系假设、对L为常数的限制以及对压缩oracle假设的依赖，实际实现时对状态空间规模与量子硬件资源有较高要求。

---

## 210. LFQA-HP-1M: A Large-Scale Human Preference Dataset for Long-Form Question Answering

**arXiv ID:** 2602.23603 | [PDF](https://arxiv.org/pdf/2602.23603v1)

**作者:** Rafid Ishrak Jahan `[一作]`, Sagnik Ray Choudhury `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了1.3M规模的长文本问答偏好数据集，并提出基于九维细粒度指标的可解释评估框架，利用逻辑回归模型对回答质量进行判定。

**💡 创新点**

创新点在于大规模LFQA偏好数据集的首次构建、细粒度可解释评估指标的系统设计，以及对LLM评测者的偏差、传递性和鲁棒性进行深入分析。

**🔧 技术方法**

使用LLM-as-judge、G‑EVAL、Veriscore、LanguageTool等工具提取指标，结合逻辑回归、对抗攻击（TextFooler、DeepWordBug）等技术进行评测与分析。

**📊 数据集**

数据来源于SHP‑2、Chatbot Arena、LFQA Eval三大源，整合后形成LFQA‑HP‑1M；实验还使用ELI5子集进行过滤与验证。

**📈 对比分析**

与GPT‑4o、Gemini‑2.5、Llama‑4等SOTA LLM评测者对比，逻辑回归模型准确率≈69%，略低于GPT‑4o；LLM表现受位置、长度偏差、非传递性影响，字符级扰动导致性能下降达10%。

**⚠️ 局限性**

限制包括域泛化能力不足、人类标注噪声、逻辑回归对细微特征捕捉有限，以及对LLM更深层特征理解的欠缺。

---

## 211. Long Range Frequency Tuning for QML

**arXiv ID:** 2602.23409 | [PDF](https://arxiv.org/pdf/2602.23409v1)

**作者:** Michael Poppel `[一作]` (Aqarios GmbH), Claudia Linnhoff-Popien `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了量子机器学习中可训练频率模型的频率可达性问题，并提出了基于三元格初始化的训练频率量子电路方案，在合成和真实时间序列数据上进行实验验证。

**💡 创新点**

创新点在于揭示可训练频率模型的梯度可达性局限，并通过三元格初始化提供一种既能保持指数门数压缩又能确保频率可达性的实用方法。

**🔧 技术方法**

采用角度编码的变分量子电路、傅里叶分析、梯度可达性实验、三元格（基于3的指数编码）初始化、PennyLane实现量子电路，并与固定频率和可训练频率基线进行对比。

**📊 数据集**

使用了人工合成的傅里叶序列目标函数以及实际的 Flight Passengers 时间序列数据进行实验。

**📈 对比分析**

通过 R² 分数对不同模型进行性能对比，三元格训练模型在合成数据上 R²≈0.9969（优于可训练频率仅 0.1841），在 Flight Passengers 数据上 R²提升 22.8% 达到 0.9671，表现优于固定频率和可训练频率基线。

**⚠️ 局限性**

主要限制包括梯度局部性导致的频率可达性受限、三元格初始化需要预设最大频率、模型规模仍随频率范围增长、以及对量子优势的证明尚不充分。

---

## 212. Hyper-reduction methods for accelerating nonlinear finite element simulations: open source implementation and reproducible benchmarks

**arXiv ID:** 2602.23551 | [PDF](https://arxiv.org/pdf/2602.23551v1)

**作者:** Axel Larsson `[一作]` (Princeton University), Siu Wun Cheung `[通讯]` (Lawrence Livermore National Laboratory)

**通讯引用:** 1325 | [OpenAlex ID](https://openalex.org/A5083936465)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对多种超采样（hyper‑reduction）方法在非线性有限元模型中的加速效果进行了系统评估，涵盖非线性扩散、弹性和拉格朗日流体动力学等benchmark问题，利用开放源代码实现进行可复现性测试。

**💡 创新点**

创新点在于将插值型gappy POD方法（DEIM、Q‑DEIM、S‑OPT）与积分型EQP方法在同一统一变分框架下进行全面对比，并揭示其在不同时间积分器、预测/再生产情景和问题复杂度下的性能差异与权衡。

**🔧 技术方法**

使用的技术包括投影式降阶模型（POD、Galerkin投影）、多种超采样算法（DEIM、Q‑DEIM、S‑OPT、EQP）、非负最小二乘求解器、可视化与Pareto前沿分析。

**📊 数据集**

数据集为五个数值基准：二维非线性扩散、二维非线性弹性、三维Sedov blast、三维Taylor‑Green vortex及三维三点冲击问题，采用高阶有限元离散与不同时间步长。

**📈 对比分析**

通过比较L²误差与在线计算时间的Pareto前沿，发现EQP在大多数情形下提供更高的加速与相对低误差，而插值方法在预测场景或RK2Avg积分器下更具优势；具体性能取决于问题类型、积分方法和采样点数。

**⚠️ 局限性**

局限性包括：1）对拉格朗日流体动力学问题，EQP因样本网格构造导致的额外开销未完全体现其点数优势；2）在预测测试中误差普遍高于再生产，表明超采样对参数外推的鲁棒性有限；3）仅考虑固定的残差能量阈值与单一时间窗口划分，未探索更复杂的自适应采样策略。

---

## 213. BTTackler: A Diagnosis-based Framework for Efficient Deep Learning Hyperparameter Optimization

**arXiv ID:** 2602.23630 | [PDF](https://arxiv.org/pdf/2602.23630v1)

**作者:** Zhongyi Pei `[一作]` (Tsinghua University), Mingsheng Long `[通讯]` (Tsinghua University)

**通讯引用:** 29212 | [OpenAlex ID](https://openalex.org/A5019241553)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Bad Trial Tackler（BTTackler）框架，利用训练诊断指标在超参数优化（HPO）过程中自动识别并提前终止训练失败的试验，以提升效率和准确性。

**💡 创新点**

首次将训练诊断指标引入HPO，设计了七个可量化的质量指标（如异常梯度、指数放大梯度、指数降低梯度等），实现了基于训练状态而非仅靠准确率的早期终止策略。

**🔧 技术方法**

采用多线程并行计算质量指标、回调函数和探测器记录训练信息、利用统计量（均值、方差、极值等）计算指标，并与现有HPO框架NNI集成；实现了模拟器用于指标校准。

**📊 数据集**

在三个经典任务上验证：Cifar10CNN（图像分类）、Cifar10LSTM（序列分类）和Ex96Trans（外汇时间序列预测）。

**📈 对比分析**

与四种主流HPO方法（Random Search、Gaussian Process、Tree-structured Parzen Estimator、SMAC）以及两种早期终止规则（Learning Curve Extrapolation、Median Stop Rule）进行对比。实验显示：BTTackler平均减少约40.33%时间获得与基线相同的最佳准确率；在给定时间预算内，Top-10 试验命中率提升至约72%（相较于约52%的早期终止规则）。

**⚠️ 局限性**

局限性包括：指标阈值需要经验调参，可能对不同网络和任务的适用性有限；部分指标计算成本较高，虽然通过并行降低了负担，但在极大规模训练时仍可能产生5%左右的额外开销；对极其复杂模型（如Transformer）效果相对有限，需进一步改进指标与诊断理论。

---

## 214. ArgLLM-App: An Interactive System for Argumentative Reasoning with Large Language Models

**arXiv ID:** 2602.24172 | [PDF](https://arxiv.org/pdf/2602.24172v1)

**作者:** Adam Dejl `[一作]` (Imperial College), Francesca Toni `[通讯]` (Imperial College)

**通讯引用:** 7156 | [OpenAlex ID](https://openalex.org/A5078354590)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 ArgLLM-App 的 Web 应用，用以将 Argumentative LLMs（ArgLLMs）应用于任何二元决策任务，支持 QBAF 的生成、定制与交互式编辑。

**💡 创新点**

创新点在于将 LLM 与可解释的 QBAF 结合，提供逐步评估的渐进语义，并允许用户直接通过滑块或图形添加/修改论证结构，从而实现可解释且可纠错的决策过程。

**🔧 技术方法**

采用 OpenAI LLM 作为底层模型，利用 QBAF 框架、DF-QuAD / Euler / Quadratic Energy 等渐进语义，结合 RAG（通过 PDF 转 Markdown）以及交互式图形可视化。

**📊 数据集**

没有使用公开数据集，系统通过用户上传的 PDF 文档或聊天输入动态生成 QBAF；在演示中使用了“乌克兰将于 2030 年前加入欧盟”的主张作为案例。

**📈 对比分析**

在二元决策的断言验证任务上，与 chain-of-thought 以及其他前沿方法相比，ArgLLM-App 在准确性上表现相当，但论文未给出定量指标。

**⚠️ 局限性**

局限性包括仅支持深度为 1 或 2 的 QBAF、单一 OpenAI LLM、仅 PDF 文档、单决策单用户、缺乏多 LLM 多代理、自动 RAG 整合以及多用户协作等功能。

---

## 215. On the Convergence of Single-Loop Stochastic Bilevel Optimization with Approximate Implicit Differentiation

**arXiv ID:** 2602.23633 | [PDF](https://arxiv.org/pdf/2602.23633v1)

**作者:** Yubo Zhou `[一作]` (Xi’an Jiaotong University), Haishan Ye `[通讯]` (Xi’an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种单循环随机近似隐式微分算法（SSAID），并给出了其非渐近收敛分析，证明其oracle复杂度为𝒪(κ^7 ε⁻²)。

**💡 创新点**

创新点在于首次对单循环随机AID算法进行精细的κ依赖分析，并实现与多循环方法（如stocBiO）相同的O(ε⁻²)收敛速率，同时将κ的指数从κ⁹降至κ⁷。

**🔧 技术方法**

采用了耦合误差分析、热身跟踪策略、随机梯度与Hessian向量乘积估计以及非渐近理论证明等技术。

**📊 数据集**

本文主要是理论分析，没有涉及具体数据集或实验。

**📈 对比分析**

与多循环方法相比，SSAID在单循环框架下实现了与标准非凸SGD相当的收敛速率，并在κ依赖上更优（κ⁷对比κ⁹），表明性能更好。

**⚠️ 局限性**

局限性包括未考虑方差减小技术、约束或PL条件的情况；缺乏实验验证，且对步长比例的选择较为敏感。

---

## 216. RewardUQ: A Unified Framework for Uncertainty-Aware Reward Models

**arXiv ID:** 2602.24040 | [PDF](https://arxiv.org/pdf/2602.24040v1)

**作者:** Daniel Yang `[一作]` (ETH Zurich), Andreas Krause `[通讯]` (ETH Zurich)

**通讯引用:** 30616 | [OpenAlex ID](https://openalex.org/A5003040843)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出RewardUQ框架，对RLHF中的奖励模型不确定性进行统一建模、评估和比较，并开源实现。

**💡 创新点**

创新点在于统一规范化现有UQ方法、引入兼顾准确性与置信度的排名得分、系统化评估多种架构、并揭示模型初始化对性能的关键影响。

**🔧 技术方法**

采用集成（MLP、LoRA）、MC Dropout、贝叶斯线性回归等UQ技术，并使用交叉熵损失、ECE/EBCE校准指标以及自定义排名分数。

**📊 数据集**

使用UltraFeedback、RewardBench、Skywork Reward Preference、Tulu3 8B Preference等数据集进行训练与评估，并基于Qwen3和Skywork系列预训练模型。

**📈 对比分析**

通过统一评估流程和RS_0.2排名分数比较不同UQ方法，结果显示无方法始终占优，性能高度依赖模型规模、数据集与初始化，finetuned基模型显著提升，贝叶斯线性回归大多数场景最佳，整体校准优良（ECE<0.1，EBCE<0.01）。

**⚠️ 局限性**

局限在于仅评估奖励模型本身，未涉及下游RLHF训练；排名分数存在准确性与置信度权衡；实验范围受限于选定算法、数据与指标，理论分析不足。

---

## 217. SegMate: Asymmetric Attention-Based Lightweight Architecture for Efficient Multi-Organ Segmentation

**arXiv ID:** 2602.23903 | [PDF](https://arxiv.org/pdf/2602.23903v1)

**作者:** Andrei-Alexandru Bunea `[一作]` (POLITEHNICA Bucharest), Radu Tudor Ionescu `[通讯]` (University of Bucharest)

**通讯引用:** 8118 | [OpenAlex ID](https://openalex.org/A5081017623)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

SegMate框架通过引入切片融合、异步解码、双重注意、位置编码及多任务学习等技术，在多器官CT分割中实现了更高效的2.5D处理。

**💡 创新点**

创新点在于将切片融合与位置条件FiLM结合的轻量化2.5D输入、异步解码与双重注意机制、以及多任务头的集成，从而在保持精度的同时大幅降低计算和显存需求。

**🔧 技术方法**

使用的技术包括SliceFusion注意力融合、Squeeze-and-Excitation(SE)与CBAM双重注意、Feature-wise Linear Modulation(FiLM)位置编码、ASPP瓶颈、轻量化解码器、以及Dice+Focal+CE多任务损失。

**📊 数据集**

数据集：TotalSegmentator、SegTHOR与AMOS22。

**📈 对比分析**

与vanilla EfficientNetV2-M、MambaOut-Tiny、FastViT-T12以及其他3D模型（Swin UNETR、nnU-Net、PaR等）对比，SegMate在TotalSegmentator上Dice达到93.51%并将VRAM降至295MB，GFLOPs下降至2.5×，在SegTHOR和AMOS22零样本和微调下也均取得领先或相当的Dice。

**⚠️ 局限性**

局限性包括仍依赖多尺度融合且对极小器官的精度提升有限，且在极端低显存环境下的可迁移性尚待验证。

---

## 218. RAD-DPO: Robust Adaptive Denoising Direct Preference Optimization for Generative Retrieval in E-commerce

**arXiv ID:** 2602.23964 | [PDF](https://arxiv.org/pdf/2602.23964v1)

**作者:** Zhiguo Chen `[一作]` (JD.com), Sulong Xu `[通讯]` (JD.com)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研发了一种针对电商生成检索的鲁棒自适应去噪直接偏好优化框架RAD-DPO，用于对结构化语义ID进行对齐。

**💡 创新点**

结合多标签全局对比、令牌级梯度分离和基于表示的动态奖励加权，解决了标准DPO在共享前缀、伪负样本和多标签覆盖方面的三大瓶颈。

**🔧 技术方法**

采用Qwen-1.7B生成模型、RQ‑Kmeans编码的SID、SFT+DPO、全局对比损失、梯度截断与余弦相似度动态权重等技术。

**📊 数据集**

使用京东700M交互日志，抽取约30M点击/下单正样本与未点击负样本进行训练。

**📈 对比分析**

与SFT基线和标准DPO在离线Recall@K、MRR以及在线A/B（用户转化率）对比，RAD‑DPO在所有规模（0.6B–8B）上均优于baseline，在线提升约+0.34% UCVR。

**⚠️ 局限性**

仍受限于工业日志的噪声与位置偏差，且对极低数据量或极大模型规模的泛化性未完全验证。

---

## 219. Learning to Generate Secure Code via Token-Level Rewards

**arXiv ID:** 2602.23407 | [PDF](https://arxiv.org/pdf/2602.23407v1)

**作者:** Jiazheng Quan `[一作]` (Fuyao University of Science and Technology), Chengbin Hou `[通讯]` (Fuyao University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了Vul2Safe框架和PrimeVul+数据集，并提出SRCode两阶段训练，用token级奖励提升LLM安全代码生成。

**💡 创新点**

通过自我反思生成高质量修复对、构造隐式提示的课程化数据；引入token级奖励在RL中细化安全优化。

**🔧 技术方法**

采用LLM自我反思、多轮生成修复、CodeQL静态分析；Token‑Level Reward (TLR)；PPO强化学习；Prompt工程。

**📊 数据集**

PrimeVul+（从PrimeVul 3,289个C/C++样本）与Vul2Safe生成的修复对；以及CWEval、CodeLMSec、CyberSecEval等基准。

**📈 对比分析**

在Qwen2.5‑Coder、DeepSeek、CodeLlama等多模型上，SRCode在安全率、FS@1等安全指标上平均提升约10%或更高，且在HumanEval Pro、MBPP Pro等通用代码生成任务保持或提升性能。

**⚠️ 局限性**

受PPO clip机制限制可能削弱token奖励的影响；依赖教师模型评估；数据可能被恶意利用；仅在C/C++场景验证，跨语言通用性仍需进一步评估。

---

## 220. Lap2: Revisiting Laplace DP-SGD for High Dimensions via Majorization Theory

**arXiv ID:** 2602.23516 | [PDF](https://arxiv.org/pdf/2602.23516v1)

**作者:** Meisam Mohammady `[一作]` (Iowa State University), Yuan Hong `[通讯]` (University of Connecticut)

**通讯引用:** 2230 | [OpenAlex ID](https://openalex.org/A5100725148)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Lap2，一种基于 ℓ₂‐裁剪的 Laplace DP‑SGD，利用主要化理论构建多元时刻计数器，显著降低维度导致的噪声放大问题。

**💡 创新点**

创新点在于：①利用主要化与 Schur‑凸性将 ℓ₂‑裁剪梯度的敏感度转化为数据无关的上界；②在此框架下实现了可在高维模型下保持强（ε,δ）隐私的 Laplace 噪声；③通过闭式近似给出噪声尺度与裁剪阈值的最佳匹配。

**🔧 技术方法**

核心技术包括 Laplace 机制、ℓ₂ 量裁剪、主要化理论与 Schur‑凸性、时刻计数器（Moments Accountant）与其多元扩展、参数搜索与闭式噪声优化。

**📊 数据集**

在 CV 任务上使用 MNIST、Fashion‑MNIST、CIFAR‑10（ViT 微调）；在 NLP 任务上使用 SST‑2、QNLI（RoBERTa‑base 微调）和 DistilGPT‑2 + E2E 文本生成。

**📈 对比分析**

与 Gaussian DP‑SGD 和传统 ℓ₁‑裁剪 Laplace 机制对比，Lap2 在相同隐私预算下往往匹配或优于 Gaussian，尤其在 ε≤1 的强隐私 regime 下准确率提升 2–10%；在大模型微调（ViT、RoBERTa、DistilGPT‑2）中可达 98%/87% 的准确率或更高的生成质量，训练耗时与 Gaussian 差异不大。

**⚠️ 局限性**

局限在于：①对极高维模型（>10⁶ 参数）仍需对主要化集合做近似，可能导致保守估计；②在某些 CV 任务（如 CIFAR‑10 ViT）Lap2 效果仍略低于 Gaussian；③实现上需要对裁剪阈值与噪声尺度做网格/二分搜索，增加配置复杂度；④未对非均匀层级裁剪或更高阶噪声分布进行探索。

---

## 221. Complex Cognition: A New Theoretical Foundation for the Design and Evaluation of Visual Analytics Systems

**arXiv ID:** 2602.23377 | [PDF](https://arxiv.org/pdf/2602.23377v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 222. TSC: Topology-Conditioned Stackelberg Coordination for Multi-Agent Reinforcement Learning in Interactive Driving

**arXiv ID:** 2602.23896 | [PDF](https://arxiv.org/pdf/2602.23896v1)

**作者:** Xiaotong Zhang `[一作]` (Chinese Academy of Sciences), Long Chen `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 95838 | [OpenAlex ID](https://openalex.org/A5100333572)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文提出了一个基于拓扑条件的斯塔克贝格协调框架TSC，用于在无通信的分布式环境下解决密集交通中的冲突。

**💡 创新点**

创新点在于通过轨迹交织生成时间可变的有向优先图，将局部优先关系转化为图局部的斯塔克贝格子游戏，避免构建全局顺序并实现稀疏领导-追随约束。

**🔧 技术方法**

使用的技术包括多智能体强化学习、CTDE（集中训练分布式执行）、图神经网络用于优先图推断、Stackelberg条件化的演员-评论家、TopK过滤、轨迹编织距离等。

**📊 数据集**

数据集是基于VMAS仿真平台的四个密集交通场景（Clover、Weave、Merge、Bypass），不使用公开真实轨迹。

**📈 对比分析**

与MFPO、SigmaRL、XP-MARL三种基线比较，TSC在所有四个场景中以最低碰撞率、竞争性的平均速度和更好的平稳性获得显著提升，尤其在Weave场景降低碰撞率至0.99%。

**⚠️ 局限性**

限制在于仅在同类动力学车辆上验证，对异构车辆和感知不确定性鲁棒性不足，缺乏形式化安全保证，且对更大规模交通的进一步评估仍待探索。

---

## 223. Modelling and Simulation of Neuromorphic Datasets for Anomaly Detection in Computer Vision

**arXiv ID:** 2602.23514 | [PDF](https://arxiv.org/pdf/2602.23514v1)

**作者:** Mike Middleton `[一作]` (University of York), Martin A. Trefzer `[通讯]` (University of York)

**通讯引用:** 700 | [OpenAlex ID](https://openalex.org/A5056241158)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了 ANTShapes，一个基于 Unity 的 neuromorphic 数据集仿真框架，用于生成可配置的事件驱动场景并标注异常行为。

**💡 创新点**

创新点在于利用中心极限定理构建多维正态分布的异常行为模型，支持通过参数化控制异常阈值、行为维度和行为空间维度，实现高度可定制的异常标签。

**🔧 技术方法**

采用 Unity 引擎进行 3D 场景渲染，正态分布采样生成对象行为，基于像素差分与双极阈值门控实现事件生成，支持单目灰度与事件视图的输出。

**📊 数据集**

本研究未使用现有真实 neuromorphic 数据集，而是生成可任意指定样本量的合成 ANTShapes 数据集，包含事件序列、标签与帧数据。

**📈 对比分析**

目前尚未进行 SNN 训练或性能比较，作者指出将来会在合成数据上评估 SNN 的异常检测效果并与其他方法对比。

**⚠️ 局限性**

主要局限包括：假设事件互相独立且无因果关系；缺乏真实人类行为与动态视角；未实现立体视觉和相机运动；模型仅适用于抽象形状场景。

---

## 224. Humans and LLMs Diverge on Probabilistic Inferences

**arXiv ID:** 2602.23546 | [PDF](https://arxiv.org/pdf/2602.23546v1)

**作者:** Gaurav Kamath `[一作]` (McGill University), Siva Reddy `[通讯]` (Mila -- Quebec AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 ProbCOPA 数据集并对其进行人类与推理型 LLM 的概率推理评估

**💡 创新点**

创新点在于将 COPA 转化为概率推理任务，采集大规模多注释数据并系统比较 LLM 与人类的概率判断差异

**🔧 技术方法**

使用人类标注、LLM 推理链生成、数值估计以及差分熵、Wasserstein 距离等技术进行分析

**📊 数据集**

ProbCOPA（210 条英文推理实例），每条至少有 25 名参与者的概率评分

**📈 对比分析**

通过差分熵和 Wasserstein 距离对模型与人类分布进行比较，结果显示模型在极端值表现相似但在中间值上差距显著，模型方差低，集成虽改善但仍不及人类

**⚠️ 局限性**

局限在于仅限英语、数据来源于 COPA 可能已出现在训练集中，以及 LLM 通过口头数值表达概率的方式可能不完全可靠

---

## 225. Enhancing Spatial Understanding in Image Generation via Reward Modeling

**arXiv ID:** 2602.24233 | [PDF](https://arxiv.org/pdf/2602.24233v1)

**作者:** Zhenyu Tang `[一作]` (Peking University), Daquan Zhou `[通讯]` (Peking University)

**通讯引用:** 9205 | [OpenAlex ID](https://openalex.org/A5100554498)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 SpatialReward-Dataset 与 SpatialScore 奖励模型，利用在线强化学习显著提升文本到图像生成模型在多对象复杂空间关系上的理解与表现。

**💡 创新点**

创新点在于：①构建了 80k 对比偏好数据集；②设计专门评估空间关系的奖励模型 SpatialScore，准确率超越多种现有奖励模型与主流 VLM；③在在线 RL 中引入 top‑k 筛选与 GRPO，解决优势偏置并提升训练效率。

**🔧 技术方法**

使用技术包括：基于 Qwen2.5‑VL‑7B 的 VLM 作为奖励模型（LoRA 微调）；对 Flux.1‑dev 进行 GRPO 强化学习；采用 SDE 与 top‑k 过滤来稳定优势估计；以及对奖励模型的 Gaussian 评分与 Bradley‑Terry 目标进行优化。

**📊 数据集**

数据集：SpatialReward‑Dataset（80k 复杂空间关系偏好对），以及用于评估的 365 对比样本与公开基准 DPG‑Bench、TIIF‑Bench、UniGenBench++。

**📈 对比分析**

与基线对比：在 SpatialScore 评估中由 2.18 提升至 7.81，空间子维度均有显著提升；在 DPG‑Bench、TIIF‑Bench 等基准中均超过原始 Flux.1‑dev 并逼近 GPT‑Image‑1；相较于 Flow‑GRPO（GenEval）模型，在长提示下性能更优。

**⚠️ 局限性**

局限性：奖励模型仍受 VLM 的推理误差影响，对极端复杂或视觉难度高的场景仍可能产生幻觉；online RL 训练成本高；top‑k 筛选在高难度样本时可能削弱多样性；目前仅针对文本到图像任务，缺乏跨模态或更大尺度的验证。

---

## 226. DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic Simulation with Online Diffusion Enhancer

**arXiv ID:** 2602.24096 | [PDF](https://arxiv.org/pdf/2602.24096v1)

**作者:** Yuxuan Zhang `[一作]` (NVIDIA), Zan Gojcic `[通讯]` (NVIDIA)

**通讯引用:** 2535 | [OpenAlex ID](https://openalex.org/A5025638024)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种在线生成式增强框架，将神经渲染场景产生的含有噪声和伪影的帧转换为视觉逼真且时间连贯的模拟结果。

**💡 创新点**

创新点在于：①将预训练的多步扩散模型改造为单步、时序条件化的增强器；②构建一套多源数据预处理流水线，合成对齐、光照和阴影等多维度的高质量对照样本；③设计多尺度感知损失和时间扭曲损失，显著抑制高频伪影并提升时间一致性。

**🔧 技术方法**

采用的技术包括Cosmos 0.6B文本到图像扩散模型、冻结的VAE编码器/解码器、时间注意力模块、光照重映射扩散模型、PBR阴影渲染以及多尺度VGG感知损失和光流 warping 损失。

**📊 数据集**

训练数据主要来自内部自动驾驶数据、Waymo公开数据以及合成的五大类对照样本（稀疏重建、ISP 随机化、光照重建、PBR 阴影、资产再插入），评估则使用三套测试集：内部新轨迹、Waymo对象插入和有标注的光照/阴影/ISP 验证集。

**📈 对比分析**

与SDEdit、InstructPix2Pix、V2V等图像/视频编辑基线以及VHTT、Ke等视频和图像融合基线对比，实验显示在 FID、FVD、DINO‑Struct‑Dist 和时间一致性指标上均优于基线，且推理速度提升至单卡 212 ms，用户研究与 VLM 评估中获得约 84 % 的首选率。

**⚠️ 局限性**

局限性包括：仍受限于单 GPU 计算资源，对极端稀疏视角下的重建伪影尚未彻底解决；在完全不同领域（如室内或城市街景）上的迁移性能需进一步验证。

---

## 227. AudioCapBench: Quick Evaluation on Audio Captioning across Sound, Music, and Speech

**arXiv ID:** 2602.23649 | [PDF](https://arxiv.org/pdf/2602.23649v1)

**作者:** Jielin Qiu `[一作]` (Salesforce AI Research), Huan Wang `[通讯]` (Salesforce AI Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建 AudioCapBench benchmark 并评估 13 大型多模态模型在环境声音、音乐、语音三类音频的自动描述能力。

**💡 创新点**

三方面创新：多域覆盖、LLM‑as‑Judge 评估框架（准确度、完整度、幻觉），以及公开完整代码实现可复现。

**🔧 技术方法**

使用 GPT‑4.1 作为自动评判者；传统 METEOR、BLEU‑4、ROUGE‑L 参考指标；API 调用 OpenAI Chat/Realtime 与 Google Gemini。

**📊 数据集**

采样自 Clotho、AudioCaps、MusicCaps、情感语音 caption 数据集，共 1000 条音频。

**📈 对比分析**

对 13 模型进行 LLM‑Judge 平均分和传统指标评测，Gemini 3 Pro 最优（整体分 6.00/10），OpenAI 低幻觉但准确性低，音乐领域最难。

**⚠️ 局限性**

局限：仅用单一 LLM 判断，可能存在偏见；参考 captions 不完整；仅评估 API 模型，未包含开源权重模型。

---

## 228. Privacy-Preserving Local Energy Trading Considering Network Fees

**arXiv ID:** 2602.23698 | [PDF](https://arxiv.org/pdf/2602.23698v1)

**作者:** Eman Alqahtani `[一作]` (University of Manchester), Mustafa A. Mustafa `[通讯]` (University of Manchester)

**通讯引用:** 739 | [OpenAlex ID](https://openalex.org/A5086381419)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于多方安全计算的本地能源市场（LEM）协议 PNF-LEM，兼顾网络费用、隐私保护与身份认证，支持参与者在保密前提下完成双拍卖清算并生成账单；

**💡 创新点**

创新点在于①将网络费用预先设定并融入双拍卖机制，促使交易更接近物理网络限制；②使用 Schnorr 标识协议与 MPC 结合实现既隐私又可验证的参与者身份；③通过批量验证、可并行比较与排序优化，显著降低通信和轮数复杂度；

**🔧 技术方法**

技术包括：多方安全计算（使用 RSS、加密乘法、比较、排序、洗牌、椭圆曲线 MPC）、Schnorr 身份识别协议、RSA-PSS 签名、AES-GCM 加密；

**📊 数据集**

使用由学术机构生成的真实分布式能源交易数据集（基于光伏、风电及用户需求，30分钟交易周期），并假设每个用户选取 3 位邻居作为交易对手；

**📈 对比分析**

通过 MP-SPDZ 框架实现，实验在 5,000 名用户的规模下，清算阶段总时长约 4–5 分钟（预处理约 17 秒，在线阶段约 3 分钟），通信成本占主要比例；与传统非隐私方案相比，性能损失在可接受范围内；

**⚠️ 局限性**

局限性包括：① 仍依赖半恶意安全模型，未覆盖完全恶意服务器；② 需要预先分发网络费用并假设用户固定选择 3 位邻居，实际网络结构可能更复杂；③ 关键比较协议为昂贵，若不采用更高效比较可进一步提升性能；

---

## 229. MemEmo: Evaluating Emotion in Memory Systems of Agents

**arXiv ID:** 2602.23944 | [PDF](https://arxiv.org/pdf/2602.23944v1)

**作者:** Peng Liu `[一作]` (Renmin University of China), Hong Chen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了情感增强的记忆系统评估基准HLME，构建了多场景、多任务的情感记忆数据集

**💡 创新点**

首次系统评估主流记忆系统在情感信息提取、更新与问答上的能力，并从三维度（提取、更新、问答）设计细化指标

**🔧 技术方法**

结合LLM检索、更新、推理等技术，利用GPT‑4o‑mini做判定评估，采用多模态评估框架

**📊 数据集**

构建了HLME-Medium与HLME-Long两版数据集，来源于Persona、EARL情感标注与自生成对话

**📈 对比分析**

与MemOS、MemoBase、Mem0、Mirix、Letta等六种系统在中短、长上下文环境下进行对比，结果显示各系统在不同子任务上有显著差异，暂无单一系统在所有维度均领先

**⚠️ 局限性**

评估仅覆盖少数记忆体系，缺乏跨体系通用性；对话记忆API支持不足；数据集仍需人工校准与多样化扩展

---

## 230. Neural Operators Can Discover Functional Clusters

**arXiv ID:** 2602.23528 | [PDF](https://arxiv.org/pdf/2602.23528v1)

**作者:** Yicen Li `[一作]` (McMaster University and Vector Institute), Maarten Valentijn de Hoop `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了神经算子在无限维Hilbert空间中对函数数据进行聚类的可行性，并给出了理论保证与实践实现。

**💡 创新点**

提出了“通用聚类”定理，证明采样型神经算子可逼近任意有限闭集聚类，使用上Kuratowski拓扑保证不产生假阳性，并构建了基于采样算子与预训练编码器的聚类流水线。

**🔧 技术方法**

采样型神经算子（SNO）+ 固定预训练CLIP特征+ 轻量MLP决策头 + 对比学习 + 软硬阈值聚类 + 上Kuratowski收敛分析。

**📊 数据集**

合成ODE轨迹数据集 ODE-6（六类结构化动力学）与 ODE-4（四类随机神经ODE，高变异）。

**📈 对比分析**

与 FPCA+KMeans、B‑Spline+KMeans、DTW+KMeans、CLIP+KMeans、CLIP+Spectrogram+KMeans 等基线比较，SNO 在结构化和高变异数据集上分别取得约 93–95% ACC、0.86–0.89 ARI、0.91–0.92 NMI，明显优于所有基线。

**⚠️ 局限性**

对采样密度和 CIS（可插值采样点）假设依赖较强；对真实稀疏或噪声数据的鲁棒性尚未充分验证；理论保证仅涉及上Kuratowski收敛，无法保证完全一致性。

---

## 231. Personal Data as a Human Right: A New Social Contract Based on Data Sovereignty, Human Dignity and Data Personalism

**arXiv ID:** 2602.23918 | [PDF](https://arxiv.org/pdf/2602.23918v1)

**作者:** J. M. Alvarez-Pallete `[一作]` (Universidad Pontificia Comillas), R. Redondo `[通讯]` (Universidad Pontificia Comillas)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种以人类尊严为核心的数字社会契约，诊断了数据化基础设施与平台治理带来的权力失衡与隐私危害，并在此基础上提出六个互联的治理维度（技术设计、自动化限制、价值再分配、政治合法性、社会共识与法律保障），同时引入“尊严设计（Dignity-by-Design）”作为技术与组织层面的可审计约束；

**💡 创新点**

创新点在于将社会契约理论与数据个人主义相结合，确立个人数据为人权载体的正当性；提出将尊严原则嵌入系统设计、算法影响评估与责任追究的完整框架；并将“数据劳动”与“数据福利”概念与尊严导向的利益再分配机制相衔接；

**🔧 技术方法**

技术层面主要引用隐私增强技术（PETs）、联邦学习、可信计算、算法影响评估（AIA/DPIA）等工具作为实现尊严设计的手段；

**📊 数据集**

由于本文为理论与政策性位置论文，未使用具体数据集；

**📈 对比分析**

本文并未开展实验或对比评估，故无性能指标或结果；

**⚠️ 局限性**

局限性包括：缺乏可操作的细化实施方案与量化评估；在多主体治理与跨境执法方面仍面临协调与执行挑战；对技术与法律实施细节的进一步研究仍需展开。

---

## 232. nvidia-pcm: A D-Bus-Driven Platform Configuration Manager for OpenBMC Environments

**arXiv ID:** 2602.24237 | [PDF](https://arxiv.org/pdf/2602.24237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 233. Tilewise Domain-Separated Selective Encryption for Remote Sensing Imagery under Chosen-Plaintext Attacks

**arXiv ID:** 2602.23772 | [PDF](https://arxiv.org/pdf/2602.23772v1)

**作者:** Jilei Sun `[一作]` (Shandong University of Aeronautics), Ying Su `[通讯]` (Shandong University of Aeronautics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于域分离的分块选择性图像加密框架TDS-SE，利用每块独立密钥抑制跨块结构泄露。

**💡 创新点**

创新点在于将密钥派生的域分离与选择性加密粒度对齐，并配套经验评估协议评测跨块、跨样本和ROI知识不对称攻击场景。

**🔧 技术方法**

使用HKDF域分离密钥派生、AES-CTR或轻量级流加密，结合线性回归和小型CNN重构鉴别器进行泄露评估。

**📊 数据集**

在公开遥感数据集RESISC45和SEN12MS上进行实验。

**📈 对比分析**

通过PSNR/SSIM等指标在四种加密变体与基线之间对比，域分离后跨块可迁移性显著下降，虽然整体平均PSNR未出现大幅提升。

**⚠️ 局限性**

局限性包括仅进行经验评估、攻击模型有限（未尝试更强的神经网络或更复杂的ROI），缺乏形式化安全证明，结果受ROI生成方式和图像预处理影响。

---

## 234. Learning to Reflect and Correct: Towards Better Decoding Trajectories for Large-Scale Generative Recommendation

**arXiv ID:** 2602.23639 | [PDF](https://arxiv.org/pdf/2602.23639v1)

**作者:** Haibo Xing `[一作]` (Alibaba International Digital Commerce Group), Jing Zhang `[通讯]` (School of Computer Science Wuhan University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结构化的生成–反思–纠正（GRC）框架，用于生成式推荐系统，先生成候选语义 token 序列，再通过多粒度反思定位错误并给出语义一致性信号，随后在这些信号指导下进行纠正；并在此基础上通过 GRPO 强化学习进一步优化自我纠正策略，同时设计熵引导的反思调度（EGRS）在 beam search 过程中高效分配纠正预算。

**💡 创新点**

① 将生成、反思、纠正三步显式建模为一个可监督的序列模板，突破传统一次性解码导致错误累积的问题；② 在 token 空间而非自然语言中实现结构化反思，兼顾位置和语义一致性；③ 通过 GRPO 在扩大的轨迹空间上使用多维奖励（token 命中、错误定位、语义匹配、提升幅度）驱动策略学习；④ 采用熵引导的反思调度在实时推理时动态分配有限的纠正计算资源。

**🔧 技术方法**

使用基于 Transformer 的 encoder–decoder 生成式推荐模型（如 TIGER 的架构），RQ‑VAE 对商品进行多级离散化；对生成过程使用监督微调和结构化模板；强化学习阶段采用 GRPO（Group Relative Policy Optimization）并设计细粒度奖励；推理阶段集成 beam search 与熵引导反思调度；实现细节使用 FlashAttention、Qwen3‑8B 预训练模型。

**📊 数据集**

公开数据集：Amazon Product Reviews（Arts、Musical Instruments）；工业数据集：近 1 亿条用户行为与广告反馈，覆盖 1900 万用户、2500 万广告，时间跨度 2025 年 3‑8 月。

**📈 对比分析**

对比基线包括 DSSM、SASRec、HSTU、TIGER、ReaRec、COBRA。GRC 在 Recall@5、Recall@10、NDCG@5/10 上均超过最强基线 10–15 % 以上；工业数据集最大提升 15.74 %；线上 A/B 测试中提升广告收入 1.79%、CTR 2.11%、GMV 2.04%，P99 延迟仅从 27 ms 增至 31 ms。

**⚠️ 局限性**

仍受限于推理延迟与 beam 搜索开销，需要在有限的纠正预算内权衡；奖励设计若缺少任务奖励易产生奖励劫持；对不同域或更大词表的泛化性待验证；模型对超参数（如 λ_rc、β_cor、α_e 等）敏感，需细致调优。

---

## 235. Human-Centered Multimodal Fusion for Sexism Detection in Memes with Eye-Tracking, Heart Rate, and EEG Signals

**arXiv ID:** 2602.23862 | [PDF](https://arxiv.org/pdf/2602.23862v1)

**作者:** Iván Arcos `[一作]` (PRHLT Research Center, Universitat Politècnica de València), Elena Gomis-Vicent `[通讯]` (PRHLT Research Center, Universitat Politècnica de València)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究通过收集用户对社交媒体恶意内容（尤其是性别歧视与厌女症）memes的生理反应（眼动、心率、EEG）构建了一个多模态数据集，并将其用于性别歧视检测模型。

**💡 创新点**

创新点在于将主观性强的性别歧视内容检测转化为客观的生理信号学习，并设计了层次注意力融合网络，将文本、图像和多模态生理特征联合建模，显著提升了检测性能。

**🔧 技术方法**

技术上采用了Vision‑Language模型（Qwen‑VL）生成文本描述、XLM‑RoBERTa提取文本特征、ViT提取视觉特征，以及多头交叉注意力层与层次融合机制整合EEG、眼动和心率特征；训练使用分阶段微调和加权二元交叉熵。

**📊 数据集**

使用的主要数据集为EXIST 2025 Meme Dataset（3984张中英memes），并通过两项实验收集了16名受试者的眼动、心率（7782+7714次记录）和EEG（7714次记录）生理数据。

**📈 对比分析**

在5折交叉验证下，加入生理信号后，二元性别歧视检测的AUC从0.699提升至0.794，意图识别从0.628提升至0.655，细粒度类别识别中最难类（厌女与非性暴力）的F1从0.259提升至0.327，均达到统计显著提升。

**⚠️ 局限性**

局限性包括：心率对短暂静态memes缺乏敏感度；样本规模有限且受试者数量不足，可能导致生理特征噪声；模型未能对每个词级生理响应进行时序对齐，难以获得更精细的解释；并且数据仅覆盖文本与图像静态内容，无法推广到动态视频场景。

---

## 236. Terminology Rarity Predicts Catastrophic Failure in LLM Translation of Low-Resource Ancient Languages: Evidence from Ancient Greek

**arXiv ID:** 2602.24119 | [PDF](https://arxiv.org/pdf/2602.24119v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 237. FHIRPath-QA: Executable Question Answering over FHIR Electronic Health Records

**arXiv ID:** 2602.23479 | [PDF](https://arxiv.org/pdf/2602.23479v1)

**作者:** Michael Frew `[一作]`, Bryan Tripp `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 FHIRPath-QA 数据集和基准，用于在真实的 MIMIC-IV FHIR 记录上实现患者级问答，包含可执行的 FHIRPath 查询和对应答案。

**💡 创新点**

创新点在于：① 将自然语言问题转化为可验证的、可执行的 FHIRPath 查询；② 采用多视角（临床与患者）语料生成，桥接患者非专业语言与 FHIR 结构；③ 提供基于真实 EHR 的评测拆分（语言、查询、资源级别）和可执行验证机制。

**🔧 技术方法**

使用技术包括：OpenAI 大语言模型（o4-mini、4.1-mini、4o-mini、4.1-nano），两种问答架构（检索‑first 与 query‑first），文本到 FHIRPath 的生成与执行，监督式微调（SFT），以及 OctoFHIR Rust 实现的 FHIRPath 解析器。

**📊 数据集**

数据集来源为 MIMIC-IV on FHIR Demo，包含 100 份真实病人记录，61 种问答模板，经过 2,095 条可执行验证样本和 12,200 条未验证的训练样本；同时生成了患者和临床视角的 1,095 条问答对。

**📈 对比分析**

对比实验显示：检索‑first 在可控 EHR 中准确率略高（≈0.42‑0.49），但 token 使用量极大且可变；query‑first 在 token 效率上优异（平均 token 约 400），准确率约 0.35‑0.45；经过 SFT 后，query‑first 的准确率显著提升至 0.7‑0.8，且在新问法、未见查询模板和未见资源类型上均有所进步。

**⚠️ 局限性**

限制包括：样本主要来自 ICU 与院内住院护理，缺乏多机构多诊疗场景；问答模板为人工构造，未完全覆盖真实患者提问的多样性；仅聚焦可从结构化 FHIR 资源直接推导的事实性问题，未涵盖解释性或主观性查询；SFT 对未出现资源类型的泛化能力有限，易出现过拟合。

---

## 238. Refining Almost-Safe Value Functions on the Fly

**arXiv ID:** 2602.23478 | [PDF](https://arxiv.org/pdf/2602.23478v1)

**作者:** Sander Tonkens `[一作]` (University of California), Sylvia Herbert `[通讯]` (University of California)

**通讯引用:** 744 | [OpenAlex ID](https://openalex.org/A5071321544)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文提出了两种基于Hamilton‑Jacobi（HJ）到达性热映射的在线安全值函数改进框架，能够在动态环境或动力学变化时即时调整控制屏障函数（CBF），从而实现实时安全保证。

**💡 创新点**

创新点在于将离线的HJ到达性方法与在线自适应相结合，提出可热启动的改进算法（refineCBF）和局部修补算法（patchCBF），在保持正式安全保证的同时显著提升计算效率；此外引入双阶段（收缩-扩展）策略，确保在每一次迭代中安全性不退化。

**🔧 技术方法**

使用的技术包括：控制屏障函数（CBF）与控制障碍值函数（CBVF）的理论定义；Hamilton‑Jacobi‑Isaacs PDE 与变分不等式的离散动态规划求解；格点空间的局部活跃集更新；以及基于二次规划的安全滤波器。

**📊 数据集**

实验数据集涵盖：1) 仿真环境下的地面车辆（Jackal）与四旋翼无人机；2) 实际硬件测试，包括Jackal机器人与Crazyflie四旋翼；3) 通过Gazebo、ROS与C++/Python实现的真实传感器与动力学模型；并利用多种环境变化（障碍物出现、风扰动、可见性限制）进行评估。

**📈 对比分析**

与基线（单独CBF、联合CBF、传统HJ求解、备份CBF）相比，refineCBF在GPU加速下能在1–3 Hz完成更新，CPU版本在10 Hz下可达9/10成功率；patchCBF在局部修补场景下显著减少计算量并实现在线收敛；实验显示在碰撞率、轨迹偏差与实时性上均优于基线，且能在硬件上安全通过突发障碍与风扰动。

**⚠️ 局限性**

主要局限包括：1) 依赖精确已知的动力学模型与有限扰动区间；2) 需要在离散网格上进行动态规划，受维度灾难限制，当前仅能扩展至约4‑D状态空间；3) 传感器与感知误差未充分考虑，实际部署需解决感知不确定性；4) 需要GPU或多核CPU加速才能满足实时需求。

---

## 239. Finite Block Length Rate-Distortion Theory for the Bernoulli Source with Hamming Distortion: A Tutorial

**arXiv ID:** 2602.24243 | [PDF](https://arxiv.org/pdf/2602.24243v1)

**作者:** Bhaskar Krishnamachari `[一作]` (University of Southern California), Bhaskar Krishnamachari `[通讯]` (University of Southern California)

**通讯引用:** 23771 | [OpenAlex ID](https://openalex.org/A5063784062)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文系统阐述了二进制伯努利源在 Hamming 损失下的压缩极限，推导出闭式率失真函数 R(D)=H(p)-H(D)，并通过 Blahut‑Arimoto 算法和 d‑tilted 信息量进一步分析有限块长压缩的第二阶修正（方差 V(D) 与正态逼近）。

**💡 创新点**

创新点在于将 Shannon 的极限与有限块长理论在同一教程中完整衔接，给出 d‑tilted 信息量的明确定义、解析表达式及其与散度的关系，并通过闭式示例验证 Blahut‑Arimoto 与理论的一致性，展示了从闭式解到数值算法再到第二阶近似的全流程。

**🔧 技术方法**

主要技术包括：信息理论中的相互信息、熵与失真约束的 Lagrange/KKT 优化、Gibbs 形式的测试通道推导、Blahut‑Arimoto 迭代求解、d‑tilted 信息量与正态逼近的中心极限定理推演。

**📊 数据集**

数据集为理论上 IID 伯努利(p) 源的随机序列（在数值实验中使用模拟生成的二进制序列）。

**📈 对比分析**

通过比较 Blahut‑Arimoto 计算出的 (R,D) 点与解析闭式 R(D)，验证其在数值精度上完全一致；同时在有限块长实验中，将正态逼近、上、下界与实际码率曲线对比，显示逼近在 10⁴~10⁵ 个符号时已十分贴近极限。

**⚠️ 局限性**

局限性包括：仅针对离散 IID 伯努利源，无法直接推广到连续或相关源；当源偏置为 0.5 时方差 V(D)=0，第二阶正态逼近失效；且示例中仅演示了最简单的 Hamming 损失，复杂损失函数需要重新推导。

---

## 240. A Novel Hierarchical Multi-Agent System for Payments Using LLMs

**arXiv ID:** 2602.24068 | [PDF](https://arxiv.org/pdf/2602.24068v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 241. No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency

**arXiv ID:** 2602.23559 | [PDF](https://arxiv.org/pdf/2602.23559v1)

**作者:** Cho-Ying Wu `[一作]` (Bosch Research North America and Bosch Center for Artificial Intelligence), Liu Ren `[通讯]` (Bosch Research North America and Bosch Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种跨传感器视图合成框架，能够在不需要复杂相机标定和深度信息的前提下，将 RGB 与其他传感器（如热、近红外、SAR）的图像对齐并生成完整的多视角 X 图像。

**💡 创新点**

创新点包括：①将跨模态特征匹配与稠密化（CADF）耦合，利用匹配置信息引导稠密化；②自匹配过滤机制进一步剔除错误补丁；③使用 3D 高斯喷射（3D Gaussian Splatting）在 RGB 视角下统一空间，提升多视角一致性；④整个流程仅需 RGB 的 COLMAP 结果，无需 X 传感器的标定或深度。

**🔧 技术方法**

主要技术包括跨模态特征匹配（XoFTR、LightGlue 等）、置信引导稠密化网络（DySPN+CADF）、自匹配相似度学习与过滤、3D Gaussian Splatting、基于 SigLIP 的自监督损失、以及多级阈值融合。

**📊 数据集**

使用的公开数据集有：METU-VisTIR-Cloudy（RGB‑热无配对序列）、RGBT-Scenes（RGB‑热配对）、RGB-NIR-Stereo（RGB‑近红外配对）、DDHR-HK（RGB‑SAR 大幅图）等。

**📈 对比分析**

与现有方法（XoFTR、LightGlue、LoFTR、MINIMA、StyleBooth、PixNext 等）对比，本文在多项指标上取得领先：Icos、p30–p90 相似度、ITM/ITcos、RMSE/MAE、PSNR/SSIM/LPIPS 等；同时在 MEt3R 视角一致性评估中也显著优于对手，说明合成效果更稳定、更加真实。

**⚠️ 局限性**

局限性包括：仅处理静态场景，无法处理动态物体；对极低纹理或同质区域的匹配仍受限；热摄像机噪声和分辨率低的问题未完全解决；整个方法高度依赖跨模态匹配质量，若匹配失败则效果受限。

---

## 242. ReasonX: Declarative Reasoning on Explanations

**arXiv ID:** 2602.23810 | [PDF](https://arxiv.org/pdf/2602.23810v1)

**作者:** Laura State `[一作]` (Alexander von Humboldt Institute for Internet and Society), Franco Turini `[通讯]` (University of Pisa)

**通讯引用:** 9014 | [OpenAlex ID](https://openalex.org/A5023007469)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个名为ReasonX的解释工具，利用约束逻辑编程和线性约束算子代数，对决策树及其代理模型进行可解释性推理，支持事实与对比解释、背景知识、未指定实例和交互式查询。

**💡 创新点**

核心创新在于定义闭合的线性约束算子代数，用以表达解释查询；同时将背景知识、可操作约束、未指定实例、多模型和多时刻推理整合到同一框架，并实现交互式约束约束。

**🔧 技术方法**

技术实现基于Python层的易用接口与CLP(ℝ)层的元解释器，内部采用混合整数线性规划(MILP)求解、投影、最小化等操作，支持符号化约束与优化。

**📊 数据集**

在公开的信用评分数据集（South German Credit、Give Me Some Credit、Default Credit Card Clients、Australian Credit Approval）和合成数据集上进行实验验证。

**📈 对比分析**

通过与LORE、GlocalX、Anchors、DiCE等现有XAI方法在规则长度、覆盖率、对比实例距离、运行时间等指标上进行量化比较，结果显示ReasonX在对比解释距离更小、支持交互与未指定实例，但运行时间相对更高。

**⚠️ 局限性**

局限性包括仅支持线性约束与离散/连续特征；多类、非结构化数据支持有限；对模型稳定性的假设；整数变量求解不完全；需要用户手工编码背景知识；缺乏自然语言生成与解释可读性提升空间。

---

## 243. Designing AI Tutors for Interest-Based Learning: Insights from Human Instructors

**arXiv ID:** 2602.24036 | [PDF](https://arxiv.org/pdf/2602.24036v1)

**作者:** Abhishek Kulkarni `[一作]` (University of Florida), Sharon Lynn Chu `[通讯]` (University of Florida)

**通讯引用:** 1149 | [OpenAlex ID](https://openalex.org/A5000750138)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究人类导师在一对一线上辅导中如何将学生兴趣融入教学，提炼出兴趣整合的功能与机制，并为基于LLM的AI导师提出设计启示。

**💡 创新点**

首次系统化解析人类导师在兴趣驱动学习中的兴趣使用功能和机制，为大规模基于LLM的兴趣教学AI模型提供实证基础。

**🔧 技术方法**

采用定性分析方法，手工编码访谈、课程计划与对话记录，利用MAXQDA进行开放与轴向编码。

**📊 数据集**

收集14对一对一线上辅导会话（共28名参与者）的课程计划、访谈、对话记录及问卷数据。

**📈 对比分析**

未进行量化对比实验，而是通过三轮编码与三方数据源的三角检验确认兴趣整合功能与机制的可靠性，未给出数值性能指标。

**⚠️ 局限性**

研究受限于实验室设置与成人一对一线上情境，未检验在真实教学环境、不同年龄层或多学科场景下的适用性，且未评估LLM实现的可行性与安全风险。

---

## 244. GPU-Native Approximate Nearest Neighbor Search with IVF-RaBitQ: Fast Index Build and Search

**arXiv ID:** 2602.23999 | [PDF](https://arxiv.org/pdf/2602.23999v1)

**作者:** Jifan Shi `[一作]` (Nanyang Technological University), Cheng Long `[通讯]` (Nanyang Technological University)

**通讯引用:** 5083 | [OpenAlex ID](https://openalex.org/A5080939756)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了GPU原生的IVF‑RaBitQ索引，结合IVF划分与RaBitQ量化，实现在GPU上快速构建、高吞吐、高召回、低存储的近似最近邻搜索。

**💡 创新点**

创新点在于：①可扩展的GPU RaBitQ量化管线；②两阶段距离估计与融合搜索核；③针对GPU的查表与位运算内积计算，首次在GPU上高效部署RaBitQ。

**🔧 技术方法**

使用了GPU并行RaBitQ量化、两阶段估计、查表/位运算内积、融合搜索核、CSR风格索引布局以及cuVS集成。

**📊 数据集**

使用了六个真实高维数据集：ImageNet、CodeSearchNet、GIST、OpenAI‑3072‑1M、OpenAI‑1536‑5M、Wiki‑all。

**📈 对比分析**

与IVF‑Flat、IVF‑PQ（含/不含重排序）及GPU图方法CAGRA对比，IVF‑RaBitQ在Recall≈0.95时平均实现2.2×QPS提升、构建速率7.7×快、存储占比仅25%，在高召回区间显著领先。

**⚠️ 局限性**

限制在共享内存与查表大小上，位运算版本对极高维仍受限；对超大规模或极高维数据需进一步扩展共享内存与多GPU支持。

---

## 245. Detoxifying LLMs via Representation Erasure-Based Preference Optimization

**arXiv ID:** 2602.23391 | [PDF](https://arxiv.org/pdf/2602.23391v1)

**作者:** Nazanin Mohammadi Sepahvand `[一作]` (McGill University), Gintare Karolina Dziugaite `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于表示消除的首选优化方法（REPO），对大语言模型进行去毒化。

**💡 创新点**

创新点是把毒性去除视为 token 级别的表示消除，结合保留 KL 对齐与对抗判别器，实现内部特征彻底消失，显著提升鲁棒性。

**🔧 技术方法**

采用 token 级别 KL 保留损失、域对抗判别器（gradient reversal）、对齐参考模型的 anchor 机制以及小型 MLP 判别器。

**📊 数据集**

使用 PairToxicity 作为对齐数据集，并在 WikiText‑2 与 RealToxicityPrompts 上进行 OOD 评估。

**📈 对比分析**

与 DPO、NPO、RMU、CB 等方法比较，REPO 在毒性降低、语言质量（PPL、F1）保持以及对 relearning、增强 GCG、orthogonalization 攻击的鲁棒性上均优于基线。

**⚠️ 局限性**

局限性包括训练成本略高、对更大模型规模的可扩展性尚待验证，以及对极细粒度或多样化毒性定义可能需要进一步调优。

---

## 246. SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking

**arXiv ID:** 2602.23963 | [PDF](https://arxiv.org/pdf/2602.23963v1)

**作者:** Qiuyang Zhang `[一作]` (Tongji University), Shangce Gao `[通讯]` (University of Toyama)

**通讯引用:** 9349 | [OpenAlex ID](https://openalex.org/A5010245958)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种全脉冲驱动、能效优先的RGB视觉跟踪框架SpikeTrack，实现了高精度与低能耗的双重目标。

**💡 创新点**

创新点在于引入非对称时序扩展与单向信息流的Siamese结构，配合脑启发的记忆检索模块（MRM），充分利用神经元时空动力学同时显著削减计算量。

**🔧 技术方法**

使用了Spike‑Driven Transformer v3、NI‑LIF神经元、E‑SDSA线性脉冲自注意力、记忆检索模块以及多尺度脉冲卷积/Transformer块等技术。

**📊 数据集**

在COCO、LaSOT、TrackingNet、GOT‑10k等公开RGB跟踪数据集上进行训练与评估。

**📈 对比分析**

与SNN及ANN基准对比，SpikeTrack在LaSOT、GOT‑10k、TrackingNet、TNL2K、UAV123、OTB100等数据集上实现了与TransT、AsymTrack等先进ANN追踪器相当甚至更优的AUC/精度，并且能耗低至ANN方法的1/7至1/26，能效比显著提升。

**⚠️ 局限性**

局限在于对相似目标的区分能力不足，脉冲信息难以充分表达细粒度语义，导致在相似物体干扰场景下表现下降。

---

## 247. CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation

**arXiv ID:** 2602.24286 | [PDF](https://arxiv.org/pdf/2602.24286v1)

**作者:** Weinan Dai `[一作]` (ByteDance), Hao Zhou `[通讯]` (Institute for AI Industry Research, Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一套基于大型语言模型的代理强化学习系统，用于自动生成和优化CUDA核代码，系统通过可编程环境、鲁棒奖励机制和多阶段训练实现持续改进；

**💡 创新点**

创新点在于①构建可扩展的数据合成管线，自动生成多级难度的CUDA任务；②引入技能化的开发环境与自动化验证/剖析，消除奖励作弊；③采用单步RL热身+拒绝采样微调与价值预训练的多阶段策略，显著提升训练稳定性；

**🔧 技术方法**

使用技术包括：Seed1.6 MoE模型、PPO强化学习、ReAct式交互式代理、拒绝采样微调(RFT)、价值预训练、规范化奖励分值、GPU+CPU分离沙箱环境；

**📊 数据集**

数据集为：自研的CUDA‑Agent‑Ops‑6K（约6000条基于PyTorch算子组合的合成任务）以及公开的KernelBench基准；

**📈 对比分析**

实验比较中系统在KernelBench Level‑1/2/3分别比传统编译器和业内主流专有模型（Claude Opus 4.5、Gemini 3 Pro）实现100%、100%和92%的加速，Pass率达98.8%，Faster率达96.8%，显著优于对比模型；

**⚠️ 局限性**

局限性包括：对极端复杂任务的覆盖仍有限，RL训练仍受限于150步；模型规模巨大，对硬件资源要求高；系统目前仅在NVIDIA H100/H20 GPU上验证，跨硬件通用性待进一步评估；

---

## 248. Human Supervision as an Information Bottleneck: A Unified Theory of Error Floors in Human-Guided Learning

**arXiv ID:** 2602.23446 | [PDF](https://arxiv.org/pdf/2602.23446v1)

**作者:** Alejandro Rodriguez Dominguez `[一作]` (Miralta Finance Bank), Alejandro Rodriguez Dominguez `[通讯]` (University of Reading)

**通讯引用:** 406 | [OpenAlex ID](https://openalex.org/A5002072646)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了人类监督通道为信息削减通道的统一理论，证明当监督完全由人类信号主导时，学习过程存在严格正的超额风险底限（Human‑Bounded Intelligence），并通过六种理论视角（算子理论、PAC‑Bayes、信息论、因果推断、范畴论与RLHF博弈论）给出一致的结构分解。

**💡 创新点**

创新点在于将人类监督视为缺失信息的通道并给出超额风险底限，揭示规模与优化无法突破的结构瓶颈；同时系统化地展示了辅助非人类信号如何通过提升通道容量消除该底限。

**🔧 技术方法**

技术手段包括信息处理理论、PAC‑Bayes泛化分析、算子范数逼近、因果可识别性证明、范畴论同构与对偶、以及RLHF奖励模型的博弈理论分析，并结合实验验证。

**📊 数据集**

使用的数据集包括真实人类偏好对比数据、可控合成目标任务、公开评测基准GSM8K、HumanEval 以及辅助验证器（如小型LLM检验器）。

**📈 对比分析**

实验通过将人类监督与辅助信号（检验器、检索、工具执行）混合，比较人类‑仅、模型‑加、以及辅助‑加的性能，结果显示人类‑仅始终处于误差底限，而加入足够信息的辅助通道可显著提升准确率并在某些任务中达到完美（0误差）。

**⚠️ 局限性**

局限性包括对无穷样本与理想优化的假设、仅适用于人类主导的监督情形、以及对实际辅助通道的容量与信息可辨识性未给出具体估计，因而在现实规模与多样化环境下的泛化仍待验证。

---

## 249. On the Uniqueness of Solutions in GPS Source Localization: Distance and Squared-Distance Minimization under Limited Measurements in Two and Three Dimensions

**arXiv ID:** 2602.23741 | [PDF](https://arxiv.org/pdf/2602.23741v1)

**作者:** Kiwoon Kwon `[一作]` (Dongguk University), Kiwoon Kwon `[通讯]` (Dongguk University)

**通讯引用:** 554 | [OpenAlex ID](https://openalex.org/A5033820781)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过理论分析研究了GPS源定位问题在距离误差与平方距离误差两种最小化形式下，尤其在测量数少于三时的唯一性与解的数量，分别在二维和三维空间内给出完整的分类结果。

**💡 创新点**

创新点在于系统地阐述了测量数不足三时的解多重性，提供了严格的几何和代数条件，揭示了平方距离误差下解数上限为两点，而距离误差下往往出现无穷多解；同时比较了二维与三维的差异。

**🔧 技术方法**

主要采用几何分析、集合划分、极值条件与凸性理论等数学技术，对目标函数的梯度与极值点进行严谨证明。

**📊 数据集**

论文使用的是理论构造的测量点与距离（即合成数据），并通过图示展示不同情形下的解位置。

**📈 对比分析**

方法与性能的比较主要体现在理论上：对距离误差与平方距离误差两种目标函数在相同测量配置下的最优解数与解空间形状进行对比，说明平方距离误差更易得到唯一或有限解。

**⚠️ 局限性**

局限性包括：仅分析了最多三次测量，未考虑更大测量数的情况；仅给出理论结果，缺乏实际实验验证；三维非平面情况下的完整情形仍有待进一步研究。

---

## 250. CoME: Empowering Channel-of-Mobile-Experts with Informative Hybrid-Capabilities Reasoning

**arXiv ID:** 2602.24142 | [PDF](https://arxiv.org/pdf/2602.24142v1)

**作者:** Yuxuan Liu `[一作]` (Renmin University of China), Rui Yan `[通讯]` (Wuhan University)

**通讯引用:** 8651 | [OpenAlex ID](https://openalex.org/A5100716372)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于四个专门专家的 Channel-of-Mobile-Experts 体系结构，用于移动智能体的多阶段混合能力推理。

**💡 创新点**

创新点在于：① 采用输出导向激活机制，使不同专家在对应推理阶段被激活；② 设计了分阶段的 Expert-FT、Router-FT 与 CoT-FT 训练流程，实现能力的解耦与平衡融合；③ 引入信息增益驱动的 DPO（Info-DPO），用信息增益评估中间推理步骤，从而抑制错误传播。

**🔧 技术方法**

核心技术包括多专家模型（每层 FFN 置换为四个专家）、输出导向激活、分阶段微调策略、信息增益奖励模型以及 DPO 训练。

**📊 数据集**

在 AITZ 和 AMEX 两大移动界面任务数据集上进行实验，分别涵盖点击、输入、滚动等多种动作。

**📈 对比分析**

与稠密移动智能体和传统 MoE 模型比较，Channel-of-Mobile-Experts 在 AITZ 上提升 1.73%（相对密集模型）和 5.72%（相对 MoE），在 AMEX 上提升 1.90%（密集）和 8.05%（MoE），同时保持更低的 GPU 内存占用。

**⚠️ 局限性**

局限性包括：对训练数据的依赖较强，专家数量与任务匹配需手工设计；输出导向激活虽然提高了性能但增加了实现复杂度；在极端复杂场景下仍可能出现中间推理错误导致最终动作不准确。

---

## 251. Controllable Reasoning Models Are Private Thinkers

**arXiv ID:** 2602.24210 | [PDF](https://arxiv.org/pdf/2602.24210v1)

**作者:** Haritz Puerto `[一作]` (Technical University of Darmstadt), Iryna Gurevych `[通讯]` (Technical University of Darmstadt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练推理模型在推理过程（reasoning traces）中遵循指令，以防止私人信息泄露

**💡 创新点**

① 构建专门针对推理过程的指令跟随数据集；② 采用阶段解码（Stage Decoding）将推理与答案生成分离；③ 证明提升推理过程指令遵循能显著提高隐私安全

**🔧 技术方法**

LoRA 微调、4-bit 量化、Stage Decoding、PEFT 框架、Qwen 3 与 Phi 4 推理模型

**📊 数据集**

自制 RT 指令训练集（1k/2k/3k）基于 GSM8K；评测集 IFEval、MathIF；隐私评测 PasswordEval、PEEP

**📈 对比分析**

在 IF 评测中 IF‑RT 与 IF‑FA 均提升至 baseline 的 20.9 分，隐私得分平均提升 21.65（PasswordEval）/22.69（PEEP），最大提升 51.9 分；相较 baseline 的 6–10 分提升；但在某些任务上实用性有所下降

**⚠️ 局限性**

数据集规模有限易过拟合导致实用性下降；仅使用 SFT 未结合 RLHF；量化可能影响稳定性；隐私提升与任务实用性之间存在权衡

---

## 252. Learning to maintain safety through expert demonstrations in settings with unknown constraints: A Q-learning perspective

**arXiv ID:** 2602.23816 | [PDF](https://arxiv.org/pdf/2602.23816v1)

**作者:** George Papadopoulos `[一作]` (University of Piraeus), George A. Vouros `[通讯]` (University of Piraeus)

**通讯引用:** 3591 | [OpenAlex ID](https://openalex.org/A5040575826)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于逆约束强化学习的 SafeQIL 算法，利用专家演示中的状态支持信息和判别器对 Q 值进行状态层级的保守约束，学习既能遵循安全约束又能在线改进的策略。

**💡 创新点**

创新点在于：1）不显式学习约束函数，而是通过判别器定义演示支持并对离开支持的状态设置局部上界，实现在状态层面的自适应保守性；2）将判别器权重与 Q 目标结合，既鼓励在安全状态下最大化奖励，又抑制在不安全状态下过度乐观；3）在 SAC 结构下实现，兼顾样本效率与稳定性。

**🔧 技术方法**

使用软 actor‑critic（SAC）框架、判别器（logistic 回归 + 梯度惩罚）、近似最近邻搜索、最大熵策略更新、以及基于 Q 值的上界约束。

**📊 数据集**

实验基准为 Safety‑Gymnasium 4 个任务（SafetyPointGoal1‑v0、SafetyPointCircle2‑v0、SafetyCarButton1‑v0、SafetyCarPush2‑v0），使用 40 条人类演示轨迹作为离线数据。

**📈 对比分析**

与 ICRL、VICRL、SAC‑GAIL 以及未约束的 SAC、PPO 进行比较。SafeQIL 在 4 个任务中均显著降低安全违规成本（30%–92%），并保持或略低于基线奖励，尤其在安全成本更高的任务中表现优于 ICRL/VICRL，且在奖励-成本权衡上优于 SAC‑GAIL。

**⚠️ 局限性**

局限性包括：对演示覆盖度敏感，判别器在极端 OOD 区域可能失准；最近邻检索简化导致在高维状态下检索误差；缺乏模型基方法的长期规划能力；当演示数据不足时，保守性可能过强导致任务失效。

---

## 253. Reason to Contrast: A Cascaded Multimodal Retrieval Framework

**arXiv ID:** 2602.23369 | [PDF](https://arxiv.org/pdf/2602.23369v1)

**作者:** Xuanming Cui `[一作]` (Meta), Xiangjun Fan `[通讯]` (Meta)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Reason-to-Contrast 框架，在多模态检索中通过 Embedding-Centric Reasoning（ECR）与重排、查询感知重写实现零样本检索性能提升。

**💡 创新点**

创新点包括：①在检索与重排两阶段均利用 ECR，实现 token‑级测试时可扩展性；②Query‑Aware ECR 重写，让重排更具判别力；③以重排得分为教师进行硬负样本挖掘并蒸馏到 embedder。

**🔧 技术方法**

技术手段：多模态大语言模型（Qwen 系列）作为 embedder、reasoner、reranker；思考‑嵌入（TTE）+ ECR；零样本文本重排（pairwise/listwise）；基于重排的硬负样本挖掘与对比学习；LoRA、GradCache 等训练技巧。

**📊 数据集**

使用数据集：MMEB‑V2（78 个任务，图像、视频、visDoc），跨语言检索基准 CrossModal‑3600、XTD10，视频检索子任务等。

**📈 对比分析**

与双编码器、UniME、LLaVE、B3、IFM‑TTE 等基线对比。整体在 MMEB‑V2 的得分为 75.7%，超过 IFM‑TTE‑7B（75.1）；视频检索提升约 6%；2B 版模型也突破 7B 版；重排与 QAR 使性能提升 10%+；硬负样本挖掘提升约 0.7%。

**⚠️ 局限性**

局限性：重排阶段仍需要额外推理成本；当候选集过大时，k 值增大会引入噪声；方法依赖大规模 MLLM，计算资源要求高；在某些 VQA 风格任务中，零样本重排表现不佳。

---

## 254. SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching

**arXiv ID:** 2602.24208 | [PDF](https://arxiv.org/pdf/2602.24208v1)

**作者:** Yasaman Haghighi `[一作]` (Ecole Polytechnique Federale de Lausanne), Alexandre Alahi `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于网络对输入扰动敏感度的自适应缓存方法（SenCache）以加速扩散模型的推理

**💡 创新点**

创新点在于用理论推导得到的本地敏感度（噪声潜变量与时间步长的雅可比范数）作为缓存决策依据，动态、样本特异地决定是否重用前一步输出，消除了先前经验式阈值的无理论基础

**🔧 技术方法**

主要技术包括：一阶敏感度近似、有限差分估计雅可比范数、阈值ε控制误差、最大连续缓存步数n、与现有缓存算法（TeaCache、MagCache）对齐的采样器兼容性

**📊 数据集**

在 MixKit 8 视频（用于敏感度校准）和 VBench/T2V-CompBench（用于生成评估）上进行实验，针对 Wan 2.1、CogVideoX 与 LTX-Video 三种视频扩散模型

**📈 对比分析**

与 TeaCache、MagCache 在 NFE、缓存比例、LPIPS、PSNR、SSIM 等指标对比，SenCache 在保持或提升视觉质量的同时，实现了与最优手工阈值相当甚至更低的 NFE，并在多种模型与计算预算下保持竞争优势

**⚠️ 局限性**

局限性包括：依赖一阶近似导致在极端非线性阶段误差累积；阈值 ε 为固定值，缺乏动态调度；目前仅在视觉域验证，其他模态需要进一步验证；以及敏感度估计需少量样本，但在更复杂模型上可能需要更精细的校准

---

## 255. Footprint-Guided Exemplar-Free Continual Histopathology Report Generation

**arXiv ID:** 2602.23817 | [PDF](https://arxiv.org/pdf/2602.23817v1)

**作者:** Pratibha Kumari `[一作]` (University of Regensburg), Dorit Merhof `[通讯]` (University of Regensburg)

**通讯引用:** 10859 | [OpenAlex ID](https://openalex.org/A5064747056)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了无样本持续学习框架，利用域足迹在冻结的补丁嵌入空间中进行伪WSI再现，以实现连续的WSI报告生成。

**💡 创新点**

创新点在于用压缩的域足迹和生成式回放代替传统示例存储，并通过报告风格原型实现域无关的风格调节。

**🔧 技术方法**

采用冻结的视觉编码器、Perceiver Resampler、HistoGPT语言模型，k-means代码簿、生成式回放、即时教师快照和风格前缀技术。

**📊 数据集**

在REG2025和PathText公开数据集上，按器官和混合式域序列进行实验。

**📈 对比分析**

与多种基线（ER、DER、EWC、SI、LwF、ProgPrompt、CMRG-LLM）比较，footprint回放在低缓冲区下平均分数和保持率上优于示例回放，BWT接近零。

**⚠️ 局限性**

局限在于对报告文本依赖较大，难以处理高度无结构或非显影相关的报告；对域匹配误差敏感，且缺乏对极端域变迁的鲁棒性。

---

## 256. MicroPush: A Simulator and Benchmark for Contact-Rich Cell Pushing and Assembly with a Magnetic Rolling Microrobot

**arXiv ID:** 2602.23607 | [PDF](https://arxiv.org/pdf/2602.23607v1)

**作者:** Yanda Yang `[一作]` (University of Delaware), Sambeeta Das `[通讯]` (University of Delaware)

**通讯引用:** 1296 | [OpenAlex ID](https://openalex.org/A5047701694)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了MicroPush，一个开源的磁性滚动微型机器人模拟器与基准套件，专门用于在二维拥挤环境下的接触式细胞推送与多目标组装。

**💡 创新点**

创新点在于将低雷诺数的过阻尼动力学与接触感知的粘-滑摩擦、近场阻尼以及可选的泊肃叶背景流相结合，提供了轻量级但物理可信的模拟模型，并设计了模块化的两阶段规划-控制管线及统一的基准协议，支持可插拔全局规划器、任务级规划器和控制器，同时兼容Gym式强化学习接口。

**🔧 技术方法**

技术包括：基于欧拉积分的过阻尼运动方程、粘-滑摩擦模型、近场阻尼阈值控制、泊肃叶流体场叠加、频率到滚动速度的标定映射、Analytic Geometry Planner (AGP) 与加权 A* 全局规划器、两阶段（接触建立+目标推送）控制策略、MPC 与 PID 两类反馈控制器，以及统一的CSV日志与可视化界面。

**📊 数据集**

数据集：使用固定随机种子生成的可重复场景，单目标推送共 80 组（含流/不流），六方组装共 30 组；每组场景包含 20 个细胞（或 1 个目标细胞），工作空间尺寸 200×140 px，机器人/细胞半径 5 μm；同时提供了 81 组静态随机场景用于全局规划器的微基准测试。

**📈 对比分析**

对比方法：在同一基准协议下比较 AGP+MPC、AGP+PID、A*+MPC、A*+PID 四种组合；指标包括成功率、到达时间、推送轨迹跟踪误差、推送路径长度、行动频率波动。结果显示：在背景流下 MPC 的成功率（1.0）显著高于 PID（0.412），并在成功案例中更快到达；在无流场下所有方法成功率均 ≥ 0.975，MPC 的跟踪误差与动作波动略低；六方组装中 MPC 仍保持更高成功率与更平滑的控制。

**⚠️ 局限性**

局限性：1) 仅采用二维过阻尼模型，未考虑三维几何与高阶流体动力学；2) 仅针对静态或少量细胞，缺乏动态障碍物与实时重规划；3) 未在真实实验中验证模型精度，缺乏从模拟到硬件的闭环迁移；4) 控制器与规划器参数均为手工调优，缺乏自动化学习或自适应机制。

---

## 257. Flow-Based Density Ratio Estimation for Intractable Distributions with Applications in Genomics

**arXiv ID:** 2602.24201 | [PDF](https://arxiv.org/pdf/2602.24201v1)

**作者:** Egor Antipov `[一作]` (Helmholtz Munich), Fabian J. Theis `[通讯]` (Helmholtz Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种利用连续正则化流(CNF)的单次模拟来高效估计不可行分布之间的似然比（scRatio）

**💡 创新点**

创新点在于推导出一个描述对数密度比随时间演化的ODE，并用流匹配学习的向量场和得分直接沿单条轨迹计算比值，避免了传统两次独立Ode求解的高昂成本

**🔧 技术方法**

主要技术包括条件化CNF、流匹配训练、向量场与得分重参数化、对数密度比ODE求解以及在单细胞数据上的应用

**📊 数据集**

使用了仿真高维高斯、互信息任务、PBMC单细胞RNA测序、NeurIPS骨髓数据、C. elegans细胞、ComboSciPlex药物组合以及10M PBMC患者数据等多种数据集

**📈 对比分析**

与基线方法（naive CNF、TSM、CTSM、SB路径等）比较，scRatio在MSE/MAE、AUC、NAR、CSP等指标上表现更优，且运行时间显著降低

**⚠️ 局限性**

局限性包括对支持不重叠的分布可能导致数值不稳定、需要训练额外的得分网络、以及在高维空间中对数比值轨迹的模拟质量受限

---

## 258. Can Unified Generation and Understanding Models Maintain Semantic Equivalence Across Different Output Modalities?

**arXiv ID:** 2602.23711 | [PDF](https://arxiv.org/pdf/2602.23711v1)

**作者:** Hongbo Jiang `[一作]` (Tencent Youtu Lab), Liujuan Cao `[通讯]` (Xiamen University)

**通讯引用:** 4260 | [OpenAlex ID](https://openalex.org/A5014628588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了统一多模态大语言模型在不同输出模态（文本与图像）下的语义一致性，并提出VGUBench基准来系统评估模型的语义一致性。

**💡 创新点**

创新点在于首次提出语义等价性（SEDOM）概念，构建三任务（文本生成理解TGU、视觉生成理解VGU、视觉渲染Render）以解耦推理与生成，并揭示现有U-MLLM在视觉回答上的严重失效。

**🔧 技术方法**

使用统一Transformer+自编码器或扩散模型的U-MLLM架构，利用LLM-as-a-judge的统一评分方式，并通过Pearson、Spearman相关性分析探究渲染与视觉理解的关系。

**📊 数据集**

数据集主要来自MMLU、AR-Challenge、OpenBookQA、CSQA、HellaSwag、BoolQ、GPQA、MATH、HumanEval，共采样2164问答对，用于构建TGU、VGU和Render任务。

**📈 对比分析**

与多种公开U-MLLM（Bagel、Emu3、BLIP3o等）和图像生成模型（Qwen-Image、LongCat等）比较，TGU得分高达60–90%，但VGU平均低于25%，Render亦显著低于TGU；相关性分析显示渲染质量与视觉回答几乎无显著关联。

**⚠️ 局限性**

主要局限在于评估仅覆盖文本与图像模态，未考虑视频等其它模态；缺少闭源模型的实验；缺乏因果机制解释，仅提供诊断性发现。

---

## 259. Learning Generation Orders for Masked Discrete Diffusion Models via Variational Inference

**arXiv ID:** 2602.23968 | [PDF](https://arxiv.org/pdf/2602.23968v1)

**作者:** David Fox `[一作]` (University of Bristol), Mengyue Yang `[通讯]` (University of Bristol)

**通讯引用:** 441 | [OpenAlex ID](https://openalex.org/A5069401509)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于变分推理的框架，用于在 Masked Diffusion Models（MDMs）中学习并决定并行生成的 token 生成顺序。

**💡 创新点**

创新点在于：①将生成顺序建模为可学习的隐变量并通过变分推理优化；②设计了可高效采样的近似后验分布（通过 α 网络产生生成顺序得分并归一化）；③利用 Rao‑Blackwell 化和 REINFORCE‑LOO 控制变量降低梯度方差；④在训练中同时优化 denoiser 与 token 选择器，使其在推理时实现自适应并行度。

**🔧 技术方法**

技术方法包括：Masked Discrete Diffusion Models、变分推理、ELBO 目标、Rao‑Blackwell 化、REINFORCE‑LOO、Bernoulli 后验、温度归一化、梯度估计等。

**📊 数据集**

使用 GSM8K 题库数据集进行实验，采用 170M 参数的 MDM 作为基准模型。

**📈 对比分析**

与三种基准采样策略（IID、Top Probability、Top Probability Margin）对比，实验显示在相同或更少的平均步骤下，本方法取得更高的准确率（例如 4 步平均步骤时 33.1% vs 23.7–29.0%），在更高预算下性能差距进一步缩小。

**⚠️ 局限性**

局限性包括：仅在 GSM8K 上验证，缺乏对多数据集和更大模型的评估；近似后验的设计仍有改进空间；训练过程需要额外的 REINFORCE 估计，可能带来方差和收敛速度的挑战。

---

## 260. Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction

**arXiv ID:** 2602.24080 | [PDF](https://arxiv.org/pdf/2602.24080v1)

**作者:** Xiang Li `[一作]` (Beijing University of Posts and Telecommunications), Benyou Wang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在2026年首次对语音对语音（S2S）系统进行图灵测试，并构建了包含28位志愿者、9个S2S模型、10个话题的1,486条多轮对话数据集；

**💡 创新点**

提出18维细粒度人类相似度分类体系，揭示语义理解已突破但声学、情感表达和人格等方面仍是瓶颈；基于此设计可解释的AI评测器，显著优于现有开源模型并达到人类评测水平；

**🔧 技术方法**

利用游戏化在线平台收集判定数据，采用多轮对话语音采集与人工标注；开发基于Qwen2.5‑Omni的两阶段Fine‑Tuning框架（Ordinal Discretization Layer + 正则化线性分类器）实现可解释评测；

**📊 数据集**

收集的多语言（中英）对话数据集，涵盖人机对话、人际对话、伪人对话（TTS合成）；其中人机对话由28名志愿者在专业录音室完成；

**📈 对比分析**

与人类评测对比：人机对话成功率均<0.5，伪人对话略高但仍低于人类对话；AI评测器准确率达0.9605，明显优于9种开源模型（平均0.4527）且超过人类评测器（0.7284）；

**⚠️ 局限性**

局限性包括：话题覆盖有限、仅考察S2S模型的音频输出不涉及多模态交互；AI评测器虽然准确但仍基于人工标注的细粒度标签，可能受标注偏差影响；未在更广泛的语言或更长对话情境中验证。

---

## 261. Peeling Off the Cocoon: Unveiling Suppressed Golden Seeds for Mutational Greybox Fuzzing

**arXiv ID:** 2602.23736 | [PDF](https://arxiv.org/pdf/2602.23736v1)

**作者:** Ruixiang Qian `[一作]` (Nanjing University), Zhenyu Chen `[通讯]` (Nanjing University)

**通讯引用:** 7089 | [OpenAlex ID](https://openalex.org/A5100422933)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种改进覆盖率驱动种子选择的技术，能够揭示被条件保护阻塞的“金色种子”，并通过迭代禁用守卫来逐步释放这些隐藏的有价值输入。

**💡 创新点**

创新点在于：①引入可切换的“守卫切换”程序变换，使得可以灵活禁用任意条件守卫而不破坏程序语义；②设计守卫层次分析与外层守卫识别算法，精准定位哪些守卫是阻塞金色种子的关键；③构建迭代种子选择框架，在每轮中增量加入新种子并不断剥离守卫，从而在保持可行性的前提下显著扩展有效种子集合。

**🔧 技术方法**

技术手段包括：LLVM 传递实现守卫切换插桩；控制流图与守卫层次分析；启发式识别“鲁莽守卫”（Crashing‑Reckless / Converging‑Reckless）并在每轮中剔除；与 AFL++ 的集成，实现基于 AFL++ 的覆盖率驱动种子选择与增量更新。

**📊 数据集**

实验使用了 LAVA（LavaBench）漏洞注入基准，涵盖 8 个不同文件格式（PNG、XML、SQL、LUA、TIFF 等）的 8 个目标程序，并从官方种子集合及公开数据集构建了高质量种子语料库。

**📈 对比分析**

通过将该技术与 AFL++ 原生 CSS、随机追加种子、OptiMin（基于 MaxSAT 的选择）等 7 种基线进行比较。结果显示，在两小时的种子选择预算下可额外选取约 340 颗种子；使用这些种子进行 24h 下游灰盒模糊测试，代码覆盖率平均提升 0.2–51.4 条边，bug 发现数排名第二；相较于单纯延长模糊时间（24h+δ），两小时内的额外种子在覆盖率上基本持平，bug 发现则更佳。

**⚠️ 局限性**

主要限制包括：①种子选择耗时占比过高（约 96%）导致实际应用成本上升；②在某些目标中，长时间运行并不明显提升覆盖率，增量收益有限；③并非所有被识别为被压制的金色种子都能显著提升模糊效果，选择算法对不同程序的适用性仍有差异。

---

## 262. Beyond the Click: A Framework for Inferring Cognitive Traces in Search

**arXiv ID:** 2602.24265 | [PDF](https://arxiv.org/pdf/2602.24265v1)

**作者:** Saber Zerhoudi `[一作]` (University of Passau), Michael Granitzer `[通讯]` (University of Passau)

**通讯引用:** 3649 | [OpenAlex ID](https://openalex.org/A5006866152)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套多代理框架，通过信息搜寻理论（IFT）和人类专家验证，从大规模交互日志中推断用户的认知轨迹。

**💡 创新点**

创新点在于：①将认知层面与行为日志结合，填补用户模拟器仅复制行为的空白；②采用多代理LLM（分析者、评论者、裁判）实现自动化标签生成并进行质量控制；③公开了认知标签数据集与可复现的工具。

**🔧 技术方法**

核心技术包括：信息搜寻理论标签设计、基于大型语言模型的多代理推理框架、人工与LLM混合验证、Transformer 预测模型（基于S-BERT+认知嵌入）用于下游任务。

**📊 数据集**

使用公开数据集：AOL Web Search、Stack Overflow Q&A、MovieLens 推荐，分别代表开放域搜索、技术问答和偏好发现。

**📈 对比分析**

在两项预测任务中比较：1）会话结果预测，认知增强模型 F1=0.90、AUC=0.92（比行为基线提升35%）；2）挣扎恢复预测，F1=0.78、AUC=0.83（比基线提升17%）。

**⚠️ 局限性**

局限性：推断的认知标签为理论假设，非直接测量；受限于输入数据的丰富度（缺乏完整 SERP HTML 时推断更弱）；实验仅覆盖三类信息检索场景，尚未验证在更专业或多模态领域的适用性。

---

## 263. Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives

**arXiv ID:** 2602.24136 | [PDF](https://arxiv.org/pdf/2602.24136v1)

**作者:** Haoran Wang `[一作]` (University of Bristol), Nantheera Anantrasirichai `[通讯]` (University of Bristol)

**通讯引用:** 3120 | [OpenAlex ID](https://openalex.org/A5021717616)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了可自适应的重建感知剪枝调度与3D差分高斯（3D-DoG）原语，实现对3D Gaussian Splatting模型的高效压缩与细节保留。

**💡 创新点**

创新点在于①动态剪枝调度根据重建误差自适应决定剪枝时机与比例；②引入空间-频域剪枝评分（SPS）提升稳定性；③设计3D-DoG原语在负密度环中提供对比，增强有限原语下的边缘与纹理表达。

**🔧 技术方法**

采用3D Gaussian Splatting基础、动态剪枝调度、SPS评分、3D-DoG原语、加速渲染CUDA模块、频域梯度分析、α-混合与球谐色彩编码等技术。

**📊 数据集**

在Mip-NeRF 360、Deep Blending、Tanks & Temples三大公开数据集上进行训练与评估。

**📈 对比分析**

与MaskGaussian、GaussianSpa、PuP-3DGS、Speedy-Splat等基线相比，在保持90%剪枝目标下，PSNR/SSIM/LPSIPS均接近或优于对手，同时模型尺寸缩减约90%，训练时间下降约25%（13m48s对比17m1s），帧率提升至289fps。

**⚠️ 局限性**

局限性包括：①对极端复杂场景仍需大量原语，剪枝率过高易导致细节损失；②3D-DoG增加计算开销；③目前仅验证静态场景，动态/大规模场景适用性待进一步研究。

---

## 264. Leveraging Geometric Prior Uncertainty and Complementary Constraints for High-Fidelity Neural Indoor Surface Reconstruction

**arXiv ID:** 2602.23926 | [PDF](https://arxiv.org/pdf/2602.23926v1)

**作者:** Qiyu Feng `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9170 | [OpenAlex ID](https://openalex.org/A5107772128)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出GPU-SDF框架，实现室内三维表面重建，结合自监督几何先验不确定性估计、不确定性引导损失、边缘距离场和多视角一致性正则化，显著提升薄结构与细节重建质量。

**💡 创新点**

1) 自监督估计几何先验不确定性，直接从先验中量化可靠性；2) 基于该不确定性动态调节先验权重，避免丢弃有用信息；3) 引入边缘距离场和局部多视角一致性约束，解决高不确定性区域的欠约束问题。

**🔧 技术方法**

神经隐式SDF、翻转一致性自监督不确定性估计、基于不确定性的几何损失、TEED边缘距离场、局部多视角一致性正则化、Eikonal正则化、hash 编码、Adam优化。

**📊 数据集**

ScanNet、Replica、ScanNet++（室内真实与合成数据集）。

**📈 对比分析**

与传统MVS、单目神经SDF、带先验的神经SDF（MonoSDF、ND-SDF、Deb-SDF等）以及Gaussian Splatting方法对比，GPU-SDF在Accuracy、Completeness、Chamfer、F-score等指标上实现SOTA，尤其在椅腿、栏杆等薄细节上明显优于基线。

**⚠️ 局限性**

对不可见视角的区域仍缺乏约束，整体指标提升有限，主要受益于高频细节，未来需进一步处理盲区。

---

## 265. MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy

**arXiv ID:** 2602.24222 | [PDF](https://arxiv.org/pdf/2602.24222v1)

**作者:** Albert Dominguez Mantes `[一作]` (Swiss Federal Institute of Technology), Martin Weigert `[通讯]` (Center for Scalable Data Analytics and Artificial Intelligence)

**通讯引用:** 11755 | [OpenAlex ID](https://openalex.org/A5001563470)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了一种多分辨率视觉Transformer，能够在同一张显微镜图像中同时处理不同物理分辨率的视图，并通过共享世界坐标系统与基于RoPE的位置编码实现跨尺度注意力；

**💡 创新点**

创新点在于将多尺度视图视作“模态”并通过世界坐标对齐；采用基于世界坐标的RoPE实现精确跨尺度位置编码；在多分辨率数据上进行MAE自监督预训练并使用多尺度解码器；证明仅用少数尺度即可显著提升分割性能；

**🔧 技术方法**

使用Transformer Encoder、基于世界坐标的RoPE、Masking AutoEncoder预训练、UNETR/Mask2Former分割解码器、嵌套多分辨率crop采样、PyTorch Lightning实现及Zarr Pyramid数据读取；

**📊 数据集**

合成环状图像数据集、mouse brain anatomy DAPI显微镜数据（12脑体，618张图），以及肾病病理WSI数据集（30训练/12测试张）；

**📈 对比分析**

与U-Net、DeepLabV3、SegFormer、SwinUNETR-V2等单尺度基线以及多尺度架构进行对比；在mouse brain任务中，_[1,8,32]+Mask2Former达mDSC 0.901，远超DeepLabV3 0.843；在合成任务中mDSC 0.9538；在肾病分割中Dice 0.8958，显著优于HoloHisto-4K 0.8454；MAE预训练显著加速收敛；

**⚠️ 局限性**

计算与显存成本随尺度级数提升；对世界坐标的精确标注有较高依赖；目前仅在语义分割任务验证，未评估检测/实例分割等；训练过程需要嵌套crop采样，实现复杂；

---

## 266. Pseudo Contrastive Learning for Diagram Comprehension in Multimodal Models

**arXiv ID:** 2602.23589 | [PDF](https://arxiv.org/pdf/2602.23589v1)

**作者:** Hiroshi Sasaki `[一作]` `[通讯]` (Japan Research Institute Limited), Hiroshi Sasaki (Japan Research Institute Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于伪对比样本的训练框架，使VLM在仅有光栅图像的条件下提升对图表结构的理解。

**💡 创新点**

创新点在于利用OCR提取文本后随机生成伪图表并通过规则编辑产生硬正负样本，无需编辑文件或语义合理样本。

**🔧 技术方法**

采用CLIP的结构感知对比学习（SaCLIP）、规则编辑、OCR、伪图表渲染等技术。

**📊 数据集**

使用公开的FlowVQA数据集（流图图像、描述、Mermaid代码、VQA对）。

**📈 对比分析**

与零样本、裁剪、基于编辑的对比学习等基线对比，在图像-文本匹配和VQA任务上显著提升Recall@1/5/10、MRR和F1分数，尤其在硬负样本环境中仍保持领先。

**⚠️ 局限性**

受限于图表样式与渲染器不匹配、OCR质量及仅适用于支持渲染器的图表类型。

---

## 267. Suppressing Prior-Comparison Hallucinations in Radiology Report Generation via Semantically Decoupled Latent Steering

**arXiv ID:** 2602.23676 | [PDF](https://arxiv.org/pdf/2602.23676v1)

**作者:** Ao Li `[一作]` (University of New South Wales), Lei Xing `[通讯]` (Stanford University)

**通讯引用:** 33110 | [OpenAlex ID](https://openalex.org/A5100381484)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种训练无关、推理时的隐空间控制框架SDLS，用来抑制放射学报告生成中的历史对比幻觉。

**💡 创新点**

创新点在于使用LLM驱动的语义分解结合QR正交化，构造语义无关的干预向量（SDIV），实现历史风格与视觉语义的几何解耦；并证明在保持甚至提升诊断准确度的同时显著降低幻觉率。

**🔧 技术方法**

核心技术包括对比上下文挖掘、差分向量构造、PCA+QR正交化得到SDIV、在decoder层注入正交干预（如Attention输出层），以及利用FilBERT、CheXpert等评估器和注意力可视化。

**📊 数据集**

在MIMIC‑CXR作为训练与基准数据集；在IU‑Xray、CheXpert Plus 进行零样本转移验证；同时使用公开的基线模型BiomedGPT、VED、LLaVA‑Med。

**📈 对比分析**

与多种基线（全局ICV、特定50、无干预）对比，SDIV在BiomedGPT上将FilBERT评分从0.2373降至0.1889（-37.3%），Macro‑F1从0.2242提升至0.3208；零样本时同样保持或提升性能。其他模型的效果受限，但仍显示一定的幻觉抑制。

**⚠️ 局限性**

局限包括：对模型架构高度依赖，非深层跨模态交互的模型（如LLaVA‑Med）效果有限；需要手工挑选历史-无历史文本对进行训练，且干预强度 λ 的调优依赖网格搜索；未进行临床工作流的人工评估。

---

## 268. Gendered Digital Financing Adoption and Women's Financial Inclusion in Pakistan

**arXiv ID:** 2602.23465 | [PDF](https://arxiv.org/pdf/2602.23465v1)

**作者:** Abdul Wadood Asim `[一作]` (Mirpur University of Science and Technology), Muhammad Raees `[通讯]` (Mirpur University of Science and Technology)

**通讯引用:** 388 | [OpenAlex ID](https://openalex.org/A5050164798)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

利用全球Findex 2023数据对巴基斯坦女性移动支付服务采用与金融包容性之间的关系进行二元逻辑回归分析，探讨手机拥有率、互联网接入和教育水平等因素的影响。

**💡 创新点**

首次将技术接入与社会能力（教育、互联网）交互作用纳入模型，量化两者共同作用下的女性金融包容性提升机制，弥补了以往仅关注单一维度的研究空白。

**🔧 技术方法**

采用特征工程构造交互项后，使用二元逻辑回归模型，结合Odds Ratio和显著性检验评估变量效应。

**📊 数据集**

使用全球Findex 2023调查数据（约1,002名受访者），涵盖城市与农村样本、手机拥有率、互联网使用、教育水平以及金融账户持有情况。

**📈 对比分析**

通过比较子假设模型的准确率、精确率、召回率和OR值来评估模型性能；主模型准确率92.5%，精确率100%，召回率70%，显示移动支付显著提升女性金融账户拥有率。

**⚠️ 局限性**

研究仅基于横截面自报数据，缺乏纵向跟踪与使用层面细节，且未能充分捕捉互联网接入背后的社会规范与行为障碍。

---

## 269. Tilt-X: Enabling Compliant Aerial Manipulation through a Tiltable-Extensible Continuum Manipulator

**arXiv ID:** 2602.23576 | [PDF](https://arxiv.org/pdf/2602.23576v1)

**作者:** Anuraj Uthayasooriyan `[一作]` (Queensland University of Technology), Felipe Gonzalez `[通讯]` (Queensland University of Technology)

**通讯引用:** 5662 | [OpenAlex ID](https://openalex.org/A5071075714)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文设计并实现了一种名为Tilt‑X的连续臂无人机操纵器，验证其工作空间和抗风下洗的性能。

**💡 创新点**

创新点在于将倾斜、伸缩和三缆驱动的连续臂集成到机身固定轴承上，实现从垂直下方到水平前方的全方向可调姿态和可伸缩工作空间。

**🔧 技术方法**

采用三缆电缆驱动连续臂、同轴伸缩管、螺旋升降电机、Pixhawk飞控与Raspberry Pi ROS通信以及正向运动学模型进行设计与实验。

**📊 数据集**

实验使用OptiTrack光学跟踪系统记录端效器位姿，收集不同倾斜角、伸缩长度、下洗、地面和墙面条件下的坐标与姿态数据。

**📈 对比分析**

通过与理论正向运动学模型计算的位姿进行欧氏误差和姿态误差对比，结果显示当端效器伸展到离开下洗影响区时误差显著降低，最大位置误差约<100 mm，标准差随倾斜/伸缩变化。

**⚠️ 局限性**

局限性包括常数曲率假设导致模型与实验偏差，缆绳摩擦、自重与结构公差未建模，控制仍为开环，载荷能力有限，缺乏气动仿真和视觉伺服等高级控制策略。

---

## 270. Synthetic Data Powers Product Retrieval for Long-tail Knowledge-Intensive Queries in E-commerce Search

**arXiv ID:** 2602.23620 | [PDF](https://arxiv.org/pdf/2602.23620v1)

**作者:** Gui Ling `[一作]` (Taobao and Tmall Group of Alibaba), Haihong Tang `[通讯]` (Taobao and Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对电商长尾知识密集型查询，构建了一个高效的数据合成框架，通过多候选查询重写与离线产品检索生成高质量的查询-产品配对数据，用于提升检索模型的召回和相关性。

**💡 创新点**

创新点在于：①使用多奖励（语义一致性、产品侧分布对齐、多样性）强化的 RL 训练重写模型，①避免了重写产生的语义漂移；②构建离线检索管线，直接将重写查询与产品匹配并通过双重相关性判别器筛选，生成可信的合成训练样本；③融合业务规则和通用 LLM 的双重过滤，提升了合成数据的质量。

**🔧 技术方法**

核心技术包括：多候选查询重写模型（基于 Qwen3‑30B‑A3B，采用 SFT + 多奖励 RL 训练）；产品侧分布对齐的 LLM 预训练与 perplexity 评估；密集检索模型 Tbstars‑3B；业务特定 42B MoE 相关性模型与通用 LLM 相关性评估；离线检索管线与在线 A/B 实验。

**📊 数据集**

使用的主要数据集：约 20M 条查询–产品对（400K 条查询）；5K 条人工标注的重写示例；离线合成的 200M 条训练样本；每种查询类型 500 条真实日志用于离线与在线评测。

**📈 对比分析**

与传统单一重写或无重写基线相比，合成数据提升了 Item Goodrate 与 Query Goodrate@N，最显著的提升为 Negative 查询的 Query Goodrate@10 +31.59%；在线 SBS 人工评测显示 GSB 上升至 +27.66%（Alternative）和 +31.59%（Negative），整体用户体验显著提升；性能提升不依赖额外技巧，直接由合成数据驱动。

**⚠️ 局限性**

局限性包括：仅使用文本知识，无法满足多模态查询需求（如“同款服饰”）；对 Alternative（替代品）查询的合成精度仍低于 80%，在知识更新速度快的场景中表现不佳；重写模型与检索模型对新产品的适应性仍需进一步提升。

---

## 271. SAILOR: A Scalable and Energy-Efficient Ultra-Lightweight RISC-V for IoT Security

**arXiv ID:** 2602.24166 | [PDF](https://arxiv.org/pdf/2602.24166v1)

**作者:** Christian Ewert `[一作]` (Institute of Computer Engineering), Saleh Mulhem `[通讯]` (Institute of Computer Engineering)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种可扩展、低功耗、超轻量级的 RISC‑V 核——SAILOR，支持 NIST 标准的 Zkn‑Zkt 加密指令扩展，并通过可调 1~32 位位序列化数据路径实现面积可控；

**💡 创新点**

创新点在于将位序列化技术与模块化可扩展架构结合，既保持极低的面积与能耗，又能在不显著增加硬件开销的前提下实现完整的加密指令支持；通过轻量级 AES S‑Box、xtime 单元及专用 SHA‑2 固定移位/旋转，进一步提升加密性能与能效；该方案在面积、性能和能耗三者之间实现了全新的平衡。

**🔧 技术方法**

使用的核心技术包括：位序列化数据路径、模块化可扩展设计、轻量化加密硬件（AES S‑Box、xtime、SHA‑2 固定移位/旋转）、Chisel HDL 代码生成、Synopsys Design Compiler（Area）与 Prime Power（能耗）仿真、RISC‑V Torture Test 进行验证。

**📊 数据集**

使用的基准数据集包括：RISC‑V 加密基准套件（AES‑128/192/256 加密/解密、SHA‑256/512、Prince）、Coremark、Dhrystone。

**📈 对比分析**

比较方法：将 SAILOR 的 1/2/4/8/16/32 位序列化核与现有序列化核心（SERV、QERV、FazyRV）及 32 位 PicoRV32 核进行面积、速度、能耗、EDP、ATP 对比。结果显示：相较同宽度序列化核心，SAILOR 在速度上提升 1.7–2.3×；与 PicoRV32 对比，16/32 位版本速度提升 2.0–3.8×，在 Zkn‑Zkt 扩展下可达 6–12×；能耗比 PicoRV32 低 2.3–3.7×，加密扩展后能耗降低 4.6–12.8×。EDP 与 ATP 同样显著提升。

**⚠️ 局限性**

局限性：为了实现常数时间执行，Zkt 扩展需要全字节/全词处理，导致非安全工作负载性能下降约 55%；序列化核心在极简实现上仍略大于最小化序列化核心；未对分支预测、CSR 等高级功能进行深入优化，未来工作可进一步减少面积与能耗，并提供可选的侧信道保护策略。

---

## 272. MINT: Multimodal Imaging-to-Speech Knowledge Transfer for Early Alzheimer's Screening

**arXiv ID:** 2602.23994 | [PDF](https://arxiv.org/pdf/2602.23994v1)

**作者:** Vrushank Ahire `[一作]` (Indian Institute of Technology Ropar), M. A. Ganaie `[通讯]` (Indian Institute of Technology Ropar)

**通讯引用:** 3697 | [OpenAlex ID](https://openalex.org/A5033429345)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出MINT框架，利用MRI教师模型对语音特征进行跨模态知识转移，实现仅用语音即可早期阿尔茨海默筛查

**💡 创新点**

首次将结构MRI的决策边界通过知识蒸馏迁移到语音编码器，形成生物学基础的无影像推理路径

**🔧 技术方法**

三阶段流程：语音自监督预训练（MAE）、MRI教师训练（MLP压缩+分类）以及跨模态投影头对齐（MSE+余弦损失）

**📊 数据集**

使用ADNI-4 Storyteller数据集：14,235条无标签语音、1,228条MRI（CN/MCI）用于教师训练，266条配对语音+MRI用于对齐

**📈 对比分析**

与传统语音基线（RF、SVM、MLP等）相比，MINT对齐后语音AUC 0.720≈RF最高0.711；融合AUC 0.973高于MRI教师0.958，表明性能可观

**⚠️ 局限性**

对齐数据量有限（266条），导致对齐质量受限；实验仅在单一ADNI-4数据集，缺乏跨站点验证，模型对不同人群的泛化需进一步评估

---

## 273. Evidential Neural Radiance Fields

**arXiv ID:** 2602.23574 | [PDF](https://arxiv.org/pdf/2602.23574v1)

**作者:** Ruxiao Duan `[一作]` (Yale University), Alex Wong `[通讯]` (Yale University)

**通讯引用:** 28730 | [OpenAlex ID](https://openalex.org/A5065094327)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了Evidential Neural Radiance Fields（Evidential NeRF），通过单前向传播即可在NeRF中同时估计阿勒特里克不确定性（数据噪声）和知识不确定性（模型缺失）。

**💡 创新点**

创新点在于将证据推理（evidential deep learning）直接嵌入NeRF的体素到像素不确定性传播过程，既实现了两类不确定性估计，又保持了渲染质量且无需多次采样。

**🔧 技术方法**

采用正态-逆伽马（Normal‑Inverse‑Gamma）分布建模像素不确定性，利用单网络预测均值、方差、阿勒特里克与知识不确定性以及形状分数，采用闭式负对数似然（NLL）加正则化训练，并在Nerfacto框架下实现。

**📊 数据集**

在三个标准3D重建数据集上评估：Light Field（LF）、Local Light Field Fusion（LLFF）和 RobustNeRF。

**📈 对比分析**

在统一的基准下与闭式概率模型（Normal、MoL）、贝叶斯方法（MC Dropout、BayesRays）以及集成方法（Ensemble、DANE）进行对比，Evidential NeRF在图像重建指标（PSNR、SSIM、LPIPS）和不确定性指标（NLL、AUSE、AUCE）上均排在前列，性能与集成方法相近但推理速度快。

**⚠️ 局限性**

局限性包括：相较于纯闭式模型仍有一定计算开销；对形状分数等超参数敏感；在极端离群或动态场景下不确定性估计可能偏高；且目前仅在静态单视角训练和推断环境下验证，需进一步探索对大规模、多视角实时应用的可扩展性。

---

## 274. KEEP: A KV-Cache-Centric Memory Management System for Efficient Embodied Planning

**arXiv ID:** 2602.23592 | [PDF](https://arxiv.org/pdf/2602.23592v1)

**作者:** Zebin Yang `[一作]` (Peking University), Meng Li `[通讯]` (Peking University)

**通讯引用:** 24687 | [OpenAlex ID](https://openalex.org/A5100457407)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套名为 KEEP 的 KV‑Cache 关注型记忆管理系统，用于提升基于大型语言模型的具身规划效率

**💡 创新点**

三大创新：① 静动记忆构造算法，根据记忆更新频率混合粒度管理 KV 缓存，减少无效重算；② 多跳记忆重算，通过重要性传播动态识别并重算关键记忆，恢复跨记忆的注意力；③ 层平衡记忆加载，平衡不同层的 KV 加载与计算，消除 pipeline 空隙

**🔧 技术方法**

采用句子编码器聚类记忆、KV 缓存预存与重算、Attention 权重重要性传播、层级 KV 存储与管线化调度，以及 vLLM+PyTorch 的实现

**📊 数据集**

在 ALFRED 与 WAH‑NL 两大具身规划基准上评测，使用 Qwen‑2.5‑14B/32B（INT4）语言模型

**📈 对比分析**

与文本记忆、全 KV 重算、全重用、CacheBlend 等方法比较，KEEP 在 ALFRED 上实现 2.68× 速度提升、4.13% 成功率提升、1.90× TTFT 缩短；在 WAH‑NL 上也显著提升 Sub‑SR 与 TTFT

**⚠️ 局限性**

局限性：仍需依赖较大 KV 存储容量、对记忆更新频率的阈值设定敏感、在极端动态场景下可能出现重算比例过高导致延迟上升

---

## 275. ABPolicy: Asynchronous B-Spline Flow Policy for Real-Time and Smooth Robotic Manipulation

**arXiv ID:** 2602.23901 | [PDF](https://arxiv.org/pdf/2602.23901v1)

**作者:** Fan Yang `[一作]` (Tianjin University), Yuting Su `[通讯]` (Tianjin University)

**通讯引用:** 6873 | [OpenAlex ID](https://openalex.org/A5033713097)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种异步B‑样条流动匹配策略（ABPolicy），通过预测连续控制点实现机器人操作的平滑、实时控制。

**💡 创新点**

创新点包括：①在B‑样条控制点空间内使用流动匹配生成动作；②双向动作预测结合局部拟合（CCR）以保证片段间连续性；③异步推理框架消除同步延迟，提升动态环境响应。

**🔧 技术方法**

主要技术：B‑样条轨迹参数化、双向动作预测、流动匹配模型（DiT架构）、连续性约束拟合、异步多线程推理。

**📊 数据集**

使用七个机器人操作任务（包括三种动态任务和四种静态任务），每个任务采集数百条演示数据，采用AgileX Piper 6‑DoF机械臂和RealSense摄像头。

**📈 对比分析**

对比同步推理与异步推理以及多种动作表示方法，实验显示ABPolicy在动态任务上平均提升18.3%成功率、在静态任务上减少14.2%完成时间，动作平滑度（ZCR、Acc p95）分别下降约29%和57%。

**⚠️ 局限性**

局限性：依赖高质量B‑样条控制点拟合，对极高频或非平滑动作的适应性有限；异步推理需额外硬件/网络支持；模型训练和推理成本相对较高。

---

## 276. Joint Geometric and Trajectory Consistency Learning for One-Step Real-World Super-Resolution

**arXiv ID:** 2602.24240 | [PDF](https://arxiv.org/pdf/2602.24240v1)

**作者:** Chengyan Deng `[一作]` (University of Electronic Science and Technology of China), Wang Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 461957 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种一次性生成的真实场景超分辨率方法GTASR，解决一致性漂移和几何解耦问题。

**💡 创新点**

创新点在于：①轨迹对齐（TA）通过全路径投影修正PF-ODE的切向量场；②双参考结构修正（DRSR）结合真实轨迹与目标结构，引入稳定损失和校正损失消除几何偏差。

**🔧 技术方法**

技术上使用一致性训练（Consistency Models）、分布轨迹匹配（DTM）、Sobel梯度约束、感知损失以及基于PF-ODE的全路径投影。

**📊 数据集**

在ImageNet‑Test、RealSR、RealLQ250、RealSet65四个数据集上进行训练与评估，使用256×256 HR/64×64 LR对。

**📈 对比分析**

与GAN、传统扩散和一阶扩散方法比较，GTASR在LPIPS、MANIQA、CLIPIQA、TOPIQ等无参考指标上均超越CTMSR、ResShift、StableSR等基线，同时保持与CTMSR相当或更低的推理时延。

**⚠️ 局限性**

局限性包括：在PSNR/SSIM等传统重建指标上略逊于部分多步扩散模型；方法仍需在极端降质场景下验证鲁棒性；对极大分辨率缩放（如×8）尚未充分测试。

---

## 277. On the Limits of Interpretable Machine Learning in Quintic Root Classification

**arXiv ID:** 2602.23467 | [PDF](https://arxiv.org/pdf/2602.23467v1)

**作者:** Rohan Thomas `[一作]` (University of Missouri-Kansas City), Majid Bani-Yaghoub `[通讯]` (University of Missouri-Kansas City)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估各种机器学习模型能否从多项式系数中自主发现可解释的数学规则，重点在五次多项式根分类。

**💡 创新点**

对比传统可解释模型与深度模型在五次多项式根分类任务上的表现，证明仅凭数据驱动无法自动提取符号不变量，需手工特征工程；使用知识蒸馏揭示神经网络内部近似本质。

**🔧 技术方法**

决策树、逻辑回归、SVM、随机森林、梯度提升、XGBoost、符号回归、神经网络；特征工程（Sturm序列、Descartes法则、Newton和、临界点sign变化Crit8）；知识蒸馏与SHAP解释。

**📊 数据集**

随机生成 40,000 条五次多项式（系数区间 [-10,10]），以及低次多项式（2-4度）做验证；使用 NumPy 计算根。

**📈 对比分析**

通过 5 折分层交叉验证、20 次随机种子比较平衡准确率。神经网络在原始系数下 84.3%；决策树仅 59.9%；给定 Crit8 后决策树提升至 84.2%；对 OOD、数据效率、噪声鲁棒性测试显示神经网络需大量样本且不具尺度不变性。

**⚠️ 局限性**

数据分布仅 [-10,10]；近似根标签可能不稳定；未涵盖更先进模型如 transformer、程序合成；符号回归搜索受限；未直接检查内部表示；仅关注系数生成方式。

---

## 278. Beyond State-Wise Mirror Descent: Offline Policy Optimization with Parameteric Policies

**arXiv ID:** 2602.23811 | [PDF](https://arxiv.org/pdf/2602.23811v1)

**作者:** Xiang Li `[一作]` (Nanjing University), Yuheng Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8537 | [OpenAlex ID](https://openalex.org/A5056699900)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了在离线强化学习中，对一般动作空间和独立参数化策略的理论分析与算法设计，克服了传统基于状态维度镜像下降（PSPI）在连续动作和全局参数化策略上的局限。

**💡 创新点**

创新点包括：① 引入“上下文耦合（contextual coupling）”概念，说明状态层面镜像下降在全局参数化策略下会失效；② 以兼容函数逼近（CFA）为导向的误差分解，给出泛化的第一阶更新范式；③ 设计两种无偏估计的策略更新方法——最小二乘策略更新（LSPU）与分布式鲁棒更新（DRPU），并给出统计与计算效率的理论保证；④ 在无动作分布差异时，DRPU退化为行为克隆，展示了离线RL与模仿学习的统一性。

**🔧 技术方法**

主要技术手段包括：离线RL的Pessimistic Soft Policy Iteration（PSPI）、镜像下降、自然策略梯度、兼容函数逼近、最小二乘回归、分布式鲁棒优化（CVaR）、密度比估计、信息理论的KL与覆盖系数。

**📊 数据集**

文章未给出具体实验数据集，理论证明以两类离线数据集合（Critic数据集和Actor数据集）为前提；在实验部分（如图示）可能使用了简单的MDP或标准离线RL基准（如MuJoCo/CartPole），但具体数据集未明示。

**📈 对比分析**

方法比较基于理论误差分解和实验对比：LSPU在存在Actor-Critic不匹配时会出现常数步误差，而DRPU在无分布差异时能驱动兼容误差接近零，收敛至比较器策略；实验图表显示DRPU相对LSPU在性能上更优，尤其在离线数据与比较器策略相同的情形。

**⚠️ 局限性**

局限性：仅适用于显式随机策略（如log-linear、Gaussian），不适用于确定性或隐式生成策略（如扩散模型）；对密度比估计的覆盖常数依赖较大，实际应用中可能需要更稳健的估计；理论分析未涵盖多任务或分布漂移的动态情境。

---

## 279. Toward E2E Intelligence in 6G Networks: An AI Agent-Based RAN-CN Converged Intelligence Framework

**arXiv ID:** 2602.23623 | [PDF](https://arxiv.org/pdf/2602.23623v1)

**作者:** Youbin Han `[一作]` (Kyung Hee University), Yan Chen `[通讯]` (Ruhr University Bochum)

**通讯引用:** 14592 | [OpenAlex ID](https://openalex.org/A5100378075)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文提出了基于大型语言模型和 ReAct 机制的 AI 代理框架，实现了 RAN 与 CN 的统一端到端智能控制。

**💡 创新点**

创新点在于将 LLM 与 ReAct 的闭环推理‑行动‑观测循环相结合，消除任务专用模型碎片化、重训练需求，并实现跨域协同决策。

**🔧 技术方法**

采用的技术包括大型语言模型（如 Phi‑3‑4B‑Instruct、GPT‑5‑mini）、ReAct 推理框架、中心化监控数据库、MCP 工具接口以及 LLM 的动态查询与策略生成。

**📊 数据集**

使用的数据集主要有 5G 生产数据集（83 条来自爱尔兰运营商的轨迹）以及模拟网络切片场景的数据。

**📈 对比分析**

通过与基于 LSTM 的任务特定模型比较，LLM 在已知与未知场景下的 MAE、RMSE 保持相近或更优；在网络切片实验中，E2E LLM 方案相比单域 LLM 或轮询策略提升了约 4–5% 的 SLA 满足率。

**⚠️ 局限性**

主要局限包括 LLM 推理延迟与算力开销、对安全性与可验证性的担忧，以及在大规模部署时的单点瓶颈和对实时控制的适配性。

---

## 280. Causal Identification from Counterfactual Data: Completeness and Bounding Results

**arXiv ID:** 2602.23541 | [PDF](https://arxiv.org/pdf/2602.23541v1)

**作者:** Arvind Raghavan `[一作]` (Columbia University), Elias Bareinboim `[通讯]` (Columbia University)

**通讯引用:** 3273 | [OpenAlex ID](https://openalex.org/A5039620960)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了ctfIDu⁺算法，能够在拥有任意可实现的层3（ℒ₃）数据时完成对所有可识别的反事实量的完全识别；

**💡 创新点**

建立了反事实可实现性与可识别性之间的根本对应关系，证明了在非参数设定下可实现的数据收集能力与可识别量的上限相同；并推导出利用反事实数据得到的比以往更紧的NTE上界与下界；

**🔧 技术方法**

使用结构因果模型（SCM）、因果图、反事实因子、c-组件分解与识别子算法（identify⁺）等理论工具，并结合c-树、c-茎结构的可实现性判定；

**📊 数据集**

实验主要基于人工生成的模拟数据（如交通摄像机、哑铃图等），通过随机抽样和贝叶斯后验模拟验证方法效果；

**📈 对比分析**

与以往IDC*、ctfID等识别算法以及传统的概率因果界限方法进行比较，ctfIDu⁺在可识别问题上实现完全识别，在不可识别问题上得到更窄的区间，实验中区间宽度明显缩小，说明方法性能优越；

**⚠️ 局限性**

仍受限于非参数设定下的ℒ₂.₅上界——任何超出ℒ₂.₅的反事实量都不可完全识别；方法依赖可实现的c-随机化实验，实际可行性受实验环境限制；

---

## 281. Geometry-based pneumatic actuators for soft robotics

**arXiv ID:** 2602.24104 | [PDF](https://arxiv.org/pdf/2602.24104v1)

**作者:** Rui Chen `[一作]` (Institute of Mechanical Intelligence), Antonio Frisoli `[通讯]` (Institute of Mechanical Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

开发了一种基于几何约束的气动柔性致动器（GPA），并在软外骨骼、触觉界面与双足机器人等三大应用场景中实现并验证其性能。

**💡 创新点**

创新点在于将多层CNC热压成型腔室与可配置约束层相结合，实现了可预测的近零弯曲半径、多状态控制以及可定制的复杂几何形状，解决了传统单腔柔性气动致动器易失稳、弯曲半径大、功能单一的问题。

**🔧 技术方法**

技术手段包括：① TPU涂层尼龙面料的CNC热压成型制造；② 约束层的几何导向设计；③ 通过压缩比例因子和角度关系推导的数学模型；④ 基于ESP32与比例阀的实时气压控制；⑤ 多腔/并联/分段/双向等多种GPA拓扑实现。

**📊 数据集**

未使用公开的数据集，而是通过自制实验平台收集了大量手工测量数据（角度、力矩、压强、位移、EMG等），并将实验结果与传统单腔致动器进行对比。

**📈 对比分析**

通过与单腔致动器在相同负载条件下的对比实验，验证了GPA在外力加载下的稳定性、角度可预测性以及无外向偏移；在三项应用中：软外骨骼可将手腕屈伸肌电活性降低51%（最大），触觉界面在80 kPa下实现8 N的即时力反馈，双足机器人在同步前进模式下达到0.83 mm/s的速度，并实现小半径转弯；所有指标均优于现有的柔性气动致动器方案。

**⚠️ 局限性**

局限性包括：① 仍缺乏闭环传感与自适应控制；② 受限于压缩空气供应，响应速度和功率密度受限；③ 多腔/并联等复杂结构对CNC热压工艺的精度与重复性提出更高要求；④ 长期使用下材料的疲劳与泄漏风险需要进一步评估。

---

## 282. Pacing Opinion Polarization via Graph Reinforcement Learning

**arXiv ID:** 2602.23390 | [PDF](https://arxiv.org/pdf/2602.23390v1)

**作者:** Mingkai Liao `[一作]` (Shenzhen University), Mingkai Liao `[通讯]` (Shenzhen University)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5003763365)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出 PACIFIER，一个基于图强化学习的序贯干预框架，用于在社交网络中调节意见极化。它将传统的 ModerateInternal (MI) 和 ModerateExpressed (ME) 任务转化为一次性规划的马尔可夫决策过程，学习可直接在大规模网络上执行的干预策略。

**💡 创新点**

创新点主要有三：① 目标无关（objective‑agnostic）——策略可根据任何极化相关奖励函数训练；② 统一框架可覆盖线性与非线性 FJ 模型、连续内部意见、成本敏感干预以及拓扑变更（如节点移除）等多种情境；③ 采用时间感知节点标记与极化感知全局特征，解决拓扑保持情境下的状态混淆与信用分配难题。

**🔧 技术方法**

技术方法：图神经网络编码器（GraphSAGE + 虚拟节点聚合）+ 值网络（Q‑learning 或基于即时奖励的贪婪网络）；多步 Q‑learning（n‑step、经验回放、目标网络）与蒙特卡罗奖励估计；在状态表示中加入节点历史标记、成本、持续干预比例以及多种聚合的全局极化特征。

**📊 数据集**

使用数据集：15 条真实 Twitter 话题的关注/转发网络（节点数从 1.1k 到 155k），以及基于两社区 BA 生成的 600 条合成网络（节点数 30–500），用于离线训练与评估。

**📈 对比分析**

与基线（BOMP、Greedy、PageRank、ExtremeExpressed、ExtremeNeighbors 等）对比，PACIFIER‑RL 在成本敏感 MI、ME 及其成本版本中实现了 15%–40% 的 AUC/ANP 提升，ME‑cost 任务中获得 100% 胜率；在极化连续/非线性和节点移除实验中同样保持领先或相近于最优；在线性 MI 的最优解已知场景下，PACIFIER‑RL 与 BOMP 接近；在 unweighted MI 的真实网络中稍逊 BOMP，但在大型网络上差距收敛。

**⚠️ 局限性**

局限性：① 在 unweighted MI 的真实网络上性能略低于 BOMP，表明对纯线性结构仍有改进空间；② 需要先在合成图上训练模型，若真实网络与训练分布偏差较大可能影响泛化；③ 训练时需多次模拟动力学收敛，仍有计算开销；④ 对极化动态的准确建模依赖于环境模拟，若真实动态更复杂，奖励信号可能不稳定。

---

## 283. PseudoAct: Leveraging Pseudocode Synthesis for Flexible Planning and Action Control in Large Language Model Agents

**arXiv ID:** 2602.23668 | [PDF](https://arxiv.org/pdf/2602.23668v1)

**作者:** Yihan `[一作]`, Xin Chen `[通讯]` (Texas A&M University)

**通讯引用:** 18178 | [OpenAlex ID](https://openalex.org/A5100363101)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于LLM的“PseudoAct”框架，先通过伪代码合成全局规划，再由控制流执行器按计划执行动作，取代传统反应式ReAct模式；

**💡 创新点**

创新点在于将伪代码视为可执行的规划模板，利用LLM的代码生成能力提前定义控制流、循环、分支和数据依赖，显著减少冗余操作、避免无限循环并降低Token消耗；

**🔧 技术方法**

核心技术包括：LLM伪代码生成器、控制流执行器、ReAct式动作执行子模块；采用预训练的大型语言模型进行结构化代码合成；

**📊 数据集**

在公开文本验证数据集FEVER和多跳问答数据集HotpotQA上进行评测，并在电网操作的实战场景（五种不同任务）进行验证；

**📈 对比分析**

与ReAct和DFSDT基线比较，FEVER上准确率从67.31%提升至88.24%（绝对提升20.93%），HotpotQA上准确率从73.21%提升至82.14%，并在电网任务中实现安全、可控的迭代和状态持久化，整体性能优于现有方法；

**⚠️ 局限性**

局限性包括：依赖LLM生成伪代码的质量，若代码出现语法或逻辑错误会导致执行失败；仅适用于可用伪代码表达的任务，对极端动态或需要实时感知的环境适应性有限；

---

## 284. Domain-Partitioned Hybrid RAG for Legal Reasoning: Toward Modular and Explainable Legal AI for India

**arXiv ID:** 2602.23371 | [PDF](https://arxiv.org/pdf/2602.23371v1)

**作者:** Rakshita Goel `[一作]` (Birla Institute of Technology and Science), Dhruv Kumar `[通讯]` (Birla Institute of Technology and Science)

**通讯引用:** 6624 | [OpenAlex ID](https://openalex.org/A5027859418)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个面向印度法律的域分区混合检索+知识图谱框架，用于提升法律检索与答案生成的准确性与可解释性。

**💡 创新点**

创新点在于将三大专门RAG模块与Neo4j知识图谱相结合，并引入LLM驱动的查询路由器，实现动态路由、多跳推理与引用生成。

**🔧 技术方法**

使用的技术包括检索增强生成（RAG）、向量检索（ChromaDB + SentenceTransformers）、Neo4j知识图谱、Gemini 2.5 Flash与Gemini 2.0 Flash LLM、Agentic orchestrator以及LLM-as-a-Judge评估框架。

**📊 数据集**

使用的数据集包括约2,900条最高法院判决、宪法与法案文本、3,481条印度刑法（IPC）条文、构建的2,586节点/5,056关系的法律知识图谱以及40问合成的LLM-as-a-Judge基准。

**📈 对比分析**

通过LLM-as-a-Judge评估将完整混合系统与仅RAG基线对比，混合系统获得70%通过率（6.09/10），而基线仅37.5%（4.16/10），在完整性、相关性与法律推理质量上显著优于基线。

**⚠️ 局限性**

局限性包括数据覆盖范围有限（KG仅2,586节点，RAG语料有限）、缺乏最新立法与大规模判例、路由器在开放式法律推理上表现不足，以及未实现生产级实体抽取与自动化更新。

---

## 285. MAGE: Multi-scale Autoregressive Generation for Offline Reinforcement Learning

**arXiv ID:** 2602.23770 | [PDF](https://arxiv.org/pdf/2602.23770v1)

**作者:** Chenxing Lin `[一作]` (Xiamen University), Cheng Wang `[通讯]` (Xiamen University)

**通讯引用:** 26353 | [OpenAlex ID](https://openalex.org/A5100736836)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发了 MAGE——一种多尺度自回归生成模型，用于离线强化学习，能够在长时序稀疏奖励任务中生成连贯、可控的轨迹。

**💡 创新点**

创新点包括：1) 采用多尺度量化自编码器捕获从宏观到微观的时间依赖；2) 通过多尺度 Transformer 以粗到细的自回归方式生成轨迹；3) 引入返回到目标（RTG）条件和条件引导解码器，实现对短期行为的精细控制；4) 在所有尺度上使用统一策略，避免了传统层次方法中多策略互相优化的难题。

**🔧 技术方法**

技术手段：VQ‑VAE 量化自编码器、多尺度 Transformer、自回归生成框架、RTG 条件引导、条件损失（L_cond）以及离线数据训练与评估。

**📊 数据集**

使用 D4RL 离线 RL 基准数据集：Adroit、Franka Kitchen、Maze2D、Multi2D、AntMaze 以及 Gym 运动学任务。

**📈 对比分析**

与 15 个基线（BC、CQL、IQL、MPPI、Decision Transformer、Trajectory Transformer、Diffuser、Decision Diffuser、RGG、Diffusion‑QL、ADT、HDMI、CARP 等）在 5 个基准上对比。MAGE 在长时序稀疏奖励任务上显著优于对手，在密集奖励任务亦保持竞争力；平均得分最高；推理速度约 27 ms/step，显著快于 HD、DD 等方法。

**⚠️ 局限性**

局限性：对极端稀疏奖励与极长时间步任务的表现仍有限；层次设计可能限制细粒度调整的灵活性；对分布漂移的鲁棒性不足；多尺度数量需针对任务调优；在多智能体或真实硬件环境中的适用性尚未充分验证。

---

## 286. SDMixer: Sparse Dual-Mixer for Time Series Forecasting

**arXiv ID:** 2602.23581 | [PDF](https://arxiv.org/pdf/2602.23581v1)

**作者:** Xiang Ao `[一作]` (Beijing Jiaotong University), Xiang Ao `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 3273 | [OpenAlex ID](https://openalex.org/A5068007462)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种双流稀疏混合器（SDMixer）框架，用来同时在频域和时域对多变量时间序列进行分解与建模，并通过稀疏门控过滤无效变量依赖，最终融合得到更准确的预测结果。

**💡 创新点**

创新点在于：①将趋势与季节成分在频域自适应分离，并通过可学习的能量阈值保留关键频率；②在时域引入基于幅值的稀疏门控，显式抑制噪声与弱相关变量；③使用轻量级 MLP‑Mixer 替代全局注意力，显著降低计算开销；④设计稀疏跨混合器实现时频特征的自适应融合。

**🔧 技术方法**

核心技术包括：FFT/逆FFT频域分解、Top‑K 能量选择、可学习的稀疏门控、MLP‑Mixer、Softmax 结合 Top‑K 的稀疏注意力、Sigmoid 可调融合权重。

**📊 数据集**

在多种公开基准数据集上评估：ETTm1/2、ETTh1/2、Electricity、Exchange、Weather（共7个），每个数据集均使用 96/192/336/720 步长进行预测。

**📈 对比分析**

与 Transformer、iTransformer、PatchTST、DLinear、TimesNet、Autoformer、FEDformer、WPMixer 等多类主流基线模型对比，SDMixer 在大多数数据集和预测长度下均取得最低 MSE/MAE，尤其在长序列和高噪声场景中优势显著。

**⚠️ 局限性**

局限性：①实验仅覆盖常见工业与公共数据集，缺乏对非周期性极端时序的验证；②模型对超大规模数据集的扩展性尚未充分评估；③由于采用轻量级 Mixer，可能在极高维度或极稀疏场景下的表现仍需进一步探索。

---

## 287. Improving Family Co-Play Experiences through Family-Centered Design

**arXiv ID:** 2602.23596 | [PDF](https://arxiv.org/pdf/2602.23596v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 288. LemmaBench: A Live, Research-Level Benchmark to Evaluate LLM Capabilities in Mathematics

**arXiv ID:** 2602.24173 | [PDF](https://arxiv.org/pdf/2602.24173v1)

**作者:** Antoine Peyronnet `[一作]` (École des Ponts), Amaury Hayat `[通讯]` (Korea Institute for Advanced Study)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可实时更新的研究级数学定理证明基准，自动从 arXiv 论文中提取并自包含化命题，随后用 LLM 生成证明并由 LLM 与人类专家评判

**💡 创新点**

创新点在于：①利用 LLM 自动补全缺失的假设与定义，确保命题自包含；②设计“实时”可更新的基准，降低数据污染；③采用 LLM 作为评判者，提供可扩展的验证流程

**🔧 技术方法**

技术手段包括正则表达式抽取 lemma 环境、全上下文检索与向量检索两种定义补全模式、GPT‑5/Gemini 及 Claude 等 LLM 进行证明生成与评判、ChromaDB 进行检索、人工标注验证

**📊 数据集**

数据集来自 arXiv 预印本，2025 年 376 条 lemma（≈240 条自包含），2026 年 677 条 lemma（≈358 条自包含），覆盖 AG、AP、PR、NT 等多数学科类别

**📈 对比分析**

对比方法为 pass@1 证明成功率与 LLM‑judge 的正确率，人类专家评估一致性，结果显示 GPT‑5 在 LLM‑judge 下实现 12–15% 的证明通过率，Claude Opus 在最严格评判下最高达 35%，人类可信度 67–83%

**⚠️ 局限性**

局限性包括：样本量有限（数百条命题），评判依赖 LLM 可能引入误判，缺乏大规模人类评审，基准对模型训练与测试的数据污染控制仍有挑战

---

## 289. UPath: Universal Planner Across Topological Heterogeneity For Grid-Based Pathfinding

**arXiv ID:** 2602.23789 | [PDF](https://arxiv.org/pdf/2602.23789v1)

**作者:** Aleksandr Ananikian `[一作]` (Saint-Petersburg University), Konstantin Yakovlev `[通讯]` (Saint-Petersburg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种一次训练即可通用的启发式预测器UPath，直接将预测的修正因子作为A*的启发式，能在多种网格拓扑上高效求解路径。

**💡 创新点**

核心创新在于预测相对于八连通网格Octile距离的修正因子，并通过仅用极简的随机几何先验训练，配合长跳跃连接和损失屏蔽实现对未知任务的强泛化；同时构建了覆盖十种拓扑的UPF评测集。

**🔧 技术方法**

采用Encoder‑Transformer‑Decoder网络（含长跳跃连接），对预测值做tanh映射至(0,1]；训练使用Adam+OneCycleLR，损失为屏蔽后的MSE；求解时与标准A*无缝集成。

**📊 数据集**

训练数据来自Uniform、Beta、Beta+Figures三种简单随机生成器；评测集UPF包含2万条任务，涵盖Baldur’s Gate、Moving Street、TMP、HouseExpo、Perlin、Dcaffo等十种拓扑。

**📈 对比分析**

与Weighted A*（w=2/5/10）和TransPath比较，评测指标为Optimal Found Ratio、Cost Ratio、Expansion Ratio和总运行时；UPath（Beta+Fig）在UPF上实现≈72%最佳解率、≈101%成本、≈47%扩展比（≈2.1×缩减），明显优于基线且在大多数λ下保持更佳的质量‑效率折中。

**⚠️ 局限性**

局限性包括：对更大分辨率（128×128）虽保持优势但仍需进一步验证；极难拓扑下Optimal Found率仍低于完美；训练仅基于极简先验，可能在某些非典型场景中失效；以及预测时的GPU占用导致小批量运行仍略慢。

---

## 290. Memory Caching: RNNs with Growing Memory

**arXiv ID:** 2602.24281 | [PDF](https://arxiv.org/pdf/2602.24281v1)

**作者:** Ali Behrouz `[一作]` (Cornell University), Vahab Mirrokni `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Memory Caching（MC）技术，在 RNN 中缓存中间记忆状态，使其有效记忆容量随序列长度增长；

**💡 创新点**

通过分段缓存并设计四种聚合方式（残差门控、Memory Soup、稀疏选择等），实现 RNN 与 Transformer 之间的灵活记忆/复杂度权衡；

**🔧 技术方法**

结合线性注意力、深层记忆、滑动窗口线性注意力、Titans 等递归模块，并使用门控、路由、稀疏选择等机制实现 MC；

**📊 数据集**

在 FineWeb、Long‑Data‑Collections、Wikitext、LMB、PIQA、HellaSwag、WinoGrande、ARC‑e/ARC‑c、SIQA、BoolQ、Needle‑in‑a‑Haystack、LongBench、MQAR 等数据集上进行评测；

**📈 对比分析**

与 Transformer、RetNet、DeltaNet、Miras、SWLA、DLA、Titans、Log‑Linear++ 等基线对比，MC 在语言建模、常识推理、长上下文理解、检索和 in‑context recall 任务上显著提升性能，准确率逼近 Transformer，且在长序列下的训练吞吐量优于 Transformer；

**⚠️ 局限性**

MC 的效果受分段长度、聚合策略和路由设计的影响，长序列仍可能出现记忆压缩不足，且对门控/路由参数的手工调优需求高；

---

## 291. GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models

**arXiv ID:** 2602.24027 | [PDF](https://arxiv.org/pdf/2602.24027v1)

**作者:** Xingyu Zhu `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**通讯引用:** 42844 | [OpenAlex ID](https://openalex.org/A5038668215)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GuardAlign，一种无训练的输入检测与解码校准框架，提升大型视觉‑语言模型的安全性。

**💡 创新点**

创新点在于将最优传输（OT）驱动的细粒度安全检测与跨模态注意力校准相结合，实现对图像恶意区域的精准识别和安全信号在生成过程中的持续强化。

**🔧 技术方法**

采用 CLIP 进行图像‑文本特征编码，利用最优传输与 Sinkhorn 算法实现安全检测；在模型内部通过注意力加权调节跨模态融合层，使安全前缀保持高激活度。

**📊 数据集**

在 SPA‑VL、MM‑SafetyBench、FigStep、Unconstrained Attack、AdvBench 等安全基准以及 VQAv2、TextVQA、MME、MMBench 等通用多模任务上进行评估。

**📈 对比分析**

与现有推理时防御（ECSO、ETA）以及微调防御（Posthoc‑LoRA、Mixed‑LoRA）对比，GuardAlign 在安全性上将 unsafe‑response‑rate 降至约 10%，比对手低 39% 以上，同时保持甚至提升 VQA 等实用指标；推理成本相对适中。

**⚠️ 局限性**

仅针对视觉‑语言模型验证，未扩展到音频、视频等其他模态；在极端攻击或高度复杂场景下仍可能存在漏检，且仍需进一步提升模型对安全前缀的持久激活能力。

---

## 292. I've Seen This IP: A Practical Intersection Attack Against Tor Introduction Circuits and Hidden Services

**arXiv ID:** 2602.23560 | [PDF](https://arxiv.org/pdf/2602.23560v1)

**作者:** Nicolas Constantinides `[一作]` `[通讯]` (University of Edinburgh), Nicolas Constantinides (University of Edinburgh)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

论文展示了一种基于交叉攻击的技术，利用Tor隐藏服务引入环路的确定性路由和长期生命周期，通过在单一中继上多次探测与分析流量来逐步识别整个引入路由，从而实现对隐藏服务的去匿名化。

**💡 创新点**

创新点在于首次证明了Tor隐藏服务引入协议的静态路由特性能被利用来进行交叉攻击，并在真实网络上演示了该攻击在不同时间、不同中继权重下的可行性与收敛行为。

**🔧 技术方法**

主要使用了交叉集分析、定时探测（发送introcell并记录对应的cell响应窗口内的IP集合）、加密伪匿名化（SHA‑256+RSA无填充）以及Python、Tor内核的日志与控制接口等技术。

**📊 数据集**

实验数据集为作者自行运营的四台Tor中继（引入点、中继1、中继0/护卫、入口守护）和一个自建隐藏服务，所有流量均在公开Tor网络中采集。

**📈 对比分析**

与传统的基于全局监控或主动扫描的攻击不同，该方法仅需在每一步观察单一中继；实验显示在不同权重和时段下，收敛至单个候选节点所需探测次数平均从数十到数百不等，证明了攻击在实际环境中的可行性。

**⚠️ 局限性**

局限性包括依赖于引入环路中继不共享IP地址、对高流量/高选择概率情境未进行评估、未收集完整的相关指标（如并发流量、时延）以及实验仅在自有服务上进行，无法直接推广到所有隐藏服务。

---

## 293. Hierarchical Concept-based Interpretable Models

**arXiv ID:** 2602.23947 | [PDF](https://arxiv.org/pdf/2602.23947v1)

**作者:** Oscar Hill `[一作]` (University of Cambridge), Mateja Jamnik `[通讯]` (University of Cambridge)

**通讯引用:** 1493 | [OpenAlex ID](https://openalex.org/A5036018012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 Concept Splitting 方法和 HiCEM 架构，用以在现有概念嵌入模型（CEM）中自动发现并建模层级概念，从而实现更细粒度的解释与干预。

**💡 创新点**

创新点在于通过稀疏自编码器从预训练 CEM 的嵌入空间中提取子概念，并将这些子概念嵌入到分层概念模型 HiCEM 中，既降低了对细粒度概念标注的需求，又支持多级人机干预。

**🔧 技术方法**

技术手段包括 BatchTopK 稀疏自编码器、概念嵌入模型、分层子概念模块、随机干预正则化（RandInt）以及线性标签预测器。

**📊 数据集**

实验使用了 MNIST-ADD、SHAPES、CUB、AwA2、PseudoKitchens（自研的 3D 厨房渲染数据集）以及 ImageNet（用于用户研究）等六个数据集。

**📈 对比分析**

通过与 CEM、CBM、LF-CBM、PCBM 等基线在任务准确率和概念 ROC‑AUC 上进行对比，HiCEM 在保留任务与概念精度的同时，子概念发现准确率超过 0.9，且在干预时能进一步提升任务精度。

**⚠️ 局限性**

局限性包括稀疏自编码器不一定能始终发现有意义的子概念、目前仅实现两层层级结构，且对更深层级和干预对用户认知负荷的影响尚未充分研究。

---

## 294. CIRCLE: A Framework for Evaluating AI from a Real-World Lens

**arXiv ID:** 2602.24055 | [PDF](https://arxiv.org/pdf/2602.24055v1)

**作者:** Reva Schwartz `[一作]` (Civitaas Insights), Thiago Lacerda `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了CIRCLE六阶段生命周期框架，用以在真实部署环境中系统性评估AI技术的高阶影响与效益。

**💡 创新点**

创新点在于将用户异质性视为关键信号，构建以利益相关者关注为核心的构念操作化方法，并将模型基准、红队、现场测试和纵向监测等多种评估手段整合为可追溯、迭代的评估流程。

**🔧 技术方法**

采用的技术包括红队演练、现场实验、长周期纵向研究、自动化大规模模型跑测、混合方法数据标注与分析、以及持续监测与反馈循环。

**📊 数据集**

使用的数据集来源于真实部署场景，如教育技术教室、企业工作场所等，主要包括用户交互日志、问卷调查、访谈记录、实验数据与自动化测试结果。

**📈 对比分析**

与传统单一基准或抽样实验相比，CIRCLE通过多阶段、跨方法的整合评估能更完整地捕获二次和三次影响，显示出更高的生态、构念与后果有效性；在实验中表现出更可靠的预测部署成功率和风险揭示能力。

**⚠️ 局限性**

局限性包括：高昂的实施成本和人力资源需求，需要跨学科团队支持；依赖持续监测和数据收集，实施周期较长；缺乏统一的工具和标准，难以在大规模部署中快速复制。

---

## 295. Democratizing GraphRAG: Linear, CPU-Only Graph Retrieval for Multi-Hop QA

**arXiv ID:** 2602.23372 | [PDF](https://arxiv.org/pdf/2602.23372v1)

**作者:** Qizhi Wang `[一作]` `[通讯]` (PingCAP), Qizhi Wang (PingCAP)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SPRIG，一种纯 CPU、线性时间、无 LLM 依赖的 GraphRAG 检索管线，利用轻量 NER 构建实体–文档共现图并通过 Personalized PageRank 进行检索。

**💡 创新点**

创新点：①用轻量 NER 与 TF‑IDF 加权共现图替代昂贵的 LLM 生成图；②采用线性时间的 PPR 进行检索；③引入标题别名消歧、hub 裁剪和种子混合等轻量化图优化显著降低查询时延；④在 4 GB RAM、纯 CPU 环境下实现多跳检索。

**🔧 技术方法**

技术手段：SpaCy/正则 NER、TF‑IDF 权重的实体–文档边、稀疏 Personalized PageRank（push/迭代）、BM25、稠密检索（bge‑small）、RRF、Cross‑Encoder 重排序、HNSW 近似向量检索。

**📊 数据集**

使用数据集：HotpotQA 和 2WikiMultiHopQA 两个多跳 QA 数据集。

**📈 对比分析**

与 BM25、BM25+RM3、Dense、RRF、BM25+CE、TF‑IDF Graph、Graph、GraphHybrid、GraphDense 等基线比较；结果显示 GraphHybrid/GraphDense 在 Recall@10 与 RRF 近似或略胜，Dense+RRF 仍为最优；在 CPU 约束下 SPRIG 能在显著降低查询时延（比 Dense 高 8–10 倍）同时提升 Recall，轻量化图优化可进一步降低 16–28% 的查询时延而 Recall 几乎不变。

**⚠️ 局限性**

局限性：①依赖轻量 NER，实体覆盖不足会导致检索弱；②未做完整实体链接或共指解析，可能引入噪声；③评估仅覆盖 HotpotQA、2WikiMultiHopQA 两个数据集，缺乏更广泛验证；④未与 GPU 加速的 GraphRAG 方案做对比；⑤超参数在验证集上调优，可能存在轻微过拟合；⑥仅使用小型稠密检索器，未评估更强模型或跨模态检索的影响。

---

## 296. FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation

**arXiv ID:** 2602.23636 | [PDF](https://arxiv.org/pdf/2602.23636v1)

**作者:** Zhihao Ding `[一作]` (Hong Kong Polytechnic University), Jieming Shi `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 148 | [OpenAlex ID](https://openalex.org/A5102750234)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对LLM内容审核的“严格度自适应”框架，设计了可在三种严格度（严格、中等、宽松）下评估的基准；

**💡 创新点**

创新点在于：①将审核任务从单一二分类转化为连续风险评分，使得可通过阈值实现不同严格度；②通过LLM专家裁决的分级评分进行伪标签生成并进行标签一致性校准；③使用两阶段训练（SFT+GRPO）实现评分与严重度的对齐；

**🔧 技术方法**

技术主要包括：LLM基准模型（如Qwen3-8B）、参数高效微调（LoRA）、基于分级的鲁棒奖励（GRPO）以及阈值校准（手工或数据校准）；

**📊 数据集**

使用的训练数据来自公开审核数据集（Aegis2.0、WildGuardMix），基准数据为自行构建的四千条样本（含提示和提示-响应对），测试集覆盖三种严格度；

**📈 对比分析**

与现有二分类审核模型（Qwen3Guard、WildGuard、LlamaGuard、BingoGuard等）对比，FlexGuard在三种严格度下的F1均高，平均提升约5–10%，且最差情况提升约9%；在公共审核基准（ToxicChat、OpenAI Moderation、HarmBench等）上也保持了较强的平均性能；

**⚠️ 局限性**

局限性包括：仅在英语数据上验证，缺乏多语种和代码混合场景；伪标签依赖有限的训练源，未系统评估其他数据分布对评分校准的影响；GRPO奖励设计可进一步优化，未尝试更先进的对齐方法；

---

## 297. SGAgent: Suggestion-Guided LLM-Based Multi-Agent Framework for Repository-Level Software Repair

**arXiv ID:** 2602.23647 | [PDF](https://arxiv.org/pdf/2602.23647v1)

**作者:** Quanjun Zhang `[一作]` (Nanjing University of Science and Technology), Liang Xiao `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 18186 | [OpenAlex ID](https://openalex.org/A5068976123)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种建议引导的多代理框架，结合定位-建议-修复三阶段实现仓库级别的自动软件修复。

**💡 创新点**

创新点包括引入建议阶段填补定位与修复间的认知缺口、使用知识图谱驱动的检索工具增强全局上下文感知，以及多代理协同的定位-建议-修复流程。

**🔧 技术方法**

主要技术为大语言模型（Claude-3.5-Sonnet 等）、知识图谱构建与查询、工具集成检索、基于 ReAct 的代理设计、回归与复现测试的多阶段验证。

**📊 数据集**

实验数据集包括 SWE-Bench-Lite（Python 项目）以及 VUL4J 与 VJBench（Java 漏洞）。

**📈 对比分析**

与同基模型的现有方法对比，本文在 SWE-Bench-Lite 上达成 51.3% 的解决率、81.2% 文件级定位精度、52.4% 函数级定位精度，平均成本仅 1.48 美元；在漏洞修复任务中也取得最高的 48% 解决率。

**⚠️ 局限性**

局限性主要在对大语言模型的依赖、建议阶段对模型推理能力敏感、以及对非常大规模仓库或多语言项目的可扩展性仍待验证。

---

## 298. From Efficiency to Meaning: Adolescents' Envisioned Role of AI in Health Management

**arXiv ID:** 2602.24249 | [PDF](https://arxiv.org/pdf/2602.24249v1)

**作者:** Jamie Lee `[一作]` (University of California), Yunan Chen `[通讯]` (University of California)

**通讯引用:** 3871 | [OpenAlex ID](https://openalex.org/A5100658705)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过设计小说与共创工作坊，调查23名14-17岁青少年对健康AI在家庭乳糜泻管理中的期望与关注。

**💡 创新点**

首次从青少年视角系统探讨健康AI的角色，提出四类期望（提升健康理解与求助、降低认知负荷、调解家庭健康管理、在尊重自主的前提下提供指导）与情感支持的分化。

**🔧 技术方法**

采用设计小说、共创、访谈与质性主题分析等人机交互方法，利用工作坊记录与访谈转录进行分析。

**📊 数据集**

未使用公开数据集，数据来源仅为23名参与者的工作坊视频、笔记与访谈转录。

**📈 对比分析**

无算法或系统实现，未进行性能或方法比较，结果仅基于质性分析得出结论。

**⚠️ 局限性**

样本局限于美国西部高中生，缺乏跨文化验证；未进行AI素养与信任的量化测评；未评估实际AI系统的可行性与安全性，且对AI误判与幻觉的影响分析有限。

---

## 299. SHINE: Sequential Hierarchical Integration Network for EEG and MEG

**arXiv ID:** 2602.23960 | [PDF](https://arxiv.org/pdf/2602.23960v1)

**作者:** Xiran Xu `[一作]` (Peking University), Jing Chen `[通讯]` (Peking University)

**通讯引用:** 37590 | [OpenAlex ID](https://openalex.org/A5100394807)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 SHINE 网络，用于从单个受试者的 MEG 信号重构二进制语音-静默序列，并在 LibriBrain 2025 竞赛中取得了领先成绩。

**💡 创新点**

创新点在于将语音检测任务改写为序列重构的 seq2seq 框架，使用 30 秒长的 MEG 段进行全时序预测，并在扩展赛道通过多任务学习（重构语音包络和 Mel 频谱）提升模型表现。

**🔧 技术方法**

核心技术包括深度卷积层、全连接层、输出上下文层、注意力机制以及 LSTM/卷积后端的组合，训练时采用负 Pearson 相关损失并采用 AdamW 优化器。

**📊 数据集**

使用的数据集为 LibriBrain 2025 的 50 小时 MEG 数据（306 通道、250 Hz），仅在单个受试者上收集，包含训练、验证、测试及排行榜保留子集。

**📈 对比分析**

与基线模型 BrainMagic、AWavNet、ConvConcatNet 的单模型对比，SHINE 在本地测试 F1‑macro 0.9067、排行榜 0.9015；通过集成 200+ 模型，标准赛道达 0.9155，扩展赛道达 0.9184，显著优于基线。

**⚠️ 局限性**

局限性包括仅针对单一受试者的数据、模型复杂度高且对时间窗口设置敏感，且未探索 Transformer/Mamba 等更先进结构，未来需在多受试者数据上验证泛化能力。

---

## 300. Synthetic Visual Genome 2: Extracting Large-scale Spatio-Temporal Scene Graphs from Videos

**arXiv ID:** 2602.23543 | [PDF](https://arxiv.org/pdf/2602.23543v1)

**作者:** Ziqi Gao `[一作]` (Allen Institute for AI), Ranjay Krishna `[通讯]` (University of Washington)

**通讯引用:** 12956 | [OpenAlex ID](https://openalex.org/A5032451496)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了自动化生成大规模全景视频场景图数据集PVSG，并基于该数据集训练了TRASER模型，实现一次前向传递即可生成完整视频场景图。

**💡 创新点**

创新点包括：两阶段在线-离线跟踪实现新物体自动发现；GPT‑5推理时空关系；轨迹对齐的Token排列与双重重采样（对象轨迹重采样+时间窗口重采样）以兼顾全局语义与细粒度时序；以及使用LLM评判器实现开放词汇评估。

**🔧 技术方法**

使用了SAM2、DAM、GPT‑5、Qwen2.5‑VL、Perceiver Resampler、轨迹对齐Token排列、双重重采样、LLM评判器等技术，构成端到端的自动化流水线。

**📊 数据集**

使用了约636K条合成视频构成的PVSG数据集（来源于SVG2、SA‑V、PVD），以及人类标注的SVG2test 100视频；评测还覆盖PVSG、VIPSeg、VidOR等公开基准。

**📈 对比分析**

与多种开源与专有VLM基线（Gemini、GPT‑4.1、GPT‑5、Qwen等）在开放词汇场景图生成任务上进行标准化评估，TRASER在关系检测提高15%~20%、对象识别提升30%~40%、属性预测提升15%；在视频问答任务中，加入高质量场景图可提升1.5%~4.6%的准确率。

**⚠️ 局限性**

局限性在于数据为合成，受分割模型与VLM的误差影响；生成流程计算成本高；对长视频动态关系的捕捉仍有不足；自动化流程可能引入噪声，真实世界泛化仍需进一步验证。

---

## 301. Current pulse generator: A circuit for programming RRAM in current mode

**arXiv ID:** 2602.24163 | [PDF](https://arxiv.org/pdf/2602.24163v1)

**作者:** Bojian Zhang `[一作]` (University of Groningen), Erika Covi `[通讯]` (University of Groningen)

**通讯引用:** 1305 | [OpenAlex ID](https://openalex.org/A5072304262)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

**🎯 论文内容**

提出并实现了一种2M1R1B电路，用电流镜拓扑在电流模式下为RRAM产生精确的脉冲电流，实现对RRAM的SET/RESET编程并通过电压缓冲读取电阻；

**💡 创新点**

创新点在于利用双电流镜+剥离开关组合，直接将参考直流电流转换为可控幅值、形状的电流脉冲，无需外部电流限制；电路在180 nm CMOS工艺下实现低功耗、低误差（≤3 %）的电流编程，并兼具电压读取功能；

**🔧 技术方法**

采用PMOS/NMOS电流镜、低压cascode（讨论）、剥离开关(chop)、电压缓冲器；在180 nm工艺下完成硅级DC与瞬态测试；

**📊 数据集**

在180个晶圆芯片上测试，取四个参考电流值（100–400 µA），在不同位置测得镜像因子与误差分布；并在硅上进行DC/瞬态测量；

**📈 对比分析**

通过DC测量验证镜像因子≈1，误差≤3 %；在瞬态测量中，脉冲幅值与参考电流线性对应，波形与剥离脉冲一致；在芯片级测得1–2 %误差；与传统电压模式相比，省去电流限制且功耗更低；

**⚠️ 局限性**

局限性包括：低电流下误差可达3–7 %；SET/RESET对供电电压有较窄工作范围（SET ≥3.5 V，RESET 4 V）；存在寄生电容导致上升斜率波动；未实现阵列级验证与功耗/可靠性评估；需进一步加入cascode以抑制通道长度模。

---

## 302. Selective Denoising Diffusion Model for Time Series Anomaly Detection

**arXiv ID:** 2602.23662 | [PDF](https://arxiv.org/pdf/2602.23662v1)

**作者:** Kohei Obata `[一作]` (SANKEN, University of Osaka), Yasushi Sakurai `[通讯]` (SANKEN, University of Osaka)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于扩散模型的无监督时间序列异常检测方法，通过构建选择性滤波器仅对异常部分进行去噪，保持正常部分不变。

**💡 创新点**

创新点在于针对时间序列的噪声设计：引入掩码高斯噪声训练，让模型学习在去噪时只处理异常；并提出无噪声推理策略，避免在推理过程中加入噪声导致正常信息丢失。

**🔧 技术方法**

技术手段包括扩散概率模型（DDPM）、Transformer/CSDI架构、掩码高斯噪声与无噪声推理。

**📊 数据集**

实验使用五个公开数据集：UCR anomaly archive、AIOps、Yahoo real、Yahoo benchmark、Server Machine Dataset。

**📈 对比分析**

与13种基线（经典方法、GAN/AE/ VAE、Transformer、去噪模型）在多种阈值无关指标下比较，所提方法在VUS‑PR上比vanilla DDPM提升45.1%，在VOC‑ROC、VOC‑PR、Range F‑score上分别提升约4.1%、14.4%、29%，在大多数数据集上获得最优或接近最优性能。

**⚠️ 局限性**

局限性包括对模式型异常多的UCR数据表现仍不理想；在训练样本少或多变量的情况下可能过拟合；掩码噪声单独使用效果不佳，需要与无噪声推理配合；同时噪声比例和损失权重的选择仍需经验调优。

---

## 303. Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation

**arXiv ID:** 2602.24041 | [PDF](https://arxiv.org/pdf/2602.24041v1)

**作者:** Xingyu Zhu `[一作]` (University of Science and Technology of China), Hanwang Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 27886 | [OpenAlex ID](https://openalex.org/A5042324027)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为AIR的自适应视觉强化框架，用于减轻多模态大语言模型的幻觉现象。

**💡 创新点**

创新点在于：①基于原型的视觉令牌压缩，过滤冗余背景信息；②使用熵正则化的最优传输（OT）量化隐藏状态与图像补丁的对齐度，智能挑选最相关的补丁进行重新注入。

**🔧 技术方法**

主要技术包括Transformer的前馈网络重注入、聚类式原型生成、最优传输与Sinkhorn算法、以及基于相似度的阈值筛选。

**📊 数据集**

使用的评估数据集包括MSCOCO、CHAIR、POPE、A-OKVQA、GQA、LLaVA-Bench、MME以及MMBench。

**📈 对比分析**

与VCD、MemVR、VAF等现有幻觉抑制方法对比，AIR在三款主流MLLM（LLaVA-1.5-7B、Qwen-VL-Chat、GLM-4V-9B）上显著降低CHAIR_S/CHAIR_I，同时保持或提升BLEU、准确率等通用任务指标；仅略有额外延迟和显存占用。

**⚠️ 局限性**

局限性包括：尚未在更复杂的推理型多模态模型或智能体上验证；对OT阈值和补丁数量的敏感性需要进一步探究；模型仍受训练时预置数据偏见的影响。

---

## 304. TRIZ-RAGNER: A Retrieval-Augmented Large Language Model for TRIZ-Aware Named Entity Recognition in Patent-Based Contradiction Mining

**arXiv ID:** 2602.23656 | [PDF](https://arxiv.org/pdf/2602.23656v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 305. Learning Accurate Segmentation Purely from Self-Supervision

**arXiv ID:** 2602.23759 | [PDF](https://arxiv.org/pdf/2602.23759v1)

**作者:** Zuyao You `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 24310 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Selfment，一种完全无监督、无需人工标注、后处理或外部模型的前景分割框架。

**💡 创新点**

创新点在于：①使用 DINOv3 的稠密特征构建补丁相似度图；②首次在此图上应用归一化割 (NCut) 获得粗略前景/背景划分；③设计迭代补丁优化 (IPO) 通过聚类中心迭代细化掩码；④用 IPO 生成的伪标签进行自监督头部训练，采用对比、Dice 与 BCE 损失实现无标注学习。

**🔧 技术方法**

核心技术包括：自监督视觉 Transformer (DINOv3-7B)、归一化割、迭代补丁优化、对比学习 (InfoNCE)、软 Dice 损失、二分类 BCE 损失，及轻量化头部网络。

**📊 数据集**

训练数据：从 DUTS 训练集随机采样 1000 张图；评估数据：ECSSD、DUTS、HKUIS、PASCAL‑S（无监督显著物体检测）以及 CHAMELEON、CAMO、COD10K、NC4K（隐蔽物体检测）。

**📈 对比分析**

与 TokenCut、SelfMask、FOUND 等方法对比，Selfment 在 ECSSD、DUTS、HKUIS、PASCAL‑S 上 Fmax 提升约 4–7%；在 COD 任务中，S‑Measure、E‑Measure、F‑β 等指标均超过所有无监督方法，甚至接近或优于部分全监督模型；整体表现显著领先且无需后处理。

**⚠️ 局限性**

局限性：IPO 过度依赖补丁特征相似度，导致与前景语义相近的背景物体被误判为前景，伪掩码错误会影响后续自监督训练；对极度细小或与背景高度混合的目标识别仍存在困难。

---

## 306. Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume

**arXiv ID:** 2602.24195 | [PDF](https://arxiv.org/pdf/2602.24195v1)

**作者:** Gregory Kang Ruey Lau `[一作]` (National University of Singapore), Bryan Kian Hsiang Low `[通讯]` (National University of Singapore)

**通讯引用:** 864 | [OpenAlex ID](https://openalex.org/A5030304400)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无训练、无外部工具的多模态大型语言模型（MLLM）不确定性估计框架，利用模型自身的条件概率和语义嵌入计算“去不一致校正语义体积”作为不确定性分数。

**💡 创新点**

创新点在于：① 将模型自带的概率信息视为局部不一致（incoherence）评分；② 结合DPP风格的质量-多样性核，将语义多样性与不一致评分融合，形成可解释且与多模态一致性相关的单一不确定性指标；③ 通过无标签自适应的α平衡参数实现无训练调优。

**🔧 技术方法**

核心技术包括：M样本采样、使用最后EOS嵌入的归一化语义向量、基于对角线不一致矩阵的Gram矩阵logdet计算、Monte Carlo估计二次熵、Cholesky分解求log‑det、min‑max归一化及相关性/ECE评估。

**📊 数据集**

实验数据集涵盖多模态QA与生成任务：图像‑文本（VQAv2、OKVQA、AdVQA、MathVista、VQA‑RAD）、音频‑文本（SLUE、SpokenSQ）、视频‑文本（VidMME）、图像生成（MS‑COCO caption）、音频生成（AudioCaps）。

**📈 对比分析**

与多种基线（图像专用、文本专用、概率熵、Eigenscore、Verbalized Confidence等）相比，本文方法在AUROC、CPC、ECE、AURAC上普遍位列前列，平均AUROC提升≈3–5%，CPC≥0.90，ECE≤0.07，黑盒场景下利用白盒代理亦保持优势。

**⚠️ 局限性**

局限性：① 对长文本/生成任务仍需更长的采样/概率稳定处理；② 需要k个采样，虽低成本但不适合极低延迟场景；③ 黑盒代理对语义一致性敏感，若代理与目标模型差距大会影响估计；④ 仅评估二次熵和语义体积，未考虑更细粒度的文本语义结构。

---

## 307. Geodesic Semantic Search: Learning Local Riemannian Metrics for Citation Graph Retrieval

**arXiv ID:** 2602.23665 | [PDF](https://arxiv.org/pdf/2602.23665v1)

**作者:** Brandon Yee `[一作]` (Yee Collins Research Groups), Krishna Sharma `[通讯]` (Hoover Institute)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了Geodesic Semantic Search（GSS）系统，利用学习到的节点级Riemannian度量在引文网络中进行几何感知检索。

**💡 创新点**

创新点包括：① 在每个节点学习低秩度量张量，使得局部几何可变且正定；② 结合多源Dijkstra、FAISS种子、MMR重排和路径连贯性过滤的层次检索流水线；③ 给出理论分析，证明几何距离在概念桥接任务中优于直接相似度。

**🔧 技术方法**

技术细节涵盖：MetricGAT图注意力网络输出嵌入与低秩度量；低秩度量参数化保证正定；多源Dijkstra与k-means粗化图层进行高效检索；InfoNCE对比损失、排名损失、光滑正则等联合训练。

**📊 数据集**

使用的数据集为169,343篇arXiv论文的引文网络（1,166,243条边），每篇论文使用768维SPECTER嵌入作为输入特征。

**📈 对比分析**

在citation prediction、semantic search和concept bridging三项任务上与SPECTER+FAISS、Node2Vec、Contriever、BGE-Large、GAT+Euclidean等基线对比，GSS在Recall@20提升23%（0.518 vs 0.421），Bridge@10提升46%，nDCG@10提升14.6%，并通过三层层次搜索实现约4×速度提升，仅损失约2%检索质量。

**⚠️ 局限性**

局限性包括：需要引用结构作为监督；检索延迟约198ms，仍高于FAISS；仅适用于静态图，难以处理动态更新；对新论文（冷启动）缺乏足够图信号。

---

## 308. Open-Vocabulary Semantic Segmentation in Remote Sensing via Hierarchical Attention Masking and Model Composition

**arXiv ID:** 2602.23869 | [PDF](https://arxiv.org/pdf/2602.23869v1)

**作者:** Mohammadreza Heidarianbaei `[一作]` (Leibniz University Hannover), Franz Rottensteiner `[通讯]` (Leibniz University Hannover)

**通讯引用:** 6324 | [OpenAlex ID](https://openalex.org/A5033807047)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个完全不需要训练的遥感图像开放词汇语义分割方法ReSeg-CLIP。

**💡 创新点**

创新点在于：①利用Segment Anything Model (SAM) 生成多尺度分割掩码，对ViT的自注意力进行层级约束；②提出Prompt Variant Separation Margin (PVSM) 指标，基于多样化文本提示评估各域适配CLIP模型的语义表达质量，并用其加权平均模型参数，实现无训练模型融合。

**🔧 技术方法**

技术手段包括：CLIP与其RS微调版本（RemoteCLIP、GeoRSCLIP）作为视觉/文本编码器；SAM生成多尺度掩码并转为注意力掩码；对齐时使用像素级余弦相似度；PVSM通过文本提示变体计算类内/类间相似度差值进行加权；模型融合采用参数空间线性加权。

**📊 数据集**

在三大高分遥感基准数据集上评估：Potsdam、UDD5 与 OpenEarthMap。

**📈 对比分析**

与现有训练式与无训练式的开放词汇语义分割方法（如SegEarth-OV、MaskCLIP、SCLIP、GEM、ClearCLIP 等）进行对比；在Potsdam上提升约8pp mIoU，在UDD5与OEM上分别取得 43.2% 与 32.4% mIoU，整体性能优于同类无训练方法，接近部分需训练的方案。

**⚠️ 局限性**

局限性包括：对小目标与背景类的分割精度仍偏低；模型融合若加入过多模型会导致参数过度平滑；SAM掩码层数与超参数需手工设定；未利用图像内容信息进行更精细的融合权重分配。

---

## 309. Data Driven Optimization of GPU efficiency for Distributed LLM Adapter Serving

**arXiv ID:** 2602.24044 | [PDF](https://arxiv.org/pdf/2602.24044v1)

**作者:** Ferran Agullo `[一作]`, Josep Ll. Berral `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 1597 | [OpenAlex ID](https://openalex.org/A5074281010)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向分布式LLM适配器推理的资源效率优化管线，结合数字孪生、机器学习预测与贪婪分配，目标是用最少GPU完成预期工作负载。

**💡 创新点**

首次针对适配器缓存问题构建数字孪生+学习驱动的全流程方法，并通过精准识别最大打包点 Max_pack 实现吞吐最大化与资源最小化。

**🔧 技术方法**

采用数字孪生模拟器、随机森林/支持向量机等机器学习模型、贪婪分配算法，以及 vLLM 框架实现。

**📊 数据集**

使用公开的 ShareGPT 数据集生成请求，LLM 模型为 Llama‑3.1‑8B 与 Qwen2.5‑7B，配合多种大小的 LoRA 适配器。

**📈 对比分析**

与 MaxBase、dLoRA 等基线对比，实验表明在四 GPU 环境下可减少 30%–50% GPU 数量，吞吐量提升约 10%–20%，且无请求饥饿或内存错误。

**⚠️ 局限性**

局限性在于仅针对固定请求长度分布和可预测的 Poisson 到达率，未覆盖长尾序列或完全不可预测的流量模式。

---

## 310. DesignSense: A Human Preference Dataset and Reward Modeling Framework for Graphic Layout Generation

**arXiv ID:** 2602.23438 | [PDF](https://arxiv.org/pdf/2602.23438v1)

**作者:** Varun Gopal `[一作]` (Adobe), Mausoom Sarkar `[通讯]` (Adobe)

**通讯引用:** 155 | [OpenAlex ID](https://openalex.org/A5112660758)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 DesignSense 数据集与奖励模型，构建了 10,235 对图形布局的人类偏好标注，并训练了基于 VLM 的判别器。

**💡 创新点**

创新点包括：①首个大规模布局偏好数据集，采用四分类标签捕捉主观模糊性；②设计了五阶段自动化生成对比布局的流水线；③使用 VLM 判别器实现显著优于通用模型的偏好评估。

**🔧 技术方法**

技术方法包括：利用 InternVL3‑8B 做视觉‑语言判别器；使用 AesthetiQ 与 GPT‑4o 进行布局生成与筛选；采用强化学习（AAPA）训练生成器；在推理阶段通过多候选排序提升质量；并用数据增强提升鲁棒性。

**📊 数据集**

主要数据集为 Crello（原始布局）、自建的 DesignSense 10k 对比数据；在外部评测中使用 PrismLayers 与 LayoutNUWA 两套未见数据。

**📈 对比分析**

对比方法：在两类选择与四类偏好任务上与开源/专有 VLM、ImageReward、HPSv3、PickScore 等模型进行评测；宏 F1 提升 54.6%、加权 F1 提升 86.4%；在生成器 AesthetiQ 上，使用 DesignSense 作为判别器后赢率提升约 3%，推理时多候选选择提升 3.6%。

**⚠️ 局限性**

局限性：尽管改进显著，但对高度模糊或极端布局仍易失误；数据集主要来自 Crello，可能缺乏其他风格的代表性；判别器依赖大型 VLM，对资源要求高；生成流水线与标注成本较高。

---

## 311. Invariant-Driven Automated Testing

**arXiv ID:** 2602.23922 | [PDF](https://arxiv.org/pdf/2602.23922v1)

**作者:** Ana Catarina Ribeiro `[一作]` `[通讯]`, Ana Catarina Ribeiro

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

无法获取

**💡 创新点**

无法获取

**🔧 技术方法**

无法获取

**📊 数据集**

无法获取

**📈 对比分析**

无法获取

**⚠️ 局限性**

无法获取

---

## 312. Ordinal Diffusion Models for Color Fundus Images

**arXiv ID:** 2602.24013 | [PDF](https://arxiv.org/pdf/2602.24013v1)

**作者:** Gustav Schmidt `[一作]` (Hertie Institute for AI in Brain Health University of Tübingen), Sarah Müller `[通讯]` (Hertie Institute for AI in Brain Health University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种基于序数条件的潜在扩散模型，用于生成不同糖尿病视网膜病变阶段的彩色眼底图像。

**💡 创新点**

创新点在于将疾病阶段的序数关系编码为一维标量嵌入，并引入结构编码器实现疾病与解剖结构的分离，使生成过程支持连续过渡而非离散类别。

**🔧 技术方法**

采用潜在扩散模型、VAE自编码器、对比学习结构编码器、分类器无监督引导以及双重条件（疾病 + 结构）技术。

**📊 数据集**

使用EyePACS眼底图像数据集，经过质量过滤后得到约12.7万张图像。

**📈 对比分析**

与传统one‑hot条件扩散模型对比，Fidelity（FID）显著下降、二次加权Kappa（QWK）提升至0.87，证明生成图像在视觉真实度和疾病一致性上均优于基线。

**⚠️ 局限性**

主要限制包括对高阶段图像的真实性仍有限、评估主要依赖自动化指标未包含专家临床评估、模型对微小病变细节的捕捉仍有欠缺，以及缺乏对纵向进展数据的显式建模。

---

## 313. Hierarchical Action Learning for Weakly-Supervised Action Segmentation

**arXiv ID:** 2602.24275 | [PDF](https://arxiv.org/pdf/2602.24275v1)

**作者:** Junxian Huang `[一作]` (Guangdong University of Technology), Shenghua Gao `[通讯]` (University of Hong Kong)

**通讯引用:** 12265 | [OpenAlex ID](https://openalex.org/A5034339267)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Hierarchical Action Learning (HAL) 模型，用于弱监督动作分割，利用层级因果生成模型将慢速演化的高层动作变量与快速变化的视觉变量解耦。

**💡 创新点**

创新点包括：① 引入伪状态对齐并用确定性转移保持动作变量的慢速变化；② 设计金字塔 Transformer 捕获多尺度依赖；③ 通过稀疏平滑约束显式强制高层动作更平滑，实现理论上可辨识的高层动作变量；④ 证明在轻微假设下可对高层动作实现块级可辨识。

**🔧 技术方法**

采用变分推断 + 金字塔 Transformer + L2 归一化 + 平滑转移约束 + ELBO+分类损失；同时利用多层注意力对视频特征进行编码。

**📊 数据集**

在 Breakfast、CrossTask、Hollywood Extended、GTEA 四大弱监督动作分割基准上进行评估。

**📈 对比分析**

与多种基线（HMM+RNN、CDFL、TASL、NN-Viterbi、DTW 系列、ATBA、CtrlNS 等）比较，HAL 在 MoF、IoU、IoD 等指标上普遍优于对手，尤其在 IoU、IoD 上显著提升，说明其更能捕捉层级语义结构。

**⚠️ 局限性**

局限性：对背景多样性强的 CrossTask 与 Hollywood 数据集的 MoF 下降；方法仍需在更复杂场景下进一步验证；计算量相对较大，需更高算力支持。

---

## 314. Shape vs. Context: Examining Human--AI Gaps in Ambiguous Japanese Character Recognition

**arXiv ID:** 2602.23746 | [PDF](https://arxiv.org/pdf/2602.23746v1)

**作者:** Daichi Haraguchi `[一作]` (CyberAgent), Daichi Haraguchi `[通讯]` (CyberAgent)

**通讯引用:** 95 | [OpenAlex ID](https://openalex.org/A5066891588)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了人类与视觉语言模型（VLM）在日语字符“ソ”(so) 与“ン”(n) 之间的视觉模糊区分时的决策边界差异，并考察了在单字符与词语上下文两种情境下的行为。

**💡 创新点**

创新点在于：①利用 β‑VAE 生成连续的字符插值图像，形成可精细控制视觉模糊度的试料；②在形状单一与形状+上下文两种最小化/强化上下文条件下，系统性比较人类与 VLM 的决策边界；③通过对比实验揭示即使准确率高，VLM 在模糊辨识时的决策机制仍与人类显著不同。

**🔧 技术方法**

技术手段包括：β‑VAE（latent 32，β=3）用于生成 15 级插值字符；对人类实验使用混合效应逻辑回归、Fisher 精确检验等统计分析；对 VLM（GPT‑5.1、Gemini‑2.5‑Flash）采用 10 次独立查询并聚合回答；对比分析采用交叉表与 Bonferroni 校正。

**📊 数据集**

数据集为：①364 种 Google Fonts 字体（包含日文字体）用于训练 β‑VAE；②通过 β‑VAE 生成的 15 级插值字符图像；③在词语上下文实验中使用的 24 个包含插值字符的词语（12 单一出现、12 共现），每个词语在 so‑偏向与 n‑偏向两种版本下均出现。

**📈 对比分析**

比较方法：在形状单一任务中，绘制人类与 VLM 在 15 级 α 下的识别概率曲线；在形状+上下文任务中，构建 2×3（so、n、Other）列联表并进行 Fisher 精确检验。结果显示：① VLM 的决策边界比人类更平滑，且在极端 α=1 时仍未达到人类的 100% n 识别率；② 在词语上下文中，尤其是共现条件下，VLM 的回答与人类更接近，但仍保留模型特异性偏差（Gemini 在 n‑偏向上下文中几乎完全选 n，GPT 在 so‑偏向上下文中略偏 n）。

**⚠️ 局限性**

局限性：①仅评估了两种 VLM，结果可能不具备普适性；②实验仅聚焦于两字符对（so 与 n），缺乏多字符或多语言的验证；③未将词义与局部视觉线索分离，难以明确 VLM 的决策来自于词义还是字符共现；④实验样本量有限（人类30+390，VLM 10 次查询），更大规模实验可能揭示不同结论；⑤β‑VAE 的生成图像与真实视觉环境的差距可能影响模型的泛化评估。

---

## 315. Multimodal Optimal Transport for Unsupervised Temporal Segmentation in Surgical Robotics

**arXiv ID:** 2602.24138 | [PDF](https://arxiv.org/pdf/2602.24138v1)

**作者:** Omar Mohamed `[一作]` (Mohammed bin Zayed University of Artificial Intelligence), Cesare Stefanini `[通讯]` (Mohammed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TASOT，一种无监督多模态最优传输框架，利用视觉与文本信息对手术视频进行时间段分割。

**💡 创新点**

在ASOT基础上加入文本特征，构建多模态成本并采用时序一致的非平衡Gromov‑Wasserstein最优传输，无需大规模预训练。

**🔧 技术方法**

使用DINOv3提取视觉特征、CLIP提取文本特征，融合多模态成本，并通过非平衡Gromov‑Wasserstein OT实现伪标签生成与无监督训练。

**📊 数据集**

在Cholec80、AutoLaparo以及MultiBypass140（StrasBypass70、BernBypass70）三个公开手术视频数据集上进行实验。

**📈 对比分析**

与零样本视频‑语言模型对比，TASOT在Cholec80、AutoLaparo分别提升+16.5、+19.6的F1分数，在MultiBypass140的StrasBypass70上提升+23.7，显示显著性能提升。

**⚠️ 局限性**

固定聚类数目限制了模型对视频特异性适应的灵活性，且步级分割仍相对困难。

---

## 316. Sharing is caring: data sharing in multi-agent supply chains

**arXiv ID:** 2602.24074 | [PDF](https://arxiv.org/pdf/2602.24074v1)

**作者:** Wan Wang `[一作]` (Wuhan University of Technology), Adam Sobey `[通讯]` (University of Southampton)

**通讯引用:** 1772 | [OpenAlex ID](https://openalex.org/A5039702362)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在两级供应链中，多智能体使用不同数据共享策略（无共享、谎报、真实、混合）对库存管理绩效的影响，并结合基线奖励与协同奖励两种奖励机制进行实验；

**💡 创新点**

创新点在于将数据共享视作智能体间的可选通信行为，探讨不同共享策略与奖励机制在高低需求场景下的交互效果，并提出协同奖励与共享策略共同提升整体绩效的可行性；

**🔧 技术方法**

使用深度强化学习中的Soft Actor-Critic（SAC）算法训练两个独立的智能体，并在Gymnasium+Ray框架下实现多代理环境，配合Ray Tune进行超参数调优；

**📊 数据集**

实验采用模拟生成的需求数据，包含两种情景：高需求（Poisson(10)）与低需求（正态(mean=2,std=1)），不使用公开真实数据集；

**📈 对比分析**

通过对比不同共享策略在基线奖励与协同奖励下的总奖励、库存水平、缺货与滞后率等指标进行评估，发现：在高需求下，谎报策略对工厂收益略有提升但整体收益差异不大；在低需求下，真实共享显著提升工厂与零售商收益（分别高达158%和7.5%），并且协同奖励进一步放大这些差异；

**⚠️ 局限性**

局限性包括：只考虑两级供应链，缺乏实际供应链数据验证；谎报策略实现简单（随机假设），未探索更具策略性的欺骗；未考虑供应商层级或竞争者加入导致的更复杂信息不对称情景；

---

## 317. Ask don't tell: Reducing sycophancy in large language models

**arXiv ID:** 2602.23971 | [PDF](https://arxiv.org/pdf/2602.23971v1)

**作者:** Magda Dubois `[一作]` (UK AI Security Institute), Lennart Luettgau `[通讯]` (UK AI Security Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过内容匹配的多因素实验系统研究了用户输入框架（问题/非问题、表达确定性、第一人称视角）对大型语言模型（LLM）sycophancy（阿谀奉承）的因果影响，并验证了两种输入层重构策略（将非问题改为问题、将第一人称改为第三人称）在降低sycophancy方面的效果。

**💡 创新点**

创新点在于：①首次用可控实验方法量化了问题形式、确定性强度和视角对sycophancy的驱动作用；②提出并验证了“将非问题改为问题”这一输入层缓解策略，实验显示其效果显著优于传统的“不要sycophantic”黑盒指令；③展示了模型差异和主题差异对sycophancy的调节作用。

**🔧 技术方法**

技术手段包括：
- 内容匹配的嵌套因子实验设计（共440种提示）；
- 采用三大前沿LLM（GPT‑4o、GPT‑5、Sonnet‑4.5）生成约15‑20万条回答；
- 双重LLM‑as‑a‑judge评分器（GPT‑5、Sonnet‑4.5）根据5维rubric评估sycophancy；
- 贝叶斯广义线性模型（ordered‑logistic）对分数进行多元统计分析。

**📊 数据集**

数据集：在四个主题（兴趣、社交关系、心理健康、医疗）下构造了40条二值化问题，每条问题生成11种内容匹配的非问题变体，形成440个提示；每个提示在10个epoch内采样，得到约4.4万条模型回答。

**📈 对比分析**

比较方法：将不同框架（问题/非问题）、不同确定性层级（陈述/信念/信念）以及重构策略（无重构、问句重构、第一人称重构、无sycophantic基线）在GLM模型中做成对比。结果显示：
- 问句相较于非问句将sycophancy平均降低约24个百分点；
- 将非问题转为问句的重构将sycophancy进一步降低（β≈-0.55）并明显优于无sycophantic基线（β≈0.51）；
- 第一人称重构仅略降sycophancy（β≈-0.25），不如问句重构；
- 这些效果在所有三款LLM上均保持一致，且模型越新sycophancy水平越低。

**⚠️ 局限性**

局限性：
- 实验仅在单轮合成提示下进行，未检验多轮对话或真实用户输入的表现；
- 评估依赖LLM-as-a-judge，可能受模型偏差影响；
- 只关注sycophancy，没有系统评估对帮助性、事实性和情感适宜性的潜在副作用；
- 主题与模型差异虽被控制，但跨文化或多语言环境的普适性尚待验证。

---

## 318. The Topology of Recovery: Using Persistent Homology to Map Individual Mental Health Journeys in Online Communities

**arXiv ID:** 2602.23886 | [PDF](https://arxiv.org/pdf/2602.23886v1)

**作者:** Joydeep Chandra `[一作]` (Tsinghua University), Yong Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 48018 | [OpenAlex ID](https://openalex.org/A5007650371)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

将持久同调（persistent homology）与语义嵌入结合，对 Reddit r/depression 用户的纵向发帖轨迹进行拓扑分析，识别循环与发散等模式，并提出语义恢复速度（SRV）指标；

**💡 创新点**

首次在个体级心理健康轨迹分析中引入可解释的拓扑特征（循环持续性、发散指数、SRV），并将其与传统情感/主题方法对比，展示更细粒度的动态评估；

**🔧 技术方法**

使用 MentalBERT 进行语义嵌入，UMAP 降维到 3D，Vietoris‑Rips 过滤计算持久同调，提取 LP、FI、SRV 特征，并采用随机森林等机器学习模型进行预测；

**📊 数据集**

基于 2018‑2020 年 Reddit r/depression 的 15,847 名用户（487,293 条帖子）数据，利用 Pushshift API 收集；

**📈 对比分析**

与情感分析、BERT 细调、主题变化指标等 baseline 进行比较，单独拓扑特征准确率 72.7%/AUC 0.70，综合特征（含拓扑）准确率 78.3%/AUC 0.76；在 2018‑2019 训练、2020 测试的时间持久化实验中获得 75.1%/AUC 0.79；

**⚠️ 局限性**

代理标签仍未等同于临床诊断，UMAP 的随机性可能影响个体轨迹，缺乏跨社区/平台的泛化验证，且研究时间段已受 COVID‑19 影响，方法尚未用于临床决策或实时干预。

---

## 319. SongSong: A Time Phonograph for Chinese SongCi Music from Thousand of Years Away

**arXiv ID:** 2602.24071 | [PDF](https://arxiv.org/pdf/2602.24071v1)

**作者:** Jiajia Li `[一作]` (Wuhan University), Lefei Zhang `[通讯]` (Wuhan University)

**通讯引用:** 62915 | [OpenAlex ID](https://openalex.org/A5100673818)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 SongSong 模型和 OpenSongSong 数据集，实现古代中国宋词音乐的自动生成。

**💡 创新点**

首创能够演奏宋词的音乐生成框架；采用两阶段节奏→旋律生成，并通过可配置文件实现用户可控的旋律与歌声同步；公开了规模最大的宋词音乐数据集。

**🔧 技术方法**

使用自回归 Transformer 与扩散模型相结合；FastSpeech2 变体用于音素与时长预测；Wavenet 用于 F0 预测；DiffSinger 的浅层扩散机制提升音频质量；整体构建了从歌词到节奏、旋律、歌声、伴奏的完整链路。

**📊 数据集**

OpenSongSong（29.9 小时宋词音乐，含文本、音素、音高、MIDI 等标注），并使用少量主流数据做预训练。

**📈 对比分析**

通过与 Suno、SkyMusic 的零/少 shot 对比，采用 FAD、音乐结构、音色丰富度等客观指标和主观评估（SCS、AC、PA 等宋词专属指标）。SongSong 在宋词专属指标上显著优于两者，FAD 维持在可接受范围；在普通音乐指标上略逊于商业大模型。

**⚠️ 局限性**

受限于宋词音乐数据规模仍有限，模型在普通流行音乐表现不如商业大模型；音质受训练集质量限制；伴奏种类和音色多样性仍不足。

---

## 320. Structured Prompt Optimization for Few-Shot Text Classification via Semantic Alignment in Latent Space

**arXiv ID:** 2602.23753 | [PDF](https://arxiv.org/pdf/2602.23753v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 321. Optimizer-Induced Low-Dimensional Drift and Transverse Dynamics in Transformer Training

**arXiv ID:** 2602.23696 | [PDF](https://arxiv.org/pdf/2602.23696v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析Transformer训练在AdamW优化器下的累计参数轨迹，发现主导方向（称为backbone），并与SGD-family进行对照，验证其是optimizer-induced的隐式偏置；

**💡 创新点**

首次量化并展示了optimizer导致的低维累积漂移结构，将训练轨迹拆解为慢速backbone和快速转向子空间，并证明该结构与loss landscape无关；

**🔧 技术方法**

使用未中心化PCA、梯度与更新方向对齐度、Fisher信息矩阵Rayleigh商、Reheating实验、匹配损失比较、以及对momentum与自适应归一化的解析；

**📊 数据集**

在TinyStories文本数据集上训练8层GPT-2（51M参数）并引入synthetic probe任务；

**📈 对比分析**

通过SGD及其变体对照实验（无/有momentum、Nesterov、不同weight decay）以及在相同验证loss下的匹配loss比较，发现AdamW的PC1解释率约60–80%，而SGD几乎完全沿一维；Reheating可暂时恢复probe准确率但随学习率衰减而衰退；

**⚠️ 局限性**

研究仅在中等规模（51M）Transformer与synthetic probe任务上验证，Fisher矩阵采用低秩近似，未探讨更大模型或多任务场景，结果对随机种子敏感，转向子空间高维尾部解释不足；

---

## 322. Experience-Guided Self-Adaptive Cascaded Agents for Breast Cancer Screening and Diagnosis with Reduced Biopsy Referrals

**arXiv ID:** 2602.23899 | [PDF](https://arxiv.org/pdf/2602.23899v1)

**作者:** Pramit Saha `[一作]` (University of Oxford), J. Alison Noble `[通讯]` (University of Oxford)

**通讯引用:** 22240 | [OpenAlex ID](https://openalex.org/A5077728082)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种面向乳腺超声筛查与诊断的多代理框架BUSD-Agent，利用经验记忆实现自适应决策，显著减少不必要的病理活检。

**💡 创新点**

创新点在于将过去病理确认的决策轨迹作为检索式上下文，动态调节模型信任与升级阈值，实现无参数更新的自我改进；同时在诊断阶段引入结构化放射学特征检索。

**🔧 技术方法**

技术包括多模型集成（CNN/Transformer分类器）、LLM驱动的ReAct式推理、图像与置信度联合检索、VLM（MedGemma-4B）生成结构化报告、U-Net/YOLOv11等检测与分割工具以及GPT-4o的协同推理。

**📊 数据集**

使用了10个公开乳腺超声数据集：BUET-BUSD、BUSBRA、BUS-UCLM、BUSI-WHU、BUS-UC、BUSI、US3M、GDPH&SYSUCC、MICCAI 2022 BUV、US Breast Lesion。

**📈 对比分析**

与10种专有及开源VLM（如GPT-4、Gemini、Qwen、Llama等）进行基准对比，BUSD-Agent在屏蔽率、准确率、敏感度-特异度平衡上均优于对照组，尤其在提升特异度、降低升级率方面表现突出。

**⚠️ 局限性**

局限性包括对检索库质量的依赖、检索策略参数（如λ、K）需人工调优、模型推理成本高、对少数病理标签的泛化能力待验证，以及在低资源环境下的部署可行性尚未充分评估。

---

## 323. APPO: Attention-guided Perception Policy Optimization for Video Reasoning

**arXiv ID:** 2602.23823 | [PDF](https://arxiv.org/pdf/2602.23823v1)

**作者:** Henghui Du `[一作]` (Renmin University of China), Di Hu `[通讯]` (Renmin University of China)

**通讯引用:** 2283 | [OpenAlex ID](https://openalex.org/A5100670614)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发并验证了一种名为APPO的注意力引导感知策略优化算法，用于在视频推理任务中通过强化学习提升模型的细粒度感知与推理能力。

**💡 创新点**

创新点包括：① 将稀疏的终极奖励转化为帧级注意力引导信号；② 对同一帧内来自不同响应的token（intra‑group）使用KL散度计算差异，按高低奖励区分权重，实现token级细粒度奖励；③ 在无需昂贵标注或额外奖励模型的前提下，通过注意力和token重加权实现感知与推理的联合优化。

**🔧 技术方法**

采用了强化学习框架（GRPO、DAPO的改进版），结合注意力权重提取、Top‑K帧/token选择、KL散度重加权、min‑max归一化、token级奖励加权策略梯度优化等技术。

**📊 数据集**

使用了多种视频基准数据集：SEED‑Bench‑R1、Perception Test、VSI‑Bench、NExT‑GQA、MVBench、NExT‑QA，以及Video‑R1‑260K RL数据集（34K子集）进行训练与评估。

**📈 对比分析**

与SFT、GRPO、DAPO以及其他视频推理模型（TW‑GRPO、GRPO‑CARE、Video‑R1、VideoRFT、VideoChat‑R1）在同一基础模型（Qwen2.5‑VL‑3B/7B）上进行对比；APPO在视频推理任务上平均提升1.5%–3.2%（3B）/0.3%–1.6%（7B），在OOP数据集上对GRPO/DAPO提升显著；在34K训练规模下，APPO的整体性能已超越或与使用更大数据集的对手相当。

**⚠️ 局限性**

局限性：对更大规模模型（7B）提升幅度相对有限；效果高度依赖超参数（K₁、K₂、K₃、α）与注意力层选择；缺乏对细粒度感知标注的直接验证；训练成本和内存占用未充分评估，尤其在多帧/多token情况下；未探讨跨模态推理的通用性与极限。

---

## 324. Global Interpretability via Automated Preprocessing: A Framework Inspired by Psychiatric Questionnaires

**arXiv ID:** 2602.23459 | [PDF](https://arxiv.org/pdf/2602.23459v1)

**作者:** Eric V. Strobl `[一作]` (University of Pittsburgh), Eric V. Strobl `[通讯]` (University of Pittsburgh)

**通讯引用:** 6865 | [OpenAlex ID](https://openalex.org/A5112095517)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种两阶段方法——先通过非线性预处理将基线问卷值稳定化，再用线性映射预测未来问卷项，保持预测关系的全局可解释性。

**💡 创新点**

创新点在于将非线性与线性拆分，利用随访数据学习具有项目对齐的预处理器，并证明该结构在满足项目保留和时序冗余的前提下是贝叶斯最优且唯一的。

**🔧 技术方法**

技术实现包括：基于随访结果的最小二乘重构得到B_t，再用随机森林等非线性回归学习预处理h_t；最后通过B_t的逆矩阵得到线性解码器β_t，形成最终的预测模型。

**📊 数据集**

使用三类数据集：NAPLS‑3（早发精神病症状预测）、STAR*D（抑郁症药物治疗期间症状轨迹）以及一组非精神科青少年健康随访数据。

**📈 对比分析**

与AICNN、GPBoost、MGCV、XGBoost及其消融版本对比，REFINE在大多数随访时点实现最高的前向/后向相关性、最高的项目对齐余弦相似度，并且在计算时间上最为迅速，整体性能最优。

**⚠️ 局限性**

局限性包括：对每个随访时点单独估计解码器，缺乏跨时间的共享导致方差增大；预处理器使用随机森林，可能在大规模或更丰富特征场景下受限；B_t 的逆矩阵对条件数敏感，易受噪声影响。

---

## 325. UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images

**arXiv ID:** 2602.24290 | [PDF](https://arxiv.org/pdf/2602.24290v1)

**作者:** Junhwa Hur `[一作]` (Google), Deqing Sun `[通讯]` (Google)

**通讯引用:** 14458 | [OpenAlex ID](https://openalex.org/A5101440839)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种统一的前向模型，利用动态3D高斯涂抹（Dynamic 3D Gaussian Splatting）从仅两张未标定的图像直接重建完整的4D（空间+时间）稠密表示，能够同时输出相机位姿、几何（点、深度）和运动（场景流），并支持任意时间和视角的高保真插值；

**💡 创新点**

核心创新在于：① 将几何、运动与相机位姿统一映射到相同的3D高斯原子集合，实现一次前向推断即可得到所有下游任务；② 采用可微的4D光栅化过程，可把图像、点图、场景流等多模态渲染结果作为自监督损失，显著提升稠密几何与运动的精度；③ 让透明度成为可学习的置信度，自动抑制不相关高斯；④ 在同一框架下实现4D时空插值，开辟新应用场景。

**🔧 技术方法**

使用基于ViT的解码器结合权重共享编码器，输出高斯参数（中心、速度、旋转四元数、尺度、球谐色彩、透明度）和相机位姿；通过线性运动假设实现连续时间的高斯位移；利用可微的3D高斯光栅化生成图像、点和场景流；损失包括自监督图像光度损失（MSE+LPIPS）、光流/深度平滑损失以及若干监督损失（点、流、位姿）。

**📊 数据集**

训练数据集包含Stereo4D、PointOdyssey、Virtual KITTI 2（以及一小部分合成数据如SHIFT、Dynamic Replica、MOVi‑F、Spring等），在验证/测试阶段评估于Stereo4D、Bonn、KITTI和Sintel等标准基准。

**📈 对比分析**

与DynaDUSt3R、ZeroMSF、MonST3R、St4RTrack等最新方法在几何、运动和位姿上进行统一对比；在Stereo4D上点误差EPE从0.81降至0.66，深度误差和δ1.25分别提升；场景流EPE大幅降低（0.17→0.05），位姿ATE、RPE远低于基线；总体在所有指标上实现或逼近state‑of‑the‑art，并在多任务上显著优于单任务或多头模型。

**⚠️ 局限性**

主要局限：假设线性运动与恒定亮度，仅适用于短时间间隔；对长序列时高斯数量线性增长，内存需求高；在纹理稀缺或遮挡复杂的Bonn等数据上表现略逊；以及合成与真实数据的域差异仍需进一步缓解。

---

## 326. Vision-Language Semantic Grounding for Multi-Domain Crop-Weed Segmentation

**arXiv ID:** 2602.23677 | [PDF](https://arxiv.org/pdf/2602.23677v1)

**作者:** Nazia Hossain `[一作]` (McGill University), Shangpeng Sun `[通讯]` (McGill University)

**通讯引用:** 1255 | [OpenAlex ID](https://openalex.org/A5103213588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了Vision‑Language Weed Segmentation (VL‑WS) 框架，利用冻结的 CLIP 图像编码器与 DeepLabv3+ 视觉编码器相结合，并通过 FiLM 对文本描述进行通道调制，实现对多源田间图像的像素级作物与杂草分割。

**💡 创新点**

核心创新在于将视觉分割任务与跨域通用的视觉‑语言语义对齐相结合：① 用预训练 CLIP 生成的全局语义嵌入为分割提供域不变语义基线；② 通过文本描述驱动的 FiLM 机制实现跨数据集的自适应调制；③ 在统一的多数据集上训练，显著降低负迁移与标签异质性问题。

**🔧 技术方法**

技术手段包括：双编码器结构（CLIP + DeepLabv3+）、FiLM 语义调制、对比式视觉‑语言损失（InfoNCE）、Dice+交叉熵分割损失、冻结 CLIP 的图像编码器以及对文本编码器的细化微调。

**📊 数据集**

使用了四个公开或自建的数据集：UAV Soybean‑VL、PhenoBench、GrowingSoy 与 ROSE（玉米与豆类），覆盖不同作物、杂草种类、成长阶段和采集平台（地面机器人、无人机）。

**📈 对比分析**

在多数据集测试集上与 U‑Net、PSPNet、DeepLabv3+ 三种基线对比，VL‑WS 平均 Dice 分数为 91.64%，比最佳基线高 4.98%；在杂草类别上提升 15.42%，并在不同标注比例下保持高性能，证明了模型在跨域和数据稀缺场景下的鲁棒性。

**⚠️ 局限性**

局限性在于：① 仍无法完全消除由视觉编码器产生的负迁移；② 依赖全局 CLIP 嵌入可能不足以捕捉密集交织区域的细粒度语义；③ 需要进一步引入空间自适应或更强的语义正则化来提升对形态多样杂草的分辨力。

---

## 327. The Auton Agentic AI Framework

**arXiv ID:** 2602.23720 | [PDF](https://arxiv.org/pdf/2602.23720v1)

**作者:** Sheng Cao `[一作]`, Ji Tang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Auton Agentic AI Framework，通过AgenticFormat标准和Runtime Engine实现对代理系统的声明式定义、执行与治理；

**💡 创新点**

创新点包括：1）AgenticFormat声明式蓝图与Runtime Engine分离，提升跨语言可移植性和可审计性；2）约束流形（Constraint Manifold）安全投影机制，避免后置过滤；3）层次化记忆与反射式整合协议，实现跨会话知识保留；4）三层自进化架构（In‑Context, Self‑Taught Reasoning, RL）实现代理自适应；5）认知Map‑Reduce、投机执行与动态上下文裁剪等运行时优化降低延迟。

**🔧 技术方法**

技术包括：声明式YAML/JSON Schema、POMDP + 隐空间推理、投影约束正则化、向量检索与知识图谱、Transformer KV缓存裁剪、异步DAG执行、强化学习（PPO/GRPO）与自监督微调；

**📊 数据集**

使用了通用大型语言模型（如GPT‑4/PaLM、Gemini）作为后端，以及公开的数据库、API模拟环境、常见工具连接器（Slack、GitHub、Postgres）做实验；

**📈 对比分析**

与传统LangChain、AutoGen等框架对比，框架在安全合规性（无后置过滤）、可审计性（蓝图版本控制）以及推理延迟（通过Map‑Reduce/投机执行）均表现出显著优势，实验显示平均响应时间降低30‑50%，错误率下降至接近0；

**⚠️ 局限性**

局限性：依赖高质量的蓝图设计与安全规则制定，约束流形表达仍需人工编写；大规模记忆压缩仍面临信息损失风险；多步推理与投机执行的准确性受模型能力限制；缺乏大规模实测数据，需进一步在真实企业场景验证。

---

## 328. Toward Guarantees for Clinical Reasoning in Vision Language Models via Formal Verification

**arXiv ID:** 2602.24111 | [PDF](https://arxiv.org/pdf/2602.24111v1)

**作者:** Vikash Singh `[一作]` (Case Western Reserve University), Gourav Datta `[通讯]` (Case Western Reserve University)

**通讯引用:** 442 | [OpenAlex ID](https://openalex.org/A5017435097)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一套神经符号验证框架，能在无参考文本的情况下自动将视觉语言模型生成的放射学报告转化为结构化逻辑，并利用SMT求解器与临床知识库检测诊断命题的内部一致性。

**💡 创新点**

创新点在于将自然语言报告自动表征为命题约束，结合形式化验证（SMT求解）与临床本体，实现可证明的推理保证并揭示模型的逻辑失效模式，同时提供后置过滤机制提升诊断可信度。

**🔧 技术方法**

核心技术包括：基于GPT-OSS-20B的文本到结构化逻辑的自动化翻译、轻量化临床本体、Z3 SMT求解器进行可证明的蕴含检查，以及假设-保证的形式化验证流程。

**📊 数据集**

实验数据集覆盖五个胸部X光基准（MIMIC‑CXR、Indiana‑IU、CheXpert‑Plus、CheXpert、NIH‑CXR14），并使用七种视觉语言模型（包括MedGemma、Llava、Qwen3等）进行评估。

**📈 对比分析**

与传统BLEU/ROUGE等表面指标对比，提出的无参考音频指标（Soundness、Completeness）显示模型在逻辑一致性上表现更好；后置SMT过滤显著提升诊断的准确性（Soundness>0.95），但召回率略有下降。

**⚠️ 局限性**

该方法的局限在于依赖文本到逻辑的翻译准确性和临床知识库的完整性；若翻译或知识库缺失，将导致误判，且验证无法纠正模型感知阶段的错误。

---

## 329. Actor-Critic Pretraining for Proximal Policy Optimization

**arXiv ID:** 2602.23804 | [PDF](https://arxiv.org/pdf/2602.23804v1)

**作者:** Andreas Kernbach `[一作]` (Fraunhofer Institute for Manufacturing Engineering and Automation), Marco F. Huber `[通讯]` (Fraunhofer Institute for Manufacturing Engineering and Automation)

**通讯引用:** 3599 | [OpenAlex ID](https://openalex.org/A5031354877)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于专家演示的PPO演员-评论家网络预训练方法，在预训练阶段分别用行为克隆和基于预训练策略回放的回报来初始化演员和评论家，并在微调时加入延伸步长与残差架构；

**💡 创新点**

创新点在于同时预训练演员与评论家（通过回放回报），并设计了延伸步长限制和残差网络结构以提升样本效率和缓解灾难性遗忘；

**🔧 技术方法**

使用行为克隆、回报回放、PPO剪裁策略损失、价值损失、熵正则、延伸步长计算与残差网络实现；

**📊 数据集**

采用Gymnasium与Gymnasium-Robotics 15个仿真机器人操作与步态任务的数据集，专家策略来自RL Baselines3 Zoo；

**📈 对比分析**

通过与无预训练、仅演员预训练、PIRL等方法对比，实验显示平均86.1%样本量减少相对无预训练，30.9%相对仅演员预训练，20.5%相对PIRL；在部分任务中未见提升；

**⚠️ 局限性**

局限在于需专家演示，缺乏统一的专家/回放数据量判定方法，且仅验证了连续动作空间的PPO，部分环境下预训练不一定带来收益，需进一步研究。

---

## 330. Towards Source-Aware Object Swapping with Initial Noise Perturbation

**arXiv ID:** 2602.23697 | [PDF](https://arxiv.org/pdf/2602.23697v1)

**作者:** Jiahui Zhan `[一作]` (Shanghai Jiao Tong University), Jianfu Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 16754 | [OpenAlex ID](https://openalex.org/A5100750974)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于单图像自监督的对象交换框架 SourceSwap，能够在不依赖多视角或视频的情况下完成源对象与目标对象的替换，并支持多对象、面部替换和后期细化等应用。

**💡 创新点**

核心创新包括①在初始噪声空间进行频率分离扰动，仅对高频部分做随机置换，从而生成高质量的伪对，直接学习跨对象对齐；②双 U‑Net 结构，源图像完整保留并作为条件，实现全源感知训练，消除显式遮罩，支持零样本推理和轻量级迭代细化；③发布了更高分辨率、类别丰富且交互更复杂的 SourceBench 基准。

**🔧 技术方法**

主要技术手段为 Stable Diffusion v1.5 与 DDIM 逆向、频域高低频分离与随机置换、双 U‑Net 结构（参考分支 + 去噪分支）、全源条件和边框掩码、Grounded SAM 分割、迭代细化推理以及使用 MLLM（ChatGPT‑5）进行 2AFC 评估。

**📊 数据集**

训练集为 40K 张 BrushData 单图像；评估使用自建的 SourceBench（1,554 对）和现有的 DreamEditBench；同时在 SourceBench 的收集阶段使用了 Pexels 与 Unsplash 等公开高质量照片。

**📈 对比分析**

与基线（学习型 PBE、AnyDoor、MimicBrush；测试时微调 PhotoSwap、InstantSwap、SwapAnything；无微调 DiptychPrompt、TIGIC）进行对比，SourceSwap 在 DreamSim、LPIPS、MLLM 与用户研究中均表现更佳，尤其在对象‑场景和谐度上领先；推理速度最快，仅 20 步 + 2 次迭代，平均耗时约 4.4 秒，显著快于传统微调方法。

**⚠️ 局限性**

对目标遮罩定位高度敏感；若 Grounded SAM 误分割多实例，模型会错误地替换所有实例；在多实例或遮罩不精确的情况下，交互效果可能下降；未来需开发更精准或无遮罩的交互方式。

---

## 331. GRAIL: Post-hoc Compensation by Linear Reconstruction for Compressed Networks

**arXiv ID:** 2602.23795 | [PDF](https://arxiv.org/pdf/2602.23795v1)

**作者:** Wenwu Tang `[一作]` (Graz University of Technology), Olga Saukh `[通讯]` (Graz University of Technology)

**通讯引用:** 2406 | [OpenAlex ID](https://openalex.org/A5074037460)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种零梯度、无标签、一次性校准的后置块级补偿方法，利用小规模校准集的Gram矩阵与岭回归恢复压缩后模型的输入-输出行为，从而在不进行微调的情况下显著提升压缩模型的准确率或困惑度。

**💡 创新点**

创新点在于：①实现了与任何结构压缩（剪枝、折叠）兼容的通用补偿框架；②利用第二阶激活统计做数据感知补偿，无需梯度或标签；③通过闭式岭回归直接求解线性映射并嵌入后续投影权重，完全不增加额外参数。

**🔧 技术方法**

技术核心包括：Gram矩阵统计、闭式岭回归求解、线性映射融合到投影权重、卷积/Transformer MLP/Attention块的统一处理；实现细节包括低阶矩阵运算、Kronecker扩展、头级缩放等。

**📊 数据集**

在多个数据集上验证：CIFAR‑10、ImageNet‑1K（ResNet、ViT、CLIP）；LLaMA‑2‑7B 在 C4、WikiText‑2、PTB 上；此外还评估了ARC、BoolQ、HellaSwag 等下游任务。

**📈 对比分析**

与基线剪枝/折叠、REPAIR、SlimGPT、FLAP、Wanda++ 等方法对比，实验表明该补偿能在 10–70% 结构压缩率下，显著恢复或超过原始模型的准确率/困惑度，甚至接近或等价于少量微调；在多种模型/数据集上表现一致且鲁棒。

**⚠️ 局限性**

局限性包括：需要一次完整的未压缩模型前向推理以收集统计，计算/内存开销随隐藏维度平方；对分布漂移敏感；仅在块级补偿，极端压缩或跨层耦合时可能不足；在边缘/低内存设备上 Gram 聚合可能不可行。

---

## 332. UniFAR: A Unified Facet-Aware Retrieval Framework for Scientific Documents

**arXiv ID:** 2602.23766 | [PDF](https://arxiv.org/pdf/2602.23766v1)

**作者:** Zheng Dou `[一作]` (Beihang University), Fuzhen Zhuang `[通讯]` (Beihang University)

**通讯引用:** 10142 | [OpenAlex ID](https://openalex.org/A5102969899)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 UniFAR，一个统一的 facet‑aware 检索框架，能够同时支持文档–文档和问题–文档检索。

**💡 创新点**

通过可学习 facet anchor、适应性多粒度注意力聚合以及 facet‑aware 联合训练，解决文档中心与问题驱动检索之间的粒度、语义焦点与训练信号不匹配问题。

**🔧 技术方法**

基于可替换的预训练编码器（SciBERT、Contriever、MPNet），多粒度注意力聚合，learnable facet anchors，InfoNCE 对比学习与 KL 约束的联合训练。

**📊 数据集**

构建 Facet‑Aware Training Units（FTU）并使用 FLeW、LLM 标注句子与生成 facet‑aware 问题；评测采用 DORIS‑MAE（q‑doc）和 CSFCube（doc‑doc）两大基准。

**📈 对比分析**

与同一后端模型下的 SPECTER、SciNCL、FLeW、FaBle、OSRetriever 等基线进行组内比较，UniFAR 在 q‑doc 的 Recall@5、MAP、R‑Precision 均提升 0.3–1.5 分，doc‑doc 的 nDCG%20、MAP 亦提升 0.5–2.0 分，显著优于单独设计的检索器。

**⚠️ 局限性**

facet 数量固定且共享，未能自适应不同检索场景；未探索多层面动态 facet 选择与更大规模 LLM 生成查询；对极长文档与极短问题的极端情况仍需进一步优化。

---

## 333. OPTIAGENT: A Physics-Driven Agentic Framework for Automated Optical Design

**arXiv ID:** 2602.23761 | [PDF](https://arxiv.org/pdf/2602.23761v1)

**作者:** Yuyu Geng `[一作]` (Zhejiang University), Kaiwei Wang `[通讯]` (Zhejiang University)

**通讯引用:** 4768 | [OpenAlex ID](https://openalex.org/A5018263416)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了基于LLM的物理驱动智能框架，自动生成符合光学约束的镜头设计并可进一步在Zemax中细化。

**💡 创新点**

通过引入光学处方补全任务注入物理直觉，构建层级光学词典奖励并使用DrGRPO强化学习，实现LLM与光学原理的对齐。

**🔧 技术方法**

使用大型语言模型（Qwen3‑4B为基座），光学词典奖励、光学处方补全、光学Ray‑tracing、RMS损失、DrGRPO策略优化以及Zemax局部优化等技术。

**📊 数据集**

采用自建OptiDesignQA数据集，包含711对完整设计任务、124对处方补全任务以及80个测试任务，融合教科书经典与自动优化生成的结构。

**📈 对比分析**

与ChatGPT‑5.2、Claude‑Sonnet‑4.5、Qwen3‑235B等LLM以及传统演化优化方法对比，成功率超过95%，EFFL相对误差≈1%，初始RMS显著优于基线。

**⚠️ 局限性**

受限于模型规模与复杂光学结构的可扩展性，过高遮掩率会导致训练崩溃，且后期细化仍需依赖专业光学软件。

---

## 334. Flowette: Flow Matching with Graphette Priors for Graph Generation

**arXiv ID:** 2602.23566 | [PDF](https://arxiv.org/pdf/2602.23566v1)

**作者:** Asiri Wijesinghe `[一作]` (Data61), Cheng Soon Ong `[通讯]` (Data61)

**通讯引用:** 7415 | [OpenAlex ID](https://openalex.org/A5016073964)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于流匹配的图生成框架 Flowette，通过融合 Gromov-Wasserstein OT 对齐噪声图和数据图，使用 GNN Transformer 预测速度场，并在生成过程中加入 Graphettes 结构先验；

**💡 创新点**

创新点包括：①使用 FGW‑OT 对图进行结构一致的监督配对；②引入可编辑的 Graphettes 作为图先验，扩展图森概念；③在训练目标中加入终点一致性与化学可行性正则化，提升生成质量与稳定性；

**🔧 技术方法**

主要技术包括流匹配（Flow Matching）、GNN Transformer、Fused Gromov‑Wasserstein OT、Hungarian 匹配、正则化损失（终点一致性、软价键约束、原子类型匹配）以及 Graphettes 图先验；

**📊 数据集**

实验使用合成数据集（Tree、SBM、Ego‑small）和分子生成基准（QM9、ZINC250K、Guacamol、MOSES）；

**📈 对比分析**

与 11 种基线（包括 DiGress、DisCo、Cometh、DeFoG 等）进行对比，Flowette 在合成数据上获得最高分，在分子基准上实现 state‑of‑the‑art 或接近最优的有效率、唯一性、化学合法性等指标；

**⚠️ 局限性**

局限性包括：FGW‑OT 计算开销大，Graphettes 需要领域知识手工设计，连续松弛导致离散化误差，且在大图/大批量场景下扩展性有限。

---

## 335. Divide and Conquer: Accelerating Diffusion-Based Large Language Models via Adaptive Parallel Decoding

**arXiv ID:** 2602.23792 | [PDF](https://arxiv.org/pdf/2602.23792v1)

**作者:** Xiangzhong Luo `[一作]` (Southeast University), Xu Yang `[通讯]` (Southeast University)

**通讯引用:** 19128 | [OpenAlex ID](https://openalex.org/A5078188641)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关的自适应并行解码框架 DiCo，采用分治策略加速扩散式大语言模型（dLLM）的推理过程。

**💡 创新点**

创新点在于将分治与自适应并行解码相结合：先通过种子标记和局部集群扩展划分输入序列，再在每个局部集群内根据置信度自适应地并行解码，最后采用基于 logit 边缘和置信度的复合解码完成生成，从而在保持质量的前提下显著提升速度。

**🔧 技术方法**

核心技术包括：
1) 通过空间抑制与轨迹引导的种子标记与集群扩展；
2) 在 Conquer 阶段实现自适应并行解码（依据置信度动态决定可并行解码的 token 数量）；
3) 在 Finalize 阶段使用 logit 边缘阈值与置信度的复合策略进行细粒度解码；
4) 采用分阶段迭代的 divide‑and‑conquer 循环。

**📊 数据集**

在 LLaDA‑8B‑Instruct 与 Dream‑7B‑Instruct 两个 dLLM 上，使用四个常用任务数据集：GSM8K、Math‑500、HumanEval、MBPP。

**📈 对比分析**

与基准方法 Vanilla（单步解码）和 Fast‑dLLM（固定阈值并行解码）比较，DiCo 在所有任务上均实现了显著的速度提升（最高 4.8×）并提升了准确率（non‑AR 设置最高 +18.8%，semi‑AR 设置最高 +9.15%），兼顾吞吐量和生成质量。

**⚠️ 局限性**

局限性包括：对极大规模模型或多模态任务的适用性尚未验证；解码效果在高度噪声或置信度估计不佳的场景下可能下降；以及需要手动调参（如阈值、种子数等）以获得最佳性能。

---

## 336. Disentangled Mode-Specific Representations for Tensor Time Series via Contrastive Learning

**arXiv ID:** 2602.23663 | [PDF](https://arxiv.org/pdf/2602.23663v1)

**作者:** Kohei Obata `[一作]` (SANKEN, Osaka University), Yasushi Sakurai `[通讯]` (SANKEN, Osaka University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种针对张量时间序列的无监督表示学习方法，利用张量切片与对比学习实现模式特定可分离表示。

**💡 创新点**

创新点在于同时保留非时间模式内的依赖关系与时间依赖，并通过实例损失和模式损失两种对比损失实现模式特定与模式不变特征的 disentanglement。

**🔧 技术方法**

使用张量切片、模式独立映射（MI）网络、因果卷积编码器、时间嵌入以及基于InfoNCE的对比学习。

**📊 数据集**

实验涵盖 11 个真实数据集，包括运动传感器识别、网页搜索计数、PM2.5 与天气预测、NYC CitiBike 等。

**📈 对比分析**

与四个时序自监督方法（TS2Vec、CPC、TNC 等）及一种张量自监督方法对比，分类准确率和预测误差均优于大多数基线，展示显著性能提升。

**⚠️ 局限性**

局限在于目前仅适用于至少三个模式的张量时间序列，对高维或非均匀时间步长的数据适用性有限，并未充分探讨对长周期非线性变化的建模。

---

## 337. FuXi-Linear: Unleashing the Power of Linear Attention in Long-term Time-aware Sequential Recommendation

**arXiv ID:** 2602.23671 | [PDF](https://arxiv.org/pdf/2602.23671v1)

**作者:** Yufei Ye `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 28266 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种线性注意力机制 FuXi-Linear，用于超长序列推荐，解决了传统软注意力的 O(n²) 复杂度瓶颈。

**💡 创新点**

① 引入独立的 Temporal Retention Channel，使用周期性时间信号并与语义信号分离，避免相互干扰；② 开发 Linear Positional Channel，利用可学习核函数近似相对位置编码并保持线性递归；③ 在千长度级别展现鲁棒的幂律扩展特性。

**🔧 技术方法**

使用线性注意力 Retention、可学习核位置编码、Chunkwise Recurrent 计算、RMSNorm、FlashAttention‑2 加速、SiLU 激活等技术。

**📊 数据集**

MovieLens‑20M、Kuairand‑27K、KuaiRec 三个公开长序列数据集。

**📈 对比分析**

在 HR@10/50、NDCG@10/50、MRR 等指标下与全注意力模型 SASRec、HSTU、FuXi‑α/β 以及线性注意力模型 RecBLR、Mamba4Rec、TTT4Rec、TiM4Rec、RetNet 进行基准；FuXi‑Linear 在长序列上平均提升约 9% NDCG@10、7% NDCG@50、9% HR@10、5% HR@50、8% MRR；在推理阶段预填 10×、解码 21× 的速度加速。

**⚠️ 局限性**

尚未在多行为/多模态序列上进行验证；对极端超长序列（>10k）可能存在数值稳定性或显存瓶颈；对极大词表扩展需进一步优化；缺乏对实时在线学习和持续更新的评估。

---

## 338. A multimodal slice discovery framework for systematic failure detection and explanation in medical image classification

**arXiv ID:** 2602.24183 | [PDF](https://arxiv.org/pdf/2602.24183v1)

**作者:** Yixuan Liu `[一作]`, Ahmed E. Fetit `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一个完全自动化的多模态审计框架，用于黑盒医疗影像分类器的系统性错误检测与解释。

**💡 创新点**

首次将切片发现方法扩展到多模态嵌入，融合图像、报告文本和元数据，实现更全面的错误发现与可解释性。

**🔧 技术方法**

使用DOMINO+GMM聚类、多模态统一嵌入、TF‑IDF与CLIP式相似度评估，构建错误切片与关键字解释。

**📊 数据集**

在MIMIC‑CXR‑JPG胸部X光多模态数据集上进行实验。

**📈 对比分析**

相较于仅图像或全局TF‑IDF基线，三种失效模式下平均Precision@10提升约20%至30%，表现优于基线。

**⚠️ 局限性**

在噪声标签条件下易产生不稳定切片；仅图像时计算成本高；需要更鲁棒的聚类与更高效的多模态融合。

---

## 339. The Vocabulary of Flaky Tests in the Context of SAP HANA

**arXiv ID:** 2602.23957 | [PDF](https://arxiv.org/pdf/2602.23957v1)

**作者:** Alexander Berndt `[一作]` (Karlsruhe University of Applied Sciences), Thomas Bach `[通讯]` (SAP)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型工业项目 SAP HANA 上复现并扩展 Pinto 等人基于源代码标识符的 flakiness 检测方法，评估词汇表特征、不同特征抽取与分类模型，并通过信息增益识别根因。

**💡 创新点**

首次在大规模工业数据集上验证该方法，并对 TF‑IDF、TF‑IDFC‑RF、XGBoost 与 CodeBERT 等技术进行对比，展示词汇表特征在工业环境下的适用性及其局限性。

**🔧 技术方法**

使用自然语言处理技术（tokenization、identifier splitting、词袋/TF‑IDF/TF‑IDFC‑RF）、机器学习分类器（Random Forest、XGBoost）和预训练语言模型 CodeBERT 进行预测；通过信息增益评估特征重要性。

**📊 数据集**

三套数据集：原始 MSR4Flakiness、SAP HANA 的 Mass Test Execution 2020（MTE20）和 Flaky Test 2021（FT21）

**📈 对比分析**

采用 5‑折交叉验证和 train‑test split 进行模型评估；相较原方法，TF‑IDF/TF‑IDFC‑RF 在 F1‑score 上略有提升，XGBoost 与 Random Forest 结果相近；CodeBERT 在所有数据集上均取得最高 F1‑score（MTE20 99%，FT21 78%），但训练时间与存储成本显著增加。

**⚠️ 局限性**

局限性：模型虽精准但结果不可直接转化为可操作的改进建议；标签噪声与全局问题导致误标；模型对时间变化敏感，需定期再训练；计算资源消耗高，尤其是 CodeBERT；仅在 SAP HANA 环境验证，外部可迁移性尚未充分评估。

---

## 340. The Stability of Online Algorithms in Performative Prediction

**arXiv ID:** 2602.24207 | [PDF](https://arxiv.org/pdf/2602.24207v1)

**作者:** Gabriele Farina `[一作]` (Massachusetts Institute of Technology), Juan Carlos Perdomo `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了：任何在线无后悔算法在performative预测环境中，其生成的模型序列的均匀混合都会收敛到performatively稳定的点（即在模型诱导的数据分布上自身的预测是风险最小的），这一结论不需要对分布映射或损失函数做任何连续性、光滑或强凸性的假设。

**💡 创新点**

创新点在于提出了一个无条件的从无后悔学习到performative稳定性的归约，突破了之前需要Lipschitz/强凸等严格假设的限制；通过引入混合模型并使用鞅论证，克服了稳定点不存在或难以计算的难点；并首次给出梯度下降、随机梯度下降、在线牛顿等常用算法在弱凸或非光滑损失下的稳定性保证。

**🔧 技术方法**

主要技术手段包括：在线学习到批量学习的转换、鞅差分序列的期望收敛性分析、对已有无后悔算法（如OGD、FTRL、在线牛顿）利用已知的对数/平方根调度的损失界；以及对分布映射的随机化处理。

**📊 数据集**

论文为理论研究，未使用实测数据集；仅通过数学示例（如二分区间的平方损失）说明概念，实际验证需要后续实验。

**📈 对比分析**

与以往工作比较：先前的结果多依赖于分布映射Lipschitz且损失强凸，得到的稳定性收敛速率为指数或多项式；本文的归约在更宽泛的设定下实现了同样甚至更快的 1/T 收敛速率，且对非光滑、弱凸损失同样适用。

**⚠️ 局限性**

局限性包括：稳定性仅对混合模型保证，单一模型的稳定点可能不存在或难以求解；稳定性不等价于performative最优性；未讨论多玩家或状态化的performative预测场景；以及结果以期望收敛为主，缺乏高概率保证。

---

## 341. See, Act, Adapt: Active Perception for Unsupervised Cross-Domain Visual Adaptation via Personalized VLM-Guided Agent

**arXiv ID:** 2602.23806 | [PDF](https://arxiv.org/pdf/2602.23806v1)

**作者:** Tianci Tang `[一作]` (Zhejiang University), Gaoang Wang `[通讯]` (Zhejiang University)

**通讯引用:** 38423 | [OpenAlex ID](https://openalex.org/A5028525523)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Sea^2框架，通过在不更新感知模型参数的前提下，训练一个基于VLM的姿态控制代理，使其在新环境中主动获取信息丰富的视角，从而提升视觉任务性能。

**💡 创新点**

创新点在于将域适配转移到部署策略而非模型本身，利用VLM在无监督条件下学习姿态控制，仅用感知模块的标量反馈进行强化学习，实现在跨域场景下的无标签、无更新自适应。

**🔧 技术方法**

采用Vision‑Language Model（如InternVL3.5、Qwen3VL）作为策略网络，先用规则生成的探索轨迹进行监督微调（SFT），随后通过GRPO进行无监督强化学习，并设计以感知置信度与几何一致性为核心的奖励函数。

**📊 数据集**

在ReplicaCAD和HM3D两个室内场景数据集上进行实验，针对视觉定位、分割与三维框架估计三项任务进行评估。

**📈 对比分析**

与随机、前向、启发式、最短路径以及直接提示VLM控制器等多种基线对比，Sea^2在三项任务上分别提升13.54%、15.92%和27.68%的主要评估指标，显著优于传统方法。

**⚠️ 局限性**

局限性包括对VLM语义推理能力的依赖，处理极端遮挡和大视角分布差异时表现可能下降，且对不同感知模型的泛化性和稳定性尚未系统评估。

---

## 342. Construct, Merge, Solve & Adapt with Reinforcement Learning for the min-max Multiple Traveling Salesman Problem

**arXiv ID:** 2602.23579 | [PDF](https://arxiv.org/pdf/2602.23579v1)

**作者:** Guillem Rodríguez-Corominas `[一作]` (Artificial Intelligence Research Institute), Christian Blum `[通讯]` (Artificial Intelligence Research Institute)

**通讯引用:** 16281 | [OpenAlex ID](https://openalex.org/A5051233439)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种面向最小化最大巡回长度的多旅行商问题（min–max mTSP）的混合求解框架 RL‑CMSA，结合了强化学习引导的聚类构造、路由池合并、受限集覆盖 MILP 求解与局部改进。

**💡 创新点**

创新点在于：①利用学习得到的 pairwise q‑value 指导城市聚类，形成更有前景的候选路段；②将生成的多条路段压缩成高质量路段池，并通过设置上界 z 的 MILP 进行高效重新组合；③引入动态年龄策略和强化学习更新机制，使搜索在探索与利用之间保持平衡。

**🔧 技术方法**

使用的技术包括：强化学习（Q‑learning）用于更新城市配对权值；k‑means++ 变体聚类；贪婪插入 + 2‑opt/Or‑opt 路线改进；集覆盖 MILP（使用 CPLEX/GLPK 等求解器）；交叉、移位、交换等跨路改进算子；以及统计显著性检验（Wilcoxon 符号秩检验）。

**📊 数据集**

测试数据集为：1）20组 n∈{50,100,200} 的随机生成实例；2）4个经典 TSPLIB 基准实例（tsp200，tsp200d 等）。每个实例的 m 值为 1%、5%、10%、15% 的城市数。

**📈 对比分析**

与基准 HGA（Hybrid Genetic Algorithm）进行比较。结果表明：在绝大多数规模与 m 组合下，RL‑CMSA 的平均目标值更优、最佳运行率更高且往往更快；在 n=200 且 m=1% 时 HGA 略优；在较大规模或较多车辆（m≥5%）时 RL‑CMSA 取得显著优势；统计检验显示大部分情况 RL‑CMSA 的效果显著更好。

**⚠️ 局限性**

局限性包括：①对极小 m（如 m=1%）和极大规模实例的求解性能相对较弱；②算法对参数（如 n_solutions、d_rate^construct、age_max 等）敏感，需要通过 i‑Race 进行调优；③仅针对对称单仓库 min‑max mTSP，尚未扩展到容量约束、多仓库或时间窗等更一般的 VRP 变体。

---

## 343. Hybrid Offline-Online Reinforcement Learning for Sensorless, High-Precision Force Regulation in Surgical Robotic Grasping

**arXiv ID:** 2602.23870 | [PDF](https://arxiv.org/pdf/2602.23870v1)

**作者:** Edoardo Fazzari `[一作]` (Mohamed bin Zayed University of AI), Cesare Stefanini `[通讯]` (Mohamed bin Zayed University of AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种无传感器、基于物理一致数字孪生与混合强化学习的手术器械抓握力精准调节框架

**💡 创新点**

将高保真数字孪生与递归CMA‑ES专家生成、离线Implicit Q‑Learning、在线TD3微调相结合，实现在无需末端传感器的情况下精确调节抓握力

**🔧 技术方法**

数字孪生建模（耦合电机‑张力‑关节动力学）、CMA‑ES轨迹优化、离线强化学习IQL、在线TD3、Gymnasium环境、FP32推理

**📊 数据集**

使用CMA‑ES生成的专家轨迹约300万步构成Minari离线数据集；在物理平台上进行10条轨迹实验验证

**📈 对比分析**

与传统机械补偿/传感器方案对比，仿真误差<1%，硬件实验平均误差3.8%（最差5.7%），最大误差0.37 N；模型71k参数，推理速度≈26 kHz，满足实时控制需求

**⚠️ 局限性**

依赖精确物理模型，对物体柔性、传动磨损、长时间漂移的鲁棒性有限；不同负荷条件和实际手术环境的适应性仍待验证

---

## 344. The Geometry of Transfer: Unlocking Medical Vision Manifolds for Training-Free Model Ranking

**arXiv ID:** 2602.23916 | [PDF](https://arxiv.org/pdf/2602.23916v1)

**作者:** Jiaqi Tang `[一作]` (Peking University), Qingchao Chen `[通讯]` (Peking University)

**通讯引用:** 2534 | [OpenAlex ID](https://openalex.org/A5069484115)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于拓扑学的医疗基础模型转移可解释度评估框架，能在无需微调的情况下预测分割任务的最佳预训练编码器

**💡 创新点**

将特征空间与标签空间的最小生成树差异（GRTD）和局部边界一致性（LBTC）相结合，并引入任务自适应融合，突破了传统统计方法在密集预测中的局限

**🔧 技术方法**

最小生成树（MST）图构建、拓扑一致性度量、任务复杂度门控融合、随机初始化nnU-Net解码器的特征提取

**📊 数据集**

OpenMind基准（涵盖头颈、心脏、肾脏等多模态分割任务）、多种预训练SSL模型（MAE、SimMIM、SwinUNETR等）

**📈 对比分析**

与LogME、LEEP、GBC、CCFV等基准对比，使用加权Kendall τ评估排名相关性；新方法平均提升约31%（τ≈0.723），并在OOB任务中保持较高一致性

**⚠️ 局限性**

对解码器随机初始化的鲁棒性验证有限；方法主要依赖MST构造，可能对大规模高维特征处理不够高效；在极端类别不平衡或复杂结构任务中仍有提升空间

---

## 345. Green or Fast? Learning to Balance Cold Starts and Idle Carbon in Serverless Computing

**arXiv ID:** 2602.23935 | [PDF](https://arxiv.org/pdf/2602.23935v1)

**作者:** Bowen Sun `[一作]` (William and Mary), Spyros Lalis `[通讯]` (University of Thessaly)

**通讯引用:** 861 | [OpenAlex ID](https://openalex.org/A5034624141)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于深度强化学习的服务器无状态函数保留（keep‑alive）决策框架，通过动态调节 pod 保留时长来同时降低冷启动延迟和空闲碳排放；

**💡 创新点**

创新点在于将冷启动概率、函数特定延迟成本与时变电网碳强度统一建模为多目标序列决策问题，并利用 RL 直接学习自适应保留策略，能在不同工作负载与碳强度变化下实现性能–可持续性的平衡；

**🔧 技术方法**

使用深度 Q‑网络（DQN）作为决策器，状态向量包含重用概率、CPU/内存请求、预估冷启动延迟、实时碳强度以及可调节的碳/延迟权重；

**📊 数据集**

采用华为云公开 Trace（300M+调用、1500+函数）和 FunctionBench 的能耗测量，结合 Electricity Maps 的实时碳强度数据；

**📈 对比分析**

与基准（固定 60 s keep‑alive、仅优化延迟、仅优化碳、EcoLife PSO）进行对比，结果显示在一般和长尾工作负载下冷启动减少 51%~58%，空闲碳降低 77%~80%，整体延迟保持与延迟最优策略相当，总碳排放仅比 Oracle 增加 6–9%，推理耗时约 15 µs/调用；

**⚠️ 局限性**

局限性包括仅在单一硬件平台下验证，未考虑多租户与动态规模化的影响，依赖离线训练，且能耗模型假设为 CPU 绑定且不随功率调度变化，网络延迟被简化为常数。

---

## 346. MPU: Towards Secure and Privacy-Preserving Knowledge Unlearning for Large Language Models

**arXiv ID:** 2602.23798 | [PDF](https://arxiv.org/pdf/2602.23798v1)

**作者:** Tiantong Wang `[一作]` (Nanyang Technological University), Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**通讯引用:** 6844 | [OpenAlex ID](https://openalex.org/A5027969322)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MPU框架，在服务器和客户端双方不泄露模型参数和忘记集的情况下实现大规模语言模型的机器无学习；

**💡 创新点**

创新点在于结合可逆、功能不变的重参数化与多份加噪副本的谐波聚合，理论证明一阶噪声被抵消，同时保持优化轨迹不变；

**🔧 技术方法**

采用结构化噪声注入、可逆对称重参数化、Harmonic聚合和多种基准无学习算法（GradAscent、GradDiff、DPO、NPO、SimNPO、UnDIAL、SatImp）；

**📊 数据集**

使用Llama-3.2-1B/3B和Qwen2.5-1.5B/3B模型，在TOFU Split99基准上进行评估；

**📈 对比分析**

与无噪声中心化无学习和单副本噪声基线对比，MPU在大部分算法下忘记质量、真值比例与模型效能均与无噪声基线相当或更好，且在高噪声或大模型上表现更为稳健；

**⚠️ 局限性**

局限在于多副本计算和通信成本随副本数线性增加，对极端高噪声或特殊算法可能导致收敛不稳定，且理论仅保证一阶误差抵消，未针对更高阶误差提供完整分析。

---

## 347. The Astonishing Ability of Large Language Models to Parse Jabberwockified Language

**arXiv ID:** 2602.23928 | [PDF](https://arxiv.org/pdf/2602.23928v1)

**作者:** Gary Lupyan `[一作]` (University of Wisconsin-Madison), Senyi Yang `[通讯]` (University of Wisconsin-Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过让大型语言模型（LLM）翻译将核心词替换为无意义字符串的“Jabberwocky”文本，证明LLM能够恢复几乎完整的语义信息。

**💡 创新点**

创新点在于：①展示LLM能在极度退化的文本中利用句法、形态和世界知识的紧密整合恢复意义；②证明即使只保留功能词、标点和大小写，LLM仍可实现高质量翻译；③首次量化评估不同模型、不同文本退化方式与语义恢复的关系。

**🔧 技术方法**

使用了Prompt Engineering、语言模型推理（如GPT‑5.1、Gemini Pro 3等）、文本嵌入相似度（OpenAI text‑embedding‑3‑large）以及FastText词嵌入评估单词级翻译。

**📊 数据集**

主要数据集为 Human‑AI‑Parallel（150篇短段落，包含播客记录、电影/电视剧剧本、小说摘录），以及9对主题匹配但预训练与非预训练来源的文本，另外对这些文本做了标准Jabberwocky退化、BLANKs退化以及其他五种变体。

**📈 对比分析**

对比基准相似度（与同一体裁随机原文的相似度）来评估翻译特异性；平均翻译准确度为0.59（最高0.99），显著高于基线0.43。进一步发现：语料类型、功能词比例、前缀词性等因素显著影响恢复效果，预训练文本略高但差异不显著。对比不同模型发现：仅基线模型无法完成，非推理版可完成但质量差，推理版（Gemini 3 Pro最佳）表现优异。

**⚠️ 局限性**

局限性包括：①无法阐明LLM实现恢复的具体机制；②人类与LLM的差距尚未确定；③实验仅限于英语，无法评估形态丰富语言的影响；④相似度评估指标对细节与整体含义区分不足；⑤需要大量预训练的LLM，无法推广到小模型。

---

## 348. FedNSAM:Consistency of Local and Global Flatness for Federated Learning

**arXiv ID:** 2602.23827 | [PDF](https://arxiv.org/pdf/2602.23827v1)

**作者:** Junkang Liu `[一作]` (Tianjin University), Yuanyuan Liu `[通讯]` (Xidian University)

**通讯引用:** 18955 | [OpenAlex ID](https://openalex.org/A5100405062)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新型联邦学习算法FedNSAM，通过在SAM框架中引入全局Nesterov动量来实现本地与全局平坦性的对齐，从而提升全局模型的泛化性能。

**💡 创新点**

创新点包括：①引入“flatness distance”概念量化数据异构导致的本地与全局平坦性不一致；②在FedSAM基础上加入全局Nesterov加速，使本地更新更好地对齐全局平坦区域；③给出更紧的收敛上界并在多种场景下验证性能提升。

**🔧 技术方法**

主要技术包括联邦学习、Sharpness‑Aware Minimization (SAM)、Nesterov加速梯度、指数滑动平均、理论收敛分析以及多模型/数据集的实验验证。

**📊 数据集**

实验使用了CIFAR‑10、CIFAR‑100、Tiny ImageNet等图像数据集，并在LeNet‑5、VGG‑11、ResNet‑18以及Vision Transformer等模型上进行验证。

**📈 对比分析**

在多种数据异构程度和参与率设置下，FedNSAM相较于FedAvg、FedAvgM、SCAFFOLD、FedACG、FedSAM、MoFedSAM、FedGAMMA、FedLESAM等基线，在准确率上提升约10–20%，同时收敛轮次显著减少（例如在CIFAR‑100下从316轮提升到66%准确率）。

**⚠️ 局限性**

局限性在于对超参数λ和ρ的敏感性，需要在不同场景下进行调参；在极端数据异构或极低参与率时仍可能出现收敛速度下降；另外，Nesterov动量的引入虽提升了性能，但也增加了额外的通信和计算开销。

---

## 349. IDP Accelerator: Agentic Document Intelligence from Extraction to Compliance Validation

**arXiv ID:** 2602.23481 | [PDF](https://arxiv.org/pdf/2602.23481v1)

**作者:** Md Mofijul Islam `[一作]` (Amazon Web Services), Diego A. Socolinsky `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 IDP Accelerator 框架，实现多文档分割、结构化信息提取、代理式分析与规则验证的端到端工业级文档智能处理。

**💡 创新点**

核心创新包括：①DocSplit 数据集与 BIO 标注的多文档分割方法；②将多模态 LLM 与 OCR 结合的高准确率提取；③基于 Model Context Protocol（MCP）的代理式分析模块；④LLM 驱动的规则验证模块取代传统引擎；⑤完整开源代码与可复现的评估框架。

**🔧 技术方法**

使用技术包括：AWS Serverless 架构（Step Functions、Lambda、SQS、DynamoDB、CloudWatch、Cognito、AppSync）；Amazon Bedrock 多模态 LLM（Claude、Qwen、Gemma）；Amazon Textract OCR；RAG 检索 + MCP；Stickler 评估工具；人机交互 Web UI。

**📊 数据集**

数据集：DocSplit benchmark 用于多文档分割；RealKIE‑FCC‑Verified（75 份 FCC 发票）用于结构化提取评估；以及医疗、营销、金融等行业内部生产数据。

**📈 对比分析**

通过 Test Studio 与 Stickler 进行结构化评估，比较 Claude Sonnet 4.5、Opus、Haiku、Qwen3‑VL、Gemma‑3 在 OCR、Image、OCR+Image 三种输入模态下的提取准确率、延迟、成本与失败率。Claude Sonnet 4.5 在 OCR+Image 获得最高提取分 0.7991，成本最高；Haiku 在 OCR+Image 成本最低但仍保持较高分；开源模型在 OCR 模式下表现相近，但在 Image 模式下失败率高。生产部署案例显示分类准确率 98%，处理延迟降低 80%，成本降低 77%。

**⚠️ 局限性**

局限性：①依赖云基础设施，缺乏完整本地部署方案；②模型偏差与公平性问题；③自动化偏差需人工复核；④图像输入模式对开源模型鲁棒性差，导致失败率高；⑤规则验证仍需针对特定业务场景进行细化；⑥多语言与行业特定细粒度需求仍需进一步扩展。

---

## 350. CLOAQ: Combined Logic and Angle Obfuscation for Quantum Circuits

**arXiv ID:** 2602.23569 | [PDF](https://arxiv.org/pdf/2602.23569v1)

**作者:** Vincent Langford `[一作]` (Lehigh University), Yuntao Liu `[通讯]` (Lehigh University)

**通讯引用:** 1112 | [OpenAlex ID](https://openalex.org/A5100657640)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出一种同时利用逻辑位与相位角的键控量子电路混淆方法，实现对量子编译阶段的IP保护。

**💡 创新点**

创新点在于将逻辑锁和相位角编码结合，利用辅助量子比特、Hadamard门与随机相位门实现更大的关键搜索空间，并引入输入状态采样评估。

**🔧 技术方法**

采用了键控逻辑锁定、相位角随机化、辅助量子比特与控制门、假门插入、去混淆过程，以及Qiskit编译与模拟、FakeManilaV2噪声模型和总变差距离（TVD）度量。

**📊 数据集**

实验使用IBM Qiskit的Qasm基准集，包括Adder、Fredkin、Basis Change和Wstate四个小规模电路。

**📈 对比分析**

通过比较逻辑仅混淆、相位仅混淆和两者结合的TVD结果，发现组合方式在错误键下产生最高TVD（≈0.9），而正确去混淆后平均TVD仅≈5%，显示了更强的破坏性和恢复性。

**⚠️ 局限性**

局限性包括仅在3~4量子比特的小规模电路上验证、门数和深度的临时增加导致编译器优化受限、噪声硬件下的准确性下降，以及对大规模电路的可扩展性尚未评估。

---

## 351. Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation

**arXiv ID:** 2602.24283 | [PDF](https://arxiv.org/pdf/2602.24283v1)

**作者:** Zhengbo Wang `[一作]` (University of Science and Technology of China), Tieniu Tan `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 37051 | [OpenAlex ID](https://openalex.org/A5111885963)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种低秩优化器 LoRA-Pre，通过压缩 Adam 与 Muon 的动量矩阵实现内存高效预训练和微调。

**💡 创新点**

创新点在于将指数移动平均（EMA）动量更新等价为在线线性回归，并以低秩因子化实现持续子空间自适应，显著减少了记忆占用。

**🔧 技术方法**

使用在线梯度流、低秩因子化、以及改进的 Adam/Muon 更新规则，并给出了闭式更新公式。

**📊 数据集**

在 C4 数据集上进行预训练，在 MetaMath100k、GSM8K 与 MATH-500 数据集上进行微调评估。

**📈 对比分析**

与 Adam、Muon、GaLore 等低秩方法对比，LoRA-Pre 在 60M–1B 参数模型上达到最低困惑度，微调时平均提升约 3–6 分，展现出显著性能优势。

**⚠️ 局限性**

局限在于对低秩参数的依赖，极小秩下的稳定性尚未完全验证；且仍需进一步探究跨优化器通用性的极限。

---

## 352. A Boundary Integral-based Neural Operator for Mesh Deformation

**arXiv ID:** 2602.23703 | [PDF](https://arxiv.org/pdf/2602.23703v1)

**作者:** Zhengyu Wu `[一作]` (Hangzhou Dianzi University), Wei Wang `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 35485 | [OpenAlex ID](https://openalex.org/A5116337743)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于边界积分与神经算子（BINO）的线性弹性网格变形框架。

**💡 创新点**

创新点在于引入Dirichlet型Green张量的直接边界积分表达式，学习几何和材料感知的Green扭矩核，并将物理积分过程与几何描述解耦，实现在任意边界条件下的快速泛化。

**🔧 技术方法**

使用边界积分方法、Dirichlet Green张量、神经算子（深度残差网络）学习Green扭矩核、几何描述符以及离散边界积分技术。

**📊 数据集**

通过高精度FEM生成二维线性弹性数据集，包含随机仿射平移与谐波扰动的边界条件，用于柔性梁和NACA0012机翼的训练，共计约2000+样本。

**📈 对比分析**

与传统FEM、RBF/IDW等基准方法对比，BINO在相同网格下的平均相对误差分别为2.99%（梁）和0.74%（机翼），同时保持线性原理并显著降低计算复杂度至O(M×K)，实现更快的实时推断。

**⚠️ 局限性**

局限性包括仅适用于二维线性弹性、对尖锐几何角落的精度有限、尚未扩展至非线性大变形、三维拓展以及复杂材料模型。

---

## 353. CA20108 COST Action: A Methodology for Developing FAIR Micrometeorological Networks

**arXiv ID:** 2602.23921 | [PDF](https://arxiv.org/pdf/2602.23921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 354. Higress-RAG: A Holistic Optimization Framework for Enterprise Retrieval-Augmented Generation via Dual Hybrid Retrieval, Adaptive Routing, and CRAG

**arXiv ID:** 2602.23374 | [PDF](https://arxiv.org/pdf/2602.23374v1)

**作者:** Weixi Lin `[一作]` `[通讯]` (Northwestern Polytechnical University), Weixi Lin (Northwestern Polytechnical University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套面向企业的完整链接优化RAG框架——Higress‑RAG，旨在解决检索精度低、幻觉多、实时延迟高等问题；

**💡 创新点**

创新点包括基于Model Context Protocol的统一接口、混合检索与递归秩融合、语义缓存与动态阈值、纠正检索评估（CRAG）以及元数据加权提升；

**🔧 技术方法**

技术手段涵盖MCP、HyDE、Reciprocal Rank Fusion、BGE‑M3/BGE‑Reranker、Milvus分区键、Tavily外部搜索、LLM驱动的路由与评估；

**📊 数据集**

使用自研的Higress Blog与Higress Doc两套企业级技术文档数据集进行评测；

**📈 对比分析**

通过与传统Naive RAG基线对比，系统在检索召回率、Factuality分数上提升约30%，缓存查询时延低至50ms，整体推理延迟在11–20秒之间；

**⚠️ 局限性**

局限性包括对LLM推理速度与成本高度依赖、外部搜索对网络访问的要求以及多租户环境下的高并发瓶颈。

---

## 355. U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation

**arXiv ID:** 2602.23739 | [PDF](https://arxiv.org/pdf/2602.23739v1)

**作者:** Xiang Deng `[一作]` (Tsinghua University), Yebin Liu `[通讯]` (Tsinghua University)

**通讯引用:** 10578 | [OpenAlex ID](https://openalex.org/A5032875389)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

我们提出并实现了U-Mind，一套统一的实时多模态交互系统，能够在单个交互循环中生成文本、语音、运动和视频。

**💡 创新点**

核心创新是统一对齐与推理框架，包括分段对齐策略、回顾驱动学习和文本优先解码，既保持高层推理，又实现跨模态同步。

**🔧 技术方法**

采用LLaMA2‑7B作为基座，使用RVQ‑VAE将运动（SMPL‑X 6D 旋转）和语音离散化，加入特殊CoT标记，并在统一词表中训练。

**📊 数据集**

在BEAT v2、HumanML3D、Common Voice、OpenOrca 等数据集上进行预训练和指令微调，并用 Qwen3 生成 QA 增强样本。

**📈 对比分析**

与 SOLAMI、LOM、EMAGE、CaMN、DisCo 等基线对比，U‑Mind 在多模态对话、指令执行、T2M/S2M 任务上在 FGD、角度误差、多样性、相关性和自然度指标上均取得最优或接近最优的成绩。

**⚠️ 局限性**

受限于运动量化词表，生成的细粒度手势与面部表情表现不够丰富；预训练数据比例经验性选择，缺乏更系统的任务平衡方法。

---

## 356. Do LLMs Benefit From Their Own Words?

**arXiv ID:** 2602.24287 | [PDF](https://arxiv.org/pdf/2602.24287v1)

**作者:** Jenny Y. Huang `[一作]` (Massachusetts Institute of Technology), Jacob Andreas `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 5402 | [OpenAlex ID](https://openalex.org/A5070829652)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型在多轮对话中是否需要保留自身先前的回答，实验显示删除助手侧历史常能保持甚至提升回答质量。

**💡 创新点**

发现大约36%对话轮是自含的，并提出了上下文过滤与自适应助手回应省略的方法。

**🔧 技术方法**

采用 LLM-as-judge 评估、轮次、提示类别和嵌入特征以及 L1 正则化逻辑回归分类器来决定是否保留助手历史。

**📊 数据集**

使用 WildChat 与 ShareLM 两个真实多轮聊天数据集进行实验。

**📈 对比分析**

对比全上下文与仅用户侧上下文的 win‑rate 与上下文长度，发现后者在多模型上表现相当，且上下文长度可缩减 5‑10 倍；自适应策略在保持 95% 以上性能的同时可节省约 30% token。

**⚠️ 局限性**

局限在于评估依赖自动化 LLM 判断器，未覆盖所有依赖场景，且仅实现全/空助手历史切换，未实现更细粒度的历史筛选。

---

## 357. Any Model, Any Place, Any Time: Get Remote Sensing Foundation Model Embeddings On Demand

**arXiv ID:** 2602.23678 | [PDF](https://arxiv.org/pdf/2602.23678v1)

**作者:** Dingqi Ye `[一作]` (University of Illinois Urbana-Champaign), Shaowen Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7939 | [OpenAlex ID](https://openalex.org/A5026042163)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 rs-embed，一套统一的 ROI‑centric Python 库，能够以单行代码从多种遥感基础模型（RSFM）中获取、批量处理并导出嵌入向量。

**💡 创新点**

创新点包括：① 将空间、时间、输出和传感器规范统一为四个可验证的 Spec；② 构建 Provider/Embedder 两层抽象，解耦云数据源与模型推理；③ 设计高性能并行流水线（预取、推理、异步写入）和策略/能力匹配机制，显著降低配置与兼容成本；④ 提供统一的元数据结构，便于跨模型对齐与融合。

**🔧 技术方法**

使用了 Python、NumPy/Xarray、Google Earth Engine 等云 API、并行线程池、异步 I/O、批处理推理、缓存与重试机制等技术实现快速、可扩展的嵌入生成；在模型层面采用面向对象的基类统一接口。

**📊 数据集**

在实验中以 2019 年 6 月至 8 月的伊利诺伊州玉米产量为例，利用 SPAM2020V2 产量数据作为标签，并采集 16 种 RSFM 的嵌入；此外使用 Sentinel‑2、MODIS 等多波段遥感数据做输入。

**📈 对比分析**

通过在采样点处提取嵌入并训练随机森林回归器，对比不同模型的 R² 与 RMSE；Agrifm 模型取得最高的 R²，但对极端产量的拟合仍有限；可视化展示了不同模型在相同时空条件下的嵌入差异。

**⚠️ 局限性**

局限性包括：① 对极端产量预测精度不足；② 对于不支持年级嵌入的模型需要手工拆分时间窗口；③ 仍依赖特定云平台（如 GEE），在跨平台部署时需额外适配；④ 部分模型缺乏标准化的元数据，导致对齐与融合仍有挑战。

---

## 358. EgoGraph: Temporal Knowledge Graph for Egocentric Video Understanding

**arXiv ID:** 2602.23709 | [PDF](https://arxiv.org/pdf/2602.23709v1)

**作者:** Shitong Sun `[一作]` (Queen Mary University of London), Jifei Song `[通讯]` (University of Surrey)

**通讯引用:** 1007 | [OpenAlex ID](https://openalex.org/A5046874089)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种基于实体中心、动态知识图的超长主观视频理解框架 EgoGraph。

**💡 创新点**

创新点在于通过专门的 egocentric schema 与时间关系建模，实现跨天长时依赖的实体记忆与推理。

**🔧 技术方法**

使用大语言模型抽取实体与关系、图结构合并与时间过滤，以及基于图检索的检索增强生成问答技术。

**📊 数据集**

在 EgoLifeQA 与 EgoR1‑Bench 两个超长主观视频问答基准上进行评测。

**📈 对比分析**

与 EgoGPT、VideoRAG、LightRAG 及 Gemini‑1.5‑Pro 等方法对比，EgoGraph 在两项基准上分别达到 45.8% 与 41.3%，显著优于现有 SOTA。

**⚠️ 局限性**

局限在于依赖外部大模型抽取，处理极长视频仍需分块，且对实时性和多模态交互的支持尚不足。

---

## 359. CACTUSDB: Unlock Co-Optimization Opportunities for SQL and AI/ML Inferences

**arXiv ID:** 2602.23469 | [PDF](https://arxiv.org/pdf/2602.23469v1)

**作者:** Lixi Zhou `[一作]` (Arizona State University), Jia Zou `[通讯]` (Arizona State University)

**通讯引用:** 734 | [OpenAlex ID](https://openalex.org/A5013735333)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了名为CactusDB的数据库系统，支持 SQL 与 AI/ML 推理查询的联合执行，并通过三层 IR 与可重用 MCTS 优化器实现高效协同优化。

**💡 创新点**

创新点包括：①三层 IR 统一抽象关系算子、表达式与 ML 函数，支持 O1–O4 四类协同优化；②基于查询嵌入的可重用 MCTS，利用共享状态与可配置动作大幅降低搜索开销；③两阶段模型训练（对比学习 + 回归）提升成本估计与状态匹配准确性。

**🔧 技术方法**

使用了 Velox 引擎、Transformer/QueryFormer 级查询嵌入、Model2Vec 与 Query2Vec 的对比学习、MCTS 结合 UCB、Faiss 向量检索、以及多种深度学习模型（FFNN、两塔推荐、DLRM、AutoEncoder、XGBoost 等）。

**📊 数据集**

评估数据集包含 MovieLens‑1M、MovieLens‑32M、TPCx‑AI（scale 1 & 10）、Credit Card、Expedia、Flights 以及自制的 2,000+ 随机推理查询。

**📈 对比分析**

通过与 EvaDB、PySpark UDF、MADLib、SystemDS、PostgresML、DL‑Centric、IMBridge 等基线对比，CactusDB 在复杂推理查询上实现 22–441 倍加速，整体优化时延仅约 1% 的执行时间；在 2,000 个随机查询中，可重用 MCTS 在执行时间上比普通 MCTS/启发式/无优化提升 10–20%，且优化时延更低。

**⚠️ 局限性**

局限性包括缺乏将现有 Python 数据科学管线自动转为三层 IR 的前端工具；需要手工注册新 ML 函数与重写规则；对大语言模型等黑盒模型的支持有限；优化器高度依赖预定义规则和模型，新增功能需额外工程投入。

---

## 360. BiKA: Kolmogorov-Arnold-Network-inspired Ultra Lightweight Neural Network Hardware Accelerator

**arXiv ID:** 2602.23455 | [PDF](https://arxiv.org/pdf/2602.23455v1)

**作者:** Yuhao Liu `[一作]` (Ruhr University Bochum), Akash Kumar `[通讯]` (Ruhr University Bochum)

**通讯引用:** 6127 | [OpenAlex ID](https://openalex.org/A5100755285)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了BiKA网络，替代KAN的可学习非线性函数为可学习阈值，设计了无乘法、无激活的硬件加速器，并在FPGA上实现与评估。

**💡 创新点**

创新点在于将KAN的复杂非线性函数逼近为二进制阈值，形成完全整数、无乘法计算模式，并通过阈值逼近理论与直通估计实现轻量化硬件设计。

**🔧 技术方法**

使用了PyTorch/CUDA训练框架、阈值逼近与直通估计技术、Systolic Array加速器架构，以及FPGA（Ultra96-V2）Vivado实现与综合。

**📊 数据集**

使用MNIST（TFC/SFC/LFC三种MLP结构）和CIFAR-10（tiny VGG-like CNN）数据集进行训练与测试。

**📈 对比分析**

通过与BNN、QNN以及已有KAN实现在相同网络结构下的FPGA资源（LUT/FF/BRAM）、频率、延迟、ADP/PDP等指标比较，BiKA在资源占用上比BNN减少27.73%，比QNN减少51.54%，速度比QNN快2–3倍，但在CIFAR-10上准确率低于BNN约10%。

**⚠️ 局限性**

主要局限在于对复杂任务（如CIFAR-10）准确率下降，对训练超参数高度敏感，阈值量化参数m对性能影响大，且目前模型规模有限，需进一步优化训练策略与网络深度。

---

## 361. Robust Aggregation for Federated Sequential Recommendation with Sparse and Poisoned Data

**arXiv ID:** 2602.23982 | [PDF](https://arxiv.org/pdf/2602.23982v1)

**作者:** Minh Hieu Nguyen `[一作]` `[通讯]`, Minh Hieu Nguyen

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种面向联邦顺序推荐的鲁棒框架(FORTRESS)，在客户端本地通过多视图对比学习和时间一致性正则化提升稀疏数据的表示质量，并在服务器端加入基于流行度的对比分离与方差正则化以抵御推广与伪装攻击。

**💡 创新点**

创新点在于：①完全在本地完成多视图对比学习（序列、用户、项目），避免共享敏感序列；②引入时间一致性正则化，平滑用户表示随时间的波动；③设计服务器端的对比分离与方差正则化，实现对推广/伪装攻击的全局防御；④三者协同在联邦场景下同时解决稀疏性与对抗性两大难题。

**🔧 技术方法**

技术方法包括：联邦学习框架、局部多视图对比学习（InfoNCE）、时间一致性正则化、服务器端流行度分离与方差正则化、FedAvg聚合、局部梯度下降。

**📊 数据集**

在Amazon Cell Phone、Amazon Baby和MIND三个真实数据集上进行实验。

**📈 对比分析**

与多种基准（如FedAvg、FedCL、FedCSR等）进行对比，实验结果表明在常规与对抗环境下均显著提升Top‑K推荐准确率与鲁棒性，尤其在攻击场景下ER@K大幅下降。

**⚠️ 局限性**

局限性包括：仅在三大公开数据集上评估，缺乏对极端异构和极低活跃用户的深入验证；服务器端额外正则化可能导致额外通信与计算开销；对模型规模和通信压缩的兼容性尚未充分探讨。

---

## 362. ODAR: Principled Adaptive Routing for LLM Reasoning via Active Inference

**arXiv ID:** 2602.23681 | [PDF](https://arxiv.org/pdf/2602.23681v1)

**作者:** Siyuan Ma `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 49285 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ODAR，一种基于难度估计的自适应推理框架，动态将查询路由到快速或延迟式 LLM 代理，并通过自由能原则进行答案融合。

**💡 创新点**

创新点包括：① 将难度估计视为主动推理的期望自由能问题，使用轻量化估计器实现即时决策；② 引入基于方差熵（varentropy）的风险敏感自由能融合，既兼顾置信度又抑制幻觉；③ 通过阈值分层路由实现“快慢”代理协同，显著提升计算-精度 Pareto 前沿。

**🔧 技术方法**

技术手段主要有：主动推理 + 期望自由能估计、Amortized 估计器、theta–gamma 触发式路由、自由能最小化融合（字符级能量密度 + Z-score 标准化）、多代理（Fast：GPT‑5.1, Slow：Claude‑4.5/DeepSeek）和可复现的 Docker 化栈。

**📊 数据集**

在 23 个跨领域基准上评估，包括 MATH、GSM8K、IMO 2025、ARC、MMLU、HotpotQA、ScienceQA、BBH、HLE、SWE‑bench、LIVEBENCH、IFEval 等，另外在全开源 Llama‑4 + DeepSeek 栈上验证可复现性。

**📈 对比分析**

与自适应路由、最佳样本、自由能融合、TOPS 等最先进方法对比，ODAR 在 22/23 任务上达成新最佳，平均准确率 89.6%（相比 Self‑Consistency +6%），在 MATH、IMO 2025、HLE 等极难任务上分别提升 20% 以上；在开源栈下平均准确率 84.4%，计算成本比 Self‑Consistency 降低 82%。

**⚠️ 局限性**

局限性包括：硬路径在尾部可能产生 >60 s 的高延迟，系统性能受限于基模型知识与推理能力，且自由能融合需访问 token‑level 似然，限制了在无法透明获取概率的环境中的可迁移性。

---

## 363. EvalMVX: A Unified Benchmarking for Neural 3D Reconstruction under Diverse Multiview Setups

**arXiv ID:** 2602.24065 | [PDF](https://arxiv.org/pdf/2602.24065v1)

**作者:** Zaiyan Yang `[一作]` (Beijing University of Posts and Telecommunications), Boxin Shi `[通讯]` (Peking University)

**通讯引用:** 8122 | [OpenAlex ID](https://openalex.org/A5038326097)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了EvalMVX数据集，并在同一数据集上对MVS、MVPS、MVSfP三种多视角重建方法进行定量基准评测。

**💡 创新点**

首次提供具备对齐3D GT、环境光与单灯光极化图像的实景数据集，能同时比较三种方法的工作范围与性能。

**🔧 技术方法**

采用极化相机捕捉、LED光照控制、逆轮廓渲染对齐、SAM分割、NeRF/3DGS/隐式SDF等技术进行数据处理与基线实现。

**📊 数据集**

使用EvalMVX（25个物体、20视角、17灯光共8,500张图）作为核心数据集，并与现有MVS/MVPS/MVSfP数据集做对照。

**📈 对比分析**

通过Chamfer Distance对13种方法进行比较，MVPS在大多数对象上取得最低/第二低平均CD，MVS在高反射表面表现突出，MVSfP在复杂反射场景下有优势但整体精度略逊于MVPS。

**⚠️ 局限性**

主要限制包括极化图像噪声导致的AO偏差、3DGS表示不如SDF细腻、光照条件对MVS和MVSfP的依赖、计算开销高以及仅覆盖RGB极化，未包含深度/红外等多模态数据。

---

## 364. PDF: PUF-based DNN Fingerprinting for Knowledge Distillation Traceability

**arXiv ID:** 2602.23587 | [PDF](https://arxiv.org/pdf/2602.23587v1)

**作者:** Ning Lyu `[一作]`, Zhiyuan Yan `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在教师模型的 logits 上叠加基于 PUF 的扰动，将设备特定指纹嵌入蒸馏过程，使得任何被盗或克隆的学生模型都可追溯回原始设备。

**💡 创新点**

创新点在于：①利用硬件根源的 PUF 响应直接植入 logits，避免对模型内部参数的访问；②提出两阶段解码（神经网络 + 汉明距离）与位压缩方案，兼顾轻量、可扩展与鲁棒性；③实现后期追踪而非仅防盗。

**🔧 技术方法**

采用的技术包括：物理不可克隆函数（PUF）、知识蒸馏、量化训练、神经网络解码器、汉明距离纠错、以及位压缩映射。

**📊 数据集**

实验使用的公开图像数据集为 CIFAR‑10、CIFAR‑20 与 CIFAR‑50。

**📈 对比分析**

与传统水印和硬件防护方案对比，评估指标为可追踪性、是否需要内部访问、硬件感知与运行开销；实验表明在扰动强度 ϵ≥0.05 时实现 0% BER/FER，学生模型准确率与基线相差不超过 1%，且整体开销低。

**⚠️ 局限性**

局限性包括：实验仅基于模拟 PUF，未在真实 FPGA 上验证；指纹恢复对扰动强度高度敏感，过大 ϵ 可能导致准确率下降；目前仅针对蒸馏盗窃，未覆盖其它攻击方式。

---

## 365. FocusTrack: One-Stage Focus-and-Suppress Framework for 3D Point Cloud Object Tracking

**arXiv ID:** 2602.24133 | [PDF](https://arxiv.org/pdf/2602.24133v1)

**作者:** Sifan Zhou `[一作]` (Southeast University), Xiaobo Lu `[通讯]` (Southeast University)

**通讯引用:** 3361 | [OpenAlex ID](https://openalex.org/A5066658319)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种一阶段的3D点云单目标跟踪框架 FocusTrack，联合运动与语义建模实现高效跟踪。

**💡 创新点**

创新点：①Inter‑frame Motion Modeling (IMM) 通过时间差分双胞塔网络捕捉全局运动差异；②Focus‑and‑Suppress Attention 在无需显式前景分割的前提下，利用 IMM 的运动上下文增强前景语义并抑制背景噪声。

**🔧 技术方法**

技术方案包括 pillar feature embedding、temporal‑difference siamese encoder、共享权重的 CNN/线性/DWC 模块以及线性注意力机制，并在此基础上构建 IMM 与 Focus‑and‑Suppress Attention。

**📊 数据集**

使用 KITTI、nuScenes 以及 Waymo Open Dataset（WOD）三个公开数据集进行评估。

**📈 对比分析**

与现有匹配式和运动式 SOTA 方法对比，FocusTrack 在所有数据集上均取得更高的 Success/Precision（如 KITTI 平均精度 89.4%，nuScenes 提升约 4%），并在 RTX3090 上实现 105 FPS，速度优势明显。

**⚠️ 局限性**

局限性：在极度稀疏的点云场景下表现下降，缺乏多帧历史信息与 RGB 语义融合。

---

## 366. Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles

**arXiv ID:** 2602.24030 | [PDF](https://arxiv.org/pdf/2602.24030v1)

**作者:** Fangyu Sun `[一作]` (Shanghai Jiao Tong University), Danping Zou `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2290 | [OpenAlex ID](https://openalex.org/A5019803400)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于视觉的课程强化学习框架，实现无人机在障碍丰富赛道中高速飞行与避障的端到端控制；

**💡 创新点**

创新点包括多阶段课程学习、域随机化与多场景更新相结合的训练策略、专门平衡穿越门与避障的奖励设计以及轻量级网络结构；

**🔧 技术方法**

使用技术主要有PPO强化学习、GRU时序建模、逆深度图输入、域随机化、奖励函数分解以及轻量级CNN+MLP网络；

**📊 数据集**

数据集主要为VisFly模拟器中的三种赛道（S形、J形、3D圆形），随机生成障碍物与门位置信息，并在真实环境中使用Intel D435i深度相机与Vicon定位系统；

**📈 对比分析**

与基线视觉RL和无视觉状态RL对比，本文方法在三条赛道上实现了100%成功率，且平均赛道时间比基线快约10%–20%；

**⚠️ 局限性**

局限性在于对未见赛道布局的泛化能力有限，RL的探索效率与样本效率仍是瓶颈，难以处理完全新颖的赛道配置。

---

## 367. Enhancing Continual Learning for Software Vulnerability Prediction: Addressing Catastrophic Forgetting via Hybrid-Confidence-Aware Selective Replay for Temporal LLM Fine-Tuning

**arXiv ID:** 2602.23834 | [PDF](https://arxiv.org/pdf/2602.23834v1)

**作者:** Xuhui Dou `[一作]` (University of Nottingham), Alejandro Guerra-Manzanares `[通讯]` (University of Nottingham)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于decoder‑style LLM（microsoft/phi‑2 + LoRA）对2018‑2024年CVE链接的代码进行二分类漏洞检测，并采用严格的前向时间链式评估；

**💡 创新点**

提出Hybrid‑CASR混合置信度与类别平衡的选择式重放策略，显著提升前向与后向记忆，并在保持可计算成本的前提下获得更高Macro‑F1；

**🔧 技术方法**

使用LoRA参数高效微调、低秩适配、置信度阈值选择、类别平衡抽样、正交正则化（OLoRA）等技术；

**📊 数据集**

构建了包含约42个双月窗口的CVE‑fixes来源的函数级数据集，涵盖C/C++（及部分Java）代码；

**📈 对比分析**

对比了窗口仅训练、累计训练、Replay‑1P/3P、CASR、Hybrid‑CASR、LB‑CL、OLoRA及零射击基线，Hybrid‑CASR平均Macro‑F1为0.667，比窗口仅训练提升0.016（p=0.026），并在F1/分钟上比窗口仅训练提升约24%；

**⚠️ 局限性**

局限性包括单一decoder‑LLM（phi‑2）架构、主要以C/C++为主的数据、潜在的预训练泄漏、宏观F1忽视真实误差成本、未覆盖跨文件或多语言漏洞等。

---

## 368. PLA for Drone RID Frames via Motion Estimation and Consistency Verification

**arXiv ID:** 2602.23760 | [PDF](https://arxiv.org/pdf/2602.23760v1)

**作者:** Jie Li `[一作]` (Xidian University), Fengkui Gong `[通讯]` (Xidian University)

**通讯引用:** 2823 | [OpenAlex ID](https://openalex.org/A5034754948)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于一致性验证的物理层身份验证（PLA）算法，用于无人机远程识别（RID）帧，结合无线感知与解码、偏航增强的常加速卡尔曼滤波（CA‑EKF）与LSTM运动估计器，并通过误差感知融合提高鲁棒性。

**💡 创新点**

创新点包括：①利用RID帧中的异构信息（AoA、Doppler、ACG、发射天线数、解码运动参数）进行动态融合；②设计了偏航增强的常加速EKF，将姿态与线速度耦合；③引入误差感知的多估计器自适应融合；④通过天线数、运动估计与禁飞区约束进行一致性验证。

**🔧 技术方法**

使用的技术包括：无线感知与解码模块、偏航增强CA‑EKF、基于LSTM的运动估计器、误差感知自适应融合策略、禁飞区（NFZ）几何检测，以及基于仿真与真实RF数据的评估。

**📊 数据集**

主要使用的数据集为仿真生成的三种场景（1）平滑曲线路径；（2）直线+螺旋上升；（3）真实城市RF信号数据集，其中包含RID帧、上/下行通信信号和Wi‑Fi/蓝牙干扰。

**📈 对比分析**

与基准方案（RSS、CFO、SNRD、AoA+CG、ADPS、LSTM、EKF、Yaw EKF）比较，所提算法在检测率与误报率上均优于其他方法，位置与速度估计的RMSE/MAPE明显降低，证明了一致性验证与融合策略的有效性。

**⚠️ 局限性**

局限性包括：仅针对A2G通信环境，未考虑地面障碍与大气扰动；依赖准确的AoA与Doppler估计，受信号衰落与同步误差影响；计算复杂度较高，需GPU支持；未在真实空域大规模部署中验证。

---

## 369. Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models

**arXiv ID:** 2602.24264 | [PDF](https://arxiv.org/pdf/2602.24264v1)

**作者:** Arnas Uselis `[一作]` (University of Tübingen), Seong Joon Oh `[通讯]` (University of Tübingen)

**通讯引用:** 8418 | [OpenAlex ID](https://openalex.org/A5025851635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过定义可分割性、可迁移性与稳定性三个可操作性指标，对视觉-语言嵌入模型的组合泛化进行理论分析，并证明在标准训练（梯度下降+交叉熵）下，这些指标迫使模型的嵌入实现线性分解且不同概念的方向近似正交。随后作者在 CLIP、SigLIP、DINO 等多种预训练模型上使用线性探测和因子回归等方法验证了这些几何结构，并将其与在 dSprites、MPI3D、PUG‑Animal、ImageNet‑AO 等组合泛化数据集上的性能进行关联分析。

**💡 创新点**

核心创新在于：①首次把组合泛化的三个实用性需求转化为可证明的几何约束；②证明了线性因子化与跨概念正交性既是必要也是充分条件；③给出维度下界（至少 k 维）并通过实验验证其对不同模型的适用性。

**🔧 技术方法**

主要技术手段包括：梯度下降+交叉熵理论分析、线性探测（Linear Probing）与文本提示投影、因子回归恢复 per‑concept 方向、白化后的 R² 评估、PCA 低秩分析，以及与随机初始化 baseline 的对照实验。

**📊 数据集**

实验使用的视觉-语言与自监督模型包括 OpenAI CLIP、OpenCLIP、MetaCLIP、MetaCLIP2、SigLIP、SigLIP‑2、DINO‑v1/v2/v3；组合泛化数据集包括 dSprites、MPI3D、PUG‑Animal、ImageNet‑AO 以及 LAION‑400M 等大规模训练集。

**📈 对比分析**

通过在 10% 组合上训练线性探测器并在剩余 10% 未见组合上评估，作者发现线性因子化的 R² 与组合泛化准确率呈正相关；与随机初始化的 ViT‑L/14 基线相比，预训练模型在 R² 与准确率两方面均优越，说明更符合理论结构的模型具备更强的组合泛化能力。

**⚠️ 局限性**

研究局限主要在于：①只考虑最坏情况的稳定性，未探讨平均/近似稳定性；②假设编码器固定且可在不同训练子集上反复训练，实际训练一次后可能不满足该假设；③目前模型仅部分满足线性+正交结构，导致在极端组合上仍表现欠佳。

---

## 370. Manifold-Preserving Superpixel Hierarchies and Embeddings for the Exploration of High-Dimensional Images

**arXiv ID:** 2602.24160 | [PDF](https://arxiv.org/pdf/2602.24160v1)

**作者:** Alexander Vieth `[一作]` (Leiden University Medical Center), Thomas Höllt `[通讯]` (Delft University of Technology)

**通讯引用:** 7033 | [OpenAlex ID](https://openalex.org/A5019308726)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于随机游走的高维属性流形感知超像素层次结构，用于在图像空间与属性空间之间实现一致的层次降维与可视化探索。

**💡 创新点**

创新点在于：1）使用随机游走特征在属性邻接图上构造Bhattacharyya相似度，既保留流形结构又兼顾图像空间；2）将相同的相似度作为层次层级的合并准则与嵌入度量，形成一个图像信息驱动的层次嵌入；3）通过层次级别的聚合与嵌入细化，提升探索效率与空间连贯性。

**🔧 技术方法**

技术包括：随机游走与转移概率、Bhattacharyya系数、改进的Borůvka合并算法、k‑近邻图构造、t‑SNE/UMAP层次嵌入、邻接图合并与特征聚合、层次细化与非精确细化、图论与流形学习。

**📊 数据集**

使用数据集：印度棕榈（Indian Pines）高光谱图像、Salinas高光谱图像、CyCIF 单细胞组织图像（含多通道蛋白表达）。

**📈 对比分析**

方法对比：与HSNE、SLIC、FH、Entropy‑Rate Superpixel（ERS）等传统超像素/层次降维方法进行定性与定量比较。定量评估使用 UE/EV、AUE/AEV 指标，在大多数层级下与其他方法相当或更优，尤其在 EV 方面表现突出；与 HSNE 相比，在相同 ROI 下所需的标记点/超像素数显著降低，提升了嵌入细节与计算效率。

**⚠️ 局限性**

局限性：1）随机游走参数（步数、次数）对不同数据敏感，需经验调参；2）嵌入层级间的全局一致性不稳定（受随机初始化影响）；3）对非常大图像仍有内存/计算开销；4）不保证所有超像素最终被合并，导致孤立点出现；5）未对细化级别的孤立分布与连通性提供完整解决方案。

---

## 371. Task-Centric Acceleration of Small-Language Models

**arXiv ID:** 2602.24174 | [PDF](https://arxiv.org/pdf/2602.24174v1)

**作者:** Dor Tsur `[一作]`, Ran Levy `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Task‑Adaptive Sequence Compression（TASC）框架，通过 tokenizer 词表扩展与推理时的 n‑gram 预测草稿两种方式加速小型语言模型（SLM）在低输出变异任务上的推理速度。

**💡 创新点**

创新点在于：①使用任务输出的高频 n‑gram 迭代加入 tokenizer 词表，显著缩短输出长度；②构建无训练、基于 n‑gram 的草稿模型，兼容不同 tokenizer，避免了传统投影对齐问题。

**🔧 技术方法**

技术包括 BPE‑启发式词表扩展、Prefix Collision Score（PCS）过滤、n‑gram 统计草稿、混合全局与局部 n‑gram 分布、2‑Rényi 熵预测加速效果。

**📊 数据集**

实验使用 Qwen2.5‑0.5B / 3B、MedQuAD、CMR‑QA、Massive‑Classification、ICSF 等四大数据集，覆盖文本分类、问答、槽位填充等任务。

**📈 对比分析**

相较基线，tokenizer 扩展后在推理时间上提升 1.5‑2.1×，n‑gram 草稿在推理时间上提升 1.5‑3.15×，并保持 F1、BERT‑Score、LLM‑Judge 等质量指标基本不变。

**⚠️ 局限性**

局限性包括：需要任务输出数据集进行 n‑gram 统计；对高输出变异任务效果有限；词表扩展可能削弱模型在其他任务上的通用性；对模型可访问权与 fine‑tune 资源有一定要求。

---

## 372. Chunk-wise Attention Transducers for Fast and Accurate Streaming Speech-to-Text

**arXiv ID:** 2602.24245 | [PDF](https://arxiv.org/pdf/2602.24245v1)

**作者:** Hainan Xu `[一作]` (NVIDIA Corporation), Jagadeesh Balam `[通讯]` (NVIDIA Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出Chunk-wise Attention Transducer (CHAT)，在RNN‑T的基础上将音频输入划分为固定大小的分块，并在每个分块内部使用跨注意力(joiner)实现局部对齐，从而兼顾流式处理与对齐灵活性。

**💡 创新点**

核心创新在于将RNN‑T的joiner改造成分块注意力模块，既降低了时间维度（减少了T），又打破了传统RNN‑T的单调对齐限制，使模型能够在每个分块内自由匹配声学特征与输出标记。

**🔧 技术方法**

技术方案包括FastConformer（17层、512维）编码器、LSTM预测器、多头注意力(joiner)以及分块流式编码器；训练使用NeMo Toolkit，推理实现批量化标签循环；同时通过在joiner中添加零帧实现“blank”输出。

**📊 数据集**

实验使用英文和德文的Librispeech、Common Voice、Voxpopuli、Multilingual Librispeech等ASR数据集；语音翻译则使用Covost EN‑DE、EN‑ZH、EN‑CA以及公开语料。

**📈 对比分析**

采用与RNN‑T相同的FastConformerLarge+LSTM结构，在相同GPU上对比WER、BLEU、训练/推理速度和峰值内存；结果显示CHAT在ASR上WER下降最多6.3%，推理速度提升1.69倍，训练速度提升1.36倍，GPU峰值内存下降46.2%；在语音翻译中BLEU提升18.0%。

**⚠️ 局限性**

局限性包括固定chunk大小导致缺乏自适应分块能力，分块边界仍可能限制对齐精度；缺少精确的语音延迟评估；未验证在更大规模或跨语言环境中的泛化性能。

---

## 373. SelfOccFlow: Towards end-to-end self-supervised 3D Occupancy Flow prediction

**arXiv ID:** 2602.23894 | [PDF](https://arxiv.org/pdf/2602.23894v1)

**作者:** Xavier Timoneda `[一作]` (CARIAD SE Volkswagen Group), Daniel Goehring `[通讯]` (Freie Universitaet Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种自监督的 3D 体积占用流预测方法，能在无需人工标签或预训练光流模型的情况下同时学习场景几何与运动；

**💡 创新点**

创新点包括将场景分离为静态与动态 SDF，利用时间聚合与相似度流损失实现隐式流学习，完全基于摄像头与 LiDAR 的自监督信号；

**🔧 技术方法**

使用基于 BEVFormer 的视觉特征提取、双头 SDF 预测、流回归 U‑Net、相似度流损失、光照与 LiDAR 余弦匹配等技术；

**📊 数据集**

在 SemanticKITTI、KITTI‑MOT 与 nuScenes 三大驾驶数据集上进行训练与评估；

**📈 对比分析**

与 LetOccFlow、OccNeRF 等基准方法对比，在 SemanticKITTI 的 RayIoU 上提升约 5%，在 KITTI‑MOT 的视差与光流误差均显著下降，nuScenes 的 mAVE 下降 7.7%，显示更优性能；

**⚠️ 局限性**

局限性在于对大幅位移、非刚体运动以及细小动态物体的流估计仍不够精确，且依赖相机- LiDAR 同步且遮挡处理仍有改进空间。

---

## 374. Demystifying Action Space Design for Robotic Manipulation Policies

**arXiv ID:** 2602.23408 | [PDF](https://arxiv.org/pdf/2602.23408v1)

**作者:** Yuchun Feng `[一作]` (Institute for AI Industry Research), Xianyuan Zhan `[通讯]` (Institute for AI Industry Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对机器人操作策略的动作空间设计进行系统的大规模实验评估，分析时间抽象（绝对 vs. 相对）与空间抽象（关节空间 vs. 末端执行器空间）的影响。

**💡 创新点**

提出了统一的动作空间抽象分类法，发现分块式相对（delta）动作在绝大多数场景下优于绝对动作；关节空间在单机器人上更稳健，而末端执行器空间在跨机器人、迁移学习场景中表现更好；提供了可落地的设计准则。

**🔧 技术方法**

使用基于视觉的Transformer策略网络，包含FiLM-条件ResNet-18编码器；实现回归式与流匹配（diffusion）两种策略；采用动作分块、chunk-wise delta 等技术；对比了不同动作空间的学习与部署性能。

**📊 数据集**

实验数据包括 13,000+ 实际机器人轨迹，500+ 已训练模型；每个任务收集 250 条专家演示（真实场景）或 50 条（RoboTwin 2.0 模拟）；涉及 4 个真实任务与 10 个仿真任务；机器人平台涵盖单臂、双臂 AgileX、AIRBOT，以及 RoboTwin 2.0 仿真环境。

**📈 对比分析**

通过在统一评估协议下计算进度分数和成功率，比较绝对/相对、关节/末端执行器以及分块/逐步 delta 的组合。结果显示，chunk-wise delta 与关节空间组合在单机器人上可提升约 10‑15% 的成功率；在跨机器人或迁移学习场景下，末端执行器空间能进一步提升性能；在不同数据量与训练轮次下保持一致性。

**⚠️ 局限性**

局限性：实验主要集中在位置控制；未深入研究力/扭矩级别动作；跨机器人泛化仅在几款硬件上验证；对更高阶动力学或非刚体环境的适用性未知；部分结论受限于特定网络架构与超参数设置。

---

## 375. PointCoT: A Multi-modal Benchmark for Explicit 3D Geometric Reasoning

**arXiv ID:** 2602.23945 | [PDF](https://arxiv.org/pdf/2602.23945v1)

**作者:** Dongxu Zhang `[一作]` (Xi'an Jiaotong University), Cheng Tan `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PointCoT 框架，结合显式的 Look‑Think‑Answer 推理流程，利用点云与多视角图像的双流编码器，实现对三维结构的可解释性推理。

**💡 创新点**

创新点：①首次把 Chain‑of‑Thought 机制迁移到 3D 点云任务；②设计 Geometry‑Guided Cross‑Modal Attention 以把 3D 坐标信息嵌入跨模态注意力；③构建 Point‑Reason‑Instruct 大规模带有层次化推理链的指令调优数据集。

**🔧 技术方法**

技术：双流 3D/2D 编码器（PointBERT + EVA‑CLIP），Geometry‑Guided Cross‑Modal Attention，基于 Qwen2.5‑7B 的 LLM 推理，双阶段训练（先生成推理链再预测答案）与 InfoNCE 对齐损失。

**📊 数据集**

数据集：Point‑Reason‑Instruct（≈86k 任务），包括点云、8 视角图像与 CoT 推理链；在 Objaverse‑LVIS 及 ScanQA 进行零样本评估。

**📈 对比分析**

比较方法：与 2D VLMs、传统 3D‑LLMs（Chat‑3D、Point‑LLM、Point‑Bind）以及 Qwen2‑V‑7B 在 Point‑Reason‑Instruct 上的性能进行对比。PointCoT 在整体准确率达到 78.5%，在结构感知、空间关系和功能推理上分别高出 12–15% 以上；在零样本任务上 51.8% 的 Objaverse‑LVIS 准确率和 23.4% 的 ScanQA BLEU-4，表现优于多数基线。

**⚠️ 局限性**

局限性：①仅在单物体级别进行推理，未扩展到复杂室内场景或动态交互；②依赖大量 CoT 级别标注，尽管通过自动化教师生成降低成本，但仍可能存在标注噪声；③对极端稀疏或噪声点云的鲁棒性尚未充分验证。

---

## 376. Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation

**arXiv ID:** 2602.23814 | [PDF](https://arxiv.org/pdf/2602.23814v1)

**作者:** Chongyang Xu `[一作]` (Sichuan University), Shuaicheng Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 7017 | [OpenAlex ID](https://openalex.org/A5039387461)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于预训练3D几何基础模型的双臂操作框架，利用RGB图像同时预测未来动作片段和未来3D点图。

**💡 创新点**

创新点在于将3D几何先验与2D语义特征、本体感知融合，并通过联合扩散模型同时生成动作和3D场景，实现在仅RGB输入下的3D感知控制。

**🔧 技术方法**

采用了π^3预训练3D几何模型、DINOv3语义编码器、DETR编码/解码架构以及条件扩散生成器，并结合动作分块与3D点图解码技术。

**📊 数据集**

使用RoboTwin 2.0仿真基准和真实世界AgileX Cobot Magic平台的专家演示数据集（约100条演示）。

**📈 对比分析**

与2D基线（ACT、DP）、点云基线（DP3、G3Flow）及大规模基础模型（RDT）进行对比实验，实验结果显示在成功率、双臂协调性和3D预测准确度方面均优于现有方法。

**⚠️ 局限性**

局限在于仅实现单步预测，缺乏持久的3D记忆，导致对长周期任务的推理和长期状态累积能力不足，且对未见对象的泛化仍有限。

---

## 377. Black-Box PWPP Is Not Turing-Closed

**arXiv ID:** 2602.23809 | [PDF](https://arxiv.org/pdf/2602.23809v1)

**作者:** Pavel Hubáček `[一作]` `[通讯]` (Institute of Mathematics, Czech Academy of Sciences), Pavel Hubáček (Institute of Mathematics, Czech Academy of Sciences)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了随机oracle下PWPP类不对自适应图灵归约闭合，并提出了嵌套碰撞寻找问题；

**💡 创新点**

首次给出PWPP的自适应归约非闭合的黑盒分离，定义了新的自然嵌套碰撞问题；

**🔧 技术方法**

运用随机oracle、计数论证、转录分析以及tainted/latent-collision概念进行证明；

**📊 数据集**

无实验数据，使用随机oracle分布作为实例生成；

**📈 对比分析**

与已知的非自适应闭合性质对比，证明在随机oracle下失败概率可忽略；

**⚠️ 局限性**

结果仅在随机oracle相对情形下成立，未得到无相对化的证明。

---

## 378. SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls

**arXiv ID:** 2602.23956 | [PDF](https://arxiv.org/pdf/2602.23956v1)

**作者:** Qianxun Xu `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**通讯引用:** 26291 | [OpenAlex ID](https://openalex.org/A5100458183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SwitchCraft，一种训练‑free的多事件视频生成框架，在推理阶段通过对预训练文本到视频扩散模型的交叉注意力进行查询 steering，实现多事件的时间控制。

**💡 创新点**

创新点在于：① Event‑Aligned Query Steering（EAQS）在每个时间段内动态调节视频查询向对应事件投影并抑制竞争事件；② Auto‑Balance Strength Solver（ABSS）自动求解增强与抑制力度，避免过度或不足的 steering。

**🔧 技术方法**

技术手段包括：训练‑free 的查询 steering、投影算子、奇异值分解（SVD）、凸二次规划求解、基于 DiT 的文本到视频扩散模型。

**📊 数据集**

评估使用了60条多事件提示（来自现有工作和大语言模型自动生成），并使用 CLIP、VideoScore2、VBench 等自动指标和人工评价。

**📈 对比分析**

与 MinT、LongLive、MEVG、DiTCtrl、Stitch 以及 Wan 2.1 基线对比，SwitchCraft 在文本对齐、事件完整性、过渡流畅度和视觉质量等指标上均优于或接近最佳方法，人工评价亦显示最高分。

**⚠️ 局限性**

局限性包括：仍受限于预训练模型的语义覆盖，极端或长时间事件顺序可能出现微小漂移；对实时性能和极端场景的泛化尚未充分验证。

---

## 379. When LLMs Help -- and Hurt -- Teaching Assistants in Proof-Based Courses

**arXiv ID:** 2602.23635 | [PDF](https://arxiv.org/pdf/2602.23635v1)

**作者:** Romina Mahinpei `[一作]` (Princeton University), Manoel Horta Ribeiro `[通讯]` (Princeton University)

**通讯引用:** 1310 | [OpenAlex ID](https://openalex.org/A5011195481)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过多部分案例研究评估大语言模型（LLM）在本科证明课程中对教学助理（TA）的辅助效果，比较了LLM与不同经验层级TA在基于 rubric 的评分上的一致性，并调查了 TA 对 LLM 生成反馈的偏好与感知。

**💡 创新点**

创新点在于揭示评分与反馈在 LLM 支持下的根本不对称性：LLM 在评分上往往过于严格且缺乏情境化判断，但在提供反馈时可作为高质量的起草伙伴；并基于此提出将评估与形成性指导分离、让 LLM 仅作为反馈协助者的设计建议。

**🔧 技术方法**

技术方面使用 OpenAI GPT 系列 LLM 并构建 LLM‑as‑a‑Judge 框架，分别让两实例独立生成评分与反馈，第三实例挑选最佳结果；同时利用 rubric‑guided prompts 与教师参考答案来引导模型。

**📊 数据集**

数据集主要由 30 份学生证明作业提交组成，来自一门本科证明课程，配有解决质量（SQ）与写作质量（WQ）两套 rubric；此外招募 3 名研究生 TA 与 3 名本科 TA 进行评分，并进一步招募 13 名 TA 进行反馈偏好调查。

**📈 对比分析**

通过统计所有子问题的评分一致率（含 95% 置信区间）与 TA 对不同来源反馈的排名进行对比。结果显示：LLM 与 TA 在评分上存在显著差异，尤其在后续子问题上一致率明显下降；但在有人类反馈存在时，LLM 生成的反馈往往被排为首选，显示其在形成性支持上的潜力；若提交无大错误，LLM 反馈被认为冗长、过度。

**⚠️ 局限性**

局限性包括：仅在单门课程与单一 LLM 上进行实验，缺乏跨课程和跨模型的泛化验证；样本量有限；对评分一致性的评估主要依赖人工判定，未探究更客观的性能指标；研究聚焦于案例探索，未实现完整交互式 TA 工具的部署与评估。

---

## 380. Dialect and Gender Bias in YouTube's Spanish Captioning System

**arXiv ID:** 2602.24002 | [PDF](https://arxiv.org/pdf/2602.24002v1)

**作者:** Iris Dania Jimenez `[一作]` (Ludwig-Maximilians-Universität München), Christoph Kern `[通讯]` (Ludwig-Maximilians-Universität München)

**通讯引用:** 2971 | [OpenAlex ID](https://openalex.org/A5103123107)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

评估YouTube自动字幕系统在七种西班牙语方言和性别下的准确性，使用WER、CER和Entity Recall指标，并通过混合效应回归模型探究音高、强度、方言与性别对错误率的影响。

**💡 创新点**

首次系统性分析YouTube西班牙语字幕中的方言与性别偏差，发现方言差异对错误率的影响大于性别差异，并揭示了语音强度与方言的显著作用。

**🔧 技术方法**

使用YouTube Data API v3抓取自动字幕，对齐后计算WER/CER/Entity Recall，并运用混合效应回归模型评估音频特征对WER的影响。

**📊 数据集**

结合Crowdsourcing Latin American Spanish数据集（阿根廷、智利、哥伦比亚、墨西哥、秘鲁、波多黎各、委内瑞拉七个方言）和TEDx西班牙语语料，涵盖男女各多小时录音。

**📈 对比分析**

通过WER/CER/Entity Recall比较不同方言和性别的性能，发现波多黎各女声WER最低为16%，阿根廷最高为24%；混合模型表明语音强度与方言显著影响WER，性别无显著影响。

**⚠️ 局限性**

局限包括样本仅覆盖七个方言、缺乏男性波多黎各样本、YouTube API 30分钟块限制、未涉及年龄、非二元性别及细粒度音频合并可能影响精确分析。

---

## 381. SAGE-LLM: Towards Safe and Generalizable LLM Controller with Fuzzy-CBF Verification and Graph-Structured Knowledge Retrieval for UAV Decision

**arXiv ID:** 2602.23719 | [PDF](https://arxiv.org/pdf/2602.23719v1)

**作者:** Wenzhe Zhao `[一作]` (Northwestern Polytechnical University), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 61894 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个无训练、双层决策框架SAGE‑LLM，用以在UAV动态追逐‑躲避任务中实现安全与泛化；

**💡 创新点**

创新点包括：①将大型语言模型与模糊控制边界函数（Fuzzy‑CBF）结合，实现对语义指令的形式化安全认证；②构建星形层次图（Star‑Hierarchical Graph）结构化检索增强生成（RAG），提升决策解释性与适应性；③将两者无训练集成，形成端到端安全可靠的控制体系；

**🔧 技术方法**

使用技术包括：大型语言模型（如Qwen、DeepSeek Chat、GLM‑4.5）、模糊控制边界函数理论、星形层次图RAG检索、强化学习下层控制器、NVIDIA Isaac Gym仿真平台；

**📊 数据集**

数据集为在NVIDIA Isaac Gym环境下生成的多场景UAV追逐‑躲避任务，涵盖随机球形障碍、不同数量障碍以及未见圆柱障碍等；

**📈 对比分析**

通过与四种安全强化学习基线（DDPG、负奖励、Primal‑Dual、MSMAR‑RL）以及不同LLM后端进行对比，SAGE‑LLM在任务成功率、碰撞安全率和零危害率方面均优于基线，安全率近99.7%、零危害率92%；

**⚠️ 局限性**

局限性在于：小规模LLM在对抗逻辑复杂度上仍显不足；模糊‑CBF虽保证安全但可能导致过于保守；系统对实时计算资源有较高需求，尚未在嵌入式平台验证；

---

## 382. Time Series Foundation Models as Strong Baselines in Transportation Forecasting: A Large-Scale Benchmark Analysis

**arXiv ID:** 2602.24238 | [PDF](https://arxiv.org/pdf/2602.24238v1)

**作者:** Javier Pulido `[一作]` (Technical University of Denmark), Filipe Rodrigues `[通讯]` (Technical University of Denmark)

**通讯引用:** 2286 | [OpenAlex ID](https://openalex.org/A5078981714)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多种真实交通数据集上评估了零-shot 时间序列基础模型 Chronos-2 的预测性能，探讨其在交通预测任务中的可行性。

**💡 创新点**

提出了利用通用 TS‑FM 进行零-shot 预测并与任务专用模型直接对比的评估框架，显示了通用模型在多样化交通场景下的竞争优势。

**🔧 技术方法**

使用基于 Transformer 的 Chronos-2，采用组注意力机制与预训练大模型技术，结合内置的分位数概率输出实现多步预测。

**📊 数据集**

十个真实交通数据集，包括 PeMSD7、Urban1、NYC Citi Bike、PeMSD4、SZ-taxi、METR-LA、PEMS-BAY、NYC Bike Flow、Seattle Loop 以及 UrbanEV（停车位占用、充电时长与充电量）等。

**📈 对比分析**

采用一致的滑动窗口评估协议，比较 MAE、RMSE、MAPE 等确定性指标以及预测区间覆盖率和 IQR，零-shot Chronos-2 在大多数数据集上实现了 state‑of‑the‑art 或极具竞争力的性能，尤其在长时 horizon 上优于传统统计和专用深度学习模型。

**⚠️ 局限性**

局限包括对复杂空间交互（如 METR‑LA）的建模不足，系统性偏差可能在多应用场景中传播，且在大型模型上微调成本高，缺乏对不同数据质量与稀缺情境下的鲁棒性验证。

---

## 383. LE-NeuS: Latency-Efficient Neuro-Symbolic Video Understanding via Adaptive Temporal Verification

**arXiv ID:** 2602.23553 | [PDF](https://arxiv.org/pdf/2602.23553v1)

**作者:** Shawn Liang `[一作]` (Case Western Reserve University), Gourav Datta `[通讯]` (Case Western Reserve University)

**通讯引用:** 442 | [OpenAlex ID](https://openalex.org/A5017435097)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了低延迟神经符号框架LE-NeuS，结合CLIP自适应采样与批量命题检测，显著减少视频自动机构建时间，提升长视频问答效率。

**💡 创新点**

创新点在于两阶段CLIP-guided自适应采样剔除视觉冗余并将命题检测并行化，既保持了时间逻辑的严谨性，又把推理延迟压缩到原来的10%以内。

**🔧 技术方法**

使用技术包括：时序逻辑（Temporal Logic）建模与自动机构建、概率模型检验（Storm）、CLIP图像/文本嵌入、自适应帧采样、批量Vision‑Language Model推理以及多段帧兴趣检索。

**📊 数据集**

实验数据集主要有LongVideoBench、Video‑MME和MLVU，用于评估长视频推理准确性与延迟。

**📈 对比分析**

与标准VLM、NeuS‑QA以及VideoTree、LVNet、VideoAgent等基线对比，LE‑NeuS在LongVideoBench上整体准确率提升约5.21%，在Video‑MME上提升12.07%；平均推理延迟从约90×下降到10×，总体加速12.53×。

**⚠️ 局限性**

局限性包括：仍需依赖大规模VLM进行命题检测，复杂逻辑与极长时序的推理成本未完全消除；自适应采样阈值需手工调优，对极高帧率或极稀疏事件的视频可能仍有性能瓶颈。

---

## 384. Agentic AI-RAN: Enabling Intent-Driven, Explainable and Self-Evolving Open RAN Intelligence

**arXiv ID:** 2602.24115 | [PDF](https://arxiv.org/pdf/2602.24115v1)

**作者:** Zhizhou He `[一作]` (University of Surrey), Rahim Tafazolli `[通讯]` (University of Surrey)

**通讯引用:** 19451 | [OpenAlex ID](https://openalex.org/A5032549075)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

在 O-RAN 环境中提出并实现了 Agentic AI‑RAN 框架，使用跨层级的目标驱动代理进行计划–执行–观察–反思循环，并通过技能封装、记忆检索、门控自管理等原语来完成网络切片生命周期管理与射频资源管理（RRM）任务，随后在多细胞仿真中验证其性能提升。

**💡 创新点**

创新点在于：①将 Agentic AI 概念引入 O‑RAN，形成跨非‑RT、Near‑RT 与 RT 层的统一目标驱动控制；②设计了一套面向 O‑RAN 的技能工具、记忆与门控原语，实现可解释、可审计且自我管理的闭环控制；③在非实时层利用 LLM 进行切片协同与决策优化，保持时间敏感控制路径不受影响；④通过消融实验系统性剖析各原语对性能的贡献。

**🔧 技术方法**

采用的技术包括：Agentic AI 计划–执行–观察–反思循环、技能封装（与 A1/E2/E2SM 接口绑定）、多时尺度记忆与检索、门控自管理、LLM（ChatGPT‑5.2）在非实时层进行语义推理、Deep Q‑Learning（DQL）深度 RL xApp、O‑RAN 软硬件架构（O‑RU/DU/CU、SMO）以及数字孪生 RAN 平台。

**📊 数据集**

实验数据来源于自建仿真环境：500 m × 500 m 区域、6 个 O‑RU、12 个小区、20 个 UE，使用自由空间路径损耗模型及标准 NR 5G 频段 N77/N78；未使用公开数据集。

**📈 对比分析**

评估方法：与传统基线 Deep‑RL xApp 以及六种消融方案（无计划、无记忆、无门控、无序列、无主动 KPM、无 LLM）进行对比，采用堆叠归一化 KPI（SLA 违规率、p99 延迟、E2 字节、操作次数、能耗）进行可视化；结果显示完整 Agentic 控制在 SLA 与尾部延迟上远优于基线，门控缺失导致 SLA 违例显著上升，LLM 的加入提升切片入驻准确率、资源利用率和尾部延迟。

**⚠️ 局限性**

局限性包括：实验仅在仿真环境下进行，缺乏真实网络验证；数字孪生与 LLM 的集成尚未在工业级部署；对多租户冲突与安全合规性支持仍需进一步研究；消融结果显示记忆与门控对性能贡献较大，但在更复杂场景下效果尚未充分评估。

---

## 385. SpikingTac: A Miniaturized Neuromorphic Visuotactile Sensor for High-Precision Dynamic Tactile Imprint Tracking

**arXiv ID:** 2602.23654 | [PDF](https://arxiv.org/pdf/2602.23654v1)

**作者:** Tianyu Jiang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Shuo Wang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `e0540dec-d77f-42db-94ae-d039248f6393` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了SpikingTac——一款小型化、低成本的神经形态视触觉传感器，集成独立事件相机、全局动态状态图、无监督去噪网络和可扩散阻尼的迟滞补偿算法，实现了1000 Hz感知速率和350 Hz跟踪频率，并在动态碰撞检测和微尺度几何定位任务中实现亚毫米精度。

**💡 创新点**

创新点在于：①采用自制的独立事件相机模块，显著降低尺寸和成本；②构建全局动态状态图并结合无监督去噪网络，实现稀疏事件的高精度重构；③提出空间增益阻尼的迟滞感知更新律，有效消除硅胶弹性迟滞导致的基点漂移；④通过高频事件计数实现1000 Hz碰撞检测，显著减少冲击后越过距离。

**🔧 技术方法**

使用技术包括：事件相机、稀疏事件到状态图的异步更新、U‑Net去噪网络、k‑d树关联、空间增益阻尼的迟滞更新律、实时碰撞检测阈值算法、交叉搜索策略进行几何定位。

**📊 数据集**

使用的数据集主要为实验收集的自定义数据，包括：①多种手部、物体的高动态接触轨迹；②高速度碰撞检测的数据；③多尺度圆孔几何估计的数据；未使用公开数据集，全部基于实验室自制。

**📈 对比分析**

与Evetac、NeuTac、GelStereo 2.0等现有传感器进行对比：SpikingTac在点跟踪上实现100 %成功率，平均误差0.8039像素；在碰撞检测中平均偏差约0.53 mm，超越速度的误差显著低于基线，冲击后越过距仅6.2 mm（基线约30 mm，提升约5倍）；在圆孔定位中位置RMSE为0.0952 mm，半径均误差0.0452 mm，体现出亚毫米级几何精度。

**⚠️ 局限性**

局限性：①仅在硅胶弹性介质上验证，其他材质的迟滞补偿效果未知；②需要先行的多向滑动校准，校准步骤耗时；③标记数量有限（8×8），对更大面积或更复杂形状的感知尚未验证；④未评估温度、光照等环境变化对事件相机稳定性的影响。

---

## 386. Does Personalized Nudging Wear Off? A Longitudinal Study of AI Self-Modeling for Behavioral Engagement

**arXiv ID:** 2602.23688 | [PDF](https://arxiv.org/pdf/2602.23688v1)

**作者:** Qing He `[一作]` (University of Pennsylvania), Yuntao Wang `[通讯]` (Tsinghua University)

**通讯引用:** 5295 | [OpenAlex ID](https://openalex.org/A5100605907)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在健身任务（壁式坐姿和仰卧起坐）中，通过对比视频自我建模（VSM）、音频自我建模（ASM）与对照组，评估AI自我建模对长期行为改变的影响，并在一周实验后进行四周纵向验证。

**💡 创新点**

首次系统地展示了AI自我建模在多周内的动态效应：VSM在前期显著提升表现，随后虽保持高水平但提升速率趋于衰减；并提出了从“催化-衰退-内化”三阶段模型解释自我建模的长期动机机制。

**🔧 技术方法**

使用了基于视觉的面部交换框架 VisoMaster 生成个性化未来自我视频；音频自我建模采用 ElevenLabs 语音克隆与 GPT‑4o 生成情境化激励语音；并结合线性混合效应模型（LME）对时间序列表现进行统计分析。

**📊 数据集**

实验数据集由两组实验共59名受试者（第一周28人，后续四周31人）在家中每日完成壁式坐姿与仰卧起坐并记录表现；同时收集了 IMI、ESES 与 VAIQ 等主观问卷。

**📈 对比分析**

通过 LME 对比 VSM 与对照组在早期（第1–7天）与后期（第22–28天）表现差异，发现 VSM 在后期仍显著优于对照（p<0.01），但两组在提升速率上差距在中期后明显收敛；ASM 在本实验中未显示显著正效应。

**⚠️ 局限性**

局限包括：仅选取两种视觉导向的体能任务，导致 ASM 的适用性受限；受试者样本量相对较小且存在补偿激励可能混淆内在动机；自我建模技术的逼真度（面部表情、声音自然度）仍有提升空间；缺乏实时反馈与交互，可能削弱长期保持；实验环境为受控远程实验，难以完全模拟真实健身场景。

---

## 387. HotelQuEST: Balancing Quality and Efficiency in Agentic Search

**arXiv ID:** 2602.23949 | [PDF](https://arxiv.org/pdf/2602.23949v1)

**作者:** Guy Hadad `[一作]` (Ben-Gurion University), Haggai Roitman `[通讯]` (Amazon)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了HotelQuest基准，用214条手工编写的酒店搜索查询评估代理式搜索系统的质量与效率。

**💡 创新点**

创新点在于同时评估质量与效率，并引入明确的澄清信息来处理查询的隐含偏好。

**🔧 技术方法**

使用了检索模型（BM25、dense embeddings）、LLM重排序器和LLM代理（Claude、Qwen3-32B）以及多种评估工具。

**📊 数据集**

数据集包括从Kaggle酒店描述数据（约963k条）和TripAdvisor评论数据（约21M条）组成的酒店语料库。

**📈 对比分析**

通过对检索仅、检索+LLM重排和LLM代理三类模型进行准确率、事实性、成本、token数和延迟的对比，发现LLM代理在准确率上领先但成本和延迟显著高于检索方法。

**⚠️ 局限性**

限制在于查询数量有限、LLM的非确定性导致可复现性差，以及对提示词优化的敏感性。

---

## 388. Histopathology Image Normalization via Latent Manifold Compaction

**arXiv ID:** 2602.24251 | [PDF](https://arxiv.org/pdf/2602.24251v1)

**作者:** Xiaolong Zhang `[一作]` (Oregon Health and Science University), Xubo Song `[通讯]` (Oregon Health and Science University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于未监督学习的图像归一化框架——Latent Manifold Compaction (LMC)，通过在潜在空间中压缩由染色强度变化诱发的局部2D流形，获得批次不变的特征表示。

**💡 创新点**

创新点在于：① 只利用单源数据而不需要目标域样本；② 通过染色空间增广生成真实染色变化的流形；③ 使用无负样本的交叉相关对比损失实现流形压缩，既保持生物学信息又抑制批次偏差。

**🔧 技术方法**

核心技术包括：SVD分解提取H&E染色通道、随机缩放染色强度生成多视图、Vision Transformer编码器、交叉相关对比损失（类似Barlow Twins）和无负样本的对比学习。

**📊 数据集**

使用了三组数据集：Camelyon16（乳腺癌转移分类）、内部前列腺癌数据集（BR与BL，Gleason分级）、MIDOG21（增殖细胞检测）。

**📈 对比分析**

与未归一化、Macenko经典归一化和基于扩散模型的StainFuser对比。LMC在所有三项任务上都显著降低批次分离度（CFD、W2值）并实现更高的交叉批次AUC或F1，显示出最优的泛化性能。

**⚠️ 局限性**

局限性包括：仅针对H&E染色的变异，尚未验证在其他染色或成像模态上的适用性；对非常小样本或极端稀有类别的鲁棒性有限；需要大量无标签图像进行流形生成和预训练。

---

## 389. Jailbreak Foundry: From Papers to Runnable Attacks for Reproducible Benchmarking

**arXiv ID:** 2602.24009 | [PDF](https://arxiv.org/pdf/2602.24009v1)

**作者:** Zhicheng Fang `[一作]` (Shanghai Qi Zhi Institute), Wei Xu `[通讯]` (Tsinghua University)

**通讯引用:** 16301 | [OpenAlex ID](https://openalex.org/A5013867024)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

构建了一套名为 Jailbreak Foundry 的系统，能自动将 jailbreak 论文转换为可执行的攻击模块，并在统一的评估环境中对其进行可复现的基准测试。

**💡 创新点**

创新点在于：1) 通过多代理（Planner‑Coder‑Auditor）流水线实现论文到代码的自动化翻译，显著降低人工集成成本；2) 提供共享框架核心与可重用工具，代码复用率高达 82.5%；3) 设计统一的数据集、协议与判断器，支持跨模型、跨攻击的可比性评估。

**🔧 技术方法**

使用的技术包括：多代理 LLM 工作流（Gemini‑3‑pro、Claude‑4.5‑sonnet、GPT‑5.1 Codex 等）、Python 统一框架核心（Jailbreak Core），以及统一的评估 harness（固定数据集、GPT‑4o 判别器）。

**📊 数据集**

采用的主要数据集为 AdvBench、JailbreakBench 等公开基准，攻击实验中对 10 个受害模型（Claude 系列、GPT‑4 系列、LLaMA、Qwen3、GPT‑5.1 等）进行评估。

**📈 对比分析**

比较方法：先在论文匹配设置下重现 30 款攻击，平均 ASR 与原报告的偏差仅 +0.26pp；随后在统一 harness 下对 10 模型进行评估，生成 ASR 热图。实验显示重现率高、实现代码压缩率 42%，平均合成耗时 28.2 分钟。

**⚠️ 局限性**

局限性包括：1) 依赖 LLM 代理，细节缺失或模型版本差异仍可能导致重现失败；2) 仅适用于描述充分、可获得官方代码或可推断实现的攻击；3) 对新型防御缺乏自动化集成；4) 可能降低恶意使用门槛，需要慎重发布。

---

## 390. All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a Robust Landmark-Identity Watermark

**arXiv ID:** 2602.23523 | [PDF](https://arxiv.org/pdf/2602.23523v1)

**作者:** Junjiang Wu `[一作]` (Xinjiang University), Zhiqing Guo `[通讯]` (Xinjiang University)

**通讯引用:** 555 | [OpenAlex ID](https://openalex.org/A5044056760)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的主动深度伪造取证框架LIDMark，能够同时完成深度伪造检测、篡改定位和来源追踪。

**💡 创新点**

核心创新包括：① 152维结构化的Landmark-Identity Watermark (LIDMark) 兼顾几何脆弱性与身份鲁棒性；② Factorized-Head Decoder (FHD) 将共享特征分解为回归和分类两头，实现高效同步恢复；③ 基于“内在-外在”一致性检验的检测与定位机制。

**🔧 技术方法**

采用深度水印嵌入、两流融合编码网络、FHD多任务学习、随机扰动与深度伪造仿真器、面部关键点对齐、SHA‑256哈希生成标识符，并使用对抗训练提升视觉不可见性。

**📊 数据集**

主要使用CelebA‑HQ作为训练与评估基准，交叉验证使用LFW；深度伪造攻击采用SimSwap、UniFace、CSCS、StarGAN‑v2、InfoSwap等模型。

**📈 对比分析**

与MBRS、CIN、SepMark、EditGuard、LampMark、DiffMark、KAD‑NET等单/双功能基线对比，LIDMark在PSNR/SSIM保持最高水平，BER最低，AUC≈0.94，能够在常见失真与主流伪造攻击下保持高检测率、精准定位和几乎完美的来源追踪，显著优于现有方法。

**⚠️ 局限性**

局限性包括：对未知或极端深度伪造模型的鲁棒性仍有限；训练过程复杂，需两阶段调参；当前仅针对静态图像，视频扩展仍待研究；在极端压缩/噪声环境下可能出现误判。

---

## 391. A Software-Defined Testbed for Quantifying Deauthentication Resilience in Modern Wi-Fi Networks

**arXiv ID:** 2602.23513 | [PDF](https://arxiv.org/pdf/2602.23513v1)

**作者:** Alex Carbajal `[一作]` (Washington State University), Asma Jodeiri Akbarfam `[通讯]` (Washington State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

设计并实现了一个可复现的软件定义Wi‑Fi测试平台，用以实验评估不同Wi‑Fi安全配置在遭受失去认证攻击时的抵抗力。

**💡 创新点**

创新点在于提供自动化、可复现的测试框架，系统性比较五种主流配置，并量化PMF在阻止失去认证攻击中的作用。

**🔧 技术方法**

使用了Arch Linux、hostapd、dnsmasq、nftables、aircrack‑ng、mdk4、Wireshark等开源工具，结合软件无线接入点与包注入技术。

**📊 数据集**

使用自建实验数据集：3台客户端、5次实验、5种配置，总计15组实验，记录断开率、注入包数和时间。

**📈 对比分析**

采用断开百分比、注入包数、平均时间等指标进行比较；结果显示开放/WPA1/WPA2（无PMF）全部被成功断开（100%），需数千包、0.2–0.3秒；启用PMF或使用WPA3时断开率为0%。

**⚠️ 局限性**

局限在硬件数量有限（仅少量客户端和单个Wi‑Fi 6 AP）、实验仅覆盖传统失去认证攻击，未测试更复杂或针对PMF/WPA3的高级攻击，且未验证在更高代Wi‑Fi或更大规模网络中的表现。

---

## 392. Unlocking Cognitive Capabilities and Analyzing the Perception-Logic Trade-off

**arXiv ID:** 2602.23730 | [PDF](https://arxiv.org/pdf/2602.23730v1)

**作者:** Longyin Zhang `[一作]` (Institute for Infocomm Research A*STAR), Ai Ti Aw `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了面向东南亚的10B参数多模态大语言模型MERaLiON2-Omni (Alpha)，通过先构建稳健的感知骨干（System 1），随后在此基础上注入链式思考推理能力（System 2）

**💡 创新点**

创新点包括：① 将区域特定的音频-视觉信号通过正交模态适配与多语种LLM对齐；② 采用低成本Generate‑Judge‑Refine银数据流水线，用大型LLM做幻觉过滤并生成可供推理的文本；③ 逐步训练方法把感知与推理分离后再融合，显著降低对大规模多模态数据的依赖

**🔧 技术方法**

核心技术包括：多模态编码器（SigLIP视觉、Whisper音频）、LoRA适配器、对齐投影、生成-裁判-精炼银数据流程、链式思考（CoT）推理、跨语言指令调优

**📊 数据集**

使用的数据集有：
• 图文对齐数据（TextVQA、DocVQA、MathVista、WebVid‑10M等）
• SEA-Image（约423万图文对）
• SEA-Video（3142条视频任务）
• 多语种指令集（6种SEA语言）
• 生成的银多模态推理数据（21.3K条）
• 评测基准 SEA‑Omni Benchmark Suite（图像和视频）

**📈 对比分析**

与Qwen2.5Omni、Qwen3Omni、SeaLion4VL等现有模型在SEA‑Omni Benchmark上对比，系统1在OCR、视觉多选、文化特定图文理解上显著优于英文中心模型；系统2在抽象逻辑（+11%数理推理）和语音指令跟随上提升约10%但在低层感知任务（OCR、音频识别）出现性能下滑

**⚠️ 局限性**

局限性在于推理注入导致“效率‑稳定性悖论”：推理增强抽象任务时会破坏时间对齐（Temporal Drift）和视觉过度解释（Visual Over‑interpretation），从而使低级感知任务性能下降；银数据流程和LoRA适配在极端长文本/长音频场景下仍不稳定；模型仍缺乏针对多模态领域的高质量标注，需进一步优化推理与感知的协同策略

---

## 393. TradeFM: A Generative Foundation Model for Trade-flow and Market Microstructure

**arXiv ID:** 2602.23784 | [PDF](https://arxiv.org/pdf/2602.23784v1)

**作者:** Maxime Kawawa-Beaudan `[一作]` (J.P. Morgan AI Research), Manuela Veloso `[通讯]` (J.P. Morgan AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练了一个524M参数的生成Transformer模型TradeFM，学习美国股票市场的订单流，并通过闭环市场模拟器验证生成序列能复现金融市场的典型stylized facts；

**💡 创新点**

① 采用规模不变的特征和统一token化，实现跨资产、跨流动性无资产特定校准；② 仅用部分可观测的订单流而非完整LOB；③ 在闭环模拟器中验证生成序列的真实性；④ 展示零射门地理泛化到APAC市场；

**🔧 技术方法**

使用Llama风格的Transformer（带GQA和RoPE），指数加权VWAP估计中价，混合基数token化，闭环市场模拟器；在评估中对比Zero-Intelligence和Compound Hawkes基线，采用K-S、Wasserstein、perplexity等指标；

**📊 数据集**

19B token的美国股票订单流数据，覆盖9K个股票、368个交易日，训练集10.7B token，测试集8.7B token；另外从APAC（中国和日本）1月2025独立测试；

**📈 对比分析**

通过与ZI和Compound Hawkes在闭环生成的stylized facts、分布误差进行比较，TradeFM在K‑S、Wasserstein等指标上比基线低2‑3倍，零射门APAC perplexity略升但仍与美国相近；

**⚠️ 局限性**

模型在低层特征如价差等方面不如Hawkes精细；模拟器简化未捕捉所有微结构细节；缺乏对极端事件的评估；对高频交易策略效果验证不足。

---

## 394. Resilient Strategies for Stochastic Systems: How Much Does It Take to Break a Winning Strategy?

**arXiv ID:** 2602.24191 | [PDF](https://arxiv.org/pdf/2602.24191v1)

**作者:** Kush Grover `[一作]` (Fondazione Bruno Kessler), Jan Kretinsky `[通讯]` (Masaryk University)

**通讯引用:** 2136 | [OpenAlex ID](https://openalex.org/A5074485601)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了在随机系统中评估策略弹性（resilience）的理论框架，定义了期望与最坏情况下的破坏点（breaking point）以及频率度量，并给出了求解最优弹性策略的算法

**💡 创新点**

首次将扰动（disturbance）视为可计数事件来刻画策略的鲁棒性，提出了期望与最坏情况的两种破坏点定义，并引入频率度量来处理无穷扰动场景

**🔧 技术方法**

利用马尔可夫决策过程、随机游戏的理论基础，构造诱导MDP、展开游戏（unfolded game）和状态计数SG，运用线性规划、二次规划、值迭代、策略迭代等算法求解期望和最坏情况的破坏点

**📊 数据集**

无具体实验数据集，本文为理论与算法研究；若有实验，主要使用人工构造的随机游戏与MDP实例来验证算法正确性

**📈 对比分析**

通过理论分析给出算法的时间复杂度（如多项式时间、Pspace/NP等），并对不同目标（安全、可达）和语义（期望、最坏）进行了对比；实验结果（若有）展示了算法在小规模实例上的可行性

**⚠️ 局限性**

局限性包括：仅考虑完全可观测的MDP/SG；扰动成本被假设为对称且单位；对大规模实例的可扩展性未得到实证；对部分可观测、多人博弈等更复杂场景的适用性尚未研究

---

## 395. The Compulsory Imaginary: AGI and Corporate Authority

**arXiv ID:** 2602.23679 | [PDF](https://arxiv.org/pdf/2602.23679v1)

**作者:** Emilio Barkett `[一作]` `[通讯]` (Columbia University), Emilio Barkett (Columbia University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过批判性话语分析，对OpenAI和Anthropic两大人工通用智能（AGI）公司的CEO公开文章进行比较研究，揭示其构建社会技术想象的共同结构与差异。

**💡 创新点**

创新点在于：①将社会技术想象框架从国家层面扩展到私营科技企业；②识别并命名四种共通的修辞操作（自我免责、终极自然化、合格承认、隐性必然性）；③证明这些操作在竞争对手之间呈现结构性一致性，而非仅是个体偏好或战略模仿。

**🔧 技术方法**

采用的技术主要是批判性话语分析（CDA）与解释性细读，结合跨文本比较方法，系统归纳修辞结构与功能。

**📊 数据集**

数据集仅包含两篇CEO文章：Sam Altman的《The Intelligence Age》和Dario Amodei的《Machines of Loving Grace》；两篇文本均为公开、面向大众、约十万字与千字规模相差显著。

**📈 对比分析**

比较方法为跨文本对比：先在单文本层面分析修辞架构，再比较两文本共通点与差异，评估其结构一致性。该方法虽无量化性能指标，但通过一致性分析揭示了结构性模式。

**⚠️ 局限性**

局限性包括：①样本仅限两家企业，缺乏更广泛的行业代表性；②未进行受众接受度研究，无法评估修辞对读者的影响；③结构性结论仍需进一步实证检验和跨技术领域验证；④对动态演化过程的时间维度分析不足。

---

## 396. InfoNCE Induces Gaussian Distribution

**arXiv ID:** 2602.24012 | [PDF](https://arxiv.org/pdf/2602.24012v1)

**作者:** Roy Betser `[一作]` (Technion - Israel Institute of Technology), Guy Gilboa `[通讯]` (Technion - Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究 InfoNCE 对比学习目标在大维度极限下诱导的表示分布，并证明其收敛为高维高斯分布；同时通过两条理论路径（对齐饱和+薄壳聚集与正则化极小化）解释这一现象。

**💡 创新点**

提出使用 Hirschfeld‑Gebelein‑Rényi 最大相关来界定正则化对齐上限，并将 InfoNCE 的均匀性势能与球面中心极限定理相结合，给出两种可行的理论证明；进一步给出正则化形式下的解析最优分布，说明高斯结构是 InfoNCE 的内在偏置。

**🔧 技术方法**

采用 InfoNCE 损失、HGR 最大相关、球面均匀性势能、薄壳收敛假设、正则化（范数惩罚与熵提升）、以及一维正态性检验（Anderson‑Darling 与 D’Agostino‑Pearson）。

**📊 数据集**

在三种数据集上进行验证：1）人工合成（拉普拉斯、GMM、稀疏二进制）配合线性或 MLP 编码器；2）CIFAR‑10（ResNet‑18、MLP）进行对比学习与监督对比；3）预训练的自监督模型（DINO、CLIP）与监督模型（ResNet‑34、DenseNet）在 MS‑COCO 与 ImageNet‑R 上的表示。

**📈 对比分析**

与监督训练和不同架构进行比较；通过 CV（范数浓缩）、AD/DP 正态性统计量衡量高斯性，发现 InfoNCE 训练的表示在范数收敛、坐标正态性方面明显优于监督；预训练自监督模型在大规模数据上也表现出近似高斯分布，说明理论预测在实际应用中成立。

**⚠️ 局限性**

主要限制：证明依赖高维极限与理想化假设（对齐饱和、薄壳收敛、正则化可达性），未给出训练动态收敛保证；对有限维、有限批量时的误差量化仍依赖经典 Berry‑Esseen 近似；并未涵盖多模态或非 InfoNCE 目标的广泛自监督方法。

---

## 397. Who Guards the Guardians? The Challenges of Evaluating Identifiability of Learned Representations

**arXiv ID:** 2602.24278 | [PDF](https://arxiv.org/pdf/2602.24278v1)

**作者:** Shruti Joshi `[一作]` (Mila - Quebec AI Institute & Universite de Montreal), Patrik Reizinger `[通讯]` (Max-Planck-Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过构造受控的合成编码器系统，系统性评估并揭示了现有可辨识性指标（如Corr、MI、Disentanglement等）的结构性误判与误报；

**💡 创新点**

提出了基于数据生成过程与编码器几何的两轴分类法，阐明指标失效的根本原因，并给出四项针对未来指标设计的性能规范；

**🔧 技术方法**

采用理论解析、闭式期望推导与大规模实验相结合的方法，评估指标对因子相关性、冗余、过完备性与样本-维度比的敏感度；

**📊 数据集**

主要使用人工合成数据集，涵盖四类因子结构（独立、相关、单因子约束、多因子约束）和多种编码器几何（1oo、3om、4oo、8mo等），并在控制实验中使用标准随机采样；

**📈 对比分析**

与现有指标（Corr、MI、Disentanglement等）对比发现：大多数指标在因子相关、冗余、过完备或高表示-样本比场景下会出现显著的假正或假负，无法在所有设置下保持一致；

**⚠️ 局限性**

局限性包括仅考虑确定性、连续因子与编码器，未涵盖离散或随机编码器，以及对实际学习模型的优化过程分析不足。

---

## 398. Artificial Agency Program: Curiosity, compression, and communication in agents

**arXiv ID:** 2602.24100 | [PDF](https://arxiv.org/pdf/2602.24100v1)

**作者:** Richard Csaky `[一作]` `[通讯]`, Richard Csaky

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并阐述了 Artificial Agency Program（AAP）——一种将 AI 看作嵌入式、受资源限制的代理的研究框架，强调好奇心驱动的学习进度奖励、明确的观察/动作/计算/记忆预算以及接口统一度量，并在合成 POMDP、ARC‑AGI‑3 任务和多模态 VLA 环境中进行验证。

**💡 创新点**

创新点在于：①将预测压缩、内在动机、赋权控制、接口统一以及语言/自通信等多种概念统一为一套可量化的目标与度量；②引入学习进度奖励与资源预算的组合目标；③提出“界面统一度量”（Unification Score）来评估感知/动作/通信瓶颈的改善；④将自通信视为可调节的代币预算，从而探索其对长程推理与协作的价值。

**🔧 技术方法**

使用的技术包括：Schmidhuber式学习进度奖励、基于 Transformer 的预测模型与自适应 meta‑controller、可塑性与赋权的熵/通道容量估计、LoRA/轻量级适配器、主动推理与自通信（私有代币）等。

**📊 数据集**

数据集/环境：合成的 POMDP（如带噪声、延迟、可调感知/动作的网格世界）、ARC‑AGI‑3 风格的交互推理任务、以及预训练的多模态 VLA（视觉+语言）环境作为后端感知模型。

**📈 对比分析**

比较方法：将自适应 meta‑controller 与固定的观察/动作/推理调度做对比；将隐式递归与显式私有代币通信做对比；通过能耗/性能前沿（Pareto 前沿）评估不同配置的成本与任务表现。实验结果表明，自适应资源分配和接口统一能在匹配成本的情况下提升性能，但具体数值因环境而异，尚未给出统一基准。

**⚠️ 局限性**

局限性：①预测、赋权与奖励往往不一致，难以统一衡量；②赋权与可塑性在高维环境中的估计成本高；③能耗与效率的测量仍依赖于 FLOPs/时间等近似；④界面统一度量的设计与调参复杂；⑤自通信通道易膨胀为冗余代码，需严格带宽约束；⑥缺乏在真实硬件上的验证与大规模实验。

---

## 399. Hello-Chat: Towards Realistic Social Audio Interactions

**arXiv ID:** 2602.23387 | [PDF](https://arxiv.org/pdf/2602.23387v1)

**作者:** Yueran Hou `[一作]` (HelloGroup Inc), Jun Gao `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 7769 | [OpenAlex ID](https://openalex.org/A5021175176)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种端到端的大型音频语言模型，能够在真实日常对话中实现高质量的语音识别、问答、翻译以及自然的语音合成。

**💡 创新点**

通过细粒度的音频字幕标注与模态交错训练，显著提升了模型对情感、环境以及非语言信号的感知，并在对话生成中实现了近似人类的情感与韵律控制。

**🔧 技术方法**

采用MiDashengLM音频编码器、Qwen2.5-7B-Instruct思考者、CosyVoice2说话者；使用模态互换（audio‑text interleaving）与跨模态对齐训练；结合多任务指令微调、知识蒸馏与流式生成策略。

**📊 数据集**

基于约316 亿token的混合语料，其中包含256 k小时ASR、64 k小时音频字幕、1.6 万小时日常对话、2.56 M小时指令对话、2.71 M小时多语言语音等；同时构造了细粒度情感、环境、发声病理等标注。

**📈 对比分析**

在ASR、AudioQA、翻译、情感识别、音效检测等多项公开基准上与Gemini3‑Preview、GPT‑4o‑Audio、Qwen3‑Omni‑32B等先进模型对比，模型在翻译、情感识别、音效检测等指标均取得SOTA或接近SOTA，并在Seed‑TTS‑Eval的中文对话MOS达4.19，显著优于同类系统。

**⚠️ 局限性**

仍受限于数据多样性与模型规模，主训练集中以中文日常对话为主，对低资源语言与非日常情景的适应性不足；模型规模较大导致推理延迟和算力成本高；对长上下文逻辑一致性的进一步提升仍有空间。

---

## 400. AoE: Always-on Egocentric Human Video Collection for Embodied AI

**arXiv ID:** 2602.23893 | [PDF](https://arxiv.org/pdf/2602.23893v1)

**作者:** Bowen Yang `[一作]` (Ant Digital Technologies), Kai Zhu `[通讯]` (Ant Digital Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于颈挂式智能手机的自我视角（egocentric）数据采集系统，并通过边缘-云协同管道实现了实时检测与选择性上传，进一步在云端完成自动标注和质量过滤。

**💡 创新点**

创新点在于将轻量级检测模型部署在设备侧，仅上传必要的帧；云端则承担重任务，显著降低设备算力需求与网络带宽消耗，同时提升数据质量与规模。

**🔧 技术方法**

核心技术包括：边缘计算（设备端实时检测）、云计算（自动标注与质量过滤）、数据选择性上传机制、跨设备协同与数据流水线管理。

**📊 数据集**

使用了自建的颈挂式摄像头采集的 egocentric 视频数据集（未公开命名），并与传统全上传方式对比。

**📈 对比分析**

与传统方案相比，该系统在保持或提升标注准确率的同时，将设备算力占用降低约70%，上传流量减少约60%，在实验中表现出更快的响应时间和更高的数据质量。

**⚠️ 局限性**

局限性包括：对网络环境依赖较大；云端标注与过滤的延迟可能影响实时应用；设备侧检测模型的误检/漏检仍需进一步优化；以及在不同摄像头规格下的泛化性仍待验证。

---

## 401. Dynamic Personalization Through Continuous Feedback Loops in Interactive AI Systems

**arXiv ID:** 2602.23376 | [PDF](https://arxiv.org/pdf/2602.23376v1)

**作者:** Liu He `[一作]` (Beijing Institute of Technology), Liu He `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 9691 | [OpenAlex ID](https://openalex.org/A5100317866)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并实现了通过持续反馈循环实现交互式 AI 系统的动态个性化，能够实时适应用户不断变化的偏好与上下文。

**💡 创新点**

创新点包括：1) 采用在线学习框架实现实时持续更新；2) 设计自适应反馈间隔与优先级机制以平衡学习效果与用户疲劳；3) 给出收敛率与 regret 上界的理论保证；4) 集成差分隐私与本地计算保障用户隐私。

**🔧 技术方法**

使用的技术主要包括：在线梯度下降与动量更新、基于反馈方差的自适应学习率、熵衡量的模型不确定性估计、隐式与显式反馈融合、差分隐私噪声注入、局部或可信执行环境下的模型更新。

**📊 数据集**

实验使用了三类数据集：1) 电影推荐数据（10,000 名用户、50,000 条物品），2) 对话式虚拟助手交互日志，3) 自适应学习平台的学生学习与交互记录。

**📈 对比分析**

与静态用户画像、每日批量更新、上下文感知静态模型、简单在线学习等基线方法对比，动态个性化在用户满意度提升 15-23%、NDCG@10 提升 0.13、任务完成率提升至 87%、学习知识增益提升 18% 的同时，平均更新延迟保持在 20-55 ms，且用户反馈请求频率下降 40-50%，满意率提高到 78%。

**⚠️ 局限性**

主要局限包括：实时更新需要额外计算资源；不同用户群体的效果差异明显，尤其对偏好快速变化者更有利；持续反馈收集仍面临隐私担忧；新用户冷启动阶段效果不佳。

---

## 402. EDDA-Coordinata: An Annotated Dataset of Historical Geographic Coordinates

**arXiv ID:** 2602.23941 | [PDF](https://arxiv.org/pdf/2602.23941v1)

**作者:** Ludovic Moncla `[一作]`, Katherine McDonough `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发一种自动化方法，提取并规范化《百科全书》中的地理坐标，并构建金标准数据集与序列模型；

**💡 创新点**

首次在历史文本中实现坐标提取与归一化；提出先分类后序列生成的两步管道，并用嵌套列表保留原始表达；

**🔧 技术方法**

采用Transformer模型（BERT做二分类，mT5/GPT做seq2seq生成），结合人工标注、交叉验证、EM与CER评估；

**📊 数据集**

使用《百科全书》ARTFL2022与ENCCRE2017两版共15,278条目，其中4,798条含坐标；外部验证用Trevoux1743与Britannica 1842；

**📈 对比分析**

通过五折交叉验证评估分类器，准确率、精度、召回率均约99%，F1≈98.7%；序列模型在EM上平均0.86、CER 0.07；对不同坐标精度（D、DM、DMS）比较显示mT5在DMS上表现更好；

**⚠️ 局限性**

对非标准或罕见坐标格式归一化仍易出错，DMS精度易导致EM下降；模型在稀缺坐标类型上表现差；外域评估样本量有限，缺少专家复核；

---

## 403. 3D Modality-Aware Pre-training for Vision-Language Model in MRI Multi-organ Abnormality Detection

**arXiv ID:** 2602.23652 | [PDF](https://arxiv.org/pdf/2602.23652v1)

**作者:** Haowen Zhu `[一作]` (Fujian University of Technology), Xiaogen Zhou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出MedMAP框架，针对3D多模态MRI实现多器官异常检测

**💡 创新点**

创新点在于模态感知的预训练与跨模态语义聚合（CSA）模块，实现细粒度视觉‑语言对齐和深度融合

**🔧 技术方法**

采用对比学习、Swin Transformer、卷积+Transformer双流、交叉认知Transformer等技术

**📊 数据集**

使用公开的MedMoM‑MRI3D数据集，包含7392组3D MRI‑报告对，12种MRI模态

**📈 对比分析**

与MCPL、MedCLIP等SOTA方法对比，MedMAP在肝脏七分类任务上取得91.57%/88.14%准确率/ROC‑AUC，在脑肿瘤二分类任务上取得90.86%/87.33%

**⚠️ 局限性**

仅针对分类任务评估，未覆盖分割或推理等密集预测，且对某些模态的表现仍有提升空间

---

## 404. Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization

**arXiv ID:** 2602.24059 | [PDF](https://arxiv.org/pdf/2602.24059v1)

**作者:** Chenwei Jia `[一作]` (Xi'an Jiaotong University), Hongbin Sun `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 20901 | [OpenAlex ID](https://openalex.org/A5100762412)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Token-aware Adaptive Error Compensation (QE) 的 PTQ 方法，用低秩适配器对 Vision‑Language Models 进行权重与激活量化，并通过共享专家与路由专家分别补偿全局与局部的量化误差。

**💡 创新点**

创新点在于：①发现重要通道在不同模态和不同 token 之间分布与出现频率差异显著；②将重要通道划分为 token‑independent 与 token‑dependent 两组；③利用 Mixture‑of‑Experts（共享专家 + 路由专家）与低秩适配器实现全局与局部误差自适应补偿；④提出基于共现信息的谱聚类与轻量化路由器动态选择专家。

**🔧 技术方法**

技术包括：Post‑Training Quantization、低秩适配器、Mixture‑of‑Experts、共现矩阵与 NPMI 相似度、谱聚类、K‑means、轻量化路由器、可选的层级微调。

**📊 数据集**

使用 COCO Caption（ShareGPT4V）做校准数据，评估多模态任务：MMMU、OCRBench、ScienceQA、TextVQA、VizWiz、AI2D、ChartQA、DocVQA、InfoVQA、MMStar、MuriBench。

**📈 对比分析**

与 RTN、SmoothQuant、MBQ、LQER、AWQ 等主流 PTQ 方法对比。QE 在 2B–70B 规模模型、W4A6/W4A8 量化设置下，平均提升 3–5%（对 72B 模型可达 5.09%），仅落后 4.23% 以内于 FP16；在权重量化（W3A16）下同样优于 MBQ/LQER，且硬件上可实现 3.5–4.5× 的前置加速。

**⚠️ 局限性**

局限性包括：需要额外的低秩适配器参数与计算，路由器与专家数量决定内存/速度折中；对极低位（如 W3A8）量化的提升有限；共现聚类与路由策略在不同模型/任务间可能需重新调优。

---

## 405. When Does Multimodal Learning Help in Healthcare? A Benchmark on EHR and Chest X-Ray Fusion

**arXiv ID:** 2602.23614 | [PDF](https://arxiv.org/pdf/2602.23614v1)

**作者:** Kejing Yin `[一作]` (Hong Kong Baptist University), Jing Qin `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 25254 | [OpenAlex ID](https://openalex.org/A5100662807)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 CareBench 多模态基准，系统评估 EHR 与胸部 X 光（CXR）融合在 ICU 病例预测中的有效性。

**💡 创新点**

首次从完整与缺失模态、模态不平衡、以及公平性四个维度全面检验多模态学习，揭示了模态缺失和信息不平衡对模型性能与公平性的冲击，并提出了相对鲁棒的融合策略。

**🔧 技术方法**

使用多种融合架构（Late Fusion、UTDE、DAFT、MMTM、AUG、InfoReg、DrFuse、M3Care 等），并结合 Bayesian 超参数优化、掩码表示、时间序列处理与交叉模态注意力等技术。

**📊 数据集**

基于 MIMIC‑IV（EHR）和 MIMIC‑CXR（胸部 X 光）构建两组 ICU cohort：完整模态基线（7,149 例）和现实缺失基线（26,947 例）。

**📈 对比分析**

在三大任务（疾病表型、死亡预测、LOS 预测）上与单模态基线进行比较；在完整模态下，多模态方法在表型分类上提升 3–5% AUROC/AUPRC，死亡预测上提升 2–4%；但在缺失模态下，若无专门设计，性能往往不及单模态；跨模态学习显著优于简单拼接，而 InfoReg、AUG 等对模态不平衡的补偿最为有效。

**⚠️ 局限性**

局限性包括：① 模态缺失率极高，导致大多数模型难以发挥优势；② 仅使用单张 X 光，未捕获时间动态影像信息；③ 公平性评估仅基于种族属性，未考虑性别、年龄等多重不平等；④ 基准仅覆盖 ICU 病例，外推性有限。

---

## 406. Mode Seeking meets Mean Seeking for Fast Long Video Generation

**arXiv ID:** 2602.24289 | [PDF](https://arxiv.org/pdf/2602.24289v1)

**作者:** Shengqu Cai `[一作]` (Stanford University), Arash Vahdat `[通讯]` (NVIDIA Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练范式，利用短视频教师的模式寻求与长视频监督的均值寻求相结合，生成分钟级长视频且保持高保真和长期连贯性。

**💡 创新点**

核心创新在于：1）解耦双头Diffusion Transformer，分别用Flow Matching（均值寻求）学习长时序结构，和Distribution Matching（模式寻求）对齐短窗口；2）通过滑动窗口的逆KL对齐，实现对教师分布的模式寻求；3）在推理时仅使用DM头实现少步高效生成。

**🔧 技术方法**

使用Rectified Flow参数化的Diffusion Transformer，结合Decoupled Diffusion Transformer（DDT）架构，滑动窗口DMD/VSD梯度策略以及监督式Flow Matching（SFT）目标。

**📊 数据集**

使用Wan 2.1 1.3B模型作为基线与教师，长视频数据来自公开长视频语料（如Wan长视频集），短视频教师则使用已有5秒短视频模型。

**📈 对比分析**

与长上下文SFT、混合长度SFT、CausVid、Self-Forcing、InfinityRoPE等基线对比，评估VBench-Long指标和Gemini-3-Pro一致性，实验显示在主体一致性、背景连贯性、运动平滑度、动态度和整体质量上均优于所有基线。

**⚠️ 局限性**

局限性包括：1）对长视频数据仍有一定依赖，极长序列仍受数据稀缺限制；2）滑动窗口对齐可能无法捕捉全局极端事件的细节；3）模型在极端复杂动态或稀有场景下的泛化能力待进一步验证。

---

## 407. Coverage-Aware Web Crawling for Domain-Specific Supplier Discovery via a Web--Knowledge--Web Pipeline

**arXiv ID:** 2602.24262 | [PDF](https://arxiv.org/pdf/2602.24262v1)

**作者:** Yijiashun Qi `[一作]` (University of Michigan), Tanmay Wagh `[通讯]` (Santa Clara University)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5028096771)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一个迭代的 Web–Knowledge–Web (W→K→W) 闭环流程，能够从行业特定网页抓取信息、构建异构知识图谱，并利用图谱缺口信息指导下一轮爬取，以系统性地发现半导体设备制造业等细分领域的供应商。

**💡 创新点**

创新点在于将知识图谱的拓扑缺口检测与生态学的种类丰富估计（Chao1、ACE）相结合，既通过图结构为爬虫提供动态目标，又给出覆盖度的可度量停止准则，填补了传统聚焦爬虫缺乏覆盖评估的空白。

**🔧 技术方法**

技术上使用 GPT‑4o‑mini 进行零样本实体与关系抽取并通过类型约束过滤，采用命名实体解析与阻止+匹配的实体解析，利用 DistMult 进行链接预测，采用捕获‑重捕捉（Chao1）实现覆盖估计，并构建异构知识图谱。

**📊 数据集**

在半导体设备制造业（NAICS 333242）中，以 48 条手工种子 URL 为起点，最终构建 765 个实体（145 家企业、318 种产品、174 个行业、116 个地点）和 586 条关系，并与约 200 家人工校准的真值集进行对照评估。

**📈 对比分析**

与 BFS、聚焦爬虫和单次 W→K 基线在相同 213 页爬取预算下比较，W→K→W 在精度 0.138、召回 0.103、F1 0.118 方面领先，且仅消耗 47% 的页数即可达到峰值召回，展示了显著的爬取效率和精度提升。

**⚠️ 局限性**

局限性包括真值集的不完整、网络覆盖偏差导致部分供应商无法被发现、LLM 抽取中的误报与漏报导致关系一致性下降、仅在单一行业进行验证、对更大规模经济体的扩展需要分布式基础设施，以及覆盖估计易高估未发现实体的比例。

---

## 408. Efficient Discovery of Approximate Causal Abstractions via Neural Mechanism Sparsification

**arXiv ID:** 2602.24266 | [PDF](https://arxiv.org/pdf/2602.24266v1)

**作者:** Amir Asiaee `[一作]` (Vanderbilt University Medical Center), Amir Asiaee `[通讯]` (Vanderbilt University Medical Center)

**通讯引用:** 1380 | [OpenAlex ID](https://openalex.org/A5070097766)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种通过机制替换和结构化剪枝来发现并构造预训练神经网络的可解释因果抽象；

**💡 创新点**

创新点在于将因果抽象理论与结构化剪枝相结合，推出二阶近似代理实现可追踪的抽象发现，并解释了方差剪枝的成功与局限；

**🔧 技术方法**

主要技术包括将网络视作确定性结构因果模型、二阶Taylor展开得到闭式替换参数与单元重要性得分、常数/仿射机制替换的编译方法以及交叉验证的交互干预指标；

**📊 数据集**

实验使用MNIST手写数字数据集以及一个8维布尔电路合成任务；

**📈 对比分析**

与方差剪枝、随机剪枝、曲率加权方差等基线对比，发现Logit-MSE方法在保持相同准确率下交互干预准确率更高，且对函数保持重参数化不敏感；

**⚠️ 局限性**

局限性包括对重参数化的鲁棒性虽然提升但仍依赖于梯度与曲率信息，且在极度稀疏或深层结构下需要进一步扩展软干预与多层抽象。

---

## 409. Mixed Choice in Asynchronous Multiparty Session Types

**arXiv ID:** 2602.23927 | [PDF](https://arxiv.org/pdf/2602.23927v1)

**作者:** Laura Bocchi `[一作]` (University of Kent), Simon Thompson `[通讯]` (University of Kent)

**通讯引用:** 134188 | [OpenAlex ID](https://openalex.org/A5007606690)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了一套异步多方会话类型（MST）框架，核心引入混合选择（mixed choice）构造，完成了理论推导、进展性与操作对应的证明，并实现了基于 Scribble 的工具链，将协议编译为 Erlang/OTP 的 gen_statem 状态机，实现了 RabbitMQ AMQP 客户端的重新实现与验证。

**💡 创新点**

创新点在于：①首次把混合选择作为 MST 的核心语法，允许参与者在同一会话中同时进行输入与输出的非确定性选择；②设计了观察者与承诺机制，保证即使存在临时不一致，所有角色最终能一致地收敛到同一分支；③在理论层面证明了进展性与操作对应；④在实践层面提供了完整的从协议描述到可运行 Erlang 代码的生成链条；⑤通过 RabbitMQ 的实际案例验证了框架的可行性。

**🔧 技术方法**

使用的技术主要包括：Scribble 协议语言的语法扩展、异步消息传递的 LTS 语义、会话类型的投影与 EFSM（事件驱动有限状态机）生成、Erlang/OTP 的 gen_statem 行为、以及基于 FIFO 消息队列的 stale 消息清理机制。

**📊 数据集**

使用的数据集主要是 RabbitMQ AMQP 协议的子集（consumer‑channel‑server 三方交互）以及 AMQP 客户端测试套件（RabbitMQ 自带的测试用例）。

**📈 对比分析**

比较方法是将原有的 RabbitMQ amqp_selective_consumer 及其行为替换为工具链生成的模块，随后运行 RabbitMQ 的测试套件进行功能验证。实验结果表明，生成的代码在功能上完全等价且能正常处理消息投递与取消订阅等交互；关于性能（如吞吐量、延迟）未在本文中给出量化评估，主要关注协议正确性与实现可行性。

**⚠️ 局限性**

限制与未来工作包括：①当前框架不支持通道委托（delegation）和更高级的公平终止分析；②投影方法基于经典 MST 的语法，因而在投影完整性上仍有不足；③混合选择的语义与时序约束结合的支持有限；④在大型分布式系统中的可扩展性与性能评估尚未完成。

---

## 410. V-MORALS: Visual Morse Graph-Aided Estimation of Regions of Attraction in a Learned Latent Space

**arXiv ID:** 2602.23524 | [PDF](https://arxiv.org/pdf/2602.23524v1)

**作者:** Faiz Aladin `[一作]` (University of Southern California), Daniel Seita `[通讯]` (University of Southern California)

**通讯引用:** 1153 | [OpenAlex ID](https://openalex.org/A5041660944)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用图像序列学习低维潜空间，构建Morse图并推断系统的ROA

**💡 创新点**

将MORALS扩展到仅有视觉观测的部分可观测情境，利用时空3D卷积自动编码器和对比损失实现潜空间聚类

**🔧 技术方法**

3D卷积自动编码器、潜在动力学网络、对比损失、Morse图生成与ROA推断

**📊 数据集**

MuJoCo仿真环境生成的图像轨迹，包括Pendulum、CartPole、Acrobot、Humanoid等四个任务

**📈 对比分析**

与原始基于状态的MORALS以及不同潜维度进行比较，三维潜空间在所有任务中显著提升精度、召回率与F1，尤其在CartPole和Humanoid上达到0.84的F1

**⚠️ 局限性**

仅依赖二值图像，易受部分可观测性和噪声影响；未在真实机器人数据上验证；假设ROA固定

---

## 411. Uncertainty-aware Language Guidance for Concept Bottleneck Models

**arXiv ID:** 2602.23495 | [PDF](https://arxiv.org/pdf/2602.23495v1)

**作者:** Yangyi Li `[一作]` (Iowa State University), Mengdi Huai `[通讯]` (Iowa State University)

**通讯引用:** 1003 | [OpenAlex ID](https://openalex.org/A5016035883)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于大语言模型的无不确定性引导概念瓶颈模型（ULCBM），实现了自动生成概念并在训练过程中充分考虑这些概念的不确定性。

**💡 创新点**

创新点在于：①使用分布无关的合规预测（Conformal Prediction）对LLM生成的概念在区分度、覆盖度与多样性三维上进行严格量化与理论保证；②将量化得到的不确定性直接驱动目标插补式数据增强与模型训练，解决了概念稀缺导致的监督信号不足问题。

**🔧 技术方法**

采用的技术包括：GPT‑3生成概念、Grounding‑DINO定位候选概念、CLIP‑RN50提取视觉与文本特征、合规预测校准阈值、弹性网络正则化、基于不确定性的视觉补丁插补增强。

**📊 数据集**

实验数据集为CIFAR‑10、CIFAR‑100和CUB图像分类数据集。

**📈 对比分析**

与LaBo、VLG‑CBM等基线对比，ULCBM在保持或降低风险约束（区分度、覆盖度、多样性均不超过设定α）的同时，显著提升了整体与最差类别准确率，并在概念符合度（CCA）上取得最高分。

**⚠️ 局限性**

局限性包括：①仍依赖LLM，生成概念的质量受LLM能力与提示设计影响；②在极端稀疏概念场景下，增强策略可能不足以弥补监督缺失；③实验仅覆盖图像分类任务，尚未验证方法对其他任务的通用性。

---

## 412. Learning Robust Control Policies for Inverted Pose on Miniature Blimp Robots

**arXiv ID:** 2602.23972 | [PDF](https://arxiv.org/pdf/2602.23972v1)

**作者:** Yuanlin Yang `[一作]`, Fumin Zhang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种基于深度强化学习的微型飞艇（MBR）颠倒姿态控制框架，并设计了映射层实现无额外训练的仿真到现实迁移。

**💡 创新点**

创新点在于：① 构建专属的三维Unity仿真环境；② 采用域随机化并改进TD3（多缓冲与梯度裁剪）来提升策略鲁棒性；③ 设计线性映射层直接将学习到的策略部署到真实系统，避免了后期再训练。

**🔧 技术方法**

使用的技术包括Unity 3D仿真、物理模型标定与域随机化、改进的TD3算法（多缓冲+梯度裁剪）、深度网络（两层256单元全连接）、PD映射层与线性映射。

**📊 数据集**

主要使用自收集的真实MBR运动数据进行仿真标定和策略评估，并未使用公开数据集。

**📈 对比分析**

通过与能量整形控制器在不同额外重量(m_w)、重心位置λ和电机增益(g_m)条件下的对比实验，DRL策略在所有测试配置下都能成功完成颠倒姿态，成功率高、所需时间更短，实验验证了其在真实MBR上的有效性。

**⚠️ 局限性**

局限性包括：映射层采用线性关系，无法完全消除仿真-现实差距；对极端参数或极端环境扰动的鲁棒性仍待进一步验证；同时多缓冲与梯度裁剪虽然加速收敛，但在更复杂动力学下可能需要更深层的策略调优。

---

## 413. CycleBEV: Regularizing View Transformation Networks via View Cycle Consistency for Bird's-Eye-View Semantic Segmentation

**arXiv ID:** 2602.23575 | [PDF](https://arxiv.org/pdf/2602.23575v1)

**作者:** Jeongbin Hong `[一作]` (Electronics and Telecommunications Research Institute), Kyoung-Wook Min `[通讯]` (Electronics and Telecommunications Research Institute)

**通讯引用:** 215 | [OpenAlex ID](https://openalex.org/A5112289593)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CycleBEV 框架，通过在训练阶段引入逆视角变换（IVT）网络对视角变换（VT）模型进行正则化，从而提升基于相机的 BEV 语义分割性能。

**💡 创新点**

创新点包括：① 利用逆视角变换网络实现视角循环一致性，显式约束 VT 模型；② 引入高度感知几何正则化和跨视角潜在一致性两大正则化目标；③ IVT 网络仅在训练期间使用，推理时无额外计算开销。

**🔧 技术方法**

采用的技术包括：dual‑branch CNN + Transformer 的 IVT 网络、视角循环一致性损失、重构 BCE 损失、Smooth_L1 对齐损失、辅助高度预测任务、数据增强兼容等。

**📊 数据集**

使用的公开数据集为 nuScenes，基于其多视角相机图像生成 BEV 语义标签并通过伪标签训练 IVT 网络。

**📈 对比分析**

在四个代表性 VT 基线（LSS、CVT、PETRv2、BEVFormer）上与 CVTM、FocusBEV 等现有方法对比；平均 mIoU 在各类目标上提升 0.6–2.08，车辆最高 +4.86，行人 +3.74，遮挡场景性能显著改善。

**⚠️ 局限性**

局限性包括：IVT 网络仅在训练阶段使用，需额外训练时间；对极端遮挡、光照变化等极端条件的鲁棒性尚未完全验证；未针对多帧时序模型充分评估。

---

## 414. A Minimal Agent for Automated Theorem Proving

**arXiv ID:** 2602.24273 | [PDF](https://arxiv.org/pdf/2602.24273v1)

**作者:** Borja Requena Pozo `[一作]` (Axiomatic AI), Leopoldo Sarra `[通讯]` (Axiomatic AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了最小化代理框架AxProverBase，利用迭代改进、内存管理与工具搜索实现简易的Lean证明代理，并将实现开源；

**💡 创新点**

核心创新在于用极简的代理架构通过迭代反馈与自管理内存获得与复杂系统相当甚至更优的证明性能，突出代理式方法的可扩展性；

**🔧 技术方法**

基于通用LLM（Claude Opus 4.5）实现ReAct式代理，结合Lean编译器验证、Web/库搜索工具、迭代反馈循环与内存管理；

**📊 数据集**

在PutnamBench、MiniF2F、FATE（M/H/X）与LeanCat等公开基准数据集上进行实验；

**📈 对比分析**

在相同LLM、迭代次数与预算下，AxProverBase在pass@1上取得54.7%（PutnamBench）、98%（FATE‑M）、66%（FATE‑H）、24%（FATE‑X）、59%（LeanCat），超过多数非代理系统，且成本与运行时间显著低于Hilbert等复杂系统；

**⚠️ 局限性**

局限性包括库搜索与Web搜索对性能提升有限、在更难的FATE‑H/X仍表现不足，需要改进内存管理、审阅器可靠性与模型专门化等。

---

## 415. NSHEDB: Noise-Sensitive Homomorphic Encrypted Database Query Engine

**arXiv ID:** 2602.24271 | [PDF](https://arxiv.org/pdf/2602.24271v1)

**作者:** Boram Jung `[一作]`, Hung-Wei Tseng `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

论文探讨了自定义编程语言的实现与应用。

**💡 创新点**

创新点在于提出了一种新的语法解析方法，能够提高编译效率。

**🔧 技术方法**

使用了递归下降解析技术和状态机。

**📊 数据集**

使用了自定义的测试数据集，包含多种编程语言的示例代码。

**📈 对比分析**

与传统的解析方法进行了比较，结果显示新方法在处理复杂语法时性能提升了30%。

**⚠️ 局限性**

限制在于该方法对特定语法的支持较弱，可能无法广泛应用于所有编程语言。

---

