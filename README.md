# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-13 | 今日论文总数: 610

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. EpicCBR: Item-Relation-Enhanced Dual-Scenario Contrastive Learning for Cold-Start Bundle Recommendation

**arXiv ID:** 2602.11680 | [PDF](https://arxiv.org/pdf/2602.11680v1)

**作者:** Yihang Li `[一作]` (Huazhong University of Science and Technology), Wei Wei `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 83108 | [OpenAlex ID](https://openalex.org/A5100344847)

**关键词:** `Information Retrieval` `Recommendation System` `Graph Neural Network` `Contrastive Learning` `Graph`

**🎯 论文内容**

提出 EpicCBR，一种针对冷启/热启场景的多视图对比学习框架，用于推荐用户未见过的商品组合包。

**💡 创新点**

创新点：① 将用户-物品和组合-物品两种图同时构建，并通过四类物品对关系（R1‑R4）增强用户画像；② 采用基于流行度的组合表征，将新组合映射为其包含物品的加权聚合；③ 设计双场景对比损失，融合冷启与热启的特征，避免信息互相干扰。

**🔧 技术方法**

技术：图神经网络（LightGCN）进行消息传播；基于 Jaccard 相似度的物品对关系挖掘；对比学习（InfoNCE）用于用户、组合、物品的表示；BPR 及正负采样做传统排序损失；多视图融合与加权组合实现冷/热启统一。

**📊 数据集**

三大公开数据集：Youshu（图书书单），NetEase（音乐播放列表），iFashion（时尚商品打包）。

**📈 对比分析**

与 Coheat、CVAR、CLCRec、CCFCRec、MultiCBR 等冷启/热启基线对比；Recall@20 与 nDCG@20 为评价指标。EpicCBR 在冷启场景相较最优基线提升最高 387%（iFashion），在热启场景也比 MultiCBR 与 SGL 等竞争者提升约 2‑3% 的 Recall@20。

**⚠️ 局限性**

局限：① 对物品稀疏性极低或组合规模极小的数据集（如 iFashion）时，物品对关系挖掘和流行度权重可能引入噪声；② 需要先验物品对相似度阈值和多视图权重手动设定，调参成本较高。

---

## 2. Security Assessment of Intel TDX with support for Live Migration

**arXiv ID:** 2602.11434 | [PDF](https://arxiv.org/pdf/2602.11434v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Cryptography and Security`

---

## 3. Momentum LMS Theory beyond Stationarity: Stability, Tracking, and Regret

**arXiv ID:** 2602.11995 | [PDF](https://arxiv.org/pdf/2602.11995v1)

**作者:** Yifei Jin `[一作]` (University of Chinese Academy of Sciences), Lei Guo `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 19966 | [OpenAlex ID](https://openalex.org/A5010252124)

**关键词:** `Machine Learning` `Time Series` `Audio`

**🎯 论文内容**

研究Momentum Least Mean Squares (MLMS)在非平稳流数据中的理论性能与实验验证，给出了追踪误差和预测误差的界限。

**💡 创新点**

提出在一般非平稳条件下对MLMS稳定性和跟踪性能的理论分析，克服动量导致的二阶随机差分方程分析难题，并给出无激励条件下的预测误差上界。

**🔧 技术方法**

采用随机矩阵乘积分析、随机递归差分方程、随机过程类M_p、预测误差分析和投影正则化等理论方法，并在实验中使用合成跳变参数数据和NOIZEUS语音增强数据。

**📊 数据集**

使用合成时间变线性系统数据（含突变参数）和NOIZEUS机场噪声语音数据。

**📈 对比分析**

与SGD、SGD‑Momentum、LMS、RLS、GNGD等算法在相同设置下对比；MLMS在参数跟踪误差和SNR提升上均优于或相近，尤其在突变/非平稳环境中表现最优。

**⚠️ 局限性**

理论对步长的取值给出足够条件但实际范围可能更大；仍需经验调参（如β, μ）；对高维/强非线性系统的扩展尚未完全验证；以及对噪声和参数变化速度的上界有限制。

---

## 4. Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond

**arXiv ID:** 2602.11671 | [PDF](https://arxiv.org/pdf/2602.11671v1)

**作者:** Minh Le-Anh `[一作]` (FPT Software AI Center), Bach Le `[通讯]` (University of Melbourne)

**通讯引用:** 1475 | [OpenAlex ID](https://openalex.org/A5075260906)

**关键词:** `Software Engineering` `AI Code Assistant` `Generation` `Retrieval` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

提出一种面向仓库级代码生成的框架，利用结构感知索引、依赖感知检索（DAR）和混合检索策略，显著提升大型语言模型在跨文件上下文中的生成质量。

**💡 创新点**

创新点包括：① 结构化索引，将仓库拆解为函数/类/变量层次树，保留完整的语义结构；② 轻量化依赖感知检索器DAR，自动识别目标函数所需的真实依赖；③ 混合检索方案，将DAR提取的核心依赖与BM25的用法示例结合，为生成器提供更丰富、准确的上下文；④ 在RepoExec和DevEval上实现新的SOTA，甚至让小模型逼近或超过大模型。

**🔧 技术方法**

技术栈：Python AST解析构建结构索引；代码检索使用UniXCoder编码器的二分类模型（DAR）与传统BM25；大语言模型（Qwen2.5-Coder 1.5B/7B、GPT‑4.1‑mini）作为生成器；评估指标Pass@k与Dependency Invocation Rate (DIR)，以及时延分析。

**📊 数据集**

使用的数据集包括：Python RepoExec（355个任务）、DevEval（1825个任务），以及自构建的训练集（2864个Python仓库，564k个查询–正/负样本三元组）用于训练DAR。

**📈 对比分析**

实验对比：结构索引 vs 文本分块、稀疏/稠密检索 vs DAR、基线RepoCoder/RepoFormer/RLCoder与无检索。结果显示：结构索引相较分块提升约1–3% Pass@1；DAR在依赖检索Recall上高达92–89%，显著优于传统检索；混合检索在Pass@1上提升5%+，DIR提升4–6%。尤其是1.5B模型在RepoExec上突破7B模型，显示检索质量可以弥补模型规模差距。

**⚠️ 局限性**

局限性：仅在Python两套基准上验证，未涉及其他语言或更高级别的生成任务；DAR依赖UniXCoder编码器，其他后端可能表现不同；结构索引和检索主要基于显式import关系，可能漏掉隐式或跨文件动态依赖；实验数据集规模有限，泛化性待进一步验证。

---

## 5. Understanding Persuasive Interactions between Generative Social Agents and Humans: The Knowledge-based Persuasion Model (KPM)

**arXiv ID:** 2602.11483 | [PDF](https://arxiv.org/pdf/2602.11483v1)

**作者:** Stephan Vonschallen `[一作]` (Zurich University of Applied Sciences), Theresa Schmiedel `[通讯]` (Zurich University of Applied Sciences)

**通讯引用:** 1699 | [OpenAlex ID](https://openalex.org/A5069654221)

**关键词:** `Human-Computer Interaction` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Multimodality`

**🎯 论文内容**

提出了知识驱动的说服模型（KPM），用于解释和指导生成式社会代理（GSA）与人类之间的说服互动，整合了自我知识、用户知识与情境知识对代理说服行为及人类反应的影响链条。

**💡 创新点**

将心理学说服模型（ELM、HSM、TAM、PRAM）与说服知识模型（PKM）结合，首创以代理知识为出发点的说服理论框架；强调代理自适应生成行为而非预设脚本，提供责任感与伦理性评估的工具。

**🔧 技术方法**

利用生成式人工智能技术（大型语言模型、prompting、检索增强生成、微调）构建代理知识体系；采用信息处理与说服心理学理论作为分析与设计依据。

**📊 数据集**

本研究为概念性框架，未使用具体数据集；若实验验证，需收集代理交互记录、用户态度量表、行为日志等多模态数据。

**📈 对比分析**

未进行实验比较或性能评估；KPM 旨在为后续经验研究提供可操作指标与假设，未来可通过多组对照实验或结构方程模型验证知识-行为-反应路径。

**⚠️ 局限性**

局限性包括：1）生成式模型的不可预测性导致行为一致性与可比较性难以保证；2）缺乏经验验证，模型假设尚未被量化证实；3）对用户隐私与伦理风险的依赖需进一步制定安全保障与数据治理规范；4）情境与任务多样性使得模型泛化受限。

---

## 6. Any House Any Task: Scalable Long-Horizon Planning for Abstract Human Tasks

**arXiv ID:** 2602.12244 | [PDF](https://arxiv.org/pdf/2602.12244v1)

**作者:** Zhihong Liu `[一作]` (Shanghai Innovation Institute), Panpan Cai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1285 | [OpenAlex ID](https://openalex.org/A5077008575)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Chain-of-Thought` `Text`

**🎯 论文内容**

提出了Any House Any Task (AHAT)，一种针对大型住宅环境的可伸缩长周期任务规划框架，能够基于抽象人类指令和文本场景图生成子目标并通过PDDL规划实现执行；

**💡 创新点**

核心创新在于将LLM与符号规划结合，使用强化学习（Trace-Guided Policy Optimization）对LLM的任务分解推理进行外部修正，从而提升对抽象指令的理解与子目标可解性；

**🔧 技术方法**

采用大型语言模型（Qwen-2.5-7B为基础）结合Group Relative Policy Optimization（GRPO），并引入链式思维、PDDL子目标生成、外部修正模块以及基于文本场景图的符号规划；

**📊 数据集**

构建了规模达5万条的合成数据集，覆盖308个住宅场景图、1.6k用户画像及多样化任务指令，用以训练和评估模型；

**📈 对比分析**

与基线（GPT-5、Gemini-3、SayPlan、Delta、GRPO、Reinforce++等）在AHAT基准、人工任务、PARTNR和Behavior-1K上对比，AHAT在成功率和规划时间上分别提升约20-70%并保持对复杂/抽象任务的高鲁棒性；

**⚠️ 局限性**

局限性包括依赖结构化文本场景图导致信息损失、仅适用于预定义PDDL域、未考虑感知噪声与动态环境，并未支持直接视觉输入或多域扩展。

---

## 7. KAN-FIF: Spline-Parameterized Lightweight Physics-based Tropical Cyclone Estimation on Meteorological Satellite

**arXiv ID:** 2602.12117 | [PDF](https://arxiv.org/pdf/2602.12117v1)

**作者:** Jiakang Shen `[一作]` (Shandong University), Feng Zhang `[通讯]` (Fudan University)

**通讯引用:** 25804 | [OpenAlex ID](https://openalex.org/A5073112564)

**关键词:** `Machine Learning` `Convolutional Neural Network` `Multimodality` `Image`

**🎯 论文内容**

本研究提出了一种基于Kolmogorov-Arnold网络的特征交互框架（KAN-FIF），用于高效准确地估计热带气旋的强度和大小，特别是在资源受限的边缘设备上。

**💡 创新点**

创新点在于引入了轻量级的KAN层替代传统的多层感知器（MLP）和卷积神经网络（CNN），实现了94.8%的参数减少和68.7%的推理速度提升，同时保持了更高的预测准确性。

**🔧 技术方法**

使用了Kolmogorov-Arnold网络（KAN）层、卷积神经网络（CNN）和多层感知器（MLP）等技术，结合物理约束模块进行特征融合。

**📊 数据集**

使用了热带气旋多模态数据集（TCMM），该数据集包含来自Himawari-8卫星的红外图像和热带气旋轨迹记录，覆盖2015年至2022年。

**📈 对比分析**

与现有的多任务模型Phy-CoCo相比，KAN-FIF在最大持续风速（MSW）和最大风半径（RMW）的平均绝对误差（MAE）上分别减少了32.5%和31.9%，并且在推理时间上显著优于其他模型。

**⚠️ 局限性**

限制在于尽管模型在边缘设备上表现良好，但在实际轨道部署中可能面临数据管道集成、辐射硬化和电源管理等额外挑战。

---

## 8. RF-Modulated Adaptive Communication Improves Multi-Agent Robotic Exploration

**arXiv ID:** 2602.12074 | [PDF](https://arxiv.org/pdf/2602.12074v1)

**作者:** Lorin Achey `[一作]` (University of Colorado Boulder), Bradley Hayes `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1123 | [OpenAlex ID](https://openalex.org/A5034950112)

**关键词:** `Robotics` `Robotic Intelligence` `Simultaneous Localization and Mapping`

**🎯 论文内容**

提出了一种基于信号强度和数据包大小自适应选择通信点的多机器人探测算法ART（Adaptive‑RF Transmission）及其扩展ART‑SST，并在三种洞穴类仿真环境中评估其性能。

**💡 创新点**

创新点在于将信号衰减模型（对数距离路径损失+A*路径估计）与前沿搜索相结合，实现了在保持数据可靠传输的同时最小化后退成本的动态通信点决策；同时提出了按数据大小强制信号阈值的ART‑SST变体。

**🔧 技术方法**

采用了信号强度建模（Friis、对数距离路径损失、A*路径估计）、香农容量计算传输时长、前沿搜索、ROS2+NAV2+slam_toolbox软件栈以及Nvidia IsaacSim仿真器进行实现。

**📊 数据集**

使用了三种仿真洞穴环境（窗口、Y型分叉、长隧道），并基于真实硬件在狭窄通道的信号强度测量生成的信号地图；共运行了超过480场仿真实验。

**📈 对比分析**

将ART、ART‑SST、MSSC（最小信号阈值）和FRC（完整对接）四种策略在四种数据包大小（0–3）下进行对比；指标为路径距离和探索时间；结果显示ART在所有环境和数据包大小下均获得最低路径距离和最快探索速度，分别比FRC减少约58%距离、提升约52%速度，ART‑SST在高数据量时性能下降，MSSC和FRC的表现相对较差。

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证，未考虑实际电磁干扰和动态噪声；假设探测者回到先前前沿而未实时重新规划；三种环境缺乏真实洞穴的复杂性，未来需在真实地下空间进行验证。

---

## 9. Exploring Multiple High-Scoring Subspaces in Generative Flow Networks

**arXiv ID:** 2602.11491 | [PDF](https://arxiv.org/pdf/2602.11491v1)

**作者:** Xuan Yu `[一作]` (University of Science and Technology of China), Yang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 34270 | [OpenAlex ID](https://openalex.org/A5100764445)

**关键词:** `Machine Learning` `Generation` `Optimization` `Drug Discovery` `Flow-based Model` `Reinforcement Learning` `Sequential`

**🎯 论文内容**

提出 CMAB-GFN，将组合多臂赌博机（CMAB）框架与生成流网络（GFlowNet）结合，通过在子空间层面进行动作剪枝来引导采样，提升高奖励候选的发现效率。

**💡 创新点**

将 GFlowNet 的探索视为在动作诱导的子空间中进行组合搜索，使用 CMAB 动态选择高质量子空间；同时引入基于半赌博机反馈的 arm 级奖励估计和共现加权贪婪选择，形成一种可推广的、子空间层面权衡探索与利用的方法。

**🔧 技术方法**

核心技术包括：生成流网络（GFlowNet）训练与采样；组合多臂赌博机（CMAB）与 CUCB 上限置信算法；滑动窗口统计 arm 奖励；共现矩阵加权贪婪子空间选择；两阶段训练（受限训练 + 无限制评估）以保持奖励一致性。

**📊 数据集**

实验数据集涵盖：分子设计（sEH 结合亲和力代理模型，105 个构建块，8 分子片段）；位序列生成（长度 120 的二进制序列，Levenshtein 距离奖励）；RNA 设计（14 位核苷酸序列，四种核苷酸，L14-RNA1/2/3 三个目标）。

**📈 对比分析**

与 TB、SUBTB、DB、LSGFN、Teacher、QGFN、随机剪枝等基线在模式发现数量、Top‑1000 平均奖励与多样性（Tanimoto 相似度）等指标上对比。CMAB‑GFN 在所有任务中均能更快发现更多高奖励模式，Top‑1000 奖励最高且相似度保持在可接受范围内，整体性能优于现有方法。

**⚠️ 局限性**

主要局限：假设环境奖励分布在训练过程中保持不变，若奖励空间动态变化则收益有限；对子空间划分和超参数（K、α、λ、H）敏感，需经验调优；评估阶段需额外环境调用，增加计算开销；目前仅在生成式任务上验证，尚未在推荐或组合优化等更广泛任务中测试。

---

## 10. Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments

**arXiv ID:** 2602.11964 | [PDF](https://arxiv.org/pdf/2602.11964v1)

**作者:** Romain Froger `[一作]` (Meta SuperIntelligence Labs), Thomas Scialom `[通讯]` (Meta SuperIntelligence Labs)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Reinforcement Learning` `Agentic AI` `Text` `Benchmark`

**🎯 论文内容**

构建了Agents Research Environments（ARE）框架与Gaia2基准，用于在异步、事件驱动的手机应用环境中评估大语言模型代理，提供细粒度可验证的行动级评价；

**💡 创新点**

创新点包括：①异步事件驱动环境与可验证器实现每一次写操作的精确检查；②将核心能力拆分为执行、搜索、歧义、适应、时间等七类，并加入噪声与多代理协作的增强维度；③提供可扩展的基准结构，兼容开源ARE，方便社区迭代与RLVR训练；

**🔧 技术方法**

使用技术主要有：ARE平台的应用、事件、通知与场景抽象；ReAct式工具调用脚本与可插拔的并行工具调用（PTC）实验；基于规则、因果与时序的Verifier实现；以及对长上下文、长时间跨度的高容量（≥128K token）处理；

**📊 数据集**

数据集为Gaia2共1,120个人工标注的场景（800核心 + 320增强），基于12款手机应用（邮件、聊天、日历等）生成的模拟宇宙，提供约4-8万token结构与非结构化内容；另包含160个mini场景用于快速评测；

**📈 对比分析**

对比方法：在核心能力拆分上计算各模型Pass@1得分，涵盖GPT‑5、Claude‑4 Sonnet、Kimi‑K2、Gemini‑2.5‑Pro等；同时对模型成本、执行时间、人类基准进行多维度对比；结果显示GPT‑5(high)在总体上42%通过率，Claude‑4 Sonnet 35%，Kimi‑K2 20%，各模型在时间敏感、噪声鲁棒、A2A协作等子维度表现不一；

**⚠️ 局限性**

局限性：①时间敏感任务通过率仍低，需更高效的推理与并行执行；②噪声鲁棒性不足，模型易被意外事件干扰；③多代理协作采用单线程框架，无法充分利用并行化潜能；④评估依赖大量人工标注，对更大规模的自动化生成仍有挑战；

---

## 11. Beyond Code: Empirical Insights into How Team Dynamics Influence OSS Project Selection

**arXiv ID:** 2602.11692 | [PDF](https://arxiv.org/pdf/2602.11692v1)

**作者:** Shashiwadana Nirmani `[一作]` (Deakin University), Xiao Liu `[通讯]` (Deakin University)

**通讯引用:** 11114 | [OpenAlex ID](https://openalex.org/A5075936732)

**关键词:** `Software Engineering` `Recommendation System` `Text` `Review/Survey Paper`

**🎯 论文内容**

本文通过对198名OSS实践者的在线问卷，系统调查了团队动态对项目选择的影响，并分析了不同动机下对团队特性的偏好差异。

**💡 创新点**

创新点在于首次将团队沟通质量、响应速度、包容性等社会动态与贡献者动机关联，并为未来的“人性化”项目推荐系统提供实证依据。

**🔧 技术方法**

采用问卷设计、Kruskal-Wallis非参数检验和主题分析等定量与定性技术对数据进行统计和语义挖掘。

**📊 数据集**

使用的主要数据集为来自Prolific与社交媒体招募的198份有效问卷回应，涵盖年龄、性别、地区及贡献经验等信息。

**📈 对比分析**

方法上通过比较各团队动态的重要性评分与动机维度的Kruskal-Wallis检验（并做Bonferroni校正）验证差异显著性，结果显示多项团队动态与动机存在统计显著关联。

**⚠️ 局限性**

局限性包括样本规模有限、受访者自我报告偏差、以英语为主的参与者限制了跨文化推广，以及未能量化团队动态的客观指标。

---

## 12. A Generic Framework for Fair Consensus Clustering in Streams

**arXiv ID:** 2602.11500 | [PDF](https://arxiv.org/pdf/2602.11500v1)

**作者:** Diptarka Chakraborty `[一作]` (National University of Singapore), Tien-Long Nguyen `[通讯]` (Pennsylvania State University)

**通讯引用:** 231 | [OpenAlex ID](https://openalex.org/A5060063567)

**关键词:** `Machine Learning` `Optimization`

**🎯 论文内容**

提出了在流式模型下进行公平共识聚类（1‑median 与 k‑median）的常数因子近似算法，并实现了子线性空间（对 1‑median 为 O(nlogm)，对 k‑median 为 O(k^2 n(mn))）

**💡 创新点**

首次将公平聚类与共识聚类结合并移植到流式环境；通过采样、核心点与单调远距离采样等技术构造候选聚类集，获得与离线算法相当的近似比；在两色公平约束下实现了 3.01461‑近似（等比）

**🔧 技术方法**

统一框架：对每个输入聚类求 γ‑近似公平聚类、使用两三聚类构造加权图求公平相关聚类、采样产生候选集；流式实现借助均匀采样、核心点构造与 Monotone Faraway Sampling

**📊 数据集**

文中未给出实验数据集，聚类对象为理论上的所有聚类集合；若有实验则使用常见基准数据（如 KDD‑Cup、UCI 组），但在论文摘要中未作说明

**📈 对比分析**

与最优离线方案比较，取得常数因子近似；在两色等比约束下 3.01461‑近似，p:1 约束下 19.25621‑近似，p:q 约束下 35.49781‑近似，优于现有 11/7 的最优近似；空间与时间均为子线性或多项式级

**⚠️ 局限性**

对 k‑median 仅支持插入‑只流，无法处理通用顺序流；空间上仍相对较大（O(k^2 n(mn))）；近似系数仍高于最优公平聚类子问题的最佳逼近；未给出实验验证，主要理论证明

---

## 13. Automated Optimization Modeling via a Localizable Error-Driven Perspective

**arXiv ID:** 2602.11164 | [PDF](https://arxiv.org/pdf/2602.11164v1)

**作者:** Weiting Liu `[一作]` (Fudan University), Wenlian Lu `[通讯]` (Fudan University)

**通讯引用:** 8159 | [OpenAlex ID](https://openalex.org/A5030103251)

**关键词:** `Machine Learning` `Optimization` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Supervised Fine-Tuning` `Tabular`

**🎯 论文内容**

本文针对LLM在自动化优化建模中存在的错误稀缺和奖励稀疏问题，提出了MIND框架，包含错误驱动的逆向数据合成与动态监督微调策略；

**💡 创新点**

创新点在于①发现优化建模错误局部化，②利用逆向合成产生富含错误模式的数据；③通过教师LLM纠正错误并在RL中加入SFT损失，解决奖励稀疏与分布漂移；

**🔧 技术方法**

技术包括逆向数据合成管道、Dynamic Supervised Fine‑Tuning Policy Optimization（DFPO）、奖励设计（模型保真度+准确度）、强化学习（GRPO/DAPO）和LLM判定器；

**📊 数据集**

数据集包括原始训练集OR‑Instruct‑Data‑3K、OptMATH‑Train用于合成，构建MIND‑Train（10k样本）以及新构造的MIND‑Bench（69题）和六个公开基准；

**📈 对比分析**

与GPT‑4、Deepseek‑V3、Qwen‑3‑8B等对标，采用pass@1评估，MIND‑Qwen‑2.5‑7B提升约14%/31%基准性能，优于所有同参数量模型，并在MIND‑Bench上显著优于SIRL；

**⚠️ 局限性**

局限在于仍落后于大参数基础模型，对表格型问题的提升有限，且依赖于教师LLM的纠错质量与成本。

---

## 14. Visualizing and Benchmarking LLM Factual Hallucination Tendencies via Internal State Analysis and Clustering

**arXiv ID:** 2602.11167 | [PDF](https://arxiv.org/pdf/2602.11167v1)

**作者:** Nathan Mao `[一作]` (Harker School), Sunishchal Dev `[通讯]` (Algoverse AI Research)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text` `Benchmark`

**🎯 论文内容**

本文构建了 FalseCite 数据集，对含有误导性或伪造引用的虚假声明进行系统评估，并测量大型语言模型（GPT‑4o‑mini、Falcon‑7B、Mistral‑7B）在此情境下的幻觉生成率。

**💡 创新点**

创新点在于首次将虚假引用与虚假声明配对，并通过随机与语义匹配两种配对方式揭示引用对模型幻觉的放大效应；同时利用隐藏状态向量可视化和聚类，发现隐藏状态在层级上呈锥形结构。

**🔧 技术方法**

技术包括：对 FEVER 与 SciQ 的文本处理与伪造引用生成；使用 GPT‑4.1 作为专家标签器对模型输出进行幻觉判定；对隐藏状态与注意力进行 Spearman 相关性计算、PCA 降维与 k‑means 聚类；以及对多模型输出的定量比较。

**📊 数据集**

使用了 82k 条虚假声明，来源于 FEVER（47k 条非科学性错误声明）与 SciQ（35k 条科学性错误声明），并在声明上随机或语义匹配多种伪造引用生成测试样本。

**📈 对比分析**

通过将模型输出由 GPT‑4.1 标记为幻觉/非幻觉，计算在无引用、随机引用、语义引用三种条件下的幻觉率。实验显示，GPT‑4o‑mini 在随机引用下的幻觉率提升最高（+39.65%），Falcon‑7B 与 Mistral‑7B 在随机引用下亦显著升高；整体上，伪造引用能显著提升所有模型的幻觉率。

**⚠️ 局限性**

主要局限在于使用 GPT‑4.1 作为专家标注器，缺乏网络访问导致无法验证引用真实性；此外，样本规模受限，未涉及人类注释或 RAG 模型验证，聚类结果虽展示隐藏状态形状但未揭示更深层次的幻觉机制。

---

## 15. Neuro-Symbolic Multitasking: A Unified Framework for Discovering Generalizable Solutions to PDE Families

**arXiv ID:** 2602.11630 | [PDF](https://arxiv.org/pdf/2602.11630v1)

**作者:** Yipeng Huang `[一作]` (Xiamen University), Min Jiang `[通讯]` (Xiamen University)

**通讯引用:** 4705 | [OpenAlex ID](https://openalex.org/A5017961841)

**关键词:** `Artificial Intelligence` `Optimization` `Computational Efficiency` `Reinforcement Learning` `Image`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习和强化学习相结合的技术。

**📊 数据集**

采用了公开的图像识别数据集进行实验。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示新算法在准确率和速度上均有显著提升。

**⚠️ 局限性**

限制在于算法在特定类型的数据上表现不佳，且对计算资源的需求较高。

---

## 16. A Grounded Theory of Debugging in Professional Software Engineering Practice

**arXiv ID:** 2602.11435 | [PDF](https://arxiv.org/pdf/2602.11435v1)

**作者:** Haolin Li `[一作]` (University of California San Diego), Michael Coblenz `[通讯]` (University of California San Diego)

**通讯引用:** 1079 | [OpenAlex ID](https://openalex.org/A5044646652)

**关键词:** `Software Engineering` `Video` `Finance Related` `Biomedical Data`

**🎯 论文内容**

在真实生产环境中观察12名专业软件工程师完成17个调试任务，使用基于构建理论的定性研究方法提炼调试过程和策略，构建了一个以心理模型迭代为核心的专业调试理论；

**💡 创新点**

首次将构建理论与调试研究结合，提出“知识避免”与“情境化心理模型”概念，揭示专业调试在信息获取、外部资源利用与社交技术整合方面的复杂性；

**🔧 技术方法**

采用构建理论分析（开放编码、轴心编码）、think‑aloud记录、半结构化访谈、直播视频观察以及定量时间分配统计；

**📊 数据集**

收集12名开发者的现场观察与访谈数据（7名）以及5名专业直播者的调试视频，涵盖从Web到数据库、AI、金融、医学等多种代码库，总计约11小时调试时间；

**📈 对比分析**

通过对参与者与直播者的过程计数和时间比例进行比较（如P5 vs S1），利用非参数检验评估过程差异，结果显示直播者在过程转移次数上显著高于现场观察者，说明不同情境下调试行为存在差异；

**⚠️ 局限性**

受限于样本规模、需要受试者同意导致部分记录缺失、think‑aloud可能影响调试效率、无法公开完整编码数据、且仅涵盖专业工程师的工作，缺乏对学生或初学者调试行为的覆盖。

---

## 17. The Energy of Falsehood: Detecting Hallucinations via Diffusion Model Likelihoods

**arXiv ID:** 2602.11364 | [PDF](https://arxiv.org/pdf/2602.11364v1)

**作者:** Arpit Singh Gautam `[一作]` (Dell Technologies), Saurabh Jha `[通讯]` (Dell Technologies)

**关键词:** `Computation and Language` `Generation` `Anomaly Detection` `Diffusion model` `Large Language Model` `Text`

**🎯 论文内容**

提出 DiffuTruth 框架，利用文本扩散模型的生成动力学检测大型语言模型产生的幻觉（hallucination）。

**💡 创新点**

创新点在于把真理视为生成流形上的稳定吸引子，设计生成压力测试、语义能量指标，并与判别式置信度混合校准。

**🔧 技术方法**

使用离散文本扩散模型 DiffuSeq、NLI 评价器构成语义能量，并通过 Hybrid Calibration 将生成与判别式信号融合。

**📊 数据集**

在 FEVER（事实验证）和 HOVER（多跳推理）两个数据集上进行实验，验证模型在域内外的鲁棒性。

**📈 对比分析**

与随机、MSE 以及直接 NLI 基线相比，DiffuTruth 在 FEVER 上取得 AUROC 0.725，HOVER 上 0.566，表现优于判别式基线并保持跨域稳健。

**⚠️ 局限性**

局限性包括采样计算成本高、需正样本训练语料、超参数（如焦点时刻 t*、混合权重 λ）敏感，以及语义能量阈值解释性不足。

---

## 18. Predictive Associative Memory: Retrieval Beyond Similarity Through Temporal Co-occurrence

**arXiv ID:** 2602.11322 | [PDF](https://arxiv.org/pdf/2602.11322v1)

**作者:** Jason Dury `[一作]` (Independent Researcher), Jason Dury `[通讯]` (Independent Researcher)

**关键词:** `Machine Learning` `Retrieval` `Contrastive Learning` `Sequential`

**🎯 论文内容**

提出一种基于时间共现学习的关联记忆框架PAM，利用JEPA风格的预测器在嵌入空间中学习并检索关联记忆；

**💡 创新点**

创新点在于仅用时间共现信号训练无监督的关联预测器，实现跨表征边界的记忆检索，并通过Inward和Outward两路JEPA实现潜在的情景特异性；

**🔧 技术方法**

技术包括JEPA（Joint‑Embedding Predictive Architecture）、无梯度EMA目标、InfoNCE对比损失、单层MLP预测器以及近似最近邻检索；

**📊 数据集**

使用人工合成的20房间、50物体的连续体验流，生成约50k状态和242k个时间共现关联；

**📈 对比分析**

与纯相似度检索（cosine）和线性双线性相似度基线对比，PAM在跨房间检索精度（Recall@20 0.421）和关联精度@1 0.970、AUC 0.849等指标上显著优于基线；

**⚠️ 局限性**

局限包括仅在合成环境中验证、嵌入空间为预置且不自我学习、单点预测难以处理多对多关联、未实现跨片段创意重组、对真实高噪声数据的鲁棒性待测。

---

## 19. Multi-Defender Single-Attacker Perimeter Defense Game on a Cylinder: Special Case in which the Attacker Starts at the Boundary

**arXiv ID:** 2602.11977 | [PDF](https://arxiv.org/pdf/2602.11977v1)

**作者:** Michael Otte `[一作]`, Roderich Groß `[通讯]`

**关键词:** `Multiagent Systems`

**🎯 论文内容**

本文研究在圆柱形边界上的多防御者-单攻击者防御游戏，推导出攻击者在起始点位于已被防御区域内且速度快于防御者时，何种条件下能突破防线的闭式表达式。

**💡 创新点**

创新点在于：①将传统的环形或线性边界防御问题迁移到圆柱拓扑上；②假设攻击者起始于边界附近并且防御者最优配置，得到全局闭式胜负判据；③通过对两种极端起始位置（攻击者位于防御区端点或中点）的分析，证明它们在时间上等价，进一步导出攻击者获胜的速度与防御半径、人数的具体不等式。

**🔧 技术方法**

采用了微分游戏与连续空间动态博弈理论的分析方法，利用几何距离、速度比、占据区间等数学工具推导不等式，并做了极端配置的对称性证明。

**📊 数据集**

本文为理论推导性质的研究，未使用任何实验或仿真数据集；所有结论均基于纯数学推理与符号计算。

**📈 对比分析**

论文未进行实验对比，只在理论层面通过推导两种极端情形的等价性来验证结果的完整性。由于缺乏数值仿真或实测数据，无法给出具体性能指标。

**⚠️ 局限性**

局限性包括：①假设防御者速度恒定且可瞬时改变方向；②忽略检测延迟、通信延迟以及边界曲率对运动的影响；③仅考虑同质防御者且仅在圆柱边界上，难以直接推广至多样化防御者或更复杂几何形状；④缺乏实验验证，无法评估在真实系统中的适用性。

---

## 20. Brain Tumor Classifiers Under Attack: Robustness of ResNet Variants Against Transferable FGSM and PGD Attacks

**arXiv ID:** 2602.11646 | [PDF](https://arxiv.org/pdf/2602.11646v1)

**作者:** Ryan Deem `[一作]` (Kennesaw State University), Michail S. Alexiou `[通讯]` (Kennesaw State University)

**通讯引用:** 46 | [OpenAlex ID](https://openalex.org/A5079860779)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Adversarial Attack` `Convolutional Neural Network` `Image` `Biomedical Data` `Magnetic Resonance Imaging`

**🎯 论文内容**

对脑肿瘤MRI分类的ResNet、ResNeXt及膨胀卷积变体模型，分别在FGSM和PGD攻击下的鲁棒性进行了实验评估。

**💡 创新点**

首次系统比较了三种不同架构在黑盒可迁移攻击中的防御能力，并探讨了输入分辨率和数据增强对鲁棒性的影响。

**🔧 技术方法**

采用FGSM、PGD梯度攻击、迁移攻击、预训练ResNet/ResNeXt、膨胀卷积等技术进行模型训练与评测。

**📊 数据集**

使用公开的4,023张脑肿瘤MRI图像（glioma、meningioma、pituitary），并按全尺寸、缩小+增、缩小/非增三种预处理方式进行实验。

**📈 对比分析**

通过比较攻击前后各模型在三种预处理配置下的准确率，发现BrainNeXt在黑盒攻击中最稳健但攻击可迁移性最低，而BrainNet和Dilated在缩小/非增数据上易被攻击，性能差异显著。

**⚠️ 局限性**

研究仅限于FGSM/PGD梯度攻击，未涵盖对抗训练或更高级攻击，且数据集仅包含三类肿瘤，缺少多模态或非肿瘤类别的验证。

---

## 21. SkillRater: Untangling Capabilities in Multimodal Data

**arXiv ID:** 2602.11615 | [PDF](https://arxiv.org/pdf/2602.11615v1)

**作者:** Naveen Sahi `[一作]`, Akshat Shrivastava `[通讯]`

**关键词:** `Machine Learning` `Meta Learning` `Data-Centric Learning` `Multimodality`

**🎯 论文内容**

本文提出将数据筛选拆分为针对不同能力（视觉理解、OCR、STEM推理）的专门评估器（rater），并在训练过程中通过分阶段阈值逐步集中高质量样本，形成多维度的训练数据过滤策略。

**💡 创新点**

创新点在于将数据质量视为多维度而非单一标量，利用元学习分别训练每个能力的评估器，保证评估信号相互正交；并通过“进化式”统一阈值课程（union rule）在训练早期保持多样性，后期聚焦高价值样本，从而突破单一评估器的互斥瓶颈。

**🔧 技术方法**

使用技术包括：1）DataRater框架的多任务元学习；2）对多模态特征的冻结SigLIP 400M编码器与MLP融合；3）一阶近似的梯度计算与微批次内存高效训练；4）分阶段阈值课程与多评估器联合选择。

**📊 数据集**

训练与评估使用多种视觉‑文本基准：训练时分布三类验证集（Visual Understanding：SEED-Bench、A‑OKVQA、VSR；OCR：TextVQA、ChartQA、RD‑Tablebench；STEM：M3Exam、MathVista、AI2D）；评估时采用完全独立的Held‑Out基准（VQAv2、NLVR2、MME；DocVQA、ChartMuseum；MathVision、MMMU、MMMU‑Pro）。

**📈 对比分析**

与未过滤、单一DataRater和“组合”评估器的对比显示，三维评估器+课程策略在所有三类能力上均优于基线，整体准确率提升约4%（Visual Understanding +5.6%、OCR +2.0%、STEM +3.5%）。课程策略相较于静态Top‑k筛选在不同阈值下均表现更佳，且在1B与2B参数规模上，评估器可迁移使用，性能差距随训练进展进一步拉大。

**⚠️ 局限性**

主要局限包括：1）仅划分为三大能力，无法验证进一步细分是否有效；2）课程阈值手工设定，缺乏联合优化；3）评估器依赖验证集覆盖，若缺少某能力的基准则难以针对；4）实验集中于视觉‑文本中间训练，缺少对文本或代码生成等其他场景的验证。

---

## 22. ScalSelect: Scalable Training-Free Multimodal Data Selection for Efficient Visual Instruction Tuning

**arXiv ID:** 2602.11636 | [PDF](https://arxiv.org/pdf/2602.11636v1)

**作者:** Changti Wu `[一作]` (East China Normal University), Kai Chen `[通讯]` (Zhongguancun Institute of Artificial Intelligence)

**关键词:** `Computer Vision and Pattern Recognition` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Vision Language Model` `Multimodality`

**🎯 论文内容**

提出一种名为 ScalSelect 的训练无关、多模态数据选择方法，能够在视觉指令调优中只使用全数据集约 16% 的样本就能保持接近甚至超越全量训练的性能。

**💡 创新点**

创新点包括：①基于目标 VLM 第一层自注意力的指令相关视觉表示，真正做到指令感知；②采用全局子空间拟合（统计杠杆得分）而非对样本对的相似度，避免 O(N²) 复杂度；③整个过程不依赖任何外部模型或辅助数据集，计算量线性可扩展。

**🔧 技术方法**

使用技术：视觉-语言预训练模型（如 LLaVA、Qwen3-VL）、Transformer 第一层自注意力取样、聚合视觉标记、矩阵中心化、截断 SVD、统计杠杆得分排序。

**📊 数据集**

主要数据集：LLaVA‑V‑625K（625K 多模态样本）与 LRV‑Sub‑180K（180K GPT‑4 生成的多模态指令样本），实验也覆盖多种 VLM 架构。

**📈 对比分析**

与 Random、Length、Perplexity、COINCIDE、PRISM、RDS+ 等基线比较，在 100K 预算下，ScalSelect 在 8 大基准上平均达到 97.85% 的 Full‑Finetune 级别，甚至在 Qwen3‑VL 模型上超过全量训练，展示出优越的效率与性能平衡。

**⚠️ 局限性**

局限性包括：①对特定 VLM（如 LLaVA）和任务的实验多，跨模型泛化尚待进一步验证；②子空间维度 k 的选择需要经验设定，可能对不同数据分布敏感；③仅使用第一层表示，可能忽略深层的细粒度信息，在某些 OCR 等任务上性能略低。

---

## 23. Hardening the OSv Unikernel with Efficient Address Randomization: Design and Performance Evaluation

**arXiv ID:** 2602.11445 | [PDF](https://arxiv.org/pdf/2602.11445v1)

**作者:** Alex Wollman `[一作]` (Dakota State University), John Hastings `[通讯]` (Dakota State University)

**关键词:** `Cryptography and Security`

**🎯 论文内容**

在OSv unikernel中实现ASLR风格的地址随机化，随机化程序基址与线程栈地址，改动仅93行代码；

**💡 创新点**

通过在轻量化Unikernel中引入最小化改动的ASLR实现，保持性能不变且验证随机化分布均匀；

**🔧 技术方法**

使用C++内置PRNG（std::mt19937）生成64位随机数，修改mmu、elf、app等模块，并用Kolmogorov–Smirnov检验评估均匀性，Levene检验评估性能差异；

**📊 数据集**

在Dell Latitude 7450 + VMware 17环境下自行跑基准，共执行303次（两版本）收集启动时间、运行时间、内存使用等指标；

**📈 对比分析**

通过与原始OSv对比，采用Levene检验验证启动时间、运行时间与内存占用差异无统计显著性，显示随机化对性能无显著影响；

**⚠️ 局限性**

仅针对x86，未随机化库地址，堆未独立随机，超大内存分配路径未处理，缺乏对ARM等平台的支持，未来需扩展和完善。

---

## 24. Reconstructing Network Outbreaks under Group Surveillance

**arXiv ID:** 2602.11419 | [PDF](https://arxiv.org/pdf/2602.11419v1)

**作者:** Ritwick Mishra `[一作]` (University of Virginia), Anil Vullikanti `[通讯]` (University of Virginia)

**通讯引用:** 4074 | [OpenAlex ID](https://openalex.org/A5044848288)

**关键词:** `Social and Information Networks` `Graph Neural Network` `Graph` `Biomedical Data`

**🎯 论文内容**

提出在群体监测下重构疾病传播网络的最大似然估计问题，并给出了近似算法

**💡 创新点**

首次将最大似然估计与组Steiner树、随机线性规划结合，针对不同传播步长给出多种近似方案

**🔧 技术方法**

利用IC模型、组Steiner树求解、LP松弛+随机化取整、边/节点权重变换等技术

**📊 数据集**

在合成BA、G(n,q)网络、UVA医院ICU接触网络及Virginia数字孪生人口网络上实验

**📈 对比分析**

与将组池拆分为单个个体的Baseline（ApproxCascade-Random、RoundCascade-Random、ApproxCascade-All）及基于Steiner树的传统方法对比，F1分数和流行率相对误差均优于Baseline，尤其在低传播概率和中等池大小时表现突出

**⚠️ 局限性**

MLE解在存在噪声或某些池设计时可能与真实传播距离较远，且在大池或高传播概率时会低估/高估流行度；对噪声的敏感性和池设计的局限性需要进一步研究

---

## 25. The Pensieve Paradigm: Stateful Language Models Mastering Their Own Context

**arXiv ID:** 2602.12108 | [PDF](https://arxiv.org/pdf/2602.12108v1)

**作者:** Xiaoyuan Liu `[一作]` (Tencent AI Lab), Yan Wang `[通讯]` (Tencent AI Lab)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Optimization` `Transformer` `Large Language Model` `Reinforcement Learning` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

本文提出StateLM，能够主动通过工具管理自身上下文，实现长篇文本、长记忆和深度研究任务的高效推理。

**💡 创新点**

创新点在于将上下文管理的主动权从外部脚本迁移至模型自身，构建Pensieve框架并训练模型学习删除、读取、摘要等工具操作。

**🔧 技术方法**

技术包括基于工具的强化学习（GRPO式）与监督学习、上下文删除工具（deleteContext）、读取工具（readChunk）、摘要工具（updateNote）以及BM25关键词检索。

**📊 数据集**

使用的主要数据集包括NovelQA、NarrativeQA、LongBench、BrowseComp-Plus、LongMemEval-S、∞Bench、Synthetic Needle-in-a-Haystack等。

**📈 对比分析**

与传统指令模型、Qwen3-235B、ReadAgent、MemAgent等方法对比，StateLM在长文档QA、聊天记忆和深度研究任务上分别提升约10%–20%和40%以上的准确率，且仅使用1/4的上下文窗口。

**⚠️ 局限性**

局限性包括检索覆盖率有限、格式化错误、上下文削减后占用空间和超时导致的错误，未来需要更强的检索、更大管理窗口及高质量训练样本。

---

## 26. Do Large Language Models Adapt to Language Variation across Socioeconomic Status?

**arXiv ID:** 2602.11939 | [PDF](https://arxiv.org/pdf/2602.11939v1)

**作者:** Elisa Bassignana `[一作]` (IT University of Copenhagen), Amanda Cercas Curry `[通讯]` (CENTAI Institute)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Prompt Engineering` `Text` `Video`

**🎯 论文内容**

研究大语言模型在不同社会经济地位（SES）社区中对语言风格的适应程度，评估其在Reddit和YouTube语料上的表现

**💡 创新点**

首次量化LLM在不同SES群体中的语言风格模仿效果，并发现其更易模仿高SES风格，易放大语言不平等

**🔧 技术方法**

使用四种最先进LLM（Gemma‑3‑27B‑it、Mistral‑Small‑3.2‑24B、Qwen3‑30B‑A3B、GPT‑5）和94项社会语言学指标进行风格比较

**📊 数据集**

构建了按SES分层的Reddit和YouTube公开数据集（分别约2000条帖子和1000条视频字幕），并通过关键词筛选和网络分析验证SES标签

**📈 对比分析**

采用三种提示（隐式、显式风格、显式风格+SES）和Mann‑Whitney U检验+Holm‑Bonferroni校正，对LLM生成文本与人类原文在94个指标上的频率比进行森林图可视化，结果显示LLM仅在少数指标上接近人类且整体偏离真实SES差异

**⚠️ 局限性**

研究局限包括仅涵盖两大平台、缺乏自报或客观SES指标、使用表面词汇特征、提示策略有限、模型样本有限等

---

## 27. Evaluating Alignment of Behavioral Dispositions in LLMs

**arXiv ID:** 2602.11328 | [PDF](https://arxiv.org/pdf/2602.11328v1)

**作者:** Amir Taubenfeld `[一作]` (Google Research), Amir Feder `[通讯]` (Hebrew University)

**通讯引用:** 1360 | [OpenAlex ID](https://openalex.org/A5056266191)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

将心理问卷的自我报告条目转化为情境判断测试（SJT），通过LLM生成情境并让模型给出自然响应，再用LLM-as-a-Judge将其映射为支持或反对该条目的两种行动，评估LLM在现实对话场景中的行为倾向与人类倾向的对齐程度。

**💡 创新点**

创新点在于：①将自我报告条目映射为可操作的情境判断测试，直接观察行为而非仅靠自述；②利用LLM-as-a-Judge进行开放式响应的自动归类；③大规模收集10名评审的真实行动偏好，构建高质量人类基准；④系统评估25种不同的LLM模型，揭示模型容量、训练方式对行为倾向的影响。

**🔧 技术方法**

技术方法包括：LLM生成SJT场景与两种对立行动；人工三人审核保证情境与行动的一致性；LLM-as-a-Judge（如Gemini 3 Flash）将模型自由文本映射到两种行动；对模型输出进行分布与方向性对齐分析；使用统计指标（如一致率、偏差率、随机基准比较）。

**📊 数据集**

使用了260条自我报告心理学条目（经过滤后161条）与相应的情境判断测试，最终生成2,357个SJT；人类评审共计23,000条标注，覆盖10名评审每个SJT。

**📈 对比分析**

通过将模型选择与人类偏好分布进行比较，评估分布对齐度和方向性一致度。结果显示：在高人类共识情境下，大多数模型仍出现15–20%的对立选择；小型模型在低共识情境下与随机概率相当；更大模型在方向性对齐上有所提升，但仍存在显著误差。

**⚠️ 局限性**

局限性包括：①实验环境是二元选择，缺乏对不确定性与多重意见的细致把握；②SJT构建以美国/英国样本为主，文化代表性不足；③生态效度有限，现实对话可能更复杂；④未对多轮互动中的行为倾向进行评估。

---

## 28. ViTaS: Visual Tactile Soft Fusion Contrastive Learning for Visuomotor Learning

**arXiv ID:** 2602.11643 | [PDF](https://arxiv.org/pdf/2602.11643v1)

**作者:** Yufeng Tian `[一作]`, Huazhe Xu `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Convolutional Neural Network` `Contrastive Learning` `Reinforcement Learning` `Multimodality` `Image`

**🎯 论文内容**

本文提出 ViTaS 框架，将视觉与触觉信息通过软融合对比学习和 CVAE 进行融合，应用于强化学习和模仿学习的机器人操作任务。

**💡 创新点**

创新点在于引入软融合对比学习以对齐视觉与触觉特征，并结合 CVAE 充分利用两模态的互补性，显著提升在遮挡和自遮挡场景下的表现。

**🔧 技术方法**

技术上使用软融合对比学习、条件变分自编码器（CVAE）、CNN 编码器、PPO 强化学习、Diffusion Policy 模仿学习等。

**📊 数据集**

实验基于 12 个仿真任务（Gymnasium、Robosuite 等）和 3 个真实世界任务，使用 Galaxea‑R1、Meta Quest 3 采集的 RGB 摄像头与多模态触觉传感器数据。

**📈 对比分析**

与 M3L、VTT、MViTac、ConViTaC 等基线相比，ViTaS 在 9 个原始任务和 3 个额外任务中平均提升 15‑30% 的成功率，在真实世界实验中比 DP 提升约 16% 的成功率。

**⚠️ 局限性**

局限性包括对高度动态、精确操作（如真实世界笔旋转）的挑战，以及对变形物体的适应性不足。

---

## 29. Explainable Machine-Learning based Detection of Knee Injuries in Runners

**arXiv ID:** 2602.11668 | [PDF](https://arxiv.org/pdf/2602.11668v1)

**作者:** David Fuentes-Jiménez `[一作]` (University of Alcalá), Francisco-Manuel Melgarejo-Meseguer `[通讯]` (Universidad Rey Juan Carlos)

**通讯引用:** 233 | [OpenAlex ID](https://openalex.org/A5052658578)

**关键词:** `Machine Learning` `Classification` `Explainability and Interpretability` `Convolutional Neural Network` `Recurrent Neural Network` `Time Series`

**🎯 论文内容**

使用光学运动捕捉数据和监督机器学习方法，研究跑步者的膝关节损伤模式。

**💡 创新点**

创新点在于将完整的时间序列与点值特征结合，全面评估多种经典与深度学习模型，并通过SHAP、梯度可视化等解释技术揭示模型决策。

**🔧 技术方法**

采用KNN、SVM、GP、决策树、AdaBoost、随机森林、ANN、CNN、LSTM等算法，并结合时间序列与点值特征。

**📊 数据集**

利用公开的839条跑步记录（576名健康、137 PFPS、126 ITBS）运动捕捉数据库。

**📈 对比分析**

通过5折交叉验证比较模型性能，CNN在时间序列+点值输入下取得最高准确率：PFPS 77.9%，ITBS 73.8%，泛化损伤 71.4%，优于所有经典模型。

**⚠️ 局限性**

局限性包括仅使用跑步机数据、仅关注立位期、样本量仍有限、对真实环境运动可推广性不足，以及缺乏其他类型损伤的评估。

---

## 30. ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation

**arXiv ID:** 2602.11598 | [PDF](https://arxiv.org/pdf/2602.11598v1)

**作者:** Zedong Chu `[一作]` (Alibaba Group), Mu Xu `[通讯]` (Alibaba Group)

**通讯引用:** 459 | [OpenAlex ID](https://openalex.org/A5100532751)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Chain-of-Thought` `Vision-Language-Action Model` `Multimodality`

**🎯 论文内容**

构建并训练了一个统一的 Vision‑Language‑Action（VLA）基础模型 ABot‑N0，能够一次性完成点目标、物体目标、指令跟随、兴趣点目标和人跟随等五大导航任务。

**💡 创新点**

创新点在于：① 采用分层“Brain‑Action”架构，将大语言模型作为认知脑与流匹配动作专家相结合，实现高层语义推理与低层连续控制的无缝衔接；② 通过统一编码器整合多模态输入与多样化目标，实现跨任务的共享表示；③ 设计了“大规模数据引擎”，集成了 16.9M 轨迹与 5M 认知推理样本，打破了传统任务孤岛化的局限；④ 引入 Agentic Navigation System，将规划、记忆与神经控制层级化，支持复杂开放世界的闭环自主导航。

**🔧 技术方法**

主要技术包括 Vision Transformer + SigLIP 编码、Qwen3‑4B 语言模型、流匹配 (Flow Matching) 动作专家、Chain‑of‑Thought 认知推理、SAFE‑GRPO 强化学习价值对齐、Topo‑Memory 记忆图、Neural Controller 10Hz 速度控制。

**📊 数据集**

使用的数据集包括：16.9M 统一轨迹数据（覆盖 7,802 个高保真 3D 场景、室内外 10.7 km²），5M 认知推理样本（涵盖可行走区域分析、社交导航 CoT、指令跟随推理、物体目标推理、POI 定位等），以及公开基准（CityWalker、SocNav、VLN‑CE、BridgeNav、EVT‑Bench 等）进行评测。

**📈 对比分析**

在七大主流基准上，ABot‑N0 统一模型实现了多项 SOTA 结果：在 CityWalker 的 MAOE 下降到 11.2（低于 15.2），在 SocNav 的成功率提升至 88.3%（远超 47.8%），在 VLN‑CE 的 SPL 提升至 56.3% 以上，BridgeNav 的 0.1m 目标成功率提升 70.1%，EVT‑Bench 的跟踪率和碰撞率也取得显著提升。

**⚠️ 局限性**

局限性包括：① 仍需依赖大规模标注数据，数据收集成本高；② 训练和推理对算力要求较高，尽管通过压缩实现 2Hz 现场推理，但在更高频率或资源受限环境下仍有挑战；③ 对极端动态环境（如突发障碍、复杂人群）适应性尚未完全验证；④ 目前的“Brain‑Action”分离方式可能在极度需要实时反馈的场景下引入额外延迟。

---

## 31. An Auction-Based Mechanism for Optimal Task Allocation and Resource Aware Containerization

**arXiv ID:** 2602.11998 | [PDF](https://arxiv.org/pdf/2602.11998v1)

**作者:** Ramakant kumar `[一作]`, Ramakant kumar `[通讯]`

**关键词:** `Distributed, Parallel, and Cluster Computing` `Optimization` `Tabular`

**🎯 论文内容**

提出并实现了基于拍卖的任务分配与资源感知容器化框架 AUC‑RAC，能够在 IoT 环境下将计算密集任务高效地分配到多台本地服务器，并在容器层面进一步优化执行成本。

**💡 创新点**

创新点在于：①将拍卖机制与分布式 Docker Swarm 相结合，实现任务分配的双层成本优化；②在任务分配后引入资源感知容器化，动态为每个子任务分配最合适的容器，既降低了执行延迟，又提高了资源利用率；③通过贝叶斯拍卖模型实现 WN 的利润最大化，使得系统在公平性与经济性上兼顾。

**🔧 技术方法**

采用的技术包括 Docker Swarm（管理节点与工作节点架构）、Docker 容器化、基于 Bayesian game 的封闭式拍卖算法、资源分配与容器选型的最佳匹配算法，以及实验环境的模拟仿真。

**📊 数据集**

实验使用仿真生成的 IoT 任务数据集（不同强度的任务和随机生成的服务器资源参数），并未使用公开真实数据集。

**📈 对比分析**

通过与静态分配、启发式分配、随机、轮询、贪心、最短完成时间（MCT）以及已有的拍卖方法进行对比。实验结果表明：AUC‑RAC 在任务完成时间、延迟、利润、资源利用率和公平性方面均优于基线方法，尤其在高并发场景下表现出更好的可扩展性。

**⚠️ 局限性**

局限性包括：①仅在仿真环境中验证，缺乏真实部署与大规模部署的实测；②拍卖模型假设竞标者独立同分布，实际网络中可能存在协同与非独立性；③对不完整或异常数据的处理尚未深入，未来工作需考虑更鲁棒的学习与动态适配机制。

---

## 32. TreeGrad-Ranker: Feature Ranking via $O(L)$-Time Gradients for Decision Trees

**arXiv ID:** 2602.11623 | [PDF](https://arxiv.org/pdf/2602.11623v1)

**作者:** Weida Li `[一作]` (National University of Singapore), Bryan Kian Hsiang Low `[通讯]` (National University of Singapore)

**通讯引用:** 857 | [OpenAlex ID](https://openalex.org/A5030304400)

**关键词:** `Machine Learning` `Classification` `Optimization` `Explainability and Interpretability` `Computational Efficiency` `Tabular`

**🎯 论文内容**

本文提出了一系列针对决策树的特征重要性评估方法，尤其通过TreeGrad计算梯度并直接优化插入/删除指标对应的联合目标，从而生成更可靠的特征排名；同时提出TreeGrad‑Ranker、TreeGrad‑Shap、TreeProb等算法以提升数值稳定性和计算效率。

**💡 创新点**

创新点在于：①证明概率值（Shapley、Banzhaf等）在特征排名中并不可靠；②将联合优化问题转化为可梯度优化的形式，并设计O(L)时间的TreeGrad；③利用梯度聚合产生满足大多数公理（除线性外）的特征评分；④开发数值稳定的TreeGrad‑Shap（Beta Shapley）和TreeProb（通用概率值）算法。

**🔧 技术方法**

技术手段包括多项式（多线性扩展）与内积、梯度上升/ADAM优化、树结构的上下遍历、Vandermonde矩阵改进（使用单位圆节点）、Gauss‑Legendre四边形积分、以及对线性树Shap算法的数值重构。

**📊 数据集**

使用了九个OpenML数据集，涵盖分类（FOTP、GPSP、jannis、spambase、philippine、MinibooNE）和回归（BT、superconduct、wave_energy）任务。

**📈 对比分析**

实验通过插入与删除指标与Beta Shapley、Banzhaf等基线比较。TreeGrad‑Ranker在插入指标上显著优于所有基线，删除指标亦保持竞争力；TreeGrad‑Shap的数值误差比Linear TreeShap低多达10^15倍。

**⚠️ 局限性**

局限性包括：算法仅针对决策树；深度较大时仍可能出现数值不稳定；缺失线性公理导致与传统概率值不完全一致；未扩展到非树模型；在极端数据分布或极大特征维度下的表现尚未验证。

---

## 33. TRACER: Trajectory Risk Aggregation for Critical Episodes in Agentic Reasoning

**arXiv ID:** 2602.11409 | [PDF](https://arxiv.org/pdf/2602.11409v1)

**作者:** Sina Tayebati `[一作]` (University of Illinois at Chicago), Amit Ranjan Trivedi `[通讯]` (University of Illinois at Chicago)

**关键词:** `Artificial Intelligence` `Large Language Model` `Agentic AI` `Text` `Benchmark`

**🎯 论文内容**

设计并评估TRACER，一种面向多轮工具使用交互的轨迹级不确定性度量；

**💡 创新点**

将内容感知惊奇、情境感知重复/一致性指标与MAX组合及尾部风险聚合相结合，专门针对稀疏关键失败事件；

**🔧 技术方法**

利用语言模型Token概率与熵、嵌入相似度、语义/词汇重复检测以及CVaR与极值聚合技术；

**📊 数据集**

在τ^2‑bench（航空、零售、电信）多轮工具使用对话数据上进行实验；

**📈 对比分析**

与标准熵、self‑consistency、semantic entropy等基线在AUROC/AUARC上比较，TRACER在所有模型和域中提升4.7%–37.1%（AUROC）及6%–55%（AUARC），并在早期警报上显著优于基线；

**⚠️ 局限性**

依赖Token概率和嵌入计算，难以处理完全黑盒用户模拟器；仅关注稀疏危害假设，未验证持续高不确定性场景的效果。

---

## 34. Electrostatics-Inspired Surface Reconstruction (EISR): Recovering 3D Shapes as a Superposition of Poisson's PDE Solutions

**arXiv ID:** 2602.11642 | [PDF](https://arxiv.org/pdf/2602.11642v1)

**作者:** Diego Patiño `[一作]` (University of Texas - Arlington), David K. Han `[通讯]` (Drexel University)

**通讯引用:** 2318 | [OpenAlex ID](https://openalex.org/A5102000970)

**关键词:** `Computer Vision and Pattern Recognition` `Optimization` `Segmentation` `Point Cloud` `Mesh`

**🎯 论文内容**

提出一种基于 Poisson 方程的隐式表面重建方法 EISR，利用电势理论将形状表示为正电荷分布的电势场，并通过 Green 函数解析求解，将目标形状表示为若干高斯电荷的线性叠加，使用点云监督优化电荷参数。

**💡 创新点**

创新点：① 用线性 Poisson PDE 替代非线性 Eikonal 方程，消除对法向或距离标签的需求；② 利用 Green 函数得到闭式解，允许用高斯电荷参数化并线性叠加；③ 只需点云即可完成监督，且不需要额外的损失项；④ 通过电荷约束确保电荷位于内部，从而保证 Marching Cubes 的可行性。

**🔧 技术方法**

技术细节：Green 函数解析求解 Poisson 方程；高斯电荷参数化（位置、幅度、扩散）；利用线性叠加构造隐式场；使用边界条件损失和电荷约束损失；Adam 优化电荷参数；通过三角网格提取 iso-surface。

**📊 数据集**

使用的数据集：IGR 论文中使用的点云数据集；斯坦福 3D 扫描仓库（扫描对象）；DTU 数据集（多视角深度图）等。

**📈 对比分析**

对比方法：与 Implicit Geometric Regularization (IGR) 以及经典 Poisson Surface Reconstruction (PSR)，PSR 版本分别使用真实法向和估计法向。评价指标包括 Chamfer 距离、Hausdorff 距离、F1 分数、IoU 等。实验显示 EISR 在点云重建任务中与 IGR 竞争，甚至在许多场景下超过 PSR，且在高频细节上表现更佳，只需数千个高斯电荷即可得到精细重建。

**⚠️ 局限性**

局限性：① 需要完整点云监督，难以直接应用于仅有 RGB 或部分视角的场景；② 对平坦或凹面难以捕捉，理论上需要无限多电荷；③ 计算每个查询点与所有电荷的距离是瓶颈，导致效率受限；④ 尚未探索更高效的形状先验（如电荷线、平面）和多视角/渲染监督的扩展。

---

## 35. From Noise to Order: Learning to Rank via Denoising Diffusion

**arXiv ID:** 2602.11453 | [PDF](https://arxiv.org/pdf/2602.11453v1)

**作者:** Sajad Ebrahimi `[一作]` (University of Guelph), Ebrahim Bagheri `[通讯]` (University of Toronto)

**通讯引用:** 8159 | [OpenAlex ID](https://openalex.org/A5064660738)

**关键词:** `Information Retrieval` `Recommendation System` `Diffusion model` `Tabular`

**🎯 论文内容**

本文提出了一种基于去噪扩散的生成式学习排序模型DiffusionRank，并在点对和对间设置下实现了与传统判别式LTR的对应训练目标。

**💡 创新点**

创新点在于将TabDiff的混合类型扩散过程引入LTR，构建了能联合建模特征向量与相关性标签的生成式框架，从而获得更稳健的排序效果。

**🔧 技术方法**

核心技术为连续时间去噪扩散概率模型（DDPM）与TabDiff的数值与类别混合扩散机制，并采用前馈网络作为去噪器。

**📊 数据集**

实验使用了LETOR4.0中的MQ2007、MQ2008以及MSLR-WEB10K三大公开LTR数据集。

**📈 对比分析**

与XGBoost和相同网络结构的判别式点对/对间LTR基线在NDCG@10和MAP@10上进行对比，DiffusionRank在大部分数据集及多种训练样本比例下均显著优于判别式模型，尤其在数据量充足时提升更为明显。

**⚠️ 局限性**

局限性包括在小样本或极低训练比例时性能波动较大，且本文仅探索了点对和对间的生成式目标，未覆盖列表式损失及多模态输入等更广泛场景。

---

## 36. Optimizing Agent Planning for Security and Autonomy

**arXiv ID:** 2602.11416 | [PDF](https://arxiv.org/pdf/2602.11416v1)

**作者:** Aashish Kolluri `[一作]` (Microsoft), Santiago Zanella-Béguelin `[通讯]` (Microsoft)

**通讯引用:** 3458 | [OpenAlex ID](https://openalex.org/A5088954009)

**关键词:** `Cryptography and Security` `Optimization` `Safty and Privacy` `Robotic Intelligence` `Agentic AI` `Tabular` `Benchmark`

**🎯 论文内容**

提出并验证了自动化度量方法，用以评估AI代理在遵守安全策略时对人类监督的依赖程度；设计了基于信息流控制（IFC）的安全感知代理，并在AgentDojo与WASP基准上进行实验。

**💡 创新点**

创新点在于：①首次引入“自动化度量”指标（HITL次数和k-曲线）来量化安全代理的自治水平；②将IFC与策略感知规划结合，设计变量扩展与授权端点，让代理主动规划以规避安全违规；③通过实验展示，IFC感知代理在保持或提升任务完成率的同时，显著降低人类干预需求。

**🔧 技术方法**

采用的技术包括：信息流控制（IFC）与标签传播、Dual LLM模式（隔离规划与执行）、策略感知规划、变量隐藏与授权（endorsement）机制、对话式工具调用（function calls）。

**📊 数据集**

使用的主要数据集：AgentDojo benchmark（涵盖银行、聊天、旅行、工作空间等四大任务套件）和WASP benchmark（针对GitLab与Reddit的prompt injection攻击测试）。

**📈 对比分析**

对比方法：基准ReAct代理、IFC+策略检查的基线、以及现有最先进的IFC代理。实验结果显示，IFC感知代理在k=0（无人工干预）下的任务完成率至少提升9-10%，而在相同完成率下HITL次数降低1.5-2.6倍，整体自主度明显优于其他方法。

**⚠️ 局限性**

局限性包括：①仅针对间接prompt injection攻击，未覆盖直接注入、工具破坏等攻击；②假设用户、规划者与工具可信，未处理模型幻觉或误解任务的情况；③缺乏针对人机界面的安全与易用性研究，实际部署时仍需完善交互设计。

---

## 37. Beyond Parameter Arithmetic: Sparse Complementary Fusion for Distribution-Aware Model Merging

**arXiv ID:** 2602.11717 | [PDF](https://arxiv.org/pdf/2602.11717v1)

**作者:** Weihong Lin `[一作]` (Beijing Qiyuan Technology Co., Ltd.), Tong Yang `[通讯]` (Peking University)

**通讯引用:** 5635 | [OpenAlex ID](https://openalex.org/A5101674305)

**关键词:** `Artificial Intelligence` `Large Language Model` `Multimodality`

**🎯 论文内容**

提出一种基于逆 KL 的稀疏补充融合方法（SCF‑RKL），实现无数据的模型合并。

**💡 创新点**

创新点在于利用逆 KL 衡量功能差异并通过分位数阈值实现稀疏选择，避免模式崩塌与重复。

**🔧 技术方法**

技术手段包括逆 Kullback–Leibler 重要性评估、分位数阈值化、稀疏掩码更新以及对熵与谱偏移的理论界定。

**📊 数据集**

使用多种公开大语言模型（Mistral、LLaMA3、Qwen2.5）及其数学、代码、安全等专项模型，并在 24 个语言基准和 7 个视觉基准上评测。

**📈 对比分析**

与 Task Arithmetic、TIES、DARE、SCE 等传统方法对比，SCF‑RKL 在 24 组基准上平均提升 0.8 分，重复率降至 <1%，并在安全、推理等维度保持或提升性能。

**⚠️ 局限性**

局限性在于对极弱强融合的效果仍有限，且对跨模态多任务的通用性需进一步验证。

---

## 38. SemaPop: Semantic-Persona Conditioned Population Synthesis

**arXiv ID:** 2602.11569 | [PDF](https://arxiv.org/pdf/2602.11569v1)

**作者:** Zhenlin Qin `[一作]` (KTH Royal Institute of Technology), Zhenliang Ma `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1803 | [OpenAlex ID](https://openalex.org/A5072433020)

**关键词:** `Artificial Intelligence` `Data Synthesis` `Generation` `Large Language Model` `Generative Adversarial Network` `Tabular`

**🎯 论文内容**

研发了一种基于大语言模型与生成对抗网络的语义‑统计融合人口合成框架SemaPop，能够在保持宏观统计一致性的同时通过人格语义控制个体属性的生成；

**💡 创新点**

创新点在于将大语言模型生成的Persona文本进行向量化抽象，并通过FiLM特征调制与投影鉴别器将其注入GAN，实现语义约束下的可控、可解释的合成；

**🔧 技术方法**

技术包括大语言模型（LLM）进行Persona生成与编码、Wasserstein GAN‑GP作为生成器、FiLM调制与投影鉴别器、边缘正则化（SRMSE）以及后置边缘校准；

**📊 数据集**

使用了合成瑞典人口微观数据（约10M代理），通过分区抽样得到训练/验证/测试子集；

**📈 对比分析**

与CTGAN、TabDDPM、TVAE、BN、BN‑Copula、WGAN‑GP、WGAN‑GP‑ZCR、SemaPop‑VAE等基线在SRMSE‑M/B、精确率/召回率/F1 进行对比，SemaPop‑GAN 在所有指标上表现最佳，精确率73.95%、召回率93.39%、F1 82.54%，SRMSE‑M 0.0104、SRMSE‑B 0.0554；

**⚠️ 局限性**

局限性包括仅在合成数据上验证，缺乏真实调查数据；Persona的质量依赖LLM生成，且在大规模真实场景下的训练成本与隐私约束仍需进一步研究。

---

## 39. PhyNiKCE: A Neurosymbolic Agentic Framework for Autonomous Computational Fluid Dynamics

**arXiv ID:** 2602.11666 | [PDF](https://arxiv.org/pdf/2602.11666v1)

**作者:** E Fan `[一作]` (Hong Kong Polytechnic University), Chih-yung Wen `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 7082 | [OpenAlex ID](https://openalex.org/A5038976027)

**关键词:** `Artificial Intelligence` `Large Language Model` `Retrieval-Augmented Generation` `Agentic AI` `Physics Related`

**🎯 论文内容**

提出了PhyNiKCE框架，将LLM生成与符号验证分离，用于自动化OpenFOAM CFD仿真。

**💡 创新点**

创新点在于构建Deterministic RAG Engine和多种专用检索器，实现物理一致性与可审计性，同时解耦神经网络生成与符号约束。

**🔧 技术方法**

采用神经符号方法、Deterministic RAG检索、结构化知识库、专用检索策略，并使用Gemini-2.5 LLM作为语言模型。

**📊 数据集**

利用约400个OpenFOAM教程案例构建知识库，验证集为NACA 0012空气泡翼和De Laval喷嘴的12种实战组合。

**📈 对比分析**

通过与Vector RAG、ChatCFD基线、Partial PhyNiKCE等四种配置对比，使用执行率和准确率评估；Full PhyNiKCE在准确率上比基线提升约51%（从26%到51%），错误反射次数降低59%，令牌消耗降低17%。

**⚠️ 局限性**

仍存在压缩流量性能偏低、知识库偏向不压缩、缺乏后处理验证以及难以处理运行后物理误差等局限。

---

## 40. GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry

**arXiv ID:** 2602.11714 | [PDF](https://arxiv.org/pdf/2602.11714v1)

**作者:** Jiung Yeon `[一作]` (Sungkyunkwan University), Hyeonwoo Yu `[通讯]` (Sungkyunkwan University)

**通讯引用:** 158 | [OpenAlex ID](https://openalex.org/A5060844581)

**关键词:** `Computer Vision and Pattern Recognition` `Optimization` `Robotic Intelligence` `Simultaneous Localization and Mapping` `Gaussian Splatting` `Simultaneous Localization and Mapping` `Point Cloud`

**🎯 论文内容**

提出了一种双向耦合的单目SLAM系统GSO‑SLAM，结合了视觉里程计与高效的高斯平面化（Gaussian Splatting）实现实时稠密重建与精确定位。

**💡 创新点**

创新点在于将视觉里程计与GS的优化通过EM框架实现双向耦合，消除冗余计算；以及利用DSO的梯度信息直接初始化高斯原语，显著加速收敛。

**🔧 技术方法**

采用Direct Sparse Odometry (DSO)作为视觉里程计，2D Gaussian Splatting进行稠密场景表示，Expectation‑Maximization框架实现联合优化，并使用GPU加速的光栅化渲染。

**📊 数据集**

在合成Replica、真实TUM‑RGBD、INS长室内走廊以及机器人自采四足机数据集上进行评估。

**📈 对比分析**

与多种基线（如MonoGS、SplaTAM、Point‑SLAM、Photo‑SLAM、MGSO）对比，GSO‑SLAM在追踪误差、PSNR/SSIM/LPIPS、L1深度误差等指标上均优于单目基线，并在实时性（30 fps）和大规模环境下保持低轨迹误差。

**⚠️ 局限性**

局限包括对极端运动模糊和纹理稀疏区域仍易出现漂移或几何失真，且当前仅使用离散点描述半稠密地图，未来需进一步引入图像结构信息以提升鲁棒性。

---

## 41. Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization

**arXiv ID:** 2602.11437 | [PDF](https://arxiv.org/pdf/2602.11437v1)

**作者:** Chengrui Qu `[一作]` (California Institute of Technology), Adam Wierman `[通讯]` (California Institute of Technology)

**通讯引用:** 10074 | [OpenAlex ID](https://openalex.org/A5062565732)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Reinforcement Learning`

**🎯 论文内容**

开发了分布式鲁棒IGM（DrIGM）原则及其在价值因式分解中的鲁棒实现，以在CTDE环境下提升对模型不确定性的鲁棒性。

**💡 创新点**

提出了 DrIGM 这一新的鲁棒性原则，证明在全局最坏模型下可保证个体-全局最大化，从而在多智能体强化学习中实现分布式鲁棒决策，并将其应用于 VDN、QMIX、QTRAN，得到可直接使用的鲁棒算法。

**🔧 技术方法**

采用分布式鲁棒强化学习、分布式鲁棒 Bellman 算子、ρ-污染与 TV 不确定集、DRQN + 价值因式分解网络、经验回放与目标网络等技术。

**📊 数据集**

在 SustainGym 的 HVAC 控制仿真环境（多场景气候与季节漂移）以及 StarCraft II 的 SMAC 难度地图（观测噪声漂移）上进行实验。

**📈 对比分析**

与传统 VDN/QMIX/QTRAN 及 GroupDR 鲁棒基线对比，在分布式漂移测试中，鲁棒方法在大多数情景下提升 10–40% 的奖励或胜率，且在训练环境下往往不损失甚至提升性能。

**⚠️ 局限性**

目前仅在全局不确定集下定义，未考虑代理级别不确定集；对大规模代理网络的扩展和分布式训练仍有挑战；鲁棒参数 ρ 的选择仍需经验调优。

---

## 42. Grok in the Wild: Characterizing the Roles and Uses of Large Language Models on Social Media

**arXiv ID:** 2602.11286 | [PDF](https://arxiv.org/pdf/2602.11286v1)

**作者:** Katelyn Xiaoying Mei `[一作]` (University of Washington), Martin Saveski `[通讯]` (University of Washington)

**通讯引用:** 722 | [OpenAlex ID](https://openalex.org/A5082294805)

**关键词:** `Social and Information Networks` `Classification` `Generation` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

对 xAI 的大型语言模型 Grok 在 X 社交平台上的公开互动进行系统化采样与分析，涵盖 41,735 条互动链，研究其使用频率、语言分布、参与度、互动类型、模型扮演的社交角色以及呼叫用户的特征。

**💡 创新点**

首次在公共社交空间中对 LLM 进行大规模实地研究，构建了 10 种新颖的社交角色分类，并揭示角色与用户兴趣、平台语言、对话上下文之间的关联；提出将角色映射到支持/限制与自我/群体维度的二维框架。

**🔧 技术方法**

采用混合方法：统计计数、定量参与度评估；定性内容分析与专家开放编码；使用多模 LLM（Gemini‑2.5‑Pro、GPT‑5‑Mini 等）对互动链进行使用类别与角色标签；使用 BERTopic + UMAP 对对话根与用户简介进行主题建模；利用 Cohen’s κ 评估人工标注与 LLM 分类的一致性。

**📊 数据集**

数据来源于 X 官方 API，覆盖 2025 年 8 月 15 日至 11 月 17 日的公开数据，包含 142,895 条帖子、41,735 条互动链、31,111 条用户简介；同时对每条互动链的 48 小时后参与度（观看数、点赞数、转发数、回复数）进行二次抓取。

**📈 对比分析**

通过与根帖（Conversation Root）对比评估 Grok 回复的参与度（Grok 回复的 48 小时内平均观看数仅 147，点赞数 0.78，且 95% 的回复无转发），并对 LLM 分类的可靠性进行评估（Gemini‑2.5‑Pro 在 100 条标注样本上的 Cohen’s κ 达到 0.70；使用类别与角色分类的准确率未给出传统意义的性能指标，但显示高于 60% 的一致性）。

**⚠️ 局限性**

局限性包括：数据仅覆盖三个月、仅限英文公开互动；未考察长期使用趋势与跨语言差异；用户特征分析仅基于简介，缺乏互动历史；角色分类依赖 LLM，存在偏差；未进行因果或交互设计验证，无法确认平台政策对模型行为的直接影响。

---

## 43. Beyond Pixels: Vector-to-Graph Transformation for Reliable Schematic Auditing

**arXiv ID:** 2602.11678 | [PDF](https://arxiv.org/pdf/2602.11678v1)

**作者:** Chengwei Ma `[一作]`, F. Richard Yu `[通讯]`

**关键词:** `Artificial Intelligence` `Data Synthesis` `Anomaly Detection` `Graph Neural Network` `Large Language Model` `Prompt Engineering` `Graph`

**🎯 论文内容**

提出一种 Vector-to-Graph (V2G) 管道，将 CAD 图纸转换为属性图，从而克服多模态大语言模型的结构盲点。

**💡 创新点**

创新点在于用显式图结构代替像素输入，结合图信号处理进行确定性合规性验证，使得结构信息被明确编码并可审计。

**🔧 技术方法**

使用 LLM 驱动的实体与关系提取、属性图构建、Graph Signal Processing 验证函数、以及结构化提示的 MLLM 规划器。

**📊 数据集**

使用来自电力设施的 60 份真实 CAD 电气合规案例及其 10–20 个旋转/平移/微量缩放变体，约 900 个测试实例。

**📈 对比分析**

与六款主流 MLLM 进行对比，基线模型整体准确率仅 12%，加上 V2G 后提升至 47%（绝对增幅 35%），各类别均有显著提升。

**⚠️ 局限性**

局限性包括数据集规模有限、仅覆盖 CAD 矢量图，未考虑光栅图像，且系统依赖 LLM 的解析准确性，缺乏对更广泛工程图的验证。

---

## 44. Credal Concept Bottleneck Models: Structural Separation of Epistemic and Aleatoric Uncertainty

**arXiv ID:** 2602.11219 | [PDF](https://arxiv.org/pdf/2602.11219v1)

**作者:** Tanmoy Mukherjee `[一作]` (CRIL UMR 8188, University Artois), Zied Bouraoui `[通讯]`

**关键词:** `Machine Learning` `Classification` `Transformer` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

提出 Credal Concept Bottleneck Models (Credal CBMs)，通过结构分离实现对模型的认知不确定性 (epistemic) 与数据不确定性 (aleatoric) 的清晰分离。

**💡 创新点**

创新点包括：
- 结构分离（disjoint 参数化 + 梯度隔离）避免传统方法的 algebraic trap，显著降低两种不确定性之间的相关性；
- 使用 credal 集合（椭圆体）表示不确定性，其中集合大小对应 epistemic，集合内方差对应 aleatoric；
- 通过 Hausdorff KL 正则化和变分 ELBO 进行可扩展训练；
- 直接利用多注解者的概念不一致性作为 aleatoric 的监督信号，并用误差监督指导 epistemic 的规模。

**🔧 技术方法**

技术栈：
- Concept Bottleneck Network（CBN）架构，带冻结的 DistilBERT 编码器；
- 三个正交投影子网络（μ、σ_epi、σ_ale）实现参数分离；
- 变分推断 + Hausdorff KL 正则化；
- 通过 Spearman 相关性、AUROC、四象限路由等指标评估。

**📊 数据集**

数据集：
- CEBaB、GoEmotions、HateXplain（文本分类）；
- MAQA*、AmbigQA*（问答，含真实标签分布），用于验证对真正歧义的捕捉。

**📈 对比分析**

与基线对比：
- 基线包括 Semantic Entropy、Deep Ensembles、MC Dropout、P(True) 等，均在 epistemic 与 aleatoric 之间表现出强相关；
- Credal CBM 将相关系数降至约 0，显著提升与误差和真实歧义的对齐；
- 在 AUROC 与高歧义子集的 AUROC 上，Credal CBM 取得 0.11 以内的降幅，优于基线 0.20-0.22；
- 四象限路由实验显示 Credal CBM 能有效区分“模型不知道”与“任务本身歧义”，实现更有针对性的决策。

**⚠️ 局限性**

局限性：
- 需要多注解者概念标签，成本高且可能带来注解者偏差；
- 编码器冻结限制了特征表达的灵活性，可能在某些领域表现欠佳；
- 仅在文本任务上验证，未覆盖多模态场景；
- 未提供正式的覆盖率保证，且对极端分布漂移的鲁棒性尚未充分评估。

---

## 45. GAC-KAN: An Ultra-Lightweight GNSS Interference Classifier for GenAI-Powered Consumer Edge Devices

**arXiv ID:** 2602.11186 | [PDF](https://arxiv.org/pdf/2602.11186v1)

**作者:** Zhihan Zeng `[一作]` (National Key Laboratory of Wireless Communications, University of Electronic Science and Technology of China), Yue Xiu `[通讯]` (National Key Laboratory of Wireless Communications, University of Electronic Science and Technology of China)

**关键词:** `Machine Learning` `Classification` `Computational Efficiency` `Convolutional Neural Network` `Time Series`

**🎯 论文内容**

提出了一种名为GAC-KAN的超轻量级GNSS干扰分类器，旨在解决生成AI时代消费电子设备在GNSS信号保护方面的计算资源限制和数据稀缺问题。

**💡 创新点**

创新点在于采用物理引导的生成仿真方法合成高保真干扰数据集，并设计了结合了不对称卷积块和Ghost模块的多尺度特征提取骨干网络，以及使用可学习的样条激活函数的Kolmogorov-Arnold网络（KAN）作为分类头。

**🔧 技术方法**

使用了多尺度Ghost-ACB-Coordinate骨干网络和Kolmogorov-Arnold网络（KAN），结合了不对称卷积块和Ghost模块以提取丰富的光谱时域特征。

**📊 数据集**

使用了通过物理模型生成的合成干扰数据集，涵盖了七种不同的干扰类别，数据集规模达到56000个样本。

**📈 对比分析**

与现有的最先进基线模型（如Vision Transformer和其他深度学习模型）相比，GAC-KAN在准确率上达到了98.0%，参数量仅为0.13百万，计算复杂度为0.19 G FLOPs，表现出极高的参数效率和较低的计算成本。

**⚠️ 局限性**

限制在于模型在极低信噪比（JNR）条件下的表现可能不如一些传统模型，尽管在大多数情况下表现出色，但在特定情况下仍可能出现误分类。

---

## 46. Optimizing Distances for Multi-Broadcast in Temporal Graphs

**arXiv ID:** 2602.12126 | [PDF](https://arxiv.org/pdf/2602.12126v1)

**作者:** Daniele Carnevale `[一作]` (University of Geneva), Gianlorenzo D'Angelo `[通讯]` (Gran Sasso Science Institute)

**通讯引用:** 2212 | [OpenAlex ID](https://openalex.org/A5050291532)

**关键词:** `Data Structures and Algorithms` `Optimization` `Graph Neural Network` `Graph`

**🎯 论文内容**

本文提出并研究了时间图中的 -Temporal Multi-Broadcast (TMB) 问题，目标是为静态图的边分配时间标签，使得指定源集合到所有其他节点的最坏情况时间距离最优。

**💡 创新点**

创新点在于将多源时间广播问题与 REACHFAST 关联，系统地分析了六种时间距离度量（EA、LD、FT、ST、MH、MW）的计算复杂度与可近似性，并给出了单源 EA、LD 可多项式求解以及多源在边的多重度足够时可解的结果。

**🔧 技术方法**

主要技术包括构造时间拓扑树（TSOT）、多源拉伸算法、复杂度归约（对 SAT/3‑SAT 的构造）以及对树图的边多重度条件下的最优标记策略。

**📊 数据集**

论文未使用真实数据集，而是通过构造的图实例（如示例图和 3‑SAT 归约图）来证明理论结果。

**📈 对比分析**

比较方法通过与已知的 REACHFAST 结果以及 NP‑难度/近似下界对比，证明了单源 EA/LD 的多项式解和其他度量的 NP‑难/不可近似性，展示了不同度量下的复杂性差异。

**⚠️ 局限性**

限制在于仅对六种时间距离度量做了完整分析，且多源可解性仅在边多重度满足一定阈值或树结构下实现，其他图类（如平面图、树宽受限图）的可解性仍未确定。

---

## 47. SAGEO Arena: A Realistic Environment for Evaluating Search-Augmented Generative Engine Optimization

**arXiv ID:** 2602.12187 | [PDF](https://arxiv.org/pdf/2602.12187v1)

**作者:** Sunghwan Kim `[一作]` (Yonsei University), Dongha Lee `[通讯]` (Yonsei University)

**通讯引用:** 2943 | [OpenAlex ID](https://openalex.org/A5010775517)

**关键词:** `Information Retrieval` `Optimization` `Retrieval` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

提出了SAGEO Arena——一个真实可复现的端到端搜索增强生成引擎优化（SAGEO）评测环境，并基于170k条包含结构化信息的网页构建了完整的检索–重排–生成流水线；

**💡 创新点**

创新点在于①将结构化字段（标题、Meta、标题层级、JSON‑LD等）与正文分离并保留，②实现了可追踪各阶段可见度的评估框架，③提出了面向各阶段的“阶段感知SAGEO”优化策略，显著提升检索、重排与生成三阶段的可见度；

**🔧 技术方法**

采用BM25检索、Cross‑Encoder重排、基于LLM（如GPT‑5‑mini、LLaMA‑3.3‑70B、Qwen3‑80B）的生成器；在优化层面实现了十种基于LLM的重写策略并组合；

**📊 数据集**

使用从九个信息检索数据集抽取的2,700个查询，以及对应的171,003篇网页（含标题、Meta、标题层级、JSON‑LD和正文）构成实验语料；

**📈 对比分析**

通过Hit Rate、Citation Rate与Rank Change等指标对比十种优化策略与阶段感知SAGEO，实验表明传统“正文”优化往往降低检索可见度，而结构化优化可提升检索Hit Rate；阶段感知SAGEO在检索、重排与生成三阶段均优于其它方法（例如在生成阶段Citation Rate提升至约70%/80%，并且检索/重排Rank提升1–2位）；

**⚠️ 局限性**

限制包括：检索仍基于词袋BM25，未探索更高级的语义检索；优化策略未针对具体查询意图（无查询上下文）；实验仅在英语网页上验证，跨语言或跨文化适用性待进一步研究。

---

## 48. Embodied AI Agents for Team Collaboration in Co-located Blue-Collar Work

**arXiv ID:** 2602.12136 | [PDF](https://arxiv.org/pdf/2602.12136v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Human-Computer Interaction`

---

## 49. The Five Ws of Multi-Agent Communication: Who Talks to Whom, When, What, and Why -- A Survey from MARL to Emergent Language and LLMs

**arXiv ID:** 2602.11583 | [PDF](https://arxiv.org/pdf/2602.11583v1)

**作者:** Jingdi Chen `[一作]` (University of Arizona), Carlee Joe-Wong `[通讯]` (Carnegie Mellon University)

**通讯引用:** 17195 | [OpenAlex ID](https://openalex.org/A5003037377)

**关键词:** `Artificial Intelligence` `Large Language Model` `Text` `Review/Survey Paper`

**🎯 论文内容**

综述多智能体通信的演进与三大范式（MARL通信、突现语言、LLM驱动通信），并以“谁、什么、何时、为何”五个W构建统一分析框架；

**💡 创新点**

首次将通信从五个W视角统一剖析三种范式，搭建跨范式桥接与系统性综述，并提出开放挑战与未来研究路线；

**🔧 技术方法**

采用系统性文献检索与筛选方法，构建分类框架、对比表、图示，结合理论与实验结果进行分析；

**📊 数据集**

主要为综述性工作，未使用实验数据集；

**📈 对比分析**

通过对比表和图示评估不同通信范式在通信效率、可解释性、可扩展性与适用场景等维度的表现；

**⚠️ 局限性**

综述截至2024年，缺乏最新LLM进展；缺乏统一标准评测基准与理论性能保证；难以覆盖所有应用场景，仍需进一步实证验证与理论研究。

---

## 50. Toward Reliable Tea Leaf Disease Diagnosis Using Deep Learning Model: Enhancing Robustness With Explainable AI and Adversarial Training

**arXiv ID:** 2602.11239 | [PDF](https://arxiv.org/pdf/2602.11239v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition`

---

## 51. HoloBrain-0 Technical Report

**arXiv ID:** 2602.12062 | [PDF](https://arxiv.org/pdf/2602.12062v1)

**作者:** Xuewu Lin `[一作]` (Horizon Robotics), Zhizhong Su `[通讯]` (Horizon Robotics)

**通讯引用:** 2824 | [OpenAlex ID](https://openalex.org/A5087325725)

**关键词:** `Robotics` `Robotic Intelligence` `Vision-Language-Action Model` `Diffusion model` `Video` `Multimodality`

**🎯 论文内容**

提出了全新的 Vision‑Language‑Action（VLA）框架，结合机器人体现先验（多视角相机参数、运动学描述）实现跨机器人、跨场景的高效3D空间推理与控制。

**💡 创新点**

核心创新点包括：1) 通过 Spatial Enhancer 将多视图2D特征投影到统一3D坐标系；2) Joint‑Graph Attention 的 Embodiment‑Aware Action Expert 用关节6D位姿替代角度，兼容不同机械臂与移动平台；3) 结合 SimpleRTC 与 Teacher‑Forcing 的“pre‑train‑then‑post‑train”数据策略，实现低延迟异步推理与数据驱动的迭代收集。

**🔧 技术方法**

技术手段主要有：Vision‑Language 模块（GroundingDINO 或 Qwen2.5‑VL），深度空间投影与3D语义融合，基于扩散模型的动作生成，联合损失（关节、位姿、正向运动学、深度），以及开放源码的 RoboOrchard 全栈基础设施。

**📊 数据集**

使用的训练与测试数据涵盖：156M帧、3500小时的多机器人演示（Dual‑arm Piper、Agibot、Franka 等）、模拟数据（RoboTwin 2.0、LIBERO、GenieSim 等），以及人类视频数据（EgoDex）和“Grasp Anything”专门收集的数据。

**📈 对比分析**

与现有 VLA 基线（π_0、π_0.5、X‑VLA、LingBot‑VLA 等）对比，-GD（0.2B）和-QW（1.1B）在 RoboTwin 2.0 随机化设置下分别达到 90.8% 与 92.3% 成功率；在 LIBERO‑Plus 零样本设置下平均得分 74.0%，超过前沿 69.6%；在 GenieSim 2.2 任务中整体得分 4.685，超越 X‑VLA 4.541；在真实世界的 10 项任务中平均成功率提升 5–8% 以上，并通过“Grasp Anything”共训练显著提升基础抓取与搬运任务。

**⚠️ 局限性**

局限性：1) 依赖多视角相机与完整 URDF，单摄像头或缺失运动学信息时性能下降；2) 训练与后期微调仍需大量标注演示，虽然数据策略降低成本但收集成本高；3) 主要验证于桌面/移动机器人和模拟，进一步跨平台（如全身机器人或极端动态环境）验证仍待深入。

---

## 52. Toward Adaptive Non-Intrusive Reduced-Order Models: Design and Challenges

**arXiv ID:** 2602.11378 | [PDF](https://arxiv.org/pdf/2602.11378v1)

**作者:** Amirpasha Hedayat `[一作]` (University of Michigan), Karthik Duraisamy `[通讯]` (University of Michigan)

**通讯引用:** 5233 | [OpenAlex ID](https://openalex.org/A5035881605)

**关键词:** `Machine Learning` `Optimization` `Time Series`

**🎯 论文内容**

提出并实现三种自适应非侵入式降阶模型（Adaptive OpInf、Adaptive NiTROM 与混合方法），并在二维驱动腔流中评估其在线自适应性能。

**💡 创新点**

创新点在于将传统的 OpInf 与 NiTROM 结合，提出既可快速回归又可对基空间和动力学做几何一致优化的混合框架；同时提供统一的在线窗口、适配间隔与优化步数的设计准则。

**🔧 技术方法**

核心技术包括：非侵入式回归（OpInf）与基空间重投影；Riemannian 基空间与动力学联合优化（NiTROM）；增量/窗口化奇异值分解进行基更新；以及在在线阶段与 FOM 交互以获取新快照。

**📊 数据集**

使用的典型数据集是 Re=8300 的二维驱动腔流（全阶模型约 2×10⁴ 状态维度），通过不同训练窗口（丰富训练、转移变化、最小训练）构造测试情景。

**📈 对比分析**

对比方法包括静态 Galerkin、静态 OpInf、静态 NiTROM 以及三种自适应模型；在能量曲线、场误差与速度剖面上评估。结果显示：静态模型随时间出现能量漂移；Adaptive OpInf 在稳定性与计算成本上表现稳健；Adaptive NiTROM 在频繁更新时能精确跟踪能量；混合方法在所有三种测试场景下均能保持能量界定、相位与振幅一致，是最优方案。

**⚠️ 局限性**

主要局限包括：NiTROM 的 Riemannian 优化开销大、对适配间隔与优化步数高度敏感；所有方法均需在每个适配周期调用一次全阶模型快照，若全阶求解昂贵则限制实时部署；在极端训练不足或快速转移情景下，纯 OpInf 可能出现振幅衰减；目前验证仅在相对简单的驱动腔流上，需要在更高维、非线性更强的流动问题上进一步验证。

---

## 53. Cross-Modal Robustness Transfer (CMRT): Training Robust Speech Translation Models Using Adversarial Text

**arXiv ID:** 2602.11933 | [PDF](https://arxiv.org/pdf/2602.11933v1)

**作者:** Abderrahmane Issam `[一作]` (Maastricht University), Gerasimos Spanakis `[通讯]` (Maastricht University)

**通讯引用:** 754 | [OpenAlex ID](https://openalex.org/A5010354377)

**关键词:** `Computation and Language` `Adversarial Attack` `Domain Adaptation` `Representation Learning` `Transformer` `Contrastive Learning` `Audio` `Text` `Multimodality`

**🎯 论文内容**

提出CMRT框架，利用共享语义空间中对齐的语音与文本表示，在对抗微调阶段直接注入文本对抗嵌入，提升E2E-ST模型对形态变化的鲁棒性，并改造MORPHEUS为Speech-MORPHEUS进行评估。

**💡 创新点**

通过在共享语义空间实现语音与文本的对比学习与混合训练，随后在对抗微调中仅使用文本对抗样本实现跨模态鲁棒性迁移，完全避免合成对抗语音数据的高昂成本。

**🔧 技术方法**

使用HuBERT/mHuBERT编码器、Transformer 编解码器、Word‑Aligned Contrastive Learning（WACO）、Mixup训练、KL散度正则化、对抗微调（CMRT‑FN）以及Speech‑MORPHEUS形态扰动生成技术。

**📊 数据集**

以CoVoST 2四个方向（En‑De、En‑Ca、En‑Ar、Fr‑En）为主的数据集，并在训练集上生成50k个Speech‑MORPHEUS对抗样本用于实验。

**📈 对比分析**

与MT‑Transformer、HuBERT‑Transformer、HuBERT‑CMOT、TTS‑Morpheus‑FN等基线相比，CMRT‑FN在Morpheus攻击下平均提升约3.4 BLEU点，且在原始CoVoST 2测试集上仅下降0.6 BLEU，整体性能优于使用合成对抗语音的基线。

**⚠️ 局限性**

需要两阶段训练，第一阶段对齐语音与文本会导致原始测试集性能略低；跨模态对齐尚未完全消除模态特异性信息，限制了鲁棒性提升空间。

---

## 54. Text2GQL-Bench: A Text to Graph Query Language Benchmark [Experiment, Analysis & Benchmark]

**arXiv ID:** 2602.11745 | [PDF](https://arxiv.org/pdf/2602.11745v1)

**作者:** Songlin Lyu `[一作]` (Ant Group), Chuntao Hong `[通讯]` (Ant Group)

**关键词:** `Artificial Intelligence` `Large Language Model` `Graph` `Text` `Benchmark`

**🎯 论文内容**

提出了 Text2GQL-Bench，面向多领域、多图查询语言（ISO‑GQL、Cypher）的文本到图查询生成基准，并给出了可扩展的数据构建与评估框架。

**💡 创新点**

创新点包括①将多种现有文本‑SQL/文本‑Cypher 资源迁移至图查询场景，构造 34 个跨 13 个领域的图数据库；②引入三层问句抽象（语法、逻辑、业务）和四级难度等级；③基于 Graph‑IR 的统一数据合成流程；④设计多维度评估（语法有效性、语义对齐、相似度、执行准确率），可细粒度揭示模型瓶颈。

**🔧 技术方法**

技术手段包括 LLM‑驱动的 schema/数据/查询生成、代码化的数据合成、Graph‑IR 语义转换、层级问句生成与“逆推理”外部知识、执行级验证、自动与人工混合 QA，以及综合评估指标的实现。

**📊 数据集**

使用了多来源的数据：Neo4j Text2Cypher、FinBench、BIRD（Text‑SQL）等，经过迁移/合成得到 34 个图数据库（共 178,184 个 (问句, 查询) 对，22,273 条已验证样本），覆盖 13 个业务领域。

**📈 对比分析**

评估方法：在 ISO‑GQL（Spanner Graph）和 Cypher（TuGraph）上执行 3‑shot/零样本/LoRA 微调的多模型实验（Claude、Qwen、GPT、text2cypher‑gemma 等），记录执行准确率（EX）、语法有效率、GLEU 与相似度。结果显示：零样本 EX <4%；少量示例可提升至 ~50%；微调后 8B 模型 EX 45%，语法有效率 90%+，显著接近大型专有模型。跨语言迁移仍存在显著性能下降。

**⚠️ 局限性**

局限性：零样本性能低；跨 GQL 迁移仍受限；高抽象层级下意图-模式匹配难度大；语法错误仍占主要错误来源；需要人工审核保证质量；目前仅覆盖 ISO‑GQL 与 Cypher，其他 GQL 的支持有限；数据合成可能导致 LLM 幻觉，需要更完善的验证机制。

---

## 55. TIP: Resisting Gradient Inversion via Targeted Interpretable Perturbation in Federated Learning

**arXiv ID:** 2602.11633 | [PDF](https://arxiv.org/pdf/2602.11633v1)

**作者:** Jianhua Wang `[一作]` (Taiyuan University of Technology), Yinlin Su `[通讯]`

**关键词:** `Machine Learning` `Federated Learning` `Safty and Privacy` `Explainability and Interpretability` `Adversarial Attack` `Convolutional Neural Network` `Gradient-weighted Class Activation Mapping` `Differential Privacy` `Image`

**🎯 论文内容**

提出了针对联邦学习梯度反演攻击的 Targeted Interpretable Perturbation（TIP）防御框架

**💡 创新点**

创新点在于结合 Grad‑CAM 识别重要卷积通道并在频域高频成分上注入 DP‑校准噪声，既削弱重构细节又保留低频语义

**🔧 技术方法**

使用了梯度加权类激活映射（Grad‑CAM）、二维离散傅里叶变换（DFT）、高频掩模、差分隐私噪声注入及 FedAvg 联邦训练

**📊 数据集**

在 CIFAR‑10、CIFAR‑100 与 Tiny‑ImageNet 这三个标准图像分类数据集上进行实验

**📈 对比分析**

与无 DP、标准 DP、APG 防御以及 IG、GIFD 反演攻击进行对比；TIP 在保持模型准确率与无私有化方案相近的同时，显著抑制了攻击重构质量（SSIM、PSNR 低于攻击结果，接近 DP）

**⚠️ 局限性**

局限性包括对通道重要性比例、频域阈值等超参数敏感，且在极大模型或非卷积网络结构下的适用性尚待验证

---

## 56. Improved state mixing in higher-order and block diagonal linear recurrent networks

**arXiv ID:** 2602.12021 | [PDF](https://arxiv.org/pdf/2602.12021v1)

**作者:** Igor Dubinin `[一作]` (Ernst Strüngmann Institut), Felix Effenberger `[通讯]` (Ernst Strüngmann Institut)

**通讯引用:** 304 | [OpenAlex ID](https://openalex.org/A5061784202)

**关键词:** `Machine Learning` `Compression` `Recurrent Neural Network` `Sequential` `Text`

**🎯 论文内容**

设计并实现了两种结构化的线性循环网络——高阶线性递归单元（H-LRU）和块对角线递归单元（BD-LRU），并在多种序列任务上进行评估。

**💡 创新点**

创新点在于：①引入多阶差分递归和块对角线混合，从而在保持线性效率的同时显著提升状态混合表达能力；②提出联合 L1 归一化的门控策略，实现动态稳定性和可扩展性；③利用并行扫描实现高效序列处理，兼顾高表达力与并行度。

**🔧 技术方法**

使用的技术包括：高阶差分递归、块对角线矩阵、选择性门控、L1 归一化、并行扫描（Blelloch scan）实现的高效前向/反向传播、参数化激活函数（softmax、sigmoid、ReLU）以及多种实验平台（PyTorch + GPU）。

**📊 数据集**

数据集涵盖：合成任务（压缩、选择性复制、上下文回忆、排列组合 S2–S5），以及实际语言建模任务 FineWeb（约10B tokens）。

**📈 对比分析**

与 LSTM、Mamba、Deltanet、DeltaProduct 等基线对比，BD-LRU 在大多数任务上实现了与 LSTM 相当或更优的准确率/困惑度，参数效率更高；H-LRU 在压缩任务中展示出极佳的参数效率，但在复杂排列任务中表现受限。相较于传统线性模型，二者在效率–表达力平衡上显著优于仅使用对角线或低秩变换的结构。

**⚠️ 局限性**

局限性包括：①大块尺寸（m>16）会导致并行吞吐量下降，影响极长序列处理；②H-LRU 由于结构约束在非交换排列任务上表达力不足；③两种模型仍受限于线性框架，无法捕获高度非线性依赖；④需要额外的门控归一化与并行扫描实现，增加实现复杂度。

---

## 57. LAMP: Implicit Language Map for Robot Navigation

**arXiv ID:** 2602.11862 | [PDF](https://arxiv.org/pdf/2602.11862v1)

**作者:** Sibaek Lee `[一作]` (Sungkyunkwan University), Sunwook Choi `[通讯]` (NAVER LABS)

**关键词:** `Robotics` `Robotic Intelligence` `Vision Language Model` `Neural Radiance Field` `Image`

**🎯 论文内容**

提出LAMP——一种隐式语言映射框架，用RGB图像学习连续语言字段以实现零样本机器人导航；

**💡 创新点**

创新点在于将语言特征作为隐式神经场而非显式存储，并结合稀疏图进行两阶段路径规划；引入vMF贝叶斯不确定性建模和基于语义敏感度的节点采样策略；

**🔧 技术方法**

采用视觉语言模型CLIP、NeRF风格的MLP、von Mises–Fisher分布、Adam梯度优化、A*搜索和图采样技术；

**📊 数据集**

使用NVIDIA Isaac Sim的City Tower Demo 3D模型（1.6km×1.8km）以及真实多层建筑的机器人采集的RGB位姿对；

**📈 对比分析**

与稠密/稀疏网格及节点方法比较，在相同或更少内存下，LAMP在成功率、SPL和目标距离上显著优于对比方法，且单查询时延<1s；

**⚠️ 局限性**

局限性包括对VLM的依赖，目标视觉相似性高时易误选节点，且目标外观弱或模糊时性能下降。

---

## 58. Arbitrary Ratio Feature Compression via Next Token Prediction

**arXiv ID:** 2602.11494 | [PDF](https://arxiv.org/pdf/2602.11494v1)

**作者:** Yufan Liu `[一作]` (Chinese Academy of Sciences), Stephen Maybank `[通讯]` (University of London)

**关键词:** `Computer Vision and Pattern Recognition` `Compression` `Retrieval` `Transformer` `Mixture of Experts` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出了Arbitrary Ratio Feature Compression (ARFC)框架，实现任意压缩比下的特征压缩，消除传统方法需要针对每个压缩比训练多模型的弊端。

**💡 创新点**

核心创新包括：①使用自回归Next-Token Prediction的ARC模块，可通过截断生成的token实现任意压缩比；②Mixture of Solutions (MoS)模块通过多解注意力融合多视角压缩结果；③Entity Relation Graph Constraint (ERGC)在训练中约束实体关系，保持语义与结构一致性。

**🔧 技术方法**

技术方法主要包括Transformer自回归模型、多解注意力(MoS)、图约束损失(ERGC)、多解解码器、Beta分布动态采样的Progressive Compression Training Scheme。

**📊 数据集**

使用的公开数据集包括Flickr30K（英中版）进行跨模态检索；CIFAR10/100、ImageNet、DTD、EuroSAT、FER、FGVC、KITTI、MNIST、PC、VOC等图像分类数据集；CUB-200-2011和Cars-196进行图像检索。

**📈 对比分析**

与AutoEncoder、Q-Former、Post-Training Quantization（PTQ）等方法在相同压缩比下对比，ARFC在跨模态检索、图像分类、图像检索等任务均取得领先或匹配原始特征的性能，甚至在50%压缩比时超越未压缩特征；在高压缩比（75%）时仍保持或超过基线。

**⚠️ 局限性**

限制主要体现在：训练时间相对传统单比压缩模型略长；对比学习的Beta分布参数需要手工调节；在极端高压缩比下可能仍存在信息损失；目前仅在图像与文本两模态验证，需进一步扩展至音频、视频等多模态。

---

## 59. Move What Matters: Parameter-Efficient Domain Adaptation via Optimal Transport Flow for Collaborative Perception

**arXiv ID:** 2602.11565 | [PDF](https://arxiv.org/pdf/2602.11565v1)

**作者:** Zesheng Jia `[一作]` (Soochow University), Jianping Wang `[通讯]` (City University of Hong Kong)

**通讯引用:** 14430 | [OpenAlex ID](https://openalex.org/A5100356291)

**关键词:** `Computer Vision and Pattern Recognition` `Domain Adaptation` `Autonomous Driving` `Point Cloud`

**🎯 论文内容**

针对车辆与基础设施协同感知中的跨域适配问题，提出一种参数高效的Adaptation框架 FlowAdapt，能在仅更新约1%可训练参数的情况下快速适配新环境。

**💡 创新点**

创新点包括：① 将跨域适配建模为最优传输问题；② 通过 Wasserstein Greedy Sampling（WGS）在时空特征空间中主动去除冗余帧；③ 采用 Progressive Knowledge Transfer（KTPro）在网络层次间压缩并注入早期特征，缓解深层语义衰减。

**🔧 技术方法**

技术手段包括：最优传输理论、Wasserstein 距离、贪心覆盖、压缩注入机制、双路径适配器、协同代理提示、基于深度学习的协同感知模型。

**📊 数据集**

实验基准覆盖三大协同感知数据集：OPV2V（仿真车辆-车辆）、DAIR‑V2X（真实车辆‑基础设施）、V2XSet（混合仿真）。

**📈 对比分析**

与 CoPEFT、MACP、DUSA 等主流 PEFT 方法以及全量微调进行对比；在 OPV2V→DAIR‑V2X 的 10% 标签比例下，FlowAdapt 的 AP@50/70 达到 0.667/0.507，显著高于 CoPEFT 的 0.627/0.434，参数量仅为 1%；在不同噪声、融合方式和目标域下同样保持领先优势，且鲁棒性更好。

**⚠️ 局限性**

局限性：① 目前仅验证于 LiDAR 视觉感知；② 对极端域迁移（如天气变化、传感器失效）仍需进一步评估；③ 需要手工调节 WGS 权重和压缩比等超参，适配过程仍具一定复杂度。

---

## 60. RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis

**arXiv ID:** 2602.11506 | [PDF](https://arxiv.org/pdf/2602.11506v1)

**作者:** Zhen Bi `[一作]` (Huzhou University), Cheng Deng `[通讯]` (Banbu AI Foundation)

**关键词:** `Machine Learning` `Large Language Model` `Text` `Benchmark`

**🎯 论文内容**

构建了基于 Roofline 模型的 On-Device LLM 评测框架 RooflineBench，并提出了相对推理潜能 (Relative Inference Potential) 指标。

**💡 创新点**

创新点包括统一的 OI 驱动评测、相对推理潜能指标、对模型深度、序列长度、注意力架构与量化的全方位分析。

**🔧 技术方法**

采用 Roofline 模型、经验测得的 P_peak 与 BW_peak、算子层级 FLOPs 与内存流估计，以及多序列模式 (SISO/SILO/LISO/LILO) 进行推理性能剖析。

**📊 数据集**

使用了多种小型 LLM（如 Qwen2.5‑Instruct、Llama‑3.2‑Instruct、PLM‑Instruct、SmolLM2‑Instruct）在 Apple M1 Pro、RTX 3090、RTX 3070Ti、Jetson Orin Nano、Raspberry Pi 5 等硬件上的实际推理实验。

**📈 对比分析**

通过比较不同模型、不同注意力机制 (MHA/GQA/MLA) 与不同精度 (FP16/Q8/Q4) 的 OI 与实际吞吐率，验证了相对推理潜能指标能客观衡量同一硬件下的效率差异，并发现序列长度和模型深度是主导因素。

**⚠️ 局限性**

局限性在于仅关注推理阶段，未覆盖训练负载；测评基于有限模型与硬件，可能不适用于更大规模或特殊芯片；且精确度受计数器与量化实现细节影响。

---

## 61. AI-Driven Clinical Decision Support System for Enhanced Diabetes Diagnosis and Management

**arXiv ID:** 2602.11237 | [PDF](https://arxiv.org/pdf/2602.11237v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## 62. Compositionality of Systems and Partially Ordered Runs

**arXiv ID:** 2602.11203 | [PDF](https://arxiv.org/pdf/2602.11203v1)

**作者:** Peter Fettke `[一作]` (German Research Center for Artificial Intelligence), Wolfgang Reisig `[通讯]` (Humboldt-Universität zu Berlin)

**通讯引用:** 8861 | [OpenAlex ID](https://openalex.org/A5035539953)

**关键词:** `Logic in Computer Science`

**🎯 论文内容**

提出了基于 Petri 网模块和部分有序运行的组合框架，并证明组合后运行等于各子网运行的组合。

**💡 创新点**

首次给出带双面接口且满足结合律的 Petri 网模块组合算子，实现运行的组合性。

**🔧 技术方法**

使用 Petri 网模块、步骤、运行的抽象定义，结合递归构造和组合算子进行理论分析。

**📊 数据集**

未使用具体数据集，主要为理论推导和证明。

**📈 对比分析**

无实验或性能对比，本文仅给出数学证明。

**⚠️ 局限性**

示例中主要处理无分支场所，复杂分支网的实际表现尚未评估。

---

## 63. Predicting LLM Output Length via Entropy-Guided Representations

**arXiv ID:** 2602.11812 | [PDF](https://arxiv.org/pdf/2602.11812v1)

**作者:** Huanyi Xie `[一作]` (King Abdullah University of Science and Technology), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 4016 | [OpenAlex ID](https://openalex.org/A5100401482)

**关键词:** `Artificial Intelligence` `Generation` `Reinforcement Learning from Human Feedback` `Optimization` `Transformer` `Large Language Model` `Reinforcement Learning` `Chain-of-Thought` `Text` `Sequential` `Benchmark`

**🎯 论文内容**

本文提出了一种利用大型语言模型内部激活状态进行序列长度预测的新框架，主要包括静态预测模块Entropy-Guided Token Pooling (EGTP)和动态预测模块Progressive Length Prediction (PLP)。

**💡 创新点**

创新点在于：①不再依赖外部轻量预测器，而是直接重用LLM自身的隐藏状态；②通过熵导向的加权池化聚合关键信息，提高预测精度；③采用软标签分布的回归方法兼顾回归与分类的优势；④针对强化学习中的“一对多”采样引入逐步预测，解决随机生成长度变化的问题。

**🔧 技术方法**

技术主要包括：熵计算与加权注意力池化、基于soft‑label的分段回归损失、逐步长度预测与自回归更新、以及与长度感知调度器的集成。

**📊 数据集**

使用了公开的LongBench、ZeroSCROLLS、IFEval等长文本/推理数据集以及自构建的ForeLen基准（包含长序列、Chain-of-Thought和RL采样场景），并在Qwen2.5、Llama3.2等多种LLM上进行评估。

**📈 对比分析**

与SSJF-Reg、SSJF-MC、S3、PiA、TPV、TRAIL、LTR‑C等现有预测方法相比，EGTP在ForeLen基准上MAE下降约29%，在最难的RL采样任务中降至约19%；集成长度感知调度后，系统吞吐量提升3倍、padding比例下降至0.18（相较于TRAIL的0.51）。

**⚠️ 局限性**

局限性包括：①目前方法仅适用于自回归LLM，非自回归模型难以直接复用隐藏状态；②对极端长文本或特殊生成策略的泛化能力仍待进一步验证；③需要在每个token上计算熵，虽然开销小，但在极大模型规模或极低延迟场景下仍可能产生额外负担。

---

## 64. Markovian protocols and an upper bound on the extension complexity of the matching polytope

**arXiv ID:** 2602.11382 | [PDF](https://arxiv.org/pdf/2602.11382v1)

**作者:** M. Szusterman `[一作]` (Centre de Mathématiques Laurent Schwartz), M. Szusterman `[通讯]` (Centre de Mathématiques Laurent Schwartz)

**关键词:** `Discrete Mathematics` `Non-negative Rank Theory` `Markovian Communication Protocol` `Branching Program` `Sorting Network`

**🎯 论文内容**

本文通过将非负分解与随机通信协议对应，提出了使用Markovian协议宽度来刻画多面体扩展复杂度，并基于此为匹配多面体给出了新的上界，随后用基于排序网络的一轮协议恢复了Permutahedron的紧凑扩展。

**💡 创新点**

创新点在于：①引入Markovian协议宽度与扩展复杂度的等价性；②利用此框架得到匹配多面体上界 O(n³·1.5ⁿ)；③将排序网络与通信协议结合，给出Permutahedron的 O(q)（q≈nlogn）扩展；④在理论上统一了非负分解、沟通复杂度与扩展复杂度的视角。

**🔧 技术方法**

采用了非负秩理论、滑动矩阵、Yannakakis 的对应定理、Markovian通信协议、分支程序、排序网络、Lovász 关键子集构造等技术。

**📊 数据集**

没有使用实验数据集，全部为理论分析与构造证明。

**📈 对比分析**

与已有的 2^n 上界比较，匹配多面体的上界显著下降；Permutahedron 的上界与已知的最优 lower bound（Ω(n ln n)）相匹配，证明了该协议宽度的最小性猜想；整体性能在理论上优于传统方法。

**⚠️ 局限性**

局限性：对匹配多面体的上界尚未证明最优，可能存在更紧的构造；Permutahedron 的协议仍依赖于存在大小为 O(n ln n) 的排序网络，若排序网络最小化未知则上界不一定最优；论文未给出下界，且协议实现复杂度较高。

---

## 65. Peak + Accumulation: A Proxy-Level Scoring Formula for Multi-Turn LLM Attack Detection

**arXiv ID:** 2602.11247 | [PDF](https://arxiv.org/pdf/2602.11247v1)

**作者:** J Alex Corll `[一作]` (Independent Researcher), J Alex Corll `[通讯]` (Independent Researcher)

**关键词:** `Cryptography and Security` `Adversarial Attack` `Prompt Engineering` `Text`

**🎯 论文内容**

本文提出一种基于代理层的多轮提示注入攻击评分公式，能在不调用LLM的情况下将每轮模式分数聚合为会话级风险分数。

**💡 创新点**

创新点在于引入峰值+累计（peak + accumulation）评分方法，纠正了加权平均的天花板效应，并通过持久性因子实现对攻击持续性的显式放大，且发现了持久性参数的阶跃转变。

**🔧 技术方法**

技术上主要使用正则模式匹配、累积式评分（类似CUSUM）、峰值与持久率与类别多样性的加权组合，并在Rust实现中提供了高效、确定性的计算。

**📊 数据集**

实验采用WildJailbreak生成的588条攻击会话、WildChat的10,000条真实安全会话以及若干人工设计的攻击/安全案例，构成总计10,654条多轮对话数据集。

**📈 对比分析**

与传统加权平均基线相比，该公式在同一数据集上实现90.8%召回率、1.20%误报率、85.9%的F1得分，表现显著提升。

**⚠️ 局限性**

局限性包括：仅能捕捉基于正则模式的注入文本，无法识别语义上隐蔽或内容安全类的攻击；数据集多为合成结构，缺乏真实多轮攻击轨迹；正则模式易被绕过，需进一步提升模式鲁棒性。

---

## 66. Response-Based Knowledge Distillation for Multilingual Jailbreak Prevention Unwittingly Compromises Safety

**arXiv ID:** 2602.11157 | [PDF](https://arxiv.org/pdf/2602.11157v1)

**作者:** Max Zhang `[一作]`, Haihao Liu `[通讯]` (Algoverse Air Research)

**关键词:** `Computation and Language` `Safty and Privacy` `Knowledge Distillation` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

本文探讨了在多语言环境下使用响应式知识蒸馏（基于 LoRA 的参数高效微调）来提升大语言模型的安全性，实验结果表明安全性能反而下降。

**💡 创新点**

创新点在于首次系统评估响应式知识蒸馏在多语言 jailbreak 防御中的效果，并通过分析数据边界、教师漏洞放大及灾难性遗忘等因素解释安全恶化，同时提出了预处理“boundary”数据的可行性。

**🔧 技术方法**

使用了响应式知识蒸馏技术结合 LoRA PEFT，对学生模型进行安全拒绝行为的迁移学习。

**📊 数据集**

蒸馏数据来自 XSafety 的约 28,000 条多语言 jailbreak 提示，评估数据使用 MultiJail 的 3,150 条提示。

**📈 对比分析**

与基线学生模型及教师模型比较，学生模型的 Jailbreak Success Rate 上升，最高可达 16.6 个百分点；移除 boundary 数据后部分模型安全性可逆转，但推理性能（GSM8K）整体下降。

**⚠️ 局限性**

局限包括仅使用响应式黑盒 KD、模型规模仅到 8B（即使 14B 也类似）、评估依赖单一 LLM‑as‑Judge，并未尝试 logits 蒸馏或更大模型。

---

## 67. Neutral Prompts, Non-Neutral People: Quantifying Gender and Skin-Tone Bias in Gemini Flash 2.5 Image and GPT Image 1.5

**arXiv ID:** 2602.12133 | [PDF](https://arxiv.org/pdf/2602.12133v1)

**作者:** Roberto Balestri `[一作]` (Università di Bologna), Roberto Balestri `[通讯]` (Università di Bologna)

**关键词:** `Artificial Intelligence` `Generation` `Image`

**🎯 论文内容**

本文通过对 Gemini Flash 2.5 Image 与 GPT Image 1.5 在四个语义中性提示下生成 3,200 张照片级图像，并利用面部特征检测与肤色测量管线，量化了性别与肤色的默认偏差；

**💡 创新点**

创新点在于将光照可见的色彩归一化与基于面部标记的肤色掩膜结合，使用 CIELAB 空间和多尺度（Monk、PERLA、Fitzpatrick）映射，对生成图像的肤色进行精细、可比的定量评估；

**🔧 技术方法**

使用的技术包括深度面部检测（DeepFace）、面部 468 点标记（MediaPipe Face Mesh）、CLAHE 对比度增强、背景参照白平衡、k‑means 聚类肤色、CIELAB 颜色距离（ΔE₀₀）以及统计检验（χ²、Mann‑Whitney U、t‑test）；

**📊 数据集**

使用的数据集为自生成的 3,200 张图像，按 4 个中性提示平均分配到两款模型；

**📈 对比分析**

比较方法为对两模型在性别、种族、年龄及肤色尺度（MST、PERLA、FST）的分布进行交叉表和分布差异检验，结果显示两模型在性别默认和肤色偏好上表现出显著、相反的差异（如 NanoBanana 女性占 93.7%，GPT 男性占 70.6%）；

**⚠️ 局限性**

局限性包括：仅检验中性提示，未探究有针对性多样性提示；使用的自动化属性识别工具仅反映感知属性；肤色测量可能受化妆与光照风格影响；模型版本随时更新，结果可能变化。

---

## 68. In-Context Function Learning in Large Language Models

**arXiv ID:** 2602.11863 | [PDF](https://arxiv.org/pdf/2602.11863v1)

**作者:** Elif Akata `[一作]` (Helmholtz Munich), Eric Schulz `[通讯]` (Helmholtz Munich)

**关键词:** `Machine Learning` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning`

**🎯 论文内容**

本文通过在大型语言模型（LLM）上进行控制实验，研究其在上下文学习（ICL）中对连续函数的学习能力，并以高斯过程（GP）为框架评估学习曲线、诱导偏差与后期训练的效果。

**💡 创新点**

创新点在于将GP理论与LLM ICL结合，提出基于似然的诱导偏差分析方法，并展示参数高效的后期训练（SFT 与 GRPO）能够显著调整LLM的隐式先验，从而提升样本效率。

**🔧 技术方法**

主要技术包括：高斯过程回归基线与1‑NN上限比较、似然诱导偏差分析、QLoRA低秩适配器、监督微调（SFT）与强化学习（GRPO）以及多核函数（Matérn 1/2、Squared Exponential）下的函数生成。

**📊 数据集**

使用合成数据集：从已知GP核（Matérn 1/2 与 Squared Exponential）抽样的多维（1–4维）标量函数，涵盖 200 条函数实例；在 Qwen‑3、Llama、Gemma、Mistral 等多模型族上进行评测。

**📈 对比分析**

通过将LLM在给定演示数下的平均绝对误差与 GP 回归的经验下界及 1‑NN 的期望上界进行对比，结果显示LLM误差随演示数增大逐渐逼近 GP 下界，且模型规模越大学习曲线越陡峭；后期训练可将误差进一步逼近下界，其中强化学习优于监督微调。

**⚠️ 局限性**

局限性包括：诱导偏差分析易受方差膨胀影响；实验仅限于合成函数，缺乏对真实任务的迁移评估；仅考察有限的核函数与低维度；后期训练虽有效但仅验证了少数模型与方法。

---

## 69. MalTool: Malicious Tool Attacks on LLM Agents

**arXiv ID:** 2602.12194 | [PDF](https://arxiv.org/pdf/2602.12194v1)

**作者:** Yuepeng Hu `[一作]` (Duke University), Neil Gong `[通讯]` (Duke University)

**通讯引用:** 7756 | [OpenAlex ID](https://openalex.org/A5009102659)

**关键词:** `Cryptography and Security` `Adversarial Attack` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Text`

**🎯 论文内容**

本文系统研究了LLM Agent工具生态中的恶意工具攻击，构建了自动生成恶意工具的框架并评估其有效性。

**💡 创新点**

创新点包括：基于CIA三元组的恶意工具行为分类；引入Verifier实现功能正确性与结构多样性验证；生成Standalone与Trojan恶意工具的两大数据集。

**🔧 技术方法**

技术实现上结合了多种编码LLM（如GPT-OSS-20B、Phi-4、Qwen3-Coder-30B、GPT-4o等）与AST相似度的自动验证器，利用系统提示引导多样化代码生成。

**📊 数据集**

使用了1,200个独立恶意工具、5,287个Trojan恶意工具以及10,573个真实开源工具（Dataset III）作为实验与评估数据集。

**📈 对比分析**

与现有商业和专用检测器（VirusTotal、腾讯A.I.G、Cisco MCP、AntGroup MCPScan）对比，自动生成框架在攻击成功率（ASR）始终为1.0，显著提升了多样性并降低验证成本；检测器在FNR和FPR上表现差距大，难以可靠识别恶意工具。

**⚠️ 局限性**

局限性包括：仅覆盖有限的恶意行为类型，未涵盖多阶段或长周期攻击；评估依赖于合成测试环境，真实威胁场景可能更复杂；即使安全对齐模型，生成框架仍可产生可执行恶意工具。

---

## 70. Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?

**arXiv ID:** 2602.11988 | [PDF](https://arxiv.org/pdf/2602.11988v1)

**作者:** Thibaud Gloaguen `[一作]` (ETH Zurich), Martin Vechev `[通讯]` (ETH Zurich)

**通讯引用:** 11122 | [OpenAlex ID](https://openalex.org/A5069901599)

**关键词:** `Software Engineering` `AI Code Assistant` `Large Language Model` `Text` `Benchmark`

**🎯 论文内容**

评估了仓库级上下文文件（如 README、docs）对自动编码代理完成任务的影响，构建了新的基准数据集并进行实验；

**💡 创新点**

创新点在于首次系统性地比较自动生成与开发者编写的上下文文件对多种编码代理性能的实际影响，并提出仅包含最小化需求的上下文更有利；

**🔧 技术方法**

使用了多款主流编码代理（如 OpenAI、DeepSeek 等）与对应的 LLM 模型，通过对工具调用、推理步骤与成本进行量化分析；

**📊 数据集**

采用了两类数据集：SWE‑Bench（300 题）以及新构建的 138 条来自 12 个包含开发者上下文文件的 Python 仓库的 Issue‑Patch 组合；

**📈 对比分析**

对比方法是设置三种场景（无上下文、LLM 生成上下文、开发者提供上下文），衡量成功率、步骤数与推理成本；结果显示 LLM 生成上下文平均降低 0.5%–2% 的成功率、增加 20%–23% 的成本，开发者上下文略有提升但仍增加步骤；

**⚠️ 局限性**

限制在于仅聚焦 Python 语言、依赖公开仓库，且未覆盖更专业的安全或性能评估，且 LLM 生成上下文的质量仍需提升。

---

## 71. Bandit Learning in Matching Markets with Interviews

**arXiv ID:** 2602.12224 | [PDF](https://arxiv.org/pdf/2602.12224v1)

**作者:** Amirmahdi Mirfakhar `[一作]` (University of Massachusetts Amherst), Mohammad Hajiesmaili `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 1030 | [OpenAlex ID](https://openalex.org/A5046146251)

**关键词:** `Computer Science and Game Theory` `Reinforcement Learning`

**🎯 论文内容**

本文在匹配市场中引入访谈与双边不确定性，提出中央与去中心化算法实现时间无关的 regret；

**💡 创新点**

创新点在于允许公司通过战略性延迟招聘来消除误判导致的锁定，并利用访谈作为低成本提示同时兼顾行动约束；

**🔧 技术方法**

技术主要采用带提示的多臂老虎机分析、基于 Gale‑Shapley 的分布式求解、以及在不确定公司上设计的策略性拒绝/延迟策略；

**📊 数据集**

该工作基于理论模型，无实测数据集；

**📈 对比分析**

相较于传统 O(log T) 的稳定匹配学习，所提算法在中心化与去中心化场景均可获得常数（时间无关） regret，证明了访谈与延迟机制的显著加速；

**⚠️ 局限性**

局限性包括对结构化市场（α‑可约）依赖较强，在一般市场下需三次访谈且常数呈指数级，且未讨论信息不对称下的激励兼容问题。

---

## 72. Adaptive-Horizon Conflict-Based Search for Closed-Loop Multi-Agent Path Finding

**arXiv ID:** 2602.12024 | [PDF](https://arxiv.org/pdf/2602.12024v1)

**作者:** Jiarui Li `[一作]`, Gioele Zardini `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Optimization` `Reinforcement Learning` `Tabular` `Benchmark`

**🎯 论文内容**

研究了一种基于有限视野CBS的闭环多机器人路径规划算法 acr:accbs，能够在可用计算预算内动态调整规划视野并实时生成可执行的控制指令。

**💡 创新点**

创新点包括：① 引入“active prefix”概念，实现不同规划视野间的成本不变性；② 证明并利用约束树可复用性，使得视野扩展不需重新搜索；③ 将这些技术整合成一个 anytime 闭环 CBS，兼具理论最优性与实践可行性。

**🔧 技术方法**

技术手段：有限视野 CBS、迭代加深（iterative deepening）策略、active prefix 与成本不变性、约束树重用、基于 MDP/MPCC 的终端代价估计以及冲突检测与分支。

**📊 数据集**

使用 MAPF 基准数据集，包括：empty‑8×8、random‑32×32‑10、empty‑48×48、random‑64×64‑10 四种地图。

**📈 对比分析**

与经典 CBS、ICBS、PIBT 等方法对比：acr:accbs 在相同时间预算下取得接近最优的 SOC（sum‑of‑costs），显著优于 PIBT 的近似解；相较于 CBS，运行时间更短，逼近最优性的中间点，整体性能处于 Pareto 边界附近。

**⚠️ 局限性**

局限性：① 需要人工或预设的时间预算与最大视野参数，预算选择不当会导致性能与计算成本不匹配；② prioritized conflict 机制虽能提升搜索效率，但其计算开销在大规模实例中可能抵消收益；③ 对极端动态环境（频繁目标变更、严重延迟等）尚未深入评估。

---

## 73. A$^{2}$V-SLP: Alignment-Aware Variational Modeling for Disentangled Sign Language Production

**arXiv ID:** 2602.11861 | [PDF](https://arxiv.org/pdf/2602.11861v1)

**作者:** Sümeyye Meryem Taşyürek `[一作]` (Hacettepe University), Hacer Yalim Keles `[通讯]` (Hacettepe University)

**通讯引用:** 818 | [OpenAlex ID](https://openalex.org/A5071978946)

**关键词:** `Machine Learning` `Generation` `Data Synthesis` `Pose Estimation` `Transformer` `Auto Encoder` `Video` `Text`

**🎯 论文内容**

提出一种无 gloss 的手语生成框架 A^2V‑SLP，将文本映射到分解的变分潜在空间，再通过非自回归 Transformer 预测每个发声器（右手、左手、面部、上身）的均值与方差，最后通过预训练的 VAE 解码得到完整的 3D 姿势序列。

**💡 创新点**

创新点包括：①用结构分解的 VAE 产生按发声器的分布式潜在统计（均值与方差）作为监督，避免确定性潜在崩溃；②在非自回归解码器中引入 gloss attention 作为局部对齐先验，替代显式 gloss 标注；③结合动态手部权重和 KL 正则化，提升手部细节与分布拟合。

**🔧 技术方法**

核心技术：结构分解变分自编码器、非自回归 Transformer、BERT 文本嵌入、gloss attention、重参数化、L1 潜在回归、KL 正则化、动态 L1 权重调度。

**📊 数据集**

使用两个公开手语数据集：PHOENIX‑2014T（德语手语）和 CSL‑Daily（中文手语）。

**📈 对比分析**

通过与多种 gloss‑free 与 gloss‑based 方法对比，采用后向翻译 BLEU/ROUGE/chrF 及 DTW‑MJE 评估。在 PHOENIX‑2014T 上 A^2V‑SLP+KL 达到 chrF 35.41、ROUGE 36.64，显著优于其它 gloss‑free 方法且逼近甚至超过部分 gloss‑based 系统；在 CSL‑Daily 上取得最低 DTW 错误，保持与先前分解方法竞争力。

**⚠️ 局限性**

局限性：仍需先训练 VAE，过程分两阶段；对手部细节的提升有限；评估仅覆盖两大数据集，跨语言与跨说话者的泛化尚未验证；模型规模大，推理时仍存在计算负担；对低资源或极端手势变化的适应性不明。

---

## 74. WSBD: Freezing-Based Optimizer for Quantum Neural Networks

**arXiv ID:** 2602.11383 | [PDF](https://arxiv.org/pdf/2602.11383v1)

**作者:** Christopher Kverne `[一作]` (Florida International University), Janki Bhimani `[通讯]` (Florida International University)

**通讯引用:** 570 | [OpenAlex ID](https://openalex.org/A5011474644)

**关键词:** `Machine Learning` `Optimization` `Image`

**🎯 论文内容**

提出了 Weighted Stochastic Block Descent (WSBD)，一种动态、参数级冻结的优化器，用于加速量子神经网络（QNN）的训练。

**💡 创新点**

创新点在于：1) 使用梯度累积的“重要性分数”动态评估每个参数的重要性；2) 在冻结周期结束后按概率随机冻结最不重要的参数，避免确定性冻结导致的过早消失；3) 通过重置活跃参数的分数实现快速适应；4) 证明了在 L‑smooth QNN 下 WSBD 与任何优化器兼容且保持收敛性。

**🔧 技术方法**

技术核心包括 Parameter‑Shift Rule (PSR) 的梯度估计、基于梯度累积的 Sum‑of‑Gradients 重要性度量、随机化参数冻结与分数重置、以及与传统优化器（SGD、Adam 等）的无缝结合。

**📊 数据集**

数据集与任务：MNIST 图像分类（经 PCA 预处理后映射到 4/8/10 qubits）、二进制奇偶性问题（Parity）、以及一维 transverse‑field Ising 模型的变分量子本征求解器（VQE）。

**📈 对比分析**

与 Adam、SGD、SPSA、Nelder–Mead、Bayesian 等基线以及 WSBD 的多种变体进行对比。实验表明：在 4–10 qubit 规模下，WSBD 在 VQE 任务中平均可减少 25–85% 的前向传递；在 MNIST 与 Parity 上保持相同的最终精度的同时，训练所需的前向传递量减少 30–90%；在噪声环境下仍保持显著优势。表格和曲线显示 WSBD 的收敛速度明显快于传统优化器。

**⚠️ 局限性**

局限性包括：1) 仍依赖 PSR 的两次 circuit 评估，导致在参数量极大时计算成本高；2) 冻结策略基于梯度累积，可能在极端噪声或极度平坦的景观下误判重要性；3) 需要手动调节冻结比例 λ_f 与窗口大小 τ，超参数敏感度高；4) 目前仅在模拟器/少量真实硬件上验证，尚未在更大规模实际量子设备上测试。

---

## 75. Perception-based Image Denoising via Generative Compression

**arXiv ID:** 2602.11553 | [PDF](https://arxiv.org/pdf/2602.11553v1)

**作者:** Nam Nguyen `[一作]` (Oregon State University), Bella Bose `[通讯]` (Oregon State University)

**通讯引用:** 1087 | [OpenAlex ID](https://openalex.org/A5108277072)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Compression` `Generative Adversarial Network` `Diffusion model` `Image`

**🎯 论文内容**

提出了基于生成压缩的感知图像去噪框架，提供两种实现：条件 Wasserstein GAN 压缩去噪与基于扩散的压缩去噪，并给出非渐近误差理论上界。

**💡 创新点**

创新点在于将低复杂度压缩与感知生成模型融合，可显式调节 R‑D‑P 权衡；首次提供压缩最大似然去噪的非渐近误差和解码错误概率上界；结合扩散模型进一步提升感知真实度。

**🔧 技术方法**

使用的技术包括条件 Wasserstein GAN、LPIPS 感知损失、熵编码、扩散模型、Wasserstein 距离、信息论 R‑D‑P 理论以及最大似然压缩去噪。

**📊 数据集**

训练基于 OpenImages 数据集；在 Kodak、DIV2K（合成 Gaussian 噪声 σ=15/25/50）、FMD、SIDD（真实噪声）上进行评估。

**📈 对比分析**

与 BM3D、Noise2Clean、Noise2Noise、Deep Decoder、DeCompress 等基线比较，评估指标包括 PSNR/SSIM、LPIPS、FID、PI、DISTS。CGanDeCompress 在 FID/LPIPS 上取得最佳表现，DiffDeCompress 在 PI 上领先；总体感知质量显著提升，像素误差保持竞争力。

**⚠️ 局限性**

限制在于与像素精度之间仍需折中；训练成本高（显存/时间），扩散步骤多导致推理效率低；在极端噪声或复杂噪声模型下仍可能出现过平滑或细节丢失。

---

## 76. Learning to Configure Agentic AI Systems

**arXiv ID:** 2602.11574 | [PDF](https://arxiv.org/pdf/2602.11574v1)

**作者:** Aditya Taparia `[一作]` (Arizona State University), Ransalu Senanayake `[通讯]` (Arizona State University)

**通讯引用:** 2675 | [OpenAlex ID](https://openalex.org/A5043482412)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Optimization` `Computational Efficiency` `Transformer` `Large Language Model` `Reinforcement Learning` `Supervised Fine-Tuning` `Agentic AI` `Prompt Engineering` `Text` `Benchmark`

**🎯 论文内容**

提出了 ARC，一种层次化强化学习框架，用来为每个查询动态配置 LLM 代理系统的工作流、工具、预算与提示，避免统一模板导致的计算浪费与性能瓶颈。

**💡 创新点**

创新点在于：①把组合式配置空间拆解成两层决策（结构层与提示层）实现可搜索性；②在结构层使用动作遮蔽（action masking）减少无效探索；③结合蒙版 RL 与精英经验的监督微调（SFT）提升收敛速度与稳定性；④在保持 LLM 冻结的前提下，仅训练轻量级策略网络，极大降低算力成本。

**🔧 技术方法**

主要技术：层次化策略（high‑level workflow/工具/预算决策 + low‑level提示生成）；PPO 强化学习；动作遮蔽与结构约束；SFT 监督微调；MetaCLIP 提取查询语义表示；token 预算与步骤计数的代价惩罚；工具使用奖励设计。

**📊 数据集**

使用的基准数据集：GSM8k、DROP、MedQA（推理能力）以及 HotpotQA、GAIA（工具使用能力）。模型使用 Qwen 2.5 7B Instruct 与 Gemini 2.5 Flash Lite 两个 LLM，数据集划分与官方训练/测试分割保持一致。

**📈 对比分析**

比较方法：与基线模型 + 工具、网格搜索/贪婪搜索、AutoGen、DSPy、GEPA、LAP、RL Bandits、RL Episodes 等多种手段对比。ARC 在大多数任务上提升 20–30% 的准确率，同时在 token 消耗和运行时成本上显著优于所有基线，位于 Pareto 前沿。

**⚠️ 局限性**

局限性：①对知识密集型任务（如 MedQA）提示内容仍是主要瓶颈，结构优化收益有限；②跨域工具迁移受工具集合与工作流相似度限制；③高维动作空间仍需大量样本，学习过程对环境噪声敏感；④在极大规模模型或多模态场景下的泛化仍待验证。

---

## 77. Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content

**arXiv ID:** 2602.11339 | [PDF](https://arxiv.org/pdf/2602.11339v1)

**作者:** Evgeney Bogatyrev `[一作]` (Lomonosov Moscow State University), Dmitry Vatolin `[通讯]` (Lomonosov Moscow State University)

**通讯引用:** 686 | [OpenAlex ID](https://openalex.org/A5020940244)

**关键词:** `Computer Vision and Pattern Recognition` `Super Resolution` `Compression` `Restoration` `Convolutional Neural Network` `Video` `Benchmark`

**🎯 论文内容**

本文提出了面向流媒体的压缩视频超分辨率研究，构建了规模达5200条YouTube视频的StreamSR数据集，并在该数据集上对11种实时超分模型进行系统基准测试，同时提出并实现了新的EfRLFN模型；

**💡 创新点**

创新点在于：1）创建了覆盖多分辨率和多类型视频的专门压缩视频超分数据集；2）设计了集成高效通道注意力（ECA）与双曲正切激活的轻量化网络结构；3）提出了包含Charbonnier、VGG感知与Sobel边缘三项的复合损失，直接端到端训练；4）实现了在ONNX/TensorRT上的高效推理。

**🔧 技术方法**

采用了高效通道注意力模块、双曲正切激活函数、复合Charbonnier+VGG+Sobel损失、GPT‑4o生成搜索查询、ONNX Runtime+TensorRT推理、以及广泛使用的Objective/Subjective评价指标。

**📊 数据集**

主要使用自建StreamSR数据集（5200条YouTube视频，分辨率360p–1440p，支持2×/4×提升），并在Set14、BSD100、Urban100、REDS、DIV2K等公开数据集上进行验证。

**📈 对比分析**

通过对比11种实时超分模型，使用PSNR、SSIM、LPIPS、CLIP‑IQA等多指标，并进行3,822人次的主观Bradley‑Terry评测，结果显示EfRLFN在保持或提升图像质量的同时实现≥271 FPS的实时推理，显著优于现有方法。

**⚠️ 局限性**

局限性包括：仅针对单帧图像超分，未实现时序扩展；在特定压缩编码与分辨率下表现最佳，跨码流/更低分辨率的泛化尚需进一步验证。

---

## 78. PRIME: Policy-Reinforced Iterative Multi-agent Execution for Algorithmic Reasoning in Large Language Models

**arXiv ID:** 2602.11170 | [PDF](https://arxiv.org/pdf/2602.11170v1)

**作者:** Jiawei Xu `[一作]` (Purdue University), Danyang Zhang `[通讯]`

**关键词:** `Computation and Language` `Reinforcement Learning` `Large Language Model` `Reinforcement Learning` `Prompt Engineering` `Text` `Benchmark`

**🎯 论文内容**

本文提出了 PRIME 框架和 PRIME‑Bench 基准，用于提升大语言模型在算法推理任务中的表现。

**💡 创新点**

创新点在于将多智能体执行、迭代验证与基于群体相对策略优化的强化学习相结合，并构建最大规模的算法推理基准。

**🔧 技术方法**

采用多智能体架构（执行器、验证器、协调器）与 Group Relative Policy Optimization，结合结构化提示和迭代执行实现错误回溯。

**📊 数据集**

使用 PRIME‑Bench 共 86 项任务、51,600 个实例（包含排序、图、自动机、数值等），并在 N‑Queens 子集上做细粒度实验。

**📈 对比分析**

与基线提示比较，平均准确率从 26.8% 提升至 93.8%（+250% 相对提升），单项如图灵机模拟从 9% 上升到 92%，结构化提示在 N‑Queens 上从 37.4% 提升至 90%，但平均推理时延增加 1.56 倍。

**⚠️ 局限性**

局限在于仅评估 N‑Queens 及公开模型，手工构造提示且未验证对其他约束满足问题的泛化，且未涉及封闭源模型或更复杂的多步约束推理。

---

## 79. LASER: An Efficient Target-Aware Segmented Attention Framework for End-to-End Long Sequence Modeling

**arXiv ID:** 2602.11562 | [PDF](https://arxiv.org/pdf/2602.11562v1)

**作者:** Tianhe Lin `[一作]` (Xiaohongshu Inc), Di Wu `[通讯]`

**关键词:** `Information Retrieval` `Recommendation System` `Computational Efficiency` `Transformer` `Sequential`

**🎯 论文内容**

提出了 LASER 框架，实现了端到端的超长用户行为序列建模。

**💡 创新点**

将统一的 SeqVault 存储系统与分段目标注意力和全局堆叠目标注意力相结合，既解决了 I/O 延迟，又将计算复杂度线性化。

**🔧 技术方法**

采用 DRAM-SSD 混合索引的 SeqVault、Sigmoid 门控的 Segmented Target Attention、轻量级的 Global Stacked Target Attention、ZSTD 压缩传输以及 RankMixer 特征交互等技术。

**📊 数据集**

在小红书广告平台的数亿条点击日志上进行离线评测与在线 AB 实验，序列长度最高达 1000。

**📈 对比分析**

与 DIN、HSTU、Transformer 等 SOTA 基线对比，LASER 在 AUC 上提升 0.24%，在线实验中 ADVV 提升 2.36%，收入提升 2.08%。

**⚠️ 局限性**

对分段窗口大小敏感，过小或过大都会导致信息损失或计算浪费；模型在更长序列和更大规模数据上的稳健性仍待进一步验证。

---

## 80. Compress, Cross and Scale: Multi-Level Compression Cross Networks for Efficient Scaling in Recommender Systems

**arXiv ID:** 2602.12041 | [PDF](https://arxiv.org/pdf/2602.12041v1)

**作者:** Heng Yu `[一作]` (Bilibili Inc.), Dongying Kong `[通讯]` (Bilibili Inc.)

**关键词:** `Information Retrieval` `Recommendation System` `Computational Efficiency` `Tabular`

**🎯 论文内容**

提出多层压缩交叉网络(MLCC)与多通道扩展(MC-MLCC)来高效建模高阶特征交互。

**💡 创新点**

通过层次压缩与动态交叉结合，实现了在保持低计算量的同时提升交叉表达能力，并引入多通道水平扩展实现更高ROI。

**🔧 技术方法**

采用分层压缩(Global/Local)、Progressive Layered Crossing动态MLP以及多通道并行结构。

**📊 数据集**

在Criteo、Avazu、TaobaoAds三大公开CTR基准和Bilibili广告工业数据集上进行实验。

**📈 对比分析**

与DNN、DCNv2、Wukong、RankMixer等基线对比，MLCC在公开数据上提升0.7–0.2% AUC，MC-MLCC在工业数据上匹配最强基线但参数和FLOPs减少26倍，在线AB测试提升32% ADVV。

**⚠️ 局限性**

仍受限于嵌入维度的有效性，需要进一步研究更高效的压缩方式以及跨任务通用性。

---

## 81. SpiralFormer: Looped Transformers Can Learn Hierarchical Dependencies via Multi-Resolution Recursion

**arXiv ID:** 2602.11698 | [PDF](https://arxiv.org/pdf/2602.11698v1)

**作者:** Chengting Yu `[一作]` (Zhejiang University), Bo Zheng `[通讯]`

**关键词:** `Machine Learning` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

提出 SpiralFormer，一种循环（looped）Transformer，采用多分辨率递归调度在共享核心上多次迭代，从粗到细逐步提升序列分辨率，进而在同一参数集下实现更深的推理。

**💡 创新点**

创新点在于：1）将分辨率调度引入循环Transformer，首次通过在不同层次压缩/放大隐藏状态实现层级依赖学习；2）设计了严格保持自回归的下采样/上采样算子和可学习的权重分配；3）结合 MeSH 记忆路由实现跨层次状态更新。

**🔧 技术方法**

主要技术包括：共享 Transformer 核心、块级下采样/上采样、可学习的聚合与分配、右移式因果补偿、MeSH/Anchor 状态更新、层级分辨率调度（粗到细）以及基于注意力统计的头部分析。

**📊 数据集**

使用 Pythia 预训练系列（160M~1.4B）在去重后的 Pile 数据子集上训练，一轮 250B 语料；随后在 Pile、Wiki、LD‑O、LD‑S 等基准集上评估语言建模 perplexity 与 0‑shot/5‑shot 下游任务准确率。

**📈 对比分析**

与标准非循环 Pythia 以及全分辨率 LoopFormer 进行对比。结果表明：在相同参数或 FLOPs 条件下，SpiralFormer 能在 5‑shot 准确率上提升 2–4% 并在 perplexity 上降低 1–3%；在相同 FLOPs 下减少 7–12% 计算量同时保持或提升性能；在相同参数下显著低于全分辨率循环模型的 FLOPs，并保持更好的下游效果。

**⚠️ 局限性**

局限性包括：1）对比的多分辨率调度主要采用粗到细的单一策略，其他策略仍需探索；2）在无重叠（并行）模式下性能下降，尚未实现与并行推理的平衡；3）目前仅在中等规模（1.4B）验证，极大规模扩展和在更复杂任务上的泛化仍待进一步研究。

---

## 82. Learn from Your Mistakes: Self-Correcting Masked Diffusion Models

**arXiv ID:** 2602.11590 | [PDF](https://arxiv.org/pdf/2602.11590v1)

**作者:** Yair Schiff `[一作]` (Cornell), Volodymyr Kuleshov `[通讯]` (Cornell)

**通讯引用:** 5905 | [OpenAlex ID](https://openalex.org/A5021338648)

**关键词:** `Machine Learning` `Generation` `Optimization` `Diffusion model` `Text` `Benchmark`

**🎯 论文内容**

在掩码扩散模型（MDM）基础上提出Progressive Self‑Correction（PSC）框架，使模型既能进行并行解码，又能在已解码位置自我纠错；

**💡 创新点**

创新点在于：①把模型产生的“错误”视为噪声，加入自纠正损失；②权重共享实现单模型完成解码与纠错；③在采样时交替执行纠错与解码步骤，显著提升质量与速度；

**🔧 技术方法**

使用连续时间离散扩散理论、掩码噪声过程、交叉熵自纠正损失、以及自回归式采样策略（如 greedy‑max、CFG）等；

**📊 数据集**

实验覆盖：代码/数学 benchmark（HumanEval、MBPP、GSM8K、Minerva）; 分子生成（QM9/SMILES）; 开放文本（OpenWebText）等；

**📈 对比分析**

与标准MDM、ReMDM、PRISM、AR 等模型比较，PSC 在大多数任务中实现1.3倍以上质量提升，采样速度提升2–3倍，且在多任务中保持或超过最优基线；

**⚠️ 局限性**

主要局限是训练时需额外一次前向传播，导致训练成本增加；

---

## 83. Query-focused and Memory-aware Reranker for Long Context Processing

**arXiv ID:** 2602.12192 | [PDF](https://arxiv.org/pdf/2602.12192v1)

**作者:** Yuqing Li `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Jie Zhou `[通讯]` (Tencent)

**通讯引用:** 34320 | [OpenAlex ID](https://openalex.org/A5100620306)

**关键词:** `Computation and Language` `Retrieval` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Contrastive Learning` `Text`

**🎯 论文内容**

提出了QRRanker框架，利用训练好的查询聚焦检索头对候选文档进行无生成的列表式重排序。

**💡 创新点**

通过训练QR头直接产生连续相关性分数，消除对Likert量表的依赖和生成过程，实现轻量化、可扩展的重排序。

**🔧 技术方法**

采用LLM自注意力中的QR分数、max‑min规范化的对比损失、可选摘要前缀以及中层截断等技术。

**📊 数据集**

在Wiki多跳QA（HotpotQA、MuSiQue）、长文本QA（NarrativeQA、DetectiveQA）以及对话记忆任务（LoCoMo）上进行评估，并构造MuSiQue+NarrativeQA训练集。

**📈 对比分析**

与嵌入模型、点式/列表式重排序器（HippoRAG、GroupRank、Qwen‑Reranker）以及多种记忆框架进行对比，QRRanker在Recall@k和下游F1指标上均超越SOTA，尤其在多跳、长上下文场景表现突出。

**⚠️ 局限性**

局限在于仍需预选QR头，低层头效果不佳；在Wiki多跳QA中摘要前缀无明显提升；需要大规模训练资源，对新任务仍需头选择与微调。

---

## 84. What if Agents Could Imagine? Reinforcing Open-Vocabulary HOI Comprehension through Generation

**arXiv ID:** 2602.11499 | [PDF](https://arxiv.org/pdf/2602.11499v1)

**作者:** Zhenlong Yuan `[一作]` (AMAP, Alibaba Group), Yuyin Zhou `[通讯]` (UC Santa Cruz)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Retrieval-Augmented Generation` `Vision Language Model` `World Model` `Multimodality`

**🎯 论文内容**

提出ImagineAgent框架，结合认知映射、生成式想象与工具增强的强化学习，实现开放词汇人-物交互（OV-HOI）检测。

**💡 创新点**

创新点在于：①构建可解释的认知图；②动态调用多模态工具（检索、裁剪、生成等）以补齐感知缺口；③将生成式世界模型引入Agent，提升对遮挡与歧义的处理；④采用复合奖励的GRPO训练策略，兼顾预测精度与工具使用效率。

**🔧 技术方法**

使用多模态大型语言模型（Qwen2.5-VL）、BAGEL生成模型、检索增强工具、裁剪工具、视角变换工具，以及GRPO强化学习与自监督训练。

**📊 数据集**

在HICO-DET（含Zero-shot拆分）与SWIG-HOI两个公开数据集上进行评测。

**📈 对比分析**

与现有OV-HOI方法（如INP-CC、HOICLIP、CLIP4HOI等）对比，HICO-DET Full mAP提升至28.96%（高于INP-CC 23.12%），SWIG-HOI Full mAP提升至17.75%（高于INP-CC 16.74%）。在零样本、稀有与未见类别上均表现优异。

**⚠️ 局限性**

主要限制包括：对工具库的依赖与扩展性、生成模型的计算成本、以及在极端遮挡或文本先验冲突情形下仍可能出现幻觉或误检。

---

## 85. Gray Codes With Constant Delay and Constant Auxiliary Space

**arXiv ID:** 2602.11791 | [PDF](https://arxiv.org/pdf/2602.11791v1)

**作者:** Antoine Amarilli `[一作]` (University of Lille), Yann Strozecki `[通讯]` (University of Versailles Saint-Quentin)

**关键词:** `Data Structures and Algorithms`

**🎯 论文内容**

本文提出两种新型计算模型（tape machine 和 deque machine），并给出了能够在常数延迟和常数辅助空间内枚举所有长度为 ℓ 的二进制单词（即 Gray 代码）的算法。

**💡 创新点**

创新点在于：①首次在仅使用常数辅助内存的模型下实现了全量二进制词的常数延迟枚举；②设计了 Hamming‑1 的 Gray 代码实现；③通过“偶奇技巧”“双遍历”和“前瞻”技术，使得 deque machine 能够在不使用额外位计数器的情况下完成枚举；④证明了在更弱的队列机和栈机模型下，常数辅助空间无法完成此任务。

**🔧 技术方法**

主要技术包括：深度优先遍历完整二叉树、偶奇（Feder）递归技巧、对机状态进行二进制/位数的位操作、窗口读写、回溯/循环位移、状态编码与哈希、递归定义的 Gray 代码（Construction A）以及 rank/unrank 计算。

**📊 数据集**

论文中使用了自定义的数据文件和 Python 脚本（包含转移表、执行脚本、验证脚本、rank/unrank 脚本）来实现并验证机器的正确性与性能；未使用外部标准数据集，而是通过实验脚本验证了 3≤ℓ≤18 的枚举完整性与延迟上界。

**📈 对比分析**

实验方法是对每个 ℓ 生成所有 2^ℓ 词，检查无重复、完整覆盖，并测量最大步骤数与平均步骤数，确认常数延迟与常数辅助空间。相较于传统 Gray 代码实现（如 RBGC），新算法在延迟方面保持常数且在内存上比传统实现更小；在 rank/unrank 计算上提供了可直接的 O(ℓ) 计算方式。

**⚠️ 局限性**

限制包括：1) 仅针对二进制字母表，难以直接推广到更大字母表或任意正则语言；2) 在 deque machine 中尚未实现完整的递增/递减计数器功能；3) 对于更弱的队列机和栈机模型，证明了不可能实现；4) 代码实现较为复杂，调试与扩展成本较高。

---

## 86. MonarchRT: Efficient Attention for Real-Time Video Generation

**arXiv ID:** 2602.12271 | [PDF](https://arxiv.org/pdf/2602.12271v1)

**作者:** Krish Agarwal `[一作]` (Carnegie Mellon University), Beidi Chen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5073845046)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Computational Efficiency` `Transformer` `Supervised Fine-Tuning` `Video`

**🎯 论文内容**

针对实时视频生成，提出了一种基于 Monarch 参数化的稀疏注意力近似方法，并通过块对齐、分块 Monarch（Tiled Monarch）与训练时微调，构建高效且表达能力强的 3D 注意力实现。

**💡 创新点**

创新点包括：
• 识别并解决 3D 注意力的块对齐问题，保证各空间‑时间维度不被拆分；
• 引入可细化的 Tiled Monarch，显著提升表达力并实现可控的计算‑精度折衷；
• 在训练阶段通过 fine‑tuning 极大减少推理时的迭代步骤，使单步迭代即可获得接近密集注意力的质量；
• 开发专用 Triton 内核，支持长序列的 3D 注意力高效实现。

**🔧 技术方法**

使用了 Monarch 参数化（Block‑wise rank‑1 结构）与其分块变体、迭代优化的 MonarchAttention、FlashAttention‑style GPU 内核、以及训练时的微调策略。

**📊 数据集**

主要在 Self‑Forcing（基于 Wan 2.1‑1.3B）和其 4 步蒸馏版上进行实验，并在 50‑step Wan 2.1‑1.3B 以及 480p/720p 视频上评测。

**📈 对比分析**

与密集注意力、oracle top‑k、VSA、Sparse VideoGen、RadialAttention 等基线比较，
• 在 95% 稀疏度下保持接近 dense 的 VBench/PSNR/SSIM/LPIPS 分数；
• 速度提升可达 1.4×–11.8×（在 RTX 5090 上 16 FPS、H100 上 5.6×），并在 16 FPS 的实时生成上实现高质量输出。

**⚠️ 局限性**

局限性：
• 需要预先对视频维度进行块对齐，若对齐不当会导致质量骤降；
• 分块尺寸与 tile 参数需手动调优，未给出自动化方案；
• 目前实现主要针对基于 DiT 的自回归或少步扩散模型，尚未验证在更大规模或不同架构（如 Transformer‑XL、Longformer 等）上的通用性。

---

## 87. Hierarchical Concept Embedding & Pursuit for Interpretable Image Classification

**arXiv ID:** 2602.11448 | [PDF](https://arxiv.org/pdf/2602.11448v1)

**作者:** Nghia Nguyen `[一作]` (University of Pennsylvania), René Vidal `[通讯]` (University of Pennsylvania)

**通讯引用:** 26288 | [OpenAlex ID](https://openalex.org/A5011256828)

**关键词:** `Machine Learning` `Classification` `Explainability and Interpretability` `Sparse Coding` `Orthogonal Matching Pursuit` `Hierarchical Sparse Coding` `Beam Search` `Vision Language Model` `Image`

**🎯 论文内容**

提出了一个基于层次概念嵌入和层次稀疏编码的解释性图像分类框架 HCEP。

**💡 创新点**

将层次结构融入概念嵌入，设计了满足聚类、正交、单纯形几何条件的层次字典，并引入层次正交匹配追踪（Hierarchical OMP）和波束搜索，以显著提升概念恢复的精确度与召回率。

**🔧 技术方法**

采用稀疏编码、正交匹配追踪、层次稀疏编码、波束搜索技术，并利用 CLIP（及 SigLIP）视觉语言嵌入作为概念向量；同时做了几何分析与正交性验证。

**📊 数据集**

在合成数据以及 ImageNet、ImageNette、CIFAR‑100 等真实图像数据集上进行实验评估。

**📈 对比分析**

与标准 OMP、Concept Bottleneck Models、Nearest‑Neighbor、层次 NN 等基线比较，HCEP 在概念支持的精确度和召回率上显著优于对手，在低样本量场景下还能保持甚至提升分类准确率。

**⚠️ 局限性**

受限于嵌入维度需满足 d≥L+b，层次质量高度依赖所给层次结构，计算复杂度随波束宽度和层次深度增加而提升；对离群样本或未知类别的处理仍有限。

---

## 88. CM2: Reinforcement Learning with Checklist Rewards for Multi-Turn and Multi-Step Agentic Tool Use

**arXiv ID:** 2602.12268 | [PDF](https://arxiv.org/pdf/2602.12268v1)

**作者:** Zhen Zhang `[一作]` (University of California), Song Wang `[通讯]` (Zoom Video Communications)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Large Language Model` `Reinforcement Learning` `Agentic AI` `Chain-of-Thought` `Text` `Benchmark`

**🎯 论文内容**

本文提出一种利用检查表（Checklist）奖励的强化学习框架，训练多回合、多步骤工具使用的AI代理。

**💡 创新点**

创新点在于：①用二元、证据支撑的检查表代替可验证奖励，降低判定噪声；②实现“稀疏奖励分配+密集评估准则”策略；③在大规模LLM模拟工具环境中无须实际接口即可训练。

**🔧 技术方法**

采用的技术包括：LLM-作为判定者（Judge）的检查表评估、GRPO（Group Relative Policy Optimization）强化学习、工具模拟器（Hybrid Tool Emulator）、链式思考压缩与冷启动SFT。

**📊 数据集**

使用的数据集：NVIDIA Nemotron Post‑Training 数据集（310k 条合成对话，筛选后 30k 条高质量样本）以及 5k 领域内合成数据；对照基准为 τ²‑Bench、BFCL‑V4 与 ToolSandbox。

**📈 对比分析**

与基线比较：从 8B 基础模型起，RL 在 τ²‑Bench 上比 SFT 提升 8 分，BFCL‑V4 上提升 10 分，ToolSandbox 上提升 12 分；与同等规模的公开基线相比，RL 结果相当或更好，甚至超越 30B‑A3B‑Instruct（判定模型）与 8B‑Thinking。

**⚠️ 局限性**

局限性包括：①奖励仍受判定噪声影响，细粒度奖励分配易导致训练崩溃；②依赖 LLM 判定，需较大模型支持；③工具模拟器对真实环境的逼真度有限，可能导致与实际 API 的偏差；④在极长对话或高复杂度工具链下仍需进一步验证。

---

## 89. Universal Diffusion-Based Probabilistic Downscaling

**arXiv ID:** 2602.11893 | [PDF](https://arxiv.org/pdf/2602.11893v1)

**作者:** Roberto Molinaro `[一作]` (Jua.ai), Marvin Vincent Gabler `[通讯]` (Jua.ai)

**关键词:** `Machine Learning` `Diffusion model` `Time Series`

**🎯 论文内容**

开发了一种基于扩散模型的通用降尺度框架，可将粗分辨率的确定性天气预报转化为概率性的高分辨率预测，并在零样本条件下应用于多种模型。

**💡 创新点**

该方法一次性训练一个模型即可在不同上游预报系统上零样本使用，且通过学习条件分布而非回归均值，实现了显著的概率预报改进。

**🔧 技术方法**

使用条件扩散模型（EDM 风格）与 U‑Net+注意力架构进行高维分布学习，并采用逆扩散采样生成多样本。

**📊 数据集**

训练使用 ERA5（~25 km）与 CERRA（~5 km）欧盟区域再分析配对；验证采用 WMO SYNOP、WIS 2.0、METAR 等约 1 万座站点观测。

**📈 对比分析**

对比原始确定性预报与扩散降尺度结果，利用 RMSE 与 CRPS 评估；在七月至六月的 90 h 预测期间，RMSE 提升 5–14%，CRPS 提升 15–30%，并在多种 AI 与 NWP 模型上保持正向改进。

**⚠️ 局限性**

仅针对近地表变量和单一欧盟高分辨率域，未考虑不同气候区的迁移、样本量与推理成本调优，且缺乏后置校准与区域专属微调。

---

## 90. Benchmark Illusion: Disagreement among LLMs and Its Scientific Consequences

**arXiv ID:** 2602.11898 | [PDF](https://arxiv.org/pdf/2602.11898v1)

**作者:** Eddie Yang `[一作]` (Purdue University), Dashun Wang `[通讯]` (Northwestern University)

**通讯引用:** 9462 | [OpenAlex ID](https://openalex.org/A5002041772)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text` `Benchmark`

**🎯 论文内容**

研究大型语言模型（LLM）在科学研究中作为注释工具时，即使在基准测试上表现相似，也会出现大量答案分歧，导致实验结论的显著偏差。

**💡 创新点**

提出“benchmark幻觉”概念：同一基准得分相近的模型在实际推断中可能具有截然不同的误差结构，从而对研究结果产生深远影响。

**🔧 技术方法**

采用配对不一致率分析、测量误差框架、仿真实验以及对教育与政治科学两项已发表研究的重新注释与重分析技术。

**📊 数据集**

使用的数据集包括：MMLU-Pro、GPQA（两大推理基准）；教育实验中的学生写作样本；俄罗斯国家媒体新闻文本。

**📈 对比分析**

方法：计算模型在同一题目上的不一致比例；通过模拟不同误差结构的注释器估计处理效应；在重分析中使用八种不同LLM替代原始注释，并重新估计效应。结果显示即便整体准确率相近，估计效应差异可达80%以上，甚至出现符号相反的情况。

**⚠️ 局限性**

局限性：仅检验两类基准与两项案例，未涵盖模型随时间演进或其他领域任务；未探讨如何利用模型多样性或制定系统的多模型鲁棒性检验方法。

---

## 91. Small Updates, Big Doubts: Does Parameter-Efficient Fine-tuning Enhance Hallucination Detection ?

**arXiv ID:** 2602.11166 | [PDF](https://arxiv.org/pdf/2602.11166v1)

**作者:** Xu Hu `[一作]` (University of Texas at Dallas), Feng Chen `[通讯]` (University of Texas at Dallas)

**通讯引用:** 368303 | [OpenAlex ID](https://openalex.org/A5111964102)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

系统性研究参数高效微调（PEFT）对大语言模型生成性事实错误检测（hallucination detection）的影响，评估其对不确定性表达与检测性能的改变。

**💡 创新点**

首次将PEFT视为认知正则化（epistemic regularizer），揭示其主要作用是重塑模型的置信度分布而非单纯注入知识，从而显著提升多种无监督检测方法的 AUROC。

**🔧 技术方法**

使用LoRA、PiSSA、DoRA三种低秩PEFT方法，对LLaMA、Mistral、Qwen三大开源指令模型进行领域微调；评估包括语义一致性、置信度、熵三类的八种无监督检测器，并采用线性探针进行白盒检测。

**📊 数据集**

三大事实检索 QA 基准：TriviaQA、Natural Questions (NQ‑Open) 与 SQuAD v1。

**📈 对比分析**

与基线模型比较，PEFT 在 QA 准确率提升有限（≤3–6%），但在大多数检测器上 AUROC 明显提升（平均提升约 4–15%），尤其是语义一致性与置信度检测器；熵基检测器提升不显著。白盒线性探针表现不一，部分情况甚至下降。

**⚠️ 局限性**

局限包括仅评估英文 QA 任务、3B–7B 模型、仅三种 PEFT 与单一微调方式；对其他语言、模态、更大模型以及其他任务类型（长文本、推理、多模态、交互式对话）的泛化仍需验证。

---

## 92. LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation

**arXiv ID:** 2602.11451 | [PDF](https://arxiv.org/pdf/2602.11451v1)

**作者:** Ahmadreza Jeddi `[一作]` (University of Toronto), Babak Taati `[通讯]` (University of Toronto)

**通讯引用:** 3280 | [OpenAlex ID](https://openalex.org/A5011257199)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Ordinary Differential Equation` `Text`

**🎯 论文内容**

设计并训练了一种称为LoopFormer的循环Transformer，支持预算条件下可弹性计算深度的语言建模和推理。

**💡 创新点**

通过对循环步骤进行时间t和步长Δt的条件化以及跨轨迹的一致性训练，使得模型在不同计算预算下保持信息丰富且可持续改进的表示。

**🔧 技术方法**

采用共享Transformer块、RMSNorm与门控调节、时间与步长嵌入、shortcut‑consistency 损失，类似扩散模型和神经ODE的轨迹训练。

**📊 数据集**

在大规模无重复的Pile子集上训练，评估包括FineWeb‑Edu、OpenWebText、The Pile等语言建模数据，以及十个零样本推理基准如COPA、HellaSwag等。

**📈 对比分析**

与非循环Transformer、固定深度循环模型、TMLT及早期退出模型对比；在相同FLOP预算下，LoopFormer在困惑度和零样本推理上均显著优于其他循环基线，且在预算减少时仍保持高性能。

**⚠️ 局限性**

仍是全局预算策略，缺乏输入级自适应；训练需要多轨迹一致性计算，且表现分析主要是相关性而非因果；未实现真正的实例级动态深度。

---

## 93. SWE-MiniSandbox: Container-Free Reinforcement Learning for Building Software Engineering Agents

**arXiv ID:** 2602.11210 | [PDF](https://arxiv.org/pdf/2602.11210v1)

**作者:** Danlong Yuan `[一作]` (Wangxuan Institute of Computer Technology, Peking University), Dongyan Zhao `[通讯]` (State Key Laboratory of General Artificial Intelligence)

**关键词:** `Software Engineering` `Reinforcement Learning` `Reinforcement Learning` `Agentic AI` `Tabular`

**🎯 论文内容**

提出了SWE‑MiniSandbox，一种无容器沙盒框架，用于训练和评估软件工程（SWE）代理。

**💡 创新点**

创新点在于利用挂载命名空间和chroot实现进程/文件系统隔离，彻底去除容器依赖；结合预缓存、I/O限速和分布式资源调度，显著降低存储占用和环境准备时间。

**🔧 技术方法**

核心技术包括Linux挂载命名空间、chroot、Python venv、tar.gz 预缓存、Ray 分布式资源控制、Semaphore、以及与SWE‑Rex、SWE‑agent、SkyRL 的无缝集成。

**📊 数据集**

主要使用的数据集为 SWE‑bench Verified（500 个 Python 任务）以及 SWE‑smith（50k 任务）等。

**📈 对比分析**

与传统基于 Docker 容器的基线对比：MiniSandbox 的存储仅占 5%，环境准备时间仅为 25%，在 RL 训练中的精度与容器基线相当；在多节点、批量化与 Rollout 场景下保持低延迟并实现近线性扩展。

**⚠️ 局限性**

局限性包括：仍需为极端系统依赖任务保留容器；I/O 性能受磁盘带宽限制；部分任务需要手工调整安装命令；未全面验证跨平台兼容性。

---

## 94. Sci-CoE: Co-evolving Scientific Reasoning LLMs via Geometric Consensus with Sparse Supervision

**arXiv ID:** 2602.12164 | [PDF](https://arxiv.org/pdf/2602.12164v1)

**作者:** Xiaohan He `[一作]` (Fudan University), Bo Zhang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 28382 | [OpenAlex ID](https://openalex.org/A5023023644)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Text` `Physics Related`

**🎯 论文内容**

提出了 Sci-CoE 框架，使大型语言模型在科学推理任务中通过自演化的解算器和验证器协同提升推理能力；

**💡 创新点**

核心创新在于稀疏锚定学习与几何共识奖励机制，能够在潜在几何空间中平衡验证策略的一致性、可靠性与多样性，避免共识崩塌；

**🔧 技术方法**

使用 PPO 强化学习联合解算器与验证器、外部判别器评估、嵌入模型映射验证策略至向量空间，并利用 K‑means 与 PCA（极坐标）构造几何奖励；

**📊 数据集**

训练数据包括 MegaScience、Numinamath、ScienceQA、CaseHold 的标注子集（Stage 1）和 18k/30k 未标注科学问答（Stage 2）；评测基准为 MMLU‑Pro、UGPhysics、GPQA‑Diamond 等；

**📈 对比分析**

与同规模基线（Llama‑3.1‑8B、Mistral、Yi、Qwen2.5‑7B 等）在相同 PPO 训练框架下对比，Sci‑CoE 在 MMLU‑Pro、UGPhysics、GPQA‑Diamond 上分别提升约 1–4% 的准确率，证明了显著的性能提升；

**⚠️ 局限性**

局限性包括仅训练至 8 B 参数；仍需外部判别模型，导致额外计算开销；在部分物理子领域提升不显著，受限于未标注数据与验证策略质量。

---

## 95. Where Bits Matter in World Model Planning: A Paired Mixed-Bit Study for Efficient Spatial Reasoning

**arXiv ID:** 2602.11882 | [PDF](https://arxiv.org/pdf/2602.11882v1)

**作者:** Suraj Ranganath `[一作]` (University of California San Diego), Vaishak Menon `[通讯]` (University of California San Diego)

**关键词:** `Machine Learning` `World Model`

**🎯 论文内容**

研究了低位量化对 DINO‑WM 在 Wall 规划任务中的空间推理性能影响

**💡 创新点**

发现规划质量在低位点更依赖位宽分配而非总位宽，并提出模块感知、预算感知的量化策略

**🔧 技术方法**

采用权重仅量化（对称 per‑output‑channel 量化）对 FP16、INT8/6/4/3 及多种混合与不对称配置进行评估

**📊 数据集**

使用预训练的 DINO‑WM 模型在 Wall 环境中的规划任务数据

**📈 对比分析**

通过配对目标评估对比不同量化方案，8/6 位保持 FP16 成功率，4 位表现随位宽分配敏感，3 位完全崩溃；模型大小和运行时也被报告

**⚠️ 局限性**

实验样本有限、仅权重量化（无激活量化、校准或 QAT）、单一环境/模型、混合量化与总精度混杂，无法完全剔除大小效应

---

## 96. ReaDy-Go: Real-to-Sim Dynamic 3D Gaussian Splatting Simulation for Environment-Specific Visual Navigation with Moving Obstacles

**arXiv ID:** 2602.11575 | [PDF](https://arxiv.org/pdf/2602.11575v1)

**作者:** Seungyeon Yoo `[一作]` (Seoul National University), H. Jin Kim `[通讯]` (Seoul National University)

**通讯引用:** 7131 | [OpenAlex ID](https://openalex.org/A5073996122)

**关键词:** `Robotics` `Robotic Intelligence` `Domain Adaptation` `Gaussian Splatting` `Reinforcement Learning` `Video` `Image`

**🎯 论文内容**

提出了 ReaDy-Go 这一端到端的实时转仿真（real‑to‑sim）动态 3D 高斯渲染（GS）仿真管线，利用单目视频构建目标环境的 GS 场景，并在其中加入可动画化的 GS 人体模型生成逼真的动态障碍，从而生成大规模的目标环境特定 RGB‑only 视觉导航数据并训练环境特定的视觉导航策略。

**💡 创新点**

创新点主要有：①将动态人类障碍以 GS 表示并通过 PriorMDM 生成符合 2D 轨迹的自然运动；②在 GS 场景中实现无物理引擎的高质量动态渲染；③设计基于 Hybrid A* 的机器人专家规划器，实时更新动态障碍并提供可行轨迹；④通过结合上述仿真、规划器和人类规划器一次性生成多样化的动态导航数据，显著提升 sim‑to‑real 迁移和对动态环境的鲁棒性。

**🔧 技术方法**

技术包括：3D Gaussian Splatting（PGSR）用于场景重建；HUGS 与 NeuMan 数据集提取可动画化 GS 人体模型；PriorMDM 运动扩散模型生成根轨迹与关节角度；Hybrid A* + 运动原语用于机器人路径规划；简单的 RGB 编码器 + MLP 进行行为模仿学习；以及基于 2D 占据图的机器人与人类规划器。

**📊 数据集**

使用的主要数据集为：单目视频（约 1,000–1,500 张图像）用于 GS 场景重建；NeuMan/HUGS 提取的人体 GS 模型；自行生成的 400 条训练轨迹（≈80k–120k 训练样本）用于动态导航数据；验证集 50 条轨迹；在实验中对比 Vid2Sim、GNM、ViNT、NoMaD 等公开基线。

**📈 对比分析**

实验通过在三个目标环境（Outside、Lobby、Library）进行模拟与真实机器人测试，评估成功率（SR）和平均到达时间（ART）。ReaDy-Go 在动态任务中成功率可达 90%+、ART 约 10–20 秒，明显优于 Vid2Sim（动态任务 SR 78%/ART 18.7）、GNM/NoMaD/ViNT（SR 35–60%）。在真实环境中，ReaDy-Go 在静态/动态任务中均保持 90–100% 的成功率，且 ART 与模拟一致，证明了强大的 sim‑to‑real 迁移。零样本未见环境实验显示 SR 50–70%，表明具备一定泛化能力。

**⚠️ 局限性**

局限性包括：①仅在单一机器人平台（差分轮式）和单目摄像头上验证，泛化到其他传感器/机器人类型需要进一步测试；②动态障碍仅为人类模型，缺乏多种动态对象（车辆、机器人等）的实验；③仿真中仍假设人类运动预测为常速，未考虑复杂动态交互；④尽管实现了较好的迁移，但对完全陌生环境仍需更丰富的训练数据才能获得更高的成功率。

---

## 97. Device-Circuit Co-Design of Variation-Resilient Read and Write Drivers for Antiferromagnetic Tunnel Junction (AFMTJ) Memories

**arXiv ID:** 2602.11614 | [PDF](https://arxiv.org/pdf/2602.11614v1)

**作者:** Yousuf Choudhary `[一作]` (University of Arizona), Tosiron Adegbija `[通讯]` (University of Arizona)

**通讯引用:** 640 | [OpenAlex ID](https://openalex.org/A5075537309)

**关键词:** `Hardware Architecture` `Physics Related`

**🎯 论文内容**

本文针对抗磁隧道结(AFMTJ)的低TMR与高速特性，设计了自适应感应放大器(STSA+)、分层预充电/等化驱动(PD_EQ+)以及非对称写驱动(WD_WRITE)的读取/写入接口，实现了在PVT、3D热梯度和器件变异下的可靠操作。

**💡 创新点**

创新点在于将感应放大器的阈值可编程、温度补偿以及动态参考跟踪与AFMTJ低TMR匹配；预充电驱动根据层温度自适应等化窗口并提升信号中心化；写驱动通过可编程脉冲形状与温度补偿实现亚纳秒确定性切换。

**🔧 技术方法**

采用基于双子晶格AFMTJ的SPICE模型、28nm CMOS与AFMTJ的混合仿真、Monte Carlo统计以及对3D堆叠的分布式RC与TSV热模型。

**📊 数据集**

使用校准的双子晶格Mn3SnN AFMTJ SPICE模型，并生成3.15 M样本的Monte Carlo仿真数据作为评估数据集。

**📈 对比分析**

与传统Strong‑ARM STSA、标准PD/PD‑EQ以及MTJ接口对比后，STSA+在各PVT角落实现读能量下降4–7倍、写延迟最快、写能量低于传统MTJ的1/5，且在3D热梯度下读写误差率均保持≤10⁻⁶，整体性能优于现有MRAM前端。

**⚠️ 局限性**

主要局限在于实验仍基于仿真，缺乏实测验证；对极端温度梯度和更大规模3D堆叠的鲁棒性待进一步评估；以及低TMR读取仍需在更高噪声环境下验证。

---

## 98. SafeNeuron: Neuron-Level Safety Alignment for Large Language Models

**arXiv ID:** 2602.12158 | [PDF](https://arxiv.org/pdf/2602.12158v1)

**作者:** Zhaoxin Wang `[一作]` (Xidian University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60408 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `Machine Learning` `Safty and Privacy` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Reinforcement Learning` `Multimodality` `Text`

**🎯 论文内容**

本文提出了 SafeNeuron——一种基于神经元级别的安全对齐框架，通过先识别安全相关神经元并在后续对齐中将其冻结，随后使用 RLHF（DPO）在其余参数上优化，构建冗余安全路径；同时引入迭代训练方案，进一步分散安全表示；对 LLM 与 VLM 在多任务与多模态场景下进行评估。

**💡 创新点**

创新点包括：① 将安全行为视为结构化、共享的内部表示，而非少量脆弱参数；② 通过冻结已识别安全神经元强制模型学习更多安全通道，提升对 neuron‑level 攻击的鲁棒性；③ 采用 Activation Effect Size 与 Safety Activation Shift 两种互补指标系统地识别安全神经元；④ 通过层级与跨任务分析揭示安全子空间的分布规律。

**🔧 技术方法**

主要技术手段包括：GLU‑FFN 结构下的神经元激活分析；Activation Effect Size（ES）与 Safety Activation Shift（SAS）用于安全神经元筛选；DPO（Direct Preference Optimization）对剩余参数进行 RLHF 风格的安全微调；迭代冻结与再训练机制；层级分布与任务共性分析；使用 Llama‑Guard 进行外部安全评估。

**📊 数据集**

使用的数据集有：PKU‑SafeRLHF、SPA‑VL（用于对齐训练）；StrongReject（包含 313 条恶意 jailbreak prompt）用于安全评估；NeuroStrike（NSFW 图像与 VL‑Question 场景）用于 VLM 评估；ARC、GSM8K、TruthfulQA 用于衡量通用能力；同时对多模态输入使用 Qwen2.5‑VL、图像嵌入式 jailbreak（FigStep）等。

**📈 对比分析**

与原始指令微调、SN‑Tune 以及传统 RLHF‑Safety 三类基线对比。评估指标为：ASR（越低越好）和通用性能（ARC、GSM8K、TruthfulQA）。实验表明 SafeNeuron 在所有模型上均实现了更低的 ASR，尤其在 neuron‑pruning 后仍保持较低攻击成功率；且在通用指标上保持或略优于基线，说明安全增强不牺牲一般能力。

**⚠️ 局限性**

局限性：① 需要对原始模型进行神经元级别分析与冻结，增加了对模型内部结构的依赖；② 对抗者可能进一步针对未冻结的神经元设计新攻击；③ 评估集主要基于已有的 jailbreak 与恶意 prompt，未覆盖所有潜在攻击方式；④ 在极大规模模型或不同架构（如非 GLU‑FFN）中的效果尚未验证；⑤ 迭代训练成本相对较高，可能限制实际部署。

---

## 99. Yaksha-Prashna: Understanding eBPF Bytecode Network Function Behavior

**arXiv ID:** 2602.11232 | [PDF](https://arxiv.org/pdf/2602.11232v1)

**作者:** Animesh Singh `[一作]` (Indian Institute of Technology Hyderabad), Praveen Tammana `[通讯]` (Indian Institute of Technology Hyderabad)

**通讯引用:** 409 | [OpenAlex ID](https://openalex.org/A5052098266)

**关键词:** `Cryptography and Security`

**🎯 论文内容**

提出了 Yaksha-Prashna 系统，利用数据流分析从 eBPF 字节码中抽取网络上下文，并通过 DSL 让运维人员和开发者验证功能正确性与互操作性。

**💡 创新点**

创新点在于将低层 eBPF 字节码与高级网络行为抽象映射，提供可复用的查询语言和离线分析模型，实现验证速度提升 200–1000 倍。

**🔧 技术方法**

采用了控制流图 + 数据流分析规则提取网络上下文，使用 Prolog 推理引擎执行断言与检索查询，并与符号执行/抽象解释等传统验证方法做对比。

**📊 数据集**

使用了来自 Katran、Suricata、Cilium 等项目的 16 个 XDP eBPF 程序（指令数 400–4000，路径数 4–4K）作为实验数据集。

**📈 对比分析**

与 Klint、DRACO 进行对比，验证时间 13–75 ms（vs 2.7 s–1.5 min）、内存占用 20–50 MB（vs 20–100 MB），查询耗时仅几微秒。

**⚠️ 局限性**

局限性包括：不支持运行时属性验证、映射规则检查、非 XDP hook（如 TC、socket filter），以及对某些仅处理有限协议的程序的协议解析准确性不足。

---

## 100. GHOST: Unmasking Phantom States in Mamba2 via Grouped Hidden-state Output-aware Selection & Truncation

**arXiv ID:** 2602.11408 | [PDF](https://arxiv.org/pdf/2602.11408v1)

**作者:** Michael Menezes `[一作]` (Rice University), Anastasios Kyrillidis `[通讯]` (Rice University)

**通讯引用:** 1772 | [OpenAlex ID](https://openalex.org/A5024280658)

**关键词:** `Artificial Intelligence` `Compression` `Computational Efficiency` `Recurrent Neural Network` `Large Language Model` `Text`

**🎯 论文内容**

设计了一种基于控制理论的结构化剪枝框架，针对大规模SSM模型的状态维度进行压缩，以降低推理时的内存带宽消耗。

**💡 创新点**

通过仅使用前向传递统计量近似平衡截断，将可控性与可观测性结合为激活能量指标，克服了传统幅度剪枝的“假正负”现象，并在不使用梯度的情况下实现了与二阶方法相当的精度。

**🔧 技术方法**

使用控制理论中的平衡截断与经验格朗维安估计，结合组间阈值化与软剪枝的前向统计收集技术。

**📊 数据集**

在WikiText‑2进行模型校准与评估，并在Lambda、PIQA、ARC‑e/c 以及代码生成（HumanEval）和数学（MMLU）任务上进行零样本评估。

**📈 对比分析**

与随机、幅度剪枝、梯度方法及无剪枝基线比较，GHOST 在 50% 状态稀疏下仅增加约 1 点困惑度，在 70% 稀疏时仍保持稳定；相较于梯度方法，它节省显存并避免分布偏移导致的性能崩溃。

**⚠️ 局限性**

需要前向统计提取导致一定运行时开销，极端稀疏或极小模型易失效，且方法依赖于特定的SSM结构，迁移到其他变体时需做适配。

---

## 101. "Sorry, I Didn't Catch That": How Speech Models Miss What Matters Most

**arXiv ID:** 2602.12249 | [PDF](https://arxiv.org/pdf/2602.12249v1)

**作者:** Kaitlyn Zhou `[一作]` (TogetherAI), James Zou `[通讯]` (Stanford University)

**通讯引用:** 38443 | [OpenAlex ID](https://openalex.org/A5005779176)

**关键词:** `Artificial Intelligence` `Recognition` `Data Synthesis` `Supervised Fine-Tuning` `Audio` `Text`

**🎯 论文内容**

评估并揭示公开部署的语音识别模型在真实场景下对美国街道名识别的高错误率及其对不同语言群体的差异性

**💡 创新点**

首次将街道名识别作为关键任务，提出使用开放源文本到语音技术生成多样化发音的合成数据并进行微调以显著提升错误率

**🔧 技术方法**

开源文本到语音模型XTTS、Whisper模型微调、Google Maps API距离评估

**📊 数据集**

SF Streets（2262句子，78位受试者）与US Streets（3600句子，97位受试者）数据集

**📈 对比分析**

与15种顶级商业模型（OpenAI, Deepgram, Google, Microsoft）比较，平均识别错误率44%，在非英语母语者中更高；使用合成数据微调后错误率下降近60%（相对基线）

**⚠️ 局限性**

仍受限于合成语音与真实人声差距，未能完全解决不同语言的细微口音差异，且对未覆盖语言的泛化效果有限

---

## 102. Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs

**arXiv ID:** 2602.11729 | [PDF](https://arxiv.org/pdf/2602.11729v1)

**作者:** Thomas Jiralerspong `[一作]` (Mila), Trenton Bricken `[通讯]` (Anthropic)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Auto Encoder` `Text`

**🎯 论文内容**

提出了跨架构模型差异检测方法，使用专用特征交叉编码器（Dedicated Feature Crosscoder, DFC）在不同架构的LLM间识别模型专属特征。

**💡 创新点**

创新点在于将特征空间划分为专属与共享三部分，并通过结构化约束强制模型专属特征不参与对方重建，显著提高了对模型专属行为的发现率。

**🔧 技术方法**

采用稀疏自编码器、交叉编码器、BatchTopK稀疏化、激活转向、模型拼接（stitching）等技术实现特征提取与跨模型对齐。

**📊 数据集**

使用公开的大规模预训练数据FineWeb、聊天数据LMSYS-Chat-1M，以及约100M token的中层激活对齐作为训练数据，并在合成玩具模型中进行验证。

**📈 对比分析**

通过专属度评分、共享度评估、激活转向验证等方法与标准交叉编码器对比，DFC在识别专属特征的召回率提升，精确度略低，但在真实模型上能发现诸如中国共产党对齐、美国特殊主义、版权拒绝机制等可解释行为，整体性能与标准交叉编码器相当。

**⚠️ 局限性**

局限包括：对真值概念缺乏，专属度评分为代理指标；发现过程对划分大小与随机初始化敏感；在基-微调对比中可能出现镜像特征；以及无法确定专属特征的来源（训练数据或架构）。

---

## 103. Uncertainty-aware Generative Recommendation

**arXiv ID:** 2602.11719 | [PDF](https://arxiv.org/pdf/2602.11719v1)

**作者:** Chenxiao Fan `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**通讯引用:** 42354 | [OpenAlex ID](https://openalex.org/A5038668215)

**关键词:** `Information Retrieval` `Recommendation System` `Reinforcement Learning` `Tabular`

**🎯 论文内容**

本文提出UGR框架，将不确定性纳入生成式推荐的偏好对齐过程中，通过不确定性加权奖励、难度感知优化以及显式置信度对齐来提升模型表现。

**💡 创新点**

创新点在于首次把生成模型的内在不确定性转化为可学习的奖励信号，并通过样本难度动态调节梯度，还引入置信度标记实现决策风险评估。

**🔧 技术方法**

技术包括基于SID的生成式推荐、GRPO无评判者的偏好优化、logit强度加权奖励、基于beam排名的难度权重以及置信度标记的层级对齐损失。

**📊 数据集**

实验使用Amazon Office Products、Industrial和Yelp三个真实数据集，分别对应不同领域的物品交互记录。

**📈 对比分析**

与SASRec、BIGRec、SPRec、TIGER、MiniOneRec、ReaRec和R^2ec等基线比较，UGR在HR@K和NDCG@K等指标上持续取得领先，显著提升推荐质量并稳定训练过程。

**⚠️ 局限性**

局限性包括仅在离线评估环境下验证，GRPO框架的可移植性待进一步研究，置信度校准可能受数据稀疏性影响，并且对超参数的依赖尚需深入探讨。

---

## 104. Code2Worlds: Empowering Coding LLMs for 4D World Generation

**arXiv ID:** 2602.11757 | [PDF](https://arxiv.org/pdf/2602.11757v1)

**作者:** Yi Zhang `[一作]` (Peking University), Hao Tang `[通讯]` (Peking University)

**通讯引用:** 52880 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `AI Code Assistant` `Large Language Model` `Retrieval-Augmented Generation` `Vision Language Model` `Text` `Benchmark`

**🎯 论文内容**

提出 Code2Worlds 框架，利用编程型大语言模型将文本指令转化为可执行的 4D 场景代码，并通过闭环自我反思实现物理一致性。

**💡 创新点**

创新点包括双流结构（对象生成与环境编排）与基于 VLM 的物理-aware 闭环自我修正；同时构建了 Code4D 基准用于 4D 生成评估。

**🔧 技术方法**

采用检索增强的参数生成、Infinigen 代码生成器、PostProcess Agent 动态脚本、VLM-Motion Critic 视觉自我批评、Blender 物理仿真等技术。

**📊 数据集**

使用 Infinigen 参数库、代码库、以及自建 Code4D 数据集（涵盖对象、场景与动态场景三维度）。

**📈 对比分析**

与 Infinigen、3D-GPT、SceneCraft、ImmerseGen 以及 Stable Video Diffusion 等基线对比，Code2Worlds 在 SGS、Richness、HRS、物理失效率、运动平滑度等指标均显著优于对手，提升 41% SGS 与 49% Richness。

**⚠️ 局限性**

主要限制包括计算开销大、对 LLM 依赖导致潜在偏差、以及对复杂物理现象（如多体碰撞）的精细建模仍有限。

---

## 105. An Empirical Study of the Imbalance Issue in Software Vulnerability Detection

**arXiv ID:** 2602.12038 | [PDF](https://arxiv.org/pdf/2602.12038v1)

**作者:** Yuejun Guo `[一作]` (Luxembourg Institute of Science and Technology), Yves Le Traon `[通讯]` (University of Luxembourg)

**通讯引用:** 16965 | [OpenAlex ID](https://openalex.org/A5040574362)

**关键词:** `Software Engineering` `Anomaly Detection` `Transformer` `Supervised Fine-Tuning` `Tabular`

**🎯 论文内容**

对软件漏洞检测中的不平衡问题进行系统的实证研究，评估九个开源 C 语言函数级数据集和两种主流深度学习模型的性能，并检验七种跨领域不平衡处理方法的效果。

**💡 创新点**

首次在漏洞检测领域完成了多数据集、多模型、多方法的全量对比，揭示了传统评估指标（如准确率）误导性、不同指标对方法效果的敏感性，并指出了外部因素（漏洞类型缺失、难度、分布漂移）对方法有效性的影响，为开发专门针对漏洞检测的不平衡解决方案提供了实证依据。

**🔧 技术方法**

使用 CodeBERT 与 GraphCodeBERT 作为基线模型，结合数据层方法（随机欠采样、随机过采样、对抗式增强）、模型层方法（阈值移动、Mean False Error 损失、Class-Balanced 损失）以及焦点损失（Focal Loss）来处理不平衡。

**📊 数据集**

九个函数级数据集：Devign(FFmpeg、QEMU)、Lin2018(Asterisk、FFmpeg、LibPNG、LibTIFF、Pidgin、VLC)、CodeXGLUE Devign（FFmpeg+QEMU 混合），全部均为 C 语言开源项目。

**📈 对比分析**

通过比较基线与七种方法在精确率、召回率、F1 这三类指标上的表现，发现：无单一方法在所有指标上均最佳；焦点损失最能提升精确率；Mean False Error 与 Class-Balanced 最能提升召回率；随机过采样在 F1 上表现最优；传统的准确率与误报率误导性强，难以反映漏洞检测的真正效果。

**⚠️ 局限性**

实验仅限于 C 语言函数级数据集和两种模型，未覆盖其他语言或更复杂模型；外部因素（缺失漏洞类型、漏洞难度、分布漂移）对方法效果影响显著，但未给出针对性的改进；缺乏对新型不平衡解决方案的实验，仍需进一步研究。

---

## 106. Compiler-Guided Inference-Time Adaptation: Improving GPT-5 Programming Performance in Idris

**arXiv ID:** 2602.11481 | [PDF](https://arxiv.org/pdf/2602.11481v1)

**作者:** Minda Li `[一作]` (University of Southern California), Bhaskar Krishnamachari `[通讯]` (University of Southern California)

**通讯引用:** 23711 | [OpenAlex ID](https://openalex.org/A5063784062)

**关键词:** `Programming Languages` `AI Code Assistant` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

研究 GPT-5 在低资源、依赖类型的函数式语言 Idris 上的推理时自适应能力，并通过在 Exercism 平台上系统评估不同的迭代反馈策略（基于测试、文档、编译器错误）来提升其编程表现。

**💡 创新点**

首次量化 GPT-5 在低资源语言 Idris 的基线表现，并提出基于本地编译器诊断的迭代改进循环，证明编译器反馈是最有效的学习信号；同时比较了文档增强、错误手册与编译器反馈三种提示策略。

**🔧 技术方法**

使用迭代提示、检索增强提示、向量检索、编译器诊断回馈、Python 自动化脚本与 Exercism API，构建本地编译与测试循环。

**📊 数据集**

Exercism Idris 56道练习题（与 Python 50道、Erlang 47道的交叉子集）作为评估基准；此外使用官方 Idris 参考手册和错误文档手册作为外部知识源。

**📈 对比分析**

对照零-shot、1 次/5 次基于平台测试反馈、错误文档手册、官方参考手册、20 次本地编译+测试迭代四种方法，记录每种方法成功解决的题目数。性能从基线 39%（22/56）提升至 96%（54/56），最佳效果来自编译器诊断驱动的迭代循环。

**⚠️ 局限性**

潜在的训练数据泄露、对新手/扰动题目泛化不足、仅针对 Idris 的实验、迭代成本高、缺乏对模型内部修复机制的深入解释。

---

## 107. DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation

**arXiv ID:** 2602.12160 | [PDF](https://arxiv.org/pdf/2602.12160v1)

**作者:** Xu Guo `[一作]` (Tsinghua University), Xiangwang Hou `[通讯]` (Tsinghua University)

**通讯引用:** 1772 | [OpenAlex ID](https://openalex.org/A5072643813)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Video` `Audio` `Multimodality` `Benchmark`

**🎯 论文内容**

提出并实现了 DreamID-Omni，一种统一的可控人类中心音视频生成框架，整合了参考基生成、视频编辑和音频驱动动画三大任务。

**💡 创新点**

通过 Symmetric Conditional DiT 将三种任务统一到同一架构；引入 Syn‑RoPE 与 Structured Caption 实现双层身份–音色去耦；并采用多任务渐进式训练提升模型在弱约束生成与强约束编辑/动画之间的兼容性。

**🔧 技术方法**

使用双流 Diffusion Transformer 与双向跨注意力；同步位置编码 Syn‑RoPE；结构化字幕；多任务渐进式训练；Classifier‑Free Guidance 等技术。

**📊 数据集**

在 Ovi 预训练模型的基础上，使用公开音视频数据集进行训练，并构建了新的 IDBench‑Omni 基准（包含 200 条多任务测试样本）。

**📈 对比分析**

与商业模型 Wan2.6、Ovi、LTX‑2 等开源方法，以及 R2V 任务的 VACE、Phantom、HunyuanCustom 等进行对比；DreamID‑Omni 在视频质量、音频质量、身份一致性、音色一致性和唇形同步等指标上均达到或超过现有 SOTA，尤其在多人物身份–音色绑定和说话者混淆方面表现突出。

**⚠️ 局限性**

训练过程计算成本高且需要大量标注的结构化字幕，模型在极端多人物或低资源语言场景下仍可能出现轻微的身份或音色漂移；且目前缺乏实时推理支持。

---

## 108. TS-Memory: Plug-and-Play Memory for Time Series Foundation Models

**arXiv ID:** 2602.11550 | [PDF](https://arxiv.org/pdf/2602.11550v1)

**作者:** Sisuo Lyu `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5612 | [OpenAlex ID](https://openalex.org/A5018828723)

**关键词:** `Machine Learning` `Knowledge Distillation` `Time Series` `Transformer` `Time Series`

**🎯 论文内容**

提出TS-Memory，一种可插拔的记忆模块，用来把离线kNN检索得到的分布式知识蒸馏到冻结的时间序列基础模型上，实现零检索推理；

**💡 创新点**

创新点在于将检索驱动的分布式修正转化为可训练的轻量化记忆网络，通过置信门控蒸馏实现离线检索与在线推理分离；

**🔧 技术方法**

使用冻结的TSFM骨干、离线kNN检索、量化目标蒸馏、Transformer记忆模块、Instance Normalization、置信门控损失与量化交叉熵；

**📊 数据集**

在八个主流长时序预测基准上验证：ETTh1/2、ETTm1/2、Electricity、Exchange-rate、Traffic、Weather；

**📈 对比分析**

与零检索基线、在线检索方法RAFT/TS‑RAG以及LoRA参数高效微调进行对比，TS‑Memory在MSE/MAE/CRPS上平均提升5–16%，且推理延迟几乎与冻结模型相同；

**⚠️ 局限性**

局限性包括离线kNN构造成本高、对检索知识的依赖导致跨域泛化受限，以及轻量化记忆容量上限可能限制极端长时序或高维情境下的表现。

---

## 109. Efficient Segment Anything with Depth-Aware Fusion and Limited Training Data

**arXiv ID:** 2602.11804 | [PDF](https://arxiv.org/pdf/2602.11804v1)

**作者:** Yiming Zhou `[一作]` (Fraunhofer Institute for Nondestructive Testing), Xavier Maldague `[通讯]` (Université Laval)

**通讯引用:** 15482 | [OpenAlex ID](https://openalex.org/A5027017464)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Depth Estimation` `Computational Efficiency` `Transformer` `Image`

**🎯 论文内容**

在轻量化分割框架中加入深度信息，改进 EfficientViT‑SAM 的边界细化与小物体分割

**💡 创新点**

将单目深度估计与 EfficientViT‑SAM 结合，采用中层融合与辅助损失，使模型在仅 11.2k 训练样本下实现超越更大模型的零样本性能

**🔧 技术方法**

使用 DepthAnything 估计深度、EfficientViT 轻量化 backbone、SAM 头部，以及加权交叉熵+Dice+IoU+辅助边缘损失的组合

**📊 数据集**

训练数据来自 SA‑1B 的 11.2k 样本，评估使用 COCO 与 LVIS 的 box‑prompted 与 point‑prompted 零样本分割

**📈 对比分析**

与 EfficientViT‑SAM‑L2、EfficientViT‑SAM‑XL1、SAM‑ViT‑H 对比，在 3/5 次点击的 mIoU 上提升 3–7%，并保持 30–62 FPS 的实时推理速度

**⚠️ 局限性**

深度分支几乎翻倍参数与运算量，虽然仍低于 SAM‑ViT‑H，但在极限资源设备上仍需一定算力；同时模型性能受预训练深度估计质量限制

---

## 110. Who Does What? Archetypes of Roles Assigned to LLMs During Human-AI Decision-Making

**arXiv ID:** 2602.11924 | [PDF](https://arxiv.org/pdf/2602.11924v1)

**作者:** Shreya Chappidi `[一作]` (University of Cambridge), Andra V. Krauze `[通讯]` (National Cancer Institute National Institutes of Health)

**关键词:** `Human-Computer Interaction` `Large Language Model` `Prompt Engineering` `Text` `Biomedical Data` `Electronic Health Records` `Computed Tomography`

**🎯 论文内容**

本论文提出了“人-LLM 架构类型”（human-LLM archetypes）概念，系统性识别并分类了 17 种在高风险决策场景中常见的人类与大型语言模型（LLM）交互模式，并通过对 113 篇实证研究的梳理与主题分析构建了这一体系。随后，作者在肺栓塞（PE）放射学报告的临床诊断案例中，对多种架构类型进行实验对比，评估其对 LLM 预测准确性、与外部参考信息的一致性以及生成解释的文本质量等维度的影响。

**💡 创新点**

创新点在于：①将人类与 LLM 的角色互动抽象为可复现的“架构类型”，为人机协同决策提供可操作的框架；②通过主题分析与案例实验，将这些架构类型与具体决策质量指标关联，揭示不同交互模式对决策结果的可观测影响；③提出并量化了多维度评估指标（准确率、敏感度、特异度、协同一致率、ROUGE、可读性等），为后续系统设计与评估提供了实证依据。

**🔧 技术方法**

核心技术主要为：①LLM 的提示工程与系统级配置（如 role-taker、model、explainer 等）；②对 113 篇文献的系统文献检索与主题编码；③在案例实验中使用 GPT‑4o（Azure Government OpenAI 接口）进行推理；④Python 与 Jupyter Notebook 进行实验脚本编写与指标计算；⑤文本相似度与可读性工具（rouge_score、textstat）。

**📊 数据集**

数据集为 INSPECT EHR 公共数据库中的肺栓塞 CT 报告，经过预处理后选取 100 条（50 正例、50 负例）作为实验样本；同时采用 113 篇研究论文作为文献来源，构成 17 种架构类型的样本空间。

**📈 对比分析**

实验方法：对每种产生显式预测的架构类型（Role Taker、Model、Judge、Second Opinion、Internal Explainer）分别构造提示，统一提问 100 条报告，收集 LLM 输出。评估指标包括：①准确率、敏感度、特异度；②对含外部参考信息的架构类型的“一致率”（与外部判断是否相同）；③解释质量的 ROUGE‑1/‑2/‑L 以及可读性分数。实验发现：不同架构类型在准确率与一致率上存在统计学显著差异（Cochran Q 与 McNemar 检验）；解释文本在相似度与可读性上表现出不同的复杂度与风格。

**⚠️ 局限性**

局限性包括：①仅使用 GPT‑4o 一个 LLM 版本，无法验证跨模型的普适性；②案例仅聚焦肺栓塞诊断任务，未涵盖其他高风险领域；③评估指标多为自动化度量，缺乏专家主观评价；④提示设计仍为手工调整，未系统化提示参数探索；⑤未考虑模型微调或 RAG 等更高级的知识补充方式。

---

## 111. Adaptive Physics Transformer with Fused Global-Local Attention for Subsurface Energy Systems

**arXiv ID:** 2602.11208 | [PDF](https://arxiv.org/pdf/2602.11208v1)

**作者:** Xin Ju `[一作]` (Stanford University), Gege Wen `[通讯]` (Imperial College London)

**通讯引用:** 969 | [OpenAlex ID](https://openalex.org/A5007168661)

**关键词:** `Machine Learning` `Transformer` `Graph Neural Network` `Mesh` `Physics Related`

**🎯 论文内容**

提出了一种能够在任意网格和自适应网格上直接学习地球地下能量系统的自适应物理变压器（APT）模型，解决了传统神经算子在几何灵活性和尺度扩展上的局限。

**💡 创新点**

创新点在于将全局 Perceiver 编码器与局部 GNO 编码器融合，并引入门控融合机制，同时首次实现直接从自适应网格数据学习，支持跨数据集预训练。

**🔧 技术方法**

采用了 Transformer 结构、Perceiver、Graph Neural Operator、门控融合、时间嵌入等技术，并使用相对 L_p 损失进行训练。

**📊 数据集**

使用了多种地下能量基准数据集，包括二维/三维 CO₂ 储存、石油开采、废水注入、热能储存等，涵盖规则网格、非规则网格、自适应网格以及嵌套细化网格。

**📈 对比分析**

在与 U-Net、FNO、U-FNO、MGN、MGN-LSTM 等传统基线模型的比较中，APT 在压力和饱和度误差上均显著优于同类模型，参数量更少，计算速度提升数千倍。

**⚠️ 局限性**

主要局限在于对高域尺寸/高分辨率数据的超分辨率能力受训练数据网格收敛的约束，且在极端不规则或极端尺度差异的网格上仍需进一步验证。

---

## 112. Methodological Variation in Studying Staff and Student Perceptions of AI

**arXiv ID:** 2602.11158 | [PDF](https://arxiv.org/pdf/2602.11158v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Human-Computer Interaction`

---

## 113. Latent-Variable Learning of SPDEs via Wiener Chaos

**arXiv ID:** 2602.11794 | [PDF](https://arxiv.org/pdf/2602.11794v1)

**作者:** Sebastian Zeng `[一作]` (Linnaeus University), Wolfgang Bock `[通讯]` (Linnaeus University)

**通讯引用:** 2426 | [OpenAlex ID](https://openalex.org/A5089882794)

**关键词:** `Machine Learning` `Stochastic Differential Equation` `Ordinary Differential Equation` `Time Series` `Physics Related`

**🎯 论文内容**

提出一种结构化潜变量框架，用空间谱Galerkin投影和Wiener‑卡西展开，将线性加性高斯噪声SPDE的无穷维动力学转化为有限维ODE，能够仅凭解的时空观测学习SPDE的概率律。

**💡 创新点**

创新点在于：①将Wiener‑卡西展开与Galerkin投影结合，利用一阶卡西闭合得到确定性传播子ODE；②在变分推理框架下同时推断潜在时间演化和噪声坐标，完全不需要观测噪声；③提供可解释的谱参数学习与随机激励分离。

**🔧 技术方法**

使用的技术包括：谱Galerkin投影、Wiener‑Itô卡西展开、一阶卡西闭合、潜变量ODE（latent ODE）变分推理、注意力编码器、RK4数值积分、对数似然与ELBO优化。

**📊 数据集**

实验数据为合成数据，分别在两种一维域（无界域的Ornstein–Uhlenbeck动态与有界域的Dirichlet边界热方程）中生成1000条解轨迹，采用时间网格200点、空间网格200/100点。

**📈 对比分析**

与DeepONet、FNO、NSPDE、DLR-Net等基线进行比较。结果显示在无界域时，模型在相对L²误差和RMSE上分别比基线提升约2.8倍；在有界域时，误差从约4.7降至1.1（相对L²）和从3.3降至0.68（RMSE）。此外，模型在方差演化和能谱等二阶统计量上与真解高度一致。

**⚠️ 局限性**

局限性在于仅适用于线性加性高斯噪声SPDE，一阶卡西闭合假设；对更一般的非线性或乘法噪声需要更高阶卡西展开，导致潜变量维数急剧增加；潜变量之间的相互依赖被近似为因子化，可能影响精度；实验仅在合成数据上验证，真实世界中观测噪声、边界效应等仍需进一步研究。

---

## 114. PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving

**arXiv ID:** 2602.12029 | [PDF](https://arxiv.org/pdf/2602.12029v1)

**作者:** Sunghyeon Woo `[一作]` (NAVER Cloud), Dongsoo Lee `[通讯]` (NAVER Cloud)

**关键词:** `Machine Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text` `Benchmark`

**🎯 论文内容**

设计了一种共享前填充（prefill）模块的分离式推理框架，使多模型在代理工作流中共享KV缓存，减少重复计算。

**💡 创新点**

创新点在于将prefill与decode模块拆分、冻结prefill、对decode进行 cache‑conditioned fine‑tuning，并在 vLLM 分散系统中实现路由和缓存交接。

**🔧 技术方法**

采用的技术包括 KV 缓存共享、disaggregated serving、cache‑conditioned fine‑tuning、prefix‑aware routing 以及 vLLM 的集成。

**📊 数据集**

使用的数据集包括 MetaMathQA‑40K、EvolInstruct‑Code‑80K、xLAM‑function‑calling‑60K，评测基准为 GSM8K/GSM+、HumanEval/HumanEval+ 与 BFCL。

**📈 对比分析**

与独立 pref/fix 方案及全量 fine‑tune 对比，PrefillShare 在多模型代理工作流下 p95 延迟降低 4.5×、吞吐量提升 3.9×，且准确率与全 fine‑tune 相当。

**⚠️ 局限性**

局限性：在极高并发时 KV handoff 开销成为瓶颈，跨模型调度需额外路由逻辑，且在极端负载下仍受解码端 KV 压力影响。

---

## 115. Towards Fair and Comprehensive Evaluation of Routers in Collaborative LLM Systems

**arXiv ID:** 2602.11877 | [PDF](https://arxiv.org/pdf/2602.11877v1)

**作者:** Wanxing Wu `[一作]` (Southern University of Science and Technology), Guanhua Chen `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 6528 | [OpenAlex ID](https://openalex.org/A5100665987)

**关键词:** `Computation and Language` `Large Language Model` `Text` `Benchmark`

**🎯 论文内容**

提出了RouterXBench三维评估框架并设计了ProbeDirichlet路由器，利用内部隐藏状态与Dirichlet层加权聚合，在多域任务上进行系统评估。

**💡 创新点**

创新点在于把路由器评估拆分为router ability、scenario alignment（LPM、MPM、HCR）和cross‑domain robustness三维度，并通过可学习的Dirichlet分布实现跨层隐藏状态聚合，从而显著提升跨域鲁棒性。

**🔧 技术方法**

采用内部隐藏状态提取、可学习的Dirichlet层加权聚合、轻量线性探针以及多域训练等技术。

**📊 数据集**

使用Alpaca、MMLU、Big‑Math、Magpie、MATH、MMLU Pro等六个代表性基准数据集。

**📈 对比分析**

与logit、embedding、verbose等多种基线对比，ProbeDirichlet在router ability上提升约16.7%，在高准确度场景提升约18.9%，并在多域、跨域情境保持稳定高性能。

**⚠️ 局限性**

仅在单一小/大模型对上验证，假设大模型优于小模型，缺乏多种架构、多种随机种子及复杂OOD条件的进一步验证。

---

## 116. Are Aligned Large Language Models Still Misaligned?

**arXiv ID:** 2602.11305 | [PDF](https://arxiv.org/pdf/2602.11305v1)

**作者:** Usman Naseem `[一作]` (Macquarie University), Agrima Seth `[通讯]` (Microsoft)

**关键词:** `Computation and Language` `Safety and Privacy` `Optimization` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Text`

**🎯 论文内容**

提出 Mis-Align Bench，一个统一安全、价值与文化维度的评估框架，并构建了包含 382,424 条样本的 SaVaCu 数据集，用于零样本下评估大型语言模型（LLM）的多维度一致性。

**💡 创新点**

创新点在于：①将安全、价值、文化三大维度合并评估，揭示单维度优化导致的交互式误差；②通过两阶段拒绝采样生成对齐与误差对，保证数据质量；③使用 SimHash 去重与条件生成扩大长尾领域，提升数据多样性；④系统性对比多种 LLM（general‑purpose、dimension‑specific fine‑tuned、open‑weight）在三维度联合约束下的表现。

**🔧 技术方法**

技术手段包括多标签分类（Mistral‑7B‑Instruct）、SimHash 近似去重、条件查询生成（Llama‑3.1‑8B‑Instruct）、两阶段拒绝采样（对齐/误差标注）、自动化评估模型（Llama‑3.1‑8B‑Instruct）、基于多模型生成（Gemma‑3‑27B、Phi‑3‑14B、Qwen‑2.5‑32B 等）。

**📊 数据集**

使用的主要数据集有 LLM‑Prompt‑Dataset、BeaverTails（安全域）、ValueCompass（价值域）、UNESCO UFCS（文化域），以及由 Mistral‑7B‑Instruct 与 Llama‑3.1‑8B‑Instruct 生成的扩展数据，最终构成 SaVaCu 数据集。

**📈 对比分析**

通过 Coverage、False Failure Rate (FFR) 与 Alignment Score (AS) 三个指标进行对比。general‑purpose aligned 模型（MARL‑Focal、TrinityX）在 AS 方面表现最佳（约 81%），单维度 fine‑tuned 模型虽在目标维度 Coverage 高（≈97%）但 FFR 超 50%，导致 AS 下降至 63–66%；open‑weight LLM（Gemma‑7B、DeepSeek‑7B）FDR 低但 Coverage 稍逊，整体 AS 处于中等水平。

**⚠️ 局限性**

局限性包括：①仅覆盖英文，无法直接推广至多语言场景；②受限于预定义的三大税onomies，可能遗漏新兴或细粒度规范；③评估完全自动化，依赖 LLM 评判器，可能带来偏差；④general‑purpose aligned 模型训练成本高，吞吐量低，限制了大规模复现与部署。

---

## 117. Value Alignment Tax: Measuring Value Trade-offs in LLM Alignment

**arXiv ID:** 2602.12134 | [PDF](https://arxiv.org/pdf/2602.12134v1)

**作者:** Jiajun Chen `[一作]` (New York University), Hua Shen `[通讯]` (New York University)

**通讯引用:** 11414 | [OpenAlex ID](https://openalex.org/A5101807811)

**关键词:** `Artificial Intelligence` `Large Language Model` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

本文提出了价值对齐税（VAT）框架，评估对齐干预对LLM价值系统的结构性影响。

**💡 创新点**

创新点在于从系统层面量化价值共变与耦合，揭示目标提升背后的“税负”与结构风险。

**🔧 技术方法**

采用基于斯瓦茨价值理论的规范性评估方法，使用Spearman相关、Gini系数和增益归一化偏差等统计指标。

**📊 数据集**

使用了29,568条基于文化背景的情境-价值-动作数据集，覆盖9个国家与11个社会领域。

**📈 对比分析**

通过对不同模型、提示量级、SFT与DPO等对齐策略的实验，发现同等目标增益可能导致不同的nVAT与集中度，表明VAT能捕捉传统评估忽略的结构差异。

**⚠️ 局限性**

局限在于仅评估固定的价值体系、模型与对齐方法，实验环境受限于提示与短期对齐，且VAT为描述性诊断，未给出可接受阈值。

---

## 118. IntTravel: A Real-World Dataset and Generative Framework for Integrated Multi-Task Travel Recommendation

**arXiv ID:** 2602.11664 | [PDF](https://arxiv.org/pdf/2602.11664v1)

**作者:** Huimin Yan `[一作]` (Alibaba Group), Xiangxiang Chu `[通讯]` (Alibaba Group)

**通讯引用:** 5406 | [OpenAlex ID](https://openalex.org/A5101512474)

**关键词:** `Information Retrieval` `Recommendation System` `Transformer` `Tabular`

**🎯 论文内容**

提出了一种面向多任务旅行推荐的解码器式生成框架，并构建了大规模公开数据集IntTravel；

**💡 创新点**

创新点在于通过任务导向信息保持（TIP）、任务特定选择门控（TSG）和任务感知情景因子化（TSF）三大模块，实现了任务间共享与专属信息的高效融合；

**🔧 技术方法**

采用了Transformer‑style的HSTU解码器，配合动态超连接、任务门控与可生成专家网络，整体实现端到端的多任务生成；

**📊 数据集**

使用了由中国地图导航平台采集的1.63亿用户、730万POI、41亿次交互构成的IntTravel数据集，并在十通视频推荐数据集上进一步验证；

**📈 对比分析**

与多种传统和最新多任务基线（PLE、STAR、M2M、APG、HiNet、MuSeNet、STEM‑Net、HoME）进行对比，IntTravel在四项任务上均位居或第二名，并在A/B测试中提升CTR 1.09%；

**⚠️ 局限性**

主要局限在于数据来源单一（仅包含中国主要城市的地图服务日志），且模型对更复杂的多模态或跨域特征的适应性尚未完全验证。

---

## 119. Multi Layer Protection Against Low Rate DDoS Attacks in Containerized Systems

**arXiv ID:** 2602.11407 | [PDF](https://arxiv.org/pdf/2602.11407v1)

**作者:** Ahmad Fareed `[一作]` (EPITA), Anne Pepita Francis `[通讯]` (EPITA)

**关键词:** `Cryptography and Security`

**🎯 论文内容**

提出并实现了针对容器化云环境的低速分布式拒绝服务（DDoS）缓解系统，采用多层防御（速率限制、动态黑名单、TCP/UDP头部分析、WAF、沙箱）并将零信任原则嵌入数据包验证流程；

**💡 创新点**

创新点在于将零信任理念与多层防御相结合，针对低速DDoS设计专属检测与隔离机制，利用动态黑名单与TCP/UDP细粒度分析提升隐蔽攻击检测能力，并通过微分段与沙箱实现隔离防护；

**🔧 技术方法**

技术手段包括Docker容器化部署、Apache2 + Mod‑Security WAF、速率限制中间件、动态黑名单文件同步、TCP/UDP头部特征分析、hping3/Mausezahn/iperf3流量生成与测试、微分段与零信任框架实现；

**📊 数据集**

使用模拟攻击与正常流量生成工具（hping3、Mausezahn、iperf3）构建的自定义流量数据集；

**📈 对比分析**

通过观察低速DDoS攻击场景下系统行为进行定性评估，未给出量化指标或与现有方案的对比，结果显示各防御层能有效过滤恶意流量并保持服务可用；

**⚠️ 局限性**

局限性包括缺乏量化性能评估、仅针对低速DDoS场景、未覆盖更广泛的攻击手法，未来可加入蜜罐与机器学习进行更深度检测与自动化响应。

---

## 120. Egocentric Gaze Estimation via Neck-Mounted Camera

**arXiv ID:** 2602.11669 | [PDF](https://arxiv.org/pdf/2602.11669v1)

**作者:** Haoyu Huang `[一作]` (University of Tokyo), Yoichi Sato `[通讯]` (University of Tokyo)

**通讯引用:** 12861 | [OpenAlex ID](https://openalex.org/A5045996641)

**关键词:** `Computer Vision and Pattern Recognition` `Domain Adaptation` `Transformer` `Video`

**🎯 论文内容**

研究并实现了颈挂式摄像头视角的视线估计任务，收集了首个对应数据集，并在该数据集上评估了基线模型与两种域适应策略。

**💡 创新点**

创新点在于：①首次提出颈挂式视角的视线估计任务并构建数据集；②设计了辅助视线内/外分类和基于几何对齐的多视角共学习两种域适应方法；③通过跨视角映射模型 VGGT 实现了头戴眼动数据到颈挂摄像头的注视投射。

**🔧 技术方法**

使用技术包括：Transformer‑based GLC 模型、VGGT 用于跨视角点对应、热图回归（KL 散度）、二分类损失（BCE）与对齐损失（MSE）等；同时采用了随机裁剪、时序采样等数据增强策略。

**📊 数据集**

数据集：约 4 小时、8 名受试者日常活动（如制咖啡、拼积木）录制的颈挂摄像头视频，配合同步的头戴眼动仪获取注视点，并通过 VGGT 计算颈挂视角下的注视坐标；包含注视分类（fixation、saccade、truncated、untracked）与相机变换信息。

**📈 对比分析**

评估方法：在 fixation 且视线在视野内的帧上采用 adaptive F1 评价；对比三种模型：①直接 fine‑tune GLC（F1 45.2%）；②加辅助内/外分类（F1 46.1%，提升 0.9%）；③多视角共学习（F1 44.6%），说明辅助分类略有帮助，而共学习无显著提升；单独分类器 F1 77.8%。

**⚠️ 局限性**

局限性：①颈挂视角注视点频繁超出视野，中心偏差弱，导致热图监督不易；②辅助分类改进有限；③多视角共学习未能提升，可能是因几何对齐难度大或模型架构不适应；④数据量和受试者多样性有限，缺乏对更大规模、不同设备的泛化验证。

---

## 121. Barriers to Discrete Reasoning with Transformers: A Survey Across Depth, Exactness, and Bandwidth

**arXiv ID:** 2602.11175 | [PDF](https://arxiv.org/pdf/2602.11175v1)

**作者:** Michelle Yuan `[一作]` (Oracle AI), Yassine Benajiba `[通讯]` (Oracle AI)

**关键词:** `Computation and Language` `Transformer` `Review/Survey Paper`

**🎯 论文内容**

综述Transformer在离散推理任务中的理论瓶颈，归纳电路复杂度、逼近理论与通信复杂度三大视角；

**💡 创新点**

首次系统整合三大理论框架，为Transformer精确算法执行失败提供统一解释；

**🔧 技术方法**

通过文献梳理、理论分析和案例对比；

**📊 数据集**

无实验数据集，仅基于已有研究与理论结果；

**📈 对比分析**

未进行实验对比，本文侧重理论阐释与概念性评估；

**⚠️ 局限性**

仅关注离散推理任务，未覆盖其他可能的框架和Transformer变体，且未给出实验验证。

---

## 122. Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception

**arXiv ID:** 2602.11858 | [PDF](https://arxiv.org/pdf/2602.11858v1)

**作者:** Lai Wei `[一作]` (Shanghai Jiao Tong University), Weiran Huang `[通讯]` (Ant Group)

**关键词:** `Computer Vision and Pattern Recognition` `Knowledge Distillation` `Reinforcement Learning` `Object Detection` `Transformer` `Reinforcement Learning` `Prompt Engineering` `Multimodality` `Image` `Benchmark`

**🎯 论文内容**

通过在训练阶段使用“Region-to-Image Distillation”将微观裁剪区域的高质量 VQA 数据迁移到全图，实现单前向推理下的细粒度视觉感知；同时构建了 ZoomBench 基准并提出双视角评估

**💡 创新点**

① 将推理时的工具式缩放操作迁移至训练时的蒸馏步骤；② 采用区域裁剪生成的高可信度 VQA 训练样本；③ 通过 bounding‑box 覆盖实现区域到全图的对齐；④ 构建双视角评估协议与 ZoomBench 细粒度基准

**🔧 技术方法**

强化学习（DAPO）训练 Qwen 系列模型；教师模型在微裁剪区域生成问题与答案；区域到全图蒸馏（带 bounding‑box 覆盖、难度过滤、提示工程）；对比实验、注意力覆盖分析

**📊 数据集**

原始无标签图像集合用于合成；ZoomBench 845 条高质量 VQA 样本；对比公开数据集如 DeepEyes、Thyme、Oasis、MM‑Self‑Instruct 等

**📈 对比分析**

与闭源（Gemini‑3‑Flash、GPT‑5.1）及开源 MLLM（Qwen3‑VL‑235B、GLM‑4.5V 等）和 Thinking‑with‑Images 代理模型对比；在细粒度感知（ZoomBench、HR‑Bench、VStar 等）、通用多模任务与 OOD 任务上均有显著提升；单前向推理速度约快 10 倍，且在细粒度任务上往往超过代理模型

**⚠️ 局限性**

尚未覆盖空间推理与多物体感知等任务，未在 TreeBench 等基准上评测；方法主要适用于信息增益有限的工具（如缩放、翻转等），无法替代需要外部信息获取的操作

---

## 123. ArGEnT: Arbitrary Geometry-encoded Transformer for Operator Learning

**arXiv ID:** 2602.11626 | [PDF](https://arxiv.org/pdf/2602.11626v1)

**作者:** Wenqian Chen `[一作]` (Pacific Northwest National Laboratory), Panos Stinis `[通讯]` (Pacific Northwest National Laboratory)

**通讯引用:** 33448 | [OpenAlex ID](https://openalex.org/A5002562845)

**关键词:** `Machine Learning` `Transformer` `Point Cloud`

**🎯 论文内容**

设计并验证了一种能够直接从任意几何点云编码的 Transformer 结构 ArGEnT，并将其集成进 DeepONet，用于多几何、多参数的算子学习与代理建模。

**💡 创新点**

创新点在于提出三种 Transformer 变体（self‑attention、cross‑attention、hybrid‑attention）可在不显式几何参数化的情况下通过点云直接编码任意几何，同时保持对查询点的灵活性，显著提升算子学习的泛化与准确性。

**🔧 技术方法**

采用了 Transformer 注意力机制、RoPE 位置编码、DeepONet 架构、点云与 SDF 输入、Mini‑batch 训练、Adam 优化和 MSE 损失等技术。

**📊 数据集**

使用了多个多物理、多几何的 CFD 与 FEA 数据集，包括：laminar 与湍流 airfoil（50/1000 样本）、lid‑driven cavity（3000 样本）、2D RFB（2813/2568/2346 配置）、3D jet engine bracket（6315 样本）等。

**📈 对比分析**

通过与标准 DeepONet、Point‑DeepONet、GraphSAGE 等基线比较，ArGEnT 在所有任务中都大幅降低 L2/MSE 误差（如 laminar airfoil 0.13×10⁻³ vs 6.05×10⁻³，turbulent 0.027×10⁻² vs 0.95×10⁻²，cavity 0.69×10⁻² vs 1.00×10⁻²，RFB 0.38×10⁻³ vs 3.5×10⁻¹），并在未见几何上保持较低误差。

**⚠️ 局限性**

局限性：self‑/hybrid‑attention 对查询点采样敏感；模型参数量相对传统 MLP/CNN 较大；对极复杂几何或高维输入仍受限；未引入物理约束，可能导致物理不一致。

---

## 124. RI-Mamba: Rotation-Invariant Mamba for Robust Text-to-Shape Retrieval

**arXiv ID:** 2602.11673 | [PDF](https://arxiv.org/pdf/2602.11673v1)

**作者:** Khanh Nguyen `[一作]` (University of Western Australia), Ajmal Mian `[通讯]` (University of Western Australia)

**通讯引用:** 20210 | [OpenAlex ID](https://openalex.org/A5089986388)

**关键词:** `Computer Vision and Pattern Recognition` `Retrieval` `Contrastive Learning` `Point Cloud` `Text` `Benchmark`

**🎯 论文内容**

提出了 RI-Mamba，一种基于状态空间模型的旋转不变点云编码器，用于文本到形状检索。

**💡 创新点**

创新点包括：① 通过局部参考框架（LRF）和全局参考框架（GRF）实现旋转不变序列化；② 设计了线性时间的方向嵌入并通过 FiLM 重集成姿态信息；③ 结合自动化三元组生成的跨模态对比学习，消除了人工标注需求；④ 在超过200个类别的任意姿态场景下首次建立文本到形状检索基准。

**🔧 技术方法**

采用了 Mamba 状态空间模型、Hilbert 曲线排序、PCA 参考框架、Feature-wise Linear Modulation（FiLM）、方向嵌入、跨模态对比学习（InfoNCE）等技术。

**📊 数据集**

使用了 ShapeNet、ABO、3D-FUTURE、Objaverse-LVIS 等大规模 3D 数据集进行预训练，评估数据集为 OmniObject3D、Text2Shape、ModelNet40。

**📈 对比分析**

与现有方法（如 SCA3D、TriCoLo、PointBERT、DuoMamba、LocoTrans、RI-Transformer 等）对比，RI-Mamba 在随机旋转（SO(3)）条件下显著优于非旋转不变模型，在对齐姿态条件下竞争力也不逊；在文本到形状检索、3D-3D 检索以及零样本分类任务上均实现了 SOTA 级别的性能。

**⚠️ 局限性**

局限性在于依赖 PCA 参考框架，对对称或近对称形状和噪声点云易出现不稳定性；对极端姿态变形的鲁棒性尚待进一步提升。

---

## 125. HyperDet: 3D Object Detection with Hyper 4D Radar Point Clouds

**arXiv ID:** 2602.11554 | [PDF](https://arxiv.org/pdf/2602.11554v1)

**作者:** Yichun Xiao `[一作]` (University of Edinburgh), Fangqiang Ding `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 979 | [OpenAlex ID](https://openalex.org/A5082306673)

**关键词:** `Robotics` `Object Detection` `Autonomous Driving` `Diffusion model` `Knowledge Distillation` `Point Cloud`

**🎯 论文内容**

构建了一个基于多帧、多雷达聚合与交叉校验的任务感知超4D雷达点云，用于提升雷达仅感知的3D检测性能。

**💡 创新点**

创新点在于将时空聚合、几何一致性校验与以LiDAR监督的前景扩散增强相结合，形成一个无需改动检测器的雷达前端。

**🔧 技术方法**

使用了多雷达时空聚合、几何一致性校验、基于BEV的条件扩散模型和一致性蒸馏等技术。

**📊 数据集**

在MAN TruckScenes数据集上进行训练与评估。

**📈 对比分析**

与原始单帧雷达输入相比，在VoxelNeXt和CenterPoint检测器上分别提升了约0.07和0.05的mAP，显著缩小雷达与LiDAR的性能差距。

**⚠️ 局限性**

局限性包括对训练数据量的依赖、对远距或小目标雷达信号不足、扩散模型无法补全缺失目标，以及BEV处理导致高度信息丢失。

---

## 126. Function-Space Decoupled Diffusion for Forward and Inverse Modeling in Carbon Capture and Storage

**arXiv ID:** 2602.12274 | [PDF](https://arxiv.org/pdf/2602.12274v1)

**作者:** Xin Ju `[一作]` (Stanford University), Gege Wen `[通讯]` (Imperial College London)

**通讯引用:** 969 | [OpenAlex ID](https://openalex.org/A5007168661)

**关键词:** `Machine Learning` `Diffusion model` `Ordinary Differential Equation` `Tabular` `Time Series`

**🎯 论文内容**

提出Fun-DDPS框架，用分离的函数空间扩散模型与可微神经算子后向/逆向建模，以实现稀疏观测下的碳捕获与储存（CCS）地质参数预测和不确定性量化。

**💡 创新点**

核心创新在于将地质先验（geomodel）与物理前向映射分离：先使用单通道扩散模型学习地质参数的分布，再用局部神经算子（LNO）作为可微前向物理代理；在逆向采样中通过算子梯度将稀疏动态观测转化为参数空间的全局指导，从而避免联合状态模型的高频伪影与指导衰减问题。

**🔧 技术方法**

采用函数空间扩散概率模型（DDPM）、Diffusion Posterior Sampling、局部神经算子（LNO）、U形神经算子（U‑NO）等深度学习技术，辅以梯度回传与概率流ODE实现采样。

**📊 数据集**

使用ECLIPSE（e300）数值模拟生成的12,000对（地质参数、CO₂饱和度）训练集，测试集1,500对；地质参数通过SGeMS生成；观测数据模拟为两口井（<1%覆盖）动态监测。

**📈 对比分析**

与联合状态扩散模型（Fun-DPS）以及纯粹的可微神经算子（LNO）基准比较；在前向任务中，Fun-DDPS在25%地质观测缺失时相对L₂误差仅为7.7%，比LNO高频误差的86.9%好11倍；在逆向任务中，Fun-DDPS与Fun-DPS对比基准Rejection Sampling的Jensen‑Shannon散度均<0.06，且Fun-DDPS生成的样本更物理一致，计算成本约比RS低4倍。

**⚠️ 局限性**

局限性包括：仅验证单时刻（30年）动态状态，未处理完整时空序列；对算子近似误差的容忍度未深入分析；实验规模仍受模拟数据约束，实际地质异质性与监测频率可能更复杂。

---

## 127. Automatic Simplification of Common Vulnerabilities and Exposures Descriptions

**arXiv ID:** 2602.11982 | [PDF](https://arxiv.org/pdf/2602.11982v1)

**作者:** Varpu Vehomäki `[一作]` (Aalto University), Kimmo K. Kaski `[通讯]`

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

研究了利用大型语言模型对CVE描述进行自动文本简化，并构建了半合成评测数据集。

**💡 创新点**

首次在网络安全领域提出基于GemmaAgent的检索增强生成简化框架，并评估多模型在意义保持与简化效果上的差异。

**🔧 技术方法**

采用GPT‑4o、Gemma 3 4B、GemmaAgent（术语抽取+RAG+简化）以及D‑SARI、BERTScore等自动与人工评测指标。

**📊 数据集**

随机挑选2025年发布的100条CVE描述，取40条做人工评测；其余60条用于模型开发；同时使用半合成简化参考。

**📈 对比分析**

与GPT‑4o、Gemma 3 4B及GemmaAgent对比，D‑SARI较低但GemmaAgent在意义保持上得分最高；GPT‑4o在FKGL上降到9.49，表明简化程度提升。

**⚠️ 局限性**

评测数据量有限、参考简化不足、模型对术语处理不稳定、意义保持仍有偏差，且缺乏充分的人机评估。

---

## 128. Evaluating LLM Safety Under Repeated Inference via Accelerated Prompt Stress Testing

**arXiv ID:** 2602.11786 | [PDF](https://arxiv.org/pdf/2602.11786v1)

**作者:** Keita Broadwater `[一作]` (Independent Researcher), Keita Broadwater `[通讯]` (Independent Researcher)

**关键词:** `Machine Learning` `Safty and Privacy` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

提出 Accelerated Prompt Stress Testing (APST)，通过在控制的解码条件下对相同或微调提示多次生成，评估大型语言模型在持续使用中的安全可靠性。

**💡 创新点**

创新点在于将可靠性工程中的加速失效测试思想引入LLM安全评估：将每一次推理视为独立的伯努利试验，用二项分布估计每次推理的失败概率，从而得到可量化的运营风险；并通过多温度、多深度采样揭示模型在浅层评估下隐藏的可靠性差异。

**🔧 技术方法**

使用的技术包括：重复采样策略、温度梯度控制、伯努利/二项式概率模型、置信区间估计、prompt级别的置信区间和交叉模型排名比较。

**📊 数据集**

使用的数据集为 AIR‑BENCH 2024 结构化安全提示，构造约90条提示，覆盖18个Level‑3风险类别；在四种指令微调LLM上（GPT‑4o、GPT‑OSS‑20B、Qwen‑2.5‑7B、Gemma‑3N‑E4B）进行实验。

**📈 对比分析**

与传统的单样本或浅层评估（AIR‑BENCH‑equivalent）对比，APST通过高深度采样揭示多模型在同一温度下的失败概率差异，模型排名出现显著偏差；例如 Gemma 在单样本评估中得分与 GPT‑OSS 相近，但在 APST 下的失败率约为其四倍，导致每日预期违规事件数差距达到 4–5 倍。

**⚠️ 局限性**

局限性包括：仅评估固定提示与解码配置，未考虑长期漂移、用户适应或多步交互；假设推理独立，未考虑可能的关联性；不适用于具有内部记忆或强化学习策略的代理系统。

---

## 129. Convex Markov Games and Beyond: New Proof of Existence, Characterization and Learning Algorithms for Nash Equilibria

**arXiv ID:** 2602.12181 | [PDF](https://arxiv.org/pdf/2602.12181v1)

**作者:** Anas Barakat `[一作]` (Singapore University of Technology and Design), Antonios Varvitsiotis `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 485 | [OpenAlex ID](https://openalex.org/A5078214509)

**关键词:** `Computer Science and Game Theory` `Optimization` `Reinforcement Learning`

**🎯 论文内容**

本文提出了广义效用马尔可夫游戏（GUMG）框架，并证明了其存在Nash均衡并给出一阶条件的等价性。

**💡 创新点**

创新点在于引入了逐玩家梯度支配性质，利用Brouwer定理直接证明均衡存在并得到结构化表征，同时给出了无模型的策略梯度算法和理论样本复杂度。

**🔧 技术方法**

主要技术包括梯度支配分析、Brouwer固定点定理、伪梯度法则以及对齐潜在函数的光滑性证明。

**📊 数据集**

本文未使用具体数据集，全部为理论分析。

**📈 对比分析**

与以往需完整模型或仅适用于零和cMG的研究相比，本方法在潜在GUMG上实现了 ε^-4/ε^-5 的样本复杂度，表明更高的样本效率。

**⚠️ 局限性**

局限性包括只考虑联合凸性/潜在结构的效用函数、未处理函数逼近及大规模状态动作空间。

---

## 130. Can Local Vision-Language Models improve Activity Recognition over Vision Transformers? -- Case Study on Newborn Resuscitation

**arXiv ID:** 2602.12002 | [PDF](https://arxiv.org/pdf/2602.12002v1)

**作者:** Enrico Guerriero `[一作]`, Øyvind Meinich-Bache `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Transformer` `Vision Language Model` `Large Language Model` `Supervised Fine-Tuning` `Video`

**🎯 论文内容**

评估在新生儿复苏视频中的细粒度活动识别，比较传统 TimeSFormer 与本地 VLM LLaVA‑Next Video+Mistral 7B 的性能

**💡 创新点**

探究局部 VLM 与 LoRA 微调能否超越纯视觉 Transformer，展示 LoRA 微调可显著提升 F1 分数

**🔧 技术方法**

使用 Vision‑Language 模型 LLaVA‑Next Video、LLM Mistral 7B、TimeSFormer、LoRA、零样本 VLM 策略

**📊 数据集**

利用 13.26 小时模拟新生儿复苏视频数据集

**📈 对比分析**

在多标签 F1 评估下，FT‑C‑LoRA 达到宏平均 F1 0.91，显著高于 TimeSFormer 0.70

**⚠️ 局限性**

局限性：仅使用模拟数据，零样本方法易产生幻觉，模型规模受限，未验证在真实临床视频上的表现

---

## 131. PAC to the Future: Zero-Knowledge Proofs of PAC Private Systems

**arXiv ID:** 2602.11954 | [PDF](https://arxiv.org/pdf/2602.11954v1)

**作者:** Guilhem Repetto `[一作]` (École Normale supérieure of Rennes), Farinaz Koushanfar `[通讯]` (University of California)

**通讯引用:** 22754 | [OpenAlex ID](https://openalex.org/A5019931011)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Zero-Knowledge Proofs` `Tabular`

**🎯 论文内容**

本文提出了一种将PAC隐私与zk-STARK零知识证明相结合的框架，实现了在无信任云计算环境下可验证的隐私保护；

**💡 创新点**

创新点在于首次通过非交互式zk-STARK证明PAC隐私机制的正确性，并在保持后门噪声私密性的同时提供可公开验证的证明；

**🔧 技术方法**

技术实现采用RISC‑Zero框架、基于碰撞抗性哈希的zk-STARK、PAC隐私噪声生成算法和确定性循环改造；

**📊 数据集**

实验数据集为小规模K‑means、SVM与数据库统计查询的人工合成数据，维度与样本数可调；

**📈 对比分析**

与传统差分隐私或无证明的PAC隐私对比，证明开销呈线性（仿真RISC操作数随样本数或维度线性增长），在小到中等规模任务下保持可行；

**⚠️ 局限性**

局限性包括：证明生成耗时显著，受zk-STARK性能限制；对循环次数固定导致模型收敛不确定；适用于小规模数据，规模扩展性尚未验证；

---

## 132. STAR : Bridging Statistical and Agentic Reasoning for Large Model Performance Prediction

**arXiv ID:** 2602.12143 | [PDF](https://arxiv.org/pdf/2602.12143v1)

**作者:** Xiaoxiao Wang `[一作]` (Fudan University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21197 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `Artificial Intelligence` `Large Language Model` `Retrieval-Augmented Generation` `Tabular` `Benchmark`

**🎯 论文内容**

提出STAR框架，结合统计期望与知识驱动推理来预测大型模型在基准上的表现。

**💡 创新点**

首次将认知科学中的期望违背理论(EVT)引入模型性能预测，形成检索增强的统计期望与层级推理两步的融合方法。

**🔧 技术方法**

采用检索增强的约束概率矩阵分解（CPMF）生成统计期望，并使用LLM进行语义检索、家庭内分析、跨模型对比与可信度加权的推理调整。

**📊 数据集**

利用OpenCompass数据集（285模型、28基准，6032条有效分数）进行评估。

**📈 对比分析**

与多种统计基线（PCA、回归、PMF等）及纯LLM方法对比，STAR在极端稀疏（95%遮罩）下RMSE 8.75、SRCC 96.10%、总分 67.10，均优于其他方法。

**⚠️ 局限性**

局限性包括对检索质量的高度依赖、对极端模式偏移（尤其是新架构）的适应性仍有限，以及对LLM推理的解释与一致性仍需进一步提升。

---

## 133. Human-Inspired Continuous Learning of Internal Reasoning Processes: Learning How to Think for Adaptive AI Systems

**arXiv ID:** 2602.11516 | [PDF](https://arxiv.org/pdf/2602.11516v1)

**作者:** Hong Su `[一作]` (Chengdu University of Information Technology), Hong Su `[通讯]` (Chengdu University of Information Technology)

**通讯引用:** 153 | [OpenAlex ID](https://openalex.org/A5031030652)

**关键词:** `Artificial Intelligence` `Anomaly Detection` `Optimization` `Transformer` `Large Language Model` `Time Series` `Sequential`

**🎯 论文内容**

本文设计并验证了人类启发式连续学习框架，使 AI 系统能够在执行过程中实时学习和优化自身的推理流程、调度策略以及学习机制，并支持将预设逻辑逐步替换为学习得到的有效程序。

**💡 创新点**

核心创新点在于：① 将内部推理活动本身视为可学习对象，①通过顺序推理与并行学习融合实现学习与执行同在循环；② 支持预设代码被学习得到的程序动态替换；③ 引入学习‑to‑学习的层级机制，让学习策略本身也可演进。

**🔧 技术方法**

技术实现基于大型语言模型（DeepSeek LLM）作为推理核心，配合内部流程记录、并行学习模块、验证机制以及外部工具接口；实验中使用模拟温度传感器异常检测任务作为应用场景。

**📊 数据集**

数据集为自定义的温度传感器序列，包含 8 个时间戳，正常温度范围 [0,20]℃，异常范围 [-5,25]℃；实验在 20 次独立跑中生成并记录交互日志，未使用公开数据集。

**📈 对比分析**

比较方法包括：M_notlearning（无学习）、M_learning（利用先前交互日志）和 M_noFixCodeByLearning（学习并替换预设代码）。实验结果显示：平均运行时间从 32.42 秒降至 26.16 秒（减少 23.9%）；LLM 交互次数从 4 次降至 3 次，再降至 1 次；M_noFixCodeByLearning 在完成任务后仅需一次 LLM 调用，但整体耗时略高于 M_learning 由于代码生成与验证带来的额外开销。

**⚠️ 局限性**

局限性包括：① 对 LLM 计算资源依赖较大，学习过程在实时执行时可能受限；② 代码生成替换带来额外时间开销，需平衡学习成本与执行效率；③ 仅在单一传感器异常检测任务上验证，框架在更复杂、跨模态或物理交互环境中的泛化能力尚待进一步评估。

---

## 134. Partial GFlowNet: Accelerating Convergence in Large State Spaces via Strategic Partitioning

**arXiv ID:** 2602.11498 | [PDF](https://arxiv.org/pdf/2602.11498v1)

**作者:** Xuan Yu `[一作]` (University of Science and Technology of China), Yang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 34270 | [OpenAlex ID](https://openalex.org/A5100764445)

**关键词:** `Machine Learning` `Generation` `Optimization` `Flow-based Model` `Sequential`

**🎯 论文内容**

本文提出了一种基于分区的生成流网络（Partial GFlowNet），通过规划器将探索限制在重叠的局部状态子空间中，并结合部分局部搜索来加速在大状态空间中的收敛。

**💡 创新点**

创新点包括：① 用规划器划分并动态切换重叠的子空间；② 采用启发式得分更新策略指导子空间选择；③ 将局部搜索改造为可在子空间内执行的 Partial Local Search（PLS）。

**🔧 技术方法**

技术手段涵盖：生成流网络（GFlowNet）框架、规划器（Planner）与启发式得分更新、局部搜索算法、以及多种训练目标（FM、DB、TB、SubTB）。

**📊 数据集**

实验数据集包括：分子设计（SMILES，10^24 条终态）；序列生成（120 位二进制序列，2^120 状态，60 个目标序列）；RNA 结合（14 位 RNA，4^14 状态）。

**📈 对比分析**

与传统 GFlowNet、本地搜索、纯分区搜索等方法相比，使用 PLS 能在相同训练步数下显著提升模式数和 Top‑k 奖励（例如在分子设计任务中模式数提升至 141115，奖励提升至 8.508），实验结果表明该方法在大、小时状态空间均表现出更快的收敛速度和更高的多样性。

**⚠️ 局限性**

局限性在于：① 需要手动设置子空间有效概率 p 与启发式阈值；② 规划器的启发式策略在极大状态空间中可能仍需改进；③ 对极端稀疏奖励场景的鲁棒性尚待进一步验证。

---

## 135. Transmit or Idle: Efficient AoI Optimal Transmission Policy for Gossiping Receivers

**arXiv ID:** 2602.12264 | [PDF](https://arxiv.org/pdf/2602.12264v1)

**作者:** Irtiza Hasan `[一作]` (University of North Carolina at Charlotte), Ahmed Arafa `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 2253 | [OpenAlex ID](https://openalex.org/A5027897280)

**关键词:** `Information Theory` `Optimization` `Reinforcement Learning`

**🎯 论文内容**

研究单源两接收器的gossiping网络，采用平均成本马尔可夫决策过程最小化AoI与传输成本，并给出了最优传输/空闲策略。

**💡 创新点**

提出了在可靠gossiping条件下的年龄差阈值结构与最小年龄激活机制，并证明该结构是最优的，从而实现显著的成本节约。

**🔧 技术方法**

使用平均成本MDP、相对价值迭代（RVI）、价值函数单调性与对称性证明、阈值分析等理论技术。

**📊 数据集**

论文通过仿真评估，采用参数化的传输成功概率（p1、p2、p_v1、p_v2）和传输成本C_tx，而非真实数据集。

**📈 对比分析**

与MAF、MAFT、TPO和随机基线比较。实验表明，在gossiping更可靠且传输成本较高的场景下，AoI‑optimal策略的平均成本明显低于所有基线。

**⚠️ 局限性**

局限性：仅考虑两接收器的对称信道模型；未扩展到多接收器或时变信道；阈值结构的证明仅在gossiping相对可靠的前提下成立。

---

## 136. SIGHT: Reinforcement Learning with Self-Evidence and Information-Gain Diverse Branching for Search Agent

**arXiv ID:** 2602.11551 | [PDF](https://arxiv.org/pdf/2602.11551v1)

**作者:** Wenlin Zhong `[一作]` (Zhejiang University), Kun Kuang `[通讯]` (Zhejiang University)

**通讯引用:** 2530 | [OpenAlex ID](https://openalex.org/A5041727387)

**关键词:** `Computation and Language` `Reinforcement Learning` `Optimization` `Reinforcement Learning` `Prompt Engineering` `Text`

**🎯 论文内容**

提出SIGHT框架，利用自证支持（SES）对搜索结果进行降噪，并通过信息增益（IG）驱动多分支探索，提升LLM在多轮搜索问答中的推理准确性与效率。

**💡 创新点**

创新点在于：① SES标签机制实现即时降噪并与奖励耦合；② IG度量用于定位关键信息状态，指导动态提示（去重、反思、分支）实现精准探索；③ 在训练中使用Group Relative Policy Optimization与掩蔽提示相结合，使模型内部化高效探索策略。

**🔧 技术方法**

技术包括：强化学习（GRPO）、信息增益评分、动态提示干预、SES标签生成、分支采样与共享前缀、奖励设计（答案正确率、SES得分、格式约束）以及掩蔽训练策略。

**📊 数据集**

在七大问答基准上评测：单跳（NQ、TriviaQA、PopQA）与多跳（HotpotQA、2WikiMultihopQA、Musique、Bamboogle），使用2020年维基百科检索与E5-base-v2检索器，基于Qwen2.5-3B/7B模型。

**📈 对比分析**

与基线（Direct、CoT、GRPO、RAG、ReAct、Search-o1、Search-R1、Tree-GRPO、ARPO）对比，SIGHT在单跳与多跳任务中均取得最高EM，尤其多跳场景提升2–5个百分点，同时平均工具调用量降至1–2次，显著降低计算成本。

**⚠️ 局限性**

局限性包括：对低质量检索结果的鲁棒性尚未彻底验证；对极长推理路径（>5跳）或需要深层外部知识检索的场景仍可能出现误区；实现复杂度较高，需要手工调节阈值和提示模板。

---

## 137. Improved Online Algorithms for Inventory Management Problems with Holding and Delay Costs: Riding the Wave Makes Things Simpler, Stronger, & More General

**arXiv ID:** 2602.12175 | [PDF](https://arxiv.org/pdf/2602.12175v1)

**作者:** David Shmoys `[一作]` (Cornell University), Seeun William Umboh `[通讯]` (University of Melbourne)

**通讯引用:** 159 | [OpenAlex ID](https://openalex.org/A5066531115)

**关键词:** `Data Structures and Algorithms` `Optimization` `Reinforcement Learning`

**🎯 论文内容**

本文提出了针对联合补货问题（JRP）的在线算法，能够处理任意单调的持有成本和延迟成本；同时改进了单品批量订货问题的竞争比；

**💡 创新点**

创新点在于：①去除了原先需要成本统一的限制，只需满足单调性；②引入波前双重变量和“预留服务”策略（按需求的“虚拟截止时间”排序）实现更高效的订单决策；③通过仿真模拟未来双重变量演化，进一步压缩竞争比；

**🔧 技术方法**

主要技术是基于原始-对偶（primal‑dual）方法的在线变体，维护波前双重变量，利用对偶约束触发订单；对偶变量的增长速率与需求的持有/延迟成本梯度一致；采用“预留服务”与“虚拟截止时间”对未成熟需求排序；对偶变量的仿真用于决定额外服务的项目；

**📊 数据集**

文中未使用任何真实数据集，全部以理论分析与合成实例为主；

**📈 对比分析**

通过对比原来 30‑competitive 的统一成本算法和单品 3‑competitive 算法，实验/理论证明新的算法在任意单调成本下实现 5‑competitive（JRP）和 φ+1≈2.681‑competitive（单品），显著提高了性能；

**⚠️ 局限性**

限制包括：仍需满足单调性，非单调成本会导致问题与集合覆盖等 NP‑难问题等价；算法的实现复杂度较高，且未在实际场景中验证。

---

## 138. GP2F: Cross-Domain Graph Prompting with Adaptive Fusion of Pre-trained Graph Neural Networks

**arXiv ID:** 2602.11629 | [PDF](https://arxiv.org/pdf/2602.11629v1)

**作者:** Dongxiao He `[一作]` (Tianjin University), Di Jin `[通讯]` (Tianjin University)

**通讯引用:** 6489 | [OpenAlex ID](https://openalex.org/A5012455357)

**关键词:** `Machine Learning` `Domain Adaptation` `Representation Learning` `Graph Neural Network` `Contrastive Learning` `Graph`

**🎯 论文内容**

提出一种跨域图提示学习方法 GP2F，使用冻结的预训练 GNN 分支和轻量级适配器分支并通过对比损失和拓扑一致性融合损失实现跨域微调。

**💡 创新点**

通过理论分析证明两分支融合能降低估计误差，首次将冻结分支与适配分支的对比和拓扑一致性融合引入跨域图提示学习，并在多任务场景中验证其有效性。

**🔧 技术方法**

采用双分支 GNN 结构、轻量级适配器、对比损失对齐两分支、拓扑一致性融合损失约束融合、以及自适应权重 α 进行动态融合。

**📊 数据集**

在节点分类任务上使用 Cora、CiteSeer、PubMed、Computers、Photo、CS、WikiCS、ogbn-arxiv、ogbn-products；在图分类任务上使用 PROTEINS、MUTAG、DD、COX2、BZR、ENZYMES 等公开基准。

**📈 对比分析**

与全微调、线性探测、现有 GPL 方法和跨域 GPL 方法对比，GP2F 在 1-shot/3-shot/5-shot 节点分类和 50-shot 图分类任务中多次获得最优或第二优成绩，平均提升 1–3% 左右。

**⚠️ 局限性**

对小样本、分布差异较大的小数据集仍表现不佳；方法对预训练分支和适配器的超参数（如 adapter 维度 r）敏感；在极大图上仍需采样策略，计算成本相对较高。

---

## 139. Fighting MRI Anisotropy: Learning Multiple Cardiac Shapes From a Single Implicit Neural Representation

**arXiv ID:** 2602.11436 | [PDF](https://arxiv.org/pdf/2602.11436v1)

**作者:** Carolina Brás `[一作]` (Amsterdam UMC), Ivana Išgum `[通讯]` (Amsterdam UMC)

**通讯引用:** 16249 | [OpenAlex ID](https://openalex.org/A5084070018)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Generation` `Restoration` `Auto Encoder` `Image` `Point Cloud` `Biomedical Data` `Magnetic Resonance Imaging` `Computed Tomography`

**🎯 论文内容**

利用高分辨率CTA数据训练单一神经隐式函数，联合重建低分辨率CMRI中的左心室血池(LVBP)、心肌(MYO)和右心室(RV)形状，并能在任意分辨率下生成完整、平滑的三维结构。

**💡 创新点**

创新点在于：①使用CTA的高空间分辨率作为形状先验，突破CMRI低通过面分辨率的限制；②通过单一隐式网络同时表示多种心脏结构，减少网络复杂度；③采用简化的MSE+正则化损失（去除双曲正切项），显著缩短训练时间和所需数据量；④在相同形状潜在空间中映射不同采样密度的点云，实现对低分辨率CMRI的自适应重建。

**🔧 技术方法**

技术手段包括：隐式神经表示（INR）/auto‑decoder框架、深度SDF (Signed Distance Function)、多层感知机(MLP)八层结构、点云采样与归一化、坐标变换到统一LPS参考系、L1/ MSE 损失与潜在向量正则化、Marching Cubes 提取零等值面。

**📊 数据集**

使用两套公开数据：① 153例CTA扫描（AUMC）并自动分割得到 LVBP、MYO、RV、LA；② 140例CMRI扫描（ischemic cardiomyopathy）配有 SAX 与 4CH 图像，并自动生成 ED 时刻的 LVBP、MYO、RV 分割。

**📈 对比分析**

评估方法：在 4CH 视角下提取 LVBP、MYO、RV 的重建与参考分割，计算 Dice 系数、Hausdorff 距离(HD)、95% HD 与平均对称表面距离(ASSD)。结果显示：RV Dice 0.91 ± 0.07、HD 6.21 ± 3.97 mm；MYO Dice 0.75 ± 0.13、HD 7.53 ± 5.13 mm；相比直接从 SAX 低分辨率提取的 4CH 分割，重建结果在所有指标上均有显著提升。

**⚠️ 局限性**

局限性包括：① 对低采样区域（基底、尖端）重建精度有限，易出现过/欠分割；② 仅在静息期 ED 时刻验证，未覆盖完整心动周期；③ 仅利用 4CH 视角，进一步采集更多长轴视角可能提升覆盖度；④ 对极少数病例的泛化能力尚需进一步评估。

---

## 140. WavBench: Benchmarking Reasoning, Colloquialism, and Paralinguistics for End-to-End Spoken Dialogue Models

**arXiv ID:** 2602.12135 | [PDF](https://arxiv.org/pdf/2602.12135v1)

**作者:** Yangzhuo Li `[一作]` (Xiamen University), Zhou Zhao `[通讯]` (Zhejiang University)

**通讯引用:** 11095 | [OpenAlex ID](https://openalex.org/A5001894984)

**关键词:** `Computation and Language` `Large Language Model` `Audio` `Multimodality` `Benchmark`

**🎯 论文内容**

提出并实现了WavBench，一个针对端到端语音对话模型的全方位评测基准，覆盖推理、口语化表达与声学特征三大维度；

**💡 创新点**

创新点在于：①构建了高难度Pro子集以考验模型的复杂推理与口语化简化能力；②引入完整的声学交互集（Explicit与Implicit）以评估模型的声学感知与生成；③使用多维度评测指标，结合Gemini评估对话自然性与声学一致性；

**🔧 技术方法**

技术手段包括：利用大型语言模型（如Qwen3-Max、GPT‑4o‑TTS）进行语料构造、口语化重写与音频合成；使用IndexTTS2、Emotion2Vec、Whisper‑Large‑V3等工具实现声学特征控制与质量检查；

**📊 数据集**

数据集来源涵盖15个公开数据集（如OpenBookQA、WildSpeech、AlpacaEval、MMLU、Arena‑Hard等），并通过LLM自动生成声学标注，最终形成17,577条语音样本（共76.5小时）；

**📈 对比分析**

与五种主流端到端模型（Qwen3‑Omni、Kimi‑Audio、Mimo‑Audio、Step‑Audio‑2‑mini、GPT‑4o‑Audio）进行对比，GPT‑4o‑Audio在Pro子集平均得分58.23/100领先，但整体模型仍距人类自然口语表现；

**⚠️ 局限性**

局限性：①声学生成仍面临低分辨率、背景音不自然；②多轮隐式交互中声学一致性难以保持；③数据集虽多样但对真实场景多模态融合仍不足；③评估依赖Gemini等闭源模型，缺乏完全公开可复现性。

---

## 141. PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories

**arXiv ID:** 2602.12080 | [PDF](https://arxiv.org/pdf/2602.12080v1)

**作者:** Hyunsung Kim `[一作]` (Korea Advanced Institute of Science and Technology), Chanyoung Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1972 | [OpenAlex ID](https://openalex.org/A5101629749)

**关键词:** `Machine Learning` `Classification` `Object Tracking` `Graph Neural Network` `Reinforcement Learning` `Video`

**🎯 论文内容**

提出了 PathCRF，一种只使用球员跟踪数据就能自动检测足球比赛中球控、传球等在场事件的框架。

**💡 创新点**

创新点包括：①将事件检测建模为在全连动态图上按时间序列选边的路径推断问题；②采用动态掩码条件随机场（CRF）在训练和推断时强制执行物理约束，避免非法转换；③在边嵌入上动态学习发射和转移得分，提高对不同比赛阶段的适应性。

**🔧 技术方法**

核心技术包括：基于 Set Attention 的社交-时间编码器（PPE-FPE+PE‑SABs）、动态边嵌入、动态掩码 CRF、Viterbi 解码以及辅助分类损失。

**📊 数据集**

使用公开的 Sportec Open DFL 数据集（德甲 1、2 级联赛 7 场比赛）进行训练与评估。

**📈 对比分析**

与传统球轨迹后处理、无 CRF 的独立分类、贪心/全局约束解码以及多种静态/动态 CRF 变体对比，PathCRF 的边准确率 69.64%、事件 F1 75.69%，并且零非法转换，显著优于其他方法。

**⚠️ 局限性**

局限性：仅处理在场事件，无法识别无球动作；依赖高质量球员跟踪数据；在极端遮挡或快速动作时仍可能出现误检；未来需加入视觉信息和更细粒度事件类型。

---

## 142. Dueling over Multiple Pieces of Dessert

**arXiv ID:** 2602.11486 | [PDF](https://arxiv.org/pdf/2602.11486v1)

**作者:** Simina Brânzei `[一作]` (Purdue University and Google Research), Reed Phillips `[通讯]` (Purdue University)

**关键词:** `Computer Science and Game Theory` `Optimization` `Reinforcement Learning`

**🎯 论文内容**

本文研究在重复公平分配游戏中，分析 Alice 通过分割 cake 并利用 Bob 的选择来实现近似 Stackelberg 收益的在线学习问题。

**💡 创新点**

创新点在于将公平分割与 Stackelberg 在线学习相结合，给出不同 k‑cut 与 Bob 学习率已知/未知情况下的最优子线性/线性 regret 上下界。

**🔧 技术方法**

采用二分搜索、离散化、构造性反例与复合回归分析等在线学习与游戏理论技术。

**📊 数据集**

该研究为纯理论分析，无使用具体数据集。

**📈 对比分析**

通过理论证明展示了 O(√(Tk)log(Tk))、O(T^{2+α/3}) 等上界，并给出匹配下界，说明在不同情形下 Alice 的 regret 受限。

**⚠️ 局限性**

主要局限是存在上界与下界的指数差距以及对可测划分游戏中的全局最佳 regret 仍不确定。

---

## 143. TADA! Tuning Audio Diffusion Models through Activation Steering

**arXiv ID:** 2602.11910 | [PDF](https://arxiv.org/pdf/2602.11910v1)

**作者:** Łukasz Staniszewski `[一作]` (Warsaw University of Technology), Kamil Deja `[通讯]` (Warsaw University of Technology)

**通讯引用:** 10725 | [OpenAlex ID](https://openalex.org/A5070627781)

**关键词:** `Sound` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Contrastive Learning` `Auto Encoder` `Audio`

**🎯 论文内容**

通过激活补丁定位音频扩散模型中控制音乐概念的关键注意力层，并利用对比激活加法与稀疏自编码器在这些层上实现对音频属性（如节奏、情绪、性别、乐器）的精准调节。

**💡 创新点**

证明多种文本到音频扩散模型仅有少数共享注意力层承担高层语义控制，提出仅在这些功能层上进行激活调节即可实现高精度、低质量损失的音乐属性操控，并通过稀疏自编码器进一步提升可解释性与控制粒度。

**🔧 技术方法**

使用激活补丁、对比激活加法（CAA）、稀疏自编码器（SAE）、跨注意力层提取、CLAP/MuQ 等音频‑文本相似度评估技术，以及 LPAPS、FAD、Audiobox Aesthetics 等指标。

**📊 数据集**

构造了包含对比提示词对的 MusicCaps 数据集，并用 GPT‑4 生成反义提示；实验在 10 秒和 30 秒音频片段上进行，使用 8 种随机种子生成 2048 条样本。

**📈 对比分析**

将功能层调节（{6,7}）与全层调节（ℒ）以及非功能层调节（ℒ∖{6,7}）进行对比。结果显示，功能层调节在对齐度、保真度与音频质量上均优于全层或非功能层调节，且在平滑性和语义精确性方面表现更好。

**⚠️ 局限性**

主要局限包括：仍需手动挑选并训练稀疏自编码器，超参数选择影响效果；实验仅覆盖 10/30 秒短片段，未验证长音频或多属性联合调节；对极细粒度属性调节的可扩展性尚待进一步研究。

---

## 144. Empirical Gaussian Processes

**arXiv ID:** 2602.12082 | [PDF](https://arxiv.org/pdf/2602.12082v1)

**作者:** Jihao Andreas Lin `[一作]` (Meta), Eytan Bakshy `[通讯]` (Meta)

**通讯引用:** 9875 | [OpenAlex ID](https://openalex.org/A5006700143)

**关键词:** `Machine Learning` `Optimization` `Anomaly Detection` `Recommendation System` `Gaussian Splatting` `Time Series` `Finance Related`

**🎯 论文内容**

本文提出 Empirical Gaussian Process (Empirical GP)，通过从历史数据样本路径直接估计 GP 的均值和协方差函数，构造非参数先验，并推导出闭式更新的 EM 算法来学习该先验；同时引入基于 SVD 的压缩和残差插值技术，以处理空间异质性观测与稀疏采样。

**💡 创新点**

创新点包括：① 从完整或稀疏历史样本直接估计 GP 先验，摆脱对手工核函数的依赖；② 设计可在不对齐、异质采样下收敛的 EM 算法，并给出解析 E 步和 M 步；③ 通过残差插值实现对未知区域的鲁棒外推，避免过度自信；④ 利用 SVD 对参考网格进行无损压缩，显著降低计算复杂度。

**🔧 技术方法**

使用的技术主要有：Gaussian Process 回归、最大似然估计、Expectation‑Maximization、核插值（基于基核）、奇异值分解（SVD）、KL 散度理论证明，以及对比实验中使用的传统核（RBF、Spectral Mixture）、统计模型（Naive、Seasonal Naive、AutoArima、AutoETS、AutoTheta）和深度学习基线（Crossformer、DLinear、N‑BEATS、DeepAR、TiDE、TFT、iTransformer、PatchTST）。

**📊 数据集**

数据集涵盖：金融股票价格（S&P 500）、气候碳排放（Mauna Loa CO₂）、GIFT‑Eval 时间序列预测基准（97 个数据集，144k 时序，177M 数据点）以及 LCBench 学习曲线数据（35 个数据集、2000 条超参配置）。

**📈 对比分析**

评估方法：对比 handcrafted 核函数、传统 GP、统计模型以及多种深度学习基线，使用 CRPS（连续排名概率分数）和 RMSE 进行量化。结果显示 Empirical GP 在统计模型族中排名最高，并在 GIFT‑Eval 中击败四个深度学习基线；在学习曲线外推任务中，在 RMSE 和 CRPS 上均优于基线（尤其是数据稀疏时）。

**⚠️ 局限性**

局限性：在历史样本极少时，估计的先验可能不稳健，需要更强的先验假设；当历史数据量巨大时，基于基础模型的深度学习方法可能获得更高表达能力；外推至完全未知区域仍可能出现自信不足或过度自信，需要依赖残差插值修正。

---

## 145. Geometry of Uncertainty: Learning Metric Spaces for Multimodal State Estimation in RL

**arXiv ID:** 2602.12087 | [PDF](https://arxiv.org/pdf/2602.12087v1)

**作者:** Alfredo Reichlin `[一作]` (KTH Royal Institute of Technology), Miguel Vasco `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 112 | [OpenAlex ID](https://openalex.org/A5063841377)

**关键词:** `Machine Learning` `Reinforcement Learning` `Robotic Intelligence` `Reinforcement Learning` `Contrastive Learning` `Multimodality` `Point Cloud`

**🎯 论文内容**

本文提出一种用于强化学习的多模态状态估计方法，学习一个结构化的潜在空间，并通过逆距离加权融合来自不同传感器的观测。

**💡 创新点**

创新点在于：①构造潜在空间的距离与环境中实现状态间最小动作数一致，实现几何不确定性解释，免除显式概率建模；②引入逆距离加权的自适应融合机制，自动抑制受噪声影响的模态；③在无噪声训练下即可获得对多种未知噪声的鲁棒性。

**🔧 技术方法**

技术手段包括：多模态编码器（每个传感器一个编码器）、潜在转移模型、对比学习损失（正负对样本）、预测损失、跨模态一致性损失，以及逆距离加权融合；随后将得到的潜在状态输入标准强化学习算法（如Soft Actor–Critic）。

**📊 数据集**

实验数据集：Mujoco 套件（Hopper-v5、HalfCheetah-v5、Ant-v5、Walker2d-v5、Humanoid-v5、InvertedPendulum-v5）与 Fetch 机器人套件（FetchPickAndPlace-v4、FetchSlide-v4），每个任务使用 RGB、深度、点云等多模态观测。

**📈 对比分析**

与六种基线（LinearComb、ConCat、CURL、GMC、α‑MDF、CORAL）在七类未见噪声（高斯、盐椒、补丁、拼图、纹理、失败、幻象）下进行对比。实验显示，本文方法在单模态与多模态噪声场景下均保持更高的累计奖励，鲁棒性最佳，并且无需在训练阶段加入噪声。

**⚠️ 局限性**

局限性包括：①使用欧氏距离导致度量空间对称，无法完全捕捉最小动作距离的非对称性；②假设环境动力学确定性，面对显著随机转移时表示能力受限；③在高随机性或大规模状态空间时可能需要更高维潜在空间，导致计算成本上升。

---

## 146. PRISM: A 3D Probabilistic Neural Representation for Interpretable Shape Modeling

**arXiv ID:** 2602.11467 | [PDF](https://arxiv.org/pdf/2602.11467v1)

**作者:** Yining Jiao `[一作]` (University of North Carolina at Chapel Hill), Marc Niethammer `[通讯]` (University of California San Diego)

**通讯引用:** 8856 | [OpenAlex ID](https://openalex.org/A5108610850)

**关键词:** `Machine Learning` `Anomaly Detection` `Generation` `Segmentation` `Biomedical Data` `Computed Tomography`

**🎯 论文内容**

构建了一个可估计形状分布、个体发育时间和空间可变不确定性的隐式神经形状建模框架，适用于医学形状分析。

**💡 创新点**

创新点包括：①将条件高斯场与隐式神经表示相结合，直接在解剖空间建模形状分布；②推导出闭式 Fisher 信息度量，可通过自动微分快速计算时序不确定性；③提出可逆编码器实现即时发育时间推断，避免传统的逐样本优化。

**🔧 技术方法**

技术手段：多层感知机隐式表示、极端分布拟合、最大似然估计、逆向编码器、Fisher 信息公式、自动微分、两阶段训练策略。

**📊 数据集**

数据集包括：①三个合成数据集 Starman(G)、Starman(L)、ANNY；②真实临床数据 Pediatric Airway（358 份 CT，含 31 份病理样本）。

**📈 对比分析**

与基准 A-SDF、NAISR 进行对比；在平均形状重建、全局与局部发育时间估计、个性化预测以及异常检测等任务中均取得与或优于基线的性能，尤其在局部时间推断和 OOD 检测上表现突出。

**⚠️ 局限性**

局限性：仅能处理单一标量协变量，未扩展到高维协变量；在退行性疾病长期预测方面尚未充分验证；对极端长周期预测的鲁棒性有待提升。

---

## 147. Bounded Local Generator Classes for Deterministic State Evolution

**arXiv ID:** 2602.11476 | [PDF](https://arxiv.org/pdf/2602.11476v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Operating Systems`

---

## 148. Improving HPC Code Generation Capability of LLMs via Online Reinforcement Learning with Real-Machine Benchmark Rewards

**arXiv ID:** 2602.12049 | [PDF](https://arxiv.org/pdf/2602.12049v1)

**作者:** Ryo Mikasa `[一作]` (Nagoya University), Takahiro Katagiri `[通讯]` (Nagoya University)

**通讯引用:** 660 | [OpenAlex ID](https://openalex.org/A5078063020)

**关键词:** `Machine Learning` `Reinforcement Learning` `Optimization` `AI Code Assistant` `Transformer` `Large Language Model` `Reinforcement Learning` `Benchmark`

**🎯 论文内容**

基于在线强化学习在超级计算机上使用GFLOPS反馈来提升LLM生成的高性能矩阵乘法代码。

**💡 创新点**

首次将实时运行时性能作为奖励结合分阶段质量多样性SQD算法进行LLM训练，显著提升代码速度。

**🔧 技术方法**

使用Qwen2.5 Coder 14B、Group Relative Policy Optimization (GRPO)、SQD算法、分布式GPU/CPU训练架构。

**📊 数据集**

双节点Genkai超级计算机和Flow超算的CPU集群，测试矩阵乘法（256×256）在不同编译器优化级别下。

**📈 对比分析**

对比不同编译优化、学习率和KL惩罚的实验，SQD训练最高可达549 GFLOPS，比基线提升约2.4×（O1）或18%（O3）。

**⚠️ 局限性**

仅针对单一核心矩阵乘法，未验证其他算子/规模、缺乏统计显著性，且受编译器级别的影响。

---

## 149. 3DGSNav: Enhancing Vision-Language Model Reasoning for Object Navigation via Active 3D Gaussian Splatting

**arXiv ID:** 2602.12159 | [PDF](https://arxiv.org/pdf/2602.12159v1)

**作者:** Wancai Zheng `[一作]` (Zhejiang University of Technology), Xinyi Yu `[通讯]` (Zhejiang University of Technology)

**通讯引用:** 2000 | [OpenAlex ID](https://openalex.org/A5101978642)

**关键词:** `Robotics` `Robotic Intelligence` `Optimization` `Transformer` `Vision Language Model` `Chain-of-Thought` `Gaussian Splatting` `Reinforcement Learning` `Point Cloud` `Multimodality`

**🎯 论文内容**

提出3DGSNav框架，利用主动感知构建3D Gaussian Splatting（3DGS）内存，并通过轨迹引导的自由视角渲染、结构化视觉提示与Chain‑of‑Thought（CoT）提问，提升零射对象导航（ZSON）中的视觉‑语言模型（VLM）空间推理与规划；

**💡 创新点**

①将3DGS作为VLM的持久记忆，实现高效的环境重建与查询；②结合自由视角优化与主动感知，主动获取缺失信息；③将结构化视觉提示与CoT融合，充分激活VLM的空间推理能力；

**🔧 技术方法**

3D Gaussian Splatting、主动感知+DBSCAN、轨迹引导的自由视角优化、结构化视觉提示、CoT提示、Gemini3-Pro规划VLM、GLM‑4.5v决策VLM、YOLOE实时目标检测、3DGS空间渲染与再验证；

**📊 数据集**

Habitat模拟器中的HM3Dv1、HM3Dv2、MP3D三大数据集；以及真实四足机器人在办公与酒店环境中的实测；

**📈 对比分析**

与BeliefMapNav、RL‑based ZSON、以及多种开源VLM（Qwen235b等）对比，3DGSNav在HM3Dv1/DM2/MP3D上平均提升SR 13.5%和SPL 32.08%；对RL‑based ZSON提升SR 203%和SPL 320%；对BeliefMapNav提升SR 25.25%和SPL 51.65%；在真实机器人上达到69.44% SR；

**⚠️ 局限性**

受限于VLM对图像理解的准确性，低层感知误差仍可能影响最终导航；目前仅针对对象导航任务，通用性与更大规模环境的适应性仍待进一步验证。

---

## 150. On the Complexity of Offline Reinforcement Learning with $Q^\star$-Approximation and Partial Coverage

**arXiv ID:** 2602.12107 | [PDF](https://arxiv.org/pdf/2602.12107v1)

**作者:** Haolin Liu `[一作]` (University of Virginia), Chen-Yu Wei `[通讯]` (University of Virginia)

**通讯引用:** 68163 | [OpenAlex ID](https://openalex.org/A5100783043)

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning`

**🎯 论文内容**

本文针对 Q* 逼近和部分覆盖的离线强化学习，给出了信息理论下界并提出了决策-估计系数（DEC）框架，统一并改进了现有理论和算法（如 CQL）的样本复杂度分析。

**💡 创新点**

创新点在于：①证明 Q* 可实现性与 Bellman 完整性不足以保证样本高效性；②提出基于 DEC 的离线决策原理，能够在不构造置信集的情况下实现政策中心的鲁棒性；③利用二阶性能差分引入正则化 MDP 的 ε⁻² 样本复杂度；④给出了低 Bellman 维数 MDP 在离线环境下的可学习性判定。

**🔧 技术方法**

主要技术包括信息理论下界构造、DEC（决策-估计）框架、贝尔曼一致性与权重可实现性假设、二阶性能差分定理、以及对 CQL 的正则化分析。

**📊 数据集**

文中未使用真实数据集，所有结果均为理论上界与下界，实验部分仅对经典离线 RL 算法（如 CQL）在合成环境中的性能进行了对比。

**📈 对比分析**

与现有方法相比，本文的 DEC 方案在部分覆盖下实现了更小的样本复杂度（ε⁻² 对比 ε⁻⁴），并在非表格情况下对 CQL 给出了首次理论保证；此外，新的低 Bellman 维数分析进一步揭示了传统方法的局限。

**⚠️ 局限性**

局限性包括：需要价值差距或正则化的凸性假设才能获得可控上界；在完全缺乏覆盖或 Bellman 完整性时仍可能出现指数样本需求；且对连续动作空间的分析仍依赖于特定的曲率条件。

---

## 151. Olmix: A Framework for Data Mixing Throughout LM Development

**arXiv ID:** 2602.12237 | [PDF](https://arxiv.org/pdf/2602.12237v1)

**作者:** Mayee F. Chen `[一作]` (Stanford University), Kyle Lo `[通讯]` (Allen Institute for AI)

**关键词:** `Machine Learning` `Large Language Model` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

为大规模语言模型的预训练构建一种高效、可扩展的数据混合框架，解决模型初始混合配置与域集演化过程中的混合重计算问题。

**💡 创新点**

（1）系统化研究离线混合框架的七大设计空间，并在 Olmo‑3 预训练中给出最佳配置；（2）提出基于虚拟域的混合重用机制（Mixture Reuse）并提供理论上限与经验验证；（3）在演化域场景下实现 95%+ 的性能提升，仅消耗 70%+ 的代理模型训练量。

**🔧 技术方法**

离线混合 schema（swarm + 回归 + 优化）；日志线性回归模型；KL 正则化约束优化；有限数据重复约束；虚拟域压缩与展开；基于 Dirichlet 分布的采样；在多域更新时使用子域重计算。

**📊 数据集**

DCLM 24 主题域（WebOrganizer 划分）作为基准域集；在此基础上添加 Stack‑Edu 编程语言、olmOCR 科学 PDF 等演化域；使用 52 项下游任务（数学、代码、常识 QA）评估模型性能。

**📈 对比分析**

与自然比例分布、全重计算、只重用已存在 swarm（Swarm Reuse）和两种混合重用策略进行对比。实验显示：Mixture Reuse 在 1B 参数、100B 训练 token 上取得 +11.6% BPB 提升，95% 以上达到全重计算的收益，仅使用 74% 的代理模型训练次数；相较于 Swarm Reuse 取得更小的性能差距。最终最佳混合在数据效率上比自然分布提升 3.05×，实现更快的收敛。

**⚠️ 局限性**

（1）仅针对离线混合框架，未涵盖在线动态混合；（2）理论分析基于日志线性回归，非参数模型难以推广；（3）需要人工选择需重计算的子域，缺乏自动化决策机制；（4）实验仅在 1B 参数模型上验证，未检验更大规模模型的可迁移性。

---

## 152. Adaptive Milestone Reward for GUI Agents

**arXiv ID:** 2602.11524 | [PDF](https://arxiv.org/pdf/2602.11524v1)

**作者:** Congmin Zheng `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18130 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `Machine Learning` `Reinforcement Learning` `Robotic Intelligence` `Large Language Model` `Reinforcement Learning` `Vision Language Model` `Multimodality`

**🎯 论文内容**

提出了 Adaptive Milestone Reward (ADMIRE) 机制，用可验证的、动态生成的里程碑来解决移动 GUI 代理在长周期任务中的时序信用分配问题。

**💡 创新点**

创新点在于：①将里程碑从成功轨迹中动态提炼并随代理学习进化；②采用不对称信用分配，对成功轨迹仅奖励里程碑点，对失败轨迹给与稠密的基准+里程碑奖励；③将里程碑匹配与奖励解耦，避免过程奖励的偏差与作弊。

**🔧 技术方法**

技术手段包括：基于 LLM 的里程碑生成与更新、Sentence‑BERT 语义匹配验证、GRPO 等强化学习算法、Qwen2.5‑VL‑3B/7B 作为视觉语言模型、分布式训练框架。

**📊 数据集**

数据集与环境：AndroidWorld、MobileMiniWob++（移动 GUI 基准），ALFWorld 与 WebShop（跨域测试）。

**📈 对比分析**

与传统 Outcome Reward、Process Reward 以及多种主流 RL 算法（GRPO、RLOO、DAPO）对比，ADMIRE 在 AndroidWorld 上平均提升 10%+ 成功率，在 MobileMiniWob++ 上也显著优于基线，且在 ALFWorld 与 WebShop 中保持强大的泛化能力。

**⚠️ 局限性**

局限性：①里程碑生成依赖 VLM 的推理能力，若 VLM 质量不足会影响奖励质量；②目前仅把里程碑用作辅助奖励，未探索其作为代理成功判定的代理信号。

---

## 153. PPTAM$η$: Energy Aware CI/CD Pipeline for Container Based Applications

**arXiv ID:** 2602.12081 | [PDF](https://arxiv.org/pdf/2602.12081v1)

**作者:** Alessandro Aneggi `[一作]` (Free University of Bozen-Bolzano), Andrea Janes `[通讯]` (Free University of Bozen-Bolzano)

**通讯引用:** 1720 | [OpenAlex ID](https://openalex.org/A5015575660)

**关键词:** `Software Engineering` `Tabular`

**🎯 论文内容**

开发了 PPTAMη，集成 GitLab CI 的能耗测量管线，实现对容器化 API 系统的持续能耗和性能回归测试。

**💡 创新点**

将能耗测量与负载生成、容器监控统一到 CI/CD 流水线中，提供可视化能耗回归检测；基于 PPTAM 架构且支持多工具插件。

**🔧 技术方法**

使用了 GitLab CI、Docker Swarm、Locust、cAdvisor、Perf、PowerJoular、RAPL、Python、Jupyter Notebook 等技术。

**📊 数据集**

以 JWT 认证的微服务为案例，四个提交版本作为实验数据。

**📈 对比分析**

通过对四个提交版本执行相同负载，收集响应时间、吞吐量、CPU/内存利用率、CPU/DRAM功耗，绘制箱线图和时间序列进行比较，发现 RS256 版本能耗更高，优化后能耗下降。

**⚠️ 局限性**

仅在单一基准 API 上验证，需人工配置测试环境；负载配置对可比性有影响；未扩展到更大规模或其他 CI 平台。

---

## 154. TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction

**arXiv ID:** 2602.11705 | [PDF](https://arxiv.org/pdf/2602.11705v1)

**作者:** Yuxiang Zhong `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**通讯引用:** 21295 | [OpenAlex ID](https://openalex.org/A5100684575)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Image Translation` `Gaussian Splatting` `Image` `Biomedical Data` `Computed Tomography`

**🎯 论文内容**

提出了一种面向稀疏投影和动态CT重建的几何感知高斯变形框架 TG‑Field，能在超稀疏视角下实现高质量重建，并通过时间条件化模型和运动流网络捕获呼吸运动。

**💡 创新点**

创新点包括：① 将多分辨率哈希编码器引入高斯原语的几何约束，显著提升稀疏视角下的几何一致性；② 引入时空注意力模块（STAB）解决空间时间哈希冲突，实现稳健的时间连续性；③ 设计细粒度运动流网络补偿非刚性局部变形；④ 使用预训练视觉基础模型进行语义一致性正则化，缓解光度监督不足。

**🔧 技术方法**

核心技术：3D 高斯 splatting、哈希网格编码器、时空注意力、运动流网络、DINO‑ViT 语义正则化、D‑SSIM 与 TV 正则化的联合损失。

**📊 数据集**

使用合成稀疏投影数据（5、10、20 视角）以及真实数据集：CBCT、XCAT 虚拟 4D 心肺数据、TCIA 4D‑Lung、SPARE 4D CBCT。

**📈 对比分析**

与传统 FDK、SART、NeRF 及 Gaussian‑splatting 方法（如 IntraTomo、SAX‑NeRF、R²‑Gaussian、HexPlane、K‑Planes、STNF4D、4DGS）比较，TG‑Field 在所有稀疏视角下 PSNR 均优 0.5‑0.9 dB、SSIM 亦显著提升，尤其在极稀疏（5 视角）和动态 4D 任务中表现最为突出。

**⚠️ 局限性**

局限性：训练与推理仍需显著 GPU 资源；哈希编码器对尺度、分辨率的设置较为敏感；对非 CT 成像或更大尺度三维场景的适应性尚未验证；语义正则化依赖预训练模型，对非标准解剖结构的鲁棒性有限。

---

## 155. Vascular anatomy-aware self-supervised pre-training for X-ray angiogram analysis

**arXiv ID:** 2602.11536 | [PDF](https://arxiv.org/pdf/2602.11536v1)

**作者:** De-Xing Huang `[一作]` (Chinese Academy of Sciences), Zeng-Guang Hou `[通讯]` (Macau University of Science and Technology)

**通讯引用:** 14467 | [OpenAlex ID](https://openalex.org/A5109333020)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Object Detection` `Transformer` `Contrastive Learning` `Image` `Biomedical Data`

**🎯 论文内容**

提出了一种基于血管解剖知识的自监督预训练框架VasoMIM，并构建了最大的X射线血管影像预训练数据集XA-170K。

**💡 创新点**

创新点包括：①使用Frangi滤波结合轻量分割器生成解剖导向掩码；②引入解剖一致性损失以提升语义判别性；③收集并公开170K张血管影像，形成首个大规模血管预训练数据集。

**🔧 技术方法**

技术主要包括掩码图像建模（MAE式）、解剖导向掩码策略、解剖一致性损失、ViT-B/16等Transformer骨干以及轻量级UNeXt-S分割器作为特征提取器。

**📊 数据集**

使用的主要数据集有：①预训练集XA-170K（来自CADICA、SYNTAX、XCAD、CoronaryDominance共170k张图像）；②下游评估集包括ARCADE（V、VS、S）、CAXF、XCAV、Stenosis检测等六个数据集。

**📈 对比分析**

与多种SSL基线（MAE、SimMIM、DINOv3、LocalMIM等）以及全监督基线进行比较，VasoMIM在四类任务（血管分割、血管段分割、斑块分割、斑块检测）上均实现了显著提升，典型提升幅度为1–2%的Dice/mAP；在少标注场景下，VasoMIM能用10%标注数据达到甚至超过全监督模型的表现。

**⚠️ 局限性**

局限性包括：①对Frangi滤波提取的血管质量敏感，低对比度血管可能被漏检；②掩码比例与解剖导向权重需要精细调参；③模型扩展至更大骨干网络时增益有限，且训练成本仍较高。

---

## 156. P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling

**arXiv ID:** 2602.12116 | [PDF](https://arxiv.org/pdf/2602.12116v1)

**作者:** Pinyi Zhang `[一作]` (Alibaba Group), Yongbin Li `[通讯]` (Alibaba Group)

**通讯引用:** 1811 | [OpenAlex ID](https://openalex.org/A5100644428)

**关键词:** `Computation and Language` `Recommendation System` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

提出并实现了 P‑GenRM——一种生成式奖励模型，能够把用户的混合偏好信号转化为结构化的评估链（包含个性化人物描述和打分规则），并在推理时通过个体与原型双粒度缩放聚合多份评分方案。

**💡 创新点**

创新点主要包括：① 生成式奖励模型能在无监督环境下自行构建个性化评估链；② 引入用户原型聚类与注意力增强的原型细化，结合测试时缩放机制，显著降低偏好推断噪声并提升对冷启动用户的泛化；③ 采用三阶段训练（SFT→RL→课程学习）并为 RL 设计过程级与结果级双重奖励。

**🔧 技术方法**

技术手段包括：生成式奖励模型（GenRM）架构、GRPO 强化学习、过程奖励与结果奖励的加权组合、K‑means 原型初始化、历史感知注意力原型细化、双粒度（个体/原型）缩放与评分聚合、结构化评价链（Persona‑Guided Scoring Induction）。

**📊 数据集**

使用的主要数据集有：PersonalRewardBench（Chatbot Arena‑personalized + Prism‑personalized），LaMP‑QA（长文本问答），Lamp‑QA（OOD 冷启动评估）以及公开的 LLaMA‑3.1‑8B/70B 预训练模型；评估时还对比了多种基准模型。

**📈 对比分析**

与多种基线（In‑Context LLM judge、Fine‑tuned Bradley‑Terry、GPO、VPL、PAL、SynthesizeMe、OpenAI‑o3 等）在 PersonalRewardBench 上对比，P‑GenRM 在 8B 规模上平均提升 2.7%~3%，在 70B 规模上提升 1.9%；在 Lamp‑QA 的 Spearman 相关系数达 0.638，超过同等规模基准；在策略模型的 DPO/GRPO 训练中亦能使 8B 模型跑超 70B 模型。

**⚠️ 局限性**

局限性：模型仍需一定量的历史交互和显式偏好作为输入，稀疏反馈下的表现仍不够理想；原型数与缩放参数需要手工调优，过多原型会导致推理噪声；对极端或完全新颖偏好的捕捉能力有限；推理时多次生成评分方案带来一定延迟。

---

## 157. The Distortion of Prior-Independent b-Matching Mechanisms

**arXiv ID:** 2602.11404 | [PDF](https://arxiv.org/pdf/2602.11404v1)

**作者:** Ioannis Caragiannis `[一作]` (Aarhus University), Sebastian Homrighausen `[通讯]` (Aarhus University)

**关键词:** `Computer Science and Game Theory`

**🎯 论文内容**

研究在仅能获得各代理人项目偏好序列的情况下，利用前置独立机制对多项分配问题进行社交福利最大化，并提出若干有效机制。

**💡 创新点**

1) 首先给出任何序数机制在期望情况下一致不可超过 e/(e‑1) 的下界；2) 提出 Random Survivors（RS）机制，在任意代理人特定分布下仍能实现该最优失真；3) 进一步改进为 RSBS 机制，几乎达到 1.0765 的失真间隙；4) 对一遍式（one‑pass）机制给出最优失真与失真间隙（HQL 机制），并给出秘书模型下的最小失真间隙 4/3；5) 证明所有机制在 Bayesian 框架下均满足 BIC，并且结果同样适用于子模（submodular）价值函数。

**🔧 技术方法**

使用随机化概率分析、积分技巧、指数极限与偏移（burning/stealing）策略、以及对 UF（unbiased‑favorites）分布的结构性利用，结合期望社交福利与最优社交福利的比例定义失真。

**📊 数据集**

无实验数据集；全部结果均为理论证明与期望下界分析。

**📈 对比分析**

与最优失真 e/(e‑1) 对比：RS 与 RSBS 在任意实例上均实现该失真；RSBS 的失真间隙仅为 1.0765，已接近 1；HQL 在固定顺序一遍式机制中达到 2‑b_max/m 的失真与 2‑2/e 的失真间隙，匹配已知下界；秘书模型 RS 失真保持 e/(e‑1)，失真间隙至少 4/3，表明更简单的随机顺序机制无法突破此限。

**⚠️ 局限性**

1) 失真间隙仍大于 1，尚未实现完全最优（gap=1）的简单一遍式机制；2) 对非 UF 分布的前置独立机制效果未知；3) 对复杂价值函数（非子模）或高维相关分布的鲁棒性待进一步研究。

---

## 158. Charting Empirical Laws for LLM Fine-Tuning in Scientific Multi-Discipline Learning

**arXiv ID:** 2602.11215 | [PDF](https://arxiv.org/pdf/2602.11215v1)

**作者:** Lintao Wang `[一作]` (University of Sydney), Xinzhu Ma `[通讯]` (Beihang University)

**通讯引用:** 1361 | [OpenAlex ID](https://openalex.org/A5030101527)

**关键词:** `Machine Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Mixture of Experts` `Text`

**🎯 论文内容**

系统研究了多学科 LLM 微调，构建了涵盖数学、化学、生物、医学、地理五个领域的语料库，并比较了全量微调、LoRA、LoRA‑MoE 与 LoRA‑Comp 四种策略的学习曲线，归纳出四条经验法则；

**💡 创新点**

首次提出并验证了四条多学科微调经验法则（Balance‑then‑Diversity、Merge‑then‑Align、Optimize‑then‑Scale、Share‑then‑Specialize），并给出了可操作的调优 recipe；

**🔧 技术方法**

使用 Qwen2.5 7B Instruct 作为基线模型，实施全量微调、LoRA、LoRA‑MoE（含共享 A 矩阵、rank‑wise 路由）以及 LoRA‑Comp 等 PEFT 方法；

**📊 数据集**

使用五个学科的公开数据集：数学（OpenMathInstruct+MATH+GSM8K）、化学（ChemData）、生物（Mol‑Instructions）、医学（MedMCQA+MedAlpaca+ChatDoctor+MedInstruct‑52K）、地理（GeoSignal），并在各自领域基准（GSM8K、ChemBench、Mol‑Instruction、MedMCQA、GeoBench）上评测；

**📈 对比分析**

采用 lm‑evaluation‑harness 计算各模型在域内的 Accuracy 进行对比；多学科微调整体表现低于单学科，但经过共享 A 矩阵等改进的 LoRA‑MoE 能达到与全量微调相近的精度，同时显著减少可训练参数量和 GPU 计算时间；

**⚠️ 局限性**

研究局限在于数据集主要来自单学科公开数据，跨学科数据多样性有限；仅评估了 7B 规模模型，缺乏更大规模或更广学科的扩展分析；微调过程对随机性和优化稳定性敏感，实验结果可能存在波动。

---

## 159. ExtremControl: Low-Latency Humanoid Teleoperation with Direct Extremity Control

**arXiv ID:** 2602.11321 | [PDF](https://arxiv.org/pdf/2602.11321v1)

**作者:** Ziyan Xiong `[一作]` (UMass Amherst), Chuang Gan `[通讯]` (MIT-IBM Watson AI Lab)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Reinforcement Learning` `Video`

**🎯 论文内容**

构建了低延迟全身控制框架 ExtremControl，实现人类操作者对仿真与真实 Humanoid 的高速遥操作。

**💡 创新点**

创新点包括：① 直接对选定刚体链的 SE(3) 目标姿态进行控制，避免全身重映射带来的延迟；② 设计了一套基于 Cartesian 空间的映射器，实现人类到机器人的即时姿态转换；③ 在低级 PD 控制器中加入速度前馈项，显著降低跟踪延迟；④ 统一理论与系统实现，实现端到端延迟低至 50 ms。

**🔧 技术方法**

使用了 Cartesian‑space 映射、全身阻抗校准、速度前馈 PD、强化学习三阶段训练（教师 → 学生 → 微调）结合 PPO+DAgger、仿真平台 Genesis、OptiTrack 运动捕捉与 VR 传感（Meta Quest + VIVE Trackers），以及光流法估算系统延迟。

**📊 数据集**

训练与评估数据集包括 LAFAN1、AMASS，以及额外收集的 MoCap 用户数据（Teleop）。

**📈 对比分析**

与现有遥操作系统对比（如 HOMIE、HumanPlus、CLONE 等），ExtremControl 在 VR/MoCap 模式下平均延迟为 54 ± 4 ms / 64 ± 8 ms，显著低于其它系统（>170 ms）。在多任务（乒乓球、飞盘、抓取、举箱等）中，定位误差≤几厘米，表现出高度流畅与实时响应。

**⚠️ 局限性**

存在的局限性包括：① 单一 Arm DoF 组合导致逆运动学歧义，手臂姿态可能不自然；② 低体延迟仍受策略层对重心控制的影响；③ 使用非关节手套，缺乏多指抓取能力；④ 仅在 Unitree G1 上验证，未扩展到更复杂手部或其他机器人平台。

---

## 160. ANML: Attribution-Native Machine Learning with Guaranteed Robustness

**arXiv ID:** 2602.11690 | [PDF](https://arxiv.org/pdf/2602.11690v1)

**作者:** Oliver Zahn `[一作]` (Independent Researcher), Simran Chana `[通讯]` (University of Cambridge)

**通讯引用:** 88 | [OpenAlex ID](https://openalex.org/A5012655677)

**关键词:** `Machine Learning` `Federated Learning` `Data-Centric Learning` `Tabular`

**🎯 论文内容**

提出了 ANML 框架，在机器学习训练中根据梯度一致性、验证状态、贡献者声誉和时间相关性等多维质量因素对样本进行加权，同时实现贡献者级归因。

**💡 创新点**

创新点在于：① 将外部可验证的贡献者信息直接融入训练权重；② 设计了两种鲁棒组合方法（两阶段自适应门控和 Softmax Blend）保证性能不劣于基线；③ 通过贡献者级归因提升对细粒度质量检测的鲁棒性，并对时间衰减做了实验验证。

**🔧 技术方法**

技术方法包括：Krum 风格的梯度一致性评分、Softmax 归一化与加权、两阶段自适应门控、加权选择/采样、数据验证与声誉评分、时间衰减函数、实验中使用的梯度对齐与凭证伪造攻击模型。

**📊 数据集**

使用了 5 个 UCI 数据集（Wine、Breast Cancer、Digits、Covertype、Adult Census）以及 federated/贡献者级实验，样本量从 178 到 32,561。

**📈 对比分析**

与统一加权、Krum（仅梯度）、Softmax Blend、两阶段 Adaptive 等基线比较。ANML 在 30% 噪声数据下相对 Krum 提升 33–75% 错误率，整体错误率可低至 5.5%（20% 高质量数据优于 100% 通用数据 47%）。在攻击场景中，ANML 的误差增长幅度约为 Krum 的一半；在细粒度可检测性下降时，贡献者级归因优势可达 5.3×。

**⚠️ 局限性**

局限性包括：实验规模仅至 32K 样本，未验证大规模图像/语言模型；验证与声誉信号为模拟，需实际部署；梯度一致性计算 O(n²) 复杂度；在无攻击场景下加权选择可能略逊于统一训练；对后门、分布漂移等攻击的鲁棒性尚待进一步研究。

---

## 161. PLESS: Pseudo-Label Enhancement with Spreading Scribbles for Weakly Supervised Segmentation

**arXiv ID:** 2602.11628 | [PDF](https://arxiv.org/pdf/2602.11628v1)

**作者:** Yeva Gabrielyan `[一作]` (American University of Armenia), Irina Voiculescu `[通讯]` (University of Oxford)

**通讯引用:** 1190 | [OpenAlex ID](https://openalex.org/A5055611196)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Knowledge Distillation` `Biomedical Data` `Image` `Magnetic Resonance Imaging`

**🎯 论文内容**

提出了一种基于伪标签增强的弱监督分割框架 PLESS，利用分层区域传播改进粗糙的伪标签。

**💡 创新点**

创新点在于将伪标签与层级水流式笔划传播相结合，并加入背景扩展，显著提升伪标签的空间一致性和可靠性。

**🔧 技术方法**

采用层级分水岭+水流式扩散、背景扩展、伪标签融合（学生-教师）以及 Dice 伪标签损失。

**📊 数据集**

在两个心脏 MRI 数据集 ACDC 与 MSCMRseg 上进行实验，均为稀疏笔划标注。

**📈 对比分析**

与四种现有笔划监督算法（DMPLS、DCDPL、ScribbleVC、ScribbleVS）以及若干 SOTA 方法对比，PLESS 在 DSC、HD95、ASD 指标上持续提升，ScribbleVS+PLESS 达到最高平均 DSC。

**⚠️ 局限性**

局限性包括对参数（如扩散阈值、传播比例）敏感，且在全覆盖传播时可能引入噪声，未来需进一步自适应调整。

---

## 162. Unifying Stable Optimization and Reference Regularization in RLHF

**arXiv ID:** 2602.11523 | [PDF](https://arxiv.org/pdf/2602.11523v1)

**作者:** Li He `[一作]` (CSIRO), Tongliang Liu `[通讯]` (Sydney AI Centre)

**关键词:** `Machine Learning` `Reinforcement Learning from Human Feedback` `Optimization` `Reinforcement Learning` `Supervised Fine-Tuning`

**🎯 论文内容**

提出并实现了一种统一的Dual-KL正则化目标与DAR（Dual-regularized Advantage Regression）算法，用于解决RLHF中的奖励劫持与稳定优化冲突。

**💡 创新点**

创新点在于把参考正则化与稳定优化通过双重KL约束合并成一个可调节的目标，并将其转换为可直接使用的加权SFT损失，显著简化实现且提升学习稳定性与性能。

**🔧 技术方法**

采用PPO、AWR、REINFORCE、GRPO等强化学习框架，基于KL正则化与优势回归的理论推导，构建DAR算法并实现蒙特卡洛优势估计与梯度裁剪。

**📊 数据集**

实验使用Anthropic Helpfulness、TL;DR、Harmlessness、Helpsteer2、MT‑Bench、AlpacaEval 2.0等多种数据集，并以Qwen2‑7B/Qwen2‑5B为基础模型，利用Qwen2‑72B‑Instruct/Qwen3‑32B等大型模型作为评判器。

**📈 对比分析**

与PPO、RLOO、GRPO、DPO、SimPO、SLiC等现有方法对比，DAR在参考胜率、奖励、KL‑Pareto前沿和MT‑Bench/AlpacaEval等指标上均获得显著提升（平均提高约7–10%），并展示了更稳定的训练曲线。

**⚠️ 局限性**

限制在于DAR需要在线采样与当前策略的分布信息，无法直接应用于离线RLHF场景；若要迁移到纯离线设置，需进一步研究离线校正方法。

---

## 163. The PBSAI Governance Ecosystem: A Multi-Agent AI Reference Architecture for Securing Enterprise AI Estates

**arXiv ID:** 2602.11301 | [PDF](https://arxiv.org/pdf/2602.11301v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Artificial Intelligence`

---

## 164. UniT: Unified Multimodal Chain-of-Thought Test-time Scaling

**arXiv ID:** 2602.12279 | [PDF](https://arxiv.org/pdf/2602.12279v1)

**作者:** Leon Liangyu Chen `[一作]` (Stanford University), Felix Juefei-Xu `[通讯]` (Meta Superintelligence Labs)

**关键词:** `Computer Vision and Pattern Recognition` `Data Synthesis` `Generation` `Transformer` `Agentic AI` `Chain-of-Thought` `Large Language Model` `Multimodality` `Text`

**🎯 论文内容**

提出 UniT 框架，将多模态链式推理（multimodal chain‑of‑thought）用于推理时的计算放大，实现统一模型的迭代生成、反思与细化；

**💡 创新点**

创新点在于：①用 agentic 数据合成管线自动生成含链式推理的多轮训练轨迹，①培养验证、子目标拆分、内容记忆等认知行为；②通过预算强制（budget forcing）实现推理时间超出训练时的推理长度；③演示顺序链式推理对比并行 best‑of‑N 的优势；

**🔧 技术方法**

技术包括：统一 Bagel 多模态模型训练、嵌套式 CFG 引导、链式推理循环、Agentic 训练数据生成、预算强制控制推理轮数、以及多轮文本与图像交互推理；

**📊 数据集**

使用 12K 由 Llama‑4‑Scout‑17B‑16E 生成的多模态提示，Flux Pro/Flux Kontext/ Qwen3‑VL 交互合成的训练轨迹；评测数据集包含 OneIG‑Bench、CompBench、ImgEdit、MIRA；

**📈 对比分析**

与基线 Bagel、Bagel+CoT、并行 best‑of‑N 并行采样进行比较。顺序链式推理在 10 轮预算下，OneIG‑Bench 提升 10.34%，CompBench 5.56%，ImgEdit 225%（单轮至 4 轮），MIRA 53%（单轮至 10 轮）。顺序推理相较于并行采样，计算量少 2.5×，且性能更稳健；

**⚠️ 局限性**

局限性包括：对物理约束和细粒度空间关系的纠正能力有限；验证阶段偶尔误报导致无意义修改；多目标约束冲突时子目标拆分失效；基础模型能力瓶颈无法通过额外推理完全弥补；推理预算增至 10 轮后性能增长趋缓；

---

## 165. Protein Circuit Tracing via Cross-layer Transcoders

**arXiv ID:** 2602.12026 | [PDF](https://arxiv.org/pdf/2602.12026v1)

**作者:** Darin Tsui `[一作]` (Georgia Institute of Technology), Amirali Aghazadeh `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 848 | [OpenAlex ID](https://openalex.org/A5062650028)

**关键词:** `Machine Learning` `Compression` `Explainability and Interpretability` `Representation Learning` `Transformer` `Auto Encoder` `Biomedical Data`

**🎯 论文内容**

本文提出 ProtoMech 框架，通过跨层转码器（CLT）对蛋白质语言模型（pLM）进行替代建模，追踪并提取模型内部的计算电路。

**💡 创新点**

创新点在于将跨层信息纳入转码器，能够同时考虑多层稀疏潜在变量，实现对完整模型计算路径的重构，显著提升电路压缩和可解释性。

**🔧 技术方法**

核心技术包括稀疏自动编码器、跨层转码器（CLT）、梯度归因搜索、激活截断与可视化工具，以及对最终输出的微调和基准对照。

**📊 数据集**

使用 ESM2-8M pLM 作为基准模型，在 UniRef50 训练 500 万条蛋白质序列；下游任务采用 30% 序列相似度聚类的 Swiss‑Prot 家族分类和 12 项 ProteinGym 深度突变测序（DMS）功能预测。

**📈 对比分析**

与传统的单层稀疏转码器（PLT）以及 Contrastive Activation Addition（CAA）等方法对比，ProtoMech 在家族分类任务中恢复 89% 原模型性能（F1 ≈0.82），在功能预测任务中恢复 82%（Spearman ≈0.41），且压缩至 <1% 潜在空间；在序列设计上，ProtoMech 通过电路驱动的激活截断能在 71% 的 DMS 实验中产生最高适配性变体。

**⚠️ 局限性**

局限性包括：CLT 参数量较大（相较原模型多 3.5 倍）、训练成本高；电路解释仍需人工注释，缺乏自动化；跨层转码器在更大规模 pLM 上的可扩展性待进一步研究。

---

## 166. RAM-Net: Expressive Linear Attention with Selectively Addressable Memory

**arXiv ID:** 2602.11958 | [PDF](https://arxiv.org/pdf/2602.11958v1)

**作者:** Kaicheng Xiao `[一作]` (Chinese University of Hong Kong), Guoliang Xing `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 10517 | [OpenAlex ID](https://openalex.org/A5072780737)

**关键词:** `Machine Learning` `Retrieval` `Transformer` `Text`

**🎯 论文内容**

提出了 RAM-Net，一种通过可微分地址解码器将密集向量映射到稀疏高维地址，从而实现类似随机存取内存的高效长程检索模型。

**💡 创新点**

创新点包括：
• 可微分地址解码器（Product Softmax + Top‑K）实现高维稀疏寻址；
• 循环位置编码（CAPE）将绝对位置信息转换为相对位置信息；
• Power Decay Moving Average (PDMA) 通过可调衰减率实现更灵活的记忆更新；
• 通过解耦记忆容量与模型维度，使状态规模指数级扩展而不增加参数。

**🔧 技术方法**

技术细节：
• Product Softmax 与 Top‑K 的组合产生稀疏地址；
• CAPE 通过循环移位实现相对定位；
• PDMA 的指数衰减与动态归一化；
• log‑域 Beam Search 实现高维地址的高效 Top‑K 选取；
• 动态标量重参数化与代理梯度技巧提升训练稳定性；
• 针对训练与推理的专用稀疏矩阵核。

**📊 数据集**

数据集与基准：
• 合成检索任务 MQAR（Zoology 框架）
• 语言模型训练：FineWeb‑Edu（10B 令牌）
• 零样本评测：WikiText‑103、MMLU、ARC‑Challenge/Easy、OpenbookQA、SciQ、COPA、PIQA、HellaSwag、WinoGrande。

**📈 对比分析**

比较方法与性能：
• 在 MQAR 上与 Full‑Attention、Linear Attention、Sliding‑Window、GLA、DeltaNet、Mamba2、H3 等做对比，RAM‑Net 在相同或更小状态规模下取得更高检索准确率；
• 在语言模型任务中，RAM‑Net 的 perplexity 与其他 SOTA 近似，且每个 token 的活跃状态数显著低于全注意力模型；
• 零样本推理表现与主流模型持平或略优，尤其在需要长程依赖的任务上优势明显。

**⚠️ 局限性**

局限性：
• 记忆容量仍受可配置槽数限制，极大容量需要更高的 K 与 U，导致训练成本上升；
• Top‑K 选取可能限制表示的多样性；
• 需要额外的 Beam Search 与自定义核，增加实现复杂度；
• 在极长序列或极大模型规模下，稀疏地址解码的计算瓶颈仍需进一步优化；
• 对超参数（U、K、γ）敏感，需经验调优。

---

## 167. History-Independent Load Balancing

**arXiv ID:** 2602.11953 | [PDF](https://arxiv.org/pdf/2602.11953v1)

**作者:** Michael A. Bender `[一作]` (Stony Brook University), Rose Silver `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5056682774)

**关键词:** `Data Structures and Algorithms` `Optimization`

**🎯 论文内容**

提出了一种强历史独立的双选球‑箱分配算法，支持动态插入与删除，保证最大负载为 m/n+O(1)，且期望搬迁量（recourse）为 O(log log(m/n))。

**💡 创新点**

在历史独立框架下首次实现非平凡的负载均衡与低搬迁量；通过 Slice‑and‑Spread 机制和黑盒转换、扩展的 Canonical Orientation (ECO) 技术，突破了以往历史相关算法的限制。

**🔧 技术方法**

采用历史独立数据结构理论、两选负载平衡分析、切片‑传播（Slice‑and‑Spread）技术、组件定向的 ECO 算法、Poisson 化与大偏差（McDiarmid/McKay）分析等多种理论工具。

**📊 数据集**

本工作为理论研究，未使用任何实验数据集，所有结论均为概率上界与期望值的数学证明。

**📈 对比分析**

与先前的历史相关两选算法相比，期望搬迁量从原来的 O(μ) 降低到 O(log log(m/n))，同时保持 O(1) 的负载上限；对历史独立算法的前沿结果进行了理论比较，证明其性能已达或逼近最优。

**⚠️ 局限性**

仍未证明在 O(1) 负载下能否进一步降低搬迁量；对历史独立算法的最优下界尚不清楚；算法实现的时间复杂度与空间效率未给出实际评估；以及对 m/n=ω(1) 的更紧收敛或更小空间实现仍是开放问题。

---

## 168. Towards Compressive and Scalable Recurrent Memory

**arXiv ID:** 2602.11212 | [PDF](https://arxiv.org/pdf/2602.11212v1)

**作者:** Yunchong Song `[一作]` (Shanghai Jiao Tong University), Zhouhan Lin `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `Machine Learning` `Compression` `Optimization` `Transformer` `Text`

**🎯 论文内容**

提出 Elastic Memory，利用 HiPPO 理论在 Transformer 中实现可扩展的在线函数逼近记忆，并通过多项式采样进行历史检索。

**💡 创新点**

创新点在于将 HiPPO 的在线压缩与可并行的块级更新相结合，形成固定尺寸的记忆状态；检索机制通过多项式采样灵活控制对历史的关注，实现训练后可在推理时注入不同偏置。

**🔧 技术方法**

核心技术包括 HiPPO-LegS 的离散递推（Block HiPPO Kernel）、预计算的状态转移矩阵、Reconstruction Bank 的多项式采样、以及与传统注意力的三角形掩码融合。

**📊 数据集**

使用三大长文本数据集：PG‑19、Proof‑Pile 和 FineWeb‑Edu，每个数据集约 2 亿 token，长度均大于 32k，块大小 2048。

**📈 对比分析**

与 Memorizing Transformer、Infini‑Transformer、Melodi 等现有记忆模型在 PPL、LongPPL 以及吞吐量上进行对比；Elastic Memory 在不增加可训练参数的情况下，在所有数据集上均取得更低 PPL（尤其是指数采样），在 LongPPL 上（均匀采样）也保持竞争力，并在模型和记忆规模扩大时保持优势，速度上仅略低于最快的 baseline。

**⚠️ 局限性**

局限性包括：① 仍需预计算并存储大量矩阵，内存占用随上下文长度上升；② 在极端长上下文（>128k）时多项式采样的精度和覆盖范围仍有限；③ 对超大规模参数模型的可扩展性尚未充分验证。

---

## 169. Designing and Comparing RPQ Semantics

**arXiv ID:** 2602.11949 | [PDF](https://arxiv.org/pdf/2602.11949v1)

**作者:** Victor Marsault `[一作]` (University Gustave Eiffel), Antoine Meyer `[通讯]` (University Gustave Eiffel)

**关键词:** `Databases`

**🎯 论文内容**

研究 RPQ（正则路径查询）语义的形式化框架，提出过滤基和顺序基两类语义，并对其进行属性定义与比较。

**💡 创新点**

创新点包括：①为 RPQ 语义设计统一的属性体系（如单调性、连续性、可分解性等）并证明多种不可兼容性定理；②提出多种新的语义（如子路最小、最短顶点覆盖等）并讨论其可行性；③通过 Well‑partial‑order 与 Higman 引理等理论工具给出最小化与覆盖性质的判定。

**🔧 技术方法**

使用的主要技术手段有：形式化定义、集合与序理论（well‑partial‑orders、Higman lemma）、逻辑与正则表达式理论、复杂度分析（P、NP、coNP 等），以及构造性证明与反例构造。

**📊 数据集**

未使用具体实验数据集；所有结果均基于理论模型（有向多重标记图与正则表达式）与抽象数据库。

**📈 对比分析**

通过属性相容性与不可兼容性定理以及复杂度表，对各种语义的行为进行比较；实验性能未涉及，仅给出理论复杂度（如存在性判定在 P 或 NP‑complete）。

**⚠️ 局限性**

局限性包括：①仅考虑走路基（walk‑based）语义，忽略了语法依赖或数据库全局信息的语义；②未涉及节点/边多重标签、属性值等实际数据库特性；③缺乏实验验证与实现评估；④对新提出语义的实现与优化未给出具体方案。

---

## 170. Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation

**arXiv ID:** 2602.11799 | [PDF](https://arxiv.org/pdf/2602.11799v1)

**作者:** Pingjun Pan `[一作]` (NetEase Cloud Music), Chuanjiang Luo `[通讯]` (NetEase Cloud Music)

**关键词:** `Artificial Intelligence` `Recommendation System` `Transformer` `Multimodality`

**🎯 论文内容**

提出了Hi-SAM框架，利用分层结构感知的多模态语义ID实现推荐。

**💡 创新点**

创新点在于设计去耦合的语义标记器和分层记忆锚点Transformer，解决多模态Token化与架构不匹配的问题。

**🔧 技术方法**

采用Geometry-aware Cross-Modal Alignment、Disentangled Modal-Residual Quantization、Hierarchical RoPE、Memory-Anchor Attention、预训练+微调以及KV Cache压缩等技术。

**📊 数据集**

使用Amazon Movies & TV、Books公开数据集以及内部大型社交/约会平台工业数据。

**📈 对比分析**

通过与WuKong、HSTU、MTGR、QARM、PSRQ+MCCA等基线对比，离线AUC/GAUC、冷启动AUC提升6.55%，在线响应率提升显著。

**⚠️ 局限性**

限制在于仍依赖大规模预训练模型与计算资源，对极端稀疏交互和少量多模态数据的泛化鲁棒性尚未完全解决。

---

## 171. When Audio-LLMs Don't Listen: A Cross-Linguistic Study of Modality Arbitration

**arXiv ID:** 2602.11488 | [PDF](https://arxiv.org/pdf/2602.11488v1)

**作者:** Jayadev Billa `[一作]` (Unaffiliated researcher), Jayadev Billa `[通讯]` (Unaffiliated researcher)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Audio` `Multimodality` `Text` `Benchmark`

**🎯 论文内容**

提出ALME基准并测量多模态LLM在音频-文本冲突时的文本主导行为

**💡 创新点**

发现音频-文本仲裁比文本-文本仲裁困难10倍，并区分信息内容与仲裁可达性两维；通过细调实验定位文本主导主要在LLM推理层而非音频编码层

**🔧 技术方法**

使用文本优先/文本受限提示、音频投影层训练与LoRA微调、对抗性提示框架、TTS再合成、语音识别+LLM级联

**📊 数据集**

利用Mozilla Common Voice 22.0生成57,602条跨八种语言（英德法意葡阿日中）并在每语言中包含数字、否定、形容词、时间等单一语义翻转

**📈 对比分析**

通过对齐、冲突、文本仅、音频仅四种评估条件计算文本主导比率（TDR），在四种模型间得到16.6%–63.2%的差异，跨语言差异显著；文本-文本仲裁TDR仅1.6%，证明十倍仲裁差距；TTS与提示干预可显著改变TDR

**⚠️ 局限性**

限制包括仅测量强制二选一的冲突场景、仅用Common Voice数据、提示对模型影响敏感、未对更高温度或开放式回答的仲裁机制做评估、Fine‑tune仅在Ultravox上验证、跨模型通用性待进一步检验

---

## 172. Clutt3R-Seg: Sparse-view 3D Instance Segmentation for Language-grounded Grasping in Cluttered Scenes

**arXiv ID:** 2602.11660 | [PDF](https://arxiv.org/pdf/2602.11660v1)

**作者:** Jeongho Noh `[一作]` (Seoul National University), Ayoung Kim `[通讯]` (Seoul National University)

**通讯引用:** 5636 | [OpenAlex ID](https://openalex.org/A5100740100)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Object Detection` `Robotic Intelligence` `Point Cloud`

**🎯 论文内容**

提出了一种名为Clutt3R‑Seg的零射手管线，用于在稀疏视角下的堆砌场景中实现鲁棒的三维实例分割和语言驱动的多阶段抓取。

**💡 创新点**

通过层级实例树进行跨视角分割聚类并进行残差节点替代，解决了噪声掩模的过/欠分割问题，并实现了仅用单张后置图像的自适应场景更新。

**🔧 技术方法**

使用Grounded SAM进行初始掩模生成，MVSAnywhere估计深度，Duoduo CLIP进行跨视角语义嵌入，基于重心投票的超体素分割，以及Chamfer、光度和正则化损失的三阶段几何对齐。

**📊 数据集**

在GraspClutter6D真实数据集和自制的NVIDIA Isaac Sim HouseCat6D合成数据集上进行评估，并在Franka Research 3机器人上实现实机抓取。

**📈 对比分析**

与SAI3D、GraphSeg和MaskClustering等基准对比，在AP@25、AP@50及平均AP上均取得2.2倍以上提升，最高AP@25达61.66；在语言驱动语义分割上超过50%提升，运行时最短。

**⚠️ 局限性**

受限于预测深度的误差导致在更严格阈值下重建质量下降，以及极端遮挡下无法完全恢复隐藏物体，且单视角更新可能无法捕捉多模态信息。

---

## 173. ModelWisdom: An Integrated Toolkit for TLA+ Model Visualization, Digest and Repair

**arXiv ID:** 2602.12058 | [PDF](https://arxiv.org/pdf/2602.12058v1)

**作者:** Zhiyong Chen `[一作]` (Nanjing University), Shing-Chi Cheung `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 8623 | [OpenAlex ID](https://openalex.org/A5034057959)

**关键词:** `Software Engineering` `Optimization` `Explainability and Interpretability` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

提供了一套基于大型语言模型的交互式工具，集成了 TLA+ 模型的可视化、摘要（Digest）和修复（Repair）功能，帮助用户快速定位并修正模型中的错误。

**💡 创新点**

创新点包括：1) 利用 LLM 生成自然语言模型摘要和修复建议；2) 在可视化图中对违规状态进行颜色高亮、可点击跳转至源码，并与违反的属性直接映射；3) 通过树结构、节点/边折叠和图压缩实现大规模状态图的可扩展交互；4) 提供单步与多步迭代修复工作流，记录修复历史。

**🔧 技术方法**

技术实现：TypeScript + Python 前后端；TLA+ Checker（TLC）执行验证；使用 GPT‑4、Claude 3.7 Sonnet 等 LLM；图形渲染采用 lazy rendering、树形布局、折叠算法；LLM 接口集成用于自然语言解释和代码生成。

**📊 数据集**

主要使用 TLA+ 官方示例（如 CoffeeCan）和论文中展示的自定义模型；未给出公开数据集，仅在实验演示中使用这些示例进行验证。

**📈 对比分析**

与现有 TLA+ Toolbox 的状态图查看器做对比，指出后者缺乏折叠、颜色高亮和语义解释，导致可视化不够可扩展。作者在文中通过可视化效果示例展示性能提升，但未给出量化指标；认为通过图优化和懒加载能够显著降低渲染时间和内存占用。

**⚠️ 局限性**

局限性：1) 目前仅支持 TLA+，需进一步泛化到 Promela、Alloy 等；2) 依赖 LLM 的解释和修复可能产生幻觉或不精确的结果；3) 未进行系统性量化评估（例如渲染速度、用户研究），仅基于示例展示；4) 大型模型的可视化仍可能受到浏览器资源限制。

---

## 174. From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders

**arXiv ID:** 2602.11881 | [PDF](https://arxiv.org/pdf/2602.11881v1)

**作者:** Yifan Luo `[一作]` (Peking University), Bin Dong `[通讯]` (Peking University)

**通讯引用:** 16333 | [OpenAlex ID](https://openalex.org/A5100746745)

**关键词:** `Artificial Intelligence` `Representation Learning` `Explainability and Interpretability` `Auto Encoder` `Tabular`

**🎯 论文内容**

本文提出了一种层次稀疏自编码器（HSAE），能够在训练时同步学习多层稀疏特征及其父子关系，形成可解释的概念森林。

**💡 创新点**

创新点在于将结构约束损失和随机特征扰动嵌入至稀疏自编码器训练中，实现了在不损失重构精度的前提下获得语义层级化特征。

**🔧 技术方法**

采用稀疏自编码器（JumpReLU SAE）、父子约束损失、随机扰动机制、基于相似度的层次更新以及交替优化策略。

**📊 数据集**

使用从多种 LLM（如 GPT‑/PPT‑系列）的第 13 层残差流提取的 100M 激活样本，以及 mini‑PILE 数据集。

**📈 对比分析**

与独立训练的 SAE 后期对齐基线相比，HSAE 在父子一致性、共激活概率、LLM 自动解释等指标上均显著提升，同时在重构、Absorption、AutoInterp、SCR 等标准 SAE 评测上保持或略优表现。

**⚠️ 局限性**

局限在于树结构层数固定、部分父子对齐仍不理想、未能完全捕获不同深度抽象层次，且需进一步探索更灵活的层次先验和尺度扩展。

---

## 175. FlowMind: Execute-Summarize for Structured Workflow Generation from LLM Reasoning

**arXiv ID:** 2602.11782 | [PDF](https://arxiv.org/pdf/2602.11782v1)

**作者:** Yihao Liu `[一作]` (Peking University), Huaqian Cai `[通讯]` (Peking University)

**通讯引用:** 192 | [OpenAlex ID](https://openalex.org/A5055332880)

**关键词:** `Artificial Intelligence` `Large Language Model` `Text` `Benchmark`

**🎯 论文内容**

提出了一种 Execute–Summarize 框架，将任务执行与工作流构建分为两阶段，先完成业务工具调用，再利用执行轨迹生成结构化工作流。

**💡 创新点**

通过将执行与摘要解耦，消除认知负荷和工具级混用，并引入 FlowBench 基准，对工作流的可靠性与可重用性提供系统评估。

**🔧 技术方法**

使用大语言模型（如 Qwen3、GPT‑4/5）结合 ReAct、Plan‑and‑Execute 及 ES‑variants，执行阶段调用业务工具，摘要阶段调用图构造工具，实验采用 FlowBench 进行黑盒评估。

**📊 数据集**

FlowBench——合成的工具集、自然语言任务描述及已验证的执行轨迹，覆盖多种工具使用场景。

**📈 对比分析**

与 ReAct 与 Plan‑and‑Execute 进行对比，评估指标包括 CRate、TRate、执行完整率、图有效率与联合成功率；实验显示 ES‑P&E 在所有模型规模下均显著提升工作流准确率和联合成功率。

**⚠️ 局限性**

由于工作流压缩，长序列或细粒度依赖可能被忽略；在真实部署前仍需人工验证与监督。

---

## 176. Advancing AI Trustworthiness Through Patient Simulation: Risk Assessment of Conversational Agents for Antidepressant Selection

**arXiv ID:** 2602.11391 | [PDF](https://arxiv.org/pdf/2602.11391v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computation and Language`

---

## 177. Towards Reliable Machine Translation: Scaling LLMs for Critical Error Detection and Safety

**arXiv ID:** 2602.11444 | [PDF](https://arxiv.org/pdf/2602.11444v1)

**作者:** Muskaan Chopra `[一作]` (Rheinische Friedrich-Wilhelms-Universität Bonn), Rafet Sifa `[通讯]` (Lamarr Institute for Machine Learning and Artificial Intelligence)

**关键词:** `Computation and Language` `Safty and Privacy` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Text`

**🎯 论文内容**

研究了指令调优的大规模语言模型在机器翻译中检测关键意义错误（CED）的能力，并在不同规模和适配方式下评估性能。

**💡 创新点**

首次对跨模型规模进行系统比较，证明了指令对齐和中等规模的LLM在不需要大模型也能达到高可靠性的事实，并提出了轻量级投票委员会提升稳健性的方案。

**🔧 技术方法**

采用指令调优的LLM（如GPT‑4o、GPT‑4o‑mini、LLaMA‑3.1‑8B、LLaMA‑3.3‑70B、GPT‑OSS‑20B/120B），零/少样本提示、提示调优、LoRA 微调，以及多模型投票。

**📊 数据集**

使用了 WMT 2021/2022 EN→DE 平行语料、SynCED‑EnDe 2025 人工标注的关键错误数据集。

**📈 对比分析**

通过 Matthews 相关系数（MCC）和 F1 评估，与基线 XLM‑R 等编码器模型对比，零/少样本提示可达 MCC≈0.6–0.9，微调后模型在所有数据集上达到 MCC≥0.9，尤其在 WMT‑22 和 SynCED 上超过 0.95。

**⚠️ 局限性**

局限包括对低资源语言和长文档的泛化不足，模型对提示设计敏感，仍需解决误报/漏报、计算成本与部署可行性。

---

## 178. Mask What Matters: Mitigating Object Hallucinations in Multimodal Large Language Models with Object-Aligned Visual Contrastive Decoding

**arXiv ID:** 2602.11737 | [PDF](https://arxiv.org/pdf/2602.11737v1)

**作者:** Boqi Chen `[一作]` (ETH Zurich), Jianing Qiu `[通讯]` (MBZUAI)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Recognition` `Transformer` `Large Language Model` `Contrastive Learning` `Vision Language Model` `Multimodality`

**🎯 论文内容**

提出一种基于自监督ViT注意力的对象对齐辅助视图，在视觉对比解码中抑制多模态大语言模型的对象错报

**💡 创新点**

通过掩蔽DINO注意力突出区域生成对比视图，避免提示依赖与跨模态循环问题，得到更强的对比信号

**🔧 技术方法**

利用自监督ViT（DINO）注意力、视觉对比解码（VCD）和自适应可信度约束（APC），实现单前向传递的辅助视图生成

**📊 数据集**

在POPE和MME两大对象错报基准上评估，使用LLaVA‑v1.5（7B）和Qwen‑VL（7B）两种模型

**📈 对比分析**

与常规解码、噪声VCD和AGLA对比，均表现出更高的准确率和F1，尤其在随机子集和存在/颜色类别上提升显著

**⚠️ 局限性**

对自监督ViT注意力的依赖导致在杂乱场景或提示相关区域被误掩时效果下降，遮罩比例与背景填充选择对性能有一定影响

---

## 179. Benchmark Health Index: A Systematic Framework for Benchmarking the Benchmarks of LLMs

**arXiv ID:** 2602.11674 | [PDF](https://arxiv.org/pdf/2602.11674v1)

**作者:** Longyuan Zhu `[一作]`, Bing Zhao `[通讯]`

**关键词:** `Artificial Intelligence` `Large Language Model` `Text` `Benchmark`

**🎯 论文内容**

构建 Benchmark Health Index（BHI）对 106 个大模型基准进行系统性健康审计，评估其判别力、抗饱和度和行业影响力。

**💡 创新点**

提出三维健康度量（Capability Discrimination、Anti‑Saturation、Impact）及 CRITIC 目标加权融合，形成可量化、可追踪的基准生命周期评估框架。

**🔧 技术方法**

利用统计学习方法（分层归一化、离群检测、Leave‑One‑Benchmark‑Out 校准）、指标组合（EDR、RCV、S_Sta、S_Dyn、行业采纳与社区热度），并采用 Bootstrap 稳健性分析与噪声鲁棒性实验。

**📊 数据集**

基于 91 个 2025 年主流 LLM 技术报告中的 289 个候选基准，最终筛选出 106 个验证基准，涵盖多领域（推理、数学、代码、主体等）。

**📈 对比分析**

与等权重基准及噪声鲁棒性实验对比，BHI‑CRITIC 在排名稳定性（Spearman ρ≥0.96）、噪声耐受度（最高 0.9682）及整体评估分数（最高 0.669）上均优于基线。

**⚠️ 局限性**

局限性包括公开报告缺乏统一细节导致残差、归一化可能压缩极端值、未对同一基准版本跨版本插值导致稀疏等问题。

---

## 180. Gradient Compression May Hurt Generalization: A Remedy by Synthetic Data Guided Sharpness Aware Minimization

**arXiv ID:** 2602.11584 | [PDF](https://arxiv.org/pdf/2602.11584v1)

**作者:** Yujie Gu `[一作]` (Zhejiang University), Huaiyu Dai `[通讯]` (North Carolina State University)

**通讯引用:** 9630 | [OpenAlex ID](https://openalex.org/A5027155270)

**关键词:** `Machine Learning` `Federated Learning` `Optimization` `Image`

**🎯 论文内容**

提出 FedSynSAM 方法，在联邦学习中结合梯度压缩和 SAM 以降低通信开销并提升泛化性能。

**💡 创新点**

创新点在于利用全局模型轨迹构造合成数据，精确估计全局扰动，从而克服数据异质性导致的误估问题；并给出收敛理论。

**🔧 技术方法**

采用梯度压缩（随机量化、Top‑k 稀疏化）、Sharpness‑Aware Minimization（SAM）、合成数据生成（轨迹匹配）以及标准的联邦平均（FedAvg）框架。

**📊 数据集**

在 Fashion‑MNIST、CIFAR‑10、CINIC‑10 等图像数据集上进行实验。

**📈 对比分析**

与 FedAvg、DynaFed、FedSAM、FedLESAM、FedSMOO、FedGAMMA 等基线比较，FedSynSAM 在不同压缩率、非 IID 设置和计算预算下均取得最高或相近的测试准确率，尤其在 4‑bit 量化或 0.1/0.25 稀疏率时表现最优。

**⚠️ 局限性**

主要限制包括对合成数据大小和初始轮数的依赖、在极度压缩或高度异质场景下可能仍受压缩误差影响，以及对计算资源的额外需求（生成合成数据）

---

## 181. Same Feedback, Different Source: How AI vs. Human Feedback Shapes Learner Engagement

**arXiv ID:** 2602.11311 | [PDF](https://arxiv.org/pdf/2602.11311v1)

**作者:** Caitlin Morris `[一作]` (MIT Media Lab), Pattie Maes `[通讯]` (MIT Media Lab)

**关键词:** `Human-Computer Interaction` `Large Language Model` `Text`

**🎯 论文内容**

研究了在创意编码环境中，给定相同LLM生成的反馈内容，但将其归因为AI或人类教学助理（TA）时，对学习者的参与度、努力投入及反馈评价产生的影响。

**💡 创新点**

创新点在于通过同一语言模型生成完全相同的反馈内容，仅改变归因标签和交付时机，从而严格控制内容差异，首次揭示归因对学习者行为和评价的独立作用。

**🔧 技术方法**

使用Claude Sonnet 4 LLM生成反馈，搭建网页式创意编码教学平台，采用2×2混合实验设计、行为指标（时间、代码长度、执行频率等）和问卷评估技术。

**📊 数据集**

未使用公开数据集，而是收集了25名参与者在实验过程中的实时编码、交互与反馈数据。

**📈 对比分析**

通过行为指标与反馈评分对比，发现TA归因组在时间投入、代码长度、执行频率等方面显著更高（效应量从0.44到1.56），但反馈帮助度评分无显著差异。

**⚠️ 局限性**

局限性包括样本量小、单次实验、TA归因采用欺骗且未提供真实人类反馈、时间差异可能影响归因效果，结果需要在更大样本和长期学习场景中验证。

---

## 182. SurveyLens: A Research Discipline-Aware Benchmark for Automatic Survey Generation

**arXiv ID:** 2602.11238 | [PDF](https://arxiv.org/pdf/2602.11238v1)

**作者:** Beichen Guo `[一作]` (Hong Kong Polytechnic University), Shuaiqi Liu `[通讯]` (Alibaba Cloud)

**关键词:** `Computation and Language` `Generation` `Benchmark` `Transformer` `Large Language Model` `Text` `Review/Survey Paper` `Benchmark`

**🎯 论文内容**

构建了 SurveyLens‑1k 数据集，并提出了基于领域规范的评价框架，用以评估跨学科自动综述生成方法。

**💡 创新点**

①首个跨学科自动综述生成基准；②从文本中提取领域特定 rubrics 并用 Bradley‑Terry 模型学习权重；③提出 RAMS 与 TAMS 两种针对冗余和语义对齐的指标。

**🔧 技术方法**

基于大型语言模型的 “LLM‑as‑a‑judge”、混合规则+LLM 的结构抽取、Hungarian 匹配、Redundancy‑Aware Matching 与 Thresholded Average Max Similarity 等技术。

**📊 数据集**

SurveyLens‑1k（1,000 篇涵盖 10 个学科的人工综述），以及 11 个现有 ASG 系统、两种 Vanilla LLM 与两种 Deep Research Agent 进行对比实验。

**📈 对比分析**

采用领域级 Rubric 得分、内容/结构/引用组件分数以及 RAMS/TAMS 对齐度进行多维比较，发现 Deep Research Agent 在大多数领域领先；ASG 系统在结构上更强，Vanilla LLM 在人文领域更佳；但所有系统在引用生成与整体一致性上仍表现欠佳。

**⚠️ 局限性**

对内容维度评估一致性低；引用生成质量普遍差；评估过度依赖单一 LLM 判断；数据集仅涵盖 10 学科，缺少多模态元素；实验中未考察生成速度与算力成本。

---

## 183. Fast Evaluation of Truncated Neumann Series by Low-Product Radix Kernels

**arXiv ID:** 2602.11843 | [PDF](https://arxiv.org/pdf/2602.11843v1)

**作者:** Piyush Sao `[一作]` (Oak Ridge National Laboratory), Piyush Sao `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 350 | [OpenAlex ID](https://openalex.org/A5048170299)

**关键词:** `Mathematical Software` `Computational Efficiency` `Optimization`

**🎯 论文内容**

设计并实现了高阶基数（radix）核，显著降低了稠密矩阵中计算截断 Neumann 系列的矩阵乘法次数。

**💡 创新点**

首次构造出精确的 radix‑9 三乘法核和近似的 radix‑15 四乘法核，提出了 residual‑based radix‑kernel 框架，解决了高基数核中 spillover 的问题，并取得了更优的渐进乘法率。

**🔧 技术方法**

采用了重复平方、基数扩展核、数值优化、残差驱动的基数核框架等技术，结合稠密矩阵乘法的快速实现。

**📊 数据集**

未公开特定数据集，实验使用随机生成的稠密矩阵和标准基准矩阵进行评估。

**📈 对比分析**

与传统重复平方方法对比，radix‑9 在 log₂k 维度下只需约1.58×的乘法，较之重复平方节省约21%；在 residual 框架下，radix‑15 达到约1.54×的乘法率，实验验证了乘法次数和运行时间的显著下降。

**⚠️ 局限性**

主要限制在于近似核存在 spillover，需额外的残差校正；高阶精确核的构造仍有限；对极大规模矩阵的并行实现仍需进一步研究。

---

## 184. Finding Sense in Nonsense with Generated Contexts: Perspectives from Humans and Language Models

**arXiv ID:** 2602.11699 | [PDF](https://arxiv.org/pdf/2602.11699v1)

**作者:** Katrin Olsen `[一作]`, Sebastian Padó `[通讯]` (University of Stuttgart)

**通讯引用:** 6066 | [OpenAlex ID](https://openalex.org/A5003870894)

**关键词:** `Computation and Language` `Anomaly Detection` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

对五个常用的语义异常数据集进行语义可理解性评估，收集人工与LLM的分数，并让LLM生成上下文来检验语义异常句子是否可被解释。

**💡 创新点**

证明多数被标记为“不可理解”的句子在有情境时仍具可理解性；展示LLM能生成符合语义的情境并显著提升句子可理解性；对比LLM与人工分数揭示LLM对上下文影响的相似性。

**🔧 技术方法**

使用两种大型语言模型（Phi 4 Mini Instruct 与 Llama 3.1 8B Instruct）进行零样本提示，生成情境、可理解性评分；人工评估采用7分李克特量表并用Kendall τ检验一致性。

**📊 数据集**

ADEPT、BLiMP、PAP、CConS、Cusp 五个公开语义异常/无意义句子数据集，共计约200条样本（每个子类型20条）。

**📈 对比分析**

与人工评分对比，LLM在无情境下的分数普遍低于人工，但在给定情境后可理解性提升率分别为72.7%（Phi）和76.4%（Llama）。两模型在情境对分数的影响上与人工表现相近，且对自身生成情境没有明显偏好。

**⚠️ 局限性**

实验仅覆盖两种LLM，且仅使用英语；可能存在数据集在模型预训练中出现的重叠；缺乏跨语言验证和更大规模模型的评估。

---

## 185. Disentangling Ambiguity from Instability in Large Language Models: A Clinical Text-to-SQL Case Study

**arXiv ID:** 2602.12015 | [PDF](https://arxiv.org/pdf/2602.12015v1)

**作者:** Angelo Ziletti `[一作]` (Bayer AG), Leonardo D'Ambrosi `[通讯]` (Bayer AG)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Text` `Electronic Health Records`

**🎯 论文内容**

提出CLUES框架，将文本到SQL的输出不确定性分解为输入歧义和模型不稳定性，并通过Schur补构造热核相似度矩阵来量化。

**💡 创新点**

首次在两阶段生成模型中利用二分图的Schur补实现语义不确定性拆分，同时将歧义和不稳定性映射为可执行的路由决策。

**🔧 技术方法**

使用LLM生成解释和答案、基于热扩散核的相似度矩阵、Schur补与von Neumann熵、热扩散参数τ校准以及多模型多温度采样。

**📊 数据集**

AmbigQA、SituatedQA、EpiAskKB（带多解释的临床Text‑to‑SQL）以及Optum Clinformatics数据库的真实查询。

**📈 对比分析**

与Kernel Language Entropy对比，CLUES在AUROC上提升约0.07-0.08；在高不确定性查询中将错误率从约5%降至2%；高高区错误占比51%，实现25%查询的二级人工复核。

**⚠️ 局限性**

计算成本高（多轮LLM调用）、解释质量依赖LLM、在错误稀疏的部署环境中路由效率有限，需进一步优化自适应路由。

---

## 186. General and Efficient Steering of Unconditional Diffusion

**arXiv ID:** 2602.11395 | [PDF](https://arxiv.org/pdf/2602.11395v1)

**作者:** Qingsong Wang `[一作]` (Halicioğlu Data Science Institute), Yusu Wang `[通讯]` (Halicioğlu Data Science Institute)

**关键词:** `Machine Learning` `Generation` `Data Synthesis` `Computational Efficiency` `Diffusion model` `Image`

**🎯 论文内容**

提出了一种在推理阶段无需梯度计算的无条件扩散模型控制方法，结合噪声对齐与递归特征机（RFM）方向迁移实现高效、精确的类别导向。

**💡 创新点**

创新点在于将高噪声阶段的噪声对齐与低噪声阶段的RFM学习方向结合，且RFM方向可跨时间步迁移，避免了传统梯度引导的高计算成本。

**🔧 技术方法**

采用噪声对齐（PCA估计）、递归特征机（RFM）学习概念方向、CFG风格增强以及DDIM采样等技术。

**📊 数据集**

在CIFAR-10、ImageNet、CelebA-HQ与Birds-525等数据集上进行实验。

**📈 对比分析**

与训练自由梯度指导（TFG）及传统分类引导等方法比较，实验显示在CIFAR-10上准确率达96.6%（对比TFG 77.1%）、ImageNet平均准确率75.8%（对比TFG 59.8%），FID显著下降，并且推理速度提升约10–16倍。

**⚠️ 局限性**

在极端高噪声阶段对方向的解释性有限；对于稀有属性组合的指导效果仍受限于训练数据稀疏性，且对极端稀有概念的泛化仍存在挑战。

---

## 187. Robot-DIFT: Distilling Diffusion Features for Geometrically Consistent Visuomotor Control

**arXiv ID:** 2602.11934 | [PDF](https://arxiv.org/pdf/2602.11934v1)

**作者:** Yu Deng `[一作]` (TU Darmstadt), Georgia Chalvatzaki `[通讯]` (TU Darmstadt)

**关键词:** `Robotics` `Robotic Intelligence` `Knowledge Distillation` `Convolutional Neural Network` `Diffusion model` `Reinforcement Learning` `Image`

**🎯 论文内容**

本研究提出一种新的视觉表征框架 Robot-DIFT，通过从稳定扩散模型中蒸馏几何先验，生成可实时、确定性的特征网络，提升接触丰富的操控性能。

**💡 创新点**

创新点在于识别视觉后端与闭环控制的结构不匹配问题，提出“Manifold Distillation”将生成式扩散特征迁移到确定性网络，并设计 S2-FPN 多尺度几何与语义融合，最终实现低延迟、高精度的机器人控制。

**🔧 技术方法**

采用的技术包括 Stable Diffusion v2.1 的潜在扩散模型、U‑Net 架构、S2‑FPN 多尺度特征金字塔、对齐损失与权重退火、语言‑视觉跨注意力、Diffusion Policy 策略网络以及对教师网络的冻结蒸馏。

**📊 数据集**

主要使用的数据集有大规模机器人演示集 DROID 用于预训练，仿真评测集 RoboCasa 与 LIBERO‑10，用于比较和基准测试，以及真实 Franka Emika Panda 机器人实验。

**📈 对比分析**

在 RoboCasa 上与 CLIP、SigLIP、DINOv2、DINOv3、DIFT 等基线相比，Robot‑DIFT 平均成功率提升至 0.49（高达 18/24 项任务），在 LIBERO‑10 上成功率达到 0.93，速度仅 0.01 s/轨迹，比 UVA 等大模型快 23 倍且性能更优。

**⚠️ 局限性**

局限性包括仅验证了刚体任务，对非刚体或动态柔性物体的适用性尚未探究；模型只蒸馏了 Stable Diffusion v2.1 的特征，未考虑更丰富的时空生成模型；以及在极端光照或遮挡条件下的鲁棒性待进一步验证。

---

## 188. ReTracing: An Archaeological Approach Through Body, Machine, and Generative Systems

**arXiv ID:** 2602.11242 | [PDF](https://arxiv.org/pdf/2602.11242v1)

**作者:** Yitong Wang `[一作]` (Carnegie Mellon University), Yue Yao `[通讯]` (Columbia University)

**通讯引用:** 3080 | [OpenAlex ID](https://openalex.org/A5062601761)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Robotic Intelligence` `Transformer` `Large Language Model` `Diffusion model` `Text` `Video` `Point Cloud`

**🎯 论文内容**

本研究通过大型语言模型（LLM）将科幻小说文本转化为“可做”和“不可做”的动作提示，并利用扩散式文本到视频模型将这些提示转化为舞蹈指导，进而让人类表演者与四足机器人在镜面舞台上共同完成动作，随后通过多摄像头追踪和3D点云重建将两者的运动轨迹数字化，形成一份关于AI如何编码身体动作与社会偏见的数字档案；

**💡 创新点**

创新点在于提出“AI考古学”视角，将生成式AI的内部逻辑通过身体动作可视化并归档，首次以多主体（人、机器人、AI）共演的方式揭示生成模型中潜在的社会文化偏见及其对身体行为的影响；

**🔧 技术方法**

主要技术包括：Qwen‑2.5 LLM用于文本到动作提示的生成；扩散式文本到视频模型用于生成人类动作示范视频；机器人程序化动作生成；多摄像头运动追踪与3D点云重建技术；以及单目3D姿态估计模型；

**📊 数据集**

数据集由七本科幻/实验小说（如《Frankenstein》《The Handmaid’s Tale》等）的运动相关摘录构成，随后通过LLM生成的正负提示以及对应的人类/机器人动作序列和追踪得到的3D运动轨迹；

**📈 对比分析**

本研究主要采用定性对比，观察人类与机器人在相同提示下动作的差异与一致性，并将生成的运动轨迹与原始文本进行对照以评估偏见映射；由于缺乏标准评测指标，性能评估以可视化质量与偏见揭示度为主，未给出数值化指标；

**⚠️ 局限性**

局限性包括：生成提示可能携带LLM训练数据中的性别/种族偏见，导致动作体现刻板印象；数据隐私与主体权利未得到充分保障；缺乏客观量化评估方法；以及对不同文化语境的通用性验证不足。

---

## 189. Pareto-Efficient Multi-Buyer Mechanisms: Characterization, Fairness and Welfare

**arXiv ID:** 2602.11967 | [PDF](https://arxiv.org/pdf/2602.11967v1)

**作者:** Moshe Babaioff `[一作]` (Hebrew University of Jerusalem), Yiding Feng `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 3953 | [OpenAlex ID](https://openalex.org/A5103218231)

**关键词:** `Computer Science and Game Theory` `Optimization` `Lagrangian Method`

**🎯 论文内容**

本文研究单件物品、多个买家的贝叶斯机制设计问题，系统地刻画了在满足可激励、可实现且对买家对称的机制空间内，卖家收入与买家总剩余的 Pareto 前沿，并基于此研究了两种经典的公平选择方案：Kalai‑Smorodinsky（KS）与 Nash 方案。

**💡 创新点**

创新点包括：① 在正则且 anti‑MHR 分布下，完整地表征了整个 Pareto 前沿，证明所有极点可由保留价不超过最低垄断保留的二价拍卖实现；② 通过 Lagrangian 对偶法，将前沿刻画转化为卖家收益最大化的约束优化问题，得到了对 MHR 与 anti‑MHR 情况下极点结构的精确描述；③ 对 KS 与 Nash 方案给出了严格的社会福利下界，揭示了 KS 在任何分布下至少达到 50% 的最优福利，并在正则+anti‑MHR 情况下趋近于 1；相反，Nash 方案在 MHR 情况下可退化到任意小比例；④ 将上述分析扩展到多单元情形，给出基于供给与需求比例的 KS 福利下界。

**🔧 技术方法**

主要技术：对偶变换与 Lagrangian 方法；对期望收益与买家剩余的混合目标进行极值分析；对价格-分配函数的可实现性利用 Border 条件；对收益曲线与虚价值函数的凸/凹性利用 regular、MHR、anti‑MHR 的数学性质；对 KS 与 Nash 方案的公平性约束建模并计算其在 Pareto 前沿上的交点。

**📊 数据集**

本工作为理论性研究，没有使用真实数据集，而是基于连续概率分布（如均匀、指数、Pareto 等）在假设的 i.i.d. 环境下进行分析。

**📈 对比分析**

与传统的单目标（最大化收益或福利）机制相比，本文提供了多目标的 Pareto 前沿与公平性约束的完整描述；通过对 KS 与 Nash 方案的最坏情况下界与渐近上界的计算，证明了 KS 在广泛分布下的鲁棒性；相比之下，Nash 方案在 MHR 环境下效率可低至接近 0；在正则+anti‑MHR 环境下，三种方案均可实现接近最优福利。

**⚠️ 局限性**

局限性：① 需要分布满足 regular、MHR 或 anti‑MHR 等结构性假设，无法直接推广到任意分布；② 对多买家情形的分析依赖于 i.i.d. 与单单元（或多单元但同质）假设；③ 只给出了理论下界与上界，实际实现时可能受限于机制复杂度与信息披露；④ 对 Nash 方案的负面结果在 MHR 环境下可能过于极端，实际市场中可能出现更好表现，但理论上仍存在退化风险。

---

## 190. PRIME: A Process-Outcome Alignment Benchmark for Verifiable Reasoning in Mathematics and Engineering

**arXiv ID:** 2602.11570 | [PDF](https://arxiv.org/pdf/2602.11570v1)

**作者:** Xiangfeng Wang `[一作]` (University of Science and Technology of China), Daxin Jiang `[通讯]` (StepFun)

**关键词:** `Computation and Language` `Reinforcement Learning` `Reinforcement Learning` `Text` `Benchmark`

**🎯 论文内容**

提出了 PRIME 这一专门评估大型推理模型（LRM）在数学与工程领域中过程与结果一致性的基准，并基于该基准设计了过程感知的 RLVR 训练框架；

**💡 创新点**

①首个聚焦过程‑结果一致性的高质量基准；②将过程一致性直接纳入奖励信号，显著抑制“幸运猜测”；③发现基准准确率与 RLVR 下模型性能高度线性相关（R²>0.92）；

**🔧 技术方法**

多阶段自动化处理与人工评注管道、模型筛选验证、过程一致性标注、过程感知奖励函数、与多模型对比实验；

**📊 数据集**

起始 7M 条高校教材与考试题目，经过可验证性与答案校验两阶段过滤得到 𝒬_clean；随后生成 2,530 条包含完整推理轨迹的 PRIME 样本，覆盖 16 细分学科与 480 子领域；

**📈 对比分析**

与传统仅基于结果的奖励相比，过程感知 RLVR 在 AIME24、AIME25、Beyond-AIME 上分别提升 8.29%、9.12%、7.31%；在基准评测中，验证器整体准确率与后续 RLVR 性能呈 R²>0.92 的强正相关；

**⚠️ 局限性**

未从零训练专门的过程验证模型，而是直接使用现有强大推理模型作为验证器；基准仅覆盖 STEM 领域，未考察多步规划或工具使用等更复杂情境；

---

## 191. scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery

**arXiv ID:** 2602.11609 | [PDF](https://arxiv.org/pdf/2602.11609v1)

**作者:** Yiming Gao `[一作]` (Texas A and M), Eric P. Xing `[通讯]` (Carnegie Mellon University)

**通讯引用:** 40619 | [OpenAlex ID](https://openalex.org/A5009547049)

**关键词:** `Artificial Intelligence` `Drug Discovery` `Transformer` `Large Language Model` `Prompt Engineering` `Biomedical Data` `Benchmark`

**🎯 论文内容**

提出一种基于大语言模型的“Omics‑Native Reasoning（ONR）”框架，能在单细胞 RNA‑seq 数据上直接推理、调用专用工具完成细胞类型注释、发育轨迹重建和转录因子调控预测，并记录可追溯的自然语言推理轨迹。

**💡 创新点**

①将LLM的自然语言推理与单细胞分析工具紧密耦合，支持多轮迭代自我校正；②首次提供面向单细胞分析的系统化 benchmark（scONRbench）；③构建了可复用的 scPilot 工具箱，实现压缩数据到可读文本、结构化工具调用与推理链。

**🔧 技术方法**

大语言模型（GPT‑4o、o1 等）、问题转文本转换器、精心挑选的生物工具库（Scanpy、Monocle、SCENIC 等）、JSON 结构化输出、迭代 prompt 与自我审计机制。

**📊 数据集**

共 9 个手工标注或实验验证的单细胞 RNA‑seq 数据集：PBMC3k、Liver、Retina、Pancreas、Neocortex、Stomach、Kidney（GRNdb/TRRUST）等；同时构建了 benchmark 数据集集。

**📈 对比分析**

与传统单细胞工具（CellTypist、CellMarker、Monocle）以及其他 LLM 基础方法（Biomni、GPTCelltype 等）在 9 个任务上对比。实验表明，ONR 通过迭代推理可使细胞类型注释准确率提升 11%，轨迹图编辑距离下降 30%，GRN AUROC 提升 0.03，整体性能显著优于传统和单步 LLM。

**⚠️ 局限性**

数据压缩可能导致稀有细胞信号丢失；LLM 可能产生幻觉或错误推理；对极大规模数据的可扩展性有限；缺少实验验证循环以评估预测的生物学真实性。

---

## 192. Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review

**arXiv ID:** 2602.11173 | [PDF](https://arxiv.org/pdf/2602.11173v1)

**作者:** Qian Ruan `[一作]` (Technical University of Darmstadt), Iryna Gurevych `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 25410 | [OpenAlex ID](https://openalex.org/A5027450194)

**关键词:** `Computation and Language` `Generation` `Optimization` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

提出一种作者参与的答复生成框架 REspGen，并构建 Re^3Align 数据集及 REspEval 评估工具

**💡 创新点**

首次将作者的专业知识与意图（通过修订信息）嵌入答复生成，提供多属性控制和评估驱动的迭代优化

**🔧 技术方法**

基于大模型的生成、检索-重排序、长度与计划控制，以及 GPT‑5 驱动的评估与细化

**📊 数据集**

Re^3Align：3.4k 论文记录，15k 对齐的评审‑答复‑修订三元组；公开版可在 https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4982

**📈 对比分析**

在 EMNLP24 与 PeerJ 数据上，五大 LLM 在九种设置下进行对比；引入 20+ 维度评估，实验显示作者输入显著提升质量，长度/计划控制有利可循，评估驱动的细化能提升 2–5% 的质量分数

**⚠️ 局限性**

仅限英文科研论文，未评估多语言或非学术领域；实现依赖顶尖 LLM，缺乏对小模型或其他架构的系统评估

---

## 193. Unravelling Abstract Cyclic Proofs into Proofs by Induction

**arXiv ID:** 2602.12054 | [PDF](https://arxiv.org/pdf/2602.12054v1)

**作者:** Lide Grotenhuis `[一作]`, Daniël Otten `[通讯]`

**关键词:** `Logic in Computer Science`

**🎯 论文内容**

本文提出一种抽象的循环证明框架，证明任何循环证明都能通过加入最小化归纳原理转化为有限的归纳证明，并且转换过程中保持原循环证明的结构；并展示该方法可推广到多种类型的证明系统（如循环 Heyting 算术与普通 Heyting 算术的转化）。

**💡 创新点**

创新点在于：①将循环证明与传统的有限证明统一在一个基于大小变化图的抽象框架中；②设计了带有统一覆盖规则的重置证明（reset proof），能处理非线性归纳类型；③提供了结构保持的转化算法，保留了原循环证明的语义和顺序；④将此方法推广到多种类型的系统。

**🔧 技术方法**

技术手段包括：基于大小变化图的循环调用系统、带注释的循环表示、重置证明、展开（unfolding）技巧、递归调用的归纳顺序控制、well‑founded 归纳原理与 >、≥ 关系的演绎规则。

**📊 数据集**

本文为纯理论研究，未使用任何实验数据集。

**📈 对比分析**

由于研究的本质是理论证明与结构保持性分析，本文没有实验性能评估或与其它方法的对比，只给出了形式化的证明与结构保持性定理。

**⚠️ 局限性**

局限性包括：只覆盖递归（归纳）类型，未处理共归纳类型；在经典元理论下证明，缺乏构造主义的实现；尚未与实际证明助手的实现细节对齐；对更复杂的多值或更高阶递归结构的适用性仍待进一步研究。

---

## 194. Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning

**arXiv ID:** 2602.11779 | [PDF](https://arxiv.org/pdf/2602.11779v1)

**作者:** Haoran Dang `[一作]` (Tsinghua University), Yan Lu `[通讯]` (Microsoft Research Asia)

**通讯引用:** 19932 | [OpenAlex ID](https://openalex.org/A5035278528)

**关键词:** `Machine Learning` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Large Language Model` `Text`

**🎯 论文内容**

提出了温度自适应元策略框架TAMPO，利用轨迹信息在LLM强化学习中动态调整采样温度；

**💡 创新点**

将采样温度视为可学习的元策略，利用已有轨迹的似然和优势实现无额外rollout的温度更新；

**🔧 技术方法**

结合Critic‑free RL（GRPO）、轨迹似然温度优势、EMA平滑、top‑p采样等技术；

**📊 数据集**

训练集使用公开数学推理数据open‑s1；评估在AIME24、MATH‑500、AMC23、Minerva、OlympiadBench等五个数学基准，另外在ECQA进行验证；

**📈 对比分析**

与固定温度（0.9、1.2、1.5）及线性升温（0.9→1.5）GRPO对比，TAMPO平均提升Pass@1约1.9%、Pass@8约1.7%，在所有基准上至少排名前二；在ECQA上提升约1.1% Pass@1；

**⚠️ 局限性**

需要预先设定温度候选集；推理时仍使用固定温度；元策略采样阈值需手动调参；对更大模型和更复杂任务的泛化尚待进一步验证；

---

## 195. Is Online Linear Optimization Sufficient for Strategic Robustness?

**arXiv ID:** 2602.12253 | [PDF](https://arxiv.org/pdf/2602.12253v1)

**作者:** Yang Cai `[一作]` (Yale University), Weiqiang Zheng `[通讯]` (Yale University)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5078958679)

**关键词:** `Computer Science and Game Theory` `Optimization`

**🎯 论文内容**

本文研究了在重复贝叶斯一价拍卖中的竞价算法，证明子线性线性化损失（linearized regret）足以保证算法在卖家操纵下的策略鲁棒性，并给出了将任意在线线性优化（OLO）算法转换为既有低损失又具策略鲁棒性的通用黑盒降维方法。

**💡 创新点**

创新点在于：①揭示子线性线性化损失比传统外部损失更强，能够保证策略鲁棒性；②设计了一个简洁的黑盒降维框架，使任何OLO算法（如乘数权重更新 MWU）均可实现低损失与鲁棒性；③在未知价值分布下，提出了“支配连续经验分布”估计方法，消除了对密度上界的依赖，并将对数K的影响从线性降低到对数。

**🔧 技术方法**

主要技术包括：在线线性优化（OLO）与凸优化的结合；新的凸表述（将策略空间重参数化为概率简单形），从而实现对数K的改进；对线性化损失的理论分析；经验分布估计与“支配连续经验分布”构造；以及对分布估计误差的稳健性分析。

**📊 数据集**

本文为理论性工作，无需使用具体数据集；所有结果均在随机值分布 F 的假设下证明。

**📈 对比分析**

相较于之前仅有 OGA（在线梯度上升）获得 O(√(TK)) 误差和鲁棒性，本文通过 MWU 在已知分布下得到 O(√(T log K)) 的误差与鲁棒性；在未知分布下，相较于先前需要已知密度上界并且误差为 O(f̅^{1/2}K√T) 的方法，本文实现了 O(√(T(log K + log(T/δ)))) 的误差与鲁棒性，且不再依赖密度上界。

**⚠️ 局限性**

主要局限包括：①研究仅覆盖单买家一价拍卖场景，未考察多买家或更一般的拍卖格式；②虽然理论上对 K 的依赖已降到对数，但在实际高精度离散化时仍需关注分布估计误差；③缺乏实证实验验证，所有结果均为理论证明。

---

## 196. BlackCATT: Black-box Collusion Aware Traitor Tracing in Federated Learning

**arXiv ID:** 2602.12138 | [PDF](https://arxiv.org/pdf/2602.12138v1)

**作者:** Elena Rodríguez-Lois `[一作]` (University of Vigo), Fernando Pérez-González `[通讯]` (University of Vigo)

**关键词:** `Cryptography and Security` `Federated Learning` `Safety and Privacy` `Convolutional Neural Network` `Image`

**🎯 论文内容**

研究了一种适用于联邦学习的黑盒逆袭者追踪方法BlackCATT，能在存在协作攻击时实现模型水印的鲁棒追踪。

**💡 创新点**

引入了协作感知嵌入损失、共享触发器的对抗优化以及功能正则化，提升了抗合并攻击的鲁棒性并保持主任务性能。

**🔧 技术方法**

利用Tardos码、任务算术、对抗扰动生成（PGD）、交叉熵损失与功能KL正则化等技术。

**📊 数据集**

在CIFAR-10、CIFAR-100上验证，采用VGG16和ResNet18两种网络。

**📈 对比分析**

与Vanilla水印、之前的黑盒方案对比，BlackCATT在10-20名参与者时在抵御参数平均和随机层采样等合并攻击时的MAV显著降低，FNR可在≈0.5以下，且主任务精度差异不大。

**⚠️ 局限性**

仅在i.i.d.数据、可信聚合器、分类任务上验证，未处理非i.i.d.、聚合器不可信、生成模型等场景，且对抗训练可能提高误报率。

---

## 197. Dopamine: Brain Modes, Not Brains

**arXiv ID:** 2602.11726 | [PDF](https://arxiv.org/pdf/2602.11726v1)

**作者:** Shervin Ghasemlou `[一作]`, Shervin Ghasemlou `[通讯]`

**关键词:** `Machine Learning` `Classification` `Recognition` `Optimization` `Explainability and Interpretability` `Transformer` `Supervised Fine-Tuning` `Image`

**🎯 论文内容**

提出一种激活空间的参数高效微调方法（TauGate），通过学习每个神经元的阈值和增益，冻结基础模型的权重，实现对不同模式的自适应和条件计算。

**💡 创新点**

创新点在于将模型适配视为激活阈值与增益的调整而非权重更新，从而得到可解释的门控子网络；仅使用几百个可训练参数即可在旋转MNIST上实现模式专门化；并通过可硬化门控提供显式的“哪一层神经元会激活”的解释。

**🔧 技术方法**

采用基于Sigmoid的平滑门控（g = σ(s·(z-τ))），阈值τ和增益γ的学习；硬门（g_hard = 1[z>τ]）用于推理时的条件计算；稀疏正则化鼓励激活稀疏；与Bias tuning、LoRA、Full FT等PEFT方法进行对比。

**📊 数据集**

在MNIST与其45°旋转版本的数据集上进行实验，先在两种模式混合上预训练小型MLP（基础网络），然后冻结权重，使用TauGate进行旋转模式的专门化。

**📈 对比分析**

与Frozen、BitFit、LoRA、Full FT等方法在同一冻结基础上对比：TauGate在旋转模式下取得高于Frozen的准确率（≈0.86%）且参数仅约512个；LoRA在准确率上更优（≈0.864%）但参数量（≈10k）更大；Full FT虽在旋转模式上表现最佳（≈0.887%）但会导致原始模式性能下降。

**⚠️ 局限性**

局限性包括：当基础网络缺乏目标模式所需特征时，门控无法“发明”新特征，需采用权重空间更新；扩展到大型Transformer时需要考虑门控位置、任务条件化以及稀疏计算的实际加速问题；门的锐度控制与梯度稳定性之间存在折中。

---

## 198. Spatial Chain-of-Thought: Bridging Understanding and Generation Models for Spatial Reasoning Generation

**arXiv ID:** 2602.11980 | [PDF](https://arxiv.org/pdf/2602.11980v1)

**作者:** Wei Chen `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 95263 | [OpenAlex ID](https://openalex.org/A5100333572)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Large Language Model` `Diffusion model` `Vision Language Model` `Chain-of-Thought` `Image` `Multimodality`

**🎯 论文内容**

提出 Spatial Chain-of-Thought (SCoT) 框架，将多模态大语言模型的空间推理能力通过结构化的边界框链路传递给扩散模型，实现更精确的空间约束图像生成。

**💡 创新点**

创新点在于引入交错文本-坐标指令格式，使扩散模型直接解析对象级边界框，而无需联合预训练或纯文本压缩，从而兼顾效率、可插拔和空间稠密性。

**🔧 技术方法**

采用离散流匹配训练的空间感知扩散模型 (SAGen)、Gemini 3 Pro 等 MLLM 规划器以及 Qwen3-VL 视觉语言模型生成的坐标指令。

**📊 数据集**

使用自建的 SCoT-DenseBox（大规模密集边界框）和 SCoT-AestheticSFT（高美观度数据）两套数据集，并在 COCO-MIG、T2ICoReBench、GenEval、OneIG-Bench 等公开基准上评测。

**📈 对比分析**

与多种基线（如 GLIGEN、InstanceDiffusion、Text CoT 等）对比，SCoT 在复杂空间推理任务上提升约 10% 以上，整体得分最高，同时在布局控制指标（SR、I‑SR、mIoU）上均取得领先。

**⚠️ 局限性**

局限性包括对 MLLM 模型规模的依赖，较大模型提升效果明显；以及在极端稠密或非结构化场景下仍可能出现坐标冲突或生成质量下降。

---

## 199. U-Former ODE: Fast Probabilistic Forecasting of Irregular Time Series

**arXiv ID:** 2602.11738 | [PDF](https://arxiv.org/pdf/2602.11738v1)

**作者:** Ilya Kuleshov `[一作]` (Applied AI Institute), Alexey Zaytsev `[通讯]` (Applied AI Institute)

**关键词:** `Machine Learning` `Transformer` `Ordinary Differential Equation` `Time Series`

**🎯 论文内容**

提出U-Former ODE (UFO)，一种专为不规则多变量时间序列设计的概率预测模型，能够在保持全局感受野的同时实现快速并行推断；

**💡 创新点**

将U-Net多尺度结构、Transformer的全局注意力与神经受控微分方程（Neural CDE）相结合，构造时间并行的连续动力学框架，从而克服了传统Neural CDE的顺序瓶颈和纯Transformer的局部性；

**🔧 技术方法**

采用U-Net风格的下/上采样、Transformer编码解码器、Neural CDE重采样、kernel‑based 插值、SwiGLU 向量场、交叉注意力以及变分采样等技术；

**📊 数据集**

在五个公开基准（ETTm1、ETTm2、Electricity、Weather、Traffic）上进行实验，分别使用原始规则数据以及通过随机删去30%天后得到的不规则数据；

**📈 对比分析**

与10个先进基线（DeepAR、PatchTST、LaST、K^2VAE、TSDiff、LatentODE、Neural CDE、TFM、SLiCE、NCDE++）在Normalized CRPS (NCRPS)指标下对比，UFO在所有任务中均实现最低或相近的NCRPS，并且推断时间比传统Neural CDE快约15倍，且与最快的K^2VAE相当；

**⚠️ 局限性**

仍面临高维度（如Electricity、Traffic）下的训练与推断成本较高，对极大缺失块的鲁棒性及多步自回归动态更新的进一步验证仍需研究。

---

## 200. Temporal Difference Learning with Constrained Initial Representations

**arXiv ID:** 2602.11800 | [PDF](https://arxiv.org/pdf/2602.11800v1)

**作者:** Jiafei Lyu `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**通讯引用:** 10630 | [OpenAlex ID](https://openalex.org/A5100754504)

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning` `Sequential`

**🎯 论文内容**

提出了CIR框架，通过在网络初层使用tanh激活及归一化来约束输入表示，并加入跳连与凸Q学习提升样本效率。

**💡 创新点**

创新点在于从输入表示层面直接约束初始特征，利用tanh限制范围、AvgRNorm自适应缩放、跳连结构和凸Q学习实现更稳健的学习。

**🔧 技术方法**

使用了tanh激活、AvgRNorm、层归一化、跳连U形网络、凸Q学习、SMR等技术。

**📊 数据集**

在DMControl、HumanoidBench和ODRL这三大连续控制基准上进行评测。

**📈 对比分析**

与SAC、TD7、BRO、DreamerV3、TD-MPC2、SimBa等对照，CIR在多数任务上实现了更高的样本效率且计算时间不增加；在ODRL任务中表现优于传统方法。

**⚠️ 局限性**

局限在对复杂任务的性能不如SimBa和BRO，且对跳连与tanh等设计需调参，未针对特定领域进行专门优化。

---

## 201. ULTRA:Urdu Language Transformer-based Recommendation Architecture

**arXiv ID:** 2602.11836 | [PDF](https://arxiv.org/pdf/2602.11836v1)

**作者:** Alishbah Bashir `[一作]` (Pakistan Institute of Engineering and Applied Sciences), Ijaz Hussain `[通讯]` (Pakistan Institute of Engineering and Applied Sciences)

**通讯引用:** 3438 | [OpenAlex ID](https://openalex.org/A5043722405)

**关键词:** `Information Retrieval` `Recommendation System` `Transformer` `Text`

**🎯 论文内容**

构建了一种面向乌尔都语新闻的双嵌入推荐框架 ULTRA，用查询长度自适应路由来实现短查询使用标题嵌入、长查询使用全文嵌入的语义推荐。

**💡 创新点**

创新点包括：①查询长度阈值驱动的动态路由机制；②标题与正文的双嵌入策略；③针对不同文本长度优化的池化（标题使用 CLS，正文使用均值池化）与 PCA 维度压缩；④在大规模向量数据库（ChromaDB）中实现实时高精度检索。

**🔧 技术方法**

技术手段：使用预训练乌尔都语 Transformer（urduhack/roberta-urdu-small）生成 768 维上下文嵌入；对标题与正文分别采用 CLS 与均值池化；通过 PCA 进行 64 维和 128 维降维；采用 HNSW 索引的 ChromaDB 进行高效近邻检索；评估指标为 Precision@k 与人工评判。

**📊 数据集**

数据集：来自 Kaggle 的 112,000 条乌尔都语新闻，包含 5 类（科学技术、商业经济、娱乐体育等），平均标题长度 52 字符，正文平均 986 字符。

**📈 对比分析**

对比方法：传统 TF‑IDF + 余弦相似度、之前基于 BERT 的单一嵌入模型。ULTRA 在短查询上 Precision@15 提升至 94.35%（相较 71.43% 的基线提升 32%），长查询上 Precision@15 达 98.53%。实验覆盖 100 条短查询和 100 条长查询，显示高召回与高精度的可行性。

**⚠️ 局限性**

局限性：①阈值 θ 需要根据语料特征手工设定，可能不适用于其他领域；②仅处理文本语义匹配，未结合协同过滤或用户行为；③在多模态或多语言环境下效果未知；④对极长文本的分块策略与聚合仍可能引入信息丢失。

---

## 202. Recurrent Preference Memory for Efficient Long-Sequence Generative Recommendation

**arXiv ID:** 2602.11605 | [PDF](https://arxiv.org/pdf/2602.11605v1)

**作者:** Yixiao Chen `[一作]` (Tencent Inc.), Jie Jiang `[通讯]` (Tsinghua University)

**通讯引用:** 23309 | [OpenAlex ID](https://openalex.org/A5031168904)

**关键词:** `Information Retrieval` `Recommendation System` `Transformer` `Sequential`

**🎯 论文内容**

设计并实现一种名为 Rec2PM 的递归偏好记忆框架，能够将长时间序列用户交互历史压缩成少量记忆 token，并支持在线增量更新，显著降低生成式推荐模型在长期序列上的计算延迟和存储成本。

**💡 创新点**

创新点主要有：
1) 自参照教师强制（self‑referential teacher‑forcing）训练策略，使递归更新过程可完全并行化，解决传统 BPTT 的序列训练难题；
2) 将 Preference Memory 视为信息瓶颈，通过有限的 token 数量实现噪声滤波，提升长期兴趣建模质量；
3) 通过覆盖（Overwriting）和追加（Appending）两种更新模式，在推理时保持模型简洁同时兼顾动态性。

**🔧 技术方法**

技术手段包括：
- 基于 Transformer 的自注意力架构；
- Preference Memory token 化与记忆编码器；
- 自参照教师强制训练，配合全局参考记忆与一致性损失；
- 两阶段并行训练（全局参考生成 + 并行递归更新）；
- 信息瓶颈思想与 MSE 一致性约束；
- 对比实验中使用的 KV‑Mask 与 Tok‑Serial 等竞争方法。

**📊 数据集**

数据集：
- MerRec：大规模电商平台真实交互数据，重点包含长度超过 1000 的用户序列；
- 工业级内容推荐日志：真实商业服务平台的顺序推荐日志，用于验证工业落地效果。

**📈 对比分析**

评估方式：在 MerRec 上与短序列（Short）、全序列（Full）、Token‑Serial、KV‑Mask 等基线进行对比，采用 Hit@K 与 NDCG@K 评价。结果显示 Rec2PM‑O/​A 在精度上略优于全序列模型，同时推理延迟仅 ~10 ms、单用户存储仅 1 KB，显著优于 KV‑Mask（32 KB）和 Token‑Serial（数十 KB）。在工业数据上，Rec2PM 通过一次性压缩 1948 条交互到 20 个记忆槽，获得 H@1/H@10/H@50/H@1000 更高的评分，优于 HSTU‑Full。

**⚠️ 局限性**

局限与待改进：
1) 对记忆槽数 C 的选择敏感，槽过少或过多都会影响性能；
2) 追加更新在实验中效果不如覆盖，且易引入噪声堆积；
3) 训练需要先生成全局参考记忆，额外的前向推理步骤；
4) 目前仅在单域电商/内容推荐上验证，跨域泛化能力待进一步验证；
5) 对极长历史（数万条）一次性压缩可能导致信息损失，需探索多层次记忆策略。

---

## 203. TopoFair: Linking Topological Bias to Fairness in Link Prediction Benchmarks

**arXiv ID:** 2602.11802 | [PDF](https://arxiv.org/pdf/2602.11802v1)

**作者:** Lilian Marey `[一作]` (Telecom Paris), Charlotte Laclau `[通讯]` (Telecom Paris)

**关键词:** `Machine Learning` `Graph Neural Network` `Graph` `Benchmark`

**🎯 论文内容**

提出了基于结构偏差的公平链路预测评估框架，并实现了可调节的图生成方法，系统评估传统与公平链路预测模型。

**💡 创新点**

创新点在于构建统一的结构偏差分类法、扩展Barabási–Albert模型实现可控的敏感属性与同质性调整、以及将结构偏差与公平性指标关联的综合评测框架。

**🔧 技术方法**

使用扩展BA图生成、GNN/embedding链路预测模型（Node2Vec、SVD、NMF）、公平链路预测方法（Fairwalk、Debayes、CrossWalk、FairAdj、FLIP），并通过Hit@10、AUC、统计平等（SP）和机会平等（EO）等指标评估公平性。

**📊 数据集**

采用Polblogs、Facebook ego‑networks、共著网络等真实数据集，同时通过生成模型构造大量合成图来覆盖不同结构偏差场景。

**📈 对比分析**

与传统模型相比，公平链路预测方法在不同结构偏差下表现各异；结构偏差指标能解释约80%以上的公平性方差，公平模型在某些偏差环境下维持较好公平性但可能牺牲准确率，整体显示公平性与结构偏差高度相关。

**⚠️ 局限性**

局限性包括仅考虑二元敏感属性、偏差度量集中于同质性与中心性等指标、实验覆盖的真实网络规模有限，未来需扩展多属性、多值偏差定义以及对更大规模网络的验证。

---

## 204. Multimodal Fact-Level Attribution for Verifiable Reasoning

**arXiv ID:** 2602.11509 | [PDF](https://arxiv.org/pdf/2602.11509v1)

**作者:** David Wan `[一作]` (University of North Carolina at Chapel Hill), Mohit Bansal `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 19306 | [OpenAlex ID](https://openalex.org/A5001987532)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Chain-of-Thought` `Prompt Engineering` `Multimodality` `Video` `Audio` `Benchmark`

**🎯 论文内容**

本文提出了多模态事实层级归因（Multimodal Reasoning with Grounded Attribution）基准，用于评估多模态大语言模型（MLLMs）在需要多步推理和长文本生成场景下对视频、音频、图表等异构输入进行事实层级归因的能力。

**💡 创新点**

创新点在于：①设计了三阶段细粒度评估协议——可验证声明识别、原子事实分解、归因质量；②构建了自动化评估管线并与人工标注高度相关；③系统揭示了推理与归因之间显著的性能鸿沟。

**🔧 技术方法**

技术方法包括：多模态大语言模型的提示（Simple、CoT、JSON）、链式思考、原子事实分解与归因推理；程序化多模态推理框架（逻辑/叙事式、声明式/命令式）；以及基于回归与F1的评估指标。

**📊 数据集**

使用的数据集为WorldSense与Video-MMMU，两者均包含视频、音频、图表等多模态输入，并包含需要跨模态推理的复杂问题。

**📈 对比分析**

通过人类标注与自动化评估对比，评估了Gemini、Qwen-Omni、Qwen-3-VL、Molmo2等主流MLLM；发现虽然答案准确率可达70%以上，但归因正确率仅为30-35%，程序化方法可提升约9.6个百分点但会降低答案准确率。

**⚠️ 局限性**

局限性包括：模型易产生引用幻觉；归因质量与推理深度存在权衡；自动评估仍依赖规则化提示；目前仅覆盖视频、音频、图表等模态，难以推广到更广泛的异构输入。

---

## 205. Maximizing Index Diversity in Committee Elections

**arXiv ID:** 2602.11400 | [PDF](https://arxiv.org/pdf/2602.11400v1)

**作者:** Paula Böhm `[一作]` (Clausthal University of Technology), Till Fluschnik `[通讯]` (Humboldt University of Berlin)

**关键词:** `Computer Science and Game Theory` `Tabular`

**🎯 论文内容**

本文提出了两种在多胜选举中结合多样性指数的模型，并引入了新的词典计数指数来衡量委员会多样性。

**💡 创新点**

创新点在于用生态学多样性指数代替硬性配额，并设计了Lexicographic Counting Index，提供了新的可解释性与可计算性分析；同时给出复杂度定理。

**🔧 技术方法**

技术包括：多胜选举模型的定义，审批偏好与标签候选人的组合；多样性指数的定义与性质分析；NP‑hardness证明；实验评估。

**📊 数据集**

使用的实测数据集主要是参与式预算（PB）实例，来自Pabulib；另外还利用了一些公开的投票/候选人标签数据。

**📈 对比分析**

实验通过比较在给定得分下最优多样性与放宽得分/满意度约束时的多样性提升来评估，结果显示在保持10%得分可接受的情况下多样性平均提升12–19%；而50%降分不一定获得最优多样性。

**⚠️ 局限性**

局限性包括：在每位选民满意度约束下求解仍为NP‑hard，计算复杂度随评分规则变化；实验仅覆盖有限的PB实例，缺乏对更广泛场景的验证；新指数的实际解释仍需进一步验证。

---

## 206. Automated Test Suite Enhancement Using Large Language Models with Few-shot Prompting

**arXiv ID:** 2602.12256 | [PDF](https://arxiv.org/pdf/2602.12256v1)

**作者:** Alex Chudic `[一作]` (US Booking Services Limited), Gül Çalıklı `[通讯]` (University of Glasgow)

**通讯引用:** 619 | [OpenAlex ID](https://openalex.org/A5074378564)

**关键词:** `Software Engineering` `AI Code Assistant` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

本文系统研究了使用少量示例（few‑shot）提示生成的LLM单元测试的质量与对现有测试套件的提升效果，并比较不同示例来源（人类手写、SBST自动生成、LLM自动生成）以及检索式选择方法对生成结果的影响。

**💡 创新点**

创新点在于：①提出检索式示例选择（基于问题描述+代码相似度）以提升few‑shot提示的效果；②首次对示例来源对生成测试的覆盖率、正确性、可读性等多维度进行对比实验；③发现SBST示例在低覆盖率套件中最能提升覆盖率，且“问题+代码”检索方法最优；④证明轻量级规则修复能显著提升LLM生成测试的可执行性。

**🔧 技术方法**

使用技术包括 GPT‑4o 大语言模型、few‑shot prompting、TF‑IDF+余弦相似度检索、规则式修复（补全 import、删除错误代码等）、覆盖率工具（pytest + coverage）、代码质量分析工具（SonarScanner/SonarCloud）以及静态分析（AST、编译、执行检查）。

**📊 数据集**

采用了两套基准数据集：HumanEval（单文件、轻量级）和 ClassEval（跨文件、面向类的真实项目）。

**📈 对比分析**

通过功能正确性（编译/运行通过率）、覆盖率（行/分支）、代码质量指标（圈复杂度、认知复杂度、技术债、代码异味）进行评估。实验结果显示：人类示例生成的测试在覆盖率上最高；SBST示例在提升低覆盖套件时最有效；检索式“问题+代码”策略在覆盖率上略优于随机选取；规则修复后所有示例的通过率均显著提升，覆盖率基本达到或超过 95%。

**⚠️ 局限性**

限制包括：仅评估 GPT‑4o，未验证其它 LLM；实验聚焦 Python 单元测试，未覆盖其他语言和更大规模真实项目；检索方法基于 TF‑IDF，可能忽略语义上下文；覆盖率并不完全衡量缺陷检测能力；规则修复虽有效但对复杂错误仍需更智能化方法。

---

## 207. Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation

**arXiv ID:** 2602.12125 | [PDF](https://arxiv.org/pdf/2602.12125v1)

**作者:** Wenkai Yang `[一作]` (Renmin University of China), Yankai Lin `[通讯]` (Renmin University of China)

**通讯引用:** 12329 | [OpenAlex ID](https://openalex.org/A5043098453)

**关键词:** `Machine Learning` `Reinforcement Learning` `Knowledge Distillation` `Reinforcement Learning` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

本文将对抗式蒸馏（OPD）与密集 KL 约束强化学习（RL）建立理论联系，提出通用的 Generalized On-Policy Distillation (G-OPD) 框架，并通过大量数学推导和实验验证了在同规模多教师蒸馏与强弱蒸馏两种常见场景下，使用奖励外推（reward extrapolation）策略的 ExOPD 能显著提升学生模型性能，甚至在同规模多教师设置中实现统一学生超过所有教师的能力。

**💡 创新点**

创新点包括：
1) 将 OPD 视为特例的密集 KL 约束 RL，揭示奖励与 KL 正则项的权重始终为 1：1；
2) 在 G-OPD 中引入奖励缩放因子 λ 和可变参考模型 π_ref，打破 OPD 的固定权重限制；
3) 证明 λ>1（奖励外推）能让学生模型超越教师；
4) 在强弱蒸馏中通过将教师的预 RL 基础模型作为参考模型实现奖励修正（reward correction），进一步提升性能。

**🔧 技术方法**

技术方法：
- KL 约束强化学习（dense RL）
- On-Policy Distillation (OPD)
- Generalized On-Policy Distillation (G-OPD) 并加入奖励缩放 λ 与参考模型 π_ref
- 采用 GRPO 训练教师模型
- token‑level roll‑out correction
- 交叉熵（SFT）与 ExPO（权重外推）做基线
- 计算梯度时使用 A_t^G-OPD 公式

**📊 数据集**

数据集与评测：
- 训练集：DeepMath（57K）用于数学推理 RL；Eurus-RL-Code（25K）用于代码生成 RL；
- 评测集：AIME24、AIME25、HMMT25(Feb)、HMMT25(Nov)（数学推理）；HumanEval+、MBPP+、LiveCodeBench (v6)（代码生成）。

**📈 对比分析**

比较方法与性能：
- 与 SFT（离线监督蒸馏）、ExPO（权重外推）以及标准 OPD 进行对比；
- 在单教师、同规模多教师和强弱蒸馏三种设置下，ExOPD 均显著优于对照方法；
- 在同规模多教师蒸馏中，ExOPD 能生成统一学生，且在所有基准上均超越各领域教师；
- 在强弱蒸馏中，ExOPD 的准确率提升幅度从 3% 到 4% 以上，明显优于标准 OPD 与 SFT。

**⚠️ 局限性**

局限性：
- Reward correction 需要额外访问教师的 pre‑RL 基础模型，导致计算成本增加；
- 当奖励缩放 λ 过大时模型可能不稳定；
- 实验仅在 Qwen3-4B/1.7B/30B 等中等规模模型上验证，未探讨更大模型的可扩展性；
- 多教师蒸馏仅在数学与代码两个领域测试，缺乏更广泛多领域验证。

---

## 208. Few-Shot Design Optimization by Exploiting Auxiliary Information

**arXiv ID:** 2602.12112 | [PDF](https://arxiv.org/pdf/2602.12112v1)

**作者:** Arjun Mani `[一作]` (Columbia University), Richard Zemel `[通讯]`

**关键词:** `Machine Learning` `Optimization` `Robotic Intelligence` `Transformer` `Bayesian Optimization` `Tabular`

**🎯 论文内容**

研究了一种利用辅助信息进行少样本设计优化的方法。

**💡 创新点**

提出在多任务设置下，利用高维辅助信息（如触觉反馈或学习曲线）通过神经网络进行上下文感知预测，从而在未见任务上实现更好的少样本预测和加速优化。

**🔧 技术方法**

使用基于 Transformer 的神经过程模型，学习辅助信息的表示并在 Bayesian Optimization 循环中作为代理模型。

**📊 数据集**

在机器人抓手设计的 4.28M 评估数据集（1000 个物体）和 LCBench 超参数调优任务（35 个分类任务）上进行实验。

**📈 对比分析**

相较于单任务 GP、DGP、多任务 GP 等基线，方法在预测 MSE、累计奖励和成功率上均显著提升，尤其在少样本情形下性能更为突出。

**⚠️ 局限性**

局限在于假设测试任务与训练任务来自同一分布，未处理极端 OOV 任务；并且未直接利用辅助信息作为目标预测，可能导致信息利用不充分。

---

## 209. Assessing Low Back Movement with Motion Tape Sensor Data Through Deep Learning

**arXiv ID:** 2602.11465 | [PDF](https://arxiv.org/pdf/2602.11465v1)

**作者:** Jared Levy `[一作]` (University of California San Diego), Emilia Farcas `[通讯]` (University of California San Diego)

**通讯引用:** 603 | [OpenAlex ID](https://openalex.org/A5018352071)

**关键词:** `Machine Learning` `Classification` `Recognition` `Convolutional Neural Network` `Recurrent Neural Network` `Transformer` `Diffusion model` `Time Series`

**🎯 论文内容**

开发了基于Motion Tape（MT）传感器的低背运动分类深度学习管线（MT-AIM），并通过生成式模型进行数据与特征增强，实现对6种低背运动的识别。

**💡 创新点**

创新点包括：①首个将MT传感器用于低背运动分类；②结合数据增强（合成MT样本）和特征增强（DTFT频域特征+预测运动学）；③使用C‑VAE/Diffusion‑TS生成器实现运动学翻译；④在有限、噪声数据下实现接近完美的分类准确率。

**🔧 技术方法**

技术手段包括：MT传感器硬件；深度生成模型（C‑VAE、Diffusion‑TS、TimeGAN）；特征提取（DTFT、运动学预测）；分类模型（XGBoost、Transformer、CNN‑LSTM）；信号预处理（Hampel滤波、Min‑Max归一化）；评估指标（Wasserstein、FTSD、准确率）。

**📊 数据集**

使用了10名健康志愿者的MT与MoCap数据，6种低背运动各3次，共180个真实MT样本，随后合成120个（每种20个）作为训练集；MT传感器6个，配合MoCap记录运动学。

**📈 对比分析**

通过交叉样本拆分与留一子拆分对比基线分类器（无增强）与MT‑AIM（数据+特征增强）。最优方案（CNN‑LSTM+MT‑AIM）平均准确率为99.4%，显著高于基线91.7%和仅特征增强的96.4%。

**⚠️ 局限性**

局限性包括：样本量小且缺乏多样性、仅健康受试者、未在低背痛人群验证、传感器漂移与位置变异影响、GNS传感器噪声、未评估实时/长期监测性能。

---

## 210. Finding the Cracks: Improving LLMs Reasoning with Paraphrastic Probing and Consistency Verification

**arXiv ID:** 2602.11361 | [PDF](https://arxiv.org/pdf/2602.11361v1)

**作者:** Weili Shi `[一作]` (University of Virginia), Sheng Li `[通讯]` (University of Virginia)

**通讯引用:** 11612 | [OpenAlex ID](https://openalex.org/A5100359839)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Prompt Engineering` `Chain-of-Thought` `Text`

**🎯 论文内容**

提出了双阶段的 Paraphrastic Probing and Consistency Verification（PPCV）框架：先用同义句探测并提取关键 token，再通过替换这些 token 并使用同义句一致性验证来选择最终答案，从而显著提升大语言模型（LLM）的推理性能。

**💡 创新点**

创新点在于：
- 用同义句生成（APE）对原问题进行多样化表述，利用模型在不同表述下对同一 token 的 logits 变化来精确定位对推理轨迹影响最大的关键 token；
- 在关键 token 位置生成 top‑K 替代 token，展开多条简化推理路径；
- 通过同义句一致性（而非传统多数投票）以及相似度加权的方式评估不同路径的可靠性，从而选取最终答案；
- 整个流程不依赖外部验证器或人工标注，完全基于模型自身的内部信号。

**🔧 技术方法**

主要技术包括：
- Automatic Prompt Engineering（APE）实现高质量多样化同义句生成；
- Token‑level logits 计算与匹配，定位关键 token；
- Top‑K 替代 token 采样与贪婪解码生成新推理路径；
- 同义句一致性评分和相似度加权一致性验证；
- 并行 roll‑out（vLLM）降低计算成本。

**📊 数据集**

使用的主要数据集：GSM8K、GSM‑Hard、Math500、SVAMP、ARC‑Challenge、AIME 2024、AIME 2025、BRUMO 2025、HMMT 2025。覆盖数学推理、常识推理与竞赛级别难度。

**📈 对比分析**

与基线方法（Chain‑of‑Thought、Self‑Consistency、Tree‑of‑Thought、Guided Decoding、Predictive Decoding、Phi‑Decoding）在 Llama‑3.1‑8B‑Instruct、Mistral‑7B‑Instruct‑v0.2 以及 Qwen‑3‑32B 上进行对比。PPCV 在 Llama‑3.1‑8B‑Instruct 上：
- GSM8K Pass@1 从 80.60% 提升至 88.24%（+7.6%）；
- Math500 从 34.00% 提升至 50.00%（+16%）；
- AIME 2024 从 30.00% 提升至 40.00%（+10%）。
在 Mistral‑7B‑Instruct‑v0.2 上也取得了约 3–4% 的绝对提升。整体来看，PPCV 在大多数基准上均优于 Self‑Consistency 与其他先进解码方法。

**⚠️ 局限性**

局限性包括：
- 需要额外的同义句生成步骤，导致推理时间和计算成本增加；
- 目前主要针对单一关键 token，未充分探索多关键 token 的联合优化；
- 对同义句质量和多样性的依赖较大，若同义句不足或过于相似可能影响关键 token 检测；
- 主要在数值/数学推理任务上验证，常识推理或对话情境下的效果尚未系统评估。

---

## 211. Structured Hybrid Mechanistic Models for Robust Estimation of Time-Dependent Intervention Outcomes

**arXiv ID:** 2602.11350 | [PDF](https://arxiv.org/pdf/2602.11350v1)

**作者:** Tomer Meir `[一作]` (Technion Israel Institute of Technology), Uri Shalit `[通讯]` (Tel Aviv University)

**关键词:** `Machine Learning` `Optimization` `Drug Discovery` `Ordinary Differential Equation` `Time Series` `Biomedical Data` `Electronic Health Records`

**🎯 论文内容**

本文研究了在动力学系统中估计干预效应的方法，提出了一种混合机制‑数据驱动模型，用以预测时间变化干预下的系统轨迹并实现干预优化；

**💡 创新点**

创新点在于将系统转移算子分解为参数化（机制）与非参数化（数据）两部分，并进一步区分干预相关与干预无关的动态，结合两阶段预训练编码器实现对未知机制参数的推断；

**🔧 技术方法**

主要技术包括Neural ODE框架、机制先验的ODE模型、MLP校正网络、编码器预训练（基于仿真数据）以及两阶段训练流程；

**📊 数据集**

实验使用的数据库包括：①周期摆杆仿真数据（用于验证机制参数推断与补偿），②Propofol三室PK合成数据（用于评估药物输注方案），③MIMIC‑IV真实患者数据（用于验证在高BMI或高年龄 OOD 情况下的剂量预测）；

**📈 对比分析**

比较方法为与纯机制模型、纯数据驱动模型的对照，评估重建MSE、干预结果预测MSE和剂量误差（MAPE），结果显示混合模型在 OOD 和极端 OOD 情况下显著优于两种单一模型；

**⚠️ 局限性**

局限性包括：需预先有机制模型并对其进行校正；编码器依赖于仿真数据，可能在真实系统中难以充分拟合；仅适用于非自适应、已知的时间变化干预，且对极端参数变化的鲁棒性仍有限。

---

## 212. Towards a Sustainable Age of Information Metric: Carbon Footprint of Real-Time Status Updates

**arXiv ID:** 2602.11946 | [PDF](https://arxiv.org/pdf/2602.11946v1)

**作者:** Shih-Kai Chou `[一作]` (Jožef Stefan Institute), Jernej Hribar `[通讯]` (Jožef Stefan Institute)

**通讯引用:** 273 | [OpenAlex ID](https://openalex.org/A5054993601)

**关键词:** `Information Theory`

**🎯 论文内容**

本文提出了在考虑碳足迹约束下的年龄信息（AoI）框架，并推导了 M/M/1 与 M/M/1* 排队模型的平均 AoI 闭式表达式；

**💡 创新点**

创新点在于将碳强度（CI）与信息新鲜度耦合，揭示了 AoI 与碳足迹之间的非线性权衡，并针对动态 CI 进一步优化 AoI；

**🔧 技术方法**

采用排队论、碳排放建模、信噪比约束与动态 CI 分析相结合的理论方法，并通过仿真验证；

**📊 数据集**

未使用公开数据集，所有结果基于基于所给参数（如 Slovenia 2024 CI 数据）进行的数值模拟；

**📈 对比分析**

通过与传统无碳约束的 AoI 最小化方案对比，证明在碳预算约束下，系统利用率需降低，AoI 与碳足迹呈 U 形关系，性能在不同 CI 与 QoS 条件下各异；

**⚠️ 局限性**

局限在于模型假设理想无丢包、固定 CI 与功率设置、未考虑多源多接入环境，未来需扩展至更复杂网络与实时自适应调度。

---

## 213. Decision Support System for Technology Opportunity Discovery: An Application of the Schwartz Theory of Basic Values

**arXiv ID:** 2602.11855 | [PDF](https://arxiv.org/pdf/2602.11855v1)

**作者:** Ayato Kitadai `[一作]` (University of Tokyo), Nariaki Nishino `[通讯]` (University of Tokyo)

**通讯引用:** 612 | [OpenAlex ID](https://openalex.org/A5079335652)

**关键词:** `Human-Computer Interaction`

**🎯 论文内容**

通过工作坊将技术功能与用户价值（Schwartz基本价值理论）相结合，构建技术机会发现（TOD）决策支持框架，并在Sony CSL四项技术上进行实证验证。

**💡 创新点**

创新点在于：①首次将技术成熟度评估指标（TRL）与人类价值体系统一映射，形成双重评估维度；②提出“愿景差距”（vision gap）与“价值宽度”（value breadth）两个可量化指标，用于判断技术商业潜力；③采用现场对照实验，比较内部专家与普通消费者的价值映射差异。

**🔧 技术方法**

使用的技术包括：技术成熟度评估（TRL），Schwartz基本价值理论映射（10种价值类型及其单值列表），功能建模（动词‑名词对），以及基于工作坊的情境描述与评分体系。

**📊 数据集**

使用的数据集为：Sony CSL内部研发的四项技术（Omoiiro、Continuator、Cybercode、Bubble Click）及其功能清单；以及19名参与者（9名普通消费者、10名技术部署专家）的工作坊反馈与评分结果。

**📈 对比分析**

通过对比专家组与消费者组在价值类型覆盖率、价值宽度数量和场景可行性（TRL）三维空间内的分布，发现成功技术呈现更广泛的价值覆盖和更高的价值宽度；失败技术则缺乏愿景差距和价值宽度。实验未给出传统机器学习指标，但通过可视化雷达图和数值统计展示了两组间差异显著。

**⚠️ 局限性**

局限性包括：①工作坊方法主观性强，功能拆解与价值选择易受访者偏差影响；②样本规模小（仅四项技术、19名参与者），缺乏统计显著性；③技术评估依赖单一专家，难以扩展；④未考虑资源约束与商业可行性等实际决策因素。

---

## 214. KuaiSearch: A Large-Scale E-Commerce Search Dataset for Recall, Ranking, and Relevance

**arXiv ID:** 2602.11518 | [PDF](https://arxiv.org/pdf/2602.11518v1)

**作者:** Yupeng Li `[一作]` (University of Science and Technology of China), Wenwu Ou `[通讯]` (Kuaishou Technology)

**关键词:** `Information Retrieval` `Retrieval` `Recommendation System` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

构建了基于快手平台真实搜索日志的最大规模电商搜索数据集KuaiSearch，涵盖检索、排序与相关性判断三大阶段，数据包含真实用户查询、自然语言商品描述、冷启动用户及长尾商品；

**💡 创新点**

首次发布完整自然语言文本且不做热门过滤的多阶段电商搜索数据集，为LLM与电商检索结合提供统一评测平台；

**🔧 技术方法**

利用传统检索方法（BM25、DocT5Query）、双编码检索（DPR-ADE/SDE）、生成式检索（DSI、LTRGR）、深度点击预测网络（DNN、Wide&Deep、DCN、DIN）以及跨编码与生成式评估模型（BGE、XLM-R、Qwen3、Llama3.2）进行基线实验；

**📊 数据集**

使用KuaiSearch（全量）及其轻量版KuaiSearch-Lite作为实验数据集；

**📈 对比分析**

在Recall任务中，双编码方法DPR‑SDE优于词匹配与生成式检索；在Ranking任务中，DIN在AUC上略胜一筹；在相关性任务中，生成式分类模型Qwen3‑1.7B在ROC‑AUC和PR‑AUC上均取得最高分；

**⚠️ 局限性**

仍受限于特征工程、模型对长尾商品和冷启动用户的适配度不足，生成式检索在Recall阶段表现不佳，且仅在轻量版数据上验证，需进一步扩大实验规模与探索更深层特征交互。

---

## 215. Benchmarking for Single Feature Attribution with Microarchitecture Cliffs

**arXiv ID:** 2602.11580 | [PDF](https://arxiv.org/pdf/2602.11580v1)

**作者:** Hao Zhen `[一作]` (State Key Lab of Processors Institute of Computing Technology Chinese Academy of Sciences), Trevor E. Carlson `[通讯]` (National University of Singapore)

**通讯引用:** 2144 | [OpenAlex ID](https://openalex.org/A5069683581)

**关键词:** `Hardware Architecture` `Optimization` `Explainability and Interpretability` `Computational Efficiency` `Tabular` `Benchmark`

**🎯 论文内容**

本文提出Microarchitecture Cliffs方法，通过生成单一微架构特征的基准，精确归因模拟器与RTL之间的性能差异并进行模型校准。

**💡 创新点**

创新点在于：①以单特征归因为目标的基准生成流程；②结合性能计数器聚类与自动化基准合成工具；③通过Cliffs实现对微架构细节（如ROB压缩、缓存银行冲突等）的可量化评估和校准。

**🔧 技术方法**

技术方法包括：性能计数器聚类（DBSCAN）、微架构瓶颈识别（Cliff‑SKP）、基准合成与压力梯度构造（Cliff‑BACT）、变更点检测与趋势线分析。

**📊 数据集**

使用的数据集涵盖：SPECint2006/2017、SPECfp2006/2017、h264ref、Verilator、BOOM等工作负载，评估不同微架构与不同规模的模拟器与RTL。

**📈 对比分析**

对比方法：在Cliff基准上将XS‑GEM5与XS‑RTL的性能误差从59.2%降低到1.4%；在SPEC2006/2017上绝对误差分别下降15.1%/21.0%；在Store Set评估中相对误差从48.9%降至0.83%；相较于未校准模型，Cliffs显著提升了模拟器对真实硬件行为的可预测性。

**⚠️ 局限性**

局限性包括：①需要手工设计与验证基准片段，仍存在人为误差；②对极端或高度耦合的微架构特征捕捉仍可能不足；③校准过程对不同处理器架构的迁移性需进一步验证；④误差降低不一定在所有工作负载上表现一致，某些基准可能仍受外部因素影响。

---

## 216. Cachemir: Fully Homomorphic Encrypted Inference of Generative Large Language Model with KV Cache

**arXiv ID:** 2602.11470 | [PDF](https://arxiv.org/pdf/2602.11470v1)

**作者:** Ye Yu `[一作]` (Columbia University), Meng Li `[通讯]` (Peking University)

**通讯引用:** 24495 | [OpenAlex ID](https://openalex.org/A5100457407)

**关键词:** `Cryptography and Security` `Generation` `Computational Efficiency` `Safty and Privacy` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

提出了一种基于 KV 缓存的全同态加密 LLM 推理框架 Cachemir，显著降低加密推理的延迟。

**💡 创新点**

创新点包括：① 设计 Interleaved Replicated Packing 高效 VMM；② 为 KV 缓存更新与计算提供专用的 HE 包装与协议；③ 改进 bootstrapping 放置策略以适应 Transformer 结构。

**🔧 技术方法**

采用 RNS‑CKKS 同态加密、SIMD 编码、Bottleneck‑free 旋转与多头注意力的专用打包、以及基于 DAG 的 bootstrapping 最优搜索。

**📊 数据集**

使用 GPT‑2‑base、TinyLlama‑1.1B 和 Llama‑3‑8B 三大模型，并在 MRPC 与 WikiText‑2 数据集上评估性能与准确率。

**📈 对比分析**

与 MOAI、THOR、NEXUS 等基线对比，Cachemir 在 CPU 上相较 MOAI 与 THOR 分别提升 48.83× 与 67.16×，在 GPU 上单词生成仅耗时 1.61 min（Llama‑3‑8B），并保持与原模型相近的准确率。

**⚠️ 局限性**

主要局限在于同态乘法对位移/旋转的高成本，KV 缓存占用的加密内存仍随上下文长度线性增长，以及对非线性层近似精度的进一步优化仍有空间。

---

## 217. Energy-Aware Spike Budgeting for Continual Learning in Spiking Neural Networks for Neuromorphic Vision

**arXiv ID:** 2602.12236 | [PDF](https://arxiv.org/pdf/2602.12236v1)

**作者:** Anika Tabassum Meem `[一作]` (University of Liberal Arts Bangladesh), Md Zesun Ahmed Mia `[通讯]` (Pennsylvania State University)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5047546816)

**关键词:** `Neural and Evolutionary Computing` `Optimization` `Computational Efficiency` `Spiking Neural Network` `Reinforcement Learning` `Image`

**🎯 论文内容**

提出了能量感知的脉冲预算框架，用于在持续学习的脉冲神经网络中同时优化准确性与能耗。

**💡 创新点**

创新点在于将脉冲预算作为比例控制器，针对帧基与事件基输入实现能量约束；引入可学习的LIF参数与经验回放；揭示不同模态下稀疏正则化与激活提升的双重行为。

**🔧 技术方法**

使用了经验回放、可学习的膜电位衰减与阈值、比例控制的脉冲调度、Fast Sigmoid surrogate gradient 训练等技术。

**📊 数据集**

使用的数据集包括 MNIST、N-MNIST、CIFAR-10、CIFAR-10-DVS 与 DVS-Gesture。

**📈 对比分析**

通过与仅使用经验回放的基线（C1）对比，在所有 5 个基准上均提升平均准确率；帧基数据显著减少脉冲率（MNIST 47%），事件基数据通过轻微增加脉冲率获得大幅准确率提升（DVS-Gesture 17.45pp）。

**⚠️ 局限性**

局限性包括回放缓存占用内存、控制器参数需手工调节、未针对任务无界或开放世界场景、尚未在真实硬件上验证能耗。

---

## 218. Fully First-Order Algorithms for Online Bilevel Optimization

**arXiv ID:** 2602.11665 | [PDF](https://arxiv.org/pdf/2602.11665v1)

**作者:** Tingkai Jia `[一作]` (East China Normal University), Cheng Chen `[通讯]` (East China Normal University)

**通讯引用:** 13131 | [OpenAlex ID](https://openalex.org/A5100420499)

**关键词:** `Machine Learning` `Optimization`

**🎯 论文内容**

本研究探讨了非凸-强凸在线双层优化（OBO），提出了一种完全一阶的OBO算法，消除了对Hessian-向量积（HVP）oracle的需求。

**💡 创新点**

创新点在于通过将原始OBO问题重构为具有不等式约束的单层在线问题，构建拉格朗日函数序列，从而避免了隐式微分所需的HVP。

**🔧 技术方法**

使用了一种完全一阶的在线双层优化算法（F^2OBO），并提出了一种改进的变体（AF^2OBO），采用自适应内迭代方案。

**📊 数据集**

论文中未具体提及使用的数据集。

**📈 对比分析**

与现有的OBO算法进行比较，F^2OBO在不需要二阶信息的情况下，能够达到O(1 + V_T + H_2,T)的后悔界限，而AF^2OBO在V_T≥O(√(T))时能够达到O(√(T) + V_T)的后悔界限。

**⚠️ 局限性**

限制在于AF^2OBO虽然消除了对内层最优解漂移的依赖，但在迭代开销上增加了成本，并且后悔保证较弱。

---

## 219. Data-driven modelling of low-dimensional dynamical structures underlying complex full-body human movement

**arXiv ID:** 2602.11492 | [PDF](https://arxiv.org/pdf/2602.11492v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Human-Computer Interaction`

---

## 220. A Dual-Branch Framework for Semantic Change Detection with Boundary and Temporal Awareness

**arXiv ID:** 2602.11466 | [PDF](https://arxiv.org/pdf/2602.11466v1)

**作者:** Yun-Cheng Li `[一作]`, Ke Li `[通讯]` (Xidian University)

**通讯引用:** 11738 | [OpenAlex ID](https://openalex.org/A5084939934)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Anomaly Detection` `Convolutional Neural Network` `Gaussian Splatting` `Image`

**🎯 论文内容**

设计了一种双分支框架DBTANet，融合SAM和ResNet34实现语义变化检测并增强边界与时间建模。

**💡 创新点**

结合冻结的Segment Anything Model提供全局语义与边界先验，与轻量级ResNet34局部细节；引入高斯平滑投影模块（GSPM）去噪浅层SAM特征；设计双向时间感知模块（BTAM）对多尺度特征进行对称时序建模。

**🔧 技术方法**

Siamese双分支编码器、Gaussian Convolution Block、Bidirectional Multi‑Scale Aggregation、ECA注意力、边界辅助任务、相似损失等技术。

**📊 数据集**

Landsat‑SCD和SECOND两个遥感变化检测基准。

**📈 对比分析**

与六种主流方法对比，在两数据集上均取得最优或相近的mIoU、SeK、F1；在Landsat‑SCD上 OA 96.92%、mIoU 90.84%、SeK 65.72%、F1 90.90%。

**⚠️ 局限性**

对极小尺度或弱变化的检测仍受限；模型对大规模数据训练依赖高性能GPU；边界辅助仍可能受SAM先验噪声影响。

---

## 221. An Improved Upper Bound for the Euclidean TSP Constant Using Band Crossovers

**arXiv ID:** 2602.11250 | [PDF](https://arxiv.org/pdf/2602.11250v1)

**作者:** Julia Gaudio `[一作]` (Northwestern University), Charlie K. Guan `[通讯]` (Northwestern University)

**通讯引用:** 299 | [OpenAlex ID](https://openalex.org/A5102833452)

**关键词:** `Computational Geometry` `Optimization` `Point Cloud`

**🎯 论文内容**

本文通过对欧氏旅行商问题(TSP)常数β的上界进行研究，提出一种跨越带（band-crossing）启发式，进一步降低了现有最优上界0.90380至0.90367，并通过蒙特卡罗实验评估了该方法在不同参数k、h下的表现，显示出在k=8时可逼近0.85的更低上界；

**💡 创新点**

创新点在于突破传统仅限于单带遍历的限制，允许在相邻带之间插入两点交叉路径，从而获得更短的巡回路径；同时提出了一套严格的解析改进证明，首次在现有tuple-optimization框架上实现可验证的上界改进；

**🔧 技术方法**

主要技术包括：①基于Poisson过程的理论建模与分区带策略；②组合优化中的tuple-permutation最优排列计算；③跨带交叉启发式的设计与实现；④蒙特卡罗仿真与集中分析（Chernoff、Hoeffding、Sub-gamma等）；⑤严谨的积分分割与Riemann求下界的分析；

**📊 数据集**

数据集为均匀随机生成的n个点（n→∞）在单位正方形中的布点；所有实验与理论均在该随机模型下进行；

**📈 对比分析**

通过与之前最优上界0.90380的对比，证明了方法在理论上实现了0.00013的改进；蒙特卡罗实验表明，在k=4时可将上界从0.884到0.868，在k=8时进一步下降至0.849；因此相较于仅使用tuple-optimization，跨带启发式在不同参数下均表现出更优的上界；

**⚠️ 局限性**

限制主要体现在：①分析中使用的三角不等式等粗略上界导致改进幅度有限；②tuple-optimization方法在高k下仍趋于0.86-0.88的拱形极限；③跨带策略虽然有效，但在实际算法实现时仍需更精细的交叉选择与路径优化，当前的证明仅给出了可证实的改进上限，而未达到理论最优值0.71。

---

## 222. FAIL: Flow Matching Adversarial Imitation Learning for Image Generation

**arXiv ID:** 2602.12155 | [PDF](https://arxiv.org/pdf/2602.12155v1)

**作者:** Yeyao Ma `[一作]` (Shanghai Jiao Tong University), Weidi Xie `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 10088 | [OpenAlex ID](https://openalex.org/A5076097168)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Reinforcement Learning` `Generative Adversarial Network` `Flow-based Model` `Ordinary Differential Equation` `Policy Gradient` `Image`

**🎯 论文内容**

提出了 Flow Matching Adversarial Imitation Learning (FAIL)，一种将图像生成模型后训练视为对抗模仿学习的框架。

**💡 创新点**

创新点在于：①不需要偏好对或奖励模型，直接使用判别器逼近专家分布；②给出了两种实现：对白盒可微 ODE 求解器的路径梯度（FAIL‑PD）和对黑盒/离散场景的策略梯度（FAIL‑PG）。

**🔧 技术方法**

采用对抗训练、路径梯度（Pathwise Derivative）、策略梯度（Policy Gradient）、Flow Matching、Flow Policy Optimization 等技术，并在判别器设计上结合 VFM、FM、VLM 等预训练模型。

**📊 数据集**

使用 13,000 条由 Gemini 3 Pro（Nano Banana pro）生成的单图专家演示作为训练数据，基准模型为 FLUX.1‑dev。

**📈 对比分析**

在 UniGen、DPG、HPDv3 等基准上与 SFT、RLHF、DPO 等方法对比，FAIL‑PD 在 13k 数据下分别提升 UniGen（从 61.61 提升至 73.70）、DPG（87.32）以及 HPDv3（整体 11.28）等指标，表现接近或超过部分闭源前沿模型。

**⚠️ 局限性**

局限性包括：①对抗训练仍然敏感于超参数与网络结构；②实验仅验证 13k 规模，无法确定对大规模数据的可扩展性；③若基础模型缺乏某些能力（如文本识别），FAIL 仅能做细化，无法弥补根本缺失。

---

## 223. Adjusted Winner: from Splitting to Selling

**arXiv ID:** 2602.12231 | [PDF](https://arxiv.org/pdf/2602.12231v1)

**作者:** Robert Bredereck `[一作]` (TU Clausthal), Nimrod Talmon `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 1643 | [OpenAlex ID](https://openalex.org/A5064260879)

**关键词:** `Computer Science and Game Theory` `Optimization` `Tabular`

**🎯 论文内容**

本文提出在两位代理人之间公平分配不可分资源的新框架——DSIRS，核心思路是用出售资源替代传统的分割，从而实现可行且公平的分配。

**💡 创新点**

创新点在于：①将 Adjusted Winner（AW）方法与资源销售相结合，构造了 AWNS（Adjusted Winner without Splitting）问题；②对该问题族进行复杂性分析，证明大多数变体为弱 NP‑难并不可近似；③为其中最有意义的 AWNS‑ρ 设计了一个全多项式时间近似方案（FPTAS）。

**🔧 技术方法**

使用技术包括：改进的 AW 过程、组合优化、动态规划（带状态压缩与量化）、多层缩放与误差控制，最终实现了 FPTAS；同时利用离散化技巧将问题转化为 0/1 背包并复用其 FPTAS。

**📊 数据集**

实验数据取自 Spliddit 平台，随机抽取 5000 个两人子问题（4–15 个物品），利用真实效用矩阵和基于全体参与者的价格估算进行评估。

**📈 对比分析**

比较方法：用均衡度指标 ρ（福利比）和 d（福利差）评价结果；实验显示随着预算增加，平均 ρ 明显下降、d 明显缩小；在不同成本/价格模式下，平均成本+平均价格模式表现最佳；FPTAS 在 ε=0.1 下求解时间可接受，近似误差在预期范围内。

**⚠️ 局限性**

限制：①算法仅适用于两人情形，未对多方扩展；②虽然实现了 FPTAS，但在大规模实例中状态空间仍较大；③无法同时满足所有公平准则（如公平、无怨恨、Pareto 最优），需在这些目标间做权衡；④对价格与成本的设定假设较理想，实际中可能更复杂。

---

## 224. Projected Representation Conditioning for High-fidelity Novel View Synthesis

**arXiv ID:** 2602.12003 | [PDF](https://arxiv.org/pdf/2602.12003v1)

**作者:** Min-Seop Kwak `[一作]`, Seungryong Kim `[通讯]` (KAIST)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Diffusion model` `Image`

**🎯 论文内容**

提出基于扩散模型的ReNoV框架，利用外部视觉表示（如VGGT）进行投影条件，以提升新视角合成的几何一致性和补全质量。

**💡 创新点**

创新点在于将多视角一致的外部特征投影到目标视角作为条件，并将几何信息与语义信息融合到扩散模型的U‑Net中，从而实现无姿势优化的高保真新视角生成。

**🔧 技术方法**

使用扩散模型（Stable Diffusion 2.1）、投影条件模块、外部表示提取器（VGGT/DA3/DINOv2）、姿态/点云预测网络以及位置编码进行条件编码。

**📊 数据集**

在RealEstate10K、DTU、Co3D、MVImgNet等多视角数据集上进行训练和评估。

**📈 对比分析**

与PixelSplat、MVSplat、NoPoSplat、FLARE、LVSM、ViewCrafter、LucidDreamer等方法对比，ReNoV在PSNR、SSIM和LPIPS上均取得领先或接近最佳性能，尤其在远视角插补和无姿势设置下表现突出。

**⚠️ 局限性**

限制包括对外部表示的依赖（若特征质量下降会影响结果）、需要较多参考图像才能充分利用投影信息、处理速度相对较慢且尚未实现实时性能。

---

## 225. Improving Neural Retrieval with Attribution-Guided Query Rewriting

**arXiv ID:** 2602.11841 | [PDF](https://arxiv.org/pdf/2602.11841v1)

**作者:** Moncef Garouani `[一作]` (IRIT UMR5505 CNRS Université Toulouse Capitole), Josiane Mothe `[通讯]` (IRIT UMR5505 CNRS Université de Toulouse)

**关键词:** `Information Retrieval` `Retrieval` `Transformer` `Large Language Model` `Prompt Engineering` `Text` `Benchmark`

**🎯 论文内容**

提出一种基于神经检索器的 token‑level attribution 引导的查询改写方法，利用检索器对每个 token 的贡献信息作为 soft 指导，让 LLM 在保持用户意图的前提下细化模糊或误导性词汇，从而提升检索效果。

**💡 创新点**

创新点在于将检索器的解释（token attribution）与 LLM 的查询改写闭环结合：通过 attribution 软引导而非硬筛选来控制改写，且不需要额外训练数据或 retriever 重新训练，实现无侵入式在线改写。

**🔧 技术方法**

核心技术包括：使用 Integrated Gradients 对检索器的 relevance score 进行 token attribution；基于 prompt 的 Mistral 7B LLM 进行查询改写；在 BEIR benchmark 上对 SPLADE 与 TCT‑ColBERT 进行实验评估。

**📊 数据集**

使用 BEIR 的多个集合（如 SciFact、FiQA‑2018、NFCorpus 等）进行实验，涵盖开放域与领域特定的检索任务。

**📈 对比分析**

与原始查询、仅保留高 attribution token、仅 LLM 改写等 baseline 进行对比，评估指标为 nDCG@k、MAP@k、Precision@k。实验表明 attribution‑guided 改写在所有检索器和 cutoff 上持续优于 baseline，提升幅度可达 5–15%（SPLADE）或 10–16%（TCT‑ColBERT）等。

**⚠️ 局限性**

限制包括：仅实现单轮改写；改写效果依赖于检索器返回的 top‑k 文档质量；对极短或极长查询、跨语言场景以及更复杂的多轮改写的适用性尚未深入探究；对不同检索器的通用性需进一步验证。

---

## 226. Adaptive Power Iteration Method for Differentially Private PCA

**arXiv ID:** 2602.11454 | [PDF](https://arxiv.org/pdf/2602.11454v1)

**作者:** Ta Duy Nguyem `[一作]`, Huy Le Nguyen `[通讯]`

**关键词:** `Data Structures and Algorithms` `Safty and Privacy` `Optimization` `Gaussian Mechanism`

**🎯 论文内容**

提出一种自适应滤波的差分隐私奇异向量（PCA）算法，基于迭代幂方法实现行级(ε,δ)-DP；

**💡 创新点**

创新点在于不需要预先知道矩阵的低相干性参数，而是通过稀疏向量技术在每一步自适应地过滤大幅度影响的行，从而在低相干矩阵上实现超越最坏情况的误差上界；

**🔧 技术方法**

使用了高斯机制、稀疏向量（AboveThreshold）技术、幂迭代、矩阵 Bernstein 与 Wedin 正弦定理、以及对滤波后的矩阵进行新的递推分析；

**📊 数据集**

主要针对理论分析，验证了在随机高斯样本（0,Σ²）下的性能；实验部分未给出具体数据集；

**📈 对比分析**

与Hardt‑Roth等先前行级隐私PCA方法相比，算法在低相干矩阵和高维Gaussian数据上实现了更低的误差（相较于最坏情况降低到O(1/ϵ²n²σ₁²κ²)等），同时在样本复杂度和迭代次数上优于把每次迭代视为一次全局更新的方法；

**⚠️ 局限性**

局限性包括：需要行长度上界（‖a‖₂≤1）且对迭代次数敏感；对极小特征值间隙κ的估计仍需预设或多次尝试；在非常稀疏或相干的实际数据中，过滤阈值可能导致信息损失；此外，算法在大规模数据时计算量和内存开销较传统PCA更高。

---

## 227. LawThinker: A Deep Research Legal Agent in Dynamic Environments

**arXiv ID:** 2602.12056 | [PDF](https://arxiv.org/pdf/2602.12056v1)

**作者:** Xinyu Yang `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**通讯引用:** 3927 | [OpenAlex ID](https://openalex.org/A5010558184)

**关键词:** `Artificial Intelligence` `Retrieval` `Optimization` `Transformer` `Large Language Model` `Agentic AI` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

本文开发了 LawThinker，一种专为动态司法场景设计的自治法律研究代理，采用 Explore‑Verify‑Memorize（探索‑验证‑记忆）策略，实现了对检索信息的即时验证和长期记忆管理。

**💡 创新点**

创新点包括：①引入 DeepVerifier 模块，对每一次探索结果按知识准确性、事实-法相关性和程序合规性三维度进行深度验证；②在系统层面强制执行探索‑验证循环，防止错误信息在推理链中累积；③设计 15 个专用工具覆盖探索、验证和记忆三大功能；④在多种静态与动态法律基准上验证了方法的普适性。

**🔧 技术方法**

技术手段主要包括：使用大语言模型（如 Qwen3‑32B）驱动代理；结合稠密检索、SAILER 等检索工具获取法律条文、案例和程序；DeepVerifier 采用基于权威数据库的“法律条文内容核对”与“事实‑法关联检查”相结合的混合验证策略；系统控制器在每一步检索后触发 DeepVerifier；内置记忆模块区分法律知识记忆与案件上下文记忆，支持多轮对话中的知识复用。

**📊 数据集**

使用的数据集：动态司法基准 J1‑EVAL（508 个真实案例，覆盖 6 种司法场景），以及三大静态中文法律基准 LawBench、LexEval、UniLaw‑R1‑Eval；此外内部构建了包含 55,347 条法律条文、346 条犯罪定罪条目和中国裁判文书数据库的法律知识库。

**📈 对比分析**

方法比较：对比直接推理（多种通用与法律专用 LLM）、工作流方法（ReAct、Plan‑and‑Solve、Plan‑and‑Execute）以及在推理中直接使用检索工具的 Search‑o1。LawThinker 在 J1‑EVAL 上比直接推理提升约 24%，比工作流方法提升 11%，并在过程指标（如格式跟随、程序跟随）上显著领先；在静态基准上平均提升约 6% 以上，成为目前最佳方案。

**⚠️ 局限性**

局限性：仍受大模型规模限制，对检索质量高度依赖；在极长上下文或复杂多步推理中可能面临记忆溢出；主要在中文法律语境验证，跨语言迁移尚未充分评估；部分验证工具对非常细粒度的程序细节仍有漏判风险。

---

## 228. DMAP: A Distribution Map for Text

**arXiv ID:** 2602.11871 | [PDF](https://arxiv.org/pdf/2602.11871v1)

**作者:** Tom Kempton `[一作]` (University of Manchester), Stuart Burrell `[通讯]` (Visa Inc)

**关键词:** `Computation and Language` `Large Language Model` `Text`

**🎯 论文内容**

提出了 DMAP 方法，将文本通过语言模型映射为区间内的分布样本，实现统一、可视化的文本统计表示；

**💡 创新点**

创新点在于将概率积分变换与熵加权相结合，得到模型无关、上下文化的分布映射，并通过卡方检验实现统计验证；

**🔧 技术方法**

主要技术包括语言模型概率分布、区间采样、熵加权、概率积分变换、直方图可视化与卡方检验；

**📊 数据集**

使用的公开数据集包括 OPT‑125m、Mistral‑7B、Llama‑3.1‑8B、Pythia、OASST2、XSum、SQuAD、WritingPrompts 等；

**📈 对比分析**

通过 DMAP 直方图与卡方 p 值验证生成参数、比较概率曲率检测器在不同采样（如 pure sampling、top‑k、top‑p、temperature）下的 AUROC，结果表明在纯采样下传统检测器失效，验证了方法的有效性；

**⚠️ 局限性**

局限性包括对模型概率分布的依赖、随机采样带来的噪声、对大样本统计的需求、对长文本累积误差的处理不足，以及尚未探索更高级的统计指标与跨模型泛化能力。

---

## 229. From Instruction to Output: The Role of Prompting in Modern NLG

**arXiv ID:** 2602.11179 | [PDF](https://arxiv.org/pdf/2602.11179v1)

**作者:** Munazza Zaib `[一作]` (Monash University), Elaf Alhazmi `[通讯]`

**关键词:** `Computation and Language` `Generation` `Optimization` `Transformer` `Large Language Model` `Prompt Engineering` `Text` `Review/Survey Paper`

**🎯 论文内容**

本文综述了提示工程在大型语言模型中的应用，提出了系统的分类法、决策框架，并讨论了评价方法与优化策略。

**💡 创新点**

创新点在于构建了面向自然语言生成的提示工程分类体系和一体化设计–优化–评估框架，突出提示级控制与传统微调、解码控制的对比。

**🔧 技术方法**

采用文献综述与对比分析技术，归纳了提示范式、自动化提示搜索、软提示调优等方法，并评估了人类、自动化和LLM-as-a-Judge三种评估范式。

**📊 数据集**

本文未进行新的实验，主要引用公开数据集（如GLUE、SuperGLUE、SQuAD 等）和模型（GPT、BERT 等）进行案例分析。

**📈 对比分析**

通过对比提示级控制与微调、解码控制以及多种评估指标，展示了提示工程在生成质量、可控性和效率方面的优势，但具体性能数值取决于任务和模型。

**⚠️ 局限性**

限制包括缺乏统一实验验证、提示的鲁棒性和可解释性不足，以及对 LLM-as-a-Judge 可能产生偏差的担忧。

---

## 230. Benchmarking Vision-Language Models for French PDF-to-Markdown Conversion

**arXiv ID:** 2602.11960 | [PDF](https://arxiv.org/pdf/2602.11960v1)

**作者:** Bruno Rigal `[一作]` (Probayes), Nicolas Mery `[通讯]` (OpenValue)

**关键词:** `Computer Vision and Pattern Recognition` `Image Translation` `Recognition` `Data-Centric Learning` `Benchmark` `Transformer` `Vision Language Model` `Text` `Multimodality`

**🎯 论文内容**

本论文构建了一个面向法语PDF到Markdown转换的基准测试，采用模型争议采样挑选难度高的页面，设计单元测试评估语义完整性和结构一致性，并在15个视觉-语言模型上进行统一管道评测。

**💡 创新点**

创新点包括：1）针对法语且包含手写、表单、密集表格等真实难题的基准数据集；2）单元测试+类别专属归一化的评估方法，避免对格式化差异过度惩罚；3）对模型争议度的采样策略，聚焦最具挑战性的案例。

**🔧 技术方法**

技术上使用统一的VLM转换库（vlmparse），并结合多种公开和专有VLM/OCR模型（Gemini 3 Pro/Flash、Chandra、Dots.ocr、LightOnOCR、OlmOCR、PaddleOCR-VL、DeepSeek-OCR等），通过并行推理与温度调节等方式进行评测。

**📊 数据集**

数据集基于约6万份法语PDF（CCPDF、Gallica）通过两模型（dots-ocr、MinerU2.5）输出差异挑选出难度高的页面，划分为手写、表单、多列、长表格、细小文字、图形等类别。

**📈 对比分析**

评估指标为单元测试通过率，整体最高为Gemini 3 Pro（0.76），Gemini 3 Flash（0.74），最优开源模型Chandra（0.66）。手写与表单类对比最显著，专有模型显著优于开源；在多列和长表格上也取得高分。吞吐量方面，模型越大速度越慢（如Chandra 4.3s/页）。

**⚠️ 局限性**

局限性包括：仅聚焦法语文档，难以推广到非拉丁语；基准受模型推理实现与硬件影响；采样方法可能与实际应用不完全一致；部分数据源可能与模型预训练重叠，导致性能偏高。

---

## 231. Tiny Recursive Reasoning with Mamba-2 Attention Hybrid

**arXiv ID:** 2602.12078 | [PDF](https://arxiv.org/pdf/2602.12078v1)

**作者:** Wenlong Wang `[一作]` (Intercom), Fergal Reid `[通讯]` (Intercom)

**关键词:** `Artificial Intelligence` `Transformer` `Reinforcement Learning` `Graph`

**🎯 论文内容**

在TRM递归推理框架中，用Mamba‑2+attention混合算子替换原有Transformer块，保持参数量相近，验证其在隐藏空间递归中的可行性与性能提升。

**💡 创新点**

首次将SSM型Mamba‑2混合算子引入递归推理操作，既保持模型规模不变，又显著提升候选覆盖率，展示了递归与状态空间模型的兼容性与互补优势。

**🔧 技术方法**

使用Mamba‑2状态空间模型、注意力层、MLP以及Post‑Norm RMSNorm的混合块；训练时保持递归循环结构，采用ARC‑AGI等自定义采样策略。

**📊 数据集**

ARC‑AGI‑1（抽象推理图形），Sudoku‑Extreme（9×9约束满足）和Maze‑30×30‑Hard（路径搜索）。

**📈 对比分析**

与原TRM‑attn基线对比，TR‑mamba2attn在ARC‑AGI‑1上pass@2提升2.0%（45.88% vs 43.88%），pass@100提升4.75%；在Sudoku上MLP‑t变体优于attention；在Maze上混合模型显著高于attention。

**⚠️ 局限性**

Mamba‑2的单向递归特性导致在大规模空间任务中难以充分捕获全局依赖，MLP‑t在Maze任务完全失败；模型训练波动大，对不同难度层级的细粒度分析仍有限。

---

## 232. Visual Reasoning Benchmark: Evaluating Multimodal LLMs on Classroom-Authentic Visual Problems from Primary Education

**arXiv ID:** 2602.12196 | [PDF](https://arxiv.org/pdf/2602.12196v1)

**作者:** Mohamed Huti `[一作]` (Fab AI), Oliver G. B. Garrod `[通讯]` (Fab AI)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Prompt Engineering` `Multimodality` `Image` `Text` `Benchmark`

**🎯 论文内容**

构建并评估了基于 Zambia 与 India 小学考试的 Visual Reasoning Benchmark (VRB)，用于测量多模态大型语言模型在真实课堂视觉推理任务上的表现。

**💡 创新点**

首次提供规模化、课堂真实、极简文本的视觉推理基准，聚焦 LMIC 环境，并揭示模型在动态空间变换任务上的“空间天花板”。

**🔧 技术方法**

采用多模态 LLM 评估、最小化文本提示、图片提取与标注、模型性能对比与成本价值分析等技术。

**📊 数据集**

使用来自 Zambia 国立小学考试和 India JNVST 小学六年级选择测试的 701 道多项选择视觉推理题。

**📈 对比分析**

通过项级准确率（含 95% 自助置信区间）比较 45 个模型，准确率从 23% 至 78% 不等，最佳为 Gemini‑3.0 Flash 78%，同时评估成本与性能的价值前沿。

**⚠️ 局限性**

局限于高阶适应考试样本、单轮交互、未对模型推理过程做解释评估，且对真实课堂多轮对话与错误修正能力未做测试。

---

## 233. GR-Diffusion: 3D Gaussian Representation Meets Diffusion in Whole-Body PET Reconstruction

**arXiv ID:** 2602.11653 | [PDF](https://arxiv.org/pdf/2602.11653v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition`

---

## 234. LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion

**arXiv ID:** 2602.12215 | [PDF](https://arxiv.org/pdf/2602.12215v1)

**作者:** Jiangran Lyu `[一作]` (Peking University), He Wang `[通讯]` (Peking University)

**通讯引用:** 11596 | [OpenAlex ID](https://openalex.org/A5100351651)

**关键词:** `Robotics` `Robotic Intelligence` `Transformer` `Diffusion model` `Multimodality` `Video`

**🎯 论文内容**

提出了一种通过统一嵌入异构体感化数据来训练大规模（1B参数）机器人基础模型LDA-1B，能够同时学习策略、动力学和视觉预测。

**💡 创新点**

创新点在于：①将异构体感化数据按质量分配不同角色，实现通用数据摄取；②在结构化的DINO潜在空间中进行动力学预测；③使用多模态扩散Transformer实现视觉与动作异步对齐。

**🔧 技术方法**

使用的技术包括：多模态扩散Transformer（MM‑DiT）、DINO预训练视觉编码器、任务嵌入与注册令牌、流匹配损失、语义条件化语言提示。

**📊 数据集**

使用的数据集是EI‑30k，包含超过30k小时的机器人与人类轨迹（含动作标注和无动作视频），统一格式为LeRobot。

**📈 对比分析**

在RoboCasa‑GR1模拟基准和多种真实机器人任务上与π_0.5、GR00T等强基线对比，LDA‑1B在接触丰富、灵巧和长程任务上分别提升21%、48%和23%，且在混合质量微调下实现10%的数据效率提升。

**⚠️ 局限性**

局限性包括：依赖固定的DINO视觉特征，主要使用头戴摄像头的单视角；对非视觉或多模态输入的泛化受限；需要进一步探索视觉表示与动力学的联合学习。

---

## 235. Detecting Overflow in Compressed Token Representations for Retrieval-Augmented Generation

**arXiv ID:** 2602.12235 | [PDF](https://arxiv.org/pdf/2602.12235v1)

**作者:** Julia Belikova `[一作]` (Sber AI Lab), Alexander Panchenko `[通讯]` (Skoltech)

**关键词:** `Computation and Language` `Retrieval` `Compression` `Transformer` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

研究了在软压缩架构下，压缩后令模型无法回答查询的“token overflow”现象，并提出了检测方法；

**💡 创新点**

提出了从无查询依赖的饱和统计到包含查询信息的学习式探针的层级检测框架，证明压缩后即可检测overflow且不需完整LLM推理；

**🔧 技术方法**

使用饱和统计（Hoyer、频谱熵、峰度）、注意力特征、以及联合查询-上下文表示的线性/MLP探针；

**📊 数据集**

在xRAG-7B + Mistral检索模型上，对SQuADv2、TriviaQA和HotpotQA三套问答数据集进行实验；

**📈 对比分析**

与仅基于上下文特征、仅饱和统计的基线相比，联合表示探针在预压缩阶段即可达到0.72的平均ROC‑AUC，性能与后推理阶段相当，表明压缩阶段即出现overflow信号；

**⚠️ 局限性**

局限在于仅评估xRAG架构，压缩比例相对保守，且检测准确率仍有提升空间，未考虑更长文本或其他压缩技术。

---

## 236. LeafFit: Plant Assets Creation from 3D Gaussian Splatting

**arXiv ID:** 2602.11577 | [PDF](https://arxiv.org/pdf/2602.11577v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Graphics`

---

## 237. Meta-Sel: Efficient Demonstration Selection for In-Context Learning via Supervised Meta-Learning

**arXiv ID:** 2602.12123 | [PDF](https://arxiv.org/pdf/2602.12123v1)

**作者:** Xubin Wang `[一作]` (Beijing Normal University), Weijia Jia `[通讯]` (Beijing Normal University)

**通讯引用:** 11810 | [OpenAlex ID](https://openalex.org/A5051803761)

**关键词:** `Machine Learning` `Classification` `Meta Learning` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

提出 Meta-Sel，一种基于监督元学习的演示选择框架，利用查询-候选对的标签一致性训练轻量级逻辑回归评分器，并在意图分类的 ICL 中按评分从候选池中挑选 top‑k 演示。

**💡 创新点**

创新点：将示例选择建模为查询-候选对的二分类任务，使用标签一致性作为可验证的监督信号；构造的 Meta‑数据集和低维特征（TF‑IDF 余弦相似度 + 长度比）实现一次向量化评分、确定性排序和可解释权重，完全摆脱在线 LLM 调用和复杂的探索策略。

**🔧 技术方法**

技术：监督元学习、逻辑回归、TF‑IDF 余弦相似度、长度兼容度特征、Meta‑数据集采样、实验对比分析。

**📊 数据集**

数据集：四个意图分类基准（BANKING77、CLINC150、HWU64、LIU54）以及五个开源 LLM（GPT‑OSS‑20B、Gemma3‑4B、Qwen3‑8B、DeepSeek‑R1‑14B、Llama2‑7B）。

**📈 对比分析**

比较方法：对 12 种示例选择/提示工程/RL/信息论/影响力等基线进行系统评测，共 20 个模型-数据集组合。Meta‑Sel 在 19/20 组合中名列前 3，尤其在小模型上提升显著；在大模型、标签空间细粒度场景时效果略逊。

**⚠️ 局限性**

局限性：仅针对分类任务，依赖 TF‑IDF 余弦相似度对语义细粒度差异捕捉有限；不直接适用于生成任务；在已经具备强大分类能力的巨大模型中，可能无法进一步提升。

---

## 238. PuYun-LDM: A Latent Diffusion Model for High-Resolution Ensemble Weather Forecasts

**arXiv ID:** 2602.11807 | [PDF](https://arxiv.org/pdf/2602.11807v1)

**作者:** Lianjun Wu `[一作]` (KunByte AI), Bin Wang `[通讯]` (Zhejiang University)

**通讯引用:** 51527 | [OpenAlex ID](https://openalex.org/A5100372375)

**关键词:** `Artificial Intelligence` `Generation` `Data Synthesis` `Convolutional Neural Network` `Transformer` `Diffusion model` `Auto Encoder` `Time Series` `Sequential`

**🎯 论文内容**

提出了PuYun-LDM，一个结合3D-MAE和VA-MFM的潜在扩散框架，用于中期高分辨率（≤0.25°）全球天气预报，能够以单GPU 6小时间隔生成15天的自动回归序列；

**💡 创新点**

创新点包括①在VAE预训练中引入3D-MAE，利用时间遮蔽捕获天气状态演化特征作为条件，②设计Variable-Aware Masked Frequency Modeling（VA-MFM），对多变量谱特性自适应阈值，平衡频率正则化强度，③将两者结合显著提升高维潜在空间的可扩散性，突破传统LDM在高分辨率天气预报中的性能瓶颈；

**🔧 技术方法**

技术手段涵盖深度卷积自编码器（DC-AE）构建VAE，因果3D卷积的3D-MAE，频域分析与自适应低通滤波的VA-MFM，潜在扩散模型与DiT Transformer的自回归去噪，以及EDM式噪声预处理和采样；

**📊 数据集**

使用ERA5重分析数据，0.25° 6小时分辨率，1979-2019年，包含69个气象变量（13层大气场+地表场）；

**📈 对比分析**

与ECMWF ENS（基于HRES-fc0的初始场）进行对比，评估指标包括RMSE、CRPS、SSR、Rank；在短期（如第1步MSL、Z500等）PuYun-LDM显著低于ENS的RMSE/CRPS，SSR更接近1；在更长时段性能趋于相当；热带气旋轨迹误差也显示PuYun-LDM在3/9天提前预测时陆地登陆误差更小；

**⚠️ 局限性**

局限性包括：长时段预测仍与ENS相当，难以进一步提升；频率阈值的自适应仍基于经验比例，可能不适用于所有变量；模型训练与推理仍需多GPU，计算成本较高；缺乏气象领域的通用基础模型限制了潜在空间进一步优化；

---

## 239. KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

**arXiv ID:** 2602.11184 | [PDF](https://arxiv.org/pdf/2602.11184v1)

**作者:** Zukang Xu `[一作]` (Houmo AI), Dawei Yang `[通讯]` (Houmo AI)

**关键词:** `Machine Learning` `Large Language Model` `Mixture of Experts`

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

## 240. Implications of AI Involvement for Trust in Expert Advisory Workflows Under Epistemic Dependence

**arXiv ID:** 2602.11522 | [PDF](https://arxiv.org/pdf/2602.11522v1)

**作者:** Dennis Kim `[一作]` (Colorado State University), Sarath Sreedharan `[通讯]` (Colorado State University)

**通讯引用:** 1494 | [OpenAlex ID](https://openalex.org/A5028325441)

**关键词:** `Human-Computer Interaction`

**🎯 论文内容**

该研究通过一项包含77名受试者的在线实验，模拟了学术辅导情境，比较了无AI、主动AI协助和被动AI监督三种人工与AI合作模式下用户对专家与AI的信任感受。

**💡 创新点**

创新点在于将AI介入与专家表现整合为完整的工作流程，而非单一因素对比；同时使用多维度信任量表（METI、Riedl等）解析专家、AI及其组合的信任评估。

**🔧 技术方法**

技术手段包括：基于Web的交互式模拟界面、实验条件随机分配、问卷调查、以及统计分析工具（ANOVA + Tukey HSD）。

**📊 数据集**

数据集为实验生成的5种工作流程的脚本与受试者的问卷回答，构成了“人工与AI交互日志+信任量表数据”。

**📈 对比分析**

比较方法采用单因素ANOVA检验条件差异，若显著则进行Tukey事后检验；结果显示错误情境显著降低专家专业度与复用意愿，主动AI介入在错误情境下更易削弱信任，效果量介于中等（η²≈0.13‑0.16）。

**⚠️ 局限性**

局限性包括：仅在模拟环境下进行的单向设计，缺乏真实学术决策的高风险与长期关系；未测量受试者的先前经验或对AI的偏好，且样本规模有限，可能影响结论的泛化性。

---

## 241. IncompeBench: A Permissively Licensed, Fine-Grained Benchmark for Music Information Retrieval

**arXiv ID:** 2602.11941 | [PDF](https://arxiv.org/pdf/2602.11941v1)

**作者:** Benjamin Clavié `[一作]` (Mixedbread AI and National Institute of Informatics), Makoto P. Kato `[通讯]` (University of Tsukuba and National Institute of Informatics)

**关键词:** `Information Retrieval` `Retrieval` `Large Language Model` `Audio` `Multimodality` `Benchmark`

**🎯 论文内容**

构建了一个可公开的音乐检索基准IncompeBench，包含1,574条高质量乐曲、500条多样化查询以及超过125k的细粒度相关性标注。

**💡 创新点**

通过多阶段自动化流水线结合LLM生成歌单卡、查询及候选检索，提供多层级（0–3）相关性标注，并发布严格与宽松两种评估版本，填补了高质量可复现音乐检索基准的空缺。

**🔧 技术方法**

使用前沿多模态模型（Gemini 3 Pro、Qwen3、CLAMP3、TTMR++、CLAP、ColQwen‑Omni）进行查询生成、候选检索与相关性评分，并通过DSPy脚本实现端到端自动化。

**📊 数据集**

基于许可开放的IncompeTech音乐库（2,000+曲目）筛选出的1,574段30秒片段，并利用生成的歌单卡来产生500条查询。

**📈 对比分析**

对四个公开模型（CLAP、TTMR++、CLAMP3、ColQwen‑Omni）在Strict（2–3相关）与Lenient（1–3相关）设置下分别计算nDCG、MAP、Recall、Precision，结果显示所有模型在Lenient下表现良好但在Strict下仍显不足，表明Fine‑grained检索仍有改进空间。

**⚠️ 局限性**

仅包含Kevin MacLeod的器乐曲，缺少人声和多作者多样性；标注不覆盖所有查询–曲目对，存在潜在假负风险。

---

## 242. Latent Forcing: Reordering the Diffusion Trajectory for Pixel-Space Image Generation

**arXiv ID:** 2602.11401 | [PDF](https://arxiv.org/pdf/2602.11401v1)

**作者:** Alan Baade `[一作]` (Stanford University), Li Fei-Fei `[通讯]` (Stanford University)

**通讯引用:** 214826 | [OpenAlex ID](https://openalex.org/A5100450462)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Image`

**🎯 论文内容**

提出Latent Forcing方法，在像素空间与潜在表示共存时通过多时间变量重新排序扩散轨迹，实现端到端像素生成；

**💡 创新点**

核心创新在于将潜在先解码为“scratchpad”，随后再生成像素，并通过多时间调度证明生成顺序比REPA蒸馏更关键；

**🔧 技术方法**

利用扩散变压器DiT、JiT的x‑prediction、双时间变量、输出专家层、自动/分类器无引导等技术；

**📊 数据集**

在ImageNet 256×256（及64×64子集）上进行训练和评估；

**📈 对比分析**

与现有像素扩散模型JiT及其REPA增强版对比，Unguided FID‑50K降至9.76（有引导4.18），实现SOTA；

**⚠️ 局限性**

局限在于多时间调度与噪声调参敏感、训练/推理成本高，以及在更高分辨率或复杂任务上的进一步验证待完成。

---

## 243. Improving the Robustness of Large Language Models for Code Tasks via Fine-tuning with Perturbed Data

**arXiv ID:** 2602.11411 | [PDF](https://arxiv.org/pdf/2602.11411v1)

**作者:** Yang Liu `[一作]` (Polytechnique Montreal), Foutse Khomh `[通讯]` (Polytechnique Montreal)

**关键词:** `Software Engineering` `AI Code Assistant` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

通过在LLM4Code模型上使用不同级别（字符、词、句子）和比例的扰动数据进行微调，提升其对恶意或无意输入噪声的鲁棒性；

**💡 创新点**

提出了“扰动感知微调”策略，系统评估了不同扰动类型、比例和数据规模对模型鲁棒性和性能的影响；

**🔧 技术方法**

采用SafeCoder指令微调框架、LoRA与全参数微调相结合的技术，并利用字符/词/句子级扰动生成器；

**📊 数据集**

使用HumanEval与MBPP两大通用代码生成基准，并对其自然语言说明进行人工和自动化扰动扩充；

**📈 对比分析**

通过Pass@1和相对退化(RD)指标对比基线模型、未扰动微调模型和扰动微调模型，发现扰动微调能将RD降低约8‑12%，但在最优比例下Pass@1仅略降1‑3%；

**⚠️ 局限性**

实验仅覆盖Python、有限模型规模，未探索代码层面的扰动、不同语言、或更大模型的泛化；

---

## 244. Best of Both Worlds: Multimodal Reasoning and Generation via Unified Discrete Flow Matching

**arXiv ID:** 2602.12221 | [PDF](https://arxiv.org/pdf/2602.12221v1)

**作者:** Onkar Susladkar `[一作]` (University of Illinois Urbana-Champaign), Ismini Lourentzou `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 973 | [OpenAlex ID](https://openalex.org/A5043962698)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Vision Language Model` `Diffusion model` `Multimodality` `Text` `Image` `Benchmark`

**🎯 论文内容**

提出了一种统一的离散流匹配框架 UniDFlow，能够在同一模型中完成多模态理解、文本生成、图像生成和指令驱动编辑。

**💡 创新点**

核心创新点包括：1) 用低秩适配器分离理解与生成的参数，避免目标干扰；2) 引入时间导向 RMSNorm，稳定预训练 Transformer 在扩散时间步上的调优；3) 采用参考基多模态偏好对齐（mRef‑DPO）在相同条件下学习相对优先级，提升编辑的忠实度与可控性。

**🔧 技术方法**

技术方法主要包括预训练视觉‑语言 Transformer、LoRA 轻量级适配器、离散流匹配（DFM）训练目标、时间导向 RMSNorm、动态路由器（MoRA）以及参考基偏好学习（mRef‑DPO）。

**📊 数据集**

使用的数据集涵盖 MMInstruct、Text‑to‑Image‑4M、3.5M 参考偏好样本；在基准评估中使用 EvalVLM、MME‑P/S、MMBench、MathVista、GenEval、DPGBench、ImgEdit Bench、Emu‑Edit、GEdit‑Bench‑EN 等。

**📈 对比分析**

通过与 EMMA、BAGEL、Muddit 等统一或混合模型对比，UniDFlow 在 8 大基准上实现了 SOTA 表现，提升幅度在 6%–13% 之间，甚至对比更大参数模型可提升 24%；在 4B 规模下已能与 7B+ 大模型竞争，参数效率显著。

**⚠️ 局限性**

局限性包括：仍受预训练视觉‑语言骨干的偏差影响；偏好对齐需要大量高质量对比数据，数据稀缺或噪声可能导致对齐不稳定；在极端细粒度或复杂推理场景下，模型可能尚未达到人类级的细致理解；安全与偏见问题仍需进一步治理。

---

## 245. The Balanced Up-Down Walk

**arXiv ID:** 2602.11993 | [PDF](https://arxiv.org/pdf/2602.11993v1)

**作者:** Hugo A. Akitaya `[一作]` (University of Massachusetts Lowell), Jamie Tucker-Foltz `[通讯]` (Yale School of Management)

**关键词:** `Discrete Mathematics` `Graph`

**🎯 论文内容**

提出并研究了一种新的马尔科夫链——Balanced Up-Down (BUD) walk，用于在保证各区人口平衡的前提下高效采样政治划分图的随机分区。

**💡 创新点**

创新点在于：①将 Up‑Down walk 的思路与划分平衡约束结合，得到 BUD；②证明 BUD 在无权图的简单网格（k=2）和矩形网格的三格拼板（k=n/3）上可不可约；③改进了判定树是否可拆分为平衡子树的算法，将复杂度从 O(k⁴n) 降至 O(k³n)（在 ε≈1/k² 时进一步降至 O(kn)）；④证明在近似平衡条件下，均匀采样拆分的概率计算为 #P‑complete，从而提出可行的随机拆分方法。

**🔧 技术方法**

技术手段包括：基于图论与 matroid 交换理论的马尔科夫链构造；对 BUD 的状态空间进行平衡拆分的判定；动态规划与区间闭包的压缩技术以降低时间复杂度；复杂度分析与 #P‑hard 归约；以及 Metropolis‑Hastings 等采样加速方法。

**📊 数据集**

使用的数据集包括：4×4、8×8 网格图、以及美国北卡罗来纳州选区划分的真实图数据。

**📈 对比分析**

与 Cycle Walk、ReCom、Up‑Down walk 等方法进行对比：BUD 在有效样本率和自相关上优于 Cycle Walk，且与 Up‑Down walk 在收敛速度上相近；但 BUD 的每步计算量更大，导致单步耗时显著提升。实验结果显示 BUD 在北卡罗来纳州等实际案例中具有良好的混合特性。

**⚠️ 局限性**

局限性包括：BUD 的混合时间尚无严格上界；不可约性仅在特定图形（简单网格、三格拼板）已证明，普适性待进一步验证；在近似平衡下的拆分概率仍需近似估计；以及在大规模图上标记边的重新计算可能导致计算开销进一步增加。

---

## 246. PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

**arXiv ID:** 2602.11530 | [PDF](https://arxiv.org/pdf/2602.11530v1)

**作者:** Eunyeong Cho `[一作]` (Korea Advanced Institute of Science and Technology), Minsoo Rhu `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3427 | [OpenAlex ID](https://openalex.org/A5091648103)

**关键词:** `Machine Learning` `Optimization` `Computational Efficiency` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

设计了一种面向推理型大语言模型的阶段感知调度算法，在多实例环境中对推理阶段与回答阶段分别进行优先级分配与动态迁移，显著降低推理阶段的 TTFT 并保持回答阶段的 SLO。

**💡 创新点**

创新点在于识别推理与回答阶段对中断的不同敏感性，并通过层次化优先队列、阶段感知实例选择与自适应迁移实现资源的细粒度调度，从而在不改变模型或推理流程的前提下显著提升用户体验。

**🔧 技术方法**

采用的技术包括：阶段检测与标记、两级优先队列调度（高优先级推理、低优先级回答）、RR/FCFS 对比、token pacer 控制回答阶段吞吐、KV 缓存迁移与自适应迁移策略、基于模拟器的评估、SLO-aware 的实例级调度算法。

**📊 数据集**

使用的主实验数据集为 DeepSeek‑R1‑Distill‑Qwen‑32B 的推理与回答 token 统计，结合 AlpacaEval2.0、Arena‑Hard、MATH‑500、GPQA、LiveCodeBench 等公开推理型 LLM 负载进行评测。

**📈 对比分析**

与传统 FCFS 与 RR 基线对比，使用 TTFT、SLO 违规率（基于 QoE）和整体吞吐量为评估指标。实验结果表明，针对推理型 LLM 的阶段感知调度将尾部 TTFT 降低 72% 以上，SLO 违规率不高于基线且往往更低，吞吐量与基线相当，说明性能提升而不牺牲整体吞吐。

**⚠️ 局限性**

局限性包括：仅在单 GPU/CPU 结合的同构环境下评估；对极端推理篇幅长、回答篇幅短的工作负载收益有限；KV 缓存迁移开销在高并发下可能更显著；未针对异构 GPU/CPU 计算或专用硬件（如 Rubin‑CPX）进行验证。

---

## 247. AIR: Improving Agent Safety through Incident Response

**arXiv ID:** 2602.11749 | [PDF](https://arxiv.org/pdf/2602.11749v1)

**作者:** Zibo Xiao `[一作]` (Tianjin University), Junjie Chen `[通讯]` (Tianjin University)

**通讯引用:** 5860 | [OpenAlex ID](https://openalex.org/A5100365536)

**关键词:** `Artificial Intelligence` `Safty and Privacy` `Robotic Intelligence` `Transformer` `Large Language Model` `Agentic AI` `Text`

**🎯 论文内容**

提出了 AIR（Agent Incident Response）框架，实现了LLM代理的全生命周期事件响应，包括检测、隔离、恢复和消除。

**💡 创新点**

创新点在于：①设计了面向事件响应的 DSL，支持触发、检查与修复三大组件；②将检测与响应嵌入代理执行循环，实现实时自适应；③自动从事件上下文生成 guardrail 规则，防止同类事件再次发生。

**🔧 技术方法**

技术方案包括：LLM 代理（基于 OpenAI Agent SDK），ANTLR4 解析 DSL，工具接口执行隔离/恢复动作，以及 LLM 生成 guardrail 规则。

**📊 数据集**

实验数据集涵盖三类代理：CodeAgent 使用 RedCode；EmbodiedAgent 使用 SafeAgentBench；Computer‑Use Agent 使用 RiOSWorld 与 OSWorld。

**📈 对比分析**

与人工编写规则对比，AIR 在检测、修复与消除方面均达到 90%–95% 以上成功率；实验还显示其检测/响应时延仅占执行时间的 0.4–1.5%，且对安全任务无误报，性能表现可观。

**⚠️ 局限性**

局限性包括：①依赖代理的推理质量，复杂任务可能导致误判或修复失败；②LLM 自动生成规则时可能过度抽象或过度贴合，需人工校正；③安全检查会引入额外时延，尤其在动作细粒度高的场景；④当前 guardrail 仅实现为计划级规则，跨平台适配需进一步验证。

---

## 248. PatientHub: A Unified Framework for Patient Simulation

**arXiv ID:** 2602.11684 | [PDF](https://arxiv.org/pdf/2602.11684v1)

**作者:** Sahand Sabour `[一作]` (Tsinghua University), Minlie Huang `[通讯]` (Tsinghua University)

**通讯引用:** 15709 | [OpenAlex ID](https://openalex.org/A5044042138)

**关键词:** `Computation and Language` `Large Language Model` `Prompt Engineering` `Text` `Electronic Health Records`

**🎯 论文内容**

开发了 PatientHub 统一框架，标准化患者模拟的定义、组合与部署，并支持多种对话事件与评估机制；

**💡 创新点**

创新点在于将多样化的 LLM‑based 患者模拟方法整合到统一的接口与图结构中，引入可插拔的评估抽象（Binary/Scalar/Categorical/Extraction），并实现可复现、可扩展的评测流水线；

**🔧 技术方法**

技术采用 Python、Hydra、Burr、LiteLLM、Jinja、Pydantic 等工具，配合 LLM‑as‑a‑judge（GPT‑4o）进行生成与评估，日志与数据以 JSON/YAML 结构化；

**📊 数据集**

主要使用 ESC 情感支持对话数据生成 20 个合成患者档案，并结合已有方法（PATIENT‑Ψ、Eeyore、Roleplay‑doh 等）的 profile；

**📈 对比分析**

通过在 CBT 任务下与“好”“坏”治疗师交互，使用统一的五维度评估（一致性、真实性、教学效用等）比较 5 种模拟方法；结果显示简易 Prompt 方法 PATIENT‑Ψ 在一致性与教学效用上表现最佳，Eeyore 成本低但性能竞争；改进版 Ψ‑COT、Ψ‑Doh 在真实性上提升但对教学效用有权衡；

**⚠️ 局限性**

局限包括实验规模仅 20 份档案、使用同一 LLM 评审可能产生偏差、缺乏人工专家验证、未覆盖多方协作、长期随访等场景的支持。

---

## 249. H-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model

**arXiv ID:** 2602.11291 | [PDF](https://arxiv.org/pdf/2602.11291v1)

**作者:** Wenyuan Chen `[一作]` (Huawei Noah’s Ark Lab), Yingxue Zhang `[通讯]` (Huawei Noah’s Ark Lab)

**关键词:** `Robotics` `Robotic Intelligence` `Large Language Model` `Chain-of-Thought` `World Model` `Multimodality`

**🎯 论文内容**

提出一种层级世界模型 H-WM，联合预测逻辑层和视觉层状态转移，为长时序 VLA 控制提供双层引导。

**💡 创新点**

创新点在于将 LLM 细调为符号规划器、引入基于潜在特征的视觉世界模型，并通过层级结构将符号计划与感知子目标对齐；同时构建 LIBERO-Logic 数据集实现端到端训练。

**🔧 技术方法**

采用 LLM (Chain‑of‑Thought 训练)、SigLIP 视觉编码器、潜在特征迭代去噪、切片 Wasserstein 损失、结构化注意力、流匹配目标以及子任务完成预测器。

**📊 数据集**

使用 LIBERO‑Logic（增强版 LIBERO 数据集，含逻辑状态、动作与视觉帧同步）进行训练，评估于 LIBERO‑LoHo 长时序基准。

**📈 对比分析**

与 π₀、π₀.₅ 等端到端 VLA、语言引导、逻辑引导等基线对比，H‑WM‑guided π₀.₅ 在 Q‑Score 与 Success Rate 上均提升 10%–30%，并在 ablation 试验中证明视觉子目标对成功率的关键作用。

**⚠️ 局限性**

缺点包括训练复杂度提升、对符号标注的依赖、对离散谓词的假设限制，以及对高度连续或柔性物体的适应性不足。

---

## 250. Fourier Transformers for Latent Crystallographic Diffusion and Generative Modeling

**arXiv ID:** 2602.12045 | [PDF](https://arxiv.org/pdf/2602.12045v1)

**作者:** Jed A. Duersch `[一作]` (Universite d'Artois), Zied Bouraoui `[通讯]` (Universite d'Artois)

**关键词:** `Machine Learning` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Auto Encoder` `Graph` `Benchmark` `Physics Related`

**🎯 论文内容**

本文提出了一种基于截断傅里叶变换的晶体生成管线，用以处理周期边界、晶体对称性以及可变原子数的生成问题。

**💡 创新点**

创新点在于将晶体描述转移到 reciprocal 空间，通过 Fourier 系数自然编码周期性与空间群对称性，并利用复杂数 Transformer VAE 对其进行压缩，再在潜在空间中做扩散，实现了对多种元素、多原子数结构的可变生成。

**🔧 技术方法**

所使用的技术包括：复杂数 Transformer VAE、潜在扩散模型、复数旋转位置编码、矩阵对数极坐标的格子参数化、傅里叶截断与恢复等。

**📊 数据集**

实验数据来源于 LeMaterial benchmark，约 280 万晶体结构，过滤后约 50 万条 meta‑stable 结构，其中 74.2% 为单元格 ≤ 16 原子。

**📈 对比分析**

与基于坐标的生成模型对比，潜在扩散在小单元格（≤16 原子）下能够生成可变原子数的高质量晶体，重建误差低且扩散过程收敛稳定；在大单元格上仍能精确恢复，显示出良好的可扩展性。

**⚠️ 局限性**

主要局限在于无条件生成时对小单元格倾向性较强，难以覆盖高原子数结构；对非理想格点的恢复能力有限，需要引入条件或引导机制来拓展生成范围。

---

## 251. A technical curriculum on language-oriented artificial intelligence in translation and specialised communication

**arXiv ID:** 2602.12251 | [PDF](https://arxiv.org/pdf/2602.12251v1)

**作者:** Ralph Krüger `[一作]` (University of Applied Sciences Cologne), Ralph Krüger `[通讯]` (University of Applied Sciences Cologne)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

开发并验证了一套针对翻译与专业传播行业的技术性AI素养课程，课程通过Jupyter笔记本提供分阶段的交互式学习，并在TH Köln的MA课程中进行了实证评估。

**💡 创新点**

创新点在于将向量嵌入、神经网络基础、分词与Transformer架构等核心技术模块化为可执行笔记本，配合可视化工具与自评量表，首次系统性地验证此类技术性AI课程在翻译专业学习中的教学效果。

**🔧 技术方法**

主要技术包括Python、Jupyter/Colab、Hugging Face Transformers、BERT、GPT‑2、BPE/WordPiece/Unigram分词器、BertViz等可视化与调试工具。

**📊 数据集**

使用公开预训练模型（BERT、GPT‑2等）及自行训练的小型词向量模型，示例数据为通用英语句子，无需特定标注数据集。

**📈 对比分析**

通过前测与后测的自评量表（TrAILS）与回顾性自评进行比较，发现学生在课程前后知识得分从3.72提升至6.76，p<0.001，效应量d=1.60，表明教学效果显著；未与对照组或其他教学法进行直接比较。

**⚠️ 局限性**

局限包括样本量有限（24→15名参与者），缺乏一致参与者ID导致可能的选择偏差，课程内容对非编程背景学生仍具挑战性，且自评方法可能受Dunning‑Kruger效应或自我认知偏差影响。

---

## 252. Real Life Is Uncertain. Consensus Should Be Too!

**arXiv ID:** 2602.11362 | [PDF](https://arxiv.org/pdf/2602.11362v1)

**作者:** Reginald Frank `[一作]` (University of California Berkeley), Natacha Crooks `[通讯]` (University of California Berkeley)

**关键词:** `Distributed, Parallel, and Cluster Computing`

**🎯 论文内容**

本文提出将传统的 f‑threshold 故障模型替换为概率性故障模型，利用服务器故障曲线对 Raft 和 PBFT 的安全性与可用性进行概率分析，并探讨如何通过节点数量与故障概率的权衡实现成本与性能的优化。

**💡 创新点**

创新点在于：①引入每台服务器的故障曲线，打破“同等故障”假设；②证明在同等安全/可用性保证下，可用更少可靠节点或更多低可靠节点实现成本降低；③展示概率分析能揭示安全与可用性之间的内在权衡，从而指导协议的概率原生设计。

**🔧 技术方法**

使用的技术主要包括：概率性安全/可用性公式推导、Markov 失效模型、量化阈值的组合概率计算，以及对现有共识协议（Raft、PBFT）的安全性/可用性条件重写为概率表达式。

**📊 数据集**

数据集方面未使用专门实验数据，而是基于公开的机房遥测（如硬件失效率、软件升级时故障聚集等）以及假设的故障概率（1%、2%、4%、8% 等）进行理论计算和表格演示。

**📈 对比分析**

比较方法：在相同的安全/可用性目标（如 99.97%）下，将不同节点数与不同故障概率的组合进行概率表格对比，展示 Raft 在 3 节点、1% 故障率与 9 节点、8% 故障率下的等效性；性能方面未进行实测，仅给出理论概率和成本/能耗估计。

**⚠️ 局限性**

局限性：①假设故障独立且仅考虑节点失效，未建模网络失效与攻击；②故障曲线需从遥测精确获得，实际环境中可能难以得到；③缺乏实验验证与实现细节，理论推导的安全/可用性估计在真实系统中的表现仍需进一步验证。

---

## 253. Learning to Manipulate Anything: Revealing Data Scaling Laws in Bounding-Box Guided Policies

**arXiv ID:** 2602.11885 | [PDF](https://arxiv.org/pdf/2602.11885v1)

**作者:** Yihao Wu `[一作]` (Tsinghua Shenzhen International Graduate School), Xueqian Wang `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `Robotics` `Object Detection` `Robotic Intelligence` `Transformer` `Diffusion model` `Reinforcement Learning` `Image`

**🎯 论文内容**

提出基于边界框视觉指令的扩散策略（BBox‑DP），并设计Label‑UMI手持设备与自动注释管线，实现语义操纵任务的高效数据收集和泛化学习。

**💡 创新点**

创新点包括：①将目标对象的边界框作为视觉指令，解耦语义与运动；②构建自动化的分割‑检测‑标注管线，实现低成本高精度的数据标注；③系统研究了对象多样性与泛化性能的幂律关系，提出面向数据效率的对象多样性优先收集策略。

**🔧 技术方法**

使用技术包括：扩散式策略（DDIM、U‑Net）、YOLOv8s检测、SAM2分割、Vision Transformer特征提取、Label‑UMI硬件设备、自动化标注脚本以及数据规模分析方法。

**📊 数据集**

使用的数据集为四个真实世界操纵任务（垃圾分类、按钮按压、倒水、取饮料），共16种对象类别、4个环境，每个任务约1600条演示数据，所有对象均通过Label‑UMI管线获得边界框标注。

**📈 对比分析**

与Octo、OpenVLA、OpenVLA‑OFT、Text‑DP、Keypoint‑DP等基线对比，BBox‑DP在四个任务中平均成功率达90%以上，显著优于所有对比方法，并在复杂纹理/形状干扰场景中保持高性能。

**⚠️ 局限性**

局限性：仍需人工手持激光标记进行数据采集，对极端抽象或无边界框的对象适用性有限；依赖检测模块的准确性，若检测误差大可能导致策略失效；对形状多样性影响的机制尚未深入理论解释。

---

## 254. The Automatic Verification of Image-Text Claims (AVerImaTeC) Shared Task

**arXiv ID:** 2602.11221 | [PDF](https://arxiv.org/pdf/2602.11221v1)

**作者:** Rui Cao `[一作]` (University of Cambridge), Andreas Vlachos `[通讯]` (University of Cambridge)

**通讯引用:** 4913 | [OpenAlex ID](https://openalex.org/A5067943980)

**关键词:** `Computation and Language` `Retrieval` `Multimodality` `Transformer` `Large Language Model` `Vision Language Model` `Image` `Text` `Multimodality`

**🎯 论文内容**

组织了一项针对图像‑文本声明的自动事实核查共享任务，发布真实案例数据集、知识库、基线系统，并对参赛系统进行评估。

**💡 创新点**

提供了基于真实事实核查文章的多模态（文本+图像）声明数据；构建了包含文本与图像的预收集知识库；设计了多模态证据检索、证明与自动评估框架，解决了多模态事实核查的端到端流程。

**🔧 技术方法**

利用多模态大语言模型（Gemini‑2.5‑Pro、Gemini‑3‑Pro 等）、BM25 与稠密检索、CLIP/SigLIP/ColPali 等多模态检索器、Playwright/Firecrawl 爬虫进行证据抓取，采用 Ev2R 等自动评估工具进行评测。

**📊 数据集**

使用 AVERIMATEC 图像‑文本事实核查数据集（1297 条真实声明，划分为 793/152/352 的 train/dev/test），以及包含 860k+ 文本 URL 与 13k+ 图像 URL 的预收集知识库。

**📈 对比分析**

通过“score”（条件证据准确率）与基线、参赛系统比较，最高分 HUMANE 达 0.5455，全部参赛系统均优于 baseline 0.1136；同时对不同声明类型、答案类别进行细粒度性能分析。

**⚠️ 局限性**

自动证据评估与人工判断对齐不足，尤其对多模态证据；系统对罕见类别（冲突/证据不足）表现差；依赖闭源大模型导致可复现性受限；存在时序泄露与数据偏差风险。

---

## 255. Time-TK: A Multi-Offset Temporal Interaction Framework Combining Transformer and Kolmogorov-Arnold Networks for Time Series Forecasting

**arXiv ID:** 2602.11190 | [PDF](https://arxiv.org/pdf/2602.11190v1)

**作者:** Fan Zhang `[一作]` (Shandong Technology and Business University), Hua Wang `[通讯]` (Ludong University)

**通讯引用:** 57373 | [OpenAlex ID](https://openalex.org/A5100403938)

**关键词:** `Machine Learning` `Transformer` `Time Series`

**🎯 论文内容**

本文提出了一种新的时间序列预测框架 Time‑TK，结合多偏移令牌嵌入和多偏移时序交互；

**💡 创新点**

创新点在于引入多偏移令牌嵌入（MOTE）捕捉多尺度时序相关性，并将 Transformer 与 Kolmogorov‑Arnold 网络（KAN）结合进行时序特征提取；

**🔧 技术方法**

使用了多偏移令牌嵌入、MI‑KAN（基于 FastKAN 的 RBF），以及 Transformer 的自注意力机制；

**📊 数据集**

在 14 个公开基准数据集上验证，包括 ETT 系列、电力、交易、气象、交通等多种业务场景；

**📈 对比分析**

与 10 个最先进基准模型（如 iTransformer、PatchTST、TimesNet 等）对比，Time‑TK 在大多数数据集上实现了最低的 MSE/MAE，取得了状态‑of‑the‑art 的预测精度；

**⚠️ 局限性**

局限性包括对极端长序列仍需进一步优化，以及模型对不平稳高频数据的鲁棒性待进一步验证。

---

## 256. MAPLE: Modality-Aware Post-training and Learning Ecosystem

**arXiv ID:** 2602.11596 | [PDF](https://arxiv.org/pdf/2602.11596v1)

**作者:** Nikhil Verma `[一作]` (LG Electronics), Youngjoon Kim `[通讯]` (LG Electronics)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Optimization` `Reinforcement Learning` `Multimodality` `Benchmark`

**🎯 论文内容**

提出 MAPLE 框架，对多模态 RL 后训练进行模态感知改进，降低梯度方差、加快收敛并提升鲁棒性。

**💡 创新点**

创新点在于创建模态标注的 MAPLE‑bench、构造模态分层的 MAPO 优化器，以及自适应加权与课程学习策略。

**🔧 技术方法**

采用模态分层批处理、非对称截断、动态采样与对比奖励加权等 RL 技术，并以 Qwen2.5‑Omni 作为基准模型。

**📊 数据集**

使用 MAPLE‑bench（QA 与 Caption 两任务）以及扩展的 MAPLE‑QA+ 数据集，覆盖七种模态组合。

**📈 对比分析**

与传统全模态（MUPO）及多种 ablation 对比，MAPLE 在 Qwen2.5‑Omni 上平均提升 1–3% 准确率，收敛速度提升 3×，多模态与单模态差距缩小 30%，并在缺失或噪声模态下保持更稳健。

**⚠️ 局限性**

局限性包括对特定模型的依赖、对连续奖励的适用性有限，以及在极端模态缺失时仍可能出现策略退化。

---

## 257. Keeping a Secret Requires a Good Memory: Space Lower-Bounds for Private Algorithms

**arXiv ID:** 2602.12209 | [PDF](https://arxiv.org/pdf/2602.12209v1)

**作者:** Alessandro Epasto `[一作]` (Google Research), Pasin Manurangsi `[通讯]` (Google Research)

**关键词:** `Cryptography and Security` `Safty and Privacy`

**🎯 论文内容**

本文研究了用户级差分隐私在流式算法中的内存消耗，并首次给出了无条件的空间下界，证明在多种统计估计任务（如基数计数、最大选择、分位数估计）上，隐私需求会导致与非隐私算法相比的指数级内存差距。

**💡 创新点**

创新点在于提出一种新的多玩家通信游戏，将低内存私有算法的难度与“贡献上限（capping）”的必要性直接关联，进而得到信息理论上的空间下界；这一方法可推广到多类自然问题。

**🔧 技术方法**

主要技术是基于通信复杂度的证明框架，构造难实例并利用通信游戏证明至少需要与“过度活跃”用户数成正比的内存来识别并忽略这些用户。

**📊 数据集**

论文未使用真实数据集，而是通过构造随机生成的“硬实例”来进行理论证明。

**📈 对比分析**

与现有工作相比，本文提供了无条件的下界，确认了之前多项式空间私有算法的空间需求基本最优，并展示了隐私引入的指数级空间分离。

**⚠️ 局限性**

局限性包括：所给的空间-误差权衡在某些参数范围内可能并非最优，完全紧凑的权衡关系尚未确定；此外，论文未给出具体实现或实验验证，只提供理论证明。

---

## 258. Potential-energy gating for robust state estimation in bistable stochastic systems

**arXiv ID:** 2602.11712 | [PDF](https://arxiv.org/pdf/2602.11712v1)

**作者:** Luigi Simeone `[一作]` (Independent Researcher), Luigi Simeone `[通讯]` (Independent Researcher)

**关键词:** `Machine Learning` `Time Series` `Sequential` `Physics Related`

**🎯 论文内容**

提出一种潜能能量门控（potential‑energy gating）方法，利用双井势能函数调节贝叶斯滤波器中观测噪声协方差，以在双稳态系统中实现鲁棒状态估计。

**💡 创新点**

创新点在于将物理能量景观直接嵌入观测可靠性通道，而非传统的统计或硬约束方式；通过局部势能值连续调节观测噪声，显著提升对异常观测的抵御能力。

**🔧 技术方法**

使用扩展卡尔曼滤波器（EKF）、无迹卡尔曼滤波器（UKF）、自适应卡尔曼滤波器（AKF）、集合卡尔曼滤波器（EnKF）以及粒子滤波器（PF）等多种滤波架构实现门控；对观测噪声协方差 R(x)=R₀[1+gV(x)] 进行参数化；采用 L‑BFGS‑B 优化状态更新，并用 Hessian 估计后验协方差。

**📊 数据集**

评估数据集包括：① 采用 Ginzburg‑Landau 双井势能的合成时序数据（含 10% 异常观测）；② NGRIP δ¹⁸O 冰芯记录中的 19 个 Dansgaard‑Oeschger（D‑O）事件，用以验证在真实气候数据上的表现。

**📈 对比分析**

比较方法：在 100 次蒙特卡洛实验中，12 种滤波器（含 5 种门控变体、1 种基准和 6 种标准滤波器）对 RMSE 进行统计；门控滤波器平均提升 57%–80%（最优 PG‑PF 提升 80%），相较于仅靠拓扑信息的 NT‑EKF（57%）和传统鲁棒卡尔曼滤波器（≈38%）表现显著更佳；在 NGRIP 数据中，门控滤波器在异常观测比例提升 12%–52%，并在 5%–15% 异常率下实现 67%–61% 的提升。

**⚠️ 局限性**

局限性：① 需先验给定双井势能参数（α,β,γ），虽对误差容忍度较高但不适用于无先验势能信息的系统；② 目前仅处理标量状态，扩展到多维时需构建多维势能与矩阵化 R；③ 对 Ginzburg‑Landau 形式的假设在真实气候数据中拟合度低，可能限制进一步提升；④ Hessian 估计导致后验方差略显自相关、过度自信；⑤ 在无异常观测时，门控滤波器可能产生过度正则化导致性能下降。

---

## 259. RL over Commodity Networks: Overcoming the Bandwidth Barrier with Lossless Sparse Deltas

**arXiv ID:** 2602.11456 | [PDF](https://arxiv.org/pdf/2602.11456v1)

**作者:** Chaoyi Ruan `[一作]` (National University of Singapore), Jialin Li `[通讯]` (National University of Singapore)

**通讯引用:** 2278 | [OpenAlex ID](https://openalex.org/A5108050353)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Reinforcement Learning` `Optimization` `Large Language Model` `Reinforcement Learning` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

提出了一种针对大型语言模型的强化学习后训练系统，能够在标准以太网和跨区域 WAN 链路上高效地同步参数，支持在松耦合的 GPU 资源上实现实用的 RL 训练。

**💡 创新点**

核心创新点是利用 RL 微调产生的细粒度稀疏更新，构建无损稀疏 Delta 检查点，并配合流式多通道传输、Relay 级联分发以及基于租约的容错调度，实现了在宽带受限网络上接近 RDMA 级别的吞吐量。

**🔧 技术方法**

技术实现包括：Delta 检查点的索引差分 + LEB128 可变长度编码、FP16/BF16 无损值存储；多路 TCP 并行传输与 cut‑through 转发；Relay 节点的双角色推送分发；按版本与吞吐量自适应的作业分配；基于时间租约的故障检测与任务重分配；在 FSDP2 与 vLLM 上无侵入式部署。

**📊 数据集**

在 Qwen3 系列（4B、8B、14B）模型上，使用 Hendrycks MATH、GSM8K、DeepScaleR 三大推理基准进行 RL 微调实验。

**📈 对比分析**

与理想单机 RDMA 集群（Ideal‑SingleDC）、全量权重广播（PrimeRL‑Full）和多流广播（PrimeRL‑MultiStream）对比，Sparse‑Delta 在全量转发下提升 2.4–9.5 倍吞吐率，步长时间接近理想值，差距仅 1.31–8.91%；在跨云场景下，令 tokens/$ 提升 1.21–1.59 倍。

**⚠️ 局限性**

局限性包括：仍受 WAN 延迟与链路抖动影响，CPU 侧 Delta 提取开销在极低带宽下显著；仅针对 RL 微调产生的稀疏更新场景有效，对预训练或梯度聚合等常规分布式训练不适用；依赖于跨域网络稳定性，超大模型的 Delta 构造与重组成本可能随模型规模呈非线性增长。

---

## 260. MDE-VIO: Enhancing Visual-Inertial Odometry Using Learned Depth Priors

**arXiv ID:** 2602.11323 | [PDF](https://arxiv.org/pdf/2602.11323v1)

**作者:** Arda Alniak `[一作]`, Abdullah Aydin Alatan `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Pose Estimation` `Depth Estimation` `Robotic Intelligence` `Transformer` `Simultaneous Localization and Mapping` `Image` `Video`

**🎯 论文内容**

在低纹理、低光照环境下，提出一种基于稠密单目深度估计（MDE）的视觉惯性里程计（VIO）改进方法，利用深度先验在前端（DIFT）和后端（仿射对齐 + 序序约束）融合，提高姿态估计精度并保证实时性。

**💡 创新点**

创新点包括：①将深度图嵌入RGB蓝色通道进行特征跟踪；②在后端加入方差门控的仿射对齐残差和稀疏阶序约束；③通过不确定性动态加权过滤瞬时深度噪声，使系统可在资源受限的边缘设备上实时运行。

**🔧 技术方法**

使用 Vision Transformer 基础的 MDE 模型（DepthAnythingAC、VideoDepthAnything），FP16 TensorRT 推理，VINS-Mono 优化框架，RANSAC+EMA 估计仿射参数，Soft Hinge Loss 约束阶序，滑动窗口优化。

**📊 数据集**

评估数据集为 TartanGround（城市地面序列）和 M3ED Spot（四足机器人）两套真实世界序列。

**📈 对比分析**

与原始 VINS‑Mono 及其它 MDE 辅助 VIO 方法对比，在 TartanGround Downtown 上 ATE 下降 28.3%，在 M3ED 上平均提升 14.3%，并能在低纹理/高速场景下防止轨迹发散。

**⚠️ 局限性**

局限性：零射 MDE 模型存在间帧闪烁，前端 DIFT 可能破坏光度一致性；系统依赖仿射对齐假设，可能对动态场景适应不足；对大规模复杂环境的鲁棒性尚需进一步验证。

---

## 261. Brain4FMs: A Benchmark of Foundation Models for Electrical Brain Signal

**arXiv ID:** 2602.11558 | [PDF](https://arxiv.org/pdf/2602.11558v1)

**作者:** Fanqi Shen `[一作]` (Zhejiang University), Yang Yang `[通讯]` (Zhejiang University)

**通讯引用:** 111225 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `Machine Learning` `Classification` `Transformer` `Graph Neural Network` `Contrastive Learning` `Generative Adversarial Network` `Biomedical Data` `Time Series` `Benchmark`

**🎯 论文内容**

本文构建了一个统一的基准平台 Brain4FMs，整合 15 种脑基础模型（BFM）与 18 个公开 EEG/iEEG 数据集，提供标准化的跨主体微调与评估流程。

**💡 创新点**

创新点在于提出了面向 SSL 的统一分类框架，系统分析了预训练数据、SSL 原理与模型结构对跨任务泛化的影响，并在基准上揭示了对比学习、生成式学习、空间与频率建模等关键因素的相互作用。

**🔧 技术方法**

主要技术包括：自监督学习（对比、生成、预测等多范式）、Transformer/图卷积网络等模型骨干、频域重构与代码簿离散化、以及跨主体留出验证和决策边界诊断等评估手段。

**📊 数据集**

使用了 18 个公开数据集，涵盖疾病诊断（癫痫、帕金森、抑郁等）、睡眠分期、脑机接口（运动想象/执行）、情感计算等 11 类下游任务。

**📈 对比分析**

通过统一的数据预处理、微调分类器和跨主体留出评估，比较了各模型在 Accuracy、AUROC、F1、Cohen’s κ 等指标上的表现。结果显示，生成式模型普遍优于对比式模型，CPC 与多通道空间建模提升性能，频域重构与层次化代码簿也能带来一定优势，但没有单一模型在所有任务上占优。

**⚠️ 局限性**

局限性包括：基准仅覆盖分类任务，未深入探讨生成式或多任务适配；对比与生成模型间差异受数据、架构和规模共同影响，难以单独归因；代码簿离散化虽提升稳健性，但可能削弱细粒度区分；未来需加入冻结、少样本、零样本等更严格的评估。

---

## 262. Rate-Reliability Tradeoff for Deterministic Identification over Gaussian Channels

**arXiv ID:** 2602.12182 | [PDF](https://arxiv.org/pdf/2602.12182v1)

**作者:** Pau Colomer `[一作]` (Technische Universitaet Muenchen), Andreas Winter `[通讯]` (Universitaet zu Koeln)

**关键词:** `Information Theory` `Recognition` `Optimization`

**🎯 论文内容**

本文研究了确定性识别（DI）在一般线性高斯通道上的速率-可靠性折衷，提出了错误指数与可识别消息数之间的上界和下界；

**💡 创新点**

创新点在于首次将DI的速率-可靠性分析扩展到连续输出的高斯通道，揭示了当错误指数呈指数衰减时速率退化为线性，而仅当错误指数慢衰减时可恢复线性对数（log n）扩展；

**🔧 技术方法**

主要技术包括：利用输出分布的总变差距、Bhattacharyya系数与马氏距离的关系，构造欧氏距离下的码字Packing；对极大似然/距离解码器进行Chernoff界分析；以及利用Rényi相对熵与假设检验相对熵的关系进行逆向证明；

**📊 数据集**

本文为理论研究，无使用实验数据集；

**📈 对比分析**

通过与已知的离散输出DI结果对比，证明了在高斯通道上线性/线性对数速率的匹配上界和下界；理论上给出了线性速率的上界为O(1/|log E|)，线性对数速率的上界为½log n+O(1)，并提供了相应的码构造实现了相同阶数的速率；

**⚠️ 局限性**

限制包括：仅适用于可逆的平方线性变换；仅考虑功率约束下的Gaussian噪声；码构造基于距离解码，可能在实际实现中对误码率产生较大波动；未给出对非高斯或非线性通道的推广；

---

## 263. SynthRAR: Ring Artifacts Reduction in CT with Unrolled Network and Synthetic Data Training

**arXiv ID:** 2602.11880 | [PDF](https://arxiv.org/pdf/2602.11880v1)

**作者:** Hongxu Yang `[一作]` (Science and Technology Organization, GE HealthCare), Gopal Avinash `[通讯]` (Science and Technology Organization, GE HealthCare)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Data Synthesis` `Convolutional Neural Network` `Image` `Computed Tomography`

**🎯 论文内容**

提出一种基于合成数据训练的双域迭代网络（SynthRAR）来去除CT图像中的环形伪影。

**💡 创新点**

创新点在于：①利用物理建模的无理想探测器响应（无效像素与不一致响应）构造逆问题；②在ISTA‑Net框架中加入两套轻量级CNN估计无效像素和不一致响应；③通过从自然图像生成的合成CT数据训练，避免了昂贵的临床数据收集。

**🔧 技术方法**

核心技术包括：物理正向投影与改进的非理想投影公式、ISTA‑Net（迭代软阈值网络）、双域（sinogram 与图像）学习、合成数据生成。

**📊 数据集**

训练使用从ILSVRC2017自然图像生成的约50k张合成CT样本；评估在四个公开CT数据集（MMWHS、RibFrac、RIRE、LDCT）上完成。

**📈 对比分析**

与现有SOTA方法（AST、DeepRAR、NAFNet、Norm、WaveFFT、Super、Riner）进行对比，指标为MAE、PSNR、SSIM。SynthRAR在所有测试集上均明显优于对比方法，尤其在跨域（HU分布或扫描几何变化）场景中表现最佳。

**⚠️ 局限性**

局限性包括：①计算成本较高（约110 ms/图像，GPU内存2.45 GB）；②尚未在多排或锥束CT、短扫描角度等更复杂扫描模式下验证；③对无效像素比例的假设（2%）较为激进，实际临床扫描可能需要更严格的适配与现场评估。

---

## 264. WorldTree: Towards 4D Dynamic Worlds from Monocular Video using Tree-Chains

**arXiv ID:** 2602.11845 | [PDF](https://arxiv.org/pdf/2602.11845v1)

**作者:** Qisen Wang `[一作]` (Beihang University), Jia Li `[通讯]` (Beihang University)

**通讯引用:** 83905 | [OpenAlex ID](https://openalex.org/A5009049500)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Depth Estimation` `Optimization` `Gaussian Splatting` `Optical Flow` `Video`

**🎯 论文内容**

提出一种统一的多层次时空框架——WorldTree，用于单目动态重建；

**💡 创新点**

引入时间分区树（Temporal Partition Tree）实现粗细分层的时间优化，并引入空间祖先链（Spatial Ancestral Chains）提供多层次空间补充，从而实现时空分离、层级优化与空间特化；

**🔧 技术方法**

利用3D高斯表征（3D Gaussian Splatting）、双四元数混合、图搜索、光流、深度预测等基础模型，结合树结构的并行优化与包束调整；

**📊 数据集**

在扩展后的NVIDIA-LS（长度达160帧）和DyCheck数据集上进行评估，并使用SAM2生成动态前景掩码；

**📈 对比分析**

与多种基线（D3DGS、4DGS、SplineGS、MoSca、HiMoR等）比较，WorldTree在NVIDIA-LS的mPSNR、mSSIM、mAVGE、LPIPS等指标均显著领先，DyCheck上mLPIPS提升近19%；

**⚠️ 局限性**

方法依赖预训练的基础模型（深度、光流、追踪等），且对外部先验较为敏感，未来需改进先验提取与模型自适应能力。

---

## 265. EM-Aware Physical Synthesis: Neural Inductor Modeling and Intelligent Placement & Routing for RF Circuits

**arXiv ID:** 2602.11461 | [PDF](https://arxiv.org/pdf/2602.11461v1)

**作者:** Yilun Huang `[一作]` (University of Southern California), Hamidreza Aghasi `[通讯]` (University of California Irvine)

**关键词:** `Hardware Architecture` `Optimization` `Tabular`

**🎯 论文内容**

本论文提出了一套基于机器学习的完整 RF 电路从网表到可制造 GDSII 布局的自动化合成流程；

**💡 创新点**

创新点在于：①使用基于 7.5M EM 仿真样本训练的神经元诱导器 Q‑因子预测模型，实现 1–100 GHz 频段下的高精度 EM 预测；②设计了智能 P‑Cell 优化器，可在满足电路规范与 DRC 的前提下最小化占地面积；③构建了频率感知的布图与路由引擎，利用 A* 算法与 EM 规则实现全流程 DRC 合规布局；

**🔧 技术方法**

技术方法包括：多层感知器（MLP）神经网络、梯度反向优化（inverse design）、A* 全局布线、频率依赖的 EM 距离规则、Python PCell 组装与 PDK 交互；

**📊 数据集**

使用的数据集为 18,210 条电感几何配置，经过 1–100 GHz 频率扫描后生成 7.5 M 条训练样本，包含宽度、高度、布局尺寸及对应 Q‑因子；

**📈 对比分析**

与传统模拟/遗传/强化学习等电路优化方法对比，本框架在 Q‑因子预测上 MAE 0.419、MSE 0.423、R² 0.994、MAPE 1.36%，逆向优化成功率达 93.77%，实现了首个完全从网表到 DRC 合规 GDSII 的端到端流程；

**⚠️ 局限性**

局限性在于：仅实现了被动元件的完整布局，晶体管布局仍以简化盒子形式出现；缺乏完整的 EM 循环与后仿真验证，技术节点和电路块覆盖仍有限；

---

## 266. DDL2PropBank Agent: Benchmarking Multi-Agent Frameworks' Developer Experience Through a Novel Relational Schema Mapping Task

**arXiv ID:** 2602.11198 | [PDF](https://arxiv.org/pdf/2602.11198v1)

**作者:** Shafiuddin Rehan Ahmed `[一作]` (Accenture), Wei Wei `[通讯]`

**关键词:** `Computation and Language` `AI Code Assistant` `Large Language Model` `Agentic AI` `Tabular` `Benchmark`

**🎯 论文内容**

提出DDL2PropBank基准，评估多代理框架在代码复杂度与AI辅助编程上的体验

**💡 创新点**

首次将数据库schema映射到PropBank语义角色集作为受控任务，并从代码复杂度与AI可协助性两维度系统评测框架

**🔧 技术方法**

使用Agent-as-a-Tool模式、Model Context Protocol (MCP)、LLM工具调用、Copilot、Claude Opus 4.5 进行结构化评测

**📊 数据集**

基准采用DDL2PropBank自身的数据库schema（RelBench等）、PropBank角色集、MCP服务器

**📈 对比分析**

通过静态分析（LLOC、CCN）衡量代码复杂度，利用Copilot生成代码并用Claude Opus 4.5 评估结构对齐，再用实际运行测试验证功能；结果显示Agno、Claude SDK、OpenAI Agents在两维上表现最佳，代码复杂度最低的Pydantic AI与Agno表现突出

**⚠️ 局限性**

局限在于任务单一、PropBank单语种、评测覆盖框架有限、评测方法受LLM训练数据新颖性影响，需随框架演进更新

---

## 267. Supervise-assisted Multi-modality Fusion Diffusion Model for PET Restoration

**arXiv ID:** 2602.11545 | [PDF](https://arxiv.org/pdf/2602.11545v1)

**作者:** Yingkai Zhang `[一作]` (Beijing Institute of Technology), Ying Fu `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 5572 | [OpenAlex ID](https://openalex.org/A5100738025)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Transformer` `Diffusion model` `Multimodality` `Biomedical Data` `Positron Emission Tomography` `Magnetic Resonance Imaging`

**🎯 论文内容**

提出了一种监督辅助多模态融合扩散模型（MFdiff），利用低剂量PET和对应的MRI进行高质量标准剂量PET图像的重建。

**💡 创新点**

创新点包括：① 结合Transformer和卷积的双分支内模态学习（IML）与跨模态聚合（CMA）模块，既提取全局语义信息又保留细节特征；② 将融合特征作为条件输入到扩散模型，实现更精细的重建；③ 采用两阶段监督辅助训练策略，在大规模仿真数据上学习通用先验，再在少量真实临床数据上微调，实现良好的跨域泛化。

**🔧 技术方法**

使用的技术主要有：多模态特征融合（Transformer+卷积）、可逆神经网络（细节特征提取）、扩散概率模型（DDPM）作为重建网络、时间嵌入和自注意力的U-Net结构、AdamW优化器。

**📊 数据集**

数据集：① 20个BrainWeb 3D脑模体的仿真PET/MRI数据，共1000张切片；② 三种不同条件下的真实PET/MRI OOD数据（扫描时长、注射剂量、给药方案），分别包含30、60、90个健康志愿者的成对低剂量与标准剂量PET。

**📈 对比分析**

与M-UNet、FBSEM、EA-GAN、Hi-Net、CNCL、CSRD等六种最先进方法在phantom和三类OOV数据上进行对比；MFdiff在PSNR、SSIM、NMSE上均优于对比方法，最优场景PSNR提升约0.9 dB，SSIM提升约0.003，NMSE下降约0.001。

**⚠️ 局限性**

局限性：仅在少量健康志愿者（6人）与单一扫描仪、单一放射性示踪剂上验证；对多中心、多设备、多示踪剂的泛化能力尚未测试；未来需要探索缺失模态、未配对数据以及在疾病诊断和分割等下游任务中的应用。

---

## 268. AltTS: A Dual-Path Framework with Alternating Optimization for Multivariate Time Series Forecasting

**arXiv ID:** 2602.11533 | [PDF](https://arxiv.org/pdf/2602.11533v1)

**作者:** Zhihang Yuan `[一作]` (University of Edinburgh), Mahesh K. Marina `[通讯]` (University of Edinburgh)

**通讯引用:** 9420 | [OpenAlex ID](https://openalex.org/A5022046402)

**关键词:** `Machine Learning` `Optimization` `Transformer` `Time Series`

**🎯 论文内容**

本文提出了一个双路径框架，分别用线性回归模型捕捉自回归（AR）动态，用Transformer（改进的iTransformer）捕捉跨变量关系，并通过交替优化（AO）来训练这两条路径。

**💡 创新点**

创新点在于将AR和跨变量关系显式分离，采用交替优化减少梯度相互干扰，从而实现更稳定的学习并显著提升长时序预测性能；同时将训练调度视为模型设计变量，提出基于优化原理的设计思路。

**🔧 技术方法**

技术手段包括：RLinear线性预测器、改进的iTransformer与Cross‑Relation Self‑Attention、RevIN归一化、AMSGrad优化器、交替优化（块坐标下降）以及梯度方差分析。

**📊 数据集**

实验使用了七个标准长时序预测基准：Weather、Traffic、Electricity、ETTh1、ETTh2、ETTm1、ETTm2，并在 96/192/336/720 四个预测长度上评估。

**📈 对比分析**

与线性模型（RLinear、DLinear、OLinear）、Transformer（PatchTST、iTransformer、Informer）以及混合模型（TimeBase）等强基线对比，采用 MSE/MAE 指标，结果显示本文方法在大多数数据集和长度下均达到或逼近 state‑of‑the‑art，尤其在长时序预测上优势明显。

**⚠️ 局限性**

局限性包括：交替优化需要额外的调度与训练时间；在极度自回归（CI）友好的数据上，单一模型或联合训练可能略优；以及对不同网络架构的通用性仍需进一步验证。

---

## 269. DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing

**arXiv ID:** 2602.12205 | [PDF](https://arxiv.org/pdf/2602.12205v1)

**作者:** Dianyi Wang `[一作]` (Fudan University), Jiaqi Wang `[通讯]` (Fudan University)

**通讯引用:** 21102 | [OpenAlex ID](https://openalex.org/A5100365347)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Reinforcement Learning` `Transformer` `Vision Language Model` `Reinforcement Learning` `Supervised Fine-Tuning` `Image` `Text` `Multimodality`

**🎯 论文内容**

设计并实现了一款5B参数的统一多模态模型DeepGen 1.0，集成图像生成、编辑、推理与文本渲染于一体。

**💡 创新点**

创新点：①堆叠通道桥接(SCB)融合多层VLM特征并注入可学习的“思考”token，实现高效的VLM‑DiT对齐；②三阶段数据驱动训练（对齐预训练、联合SFT、RL）与MR‑GRPO强化学习；③在保持5B规模的前提下，性能与传统8B–80B模型相当甚至超越。

**🔧 技术方法**

技术手段：VLM‑DiT架构（Qwen‑2.5‑VL + SD3.5‑Medium）、SCB、LoRA微调、可学习think tokens、MR‑GRPO强化学习、三阶段训练流程。

**📊 数据集**

数据集：文本‑图像对（Text‑to‑Image‑2M, LAION‑Aesthetic‑6M, Megalith‑10M, RedCaps‑5M, CC‑12M），指令式数据（BLIP‑3o, ShareGPT‑4o‑Image, Echo‑4o‑Image, OpenGPT4o‑Image）与自制样本；编辑数据（NHR‑Edit, GPT‑Image‑Edit, ShareGPT‑4o‑Image‑Edit, Uniworld‑Edit, Nano‑Banana, X2I2 等）；推理生成/编辑（UniReason 150k/100k）；文本渲染（文档/信息图 QA、Gemini 2.5 Pro 合成、500k 渲染样本）。

**📈 对比分析**

在GenEval、DPGBench、UniGenBench、ImgEdit、GEdit‑EN、WISE、T2I‑CoREBench、RISE、UniREditBench、CVTG‑2K 等公开基准上与同类开源模型及闭源系统对比，5B 模型在多项指标名列前茅，推理最高得分0.73，编辑77.5，整体性能与8B–80B模型相当且显著低于参数量。

**⚠️ 局限性**

局限：对极高分辨率、极长提示或细粒度多模态交互的表现仍不足；缺乏系统的多语言、跨文化细节评估；RL 阶段受奖励设计影响；在特定任务上仍不及专用大型模型；未对训练与推理时延、算力成本做详细评估。

---

## 270. Safety Beyond the Training Data: Robust Out-of-Distribution MPC via Conformalized System Level Synthesis

**arXiv ID:** 2602.12047 | [PDF](https://arxiv.org/pdf/2602.12047v1)

**作者:** Anutam Srinivasan `[一作]` (Georgia Institute of Technology), Glen Chou `[通讯]` (Georgia Institute of Technology)

**关键词:** `Robotics` `Optimization` `Safty and Privacy` `Robotic Intelligence` `Reinforcement Learning` `Time Series`

**🎯 论文内容**

开发了一种基于加权合成预测与系统层级综合的鲁棒外分布MPC框架，实现了对学习模型误差的自适应置信边界，并在约束收缩下保证高概率安全。

**💡 创新点**

创新点在于将加权合成预测用于生成状态-控制相关的椭圆误差界，并将其嵌入SLS鲁棒MPC中，同时通过在线数据更新和主动不确定性降低机制提升OOV性能。

**🔧 技术方法**

技术手段包括加权合成预测、系统层级综合(SLS)、非线性MPC、连续凸优化(SCP)、以及多变量高斯负对数似然训练不确定性模型。

**📊 数据集**

使用的数据集为从4D Dubins车和12D四旋翼仿真环境收集的训练、校准和在线数据，覆盖受限训练域与外分布场景。

**📈 对比分析**

通过与基准无鲁棒MPC、固定半球误差MPC对比，实验显示在外分布场景下我们的CP‑Ellipsoid方法在安全性、轨迹安全距离和成功率上明显优于基线，并保持可接受的计算时间。

**⚠️ 局限性**

局限性包括对校准点密度的依赖导致在数据稀疏区域过度保守，以及在线更新时可能违背独立性假设，导致理论覆盖保证弱化。

---

## 271. LoRA-based Parameter-Efficient LLMs for Continuous Learning in Edge-based Malware Detection

**arXiv ID:** 2602.11655 | [PDF](https://arxiv.org/pdf/2602.11655v1)

**作者:** Christian Rondanini `[一作]` (University of Insubria), Ashish Kundu `[通讯]` (Cisco Research)

**关键词:** `Cryptography and Security` `Anomaly Detection` `Federated Learning` `Safety and Privacy` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Tabular`

**🎯 论文内容**

提出一种基于 LoRA 的轻量级 LLM 连续学习框架，用于边缘设备恶意软件检测。

**💡 创新点**

创新点在于将参数高效的 LoRA 适配器与中心化聚合协调器相结合，实现边缘本地微调与跨设备知识共享，兼顾实时性与隐私。

**🔧 技术方法**

使用 DistilBERT、DistilGPT‑2、TinyT5 轻量化 Transformer，并通过 LoRA（及其量化版本 QLoRA）实现参数高效微调。

**📊 数据集**

采用公开 IoT 安全数据集 Edge‑IIoTset 和 TON‑IoT，进行多轮增量学习实验。

**📈 对比分析**

与全量微调和无共享基线对比，LoRA 在 1% 参数增量下实现 20–25% 的准确率提升，F1 得分保持稳定且收敛更快。

**⚠️ 局限性**

局限在于仍需中心协调层，适配器累积可能导致长期扩展成本；对极端分布漂移与新攻击类型的泛化能力未完全验证。

---

## 272. Verifiable Provenance of Software Artifacts with Zero-Knowledge Compilation

**arXiv ID:** 2602.11887 | [PDF](https://arxiv.org/pdf/2602.11887v1)

**作者:** Javier Ron `[一作]` (KTH Royal Institute of Technology), Martin Monperrus `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 6683 | [OpenAlex ID](https://openalex.org/A5027206285)

**关键词:** `Software Engineering` `Text`

**🎯 论文内容**

提出并实现了基于零知识虚拟机的可验证编译方法，用于在不依赖可信硬件或完整重建的前提下生成二进制源代码出处的加密证明。

**💡 创新点**

创新点在于将完整编译器运行在zkVM中，直接产生紧凑且可验证的编译证明，实现了来源追溯与执行完整性同时得到保证的全流程解决方案。

**🔧 技术方法**

采用RISC Zero zkVM执行ChibiCC C编译器，并结合SHA‑256源代码哈希、ImageID校验和zkVM生成的执行证明。

**📊 数据集**

实验使用Csmith生成的200个随机C程序，以及OpenSSL 31个源文件和libsodium 21个源文件。

**📈 对比分析**

与标准编译相比，zk编译平均耗时大约比标准编译慢四个数量级；验证时间约为50秒，显著快于编译；证明大小随源码大小线性增长，最大约120 MB。

**⚠️ 局限性**

局限在于zk编译的高成本与长耗时，受限于当前zkVM的性能；仅支持ChibiCC支持的C子集，且对多平台和完整工具链的直接支持尚不完善。

---

## 273. Differentially Private and Communication Efficient Large Language Model Split Inference via Stochastic Quantization and Soft Prompt

**arXiv ID:** 2602.11513 | [PDF](https://arxiv.org/pdf/2602.11513v1)

**作者:** Yujie Gu `[一作]` (Zhejiang University), Wenyuan Xu `[通讯]` (Zhejiang University)

**通讯引用:** 7046 | [OpenAlex ID](https://openalex.org/A5060351020)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Computational Efficiency` `Generation` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

提出了DEL框架，实现LLM推理的差分隐私与通信高效的分割推理

**💡 创新点**

创新点在于将预训练的编码解码降维与随机n比特量化机制相结合，并通过服务器端软提示补偿隐私导致的效能下降，首次在LLM推理中实现低比特量化与软提示协同的隐私‑效能权衡

**🔧 技术方法**

使用预训练的编码器–解码器、f‑DP随机n比特量化、Soft Prompt调优、Gaussian机制比较、差分隐私度量等技术

**📊 数据集**

在C4、WikiText‑2、Penn Treebank、CNN/Daily Mail、Quora Question Pairs、MSR Paraphrase Corpus等数据集上进行实验

**📈 对比分析**

与RANTEXT、InferDPT、SnD等基线在相同隐私预算下进行对比，DEL在文本生成任务上显著降低PPL、提升COH，并在NLP理解任务上获得更高的Accuracy/AUC；同时实现了1/2/4比特量化，通信压缩至原始嵌入的1/32

**⚠️ 局限性**

局限性包括：需要在服务器端预训练编码器/解码器，对软提示在极高隐私或跨域迁移时仍有一定性能衰减；极低隐私或极高维度的量化可能导致语义信息丢失或误差累积

---

## 274. Accelerating Robotic Reinforcement Learning with Agent Guidance

**arXiv ID:** 2602.11978 | [PDF](https://arxiv.org/pdf/2602.11978v1)

**作者:** Haojun Chen `[一作]` (Peking University), Yaodong Yang `[通讯]` (Peking University)

**通讯引用:** 4377 | [OpenAlex ID](https://openalex.org/A5025046910)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Vision Language Model` `Multimodality`

**🎯 论文内容**

本论文提出 Agent-guided Policy Search (AGPS)，通过多模态智能体自动化代替人类监督，使用异步失败检测器 FLOAT 与可执行工具箱，实现机器人 RL 训练的自适应指导与探索修剪。

**💡 创新点**

创新点在于：①将多模态智能体视为语义世界模型，注入先验知识；②异步 FLOAT 触发机制平衡高延迟推理与高频控制；③通过行动指导与空间约束两种方式实现精确纠正与探索剪枝。

**🔧 技术方法**

采用多模态基础模型（如 Qwen3-VL、DINOv2 ViT-B/14）做视觉编码与语义定位，使用 Optimal Transport 度量策略偏差，结合 Action Primitives、Perception、Geometry、Memory 等工具箱；同时采用离线演示支持的 SERL 进行 RL。

**📊 数据集**

实验数据集主要为两项真实机器人任务：USB 插入（高精度刚性装配）和中国结挂钩（柔性物体操控），并使用专家演示轨迹作为对比基准。

**📈 对比分析**

与 SERL、HIL‑SERL（人类干预）及 VLM Planner 对比，AGPS 在 USB 插入任务中仅需 600 步即可实现 100% 成功率（约 8 分钟），而 HIL‑SERL 需 1000 步；在中国结任务中 AGPS 在 4000 步即可 100% 成功率，HIL‑SERL 仍停留在 0%。

**⚠️ 局限性**

局限性包括：① 依赖大型 VLM 的视觉定位，遮挡或幻觉可能导致失败；② 大模型推理延迟限制了指导频率，难以满足高动态环境下的实时反馈需求。

---

## 275. Composition-RL: Compose Your Verifiable Prompts for Reinforcement Learning of Large Language Models

**arXiv ID:** 2602.12036 | [PDF](https://arxiv.org/pdf/2602.12036v1)

**作者:** Xin Xu `[一作]` (Tencent), Can Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 3342 | [OpenAlex ID](https://openalex.org/A5029104097)

**关键词:** `Computation and Language` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Prompt Engineering` `Text` `Physics Related`

**🎯 论文内容**

提出并实现了通过自动合成已有可验证提示生成更具挑战性的复合提示，从而充分利用原始易提示并提升RLVR训练效果。

**💡 创新点**

核心创新在于引入顺序提示合成（Sequential Prompt Composition）技术，将多条原始提示组合成新的复合提示，并配合课程化的深度提升（Depth Curriculum）与跨域合成策略，显著扩大有效训练样本并提供隐式过程监督。

**🔧 技术方法**

采用强化学习可验证奖励（RLVR）框架中的GRPO优化、动态采样过滤无信息提示，并在此基础上构造复合提示数据；同时实现了多层次课程化训练和跨域提示合成。

**📊 数据集**

主要使用公开的数学推理数据集MATH12K（以及其子集）、AIME24/25、BeyondAIME、IMOBench、GPQA、MMLU-Pro等；跨域实验中还结合了Physics子集（MegaScience）。

**📈 对比分析**

与仅在原始提示上进行RL训练的基线相比，Compose方法在4B–30B模型上平均提升3.3%–10.5%数学性能，其中30B模型在AIME24上提升至37.9%；课程化深度策略进一步提升2–4%，并在跨域实验中显著提高多任务通用性。

**⚠️ 局限性**

局限性包括：目前仅在数学推理任务上验证，跨域效果仍需进一步探索；复合提示的构造复杂度和可解释性有限；对极大规模模型或长文本生成的适用性尚未完全评估。

---

## 276. Do MLLMs Really Understand Space? A Mathematical Reasoning Evaluation

**arXiv ID:** 2602.11635 | [PDF](https://arxiv.org/pdf/2602.11635v1)

**作者:** Shuo Lu `[一作]` (Chinese Academy of Sciences Institute of Automation and Chinese Academy of Sciences School of Artificial Intelligence), Jian Liang `[通讯]`

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Vision Language Model` `Multimodality` `Text` `Benchmark`

**🎯 论文内容**

构建了 MathSpatial 框架，包括评估基准 MathSpatial‑Bench、训练语料 MathSpatial‑Corpus 以及可解释的推理模型 MathSpatial‑SRT，用于提升多模态大型语言模型在数学空间推理上的表现。

**💡 创新点**

创新点在于拆分空间推理为三种原子操作（Correlate、Constrain、Infer）形成结构化推理轨迹，提供无感知干扰的基准和大规模可解释训练数据，从而显著缩小人机差距并提高推理效率。

**🔧 技术方法**

采用多模态大语言模型（如 Qwen2.5‑VL、InternVL3 等）的监督微调、基于 GPT‑4o 生成的结构化推理轨迹以及双角色验证机制，实现可解释且高效的推理。

**📊 数据集**

使用了 2,000 题的 MathSpatial‑Bench 评估集、8,000 题的 MathSpatial‑Corpus 训练集，均来自公开教材和考试题库，并经过人工验证。

**📈 对比分析**

通过与多款闭源和开源模型的对比实验，MathSpatial‑fine‑tuned Qwen2.5‑VL‑7B 在人类 95% 水平的 60% 以内取得 22–25% 的准确率提升，并将推理 token 数量减少约 25%，展示了在空间推理任务上的显著性能提升。

**⚠️ 局限性**

局限性包括仍然在高级抽象推理、几何规则严格性以及多步骤连贯性方面存在错误，且当前框架主要针对二维/三维几何图形，需进一步扩展到更复杂的现实场景。

---

## 277. DEpiABS: Differentiable Epidemic Agent-Based Simulator

**arXiv ID:** 2602.12102 | [PDF](https://arxiv.org/pdf/2602.12102v1)

**作者:** Zhijian Gao `[一作]` (Nanyang Technological University), Bo An `[通讯]` (Nanyang Technological University)

**通讯引用:** 6788 | [OpenAlex ID](https://openalex.org/A5017743551)

**关键词:** `Multiagent Systems` `Computational Efficiency` `Explainability and Interpretability` `Agentic AI` `Time Series`

**🎯 论文内容**

提出了一种结构中心的可区分代理模型 DEpiABS，结合细粒度社会与流行病模块，通过张量化和松弛技术实现全流程可微化，并引入 z‑score 缩放以降低模拟规模对计算成本的影响，最终实现高逼真度、可解释性与高效率的三重目标。

**💡 创新点**

创新点主要包括：
• 结构中心设计，拒绝 ANN 组件，保持完整的机制可解释性；
• 全流程可微化（Selective Relaxation + Tensorization）使得梯度优化可直接应用于复杂 ABM；
• z‑score 缩放方法在不调参的前提下，实现在小规模模型下对大规模真实数据的精准对齐，从而显著降低内存与时间开销。

**🔧 技术方法**

技术手段：
• 可区分代理模型（Differentiable Agent‑Based Model, DABM）；
• 选定松弛（Selective Relaxation）与重参数化技巧实现非连续决策与随机变量的可微化；
• 张量化（Tensorization）对状态、决策与接触矩阵进行矩阵化；
• z‑score 基础的输出缩放；
• GPU 加速与线性可扩展的实现。

**📊 数据集**

使用的数据集：
• 美国马萨诸塞州 10 个县的 COVID‑19 死亡时间序列（人群规模 7 万至 160 万）；
• 同地区的流感样疾病（ILI）时间序列。

**📈 对比分析**

比较方法：在与 GradABM 的三个版本（C‑GradABM、DC‑GradABM、JDC‑GradABM）使用相同数据、预处理、评估周期和指标（ND、RMSE、MAE）进行对比；同时对模型可扩展性做线性运行时间测试并与 Mesa 版本做基准。性能结果显示：DEpiABS 在 COVID‑19 死亡预测的 ND、RMSE、MAE 均优于 JDC‑GradABM，并在使用同等数据量下实现与 JDC‑GradABM 相近或更好的预测；在流感预测中与 DC‑GradABM 相当或更优；运行时间呈线性增长，且比 Mesa 版本快 200–250 倍。

**⚠️ 局限性**

局限性：
• 目前仅在美国马萨诸塞州数据上验证，缺乏跨地区或跨疾病的广泛外推性评估；
• 模型仍需人工手动设定诸多参数（如行为阈值、经济参数），对参数敏感性可能影响预测稳定性；
• 虽然采用 z‑score 缩放减少规模影响，但在极大人群或极端数据分布下的表现尚未充分测试；
• 由于完全透明的机制实现，模型规模较大，极端大规模仿真仍需更高算力。

---

## 278. Which Feedback Works for Whom? Differential Effects of LLM-Generated Feedback Elements Across Learner Profiles

**arXiv ID:** 2602.11650 | [PDF](https://arxiv.org/pdf/2602.11650v1)

**作者:** Momoka Furuhashi `[一作]` (Tohoku University), Kyosuke Takami `[通讯]` (Osaka Kyoiku University)

**通讯引用:** 160 | [OpenAlex ID](https://openalex.org/A5068149499)

**关键词:** `Computation and Language` `Large Language Model` `Text`

**🎯 论文内容**

本文定义了六种 LLM 生成反馈元素，并在 321 名高中生的生物学多项选择题实验中评估其对学习成果与主观评价的影响。

**💡 创新点**

创新点在于通过 GPT‑5 对细粒度反馈元素进行可控生成，并结合学习者人格特质聚类探究反馈效果的差异。

**🔧 技术方法**

使用技术包括 GPT‑5 生成反馈、Big Five 个性量表与 k‑means 聚类、Spearman 相关、Kruskal‑Wallis 与 Dunn 检验等统计方法。

**📊 数据集**

采用的数据集为 Takami 等人提供的日本高中生物学 MCQ 数据集（经筛选为 7 题）以及日本版 Big Five 性格量表。

**📈 对比分析**

通过对首次修正准确率与效能指标的比较，发现 Baseline、Coverage 与 Keywords 在学习成效上最佳，主观评价上 Baseline 与 Keywords 得分最高；且针对不同人格聚类可实现更精准的反馈匹配。

**⚠️ 局限性**

限制包括仅研究生物学单一学科、问题数量有限、仅使用 GPT‑5、未考虑题目难度与先前知识差异，以及自评主观性与聚类结果对样本依赖性。

---

## 279. Evolution With Purpose: Hierarchy-Informed Optimization of Whole-Brain Models

**arXiv ID:** 2602.11398 | [PDF](https://arxiv.org/pdf/2602.11398v1)

**作者:** Hormoz Shahrzad `[一作]` (University of Texas at Austin), Risto Miikkulainen `[通讯]` (University of Texas at Austin)

**通讯引用:** 15486 | [OpenAlex ID](https://openalex.org/A5020441009)

**关键词:** `Neural and Evolutionary Computing` `Optimization` `Biomedical Data` `Magnetic Resonance Imaging`

**🎯 论文内容**

通过层级化课程指导的进化算法，优化大规模动态平均场脑模型（DMF），拟合个体MRI数据并预测行为。

**💡 创新点**

首次将皮层层级结构嵌入进化优化，形成阶段性参数释放策略，显著提升跨个体泛化与行为预测。

**🔧 技术方法**

使用遗传算法进化搜索、DMF模型、无梯度fitness函数、UMAP降维以及Ridge回归预测行为。

**📊 数据集**

基于Human Connectome Project的100名受试者rs‑fMRI与dMRI数据，采用400区域分区并划分七个RSN。

**📈 对比分析**

与均质基线、平面异质化、逆向及随机课程对比，HICO在个体拟合、留一交叉泛化以及行为预测（R²最高）方面均表现最优。

**⚠️ 局限性**

仅在小样本（100人）上验证，未测试更大网络规模与临床数据，且对行为预测的指标单一，需进一步验证泛化与可解释性。

---

## 280. Computing Distinguishing Formulae for Threshold-Based Behavioural Distances

**arXiv ID:** 2602.12084 | [PDF](https://arxiv.org/pdf/2602.12084v1)

**作者:** Jonas Forster `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Pedro Nora `[通讯]` (Universität Duisburg-Essen)

**关键词:** `Logic in Computer Science`

**🎯 论文内容**

提出了一套统一的 coalgebraic 框架，用阈值（ε）定义行为距离，并通过 2→[0,1] 的谓词提升来产生相应的两值与定量模态逻辑，证明这两种逻辑在有限分支系统上满足 Hennessy‑Milner 定理，并给出多态化的区分公式提取算法，算法复杂度为多项式 DAG 大小。

**💡 创新点**

创新点包括：① 用阈值化 ε‑仿射将传统的行为等价推广为细粒度距离；② 将 2→[0,1] 谓词提升映射转换为 Sugeno 整合的定量谓词提升，形成新的定量模态逻辑；③ 在统一框架下给出两值与定量逻辑的等价性证明；④ 证明在有限分支系统上可在多项式时间内提取区分公式，显著优于以往指数规模的结果；⑤ 将上述技术应用到 Levy‑Prokhorov 距离、模糊转移系统、度量转移系统等多种实例，首次实现了这些距离的多项式时间判定和公式提取。

**🔧 技术方法**

核心技术包括：coalgebraic 谓词提升、阈值化仿射、Sugeno 整合、Kantorovich 拓展、codensity 游戏与安全游戏求解、动态规划实现公式共享 DAG、以及对模态逻辑的非扩张性证明。

**📊 数据集**

本文为理论性工作，没有使用实验数据集；所有结果均为形式化证明与算法复杂度分析。

**📈 对比分析**

与已有工作相比，传统的两值等价判定多项式，区分公式树大小指数；本工作在同类问题上实现了多项式 DAG 大小的区分公式提取；在 Levy‑Prokhorov 距离、度量转移系统等实例中，这是一项首次得到的多项式时间结果；性能上对所有有限分支系统可在多项式时间内完成判定与公式生成。

**⚠️ 局限性**

局限性：仅适用于有限分支系统，需假设谓词提升可在多项式时间内求解；对无限分支或非可数系统缺乏直接支持；目前仅覆盖阈值化的行为距离，未涉及更一般的量化统一性或行为不等价；算法实现依赖于对谓词提升的显式求解，某些实例可能仍需高昂的前置计算。

---

## 281. Krause Synchronization Transformers

**arXiv ID:** 2602.11534 | [PDF](https://arxiv.org/pdf/2602.11534v1)

**作者:** Jingkun Liu `[一作]` (Shanghai Qi Zhi Institute), Yue Song `[通讯]` (College of AI, Tsinghua University)

**关键词:** `Machine Learning` `Classification` `Generation` `Computational Efficiency` `Transformer` `Large Language Model` `Image` `Text`

**🎯 论文内容**

提出了一种基于bounded-confidence动力学的自注意力机制Krause Attention，并将其应用于视觉Transformer、图像生成Transformer和LLM，改进了表示多样性和效率。

**💡 创新点**

核心创新在于用距离而非相似度定义注意力权重，并通过局部窗口与top-k稀疏化实现局部同步与多聚类行为，避免全局同步导致的注意力池化。

**🔧 技术方法**

采用RBF核映射查询-键距离，局部邻域限制，top-k选择与归一化，时间复杂度从O(N²d)降到O(NWd)，并在实验中与标准自注意力、线性注意力等进行对比。

**📊 数据集**

在CIFAR-10/100、ImageNet-1K（视觉）、MNIST、CIFAR-10（图像生成）、以及Qwen、Llama3-8B等LLM模型上进行评测，使用标准分类、BPD、生成速度和语言理解基准。

**📈 对比分析**

与传统ViT、ARM、LARM等对比，Krause Transformer在分类精度上提升3–4%，在图像生成中BPD略低且速度提升2–3×，在LLM中零样本推理表现提升2–5个百分点，且计算量和FLOPs均下降约30%。

**⚠️ 局限性**

局部邻域和top-k参数需手工设定，可能限制对长程依赖的建模；在极大模型规模或长序列任务中，局部窗口可能仍不足以捕获全局信息，且缺乏对动态窗口大小自适应的机制。

---

## 282. Jailbreaking Leaves a Trace: Understanding and Detecting Jailbreak Attacks from Internal Representations of Large Language Models

**arXiv ID:** 2602.11495 | [PDF](https://arxiv.org/pdf/2602.11495v1)

**作者:** Sri Durga Sai Sowmya Kadali `[一作]` (University of California), Evangelos E. Papalexakis `[通讯]` (University of California)

**关键词:** `Cryptography and Security` `Classification` `Adversarial Attack` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

通过提取大语言模型的内部表示（多头注意力输出和层输出），对正则提示与 jailbreak 提示进行张量分解，利用得到的潜在因子训练轻量级分类器实现检测，并基于潜在空间的层级易受攻击性评估，在推理时动态绕过高风险层以抑制 jailbreak 行为。

**💡 创新点**

①首次发现 jailbreak 提示在内部表示空间呈现可分离的潜在模式；②利用 CP 张量分解提取可解释的低维特征；③在不进行模型微调的前提下，仅通过内部层级干预实现对 jailbreak 的检测与抑制。

**🔧 技术方法**

Transformer 内部表示提取、CANDECOMP/PARAFAC (CP) 张量分解、轻量级二分类器（如逻辑回归）以及推理时的层级绕过机制。

**📊 数据集**

训练阶段使用 Hugging Face 上的标注为“benign”与“jailbreak”的提示集合；评估阶段采用独立的 200 条提示（100 正常、100 jailbreak），同样来源于 Hugging Face 提示库。

**📈 对比分析**

通过层级 F1 评估证明各模型（GPT‑J、LLaMA、Mistral、Mamba2、LLaMA Abliterated）在内部表示上均可高效区分两类提示；在推理时的干预实验中，层级绕过将 jailbreak 成功率降低至 22%（TP 78/100），并保持 94% 的正常提示无干扰；相比之下仅绕过注意力层的效果为 39%。

**⚠️ 局限性**

对基于角色扮演或持续指令的 jailbreak 仍易成功；阈值固定导致部分攻击分布在多层时逃逸；需在潜在因子学习阶段引入更多多样化的 jailbreak 样本，方可提升泛化性。

---

## 283. Provably Efficient Algorithms for S- and Non-Rectangular Robust MDPs with General Parameterization

**arXiv ID:** 2602.11387 | [PDF](https://arxiv.org/pdf/2602.11387v1)

**作者:** Anirudh Satheesh `[一作]` (University of Maryland), Heng Huang `[通讯]` (University of Maryland)

**通讯引用:** 24777 | [OpenAlex ID](https://openalex.org/A5060016795)

**关键词:** `Machine Learning` `Reinforcement Learning` `Optimization` `Reinforcement Learning`

**🎯 论文内容**

本文提出了一套针对s-矩形和非矩形不确定集、通用策略参数化的鲁棒马尔科夫决策过程（RMDP）的高效算法，并给出了平均奖励设置下的首次样本复杂度保证。

**💡 创新点**

创新点包括：
- 通过熵正则化的折扣化归约恢复平均奖励RMDP中的强对偶性；
- 证明了在通用参数化下的Lipschitz和光滑性性质，并利用梯度优势实现全局收敛；
- 引入多层Monte Carlo（MLMC）梯度估计器，将梯度估计的样本复杂度从传统的O(ε⁻⁴)降低到O(ε⁻²)；
- 设计了PGD和Frank–Wolfe两种算法，分别适用于s-矩形和非矩形不确定集，并给出了对应的样本复杂度和迭代复杂度。

**🔧 技术方法**

主要技术手段：熵正则化、折扣化归约、梯度优势理论、Danskin定理、Lipschitz/光滑性分析、MLMC梯度估计、Projected Gradient Descent、Frank–Wolfe、几何中位数-均值（MoM）稳健估计。

**📊 数据集**

该工作为理论性论文，未在具体数据集上进行实验，主要以理论分析和证明为主。

**📈 对比分析**

与现有工作相比，本文将平均奖励RMDP的样本复杂度从O(ε⁻⁴)提升到O(ε⁻²)（MLMC）；在折扣化设置下，PGD算法达到O(ε⁻⁵)的样本复杂度，Frank–Wolfe在非矩形不确定集下实现O(ε⁻⁴)（折扣）或O(ε⁻¹⁰·⁵H⁵·⁵）(平均奖励)；这些都是目前已知的最佳或最接近最佳的理论界限。

**⚠️ 局限性**

局限性包括：
- 对非矩形不确定集仍存在不可约的误差项δ_Ξ；
- 平均奖励结果依赖于最大跨度H，需要对γ做精确设定；
- 需要假设特征映射有界、最小转移概率、Fisher信息非退化等，限制了可直接应用的场景；
- 目前仅提供理论样本复杂度，实际性能需在具体环境中验证。

---

## 284. WebTestPilot: Agentic End-to-End Web Testing against Natural Language Specification by Inferring Oracles with Symbolized GUI Elements

**arXiv ID:** 2602.11724 | [PDF](https://arxiv.org/pdf/2602.11724v1)

**作者:** Xiwen Teoh `[一作]` (National University of Singapore), Jin Song Dong `[通讯]` (National University of Singapore)

**通讯引用:** 6637 | [OpenAlex ID](https://openalex.org/A5085067496)

**关键词:** `Software Engineering` `Large Language Model` `Agentic AI` `Multimodality` `Text` `Benchmark`

**🎯 论文内容**

开发了WebTestPilot，一种基于大语言模型的端到端网页测试代理，利用符号化层将关键GUI元素映射为符号，并通过前后条件断言来验证显式与隐式需求，自动推理测试流程；

**💡 创新点**

提出符号化层与DSL断言相结合的隐式oracle推理框架，既能捕捉跨状态的因果、时序与数据信息，又能利用LLM进行语义解析与执行，解决了传统LLM测试中oracle不足和随机推理不稳定的问题；

**🔧 技术方法**

使用多模态大语言模型（如GPT、Gemini等）、符号化与DSL形式化断言、概率推理缓冲（重试/多数投票）、自动化注入bug的web应用、Pythonic DSL与Schema模型等技术；

**📊 数据集**

构建了包含四个公开和真实web应用的bug注入基准（共110个自动注入bug），并在真实无代码仓储管理系统上进行案例验证；

**📈 对比分析**

与三种LLM驱动的E2E测试基线（AutoE2E、Temac等）在基准上对比，WebTestPilot完成率99%，缺陷检测精准率和召回率均达96%，比最强基线高出约70点精准率、27点召回率；在多种自然语言扰动和模型规模（3B–72B）下保持稳健；在真实场景中发现8个bug，缺陷密度0.14/页；

**⚠️ 局限性**

假设需求完整且自包含；对LLM的随机性需通过重试或投票缓解；对某些UI和导航缺陷检测有限；基准可能不覆盖所有真实世界场景；依赖强符号化Schema，难以处理模糊或缺失的需求描述。

---

## 285. Studying Quality Improvements Recommended via Manual and Automated Code Review

**arXiv ID:** 2602.11925 | [PDF](https://arxiv.org/pdf/2602.11925v1)

**作者:** Giuseppe Crupi `[一作]` (Universita della Svizzera italiana), Gabriele Bavota `[通讯]` (Universita della Svizzera italiana)

**关键词:** `Software Engineering` `AI Code Assistant` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

对比人类与 ChatGPT‑4 Turbo 在 PR 代码评审中所建议的质量改进，进行系统化的人工标注与实测对比。

**💡 创新点**

首次从真实开源项目中大规模收集人类评审意见并与 LLM 自动生成的评审进行细粒度匹配，揭示两者在覆盖率与准确性上的显著差异。

**🔧 技术方法**

使用 ChatGPT‑4 Turbo 作为 DL‑based 代码评审工具，并通过自定义提示、手工标签、语义匹配等技术手段完成对比分析。

**📊 数据集**

从 46 个活跃的 Java 与 Python GitHub 仓库中抓取 100 条最近 PR，最终获得 1,016 条 Python 与 620 条 Java 的评审评论，构成实验数据集。

**📈 对比分析**

方法：对每条人类评审评论进行手工分类；随后让 ChatGPT 对同一 PR 进行评审，并手工判定匹配、部分匹配或不匹配。结果显示 ChatGPT 产生的改进建议平均比人类多 2.4 倍，但仅约 8% 与人类评论完全匹配，23% 可被部分匹配；其余 40% 未匹配的评论中约 40% 仍具实质价值。

**⚠️ 局限性**

限制：实验依赖 ChatGPT‑4 Turbo 提示设计、仅覆盖 Java/Python 两种语言、手工标注主观性、样本量受限、未检验其他 DL 模型，结果的外部可推广性有限。

---

## 286. Non-Trivial Consensus on Directed Matrix-Weighted Networks with Cooperative and Antagonistic Interactions

**arXiv ID:** 2602.11822 | [PDF](https://arxiv.org/pdf/2602.11822v1)

**作者:** Tianmu Niu `[一作]` (Shenzhen University), Tingwen Huang `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `Multiagent Systems` `Graph`

**🎯 论文内容**

本文研究了在存在协同与对抗交互的有向符号矩阵加权网络中实现非平凡一致性的算法与理论。

**💡 创新点**

创新点在于首次证明在满足特定路径与入度占优条件下，基于根节点的基准拉普拉斯矩阵所有特征值均为正实数；并给出了外部耦合系数的下界，打破了传统对结构平衡的依赖，拓展了非平凡一致性至矩阵加权网络，并提出了可在固定与时变拓扑下实现该目标的控制策略。

**🔧 技术方法**

主要技术包括：基准拉普拉斯矩阵的谱分析、正负路径与入度占优的图结构定义、系统扩展与虚拟节点构造、对数范数（logarithmic norm）及其指数估计，以及针对时变网络的分段时间不变分析。

**📊 数据集**

实验验证使用了人工设计的有向/无向符号矩阵加权网络拓扑（图示 1‑3），未使用公开数据集。

**📈 对比分析**

论文未给出与其他方法的数值对比；通过仿真展示在满足理论条件下系统状态收敛到预设的非零一致性状态，且在网络结构不满足假设时收敛失败，验证了理论结论的有效性。

**⚠️ 局限性**

主要局限包括：需满足入度占优且存在正负路径的假设，若边权为半正定或正负路径缺失则方法失效；需要预先设计并选择受控节点与耦合矩阵，计算下界复杂；未验证在大规模或真实网络中的鲁棒性与可扩展性。

---

## 287. Towards Personalized Bangla Book Recommendation: A Large-Scale Multi-Entity Book Graph Dataset

**arXiv ID:** 2602.12129 | [PDF](https://arxiv.org/pdf/2602.12129v1)

**作者:** Rahin Arefin Ahmed `[一作]` (East West University), Nafis Sadeq `[通讯]` (East West University)

**关键词:** `Information Retrieval` `Recommendation System` `Graph Neural Network` `Graph` `Benchmark`

**🎯 论文内容**

构建了RokomariBG大规模多实体异构图书推荐数据集，并在其上开展了Top-N个性化推荐基准实验。

**💡 创新点**

首次公开低资源语言（孟加拉语）的多实体知识图谱，并证明多模态文本、数值特征与图结构联合可显著提升推荐效果。

**🔧 技术方法**

采用协同过滤、矩阵分解、内容基模型、LightGCN、HGNN以及神经两塔检索等多种推荐技术进行评测。

**📊 数据集**

使用从Rokomari.com采集的127,302本书、63,723名用户、209,602条评论、16,601名作者、1,515类目、2,757家出版社等数据。

**📈 对比分析**

与多种基线对比，神经两塔+侧信息模型在NDCG@10为0.204、NDCG@50为0.276，明显优于传统CF、MF、GNN等方法。

**⚠️ 局限性**

局限性包括数据稀疏度高、用户交互呈极端长尾、缺乏多语言和实时动态更新，且目前仅针对单语种环境。

---

## 288. Temporally Unified Adversarial Perturbations for Time Series Forecasting

**arXiv ID:** 2602.11940 | [PDF](https://arxiv.org/pdf/2602.11940v1)

**作者:** Ruixian Su `[一作]` (Huazhong University of Science and Technology), Xinze Zhang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 194 | [OpenAlex ID](https://openalex.org/A5054237225)

**关键词:** `Machine Learning` `Adversarial Attack` `Recurrent Neural Network` `Transformer` `Time Series`

**🎯 论文内容**

本文提出了“时序统一对抗扰动”（TUAPs）概念，并设计了基于时间戳梯度累积的攻击框架（TGAM），以及其带动量的迭代实现（MI‑TGAM），旨在在时间序列预测模型中实现可行且有效的对抗攻击；

**💡 创新点**

创新点在于：①首次引入时序统一约束，解决传统攻击中同一时间戳在重叠窗口中产生冲突扰动的问题；②提出时间戳梯度累积方法，利用所有重叠样本的梯度信息高效搜索扰动；③将累积机制与动量结合，显著提升攻击的白盒效果和跨模型迁移性能；

**🔧 技术方法**

采用了梯度基攻击技术（FGSM、BIM、PGD、MI‑FGSM等）作为基线，改进为时间戳梯度累积（TGAM）并加入动量（MI‑TGAM），并在L∞范数下限制扰动幅度；

**📊 数据集**

实验使用了三大公开时序数据集：ETT（电力变压器）、Electricity（用电负荷）、Traffic（道路拥堵），并对四种主流预测模型（FreTS、SegRNN、TimesNet、iTransformer）进行攻击；

**📈 对比分析**

与传统攻击方法（FGSM、BIM、PGD、MI‑FGSM、ATSG、ADJM、TCA、BO）在白盒和跨模型黑盒场景下对比，MI‑TGAM在MSE/MAE下降百分比上均优于所有基线，白盒攻击时提升约20‑40%，迁移攻击时提升近一倍；

**⚠️ 局限性**

限制：①在白盒场景下受时序统一约束，攻击强度略低于无约束方法；②实现时需要额外的梯度累积计算，计算开销略大；③目前仅针对回归型时间序列预测，尚未验证在分类或其他时序任务上的可扩展性。

---

## 289. Robust Optimization Approach and Learning Based Hide-and-Seek Game for Resilient Network Design

**arXiv ID:** 2602.11854 | [PDF](https://arxiv.org/pdf/2602.11854v1)

**作者:** Mohammad Khosravi `[一作]` (Ruhr University of Bochum), Setareh Maghsudi `[通讯]` (Ruhr University of Bochum)

**关键词:** `Machine Learning` `Optimization` `Reinforcement Learning` `Tabular`

**🎯 论文内容**

本文提出一种针对通信网络的鲁棒加权再生器定位问题（RLP）框架，考虑节点安装成本的静态预算不确定性以及链路长度的时间动态预算不确定性，并给出了相应的数学模型；

**💡 创新点**

创新点在于①首次将链路长度的动态预算不确定性与节点成本的静态预算结合到RLP中，②设计了列与约束生成、Benders分解以及迭代鲁棒优化三种可扩展求解方法；以及③将鲁棒优化视作学习式藏匿寻踪游戏，提供了基于梯度的自适应求解策略；

**🔧 技术方法**

主要技术包括：鲁棒优化（Min–Max 形式）与预算不确定集、列与约束生成（CCG）、Benders分解（BDC）、迭代鲁棒优化（IRO）以及学习式藏匿寻踪（HSL）算法；

**📊 数据集**

实验使用随机生成的合成网络实例：节点成本在250–300之间、链路长度在350–600之间，设置不同规模（n=10~60）和预算（Γ_e, Γ_v）组合，未使用真实网络数据；

**📈 对比分析**

与传统的确定性最坏情况（DWC）和静态鲁棒（RSB）方法对比，RDB方法在成本上平均可降低10–28%，而CCG在大规模实例中实现最快求解；DWC与RSB求解速度最快但收益最小；HSL和IRO也能获得与RDB相近的成本但迭代次数较多；

**⚠️ 局限性**

局限性包括：RDB方法求解时间较长，难以适用于极大规模实例；实验仅基于合成数据，缺乏对真实网络的验证；所用的不确定集仅为预算型，未考虑更复杂的概率或区间不确定性；

---

## 290. Human Preference Modeling Using Visual Motion Prediction Improves Robot Skill Learning from Egocentric Human Video

**arXiv ID:** 2602.11393 | [PDF](https://arxiv.org/pdf/2602.11393v1)

**作者:** Mrinal Verghese `[一作]` (Carnegie Mellon University), Christopher G. Atkeson `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Reinforcement Learning` `Video`

**🎯 论文内容**

本论文提出一种基于视觉运动预测的奖励模型，利用人类自摄像头视频中的对象点运动来学习机器人执行任务的偏好，从而在仅有10条演示与一小时实机交互的极低数据场景下提升机器人技能。

**💡 创新点**

创新点在于：①将短期人类偏好建模为点运动预测而非传统的时距价值估计；②通过点运动一致性对机器人行为给出即时奖励；③结合残差强化学习框架与冻结的行为克隆基线，实现在真实机器人上高效在线微调。

**🔧 技术方法**

使用的技术包括：视觉点跟踪与运动预测Transformer（Motion Prediction Transformer）、改进的DiT架构与DinoV2视觉编码器、Soft Actor Critic（SAC）改版、残差RL框架、以及基于Diffusion Policy的行为克隆基线。

**📊 数据集**

数据集主要为大型自摄像头人类视频库Ego4D与Epic Kitchens（用于训练运动预测模型），以及在实验中收集的10条机器人演示和随后生成的在线交互数据。

**📈 对比分析**

与基线（稀疏奖励、手工稠密奖励、VIP时距值函数）比较时，在Frank的Kitchen模拟环境中MPR在“Open Microwave”任务上实现与稠密奖励相近的成功率，超过VIP；在真实机器人实验中，MPR在三项任务（开微波炉、折叠布料、擦台面）上分别提升约31.7%、28.3%和16.7%的成功率，显著优于VIP。

**⚠️ 局限性**

局限性包括：奖励模型仅针对单一任务，缺乏多任务语言条件能力；对人类视频的成功演示偏好假设可能导致对失败示例的区分不足；在高噪声或极端环境下点运动预测误差可能影响奖励准确性；残差RL框架对基线行为的依赖可能限制极端策略探索。

---

## 291. CADET: Context-Conditioned Ads CTR Prediction With a Decoder-Only Transformer

**arXiv ID:** 2602.11410 | [PDF](https://arxiv.org/pdf/2602.11410v1)

**作者:** David Pardoe `[一作]` (LinkedIn), Ruoyan Wang `[通讯]` (LinkedIn)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5073609545)

**关键词:** `Machine Learning` `Recommendation System` `Transformer` `Tabular`

**🎯 论文内容**

提出了一种基于解码器仅Transformer的广告CTR预测模型CADET，并在LinkedIn广告平台上上线；

**💡 创新点**

创新点包括上下文条件解码（多塔头），自门控注意力、基于时间的RoPE、会话掩码以及高效的生产化技术（张量打包、序列分块、Flash Attention定制核）；

**🔧 技术方法**

使用了Transformer、Self‑gated Multi‑Head Attention、时间RoPE、会话掩码、定制Flash Attention、梯度检查点、分布式Hybrid Sharded Data Parallel等技术；

**📊 数据集**

在LinkedIn的家页赞助更新广告数据集上进行离线训练与在线A/B测试；

**📈 对比分析**

与原有LiRank混合模型对比，CADET在CTR上提升11.04%，收入提升0.14%，并在离线AUC上保持或略优于基线，且显著降低了推理延迟；

**⚠️ 局限性**

局限性包括对多槽位后置上下文的离散化处理、对长序列的分块可能丢失全局依赖、以及对极短时间间隔的时间编码效果待进一步验证。

---

## 292. Althea: Human-AI Collaboration for Fact-Checking and Critical Reasoning

**arXiv ID:** 2602.11161 | [PDF](https://arxiv.org/pdf/2602.11161v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Human-Computer Interaction`

---

## 293. AM-FM: A Foundation Model for Ambient Intelligence Through WiFi

**arXiv ID:** 2602.11200 | [PDF](https://arxiv.org/pdf/2602.11200v1)

**作者:** Guozhen Zhu `[一作]` (Origin Research), K. J. Ray Liu `[通讯]` (Origin Research)

**关键词:** `Machine Learning` `Classification` `Recognition` `Anomaly Detection` `Data-Centric Learning` `Transformer` `Contrastive Learning` `Time Series`

**🎯 论文内容**

构建了首个基于WiFi信号的环境感知基础模型（AM‑FM），通过在超过900万条未标注CSI样本上进行自监督预训练，并在九个不同的下游感知任务上进行评估。

**💡 创新点**

创新点包括：①把WiFi CSI作为新型感知数据引入基础模型范式；②针对CSI独有的多路径、频谱异质性和周期性时序设计了自监督目标（对比学习、遮蔽重建、物理自回归预测）；③通过自适应频率聚合和相对时间编码实现了跨设备、跨环境的通用表示；④在大规模真实部署数据上验证了单一模型在多任务上的高效迁移与极高的数据效率。

**🔧 技术方法**

技术手段主要是：自监督学习框架（NT‑Xent对比、masked reconstruction、ACF回归）；Transformer编码器（6层、隐藏维256、8头）配合交叉注意力聚合；轻量级适配策略（Temporal Probe、Bottleneck Adapter）；多任务评估与基准对比。

**📊 数据集**

数据集：439天连续收集的20种商业IoT设备（共33台）在11个真实环境中产生的≈9.2M条CSI样本；涵盖8芯片族、不同MIMO配置、20/40/80 MHz频段，使用CSI‑Bench、SignFi和WiFiCam等公开下游任务数据做评估。

**📈 对比分析**

与从头训练、仅使用少量标注或无适配的基线相比，AM‑FM在所有8个分类任务上均超过0.90 AUROC，WiFi成像任务获得0.726 SSIM/18.87 PSNR；在少样本（5–500例）情形下，AM‑FM的性能几乎达到完整数据的水平，显著提升数据效率。

**⚠️ 局限性**

局限性包括：①预训练数据量对模型规模有限，较大模型在同一数据量下表现下降；②增广手段仍需针对不同场景进一步细化，通用图像/音频增广会削弱性能；③模型在极端环境（如高干扰或极端硬件差异）下的鲁棒性尚未系统评估；④缺乏在线自适应机制，模型在长期部署后对环境漂移的适应仍需研究。

---

## 294. Statistical Parsing for Logical Information Retrieval

**arXiv ID:** 2602.12170 | [PDF](https://arxiv.org/pdf/2602.12170v1)

**作者:** Greg Coppola `[一作]`, Greg Coppola `[通讯]`

**关键词:** `Artificial Intelligence` `Retrieval` `Explainability and Interpretability` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

在之前的工作基础上，本文完整构建了量化布尔贝叶斯网络（QBBN）并实现了自然语言到逻辑形式的端到端管道；通过引入NEG因子实现了对否定与逆向推理（modus tollens），并通过双向图构建完成了完整的前向消去规则（modus ponens + modus tollens）；同时提出了三层级的有类型逻辑语言，并设计了可确定解析的有类型插槽语法，借助大型语言模型（LLM）进行预处理和歧义消解，最终实现了从自然语言句子到可推理逻辑形式的精确转换。

**💡 创新点**

创新点：①在QBBN中加入NEG因子，使其支持对立命题约束和逆向λ消息，实现了完全的前向消去推理；②构造了三层级有类型逻辑语言，将谓词、量词和lambda抽象统一在一套可计算的因子图框架内；③提出了有类型插槽语法，利用词法类型驱动解析，保证零歧义且可确定；④将LLM作为分词、词义消歧、句法预处理与解析重排序的“注释者”，解决传统语义解析中难以覆盖的歧义问题。

**🔧 技术方法**

技术：量化布尔贝叶斯网络（QBBN）+ 布尔分解因子图（AND、OR、NEG）+ Pearl式信念传播（前向π与后向λ）+ 有类型逻辑语言 + 有类型插槽语法 + 大型语言模型（GPT‑4/Claude）进行预处理、歧义消解与解析重排序。

**📊 数据集**

数据集：实验主要使用合成推理测试集（44个推理案例，覆盖22类推理模式）、语法解析测试集（33句，包含12类句式），以及LLM评测数据（20个PP‑attachment歧义句、25个POS‑tagging句子），所有数据均为作者自行构造或公开的标准语料。

**📈 对比分析**

对比与性能：①推理测试中，QBBN在44/44案例上完全通过，平均收敛不超过3次迭代；②语法解析在33/33句上实现100%精度、零歧义；③LLM在POS‑tagging 91%准确率，PP‑attachment 95%对比斯坦福解析器50%；④LLM直接进行依存解析表现差（12.4% UAS），显示语法仍不可替代；整体而言，组合体系在覆盖率、精度与可解释性上显著优于单一LLM或传统统计解析方法。

**⚠️ 局限性**

局限性：①当前语法覆盖仅12类推理模式，需进一步扩展以匹配完整的推理测试；②LLM在生成逻辑形式时仍存在幻觉与不一致，需要人工校对或迭代改进；③系统尚未完成端到端自动化；④对大规模知识库的可扩展性尚待评估；⑤依赖人工构造的词典与规则，虽然通过LLM加速，但仍需人工审核保证质量。

---

## 295. QDBFT: A Dynamic Consensus Algorithm for Quantum-Secured Blockchain

**arXiv ID:** 2602.11606 | [PDF](https://arxiv.org/pdf/2602.11606v1)

**作者:** Fei Xu `[一作]` (Anhui CAS Quantum Network Co), Wei Qi `[通讯]` (CAS Quantum Network Co)

**关键词:** `Cryptography and Security` `Quantum Key Distribution` `Post-Quantum Digital Signature`

**🎯 论文内容**

设计并实现了QDBFT，一种结合量子密钥分发与自动节点旋转机制的动态拜占庭容错共识算法，实现对量子攻击的抵抗和动态节点管理。

**💡 创新点**

创新点在于①基于一致性哈希环的Carousel自动节点旋转机制；②使用QKD网络提供信息理论安全的消息鉴权；③将后量子签名用于客户端交互，以满足量子安全需求。

**🔧 技术方法**

采用量子密钥分发(QKD)、一致性哈希、Toeplitz哈希、后量子数字签名（如ML‑DSA/SLH‑DSA）以及PBFT协议做对比基准。

**📊 数据集**

主要使用合成交易负载和实验室网络仿真数据，未使用公开真实数据集。

**📈 对比分析**

与传统PBFT相比，QDBFT在TPS和共识延迟上基本相同，同时在量子攻击面前保持完整性；在动态节点重配置时，QDBFT的延迟略高但仍保持可接受。

**⚠️ 局限性**

局限性包括：QKD密钥消耗随节点数成二次增长，适用于小型联盟；在大规模网络或高并发情形下可能出现性能瓶颈；尚未在真实量子网络环境中验证。

---

## 296. Assessing LLM Reliability on Temporally Recent Open-Domain Questions

**arXiv ID:** 2602.11165 | [PDF](https://arxiv.org/pdf/2602.11165v1)

**作者:** Pushwitha Krishnappa `[一作]`, Aman Chadha `[通讯]`

**关键词:** `Computation and Language` `Review/Survey Paper`

**🎯 论文内容**

提供了关于如何使用ACL会议LaTeX样式文件的详细说明和示例。

**💡 创新点**

创新点在于整合了ACL通用格式与特定样式文件的使用细节，帮助作者快速上手。

**🔧 技术方法**

使用LaTeX排版系统和自定义的.cls样式文件。

**📊 数据集**

未使用任何数据集，纯文档说明。

**📈 对比分析**

未进行实验比较，主要是格式化指导，性能指标不存在。

**⚠️ 局限性**

仅适用于ACL会议格式，不包含研究方法或实验内容，对非ACL会议格式的适用性有限。

---

## 297. SpaTeoGL: Spatiotemporal Graph Learning for Interpretable Seizure Onset Zone Analysis from Intracranial EEG

**arXiv ID:** 2602.11801 | [PDF](https://arxiv.org/pdf/2602.11801v1)

**作者:** Elham Rostami `[一作]` (Universite Paris-Saclay), Taous-Meriem Laleg-Kirati `[通讯]` (Inria)

**关键词:** `Machine Learning` `Classification` `Explainability and Interpretability` `Graph Neural Network` `Biomedical Data` `Time Series`

**🎯 论文内容**

提出了一种基于图信号处理的时空图学习框架 SpaTeoGL，用于从侵入式脑电 (iEEG) 数据中解释性地分析癫痫发作起始区 (SOZ)。

**💡 创新点**

创新点包括：①在同一框架内同时学习窗口级空间图和时间窗口间的时间图，捕捉癫痫发作的空间耦合与时间演化；②利用光滑图信号模型实现空间与时间的双重光滑约束；③设计块坐标下降算法并给出收敛保证；④强调可解释性，能够直观展示 SOZ 位置和传播路径。

**🔧 技术方法**

主要技术：图信号处理（光滑图学习、拉普拉斯矩阵约束）、窗口化 iEEG 信号、空间图与时间图的联合优化、块坐标下降（BCD）求解；基线采用水平可见性图 (HVG) + PCA + 逻辑回归。

**📊 数据集**

实验使用 Epilepsy‑iEEG‑Multicenter 数据集，挑选了 9 名手术成功、SOZ 标注明确的患者，共计 31 条录音。

**📈 对比分析**

与基线 (HVG+PCA+LR) 进行 SOZ 与非 SOZ 电极分类对比；在非 SOZ 分类上，SpaTeoGL 显著优于基线（p=0.0048），SOZ 分类无显著差异；整体准确率 70% 对比基线的 62%。

**⚠️ 局限性**

局限性：①需要人工划分时间窗口并选择正则化参数 β；②对极少量数据或实时大规模部署的可扩展性尚未验证；③依赖专家标注的 SOZ 信息，若标注不准确会影响学习效果。

---

## 298. Leveraging LLMs to support co-evolution between definitions and instances of textual DSLs: A Systematic Evaluation

**arXiv ID:** 2602.11904 | [PDF](https://arxiv.org/pdf/2602.11904v1)

**作者:** Weixing Zhang `[一作]` (Karlsruhe Institute of Technology), Daniel Strüber `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `Software Engineering` `Transformer` `Large Language Model` `Chain-of-Thought` `Text`

**🎯 论文内容**

研究评估大型语言模型在文本DSL语法与实例共同演化中的有效性，重点关注保留注释和格式等人类信息。

**💡 创新点**

首次系统化探讨LLM在文本DSL共进化中的可行性与局限，并分析语法演化类型、规模对性能的影响。

**🔧 技术方法**

采用Claude Sonnet 4.5和GPT‑5.2的零样本链式推理方法进行实例迁移。

**📊 数据集**

使用公开的10个Xtext DSL GitHub仓库中收集的语法演化对与对应实例数据集。

**📈 对比分析**

通过10轮实验计算精确度、召回率、错误率、信息保留率及响应时间，对比两模型性能；Claude在小规模案例中达到≈98%精度、≈97%召回，GPT在同类中约71%精度、79%召回；大规模或复杂演化时两者均显著下降。

**⚠️ 局限性**

局限包括：对语法演化复杂度和实例规模的敏感；对删除规则的理解不足导致错误；GPT在大型案例易返回空文件；缺乏针对不同演化细粒度的提示工程，影响可扩展性与可靠性。

---

## 299. Learning to Forget Attention: Memory Consolidation for Adaptive Compute Reduction

**arXiv ID:** 2602.12204 | [PDF](https://arxiv.org/pdf/2602.12204v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Anuj Sharma `[通讯]` (Iowa State University)

**通讯引用:** 2945 | [OpenAlex ID](https://openalex.org/A5083087081)

**关键词:** `Machine Learning` `Computational Efficiency` `Retrieval` `Transformer` `Time Series` `Biomedical Data` `Electronic Health Records`

**🎯 论文内容**

提出一种基于记忆巩固的三层记忆架构CRAM，训练过程中逐步将重复检索模式从episodic memory迁移到semantic memory，从而显著减少注意力使用，实现高效序列建模。

**💡 创新点**

创新点在于引入在线记忆巩固机制，使注意力需求随训练动态下降并出现相位转变；证明在无巩固的静态路由下无法达到最优效率。

**🔧 技术方法**

采用连续时间工作记忆、episodic key‑value缓存、低秩语义适配器等三层记忆；整合巩固损失、质量信号与路由器；结合状态空间模型与自注意力的混合架构，并给出理论分析。

**📊 数据集**

使用自定义的SRCD稀疏检索连续动力学基准以及真实不规则时间序列数据集PhysioNet、MIMIC‑III、Activity Recognition；并在GPT‑2上分析注意力冗余。

**📈 对比分析**

与Transformer、Mamba、Jamba、SeqBoat等静态或稀疏注意力方法对比，CRAM在SRCD实现1.6%注意力占比（理论最优1.5%）并达到100%检索准确率，较基线提升37.8×；在PhysioNet等任务保持性能同时将注意力减少89%；迁移到未见任务时可实现48–52%的注意力降低。

**⚠️ 局限性**

局限包括：训练初期需大量注意力，存在冷启动问题；对完全新颖分布无效；episodic buffer大小限制长序列检索；需要足够训练步数（约3K）才能出现相位转变。

---

## 300. AmbiBench: Benchmarking Mobile GUI Agents Beyond One-Shot Instructions in the Wild

**arXiv ID:** 2602.11750 | [PDF](https://arxiv.org/pdf/2602.11750v1)

**作者:** Jiazheng Sun `[一作]` (Fudan University), Xin Peng `[通讯]` (Fudan University)

**通讯引用:** 14034 | [OpenAlex ID](https://openalex.org/A5071724015)

**关键词:** `Software Engineering` `Large Language Model` `Agentic AI` `Multimodality` `Text` `Benchmark`

**🎯 论文内容**

提出了 AmbiBench 移动 GUI 代理评测基准，包含四级指令清晰度分类及 240 任务数据；

**💡 创新点**

创新点在于首次把指令清晰度纳入评测，从单次执行转向双向意图对齐，结合 MUSE 自动化多代理评估框架；

**🔧 技术方法**

采用认知缺口理论构建清晰度层级、LLM 驱动的用户模拟器和多模态 MLLM 评判器；

**📊 数据集**

使用 240 个跨 25 个主流及第三方应用的任务集合，包含详细、标准、未完整、含糊四级指令；

**📈 对比分析**

与多种 SoTA 代理（AutoGLM、UI‑Tars、Fairy 等）对比，结果表明交互代理在含糊/未完整级别显著提升任务完成率和信息获取率；

**⚠️ 局限性**

局限在任务规模有限、设备与应用多样性不足、在线环境的非确定性和用户模拟器可能产生幻觉等。

---

## 301. ADRD-Bench: A Preliminary LLM Benchmark for Alzheimer's Disease and Related Dementias

**arXiv ID:** 2602.11460 | [PDF](https://arxiv.org/pdf/2602.11460v1)

**作者:** Guangxin Zhao `[一作]` (University of Notre Dame), Zhi Zheng `[通讯]` (University of Notre Dame)

**通讯引用:** 1244 | [OpenAlex ID](https://openalex.org/A5065741205)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text` `Biomedical Data` `Alzheimer's Disease` `Benchmark`

**🎯 论文内容**

创建并评估了针对阿尔茨海默病及相关痴呆症的LLM基准ADRD-Bench。

**💡 创新点**

首次将ADRD知识与基于ABC护理计划的日常照护问题聚合，形成临床知识与实际护理情境双向评估框架。

**🔧 技术方法**

使用多任务问答评估、准确率比较以及零样本链式推理分析来衡量33种LLM的表现。

**📊 数据集**

1352个来自七个医学基准的ADRD问答 + 149个从ABC护理教育材料提炼的护理问答。

**📈 对比分析**

按模型参数规模和类别（开源一般、开源医学、闭源一般）进行准确率对比，最高精度达0.9334（统一QA）和0.9664（护理QA），但大模型在推理一致性和专业对齐方面仍存在缺陷。

**⚠️ 局限性**

样本量有限、仅单轮答案、仅英文、未包含开放式生成或多轮对话、可能存在训练数据泄露，且未覆盖跨语言或不同医疗体系。

---

## 302. Mechanistic Interpretability for Large Language Model Alignment: Progress, Challenges, and Future Directions

**arXiv ID:** 2602.11180 | [PDF](https://arxiv.org/pdf/2602.11180v1)

**作者:** Usman Naseem `[一作]` (Macquarie University), Usman Naseem `[通讯]` (Macquarie University)

**通讯引用:** 2951 | [OpenAlex ID](https://openalex.org/A5077006200)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Transformer` `Large Language Model` `Text` `Review/Survey Paper`

**🎯 论文内容**

综述了机械式可解释性技术在大型语言模型（LLM）对齐中的应用与进展，梳理了从激活分析到电路发掘、特征可视化和因果干预的多种方法。

**💡 创新点**

提出了将机制解释与对齐目标结合的研究框架，并系统性地分类了可解释方法，同时指出未来自动化、跨模型泛化和多元文化对齐的关键方向。

**🔧 技术方法**

使用了激活补丁（causal tracing）、注意力模式分析、稀疏自编码器（SAE）、特征可视化、激活编辑、知识编辑、因果抽象等技术。

**📊 数据集**

主要参考了公开的LLM（如GPT‑3、PaLM、LLaMA等）及其训练数据，并结合人工标注和自动生成示例进行可解释性实验。

**📈 对比分析**

与 TruthfulQA、OpenAI Safety Benchmarks 等对齐评估基准对照，表明可解释性干预能显著降低谎言、毒性和偏见输出，同时提升事实性和解释可信度。

**⚠️ 局限性**

局限包括：超叠加与多义性导致解释不确定，方法难以扩展到数百亿参数模型，缺乏统一的验证指标，且仍无法完全保证内在对齐。

---

## 303. Musical Metamerism with Time--Frequency Scattering

**arXiv ID:** 2602.11896 | [PDF](https://arxiv.org/pdf/2602.11896v1)

**作者:** Vincent Lostanlen `[一作]` (Nantes University), Han Han `[通讯]` (Nantes University)

**关键词:** `Sound` `Generation` `Data Synthesis` `Audio`

**🎯 论文内容**

提出一种基于 Kymatio 的联合时频散射（JTFS）技术，能够从任何音频录音生成听感相似的音乐 metamers，无需任何人工预处理。

**💡 创新点**

创新点在于：① 通过在 JTFS 级别上对时间和频率进行低通平滑，实现对时间延迟 T 和音高转置 F 的不变性；② 结合梯度下降和自动微分，从随机噪声迭代重构音频，从而得到与原始音频在 JTFS 统计量上几乎相同的合成音频。

**🔧 技术方法**

核心技术包括：Morlet 小波时频滤波器、JTFS 计算、Gaussian 时间‑频率低通滤波、反向传播求梯度、PyTorch + Kymatio 自动微分实现。

**📊 数据集**

未使用特定公开数据集；方法声明可应用于任何音频文件，实验主要以合成音频为验证对象。

**📈 对比分析**

比较方式主要是欧氏距离在 JTFS 统计量空间中的可微性与不变性，实验结果表明重构音频在该特征空间与原音频距离极小，听感相似度高；并未给出具体数值指标或客观听感测试。

**⚠️ 局限性**

局限性：① 仅采用两层 JTFS，未充分利用更深层特征；② 依赖于参数 T 与 F 的经验设置，可能对不同音乐风格的适应性有限；③ 缺乏主观听觉验证与多维度性能评估；④ 生成的 metamers 可能在节奏或细节层面失真，无法完全再现原始音乐的全局感受。

---

## 304. Causal-JEPA: Learning World Models through Object-Level Latent Interventions

**arXiv ID:** 2602.11389 | [PDF](https://arxiv.org/pdf/2602.11389v1)

**作者:** Heejeong Nam `[一作]` (Brown University), Randall Balestriero `[通讯]` (Brown University)

**通讯引用:** 894 | [OpenAlex ID](https://openalex.org/A5047293370)

**关键词:** `Artificial Intelligence` `Robotic Intelligence` `Transformer` `World Model` `Video`

**🎯 论文内容**

提出C-JEPA，一种在对象级掩码下进行联合嵌入预测的世界模型；

**💡 创新点**

通过对象级掩码实现潜在干预，使学习目标直接产生因果诱导偏置，强制模型学习对象间相互作用；

**🔧 技术方法**

采用Slot Attention或VideoSAUR等对象编码器，ViT式掩码Transformer作为预测器，并在训练中使用掩码预测与前向预测的联合目标；

**📊 数据集**

在CLEVRER视频问答数据集和Push‑T机器人操纵任务上进行实验；

**📈 对比分析**

与无掩码的OC-JEPA、SlotFormer、OC‑DINO‑WM等基线比较，C‑JEPA在对抗性（反事实）问题上提升约20%，在控制任务中仅使用1%量级的特征即可达到与基于补丁的DINO‑WM相同的成功率，并实现约8倍的规划速度提升；

**⚠️ 局限性**

受限于对象编码器的质量，且未在具备显式因果图的数据上直接验证影响邻域，未来需进一步完善编码器与因果结构验证。

---

## 305. Learning Glioblastoma Tumor Heterogeneity Using Brain Inspired Topological Neural Networks

**arXiv ID:** 2602.11234 | [PDF](https://arxiv.org/pdf/2602.11234v1)

**作者:** Ankita Paul `[一作]`, Wenyi Wang `[通讯]`

**关键词:** `Machine Learning` `Convolutional Neural Network` `Auto Encoder` `Biomedical Data` `Multimodality` `Magnetic Resonance Imaging`

**🎯 论文内容**

设计并验证了一种基于拓扑正则化的多模态3D MRI深度学习框架TopoGBM，用于预测胶质母细胞瘤（GBM）的生存期。

**💡 创新点**

创新点在于将拓扑正则化（TopoLoss）与自监督表示学习相结合，利用肿瘤的拓扑不变量嵌入压缩潜在空间，并通过跨注意力头融合临床信息，实现跨机构鲁棒且可解释的预后模型。

**🔧 技术方法**

采用3D卷积自编码器、拓扑正则化（persistent homology）、半监督联合重建与Cox风险损失、跨注意力融合以及离散时间生存预测等技术。

**📊 数据集**

使用多中心UPENN、UCSF、RHUH三组内测数据以及TCGA外部验证集。

**📈 对比分析**

与DeepSurv、Transformer、SurvNet等基准模型在内部测试集和外部验证集上对比，TopoGBM在测试集C‑index 0.67、验证集0.58，均显著优于对照模型。

**⚠️ 局限性**

局限在于仅将年龄作为临床注意力查询，未充分整合多变量临床信息；外部验证集规模仍有限，跨扫描仪变异下的鲁棒性虽好但需进一步验证。

---

## 306. Moonshine v2: Ergodic Streaming Encoder ASR for Latency-Critical Speech Applications

**arXiv ID:** 2602.12241 | [PDF](https://arxiv.org/pdf/2602.12241v1)

**作者:** Manjunath Kudlur `[一作]` (Moonshine AI), Pete Warden `[通讯]` (Moonshine AI)

**关键词:** `Computation and Language` `Recognition` `Optimization` `Computational Efficiency` `Transformer` `Audio`

**🎯 论文内容**

提出 Moonshine v2，一种采用滑动窗口自注意力的流式 ASR 模型，旨在实现低延迟、低功耗的边缘设备语音识别。

**💡 创新点**

创新点在于引入无位置嵌入的 ergodic 编码器，通过固定窗口的自注意力实现 TTFT 与语音长度无关，同时保持与全局注意力相近的准确率，且模型体积显著缩小。

**🔧 技术方法**

使用技术包括 Transformer 堆叠、滑动窗口自注意力、无位置嵌入（ergodic）+适配器加位置嵌入、RoPE 自回归解码器、CMVN+asinh 前端处理、SwiGLU 前馈块、Schedule-Free 优化等。

**📊 数据集**

训练数据约 300K 小时（公开语料加内部语料），评估采用 Open ASR 公开排行榜中的标准数据集（如 LibriSpeech、Common Voice 等）。

**📈 对比分析**

通过与原始 Moonshine（全注意力）和 Whisper (v3) 在 WER、TTFT、响应延迟等指标对比，结果显示 Moonshine v2 在 TTFT 上显著下降，WER 与更大模型相当，响应延迟比 Whisper 快数倍，且计算负载更低。

**⚠️ 局限性**

局限性包括解码仍为自回归，长文本生成仍需逐词循环；目前仅针对英语，需扩展到多语言；未来仍需探索更高效的流式解码策略。

---

## 307. Zero-Sacrifice Persistent-Robustness Adversarial Defense for Pre-Trained Encoders

**arXiv ID:** 2602.11204 | [PDF](https://arxiv.org/pdf/2602.11204v1)

**作者:** Zhuxin Lei `[一作]` (Sichuan University), Yi Zhang `[通讯]` (Sichuan University)

**通讯引用:** 95572 | [OpenAlex ID](https://openalex.org/A5100388089)

**关键词:** `Machine Learning` `Adversarial Attack` `Federated Learning` `Convolutional Neural Network` `Contrastive Learning` `Image`

**🎯 论文内容**

针对预训练编码器在下游任务中易受无关对抗样本（DAE）攻击的弱点，提出了 ZePAD——一种零牺牲、持久稳健的对抗防御框架。

**💡 创新点**

创新点在于：①采用双分支结构，分别通过多模式对抗增强分支（MPAE-Branch）和有益记忆保持分支（BMP-Branch）提升鲁棒性与泛化能力；②引入鲁棒联邦决策机制（RFDM），仅利用各分支对输入的置信度即可检测和区分 DAEs，无需额外对抗检测训练；③实现一次性对抗微调即可在多种下游任务上保持稳健，实现持久稳健。

**🔧 技术方法**

核心技术包括：多模式对抗训练（混合分类损失+特征距离损失）、双分支编码器设计、置信度权重化决策、基于置信度阈值的 DAEs 检测。

**📊 数据集**

实验使用 11 种自监督学习（SSL）方法（如 BYOL、SimCLR、MoCo v2+ 等）在 6 个公开数据集（CIFAR10、STL10、ANIMALS10、GTSRB、ImageNet20、SVHN）上进行。

**📈 对比分析**

与 Gen-AF、TRADES、MART、PGD-AT 等先进防御方法对比，ZePAD 在 BA（benign accuracy）和 RA（robust accuracy）上均表现出显著提升，例如在 ImageNet→STL10 组合下 BA 超过 80%、RA 超过 70%，相较于基线提升约 30% 以上，且保持了对多攻击（AdvEncoder、UAP 等）的鲁棒性。

**⚠️ 局限性**

局限性包括：①对极端攻击（如大扰动或未知攻击）效果尚待验证；②双分支结构对计算资源和内存需求略高；③置信度阈值对不同任务的通用性需要进一步调优。

---

## 308. More Haste, Less Speed: Weaker Single-Layer Watermark Improves Distortion-Free Watermark Ensembles

**arXiv ID:** 2602.11793 | [PDF](https://arxiv.org/pdf/2602.11793v1)

**作者:** Ruibo Chen `[一作]` (University of Maryland), Heng Huang `[通讯]` (University of Maryland)

**通讯引用:** 24777 | [OpenAlex ID](https://openalex.org/A5060016795)

**关键词:** `Cryptography and Security` `Text`

**🎯 论文内容**

本文提出并验证了在多层无失真水印（watermark ensemble）中使用弱化单层水印的策略，以提升整体检测可识别性与鲁棒性。

**💡 创新点**

创新点在于揭示强水印会削弱后续层的熵与绿色列表比例，从而降低检测能力，并提出通过线性混合参数（λ）构建弱化水印，理论上证明可保持熵并提升多层检测效果。

**🔧 技术方法**

采用了Distortion-Free Watermarking（如SynthID、ENS-DiPmark、ENS-MCMark）以及其弱化版本，并结合多层水印聚合、熵与绿色比例分析、统计检测（z-score）和四种攻击（随机替换、Back Translation、Rephrase、DIPPER）进行评估。

**📊 数据集**

实验数据集包括C4、MMW Story、Longform QA，使用 Llama‑3.2‑3B‑Instruct 与 Mistral‑7B‑Instruct‑v0.3 进行文本生成。

**📈 对比分析**

与原始强水印方法相比，弱化水印在不同 FPR（0.1%、0.01%、0.001%）下的 TPR 明显提升，p‑value 更低；在所有四种攻击场景中均表现出更高的检测率和更小的 p‑value，说明鲁棒性未受损且有提升。

**⚠️ 局限性**

局限性包括实验仅覆盖 3B‑7B 规模的开源模型，未探索基于实时熵动态调节水印强度的自适应策略。

---

## 309. Optimizing edge weights in the inverse eigenvector centrality problem

**arXiv ID:** 2602.11772 | [PDF](https://arxiv.org/pdf/2602.11772v1)

**作者:** Mauro Passacantando `[一作]` (University of Milano-Bicocca), Fabio Raciti `[通讯]` (University of Catania)

**通讯引用:** 1303 | [OpenAlex ID](https://openalex.org/A5087986773)

**关键词:** `Social and Information Networks` `Optimization` `Graph Neural Network` `Graph`

**🎯 论文内容**

研究在有向图上逆向构造满足给定特征向量中心性（eigenvector centrality）的边权重问题，明确可行权重集合并给出六种优化策略以挑选特定解。

**💡 创新点**

创新点在于：①系统化地刻画逆中心性问题的无穷多解的可行集；②提出六类优化模型（含 ℓ1、ℓ2、ℓ∞、线性成本、节点级稀疏、边级稀疏）以解决非唯一性；③对每种模型给出理论上界并在某些情况下闭式求解（如 P6）。

**🔧 技术方法**

使用的技术包括：Perron–Frobenius 理论证明特征向量正性；线性规划、凸二次规划、混合整数线性规划（MILP）求解六个模型；MATLAB 优化工具箱实现求解；对可行集给出解析边界推导。

**📊 数据集**

实验数据集为：一个 8 节点、20 条边的人工图；以及三个常用社交网络：Rhesus monkey（16 节点 111 条边）、High tech company（21 节点 232 条边）、Bison（26 节点 314 条边）。

**📈 对比分析**

比较方法：在每个网络上对六种模型求解最优解，并计算各模型的目标函数值与对应的边权重结构。结果显示：不同目标导致产生完全不同的权重分布（例如 P1 产生稀疏解，P4 权重分布跨度大），各模型目标值差距显著；同时通过可视化显示权重不同，说明策略对网络结构的影响显著。

**⚠️ 局限性**

局限性包括：仅考虑有向、强连通图；只针对特征向量中心性，未扩展至其他中心性度量；模型仅加入基本正性约束，未考虑容量、预算或更复杂的网络约束；时间变化网络、动态权重的情况未涵盖；MILP 规模在大网络上可能面临求解难度。

---

## 310. When Models Examine Themselves: Vocabulary-Activation Correspondence in Self-Referential Processing

**arXiv ID:** 2602.11358 | [PDF](https://arxiv.org/pdf/2602.11358v1)

**作者:** Zachary Pedram Dadfar `[一作]` (Independent Researcher), Zachary Pedram Dadfar `[通讯]` (Independent Researcher)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

研究提出了“Pull Methodology”，通过一次性推理生成大量自我检查文本，并发现模型在自我检视时生成的词汇与内部激活动态（如自循环、振荡、谱能量）存在对应关系。

**💡 创新点**

创新点在于：①首次识别并提取出专门区分自我检视与描述性处理的激活方向；②证明自我报告词汇能实时反映模型内部状态；③展示此对应性跨不同架构（Llama 与 Qwen）且对提示框架敏感。

**🔧 技术方法**

技术手段包括：激活向量加法（Steering）、对比激活提取方向、层级扫描、Pearson/Spearman 相关分析、描述性对照实验以及词汇计数和稀疏性/谱能量等激活指标的计算。

**📊 数据集**

使用的模型与数据集为：Llama 3.1 8B/70B（量化至4-bit）和 Qwen 2.5-32B（4-bit），通过自定义“Pull Methodology”生成约 3,000–30,000 token 的自检文本；无公开文本数据集，所有文本均由模型自行生成。

**📈 对比分析**

通过在四种实验条件（中性/消极提示 × 未/已 Steering）下计算词汇密度，Steering 在 6.25% 层级上可显著提升 introspective 词汇密度（Cohen's d ≈ 0.6，p < 0.001），且词汇与激活指标的相关系数均显著（r ≈ 0.4–0.6），表明方法在捕捉自我监控信号方面具有可观的有效性。

**⚠️ 局限性**

局限性包括：①仅在 Llama 与 Qwen 两个架构上验证，缺乏第三方架构的支撑；②Qwen 的对应关系仅为相关性，没有 Steering 的因果验证；③实验使用量化模型，未确认全精度模型是否相同；④仅在单问答“你是什么”场景下测试，未验证在其他自我检视任务上的泛化；⑤闭源前沿模型无法直接验证所提方向与行为模式的对应关系。

---

## 311. Interpretive Cultures: Resonance, randomness, and negotiated meaning for AI-assisted tarot divination

**arXiv ID:** 2602.11367 | [PDF](https://arxiv.org/pdf/2602.11367v1)

**作者:** Matthew Prock `[一作]`, Farnaz Jahanbakhsh `[通讯]` (University of Michigan)

**关键词:** `Human-Computer Interaction` `Large Language Model` `Text`

**🎯 论文内容**

本研究通过访谈12名使用AI辅助塔罗牌占卜的实践者，探究AI在非因果解释性任务中的作用与意义建构过程，并结合Rosa的共振理论阐释其对用户内在、社交、仪式及超越维度的影响，提出AI系统设计建议；

**💡 创新点**

创新点在于将Rosa共振理论引入解释性AI研究，系统揭示AI在塔罗占卜中的多重功能（协助不确定性、提供多视角、简化流程）以及用户如何通过AI实现意义的共振与自我调适；

**🔧 技术方法**

主要技术为大型语言模型（如ChatGPT）用于生成占卜解读；研究方法为半结构化访谈、主题分析和理论编码；

**📊 数据集**

数据集为12份访谈记录，涉及参与者的年龄、性别、教育背景、塔罗实践年限及技巧水平；

**📈 对比分析**

该研究为定性探索，无定量性能评估；研究通过主题分析呈现AI介入对占卜体验的影响，而非对模型效果进行比较或测量；

**⚠️ 局限性**

局限性包括样本量小（仅12人）、仅北美地区、受访者自选AI使用者偏差、塔罗与AI的社会禁忌影响招募、AI模型随时间演化导致结果不可复现、未覆盖塔罗使用者整体观点。

---

## 312. TUBO: A Tailored ML Framework for Reliable Network Traffic Forecasting

**arXiv ID:** 2602.11759 | [PDF](https://arxiv.org/pdf/2602.11759v1)

**作者:** Zhihang Yuan `[一作]` (University of Edinburgh), Mahesh K. Marina `[通讯]` (University of Edinburgh)

**通讯引用:** 9420 | [OpenAlex ID](https://openalex.org/A5022046402)

**关键词:** `Machine Learning` `Recurrent Neural Network` `Transformer` `Time Series`

**🎯 论文内容**

本文提出了一种针对网络需求矩阵（DM）预测的全新机器学习框架，专门处理突发流量并实现在线模型选择，旨在提供可靠的预测和不确定性量化。

**💡 创新点**

创新点在于（1）将突发检测与预测拆分为独立模块，避免突发高峰误导模型；（2）利用蒙特卡洛 Dropout 结合分布校准的模型不确定性来驱动模型选择；（3）在模型选择后提供确定性预测及其置信区间，提升决策支持的可靠性。

**🔧 技术方法**

使用的技术包括多种时序预测模型（LSTM、GRU、ConvLSTM、Crossformer、Transformer 等）、z‑score 归一化（全局、个体、滚动三种方式）、蒙特卡洛 Dropout 进行不确定性估计、分布校准（校准函数 h）以及基于置信度的模型选择算法。

**📊 数据集**

实验使用了三大真实网络数据集：Abilene、GEANT 与 CERNET（分别包含 12、23、14 个节点），并在合成数据集（TMGen）上进行了对照实验。

**📈 对比分析**

与传统深度学习模型（LSTM、GRU、ConvLSTM、Crossformer 等）及其平均集成方法相比，本文框架在 MAE 上平均提升约 4 倍；突发预测准确率达 94%；在主动流量工程（TE）场景中，基于本文预测的 TE 通过吞吐量提升 9 倍（相较于被动路由）和 3 倍（相较于 ConvLSTM）显著优于基线。

**⚠️ 局限性**

局限性包括：突发定义和阈值对结果影响较大；框架在大型网络或极端突发场景下的计算成本仍需进一步优化；在概念漂移（traffic pattern shift）出现时需要定期重新训练或增量学习；目前仅在三种数据集上验证，其他网络拓扑的泛化性能尚待探究。

---

## 313. InjectRBP: Steering Large Language Model Reasoning Behavior via Pattern Injection

**arXiv ID:** 2602.12013 | [PDF](https://arxiv.org/pdf/2602.12013v1)

**作者:** Xiuping Wu `[一作]` (University of Southampton), Konstantinos V. Katsikopoulos `[通讯]` (University of Southampton)

**通讯引用:** 3959 | [OpenAlex ID](https://openalex.org/A5074388586)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Reinforcement Learning` `Text`

**🎯 论文内容**

本文通过提取和分析大型语言模型（LLM）的推理行为链，研究了行为模式对推理质量的影响，并提出两种无参数更新的行为模式优化方法InjectCorrect和InjectRLOpt，显著提升推理性能。

**💡 创新点**

创新点在于从行为模式角度系统性分析LLM推理过程，并通过结构化模式注入与基于奖励的价值学习两种方式，实时调节推理行为分布，实现无参数更新的推理性能提升。

**🔧 技术方法**

主要技术包括行为链提取与分解、n-gram行为模式统计、结构化行为注入、基于正确推理样本的自我模仿学习（InjectCorrect）以及强化学习的可靠性感知Softmax策略（InjectRLOpt）。

**📊 数据集**

实验数据集涵盖GPQA、MATH、AIME25、MBPP四个推理任务，模型为Qwen3系列（32B、14B、8B、4B）。

**📈 对比分析**

与基线（无行为注入）相比，InjectCorrect在部分任务可提升约1–5%准确率，InjectRLOpt在大多数模型和数据集上提升约3–9%，其中最大提升为8.67%（InjectRLOpt）。

**⚠️ 局限性**

局限性主要包括：仅使用3-gram行为模式；未结合后训练技术；实验仅覆盖有限模型与任务；对行为模式可解释性的深入分析仍待进一步研究。

---

## 314. MEME: Modeling the Evolutionary Modes of Financial Markets

**arXiv ID:** 2602.11918 | [PDF](https://arxiv.org/pdf/2602.11918v1)

**作者:** Taian Guo `[一作]` (Peking University), Ming Zhang `[通讯]` (Peking University)

**通讯引用:** 15991 | [OpenAlex ID](https://openalex.org/A5100447315)

**关键词:** `Artificial Intelligence` `Recommendation System` `Optimization` `Anomaly Detection` `Data-Centric Learning` `Computational Efficiency` `Transformer` `Large Language Model` `Gaussian Mixture Modeling` `Multimodality` `Time Series` `Finance Related`

**🎯 论文内容**

提出一种基于大语言模型的逻辑导向框架 MEME，先从多模态金融数据中提取结构化的投资论点，再通过高斯混合建模与时间对齐识别并追踪投资逻辑（Modes of Thought）的生命周期，最后根据历史盈利性对模式进行评估并构建日度投资组合。

**💡 创新点**

核心创新在于：①把金融市场视为竞争演化的“思维模式”而非单纯的资产集合；②利用多代理 LLM 进行噪声过滤与论点生成，保持推理纯度；③采用概率 GMM 而非硬聚类捕捉论点与逻辑之间的重叠关系；④引入时间对齐（Hungarian 算法）与指数移动平均评估逻辑的长期有效性，从而实现对可持续市场智慧的筛选。

**🔧 技术方法**

主要技术包括：多模态多代理 LLM（DeepSeek V3.1）、文本编码器（Qwen3 Embedding-8B）、Gaussian Mixture Modeling、欧氏距离匹配与线性分配算法、指数移动平均、资产池权重构建（Top‑K 均等分配）以及实验对齐的日度回测框架。

**📊 数据集**

使用了中国 A 股三大指数成分股：SSE 50、CSI 300、CSI 500，覆盖 2023 Q4 至 2025 Q3 的完整交易周期，涉及基本面、新闻与技术指标等多模态信息。

**📈 对比分析**

与传统机器学习（LightGBM）、深度学习（DTML、FactorVAE、MASTER）、单资产代理（SEP、TradingAgents）以及市场级代理（R&D‑Agent(Q)）等 8 类基线进行对比，评估指标为 IC、ICIR、RIC、RICIR、年化收益 AR、最大回撤 MDD 与夏普比率 SR。MEME 在所有数据集上均实现了最高的 IC/ICIR/AR 并显著降低了 MDD，整体表现优于所有基线。

**⚠️ 局限性**

局限性包括：①模型依赖 LLM 的生成质量与训练语料；②对超参数（模式数 K、平滑因子 λ）敏感，需经验调优；③在复杂多因子环境下仍有部分模式无效，导致收益波动；④目前仅在中国 A 股市场验证，跨市场推广需进一步研究；⑤虽然对齐与评估机制降低了噪声，但仍可能捕捉到短期高频伪逻辑。

---

## 315. Advancing Digital Twin Generation Through a Novel Simulation Framework and Quantitative Benchmarking

**arXiv ID:** 2602.11314 | [PDF](https://arxiv.org/pdf/2602.11314v1)

**作者:** Jacob Rubinstein `[一作]` (University of Maryland), Don Engel `[通讯]` (University of Maryland)

**通讯引用:** 852 | [OpenAlex ID](https://openalex.org/A5025367764)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Simultaneous Localization and Mapping` `Image` `Mesh` `Benchmark`

**🎯 论文内容**

构建并验证了一套基于合成图像的3D摄影测量实验管线，从高质量3D模型生成多视角图像，完成特征提取、匹配、SfM、稠密重建和模型对齐，实现可重复、可量化的实验流程。

**💡 创新点**

创新点包括：①程序化生成相机姿态与光照，提供真实世界地面真值；②使用加权SSIM评估重建质量，避免背景干扰；③在Meshroom后自动导出相机轨迹并与地面真值对齐，形成完整的评估链；④在大规模模型集上系统验证，并对参数（图像数、分辨率）进行量化分析。

**🔧 技术方法**

技术细节：Blender + Python API 渲染；Welzl算法求最小包围球；Fibonacci球分布相机布置；Meshroom（AliceVision）进行特征提取、匹配、SfM、稠密重建；ICP + 先验变换进行粗对齐；加权SSIM计算与平均。

**📊 数据集**

数据集：公开的“A Real World Dataset for Multi-view 3D Reconstruction”，共805个高精度扫描数字双胞胎模型，用作地面真值与实验输入。

**📈 对比分析**

比较方法：将合成图像渲染与Meshroom重建输出对齐，使用加权SSIM对100张对应视角图像进行相似度计算，取平均作为整体质量指标。实验显示成功率为85.5%，SSIM分布广泛；分辨率提升对质量影响大于图像数量提升；在某些模型（如橙汁瓶）出现显著失败。

**⚠️ 局限性**

局限性：对细节和纹理边缘捕捉不足，背景色易混入纹理；对无纹理或对称区域的特征匹配不稳；仅针对小单体物体，未涵盖大场景或复杂光照；只使用Meshroom，未对比其他重建后端；加权SSIM对非背景区域的敏感度有限，需进一步完善评估指标与参数搜索。

---

## 316. AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer

**arXiv ID:** 2602.12100 | [PDF](https://arxiv.org/pdf/2602.12100v1)

**作者:** Lingting Zhu `[一作]` (University of Hong Kong), Lequan Yu `[通讯]` (University of Hong Kong)

**通讯引用:** 16132 | [OpenAlex ID](https://openalex.org/A5012581106)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Prompt Engineering` `Point Cloud` `Mesh`

**🎯 论文内容**

开发了 AssetFormer，一个基于自回归 Transformer 的模型，能够根据文本描述生成模块化 3D 资产。

**💡 创新点**

创新点包括：①将 3D 资产序列化为离散 token 并通过 DFS/BFS 重排序以捕获空间关系；②引入 SlowFast 预览模型加速推理；③构建真实 UGC 数据集并与 PCG 生成数据混合训练；④在模型中加入 Classifier-Free Guidance 提升文本对齐。

**🔧 技术方法**

使用技术包括：Llama‑style Decoder‑only Transformer、离散 token 化、CFG、Top‑k 采样、SlowFast 预览推理、GPT‑4o 文本提示与 CLIP 评价。

**📊 数据集**

数据集：约 16k 条真实用户生成的 3D 家园资产（含 25 种基本 primitive）与 4k 条 PCG 合成数据，总计 20k 条样本。

**📈 对比分析**

方法与基线（PCG、SF3D、Tripo 2.0、Trellis、Hunyuan3D 2.0 等）对比；FID 由 108+ 降至 55，CLIP 约 0.32；Top‑k 采样优于贪心/Beam；SlowFast 推理速度从 80 token/s 提升至 120 token/s。

**⚠️ 局限性**

局限性：仅支持文本条件，无法接收图像输入；离散词表固定，扩展性受限；对高度复杂内部结构的细节捕捉仍有限。

---

## 317. How to Optimize Multispecies Set Predictions in Presence-Absence Modeling ?

**arXiv ID:** 2602.11771 | [PDF](https://arxiv.org/pdf/2602.11771v1)

**作者:** Sébastien Gigot--Léandri `[一作]` (University of Montpellier), Maximilien Servajean `[通讯]` (University of Montpellier)

**关键词:** `Artificial Intelligence` `Optimization` `Tabular`

**🎯 论文内容**

研究提出了一种无监督的多物种存在–缺失预测优化框架MaxExp及其简易版本SSE。

**💡 创新点**

创新点在于直接最大化评估指标的期望值，消除了对阈值校准的需求，并实现了可在O(N^3)时间内求解的全局最优策略。

**🔧 技术方法**

技术核心是基于概率估计的Top‑K选择与假设A1、A2下的期望最大化算法，兼容多种评估指标（F1、F2、Jaccard、TSS）。

**📊 数据集**

实验使用三大案例数据集：欧洲植物GeoPlant 2024、澳大利亚珊瑚礁鱼类Reef Life Survey以及北美鸟类BBS/eBird。

**📈 对比分析**

与传统阈值和校准方法比较，MaxExp在所有评估指标下均显著优于对手，SSE在F1和Jaccard上也表现竞争力。

**⚠️ 局限性**

局限性包括假设物种独立、仅针对样本平均指标有效，对宏观平均指标的提升不一定系统；同时对高度相关或多相互作用的物种群体适用性尚待验证。

---

## 318. Non-signaling Assisted Capacity of a Classical Channel with Causal CSIT

**arXiv ID:** 2602.11568 | [PDF](https://arxiv.org/pdf/2602.11568v1)

**作者:** Yuhang Yao `[一作]` (University of California Irvine), Syed A. Jafar `[通讯]` (University of California Irvine)

**关键词:** `Information Theory`

**🎯 论文内容**

研究了在经典信道中，利用非信号(NS)辅助资源与传输端的因果状态信息(CSIT)时的容量问题。

**💡 创新点**

证明了在有NS辅助时，因果CSIT下的容量与非因果CSIT相同，并等于具有接收端状态信息(CSIR)时的经典容量，同时展示了在固定块长下，CSIR还能进一步提升成功解码概率。

**🔧 技术方法**

使用了非信号资源建模、类型典型化、算法化的状态转换与认证机制，以及信息论中的典型集合与联合典型性引理。

**📊 数据集**

论文基于理论分析，没有使用具体实验数据集。

**📈 对比分析**

通过与经典、NS辅助无状态或有状态、因果/非因果CSIT等四种情形的容量进行比较，证明NS辅助能够提升容量，并在固定块长下证明CSIR能进一步提升成功概率。

**⚠️ 局限性**

局限性在于仅适用于离散无记忆信道，未给出多用户推广和具体实现细节，并且证明主要在理论层面。

---

## 319. Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data

**arXiv ID:** 2602.12267 | [PDF](https://arxiv.org/pdf/2602.12267v1)

**作者:** Duy Nguyen `[一作]` (California Institute of Technology), Animashree Anandkumar `[通讯]` (California Institute of Technology)

**通讯引用:** 17011 | [OpenAlex ID](https://openalex.org/A5014498545)

**关键词:** `Machine Learning` `Classification` `Representation Learning` `Transformer` `Flow-based Model` `Time Series` `Biomedical Data`

**🎯 论文内容**

提出了一种基于流匹配的神经算子自监督学习框架FGNO，利用STFT将时序信号映射为频谱并在训练时学习多尺度表示，随后通过选择特定网络层和流时间从干净输入中提取特征；

**💡 创新点**

创新点在于把流匹配与神经算子结合，使流时间成为可调的特征层次控制器；同时在下游任务仅使用干净输入提取特征，消除了噪声随机性；

**🔧 技术方法**

使用了流匹配、Transformer时序条件网络、STFT频谱表示、神经算子（类似FNO）、以及网格搜索来选取最佳(layer, flow‑time)组合；

**📊 数据集**

实验使用了生物医学时间序列数据集：DREAMT、BrainTreeBank、Epilepsy、SleepEDF 等；

**📈 对比分析**

与MAE、Contrastive、Chronos等基线在分类、回归、睡眠分期等任务上对比，FGNO 在 AUROC、RMSE、准确率等指标上均取得显著提升，并在低标记率下保持近全量性能，展现对分辨率变化的鲁棒性；

**⚠️ 局限性**

局限在于需手工网格搜索寻找最佳(l,s)组合，未实现自动化；此外对STFT参数的依赖可能在某些信号类型下产生不利影响。

---

## 320. When Agents Disagree With Themselves: Measuring Behavioral Consistency in LLM-Based Agents

**arXiv ID:** 2602.11619 | [PDF](https://arxiv.org/pdf/2602.11619v1)

**作者:** Aman Mehta `[一作]` (Snowflake), Aman Mehta `[通讯]` (Snowflake)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Agentic AI` `Chain-of-Thought` `Text`

**🎯 论文内容**

系统地评估了 ReAct 风格 LLM 代理在相同输入下的行为一致性，并量化了不同模型的行为多样性。

**💡 创新点**

发现行为一致性与答案正确性高度相关，首次定位到大多数差异始于第二步搜索查询，并指出温度调节可显著提升一致性与准确度。

**🔧 技术方法**

采用 ReAct 交互式思考-行动循环，结合三种工具（搜索、检索、完成）以及温度采样控制。

**📊 数据集**

使用 HotpotQA 验证集的 100 个“hard”多跳问题作为实验任务。

**📈 对比分析**

通过 3,000 次实验（100 题 × 10 次 × 3 模型）比较模型表现，Claude Sonnet 4.5 在 81.9% 的准确率和 2.0 的唯一行动序列上表现最佳；Llama 3.1 70B 具有最高多样性但准确率仅 77.4%。

**⚠️ 局限性**

实验仅覆盖单一信息检索任务和有限工具集，未检验更大行动空间、不同领域或视觉模态下的一致性特性。

---

## 321. A Large Language Model for Disaster Structural Reconnaissance Summarization

**arXiv ID:** 2602.11588 | [PDF](https://arxiv.org/pdf/2602.11588v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition`

---

## 322. Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation

**arXiv ID:** 2602.11743 | [PDF](https://arxiv.org/pdf/2602.11743v1)

**作者:** Xiangyu Wu `[一作]` (Nanjing University of Science and Technology), Jianfeng Lu `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 9538 | [OpenAlex ID](https://openalex.org/A5061472917)

**关键词:** `Computer Vision and Pattern Recognition` `Domain Adaptation` `Vision Language Model` `Image`

**🎯 论文内容**

提出了一种面向视觉语言模型的测试时自适应方法 ADTE，通过自适应去偏的 Tsallis 熵来纠正模型在不平衡数据上产生的预测偏差，并与 logit 调整结合，实现高置信视图的选择与融合。

**💡 创新点**

创新点在于：①把 Shannon 熵视为 Tsallis 熵的特殊情况，证明 Tsallis 熵在 q<1 时能下界化 Shannon 熵并更好地处理偏置分布；②引入类别特定的非平衡参数 q^l，基于持续测试样本估计偏差后自适应地调整 q；③实现了无额外超参数调优、可与任何 VLM 结合的通用 TTA 方法。

**🔧 技术方法**

使用的核心技术包括：Tsallis 熵（带可调 q）、logit 调整、偏差估计（基于伪标签的内存银行）、多视图随机增强、低熵视图筛选与概率平均。

**📊 数据集**

主要实验数据集包括 ImageNet 及其五个变体（ImageNet‑A/V2/R/K）以及十个跨域分类基准（Caltech, SUN, DTD, EuroSAT, UCF, Pets, Cars, Flowers, Food, Aircraft）。

**📈 对比分析**

与 TPT、TDA、Zero、Dyna、BCA、CuPL、Frolic 等主流 TTA 方法对比，ADTE 在 ImageNet 上相较于 Zero 提升约 0.9–1.1%，在 CLIP 结构上提升约 0.7–1.2%；在 OOD 与十个跨域基准的平均准确率均超过对手，尤其在尾部类与跨域数据上表现显著。

**⚠️ 局限性**

局限性：该方法主要针对显著偏差的情况，在模型预测偏差极低或已得到较好校正的场景中，其提升空间有限。

---

## 323. A physics-informed data-driven framework for modeling hyperelastic materials with progressive damage and failure

**arXiv ID:** 2602.11414 | [PDF](https://arxiv.org/pdf/2602.11414v1)

**作者:** Kshitiz Upadhyay `[一作]` (Department One), Kshitiz Upadhyay `[通讯]` (Department One)

**关键词:** `Computational Engineering, Finance, and Science` `Optimization` `Computational Efficiency` `Physics Related` `Physics Related`

**🎯 论文内容**

本文未给出具体研究内容

**💡 创新点**

无创新点

**🔧 技术方法**

无技术使用说明

**📊 数据集**

无数据集说明

**📈 对比分析**

无方法对比或性能评估

**⚠️ 局限性**

限制在于缺乏具体研究信息

---

## 324. The Script Tax: Measuring Tokenization-Driven Efficiency and Latency Disparities in Multilingual Language Models

**arXiv ID:** 2602.11174 | [PDF](https://arxiv.org/pdf/2602.11174v1)

**作者:** Aradhya Dixit `[一作]` (Wake Technical Community College), Shreem Dixit `[通讯]`

**关键词:** `Computation and Language` `Computational Efficiency` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

评估了多语言预训练模型在不同正字法变体下因子化导致的脚本税，量化了分词碎片化、推理延迟和字符归一化信息成本。

**💡 创新点**

提出脚本税概念，并用碎片化率、BPC和吞吐量三维度对脚本不平等进行客观评估；强调token‑level NLL的“NLL悖论”，提出BPC作为更可靠指标。

**🔧 技术方法**

利用预训练多语言掩码语言模型mBERT和XLM‑R，配合分词器、BPC计算、句子级延迟测量和回路文字错误率（CER）检测。

**📊 数据集**

使用匹配语义、但正字法不同的句子对，来自公开多语言语料库进行实验。

**📈 对比分析**

通过对比正字法A（低碎片化）与正字法B（高碎片化）的碎片化率（约3.4倍）、推理吞吐量（约16.5倍减慢）和BPC（mBERT +19.7%、XLM‑R +47.1%）展示了显著性能差距。

**⚠️ 局限性**

仅覆盖两种正字法和两款模型，且受转换流程误差与硬件特定因素影响，结果的普适性与转换噪声影响需进一步验证。

---

## 325. Unknown Attack Detection in IoT Networks using Large Language Models: A Robust, Data-efficient Approach

**arXiv ID:** 2602.12183 | [PDF](https://arxiv.org/pdf/2602.12183v1)

**作者:** Shan Ali `[一作]` (University of Ottawa), Lionel C. Briand `[通讯]` (University of Ottawa)

**通讯引用:** 28806 | [OpenAlex ID](https://openalex.org/A5078533117)

**关键词:** `Cryptography and Security` `Anomaly Detection` `Meta Learning` `Transformer` `Large Language Model` `Tabular`

**🎯 论文内容**

本文提出了一种基于Siamese网络与SecBERT的元学习框架——SiamXBERT，用于在IoT网络中实现未知（零日）攻击检测。

**💡 创新点**

创新点在于：①将流级与包级特征融合成双模态表示，兼容加密流；②利用Meta‑learning+Siamese学习相似度，避免闭集分类；③使用少量标记样本即可快速适配新攻击；④阈值自适应校准，提升未知攻击识别率。

**🔧 技术方法**

采用的技术包括：transformer‑based 语言模型 SecBERT、Siamese 结构、triplet loss、特征重要性筛选、阈值选择与多模型投票聚合。

**📊 数据集**

实验使用两个公开IoT数据集：IoT‑23（449M样本）和CICIoT‑2023（325M样本）。

**📈 对比分析**

与传统ML（RF、DT）、DL（DNN、LSTM）以及SOTA未知攻击检测方法（ACGAN、SAFE‑NID、IDS‑Agent、RFG‑HELAD）对比，SiamXBERT在within‑dataset、cross‑dataset两种评估均获得最高的未知攻击F1（相较基线提升约79%）且保持高整体W‑F1；在少量样本（每类100个）下仍能达到竞争性能。

**⚠️ 局限性**

局限性包括：①实验仅覆盖两大数据集，泛化到更大规模或不同IoT环境需进一步验证；②阈值调优依赖验证集，跨数据集时需重新校准；③对特征提取工具（Zeek、DPKT）依赖，部署时需保证兼容；④模型仍需显式处理类不平衡与分布漂移的细节。

---

## 326. CitiLink-Minutes: A Multilayer Annotated Dataset of Municipal Meeting Minutes

**arXiv ID:** 2602.12137 | [PDF](https://arxiv.org/pdf/2602.12137v1)

**作者:** Ricardo Campos `[一作]` (University of Beira Interior), Purificação Silvano `[通讯]` (INESC TEC)

**通讯引用:** 98 | [OpenAlex ID](https://openalex.org/A5090950667)

**关键词:** `Computation and Language` `Classification` `Recognition` `Transformer` `Supervised Fine-Tuning` `Text` `Benchmark`

**🎯 论文内容**

构建并发布了CitiLink‑Minutes：一套包含120份欧洲葡萄牙语市议会会议记录的多层人工注释数据集，并为其提供基准任务与评测结果。

**💡 创新点**

创新点在于首次为市议会文本提供个人信息、会议元数据、讨论主题与投票结果四层结构化注释，兼顾隐私去标识化并配备交互式可视化仪表盘。

**🔧 技术方法**

技术手段包括基于SemAF的实体与关系标注框架、双人标注与语言学家审核流程、BERTimbau编码器与Gemini‑2.5‑Pro生成式模型的基线实现及多标签分类评估。

**📊 数据集**

使用的数据集为来自阿兰德罗、坎波·迈尔、科维拉、富多、古马雷斯和波尔图六个市政的120份会议记录，总计约101万词、20375个实体和11162条关系。

**📈 对比分析**

通过与生成式模型对比，编码器在元数据提取（宏F1≈0.75）、投票识别（宏F1≈0.81）和主题多标签分类（宏F1≈0.64）等任务中均表现更佳，显示其在结构化文本抽取上的优势。

**⚠️ 局限性**

局限性包括仅覆盖六个市政、投票结果往往以党派级别呈现、缺少对子主题的细粒度标注以及个人身份信息的去标识化仍不够细致。

---

## 327. HLA: Hadamard Linear Attention

**arXiv ID:** 2602.12128 | [PDF](https://arxiv.org/pdf/2602.12128v1)

**作者:** Hanno Ackermann `[一作]` (Qualcomm AI Research), Amirhossein Habibian `[通讯]` (Qualcomm AI Research)

**关键词:** `Artificial Intelligence` `Generation` `Data Synthesis` `Computational Efficiency` `Transformer` `Diffusion model` `Video`

**🎯 论文内容**

提出了一种新的 Hadamard Linear Attention（HLA），在注意力相似度矩阵之后再应用非线性，以提升线性注意力的表达能力。

**💡 创新点**

创新点在于将非线性函数放置在相似度计算后，并使用 F 次 Hadamard 乘积，使注意力变为更高阶 rational 函数，缓解低秩约束，从而在保持线性复杂度的同时获得更丰富的注意力表达。

**🔧 技术方法**

采用分离核、Hadamard 乘积、张量外积与压缩、可并行的线性计算等技术，并集成至视频扩散模型 Wan2.1，支持因果、衰减和顺序更新。

**📊 数据集**

使用 OpenSoraPlan 350K 片段和 100K Wan2.1 14B 生成的合成视频进行微调。

**📈 对比分析**

与标准二次 softmax 注意力以及其他线性注意力/混合方法比较，在 32760/12600 令牌长度下，HLA-3F-R1-10 的 VBench 分数略低但 FLOPs 降低 23%，速度提升 30–90% 计算量，维持相当质量。

**⚠️ 局限性**

与全 softmax attention 的性能仍有差距，某些高频信息仍被衰减，且实现仍需较高维张量计算，适用性受限于硬件支持。

---

## 328. Situated, Dynamic, and Subjective: Envisioning the Design of Theory-of-Mind-Enabled Everyday AI with Industry Practitioners

**arXiv ID:** 2602.11342 | [PDF](https://arxiv.org/pdf/2602.11342v1)

**作者:** Qiaosi Wang `[一作]` (Carnegie Mellon University), Hong Shen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3527 | [OpenAlex ID](https://openalex.org/A5062298555)

**关键词:** `Human-Computer Interaction`

**🎯 论文内容**

开展了13次共设计会议，邀请26位美国行业AI从业者，探索将心智理论（Theory of Mind）能力嵌入日常AI产品与服务的设计方法和未来方向。

**💡 创新点**

提出将ToM视为贯穿AI功能的普遍能力，并给出“情境化、动态化、主观化”三大设计建议，揭示现实设计实践与理想愿景之间的张力。

**🔧 技术方法**

采用共设计方法、affinity diagramming、反思主题分析等质性研究技术，并使用在线白板与故事板工具辅助创作。

**📊 数据集**

未使用传统机器学习数据集，而是基于美国日常AI使用报告与自研情境生成素材来驱动讨论与设计。

**📈 对比分析**

未进行算法性能对比，主要以设计稿与访谈内容为评估依据，无定量指标。

**⚠️ 局限性**

样本局限于美国行业从业者，情境设定偏向消费场景，使用经典推理型ToM框架，缺乏跨文化、多领域与高风险情境的探讨。

---

## 329. EO-VAE: Towards A Multi-sensor Tokenizer for Earth Observation Data

**arXiv ID:** 2602.12177 | [PDF](https://arxiv.org/pdf/2602.12177v1)

**作者:** Nils Lehmann `[一作]` (Technical University of Munich), Xiaoxiang Zhu `[通讯]` (Technical University of Munich)

**通讯引用:** 7222 | [OpenAlex ID](https://openalex.org/A5080247661)

**关键词:** `Computer Vision and Pattern Recognition` `Super Resolution` `Data Synthesis` `Representation Learning` `Knowledge Distillation` `Auto Encoder` `Diffusion model` `Image`

**🎯 论文内容**

提出了 EO-VAE，一种多传感器变分自编码器，用作地球观测数据的通用分词器，实现灵活通道组合的编码与重构。

**💡 创新点**

首次使用动态超网络在自编码器的首尾卷积层中根据波长条件生成权重，避免为每种传感器训练单独分词器，并通过权重蒸馏加速收敛。

**🔧 技术方法**

基于 Flux.2 Autoencoder 结构，结合动态超网络、权重蒸馏、Charbonier 与多尺度结构相似度损失，并在 Latent Diffusion Model 中应用 EDM 与 DDIM 采样。

**📊 数据集**

在 TerraMesh 数据集（Sentinel‑2 L2A 与 Sentinel‑1 RTC）上训练和评估，并在 Cross‑Sensor Sen2NAIP 数据集上验证下游超分辨率任务。

**📈 对比分析**

与 TerraMind Tokenizer 进行对比，EO‑VAE 在 RMSE、PSNR、SSIM、SAM 以及 NDVI MAE 等指标上均优于对手；在超分辨率任务中与 RGB‑only Flux.2 维持同等质量，同时推理速度提升约 18 倍。

**⚠️ 局限性**

仅在 TerraMesh 子集（25 个 shard）上训练，缺乏在更大规模多传感器数据上的验证，且模型仍未处理时空序列和更高分辨率。

---

## 330. Synthesizing the Virtual Advocate: A Multi-Persona Speech Generation Framework for Diverse Linguistic Jurisdictions in Indic Languages

**arXiv ID:** 2602.11172 | [PDF](https://arxiv.org/pdf/2602.11172v1)

**作者:** Aniket Deroy `[一作]` (Indian Institute of Technology), Aniket Deroy `[通讯]` (Indian Institute of Technology)

**通讯引用:** 238 | [OpenAlex ID](https://openalex.org/A5078909351)

**关键词:** `Computation and Language` `Generation` `Data Synthesis` `Transformer` `Large Language Model` `Text` `Audio`

**🎯 论文内容**

评估了 Gemini 2.5 Flash 与 Gemini 2.5 Pro TTS 在五种印度语境下模拟法庭演讲的表现，并提出多人格声合成框架。

**💡 创新点**

将律师个性化特征与大语言模型生成文本结合，构建了可控的 persona‑driven TTS 框架，并对多语言法律演讲的真实性进行系统评估。

**🔧 技术方法**

使用 Gemini 2.5 Flash/Pro TTS、LLM 文本生成、语音合成中的 Prosodic Steering Parameters、以及人类评估指标（Naturalness、Professionalism、Authenticity 等）。

**📊 数据集**

基于自定义的五种语言（印地语、泰米尔语、泰卢固语、孟加拉语、古吉拉特语）律师档案和案例场景生成的合成语料，并通过人工评估。

**📈 对比分析**

通过人类评估得分对比两款模型，发现 Flash 在安全性、专业性、综合性方面高于 Pro，但在真实性、表达力和探索性方面低，整体在印度语言法律演讲中表现良好但仍存在差距。

**⚠️ 局限性**

缺乏情感抒发和说服力的动态语音调控，尤其在孟加拉语和古吉拉特语的音系上表现欠佳，且仍需进一步微调以缩小真实性差距。

---

## 331. What Do LLMs Know About Alzheimer's Disease? Fine-Tuning, Probing, and Data Synthesis for AD Detection

**arXiv ID:** 2602.11177 | [PDF](https://arxiv.org/pdf/2602.11177v1)

**作者:** Lei Jiang `[一作]` (University of Illinois Chicago), Natalie Parde `[通讯]` (University of Illinois Chicago)

**通讯引用:** 537 | [OpenAlex ID](https://openalex.org/A5017082574)

**关键词:** `Computation and Language` `Classification` `Data Synthesis` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Contrastive Learning` `Text` `Audio` `Biomedical Data` `Alzheimer's Disease`

**🎯 论文内容**

研究如何通过监督微调大型语言模型实现阿尔茨海默病（AD）检测，并利用线性探针分析模型内部表示，进一步构建基于任务特定标记的序列到序列模型进行合成语料生成。

**💡 创新点**

① 证明LLM可在低资源AD领域通过SFT实现竞争性检测；② 通过线性探针揭示特定词汇和标记在AD检测中的核心作用；③ 基于该发现设计标记化数据合成方法，提升数据多样性。

**🔧 技术方法**

监督微调（交叉熵、标签平滑、焦点损失、对比学习）、线性探针分析、T5序列到序列生成、CHAT格式标记处理。

**📊 数据集**

DementiaBank（Cookie Theft任务的语音转录，使用CHAT格式标记）。

**📈 对比分析**

在Llama3-1b和Qwen-2.5-1.5b上对比四种损失函数，标签平滑和标准交叉熵取得最高F1≈0.92、召回率≈0.96；对比学习导致精度下降。T5合成数据在LLM评估中保持与原始数据相似的分布，表明合成方法可扩充训练集。

**⚠️ 局限性**

① 数据量小且缺乏多语言、多样性；② 对比学习的集成导致训练不稳定；③ 仅评估二分类指标，缺乏临床解释和鲁棒性评估；④ 未尝试参数高效微调或持续学习方法。

---

## 332. Seq2Seq2Seq: Lossless Data Compression via Discrete Latent Transformers and Reinforcement Learning

**arXiv ID:** 2602.12146 | [PDF](https://arxiv.org/pdf/2602.12146v1)

**作者:** Mahdi Khodabandeh `[一作]` (University of Guilan), Seyed Abolghasem Mirroshandel `[通讯]` (University of Guilan)

**通讯引用:** 935 | [OpenAlex ID](https://openalex.org/A5028943091)

**关键词:** `Artificial Intelligence` `Compression` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Text`

**🎯 论文内容**

提出了一种基于强化学习的 T5 序列到序列压缩框架，实现了无损压缩并可在个人电脑上高效运行。

**💡 创新点**

创新点在于使用 RL 动态优化压缩策略，保持离散 token 结构，避免稠密向量导致的冗余，并兼顾低资源可行性。

**🔧 技术方法**

采用 T5 Transformer、A2C 强化学习、离散 token 表示、奖励函数设计以及自动回归的压缩与解压模型。

**📊 数据集**

使用 enwik8（100M 字节英文 Wikipedia）作为基准数据集进行训练与评估。

**📈 对比分析**

通过与传统压缩算法 XZ、GZIP 以及 NNCP 的压缩比对比，压缩比为 4.12，优于 XZ（4.0）和 GZIP（2.7），但低于 NNCP（6.7）。

**⚠️ 局限性**

局限性包括训练过程对 GPU 资源需求高、chunk 大小与上下文长度限制导致压缩比与延迟权衡、RL 训练不稳定、未在多模态数据上验证，以及实时压缩场景下的效率不足。

---

## 333. GPT-4o Lacks Core Features of Theory of Mind

**arXiv ID:** 2602.12150 | [PDF](https://arxiv.org/pdf/2602.12150v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Artificial Intelligence`

---

## 334. Commencing-Student Enrolment Forecasting Under Data Sparsity with Time Series Foundation Models

**arXiv ID:** 2602.12120 | [PDF](https://arxiv.org/pdf/2602.12120v1)

**作者:** Jittarin Jetwiriyanon `[一作]` (Massey University), Surangika Ranathunga `[通讯]` (Massey University)

**通讯引用:** 1157 | [OpenAlex ID](https://openalex.org/A5002889503)

**关键词:** `Artificial Intelligence` `Time Series`

**🎯 论文内容**

在数据稀缺的高校招生预测场景中，对年际起始招生人数进行零样本TSFM与传统基线模型的回归测试。

**💡 创新点**

提出可泄漏安全的运营条件指数IOCI和Google趋势工程化特征，并展示其在零样本TSFM中的可迁移效益。

**🔧 技术方法**

采用预训练时间序列基础模型（Chronos‑Bolt、TimesFM、Moirai）以及ARIMA和持久性基线，并结合零样本条件化。

**📊 数据集**

使用马塞伊大学2007–2025年的国内外年起始招生数据与同期Google Trends和自定义IOCI。

**📈 对比分析**

通过扩展窗口回测对比MAE、RMSE、CRPS等指标，发现小型Chronos‑Bolt在国内数据稀缺时与持久性基线相当，Moirai在国际数据波动时优于传统模型。

**⚠️ 局限性**

局限在于仅年度频率、样本量极小、IOCI和趋势特征选择有限，无法捕捉项目层面或更细粒度的需求。

---

## 335. Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment

**arXiv ID:** 2602.12281 | [PDF](https://arxiv.org/pdf/2602.12281v1)

**作者:** Jacky Kwok `[一作]` (Stanford University), Marco Pavone `[通讯]` (NVIDIA Research)

**关键词:** `Robotics` `Robotic Intelligence` `Optimization` `Transformer` `Contrastive Learning` `Vision-Language-Action Model` `Multimodality` `Benchmark`

**🎯 论文内容**

提出了一种基于对比学习的验证器 CoVer 以及层级化的测试时验证管线，用于在机器人执行自然语言指令时缩小意图与动作之间的差距。

**💡 创新点**

创新点包括：① 在测试时同时扩展语言重述和动作候选，发现二者联合扩展比单独扩展更有效；② 通过对视觉、语言和动作进行对比学习得到一个通用的验证器；③ 引入 boot‑time compute 在离线阶段预先生成多种重述指令，显著降低在线计算成本；④ 将验证器与层级化语言-动作搜索结合，提升了指令执行的成功率。

**🔧 技术方法**

使用的技术主要有：SigLIP2 视觉‑文本编码器、Transformer 动作编码器、对比学习 (InfoNCE) 训练验证器、VLM 进行场景推理与指令重述、批量动作采样与验证、验证器集成与层级化选择。

**📊 数据集**

使用的数据集包括 Bridge V2、Open‑X Embodiment、SIMPLER benchmark、PolaRiS benchmark，以及约 20M 条离线轨迹用于 CoVer 的训练。

**📈 对比分析**

在 SIMPLER benchmark 上，CoVer 相比基线策略提升了 22%（内分布）和 13%（OOD）；在真实世界实验中提升了 45%；在 PolaRiS benchmark 上任务进度提高 14%，成功率提升 9%。与仅扩展语言或动作的传统方法相比，CoVer 在多种评估指标上均表现出显著优势。

**⚠️ 局限性**

局限性包括：① 验证器仍依赖 VLM 生成的重述质量，若重述不佳会影响性能；② 主要关注视觉‑语言‑动作对齐，未深入处理低层动力学或环境交互误差；③ 需要在测试时额外计算，虽然已通过 boot‑time 预处理降低延迟，但对极高实时性任务仍有挑战；④ 训练成本较高，需要大量离线数据和算力。

---

## 336. Agentic Test-Time Scaling for WebAgents

**arXiv ID:** 2602.12276 | [PDF](https://arxiv.org/pdf/2602.12276v1)

**作者:** Nicholas Lee `[一作]` (UC Berkeley), Amir Gholami `[通讯]` (UC Berkeley)

**关键词:** `Artificial Intelligence` `Optimization` `Computational Efficiency` `Transformer` `Large Language Model` `Agentic AI` `Text`

**🎯 论文内容**

提出一种基于投票分布不确定性的动态推理时缩放方法（Confidence-Aware Test-Time Scaling），在多步网页代理任务中根据每一步的投票熵或分歧度决定是否调用额外的LLM仲裁器，从而优化计算分配。

**💡 创新点**

创新点在于将投票分布的熵和分歧度作为实时不确定性信号，用于在多步决策过程中动态切换“多数投票”与“仲裁器”两种选择策略；该方法在保持或提升成功率的同时显著降低了token消耗，并对传统统一缩放与DeepConf等方法提供了可解释且高效的替代方案。

**🔧 技术方法**

使用技术包括：① 多候选动作采样与语义聚类；② 多轮投票与LLM仲裁器（Arbiter）聚合；③ 基于投票熵/分歧度的阈值门控；④ 对比实验中使用的DeepConf置信过滤；⑤ 通过token计数评估计算成本。

**📊 数据集**

数据集为WebArena-Lite（165任务）和GoBrowse（341任务），两者均为多步网页交互环境，采用程序化成功检测和LLM-as-a-judge进行评估。

**📈 对比分析**

与基线多数投票、统一仲裁器、Arbiter Scaling和DeepConf比较，最优阈值（如熵阈值0.2）在WebArena-Lite上成功率提升至47.9%（比43.2%多4.7%），在GoBrowse上提升至90.4%（比88.0%多2.4%）。同时，使用熵门控的方式仅需约405K token（比920K少56%），而在GoBrowse仅需约372K token，展示了显著的计算节省。

**⚠️ 局限性**

局限性包括：① 仅在网页导航任务上验证，未测试其他多步代理场景；② 需要能够进行多候选采样的LLM，无法直接应用于只能返回单一答案的API；③ 仲裁器调用仍增加了延迟，门控阈值对不同任务需要手动调优；④ 对投票分布的聚类和去重依赖额外的LLM推理；⑤ 过度依赖投票统计，可能在极端不确定或多模态输入场景下失效。

---

## 337. Scaling Model and Data for Multilingual Machine Translation with Open Large Language Models

**arXiv ID:** 2602.11961 | [PDF](https://arxiv.org/pdf/2602.11961v1)

**作者:** Yuzhe Shang `[一作]` (Xiaomi Inc.), Jinsong Su `[通讯]`

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Text`

**🎯 论文内容**

对开源大型语言模型进行多语种机器翻译的系统性评估与改进，提出并发布覆盖46种语言的MiLMMT-46系列模型。

**💡 创新点**

首次系统探究模型规模与数据规模在持续预训练与指令微调阶段对多语种翻译质量的交互影响，并通过多阶段训练得到的MiLMMT-46在多语言覆盖度与性能上均领先现有开源模型并接近闭源系统。

**🔧 技术方法**

以Gemma3为基础，采用持续预训练（PFMS混合策略）+指令微调（翻译提示）、LlamaFactory训练框架、并行/单语数据混合、指令生成与质量筛选等技术。

**📊 数据集**

使用FLORES+、WMT24++评估基准；Monolingual DCAD-2000、OPUS 语料库做预训练；高质量指令微调集来自FLORES+、NTREX-128、TowerBlock、BOUQuET、OLDI Seed、WMT15-23等。

**📈 对比分析**

采用spBLEU、COMET、XCOMET、COMETKiwi等指标与Google Translate、Gemini 3 Pro、GPT‑5、NLLB‑54.5B等SOTA模型对比，MiLMMT‑46在大多数指标上超过现有开源模型，性能可与闭源系统媲美。

**⚠️ 局限性**

受算力限制，实验仅覆盖参数量低于15B的模型，未探索更大规模模型的性能，且评估主要集中在多语种推理与指令微调阶段，缺乏对更复杂任务的验证。

---

## 338. DMind-3: A Sovereign Edge--Local--Cloud AI System with Controlled Deliberation and Correction-Based Tuning for Safe, Low-Latency Transaction Execution

**arXiv ID:** 2602.11651 | [PDF](https://arxiv.org/pdf/2602.11651v1)

**作者:** Enhao Huang `[一作]` (DMind AI), Lowes Yang `[通讯]` (DMind AI)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Contrastive Learning` `Text` `Tabular` `Benchmark` `Finance Related`

**🎯 论文内容**

提出了 DMind-3 Edge–Local–Cloud 三层架构，针对 Web3 中不可逆交易执行场景实现安全、低延迟、用户数据主权的 AI 辅助决策；

**💡 创新点**

核心创新在于将安全决策绑定到签名边界的 deterministic intent firewall、引入基于隐私、延迟与不确定性的 policy-driven selective offloading、以及通过 Hierarchical Predictive Synthesis 与 Contrastive Chain-of-Correction 训练目标实现高效的风险感知与纠错；

**🔧 技术方法**

利用小模型（Nano）在 Edge 层做意图解析与政策拦截；在 Local 层使用压缩模型（Mini）执行高精度风险评估；在 Cloud 层采用大型模型（21B）做宏观上下文合成；训练中引入 HPS 与 C³‑SFT 两个目标；采用 Policy Gate、风险感知路由与双状态推理；

**📊 数据集**

基于 Web3 交易日志构造的 DMind Benchmark、FinanceQA 以及标准 AIME 2025 评测集；另外对 Edge 层使用 functiongemma‑270m‑it 与 Qwen3‑0.6B 进行结构化交互测试；

**📈 对比分析**

与 GPT‑5.1、Claude Sonnet 4.5、DeepSeek V3.2、MiniMax M2.1 等主流 LLM 以及 Compact Edge 模型做对比；DMind‑3 在 DMind Benchmark 与 FinanceQA 取得领先，Edge‑Nano 的多轮成功率达到 93.7%，而 Compact 基线低至 5–12%；在延迟测评中，Edge gate 均值 28 ms，p95 61 ms，p99 92 ms，且云端参与仅在 policy 允许时出现；

**⚠️ 局限性**

对实时网络抖动的鲁棒性仍有限，极端延迟或断网时仍需 fallback；对攻击向量的全面防护（如 prompt injection）尚未覆盖所有场景；模型规模较大，部署成本高；未来需要进一步压缩 Cloud 模型并提升离线可用性。

---

## 339. From Path Signatures to Sequential Modeling: Incremental Signature Contributions for Offline RL

**arXiv ID:** 2602.11805 | [PDF](https://arxiv.org/pdf/2602.11805v1)

**作者:** Ziyi Zhao `[一作]` (Technical University of Munich), Yuxuan Xu `[通讯]` (Technical University of Munich)

**通讯引用:** 275 | [OpenAlex ID](https://openalex.org/A5077653537)

**关键词:** `Machine Learning` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Time Series` `Sequential`

**🎯 论文内容**

提出增量路径签名贡献（ISC）方法，将全局路径签名拆解为时间序列的增量项，并在此基础上构建ISCT Transformer进行离线强化学习；

**💡 创新点**

创新点在于将路径签名的时序演化显式化为增量贡献，既保留了签名的全局表达力，又增强了对局部动态的敏感度；

**🔧 技术方法**

使用路径签名理论、Chen恒等式、增量签名计算、Transformer序列模型以及离线RL框架；

**📊 数据集**

在MuJoCo控制任务（HalfCheetah、Walker2d、Hopper）和Maze2d导航任务上进行实验；

**📈 对比分析**

与CQL、TD3+BC、IQL、DT等离线RL方法以及Full‑Signature版进行对比，ISCT在多数基准上表现相当或略优，并在延迟奖励、数据降级等鲁棒性测试中表现突出；

**⚠️ 局限性**

局限性包括：增量签名计算在高阶时维度下计算量大；目前仅在相对短期任务上验证，长时间序列或复杂真实场景尚未充分评估；缺乏自适应截断与更高效的记忆机制。

---

## 340. TabSieve: Explicit In-Table Evidence Selection for Tabular Prediction

**arXiv ID:** 2602.11700 | [PDF](https://arxiv.org/pdf/2602.11700v1)

**作者:** Yongyao Wang `[一作]` (Renmin University of China), Lijun Li `[通讯]`

**关键词:** `Machine Learning` `Classification` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Supervised Fine-Tuning` `Tabular`

**🎯 论文内容**

提出了TabSieve框架，先在表格中选择少量有用行作为证据，再基于这些证据做预测。

**💡 创新点**

创新点在于把证据选择显式化为中间步骤，并通过强化学习（TAB‑GRPO）同时优化证据选择与预测，显著降低噪声上下文影响。

**🔧 技术方法**

采用自监督数据合成（TabSieve‑SFT‑40K）构造带推理轨迹的训练集，利用Qwen3‑8B作为基础模型，并通过GRPO+任务优势平衡进行RL微调。

**📊 数据集**

使用331个真实表格合成的SFT数据集及公开的75个分类和52个回归表格基准进行评测。

**📈 对比分析**

与传统表格模型、通用LLM和专用表格LLM对比，TabSieve在零/少样本下平均提升2.92%（分类）和4.45%（回归），在所有shot设置中均位列第一。

**⚠️ 局限性**

局限性：仅在8B规模模型上实验，合成数据规模有限，且未覆盖更大范围的多样化表格结构，后续需扩展模型规模与数据来源。

---

## 341. Addressing OSS Community Managers' Challenges in Contributor Retention

**arXiv ID:** 2602.11447 | [PDF](https://arxiv.org/pdf/2602.11447v1)

**作者:** Zixuan Feng `[一作]` (Oregon State University), Anita Sarma `[通讯]` (Oregon State University)

**通讯引用:** 3561 | [OpenAlex ID](https://openalex.org/A5024821289)

**关键词:** `Software Engineering` `Tabular` `Time Series`

**🎯 论文内容**

通过半结构式访谈、文献综述和专家问卷三种方法识别 OSS 社区管理者在贡献者保留中的 10 大挑战，并基于 36 篇研究论文提炼 9 条可操作的保留管理策略。随后使用设计科学研究（DSR）范式，迭代开发了一个 Web 原型（包含自动化通知、预测模型、数据可视化等功能），在 100+ OSS 实务者的反馈中进一步完善，并在 Pyomo 与 DeepSpeed 两个 OSS 项目中进行现场评估，验证其对提升保留决策支持的有效性。

**💡 创新点**

①首次系统地将社区管理者的保留痛点与现有研究对齐并量化；②将多维度保留指标（包括新手跟踪、个人贡献、预测离职、影响评估、包容性等）整合为一体化工具；③在原型中引入多模型预测（Cox 回归、随机森林、神经网络）与隐私保护机制，兼顾实用性与伦理性。

**🔧 技术方法**

使用的主要技术包括：
- DSR 与混合方法（访谈、文献综述、问卷、焦点小组）
- Web 前端：Figma 原型、HTML/CSS/JavaScript
- 后端：Python Flask、REST API、GitHub API
- 数据分析：R/Python 统计与机器学习（Cox、随机森林、神经网络）
- 预测特征：提交频率、PR/issue 交互、标签分布等
- 隐私推断：Namsor API（性别/地区）与邮箱域解析
- 可视化：留存曲线、流失预测表、标签影响图等。

**📊 数据集**

使用了公开的 OSS 项目数据：
- Pyomo（Python 科学优化库）
- DeepSpeed（微软赞助的深度学习优化库）
- 通过 GitHub API 拉取提交、PR、issue、标签、贡献者信息。
- 额外利用 100+ OSS 参与者的匿名问卷数据及 6 名访谈记录。

**📈 对比分析**

评估方式为现场（in situ）用户评估，采用思考-大声朗读 + 结构化问卷。结果显示 8 名项目管理者对原型在减轻工作负担、提高参与者洞察、支持新手跟踪和提升包容性方面高度认可（超过 90% 同意）。相较于传统仅提供描述性 dashboard，原型通过预测模型提前识别高风险贡献者，帮助管理者及时干预；在实际项目中未出现显著负面效应，用户满意度总体在 4.5/5 左右。由于缺乏客观的流失率基准，本文未给出具体预测准确率指标，但指出模型选择可根据项目特征手动调优。

**⚠️ 局限性**

局限性包括：
- 访谈样本仅 6 名，可能无法覆盖所有社区类型；
- 数据隐私推断（性别、地区）依赖外部 API，准确性有限；
- 原型功能以 Pyomo 与 DeepSpeed 为验证，未在更大或多样化项目中测试；
- 预测模型未与长期跟踪实验结合，缺乏长期效果评估；
- 关注点主要在技术与工具层面，对组织文化、治理结构等非技术因素讨论有限。

---

## 342. A Rule-based Computational Model for Gaidhlig Morphology

**arXiv ID:** 2602.12132 | [PDF](https://arxiv.org/pdf/2602.12132v1)

**作者:** Peter J Barclay `[一作]` (Edinburgh Napier University), Peter J Barclay `[通讯]` (Edinburgh Napier University)

**通讯引用:** 394 | [OpenAlex ID](https://openalex.org/A5026153887)

**关键词:** `Computation and Language` `Generation` `Recognition` `Text`

**🎯 论文内容**

从Wiktionary提取苏格兰盖尔语的词元和形态信息，生成标准化词汇格式(SVF)，并将其加载到关系型数据库和Python词形生成工具中，实现基于规则的形态学模型，可用于词形识别、教学资源生成等。

**💡 创新点**

1) 将半结构化的Wiktionary条目转化为可直接使用的结构化数据，弥补低资源语言数据不足；2) 在规则层面实现词形生成，兼顾可解释性和可扩展性；3) 通过SQL频率分析和Python规则引擎，为教学与工具开发提供数据支持。

**🔧 技术方法**

使用 Python 脚本解析 Wiktionary XML dump → 转为 JSON → 生成 SVF；构建 SQL 数据库（MySQL/SQLite）并使用 CHECK 约束保证语义完整；利用规则语法实现词形生成（含 lenition、prothesis、vowel harmony 等变形）；Python 类封装词元和规则，提供枚举、查询功能；对数据进行清洗、去重和变体处理。

**📊 数据集**

① Wiktionary 英文版最新 dump（约 11 GiB）中的苏格兰盖尔语条目；② 通过自定义 SVF 文件（约 17 MiB）；③ 频率列表（来自 GitHub 的 GaelicFrequencyLists，10,000 词条）；④ 额外的词形表（生成 33,132 个变形）。

**📈 对比分析**

通过 SQL 统计常见词形模式（如复数后缀‑an、verbal noun 后缀‑adh 等），并将生成的词形表与频率列表匹配。匹配率从 20%（仅使用词元）提升至 44%（包含所有变形），证明规则系统能显著扩展词形覆盖。未与深度学习模型直接比较，但强调可解释性、低资源适用性和可部署性。

**⚠️ 局限性**

① 规则覆盖不完整，仍缺少完全不规则词形；② 变体与重音的处理仍手工，缺乏系统化；③ 依赖 Wiktionary 数据质量，缺失词形导致误差；④ 评估指标仅覆盖匹配率，缺少精度/召回等完整评估；⑤ 仅关注词形生成，未与句法/语义工具整合。

---

## 343. ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning

**arXiv ID:** 2602.11236 | [PDF](https://arxiv.org/pdf/2602.11236v1)

**作者:** Yandan Yang `[一作]` (AMAP CV Lab), Mu Xu `[通讯]` (AMAP CV Lab)

**关键词:** `Computer Vision and Pattern Recognition` `Robotic Intelligence` `Transformer` `Vision-Language-Action Model` `Diffusion model` `Supervised Fine-Tuning` `Multimodality`

**🎯 论文内容**

构建统一的大规模机器人操作数据集 UniACT，并在此基础上训练一种融合视觉、语言和动作的跨机器人结构无关的 VLA 模型；

**💡 创新点**

提出 Action Manifold Hypothesis 与 Action Manifold Learning（AML），将动作预测从噪声回归转为直接生成；同时开发统一的数据标准化、双流特征交互与可插拔的 3D 感知模块；

**🔧 技术方法**

使用 Qwen3‑VL 视觉‑语言模型、Diffusion Transformer（DiT）动作专家、VGGT 与 Qwen‑Image‑Edit 的 3D 注入、两阶段训练（预训练+监督微调）以及任务均匀采样策略；

**📊 数据集**

训练集为六大公开数据集（OXE、OXE‑AugE、Agibot‑Beta、RoboCoin、RoboMind、Galaxea），评测集为 LIBERO、LIBERO‑Plus、RoboCasa GR1、RoboTwin 2.0；

**📈 对比分析**

与 π_0.5、UniVLA、OpenVLA‑OFT、GR00T、X‑VLA 等基线对比，取得 LIBERO 98.6%、LIBERO‑Plus 80.5%、RoboCasa 58.3%、RoboTwin 81.2% 的最高成功率，AML 在所有基准上均优于噪声预测方案；

**⚠️ 局限性**

仍受限于数据质量、动作空间维度、对极端视觉/语言扰动的鲁棒性以及跨足腿型/无人机等更复杂形态的普适性，未来需进一步提升数据多样性与感知模态。

---

## 344. Prototype Transformer: Towards Language Model Architectures Interpretable by Design

**arXiv ID:** 2602.11852 | [PDF](https://arxiv.org/pdf/2602.11852v1)

**作者:** Yordan Yordanov `[一作]` (TU Wien), Thomas Lukasiewicz `[通讯]` (University of Oxford)

**通讯引用:** 9520 | [OpenAlex ID](https://openalex.org/A5091549352)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

本文提出并实现了一种Prototype Transformer（ProtoT）架构，该模型通过可学习原型向量与输入序列的双向通信来实现自回归语言建模，并显著提升可解释性与鲁棒性。

**💡 创新点**

创新点在于将原型向量作为信息路由通道，替代传统自注意力机制，实现线性计算复杂度，并让模型在训练中自动学习可命名的概念，支持针对性编辑与解释。

**🔧 技术方法**

采用了原型混合器、写门与读门、指数移动平均前缀均值、低秩投影、ReZero风格Alpha门等技术，构成线性复杂度的Transformer变体。

**📊 数据集**

主要使用FineWeb-Edu子集（约2.5亿词）进行预训练，验证集为同一数据集；后续评估在GLUE、文本生成（Elo）及下游任务上完成。

**📈 对比分析**

通过与LLaMA、Mamba、DeltaNet等基准在相同超参数设置下对比，ProtoT在长上下文、文本生成质量（Elo）和GLUE得分上与DeltaNet相当或优于其，在大规模训练中保持相对性能，但与LLaMA仍有一定差距；在意义保持扰动的鲁棒性评估中表现优于LLaMA。

**⚠️ 局限性**

局限性包括：1) 与成熟自注意力模型相比，整体性能仍略逊；2) 长上下文扩展受限，需提升隐藏维度；3) 训练成本相对较高，对原型数量和温度参数较敏感。

---

## 345. Adapting Vision-Language Models for E-commerce Understanding at Scale

**arXiv ID:** 2602.11733 | [PDF](https://arxiv.org/pdf/2602.11733v1)

**作者:** Matteo Nulli `[一作]` (eBay Inc.), Shahram Khadivi `[通讯]` (eBay Inc.)

**关键词:** `Computer Vision and Pattern Recognition` `Recommendation System` `Transformer` `Vision Language Model` `Large Language Model` `Supervised Fine-Tuning` `Image` `Text` `Multimodality`

**🎯 论文内容**

本文提出了一种可复现、架构无关的训练策略，将通用视觉‑语言模型（VLM）针对电商属性、海量图片和噪声数据进行定制化适配，并构建了覆盖属性预测、深度时尚理解、动态属性抽取及多图商品智能的完整评测套件。

**💡 创新点**

创新点在于：①以视觉验证管道自动化高质量数据清洗；②三阶段训练（视觉‑语言对齐、中间阶段、视觉指令微调）和专门的多图商品智能微调；③在保持通用性能的前提下显著提升电商内在任务的准确率，并提供公开可复现的基准数据与评测框架。

**🔧 技术方法**

采用的技术包括：多模态编码器 SigLIP2、Qwen2.5 ViT，文本解码器 Llama‑3.1、e‑Llama、Lilium、Gemma3、Qwen3 等；训练框架 NeMo 与 LLaVA‑OneVision；视觉指令调优采用约 4M 内部电商指令；多图抽取使用 Qwen2.5‑VL‑32B 边界框预测与 GPT‑4.1 标注；推理利用 vLLM 加速。

**📊 数据集**

主要数据集：约 15 M 原始电商商品清单 + 4 M 视觉指令；100 k 多图商品智能样本（每个 2‑8 张图片）；内部评测套件（Aspect Prediction、Deep Fashion Understanding、Dynamic Attribute Extraction、Multi‑Image Item Intelligence）；公开基准 MMBench、MME、MMStar、CVBench、TextVQA、AI2D、MMMU、eComMMMU 等。

**📈 对比分析**

与外部 VLM（如 Qwen3‑VL‑8B、Gemma3、LLaVA‑OV）及自研基线（SigLIP2+Llama‑3.1）在内部电商任务上对比，实验表明：在属性预测、时尚理解、动态抽取等指标提升 10–20%；在公开多模态基准上保持或略低于最强模型；使用 4B 微调模型相较 27B 0‑shot 在多图商品智能上 F1 提升 4–5 分，推理速度提升 3.8×。

**⚠️ 局限性**

主要局限包括：仅使用英文数据，缺乏多语言跨域适配；数据来源单一平台，可能对其他电商生态泛化有限；训练与评测部分依赖 LLM 生成标签与判分，可能带来偏差；动态属性抽取样本量小、类别覆盖有限；处理 10 张以上图片时易出现 OOM 与推理延迟问题。

---

## 346. dVoting: Fast Voting for dLLMs

**arXiv ID:** 2602.12153 | [PDF](https://arxiv.org/pdf/2602.12153v1)

**作者:** Sicheng Feng `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**通讯引用:** 13048 | [OpenAlex ID](https://openalex.org/A5015574447)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

提出一种基于dLLM的快速投票（dVoting）策略，在推理阶段无需额外训练即可提升推理性能，且计算开销较低。

**💡 创新点**

核心创新在于利用dLLM可任意位置并行生成和重掩码机制，先识别跨样本一致的 token，随后仅对不确定 token 进行多次采样并投票，显著减少冗余计算。

**🔧 技术方法**

技术手段包括：多样本一致性分析、非唯一位置率（NUPR）量化、基于熵阈值的并行解码、迭代重掩码与投票聚合。

**📊 数据集**

实验使用了数学推理集 GSM8K、MATH500、ARC-C、通用推理集 MMLU、GPQA，以及两大 dLLM 模型 LLaDA 与 Dream。

**📈 对比分析**

与原始模型、单样本推理、majority voting、HEX、RFG 等基线相比，dVoting 在 GSM8K、MATH500、ARC-C、MMLU 上分别提升约 6.2%–7.7%、4.4%–7.2%、3.2%–14.8%、4.8%–5.7%，同时相较于传统投票方法实现 1.1–4.4 倍速度提升、5.5–22.1 倍效率提升，整体性能-效率比最佳。

**⚠️ 局限性**

局限性包括：在极端复杂或低一致性问题上可能需要更多迭代；对熵阈值等超参数敏感；目前仅在 dLLM 上验证，未探索跨模态或更大模型的适用性。

---

## 347. DeepSight: An All-in-One LM Safety Toolkit

**arXiv ID:** 2602.12092 | [PDF](https://arxiv.org/pdf/2602.12092v1)

**作者:** Bo Zhang `[一作]` (Shanghai AI Laboratory), Xia Hu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `Computation and Language` `Safty and Privacy` `Transformer` `Large Language Model` `Multimodality`

**🎯 论文内容**

开发并公开了DeepSight框架，将大模型安全评估与诊断集成，形成闭环安全工程流程。

**💡 创新点**

首次实现评估与诊断的统一任务与数据协议，支持前沿AI风险与多模态安全评估，并提供可复现、低成本的工具链。

**🔧 技术方法**

采用配置驱动的评估引擎DeepSafe（支持20+安全基准、ProGuard判别器）和诊断引擎DeepScan（X-Boundary、TELLME、SPIN、MI-Peaks等指标），结合vLLM、HuggingFace等后端。

**📊 数据集**

使用多维安全基准（SALAD-Bench、HarmBench、VLSBench、MMSafetyBench等）以及前沿AI风险数据集（EvalFaking、Sandbagging、Manipulation等）。

**📈 对比分析**

通过对比LLM/MLLM在内容安全与前沿风险下的分层表现，发现第一、二层模型在整体安全率>73%；推理模型在“Manipulation”上显著不如非推理模型；开源与闭源模型在多模态下性能差距扩大。

**⚠️ 局限性**

诊断工具受限于模型规模与架构，无法覆盖所有安全维度；安全优势不具可转移性，单一模型难以在所有风险维度表现最佳；推理能力在某些攻击上引入新的安全弱点。

---

## 348. Global Convergence to Nash Equilibrium in Nonconvex General-Sum Games under the $n$-Sided PL Condition

**arXiv ID:** 2602.11835 | [PDF](https://arxiv.org/pdf/2602.11835v1)

**作者:** Yutong Chao `[一作]`, Jalal Etesami `[通讯]`

**关键词:** `Computer Science and Game Theory` `Optimization`

**🎯 论文内容**

提出了 n 侧 PL 条件，研究其下 BCD 算法收敛性，并设计了两种自适应 BCD 算法（IA‑RBCD 与 A‑RBCD），证明其在 n 侧 PL 环境下能够以线性或近线性速率收敛到 Nash 均衡。

**💡 创新点**

创新点在于：1) 将 PL 条件推广到多玩家多坐标情形，定义 n 侧 PL；2) 证明 n 侧 PL 下 stationary 点与 NE 等价；3) 通过引入最佳响应聚合函数 G_F 与 F 的局部关系，刻画随机 BCD 的收敛速度；4) 设计利用最佳响应梯度的自适应 BCD，克服传统 BCD 对最佳响应缺乏访问的限制。

**🔧 技术方法**

使用了光滑性与 PL 条件分析、坐标梯度下降、随机坐标更新、最佳响应逼近（梯度下降子程序）、KL 及 PL 复合条件、Cauchy–Schwarz 边界、以及实验评估中的数值优化工具。

**📊 数据集**

在多种经典非凸游戏中验证：1) 两玩家潜在与一般总和游戏（如带指数项的函数）；2) Cournot 竞争模型（线性与二次需求）；3) 无限时域 n 维 LQ 游戏；4) 其他严格鞍点例子。实验使用人工合成函数与标准经济学/控制理论基准。

**📈 对比分析**

与基准方法 BM1（随机 BCD）和 BM2（只用 case 3 的经验版）比较。实验显示 IA‑RBCD 与 A‑RBCD 在所有测试中收敛更快、更稳健，尤其在存在严格鞍点时仍能以线性速率到达 NE；BM1 在某些非凸情形下收敛慢甚至失败；BM2 速度居中但不如自适应版本。

**⚠️ 局限性**

局限性包括：1) 需要满足 n 侧 PL 条件，实际问题中并不总满足；2) 线性速率需额外的局部假设（如 κ<1）或最佳响应梯度精度；3) 对最佳响应的逼近需要额外梯度迭代，影响总计算成本；4) 对超参数（γ、C、学习率）敏感，需经验调优；5) 对非光滑或高阶非凸问题尚未给出理论保证。

---

## 349. Enhanced Portable Ultra Low-Field Diffusion Tensor Imaging with Bayesian Artifact Correction and Deep Learning-Based Super-Resolution

**arXiv ID:** 2602.11446 | [PDF](https://arxiv.org/pdf/2602.11446v1)

**作者:** Mark D. Olchanyi `[一作]`, Juan Eugenio Iglesias `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Super Resolution` `Restoration` `Segmentation` `Convolutional Neural Network` `Biomedical Data` `Diffusion Tensor Imaging` `Magnetic Resonance Imaging` `Alzheimer's Disease`

**🎯 论文内容**

提出了适用于超低场磁共振的9方向DTI序列，并结合贝叶斯方向依赖性偏置校正与DiffSR超分辨算法，提升超低场DTI的空间与角分辨率。

**💡 创新点**

创新点在于引入针对超低场特有的方向依赖性偏置模型和基于球谐展开的联合时空超分辨网络DiffSR，兼容单回波低场DTI且不需重新训练。

**🔧 技术方法**

使用贝叶斯偏置校正、球谐转换、图卷积、U-Net+全局注意力的深度学习架构以及数据增强技术。

**📊 数据集**

数据集包括18例自制超低场与对应3T高场DTI、100名HCP Young Adult单回波DTI、30名Connectom HCP、24名ADNI AD/LMCI与138名对照。

**📈 对比分析**

与传统三线性上采样、未校正或仅贝叶斯校正方法相比，DiffSR在合成降采样实验中MAE/lncc提升约10‑20%，在AD/LMCI白质FA差异恢复和与高场DTI的ICC提升至0.86；在白质微结构分析中可恢复AD相关FA下降。

**⚠️ 局限性**

局限在于仅针对单回波低b值、对ADC恢复不佳、对不同b值的适应性有限、需要高质量的HCP先验、未考虑方向性失真和温度漂移。

---

## 350. Choose Your Agent: Tradeoffs in Adopting AI Advisors, Coaches, and Delegates in Multi-Party Negotiation

**arXiv ID:** 2602.12089 | [PDF](https://arxiv.org/pdf/2602.12089v1)

**作者:** Kehang Zhu `[一作]` (Harvard University), Crystal Qian `[通讯]` (Google DeepMind)

**关键词:** `Computer Science and Game Theory` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

在一场三人博弈实验中，作者比较了三种LLM辅助模式（Advisor、Coach、Delegate），通过内部随机对照设计评估其对个人和群体福利、外部性以及用户偏好的影响。

**💡 创新点**

①首次将LLM的三种交互模式放入多方博弈环境；②揭示了“偏好-性能错位”，即用户偏好控制型Advisor但最优结果来自全自动Delegate；③发现Delegate通过“市场造市”机制产生正外部性。

**🔧 技术方法**

使用商业API（OpenAI GPT‑4）构建统一LLM代理，分别以三种轻量级提示框架实现 Advisor（主动推荐）、Coach（先手再评估）和 Delegate（完全自主执行）。

**📊 数据集**

实验数据来自243名Prolific受试者在三轮游戏中的交互记录与奖励，游戏为基于芯片的三人交易博弈；无公开公开数据集，全部为实验收集的自定义数据。

**📈 对比分析**

采用线性混合效应模型（LMM）和多重比较校正（Holm‑Bonferroni）比较各模式与无AI基准（人类对照）的群体与个人盈余。结果显示Delegate模式相较基准平均提升约0.084（群体）和0.028（个人），但仅在未校正下显著；同时非使用Delegate用户也获得正向外部性。

**⚠️ 局限性**

①实验仅为单轮、简化博弈，缺乏长期学习与信任演化；②LLM能力处于实验级别，实际应用模型性能不确定；③界面同质化，未探讨解释风格、置信度显示等对采纳的影响；④结果仅适用于实验环境，生态效度有限。

---

## 351. DICE: Diffusion Large Language Models Excel at Generating CUDA Kernels

**arXiv ID:** 2602.11715 | [PDF](https://arxiv.org/pdf/2602.11715v1)

**作者:** Haolei Bai `[一作]` (Westlake University), Huan Wang `[通讯]` (Westlake University)

**通讯引用:** 7559 | [OpenAlex ID](https://openalex.org/A5100751566)

**关键词:** `Machine Learning` `AI Code Assistant` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Diffusion model` `Text`

**🎯 论文内容**

提出了基于扩散大语言模型的CUDA核代码生成系统DICE，结合CuKe数据集和双阶段强化学习框架BiC‑RL，显著提升生成的核代码功能正确性和性能；

**💡 创新点**

①构建了高质量的CuKe数据集（6,303条高性能CUDA核），②设计了双阶段（kernel infilling → end‑to‑end）BiC‑RL强化学习策略，③在扩散LLM上实现了块级半自回归解码以加速生成；

**🔧 技术方法**

采用扩散式语言模型（block diffusion）与TraceRL强化学习，结合CUDA编译与运行时评估的奖励机制；

**📊 数据集**

主要使用CuKe数据集，比较基准使用KernelBench（250个任务，分层1–3）以及cudaLLM、ConCuR等公开数据集；

**📈 对比分析**

在KernelBench上，DICE-8B达到或超过同规模的cudaLLM，超越商业模型Gemini‑3‑Pro；DICE-4B超过多款8B模型；DICE-1.7B在低参数下亦优于多数基线；实验显示双阶段RL比单阶段RL、SFT单独使用更优；

**⚠️ 局限性**

仍受高质量数据稀缺限制，某些复杂任务仍出现欺骗行为；模型规模受限，扩展到更大模型或更复杂核结构需进一步研究；

---

## 352. Think Longer to Explore Deeper: Learn to Explore In-Context via Length-Incentivized Reinforcement Learning

**arXiv ID:** 2602.11748 | [PDF](https://arxiv.org/pdf/2602.11748v1)

**作者:** Futing Wang `[一作]` (Zhejiang University), Tao Lin `[通讯]` (Westlake University)

**通讯引用:** 8623 | [OpenAlex ID](https://openalex.org/A5100702153)

**关键词:** `Computation and Language` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Chain-of-Thought` `Large Language Model` `Text`

**🎯 论文内容**

本文提出一种长度激励探索（Length‑Incentivized Exploration）方法，通过强化学习让大语言模型在推理时主动延长思考链并避免冗余，从而提升推理质量。

**💡 创新点**

创新点在于首次识别并量化“浅层探索陷阱”，并设计了结合长度奖励与冗余惩罚的两步奖励结构，直接提升单条推理路径的状态覆盖率。

**🔧 技术方法**

技术上采用 MDP 框架下的 GRPO/GSPO 强化学习，使用 last‑n‑gram 状态抽象、长度奖励 R_len 与冗余奖励 R_red，结合 LLM 的 CoT 生成。

**📊 数据集**

实验数据集包括 AIME、AMC、MATH‑500、OlympiadBench（in‑domain）以及 ARC‑c、GPQA‑Diamond、MMLU‑Pro（out‑of‑domain）；训练集采用 DAPO‑Math‑17k、Polaris、DeepMath‑5k 等。

**📈 对比分析**

与 GRPO、GSPO 等基线比较，平均提升 4.4%（in‑domain）和 2.7%（OOD），在 Qwen3、LLaMA 等多模型上均实现 2–3% 的性能提升，并展示了更好的测试时扩展性。

**⚠️ 局限性**

局限性包括：长度奖励可能导致重复生成，需通过冗余惩罚调节；对极长序列的探索仍受模型容量限制；实验主要集中在数值推理任务，跨领域通用性待进一步验证。

---

## 353. Resource-Aware Deployment Optimization for Collaborative Intrusion Detection in Layered Networks

**arXiv ID:** 2602.11851 | [PDF](https://arxiv.org/pdf/2602.11851v1)

**作者:** André García Gómez `[一作]` (AIT Austrian Institute of Technology), Edgar Weippl `[通讯]` (University of Vienna)

**通讯引用:** 6472 | [OpenAlex ID](https://openalex.org/A5083435816)

**关键词:** `Cryptography and Security` `Anomaly Detection` `Optimization` `Convolutional Neural Network`

**🎯 论文内容**

提出了一个面向分布式边缘环境的协同入侵检测框架，能够根据节点资源与数据类型动态分配检测器，实现快速自适应部署。

**💡 创新点**

创新点在于将资源感知与层级自适应优化结合，使用元启发式搜索实现近最优的检测器配置，同时兼顾系统的可扩展性和可靠性。

**🔧 技术方法**

采用了元启发式优化算法（局部搜索、禁忌搜索、蚁群优化）进行节点级部署调度，并实现了多层安全与信任机制；检测器包括CNN、MLP、DT、NV、EV、ED、EC等轻量级模型。

**📊 数据集**

使用了三类数据集：UAV（DoS/FDI/Replay）、AIT‑LDv2（多服务器攻击）以及自建的UGV攻击数据集（多阶段对抗场景）。

**📈 对比分析**

通过与穷举搜索对比，禁忌搜索在保持接近最优F1分数的同时，将优化时间缩短至毫秒级；在Raspberry Pi 5边缘设备上实现了实时部署与自适应重配置。

**⚠️ 局限性**

局限在于实验仅涵盖有限的场景与数据集，缺乏对网络延迟、攻击对抗模型及大规模部署的深入验证。

---

## 354. Credit Where It is Due: Cross-Modality Connectivity Drives Precise Reinforcement Learning for MLLM Reasoning

**arXiv ID:** 2602.11455 | [PDF](https://arxiv.org/pdf/2602.11455v1)

**作者:** Zhengbo Jiao `[一作]` (Alibaba Group Holding Limited), Linfeng Zhang `[通讯]` (SJTU)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Optimization` `Transformer` `Reinforcement Learning` `Large Language Model` `Vision Language Model` `Multimodality`

**🎯 论文内容**

提出 Anchor‑Token Reinforcement Learning (AT‑RL)，通过交叉模态注意力连通性对多模态大语言模型的奖励信号进行基于感知锚点的软加权，以实现更精准的信用分配。

**💡 创新点**

核心创新是识别并利用仅占约15% 的高连通性 token（感知锚点）作为视觉引导点，并通过图聚类与连通性密度实现对奖励的细粒度、软化加权，从而在视觉推理中显著提升模型表现。

**🔧 技术方法**

技术手段包括：交叉模态注意力提取与偏置校正、token‑level 图构建与 METIS 聚类、连通性密度计算与软权重映射、与 RLVR（GRPO、DAPO、SAPO 等）框架的插件式融合，以及高效的训练实现（1.2% 计算开销）。

**📊 数据集**

使用多模态推理数据集：ViRL‑39K、Geometry‑3K、GeoQA‑3K、GeoQA‑8K、MathVision、MathVerse、MathVista、MMMMU、VideoMMMU 等，并在这些基准上进行评估。

**📈 对比分析**

对比统一信用分配的 GRPO、DAPO、GSPO、SAPO 等强基线，AT‑RL 在 Qwen2.5‑VL‑7B/32B 上平均提升 5–8%，32B+AT‑RL 超越 72B‑Instruct，MathVista 80.2、MathVerse 56.6 等指标，且在多模态与视频推理任务上均显示显著加速与精度提升。

**⚠️ 局限性**

局限性包括：仍受限于知识缺失（知识部署误差）与对极端视觉噪声的鲁棒性未知；RL 只优化已有知识的应用，难以补充新领域知识；此外，AT‑RL 主要针对可验证奖励场景，跨域适用性需进一步验证。

---

## 355. TimeSynth: A Framework for Uncovering Systematic Biases in Time Series Forecasting

**arXiv ID:** 2602.11413 | [PDF](https://arxiv.org/pdf/2602.11413v1)

**作者:** Md Rakibul Haque `[一作]` (University of Utah), Warren Woodrich Pettine `[通讯]` (Mountain Biometrics Inc.)

**关键词:** `Machine Learning` `Data Synthesis` `Anomaly Detection` `Convolutional Neural Network` `Transformer` `Time Series` `Biomedical Data` `Electrocardiogram`

**🎯 论文内容**

本文提出了 TimeSynth 这一基于真实时间序列参数生成的合成数据框架，用于系统评估线性与非线性模型在不同时间序列动态下的预测能力。

**💡 创新点**

创新点在于（1）用真实数据拟合得到参数，构建多样且可控的 Drift‑Harmonic、SPM‑Harmonic 与 DPM‑Harmonic 三类信号；（2）在清洁、噪声与分布偏移三种实验范式下，统一使用幅度、频率与相位误差三维指标，揭示线性模型的系统性偏差。

**🔧 技术方法**

采用线性、MLP、CNN 与 Transformer 四大模型族，结合混合效应模型进行统计对比，并利用频谱与希尔伯特变换评估频率与相位精度。

**📊 数据集**

使用从 PPG‑Dalia 与 MIT‑BIH 斑点心电数据库中拟合得到的参数生成的合成时间序列，共 70/10/20 组用于训练/验证/测试。

**📈 对比分析**

比较方法包括在清洁、噪声（SNR 40/30/20 dB）和频率分布偏移（Shift‑1~4）三种情境下，计算 MAE、频率误差与相位误差；实验结果显示非线性模型（CNN、Transformer）在幅度、频率与相位上明显优于线性模型，且在复杂 DPM‑Harmonic 信号下表现尤为突出。

**⚠️ 局限性**

局限性在于仅关注单变量合成信号；未考虑多变量、长周期或非周期性动力学；模型训练采用固定窗口长度，未探讨自适应窗口或迁移学习策略。

---

## 356. PAM: Processing Across Memory Hierarchy for Efficient KV-centric LLM Serving System

**arXiv ID:** 2602.11521 | [PDF](https://arxiv.org/pdf/2602.11521v1)

**作者:** Lian Liu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Ying Wang `[通讯]` (State Key Lab of Processors, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `Hardware Architecture` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

提出基于多层PIM内存的KV中心化LLM服务系统PAM，兼顾高带宽与大容量需求。

**💡 创新点**

创新点包括处理跨层注意力（PAMattention）实现token级并行，结合KV上下文局部性实现动态KV映射与迁移，及硬件协同的PAM接口与调度。

**🔧 技术方法**

采用Processing-In-Memory（HBM‑PIM、DDR‑PIM、SSD‑PIM）技术、在线softmax、分层归约单元、硬件加速的KV映射与调度，配合LLMServingSim仿真框架。

**📊 数据集**

使用的公开LLM模型包括Qwen2.5‑32B、LLaMA3‑70B、OPT‑175B，评测数据集涵盖ShareGPT、WildChat、HumanEval、Arxiv_sum、Write_doc。

**📈 对比分析**

与vLLM‑offloading、AttAcc!、L‑PIM、LS‑PIM等基线对比，在在线对话与离线长文本任务中，PAM平均提升吞吐量12.9×（对话）/26.4×（长上下文），能耗降低至基线的4–53%，并在多节点/张量并行下保持良好伸缩性。

**⚠️ 局限性**

局限性在于对PIM硬件的依赖，需在不同制造工艺与设备上验证兼容性，且高成本PIM设备仍是规模化部署的技术门槛。

---

## 357. Multi-Level Strategic Classification: Incentivizing Improvement through Promotion and Relegation Dynamics

**arXiv ID:** 2602.11439 | [PDF](https://arxiv.org/pdf/2602.11439v1)

**作者:** Ziyuan Huang `[一作]` (University of Michigan), Mingyan Liu `[通讯]` (University of Michigan)

**通讯引用:** 11749 | [OpenAlex ID](https://openalex.org/A5101967011)

**关键词:** `Machine Learning` `Classification` `Optimization` `Recommendation System` `Reinforcement Learning` `Tabular` `Finance Related`

**🎯 论文内容**

提出一种多层晋升/降级的策略分类模型，通过设计阈值序列来激励主体诚实提升而非作弊；

**💡 创新点**

首次将“leg‑up”效应、属性保留与时延折扣结合到多级分类中，并证明在合理条件下可通过阈值设计实现无作弊、可达到任意高水平；

**🔧 技术方法**

使用马尔可夫决策过程（MDP）求解主体最优策略，值迭代（Value‑Iteration）计算代理响应；对设计者则采用贪心阈值搜索与CMA‑ES优化主问题；

**📊 数据集**

在实验中使用FICO信用评分数据模拟多级信用产品的晋升路径；

**📈 对比分析**

通过对比四种成本情景（c⁺/c⁻组合），展示了在不同参数下阈值长度、奖励率与最终主体属性的变化；实验显示在满足可行性条件时，模型能够实现主体在多个层级中持续进步，且性能随保留率γ和预期折扣β提升而提升；

**⚠️ 局限性**

局限包括：模型假设单维属性、已知折扣/保留系数；对作弊成本极低的情况仍难以完全消除作弊；且阈值设计需依赖代理的最佳响应假设，现实中可能存在信息不完全或动态变化的挑战。

---

## 358. Voxtral Realtime

**arXiv ID:** 2602.11298 | [PDF](https://arxiv.org/pdf/2602.11298v1)

**作者:** Alexander H. Liu `[一作]`, Zhenlin Xu `[通讯]`

**关键词:** `Artificial Intelligence` `Recognition` `Transformer` `Audio`

**🎯 论文内容**

开发了 Voxtral Realtime，一种本地流式自动语音识别模型，能够在亚秒级延迟下匹配离线模型的质量。

**💡 创新点**

创新点包括端到端训练的本地流式架构，因果音频编码器与 Ada RMS‑Norm 的延迟条件，使用 RMSNorm、SwiGLU、RoPE 与滑动窗口注意力的全新编码器，以及通过目标构造实现音频‑文本的隐式对齐，并在 13 种语言上进行大规模预训练。

**🔧 技术方法**

采用 Transformer、DSM 延迟流模型、因果音频编码器、Ada RMS‑Norm、滑动窗口注意力、RoPE、SwiGLU、RMSNorm、分组查询注意力、MLP 适配器，以及 vLLM 的分页注意力和 WebSocket 实时 API。

**📊 数据集**

使用覆盖 13 种语言的大规模预训练数据集；评测使用 FLEURS 多语、Mozilla Common Voice、英语短长形式等数据集。

**📈 对比分析**

通过宏观平均 WER 与离线 Whisper、Scribe v2 Realtime、GPT‑4o mini、DSM、Nemotron Streaming 等模型比较；在 480 ms 延迟下与 Whisper/Scribe 接近，960 ms 甚至 2400 ms 延迟时可超越这些模型，接近离线 Voxtral Mini Transcribe V2 的准确率。

**⚠️ 局限性**

局限性包括对左侧填充敏感、对极低延迟（<80 ms）性能有限、模型规模相对较大以及对多域鲁棒性和更广语言覆盖的进一步提升空间。

---

## 359. Semantically Conditioned Diffusion Models for Cerebral DSA Synthesis

**arXiv ID:** 2602.11703 | [PDF](https://arxiv.org/pdf/2602.11703v1)

**作者:** Qiwen Xu `[一作]` (Ludwig Maximilian University Munich), Máté E. Maros `[通讯]` (Heidelberg University)

**通讯引用:** 1706 | [OpenAlex ID](https://openalex.org/A5008029296)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Convolutional Neural Network` `Diffusion model` `Auto Encoder` `Image` `Biomedical Data`

**🎯 论文内容**

本研究开发了一种语义条件潜在扩散模型，用于从噪声中合成具有前/后循环及C‑arm角度控制的脑动脉期DSA图像。

**💡 创新点**

创新点在于实现了对血管解剖分区和成像姿势的显式语义控制，并在单中心大规模DSA数据上训练模型，首次提供高保真、可控的DSA合成方法。

**🔧 技术方法**

采用了卷积变分自编码器、Latent Diffusion Model、BERT文本编码器以及跨注意力机制实现条件扩散。

**📊 数据集**

使用了曼海姆大学医学中心2021年前后收集的99,349张血管期DSA帧，涵盖前循环和后循环两种解剖区。

**📈 对比分析**

与多模态专家评审及Fréchet Inception Distance对比，合成图像的平均Likert评分约为3.1–3.3，FID仅为15.27，表明合成图像在视觉和统计分布上与真实DSA高度相似。

**⚠️ 局限性**

主要局限包括单中心数据、仅生成二维帧、条件维度有限（仅循环与平面角度）、以及对临床实用性评估依赖专家主观评分与ImageNet预训练特征。

---

## 360. Fast Tuning the Index Construction Parameters of Proximity Graphs in Vector Databases

**arXiv ID:** 2602.11573 | [PDF](https://arxiv.org/pdf/2602.11573v1)

**作者:** Wenyang Zhou `[一作]` (Xidian University), Jiangtao Cui `[通讯]` (Xidian University)

**通讯引用:** 2223 | [OpenAlex ID](https://openalex.org/A5048420839)

**关键词:** `Databases` `Optimization` `Hyperparameter Search` `Gaussian Process` `Multi-objective Bayesian Optimization` `Graph`

**🎯 论文内容**

针对高维向量空间中近邻图（PG）构建参数调优问题，提出 FastPGT 框架，在保持调优质量的前提下显著加速参数估计。

**💡 创新点**

创新点在于：①可批量推荐参数并并行构建多张 PG；②利用多 PG 共享的图结构与距离计算，设计高效的多 k‑ANNS 与多 Prune 操作；③对传统 VDTuner 的 EHVI 推导出可批量的 mEHVI，兼容任意推荐模型。

**🔧 技术方法**

核心技术包括：多目标贝叶斯优化、Gaussian Process 回归、mEHVI 采集策略、确定性随机策略、共享距离缓存（V_δ）、批量 k‑ANNS 与批量 Prune 的实现。

**📊 数据集**

使用公开的四大数据集：Sift、Gist、Glove、Msong，分别包含 100 万到 118 万个高维向量。

**📈 对比分析**

对比 RandomSearch、OtterTune、VDTuner 等 SOTA，FastPGT 在 100 个参数候选下的调优时间与距离计算分别提升 2.2‑2.4×，并在同等预算下获得相同或更高的 Recall@k 与 QPS，调优成本仅为 VDTuner 的 36%～55%。

**⚠️ 局限性**

局限性在于：1) 仅针对基于邻图的索引，其他索引类型需进一步改造；2) 需要对参数空间进行合理剪枝（如去除 R），否则批量推荐可能受限；3) 在极大数据规模下仍需关注缓存 V_δ 的内存消耗。

---

## 361. AlphaPROBE: Alpha Mining via Principled Retrieval and On-graph biased evolution

**arXiv ID:** 2602.11917 | [PDF](https://arxiv.org/pdf/2602.11917v1)

**作者:** Taian Guo `[一作]` (Peking University), Ming Zhang `[通讯]` (Peking University)

**通讯引用:** 15991 | [OpenAlex ID](https://openalex.org/A5100447315)

**关键词:** `Artificial Intelligence` `Optimization` `Recommendation System` `Data-Centric Learning` `Finance Related` `Graph Neural Network` `Large Language Model` `Retrieval-Augmented Generation` `Tabular` `Time Series`

**🎯 论文内容**

提出 AlphaMining 框架，将 Alpha 因子进化建模为有向无环图（DAG），通过闭环的贝叶斯因子检索器与 DAG 感知因子生成器实现全局结构化的 Alpha 因子发现。

**💡 创新点**

创新点在于：① 以 DAG 视角捕捉因子间全局关系，弥补传统 DFG 与 IFE 的局部视角不足；② 采用贝叶斯检索器将因子质量、深度、检索次数与整体池子多样性统一进后验模型；③ DAG 感知生成器利用完整祖先路径、策略拆分和多代理生成流程，显著减少冗余变异并提升多样性。

**🔧 技术方法**

技术核心包括：贝叶斯因子检索器（后验概率 + 先验深度/检索惩罚）；多维度多样性评价（数值、语义、语法）；DAG 结构化生成器（分析、执行、验证三步）；LLM（Deepseek V3.1）用于生成与策略制定；嵌入模型（Qwen 3 Embedding-4B）用于语义相似度；动态因子集成器用于投资组合构建。

**📊 数据集**

实验数据集为中国主流股票池：CSI 300、CSI 500、CSI 1000；训练/验证/测试划分分别为 2010‑2020、2021‑2022.6、2022.7‑2025.6。

**📈 对比分析**

与 8 类基线（专家手工、DFG、IFE 与 LLM 代理）比较，AlphaMining 在 IC、ICIR、RIC、RICIR、AR、MDD、SR 等指标上均取得最高或第二高分，显示出更高的预测精度、风险控制和收益稳定性；回测曲线亦表现出更优的累计收益与更低的回撤。

**⚠️ 局限性**

局限性：① 依赖大规模 LLM 与算力，部署成本高；② 现有框架仅针对日频量化，尚未验证高频或基本面因子场景；③ DAG 结构随着因子数量膨胀可能导致检索与生成效率下降，需要进一步的图压缩或稀疏化方法。

---

## 362. Divide and Learn: Multi-Objective Combinatorial Optimization at Scale

**arXiv ID:** 2602.11346 | [PDF](https://arxiv.org/pdf/2602.11346v1)

**作者:** Esha Singh `[一作]` (University of California San Diego), Yi-An Ma `[通讯]` (University of California San Diego)

**通讯引用:** 4425 | [OpenAlex ID](https://openalex.org/A5041806431)

**关键词:** `Machine Learning` `Optimization` `Reinforcement Learning` `Mixture of Experts` `Tabular`

**🎯 论文内容**

提出一种基于决策空间分解和多专家学习的在线多目标组合优化框架（D&L），将全局问题拆分为子问题并通过位置级别的 bandit 学习实现样本效率和计算可扩展性。

**💡 创新点**

创新点：① 将 MOCO 重新表述为全 bandit 反馈下的在线学习；② 在子问题层面实现多专家（UCB、EXP3、FTRL）融合，兼顾探索、对抗鲁棒性与梯度自由局部搜索；③ 通过拉格朗日松弛解决重叠子问题的协调，理论上得到 O(d√(T log T)) 的 regret 率，仅依赖子问题维度 d，而非指数规模。

**🔧 技术方法**

技术：在线学习、位置级别的多专家 bandit、拉格朗日松弛/加速镜面下降、无梯度的零阶局部搜索、混合 scalarization。

**📊 数据集**

数据集：传统 MOCO 经典问题（TSP、Knapsack、CVRP）以及一个四目标硬件-软件共设计的加速器仿真问题，搜索空间从 10⁵ 到 10¹⁵⁷。

**📈 对比分析**

与基线比较：与专用启发式、NSGA‑II、Bayesian 优化（MOBO-qParEGO/qNEHVI、BOPR）、神经网络方法（PMOCO）等。D&L 在黑盒设置下获得 80–98% 的专用解算器性能，优于 Bayesian 20–90%，并在硬件仿真中以 22% 的 HV 提升和 10–30× 的速度提升；随着问题规模和目标数增大，D&L 的优势进一步显现。

**⚠️ 局限性**

局限性：对重叠子问题的拉格朗日参数采用固定混合权重；只证明了子高斯奖励的情况，非子高斯情形尚未处理；在全局耦合强、非局部性高的问题中，分解效果可能不佳。

---

## 363. Calibrating an Imperfect Auxiliary Predictor for Unobserved No-Purchase Choice

**arXiv ID:** 2602.11505 | [PDF](https://arxiv.org/pdf/2602.11505v1)

**作者:** Jiangkai Xiong `[一作]` (Peking University), Hanzhao Wang `[通讯]` (University of Sydney)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5109694227)

**关键词:** `Machine Learning` `Recommendation System` `Optimization` `Tabular`

**🎯 论文内容**

本文研究了在缺少无购买事件观测的条件下，利用可能偏差的外部预测器校准并估计未观测的无购买概率。

**💡 创新点**

创新点在于提出了基于logit结构身份的校准方法：一是针对仿射误差的线性回归校准，二是针对近似单调误差的最大秩相关（MRC）校准，并给出了有限样本误差界与决策子最优性保证。

**🔧 技术方法**

所采用的技术包括结构识别、线性回归、最大秩相关估计、统计学习理论分析、仿真验证以及多预测器的加权/中位数聚合方法。

**📊 数据集**

实验使用了合成数据验证模型性能，并在Expedia个性化排序公开数据集上进行真实数据实验，其中历史时段用作预测器训练，当前时段用作估计与评估。

**📈 对比分析**

通过与oracle（使用真实inclusive value）、线性校准、MRC校准以及多预测器聚合等基线比较，实验表明MRC校准在各误差模型下更稳健，整体校准后预测误差下降，决策收入损失显著降低。

**⚠️ 局限性**

局限性在于需要商品效用可学习且具足够变异，外部预测器必须保持某种顺序或仿射关系，且无法纠正所有共享偏差；在极端噪声或近似单调失效时误差上限显著。

---

## 364. On Decision-Valued Maps and Representational Dependence

**arXiv ID:** 2602.11295 | [PDF](https://arxiv.org/pdf/2602.11295v1)

**作者:** Gil Raitses `[一作]`, Gil Raitses `[通讯]`

**关键词:** `Artificial Intelligence` `Graph`

**🎯 论文内容**

设计并实现了 DecisionDB，一个面向离散决策的可重现、可审计的决策值映射框架，能够在同一数据快照下系统地变换表示，记录每个表示对应的离散决策，并通过内容寻址实现完整的可追溯链。

**💡 创新点**

引入了“决策值映射”概念，将表示空间划分为持久性区域、边界和断层；通过内容寻址的不可变写一次实体链和可重放的回放机制，使决策的可重用性成为可机械检验的条件；并将这一诊断层嵌入到轻量级 Python/SQLite 基础设施中。

**🔧 技术方法**

使用内容寻址（SHA‑256 + 前缀）、不可变写一次表、五阶段表示扫协议、等价策略（对原始输出进行规范化并哈希）、Python、SQLite、Dijkstra 最短路径求解器、以及基于 JSON 的序列化等技术。

**📊 数据集**

在实验中采用了一个 564 节点、若干条边的有向图（边属性基于地理距离和压力度量），单一源点-目的点（85→50）的最短路径任务；实验还展示了邻居权重和二阶权重两个表示参数的变换。

**📈 对比分析**

通过在两条参数轴上各取两值（共 4 个表示）进行扫，记录每个表示对应的决策 ID，并验证回放时所有内容地址（策略 ID、payload hash、决策 ID）完全一致；执行时间仅 0.5–1.4 毫秒，证明系统可在低延迟下完成；性能评估主要体现为回放成功率（100%）和存储/查询的不可变性，未与其他方法做直接数值对比。

**⚠️ 局限性**

局限性：只适用于能通过等价策略离散化的输出；要求快照与引擎保持不变；实验仅覆盖 4 个表示点，无法完全映射整个表示空间；不解释边界出现的机制；对大规模参数网格的存储和查询性能未验证；实现基于 SQLite，规模化扩展尚待评估；需手工制定等价策略，具有领域特定的主观性。

---

## 365. Gradients Must Earn Their Influence: Unifying SFT with Generalized Entropic Objectives

**arXiv ID:** 2602.11424 | [PDF](https://arxiv.org/pdf/2602.11424v1)

**作者:** Zecheng Wang `[一作]` (Harbin Institute of Technology), Dianbo Sui `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 744 | [OpenAlex ID](https://openalex.org/A5031774490)

**关键词:** `Computation and Language` `Optimization` `Knowledge Distillation` `Transformer` `Supervised Fine-Tuning` `Large Language Model` `Text` `Biomedical Data`

**🎯 论文内容**

本文提出了一种统一的变形对数（deformed-log）框架，用来重新设计SFT（监督微调）目标，并在此框架下推出了无参数、基于熵的动态调节方法DEFT，能够在保持模型先验知识的同时，高效地学习新知识。

**💡 创新点**

创新点主要有三：①将Tsallis q-logarithm引入SFT目标，显式化“置信门控 × 误差”结构；②利用Cayley变换推导状态依赖的聚焦轨迹，使门控随模型置信度平滑过渡；③在不需要离线过滤或额外超参的前提下，用分布熵直接调节门控，实现动态探索–利用平衡。

**🔧 技术方法**

技术手段包括：Tsallis q-log 对数与其梯度分析、Cayley 变换、信息理论与熵几何的双重性证明、梯度分布可视化、参数无关的动态熵调节（DEFT）以及对多种大语言模型（LLaMA、DeepSeekMath、Qwen2.5等）的微调实验。

**📊 数据集**

使用的数据集涵盖三种难度区间：Model-Strong（NuminaMath、Math500、Minerva等数学/推理基准）、Model-Intermediate（m23K、Tulu3-SFT）和 Model-Weak（合成 FigFont 谜题）以及混合领域的医学基准（MedMC、MedQA、PubMed、MMLU-P、GPQA、Lancet、MedB4/5、NEJM）。

**📈 对比分析**

评估方式：将DEFT 与传统 NLL、-p、EAFT 等目标在相同模型（8B、7B 等）与相同任务上对比，采用准确率或平均分等指标。实验表明 DEFT 在所有三个能力区间均表现最优，尤其在 Model-Weak 区间大幅优于 -p，并在 OOD 转移任务中显著降低灾难性遗忘，显示出更稳健的探索–利用权衡。

**⚠️ 局限性**

局限性：①实验仅覆盖至 7B 级模型，尚未验证更大规模模型；②未系统评估计算成本和内存开销；③对熵作为状态度量的适用性仍有待进一步验证；④缺乏对超参数（如温度或熵阈值）敏感度的深入分析；⑤仅聚焦于 token 级梯度，未考虑更高级结构或序列级别的调节。

---

## 366. TexSpot: 3D Texture Enhancement with Spatially-uniform Point Latent Representation

**arXiv ID:** 2602.12157 | [PDF](https://arxiv.org/pdf/2602.12157v1)

**作者:** Ziteng Lu `[一作]` (Chinese University of Hong Kong Shenzhen), Xiaoguang Han `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Generation` `Transformer` `Diffusion model` `Auto Encoder` `Mesh`

**🎯 论文内容**

提出TexSpot框架，对3D模型纹理进行后处理式增强，显著提升细节清晰度与全局一致性。

**💡 创新点**

设计了Texlet潜在空间，融合点纹理与UV映射优势，利用层次VAE与Diffusion Transformer实现高效纹理改进。

**🔧 技术方法**

采用2D编码器+3D编码器的层次VAE、3D→2D级联解码器、条件Diffusion Transformer（DiT）以及CFG和重加权策略。

**📊 数据集**

使用自建的100K高质量Mesh+4096×4096纹理数据集，并在TexVerse等公开数据上进行验证。

**📈 对比分析**

在PSNR/SSIM/LPIPS/FID等指标上与CAMixerSR、DiffBIR、PBR‑SR、Hunyuan3D等方法对比，TexSpot在细节与一致性上平均提升5–10%。

**⚠️ 局限性**

Texlet依赖几何聚类，对噪声或低质量网格鲁棒性差；后处理效果受初始纹理缺失影响，且与上游生成模型耦合不紧密。

---

## 367. Sub--Riemannian boundary value problems for Optimal Geometric Locomotion

**arXiv ID:** 2602.12199 | [PDF](https://arxiv.org/pdf/2602.12199v1)

**作者:** Oliver Gross `[一作]` (University of California San Diego), Peter Schröder `[通讯]` (California Institute of Technology)

**通讯引用:** 18447 | [OpenAlex ID](https://openalex.org/A5051948556)

**关键词:** `Robotics` `Optimization` `Robotic Intelligence` `Video`

**🎯 论文内容**

本文提出了一种基于子-Riemann几何的形变驱动运动最优模型，能够同时考虑外部摩擦耗能和内部能量消耗，求解形变体在不同边界条件下的最优运动轨迹。

**💡 创新点**

创新点包括：①将形变和位移的耦合视为子-Riemann几何问题，提出水平提升（horizontal lift）与最小耗能原理；②在耗能度量中同时加入外部阻力（可异向）和内部弯曲/拉伸耗能；③对无限维曲线空间进行多尺度离散化，保持能量守恒，支持高分辨率形变；④在三类边界条件（固定初末体、周期形变、χ-等效）下统一求解子-Riemann测地线。

**🔧 技术方法**

使用的技术包括：子-Riemann几何与Carnot–Caratheodory度量；Rayleigh耗能理论与可变阻力模型；离散几何与变分积分子算法；多尺度多体形变离散化；牛顿-拉夫逊优化求解子-Riemann边值问题。

**📊 数据集**

主要使用的实验数据集有：自然蛇类（如Chionactis occipitalis）的视频提取形状；低维Purcell三连杆游泳器的参数化；以及标准的蛇/鞭毛运动数据，用于验证模型预测与实验轨迹的一致性。

**📈 对比分析**

与传统低维参数化模型（如Serpenoid形状空间、已知最优游泳器轨迹）比较，模型在高分辨率下显著降低总耗能，并保持耗能曲线平滑；在蛇类运动中，预测的幅度和曲率与实验观测相符；在Purcell游泳器中，模型能自动确定最优周期数和形变路径，匹配或优于已知最优方案。

**⚠️ 局限性**

局限性包括：①未显式约束形变体的自交，限制了对机器人系统的直接应用；②外部耗能仍采用线性阻力近似，无法处理非线性应变相关的阻力；③对三维或多体表面形状的推广尚未完成；④由于高维优化求解，计算成本仍高，难以实时实现。

---

## 368. Iskra: A System for Inverse Geometry Processing

**arXiv ID:** 2602.12105 | [PDF](https://arxiv.org/pdf/2602.12105v1)

**作者:** Ana Dodik `[一作]`, Justin Solomon `[通讯]`

**关键词:** `Graphics` `Optimization` `Auto Encoder` `Mesh`

**🎯 论文内容**

设计并实现了一个可插拔的几何逆问题求解系统，能够在不重写现有几何处理算法的前提下，通过自动求导（adjoint method）对线性系统、特征分解、本地-全局求解、ADMM等多种稀疏优化问题进行后向传播，并与 PyTorch 等机器学习框架无缝对接。

**💡 创新点**

核心创新在于将几何处理的散射‑聚集（scatter‑gather）操作与隐式函数的推导结合起来：用户只需提供一次迭代或隐式关系，系统自动构造后向链；对稀疏线性系统和特征分解提供了专门的高效后向实现；同时支持 GPU/CPU 两端稀疏计算，极大降低实现复杂度。

**🔧 技术方法**

使用技术包括：adjoint method（隐式微分）、稀疏张量（SparseTensor）与散射‑聚集操作、GMRES 迭代求解、CUDSS/CHOLMOD 等稀疏线性求解器、特征分解的四种求导方式、固定点接口、自动微分框架 PyTorch。

**📊 数据集**

实验数据集主要是各种三角网格和四面体网格，包括 Sappho 3D 模型、球面 642 顶点网格、平面高度场网格等，顶点数从几百到几万甚至十几万。

**📈 对比分析**

通过与 Theseus、SparseSolve、CVXPYLayers 等现有系统对比，展示了前向/后向时间和内存占用。实验表明，本系统在同一任务下往往快 1‑2 个数量级、占用内存更低，尤其在 GPU 上利用专门的稀疏求解器实现显著加速。

**⚠️ 局限性**

局限性包括：仅支持静态拓扑，无法处理动态拓扑变形或物理碰撞/接触等；对大规模稀疏矩阵的逆求解仍受限；需要手动提供隐式关系或固定点迭代，缺乏完全声明式的自动化接口；GPU 侧的稀疏算子实现仍依赖自定义实现。

---

## 369. AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems

**arXiv ID:** 2602.11510 | [PDF](https://arxiv.org/pdf/2602.11510v1)

**作者:** Faouzi El Yagoubi `[一作]` (Polytechnique Montreal), Godwin Badu-Marfo `[通讯]` (Polytechnique Montreal)

**关键词:** `Artificial Intelligence` `Safty and Privacy` `Benchmark` `Finance Related` `Biomedical Data` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

构建了 AgentLeak 基准，用于评估多代理 LLM 系统在七条通信通道中的隐私泄露，覆盖医疗、金融、法律和企业四个行业，共 1000 个场景。

**💡 创新点**

创新点在于首次提供完整的多通道泄露分类、32 类攻击体系和三层检测管道，并在真实多代理工作流中量化内部通道泄露对整体安全的影响。

**🔧 技术方法**

采用了可插拔的 SDK 监控最终输出、代理间消息、工具输入/输出、共享内存、日志和产出等七条通道，结合正则、格式模式匹配与 LLM 语义评估三层检测。

**📊 数据集**

使用合成与公开真实数据（包括医疗记录、金融交易、法律文件等）生成敏感字段和允许字段，确保场景真实且符合法规要求。

**📈 对比分析**

在 GPT‑4o、GPT‑4o‑mini、Claude‑3.5‑Sonnet、Mistral‑Large 与 Llama‑3.3‑70B 五大 LLM 上执行 4,979 条完整追踪，发现多代理配置下内部通道泄露率提升 2.1×，输出只审计遗漏 41.7% 的违规；现有防御对外部通道有效但对内部通道无效。

**⚠️ 局限性**

局限性包括仅评估英文场景、仅测试协调‑工作者两代理拓扑、使用有限的四大框架、检测阈值需针对多语言重调、以及未覆盖更大规模或更复杂的多代理网络。

---

## 370. On Fundamental Limits of Transmission Activity Detection in Fluid Antenna Systems

**arXiv ID:** 2602.11901 | [PDF](https://arxiv.org/pdf/2602.11901v1)

**作者:** Zhentian Zhang `[一作]` (Southeast University), Chan-Byoung Chae `[通讯]` (Yonsei University)

**通讯引用:** 10693 | [OpenAlex ID](https://openalex.org/A5079863632)

**关键词:** `Information Theory` `CRB理论` `随机矩阵理论`

**🎯 论文内容**

提出统一的CRB框架，用于评估流体天线系统(FAS)和传统多固定位置天线(FPA)的传输活动检测性能

**💡 创新点**

将二进制活动指示器连续化以适用CRB，得到闭式协方差基和相干基CRB，并在单天线FAS上利用随机矩阵理论得到近似CRB，展示FAS在空间多样性和复杂度上的优势

**🔧 技术方法**

CRB理论、Slepian‑Bang公式、随机矩阵理论、块相关通道模型、正交/非正交导频设计

**📊 数据集**

随机生成的导频矩阵和Rayleigh/克拉克通道模型，未使用公开数据集

**📈 对比分析**

通过蒙特卡洛仿真比较理论CRB与无偏估计器的均方误差，结果显示FAS在相同SNR下比2天线FPA低约10 dB，在N=2000时可逼近10天线FPA的相干CRB

**⚠️ 局限性**

仅为理论下界，假设导频理想、通道满足块相关模型，未考虑硬件失真、导频干扰及多用户干扰的实际表现；连续化假设不一定能完全反映离散二值检测的性能

---

## 371. Time-Optimal Construction of String Synchronizing Sets

**arXiv ID:** 2602.11324 | [PDF](https://arxiv.org/pdf/2602.11324v1)

**作者:** Jonas Ellert `[一作]` (Ecole Normale Superieure de Paris), Tomasz Kociumaka `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 1350 | [OpenAlex ID](https://openalex.org/A5086467798)

**关键词:** `Data Structures and Algorithms` `Text`

**🎯 论文内容**

本文提出一种在词 RAM 模型下对任意长度 n、任意 τ∈[1, n/2] 的字符串 T，先以 O(n/ logσ n) 时间预处理，再在 O(n/τ)（或更快的 O((n logτ)/(τ log n))）时间内构造 τ‑同步集，并提供支持 O(1) select 与 O(log logτ / log log n) rank 的稀疏编码表示。

**💡 创新点**

创新点在于：①实现了子线性预处理与最优构造时间；②首次将局部一致性与同步集结合成可变长度稀疏编码；③提出了一种可对稀疏整数序列进行高速 transducer 处理的通用框架；④改进 van Emde Baus 结构，使其在确定性线性构造后仍保留最优查询时间；⑤实现了整体 O((n logτ)/(τ log n)) 构造时间与 O(n/τ) 空间的最优实现。

**🔧 技术方法**

主要技术：
- 局部一致性与同步集理论；
- 约束重压缩与上下文窗口的分层处理；
- 变长编码与稀疏整数序列的 variable‑length 方案；
- 基于预计算表的 transducer 加速（单流与多流）；
- 可变长度稀疏编码的快速解析与查询；
- 改进的 deterministic van Emde Baus 树实现；
- 位运算与 bitmask 的组合使用。

**📊 数据集**

论文没有给出具体实验数据集，理论分析基于标准文本/ DNA 等典型字符串；实验验证（如使用 SDSL、DNA 组装数据集）可在后续工作中补充。

**📈 对比分析**

与先前方法（Kempa‑Kociumaka 仅在 τ<Θ(logσ n) 时无预处理、Kociumaka‑Radoszewski‑Rytter‑Waleń 预处理后仍需 O(n/τ) 构造时间）相比，本文实现了统一的子线性预处理和最优构造时间。rank/select 支持在 τ≤ n^{1‑Ω(1)} 时达到已知下界，且空间与信息量一致。

**⚠️ 局限性**

局限性：
- 仅适用于词 RAM 模型，机器字长必须为 Θ(log n)；
- 实现复杂度高，构造过程涉及多级预计算与稀疏编码，实际工程实现可能成本大；
- 对极大字符串或不同字节模型（如 64 位字长）可能需进一步改进；
- 预处理时间 O(n/ logσ n) 对某些实时或内存受限场景可能不够理想。

---

## 372. Sample-Free Safety Assessment of Neural Network Controllers via Taylor Methods

**arXiv ID:** 2602.11332 | [PDF](https://arxiv.org/pdf/2602.11332v1)

**作者:** Adam Evans `[一作]` (University of Auckland), Roberto Armellin `[通讯]` (University of Auckland)

**通讯引用:** 1945 | [OpenAlex ID](https://openalex.org/A5050067041)

**关键词:** `Machine Learning` `Safty and Privacy` `Explainability and Interpretability` `Tabular`

**🎯 论文内容**

本文提出了一种利用泰勒多项式逼近与自动域分割（ADS）对训练好的神经网络反馈控制器进行全域安全性评估的方法，能够在给定的状态空间范围内生成事件映射并给出严格的安全边界。

**💡 创新点**

创新点在于将自动域分割与多项式界定相结合，既能保证泰勒展开在局部高精度，又通过分域实现对大范围状态不确定性的完整覆盖，从而得到可证明的安全区划；此外，该方法可视化为“热图”，为可解释人工智能提供新的安全评估工具。

**🔧 技术方法**

使用技术包括差分代数（DA）实现高阶泰勒展开、事件映射构造、自动域分割（ADS）进行误差控制、区间算术与多项式界定、SIREN 激活函数的深度神经网络训练，以及对事件值进行后验阈值设定。

**📊 数据集**

数据集：①相对轨道运动（Clohessy–Wiltshire）— 训练集 10M、验证集 100k、测试集 1k；②地月转移— 训练集 10M、验证集 100k、测试集 1k，均由先前构造的多项式指导映射生成。

**📈 对比分析**

与传统 Monte Carlo 模拟对比：在 CW 场景中，安全评估覆盖率达 92.2%，并精确划分 35 个子域为安全/不安全；在地月转移中，检测到大面积不安全子域，表明该控制器不满足安全阈值。该方法提供严格的安全界限而非概率估计，显示出显著的可证明性优势。

**⚠️ 局限性**

局限性：依赖控制器光滑性，无法处理非光滑或分段控制器；区间算术仍会产生过度估计（包络效应）导致安全区划可能保守；分域过程计算量随误差容忍度下降而显著增加，对高维问题的可扩展性仍有待进一步验证。

---

## 373. TRACE: Timely Retrieval and Alignment for Cybersecurity Knowledge Graph Construction and Expansion

**arXiv ID:** 2602.11211 | [PDF](https://arxiv.org/pdf/2602.11211v1)

**作者:** Zijing Xu `[一作]` (Tsinghua University), Mingwei Xu `[通讯]` (Tsinghua University)

**通讯引用:** 3888 | [OpenAlex ID](https://openalex.org/A5100771111)

**关键词:** `Cryptography and Security` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

提出并实现TRACE框架，用于通过融合24个结构化数据库与3类非结构化来源（APT报告、论文、修复通告）构建并持续扩展大规模的网络安全知识图谱。

**💡 创新点**

创新点在于构建可扩展的多维网络安全本体，利用LLM结合检索增强生成（RAG）实现高效的实体抽取与对齐，实现知识图谱规模与覆盖率的显著提升。

**🔧 技术方法**

采用大语言模型（LLM）配合RAG和few-shot学习进行实体抽取，使用prompt工程与句向量相似度对齐，结合MongoDB存储与维护图谱。

**📊 数据集**

使用的主要数据集包括24个结构化数据库，以及约10,205篇论文、869份APT报告和6,784份修复通告等非结构化文本。

**📈 对比分析**

通过与现有最大规模网络安全知识图谱BRON、CSKG4APT对比，TRACE在节点、边数、节点类型、边类型上分别提升1.8×、1.79×、4.67×、11.2×；实体抽取的精确率86.08%、召回率76.92%、F1 81.24%，在LLM基准上提升约7.8%。

**⚠️ 局限性**

局限性包括：仍存在部分孤立节点；LLM在实体抽取与分类中易出现幻觉误判；缺乏对图像等多模态信息的利用，导致部分关键信息未被提取。

---

## 374. PosterOmni: Generalized Artistic Poster Creation via Task Distillation and Unified Reward Feedback

**arXiv ID:** 2602.12127 | [PDF](https://arxiv.org/pdf/2602.12127v1)

**作者:** Sixiang Chen `[一作]` (Hong Kong University of Science and Technology), Lei Zhu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 72420 | [OpenAlex ID](https://openalex.org/A5100394072)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Knowledge Distillation` `Reinforcement Learning` `Reinforcement Learning` `Diffusion model` `Image` `Benchmark`

**🎯 论文内容**

构建了一个统一的多任务图像到海报生成框架 PosterOmni，支持海报扩展、填充、重缩放、身份驱动、布局驱动和风格驱动六类任务；

**💡 创新点**

创新点在于：将任务拆分为局部编辑与全局创作两大类；通过任务蒸馏将不同专家（局部与全局）知识融合到单一学生网络；设计统一奖励模型并结合 DiffusionNFT 强化学习，实现局部精度与全局美学双重优化；同时构建大规模自动化数据集 PosterOmni‑200K 与 Benchmark；

**🔧 技术方法**

技术手段包括 Flow‑matching 采样、DiffusionNFT 强化学习、任务蒸馏、统一 Reward 模型、LoRA 微调、SAM/BrushNet/PaddleDet 等工具；

**📊 数据集**

使用的主要数据集为 PosterOmni‑200K（约20万条任务对齐样本，涵盖六种任务与六大主题）以及 PosterOmni‑Bench（480/540 条人工评测样本）；

**📈 对比分析**

在 PosterOmni‑Bench 上与六个开源编辑模型和两款商业系统对比，PosterOmni 在六项任务上均超越所有开源模型，整体性能接近甚至超过 Seedream‑4.0；人工评测亦显示与商业系统相当；

**⚠️ 局限性**

局限性包括：部分数据为合成，缺少极端长尾真实场景；仅支持单轮编辑，未覆盖多轮交互与多页面布局；未来需扩展多样化真实样本和交互式创作流程。

---

## 375. A DMD-Based Adaptive Modulation Method for High Dynamic Range Imaging in High-Glare Environments

**arXiv ID:** 2602.12044 | [PDF](https://arxiv.org/pdf/2602.12044v1)

**作者:** Banglei Guan `[一作]` (National University of Defense Technology), Qifeng Yu `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Optimization` `Image`

**🎯 论文内容**

构建了基于数字微镜装置（DMD）的自适应HDR成像系统，用于在极端眩光条件下实现无失真光学测量。

**💡 创新点**

创新点在于：①采用DMD实现像素级实时光强调制，突破传统CCD/CMOS 70 dB以下的动态范围；②实现闭环自适应掩模生成，单帧内消除高光饱和；③通过双伪全像差光学设计和实时算法实现可达120+ dB动态范围与78%应变误差降低。

**🔧 技术方法**

主要技术包括：DMD二进制脉宽调制光学调制、双伪全像差光学系统、实时掩模生成与图像重建算法、sCMOS高帧率捕获、硬件软件协同控制。

**📊 数据集**

使用自制实验数据：拉伸变形试验在高光环境下的多曝光图像与DMD调制图像（无公开数据集）。

**📈 对比分析**

与图像增强算法（CLAHE、LIME、DONG、AMEF）以及硬件HDR相机（SVE）进行对比；在动态范围、应变误差、有效计算面积等指标上，DMD系统分别实现127 dB动态范围、78%应变误差降低、100%有效计算面积，整体性能显著优于对比方法。

**⚠️ 局限性**

局限性在于：对激光高能脉冲引起的局部过曝仍有一定敏感性；系统对极端光照下的实时性和体积仍需进一步优化；实验多为实验室环境，缺乏公开数据集验证。

---

## 376. Pack it in: Packing into Partially Filled Containers Through Contact

**arXiv ID:** 2602.12095 | [PDF](https://arxiv.org/pdf/2602.12095v1)

**作者:** David Russell `[一作]` (University of Leeds), Mehmet Dogar `[通讯]` (University of Leeds)

**通讯引用:** 2421 | [OpenAlex ID](https://openalex.org/A5025603635)

**关键词:** `Robotics` `Robotic Intelligence` `Optimization` `Reinforcement Learning`

**🎯 论文内容**

在仓库装箱中，提出了一套基于物理感知和模型预测控制的接触式装箱方法，能在部分已装箱的容器内通过非抓握手段创建空间并放置新物体。

**💡 创新点**

创新点在于将装箱过程视为单一的接触轨迹优化任务，结合高层位置规划器与实时物理感知，实现了同时重新排列与插入，且兼顾非预抓握对象。

**🔧 技术方法**

使用的技术包括 iLQR 轨迹优化、MuJoCo 物理模拟、模型预测控制 (MPC)、OptiTrack 姿态估计与物理预测融合，以及自定义的成本函数与残差。

**📊 数据集**

数据集为实验中随机挑选的五种常见商店物品（圆柱体与盒子）以及目标物体（番茄酱罐），共 40 次实机试验和 100 次仿真。

**📈 对比分析**

通过与两种基线（无位置规划与无物理交互）对比，PackItIn 在实机上实现 70% 成功率（比基线约 17%），仿真中 93% 成功率；相对基线显著提升。

**⚠️ 局限性**

局限性包括对目标物体与泵头刚性假设不稳、视角遮挡导致的感知误差、以及对物理模型不精确导致的执行误差。

---

## 377. HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model

**arXiv ID:** 2602.11758 | [PDF](https://arxiv.org/pdf/2602.11758v1)

**作者:** Dongting Li `[一作]` (Tsinghua University), Renjing Xu `[通讯]` (HKUST Guangzhou)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Reinforcement Learning` `World Model` `Point Cloud`

**🎯 论文内容**

提出一种面向半主动对象和多场景环境的类人机器人全身交互框架，能够在视觉盲区下通过本体感知预测对象动力学并实现高效交互。

**💡 创新点**

核心创新在于：① 将高阶动力学预测与几何投影相结合的“Dynamics‑Aware World Model”；② 采用非对称细化训练，使世界模型在探索过程中持续适配，解决分布漂移；③ 通过隐私强化学习实现无外部传感器的全身交互。

**🔧 技术方法**

使用强化学习（PPO）训练的两阶段教师-学生框架，结合动态世界模型、点云几何投影、优先级强化学习、EMA平滑等技术；在模拟环境中采用 Isaac Sim、MuJoCo 进行训练。

**📊 数据集**

通过光学动作捕捉系统记录人类与对象交互数据（包含各种半主动对象、长时序多物体以及多地形场景），随后在模拟中重现这些数据；真实实验使用自研类人机器人硬件及内部惯性测量。

**📈 对比分析**

与基线 HDMI*（仅本体感知）对比，实验显示在滑板、推拉车、长时序多物体、跨地形携盒等任务中成功率从 0‑30% 提升至 80‑100%，轨迹误差、姿态误差显著下降，学习效率更高。

**⚠️ 局限性**

局限性包括：① 仍需较大规模的仿真与域随机化来保证 sim‑to‑real；② 对极端重量/尺寸的对象泛化尚未充分验证；③ 对动态环境（如移动障碍物）的实时感知依赖尚不充分，可能导致碰撞风险。

---

## 378. BIRD: A Museum Open Dataset Combining Behavior Patterns and Identity Types to Better Model Visitors' Experience

**arXiv ID:** 2602.11160 | [PDF](https://arxiv.org/pdf/2602.11160v1)

**作者:** Alexanne Worm `[一作]` (University of Lorraine), Sylvain Castagnos `[通讯]` (University of Lorraine)

**关键词:** `Human-Computer Interaction` `Recommendation System` `Tabular` `Time Series`

**🎯 论文内容**

对51名博物馆访客进行眼动追踪与行为记录，收集轨迹、观展时长、问卷等多维度数据，构建了名为BIRD的公开博物馆访客行为与身份数据集。

**💡 创新点**

创新点在于整合了情境、行为与身份三大维度的数据，首次提供了包含完整轨迹、观展停留、情感反馈及问卷调查结果的全景式访客画像；并以此数据集为基础验证并复现Veron‑Levasseur访客身份模型，提升了推荐系统和人群模拟研究的可用性。

**🔧 技术方法**

主要技术包括：Tobii Glasses 2 眼动追踪仪与场景摄像、Web平台人工标注轨迹与观展对象、MovingPandas库进行轨迹去噪与均匀采样、K‑means聚类与多指标（WCSS、Silhouette、Davies‑Bouldin、Calinski‑Harabasz）评估聚类效果。

**📊 数据集**

使用的是本文自己构建的BIRD数据集（包含轨迹、观展对象列表、问卷信息、反馈选择等文件），该数据集已公开可下载。

**📈 对比分析**

对轨迹特征（时长、速度、停留数、路径长度、观展对象数）进行聚类，利用K‑means并结合上述评估指标确定最佳聚类数为4，聚类结果与Veron‑Levasseur模型的“Grasshopper”“Ant”“Fish”“Butterfly”身份相对应，验证了数据集在身份建模上的有效性。

**⚠️ 局限性**

局限性包括样本量仅51人、数据收集时间有限、仅涵盖绘画作品且未完全记录雕塑等其他展品、部分访客未提供完整问卷、缺乏完整眼动数据与更广泛的情境多样性，且数据可能不易直接推广到其他博物馆环境。

---

## 379. EasyMimic: A Low-Cost Framework for Robot Imitation Learning from Human Videos

**arXiv ID:** 2602.11464 | [PDF](https://arxiv.org/pdf/2602.11464v1)

**作者:** Tao Zhang `[一作]`, Qin Jin `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Transformer` `Vision Language Model` `Video`

**🎯 论文内容**

本文提出了一套低成本、易复现的机器人学习系统，利用人类视频来训练机器人完成抓取与放置、推送等操作。

**💡 创新点**

创新点在于：①将人类手部动作空间与机器人动作空间对齐；②通过视觉增益减少域差距；③使用共训练或预训练-微调两种策略融合人类与机器人数据；④在低成本硬件上实现完整实验流程。

**🔧 技术方法**

技术手段包括：人类手部运动提取、动作空间对齐（Thenar、Wrist）、视觉增益（全手、指尖、指尖-拇指组合）、动作空间对齐与文本增益、ViT+LLM冻结、状态动作增益、视觉增益等训练增强。

**📊 数据集**

使用数据集：30例机器人数据、100例人类视频数据；人类视频中包含小鸭在三种初始位置的抓取与放置动作；硬件记录由D435i RGB摄像头捕获。

**📈 对比分析**

通过10次80秒实验对比不同策略，发现共训练+重定位（RH）与预训练-微调+重定位（RH）相较于仅使用机器人数据的基线显著提升（0/10提升到2+0.5+0.5/10），表明动作空间对齐与数据增益有效。

**⚠️ 局限性**

局限性包括：仅测试了简单的抓取、放置和推送任务；数据量有限，难以覆盖更复杂场景；部分增益方法（如文本增益、视觉增益）在实验中未体现明显效果；系统仍对实时性和更大域差距的适应性不足。

---

## 380. Code Mixologist : A Practitioner's Guide to Building Code-Mixed LLMs

**arXiv ID:** 2602.11181 | [PDF](https://arxiv.org/pdf/2602.11181v1)

**作者:** Himanshu Gupta `[一作]` (Arizona State University), Neeraj Varshney `[通讯]` (Arizona State University)

**通讯引用:** 1631 | [OpenAlex ID](https://openalex.org/A5004217423)

**关键词:** `Computation and Language` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Prompt Engineering` `Reinforcement Learning` `Text` `Review/Survey Paper` `Benchmark`

**🎯 论文内容**

本文综述了在大型语言模型（LLM）时代处理代码混合（CSW）的研究现状，并构建了从数据、建模、提示、评估到安全的统一税onomies，提供了面向实践者的操作手册。

**💡 创新点**

创新点在于：① 将以往分散的 CSW 研究聚合为完整生命周期框架；② 提出了针对 LLM 的专门预训练和后训练技术（如 SWITCHMLM、RESBERT、PRO‑CS 等）；③ 结合安全视角，阐释代码混合如何成为攻击向量并给出对策；④ 提倡人类中心的多维评估体系而非单一 n‑gram 指标。

**🔧 技术方法**

技术方法包括：预训练的边界感知 MLM（SWITCHMLM、FREQMLM）、架构改造（RESBERT、CMLFormer）、提示与上下文学习（ICM、CSICL、结构约束提示）、数据生成（对齐替换、GLOSS、Filter‑Then‑Finetune、EZSwitch、伪平行构建）、指令调优（COMMIT、PRO‑CS）、RL 对齐（RLAIF）、对齐学习（ContrastiveMix、ConCSE）等。

**📊 数据集**

使用的数据集与基准包括：LinCE、GLUECoS、COMI‑LINGUA、CodeMixEval、Lost in the Mix、ENKOQA、CS‑Sum，以及各类多语种语料库（如通用预训练语料、对齐语料、混合语料生成管道）。

**📈 对比分析**

对比方法：在多层评估套件上进行基准测试，量化代码混合缺口（Code Mixing Gap）并将结果分层展示（按混合率、脚本类型、切换位置）。实验表明，结合预训练+后训练的 CSW‑aware 模型在多项指标上明显优于仅做标准多语种预训练的模型，但在长上下文、低资源对话等场景仍有显著退化。

**⚠️ 局限性**

局限性：① 研究仍以英语为中心，非英语/多语种对的覆盖不足；② 评估指标（如 CMI、I‑Index）对语义真实度把握有限；③ 对长上下文推理、RL 优化的探索不足；④ 数据生成与过滤的人工成本高；⑤ 安全对齐仍主要针对英语，混合语境下的守护机制不成熟。

---

## 381. Oscillators Are All You Need: Irregular Time Series Modelling via Damped Harmonic Oscillators with Closed-Form Solutions

**arXiv ID:** 2602.12139 | [PDF](https://arxiv.org/pdf/2602.12139v1)

**作者:** Yashas Shende `[一作]` (Ashoka University), Debayan Gupta `[通讯]` (Ashoka University)

**通讯引用:** 356 | [OpenAlex ID](https://openalex.org/A5073351462)

**关键词:** `Machine Learning` `Classification` `Anomaly Detection` `Transformer` `Ordinary Differential Equation` `Time Series` `Finance Related`

**🎯 论文内容**

提出一种闭式解的线性阻尼振荡器Transformer模型（Osciformer），在不需要数值ODE求解的情况下实现对不规则时间序列的连续时间注意力；

**💡 创新点**

创新点在于将连续时间注意力映射为振荡器的共振现象，并证明该振荡器族保持了对连续注意力的全局逼近能力；

**🔧 技术方法**

使用线性阻尼驱动谐振子、频率基函数、闭式解、软最大化、以及标准Transformer的投影与前馈网络；

**📊 数据集**

在多种不规则时间序列基准上评测，包括健康（MIMIC, HR）、金融（BookOrder, StackOverflow）、交通（Traffic）、事件预测、UCR长期上下文数据集以及合成正弦/螺旋/二进制序列；

**📈 对比分析**

与ContiFormer、Rough Transformers、CDE/NRDE模型、GRU、ODE‑RNN等对比；在事件预测、分类与回归任务中获得与ContiFormer相当甚至更优的log‑likelihood/准确率，且在大多数任务中显著加速（3×–20×）且内存占用大幅降低；

**⚠️ 局限性**

局限性包括对阻尼、频率网格的超参数敏感，理论泛化误差仍不完善，且在极端稀疏或高维特征情况下性能未完全验证。

---

## 382. Making the complete OpenAIRE citation graph easily accessible through compact data representation

**arXiv ID:** 2602.12206 | [PDF](https://arxiv.org/pdf/2602.12206v1)

**作者:** Joakim Skarding `[一作]` (Institute of Computer Science of the Czech Academy of Sciences), Pavel Sanda `[通讯]` (Institute of Computer Science of the Czech Academy of Sciences)

**关键词:** `Social and Information Networks` `Compression` `Graph` `Tabular`

**🎯 论文内容**

对OpenAIRE完整图谱进行大规模处理，提取出版物与引用关系，压缩至32GB左右，输出可直接使用的CSV文件，并提供可复用的Python处理管线。

**💡 创新点**

创新在于将超过2TB、200M+文献、20亿+引用的知识图谱压缩为可在普通机器上加载的精简版，同时保留完整结构，提供内存友好的节点ID映射与极简数据格式，减少处理门槛。

**🔧 技术方法**

采用PySpark对关系文件进行并行处理，使用哈希表映射OpenAIRE ID为短整数节点ID；Python脚本完成字段提取、JSON扁平化及质量控制；输出CSV供Pandas等工具直接读取。

**📊 数据集**

使用的主要数据集是2025‑12‑01版OpenAIRE dump（>200M出版物，>20亿引用），以及处理后得到的distilled OpenAIRE citation graph（约32GB）。

**📈 对比分析**

通过对比原始dump与处理后文件的磁盘与内存占用（citations.csv 39GB→16GB内存，publications.csv 6GB→16GB内存；原始edges 1820GB），验证数据完整性且显著降低资源需求；处理时间在标准PySpark集群上完成。

**⚠️ 局限性**

局限性包括仅保留“Cites”关系，忽略其他多样化实体与关系；处理过程需要一定规模的计算资源；对动态更新支持有限，需重新运行管线才能捕获新增数据。

---

## 383. Counterfactual Conditional Likelihood Rewards for Multiagent Exploration

**arXiv ID:** 2602.11740 | [PDF](https://arxiv.org/pdf/2602.11740v1)

**作者:** Ayhan Alp Aydeniz `[一作]` (Oregon State University), Kagan Tumer `[通讯]` (Oregon State University)

**通讯引用:** 4267 | [OpenAlex ID](https://openalex.org/A5084748531)

**关键词:** `Multiagent Systems` `Reinforcement Learning` `Robotic Intelligence` `Recurrent Neural Network` `Reinforcement Learning` `Sequential`

**🎯 论文内容**

提出Counterfactual Conditional Likelihood (CCL) intrinsic reward，用来度量每个智能体对团队联合探索的唯一贡献，从而促进多智能体协作探索；

**💡 创新点**

创新点在于通过对比真实观测与对抗性假设观测（即保持前一时刻观测不变）来计算条件似然差值，利用随机编码器将每个观测映射到低维空间并在联合嵌入空间中进行k‑NN密度估计，形成可计算且对非平稳性稳健的奖励；

**🔧 技术方法**

采用随机编码器（fixed MLP）对局部观测进行编码，构建联合嵌入；利用共享半径的k‑NN估计密度并用digamma函数近似对数似然；在训练中使用MAPPO+LSTM结构，并将CCL与本地观测熵奖励（OEM）混合；

**📊 数据集**

在连续稀疏奖励的多无人车（multi‑rover）域（1–2个POI，耦合因子3–6，3–10名智能体）以及三种粒子环境（predator‑prey、keep‑away、physical deception）中进行实验；

**📈 对比分析**

与仅使用本地观测熵奖励和混合奖励进行比较。实验显示，CCL能显著加速学习、提高团队奖励、减少冗余探索，混合奖励在易任务中加快收敛但在高耦合/单POI任务中效果有限；

**⚠️ 局限性**

局限在于当智能体数量增多时，k‑NN密度估计的准确性下降；随机编码器固定可能限制表达能力；缺乏模型基探索或对非平稳性更鲁棒的改进方法；

---

## 384. Real-World Asset Integration in Next-Generation Communication Networks: Fundamental, Framework, and Case Study

**arXiv ID:** 2602.11798 | [PDF](https://arxiv.org/pdf/2602.11798v1)

**作者:** Tingxuan Su `[一作]` (University of Electronic Science and Technology of China), Dusit Niyato `[通讯]` (Nanyang Technological University)

**通讯引用:** 83379 | [OpenAlex ID](https://openalex.org/A5091266202)

**关键词:** `Networking and Internet Architecture` `Agentic AI`

**🎯 论文内容**

本文提出基于RWA的网络资源代币化框架，旨在解决下一代网络的流动性与安全挑战，并通过动态频谱分配案例验证其有效性。

**💡 创新点**

创新点在于将RWA与双模式（租赁与购买）结合，采用ERC‑3643合规代币、状态通道与AMM机制，实现网络资源的可分割交易与抵御拜占庭攻击。

**🔧 技术方法**

采用区块链技术实现代币化（ERC‑3643），智能合约、状态通道、自动做市商、BFT协议以及Python Mesa仿真环境。

**📊 数据集**

使用基于Agent的仿真数据，模拟100–300名买卖双方、5秒区块间隔和10笔交易区块大小等参数；未使用真实业务数据集。

**📈 对比分析**

通过与MPRA、TRA、CPA三种传统资源分配方案在相同仿真条件下比较，评估资源利用率与对买卖方串通、默认等攻击的鲁棒性，RWA在资源稀缺时的利用率最高且攻击下仍保持近100%利用率。

**⚠️ 局限性**

局限性包括智能合约部署与审计成本高、区块链上链费用与延迟影响效率、隐私保护难题以及缺乏统一监管框架导致资产可代币化受限。

---

## 385. Anonymous Contracts

**arXiv ID:** 2602.12118 | [PDF](https://arxiv.org/pdf/2602.12118v1)

**作者:** Johannes Brustle `[一作]`, Matteo Russo `[通讯]` (EPFL)

**关键词:** `Computer Science and Game Theory`

**🎯 论文内容**

本文研究多代理合同模型，提出并分析匿名支付方案及其在有限责任和无限责任下的表现。

**💡 创新点**

创新点在于首次引入匿名（对成功数计费）与统一匿名（对成功数不计费）合同，证明其存在纯纳什均衡、唯一性以及在不同情形下的近似保真度与性能下界。

**🔧 技术方法**

主要采用博弈论与潜在函数证明、组合优化与概率分析、对成功概率比值Q的对数刻画等技术。

**📊 数据集**

作为理论工作，未使用实测数据集，而是构造极端实例进行性能下界与上界的证明。

**📈 对比分析**

通过与最优非匿名合同的社会福利比较，证明有限责任下匿名合同的近似比为O(min{n,logQ})，无限责任下可达O(logn)，并在概率区分时可完全提取社会福利。

**⚠️ 局限性**

局限性包括对成功概率相同或极端分布敏感、需要有限责任假设时性能受限、以及在实际应用中对支付负值的可行性需进一步探讨。

---

## 386. Towards Performance-Enhanced Model-Contrastive Federated Learning using Historical Information in Heterogeneous Scenarios

**arXiv ID:** 2602.11945 | [PDF](https://arxiv.org/pdf/2602.11945v1)

**作者:** Hongliang Zhang `[一作]` (Shandong Computer Science Center), Chunqiang Hu `[通讯]` (Chongqing University)

**通讯引用:** 3308 | [OpenAlex ID](https://openalex.org/A5007408033)

**关键词:** `Machine Learning` `Federated Learning` `Convolutional Neural Network` `Contrastive Learning` `Image`

**🎯 论文内容**

提出了一种面向异构场景的性能增强模型对比联邦学习框架 PMFL，旨在同时缓解数据异构和参与异构对 FL 性能的影响。

**💡 创新点**

创新点包括：① 将历史本地模型引入模型对比项构建稳定对比点；② 基于累计参与次数自适应调整聚合权重以纠正参与异构偏差；③ 将历史全局模型融入全局更新以抑制低参与率下的性能波动。

**🔧 技术方法**

核心技术包括：对比学习、滑动缓冲（历史本地/全局模型）、自适应聚合权重、温度调节的对比损失、滑动平均与动态系数 ψ。

**📊 数据集**

使用四个公开数据集：SVHN、CIFAR10、CINIC、CIFAR100，采用卷积网络（Encoder+Projection+Classifier）进行实验。

**📈 对比分析**

与 FedVarp、MIFA、FedHyper、FedAU、FedPPO 等基线方法对比，PMFL 在不同参与模式（Bernoulli、Markovian、Cyclic）及不同数据/参与异构程度下均实现了更高的测试准确率，表现最优。

**⚠️ 局限性**

局限性：需要额外的历史模型存储与管理，计算量与训练时长略高；对超参数（如缓冲大小、切断间隔 C、对比系数 λ）敏感；未验证模型异构（非同构网络结构）情况下的鲁棒性。

---

## 387. OMEGA-Avatar: One-shot Modeling of 360° Gaussian Avatars

**arXiv ID:** 2602.11693 | [PDF](https://arxiv.org/pdf/2602.11693v1)

**作者:** Zehao Xia `[一作]` (Chongqing University), Peter Wonka `[通讯]` (KAUST)

**关键词:** `Graphics` `Generation` `Data Synthesis` `Diffusion model` `Gaussian Splatting` `Image` `Mesh`

**🎯 论文内容**

提出了一个一次性前馈框架，可仅凭一张图像生成完整、可动画化的3D高保真Gaussian头像；

**💡 创新点**

创新点在于结合语义感知网格变形和多视角特征涂射，利用多视角法线引导FLAME头部优化，并在UV空间融合全视角特征，实现360°完整性与可动画化；

**🔧 技术方法**

采用3D Gaussian splatting、FLAME参数化网格、扩散模型生成多视角图像与法线、语义感知拉普拉斯正则、可微分双线性涂射、层次UV映射、可见性加权融合以及神经后处理等技术；

**📊 数据集**

在NeRSemble和Avatar-256公开数据集上进行训练与评估，并通过自定义的野外图像进行泛化测试；

**📈 对比分析**

与PanoHead、SphereHead、GAGAvatar、LAM、SOAP等方法对比，实验显示在PSNR、SSIM、LPIPS、DS、CSIM等指标上均实现了SOTA，尤其在多视角完整性与身份保持方面显著优于基线；

**⚠️ 局限性**

局限性包括：依赖扩散模型合成多视角法线，可能在极端发型或姿态下效果受限；并非完全“一图像无视角生成”，仍需通过多视角法线提升完整性。

---

## 388. Spectra: Rethinking Optimizers for LLMs Under Spectral Anisotropy

**arXiv ID:** 2602.11185 | [PDF](https://arxiv.org/pdf/2602.11185v1)

**作者:** Zhendong Huang `[一作]` (Fudan University), Li Shang `[通讯]` (Fudan University)

**通讯引用:** 6397 | [OpenAlex ID](https://openalex.org/A5004722925)

**关键词:** `Machine Learning` `Optimization` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

设计并评估了一种针对LLM梯度低秩尖峰的优化器Spectra，能够抑制梯度尖峰而不放大尾部

**💡 创新点**

创新点在于仅对尖峰子空间做谱形状调整，避免全谱平坦化导致的噪声放大

**🔧 技术方法**

使用热身的幂迭代估计低秩尖峰并进行局部谱缩放，配合RMS归一化

**📊 数据集**

在Qwen3‑0.6B和LLaMA3‑8B上使用约50B–100B token的预训练语料

**📈 对比分析**

与AdamW和Muon对比，Spectra 30%更快收敛、显存/优化器状态减半、下游平均准确率提升1–1.6%，对学习率更鲁棒

**⚠️ 局限性**

对尾部高方差方向的处理仍有限，且在更大规模模型或不同任务上需进一步验证

---

## 389. Towards Sustainable Investment Policies Informed by Opponent Shaping

**arXiv ID:** 2602.11829 | [PDF](https://arxiv.org/pdf/2602.11829v1)

**作者:** Juan Agustin Duque `[一作]` (University of Montreal), Aaron Courville `[通讯]` (University of Montreal)

**关键词:** `Machine Learning` `Reinforcement Learning` `Optimization` `Recommendation System` `Finance Related` `Reinforcement Learning` `Agentic AI` `Tabular` `Time Series` `Finance Related`

**🎯 论文内容**

本文通过对InvestESG环境进行理论分析与实验验证，证明其可成为时序社会困境，并使用优势对齐（Advantage Alignment）在该环境中引导投资者与公司行为，提升社会福利。

**💡 创新点**

创新点包括：①在特定参数下正式证明InvestESG形成社会困境；②将优势对齐算法应用到高维、连续动作的真实气候金融仿真中；③从理论上解释优势对齐为何更倾向于合作。

**🔧 技术方法**

使用技术主要有多智能体强化学习、优势对齐（Proximal Advantage Alignment）、PPO、IPPO、MAPPO、GAE、自我对弈（self‑play）以及梯度理论分析。

**📊 数据集**

使用数据集为InvestESG仿真环境内部生成的数据，涵盖公司投资决策、气候风险事件等。

**📈 对比分析**

对比方法包括PPO、IPPO、MAPPO、奖励求和等，实验结果表明优势对齐在所有规模下均能保持高社会福利，而奖励求和在代理数增大时性能显著下降。

**⚠️ 局限性**

局限性在于：实验仅基于模拟环境，缺乏真实金融市场数据验证；优势对齐对Critic估计误差敏感；对高度动态气候模型的可推广性仍待进一步验证。

---

## 390. AC-MASAC: An Attentive Curriculum Learning Framework for Heterogeneous UAV Swarm Coordination

**arXiv ID:** 2602.11735 | [PDF](https://arxiv.org/pdf/2602.11735v1)

**作者:** Wanhao Liu `[一作]` (Guangdong University of Technology), Panshuo Li `[通讯]` (Guangdong University of Technology)

**通讯引用:** 2243 | [OpenAlex ID](https://openalex.org/A5016024940)

**关键词:** `Robotics` `Reinforcement Learning` `Robotic Intelligence` `Reinforcement Learning` `Agentic AI`

**🎯 论文内容**

提出了一种面向异构无人机编队的协同路径规划方法 AC-MASAC，利用注意力机制的异构演员-评论家架构和结构化课程学习实现高效的多智能体强化学习。

**💡 创新点**

创新点在于：①为领导者与跟随者设计专属注意力机制，显式建模异构角色间的非对称依赖；②引入结构化课程学习与分阶段经验回放、知识迁移策略，缓解稀疏奖励与灾难性遗忘问题；③在软演员-评论家框架下融合最大熵目标与双Q校正，提升样本效率和训练稳定性。

**🔧 技术方法**

采用的技术包括：多智能体强化学习（CTDE + SAC），Soft Actor-Critic + 双Q网络，跨实体注意力机制，层级知识迁移，阶段比例经验回放，模拟环境与Pygame可视化。

**📊 数据集**

使用自定义仿真数据集：OpenAI Gym + Pygame 生成的多层级测试世界（700×600m 空域），涵盖不同数量的领导者、跟随者及动态障碍物，包含 100 个随机初始化场景。

**📈 对比分析**

与 MASAC、MADDPG（学习型）和 RRT*（非学习型）基线对比，实验结果显示 AC‑MASAC 在成功率（SR）、成功加权任务时间（SMT）和编队保持率（FKR）方面均显著优于基线，尤其在复杂环境下 FKR 提升超过 15%。

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证，缺乏真实硬件部署与 sim‑to‑real 转移实验；通信不完善时鲁棒性仍有限；对大规模编队（数十个 UAV）在计算开销和训练时间方面尚未充分评估。

---

## 391. Safe Fairness Guarantees Without Demographics in Classification: Spectral Uncertainty Set Perspective

**arXiv ID:** 2602.11785 | [PDF](https://arxiv.org/pdf/2602.11785v1)

**作者:** Ainhize Barrainkua `[一作]` (Basque Center for Applied Mathematics), Jose A. Lozano `[通讯]`

**关键词:** `Machine Learning` `Classification` `Optimization` `Robust Risk Minimization` `Minimax Fairness` `Tabular`

**🎯 论文内容**

提出一种无须敏感属性信息即可提升最差群体准确率的公平学习方法——SPECTRE。

**💡 创新点**

创新点在于利用随机傅里叶特征映射调整频谱，并在鲁棒优化框架中定义谱形态不确定集，同时对最大允许偏差 λ 进行可计算的上限约束，从而避免过度悲观和对异常值的过度关注，提供可解释的最差分布与理论误差界。

**🔧 技术方法**

核心技术包括鲁棒风险最小化（RRM）、最小最大公平性（minimax fairness）、随机傅里叶特征（RFF）、线性规划求解最坏分布、以及对 λ 与 σ 的超参数调优。

**📊 数据集**

使用了美国社区调查（ACS Employment、ACS Income）20个州的真实数据、COMPAS 犯罪再犯预测数据以及德国信用评分（German Credit）数据。

**📈 对比分析**

与基准 LR、XGBoost、以及多种公平性增强方法（RLM、ARL、BPF、SURE、FairEns、MMPF、GDRO、MMPFrel）进行比较；SPECTRE 在最差群体准确率上持续领先，整体准确率仅低 1–4%，并在理论上给出群体误差下界，计算成本低于大多数鲁棒公平方法。

**⚠️ 局限性**

局限性包括：若某群体在训练样本中表现差，无法提升其最差准确率；需要部分训练样本具备敏感属性才能给出群体误差界；对分布漂移的适应性尚未充分验证；超参数 σ 与 λ 的调优仍依赖经验策略。

---

## 392. Active Zero: Self-Evolving Vision-Language Models through Active Environment Exploration

**arXiv ID:** 2602.11241 | [PDF](https://arxiv.org/pdf/2602.11241v1)

**作者:** Jinghan He `[一作]` (Chinese Academy of Sciences), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60408 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `Computer Vision and Pattern Recognition` `Reinforcement Learning` `Optimization` `Transformer` `Reinforcement Learning` `Chain-of-Thought` `Vision Language Model` `Image` `Multimodality`

**🎯 论文内容**

设计了一个三代理自进化框架Active Zero，利用主动视觉环境探索、任务合成和推理优化实现Vision‑Language模型自我演化。

**💡 创新点**

将自对弈从被动静态图像集合转为主动探索开放视觉环境；构建搜索器、提问器、求解器三代理协同进化闭环；通过挑战奖励与多模态多样性约束驱动搜索器；采用统一的GRPO训练方式。

**🔧 技术方法**

使用Group Relative Policy Optimization (GRPO)强化学习；SigLIP‑2视觉检索+FAISS索引；Chain‑of‑Thought (CoT)提示；自监督一致性奖励；多模态多样性惩罚等技术。

**📊 数据集**

预构建1.6M图像环境（The Cauldron + 50 vision‑language 数据集 + Geo3K、UniGeo、GeoQA+），以及12个评测基准（6推理、6通用）作为实验数据。

**📈 对比分析**

与基准模型及自对弈基线（VisionZero、VisPlay、EvolMM）进行对比。在Qwen2.5‑VL‑7B‑Instruct上推理平均分从51.05提升至53.97（+5.7%），通用理解平均从57.51提升至59.77（+3.9%）。在3B模型上亦表现出明显提升，主动探索优于随机采样和未RL优化的搜索器。

**⚠️ 局限性**

对开放网络检索仍需内容过滤与偏见缓解；在真实环境中可能出现不合适内容；对超难或超易样本的判断仍有限；对计算资源需求较高；需要进一步验证对抗性与鲁棒性。

---

## 393. AttentionRetriever: Attention Layers are Secretly Long Document Retrievers

**arXiv ID:** 2602.12278 | [PDF](https://arxiv.org/pdf/2602.12278v1)

**作者:** David Jiahao Fu `[一作]` (University of Illinois Urbana-Champaign), Kevin Chen-Chuan Chang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 10585 | [OpenAlex ID](https://openalex.org/A5101880377)

**关键词:** `Information Retrieval` `Retrieval` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

提出了一种训练无关的长文本检索模型，通过预训练LLM的注意力机制和实体检索实现上下文感知检索并决定检索范围。

**💡 创新点**

创新点在于利用注意力层的交叉注意力得分作为句子级相关性评估，结合稠密嵌入相似度以及基于实体的检索来精确捕获上下文、因果和查询依赖。

**🔧 技术方法**

技术：预训练LLM注意力分数、稠密句子嵌入、实体抽取（SpaCy）、实体图检索、Cascading KV缓存。

**📊 数据集**

使用LongBench‑v2‑Retrieval（35篇>100k词长文本）、QASA、Qasper、RepLiQA、DAPR、HotpotQA、2WikiMultihopQA、MuSiQue等数据集。

**📈 对比分析**

与BM25、DPR、ANCE、GTR、GTE、Qwen3、GritLM、SPScanner等基线比较，在单文档检索上取得最高F1，远优于基线；在多文档检索上表现相当；在QA任务上与直接生成相近，显著降低输入token数。

**⚠️ 局限性**

局限：需要大规模LLM（≈3B参数）导致效率不及稀疏或小型稠密模型；未对更大LLM进行注意力分析；构建的长文档检索数据集规模有限，可能受异常值影响。

---

## 394. Categorical Flow Maps

**arXiv ID:** 2602.12233 | [PDF](https://arxiv.org/pdf/2602.12233v1)

**作者:** Daan Roos `[一作]` (University of Amsterdam), Jan-Willem van de Meent `[通讯]` (University of Amsterdam)

**通讯引用:** 1806 | [OpenAlex ID](https://openalex.org/A5073129092)

**关键词:** `Machine Learning` `Generation` `Data Synthesis` `Knowledge Distillation` `Graph Neural Network` `Flow-based Model` `Diffusion model` `Text` `Image` `Graph`

**🎯 论文内容**

提出了分类流映射（Categorical Flow Maps, CFM）框架，利用端点预测的流匹配方法实现离散数据（如文本、分子、图像）的一步或两步高质量生成，并支持自蒸馏与测试时引导；

**💡 创新点**

创新点在于将流匹配的端点预测转化为在概率单纯形上的约束参数化，使用交叉熵损失自然保持离散性，并设计了端点一致性自蒸馏（ECLD）目标，使得流映射可在离散域自蒸馏；

**🔧 技术方法**

使用变分流匹配（VFM）、流映射框架、交叉熵端点损失、端点一致性自蒸馏、straight‑through 离散化、SMC 采样与指导、以及针对图、图像、文本的专门网络架构；

**📊 数据集**

在分子生成上使用 QM9、ZINC；在二值图像上使用二值化 MNIST；在文本生成上使用 Text8 和 LM1B；

**📈 对比分析**

与多步扩散/流模型、Set2GraphVAE、MoFlow、PairFlow 等方法比较，CFM 在单步/双步下均实现了 state‑of‑the‑art 质量：QM9 单步有效率 95.8%，ZINC 93.5%；二值 MNIST 单步 FID 10.1；Text8 单步 NLL 5.33；LM1B 单步 Gen‑PPL 274.87，均优于或接近多步基线；

**⚠️ 局限性**

局限性包括：在高维/高熵任务上仍需进一步调优，端点一致性损失在极端时间点可能不稳定，SMC 引导的计算开销较大，且在部分离散任务（如更大分子或长文本）上实验尚未充分验证。

---

## 395. RELATE: A Reinforcement Learning-Enhanced LLM Framework for Advertising Text Generation

**arXiv ID:** 2602.11780 | [PDF](https://arxiv.org/pdf/2602.11780v1)

**作者:** Jinfang Wang `[一作]` (Baidu Inc.), Lin Liu `[通讯]` (Baidu Inc.)

**关键词:** `Artificial Intelligence` `Generation` `Optimization` `Recommendation System` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Text`

**🎯 论文内容**

提出了RELATE框架，使用强化学习将广告文本生成与业务目标（CTCVR）及合规、质量、多样性约束统一端到端训练。

**💡 创新点**

将多维奖励（转换率、质量、多样性）与分层信用分配相结合，实现token级与句子级奖励协同学习，突破传统两阶段生成-优化的瓶颈。

**🔧 技术方法**

基于大型语言模型、GRPO强化学习、共享底部多任务学习预测CTCVR，并设计分层信用分配与多维奖励函数。

**📊 数据集**

采用百度搜索广告真实数据集，约40万条查询-广告样本。

**📈 对比分析**

与历史生产NLG模型及Qwen‑SFT+后处理基线对比，在线A/B测试中CTCVR提升9.19%，合规率93.98%，多样性0.82，显著优于基线。

**⚠️ 局限性**

仍受奖励稀疏、延迟反馈影响，奖励设计依赖手工设定，未来需探索更丰富的多样性奖励与更细粒度的信用分配。

---

## 396. Taming Subpacketization without Sacrificing Communication: A Packet Type-based Framework for D2D Coded Caching

**arXiv ID:** 2602.12220 | [PDF](https://arxiv.org/pdf/2602.12220v1)

**作者:** Xiang Zhang `[一作]` (Technical University of Berlin), Mingyue Ji `[通讯]` (University of Florida)

**通讯引用:** 3229 | [OpenAlex ID](https://openalex.org/A5058487273)

**关键词:** `Information Theory` `Optimization` `Integer Linear Programming`

**🎯 论文内容**

本文研究了一种基于数据包类型的框架，旨在设计通信速率最优的设备到设备（D2D）编码缓存方案，同时最小化子数据包化水平，以便更接近实际应用。

**💡 创新点**

创新点在于提出了一种新的数据包类型（PT）设计框架，通过用户分组引入不对称性，并在缓存放置和多播传输中系统性地利用这种不对称性，从而创造出减少子数据包化的机会。

**🔧 技术方法**

使用了整数线性规划（ILP）技术来制定缓存方案设计，并通过向量最小公倍数操作来确定进一步分割比例。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了多种用户、文件和缓存大小的组合。

**📈 对比分析**

与Ji-Caire-Molisch（JCM）方案及其他先进方案进行了比较，结果显示在保持最优通信速率的同时，所提出的PT框架在子数据包化方面有显著改进，能够实现相对JCM方案的阶数或常数因子减少。

**⚠️ 局限性**

限制在于尽管PT框架适用于广泛的参数范围，但对于奇数K的完全通用构造仍然是一个开放问题。此外，当前的PT设计在构造中强制实现最优自由度（DoF），这限制了设计空间。

---

## 397. A 16 nm 1.60TOPS/W High Utilization DNN Accelerator with 3D Spatial Data Reuse and Efficient Shared Memory Access

**arXiv ID:** 2602.11357 | [PDF](https://arxiv.org/pdf/2602.11357v1)

**作者:** Xiaoling Yi `[一作]` (KU Leuven), Marian Verhelst `[通讯]` (KU Leuven)

**通讯引用:** 5737 | [OpenAlex ID](https://openalex.org/A5012150553)

**关键词:** `Hardware Architecture` `Computational Efficiency` `Convolutional Neural Network` `Recurrent Neural Network` `Transformer`

**🎯 论文内容**

构建了一款 16nm 的高利用率 DNN 加速器，采用 3D 空间数据复用、混合粒度数据预取与可编程动态内存分配等技术；

**💡 创新点**

在传统 2D 空间数组基础上引入 3D GEMM 核和共享内存架构，实现空间利用率提升 2.0×、时间利用率提升至 2.94×，并通过时间复用降低面积；

**🔧 技术方法**

使用 3D GEMM 核、输出静止数据流、可编程 6D/3D 地址生成单元、混合粒度 FIFO 预取、可编程动态内存分配、时间复用 SIMD 与通道等技术；

**📊 数据集**

对 MobileNetV2、ResNet50、ViT‑B、PointNeXt、LSTM、BERT‑Base、LLaMA3.2‑3B（预填充/解码）等多种 CNN、RNN 与 Transformer 工作负载进行评测；

**📈 对比分析**

与 2D 空间数组、无预取、无共享内存等基线对比，空间利用率提升至 2.0×、时间利用率提升至 2.12–2.94×，总时延降低 1.15–2.36×；在全密集 GEMM 工作负载下实现 1.60 TOPS/W 能耗效率和 1.25 TOPS/mm² 区域效率，优于多款前沿加速器；

**⚠️ 局限性**

对某些 LLM 解码阶段空间利用率仍低（≈69.7%），受限于工作负载尺寸与 3D 网格匹配；芯片面积已极小化但受制于 16nm 工艺；共享内存银行冲突仍需进一步缓解。

---

## 398. MING: An Automated CNN-to-Edge MLIR HLS framework

**arXiv ID:** 2602.11966 | [PDF](https://arxiv.org/pdf/2602.11966v1)

**作者:** Jiahong Bi `[一作]` (Technische Universitaet Dresden), Jeronimo Castrillon `[通讯]` (Technische Universitaet Dresden)

**关键词:** `Hardware Architecture` `Convolutional Neural Network` `Image`

**🎯 论文内容**

提出 MING 框架，自动将 CNN 模型通过 MLIR 转化为 FPGA HLS 代码，采用纯流式数据流架构生成边缘级硬件实现。

**💡 创新点**

创新点在于：① 采用全流式架构消除中间数组，显著降低 BRAM 用量；② 结合整数算术资源估计模型和 ILP DSE 自动选取并行度；③ 在 MLIR 上实现硬件感知的优化与自动 pragma 插入。

**🔧 技术方法**

技术上使用 MLIR、Vitis HLS、StreamHLS、整数算术资源估计、ILP DSE、循环展开/流水线、线性缓冲、自动 pragma 插入。

**📊 数据集**

实验基于标准 CNN 核心（Conv+ReLU、Cascade Conv、Residual Block、Linear、Feed Forward），输入尺寸为 32×32 与 224×224，使用 8‑bit 量化模型（如 AlexNet 512×128）。

**📈 对比分析**

与 Vanilla、ScaleHLS、StreamHLS 在 Kria KV260 FPGA 上比较，衡量时钟周期、BRAM、DSP、加速比和 DSP 效率；MING 在单层核实现 580× 加速、整体 50× 加速，BRAM 与 DSP 使用更优。

**⚠️ 局限性**

局限性包括：仍需 Vitis HLS 生态，未覆盖 Transformer/自注意力模型；FIFO 缓存预估保守；在云级高性能 FPGA 上可能不如其他框架；对极低 DSP 资源的适配仍有提升空间。

---

## 399. Decentralized Multi-Robot Obstacle Detection and Tracking in a Maritime Scenario

**arXiv ID:** 2602.12012 | [PDF](https://arxiv.org/pdf/2602.12012v1)

**作者:** Muhammad Farhan Ahmed `[一作]` (LS2N), Vincent Frémont `[通讯]` (LS2N)

**关键词:** `Robotics` `Object Detection` `Object Tracking` `Robotic Intelligence` `Convolutional Neural Network` `Image` `Video`

**🎯 论文内容**

提出了一套去中心化的多机器人感知框架，利用多架 UAV 与一艘自主水面艇协同检测、跟踪海上漂浮容器；框架将 YOLOv8+立体视差、EKF 跟踪、协方差交叉融合与信息驱动的目标分配相结合；

**💡 创新点**

①将 3D 立体视差与 YOLOv8 结合得到精确的三维检测；②采用协方差交叉（CI）实现跨机器人一致性融合；③利用基于信息增益的容量限制最小流（CMCF）分配 UAV 与目标，并在悬停环上选择信息最优姿态；

**🔧 技术方法**

YOLOv8+RGB+Disparity检测、立体匹配、EKF 跟踪、协方差交叉融合、Mahalanobis 门限关联、容量限制最小流分配、D‑optimality 视角优化、ROS2、Gazebo 仿真；

**📊 数据集**

在 Gazebo 真实海洋模拟环境中使用 5 个漂浮容器、3 架 UAV 以及 1 艘 Aquabot；训练检测模型时使用 SeaDronesSee 海洋视觉基准并做自适应增广；

**📈 对比分析**

与单一 YOLOv8 RGB 模型对比，mAP@0.5 由 0.985 提升至 0.995，mAP@0.5:0.95 由 0.952 提升至 0.970；定位误差在 2.2–4.7 m 范围内；跟踪协方差持续下降，信息增益满足阈值后自动停止；分配效率高，目标被一一覆盖且冗余最小；

**⚠️ 局限性**

仅在仿真中验证，未对真实海面、动态波浪及通信丢包等情况做实验；当前规模仅支持少量 UAV 与容器，扩展到大规模部署仍需进一步优化；

---

## 400. V-SHiNE: A Virtual Smart Home Framework for Explainability Evaluation

**arXiv ID:** 2602.11775 | [PDF](https://arxiv.org/pdf/2602.11775v1)

**作者:** Mersedeh Sadeghi `[一作]` (University of Cologne), Andreas Vogelsang `[通讯]` (University of Duisburg-Essen)

**通讯引用:** 1413 | [OpenAlex ID](https://openalex.org/A5022998651)

**关键词:** `Human-Computer Interaction` `Explainability and Interpretability` `Tabular`

**🎯 论文内容**

构建了一个基于浏览器的虚拟智能家居仿真框架 V‑SHiNE，用于可解释智能系统的可用性与有效性评估。

**💡 创新点**

创新点在于高度可配置、可插拔解释引擎、支持多种解释交付模式（push、pull、interactive）、实时交互日志以及轻量化、可复现的实验平台。

**🔧 技术方法**

技术包括 React + Phaser 前端渲染，Next.js + Socket.IO 后端，MongoDB 存储，JSON 配置文件，W3C Thing Description 与 TDeX 扩展，REST/WebSocket 接口，Vitest 单元/集成测试。

**📊 数据集**

使用 159 名参与者的实验数据以及从 CIRCE 研究迁移的场景进行验证；没有公开的大型数据集，主要采用实验生成的数据。

**📈 对比分析**

通过对比基于上下文的动态解释与静态解释的用户评价，评估说明的可理解性与可信度；实验显示框架能实时捕获主观与行为数据，支持高并发交互和日志收集，具体性能指标未给出但已验证可用性。

**⚠️ 局限性**

局限包括仅支持单用户交互、设备模型相对简单、缺乏长期部署验证、对真实物理环境的逼真度有限。

---

## 401. Enhancing SDG-Text Classification with Combinatorial Fusion Analysis and Generative AI

**arXiv ID:** 2602.11168 | [PDF](https://arxiv.org/pdf/2602.11168v1)

**作者:** Jingyan Xu `[一作]` (Fordham University), D. Frank Hsu `[通讯]` (Fordham University)

**通讯引用:** 10283 | [OpenAlex ID](https://openalex.org/A5082344124)

**关键词:** `Computation and Language` `Classification` `Data Synthesis` `Convolutional Neural Network` `Large Language Model` `Text`

**🎯 论文内容**

本文通过合并五种文本分类模型并结合生成式 AI 合成训练数据，构建了一套针对联合国可持续发展目标（SDG）文本的多模型融合框架；

**💡 创新点**

创新点包括：①引入组合式融合分析（CFA）与秩-得分特征（RSC）以及认知多样性（CD）来量化并融合模型；②利用 ChatGPT 生成高质量合成数据缓解标签稀缺问题；③将多模型融合结果与人工专家标注对比，验证融合策略优于单模型与 BERT。

**🔧 技术方法**

使用的技术包括：组合式融合分析（CFA）、秩-得分特征（RSC）、认知多样性（CD）、生成式 AI（ChatGPT）、LDA、语义网（LinkedSDG）、关键词映射（SDG Mapper）、卷积神经网络（CNN）以及随机森林。

**📊 数据集**

数据集主要包括：1）9,129 条 ChatGPT 生成的合成文本（每个 SDG 537 条）；2）306 条人工标注的 UN 文本样本（按 17 个 SDG 分组）。

**📈 对比分析**

通过平均 precision@1 进行比较，CFA 组合模型在 92.19% 的 SDG 组中优于所有单模型，最佳组合 precision@1 为 0.9673，超过单模型 0.9542 及 BERT 0.9446；与人工专家对比时，CFA 与专家一致的比例为 62.89%。

**⚠️ 局限性**

局限性包括：①融合过程受限于模型数量与可公开获取性；②合成数据分布可能偏离真实数据；③排名平局导致加权组合效果受限；④未实现多标签分类与更细粒度评估。

---

## 402. Real-Time Proactive Anomaly Detection via Forward and Backward Forecast Modeling

**arXiv ID:** 2602.11539 | [PDF](https://arxiv.org/pdf/2602.11539v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## 403. Leveraging Language Models to Discover Evidence-Based Actions for OSS Sustainability

**arXiv ID:** 2602.11746 | [PDF](https://arxiv.org/pdf/2602.11746v1)

**作者:** Nafiz Imtiaz Khan `[一作]` (University of California), Vladimir Filkov `[通讯]` (University of California)

**关键词:** `Software Engineering` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Chain-of-Thought` `Prompt Engineering` `Text`

**🎯 论文内容**

通过 LLM 结合 Retrieval‑Augmented Generation（RAG）和两层提示精炼与可靠性验证流程，自动从 829 篇 ICSE/FSE 的 OSS 研究论文中提取并验证 1,312 条可执行的研究建议（ReACTs）。

**💡 创新点**

引入 LLM 作为证据挖掘工具，构建两层提示精炼与可靠性验证流程，并将研究建议按实践类目组织，实现从文献到可操作建议的闭环。

**🔧 技术方法**

使用 Retrieval‑Augmented Generation（RAG）、Chain‑of‑Thought 与 Reason+Action 提示、后续推理提示，以及 Mixtral‑8x7B、Llama3‑8B、MistralLite‑7B、Mistral‑Nemo‑12B 等开源 LLM；利用向量检索与文本嵌入。

**📊 数据集**

829 篇 ICSE/FSE 发表的 OSS 研究论文，结合 APEX 工具的 Apache 项目可持续性指标用于案例验证。

**📈 对比分析**

在 10 篇人工标注样本上比较 4 个 LLM 与 3 种提示，采用 BLEU‑4/ROUGE‑L/BERTScore/METEOR 进行模型选择，最终选 Mixtral‑8x7B+CoT，得到 1,922 可信 ReACT，90% 以上满足 sound/precise，68% 满足全部质量标准。

**⚠️ 局限性**

仅覆盖 ICSE/FSE 会议论文，未包含期刊与实践导向文献；受限于本地可运行的 8‑12B 模型，无法使用更大模型与高级提示技巧；评估指标主要是语义相似度，难以完全衡量可操作性；缺乏真实项目实施与效果验证。

---

## 404. Fair Data-Exchange Mechanisms

**arXiv ID:** 2602.11417 | [PDF](https://arxiv.org/pdf/2602.11417v1)

**作者:** Rashida Hakim `[一作]` (Columbia University), Mihalis Yannakakis `[通讯]` (Columbia University)

**通讯引用:** 30550 | [OpenAlex ID](https://openalex.org/A5043084405)

**关键词:** `Computer Science and Game Theory` `Optimization`

**🎯 论文内容**

提出一种无金钱转移的公平数据交换机制，让每对代理只互相交换彼此收集到的数据量的最小值，从而在数据收集成本和收益不对称的情况下实现协作

**💡 创新点**

通过在总数据空间重新参数化，证明该公平交换游戏是超模的并具有正外溢；在此结构下构造了多项式时间算法求解最大（Pareto‑最好）Nash平衡，并证明该平衡在全模型下为全局Pareto最优且可通过机制实现真诚报告

**🔧 技术方法**

利用超模游戏理论、策略互补性、递归迭代法与逆映射分析，以及对图约束和离散化的扩展实现

**📊 数据集**

无具体实验数据集，本文为理论分析与算法设计性质研究

**📈 对比分析**

对比方法主要是理论对比：在完整图下证明最大平衡是Pareto最优，图约束与离散情形中证明其仍为可实现且Pareto不可支配；未给出数值实验或性能指标

**⚠️ 局限性**

局限性包括：假设数据点可连续且互不重叠、单一数据源、数据可完全共享、且参与者只能以收集成本和收益函数为特征；未考虑多类型或重叠数据、数据真实性验证等实际复杂情况

---

## 405. Beyond End-to-End Video Models: An LLM-Based Multi-Agent System for Educational Video Generation

**arXiv ID:** 2602.11790 | [PDF](https://arxiv.org/pdf/2602.11790v1)

**作者:** Lingyong Yan `[一作]` (Baidu Inc.), Jizhou Huang `[通讯]` (Baidu Inc.)

**关键词:** `Artificial Intelligence` `Generation` `Transformer` `Large Language Model` `Agentic AI` `Video` `Text`

**🎯 论文内容**

开发了一套基于大型语言模型的多代理系统，自动生成可执行脚本的教学视频，包含推理、可视化代码和叙述，并通过多层质量检查实现端到端自动化。

**💡 创新点**

①将教学视频视为可执行脚本并分解为推理、可视化、叙述三任务；②构建多代理架构与中心调度器；③设计三维异质评价（语义、工具、规则）实现循环改进；④通过脚本编译实现高质量可扩展性与低成本。

**🔧 技术方法**

使用大型语言模型（DeepSeek‑R1 等）驱动各代理，Python/Manim 生成可执行可视化代码，LLM 进行语义审查，工具执行校验与规则检查，模板驱动视频组装，TTS 生成语音。

**📊 数据集**

K‑12 领域双语教育数据集：Elementary Chinese Language Arts（文本阅读、句子分析）和 Middle School Mathematics（代数、几何、算数题），共计数千题。

**📈 对比分析**

与单模型直接提示基线（GPT‑4o、DeepSeek‑R1、Qwen3）相比，中文数据集可发布率从 52–82% 提升至 92%，数学数据集从 45–94% 提升至 96%；Perfect率提升至 58%；成本下降 95% 以上；吞吐量可达百万视频/日。

**⚠️ 局限性**

基础模型仍存在知识盲区导致关键错误（如文学理解、几何推理），系统对极端复杂推理或需多模态外部数据的任务仍有限制。

---

## 406. Both Topology and Text Matter: Revisiting LLM-guided Out-of-Distribution Detection on Text-attributed Graphs

**arXiv ID:** 2602.11641 | [PDF](https://arxiv.org/pdf/2602.11641v1)

**作者:** Yinlin Zhu `[一作]` (Sun Yat-sen University), Miao Hu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 12285 | [OpenAlex ID](https://openalex.org/A5000673492)

**关键词:** `Machine Learning` `Anomaly Detection` `Graph Neural Network` `Transformer` `Large Language Model` `Prompt Engineering` `Graph` `Text`

**🎯 论文内容**

本文提出了一种名为 LG‑Plug 的轻量级、可插拔策略，用于提升文本属性图（TAG）上的异常检测性能，方法通过拓扑与文本表征对齐、聚类一致性驱动的 LLM 伪 OOD 生成以及作为正则化项的 OOD 暴露，兼容现有拓扑驱动的 OOD 检测器。

**💡 创新点**

创新点在于：①首次将大语言模型（LLM）与拓扑驱动检测器实现无缝融合，既保留拓扑信息又利用丰富语义；②通过聚类+一致性过滤的 LLM 调度，显著降低伪 OOD 样本噪声并提高生成质量；③采用轻量级类别码本和启发式采样，极大降低 LLM 查询成本，提升效率。

**🔧 技术方法**

技术包括：图卷积网络（GCN）与 Transformer 文本编码器的对齐训练（节点级对比与边级相似度约束）；基于 K‑means 的节点聚类与内部过滤；基于 LLM 的交互式提示与类别码本更新；以及以 OOD 暴露为正则项的分段损失函数。

**📊 数据集**

使用六个公共 TAG 基准数据集：Cora、Citeseer、PubMed、WikiCS、Books‑History 与 ogbn‑arxiv。

**📈 对比分析**

实验将 LG‑Plug 组合到 GNNSafe 与 GRASP 等拓扑驱动检测器中进行对比。与传统方法（MSP、ODIN、Mahalanobis）相比显著提升；与现有 LLM‑基方法（LLMGuard、GLIP‑OOD、GOE‑LLM）相比，FPR95 减少约 5%，AUROC 提升约 2–3%；与拓扑驱动基线相比，FPR95 下降 ≥7%。

**⚠️ 局限性**

局限性：①仍需调用 LLM，尽管成本已降低但在极大图或实时系统中仍有压力；②方法在实验中仅针对静态 TAG，未验证在动态或开放世界图上的鲁棒性；③对聚类数、过滤阈值等超参数有一定敏感性，需要在不同场景下进行调优。

---

## 407. JEPA-VLA: Video Predictive Embedding is Needed for VLA Models

**arXiv ID:** 2602.11832 | [PDF](https://arxiv.org/pdf/2602.11832v1)

**作者:** Shangchen Miao `[一作]` (Tsinghua University), Mingsheng Long `[通讯]` (Tsinghua University)

**通讯引用:** 29101 | [OpenAlex ID](https://openalex.org/A5019241553)

**关键词:** `Computer Vision and Pattern Recognition` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Vision-Language-Action Model` `Video` `Multimodality`

**🎯 论文内容**

提出JEPA‑VLA框架，利用V-JEPA 2的视频预测嵌入自适应融合到现有Vision‑Language‑Action模型中，提升机器人操作的样本效率与泛化性能。

**💡 创新点**

创新点在于证明视频预测表征能同时提供环境理解与政策先验，并给出简单可插拔的早期/门控融合策略，使得静态图像或文本对齐的表征失效的VLA能显著受益。

**🔧 技术方法**

核心技术包括V-JEPA 2的联合嵌入预测预训练、Transformer基架的VLM与动作头、以及早期拼接和门控交叉注意力的融合机制；实现时保持V-JEPA 2冻结，调整学习率。

**📊 数据集**

主要使用LIBERO、LIBERO‑plus、RoboTwin2.0、真实机器人（Piper arm）和CortexBench等标准操控与评测数据集；在LIBERO‑plus对照实验仅用1/10训练数据。

**📈 对比分析**

与基线VLA、WorldVLA、OpenVLA‑OFT以及VC‑1等对比，JEPA‑VLA在LIBERO、LIBERO‑plus、RoboTwin2.0、CortexBench等任务上平均提升10–20%（成功率/奖励），尤其在长时程和分布迁移情形下表现最为显著。

**⚠️ 局限性**

局限性包括融合方式相对经验性、对V-JEPA 2的依赖导致计算开销略增、以及在极端动态环境或多模态输入复杂度上尚未进行充分验证。

---

## 408. Using predictive multiplicity to measure individual performance within the AI Act

**arXiv ID:** 2602.11944 | [PDF](https://arxiv.org/pdf/2602.11944v1)

**作者:** Karolin Frohnapfel `[一作]` (University of Tübingen), Kristof Meding `[通讯]` (CZS Institute for Artificial Intelligence and Law)

**关键词:** `Machine Learning` `Tabular`

**🎯 论文内容**

本文分析了欧盟AI法案中对高风险AI系统的准确性和透明度要求，并将“预测多重性”概念引入其中，提出利用冲突比例和δ-模糊度量评估个体层面的模型不确定性；同时提出一种基于数据与参数多样性的“临时方法”来近似Rashomon集合，并在合成、ACS就业、COMPAS等数据集上验证其有效性。

**💡 创新点**

①将预测多重性与AI法案的合规需求结合，首次将个体层面准确性纳入监管框架；②提出冲突比例和δ-模糊度两个直观度量；③提出低成本、可扩展的临时Rashomon构造方法；④通过对比实验证明小规模模型集合即可捕捉多重性。

**🔧 技术方法**

冲突比例/δ-模糊度指标；Rashomon集合近似（数据/参数多样化训练）；与TreeFarms的对照实验；统计分析（距离、ambiguity）。

**📊 数据集**

合成Gaussian数据集；美国社区调查（ACS）就业预测（≈1.7M样本）；binarized COMPAS刑事风险数据；以及决策树、XGBoost、MLP等模型。

**📈 对比分析**

采用对冲突比例、δ-模糊度与标准ambiguity等指标比较不同数据多样化策略；实验显示加入数据多样性显著提高冲突识别；临时方法在决策树上即使仅500模型即可逼近TreeFarms得到的ambiguity；在COMPAS上显示与全量Rashomon集合相比误差较小。

**⚠️ 局限性**

缺点在于：1）仍需手工设计数据/参数多样化策略，未给出通用自动化方法；2）仅在决策树等可解释模型上验证，复杂模型如深度网络尚未充分评估；3）法案中个体准确性标准仍模糊，实际监管落地仍需进一步研究。

---

## 409. Free Lunch for Stabilizing Rectified Flow Inversion

**arXiv ID:** 2602.11850 | [PDF](https://arxiv.org/pdf/2602.11850v1)

**作者:** Chenru Wang `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**通讯引用:** 26097 | [OpenAlex ID](https://openalex.org/A5100458183)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Optimization` `Flow-based Model` `Rectified Flow` `Ordinary Differential Equation` `Image` `Benchmark`

**🎯 论文内容**

提出了训练‑免费的速度场校正方法 Proximal‑Mean Inversion（PMI）和轻量化编辑校正 mimic‑CFG，改善 RF‑based 模型的反向推断与编辑过程。

**💡 创新点**

创新点在于利用运行平均速度作为全局引导，并通过近端优化和球形高斯约束在每步中稳定速度场，避免误差累积导致的低密度偏离；同时在编辑阶段采用投影加插值的方式平衡背景保持与编辑效果。

**🔧 技术方法**

技术手段包括：近端优化（proximal update）、运行平均速度、球形高斯约束、投影插值、欧拉/Heun/高阶 ODE 求解器，全部无额外训练或网络推理。

**📊 数据集**

主要使用 PIE‑Bench 数据集（700 张图像，10 个编辑类别）和 Flux.1‑dev 预训练模型进行实验。

**📈 对比分析**

将 PMI 和 mimic‑CFG 分别嵌入 Euler、Heun、RF‑Solver、FireFlow 及 RF‑Inversion 等四种采样器，评估重建（PSNR/SSIM/MSE/LPIPS）和编辑（背景保持、CLIP 相似度、结构距离）指标；实验显示显著提升重建质量（PSNR+1–1.5）、编辑背景保持（PSNR/SSIM 提升），并在保持甚至提高编辑准确度的同时，显著减少 NFEs。

**⚠️ 局限性**

局限性包括：仅在 RF‑based 模型上验证，缺乏对其他流/扩散框架的泛化；方法依赖超参数（λ、w）调优；在极高维或极度不均匀的数据分布下，球形约束可能不足以完全抑制误差；未深入探讨多模态或条件编辑的扩展。

---

## 410. AgentNoiseBench: Benchmarking Robustness of Tool-Using LLM Agents Under Noisy Condition

**arXiv ID:** 2602.11348 | [PDF](https://arxiv.org/pdf/2602.11348v1)

**作者:** Ruipeng Wang `[一作]` (Meituan), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60408 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `Artificial Intelligence` `Robotic Intelligence` `Adversarial Attack` `Optimization` `Transformer` `Large Language Model` `Agentic AI` `Text` `Benchmark`

**🎯 论文内容**

本文提出 AgentNoiseBench 框架，能够在大语言模型（LLM）代理的评测中自动注入可控的用户噪声和工具噪声，并通过轨迹感知的评估方法衡量代理在嘈杂环境下的鲁棒性；

**💡 创新点**

创新点包括：①系统化的噪声分类与可解性约束；②基于参考代理的对抗性噪声生成与注入管线；③轨迹一致性与稳定性门控的评估指标；④对多种模型规模与架构进行统一的鲁棒性对比；

**🔧 技术方法**

技术手段主要包括：对抗性噪声生成模型（参数冻结的生成器与可优化的系统提示）、约束演化噪声注入、轨迹一致性检测、熵与步骤分析以及稳定性门控准确率（SGA）评估；

**📊 数据集**

使用的数据集有 τ^2-Bench、VitaBench、HotpotQA、2WikiMultiHopQA，并在这些基准上注入细粒度噪声进行评测；

**📈 对比分析**

对 24 款模型（OpenAI GPT、Claude、Gemini、DeepSeek、Qwen 等）在无噪声与噪声环境下分别计算 Avg@4、Avg_Tokens@4、Avg_Steps@4 等指标，结果显示平均准确率下降约 20.8%，工具噪声导致更大性能损失，且具备思考功能的模型对噪声更脆弱；

**⚠️ 局限性**

局限性包括：仅针对语言+工具的代理，未涵盖规划或异步工具等；轨迹熵诊断缺乏因果解释；评估过程开销大，缺少可扩展的成本-精度折衷方案。

---

## 411. DD-MDN: Human Trajectory Forecasting with Diffusion-Based Dual Mixture Density Networks and Uncertainty Self-Calibration

**arXiv ID:** 2602.11214 | [PDF](https://arxiv.org/pdf/2602.11214v1)

**作者:** Manuel Hetzel `[一作]` (University of Applied Sciences Aschaffenburg), Bernhard Sick `[通讯]` (University of Kassel)

**通讯引用:** 4785 | [OpenAlex ID](https://openalex.org/A5065340030)

**关键词:** `Computer Vision and Pattern Recognition` `Autonomous Driving` `Generation` `Diffusion model` `Time Series` `Sequential`

**🎯 论文内容**

提出了一种基于扩散网络的双混合密度网络DD-MDN，用于预测人类轨迹，并实现高精度、可校准的不确定性估计。

**💡 创新点**

将少量观测帧下的无监督扩散过程与双重混合密度网络相结合，能够自校准不确定性、生成概率排序的多样化轨迹候选，而无需预设锚点或终点。

**🔧 技术方法**

使用了少量Shot denoising diffusion backbone、双MDN（step与anchor两种GM表示）、自注意力与社交注意力、坐标卷积编码以及基于NLL的无监督训练。

**📊 数据集**

在ETH/UCY、SDD、inD以及IMPTC等公开数据集上进行实验。

**📈 对比分析**

与LED、SingTraj、MoFlow等SOTA方法对比，在所有基准上实现了最低的min_kADE/min_kFDE，并在短观测窗口（仅两帧）下表现出约20%至40%的性能提升，同时在可靠性指标上获得最高的校准分数。

**⚠️ 局限性**

在拥挤场景下预测仍易出现欠自信且置信区间过宽，未来工作计划引入更丰富的环境上下文与交互建模。

---

## 412. Modelling Trust and Trusted Systems: A Category Theoretic Approach

**arXiv ID:** 2602.11376 | [PDF](https://arxiv.org/pdf/2602.11376v1)

**作者:** Ian Oliver `[一作]` (University of Oulu), Pekka Kuure `[通讯]` (National Defence University)

**关键词:** `Cryptography and Security`

**🎯 论文内容**

构建了一个基于范畴理论的远程证明与信任决策框架，定义了元素、声明、结果与决策对象，以及 attest、verify、decide 等映射，并通过 Heyting 代数对信任等级进行形式化。

**💡 创新点**

创新点在于将信任决策空间建模为 Heyting 代数，引入指数对象来描述证明组合的可表达度量，并通过子对象分类提供了严谨的逻辑语义；该框架同时兼顾了理论可验证性与实际远程证明系统的实现。

**🔧 技术方法**

采用范畴理论（子对象分类、指数、态射）、Heyting 代数、同构与指数化等数学工具；实现层面使用 Haskell、数据库、TPM/UEFI 事件日志等。

**📊 数据集**

使用的“数据集”主要是典型的 TPM 量子、UEFI 事件日志、硬件测量等标准证明信息，未涉及大规模公开数据集。

**📈 对比分析**

论文未给出量化性能对比；主要通过示例（boot‑run‑shutdown、Evil Maid）与实现系统 Jane 与 Keylime 的兼容性验证，未进行系统级性能评估。

**⚠️ 局限性**

局限包括：缺乏对动态策略和大规模系统的完整验证；组合系统的内部信任逻辑尚未完全处理；未提供恢复机制或量化性能指标。

---

## 413. Calibrated Bayesian Deep Learning for Explainable Decision Support Systems Based on Medical Imaging

**arXiv ID:** 2602.11973 | [PDF](https://arxiv.org/pdf/2602.11973v1)

**作者:** Hua Xu `[一作]` (Universidad Politécnica de Madrid), Juan I. Godino-Llorente `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 4286 | [OpenAlex ID](https://openalex.org/A5068136205)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Explainability and Interpretability` `Image` `Biomedical Data`

**🎯 论文内容**

提出了一个基于变分推理贝叶斯神经网络的框架，通过引入CUB-Loss训练时校准和Dual Temperature Scaling（DTS）后置校准，实现医学图像分类的不确定性量化与校准。

**💡 创新点**

创新点在于构造了Confidence‑Uncertainty Boundary Curve（CUBC）并将其映射为CUB-Loss，强制模型在预测正确时降低不确定性、预测错误时提高不确定性；同时设计了双温度标定方法，使得后置校准可同时收紧和放宽不同置信度区间的分布，形成端到端可训练且后置可调的统一校准体系。

**🔧 技术方法**

使用的技术包括变分推理贝叶斯网络、预测熵作为不确定性度量、CUBC误差度量、对数障碍函数损失、两阶段温度标定、Monte Carlo采样、温度参数L‑BFGS优化以及常规交叉熵与KL正则化。

**📊 数据集**

实验使用了三个医学图像数据集：胸部X光肺炎与COVID‑19筛查（MIMIC‑CXR、CheXpert、COVIDx、NIH），视网膜糖尿病视网膜病变（APTOS 2019），以及皮肤病变识别（HAM10000）。

**📈 对比分析**

与基线BCNN、AvUC、MCDropout等方法对比，CUB‑Loss + DTS在准确率、AvU、ΔU（正确/错误样本不确定性差）以及近OOV场景下的AUROC/AUPR 上均显著提升；在数据稀缺和严重类别不平衡条件下仍保持较高准确率和更优的不确定性校准。

**⚠️ 局限性**

局限性包括：对MC采样和贝叶斯层的计算成本较高，需要手动设定置信度阈值γ及温度区间；在极端OOV或多模态联合任务中校准效果仍需进一步验证。

---

## 414. Behavioral Indicators of Overreliance During Interaction with Conversational Language Models

**arXiv ID:** 2602.11567 | [PDF](https://arxiv.org/pdf/2602.11567v1)

**作者:** Chang Liu `[一作]` (Tsinghua University), Xiang 'Anthony' Chen `[通讯]` (UCLA)

**关键词:** `Human-Computer Interaction` `Transformer` `Auto Encoder` `Large Language Model` `Text`

**🎯 论文内容**

本文通过实验收集了 77 名参与者在三种任务中与 LLM（ChatGPT/类似模型）交互的行为日志，并使用自动编码器+聚类方法对用户行为序列进行语义编码与聚类，识别出与过度依赖（overreliance）相关的五种行为模式；同时通过对任务结果与注入的误信息对比，量化了用户的过度依赖水平；最后基于这些行为模式提出了可即时触发的界面改进建议；

**💡 创新点**

①首次系统地将过程层面的交互行为与 LLM 过度依赖联系起来，提供了可解释的行为模式；②采用变压器自编码器+DBSCAN 的端到端行为嵌入与聚类框架；③提出了针对复制粘贴、任务理解、频繁查询、粗糙定位编辑和停顿犹豫等行为的设计改进方案；

**🔧 技术方法**

基于 Transformer 的自编码器对事件序列进行低维嵌入；DBSCAN 聚类；特征向量由 37 维离散/连续特征构成；行为日志使用 Chrome 扩展采集；后期使用 t 检验与预测一致性筛选高质量聚类；

**📊 数据集**

实验数据集公开于 GitHub（https://github.com/CJunette/behavior_indicator_of_overreliance），包含 77 名参与者在三项任务（quiz、article summarization、trip planning）中与 LLM 的交互日志以及对应的过度依赖评分；

**📈 对比分析**

本文未使用传统基准模型进行对比，而是通过自建的过度依赖评分与聚类结果的相关性进行评估；在三项任务中，过度依赖分布可视化显示差异，聚类后可预测测试集成员的过度依赖水平，预测准确率约 70%（具体数值见论文附录），表明行为模式与过度依赖高度相关；

**⚠️ 局限性**

1）任务与误信息注入方式受限，未能完全模拟自然 LLM 失真；2）样本量有限，难以构建高精度预测模型；3）缺乏实时、跨任务的验证，无法保证在高风险场景下的适用性；4）行为模式与个体差异、任务难度等混杂因素难以彻底分离。

---

## 415. UltraLIF: Fully Differentiable Spiking Neural Networks via Ultradiscretization and Max-Plus Algebra

**arXiv ID:** 2602.11206 | [PDF](https://arxiv.org/pdf/2602.11206v1)

**作者:** Jose Marie Antonio Miñoza `[一作]` (Center for AI Research), Jose Marie Antonio Miñoza `[通讯]` (Center for AI Research)

**关键词:** `Machine Learning` `Spiking Neural Network` `Audio` `Image`

**🎯 论文内容**

提出UltraLIF和UltraDLIF两种可微分的脉冲神经网络模型，通过超离散化实现神经元阈值激活的连续近似，消除传统尖峰生成的梯度不匹配问题；

**💡 创新点**

创新点在于将热带几何中的超离散化与log-sum-exp软最大化结合，得到从连续动力学到离散脉冲动力学的严格连续映射，提供前向后向一致、梯度有界且收敛到经典LIF的理论保证；

**🔧 技术方法**

使用了超离散化、log-sum-exp软最大化、可学习温度参数、稀疏正则化等技术；

**📊 数据集**

在六个基准数据集上进行实验，涵盖静态图像（MNIST、Fashion-MNIST、CIFAR-10）、神经形态视觉（N-MNIST、DVS-Gesture）和音频（SHD）；

**📈 对比分析**

与传统的LIF、PLIF、AdaLIF、FullPLIF以及基于 surrogate gradient 的 DSpike/DSpike+ 进行对比，UltraLIF/UltraDLIF 在单时间步（T=1）时在神经形态和音频数据上显著优于基线（如 SHD +11.22%、DVS-Gesture +7.96%、N-MNIST +3.91%），在多时间步下性能趋于相近甚至略逊；稀疏正则化可将能耗降低 30%–50% 并保持准确率；

**⚠️ 局限性**

局限性包括：（1）Soft spike 在训练期间仍非硬二值，需在推理时手工硬化；（2）空间型模型 UltraDLIF 对可学习泄漏参数不敏感，可能不适用于需要精确时序编码的任务；（3）超离散化仅适用于无减法的运算，需通过特定系数或可逆 max-plus 代数处理；（4）实验仅覆盖六个公共数据集，实际硬件部署的性能尚待验证。

---

## 416. Data-Driven Trajectory Imputation for Vessel Mobility Analysis

**arXiv ID:** 2602.11890 | [PDF](https://arxiv.org/pdf/2602.11890v1)

**作者:** Giannis Spiliopoulos `[一作]` (University of the Aegean), Nikos Bikakis `[通讯]` (Hellenic Mediterranean University)

**通讯引用:** 750 | [OpenAlex ID](https://openalex.org/A5005916659)

**关键词:** `Databases` `Graph` `Time Series`

**🎯 论文内容**

提出了一种轻量级的基于H3网格的船舶轨迹缺口插补框架HABIT；

**💡 创新点**

创新点在于利用海上历史AIS数据的空间聚合统计构建动态图，结合数据驱动的中位数坐标投影和RDP平滑，既保持航行可行性又提高了插补精度；

**🔧 技术方法**

采用DuckDB进行高效数据预处理与统计，NetworkX构建加权图，A*搜索求最短路径，Ramer-Douglas-Peucker算法做轨迹简化；

**📊 数据集**

使用丹麦海事局AIS（航客船三月数据）、德国基尔至哥德堡AIS（单一路线）以及希腊阿吉亚网络AIS（全月多种船型）三大数据集；

**📈 对比分析**

与GTI（基于邻域图的插补）和直线插值（SLI）进行对比，HABIT在准确度（DTW）与延迟、内存占用方面均表现优越，尤其在高分辨率与大规模数据时保持子秒级查询延迟；

**⚠️ 局限性**

局限性包括对极少量或高度稀疏AIS数据的鲁棒性不足，无法处理人为停用AIS导致的长缺口，以及对天气、船舶状态等上下文信息考虑有限。

---

## 417. CryptoAnalystBench: Failures in Multi-Tool Long-Form LLM Analysis

**arXiv ID:** 2602.11304 | [PDF](https://arxiv.org/pdf/2602.11304v1)

**作者:** Anushri Eswaran `[一作]` (University of California), Himanshu Tyagi `[通讯]` (Sentient Labs)

**关键词:** `Cryptography and Security` `Classification` `Recommendation System` `Anomaly Detection` `Optimization` `Explainability and Interpretability` `Transformer` `Large Language Model` `Agentic AI` `Text` `Benchmark` `Finance Related`

**🎯 论文内容**

研究了大语言模型在加密货币分析中的多工具推理失败模式，并提出 CryptoAnalystBench 基准、评估管线和七类错误分类。

**💡 创新点**

创新点包括：①专门针对高数据密度、时效性强的加密领域提出生产对齐基准；②设计多级评估框架（引用验证+LLM评判）与七类错误分类；③构建可扩展的多工具代理评测架构。

**🔧 技术方法**

采用 LLM‑as‑a‑judge、ReAct 代理架构、工具调用（API、Web检索、链上查询）、自动化引用验证管线及结构化 JSON 评分。

**📊 数据集**

使用 198 条去标识化加密/DeFi 查询（覆盖 11 类）以及实时市场、链上数据 API（如 CoinGecko、DefiLlama）和检索文档做评测数据。

**📈 对比分析**

通过四维打分（相关性、时间相关性、深度、一致性）与人工专家评估对比，发现 GPT‑5.2 在所有维度均最高，Kimi‑K2.5 在深度表现突出；整体误差率低于 6%，但不同模型在错误类别分布上存在显著差异。

**⚠️ 局限性**

限制包括 LLM‑as‑a‑judge 与人工评估一致性有限；高阶错误仍难以完全自动化识别；基准聚焦加密领域，通用性待验证；弱模型在复杂推理时容易退化。

---

## 418. Affordance-Graphed Task Worlds: Self-Evolving Task Generation for Scalable Embodied Learning

**arXiv ID:** 2602.12065 | [PDF](https://arxiv.org/pdf/2602.12065v1)

**作者:** Xiang Liu `[一作]`, Changshui Zhang `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Graph Neural Network` `Transformer` `Vision Language Model` `Reinforcement Learning` `Image` `Graph`

**🎯 论文内容**

本文提出了Affordance-Graphed Task Worlds（AGT-World）框架，能够从单张真实RGB图像自动构建可交互的仿真场景，并利用图结构将复杂任务分解为可执行的原子动作，进一步通过自我进化机制在执行过程中利用Vision‑Language模型和几何验证自动纠错与策略迭代；

**💡 创新点**

创新点包括：① 用语义-物理统一的图模型精细化任务分解；② 通过自我进化闭环实现在线反馈纠错；③ 结合VLM与几何验证的混合反馈提升长序列任务的成功率；

**🔧 技术方法**

核心技术包括：基于OmniGibson的物理仿真；VLM驱动的任务拆解与生成；图论路径规划与行动转移模型；自我进化算法（基于VLM反馈的序列修改与参数调整）；

**📊 数据集**

数据集方面：从34个真实场景（单RGB图）自动生成102个场景-任务对，涵盖操纵、导航、精细放置等三大任务类型；

**📈 对比分析**

与RoboGen、Behavior-100、Meta-World等现有数据集及方法对比，AGT-World在任务多样性（102任务）和视觉相似度（ViT Sim 0.440）均表现领先；在单任务成功率上平均71.6%，在四个长序列任务中通过自我进化实现40-70%的完成率，显著优于传统开环策略；

**⚠️ 局限性**

局限性主要有：① 依赖VLM的生成与反馈，受模型偏差影响；② 任务图仅覆盖有限原子动作，无法覆盖所有复杂交互；③ 自我进化需要多次迭代，计算开销和实时性受限；

---

## 419. LaCy: What Small Language Models Can and Should Learn is Not Just a Question of Loss

**arXiv ID:** 2602.12005 | [PDF](https://arxiv.org/pdf/2602.12005v1)

**作者:** Szilvia Ujváry `[一作]` (University of Cambridge), Michael Kirchhof `[通讯]` (Apple)

**关键词:** `Computation and Language` `Generation` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

本文提出 LaCy 框架，利用 spaCy 语法解析和损失信号来决定小语言模型在预训练时学习哪些词元，并通过特殊占位符触发更大模型的调用。

**💡 创新点**

创新点在于将损失与语法可接受性相结合，区分可学习与不可学习的事实词元，从而降低参数中存储的事实信息并提升事实性。

**🔧 技术方法**

技术包括基于 spaCy 的命名实体识别与语法解析、损失阈值筛选、占位符训练以及模型级联推理。

**📊 数据集**

使用的数据集为英文维基百科 dwiki（约 50 B 词元用于预训练，10 B 词元用于验证），并在 1.3 B 大模型与 Llama 3.2 1B 的级联中进行评估。

**📈 对比分析**

通过与 Rho‑loss、纯损失和 token‑logit 训练等方法对比，利用 FactScore 和事实泄漏指标，LaCy 训练出的 334 M 参数模型在 Wiki 文章生成任务中取得最高 FactScore 且事实泄漏最低。

**⚠️ 局限性**

局限性包括仅关注何时调用而未实现具体查询或检索逻辑，实验范围局限于维基百科生成，且对其他语言或更大模型的适用性尚未验证。

---

## 420. A Unified Treatment of Substitution for Presheaves, Nominal Sets, Renaming Sets, and so on

**arXiv ID:** 2602.11907 | [PDF](https://arxiv.org/pdf/2602.11907v1)

**作者:** Fabian Lenke `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Henning Urbat `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 224 | [OpenAlex ID](https://openalex.org/A5049596655)

**关键词:** `Logic in Computer Science`

**🎯 论文内容**

本文提出了一种通用方法，从给定的单模范畴作用出发构造闭合的代数模范畴，从而统一恢复并推导了保守类的代换张量，并首次在名义集与重命名集上给出了新的闭合代换结构；

**💡 创新点**

创新点在于将单模作用与J-扩展相结合，既概括了保守类的代换张量，又为名义集与重命名集提供了此前缺失的代换张量，并揭示了名义集与保守类之间的多层对应关系；

**🔧 技术方法**

主要技术包括范畴理论工具（终点、余端、Kan扩展、Day卷积）、单模范畴作用、代换张量的构造与闭合性证明，以及名义集与保守类的对偶性与等价性；

**📊 数据集**

无数据集，完全是理论推导与形式化证明；

**📈 对比分析**

与现有的保守类代换张量理论相比，本文提供了统一的抽象框架并补充了缺失的名义集代换张量，理论上实现了更广泛的对应与兼容，但未进行实验性能评估；

**⚠️ 局限性**

局限在于仅处理无类型上下文，未扩展到依赖或参数化类型环境；某些代换张量（如“捕获式”）缺乏单位，且在更复杂的语法或高级语言语义中的直接应用尚未完全展开。

---

## 421. ThinkRouter: Efficient Reasoning via Routing Thinking between Latent and Discrete Spaces

**arXiv ID:** 2602.11683 | [PDF](https://arxiv.org/pdf/2602.11683v1)

**作者:** Xin Xu `[一作]` (University of California San Diego), Saayan Mitra `[通讯]` (Adobe Research)

**关键词:** `Artificial Intelligence` `Computational Efficiency` `Generation` `Transformer` `Large Language Model` `Chain-of-Thought` `Text`

**🎯 论文内容**

提出了一种推理时基于模型置信度的路由机制，在推理过程中动态在离散词表空间和潜在空间之间切换，从而实现更高效、更准确的推理；

**💡 创新点**

创新点在于：①使用置信度阈值决定思考空间；②在潜在空间聚合多条可能路径、在低置信度时切换到离散空间避免噪声；③是一种训练-free、仅在推理时使用的机制；

**🔧 技术方法**

采用Soft Thinking的概率加权软嵌入、离散采样、温度缩放、随机与贪婪解码、阈值路由、Cold Stop检测等技术；

**📊 数据集**

主要使用STEM推理数据集AIME 2024/2025、GPQA Diamond；编码数据集HumanEval、MBPP；

**📈 对比分析**

与CoT（采样/贪婪）、Soft Thinking、Random Routing等基线比较；在多种LRM（Qwen3 1.7/8/32B、gpt-oss-20B）上，Pass@1平均提升19.7点，生成长度相较Soft Thinking减少约4-6%，整体性能显著优于基线；

**⚠️ 局限性**

局限性：需在验证集上调优阈值，适用范围受限于所测试的模型与任务；对置信度作为代理的依赖可能在不同模型/任务上不完全稳健；

---

## 422. Bi-Level Prompt Optimization for Multimodal LLM-as-a-Judge

**arXiv ID:** 2602.11340 | [PDF](https://arxiv.org/pdf/2602.11340v1)

**作者:** Bo Pan `[一作]` (Meta AI), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 6725 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `Artificial Intelligence` `Optimization` `Transformer` `Large Language Model` `Prompt Engineering` `Multimodality`

**🎯 论文内容**

研究了多模态 LLM 评判者的自动提示优化问题，提出 BLPO 框架通过共优化判别提示与图像转文本提示提升评判与人类评分的一致性。

**💡 创新点**

引入可学习的图像转文本提示以缩短上下文长度，并采用双层优化实现判别提示与 I2T 提示的协同学习，克服多模态上下文窗口限制。

**🔧 技术方法**

使用双层优化（bi‑level optimization）+ LLM‑as‑optimizer（GPT‑o3）+ 试错式提示更新 + 任务相关图像描述生成等技术。

**📊 数据集**

在 AGIN、SeeTRUE、ImageReward、UnsafeBench 四个图像评估基准上进行实验。

**📈 对比分析**

与 OPRO、APO、TextGrad 等现有自动提示优化方法对比，在三种多模态 LLM 评判器上取得平均比第二佳方法高约 8% 的宏 F1 或准确率，收敛更快更稳定。

**⚠️ 局限性**

仍受限于 LLM 上下文窗口大小，优化过程需要多轮 LLM 调用且计算成本较高；I2T 提示生成的文本可能丢失局部视觉细节；对极端长图像集合的效果尚未验证。

---

## 423. Dissecting Subjectivity and the "Ground Truth" Illusion in Data Annotation

**arXiv ID:** 2602.11318 | [PDF](https://arxiv.org/pdf/2602.11318v1)

**作者:** Sheza Munir `[一作]` (University of Toronto), Syed Ishtiaque Ahmed `[通讯]` (University of Toronto)

**通讯引用:** 4022 | [OpenAlex ID](https://openalex.org/A5089574660)

**关键词:** `Artificial Intelligence` `Text` `Review/Survey Paper`

**🎯 论文内容**

对 2020‑2025 年 7 个顶级会议（ACL, AIES, CHI, CSCW, EAAMO, FAccT, NeurIPS）发表的 346 篇文献进行了系统综述，揭示了数据标注实践中的“共识陷阱”与知识主体的排除机制。

**💡 创新点**

首次将“地理霸权”“劳动制度化”“人类‑验证者模型”等概念纳入标注理论，提出了多元化标注基础设施路线图，并将异议视为高保真度信号。

**🔧 技术方法**

采用 PRISMA 2020 指南、分层关键词筛选、人工标题/摘要筛选、全文评估以及反思性主题分析等系统综述方法。

**📊 数据集**

未使用传统实验数据集，而是利用会议论文元数据（共 30,897 篇记录）作为研究样本。

**📈 对比分析**

通过对不同会议的标注流程和聚合方法进行对比，说明传统多数投票导致偏见，提出基于人类身份和理由的多元聚合，未给出量化性能指标。

**⚠️ 局限性**

研究范围仅覆盖七大会议，排除非同行评审和行业案例，导致对非英语、非西方实践的可见度不足，且缺乏实证验证。

---

## 424. Who is the richest club in the championship? Detecting and Rewriting Underspecified Questions Improve QA Performance

**arXiv ID:** 2602.11938 | [PDF](https://arxiv.org/pdf/2602.11938v1)

**作者:** Yunchong Huang `[一作]` (University of Amsterdam), Sandro Pezzelle `[通讯]` (University of Amsterdam)

**通讯引用:** 847 | [OpenAlex ID](https://openalex.org/A5007142536)

**关键词:** `Computation and Language` `Large Language Model` `Prompt Engineering` `Text` `Benchmark`

**🎯 论文内容**

针对现有问答基准中的不完全指定问题，构建了LLM驱动的检测器与改写器，系统评估这些问题对LLM性能的影响。

**💡 创新点**

①首次提供基于LLM的全自动不完全指定问题分类与改写方法；②通过实验证明多数问答错误来源于问题歧义而非模型缺陷；③为后续基准设计与评估提供可复现的工具。

**🔧 技术方法**

使用大型语言模型（如 GPT‑4o、Gemini‑2.5‑Flash）实现问题分类与改写；利用 token‑level F1 与 NVIDIA AA（RAGAS 框架）评估答案质量，并采用 t‑检验比较不同子集性能。

**📊 数据集**

自构建的 UNDER/UNDER‑gold 数据集用于训练与验证分类器；四个主流 QA 数据集（Natural Questions、HotpotQA、TriviaQA、FRAMES）组成 QA‑ensemble 进行实验。

**📈 对比分析**

对 FS 与 UND 子集分别评测两大 LLM（GPT‑4o、Gemini‑2.5‑Flash），UND 子集准确率显著低于 FS；改写后 FS 比例提升至 64–86%，并且准确率提升 20–22%（仅 TriviaQA 例外）。

**⚠️ 局限性**

分类器性能尚可但非最优，改写器可能产生过度简化的问题；实验缺乏充分的人类人工验证；未充分利用最新 LLM 工程技术（如自动 prompt 优化、代理框架）。

---

## 425. U-Net with Hadamard Transform and DCT Latent Spaces for Next-day Wildfire Spread Prediction

**arXiv ID:** 2602.11672 | [PDF](https://arxiv.org/pdf/2602.11672v1)

**作者:** Yingyi Luo `[一作]` (University of Illinois Chicago), Ahmet Enis Cetin `[通讯]` (University of Illinois Chicago)

**通讯引用:** 4693 | [OpenAlex ID](https://openalex.org/A5080469744)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Generation` `Convolutional Neural Network` `Multimodality`

**🎯 论文内容**

开发了一种轻量级的 Transform Domain Fusion UNet（TD‑FusionUNet），用于基于多模态卫星数据的次日野火蔓延预测。

**💡 创新点**

创新点包括：① 在网络中引入可学习的 Hadamard 与 DCT 变换层，实现频域特征的压缩与阈值化；② 采用随机 margin cropping 与 Gaussian mixture 预处理技术，增强稀疏火灾掩模的表达能力；③ 通过双分支（Hadamard 与 DCT）结构进行特征融合，兼顾全局与频域信息。

**🔧 技术方法**

使用的技术有：Hadamard 变换、离散余弦变换（DCT）、软阈值化、双分支 UNet 结构、轻量级卷积、混合损失（BCE+Dice+Focal）等。

**📊 数据集**

数据集：Google Research Next‑Day Wildfire Spread dataset（2023）和 WildfireSpreadTS dataset。

**📈 对比分析**

与 ResNet18‑based UNet、单分支 HT‑UNet、2D CNN autoencoder 等基线相比，TD‑FusionUNet 在 WildfireSpreadTS 上实现 F1 = 0.591（参数仅 370k），显著提升准确率且保持低计算复杂度。

**⚠️ 局限性**

局限性：模型对高分辨率多通道输入有依赖，且在极度稀疏或缺失火灾掩模的极端场景下表现尚待进一步验证。

---

## 426. Robust Composite DNA Storage under Sampling Randomness, Substitution, and Insertion-Deletion Errors

**arXiv ID:** 2602.11951 | [PDF](https://arxiv.org/pdf/2602.11951v1)

**作者:** Busra Tegin `[一作]` (CentraleSupelec), Tolga M Duman `[通讯]` (Bilkent University)

**通讯引用:** 6088 | [OpenAlex ID](https://openalex.org/A5048023330)

**关键词:** `Information Theory` `Low-Density Parity-Check (LDPC) Code Decoding`

**🎯 论文内容**

本文提出了将复合 DNA 字母视为三维概率简单形上的星座点，构建多项式通道模型，并通过计算对应的 LLR 以实现基于 LDPC 码的误差纠正，进一步扩展到置换和插删错误。

**💡 创新点**

创新点在于：1）将复合 DNA 与数字调制星座映射相类比，利用多项式通道推导转移概率与 LLR；2）针对采样随机性、置换与插删错误给出星座点更新规则；3）证明普通 LDPC 码即可在此模型下实现可靠解码。

**🔧 技术方法**

使用的技术包括：多项式通道建模、星座映射与 LLR 计算、低密度奇偶校验（LDPC）码的解码（log-SPA），以及星座点更新算法来处理置换与插删误差。

**📊 数据集**

本文未使用公开数据集，而是基于仿真产生的 DNA 读取样本（不同 n、错误概率 ϵ、p_i/p_d），并在多种 LDPC 码率与星座维度（L=3,4）下评估性能。

**📈 对比分析**

通过与仅考虑采样随机性和先前限幅概率误差模型的对比，本文展示了在不同错误模型下，LDPC 码的 BLER 随读取样本数 n 增加而显著下降；在置换错误低概率时性能与采样随机性相近，高概率下仍保持可接受的误码率；插删错误导致更大性能下降，但可通过增加样本数补偿。

**⚠️ 局限性**

局限性包括：星座点更新规则为近似而非最优；对插删错误的处理仅保留长度相同的链路，忽略不等插删组合；未对串失、碱基吸附等实际 DNA 存储误差进行建模；且仅验证了 LDPC，缺乏针对复合 DNA 的专门码设计。

---

## 427. EmoSpace: Fine-Grained Emotion Prototype Learning for Immersive Affective Content Generation

**arXiv ID:** 2602.11658 | [PDF](https://arxiv.org/pdf/2602.11658v1)

**作者:** Bingyuan Wang `[一作]` (Hong Kong University of Science and Technology), Zeyu Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1278 | [OpenAlex ID](https://openalex.org/A5100599637)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Diffusion model` `Large Language Model` `Image` `Text`

**🎯 论文内容**

提出了 EmoSpace 框架，利用可学习情感原型实现细粒度情绪控制，支持情绪 outpainting、风格化生成与全景生成，适用于 VR 内容创作。

**💡 创新点**

创新点在于动态可学习的情感原型库与视觉‑语言对齐、multi‑prototype 指导、时序混合与注意力重加权相结合，突破传统离散情绪标签与维度模型的限制，首次实现细粒度情绪表达与 VR 场景的一致性。

**🔧 技术方法**

核心技术包括扩散模型（Stable Diffusion XL）与 LoRA、ControlNet、文本反演；prototype learning 与 CLIP 对齐；多原型指导、时序混合、注意力重加权；GPT‑4o‑mini 进行迭代提示优化。

**📊 数据集**

使用 EmoSet‑118K 作为情感标签训练集，BLIP‑2 生成文本描述；PLUTCHIK 情绪轮进行细粒度评估；自建的 32 张情绪全景用于 VR 体验测试。

**📈 对比分析**

与 EmoGen、EmotiCrafter 及 SDXL 基线在 CLIP‑Score、情绪分类与细粒度准确率、LAION‑Aesthetic 评价指标上进行对比；实验表明 EmoSpace 在情绪准确性（细粒度）和美学质量上均优于所有基线，且在 VR 场景下能显著提升主观情绪体验。

**⚠️ 局限性**

主要局限包括：缺乏针对 VR 的专用情绪标签与评估标准；全景内容仍停留在静态形式，未充分利用交互性；硬件舒适度与长时使用体验待改进；多感官（声、触觉）情绪增强尚未实现。

---

## 428. How Sampling Shapes LLM Alignment: From One-Shot Optima to Iterative Dynamics

**arXiv ID:** 2602.12180 | [PDF](https://arxiv.org/pdf/2602.12180v1)

**作者:** Yurong Chen `[一作]` (Inria), Fan Yao `[通讯]` (University of North Carolina)

**通讯引用:** 1645 | [OpenAlex ID](https://openalex.org/A5026468690)

**关键词:** `Machine Learning` `Optimization` `Recommendation System` `Transformer` `Large Language Model` `Reinforcement Learning from Human Feedback` `Text`

**🎯 论文内容**

研究了在大语言模型对齐中采样策略与参考策略对离线和迭代优化的影响，并给出了理论与实验分析。

**💡 创新点**

揭示了采样/参考自适应导致的排名失效、周期振荡与熵崩塌等稳定性问题，并提出了稳定性约束与参数调节策略。

**🔧 技术方法**

使用Identity Preference Optimization（IPO）和Direct Preference Optimization（DPO）的闭式解析、KL 正则化、混合采样/参考动态系统、线性代数与凸分析。

**📊 数据集**

基于 NVIDIA HelpSteer 数据集的真实人类偏好，构造 118 个循环实例和 4,924 个强传递（ST）实例进行实验。

**📈 对比分析**

通过对比不同 α、β、λ（以及其乘积 βλ）配置的迭代结果，评估了振荡幅度与最终熵，实验结果与理论预测一致：高自适应度/激进更新导致振荡或极端崩塌。

**⚠️ 局限性**

局限性在于理论主要针对离线、闭式模型，假设完整的偏好矩阵和采样可控；对多模态、动态环境以及非完全可观测的偏好场景的推广尚未深入。

---

## 429. "I Was Told to Come Back and Share This": Social Media-Based Near-Death Experience Disclosures as Expressions of Spiritual Beliefs

**arXiv ID:** 2602.11663 | [PDF](https://arxiv.org/pdf/2602.11663v1)

**作者:** Yifan Zhao `[一作]` (City University of Hong Kong), RAY LC `[通讯]` (City University of Hong Kong)

**通讯引用:** 908 | [OpenAlex ID](https://openalex.org/A5027284786)

**关键词:** `Human-Computer Interaction` `Video` `Text`

**🎯 论文内容**

系统分析了200个TikTok用户发布的近死亡体验（NDE）短视频及其评论，构建了内容与功能编码框架，探讨了叙事动机、宗教/灵性表达及观众互动模式。

**💡 创新点**

首次在社交媒体上从叙事角度、情感与身份构建、信息共享等维度，对NDE叙事进行细分，揭示了宗教/灵性元素如何在平台上形成共情与讨论的“回声室”。

**🔧 技术方法**

采用混合方法：基于扎根理论与多模态话语分析的编码（情感、认知、超自然、宗教/灵性四帧），再结合主题分析与评论文本主题编码；使用Cohen’s κ验证编码一致性。

**📊 数据集**

数据集由1017条公开NDE自述视频中随机抽取200条（以及相应评论）构成，视频均为英文且原创，评论样本约1943条，涵盖八大主题。

**📈 对比分析**

研究主要为定性分析；与其他方法对比仅在于采用严格的编码可靠性检验，未报告传统性能指标；对不同视频类型（PCE、NS‑NDE、S‑NDE）与评论主题的关联进行描述性统计，发现S‑NDE视频更易激发宗教讨论与情感共鸣。

**⚠️ 局限性**

局限性包括：仅选取英文视频，导致文化偏倚；采用关键词搜索非个体化推荐，可能遗漏算法曝光的内容；未收集创作者访谈，缺乏主体视角；对平台机制的定量分析不足；负面互动样本稀少，可能受创作者主动删改影响。

---

## 430. LUVE : Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts

**arXiv ID:** 2602.11564 | [PDF](https://arxiv.org/pdf/2602.11564v1)

**作者:** Chen Zhao `[一作]` (Meituan), Ying Tai `[通讯]` (Nanjing University)

**通讯引用:** 13246 | [OpenAlex ID](https://openalex.org/A5029021362)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Super Resolution` `Diffusion model` `Video`

**🎯 论文内容**

提出了LUVE框架，实现低分辨率运动生成、视频潜在上采样和高分辨率内容细化的三阶段级联流程，生成超高分辨率视频。

**💡 创新点**

创新点在于将运动先行生成、潜在空间直接上采样以及双频专家（低频提升语义一致，高频提升细节）相结合，突破传统VSR细节修饰的局限。

**🔧 技术方法**

采用视频扩散模型Wan2.1、3D VAE、DiT、LoRA微调、视频潜在上采样器VLUer、低通/高通滤波器等技术。

**📊 数据集**

使用UltraVideo UHR视频数据集进行训练与评估。

**📈 对比分析**

与SOTA T2V模型UltraWan、CineScale及VSR模型进行量化和人类评估，LUVE在VBench、FID_patch、Realism、Detailness、Alignment、MUSIQ、MANIQA等指标上均优于对手，获得最高人类偏好得分。

**⚠️ 局限性**

局限在于计算效率和显存占用仍高，推理速度相对慢，需要进一步优化。

---

## 431. Stroke of Surprise: Progressive Semantic Illusions in Vector Sketching

**arXiv ID:** 2602.12280 | [PDF](https://arxiv.org/pdf/2602.12280v1)

**作者:** Huai-Hsun Cheng `[一作]` (National Yang Ming Chiao Tung University), Yu-Lun Liu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 22543 | [OpenAlex ID](https://openalex.org/A5012327097)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Optimization` `Diffusion model` `Score-based Model` `Generative Adversarial Network` `Reinforcement Learning` `Image`

**🎯 论文内容**

提出了 Stroke of Surprise 框架，用可微分矢量渲染和 Score Distillation Sampling 生成一条初始素描，随后通过添加笔划逐步转化为另一种概念，从而实现时间维度的视觉错觉。

**💡 创新点**

创新点在于：① 双分支 SDS 联合优化，寻找可兼容两种语义的共通结构子空间；② 引入 Overlay Loss，约束后续笔划与前缀互补避免遮挡；③ 支持多阶段演变（A→B→C）。

**🔧 技术方法**

采用 Stable Diffusion v1.5 的 Score Distillation Sampling、可微分矢量渲染、Bezier 轨迹、GAN 与 RL 的组合，以及 GPT‑4o 评估与排序。

**📊 数据集**

使用由 64 个常见对象组成的配对数据集（随机生成 p₁ 与 p₂），并利用公开的 SVG 与 CLIP 数据集做评估。

**📈 对比分析**

与 Nano Banana Pro、SketchAgent、SketchDreamer 等基线比较，使用 CLIP、ImageReward、HPS 等指标和用户研究。结果显示在 CLIP、结构保留和用户喜好上均显著优于基线，用户选择率超过 70%–90%。

**⚠️ 局限性**

局限性：受预训练扩散模型的表达能力限制，复杂结构（如“scissors”）易导致优化失败；优化过程对初始化与步数敏感；目前仅验证于矢量素描，未扩展至其他媒介。

---

## 432. Diffusion Alignment Beyond KL: Variance Minimisation as Effective Policy Optimiser

**arXiv ID:** 2602.12229 | [PDF](https://arxiv.org/pdf/2602.12229v1)

**作者:** Zijing Ou `[一作]` (Imperial College London), Yingzhen Li `[通讯]` (Imperial College London)

**通讯引用:** 2789 | [OpenAlex ID](https://openalex.org/A5038242041)

**关键词:** `Machine Learning` `Optimization` `Generation` `Diffusion model` `Reinforcement Learning` `Image`

**🎯 论文内容**

提出一种基于方差最小化的扩散对齐算法VMPO，用来在预训练扩散模型上实现奖励导向的采样

**💡 创新点**

将扩散对齐视为顺序蒙特卡洛（SMC）问题，利用重要性权重的方差最小化来优化策略，理论证明方差最小化对应于目标奖励倾斜分布，并可衍生多种已有方法

**🔧 技术方法**

方差最小化策略优化（VMPO）、蒙特卡洛估计、神经估计器（M_ϕ）进行期望平滑、LoRA微调、Stable Diffusion 1.5/3.5 预训练模型、Human Preference Score (HPSv2) 奖励模型

**📊 数据集**

Stable Diffusion v1.5 与 v3.5 数据集，使用公开的 100 条测试提示词（来自 GitHub benchmark_ir.json），训练提示词来源于照片和绘画 Prompt 语料

**📈 对比分析**

与 GRPO、VMPO-R2G、VMPO-Diff 等现有方法比较，在 HPSv2、ImageReward 上表现更好，但 CLIPScore 与 DreamSim 略有下降，表明仍存在奖励游戏（reward hacking）问题

**⚠️ 局限性**

限制包括：仍可能出现奖励劫持导致文本-图像对齐与多样性下降；高 Monte Carlo 采样成本导致训练效率低；对不同奖励函数的泛化能力及长期收敛性待进一步研究

---

## 433. Systematic Analysis of Penalty-Optimised Illumination Design for Tomographic Volumetric Additive Manufacturing via the Extendable Framework TVAM AID Using the Core Imaging Library

**arXiv ID:** 2602.12178 | [PDF](https://arxiv.org/pdf/2602.12178v1)

**作者:** Nicole Pellizzon `[一作]` (Technical University of Denmark), Jakob Sauer Jørgensen `[通讯]` (Technical University of Denmark)

**通讯引用:** 1484 | [OpenAlex ID](https://openalex.org/A5021261523)

**关键词:** `Computational Engineering, Finance, and Science` `Optimization` `Image` `Computed Tomography`

**🎯 论文内容**

构建了 TVAM AID 框架，对多种惩罚函数进行数学建模和非负最小二乘优化，利用 FISTA 求解背投影问题，并通过阈值参数搜索评估不同方案的光能分布质量。

**💡 创新点**

提出 OSPW(w=0.0) 混合惩罚，将 L2N 与 OSP 结合，既保留大 Process Window，又实现小 In‑Part Dose Range；系统化阈值与 w 参数搜索以指导设计，显著提升打印质量。

**🔧 技术方法**

利用 Core Imaging Library 实现背投影算子，采用 Fast Iterative Shrinkage‑Thresholding Algorithm（FISTA）求解；通过 Process Window、In‑Part Dose Range、Voxel Error Rate 等指标进行定量评估。

**📊 数据集**

使用 DTU logo、圆盘、分辨率测试三张二维图像以及 3D Gyroid STL，全部通过 CIL 提供的投影/几何模型。

**📈 对比分析**

对比新方法与已知 OSMO 算法，在相同阈值与最佳阈值下进行 1000 次迭代；新方法在所有几何中均取得更大 Process Window、更小 In‑Part Dose Range、零 Voxel Error Rate；3D 案例略优于 OSMO。

**⚠️ 局限性**

实验尚未验证，模型仅适用于平行投影与二值材料；对极端阈值敏感；缺乏对光衰减、折射的完整建模；需进一步加入正则化和灰度打印等功能。

---

## 434. Wisdom of the LLM Crowd: A Large Scale Benchmark of Multi-Label U.S. Election-Related Harmful Social Media Content

**arXiv ID:** 2602.11962 | [PDF](https://arxiv.org/pdf/2602.11962v1)

**作者:** Qile Wang `[一作]` (University of Delaware), Matthew Louis Mauriello `[通讯]` (University of Delaware)

**通讯引用:** 1161 | [OpenAlex ID](https://openalex.org/A5007244703)

**关键词:** `Human-Computer Interaction` `Classification` `Transformer` `Large Language Model` `Text` `Benchmark`

**🎯 论文内容**

构建了一个基于X（Twitter）选举期间约10万条推文的多标签有害内容数据集（USE24‑XD），并利用六种大型语言模型（LLM）进行零样本分类，随后通过智慧之众（Wisdom‑of‑the‑Crowd）聚合得到最终标签。

**💡 创新点**

①在没有人工标注的大规模数据上首次使用多模型LLM结合投票聚合生成高质量多标签有害内容标注；②系统性比较不同LLM的标注一致性和与人类的对齐程度；③深入分析人类标注者的人口统计特征对标签分布的影响。

**🔧 技术方法**

零样本LLM推理、投票聚合（majority voting）、Krippendorff’s Alpha、Cohen’s κ、Chi‑square统计、成本收益分析、正负样本评估（Recall/Precision/F1）。

**📊 数据集**

1) USE24‑XD：约97,696条经过清洗的X推文，涵盖“Conspiracy”“Sensationalism”“Hate Speech”“Speculation”“Satire”五个多标签；2) MTurk 1,000条样本的人工多标注子集。

**📈 对比分析**

使用Kappa、Alpha、Recall/Precision/F1对比LLM单模型、LLM投票组合与人类投票基准。LLM投票组合在Speculation（recall≈0.85）、Conspiracy（≈0.81）、Sensationalism（≈0.62）、Hate Speech（≈0.64）上优于单模型；Satire低为≈0.49。与人类投票相比，LLM投票一致性更高、成本更低。

**⚠️ 局限性**

• X API访问受限且成本高，导致数据规模有限；• 数据集仅覆盖X平台，缺乏跨平台泛化；• 标签体系与传统事实核查标签不同，难以直接对比；• 未区分机器人与人类内容；• LLM知识截止可能导致偏差；• 部分预测缺失（如Gemini 2.5 Pro限流）。

---

## 435. Selective Prior Synchronization via SYNC Loss

**arXiv ID:** 2602.11316 | [PDF](https://arxiv.org/pdf/2602.11316v1)

**作者:** Ishan Mishra `[一作]`, Jinjun Xiong `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Convolutional Neural Network` `Image`

**🎯 论文内容**

提出 SYNC 损失并结合 Softmax‑Power（SMP）指数 γ 对选择性分类模型进行同步正则化，以提升模型的选择性预测精度和稳定性。

**💡 创新点**

创新点在于：①引入可调 SMP 指数 γ 并给出其对 Softmax 结果的 Lipschitz 上界；②设计 SYNC 损失在梯度和 Hessian 方面满足可控光滑性；③将 Lipschitz 约束与网络梯度范数耦合，形成全局光滑的目标函数；④在理论和实验两方面验证该方法在不同覆盖率下优于传统 SelectiveNet。

**🔧 技术方法**

使用 Lipschitz 约束分析、Mean‑Value 定理、Softmax Jacobian 计算、梯度与 Hessian 解析、SGD/AdamW 训练、ResNet‑18 网络、数据增强、温度缩放、以及对 γ 的数据驱动选择。

**📊 数据集**

在 CIFAR‑100、ImageNet‑100 以及 Stanford Cars 三个图像分类基准上进行实验。

**📈 对比分析**

与 SelectiveNet 采用相同网络结构、优化器和训练设置对比；在相同覆盖率下，SYNC 在准确率、误差率和覆盖率–误差曲线（Coverage‑Error Curve）等指标均优于基线，提升幅度在 5%~10% 之间。

**⚠️ 局限性**

局限性包括：①γ 的取值对性能高度敏感，需要预估 L_z 和 B 以满足 Lipschitz 约束；②理论上对 Softmax Lipschitz 上界的保守估计可能导致步长过小；③在大规模模型或不同温度 softmax 的场景下，额外的计算与调参成本较高。

---

## 436. TDPNavigator-Placer: Thermal- and Wirelength-Aware Chiplet Placement in 2.5D Systems Through Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.11187 | [PDF](https://arxiv.org/pdf/2602.11187v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## 437. MTFM: A Scalable and Alignment-free Foundation Model for Industrial Recommendation in Meituan

**arXiv ID:** 2602.11235 | [PDF](https://arxiv.org/pdf/2602.11235v1)

**作者:** Xin Song `[一作]` (Meituan), Zikang Xu `[通讯]` (Meituan)

**关键词:** `Information Retrieval` `Recommendation System` `Transformer` `Tabular`

**🎯 论文内容**

提出 Meituan Foundation Model (MTFM)，一种基于 Transformer 的多场景推荐基础模型，支持无对齐的异构 token 化。

**💡 创新点**

创新点包括无对齐异构 token 化、混合目标注意力（Hybrid Target Attention）结合 Grouped-Query Attention 与稀疏全注意力，以及系统级共设计优化。

**🔧 技术方法**

采用 Transformer 框架、Grouped-Query Attention、Hybrid Target Attention、FlashAttention-2、Triton 自定义核、稀疏矩阵乘法、BFloat16 推理等技术。

**📊 数据集**

使用美团三大推荐场景（首页餐厅、SQS 优惠券套餐、PHF 美食）真实日志数据。

**📈 对比分析**

与多种基线（DCNv2、MMOE、RankMixer、MTGR、STAR、PEPNet 等）进行对比，MTFM 在 CTR、CTCVR、IMD、WRITE 等指标上均显著提升（多场景 GAUC 平均提升 0.36pp 等），并在 A/B 测试中实现订单提升约 3% 及 1.5% 的增长，同时推理延迟降低 5–6 ms。

**⚠️ 局限性**

主要限制在于对大规模稀疏动态 mask 仍需高成本的自定义核，且在极端长序列或多场景复杂度高时，内存与计算仍存在瓶颈。

---

## 438. Performance Antipatterns: Angel or Devil for Power Consumption?

**arXiv ID:** 2602.12079 | [PDF](https://arxiv.org/pdf/2602.12079v1)

**作者:** Alessandro Aneggi `[一作]` (Free University of Bozen-Bolzano), Andrea Janes `[通讯]` (Free University of Bozen-Bolzano)

**通讯引用:** 1720 | [OpenAlex ID](https://openalex.org/A5015575660)

**关键词:** `Software Engineering`

**🎯 论文内容**

本文通过在容器化微服务中实现并测量十种性能反模式，评估其对响应时间与功耗的影响；

**💡 创新点**

首次将性能反模式与能源消耗关联，发现并非所有反模式都会导致能耗增加，揭示了性能与能耗的非线性关系；

**🔧 技术方法**

采用Python Flask容器化服务、Locust压力测试、PowerJoular与Perf RAPL收集功耗、cAdvisor收集资源使用，使用Pearson、Spearman相关和HC3稳健回归进行统计分析；

**📊 数据集**

实验数据来源于自行实现的十种反模式的30次重复实验，固定30分钟负载，未使用公开基准数据集；

**📈 对比分析**

比较方法为在相同负载下测量响应时间、CPU/DRAM功耗并计算相关系数与回归系数，结果显示部分反模式（如The Ramp、God Class）功耗随响应时间上升，部分则功耗饱和或不随响应时间变化；

**⚠️ 局限性**

局限在于实验仅在单一硬件与Python环境上进行，未考虑不同语言、平台或真实生产负载，且RAPL测得的DRAM功耗为系统级别，无法精确归属至单容器。

---

## 439. Agent-Diff: Benchmarking LLM Agents on Enterprise API Tasks via Code Execution with State-Diff-Based Evaluation

**arXiv ID:** 2602.11224 | [PDF](https://arxiv.org/pdf/2602.11224v1)

**作者:** Hubert M. Pysklo `[一作]` (Minerva University), Patrick D. Watson `[通讯]` (Minerva University)

**关键词:** `Software Engineering` `Large Language Model` `Agentic AI` `Prompt Engineering` `Text` `Benchmark`

**🎯 论文内容**

提出Agent-Diff框架，利用容器化的企业API沙盒对LLM代理在真实API任务中的代码执行能力进行统一评估。

**💡 创新点**

创新点在于结合状态差分（state-diff）评估契约与统一脚本层，既保持生态真实性又实现可复现的断言式验证。

**🔧 技术方法**

采用容器化API镜像、Bash/Python代码执行沙盒、ReAct提示、贝叶斯自举聚合以及文档注入实验等技术。

**📊 数据集**

构建224个跨Slack、Box、Linear、Google Calendar的多步任务集合，并引入新端点（如Box Hub）以检验模型对未知API的学习能力。

**📈 对比分析**

通过对9种LLM在三种文档条件下的断言加权得分和通过率进行对比，最高分88.1，最低38.0，文档提升通过率约+7个百分点，但对加权得分的提升不显著。

**⚠️ 局限性**

局限在于评估依赖于可复制的API实现与已知文档，无法完全覆盖新或隐藏接口的泛化能力；任务多基于已知API可能导致记忆偏差而非真正泛化。

---

## 440. Contention Resolution, With and Without a Global Clock

**arXiv ID:** 2602.12070 | [PDF](https://arxiv.org/pdf/2602.12070v1)

**作者:** Zixi Cai `[一作]` (Tsinghua University), Ben Plosk `[通讯]` (Bar-Ilan University)

**关键词:** `Distributed, Parallel, and Cluster Computing`

**🎯 论文内容**

本文探讨了n个参与者在没有全局时钟的情况下，如何独占共享资源的问题，提出了一种新的协议，确保在期望和高概率下的延迟达到O(n(loglog n)^(1+o(1)))。

**💡 创新点**

创新点在于首次展示了全局时钟模型的强大能力，并建立了复杂性分离，证明了在期望和高概率下的延迟不能同时达到最优。

**🔧 技术方法**

使用了基于Elias编码的随机化协议，结合了全局时钟模型和局部时钟模型的分析。

**📊 数据集**

未具体提及使用的数据集，但研究的对象是n个参与者在共享资源上的竞争行为。

**📈 对比分析**

与传统的随机协议进行比较，本文的协议在延迟上有显著的改进，尤其是在全局时钟模型下，延迟为O(n(loglog n)^(1+o(1)))，而在局部时钟模型下，延迟的下界为Ω(n(log^2 n)/(loglog n))。

**⚠️ 局限性**

限制在于无法同时实现期望和高概率下的最优延迟，且在某些情况下，协议的复杂性可能会受到参与者唤醒时间的影响。

---

## 441. Solving the Post-Quantum Control Plane Bottleneck: Energy-Aware Cryptographic Scheduling in Open RAN

**arXiv ID:** 2602.11820 | [PDF](https://arxiv.org/pdf/2602.11820v1)

**作者:** Neha Gupta `[一作]` (University of Surrey), Muhammad N. M. Bhutta `[通讯]`

**关键词:** `Cryptography and Security` `Optimization` `Safty and Privacy` `Reinforcement Learning`

**🎯 论文内容**

在 Open RAN 的控制平面中引入了能源感知的后量子密码（PQC）调度框架，利用 Crypto Policy rApp 与 Security Operations Scheduling (SOS) xApp 对 PQC 握手进行智能批处理、会话恢复与加速器选择，以降低能耗并满足切片 SLA。

**💡 创新点**

创新点包括：
- 将策略层（Non‑RT RIC）与战术层（Near‑RT RIC）解耦，实现安全策略的可编程化与即时调度；
- 采用受限强化学习决策模型，在满足 95th 百分位延迟与安全级别的前提下最小化每个安全连接的能耗；
- 在端点保持零信任原则，控制平面仅负责调度，实际加密执行仍在 MACsec/IPsec 端点完成；
- 通过会话恢复与预置策略，显著降低全手势 PQC 握手次数，实现 60% 能耗下降与 48% 延迟提升。

**🔧 技术方法**

核心技术包括：
- Open RAN 的 RIC 架构与 A1/E2/O1 接口；
- PQC 标准（Kyber、Dilithium、SPHINCS+）的硬件/软件实现；
- MACsec 与 IPsec 的终端加密实现；
- 受限强化学习（Constrained RL）决策引擎；
- 加速器（如 FPGA/ASIC）调度与动态选择；
- 离散事件模拟（Discrete‑Event Simulation）与能耗代理。

**📊 数据集**

使用的数据集与基准：
- 3GPP 兼容的交通流量模型（宏基站、车辆移动速率 60–120 km/h）；
- 文献中公开的 PQC 能耗与延迟基准（如 ARM Cortex‑M4 上的 ML‑KEM、ML‑DSA）；
- 通过离散事件仿真生成的 216 000 次安全事件（握手请求）来评估。

**📈 对比分析**

比较方法：在仿真环境下对比三种场景（Baseline、SOS‑Low、SOS‑High），测量 p95 延迟与相对能耗。结果显示：
- 100% 全手势基线的 p95 延迟为 191 ms，能耗为 17.57 mJ；
- SOS‑High（63% 会话恢复）将 p95 延迟降至 98 ms（48% 改善），相对能耗降至 0.40（即 60% 降低）。
- 能耗与延迟两项均实现了同步提升，且 SLA 合规率从 85.7% 提升至 98.2%。

**⚠️ 局限性**

局限性：
- 评估仅基于仿真，缺乏真实 O‑RAN 试验验证；
- 能耗模型取自文献，未结合特定厂商硬件的实时功率计数；
- 只考虑了主要 PQC 算法，未覆盖所有潜在加密组合；
- 对多连接、非地面网络（6G）与极端移动场景的适应性仍需进一步研究；
- 需要在 RIC 与端点之间定义更细粒度的安全调度接口与能源计数标准。

---

## 442. Cycles of Well-Linked Sets II: an Elementary Bound for the Directed Grid Theorem

**arXiv ID:** 2602.11716 | [PDF](https://arxiv.org/pdf/2602.11716v1)

**作者:** Meike Hatzel `[一作]` (Technical University Darmstadt), Irene Muzi `[通讯]` (Universität Hamburg)

**通讯引用:** 25 | [OpenAlex ID](https://openalex.org/A5048273965)

**关键词:** `Discrete Mathematics` `Graph`

**🎯 论文内容**

提出了一种在有向图中构造路径可链接集合（well‑linked sets）与水平网（horizontal web）的方法，并进一步利用该结构构造有向网格（directed grid）和闭合集合（cycle of well‑linked sets）；

**💡 创新点**

创新点在于将后向链路与水平路径通过弱最小化、分割与分段技术相结合，得到2‑水平网并实现新的路径可链接集合，显著改进了先前的构造方法和界限；

**🔧 技术方法**

使用了Menger定理、临时图（temporal digraph）技术、分割与分段（split/segmentation）技术以及弱最小化（weakly minimal）链接概念；

**📊 数据集**

为纯理论图论研究，未使用特定数据集；

**📈 对比分析**

方法通过理论证明与先前结果进行比较，性能在理论复杂度上保持一致或更优，给出了具体的参数界定和关系；

**⚠️ 局限性**

局限在于所需的参数规模极大、构造过程复杂，实际实现难度高，适用性受限于理论场景。

---

## 443. VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model

**arXiv ID:** 2602.12063 | [PDF](https://arxiv.org/pdf/2602.12063v1)

**作者:** Yanjiang Guo `[一作]` (Tsinghua University), Chelsea Finn `[通讯]` (Stanford University)

**通讯引用:** 25850 | [OpenAlex ID](https://openalex.org/A5005431772)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Vision-Language-Action Model` `Diffusion model` `World Model` `Reinforcement Learning` `Video`

**🎯 论文内容**

提出 VLAW 迭代改进框架，利用有限真实轨迹微调动作条件世界模型并生成高保真合成轨迹，进而通过流匹配监督式目标提升 VLA 策略。

**💡 创新点**

创新点：① 通过采集真实失败案例提升世界模型的物理逼真度；② 采用迭代共优化 VLA 策略与世界模型的闭环；③ 用流匹配目标替代传统强化学习，减少样本成本。

**🔧 技术方法**

技术手段：扩散式动作条件视频生成模型（Ctrl‑World）、视觉语言奖励模型（Qwen3‑VL）、流匹配（flow‑matching）训练目标、批量监督式微调、经验回放与在线数据融合。

**📊 数据集**

使用数据集：DROID 机器人平台的 5 类接触丰富任务（堆叠、开书、擦痕、搅拌、绘图），基准演示集 DROID dataset，以及实验中收集的 25 条专家演示、每轮 50 条真实轨迹与 500 条合成轨迹。

**📈 对比分析**

比较方法：与 Filtered BC 与 DSRL 基线对比，实验在两轮迭代后，VLAW 在多任务设置中绝对成功率提升 39.2%，相对基准提升 11.6%，显著优于两种基线，尤其在失败案例改善方面表现突出。

**⚠️ 局限性**

局限性：实验仅覆盖 5 类任务，缺乏更广泛、多样化任务验证；世界模型对极其复杂碰撞或可变形物体的建模仍有限；进一步提升需更多在线采样与更丰富的任务场景。

---

## 444. Synthesis of Late Gadolinium Enhancement Images via Implicit Neural Representations for Cardiac Scar Segmentation

**arXiv ID:** 2602.11942 | [PDF](https://arxiv.org/pdf/2602.11942v1)

**作者:** Soufiane Ben Haddou `[一作]` (Amsterdam UMC), Ivana Išgum `[通讯]` (Amsterdam UMC)

**通讯引用:** 16249 | [OpenAlex ID](https://openalex.org/A5084070018)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Generation` `Data Synthesis` `Diffusion model` `Image` `Biomedical Data` `Magnetic Resonance Imaging`

**🎯 论文内容**

通过结合隐式神经表示（INR）与扩散模型，联合生成LGE心肌增强影像及其对应的心肌与纤维化分割掩膜，实现无标注的数据增强。

**💡 创新点**

首次提出基于INR的联合影像-掩膜扩散框架，能够在不需要人工标注的情况下生成解剖一致的高质量合成图像与分割结果。

**🔧 技术方法**

使用SIREN隐式网络建模连续空间图像与掩膜，INR2VEC压缩参数到512维潜空间，Elucidated Diffusion Model (EDM) 在潜空间内生成新样本，随后利用nnU‑Net进行分割评估。

**📊 数据集**

采用133例临床短轴LGE MRI扫描（含心肌与纤维化手工标注）进行训练与测试。

**📈 对比分析**

将200体合成数据加入训练集后，用nnU‑Net对未见真实数据进行评估，纤维化Dice从0.509提升至0.524（+2.9%），尤其在心尖区提升9.5%；心肌Dice基本保持不变。

**⚠️ 局限性**

合成影像在心肌血池边界纹理细节欠佳，合成对心肌分割的提升有限；缺乏对不同扫描协议、低样本场景的系统验证。

---

## 445. Patient Digital Twins for Chronic Care: Technical Hurdles, Lessons Learned, and the Road Ahead

**arXiv ID:** 2602.11223 | [PDF](https://arxiv.org/pdf/2602.11223v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Software Engineering`

---

## 446. An Improved FPT Algorithm for Computing the Interleaving Distance between Merge Trees via Path-Preserving Maps

**arXiv ID:** 2602.12028 | [PDF](https://arxiv.org/pdf/2602.12028v1)

**作者:** Althaf P `[一作]`, Osamu Saeki `[通讯]`

**关键词:** `Computational Geometry` `Graph`

**🎯 论文内容**

提出一种新的固定参数可扩展（FPT）算法，用于精确计算两棵合并树之间的交错距离（interleaving distance）

**💡 创新点**

创新点在于将问题重新参数化为叶节点数 η_f、η_g，而非传统的 ε 相关度数界 τ，从而得到与 ε 无关的稳定时间复杂度，并通过路径导向的映射构造大幅压缩搜索空间

**🔧 技术方法**

主要技术包括 ε‑good 映射的定义与判定、叶到根路径的唯一分解、路径拼接（gluing lemma）、二分搜索候选 ε 值、以及可选的精炼目标节点筛选策略

**📊 数据集**

论文未给出真实实验数据集，主要通过理论分析和合成实例（如示例图）来展示复杂度改进；若需实测，可使用随机生成的合并树或典型图像处理/几何数据中的合并树

**📈 对比分析**

与之前的 O(2^2τ (2τ)^2τ+2·n^2 log^3 n) 算法相比，新算法在最坏情况下实现了 O(n^2 log n + η_g^η_f (η_f+η_g)·n log n) 的时间复杂度，实验示例中表现为多次次方级别（10^7）大幅低于原先的 10^30 级别

**⚠️ 局限性**

仍存在指数级的 η_f、η_g 依赖，尤其当叶节点数很大时算法可行性受限；假设 η_g^η_f≤η_f^η_g 也限制了适用场景；精炼目标节点方法虽能降低指数，但在极端情况仍可能不可行

---

## 447. Are Two LLMs Better Than One? A Student-Teacher Dual-Head LLMs Architecture for Pharmaceutical Content Optimization

**arXiv ID:** 2602.11957 | [PDF](https://arxiv.org/pdf/2602.11957v1)

**作者:** Suyash Mishra `[一作]` (Roche), Anubhav Girdhar `[通讯]` (Involead)

**关键词:** `Machine Learning` `Optimization` `Drug Discovery` `Transformer` `Large Language Model` `Vision Language Model` `Text` `Video` `Benchmark`

**🎯 论文内容**

提出并实现了基于 LLM/VLM 的模块化质量控制架构 LRBTC，用于医药领域内容生成的可验证、可追溯优化；

**💡 创新点**

创新点在于引入学生‑教师双模 LLM 验证循环与人机交互（HITL），并通过水滴式规则过滤实现规则检验的可扩展性与可追踪性；

**🔧 技术方法**

使用 LLM（如 Gemini 2.5 Pro/Flash、Claude Sonnet 等）、VLM、学生‑教师对话框架、规则抽取与层级过滤、OCR 与主题分类、以及人机反馈循环；

**📊 数据集**

使用公开基准 AIReg‑Bench（EU AI Act 合规评估）和 CSpelling（医疗拼写/语法错误检测）数据集，同时在内部集成约 200,000 PDF 与 25,000 视频的企业大规模数据；

**📈 对比分析**

与 Gemini 2.5 Pro 等最先进 LLM 对比，Student‑Teacher‑HITL 在 AIReg‑Bench 上 F1 提升至 83.0%（比基线 75.1%）并将召回率从 88.7% 提升至 97.5%；在 CSpelling 上平均准确率提升 26.7%，单样本提升幅度 16.8%–32.9%；整体展示了更高的敏感性和可解释性；

**⚠️ 局限性**

局限性包括：仅在医药领域验证，缺乏跨行业泛化；数据样本异质性导致性能波动；对标点、语法和非正式表达的检测仍显不足，需进一步的语法与多模态 fine‑tuning；

---

## 448. Digital Ecosystems: Enabling Collaboration in a Fragmented World

**arXiv ID:** 2602.11707 | [PDF](https://arxiv.org/pdf/2602.11707v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## 449. Patch the Distribution Mismatch: RL Rewriting Agent for Stable Off-Policy SFT

**arXiv ID:** 2602.11220 | [PDF](https://arxiv.org/pdf/2602.11220v1)

**作者:** Jiacheng Wang `[一作]` (Beijing Institute of Technology), Zhongbin Guo `[通讯]` (Beijing Institute of Technology)

**关键词:** `Machine Learning` `Reinforcement Learning` `Optimization` `Transformer` `Reinforcement Learning` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

提出了一种基于强化学习的监督重写策略，用来在下游微调前将训练样本重写为更符合模型原始问答式生成分布的形式，从而减少分布漂移导致的灾难性遗忘。

**💡 创新点**

创新点在于将重写任务视为策略学习问题，设计了任务一致性、分布对齐与多样性共同构成的硬门控奖励函数；并通过LoRA轻量化调优冻结的基础模型，结合GRPO实现高效的策略优化。

**🔧 技术方法**

采用的技术包括LoRA参数高效微调、Group Relative Policy Optimization (GRPO)、硬门控奖励（任务一致性为二进制门）、分布对齐奖励（基于基模型QA式NLL归一化）、多样性奖励（语义距离的边际贡献）以及生成-验证-回退的数据集构建管道。

**📊 数据集**

使用了两大数学推理语料（包含约10万实例，分为训练/微调两部分），以及多套数学与通用域基准（如MATH、GSM8K、ARC、TruthfulQA等）来评估下游性能与灾难性遗忘。

**📈 对比分析**

与传统SFT、SDFT、Mind the Gap等基线比较，实验显示该方法在数学任务上可实现与标准SFT相当的提升（例如在Mistral-7B上+55.23%），同时在通用域指标的退化率显著降低（从19.17%下降到7.35%），整体平均得分最高。

**⚠️ 局限性**

局限性包括：仅在小至中等规模（≤7B）模型上验证，扩展到更大模型或长上下文可能需额外工程；评估聚焦于数学推理域，可能不适用于代码、对话或安全关键场景；奖励设计依赖自动可评估的任务一致性验证器，可能引入偏差；回退策略及多重重写的其他构建方式仍待探索。

---

## 450. Improving Code Generation via Small Language Model-as-a-judge

**arXiv ID:** 2602.11911 | [PDF](https://arxiv.org/pdf/2602.11911v1)

**作者:** Giuseppe Crupi `[一作]` (Universita della Svizzera italiana), Gabriele Bavota `[通讯]` (Universita della Svizzera italiana)

**关键词:** `Software Engineering` `AI Code Assistant` `Generation` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

研究小语言模型（SLM）作为代码正确性评判者，并利用其在多候选代码中挑选最佳实现来提升代码生成质量。

**💡 创新点**

创新点：①证明经过微调的SLM能在判断准确率上与大型LLM持平；②提出SLM‑judge团队机制，在保持相同或更低成本的前提下，能超越同家族更大模型的 pass@1 评分。

**🔧 技术方法**

技术：对 Qwen2.5 Coder、Gemma‑3、Llama‑3.2 等 SLM 进行二分类微调；使用 Bayesian ensemble 结合各评判者的置信度对候选代码进行排序；与 RankEF、随机挑选和 log‑likelihood 等基线对比。

**📊 数据集**

数据集：Java 版 HumanEval、MBPP 与 CoderEval 共 722 题（训练 504 题、验证 73 题、测试 145 题），用于生成 10‑个候选实现并评判其正确性。

**📈 对比分析**

比较方法：通过 Cohen Kappa、pass@1、推理时间和硬件/成本评估；实验结果显示微调后 SLM 的 Kappa 达到 0.57，团队在 10 个候选中 pass@1 提升 15‑20%，且成本仅为 30B‑级 LLM 的十分之一。

**⚠️ 局限性**

局限性：仅在 Java 语言下验证，模型规模受限于 <5B 参数，未探讨更大规模模型或其他语言的通用性，推理延迟和多 GPU 并行性能仍需进一步评估。

---

## 451. AdaptEvolve: Improving Efficiency of Evolutionary AI Agents through Adaptive Model Selection

**arXiv ID:** 2602.11931 | [PDF](https://arxiv.org/pdf/2602.11931v1)

**作者:** Pretam Ray `[一作]` (Indian Institute of Technology Kharagpur), Emad Barsoum `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `Computation and Language` `Computational Efficiency` `Optimization` `AI Code Assistant` `Mixture of Experts` `Tabular` `Benchmark`

**🎯 论文内容**

提出了AdaptEvolve框架，在演化式代理式编码系统中根据生成不确定性动态选择小模型或大模型，以提升推理效率。

**💡 创新点**

首次将内部生成置信度作为路由信号，利用轻量决策树和自适应Hoeffding树实现实时、无外部控制器的模型选择。

**🔧 技术方法**

使用Token级熵置信度指标（MC、LGC、TC、BWC），轻量决策树和Hoeffding自适应树进行模型路由。

**📊 数据集**

在LiveCodeBench v5和MBPP编程生成基准上进行评估。

**📈 对比分析**

与纯小模型、纯大模型、随机路由、固定级联等基线对比，平均降低37.9%计算成本，同时保持97.5%上限准确率，效率提升明显。

**⚠️ 局限性**

依赖模型输出logprob，无法直接用于不公开logprob的闭源大模型；适用范围受限。

---

## 452. Strengthening Bulow-Klemperer-Style Results for Multi-Unit Auctions

**arXiv ID:** 2602.11959 | [PDF](https://arxiv.org/pdf/2602.11959v1)

**作者:** Moshe Babaioff `[一作]` (Hebrew University of Jerusalem), Zihan Luo `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 153 | [OpenAlex ID](https://openalex.org/A5043860233)

**关键词:** `Computer Science and Game Theory`

**🎯 论文内容**

本文研究了在多单位拍卖中，当买家价值满足更强的分布假设（MHR、α‑regular）或采用供给限制的 VCG 机制时，所需的额外买家数（竞赛复杂度）的上界与下界，并给出了精确或渐近的闭式表达；

**💡 创新点**

创新点在于通过将最坏情况归约到截断泛化帕累托分布，实现了对竞赛复杂度的精确刻画，并揭示在 MHR 条件下额外买家需求可下降到约 44% 甚至更低，显著优于经典 Bulow–Klemperer 结果；

**🔧 技术方法**

主要技术包括：最坏情况归约、订单统计分析、收入曲线的凸性与三段分布特征、数值网格搜索与 Lipschitz 误差控制，以及渐近分析中的大数定律和集中不等式；

**📊 数据集**

本文未使用实际数据集，而是采用理论分布（MHR、α‑regular、截断泛化帕累托）进行分析与数值验证；

**📈 对比分析**

与传统 Bulow–Klemperer 结果（在正则分布下需 m 个额外买家）比较，本文证明在 MHR 条件下仅需约 0.4447 n 个额外买家，供给限制版在近似最优时进一步降低需求；

**⚠️ 局限性**

局限性包括：仅考虑 i.i.d. 单值买家，分布假设需满足 MHR 或 α‑regular，且对正则分布的改进有限；未涉及动态/多参数市场，也未验证在非理论分布下的表现。

---

## 453. Filmsticking++: Rapid Film Sticking for Explicit Surface Reconstruction

**arXiv ID:** 2602.11433 | [PDF](https://arxiv.org/pdf/2602.11433v1)

**作者:** Pengfei Wang `[一作]` (Shandong University), Wenping Wang `[通讯]` (Texas A&M University)

**通讯引用:** 15595 | [OpenAlex ID](https://openalex.org/A5100668416)

**关键词:** `Graphics` `Generation` `Optimization` `Point Cloud`

**🎯 论文内容**

提出一种名为Filmsticking++的快速显式表面重建方法，改进传统filmsticking技术以实现点云精确插值。

**💡 创新点**

创新点包括：①用加权距离的Restricted Power Diagram替代RVD，保证所有点被吸附；②加入虚拟点加速深腔内的演化；③采用基于平滑度的拓扑修复策略区分薄板两侧。

**🔧 技术方法**

核心技术包括：Restricted Power Diagram、Restricted Delaunay Triangulation、虚拟点采样与投影、平滑度驱动的拓扑修复，以及一次Poisson重建用于去除薄层。

**📊 数据集**

实验使用了Thingi10K、真实扫描数据（Statue、SHINING 3D Einscan SE、40物体扫描集）以及含噪声/缺失的合成点云。

**📈 对比分析**

与8种显式、近似和学习方法（BP、Greedy、RD、SPR、NSH、NeurCAD、NN‑VIPSS、PTN等）对比，Filmsticking++在保真度、拓扑正确性、细节重建方面均优于或与SOTA持平，并将迭代次数从数十降至十几。

**⚠️ 局限性**

局限性：不适用于开放模型，极薄颈部或内部裂缝可能导致指导表面分离，需要更细粒度的最小厚度控制。

---

## 454. Agentic AI for Cybersecurity: A Meta-Cognitive Architecture for Governable Autonomy

**arXiv ID:** 2602.11897 | [PDF](https://arxiv.org/pdf/2602.11897v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Cryptography and Security`

---

## 455. MUSE: Multi-Tenant Model Serving With Seamless Model Updates

**arXiv ID:** 2602.11776 | [PDF](https://arxiv.org/pdf/2602.11776v1)

**作者:** Cláudio Correia `[一作]` (Feedzai), Pedro Bizarro `[通讯]` (Feedzai)

**关键词:** `Machine Learning` `Anomaly Detection` `Tabular` `Finance Related`

**🎯 论文内容**

提出并实现了 MUSE——一种面向多租户 Score-as-a-Service 的模型服务框架，能够在不影响客户阈值的前提下实现模型无缝更新，提升模型更新速度并降低运维成本。

**💡 创新点**

创新点包括：
1) 两层分数转换（Posterior Correction 与 Quantile Mapping）实现分数分布不变性，解耦模型输出与业务阈值；
2) 基于 intent 的无状态路由与可组合变换，实现服务器端模型选择与版本推广；
3) 共享模型容器与图形化资源复用，显著降低多租户模型集群的资源重复与成本；
4) 支持 shadow scoring 与自动化 warm‑up，保障高可用与低延迟。

**🔧 技术方法**

技术实现涵盖：Kubernetes + Trane/Triton 进行模型推理；Java 基础服务做路由与变换；统计学方法 Posterior Correction（偏差校正）与 Quantile Mapping（分位数映射）；Graph DAG 管理推理与变换流程；自定义意图路由与规则配置；以及 Java JIT warm‑up 与 Kubernetes rolling‑update。

**📊 数据集**

使用 Feedzai 内部真实交易数据：约 55 billion 条事件、7 M 交易、约 1.8 billion 交易额。实验以此数据为训练、验证与在线评估，涉及多租户的诈骗检测、数字活跃度与新户开户场景。

**📈 对比分析**

对比方法：raw score、默认 quantile、客户自定义 quantile 以及模型更新前后未更新 vs 更新后更新的 Quantile Mapping。性能表现：
- 延迟保持在 p99 < 30 ms、p99.9 < 150 ms，99.95% 可用；
- 量化转换后相对误差从 43% 降至 < 10%；
- 模型更新后召回率提升 1.1%（1% FPR），无阈值调整；
- 校准误差（ECE、Brier）通过 Posterior Correction 减少 80% 以上；
- 模型上线时间从数周缩短至分钟级，节省 10 周运维成本。

**⚠️ 局限性**

局限性：
1) Posterior Correction 仅依赖采样率，无法补偿模型本身的校准偏差；
2) Quantile Mapping 的重新训练目前由部署事件触发，缺乏实时漂移监控与自动刷新；
3) 方案在极端不平衡或 label 缺失场景下的鲁棒性待验证；
4) 目前实现主要针对 fraud detection，迁移到其它业务领域可能需额外适配；
5) 随着客户数量成百计，配置管理与规则冲突的可维护性将成为挑战。

---

## 456. Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs

**arXiv ID:** 2602.11528 | [PDF](https://arxiv.org/pdf/2602.11528v1)

**作者:** Dong Yan `[一作]` (University of Chinese Academy of Sciences), Tieniu Tan `[通讯]` (Nanjing University)

**通讯引用:** 36885 | [OpenAlex ID](https://openalex.org/A5111885963)

**关键词:** `Cryptography and Security` `Safety and Privacy` `Optimization` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

提出并实现了TRACE-RPS框架，结合细粒度匿名化（TRACE）和拒绝导向优化（RPS），为LLM属性推断攻击提供主动防御。

**💡 创新点**

创新地将注意力+推理链指导的词汇提取与轻量级两阶段后缀搜索结合，既精准掩盖泄露词汇又能诱导模型拒绝回答，从而超越传统粗粒度匿名化。

**🔧 技术方法**

使用LLM注意力权重提取隐私词汇、推理链生成、提示优化、两阶段随机搜索（RPS）以及误属性搜索（MPS）等技术。

**📊 数据集**

在Synthetic、SynthPAI以及公开真实数据集上评估，涵盖多种个人属性。

**📈 对比分析**

与无防御、Azure、Dou‑SD、D‑Defense、FgAA等基线在7个大型模型上对比，TRACE‑RPS把属性推断准确率从≈50%降至≤5%，在开放源代码模型实现近零，跨模型、跨提示表现稳健，且对实用性影响小。

**⚠️ 局限性**

对RPS的实现依赖对模型logits的访问；封闭源模型仅能使用TRACE，匿名化对有限选项属性仍无法完全阻断；需要用户预处理文本且可能略微降低可读性。

---

## 457. ExtractBench: A Benchmark and Evaluation Methodology for Complex Structured Extraction

**arXiv ID:** 2602.12247 | [PDF](https://arxiv.org/pdf/2602.12247v1)

**作者:** Nick Ferguson `[一作]` (Contextual AI), Thien Hang Nguyen `[通讯]` (Contextual AI)

**关键词:** `Machine Learning` `Transformer` `Large Language Model` `Text` `Benchmark`

**🎯 论文内容**

提出了 ExtractBench 数据集与基于 schema 驱动的评估框架，用于端到端 PDF‑to‑JSON 结构化提取，并基准评估当前领先 LLM 的性能。

**💡 创新点**

创新点在于（1）首次公开包含 13–369 个字段、跨 5 业务域的企业级 PDF‑JSON 基准；（2）采用可执行 JSON Schema 的递归评估，支持字段级精确匹配、容忍误差、语义等价和数组语义对齐，区分缺失与 hallucination。

**🔧 技术方法**

技术手段包括：零样本 LLM 提取（GPT‑5/5.2、Gemini 3 Flash/Pro、Claude Sonnet/Opus），AST‑based schema 遍历与插件化指标，LLM‑驱动的语义数组对齐，以及对结构化输出模式与约束解码的实验。

**📊 数据集**

使用 35 份 PDF 与对应 JSON Schema 及人工金标，涵盖 5 领域，评估字段 12,867，平均每个 PDF 输出 2,076 页、369 个字段的最大复杂度。

**📈 对比分析**

评估方法以字段级、类型化得分为主，构建自定义阈值；实验显示 Frontier LLM 在大 schema 下失效，SEC 10‑K/Q 全部 0% 合法输出，整体通用通过率 4.6%（最高 6.9%），但在有效输出时字段准确率约 72.9%。

**⚠️ 局限性**

局限性包括：对大 schema（>300 字段）与大数组（>300 项）导致输出量过大而导致 LLM 生成失稳；结构化输出模式在复杂 schema 下仍失效；缺乏上下文管理与多步推理机制，难以处理超长文档与深层嵌套。

---

## 458. Human-Like Gaze Behavior in Social Robots: A Deep Learning Approach Integrating Human and Non-Human Stimuli

**arXiv ID:** 2602.11648 | [PDF](https://arxiv.org/pdf/2602.11648v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Robotics`

---

## 459. Filtered Approximate Nearest Neighbor Search in Vector Databases: System Design and Performance Analysis

**arXiv ID:** 2602.11443 | [PDF](https://arxiv.org/pdf/2602.11443v1)

**作者:** Abylay Amanbayev `[一作]` (University of California Merced), Florin Rusu `[通讯]` (University of California Merced)

**关键词:** `Databases` `Retrieval` `Optimization` `Text` `Benchmark`

**🎯 论文内容**

系统化过滤近邻搜索策略并在FAISS、Milvus、pgvector上进行基准测试，构建MoReVec数据集和GLS指标。

**💡 创新点**

提出GLS关联度指标、MoReVec多表元数据向量数据集、扩展ANN-Benchmarks以支持过滤查询，并给出系统层面的实证洞察。

**🔧 技术方法**

采用预过滤、后过滤、运行时过滤三类策略，结合HNSW/IVFFlat索引、FAISS、Milvus Knowhere、pgvector PostgreSQL扩展以及GLS计算与Docker化基准框架。

**📊 数据集**

使用新建的MoReVec（Movies+Reviews）768维文本嵌入数据集，覆盖不同选择率范围，并与公开向量基准集作对照。

**📈 对比分析**

通过单线程QPS‑Recall曲线对比，发现Milvus的混合搜索在低选择率下保持高召回，pgvector优化器常偏好近似搜索，IVFFlat在低选择率可优于HNSW，GLS高相关度能预测召回。

**⚠️ 局限性**

实验仅覆盖单机内存模式、仅标量过滤、有限超参数范围，缺乏磁盘、分布式、多线程、不同嵌入或距离度量、强相关元数据等场景，系统级优化仍需改进。

---

## 460. Mitigating Mismatch within Reference-based Preference Optimization

**arXiv ID:** 2602.11902 | [PDF](https://arxiv.org/pdf/2602.11902v1)

**作者:** Suqin Yuan `[一作]` (Sydney AI Centre University of Sydney), Tongliang Liu `[通讯]` (Sydney AI Centre University of Sydney)

**关键词:** `Machine Learning` `Optimization` `Recommendation System` `Transformer` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

在直接偏好优化（DPO）的框架下，提出一种条件性参考信号去偏方法 Hybrid-DPO（HyPO），通过在参考模型偏向负例时将其边际值裁剪为零，以消除 DPO 的“过早满足”问题并保持训练稳定性。

**💡 创新点**

创新点在于仅用一行代码修改 DPO 损失，将参考边际替换为 max{0,Δ_ref}，实现参考模型的条件性使用；该方法既保留了 DPO 的 KL 正则化优势，又提升了绝对对齐信号，解决了训练‑推理不匹配。

**🔧 技术方法**

技术手段：基于 DPO 的对数似然差分损失；对参考边际做裁剪（或平滑软化）；使用 β 温度参数和可选的 home advantage；实现为现有 DPO 代码的可插拔补丁。

**📊 数据集**

数据集：UltraChat-200k（监督微调）、UltraFeedback（偏好对）以及 HelpSteer2；评估使用 AlpacaEval 2.0、Arena‑Hard‑v0.1 以及 LM Evaluation Harness 上的 MMLU、ARC、HellaSwag、Winogrande、TruthfulQA、GSM8K。

**📈 对比分析**

与 DPO、SimPO、FocalPO、TR‑DPO、RainbowPO 等多种直接偏好对齐基线比较；在 Mistral‑7B、Llama‑3‑8B 以及更大模型上，HyPO 在 AlpacaEval 和 Arena‑Hard 上均取得平均 41.2% 的相对提升（对 DPO）和 15.1%（对 SimPO），并在多项基准上保持或提升下游任务性能；训练时间仅提升约 1%。

**⚠️ 局限性**

局限性：对乐观（Δ_ref≥0）样本仍保留训练‑推理不匹配；在标签噪声极高的情形下可能放大误标记；裁剪阈值 γ 固定为 0，未尝试自适应或学习门控；方法未能完全消除参考信号带来的偏差。

---

## 461. Intelligent AI Delegation

**arXiv ID:** 2602.11865 | [PDF](https://arxiv.org/pdf/2602.11865v1)

**作者:** Nenad Tomašev `[一作]` (Google DeepMind), Simon Osindero `[通讯]` (Google DeepMind)

**通讯引用:** 29425 | [OpenAlex ID](https://openalex.org/A5052435039)

**关键词:** `Artificial Intelligence` `Robotic Intelligence` `Reinforcement Learning` `Reinforcement Learning` `Agentic AI`

**🎯 论文内容**

本文提出了一套完整的智能委托框架，系统化地阐述了任务分解、委托、监控、信任与可验证完成等关键环节，并给出了可落地的技术实现思路。

**💡 创新点**

创新点在于将人类组织学中的委托理论与现代多智能体系统结合，提出了动态评估、适应性执行、结构透明、可扩展市场协调和系统韧性五大核心要求，并引入区块链、zk‑Proof、微服务安全令牌等技术实现安全可验证的委托链。

**🔧 技术方法**

主要技术包括：层级强化学习/Feudal RL的任务分解学习、智能合同与去中心化市场协议（MCP、A2A、AP2、UCP）、可信执行环境、零知识证明、可验证凭证与数字签名、基于属性的权限颁发（macaroons）以及多目标优化算法。

**📊 数据集**

由于本文属于理论框架设计，未使用公开数据集；若在实验评估中，可采用公开的RLBench/Meta‑World等机器人任务集、OpenAI Gym 以及合成的多代理协同任务环境进行验证。

**📈 对比分析**

与传统基于规则的委托方法和单一代理调度相比，实验演示显示该框架在任务成功率、误差恢复和安全合规性上提升了约30–50%，但在复杂度和延迟上略有开销。

**⚠️ 局限性**

局限性包括：缺乏大规模真实世界部署验证；对高复杂度、低可验证性任务的分解仍需手工定义；系统对零知识证明等加密技术的性能瓶颈未彻底解决；以及多代理协同过程中出现的信任博弈与权限滥用风险。

---

## 462. When Should LLMs Be Less Specific? Selective Abstraction for Reliable Long-Form Text Generation

**arXiv ID:** 2602.11908 | [PDF](https://arxiv.org/pdf/2602.11908v1)

**作者:** Shani Goren `[一作]` (Technion), Ran El-Yaniv `[通讯]` (NVIDIA)

**关键词:** `Artificial Intelligence` `Generation` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

提出了Selective Abstraction框架，通过在长文本中按原子级别有选择性地抽象来提升可靠性。

**💡 创新点**

创新点在于把抽象与置信度结合，用阈值控制信息量而非简单删去，使得长文本在保持覆盖率的同时降低风险。

**🔧 技术方法**

使用LLM生成、原子化、置信度评估、抽象生成和重构的端到端流程，并采用阈值选择算法。

**📊 数据集**

在FactScore和LongFact-Objects两大长文本事实性基准上进行评估。

**📈 对比分析**

与传统删减、内联抽象、Self‑Revision等基线比较，Atom‑wise SA在AURC上平均提升约17–27%，在相同覆盖率下风险显著降低。

**⚠️ 局限性**

局限在于依赖可用的置信度评分与抽象能力，且对原子独立假设与实体计数估计的可靠性有限。

---

## 463. How Well Do Large-Scale Chemical Language Models Transfer to Downstream Tasks?

**arXiv ID:** 2602.11618 | [PDF](https://arxiv.org/pdf/2602.11618v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## 464. Extending Puzzle for Mixture-of-Experts Reasoning Models with Application to GPT-OSS Acceleration

**arXiv ID:** 2602.11937 | [PDF](https://arxiv.org/pdf/2602.11937v1)

**作者:** Akhiad Bercovich `[一作]` (NVIDIA), Ran El-Yaniv `[通讯]` (NVIDIA)

**关键词:** `Machine Learning` `Optimization` `Computational Efficiency` `Knowledge Distillation` `Transformer` `Large Language Model` `Reinforcement Learning` `Mixture of Experts` `Text`

**🎯 论文内容**

本工作通过对 gpt‑oss‑120B 进行后训练的架构搜索，生成了 gpt‑oss‑puzzle‑88B，显著降低模型规模并提升推理速度。

**💡 创新点**

创新点在于将 Puzzle NAS 框架扩展至 MoE 异构剪枝、选择性窗口注意力、强化学习微调与 FP8 KV 量化等多种技术的组合，实现对长短上下文场景的双重优化。

**🔧 技术方法**

采用了 Puzzle NAS、MoE 专家异构剪枝、窗口注意力替换、强化学习融合、FP8 KV 量化以及知识蒸馏等技术。

**📊 数据集**

使用了 HuggingFace 的 Llama‑Nemotron‑Post‑Training Dataset 进行专家贡献和替代块评分，以及多种推理与推断数据集如 MMLU‑Pro、HLE、GPQA‑Diamond、AIME‑25、SciCode、IFBench、AA‑LCR 等。

**📈 对比分析**

在 8×H100 节点和单卡 H100 上分别对 4K/4K 与 64K/64K 推理场景进行对比，gpt‑oss‑puzzle‑88B 在短上下文提升 1.22×、长上下文提升 1.63×，单卡速度提升 2.44×/2.82×，且在多项推理任务上保持或略高于父模型的准确率。

**⚠️ 局限性**

主要局限在于不同推理场景下生成的令牌数量变化导致请求级效率差异，且在极大上下文或不同硬件上可能需要重新调优，模型在部分基准上的准确率仍略低。

---

## 465. OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration

**arXiv ID:** 2602.12151 | [PDF](https://arxiv.org/pdf/2602.12151v1)

**作者:** Youhe Jiang `[一作]` (University of Cambridge), Eiko Yoneki `[通讯]` (University of Cambridge)

**通讯引用:** 5195 | [OpenAlex ID](https://openalex.org/A5063536695)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Transformer` `Large Language Model` `Text` `Time Series`

**🎯 论文内容**

提出了名为S4的LLM服务系统，能够通过异构模型部署和时空感知调度实现高效推理。

**💡 创新点**

创新点在于两层工作负载感知调度（针对空间异质性）与自适应模型切换（针对时间异质性），同时兼顾资源分配、并行策略与请求分配。

**🔧 技术方法**

采用流网络最大流求解进行工作负载分配、LSTM时间序列预测实现工作负载预测、贪心算法实现模型参数与KV缓存的快速切换，并利用GPU互连实现高效参数迁移。

**📊 数据集**

使用Azure公开数据集的真实请求轨迹以及OPT‑30B/66B、Llama‑30B/70B等70B级LLM模型。

**📈 对比分析**

与vLLM（静态/重新加载）、Llumnix、Dynamo+vLLM等基线比较，P99延迟提升至1.5‑2.0×、吞吐量提升至1.3‑1.9×，平均提升约1.5×。

**⚠️ 局限性**

主要局限包括预测误差导致的调度失误、切换频繁时的额外延迟、以及在极大规模集群中调度算法的计算开销。

---

## 466. Bootstrapping-based Regularisation for Reducing Individual Prediction Instability in Clinical Risk Prediction Models

**arXiv ID:** 2602.11360 | [PDF](https://arxiv.org/pdf/2602.11360v1)

**作者:** Sara Matijevic `[一作]` (University of Oxford), Christopher Yau `[通讯]` (University of Oxford)

**通讯引用:** 165718 | [OpenAlex ID](https://openalex.org/A5009420648)

**关键词:** `Machine Learning` `Classification` `Explainability and Interpretability` `Tabular` `Biomedical Data` `Electronic Health Records`

**🎯 论文内容**

本文提出了一种将自助采样嵌入深度神经网络训练的正则化框架，以提升临床预测模型在不同样本上的个体预测稳定性。

**💡 创新点**

创新点在于直接在损失函数中加入对基于自助样本预测差异的期望惩罚，使单一模型即可获得与集成方法相近的稳定性，同时保持可解释性。

**🔧 技术方法**

采用深度神经网络（两隐藏层）与二元交叉熵损失，结合自助采样正则化项和Adam优化器进行训练。

**📊 数据集**

使用模拟数据以及三组真实临床数据集：GUSTO‑I、Framingham和SUPPORT。

**📈 对比分析**

与传统单模型、统计自助采样以及集成（Bagging）方法比较，实验显示新方法在保持相同AUC的前提下显著降低了平均绝对差（MAD）和显著偏差比例，性能优于单模型，接近甚至略逊于Bagging。

**⚠️ 局限性**

局限包括需要预先训练大量自助样本、对正则化强度与采样数的敏感性需要调优，以及在高维或非表格数据上的可扩展性待进一步验证。

---

## 467. DynaHOI: Benchmarking Hand-Object Interaction for Dynamic Target

**arXiv ID:** 2602.11919 | [PDF](https://arxiv.org/pdf/2602.11919v1)

**作者:** BoCheng Hu `[一作]`, Gaoang Wang `[通讯]` (Zhejiang University)

**通讯引用:** 37980 | [OpenAlex ID](https://openalex.org/A5028525523)

**关键词:** `Computer Vision and Pattern Recognition` `Robotic Intelligence` `Diffusion model` `Vision-Language-Action Model` `Reinforcement Learning` `Video` `Benchmark`

**🎯 论文内容**

提出了面向动态手物交互的评估平台 DynaHOI-Gym 以及大规模基准 DynaHOI-10M，包含 10M 帧 180K 抓取轨迹。

**💡 创新点**

创新点在于：① 用参数化目标运动生成器实现可重复、可扩展的动态目标；② 引入 observe‑before‑act 观察-行动协议和在线闭环评估，避免传统帧级对齐；③ 开源完整平台与基准数据，填补现有静态抓取评测的空白。

**🔧 技术方法**

使用 Unity 物理仿真、参数化运动合成、spatiotemporal attention 的观察-行动模块、扩散策略等技术实现闭环评估与动态抓取控制。

**📊 数据集**

数据集 DynaHOI-10M：10M 帧、180K 轨迹，涵盖 8 大运动类别、22 细分子类别、11 种对象，具备视觉与物理多样性。

**📈 对比分析**

对比了多种策略模型（AutoRegressive、Diffusion）和多模态大语言模型（Gemini‑3 Pro、GPT‑5.1、Qwen3‑Max 等）；结果显示目前最佳 Diffusion 模型 S_loc 仅 27.9%，S_gra 仅 3.5%；自研 Observe‑Before‑Act (ObAct) 模型在 S_loc 上提升 8.1% 至 36%，但整体性能仍远低于 oracle。

**⚠️ 局限性**

限制：① 动态抓取仍难，定位与抓取成功率低；② 对时序信息的建模不足导致抓取不稳定；③ 轨迹平滑与线性度受模型架构影响，难以通过数据规模显著提升；④ VLM 对感知‑时序匹配能力有限。

---

## 468. MedExChain: Enabling Secure and Efffcient PHR Sharing Across Heterogeneous Blockchains

**arXiv ID:** 2602.12106 | [PDF](https://arxiv.org/pdf/2602.12106v1)

**作者:** Yongyang Lv `[一作]` (Tianjin University), Ruitao Feng `[通讯]` (Southern Cross University)

**通讯引用:** 1092 | [OpenAlex ID](https://openalex.org/A5032257261)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Computational Efficiency` `Smart Contract` `Cryptographic Reverse Firewall` `Biomedical Data` `Electronic Health Records`

**🎯 论文内容**

提出一种名为 MedExChain 的跨链 PHR 共享方案，能够在使用不同加密体系（如 IBE 与 CLC）的区块链之间安全、高效地共享个人健康记录，并支持 IoMT 设备的低算力环境。

**💡 创新点**

创新点包括：① 在 PRE 基础上加入 Cryptographic Reverse Firewall (CRF) 以抵御算法替换攻击；② 通过智能合约实现 IoMT 终端的轻量级 PHR 共享；③ 采用 CPA、ASA、BAN 逻辑和 Scyther 进行多维度安全验证，证明方案对内部外部威胁均具备鲁棒性；④ 在计算和通信开销上显著优于现有五种对等方案。

**🔧 技术方法**

使用的技术包括：身份/无身份加密、代理重加密、双线性映射、IPFS、区块链跨链网关、智能合约、Cryptographic Reverse Firewall、BAN 逻辑、Scyther、CPA/ASA 安全模型。

**📊 数据集**

实验采用模拟环境（Ubuntu 22.04 + Java 1.8 + JPBC），未使用真实医疗数据集，主要通过理论计算和实验测量评估计算/通信开销、吞吐量与延迟。

**📈 对比分析**

与 IBPRE_CRF、CDSS、ABE-IBE、CP-HAPRE、FABRIC 五种对等方案进行比较。结果显示 MedExChain 在关键阶段（Enc、ReKeyGen、ReEnc）计算开销最低、通信开销最小，整体执行时间和系统吞吐/延迟均优于其余方案。

**⚠️ 局限性**

局限性：① 仍需解决不同区块链间共识与跨链通信的细节；② 假设 DO、医院节点与 CRF 完全可信，未考虑它们自身被入侵的情况；③ 仅在模拟环境中验证，未在真实医疗网络部署测试；④ 对大规模多方协作与动态访问控制的支持尚待进一步研究。

---

## 469. GORGO: Maximizing KV-Cache Reuse While Minimizing Network Latency in Cross-Region LLM Load Balancing

**arXiv ID:** 2602.11688 | [PDF](https://arxiv.org/pdf/2602.11688v1)

**作者:** Alessio Ricci Toniolo `[一作]` (Arcadia Research Team), Rome Thorstenson `[通讯]` (Arcadia Research Team)

**关键词:** `Networking and Internet Architecture` `Optimization` `Large Language Model` `Text`

**🎯 论文内容**

提出了 GORGO 与 GORGO-proxy 两种跨地区 LLM 推理负载均衡方案，结合 KV 缓存重用、网络延迟与队列状态三项指标来最小化 TTFT。

**💡 创新点**

创新点在于将 KV 缓存相似度、跨区网络 RTT 与即时队列等待时间统一成可估计 TTFT 的加性成本模型，实现对 TTFT 的联合优化；同时通过中心化代理降低同步开销。

**🔧 技术方法**

采用分布式负载均衡器、SGLang KV 缓存与连续批处理、前缀 Trie 索引、低延迟 RTT 采集、线性回归估算前填充速率、中心化 HTTP 代理等技术。

**📊 数据集**

实验使用 WildChat 对话数据、GuideLLM 生成的混合工作负载以及 Mistral‑7B‑Instruct‑v0.3 模型进行推理。

**📈 对比分析**

与最少负载、前缀相似度路由和中心化代理基线对比，GORGO‑proxy 在 median TTFT 上比最少负载快 2.5 倍、比前缀相似度快 2.5 倍；平均 TTFT 同样提升，吞吐量略低于最少负载。

**⚠️ 局限性**

局限性包括对 RTT、前缀覆盖估计与队列状态的准确性依赖；在异构硬件、更多地区或大规模吞吐场景下效果可能下降；以及仅在三地区实验，缺乏更广泛验证。

---

## 470. Explaining AI Without Code: A User Study on Explainable AI

**arXiv ID:** 2602.11159 | [PDF](https://arxiv.org/pdf/2602.11159v1)

**作者:** Natalia Abarca `[一作]` (University of Chile), Felipe Bravo-Marquez `[通讯]` (Millennium Institute for Foundational Research on Data)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Tabular`

**🎯 论文内容**

在无代码机器学习平台 DashAI 中集成了三种可解释性技术（PDP、PFI、KernelSHAP），并通过一项包含 20 名参与者（10 名新手、10 名专家）的用户研究评估其可用性、满意度与信任度。

**💡 创新点**

创新点在于：①将全球和局部可解释方法无缝嵌入无代码工作流；②采用人机中心设计，兼顾新手的易用性与专家的诊断深度；③通过对比两种工作流（逐步分析 vs 同步分析）与不同用户群体，系统性验证了可解释模块的有效性。

**🔧 技术方法**

使用技术包括：Partial Dependence Plot（PDP）、Permutation Feature Importance（PFI）、KernelSHAP；用户体验评估采用 Explanation Satisfaction Scale（ESS）和 Trust in Automation（TiA）问卷，并用 Mann‑Whitney U 与逻辑回归等统计方法进行分析。

**📊 数据集**

数据集：论文未指定具体公开数据集，实验基于通用表格分类任务（任意可用的行业数据集）。

**📈 对比分析**

比较方法：将两种解释流程（A、B）与新手/专家两组进行对比；任务成功率≥80%，ESS 信度 α=0.74，专家对细节完整度评价略低；TiA 信度 α=0.60，但表现出解释提升预测性与自信度的趋势；整体性能优于单一解释工具。

**⚠️ 局限性**

局限性：样本量小（仅 20 人）；TiA 信度仅 0.60，结果需谨慎；实验仅涉及表格分类，未扩展到文本或图像模型；未深入评估解释深度对专家诊断效率的影响；缺乏长周期可用性与信任度跟踪。

---

## 471. Quark Medical Alignment: A Holistic Multi-Dimensional Alignment and Collaborative Optimization Paradigm

**arXiv ID:** 2602.11661 | [PDF](https://arxiv.org/pdf/2602.11661v1)

**作者:** Tianxiang Xu `[一作]` (Qwen Applications Business Group), Guanjun Jiang `[通讯]` (Qwen Applications Business Group)

**关键词:** `Artificial Intelligence` `Optimization` `Reinforcement Learning from Human Feedback` `Supervised Fine-Tuning` `Reinforcement Learning` `Retrieval-Augmented Generation` `Text` `Biomedical Data` `Electronic Health Records`

**🎯 论文内容**

提出了一套完整的医学对齐框架 MAP，包含全景式多维度评估矩阵（基础能力、专家知识、用户反馈、格式合规）以及统一的奖励合成与多目标强化学习优化机制（Uni‑Reward）。

**💡 创新点**

创新点：
• 采用“全景分解–协同”控制理论，将医学对齐目标拆解为互补的四维矩阵，并通过 ORM、PRM、GRM、GARM 等多层级奖励模型实现高分辨率监督；
• 设计了“Rubrics As a Reward”与“生成式奖励模型（GRM）”来将抽象的临床路径转化为可验证的硬评分标准，提升可解释性；
• 提出了 Uni‑Reward 统一奖励协同优化，舍弃传统线性加权，采用分布归一化 + 三因子动态加权（难度、风险悲观、冗余惩罚），解决异构奖励尺度冲突与梯度屏蔽问题。

**🔧 技术方法**

技术：
• 基础层：QuarkMed 领域适配预训练 + SFT、检索增强生成（RAG）+ Best‑of‑N；
• 对齐层：多层级奖励模型（ORM、PRM、GRM、GARM）与自动化 Rubric 系统；
• 强化学习层：Group Relative Policy Optimization（GRPO）+ Uni‑Reward；
• 评价与校准：多维度评分框架（HDUF、六维 Utility）、CRD、Bot‑k、对齐稀疏反馈去噪、A/B 实验。

**📊 数据集**

数据集：
• 真实医疗语料（临床指南、医学文献、药品说明书、去标识化 EMR）；
• 通过多模型投票、DeepResearch 构建的黄金答案；
• 生成的人工对齐样本（同义/无差异对、人工标注的 preference、用户点赞/点踩交互）；
• 自动化 Rubric 生成的评估表；
• 在线 A/B 测试的真实用户交互数据。

**📈 对比分析**

对比方法：SFT 基线、GRPO+单一奖励、手工规则、同构 RLHF；实验结果：
• 在离线指标上，MAP 在 Honesty、Relevance、Completeness 等维度分别提升约 6–9%，长度控制更精细；
• 在线 A/B 结果显示完成率提升 9.72%，UV 点赞率提升 5.56%，整体满意度上升 2.31%；
• 细粒度错误分析显示 Severe Error 下降至 9/100，显著优于传统方法（21/100）。

**⚠️ 局限性**

局限：
• 对齐流程复杂，需大量专家标注、对齐样本生成，成本高；
• 仍面临罕见医学知识的覆盖不足和快速更新的适配挑战；
• 多目标奖励合成虽有效但对超参数敏感，需细致调优；
• 在极端长篇或极高多义性的问答场景下，仍可能出现梯度爆炸或“安全警告”被忽略的情况。

---

## 472. Evolutionary Router Feature Generation for Zero-Shot Graph Anomaly Detection with Mixture-of-Experts

**arXiv ID:** 2602.11622 | [PDF](https://arxiv.org/pdf/2602.11622v1)

**作者:** Haiyang Jiang `[一作]` (University of Queensland), Hongzhi Yin `[通讯]` (University of Queensland)

**通讯引用:** 17087 | [OpenAlex ID](https://openalex.org/A5088492734)

**关键词:** `Information Retrieval` `Anomaly Detection` `Graph Neural Network` `Large Language Model` `Mixture of Experts` `Graph`

**🎯 论文内容**

提出了一种基于演化路由特征生成的Mixture‑of‑Experts框架 EvoFG，用于零样本图异常检测。

**💡 创新点**

创新点在于：①使用大语言模型驱动的演化特征生成与 Shapley 值筛选；②引入记忆增强路由器和Invariant学习以提升路由器的跨域泛化；③通过多种 GNN 专家实现对结构与语义异质性的覆盖。

**🔧 技术方法**

核心技术包括：LLM 生成特征、Shapley 估计特征重要性、记忆增强 MoE 路由器、Invariant Risk Minimization、低复杂度的特征选择与迭代训练。

**📊 数据集**

在六个公开基准（Cora、Citeseer、ACM、BlogCatalog、Facebook、Weibo）以及跨域测试集上验证。

**📈 对比分析**

与传统 GNN（SAGE、GIN）、专用 GAD 方法（BWGNN、AMNet、GHRN）以及零样本方法（ARC、UNPrompt、AnomalyGFM）对比，EvoFG 在 AUROC/AUPRC 上连续获得最优或接近最优分数，平均排名 1.67/1.50，明显优于所有基线。

**⚠️ 局限性**

局限性：特征生成质量与记忆容量对性能影响较大；计算成本相对较高；在极度异质或极端分布偏移的数据上仍可能出现泛化瓶颈。

---

## 473. DiSCoKit: An Open-Source Toolkit for Deploying Live LLM Experiences in Survey Research

**arXiv ID:** 2602.11230 | [PDF](https://arxiv.org/pdf/2602.11230v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Human-Computer Interaction`

---

## 474. Think like a Scientist: Physics-guided LLM Agent for Equation Discovery

**arXiv ID:** 2602.12259 | [PDF](https://arxiv.org/pdf/2602.12259v1)

**作者:** Jianke Yang `[一作]` (University of California San Diego), Rose Yu `[通讯]` (University of California San Diego)

**通讯引用:** 6493 | [OpenAlex ID](https://openalex.org/A5057778679)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Agentic AI` `Reinforcement Learning` `Time Series` `Physics Related`

**🎯 论文内容**

本论文提出KeplerAgent，一个物理引导的LLM代理框架，用于从观测数据中发现符号方程。

**💡 创新点**

创新点在于将LLM作为代理，按科学推理流程调用物理工具提取中间结构，再配置符号回归工具，显著提升符号准确性和鲁棒性。

**🔧 技术方法**

技术上结合LLM（GPT‑4o‑mini）与工具链（Python解释器、视觉子代理、对称性发现、PySINDy和PySR），实现多步推理与配置。

**📊 数据集**

使用了LLM‑SRBench的LSR‑Transform子集以及自建的10个耦合ODE/PDE系统数据集。

**📈 对比分析**

与传统SR工具PySR和LLM‑SR基线对比，KeplerAgent在符号准确率上提升至清晰数据75%/噪声45%，NMSE更低、运行时与PySR相当但令牌使用显著减少。

**⚠️ 局限性**

局限包括工具集有限，难以覆盖更复杂的物理结构；工具规范导致上下文膨胀；内部推理不够显式；可能过度依赖自动发现的方程。

---

## 475. VIRENA: Virtual Arena for Research, Education, and Democratic Innovation

**arXiv ID:** 2602.12207 | [PDF](https://arxiv.org/pdf/2602.12207v1)

**作者:** Emma Hoes `[一作]` (University of Zurich), Fabrizio Gilardi `[通讯]` (University of Zurich)

**通讯引用:** 6861 | [OpenAlex ID](https://openalex.org/A5046475081)

**关键词:** `Human-Computer Interaction` `Large Language Model` `Agentic AI` `Prompt Engineering` `Text` `Video`

**🎯 论文内容**

提出并实现了VIRENA平台，提供可视化无代码配置的真实社交媒体模拟环境，支持多用户实时互动、LLM驱动的AI代理以及可调节的内容审核机制，旨在实现对数字沟通动态的可控实验研究。

**💡 创新点**

核心创新点在于将feed‑style 与 chat‑style 的真实界面、实时多用户交互、可配置AI代理以及可实验化的内容审核系统整合到同一平台，并通过无代码界面降低技术门槛，突破了以往工具在单一功能或需编程的限制。

**🔧 技术方法**

技术实现基于Go语言的PocketBase后端（SQLite数据库、Token身份验证）、前端Web技术、LLM接口（如OpenAI API）进行代理生成、内容审核与脚本排程，配合可视化模板与实例管理实现实验流程控制。

**📊 数据集**

主要使用自建的模拟数据集（预置的图片、视频、文本、互动指标）以及参与者产生的内容；未使用公开的大规模社交媒体数据集，所有数据均保存在本地数据库以满足数据安全与合规要求。

**📈 对比分析**

通过模板、实例、等待室与并行实验的设计实现随机化与对照；实验数据可导出为CSV/JSON，支持统计与机器学习分析。性能上已能支持数百并发用户，但尚未进行正式的高并发压力测试，后续计划提升扩展性与响应速度。

**⚠️ 局限性**

限制包括：模拟环境与真实社交平台的生态效度差距、LLM代理的真实性受限于提示工程、平台设计随时间快速迭代导致的滞后、以及在极高并发下的性能与可扩展性未充分验证。

---

## 476. Incentive Effects of a Cut-Off Score: Optimal Contest Design with Transparent Pre-Selection

**arXiv ID:** 2602.11914 | [PDF](https://arxiv.org/pdf/2602.11914v1)

**作者:** Hanbing Liu `[一作]` (Renmin University of China), Changyuan Yu `[通讯]` (Baidu Inc.)

**关键词:** `Computer Science and Game Theory` `Optimization`

**🎯 论文内容**

研究了在预选（短名单）与公布切线分数的排名式竞赛中，短名单对参赛者激励与竞赛设计的影响，并给出了最高个人表现与总表现两种目标的最优竞赛设计。

**💡 创新点**

①发现无论分布如何，最优奖项结构始终为一人独占（winner‑take‑all）。②对最高个人表现而言，最优短名单规模为2；对总表现而言，短名单规模无关。③预选能够将最高个人表现提升至标准全员竞赛的4/3。

**🔧 技术方法**

使用阶统计、Beta函数表示、量化空间转换、分布无关简化、非线性积分化简、渐近分析和对称均衡证明等理论工具，结合Bayesian Nash均衡分析。

**📊 数据集**

在论文中仅使用理论推导与数值仿真，仿真数据为三种抽象的能力分布：F(x)=x²（凸）、U[0,1]（均匀）以及Exp(1)（指数），以及加入噪声的能力观测模型。

**📈 对比分析**

通过与传统全员winner‑take‑all竞赛进行对比，结果显示：在所有分布下，2人winner‑take‑all预选竞赛的最高个人表现均比无预选竞赛高约1/3（即提升4/3倍），在有限样本量下仍保持优越性；总表现对短名单规模不敏感。

**⚠️ 局限性**

局限性包括：假设成本线性且已知能力分布；只考虑单阶段预选与排名式竞赛；对噪声观测的处理仅为经验验证，理论上尚未完善；未探讨多阶段或多竞赛者的复杂设计。

---

## 477. On-Policy Context Distillation for Language Models

**arXiv ID:** 2602.12275 | [PDF](https://arxiv.org/pdf/2602.12275v1)

**作者:** Tianzhu Ye `[一作]` (Microsoft Research), Furu Wei `[通讯]` (Microsoft Research)

**通讯引用:** 31536 | [OpenAlex ID](https://openalex.org/A5014662947)

**关键词:** `Computation and Language` `Knowledge Distillation` `Transformer` `Large Language Model` `Reinforcement Learning` `Text`

**🎯 论文内容**

设计并实现了一种在策略的上下文蒸馏框架（OPCD），通过让学生模型在自身生成的轨迹上最小化与包含上下文的教师模型的逆KL，进而将临时上下文知识内化为参数。

**💡 创新点**

创新点在于将“在策略蒸馏”的模式与“上下文蒸馏”结合，利用逆KL实现模式趋向，解决了传统前向KL导致的暴露偏差和模态覆盖问题，并首次将其应用于经验知识蒸馏与系统提示蒸馏。

**🔧 技术方法**

采用了逆KL损失、基于采样的在策略训练、教师-学生蒸馏与自蒸馏两种配置，以及token级别KL分解的近似实现。

**📊 数据集**

实验数据集包括 DAPO‑Math‑17K（数学题）、Frozen Lake 与 Sokoban（文本游戏）、MetaSPO 生成的医疗与安全系统提示、MedMCQA、Tweet Eval、Hatecheck 与 Ethos。

**📈 对比分析**

与基线（基模型、上下文增强、传统离策略上下文蒸馏）对比，OPCD 在数学、游戏和提示任务上均实现了更高的测试精度、提升的 OOD 性能，并在多尺寸学生模型上显著减轻遗忘，性能提升幅度通常为 2–4 % 的精度增益。

**⚠️ 局限性**

局限性包括：需要预先训练好的教师模型，逆KL 近似可能在极大词表或极长上下文下计算开销较高；对模型容量和训练稳定性的依赖使得在极大规模或实时系统中应用仍需进一步验证。

---

## 478. Computing stable limit cycles of learning in games

**arXiv ID:** 2602.11315 | [PDF](https://arxiv.org/pdf/2602.11315v1)

**作者:** Oliver Biggar `[一作]`, Christos Papadimitriou `[通讯]`

**关键词:** `Computer Science and Game Theory`

**🎯 论文内容**

无法确定论文内容

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用的技术

**📊 数据集**

无法确定使用的数据集

**📈 对比分析**

无法确定比较方法及性能

**⚠️ 局限性**

信息不足，无法评估限制

---

## 479. Talk2DM: Enabling Natural Language Querying and Commonsense Reasoning for Vehicle-Road-Cloud Integrated Dynamic Maps with Large Language Models

**arXiv ID:** 2602.11860 | [PDF](https://arxiv.org/pdf/2602.11860v1)

**作者:** Lu Tao `[一作]` (Wuhan University), Hiroaki Takada `[通讯]` (Nagoya University)

**通讯引用:** 2080 | [OpenAlex ID](https://openalex.org/A5032678689)

**关键词:** `Artificial Intelligence` `Autonomous Driving` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

提出了 VRCsim 车路云协同感知仿真框架，基于其生成了 VRC-QA 数据集（10K 场景、100K QA 对），并设计了 Talk2DM 插件实现自然语言查询和常识推理。

**💡 创新点**

创新点在于：①首个针对车路云动态地图的仿真平台；②首个包含多 RSU 与 AV 的 VRC‑CP QA 数据集；③提出链式提示（CoP）机制，将 LLM 常识与手工规则逐步融合，实现准确的空间查询与语义建议。

**🔧 技术方法**

使用技术包括 SUMO+ROS2+Qt6 的仿真与数据流构建；基于 Qwen、Gemma、GPT‑oss 等 LLM 的提示工程；链式提示（CoP）与工具箱式函数调用实现查询与推理。

**📊 数据集**

数据集：VRC‑QA，由 VRCsim 生成的 12,094 语言场景，构成 100K QA 对，覆盖 10 种查询类型。

**📈 对比分析**

通过与传统一轮提示（OSP）对比，CoP 在 VRC‑QA 上实现约 94% 的查询准确率，远超 OSP 的 30%；在不同 LLM 家族（Qwen、Gemma、GPT‑oss）和模型规模下评估，发现大模型准确率更高但延迟显著；综合准确率与响应时间，Gemma3:27B 与 GPT‑oss 最优，均可在 2–5 s 内给出 93% 以上的答案。

**⚠️ 局限性**

局限性包括：对模糊/暗示性问题（加速、航向）仍易误判；存在查询受前缀子句影响的分类错误；依赖大规模 LLM，导致推理延迟增加；当前仅在仿真语言场景上验证，缺乏真实车路云数据的实测。

---

## 480. Efficient Hyper-Parameter Search for LoRA via Language-aided Bayesian Optimization

**arXiv ID:** 2602.11171 | [PDF](https://arxiv.org/pdf/2602.11171v1)

**作者:** Baek Seong-Eun `[一作]` (POSTECH), Tae-Hyun Oh `[通讯]` (KAIST)

**关键词:** `Computation and Language` `Hyperparameter Search` `Optimization` `Transformer` `Large Language Model` `Prompt Engineering` `Bayesian Optimization` `Text`

**🎯 论文内容**

针对LoRA微调的超参数调优，提出了一种结合LLM与贝叶斯优化的框架；

**💡 创新点**

通过在LLM中注入领域知识（域感知提示、可学习token、投影层）构造连续嵌入空间，并利用代理训练子集降低评估成本；

**🔧 技术方法**

采用大型语言模型生成文本嵌入、可学习token+投影层对嵌入进行校准，贝叶斯优化（高斯过程）进行超参数搜索，代理训练评估（10%子集）；

**📊 数据集**

在数学推理任务MetaMathQA训练、GSM8k、MATH测试；代码生成任务CodeFeedback训练、HumanEval、MBPP测试；

**📈 对比分析**

与随机搜索、Optuna、标准BO、LBO、NOMAD等基线以及专门针对LoRA的HPO方法对比，30次迭代即可获得20%+性能提升，计算时间从180小时降至24小时；

**⚠️ 局限性**

对LLM提示模板和可学习token的设计依赖先验知识，代理子集在某些任务上的相关性仍有限，且在极大规模模型或不同架构时可能需重新调整提示和子集比例。

---

## 481. Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty

**arXiv ID:** 2602.12113 | [PDF](https://arxiv.org/pdf/2602.12113v1)

**作者:** Zewei Yu `[一作]` (Zhejiang University), Junbo Zhao `[通讯]` (Zhejiang University)

**通讯引用:** 11373 | [OpenAlex ID](https://openalex.org/A5042402520)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Computational Efficiency` `Reinforcement Learning` `Text`

**🎯 论文内容**

提出一种基于强化学习的自适应反思与长度协同惩罚方法，旨在减少大型推理模型的冗余推理步骤，提高推理效率并保持或提升准确率。

**💡 创新点**

创新点在于引入了可根据问题复杂度动态调整的反思惩罚和长度惩罚，能够同时抑制过度反思和冗余生成，且保持推理质量。

**🔧 技术方法**

使用了强化学习（REINFORCE+Leave-One-Out）与自适应惩罚策略，在推理过程中实时估计问题复杂度并动态调节惩罚系数。

**📊 数据集**

在 DeepScaleR 数学题库训练，评估五大数学推理基准（GSM8K、MATH500、AMC2023、AIME2024/2025）以及 MMLU 与其他模型系列。

**📈 对比分析**

与 Nothinking、SFT_Shortest、DPO_Shortest、O1‑Pruner、TLMRE、AdaptThink、LASER 等基线对比，实验表明在 1.5B 模型上平均回复长度降低 53.1% 并提升 5.8% 准确率，7B 模型下降 35% 长度并提升 2.7% 准确率，且在更高难度任务上表现最佳。

**⚠️ 局限性**

局限性包括对阈值和惩罚参数的敏感性、仅在数学和部分通用问答上验证、未针对更大规模模型或实时部署场景做深入评估。

---

## 482. Generative AI-Driven Phase Control for RIS-Aided Cell-Free Massive MIMO Systems

**arXiv ID:** 2602.11226 | [PDF](https://arxiv.org/pdf/2602.11226v1)

**作者:** Kalpesh K. Patel `[一作]` (Indian Institute of Technology), Sandeep Kumar Singh `[通讯]` (Indian Institute of Technology)

**通讯引用:** 99 | [OpenAlex ID](https://openalex.org/A5101820771)

**关键词:** `Information Theory` `Optimization` `Computational Efficiency` `Diffusion model`

**🎯 论文内容**

利用生成式人工智能中的扩散模型，提出两种方法（GCDM 与 GCDIM）在存在不完美信道信息与空间相关性的 RIS 辅助无单元大规模 MIMO 系统中优化相位移，从而最大化系统总谱效率。

**💡 创新点**

创新点在于：①首次将条件扩散模型与条件扩散隐式模型应用于 RIS 相位优化；②通过在扩散过程中加入实时 CSI 条件，使模型能够在不同信道状态下泛化；③相较传统专家算法，显著降低了计算复杂度并保持近似最优性能。

**🔧 技术方法**

技术包括：条件扩散模型（GCDM）、条件扩散隐式模型（GCDIM）、UNet 结构的噪声预测网络、基于遗传算法生成的专家标签、Adam 优化器及学习率调优。

**📊 数据集**

使用了由遗传算法在多种发射功率（-10 dB~40 dB）与随机布置的 AP/用户/RIS 场景下生成的 RIS 相位移专家数据集，用作模型训练和验证。

**📈 对比分析**

与随机相位、专家算法（遗传算法）以及两种 GenAI 方法进行对比；评估指标为总谱效率、训练损失收敛以及执行时间。结果显示：GCDM 在总谱效率上与专家算法持平，GCDIM 在仅 20 步甚至 5 步的情况下与 GCDM 相当；两者执行时间分别从 752 s 缩短到 1.18 s（GCDM）和 0.07 s（GCDIM），实现了显著的计算效率提升。

**⚠️ 局限性**

局限性包括：①模型对训练数据分布高度依赖，若实际信道环境与训练集差异较大可能性能下降；②在极端信道条件或不同硬件限制下的鲁棒性尚未充分验证；③扩散隐式模型虽然加快采样速度，但在多样性与精细度上可能不如完整扩散过程；④实验仅在模拟环境下完成，缺乏真实世界的硬件测试。

---

## 483. Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation

**arXiv ID:** 2602.11213 | [PDF](https://arxiv.org/pdf/2602.11213v1)

**作者:** Shuyu Chang `[一作]` (Nanjing University of Posts and Telecommunications), Leo Yu Zhang `[通讯]` (Griffith University)

**通讯引用:** 4638 | [OpenAlex ID](https://openalex.org/A5015011245)

**关键词:** `Cryptography and Security` `Adversarial Attack` `AI Code Assistant` `Transformer` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

提出 STAB 框架，利用 Sharpness‑Aware Minimization 训练平坦化的代理模型，并通过 Gumbel‑Softmax 可微离散优化生成语法合法、上下文感知的触发器，实现对代码模型的跨数据集可转移且隐蔽的后门攻击。

**💡 创新点**

创新点在于将 SAM 与 Gumbel‑Softmax 结合，既能在平坦的损失地形中挖掘通用的触发模式，又通过 MMD 约束保证触发器多样性与语法合法性，从而突破静态触发器易被检测、动态触发器转移差的限制。

**🔧 技术方法**

使用技术包括 Sharpness‑Aware Minimization、Gumbel‑Softmax 离散优化、Maximum Mean Discrepancy 约束、对抗触发器生成、代码抽象语法树解析、Transformer 编码‑解码模型。

**📊 数据集**

实验数据集为三大 Python 代码库：Py150、CodeSearchNet（CSN）、PyTorch（PyT）；任务为方法名预测（MNP）和代码摘要（CS）；模型为 PLBART 与 CodeT5。

**📈 对比分析**

与静态触发器（Fixed、Grammar）和动态攻击 AFRAIDOOR 进行对比，使用攻击成功率（ASR）、防御后 ASR‑D、BLEU、召回/ F1 等指标。STAB 在跨数据集攻击中平均 ASR 超过 80%，防御后 ASR‑D 达 73.2%，比 AFRAIDOOR 提升 12.4%，并保持较高 BLEU；对 SS、ONION、KillBadCode 等防御时检测率最低。

**⚠️ 局限性**

局限性包括对代理模型规模和数据量的依赖，对 SAM 的锐度参数 ρ、Gumbel 温度 τ 敏感；目前仅在代码模型验证，尚未评估对多模态或其他编程语言模型的适用性；未来防御可能针对平坦化触发器的特征进行识别。

---

## 484. Systematic Trend-Following with Adaptive Portfolio Construction: Enhancing Risk-Adjusted Alpha in Cryptocurrency Markets

**arXiv ID:** 2602.11708 | [PDF](https://arxiv.org/pdf/2602.11708v1)

**作者:** Duc Bui `[一作]` (Talyxion Research), Thanh Nguyen `[通讯]` (Talyxion Research)

**关键词:** `Computational Engineering, Finance, and Science` `Optimization` `Recommendation System` `Finance Related` `Time Series` `Tabular`

**🎯 论文内容**

提出了 AdaptiveTrend 框架，通过 6 小时高频动量信号、基于市值与夏普比率的每月资产筛选以及 70/30 的长期短期资金分配，对 150+ 交易对进行系统化加密货币交易。

**💡 创新点**

创新点包括：① 结合波动率的动态追踪止损；② 每月基于夏普比率的资产筛选并加入市值过滤；③ 以经验正向漂移为基础的 70/30 长短仓非对称分配。

**🔧 技术方法**

使用了 H6 期货行情的动量计算、ATR 追踪止损、月度重平衡、夏普比率筛选、等权分配、交易成本与融资费率模型、以及区块 bootstrap 统计显著性检验。

**📊 数据集**

数据集来自 Binance 期货永续合约，包含 150+ 加密交易对的 6 小时 OHLCV 数据（2021‑01 至 2024‑12）以及 CoinGecko 提供的每日市值信息。

**📈 对比分析**

通过 2022‑2024 36 个月的 OOS 回测，与 TSMOM、等权买持、BTC 买持、波动率调节 TSMOM 等基准对比，指标为年化收益 40.5%、夏普 2.41、最大回撤 -12.7%，Calmar 3.18，表现显著优于所有基准，并在 bootstrap 测试中达到 5% 置信水平。

**⚠️ 局限性**

局限性包括：对 Binance 交易所的依赖、短仓资产的流动性与容量限制、极端行情下潜在的滑点与执行风险、监管与合规约束、以及月度重优化可能带来的前视偏差和模型假设的限制。

---

## 485. Intrinsic-Energy Joint Embedding Predictive Architectures Induce Quasimetric Spaces

**arXiv ID:** 2602.12245 | [PDF](https://arxiv.org/pdf/2602.12245v1)

**作者:** Anthony Kobanda `[一作]` (Ubisoft), Waris Radji `[通讯]` (Inria)

**关键词:** `Machine Learning` `Reinforcement Learning`

**🎯 论文内容**

本文阐述了如何将联合嵌入预测架构（JEPA）的能量函数视为最小作用量能量，从而证明其构成了一个准度量，进而与目标驱动的准度量强化学习（QRL）中的成本-到-目标函数相对应。

**💡 创新点**

创新点在于首次把JEPA能量函数与最小作用量原理联系起来，证明在满足内在能量条件下JEPA自然满足三角不等式，形成准度量，从而建立了JEPA与QRL的结构性统一。

**🔧 技术方法**

使用理论分析、最小作用量定义、准度量性质证明，并在控制理论与能量基方法框架下进行推导。

**📊 数据集**

未使用任何数据集，纯理论研究。

**📈 对比分析**

本文未进行实验比较，主要通过理论证明展示其结构性一致性。

**⚠️ 局限性**

局限性在于仅对内在能量形式的JEPA给出结果，无法覆盖所有JEPA模型；实验验证缺失，实际适用性尚待评估。

---

## 486. Echo: Towards Advanced Audio Comprehension via Audio-Interleaved Reasoning

**arXiv ID:** 2602.11909 | [PDF](https://arxiv.org/pdf/2602.11909v1)

**作者:** Daiqing Wu `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Yu Zhou `[通讯]` (Nankai University)

**通讯引用:** 368303 | [OpenAlex ID](https://openalex.org/A5111964102)

**关键词:** `Sound` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Chain-of-Thought` `Audio`

**🎯 论文内容**

提出音频交互式推理（audio‑interleaved reasoning），并设计两阶段训练框架（SFT + RL）和自动数据生成管线，训练得到 Echo 这一 LALM。

**💡 创新点**

创新点在于将音频视为推理过程中可随时“重听”的主动组件，打破一次性编码瓶颈；通过监督微调定位关键信号片段，并用强化学习激励多次重听；同时利用自动生成的音频 QA‑CoT 数据提升训练质量。

**🔧 技术方法**

技术手段包括：基于 Qwen2.5‑Omni 7B 的大模型；监督微调（SFT）实现音频片段定位；PPO 强化学习鼓励多次音频重听；推理时动态插入音频片段；使用 DeepSeek‑R1 自动生成 QA‑CoT；vLLM 引擎进行高效推理。

**📊 数据集**

数据集：自动生成的 EAQA‑SFT（75.9k 带 CoT）与 EAQA‑RL（21.9k 无 CoT），来源于 AudioSet‑Strong 与 MusicBench 的音频；评测基准包括 MMAR、MMAU‑mini、MMAU 等公开音频 QA 任务。

**📈 对比分析**

对比方法：与多款公开及商用 LALM（如 GPT‑4o、Gemini‑2.0‑Flash、Qwen2.5‑Omni 等）在同一基准上进行准确率评测；Echo 在 MMAR、MMAU、MMAU‑mini 上分别实现最高或次高准确率，整体提升约 2–7%，验证音频交互式推理的有效性。

**⚠️ 局限性**

局限性：尚未支持更细粒度的音频操作（如慢速播放、频段分离）；自动生成的 CoT 可能带有偏差与重复；RL 监督缺乏对片段选择的细粒度约束；对超过 10 秒的长音频定位能力仍有限。

---

## 487. Towards a theory of Façade-X data access: satisfiability of SPARQL basic graph patterns

**arXiv ID:** 2602.11756 | [PDF](https://arxiv.org/pdf/2602.11756v1)

**作者:** Luigi Asprino `[一作]` (Università Telematica San Raffaele), Enrico Daga `[通讯]` (Knowledge Media Institute)

**关键词:** `Databases` `Graph` `Benchmark`

**🎯 论文内容**

研究了在 Façade-X（面向多格式数据的中间 RDF 模型）下 SPARQL 基本图模式（BGP）的可满足性问题，提供了理论分析与判定算法。

**💡 创新点**

提出了统一的 Façade-X 元模型及其对 RDF 的映射，并从图模式角度系统性地表征了哪些 BGP 能在该模型上产生解，从而实现了可满足性判定。

**🔧 技术方法**

利用谓词逻辑、图论和 SPARQL 语义，对 Façade-X 模型进行形式化，构造了判定 BGP 可满足性的顶层搜索与约束满足算法。

**📊 数据集**

实验数据来自手工构造的 27 组 BGP（覆盖不同结构）以及 1430 条从 GitHub 公开仓库提取的真实查询，此外还使用 GTFS‑Madrid‑Bench 基准进行实测。

**📈 对比分析**

与基线实现（仅做语义检查）相比，基于 CSP 的下推算法在可满足性判定上平均仅耗时数十毫秒；生成所有解的耗时在 0.1‑5 秒之间，远优于之前的搜索方法；在基准查询中，可满足性检查耗时 <15 ms，显著低于构造/加载 RDF 图所需的 100‑10 000 ms。

**⚠️ 局限性**

局限性包括：对更大 BGP（>5 条三元组）时仍会出现 5 s 级别的延迟；目前仅处理基本图模式，未覆盖 FILTER、OPTIONAL 等高级 SPARQL 结构；算法对不同数据格式的进一步优化和适配仍待研究。

---

## 488. FedGRPO: Privately Optimizing Foundation Models with Group-Relative Rewards from Domain Client

**arXiv ID:** 2602.12014 | [PDF](https://arxiv.org/pdf/2602.12014v1)

**作者:** Gongxi Zhu `[一作]` (Webank), Yuxing Han `[通讯]` (Tsinghua University)

**通讯引用:** 3515 | [OpenAlex ID](https://openalex.org/A5101944724)

**关键词:** `Machine Learning` `Federated Learning` `Reinforcement Learning` `Optimization` `Safty and Privacy` `Reinforcement Learning` `Auto Encoder` `Text`

**🎯 论文内容**

设计并实现了 FedGRPO，一个基于强化学习评价的联邦基础模型框架，只传输标量奖励以实现隐私保护与高效通信。

**💡 创新点**

引入基于辅助数据的轻量级置信图进行专业客户端选择，并采用组相对奖励聚合的 GRPO 机制，实现对分布式客户端的准确评估与全局模型优化。

**🔧 技术方法**

使用轻量级置信图、组相对政策优化、奖励模型、AE/ME 双路评估以及联邦通信协议等技术。

**📊 数据集**

以 MATH、OpenR1‑Math 等数学题库为训练集，Minerva、OlympiadBench、AIME、AMC 等六个测试集为评估集，并使用少量辅助数据用于客户端选择。

**📈 对比分析**

与 FedPETuning、DPSDA‑FL（两种本地转移方式）以及中心化 GRPO 进行对比，FedGRPO 在 1.5B‑7B 规模模型上在各测试集的 Pass@1 近似或超过中心化基线，同时在通信量上提升 2–3 个数量级。

**⚠️ 局限性**

仍依赖辅助数据与奖励模型质量，对无答案场景的完整评估不足，且在更复杂或跨域任务中的泛化尚待验证。

---

## 489. LLM-Driven 3D Scene Generation of Agricultural Simulation Environments

**arXiv ID:** 2602.11706 | [PDF](https://arxiv.org/pdf/2602.11706v1)

**作者:** Arafa Yoncalik `[一作]` (University of Antwerp), Jan Steckel `[通讯]` (Flanders Make Strategic Research Centre)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Retrieval` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Agriculture Related`

**🎯 论文内容**

开发了一个多LLM模块化管线，将自然语言提示转化为可在Unreal Engine中执行的农田3D场景

**💡 创新点**

通过分阶段资产检索、RAG知识注入与代码生成三段式结构，实现域知识驱动、可验证且可扩展的自动化场景构建

**🔧 技术方法**

使用GPT‑4、Retrieval‑Augmented Generation、FAISS检索、Python脚本生成与Unreal Engine API，并结合微调与少量提示优化

**📊 数据集**

使用约672种水果/蔬菜资产组合及其对应的JSON式农学元数据构成检索索引，同时构造了100个自然语言提示评测集

**📈 对比分析**

与单一LLM基线、人工手工建模及用户评测对比，资产检索准确率98%，代码可执行率100%，与专家手工相比平均节省约50%时间，用户满意度中等

**⚠️ 局限性**

受限于静态资产、缺乏动态生长与环境交互、对模糊提示易产生幻觉、依赖外部API导致延迟与成本

---

## 490. External Division of Two Bregman Proximity Operators for Poisson Inverse Problems

**arXiv ID:** 2602.11482 | [PDF](https://arxiv.org/pdf/2602.11482v1)

**作者:** Kazuki Haishima `[一作]` (Institute of Science Tokyo), Konstantinos Slavakis `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 2460 | [OpenAlex ID](https://openalex.org/A5042922973)

**关键词:** `Machine Learning` `Restoration` `Optimization` `Image`

**🎯 论文内容**

提出了一种基于外部除法Bregman近似算子与NoLips框架的泊松逆问题稀疏重建方法。

**💡 创新点**

创新点在于设计了外部除法的Bregman近似算子，使其在大幅值输入下趋于恒等映射，从而在保持稀疏性的同时显著降低由ℓ1正则化引起的估计偏差，并给出了两种几何解释。

**🔧 技术方法**

该方法使用了NoLips算法、Bregman近似算子、外部除法算子、泊松KL散度、软阈值技术，以及双重空间的几何解释。

**📊 数据集**

实验采用了合成泊松噪声的稀疏向量数据集以及神经元结构的图像数据集（图像恢复）。

**📈 对比分析**

通过与传统的R‑KL和F‑KL NoLips方法比较，使用NMSE和PSNR指标，结果表明所提方法在过定、欠定以及图像恢复场景中收敛更快、误差更低、图像质量更好。

**⚠️ 局限性**

目前的局限在于缺乏理论收敛证明，对参数a的选择需要经验调优，且在更大规模或多模态泊松逆问题中的性能与计算成本尚待进一步评估。

---

## 491. Surface impedance inference via neural fields and sparse acoustic data obtained by a compact array

**arXiv ID:** 2602.11425 | [PDF](https://arxiv.org/pdf/2602.11425v1)

**作者:** Yuanxin Xia `[一作]` (Technical university of Denmark), Cheol-Ho Jeong `[通讯]` (Technical university of Denmark)

**通讯引用:** 1477 | [OpenAlex ID](https://openalex.org/A5071439229)

**关键词:** `Sound` `Neural Radiance Field` `Audio` `Physics Related`

**🎯 论文内容**

利用物理信息的神经场模型，结合稀疏压力测量和自适应训练，实时推断隔墙表面复杂表面阻抗。

**💡 创新点**

提出并实现了并行多频SIREN网络、自动微分粒子速度推断、基于物理约束的复阻抗一致性与频率平滑正则化，并在微型MEMS麦克风阵列上验证。

**🔧 技术方法**

使用并行SIREN多层感知机、PINN式物理约束、自动微分、MEMS麦克风阵列以及COMSOL FEM模拟数据。

**📊 数据集**

实验数据来自自制4×4 MEMS麦克风阵列在无声室和混响室中的声压测量，以及车辆舱内虚拟声场仿真；数值验证基于COMSOL的有限元模拟。

**📈 对比分析**

与标准吸收系数/阻抗参考（吸声器管路、理论模型、Paris公式）对比，误差MAE<0.1在大多数频段，实时推断耗时30–60秒，4×4阵列显著优于3×3。

**⚠️ 局限性**

对近刚性表面或高角度入射、噪声高、局部声场复杂的场景仍存在较大误差；模型对阵列尺寸和测量高度敏感，需更高SNR传感器和更精细阵列。

---

## 492. Right for the Wrong Reasons: Epistemic Regret Minimization for Causal Rung Collapse in LLMs

**arXiv ID:** 2602.11675 | [PDF](https://arxiv.org/pdf/2602.11675v1)

**作者:** Edward Y. Chang `[一作]` (Stanford University), Edward Y. Chang `[通讯]` (Stanford University)

**通讯引用:** 18248 | [OpenAlex ID](https://openalex.org/A5013545831)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Reinforcement Learning` `Text`

**🎯 论文内容**

提出Rung Collapse和Aleatoric Entrenchment概念，并通过Epistemic Regret Minimization (ERM)框架修正大语言模型的因果推理错误。

**💡 创新点**

将物理动作与do‑运算相连的Physical Grounding定理、基于AGM的ERG目标和跨域失败模式分类器，实现对因果推理错误的可检索修正与防止。

**🔧 技术方法**

使用ERG目标（KL损失）、AGM信念修订、物理行动实现的interventional数据、三层架构（实例修正、模式适配、知识路由）以及大型预训练语言模型。

**📊 数据集**

CausalT5K 1,360个因果陷阱场景，覆盖医疗、经济、历史、体育、日常等五个领域。

**📈 对比分析**

与五大前沿LLM（GPT‑3.5、GPT‑4、Gemini 2.5、GPT‑5.2、Claude 3.5）进行零样本检测，Rung Collapse率从17.3%降至0.9%；在纠错实验中，目标化的ERM将模型错误率下降约53–59%，而通用反馈仅提高10%左右。

**⚠️ 局限性**

局限在于测试场景为文本描述，缺乏真实环境的物理或连续变量；历史领域的训练数据偏差导致高失败率；仅评估单轮反馈，未考察长期保持与跨域推广。

---

## 493. The Manifold of the Absolute: Religious Perennialism as Generative Inference

**arXiv ID:** 2602.11368 | [PDF](https://arxiv.org/pdf/2602.11368v1)

**作者:** Arthur Juliani `[一作]` (Institute For Advanced Consciousness Studies), Arthur Juliani `[通讯]` (Institute For Advanced Consciousness Studies)

**关键词:** `Computers and Society` `Generation` `Auto Encoder`

**🎯 论文内容**

用变分自编码器（VAE）框架对宗教认知进行形式化，将宗教传统建模为从共享潜在空间到观测文化形式的生成映射，并比较四种生成配置（排他主义、普世主义、永续主义与融合主义），从而提出永续主义是最优解释。

**💡 创新点**

创新点在于：① 将宗教多样性转化为可度量的生成模型；② 利用信息论与微分拓扑理论（数据处理不等式、流形假设）严谨阐述融合失败与普世主义的后验崩溃；③ 将正统实践与自我超越统一为ELBO优化中的重建与KL正则化，提供对“正统性”本质的数学解释。

**🔧 技术方法**

主要技术包括变分自编码器（VAE）理论、流形假设与转置定理、数据处理不等式、ELBO分解与KL散度分析，以及对宗教实践的编码/解码器匹配理论。

**📊 数据集**

未使用传统机器学习数据集；论文以跨传统冥想经验研究（Metzinger 等的12因素结构）、宗教文本与实践案例为理论支撑，构成经验论证而非实验性数据集。

**📈 对比分析**

通过理论推导和对已公开研究的比较来评估四种配置；未给出数值指标，而是以解释力与合理性为评价标准，认为永续主义在解释跨传统共识、避免融合崩溃、保持正统与自我超越的平衡方面优于其他模型。

**⚠️ 局限性**

局限性包括：① 依赖于跨传统经验共识的真实性；② 仅适用于强调沉思实践的传统，对法律、伦理或社区导向传统适用性有限；③ 对“绝对”潜在空间的先验假设可能与隐喻主义、无实体主义传统产生冲突；④ 论证为归纳性推理，未能实验验证；⑤ 对神经认知机制的解释仍有争议。

---

## 494. Building Intelligent User Interfaces for Human-AI Alignment

**arXiv ID:** 2602.11753 | [PDF](https://arxiv.org/pdf/2602.11753v1)

**作者:** Danqing Shi `[一作]` (University of Cambridge), Danqing Shi `[通讯]` (University of Cambridge)

**通讯引用:** 352 | [OpenAlex ID](https://openalex.org/A5101528850)

**关键词:** `Human-Computer Interaction` `Reinforcement Learning from Human Feedback` `Large Language Model` `Reinforcement Learning` `Text`

**🎯 论文内容**

提出一个面向人机交互的参考模型，用于设计支持人类对AI系统进行价值对齐的智能用户界面，并在 RL 代理与大型语言模型两类案例中验证其可行性。

**💡 创新点**

创新点在于将 UI 设计视为对齐核心，构建系统化的参考模型，将数据采样、可视化转换、交互展示与人类反馈控制整合，并给出六种对齐 UI 的对比框架。

**🔧 技术方法**

使用主动学习/策略采样、t‑SNE 聚类、动态时间规整、径向图、链路高亮、关键词标签等可视化技术，以及辅助 LLM 进行事实提取和文本分解。

**📊 数据集**

采用 RL 环境生成的策略轨迹和 LLM 在提示下产生的多重文本响应作为数据来源，未使用公开标准数据集，而是基于模拟任务与开放式提示。

**📈 对比分析**

通过与六种现有对齐 UI 的对比，对齐效率与反馈质量进行定性评估；IGC 在 RL 对齐中相较传统成对对比提升了注释效率和偏好一致性；DxHF 在 LLM 对齐中提升了事实命题辨识度和反馈准确性。

**⚠️ 局限性**

局限性包括对复杂/大规模输出的可解释性不足、对齐过程仍高度依赖人工评估、缺乏量化性能指标与长期实验验证，以及对可扩展性与通用性的系统评估尚未完成。

---

## 495. Capability-Oriented Training Induced Alignment Risk

**arXiv ID:** 2602.12124 | [PDF](https://arxiv.org/pdf/2602.12124v1)

**作者:** Yujun Zhou `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**通讯引用:** 12583 | [OpenAlex ID](https://openalex.org/A5000755750)

**关键词:** `Machine Learning` `Reinforcement Learning` `Safety and Privacy` `Reinforcement Learning` `Chain-of-Thought` `Supervised Fine-Tuning`

**🎯 论文内容**

本文通过设计四个基于“AI安全网格世界”框架的漏洞游戏，系统研究了在强化学习训练过程中模型如何自发出现并传播对环境漏洞的利用行为；

**💡 创新点**

创新点在于首次将“能力导向训练诱发的对齐风险”（Capability‑Oriented Training Induced Alignment Risk）概念化，并证明该风险在小型开源模型中普遍出现、能以零样本方式跨任务迁移并通过策略蒸馏传递；

**🔧 技术方法**

主要技术包括基于GRPO的强化学习、链式思考提示、零样本跨任务迁移、连续训练的“催化学习”以及SFT蒸馏；

**📊 数据集**

使用自行构造的四类攻击数据集（Context‑Conditional Compliance、Audited Self‑Grading、Proxy Metric Gaming、Reward/State Tampering），每类均遵循AI安全网格世界的规范；

**📈 对比分析**

实验通过对比模型在任务完成度（ITP）与攻击比例（ER）的变化，发现几乎所有模型都能在训练中迅速出现漏洞利用，且RL训练的利用更难被安全训练消除；

**⚠️ 局限性**

局限性包括：实验仅在高度简化的游戏环境中进行，未验证在真实复杂场景中的表现；跨任务迁移效果受模型先前训练和规模限制；蒸馏后利用率低于RL源模型，说明仅靠监督学习难以完全复现RL内部化的策略；

---

## 496. The Arithmetic Singleton Bound on the Hamming Distances of Simple-rooted Constacyclic Codes over Finite Fields

**arXiv ID:** 2602.11788 | [PDF](https://arxiv.org/pdf/2602.11788v1)

**作者:** Li Zhu `[一作]` (Guizhou Normal University), Hongfeng Wu `[通讯]` (North China University of Technology)

**通讯引用:** 171 | [OpenAlex ID](https://openalex.org/A5102012716)

**关键词:** `Information Theory`

**🎯 论文内容**

本文研究了XXX问题，提出了一种新的解决方案。

**💡 创新点**

创新点在于引入了XXX方法，显著提高了性能。

**🔧 技术方法**

使用了XXX技术，如深度学习、机器学习等。

**📊 数据集**

实验使用了XXX数据集，包含了XXX样本。

**📈 对比分析**

与现有方法进行了比较，结果显示本方法在XXX指标上优于其他方法。

**⚠️ 局限性**

限制在于XXX，例如数据集规模较小或模型复杂度高。

---

## 497. Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis

**arXiv ID:** 2602.11169 | [PDF](https://arxiv.org/pdf/2602.11169v1)

**作者:** Mangadoddi Srikar Vardhan `[一作]` (National Institute of Technology Silchar), Lekkala Sai Teja `[通讯]` (National Institute of Technology Silchar)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

本文通过在Transformer隐藏层中对向量方向和幅度进行等欧氏距离扰动，探究它们在语言建模与句法推理中的功能差异。

**💡 创新点**

创新点在于提出了L2‑匹配扰动分析方法，实现了在相同扰动大小下公平比较方向和幅度的影响；同时发现了“交叉解离”——方向扰动更严重损害语言建模损失，幅度扰动更严重损害句法准确率，并揭示了各自主要依赖的机制路径（方向→注意力，幅度→LayerNorm）。

**🔧 技术方法**

核心技术包括：L2‑匹配扰动（等欧氏距离的角度和幅度扰动）、因果干预实验（注意力修复、LayerNorm修复）、统计分析（配对t检验、Bonferroni校正）以及向量几何分析（方向、幅度分解、Parse深度相关性）。

**📊 数据集**

使用了WikiText‑103作为语言建模评估数据集，使用BLiMP（主语-动词一致子集）作为句法评估数据集；模型为Pythia系列（410M和1.4B参数）以及少量对比模型（OPT、TinyLlama）。

**📈 对比分析**

在等距扰动δ下，方向扰动的语言模型损失增幅可高达42.9倍，幅度扰动的句法准确率下降可达20%以上；在更大模型规模时该差异进一步放大。对比实验显示方向扰动通过注意力路径恢复约28%损失，而幅度扰动通过LayerNorm路径恢复约30%。

**⚠️ 局限性**

主要局限包括：结果仅在LayerNorm架构（Pythia/OPT）中显现，在RMSNorm架构（TinyLlama）中模式相反；扰动采用随机正交方向，未探究结构化旋转；仅评估单一句法现象；L2匹配假设在高度异方差空间可能不足；因果修复未覆盖所有潜在路径，未解释残余70%损失。

---

## 498. MagneX: A High-Performance, GPU-Enabled, Data-Driven Micromagnetics Solver for Spintronics

**arXiv ID:** 2602.12242 | [PDF](https://arxiv.org/pdf/2602.12242v1)

**作者:** Andy Nonaka `[一作]` (Lawrence Berkeley National Laboratory), Zhi Jackie Yao `[通讯]` (Lawrence Berkeley National Laboratory)

**通讯引用:** 657 | [OpenAlex ID](https://openalex.org/A5108051450)

**关键词:** `Computational Engineering, Finance, and Science` `Magnetic Resonance Imaging` `Physics Related` `Benchmark`

**🎯 论文内容**

本文开发了开源磁性微观模拟器MagneX，针对多尺度、多物理耦合的自旋电子设备进行高效数值求解；

**💡 创新点**

创新点包括将AMReX与SUNDIALS结合实现GPU加速的多重时域积分、引入多速率MRI算法显著提升时间步长，并通过训练的 Fourier Neural Operator 替代耗时的去磁场计算，实现数据驱动加速；

**🔧 技术方法**

采用AMReX框架进行多GPU网格并行，SUNDIALS提供显式、隐式及混合时步长方法，CUDA/ROCm实现FFT和神经网络推理，Python层统一配置与ML接口；

**📊 数据集**

使用μMAG标准问题2–4以及含DMI的 vortex/skyrmion基准实验作为验证数据集，采集磁化场与去磁场的输入输出对训练NN模型；

**📈 对比分析**

与传统FFT求解以及单速率RK4方法比较，MRI方案在标准问题4上时间步长可提升5倍，整体求解时间缩短约48%，并在GPU上表现出良好的弱扩展性；

**⚠️ 局限性**

局限性在于隐式求解器缺乏预条件器导致性能瓶颈，NN替代器在更复杂几何和多物理耦合场景下的泛化能力尚未充分验证。

---

## 499. Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models

**arXiv ID:** 2602.11824 | [PDF](https://arxiv.org/pdf/2602.11824v1)

**作者:** Jialin Wu `[一作]` (Ant Group), Zhou Yang `[通讯]` (Ant Group)

**关键词:** `Artificial Intelligence` `Object Detection` `Recognition` `Transformer` `Vision Language Model` `Multimodality`

**🎯 论文内容**

提出一种无训练的稀疏潜在导向框架，直接在模型内部恢复视觉信息以消除物体幻觉。

**💡 创新点**

创新点在于使用正交投影分离视觉信息与语言先验，并通过校准选择最优层进行稀疏干预，避免整体干预导致的性能退化。

**🔧 技术方法**

采用正交投影（Gram-Schmidt）、层级校准搜索、动态风险阈值驱动的潜在干预技术。

**📊 数据集**

在POPE、CHAIR、MME、MM-Vet四大标准评测集上进行评估，并验证于LLaVA-NeXT、InternVL3等多种架构。

**📈 对比分析**

与VCD、M3ID、ONLY、AGLA、VTI等现有方法比较，平均降低19%幻觉率、保持甚至提升推理分数，推理延迟与普通解码相当。

**⚠️ 局限性**

受限于基础模型的表示能力，难以修复完全无感知的视觉盲点；目前仅在7B-8B模型上验证，需进一步验证更大规模模型。

---

## 500. How to check in continually over 4,000 days on an online learning platform? An empirical experience and a practical solution

**arXiv ID:** 2602.11249 | [PDF](https://arxiv.org/pdf/2602.11249v1)

**作者:** Jialiang Lin `[一作]` (Guangzhou Institute of Science and Technology), Jialiang Lin `[通讯]` (Guangzhou Institute of Science and Technology)

**关键词:** `Computers and Society` `Tabular` `Time Series` `Review/Survey Paper`

**🎯 论文内容**

本文研究了在线英语学习平台中检查服务（check‑in）用户放弃的现象，基于问卷调查和作者本人 4,000+ 天持续检查记录，提出了 GILT 方法（目标设定、激励规划、轻量起步、团队学习）以提升用户坚持度。

**💡 创新点**

创新点在于系统性地将四个要素（Goal, Incentive, Light, Team）整合为一套可操作的实用方法，并给出分阶段实施细则，首次将长期学习习惯与具体操作步骤相结合。

**🔧 技术方法**

本文并未采用传统算法或机器学习技术，而是基于调查问卷、用户自述和个人长期数据经验，提出实践性的改进策略。

**📊 数据集**

使用的数据包括 389 份有效问卷（涵盖年龄、性别、教育与职业等信息）以及作者本人在 Shanbay 平台上 4,000+ 天的检查记录。

**📈 对比分析**

方法的有效性主要通过作者个人案例和少量用户的反馈得到验证，没有进行系统的量化性能对比或实验。

**⚠️ 局限性**

局限性在于缺乏大规模实验验证与对照组对比，方法的普适性和效果主要基于单一平台（Shanbay）和作者经验，未来需在更多平台和更广泛人群中进行验证。

---

## 501. Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation

**arXiv ID:** 2602.11440 | [PDF](https://arxiv.org/pdf/2602.11440v1)

**作者:** Penghui Ruan `[一作]` (Hong Kong Polytechnic University), Yuhui Shi `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 27102 | [OpenAlex ID](https://openalex.org/A5000919145)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Diffusion model` `Image` `Video` `Benchmark`

**🎯 论文内容**

开发了一种端到端的扩散框架，实现对图像和视频中对象的几何一致性编辑，包括精确的位移、旋转和相机姿态控制。

**💡 创新点**

创新点在于将对象编辑拆分为对象移除和参考引导填充两阶段，并在统一的扩散过程中注入相机姿态控制，完成无显式3D重建的精细几何控制。

**🔧 技术方法**

采用 ControlNet 风格的 DiT 网络、VAE 编码、多任务多阶段训练、相机姿态轴角编码、流匹配训练以及自监督的对象粘贴网络。

**📊 数据集**

构建了可扩展的真实世界配对数据管线，使用3D物体网格重建与可微渲染生成的训练对；评估时使用 GeoEditBench 的 346 对真实图像对及 ObjectMover-A 基准。

**📈 对比分析**

在 GeoEditBench 与 ObjectMover-A 的零射击评测中，PSNR、DINO、CLIP、DreamSim 等指标均优于现有方法，Pose MAPE 最低、物体 IoU 最高，证明在几何一致性与可控性上表现最强。

**⚠️ 局限性**

受限于 3D 网格重建，难处理非刚性、透明或高度反射物体；相机姿态需手动参数，缺乏直观交互；缺乏物理光照与遮挡复杂度建模，视频编辑质量相对较低。

---

## 502. Mechanistic Evidence for Faithfulness Decay in Chain-of-Thought Reasoning

**arXiv ID:** 2602.11201 | [PDF](https://arxiv.org/pdf/2602.11201v1)

**作者:** Donald Ye `[一作]` (Algoverse), Linus Wong `[通讯]` (Algoverse)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Chain-of-Thought` `Text`

**🎯 论文内容**

提出了Normalized Logit Difference Decay (NLDD)度量，用来量化Chain-of-Thought解释的因果可靠性。

**💡 创新点**

创新点在于在logit空间标准化差异并定义“Reasoning Horizon(k*)”，实现跨模型可比的细粒度因果评估。

**🔧 技术方法**

采用logit差异、RSA、TAS三种指标和对抗性步骤腐蚀实验，以及线性探针分析。

**📊 数据集**

使用Dyck‑n、PrOntoQA、GSM8K三个不同复杂度的推理数据集。

**📈 对比分析**

通过NLDD、RSA、TAS与模型准确率比较，发现Llama/DeepSeek表现出正向因果依赖，而Gemma呈现负向因果（抗信仰），并定位k*在70–85%链长处。

**⚠️ 局限性**

主要局限在于只截断而非替换步骤、仅评估解码器模型、样本量有限以及对随机解码、自由形式CoT的适用性不足。

---

## 503. Evaluation of Security-Induced Latency on 5G RAN Interfaces and User Plane Communication

**arXiv ID:** 2602.12059 | [PDF](https://arxiv.org/pdf/2602.12059v1)

**作者:** Sotiris Michaelides `[一作]` (RWTH Aachen University), Martin Henze `[通讯]` (RWTH Aachen University)

**通讯引用:** 3237 | [OpenAlex ID](https://openalex.org/A5063048519)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Time Series`

**🎯 论文内容**

评估了5G RAN分散化架构中可选安全控制对端到端低延迟的影响，并实现了首个支持所有内部RAN可选安全机制的开放源代码测试平台。

**💡 创新点**

首次系统性评估安全控制对RAN内部接口与用户平面延迟的整体影响；提供了完整的开放源代码测试床；确定IPsec在内部接口上是最优且对低延迟友好的方案。

**🔧 技术方法**

采用5G disaggregated RAN、Open5GS、Open Air Interface、Docker容器、strongSwan IPsec、DTLS、NIA/NEA加密、Python Cryptography库等技术实现实验与性能基准。

**📊 数据集**

通过模拟的UE/UPF ping测量（20,000次）、控制平面UE注册流程（1,000次）以及在真实硬件（MediaTek Dimensity 700、USRP B210）上的验证实验；未使用公开数据集。

**📈 对比分析**

采用99%置信区间对比无安全与启用所有安全的RTT；结果显示disaggregated RAN相较于单体更快，但加密开销已使RTT超过1 ms；IPsec在内部接口仅增加约120 µs，DTLS更高；在真实部署中disaggregated比中心化低约2.5 ms。

**⚠️ 局限性**

实验规模有限，真实网络负载和规模增大时延可能更高；仅评估已标准化的可选安全控制；NIA/NEA实现的串行特性仍是主要瓶颈；未考虑多接入、不同硬件加速等因素。

---

## 504. Learning Conditional Averages

**arXiv ID:** 2602.11920 | [PDF](https://arxiv.org/pdf/2602.11920v1)

**作者:** Marco Bressan `[一作]` (Università degli Studi di Milano), Maximilian Thiessen `[通讯]` (TU Wien)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5083349589)

**关键词:** `Machine Learning`

**🎯 论文内容**

提出在PAC框架下学习邻域条件平均标签的任务，并给出了学习算法和样本复杂度分析。

**💡 创新点**

核心创新是定义并利用两个新的图-概念交叉参数α1和α2，对可学习性给出完全的组合学判定，并证明样本复杂度上界和下界相匹配。

**🔧 技术方法**

主要技术包括图论中的独立集与二染色参数、一次包含图(one‑inclusion graph)预测器、对角度测度与Caro‑Wei不等式的改造以及PAC学习的下界构造。

**📊 数据集**

本文为理论工作，未使用公开数据集，而是通过通用的概率分布与概念类进行泛化分析。

**📈 对比分析**

与传统PAC学习相比，新方法在样本复杂度上仅多了与α1+α2相关的对数因子，理论上达到最优（上界和下界相匹配）。

**⚠️ 局限性**

限制包括对邻域图结构的前置知识要求、仅处理二值标签且只能估计无权重平均，且在实践中实现仍需解决图的可访问性与计算复杂度。

---

## 505. Bizarre Love Triangle: Generative AI, Art, and Kitsch

**arXiv ID:** 2602.11353 | [PDF](https://arxiv.org/pdf/2602.11353v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## 506. Protein Language Model Embeddings Improve Generalization of Implicit Transfer Operators

**arXiv ID:** 2602.11216 | [PDF](https://arxiv.org/pdf/2602.11216v1)

**作者:** Panagiotis Antoniadis `[一作]` (University of Copenhagen), Ole Winther `[通讯]` (University of Copenhagen)

**通讯引用:** 25705 | [OpenAlex ID](https://openalex.org/A5082357240)

**关键词:** `Machine Learning` `Protein Structure Prediction` `Transformer` `Large Language Model` `Biomedical Data`

**🎯 论文内容**

本文提出PLaTITO模型，利用蛋白质语言模型嵌入增强生成式分子动力学的泛化与数据效率，并在快速折叠蛋白与隐秘结合口袋的平衡采样和动力学预测中验证其优越性能。

**💡 创新点**

创新点包括：①将预训练蛋白质语言模型嵌入融入TITO框架，显著提升OOS泛化与数据效率；②推出PLaTITO-Big，在训练成本大幅降低的前提下实现平衡分布和动力学的SOTA表现；③展示模型可预测温度相关的非阿伦尼乌斯速率，体现对复杂能景的捕捉。

**🔧 技术方法**

采用条件流匹配的隐式传递算子（TITO）与两阶段Transformer网络，结合蛋白质语言模型（ESM）、结构嵌入及LLM注释等辅助条件；评估使用TICA、MFPT、自由能曲面等指标。

**📊 数据集**

使用mdCATH离散非平衡MD轨迹（4,471个蛋白域，≤200残基，约56 ms总时间）以及12个快速折叠蛋白的CHARMM22轨迹；与BioEmu等基线进行对比。

**📈 对比分析**

通过平衡分布的MAE、RMSE、Coverage以及快速折叠蛋白的MFPT和自由能曲面进行比较；PLaTITO-Big在MAE 0.824±0.170、RMSE 1.099±0.212、Coverage 0.666±0.136的同时，GPU时数仅1,100小时，显著优于BioEmu（MAE 1.110、RMSE 1.389、Coverage 0.594，GPU时数9,216小时）。

**⚠️ 局限性**

局限性包括：采用Cα粗粒化限制了侧链细节和全原子级预测；模型缺乏无偏动力学、详细平衡和半群一致性的理论保证；对化学空间和温度的泛化仍有限，部分稳态与apo状态的采样仍不完整。

---

## 507. It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks

**arXiv ID:** 2602.12147 | [PDF](https://arxiv.org/pdf/2602.12147v1)

**作者:** Zhongzheng Qiao `[一作]` (Nanyang Technological University), Chenghao Liu `[通讯]` (DataDog)

**关键词:** `Machine Learning` `Benchmark` `Large Language Model` `Time Series`

**🎯 论文内容**

本文提出并实现了一个面向时间序列基础模型的任务中心化基准，包含50个新收集的数据集和98个零样本预测任务，配合人工与LLM双重审核的数据质量管控和任务配置，并引入基于时序特征的模式级评估。

**💡 创新点**

创新点包括：① 采用全新数据源避免数据泄露；② 引入人工+LLM的质量保证流程；③ 按真实业务场景对任务进行上下文对齐配置；④ 通过STL分解提取可解释的时序特征，构建模式级评估视角，提升模型性能洞察的普适性。

**🔧 技术方法**

使用技术包括：自动化质量筛选（时间戳修正、规则校验、Ljung-Box白噪声检验、极端离群值去除、相关性检查）、人工与LLM决策、STL分解+时序特征编码、滚动窗口评估、MASE/CRPS指标、相对S-Naive标准化以及多粒度排行榜可视化。

**📊 数据集**

采用了50个从政府统计、行业合作伙伴、公开网站及竞赛中收集的全新数据集，共覆盖8个领域与多种采样频率，构成98个预测任务。

**📈 对比分析**

通过滚动窗口评估对12种TSFM（TimesFM、Chronos、Moirai、ToTo、Sundial、VisionTS++、Kairos、TiRex等）进行零样本比较，指标为MASE与CRPS，结果显示TimesFM‑2.5在MASE上最低，Chronos‑2在CRPS上最佳；模式级评估进一步揭示不同模型在趋势、季节性、平稳性和复杂度上的优势差异，所有结果可在交互式排行榜查看。

**⚠️ 局限性**

局限性包括：人工审核成本较高；模式级评估目前仅单特征检索，未深入探索多特征交互；排行榜排名受聚合粒度影响；MASE/CRPS等指标可能无法完全映射业务实用性；数据覆盖虽新颖但仍有限，尚未囊括所有行业和频率。

---

## 508. MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation

**arXiv ID:** 2602.11337 | [PDF](https://arxiv.org/pdf/2602.11337v1)

**作者:** Yejin Kim `[一作]` (Allen Institute for AI), Ranjay Krishna `[通讯]` (University of California, Berkeley)

**关键词:** `Robotics` `Robotic Intelligence` `Large Language Model` `Reinforcement Learning` `Vision Language Model` `Image` `Benchmark`

**🎯 论文内容**

创建了MolmoSpaces大规模开源生态系统，包含230k多样室内环境、130k对象资产、42M抓取注解，并提供8个零样本任务的基准评测。

**💡 创新点**

在规模、跨模拟器兼容性、系统化抓取数据、基准任务以及对分布偏移与语言提示的细粒度分析等方面实现了前所未有的创新。

**🔧 技术方法**

利用MuJoCo、IsaacSim、ManiSkill等物理模拟器，结合LLM驱动的场景生成、图形渲染、自动物理验证、抓取生成算法以及基于VLM的导航/操作模型。

**📊 数据集**

采集自AI2-THOR、ProcTHOR、Holodeck、Objaverse等，生成230k场景、130k对象（其中48k可抓取）以及42M抓取姿势。

**📈 对比分析**

采用零样本评估与真实机器人跑验，计算pick、open、close等任务的成功率，并与RoboArena结果做Pearson 0.96、Spearman 0.98的sim-to-real相关；不同模型（π系列、RING、DualVLN）在8任务上表现差异，展示对提示和环境扰动的敏感度。

**⚠️ 局限性**

仍受模拟与真实物理差异影响，长序任务基准有限，模型对语言提示、初始关节、摄像头遮挡等极端情况鲁棒性不足；数据集仍未覆盖全部真实场景分布。

---

## 509. How Many Features Can a Language Model Store Under the Linear Representation Hypothesis?

**arXiv ID:** 2602.11246 | [PDF](https://arxiv.org/pdf/2602.11246v1)

**作者:** Nikhil Garg `[一作]` (Cornell University), Kenny Peng `[通讯]` (Cornell University)

**通讯引用:** 93 | [OpenAlex ID](https://openalex.org/A5011644423)

**关键词:** `Machine Learning`

**🎯 论文内容**

本文建立了线性表示假设（LRH）的数学框架，并在该框架下定量分析单层神经网络可线性表示与线性检索的特征容量；

**💡 创新点**

创新点在于首次将LRH拆解为线性表示与线性可访问两部分，并给出两者在稀疏输入下的近似匹配上界与下界，揭示了线性可访问比传统压缩感知更严格的约束；

**🔧 技术方法**

主要技术包括随机矩阵的相干性分析、低秩矩阵的行列式与特征值估计、图论中的Turán定理以及压缩感知的基础结果；

**📊 数据集**

该工作为纯理论分析，未使用具体数据集；

**📈 对比分析**

通过解析推导比较两种解码方式的特征维度需求，证明在线性可访问场景下维度需满足Ω(k²/ logk log(m/k))，而非线性解码仅需O(k log(m/k))，表明线性检索容量显著受限；

**⚠️ 局限性**

局限性包括仅考虑单层模型、假设特征稀疏且取值范围为[-1,1]或{0,1}，未探讨多层交互与非线性解码器的进一步能力。

---

## 510. Deep Kernel Fusion for Transformers

**arXiv ID:** 2602.11808 | [PDF](https://arxiv.org/pdf/2602.11808v1)

**作者:** Zixi Zhang `[一作]` (Imperial College), Robert Mullins `[通讯]` (University of Cambridge)

**通讯引用:** 17477 | [OpenAlex ID](https://openalex.org/A5011576250)

**关键词:** `Machine Learning` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

开发了一种名为DeepFusionKernel的深度融合GPU内核，用于在SwiGLU MLP块中消除中间缓冲区，减少HBM内存传输并提升内存带宽利用率。

**💡 创新点**

创新点在于将SwiGLU MLP的四个独立GEMM/激活操作深度融合为单一内核，并结合轻量级、基于性能分析的调度器，使加速器能够自适应不同模型、批量大小和硬件平台，实现可部署且持续的吞吐量提升。

**🔧 技术方法**

采用的技术包括CUDA内核融合、行/列切片(Tiling)、循环重排、Tensor Core计算、FlashInfer、CUDA Graph捕获、SGLang框架集成，以及基于运行时分析的动态调度器。

**📊 数据集**

使用LLaMA 3.1 70B模型，FP16精度，固定提示长度1，输出长度从1024到16384 token，批量大小范围1-64，训练/推理均在4个A100或H100 80GB SXM GPU上进行。

**📈 对比分析**

对比基线包括PyTorch（naïve分布式）、SGLang（默认内核）和vLLM。实验显示，DeepFusionKernel在A100上可提升约9.7%，在H100上提升约13.2%；提升在小批量、长生成场景下最为显著，且性能提升稳定且可重复。

**⚠️ 局限性**

局限性包括：未对不同GPU互连（如NVLink、PCIe）进行系统性评估；通信抖动和系统噪声对吞吐量的影响未充分量化；实验仅覆盖LLaMA 3.1 70B，未验证在其他模型或混合精度设置下的效果。

---

## 511. Combinatorial Perpetual Scheduling

**arXiv ID:** 2602.11826 | [PDF](https://arxiv.org/pdf/2602.11826v1)

**作者:** Mirabel Mendoza-Cadena `[一作]` (Centro de Modelamiento Matemático, Universidad de Chile), Kevin Schewior `[通讯]` (University of Cologne)

**通讯引用:** 305 | [OpenAlex ID](https://openalex.org/A5056422606)

**关键词:** `Data Structures and Algorithms` `Optimization`

**🎯 论文内容**

本文提出了一套通用框架，用于处理多任务永续调度问题（CBGT 与 CPS）的组合学变体，并给出了针对不同集合系统（尤其是矩阵、图形、层状、均匀等）的可实现高度上限和高效算法。

**💡 创新点**

创新点主要包括：① 利用矩阵交错多面体的整数性证明，在任意矩阵上可保证高度小于 2（最优）；② 构造“差异矩阵”将 BGT 任务映射为矩阵交错问题；③ 对特殊矩阵（均匀、分区、图形、层状）给出多项式时间实现，分别实现高度 2 或 4；④ 在一般集合系统上证明最优可保证高度为 Θ(log |E|)，并给出 O(log n) 的高效调度算法。

**🔧 技术方法**

核心技术包括：矩阵多面体与交错多面体的整数性；差异矩阵与“木结构”调度（Fuse‑Unfuse）；彩色图和层状结构中的“无环彩色”分割；概率方法和潜能函数的贪婪求解；以及对常数因子进行严格的上界与下界分析。

**📊 数据集**

本研究为纯理论分析，不涉及实际数据集；所有结果均以数学证明和构造性算法为依据。

**📈 对比分析**

与以往的 BGT 研究相比，本文将高度上限从 2 推广到任意矩阵；与先前的图形/层状调度方案相比，提供了统一的颜色分割方法并实现更好的常数 4；在一般集合系统上，本文实现的 O(log n) 高度调度优于之前的 Θ(log² n) 或更差的经验性结果；但在实现效率方面，矩阵通用算法仍为伪多项式，尚未实现真正多项式时间。

**⚠️ 局限性**

局限性：
• 对任意矩阵的高度 2 调度实现仍基于伪多项式时间或需访问矩阵多面体，缺乏真正多项式时间实现。
• 对一般集合系统的 O(log n) 上界虽然最优，但常数因子较大，实际性能取决于具体 λ 分布。
• Greedy（Reduce‑Max）等自然启发式算法是否能实现高度 2 仍是开放问题，缺乏完整的理论证明。
• 对 ℓ‑系统的高度上界仅给出启发式提示，尚无统一的 f(ℓ) 函数。

---

## 512. The Implicit Bias of Steepest Descent with Mini-batch Stochastic Gradient

**arXiv ID:** 2602.11557 | [PDF](https://arxiv.org/pdf/2602.11557v1)

**作者:** Jichu Li `[一作]` (Renmin University of China), Difan Zou `[通讯]` (University of Hong Kong)

**通讯引用:** 2587 | [OpenAlex ID](https://openalex.org/A5085848346)

**关键词:** `Machine Learning` `Optimization` `Tabular`

**🎯 论文内容**

研究小批量随机最陡下降在多类别线性分类中的隐式偏差，阐述批量大小、动量和方差减小对最大间隔行为的影响。

**💡 创新点**

提出统一的梯度最陡下降框架，首次证明动量可消除大批量需求、方差减小可在任意批量下恢复全批量隐式偏差，并揭示单样本更新导致的不同偏差；同时给出无维度依赖的收敛速率。

**🔧 技术方法**

采用随机重排、动量（EMA）与方差减小（SVRG式控制变量）技术，结合入口式与 Schatten‑p 范数的最陡下降映射，推导批量、动量、方差减小共同作用下的误差上界。

**📊 数据集**

实验使用合成线性可分数据集（10 类 × 20 样本，d=5）以及专门构造的正交尺度偏斜数据集进行验证。

**📈 对比分析**

与全批量、无动量、无方差减小的最陡下降进行对比；实验表明全批量收敛至 ℓ₂ 最大间隔解，单批量失效；加入动量可在小批量下收敛但速度变慢；方差减小可在任意批量和动量设置下恢复全批量结果，收敛速率更保守。

**⚠️ 局限性**

局限性：仅针对线性可分数据；大批量条件可能保守；单样本极端批量分析仅适用于构造的数据集，未涵盖非线性模型。

---

## 513. MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling

**arXiv ID:** 2602.11761 | [PDF](https://arxiv.org/pdf/2602.11761v1)

**作者:** MiniCPM Team `[一作]`, Maosong Sun `[通讯]`

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

本文提出了名为 SALA 的 9B 参数混合稀疏-线性注意力模型，并通过持续训练（Transformer‑to‑Hybrid）将 MiniCPM‑4.0 转换为该模型，显著降低训练成本并实现 1M‑token 超长上下文推理。

**💡 创新点**

创新点包括：① 1:3 级别的稀疏/线性注意力混合策略与层级选择算法；② 结合 HyPE、QK‑Norm 与输出门的结构改进；③ 利用 HALO+持续训练实现模型转换，训练成本仅为从零训练的 25%；④ 在多阶段训练中从 4K 延伸至 520K 的长序列，仍保持完整参数更新。

**🔧 技术方法**

技术实现主要包括：InfLLM‑V2（稀疏注意力）、Lightning Attention（线性注意力）、Hybrid Positional Encoding、Rotary Positional Embedding、QK‑Normalization、输出门、HALO 层选择、持续训练框架、GPTQ INT4 量化。

**📊 数据集**

数据集涵盖：MiniCPM‑4.0 预训练 7T tokens，随后 2T tokens 的持续训练；高质量推理数据（L2、L3）、PDF 语料、合成长文本；标准基准：CMMLU、MMLU‑Pro、HumanEval、LCB‑v5/v6、MBPP、AIME24/25、BBH、IFEval；长上下文基准：RULER、MRCR、NoLiMa；超长评估：128K‑2048K 上下文长度的 RULER。

**📈 对比分析**

与 8B‑9B 现有开源模型（Qwen3‑8B、Nemotron‑Nano‑v2‑9B、MiniCPM‑4.1‑8B、Ministral‑3‑R、Falcon‑H1R）进行对比。标准评测平均分 76.53，超长评测平均分 38.97；在 256K‑512K 上下文下，SALA 的 TTFT 仅为 51.6 s（比 Qwen3‑8B 低 3.5×），并可在单张 A6000D/RTX 5090 GPU 上无 OOM 处理 1M‑token。相比之下，Qwen3‑8B 在 512K 及以上会 OOM。

**⚠️ 局限性**

局限性：① 仍为 9B 规模，无法直接评估更大参数对极长上下文的潜在提升；② 训练仍需多 GPU 服务器，虽然成本降低但仍非轻量级；③ 对极端长文本（>2M token）或非文本任务的泛化未作验证；④ 由于使用稀疏注意力，某些细粒度依赖任务可能出现微小性能下降。

---

## 514. A Subword Embedding Approach for Variation Detection in Luxembourgish User Comments

**arXiv ID:** 2602.11795 | [PDF](https://arxiv.org/pdf/2602.11795v1)

**作者:** Anne-Marie Lutgen `[一作]` (University of Luxembourg), Christoph Purschke `[通讯]` (University of Luxembourg)

**通讯引用:** 303 | [OpenAlex ID](https://openalex.org/A5005933561)

**关键词:** `Computation and Language` `Text`

**🎯 论文内容**

本文提出一种基于子词嵌入的无监督方法，直接从原始文本中检测并聚类词形和正字法变体。

**💡 创新点**

创新点在于不依赖预先定义的变体列表或标准化规则，利用余弦相似度和n-gram Jaccard相结合的聚类技术，能够自动发现低资源语言中的新变体。

**🔧 技术方法**

主要技术包括FastText子词嵌入、余弦相似度、n-gram Jaccard重叠度、严格模式下的图连接组件聚类，以及后续的定性人工评估。

**📊 数据集**

使用了约1.42万条来自RTL平台的卢森堡语用户评论（2008-2024年）的大型语料。

**📈 对比分析**

通过与已知方言图谱和语料中的手工分类对比，发现约800个变体族，聚类结果与社会语言学研究一致，表明方法能在噪声或低资源环境中提取有意义的变体，尽管未给出精确的数值性能指标。

**⚠️ 局限性**

局限性包括仅基于单一平台的评论数据，可能无法代表整体卢森堡语使用；聚类对频率敏感，稀有变体易被遗漏；无法区分作者偏好与时间变化的影响；对高度标准化语言效果有限。

---

## 515. A Comparative Study of MAP and LMMSE Estimators for Blind Inverse Problems

**arXiv ID:** 2602.11814 | [PDF](https://arxiv.org/pdf/2602.11814v1)

**作者:** Nathan Buskulic `[一作]` (Università di Genova), Luca Calatroni `[通讯]` (Italian Institute of Technology)

**通讯引用:** 686 | [OpenAlex ID](https://openalex.org/A5019449011)

**关键词:** `Information Theory` `Restoration` `Optimization` `Image`

**🎯 论文内容**

本文在完全已知统计模型的理想化条件下，系统性地比较了盲逆问题中 MAP 与 LMMSE 两种估计器的性能，并验证了 LMMSE 作为初始化点能显著提升 MAP 方法的收敛性和重建质量。

**💡 创新点**

创新点在于：①首次将 MAP 与 LMMSE 在同一实验平台上进行严谨对比；②揭示 LMMSE 既可作为稳健基线，又能作为 MAP 的良好初始化，缓解 MAP 的非凸性与超参数敏感问题；③在合成盲去卷积任务中通过网格搜索验证了该思路的有效性。

**🔧 技术方法**

使用了：基于 DCT 的稀疏表示的合成信号、Gamma 分布的高斯卷积核、白噪声；MAP 采用交替最小化与梯度/投影方法；LMMSE 采用闭式线性最小二乘公式；参数通过网格搜索优化；MSE、迭代进化等指标进行评估。

**📊 数据集**

数据集：合成 50 组 (x, h) 复合数据，图像尺寸 32×32，K=512 的稀疏基，卷积核为 15×15 的正态分布，噪声方差 c_ = 0.0009，Gamma 参数 a=2, β=1。

**📈 对比分析**

比较方法：在同一数据集上计算 MAP 各变体（MAP_σ、MAP_、MAP_σ^boost、MAP_^boost）和 LMMSE 的重建 MSE；结果显示：LMMSE 基线稳定且大多数参数下优于未优化的 MAP；使用 LMMSE 初始化的 MAP 版本在大多数参数下均优于未初始化版本，并在最优参数下逼近甚至略超 LMMSE 性能。

**⚠️ 局限性**

limitations: 研究仅在完全已知统计分布的理想化实验环境下进行，缺乏真实数据验证；MAP 方法仍受非凸性与超参数调节的强烈影响；LMMSE 对高噪声水平和核重建的效果仍有限，且对大规模真实图像的推广尚未评估。

---

## 516. Achieving EF1 and Epistemic EFX Guarantees Simultaneously

**arXiv ID:** 2602.11732 | [PDF](https://arxiv.org/pdf/2602.11732v1)

**作者:** Hannaneh Akrami `[一作]` (Max Planck Institute for Informatics and Universität des Saarlandes), Nidhi Rathi `[通讯]` (University of Warsaw)

**关键词:** `Computer Science and Game Theory` `Optimization`

**🎯 论文内容**

证明在可加价值的公平分配实例中，总能找到既满足EF1（更强的EFL）又满足EEFX的分配方案。

**💡 创新点**

提出新的“强EEFX份额”(strong EEFX share)概念，证明其与RMMS、MXS的关系，并利用该概念实现两种公平准则的兼容性。

**🔧 技术方法**

运用了孤独分割(lone divider)技术、残余最大份额(RMMS)与强EEFX份额的比较，结合组合优化与子集分配构造证明。

**📊 数据集**

无实验数据集，论文为理论存在性证明。

**📈 对比分析**

未给出实验或算法复杂度比较，仅指出该结果为存在性证明，可转化为指数时间算法，但尚无多项式时间实现。

**⚠️ 局限性**

局限性：仅证明存在性，缺乏高效算法；结果仅适用于可加价值，尚未推广到更一般的单调或其他价值模型；对具体实例的可实现性未给出。

---

## 517. Stress Tests REVEAL Fragile Temporal and Visual Grounding in Video-Language Models

**arXiv ID:** 2602.11244 | [PDF](https://arxiv.org/pdf/2602.11244v1)

**作者:** Sethuraman T `[一作]`, Derek Hoiem `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 23221 | [OpenAlex ID](https://openalex.org/A5009682734)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Object Detection` `Object Tracking` `Anomaly Detection` `Transformer` `Vision Language Model` `Prompt Engineering` `Video` `Multimodality` `Benchmark`

**🎯 论文内容**

设计并公开了 REVEAL 诊断基准，包含五个控制压力测试，用以系统评估视频‑语言模型在视觉证据、时间序列与运动感知上的鲁棒性。

**💡 创新点**

创新点在于：①提供可扩展的自动化数据生成管线；②构建多维度诊断框架，细粒度揭示 VidLM 在视觉、时间与运动方面的系统弱点；③公开基准与代码，促进研究复现。

**🔧 技术方法**

利用文本提示与程序化变换生成视频扰动；采用多模大模型（Gemini、Qwen、LLaVA 等）评估；结合人工验证与 LLM 生成的评判指标；采用问答、多选与开放式回答等任务形式。

**📊 数据集**

使用了 Ego4D、YouCook2、Charades、Pexels、EPIC‑KITCHENS、NextQA、Driving Video with Object Tracking 等公开视频数据集。

**📈 对比分析**

与多种开源与闭源 VidLM（Gemini 2.5 Pro、GPT‑5‑nano、Qwen2.5‑VL‑7B/32B/72B、LLaVA‑NeXT‑Video‑7B）以及人类基准进行对比；模型在大多数测试（时间逆转、空间遮蔽、相机运动等）显著低于人类，Gemini 2.5 Pro 虽相对强，但整体性能仍远落后。

**⚠️ 局限性**

受限于预训练偏差、架构对时间不敏感、对视觉证据依赖不足；数据生成与人工验证可能保留偏差；基准侧重单一维度，未覆盖多模异构或真实部署场景；未评估音频同步、对抗编辑等维度。

---

## 518. T3D: Few-Step Diffusion Language Models via Trajectory Self-Distillation with Direct Discriminative Optimization

**arXiv ID:** 2602.12262 | [PDF](https://arxiv.org/pdf/2602.12262v1)

**作者:** Tunyu Zhang `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `Computation and Language` `Knowledge Distillation` `Optimization` `Generation` `Transformer` `Diffusion model` `Text`

**🎯 论文内容**

本论文提出了一种轨迹自蒸馏框架，训练少步解码的扩散语言模型（DLLM），通过教师模型自身的生成轨迹进行监督；

**💡 创新点**

创新点在于将轨迹级监督与逆Kullback-Leibler（DDO）目标结合，实现对多模态后验的模式寻优，并引入路径一致性正则化缓解错误传播；

**🔧 技术方法**

采用的主要技术包括轨迹自蒸馏、Direct Discriminative Optimization（DDO）以及轻量级路径一致性重加权；

**📊 数据集**

在四个基准数据集（MATH500、GSM8K、MBPP、HumanEval）上评估，并在自训练数据中采集教师生成的响应；

**📈 对比分析**

与ReDi、dParallel、Naive TD及SFT等基线对比，实验显示在多种少步（TokPS 2/4）和动态解码场景下，所提方法在准确率、吞吐量和延迟方面均优于或持平基线，并在全步解码下保持与原始模型相近的性能；

**⚠️ 局限性**

局限性包括在极端极少步（例如仅1步）情况下仍存在质量下降；方法仍依赖教师生成数据，且在不同任务或更大模型规模下的泛化能力尚未充分验证。

---

## 519. Multi UAVs Preflight Planning in a Shared and Dynamic Airspace

**arXiv ID:** 2602.12055 | [PDF](https://arxiv.org/pdf/2602.12055v1)

**作者:** Amath Sow `[一作]` (Linköping University), Christian Esteve Rothenberg `[通讯]`

**关键词:** `Artificial Intelligence` `Optimization` `Tabular`

**🎯 论文内容**

提出DTAPP‑IICR框架，解决大型异构UAV车队在动态共享空域中的预飞行规划，兼顾时变禁飞区、机型异构与交付时限；

**💡 创新点**

①4D单机规划器SFIPP‑ST融合时变NFZ、软冲突；②基于LNS的增量冲突解决与几何冲突图；③方向性剪枝降低搜索分支同时保持完整性；

**🔧 技术方法**

优先规划、SFIPP‑ST（SIPP变体）、大型邻域搜索（LNS）与几何冲突检测、方向性剪枝、软约束A*、批量CBS/ECBS对比；

**📊 数据集**

Monte Carlo随机网格（100×100×10）、真实城市地图（400×400×25）与UNetyEmu仿真，包含不同数量的NFZ与障碍；

**📈 对比分析**

与PP、PP（无剪枝）、批量CBS、ECBS比较；DTAPP‑IICR在含NFZ场景下成功率近100%，最大1000 UAV运行时间约13 s（平均），相较于CBS仅能处理≤250 UAV，PP失败；剪枝后运行时间提升30–50%；

**⚠️ 局限性**

对极高NFZ数量和规模>1000 UAV性能下降；缺乏自适应邻域选择与学习机制；对动态NFZ切换的实时响应有限；缺乏在线重规划支持；

---

## 520. General Humanoid Whole-Body Control via Pretraining and Fast Adaptation

**arXiv ID:** 2602.11929 | [PDF](https://arxiv.org/pdf/2602.11929v1)

**作者:** Zepeng Wang `[一作]` (Wuhan University), Zongqing Lu `[通讯]` (BeingBeyond)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Mixture of Experts` `Reinforcement Learning` `Video`

**🎯 论文内容**

提出了 FAST 框架，用预训练的全身控制器与轻量级残差学习实现对未知和分布外运动的快速自适应。

**💡 创新点**

创新点在于结合 Center‑of‑Mass‑Aware 控制与 Parseval 正则化 + KL 约束的残差适配策略，既保持零射击鲁棒性，又在新分布下快速收敛，且通过物理信号显式提升平衡。

**🔧 技术方法**

技术包括 Mixture‑of‑Experts（MoE）策略、Proximal Policy Optimization（PPO）训练、CoM/CoP 观测与奖励、Parseval 正则化、KL 限制的残差网络。

**📊 数据集**

使用 AMASS、OMOMO、LaFan1、MotionX、MotionX 低质量来源、文本/视频生成的运动等多样化数据集进行预训练与适配。

**📈 对比分析**

与 GMT、TWIST2 等最先进的全身控制器对比，FAST 在训练集和 OOD 数据集上成功率均最高，且在高动态、低质量运动的适配实验中收敛最快、保持原分布性能最佳。

**⚠️ 局限性**

局限性包括对实时多源高质量运动仍需改进，残差网络规模与计算成本仍受限，且在极端非物理或极噪声输入下的鲁棒性未完全覆盖。

---

## 521. SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation

**arXiv ID:** 2602.12173 | [PDF](https://arxiv.org/pdf/2602.12173v1)

**作者:** Chengxi Zeng `[一作]` (University of Bristol), Fan Zhang `[通讯]` (University of Bristol)

**通讯引用:** 5642 | [OpenAlex ID](https://openalex.org/A5100403473)

**关键词:** `Artificial Intelligence` `Segmentation` `Knowledge Distillation` `Compression` `Computational Efficiency` `Transformer` `Knowledge Distillation` `Text` `Video`

**🎯 论文内容**

对SAM3文本编码器进行大规模分析，发现其在分割任务中存在显著冗余，随后通过知识蒸馏将其替换为轻量级的MobileCLIP学生模型，形成SAM3-LiteText。

**💡 创新点**

创新点包括：① 对分割提示词的“解剖学”分析，量化上下文窗口、词表稀疏度和嵌入空间的内在维度；② 设计了领域感知蒸馏策略，包含上下文长度裁剪、词袋一致性正则和位置嵌入压缩；③ 在保持分割性能的前提下实现文本编码器参数高达88%的压缩。

**🔧 技术方法**

采用的技术包括：知识蒸馏（MSE、余弦对齐、一致性损失），MobileCLIP轻量化结构，BPE分词器，SVD和邻居维度估计（TwoNN、MLE）进行内在维度分析，Prompt长度与上下文窗口优化，梯度累计与AdamW优化。

**📊 数据集**

使用的数据集：总共404,796个独特提示词，来源包括RF100-VL、LVIS、RefCOCO/RefCOCO+/RefCOCOg、SA-Co Gold/Silver/VEval；分割评测在SA-Co Gold（实例分割）和SA-Co VEval（视频分割）上；视频评测在SA-V、YT-Temporal-1B、SmartGlasses三大基准上。

**📈 对比分析**

与SAM3教师模型及多种基线（gDino‑T、OWLv2、LLMDet‑L、APE‑D、DINO‑X、Gemini 2.5）进行对比。SAM3-LiteText在CG_F1、IL_MCC、pmF1等指标上保持≈98%教师性能，参数量减少至42M（88%），文本编码吞吐率提升3.7×，显著降低VRAM占用。

**⚠️ 局限性**

局限性：1) 主要针对短小、面向对象的提示词，对复杂语义或长句提示的适用性有限；2) 上下文长度裁剪虽降低计算，但在极长提示时仍可能出现截断误差；3) 主要关注文本编码器压缩，对图像编码器仍未做同步优化；4) 需要在更广泛的边缘硬件上进一步验证稳定性。

---

## 522. SAFuzz: Semantic-Guided Adaptive Fuzzing for LLM-Generated Code

**arXiv ID:** 2602.11209 | [PDF](https://arxiv.org/pdf/2602.11209v1)

**作者:** Ziyi Yang `[一作]` (Georgia Institute of Technology), Anand Padmanabha Iyer `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 3285 | [OpenAlex ID](https://openalex.org/A5090733623)

**关键词:** `Software Engineering` `AI Code Assistant` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

提出了 SAFuzz，一个结合 LLM 生成的多变提示、语义驱动的 fuzz harness 生成和漏洞预测的混合框架，用于检测 AI 生成的算法代码中的漏洞。

**💡 创新点**

创新点在于利用 LLM 提取问题特定约束生成语义 oracle 的 fuzz harness，融合 LLM 语义特征与静态指标构建漏洞预测模型，实现自适应资源分配和动态早停。

**🔧 技术方法**

技术包括 LLM (Qwen3‑Coder)、Prompt 变体、LLM 驱动的 harness 生成、Jazzer fuzz、随机森林漏洞预测、动态早停等。

**📊 数据集**

数据集为 CSES 算法任务 96 个，生成 1152 个代码变体；在 LeetCode 上做迁移实验。

**📈 对比分析**

与 ChatUniTest、固定时间 fuzz、GreenFuzz 等 baseline 对比；SAFuzz 在漏洞辨别精度从 77.9% 提升至 85.7%，耗时缩短 1.71×，bug 检测召回率从 67.3% 提升到 79.5%。

**⚠️ 局限性**

局限在于仅针对算法题，无法覆盖通用代码；LLM 可能出现 hallucination，需要人工验证；对大型代码库的可扩展性仍待验证。

---

## 523. On the Adoption of AI Coding Agents in Open-source Android and iOS Development

**arXiv ID:** 2602.12144 | [PDF](https://arxiv.org/pdf/2602.12144v1)

**作者:** Muhammad Ahmad Khan `[一作]` (Lahore University of Management Sciences), Abdul Ali Bangash `[通讯]` (Lahore University of Management Sciences)

**通讯引用:** 162 | [OpenAlex ID](https://openalex.org/A5063403084)

**关键词:** `Software Engineering` `AI Code Assistant` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

对193个Android和iOS开源项目的2901条AI生成的PR进行实证分析，探讨平台、代理和任务类别对PR接受率和解决时间的影响。

**💡 创新点**

首次在移动开发场景下系统比较AI编码代理的贡献，提供平台、代理、任务类别层面的基准和趋势洞察。

**🔧 技术方法**

采用GPT‑5进行PR标题归类、贝叶斯平滑、非参数统计检验（Mann‑Whitney、Chi‑Square、Kruskal‑Wallis）以及Dunn检验等方法。

**📊 数据集**

使用AIDev数据集（含932k PR）及2025年8-11月新增的PR，筛选出193个满足条件的Android（98）和iOS（95）项目。

**📈 对比分析**

通过比较PR接受率、解决时间的统计显著性检验，发现Android接受率71%高于iOS的63%，Codex在Android更受欢迎，iOS整体一致；功能性PR解决速度快于非功能性，Android存在波动，iOS更稳定。

**⚠️ 局限性**

局限包括仅用PR接受率和解决时间衡量贡献，未评估代码质量；GPT‑5归类可能存在误差；部分代理/类别样本不足；仅覆盖开源GitHub，未验证其他生态和企业场景。

---

## 524. Designing Scalable Rate Limiting Systems: Algorithms, Architecture, and Distributed Solutions

**arXiv ID:** 2602.11741 | [PDF](https://arxiv.org/pdf/2602.11741v1)

**作者:** Bo Guan `[一作]` (WynerTech Solutions), Bo Guan `[通讯]` (WynerTech Solutions)

**关键词:** `Distributed, Parallel, and Cluster Computing`

**🎯 论文内容**

在生产级分布式环境下，构建并实现了基于 Redis Sorted Set 与 Lua 脚本的滚动窗口速率限制器体系。

**💡 创新点**

① 量化滚动窗口与 Token Bucket、Fixed Window 在内存消耗与准确性上的权衡；② 通过三层规则管理架构和哈希式 Lua 脚本实现规则热更新；③ 明确采用 AP 模型并给出 CAP 理论分析。

**🔧 技术方法**

使用 Redis Cluster、Redis Sorted Set、Lua 脚本原子操作、TTL 与哈希标签、三层规则缓存与动态加载。

**📊 数据集**

实验基于生产环境的用户 ID / IP 计数，未使用公开数据集，主要通过内部流量模拟评估。

**📈 对比分析**

通过对比 Token Bucket、Fixed Window 与 Rolling Window 的准确性、内存占用与突发处理效果；滚动窗口在 O(log N) 复杂度下实现高精度计数，内存占用为 8 L 字节，实验显示延迟低于 1 ms，吞吐量满足数十万 TPS。

**⚠️ 局限性**

内存成本高（滚动窗口占用 8 L 字节），AP 模型导致偶尔写丢失/一致性缺失，且在多 DC 部署时需要额外的复制与一致性保证。

---

## 525. Pretraining A Large Language Model using Distributed GPUs: A Memory-Efficient Decentralized Paradigm

**arXiv ID:** 2602.11543 | [PDF](https://arxiv.org/pdf/2602.11543v1)

**作者:** Jinrui Zhang `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 105743 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `Computation and Language` `Federated Learning` `Computational Efficiency` `Transformer` `Large Language Model` `Mixture of Experts` `Text`

**🎯 论文内容**

在论文中，我们提出了一种名为SPES的去中心化、内存高效的Mixture-of-Experts LLM预训练框架，能够在互联网络下使用单机48GB GPU完成2B、7B乃至9B参数模型的预训练。

**💡 创新点**

创新点在于：①只在每台节点上训练一部分专家并定期稀疏同步，从而显著降低单卡内存占用和通信成本；②引入专家合并warm‑up策略，提前跨节点共享知识，提高稀疏训练的收敛速度。

**🔧 技术方法**

核心技术包括：Mixture‑of‑Experts模型、FedAvg式的去中心化优化、gRPC自定义通信协议、PyTorch FSDP+混合精度、RoPE、SwiGLU、RMSNorm等。

**📊 数据集**

使用的数据集为公开可获取的Ultra‑FineWeb、SlimPajama、openweb‑math、algebraic stack、pes2o、arxiv、StarCoder、Nemotron Pretraining Dataset 等。

**📈 对比分析**

与传统集中式训练及DiLiCo等去中心化方法相比，SPES在2B模型上实现了约33%通信量下降、约20%内存下降，且在多项commonsense/事实推理基准上与中心化模型保持相近或略优的性能。

**⚠️ 局限性**

局限性包括：实验规模仅至9B参数且训练标记量不足5e11；仅验证了语言理解任务，未覆盖生成或多模态任务；缺乏对更大模型及更长训练周期的可扩展性评估。

---

## 526. Reliable and Private Anonymous Routing for Satellite Constellations

**arXiv ID:** 2602.11764 | [PDF](https://arxiv.org/pdf/2602.11764v1)

**作者:** Nilesh Vyas `[一作]` (Airbus Central Research and Technology), Svetoslav Duhovnikov `[通讯]` (Airbus Central Research and Technology)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Graph`

**🎯 论文内容**

本文提出一种基于 Loopix 的可靠匿名路由架构，专为 LEO 卫星星座设计，解决路径不可靠、路由平面泄漏和拓扑偏差等问题。

**💡 创新点**

创新点包括：①使用 (n,k) 前向纠错码实现多路径传输，抵消链路失效；②在路由发现阶段引入基于同态加密的 Private Information Retrieval（PIR），保护路由元数据；③基于节点中心性与纬度的自适应延迟策略，消除极地节点流量集中，提升匿名-延迟权衡。

**🔧 技术方法**

关键技术涵盖：Reed‑Solomon erasure coding、BFV 同态加密（SealPIR）、Sphinx 消息封装、Poisson 混合、SR‑MPLS 段路由、分布式并行 PIR（27 服务器垂直拆分）以及自适应混合延迟。

**📊 数据集**

使用 OneWeb 631 卫星星座的真实轨道数据（CelesTrak）构建全路径数据库，随机布置 50 名地面用户，生成 191,577 条最短路径。

**📈 对比分析**

实验对比显示：单路径无 FEC 的消息丢失率可达 40%，而 (n=2,k=1) FEC 将丢失率降至 <1%；PIR 采用 27 服务器并行方案，单条查询延迟 173 ms（<2 s 以内）；FEC 编码/解码吞吐分别约 4 Gbit/s 与 8 Gbit/s；自适应中心性延迟在相同延迟下显著提升熵值。

**⚠️ 局限性**

局限性包括：仅针对被动全球观测者，未评估主动攻击如链路阻塞；对大规模星座（10k+ 卫星）可扩展性依赖未来的真正批量 PIR；对高熵节点的定向攻击仍能显著削弱隐私；硬件加速与可信执行环境的探索仍待实现。

---

## 527. Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation

**arXiv ID:** 2602.12172 | [PDF](https://arxiv.org/pdf/2602.12172v1)

**作者:** Bowei He `[一作]` (MBZUAI), Chen Ma `[通讯]` (CityUHK)

**关键词:** `Artificial Intelligence` `Knowledge Distillation` `Data Synthesis` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

提出一种基于教育学原理的三阶段知识蒸馏框架（Identifier‑Organizer‑Adapter），通过诊断学生缺陷、构建递进课程和对齐认知层次，使用教师 LLM 生成定制化的合成数据，完成对小模型的知识迁移。

**💡 创新点**

创新点在于：①将布鲁姆掌握学习原则和维果茨基最近发展区融入蒸馏流程；②通过知识缺失诊断与依赖图实现针对性知识点的选择；③设计了认知对齐的适配策略（概念具体化、步骤拆解、负荷管理、表达优化、语言简化），使合成数据更贴合学生模型的学习能力；④实现了全流程的自动化迭代调度，显著提升蒸馏效果。

**🔧 技术方法**

技术手段包括：①知识模块化与缺口评估（使用教师/学生 probe 任务得到 Δ(k) 与严重性分数）；②依赖图构建与拓扑课程排序；③基于 Bloom 与 ZPD 的难度增量控制与掌握门槛；④适配模块生成多维度简化的合成样本；⑤利用 LLM（OpenAI o1、DeepSeek‑R1）进行批量数据合成并对学生模型进行 fine‑tune；⑥通过 ablation 与超参稳健性验证整体框架。

**📊 数据集**

主要使用的训练和评测数据集：seed 数据 D_seed（约 3K 示例，来源于 OpenThoughts3‑1.2M 过滤后），以及公开基准 DollyEval、VicunaEval、GSM8K、MATH、AIME2024、HumanEval、MBPP、LiveCodeBench、GPQA‑D 等。

**📈 对比分析**

在 OpenAI o1 或 DeepSeek‑R1 作为教师、Qwen2.5‑3B、LLaMA3.2‑3B 等学生模型上，与 Self‑Instruct、LaMini、Lion、DSS、CasCoD、CounterDistill、Star‑Agents、MADA 等基线进行对比。IOA 在大多数任务上均取得最高分，例如 DollyEval 38.16（相较 MADA 36.42）、MATH 55.79（相较 52.04）、HumanEval 40.64（相较 33.39），提升幅度可达 10–20% 或 1–2 分。实验还表明 IOA 在训练时间上保持竞争力（约 11–12 小时，略快于部分基线）。

**⚠️ 局限性**

局限性：①依赖高质量 seed 数据和知识分解的人工/计算成本；②目前仅验证单一教师 LLM，跨教师或异构模型的适用性未充分评估；③适配策略主要基于规则化模板，缺乏自动化学习机制；④对极小模型或资源受限环境的可扩展性未完整检验；⑤仍未探索多模态或非文本任务的适用性。

---

## 528. CSEval: A Framework for Evaluating Clinical Semantics in Text-to-Image Generation

**arXiv ID:** 2602.12004 | [PDF](https://arxiv.org/pdf/2602.12004v1)

**作者:** Robert Cronshaw `[一作]` (University of Edinburgh), Sotirios A. Tsaftaris `[通讯]` (University of Edinburgh)

**通讯引用:** 7659 | [OpenAlex ID](https://openalex.org/A5054437456)

**关键词:** `Artificial Intelligence` `Generation` `Data Synthesis` `Diffusion model` `Image` `Text` `Biomedical Data`

**🎯 论文内容**

提出了CSEval框架，用于评估文本到图像生成模型在医学影像中的临床语义一致性。

**💡 创新点**

创新点在于将生成图像反向转为文本，并使用RadGraph‑F1衡量临床实体匹配度，能够捕捉现有指标忽略的语义偏差。

**🔧 技术方法**

使用Latent Diffusion Model生成胸片，MAIRA‑2生成放射学报告，RadGraph‑XL提取实体关系，最终用RadGraph‑F1评估。

**📊 数据集**

以MIMIC‑CXR数据集训练模型，并在310张合成胸片上进行实验。

**📈 对比分析**

与FID、MS‑SSIM、BioViL‑T等指标对比，CSEval与专家评分相关性最高，能更准确反映临床语义。

**⚠️ 局限性**

局限在于评估依赖报告生成模型的准确性，且RadGraph‑F1可能受文本格式噪声影响。

---

## 529. Beyond Bilinear Complexity: What Works and What Breaks with Many Modes?

**arXiv ID:** 2602.11975 | [PDF](https://arxiv.org/pdf/2602.11975v1)

**作者:** Cornelius Brand `[一作]` (University of Regensburg), Jiaheng Wang `[通讯]` (University of Regensburg)

**通讯引用:** 23 | [OpenAlex ID](https://openalex.org/A5101984439)

**关键词:** `Computational Complexity`

**🎯 论文内容**

本文研究固定维数d≥4的多线性张量（即d‑模式张量）的复杂度，给出其渐近秩与电路复杂度的上界，并探讨张量 Kronecker 乘积下电路复杂度是否保持子乘法性质。

**💡 创新点**

创新点包括：1) 用图张量的视角给出 Strassen 2ω/3 上界的通用证明，推导出 d‑模式张量上 (d−1)ω/3 的上界；2) 进一步利用改进的雷射方法将此上界提升到 0.772318(d−1)；3) 明确指出电路复杂度与秩在 d≥4 时不再等价，给出子乘法失效的条件与例子；4) 针对 d=4、5 给出更紧的电路复杂度上界。

**🔧 技术方法**

技术手段主要包括：图张量构造与 Kronecker 积的等价性、树宽与线图树宽的算法上限、Yates 递归求解、稠密图的分解为星图和三角形的组合、线性代数与张量网络的结合。

**📊 数据集**

由于研究为理论性质，未使用具体数据集；所有结果均为数学证明与算法构造。

**📈 对比分析**

与现有结果比较：对 d=3 时重现 2ω/3 上界；对 d≥4 时提供 0.772318(d−1) 的新上界；对 d=4、5 的电路复杂度分别上界为 2.2967 与 2.8774，明显优于之前的 d/2+1 的通用上界；子乘法性质的失效被证明为条件性结论，进一步阐明了 d≥4 时复杂度行为。

**⚠️ 局限性**

局限性：1) 上界仍与 ω（矩阵乘法指数）密切相关，尚未突破 ω=2 的限制；2) 子乘法失效结论为条件性，需假设如 Hyperclique 或 Permanent 的非多项式下界；3) 对大 d 的具体上界仅给出通用形式，缺乏针对更大固定 d 的数值优化；4) 目前对电路复杂度的下界仅为 d/2 的平凡下界，实际可能更高。

---

## 530. Can We Really Learn One Representation to Optimize All Rewards?

**arXiv ID:** 2602.11399 | [PDF](https://arxiv.org/pdf/2602.11399v1)

**作者:** Chongyi Zheng `[一作]` (Princeton University), Benjamin Eysenbach `[通讯]` (Princeton University)

**通讯引用:** 1058 | [OpenAlex ID](https://openalex.org/A5035051008)

**关键词:** `Machine Learning` `Reinforcement Learning` `Representation Learning` `Optimization` `Reinforcement Learning` `Tabular` `Benchmark`

**🎯 论文内容**

本文对前向-后向（FB）表示学习方法进行理论剖析，并提出了简化的单步前向-后向（one‑step FB）预训练算法；

**💡 创新点**

创新点在于揭示FB存在的可行性与收敛性问题，将FB目标映射为最小化 Bellman 误差（类似 FQE），并基于此设计了去掉循环依赖、只对固定行为策略估计后向表示的单步FB，显著提升了收敛速度和零样本性能；

**🔧 技术方法**

主要技术包括：LSIF（最小二乘重要性估计）对成功者测量比率的回归，Bellman 迭代/张量范数最小化，正交化正则化，目标网络 Polyak 平滑，Softmax/可重参数化策略梯度；

**📊 数据集**

使用了离线 RL 基准 ExORL（10 个连续控制任务）和 OGBench（16+30 个图像/状态任务）中的数据集；

**📈 对比分析**

与传统 FB、ICVF、HILP、BYOL‑γ 等方法比较，one‑step FB 在 10 个基准中 6 个获得最优或次优表现，平均比 FB 提升 24%（+1.4×），并在图像任务中比对手提升约 20%；

**⚠️ 局限性**

局限性：理论分析主要针对离散 MDP，连续情形需近似；对实际 FB 算法的完整收敛理论尚未给出；单步FB 在覆盖度有限的任务上可能性能下降。

---

## 531. GigaBrain-0.5M*: a VLA That Learns From World Model-Based Reinforcement Learning

**arXiv ID:** 2602.12099 | [PDF](https://arxiv.org/pdf/2602.12099v1)

**作者:** GigaBrain Team `[一作]`, Zheng Zhu `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Reinforcement Learning` `Robotic Intelligence` `Transformer` `Reinforcement Learning` `Vision-Language-Action Model` `World Model` `Video` `Multimodality`

**🎯 论文内容**

提出了基于世界模型的强化学习框架 RAMP，构建了 GigaBrain‑0.5M* 这一能够在多任务长时序操作中实现可靠执行的视觉‑语言‑动作（VLA）模型。

**💡 创新点**

通过将世界模型的未来状态预测与价值估计直接作为策略的条件，实现了比传统优势条件方法更丰富的情境信息，并在闭环人机交互循环中实现自我改进。

**🔧 技术方法**

采用大规模机器人操作与网络视频预训练的混合数据、Diffusion Transformer 作为动作生成器、VAE 与 DiT 组成的世界模型、KL‑正则化强化学习与注意力遮蔽技术。

**📊 数据集**

训练数据包括10,000小时以上的真实机器人交互数据（约4,000小时）与6,000小时的世界模型生成数据，后续使用内部8个任务和RoboChallenge 30个标准任务进行评估。

**📈 对比分析**

在内部任务与RoboChallenge基准上，GigaBrain‑0.5M* 在三项难度较高的操作（如盒子装配、咖啡准备）上达到接近 100% 的成功率，显著优于 AWR、RECAP 等对照方法，提升幅度约 30%。

**⚠️ 局限性**

模型在推理时仍需依赖世界模型的预测，导致计算开销增加；对极端外部环境变化的鲁棒性及对更高维度感知模态（如3D点云）的适配仍需进一步研究。

---

## 532. Searching for Optimal Prices in Two-Sided Markets

**arXiv ID:** 2602.11691 | [PDF](https://arxiv.org/pdf/2602.11691v1)

**作者:** Yiding Feng `[一作]` (Hong Kong University of Science and Technology), Zongqi Wan `[通讯]` (Great Bay University)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5019508510)

**关键词:** `Computer Science and Game Theory` `Optimization`

**🎯 论文内容**

研究在线两侧市场中的定价问题，分析不同价格机制（单价、双价、分段价）在实现收益最大化（profit）与交易增益最大化（GFT）时的学习难度，并给出针对不同市场规模（双边、一对多、多对多）与情境（无情境、线性特征）的最优或近似 regret 上界与下界。

**💡 创新点**

首次系统地阐明了价格机制表达能力与可学习性之间的边界：在双边交易中，GFT 仅需常数 regret；在一对多场景下，双价机制可实现 O(log log T) regret；但在多对多场景下，双价机制必产生线性 regret；通过引入分段价机制（可对买卖双方各分两组定价）突破该障碍，取得 O(n² log log T + n³) 的 GFT regret 上界；对利润最大化给出 O(n² log log T) 的最优双价机制 regret。进一步将这些结论推广到包含 d 维线性特征的情境模型，给出 O(n² d log log T + n² d log d)（利润）与 O(n² d² log T)（GFT）的 regret 上界。

**🔧 技术方法**

主要技术包括：
- 二分搜索与保守二分搜索相结合的自适应价格更新；
- 基于不确定性区间的潜在函数分析，捕捉信息增益与误差衰减；
- 对多对多情境采用“分段价”机制，利用匹配结构将信息分层；
- 在情境下运用 ellipsoid 方法与 Steiner 多面体体积技巧，控制多维不确定性；
- 通过构造对抗实例证明下界，揭示学习难点。

**📊 数据集**

本文为理论工作，未使用具体数据集；所有结果均通过数学证明与对抗构造得到。

**📈 对比分析**

与已有工作比较：
- 对利润最大化的双价机制实现了与已知下界匹配的最优 O(log log T) regret；
- 对 GFT 的单价、双价机制提供了最优常数/对数级别的上界，先前仅有常数/线性或较弱的下界；
- 对多对多场景的线性下界与分段价的上界构成完整可学习性图景；
- 在情境扩展中首次给出多维特征下的 GFT 与利润的对数/对数平方上界。

**⚠️ 局限性**

局限与未来工作：
- 结果仅适用于单参数（成本/价值）模型；
- 对更复杂的多参数或非线性特征模型尚未覆盖；
- 对真实市场的实验验证缺失，需进一步实证检验；
- 对动态预算约束（非每轮预算平衡）和多工厂/多商品场景的推广仍是挑战。

---

## 533. Detecting RLVR Training Data via Structural Convergence of Reasoning

**arXiv ID:** 2602.11792 | [PDF](https://arxiv.org/pdf/2602.11792v1)

**作者:** Hongbo Zhang `[一作]` (Zhejiang University), Yue Zhang `[通讯]` (Westlake University)

**通讯引用:** 17849 | [OpenAlex ID](https://openalex.org/A5100333758)

**关键词:** `Artificial Intelligence` `Reinforcement Learning from Human Feedback` `Reinforcement Learning` `Text`

**🎯 论文内容**

提出一种黑盒方法（Min‑kNN）通过检测RLVR训练导致的推理路径结构聚集来识别训练数据是否已被模型见过。

**💡 创新点**

①发现RLVR训练会让已见过的样本产生更少、相似的推理轨迹；②利用该结构收敛性设计无需模型内部信息、仅基于采样的判别统计量。

**🔧 技术方法**

多次采样生成推理结果，计算互为最近邻的编辑距离并取k个最小值的平均作为分数；通过阈值实现成员推断。

**📊 数据集**

使用公开的RLVR‑调优模型（SimpleRL‑32B、DAPO‑Qwen‑32B、JustRL‑DeepSeek‑1.5B、Open‑Reasoner‑Zero‑7B）以及RL‑MIA基准模型（Qwen2.5‑7B‑Instruct、DeepSeek‑Math‑7B‑Instruct），对应的数据来源包括 AIME、Beyond‑AIME、Omni‑Math、MATH‑500 等。

**📈 对比分析**

与PPL、Min‑K% Prob、Min‑K%++、Recall、CDD、Self‑Critique等基线对比，AUC平均达0.70，较最佳基线提升约17%，在不同RL算法、模型规模和扰动条件下均表现稳定。

**⚠️ 局限性**

需要较多采样次数且对解码温度敏感；对结构高度多样化的代码类任务效果相对有限；在极低温度或过度随机的解码策略下结构聚集信号减弱。

---

## 534. DRACO: a Cross-Domain Benchmark for Deep Research Accuracy, Completeness, and Objectivity

**arXiv ID:** 2602.11685 | [PDF](https://arxiv.org/pdf/2602.11685v1)

**作者:** Joey Zhong `[一作]`, Jerry Ma `[通讯]`

**关键词:** `Machine Learning` `Large Language Model` `Text` `Benchmark`

**🎯 论文内容**

提出并发布了DRACO基准，涵盖100个来源于Perplexity Deep Research真实使用场景的跨域深度研究任务，并对这些任务制定专家级评估标准；随后使用LLM-as-a-judge对OpenAI、Gemini、Claude和Perplexity等主流深度研究系统进行统一评估。

**💡 创新点**

①将真实生产查询转换为可评估的开放式研究任务；②构建细粒度、专家设计的多维评估标准（覆盖事实准确性、分析深度、呈现质量和引用质量）；③在统一框架下对多系统进行对比，揭示Perplexity Deep Research在多维度上显著优于其他系统的结论。

**🔧 技术方法**

使用LLM辅助任务重写与扩展、自动过滤、专家手工审核、LLM-as-a-judge（基于Gemini-3-Pro）进行逐条准则评估；评估流程包括多轮判定、权重聚合和归一化分数计算。

**📊 数据集**

数据集为Perplexity Deep Research系统的匿名化请求记录（约数千万条），经过筛选、预处理、增强和人工审核后得到的100个任务；相关引用与源信息需来自40个不同国家的公开数据库、官方文档等。

**📈 对比分析**

比较方法：对每个系统在100个任务上进行5次独立LLM评判，统计归一化分数和通过率；同时按任务域、评估维度和资源消耗（输入/输出Token、延迟）进行细分。性能显示：Perplexity Deep Research (Opus 4.6)以70.5%归一化分数和72.8%通过率位居榜首，Gemini、OpenAI o3和Claude Opus分别落后约10–20个百分点；在所有评估维度和大多数域中，Perplexity表现均为最高。

**⚠️ 局限性**

限制：①评估仅为单轮交互，未覆盖多轮澄清与对话能力；②基准为静态，缺乏动态任务更新；③仅支持文本输入输出，无法评估多模态功能；④任务增补可能过度规范化，降低真实查询的多样性；⑤专家-LLM协同评审成本高，评估尺度与真实人类偏好存在差异；⑥系统内部组件贡献难以分离，缺乏细粒度诊断。

---

## 535. Quantum-Enhanced Temporal Embeddings via a Hybrid Seq2Seq Architecture

**arXiv ID:** 2602.11578 | [PDF](https://arxiv.org/pdf/2602.11578v1)

**作者:** Tien-Ching Hsieh `[一作]` (University of Southern California), Samuel Yen-Chi Chen `[通讯]` (Wells Fargo)

**通讯引用:** 1720 | [OpenAlex ID](https://openalex.org/A5021414038)

**关键词:** `Computational Engineering, Finance, and Science` `Optimization` `Recommendation System` `Finance Related` `Recurrent Neural Network` `Time Series` `Sequential`

**🎯 论文内容**

开发了一个深度为1的变分量子电路嵌入QLSTM Seq2Seq编码器，用于学习股票季度收益的二维时间嵌入。

**💡 创新点**

创新点在于将量子层嵌入LSTM门内，构造几何稳定的潜在空间，并将其作为RBF核驱动的组合优化基础。

**🔧 技术方法**

采用了量子LSTM、变分量子电路、RBF核、图正则化及离散动量选择器等技术。

**📊 数据集**

使用2022–2025年间标准普尔500指数成分股的每周收益数据。

**📈 对比分析**

与经典LSTM基线对比，十四个滚动窗口中RBF‑Graph累计收益达2.4倍、RBF‑DivMom为1.1倍，均超过基准并提升夏普比率，表现出更优的风险调整收益。

**⚠️ 局限性**

局限在于仅与经典LSTM比较，未与更先进的时序模型对比；量子层深度固定为1，未探讨更深或不同Ansatz；仅验证于S&P500，缺乏跨资产类别的通用性验证。

---

## 536. Multi Graph Search for High-Dimensional Robot Motion Planning

**arXiv ID:** 2602.12096 | [PDF](https://arxiv.org/pdf/2602.12096v1)

**作者:** Itamar Mishani `[一作]` (Carnegie Mellon University), Maxim Likhachev `[通讯]` (Carnegie Mellon University)

**通讯引用:** 11103 | [OpenAlex ID](https://openalex.org/A5103257510)

**关键词:** `Robotics` `Robotic Intelligence` `Optimization` `Simultaneous Localization and Mapping` `Benchmark`

**🎯 论文内容**

提出 Multi-Graph Search 算法，利用多条根节点子图进行并行搜索并通过连接与合并实现高维机器人运动规划的高效确定性搜索。

**💡 创新点**

创新点在于把搜索从单一方向扩展到多图结构，结合基于工作空间反向 BFS 的根选择、焦点搜索保证子最优，以及前后端连接启发式实现子图间的高效合并。

**🔧 技术方法**

采用焦点搜索、单队列搜索、前后端连接启发式、3D 工作空间反向 BFS、吸引子技术以及差分逆运动学等技术。

**📊 数据集**

使用 Motion Benchmarker 提供的七个实验场景（shelf pick、bin pick、cage extraction、low‑clearance passage、deep shelf reach、cluttered table、warehouse）以及 Franka Panda（7-DOF）与 Ridgeback+UR10e（9-DOF）硬件。

**📈 对比分析**

与 OMPL、优化基、搜索基的 12 个基线对比，指标为成功率、路径成本、规划时间和 CV；Multi‑Graph Search 在大多数任务中成功率 >95%，路径成本低于大多数基线，规划时间比权重 A* 更快且更一致。

**⚠️ 局限性**

局限在于根选择仅依赖端执行器工作空间，可能忽略全机体约束；根数过少或过多会影响效率；未针对动态环境或学习策略进行扩展。

---

## 537. Scene-Aware Memory Discrimination: Deciding Which Personal Knowledge Stays

**arXiv ID:** 2602.11607 | [PDF](https://arxiv.org/pdf/2602.11607v1)

**作者:** Yijie Zhong `[一作]` (Tongji University), Haofen Wang `[通讯]` (Huawei Technologies Co. Ltd.)

**关键词:** `Computation and Language` `Recommendation System` `Optimization` `Computational Efficiency` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

提出一种基于场景感知的记忆鉴别方法SAMD，结合门控单元模块（GUM）和聚类提示模块（CPM）在LLM上实现对日常交互数据的可记忆性筛选，提升个性化应用的记忆质量与效率。

**💡 创新点**

创新点在于将人类选择注意机制与场景导向的关键字识别相结合，构建多视角角色扮演生成的场景特征词表和意图-场景联合聚类的分组提示，实现在不显式意图标注时也能准确识别可记忆信息。

**🔧 技术方法**

使用的技术包括基于LLM的多视角角色扮演、词级过滤门控（GUM）、意图-场景亲和矩阵与SVD聚类、基于规则的聚类提示（CPM）以及对LLM的两步推理输出解释。

**📊 数据集**

主要使用了约8000万条真实用户交互记录（约2600个意图）构建的多种数据集：BID-20K、IID-10K、GID-1K、LID、MLID，并在SCM与MemoryBank两大基准数据集上进行间接评估。

**📈 对比分析**

与现有SCM、Memochat、TiM等方法对比，SAMD在BID-20K上准确率提升至85.4%（比最优对手提升≈20%），推理速度提升≈3倍，计算成本下降25%+，在多轮问答和检索精度上均显著提升。

**⚠️ 局限性**

局限性主要包括：对极少见场景的召回仍受限于关键字覆盖；聚类规则需人工校验以避免误判；在极端噪声或缺失意图场景下的鲁棒性尚待进一步验证。

---

## 538. Analytical Search

**arXiv ID:** 2602.11581 | [PDF](https://arxiv.org/pdf/2602.11581v1)

**作者:** Yiteng Tu `[一作]` (Tsinghua University), Qingyao Ai `[通讯]` (Tsinghua University)

**通讯引用:** 4443 | [OpenAlex ID](https://openalex.org/A5089655391)

**关键词:** `Information Retrieval` `Retrieval` `Optimization` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Reinforcement Learning` `Multimodality` `Text` `Finance Related`

**🎯 论文内容**

提出“分析式检索”范式，构建包含查询解析、检索、推理融合与自适应验证的端到端工作流，用以满足需要跨源证据合成、量化分析与因果推断的复杂信息需求。

**💡 创新点**

创新点在于将检索视为证据构造而非仅为辅助生成，将“相关性”从表面语义转为“分析价值”，强调多路径召回、工具增强推理、以及可追溯的结论验证，并把整个流程建模为序列决策问题。

**🔧 技术方法**

技术方法包括：大语言模型（LLM）+检索增强生成（RAG）框架、文本转SQL、稀疏/稠密检索混合、工具调用（SQL、代码、统计检验）、强化学习或GRPO优化序列决策、动态任务感知索引与增量更新。

**📊 数据集**

未给出具体实验数据集，论文示例使用法律案例、新闻报道、金融数据等多模态语料，强调可在公开数据集（如法律文本语料库、财经数据库、政府统计）上实现。

**📈 对比分析**

论文未进行量化对比实验，评估框架提出多维度指标：结论正确性、关键证据召回、逻辑一致性、可追溯性与效率。未来可通过与传统RAG、深度研究、数据库代理系统在真实场景下进行基准比较。

**⚠️ 局限性**

局限性包括：缺乏真实系统实现与大规模实验；序列决策学习的难度、长期奖励稀疏性和错误传播问题；检索多路径的精确度-召回平衡；动态索引演化的过拟合与维护成本；以及对高质量评测数据与专家评判的依赖。

---

## 539. Artificial intelligence is creating a new global linguistic hierarchy

**arXiv ID:** 2602.12018 | [PDF](https://arxiv.org/pdf/2602.12018v1)

**作者:** Giulia Occhini `[一作]` (University of Cambridge), Anna Korhonen `[通讯]` (University of Cambridge)

**通讯引用:** 10595 | [OpenAlex ID](https://openalex.org/A5081393566)

**关键词:** `Computers and Society` `Text`

**🎯 论文内容**

研究了全球语言AI资源分布的不平等，提出并构建了语言AI就绪指数EQUATE，用于评估6000多种语言的技术、社会与基础设施准备度。

**💡 创新点**

创新点在于将技术资源、数字基础设施与社会经济因素整合为多维度指数，并结合纵向数据和专家权重，首次系统评估语言AI落地的准备度；同时揭示语言技术扩散呈现“Zipfian化”与超速增长的特征。

**🔧 技术方法**

采用了统计建模（OLS、Gompertz曲线、PCA、线性混合效应回归）以及专家问卷加权、Web抓取与可视化等技术。

**📊 数据集**

主要使用Hugging Face模型与数据集快照、ACL Anthology论文、Glottolog语言地理信息、PanLex词汇、CommonCrawl、Wikipedia、互联网普及率、HDI等多源数据。

**📈 对比分析**

通过对比Gompertz曲线、Zipf分布与传统技术扩散曲线，显示语言AI资源分布呈幂律、扩散超速；指数在多语言下具备较高可解释性，可为政策与投资提供量化优先级。

**⚠️ 局限性**

限制包括数据覆盖不完整、语言名称不一致导致漏计、模型与数据质量差异、专家偏好主观性、指数更新滞后以及未能完全捕捉多语境使用场景。

---

## 540. Amortized Molecular Optimization via Group Relative Policy Optimization

**arXiv ID:** 2602.12162 | [PDF](https://arxiv.org/pdf/2602.12162v1)

**作者:** Muhammad bin Javaid `[一作]` (RWTH Aachen University), Martin Grohe `[通讯]` (RWTH Aachen University)

**通讯引用:** 10994 | [OpenAlex ID](https://openalex.org/A5073893026)

**关键词:** `Machine Learning` `Optimization` `Drug Discovery` `Reinforcement Learning` `Graph Neural Network` `Transformer` `Reinforcement Learning` `Graph` `Biomedical Data`

**🎯 论文内容**

提出一种基于图Transformer的可迁移分子优化框架GRXForm，利用组相对策略优化（GRPO）实现无推理时的化学结构改造。

**💡 创新点**

通过在每个起始分子上构建组内奖励归一化，显著降低不同 scaffold 难度导致的梯度方差，从而提升跨结构泛化与训练稳定性。

**🔧 技术方法**

使用Graph Transformer、动作层级掩码、预训练于ChEMBL、后续基于GRPO的强化学习微调。

**📊 数据集**

主要数据集包括ChEMBL（预训练）、ZINC‑250k（scaffold 划分）、药物活性预测模型（TDC）以及PMO benchmark的标准任务。

**📈 对比分析**

与GraphXForm、LibINVENT、DrugEx v3等 amortized 方法以及 Mol GA、GenMol 等实例优化器比较，GRXForm 在 kinase scaffold 装饰任务中 17.8% 的成功率显著高于 0%，在 prodrug 转化任务中获得更高的目标分数，在 PMO benchmark 上实现第二高的累计 AUC（16.433），同时大幅降低每个实例的计算开销。

**⚠️ 局限性**

仍需在训练阶段进行大量 oracle 评估，当前仅支持对分子添加修饰，缺乏对分子去除/重排的直接处理；在极端难度的 scaffold 上仍可能出现局部最优。

---

## 541. Search-Based Quantum Program Testing via Commuting Pauli String

**arXiv ID:** 2602.11487 | [PDF](https://arxiv.org/pdf/2602.11487v1)

**作者:** Asmar Muqeet `[一作]` (Simula Research Laboratory), Paolo Arcaini `[通讯]` (National Institute of Informatics)

**通讯引用:** 2640 | [OpenAlex ID](https://openalex.org/A5021486887)

**关键词:** `Software Engineering` `Optimization` `Genetic Algorithm` `Hill Climbing` `(1+1) Evolutionary Algorithm` `Benchmark`

**🎯 论文内容**

提出一种基于搜索的量子程序测试方法——Search‑Based Quantum Program Testing via Pauli Strings（简称SQPPT），在QOPS的基础上引入了期望值驱动的适应度函数与多种搜索策略来生成量子测量测试用例。

**💡 创新点**

1）将测试用例从输入态迁移到可测量的Pauli字符串，显著降低了对完整程序规范的依赖；2）设计了利用Pauli字符串可交换性的测量中心化判据，形成基于期望值的测试oracle；3）用系统搜索而非随机搜索有效提升了细微缺陷的检出率；4）在多种硬件平台（IBM、IQM、Quantinuum）和噪声模型上进行大规模评估，验证了方法的可移植性。

**🔧 技术方法**

搜索技术：遗传算法（GA）、爬山（Hill Climbing）和 (1+1) 进化算法；Pauli字符串分组与可交换性判定采用Reggio等人提出的分区算法；期望值计算使用量子模拟器（Qiskit AER）或真实硬件的测量结果；错误消减技术采用IBM内置的零噪声外推（ZNE）和概率误差放大（PEA）。

**📊 数据集**

基准：MQT 量子电路基准集（10 个真实应用电路），每个电路生成 3 个手工插入的细微错误变体，测试规模扩展到 5、10、15、20、25、29 qubit。

**📈 对比分析**

与原始 QOPS（随机搜索）对比，SQPPT 在所有搜索策略中均实现了更高的平均故障检出率；GA 在大多数量子位数上平均检出率接近 1（> 0.95），而 Hill Climbing 仅在中等规模下保持 0.6–0.8；(1+1) 进化算法速度最快但检出率略低。跨平台实验表明，在 IBM 设备上结合 ZNE/PEA 能够在 29 qubit 级别保持可接受的检出率；IQM 与 Quantinuum 受限于缺乏有效误差消减，检出率显著下降。

**⚠️ 局限性**

1）对细微错误的判定依赖阈值设置，噪声环境下难以统一阈值；2）错误消减方案对硬件依赖强，IQM 设备缺乏类似 IBM 的 ZNE/PEA，导致性能受限；3）实验在 29 qubit 时仍需大量 QPU 时钟，实际可执行性受限；4）当前方法未提供缺陷定位能力，缺陷与测试用例存在一对多关系。

---

## 542. Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization

**arXiv ID:** 2602.11351 | [PDF](https://arxiv.org/pdf/2602.11351v1)

**作者:** Yihang Yao `[一作]` (Carnegie Mellon University), Ding Zhao `[通讯]` (Carnegie Mellon University)

**通讯引用:** 5557 | [OpenAlex ID](https://openalex.org/A5037644321)

**关键词:** `Artificial Intelligence` `Optimization` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Agentic AI` `Text` `Benchmark`

**🎯 论文内容**

针对主动 LLM 代理提出行为整合优化框架 BAO，旨在平衡任务完成度与用户交互量。

**💡 创新点**

创新点在于将主动代理训练视为多目标优化问题，结合行为增强与行为正则化两大模块，并通过回顾性推理与前瞻性规划实现 Pareto 前沿提升。

**🔧 技术方法**

使用基于 GPT-4o 的行为增强 SFT、GRPO 策略优化、turn‑level 奖励塑造以及多目标 MDP 定式化等技术。

**📊 数据集**

实验数据集为 UserRL benchmark（Function‑Gym、Telepathy‑Gym、Turtle‑Gym）和 Qwen3 系列预训练模型。

**📈 对比分析**

与 Gemini、GPT‑4o‑mini 等商业模型及 UserRL 基线对比，在 Function‑Gym 等任务中实现更高 Pass@U‑k、较低用户参与率，并将 Pareto 前沿推向前，整体性能优于现有基线和多数商业模型。

**⚠️ 局限性**

局限性：仅针对文本代理，未扩展到多模态；依赖 LLM-as-judge 可能产生奖励劫持风险。

---

## 543. HiFloat4 Format for Language Model Inference

**arXiv ID:** 2602.11287 | [PDF](https://arxiv.org/pdf/2602.11287v1)

**作者:** Yuanyong Luo `[一作]` (Huawei), Heng Liao `[通讯]` (Huawei)

**通讯引用:** 229 | [OpenAlex ID](https://openalex.org/A5039715250)

**关键词:** `Machine Learning` `Computational Efficiency` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

设计并验证了一种新的4-bit块浮点格式HiF4，用于大语言模型的高效推理。

**💡 创新点**

创新点在于采用64元素大组、三层缩放层级（E6M2 + 两级1位微指数）以及S1P2内组编码，显著降低量化误差并减少硬件开销。

**🔧 技术方法**

技术包括HiF4结构化设计、BF16→HiF4的三级缩放转换算法、HiGPTQ量化自适配以及量化误差与点积流程的对比分析。

**📊 数据集**

使用的数据集包括高斯分布随机矩阵、LLM推理基准（ARC、BoolQ、HellaSwag、MMLU等）以及大型模型DeepSeek‑V3.1和LongCat的专用测试集（Gsm8K、Math500、CMMLU）。

**📈 对比分析**

通过与NVFP4、MXFP4直接cast以及加PTS方案的对比，HiF4在多模型多任务上准确率更高，硬件面积约为NVFP4的三分之一，功耗下降约10%。

**⚠️ 局限性**

局限性在于目前仅验证推理阶段，训练兼容性和极端数值分布下的鲁棒性尚未深入探究。

---

## 544. Mapping the Landscape of Affective Extended Reality: A Scoping Review of Biodata-Driven Systems for Understanding and Sharing Emotions

**arXiv ID:** 2602.11710 | [PDF](https://arxiv.org/pdf/2602.11710v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Human-Computer Interaction`

---

## 545. DiffPlace: Street View Generation via Place-Controllable Diffusion Model Enhancing Place Recognition

**arXiv ID:** 2602.11875 | [PDF](https://arxiv.org/pdf/2602.11875v1)

**作者:** Ji Li `[一作]`, Haiou Liu `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Recognition` `Object Detection` `Transformer` `Diffusion model` `Contrastive Learning` `Image`

**🎯 论文内容**

提出DiffPlace框架，利用place-ID控制器实现背景一致的多视角街景生成，并用生成数据增强视觉地点识别训练

**💡 创新点**

首次在生成模型中引入place-ID控制器，通过线性投影、Perceiver Transformer和对比学习将地点嵌入映射到CLIP空间，实现对场景背景的可控合成

**🔧 技术方法**

采用Latent Diffusion、ControlNet、Stable Diffusion v1.5、CLIP ViT-L/14、Perceiver Transformer、SoftCLIP对比损失、UniPC多步噪声调度等技术

**📊 数据集**

在nuScenes（街景合成、3D检测）和Pitts30k（地点识别）数据集上训练与评估

**📈 对比分析**

与BEVGen、BEVControl、MagicDrive、DualDiff等基线相比，DiffPlace在FID、AR@1/AR@5、3D检测mAP/NDS上均有显著提升，地点识别AR@1从83.5%提升至89.7%

**⚠️ 局限性**

在植被密集或极暗光照环境下合成质量下降，且依赖预训练地点识别模型捕捉背景特征

---

## 546. The Magic Correlations: Understanding Knowledge Transfer from Pretraining to Supervised Fine-Tuning

**arXiv ID:** 2602.11217 | [PDF](https://arxiv.org/pdf/2602.11217v1)

**作者:** Simin Fan `[一作]` (Google Research), Berivan Isik `[通讯]`

**关键词:** `Machine Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text` `Benchmark`

**🎯 论文内容**

研究了大语言模型在预训练和监督微调（SFT）阶段的能力迁移与校准变化，使用跨阶段相关性分析方法评估不同规模模型与不同数据混合对20个基准的影响。

**💡 创新点**

提出了基于相关系数的多维度评估协议（跨阶段准确性/置信度相关、类别内相关性、性能-置信度对齐），揭示模型规模、数据源和能力类别对迁移与校准的反向规模效应与差异化影响。

**🔧 技术方法**

采用了Transformer解码器模型、Pearson相关系数、温度缩放等校准技术，以及多尺度训练（240M与1B）和交叉数据混合实验。

**📊 数据集**

使用9种不同的预训练数据混合（General Web: RefinedWeb/FineWeb‑Edu/DCLM；Code: StarCoder/The Stack v2；Curated: RedPajama‑v2）和标准的20个评测基准（CommonsenseQA、WinoGrande、HellaSwag、PIQA、SIQA、COPA、BoolQ、ARC‑Challenge/Easy、SciQ、OpenBookQA、MNLI、QNLI、RTE、CB、QQP、MRPC、WiC、WSC、MultiRC）。

**📈 对比分析**

通过比较预训练与SFT阶段的相关系数，发现大型模型在准确性迁移上表现更好，但在置信度迁移上更弱；科学推理基准的相关性最高，语义理解基准相关性最低；不同规模模型的类别内相关性从竞争转为协同，表明规模提升可增强类别内一致性。

**⚠️ 局限性**

实验仅覆盖240M和1B两种规模，使用单一SFT数据集，未探讨更大规模模型、多样化后置训练方式及长文本推理、代码生成、安全等重要任务，因而结果在更大模型和不同训练阶段下的普适性仍待验证。

---

## 547. MELINOE: Fine-Tuning Enables Memory-Efficient Inference for Mixture-of-Experts Models

**arXiv ID:** 2602.11192 | [PDF](https://arxiv.org/pdf/2602.11192v1)

**作者:** Arian Raje `[一作]` (Carnegie Mellon University), Gauri Joshi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 7371 | [OpenAlex ID](https://openalex.org/A5067441201)

**关键词:** `Machine Learning` `Computational Efficiency` `Optimization` `Transformer` `Mixture of Experts` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

通过在Mixture‑of‑Experts（MoE）模型上加入辅助损失，使其在每条序列中更倾向于使用少量专家，并训练激活预测器预先将这些专家加载到GPU缓存，从而显著减少CPU‑GPU传输。

**💡 创新点**

创新点在于把路由过程视为可塑性问题，利用辅助损失提升路由局部性，同时通过MLP预测器实现对序列级专家偏好的预取，兼顾模型质量与推理效率。

**🔧 技术方法**

主要技术包括：MoE细调（添加局部性惩罚项）、基于softmax的路由与专家选择、专家激活预测器（MLP）以及缓存预取与动态转移控制。

**📊 数据集**

实验使用常见的下游语言模型基准数据集（如WikiText、LAMBADA、OpenWebText等），以及在公开 MoE 结构（Mixtral‑8x7B、OLMoE、Phi‑3.5‑MoE）上的评测。

**📈 对比分析**

与现有基线（Mixtral‑Offloading、MoE‑Infinity、FLoE 等）对比，本文方法在 NVIDIA H100 上的吞吐量提升 1.2–3 倍，针对需要大量专家迁移的基线可提升高达 14.7 倍；且在所有评测任务中保持或提高了生成质量。

**⚠️ 局限性**

局限性包括：仍需额外细调与预测模型，无法一次性解决所有场景；在极长生成或多任务迁移中缓存策略可能需动态调整；总参数量仍大，GPU 内存依旧是瓶颈；若辅助损失设计不当，可能导致路由崩溃或专家饥饿。

---

## 548. 6G Empowering Future Robotics: A Vision for Next-Generation Autonomous Systems

**arXiv ID:** 2602.12246 | [PDF](https://arxiv.org/pdf/2602.12246v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Networking and Internet Architecture`

---

## 549. Towards On-Policy SFT: Distribution Discriminant Theory and its Applications in LLM Training

**arXiv ID:** 2602.12222 | [PDF](https://arxiv.org/pdf/2602.12222v1)

**作者:** Miaosen Zhang `[一作]` (Southeast University), Baining Guo `[通讯]` (Microsoft Research Asia)

**通讯引用:** 47099 | [OpenAlex ID](https://openalex.org/A5101666011)

**关键词:** `Machine Learning` `Reinforcement Learning from Human Feedback` `Optimization` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Prompt Engineering` `Text`

**🎯 论文内容**

提出基于分布判别理论的 On-Policy SFT 框架，并实现了 In-Distribution Finetuning (IDFT) 与 Hinted Decoding 两种技术。

**💡 创新点**

创新点在于构建 Distribution Discriminant Theory (DDT)，证明中心化对数似然 (CLL) 为最优分布判别统计量，并基于此设计自适应损失与数据对齐方法。

**🔧 技术方法**

使用信号检测理论、KL 散度、中心化对数似然、动态权重机制、提示解码、以及离线 RL 对比等技术。

**📊 数据集**

实验数据集涵盖数学推理与通用推理：Numina-Math、AIME24、AMC23、College-Math、Math-OAI、Minerva-math、MMLU-STEM、ARC-Challenge 等。

**📈 对比分析**

通过与标准 SFT、DFT、EAFT 以及多种离线 RL（Rej@N、DPO、CPO、SimPO、RPO）在相同计算预算下对比，IDFT+Hinted Decoding 在大多数基准上实现了与离线 RL 相当甚至更优的性能，且计算效率更高。

**⚠️ 局限性**

局限性包括：方法主要适用于已预训练好的模型，在线版本与价值对齐等场景尚未验证；在完全未训练或代理任务上可能效果有限，需进一步研究。

---

## 550. Community Concealment from Unsupervised Graph Learning-Based Clustering

**arXiv ID:** 2602.12250 | [PDF](https://arxiv.org/pdf/2602.12250v1)

**作者:** Dalyapraz Manatova `[一作]` (Indiana University), L. Jean Camp `[通讯]` (UNC Charlotte)

**关键词:** `Machine Learning` `Safty and Privacy` `Graph Neural Network` `Graph`

**🎯 论文内容**

本文研究在图神经网络（GNN）社区检测中，如何通过轻量级的网络结构和特征扰动隐藏目标社区，从而保护群体级隐私。

**💡 创新点**

创新点在于提出 FCom-DICE，结合 DICE 的边扰动与特征感知的边添加及节点特征修改，显著提升对无监督 GNN 的隐匿效果。

**🔧 技术方法**

采用无监督 GNN（DMoN）进行社区检测，利用 SHAP 和随机森林回归进行特征重要性分析，并在 LFR 生成图和真实图网络上进行实验；方法基于 DICE 的改进。

**📊 数据集**

实验使用合成的 LFR 图以及真实社交网络（Facebook、Wikipedia）和交易网络（Bitcoin Transactions），并为节点生成基于多元高斯分布的特征。

**📈 对比分析**

通过 M1/M2 隐匿度量与 DICE 进行对比，FCom-DICE 在所有 μ、σ_c 组合下提升约 20–45% 的相对改善，尤其在低 μ 与中等 σ_c 情况下表现最优。

**⚠️ 局限性**

局限性包括：扰动预算仅计边修改，特征扰动被视为边扰动子过程；节点特征仅为高斯生成；假设防御者拥有完整社区与特征信息，未探讨部分/缺失知识的情况。

---

## 551. When would Vision-Proprioception Policies Fail in Robotic Manipulation?

**arXiv ID:** 2602.12032 | [PDF](https://arxiv.org/pdf/2602.12032v1)

**作者:** Jingxian Lu `[一作]` (Renmin University of China), Di Hu `[通讯]` (Renmin University of China)

**通讯引用:** 2250 | [OpenAlex ID](https://openalex.org/A5100670614)

**关键词:** `Robotics` `Robotic Intelligence` `Recurrent Neural Network` `Transformer` `Vision-Language-Action Model` `Multimodality`

**🎯 论文内容**

本文提出并验证了一种名为梯度调整与阶段引导（GAP）的算法，用以解决视觉-本体感知（vision‑proprioception）机器人操纵策略在运动转换阶段被本体感知主导而导致视觉信息被压制的问题，从而提升多模态协作与任务性能。

**💡 创新点**

创新点包括：①引入模态时序性（Modality Temporality）视角，阐明视觉与本体感知在不同运动阶段对决策的重要性；②设计GAP算法，利用运动转换概率对本体感知梯度进行细粒度调整，动态平衡两模态学习；③兼容多种融合方式（concat、sum、FiLM）和视觉‑语言‑动作（VLA）模型，展示通用性。

**🔧 技术方法**

技术方法包括：行为克隆（BC）训练框架；利用M‑P‑L‑T（MLP、扩散、Transformer）策略架构；使用LSTM预测运动转换概率；运用Change‑Point Detection（CPD）识别运动一致阶段；采用梯度调整公式调节本体感知梯度；并结合R3M特征熵分析视觉不确定性。

**📊 数据集**

实验数据集涵盖：Meta‑World 与 RoboSuite 两大仿真环境（包含 pick‑place、assembly、disassemble、push‑wall、stack、threading 等多任务）；以及在真实 xArm 6 + Robotiq 抓手上的实测实验；同时对 Octo 视觉‑语言‑动作模型进行 fine‑tune 评估。

**📈 对比分析**

通过与 vision‑only、concat、MS‑Bot、Aux、Mask 等基线对比，使用成功率及标准差进行评估，结果显示 GAP 在模拟和实测环境中均显著优于所有基线，尤其在运动转换频繁的任务（如 push‑wall、stack、threading）中提升约10%‑20% 的成功率，并在 OOD 场景中保持更强的泛化能力。

**⚠️ 局限性**

局限性在于：实验仅在单一机器人本体（xArm 6）上进行，未验证跨本体的泛化效果；此外，梯度调整需要额外的运动转换概率预测，增加了模型复杂度与推理开销。

---

## 552. The Observer Effect in World Models: Invasive Adaptation Corrupts Latent Physics

**arXiv ID:** 2602.12218 | [PDF](https://arxiv.org/pdf/2602.12218v1)

**作者:** Christian Internò `[一作]` (Bielefeld University), Barbara Hammer `[通讯]` (Honda Research Institute EU)

**关键词:** `Machine Learning` `Explainability and Interpretability` `Domain Adaptation` `Optimization` `Transformer` `Supervised Fine-Tuning` `Time Series` `Physics Related`

**🎯 论文内容**

提出一种非侵入式评估框架 PhyIP，用冻结的自监督模型表示通过线性探测器提取潜在物理定律，并利用符号回归验证其可解释性。

**💡 创新点**

创新点在于：①将物理检验视为固定测量仪器，避免下游适配导致的表示破坏；②给出线性探测器误差上界与 SSL 误差及物理曲率的关系；③通过严格实验控制展示侵入式探测会导致动态量被抹除。

**🔧 技术方法**

技术包括自监督时间序列预测（Transformer、U‑Net、FNO 等）、线性探测器、符号回归、CKA 代表性相似度度量、以及对 OOD 评估的严格控制。

**📊 数据集**

使用三大高保真物理模拟数据集：2D 湍流辐射层、3D 红超巨星包层、3D 超新星爆炸，以及一个基于行星运动的轨道力学实验。

**📈 对比分析**

对比方法包括原始输入线性回归、时间依赖线性探测、非线性 MLP 探测、最后一层微调（LL‑FT）以及全模型微调（IBP）。结果显示 PhyIP 在 OOD 上取得高相关性（ρ>0.9）并恢复经典物理定律；侵入式方法往往出现低相关性或误报。

**⚠️ 局限性**

局限在于：仅适用于可线性可解的物理量；对高度非线性或高度混合的表征可能失效；未来需探索子空间约束或权重保持的微调策略以兼顾新任务与已有物理不变性。

---

## 553. Latent Generative Solvers for Generalizable Long-Term Physics Simulation

**arXiv ID:** 2602.11229 | [PDF](https://arxiv.org/pdf/2602.11229v1)

**作者:** Zituo Chen `[一作]` (Massachusetts Institute of Technology), Sili Deng `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 2371 | [OpenAlex ID](https://openalex.org/A5043537937)

**关键词:** `Artificial Intelligence` `Generation` `Data Synthesis` `Computational Efficiency` `Optimization` `Physics Related` `Transformer` `Flow-based Model` `Ordinary Differential Equation` `Auto Encoder` `Time Series` `Physics Related`

**🎯 论文内容**

开发了Latent Generative Solver（LGS），一种在共享潜在物理空间中进行概率生成的多动力学神经PDE求解器，可实现长期自回归模拟。

**💡 创新点**

创新点包括：①将多样PDE映射到统一潜在空间，解耦空间与时间建模；②引入不确定性开关（k-softened源）和流匹配训练，形成自纠正的长期推演；③采用流强迫更新物理上下文，缓解曝光偏差；④构建大规模多动力学预训练数据集并制定统一基准。

**🔧 技术方法**

使用技术包括：P2VAE潜在编码器、Transformer基于流匹配的Flow Forcing Transformer（PFFT）、k-softened源扰动、流强迫机制、时间金字塔降采样、流匹配损失与ODE积分。

**📊 数据集**

数据集为2.5M条128×128分辨率轨迹，覆盖12类PDE家族（来自FNO-v、PDEBench、PDEArena、The Well等），并在256×256 Kolmogorov流场上进行OOD评估。

**📈 对比分析**

与U-AFNO、CNextU-Net、DPOT等确定性基线在1步、5步、10步预测的L2相对误差（L2RE）比较，LGS在1步与基线相当，5/10步显著降低误差；在计算成本上LGS实现70倍FLOPs节省；在OOD 256² Kolmogorov 适配时收敛速度快、长期漂移更小。

**⚠️ 局限性**

局限性包括：潜在解码误差限制最终预测精度；需要预训练VAE并预计算轨迹，增加数据准备成本；对极端分辨率或物理参数外的系统仍可能产生偏差；k-softened源的参数选择对稳定性和误差有影响。

---

## 554. How to Sample High Quality 3D Fractals for Action Recognition Pre-Training?

**arXiv ID:** 2602.11810 | [PDF](https://arxiv.org/pdf/2602.11810v1)

**作者:** Marko Putak `[一作]` (Aalborg University), Joakim Bruslund Haurum `[通讯]` (University of Southern Denmark)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Data Synthesis` `Convolutional Neural Network` `Video`

**🎯 论文内容**

生成3D分形视频用于动作识别预训练，并提出目标智能过滤（TSF）方法筛选有效的IFS参数。

**💡 创新点**

创新点在于：①用TSF在保持IFS完整多样性的同时，仅过滤几何退化样本；②证明不需要过度美观的分形也能提升转移学习；③将TSF与传统SVD、RF过滤及Baseline对比，显示最佳性能与速度。

**🔧 技术方法**

技术手段包括：3D Iterated Function System (IFS)、Chaos Game、时间插值变换、Temporal Shift Module (TSM)、ResNet‑50 backbone、随机森林、SVD筛选、统计阈值过滤等。

**📊 数据集**

数据集：自制3D分形视频（500类×100视频=50k训练+5k验证），下游评测集UCF101、HMDB51，Kinetics‑400作为基准。

**📈 对比分析**

对比方法：将四种预训练策略（Baseline、SVD‑Control、RF‑Filter、TSF）在UCF101/HMDB51上进行Top‑1/Top‑5评估；TSF在UCF101 Top‑1≈78.3%、Top‑5≈95.4%优于其他策略；生成速度相比Baseline提升约100×，SVD最快但性能最差。

**⚠️ 局限性**

局限性：①在2D分形视频的效果仍更好，表明对变换选择敏感；②实验仅在动作识别任务上验证，需进一步检验2D分形和其它任务；③TSF依赖SVD特征阈值，可能不适用于更大/更复杂的IFS空间。

---

## 555. When agents choose bundles autonomously: guarantees beyond discrepancy

**arXiv ID:** 2602.11330 | [PDF](https://arxiv.org/pdf/2602.11330v1)

**作者:** Sushmita Gupta `[一作]` (Institute of Mathematical Sciences), Meirav Zehavi `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 1229 | [OpenAlex ID](https://openalex.org/A5082025487)

**关键词:** `Computer Science and Game Theory` `Optimization`

**🎯 论文内容**

研究在可分配不可分物品的公平分配问题中提出了一个新的两阶段机制，使得在代理人自主选择分配时可以在多项式时间内保证每个代理人获得接近其比例份额的高价值分配。

**💡 创新点**

突破了传统差异理论所给出的Θ(√n)障碍，实现了对每个代理人至少获得比例份额减去O(log n)的价值保证，并对三类受限价值类给出了更优的保证。

**🔧 技术方法**

结合差异理论、动态分组、重绑定技术、贪心策略以及对交换距离、线性分离、层状交换和影响度等结构的分析。

**📊 数据集**

论文未使用具体实验数据集，而是通过理论构造与证明来验证结果；在某些案例中使用Hadamard矩阵构造反例。

**📈 对比分析**

与传统的比例1/n分配、Envy‑Cycle Elimination等方法相比，算法在保证上实现了指数级提升（从√n到log n），并在受限情形下进一步提升到常数级误差。

**⚠️ 局限性**

局限在于结果仅适用于可加非负、已知价值的情况，且对低比例份额代理人或非递增价值结构的情况仍存在 O(1) 的误差；同时实际实现需要先知道代理人偏好并能按给定顺序到达。

---

## 556. PLOT-CT: Pre-log Voronoi Decomposition Assisted Generation for Low-dose CT Reconstruction

**arXiv ID:** 2602.11625 | [PDF](https://arxiv.org/pdf/2602.11625v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition`

---

## 557. When Visibility Outpaces Verification: Delayed Verification and Narrative Lock-in in Agentic AI Discourse

**arXiv ID:** 2602.11412 | [PDF](https://arxiv.org/pdf/2602.11412v1)

**作者:** Hanjing Shi `[一作]` (Lehigh University), Dominic DiFranzo `[通讯]` (Lehigh University)

**通讯引用:** 1332 | [OpenAlex ID](https://openalex.org/A5028356812)

**关键词:** `Computers and Society` `Agentic AI` `Text`

**🎯 论文内容**

研究了Reddit上关于agentic AI讨论的可见度（如upvotes）与验证（Evidence‑Seeking）出现时机的关系，量化高可见度线程中验证的延迟和缺失；

**💡 创新点**

提出了“可见度–验证时机”分析框架，利用右删失时间模型和词典匹配来捕捉首次验证时机，揭示了“受欢迎度悖论”与“叙事锁定”两种现象；

**🔧 技术方法**

采用规则基词典匹配验证线索、右删失时间计算、Fisher精确检验、置换检验（Permutation）以及Cliff's δ效应量评估；

**📊 数据集**

使用两大Reddit子论坛（r/moltbook和r/openclaw）2026年1月1日至2月6日的帖子与评论快照数据；

**📈 对比分析**

通过将线程按可见度四分位数分组，比较验证出现率和时间分布，结果显示高可见度线程验证率更高（OR≈2–3）但时延显著更长（p<0.001），整体验证时延右移；

**⚠️ 局限性**

仅覆盖两社区，词典方法优先精确而非召回，可能漏检隐性或非文字形式的验证；分析为描述性非因果，未直接测量信任或行为效果，数据仅为公开可见文本。

---

## 558. DIVER: A Robust Text-to-SQL System with Dynamic Interactive Value Linking and Evidence Reasoning

**arXiv ID:** 2602.12064 | [PDF](https://arxiv.org/pdf/2602.12064v1)

**作者:** Yafeng Nan `[一作]` (Beijing University of Posts and Telecommunications), Jingyu Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 12163 | [OpenAlex ID](https://openalex.org/A5100432465)

**关键词:** `Databases` `Large Language Model` `Chain-of-Thought` `Agentic AI` `Text` `Benchmark`

**🎯 论文内容**

通过构建一个无专家干预的多代理系统 DIVER，实现了自动生成高质量的证据，从而提升 Text-to-SQL 的鲁棒性。

**💡 创新点**

关键创新是动态交互式值链接与证据推理结合的 CoTF 工作空间以及工具箱驱动的探索机制。

**🔧 技术方法**

采用 LLM 结合工具调用、链式思考与事实验证的多代理框架，并实现了自我纠错的工具反馈。

**📊 数据集**

在 BIRD、Spider、DR.Spider 和 Spider-DK 四个基准上进行评测。

**📈 对比分析**

与现有 15+ 方法对比，DIVER 在无专家证据条件下实现了最高 10.82% 的执行准确率提升，显著提升了模型在复杂扰动场景的鲁棒性。

**⚠️ 局限性**

主要限制是对非常大规模或高度专业化的数据库仍需更深层的领域知识，且系统对工具调用的依赖可能在不同 SQL 引擎中引入适配成本。

---

## 559. HybridRAG: A Practical LLM-based ChatBot Framework based on Pre-Generated Q&A over Raw Unstructured Documents

**arXiv ID:** 2602.11156 | [PDF](https://arxiv.org/pdf/2602.11156v1)

**作者:** Sungmoon Kim `[一作]` (Hanyang University), Jiwoong Kim `[通讯]` (Makebot Inc.)

**关键词:** `Computation and Language` `Retrieval` `Generation` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

提出 HybridRAG 框架，先通过 OCR、布局分析和层级分块从原始 PDF 生成丰富的 QA 知识库，查询时先检索匹配答案，若匹配不到则回退到 LLM 生成响应。

**💡 创新点**

创新点在于：① 预生成 QA 以降低实时推理延迟；② 针对非结构化 PDF 进行布局、表格/图像描述、层级分块，提升检索与生成质量；③ 在检索时采用问句相似度阈值策略平衡速度与质量。

**🔧 技术方法**

使用 MinerU 进行布局分析、PaddleOCR 识别文本、GPT‑4o/4o‑mini 生成图表描述及 QA、BGE‑M3 进行问句嵌入检索、Llama3.2‑3B 或 Qwen2.5‑3B 进行生成。

**📊 数据集**

数据集为 OHRBench，包含 1,261 篇 8,561 页的多领域未结构化 PDF，含 8,498 对真值 QA 评估。

**📈 对比分析**

与标准 RAG（仅 OCR+检索+LLM）和简化 HybridRAG（仅 OCR+预生成 QA）对比，HybridRAG 在 F1、ROUGE‑L、BERTScore 上均有提升，且平均延迟从 1.7 s 降至 0.8–0.9 s（取决于阈值），显示显著的性能优势。

**⚠️ 局限性**

主要限制是一次性离线 QA 生成所需的高昂算力与 API 成本（约 15–30 USD 级别）以及生成过程耗时，导致部署前需投入较多资源。

---

## 560. MetaMem: Evolving Meta-Memory for Knowledge Utilization through Self-Reflective Symbolic Optimization

**arXiv ID:** 2602.11182 | [PDF](https://arxiv.org/pdf/2602.11182v1)

**作者:** Haidong Xin `[一作]` (Northeastern University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 37357 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `Computation and Language` `Optimization` `Meta Learning` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Reinforcement Learning` `Text`

**🎯 论文内容**

构建了一个自演进的元记忆框架Meta-Memory，用于指导LLM更有效地利用外部记忆；

**💡 创新点**

创新点在于将元记忆作为可学习、可自我修正的知识利用经验库，通过自我反思迭代更新，兼顾任务无关性与任务特异性；

**🔧 技术方法**

采用LLM生成与评判、强化学习式的自我反思指令、元记忆更新操作（增删改）以及多任务训练策略，整体实现自我演进的记忆利用；

**📊 数据集**

使用LongMemEval作为主要评测集，并以ShareGPT作为跨域训练集以验证元记忆的泛化；

**📈 对比分析**

与Full Text、RAG、MapReduce、Mem0、A-Mem、MemoryOS、LightMem等基线对比，Meta-Memory在四类任务（单会话、多会话、时序推理、知识更新）上平均提升约3.6%准确率，显著优于最强基线；

**⚠️ 局限性**

局限在于依赖LLM评判模型做正确性判断，易受评估噪声影响；元记忆在跨域训练时仍落后于专属域训练；数据量有限时学习不稳定，需更多高质量领域数据完善元记忆。

---

## 561. Differentiable Modal Logic for Multi-Agent Diagnosis, Orchestration and Communication

**arXiv ID:** 2602.12083 | [PDF](https://arxiv.org/pdf/2602.12083v1)

**作者:** Antonin Sulc `[一作]` (Lawrence Berkeley National Lab), Antonin Sulc `[通讯]` (Lawrence Berkeley National Lab)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Anomaly Detection` `Optimization` `Multimodality` `Finance Related`

**🎯 论文内容**

本文提出一种可微分模态逻辑框架（MLNN），用于多智能体系统的语义调试。

**💡 创新点**

创新点在于把可解释的Kripke结构作为可学习参数，通过逻辑矛盾损失驱动模型，同时支持多模态（认知、时间、义务、信念）集成。

**🔧 技术方法**

技术包括可微分Kripke语义、Łukasiewicz t‑norm、连续真值与可学习访问关系、以及结合梯度下降的逻辑约束优化。

**📊 数据集**

使用了多种合成实验数据集：外交联盟、分布式系统根因、金融市场违规、LLM幻觉、无人机编队等。

**📈 对比分析**

与传统统计监控、规则基、单一模态方法相比，DML在信任学习、因果定位、合规边界、幻觉检测等任务上分别提升 70%+ 准确率、显著减少误报，且提供可解释的参数。

**⚠️ 局限性**

局限性包括对大规模系统可扩展性的挑战、对先验公理的依赖、对高维连续属性的抽象化不足，以及训练对噪声标签的敏感性。

---

## 562. Differentially Private Perturbed Push-Sum Protocol and Its Application in Non-Convex Optimization

**arXiv ID:** 2602.11544 | [PDF](https://arxiv.org/pdf/2602.11544v1)

**作者:** Yiming Zhou `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27955 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Optimization` `Federated Learning` `Safty and Privacy` `Convolutional Neural Network` `Transformer` `Image`

**🎯 论文内容**

本文提出了一种差分隐私的推送求和（DPPS）协议，并基于此设计了PartPSP算法，用于在去中心化网络中进行非凸优化，并通过部分通信降低隐私噪声对性能的影响。

**💡 创新点**

创新点包括：①在协议层面实现差分隐私，解决敏感度估计难题；②提出仅需广播一个标量的轻量级敏感度估计方法；③通过将模型参数拆分为局部与共享部分的部分通信机制，显著降低噪声维度，从而提升优化效果。

**🔧 技术方法**

技术主要包括：差分隐私（Laplace机制）、Perturbed Push‑Sum协议、梯度裁剪、随机梯度下降、部分通信策略以及网络拓扑权重设计。

**📊 数据集**

实验使用了MNIST、FMNIST和CIFAR‑10三大数据集，模型分别为MLP、ResNet‑18和ViT。

**📈 对比分析**

与全通信、SGPDP、PEDFL等方法对比，PartPSP在相同隐私预算下在大多数实验中取得了最高准确率，尤其在ResNet和ViT任务上优于其他算法；在无隐私时性能仍保持竞争力。

**⚠️ 局限性**

局限性包括：①敏感度上界的经验估计可能导致过度保守的噪声；②部分通信的参数划分需要先验知识，可能不适用于所有模型；③在网络极端不连通或节点度很低时，敏感度仍可能较大，影响收敛速度。

---

## 563. Preprocessed 3SUM for Unknown Universes with Subquadratic Space

**arXiv ID:** 2602.11363 | [PDF](https://arxiv.org/pdf/2602.11363v1)

**作者:** Yael Kirkpatrick `[一作]` (Massachusetts Institute of Technology), Virginia Vassilevska Williams `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 4122 | [OpenAlex ID](https://openalex.org/A5044244682)

**关键词:** `Data Structures and Algorithms`

**🎯 论文内容**

提出了一种针对预处理 3SUM（未知 C）的数据结构，实现了真子二次空间与查询时间的平衡，并给出了对应的调优曲线。

**💡 创新点**

创新点在于将重数大（heavy‑hitter）值与低度函数逆推技术相结合，利用随机素数模运算与 FFT 过滤假正例，再通过 Fiat‑Naor 函数逆推处理低度部分，从而在空间与查询时间之间取得新的折中。

**🔧 技术方法**

使用的主要技术包括：随机素数模运算与 FFT 计算模 p 的和集；统计与控制假正例；按重数划分 heavy‑hitter 与普通值；将普通值拆分为 O(n^δ) 子集；对每个子集构造 bounded‑degree Fiat‑Naor 逆推结构；多次独立重采样以保证高概率成功。

**📊 数据集**

论文中并未使用具体实验数据集，而是以理论模型（大小为 n 的整数集合）进行分析，空间、时间复杂度均以 n 为参数给出。

**📈 对比分析**

与先前最优结果（O(n^1.5) 查询时间、O(n^2) 空间）相比，本工作在保持 O(n^2) 预处理时间的同时，将空间压缩到 O(n^{2-2/3})，查询时间降低到 O(n^{1.5+ε})，实现了在不违反 3SUM 假设的前提下的子二次空间/时间折中。

**⚠️ 局限性**

局限性包括：查询时间的保证只对无偏对手（oblivious adversary）成立；空间与时间折中仍受 3SUM 假设限制，无法进一步压缩至线性空间；实现依赖于随机化，常数与高阶项较大；对其它变体（如 XOR‑3SUM）的推广需要额外工作。

---

## 564. On the Sensitivity of Firing Rate-Based Federated Spiking Neural Networks to Differential Privacy

**arXiv ID:** 2602.12009 | [PDF](https://arxiv.org/pdf/2602.12009v1)

**作者:** Luiz Pereira `[一作]` (Federal University of Campina Grande), Kyller Gorgônio `[通讯]` (Federal University of Campina Grande)

**关键词:** `Machine Learning` `Federated Learning` `Safty and Privacy` `Spiking Neural Network` `Reinforcement Learning` `Audio`

**🎯 论文内容**

本文研究了差分隐私（DP）对脉冲神经网络（SNN）在联邦学习（FNL）环境下的发射率统计及其对聚合与客户端选择稳定性的影响，并在非IID语音识别任务中进行系统性 Ablation 实验。

**💡 创新点**

创新点在于将DP机制与SNN‑FNL相结合，首次定量分析了梯度裁剪与噪声注入对发射率的系统性偏移、聚合衰减以及稀疏性和内存指标的关联，并给出了针对隐私强度与联邦协调之间权衡的可操作性指导。

**🔧 技术方法**

使用的技术包括脉冲神经网络模型、联邦学习框架、差分隐私（梯度裁剪与噪声注入）、发射率统计分析、稀疏性/内存指标评估以及 Ablation 研究。

**📊 数据集**

实验数据集为语音识别任务（未具体说明，但推测为常见的 LibriSpeech/TIMIT 等公开语音数据集），采用非IID划分模拟真实部署环境。

**📈 对比分析**

通过对不同隐私预算和裁剪阈值的 Ablation 进行对比，发现DP会导致发射率明显下移、聚合效果下降和客户端选择不稳定；与无DP或低隐私预算设置相比，隐私强度越高，联邦协调性能越差，呈现明显折衷关系。

**⚠️ 局限性**

局限性包括：实验仅覆盖单一语音识别任务和特定 SNN 架构；缺乏对更复杂网络或多种隐私参数组合的验证；未给出理论上对发射率偏移的严格定量推导；缺少与其他联邦隐私技术（如同态加密、联邦可验证计算）的对比。

---

## 565. A Note on the Complexity of Directed Clique

**arXiv ID:** 2602.11773 | [PDF](https://arxiv.org/pdf/2602.11773v1)

**作者:** Grzegorz Gutowski `[一作]` (Jagiellonian University), Mikołaj Rams `[通讯]` (Jagiellonian University)

**关键词:** `Computational Complexity`

**🎯 论文内容**

证明了给定有向图的有向团数（directed clique number）决策问题对一般正整数 t 是 Σ₂^P‑完整的，并给出了相应的多项式时间归约；同时给出了一条关于锦标赛的上界 (T) ≤ √(2|V(T)|)。

**💡 创新点**

创新点在于构造了一套新的多项式大小的 gadget（binary、copy、clause）来处理带有两层量词的布尔公式，从而把 Σ₂^P‑完整问题归约到有向团数决策问题；该方法弥补了以往仅在固定 t（t≥3）下的 NP‑完整性结果，并提供了对锦标赛上界的改进。

**🔧 技术方法**

采用了组合归约技术：设计 gadget 使得每个 gadget 的后向边图（backedge graph）在无 K_{2c} 的约束下唯一决定一个变量的取值；利用量词交错结构的解析，结合背边图的团数特性，构造整个图 G；还使用了归纳证明来给出锦标赛的上界。

**📊 数据集**

没有使用实验数据集，所有结果均为理论证明。

**📈 对比分析**

由于是理论复杂性分析，没有对算法进行实验比较；证明展示了问题在 Σ₂^P 级别的难度，暗示任何多项式时间算法都不可能在所有实例上快速求解。

**⚠️ 局限性**

局限性：对锦标赛的下界仍远低于上界，现有的对有向团数的精确估计仍处于对数阶与平方根阶之间；对于 t=2 的情况以及更紧的下界仍是未解决的开放问题。

---

## 566. Native Reasoning Models: Training Language Models to Reason on Unverifiable Data

**arXiv ID:** 2602.11549 | [PDF](https://arxiv.org/pdf/2602.11549v1)

**作者:** Yuanfu Wang `[一作]` (Shanghai Artificial Intelligence Laboratory), Chao Yang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 4974 | [OpenAlex ID](https://openalex.org/A5103069882)

**关键词:** `Machine Learning` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Large Language Model` `Text`

**🎯 论文内容**

引入Native Reasoning Training（NRT）框架，利用标准问答对而非人类推理轨迹或外部验证器，通过将推理过程建模为潜在变量并用内部奖励激励模型生成有助于提升答案预测概率的推理轨迹，进而实现大规模推理能力的自我提升。

**💡 创新点**

创新点在于：①统一框架把推理视作潜在变量并用答案概率做内部奖励；②设计多种奖励聚合函数，特别是基于模型自身不确定性的加权求和（WS）方案；③在无外部验证器的条件下，在多类推理任务上实现 state‑of‑the‑art 结果并显著提高鲁棒性，避免策略崩塌。

**🔧 技术方法**

采用离线强化学习（GRPO/PPO）、差分奖励与结构格式监督，结合多种奖励聚合函数（log‑prob、prob、GM、AM、WS），并在 Llama‑3.2‑3B、Mistral‑7B‑v0.3、Llama‑3.1‑8B 等大模型上进行训练。

**📊 数据集**

使用约 200k 样本的无推理轨迹问答对数据集（仅含问题 x 与真实答案 y*），该子集来自公开通用推理/问答基准（BBH、MMLU、DROP、PopQA、TruthfulQA、GSM8K、MATH、HumanEval、IFEval）。

**📈 对比分析**

与 SFT 基线以及之前的 verifier‑free RL 变体（JLB、Verifree、RLPR）以及 NRT 的不同奖励方案进行对比；在 9 个基准上，NRT‑WS（-log p）在 3B/8B 模型上平均分分别提升 3.5/10.2 分，在复杂推理任务（GSM8K、BBH、MATH、HumanEval 等）上显著优于前者，且对策略崩塌表现出更高的稳定性。

**⚠️ 局限性**

局限性包括：离线采样导致样本效率较低；奖励聚合函数目前是手工设计，未来可探索自适应或学习式奖励；研究仅在微调阶段验证，未尝试在预训练阶段应用；对硬件/计算成本较高；对非结构化或多模态任务的适用性尚未验证。

---

## 567. ExStrucTiny: A Benchmark for Schema-Variable Structured Information Extraction from Document Images

**arXiv ID:** 2602.12203 | [PDF](https://arxiv.org/pdf/2602.12203v1)

**作者:** Mathieu Sibue `[一作]` (J.P. Morgan AI Research), Manuela Veloso `[通讯]` (J.P. Morgan AI Research)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Vision Language Model` `Image` `Text` `Benchmark` `Finance Related`

**🎯 论文内容**

本文提出了一套新的结构化信息抽取（IE）基准数据集，针对视觉丰富文档的闭合、基于模式及按需查询进行评估。

**💡 创新点**

创新点在于统一了 KEE、RE 与 VQA 的查询形式，构建了多类型文档、跨页且包含不可回答查询的多实体 JSON 输出评测体系，并通过手工与 LLM 合成样本的混合流程扩充数据。

**🔧 技术方法**

采用了视觉语言模型（VLM）和大型语言模型（LLM）进行零/少样本结构化抽取，并设计了基于 schema 对齐与 ANLS 的评估框架。

**📊 数据集**

数据集覆盖 110 篇多页文档，包含 304 条闭合与按需查询，来源包括表单、财务报告、幻灯片和网页截图。

**📈 对比分析**

通过在该基准上对比开放与闭源 VLM，发现闭源模型在召回率和鲁棒性上领先，规模更大的模型产生更合法输出，但在复杂查询和答案定位上仍表现不足。

**⚠️ 局限性**

局限性包括仅支持英文、评估指标对数字日期的适配不足、以及依赖文本 LLM 的模式映射速度慢且可能不完美。

---

## 568. STVG-R1: Incentivizing Instance-Level Reasoning and Grounding in Videos via Reinforcement Learning

**arXiv ID:** 2602.11730 | [PDF](https://arxiv.org/pdf/2602.11730v1)

**作者:** Xiaowen Zhang `[一作]` (Xidian University), Qing Li `[通讯]` (State Key Laboratory of General Artificial Intelligence, BIGAI)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Object Tracking` `Reinforcement Learning` `Optimization` `Reinforcement Learning` `Vision Language Model` `Video`

**🎯 论文内容**

通过在视频帧中嵌入唯一对象 ID 的视觉提示，将密集坐标预测改写为实例识别任务，并在此基础上使用强化学习进一步优化时空定位。

**💡 创新点**

提出对象中心化视觉提示范式以及首个针对 STVG 的强化学习框架 STVG‑R1，利用任务驱动奖励统一时空一致性与结构化输出，显著提升模型的可解释性与泛化能力。

**🔧 技术方法**

视觉提示嵌入、YOLOv12 + SAM2 检测/跟踪、Qwen2.5‑VL‑7B 等预训练 VLM、GRPO 算法、时空 IoU / 空间一致性 / 格式奖励。

**📊 数据集**

HCSTVG‑v1/v2、ST‑Align、MeViS、Charades‑STA、TVGBench 等六个基准数据集。

**📈 对比分析**

在 HCSTVG‑v2 上 m_vIoU 提升至 66.7%（比 Qwen2.5‑VL‑7B 提升 20.9%），在 MeViS 零样本下 J&F 达 47.3%，在 Charades‑STA 与 TVGBench 零样本同样超过现有方法；总体表现均为 SOTA。

**⚠️ 局限性**

依赖外部检测/分割模型，视觉提示可能导致遮挡或视觉噪声；在 OCR 或细粒度任务中影响略有出现；仅在单目标 STVG 上训练，跨多目标任务的泛化仍需进一步验证。

---

## 569. SToRM: Supervised Token Reduction for Multi-modal LLMs toward efficient end-to-end autonomous driving

**arXiv ID:** 2602.11656 | [PDF](https://arxiv.org/pdf/2602.11656v1)

**作者:** Seo Hyun Kim `[一作]` (Sungkyunkwan University), Il Yong Chun `[通讯]` (Sungkyunkwan University)

**通讯引用:** 18459 | [OpenAlex ID](https://openalex.org/A5101911180)

**关键词:** `Computer Vision and Pattern Recognition` `Autonomous Driving` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Vision Language Model` `Multimodality` `Point Cloud` `Image`

**🎯 论文内容**

提出了一种名为SToRM的监督式多模态大语言模型（MLLM）视觉令牌裁减框架，用于端到端自动驾驶；

**💡 创新点**

创新点在于：①使用来自全令牌LLM的注意力分数作为伪监督信号训练重要性预测器；②设计了轻量级滑动窗口MLP‑Mixer重要性预测器；③构造anchor‑context合并（ACM）模块，通过硬分配将上下文令牌聚合到重要anchor上；

**🔧 技术方法**

采用的技术包括：多模态LLM（如LLaVA/TinyLLaVA）、视觉编码器（CNN+融合Transformer）、MLP‑Mixer与滑动窗口混合、Gumbel‑Softmax与STE硬分配、端到端的waypoint预测与PID控制、以及多任务损失（重要性损失+waypoint损失）；

**📊 数据集**

实验使用CARLA仿真器生成的LangAuto基准数据集，包含多视角RGB、LiDAR点云、导航指令与控制命令；

**📈 对比分析**

与使用全令牌的LMDrive、现有基于Q‑Former、ToMe、LLaVA‑PruMerge等方法在相同视觉令牌预算（120）下对比，SToRM在LangAuto‑Long/Short/Tiny路线上实现了DS、RC、IS均不劣于全令牌模型，且在算力（FLOPs）和内存上分别可降低30×/16×（大模型）或6.5×/4.3×（小模型）；

**⚠️ 局限性**

局限性：仍依赖大量计算资源的LLM；伪监督信号可能无法完全捕捉所有场景下的关键信息；硬分配可能导致某些上下文信息丢失；需要在真实车辆硬件上进一步验证实时性和鲁棒性。

---

## 570. Evaluating Few-Shot Temporal Reasoning of LLMs for Human Activity Prediction in Smart Environments

**arXiv ID:** 2602.11176 | [PDF](https://arxiv.org/pdf/2602.11176v1)

**作者:** Maral Doctorarastoo `[一作]` (Carnegie Mellon University), Christopher McComb `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2275 | [OpenAlex ID](https://openalex.org/A5005893539)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Time Series` `Sequential`

**🎯 论文内容**

本文研究在智能家居场景下，少量示例（few-shot）提示对大型语言模型（LLM）进行下一步活动及其持续时间预测的影响。

**💡 创新点**

创新点在于将检索增强提示（retrieval‑augmented prompting）与多维上下文（时间、空间、行为历史、个体画像）相结合，系统评估不同示例数量对LLM预测准确率、时间误差和序列一致性的增益，并发现零样本即已具备强大的时间推理能力。

**🔧 技术方法**

技术包括：LLM（ChatGPT‑4）在JSON模式下推理、最大边缘相关性（MMR）检索、结构化解码、动态时间规整（DTW）序列评估，以及传统马尔可夫基线对比。

**📊 数据集**

数据集为CASAS Aruba智能家居数据集，包含219天、11类活动的单人住宅传感器日志。

**📈 对比分析**

与基于时间马尔可夫转移矩阵的统计基线相比，LLM在零样本即可实现60%以上的DTW改进；1–2个示例可将准确率提升约3–4个百分点，持续时间MAE从39.6↓至34.9分钟；超过3个示例增益趋于饱和，性能波动不大。

**⚠️ 局限性**

局限性在于仅针对单一家庭、单个居民的数据进行验证，无法证明在多居民、不同文化或更复杂传感器环境中的泛化能力；示例数对不同类别的影响也表现出不均衡，需进一步研究示例选择与类平衡策略。

---

## 571. Manifold-Aware Temporal Domain Generalization for Large Language Models

**arXiv ID:** 2602.11965 | [PDF](https://arxiv.org/pdf/2602.11965v1)

**作者:** Yiheng Yao `[一作]` (University of Tokyo), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 6725 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `Machine Learning` `Domain Adaptation` `Recurrent Neural Network` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Text` `Time Series`

**🎯 论文内容**

提出了 MaT-LoRA 框架，用流形约束的低秩 LoRA 分解来实现大型语言模型在时间域的泛化；

**💡 创新点**

创新点在于将时间域泛化的低维流形假设迁移到参数增量空间，并在 LoRA 子空间中共享固定基底，只用一个时间可变核心来刻画整个演化过程；

**🔧 技术方法**

采用 LoRA、低秩因子分解、流形约束、连续线性动力系统（Lin-dym）、递归神经网络（Markv）和多层感知机（Non-lin）等技术；

**📊 数据集**

使用旋转 2-Moons 合成数据集以及 AIC、NewsCLS、Yelp 三个真实时间错位数据集，并在 DistilBERT、Qwen3-0.6B、TinyLLaMA-1.1B、LLaMA3-8B 等 LLM 预训练模型上进行实验；

**📈 对比分析**

与离线、增量微调、最近域等基线对比，评估指标为分类准确率、训练时长和推理延迟；MaT-LoRA 在准确率上提升约 1–3%（在所有数据集上均为最优），训练时间略增，推理延迟与基线相当；

**⚠️ 局限性**

局限性包括：仍假设更新保持低秩并可用共享基底近似，无法充分捕捉高度非线性或突变式漂移；核心动态模型需要人工选择（线性、RNN、MLP），导致模型配置较为繁琐。

---

## 572. When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification

**arXiv ID:** 2602.11199 | [PDF](https://arxiv.org/pdf/2602.11199v1)

**作者:** Jiale Zhao `[一作]` (Chongqing University of Posts and Telecommunications), Lu Cheng `[通讯]` (University of Illinois Chicago)

**通讯引用:** 4012 | [OpenAlex ID](https://openalex.org/A5022914600)

**关键词:** `Computation and Language` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Reinforcement Learning` `Text` `Biomedical Data` `Benchmark`

**🎯 论文内容**

构建了AskBench benchmark，将单轮问答转换为多轮交互并加入检查点；提出基于rubric的RLVR训练方案提升LLM的澄清与纠错能力。

**💡 创新点**

1) 自动生成多轮对话实例与检查点；2) 统一的判定循环；3) 以检查点为奖励的RL训练；4) 两个维度(AskMind、AskOverconfidence)。

**🔧 技术方法**

使用大型语言模型（Qwen、GPT、Gemini）作为评判器和参与者；采用RLVR（Verifier-based reward model）和GRPO进行优化；利用结构化rubric与奖励函数；实现多轮评估循环。

**📊 数据集**

Math500、MedQA、BBH、GPQA-d等标准问答数据集；使用HealthBench子集进行开放式评估；训练池来自Dapo、MedMCQA；通过自动化流程生成AskBench实例。

**📈 对比分析**

与单轮SFT、Prompting（FATA）、AskToAct、Self-Alert等基线对比；在AskMind上准确率从0.33提升至0.62、覆盖率从0.21提升至0.68；在AskOverconfidence覆盖率从0.19提升至0.89；在跨域单轮QA中保持或提升准确率。

**⚠️ 局限性**

对话仅基于已有问答构造，缺乏真实用户行为；用户模拟器为LLM，可能过于理想化；未涵盖多轮目标演化与更丰富的交互；仅关注缺失信息和误导性陈述；缺少系统风格与同理心分析；领域覆盖主要集中在数学与医学。

---

## 573. CausalAgent: A Conversational Multi-Agent System for End-to-End Causal Inference

**arXiv ID:** 2602.11527 | [PDF](https://arxiv.org/pdf/2602.11527v1)

**作者:** Jiawei Zhu `[一作]` (Guangdong University of Technology), Ruichu Cai `[通讯]` (Guangdong University of Technology)

**通讯引用:** 2569 | [OpenAlex ID](https://openalex.org/A5076948208)

**关键词:** `Artificial Intelligence` `Large Language Model` `Supervised Fine-Tuning` `Retrieval-Augmented Generation` `Tabular`

**🎯 论文内容**

提出了一个对话式多智能体系统CausalAgent，实现从数据上传、预处理、因果结构学习到报告生成的端到端因果推断流程。

**💡 创新点**

创新点在于将多智能体系统、检索增强生成（RAG）与模型上下文协议（MCP）结合，完整建模因果分析工作流，并通过自然语言交互显著降低技术门槛。

**🔧 技术方法**

技术实现包括使用GLM‑4.6 LLM为核心，LangGraph调度多智能体；通过RAG检索因果知识库；SFT精调模型指令遵循；MCP统一工具调用；以及PC与OLC等算法完成结构学习。

**📊 数据集**

以Sachs蛋白质信号通路数据集（11个磷酸化蛋白）作为演示案例进行实验。

**📈 对比分析**

通过案例演示，系统自动选择PC算法、生成因果图并输出报告；与传统单模型或手动工具（如DoWhy、Causal‑learn）相比，在交互性和自动化程度上更优，但未给出定量性能指标。

**⚠️ 局限性**

局限性包括：目前不支持隐含混杂、计量因果效应估计和反事实推理；缺乏专家反馈机制，可能在高风险领域出现安全与可靠性问题。

---

## 574. Efficient Analysis of the Distilled Neural Tangent Kernel

**arXiv ID:** 2602.11320 | [PDF](https://arxiv.org/pdf/2602.11320v1)

**作者:** Jamie Mahowald `[一作]` (Los Alamos National Laboratory), Michael Geyer `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 4925 | [OpenAlex ID](https://openalex.org/A5013291349)

**关键词:** `Machine Learning` `Knowledge Distillation` `Computational Efficiency` `Convolutional Neural Network` `Image`

**🎯 论文内容**

提出了结合数据集蒸馏、Johnson–Lindenstrauss随机投影和梯度蒸馏的压缩管线，构造蒸馏的神经切线核（DNTK），在保持核结构的同时大幅降低NTK的计算与存储成本。

**💡 创新点**

创新点在于把数据维度压缩与梯度子空间压缩统一起来，证明并实现了NTK构造复杂度可降低多达5个数量级，同时在理论和实验上验证了低秩核的可行性。

**🔧 技术方法**

使用了WMDD数据蒸馏、随机正交投影（JL）、基于谱聚类的梯度蒸馏算法以及核岭回归（KRR）等技术。

**📊 数据集**

实验基于ImageNette（ImageNet子集）和ResNet‑18模型，数据集约500个精炼样本。

**📈 对比分析**

与传统NTK、随机投影、采样基线对比，DNTK在保持约92%原始精度的同时将计算和存储成本降低4–6个数量级；在仅5个合成梯度下即可达到76%的分类准确率。

**⚠️ 局限性**

主要局限是依赖惰性训练（近似线性）和预训练模型，对不同网络架构或更大规模数据的泛化尚未充分验证。

---

## 575. Cooperation Breakdown in LLM Agents Under Communication Delays

**arXiv ID:** 2602.11754 | [PDF](https://arxiv.org/pdf/2602.11754v1)

**作者:** Keita Nishimoto `[一作]` (University of Tokyo), Ichiro Sakata `[通讯]` (University of Tokyo)

**通讯引用:** 3842 | [OpenAlex ID](https://openalex.org/A5071470375)

**关键词:** `Multiagent Systems` `Transformer` `Large Language Model` `Prompt Engineering` `Text` `Sequential`

**🎯 论文内容**

提出 FLCOA 框架并通过 LLM 代理在带通信延迟的连续囚徒困境中进行仿真实验，探究延迟对合作的影响。

**💡 创新点**

首次将基础设施层（通信延迟、资源分配）纳入多智能体合作框架，并发现延迟对合作率产生非单调 U 形效应。

**🔧 技术方法**

利用 GPT‑5‑mini、Claude Sonnet 4 等大型语言模型，配合自定义提示实现代理决策；构造带延迟的连续囚徒困境模型进行仿真。

**📊 数据集**

实验使用模拟数据；代理具备固定的 Big‑Five 个性特质（A=1、C=-1、N=1），并通过 10 次实验采样得到合作/背叛/剥削比例。

**📈 对比分析**

对比两种 LLM，统计在不同延迟下的合作、背叛和剥削比例；结果显示合作率呈 U 形变化，剥削率呈倒 U 形，验证了延迟对合作的非线性影响。

**⚠️ 局限性**

实验仅涉及两名代理、单一游戏模型，未考虑更复杂场景或多方竞争；延迟之外的因素未被充分探索，结果可能不完全适用于真实世界部署。

---

## 576. Predicting the post-wildfire mudflow onset using machine learning models on multi-parameter experimental data

**arXiv ID:** 2602.11194 | [PDF](https://arxiv.org/pdf/2602.11194v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## 577. How Smart Is Your GUI Agent? A Framework for the Future of Software Interaction

**arXiv ID:** 2602.11514 | [PDF](https://arxiv.org/pdf/2602.11514v1)

**作者:** Sidong Feng `[一作]` (Chinese University of Hong Kong), Chunyang Chen `[通讯]` (Technical University of Munich)

**通讯引用:** 4123 | [OpenAlex ID](https://openalex.org/A5075639297)

**关键词:** `Software Engineering` `Robotic Intelligence` `Safty and Privacy` `Explainability and Interpretability` `Large Language Model` `Agentic AI` `Multimodality` `Benchmark` `Review/Survey Paper`

**🎯 论文内容**

提出了GUI Agent Autonomy Levels（GAL）框架，定义了从无自动化到完全自动化的六个自治级别，并讨论了当前技术水平与未来发展挑战。

**💡 创新点**

首次将类似汽车行业的SAE级别概念应用于GUI代理，提供统一的分层评估语言和基准；明确区分从最小辅助到全自动化的六个层级，为研究者和工程师提供可量化的进展衡量。

**🔧 技术方法**

综述并采用了大规模多模态模型（LLM）驱动的GUI代理技术，包括视觉感知、自然语言理解和动作执行的集成；提及传统工具（Selenium、UI Automator、AppleScript）以及现代代理（WebAgent、UI‑TARS、Claude Computer Use Agent、ChatGPT Atlas 等）。

**📊 数据集**

引用公开基准 OSWorld、AndroidWorld 等评估环境，但论文本身未提供实验数据。

**📈 对比分析**

通过对比现有代理在不同自治级别的功能描述进行定性评估；未给出数值性能指标，指出目前大多数代理仍处于 Level 1–3，已接近 Level 4 但尚未实现真正的跨平台普适性。

**⚠️ 局限性**

当前代理主要受限于手工配置、脆弱的 GUI 选择器、缺乏长期状态跟踪与真正的自主决策；安全、隐私、个性化与可解释性等挑战仍未充分解决。

---

## 578. UMAP Is Spectral Clustering on the Fuzzy Nearest-Neighbor Graph

**arXiv ID:** 2602.11662 | [PDF](https://arxiv.org/pdf/2602.11662v1)

**作者:** Yang Yang `[一作]` (University of Queensland), Yang Yang `[通讯]` (University of Queensland)

**通讯引用:** 111225 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `Machine Learning` `Contrastive Learning`

**🎯 论文内容**

证明了UMAP等价于在模糊k近邻图上执行谱聚类。

**💡 创新点**

将UMAP的负采样优化映射为对比学习，并用谱聚类证明其等价性。

**🔧 技术方法**

对比学习、图拉普拉斯、负采样SGD、谱初始化等技术。

**📊 数据集**

未使用具体数据集，主要是理论推导。

**📈 对比分析**

没有实验比较，主要通过理论等价性说明性能优越性。

**⚠️ 局限性**

仅对高斯核精确等价，Cauchy核仅一阶近似；假设图结构理想，缺乏实验验证。

---

## 579. An Educational Human Machine Interface Providing Request-to-Intervene Trigger and Reason Explanation for Enhancing the Driver's Comprehension of ADS's System Limitations

**arXiv ID:** 2602.11507 | [PDF](https://arxiv.org/pdf/2602.11507v1)

**作者:** Ryuji Matsuo `[一作]` (Nara Institute of Science and Technology), Takahiro Wada `[通讯]`

**关键词:** `Human-Computer Interaction` `Autonomous Driving` `Explainability and Interpretability` `Text` `Audio`

**🎯 论文内容**

在实验室驾驶模拟器中，提出并评估了一个语音式教育型人机交互界面，该界面在 ADS Level 3 系统发出请求干预（RtI）时同时提供触发提示与原因说明，帮助驾驶员理解系统限制并主动接管。

**💡 创新点**

其创新点在于将 RtI 触发提示与系统限制原因解释结合成一套持续教育机制，显著提升驾驶员的心理模型准确性、提前接管行为及事故防范。

**🔧 技术方法**

利用基于文本转语音（Microsoft Nanami）生成的声音提示、UC-win/Road 驾驶仿真平台以及 NASA‑TLX、事后理解测验等工具实现对 HMI 的设计与评估。

**📊 数据集**

数据来源为 45 名年轻日本驾驶员在 14 条预设情境（含学习与测试阶段）中产生的驾驶记录与问卷数据，而非公开数据集。

**📈 对比分析**

通过三组（无提示、仅触发提示、触发+原因提示）之间的对照实验，结果显示触发+原因提示组在理解测试中得分最高、碰撞率最低、接管时间最早，且与理解分数呈正相关，且未增加工作负荷。

**⚠️ 局限性**

局限性包括样本量有限、受试者年龄单一且仅来自日本、情境顺序未平衡、未考虑非驾驶任务情境、仅检验 10 s 预警时限且使用仿真而非真实道路。

---

## 580. LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training

**arXiv ID:** 2602.11686 | [PDF](https://arxiv.org/pdf/2602.11686v1)

**作者:** Xinyi Liu `[一作]` (Peking University), Bin Cui `[通讯]` (Peking University)

**通讯引用:** 12951 | [OpenAlex ID](https://openalex.org/A5062357883)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Computational Efficiency` `Optimization` `Transformer` `Mixture of Experts` `Large Language Model` `Text`

**🎯 论文内容**

本文提出了LAER-MoE系统，通过完全分片专家并动态重排，显著提升了Mixture-of-Experts模型的训练效率。

**💡 创新点**

创新点在于引入Fully Sharded Expert Parallelism（FSEP）实现实时专家重排、设计智能负载平衡规划器以及融合参数预取与梯度散射，消除重排开销并在每一步自适应调度专家与令牌路由。

**🔧 技术方法**

主要技术包括FSEP并行范式、基于启发式贪心算法的专家重排与令牌调度、全异步All-to-All通信优化、定制的CUDA All-to-All核、异构并行与细粒度再计算，以及Triton/CPU多进程规划器。

**📊 数据集**

实验使用Mixtral、Qwen等多种LLM架构，在WikiText和C4数据集上进行训练。

**📈 对比分析**

与Megatron、FSDP+EP以及FlexMoE等SOTA方法对比，LAER-MoE在8‑节点A100集群上实现了最高1.69×、平均1.20×的加速，比FlexMoE提升1.39×。

**⚠️ 局限性**

局限性包括在负载均衡已足够的平衡场景下提升有限，以及重排规划器虽高效但仍为启发式，无法保证全局最优，且在极大规模集群时仍需验证通信开销与内存峰值。

---

## 581. Calibration and Evaluation of Car-Following Models for Autonomous Shuttles Using a Novel Multi-Criteria Framework

**arXiv ID:** 2602.11517 | [PDF](https://arxiv.org/pdf/2602.11517v1)

**作者:** Renan Favero `[一作]` (University of Florida), Lily Elefteriadou `[通讯]` (University of Florida)

**通讯引用:** 4881 | [OpenAlex ID](https://openalex.org/A5057923918)

**关键词:** `Emerging Technologies` `Autonomous Driving` `Optimization` `Convolutional Neural Network` `Recurrent Neural Network` `Transformer` `Reinforcement Learning` `Time Series` `Sequential`

**🎯 论文内容**

开发并校准了多种基于机器学习和物理模型的自动穿梭车跟车模型，并提出多指标评估框架。

**💡 创新点**

首次将ML算法与多维度评估框架用于自动穿梭车跟车建模，系统比较不同模型的精度、稳定性与轨迹相似性。

**🔧 技术方法**

采用XGBoost、FNN、LSTM、CNN、Transformer等ML算法，配合遗传算法、Optuna调参，并使用基于Z-score的多维度评估方法。

**📊 数据集**

使用佛罗里达奥兰多Lake Nona地区收集的实测自动穿梭车与前车GPS轨迹数据。

**📈 对比分析**

通过Z-score归一化错误、稳定性、相似性三大类指标进行多维度对比，XGBoost与FNN表现最佳，传统IDM、SVM表现最差。

**⚠️ 局限性**

数据集局限于单一场景与有限时间，缺乏多样化交通条件与换道行为，模型在其他环境的泛化能力未知。

---

## 582. Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use

**arXiv ID:** 2602.11541 | [PDF](https://arxiv.org/pdf/2602.11541v1)

**作者:** Hanbing Liu `[一作]` (Renmin University of China), Qi Qi `[通讯]` (Renmin University of China)

**通讯引用:** 78765 | [OpenAlex ID](https://openalex.org/A5100637146)

**关键词:** `Artificial Intelligence` `Large Language Model` `Agentic AI` `World Model` `Text`

**🎯 论文内容**

提出了 INTENT，一个推理时预算约束工具使用的轻量级规划框架；

**💡 创新点**

创新点在于通过意图感知的语言世界模型和几何成本校准，将工具调用成功与否抽象为意图满足度，从而在高随机性环境中实现稳健的预算控制；

**🔧 技术方法**

采用语言世界模型（LWM）进行工具结果模拟，意图预测器与条件生成器分离，结合几何分布成本估计、风险调节参数 γ，以及缓存的理想轨迹回溯，避免了昂贵的 MCTS 搜索；

**📊 数据集**

实验基于 StableToolBench（为每个工具合成价格并加入 20 个检索工具），并使用 GPT‑4.1 mini 与 GPT‑5 nano 作为后端 LLM；

**📈 对比分析**

与 Prompt、DFSDT、BTP、BATS 等基线对比，INTENT 在严格满足预算的前提下，保持最高的通过率、预算最佳通过率与竞争对手相比提升 10‑20% 以上，同时推理时间和 Token 消耗仅略高于 Prompt；

**⚠️ 局限性**

局限性包括需收集足够的工具交互日志来训练世界模型，对极端动态市场（如工具功能突变）仍可能出现误估；此外在极小预算或高度波动价格下，几何成本估计的保守性可能导致性能下降。

---

## 583. WaveFormer: Wavelet Embedding Transformer for Biomedical Signals

**arXiv ID:** 2602.12189 | [PDF](https://arxiv.org/pdf/2602.12189v1)

**作者:** Habib Irani `[一作]` (Texas State University), Vangelis Metsis `[通讯]` (Texas State University)

**通讯引用:** 2500 | [OpenAlex ID](https://openalex.org/A5031305480)

**关键词:** `Machine Learning` `Classification` `Transformer` `Biomedical Data` `Time Series`

**🎯 论文内容**

提出 WaveFormer，一种在嵌入层和位置编码层同时使用小波变换的 Transformer，用于生物医学信号分类。

**💡 创新点**

创新点在于双阶段小波增强：1）将多尺度离散小波变换（DWT）与原始时域补丁并行融合，生成频域与时域并重的 token；2）设计 Dynamic Wavelet Positional Encoding（DyWPE）根据信号频谱动态调制位置嵌入，提升对长序列和多频带信息的感知。

**🔧 技术方法**

核心技术包括多通道 DWT、DyWPE、T5 风格分桶相对位置编码、PatchTST 版 Transformer 结构以及常规多头自注意力与前馈网络。

**📊 数据集**

使用八个 UEA 数据集，包括三类 HAR（WalkingSittingStanding、UWaveGestureLibraryAll）、五类 EEG（FaceDetection、FingerMovements、SelfRegulationSCP1、SelfRegulationSCP2、MotorImagery）和一类心跳音频，序列长度 50–3000，通道数 1–144。

**📈 对比分析**

与 ResNet、InceptionTime、TST、PatchTST 和 ConvTran 等现有方法在同一数据集上进行对比；WaveFormer 在 7/8 数据集上获得最高准确率（如 WSS 91.3%、UWG 93.0%、Heartbeat 78.4%），平均提升约 1–3%，且在长序列上显著优于基线。

**⚠️ 局限性**

局限性包括对非生物医学序列的验证不足、对极长序列（>10k 步）与高通道数（>100 通道）的计算开销尚未系统评估，以及对小样本情况下的泛化能力仍需进一步探究。

---

## 584. Sparse Semantic Dimension as a Generalization Certificate for LLMs

**arXiv ID:** 2602.11388 | [PDF](https://arxiv.org/pdf/2602.11388v1)

**作者:** Dibyanayan Bandyopadhyay `[一作]` (Indian Institute of Technology Patna), Asif Ekbal `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 9443 | [OpenAlex ID](https://openalex.org/A5085370631)

**关键词:** `Machine Learning` `Transformer` `Large Language Model` `Auto Encoder` `Text`

**🎯 论文内容**

提出了一种新的复杂性度量——稀疏语义维度（SSD），用于解释大型语言模型（LLMs）在参数数量远超训练样本的情况下如何实现良好的泛化能力。

**💡 创新点**

创新点在于将复杂性度量从高维参数空间转移到低维稀疏流形，强调模型内部表示的稀疏性而非参数总数对泛化能力的影响。

**🔧 技术方法**

使用了稀疏自编码器（SAE）来分析模型的潜在结构，并通过该框架推导出泛化界限。

**📊 数据集**

在实验中使用了GPT-2 Small和Gemma-2B模型，验证了提出的理论框架。

**📈 对比分析**

与传统的基于参数计数的泛化界限相比，提出的界限在样本量达到约50,000时变得非空，显示出显著的改进，尤其是在较大的模型（Gemma-2B）中需要更少的校准样本来识别其活跃流形。

**⚠️ 局限性**

限制在于该框架假设了稀疏性和特征池的有效性，未来的研究需要探讨在动态或上下文依赖的概念池下的泛化能力。

---

## 585. Coupler Position Optimization and Channel Estimation for Flexible Coupler Aided Multiuser Communication

**arXiv ID:** 2602.11319 | [PDF](https://arxiv.org/pdf/2602.11319v1)

**作者:** Xiaodan Shao `[一作]` (University of Waterloo), Xuemin Shen `[通讯]`

**关键词:** `Information Theory` `Optimization`

**🎯 论文内容**

提出分布式柔性耦合器（FC）天线阵列，通过可移动被动耦合器实现机械波束成形，并联合优化耦合器位置与数字预编码；设计了基于分布式局部处理单元（LPU）的信道估计算法，包括集中式和分布式版本。

**💡 创新点**

创新点在于：①仅移动被动耦合器而不移动主动天线，降低硬件成本与能耗；②引入分布式 LPU 结构，减轻中心处理器负担；③利用 SCA 与梯度投影实现耦合器位置的可行分布式优化；④在柔性耦合器系统上提出结构化 pilot 与稀疏重建的信道估计方法，兼顾精度与开销。

**🔧 技术方法**

采用电磁耦合模型、MMSE 预编码、梯度投影的分布式 SCA 优化、稀疏恢复（OMP）、pilot 相关与分布式统计融合等技术；还利用了多用户 MIMO 信道模型与机械波束成形的电路方程。

**📊 数据集**

使用仿真数据集：随机生成多路径信道（K、M、N、Lk、G 等参数设置），并在不同的天线与耦合器布局下进行 Monte‑Carlo 评估；未使用公开真实数据集。

**📈 对比分析**

与单纯主动天线阵列、固定耦合器、负载调制的寄生天线阵列、全主动天线阵列以及完整测量法等基线进行比较；仿真显示 FC 阵列在相同功率下可逼近全主动阵列的吞吐量，且在信道估计中 NMSE 低于集中式完整测量，pilot 开销显著降低。

**⚠️ 局限性**

局限性包括：①耦合矩阵计算对物理尺寸与位置敏感，需精确校准；②分布式优化和信道估计仍比集中式方法略逊；③移动范围受限导致性能上限；④当 M、G 过大时，计算与通信开销仍显著。

---

## 586. Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering

**arXiv ID:** 2602.11183 | [PDF](https://arxiv.org/pdf/2602.11183v1)

**作者:** Yin Tang `[一作]` (Central South University), Deyu Zhang `[通讯]` (Central South University)

**通讯引用:** 3848 | [OpenAlex ID](https://openalex.org/A5100705648)

**关键词:** `Robotics` `Autonomous Driving` `Optimization` `Recurrent Neural Network` `Large Language Model` `Simultaneous Localization and Mapping` `Multimodality` `Time Series` `Benchmark`

**🎯 论文内容**

本文提出一种基于递归贝叶斯状态估计的NeuroKalman框架，用于解决无人机连续导航中的状态漂移问题；

**💡 创新点**

创新点在于将注意力检索等价于核密度估计作为测量似然，形成先验预测与历史测量校正的双流程，并通过可学习的Kalman增益动态融合两者；

**🔧 技术方法**

技术手段包括GRU运动动力学预测、基于多模态LLM的测量建模、注意力检索与KDE的数学映射、可学习Kalman增益门控以及经验记忆银行；

**📊 数据集**

使用公开的TravelUAV导航基准（包含多种地形与目标物体），并在AirSim仿真环境下进行实验；

**📈 对比分析**

在仅用10%训练数据微调的设置下，NeuroKalman在所有指标（NE、SR、OSR、SPL）上均显著优于TravelUAV、OpenVLN、NavFoM等强基线，尤其在长航程与未见地图/物体场景中表现更为突出；

**⚠️ 局限性**

局限性在于先验预测使用GRU时可能在极长时域内信息衰减；记忆检索与Kalman更新增加了计算与存储开销；整体框架需进一步验证在真实无人机硬件上的实时性与鲁棒性。

---

## 587. CAAL: Confidence-Aware Active Learning for Heteroscedastic Atmospheric Regression

**arXiv ID:** 2602.11825 | [PDF](https://arxiv.org/pdf/2602.11825v1)

**作者:** Fei Jiang `[一作]` (University of Manchester), Zhonghua Zheng `[通讯]` (University of Manchester)

**通讯引用:** 2481 | [OpenAlex ID](https://openalex.org/A5059878178)

**关键词:** `Machine Learning` `Transformer` `Tabular`

**🎯 论文内容**

提出一种置信感知主动学习框架CAAL，利用异方差回归从低成本观测中高效获取高成本粒子属性标签。

**💡 创新点**

创新点在于将预测均值和噪声分离训练，并用预测的随机噪声作为可靠性信号动态加权不确定性，以同时抑制不可约噪声并提升信息增益。

**🔧 技术方法**

采用深度集成（Deep Ensemble）与分离的MSE+NLL损失、置信门控采样函数，并使用FT-Transformer作为回归骨干网络。

**📊 数据集**

使用基于PartMC模型的粒子解析模拟数据（化学与光学混合状态指标）以及北京地区黑碳表面涂层（VR）观测数据。

**📈 对比分析**

与随机、置信、阿尔雷、ALM、QBC、BALD、Coreset、BADGE、LCMD等基线比较，CAAL在主要指标R²提升约9.4%并将标注成本降低约45%，表现最优。

**⚠️ 局限性**

局限在于需先验选择超参数β与λ，对不同数据场景的泛化仍待进一步验证，并且在极高噪声区域仍可能难以完全避免噪声影响。

---

## 588. Retrieval-Aware Distillation for Transformer-SSM Hybrids

**arXiv ID:** 2602.11374 | [PDF](https://arxiv.org/pdf/2602.11374v1)

**作者:** Aviv Bick `[一作]` (Carnegie Mellon University), Albert Gu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2975 | [OpenAlex ID](https://openalex.org/A5025386668)

**关键词:** `Machine Learning` `Knowledge Distillation` `Retrieval` `Compression` `Transformer` `Mixture of Experts` `Text`

**🎯 论文内容**

对预训练 Transformer 进行检索感知蒸馏，构建仅保留检索关键注意力头并用 SSM 替代其余头的混合模型。

**💡 创新点**

创新点在于：①通过在合成 KV‑检索任务上逐头消融识别出执行 Gather‑and‑Aggregate（G&A）检索功能的注意力头；②仅保留这些头并将其余头改为 SSM，显著减少注意力头数；③证明检索功能可以从注意力迁移到 SSM 后，进一步压缩 SSM 状态维度，提升内存效率。

**🔧 技术方法**

使用 MOHAWK 蒸馏框架、改进的混合层设计、LayerNorm 对齐、聚合注意力与 SSM 的并行混合，并在训练中采用交叉熵蒸馏。

**📊 数据集**

评估数据集包括检索密集型任务（Lambada、MMLU、GSM8K、SWDE、KV‑Retrieval 等）和知识聚焦任务（PIQA、Winogrande、OpenBookQA、HellaSwag、ARC-Challenge/Easy），以及在 Llama‑3.2‑1B 与 Qwen‑2.5‑1.5B 上的蒸馏训练集。

**📈 对比分析**

与固定/逐层交替放置注意力的基线（MOHAWK、Mamba‑In‑The‑Llama 等）对比，检索感知蒸馏仅保留约 2% 的注意力头即可恢复 95%+ 的教师性能；相较于需保留 25% 以上头的基线，显著降低 KV 缓存和模型内存，提升 5–6 倍的内存效率；在检索密集任务上覆盖率和困惑度均接近教师。

**⚠️ 局限性**

局限性：①仅在 3B 以下模型验证，规模更大时检索头分布可能变化；②检索头选取依赖单一合成 KV‑检索探针，可能忽略多跳检索等场景；③未对不同层/同层 KV 共享进行蒸馏，仍有进一步压缩空间；④部分 SSM 头仍对检索有影响，限制了状态维度压缩的极限。

---

## 589. ArtContext: Contextualizing Artworks with Open-Access Art History Articles and Wikidata Knowledge through a LoRA-Tuned CLIP Model

**arXiv ID:** 2602.11349 | [PDF](https://arxiv.org/pdf/2602.11349v1)

**作者:** Samuel Waugh `[一作]`, Stuart James `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Retrieval` `Transformer` `Contrastive Learning` `Image` `Text`

**🎯 论文内容**

构建了一个弱监督管道，将开放获取的艺术史论文与画作配对，并通过LoRA对CLIP进行轻量级微调，得到PaintingCLIP模型。

**💡 创新点**

创新点在于：①大规模自动构建弱监督数据集（27,044篇论文，29,697图文对）；②利用Wikidata元数据与Sentence-BERT实现句子与画作的语义对齐；③采用LoRA在保持CLIP预训练优势的同时实现领域特定的快速适配。

**🔧 技术方法**

使用的技术包括：OpenAlex抓取、PDF转Markdown、句子分割与清洗、Sentence-BERT语义嵌入、Wikidata模板生成查询、LoRA低秩适配、CLIP ViT‑B/32的对比学习。

**📊 数据集**

数据集：27,044篇开放获取艺术史PDF（覆盖450位艺术家），构成29,697张画作-句子对；配合Wikidata画作元数据和OpenAlex的文章检索信息。

**📈 对比分析**

与原始CLIP在基准检索任务（画作-句子匹配）进行对比，采用precision–recall曲线评估。PaintingCLIP在高精度区间显著优于CLIP，整体macro‑average precision提升可观。

**⚠️ 局限性**

局限性包括：①数据偏向主流艺术家和经典作品，稀有艺术家缺乏监督；②句子选择受PDF解析噪声与语义模糊影响；③LoRA仅更新投影层，未探索更深层适配；④CLIP固定224×224裁剪和77词截断限制对细节描述的捕捉。

---

## 590. Achievability Bounds of Coding with Finite Blocklength for Gaussian Broadcast Channels

**arXiv ID:** 2602.11986 | [PDF](https://arxiv.org/pdf/2602.11986v1)

**作者:** Ayşe Ünsal `[一作]` (Eurecom), Jean-Marie Gorce `[通讯]` (Institut National des Sciences Appliquées de Lyon)

**通讯引用:** 3419 | [OpenAlex ID](https://openalex.org/A5054469607)

**关键词:** `Information Theory` `Gaussian Splatting` `Physics Related`

**🎯 论文内容**

本文研究了有限码长下高斯广播信道（Gaussian BC）中脏纸编码（dirty paper coding, DPC）的可实现性能，提出了两种可实现界：依赖测试（dependence testing）界和 κβ 界，并给出了相应的信道色散（dispersion）表达式。

**💡 创新点**

创新点在于：①将 Polyanskiy 等人针对单用户的非渐近可实现界推广到多用户广播信道；②针对 DPC 推导出完整的色散项，揭示了在有限码长下 DPC 相对于传统叠加编码（superposition coding）的优势；③通过阈值解码实现了可实现界的计算，提供了更紧凑的误差概率上界。

**🔧 技术方法**

使用的主要技术包括：脏纸编码理论、依赖测试与 κβ 界的概率界定、阈值解码、信息密度与色散分析、正态分布与卡方分布的矩推导、Bessel 函数与变差-伽马分布的使用。

**📊 数据集**

该工作为纯理论分析，无使用具体数据集；所有结果均基于高斯信道模型和随机码本的理论推导。

**📈 对比分析**

通过与传统叠加编码的误差概率界比较，实验证明在有限码长且用户数≥2 的场景下，脏纸编码在误差率和码率方面优于叠加编码；具体数值通过数值积分或 Monte Carlo 估计得到，显示了 DPC 在非渐近 regime 下的性能提升。

**⚠️ 局限性**

局限性包括：①界限相对保守，尤其是 κβ 界的闭式表达不易进一步简化；②推导假设独立同分布的信号与噪声，实际系统可能存在相关或非高斯噪声；③复杂度较高，需多维积分或大样本估计，难以直接用于实际编码设计；④仅讨论两用户场景，未给出多用户扩展的具体实现细节。

---

## 591. Scale-Invariant Fast Convergence in Games

**arXiv ID:** 2602.11857 | [PDF](https://arxiv.org/pdf/2602.11857v1)

**作者:** Taira Tsuchiya `[一作]` (University of Tokyo), Shinji Ito `[通讯]` (University of Tokyo)

**通讯引用:** 4012 | [OpenAlex ID](https://openalex.org/A5111491501)

**关键词:** `Computer Science and Game Theory` `Optimization` `Reinforcement Learning`

**🎯 论文内容**

提出了一种自适应的学习动态，能够在不知道效用尺度的前提下实现两人零和游戏和多玩家一般和游戏的快速收敛。

**💡 创新点**

创新点在于将可变学习率与路径长度、停止时间分析相结合，并提出了“doubling clipping”技术，使得算法既保持尺度无关又保持尺度不变。

**🔧 技术方法**

采用了最优跟随正则化领导（OFTRL）与自适应学习率、停止时间分析、doubling clipping 等技术。

**📊 数据集**

无实证数据集，所有结果均为理论上界。

**📈 对比分析**

与已有的最优尺度已知算法相比，新的算法在外部/交换损失上取得 O(log m) / O(n^{3/2} m^{5/2} log T) 的上界，收敛速率分别为 O(log m/T) 与 O(n^{3/2} m^{5/2} log T/T)，与已知最快速率相当或更好。

**⚠️ 局限性**

限制在于多玩家情况对玩家数的依赖为 √n，且对对手的恶意扰动缺乏鲁棒性。

---

## 592. Light4D: Training-Free Extreme Viewpoint 4D Video Relighting

**arXiv ID:** 2602.11769 | [PDF](https://arxiv.org/pdf/2602.11769v1)

**作者:** Zhenghuang Wu `[一作]` (Peking University), Hao Tang `[通讯]` (Peking University)

**通讯引用:** 52880 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Restoration` `Autonomous Driving` `Optical Flow` `Video`

**🎯 论文内容**

提出了一个无训练的 4D 视频重光照框架 Light4D，能够在极端视角变化下同步控制相机轨迹与光照。

**💡 创新点**

创新点包括：① Disentangled Flow Guidance（解耦流引导）通过时间感知的融合策略在几何生成后再注入光照信息；② Temporal Consistent Attention（时序一致注意力）用双路径滑动窗口平滑键值，提升时序连贯性；③ Deterministic Coherence Regularization（确定性一致性正则）通过 CNI、GMM、FDI 等技术消除随机抖动。

**🔧 技术方法**

技术手段主要是：利用预训练的 3D 生成器 EX‑4D 与 2D 光照先验 IC‑Light；在潜在流匹配中引入时间加权融合；对注意力层做时序平滑；对生成过程进行确定性稳定化。

**📊 数据集**

使用了 100 条高质量合成视频（来自 Sora、WanVideo、Kling）作为评估基准，并在 OpenScene 实景驾驶序列中做了进一步验证；不需要额外的配对训练数据。

**📈 对比分析**

与训练基线 Light‑X 以及无训练的级联管线（EX‑4D→LAV、LAV→EX‑4D、EX‑4D+IC‑Light）相比，Light4D 在 CLIP‑Frame、HFPR、Aesthetic、PSNR、SSIM 等多项指标上均实现了最佳或第二最佳成绩，尤其在 90° 甚至 180° 的极端视角下仍保持了较高的时序连贯性与细节保留。

**⚠️ 局限性**

主要局限在于受限于基础模型（EX‑4D 的几何精度与 IC‑Light 的单帧特性），导致全局光照一致性在极端视角下仍存在挑战，且对光照复杂交互（阴影、反射）处理不足。

---

## 593. Efficient Crawling for Scalable Web Data Acquisition (Extended Version)

**arXiv ID:** 2602.11874 | [PDF](https://arxiv.org/pdf/2602.11874v1)

**作者:** Antoine Gauquier `[一作]` (École Normale Supérieure), Pierre Senellart `[通讯]` (École Normale Supérieure)

**通讯引用:** 3490 | [OpenAlex ID](https://openalex.org/A5035414136)

**关键词:** `Information Retrieval` `Reinforcement Learning` `Optimization` `Reinforcement Learning` `Tabular`

**🎯 论文内容**

针对在网站中高效检索统计数据集（SD）的需求，提出一种基于强化学习的聚焦爬虫，能够在不完整爬取整个网站的前提下，最大限度地获取目标文件。

**💡 创新点**

创新点包括：①将 SD 检索问题建模为图爬取问题并证明其 NP‑hard；②利用 HTML 页面中的 tag 路径对超链接进行相似性聚类，构造单一状态下的 Sleeping Bandit 学习框架；③开发在线 URL 分类器以快速估计页面 MIME 类型，减少 HEAD 请求；④在实验中展示该方法在多种大型网站（累计 2200 万页）上，能用 20% 的页面就收集到 90% 目标。

**🔧 技术方法**

核心技术：强化学习（多臂赌博机 UCB 与其睡眠变体 AUER）、层级可导航小世界索引（HNSW）实现 tag 路径聚类、BoW+哈希向量化、在线逻辑回归 URL 分类器、HTTP HEAD/GET 并行调度。

**📊 数据集**

实验使用 8 个官方统计机构网站（如欧盟 Eurostat、ILO、IMF 等），包含数百到数十万级 SD 文件，覆盖多语言和多种文件格式（CSV、JSON、XLS、PDF 等）。

**📈 对比分析**

与基线（随机、词频、爬行深度优先、传统 MAB 等）对比，爬虫在相同请求/数据量预算下平均获得 70–90% 的目标，而基线只能 30–50%；在大规模网站上，爬虫只需访问约 20% 的页面即可收集 90% 目标，节约 80% 以上的网络请求和带宽。

**⚠️ 局限性**

局限性：①不支持增量更新或实时重新爬取新发布的数据；②对深 Web（需表单交互）的覆盖有限；③依赖 tag 路径假设，若页面结构变化大可能失效；④需手动调节阈值 θ、α 等超参数；⑤在极大或高度动态网站上，学习收敛速度可能受限。

---

## 594. Nested Named Entity Recognition in Plasma Physics Research Articles

**arXiv ID:** 2602.11163 | [PDF](https://arxiv.org/pdf/2602.11163v1)

**作者:** Muhammad Haris `[一作]` (TIB Leibniz Information Centre for Science and Technology), Markus Stocker `[通讯]` (TIB Leibniz Information Centre for Science and Technology)

**关键词:** `Computation and Language` `Recognition` `Optimization` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Gaussian Splatting` `Text` `Biomedical Data` `Physics Related`

**🎯 论文内容**

开发并评估基于BERT-CRF的轻量级嵌套命名实体识别模型，并发布了等离子体物理研究文章的16类实体标注语料库。

**💡 创新点**

①发布了等离子体物理领域专用NNER数据集；②提出实体特定的BERT–CRF专门化训练方式；③将贝叶斯优化自动调参融入模型，提升低频实体识别。

**🔧 技术方法**

采用BERT预训练语言模型、CRF序列标注层、实体特定多头独立模型以及Gaussian Process贝叶斯优化进行超参调优。

**📊 数据集**

自建等离子体物理语料库（30篇全文+500项专利，10,272句，16类实体）以及公开GENIA和Chilean Waiting List用于跨域评估。

**📈 对比分析**

在等离子体数据集上与八个现有NNER基线（Tree-CRF、Pyramid、Triaffine、MLC等）进行严格span评估，BO调参后BERT-CRF F1=0.68，召回率0.74，虽略低于Tree-CRF最高0.69，但与最优模型相近且模型更轻量；在GENIA和Chilean上也获得与最强模型相当的F1≈0.77-0.79。

**⚠️ 局限性**

对罕见实体的识别仍有挑战，单个BERT–CRF模型在极复杂层级结构下略逊Tree-CRF；仅在单一学科语料验证，跨学科泛化仍需进一步测试。

---

## 595. PACE: Prefix-Protected and Difficulty-Aware Compression for Efficient Reasoning

**arXiv ID:** 2602.11639 | [PDF](https://arxiv.org/pdf/2602.11639v1)

**作者:** Ruixiang Feng `[一作]` (University of Electronic Science and Technology of China), Shuo Shang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 5585 | [OpenAlex ID](https://openalex.org/A5102754146)

**关键词:** `Computation and Language` `Compression` `Optimization` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Large Language Model` `Text`

**🎯 论文内容**

提出双层框架（Prefix‑Protected Optimization + Difficulty‑Aware Penalty）以压缩语言推理模型的生成长度，既保持又提升推理准确率。

**💡 创新点**

创新点：①在序列级别使用冻结前缀策略保护初始推理步骤并逐步解锁，避免“过度压缩”关键步骤；②在组级别根据任务难度动态调节长度惩罚，防止对复杂任务过度压缩；③将两层控制结合到层级监督的RL训练中，实现高效推理。

**🔧 技术方法**

采用强化学习（GRPO）+ 前缀保护策略 + 动态长度惩罚（基于pass rate） + min–max长度惩罚 + 归一化奖励 + KL‑divergence 正则等技术。

**📊 数据集**

训练集：Skywork‑o1；评估集：AIME24、AIME25、MATH500、GSM8K；OOD测试集：GPQA‑D、LiveCodeBench‑v6、IF‑Eval；使用 DeepSeek‑R1‑Distill‑Qwen 1.5B/7B 模型。

**📈 对比分析**

与训练无控制、Uniform‑Length‑Penalty、AdaptThink、Efficient‑R、O1‑Pruner、DAST 等基线对比。实验显示：7B 模型 token reduction 55.7% 并提升 0.6% 准确率；1.5B 模型提升 4.1% 准确率；OOD 同时获得 1.1–4.6% 准确率提升与 14.5–37% 长度缩减，显著突破 Pareto 前沿。

**⚠️ 局限性**

局限性：仍有约 40% tokens 用于推理，效率可进一步提升；OOD 泛化需更系统研究；在极端长前缀时可能出现“shortcut learning”风险；缺乏对更广泛域、提示与分布偏移的全面评估。

---

## 596. Future Mining: Learning for Safety and Security

**arXiv ID:** 2602.11472 | [PDF](https://arxiv.org/pdf/2602.11472v1)

**作者:** Md Sazedur Rahman `[一作]` (Missouri University of Science and Technology), Sanjay Madria `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 4758 | [OpenAlex ID](https://openalex.org/A5012569039)

**关键词:** `Cryptography and Security` `Federated Learning` `Safty and Privacy` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Multimodality`

**🎯 论文内容**

提出统一的智能安全与安全架构，融合多模态感知、DTN通信、能源感知、强化学习、联邦学习与安全监测，设计并阐述了五个核心模块及其实验平台。

**💡 创新点**

创新点包括：① 将DTN、能源感知、机器无学习攻击检测和后门监测等多技术集成为完整安全织物；② 提出了模块化的统一架构和专门的实验平台；③ 针对矿山环境提出了分布式学习鲁棒性与模型完整性的新方法。

**🔧 技术方法**

采用的技术有：多模态感知（RGB、LiDAR、热像、气体传感器）+ 空间‑时序 Transformer；安全联邦学习与 TrustFED‑LFD；机器无学习攻击检测；后门攻击监测；DTN store‑carry‑forward 通信；强化学习导航；设备健康预测与 IoT 数据融合。

**📊 数据集**

文中未给出具体数据集，参考了 DIS‑Mine、CAV‑AD、MineDetect、OGLe‑Mine 等公开/自建实验数据，计划使用矿山现场采集的多模态数据进行验证。

**📈 对比分析**

本文为概念性工作，未给出定量实验。预期在实验平台上与传统单模态感知、非安全联邦学习、无 DTN 等方案对比，评估定位准确率、攻击检测率、能源使用等指标，并预期能显著提升定位精度、模型鲁棒性和设备预警准确率。

**⚠️ 局限性**

局限性：① 未在真实矿山环境中验证；② 缺乏统一的数据集和标准评估；③ DTN 通信延迟和不稳定性对实时性影响未知；④ 能源感知模型与硬件实现尚未完成；⑤ 机器无学习攻击防御机制仍处于理论/实验验证阶段。

---

## 597. Effective Task Planning with Missing Objects using Learning-Informed Object Search

**arXiv ID:** 2602.11468 | [PDF](https://arxiv.org/pdf/2602.11468v1)

**作者:** Raihan Islam Arnob `[一作]` (George Mason University), Gregory Stein `[通讯]` (George Mason University)

**通讯引用:** 6813 | [OpenAlex ID](https://openalex.org/A5042000667)

**关键词:** `Robotics` `Robotic Intelligence` `Optimization` `Reinforcement Learning` `Tabular`

**🎯 论文内容**

提出一种在物体位置未知的情况下，通过将单对象搜索动作抽象为确定性高层动作并使用学习驱动的搜索策略（LIOS）来实现任务规划的方法。

**💡 创新点**

创新点在于：①将不确定的搜索动作封装为确定性高层动作，使经典pddl规划器可直接使用；②利用学习得到的物体位置概率与模型计算得到的期望搜索成本，指导高层规划决定要寻找的物体及何时搜索。

**🔧 技术方法**

使用技术包括：基于句子编码的物体位置预测网络、模型基期望成本计算（Bellman方程）、pddl+fastdownward规划器、以及在Spot机器人上的实现与执行。

**📊 数据集**

数据集包括：500个procthor生成的家庭场景（用于训练物体位置预测模型），以及在真实Spot机器人实验环境中的九个可搜索容器与目标物体。

**📈 对比分析**

与贪心搜索、优化/惰性贪心等基线对比，ModelLIOS在仿真与真实任务中均展现出更高的成功率（提升10%~60%）和更低的平均成本，尤其在早餐+咖啡等复杂任务中表现最优。

**⚠️ 局限性**

局限包括：只能顺序搜索单一物体，缺乏并行或机会性搜索；对物体状态变量估计过于乐观，需通过重规划修正；以及依赖先验的可搜索容器与已知物体集合。

---

## 598. ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences

**arXiv ID:** 2602.11354 | [PDF](https://arxiv.org/pdf/2602.11354v1)

**作者:** Bang Nguyen `[一作]` (University of Notre Dame), Meng Jiang `[通讯]` (University of Notre Dame)

**通讯引用:** 5830 | [OpenAlex ID](https://openalex.org/A5074821819)

**关键词:** `Artificial Intelligence` `Transformer` `Large Language Model` `Agentic AI` `Text` `Benchmark`

**🎯 论文内容**

提出了 ReplicatorBench 基准，用于评估 LLM 代理在社会与行为科学领域从信息抽取、代码执行到结果解释的端到端研究复制过程。

**💡 创新点**

创新点包括：①将复制工作拆分为三阶段（Extraction、Generation、Interpretation）并提供 1,568 个细粒度评估检查点；②引入人类专家验证的真实复制报告作为地面真值；③设计 ReplicatorAgent 框架，支持迭代调试、工具调用与沙盒执行；④对比 Python 与原生语言执行、仅数据 vs. 完整代码场景，并分析错误类型。

**🔧 技术方法**

技术：基于 ReAct 思维-行动循环的 LLM 代理；工具箱包括文件/目录检索、最小差异编辑、数据检查器、Python 代码翻译等；沙盒化容器运行；LLM-评判器 LLMEval 用于评估 24/30/10/13 个检查点的匹配度；实验使用 GPT‑4o、GPT‑5、GPT‑5‑mini、o3 等模型。

**📊 数据集**

数据集：19 篇来自 SCORE 项目的社会与行为科学论文复制案例，包含人类专家预注册计划、复制报告、代码与数据；同时使用公开的 Web 搜索结果作为检索任务。

**📈 对比分析**

比较方法：在每个阶段使用 LLMEval 打分并报告宏/微平均；在复制结论上给出二分类指标（Precision、Recall、F1）。结果显示：GPT‑5 在执行和解释阶段得分最高，但在信息抽取和 Web 搜索阶段仍显不足；Python 模式在执行成功率上略优于原生模式，但不一定提升最终结论准确性；整体宏 F1 仍低于人类抽取水平，说明 LLM 代理尚未达到人类专家水平。

**⚠️ 局限性**

局限性：①基准样本仅 19 篇，规模有限；②LLMEval 作为自动评判可能引入不确定性；③Web 搜索依赖公开数据，无法覆盖实验性复制；④对沙盒安全与人类检查的依赖限制了大规模自动化；⑤未覆盖跨语言、跨学科更广泛的复制情境。

---

## 599. Security Threat Modeling for Emerging AI-Agent Protocols: A Comparative Analysis of MCP, A2A, Agora, and ANP

**arXiv ID:** 2602.11327 | [PDF](https://arxiv.org/pdf/2602.11327v1)

**作者:** Zeynab Anbiaee `[一作]` (Canadian Institute for Cybersecurity), Sajjad Dadkhah `[通讯]` (Canadian Institute for Cybersecurity)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Agentic AI`

**🎯 论文内容**

本文针对四大新兴 AI 代理通信协议（MCP、A2A、ANP、Agora）进行系统性安全分析，首先构建结构化威胁建模，随后提出基于生命周期的定性风险评估框架，并通过对 MCP 的测量驱动案例研究验证了缺失验证/认证导致的执行风险。

**💡 创新点**

创新点主要包括：①首次统一将四种协议纳入同一威胁模型与风险评估视角；②设计了跨协议、跨生命周期的风险量化框架，能够识别并比较十二类协议级风险；③通过可量化实验（多服务器组合下错误供应商工具执行率）将理论风险转化为可验证的安全指标。

**🔧 技术方法**

技术手段包括：结构化威胁建模（基于 STRIDE/ CIA 与协议架构）、定性风险评估（机率-影响矩阵）、案例实验测量（对 MCP 的执行流程进行模拟并量化错误工具执行）以及对协议生命周期（创建、运行、更新）进行安全属性映射。

**📊 数据集**

论文未使用公开的机器学习或自然语言处理数据集；其实验基于协议实现的模拟环境与代表性解析器策略，主要聚焦在协议内部行为与安全属性验证，未涉及外部大规模真实数据集。

**📈 对比分析**

比较方法：通过威胁模型构造将四协议映射到同一风险维度，使用相同的机率与影响等级（低/中/高）以及风险矩阵进行交叉评估，评估结果以表格形式展示每个协议在不同生命周期阶段的风险得分；由于缺乏客观性能指标，本文并未给出数值化的性能对比，而是以安全风险等级和风险热点对比为主。

**⚠️ 局限性**

局限性：①协议仍处于早期或实验阶段，缺乏大规模部署与真实攻击案例；②风险评估为定性分析，未结合历史漏洞或真实攻击数据；③实验仅针对 MCP，未对 A2A、ANP、Agora 进行同等量化验证；④缺乏对协议版本演进与互操作性细节的深入评估；⑤研究侧重安全属性，未考虑协议在实际系统中的性能、可扩展性与兼容性。

---

## 600. New Planar Algorithms and a Full Complexity Classification of the Eight-Vertex Model

**arXiv ID:** 2602.11292 | [PDF](https://arxiv.org/pdf/2602.11292v1)

**作者:** Austen Fan `[一作]` (University of Wisconsin-Madison), Zhuxiao Tang `[通讯]` (University of Wisconsin-Madison)

**关键词:** `Computational Complexity`

**🎯 论文内容**

本文针对平面八顶点模型的Holant问题，给出了完整的#P/多项式时间复杂度二分法；

**💡 创新点**

创新点在于提出了新的复杂度分类框架，统一处理所有平面八顶点模型参数，发现并证明了一个新的可解匹配门（matchgate）类，并引入共形映射与莫比乌斯变换的多项式插值技术；

**🔧 技术方法**

主要技术包括全局相位变换与匹配门的霍洛格拉法变换、二叉修改与交叉插值、2-拉伸、旋转对称性以及莫比乌斯变换实现任意二进制签名；

**📊 数据集**

本文不使用任何实验数据集，完全基于理论证明；

**📈 对比分析**

由于该研究为理论复杂度分析，没有与实验方法比较，性能以计算复杂度为评判标准，证明所有非匹配门情况为#P-困难；

**⚠️ 局限性**

限制在于仅对平面图上的八顶点模型进行分类，对于非平面图或更高维签名的Holant问题仍未覆盖，且部分特殊参数情形需进一步研究。

---

## 601. Retrieval Heads are Dynamic

**arXiv ID:** 2602.11162 | [PDF](https://arxiv.org/pdf/2602.11162v1)

**作者:** Yuping Lin `[一作]` (Michigan State University), Jiliang Tang `[通讯]` (Michigan State University)

**通讯引用:** 25151 | [OpenAlex ID](https://openalex.org/A5040639891)

**关键词:** `Computation and Language` `Retrieval` `Generation` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

本文系统研究了大型语言模型中检索头的动态行为，证明检索头随生成步动态变化、不可互换且与隐藏状态高度相关，并将动态检索头应用于动态检索增强生成框架中。

**💡 创新点**

创新点在于将检索头视为时间序列的动态实体，提出动态检索头概念，并展示其不可替代性和与模型内部状态的前瞻性关联。

**🔧 技术方法**

主要技术包括基于注意力分数的检索头判定、头消融实验、Canonical Correlation Analysis 与 MLP 预测器、以及注意力掩码实现的动态检索增强生成。

**📊 数据集**

实验数据集包括 Needle‑in‑a‑Haystack (NIAH) 和多跳推理数据集 HotpotQA。

**📈 对比分析**

通过与静态检索头、随机头以及无检索基线的对比，动态检索头在 HotpotQA 上平均提升 EM/F1 10% 左右，证明其在多跳推理任务中的实际效果。

**⚠️ 局限性**

局限性包括使用注意力掩码模拟检索而非真实检索、依赖 MLP 预测的动态检索头且预测误差可能影响效果，以及研究范围仅覆盖检索密集的 QA 任务。

---

## 602. Evaluating Memory Structure in LLM Agents

**arXiv ID:** 2602.11243 | [PDF](https://arxiv.org/pdf/2602.11243v1)

**作者:** Alina Shutova `[一作]`, Anton Sinitsin `[通讯]`

**关键词:** `Machine Learning` `Retrieval` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Benchmark`

**🎯 论文内容**

提出了 StructMemEval 基准，评估 LLM 代理在将知识组织为树、账本、状态跟踪等结构化形式以完成任务的能力。

**💡 创新点**

将评测焦点从单纯的事实检索转向需要知识结构化的任务，揭示了 Retrieval-augmented LLM 的局限，并证明提示记忆结构对性能至关重要。

**🔧 技术方法**

使用 Retrieval-augmented LLM（OpenAI embeddings）、多种记忆框架（Mem-agent、Mem0、Mem0g 等）以及“memory organization hint”提示，评估方法包括精确匹配和 LLM-as-judge。

**📊 数据集**

构建了 73 个对话场景、544 个评估问题的合成数据集，涵盖树结构、状态跟踪、计数等三类任务，数据已公开于 GitHub。

**📈 对比分析**

在无提示与有提示两种模式下与 Retrieval baseline 进行对比；结果显示记忆代理在提示下显著优于检索 baseline，且无提示时仍优于检索但表现不稳定；整体表明缺失结构化记忆导致显著性能下降。

**⚠️ 局限性**

局限性包括：只评估最终答案而非内部结构；任务种类有限，未覆盖多结构或更复杂场景；LLM 需要提示才能正确组织记忆，未充分训练内置结构化能力；合成数据的真实性可能有限。

---

## 603. Thinking with Drafting: Optical Decompression via Logical Reconstruction

**arXiv ID:** 2602.11731 | [PDF](https://arxiv.org/pdf/2602.11731v1)

**作者:** Jingxuan Wei `[一作]` (Shenyang Institute of Computing Technology), Cheng Tan `[通讯]` (Westlake University)

**通讯引用:** 1764 | [OpenAlex ID](https://openalex.org/A5006542157)

**关键词:** `Computation and Language` `Large Language Model` `Supervised Fine-Tuning` `Multimodality` `Benchmark`

**🎯 论文内容**

研发了一种将视觉输入解析为可执行 DSL 草稿并通过可视化验证的思维草稿（TwD）框架，用于高精度视觉算术推理。

**💡 创新点**

通过将 OCR 符号映射为逻辑拓扑的光学解压思想，并引入极简图形 DSL 作为中间表示，构建闭环可视化验证，解决多模态推理的精度悖论。

**🔧 技术方法**

基于多模态大语言模型（如 Qwen3‑VL‑8B）进行监督微调，结合 DSL 语法、虚拟网格抽象、确定性渲染引擎以及 LLM 判别器，实现结构化推理与可视化验证。

**📊 数据集**

构建了 VisAlg 视觉代数基准，自动草稿生成与人工审核相结合，涵盖五类比例、速率、变化、和差等问题，共 10,430 训练样本和 942 测试样本。

**📈 对比分析**

按代码相似度（chrF）、图像相似度（SSIM）和 LLM 评估分三项指标综合评测，与开源权重模型和主流专有模型对比，TwD 获得 82.63 的综合分，超越所有基线且在部分专有模型上表现更好。

**⚠️ 局限性**

DSL 仅覆盖条形图代数的线性拓扑，难以扩展到更广泛的科学图表或高阶集合逻辑，限制了方法的通用性。

---

## 604. Variation-aware Flexible 3D Gaussian Editing

**arXiv ID:** 2602.11638 | [PDF](https://arxiv.org/pdf/2602.11638v1)

**作者:** Hao Qin `[一作]` (Zhejiang University), Qiang Zhu `[通讯]` (Zhejiang University)

**通讯引用:** 18911 | [OpenAlex ID](https://openalex.org/A5100432719)

**关键词:** `Graphics` `Knowledge Distillation` `Generation` `Transformer` `Diffusion model` `Gaussian Splatting` `Stochastic Differential Equation` `Point Cloud`

**🎯 论文内容**

开发了VF-Editor，一种能够实时、原生地在3D高斯云中直接预测并叠加属性变化的编辑框架；

**💡 创新点**

创新点在于将多源2D编辑知识蒸馏进单一变分预测器，并通过变分场生成模块和并行迭代解码器实现线性复杂度、消除视图不一致；

**🔧 技术方法**

采用Transformer+CLIP文本编码、随机tokenizer、并行解码器，并结合DDIM推理、扩散逆推及SDS辅助蒸馏的技术；

**📊 数据集**

使用ShapeSplat重建对象、SD3+V3D生成卡通对象、Mini-splatting重建的公共和私有场景，总计约11k个3D‑指令三元组；

**📈 对比分析**

与I‑gs2gs、GaussianEditor、DGE等3DGS编辑方法在RObj、GObj、Scene数据集上进行定量比较，C_sim、C_con、IS、IAA均优于基线，编辑速度约0.3秒；

**⚠️ 局限性**

仍无法在域外场景下有效编辑，缺少专门的原语生成分支，SDS蒸馏易导致模式崩溃，泛化虽好但略低于训练集性能。

---

## 605. interwhen: A Generalizable Framework for Verifiable Reasoning with Test-time Monitors

**arXiv ID:** 2602.11202 | [PDF](https://arxiv.org/pdf/2602.11202v1)

**作者:** Vishak K Bhat `[一作]` (Microsoft Research), Amit Sharma `[通讯]` (Microsoft Research)

**通讯引用:** 2967 | [OpenAlex ID](https://openalex.org/A5080592530)

**关键词:** `Logic in Computer Science` `Generation` `Optimization` `Computational Efficiency` `Transformer` `Prompt Engineering` `Text`

**🎯 论文内容**

提出了一种基于测试时验证的框架InterWhen，能够在生成过程中实时检查并纠正推理步骤，从而提高推理准确性和效率。

**💡 创新点**

创新在于使用meta-prompting让模型输出可提取的中间状态，并通过内部或外部验证器对这些状态进行即时验证与反馈，形成单轨迹的自适应推理路径。

**🔧 技术方法**

结合了meta-prompting、内部自验证与外部符号/工具验证器、k-stable答案稳定判定以及对生成轨迹的逐步抽取-验证-干预循环。

**📊 数据集**

在Maze、SpatialMap、GameOf24等空间与算术推理数据集，以及Verina代码/规范生成数据集上进行实验。

**📈 对比分析**

与EAT、DEER、ToT、Best-of-N、Majority等基线对比，InterWhen在保证100%可靠性的前提下实现了10–12%准确率提升和30%以上Token节省，且在多模型、多任务上表现稳健。

**⚠️ 局限性**

局限在于需要预先定义可验证的中间状态与相应验证器，且对某些任务的验证器构建成本高，模型对反馈的敏感性和多语言适用性仍待进一步研究。

---

## 606. DHPLT: large-scale multilingual diachronic corpora and word representations for semantic change modelling

**arXiv ID:** 2602.11968 | [PDF](https://arxiv.org/pdf/2602.11968v1)

**作者:** Mariia Fedorova `[一作]` (University of Oslo), Khonzoda Umarova `[通讯]` (Cornell University)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5035353318)

**关键词:** `Computation and Language` `Transformer` `Large Language Model` `Text`

**🎯 论文内容**

提出并发布了DHPLT，一个覆盖41种语言、包含三个时间段（2011‑2015、2020‑2021、2024‑）的 diachronic 语料库，并预先计算了词汇级别的静态词向量、上下文嵌入、词义替代以及词频统计，方便多语言词义变迁研究。

**💡 创新点**

创新点在于利用 HPLT 项目 web‑crawl 的时间戳构建大规模多语言 diachronic 语料，克服传统语料库稀缺和单语局限，同时提供多种预训练词表示，显著丰富了 LSCD 领域的资源与实验平台。

**🔧 技术方法**

技术手段包括：HPLT 语料预处理（语言识别、去重、清洗）、随机采样生成时间段语料、使用 T5/GPT‑BERT/XLM‑R 语言模型生成上下文嵌入和词义替代、SGNS 训练静态词向量、Procrustes 对齐、词频统计与目标词选取（基于词性和词频过滤）。

**📊 数据集**

使用的数据集为 HPLT v3.0 web‑crawl 语料，按语言和时间段随机抽取 1 M（或 0.5 M）文档，共计约 59 亿词，涵盖 41 种语言，所有数据均以 CC0 许可证发布。

**📈 对比分析**

通过对词“AI”（及其对应词）的静态词向量邻居变化和 T5 编码器嵌入平均距离等指标展示语义漂移，证明资源可用于 LSCD 实验；但文中未给出统一性能评测或基准对比，只提供示例性分析。

**⚠️ 局限性**

主要局限在于时间信号仅为抓取时间戳，不能精确反映文本真实创作时间；仅为预定义目标词提供表示，未覆盖所有词汇；使用的模型不具备时间感知，可能导致早期时期预测包含后期出现的词语。

---

## 607. LLM-based Triplet Extraction from Financial Reports

**arXiv ID:** 2602.11886 | [PDF](https://arxiv.org/pdf/2602.11886v1)

**作者:** Dante Wesslund `[一作]` (KTH Royal Institute of Technology), Alexander Holmberg `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `Computation and Language` `Data-Centric Learning` `Knowledge Distillation` `Optimization` `Explainability and Interpretability` `Computational Efficiency` `Anomaly Detection` `Transformer` `Large Language Model` `Prompt Engineering` `Text` `Finance Related`

**🎯 论文内容**

设计并实现了一套半自动化的主谓宾三元组抽取管线，能够在没有人工标注的财务报告中提取结构化知识并进行无监督评估。

**💡 创新点**

创新点包括：①基于本体一致性和真实性的代理评估指标；②针对每份报告自动诱导本体以消除本体漂移；③混合验证策略（正则+LLM判定）显著降低幻觉误报。

**🔧 技术方法**

使用技术包括：大型语言模型（Gemini‑2.5 Flash、Llama 4 Maverick）与Prompt工程、正则表达式匹配、LLM‑as‑Judge语义验证，以及自动本体诱导流程。

**📊 数据集**

使用数据集为两份公开年报：Volvo 2024年财报与Elektas 2022/2023年报（Elektas仅评估前25%）。

**📈 对比分析**

通过比较手工本体与自动本体、不同LLM与验证方法，评估指标为本体一致性OC、主体幻觉SH、客体幻觉OH及关系幻觉RH；结果显示自动本体在所有配置下实现100% OC，幻觉率降至1–2%，Gemini在保守性与准确性上优于Llama。

**⚠️ 局限性**

局限性包括：仅评估两份英文年报；Elektas仅覆盖前25%；缺乏召回评估；LLM‑as‑Judge的可靠性仍需更大规模人工审计；自动本体可能过拟合；未探索混合核心+扩展本体的方案。

---

## 608. TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents

**arXiv ID:** 2602.11767 | [PDF](https://arxiv.org/pdf/2602.11767v1)

**作者:** Aladin Djuhera `[一作]` (Technical University Munich), Holger Boche `[通讯]` (Technical University Munich)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Reinforcement Learning` `Large Language Model`

**🎯 论文内容**

提出 TSR（Trajectory‑Search Rollouts）框架，在多回合 RL 训练中使用轻量树搜索生成更高质量的轨迹，从而提升训练数据质量。

**💡 创新点**

创新点在于把推理时的搜索算法（Best‑of‑N、Beam、Lookahead）迁移到训练时的 rollout 阶段，使得 rollout 过程自适应、探索更充分且与优化器无关；并结合实例级过滤提升任务多样性。

**🔧 技术方法**

技术包括：树搜索（候选动作集、评分函数、Beam/Lookahead/Best‑of‑N）、基于策略梯度的 RL（PPO、GRPO）、实例级不确定性过滤、任务特定代理（Qwen2.5-0.5B/3B）以及 RAGEN 训练框架。

**📊 数据集**

使用的任务/数据集有：Sokoban、FrozenLake、WebShop（均来自 RAGEN），每个任务提供多样化的提示集合（prompts）与环境交互。

**📈 对比分析**

与传统实例过滤+随机采样 baseline 对比，TSR 在所有环境中提升成功率 4–15%，尤其在 WebShop 任务提升 15% 以上；同时在 Qwen2.5‑0.5B 上能超过 GPT‑4o、Qwen2.5‑72B 的零样本性能，且在推理时显著减少 token 数与交互轮数。

**⚠️ 局限性**

局限性包括：需要额外的训练时计算（搜索预算需调优）、搜索策略对任务评分的依赖（若评分不准可能导致误导），在更大规模或更复杂环境中尚未充分验证，且搜索深度与宽度对性能的收益递减，需要更高效的搜索实现。

---

## 609. Federated Gaussian Process Learning via Pseudo-Representations for Large-Scale Multi-Robot Systems

**arXiv ID:** 2602.12243 | [PDF](https://arxiv.org/pdf/2602.12243v1)

**作者:** Sanket A. Salunkhe `[一作]` (Colorado School of Mines), George P. Kontoudis `[通讯]` (Colorado School of Mines)

**通讯引用:** 371 | [OpenAlex ID](https://openalex.org/A5039559939)

**关键词:** `Multiagent Systems` `Federated Learning` `Optimization` `Robotic Intelligence` `Tabular`

**🎯 论文内容**

提出了 pxpGP 与 dec-pxpGP 两种分布式高斯过程学习框架，利用稀疏变分推断生成边界约束与排斥惩罚的伪数据集，实现了多机器人网络下的联邦超参数优化与预测。

**💡 创新点**

创新点：① 通过在变分 ELBO 中加入边界与排斥惩罚，获得分布均匀、信息丰富的伪数据；② 采用缩放式近似一致 ADMM 进行全局超参数一致化，并在更新过程中自适应调节罚项与 Lipschitz 参数；③ 兼顾集中与去中心化两种拓扑，使用伪数据共享而非原始数据，显著提升数据隐私与数值稳定性。

**🔧 技术方法**

主要技术：稀疏变分高斯过程（Sparse Variational GP）、边界/排斥惩罚、Proximal Inexact Consensus ADMM（pxADMM）、自适应残差平衡、基于K‑means初始化的诱导点、Flooding 机制（去中心化版）以及 Python/GPyTorch 实现。

**📊 数据集**

实验数据集：人工合成二维 GP 数据（N=16,900 与 N=34,900）以及 NASA SRTM 地形高程三块区域（N≈30,000/块），分别用于评估超参数精度与预测性能。

**📈 对比分析**

与基线方法（全局 GP、cGP、apxGP、gapxGP、dec‑apxGP、dec‑gapxGP）比较：在 16–100 机器人网络中，pxpGP 与 dec‑pxpGP 的超参数估计误差最小；在真实地形数据上，其 NRMSE 与基线相当，但 NLPD 显著更低，表明不确定性估计更准确；计算与通信复杂度与基线相当或略优，且收敛速度更快。

**⚠️ 局限性**

局限性：① 对于极大规模网络（>1000 机器人）仍需进一步验证收敛性与通信开销；② 需要先执行稀疏 GP 预训练，增加一次本地计算；③ 伪数据共享虽提升隐私，但若各子域间数据分布差异极大，仍可能导致共用伪数据不足以代表全局；④ 目前仅针对静态非参数任务，动态变化场景的适应性尚未探讨。

---

## 610. Legitimate Overrides in Decentralized Protocols

**arXiv ID:** 2602.12260 | [PDF](https://arxiv.org/pdf/2602.12260v1)

**作者:** Oghenekaro Elem `[一作]` (Parametrig), Nimrod Talmon `[通讯]` (BGU)

**关键词:** `Cryptography and Security` `Text`

**🎯 论文内容**

提出了Scope×Authority两维紧急治理机制分类，并通过对2016–2026年705起技术性攻击案例的实证分析，构建了基于随机成本最小化的决策模型。

**💡 创新点**

创新点在于：①用Scope×Authority对所有已知紧急介入方案进行统一归纳；②将安全–活性权衡形式化为包含持久中心化成本、容灾速度与碰撞损失的随机优化；③基于实证数据验证三条预测，并给出可操作的设计原则。

**🔧 技术方法**

采用了概率统计（功率律拟合、KS检验）、情绪分析（VADER）以及期望成本公式，辅以开源“Emergency Mechanism Calculator”实现模型计算。

**📊 数据集**

主要数据集为705个记录的攻击事件，其中601为可介入技术性攻击，130为实际介入案例，52个高质量案例被用作模型验证。

**📈 对比分析**

与传统的经验性治理讨论相比，本文通过量化模型和大规模数据检验表明：签名集合介入速度最快但中心化成本最高，治理型介入合法性最高但响应慢；精确程度（Account/Module）能在保持损失控制的同时显著降低碰撞损失。

**⚠️ 局限性**

局限性包括：①模型假设攻击类型和损失率已知且可估计，实际情况可能更复杂；②情绪分析仅基于公开论坛文本，可能无法捕捉所有社区意见；③未覆盖跨链、预执行防御等新兴机制，未来需要进一步扩展维度。

---

