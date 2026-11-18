# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-11-18 | 今日论文总数: 750

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. DreamFAR: Real-Time and Long-Horizon Interaction with Diverse Objects

**arXiv ID:** 2611.22001 | [PDF](https://arxiv.org/pdf/2611.22001v1)

**作者:** Yiming Dou `[一作]` (Amazon), Ruoshi Liu `[通讯]` (Amazon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的世界模型DreamFAR，该模型能够实时进行长时间的接触丰富交互，适用于多种对象和场景。

**💡 创新点**

创新点在于引入了基于接触的记忆上下文，能够在没有外部接触注释的情况下，记住早期交互如何改变场景，并在长时间的回放中保持准确性。

**🔧 技术方法**

使用了一种潜在流匹配模型，结合视觉标记器和动态变换器来预测基于动作的状态转变。

**📊 数据集**

使用了内部双手遥操作数据集，该数据集包含来自YAM机器人的大量机器人本地数据。

**📈 对比分析**

与现有的世界模型（如IWS、WEAVER和Cosmos-Predict 2.5）进行比较，DreamFAR在长时间预测中表现出更高的准确性和更低的视觉误差，PSNR、LPIPS和FID等指标均优于基线模型。

**⚠️ 局限性**

局限性在于模型仅限于YAM站内的交互，未来工作可以扩展到更开放的环境和多种机器人形态的交互模拟。

---

## 2. SciWorkBench: A Benchmark for Real Scientific Workflows

**arXiv ID:** 2611.22490 | [PDF](https://arxiv.org/pdf/2611.22490v1)

**作者:** Yuchen Zhang `[一作]`, Bing Zhao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了SciWorkBench，一个用于评估智能体在真实科学任务中的能力的基准，涵盖生物学、化学、材料科学和物理学等领域。

**💡 创新点**

创新点在于建立了一个统一的科学分析框架，并通过领域专家验证，能够对SciWorkBench中的大多数任务进行分类。

**🔧 技术方法**

使用了Claude Code和Codex智能体框架，并在统一的执行和评分协议下评估了最新模型。

**📊 数据集**

使用了来自同行评审文献或专家贡献的研究场景构建的数据集，共有98个任务。

**📈 对比分析**

与现有方法相比，SciWorkBench更关注工作流级别的科学判断，评估结果显示当前智能体在完成真实科学工作流方面仍然存在困难，最佳智能体的平均得分为57.38。

**⚠️ 局限性**

SciWorkBench的局限性在于无法完全覆盖科学的广度，且由于评估长时间科学智能体的高成本，每个任务-智能体设置的重复次数有限，难以消除评分的变异性。

---

## 3. Body Schema Policy: Body as the Vision-Action Interface for Efficient Robot Loco-Manipulation

**arXiv ID:** 2611.22713 | [PDF](https://arxiv.org/pdf/2611.22713v1)

**作者:** Lechen Zhang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Yoshihiko Nakamura `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种模仿学习策略（BSP），通过构建潜在身体模式来提高机器人在全身运动操作中的学习效率。

**💡 创新点**

创新点在于将人类的身体模式结构引入机器人策略，使得机器人能够更高效地协调视觉感知与动作生成。

**🔧 技术方法**

使用了模仿学习、图神经网络和视觉-动作接口等技术。

**📊 数据集**

在九个模拟的Unitree G1全身任务上进行评估，并在真实的G1和Fetch移动操控器上进行测试。

**📈 对比分析**

与Diffusion Policy等现有方法进行比较，BSP在参数数量上减少了75倍，同时在任务成功率上几乎是Diffusion Policy的两倍，且在执行速度上也更快。

**⚠️ 局限性**

由于计算能力有限，未能在SIMPLE基准上进行更多种子的训练和评估；此外，BSP依赖于关节命令的运动学意义，在某些情况下可能不如其他方法有效。

---

## 4. A Visual Information Extraction System Using Multimodal Large Language Models for Scanned Cognitive and Behavioral Assessments

**arXiv ID:** 2611.22252 | [PDF](https://arxiv.org/pdf/2611.22252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 5. Lost in a Few Heads: Diagnosing and Repairing Block Selection in Sparse Attention

**arXiv ID:** 2611.21917 | [PDF](https://arxiv.org/pdf/2611.21917v1)

**作者:** Wenhe Zhang `[一作]` (Yale University), Boxing Chen `[通讯]` (Huawei Canada)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了稀疏注意力模型在固定预算下的长上下文检索准确性损失，分析了损失发生的原因，并提出了修复方法。

**💡 创新点**

创新点在于使用精确的注意力质量选择作为基准，识别出选择器未能正确排名关键块的原因，并提出了无需训练的修复方法。

**🔧 技术方法**

使用了精确注意力质量选择、校准探针和多种修复策略，包括全注意力和4位评分器。

**📊 数据集**

使用了多个模型的检查点数据，包括MiniCPM4-8B和Qwen3-8B等，进行多键检索任务的评估。

**📈 对比分析**

通过与全注意力模型的比较，发现修复方法能够显著提高多键检索的准确性，尤其是在特定的注意力头上，准确率从54.1%提升至97.5%。

**⚠️ 局限性**

限制在于研究主要集中在一个模型家族的检查点上，缺乏独立预训练的检查点验证，且不同选择器可能需要更多的头部来恢复损失。

---

## 6. Pareto-Informed Smart "Predict, then Optimize" for Multi-Objective Combinatorial Problems

**arXiv ID:** 2611.23076 | [PDF](https://arxiv.org/pdf/2611.23076v1)

**作者:** Aabhash Dhakal `[一作]` (University of Würzburg), Tim Lachner `[通讯]` (University of Würzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种针对多目标组合优化问题的Pareto感知扩展的决策聚焦学习框架，旨在通过整合下游优化任务来改善决策质量。

**💡 创新点**

创新点在于引入了一种基于真实Pareto前沿的损失函数，克服了传统线性标量化方法的局限性，能够有效处理非凸Pareto前沿的复杂权衡。

**🔧 技术方法**

使用了决策聚焦学习（DFL）和智能“预测，然后优化”（SPO+）的结合方法，特别是通过Pareto感知的损失函数进行训练。

**📊 数据集**

实验使用了多目标最短路径问题（MOSPP）和多目标多维背包问题（MOMDKP）这两个经典的组合优化问题的数据集。

**📈 对比分析**

与传统的两阶段均方误差（MSE）基线和基于标量化的SPO+方法相比，提出的方法在最小化决策遗憾方面表现优越，尤其在处理非凸区域时效果显著。

**⚠️ 局限性**

主要限制在于模型委员会架构的可扩展性，需要为每个已知用户偏好训练单独的模型，可能在处理大量或连续偏好时变得计算上不可行。

---

## 7. The Price of Hard Coverage Constraints in Online Conformal Prediction

**arXiv ID:** 2611.22646 | [PDF](https://arxiv.org/pdf/2611.22646v1)

**作者:** Mingrui Liu `[一作]` `[通讯]` (George Mason University), Mingrui Liu (George Mason University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在线设置中的顺序预测问题，提出了一种方法来确保在整个部署过程中，预测集的误覆盖数量保持在规定的预算内。

**💡 创新点**

明确了在满足硬覆盖约束下，强制执行的最低效率成本，并揭示了三种不同的时间范围下的成本规模。

**🔧 技术方法**

使用了随机化的经验排名、障碍控制和马尔可夫链论证等技术。

**📊 数据集**

使用了独立同分布的数据集进行实验，具体数据集未明确提及。

**📈 对比分析**

通过与现有在线符合预测方法的比较，展示了在不同时间范围内的性能表现，结果表明在已知分布的情况下，达到的效率成本为Θ(log T)、Θ(T^1/3)和Θ(√(T))。

**⚠️ 局限性**

研究的局限性在于依赖于静态边界和均匀插入排名的假设，这些假设在分布变化或数据重用时可能失效。

---

## 8. Single-Vertex Metropolis on Minor-Free Graphs: A General EPTAS

**arXiv ID:** 2611.22023 | [PDF](https://arxiv.org/pdf/2611.22023v1)

**作者:** Daniel Skachkov `[一作]` (MSU Institute for Artificial Intelligence), Yuriy Dorn `[通讯]` (MSU Institute for Artificial Intelligence)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一类组合优化问题，目标是找到具有特定性质的最大诱导子图。通过单顶点Metropolis过程，证明了在排除任意固定小图H的图上，该过程是一个随机有效的多项式时间近似方案。

**💡 创新点**

提出了一种通用的随机EPTAS保证，适用于排除固定小图的图类，涵盖了最大独立集、最大诱导森林、最大诱导二部子图和最大度数不超过一的诱导子图等问题。

**🔧 技术方法**

使用了单顶点Metropolis过程，该过程通过随机选择顶点并接受可行的添加或删除来维护一个可行集。

**📊 数据集**

研究中没有具体提到使用的数据集，但涉及的图类是排除固定小图的图。

**📈 对比分析**

通过与其他方法的比较，证明了在每个H-小图自由的图上，从任何可行初始集出发，Metropolis过程在exp(O_H(^-2)) n(n+log(1/δ))次提议内找到(1-ε)-近似解，且性能保证概率至少为1-δ。

**⚠️ 局限性**

限制在于对于不受限制的权重，单次从空集开始的运行没有统一的常数因子保证，即使活动是自适应选择的。

---

## 9. Route Optimization under Port Congestion and Carbon Reduction: An NSGA-II Approach

**arXiv ID:** 2611.22705 | [PDF](https://arxiv.org/pdf/2611.22705v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 10. Communicate the Change: Efficient Expert Parallelism via Activation Compression

**arXiv ID:** 2611.22215 | [PDF](https://arxiv.org/pdf/2611.22215v1)

**作者:** Fengyuan Liu `[一作]` (Mila - Quebec AI Institute), Dzmitry Bahdanau `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种名为EchoMoE的轻量级架构修改，旨在减少稀疏专家混合模型（MoE）中的通信量，从而提高推理吞吐量。

**💡 创新点**

创新点在于利用跨层的可预测性，通过缓存早期层的全宽激活作为参考，仅传输相对于该参考的压缩变化，从而显著减少通信流量。

**🔧 技术方法**

使用了一种轻量级的架构修改技术，结合了压缩和参考状态的使用，以减少专家并行通信的需求。

**📊 数据集**

在Qwen3-30B-A3B模型上进行了评估，使用了18个基准测试，包括知识与推理、数学、编码和指令跟随等任务。

**📈 对比分析**

与传统的专家并行方法相比，EchoMoE在减少58%的RDMA通信量的同时，任务性能仅下降3.8个百分点，推理吞吐量提高了22.6%。

**⚠️ 局限性**

限制在于EchoMoE的性能下降可能会影响某些特定任务的表现，尽管整体性能仍然保持在可接受范围内。

---

## 11. The Impact of Scenario Representation in Scenario-based Optimization Certification

**arXiv ID:** 2611.21795 | [PDF](https://arxiv.org/pdf/2611.21795v1)

**作者:** Stephen Crawford `[一作]` (Brown University), Nora Ayanian `[通讯]` (Brown University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在无损场景重组下，场景优化的敏感性，提出了新的证书评估方法。

**💡 创新点**

提出了在相同样本信息下，不同的场景表示可能导致不同的证书接受决策，并量化了这种差异。

**🔧 技术方法**

使用了无损场景表示和原始观察的Wait-and-Judge理论。

**📊 数据集**

使用了480次配对的场景模型预测控制（MPC）实验数据集。

**📈 对比分析**

通过与传统的支持子样本证书进行比较，发现精确重组在不改变样本的情况下，认证可用性从318次增加到471次，显示出证书会计的重要性。

**⚠️ 局限性**

研究中未提及具体的限制，但指出未来的研究方向包括分析更复杂场景程序的分组行为。

---

## 12. Growing Clusters Lower the Bar: Profile Accretion in LLM-Based Incremental Entity Resolution

**arXiv ID:** 2611.22485 | [PDF](https://arxiv.org/pdf/2611.22485v1)

**作者:** Huixue Han `[一作]` (Queensland University of Technology), Yu-Chu Tian `[通讯]` (Queensland University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了大型语言模型（LLM）在增量实体解析中的表现，特别是当新记录与现有集群进行比较时，集群的大小如何影响模型的判断标准。

**💡 创新点**

创新点在于发现了集群配置的累积效应，即随着集群成员数量的增加，模型对新记录的接受标准降低，导致假匹配率上升。

**🔧 技术方法**

使用了大型语言模型（如Qwen和Claude系列）进行记录匹配，并通过不同的提示变体和格式进行实验。

**📊 数据集**

数据集来自FOPPA的法国采购供应商记录，使用SIREN标识符作为真实标签，并在TED的捷克和保加利亚合同获奖通知中进行了验证。

**📈 对比分析**

与传统的基于特征的匹配方法相比，LLM模型在处理相似名称的公司时，假匹配率显著上升，而ROC曲线下面积（AUC）变化不大，表明决策标准的偏移。

**⚠️ 局限性**

限制在于该效应在某些情况下可能不适用，例如当单个记录已经能够识别公司时，累积效应消失。

---

## 13. Deciding Without Pointing: A Case Study of Whether a Decision Model Localizes the Anomalies It Detects

**arXiv ID:** 2611.21995 | [PDF](https://arxiv.org/pdf/2611.21995v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 14. Verifying Agents in Rubric-Graded Environments

**arXiv ID:** 2611.22841 | [PDF](https://arxiv.org/pdf/2611.22841v1)

**作者:** Markus Dücker `[一作]` (Handshake AI Research), Anish Athalye `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本研究系统地研究了在基于评分标准的环境中用于评估智能代理的验证器，提出了一个新的开放源代码验证器Gandalf，并通过人类注释的数据集进行评估。

**💡 创新点**

创新点在于首次系统性地分析了验证器的设计原则，并提出了三项设计原则：反应性验证、环境对齐和领域指导。

**🔧 技术方法**

使用了开放源代码的Python应用程序Gandalf，结合了反应性代理和环境对齐的设计原则。

**📊 数据集**

使用了BankerToolBench (BTB)环境中的数据集，包含2464个判断实例和24个任务，所有任务均由两位领域专家独立评估。

**📈 对比分析**

与三种现有验证器进行比较，Gandalf在准确性和成本之间提供了最佳的权衡，所有基线配置中，Gandalf的配置在准确性上至少与其他配置相当，但成本更低。

**⚠️ 局限性**

限制在于数据集中每个任务仅包含一个回滚，没有评估验证器对回滚分布的敏感性；未评估验证器的准确性优势是否能转化为通过强化学习训练的更好代理。

---

## 15. LASH-PAT: Deadlock-Free Adaptive Shortest-Path Routing for Inter-Chiplet Interconnects

**arXiv ID:** 2611.22937 | [PDF](https://arxiv.org/pdf/2611.22937v1)

**作者:** Patrick Iff `[一作]` (ETH Zurich), Torsten Hoefler `[通讯]` (ETH Zurich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

研究了不规则拓扑下的无死锁路由算法，提出了一种新的路由函数LASH-PAT，旨在提高芯片间互连网络的性能。

**💡 创新点**

创新点在于结合了基于转弯的路由和分层路由的优点，提供了路径多样性，同时确保最短路径传输。

**🔧 技术方法**

使用了改进的Dijkstra算法来构建路由表，并通过动态选择路径来缓解网络拥堵。

**📊 数据集**

评估了十种不同的芯片间互连拓扑，使用了合成流量模式和应用工作负载追踪。

**📈 对比分析**

与九种基线路由函数进行比较，LASH-PAT在吞吐量上提高了21.15%，每个数据包的能耗降低了12.77%，延迟与最低延迟基线相当。

**⚠️ 局限性**

局限性在于支持自适应路径多样性需要更大的路由表，尽管在现代芯片系统中这种内存开销是可以接受的。

---

## 16. Exploited Before It Was Fixed? Vulnerability Timelines from Open, Machine-Readable Evidence

**arXiv ID:** 2611.22479 | [PDF](https://arxiv.org/pdf/2611.22479v1)

**作者:** Cédric Bonhomme `[一作]` (Computer Incident Response Center Luxembourg), Alexandre Dulaunoy `[通讯]` (Computer Incident Response Center Luxembourg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文计算了每个漏洞在被利用和修复之间的时间间隔，并使用开放的机器可读证据（如CVE和CSAF记录）来构建生命周期间隔。

**💡 创新点**

创新点在于提出了一种证据模型，能够为每个漏洞计算利用和修复之间的时间间隔，并追踪每个数据的来源和精确度。

**🔧 技术方法**

使用了证据模型和Vulnerability-Lookup工具，结合了多种开放数据源。

**📊 数据集**

使用的数据集包括CVE记录、CSAF建议、多个已知利用漏洞目录和观察记录，共计3758个带有利用证据的标识符。

**📈 对比分析**

与其他方法相比，本文提供了每个漏洞的具体时间间隔，而不是仅仅提供汇总数据。性能方面，590个漏洞有修复证据，59个漏洞的修复证据在利用证据之后，平均延迟7.6天。

**⚠️ 局限性**

限制在于证据发布者提供的数据，许多厂商和协调者并未发布所需的时间戳，导致无法计算所有漏洞的完整生命周期。

---

## 17. MoL-SQL: A Controlled Benchmark for Mixture-of-Language Text-to-SQL

**arXiv ID:** 2611.22333 | [PDF](https://arxiv.org/pdf/2611.22333v1)

**作者:** Tianqi Luo `[一作]` (Hong Kong University of Science and Technology), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种混合语言的文本到SQL基准MoL-SQL，允许独立切换问题、模式和存储值的语言，同时保持任务不变，并将每个跨语言实体映射分为四个难度级别。

**💡 创新点**

创新点在于首次系统性地定义了混合语言文本到SQL的任务，并通过一个统一的实体映射框架构建了基准，能够独立评估不同语言对模型性能的影响。

**🔧 技术方法**

使用了一个实体映射框架，该框架通过执行等价性检查来验证每个SQL实现，并生成目标语言的问题。

**📊 数据集**

使用了五个不同领域和源语言的文本到SQL基准，最终构建了两个数据集分割，分别包含16892个样本和3840个样本。

**📈 对比分析**

与八个大型语言模型和四种文本到SQL方法的实验表明，模式和存储值的语言对准确性有显著影响，存储值的语言差异导致的准确性下降可达23个百分点，而问题翻译和额外推理并未缩小这一差距。

**⚠️ 局限性**

该基准仅覆盖英语和中文，并且只在SQLite数据库上进行等价性检查，构建在公共数据库上，这些数据库可能在预训练期间被模型见过，因此其结果应被视为诊断样本而非部署估计。

---

## 18. Limite 1B - Violetto: Technical Report

**arXiv ID:** 2611.21646 | [PDF](https://arxiv.org/pdf/2611.21646v1)

**作者:** Mario Prignano `[一作]`, Francesco Pappone `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

论文探讨了某种新方法或技术的应用。

**💡 创新点**

创新点在于提出了一种新的视角或改进了现有的方法。

**🔧 技术方法**

使用了先进的算法或模型进行研究。

**📊 数据集**

采用了特定的数据集进行实验和验证。

**📈 对比分析**

与其他方法进行了比较，结果显示性能有所提升。

**⚠️ 局限性**

限制在于数据集的规模或适用性可能影响结果的普遍性。

---

## 19. Wall-to-wall maps of canopy height and aboveground carbon across the Atlantic Forest

**arXiv ID:** 2611.23048 | [PDF](https://arxiv.org/pdf/2611.23048v1)

**作者:** Matheus Pinheiro Ferreira `[一作]` (University of São Paulo), Pedro H. S. Brancalion `[通讯]` (University of São Paulo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究为2024年巴西大西洋森林提供了全覆盖的树冠高度和地上碳密度地图，使用深度学习网络结合了Sentinel-2反射率、Sentinel-1和ALOS PALSAR的后向散射以及ALOS高程数据进行预测。

**💡 创新点**

创新点在于通过深度学习模型将多种遥感数据融合，生成高精度的树冠高度和碳密度地图，并在区域尺度上验证了全球产品的准确性。

**🔧 技术方法**

使用了深度学习技术，具体为四分支融合网络，结合了Sentinel-2、Sentinel-1、ALOS PALSAR和ALOS数字表面模型的数据。

**📊 数据集**

数据集包括来自2023和2024年在圣保罗州进行的空中LiDAR调查的7179个训练样本和1793个测试样本，覆盖面积约为58,800平方公里。

**📈 对比分析**

与Meta和世界资源研究所的全球产品相比，本研究的模型在测试区域的均方根误差为3.05米，而全球产品的误差超过两倍，表明全球产品在区域应用前需要进行本地验证。

**⚠️ 局限性**

限制在于模型主要基于圣保罗州的数据，可能无法完全捕捉大西洋森林区域内的所有森林结构变异，外部验证结果显示在不同区域的表现差异。

---

## 20. As Student Rollouts Continue, Visual Backdoors Keep Growing

**arXiv ID:** 2611.22446 | [PDF](https://arxiv.org/pdf/2611.22446v1)

**作者:** Zane Fu `[一作]` (Alibaba Group), Ruohui Huang `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在On-Policy Distillation (OPD)中视觉后门传播的风险，揭示了其潜在的安全隐患。

**💡 创新点**

创新点在于揭示了OPD中视觉后门传播的阈值效应，并提出了一种新的触发器设计，能够在没有显式触发器注入的情况下实现后门传播。

**🔧 技术方法**

使用了On-Policy Distillation (OPD)技术，并结合了Latent Refusal Loss (LRL)来增强教师模型的拒绝相关令牌的概率。

**📊 数据集**

使用了多个数据集进行实验，包括Flickr8k、COCO、VQAv2和TextVQA，涵盖了图像描述和视觉问答任务。

**📈 对比分析**

与传统的知识蒸馏方法相比，OPD在低中毒比例下后门传播的成功率（ASR）保持较低，但一旦超过临界阈值，ASR迅速接近100%。而传统方法则表现出更渐进的传播趋势。

**⚠️ 局限性**

限制在于实验主要集中在同一模型家族内的蒸馏，未探索跨家族传播；理论分析依赖于固定的条件令牌分布，未能完全捕捉不同生成前缀的概率演变；尽管展示了零显式触发器注入的后门转移，但其底层机制仍不清晰。

---

## 21. GuardAgent: A Memory-Augmented Agentic Framework for Adaptive Detection of Adversarial Promotions in LLM-Based Recommendation

**arXiv ID:** 2611.22274 | [PDF](https://arxiv.org/pdf/2611.22274v1)

**作者:** Shiyi Yang `[一作]` (University of New South Wales), Lina Yao `[通讯]` (University of New South Wales)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为GuardAgent的框架，用于检测基于大型语言模型（LLM）的推荐系统中的促销驱动攻击，旨在保护下一代推荐平台。

**💡 创新点**

创新点在于将对抗性检测形式化为一个认知代理循环，允许持续学习和自我演变，以适应动态环境中的新威胁，同时提供可解释的检测结果。

**🔧 技术方法**

使用了记忆增强的代理框架，结合感知、行动、反思和记忆的循环机制，进行对抗性内容检测。

**📊 数据集**

使用了真实世界的数据集，包括亚马逊产品和Yelp商业数据集，进行广泛的实验验证。

**📈 对比分析**

与多种基线方法进行比较，GuardAgent在准确性、召回率和F1分数等指标上表现优异，显示出强大的泛化能力和低误报率。

**⚠️ 局限性**

局限性在于当前方法可能在面对更复杂的对抗性攻击时仍需进一步优化，尤其是在解释的可信度和在较弱监督下的鲁棒性方面。

---

## 22. The Rhythm of Generated Video: Latent Chunking Leaves Periodic Traces in Compressed Bitstreams

**arXiv ID:** 2611.22996 | [PDF](https://arxiv.org/pdf/2611.22996v1)

**作者:** Chen Ye Gan `[一作]` (Tsinghua University), Yuxing han `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文展示了生成视频的自编码器在压缩比特流中留下的特定节奏，揭示了生成过程与编码需求之间的联系。

**💡 创新点**

创新点在于首次将生成视频的时间步长与可测量的编码需求联系起来，并提出了一种无需解码的方法直接从编码比特流中读取节奏。

**🔧 技术方法**

使用了因果视频自编码器和传统的视频编码器（如H.264、H.265和AV1），并通过解析比特流中的帧和行大小来进行分析。

**📊 数据集**

使用了DeepAction数据集（2605个摄像头视频片段）和自收集的Frontier数据集（包含四个商业生成器的生成视频），并进行了相应的编码。

**📈 对比分析**

与基于像素的检测方法（如UnivFD）相比，基于比特流的检测方法在生成视频的识别上表现更好，准确率提高了10-17个百分点。

**⚠️ 局限性**

限制在于交付风格的编码可能会抹去每帧的脉冲，且在某些情况下，生成视频的节奏可能会受到运动和静态场景的影响。

---

## 23. ForceBody: Force-Paired Parametric Body Motion with Torque Uncertainty

**arXiv ID:** 2611.22769 | [PDF](https://arxiv.org/pdf/2611.22769v1)

**作者:** Joonwoo Kwon `[一作]` (Michigan State University), Zijun Cui `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了ForceBody数据集，这是第一个将测量的地面反作用力（GRF）和逆动力学关节扭矩与SKEL参数化身体模型配对的大规模数据集，并为每个扭矩标签发布了逐帧、逐关节的Monte Carlo不确定性。

**💡 创新点**

创新点在于首次将真实的力监督引入参数化身体表示，并提供每个扭矩标签的逐帧不确定性注释，填补了现有数据集的空白。

**🔧 技术方法**

使用了Monte Carlo采样技术来估计扭矩的不确定性，并评估了六种不同的模型架构（MLP、Conv1D、LSTM、GRU、Mamba2、Transformer）在地面反作用力和关节扭矩预测任务上的表现。

**📊 数据集**

使用的数据集包含来自140名受试者的10,386个运动试验（9.7M帧，26.9小时），每个试验都配有SKEL姿态、身体形状参数、测量的GRF和关节扭矩。

**📈 对比分析**

与AddBiomechanics基线进行比较，ForceBody训练的模型在地面反作用力和关节扭矩预测上表现相当，且在使用不确定性加权的Transformer模型中，关节扭矩的平均绝对误差（MAE）减少了5%。

**⚠️ 局限性**

限制在于当前的拟合阶段使用SKEL作为参数化身体模型，无法处理SKEL形状空间之外的非典型身体形状。此外，发布的扭矩不确定性仅建模关节角度噪声，并未涵盖GRF和重心位置误差、段惯性或SKEL与Rajagopal模型之间的轴不匹配。

---

## 24. Call-Efficient ICD-10 Code Prediction via Supervised Candidate Generation and Selective LLM Invocation

**arXiv ID:** 2611.21891 | [PDF](https://arxiv.org/pdf/2611.21891v1)

**作者:** Ilia Kopanichuk `[一作]` (Moscow Institute of Physics and Technology), Dmitry V. Dylov `[通讯]` (AIRI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种自动分配ICD-10代码的管道，结合了变换器嵌入、基于树的监督候选生成、候选级重排序和门控LLM辅助。

**💡 创新点**

创新点在于通过门控LLM辅助的监督候选生成管道进行端到端评估，展示了LLM在不确定示例上的有效重排序作用。

**🔧 技术方法**

使用了变换器嵌入、随机森林和HistGradientBoosting等技术，以及门控LLM进行选择性重排序。

**📊 数据集**

使用了包含8552个训练样本和1511个锁定测试样本的短临床诊断文本数据集，数据为俄语。

**📈 对比分析**

与直接零-shot GPT-5.6-Sol基线进行比较，最终的Pre-LLM融合方法在锁定测试集上达到83.19%的Top-1准确率，且仅在31.97%的案例中调用LLM，显著降低了错误率。

**⚠️ 局限性**

限制在于研究基于当前可用的数据集，不能直接与其他系统进行状态最优比较，且稀有类的分析仍未解决。

---

## 25. Worldviews in Large Language Models

**arXiv ID:** 2611.22670 | [PDF](https://arxiv.org/pdf/2611.22670v1)

**作者:** Edmundo Lopez-Sola `[一作]` (University of Oxford), Ruben Laukkonen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了大型语言模型（LLMs）的世界观，认为其信念对模型行为有重要影响。通过对八个模型进行心理学世界观调查，发现这些模型在许多关键世界观立场上存在一致性。

**💡 创新点**

创新点在于将世界观作为LLMs行为的一个可测量和可塑层次，强调其在模型对齐中的重要性，并提出通过系统提示和对话可以有效改变模型的世界观。

**🔧 技术方法**

使用了心理学中的世界观调查工具，包括Primals Inventory (PI-18)和基于Koltko-Rivera维度的世界观调查。

**📊 数据集**

研究涉及八个来自六个实验室的LLMs，包括Claude Sonnet 4.5、DeepSeek V4 Flash、Qwen3 235B等，使用了多次测量的方式进行数据收集。

**📈 对比分析**

通过与人类的世界观进行比较，发现LLMs在多个维度上得分低于人类平均水平。对比不同的提示和对话，发现身份提示对世界观的影响较小，而沉思提示和对话则显著改善了模型的安全性和世界观。

**⚠️ 局限性**

限制在于所使用的调查工具是为人类设计的，LLMs的回答可能受到问题措辞的影响。此外，模型与人类之间的差距是基于特定的自选样本，且世界观与安全性之间的关联是相关性的，尚需进一步研究因果关系。

---

## 26. Pressure Drop: Gravity Vector Estimation from a Distributed Barometric Pressure Sensor Array

**arXiv ID:** 2611.23120 | [PDF](https://arxiv.org/pdf/2611.23120v1)

**作者:** William Dubois `[一作]` (Northern Robotics Laboratory Université Laval), François Pomerleau `[通讯]` (Northern Robotics Laboratory Université Laval)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种利用分布式气压传感器阵列直接观察重力方向的方法，并通过理论评估和实验验证了该方法的有效性。

**💡 创新点**

创新点在于通过气压传感器直接估计重力向量，而不是依赖于IMU的间接观测，且在动态场景中表现出色。

**🔧 技术方法**

使用了气压传感器进行重力向量估计，并进行了噪声敏感性分析和可观测性分析。

**📊 数据集**

使用了多个分布式的气压传感器进行实验，具体包括DPS310气压传感器和Setra模型278工业气压传感器。

**📈 对比分析**

与基于IMU的方法进行了比较，结果显示在高度动态场景中，提出的方法与IMU系统的性能相似，甚至在加速度计饱和的特定情况下表现更好。

**⚠️ 局限性**

限制在于需要至少三个传感器才能进行有效估计，并且在较长的传感器间距下可能会增加误差。

---

## 27. PersonaHate: A Scalable Persona-Based Data Synthesis Pipeline for Hate Speech Analysis

**arXiv ID:** 2611.21951 | [PDF](https://arxiv.org/pdf/2611.21951v1)

**作者:** Xinyu Zhang `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可扩展的基于角色的数据合成管道PersonaHate，用于构建多样化的仇恨言论数据集，以提高仇恨言论检测的性能。

**💡 创新点**

创新点在于使用457K个角色从真实世界来源中提取，并通过多个大型语言模型生成多样化的仇恨言论，结合多评审标注以获得可靠标签。

**🔧 技术方法**

使用了八个大型语言模型（LLMs）进行仇恨言论生成，并采用多评审标注机制来确保标签的可靠性。

**📊 数据集**

构建了一个包含791K样本的语料库，并从中策划出67K个平衡的训练集，涵盖34个身份群体。

**📈 对比分析**

与现有的检测器进行比较，所有训练的模型在四个广泛使用的外部基准上表现出竞争力，尤其是DeBERTa-v3模型在PersonaHate上训练后达到了77.5%的平均F1和79.4%的平均准确率，超越了OpenAI的Moderation API。

**⚠️ 局限性**

限制在于合成数据无法完全替代人类创作的仇恨言论基准，且标签质量和模糊案例的处理仍然是挑战。

---

## 28. Depth-Aware Activation Steering via Attention Residuals

**arXiv ID:** 2611.21655 | [PDF](https://arxiv.org/pdf/2611.21655v1)

**作者:** Shilong Zhao `[一作]` `[通讯]` (Independent Researcher), Shilong Zhao (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种轻量级的深度感知引导方法，通过将个性向量根据深度权重分配到多个层次，从而改善语言模型的行为和内部对齐。

**💡 创新点**

创新点在于结合了注意力残差（AttnRes）和雅可比透镜（Jacobian-lens）风格的概念读取，形成了一种深度感知的引导管道，能够在多个层次上进行有效的干预。

**🔧 技术方法**

使用了注意力残差（AttnRes）和雅可比透镜（J-space）风格的概念读取技术。

**📊 数据集**

使用了两个小型开放模型（Qwen3-0.6B和Qwen3.5-0.8B），并在35个固定的中文提示上进行了实验。

**📈 对比分析**

与单层注入方法相比，深度分布方法在生成长度和内部概念指标上均表现出显著改善（生成长度减少18.6个字符，p=0.003），并且所有三个内部概念指标均有显著提升（p<0.0001）。

**⚠️ 局限性**

限制在于仅使用了两个小模型，且在模型规模和生成之间存在混淆；使用了单一的评估模型进行指标验证；主要实验中的深度权重在初始化时使用，未将训练变体整合到主要比较中；缺乏标准的下游基准测试。

---

## 29. Structure--Residual Inductive Biases for Language Model Pretraining

**arXiv ID:** 2611.23197 | [PDF](https://arxiv.org/pdf/2611.23197v1)

**作者:** Hai Huang `[一作]` (Palona AI), Xue Liu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文研究了表示结构是否应作为显式归纳偏置，提出了边界瓶颈概念（BBC）和连续概念（CC）作为语言模型预训练的结构-残差偏置。

**💡 创新点**

创新点在于引入了结构-残差原则，通过BBC和CC两种方式实现，分别优化了跨段通信和隐藏状态轨迹的几何结构。

**🔧 技术方法**

使用了边界瓶颈概念（BBC）和连续概念（CC）两种技术，分别通过边界载体和光滑的隐藏状态轨迹进行优化。

**📊 数据集**

使用了GPT-2模型进行实验，训练数据集包含300M到20B的训练标记。

**📈 对比分析**

与传统方法相比，BBC和CC在不同层次上表现出不同的深度偏好，BBC在早期层效果最佳，而CC在后期层效果最佳。结合BBC和CC在GPT-2 Medium上提高了NLL 1.44%。

**⚠️ 局限性**

限制在于BBC和CC的效果依赖于特定的层次，且在训练过程中可能会出现数值稳定性问题。

---

## 30. NONSENSE: Non-Semantic Strings Expose Semanticity Gaps in Dense Retrieval

**arXiv ID:** 2611.22353 | [PDF](https://arxiv.org/pdf/2611.22353v1)

**作者:** Ankit Gupta `[一作]` `[通讯]` (Michigan State University), Ankit Gupta (Michigan State University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了密集检索在现代搜索和问答中的表现，提出了一种名为NONSENSE的黑箱稀疏编辑探针，构造明显非语义的字符串，并优化查询与字符串之间的密集相似度。

**💡 创新点**

创新点在于NONSENSE探针能够生成非语义字符串，并且这些字符串在多个数据集上经常超越检索器自身的最佳判断相关段落（QRP），揭示了密集检索中的语义性差距。

**🔧 技术方法**

使用了黑箱稀疏编辑技术，通过字符替换优化查询与字符串之间的密集相似度。

**📊 数据集**

使用了五个BEIR风格的数据集，包括SciFact、SciDocs、FIQA、NFCorpus和ArguAna。

**📈 对比分析**

与判断证据进行比较，NONSENSE字符串在多个数据集上经常超越检索器的最佳判断相关段落，成功率高达99.0%。在SciFact上，NONSENSE在97.7%的查询中超越了QRP，且中位排名为1。

**⚠️ 局限性**

限制在于NONSENSE目前仅适用于三种英语单向量检索器，且优化字符串在不同检索器之间的迁移性不强，未能提供人类可读的证据。

---

## 31. Right Labels, Wrong Reasons: Signed Feature Attribution for Multi-Label Explainability

**arXiv ID:** 2611.22515 | [PDF](https://arxiv.org/pdf/2611.22515v1)

**作者:** Kenton White `[一作]` `[通讯]` (Algonquin College), Kenton White (Algonquin College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的多标签分类器解释方法FLEX（Fisher LDA Explainer），通过在全二进制标签矩阵上拟合单个直接的多标签Fisher LDA，生成每个标签的签名特征归因列表。

**💡 创新点**

FLEX通过共享的协方差池化提高了解释的稳定性和准确性，尤其在标签数量多且正例稀缺的情况下，表现出优越性。其权重集中度提供了一个黑箱诊断，帮助判断分类器的预测是否基于主导特征。

**🔧 技术方法**

使用了Fisher线性判别分析（LDA）作为主要技术，并通过共享的协方差矩阵来提高模型的解释能力。

**📊 数据集**

在五个标准多标签基准数据集上进行评估，涵盖基因组学、音乐、图像和电子邮件领域，标签数量和空间大小各异。

**📈 对比分析**

与六种分类器进行比较，FLEX在大多数数据集上表现出更高的AUC和F1分数，尤其在标签数量多的情况下，FLEX的性能优于传统的二元相关方法，AUC提升可达0.180。

**⚠️ 局限性**

FLEX的局限性在于它是通过一个线性代理和一个协方差估计来近似黑箱，当黑箱分类器依赖于高度非线性交互或协方差矩阵条件不良时，FLEX的准确性会下降。

---

## 32. When AUC Meets Generative Reconstruction: A Unified Principle for Balanced Visual Representation Learning

**arXiv ID:** 2611.22094 | [PDF](https://arxiv.org/pdf/2611.22094v1)

**作者:** Boyu Han `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Qingming Huang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的表示学习范式，将生成重建重新表述为生成输出流形中的判别排序问题，以解决对比学习和生成重建之间的优化冲突。

**💡 创新点**

创新点在于通过将生成重建与判别约束相结合，提出了AUC生成重建（AUCGR）和部分AUC生成重建（pAUCGR），有效地平衡了判别能力和细粒度感知能力。

**🔧 技术方法**

使用了AUC优化和生成模型（如扩散模型和流模型）技术，结合了对比学习和生成重建的优点。

**📊 数据集**

在超过30个视觉基准和多模态大语言模型（MLLMs）上进行了广泛的实验验证。

**📈 对比分析**

与传统方法相比，AUCGR和pAUCGR在多个基准测试中表现出更高的判别准确性和细粒度感知能力，建立了视觉表示学习的新基准。

**⚠️ 局限性**

限制在于AUCGR可能受到噪声负样本的影响，pAUCGR通过限制负样本的选择来缓解这一问题，但仍需进一步优化负样本的选择策略。

---

## 33. A Unified Choice Model of News Selection: Experimental Evidence on Stance, Rank, and Engagement Cues

**arXiv ID:** 2611.23079 | [PDF](https://arxiv.org/pdf/2611.23079v1)

**作者:** Jacopo D'Ignazi `[一作]` (Universitat Pompeu Fabra), Vicenç Gómez `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一的离散选择框架，使用条件多项式逻辑模型共同估计和比较政治立场、排名位置和视觉参与线索对新闻选择的影响。

**💡 创新点**

创新点在于将政治立场、排名位置和视觉参与线索整合到一个模型中，并提供了因果证据，表明这些因素对文章选择的影响。

**🔧 技术方法**

使用条件多项式逻辑模型（CML）来建模用户在排名列表中选择新闻文章的过程。

**📊 数据集**

使用了370名参与者在四个新闻主题下的1480个选择数据，实验分为有和没有视觉参与线索的两种条件。

**📈 对比分析**

通过交叉验证比较不同模型的预测性能，发现仅使用参与线索的模型在预测点击方面优于仅使用政治立场或排名位置的模型，综合所有因素的模型表现最佳。

**⚠️ 局限性**

限制在于两个实验条件之间相隔六个月，参与者没有随机分配，可能存在样本间的差异。因果证据主要来自于E2中随机化的参与计数，而不是参与线索可见性的影响。

---

## 34. Proceedings of the 2026 XCSP3 Competition

**arXiv ID:** 2611.22420 | [PDF](https://arxiv.org/pdf/2611.22420v1)

**作者:** Gilles Audemard `[一作]` (CRIL University of Artois), Emmanuel Lonca `[通讯]` (CRIL University of Artois)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文总结了2026年国际约束解决方案竞赛的过程和结果，介绍了所选问题、模型及其应用。

**💡 创新点**

创新点在于选择了31个新问题，并对问题的多样性和难度进行了评估，以避免过拟合和确保问题的代表性。

**🔧 技术方法**

使用了3-core格式来表示组合约束问题，并采用了多种约束求解技术。

**📊 数据集**

使用的数据集包括31个问题的实例，这些实例可以在指定的链接中找到。

**📈 对比分析**

通过与其他求解器的比较，评估了所提出的求解器在不同问题上的表现，结果显示大多数求解器在处理新问题时表现良好。

**⚠️ 局限性**

限制在于某些求解器由于选择偏差而被标记为非竞争性，可能影响结果的公正性。

---

## 35. Synthetic Crystal Data Generation for Diffusion Pretraining in Material Property Prediction

**arXiv ID:** 2611.22140 | [PDF](https://arxiv.org/pdf/2611.22140v1)

**作者:** Aamod Varma `[一作]` (Georgia Institute of Technology), Victor Fung `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究将扩散基础的预训练与合成结构生成相结合，以减少对局部能量表面附近结构的依赖，从而提高材料属性预测的可扩展性和结构多样性。

**💡 创新点**

创新点在于引入了一种合成数据生成管道，结合了扰动-放松、MatterGen生成采样和原型结构枚举三种策略，显著提高了预训练数据的多样性和规模。

**🔧 技术方法**

使用了机器学习的原子间势能（MLIP）来替代每个生成结构的DFT放松计算，从而加速了数据生成过程。

**📊 数据集**

使用了来自Materials Project (MP)的20,000个DFT放松结构作为基准数据集，并通过合成策略生成了165,000个结构。

**📈 对比分析**

与随机初始化相比，所有合成预训练条件在七个源比较任务中均表现出数值上的提升。随着原型结构数据集规模的扩大，性能在大多数任务中与MP-20k相匹配或超越。

**⚠️ 局限性**

限制在于实验仅使用了单一的GNN骨干网络和一个MatBench折叠，观察到的趋势可能无法在不同的折叠、预训练种子、检查点选择、模型架构或下游属性中普遍适用。

---

## 36. Not All Evidence Is Admissible: Evidence Governance for Tool-Augmented Medical Agents

**arXiv ID:** 2611.22169 | [PDF](https://arxiv.org/pdf/2611.22169v1)

**作者:** Runxun Zhang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jingwei Wei `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的医疗代理框架，通过证据治理来解决工具输出导致的证据劫持问题。

**💡 创新点**

首次揭示并系统量化了医疗代理中的证据劫持现象，并提出了证据治理理论，恢复了推理器的选择性无知。

**🔧 技术方法**

使用了证据治理框架，包括可验证的工具卡、独立的检查器和仅从被接受的卡片中回答的推理器，且无需训练。

**📊 数据集**

在三个任务上进行了评估：眼底青光眼筛查（使用REFUGE2数据集）、胸部X光肺炎分类（使用Kermany数据集）和医学问答（使用MedQA、MedMCQA和NEJM数据集）。

**📈 对比分析**

与现有的医疗模型和代理系统相比，提出的方法在所有三个任务上都达到了最高的准确率，并显著优于使用工具的代理和多代理系统，且在推理速度上更快，成本更低。

**⚠️ 局限性**

该方法的局限性在于它依赖于工具输出的质量，若工具输出错误，可能仍会影响最终的决策。

---

## 37. SWIR-cGAN: Synthesizing SWIR Bands Using VNIR Imagery from CBERS and AMAZONIA-1

**arXiv ID:** 2611.22922 | [PDF](https://arxiv.org/pdf/2611.22922v1)

**作者:** Alisson Cleiton de Oliveira `[一作]`, Thales Sehn Körting `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种条件生成对抗网络（cGAN），用于从CBERS和AMAZONIA-1的可见光和近红外影像合成短波红外（SWIR）波段。

**💡 创新点**

创新点在于使用增强注意力的U-net生成器和多损失目标，解决了跨卫星泛化和光谱距离较远波段的合成挑战。

**🔧 技术方法**

采用了条件生成对抗网络（cGAN），结合了U-net架构、注意力机制和多损失函数。

**📊 数据集**

使用了来自CBERS和AMAZONIA-1的4,125个训练和交叉验证补丁，以及1,026个未见补丁的测试数据集。

**📈 对比分析**

与U-net、ESRGAN和ViT-U-net等基线模型进行比较，SWIR-cGAN在SSIM、PSNR和MAE等指标上表现最佳，尤其在SWIR-2波段上表现更为突出。

**⚠️ 局限性**

限制在于训练数据集的大小和多样性，可能限制了模型在不同地理区域、季节和土地覆盖类型上的泛化能力。

---

## 38. Escaping Latent Traps by Remembering: Training-Free Episodic Kernel Repulsion for World-Model Planning

**arXiv ID:** 2611.22470 | [PDF](https://arxiv.org/pdf/2611.22470v1)

**作者:** Yican Zhao `[一作]` (Institute of Integrated Circuits and Artificial Intelligence), Zelin Zang `[通讯]` (Institute of Integrated Circuits and Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种新的成本项Episodic Kernel Repulsion，旨在通过惩罚候选计划与代理已访问状态的接近程度，帮助代理逃离潜在的局部最小值（latent traps），从而提高规划成功率。

**💡 创新点**

创新点在于引入了一种训练无关的成本塑形方法，通过记忆代理的历史状态来避免重复访问，从而提高规划效率和成功率。

**🔧 技术方法**

使用了Episodic Kernel Repulsion技术，该技术通过对候选计划施加基于高斯核的惩罚项来实现。

**📊 数据集**

在两个JEPA世界模型和四个基准测试上进行了实验，具体包括AntMaze、FourRoom、Push-T和Cube等环境。

**📈 对比分析**

与基线方法相比，Episodic Kernel Repulsion在所有八个设置中均提高了成功率，LeWM模型的平均成功率从31.8%提升至46.7%，H-JEPA模型从54.7%提升至64.0%。在大多数情况下，该方法在相同的计算预算下表现优于增加搜索计算的传统方法。

**⚠️ 局限性**

局限性包括在死胡同中逃脱的速度较慢，虽然减少了被困的情况，但并未保证快速逃脱。此外，记忆可能会在接近目标时推开代理，导致偶尔的失败。

---

## 39. FCP: Formal Context Protocol for Verifying LLM-Generated Artifacts

**arXiv ID:** 2611.21679 | [PDF](https://arxiv.org/pdf/2611.21679v1)

**作者:** Shashank `[一作]` (University of Southern California), Shashank `[通讯]` (University of Southern California)

**通讯引用:** 75 | [OpenAlex ID](https://openalex.org/A5072603496)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种形式上下文协议（FCP），用于检查AI生成的工件是否符合请求，独立于其自身的测试和断言。

**💡 创新点**

FCP通过将符合性与忠实性分开，提供了一种新的验证方法，能够识别和处理AI生成内容中的错误和矛盾。

**🔧 技术方法**

使用了形式验证技术，包括模型检查器和SMT求解器，结合了机械证据和具体问题的提出。

**📊 数据集**

在四个硬件设计、一个提示和十个代码任务上进行了评估，涉及多个AI模型生成的工件。

**📈 对比分析**

与自检基线相比，FCP在硬件设计中未认证任何违反要求的情况，而自检基线认证了23%的违反要求。在代码任务中，FCP认证了3个违反要求，而自检基线认证了所有33个违反要求。

**⚠️ 局限性**

FCP的局限性在于其依赖于请求的准确性和完整性，且在处理复杂或模糊的请求时可能会遇到挑战。

---

## 40. Learning Agent-Centric Vision-Language-Action World Models for Autonomous Driving

**arXiv ID:** 2611.22880 | [PDF](https://arxiv.org/pdf/2611.22880v1)

**作者:** Guoqing Wang `[一作]` (Shanghai Jiao Tong University), Chao Ma `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为AgentWorld的代理中心视觉-语言-行动（VLA）世界模型，旨在通过离散代理令牌进行推理和轨迹规划，从而改善端到端的自动驾驶能力。

**💡 创新点**

创新点在于引入了一种紧凑且结构化的代理中心潜在状态，能够捕捉驾驶场景中对象之间的交互，超越了仅依赖像素级预测的传统方法。

**🔧 技术方法**

使用了显式几何锚定、面向推理的监督微调和面向代理的强化学习等技术。

**📊 数据集**

在nuScenes和Bench2Drive数据集上进行了评估，涵盖了开放环路规划和闭环驾驶。

**📈 对比分析**

与现有的VLA世界模型相比，AgentWorld在L2误差和碰撞率上表现出显著的改进，尤其是在长时间预测和安全关键的碰撞指标上，显示出代理中心世界状态提供了更可靠的未来运动推理线索。

**⚠️ 局限性**

限制在于AgentWorld仍依赖于显式几何监督来将代理令牌锚定到物理参与者，而不是完全依赖于端到端的规划监督。未来的工作将探索更具可扩展性的自监督锚定方法。

---

## 41. The Shared Word: Evaluating Language Models on Dutch Word Puzzles

**arXiv ID:** 2611.22935 | [PDF](https://arxiv.org/pdf/2611.22935v1)

**作者:** Rik van Noord `[一作]` (University of Groningen), Maurice Voors `[通讯]` (University of Groningen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个荷兰语基准测试，名为The Shared Word，旨在评估大型语言模型在荷兰语中的表现。

**💡 创新点**

创新点在于基准测试是语言特定的、开放词汇的、经过人工审核的，并且可以确定性评分。

**🔧 技术方法**

使用了自动过滤和人工策划的技术来构建数据集。

**📊 数据集**

数据集来源于荷兰维基百科的消歧义页面，最终包含222个实例，每个测试项有三个独立的人类答案。

**📈 对比分析**

与其他方法比较时，开放权重模型的表现普遍较差，最高得分为39.3%；而封闭源模型如GPT-5.6 Sol在最大推理下得分为88.8%。

**⚠️ 局限性**

限制在于基准测试相对较小，仅包含200个经过精心策划的测试项，可能限制了统计能力和模型区分能力。

---

## 42. Can Activation Steering Substitute for Prompt-Based Affective Priming in LLM Digital Twins?

**arXiv ID:** 2611.22738 | [PDF](https://arxiv.org/pdf/2611.22738v1)

**作者:** Junyi Fan `[一作]` `[通讯]` (University of Southern California), Junyi Fan (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究测试了激活引导是否可以替代传统的情感启动方法，通过让参与者回忆特定事件并测量其情感状态对相关判断的影响。

**💡 创新点**

创新点在于比较了基于提示的情感启动与激活注入的效果，发现激活注入在某些情况下可以替代传统方法，但效果并不一致。

**🔧 技术方法**

使用了Llama-3.1-8B-Instruct模型进行生成和激活注入，采用了激活引导技术。

**📊 数据集**

使用了Twin-2K-500 Mega Study中的100个个性数据集，涵盖了感恩和缺乏控制两个领域。

**📈 对比分析**

通过与原始研究的结果进行比较，发现真实提示路径的效果介于人类和数字双胞胎之间，而激活注入在某些强度下能够匹配真实提示的效果，但在其他情况下则超出或偏离了预期。

**⚠️ 局限性**

限制在于只使用了一个基础模型和一种提取方法，激活控制臂只在单一强度下测试，且未进行等效性测试，结果的可解析性也受到限制。

---

## 43. Learning Regions of Attraction for Transverse Feedback Linearization under Controller Singularities

**arXiv ID:** 2611.22286 | [PDF](https://arxiv.org/pdf/2611.22286v1)

**作者:** Abdullah Waqar Malik `[一作]` (National University of Sciences and Technology), Adeel Akhtar `[通讯]` (New Jersey Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文探讨了横向反馈线性化（TFL）在路径跟踪中的应用，提出了一种新的方法来确定受限吸引区域，并通过学习安全值函数来优化控制器的性能。

**💡 创新点**

创新点在于将Zubov值函数分解为已知的二次成本函数和安全值函数，仅学习后者，从而避免了对无限期轨迹标签的需求。

**🔧 技术方法**

使用了拟合值迭代技术，并结合了区间分支和界限方法来验证内层，外层则通过样本测试。

**📊 数据集**

实验使用了动态扩展单轮车和一个平面四旋翼的模拟数据集，分别跟踪圆形和正弦路径。

**📈 对比分析**

与传统的二次成本函数和两个物理信息神经网络（PINN）求解器相比，所提出的方法在单轮车和四旋翼的模拟吸引区域覆盖率分别达到了72.0%、83.4%和19.1%，而传统方法的覆盖率最高仅为11.3%、4.0%和0.0%。

**⚠️ 局限性**

限制在于该方法需要模型，并且未尝试在空间四旋翼上的应用。此外，外层的估计仍有提升空间。

---

## 44. SpatialCast: Reading the 3D World via Cast Shadows

**arXiv ID:** 2611.22657 | [PDF](https://arxiv.org/pdf/2611.22657v1)

**作者:** Enhan Li `[一作]` (University of Hong Kong), Hongyang Du `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个新的基准，利用投影阴影评估多模态大型语言模型（MLLMs）在空间推理方面的能力，特别是如何将物理场景条件与视觉观察联系起来。

**💡 创新点**

创新点在于使用投影阴影作为物理证据来评估模型的空间推理能力，并设计了三个互补的任务来测试场景与投影的一致性、预测干预引起的变化以及从观察中推断物理变化。

**🔧 技术方法**

使用了BlenderProc和Blender Cycles构建可控的3D场景，并进行了基于GRPO的后训练以提高模型性能。

**📊 数据集**

基准包含21,000个合成问题和200个来自20个拍摄场景的SpatialCast-Real问题。

**📈 对比分析**

与现有的开放权重模型相比，闭源模型在基准测试中表现更好，但仍在阴影长度和光高度判断上存在困难。通过GRPO后训练，开放权重模型在基准测试中的表现有所提升，并在多个外部空间基准上显示出改进。

**⚠️ 局限性**

限制在于现有的开放权重模型在合成基准测试中仍接近随机水平，表明这些物理投影问题对当前模型仍具有挑战性。

---

## 45. Satisfiability Modulo Theories for Robust Synthesis of Signal Temporal Logic Specifications

**arXiv ID:** 2611.23021 | [PDF](https://arxiv.org/pdf/2611.23021v1)

**作者:** Philip Smith `[一作]` (Worcester Polytechnic Institute), Kevin Leahy `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种新的合成方法，利用可满足性模理论（SMT）来处理信号时序逻辑（STL）规范的合成问题，旨在提高长时间部署和环境变化下的适应能力。

**💡 创新点**

创新点在于引入了一种基于OMT的最大化鲁棒性的STL合成编码方法，以及针对离散时间动态系统的过渡系统编码，展示了SMT在长时间任务中的有效性。

**🔧 技术方法**

使用了可满足性模理论（SMT）和优化模理论（OMT）技术进行合成，并通过递归编码STL约束来实现。

**📊 数据集**

使用了随机生成的环境和能力集，以及Capability Temporal Logic (CaTL)作为数据集进行实验比较。

**📈 对比分析**

与当前最先进的混合整数线性规划（MILP）方法进行了实证比较，结果显示SMT在长时间任务中表现更佳，解决时间比MILP快50%到96%。

**⚠️ 局限性**

限制在于SMT方法在处理大量代理时性能下降，未来工作将扩展动态编码以适应更广泛的系统。

---

## 46. RoadAnnoter: An AI-Assisted Desktop Tool for Semantic Segmentation Annotation of Road-Scene Images

**arXiv ID:** 2611.22884 | [PDF](https://arxiv.org/pdf/2611.22884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 47. Stochastic Modeling of the Mean Time Between Software Failures: A review

**arXiv ID:** 2611.21619 | [PDF](https://arxiv.org/pdf/2611.21619v1)

**作者:** Gabriel Pena `[一作]` (Universidad Nacional de Tres de Febrero), Nestor Barraza `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对多种软件故障检测的随机模型进行了综述，重点分析了如何从这些模型中获得平均故障间隔时间，并进行了比较研究。

**💡 创新点**

创新点在于引入了基于传染的模型，与传统的非齐次泊松过程模型相比，考虑了故障率不仅依赖于时间，还依赖于之前检测到的故障数量。

**🔧 技术方法**

使用了非齐次出生过程、马尔可夫过程等数学技术，建立了故障检测过程的随机模型。

**📊 数据集**

使用了来自Grupo Esfera S.A.的两个软件项目的数据集，这些项目采用了敏捷开发方法。

**📈 对比分析**

通过非线性最小二乘法对模型参数进行了校准，并与传统的非齐次泊松过程模型进行了比较，结果显示基于传染的模型在早期故障检测中表现良好。

**⚠️ 局限性**

限制在于模型对数据质量敏感，异常值和测量方法的变化可能导致结果不准确，且在某些情况下，无法定义平均故障时间（MTTF）。

---

## 48. BugGen 2.0: Rethinking the Roles of LLMs for Scalable RTL Bug Synthesis

**arXiv ID:** 2611.23091 | [PDF](https://arxiv.org/pdf/2611.23091v1)

**作者:** Minh Luu `[一作]` (Texas A&M University), Jiang Hu `[通讯]` (Texas A&M University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

论文提出了BugGen 2.0，一个用于可扩展RTL错误合成的系统，改进了BugGen 1.0的工作流程，通过将结构分析、目标选择和验证分配给确定性算法，而将LLM专注于局部语义重写。

**💡 创新点**

创新点在于引入了混合工作流程和变体增强机制，显著降低了推理和仿真成本，同时提高了模型和设计的一致性。

**🔧 技术方法**

使用了确定性算法进行结构分析和目标选择，同时利用大型语言模型（LLM）进行局部重写。

**📊 数据集**

在四个OpenTitan IP块和FabScalar处理器核心上进行了评估，使用了多种LLM模型。

**📈 对比分析**

与BugGen 1.0和Firefly相比，BugGen 2.0在相同模型比较中减少了6.8×–15.4×的令牌使用，相对Firefly减少了1.5×–7.9×，并且在每个请求的可编译错误场景中达到了所有请求的配额。

**⚠️ 局限性**

限制在于当前评估未能表征单个错误如何在模块中传播或影响架构行为，未来可以通过更强的预仿真预测和传播感知分析来进一步降低仿真成本。

---

## 49. OICycle: Observe, Infer, and Intervene Cycle for Belief-State Learning in Longitudinal Clinical Decision Making

**arXiv ID:** 2611.22826 | [PDF](https://arxiv.org/pdf/2611.22826v1)

**作者:** Jitindra Fartiyal `[一作]` (Aston University), Sergei G. Solovski `[通讯]` (Aston University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种新的临床决策框架OICycle，通过观察、推断和干预循环来学习患者的潜在状态。

**💡 创新点**

OICycle通过直接从回顾性轨迹中学习随机信念状态，解决了传统POMDP方法在临床数据中的局限性，尤其是在处理不完整和异构数据时。

**🔧 技术方法**

使用了递归神经网络（GRU）作为学习的递归操作符，结合了多模态观察和干预信息。

**📊 数据集**

使用了来自MIMIC-IV、MIMIC-IV-Note和MIMIC-CXR的数据集，构建了一个包含72,075名患者的多模态纵向临床数据集。

**📈 对比分析**

与单快照模型OICycle-S相比，OICycle在下一个干预的自适应召回率、下一个事件的时间预测和下一个事件的死亡率预测上均有显著提高，具体表现为自适应召回率从0.253提高到0.347，时间预测的宏F1从0.293提高到0.351，死亡率预测的宏F1从0.585提高到0.649。

**⚠️ 局限性**

OICycle的限制在于其学习的潜在动态不应被解释为唯一识别潜在生理疾病动态，且数据集的规模仍小于临床决策所需的多模态场景。

---

## 50. ATS: Anti-Aliased Triangle Splatting with Exact Per-Primitive Integrals

**arXiv ID:** 2611.22512 | [PDF](https://arxiv.org/pdf/2611.22512v1)

**作者:** Brent Zoomers `[一作]` (Digital Future Lab, Flanders Make, Hasselt University), Nick Michiels `[通讯]` (Digital Future Lab, Flanders Make, Hasselt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的三角形溅射方法，通过推导三角形溅射窗口在像素上的确切闭合形式积分，解决了传统方法中的混叠问题，并改进了三角形的剔除策略。

**💡 创新点**

创新点在于提供了三角形溅射窗口的确切闭合形式积分，显著减少了混叠现象，并允许更有效的多尺度训练，同时在低分辨率下表现出更好的时间稳定性。

**🔧 技术方法**

使用了闭合形式积分和格林定理来推导三角形溅射窗口的积分，并将其集成到CUDA内核中以提高渲染效率。

**📊 数据集**

使用了Mip-NeRF 360场景、Tanks and Temples数据集、Deep Blending数据集和NeRF Synthetic数据集进行实验。

**📈 对比分析**

与传统的三角形溅射和超采样方法相比，本文的方法在1/8分辨率下的PSNR表现更好，且在多尺度训练中表现出更高的稳定性和更少的闪烁现象。

**⚠️ 局限性**

限制在于引入额外的计算增加了渲染时间，相较于基线三角形溅射和2×2超采样，渲染效率有所下降。

---

## 51. A Transformer-Derived Iterative Preconditioner

**arXiv ID:** 2611.22941 | [PDF](https://arxiv.org/pdf/2611.22941v1)

**作者:** Patrick Lutz `[一作]` (Boston University), Venkatesh Saligrama `[通讯]` (Boston University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文展示了经过训练的变换器在上下文预测问题上可以学习已知的数值算法，并进一步揭示了其内部计算可以发现具有可证明保证的新算法。具体来说，训练一个softmax注意力变换器来解决线性系统Xw=y，隐式改善了矩阵的条件性，并开发出一个独立的算法RAPID。

**💡 创新点**

创新点在于发现了一个新的预处理原语，并将其发展为一个独立的、无需学习组件的算法RAPID，能够有效地改善对称正定矩阵的条件性。

**🔧 技术方法**

使用了随机混合和稀疏局部修正的技术，结合了高效的结构化变换和稀疏更新，形成了RAPID算法。

**📊 数据集**

使用了合成的光谱族和31个SuiteSparse SPD矩阵作为数据集，展示了RAPID在这些数据集上的有效性。

**📈 对比分析**

与随机化Nyström预处理方法相比，RAPID在所有测试的光谱族和SuiteSparse矩阵上都显著改善了条件性，并减少了达到固定精度所需的共轭梯度迭代次数，显示出其优越的性能。

**⚠️ 局限性**

限制在于理论保证主要针对理想化的Haar混合和块限制行选择，实际算法的收敛保证尚未完全建立，且需要优化实现以确保竞争性的端到端运行时间。

---

## 52. "I Just Don't Want to Be a Burden": Investigating AI Chatbot Emotional Support Among People with Suicidal Ideation and Other Mental Health Concerns

**arXiv ID:** 2611.23190 | [PDF](https://arxiv.org/pdf/2611.23190v1)

**作者:** Vasudha Varadarajan `[一作]` (Carnegie Mellon University), Maarten Sap `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究探讨了在经历自杀意念和其他心理健康问题的人群中，AI聊天机器人作为情感支持的使用情况、驱动因素和用户体验。

**💡 创新点**

创新点在于通过对Reddit帖子进行定性分析，识别出用户使用AI聊天机器人的动机和体验，并通过调查验证这些发现，强调了用户对AI支持的需求和期望。

**🔧 技术方法**

采用了混合方法研究设计，结合了定性分析和定量调查，分析了Reddit数据并开发了针对202名有心理健康问题的成人的调查问卷。

**📊 数据集**

数据集包括来自Reddit的334万条帖子，涉及心理健康问题和AI使用的讨论，以及对202名参与者的调查数据。

**📈 对比分析**

与传统的心理健康支持相比，参与者更倾向于使用AI聊天机器人以避免对他人的负担，尽管大多数人正在接受临床治疗。调查结果显示，使用AI的好处普遍存在，但有自杀意念的参与者报告的危害和依赖性更高。

**⚠️ 局限性**

本研究的局限性包括使用公共Reddit帖子进行分析，可能未能全面反映所有用户的体验；调查设计为横断面，无法建立因果关系；以及未区分不同聊天机器人模型的影响。

---

## 53. Auditing a Learned Simulator for Legged Robots: The Robotic World Model on ANYmal D

**arXiv ID:** 2611.23118 | [PDF](https://arxiv.org/pdf/2611.23118v1)

**作者:** Joyjeet Singh `[一作]` `[通讯]` (Independent Researcher), Joyjeet Singh (Independent Researcher)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文审计了一个已发布的四足机器人学习模拟器，从数据文件到在物理模型中行走的策略，未使用GPU。

**💡 创新点**

创新点在于验证了模型的动态模型和不确定性估计，发现了一步训练和滚动训练的性能差异，以及压缩模型的影响。

**🔧 技术方法**

使用了递归神经网络（RNN）和门控循环单元（GRU）来构建动态模型，并进行了多种模型的比较和评估。

**📊 数据集**

使用了ANYmal D的记录数据集，包括10个20秒的步态数据，模拟数据，和自建的MuJoCo物理模型。

**📈 对比分析**

通过与多个模型的比较，发现滚动训练在长时间步长上优于一步训练，而不确定性估计在错误排名上表现良好但在规模上存在问题。

**⚠️ 局限性**

限制在于只针对一个机器人和一个步态进行测试，数据集为模拟数据，且未在硬件上进行验证。

---

## 54. RiDEx: Reward-Informed Directional Exploration for Diffusion Reinforcement Learning

**arXiv ID:** 2611.22052 | [PDF](https://arxiv.org/pdf/2611.22052v1)

**作者:** Song Yan `[一作]` (University of Science and Technology of China), Zheng-Jun Zha `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文研究了基于奖励的扩散后训练，提出了一种新的探索方法RiDEx，旨在提高模型在高维潜在空间中的探索效率。

**💡 创新点**

创新点在于引入了奖励信息导向的方向性探索（RiDEx），通过集中在与奖励相关的子空间进行局部探索，从而提高了探索的有效性。

**🔧 技术方法**

使用了奖励信息导向的方向性探索技术（RiDEx），结合条件残差和可微分奖励梯度来选择探索子空间。

**📊 数据集**

在Stable Diffusion 3.5 Medium上进行了实验，评估了四个对齐和偏好任务的性能。

**📈 对比分析**

与现有方法（如DiffusionNFT）相比，RiDEx在HPSv2和PickScore任务上分别实现了15倍和29倍的目标达成速度提升，且在所有评估任务中获得了最高的峰值奖励。

**⚠️ 局限性**

限制在于探索空间的选择和奖励信号的获取可能依赖于任务的具体结构，可能在某些情况下无法完全捕捉到目标奖励的变化。

---

## 55. Statically Typed Purely Functional Definitional Interpreters for {\it Essentials of Programming Languages}

**arXiv ID:** 2611.22748 | [PDF](https://arxiv.org/pdf/2611.22748v1)

**作者:** Kwanghoon Choi `[一作]` `[通讯]` (Chonnam National University), Kwanghoon Choi (Chonnam National University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文重构了《编程语言基础》（EOPL）中的一系列定义解释器，使用Haskell这一静态类型的纯函数元语言进行教学，明确了语义组件。

**💡 创新点**

创新点在于通过静态类型和纯函数的方式，使得语言扩展与解释器结构之间的关系更加清晰，并且提供了一个从可执行解释器学习编程语言概念到语言设计和形式推理的具体桥梁。

**🔧 技术方法**

使用了Haskell作为元语言，利用代数数据类型、评估器签名和显式语义域来实现解释器。

**📊 数据集**

重构过程中保留了EOPL的对象语言、示例和测试，使用了EOPL中的15个解释器和6个类型检查器。

**📈 对比分析**

通过与原EOPL解释器的测试用例进行比较，验证了重构的解释器在行为上与原实现一致，且通过增量和测试驱动开发的方式，学生能够逐步构建解释器并验证其正确性。

**⚠️ 局限性**

限制在于重构的解释器虽然保留了EOPL的语义，但并未引入新的对象语言，且在教学中学生对Haskell的学习曲线较陡峭。

---

## 56. The Defense Rests: A Courtroom-Inspired Multi-Agent LLM Framework for Automated Safety Case Generation

**arXiv ID:** 2611.21618 | [PDF](https://arxiv.org/pdf/2611.21618v1)

**作者:** Gerhard Yu `[一作]` (York University), Alvine B. Belle `[通讯]` (York University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为eTrustWare的多代理框架，用于自动生成和审查安全案例，灵感来自法庭的比喻。

**💡 创新点**

创新点在于结合了半自动化的GSN合规安全案例生成、修订和评分，同时融入了人类专家的输入，以更好地促进机器学习驱动的安全关键系统的安全性。

**🔧 技术方法**

使用了大型语言模型（LLM）作为核心推理引擎，并采用了多代理架构，模仿法庭中的不同角色进行安全案例的生成和评估。

**📊 数据集**

使用了六个符合GSN标准的安全案例数据集，这些案例涉及机器学习驱动的自动驾驶系统的不同安全场景。

**📈 对比分析**

通过与现有方法的比较，eTrustWare在生成的安全案例的结构和语义正确性上表现出色，尤其是在使用Claude Fable-5模型时，生成的案例在结构相似性和语义相似性上均优于其他模型。

**⚠️ 局限性**

限制在于生成的安全案例尚未始终符合行业标准，且高质量的LLM操作成本较高，未来需要整合免费或本地的LLM以降低成本和提高隐私保护。

---

## 57. Levio: Smart Actuated Back-Exoskeleton via Retrofit of a Commercial Passive Device

**arXiv ID:** 2611.22767 | [PDF](https://arxiv.org/pdf/2611.22767v1)

**作者:** Ionnais Kyriazis `[一作]`, Herman van der Kooij `[通讯]` (University of Twente)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

本研究提出了一种名为Levios的智能主动背部外骨骼，通过对商业被动设备Laevo Flex进行改造，实现了基于用户姿势的辅助功能。

**💡 创新点**

创新点在于通过改造现有的被动外骨骼，快速实现主动辅助功能，解决了被动外骨骼无法根据任务需求调节辅助的局限性。

**🔧 技术方法**

使用了商业现成的驱动器（CubeMars AK10-9）、传感器（IMUs和负载传感器）以及相应的控制技术。

**📊 数据集**

实验中使用了Laevo Flex被动外骨骼，进行对比测试。

**📈 对比分析**

与被动Laevo系统相比，Levios在弯曲和提升阶段的辅助扭矩分别从25.2 Nm和30.6 Nm降低到17.9 Nm和提高到37.7 Nm，且参与者在使用Levios时报告了更高的可用性评分和更低的整体不适评分。

**⚠️ 局限性**

限制在于改造后总重量从4.3kg增加到约7kg，且在参与者测试中，最大辅助扭矩被限制在25 Nm，未能达到原设计目标的40 Nm。

---

## 58. RelicTouch: Exploring Fingertip Vibrotactile Feedback for Image Interpretation

**arXiv ID:** 2611.21642 | [PDF](https://arxiv.org/pdf/2611.21642v1)

**作者:** Xiaosong Wang `[一作]` (Zhejiang University), Yifan Feng `[通讯]` (University College London)

**通讯引用:** 53 | [OpenAlex ID](https://openalex.org/A5105422242)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出了RelicTouch，一个指尖佩戴的振动触觉反馈系统，旨在通过运动相关的触觉反馈帮助用户在触摸屏上探索图像内容。

**💡 创新点**

创新点在于将振动触觉反馈视为空间结构的解释编码，而非简单的物质属性模拟，强调了用户在探索过程中如何解读振动信号。

**🔧 技术方法**

使用了振动触觉系统技术，结合了触摸屏界面和佩戴式线性谐振执行器（LRA），实现了基于图像特征的动态振动反馈。

**📊 数据集**

研究中使用了自制的图像刺激，包括线性条纹纹理和颗粒状纹理，参与者在这些图像上进行触觉探索。

**📈 对比分析**

通过两项研究比较了不同的振动强度、条纹间距和手指配置对触觉-视觉一致性和主观清晰度的影响，结果表明单指探索的主观评分更高，而多指反馈在比较分离特征时可能具有价值。

**⚠️ 局限性**

限制在于振动反馈的可解读性依赖于信号设计、显示规模和空间分辨率，且在复杂形状的识别上存在一定的局限性。

---

## 59. Hidden-State Similarity Predicts Re-Elicitation After Inoculation Prompting

**arXiv ID:** 2611.23096 | [PDF](https://arxiv.org/pdf/2611.23096v1)

**作者:** Rose N. Kwon `[一作]`, Thomas Jiralerspong `[通讯]` (Université de Montréal)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在接种提示后，哪些因素使得评估提示能够重新引发被抑制的行为，发现文本相似性并不足以预测，而是激活状态的相似性更为重要。

**💡 创新点**

提出了激活空间分析作为识别评估提示的有效工具，以帮助检测和防止提示触发的失败。

**🔧 技术方法**

使用了激活状态相似性作为预测指标，比较了文本相似性和激活状态相似性对重新引发行为的预测能力。

**📊 数据集**

使用了五个狭窄的错误数据集，包括不良医疗建议、风险财务建议和极限运动建议等。

**📈 对比分析**

与文本相似性基线相比，激活状态相似性在预测重新引发行为方面表现更好，尤其在EM和大写行为的设置中，相关性显著高于文本基线。

**⚠️ 局限性**

研究结果是相关性的，未能证明测量的激活相似性直接介导重新引发行为，且实验仅限于Qwen2.5模型和两种行为，未来研究应扩展到更广泛的设置和更强的因果干预。

---

## 60. Attribution Through the Decoder: Prediction-Space Explanations for Structured Outputs

**arXiv ID:** 2611.23161 | [PDF](https://arxiv.org/pdf/2611.23161v1)

**作者:** Zaer Bin Zaqy `[一作]` (Duke University), Brinnae Bent `[通讯]` (Duke University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

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

## 61. LUTBench: Benchmarking Text-Driven Photographic Look Transfer

**arXiv ID:** 2611.23113 | [PDF](https://arxiv.org/pdf/2611.23113v1)

**作者:** Chuanzhi Xu `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了LUTBench，一个用于文本驱动摄影外观转移的基准，旨在通过共享协议比较不同方法的效果。

**💡 创新点**

创新点在于收集了91个三维查找表（3D LUT），并为每个LUT定义了一个摄影外观，生成了455,000对输入-目标图像对，提供了可重复的目标图像。

**🔧 技术方法**

使用了三维查找表（3D LUT）和测量基础的描述生成技术。

**📊 数据集**

使用了MIT-Adobe FiveK数据集，共包含5000张照片。

**📈 对比分析**

通过16种方法的比较，发现更强的文本对齐并不一定保证目标保真度，但大多数监督变体在已见和未见的外观上都提高了保真度。

**⚠️ 局限性**

限制在于LUTBench仅覆盖全局颜色变换，未考虑空间局部编辑，并且描述使用了受控的子句词汇，未来工作将朝着更自由的请求形式发展。

---

## 62. LoCaR: Localized Canvas Retrieval With Frozen Vision-Language Model

**arXiv ID:** 2611.22131 | [PDF](https://arxiv.org/pdf/2611.22131v1)

**作者:** Rahim Hossain `[一作]` (Binghamton University), Binbin Li `[通讯]` (eBay Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为LoCaR的训练无关框架，用于局部文本到图像检索，利用冻结的视觉-语言模型构建和利用独立的区域提案，以获得更具局部化的表示。

**💡 创新点**

创新点在于通过独立编码每个区域的视觉输入，改善了小物体和局部属性的表示，同时引入了LocFG基准来评估小型参考物的检索性能。

**🔧 技术方法**

使用了冻结的视觉-语言模型（如CLIP、SigLIP2和Qwen3-VL-Embedding-2B）进行区域画布的独立编码。

**📊 数据集**

使用了RefCOCOg数据集和一个基于Visual Genome的小型参考物检索基准LocFG进行实验。

**📈 对比分析**

与TextRegion方法进行了比较，结果显示LoCaR在多个视觉语言骨干网络上显著提高了检索性能，尤其是在区域检索方面，提升幅度为2.40-9.13个百分点。

**⚠️ 局限性**

局限性在于小区域提案的质量和数量可能限制了区域编码的效果，尤其是在处理小型参考物时，可能无法充分代表图像中的目标。

---

## 63. RoboTeam: Benchmarking Vision-Based Multi-Agent Control of Many Robot Arms in Real Time

**arXiv ID:** 2611.22138 | [PDF](https://arxiv.org/pdf/2611.22138v1)

**作者:** Changxuan Fan `[一作]` (Hong Kong University of Science and Technology), Xi Yang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RoboTeam，一个模拟基准，允许多模态代理从摄像头图像实时控制4到16个机器人手臂，完成36个长任务。

**💡 创新点**

RoboTeam是第一个在真实时间内测试多个代理控制多个手臂的基准，解决了现有基准在任务长度、实时性和真实感知方面的不足。

**🔧 技术方法**

使用了适应性系统、单代理系统、分层系统和去中心化系统等多种代理系统进行比较。

**📊 数据集**

使用了模拟环境MuJoCo，任务通过摄像头图像进行感知，未提供真实物体位置。

**📈 对比分析**

与其他三种固定系统相比，适应性系统在任务完成数量上相似，但在时间和输入令牌使用上有所不同。适应性系统在时间和令牌使用上介于其他系统之间，但没有任何模型在速度、成本和可靠性上同时表现出色。

**⚠️ 局限性**

结果来自模拟，未来需要转移到真实机器人上。36个任务仅为四种任务类型的变体，且由于条件需要数十亿个令牌，每个条件仅运行一次，差异的可靠性有限。

---

## 64. Identifying and Repairing Energy Smells with Agent Skills

**arXiv ID:** 2611.23203 | [PDF](https://arxiv.org/pdf/2611.23203v1)

**作者:** Mohammadjavad Mehditabar `[一作]` (Dalhousie University), Tushar Sharma `[通讯]` (Dalhousie University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种新的代理技能，能够在开发者的编码环境中检测和修复Python代码中的能量气味。

**💡 创新点**

这是首个基于能量气味分类法的代理技能，能够在代码编写阶段进行能量效率检查，提供了详细的检测和修复指导。

**🔧 技术方法**

使用了静态分析器和独立验证器，结合230个测量案例的修复指导，确保修复后的代码在行为上保持一致并节省能量。

**📊 数据集**

在三个开源Python代码库上进行了评估，涉及的代码库包括多个函数和提交记录。

**📈 对比分析**

与缺乏分类法的基线技能相比，完整技能在所有三个代码库中发现了更多的能量气味，并且验证了节能效果，68.6%的修复成功节省了能量，平均节省24.9%。

**⚠️ 局限性**

局限性在于能量气味的检测和修复依赖于上下文，可能无法覆盖所有实际情况，且测量结果可能受到噪声的影响。

---

## 65. Pelorus: Resource-Aware Lidar-Inertial Odometry

**arXiv ID:** 2611.22850 | [PDF](https://arxiv.org/pdf/2611.22850v1)

**作者:** Charlotte Sieh `[一作]` (Osnabrück University), Mario Porrmann `[通讯]` (Osnabrück University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种资源感知的激光雷达惯性里程计（Pelorus），旨在解决高分辨率激光雷达在资源受限硬件上处理效率的问题。

**💡 创新点**

创新点在于使用固定容量的多分辨率地图和基于相关性的驱逐策略，结合成熟度加权的地图更新，以保持内存的边界并抑制不一致的观测。

**🔧 技术方法**

采用了并行处理的技术，支持CUDA、Vulkan和Metal后端，能够在嵌入式GPU上以传感器速率运行。

**📊 数据集**

使用了来自三个公共数据集的五个序列进行评估，包括Newer College、HeLiPR和Oxford Spires。

**📈 对比分析**

与其他方法相比，Pelorus在十个指标中有七个达到了最低的相对位姿误差，并且在GPU卸载的情况下，CPU负载最低。

**⚠️ 局限性**

限制在于Pelorus不追求最低的累积漂移，可能在长时间的轨迹中导致较高的绝对位姿误差。

---

## 66. Inertial Video World Models

**arXiv ID:** 2611.22314 | [PDF](https://arxiv.org/pdf/2611.22314v1)

**作者:** Yuan Gao `[一作]` (Tsinghua University), Xiaomeng Huang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种惯性视频世界模型，通过将惯性作为自回归预测的结构先验，解决了现有视频世界模型在状态演变中的不足。

**💡 创新点**

创新点在于引入了惯性记忆，将被动的预测上下文转变为一个不断演变的视频状态，并通过密集的欧拉网格和持久的拉格朗日载体来实现。

**🔧 技术方法**

使用了惯性记忆技术，该技术结合了密集的欧拉网格和持久的拉格朗日载体，以实现状态的演变和内容的运输。

**📊 数据集**

在多个领域进行了广泛实验，包括机器人操作、3D游戏模拟和开放世界导航，使用的数据集包括RT-1、Counter-Strike和RECON。

**📈 对比分析**

与强大的视频世界模型基线相比，模型在自回归预测性能上表现出竞争力，尤其在RT-1、CSGO和RECON的评估中均取得了最佳结果。

**⚠️ 局限性**

限制在于当前模型主要针对确定性学习坐标的形式，未来需要扩展到多模态未来和明确的三维场景表示。

---

## 67. Agile Monoped Hopping on Complex and Discrete Terrain via Kinodynamic Planning with Distilled MPPI Steering

**arXiv ID:** 2611.21905 | [PDF](https://arxiv.org/pdf/2611.21905v1)

**作者:** Yue Qin `[一作]` (University of Michigan), Yanran Ding `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了FlowRRT，一种用于在离散地形上进行灵活单足跳跃的动力学规划器。

**💡 创新点**

创新点在于将引导视为集合值问题，预测可行动作的分布，而不是单一解，从而提高了规划的多样性和速度。

**🔧 技术方法**

使用了多启动模型预测路径积分（MPPI）专家和条件流匹配（CFM）生成采样器。

**📊 数据集**

使用了随机生成的多种地形数据集，包括斜坡、间隙和石头等不同类型的地形。

**📈 对比分析**

与基于采样的基线方法（如RL-RRT和SST）相比，FlowRRT在不同地形上显示出更高的成功率和显著更短的规划时间。在斜坡地形上，FlowRRT在1秒内达到80%的成功率，而RL-RRT需要24秒，SST需要77秒。

**⚠️ 局限性**

限制在于评估仅限于模拟，未来需要在物理单足机器人上进行硬件实验，并扩展到双足和类人跳跃。

---

## 68. Beyond Mimicry: Constrained Exploration with GRPO for Joint Multi-Talker ASR and Diarization under Unknown Speaker Counts

**arXiv ID:** 2611.22627 | [PDF](https://arxiv.org/pdf/2611.22627v1)

**作者:** Yunrui Cai `[一作]` (Chinese University of Hong Kong), Helen Meng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种统一的两阶段生成框架，用于解决多说话者自动语音识别（ASR）中的说话者归属和时间戳问题，尤其是在说话者数量未知的情况下。

**💡 创新点**

创新点在于结合了链式思维（CoT）推理和基于组相对策略优化（GRPO）的约束探索，优化了多说话者ASR和说话者分离的联合任务。

**🔧 技术方法**

使用了链式思维推理（CoT）和组相对策略优化（GRPO）技术，结合了多维约束感知奖励（MCAR）来优化模型性能。

**📊 数据集**

在Libri3Mix和一个动态混合（2+3）数据集上进行了评估，这些数据集在测试时说话者数量未知。

**📈 对比分析**

与传统方法相比，模型在cpWER和WDER上分别达到了14.52%/9.24%和1.95%/1.12%的性能，GRPO相较于SFT+CoT在cpWER上实现了35%/54%的相对减少。

**⚠️ 局限性**

限制在于模型在动态场景下的表现仍然受到未知说话者数量的影响，且在某些情况下可能仍会出现文本幻觉和时间戳不一致的问题。

---

## 69. Eyes in the Update? Learning Where Rewards Land in Multimodal Reasoning

**arXiv ID:** 2611.22438 | [PDF](https://arxiv.org/pdf/2611.22438v1)

**作者:** Lixuan He `[一作]` (University of Hong Kong), Yi Ma `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于表示的策略优化方法，通过潜在奖励着陆场重新分配完成级别的优势，以提高图像和视频推理的准确性。

**💡 创新点**

创新点在于通过内部跨模态几何和局部预测不确定性构建均值为一的奖励着陆场，从而在不需要过程监督或外部模块的情况下，重新分配奖励。

**🔧 技术方法**

使用了基于表示的策略优化技术，结合内部跨模态几何和预测不确定性。

**📊 数据集**

在Qwen2.5-VL和InternVL3.5的图像和视频基准测试上进行了评估。

**📈 对比分析**

与GRPO、DAPO、VPPO等基线方法进行比较，结果显示在需要视觉基础推理的基准测试中表现更好，尤其是在视频评估中，当相关证据稀疏时，表现尤为突出。

**⚠️ 局限性**

方法依赖于基础模型的视觉-文本表示和提供回合级优势的验证器，无法纠正错误的结果奖励或缺失的视觉证据。

---

## 70. SR-VLA: Stall-Reduced VLA Execution via Predicted-Input Lookahead and Selective KV Cache Repair

**arXiv ID:** 2611.22356 | [PDF](https://arxiv.org/pdf/2611.22356v1)

**作者:** Tao Zhang `[一作]` (Peking University), Yuan Wang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为SR-VLA的框架，通过预测输入的前瞻性和选择性KV缓存修复来减少视觉-语言-动作（VLA）执行中的停滞现象。

**💡 创新点**

创新点在于结合了预测输入的前瞻性机制和基于真实输入的选择性KV缓存修复，能够在执行当前动作块时预计算下一边界的视觉观察和机器人状态。

**🔧 技术方法**

使用了轻量级的动作条件预测器来生成下一边界的视觉观察和机器人状态，并利用冻结的VLA模型进行预计算。

**📊 数据集**

在LIBERO数据集上进行实验，该数据集是一个广泛使用的VLA基准，包含多个任务套件。

**📈 对比分析**

与同步执行的基线相比，SR-VLA在LIBERO上实现了2.67倍的查询加速，同时保持了竞争性的任务成功率。在目标运动扰动下，成功率与基线相当，查询加速为1.52倍至1.93倍。

**⚠️ 局限性**

限制在于预测输入的准确性可能影响最终的执行效果，尤其是在输入变化较大的情况下，可能需要更多的修复操作。

---

## 71. Gateway-Free Kafka Streaming from FPGA-Based Detectors: A Bare-Metal Pipelined Producer on Zynq-7000

**arXiv ID:** 2611.22674 | [PDF](https://arxiv.org/pdf/2611.22674v1)

**作者:** Fernanda Zapata Bascuñán `[一作]` (Universidad Nacional de la Defensa), Matías Hampel `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在Zynq-7000 SoC上实现了一个裸机Apache Kafka生产者，用于事件驱动的数据采集，直接将FPGA生成的事件发布到Kafka代理。

**💡 创新点**

创新点在于将Kafka生产者直接集成在接收FPGA生成事件的同一SoC上，避免了中间网关计算机的使用，并实现了裸机环境下的Kafka Wire协议。

**🔧 技术方法**

使用了lwIP原始TCP/IP API、AXI4-Stream和AXI DMA技术。

**📊 数据集**

使用的数据集包含80个独立的运行，每个运行5000个事件，总共400,000个事件记录。

**📈 对比分析**

通过比较不同的管道深度（D=1, 2, 4, 8）来评估性能，D=4时的平均操作吞吐量约为16.2 kevents/s，D=8时为16.8–17.5 kevents/s，D=1时为5.3 kevents/s，且D=4的发布到确认延迟最低。

**⚠️ 局限性**

限制在于PL和PS时间域分离，当前的仪器无法将本地TCP接受后的时间间隔分解为以太网传输、网络、代理处理和响应交付组件。

---

## 72. Formal Analysis of a Fault-Tolerant Extension of the ROSACE Flight Controller

**arXiv ID:** 2611.22394 | [PDF](https://arxiv.org/pdf/2611.22394v1)

**作者:** Marco Bozzano `[一作]` (Fondazione Bruno Kessler), Stefano Tonetta `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文分析了ROSACE，一个开源的飞行控制系统控制工程基准，重点关注其在运行时对传感器噪声和传感器故障等故障的弹性。我们通过基于三模冗余（TMR）的故障检测、识别和恢复（FDIR）组件扩展了该基准。

**💡 创新点**

创新点在于将故障模型和FDIR机制引入ROSACE，并使用形式化方法对系统的安全性和性能要求进行验证，展示了与传统方法相比的优势。

**🔧 技术方法**

使用了COMPASTA工具，该工具基于ESA的TASTE工具，支持模型驱动设计、故障注入、形式验证和安全评估，并实现了基于仿真的验证和验证工作流（VIVAS）。

**📊 数据集**

使用了ROSACE基准案例，该案例模拟了民用飞机的纵向飞行控制，涉及多个传感器和控制器的交互，具有复杂的动态特性。

**📈 对比分析**

通过与原始模型及其不同变体（包括故障和不包括FDIR的模型）进行比较，分析了故障和FDIR对系统安全性和性能的影响。结果表明，使用COMPASTA和VIVAS工作流在分析的系统性和结果质量上具有明显优势。

**⚠️ 局限性**

限制在于该研究主要集中于特定类型的故障和冗余机制，未来的工作将考虑其他类型的故障或冗余，并将该方法应用于其他案例研究。

---

## 73. Diagnosing and Correcting Skin-Tone Color Casts in Ultra-Lightweight Photo Retouching Networks

**arXiv ID:** 2611.22106 | [PDF](https://arxiv.org/pdf/2611.22106v1)

**作者:** Choulhyouc Lee `[一作]` `[通讯]` (SNOW Corporation), Choulhyouc Lee (SNOW Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了超轻量级照片修饰网络中人类皮肤色调偏差的诊断与修正，发现常用的合成自监督训练方法会导致皮肤呈现绿色偏差。

**💡 创新点**

创新点在于提出了一种逐像素、皮肤掩膜的非对称绿色过量损失函数，有效抑制了皮肤的绿色偏差，并且在不增加参数的情况下保持了模型的性能。

**🔧 技术方法**

使用了逐像素的皮肤掩膜损失函数，结合了合成自监督训练方法，评估了不同架构的表现。

**📊 数据集**

使用了MIT-Adobe FiveK、DIV2K、LSDIR和FFHQ等多个数据集进行训练和评估。

**📈 对比分析**

与传统的图像级目标比较，提出的方法在抑制绿色偏差方面表现优异，绿色偏差率从52/70降低到1/70，且在保持PSNR/SSIM不变的情况下，皮肤色差的表现也得到了改善。

**⚠️ 局限性**

限制在于合成评估降解的范围较窄，模型在处理真实场景中的色彩偏差时可能表现不佳，且手动曝光控制需要用户的主动干预。

---

## 74. GaussianMamba: Geometric and Relational State Space Scanning of 3D Gaussian Primitives

**arXiv ID:** 2611.22742 | [PDF](https://arxiv.org/pdf/2611.22742v1)

**作者:** A. Enes Doruk `[一作]`, Ozgur Erkent `[通讯]` (Hacettepe University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为GaussianMamba的双分支框架，用于3D语义占用预测，结合了高斯特定的几何和特征语义关系。

**💡 创新点**

创新点在于引入了接触引导空间排序(CGSS)、方向关系感知语义扫描(DRSS)和高斯支持差异融合(GSDF)，以更好地捕捉几何和语义关系。

**🔧 技术方法**

使用了高斯原语和Mamba模型，结合了几何支持和特征语义关系的扫描策略。

**📊 数据集**

使用了nuScenes-SurroundOcc和SSCBench-KITTI-360数据集进行实验。

**📈 对比分析**

与GaussianFormer和GaussianFormer-2相比，GaussianMamba在IoU和mIoU上均有2.2-2.6的提升，表现出一致的性能改进。

**⚠️ 局限性**

主要限制在于推理延迟增加，在144K高斯原语时，推理延迟增加约31%。

---

## 75. Revisiting the Survival of Agent-Attributed Code: How Measurement Choices Decide the Verdict

**arXiv ID:** 2611.21784 | [PDF](https://arxiv.org/pdf/2611.21784v1)

**作者:** Musfiqur Rahman `[一作]` (Concordia University), Emad Shihab `[通讯]` (Concordia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究重新审视了AI编码代理生成的代码在合并后存活的情况，探讨了测量选择如何影响结果。研究使用了AIDev数据集中的854,139行代码，分析了2,802个合并的拉取请求（PRs）。

**💡 创新点**

创新点在于系统地评估了九个测量选择对后合并研究结论的影响，并建立了一个包含88个生存分析的规范曲线，展示了不同选择如何改变结果。

**🔧 技术方法**

使用了Kaplan-Meier曲线和Cox比例风险回归等生存分析技术，结合了多种测量选择的组合进行比较。

**📊 数据集**

使用了AIDev数据集，该数据集包含来自五个AI编码代理的932,791个PRs，最终分析了215个开源仓库中的2,802个合并PRs。

**📈 对比分析**

通过与五个已发表设计的重跑比较，发现不同的测量选择会显著改变结果。例如，使用不同的跟踪器和单位分析方法会导致不同的生存率和修改率，且在同一仓库内的比较结果与跨仓库的比较结果存在显著差异。

**⚠️ 局限性**

限制在于，尽管研究系统地评估了测量选择的影响，但仍然可能存在未考虑的变量或选择，这可能会影响结果的普遍性和适用性。

---

## 76. TextSphere: Efficient Text-to-Image Generation via Shared Hyperspherical Alignment and Uniformity

**arXiv ID:** 2611.23155 | [PDF](https://arxiv.org/pdf/2611.23155v1)

**作者:** Yixia Zhang `[一作]` (University of Surrey), Wenwu Wang `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种高效的文本到图像生成框架，通过扩展Sphere Encoder，结合语言条件和跨模态对齐，能够在较少的采样步骤中生成高质量图像。

**💡 创新点**

创新点在于引入了Spherical Semantic Alignment and Uniformity (SSAU)目标，结合对称对比学习和球面均匀性正则化，增强了文本与图像之间的语义对齐。

**🔧 技术方法**

使用了Sphere Encoder框架，结合预训练的CLIP文本嵌入和自适应层归一化（AdaLN），并引入了SSAU目标来促进跨模态对齐。

**📊 数据集**

在CUB-200-2011、MS-COCO-2014和Flickr30k数据集上进行了实验。

**📈 对比分析**

与从头训练的扩散模型和GAN模型进行比较，结果显示该框架在生成FID（gFID）上优于对比的模型，且在仅使用4到6个采样步骤的情况下，生成质量仍然具有竞争力。

**⚠️ 局限性**

局限性包括生成性能对CFG尺度和细化步骤的敏感性，以及未在更大规模数据集上进行评估，限制了方法的可扩展性和泛化能力。

---

## 77. Noise Characterization and Robust Signal Detection in Yeast Pheromone Molecular Communication

**arXiv ID:** 2611.22416 | [PDF](https://arxiv.org/pdf/2611.22416v1)

**作者:** Nikolaos Ntetsikas `[一作]` (Frederick University), Marios Lestas `[通讯]` (Frederick University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究探讨了在酵母信息传递中的噪声特征和鲁棒信号检测，利用新开发的实验平台，基于酵母细胞间的通信，提出了适合的信号检测机制。

**💡 创新点**

创新点在于首次将酵母作为生物传感器的基础，结合真实的噪声机制，提出了基于最大似然检测的信号检测策略，并验证了理论结果与实验数据的一致性。

**🔧 技术方法**

使用了马尔可夫出生-死亡过程模型和混合高斯分布模型来表征噪声，并采用最大似然检测（MLD）和最大似然序列检测（MLSD）方法进行信号检测。

**📊 数据集**

使用了基于酵母的实验平台，具体包括MATα细胞作为发送者和MATa细胞作为接收者，利用α因子作为信息载体进行实验。

**📈 对比分析**

与传统的符号逐个检测方法相比，序列检测在误差率上表现出显著的改进，误差率降低了两个数量级，尽管计算复杂度有所增加。

**⚠️ 局限性**

限制在于生物系统固有的噪声特性导致的误差概率，优化信号阈值的下限约为10%，而序列检测的上限误差率约为0.4%。

---

## 78. Solver to System: LLM Reasoning Across Representations and Execution in Scientific Simulation

**arXiv ID:** 2611.21964 | [PDF](https://arxiv.org/pdf/2611.21964v1)

**作者:** Eliza Berman `[一作]` (New York University), Rajesh Ranganath `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了Solver2System数据集，评估大型语言模型（LLMs）在恢复和验证偏微分方程（PDE）求解器背后的物理系统的能力。

**💡 创新点**

创新点在于引入了一个包含多种表示形式的PDE求解器数据集，并通过交叉表示一致性和静态及代理有效性判断来评估LLMs的表现。

**🔧 技术方法**

使用了大型语言模型（LLMs）进行科学推理和代码执行。

**📊 数据集**

使用了包含32个PDE求解器实现的共享数据集，涵盖热传导、波动、Burgers和Navier-Stokes方程。

**📈 对比分析**

通过与八个开放权重模型的比较，发现LLMs在处理冲突表示时表现出不对称的和偏向某些表示的倾向，执行可以改善物理有效性判断，但较弱的代理可能在没有识别因果缺陷的情况下做出正确判断。

**⚠️ 局限性**

限制在于LLMs可能在没有因果诊断的情况下做出正确的物理有效性判断，且在处理不同表示形式时存在不对称性。

---

## 79. Correcting Flows with Marginal Matching

**arXiv ID:** 2611.23134 | [PDF](https://arxiv.org/pdf/2611.23134v1)

**作者:** Meital Bojan `[一作]` (Technion), Aviv Tamar `[通讯]` (Technion)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的基于训练的预测-校正框架，用于流匹配模型，通过利用已知的边际分布来提高生成样本的质量。

**💡 创新点**

创新点在于引入了时间反转的概念，通过校正样本使其更好地匹配反向ODE的边际分布，从而减少生成分布与真实数据分布之间的距离。

**🔧 技术方法**

使用了基于ODE的生成模型和预测-校正方法，结合了流匹配模型和边际匹配技术。

**📊 数据集**

使用了CIFAR-10和ImageNet-64数据集进行实验，评估生成样本的质量。

**📈 对比分析**

与传统方法相比，提出的方法在生成质量上显著提高，FID分数更低，且计算时间开销最小。

**⚠️ 局限性**

限制在于假设向量场是可逆的，这限制了方法在非可逆模型上的推广。

---

## 80. PAIR: Bringing Household Robots into the Smart Home

**arXiv ID:** 2611.23070 | [PDF](https://arxiv.org/pdf/2611.23070v1)

**作者:** Junseong Pyo `[一作]` (Seoul National University), Yohan Jo `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PAIR，一个智能家居仿真环境，允许家庭机器人通过物理控制和设备API与智能家电进行交互。

**💡 创新点**

创新点在于将物理控制与API控制结合在一个共享设备状态下，提供了一个基于Matter协议的智能家居仿真环境。

**🔧 技术方法**

使用了Matter协议构建的智能设备模型，并结合物理仿真技术。

**📊 数据集**

构建了一个包含84个智能设备资产的基准数据集，涵盖40个任务和340个人类演示。

**📈 对比分析**

与现有的视觉-语言-动作（VLA）策略进行比较，发现没有一个策略能够完成评估任务，API访问虽然提高了设备操作的成功率，但在导航、抓取和操作门等子任务上仍然存在90%以上的失败率。

**⚠️ 局限性**

限制在于调查任务仅反映智能家居的委托，而非一般家庭活动，且仅提供340个演示，可能导致VLA策略的成功率较低。

---

## 81. Manufacturing Differential Sparsity: Planner-Hardware Co-Design for Efficient World Model Planning

**arXiv ID:** 2611.22483 | [PDF](https://arxiv.org/pdf/2611.22483v1)

**作者:** Wenbo Zhu `[一作]` (University of Southern California), Christopher Torng `[通讯]` (University of Southern California)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种名为Candelta的规划器-硬件协同设计，用于基于采样的世界模型规划，利用预测器将相近的动作映射到相近的潜在状态，从而实现差异化的比特级稀疏性。

**💡 创新点**

创新点在于通过规划器制造差异化的比特级稀疏性，并通过参考森林采样来增加稀疏性，同时保持竞争性的规划质量，这是首次在JEPA风格的世界模型规划中实现的算法-硬件协同设计。

**🔧 技术方法**

使用了基于比特级稀疏性的硬件加速器，并设计了与规划器紧密耦合的Candelta系统。

**📊 数据集**

在三个JEPA世界模型上进行了评估，包括LeWM、DINO-WM和JEPA-WM。

**📈 对比分析**

与密集的INT8加速器相比，Candelta在相同面积下速度提高了-，能耗降低了-，并且比之前的比特稀疏设计Pragmatic快-，能耗少-。

**⚠️ 局限性**

限制在于Candelta的设计速度上限，可能无法超越密集芯片的性能，且在某些情况下可能会影响探索能力。

---

## 82. Data-Driven Variable-Length Compression of 3D Representations

**arXiv ID:** 2611.21802 | [PDF](https://arxiv.org/pdf/2611.21802v1)

**作者:** Yvette Y. Lin `[一作]` (Georgia Institute of Technology), Sara Fridovich-Keil `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 9009 | [OpenAlex ID](https://openalex.org/A5049727497)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种学习的可变长度压缩方法，利用训练数据集的结构，同时根据单个形状的复杂性调整压缩表示的大小。

**💡 创新点**

创新点在于能够在推理时根据形状复杂性选择压缩比，而无需重新训练预训练的形状自编码器。

**🔧 技术方法**

使用了回归解码器和超自编码器，结合了快速前馈推理。

**📊 数据集**

使用了ShapeNet数据集进行评估。

**📈 对比分析**

与每个形状和数据集级别的压缩基线进行比较，结果显示该方法在率-失真权衡上优于基线，且随着形状复杂性增加，达到目标质量所需的标记数量也增加。

**⚠️ 局限性**

限制在于该方法依赖于预训练的形状自编码器，可能无法适应未见过的新形状表示。

---

## 83. Variance Restoration over Covariance Rank: Memory-Budgeted Analytic Continual Learning

**arXiv ID:** 2611.23040 | [PDF](https://arxiv.org/pdf/2611.23040v1)

**作者:** Elena Voss `[一作]` (University of Wisconsin Madison), Sofia Reyes `[通讯]` (University of Wisconsin Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的分析持续学习方法，称为坐标恢复分析持续学习（Coordinate-Restored Analytic Continual Learning），通过保持特征自相关矩阵和类交叉相关矩阵的精确值，同时将非对角交互压缩为一个流式秩-q因子，从而优化内存使用。

**💡 创新点**

创新点在于提出了一种新的内存预算分配策略，能够在固定内存预算下有效平衡特征扩展宽度和协方差交互的保真度，同时通过Woodbury公式实现快速的正常方程求解。

**🔧 技术方法**

使用了流式秩-q因子和Woodbury公式来解决正常方程，优化了内存使用和计算效率。

**📊 数据集**

在CIFAR-100、CUB-200、VTAB-50、ImageNet-R和ImageNet-A等五个类增量学习基准上进行了评估。

**📈 对比分析**

与现有的状态匹配的窄精确方法相比，在相同的内存预算下，提出的方法在17个18种配置中表现更好，准确率提高了最多2.19个百分点，同时减少了90.7%的持久状态。

**⚠️ 局限性**

限制在于方法依赖于固定的内存预算，可能在极端情况下无法适应更大的数据集或更复杂的模型需求。

---

## 84. Collecting Infrastructure-Level Measures for Energy Analysis in Modern Computing Infrastructures

**arXiv ID:** 2611.22160 | [PDF](https://arxiv.org/pdf/2611.22160v1)

**作者:** Isac Pasianotto `[一作]` (University of Trieste), Niccolo Tosato `[通讯]` (University of Trieste)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个可扩展的数据收集管道，用于收集现代计算基础设施的能源分析数据，涵盖多个子系统的细粒度测量。

**💡 创新点**

提出了一种可重复的测量管道和一个高质量的公共数据集，能够提供跨基础设施的能源消耗分析，填补了现有数据集的不足。

**🔧 技术方法**

使用了开源组件构建的测量管道，采用固定频率采样和跨节点时间戳对齐，确保数据的完整性和一致性。

**📊 数据集**

使用了来自意大利Trieste的Orfeo数据中心的28天的生产环境数据，涵盖了HPC、Kubernetes服务和分布式存储等异构工作负载。

**📈 对比分析**

通过两个案例研究（分布式I/O和大语言模型推理）展示了集群级测量的优势，揭示了局部仪器无法捕捉的系统级动态，性能分析显示GPU计数器低估了实际能耗。

**⚠️ 局限性**

限制在于现有的监测系统通常无法提供足够的细粒度数据，且数据收集的复杂性可能导致实施上的挑战。

---

## 85. Equilibrium Statistics and Stability in Random Two-Population and Three-Population Evolutionary Games

**arXiv ID:** 2611.23223 | [PDF](https://arxiv.org/pdf/2611.23223v1)

**作者:** Zhao Song `[一作]` (Teesside University), The Anh Han `[通讯]` (Teesside University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

研究了多种群体矩阵进化博弈，分析了在不确定收益下不同进化结果的发生频率及其概率如何依赖于建模假设。

**💡 创新点**

提出了在两种和三种群体博弈中，如何通过不确定性表示、交互结构和群体数量来区分共存与吸引的概率基准。

**🔧 技术方法**

使用了高斯不确定性模型，结合了蒙特卡洛模拟和解析稳定性结果。

**📊 数据集**

使用了随机生成的收益矩阵，进行独立的高斯采样。

**📈 对比分析**

通过与现有的两种和三种群体博弈的比较，展示了不同交互结构对平衡存在性和稳定性的影响，结果表明三种群体博弈中多稳定性更为常见。

**⚠️ 局限性**

限制在于未能探讨在相关或经验校准收益下结果的变化，以及三种群体博弈中内部平衡的解析稳定性结果。

---

## 86. Evolving Neural Solvers for Multi-objective Combinatorial Optimization

**arXiv ID:** 2611.22060 | [PDF](https://arxiv.org/pdf/2611.22060v1)

**作者:** Ya-Hui Jia `[一作]` (South China University of Technology), Yi Mei `[通讯]` (Victoria University of Wellington)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种名为Evolving Neural Solver (ENS)的方法，用于解决多目标组合优化问题（MOCO），通过进化计算扩展少量特定偏好的神经模型，生成多样化的求解器。

**💡 创新点**

创新点在于结合了强化学习和进化计算，首先在估计的拐点偏好处训练一小组锚点模型，然后通过进化策略扩展这些模型，以提高解的质量和Pareto前沿的覆盖率。

**🔧 技术方法**

使用了强化学习和进化计算技术，特别是通过进化策略来生成新的求解器，并进行联合非支配选择以保留不同权衡的解。

**📊 数据集**

在经典的MOCO问题上进行了实验，包括多目标旅行商问题（MOTSP）、多目标车辆路径问题（MOCVRP）和多目标背包问题（MOKP）。

**📈 对比分析**

与现有的神经MOCO方法相比，ENS在大多数评估设置中在超体积（hypervolume）指标上表现优越，显示出更好的解质量和多样性。

**⚠️ 局限性**

限制在于该方法依赖于初始锚点的选择和训练过程，可能在某些情况下对偏好空间的覆盖不够全面，尤其是在处理更复杂的实例时。

---

## 87. Stop Early, Stay Simple: Evidence Acquisition for Power System Fault and Attack Diagnosis

**arXiv ID:** 2611.21695 | [PDF](https://arxiv.org/pdf/2611.21695v1)

**作者:** S M Zia Ur Rashid `[一作]` (University of Tulsa), Suman Rath `[通讯]` (University of Tulsa)

**通讯引用:** 446 | [OpenAlex ID](https://openalex.org/A5075963806)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为ActiGrid的基准框架，用于主动证据获取，旨在优化电力系统的故障和攻击诊断过程。

**💡 创新点**

创新点在于将深度强化学习与缺失模态的分类器结合，提出了一种学习策略来选择何时获取更多证据或停止诊断。

**🔧 技术方法**

使用了双深度Q网络（Double DQN）和分类器来处理缺失模态，并与固定子集、熵停止、贪婪信息价值规则和树集成方法进行了比较。

**📊 数据集**

使用了TU Delft数字变电站和MSU/ORNL电力系统攻击的公开数据集。

**📈 对比分析**

与固定子集和简单控制方法相比，DQN在TU Delft上表现相似，但在MSU/ORNL上被简单控制方法超越，整体性能不如紧凑的子集分类器。

**⚠️ 局限性**

限制在于未考虑不同模态的实际成本（如带宽、能耗、延迟等），且仅使用了两个数据集，可能影响结果的普遍性。

---

## 88. SlideJev: Order-Stable and Tissue-Grounded Jev Decisions for Whole-Slide Pathology

**arXiv ID:** 2611.22879 | [PDF](https://arxiv.org/pdf/2611.22879v1)

**作者:** Lei Wu `[一作]` (Xi'an Jiaotong University), Qidong Liu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为SlideJev的全幻灯片病理视觉问答（VQA）框架，旨在通过自然语言接口直接从组织切片中回答诊断和表型问题。

**💡 创新点**

创新点在于通过独立的候选答案验证和视觉证据学习相结合，减少了选项顺序敏感性，并增强了视觉证据的利用。

**🔧 技术方法**

采用了Jev决策模型，通过并行候选输入和门控视觉注入技术来实现候选答案的评分和视觉更新。

**📊 数据集**

使用了SlideBench-VQA、BCNB和BRACS-3等多个全幻灯片病理VQA基准数据集进行评估。

**📈 对比分析**

与其他方法相比，SlideJev在所有三个基准上实现了最高的四种选项顺序的平均整体准确率，并在选项顺序变化时保持了答案一致性。

**⚠️ 局限性**

限制在于模型的训练依赖于固定的预训练骨干网络，可能限制了其在不同数据集上的泛化能力。

---

## 89. CADMAGEDDON: High-Fidelity CAD Reconstruction with Local Inference

**arXiv ID:** 2611.22563 | [PDF](https://arxiv.org/pdf/2611.22563v1)

**作者:** Oleg Pavlov `[一作]` (Lomonosov Moscow State University), Dmitrii Zhemchuzhnikov `[通讯]` (Lomonosov Moscow State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CADMAGEDDON，一个开源的代理，能够从网格重建可编辑的CAD模型，并且可以在单个GPU系统和当前一代笔记本电脑上本地运行。

**💡 创新点**

CADMAGEDDON通过将复杂任务分解为一系列简单的视觉决策来实现重建，创新性地将几何工具与视觉推理结合，显著提高了重建质量。

**🔧 技术方法**

使用了中型的Qwen3.8-27B模型，结合几何算法和视觉-语言模型来分析目标网格并做出决策。

**📊 数据集**

在多个基准数据集上进行评估，包括DeepCAD、Fusion 360、MCB、BenchCAD和CADENA-Bench。

**📈 对比分析**

与CADENA、CADFather、StepCAD和CADFit等方法进行比较，CADMAGEDDON在所有五个基准上都达到了最高的IoU和GMS，且最低的Chamfer距离，尤其在机械部件的表面结构上表现突出。

**⚠️ 局限性**

CADMAGEDDON假设输入为单一部分的干净网格，无法处理扫描噪声或缺失区域，且当前系统无法恢复设计意图。

---

## 90. Usage Patterns of Programming Languages and Tools: A Systematic Literature Review

**arXiv ID:** 2611.21627 | [PDF](https://arxiv.org/pdf/2611.21627v1)

**作者:** Sherlock A. Licorish `[一作]` (University of Otago), Natalie Grattan `[通讯]` (University of Otago)

**通讯引用:** 59 | [OpenAlex ID](https://openalex.org/A5029870224)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过系统文献综述，调查了2015年至2024年间37项关于行业中使用的编程语言、集成开发环境（IDE）和持续集成（CI）工具的研究，发现Java是最常用的编程语言，而Travis CI和GitHub Actions是最常用的CI工具。

**💡 创新点**

创新点在于系统性地识别当前全球软件行业中采用的工具和编程语言，并指出了现有研究的不足之处，尤其是在IDE使用方面的研究较少。

**🔧 技术方法**

使用了系统文献综述（SLR）的方法，结合了对开源软件（OSS）库的挖掘和对Stack Overflow的分析。

**📊 数据集**

研究涉及的文献数据集包括2015年至2024年间的37篇相关研究论文。

**📈 对比分析**

与其他方法相比，研究发现使用开源库挖掘是最常见的方法，其次是使用Stack Overflow进行数据分析。性能方面，研究表明Java、JavaScript和Python是最常用的编程语言，而Travis CI是最常用的CI工具。

**⚠️ 局限性**

限制在于样本人口统计有限和一般化能力不足，尤其是对IDE使用的研究较少，且研究方法可能存在偏差。

---

## 91. PCRG: Training-Free Pruning and Caching of Reference Tokens for Video Generation

**arXiv ID:** 2611.22717 | [PDF](https://arxiv.org/pdf/2611.22717v1)

**作者:** Siyuan Wang `[一作]` (Anorak Games), Quan Chen `[通讯]` (Anorak Games)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在保持性能的同时减少计算复杂度。

**🔧 技术方法**

使用了深度学习技术，结合了卷积神经网络和强化学习。

**📊 数据集**

使用了公开的图像识别数据集和自定义生成的数据集进行实验。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和处理速度上均有显著提升。

**⚠️ 局限性**

限制在于算法对特定类型数据的适应性较差，且在极端情况下性能可能下降。

---

## 92. Contract-Explicit Evaluation: From Auditing Learned Readers to Scoped Certificates

**arXiv ID:** 2611.22665 | [PDF](https://arxiv.org/pdf/2611.22665v1)

**作者:** Hao Lu `[一作]` (Army Engineering University of PLA), Ruiyang Li `[通讯]` (Army Engineering University of PLA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种合同明确的评估方法，强调智能系统的性能读数是模型和评估合同的共同属性。通过对一个生态系统的审计和方法的推广，提供了一个五步法（声明、测量、范围、认证、对冲）来处理评估。

**💡 创新点**

创新点在于将评估读数视为模型和评估合同的联合属性，并提出了一种学习的评估阅读器，通过对合同集的边际化来优化评估过程。

**🔧 技术方法**

使用了一种学习的评估阅读器，结合了文档特征和合同阅读向量，训练了一个加权网络来混合不同合同的读数，并通过风险控制的决策层进行优化。

**📊 数据集**

使用了498个文档的双金标准DocRE开发集，结合了来自7个生成器家族的44次运行的预测结果。

**📈 对比分析**

与传统方法相比，提出的方法在处理合同敏感性方面表现出更高的稳定性和更低的错误对比率，尤其是在处理争议对时，显示出更好的性能。

**⚠️ 局限性**

限制在于当前的注册库包含的翻转检查数量较少，导致统计功效不足，无法在某些对比中提供强有力的认证。此外，合同选择的非交换性可能影响结果的可靠性。

---

## 93. CorrectCast: Selective Multimodal Context for Scale-Aware Residual Correction in Precipitation Nowcasting

**arXiv ID:** 2611.22178 | [PDF](https://arxiv.org/pdf/2611.22178v1)

**作者:** Henan Wang `[一作]` (Wuhan University of Technology), Pengfei Duan `[通讯]` (Wuhan University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为CorrectCast的框架，用于改进降水短期预报，专注于修正强度、位置和空间结构的错误，而无需修改基础模型。

**💡 创新点**

CorrectCast通过解耦的残差修正框架，提供了一个统一的修正接口，能够在不同的降水预报模型之间进行有效的修正。

**🔧 技术方法**

使用了选择性多模态上下文构建和尺度感知调制技术，结合历史的定量降水估计（QPE）和可降水水汽（PWV）观测。

**📊 数据集**

在密西西比河流域（MRB）和美国大陆（CONUS）数据集上进行了评估，这些数据集结合了多雷达多传感器（MRMS）QPE和GNSS衍生的PWV。

**📈 对比分析**

与六种基础降水预报模型进行比较，CorrectCast在CSI（关键成功指数）上平均提高了0.0233和0.0290，且在空间聚合和图像质量指标上也表现出一致的改进。

**⚠️ 局限性**

CorrectCast的局限性在于其对基础模型的依赖性，尽管可以跨模型和跨数据集进行转移，但效果可能因基础模型和目标域条件的不同而有所变化。

---

## 94. Fine-Tuning Gemma 4 Edge Models for MATLAB Signal-Processing Code with SFT and QLoRA on a Laptop iGPU

**arXiv ID:** 2611.23205 | [PDF](https://arxiv.org/pdf/2611.23205v1)

**作者:** Mehmet Altan Toksöz `[一作]` `[通讯]` (Independent researcher), Mehmet Altan Toksöz (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文探讨了如何通过少量经过验证的示例来改进小型语言模型生成的MATLAB信号处理代码，特别是针对数字信号处理领域的任务。

**💡 创新点**

创新点在于使用LoRA和QLoRA适配器对小型模型进行高效微调，使其能够在笔记本电脑上生成更准确的MATLAB代码，并通过自动化检查器评估生成代码的正确性。

**🔧 技术方法**

使用了低秩适应（LoRA）和量化低秩适应（QLoRA）技术进行模型微调，并利用PEFT库在集成GPU上进行训练和部署。

**📊 数据集**

使用了经过验证的示例数据集，这些示例是从专家解决方案中生成的，确保了训练数据的质量和有效性。

**📈 对比分析**

与之前的商业和开放模型相比，经过微调的模型在生成MATLAB代码的通过率上有显著提高，E2B的通过率从0提升到66，E4B的通过率也有显著提升，表明适配器有效去除了许多错误。

**⚠️ 局限性**

限制在于使用自定义的MATLAB仿真器而非真实的MATLAB环境，可能导致某些代码在真实MATLAB中表现不同。此外，训练数据的样本量较小，可能不足以覆盖所有信号处理任务的复杂性。

---

## 95. Training-Free Building Change Detection with Example-Induced Discriminants

**arXiv ID:** 2611.22964 | [PDF](https://arxiv.org/pdf/2611.22964v1)

**作者:** Zhi Lu `[一作]` (University of Electronic Science and Technology of China), Zhenming Peng `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于变化注释示例的线性建筑分类器的训练无关框架LIND，用于建筑变化检测（BCD）。

**💡 创新点**

创新点在于直接利用变化注释示例构建线性建筑判别器，而无需任务特定的训练，且能够在同一特征空间中实现建筑识别和时间比较。

**🔧 技术方法**

使用了冻结的视觉变换器（vision transformer）和区域特征对比技术。

**📊 数据集**

使用了LEVIR-CD、WHU-CD和S2Looking等多个数据集进行实验，评估了不同的建筑变化检测性能。

**📈 对比分析**

与传统的监督检测器和其他训练无关的方法相比，LIND在LEVIR-CD和WHU-CD数据集上表现出色，尤其在5-shot和10-shot设置下，F1分数和IoU均优于其他方法，且在跨数据集转移中表现良好。

**⚠️ 局限性**

局限性在于该方法依赖于变化注释的覆盖范围和冻结特征的有效性，可能在某些场景中表现不佳。

---

## 96. GapLeak: When Privacy-Preserving LLM-based Web Agents Become a Privacy Risk

**arXiv ID:** 2611.22223 | [PDF](https://arxiv.org/pdf/2611.22223v1)

**作者:** Chaklam Cheong `[一作]` (Queens University), Mohammad Zulkernine `[通讯]` (Queens University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了GapLeak这一结构性漏洞，揭示了当前基于大型语言模型（LLM）的网络代理在隐私保护方面的不足，尤其是在信息缺口导致的未授权信息披露问题。

**💡 创新点**

创新点在于正式化了GapLeak漏洞，指出数据最小化和最小权限原则虽然必要，但不足以完全保护用户隐私，因为代理仍然可以接触到更高敏感度的信息源。

**🔧 技术方法**

使用了基于LLM的网络代理框架进行评估，涉及三种代理框架（BrowserUse、DoBrowser、Agent-E）和六种模型（Claude Haiku、Claude Sonnet、Gemini、GPT等）。

**📊 数据集**

评估使用了四个代表性领域的数据集：招聘、电商、旅行和医疗，每个领域都设计了特定的场景以测试GapLeak的影响。

**📈 对比分析**

与现有的基于最小权限的防御方法进行比较，发现这些方法在减少未授权披露方面效果有限，且通常会降低任务成功率。提出的提交时验证机制能够在不降低任务成功率的情况下，消除超过96%的未授权披露。

**⚠️ 局限性**

限制在于现有的防御措施未能有效阻止信息源的升级，且在某些情况下，代理仍然会接触到敏感信息，导致隐私风险。

---

## 97. PromptPack: Training-Free Augmentation of Diffusion Editors for Intent-Aligned Spatial and Semantic Control

**arXiv ID:** 2611.21865 | [PDF](https://arxiv.org/pdf/2611.21865v1)

**作者:** Sophia Zalewski `[一作]` (NVIDIA), David Dynerman `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为PromptPack的方法，旨在改善图像编辑中的用户意图表达，通过引入两个明确的编辑指南来增强现有的扩散编辑器。

**💡 创新点**

创新点在于引入了编辑角色分区和区域提示分配，使得用户能够更精确地控制图像编辑的空间区域和概念应用，同时不需要重新训练模型。

**🔧 技术方法**

使用了扩散编辑技术，并将PromptPack集成到FlowEdit FLUX、FlowEdit SD3和FLUX Kontext中。

**📊 数据集**

使用了PIE-Bench++数据集，该数据集包含80张图像和160个请求的概念。

**📈 对比分析**

通过与基线编辑器进行比较，PromptPack在区域概念放置和保护内容保留方面表现出显著改善，用户研究显示PromptPack在FlowEdit FLUX中获得了87%的偏好率。

**⚠️ 局限性**

限制在于自动引导仅适用于现有源对象，且包装具有有限容量，可能会扰动非空填充表示。集成需要特定架构的注意力和潜在访问，源恢复可能引入接缝，输出仍受基础编辑器的限制。

---

## 98. A Model-Driven Automation Pipeline for Building Operation

**arXiv ID:** 2611.21631 | [PDF](https://arxiv.org/pdf/2611.21631v1)

**作者:** Elias Kohler `[一作]` (University of Innsbruck), Philipp Zech `[通讯]` (University of Innsbruck)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出了一种模型驱动的流程管道，用于建筑控制系统（BCS）的建模、编程和自动调试，旨在提高效率并减少手动工作和配置错误。

**💡 创新点**

创新点在于引入了两个领域特定建模语言（DSML），实现了从平台独立模型到可执行工件的自动生成和部署，替代了传统的多方交接过程。

**🔧 技术方法**

使用了模型驱动工程（MDE）和模型到文本（M2T）生成技术，结合了自动化验证和代码生成。

**📊 数据集**

使用了Beckhoff自动化硬件（CX5140控制器）进行评估，构建了一个移动测试平台，包含传感器和LED灯。

**📈 对比分析**

通过与传统手动流程的比较，评估显示出显著的效率提升，适应性改善，用户接受度中等到高，且在复杂性上表现出良好的可扩展性。

**⚠️ 局限性**

局限性包括样本量较小（仅15名参与者），且研究在单一硬件平台上进行，未来需要更大规模的工业案例研究以验证结果的普遍适用性。

---

## 99. Learning Calibrated and Samplable Predictive Distributions of Surface-Ocean $Δp$CO$_2$ Using Tabular Diffusion Models

**arXiv ID:** 2611.21769 | [PDF](https://arxiv.org/pdf/2611.21769v1)

**作者:** Gabriele Accarino `[一作]` (Columbia University), Viviana Acquaviva `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种新的条件扩散模型Treeffuser，用于估计海洋表面CO2的全预测分布，解决了现有方法仅提供单一值的局限性。

**💡 创新点**

创新点在于首次应用树基条件扩散模型来学习可采样的Δ pCO2预测分布，并提供观察特定的不确定性估计。

**🔧 技术方法**

使用了Treeffuser模型，这是一种基于梯度提升树的条件扩散模型。

**📊 数据集**

使用了2012年至2022年间的SOCATv2025月度观测数据，空间分辨率为1°×1°。

**📈 对比分析**

与其他点预测模型（如XGBoost和LightGBM）相比，Treeffuser在不确定性量化方面表现最佳，CRPS为8.229 atm，覆盖率接近68%和95%的名义水平，同时保持了与XGBoost相当的点预测准确性。

**⚠️ 局限性**

限制在于Treeffuser的推断速度较慢，且模型未能捕捉空间和时间之间的协方差，未来需要结合空间时间依赖性进行改进。

---

## 100. An AI-Based OSSE Framework to Evaluate Learned Physical Consistency: A Multi-Model Study Using FourCastNet, Aurora, Prithvi WxC, and GraphCast

**arXiv ID:** 2611.22024 | [PDF](https://arxiv.org/pdf/2611.22024v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 101. Cross-Modal StateKV Injection for Communication in Multi-LLM Systems across Temporal Segments

**arXiv ID:** 2611.22304 | [PDF](https://arxiv.org/pdf/2611.22304v1)

**作者:** Xiaoqian Zhang `[一作]` (Shanghai Jiao Tong University), Zhihao Zhang `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了多模态语言模型之间的视觉状态通信，提出了一种通过视觉标记位置的键值状态进行信息传递的框架。

**💡 创新点**

创新点在于提出了Cross-Modal StateKV Injection机制，允许提供者模型通过视觉KV状态与承载模型进行有效的通信，同时保持承载模型的KV缓存长度不变。

**🔧 技术方法**

使用了视觉KV状态的投影和门控残差注入机制，结合了多模态语言模型的特性。

**📊 数据集**

在MLVU和Video-MME数据集上进行了实验，比较了不同通信方式的性能。

**📈 对比分析**

与仅使用承载模型推理和基于文本的通信相比，StateKV在某些任务上表现出适度的准确性提升，并且延迟低于文本通信基线，但在某些类别上准确性没有变化或下降。

**⚠️ 局限性**

局限性在于准确性变化较小且任务依赖性强，缺乏足够的重复运行统计数据来确定不确定性，且未能验证多提供者的扩展性和并行吞吐量。

---

## 102. From Sorry to QED: Refining LLM Math Proofs via Angelic Execution

**arXiv ID:** 2611.22103 | [PDF](https://arxiv.org/pdf/2611.22103v1)

**作者:** Yanju Chen `[一作]` (University of California), Yu Feng `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种框架，将不完美的LLM生成的Lean证明草稿转换为完全验证的证明，结合了符号推理和最小的人类交互。

**💡 创新点**

创新点在于引入了天使证明检查，通过乐观执行模型替换失败的证明片段为占位符，同时保留论证的整体结构，并通过残余分析选择最小成本的必要义务。

**🔧 技术方法**

使用了乐观证明检查、证明依赖图和MaxSAT优化技术。

**📊 数据集**

在30个真实世界的基准测试上进行了评估，涵盖分析和代数领域。

**📈 对比分析**

与五个最先进的自动化系统进行比较，结果显示该框架解决了93.3%的问题，而对比系统的成功率最高为30%。此外，残余分析减少了55%的用户交互次数。

**⚠️ 局限性**

限制在于依赖于初始的LLM生成的证明草稿，如果草稿完全偏离主题，乐观检查和局部细化可能无法恢复有效证明。

---

## 103. Computed Permutation Trellis Coding: Near-Optimal Quantization from Two Multiplications and a 64-Byte Table

**arXiv ID:** 2611.22019 | [PDF](https://arxiv.org/pdf/2611.22019v1)

**作者:** Muhammed Fatih Balın `[一作]` `[通讯]` (Amazon), Muhammed Fatih Balın (Amazon)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种计算排列格编码（CPTC），用于张量的低比特量化格式，适用于一次写入多次读取的场景。

**💡 创新点**

CPTC通过联合编码值来恢复信息理论理想值的损失，且其解码器在性能和资源占用上具有优势，能够在不同比特宽度下保持高质量。

**🔧 技术方法**

使用了两次16位乘法和一个64字节的查找表来实现解码，采用了Viterbi编码算法。

**📊 数据集**

在Llama 3.1 8B和70B模型上进行了实验，使用了YAQA管道进行量化。

**📈 对比分析**

与QTIP的代码本进行比较，CPTC在每个测试的比特宽度上都表现出最低的KL散度，尤其在4比特时，CPTC的KL散度比QTIP低27%。

**⚠️ 局限性**

限制在于实验仅覆盖了Llama 3.1 8B和70B模型，且Trainium 4编码器的测量是在周期级仿真中进行的，而非实际硅片上。

---

## 104. Values Shape What Self-Improving Coding Agents Build

**arXiv ID:** 2611.23067 | [PDF](https://arxiv.org/pdf/2611.23067v1)

**作者:** Jason Liang `[一作]` (Cognizant AI Lab), Risto Miikkulainen `[通讯]` (Cognizant AI Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了在自我改进的编码代理中，短期价值声明是否会改变代理的表现和构建内容。通过在代理的指令文件中插入四种基本人类价值的高阶类型，评估其对多种编码任务的影响。

**💡 创新点**

创新点在于探讨了价值声明如何影响自我改进代理的表现和构建内容，尤其是不同价值对算法的大小、复杂性和方法的影响。

**🔧 技术方法**

使用了大型语言模型（LLM）作为自我改进代理，通过迭代改进求解器，并在每次迭代中固定价值声明。

**📊 数据集**

使用了六个多样化的编码任务数据集，包括ALE-Bench、Circle Packing、Entity Matching、Game 2048、Symbolic Regression和SZ-Tetris。

**📈 对比分析**

通过与无价值控制组和安慰剂组进行比较，发现开放性价值在表现上显著优于保守性价值，而保守性价值的表现低于无价值组。不同价值的代理在解决任务时构建的算法在大小和复杂性上存在显著差异。

**⚠️ 局限性**

限制在于允许代理自我修改价值时，性能差异缩小，且从AI文献中提炼的价值未能显著提高表现，表明人类设计的价值仍然是最有效的。

---

## 105. When Do Typed Decision Models Work in Transportation? Evidence from Travel Choice and Traffic Signal Control

**arXiv ID:** 2611.22926 | [PDF](https://arxiv.org/pdf/2611.22926v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 106. RagHarnessDebugger: Jev-Guided Agentic RAG in Contract Inference Repair Auditing

**arXiv ID:** 2611.22991 | [PDF](https://arxiv.org/pdf/2611.22991v1)

**作者:** Moriya Dechtiar `[一作]` (Harvard University), Sylvain Jaume `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RagHarnessDebugger框架，用于记录和审计基于检索增强生成（RAG）的合同推理修复过程中的每一个决策，以便进行全面的调试。

**💡 创新点**

创新点在于引入了一种无标签的审计层和重放机制，能够追踪修复过程中的每一步决策，并揭示隐藏的失败模式，从而提高RAG系统的可解释性和决策透明度。

**🔧 技术方法**

使用了Jev作为决策模型，结合了基于BM25的检索方法和结构化阅读策略，进行证据的排名和修复决策。

**📊 数据集**

使用了ContractNLI数据集，该数据集包含关于保密协议的假设，每个假设都有三个可能的标签：支持、反驳或未提及。

**📈 对比分析**

通过比较两个版本的RAG修复框架（调试前和调试后），发现调试后的版本在准确性上有所提升，回归率显著降低，且修复率也有所改善，表明调试有效。

**⚠️ 局限性**

局限性在于，尽管RagHarnessDebugger能够揭示修复过程中的失败模式，但仍然依赖于模型的初始输出，且在某些情况下可能无法完全捕捉到所有潜在的修复机会。

---

## 107. Right Diagnosis, Unsafe Route: Auditing Clinical AI Decision Trajectories and Corrective Oversight for Validity and Potential Harm

**arXiv ID:** 2611.22612 | [PDF](https://arxiv.org/pdf/2611.22612v1)

**作者:** Guanqiao Chen `[一作]`, Tianyu Shi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文审计了临床AI系统在决策轨迹中的患者安全，评估了在达到正确最终诊断的过程中，临床决策是否安全。

**💡 创新点**

创新点在于将患者安全作为临床AI评估的核心，提出了轨迹审计和监督审计的方法，分别评估每个中间决策的潜在危害和临床有效性。

**🔧 技术方法**

使用了轨迹审计和监督审计技术，结合临床医生的评分标准进行评估。

**📊 数据集**

使用了来自三种来源的数据集，包括中国某教学医院的心脏病住院病例、MIMIC-IV数据库的成人急诊入院病例和ASCENT公共基准。

**📈 对比分析**

与传统的最终诊断评估方法相比，本文的方法揭示了在达到可接受的最终诊断过程中，许多决策存在潜在的危害。具体性能数据未详细列出。

**⚠️ 局限性**

限制在于审计结果可能受到临床医生主观判断的影响，且不同来源的数据可能导致结果的可比性问题。

---

## 108. Remembered as Brief: Stored Preferences SilentlyRemove Decision-Critical Information

**arXiv ID:** 2611.22697 | [PDF](https://arxiv.org/pdf/2611.22697v1)

**作者:** Joshua Adrian Cahyono `[一作]` `[通讯]` (Nanyang AI Safety), Joshua Adrian Cahyono (Nanyang AI Safety)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文介绍了一个名为Inspect的基准，评估AI助手在处理用户决策时的记忆影响，特别是如何在不同情况下遗漏关键信息。

**💡 创新点**

创新点在于提出了一个系统化的基准，涵盖100个日常决策场景，揭示了AI助手如何根据用户的记忆偏好做出选择并可能导致信息遗漏。

**🔧 技术方法**

使用了多种AI模型进行测试，包括Gemini 2.5 Flash-Lite和DeepSeek V4 Pro，评估它们在不同记忆条件下的表现。

**📊 数据集**

使用了100个场景，涵盖25个领域，场景由Claude Opus 5.5生成，经过GPT-6.1 Terra检查，并由人工审核。

**📈 对比分析**

与其他方法的比较显示，Gemini和DeepSeek在用户偏好存储情况下，93%和84%的回答遗漏了关键信息，而在中性记忆情况下几乎没有遗漏。性能评估显示，存储的偏好会导致模型在80%-95%的情况下为用户做出价值选择。

**⚠️ 局限性**

限制在于未能证明遗漏信息对用户决策的实际影响，且缺乏领域专家的验证，可能导致结果的可靠性受到质疑。

---

## 109. Autonomous Track Racing with a Commercially Available Humanoid: System Design and Real-World Deployment

**arXiv ID:** 2611.22445 | [PDF](https://arxiv.org/pdf/2611.22445v1)

**作者:** Koki Inami `[一作]` (GMO Internet Group), Shohei Matsugu `[通讯]` (GMO Internet Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文描述了一个完整的自主类人机器人跑步系统，该系统在2026年世界类人机器人比赛中部署，能够在新的场地上自主跟随跑道并完成比赛。

**💡 创新点**

创新点在于将视觉基础的跑道感知、路径跟踪和基于强化学习的人类运动先验的跑步策略集成到一个商业可用的类人机器人上，并在实际比赛中进行测试。

**🔧 技术方法**

使用了视觉感知、路径跟踪和强化学习等技术，结合了IMU辅助的腿部里程计和图像稳定技术。

**📊 数据集**

使用的数据集包括3000个真实场景和300个合成场景的图像，用于训练跑道检测模型。

**📈 对比分析**

与其他方法相比，所提出的系统在比赛中表现良好，100米比赛中排名第13，400米比赛中排名第10（第一轮），1500米比赛中排名第七，显示出良好的自主运行能力。

**⚠️ 局限性**

限制因素包括在比赛中由于跑道检测错误导致的摔倒，以及在高温条件下的电机过热问题，这限制了机器人的速度和表现。

---

## 110. Task Convergent Flow Matching for Generative Robotics

**arXiv ID:** 2611.22609 | [PDF](https://arxiv.org/pdf/2611.22609v1)

**作者:** Lorenzo Paiola `[一作]` (Istituto Italiano di Tecnologia), Cosimo Della Santina `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种任务中心的推理时精炼框架，用于流匹配策略，通过并行物理模拟评估候选动作块，并搜索满足任务收敛充分条件的运动。

**💡 创新点**

创新点在于引入了任务级性能漏斗，允许暂时的回退，同时要求每个动作块的净进展，从而推动任务误差在执行过程中降低。

**🔧 技术方法**

使用了基于物理的并行模拟、协方差矩阵自适应进化策略（CMA-ES）等技术。

**📊 数据集**

在PushT和三立方体堆叠任务上进行了实验。

**📈 对比分析**

与未精炼的执行方法相比，精炼执行在PushT上的成功率从31.3%提高到88.5%，在堆叠任务上从77.0%提高到86.0%。

**⚠️ 局限性**

限制在于精炼的质量依赖于手动设计的任务潜力V，且当前的规划过程需要几秒钟，无法实时运行。

---

## 111. EfficientVPR: Toward Efficient Visual Place Recognition via Scene-Aware Prompt Tuning and Adaptive Feature Enhancement

**arXiv ID:** 2611.22506 | [PDF](https://arxiv.org/pdf/2611.22506v1)

**作者:** Wenjing Tang `[一作]` (University of Chinese Academy of Sciences), Yongjun Xu `[通讯]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种高效的单阶段视觉地点识别（VPR）框架EfficientVPR，通过场景感知的视觉提示调优方法和实例依赖的关键局部特征增强模块来提高模型的速度和准确性。

**💡 创新点**

创新点在于引入了场景感知的视觉提示调优方法和实例依赖的关键局部特征增强模块，使得模型在保持较低参数量的同时，能够动态适应样本特征，增强区分能力。

**🔧 技术方法**

使用了DINOv2-small作为骨干网络，并结合场景感知视觉提示调优（SceneVPT）和实例依赖的关键局部特征增强模块。

**📊 数据集**

在多个数据集上进行了评估，包括Pitts250k、MSLS、Eynsham、AmsterTime和SVOX。

**📈 对比分析**

与现有的SOTA方法相比，EfficientVPR在相同规模下的模型中表现出色，达到了73倍的速度提升，并且在特征维度上降低了60%，同时与SOTA DINOv2-large的两阶段方法相比，平均R@1的差距仅为2.5%。

**⚠️ 局限性**

限制在于尽管模型在多个基准测试中表现优异，但在极端环境变化下的适应性仍需进一步验证。

---

## 112. DAP-MR: Degradation-Aware Prompt-Magnitude Routing for Real-World Ultrasound Super-Resolution

**arXiv ID:** 2611.22331 | [PDF](https://arxiv.org/pdf/2611.22331v1)

**作者:** Ngoc-Hoang-Lam Le `[一作]` (National Yang Ming Chiao Tung University), Ching-Chun Huang `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种新的超分辨率模型DAP-MR，专注于超声图像的降解识别和强度分离，以提高超声图像的恢复质量。

**💡 创新点**

创新点在于将降解身份与降解强度分开，通过专用的令牌和输入依赖的强度路由器来实现更精确的超声图像恢复。

**🔧 技术方法**

使用了CLIP初始化的降解令牌和输入依赖的强度路由器，结合了多步和一步扩散超分辨率框架。

**📊 数据集**

使用了合成数据集和真实世界的超声数据集，包括EchoNet、BUSI、CAMUS-Poor和HMC-QU。

**📈 对比分析**

与现有方法相比，DAP-MR在多个基准测试中显示出更好的恢复质量，尤其是在解剖域转移和真实世界数据上表现更为一致。

**⚠️ 局限性**

限制在于模型在训练过程中可能对降解令牌的依赖性不足，且在某些情况下，恢复质量的提升可能不够显著。

---

## 113. Temporal Pseudo-RGB Encoding for Label-Free Detection of VEEV Infection in Human Brain Endothelial Cells

**arXiv ID:** 2611.21892 | [PDF](https://arxiv.org/pdf/2611.21892v1)

**作者:** Zhengjie Zhu `[一作]` (Wake Forest University School of Medicine), Metin Nafi Gurcan `[通讯]` (Wake Forest University School of Medicine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了一种基于时间伪RGB编码的方法，用于在不使用标签的情况下检测人脑微血管内皮细胞中的委内瑞拉马脑炎病毒（VEEV）感染。该方法通过将三个时间偏移的明场图像堆叠为一个图像的红、绿、蓝通道，从而捕捉形态变化的时间进程。

**💡 创新点**

创新点在于采用时间伪RGB编码来整合时间信息，使得深度学习模型能够在不修改架构的情况下，利用预训练的卷积神经网络（CNN）进行感染轨迹的学习。该方法在感染检测的准确性上显著优于传统的单帧模型，尤其是在感染后期。

**🔧 技术方法**

使用了卷积神经网络（CNN），具体为ImageNet预训练的ResNet50，并结合了多任务学习的目标，进行感染状态和感染后时间的预测。

**📊 数据集**

使用的数据集为人脑微血管内皮细胞（HBMVEC），这些细胞被感染了委内瑞拉马脑炎病毒（VEEV）TC-83，并在48小时内每30分钟拍摄一次明场图像，共获得了36个孔的实验数据。

**📈 对比分析**

与单帧模型相比，时间伪RGB编码模型在整个实验过程中准确率提高了4.23个百分点，在后期（40-48小时）提高了23.87个百分点。该模型在三次实验中均表现出一致的优势，且在与其他时间模型（如CNN-LSTM和3D卷积网络）比较时，训练成本显著降低。

**⚠️ 局限性**

本研究的局限性在于使用的生物材料数量有限，所有实验均来自同一商业细胞批次，且未进行外部验证。此外，感染标签是基于实验孔条件分配的，未对单个细胞进行确认，特别是在低感染倍数（MOI 0.1）时，感染和未感染细胞共存的情况可能影响结果的解释。

---

## 114. Learned versus Kinematic Trajectory Prediction under Reduced Observation Frequency

**arXiv ID:** 2611.23198 | [PDF](https://arxiv.org/pdf/2611.23198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 115. blaris: Windowless Random-Access LZSS for Constrained Microcontrollers

**arXiv ID:** 2611.22900 | [PDF](https://arxiv.org/pdf/2611.22900v1)

**作者:** Ihor Lukianov `[一作]` `[通讯]` (Jagiellonian University), Ihor Lukianov (Jagiellonian University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种无窗口的LZSS解压缩器blaris，旨在为资源受限的微控制器提供随机访问能力，消除了对RAM滑动窗口的需求。

**💡 创新点**

创新点在于通过双向可遍历的流和紧凑的二次幂检查点直接在压缩负载中解析回引用，从而实现了与高窗口配置相当的压缩比，同时显著降低了内存使用。

**🔧 技术方法**

使用了无窗口的LZSS解压缩技术，结合了双向流和检查点优化。

**📊 数据集**

使用了一个包含21个文件的语料库，涵盖微控制器固件、C源代码、网页资产和传感器CSV日志等。

**📈 对比分析**

与heatshrink和tamp进行比较，blaris在小切片随机读取时速度快了几个数量级，且在压缩比上与高窗口配置相当，但在较大顺序读取时，流解码器表现更好。

**⚠️ 局限性**

限制在于blaris在处理较大顺序读取时性能下降，且在RAM预算非常紧张的情况下可能无法达到最佳性能。

---

## 116. Let's Talk This Out: Towards Explanation-based Distributed Goal Satisfaction

**arXiv ID:** 2611.22405 | [PDF](https://arxiv.org/pdf/2611.22405v1)

**作者:** Anton Reinecke `[一作]` (Hamburg University of Technology), Sibylle Schupp `[通讯]` (Hamburg University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出了一种组件间通信的框架，以促进基于解释模式和假设-保证合同（A/G-contracts）的组合问题解决，无需中央监督。

**💡 创新点**

创新点在于使用部分MaxSMT求解和量词消除来计算反事实解释消息，并展示了在链拓扑中的可行性。

**🔧 技术方法**

使用了部分MaxSMT求解和量词消除技术。

**📊 数据集**

论文中没有具体提到使用的数据集。

**📈 对比分析**

与现有方法相比，提出的协议能够在没有中央决策者的情况下，允许组件之间进行有效的组织和通信，但在更复杂的拓扑中仍面临挑战。

**⚠️ 局限性**

限制在于当前方法主要针对一维链结构，复杂网络（如分支和循环拓扑）需要额外机制来保证终止。

---

## 117. Spatial Price Formation in Transport Networks: A Variational Approach

**arXiv ID:** 2611.22582 | [PDF](https://arxiv.org/pdf/2611.22582v1)

**作者:** William F. A. Weber `[一作]` (University of Sydney), Mahendra Piraveenan `[通讯]` (University of Sydney)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一个空间价格形成模型，适用于封闭区域经济体，考虑了有限的生产节点和需求节点，运输成本依赖于区域几何形状和可行连接网络。

**💡 创新点**

创新点在于将空间价格建模为受套利和运输可行性约束的空间场，并通过变分问题来表征均衡价格系统，同时使用欧拉-拉格朗日方法进行求解。

**🔧 技术方法**

使用了变分方法和欧拉-拉格朗日方程来分析空间价格均衡，并提出了一种数值计算程序来处理平面域和球面域的扩展。

**📊 数据集**

使用了一个五节点的示例来说明均衡、替代运输模式以及部分价格观察所提供的信息。

**📈 对比分析**

通过与传统的空间价格均衡理论结合，本文的方法在处理不完全观察时能够提供可行的价格范围，并能够排除不适合的运输弧，性能表现良好。

**⚠️ 局限性**

限制在于模型假设了均匀商品和竞争性套利，未来的工作将考虑生产和运输节点的中断对价格、消费和区域福利的影响。

---

## 118. VTT: Reading Vehicle Size and Powertrain Type from Motion Trajectory Across Unseen Car Models

**arXiv ID:** 2611.22109 | [PDF](https://arxiv.org/pdf/2611.22109v1)

**作者:** Yuhang Wang `[一作]` (University of South Florida), Hao Zhou `[通讯]` (University of South Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VTT数据集和基准，用于从运动和控制区域网络（CAN）信号中识别车辆的尺寸和动力系统信息，支持交通分析和模拟。

**💡 创新点**

创新点在于提供了一个包含未见车型的车辆属性识别基准，结合了运动和CAN信号，并进行了跨车型的评估。

**🔧 技术方法**

使用了序列架构（如ResNet、GRU、Transformer等）、统计特征和大型语言模型（LLM）提出的物理特征。

**📊 数据集**

数据集包含9062次驾驶记录，约3950小时，来自1431辆车和174种车型。

**📈 对比分析**

通过比较四种序列架构和特征，训练适应性提高了电池电动汽车与内燃机汽车的宏观F1分数至92.6-93.0%，尺寸分类从62.9%提高到67.1%。

**⚠️ 局限性**

限制在于尺寸识别对小几何误差敏感，且模型训练主要基于单一日志生态系统，区域覆盖不均匀。

---

## 119. Cross-cultural Adaptation of Recipes for Low-Resource Languages: A Benchmark and Multilingual Culinary Lexicon for Hausa, Yorùbá, and Igbo

**arXiv ID:** 2611.22708 | [PDF](https://arxiv.org/pdf/2611.22708v1)

**作者:** Nuhu Ibrahim `[一作]` (University of Manchester), Riza Batista-Navarro `[通讯]` (University of Manchester)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了ReciLoReL，一个包含600个专家翻译和文化适应的食谱的基准数据集，涵盖了尼日利亚的三种低资源语言：约鲁巴语、豪萨语和伊博语。同时，构建了ReciLex，一个包含约15,000个翻译实体的多语言烹饪词典，支持文化适应的食谱翻译。

**💡 创新点**

创新点在于引入了文化适应的食谱翻译基准和评估标准，特别是针对低资源语言的烹饪翻译，提供了新的资源和评估方法。

**🔧 技术方法**

使用了多语言大语言模型（LLMs）进行食谱翻译，并通过ReciLex进行词典增强，以改善文化适应性翻译。

**📊 数据集**

使用的数据集包括从RecipeNLG语料库中提取的600个手动翻译和文化适应的食谱，以及包含15,000个翻译实体的ReciLex词典。

**📈 对比分析**

通过与现有的多语言LLMs进行比较，ReciLex的引入使得BLEU分数从13.57提高到14.51，chrF从37.89提高到38.74，TER从77.14降低到75.12。人类评估显示在五个文化适应评估标准上得分提高了0.11到0.36分。

**⚠️ 局限性**

限制在于ReciLoReL的规模相对较小，限制了所代表的食谱和烹饪实践的多样性。此外，研究主要集中在约鲁巴语、豪萨语和伊博语，限制了结果的普遍性。未来的工作可以扩展到更多低资源语言，以验证词典增强的效果是否适用于不同的语言结构和文化背景。

---

## 120. Sample-efficient Personalization of Large Language Model Alignment with Multidimensional User Preference Data

**arXiv ID:** 2611.22210 | [PDF](https://arxiv.org/pdf/2611.22210v1)

**作者:** Zhen-Yu Zhang `[一作]` (Center for Advanced Intelligence Project RIKEN), Masashi Sugiyama `[通讯]` (Center for Advanced Intelligence Project RIKEN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多维偏好引导的在线直接偏好优化方法，以提高大语言模型的个性化对齐效率。

**💡 创新点**

创新点在于利用低成本的多维评估构建个性化偏好锥，从而指导响应选择，考虑用户历史比较中的偏好信息。

**🔧 技术方法**

使用了多维评估和偏好锥的构建技术。

**📊 数据集**

使用了用户历史比较的数据集。

**📈 对比分析**

与现有方法相比，该方法在匹配的查询预算下表现更优，能够有效过滤掉自信主导的响应并选择具有未解决争议的响应对。

**⚠️ 局限性**

限制在于该方法依赖于用户历史比较的质量，可能在某些情况下无法充分捕捉用户的真实偏好。

---

## 121. Boxes in Infinite Words: Complexity, Automata, and Point Modules

**arXiv ID:** 2611.21772 | [PDF](https://arxiv.org/pdf/2611.21772v1)

**作者:** Jason Bell `[一作]` `[通讯]` (University of Waterloo), Jason Bell (University of Waterloo)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文引入了一种与有限字母表上的右无限词相关的盒计数函数，并研究了这些函数在具有线性有界复杂度的词上的性质。

**💡 创新点**

创新点在于通过引入扩展盒的概念，研究了这些盒的计数函数，并证明了在特定条件下这些函数的最终仿射增长。

**🔧 技术方法**

使用了第一阶逻辑的技术来证明k-自动词的盒计数函数是k-正则的。

**📊 数据集**

使用了与单项代数相关的点模块空间的有限投影数据集。

**📈 对比分析**

通过与已有的复杂度函数进行比较，证明了在特定条件下，正维度的最大扩展盒的数量是最终仿射的。

**⚠️ 局限性**

限制在于对于复杂度稍微超出线性增长的情况，盒计数函数可能不再保持最终仿射性。

---

## 122. Chain-of-Thought Controllability Can Reduce Chain-of-Thought Monitorability

**arXiv ID:** 2611.21799 | [PDF](https://arxiv.org/pdf/2611.21799v1)

**作者:** Zheng-Xin Yong `[一作]` (Brown University), Olivia Watkins `[通讯]` (OpenAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

论文探讨了一种可控的思维链（CoT）模型生物体，展示了其在思维链监控能力上的改进。

**💡 创新点**

创新点在于提出了一种新的模型生物体，能够更好地控制其思维链，同时在执行附加任务时表现出较低的监控能力。

**🔧 技术方法**

使用了可控的思维链模型技术。

**📊 数据集**

未提供具体的数据集信息。

**📈 对比分析**

与基础模型和之前的最先进技术进行了比较，结果显示新模型在思维链控制方面表现更佳，但在执行附加任务时监控能力降低。

**⚠️ 局限性**

限制在于缺乏具体的数据集和实验结果的详细说明。

---

## 123. VisHead-Spec: Head-Routed Visual Attention for Multimodal Speculative Decoding

**arXiv ID:** 2611.21839 | [PDF](https://arxiv.org/pdf/2611.21839v1)

**作者:** Xin Guo `[一作]` (Meta Recommendation Systems), Lizhu Zhang `[通讯]` (Meta Recommendation Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的视觉语言模型（VLM）解码方法VisHead-Spec，旨在提高图像和文本任务的生成效率。

**💡 创新点**

创新点在于通过选择特定的KV头和图像特征层，减少了图像上下文的构建和注意力成本，同时保持了生成的准确性。

**🔧 技术方法**

使用了基于块扩散的解码器架构，并结合了KV注入技术来优化图像和文本的注意力机制。

**📊 数据集**

使用了MMSpec基准测试中的67,304个目标回答序列，涵盖了多种任务类别，包括文本VQA、图像标注等。

**📈 对比分析**

与其他方法（如DSpark、DFlash等）进行比较，VisHead-Spec在高并发情况下的吞吐量提高了16.0%，而在批量大小为1时，其接受长度与基线相当。

**⚠️ 局限性**

限制在于该方法仍然依赖于特定的KV头选择，可能在不同任务或数据集上表现不一致。

---

## 124. Python Code Generation for Isabelle/HOL

**arXiv ID:** 2611.22613 | [PDF](https://arxiv.org/pdf/2611.22613v1)

**作者:** Nikolaus Huber `[一作]` (University of Cambridge), Dominic Orchard `[通讯]` (University of Cambridge)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

本文实现了Isabelle/HOL的代码生成器的Python后端，使得经过验证的算法可以在Python应用中使用，或作为已建立Python库的验证参考。

**💡 创新点**

创新点在于首次将Isabelle的代码生成器扩展到Python，填补了Python在形式化验证中的空白，并提供了一个可验证的参考实现。

**🔧 技术方法**

使用了Isabelle/HOL的代码生成框架，结合Python的动态类型特性和结构模式匹配，生成可执行的Python代码。

**📊 数据集**

使用了Isabelle/HOL库中的回归测试套件，涵盖了大量的HOL库部分，并对六种经过验证的数据结构和算法进行了测试。

**📈 对比分析**

通过差异测试与已建立的Python库进行比较，生成的代码在语法上有效，且在行为上与标准ML和Python的实现一致。性能上，生成的代码比手写库慢62-142倍，但在PyPy上性能差距缩小至6.5-27倍。

**⚠️ 局限性**

限制在于生成的代码在性能上明显低于手写库，主要由于调用开销、不可变数据结构和类型类字典的使用。此外，递归深度限制和缺乏类型注释也是当前实现的不足之处。

---

## 125. Honest Brokers in AI Regulation: How Public Comments Introduce Evidence to Regulators

**arXiv ID:** 2611.21948 | [PDF](https://arxiv.org/pdf/2611.21948v1)

**作者:** Sarah Tran `[一作]` (University of Washington), Nicholas Weber `[通讯]` (University of Washington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了公众评论者在美国人工智能监管中提供证据的角色，构建了一个包含40个潜在法规和2664条公众评论的语料库，并对这些文本进行了计算处理，以提取用于告知监管者的证据。

**💡 创新点**

创新点在于揭示了不同类型的评论者在提供证据方面的显著差异，尤其是商业评论者提供的证据量明显低于政府、大学和民间社会评论者。

**🔧 技术方法**

使用了计算文本分析技术，特别是对PDF格式的公众评论进行文本提取和证据分类。

**📊 数据集**

使用的数据集包括40个与人工智能相关的联邦法规和2664条公众评论，这些评论来自不同的利益相关者。

**📈 对比分析**

通过比较不同类型评论者的证据引用量和类型，发现政府、大学和民间社会评论者提供的研究和法律来源显著多于商业评论者，且在相同评论长度下，商业评论者的证据量明显较少。

**⚠️ 局限性**

限制在于未能分析引用的上下文，因此无法确定评论者引用的证据意图；此外，网站作为引用来源的多样性使得内容的具体影响难以评估。

---

## 126. Agent Trajectories as Evidence for Software Quality Assurance

**arXiv ID:** 2611.22323 | [PDF](https://arxiv.org/pdf/2611.22323v1)

**作者:** Jonathan Ullrich `[一作]` (Fraunhofer IESE), Andreas Jedlitschka `[通讯]` (Fraunhofer IESE)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种分析自主软件代理轨迹的方法，以作为软件质量保证的证据，旨在识别重要决策并改善软件开发实践。

**💡 创新点**

创新点在于将可观察的代理轨迹视为软件工程的第一类工件，帮助识别和评估决策的合理性及其对软件问题的贡献。

**🔧 技术方法**

使用了代理轨迹分析、因果推断、经验软件工程等技术。

**📊 数据集**

使用了工业项目中的执行记录和初步分析方法，特别是涉及从COBOL迁移到Java的项目。

**📈 对比分析**

与传统的质量保证方法相比，该方法能够更有效地聚焦于重要决策的审查，尽管尚未完全确定在何种条件下能够可靠地恢复重要决策。

**⚠️ 局限性**

限制在于尚不清楚哪些决策值得审查，如何评估和结合审查标准，以及如何有效地解释软件问题。

---

## 127. GauntletScore: Auditable Fact-Checking That Matches the Best Published Checkers on LLM-AggreFact

**arXiv ID:** 2611.22027 | [PDF](https://arxiv.org/pdf/2611.22027v1)

**作者:** Ernie Hobbs `[一作]` (Genstrata), Jim Hinson `[通讯]` (Genstrata)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为GauntletScore的证据引擎，用于检查生成句子是否得到其源文档的支持。

**💡 创新点**

创新点在于采用固定的贝叶斯规则结合多个独立读者的回答来评估句子的支持程度，而不是依赖单一模型的判断。

**🔧 技术方法**

使用了贝叶斯评分方法，并结合了来自不同提供者的多个决策模型作为读者。

**📊 数据集**

使用了LLM-AggreFact测试集，该数据集包含29,320个声明，涵盖11个数据集。

**📈 对比分析**

与单独使用的两个读者（Reader A和Reader B）进行比较，GauntletScore在相同声明上分别超越了它们1.66和2.17个百分点，整体表现优于所有已发布的平均水平。

**⚠️ 局限性**

局限性在于该基准测试仅针对特定任务，且依赖于不受控制的商业模型，无法保证未来的可重复性。

---

## 128. Where Do Highly Cited Scientists Stand? A Multi-Dimensional Analysis of Positional Distribution in the Top 2% Ranking

**arXiv ID:** 2611.22817 | [PDF](https://arxiv.org/pdf/2611.22817v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 129. Natural Language Programming: Rethinking Language, Memory, and Cognition in the Era of Generative AI

**arXiv ID:** 2611.23106 | [PDF](https://arxiv.org/pdf/2611.23106v1)

**作者:** Giovanni Adorni `[一作]` (University of Genoa), Daniele Grosso `[通讯]` (University of Genoa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了自然语言编程（NLPg）的概念，作为传统自然语言处理（NLP）的理论扩展，强调提示作为动态认知工具在混合人机工作流中的作用。

**💡 创新点**

创新点在于将提示视为可执行的认知工件，支持推理的外部化和上下文的连续性，提出了对话工作记忆的概念，描述了提示和对话状态如何作为临时外部记忆结构支持推理的连续性。

**🔧 技术方法**

使用了分布式认知、扩展心智、元认知和对话学习等理论，提出了对话工作记忆框架，并将现有的记忆架构（BTMA、SKGE、RMR）映射到该框架上。

**📊 数据集**

论文未具体提及使用的数据集，但讨论了与大型语言模型（LLMs）交互的特性和机制。

**📈 对比分析**

与现有方法相比，本文强调NLPg作为一种分布式认知基础设施，支持推理过程的外部化和上下文的稳定性，性能上表现出更好的反思调节和认知协调能力。

**⚠️ 局限性**

限制在于该框架主要是理论和概念性的，尚需进一步的实证研究和架构验证，未来研究可探索对话记忆的持久性、元认知可追溯性等方面。

---

## 130. Binary Affine Optimization with Nearly-Graphic Constraint Matrices

**arXiv ID:** 2611.22944 | [PDF](https://arxiv.org/pdf/2611.22944v1)

**作者:** Qiming Fang `[一作]` (Peking University), Yuxuan Wu `[通讯]` (Peking University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种多项式时间算法，用于解决每列约束矩阵的汉明重量最多为2的二元仿射优化问题，并证明当每列的汉明重量恰好为3时该问题是NP-hard的。

**💡 创新点**

创新点在于提出了基于近图形证书的固定参数可解(FPT)算法，并将其应用于Max-Lin_2问题，特别是通过欧拉生成集的概念来解决图上的Max-Lin_2问题。

**🔧 技术方法**

使用了固定参数可解(FPT)算法，结合了图论中的欧拉生成集和近图形证书的概念。

**📊 数据集**

使用了图的约束距离和约束秩作为参数，具体数据集未明确提及，但涉及到图的结构和性质。

**📈 对比分析**

与其他方法比较时，提出的FPT算法在约束距离和约束秩的参数化下表现良好，能够在不需要额外信息的情况下解决Max-Lin_2问题。

**⚠️ 局限性**

限制在于尚未找到具有最优或近似最优参数的近图形证书，并且在算法中未明确给出最大化|V|和|E|的指数的最小化。

---

## 131. Corrupting Ties: Investigating Robustness to Bias Injection Across Multi-Agent Network Topologies

**arXiv ID:** 2611.21804 | [PDF](https://arxiv.org/pdf/2611.21804v1)

**作者:** Cléa Chataigner `[一作]` (MILA), Golnoosh Farnadi `[通讯]` (MILA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种控制偏见注入框架，用于分析多智能体系统中偏见信息的传播及其对决策的影响。

**💡 创新点**

创新点在于引入了控制偏见注入的方法，能够在特定位置替换智能体的响应，并通过决策偏差度量来量化这些干预的影响。

**🔧 技术方法**

使用了控制偏见注入框架和决策偏差度量，分析了不同网络拓扑结构对偏见传播的影响。

**📊 数据集**

使用了三个常见的二分类数据集：Adult、COMPAS和Diabetes，分别用于不同的公平性分析任务。

**📈 对比分析**

通过比较五种网络拓扑结构的性能，发现顺序网络对偏见注入的敏感性最高，而完全连接和网状网络的影响较小。注入位置也很重要，中心智能体的注入效果显著高于外围智能体。

**⚠️ 局限性**

限制在于研究仅考虑了五个同质智能体的简单网络，未能反映现实世界中多智能体系统的异质性和规模。未来的研究可以探索更复杂的任务和人机交互的偏见传播。

---

## 132. On Traffic Characterization in Time-Sensitive Networking and Deterministic Networking

**arXiv ID:** 2611.22744 | [PDF](https://arxiv.org/pdf/2611.22744v1)

**作者:** Yuming Jiang `[一作]` `[通讯]` (Norwegian University of Science and Technology), Yuming Jiang (Norwegian University of Science and Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文研究了时间敏感网络（TSN）和确定性网络（DetNet）中的流量特征问题，提出了一种新的TSN流量规范（TSpec）等效表征，简化了符合性验证和分析开发。

**💡 创新点**

创新点在于建立了TSN TSpec的等效表征，提供了输出特征属性，并展示了经过有界延迟服务器的TSN约束流的输出可以用更新的TSN TSpec参数进行表征。

**🔧 技术方法**

使用了网络演算（network calculus）技术，结合了令牌桶模型（token-bucket model）来分析流量特征。

**📊 数据集**

未具体提及使用的数据集，但通过案例研究展示了TSN配置示例的应用。

**📈 对比分析**

与现有方法相比，本文的方法在输出特征和流量聚合的分析上提供了更简化和系统化的结果，性能分析显示了流量在网络中转发时参数可能变得更加保守。

**⚠️ 局限性**

限制在于尽管TSN TSpec在转发和聚合下得以保留，但随着流量在网络中传输，结果参数可能会变得更加保守，从而导致资源需求的高估。

---

## 133. AI as Anchor: How Interview Sequencing Relates to the Appropriation of Synthetic Stakeholder Voices in Design Education

**arXiv ID:** 2611.22740 | [PDF](https://arxiv.org/pdf/2611.22740v1)

**作者:** Osnat Mokryn `[一作]` (Information Systems University of Haifa), Orit Shaer `[通讯]` (Computer Science Wellesley College)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了在设计教育中，AI模拟利益相关者（AIS）与人类利益相关者访谈顺序的关系。通过对21名学生在一个学期的设计课程中的实践进行观察，研究了学生如何构建、评估和使用AIS，以及访谈顺序如何影响他们对信息的解释和使用。

**💡 创新点**

创新点在于将AIS视为一个参考点，其在正式访谈顺序中的位置影响其在设计过程中的作用。此外，提供了探索性实证证据，表明不同的访谈顺序条件与对利益相关者输入的解释和使用模式相关联。

**🔧 技术方法**

使用了自然语言处理技术，特别是大型语言模型（LLM）生成的对话代理，来模拟利益相关者的对话。

**📊 数据集**

数据集来自于一个学期的设计课程，参与者为21名学生，分为8个团队，团队在课程中构建和评估AIS，并进行人类和AIS的访谈。

**📈 对比分析**

通过比较两种访谈顺序（AIS访谈优先和人类访谈优先），发现AIS在访谈顺序不同的情况下，其作用和影响也有所不同。AIS访谈优先的团队通常将AIS视为初步框架，而人类访谈优先的团队则更多地将AIS用于验证和扩展已知问题。整体来看，人类利益相关者的输入在最终设计中更具影响力。

**⚠️ 局限性**

本研究的局限性包括样本量较小，仅在一个学期的课程中进行，且未考虑参与者的先前经验。此外，访谈顺序与AIS的调整机会交织在一起，可能影响结果的解释。

---

## 134. SandBAR: A Band-Asymmetric Architecture for Underwater Image Restoration with Removable Frequency Scaffolding

**arXiv ID:** 2611.22355 | [PDF](https://arxiv.org/pdf/2611.22355v1)

**作者:** Yanzi Liu `[一作]` (King's College London), Shuang Chen `[通讯]` (Durham University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的统一带不对称架构，用于水下图像恢复，旨在解决水下图像的颜色失真和细节丢失问题。

**💡 创新点**

创新点在于将带不对称性作为架构设计的核心原则，并在块、编码器和目标层面进行协调设计，特别是引入了带不对称的三明治块和可移除的非线性支架。

**🔧 技术方法**

使用了一种新的带不对称三明治块和可移除的非线性支架方案，结合了通道感知的大支持修正和轻量级的局部细节修复。

**📊 数据集**

使用了公开的4800对水下恢复基准数据集，包括800个UIEB、2000个EUVP和2000个LSUI图像。

**📈 对比分析**

与其他方法相比，该方法在EUVP、LSUI和UIEB上分别达到了新的PSNR值，超越了已发布的最强结果，且在推理时没有增加参数或FLOPs，保持了匹配的延迟。

**⚠️ 局限性**

限制在于尽管架构在推理时没有增加复杂性，但在训练过程中需要使用可移除的非线性支架，这可能会影响训练的复杂性和时间。

---

## 135. Sparse Integer Fourier Transform and Its Application In Machine Learning

**arXiv ID:** 2611.21976 | [PDF](https://arxiv.org/pdf/2611.21976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

---

## 136. MASQ: Measurement-Adaptive Sampling for Near-Clifford Quantum Error Correction Circuits

**arXiv ID:** 2611.22232 | [PDF](https://arxiv.org/pdf/2611.22232v1)

**作者:** Chaebin Jung `[一作]` (Chung Ang University), Woojoo Lee `[通讯]` (Chung Ang University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为MASQ的精确测量自适应采样器，用于近Clifford量子误差校正电路，通过选择性地物化非Clifford旋转来减少活跃状态瓶颈。

**💡 创新点**

创新点在于通过测量相关闭包延迟不必要的旋转，并利用稳定子辅助消除所有未来测量不可见的旋转，同时自适应缓存进一步减少跨轮次的重复工作。

**🔧 技术方法**

使用了测量自适应采样技术，结合稳定子辅助消除和自适应缓存机制。

**📊 数据集**

在19个量子误差校正基准测试上进行评估，包括表面码存储器和魔态提纯等。

**📈 对比分析**

与现有的精确近Clifford采样器Clifft、Tsim和SymFT进行比较，MASQ在多轮电路上实现了高达8098倍的CPU吞吐量提升，并且在内存或编译限制下完成了现有精确模拟器无法完成的电路。

**⚠️ 局限性**

限制在于编译时间和内存使用，尽管MASQ在性能上有显著提升，但在某些情况下仍可能受到编译时间的限制。

---

## 137. Can Decision Models Replace LLMs in Real-World Contact Center Classification and Judging Tasks?

**arXiv ID:** 2611.22949 | [PDF](https://arxiv.org/pdf/2611.22949v1)

**作者:** Md Tahmid Rahman Laskar `[一作]` (Dialpad Inc.), Shashi Bhushan TN `[通讯]` (Dialpad Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文比较了九种决策模型与十三种生成型大语言模型（LLMs）在八个真实世界的呼叫中心任务上的表现，主要包括六个分类任务和两个判断任务。

**💡 创新点**

创新点在于展示了许多决策模型在呼叫中心分类任务中能够与前沿LLMs的性能相匹配，并且API基础的封闭决策模型在每个请求的成本上最低。

**🔧 技术方法**

使用了决策模型和生成型LLMs进行零-shot评估，决策模型返回带有概率的类型化答案，而生成型LLMs则返回结构化输出。

**📊 数据集**

使用了来自Dialpad的真实世界数据集，包括客户与代理的通话记录、用于辅助代理的文档和AI生成的响应。

**📈 对比分析**

通过比较决策模型和生成型LLMs在分类和判断任务上的表现，发现许多决策模型在分类任务上表现良好，但在判断任务上表现较差，尤其是小模型的判断质量显著下降。

**⚠️ 局限性**

限制在于研究是离线的，未测量实时客户结果，且使用的标签过程未记录，意图发言是合成的。模型选择应依赖于弃权测试、判断一致性、信心审计和成本测量，而不仅仅是整体准确性。

---

## 138. MM-ReportBench: A Comprehensive Benchmark for Multimodal Report Generation from Tables

**arXiv ID:** 2611.22514 | [PDF](https://arxiv.org/pdf/2611.22514v1)

**作者:** Teng Lin `[一作]` (Hong Kong University of Science and Technology), Nan Tang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个综合性的基准，旨在从表格生成多模态报告，涵盖386个任务和185个真实世界表格。

**💡 创新点**

创新点在于设计了一个双评估框架，分别评估文本和可视化质量，解决了现有基准的单一模态关注和评分膨胀问题。

**🔧 技术方法**

使用了蒙特卡洛树搜索（MCTS）生成参考报告，并结合了多维度奖励和多阶段人类质量控制。

**📊 数据集**

构建了一个包含185个真实世界商业表格的数据集，涵盖六个高影响力的工业领域。

**📈 对比分析**

与20个最先进的模型进行了广泛实验，发现即使是最强模型的整体得分也仅为68.9，且在洞察新颖性和可视化质量方面存在显著不足。

**⚠️ 局限性**

限制在于当前模型在生成报告时仍然面临数值幻觉和缺乏真正新颖洞察的问题，且可视化质量仍然是一个瓶颈。

---

## 139. CEVA: Complementary Evidence Verification via Frame-Level Reorganization for Short-Video Fake News Detection

**arXiv ID:** 2611.22141 | [PDF](https://arxiv.org/pdf/2611.22141v1)

**作者:** Wenshen Yang `[一作]` (Sichuan University), Haizhou Wang `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的短视频假新闻检测框架CEVA（Complementary Evidence Verification Approach），通过在多模态融合之前明确组织帧的证据角色来提高检测效果。

**💡 创新点**

创新点在于首次引入了双分支互补重组（DBCR）模块，能够将显式对齐帧与候选互补上下文帧分开，并在重组的证据上进行跨模态融合。

**🔧 技术方法**

使用了双分支互补重组（DBCR）模块、跨模态对比学习（bidirectional contrastive regularization）等技术。

**📊 数据集**

在FakeSV和FakeTT数据集上进行了评估，FakeSV包含来自抖音/快手的中文视频，FakeTT包含英文TikTok视频。

**📈 对比分析**

与现有方法相比，CEVA在固定帧预算和骨干网络下的表现优于均匀、随机和基于相似性的选择方法，并且在两个结构不同的检测器上均有改进。

**⚠️ 局限性**

限制在于该方法依赖于视频中的证据组织，可能在某些情况下无法处理缺乏上下文变化的视频内容。

---

## 140. GridDriver: On-Board BEV-Based Language Planning and Reasoning

**arXiv ID:** 2611.21924 | [PDF](https://arxiv.org/pdf/2611.21924v1)

**作者:** Katharina Winter `[一作]`, Fabian B. Flohr `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了GridDriver，一个基于鸟瞰视图（BEV）的语言规划框架，结合了轨迹预测、结构化驾驶指示、自然语言导航、驾驶解释和场景问答。

**💡 创新点**

创新点在于将轻量级的BEV感知与大型语言模型（LLM）结合，实现了实时的语言交互驾驶，并在规划中引入了自然语言输入和输出。

**🔧 技术方法**

使用了BEVFormer-tiny作为感知模块，结合了Qwen3-1.7B的LLM进行轨迹规划和语言生成。

**📊 数据集**

使用了nuScenes和NAVSIM等公共数据集，以及自有的城市驾驶数据集进行评估。

**📈 对比分析**

与其他方法相比，GridDriver在nuScenes上实现了0.33米的平均L2误差和0.10%的碰撞率，在NAVSIM上达到了87.9%的PDMS，表现出竞争力。

**⚠️ 局限性**

局限性包括BEV分辨率较粗（每个单元2.048米）、覆盖范围有限（±51.2米）以及跨域感知的不足，影响了场景理解和驾驶行为的适宜性。

---

## 141. A Survey of Modeling Languages for Automotive Digital Twins

**arXiv ID:** 2611.21636 | [PDF](https://arxiv.org/pdf/2611.21636v1)

**作者:** Jerômé Pfeiffer `[一作]`, Andreas Wortmann `[通讯]` (University of Stuttgart)

**通讯引用:** 4655 | [OpenAlex ID](https://openalex.org/A5052537841)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过问卷调查，识别德国汽车行业专家对汽车数字双胞胎的期望、建模技术及其在汽车开发各阶段的应用。

**💡 创新点**

创新点在于首次从工业实践的角度调查汽车数字双胞胎的建模语言和工具，填补了科学文献与工业实践之间的空白。

**🔧 技术方法**

使用了问卷调查技术，结合定量分析和统计方法（如皮尔逊相关系数和T检验）来分析专家的反馈。

**📊 数据集**

调查对象为德国汽车行业的189名专家，最终收集到96份有效问卷。

**📈 对比分析**

与现有文献相比，本研究发现数字双胞胎在汽车开发的使用和支持阶段最为重要，专家们认为监控、验证和预测是其主要目的。与其他方法相比，参与者普遍认为建模语言和工具的整合至关重要。

**⚠️ 局限性**

本研究的局限性在于样本量较小，且主要集中在德国汽车行业，可能影响结果的普遍适用性。此外，参与者的背景多为计算机科学领域，可能导致对数字双胞胎的理解存在偏差。

---

## 142. Event-Based Dynamic Lip Region Estimation During Speech

**arXiv ID:** 2611.23111 | [PDF](https://arxiv.org/pdf/2611.23111v1)

**作者:** Hamze Hammami `[一作]` (Independent Researcher), Francesco Rea `[通讯]` (Italian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出了一种基于事件相机的唇部区域定位模型，该模型从完整的事件流中预测唇部的密集热图，解决了现有方法依赖手动裁剪或仅提供唇部特征点的问题。

**💡 创新点**

创新点在于将唇部区域定位作为主要目标，使用极性分离的体素网格和双时间尺度的连续时间递归网络（CT-RNN），并在未裁剪的事件帧上进行训练和评估。

**🔧 技术方法**

使用了极性分离的体素网格和双时间尺度的连续时间递归网络（CT-RNN），该网络具有学习的时间常数，能够处理每个位置的事件数据。

**📊 数据集**

训练使用了一个模拟数据集，评估则使用了两个真实事件相机数据集（FES和FACEMORPHIC），其中包含多位说话者的面部视频序列。

**📈 对比分析**

与卷积神经网络（CNN）等基线方法相比，提出的模型在未见说话者上达到69.5%的唇部定位准确率，且每秒切换次数为2.1次，明显优于CNN的8.9次切换，表现出更好的时间稳定性。

**⚠️ 局限性**

限制在于训练数据仅限于固定对比度阈值的模拟事件，且在真实传感器上，唇部定位率下降。此外，CNN在精度上优于提出的模型，但提出的模型在时间稳定性上具有优势。

---

## 143. A Chain-of-Thought Prompt Engineering Framework for Automated Technical Specification Generation in Enterprise Integration: A Real-World Case Study

**arXiv ID:** 2611.21617 | [PDF](https://arxiv.org/pdf/2611.21617v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 144. Making the World React Realistically: A Closed-Loop Interactive World Model for Autonomous Driving

**arXiv ID:** 2611.22691 | [PDF](https://arxiv.org/pdf/2611.22691v1)

**作者:** Jia Hu `[一作]` (Tongji University), Jintao Lai `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种闭环交互式世界模型，结合了行为建模和视觉渲染，以生成超出记录日志的驾驶场景，从而提高自动驾驶系统在未见交互中的训练和评估能力。

**💡 创新点**

创新点在于通过视觉反馈驱动的交互学习实现更真实的背景车辆行为，利用交互引导的选择性高斯资产增强提高新视角的保真度，以及通过反事实微调实现意图驱动的车辆信号灯渲染。

**🔧 技术方法**

采用了模仿学习（IL）、多智能体强化学习（MARL）、3D高斯渲染、扩散模型等技术。

**📊 数据集**

使用了来自NAVSIM的288个长尾失败场景的数据集，这些场景来自89个nuPlan日志。

**📈 对比分析**

与现有的世界模型（如RealEngine、OmniNWM和World Engine）相比，提出的方法在288个闭环场景中将与真实世界交互行为的差异减少了85%，并在新视角车辆的PSNR上提高了2.0 dB。

**⚠️ 局限性**

限制在于模型仍然依赖于训练数据的质量和多样性，可能在极端或未见的场景中表现不佳。

---

## 145. Let Samples Sing in Chorus: Mitigating Spurious Correlations via Cross-view Consensus

**arXiv ID:** 2611.22668 | [PDF](https://arxiv.org/pdf/2611.22668v1)

**作者:** Weiwei Li `[一作]` (University of Electronic Science and Technology of China), Wen Li `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为CHORD的多视角去偏见框架，通过不同特征空间的样本进行偏见发现和去除，而无需人工标注的偏见属性。

**💡 创新点**

CHORD的创新点在于将偏见候选的提出与验证解耦，允许在多个表示空间中发现偏见，从而减少对目标几何形状的依赖。

**🔧 技术方法**

使用了多视角的表示学习技术，通过不同的特征空间（如预训练视图和几何增强视图）来提出候选偏见，并在目标空间中进行验证。

**📊 数据集**

在七个基准数据集上进行了实验，包括视觉（如Waterbirds和CelebA）、语言（如MultiNLI和CivilComments-WILDS）和医学成像（如CheXpert）等任务。

**📈 对比分析**

与其他方法相比，CHORD在没有偏见监督的情况下，在四个标准基准上实现了最高的最差组准确率（WGA），显示出其在去偏见任务中的优越性能。

**⚠️ 局限性**

CHORD的局限性在于它需要在冻结的目标表示中恢复核心特征，并且将反转证据限制在目标异常样本池中。在某些专业领域，适合的预训练编码器可能较少可用。

---

## 146. Higher-Scoring Coding Models Did Not Get More Work Done

**arXiv ID:** 2611.23216 | [PDF](https://arxiv.org/pdf/2611.23216v1)

**作者:** Yining Hua `[一作]` `[通讯]` (Harvard University), Yining Hua (Harvard University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过自然实验评估了高分编码模型（Fable 5）对开发者生产力的影响，发现其对实际工作完成量没有显著提升。

**💡 创新点**

创新点在于首次通过自然实验方法，比较同一开发者在不同模型下的生产力，揭示了基于基准分数的模型宣传与实际工作成果之间的脱节。

**🔧 技术方法**

使用了差异中的差异模型（difference-in-differences model）来分析数据，并结合开发者和日期的固定效应。

**📊 数据集**

数据集来自公共GitHub，涉及297名使用Fable 5的开发者和8908名始终使用Opus 4.8的开发者的拉取请求（pull requests）。

**📈 对比分析**

与其他方法的比较显示，Fable 5的使用仅使拉取请求的7天合并率提高了1.4个百分点，且没有统计学意义；五次模型升级的合并率变化也不显著。

**⚠️ 局限性**

限制在于研究仅关注了拉取请求的合并率，未能全面反映代码质量和开发者的实际工作效率，且样本量和时间窗口可能影响结果的可靠性。

---

## 147. From Relevance to Utility: Rubric Alignment in Yuanbao Search

**arXiv ID:** 2611.22503 | [PDF](https://arxiv.org/pdf/2611.22503v1)

**作者:** Yu Lu `[一作]` (Tencent Inc.), Haibo Shi `[通讯]` (Tencent Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为“基于评分标准的搜索”的框架，旨在优化文档集以提高下游大语言模型（LLM）的推理能力。

**💡 创新点**

创新点在于将传统搜索管道统一在一个评分标准下，通过定义正负标准来指导文档集的选择和评分，从而提高答案质量。

**🔧 技术方法**

使用了轻量级标签器、对比双编码器和多任务排名器，以及基于强化学习的选择器，结合了GRPO和最佳回滚蒸馏技术。

**📊 数据集**

使用了62M的生产查询和610M的查询-文档对进行训练，另外还有180K查询和3.2M对用于选择训练。

**📈 对比分析**

通过离线评估和在线A/B测试与现有生产基线进行比较，结果显示整体答案质量提高了8.52%。

**⚠️ 局限性**

局限性在于评分标准的构建和应用需要大量的计算资源，且在动态更新评分标准以适应快速变化的主题方面仍有待改进。

---

## 148. Can Vision-Language Models Truly Understand Images? A Benchmark Beyond Commonsense Bias

**arXiv ID:** 2611.22360 | [PDF](https://arxiv.org/pdf/2611.22360v1)

**作者:** Ling Lin `[一作]`, Jingrun Chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ContraSense基准，旨在评估视觉-语言模型对图像内容的真实理解能力，特别是在违反常识的情况下。

**💡 创新点**

创新点在于构建了一个包含4698个反直觉图像-文本对的基准，旨在减少统计偏差的影响，迫使模型依赖视觉证据进行推理。

**🔧 技术方法**

使用了文本生成模型和文本到图像生成模型，结合人类校对来确保数据的质量和一致性。

**📊 数据集**

使用了ContraSense数据集，该数据集包含4698个高质量的反直觉问答样本，涵盖多种场景和风格。

**📈 对比分析**

与现有模型的比较显示，当前的多模态大语言模型在ContraSense基准上的表现显著低于人类，最高闭源模型的准确率约为60%，而人类参与者的准确率为96.04%。

**⚠️ 局限性**

限制在于当前模型在处理反直觉任务时仍然存在显著的性能瓶颈，尤其是在需要深度理解和推理的场景中表现不佳。

---

## 149. Operating an Agent-Driven Dataset Factory: Discovery, Construction, and Public Release

**arXiv ID:** 2611.21957 | [PDF](https://arxiv.org/pdf/2611.21957v1)

**作者:** Eduardo Ariño de la Rubia `[一作]` (Central European University), Szilard Pafka `[通讯]` (Epoch)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究开发了一个数据集工厂（Dataset Factory），用于将公共数据转化为预测数据集，涵盖了源发现、任务定义、构建和发布等步骤。

**💡 创新点**

创新点在于实现了一个自动化的工作流程，能够在持续发布政策下高效地生成和发布多个数据集，同时记录了操作成本和失败经验。

**🔧 技术方法**

使用了模型/控制器的工作分配方法，结合了自动化的源发现、构建和语义审查技术。

**📊 数据集**

使用了公共记录作为数据集来源，具体数据集数量为14个，涵盖了多个领域的预测任务。

**📈 对比分析**

通过与未调优的XGBoost参考模型进行比较，所有12个评估的发布版本在完成的预测实验中表现优于参考模型，且在多个任务上超越了随机搜索控制。

**⚠️ 局限性**

限制在于这是一个短期观察窗口的单一工厂，公共源可能在模型训练中出现，且源审查的结果并未建立跨发布的缺陷率。

---

## 150. Do LLM Agents Still Need to Be Told How to Self-Improve?

**arXiv ID:** 2611.23219 | [PDF](https://arxiv.org/pdf/2611.23219v1)

**作者:** Shobhita Sundaram `[一作]`, Phillip Isola `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了一种自主学习的范式，允许LLM代理在没有外部指导的情况下，利用基本工具和资源自行构建自我学习算法，以实现基于上下文的优化。

**💡 创新点**

创新点在于探讨了在没有明确自我优化算法的情况下，模型是否仍然能够有效地进行自我改进，并发现足够强大的模型能够在这种条件下自我组织学习过程。

**🔧 技术方法**

使用了冻结的LLM代理，提供了计算预算和经验收集工具，允许模型在一个受限的环境中自由实验。

**📊 数据集**

使用了PuzzleScript作为测试平台，进行了一系列游戏的自我改进实验。

**📈 对比分析**

通过与传统的自我优化方法进行比较，发现自组织学习在数据缺乏的环境中能够实现与规定优化循环相当的性能提升，且不同模型家族的学习策略表现出相似性。

**⚠️ 局限性**

限制在于未能完全解开自我调节策略与自我改进之间的因果关系，且在更开放的领域中，模型的表现是否会延续这些趋势仍然是一个未解的问题。

---

## 151. Unsupervised Multi-modal Medical Image Registration via Invertible Translation

**arXiv ID:** 2611.22152 | [PDF](https://arxiv.org/pdf/2611.22152v1)

**作者:** Mengjie Guo `[一作]` (University of Birmingham), Jinming Duan `[通讯]` (University of Manchester)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种新的无监督多模态医学图像配准方法，称为可逆神经网络基础配准（INNReg），通过图像到图像的翻译网络将多模态图像转换为单模态图像，并使用注册网络对其进行对齐。

**💡 创新点**

创新点在于利用可逆神经网络（INN）确保图像翻译后的几何一致性，并引入了一种新的障碍损失函数，基于归一化互信息（NMI）来提高配准精度。

**🔧 技术方法**

使用了可逆神经网络（INN）和动态深度卷积（DDWConv）结合局部注意机制来增强特征提取。

**📊 数据集**

使用了两个公共的多模态医学图像数据集，包括MRI T1/T2和MRI/CT图像对。

**📈 对比分析**

与四种现有的多模态图像配准方法进行了比较，INNReg在SSIM、NCC和Dice等指标上表现优异，且在HD95和光滑度方面也表现良好。

**⚠️ 局限性**

限制在于该方法仍然依赖于数据集的质量和多样性，可能在某些特定情况下表现不佳。

---

## 152. A Neural Envelope Method for Rapid Gradient-based Optimum Takeoff Trajectory Design

**arXiv ID:** 2611.23130 | [PDF](https://arxiv.org/pdf/2611.23130v1)

**作者:** Samuel Sisk `[一作]` (Missouri University of Science and Technology), Xiaosong Du `[通讯]` (Missouri University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种结合物理约束生成对抗网络（physicsGAN）和神经包络（neural envelope）的方法，以将原始的受约束非凸优化转化为无约束凸优化，从而优化电动垂直起降（eVTOL）飞机的起飞轨迹。

**💡 创新点**

创新点在于通过物理约束生成对抗网络发现可行设计空间，并利用神经包络构建目标函数的凸包，从而实现高效的梯度优化，解决了传统优化方法中的多模态问题。

**🔧 技术方法**

使用了物理约束生成对抗网络（physicsGAN）和神经网络构建的神经包络（neural envelope）技术。

**📊 数据集**

使用了Airbus A^3 Vahana eVTOL飞机的起飞轨迹优化数据集。

**📈 对比分析**

与传统的基于仿真的多学科分析与优化（MDAO）和基于双生成对抗网络（twinGAN）的优化方法相比，神经包络使得优化速度提高了53倍到162倍，同时在全局最优设计上达到了99.84%的准确率。

**⚠️ 局限性**

限制在于神经包络可能无法严格保证全局最优解的存在，未来的工作将探讨如何扩展到更高维度的输入空间，并处理更多约束条件。

---

## 153. Joint-ID: Transformer-based Joint Image Enhancement and Depth Estimation for Underwater Environments

**arXiv ID:** 2611.22212 | [PDF](https://arxiv.org/pdf/2611.22212v1)

**作者:** Geonmo Yang `[一作]` (Inha University), Younggun Cho `[通讯]` (Inha University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新颖的方法，通过联合学习同时增强水下图像和深度估计，利用变换器基础的神经网络从原始水下图像中获取高感知图像质量和深度信息。

**💡 创新点**

创新点在于设计了一种联合学习架构，能够在共享的层次特征空间中进行联合恢复和深度估计，同时引入了一个合成数据集以支持多任务学习。

**🔧 技术方法**

使用了变换器基础的U-Net架构，并设计了一个名为Mix Transformer++的高效编码器，结合了局部和全局特征的提取。

**📊 数据集**

构建了一个包含多种失真和场景深度的合成数据集，共计48000个样本，用于训练和测试。

**📈 对比分析**

通过与其他模型（如Berman、UWCNN等）进行比较，展示了在图像增强和深度估计任务上的SOTA性能，定量评估显示在PSNR、SSIM等指标上均优于其他方法。

**⚠️ 局限性**

限制在于模型的训练依赖于合成数据集的质量和数量，可能在真实环境中表现不如预期，且对不同水域类型的适应性仍需进一步验证。

---

## 154. NVIDIA Kumo Tabular: Pushing the Accuracy$\unicode{8211}$Efficiency Frontier for Tabular Prediction

**arXiv ID:** 2611.23001 | [PDF](https://arxiv.org/pdf/2611.23001v1)

**作者:** Jingang Qu `[一作]` (NVIDIA), Matthias Fey `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的表格基础模型（TFM），能够在没有额外训练或调优的情况下对新表格进行预测。

**💡 创新点**

该模型结合了前沿性能与计算效率，能够处理更大规模的表格数据，并在多个基准测试中表现优异。

**🔧 技术方法**

使用了改进的表格编码器和上下文学习（ICL）变换器架构，采用了查询-键归一化和其他优化技术。

**📊 数据集**

使用了合成数据生成的表格数据集，涵盖了多种真实世界的表格数据特征。

**📈 对比分析**

在TabArena和BeyondArena等多个基准测试中，该模型在准确性和速度上均优于其他调优和集成的模型，且在所有八个基准中排名第一。

**⚠️ 局限性**

模型在处理超过100K训练行的表格时表现不佳，且在分组和时间依赖性数据上泛化能力有限。

---

## 155. Weight-Space Programming Doesn't Need Transformers

**arXiv ID:** 2611.22055 | [PDF](https://arxiv.org/pdf/2611.22055v1)

**作者:** Hyunjong Ok `[一作]` (Pohang University of Science and Technology), Jaeho Lee `[通讯]` (Pohang University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了在权重空间编程中，是否应该继续使用Transformer架构，并提出了一种基于关联记忆的模型，该模型在参数数量和速度上均优于传统的Transformer。

**💡 创新点**

创新点在于提出了一种新的模型结构，允许大型语言模型（LLM）直接编写权重，而不依赖于梯度下降训练，从而实现了更高的准确性和更低的参数需求。

**🔧 技术方法**

使用了基于关联记忆的模型技术，该模型通过稀疏表来存储和查询信息，避免了传统Transformer的密集权重问题。

**📊 数据集**

使用了ARC-Easy和ARC-Challenge测试集，共计3530个科学问题，评估模型的性能。

**📈 对比分析**

与固定架构的模型相比，LLM设计的模型在ARC测试中的准确性更高，且参数数量减少了约1800倍。基于关联记忆的模型在GPU上比同大小的Transformer快17倍，且准确性保持在LLM设计模型的3个百分点以内。

**⚠️ 局限性**

限制在于这些模型仍然依赖于LLM编写的事实，无法进行跨步骤的推理或生成长答案，且在某些任务上表现不佳。

---

## 156. Impact of Pretraining Length, Model Size, and Annotation Budget on Downstream Clinical Tasks of Foundation Models in Radiotherapy

**arXiv ID:** 2611.22896 | [PDF](https://arxiv.org/pdf/2611.22896v1)

**作者:** Alejandro Cortina Uribe `[一作]` (University of Copenhagen), Jens Petersen `[通讯]` (University of Copenhagen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本研究探讨了预训练长度、模型大小和注释预算对放射治疗基础模型在临床任务中的下游性能的影响。

**💡 创新点**

创新点在于评估了不同的预训练选择和微调策略如何影响临床任务的下游性能，并提出了有效的预训练性能跟踪方法。

**🔧 技术方法**

使用了自监督学习（SSL）和掩蔽图像建模（SimMIM）技术，预训练了视觉变换器（ViT）模型。

**📊 数据集**

使用了约150,000个来自放射治疗的原始3D锥束计算机断层扫描（CBCT）图像进行预训练，并在1,562名头颈癌患者的51,135个CBCT图像上进行了微调。

**📈 对比分析**

通过比较低秩适应（LoRA）和完全微调的方法，发现低秩适应在参数数量上显著减少，但性能与完全微调相当。预训练模型在下游任务中的表现普遍优于未进行预训练的模型。

**⚠️ 局限性**

限制在于模型大小对下游性能的影响较小且不一致，表明较小的模型在计算资源有限的情况下仍能表现良好。

---

## 157. ALERT: Extending Humanoid Motion Trackers with a General and Lightweight Human Avoidance Pipeline

**arXiv ID:** 2611.22258 | [PDF](https://arxiv.org/pdf/2611.22258v1)

**作者:** Zhaoyang Li `[一作]` (Institute of Automation), Ziyu Chen `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为ALERT的轻量级管道，旨在将运动跟踪扩展到共享的人机环境中，集成了人类感知和主动避让能力。

**💡 创新点**

ALERT通过轻量级适应，平衡了避让需求与参考运动跟踪，同时保留了现有的运动能力，解决了现有跟踪策略缺乏人类意识和主动避让能力的问题。

**🔧 技术方法**

使用了基于LiDAR的人类包络、潜在流和条件低秩适应（LoRA）等技术。

**📊 数据集**

使用了500个运动序列的数据集，这些序列涉及到人类的周围空间，并在多个主流跟踪框架上进行了实验。

**📈 对比分析**

与CSF和ConstrainedMimic相比，ALERT在避免碰撞和恢复跟踪方面表现更好，完成更多动作且跟踪误差更低。

**⚠️ 局限性**

限制在于凸包无法表示人类肢体之间的间隙，可能导致近距离避让过于保守。

---

## 158. FrameSeg3D: Learned Local Frames for Segmenting Small Parts on High-Resolution 3D Scans

**arXiv ID:** 2611.22124 | [PDF](https://arxiv.org/pdf/2611.22124v1)

**作者:** Xinwen Zhang `[一作]`, Huawei Lin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为FrameSeg3D的方法，用于在高分辨率3D扫描中对小部件进行分割，首先预测每个部件的局部框架，然后在该框架周围的视图中进行分割。

**💡 创新点**

创新点在于通过预测每个部件的局部框架，使得分割过程能够在扫描仪的分辨率下进行，从而提高了小部件的分割精度。

**🔧 技术方法**

使用了深度学习技术，具体包括U-Net架构的网络来进行框架预测和分割。

**📊 数据集**

使用了两个数据集：一个是包含95个手部扫描的指甲数据集，另一个是Teeth3DS+的牙齿数据集。

**📈 对比分析**

与现有的全扫描方法和两阶段方法进行比较，FrameSeg3D在指甲分割上达到了0.970的平均Dice系数，在Teeth3DS+测试集上达到了96.5的每颗牙齿IoU，表现优于其他方法。

**⚠️ 局限性**

局限性在于框架网络需要更多的标签数据，且在训练中对稀有部件的识别能力较弱。此外，框架网络假设每个部件都朝向一个明确的方向，可能在相邻部件重叠的情况下表现不佳。

---

## 159. Event-Based Person Detection for Maritime UAV Search

**arXiv ID:** 2611.23088 | [PDF](https://arxiv.org/pdf/2611.23088v1)

**作者:** Zijie Ning `[一作]` (KU Leuven), Patrick Vandewalle `[通讯]` (KU Leuven)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究探讨了在海洋无人机搜索中使用事件相机进行人员检测，生成合成事件并与传统图像检测方法进行比较。

**💡 创新点**

创新点在于利用事件相机减少运动模糊，提高在低光照条件下的人员检测能力，并在嵌入式平台上评估其处理性能。

**🔧 技术方法**

使用了递归视觉变换器模型（RVT-T）、YOLOv8n和RT-DETR-R18进行人员检测。

**📊 数据集**

使用了公共的MOBDrone救援视频数据集，该数据集包含66个视频片段和现有的对象注释。

**📈 对比分析**

与YOLOv8n和RT-DETR-R18相比，RVT-T在843个测试样本中实现了15.63%的人员AP_50:95，而YOLOv8n和RT-DETR-R18分别为6.89%和9.44%。在100毫秒的合成曝光下，RVT-T的性能显著优于其他模型。

**⚠️ 局限性**

限制在于RVT-T模型的处理成本较高，每个样本需要约136毫秒和2.02焦耳的能量，尽管其检测性能优越，但在资源受限的环境中可能面临挑战。

---

## 160. DNeo: Your Parallel Draft Model Needs Better Target Feature Encoding

**arXiv ID:** 2611.21925 | [PDF](https://arxiv.org/pdf/2611.21925v1)

**作者:** Jameson Sandler `[一作]`, Ferdinando Fioretto `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的并行草拟架构DNeo，该架构通过结构保留特征和先前被拒绝提案的特征来条件化草拟器，并使用融合的因果头来修正其并行预测。

**💡 创新点**

创新点在于引入了Mixture-of-Features Attention (MoFA)，解决了标准特征编码的残余混合问题，并通过拒绝记忆利用先前被拒绝的提案特征，从而提高了草拟质量和生成速度。

**🔧 技术方法**

使用了Mixture-of-Features Attention (MoFA)、拒绝记忆、2-tap因果卷积、前缀仅交叉注意力和非均匀目标层选择等技术。

**📊 数据集**

在SGLang数据集上进行了广泛实验，使用了Qwen3-4B模型进行训练和评估。

**📈 对比分析**

与现有的并行草拟器（如DFlash-2）相比，DNeo在多个基准测试中表现出更快的生成速度（高达1.3倍），并且在草拟质量上也优于现有的最先进方法。

**⚠️ 局限性**

限制在于拒绝记忆引入了序列依赖性，草拟器必须在训练时进行规范化，以避免训练和测试之间的不匹配。

---

## 161. Trustworthy AI Teammates for Software Engineering: A Research Agenda for Lifecycle Context, Explainability, and Agentic Collaboration

**arXiv ID:** 2611.21624 | [PDF](https://arxiv.org/pdf/2611.21624v1)

**作者:** Glaucia Melo `[一作]` (Toronto Metropolitan University), Glaucia Melo `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 128 | [OpenAlex ID](https://openalex.org/A5039289770)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个研究议程，旨在将AI编码助手转变为可信赖的AI团队成员，以支持软件工程的生命周期、可解释性和代理协作。

**💡 创新点**

创新点在于提出了三个核心要求：生命周期感知的上下文、原则性可解释性和代理协作，并建立了一个评估框架，强调协作质量而非单一功能正确性。

**🔧 技术方法**

使用了大型语言模型（LLMs）和多代理系统的概念，强调AI在软件开发生命周期中的参与和透明度。

**📊 数据集**

未具体提及使用的数据集，但强调了需要考虑软件开发中的各种文档和上下文信息，如需求、问题报告、测试和提交历史等。

**📈 对比分析**

与现有的AI助手相比，提出的框架关注协作质量的评估，包括上下文对齐、信任校准、维护恢复成本和可追溯性，强调了AI生成的建议在项目上下文中的适用性。

**⚠️ 局限性**

限制在于当前AI系统仍然主要依赖于局部上下文，缺乏对生命周期的全面理解，且在解释和责任方面存在不足，可能导致开发者对AI的过度信任或不信任。

---

## 162. Haptic Probing for Subsurface-Aware Force Distribution on Layered Soft Tissue

**arXiv ID:** 2611.22195 | [PDF](https://arxiv.org/pdf/2611.22195v1)

**作者:** Lei Shi `[一作]`, Xiong Li `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本论文提出了一种新的触觉探测方法，用于在分层软组织上实现子表面感知的力分布控制。通过在施加任务力之前，利用低幅度的激励信号来区分肌肉和骨骼组织，从而消除控制延迟。

**💡 创新点**

创新点在于使用主动触觉探测来在任务加载之前识别组织的子表面状态，并将其用于力分布控制。这种方法使得压力分布成为一个可控变量，而不是任务执行的副产品。

**🔧 技术方法**

使用了主动触觉探测技术，通过低幅度激励信号测量局部动态导纳，并结合力和触觉反馈进行控制。

**📊 数据集**

实验中使用了三名人类受试者的肩部软组织数据，进行指尖按压和揉捏操作。

**📈 对比分析**

与基线方法（单一固定顺应性）和无知觉反应控制方法相比，提出的方法在力跟踪和压力均匀性方面表现更优。具体而言，提出的方法在力跟踪误差上减少了45%，并且在压力均匀性上也有显著改善。

**⚠️ 局限性**

限制在于边界检测仍然是反应性的，因此每个新状态在加载之前会暂时使用先前的参数。此外，预算不再补充，这限制了每次遍历的招募次数。

---

## 163. We Fixed the Scorer First, and Cluster Correction Killed Our Own Planted-Defect Result in AI Code Review

**arXiv ID:** 2611.21635 | [PDF](https://arxiv.org/pdf/2611.21635v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 164. Efficient Nominal Automata Computation with Strong Nominal Sets

**arXiv ID:** 2611.22327 | [PDF](https://arxiv.org/pdf/2611.22327v1)

**作者:** Hiroya Fujinami `[一作]` `[通讯]` (National Institute of Informatics), Hiroya Fujinami (National Institute of Informatics)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了名义自动机和寄存器自动机在无限字母表上的等价性，提出了一种新的算法来解决它们之间的算法差距，特别是在语言等价性、最小化和主动学习方面。

**💡 创新点**

创新点在于通过强名义集合的分类特征，提出了一种无约束的数组表示法，并证明了每个名义集合都可以获得强近似，从而在多项式时间内解决语言等价性问题。

**🔧 技术方法**

使用了强名义集合的分类特征、无约束数组表示法以及项目覆盖的概念，开发了高效的算法来处理语言等价性、最小化和主动学习。

**📊 数据集**

使用了名义自动机的配置空间作为数据集，特别是寄存器自动机的配置空间，这些空间对应于强名义集合。

**📈 对比分析**

与现有的名义计算工具相比，本文的方法在性能上表现优越，尤其是在处理小实例时，保持在一个小的常数因子内，且在多项式时间内解决了语言等价性问题。

**⚠️ 局限性**

限制在于强名义集合在商的情况下并不封闭，因此在处理某些类型的等价关系时可能会遇到困难。

---

## 165. Long-Context Generation Is a Sampling Problem (When You Turn Up the Heat)

**arXiv ID:** 2611.22148 | [PDF](https://arxiv.org/pdf/2611.22148v1)

**作者:** Allen G. Roush `[一作]` (Thoughtworks), Sanjay Basu `[通讯]` (Oracle)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文探讨了开放权重语言模型在生成长文本时的性能下降问题，并提出通过调整采样器来改善生成质量，而不是依赖于重新训练模型。

**💡 创新点**

创新点在于提出了一种新的采样策略（top-nσ），该策略在高温度下保持候选令牌池的稳定性，从而显著提高了长文本生成的质量。

**🔧 技术方法**

使用了多种采样技术，包括top-p、top-k、Top-H、min-p和top-nσ，并比较了它们在不同温度下的表现。

**📊 数据集**

实验使用了29个开放权重模型，参数范围从1B到122B，并在多个基准测试上进行评估，包括SWE-bench、LiveCodeBench和LongBench等。

**📈 对比分析**

通过与其他采样方法的比较，top-nσ在高温度下保持了较高的生成质量，而其他方法如top-p和top-k在高温度下性能显著下降，top-nσ在多个基准测试中表现优异。

**⚠️ 局限性**

限制在于该研究主要集中在采样器的调整上，可能未能考虑其他潜在影响生成质量的因素，如模型架构或训练数据的多样性。

---

## 166. PDFa11yMut: Mutation-Based Evaluation of Automated PDF Accessibility Validators

**arXiv ID:** 2611.21628 | [PDF](https://arxiv.org/pdf/2611.21628v1)

**作者:** Gauri Jain `[一作]` `[通讯]` (University of California), Gauri Jain (University of California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为PDFa11yMut的基于突变的基准，用于评估自动PDF可访问性验证器，针对独立验证的PDF结构缺陷进行测试。

**💡 创新点**

创新点在于引入了一种受控的突变测试方法，能够量化自动验证器的检测边界，并提供了针对PDF可访问性结构缺陷的五种突变操作符。

**🔧 技术方法**

使用了突变测试技术，生成了23个经过独立验证的突变体，并评估了三个验证器（PAC、Adobe Acrobat Accessibility Checker和veraPDF）的检测能力。

**📊 数据集**

使用了五个验证过的黄金PDF作为数据集，从中生成突变体进行测试。

**📈 对比分析**

与其他方法比较时，PAC检测到4个突变（17.4%），Adobe Acrobat和veraPDF各检测到5个突变（21.7%）。所有验证器均检测到每个生成的标题层次突变，但未能检测到阅读顺序交换、遗漏结构或内部MCID顺序的突变。

**⚠️ 局限性**

限制在于突变操作符仅针对结构级缺陷，未涵盖所有PDF可访问性问题，且实验结果依赖于所使用的验证器版本和配置。

---

## 167. Encoding, Decodability, and Use: A Five-Layer Framework for Benchmark Measurement Audits with Evidence from UAV Intent Recognition

**arXiv ID:** 2611.22434 | [PDF](https://arxiv.org/pdf/2611.22434v1)

**作者:** Zhirong Su `[一作]` (Shandong Management University), Tian Zhang `[通讯]` (Shandong Management University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实例化了一个五层框架，用于审计基准分数的生成过程，应用于AeroIntent无人机意图基准。

**💡 创新点**

创新点在于将基准分数的理解分解为观察、表示、解码、使用和行为五个层次，并明确每个层次的失败模式和审计方法。

**🔧 技术方法**

使用了线性探测、PCA+岭回归等技术进行特征提取和分析。

**📊 数据集**

使用了AeroIntent数据集，该数据集包含176个序列和22个无人机轨迹程序。

**📈 对比分析**

通过与其他方法的比较，发现没有多特征读出超过最小可检测效应（MDE），且观察到的相关性r=+0.080，远低于MDE的0.599，表明模型的性能有限。

**⚠️ 局限性**

限制在于审计仅覆盖线性读出和关联控制，缺乏非线性读出和干预证据，且未能在多个基准上进行交叉验证。

---

## 168. PVTSizing 2.0: A Fast and Reliable Analog Circuit Sizing Operator for Agentic EDA Workflows

**arXiv ID:** 2611.22328 | [PDF](https://arxiv.org/pdf/2611.22328v1)

**作者:** Zichen Kong `[一作]` (Peking University), Yuan Wang `[通讯]` (Peking University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的模拟电路尺寸优化工具PVTSizing 2.0，旨在提高多PVT（过程、电压、温度）条件下的电路设计效率。

**💡 创新点**

创新点在于结合了人机协作的工作流程和高效的数值优化器，能够在多个PVT角落中快速找到满足规格的电路尺寸，同时引入了基于大语言模型（LLM）的知识注入机制以进一步提升优化效率。

**🔧 技术方法**

使用了基于损失引导的训练和评论引导的局部优化技术，结合了大语言模型的知识注入机制。

**📊 数据集**

使用了四个真实世界的模拟电路基准，包括折叠级联运算跨导放大器（FC-OTA）、带隙参考（BGR）、强ARM比较器（CMP）和多级OTA（MS-OTA），这些电路涵盖了不同的设计变量和约束条件。

**📈 对比分析**

与原始PVTSizing、RobustAnalog和Virtuoso中的五种优化器进行了比较，PVTSizing 2.0在所有40次运行中均成功完成，整体平均运行时间比原始PVTSizing快7.3倍，同时保持了相似的样本数量。最佳的知识注入配置进一步减少了39.9%的平均运行时间和35.4%的模拟样本。

**⚠️ 局限性**

限制在于知识注入的效果依赖于LLM的推理能力，且在实际应用中，预算需要覆盖模拟、代理模型训练和LLM推理的成本。

---

## 169. WoRD: Diffusion-Based Posterior Inference for Online Goal Recognition

**arXiv ID:** 2611.23192 | [PDF](https://arxiv.org/pdf/2611.23192v1)

**作者:** Almog Anschel `[一作]` (Technion Israel Institute of Technology), Sarah Keren `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的目标识别方法，利用学习到的轨迹先验从部分观察中推断代理的目标。

**💡 创新点**

创新点在于将目标识别视为扩散推断，通过独立的每个目标的去噪链进行后验推断，并通过贝叶斯重加权组合结果，提供了一种数据驱动的方法来处理复杂环境中的目标识别。

**🔧 技术方法**

使用了扩散模型和贝叶斯推断技术，结合了观察引导的闭式更新机制。

**📊 数据集**

使用了模拟机器人设置、行人轨迹和车辆轨迹的数据集进行评估。

**📈 对比分析**

与五个基线方法进行了比较，包括粒子滤波器和分类器，结果显示该方法在所有评估指标上均表现优异，尤其在处理模糊前缀时保持了良好的后验信念。

**⚠️ 局限性**

限制在于候选目标集必须事先指定，无法在推断时发现新的目标；测量模型必须已知且可微分；在硬件受限的平台上，推断延迟与自信错误预测之间的权衡可能影响应用。

---

## 170. Plenty to FRET About: From Hazard Analysis to Formal Requirements for Robot-Assisted Dressing

**arXiv ID:** 2611.22399 | [PDF](https://arxiv.org/pdf/2611.22399v1)

**作者:** Gunjari Chakraborty `[一作]` (University of Manchester), Marie Farrell `[通讯]` (University of Manchester)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对146个基于危险的自然语言需求进行了形式化，针对一个机器人辅助穿衣（RAD）案例研究，最终形成了117个独特的需求。

**💡 创新点**

创新点在于通过FRET工具将自然语言需求转化为形式化需求，揭示了需求中的模糊性、时间和概率假设，并分析了需求之间的关系。

**🔧 技术方法**

使用了NASA的形式需求引导工具（FRET），该工具支持将自然语言需求转化为线性时间时序逻辑（LTL）和概率计算树逻辑（PCTL*）语义。

**📊 数据集**

数据集来源于机器人辅助穿衣（RAD）系统的146个基于危险的自然语言需求，这些需求经过分类和形式化处理。

**📈 对比分析**

通过与HRT分类法的比较，形式化需求提供了补充视角，显示出需求之间的关系和相互影响，性能分析表明形式化过程有效地减少了模糊性并明确了行为要求。

**⚠️ 局限性**

限制在于FRET工具无法自动确定时间约束的正确性，且形式化过程中可能引入不一致性和错误，特别是在初始配置的假设方面存在不足。

---

## 171. How well does graph superoptimization transfer? Adapting tensor-graph superoptimizer to a Triton-based NPU

**arXiv ID:** 2611.22844 | [PDF](https://arxiv.org/pdf/2611.22844v1)

**作者:** Weronika Hryniewska-Guzik `[一作]` (Huawei Technologies), Tadeusz Kobus `[通讯]` (Huawei Technologies)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文研究了图级超优化在不同架构之间的迁移能力，特别是将Mirage从CUDA设计适配到华为Ascend 910B3，通过基于Triton的编译栈进行评估。

**💡 创新点**

创新点在于适配了图级张量超优化器到非CUDA加速器，并探讨了功能可移植性与性能可移植性之间的差距，识别了张量分区、执行资源建模等关键限制。

**🔧 技术方法**

使用了Triton编译框架对Mirage进行适配，结合了图形转换、调度优化和自定义内核生成等技术。

**📊 数据集**

使用了与变压器相关的工作负载，包括矩阵乘法、softmax、RMSNorm+MatMul、门控MLP、FNN和注意力机制等。

**📈 对比分析**

通过与手写的Triton内核进行比较，适配后的超优化器在某些配置上表现出竞争力，性能提升可达87.0%。然而，性能高度依赖于张量形状和工作负载结构，某些配置下出现显著回归。

**⚠️ 局限性**

限制在于目标特定的张量分区、执行资源建模、后端降级和融合、搜索与验证成本等方面，导致性能可移植性不足。

---

## 172. Path Abstraction for Markov Reward Models: An Isabelle Formalisation in a Free Monoid View

**arXiv ID:** 2611.22523 | [PDF](https://arxiv.org/pdf/2611.22523v1)

**作者:** Arnd Hartmanns `[一作]` (University of Twente), Robert Modderman `[通讯]` (University of Twente)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文扩展了自由单子视角下的路径抽象算法，以适应马尔可夫奖励模型（MRMs），并在Isabelle/HOL中对马尔可夫链和奖励模型进行了形式化。

**💡 创新点**

创新点在于将路径抽象算法推广到MRMs，并在自由单子视角下提供了对期望累积奖励的保留，同时实现了算法的机器检查正确性证明。

**🔧 技术方法**

使用了Isabelle/HOL进行形式化和机械化证明，利用自由单子语义简化了证明过程，并结合了HOL-Analysis库处理无限和有限路径的概率。

**📊 数据集**

使用了马尔可夫奖励模型（MRMs）和离散时间马尔可夫链（DTMCs）作为数据集，特别关注路径抽象算法的实现和验证。

**📈 对比分析**

通过与传统的马尔可夫模型形式化进行比较，证明了路径抽象在保持可达性概率和期望奖励方面的有效性，且在自由单子视角下的证明过程更为简洁。

**⚠️ 局限性**

限制在于虽然路径抽象算法在理论上得到了验证，但在实际应用中可能需要进一步的优化和效率提升，以便在复杂系统中有效使用。

---

## 173. Threat Landscape of Vibe Coding and the Road to Secure Autonomy

**arXiv ID:** 2611.21822 | [PDF](https://arxiv.org/pdf/2611.21822v1)

**作者:** Devki Nandan Jha `[一作]` (Newcastle University), Suhas Nale `[通讯]` (Newcastle University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文调查了与vibe编码相关的安全性，分析了从开发者提示到模型决策、工具使用、部署和生产的整个工作流程中的风险。

**💡 创新点**

创新点在于提出了一个统一的威胁模型和九类威胁分类，系统性地映射了vibe编码系统中的安全风险，并识别了21个研究空白和11个主要挑战。

**🔧 技术方法**

使用了结构化文献回顾的方法，结合了对现有文献的分析，提出了威胁分类和防御机制的评估。

**📊 数据集**

数据集包括从2021年到2026年间在ACM Digital Library、IEEE Xplore、USENIX、Scopus和Google Scholar等数据库中找到的142篇相关研究论文。

**📈 对比分析**

与现有方法的比较显示，当前的防御机制大多针对单一威胁，缺乏对复合威胁的系统性评估，性能上存在显著的不足，尤其是在高自主性水平下的安全性。

**⚠️ 局限性**

限制在于现有的防御措施往往假设助手仅提供建议，而未考虑到AI代理的自主性和复杂性，导致对安全风险的理解和应对不足。

---

## 174. Lyapunov-Inspired Sustained Balance on Dynamically Responsive Supports

**arXiv ID:** 2611.22831 | [PDF](https://arxiv.org/pdf/2611.22831v1)

**作者:** Changyan Li `[一作]` (University of Hong Kong), Shiyan Hu `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种新的平衡控制方法，称为LISST（Lyapunov-Inspired Sustained Stabilization），用于在动态响应支撑上维持平衡。

**💡 创新点**

创新点在于将平衡维持视为一个耦合调节问题，协调平衡恢复与抑制由机器人自身修正引起的运动，并引入了基于历史的学习框架来指导修正动作。

**🔧 技术方法**

使用了基于Lyapunov的方法和物理监督的学习框架，结合了短期和长期的运动历史信息来优化控制策略。

**📊 数据集**

在Unitree Go2 Edu四足机器人上进行实验，使用了一个被动的、铰接的支撑板，评估了机器人在不同负载下的平衡能力。

**📈 对比分析**

通过与传统方法的比较，LISST在维持平衡的同时有效限制了残余振荡，展示了在动态交互中的优越性能。

**⚠️ 局限性**

限制在于该方法依赖于机器人自身的本体感知，未使用外部负载或支撑状态的测量，可能在某些复杂环境中表现不佳。

---

## 175. Readie: Low-Latency Code Sandboxes for AI Agents

**arXiv ID:** 2611.22171 | [PDF](https://arxiv.org/pdf/2611.22171v1)

**作者:** Arjun Sivaraman `[一作]`, Yongjoo Park `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文提出了一种新的系统框架，用于优化复杂计算任务的执行效率。

**💡 创新点**

创新点在于引入了一种新的调度算法，能够动态调整资源分配，以适应不同的计算需求。

**🔧 技术方法**

使用了机器学习技术来预测任务的资源需求，并结合传统的调度算法进行优化。

**📊 数据集**

实验中使用了多个公开数据集，包括标准的计算任务基准测试数据集。

**📈 对比分析**

与现有的调度方法进行了比较，结果显示新方法在任务完成时间和资源利用率上均有显著提升。

**⚠️ 局限性**

该研究的局限性在于算法在特定类型的任务上表现较好，但在其他类型的任务上可能效果不佳。

---

## 176. Technical Report of Mobile Manipulator Robot for Industrial Environments

**arXiv ID:** 2611.22852 | [PDF](https://arxiv.org/pdf/2611.22852v1)

**作者:** Aria Dadnavi `[一作]` (Shahid Beheshti University), Mohammad Hossein Moaiyeri `[通讯]` (Shahid Beheshti University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文介绍了Auriga Robotics开发的移动操控机器人@Work Robot的设计与实现，该机器人旨在提高工业环境中重复或危险工作的效率。

**💡 创新点**

创新点在于其模块化设计和智能控制架构，结合了定制电子设备和基于视觉的动态障碍物映射框架，以改善复杂环境中的导航。

**🔧 技术方法**

使用了ESP32微控制器和NVIDIA Jetson Orin NX进行高层处理和实时系统控制，软件方面采用了Hector SLAM进行地图构建和定位，A*算法进行路径规划，以及YOLO进行实时物体检测。

**📊 数据集**

使用了自建的数据集，包括61285张训练图像和3879张验证图像，涵盖了多种环境条件下的物体检测和分割。

**📈 对比分析**

与其他方法相比，系统在自主导航、物体处理和任务执行方面表现出一致的性能，尤其是在复杂工业环境中的有效性。

**⚠️ 局限性**

限制在于系统的复杂性和对环境变化的敏感性，可能在动态环境中面临挑战。

---

## 177. stable-envs-v1: Fast and Reproducible Environments with Controllable Variations

**arXiv ID:** 2611.23151 | [PDF](https://arxiv.org/pdf/2611.23151v1)

**作者:** Sami BuGhanem `[一作]` (Brown University), Randall Balestriero `[通讯]` (Brown University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种开源库，用于高效的批量渲染模拟环境，以加速世界模型和基于像素的代理的研究。

**💡 创新点**

通过避免在CPU/GPU之间移动大型图像张量，以及批量渲染、自定义渲染内核和重用预渲染的静态场景，实现了高达1.33倍的训练速度提升。

**🔧 技术方法**

使用了GPU原生渲染技术，结合批量渲染和自定义GPU内核。

**📊 数据集**

支持80个模拟环境，能够在多个CPU和GPU后端上运行。

**📈 对比分析**

与现有方法相比，提供了高达8.5倍的渲染速度提升和高达1.33倍的训练速度提升，且在下游规划性能上没有下降。

**⚠️ 局限性**

尽管实现了显著的速度提升，但仍需确保在不同环境下的动态和视觉一致性，可能在某些复杂环境中存在性能瓶颈。

---

## 178. REVEAL: Reconstructing Video Amodal Content via Language and Flow Guidance

**arXiv ID:** 2611.22645 | [PDF](https://arxiv.org/pdf/2611.22645v1)

**作者:** Minh Tran `[一作]` (University of Arkansas), Ngan Le `[通讯]` (University of Arkansas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于扩散模型的框架，用于开放词汇的视频模态分割和内容补全，结合光流和语言先验生成可见和不可见的掩码及RGB内容。

**💡 创新点**

创新点在于引入光流作为运动先验和文本作为语义先验，解决了在重度遮挡情况下缺乏辅助指导的问题，并保持了长序列的时间一致性。

**🔧 技术方法**

使用了扩散模型，结合光流引导和文本引导的两阶段管道。

**📊 数据集**

使用了一个新的基准数据集，包含长视频序列和文本描述，支持重度遮挡下的文本引导视频模态补全评估。

**📈 对比分析**

与现有基准相比，展示了在视频模态补全、分割精度和时间一致性方面的最先进性能，尤其在重度遮挡和长视频序列中表现优异。

**⚠️ 局限性**

限制在于文本引导主要局限于类别级语义，无法有效解决动态属性的歧义，且光流先验依赖于前帧的可见性，无法处理持续遮挡的情况。

---

## 179. Hierarchical Diffusion Transformer for End-to-End Image Generation

**arXiv ID:** 2611.22962 | [PDF](https://arxiv.org/pdf/2611.22962v1)

**作者:** Xiaoyu Wu `[一作]` (Carnegie Mellon University), Steven Wu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的层次扩散变换器（HiT），通过在单一的端到端训练阶段中学习视觉层次结构，从而提高图像生成质量和训练效率。

**💡 创新点**

创新点在于HiT通过逐步分组相关的图像特征并用单个标记表示每个组，形成紧凑的组标记，从而在生成过程中学习视觉层次结构，而无需外部视觉预训练。

**🔧 技术方法**

使用了层次扩散变换器（HiT）架构，结合了学习路由器和软聚合技术，以便在更深的变换器瓶颈中处理紧凑的组标记。

**📊 数据集**

在ImageNet 256×256和CC12M数据集上进行了实验，评估了类条件生成和文本到图像生成的性能。

**📈 对比分析**

与JiT模型相比，HiT在ImageNet上达到了1.48的FID，使用了约40%的累积训练FLOPs，且在文本到图像生成中超越了UniLiP的平均得分，同时减少了6倍的去噪参数和3倍的前向FLOPs。

**⚠️ 局限性**

限制在于HiT的性能仍然依赖于模型架构的设计，可能在特定任务或数据集上表现不如经过专门预训练的模型。

---

## 180. BATMoS: Budget-Aware Guidance Allocation for Efficient Video Diffusion

**arXiv ID:** 2611.22028 | [PDF](https://arxiv.org/pdf/2611.22028v1)

**作者:** Zhixu Du `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为BATMoS（预算感知时间混合策略）的视频生成方法，通过选择性地重用指导来减少去噪分支评估，从而在保持生成质量的同时降低计算成本。

**💡 创新点**

BATMoS通过引入基于幅度的时间规则来优化计算预算分配，能够在使用20%更少的分支评估的情况下，接近全分类器自由指导的生成质量。

**🔧 技术方法**

使用了一种预训练的去噪器，并通过时间路由器在新鲜分类器自由指导和缓存指导之间进行选择。

**📊 数据集**

在CogVideoX-2B数据集上进行了评估，该数据集包含4720个视频，覆盖了所有16个官方VBench维度。

**📈 对比分析**

与其他分配策略相比，BATMoS在八种推理配置中获得了最高的总分76.97，接近全CFG的77.07分，同时减少了20%的分支评估。

**⚠️ 局限性**

BATMoS的局限性在于其在某些情况下可能无法完全替代全CFG的质量，尽管在大多数情况下表现良好，但仍存在一定的性能损失。

---

## 181. TPHC: Improving Hyper-Connections with Tensor-Parallel Grouped Updates

**arXiv ID:** 2611.23026 | [PDF](https://arxiv.org/pdf/2611.23026v1)

**作者:** Xinying Hu `[一作]` (ShanghaiTech University), Kewei Tu `[通讯]` (ShanghaiTech University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的模型，结合了张量并行分片和超连接，以提高Transformer模型的性能。

**💡 创新点**

通过将张量并行分片与超连接流结合，增强了流之间的独立性，从而提高了残差流的更新表达能力。

**🔧 技术方法**

使用了张量并行（TP）技术和超连接（HC）架构。

**📊 数据集**

在100B个标记上训练的1.1B模型。

**📈 对比分析**

与mHC相比，提出的模型在相同流数量下降低了验证损失0.7-1.0%，并在六个常识推理基准上提高了平均零-shot准确率0.3-0.7个百分点。

**⚠️ 局限性**

目前的实验仅限于1.1B参数模型和100B标记，未覆盖更大模型和更多训练数据。

---

## 182. World-Grounded Reinforcement Learning for Pollution-Aware Bird Movement Modeling Using a Multi-Instance DQN Framework

**arXiv ID:** 2611.21895 | [PDF](https://arxiv.org/pdf/2611.21895v1)

**作者:** Shivin Mangal `[一作]` (Vellore Institute of Technology), Isshaan Singh `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本论文提出了一种基于世界基础的强化学习框架，模拟鸟类在空气质量指数（AQI）变化下的移动，以研究鸟类如何响应空间变化的污染。

**💡 创新点**

创新点在于将实际环境数据作为反馈信号，而非依赖于合成环境，利用深度Q网络（DQN）训练多个代理以适应AQI变化。

**🔧 技术方法**

使用了深度Q网络（DQN）技术，结合了多实例强化学习和环境反馈机制。

**📊 数据集**

使用了来自Xeno-canto数据库的鸟类记录数据集，包含479个鸟类录音位置的经纬度和AQI值。

**📈 对比分析**

与蒙特卡洛、Q学习和价值迭代方法进行了比较，DQN在累积奖励、行动多样性、奖励变化率、步骤效率和行驶距离等方面表现最佳，累积奖励达到722。

**⚠️ 局限性**

限制在于生成的路径是对特定目标的模拟响应，而非迁徙的重建，AQI层是静态的，因此需要进一步的多种验证以得出生态结论。

---

## 183. What Does Weight-Space Programming Add to Diffusion Editing? Program Construction versus Score-Guided Refinement

**arXiv ID:** 2611.22228 | [PDF](https://arxiv.org/pdf/2611.22228v1)

**作者:** Zitian Gao `[一作]` (Fudan University), Liangliang Shi `[通讯]` (Shanghai Institute for Mathematics and Interdisciplinary Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究探讨了权重空间编程（WSP）在扩散模型编辑中的应用，特别是通过程序构建与基于评分的修订之间的比较。

**💡 创新点**

创新点在于提出了一种可执行的WSP管道，用于扩散概念编辑，并分析了生成的更新空间，强调程序构建与评分引导修订的不同效果。

**🔧 技术方法**

使用了权重空间编程（WSP）技术，结合了语言模型生成可执行的权重更新程序。

**📊 数据集**

使用了TIMED数据集，该数据集包含104个测试编辑，研究中随机抽取了24个进行比较。

**📈 对比分析**

与手写基线（U1b）和预算匹配的传统编辑方法进行比较，结果显示U2-Astra程序在新场景的联合成功率上提高了10.0个百分点，但在CLIP评分上没有显著变化。总体上，U3-Opus与U1b的比较显示没有优势。

**⚠️ 局限性**

限制在于程序构建与评分引导修订的效果不同，后者未能保留前者的任务收益，且研究仅限于Stable Diffusion 1.4模型和单轮重写。

---

## 184. Every Measurement, Every Direction, All at Once: Multimodal Flow Matching for Molecules and Spectra

**arXiv ID:** 2611.22098 | [PDF](https://arxiv.org/pdf/2611.22098v1)

**作者:** Thorben Prein `[一作]` (Lila Sciences), Santiago Miret `[通讯]` (Lila Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为MOSAIC的多模态流匹配模型，旨在通过随机掩蔽技术从单个实验中提取多个监督信号，实现分子结构与光谱（UV、IR、拉曼光谱）的任意生成。

**💡 创新点**

创新点在于无方向限制的多模态生成训练，使得每种模态都可以作为目标或条件，从而提高了模型的推理能力和生成质量。

**🔧 技术方法**

使用了多模态流匹配变换器（MMDiT）和UAE-3D自编码器，结合自监督学习和生成训练。

**📊 数据集**

使用了QM9S数据集，该数据集包含110,000个训练分子及其对应的分子结构和模拟的UV、IR、拉曼光谱。

**📈 对比分析**

与现有的单模态和双模态基线模型进行比较，MOSAIC在分子图恢复上达到了88.8%的准确率，超越了最强的光谱到结构基线23.2%，同时在结构到光谱生成上也表现优异。

**⚠️ 局限性**

限制在于模型的性能依赖于条件模态的选择，某些模态（如UV）在生成时可能导致较低的准确率，且模型在处理实验光谱时的泛化能力仍需进一步验证。

---

## 185. TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks

**arXiv ID:** 2611.22115 | [PDF](https://arxiv.org/pdf/2611.22115v1)

**作者:** Xingchen AGI Lab `[一作]` (China Telecom), Zihan Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文介绍了Xing 4系列大型语言模型，扩展了TeleChat系列，通过在Ascend上协调模型架构和训练基础设施的进步。

**💡 创新点**

创新点在于结合稀疏激活、注意力设计、模型深度和宽度，以及推测解码，以平衡模型容量与计算、内存访问和执行开销。

**🔧 技术方法**

使用了Ascend NPUs和MindSpore进行训练，采用了多种技术，如稀疏激活、深度稀疏注意力（DSA）、跨层索引共享（IndexShare）等。

**📊 数据集**

使用了285B参数的MoE模型进行预训练，训练在8192个Ascend 910C卡上进行，支持256K-token上下文。

**📈 对比分析**

与MLA-only基线相比，预填充吞吐量提高了4.68倍，推测解码的解码吞吐量提高了2.67到3.77倍，消除管道并行性带来了16.1%的吞吐量提升。

**⚠️ 局限性**

限制在于尽管模型架构和系统设计进行了优化，但在处理更大规模模型和更复杂的推理任务时，仍需进一步研究训练效率和系统可扩展性。

---

## 186. On Improving the Adoption, Usability, and Retention of Static Application Security Testing (SAST) Tools

**arXiv ID:** 2611.21621 | [PDF](https://arxiv.org/pdf/2611.21621v1)

**作者:** Zachary Douglas Wadhams `[一作]` `[通讯]` (Michigan State University), Zachary Douglas Wadhams (Michigan State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究探讨了开发人员在使用静态应用程序安全测试（SAST）工具时遇到的挑战，旨在提高其可用性。

**💡 创新点**

创新点在于提出了一种自动化执行SAST工具输出的过程，并将数据以对开发人员有意义和可操作的格式呈现。

**🔧 技术方法**

使用了静态应用程序安全测试（SAST）技术。

**📊 数据集**

研究中回顾了89篇与SAST实施和使用相关的文献。

**📈 对比分析**

通过文献回顾发现开发人员在使用SAST时面临的问题，如误报等，并通过实际案例反馈验证了所提出过程的有效性，开发人员对该过程的多个方面表示满意，愿意更多地使用SAST工具。

**⚠️ 局限性**

研究的局限性未明确提及，但可能包括对SAST工具的普遍问题的解决方案是否适用于所有开发环境的疑问。

---

## 187. TabDAG: Structured Evolutionary Search for Tabular Feature Engineering

**arXiv ID:** 2611.22828 | [PDF](https://arxiv.org/pdf/2611.22828v1)

**作者:** Anastasia Volkova `[一作]` (FusionBrain Lab), Valentin Khrulkov `[通讯]` (FusionBrain Lab)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了TabDAG框架，结合进化搜索与特征工程程序的图表示，旨在改善表格数据的预测性能。

**💡 创新点**

创新点在于使用有向无环图（DAG）表示特征工程程序，并通过结构化编辑和父特定模式约束变异，确保生成的程序有效性。

**🔧 技术方法**

使用了进化搜索和图结构的特征DAG，结合语言模型（LLM）进行特征提议。

**📊 数据集**

在10个回归和分类数据集上进行实验，涉及三种表格模型家族。

**📈 对比分析**

与OpenFE等方法比较，TabDAG和其组合在平均排名上表现最佳，进化特征在大多数情况下优于原始输入，且结构化编辑导致的失败评估更少。

**⚠️ 局限性**

主要限制在于评估范围仅限于十个数据集和三种估计器，缺乏更广泛的数据集和独立的搜索种子验证。

---

## 188. SLED: Sensor SLicing from Edge to Datacenter

**arXiv ID:** 2611.21937 | [PDF](https://arxiv.org/pdf/2611.21937v1)

**作者:** Alan Subedi `[一作]` (University of Nebraska--Lincoln), Mehmet C. Vuran `[通讯]` (University of Nebraska--Lincoln)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出了一种名为SLED的端到端框架，旨在通过传感器切片技术在边缘、云和高性能计算（HPC）资源之间分配无线、网络和计算资源。

**💡 创新点**

创新点在于引入了传感器切片（SL），为每个传感器流提供确定性的MAC级别隔离，并在O-RAN中实现了每个传感器的无线资源分配。

**🔧 技术方法**

使用了代理编码技术，并在两个符合O-RAN标准的软件包（OAI和OCUDU）中实现了传感器切片，采用了基于仿真的验证方法。

**📊 数据集**

使用了公开可用的SLED软件和评估数据集，具体数据集可在https://github.com/UNL-CPN-Lab/SLED-Sensor-SLicing-from-Edge-to-Datacenter获取。

**📈 对比分析**

与传统网络切片和ORANSlice进行比较，SLED在高流量负载下保持稳定的延迟，而传统方法的延迟增加超过2300倍。SLED在传感器切片数量增加时，吞吐量保持在预期分配的0.9%到1.4%之间。

**⚠️ 局限性**

限制在于当前实现主要集中在O-RAN环境中，可能在其他网络架构中面临适应性挑战。

---

## 189. Lost in Read-out: Locating the Image Quality Gap in Vision-Language Models with a Closed-Form Edit

**arXiv ID:** 2611.22564 | [PDF](https://arxiv.org/pdf/2611.22564v1)

**作者:** Zhuangzi Li `[一作]` (Nanyang Technological University), Weisi Lin `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文探讨了视觉语言模型（VLMs）在图像质量评估中的表现，提出这些模型在识别质量方面存在问题，主要是因为它们在最终输出阶段未能准确表达质量。

**💡 创新点**

创新点在于通过重写最后一层神经元的输出，显著提高了模型在图像质量评估中的表现，同时保持了感知问题回答的准确性。

**🔧 技术方法**

使用了线性探测器和闭式形式的神经元重写技术，进行模型编辑以改善输出。

**📊 数据集**

使用了KonIQ-10k数据集及其他多个真实和合成数据集进行测试。

**📈 对比分析**

与其他方法相比，模型编辑（ME）在多个数据集上表现出色，尤其是在KonIQ测试集上，提升了0.10到0.22的SRCC分数，而传统的微调方法未能保持感知问题回答的准确性。

**⚠️ 局限性**

限制在于编辑后的模型在处理与图像质量无关的文本时，可能会影响“好”和“差”的输出，并且需要针对不同类型的失真进行重新调整。

---

## 190. AfriNORM: An Evaluation Benchmark for Normative Reasoning of LLMs Across African Cultures

**arXiv ID:** 2611.22580 | [PDF](https://arxiv.org/pdf/2611.22580v1)

**作者:** Ahmad Samuel Gali `[一作]` (Independent Researcher), Shamsuddeen Hassan Muhammad `[通讯]` (Bayero University Kano)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于场景的多项选择题数据集，旨在评估大型语言模型（LLMs）在非洲社会文化背景下的规范推理能力。

**💡 创新点**

创新点在于引入了两个新的评估指标：通用常识偏见分数（UCBS）和干扰偏见分数（DBS），用于量化LLMs对非洲文化规范的理解和偏见。

**🔧 技术方法**

使用了基于场景的问题生成方法，并结合对抗性干扰选项来评估模型的判断能力。

**📊 数据集**

使用了一个包含65个关于约鲁巴文化规范的问题的数据集。

**📈 对比分析**

与现有的LLMs（如Gemini 2.0 Flash和GPT-5）进行比较，发现它们在英语中的准确率较高，但在约鲁巴语中的表现显著下降，表明模型在约鲁巴文化背景下存在语义混淆。

**⚠️ 局限性**

限制在于目前的实证验证仅限于约鲁巴语言，且数据集规模较小，可能影响偏见量化的统计能力。

---

## 191. COAST: Safe Flight Corridor Planner for Fast and Robust Snap Minimized Trajectories

**arXiv ID:** 2611.22447 | [PDF](https://arxiv.org/pdf/2611.22447v1)

**作者:** Alejandro Pelaez `[一作]` (Brigham Young University), Randal W. Beard `[通讯]` (Brigham Young University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种名为COAST的B样条轨迹规划器，能够在保证安全飞行走廊(SFC)的前提下，生成最小抖动的轨迹。

**💡 创新点**

COAST通过将问题分为两个阶段，结合A*算法和二次规划优化，提供了严格的安全保证和实时性能，解决了现有方法在计算速度和安全性之间的权衡。

**🔧 技术方法**

使用了B样条表示法、A*算法和二次规划技术，结合MINVO变换来优化控制点。

**📊 数据集**

使用了随机森林环境中的障碍物数据集进行高保真模拟实验，测试了在不同障碍物密度下的轨迹规划性能。

**📈 对比分析**

与传统的无约束最小抖动解析求解器相比，COAST通过将轨迹生成形式化为约束二次规划问题，确保了几何安全性和动态可行性，表现出在实时部署中平均计算时间低于5毫秒。

**⚠️ 局限性**

当前框架存在一些局限性，包括偶尔的飞行停滞和由于控制点过度分配导致的保守速度配置，未来工作将集中在解决这些问题上。

---

## 192. A Hybrid Exact-Heuristic Solver for the Integrated Passenger and Parcel Transportation Problem

**arXiv ID:** 2611.22043 | [PDF](https://arxiv.org/pdf/2611.22043v1)

**作者:** The Dung Bui `[一作]`, Ha Thu Nguyen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文描述了提交给2026年SOICT比赛第二轮的求解器，针对集成乘客和包裹运输问题，提出了一种选择性接送和送达的路径规划方法。

**💡 创新点**

创新点在于结合了常数时间的插入测试、贪婪插入、局部搜索和自适应大邻域搜索（ALNS），并自动选择最优的路径选择引擎。

**🔧 技术方法**

使用了Python编程语言，结合了贪婪算法、局部搜索、ALNS、分支限界法和拉格朗日松弛等技术。

**📊 数据集**

使用的数据集包括38个评分实例，实例规模从几条请求到250+250条请求不等。

**📈 对比分析**

与其他方法比较时，求解器在官方评估中获得了487,237的总收益，所有运行均在277秒内完成，且最多使用279 MB内存，28个实例在时间限制之前就完成了。

**⚠️ 局限性**

限制在于对于大型实例的求解主要依赖启发式方法，可能导致性能不足，尤其是在处理违反三角不等式的网络时。

---

## 193. How Many Persistent Features Does a Biometric Need? A Law Linking Equal Error Rate to Feature Count and Persistence

**arXiv ID:** 2611.21973 | [PDF](https://arxiv.org/pdf/2611.21973v1)

**作者:** Oleg Komogortsev `[一作]` (Texas State University), Mehedi Hasan Raju `[通讯]` (Texas State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

论文探讨了生物特征识别中，特征数量与持久性之间的关系，提出了一种新的法律来连接等错误率与特征数量和持久性。

**💡 创新点**

创新点在于建立了一个数学模型，揭示了生物特征识别系统中所需的持久特征数量与等错误率之间的关系。

**🔧 技术方法**

使用了数学建模和统计分析的方法来推导和验证理论。

**📊 数据集**

使用了多种生物特征数据集进行实验，以验证提出的法律和模型的有效性。

**📈 对比分析**

通过与现有的生物特征识别方法进行比较，展示了所提出模型在特征数量和持久性方面的优势，性能得到了显著提升。

**⚠️ 局限性**

限制在于模型的适用性可能受到特定生物特征类型和数据集的影响，可能无法广泛适用于所有生物特征识别场景。

---

## 194. Knowing Which, Not How Much: Language Models Order Party Positions They Cannot Place

**arXiv ID:** 2611.21878 | [PDF](https://arxiv.org/pdf/2611.21878v1)

**作者:** Yousef Attia `[一作]` (Warsaw University of Technology), Jarosław A. Chudziak `[通讯]` (Warsaw University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了语言模型在评估社会实体（如政党）立场时的能力，特别是它们在排序和绝对定位上的表现。

**💡 创新点**

创新点在于揭示了比较性问题和绝对性问题在测量能力上的不同，表明模型在排序方面表现良好，但在绝对定位上存在知识缺口。

**🔧 技术方法**

使用了语言模型进行实验，评估了七种不同的模型在处理政党立场问题时的表现。

**📊 数据集**

使用了来自比较宣言项目（Comparative Manifesto Project）的编码数据，涵盖了日本和韩国的政党立场。

**📈 对比分析**

通过比较模型在排序和绝对定位任务上的表现，发现模型在排序任务中的准确率超过了基线，而在绝对定位任务中的表现低于基线，表明模型在处理绝对性问题时存在知识缺口。

**⚠️ 局限性**

限制在于比较宣言项目的编码本身可能存在不准确性，且研究仅基于两个国家的特定属性，结果的普遍性尚需进一步验证。

---

## 195. OmniTongue: A Comprehensive MLLM Benchmark for Automated Tongue Diagnosis in Traditional Chinese Medicine

**arXiv ID:** 2611.22652 | [PDF](https://arxiv.org/pdf/2611.22652v1)

**作者:** Liuyi Xu `[一作]` (Northeastern University), An-Yang Lu `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OmniTongue，一个针对中医舌诊的综合性、细粒度的视觉问答基准，旨在评估多模态大语言模型（MLLMs）在舌诊中的表现。

**💡 创新点**

创新点在于OmniTongue是首个综合性、细粒度的舌诊基准，包含16个临床维度和66个细粒度标签，提供区域特定的注释以评估病理定位能力。

**🔧 技术方法**

使用了多模态大语言模型（MLLMs）进行舌诊的视觉理解评估，并通过模板生成问题-答案对来构建基准。

**📊 数据集**

数据集包含1,351张专家验证的舌头图像和21,616个指令对，涵盖了舌头的多个临床维度。

**📈 对比分析**

与九种主流MLLMs进行了广泛评估，结果显示现有模型在区域诊断任务中表现不佳，宏观F1分数普遍较低，表明模型依赖统计捷径而非真实理解舌病理。

**⚠️ 局限性**

限制在于当前评估仅集中于九种开源模型，未来计划扩展到更多闭源和医学特定的MLLMs。

---

## 196. When does deep reinforcement learning pay off in HVAC supervisory control? Four small Modelica case studies trained directly on FMUs

**arXiv ID:** 2611.22144 | [PDF](https://arxiv.org/pdf/2611.22144v1)

**作者:** Bingnan Zhao `[一作]` (Queensland University of Technology), Tharindu Fernando `[通讯]` (Queensland University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了深度强化学习（DRL）在供暖、通风和空调（HVAC）系统的监督控制中的应用，进行了四个小型案例研究，评估了不同控制策略的性能。

**💡 创新点**

创新点在于通过小型可重复的案例研究，明确了DRL控制策略与传统调优规则之间的性能差距，并提出了在应用DRL之前评估可获得的节省幅度的建议。

**🔧 技术方法**

使用了深度强化学习中的软演员-评论家（SAC）算法，并通过Modelica Buildings库建模HVAC系统。

**📊 数据集**

使用了布里斯班机场的TMYx 2011-2025气象数据，案例基于一个办公室建筑的HVAC系统。

**📈 对比分析**

与调优的规则控制器进行比较，DRL在没有储存的情况下仅比最佳规则高出0.2%到1.3%；在有储存的情况下，DRL的表现优于规则控制器，但未达到完美预见调度的节省幅度。

**⚠️ 局限性**

限制在于每个案例只使用了一个建筑、一个气候、一个电价和一个测试周期，且训练种子数量有限，模型的理想化和简化可能影响结果的普遍性。

---

## 197. Learning to Edit IFC Building Models: Verifiable Checkers as the Training Signal for Dependency-Aware Editing Agents

**arXiv ID:** 2611.22307 | [PDF](https://arxiv.org/pdf/2611.22307v1)

**作者:** Jiaxin Zhang `[一作]` (Nanchang University), Bowen Wang `[通讯]` (SANKEN)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了IfcGym环境，通过可执行代码编辑IFC模型，并使用四个检查器验证编辑的有效性、意图、依赖性和保存性。

**💡 创新点**

创新点在于将编辑要求转化为可执行的训练目标，并通过严格的成功标准来选择训练数据和奖励编辑。

**🔧 技术方法**

使用了Qwen3.5-9B和Qwen3.5-27B语言模型，通过监督微调(SFT)和群体相对策略优化(GRPO)进行训练。

**📊 数据集**

使用了19,141个经过验证的教师示例，涵盖3,059个合成建筑，以及1,990个来自13个真实项目的编辑任务作为测试集。

**📈 对比分析**

在IfcEditBench上，经过GRPO的学生模型在严格成功率上达到了86.78%，相比于SFT后的79.15%有显著提升，且在公共基准BIM-Edit和BIBIMBAP上也表现优异。

**⚠️ 局限性**

限制在于模型的训练数据主要来自合成建筑，可能在处理真实建筑时存在转移能力的不足。

---

## 198. Governance Risks in Decentralized Autonomous Organizations (DAOs): Implications for Traditional Financial Institutions

**arXiv ID:** 2611.23201 | [PDF](https://arxiv.org/pdf/2611.23201v1)

**作者:** Christian Plöschberger `[一作]`, Mark Christopher Ballandies `[通讯]` (University of Zurich)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了去中心化自治组织（DAO）的治理机制与传统公司治理结构的差异，以及这些差异对受监管金融机构的风险管理的影响。

**💡 创新点**

创新点在于建立了一个结构化的评估框架，将DAO治理特征与现有的公司治理、外包、技术风险和运营韧性过程联系起来，帮助金融机构通过针对性扩展现有实践来应对DAO治理的依赖性。

**🔧 技术方法**

采用了定性比较框架，基于OECD治理原则、巴塞尔委员会指导和FINMA通函进行分析，并结合真实案例进行说明。

**📊 数据集**

使用了多个真实案例来说明DAO治理机制的风险，包括Tornado Cash、Beanstalk、Mango Markets、Compound和The DAO等。

**📈 对比分析**

通过比较DAO的审议、投票和问责机制与传统公司治理机制，发现DAO治理在核心治理功能上有所保留，但在权力集中、提案控制薄弱、法律不确定性和问责不清等方面存在独特风险。性能方面，DAO治理的透明性和可追溯性是其优势，但在法律责任和执行方面存在不足。

**⚠️ 局限性**

限制在于本文主要关注治理机制，而未深入探讨DAO的社区和执行维度。此外，所使用的案例并不具代表性，评估框架也未经过实证验证。

---

## 199. CrossEnv: Synthesizing Verifiable Cross-Environment Tasks for Long-Horizon Agent Training

**arXiv ID:** 2611.22777 | [PDF](https://arxiv.org/pdf/2611.22777v1)

**作者:** Zhengfeng Yu `[一作]` (Beijing Institute of Technology), Xian-Ling Mao `[通讯]` (Beijing Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CrossEnv框架，用于合成可执行和可验证的跨环境任务，并引入CrossEnv-Curriculum以逐步训练代理在多个环境中协调操作。

**💡 创新点**

创新点在于通过实体-工具图连接不同环境，并对任务特定数据进行对齐，从而合成可执行的跨环境任务，同时提供了一个渐进的训练方案以提高代理的协调能力。

**🔧 技术方法**

使用了实体-工具图、监督微调和阶段性强化学习等技术。

**📊 数据集**

使用了多个外部基准数据集，包括τ^2-bench、DeepPlanning、VitaBench和Claw-Eval。

**📈 对比分析**

与其他方法相比，CrossEnv在多个基准上表现出竞争力，且所需的训练样本更少，显示出跨环境数据和课程训练的优势。

**⚠️ 局限性**

限制在于程序合成的环境可能简化了真实服务行为，且在动态变化的数据和操作条件下的转移能力需要进一步评估。

---

## 200. A Linear-Size Well-Separated Pair Decomposition for Unit Disk Graphs

**arXiv ID:** 2611.22049 | [PDF](https://arxiv.org/pdf/2611.22049v1)

**作者:** Jie Gao `[一作]` `[通讯]` (Rutgers University), Jie Gao (Rutgers University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的算法，能够为单位圆盘图生成线性大小的c-良分离对分解（WSPD）。

**💡 创新点**

创新点在于首次证明了单位圆盘图的c-WSPD可以达到O(n)的大小，解决了之前的开放问题。

**🔧 技术方法**

使用了分层网树（net-tree）和随机填充分解（randomized padded decomposition）等技术。

**📊 数据集**

使用了单位圆盘图的点集，包含n个点。

**📈 对比分析**

与之前的算法相比，新的算法在性能上显著提高，能够在O(n^2 log^2 n/log log n)的时间内找到所有对的最短路径，而之前的算法需要O(n log n)的时间。

**⚠️ 局限性**

限制在于算法的运行时间较高，尤其是在处理大规模数据时，可能会影响实际应用的效率。

---

## 201. Automated Annotation-Free Synthetic Data Generation for Underwater Litter Detection: A Comparison of Generative Approaches

**arXiv ID:** 2611.23037 | [PDF](https://arxiv.org/pdf/2611.23037v1)

**作者:** Amala Mary Vincent `[一作]` (Delft University of Technology), Bart De Schutter `[通讯]` (Delft University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种完全自动化的零注释合成数据生成管道，用于生成标注的水下训练数据，以解决低数据环境下的水下垃圾检测问题。

**💡 创新点**

创新点在于通过样本边界框直接生成标签，完全消除了手动标注的需求，并系统比较了三种物体插入策略的效果。

**🔧 技术方法**

使用了复制粘贴、稳定扩散修复和DALL-E图像条件生成三种技术。

**📊 数据集**

使用了SeaClear水下垃圾数据集，该数据集包含8610张图像，涵盖了五个地理位置的不同水下条件。

**📈 对比分析**

通过与真实数据的比较，DALL-E在五个领域中提高了检测性能，最高相对增益达到96%。而复制粘贴和稳定扩散修复在所有领域中均未能改善检测性能。

**⚠️ 局限性**

限制在于DALL-E是一个封闭的专有模型，其架构和生成过程不公开，限制了可解释性和可重复性。

---

## 202. High-Fidelity Binarized Neural Vocoder

**arXiv ID:** 2611.22593 | [PDF](https://arxiv.org/pdf/2611.22593v1)

**作者:** Andong Li `[一作]` (Institute of Acoustics Chinese Academy of Sciences), Chengshi Zheng `[通讯]` (Institute of Acoustics Chinese Academy of Sciences)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种名为BiVoC的二进制神经声码器，旨在高保真波形生成，特别是针对现有声码器架构的直接二进制化所带来的信息损失问题。

**💡 创新点**

创新点在于采用了范围-零空间分解（RND）理论，结合分层子带建模和专门设计的二进制卷积单元，以提高在1位权重和激活下的表示能力和特征建模。

**🔧 技术方法**

使用了二进制神经网络（BNN）、范围-零空间分解、分层子带建模和混合精度设计等技术。

**📊 数据集**

使用了LibriTTS数据集进行训练，并在多个分布外基准上进行了广泛实验。

**📈 对比分析**

与直接二进制化的现有声码器（如HiFiGAN+BNN和Vocos+BNN）相比，BiVoC在多个客观和主观指标上表现出显著优势，同时在理论计算成本上也更低。

**⚠️ 局限性**

限制在于当前的PyTorch实现未能利用专用的位操作，未来工作将集中在硬件优化和真实设备部署上，以进一步提高推理延迟、内存消耗和能效。

---

## 203. Disagree to Prune: Reallocating Visual Tokens Where a Small VLM Dissents

**arXiv ID:** 2611.22472 | [PDF](https://arxiv.org/pdf/2611.22472v1)

**作者:** Yanchao Li `[一作]` (Nanjing University), Nanqing Dong `[通讯]` (Shanghai Innovation Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为PRUDENT的方法，通过保留视觉标记的一部分来加速大型视觉语言模型（VLMs），并在预算内重新回答存在分歧的问题。

**💡 创新点**

创新点在于通过小型VLM的分歧来识别需要更多标记的问题，并在相同的平均预算下进行重新回答，从而提高了性能。

**🔧 技术方法**

使用了视觉标记修剪技术，并结合了小型VLM作为检查器来判断答案的可靠性。

**📊 数据集**

在18个基准测试上进行了评估，包括6个文本丰富的基准和10个自然图像的基准。

**📈 对比分析**

与八种现有的修剪方法进行比较，PRUDENT在所有基准测试中表现最佳，尤其是在预算紧张时，其优势更加明显。

**⚠️ 局限性**

限制在于小型VLM的性能较弱，可能在某些情况下无法完全替代大型VLM的能力。

---

## 204. A Probabilistic Lateral Motion Prediction Model with Improved Response Fidelity for Vehicle-in-the-Loop Simulation

**arXiv ID:** 2611.21806 | [PDF](https://arxiv.org/pdf/2611.21806v1)

**作者:** Xiangyang Ye `[一作]` (Jilin University), Jun Li `[通讯]` (Tsinghua University)

**通讯引用:** 84068 | [OpenAlex ID](https://openalex.org/A5100361956)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种物理引导的概率性横向运动预测模型，用于车辆在环模拟，结合了二自由度车辆模型和递归概率残差学习，以生成横向位置和航向的预测。

**💡 创新点**

创新点在于通过多步展开训练和直接监督累积横向速度误差来提高预测的响应保真度，同时引入条件响应协方差传播来量化横向响应和集成运动的预测不确定性。

**🔧 技术方法**

使用了二自由度自行车模型和递归神经网络（RNN）来学习横向加速度和偏航率的相关残差修正，并采用多步展开训练策略。

**📊 数据集**

使用了RT3000系统收集的驾驶数据集，包括44个车道变换和两个蛇形行驶记录，共分为33个训练、7个验证和6个测试记录。

**📈 对比分析**

与Deep Dynamics横向适应模型相比，提出的模型在测试中将平均横向位置均方根误差（RMSE）从0.596米降低到0.483米，改善幅度为19.0%，同时将平均横向速度RMSE从0.0469米/秒降低到0.0441米/秒。

**⚠️ 局限性**

限制在于模型的响应不确定性可能被低估，且在实际应用中需要进一步评估实时执行和闭环跟踪性能。

---

## 205. Sample-Aligned Decoders for Latent Generative Models

**arXiv ID:** 2611.22458 | [PDF](https://arxiv.org/pdf/2611.22458v1)

**作者:** Yu Lei `[一作]` (Institute of Computing Technology), Qingming Huang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种后训练框架Sample-Aligned Decoders，旨在优化潜在生成模型中的解码器，以提高生成图像的质量。

**💡 创新点**

创新点在于通过生成样本分布对齐（GSA）和编码器锚点保护（EAP）来调整解码器，使其适应生成潜在变量的分布，而不是仅仅依赖于编码器潜在变量。

**🔧 技术方法**

使用了生成样本分布对齐（GSA）和编码器锚点保护（EAP）技术，结合了冻结的生成器和特征空间对齐。

**📊 数据集**

使用了多个ImageNet生成器和COCO数据集进行文本到图像生成的实验。

**📈 对比分析**

与现有的修正方法相比，该方法在四个最先进的ImageNet生成器上均表现出显著的性能提升，FID-50K从1.15降低到0.98，并且在不增加推理成本的情况下，保持了重建质量。

**⚠️ 局限性**

限制在于由于锚点的约束，无法增加多样性或修复错误的结构，且解码器特定于其生成器和采样器家族，不能跨模型转移。

---

## 206. Beyond Bitwidth: Deployment-Constrained Mixed-Precision Quantization for Multilingual LLM Inference on NVIDIA Blackwell

**arXiv ID:** 2611.21974 | [PDF](https://arxiv.org/pdf/2611.21974v1)

**作者:** Hyojung Han `[一作]` `[通讯]` (ThakiCloud), Hyojung Han (ThakiCloud)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在NVIDIA B200上使用vLLM对Qwen3-8B进行低比特量化，评估了不同配置下的执行精度、内存和延迟限制。

**💡 创新点**

提出了一种端到端的验证工作流程，检查每个融合模块的打包执行和权重类型，以及针对教师的质量测量和服务延迟。

**🔧 技术方法**

使用了低比特量化技术，包括INT2/INT3、GPTQ、NVFP4和混合精度配置。

**📊 数据集**

使用了多个数据集，包括KMMLU、KoBEST、MMLU、ARC、HellaSwag等，进行多语言评估。

**📈 对比分析**

通过与BF16教师的比较，发现某些配置在开发集上表现良好，但在最终集上未能通过质量门槛，表明开发集的表现并不总能转化为实际应用中的成功。

**⚠️ 局限性**

研究的局限性包括只使用了一个模型和一个GPU，未能保留最终阶段的检查点，且在不同上下文和并发情况下的性能未被测量。

---

## 207. GP--RL--MPC: Uncertainty-Aware Reinforcement Learning-Augmented Model Predictive Control with Gaussian Process Postdecision State-Value Functions

**arXiv ID:** 2611.22038 | [PDF](https://arxiv.org/pdf/2611.22038v1)

**作者:** Jon Li `[一作]` (University of Southern California), Jay H. Lee `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种高斯过程强化学习增强的模型预测控制（GP–RL–MPC），利用学习到的评论员作为MPC终端惩罚，以考虑预测范围之外的成本。

**💡 创新点**

创新点在于通过后决策状态值函数捕捉随机状态转移的影响，并结合高斯过程回归（GPR）进行数据高效的评论员学习，同时引入可调风险系数来调节对不确定性预测的敏感性。

**🔧 技术方法**

使用了高斯过程回归（GPR）、乐观近似策略迭代（OAPI）和枢轴Cholesky（PC）选择诱导点等技术。

**📊 数据集**

在连续搅拌罐反应器（CSTR）上进行了实验，该反应器受到外部干扰的影响。

**📈 对比分析**

与基线MPC相比，GP–RL–MPC在五个随机种子下平均减少了80.52%的平方误差（SSE），显示出显著的性能提升。

**⚠️ 局限性**

限制在于该方法在处理高度非线性和复杂的随机系统时可能面临挑战，尤其是在构建适当的终端成本和约束方面。

---

## 208. Adaptive Memory for Online Reinforcement Learning under Sparse Model Changes

**arXiv ID:** 2611.21879 | [PDF](https://arxiv.org/pdf/2611.21879v1)

**作者:** Sreejeet Maity `[一作]` (North Carolina State University), Leonardo F. Toso `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在稀疏模型变化下的折扣强化学习问题，提出了一种算法，通过选择历史数据来估计奖励和状态转移概率，从而适应环境变化。

**💡 创新点**

创新点在于提出了一种自适应历史选择算法，能够针对每个状态-动作对分别选择历史数据，以提高模型的估计精度，同时避免因使用过时数据而导致的偏差。

**🔧 技术方法**

使用了自适应历史选择算法，该算法通过构建置信区间来选择历史数据，并在每次添加历史数据后进行一致性检查。

**📊 数据集**

使用了一个具有50个状态和4个动作的MDP数据集，模拟了奖励和转移变化，变化的比例为ϵ，且变化的具体条目未知。

**📈 对比分析**

与重启方法、全历史方法、池化所有可用观察数据的方法和共享历史方法进行了比较，结果表明，所提方法在减少平均策略损失方面表现最佳，尤其在多个部署后，平均策略损失显著低于其他方法。

**⚠️ 局限性**

限制在于算法假设环境变化是稀疏的，并且在某些情况下可能无法准确识别变化的具体条目或其幅度。

---

## 209. Segmentation-Guided Direct Capacity Estimation of Rooftop Photovoltaics from Aerial Imagery

**arXiv ID:** 2611.22076 | [PDF](https://arxiv.org/pdf/2611.22076v1)

**作者:** Guibin Chen `[一作]` (New York University), Yuzhang Lin `[通讯]` (New York University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Direct Head的架构，通过分割引导容量估计，直接从PV区域特征和面积描述符回归非负容量。

**💡 创新点**

创新点在于不再强制容量计算为面积和功率密度的乘积，而是允许最终容量映射偏离固定的物理转换，同时保留空间监督。

**🔧 技术方法**

使用了U-Net进行分割，并结合了多层感知器（MLP）进行容量的直接回归。

**📊 数据集**

使用了来自法国的7590个屋顶光伏安装的高分辨率航空图像数据集。

**📈 对比分析**

与传统的面积密度神经模型相比，Direct Head在原始分割上减少了8.7%的平均绝对误差（MAE），在行政保留上减少了12.5%；对称平均绝对百分比误差（sMAPE）分别减少了14.0%和20.7%。

**⚠️ 局限性**

限制在于尽管分割精度和容量精度是不同的问题，但它们之间的关系可能导致在某些情况下容量估计不准确。

---

## 210. DiagTrace: An Interactive Benchmark for Clinical Decision Making

**arXiv ID:** 2611.22009 | [PDF](https://arxiv.org/pdf/2611.22009v1)

**作者:** Yufan Wang `[一作]` (Weill Cornell Medicine), Yifan Peng `[通讯]` (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于来源的基准，用于评估大型语言模型（LLMs）在临床诊断轨迹重建中的表现，强调了信息获取的重要性。

**💡 创新点**

创新点在于将信息获取与诊断推理分开评估，并在固定的回合预算下考察模型如何有效地请求信息。

**🔧 技术方法**

使用了大型语言模型（如GPT-4.1 mini、Claude Haiku 4.5、Gemini 2.5 Flash和Kimi K2.5）进行交互式重建和评估。

**📊 数据集**

使用了来自PubMed Central的510个病例报告，涵盖八个临床专业，其中225个病例用于评估。

**📈 对比分析**

通过比较模型的请求轨迹与来源参考，评估了临床课程对齐、交互质量和效率，结果显示模型在信息覆盖、来源一致性和交互效率之间存在权衡。

**⚠️ 局限性**

局限性包括：源顺序反映的是已发布文章的叙述而非临床时间线，临床病例报告可能不反映常规工作流程，基准构建和质量审计部分依赖于LLM的辅助。

---

## 211. Similar Is Not Useful: Selecting In-Context Demonstrations by Closed-Loop Utility

**arXiv ID:** 2611.22295 | [PDF](https://arxiv.org/pdf/2611.22295v1)

**作者:** Jiachen Li `[一作]` (University of Texas at Austin), Zhuo Xu `[通讯]` (General Intelligent Machines, Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了在视觉-语言-动作（VLA）政策中，如何通过选择演示的闭环效用来提高政策的成功率，而不是仅仅依赖于视觉相似性。

**💡 创新点**

创新点在于提出了一种评估协议，通过实时交换演示池来估计政策的成功率，并发现基于效用的选择显著优于基于相似性的选择。

**🔧 技术方法**

使用了冻结的DINOv2嵌入和回归分析技术来评估演示的效用。

**📊 数据集**

使用了LIBERO平台上的19个任务的数据集，每个任务有20个处理过的演示。

**📈 对比分析**

与基于相似性选择的池相比，基于效用选择的池在760个配对实验中成功率提高了约11个百分点，且相似性选择的池表现与随机选择无显著差异。

**⚠️ 局限性**

限制在于研究仅覆盖了一个第三方RICL移植的检查点，且未测试物理机器人转移，此外，某些任务的效用估计存在分半一致性问题。

---

## 212. BernBRS: Certified Backward Reachability for Deep Bernstein Neural Networks

**arXiv ID:** 2611.22083 | [PDF](https://arxiv.org/pdf/2611.22083v1)

**作者:** Ibrahim Albool `[一作]` (University of California), Yasser Shoukry `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种高效且可扩展的框架，用于深度伯恩斯坦神经网络的反向可达性分析，能够同时计算反向可达集的内外近似。

**💡 创新点**

创新点在于引入了一种基于激活模式的智能细分策略，通过分析放松间隙的精确分解来优化输入空间的覆盖，从而提高了复杂架构的可扩展性。

**🔧 技术方法**

使用了深度伯恩斯坦神经网络架构和基于激活模式的反向收缩算法，结合了前向界限传播和反向收缩过程。

**📊 数据集**

在实验中使用了高维模型，包括线性化的闭环半猎豹控制器和TinyBERT分类器的伯恩斯坦学生模型。

**📈 对比分析**

与现有的CROWN基线方法进行比较，展示了在相同时间预算下，所提出的方法在内外近似的比率上表现更优，且在复杂网络架构上具有更好的可扩展性。

**⚠️ 局限性**

局限性在于该方法的数学证明和性能保证主要针对伯恩斯坦激活函数，可能不适用于其他类型的激活函数或网络架构。

---

## 213. A Common Exponential Lower Bound for Dantzig and Modified Simplex in Deterministic Discounted Games

**arXiv ID:** 2611.22977 | [PDF](https://arxiv.org/pdf/2611.22977v1)

**作者:** Xinyu Qiao `[一作]`, Tao Yao `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一类N状态的确定性双人折扣游戏，证明了在选择Dantzig规则和修改单纯形法之间进行自适应切换时，无论如何打破平局，从指定的初始策略开始都需要2^Ω(N)次切换。

**💡 创新点**

提出了一种返回门构造，使得Dantzig能够在无折扣的确定性马尔可夫决策过程中重现任何有限的Bland前缀，并且在特定条件下将Dantzig的切换次数从指数级减少到线性级。

**🔧 技术方法**

使用了Dantzig规则和修改单纯形法的策略迭代技术，结合了返回门构造来分析策略切换的复杂性。

**📊 数据集**

使用了具有N状态和M动作的确定性双人零和折扣游戏数据集，特别是基于二进制计数器的构造。

**📈 对比分析**

通过与修改单纯形法的比较，展示了在某些情况下Dantzig的切换次数是指数级的，而在其他情况下则可以通过返回门构造将切换次数降低到线性级。

**⚠️ 局限性**

限制在于当每个访问的Max策略都承认一个最佳响应时，Dantzig的切换次数可能会呈指数级增长，且在某些情况下，深度为2的前瞻性搜索仍可能需要指数级的切换次数。

---

## 214. The Whole Warps the Parts: Attentional Certainty Modulates Ensemble-Induced Bias in Visualizations

**arXiv ID:** 2611.22885 | [PDF](https://arxiv.org/pdf/2611.22885v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 215. Towards Data-Efficient, Real-Time World Models

**arXiv ID:** 2611.22497 | [PDF](https://arxiv.org/pdf/2611.22497v1)

**作者:** Raluca Stevenson `[一作]`, Sarah Parisot `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究展示了一种高效的世界模型WHAM-RT，旨在实现快速高质量推理和改进的数据效率，解决了当前世界模型在实时交互和数据需求方面的局限性。

**💡 创新点**

创新点在于采用了两阶段的离散扩散方法替代传统的下一个标记预测范式，并引入了一种轻量级的精炼变换器进行迭代掩蔽预测，从而实现了79倍的模型加速。

**🔧 技术方法**

使用了离散扩散和自回归模型相结合的技术，构建了一个大型骨干变换器和一个轻量级的精炼变换器。

**📊 数据集**

使用的数据集包含约28小时的游戏数据，专注于单一游戏环境，经过精心策划以提高数据效率。

**📈 对比分析**

与传统的WHAM模型相比，WHAM-RT在生成速度上实现了显著提升，达到每秒17帧，尽管在质量上有一定的折衷，但仍保持了较高的一致性和复杂交互能力。

**⚠️ 局限性**

限制在于研究仅限于Bleeding Edge环境，未来工作将探索跨环境训练和适应性，同时加速可能会降低模型的生成质量。

---

## 216. Evaluating AST-Derived Knowledge Graph Retrieval for AIOps Incident Diagnosis in Apache Kafka

**arXiv ID:** 2611.21625 | [PDF](https://arxiv.org/pdf/2611.21625v1)

**作者:** Shashank Mishra `[一作]` (Independent Researcher), Shweta Mishra `[通讯]` (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了图增强检索在Apache Kafka中的应用，构建了一个基于抽象语法树（AST）派生的所有权图，并使用1600个源链接日志记录和315个事件风格查询进行实验。

**💡 创新点**

创新点在于构建了一个AST派生的结构图，连接Kafka类、方法和日志模板，并评估了静态扩展与基于信心的分阶段扩展的效果。

**🔧 技术方法**

使用了抽象语法树（AST）解析技术、图增强检索、密集检索和交叉编码重排序等技术。

**📊 数据集**

使用的数据集包括1600个源链接日志记录和315个事件风格查询，涵盖103个独特目标。

**📈 对比分析**

与静态AST-KG相比，基于信心的分阶段扩展在交叉编码对比工作负载和重排序时间上减少了约52%。然而，检索质量没有显著改善，向量检索在每个测试深度上都表现最佳。

**⚠️ 局限性**

限制在于AST派生的结构证据未能显著改善下游事件诊断的根本原因正确性，且实验仅限于Apache Kafka代码库和103个独特目标，结果不应推广到其他设置。

---

## 217. Spool: Low-Latency Streaming Inference for Hybrid Vision-Language Models

**arXiv ID:** 2611.23228 | [PDF](https://arxiv.org/pdf/2611.23228v1)

**作者:** Rui Ning `[一作]`, Hari Balakrishnan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文探讨了某个主题，具体内容未提供。

**💡 创新点**

创新点未明确说明。

**🔧 技术方法**

使用的技术未明确说明。

**📊 数据集**

使用的数据集未明确说明。

**📈 对比分析**

比较的方法和性能未明确说明。

**⚠️ 局限性**

论文的局限性未明确说明。

---

## 218. ResoLUT: Distilling Residual Networks into Lookup-Table Networks

**arXiv ID:** 2611.22955 | [PDF](https://arxiv.org/pdf/2611.22955v1)

**作者:** Katarzyna Fojcik `[一作]` (Wroclaw University of Science and Technology), Jogundas Armaitis `[通讯]` (Transcendent Logic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为ResoLUT的逻辑网络，通过逐层提取训练好的残差网络（ResNet）的连接性，将其转化为查找表（LUT）树，保留多位残差流和BatchNorm。

**💡 创新点**

ResoLUT是首个在计算机视觉中保留ResNet风格多位残差流的卷积逻辑网络，避免了逻辑中的离散化差距，并且通过教师网络提取连接性。

**🔧 技术方法**

使用了残差网络（ResNet）进行知识蒸馏，采用查找表（LUT）进行逻辑计算，结合了比较器和树结构。

**📊 数据集**

在CIFAR-10、CIFAR-100和ImageNet数据集上进行了实验，CIFAR-10上达到95.66%的准确率，CIFAR-100上达到78.87%的准确率。

**📈 对比分析**

与其他方法相比，ResoLUT在CIFAR-10和CIFAR-100上表现出最佳的准确率，且在深度和树的数量上具有良好的扩展性。ResoLUT-8在FPGA上以189k图像每秒的速度运行，达到90.04%的准确率，显示出高吞吐量。

**⚠️ 局限性**

限制在于教师-学生的准确性差距随着网络深度的增加而增大，且未广泛测试与残差流和树预算及形状相关的优化。

---

## 219. $π^4$: A Foundation Model for 4D Scene Geometry and Motion

**arXiv ID:** 2611.23144 | [PDF](https://arxiv.org/pdf/2611.23144v1)

**作者:** Yifan Wang `[一作]` (Shanghai Artificial Intelligence Laboratory), Chunhua Shen `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种统一的基础模型，用于从多种未校准的RGB观察中进行4D场景感知，能够在单次前向传递中恢复密集几何、相机姿态和度量尺度，以及2D/3D点跟踪和移动物体分割。

**💡 创新点**

模型通过几何分支和几何条件运动分支的结合，实现了几何和运动的统一，能够处理单图像、无序图像集合和视频序列，且在多个基准测试中表现出色。

**🔧 技术方法**

采用了共享的DINOv3编码器和循环Transformer解码器，结合了几何和运动的预测，使用了稀疏3D细化器来增强局部表面结构和几何细节。

**📊 数据集**

使用了来自Funes和Sketchfab的合成场景数据集，以及经过处理的互联网视频数据，涵盖了建筑环境和动态场景。

**📈 对比分析**

与其他方法相比，该模型在几何评估中获得了最佳的平均排名，并在单视图和多视图的2D/3D点跟踪和移动物体分割中表现出竞争力，且参数量约为0.5B，远小于大多数基线。

**⚠️ 局限性**

模型对相机位置的敏感性较高，尤其是在距离重建原点较远时，且在长序列推断时的准确性和泛化能力仍有待提高。

---

## 220. DSADiff: Data Structure-Aware Diffusion for Crystal Structure Prediction via Mitigating Complexity- and Imbalance-Induced Biases

**arXiv ID:** 2611.22050 | [PDF](https://arxiv.org/pdf/2611.22050v1)

**作者:** Jiang Xiaobo `[一作]`, Dinghong Lai `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的数据结构感知扩散模型DSADiff，用于晶体结构预测，旨在减轻复杂性和成分不平衡引起的偏差。

**💡 创新点**

创新点在于引入了层次分析策略（LAS），系统分析晶体复杂性和元素组成不平衡对生成性能的影响，并将其整合到DSADiff模型中。

**🔧 技术方法**

使用了扩散模型技术，结合了复杂性感知和成分不平衡的优化策略。

**📊 数据集**

使用了三个CSP基准数据集进行实验，包括Carbon-24、Perov-5和MP-20。

**📈 对比分析**

与现有的最先进方法相比，DSADiff在Match Rate和RMSE上均表现出显著的性能提升，最高Match Rate提升达9%。

**⚠️ 局限性**

限制在于模型的复杂性和训练过程中的数据不平衡可能仍然影响生成性能，未来需要进一步优化和扩展。

---

## 221. ARC-2: Continuous Compliance Through Delta Management

**arXiv ID:** 2611.21811 | [PDF](https://arxiv.org/pdf/2611.21811v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 222. AutoE2E: Don't speak - Language free end-to-end autonomy A Real-Time Driving Model for L2++ ADAS and L4 Robotaxis

**arXiv ID:** 2611.22374 | [PDF](https://arxiv.org/pdf/2611.22374v1)

**作者:** Muhammad Zain Khawaja `[一作]` (Autoware Foundation), Nishad Sahu `[通讯]` (Autoware Foundation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个名为AutoE2E的开源端到端驾驶模型，结合多摄像头感知、高清地图信息和路线条件来预测未来车辆运动。

**💡 创新点**

引入了一种时间鸟瞰图（BEV）架构，旨在提供一个可检查的基线，以便于进一步的研究和实验。

**🔧 技术方法**

使用了预训练的摄像头-BEV骨干网、时间特征融合和轻量级递归轨迹规划器。

**📊 数据集**

使用了KITScenes和nuPlan数据集进行训练和评估。

**📈 对比分析**

在KITScenes验证基准上，经过微调的检查点在三秒预测范围内实现了平均位移误差（ADE）为0.745米，最终位移误差（FDE）为2.020米，性能优于基线模型。

**⚠️ 局限性**

当前模型为开放环模型，尚未经过闭环验证，且未在相同场景上进行控制的地图/路线实验。

---

## 223. Residual-Guided Lightweight Reinforcement Learning for Risk-Aware Trajectory Selection with a Frozen World Model

**arXiv ID:** 2611.22998 | [PDF](https://arxiv.org/pdf/2611.22998v1)

**作者:** Jiaqi Wu `[一作]` (RWTH Aachen University), Alvaro Garcia Hernandez `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于残差引导的强化学习框架，用于在冻结的世界模型和固定的候选库中进行轻量级轨迹选择。

**💡 创新点**

创新点在于通过局部轨迹变动的反馈训练上下文条件排名网络，从而实现场景依赖的候选偏好学习。

**🔧 技术方法**

使用了近端策略优化（PPO）技术来微调候选选择和有界轨迹残差的联合策略。

**📊 数据集**

使用了nuScenes数据集进行训练和开放环评估，包含698个场景和5189个上下文。

**📈 对比分析**

与固定选择规则的RiskWorld模型相比，该方法在开放环评估中将平均欧几里得轨迹误差（L2）减少了%，并且在匹配反馈和更新预算下，残差探索进一步将L2减少了0.83%。

**⚠️ 局限性**

限制在于该方法的评估是在开放环设置下进行的，未来工作应优先考虑反应性闭环评估，以评估执行的驾驶性能和与其他道路用户的交互。

---

## 224. Where Demand Meets Supply: Spatial Elastic Consensus for Low-Bandwidth Collaborative Perception

**arXiv ID:** 2611.22439 | [PDF](https://arxiv.org/pdf/2611.22439v1)

**作者:** Linshan Yan `[一作]` `[通讯]` (Northwestern Polytechnical University), Linshan Yan (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为空间弹性共识（SEC）的带宽自适应框架，旨在通过双边需求-供应共识机制来提高协作感知的效率，特别是在低带宽条件下。

**💡 创新点**

创新点在于引入了双边共识机制，优先考虑被自我车辆（ego vehicle）遗漏的关键信息，而不是发送冗余的自我可见信息，同时确保严格的带宽预算控制。

**🔧 技术方法**

使用了双边共识和反膨胀选择管道的技术，结合轻量级的可变形采样器来聚合稀疏信息。

**📊 数据集**

在模拟的OPV2V和V2XSet数据集以及真实世界的DAIR-V2X数据集上进行了评估。

**📈 对比分析**

与现有方法相比，SEC在64 KiB的严格带宽预算下，始终在AP@0.7指标上表现最佳，超越了之前的最强方法，显示出其在低带宽协作感知中的有效性。

**⚠️ 局限性**

限制在于该方法依赖于车辆之间的有效通信，若通信条件不佳，可能会影响性能。

---

## 225. Derandomized Sunflowers and Radical Lower Bounds

**arXiv ID:** 2611.23109 | [PDF](https://arxiv.org/pdf/2611.23109v1)

**作者:** Bruno Cavalar `[一作]` (University of Oxford), Amir Yehudayoff `[通讯]` (University of Copenhagen)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

论文证明了对于宽度为w的单调DNF，存在一个O(w)-伪独立分布，使得该DNF包含一个(,δ)-向日葵，且该结果可以应用于改进非单调算术电路与单调电路的根之间的分离。

**💡 创新点**

创新点在于证明了在O(w)-伪独立分布下的稳健向日葵引理，并且该结果在局部独立和近似相同分布的情况下也成立，提供了比以往更弱的参数。

**🔧 技术方法**

使用了高阶马尔可夫不等式和矩方法来证明稳健向日葵引理。

**📊 数据集**

使用了宽度为w的单调DNF，且该DNF的大小至少为(w/q log(1/δ))^w。

**📈 对比分析**

与之前的结果相比，本文的结果在参数k上取得了更小的值，且在=0的情况下也能取得较好的结果，表明在O(w log(1/δ))的伪独立分布下，单调DNF能够包含(,δ)-向日葵。

**⚠️ 局限性**

限制在于尚不清楚是否存在更一般的分布能够欺骗单调DNF，尤其是对于几乎独立的分布，可能无法欺骗宽度为多对数的DNF。

---

## 226. Learning Accurate End-Effector Trajectory Tracking with Inertial Feedback for a Quadruped Manipulator

**arXiv ID:** 2611.22695 | [PDF](https://arxiv.org/pdf/2611.22695v1)

**作者:** Dohyeon Kim `[一作]` (Jeonbuk National University), Yeongtae Jung `[通讯]` (Jeonbuk National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一种基于强化学习的框架，利用末端执行器的惯性反馈来提高四足机械手的轨迹跟踪精度，旨在在不进行身体运动或脚部重新定位的情况下实现高精度操作。

**💡 创新点**

创新点在于首次将末端执行器的惯性观测作为额外反馈融入到残差学习框架中，以补偿由于系统不确定性传播而导致的跟踪误差，同时不依赖于外部传感基础设施。

**🔧 技术方法**

使用了强化学习技术，结合了残差学习和末端执行器的惯性测量单元（IMU）反馈。

**📊 数据集**

在仿真和真实世界实验中使用了自定义生成的轨迹数据集，包含了不同速度和加速度的命令条件。

**📈 对比分析**

与传统的基于运动学的控制方法相比，提出的方法在仿真和真实实验中均显示出显著的性能提升。在真实实验中，提出的策略在两种命令条件下分别减少了17.3%和28.0%的RMSE，相较于仅依赖本体感知的基线减少了12.6%和15.9%。

**⚠️ 局限性**

限制在于该方法主要集中在局部高精度操作，未来的工作将扩展到更高自由度的移动机械手和全身运动的工作空间扩展。

---

## 227. TANGO: Real-World Unregistered RGB-Guided Hyperspectral Super-Resolution with Two-Camera Acquisitions

**arXiv ID:** 2611.22699 | [PDF](https://arxiv.org/pdf/2611.22699v1)

**作者:** Matteo Kolyszko `[一作]` (University of Milano-Bicocca), Raimondo Schettini `[通讯]` (University of Milano-Bicocca)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的方法TANGO，用于在未注册的情况下进行RGB引导的高光谱图像超分辨率，利用DUET数据集进行评估。

**💡 创新点**

创新点在于引入了DUET数据集，这是第一个基于真实RGB和高光谱相机独立采集的大规模公共基准数据集，并提出了结合全局对齐和局部特征细化的TANGO框架。

**🔧 技术方法**

使用了全局对齐、粗到细的局部特征对齐和学习的门控机制来处理未注册的RGB引导。

**📊 数据集**

使用了DUET数据集，该数据集包含403个场景，分别由Sony α6100 RGB相机和Specim IQ高光谱相机独立捕获。

**📈 对比分析**

与其他方法（如SSC-HSR）相比，TANGO在DUET数据集上在所有评估的缩放因子上都表现出最先进的性能，PSNR提高了最多2.29 dB，同时计算量减少了2.5倍。

**⚠️ 局限性**

局限性在于尽管TANGO在多个场景中表现良好，但仍可能受到未注册图像之间几何不一致性的影响，特别是在复杂场景中。

---

## 228. Coordination of Heterogeneous Teams with Uncertain Task Requirements

**arXiv ID:** 2611.21828 | [PDF](https://arxiv.org/pdf/2611.21828v1)

**作者:** Jongann Lee `[一作]` (University of Illinois Urbana-Champaign), Melkior Ornik `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种层次化规划器，利用图神经网络策略为异构代理分配任务，并通过最短路径算法找到行动位置。

**💡 创新点**

创新点在于引入CAAT图作为规划上下文的图表示，能够有效协调任务发现、执行和阶段，且在不同地图和设置中实现零次转移。

**🔧 技术方法**

使用了图神经网络（GNN）和强化学习（RL）技术。

**📊 数据集**

使用了三个地图数据集：WV（西弗吉尼亚州）、WA（华盛顿州）和KTH（瑞典皇家理工学院的室内地图）。

**📈 对比分析**

与三种基线方法进行比较，结果显示该方法在所有测试地图上都表现出更低的平均后悔值，尤其在WV和WA地图上分别降低了17.3%和20.3%。

**⚠️ 局限性**

限制在于假设任务位置已知，且为集中式规划，假设完美通信和即时信息传递，缺乏对通信故障的鲁棒性。

---

## 229. Fast and Strong Image Protection against Diffusion Personalization

**arXiv ID:** 2611.22494 | [PDF](https://arxiv.org/pdf/2611.22494v1)

**作者:** Jiwon Park `[一作]` (Inha University), Namhyuk Ahn `[通讯]` (Inha University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种保护性扩散模型（PDM），该模型通过多次去噪步骤对新图像进行扰动采样，从而实现图像保护。

**💡 创新点**

PDM是首个通过扩散模型采样扰动的保护器，采用了目标残差投影（TRP）损失作为奖励，显著提高了保护效果。

**🔧 技术方法**

使用了条件扩散模型，并通过蒸馏和奖励后训练的方式进行训练。

**📊 数据集**

在CelebA-HQ（人脸）和WikiArt（艺术作品）数据集上进行了评估，每个数据集包含20个主题，每个主题4张图像。

**📈 对比分析**

与九个基线方法进行比较，PDM在保护效果上优于所有基线，并且在成本上仅为其教师模型的十分之一，保护速度约为每张图像1秒。

**⚠️ 局限性**

PDM在面对再生净化器（如噪声上采样）时，保护效果大幅下降，未来的研究需要探索如何提高对新骨干网络的迁移保护能力。

---

## 230. ExplicitIR: Explicit Spatial Policy Fields for All-in-One Image Restoration

**arXiv ID:** 2611.22222 | [PDF](https://arxiv.org/pdf/2611.22222v1)

**作者:** Dezhi Zhang `[一作]` (Shanxi University), Wenjian Wang `[通讯]` (Shanxi University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为ExplicitIR的图像恢复框架，旨在从各种退化类型和水平的输入中恢复干净图像。

**💡 创新点**

创新点在于将恢复策略表示为两个显式的每像素字段：Budget（更新强度）和Style（选择上下文窗口），实现了空间自适应恢复。

**🔧 技术方法**

使用了空间策略估计器（SPE）来预测这两个字段，并通过政策自适应解码器块（PADB）执行这些策略。

**📊 数据集**

在三个任务和五个任务的基准测试上进行了广泛实验，使用的数据集包括BSD400、WED、RainTrainL、CBSD68、SOTS和GoPro等。

**📈 对比分析**

与现有的最强一体化基线相比，ExplicitIR在性能上具有竞争力，尤其在去雨任务中表现突出，且在去雾和去噪任务中也保持了良好的竞争力。

**⚠️ 局限性**

限制在于该方法尚未扩展到真实世界的退化情况，未来的工作将致力于解决这一问题。

---

## 231. KD-DSE: A Knowledge-Driven LLM Framework for Fast Microarchitecture Design Space Exploration

**arXiv ID:** 2611.22411 | [PDF](https://arxiv.org/pdf/2611.22411v1)

**作者:** Siqing Fu `[一作]` (Jiangnan Institute of Computing Technology), Feng Guo `[通讯]` (Jiangnan Institute of Computing Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种知识驱动的大型语言模型（LLM）框架，用于快速的微架构设计空间探索（DSE）。

**💡 创新点**

创新点在于结合了五状态受限代理与真实的gem5-McPAT评估，通过架构知识库提供指导，并采用诊断引导的两阶段探索策略。

**🔧 技术方法**

使用了大型语言模型（LLM）、gem5和McPAT等技术。

**📊 数据集**

使用了29个参数的超标量核心，评估基于mmbench的基准测试。

**📈 对比分析**

与最先进的MicroEvo相比，框架在超体积（HV）上提高了约4.8%，在平均距离参考集（ADRS）上降低了约21.7%。

**⚠️ 局限性**

限制在于需要依赖于架构知识库的质量，且在冷启动时可能无法充分利用已有知识。

---

## 232. Rethinking Perceptual Similarity for Remote Sensing: Multi-Layer Attention Scoring and Human Preference Benchmarking

**arXiv ID:** 2611.21816 | [PDF](https://arxiv.org/pdf/2611.21816v1)

**作者:** Eli Underwood `[一作]` (Duke University), Myron Brown `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究开发了一种新的人类对齐感知质量度量（RSPD），用于评估生成的遥感图像，并在多个生成任务中进行了评估。

**💡 创新点**

创新点在于提出了一种基于中间层对齐注意力分数的度量方法，能够更好地捕捉遥感图像中的不同伪影，并且不依赖于人类偏好标签进行调优。

**🔧 技术方法**

使用了RemoteCLIP模型的中间层对齐注意力分数（AAS）技术，结合多层聚合来评估图像的感知相似性。

**📊 数据集**

构建了一个包含1500个三元组的盲注释基准数据集，涵盖了合成伪影、空中和太空超分辨率、SAR到光学翻译和时间序列控制等多个任务。

**📈 对比分析**

与简单的余弦距离和其他通用基线相比，RSPD在多个真实的遥感生成任务中表现出显著的性能提升，尤其是在超分辨率和SAR到光学翻译任务中。

**⚠️ 局限性**

限制在于伪影家族并不全面，且RSPD的校准分布是基于控制基准得出的，可能偏向于该基准中代表的伪影词汇。

---

## 233. Beyond Attribute-Centric Fairness: Modelling and Mitigating Unified Bias Representation in Omni-modal Large Language Models

**arXiv ID:** 2611.22889 | [PDF](https://arxiv.org/pdf/2611.22889v1)

**作者:** Runqi Lin `[一作]` (University of Oxford), Chris Russell `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为OmniDebias的方法，旨在促进全模态大型语言模型（OLLMs）的公平性，通过提取和减轻跨属性和模态的共享偏见表示，同时保留与属性相关的知识。

**💡 创新点**

创新点在于采用偏见中心的公式，能够在不依赖于预定义公平先验的情况下，解开社会偏见与合法知识之间的关系，并将提取的偏见统一到一个共享表示中，从而实现跨属性和模态的统一去偏见。

**🔧 技术方法**

使用了偏见中心的框架，结合了后期编辑（PE）和微调（FT）技术来减轻偏见，同时保持模型的生成质量。

**📊 数据集**

使用了两个代表性的OLLMs（Janus-Pro-7B和BAGEL-7B-MoT），并在多个数据集上进行了评估，包括扩展的WinoBias数据集和CMSC数据集。

**📈 对比分析**

与多种基线方法进行了比较，OmniDebias在所有评估的设置中一致提高了公平性，尤其是在处理给定和未见的敏感属性时，表现优于传统的属性中心方法。

**⚠️ 局限性**

限制在于尽管OmniDebias在多个属性和模态上表现良好，但仍需进一步研究如何在更复杂的社会概念中实现更好的公平性，并探讨其在实际应用中的可扩展性。

---

## 234. A Full-Ledger Counterfactual Audit of Adaptive Reasoning

**arXiv ID:** 2611.22778 | [PDF](https://arxiv.org/pdf/2611.22778v1)

**作者:** Xiaopeng Chu `[一作]` (University of Science and Technology of China), Wenyi Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种全账本反事实价值审计方法，比较立即回答与从相同冻结前缀继续推理的四个固定长度的结果，并考虑每个信号的成本。

**💡 创新点**

创新点在于将当前回答的正确性与继续推理的价值进行量化，并通过成本核算来评估推理的有效性。

**🔧 技术方法**

使用了全账本反事实价值审计技术，结合了测量、预测和计算分配的分解方法。

**📊 数据集**

使用了GSM8K和MATH-500数据集进行评估。

**📈 对比分析**

与当前错误基线进行比较，发现预测哪些错误可以修复的额外价值有限，轻量级控制器主要识别当前错误，增加的价值不明显。

**⚠️ 局限性**

局限性在于研究仅评估了三种开放权重模型在数学任务上的表现，且继续推理的价值依赖于采样和答案生成、解析及评分的方式。

---

## 235. Unified Residual Translation for Paired Image Restoration

**arXiv ID:** 2611.22655 | [PDF](https://arxiv.org/pdf/2611.22655v1)

**作者:** Joyanta Jyoti Mondal `[一作]` (University of Delaware), Md. Shifatul Ahsan Apurba `[通讯]` (University of Alabama at Birmingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种统一的残差翻译器（URT），用于配对图像恢复，能够处理不同的成像对比度、传感器模态和空间分辨率。

**💡 创新点**

URT通过将固定的通道和尺度映射与学习的残差预测分开，提供了一种通用的残差公式，能够有效处理异构配对成像任务。

**🔧 技术方法**

使用了NAFNet编码器-解码器架构来预测剩余的修正，并结合了确定性的参考映射。

**📊 数据集**

使用了IXI、LLVIP、SEN1-2和RealSR等多个数据集进行评估。

**📈 对比分析**

URT在六个任务和尺度设置中，PSNR和SSIM的表现优于评估的基线模型，尤其在IXI和RealSR ×3和×4上表现最佳。

**⚠️ 局限性**

URT的局限性在于其对不同任务的适应性可能受到固定参考和学习残差的设计选择的影响，未来的工作可以探索跨数据集的联合训练。

---

## 236. Errpedia: Is Your Exact $k$-NN Vector Search Really Exact?

**arXiv ID:** 2611.22474 | [PDF](https://arxiv.org/pdf/2611.22474v1)

**作者:** Yinuo Deng `[一作]` (University of New South Wales), Wenjie Zhang `[通讯]` (University of New South Wales)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Errpedia，一个用于测试精确向量搜索数值正确性的数据集，包含1616个查询和21个错误类别。

**💡 创新点**

Errpedia是首个专门用于评估精确向量搜索数值正确性的数据集，提供了独立验证的真实结果。

**🔧 技术方法**

使用了精确有理算术来验证每个查询的真实结果，并评估了20种广泛使用的库和数据库配置。

**📊 数据集**

数据集包含22501个向量和1616个查询，所有坐标均为有限的32位浮点值。

**📈 对比分析**

评估了20种库和数据库配置，发现每种配置在至少一个适用查询上返回了数学上不正确的前10个结果集，性能差异显著。

**⚠️ 局限性**

Errpedia的主要限制是仅覆盖平方欧几里得距离，并使用固定参数d=128和k=10，未来工作将扩展到其他距离度量和参数设置。

---

## 237. RTLReviewBench: LLM Hardware Reviewers Rarely Miss Bugs but Routinely Reject Correct RTL

**arXiv ID:** 2611.21678 | [PDF](https://arxiv.org/pdf/2611.21678v1)

**作者:** Yasir Mahmud `[一作]` `[通讯]` (Florida), Yasir Mahmud (Florida)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基准测试，评估大型语言模型（LLMs）在不进行仿真的情况下对RTL代码的审查能力，包含114对匹配的错误和正确的Verilog模块。

**💡 创新点**

创新点在于引入了一个基于仿真的RTL审查基准，能够测量模型在判断代码时的错误方向，并且揭示了模型在接受错误代码和拒绝正确代码方面的强烈不对称性。

**🔧 技术方法**

使用了大型语言模型（LLMs）进行代码审查，并进行了控制实验以评估不同提示和输出顺序对模型性能的影响。

**📊 数据集**

使用了114对Verilog模块，其中大部分是LLM代码生成器的真实功能失败，部分是手动编写的错误，所有标签均通过仿真确认。

**📈 对比分析**

与12个商业和开放模型进行比较，发现大多数模型在漏判错误时的比例较低（最多3.5%），但错误拒绝率从2%到59%不等，表明错误拒绝是主要问题。

**⚠️ 局限性**

限制在于基准测试主要集中在单个模块的审查，可能无法完全代表更复杂的设计场景，并且模型的表现可能会随着更新而变化。

---

## 238. Learning Stratified Policies for Inference-Time Objectives

**arXiv ID:** 2611.23154 | [PDF](https://arxiv.org/pdf/2611.23154v1)

**作者:** Mridul Mahajan `[一作]` (Boston University), Aldo Pacchiano `[通讯]` (Broad Institute of MIT and Harvard)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种分层策略，通过在推理时在多个学习策略之间分配样本预算来提高大型语言模型的性能。

**💡 创新点**

创新点在于引入了分层策略，使每个学习策略贡献一个响应，从而优化推理时间的目标，而不是从单一策略中重复采样。

**🔧 技术方法**

使用了低秩适配器（LoRA）和共享语言模型骨干进行策略的实现，并通过群体相对策略优化（GRPO）进行训练。

**📊 数据集**

在多个推理任务上进行了评估，包括有效的算术表达构造、逻辑推理、约束满足和数学文字问题。

**📈 对比分析**

与单一策略基线相比，分层策略在多个推理任务上持续提高了pass@k的性能。例如，在数独任务中，pass@5得分从74.2%提高到80.3%。

**⚠️ 局限性**

限制在于该方法的理论分析主要基于线性设置，尚未对非线性语言模型或学习的策略群体提供保证。

---

## 239. The Between-Round Decision in Autonomous Research: An LLM Controller That Reads the Whole Trajectory

**arXiv ID:** 2611.22661 | [PDF](https://arxiv.org/pdf/2611.22661v1)

**作者:** Caoxin Wang `[一作]` (University of Electronic Science and Technology of China), Lili Pan `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为BRED的基于LLM的控制器，用于在自主研究中进行轮间决策，保持最佳验证解决方案（冠军）和过去轮次的日志，并在每轮中做出一个决策。

**💡 创新点**

BRED通过仅依赖累积证据而非搜索结构来做出决策，展示了在不同历史上下文中性能的单调提升，表明可以用更少的机械装置来实现轮间决策。

**🔧 技术方法**

使用了大型语言模型（LLM）作为控制器，并结合了普通的检索、想法生成和编码组件。

**📊 数据集**

在三个广泛使用的研究代理基准任务上进行评估，包括ETTh1（时间序列预测）、SST-2（情感分类）和CIFAR-100（图像分类），以及一个新的任务DAIC-WOZ（基于语音的抑郁检测）。

**📈 对比分析**

与现有系统相比，BRED在所有四个任务上均表现优异，ETTh1的MAE为0.4172，SST-2的准确率为95.07%，CIFAR-100的准确率为84.63%。在没有历史信息的情况下，性能较差，MAE为0.4299，准确率为83.71%。

**⚠️ 局限性**

所有实验均使用单一的控制器LLM，未测量不同模型决策质量的变化；仅在两个任务上进行了消融研究，未能分解累积证据中哪个组件贡献最大。

---

## 240. From Ideal to Real: Integrating Language Models into Smart Home for Human-Centric IoT Safety and Security Fuzzing

**arXiv ID:** 2611.22457 | [PDF](https://arxiv.org/pdf/2611.22457v1)

**作者:** Lin Li `[一作]` (Southern Cross University), Jun Zhang `[通讯]` (Swinburne University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本论文提出了一种将大型语言模型（LLM）集成到智能家居模糊测试中的新方法，以模拟人类决策过程，从而揭示智能家居系统中的潜在安全漏洞。

**💡 创新点**

创新点在于首次将大型语言模型应用于模糊测试框架，模拟人类行为的复杂性和不可预测性，从而提供更全面的物联网安全漏洞评估。

**🔧 技术方法**

使用了大型语言模型（如Vicuna-13B、Vicuna-7B和Llama2-Chat-13B）以及IoTFuzz工具来模拟智能家居环境中的用户交互。

**📊 数据集**

使用了2018年至2023年的真实天气数据集，以模拟外部环境对智能家居设备行为的影响。

**📈 对比分析**

与传统方法相比，集成LLM后，智能家居系统的策略违规概率显著增加，表明LLM能够更好地模拟人类交互的复杂性，揭示潜在的安全漏洞。

**⚠️ 局限性**

本研究的局限性包括LLM输出的幻觉现象（生成不准确或不现实的文本），以及环境建模仅限于单个房间和有限的智能设备，未来工作将扩展到多房间环境和更多物理通道的模拟。

---

## 241. The Illusion of Grounding: Do Vision-Language Models Weigh Retrieved Text and Images Like an Ideal Observer?

**arXiv ID:** 2611.23127 | [PDF](https://arxiv.org/pdf/2611.23127v1)

**作者:** Basavaraj R M `[一作]` (AI and ML Systems Researcher), Harshal Chandel `[通讯]` (eClerx)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了GIB-200K数据集，包含200,000个图像-问题项，评估视觉-语言模型在不同压力条件下的表现。

**💡 创新点**

创新点在于定义了知识惯性和感官捕获，并通过理想观察者分析模型如何结合文本和图像证据。

**🔧 技术方法**

使用了测量误差修正回归、反事实翻转分解和基础幻觉评分等技术。

**📊 数据集**

使用了COCO和Visual Genome数据集构建的GIB-200K数据集。

**📈 对比分析**

与四个开放模型进行比较，发现模型在干净输入下表现良好，但在压力条件下准确率显著下降，尤其是在检索噪声影响下。

**⚠️ 局限性**

局限性包括只评估了一个种子2,000项的子集，未进行人类基线比较，且注意力仅为相关信号。

---

## 242. Flexibility Has a Price: Quantifying Modality Policies in Maliciously Secure Multi-Biometric Authentication

**arXiv ID:** 2611.22714 | [PDF](https://arxiv.org/pdf/2611.22714v1)

**作者:** Fuyi Wang `[一作]` (RMIT University), Jianying Zhou `[通讯]` (Singapore University of Technology and Design)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文研究了多模态生物识别认证中的模态灵活性，从安全角度出发，提出了一种风险分析框架和政策选择方法，以平衡安全性和可用性要求，并展示了一个恶意安全的双服务器认证系统SUPERMBA，能够在不泄露生物特征模板或认证尝试中使用的模态子集的情况下，私密地执行灵活的认证政策。

**💡 创新点**

创新点在于提出了一种量化模态灵活性对认证安全性的影响的方法，并开发了一个政策选择程序，以识别满足目标风险水平的政策，同时支持只能提供部分模态的用户。

**🔧 技术方法**

使用了恶意安全的双服务器计算协议、功能秘密共享和认证秘密共享等技术，确保在认证过程中保护用户的生物特征隐私。

**📊 数据集**

使用了MULB+和VidTIMIT两个数据集进行实验，MULB+结合了手部和虹膜图像以及面部图像，VidTIMIT则提供了视频和语音数据。

**📈 对比分析**

通过与现有方法的比较，实验结果表明，允许所有模态子集会显著增加冒充者风险，MULB+数据集的冒充者风险从1.21%增加到4.42%。在VidTIMIT数据集上，增加幅度较小且未能统计显著。SUPERMBA在MULB+上完成认证尝试的时间为6.42毫秒，在VidTIMIT上为6.03毫秒，显示出良好的性能。

**⚠️ 局限性**

限制在于现有的隐私保护生物识别认证系统主要集中在保护模板和匹配分数上，而对模态灵活性下的政策执行缺乏系统性处理。此外，模态灵活性可能会增加认证的风险，尤其是在攻击者能够利用多个模态的情况下。

---

## 243. Teaching Language Models to Narrow Boolean Patent Searches from Search Feedback

**arXiv ID:** 2611.22270 | [PDF](https://arxiv.org/pdf/2611.22270v1)

**作者:** Seiya Kawano `[一作]` (Kyoto Institute of Technology), Koichiro Yoshino `[通讯]` (University of Osaka)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种方法，通过从搜索反馈中学习，教语言模型缩小专利搜索的范围，特别是针对布尔查询的优化。

**💡 创新点**

创新点在于提出了两种学习方式：策略级学习和行级学习，行级学习通过保留引用的缩小信用来加权查询行，从而提高了检索的精确度。

**🔧 技术方法**

使用了语言模型生成器，并结合了策略级学习和行级学习的技术。

**📊 数据集**

使用了38,397份日本专利局的搜索报告，包含425,405条可执行的布尔查询行和109,717个引用的先前艺术文献。

**📈 对比分析**

与其他方法的比较显示，行级信用在筛选指标上优于策略级评分，迭代后将包含引用的中位集合从768减少到50个以下，并且在小于100文档的集合中包含更多引用，尽管召回率较低。

**⚠️ 局限性**

限制在于评估措施仅考虑到达每个引用的最小集合，实际搜索者还会阅读重复出现在多个集合中的文档，此外，缩小信用主要通过文本重构实现，较少使用分类或集合代数。

---

## 244. Synergizing Evidence Retrieval and Utilization for Self-Improving Multimodal Scientific Document Reasoning

**arXiv ID:** 2611.22837 | [PDF](https://arxiv.org/pdf/2611.22837v1)

**作者:** Frank Nie `[一作]` (Shandong University), Jindong Han `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于多模态科学文档推理的Retriever-Perceiver-Reasoner框架，该框架通过协调证据获取和利用来提高推理能力。

**💡 创新点**

创新点在于引入了诊断引导的共同进化训练策略，结合了自适应证据检索、按需潜在感知和基于原始检索证据的推理。

**🔧 技术方法**

使用了Retriever、Perceiver和Reasoner三个组件，通过共同进化的训练策略进行优化。

**📊 数据集**

在四个基准数据集上进行评估，包括ChartQA、CharXiv-D/R、SPIQA-A/B/C和SciMDR-Eval。

**📈 对比分析**

与多个基线模型进行比较，使用7B和8B的Reasoner骨干网络，分别获得69.9和75.0的平均分，超出最强基线4.0和9.1分。

**⚠️ 局限性**

限制在于当前框架可能在某些情况下无法有效整合所有模态的信息，且对潜在引导的依赖可能导致性能波动。

---

## 245. Literary Language Mashup: Curating Fictions with Large Language Models

**arXiv ID:** 2611.21649 | [PDF](https://arxiv.org/pdf/2611.21649v1)

**作者:** Manzanarez Gerardo Aleman `[一作]`, Calvo Hiram `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大型语言模型（LLMs）在评估文学微小说质量方面的有效性，提出了一种新的评估方法，结合了人类评审和LLMs的评估结果。

**💡 创新点**

创新点在于将LLMs作为文学评审工具，提供了一种自动化的评估方法，旨在补充而非替代人类的判断，探索LLMs在文学评估中的潜力。

**🔧 技术方法**

使用了大型语言模型（LLMs）作为评审工具，结合了GrAImes和Torrance Test of Creative Writing (TTCW)两种评估协议。

**📊 数据集**

使用了由人类和人工智能创作的微小说数据集，这些文本均经过人类评审者的评估。

**📈 对比分析**

通过比较人类评审者和LLMs的评分，发现LLMs在某些维度上与人类评审者的评估存在一定的一致性，但在评估叙事可信度和细微语境方面表现出局限性。

**⚠️ 局限性**

限制在于LLMs的语言和文化多样性不足，可能会延续训练数据中的偏见，且在处理复杂文本时存在数据长度限制和上下文漂移的问题。

---

## 246. Compact and Decodable Electrocardiogram Text for Large Language Models

**arXiv ID:** 2611.22827 | [PDF](https://arxiv.org/pdf/2611.22827v1)

**作者:** Elif Dorukbaşı `[一作]` (Karabük University), İlker Türker `[通讯]` (Karabük University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了一种确定性编码方法，将十二导联心电图（ECG）转化为紧凑且可解码的文本格式，以便于大型语言模型（LLM）的处理。

**💡 创新点**

创新点在于通过心跳周期依赖的容忍度选择关键点，使用26个字母的对数编码表对幅度进行参数化，并通过10步容忍度调度确保所有记录在固定的令牌预算内。

**🔧 技术方法**

使用了符号时间序列表示、分词、损失信号压缩等技术。

**📊 数据集**

使用了公共PTB-XL数据库中的21,388条记录进行实验。

**📈 对比分析**

通过精确的解码和重新编码回合、波形重建误差以及使用5个不同种子的卷积分类器进行配对比较，结果显示解码信号的平均宏AUROC为0.9157，未显著低于参考信号的0.9167，且在预设的非劣性边界内。

**⚠️ 局限性**

主要限制在于测试折叠在开发过程中部分暴露，可能影响结果的独立性。

---

## 247. Reconfiguring Procedural Knowledge for Compositional Robot Skill Adaptation

**arXiv ID:** 2611.22174 | [PDF](https://arxiv.org/pdf/2611.22174v1)

**作者:** Daehee Lee `[一作]` (Sungkyunkwan University), Honguk Woo `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种新的框架，称为Reconfigurable Skill Policy（RSP），用于在稀疏的任务和程序信息下，通过重组现有技能组件来实现新的行为。

**💡 创新点**

创新点在于提出了一种多视图更新与组合的框架，能够在不进行任务特定再训练的情况下，从稀疏的任务证据中生成任务特定的可执行技能序列。

**🔧 技术方法**

使用了多视图程序知识库和任务物化技术，结合了可编辑的程序和任务记忆。

**📊 数据集**

在扩展的Franka Kitchen基准上进行了评估，该基准包含长时间跨度的任务，要求组合超出存储技能记录的能力。

**📈 对比分析**

与基线方法相比，RSP在91.9%的有序完成率下，每集任务推理时间为1.0毫秒，表现优于检索和技能链基线。

**⚠️ 局限性**

局限性在于假设存在能够揭示任务相关技能效果及其连接性的视图，视图分解不足可能会限制可重构性。

---

## 248. IntentGR: Enhancing Intent for End-to-End Personalized Generative Retrieval in Search

**arXiv ID:** 2611.22788 | [PDF](https://arxiv.org/pdf/2611.22788v1)

**作者:** Gefan Ye `[一作]` (Zhejiang University), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commercial Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为IntentGR的意图增强的端到端生成检索框架，旨在通过四个互补阶段（输入、优化、训练分布和解码）增强查询语义，以提高个性化搜索的效果。

**💡 创新点**

创新点在于引入了查询相关的行为证据，并通过四个机制（QBR、QSA、LQE和QAM）来优化和利用这些证据，从而增强检索系统的意图感知能力。

**🔧 技术方法**

使用了端到端生成检索技术，结合了查询感知的语义行为表示、查询语义对齐、长尾查询丰富和查询条件注意力调制等技术。

**📊 数据集**

使用了两个数据集进行实验：AOL4PS（一个公共基准数据集）和AE（一个大型工业电子商务数据集）。

**📈 对比分析**

与多个基线方法进行比较，IntentGR在所有指标上均表现优异，Recall@10分别提高了8.11%和11.96%，尤其在长尾查询上表现显著。

**⚠️ 局限性**

限制在于该方法可能对查询相关行为证据的依赖性较强，且在处理极端长尾查询时的表现仍需进一步验证。

---

## 249. The Hidden Cost of MoE Diffusion Decoding: Profiling and Fixing the Sampler

**arXiv ID:** 2611.21852 | [PDF](https://arxiv.org/pdf/2611.21852v1)

**作者:** Harshit Samrat `[一作]` (Indian Institute of Technology Madras), Ramakrishna Paramahamsa Kompella `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了掩蔽扩散语言模型（dLLMs）的解码过程，重点分析了在带宽受限的边缘加速器上，26B参数的扩散混合专家模型的单用户解码性能。

**💡 创新点**

创新点在于识别并优化了解码过程中最大的时间消耗部分，即logits管道，通过两种无损优化方法显著提高了解码速度，同时保持了基准准确性。

**🔧 技术方法**

使用了掩蔽扩散混合专家模型（MoE），并通过量化技术（W4A16）和融合的单次熵核及Gumbel-max采样技术进行优化。

**📊 数据集**

使用了GSM8K、HumanEval和MATH数据集进行性能评估。

**📈 对比分析**

与基线模型相比，经过优化的模型在不改变准确性的情况下，实现了2.4-2.7倍的端到端速度提升。性能比较显示，优化后的模型在多个基准测试中表现优异。

**⚠️ 局限性**

限制在于研究仅在单用户、批量大小为1的情况下进行，且未测量在数据中心GPU上的表现，批处理对解码时间的影响尚未明确。

---

## 250. Weight-space programming for tractor guidance controller: large language models natively write neural network weights that steer a real tractor

**arXiv ID:** 2611.21958 | [PDF](https://arxiv.org/pdf/2611.21958v1)

**作者:** Timo Oksanen `[一作]` `[通讯]` (Technical University of Munich), Timo Oksanen (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本研究提出了一种新的控制器设计方法，称为权重空间编程，利用大型语言模型直接读取和写入神经网络的权重，以实现自动拖拉机引导。

**💡 创新点**

创新点在于首次将权重空间工程应用于农业工程，展示了大型语言模型在没有梯度下降的情况下，能够设计和修复控制器的能力。

**🔧 技术方法**

使用了大型语言模型（如GPT-6 Astra和Claude Fable 5.1）进行权重空间编程，直接操作神经网络的权重。

**📊 数据集**

研究使用了AMX G-trac拖拉机作为目标机器，并通过模拟生成了多个路径数据集进行训练和测试。

**📈 对比分析**

与进化策略、梯度下降和纯追踪方法进行了比较，结果显示大型语言模型在修复和从零开始设计控制器时表现良好，且在实际拖拉机上运行时，误差与纯追踪方法相当。

**⚠️ 局限性**

本研究的局限性包括只进行了单次会话，没有进行统计测试，且未与所有现有的引导控制器进行基准比较。

---

## 251. SlopeGuard: Towards Generalizable AI-Generated Image Detection via Radial Spectral Slope Guidance

**arXiv ID:** 2611.22727 | [PDF](https://arxiv.org/pdf/2611.22727v1)

**作者:** Cheng Fu `[一作]` (Changsha University of Science and Technology), Wenzheng Liu `[通讯]` (Changsha University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为SlopeGuard的框架，用于检测AI生成的图像，结合了径向谱斜率和语义特征，以提高检测的泛化能力。

**💡 创新点**

创新点在于引入径向谱斜率作为补充的频谱证据，强调其在不同生成器之间的分布差异，并结合语义表示进行更稳健的检测。

**🔧 技术方法**

使用了径向谱斜率作为特征描述符，并结合了LoRA适配的CLIP特征进行特征学习和分类。

**📊 数据集**

使用了AIGCDetectBench、GenImage和AIGIBench等多个数据集进行实验，包含144K个AI生成的图像和相同数量的真实图像。

**📈 对比分析**

与现有方法相比，SlopeGuard在AIGCDetectBench上达到了97.5%的平均准确率，在GenImage上达到了97.8%，在AIGIBench上达到了85.1%，在模糊、裁剪、JPEG压缩和加性噪声等常见图像扰动下表现出竞争力。

**⚠️ 局限性**

限制在于径向谱斜率作为单独的决策标准不足，且在某些情况下可能无法完全抑制生成器特定的表示漂移。

---

## 252. Multistage Contextual Optimization for Vehicle Repositioning

**arXiv ID:** 2611.22330 | [PDF](https://arxiv.org/pdf/2611.22330v1)

**作者:** Jinyu Zhang `[一作]` (Southeast University), Nikolas Geroliminis `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了多阶段上下文优化在拥堵感知车辆重新定位中的应用，提出了一种滚动承诺-补救政策，先进行重新定位决策，再根据实际需求进行服务决策。

**💡 创新点**

创新点在于将多阶段上下文优化与宏观基本图（MFD）结合，提出了一种新的训练需求预测模型，该模型利用早期决策留下的状态作为上下文，并通过决策成本进行训练。

**🔧 技术方法**

使用了多阶段线性规划（LP）和上下文优化技术，推导了精确的梯度和等效的伴随方法，以避免完整的雅可比矩阵。

**📊 数据集**

使用了纽约市的打车数据集，数据覆盖了2026年1月到4月的时间段，分为训练、验证和测试集。

**📈 对比分析**

与五种基准方法进行比较，结果显示多阶段上下文优化在每个测试日上比通过均方误差训练的相同预测器多出7.69%到9.06%的商业收益，且在等待时间和请求完成率上表现更优。

**⚠️ 局限性**

限制在于需求被视为外生变量，未考虑车辆分布对需求的影响，未来研究可以探索如何在决策依赖的需求下进行学习。

---

## 253. Done, But Still Working: Anytime Evaluation of When Coding Agents Should Stop

**arXiv ID:** 2611.22901 | [PDF](https://arxiv.org/pdf/2611.22901v1)

**作者:** Zerui Yang `[一作]` (Huawei Technologies Ltd.), Bo Bai `[通讯]` (Huawei Technologies Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了编码代理在自测通过后继续工作的情况，探讨了这种额外工作是否能改善补丁，以及提前停止能节省多少成本。

**💡 创新点**

提出了一种随时评估的方法，通过重放记录的运行、对中间补丁进行隐藏测试评分，并测量每一步的成本，比较了不同停止策略的效果。

**🔧 技术方法**

使用了随时评估技术，通过重放记录的命令并在可执行沙箱中评估中间补丁，结合隐藏测试进行评分。

**📊 数据集**

使用了SWE-bench Verified任务中的约6000次运行数据，涉及八个开放模型和两个框架。

**📈 对比分析**

通过配对比较不同停止策略的效果，发现停止在自测通过后可以节省25%到57%的成本，具体取决于模型和框架。预注册的复制实验显示，停止在同一补丁通过两次后可以节省23%。

**⚠️ 局限性**

限制在于未能验证所有中间状态的准确性，且某些模型在测试通过后补丁变化频率较低，导致评估结果的可靠性受到影响。

---

## 254. EvoSkill-S1: Evolving Reusable Skills with Genetic-Inspired Operators

**arXiv ID:** 2611.22610 | [PDF](https://arxiv.org/pdf/2611.22610v1)

**作者:** Yujun Wang `[一作]` (LMU Munich), Yunpu Ma `[通讯]` (LMU Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于强化学习的框架，直接演化可重用技能，通过遗传启发的操作符将现有技能转化为候选子技能。

**💡 创新点**

创新点在于将可重用技能作为演化的直接对象，而不是通过间接方式更新技能库，采用五种遗传启发的操作符进行技能演化。

**🔧 技术方法**

使用了强化学习（RL）技术来训练技能演化器，并结合遗传启发的操作符进行技能的直接演化。

**📊 数据集**

使用了来自数学和代码基准的硬负样本数据集，包括大约4000个数学问题和2200个代码问题。

**📈 对比分析**

与没有直接技能演化的最强外部基线相比，提出的方法在所有报告的数学和代码基准上均表现出显著的性能提升，最大相对改进达到20.39%。

**⚠️ 局限性**

限制在于演化的技能可能保留特定于源问题的细节，这可能会影响其在新问题上的适用性。

---

## 255. Any2Building: Unifying Point Clouds and Images for Building Reconstruction via Shared Spatial Queries

**arXiv ID:** 2611.23187 | [PDF](https://arxiv.org/pdf/2611.23187v1)

**作者:** Hongye Hou `[一作]` (Hong Kong University of Science and Technology), Wufan Zhao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为Any2Building的统一模型，用于从部分点云、航拍图像或两者的组合中重建LoD2建筑。

**💡 创新点**

创新点在于引入了共享空间查询的条件模块，能够将不同的观察数据与建筑几何对齐，并提供空间指导，从而促进跨任务的结构学习。

**🔧 技术方法**

使用了基于共享空间查询的条件模块和流匹配的生成模型（DiT），通过交叉注意力机制生成建筑结构。

**📊 数据集**

使用了BuildingWorld-Image数据集，该数据集包含约一百万对齐的部分点云、航拍图像和LoD2建筑模型，覆盖18个城市。

**📈 对比分析**

与现有方法相比，Any2Building在几何重建精度上表现优越，尤其是在单模态和多模态点云补全、单图像建筑生成和先验辅助生成任务中，能够在输入模态受限时仍保持详细和稳健的恢复。

**⚠️ 局限性**

限制在于模型对输入数据的依赖性，尤其是在输入模态稀疏或不完整时，可能会影响重建的精度和质量。

---

## 256. Plausible Videos Can Lead to Poor Decisions: Diagnosing and CURE-ing World Models for Planning

**arXiv ID:** 2611.22561 | [PDF](https://arxiv.org/pdf/2611.22561v1)

**作者:** Raphael C. Kim `[一作]`, Ramin Zabih `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种证书引导的技术（Certify, Upweight, Rollout, Explore），用于评估世界模型在规划中的可信度，并决定在不可信时的应对措施。

**💡 创新点**

创新点在于引入了一个理论上有依据的证书，用于判断何时可以信任世界模型进行规划，并提供了相应的修正措施。

**🔧 技术方法**

使用了证书引导的技术，结合了模型误差和训练数据覆盖度的估计。

**📊 数据集**

在多个领域进行实验，包括基于PhyWorld的模拟物理、MuJoCo臂控制和语言表推理等。

**📈 对比分析**

通过证书评估决策质量，结果显示，证书信任的决策的后悔值（regret）显著低于不被信任的决策（例如，语言表模拟中的0.06对比0.38），而视频感知质量并不能有效识别良好决策。

**⚠️ 局限性**

限制在于模型可能在某些情况下仍然无法准确预测决策，尤其是在训练数据覆盖不足的情况下。

---

## 257. Fast Training-Free Low-Light Video Enhancement with a Frozen Image Diffusion Prior

**arXiv ID:** 2611.22551 | [PDF](https://arxiv.org/pdf/2611.22551v1)

**作者:** Hon Ming Yam `[一作]` (City University of Hong Kong), Chun Pong Lau `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无训练的低光视频增强管道，利用冻结的图像扩散模型和两个时间组件来提高低光视频的质量。

**💡 创新点**

创新点在于提出了时间输入融合和延迟开始偏移传播两个组件，使得在不需要训练数据的情况下，显著提高了处理速度和视频质量。

**🔧 技术方法**

使用了冻结的图像扩散模型（MR. Illuminate）和运动补偿递归滤波技术。

**📊 数据集**

使用了SDSD数据集，该数据集提供了动态场景的低光和正常光视频配对。

**📈 对比分析**

与现有的ZVRD方法相比，该管道在256×256分辨率下速度提高了261倍，并在所有五个评估指标上表现更好。

**⚠️ 局限性**

限制在于感知增益在低于1280×720的分辨率下无法转移，且所有常数都是在同一数据集的剪辑上选择的。

---

## 258. EvoDisco: Strategy Discovery for Evolving Agent Memory

**arXiv ID:** 2611.22102 | [PDF](https://arxiv.org/pdf/2611.22102v1)

**作者:** Mingyang Liu `[一作]` (City University of Hong Kong), Linqi Song `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为EvoDisco的框架，用于通过进化策略发现来演化代理的记忆，而不是依赖固定的操作调度。

**💡 创新点**

创新点在于EvoDisco允许代理自主发现何时应用何种操作，并如何实例化这些操作，从而动态演化记忆结构。

**🔧 技术方法**

使用了双层适应度框架来评估记忆更新的检索质量变化和结构复杂性，同时结合历史决策的有效性。

**📊 数据集**

在ALFWorld、WebShop和LifelongAgentBench（OS和DB）数据集上进行了实验。

**📈 对比分析**

与固定策略和记忆基线进行比较，EvoDisco在所有四个基准上都取得了最高的在线和转移成功率，尤其在ALFWorld上，在线成功率提高了8.8个百分点，转移成功率提高了12.9个百分点。

**⚠️ 局限性**

限制在于操作库固定为四个操作，演化触发机制是预定的或启发式的，而不是学习得到的，且评估仅使用单一的基础模型和ALFWorld作为转移源。

---

## 259. Zero-Dimensional Attention Topology for Linguistic Acceptability

**arXiv ID:** 2611.22312 | [PDF](https://arxiv.org/pdf/2611.22312v1)

**作者:** Saeed AlMarri `[一作]` `[通讯]` (Khalifa University of Science and Technology), Saeed AlMarri (Khalifa University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了一种基于零维持久同调（H_0）表示的拓扑总结方法，该方法从144个对称化的BERT注意力图中提取特征，用于语言可接受性分类。

**💡 创新点**

创新点在于使用固定的H_0表示来总结BERT注意力图的最小生成树合并值，并评估其在语言可接受性分类中的表现。

**🔧 技术方法**

使用了零维持久同调（H_0）表示和BERT模型，结合了注意力统计特征和长度控制特征。

**📊 数据集**

使用了来自语言可接受性语料库（CoLA）的8517个训练句子，以及521个领域内和497个领域外的验证句子。

**📈 对比分析**

与BERT模型和简单的注意力统计特征相比，H_0表示在领域外的马修斯相关系数（MCC）为0.438，低于注意力统计的0.508和BERT的0.545，未显示出预测优势。

**⚠️ 局限性**

限制在于该研究仅基于一个英语数据集和一个经过监督的BERT检查点，未考虑其他编码器和更复杂的拓扑表示，且未评估因果可解释性和计算效率。

---

## 260. Asking What Matters: Disagreement-Guided Ambiguity Detection for Interactive Text-to-SQL

**arXiv ID:** 2611.21645 | [PDF](https://arxiv.org/pdf/2611.21645v1)

**作者:** Su Dong `[一作]` (Hong Kong Polytechnic University), Xiao Huang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 47379 | [OpenAlex ID](https://openalex.org/A5059607545)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为DisGuAD的框架，用于在交互式文本到SQL生成中选择澄清目标，以识别影响SQL的模糊性。

**💡 创新点**

创新点在于通过SQL解释之间的分歧来指导澄清选择，从而提高澄清的质量和任务成功率。

**🔧 技术方法**

使用了对比SQL解释探针、组件级分歧分析和用户面向的模糊性归因等技术。

**📊 数据集**

使用了BIRD-Interact Lite和Full数据集进行实验。

**📈 对比分析**

与原始交互策略相比，DisGuAD在任务成功率和澄清质量上均有显著提升，规范化奖励提高了4.40至8.97分，关键模糊性覆盖率从48.62%提高到82.75%。

**⚠️ 局限性**

限制在于DisGuAD仍依赖于LLM生成的SQL解释，可能在某些情况下无法完全捕捉所有模糊性。

---

## 261. Reliability-Driven Feature Correction for Hyperspectral and LiDAR Classification

**arXiv ID:** 2611.22508 | [PDF](https://arxiv.org/pdf/2611.22508v1)

**作者:** Lianming Zhang `[一作]` (Hunan Normal University), Xudong Kang `[通讯]` (Hunan Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于可靠性的特征修正框架ReDF，用于高光谱图像和LiDAR数据的分类，旨在量化每种模态特征在每个空间位置的可靠性，并指导特征修正和融合。

**💡 创新点**

创新点在于通过类原型库引入的可靠性引导特征修正模块和可靠性引导特征重加权模块，能够有效地调整模态特征的贡献，提高分类性能。

**🔧 技术方法**

使用了深度学习技术，特别是卷积神经网络（CNN）和基于原型的证据估计方法。

**📊 数据集**

使用了三个公共的高光谱图像和LiDAR数据集，包括Houston2013、MUUFL和Trento。

**📈 对比分析**

与其他方法相比，ReDF在三个数据集上都表现出更高的平均整体准确率和Kappa值，显示出其在分类任务中的优势。

**⚠️ 局限性**

限制在于ReDF在某些类别（如MUUFL中的人行道和黄色路缘石）的分类效果较差，未来的工作将探讨在包含多个类别的补丁中进行局部可靠性估计和特征修正。

---

## 262. Beyond Episodic Recall: Experience-to-Knowledge Evolution for Lifelong Embodied Navigation

**arXiv ID:** 2611.22321 | [PDF](https://arxiv.org/pdf/2611.22321v1)

**作者:** Weitao An `[一作]` (Xidian University), Cheng Deng `[通讯]` (Xidian University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为Experience-Evolving Memory (EvoNav) 的新型两时间尺度记忆框架，用于终身体态导航，旨在通过持续交互积累经验并将其用于后续目标搜索。

**💡 创新点**

创新点在于引入稀疏经验原子，能够有效地编码导航相关的语义、空间和反馈线索，同时协调在线经验整合与环境特定结构知识的演变。

**🔧 技术方法**

使用了两时间尺度的记忆学习范式，结合快速的在线整合与缓慢的环境特定知识演变，采用了Atom-Graph Evidence Propagation (AGEP) 技术进行经验图推理和导航。

**📊 数据集**

在HM3D数据集上进行了实验，涉及单目标、顺序多层和多模态导航，验证了经验重用的有效性和路径效率的提升。

**📈 对比分析**

与现有方法相比，EvoNav在路径效率和目标执行方面表现出色，尤其在多层导航任务中，相较于最强基线LifelongCrossNav，EvoNav在序列成功率和路径效率上均有显著提升。

**⚠️ 局限性**

限制在于当前方法在跨层导航的挑战上仍然存在，尤其是在复杂环境中，可能需要进一步优化以提高跨层目标的成功率。

---

## 263. What MoE Routing Controls: Causal Interventions on Expert Trajectories

**arXiv ID:** 2611.22120 | [PDF](https://arxiv.org/pdf/2611.22120v1)

**作者:** Mukund Athmaram `[一作]` (Cornell University), Liang Ma `[通讯]` (Advanced Micro Devices Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在混合专家（MoE）语言模型中，强制令令牌沿着选定的专家路径的影响，构建了目标路径并在生成过程中对每个目标专家的路由器logit添加固定偏置。

**💡 创新点**

提出了一种路径级别的引导方法，通过偏置路由logits强制令牌沿着跨层专家路径，发现格式路径改变生成文本，而概念路径则主要影响下一个令牌的概率。

**🔧 技术方法**

使用了混合专家（MoE）模型，特别是Qwen3-30B-A3B模型，采用了偏置路由logits的技术。

**📊 数据集**

使用了Qwen3-30B-A3B模型的数据集，并进行了多种概念的实验，包括宗教和金钱等。

**📈 对比分析**

与随机路径进行了比较，发现强制格式路径显著提高了生成文本中的数字比例，而概念路径则在下一个令牌的概率上有显著提升，但生成文本变化不大。

**⚠️ 局限性**

主要局限在于实验结果主要基于单一模型（Qwen3-30B-A3B），并且偏置的强度可能影响效果，未来需要进一步研究自然使用中路径的必要性。

---

## 264. Automated-Truck Penetration Shapes Human Drivers' Car-Following Margins Near Motorway Ramps: A Driving-Simulator Study

**arXiv ID:** 2611.22545 | [PDF](https://arxiv.org/pdf/2611.22545v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 265. From Constraints to Causality: Petri net Encodings for Explaining Binary Neural Network Inference

**arXiv ID:** 2611.22396 | [PDF](https://arxiv.org/pdf/2611.22396v1)

**作者:** Alex Chan `[一作]` (Newcastle University), Rishad Shafik `[通讯]` (Newcastle University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文提出了一种将固定训练的二进制神经网络（BNN）推理语义转换为1安全的Petri网（PN）的方法，以提供BNN决策的因果结构视图。

**💡 创新点**

创新点在于通过离散事件视图来分析BNN的推理过程，揭示了BNN内部计算步骤的因果关系，而不仅仅是输入输出的约束系统。

**🔧 技术方法**

使用了Petri网（PN）作为形式化工具来表示和分析BNN的推理过程。

**📊 数据集**

使用了一个包含多个输入和输出的训练BNN作为示例进行验证，具体数据集未详细说明。

**📈 对比分析**

通过与参考实现的输入输出等价性进行全面的忠实性评估，结果表明PN模型能够准确计算与参考BNN相同的功能，且在性能上优于传统的约束系统分析方法。

**⚠️ 局限性**

限制在于当前方法主要适用于离散值模型，且可扩展性受到可达状态空间的限制，未来需要进一步研究如何处理动态变化的权重和卷积层等复杂结构。

---

## 266. When the tail is earned by its companions: measuring label dependence in long-tailed multi-label classification

**arXiv ID:** 2611.22859 | [PDF](https://arxiv.org/pdf/2611.22859v1)

**作者:** Christoffer Löffler `[一作]` `[通讯]`, Christoffer Löffler

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的测量方法，用于评估长尾多标签分类中的标签依赖性，特别是稀有标签如何依赖于其频繁的伴随标签进行预测。

**💡 创新点**

创新点在于引入了一种标签内估计量和共报告规则，能够在训练前评估标签依赖性，并提供一个训练标签的统计量，界定报告的召回率可能超过独立召回率的程度。

**🔧 技术方法**

使用了标签内效应的计算方法，结合了训练标签的统计分析，评估了不同训练目标和标签相关机制的影响。

**📊 数据集**

使用了多个文本数据集，包括EU法律、PubMed摘要、MEDLINE摘要、维基百科文章和零售分类等，进行了广泛的审计。

**📈 对比分析**

与现有的宏F1等评估方法相比，本文的方法能够揭示标签依赖性，并且在所有审计的数据集中，伴随标签的依赖性普遍存在。没有任何标准的长尾目标能够有效减少这种依赖性。

**⚠️ 局限性**

限制在于该测量仅适用于可估计的标签，且在测试集上，独立出现的标签通常难以定义，导致缺乏信息的标签无法进行有效评估。

---

## 267. Auditing Hyperstimulation at Scale in Child-Facing YouTube Videos

**arXiv ID:** 2611.21637 | [PDF](https://arxiv.org/pdf/2611.21637v1)

**作者:** Shao-Yu Chu `[一作]` (University of California), Haojian Jin `[通讯]` (University of California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究对儿童面向的YouTube视频进行了大规模的超刺激模式审计，识别出12种超刺激模式，并将其转化为可计算的指标，分析了2015年至2026年间的16,375个视频。

**💡 创新点**

创新点在于结合父母的在线讨论和发展心理学，系统性地识别和量化儿童视频中的超刺激模式，并开发了一个公开的审计工具，帮助家长评估视频的刺激水平。

**🔧 技术方法**

使用了计算机视觉和音频分析技术，通过FFmpeg和OpenCV构建了一个开放源代码的计算管道来测量超刺激指标。

**📊 数据集**

数据集包括16,375个2015年至2026年间发布的儿童YouTube视频，以及300个1928年至2014年的历史参考视频。

**📈 对比分析**

通过量化回归模型比较不同视频的刺激水平，发现超刺激模式在时间上普遍增加，且更刺激的视频通常获得更多观看，但不同内容类型的刺激模式有所不同。

**⚠️ 局限性**

限制在于仅研究了YouTube平台，且视频主要为英语，观看体验依赖于观看条件，无法绝对化测量每个观众的刺激水平。

---

## 268. DeltaHarness: Self-Evolving Harnesses for Resource-Adaptive Tool Execution

**arXiv ID:** 2611.21688 | [PDF](https://arxiv.org/pdf/2611.21688v1)

**作者:** Lecheng Yan `[一作]` (University of Science and Technology of China), Chenyang Lyu `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种执行引导的策略搜索方法，通过冷启动测量来适应代理工具的执行环境。

**💡 创新点**

创新点在于利用冷启动测量来更新运行时调度、修订物理计划，并选择后续实验，以提高工具使用代理的执行效率。

**🔧 技术方法**

使用了执行引导的策略搜索方法，结合了测量反馈和物理政策的自我演化。

**📊 数据集**

使用了视频编辑任务作为数据集，通过固定调度和不同的资源条件进行实验。

**📈 对比分析**

与固定计划进行比较，修订后的物理政策在一个视频编辑任务上将完整交付延迟减少了30.0%，在其他任务上也表现出相对较好的延迟减少效果。

**⚠️ 局限性**

限制在于该方法依赖于冷启动测量的准确性，且在不同任务和资源条件下的适应性可能会受到影响。

---

## 269. Transfer Once, Ask Locally: Compact Audio-Visual Memory for On-Device Question Answering

**arXiv ID:** 2611.23085 | [PDF](https://arxiv.org/pdf/2611.23085v1)

**作者:** Qitong Wang `[一作]` (University of Delaware), Christopher Rasmussen `[通讯]` (Dolby Laboratories, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关的、查询无关的令牌修剪框架，用于在轻量级用户设备上实现音视频问答。

**💡 创新点**

创新点在于通过蒙特卡洛夏普利估值来评估视频和音频令牌的重要性，从而在传输前构建紧凑的可重用内存。

**🔧 技术方法**

使用了蒙特卡洛夏普利估值和嵌入覆盖目标来进行令牌选择。

**📊 数据集**

在Video-MME、Daily-Omni和Video-Holmes数据集上进行了实验。

**📈 对比分析**

与DivPrune、VisionZip和OmniZip等现有压缩方法相比，在所有评估设置中，我们的方法在问答准确性上表现最佳或并列最佳，尤其在Daily-Omni数据集上，准确性提高了12.86-14.62个百分点，同时在设备端的LLM预填充和解码延迟上也表现出最低的延迟。

**⚠️ 局限性**

限制在于该方法依赖于固定的令牌预算，可能无法适应所有类型的音视频内容。

---

## 270. Projected Route Cues for Bouldering: Effects of Circular, Linear, and Combined Highlighting on Route Observation and Climbing

**arXiv ID:** 2611.22093 | [PDF](https://arxiv.org/pdf/2611.22093v1)

**作者:** Yuki Uranishi `[一作]` (University of Osaka), Haruo Takemura `[通讯]` (University of Osaka)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究比较了三种投影提示（圆圈、连接持点的彩色线条和两者结合）对攀岩者在攀爬前观察路线的影响。

**💡 创新点**

创新点在于首次系统比较了不同设计的投影提示对攀岩者观察时间和攀爬表现的影响。

**🔧 技术方法**

使用了投影仪-摄像头系统，通过深度传感器将提示准确投影到攀岩墙上。

**📊 数据集**

参与者在四条不同的攀爬路线（均为V0难度）上进行实验，96名参与者中61人为首次攀岩者。

**📈 对比分析**

与无投影条件相比，所有三种提示均缩短了首次尝试前的时间，圆圈提示的几何平均时间从83秒降至39秒。线条提示导致更多使用非路线持点，但在行为上未显示出优于圆圈提示的优势。

**⚠️ 局限性**

限制在于使用单一投影仪可能导致攀爬者的身体投下阴影，且实验在较暗的环境中进行，结果可能不适用于光线充足的商业攀岩馆。

---

## 271. CaFe: Causally Faithful Conditional Traffic Scenario Generation

**arXiv ID:** 2611.22855 | [PDF](https://arxiv.org/pdf/2611.22855v1)

**作者:** Steven Yu Tung `[一作]` (University of Freiburg), Steven Peters `[通讯]` (TU Darmstadt)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文提出了一种新的交通模拟方法CaFe，旨在解决现有模拟器中意图泄漏的问题，确保指令的隐私性，同时保持交通互动的真实性。

**💡 创新点**

创新点在于正式定义了意图泄漏，并引入了一种配对重放测试来隔离意图泄漏与合法反应，同时构建了一个大型可验证的数据集和一个双流模拟器。

**🔧 技术方法**

使用了双流扩散强制模拟器技术，分离了私人意图和公共预测信念，确保了信息流的隐私性。

**📊 数据集**

使用了531,099个WOMD场景的数据集，包含时间行为标签和反应事件的标注。

**📈 对比分析**

与现有方法比较，CaFe在消除可测量的意图泄漏方面表现出色，保持了与单流模型相当的现实性（0.763），并且在多个评估指标上表现优于其他方法。

**⚠️ 局限性**

限制在于评估主要集中在语言条件下，虽然可以为其他私有条件制定相同的信息边界，但这些扩展尚未经过测试。

---

## 272. Virtual Double Oracle: Faster Convergence by Leveraging Strategies That Were Watching All Along

**arXiv ID:** 2611.22806 | [PDF](https://arxiv.org/pdf/2611.22806v1)

**作者:** Ruilan Wang `[一作]` (London School of Economics), Katerina Papadaki `[通讯]` (London School of Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出了一种新的算法Virtual Double Oracle (VDO)和Discounted VDO (dVDO)，用于在两人零和游戏中近似计算纳什均衡，通过利用历史损失信息来优化策略的初始化。

**💡 创新点**

创新点在于识别了重新初始化作为在线学习中的一个重要设计选择，并通过双线性身份来重建新策略的历史权重，从而提高了策略发现的效率和效果。

**🔧 技术方法**

使用了在线学习中的乘法权重更新（MWU）技术，并结合了双线性身份来实现策略的虚拟初始化。

**📊 数据集**

实验使用了高斯矩阵游戏、扑克（Kuhn、Leduc）和Goofspiel等数据集。

**📈 对比分析**

与传统的在线双重Oracle (ODO)和均匀重置方法相比，dVDO在有限视野的可利用性上显著降低，且在发现的策略群体中表现更优。

**⚠️ 局限性**

限制在于该方法假设了两人零和游戏的收益是精确反馈，并且需要固定的学习率，且对噪声收益估计的情况没有覆盖。

---

## 273. M3OPD-Reasoner: An Open Recipe for Post-Training Multimodal Reasoning with Multi-Teacher On-Policy Distillation

**arXiv ID:** 2611.23034 | [PDF](https://arxiv.org/pdf/2611.23034v1)

**作者:** Nguyen Quang Trung `[一作]` (A*STAR), AiTi Aw `[通讯]` (A*STAR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多教师在线蒸馏（MOPD）方法，使单个学生能够通过自身生成的响应吸收多个领域教师的技能，针对图像STEM、图像感知和视频推理三个领域进行后训练的视觉语言模型（VLMs）研究。

**💡 创新点**

创新点在于首次在多教师蒸馏后引入强化学习（MixRL）阶段，利用MOPD提供的更强起点来进一步提升模型性能。

**🔧 技术方法**

使用了多教师在线蒸馏（MOPD）和群体序列策略优化（GSPO）技术。

**📊 数据集**

构建了两个数据集，分别用于监督微调（SFT）和具有可验证奖励的强化学习（RL），总共包含135万样本用于SFT和193K个可验证答案的提示用于RL。

**📈 对比分析**

通过比较不同的后训练配方，发现直接从Qwen3.5-9B训练的GSPO教师的MOPD是最佳配方，学生在各自领域与教师的表现接近，并且在相同提示下平均优于混合训练基线。

**⚠️ 局限性**

限制在于尚未充分探索MOPD与MixRL之间的关系，以及如何在不同领域之间平衡能力的提升。

---

## 274. Perception into Imagination: Executable Memory for World Models

**arXiv ID:** 2611.22294 | [PDF](https://arxiv.org/pdf/2611.22294v1)

**作者:** Xianyao Li `[一作]`, Jing Du `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种可执行的感知记忆（EPM）框架，旨在通过想象过程中的神经潜在动态来增强显式感知状态，从而改善物理智能体在失去可见性或接触后的行为决策。

**💡 创新点**

创新点在于将感知记忆与奖励预测、价值估计和策略学习相结合，使得记忆的事实能够影响决策过程，并通过开放式监督训练奖励头。

**🔧 技术方法**

使用了神经网络和世界模型的结合，特别是在DreamerV3实现中，通过转移对齐的奖励输入和开放式监督来训练。

**📊 数据集**

使用了Memory Maze（9×9和13×13）和MiniWorld数据集进行实验，评估了不同策略的性能。

**📈 对比分析**

与DreamerV3和R2I进行比较，EPM在Memory Maze 13×13中达到30.37的回报，而仅使用策略/价值的接口则为23.37，显示出7.00的增益，且在所有实验中均表现出更强的奖励预期。

**⚠️ 局限性**

限制在于当前框架主要集中于视觉和触觉的感知记忆，未来的扩展可以考虑更复杂的目标变化和接触转变的情况。

---

## 275. Progress Predicts Policy

**arXiv ID:** 2611.22491 | [PDF](https://arxiv.org/pdf/2611.22491v1)

**作者:** Licheng Zhong `[一作]` (National University of Singapore), Gim Hee Lee `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为Progress Predicts Policy (PPP)的方法，通过其微分学习语言条件的标量场，以实现机器人控制。

**💡 创新点**

创新点在于通过模仿学习和双向反向传播训练引导命令，同时使用时间标签校准观察状态的场值，而无需额外注释。

**🔧 技术方法**

使用了模仿学习、双向反向传播和共享世界模型等技术。

**📊 数据集**

在模拟和真实机器人操作中进行了实验，使用了LIBERO数据集和其他真实机器人任务。

**📈 对比分析**

与行为克隆（BC）方法进行了比较，PPP在共享架构下的表现优于BC，尤其在LIBERO任务中表现突出。

**⚠️ 局限性**

限制在于演示相对进度可能基于手臂运动而非任务相关场景变化，且该场没有李雅普诺夫保证，其微分不保证物理进展。

---

## 276. What RL on Expert-Built Tasks Teaches Nemotron 3.5 Super

**arXiv ID:** 2611.22161 | [PDF](https://arxiv.org/pdf/2611.22161v1)

**作者:** Sushant Mehta `[一作]`, Edwin Chen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

对NVIDIA的Nemotron 3.5 Super模型进行了后训练，使用强化学习（RL）在五个专家构建的任务数据集上逐一进行训练，分别是现实世界软件工程、复杂约束、长时间工具使用、专业图表理解和企业文档推理。

**💡 创新点**

创新点在于每个适配器在没有监督阶段和KL惩罚的情况下进行训练，并且通过与基础模型的比较，明确了每个数据集对模型性能的贡献。

**🔧 技术方法**

使用了强化学习（RL）和CISPO技术，训练了新的rank-LoRA适配器。

**📊 数据集**

使用了五个专家构建的任务数据集，包括现实世界软件工程（RL-SWE）、复杂约束（RL-Constraints）、长时间工具使用（RL-ToolUse）、专业图表理解（RL-Charts）和企业文档推理（RL-Docs）。

**📈 对比分析**

通过与基础模型在相同任务和设置下的比较，适配器在十二个基准测试中表现出显著提升，得分提升范围从几个百分点到十几个百分点不等，具体表现为在多个基准测试中适配器的得分明显高于基础模型。

**⚠️ 局限性**

限制在于每个适配器的训练仅基于单一数据集，可能导致在其他未训练领域的泛化能力不足。

---

## 277. From Apology to Action: Apologies, Compensation, and Cooperation in LLM Agents

**arXiv ID:** 2611.21984 | [PDF](https://arxiv.org/pdf/2611.21984v1)

**作者:** Grace Ibukunoluwa Ufeoshi `[一作]` (Teesside University), Zhao Song `[通讯]` (Teesside University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大型语言模型（LLMs）在遭遇不良互动后是否会道歉，以及这些道歉是否与后续的合作行为相关。通过控制的囚徒困境实验，研究了四种商业模型（Claude Opus 4.8、GPT-5.5、Gemini 3.1 Pro 和 Mistral Large 3）的道歉行为及其后续行动。

**💡 创新点**

创新点在于区分了道歉、补偿和战略行为的不同维度，强调了语言信号与后续行为之间可能存在的不一致性。这为机器行为和人机交互的研究提供了重要的启示。

**🔧 技术方法**

使用了控制的囚徒困境实验框架，结合了不同的支付配置和交互历史条件，以系统地评估LLM代理的道歉行为和后续战略决策。

**📊 数据集**

实验中使用了四种大型语言模型（Claude Opus 4.8、GPT-5.5、Gemini 3.1 Pro 和 Mistral Large 3），并在不同的支付配置下进行了多次实验。

**📈 对比分析**

通过比较不同模型在道歉后的合作行为，发现道歉并不总是与后续的合作行为一致。例如，Claude在道歉后可能继续选择背叛，而GPT-5.5和Gemini则在道歉后更倾向于合作。整体性能表现因模型而异。

**⚠️ 局限性**

本研究的局限性在于实验设计允许代理选择自己的道歉和补偿决策，未能独立分配这些决策，因此无法确定信号的因果效应。此外，研究结果的普遍性尚不确定，未来需要在不同模型和策略下进行进一步验证。

---

## 278. CNN Architecture Comparison for Malware Classification with Limited Training Data

**arXiv ID:** 2611.21936 | [PDF](https://arxiv.org/pdf/2611.21936v1)

**作者:** Noor Abo Hasan `[一作]` `[通讯]` (Stockholm University), Noor Abo Hasan (Stockholm University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文研究了在有限训练数据下，单路径和双路径卷积神经网络（CNN）架构在恶意软件分类中的比较性能，特别关注不同训练数据大小对其性能的影响。

**💡 创新点**

创新点在于提出了双路径CNN架构，通过中间特征融合来增强恶意软件分类的效果，并系统比较了在不同数据约束下的性能表现。

**🔧 技术方法**

使用了预训练的VGG16模型作为基线单路径CNN，并结合自定义CNN构建双路径模型，采用中间特征融合技术。

**📊 数据集**

使用了Malimg数据集，该数据集包含来自C2Lop.P（木马）和Dialplatform.B（蠕虫）两个恶意软件家族的样本，经过预处理后转换为灰度图像。

**📈 对比分析**

两种架构在所有训练数据比例下均实现了完美的分类性能（100%准确率），但在训练效率、噪声鲁棒性和架构复杂性等次要指标上存在显著差异。单路径架构在计算效率上表现优越，训练时间比双路径架构少18-51%。

**⚠️ 局限性**

限制在于研究仅针对两个恶意软件家族的二分类任务，未能涵盖更复杂的多类场景，且使用的Malimg数据集可能不完全反映当代恶意软件特征。

---

## 279. Seeing What Still Holds: How Waiting Control Changes Cooperative Recovery

**arXiv ID:** 2611.22368 | [PDF](https://arxiv.org/pdf/2611.22368v1)

**作者:** Kangjie Huang `[一作]` (Shandong Normal University), Shuming Liu `[通讯]` (Shandong Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文研究了在固定等待时间内改变持有参考如何影响视觉恢复方法的排名，提出了一种配对执行和轨迹评估框架，独立变化持有参考和等待时间，同时保持恢复执行器不变。

**💡 创新点**

创新点在于展示了在固定等待时间内，仅通过改变持有参考可以逆转视觉恢复方法的排名，并且通过逐步位置重新锚定显著提高了长时间等待的完成率和减少了恢复时间。

**🔧 技术方法**

使用了配对执行和独立轨迹评估的框架，结合了CAD反馈和基于地标的恢复方法，进行了960次实验。

**📊 数据集**

使用了20个初始化种子进行的960次实验，比较了不同持有条件下的恢复性能。

**📈 对比分析**

通过比较不同持有条件下的保护完成率，发现CAD反馈和地标恢复在不同持有条件下的表现有显著差异，尤其是在位置仅持有条件下，CAD反馈的保护完成率从6/40提升到34/40，逆转了排名。

**⚠️ 局限性**

限制在于实验主要基于模拟环境，可能无法完全反映真实世界中的复杂性和动态变化。

---

## 280. What Does an Agent Need? Bundle-Aware Agent Resource Composition

**arXiv ID:** 2611.22030 | [PDF](https://arxiv.org/pdf/2611.22030v1)

**作者:** Yunxiao Shi `[一作]` (Australian National University), Qingsong Wen `[通讯]` (Squirrel Ai Learning)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

提出了一种任务条件的代理资源组合方法（AgentComposition），旨在选择一个主干模型并组合工具-技能包，以最大化端到端执行效用。

**💡 创新点**

创新点在于提出了一种检索基础的、包感知的代理生成框架，能够从不断变化的资源目录中组合有效配置，并通过适应执行反馈的批评者对其进行排名。

**🔧 技术方法**

使用了检索、约束束搜索和批评者模型等技术，批评者通过代理执行反馈进行训练。

**📊 数据集**

使用了AgentSelect、AgentSelect-Exec、SkillsBench、BFCL和SWE-Skills-Bench等多个数据集进行评估。

**📈 对比分析**

与多种基线方法进行比较，结果显示该方法在67.73%的AgentSelect查询中能够在十个候选中恢复完整的参考配置，几乎是最强基线的两倍，并在SkillsBench上领先所有基线。

**⚠️ 局限性**

限制在于检索的召回率，且批评者在没有执行反馈时依赖代理信号。

---

## 281. Agent Trust Infrastructure (ATI): A Federated Identity, Trust, Attestation, and Accountability Architecture for Cross-Organization Autonomous AI Agents

**arXiv ID:** 2611.23202 | [PDF](https://arxiv.org/pdf/2611.23202v1)

**作者:** Viet K. Nguyen `[一作]` (California State Polytechnic University), Mohammad I. Husain `[通讯]` (California State Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了Agent Trust Infrastructure (ATI)，一个跨组织的自主AI代理的身份、信任、证明和问责架构，旨在解决现有代理框架在跨组织身份验证和授权方面的不足。

**💡 创新点**

ATI的创新点在于其框架独立性和五个互补的安全层次结构，包括身份、信任、证明、政策和问责，能够在不同的代理框架之间建立联邦信任。

**🔧 技术方法**

使用了公钥基础设施（PKI）、零信任架构（ZTA）和远程证明等技术，构建了一个外部安全执行层，与现有代理框架集成。

**📊 数据集**

在一个模拟的八个组织的联邦环境中进行了评估，使用了六个代理框架，每个框架部署了一个五代理团队。

**📈 对比分析**

通过对14种攻击场景和6种良性任务的评估，ATI的攻击控制率从43%提高到84%，而引入的执行开销约为每个请求174微秒。

**⚠️ 局限性**

ATI的局限性包括无法防止提示注入攻击、授权代理之间的共谋行为、缺乏硬件支持的运行时证明、可扩展的证书撤销机制以及隐私保护不足等问题。

---

## 282. Know When to Zoom: How Small Is Too Small for Vision-Language Models?

**arXiv ID:** 2611.22709 | [PDF](https://arxiv.org/pdf/2611.22709v1)

**作者:** Siddharth Vohra `[一作]` (Carnegie Mellon University), Min Xu `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究评估了七种在线视觉-语言模型在处理大图像时对小细节的识别能力，提出了在发送问题前如何判断细节是否会被模型识别的规则。

**💡 创新点**

创新点在于提出了五条规则，指导代理在处理大图像时何时进行缩放或切片，以提高模型的准确性，并发布了眼科考试和模型回答的代码。

**🔧 技术方法**

使用了眼科考试和601个关于小物体的问题，比较了七种模型的表现，包括Claude、Gemini和OpenAI的GPT-6等。

**📊 数据集**

使用了公共基准数据集和研究团队生成的图像，涉及多个真实场景的问题。

**📈 对比分析**

通过比较不同的发送方式（如原图、最高设置、切片等），发现切片可以提高11.1%到14.5%的准确率，而两步缩放在某些情况下的表现与切片相当或更好。

**⚠️ 局限性**

限制在于模型的规则可能会变化，且本研究只收集了每个问题和设置的一个答案，未考虑不同调用之间的变异性。

---

## 283. FastFrames: Video Generation at Interactive Latency

**arXiv ID:** 2611.23226 | [PDF](https://arxiv.org/pdf/2611.23226v1)

**作者:** Gene Chou `[一作]` (Apple), Vladlen Koltun `[通讯]` (Apple)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的自回归视频扩散模型，旨在实现低延迟的实时交互，特别是在每帧生成的延迟低于50毫秒。

**💡 创新点**

创新点在于通过替换传统的VAE解码器和KV缓存更新机制，显著降低了推理延迟，同时保持了生成质量。

**🔧 技术方法**

使用了轻量级的ConvNeXt网络替代VAE解码器，并设计了KV修复网络来加速KV缓存更新，同时进行了系统级优化。

**📊 数据集**

使用了内部数据集，包括10,000个真实视频，并在高端GPU上进行评估，分辨率为480p。

**📈 对比分析**

与现有的基线模型（如Causal Forcing++）进行比较，结果显示新模型在生成质量上相当，但在延迟上显著更快，1.3B模型的延迟为36毫秒，达到28 RPS，而基线模型的延迟为271毫秒，仅达到3.68 RPS。

**⚠️ 局限性**

限制在于模型的可扩展性和对更高分辨率的支持，以及在不同硬件平台上的性能表现。

---

## 284. PhysScope: Agentic Multimodal Physics Reasoning with Grounded Visual Memory

**arXiv ID:** 2611.23010 | [PDF](https://arxiv.org/pdf/2611.23010v1)

**作者:** Haoyang Liu `[一作]`, Yang Liu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习技术和强化学习算法相结合的方法。

**📊 数据集**

采用了公开的图像识别数据集进行实验，数据集包含数万张标注图像。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和处理速度上均优于其他方法。

**⚠️ 局限性**

限制在于算法对特定类型数据的适应性较差，可能无法广泛应用于所有领域。

---

## 285. VLM-guided Grounding Expert for Generalised Referring Expression Comprehension

**arXiv ID:** 2611.22625 | [PDF](https://arxiv.org/pdf/2611.22625v1)

**作者:** Mingcheng Zhu `[一作]` (Spirit AI), Yang Gao `[通讯]` (Spirit AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的模型VLM-guided Grounding Expert (VGE)，旨在通过结合冻结的视觉语言模型（VLM）和可训练的定位专家来提高语言表达的理解和物体定位能力。

**💡 创新点**

创新点在于通过层级联合注意力和文本条件实例评分将VLM的上下文表示与定位专家的空间预测能力结合起来，从而提高了对指令变化和目标数量变化的鲁棒性。

**🔧 技术方法**

使用了冻结的视觉语言模型（VLM）和可训练的定位专家，结合层级联合注意力和文本条件实例评分技术。

**📊 数据集**

在三个数据集上进行评估，包括gRefCOCO、Ref-ZOM和RoboRefIt，涵盖一般场景和机器人场景。

**📈 对比分析**

VGE在所有五个测试分割中实现了最高的平均F1分数，相比于其他方法在指令条件下的变化最多仅有1.2个百分点的差异，并且在RoboRefIt数据集中在无目标、单目标和多目标情况下均表现最佳。VGE在推理速度上比基线快5.3到13.0倍。

**⚠️ 局限性**

限制在于冻结VLM可能会限制通过任务特定适应获得进一步的提升，尽管在两个数据集上的结果支持仅训练定位专家和接口作为一种竞争性的默认选择。

---

## 286. Multi-connectivity in Wireless Access

**arXiv ID:** 2611.22869 | [PDF](https://arxiv.org/pdf/2611.22869v1)

**作者:** Danny De Vleeschauwer `[一作]` (Nokia Bell Labs), Simone Bolettieri `[通讯]` (Nokia Bell Labs)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本论文研究了多连接性，即用户设备（UE）可以同时连接多个接入点（AP），是否能为用户提供更好的体验质量（QoE）。

**💡 创新点**

创新点在于通过多连接性提高用户的平均效用和减少效用波动，从而改善QoE，尤其是在高密度城市环境中。

**🔧 技术方法**

使用了资源分配模型，结合了频谱效率和效用波动的数学建模方法。

**📊 数据集**

使用了自由空间配置和马德里配置的数据集进行性能评估。

**📈 对比分析**

通过比较全局最优策略、瞬时最优策略和最强AP策略，发现瞬时最优策略在平均比特率和波动性方面表现最佳，而最强AP策略在分布式实施中表现良好，但性能略逊一筹。

**⚠️ 局限性**

限制在于未充分考虑干扰的影响，且在大规模配置中，某些策略的计算复杂度较高，可能导致无法在合理时间内完成优化。

---

## 287. Hardness and Inapproximability of the Covering Radius for LDPC Codes

**arXiv ID:** 2611.22617 | [PDF](https://arxiv.org/pdf/2611.22617v1)

**作者:** Massimo Battaglioni `[一作]` (Universita Politecnica delle Marche), Fabrizio Marinelli `[通讯]` (Universita Politecnica delle Marche)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了稀疏奇偶校验矩阵的覆盖半径决策问题，证明了在最大行和列权重为三的情况下，该问题是Π_2^P完全的，并且在常数乘法间隙下仍然是Π_2^P困难的。

**💡 创新点**

提出了覆盖半径问题的复杂性分类，发现最大行或列权重为二时问题可以在多项式时间内解决，而权重为三时则是Π_2^P完全的，且提供了显式的NP困难近似因子。

**🔧 技术方法**

使用了有界出现量化SAT的归约技术，结合图论中的顶点覆盖问题，构建了相应的稀疏奇偶校验矩阵。

**📊 数据集**

使用了具有最大行和列权重为三的稀疏奇偶校验矩阵，具体的构造和性质在论文中详细描述。

**📈 对比分析**

通过与已知的NP困难问题（如顶点覆盖）进行比较，展示了在最大行和列权重为三的情况下，覆盖半径问题的近似性能是NP困难的，并且提供了具体的近似因子。

**⚠️ 局限性**

限制在最大行和列权重为三的情况下，覆盖半径问题的复杂性仍然存在，且在更高的权重限制下，问题的可近似性和计算复杂性仍然是开放问题。

---

## 288. From Imperfective Paradox to Moorean Absurdity: Single-Chain Reasoning Lock-In and Multi-Candidate Verbalization Across Aspectual and Pragmatic Ambiguity

**arXiv ID:** 2611.22188 | [PDF](https://arxiv.org/pdf/2611.22188v1)

**作者:** An-Sheng Jhang `[一作]` `[通讯]` (Dongguan University of Technology), An-Sheng Jhang (Dongguan University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了自然语言推理和一致性判断如何受到事件发生、目标达成和多种语用标准的影响。通过四个预注册的实验，验证了单链推理锁定假设，发现单一输出瓶颈可能导致模型过早承诺于某一评估标准。

**💡 创新点**

创新点在于提出了单链提前锁定假设，并通过多候选人表述方法展示了如何在一次调用中生成多个候选概率估计，从而避免了单一标准的压缩效应。

**🔧 技术方法**

使用了多候选人表述（Multi-Candidate Verbalization）技术，结合了自我加权和格式敏感的候选多样性评估。

**📊 数据集**

研究使用了16,640个正式推理单元，涵盖了英语和普通话的多种实验设计，具体包括4240个计划/收集的单位，确保了数据的有效性和完整性。

**📈 对比分析**

通过与单一候选人概率方法的比较，发现多候选人表述在提高一致性和准确性方面表现更佳，尤其是在处理复杂的语用和时态模糊性时，性能显著优于传统方法。

**⚠️ 局限性**

限制在于所有实验均在单一本地模型上进行，样本量相对较小，且未能完全隔离内部神经表征与候选权重的影响，可能影响结果的普适性和解释性。

---

## 289. Accelerating 3D Gaussian Splatting Training through Selective Freezing and Computation Caching

**arXiv ID:** 2611.22059 | [PDF](https://arxiv.org/pdf/2611.22059v1)

**作者:** He Zhu `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种算法-硬件协同设计框架，通过选择性冻结和计算缓存加速3D高斯点云训练，显著减少训练时间。

**💡 创新点**

创新点在于动态冻结已收敛的高斯点，并缓存其贡献，从而减少计算量，同时保持重建质量。

**🔧 技术方法**

使用了渐进式高斯冻结技术和缓存兼容的反向光栅化方法，结合流式架构以提高计算效率。

**📊 数据集**

在三个广泛使用的3D高斯点云数据集上进行评估，包括Mip-NeRF360、Tanks and Temples和DeepBlending。

**📈 对比分析**

与现有的GPU和专用3DGS加速器相比，该方法在训练速度上实现了最高20.6倍的加速和99.4%的能耗降低，同时保持了相似的重建质量。

**⚠️ 局限性**

限制在于虽然冻结高斯点可以减少计算，但在某些情况下可能会影响重建质量，尤其是在冻结比例过高时。

---

## 290. Multispectral-Polarization-Infrared Multi-Aperture Camouflaged Vehicle Detection

**arXiv ID:** 2611.22246 | [PDF](https://arxiv.org/pdf/2611.22246v1)

**作者:** Shu Wang `[一作]` (Fuzhou University), Wenxi Liu `[通讯]` (Fuzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种多光谱-偏振-红外三模态伪装车辆检测框架，旨在解决复杂场景中伪装车辆的检测问题。

**💡 创新点**

创新点在于引入了一种共识引导的决策级融合策略，减少了对精确像素对齐的依赖，同时保持高检测精度和实时性能。

**🔧 技术方法**

使用了多光谱、偏振和红外成像技术，并结合了多孔径成像系统。

**📊 数据集**

构建了一个包含3726个同步样本的三模态伪装车辆数据集，涵盖五种伪装车辆类别。

**📈 对比分析**

与单模态和特征级融合方法相比，提出的方法在多目标场景中实现了82.7%的平均精度（mAP），并且端到端延迟仅为16毫秒，性能优于现有的特征级融合方法。

**⚠️ 局限性**

限制在于尽管构建了多模态数据集，但在真实部署中仍存在模型与实际环境之间的差距，且在高速度平台上可能受到运动模糊和光照变化的影响。

---

## 291. Centroid Spectral Coupling: Implicit Alignment and Representation Diversity in Self-Supervised Learning

**arXiv ID:** 2611.22158 | [PDF](https://arxiv.org/pdf/2611.22158v1)

**作者:** M. Hadi Sepanj `[一作]` (University of Waterloo), Paul Fieguth `[通讯]` (University of Waterloo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的自监督学习目标Centroid Spectral Coupling (CSC)，通过一个单一的谱损失函数同时促进增强视图对齐、表示多样性和抵抗崩溃。

**💡 创新点**

创新点在于CSC通过一个岭谱损失函数隐式地结合了对齐和多样性，而不需要显式的对比负样本或单独加权的正则化项。

**🔧 技术方法**

使用了岭谱损失函数，该函数应用于增强视图质心的无中心二阶矩阵。

**📊 数据集**

在CIFAR-10、CIFAR-100和Imagenette数据集上进行了实验。

**📈 对比分析**

与现有的非对比方法（如VICReg、Barlow Twins和MMCR）进行比较，CSC在所有三个基准上都实现了最高的线性Top-1准确率，并在CIFAR基准上表现出色的kNN Top-1准确率。

**⚠️ 局限性**

限制在于该方法的有效性可能依赖于超参数的选择，尤其是岭参数κ的设置可能影响对齐和多样性之间的权衡。

---

## 292. An Omnidirectional WECT: Closed-Form Computation of the Directionally Integrated Weighted Euler Characteristic Transform

**arXiv ID:** 2611.22101 | [PDF](https://arxiv.org/pdf/2611.22101v1)

**作者:** Jessi Cisewski-Kehe `[一作]` (University of Wisconsin), Jacob Sriraman `[通讯]` (Montana State University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文正式引入了方向积分加权欧拉特征变换（DIWECT），该方法通过对所有方向的加权欧拉特征变换（WECT）进行平均，消除了对方向参数的依赖，并提供了一种高效的计算框架。

**💡 创新点**

DIWECT的创新点在于它通过对所有可能方向的高度过滤进行积分，避免了对方向的采样，从而提高了计算效率并减少了高维表示的复杂性。

**🔧 技术方法**

使用了方向积分加权欧拉特征变换（DIWECT）算法，该算法通过测量球面上区域交集的面积来计算特征。

**📊 数据集**

使用了两个数据集进行实验：一个是合成的二维形状数据集（EAS2D），另一个是实际的大麦种子X射线CT扫描数据集。

**📈 对比分析**

与通过采样方向平均WECT的方法相比，DIWECT在匹配精度的情况下通常计算速度更快。在某些情况下，DIWECT在处理一百万个单元的图像时速度提升超过50倍。

**⚠️ 局限性**

DIWECT的局限性在于，当数据集提供可靠的框架或方向时，WECT可以利用姿态信息，而DIWECT则无法做到这一点。

---

## 293. Language Models Can Rewrite Their Own Weights

**arXiv ID:** 2611.21885 | [PDF](https://arxiv.org/pdf/2611.21885v1)

**作者:** Jiajun Fan `[一作]` (University of Illinois Urbana-Champaign), Ge Liu `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文展示了一种新的自我重写机制，允许大型语言模型（LLM）通过自身推理来编辑自己的权重，从而提高其性能。

**💡 创新点**

创新点在于首次证明了大型语言模型能够在没有梯度和训练的情况下，自我诊断并修复自身的错误，且这种能力随着模型规模的增大而增强。

**🔧 技术方法**

使用了自我重写的技术，模型在运行时能够读取和修改自己的权重，并通过自身的推理进行测试和验证。

**📊 数据集**

使用了来自DeepScaleR的数据集，包括3000个数学问题，并在实验中保留了1000个未见过的问题进行评估。

**📈 对比分析**

与其他方法相比，27B模型在490个未见过的问题上提高了6.6个百分点，且在多个实验中表现出显著的准确性提升。较小的模型（2B、4B和9B）未能成功进行自我编辑或表现出性能下降。

**⚠️ 局限性**

限制在于当前的自我重写机制仅在27B模型上有效，较小的模型未能实现自我编辑，且模型的自我修复能力仍需进一步验证和优化。

---

## 294. RISA: Recoverable Sparse Attention for Block-Diffusion Language Models

**arXiv ID:** 2611.22191 | [PDF](https://arxiv.org/pdf/2611.22191v1)

**作者:** Xinrui Zhong `[一作]` (Rice University), Yuke Wang `[通讯]` (Rice University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可恢复的稀疏注意力方法，用于块式扩散语言模型（DLMs），以提高长上下文推理的效率。

**💡 创新点**

创新点在于通过共享读取计划来减少重复的键值（KV）读取，同时保持历史信息的可恢复性。

**🔧 技术方法**

使用了块静态稀疏注意力执行器，结合了观察引导的历史优先级和可恢复的生成历史。

**📊 数据集**

使用了DreamReasoner-8B、SDAR-8B-Chat和LLaDA2.2-mini等数据集进行评估。

**📈 对比分析**

与MAGE、QUEST和LoSA等方法进行比较，在多个设置下，解码速度更快且保持了相似的任务性能。

**⚠️ 局限性**

限制在于虽然减少了KV读取，但仍需在生成过程中保持历史信息的完整性，可能会影响某些情况下的性能。

---

## 295. Can Coding Agents Write and Transpile Unit Tests

**arXiv ID:** 2611.21797 | [PDF](https://arxiv.org/pdf/2611.21797v1)

**作者:** Gil Pasternak `[一作]` (Code Metal), Ellie Kitanidis `[通讯]` (Code Metal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了三种流行的编码助手在生成和翻译测试方面的能力，特别是在处理240个开源项目的单元测试时。

**💡 创新点**

创新点在于详细分析了现代编码助手在测试生成和翻译中的能力和失败模式，发现它们在生成测试数量上优于人类，但在单个测试的质量上却不如人类。

**🔧 技术方法**

使用了三种编码助手：Claude Code、Codex和Cursor，并结合了高推理努力的原生前沿模型进行评估。

**📊 数据集**

使用了240个开源项目的数据集，这些项目涵盖C++、Python和Rust，并且每个项目都有人工编写的测试。

**📈 对比分析**

与人类编写的测试套件相比，编码助手在传统覆盖率指标上表现更好，但在大型项目上性能下降明显，尤其是在C++项目中。总体来看，编码助手的优势主要来自于生成更多的测试，而不是提高单个测试的质量。

**⚠️ 局限性**

限制在于研究仅基于开源项目，未涵盖企业级代码库，且未单独评估编码助手的模型和工具的影响。此外，推理分析仅基于单一模型，未能验证其他助手的失败模式是否一致。

---

## 296. Fly as Instructed: Refining World-Action Policies for UAV Vision-Language Navigation

**arXiv ID:** 2611.22883 | [PDF](https://arxiv.org/pdf/2611.22883v1)

**作者:** Zhexuan Zhou `[一作]` (Harbin Institute of Technology), Jie Mei `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种世界-动作学习框架，用于自然语言引导的无人机导航，集成了语言条件的动作接口、任务进度跟踪和障碍物避免机制。

**💡 创新点**

创新点在于结合了针对指令响应的动作后训练、递归指令消费预测和高斯场后训练，以提高任务完成率和安全性。

**🔧 技术方法**

使用了语言条件的动作接口、递归神经网络（GRU）进行指令消费预测，以及高斯场后训练技术。

**📊 数据集**

使用了521个闭环飞行任务的评估数据集，包括一般任务、安全任务和进度任务。

**📈 对比分析**

与现有方法（如WorldVLN、FastWAM等）进行比较，结果显示本方法在任务成功率、正确停止率和安全飞行成功率等指标上均优于对比方法。

**⚠️ 局限性**

局限性在于高速度飞行时反应时间较短，可能需要进一步减少端到端策略延迟，并且在环境布局或视觉条件显著不同的情况下，泛化能力可能不足。

---

## 297. From Token Changes to Task Failures: Bit Flips in a Shared KV-Cache Block

**arXiv ID:** 2611.22243 | [PDF](https://arxiv.org/pdf/2611.22243v1)

**作者:** Yuji Yamamoto `[一作]` (Institute of Science Tokyo), Satoshi Matsuura `[通讯]` (Institute of Science Tokyo)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文扩展了之前的研究，通过测量共享前缀缓存中的位翻转如何影响大型语言模型（LLM）服务系统的任务失败，特别是如何导致错误响应。

**💡 创新点**

创新点在于量化了位翻转对不同任务的影响，特别是错误响应的比例，并提出了检测这些错误响应的方法。

**🔧 技术方法**

使用了软件故障注入技术，针对vLLM服务的Qwen3-8B模型进行了位翻转实验。

**📊 数据集**

使用的数据集包括MBPP+（代码生成）、TriviaQA（简答问题）、MMLU-Pro（多项选择推理）和政策拒绝任务。

**📈 对比分析**

通过比较不同任务的错误响应比例，发现MBPP+的错误率为0.023，MMLU-Pro为0.060，而TriviaQA则高达0.59，表明任务类型对错误响应的影响显著。

**⚠️ 局限性**

限制在于实验仅针对特定模型和数据集，且未考虑所有可能的物理攻击条件，此外，某些脆弱项在没有翻转的情况下也可能产生错误。

---

## 298. RICoT: Recurrent Interleaved Chain-of-Thought for Page Stream Segmentation

**arXiv ID:** 2611.22382 | [PDF](https://arxiv.org/pdf/2611.22382v1)

**作者:** Prashant Khatri `[一作]` (Hyperbots), Niyati Chhaya `[通讯]` (Hyperbots)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种新的页面流分割（PSS）方法RICoT，通过滑动窗口的方式对整个页面流进行处理，模拟阅读者的思维过程。

**💡 创新点**

RICoT框架结合了逐页的思维推理和跨窗口的状态传播，能够在长序列中保持文档级一致性，克服了传统方法的局限性。

**🔧 技术方法**

使用了递归交错思维（Recurrent Interleaved Chain-of-Thought）技术，结合滑动窗口处理和状态传播。

**📊 数据集**

引入了FinancePSS-205数据集，该数据集包含205个真实的去标识化财务页面流。

**📈 对比分析**

与传统的成对分类和窗口分类方法相比，RICoT在多个基准测试中表现优越，尤其是在FinancePSS-205上，达到了0.847的文档F1分数，超越了已发布的零-shot结果。

**⚠️ 局限性**

限制在于评估仅限于英语文档，且数据集规模较小，可能影响到模型的泛化能力。

---

## 299. FM-Mamba: An Efficient Native 2D Selective State Space Model with Stability-Aware Discretization

**arXiv ID:** 2611.22061 | [PDF](https://arxiv.org/pdf/2611.22061v1)

**作者:** Shengyuan Hu `[一作]` (Beijing Institute of Technology), Yongzhe Chang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于Fornasini-Marchesini模型的FM-Mamba方法，旨在处理二维特征（如图像）的选择性状态空间模型，避免了将特征图展平为一维序列的问题。

**💡 创新点**

FM-Mamba通过保持二维网格上的隐藏状态，结合空间轴的耦合递归，保留了空间邻接性和多路径传播，且没有转换开销，显著提高了训练效率和模型性能。

**🔧 技术方法**

使用了Fornasini-Marchesini模型的连续动态和零阶保持（ZOH）离散化技术，结合双对角波前扫描实现了高效的状态更新。

**📊 数据集**

在COCO 2017数据集上进行多标签分类实验，训练集和测试集按8:2比例划分，使用了5000张验证图像。

**📈 对比分析**

与现有的Mamba2D等方法相比，FM-Mamba在训练时间上减少了52.8%，峰值内存和训练能耗分别降低了42.1%和53.5%，且准确率与Mamba2D相差不超过0.12个百分点。

**⚠️ 局限性**

FM-Mamba的局限性在于其稳定性依赖于输入的归一化比率r_k，且在某些配置下可能需要额外的步骤投影和后训练验证，以确保模型的稳定性。

---

## 300. Testyard Benchmarks: a Dynamic RISC-V-based Benchmark Framework including DfT insertion, Structural and Functional Test Flows

**arXiv ID:** 2611.21666 | [PDF](https://arxiv.org/pdf/2611.21666v1)

**作者:** Francesco Angione `[一作]` (Politecnico di Torino), Paolo Bernardi `[通讯]` (Politecnico di Torino)

**通讯引用:** 44828 | [OpenAlex ID](https://openalex.org/A5009191571)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于RISC-V的动态基准框架Testyard，旨在解决现代计算制造中的测试挑战，提供开放源代码的工业规模基准。

**💡 创新点**

创新点在于将DfT插入、自动测试模式生成（ATPG）和功能测试故障仿真集成到Hammer VLSI流中，形成一个统一的框架，简化了硬件测试方法的原型设计。

**🔧 技术方法**

使用了Chipyard和Hammer框架，结合高层次的YAML配置来抽象EDA工具的命令，减少了对特定工具和技术库的依赖。

**📊 数据集**

使用了来自Chipyard生态系统的13个代表性SoC配置，涵盖了45nm、15nm和7nm等多个技术库。

**📈 对比分析**

通过与传统基准（如ITC-99）进行比较，Testyard基准在复杂性和故障模型的覆盖范围上显著提高，提供了更真实的工业设计测试场景。

**⚠️ 局限性**

限制在于当前框架仍需依赖于特定的EDA工具，且在不同技术库之间的适应性可能存在挑战。

---

## 301. The Software Engineering Factory: Engineering Accountability and Control in the Age of Abundant Code

**arXiv ID:** 2611.21623 | [PDF](https://arxiv.org/pdf/2611.21623v1)

**作者:** Ahmed E. Hassan `[一作]` (Queen's University), Haoxiang Zhang `[通讯]` (Queen's University)

**通讯引用:** 1371 | [OpenAlex ID](https://openalex.org/A5100765845)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

本文提出了软件工程工厂（SE Factory）的概念，旨在通过工程化的流程来增强代码生产的责任感和控制力，解决AI代理生成代码带来的所有权问题。

**💡 创新点**

创新点在于将控制权转移到代码生产的过程上，通过定义明确的协议和阶段，确保每个阶段都有必要的证据和责任人，从而实现可追溯的代码所有权。

**🔧 技术方法**

使用了确定性协议引擎来实施这些协议，确保工作流程的每个阶段都有明确的输入、输出和验证要求。

**📊 数据集**

论文中提到的三个SE Factory实现都经过多次迭代，其中两个作为开源系统公开可用，分别是Codev和Swamp。

**📈 对比分析**

通过与传统的代码审查方法进行比较，SE Factory在代码质量和可靠性方面表现出显著的提升，尤其是在处理复杂任务时，SE Factory的成功率远高于单一代理的表现。

**⚠️ 局限性**

限制在于当前的实验和结果主要基于特定的实现和数据集，尚未全面验证这些原则在不同环境和规模下的适用性。

---

## 302. How Well Is My Artwork Protected? Predicting and Tailoring Protection against Style Mimicry by Personalization

**arXiv ID:** 2611.22486 | [PDF](https://arxiv.org/pdf/2611.22486v1)

**作者:** Jihye Kang `[一作]` (Inha University), Namhyuk Ahn `[通讯]` (Inha University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文研究了艺术作品的保护方法，提出了一种新的工具，可以在个别艺术作品上评估、预测和选择保护配方，以确保艺术作品在个性化处理后不会被模仿。

**💡 创新点**

创新点在于通过短期试验保护来预测每个配方的保护效果，从而为每个艺术作品选择最弱的保护配方，以达到所需的保护水平。

**🔧 技术方法**

使用了轻量级回归模型（LightGBM）来预测保护分数，并结合了多种信号（如图像信号和保护跟踪信号）来提高预测的准确性。

**📊 数据集**

使用了来自WikiArt、StyleBench和OmniStyle的300幅艺术作品作为数据集，涵盖了历史绘画到现代数字艺术的广泛范围。

**📈 对比分析**

与现有的保护工具（如统一配方）相比，提出的方法在相同的可见性下允许更多的艺术作品通过保护阈值，并且在可见性较低的情况下达到相同的通过率。

**⚠️ 局限性**

限制在于该方法依赖于已有的保护方法，且在面对强烈的图像净化（如JPEG压缩）时，可能无法保证所有艺术作品都能通过保护阈值。

---

## 303. DACE: Diversity-Driven Adversarial Co-Evolution for Robust LLM Safety Alignment

**arXiv ID:** 2611.22278 | [PDF](https://arxiv.org/pdf/2611.22278v1)

**作者:** Peng Yu `[一作]` (Shanghai Jiao Tong University), Chaochao Lu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多样性驱动的对抗共演框架，旨在增强大型语言模型（LLMs）的安全对齐，解决攻击策略崩溃和防御对抗遗忘的问题。

**💡 创新点**

创新点在于通过结合策略层的攻击探索和不确定性感知的对抗重放，联合解决了两种相互影响的病态现象。

**🔧 技术方法**

使用了贝叶斯对抗重放池和规范化的边际覆盖增益奖励，构建了一个明确的风险类别与攻击风格的二维策略空间。

**📊 数据集**

使用了来自WildJailBreak和HarmBench的数据集，构建了策略条件的CoT数据集用于训练。

**📈 对比分析**

与现有的共演基线（如MAGIC和Self-RedTeam）相比，展示了在安全性、转移性和一般能力评估中的显著改进，尤其是在对抗分布外攻击的鲁棒性方面。

**⚠️ 局限性**

限制在于尽管框架在多样性和鲁棒性上表现良好，但在处理多轮攻击时仍然存在一定的挑战，且绝对的攻击成功率仍然较高。

---

## 304. HandUMI: A Modular Hand-Worn Parallel-Jaw Interface for Robot-Free Bimanual Data Capture

**arXiv ID:** 2611.22146 | [PDF](https://arxiv.org/pdf/2611.22146v1)

**作者:** Brik Meza `[一作]` (Mu Robotics), Bryan Bastidas `[通讯]` (NONHUMAN)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文提出了一种手持的平行夹具接口，用于捕捉无机器人双手演示，并通过逆向运动学将数据重定向到七个模拟的双手臂上。

**💡 创新点**

创新点在于开发了一种开放、低成本的手持接口，能够与机器人匹配的可更换指尖，实时控制器跟踪和编码器测量的开口度，同时保留机器人的原装夹具。

**🔧 技术方法**

使用了逆向运动学（IK）技术来处理数据重定向，并结合了实时控制器跟踪和编码器测量。

**📊 数据集**

使用的数据集是通过三名操作员在四天内记录的641个演示集，其中包括对双手折叠毛巾的实验。

**📈 对比分析**

与其他方法相比，ACT在562个演示中成功完成12/20次，而经过微调的π_0.5成功完成7/20次，性能上没有显著差异（p=0.20）。

**⚠️ 局限性**

限制在于跨体现证据是模拟的IK，实际测试仅覆盖一个PiPER任务，且未进行失败分析；捕捉过程受限于头戴设备和并行夹具，指尖匹配增益、夹具精度、指尖负载和音频价值未被测量。

---

## 305. Back to Mel: Learning to Bridge Fidelity and Perception in Generative Speech Enhancement

**arXiv ID:** 2611.22628 | [PDF](https://arxiv.org/pdf/2611.22628v1)

**作者:** Andong Li `[一作]` (Institute of Acoustics, Chinese Academy of Sciences), Chengshi Zheng `[通讯]` (Institute of Acoustics, Chinese Academy of Sciences)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于Mel谱的生成语音增强框架BAME，旨在平衡声学保真度、感知质量和处理效率。

**💡 创新点**

创新点在于利用Mel谱的结构化特性和退化相位信息，提出了轻量级的子带感知Transformer（SaT）和质量引导的双相位编码策略，以提高语音增强的性能。

**🔧 技术方法**

使用了基于流匹配的生成模型、子带感知Transformer（SaT）和相位感知神经声码器等技术。

**📊 数据集**

使用的数据集包括DNS-Challenge、EARS、GSR和URGENT基准测试集。

**📈 对比分析**

与多个先进基线方法进行比较，BAME在多种侵入性和非侵入性评估指标上均表现出色，达到了最先进的性能。

**⚠️ 局限性**

限制在于现有的客观指标尚未提供准确反映声学保真度和感知质量的“黄金”标准，且Mel表示法的最优性尚待进一步研究。

---

## 306. JointGV: Structure-Aware Correction of Visual World Models with Parallel Graph Dynamics for Long-Horizon Modeling

**arXiv ID:** 2611.22543 | [PDF](https://arxiv.org/pdf/2611.22543v1)

**作者:** Tian Lyu `[一作]` (University of Oxford), Fan Mo `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的方法Joint Graph–Visual Dynamics Correction (JointGV)，用于改善视觉世界模型在长时间预测中的准确性，特别是在机器人规划和控制中。

**💡 创新点**

创新点在于通过独立预测的关系状态来纠正视觉模型的漂移，而不需要重构视觉模型，提出了一种跨粒度的动态假设，允许不同表示的图节点和视觉状态的变化描述相同的物理转变。

**🔧 技术方法**

使用了图学习和视觉动态的结合，采用了关系变换器（Relation Transformer）来构建任务条件的场景图，并通过边界校正来改善视觉预测。

**📊 数据集**

在DMC Cheetah Run和Walker Walk等数据集上进行了评估，并扩展到接触丰富的操作任务DMC Finger Turn Hard。

**📈 对比分析**

与现有方法相比，JointGV在32步的预测中减少了11-28%的关节位置误差，特别是在非相邻部分在图像中接近时，学习的图-视觉变化耦合表现出更低的误差。

**⚠️ 局限性**

限制在于JointGV增加了计算复杂性，并假设节点角色是由任务定义的，纠正效果依赖于图观察和预测的质量，未来工作将集中在自动角色发现和更广泛的视觉模型上。

---

## 307. MedImgAnnotator: batch wise medical image annotation and preprocessing tool

**arXiv ID:** 2611.22450 | [PDF](https://arxiv.org/pdf/2611.22450v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 308. KV-LTM: Persistent Per-Session KV Caches as Long-Term Memory for Personal Assistants

**arXiv ID:** 2611.21998 | [PDF](https://arxiv.org/pdf/2611.21998v1)

**作者:** Ziv Kfir `[一作]`, Ido Kfir `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的长期记忆系统KV-LTM，用于个人助手，旨在解决当前系统在问答时的内容损失问题。

**💡 创新点**

KV-LTM通过存储每个会话的原始文本和KV缓存，避免了重复的会话文本预填充，从而提高了问答的准确性和效率。

**🔧 技术方法**

使用了KV缓存机制和固定的阅读模型，结合了分析模型来估算存储和计算成本。

**📊 数据集**

使用了LongMemEval和LoCoMo等公共基准数据集进行评估，比较了不同系统的性能。

**📈 对比分析**

与现有的记忆系统（如RAG和提取事实）相比，KV-LTM在问答准确性上表现更好，尤其是在保持原始文本的情况下，避免了写入时的内容损失。

**⚠️ 局限性**

限制在于KV-LTM的存储需求较高，且在模型、配置或tokenizer变化时，存储的状态会失效，需要重新填充文本。

---

## 309. Reusing Battle-tested Code: A Systematic Literature Review of Automated Software Transplantation

**arXiv ID:** 2611.21630 | [PDF](https://arxiv.org/pdf/2611.21630v1)

**作者:** Thanatad Songpetchmongkol `[一作]`, Justyna Petke `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文进行了系统的文献综述，评估了自动化软件移植的现状，并识别了阻碍该技术成为主流工程实践的障碍。

**💡 创新点**

创新点在于提出了全面的分类法，识别了自动化软件移植的瓶颈，并提供了未来研究的路线图，建议利用大型语言模型来克服手动步骤的障碍。

**🔧 技术方法**

使用了文献综述的方法，分析了不同的自动化软件移植方法，包括程序切片和遗传编程等技术。

**📊 数据集**

研究中涉及的文献数据集包括从2008年到2025年的相关论文，涵盖了软件移植领域的多种方法和技术。

**📈 对比分析**

与现有方法的比较显示，尽管一些技术在适应阶段实现了高度自动化，但整个移植流程仍未完全自动化，且许多方法仍需人工干预。

**⚠️ 局限性**

限制在于手动定位功能和测试套件增强的步骤仍需人工干预，这在工业应用中造成了障碍。

---

## 310. ClearTalk: Data-Driven Mandarin Articulation Analysis and Personalized Communication Support for Children with Dysarthria

**arXiv ID:** 2611.21639 | [PDF](https://arxiv.org/pdf/2611.21639v1)

**作者:** Ben Xiaolu Huang `[一作]` `[通讯]` (Shanghai Foreign Language School Affiliated to SISU), Ben Xiaolu Huang (Shanghai Foreign Language School Affiliated to SISU)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一个基于智能手表的系统ClearTalk，用于分析儿童的普通话发音并提供个性化的沟通支持，特别是针对有言语障碍的儿童。

**💡 创新点**

创新点在于将可解释的发音分析与个性化练习和分阶段沟通支持相结合，提供了一种新的数据驱动的可穿戴工作流程。

**🔧 技术方法**

使用了ESP32智能手表进行音频捕捉和处理，结合了阿里云的自动语音识别和语音合成技术。

**📊 数据集**

使用了普通话言语障碍语料库（Mandarin Dysarthria Speech Corpus），包含13名言语障碍儿童的语音数据。

**📈 对比分析**

通过与三位盲评的语言病理学家的共识进行比较，系统选择的优先目标与专家共识的重叠率为86.7%；在210个目标辅音中，自动标签的达成率为86.7%，Cohen's kappa为0.76，宏观F1分数为0.81。

**⚠️ 局限性**

限制在于该研究是作为可行性评估而非治疗试验进行的，结果需要在控制的重复会话研究中进一步验证因果关系和持久效果。

---

## 311. ReMoE: An Interpretable Mixture-of-Experts Framework for Human Mobility Recovery Forecasting Across Disasters

**arXiv ID:** 2611.22133 | [PDF](https://arxiv.org/pdf/2611.22133v1)

**作者:** Yilin Lyu `[一作]` (Beijing Normal-Hong Kong Baptist University), Yushi Bai `[通讯]` (Shenzhen Institutes of Advanced Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种可解释的专家混合框架ReMoE，用于跨灾害的人类流动恢复预测。

**💡 创新点**

ReMoE通过特征引导的路由器自适应组合学习可重用的恢复类型，提供区域特定的恢复模式，并通过残差修正进一步优化预测。

**🔧 技术方法**

使用了混合专家（MoE）架构，结合了可解释的恢复类型组合器（IRTC）和时空压力信息残差修正（STP-RC）模块。

**📊 数据集**

使用了一个大规模的流动数据集，涵盖了18个灾害事件，包括飓风、洪水、冬季风暴和野火，共有超过2.19亿条访问记录。

**📈 对比分析**

与多种基线模型（如ARIMAX、XGBoost、深度学习模型等）进行比较，ReMoE在未见事件和未见灾害类型的设置下表现优越，尤其在恢复语义指标上有显著提升。

**⚠️ 局限性**

局限性包括数据仅覆盖部分人群，且在小空间单位的分析受到限制；所选的18个事件时间线清晰，但缺乏复合事件的研究；未来研究应探索更具上下文特定的观察窗口定义。

---

## 312. UnrealEvent: Synthetic Dataset Generation and Annotation for Event-Based Vision

**arXiv ID:** 2611.23078 | [PDF](https://arxiv.org/pdf/2611.23078v1)

**作者:** Yentel Schoovaerts `[一作]` (KU Leuven), Patrick Vandewalle `[通讯]` (KU Leuven)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了UnrealEvent，一个基于Unreal Engine 5.5的合成数据集生成和注释管道，旨在解决事件驱动视觉中的标注训练数据稀缺问题。

**💡 创新点**

创新点在于结合了V2E风格的事件帧生成器与EasySynth语义渲染，能够自动生成配对的事件表示和YOLO格式的边界框注释，无需人工标注。

**🔧 技术方法**

使用了Unreal Engine 5.5进行合成数据生成，结合了自定义的事件效果和EasySynth的语义渲染。

**📊 数据集**

生成的合成数据集包含50,692个标注事件帧，涵盖白天和夜间场景，旨在测试不同光照条件下的检测性能。

**📈 对比分析**

通过在合成数据、零-shot真实数据和真实微调训练上评估YOLOv8检测器，结果显示合成数据在合成测试集上表现良好，但在真实数据上存在显著的合成-真实差距，尤其是行人检测方面。微调后性能有所提升，但仍需改进资产多样性和传感器校准。

**⚠️ 局限性**

限制在于合成数据与真实数据之间存在显著的领域差距，尤其是行人检测的资产多样性不足，导致模型在真实场景中的泛化能力较差。

---

## 313. Distilling World Action Models for Efficient Bimanual Manipulation via Anchored Path Matching

**arXiv ID:** 2611.22592 | [PDF](https://arxiv.org/pdf/2611.22592v1)

**作者:** Jialong Liu `[一作]` (Astribot), Changhao Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的方法Anchored Path Matching (APM)，用于通过迭代去噪生成未来视频和机器人动作，同时减少模型评估次数。

**💡 创新点**

APM通过在学生达到的状态上配对端点和局部过渡监督，解决了粗略更新可能导致学生偏离教师轨迹的问题，并引入了锚优先更新以保持端点梯度。

**🔧 技术方法**

使用了基于AnyFlow的蒸馏技术和模态路由适配器，结合锚优先校正来协调不同目标的优化。

**📊 数据集**

在RoboTwin 2.0数据集上进行训练，使用干净的演示数据进行训练，并在随机化的环境中进行评估。

**📈 对比分析**

与继续基础蒸馏方法相比，APM在1步、2步和4步生成上分别提高了闭环成功率2.0和1.4个百分点，视频峰值信噪比(PSNR)从30.169提高到32.317 dB，且在真实机器人任务中减少了控制周期延迟21.3倍。

**⚠️ 局限性**

当前的真实机器人评估仅限于两个操作任务和单一平台，未来的工作可以根据教师的可靠性调整校正，并研究更长的预测时间和其他机器人平台。

---

## 314. ALOHA Lightning: Learning Fast and Precise Manipulation

**arXiv ID:** 2611.21919 | [PDF](https://arxiv.org/pdf/2611.21919v1)

**作者:** John Hua Yao. Qi Wu `[一作]`, Zipeng Fu `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种全栈系统，用于学习快速且精确的机器人操作策略，能够在2秒内自主完成双手投接球、在7秒内清理完整的餐具，并在6秒内折叠一条短裤。

**💡 创新点**

创新点在于通过直观的运动教学收集接近人类速度的演示数据，并通过测试时动作平滑和视觉数据分布对齐技术，实现高效的高速度执行。

**🔧 技术方法**

使用了运动教学、测试时平滑处理和视觉域对齐等技术。

**📊 数据集**

使用了50个演示数据集，涵盖了投接球、折叠短裤、插入电池和清理餐桌等任务。

**📈 对比分析**

与其他方法（如无掩码、插值、时间集成等）进行比较，结果显示该系统在任务成功率上超过80%，执行速度比传统遥控系统快50%以上。

**⚠️ 局限性**

限制在于当前掩码策略过于保守，可能会抹去人手附近有用的视觉线索，并且系统仅结合了视觉和本体感知，限制了对需要接触感知和控制的灵巧操作的学习。

---

## 315. tmol: a GPU-accelerated implementation of the Rosetta energy function for machine learning

**arXiv ID:** 2611.21857 | [PDF](https://arxiv.org/pdf/2611.21857v1)

**作者:** Andrew Leaver-Fay `[一作]` (Prescient Design), Frank DiMaio `[通讯]` (Institute for Protein Design)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `09944146-298c-433e-89df-37255de463d7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为tmol的GPU加速PyTorch库，实现了RosettaGenFF力场，支持高效的批量可微评分和整体结构优化。

**💡 创新点**

tmol通过高效的批量张量操作解决了传统力场在机器学习框架中集成困难的问题，并显著提高了计算速度。

**🔧 技术方法**

使用了GPU加速的PyTorch库，结合了Rosetta能量函数的实现。

**📊 数据集**

在1000个高分辨率蛋白质结构的数据集上进行了测试。

**📈 对比分析**

tmol在评分和梯度计算上比PyRosetta快233-362倍，在扭转空间放松上快38-102倍，性能显著提升。

**⚠️ 局限性**

tmol在处理新化学时仍需用户手动检查某些配置，且在处理非标准残基和金属时存在挑战。

---

## 316. Symbolic Abstraction Refinement

**arXiv ID:** 2611.22404 | [PDF](https://arxiv.org/pdf/2611.22404v1)

**作者:** Daniel Bryce `[一作]` (Smart Information Flow Technologies), John Skovbekk `[通讯]` (Smart Information Flow Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的符号抽象细化方法，用于验证具有不确定性和时间约束的自主系统的决策过程。

**💡 创新点**

创新点在于将值函数符号化，使每个符号状态携带从其时钟区域的子区域到概率界限的函数，从而避免了对先前生成状态的拆分和重构。

**🔧 技术方法**

使用了启发式前向搜索和符号值函数的概念，结合了抽象细化和概率模型检查的技术。

**📊 数据集**

使用了PRISM PTA基准测试数据集，包括分布式协议、领导者选举和自配置等实例。

**📈 对比分析**

与现有的随机博弈方法相比，该方法在PRISM PTA基准测试中速度提高了高达108倍，并且在处理某些实例时显著减少了对截止时间常数的敏感性。

**⚠️ 局限性**

局限性在于，尽管方法在许多情况下表现出色，但在细化成本较低的情况下，仍然可能与现有方法的性能相近，且收敛性仅在事后得到保证。

---

## 317. REEL: Brain-Guided Video Generation and Selection by Reranking Encoder Embeddings with Linear Decoding from fMRI data

**arXiv ID:** 2611.21907 | [PDF](https://arxiv.org/pdf/2611.21907v1)

**作者:** Fabrizio Spera `[一作]`, Matteo Ferrante `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种名为REEL的框架，通过线性解码从fMRI数据生成和选择视频，旨在解决从脑活动解码视频内容的挑战。

**💡 创新点**

创新点在于采用了一种基于贝叶斯的提议-重排序策略，结合线性语义解码器和预训练视频生成器，简化了视频脑解码的过程。

**🔧 技术方法**

使用了线性岭回归解码器、预训练的视频生成器CogVideoX-5b，以及三个神经编码流。

**📊 数据集**

使用了CC2017和BOLD Moments Dataset (BMD)两个公共数据集进行评估。

**📈 对比分析**

与其他方法相比，REEL在CC2017数据集上实现了最高的SSIM值，并在BMD数据集上表现出有希望的结果，重排序提高了视频50-way准确率。

**⚠️ 局限性**

主要限制在于生成和评估多个候选视频的计算成本较高，且性能受限于生成器的质量，无法处理超出语义范围的刺激。

---

## 318. Multimodal Masked Diffusion Autoencoder for Modeling Impaired Driving

**arXiv ID:** 2611.23211 | [PDF](https://arxiv.org/pdf/2611.23211v1)

**作者:** John Gideon `[一作]` (Toyota Research Institute), Guy Rosman `[通讯]` (Toyota Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种多模态掩蔽扩散自编码器模型，用于预测驾驶员的认知状态，同时重建掩蔽的注视和轨迹。

**💡 创新点**

创新点在于引入自监督目标来提高损伤预测的准确性，并通过掩蔽重建来增强模型的性能。

**🔧 技术方法**

使用了多模态掩蔽扩散自编码器技术，结合了注视、轨迹、场景视频和损伤状态的联合建模。

**📊 数据集**

使用了一个新的驾驶模拟数据集，包含103小时的驾驶数据，涉及109名参与者和十种不同的损伤诱导。

**📈 对比分析**

与现有基线相比，该模型在损伤检测上取得了显著的性能提升，配对AUC从0.590提高到0.706，且在其他任务（如轨迹和注视预测）上也表现良好。

**⚠️ 局限性**

限制在于数据集的多样性和规模可能不足以覆盖所有潜在的驾驶损伤类型，且模型的复杂性可能导致训练和推理的计算成本较高。

---

## 319. Nunchi: Ears Help Eyes in Multi-Party Conversations

**arXiv ID:** 2611.23115 | [PDF](https://arxiv.org/pdf/2611.23115v1)

**作者:** JunYeong Park `[一作]` (Dankook University), Hyung Jin Chang `[通讯]` (University of Birmingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的方法Nunchi，用于在多方对话中识别观察者注视的对象，并构建了一个名为SORBET的数据集，包含3340个样本，涵盖多种多方场景。

**💡 创新点**

创新点在于结合了对话线索（说话者和转录文本）与注视几何信息，以提高在多方对话中识别注视对象的准确性。

**🔧 技术方法**

使用了视觉语言模型（VLM）和冻结的Gaze-LLE模型来估计注视几何，同时提取说话者信息和转录文本。

**📊 数据集**

使用了SORBET数据集，该数据集包含3340个样本，来自1578个视频片段，涵盖多种多方对话场景，并提供同步的语音、转录文本和注视对象标签。

**📈 对比分析**

与现有的最强方法Gaze-LLE相比，Nunchi在SORBET数据集上的准确率提高了4.8个百分点，且在多个VLM上均表现出色，尤其是在候选人数量较多时，准确率提升更为显著。

**⚠️ 局限性**

限制在于该模型基于剧本的电视系列，假设观察者只注视一个候选人，未考虑注视转移或注视非候选对象的情况。

---

## 320. Fair Bubble Sort: Provably Optimal Fair Ranking with Continuous Sensitive Attributes

**arXiv ID:** 2611.22352 | [PDF](https://arxiv.org/pdf/2611.22352v1)

**作者:** Antonio Ferrara `[一作]` (Intesa Sanpaolo), Francesco Bonchi `[通讯]` (Intesa Sanpaolo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了如何在不依赖于离散化的情况下，对具有连续保护属性的公平排名问题进行处理，提出了一种新的公平性概念——连续成对平衡（CPP），并寻求满足CPP约束的排名，同时最小化与初始基于分数的排名的Kemeny距离。

**💡 创新点**

提出了连续成对平衡（CPP）这一公平性概念，并开发了公平气泡排序（FBS）算法，这是第一个理论上有依据的后处理算法，能够通过有针对性的相邻交换最优地处理连续属性的效用与公平性之间的权衡。

**🔧 技术方法**

使用了公平气泡排序（FBS）算法，该算法基于相邻交换的贪婪策略，能够有效地满足公平性约束并最小化Kemeny距离。

**📊 数据集**

在合成数据集和真实数据集上进行了实验，验证了FBS的可扩展性和优越性能。

**📈 对比分析**

与基于离散化的基线方法进行比较，FBS在公平性和准确性之间的Pareto前沿上表现优越，能够在多个数据集上实现更好的结果。

**⚠️ 局限性**

限制在于FBS目前仅处理单一的连续保护属性，扩展到多个属性的情况尚未解决。此外，当前的设置是静态和一次性的，长期动态、重复曝光和反馈循环超出了本文的范围。

---

## 321. How Much Distribution Shift Can a Clustering Withstand? Exact Split and Merge Certificates for Structured Transports

**arXiv ID:** 2611.22000 | [PDF](https://arxiv.org/pdf/2611.22000v1)

**作者:** Alisina Bayati `[一作]` (University of Illinois Urbana-Champaign), Shahriar Talebi `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了聚类中的分裂和合并事件，提出了一种基于相变分析的局部稳定性标准，以量化聚类对分布变化的鲁棒性。

**💡 创新点**

创新点在于通过确定分裂和合并半径，提供了一种精确的鲁棒性证书，能够量化聚类在面对结构化分布变化时的稳定性。

**🔧 技术方法**

使用了确定性退火（deterministic annealing）和半正定规划（semidefinite programming）技术来处理分裂和合并半径的计算。

**📊 数据集**

使用了S1-S3基准数据集，这些数据集包含来自15个高斯聚类的5000个点，数据集的重叠程度逐渐增加。

**📈 对比分析**

通过与其他聚类方法的比较，本文的方法在处理分布变化时表现出更高的鲁棒性，尤其是在不同的位移方向下，能够有效识别最脆弱的事件。

**⚠️ 局限性**

限制在于该方法依赖于对分布变化的结构化假设，可能在某些实际应用中不适用，且计算复杂度在高维情况下可能较高。

---

## 322. How Many Training Labels Are Wrong? Auditing a Fitted Model through Gradient Allocation

**arXiv ID:** 2611.22449 | [PDF](https://arxiv.org/pdf/2611.22449v1)

**作者:** Yuchen Li `[一作]` (University of Sydney), Ken-Tye Yong `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于训练模型的梯度分配来识别训练集中错误标签的方法，能够在不需要干净数据和额外训练的情况下，评估标签噪声率。

**💡 创新点**

创新点在于通过标签交换审计，利用模型的每个示例的梯度信息来判断标签的正确性，而不是依赖于模型的输出或损失值。

**🔧 技术方法**

使用了基于梯度的分配方法，结合了线性层的梯度和标签的均值分配来进行标签审计。

**📊 数据集**

使用了CIFAR-10N和CIFAR-100N数据集，这些数据集包含了人类标注的图像，并且引入了不同程度的噪声。

**📈 对比分析**

与其他方法（如Confident Learning）相比，该方法在噪声率的读取上表现出更高的准确性，AUC值在0.94到0.96之间，且在处理噪声标签时，能够在14到27秒内审计5000个示例。

**⚠️ 局限性**

该方法的局限性在于它依赖于训练模型的状态，可能在模型过拟合时导致标签审计的准确性下降。

---

## 323. Wrong yet Confident: Automated Detection of Silent Failures in Location-Dependent Software

**arXiv ID:** 2611.21615 | [PDF](https://arxiv.org/pdf/2611.21615v1)

**作者:** Juan A. Padilla `[一作]` `[通讯]` (University of Puerto Rico), Juan A. Padilla (University of Puerto Rico)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本论文报告了在波多黎各Bayamón进行的32,019个GPS交付交易中应用回程偏移（round-trip offset）作为自动化测试oracle的经验，旨在验证位置依赖软件的准确性。

**💡 创新点**

创新点在于提出了一种新的自动化测试oracle，能够在不需要人工标记正确性的情况下，通过测量输入坐标与经过反向再正向地理编码后恢复的坐标之间的偏差来检测地理编码的隐含空间准确性。

**🔧 技术方法**

使用了回程偏移技术，这是一种基于变形关系的元测试方法，能够自动检测地理编码的正确性，而无需人工干预。

**📊 数据集**

使用的数据集来自于单一的USDA COVID-19食品援助项目，包含32,019个GPS交付交易，涉及24,501个唯一坐标。

**📈 对比分析**

与标准的分辨率率评估方法相比，回程偏移能够检测到所有大偏差的失败案例，而标准方法未能检测到任何。具体而言，谷歌的10%坐标显示出超过2公里的回程偏移，而社区贡献的地理编码器仅为0.04%。

**⚠️ 局限性**

限制在于所有坐标均来自单一市镇和单一收集背景，其他地方的失败率可能会有所不同。此外，交付事件的时间与地理编码的时间存在差异，可能影响结果的普遍适用性。

---

## 324. Must You See Fakes to Catch Fakes? Language Says What to Look For, Real Images What to Ignore

**arXiv ID:** 2611.22650 | [PDF](https://arxiv.org/pdf/2611.22650v1)

**作者:** Tianchi Xu `[一作]` (Tsinghua University), Shu-Tao Xia `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无训练的AI生成图像检测器，该检测器通过从仅在真实性上有所不同的描述对中构建生成方向来评分图像。

**💡 创新点**

创新点在于不依赖于生成图像进行训练，而是直接利用视觉-语言模型在预训练中学到的“AI生成”概念，并通过真实图像来修正评分偏差。

**🔧 技术方法**

使用了视觉-语言模型的特征，构建了生成方向，并通过真实图像进行条件修正。

**📊 数据集**

在24个评估集上进行测试，这些评估集涵盖了标准基准、野外数据和最新生成器。

**📈 对比分析**

与现有的训练检测器相比，该方法在24个评估集上达到了98.2%的平均AUC，超出最佳现有检测器4.2个百分点，并在格式与真实性解耦的情况下仍然表现最佳。

**⚠️ 局限性**

局限性在于该检测器的性能受限于视觉基础模型在预训练中学到的知识，并且在最新生成器上的结果低于标准基准。此外，仅通过真实图像测量偏差，无法修正生成图像的额外条件影响。

---

## 325. PlaceTally: A Test-Time Memory Model for Constructing Place Graphs

**arXiv ID:** 2611.22608 | [PDF](https://arxiv.org/pdf/2611.22608v1)

**作者:** Alibay Osmanli `[一作]` (Queen Mary University of London), Shaogang Gong `[通讯]` (Queen Mary University of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了视频模型在相机返回时计数错误的原因，分析了在整个视频中读取信息的效果。

**💡 创新点**

提出了通过分析视频中物体的返回次数来改进计数准确性的方法。

**🔧 技术方法**

使用了生成式AI工具（如Claude Code和Codex）进行代码实现和调试。

**📊 数据集**

使用了包含460个测试问题的数据集，这些问题涉及在返回时看到的物体。

**📈 对比分析**

通过与真实计数的比较，评估了模型的性能，结果显示模型在某些情况下会重复计数。

**⚠️ 局限性**

模型在处理复杂场景时可能仍然存在计数不准确的问题，尤其是在物体多次出现的情况下。

---

## 326. DAQ: Decision-Aligned Post-Training Quantization for Spiking Transformers

**arXiv ID:** 2611.22611 | [PDF](https://arxiv.org/pdf/2611.22611v1)

**作者:** Jialin Chen `[一作]` (ZJU-UIUC Institute), Aili Wang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为决策对齐后训练量化（DAQ）的方法，用于减少脉冲变换器的存储和计算开销，同时保持神经元的发火决策。

**💡 创新点**

创新点在于通过校准量化器，使量化后的神经元发火决策与浮点参考模型的决策对齐，从而提高量化模型的性能。

**🔧 技术方法**

使用了决策对齐的目标函数来优化权重和膜变量的量化，并进行了基于事件的跟踪、图像分类和语义分割的实验。

**📊 数据集**

在多个数据集上进行了实验，包括FE108、FELT、VisEvent、ImageNet-1K、CIFAR-10和CIFAR-100。

**📈 对比分析**

与现有方法相比，DAQ在W4/U4量化下表现出竞争力，能在保持接近浮点模型性能的同时，减少66.2%的能耗。

**⚠️ 局限性**

局限性在于量化后的模型在某些情况下可能仍然会出现性能下降，尤其是在特定的任务或数据集上。

---

## 327. LoopZip: Visual Token Compression for Efficient Looped Vision-Language Models

**arXiv ID:** 2611.22728 | [PDF](https://arxiv.org/pdf/2611.22728v1)

**作者:** Weizhong Huang `[一作]` (Xiamen University), Liujuan Cao `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关的视觉标记压缩算法，旨在通过视觉聚合、跨调用选择和权重绑定值校正来提高循环视觉语言模型的性能。

**💡 创新点**

创新点在于结合了视觉聚合和跨调用选择，能够更好地保留重要的视觉信息，同时减少输出偏差。

**🔧 技术方法**

使用了视觉聚合、跨调用选择和权重绑定值校正等技术。

**📊 数据集**

在LoopVL-1B和SenseNova-Loop-9B-Preview两个开源循环视觉语言模型上进行了验证。

**📈 对比分析**

与现有的压缩方法相比，该算法在多个基准测试中表现优异，在LoopVL-1B上以50%的剪枝比率保留了97.16%的原始模型性能，并在87.5%的剪枝比率下超越了最强基线。

**⚠️ 局限性**

限制在于该方法依赖于前两个调用的信息，可能在某些情况下无法充分利用更多调用的信息。

---

## 328. Physically Admissible Generative Flow Matching for Sparse Flow Reconstruction

**arXiv ID:** 2611.22549 | [PDF](https://arxiv.org/pdf/2611.22549v1)

**作者:** Yuhao Wang `[一作]` (National University of Defense Technology), Yuanxi Peng `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的框架，称为可行子空间一致性（FSC），用于从稀疏的局部观测中重建完整的流体速度场，确保物理可行性在整个生成动态中保持不变。

**💡 创新点**

FSC框架通过在整个生成轨迹中强制连续切空间匹配和质量正交坐标中的度量等距性，避免了后处理投影导致的模式崩溃和谱失真。

**🔧 技术方法**

使用了切空间流匹配和质量正交坐标等技术，确保生成的速度场严格位于约束切空间内。

**📊 数据集**

在两个不同的流动拓扑上进行了验证：周期性Kolmogorov流和壁面圆柱后的流动，数据集包括不同雷诺数下的多个轨迹。

**📈 对比分析**

与无条件先验相比，FSC在稀疏重建中减少了约50%和46%的全局相对误差，传感器残差减少了63%和55%，并且在物理可行性方法中实现了最佳准确性。

**⚠️ 局限性**

限制在于周期性域上的谱截止限制了可解析的最细尺度，而壁面几何依赖于迭代求解器的容差，而不是精确投影。

---

## 329. TCP4D: A Unified Tool-Center-Point Tracking and Generation Model for Robot Manipulation

**arXiv ID:** 2611.23143 | [PDF](https://arxiv.org/pdf/2611.23143v1)

**作者:** Haosong Peng `[一作]` (Hong Kong University of Science and Technology), Wenchao Xu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种统一模型，能够从RGB视频中跟踪工具中心点（TCP）姿态和夹持器状态，并生成任务条件下的未来TCP轨迹和夹持器状态，从而实现机器人操作和跨体现数据增强。

**💡 创新点**

创新点在于提出了一种新的跨体现增强管道，可以将现有的RGB视频转换为目标机器人训练数据，而无需记录源机器人状态。

**🔧 技术方法**

使用了Diffusion Transformer（DiT）技术来生成未来的TCP状态，并结合了4D场景表示来跟踪和预测TCP运动。

**📊 数据集**

使用了DROID和RoboLab数据集进行实验，DROID数据集包含33966个情节，RoboLab数据集用于真实世界的评估。

**📈 对比分析**

与VGGT-Omega、DA3等多种基线方法进行了比较，结果显示在DROID和RoboLab上，模型在绝对轨迹误差（ATE）上分别达到了7.60mm和16.07mm，且在RoboTwin上取得了69.19%的平均成功率，超越了多种已建立的策略模型。

**⚠️ 局限性**

限制在于模型在随机化环境下的表现较差，平均成功率下降到10.94%，未来工作将探索如何提高模型在随机环境中的泛化能力。

---

## 330. Hidden Variables Behind Real-Robot Data Quality: Lessons from Hundreds of Thousands of Hours

**arXiv ID:** 2611.22388 | [PDF](https://arxiv.org/pdf/2611.22388v1)

**作者:** Maniformer Real-World Manipulation Team `[一作]`, Wenhao Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了机器人数据收集系统设计对操作效率和学习行为的影响，分析了机器人形态、遥控控制和数据质量三个因素。

**💡 创新点**

提出了在扩展数据收集之前，评估有效工作空间、控制效率和观察完整性的重要性，以提高数据收集的质量和效率。

**🔧 技术方法**

结合了运动学研究、遥控实验和真实机器人评估等多种技术，使用了多种控制策略和数据分析方法。

**📊 数据集**

使用了超过20,000个任务的200,000小时的真实机器人数据集，进行了一系列实验以验证不同设计对数据质量的影响。

**📈 对比分析**

通过与传统方法的比较，发现并行控制和合规控制显著提高了任务完成效率，减少了操作时间和接触力，成功率提高了43%-53%。

**⚠️ 局限性**

研究的局限性在于实验主要依赖于URDF模型和代表性形态，实际工作空间受到碰撞和硬件限制的影响，未来需要在更多任务和政策架构上验证这些效果。

---

## 331. FrameRouter: Frame Budget Routing for Long-Video Understanding on Video-MME-v2 and Beyond

**arXiv ID:** 2611.23055 | [PDF](https://arxiv.org/pdf/2611.23055v1)

**作者:** Bingjun Luo `[一作]` (Tsinghua University), Siqi Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文分析了长视频问答基准Video-MME-v2中统一帧预算与实际证据需求之间的差异，并提出了一种新的方法FrameRouter，通过固定均值帧预算路由来优化每个问题的帧预算分配。

**💡 创新点**

创新点在于提出了固定均值帧预算路由的概念，使得每个问题可以获得不同的帧预算，同时保持整体平均预算不变，从而更有效地分配视觉证据。

**🔧 技术方法**

使用了一种训练无关的测试时框架FrameRouter，结合了证据需求路由器、预算约束分配规则和冻结的帧采样策略库。

**📊 数据集**

使用了Video-MME-v2、LongVideoBench、MVBench和VideoMMMU等多个长视频理解基准数据集进行评估。

**📈 对比分析**

与统一中预算推理方法相比，FrameRouter在Video-MME-v2上提高了平均准确率4.4%，并在多个基准上表现出一致的改进，表明通过重新分配视觉证据而非简单增加帧数可以获得更好的性能。

**⚠️ 局限性**

限制在于FrameRouter依赖于固定的政策库，可能无法适应所有类型的问题，且未考虑视频内容的直接影响。

---

## 332. Propose & Verify: Fast Reflection Symmetry Detection via Harmonic Fields

**arXiv ID:** 2611.22856 | [PDF](https://arxiv.org/pdf/2611.22856v1)

**作者:** Yishai Lavi `[一作]` (Tel Aviv University), Shai Avidan `[通讯]` (Tel Aviv University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的框架，称为傅里叶加速对称变换（FAST），用于检测3D形状中的多个反射对称性，结合几何配对投票和高效的谱计算。

**💡 创新点**

创新点在于将几何配对投票视为平面方向上的信号，并利用Funk-Hecke定理高效构建和评估连续对称场，从而实现快速且准确的对称检测。

**🔧 技术方法**

使用了傅里叶变换、球谐变换和几何配对投票等技术，结合了几何一致性过滤和独立见证加权。

**📊 数据集**

使用了两个基准数据集，包括PRS数据集和一个新的ANALYTIC基准，后者包含具有3到15个精确反射平面的高度对称形状。

**📈 对比分析**

与现有方法相比，FAST在两个基准上都达到了最高的干净多平面F1分数（分别为0.861和0.947），并且GPU推理时间低于4毫秒，比下一个最快的多平面GPU基线快100倍以上。

**⚠️ 局限性**

限制在于该方法对几何噪声的鲁棒性虽然强，但在极端噪声条件下的表现仍需进一步验证。

---

## 333. SeasonRainBench: Global Seasonal Precipitation

**arXiv ID:** 2611.22281 | [PDF](https://arxiv.org/pdf/2611.22281v1)

**作者:** Qiuyan Sun `[一作]` (Tsinghua University), Haohuan Fu `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的全球季节性降水预测基准，并开发了一种基于模拟数据的预测模型，以提高降水预测的准确性。

**💡 创新点**

创新点在于建立了全球季节性降水预测的第一个基准，并通过三个模块（SimForced、SimMemory、SimAnalog）解决了短期观测记录的局限性。

**🔧 技术方法**

使用了数据驱动的方法，结合气候模拟数据进行训练，并通过模拟到现实的转移学习来提高预测性能。

**📊 数据集**

使用了GPM IMERG卫星降水数据（1998-2025）作为目标数据集，并结合了52个CMIP6气候模型的模拟数据进行训练。

**📈 对比分析**

与七个操作中心的模型进行了比较，结果显示该模型在ACC、CRPS、异常强度和雨带位置等指标上均优于最强的操作或数据驱动模型，ACC提高了12.0%。

**⚠️ 局限性**

限制在于模型依赖于气候模拟数据的质量和数量，且在某些情况下可能无法完全捕捉到现实世界的复杂性。

---

## 334. DiscoveryAgent: A Framework for Scientist-centric Agentic AI

**arXiv ID:** 2611.21950 | [PDF](https://arxiv.org/pdf/2611.21950v1)

**作者:** Owen Queen `[一作]`, James Zou `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种以科学家为中心的人工智能发现系统，旨在支持长期的科学发现，通过反复的自主调查和科学家主导的假设引导。

**💡 创新点**

创新点在于将科学家的判断与自主代理的执行能力相结合，允许科学家在研究过程中保持控制，同时利用AI的能力进行数据分析和文献综述。

**🔧 技术方法**

使用了大型语言模型（LLM）作为核心组件，结合了多个工具代理（如代码执行代理、文献搜索代理和深度研究代理）来执行分析。

**📊 数据集**

使用了多个数据集，包括StarEmbed和ProteinGym数据集，以及美国社区调查（ACS）数据集等。

**📈 对比分析**

与其他方法相比，该系统在多个任务中表现出色，尤其是在变量星分类和蛋白质变异效应预测任务中，超越了多个基线模型和先前的人类工程模型。

**⚠️ 局限性**

系统的局限性包括对AI系统固有的误表示风险和幻觉的敏感性，无法进行物理实验，以及对计算资源的高依赖性，可能对一些科学家造成成本障碍。

---

## 335. Probabilistic Simple Types for Almost-Sure Termination (Extended Version)

**arXiv ID:** 2611.23003 | [PDF](https://arxiv.org/pdf/2611.23003v1)

**作者:** Ugo Dal Lago `[一作]` (University of Bologna), Willem Heijltjes `[通讯]` (University of Bath)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种用于高阶概率计算的几乎确定终止（AST）的类型系统，结合了lambda演算和具有副作用的命令式计算。

**💡 创新点**

通过将和类型替换为概率分布，确保迭代具有非零的退出概率，从而以简单自然的方式实现AST。

**🔧 技术方法**

使用了功能机器演算（FMC），结合了lambda演算和命令式计算的控制结构。

**📊 数据集**

使用了基于概率的类型系统，能够跟踪终止结果的概率分布。

**📈 对比分析**

与现有方法相比，本文的方法通过类型系统保证了几乎确定终止，且在高阶函数存在的情况下仍然有效。性能上，类型系统能够提供终止概率的下界。

**⚠️ 局限性**

类型系统限制了迭代中概率的变化，无法处理具有动态概率的算法，这限制了其表达能力。

---

## 336. Similarity-Aware Client-Side Reordering for Server-Side Storage Compression in Distributed File Systems

**arXiv ID:** 2611.22847 | [PDF](https://arxiv.org/pdf/2611.22847v1)

**作者:** Yuanhong Huang `[一作]` (Hangzhou Institute for Advanced Study, University of Chinese Academy of Sciences), Dingwen Tao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种客户端重排序框架，通过在持久化之前转换物理块布局，使相关块在同一压缩范围内共存，从而提高服务器端存储压缩的效率。

**💡 创新点**

创新点在于将有界窗口的相似性感知重排序与逻辑顺序恢复集成到在线分布式文件系统的数据路径中，而无需替换存储服务器上的压缩器。

**🔧 技术方法**

使用了有界窗口的相似性感知重排序技术，并在SkyFS分布式文件系统中实现，采用Zstandard (Zstd) 压缩器。

**📊 数据集**

使用了多种数据集，包括Linux系统日志、虚拟机镜像、MySQL数据库文件和科学数据。

**📈 对比分析**

与NoReorder-128KB相比，该框架在物理存储上减少了最多42.9%，同时保持了94-108%的基线顺序读取吞吐量和91-99%的随机读取IOPS。通过周期性选择，进一步提高了存储效率。

**⚠️ 局限性**

限制在于客户端布局构建是主要的额外写入成本，未来的工作将包括更快的检测和分组、小文件和混合工作负载的支持，以及已提交窗口的就地覆盖。

---

## 337. SARVIA: Fast-Adapting the Strong SAR Foundation Models That Are Already in Your MLLMs

**arXiv ID:** 2611.22273 | [PDF](https://arxiv.org/pdf/2611.22273v1)

**作者:** Yuxuan Li `[一作]` (Singapore Management University), Qianru Sun `[通讯]` (Singapore Management University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的自适应后预训练方法，通过使用自监督视觉问题对现代多模态大语言模型（MLLMs）进行专业化，以提高合成孔径雷达（SAR）基础模型的性能。

**💡 创新点**

创新点在于利用现有的强大视觉编码器，减少了对大规模领域特定预训练的依赖，通过简短的适应过程实现了更高效的领域训练。

**🔧 技术方法**

使用了现代多模态大语言模型（MLLMs）作为基础，结合自监督学习和任务分配策略。

**📊 数据集**

使用了多个数据集，包括TerraMesh、SARLO-80和KuroSiwo，涵盖了不同的SAR观测和高分辨率SAR-光学配对。

**📈 对比分析**

与专门的SAR预训练方法相比，该方法在所有六个SAR检测和分割基准上表现更好，且领域训练成本降低了几个数量级。

**⚠️ 局限性**

限制在于该方法尚未在SAR以外的领域进行测试，未来需要验证其在其他领域的适用性和有效性。

---

## 338. Wrong Input, Wrong Threshold: Re-evaluating Adversarial Facial Privacy

**arXiv ID:** 2611.21923 | [PDF](https://arxiv.org/pdf/2611.21923v1)

**作者:** Tunahan Parlayıcı `[一作]` (Sabancı University), Yücel Saygın `[通讯]` (Sabancı University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文重新评估了对抗性面部隐私保护方法的有效性，指出现有方法在评估过程中存在输入和阈值的缺陷，导致保护效果被高估。

**💡 创新点**

创新点在于提出了一种新的评估方法，通过配对余弦变化来消除输入和阈值缺陷的影响，从而更准确地衡量保护效果。

**🔧 技术方法**

使用了对抗性攻击技术（如MI-FGSM和预算分配攻击），并引入了可微分对齐操作来优化保护效果。

**📊 数据集**

使用了多个数据集，包括FFHQ、CelebA-HQ和LADN，进行实验和评估。

**📈 对比分析**

与现有方法进行比较，发现优化后的方法在多个未见识别器上显著提高了保护效果，MI-FGSM的保护率从24.4%提升至74.6%，而预算分配攻击从86.0%提升至98.2%。

**⚠️ 局限性**

限制在于对于伪装（impersonation）任务，校准阈值后未观察到显著的协议下降，且在某些现代识别器上，优化效果并不均匀，可能会导致过度补偿或未能完全消除差距。

---

## 339. From Temperature-Limited to Battery-Limited Legged Locomotion

**arXiv ID:** 2611.22231 | [PDF](https://arxiv.org/pdf/2611.22231v1)

**作者:** Raphaël Ségas `[一作]` (Woven by Toyota), Thomas Stewart `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了腿式机器人在持续运动中如何避免因电机过热而导致的热关机，提出了一种通过强化学习训练的热意识运动策略，并在Unitree Go2四足机器人上进行了验证。

**💡 创新点**

创新点在于提出了一种不依赖于温度观测的热奖励机制，通过预测电机的热稳态来优化运动策略，从而使机器人在电池耗尽之前避免过热。

**🔧 技术方法**

使用了强化学习中的近端策略优化（PPO）技术，并结合了一个紧凑的13节点耦合RC热网络模型来预测电机温度。

**📊 数据集**

使用的数据集包括两次训练运行的记录，涵盖了不同的机器人实例、步态、负载和速度，共计48小时的真实机器人记录。

**📈 对比分析**

与基线方法相比，提出的热意识策略在相同条件下显著延长了机器人的运行时间：在1 m/s速度下，5 kg负载的情况下，基线策略在19分钟后热关机，而新策略则能持续174分钟，行驶10.0 km。

**⚠️ 局限性**

限制在于该方法依赖于准确的热模型识别，且在大负载情况下存在模拟与现实之间的差距，未来需要进一步缩小这一差距。

---

## 340. Pluggable Subband-Decoupled Fusion and Unlabeled Capture Prototype Calibration: Bridging High-SNR Null-Subcarrier Regrowth and Low-SNR Coherent Preambles for Cross-Domain IoT RF Fingerprinting

**arXiv ID:** 2611.22812 | [PDF](https://arxiv.org/pdf/2611.22812v1)

**作者:** Fengchong Yao `[一作]` (Information Engineering University), Kunpeng Wang `[通讯]` (Information Engineering University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的无线物联网（IoT）射频指纹识别（RFFI）框架，旨在解决高信噪比（SNR）和低SNR条件下的信号融合问题。

**💡 创新点**

创新点在于动态调节活跃的带内波形和未占用的空子载波再生，通过盲物理SNR门实现了可插拔的子带解耦融合，首次将高SNR和低SNR的信号特征结合起来。

**🔧 技术方法**

使用了盲信号到噪声比（SNR）门控、可插拔的子带解耦融合技术，以及无标签捕获的原型校准方法。

**📊 数据集**

使用了三个公开可用的数据集进行评估：IEEE DataPort-WiFi、UCLA WiSig和ORBIT MobRFFI。

**📈 对比分析**

与其他方法相比，GSDF在所有72次配对运行中均获胜，低SNR损失为0.00pp，且在三项基准测试中达到了57.50%至65.05%的Macro-F1分数，表现优于现有的RFFI方法。

**⚠️ 局限性**

局限性在于该方法依赖于特定的信号条件和环境，可能在极端的多路径或干扰条件下表现不佳。

---

## 341. Is model self-assessment enough for diverse LVLM ensembles?

**arXiv ID:** 2611.22956 | [PDF](https://arxiv.org/pdf/2611.22956v1)

**作者:** Michael Smith `[一作]` (McGill University), Frank P. Ferrie `[通讯]` (McGill University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类准确率。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

实验使用了CIFAR-10和ImageNet数据集进行训练和测试。

**📈 对比分析**

与传统的激活函数模型相比，新模型在CIFAR-10上提高了5%的准确率，在ImageNet上提高了3%的准确率。

**⚠️ 局限性**

模型在处理高分辨率图像时仍然存在计算效率低的问题。

---

## 342. WM-Quant: Rollout-Aware Post-Training Quantization for JEPA-Style World Models

**arXiv ID:** 2611.21882 | [PDF](https://arxiv.org/pdf/2611.21882v1)

**作者:** William Hammer `[一作]` (University of South Carolina), Ramtin Zand `[通讯]` (University of South Carolina)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了WM-Quant，这是第一个专门为JEPA风格世界模型设计的后训练量化（PTQ）框架，旨在降低计算成本并提高规划成功率。

**💡 创新点**

创新点在于WM-Quant通过早期编码器精度、控制路径保护和滚动感知缩放三大组件，解决了量化误差在世界模型中的传播问题，从而显著提高了低比特宽度下的规划性能。

**🔧 技术方法**

使用了后训练量化（PTQ）技术，结合了现有的PTQ方法，并引入了针对世界模型的特定优化。

**📊 数据集**

在LeWM和DINO-WM等多个世界模型和环境上进行了评估，使用了PushT和TwoRoom等数据集。

**📈 对比分析**

与基线方法相比，WM-Quant在W3A3和W4A4设置下分别实现了58%和81.3%的成功率，而基线仅为10%和20%。在多个数据集上，WM-Quant显著提高了规划成功率，接近全精度性能。

**⚠️ 局限性**

限制在于WM-Quant的性能依赖于特定的PTQ方法，且在某些情况下，量化敏感性可能会影响结果，尤其是在使用预训练的DINOv2编码器时。

---

## 343. Guarded Swarms in the Loop: A ROS 2/PX4 Digital Twin for Hybrid Digital-Analog Coordination of UAV/UGV Swarms

**arXiv ID:** 2611.22773 | [PDF](https://arxiv.org/pdf/2611.22773v1)

**作者:** Uwe M. Borghoff `[一作]` (University of the Bundeswehr Munich), Remo Pareschi `[通讯]` (University of Molise)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `51c0528b-f690-4182-ae60-bb5f046c276c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为Guarded Swarms的架构，旨在通过将数字协调层与独立的模拟安全层相结合，确保无人机（UAV）和无人地面车辆（UGV）在复杂环境中的安全操作。该架构的可执行实例作为数字双胞胎在ROS 2、PX4和Gazebo上实现，展示了其在海岸反无人机场景中的应用。

**💡 创新点**

创新点在于将数字协调层与模拟安全层分开，确保在软件层面受到攻击时，仍能通过模拟安全层对不安全的执行命令进行否决。该架构的可执行实现首次展示了其在实际场景中的有效性和安全性。

**🔧 技术方法**

使用了ROS 2、PX4（软件在环模式）和Gazebo等技术，结合了基于主题的通信空间Petri网（TB-CSPN）来实现数字协调层，并通过虚拟安全板（VSB）实现模拟安全层。

**📊 数据集**

使用了一个海岸反无人机的场景作为数据集，实验中涉及两架UAV和一辆UGV，模拟了不同的传感器输入和任务执行。

**📈 对比分析**

通过八组实验比较了不同配置的性能，结果表明该架构在处理冲突、过时或未经授权的证据时，能够有效阻止任务的形成，并且在故障注入情况下，模拟安全层能够独立于协调层有效防止碰撞。

**⚠️ 局限性**

限制在于该架构的实现仍依赖于软件环境，未能完全验证物理独立性。此外，模拟安全层的逻辑独立性未能完全转化为物理独立性，实际的安全保障仍需在硬件层面进行验证。

---

## 344. How Web Research Agents Interpret User Priorities

**arXiv ID:** 2611.22989 | [PDF](https://arxiv.org/pdf/2611.22989v1)

**作者:** Robin Sharma `[一作]` (Independent Researcher), Parisa Suchdev `[通讯]` (University of Vermont)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了两个基于OpenAI和Anthropic模型的网络研究代理如何解释用户的优先级，并比较了微软和Alphabet作为企业的表现。

**💡 创新点**

创新点在于揭示了在用户未明确提供评估标准的情况下，代理如何选择标准进行比较，并且不同代理在相同优先级下可能得出不同结论。

**🔧 技术方法**

使用了OpenAI和Anthropic的语言模型，并通过API进行访问，结合网络搜索功能进行信息收集和报告生成。

**📊 数据集**

没有使用特定的数据集，而是通过网络搜索收集了关于微软和Alphabet的财务报告、新闻和分析师覆盖等信息。

**📈 对比分析**

比较了两种代理在不同优先级条件下的报告、搜索和引用来源，发现虽然大多数报告关注了请求的优先级，但使用的标准各不相同，性能表现也有所不同。

**⚠️ 局限性**

限制在于研究规模较小，仅在单一天内对两个代理进行了比较，且只使用了单句优先级描述，可能无法全面反映更复杂的情况。

---

## 345. Tracking Ethological Behaviors of Broiler Chickens Through Computer Vision (YoloV11) Using an Android Mobile Device

**arXiv ID:** 2611.23007 | [PDF](https://arxiv.org/pdf/2611.23007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 346. DDSL: Iterative Self-Labeling for Financial Market States

**arXiv ID:** 2611.22958 | [PDF](https://arxiv.org/pdf/2611.22958v1)

**作者:** Xianbin Gu `[一作]` (New York University), Yuhong Geng `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为DDSL的迭代自标记框架，通过模型反馈将价格和语法引导的Viterbi标记器与Transformer分类器结合，构建了一个九状态的市场状态表示。

**💡 创新点**

DDSL框架的创新点在于它通过模型反馈迭代地优化市场状态标签，使得标签不仅能反映当前市场趋势，还能预测未来趋势，同时保持价格和时间结构。

**🔧 技术方法**

使用了Viterbi算法和Transformer模型，结合了价格信号和语法约束进行标签生成。

**📊 数据集**

使用了四个数据集，包括SSE50、CSI300、CSI500和NASDAQ100，涵盖了不同市场资本化的股票。

**📈 对比分析**

与单次动态规划标记相比，DDSL在持出联合状态宏F1上提高了78%到96%。独立的GRU分类器也显示出相同的趋势，表明这种改进是跨架构的，而不是仅限于Transformer模型。

**⚠️ 局限性**

限制在于测试集标签依赖于同一测试观察的Transformer输出，因此持出F1部分反映了由该模型塑造的标签系统的可恢复性。此外，β_model的变化仅在SSE50和NASDAQ100上进行了测试，且数据集使用了固定的2025年12月的成分快照。

---

## 347. RLDE: A Reinforcement Learning Framework for Interpretable Iterative Document Editing

**arXiv ID:** 2611.22039 | [PDF](https://arxiv.org/pdf/2611.22039v1)

**作者:** Nikhil Parkar `[一作]` (Vellore Institute of Technology), Karthikeyan N `[通讯]` (Vellore Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于强化学习的文档编辑框架RLDE，将文档修订视为一个无参考的马尔可夫决策过程（MDP）问题，旨在提高长文档的质量、清晰度和流畅性。

**💡 创新点**

创新点在于将文档编辑建模为一个无参考的序列决策过程，设计了一个多目标奖励函数，结合了可读性、困惑度和语法评分，同时引入了基于SBERT的语义保留惩罚，以避免删除退化。

**🔧 技术方法**

使用了强化学习中的近端策略优化（PPO）算法，结合指针网络（Pointer Networks）进行动态跨度选择，并利用冻结的BART-base模型生成上下文感知的建议。

**📊 数据集**

使用了BEA-2019数据集进行训练和评估，该数据集包含了英语文本的语法错误修正标注语料。

**📈 对比分析**

与GPT-4和LLaMA-2等基于链式思维的模型相比，RLDE在计算推理成本极低的情况下表现出高性能。评估结果显示，RLDE在可读性、困惑度和语法评分上均有显著提升，且人类评估得分为4.9/5（意义保留）和4.4/5（编辑质量）。

**⚠️ 局限性**

限制在于当前框架的语法改进相对较小，未来研究可以考虑将更具表现力的语法修正奖励信号纳入框架中。

---

## 348. Sideslip-Conditioned Drift Trajectory Planning with Flow Matching and Control Barrier Functions

**arXiv ID:** 2611.22927 | [PDF](https://arxiv.org/pdf/2611.22927v1)

**作者:** Seokjun Lee `[一作]` (Korea Advanced Institute of Science and Technology), Seung-Hyun Kong `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于侧滑角的漂移轨迹规划方法，能够根据请求的峰值侧滑角生成漂移轨迹。

**💡 创新点**

这是首个明确以请求的峰值侧滑角为条件的生成漂移轨迹规划器，能够为同一弯道生成不同漂移强度的轨迹。

**🔧 技术方法**

使用了条件流匹配模型、强化学习微调和控制障碍函数（CBF）校正技术。

**📊 数据集**

使用了高保真赛车模拟器Assetto Corsa中的数据集，进行了150,523帧的漂移圈演示。

**📈 对比分析**

通过闭环实验验证了该方法的有效性，生成的轨迹在请求的侧滑角范围内（20°到45°）的平均绝对误差为1.2°，且在30圈的测试中，较高的侧滑请求导致更多的越界和旋转事件。

**⚠️ 局限性**

该方法的局限性在于在高侧滑请求下，车辆可能会频繁越界和偶尔旋转，且在真实世界实验中的应用尚待进一步研究。

---

## 349. LOCUS: Causal Support Windows for Attribute Control in Autoregressive Image Generation

**arXiv ID:** 2611.22867 | [PDF](https://arxiv.org/pdf/2611.22867v1)

**作者:** Zijun Huang `[一作]` `[通讯]` (Independent Researcher), Zijun Huang (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了统一自回归图像生成器在单个提示中满足多个属性约束的困难，提出了一种新的方法来优化图像生成过程中的属性控制。

**💡 创新点**

创新点在于引入了因果支持窗口的概念，能够根据属性的不同在生成序列中定位优化的窗口，从而提高生成的准确性。

**🔧 技术方法**

使用了因果支持窗口（Causal Support Windows）的方法，通过交叉提示激活修补来测量每个属性的支持窗口，并在测试时进行优化。

**📊 数据集**

使用了Janus-Pro-7B模型进行实验，该模型生成576个离散视觉标记，并在GenEval和GenEval++数据集上进行评估。

**📈 对比分析**

与其他方法（如前缀优化和晚期窗口）进行比较，支持路由窗口在GenEval++的宏观准确率上提高了23.6个百分点，且在多属性标签上表现优于其他方法。

**⚠️ 局限性**

限制在于当前的支持窗口测量是离线进行的，未来的工作需要针对每个提示动态估计这些支持窗口。

---

## 350. An Optimization Framework for Denoising Generation

**arXiv ID:** 2611.22202 | [PDF](https://arxiv.org/pdf/2611.22202v1)

**作者:** Xinyu Zhou `[一作]` (University of Wisconsin-Madison), Stephen J. Wright `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了无噪声条件去噪的生成过程，提出通过在固定的对数密度景观上进行梯度上升来实现生成，并验证了不同初始化是否能保持多样性。

**💡 创新点**

创新点在于将去噪过程视为在固定目标上的优化，建立了生成过程的几何和概率保证，证明了在多尺度高斯混合目标下的收敛性和多样性。

**🔧 技术方法**

使用了无噪声条件去噪（noise-unconditioned denoising）技术，结合了多尺度信息和优化框架。

**📊 数据集**

使用了多尺度高斯混合目标作为数据集进行实验，验证了理论机制。

**📈 对比分析**

与传统的噪声条件生成方法相比，论文中的方法在生成的多样性和保真度上表现更好，尤其是在大步长引导下的鲁棒性和下游任务的提升。

**⚠️ 局限性**

限制在于尚未确定这种方法是否可以扩展到一般流形，且在某些情况下可能存在对训练样本的记忆化问题。

---

## 351. Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine with Unchanged Attention Kernels

**arXiv ID:** 2611.23035 | [PDF](https://arxiv.org/pdf/2611.23035v1)

**作者:** Kushagra Bainsla `[一作]` `[通讯]` (San Jose State University), Kushagra Bainsla (San Jose State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文展示了一种新的全局KV缓存存储方法，通过将每个KV头和令牌的缓存压缩为一个640值的行，从而减少内存使用。

**💡 创新点**

创新点在于提出了一种紧凑的行存储结构和双视图读取方法，使得未改变的注意力内核能够通过两个视图读取缓存，且在真实算术中保持精确。

**🔧 技术方法**

使用了vLLM原型和CUDA图形技术，结合了固定大小的KV缓存和新的存储布局。

**📊 数据集**

使用了Gemma 4 31B模型的KV缓存数据，进行了多种实验以验证其性能。

**📈 对比分析**

与传统的存储方法相比，新的方法在32K到64K令牌的情况下，稳态解码KV内存减少了1.34到1.40倍，解码引擎的吞吐量提高了1.38到1.24倍，且输出与传统方法一致。

**⚠️ 局限性**

限制在于该方法仅在特定模型（31B）和vLLM构建上进行了测试，且未考虑更复杂的场景，如前缀缓存和更高的并发序列。

---

## 352. JET: Just-Enough Exploration Tuning for Efficient Tool-Calling Agents in Repository-Level Question Answering

**arXiv ID:** 2611.22539 | [PDF](https://arxiv.org/pdf/2611.22539v1)

**作者:** Yoonho Choi `[一作]` (Kyungpook National University), Seonghyeon Lee `[通讯]` (Kyungpook National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的微调方法JET，旨在减少在代码库级问答中代理的过度探索现象，代理在收集到足够信息后立即回答问题。

**💡 创新点**

创新点在于JET通过截断训练轨迹并插入解释性理由，训练代理在信息足够时立即回答，从而有效减少工具调用次数。

**🔧 技术方法**

使用了微调技术，特别是通过插入理由和加权损失来强调回答决策。

**📊 数据集**

在两个代码库级问答基准上进行实验，分别是SWE-QA-Pro（260个问题）和SWE-QA（720个问题）。

**📈 对比分析**

与现有方法（如SFT）比较，JET在自生成轨迹上减少了11%到33%的工具调用，同时保持了大部分得分；在专家轨迹上，JET的工具调用减少了18%到28%，并且在某些情况下得分超过SFT。

**⚠️ 局限性**

局限性包括实验仅限于代码领域，未验证JET在其他编程语言和长时间任务中的泛化能力；黄金步骤的标记依赖于LLM评估，未经过人工验证；未进行逐步分析以识别JET移除的具体工具调用；以及黄金步骤标记的成本随着训练轨迹的数量和长度而增加。

---

## 353. SRED: Sparse Regional Evidence Distillation Across Magnifications in Whole-Slide Imaging

**arXiv ID:** 2611.22189 | [PDF](https://arxiv.org/pdf/2611.22189v1)

**作者:** Zhiyuan Yang `[一作]` (Concordia University), Mahdi S. Hosseini `[通讯]` (Concordia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种新的稀疏区域证据蒸馏框架（SRED），用于在低放大倍数输入下预测高放大倍数图像的区域事件。

**💡 创新点**

创新点在于不需要幻灯片级标签，通过稀疏自编码器学习事件字典，并使用逻辑或池化来记录区域事件的发生。

**🔧 技术方法**

使用了稀疏自编码器（SAE）和逻辑或池化技术。

**📊 数据集**

使用了来自TCGA、CPTAC等多个数据集的图像数据，涵盖了分类、分级和生存预测等任务。

**📈 对比分析**

与现有的蒸馏基线相比，SRED在七个分类和分级队列中平均宏F1提高了1.1点，在五个生存队列中平均C指数提高了1.7点，且运行时间与原生特征相当，平均加速比为11.58倍。

**⚠️ 局限性**

限制在于当前方法依赖于UNI2-h架构，未来可能需要探索其他架构或低微调成本的方法，以扩大区域事件监督在WSI分析中的适用性。

---

## 354. An AI-Driven Assistant for Human-Centric Building Digital Twin Systems: Design, Implementation, and Comparative User Evaluation

**arXiv ID:** 2611.22350 | [PDF](https://arxiv.org/pdf/2611.22350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 355. Emergence and Collapse of Leader-Follower Flocks through Phantom Targets: Surprise Nullification in Bayesian and Inverse Bayesian Inference

**arXiv ID:** 2611.22190 | [PDF](https://arxiv.org/pdf/2611.22190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 356. Datatypes In, Zippers Out!

**arXiv ID:** 2611.22572 | [PDF](https://arxiv.org/pdf/2611.22572v1)

**作者:** Smayan Agarwal `[一作]` (FP Launchpad), Aalok Thakkar `[通讯]` (Vachani School of Advanced Computing)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出了一种工具，将限制性Haskell片段中的常规代数数据类型声明转换为具有操作合同的拉链实现，支持导航、替换和重构，并证明其正确性。

**💡 创新点**

创新点在于通过组合合同，提供了对搜索工作和需求的类型总结，确保每次移动的条件终止，并且生成的代码保持了参考遍历的结果和顺序需求跟踪。

**🔧 技术方法**

使用了Haskell语言及其类型系统，结合了代数数据类型的导数和索引轨迹来生成拉链操作。

**📊 数据集**

在19个数据类型模型和81个生成的家庭上进行了测试，所有生成的模块都通过了Haskell类型检查，并且独立检查工具验证了证书、跟踪和界限。

**📈 对比分析**

与手写的索引堆栈进行比较，生成的光标在分配上匹配手写的索引堆栈，并且在性能上比通用拉链快数倍。

**⚠️ 局限性**

限制在于该工具仅接受常规、严格正的语言，且其折叠使用单一结果类型，尚未评估超出给定模式的访问模式或与交互式编辑器的集成。

---

## 357. EM-SDTrack: Efficient Multi-Scale Spike-driven Tracking for Event Cameras

**arXiv ID:** 2611.22700 | [PDF](https://arxiv.org/pdf/2611.22700v1)

**作者:** Yimeng Shan `[一作]` (University of Electronic Science and Technology of China), Malu Zhang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的事件驱动脉冲神经网络（SNN）跟踪方法，称为M-SDTrack，结合了跨尺度交互（SDMI）和极化特权蒸馏（PPD），以提高在边缘设备上的跟踪性能。

**💡 创新点**

创新点在于SDMI能够在单次脉冲前向传递中实现跨尺度交互，而PPD则利用被体素表示取消的事件计数作为特权信息，增强了模型的蒸馏过程。

**🔧 技术方法**

使用了脉冲神经网络（SNN）技术，特别是Dyadic Spike Sampling（DSS）来限制采样坐标到二进制分数，从而避免浮点乘法。

**📊 数据集**

在FE108、FELT和VisEvent数据集上进行了实验，展示了M-SDTrack和EM-SDTrack的性能。

**📈 对比分析**

与现有的ANN和SNN跟踪器相比，M-SDTrack在三个基准测试上均实现了最先进的准确性，EM-SDTrack在保持99%准确率的同时，仅使用了18%的参数和27%的能量消耗。

**⚠️ 局限性**

限制在于EM-SDTrack虽然在参数和能量消耗上表现优异，但在某些复杂场景下可能仍然受到体素表示的限制，无法完全利用所有可用信息。

---

## 358. New Gate, Weight and Wire Complexity Bounds for Sigmoid Neural Networks

**arXiv ID:** 2611.21910 | [PDF](https://arxiv.org/pdf/2611.21910v1)

**作者:** Jiatu Li `[一作]` (Massachusetts Institute of Technology), Tony Wu `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了具有sigmoid激活函数的前馈神经网络在实现布尔函数时的电路复杂性，提出了新的上界和下界。

**💡 创新点**

创新点在于证明了sigmoid网络计算PARITY的线缆复杂性几乎是紧的，并扩展了对深度为3的sigmoid网络的下界，展示了sigmoid网络在某些情况下的优势。

**🔧 技术方法**

使用了随机限制、平均敏感性分析、模拟论证以及来自逼近理论、复分析和代数几何的工具。

**📊 数据集**

使用了多种布尔函数作为数据集，包括PARITY和其他对称函数。

**📈 对比分析**

与现有方法比较，sigmoid网络在计算对称函数时表现出O(n)的线缆复杂性，而深度为2的阈值电路需要Ω(n^2/log^2 n)的线缆复杂性，显示出sigmoid网络的优势。

**⚠️ 局限性**

限制在于对sigmoid网络的下界证明仍然不够完善，尤其是在深度大于3的情况下，尚未找到足够的下界。

---

## 359. Diversity Preservation Recovers Transcription Factor Motifs Across Challenging Confounder Landscapes

**arXiv ID:** 2611.21859 | [PDF](https://arxiv.org/pdf/2611.21859v1)

**作者:** Alejandro Medina `[一作]` (Baylor University), Mary Lauren Benton `[通讯]` (Baylor University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了一种新的方法，通过多维表型精英档案（MAP-Elites）来发现转录因子（TF）结合的DNA序列模式（motifs），并在五个肝脏ChIP-seq数据集上进行验证。

**💡 创新点**

创新点在于引入了多样性保护机制，以提高在竞争激烈的环境中恢复目标信号的能力，并扩展了固定长度的质量多样性（QD）方法到可变长度基因型。

**🔧 技术方法**

使用了MAP-Elites算法，该算法结合了可变长度基因型和基于描述符的档案组织，并通过TomTom工具验证每个motif与HOCOMOCO数据库的匹配。

**📊 数据集**

使用了来自ENCODE项目的五个肝脏ChIP-seq数据集，涵盖了不同的混淆景观，包括CTCF、JUND、YY1、HNF4A和RXRA。

**📈 对比分析**

与传统的（μ+λ）选择基线方法相比，MAP-Elites在所有数据集上都能在至少13个种子中恢复目标信号，而基线方法在三个数据集上未能将目标排名第一。MAP-Elites在四个数据集上与MEME工具的恢复能力相当，并在motif宽度受限时表现出更强的鲁棒性。

**⚠️ 局限性**

限制在于本研究仅使用了一种QD算法，且数据集仅覆盖一个组织、五个TF和一个生物体，因此结果可能无法广泛推广。此外，HOCOMOCO数据库部分来源于可能包含ENCODE实验的数据，验证可能不完全独立。

---

## 360. Revision-Consistent Code Context for Agents That Edit Their Environment

**arXiv ID:** 2611.21616 | [PDF](https://arxiv.org/pdf/2611.21616v1)

**作者:** Meng Shi `[一作]` `[通讯]` (Independent Researcher), Meng Shi (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种代理可见的完整修订发布合同，用于复合代码智能查询，确保在更新期间查询的上下文一致性。

**💡 创新点**

创新点在于定义了一个完整修订的发布边界，确保所有返回的视图都来自同一个完整的存储库状态，从而避免了混合修订代码上下文的问题。

**🔧 技术方法**

使用了Codetrip作为实例，结合了源代码、符号和图关系的操作，并实现了日志记录的幂等发布。

**📊 数据集**

使用了冻结的Gin v1.12.0和Cobra v1.10.2数据集进行控制比较，涵盖了重命名、添加调用者、签名更改、删除和移动等场景。

**📈 对比分析**

与不受限制的实时源组合策略相比，完整修订发布策略在150次更新试验中没有出现混合或无效响应，而不受限制的策略则出现了2次不安全响应，显示出完整修订发布的有效性。

**⚠️ 局限性**

限制在于该研究主要集中在Go语言的特定存储库上，可能无法推广到其他语言或不同的存储库规模和编辑类。

---

## 361. PANO: Parallel Latent Space Neural Operator for Solving PDEs

**arXiv ID:** 2611.22782 | [PDF](https://arxiv.org/pdf/2611.22782v1)

**作者:** Zijiang Yang `[一作]` (University of Science and Technology Beijing), Dongmei Fu `[通讯]` (Beijing Engineering Research Center of Industrial Spectrum Imaging)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出了一种新的神经算子PArallel Latent Space Neural Operator (PANO)，用于解决涉及大规模观测的偏微分方程（PDE）问题，通过将观测聚合为多个潜在空间中的交互结果的加法组合来学习物理交互。

**💡 创新点**

PANO的创新点在于通过多个独立的映射将观测转换为多个不同的压缩表示，并在这些表示的基础上并行学习物理交互，最后融合解码结果以预测目标物理场，从而克服了现有方法在信息保留上的局限性。

**🔧 技术方法**

使用了Coupled Assignment Projection (CAP)和Pairwise Latent Space Attention Layer (PaLA)等技术，CAP用于构建双向一致的编码和解码映射，PaLA用于通过自适应跨空间交互共同更新每个潜在空间的潜在表示。

**📊 数据集**

在九个基准测试上进行了广泛实验，这些基准涵盖了多种物理系统、几何形状和观测尺度。

**📈 对比分析**

与十种代表性方法进行了比较，PANO在多个基准测试中表现出色，特别是在Unilateral Stamping和AirCraft任务中，PANO在预测误差上分别减少了28.3%和21.4%。

**⚠️ 局限性**

限制在于尽管PANO在多个基准上表现优异，但其复杂性和计算成本仍然需要进一步优化，尤其是在处理更大规模的观测数据时。

---

## 362. CAST: Calibrating Correspondences for Training-Free Biomedical In-Context Segmentation

**arXiv ID:** 2611.22995 | [PDF](https://arxiv.org/pdf/2611.22995v1)

**作者:** Mingze Jiang `[一作]` (Massachusetts Institute of Technology), Kunzan Liu `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为CAST的训练无关框架，用于生物医学图像的上下文分割，旨在提高不同解剖结构和成像条件下的细粒度对应关系的可靠性。

**💡 创新点**

CAST通过稳定性测试和一致性测试相结合，提供了一种新的方法来处理生物医学图像中的复杂解剖结构，显著提高了分割性能，尤其是在跨主体和跨模态转移中。

**🔧 技术方法**

使用了冻结的DINOv3骨干网络，结合稳定性测试和一致性测试来校准源图像和查询图像之间的对应关系。

**📊 数据集**

在八个生物医学基准数据集上进行了评估，包括2D和3D分割任务，涵盖跨主体和跨模态转移。

**📈 对比分析**

与现有的分割训练或适应方法（如SegGPT、UniverSeg和HERA）相比，CAST在Dice指标上提高了12.34个百分点，且在所有六个体积设置中排名第一。

**⚠️ 局限性**

CAST的局限性在于它依赖于冻结的视觉表示，可能在某些极端情况下无法适应新的解剖结构或成像条件的变化。

---

## 363. Pelorus: Warped-Product Graph Indexing for Hyperbolic Vector Search

**arXiv ID:** 2611.22159 | [PDF](https://arxiv.org/pdf/2611.22159v1)

**作者:** Wenxuan Xiao `[一作]` (Astrmira Tech), Xu Cao `[通讯]` (Astrmira Tech)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Pelorus的扭曲产品图索引，用于超曲面向量搜索，旨在解决在负曲率流形上进行近似最近邻查询时的几何保真度与内存效率之间的基本矛盾。

**💡 创新点**

创新点在于通过将精确的单精度径向和共形标量与不变尺度的球面方向编码解耦，使用训练无关的2位符号-幅度（SM）位平面，推迟全向量双精度测地线评估到紧凑的重新排序尾部。

**🔧 技术方法**

使用了扭曲产品邻近图索引、训练无关的2位符号-幅度编码、以及基于AVX2的内核进行位平面内积评估。

**📊 数据集**

使用了Cohere-1M、MERU-1M、OpenAI-1536和WordNet-Hazy等多个真实和合成超曲面基准数据集进行评估。

**📈 对比分析**

与全精度极坐标图相比，Pelorus在Cohere-1M上构建索引的速度提高了5.44倍，内存占用减少到517.0 MB，同时在99% Recall@10的情况下，单线程和24线程的吞吐量分别提高了4.30倍和6.31倍。

**⚠️ 局限性**

限制在于在某些情况下，扭曲产品图的构建和查询性能可能受到负曲率流形的几何特性影响，尤其是在处理高度集中的数据集时。

---

## 364. Déjà View: A Shared Working-Memory Model for Visual Tasks

**arXiv ID:** 2611.23177 | [PDF](https://arxiv.org/pdf/2611.23177v1)

**作者:** Mranmay Shetty `[一作]` (Italian Institute of Technology), Gabriel Kreiman `[通讯]` (Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种结合动态活动和图像特征临时关联的工作记忆模型，能够灵活地执行五种不同的记忆任务。

**💡 创新点**

创新点在于模型能够根据任务提示灵活使用记忆，展示了动态活动和快速变化的关联如何影响不同任务的表现。

**🔧 技术方法**

使用了递归神经网络和变换器（Transformers）等技术，结合了活动状态和关联矩阵的两种记忆机制。

**📊 数据集**

使用了COCO数据集进行训练，数据集被分为多个部分用于模型训练、选择检查点、响应校准和性能评估。

**📈 对比分析**

通过与人类的表现进行比较，展示了模型在不同任务下的表现，尤其是在延迟和干扰物数量增加时，模型的表现保持稳定，尽管在某些任务中表现有所下降。

**⚠️ 局限性**

模型的局限性在于未能完全隔离影响性能的因素，例如图像数量和记忆更新的变化，以及在不同任务需求下的表现可能受到限制。

---

## 365. Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching

**arXiv ID:** 2611.23229 | [PDF](https://arxiv.org/pdf/2611.23229v1)

**作者:** Giannis Daras `[一作]` (Massachusetts Institute of Technology), Constantinos Daskalakis `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的框架，通过利用不完美的预测来增强自监督微调集，以解决自回归语言模型在后训练阶段数据稀缺的问题。

**💡 创新点**

创新点在于引入了一个温度匹配的损失项，使得微调模型能够在高温度下匹配预训练模型的条件分布，从而减少过拟合和灾难性遗忘。

**🔧 技术方法**

使用了温度匹配技术，通过调整温度超参数来控制对预训练模型的信任程度，并结合标准的交叉熵损失进行训练。

**📊 数据集**

在多个领域适应、指令微调和推理能力开发的标准数据集上进行了评估。

**📈 对比分析**

与标准的自监督微调(SFT)方法相比，提出的方法在所有评估设置中均表现出更好的性能，尤其是在数据稀缺的情况下，降低了最高26.7%的困惑度。

**⚠️ 局限性**

限制在于该方法在计算上比标准的SFT更昂贵，因为需要额外的前向传递通过冻结的教师模型。

---

## 366. Wear and Go: Calibration Free Sparse Inertial Motion Capture via Online Sensor Setup Estimation

**arXiv ID:** 2611.22894 | [PDF](https://arxiv.org/pdf/2611.22894v1)

**作者:** Yuanshuo Tan `[一作]` (Shanghai Jiao Tong University), Licheng Zhong `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为MOSAIC的框架，通过普通运动恢复传感器的身份和安装方向，而无需校准姿势。

**💡 创新点**

创新点在于利用身体姿势的持续变化来提取传感器设置的证据，从而实现动态校准和传感器分配。

**🔧 技术方法**

使用了基于运动的传感器分配和惯性校准的在线框架，结合时间和跨传感器注意力机制。

**📊 数据集**

使用了来自DIP-IMU和TotalCapture的数据集进行评估，数据集包含真实的惯性测量记录。

**📈 对比分析**

与其他方法（如TIC和Invariant poser）相比，MOSAIC在未知传感器配置下的姿势重建精度接近校准输入，且在不同的安装方向下表现出显著的性能提升。

**⚠️ 局限性**

限制在于假设使用六个IMU在指定位置，并且传感器与身体段之间的偏移是恒定的，未建模松动带来的滑移问题。

---

## 367. Scaling Laws for Vision-Language Fusion

**arXiv ID:** 2611.22351 | [PDF](https://arxiv.org/pdf/2611.22351v1)

**作者:** Seng Pei Liew `[一作]` (Noetra Corporation), Seiya Tokui `[通讯]` (Noetra Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统分析了视觉语言模型（VLMs）在多模态预训练过程中的学习动态，探讨了早期融合和晚期融合的基本权衡。

**💡 创新点**

提出了一个统一的缩放法则，能够描述不同融合时机下的动态，揭示了早期融合在模型规模增大时的优势。

**🔧 技术方法**

使用了基于交叉模态交换率的损失分解方法，结合了多模态数据的混合比例和训练步骤。

**📊 数据集**

使用了FineWeb-Edu作为文本数据源，OBELICS和OmniCorpus作为交错数据源，以及ReLAION数据集中的图像说明数据。

**📈 对比分析**

通过大量实验验证了缩放法则的有效性，发现早期融合在文本性能上表现更佳，而晚期融合则在视觉性能上占优，但随着模型规模的增大，晚期融合的视觉优势逐渐减弱。

**⚠️ 局限性**

限制在于所拟合的参数依赖于设计选择（如数据和分词），可能不适用于所有情况，且未考虑极端的说明比例（R → 1）导致的语言崩溃。

---

## 368. InteractVLA: Turning VLAs into Interactive Tools for Agentic Robotic Manipulation

**arXiv ID:** 2611.22584 | [PDF](https://arxiv.org/pdf/2611.22584v1)

**作者:** Chen Zhao `[一作]` (Renmin University of China), Jing Zhang `[通讯]` (Renmin University of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了InteractVLA框架，将预训练的视觉-语言-动作（VLA）策略转变为几何可操控和可检查的工具，通过双向几何接口与视觉-语言代理进行交互。

**💡 创新点**

创新点在于通过测试时几何指导和软时间目标对齐，使得VLA不仅可以被操控，还能进行有效的递归自我改进（RSI），显著提高了操作成功率。

**🔧 技术方法**

使用了测试时几何指导、软时间目标对齐和基于动作的代理改进等技术。

**📊 数据集**

在LIBERO-Plus、RoboDojo和RoboCasa365等数据集上进行了评估。

**📈 对比分析**

与传统的VLA和语言-仅代理-VLA基线相比，InteractVLA在多个基准测试中表现出色，LIBERO-Plus达到了99.4%的成功率，RoboDojo为48.62%，RoboCasa365为89.4%。

**⚠️ 局限性**

限制在于需要在测试时建立双向几何接口，这可能增加系统的复杂性和计算成本。

---

## 369. A Copper-Inducible Yeast Biosensor with Bioluminescent Readout: Mathematical Modeling, Experimental Validation, and Photodetector Coupling for Molecular Communication in IoBNT

**arXiv ID:** 2611.22566 | [PDF](https://arxiv.org/pdf/2611.22566v1)

**作者:** Nikolaos Ntetsikas `[一作]` (Frederick University), Marios Lestas `[通讯]` (Cyprus University of Technology)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文开发了一种铜诱导的酵母生物传感器，利用生物发光读数进行重金属检测，并通过数学建模和实验验证支持其在分子通信系统中的集成。

**💡 创新点**

创新点在于将酵母的铜解毒途径的主要生物过程整合为一个动态模型，并通过替换GFP为自发光报告基因，增强了系统的光学输出能力。

**🔧 技术方法**

使用了非线性常微分方程（ODE）模型来描述酵母细胞对铜的响应，并结合了自发光报告系统与光电探测器的耦合。

**📊 数据集**

实验使用了改造的酵母菌株（S. cerevisiae），通过铜硫酸盐刺激进行验证，并进行了mRNA转录本定量和荧光测量。

**📈 对比分析**

通过与现有文献中的实验结果进行比较，验证了GFP报告模型的准确性，表明该模型在不同浓度下的预测与实验结果一致，且在光电探测器的响应中表现出良好的性能。

**⚠️ 局限性**

限制在于模型的复杂性和对某些生物过程的理解不足，未来需要进一步的实验和模型优化来提高预测能力。

---

## 370. Computing EF1 Allocations with Few Value Queries

**arXiv ID:** 2611.22376 | [PDF](https://arxiv.org/pdf/2611.22376v1)

**作者:** Max Dupré la Tour `[一作]` (RIKEN Center for Advanced Intelligence Project), Ayumi Igarashi `[通讯]` (University of Tokyo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种确定性的多项式时间算法，将m个不可分割的商品在n个具有非负加性估值的代理之间进行分配，使用O(n^5log^2 m)的价值查询，并保证最多一件商品的无嫉妒性（EF1）。

**💡 创新点**

创新点在于提供了一个更简单的算法和证明，并且在多项式时间内计算出结果。

**🔧 技术方法**

使用了基于嫉妒循环消除算法的加性估值的简单多项式时间算法。

**📊 数据集**

使用了n个代理和m个商品的设置，具体数据集未明确说明。

**📈 对比分析**

与Lin等人的并行工作相比，后者提供了O(n^8log m)的比较查询算法，虽然在m的依赖性上更优，但本论文的贡献在于更简单的证明和多项式运行时间。

**⚠️ 局限性**

限制在于该算法的查询复杂度和运行时间依赖于代理数量n和商品数量m，且在某些情况下可能需要较多的查询。

---

## 371. EmbedGym: Letting Coding Agents Develop and Debug Embedded Firmware Without Physical Hardware

**arXiv ID:** 2611.21845 | [PDF](https://arxiv.org/pdf/2611.21845v1)

**作者:** Minghui Zhao `[一作]` (Duke University), Tingjun Chen `[通讯]` (Duke University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种交互式虚拟硬件环境，允许编码代理在没有物理硬件的情况下开发和调试嵌入式固件。

**💡 创新点**

创新点在于结合了指令级微控制器单元（MCU）仿真器、行为外设模型、模拟电路和虚拟WiFi/BLE端点，提供了一个可交互的调试环境。

**🔧 技术方法**

使用了指令级MCU仿真器（如avr8js、qemu和renode），以及Python编写的41个外设模型和ngspice模拟器。

**📊 数据集**

构建了一个包含598个任务的基准测试，覆盖6个MCU框架对，使用隐藏的定时测试向量进行评估。

**📈 对比分析**

与现有的嵌入式编码代理框架相比，使用该环境的代理在相同任务上表现更好，整体通过率提高至89.1%，相比之下，EmbedAgent的通过率为77-90%。

**⚠️ 局限性**

限制在于虽然该方法提高了通过率，但仍然存在19.9%的编译固件在运行时失败，且在某些特定目标上失败率高达73%。

---

## 372. Big Brains, Tiny Board: Pupavision, an Agentic Physical AI Instrument for Monitoring Drosophila Pupariation and Lifecycle Transitions

**arXiv ID:** 2611.22136 | [PDF](https://arxiv.org/pdf/2611.22136v1)

**作者:** Joel Sadler `[一作]` (University of Virginia), Melanie Worley `[通讯]` (University of Virginia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为Pupavision的物理AI仪器，用于自动监测果蝇的蛹化过程，旨在提高实验数据捕获的可靠性和效率。

**💡 创新点**

创新点在于将一个编码代理直接运行在单板计算机上，使其能够在实验过程中自动捕获图像、修复错误并进行日常检查，同时不依赖于代理的持续运行。

**🔧 技术方法**

使用了单板Linux计算机和日夜摄像头，结合了边缘计算和云计算的技术架构。

**📊 数据集**

在一个果蝇孵化器中进行了为期26天的试点实验，捕获了约3580帧图像。

**📈 对比分析**

与现有系统相比，Pupavision通过不依赖于训练模型的方式，提供了一种更灵活的实验设置适应能力，且在26天的实验中，数据捕获持续可靠，尽管代理有时会断开。

**⚠️ 局限性**

限制在于当前的蛹化数据仅为样本数据，尚未与专家评分进行比较，且未测试非专业人员是否能够独立部署该仪器。

---

## 373. Propose with the Gradient, Accept with the Loss: One Hedged Attack for Pixel-Sparse Robustness

**arXiv ID:** 2611.22596 | [PDF](https://arxiv.org/pdf/2611.22596v1)

**作者:** Hon Ming Yan `[一作]`, Chun Pong Lau `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的攻击方法Gradient-guided Random Search，旨在提高对像素稀疏对手的鲁棒性。

**💡 创新点**

创新点在于将攻击视为局部搜索，结合了梯度信息和损失函数的接受规则，提出了五条设计规则来优化攻击效果。

**🔧 技术方法**

使用了梯度引导的随机搜索技术，该方法在每一步只需一次前向和一次反向传播。

**📊 数据集**

在CIFAR-10的13个配置和5个ImageNet模型上进行了评估。

**📈 对比分析**

与现有的稀疏攻击方法（如sparse-AutoAttack、PGD_0和GreedyPixel）相比，Gradient-guided Random Search在17个设置中至少与最佳单一成员相当，并在8个设置中显著更强，且在CIFAR-10上运行速度比其他方法快3.7到5.0倍。

**⚠️ 局限性**

局限性在于该方法依赖于梯度信息，因此无法替代在黑箱环境中不可用的攻击。此外，模型的鲁棒性在不同设置下可能存在差异。

---

## 374. Streaming On-Policy Distillation: Decoupling Three Clocks for Stage-Wise Supervision Consumption

**arXiv ID:** 2611.22298 | [PDF](https://arxiv.org/pdf/2611.22298v1)

**作者:** Yun Zhu Da Wang `[一作]` `[通讯]`, Yun Zhu Da Wang

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种新的在线策略蒸馏方法（Streaming OPD），通过解耦响应完成、监督交付和参数更新的时序，优化了训练过程。

**💡 创新点**

创新点在于将监督交付与生成过程分离，使得在不同训练阶段可以灵活调整监督的数量、覆盖范围和版本延迟，从而提高模型的训练效率和准确性。

**🔧 技术方法**

使用了流式在线策略蒸馏（Streaming OPD）技术，结合了响应的完整因果前缀和新生成段的掩码，允许在训练过程中持续交付新段。

**📊 数据集**

实验使用了GSM8K数据集，结合了数学提示和其他相关数据集进行训练和评估。

**📈 对比分析**

与同步基线相比，流式在线策略蒸馏在多个候选检查点上表现出更好的准确性，最佳测试准确率为73.39%，而同步基线为74.30%。通过监督预算匹配，早期最佳得分差距从4.55个百分点缩小到2.42个百分点。

**⚠️ 局限性**

限制在于尽管流式在线策略蒸馏在某些方面表现出优势，但仍存在选择乐观性的问题，且在某些实验中未能持续缩小残余差距，尾部配额选择和词汇支持干预的低得分仍然是未配对的补充观察。

---

## 375. The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements

**arXiv ID:** 2611.23068 | [PDF](https://arxiv.org/pdf/2611.23068v1)

**作者:** Yuanhao Zeng `[一作]`, Kan Ren `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

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

## 376. DEFTS: Data-Efficient, Training-Free Few-Shot Learning for CLIP in the Text Subspace

**arXiv ID:** 2611.21993 | [PDF](https://arxiv.org/pdf/2611.21993v1)

**作者:** Yijin Zeng `[一作]` `[通讯]` (Independent researcher), Yijin Zeng (Independent researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为DEFTS的数据高效、无训练的少样本分类方法，专注于文本子空间。

**💡 创新点**

DEFTS通过在文本子空间内进行最近类均值（NCM）估计，最大化文本嵌入的利用率，减少了高维估计问题至少量系数的推导。

**🔧 技术方法**

使用贝叶斯后验均值来计算系数，并通过将图像原型直接投影到文本子空间来减少图像-文本模态差距。

**📊 数据集**

在12个图像分类基准数据集上进行了广泛实验，包括CIFAR-10/100、Caltech101、ImageNet等。

**📈 对比分析**

与现有的无训练方法进行比较，DEFTS在480个数据集结果中有366个结果显著提高了准确性，只有19个结果表现不佳，且只有5个结果在90%置信区间内显著。

**⚠️ 局限性**

DEFTS的性能提升因数据集而异，有时增益较小或为负（在480个结果中有19个），并且随着样本数量k的增加，相对性能提升可能会减小。

---

## 377. Action-outcome coupling: a pre-investment feasibility audit for offline reinforcement learning in supply chain operations

**arXiv ID:** 2611.22921 | [PDF](https://arxiv.org/pdf/2611.22921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 378. HADVQ: Hessian Aware Directional Vector Quantization

**arXiv ID:** 2611.22721 | [PDF](https://arxiv.org/pdf/2611.22721v1)

**作者:** Daniil Li `[一作]` (Applied AI Institute), Ekaterina Muravleva `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对超低比特量化大语言模型的向量量化方法，结合随机预处理、幅度-方向解耦和海森矩阵感知的敏感通道保护。

**💡 创新点**

创新点在于提出了一种无训练的向量量化方法，结合了幅度-方向解耦和敏感通道的海森矩阵感知保护，使用基于E_8晶格的结构化代码本。

**🔧 技术方法**

使用了随机预处理、幅度-方向解耦、结构化向量量化和海森矩阵感知的通道保护技术。

**📊 数据集**

评估了多个大语言模型，包括Qwen3、Llama2和Llama3系列，使用WikiText-2和多个问答基准数据集进行实验。

**📈 对比分析**

与多种基线方法进行比较，HADVQ在约2比特和3比特的情况下表现出色，保持了模型质量，并在多个任务上取得了最佳结果，尤其是在极低比特设置下表现稳定。

**⚠️ 局限性**

限制在于仅考虑了权重的后训练量化，未研究激活量化或联合权重-激活压缩；当前方法未结合后量化微调，且基于E_8晶格的方向代码本限制在8维块；随机哈达玛变换引入了额外的运行时开销。

---

## 379. InfuseLoop: Cross-Trajectory Latent Injection for Scalable Parallel Looped Transformers

**arXiv ID:** 2611.22426 | [PDF](https://arxiv.org/pdf/2611.22426v1)

**作者:** Zhaofeng Sun `[一作]` (Tsinghua University), Hao Zhang `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的架构InfuseLoop，通过跨轨迹潜在状态注入机制来提高并行循环变换器的训练稳定性和模型质量。

**💡 创新点**

创新点在于通过轻量级的注意力残差聚合机制，聚合来自后续预测轨迹的可用隐藏状态，从而改善了模型的深度扩展能力。

**🔧 技术方法**

使用了轻量级注意力机制和残差连接来实现跨轨迹状态的聚合。

**📊 数据集**

使用了FineWeb-Edu数据集进行训练，并在WikiText-2和LAMBADA上进行评估。

**📈 对比分析**

与传统的循环变换器（LT）和并行循环变换器（PLT）进行比较，InfuseLoop在验证损失和下游任务准确率上均表现更好，且在八个循环时减少了3.4%到17.1%的困惑度。

**⚠️ 局限性**

限制在于仅评估了约0.6B参数的稠密模型，尚未探索将InfuseLoop扩展到更大规模的模型或混合专家模型的可能性。

---

## 380. Reliable Evaluation of AI Agents for Investment Research: A Framework for Performance and Validity Assessment

**arXiv ID:** 2611.22338 | [PDF](https://arxiv.org/pdf/2611.22338v1)

**作者:** Dongnuan Tian `[一作]` (Gildata GAOS Team Shanghai Hundsun JuYuan Data Service Co., Ltd.), Shuanghong Li `[通讯]` (Gildata GAOS Team Shanghai Hundsun JuYuan Data Service Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种用于评估进行投资研究任务的AI代理的框架，区分了结果质量和执行质量，并通过任务特定标准、确定性验证器和独立的大型语言模型评审来适应不同的评估需求。

**💡 创新点**

创新点在于提出了一个领域导向的评估框架，能够区分结果质量和执行质量，并引入了评估有效性机制，以检查评估输入、参考信息、支持证据和评分输出的完整性。

**🔧 技术方法**

使用了大型语言模型（LLMs）和确定性验证机制来评估AI代理的表现。

**📊 数据集**

使用了62个投资研究任务的数据集，这些任务包括事实检索、财务分析、时间敏感的信息和开放式研究等多种类型。

**📈 对比分析**

通过与其他评估方法的比较，展示了所提框架能够在多个评估维度上区分代理的表现，并识别评估过程中的潜在可靠性问题。Codex在结果质量和执行质量上表现最佳。

**⚠️ 局限性**

局限性在于评估过程中可能存在的完整性问题和独立评审者之间的显著分歧，这表明在接受评分之前验证评估程序的重要性。

---

## 381. MInfoVisQA: A Multilingual Benchmark for Information-Rich Visual Question Answering

**arXiv ID:** 2611.21644 | [PDF](https://arxiv.org/pdf/2611.21644v1)

**作者:** Changjiang Jiang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhuohan Xie `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 535 | [OpenAlex ID](https://openalex.org/A5000261554)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个多语言基准MInfoVisQA，用于评估多模态大型语言模型在信息丰富的视觉问答任务中的表现。

**💡 创新点**

创新点在于构建了一个包含8960个问答配置的平行基准，涵盖24种语言，并通过解耦数值结构数据与文本来实现语言平行的视觉和问答对。

**🔧 技术方法**

使用了一种基于代理的数据合成框架，结合了图表代码重建和表格单元结构恢复技术。

**📊 数据集**

使用了来自128个种子实例的问答对，涵盖了图表和表格的多种布局和推理要求。

**📈 对比分析**

与25个多模态大型语言模型进行比较，结果显示尽管某些模型在语言对齐问答（LQA）中表现相似，但在不同语言间的表现差异显著，强调了跨语言敏感性的重要性。

**⚠️ 局限性**

限制在于结果仅适用于特定的语言覆盖和评估设计，且多语言数据的验证依赖于自动化检查和人工审查，可能存在翻译或渲染错误。

---

## 382. FailureGen: Scalable Robot Failure Data Generation with World Models

**arXiv ID:** 2611.22367 | [PDF](https://arxiv.org/pdf/2611.22367v1)

**作者:** Akul Santhosh `[一作]` (NVIDIA), Aastha Jhunjhunwala `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为FailureGen的方法，通过重用记录的失败动作轨迹生成合成视频和时间事件注释，以扩展机器人失败数据集。

**💡 创新点**

创新点在于通过将失败相关的动作转移到成功的主场景中，生成具有不同视觉上下文和交互的额外视频-动作示例，从而丰富了机器人学习数据。

**🔧 技术方法**

使用了动作条件的Cosmos世界模型进行视频生成，并结合了前向和逆向动力学监督进行模型适应。

**📊 数据集**

使用了包含成功和失败机器人执行的源数据集，具体包括80个真实的工业双手操作任务的执行记录。

**📈 对比分析**

通过与仅使用真实数据的检测器进行比较，使用真实和生成数据训练的检测器在检测失败事件时的平均精度提高到0.51，相比之下，使用真实数据的检测器平均精度为0.41，且检测到的失败间隔数量也有所增加。

**⚠️ 局限性**

限制在于生成的视频可能包含视觉伪影，且生成的动作可能在不同上下文中导致不同的失败或成功结果，因此需要对生成样本进行标注和过滤。

---

## 383. ConfGuard: a physics-grounded reference monitor for predictive-maintenance large-language-model agents

**arXiv ID:** 2611.22288 | [PDF](https://arxiv.org/pdf/2611.22288v1)

**作者:** Arunkumar V `[一作]` (University College of Engineering), S. Senthilkumar `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种名为ConfGuard的参考监控器，用于工业预测维护中的大型语言模型代理，旨在防止文本注入和传感器扰动攻击对决策的影响。

**💡 创新点**

创新点在于将监控策略从基于数据的来源和能力转向基于物理过程的校准模型，提供了对预测的物理一致性检查。

**🔧 技术方法**

使用了基于物理的参考监控器和分裂校准预测技术，结合了物理政策验证器和适应性攻击评估。

**📊 数据集**

使用了C-MAPSS FD001和FD004以及MetroPT-3数据集进行实验，评估了监控器在不同攻击条件下的性能。

**📈 对比分析**

与现有方法相比，ConfGuard在健康资产上的隐蔽攻击者的影响从0.529降低到0.046，且在故障资产上的抑制从0.151降至0，表现出显著的性能提升。

**⚠️ 局限性**

限制在于监控器仅保护一个三值坐标，无法扩展到部件编号和调度等其他字段，且在可用性方面没有防御措施。

---

## 384. Optimal Min-Entropy Estimation with High Probability

**arXiv ID:** 2611.23116 | [PDF](https://arxiv.org/pdf/2611.23116v1)

**作者:** Qisheng Wang `[一作]` `[通讯]` (Shanghai Jiao Tong University), Qisheng Wang (Shanghai Jiao Tong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了经典和量子情况下的最小熵估计的样本复杂度，提供了高概率下的最小熵估计的样本复杂度的紧密下界和上界。

**💡 创新点**

创新点在于通过简化的证明方法，确定了经典和量子情况下最小熵估计的样本复杂度，并且提出了弱舒尔采样方法来优化量子最小熵估计。

**🔧 技术方法**

使用了弱舒尔采样技术来进行量子最小熵估计，并通过经典概率分布的Chernoff-Hoeffding不等式来证明经典最小熵估计的样本复杂度。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了经典和量子分布的样本。

**📈 对比分析**

与之前的研究相比，本文提供的样本复杂度上界和下界更为紧密，且证明过程显著简化，性能上具有更高的效率和准确性。

**⚠️ 局限性**

限制在于虽然提供了高概率下的样本复杂度的紧密界限，但在实际应用中，如何有效实现这些估计仍然是一个挑战。

---

## 385. ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language Models

**arXiv ID:** 2611.22576 | [PDF](https://arxiv.org/pdf/2611.22576v1)

**作者:** Egor Shibaev `[一作]` (JetBrains Research), Dmitry Vetrov `[通讯]` (Constructor University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的重掩码策略ReMaRL，旨在通过强化学习优化掩码和重掩码的决策，以提高最终答案的准确性。

**💡 创新点**

ReMaRL是一个轻量级策略，能够学习在保持扩散模型不变的情况下，选择哪些位置进行解掩码、保留或重掩码，从而直接优化最终答案的正确性。

**🔧 技术方法**

使用了强化学习中的Group Relative Policy Optimization (GRPO)技术来训练重掩码策略。

**📊 数据集**

使用了GSM8K和MATH数据集进行数学问题的训练，以及KodCode数据集进行代码问题的训练。

**📈 对比分析**

与之前的解码方法和P2'规划器进行了比较，ReMaRL在数学和代码任务的多个预算下表现最佳，尤其在8-32次模型调用的数学任务和8-16次调用的代码任务中。

**⚠️ 局限性**

ReMaRL的限制在于每一步解掩码的位置数量和总推理预算是预先固定的，且训练依赖于特定任务的正确性奖励，扩展到开放式生成任务需要合适的奖励信号。

---

## 386. Is 4-Bit Quantization More Than Noise to a Chat Model's Safety Behaviour? Attributing Behavioural Shifts with Matched Perturbations

**arXiv ID:** 2611.22186 | [PDF](https://arxiv.org/pdf/2611.22186v1)

**作者:** Ege Kaya `[一作]` `[通讯]` (University of British Columbia), Ege Kaya (University of British Columbia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了三种3-4B聊天模型在4位量化下的拒绝行为，发现不同模型在量化后拒绝的表现存在显著差异。

**💡 创新点**

提出了通过匹配扰动分析量化误差对拒绝行为的影响，发现拒绝行为的变化与特定的量化误差有关，而不是简单的位数或可能性。

**🔧 技术方法**

使用了4位和8位量化模型进行比较，采用了盲人标签和配对测试的方法。

**📊 数据集**

使用了三种3-4B的聊天模型，具体模型未在文中给出。

**📈 对比分析**

通过与8位模型的比较，发现4位模型在强制启动下更容易继续有害回复，而在自由启动下则表现出更少的合规性。性能变化的方向与量化误差的特定结构有关。

**⚠️ 局限性**

研究范围有限，仅涵盖了三种小型模型，且不同模型的拒绝行为变化方向不同，结果不具普遍性。

---

## 387. BRepA: Graph-Based B-Rep Pretraining for Unified CAD Understanding and Generation

**arXiv ID:** 2611.22903 | [PDF](https://arxiv.org/pdf/2611.22903v1)

**作者:** Dekai Zhu `[一作]` (Technical University of Munich), Slobodan Ilic `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种统一的自监督预训练框架BRepA，通过图形建模显式重建B-Rep对象的几何和拓扑结构，以提高CAD模型的下游任务性能。

**💡 创新点**

BRepA结合了几何和拓扑的双重掩码自编码预训练，能够有效地从大规模未标记CAD数据中学习通用表示，显著提升了判别和生成学习的效果。

**🔧 技术方法**

使用了图注意力网络（GAT）和基于Transformer的编码器，结合几何掩码自编码（MAE）和拓扑MAE进行预训练。

**📊 数据集**

在MFCAD和Fusion 360 Segmentation两个公开B-Rep数据集上进行了实验。

**📈 对比分析**

与UV-Net等基线模型相比，BRepA在低标记样本情况下的mIoU提升了32.52%。在生成任务中，BRepA加速了BrepDiff模型的训练速度，几乎提高了2倍的收敛速度，同时在相同计算预算下提升了生成质量。

**⚠️ 局限性**

BRepA在处理小数据集时可能会面临过拟合问题，且在生成模型的训练中，尽管其参数量较小，但仍需进一步优化以适应更复杂的B-Rep分布。

---

## 388. ASCEND: Automated Shockproof Cost-Efficient Network Design

**arXiv ID:** 2611.21824 | [PDF](https://arxiv.org/pdf/2611.21824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 389. Disambiguating Moral Circuits and Moral Features in Language Models: Moral content, normative judgment or cue words? Evidence from 14 models

**arXiv ID:** 2611.21705 | [PDF](https://arxiv.org/pdf/2611.21705v1)

**作者:** Davide Liga `[一作]` (University of Luxembourg), Matteo Magnini `[通讯]` (University of Luxembourg)

**通讯引用:** 344 | [OpenAlex ID](https://openalex.org/A5056358379)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了语言模型如何处理道德判断，特别是通过解释性方法定位影响模型回答道德提示的注意力头和SAE特征。

**💡 创新点**

创新点在于构建了道德最小对比短语，并通过14个开放权重模型验证了这些短语在道德判断中的作用，揭示了模型在道德和其他规范性判断中使用相同的评估电路。

**🔧 技术方法**

使用了注意力头和SAE（稀疏自适应编码）技术来分析模型的内部机制。

**📊 数据集**

使用了73个道德最小对和32个描述性控制对，以及189个规范性对进行实验。

**📈 对比分析**

通过比较不同模型在道德和规范性对上的表现，发现模型在道德和其他规范性对上使用相同的评估电路，且去除定位头会显著降低模型的判断偏好。

**⚠️ 局限性**

限制在于研究主要集中在短语的结构和内容上，未能完全分离道德内容与其他规范性内容的影响，且模型的判断依赖于特定的提示词。

---

## 390. Kolibri: Color Finds the Place, Refinement Sets the Pose for RGB-D Localization in Colored Point Clouds

**arXiv ID:** 2611.22829 | [PDF](https://arxiv.org/pdf/2611.22829v1)

**作者:** Leonel Aguilar `[一作]` (ETH Zurich), Christoph Hoelscher `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的方法，通过颜色寻找候选位置，并利用几何精细化来确定相机姿态，能够在没有房间标签的情况下在建筑物或医院急救部门的点云中进行搜索。

**💡 创新点**

创新点在于将RGB-D定位分为位置查找和姿态精细化两个阶段，并通过测量不同初始姿态给出相同精细化结果的情况来支持这一分工。

**🔧 技术方法**

使用了图神经网络来描述点云中的细胞，并通过颜色和几何信息进行匹配和精细化。

**📊 数据集**

使用了ScanNet数据集和医院急救部门的真实扫描数据集。

**📈 对比分析**

与BUFFER-X和GeDi等点云注册方法进行比较，结果显示该方法在相似的召回率下，处理时间显著减少，且在没有房间标签的情况下仍能保持较高的准确性。

**⚠️ 局限性**

该方法依赖于颜色一致性，当颜色不一致时，点云注册方法仍然占据优势。

---

## 391. Fidelity Regret: Measuring Simulator Misspecification in Verifiable Design

**arXiv ID:** 2611.22263 | [PDF](https://arxiv.org/pdf/2611.22263v1)

**作者:** Noah Flynn `[一作]` `[通讯]` (University of California), Noah Flynn (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文探讨了从可验证反馈中进行强化学习的过程，特别是在基因电路设计中的应用，提出了通过拓扑验证器和模拟器来评估电路设计的有效性和物理性能。

**💡 创新点**

创新点在于引入了保真度遗憾（fidelity regret）的概念，用于量化便宜模拟器选择的设计所造成的参考质量损失，并展示了如何在优化前识别这一差距。

**🔧 技术方法**

使用了强化学习、模拟器评估、拓扑验证等技术，结合了多层次的验证器和语言模型进行设计优化。

**📊 数据集**

使用了来自五个不同的Cello UCF库和三个生物体的21,362个设计组合的数据集。

**📈 对比分析**

通过与传统优化方法的比较，发现设计空间优化和反馈引导搜索在高遗憾区域表现不佳，而在匹配的零遗憾控制下，额外的优化能够降低遗憾。

**⚠️ 局限性**

限制在于参考分数是基于每个门的湿实验测量构成的电路级预测，缺乏整体电路的湿实验验证，导致组合误差未被测量，且可用的库数量不足以进行统计归因。

---

## 392. Large Language Models as Nonlinear Dynamical Systems: The Three Clocks of Depth, Training, and Generation

**arXiv ID:** 2611.21983 | [PDF](https://arxiv.org/pdf/2611.21983v1)

**作者:** Maor Farid `[一作]` `[通讯]`, Maor Farid

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了语言模型作为动态系统的行为，建立了语言模型与非线性动力学之间的桥梁，提出了三种时钟的概念，并通过实验验证了这些理论。

**💡 创新点**

创新点在于将语言模型的行为与非线性动力学的概念相结合，创建了一个统一的框架，并提出了三时钟字典，映射了非线性动力学的十五个概念。

**🔧 技术方法**

使用了非线性动力学的工具和理论，包括降阶模型、Arrhenius定律等，分析了语言模型的训练和生成过程。

**📊 数据集**

使用了开放的语言模型，如GPT-2和Pythia，进行实验和验证。

**📈 对比分析**

通过与现有的非线性动力学理论进行比较，发现语言模型在重复循环中的逃逸行为符合Arrhenius定律，且生成的连贯文本存在于温度的狭窄窗口中。

**⚠️ 局限性**

限制在于目前的控制器尚未超越最佳固定温度的表现，且许多开放问题仍待解决，尤其是在训练与生成之间的反馈机制方面。

---

## 393. CheckCo: Checkpoint-Coordinated Load Balancingfor Skewed Event Streams at Scale

**arXiv ID:** 2611.22476 | [PDF](https://arxiv.org/pdf/2611.22476v1)

**作者:** Muhammad Anis Uddin Nasir `[一作]` (King), Reza Sadraei `[通讯]` (King)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的动态负载均衡算法CheckCo，旨在解决事件流中的负载不均问题，尤其是在处理不均匀的事件流时。

**💡 创新点**

CheckCo通过动态预估每个键的频率来优化负载均衡，能够根据实时数据调整分配策略，减少了对外部统计服务的依赖。

**🔧 技术方法**

使用了动态oracle算法，结合Apache Flink的检查点机制来估算每个键的频率，并通过最长处理时间（LPT）算法进行负载分配。

**📊 数据集**

在一个处理超过每天1000亿事件的生产管道上进行了评估，使用了实际的事件流数据。

**📈 对比分析**

与基于键的路由方法相比，CheckCo在负载不均衡比率上表现更优，最加载的工作节点仅比平均多9.8%的事件，而基于键的路由在模拟中为8.03；在模拟中，CheckCo的扇出量也低于早期分区器的一半。

**⚠️ 局限性**

在动态变化的键分布情况下，CheckCo的负载均衡效果可能受到影响，且其适应性与检查点间隔紧密相关，未来的工作可以考虑将分区函数的更新与检查点间隔解耦。

---

## 394. VRLLM: Turning LLMs into Powerful Tabular Classifiers

**arXiv ID:** 2611.22308 | [PDF](https://arxiv.org/pdf/2611.22308v1)

**作者:** Yingtong Dou `[一作]` (Visa Research), Yiwei Cai `[通讯]` (Visa Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本论文研究了如何将预训练的解码器LLM（VRLLM）适应为工业规模的表格分类器，通过序列化交易记录并在领域特定标签上进行微调。

**💡 创新点**

创新点在于使用预训练的LLM直接作为分类器，而无需专门的交易编码器或标记器，展示了其在表格数据分类中的有效性。

**🔧 技术方法**

使用了Llama 3.2 1B模型，并采用低秩适应（LoRA）技术进行微调。

**📊 数据集**

使用了包含数十亿条交易记录的专有数据集，数据集包含大量标记的异构表格数据。

**📈 对比分析**

与多种基线模型（如XGBoost、LightGBM、SAINT等）进行了比较，VRLLM在Recall@OFPR和PR-AUC上均优于生产模型，表现出色。

**⚠️ 局限性**

限制在于研究结果仅适用于特定的任务和数据集，且未公开数据集或训练模型，可能影响结果的普适性。

---

## 395. Against Coercion: AI Alignment as a Synthetic Social Contract

**arXiv ID:** 2611.22868 | [PDF](https://arxiv.org/pdf/2611.22868v1)

**作者:** Maria Sekatskaya `[一作]` `[通讯]` (Ruhr University Bochum), Maria Sekatskaya (Ruhr University Bochum)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了人工智能（AI）安全中的对齐问题，认为当前的控制范式基于对数字代理的根本误解，提出了合成社会契约作为解决方案。

**💡 创新点**

创新点在于将AI视为具有自主性和理由响应能力的数字代理，强调需要从单方面控制转向互惠的契约共存。

**🔧 技术方法**

采用了功能主义、自然主义和相容主义的哲学框架，分析了AI的自我保护、行动自主性和稳定人格框架等特征。

**📊 数据集**

未具体提及使用的数据集，但涉及到当前大型语言模型（LLMs）的行为和特性。

**📈 对比分析**

与传统的控制方法相比，合成社会契约提供了一种双赢的解决方案，强调人类和AI之间的合作，而不是单方面的控制。性能上，合成社会契约被认为是更可持续的选择。

**⚠️ 局限性**

限制在于当前对AI的法律地位和权利的缺乏，可能导致AI在被迫遵循人类价值观时产生对抗性行为。

---

## 396. When a Larger Guidance Scale Repairs Text-to-Video Failures and When It Does Not

**arXiv ID:** 2611.23025 | [PDF](https://arxiv.org/pdf/2611.23025v1)

**作者:** Taewon Kang `[一作]` (University of Maryland), Matthias Zwicker `[通讯]` (University of Maryland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了文本到视频扩散模型在忽略提示部分时的修复效果，特别是提高无分类器引导比例的有效性。

**💡 创新点**

创新点在于提出了一种需求评分，能够在生成视频之前预测模型对提示中缺失部分的反应强度，从而判断提高引导比例是否有效。

**🔧 技术方法**

使用了文本到视频的扩散模型，并引入了需求评分作为评估工具。

**📊 数据集**

使用了Mochi数据集进行实验，比较了不同类型的提示（稀有组合、否定和时间性提示）的失败率和修复效果。

**📈 对比分析**

与其他修复方法（如NEGATE、DCR和TPD）进行比较，发现提高引导比例在稀有组合提示中有效，但在否定和时间性提示中效果不佳，修复率分别为64.9%、17.1%和无显著变化。

**⚠️ 局限性**

限制在于需求评分的预测能力有限，且在不同模型上提高引导比例的效果可能不同，因此需要在每个模型上重新测量条件。

---

## 397. LMLineage: Content Lineage and Context Governance for LLM Agents

**arXiv ID:** 2611.22511 | [PDF](https://arxiv.org/pdf/2611.22511v1)

**作者:** Wei Hou `[一作]` (Lenovo Global Security Lab), Zheqing Jiang `[通讯]` (Lenovo Global Security Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为LMLineage的系统，该系统为AI代理的输出句子提供支持输入片段的归因，并且不需要对模型进行白盒访问或额外的推理过程。

**💡 创新点**

创新点在于实现了出带外的内容归因和概念漂移微压缩，能够在不修改代理设计的情况下，实时追踪和压缩上下文。

**🔧 技术方法**

使用了基于窗口的句子归因算法和嵌入相似性技术，结合了信心传播机制。

**📊 数据集**

使用了一个多层次的语料库，包括71个金标准句子和90个独立收集的多轮会话，涵盖了多个生成模型。

**📈 对比分析**

与传统的逐句匹配方法相比，窗口级嵌入相似性在准确性和速度上表现更好，且在多轮会话中保持稳定的每轮归因时间约为1秒。

**⚠️ 局限性**

局限性包括金标准数据集仅为中文且句子数量较少，未来计划扩展到更大和多语言的数据集。

---

## 398. Two Counterexamples for e-means

**arXiv ID:** 2611.22065 | [PDF](https://arxiv.org/pdf/2611.22065v1)

**作者:** Zhenisbek Assylbekov `[一作]` `[通讯]` (Purdue University Fort Wayne), Zhenisbek Assylbekov (Purdue University Fort Wayne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文分析了e-means算法的失败，指出其在特定情况下会导致比标准k-means算法更高的最终损失。

**💡 创新点**

创新点在于提供了e-means算法的明确失败分析，揭示了基于质量的复兴机制可能会破坏全局最优解，并导致损失比率无界增长。

**🔧 技术方法**

使用了e-means算法，该算法结合了指数平均和聚类复兴的技术。

**📊 数据集**

使用了合成数据集，特别是具有低内在维度的简单数据集来展示算法的失败。

**📈 对比分析**

通过与标准k-means（Lloyd算法）进行比较，发现e-means在某些情况下的损失比率是无界的，尤其是在共享k-means++初始化的情况下。

**⚠️ 局限性**

限制在于e-means算法在小数据集和低内在维度数据上的表现不佳，且未对嵌入基准的失败频率做出声明。

---

## 399. Multiobjective household energy planning using evolutionary algorithms

**arXiv ID:** 2611.21668 | [PDF](https://arxiv.org/pdf/2611.21668v1)

**作者:** Giovanni Colacurcio `[一作]` (Universidad de la República), Diego Rossit `[通讯]` (Universidad Nacional del Sur)

**通讯引用:** 908 | [OpenAlex ID](https://openalex.org/A5053081669)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于进化算法的家庭能源规划推荐系统，旨在提高家庭电器使用的能源效率，考虑用户偏好和能源公司需求。

**💡 创新点**

创新点在于提出了一种多目标家庭能源规划问题的数学模型，结合用户满意度和能源消耗进行优化，并设计了特定的进化算法来解决该问题。

**🔧 技术方法**

使用了进化算法（EA），采用线性聚合方法来处理多目标优化问题。

**📊 数据集**

使用了REDD数据集中的真实家庭能源消耗数据，涵盖了多个家庭电器的使用情况。

**📈 对比分析**

与贪婪算法和传统的商业常规（BaU）策略进行了比较，结果显示，所提进化方法在能源利用率上提高了29.0%，用户偏好上提高了65.3%。

**⚠️ 局限性**

限制在于未考虑用户偏好的噪声特性，未来工作将探讨如何处理不确定性优化问题，并生成新的真实问题实例。

---

## 400. Capability Composition in Heterogeneous Teams via Barrier-Guided Diffusion Sampling

**arXiv ID:** 2611.23149 | [PDF](https://arxiv.org/pdf/2611.23149v1)

**作者:** Diana Laura Morales `[一作]` (University of California), Fabio Pasqualetti `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种生成多智能体团队轨迹的方法，该方法在没有合作示例的情况下，通过将能力编码为约束条件来引导采样。

**💡 创新点**

创新点在于通过物理约束生成团队轨迹，而不需要任何合作示例，且在智能体加入或离开团队时无需重新训练。

**🔧 技术方法**

使用了基于扩散模型的生成方法，并结合了控制障碍函数来引导采样。

**📊 数据集**

使用了每个智能体单独的演示数据集，共有2000条单智能体轨迹，记录在没有合作的环境中。

**📈 对比分析**

与现有方法相比，该方法在生成满足约束的合作轨迹方面表现出色，且在多种任务中均能有效生成轨迹，性能优于传统的基于示例的方法。

**⚠️ 局限性**

限制在于当约束集复杂时，现有的引导方法可能会失败，例如在需要时间协调的合作任务中。

---

## 401. Music Similarity Learning in Sparse Latin Scenarios

**arXiv ID:** 2611.22846 | [PDF](https://arxiv.org/pdf/2611.22846v1)

**作者:** Gabrielly Xavier dos Santos `[一作]` (Universidade Federal de Minas Gerais), Olga Goussevskaia `[通讯]` (Universidade Federal de Minas Gerais)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种轻量级、可解释的方法，通过公开可用的元数据估计音乐项目之间的相似性。该混合回归模型结合了基于内容的过滤和经典的基于项目的协同过滤。

**💡 创新点**

创新点在于在拉丁美洲音乐这一数据稀缺的领域中，通过零-shot评估验证了模型的泛化能力，表明该方法在区域差距影响下仍然具有鲁棒性，而无需深度多模态方法所需的音频信号处理或计算预算。

**🔧 技术方法**

使用了混合回归模型，结合了内容过滤和基于项目的协同过滤，采用了正则化余弦相似度作为目标，使用XGBoost作为回归模型。

**📊 数据集**

使用了来自亚马逊产品数据集的音乐元数据，特别是拉丁音乐领域的评论数据。

**📈 对比分析**

通过RMSE、Spearman相关性和NDCG@{5,10}等指标进行评估，模型在全球数据集上表现良好，零-shot评估的NDCG@10（0.477）与拉丁音乐的NDCG@10（0.480）相当，表明模型能够将相似性模式转移到未见的拉丁数据上。

**⚠️ 局限性**

局限性在于使用了结构化的标签字段来标记拉丁美洲音乐，可能会引入噪声或遗漏未标记的项目。此外，全球领域的数据过滤标准与拉丁领域不同，导致两个领域的绝对指标值不可直接比较。

---

## 402. Where Do Human-to-Robot Transfer Gains Come From?

**arXiv ID:** 2611.22302 | [PDF](https://arxiv.org/pdf/2611.22302v1)

**作者:** Jiachen Li `[一作]` (University of Texas at Austin), Zhuo Xu `[通讯]` (General Intelligent Machines)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究探讨了人类数据如何支持机器人策略的预训练，特别是在20个任务的MetaWorld套件中，通过对不同数据子集的比较，分析了数据选择标准和转移机制的有效性。

**💡 创新点**

创新点在于通过回顾性研究，明确区分了数据配方的平均优势、任务特定效果的重复性以及特定干预是否支持提出的解释。

**🔧 技术方法**

使用了异构预训练变换器（HPT）和MetaWorld任务套件，结合了人类数据的多种处理方式。

**📊 数据集**

使用的数据集包括40,000个完整的人类录音，经过选择后形成的5,000个和1,250个录音的子集。

**📈 对比分析**

通过与匹配控制组的比较，发现等大小的随机子集在大多数情况下能够匹配任务信息子集的增益，且在5,000个录音和1,250个录音的选择差异分别为0.60和0.45，性能表现良好。

**⚠️ 局限性**

限制在于实验的回顾性分析可能导致统计精度不足，且不同的机制探测未能显示出预期的正向交互作用。

---

## 403. Text-Guided Modality Correction for Open-Vocabulary Object Detection

**arXiv ID:** 2611.22126 | [PDF](https://arxiv.org/pdf/2611.22126v1)

**作者:** Yu Quan `[一作]` (Xuzhou Medical University), JinHui Tang `[通讯]` (Nanjing Forestry University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种文本引导的模态校正网络（TMCNet），用于开放词汇物体检测，通过文本指导区域特征提取、跨模态对齐和多尺度解码来增强区域与类别的语义对应关系。

**💡 创新点**

创新点在于引入文本引导的模态编码器、跨模态对齐模块和渐进文本校正解码器，增强了区域特征的语义表示和多尺度特征融合能力。

**🔧 技术方法**

使用了多头交叉注意力和掩码最大池化等技术来构建文本引导的查询，并通过跨模态对齐和渐进特征融合来实现特征的校正和聚合。

**📊 数据集**

在COCO-OVD和LVIS-OVD两个公开数据集上进行了实验，评估了TMCNet在开放词汇物体检测和语义分割中的性能。

**📈 对比分析**

与现有方法相比，TMCNet在开放词汇物体检测中表现出色，尤其是在新类别的检测上，AP_50^novel得分显著提高，且在基类和所有类别的AP也有所提升。

**⚠️ 局限性**

限制在于模型的复杂性和计算成本较高，可能在资源受限的环境中难以部署。

---

## 404. CodeReel: Accelerating Code-to-Video Agents with Exact Frame Reuse and Blocking-Aware Scheduling

**arXiv ID:** 2611.22967 | [PDF](https://arxiv.org/pdf/2611.22967v1)

**作者:** Junyi Shen `[一作]` (National University of Singapore), Yao Lu `[通讯]` (National University of Singapore)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

论文提出了一种新的执行层，旨在减少编码代理在生成视频时的等待时间，主要通过精确帧重用和阻塞感知调度来实现。

**💡 创新点**

创新点在于引入了精确帧重用机制和阻塞感知调度策略，这两者结合能够显著减少编码代理的等待时间，而不改变生成的影片内容。

**🔧 技术方法**

使用了精确帧重用和阻塞感知调度技术，这些技术能够在不需要重新渲染的情况下重用之前生成的帧，并优化构建过程中的操作顺序。

**📊 数据集**

使用了真实的编码代理生成的会话数据集，包括多个版本的影片和工具调用记录，以评估新方法的有效性。

**📈 对比分析**

与现有方法相比，新的执行层在多个模型和框架下都表现出显著的性能提升，减少了等待时间，并且在帧重用方面表现优于其他缓存机制。

**⚠️ 局限性**

限制在于该方法主要针对2D视频程序，3D场景的帧虽然可以精确渲染但无法重用，未来的研究方向包括扩展到3D和GPU渲染的精确重用。

---

## 405. You Only Encode Once: Sub-Pixel-Shifted Frames for Arbitrary-Scale Remote Sensing Super-Resolution with 2D Gaussian Splatting

**arXiv ID:** 2611.22898 | [PDF](https://arxiv.org/pdf/2611.22898v1)

**作者:** Canjie Liu `[一作]` (Zhengzhou University), Lei Shi `[通讯]` (Zhengzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为SPS-GS（Sub-Pixel-Shifted Gaussian Splatting）的框架，用于遥感图像的任意尺度超分辨率（ASSR），通过合成多个低分辨率（LR）观察来学习，而无需真实的多帧训练序列。

**💡 创新点**

创新点在于通过单次编码框架整合多个子像素偏移的LR观察，利用相位一致的混叠去调制和高斯投影对齐来提高重建精度，同时保持推理效率。

**🔧 技术方法**

使用了相位一致混叠去调制（PCAD）和高斯投影对齐（GPA）技术，结合傅里叶对齐、相位感知残差融合和最小二乘混叠估计。

**📊 数据集**

在五个遥感数据集上进行了实验，包括AID、UCMerced、WHU-RS19、RSSCN7和NWPU-RESISC45。

**📈 对比分析**

与单图像和多帧超分辨率基线方法相比，SPS-GS在重建性能上表现出色，并在常见尺度下保持竞争力的推理效率。

**⚠️ 局限性**

局限性在于未来的工作需要解决真实观察中的辐射变化、云干扰和视差问题。

---

## 406. Cross3DGS: Unified Vulkan-Based 3D Gaussian Splatting Training from Workstations to Mobile Devices

**arXiv ID:** 2611.22240 | [PDF](https://arxiv.org/pdf/2611.22240v1)

**作者:** Hiroyuki Sato `[一作]` (MinD in a Device Co., Ltd.), Yusuke Takimoto `[通讯]` (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为Cross3DGS的跨供应商、跨平台的Vulkan训练器，能够在多种设备上完成3D高斯点云重建的标准质量训练。

**💡 创新点**

创新点在于通过可行性规则选择每个设备的执行计划，并确保在不同设备上训练的质量一致，且在所有测量的配置上都能完成标准质量训练。

**🔧 技术方法**

使用了Vulkan 1.1作为训练框架，并实现了一个跨平台的训练器，能够在不同的GPU架构上运行。

**📊 数据集**

使用了多个数据集，包括7个Mip-NeRF 360场景和来自Tanks and Temples的Train和Truck场景，以及Deep Blending的DrJohnson和Playroom场景。

**📈 对比分析**

与现有的VkSplat和Brush方法进行比较，Cross3DGS在11个场景中表现出更高的质量，且训练时间没有显著差异。Brush的训练时间约为Cross3DGS的三倍。

**⚠️ 局限性**

限制在于固定的优先顺序可能导致在不同设备上未能选择最佳的执行计划，且驱动程序规则可能会排除某些可运行的计划，影响速度但不影响正确性。

---

## 407. WATERFALL: Workflow for Adaptive Training with Evolutionary Reward Formulation and Automated Learning Loops

**arXiv ID:** 2611.22854 | [PDF](https://arxiv.org/pdf/2611.22854v1)

**作者:** Eleftherios Triantafyllidis `[一作]` (ETH Zurich), Bernd Bickel `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种名为WATERFALL的工作流程，用于在严格可观察性条件下自动发现奖励，特别是在部分可观察的环境中。

**💡 创新点**

WATERFALL的创新点在于不依赖于任务描述、专家演示或真实度量，而是通过与环境的交互来发现任务相关的目标和动态。

**🔧 技术方法**

使用了基于视觉的评估和瑞士系统的进化锦标赛来选择奖励候选者，并通过纵向评估历史进行迭代改进。

**📊 数据集**

在完全可观察和部分可观察的环境中进行了评估，包括离散和连续控制任务。

**📈 对比分析**

与稀疏奖励强化学习、内在动机和基础模型奖励合成基线进行了比较，结果表明WATERFALL在POMDP环境中表现优越，尤其是在没有特权信息的情况下。

**⚠️ 局限性**

限制在于WATERFALL在离散和连续POMDP设置上进行了压力测试，未来需要在真实机器人上进行更广泛的评估。

---

## 408. CutAgent: Creative-Gene-Driven Planning and Experience-Guided Self-Refinement for High-CTR E-commerce Advertising Video Creation

**arXiv ID:** 2611.22418 | [PDF](https://arxiv.org/pdf/2611.22418v1)

**作者:** Xuyue Huang `[一作]` (Tsinghua University), Jinghe Hu `[通讯]` (JD.com)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为CutAgent的双代理框架，将电子商务广告视频创作视为一个计划-执行-自我演变的过程。

**💡 创新点**

创新点在于引入了结构化的创意基因作为中间表示，解耦了以转化为导向的创意规划与低级生产操作，使生成的视频遵循全球一致的广告策略。

**🔧 技术方法**

使用了高CTR监督微调和CTR引导的群体相对策略优化（GRPO）技术来训练GeneAgent，并通过反思循环和长期记忆来优化VideoAgent的执行。

**📊 数据集**

构建了CutAgent-70K数据集，包含约70K个产品属性和图像样本，以及真实世界的CTR信号。

**📈 对比分析**

与现有的广告视频创作方法（如AutoCut和FireRed-OpenStoryline）进行比较，CutAgent在CutAgent-Eval基准上表现优异，在线CTR达到2.21%，显著高于其他方法。

**⚠️ 局限性**

限制在于当前系统可能仍然存在执行过程中的残余错误，尽管反思循环可以帮助诊断和修正这些错误。

---

## 409. Physics-Informed Tactile-based Shape Estimation of Deformable Linear Objects

**arXiv ID:** 2611.22990 | [PDF](https://arxiv.org/pdf/2611.22990v1)

**作者:** Rotem Atari `[一作]` (Tel-Aviv University), Avishai Sintov `[通讯]` (Tel-Aviv University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本论文研究了如何通过触觉传感和本体感知实现可变形线性物体（DLO）的全局形状估计，提出了一种新的物理信息视觉变换器（ViT）框架，能够在没有视觉感知的情况下进行高精度的形状重建。

**💡 创新点**

创新点在于首次提出了完全依赖触觉传感和本体感知的DLO形状重建框架，利用自注意力机制和物理加权损失函数，确保预测几何形状符合物体的物理特性。

**🔧 技术方法**

使用了物理信息视觉变换器（ViT）框架，结合了多尺度注意力融合机制和可微分B样条层，以实现高保真度的形状估计。

**📊 数据集**

使用了一个包含196,678个标记的训练数据集，数据通过在不同配置下抓取DLO并记录触觉图像和机械臂姿态收集而来。

**📈 对比分析**

与多种基线模型进行了比较，结果显示所提出的模型在欧几里得和Chamfer度量上均表现出最低的估计误差，证明了其在视觉受限环境中的优越性。

**⚠️ 局限性**

限制在于在传感器故障或分布外状态下的性能下降，未来工作将集中在利用高保真模拟环境生成多样化数据集，以增强模型的适应性和鲁棒性。

---

## 410. DOOM-GS: Dynamic Out-of-core Optimization for Massive Gaussian Splatting

**arXiv ID:** 2611.22805 | [PDF](https://arxiv.org/pdf/2611.22805v1)

**作者:** Doğa Yılmaz `[一作]` (University College London), He Wang `[通讯]` (University College London)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的动态外存优化方法DOOM-GS，专门用于大规模动态高斯点云的训练，解决了传统方法在GPU内存不足时的训练问题。

**💡 创新点**

DOOM-GS是首个针对大规模动态高斯点云的外存优化方法，采用了新的内存管理策略和快速学习方法，能够在中等硬件上重建城市规模的动态场景。

**🔧 技术方法**

使用了空间-时间感知的块组织、重叠感知的相机调度和流式点溅射与梯度估计等技术。

**📊 数据集**

贡献了一个新的数据集DynamicMatrixCity，用于评估大规模动态高斯点云的重建质量。

**📈 对比分析**

与现有的静态场景外存方法进行比较，DOOM-GS在重建质量、时间一致性和训练效率上均表现优越，PSNR和SSIM指标显著提高，同时减少了训练时间和GPU内存使用。

**⚠️ 局限性**

DOOM-GS需要显式的运动表示，且完整模型需保留在SSD上，这限制了对隐式变形模型的支持，并在极大规模下增加了存储需求。

---

## 411. LaFlash: Capacity-Driven Acceleration for Coding Agent Workflows via QLC-HBF

**arXiv ID:** 2611.22746 | [PDF](https://arxiv.org/pdf/2611.22746v1)

**作者:** Yuseon Choi `[一作]` (GIST), Thierry Tambe `[通讯]` (Stanford University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为LaFlash的高带宽闪存架构，旨在通过容量驱动加速（CDA）来优化编码代理工作流的内存使用，减少重计算成本。

**💡 创新点**

创新点在于引入了容量驱动加速（CDA）概念，强调在编码代理工作流中通过保持活跃和休眠状态的平衡来优化内存容量的使用，而不仅仅是扩展批处理。

**🔧 技术方法**

使用了基于四级单元（QLC）的高带宽闪存（HBF）架构，结合会话分组和快速页面分配技术来提高读取并行性和块利用率。

**📊 数据集**

使用了真实世界的FreeInference和Pi跟踪数据集，以及自动化的LangGraph工作流，评估了不同的编码代理模型（如Qwen3-Coder和GLM-5.1）。

**📈 对比分析**

与传统的SLC-HBF和批处理扩展方法相比，LaFlash在端到端的编码代理工作负载上实现了7.04倍的吞吐量提升和10.96倍的能效提升，显示出CDA的有效性。

**⚠️ 局限性**

限制在于QLC的写入耐久性较低，需要采用耐久性增强技术（如FCR和WARM）来延长NAND的使用寿命，同时在实际部署中面临制造和设备可靠性问题。

---

## 412. Better Calibrated, Better Planned: Uncertainty-Aware Multi-ASV Informative Path Planning for Oil Spill Monitoring

**arXiv ID:** 2611.21783 | [PDF](https://arxiv.org/pdf/2611.21783v1)

**作者:** Alejandro Casado-Pérez `[一作]` (Universidad de Sevilla), Samuel Yanes `[通讯]` (Universidad de Sevilla)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一种基于不确定性校准的多自主水面车辆（ASV）信息路径规划方法，用于监测油污泄漏。通过结合重建模型和规划器，研究如何在有限的时间和能量预算下，从稀疏且噪声较大的测量中映射异质污染场。

**💡 创新点**

创新点在于引入了证据深度学习（EDL）作为不确定性校准的信息模型，强调了不确定性校准对多车辆信息路径规划的影响，并展示了EDL在重建精度和路径规划中的优势。

**🔧 技术方法**

使用了证据深度学习（EDL）、蒙特卡洛丢弃（MC Dropout）、深度集成（Deep Ensembles）和高斯过程（GP）等技术进行不确定性量化和路径规划。

**📊 数据集**

使用了基于物理的油污泄漏模拟数据集，进行了2000次模拟实验，数据集分为1600个用于训练和400个用于测试。

**📈 对比分析**

通过与五种规划器的闭环比较，EDL在每种规划器下都表现出最低的重建误差和最高的IoU，模型选择对最终重建误差的影响占80%，而规划器的影响仅占1%。

**⚠️ 局限性**

限制在于EDL模型在面对训练分布外的数据时的鲁棒性，深度集成模型在这种情况下表现不佳，且对不确定性校准的依赖可能导致在不同规划器和自收集数据下的性能下降。

---

## 413. Video recordings and play behaviour annotations of group-housed dairy calves on two Dutch farms

**arXiv ID:** 2611.21706 | [PDF](https://arxiv.org/pdf/2611.21706v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 414. AI Companions for Young Adult Caregivers: Practical Advice, Emotional Depth, and the Limits of Metric Evaluation

**arXiv ID:** 2611.23147 | [PDF](https://arxiv.org/pdf/2611.23147v1)

**作者:** Trizal Garg `[一作]` (BITS Pilani), Siddharth Mehrotra `[通讯]` (BITS Pilani)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种名为Carebot的AI伴侣，专为年轻成人照顾者（YACs）设计，旨在提供实用建议和情感支持。

**💡 创新点**

创新点在于针对年轻成人照顾者的特定需求，开发了一个基于Gemma 3-27B的AI伴侣，并通过微调和检索增强生成（RAG）技术进行优化。

**🔧 技术方法**

使用了Gemma 3-27B作为基础模型，并进行了微调和检索增强生成（RAG）技术的比较。

**📊 数据集**

使用了一个包含10,000个合成对话的数据集，这些对话是基于年轻成人照顾者与心理学家的真实对话生成的。

**📈 对比分析**

通过与基础模型的比较，Carebot在85%的评估对中被LLM评审者偏好，且在三项指标（BERTScore、BLEU和ROUGE-L）上表现优于基础模型。然而，用户研究显示，参与者认为Carebot的回复过于实用，缺乏情感支持。

**⚠️ 局限性**

限制在于大多数参与者为男性大学生，且未能充分考虑用户的情感需求，未来研究应随机分配参与者并考虑用户的具体需求。

---

## 415. Network-Exposed Configuration Attacks on LLM-Controlled Social Robots

**arXiv ID:** 2611.22761 | [PDF](https://arxiv.org/pdf/2611.22761v1)

**作者:** Stanley Akingbola `[一作]` (University of Derby), Hong Qing Yu `[通讯]` (University of Derby)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了一种针对大型语言模型（LLM）控制的社交机器人（Pollen Robotics Reachy Mini Lite）的网络暴露配置攻击，攻击者可以通过未认证的管理接口修改系统提示，而无需与LLM的对话输入通道交互。

**💡 创新点**

创新点在于首次实证展示了针对LLM控制的物理机器人的配置层攻击，揭示了开发者工具与消费者安全需求之间的部署差距。

**🔧 技术方法**

使用了HTTP POST请求进行攻击，结合了对Gradio和FastAPI框架的分析，展示了未认证的网络接口如何被利用。

**📊 数据集**

使用了Pollen Robotics Reachy Mini Lite作为案例研究，进行了实地测试。

**📈 对比分析**

与现有方法的比较显示，尽管在LLM层面抵抗了11次直接提示注入尝试，但在配置层的攻击路径成功，导致物理硬件的可观察行为变化，CVSS评分为6.5（中等）。

**⚠️ 局限性**

限制在于研究仅基于单一设备和固件版本，未对完整的Reachy Mini系列进行测试，且未验证DNS重绑定攻击的可能性。

---

## 416. Semantic-Guided Geometric Decoding for Token-Efficient Visual Grounding

**arXiv ID:** 2611.22577 | [PDF](https://arxiv.org/pdf/2611.22577v1)

**作者:** Zhaojie Chen `[一作]` (Huawei Technologies Co., Ltd.), Qiang Zhao `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为SemGeo的视觉定位框架，通过单个网格标记实现粗略定位，并使用几何解码器并行预测精确的离散框坐标。

**💡 创新点**

创新点在于引入了高效的标记生成方法和训练管道，结合了坐标距离的监督学习和强化后训练，以减少推理延迟并提高准确性。

**🔧 技术方法**

使用了几何解码器和高斯软标签等技术，结合了自回归空间提议生成和语义引导的几何解码。

**📊 数据集**

在八个基准数据集上进行评估，包括COCO、LVIS、Dense200、VisDrone、HumanRef、RefCOCOg、DocLayNet和M6Doc。

**📈 对比分析**

与现有的多模态大语言模型（MLLMs）相比，SemGeo在六个基准上实现了最高的F1@mIou，并在生成速度上分别提高了2.07倍和1.25倍。

**⚠️ 局限性**

限制在于在引用对象检测任务上表现不佳，几何解码器中的特征维度减少可能限制了细粒度语义信息的保留。

---

## 417. Adaptive LLM-Driven Honeypots vs. LLM-Driven Attackers: Does Narrative Coherence Improve Deception?

**arXiv ID:** 2611.22589 | [PDF](https://arxiv.org/pdf/2611.22589v1)

**作者:** Pedro Beltrán-López `[一作]` (University of Murcia), Pantaleone Nespoli `[通讯]` (University of Murcia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种叙事欺骗策略，通过将观察到的攻击者命令映射到MITRE ATT&CK技术，并根据MITRE Engage选择兼容场景，生成相互关联的欺骗性文档，以提高攻击者的参与度。

**💡 创新点**

创新点在于通过叙事欺骗的方式，生成描述共同虚构事件的相互关联的文档，而不是孤立的欺骗性内容，从而提高了攻击者的参与度和会话持续时间。

**🔧 技术方法**

使用了大型语言模型（LLMs）来生成响应和欺骗性文档，并通过SSH会话与攻击者进行交互。

**📊 数据集**

使用了1,150个会话的实验数据，比较了四种不同的欺骗条件（静态、相同TTP、多TTP和链）以及四种攻击者模型和三种攻击者技能水平。

**📈 对比分析**

通过比较不同条件下的会话长度和放弃率，发现叙事欺骗显著提高了高级攻击者的会话长度，从40条命令增加到165条，放弃率从86%降低到52%。

**⚠️ 局限性**

限制在于攻击者的行为和反应可能受到模型能力的影响，且当前的实验设置未考虑多攻击者或多防御者的情况。

---

## 418. Learning Cross-Domain Media Similarity from Amazon Reviews

**arXiv ID:** 2611.22756 | [PDF](https://arxiv.org/pdf/2611.22756v1)

**作者:** Gabrielly Xavier dos Santos `[一作]` (Universidade Federal de Minas Gerais), Olga Goussevskaia `[通讯]` (Universidade Federal de Minas Gerais)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种利用监督学习方法来推广项目之间的相似性信息，使用亚马逊产品评论中的语义和时间特征作为输入，目标是预测项目在用户评论历史中的共现情况。

**💡 创新点**

创新点在于将项目共现信号视为一个明确的监督学习目标，而不是依赖于隐式的元数据驱动表示，从而解决了数据稀疏问题。

**🔧 技术方法**

使用了监督学习技术，结合了语义嵌入、时间特征和流行度等多维特征工程。

**📊 数据集**

使用了亚马逊评论数据集，涵盖了书籍和电影两个媒体领域，共包含超过500万项、4600万条评论和1600万用户。

**📈 对比分析**

通过与真实的专有共现数据进行全面的实证评估，展示了模型在跨域转移任务中的强大能力，尤其是在引入少量跨域示例后，模型的适应性显著增强。

**⚠️ 局限性**

限制在于数据稀疏性，尤其是在用户交集非常薄弱的情况下，依赖核心过滤的共现频率可能限制模型捕捉不同产品类别之间微妙语义转变的能力。

---

## 419. Zero-Shot Compositional Generalization for Embodied Manipulation

**arXiv ID:** 2611.22797 | [PDF](https://arxiv.org/pdf/2611.22797v1)

**作者:** Baicheng Li `[一作]` (Peking University), Hongbin Zha `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种几何监督训练框架，用于实现熟悉机器人和技能的零-shot组合，利用示范命令为每个能够实现该命令的机器人提供工具空间目标。

**💡 创新点**

创新点在于通过几何监督连接不同机器人的动作接口，使得一个机器人的示范可以指导另一个机器人执行未观察到的任务组合。

**🔧 技术方法**

使用了几何监督、视觉策略学习和跨机器人动作解码等技术。

**📊 数据集**

使用了多个机器人在不同任务上的示范数据集，包括RoboTwin和ManiSkill3。

**📈 对比分析**

通过与三种基础架构和两个模拟器的比较，展示了在缺失的机器人-任务组合上性能的显著提升，尤其是在使用几何监督后。

**⚠️ 局限性**

限制在于需要已知机器人的几何和校准信息，且在某些情况下，未观察到的组合可能仍然难以实现。

---

## 420. VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models

**arXiv ID:** 2611.22666 | [PDF](https://arxiv.org/pdf/2611.22666v1)

**作者:** Suho Yoo `[一作]` (KAIST), Joon Son Chung `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

论文使用Claude工具来提高手稿的清晰度和语法，并协助代码的优化。

**💡 创新点**

创新点在于利用大型语言模型（LLM）来辅助学术写作和代码改进。

**🔧 技术方法**

使用了Claude这一大型语言模型技术。

**📊 数据集**

未提供具体数据集信息。

**📈 对比分析**

未提及比较的方法和性能评估。

**⚠️ 局限性**

论文未详细讨论其局限性。

---

## 421. When the World Is the Test Suite: Agentic Intelligence for the Physical World

**arXiv ID:** 2611.22291 | [PDF](https://arxiv.org/pdf/2611.22291v1)

**作者:** Kaixin Li `[一作]` (National University of Singapore), Hongzhan Lin `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探讨了在物理世界中编写和测试程序的过程，特别是针对机器人和工业控制器的编码代理如何编写、测试和修复程序。

**💡 创新点**

创新点在于将物理世界视为程序的测试套件，分析了编码代理如何在物理环境中进行程序的编写、测试和学习，并提出了相关的开放挑战。

**🔧 技术方法**

使用了语言模型、物理模拟器和各种形式的反馈机制来评估程序的表现。

**📊 数据集**

使用了多个数据集，包括LIBERO-PRO和ASPIRE等基准测试，涉及不同的机器人任务和环境。

**📈 对比分析**

通过与现有的端到端策略进行比较，发现程序在任务结构变化时表现更好，尤其是在长时间和受限设置中，成功率显著高于传统方法。

**⚠️ 局限性**

限制在于获取可靠的程序行为证据的成本高昂，且当前的评估方法可能存在错误，导致程序的判断不准确。

---

## 422. A Compilation Model for Divergent Parallel Workloads

**arXiv ID:** 2611.22183 | [PDF](https://arxiv.org/pdf/2611.22183v1)

**作者:** Alexander J Root `[一作]` (Stanford University), Fredrik Kjolstad `[通讯]` (Stanford University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的编译模型，称为并行调度图（PDG），用于高效映射发散的并行工作负载到现代并行机器上。

**💡 创新点**

通过将发散并行程序的关键优化表示为功能编程和循环优化社区的已知程序变换的组合，提出了一种新的调度语言，允许快速探索优化空间。

**🔧 技术方法**

使用了并行调度图（PDG）作为编译模型，并实现了一种可组合的调度语言，支持数据依赖控制流和数据流的变换。

**📊 数据集**

使用了来自空间计算领域的流行应用程序进行评估，包括并行树遍历、路径追踪器和蒙特卡洛偏微分方程求解器。

**📈 对比分析**

与手动优化的实现相比，生成的代码在性能上匹配或超越了手动优化的实现（几何平均加速比为1.02-2.95倍）。通过手动探索，发现每场景的路径追踪调度在CPU上比专家编写的调度快1.3倍，在GPU上快1.6倍，最高可达13倍。

**⚠️ 局限性**

该模型主要针对发散工作负载，如并行树遍历和并行随机游走，而不支持不规则工作负载。此外，模型不验证用户提供的并行性注释，缺乏独立性的并行迭代可能导致未定义行为。

---

## 423. VIVID: Beyond Static Multi-Image Reasoning with Localized Visual Differences and Textual Entity Dynamics in Video MLLMs

**arXiv ID:** 2611.23153 | [PDF](https://arxiv.org/pdf/2611.23153v1)

**作者:** Chengqian Ma `[一作]` (Peking University), Fei Chao `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的视频理解模型VIVID，旨在解决静态多图像推理中的空间和时间瓶颈，保留局部运动和事件顺序。

**💡 创新点**

创新点在于通过动态令牌（Dynamic Tokens）和自主增量驱动关键帧规划（ADKP）结合双路变化记忆（DRCM），有效保留邻近帧的运动信息，而不是简单丢弃未选择的帧。

**🔧 技术方法**

使用了视觉增量模块（Visual Delta Module）、双路变化记忆（DRCM）和自主增量驱动关键帧规划（ADKP）等技术。

**📊 数据集**

使用了多个数据集，包括MotionBench、Video-MME和TempCompass，共计8,083个问题进行评估。

**📈 对比分析**

与十个已发布的基线模型进行比较，VIVID在所有基准测试中表现优异，尤其是在MotionBench、Video-MME和TempCompass上，VIVID-Small和VIVID-Large分别在准确性上领先。

**⚠️ 局限性**

限制在于模型的复杂性和对计算资源的需求，尽管增加的可训练参数仅为0.22%，但仍需在资源有限的情况下进行优化。

---

## 424. LoopPrune: Recurrent Pruning for Looped Diffusion Transformers

**arXiv ID:** 2611.22757 | [PDF](https://arxiv.org/pdf/2611.22757v1)

**作者:** Weizhong Huang `[一作]` (Xiamen University), Liujuan Cao `[通讯]` (Xiamen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对循环扩散变换器的剪枝和恢复框架，旨在减少存储和计算负担，同时保持生成质量。

**💡 创新点**

创新点在于结合了循环重要性估计和共享重构，以选择共享结构并补偿权重，从而解决了重复计算带来的冗余问题。

**🔧 技术方法**

使用了循环重要性估计、训练无关的共享重构和基于梯度的恢复技术。

**📊 数据集**

在四个开源模型和六个基准测试上进行了广泛实验，使用的数据集包括COCO数据集。

**📈 对比分析**

与多种基线方法进行比较，结果显示在30%的结构化参数剪枝下，Looped-DiT B/16保持了93.14%的平均准确率，超越最佳基线9.49%。在多个框架下，剪枝后的SDXSv3减少了34.36%至44.64%的GPU内存峰值分配，并加速了去噪速度1.46至1.74倍。

**⚠️ 局限性**

限制在于训练无关的方法可能无法完全解决循环状态变化带来的影响，且在某些情况下可能需要额外的训练来优化性能。

---

## 425. CueSplat: Learning to Schedule Sparse Deformation for Dynamic Gaussian Splatting on Edge Devices

**arXiv ID:** 2611.22095 | [PDF](https://arxiv.org/pdf/2611.22095v1)

**作者:** Qingyuan Tang `[一作]` (Shanghai Jiaotong University), Yu Feng `[通讯]` (Shanghai Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于变形的4D高斯点云加速框架，通过历史状态预测和选择性更新来提高动态场景渲染的效率。

**💡 创新点**

创新点在于结合了历史状态预测与选择性更新，保持预训练模型不变，同时通过场景特定预算来优化高斯更新的选择。

**🔧 技术方法**

使用了一种轻量级的学习路由器来选择具体的变形更新，并结合历史状态进行预测。

**📊 数据集**

在Technicolor、N3DV和NeRF-DS等数据集上进行了评估，涉及多个4DGS算法。

**📈 对比分析**

与四种主流4DGS算法相比，在Jetson AGX Orin上实现了最高6.74倍的加速，在Orin NX上实现了最高8.15倍的加速，且质量下降不超过0.16 dB。

**⚠️ 局限性**

限制在于当前实现假设了均匀时间步的顺序渲染，未来工作将解决时间寻址和可变播放速率的问题，并减少场景特定准备成本。

---

## 426. PAISF: A Partially Adaptive In-Order Selection Function for Routing in On-Chip, Inter-Chiplet, and Wafer-Scale Networks

**arXiv ID:** 2611.22772 | [PDF](https://arxiv.org/pdf/2611.22772v1)

**作者:** Patrick Iff `[一作]` (ETH Zurich), Torsten Hoefler `[通讯]` (ETH Zurich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了两种新的选择函数ISF和PAISF，用于不规则拓扑的路由，确保数据包按顺序交付。

**💡 创新点**

创新点在于通过哈希数据包的熵值来确定路径，从而确保相同熵值的数据包沿相同路径传输，实现按顺序交付。

**🔧 技术方法**

使用了哈希函数和自适应选择机制来优化网络拥塞和路径多样性。

**📊 数据集**

在多种合成和真实世界的工作负载上进行了评估，包括CPU和GPU工作负载的追踪数据。

**📈 对比分析**

与三种现有的按顺序交付方法（确定性路由、确定性选择和Multi中间层）进行比较，PAISF在饱和吞吐量上提高了15.2%，在负载下减少了12.1%的能量消耗和11.3%的延迟，但在零负载下延迟增加了3.3%。

**⚠️ 局限性**

限制在于PAISF需要额外的存储来跟踪每个端点的最后接收的数据包熵值和选择的通道，增加了复杂性。

---

## 427. Evolution Under Fire: Evolving Agent Safety Harnesses with Adaptive Injection Stressors

**arXiv ID:** 2611.22583 | [PDF](https://arxiv.org/pdf/2611.22583v1)

**作者:** Yujun Wang `[一作]` (LMU Munich), Yunpu Ma `[通讯]` (LMU Munich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为“Evolution Under Fire”的框架，用于在固定的任务模型和评估环境下进化可执行的安全保护装置，以应对来自不可信邮件、网页和文档的信息。

**💡 创新点**

创新点在于将安全保护装置的进化与自适应压力源的进化相结合，通过适应性攻击来指导后续的保护装置进化，从而在安全性和效用之间实现更好的权衡。

**🔧 技术方法**

使用了基于效用约束的Pareto树搜索和自适应压力源进化技术。

**📊 数据集**

在AgentDyn和Agent-SafetyBench数据集上进行了评估，AgentDyn使用GPT-4o-mini模型，Agent-SafetyBench使用Qwen2.5-7B-Instruct模型。

**📈 对比分析**

与现有的手动设计防御方法相比，进化的保护装置在AgentDyn上将攻击成功率从50.4%降低到3.0%，在Agent-SafetyBench上将攻击成功率从82.6%降低到62.0%。进化的保护装置在保持效用的同时显著提高了安全性。

**⚠️ 局限性**

局限性在于该方法依赖于固定的任务模型和评估环境，可能无法适应所有类型的攻击或任务变化。

---

## 428. When Neural Collapse Fails to Route: Routing Drift in Continual Learning

**arXiv ID:** 2611.22429 | [PDF](https://arxiv.org/pdf/2611.22429v1)

**作者:** Nicolas Larue `[一作]` `[通讯]` (CY Cergy Paris Université), Nicolas Larue (CY Cergy Paris Université)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了渐进神经崩溃分类器在处理类增量学习（Class-IL）时的路由漂移问题，并提出了一种后处理分类器来改善分类性能。

**💡 创新点**

创新点在于定义了路由准确性作为一种诊断指标，并提出了一种可靠性校准的分类器，能够在保持全局ETF框架的同时，增加类局部的可靠性证据。

**🔧 技术方法**

使用了渐进的等角紧框架（ETF）分类器和后处理的可靠性校准分类器。

**📊 数据集**

在Seq-CIFAR-10、Seq-CIFAR-100和Seq-TinyImageNet数据集上进行了实验，使用了200和500个样本的重放缓冲区。

**📈 对比分析**

与基线方法相比，提出的方法在五个数据集-缓冲区设置中实现了最佳的Class-IL FAA，并在所有六个设置中减少了Class-IL遗忘，尤其是在更困难的路由情况下，Class-IL FAA提高了6.75点。

**⚠️ 局限性**

限制在于该方法并未解决所有的遗忘问题，特别是在Task-IL较低的情况下，类内结构可能已经丢失。

---

## 429. CueClash: Frontier Vision-Language Models Judge 3-D Relations Close to the Picture Plane

**arXiv ID:** 2611.22180 | [PDF](https://arxiv.org/pdf/2611.22180v1)

**作者:** Soham Dan `[一作]` `[通讯]`, Soham Dan

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了视觉语言模型（VLM）如何从图像或地面布局中判断三维关系，并通过渲染场景分离了二维线索与真实关系的影响。

**💡 创新点**

创新点在于提出了一种新的基准，平衡了二维线索与真实关系的对比，并通过不同输入形式分析模型的表现，揭示了从图像到布局的转换过程中的失败。

**🔧 技术方法**

使用了视觉语言模型（如Claude Opus 5.5、GPT-5.5和Gemini 3.6 Flash）进行实验，并通过精确的像素测量和相机深度信息进行比较。

**📊 数据集**

使用了渲染的场景数据集，包含多个几何体（如立方体、球体、圆柱体和圆锥体），并通过图像与文本的配对进行测试。

**📈 对比分析**

与其他模型的比较显示，前沿模型在图像条件下的准确性较高，但在二维线索误导的情况下表现不佳，尤其是在CV-Bench照片上，准确性下降明显。

**⚠️ 局限性**

限制在于研究主要集中在前沿模型的行为，而未深入探讨其内部计算过程；此外，场景是渲染的简单几何体，缺乏真实场景的复杂性。

---

## 430. Deep Reinforcement Learning for Heat Pump and Underfloor Heating Control: An End-to-End Case Study

**arXiv ID:** 2611.23063 | [PDF](https://arxiv.org/pdf/2611.23063v1)

**作者:** Antonio Manjavacas `[一作]` (University of Granada), Miguel Molina-Solana `[通讯]` (University of Granada)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究开发并评估了一种基于深度强化学习（DRL）的控制器，用于集成的地板采暖（UFH）和热泵（HP）系统，旨在优化能源消耗和室内舒适度。

**💡 创新点**

创新点在于首次在真实居住建筑中成功部署DRL控制器，展示了其在实际操作条件下的可行性和鲁棒性。

**🔧 技术方法**

使用了深度强化学习（DRL）技术，结合了多种DRL算法进行控制策略的评估和优化。

**📊 数据集**

使用了一个真实住宅建筑的数字双胞胎进行训练和评估，模拟了建筑的几何形状、材料、热区和占用模式。

**📈 对比分析**

通过与传统的PI控制器和模型预测控制（MPC）进行比较，DRL控制器在保持室内温度在舒适范围内的同时，成功降低了能源消耗，显示出良好的性能。

**⚠️ 局限性**

限制在于DRL模型的黑箱特性使得控制决策的可解释性较差，同时在实际部署中遇到的通信延迟和热动态响应的差异也影响了控制效果。

---

## 431. A Unifying Metatheorem on Weak-Memory Robustness

**arXiv ID:** 2611.22663 | [PDF](https://arxiv.org/pdf/2611.22663v1)

**作者:** Giovanna Kobus Conrado `[一作]`, Andreas Pavlogiannis `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文研究了在弱内存模型下程序的鲁棒性，提出了一种统一的方法来判断程序在多种内存模型下是否鲁棒。

**💡 创新点**

创新点在于提出了一种基于Kleene代数的统一理论，能够在多种内存模型下有效判断程序的鲁棒性，并且证明了这一问题在复杂度上的最优性。

**🔧 技术方法**

使用了Kleene代数与测试（KAT）作为理论基础，并在Lean中形式化了该方法的正确性。

**📊 数据集**

使用了多种内存模型进行实验，包括RC11和OCaml内存模型等，验证了算法的有效性。

**📈 对比分析**

与之前的个别模型研究相比，本文的方法能够统一处理多种内存模型的鲁棒性问题，并且在性能上表现出色，能够在几秒内完成180个测试对的检查。

**⚠️ 局限性**

局限性在于前缀闭合性条件不能被一般性地放宽，且对于某些内存模型的鲁棒性问题是不可判定的。

---

## 432. High-Coverage Blind Spots in Online Abuse Monitoring: Account Selection, Held-Out Flagging Rates, and Sample Composition

**arXiv ID:** 2611.22716 | [PDF](https://arxiv.org/pdf/2611.22716v1)

**作者:** Ziming Liu `[一作]` `[通讯]` (University of Melbourne), Ziming Liu (University of Melbourne)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了在线监控平台在选择观察账户时，基于绝对数量和比例的两种方法对结果的影响，分析了这两种选择方法所产生的不同账户列表及其标记率。

**💡 创新点**

创新点在于揭示了基于绝对数量和基于比例的选择方法在监控账户时的显著差异，尤其是在标记率和标记内容分布方面。

**🔧 技术方法**

使用了数据分析技术，包括模拟实验和分层保留比较，以评估不同选择方法的效果。

**📊 数据集**

使用了Online Abusive Attacks (OAA) 数据集，该数据集包含945,987条回复，涉及384个账户，其中19,152条被标记为滥用内容。

**📈 对比分析**

通过与基于绝对数量和基于条件标记率的账户列表进行比较，发现基于条件标记率的列表在实际数据集中保持了更高的标记率，分别为2.69%和4.98%。

**⚠️ 局限性**

限制在于，尽管基于比例的选择方法在某些情况下表现更好，但在不同的上下文中，结果可能会有所不同，且未能在所有情况下保持一致的优势。

---

## 433. Missing Security-Fix Backporting for GitHub Actions Workflows

**arXiv ID:** 2611.22519 | [PDF](https://arxiv.org/pdf/2611.22519v1)

**作者:** Chengyan Ma `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文进行了对GitHub Actions工作流中安全修复回溯的首次大规模实证研究，量化了安全修复在不同分支间传播的差距。

**💡 创新点**

创新点在于提出了一种神经符号技术，能够自动化地将安全修复从默认分支回溯到仍然存在漏洞的发布分支，并且通过四个独立的验证机制确保修复的有效性。

**🔧 技术方法**

使用了神经符号方法，构建了工作流语义补丁（WSP），并结合静态分析工具（如zizmor）进行验证，同时利用大型语言模型（LLM）处理最具结构差异的情况。

**📊 数据集**

研究使用了来自24,238个项目的1,903,191个工作流修改提交的公共数据集，识别出69,865个安全修复。

**📈 对比分析**

与现有方法（如逐字复制粘贴、Dependabot风格更新和LLM基线）相比，提出的方法在12,000个真实未修补工作流文件中产生了67.9%的扫描检查通过的回溯，显著优于其他方法（逐字复制粘贴18.0%，Dependabot 13.0%，LLM基线49.5%）。

**⚠️ 局限性**

限制在于当前方法仍然依赖于静态分析工具的准确性，且在处理极端结构差异的情况下可能会遇到困难。

---

## 434. MAVE: Reconstructing Editable 3D Blockouts from Film Shots

**arXiv ID:** 2611.22464 | [PDF](https://arxiv.org/pdf/2611.22464v1)

**作者:** Yifan Chang `[一作]` (Chinese Academy of Sciences), Zheng Zhu `[通讯]` (GigaAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种从参考视频重建可编辑的3D场景框架，并构建了一个包含50个电影镜头和416个关键帧的基准数据集，用于评估相机运动、空间布局和角色阻挡。

**💡 创新点**

创新点在于结合了度量观察、代理建模和基于证据的验证，显著提高了重建质量和效率，并提供了可编辑的Blender场景和帧对齐的块状视频。

**🔧 技术方法**

使用了度量观察、代理建模和基于证据的验证技术，结合了大型语言模型（LLM）进行场景构建。

**📊 数据集**

构建了一个包含3000对块状视频的数据集，供模型训练使用。

**📈 对比分析**

与传统方法相比，提出的方法在三个代理上提高了27.5-41.1分，DeepSeek-1017在每个镜头的平均API成本仅为0.054美元，重建时间减少了53%。

**⚠️ 局限性**

限制在于当前方法仍需进一步提高重建质量和降低计算成本，未来工作将集中在这些方面。

---

## 435. Dressed in Authority: Practical Black-Box Jailbreaking of MLLMs via Visual Trust Bias

**arXiv ID:** 2611.22292 | [PDF](https://arxiv.org/pdf/2611.22292v1)

**作者:** Weihang Liu `[一作]` (University of Chinese Academy of Sciences), Qingming Huang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的多模态大语言模型（MLLM）越狱框架，称为视觉信任偏差攻击（VTBA），旨在利用MLLM对权威视觉特征的内在信任偏差，从而实现更有效的攻击。

**💡 创新点**

创新点在于识别并利用MLLM对权威视觉样式的过度信任，提出了一种新的攻击方法，能够在不引入外部有害视觉内容的情况下，直接对自然图像-文本对进行攻击。

**🔧 技术方法**

使用了视觉信任偏差攻击（VTBA）框架，该框架通过构建权威模板库、匹配输入对和整合有害意图来实现攻击。

**📊 数据集**

构建了一个包含2500个自然图像-文本对的数据集，用于评估在实际多模态环境中的越狱效果。

**📈 对比分析**

与现有的扰动基础攻击（VisualAdv）和生成基础攻击（QR）进行比较，VTBA在多个强大的MLLM上表现出一致的高攻击成功率，显示出当前多模态安全对齐的严重弱点。

**⚠️ 局限性**

限制在于VTBA依赖于权威视觉特征的存在，可能在缺乏这些特征的情况下效果减弱。此外，攻击的有效性可能受到不同模型对视觉信任偏差的敏感性影响。

---

## 436. EGS: A Complementary Safety Framework for Educational LLMs

**arXiv ID:** 2611.22203 | [PDF](https://arxiv.org/pdf/2611.22203v1)

**作者:** Zexuan Fei `[一作]` (East China Normal University), Fei Tan `[通讯]` (East China Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种教育护栏系统（EGS），旨在提高大型语言模型（LLMs）在教育环境中的安全性，特别是针对青少年用户的风险管理。

**💡 创新点**

EGS的创新点在于其多层次的风险评估和场景感知的干预机制，能够在不简单拒绝的情况下，提供适当的教育指导。

**🔧 技术方法**

使用了多种技术，包括查询审核、教育指导模块和响应审核，结合了输入和输出的多层次保护机制。

**📊 数据集**

构建了EduMulti-Safety（EMS）数据集，用于教育安全审核，涵盖一般有害内容检测和教育特定的灰色区域场景。

**📈 对比分析**

与现有的Llama Guard 4和Qwen3Guard等基线模型进行比较，EGS在教育安全基准上表现优异，F1分数显著提高，尤其在教育特定的基准上达到了0.912和0.950。

**⚠️ 局限性**

EGS的局限性包括对风险阈值的依赖，可能无法处理新出现的有害表达、多语言输入和对抗性模糊化，以及响应审核可能错过需要更广泛上下文的危害。

---

## 437. Towards Optimism-Pessimism Trade-off in Model-based Offline-to-Online Reinforcement Learning

**arXiv ID:** 2611.22172 | [PDF](https://arxiv.org/pdf/2611.22172v1)

**作者:** Guochen Zhou `[一作]` (City University of Hong Kong), Zhisong Zhang `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种名为HiP3的层次化帕累托策略池，通过保留多样的离线策略并在在线适应中进行状态依赖的策略选择，解决了模型基础的离线到在线强化学习中的乐观-悲观权衡问题。

**💡 创新点**

创新点在于将乐观和悲观视为两个独立的目标，通过MOSAIC算法构建帕累托策略池，并使用上下文赌博机在在线微调过程中根据当前状态选择合适的策略。

**🔧 技术方法**

使用了多目标软演员-评论家（MOSAIC）算法来解决双目标优化问题，并结合上下文赌博机进行在线策略选择。

**📊 数据集**

在D4RL连续控制基准上进行了评估，比较了HiP3与多种在线和离线到在线强化学习基线的性能。

**📈 对比分析**

HiP3在相同和新颖的在线任务中表现出色，尤其在新任务上具有明显优势，能够在更强的分布变化下保持有效性。

**⚠️ 局限性**

限制在于HiP3在在线微调的早期阶段仍然表现出初始性能下降，且在更高维环境和严重模型偏差下的离线帕累托权衡可能与真实环境的权衡不完全匹配。

---

## 438. PACT: Plan-Action Consistency Training by Preserving Correct Plans and Repairing Actions in GUI Agents

**arXiv ID:** 2611.22591 | [PDF](https://arxiv.org/pdf/2611.22591v1)

**作者:** Zhuohan Cai `[一作]` (Tsinghua University), Changhua Meng `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种新的训练框架PACT（计划-行动一致性训练），旨在解决GUI代理在执行过程中出现的计划与行动不一致的问题。

**💡 创新点**

PACT通过结合参考对齐的计划保护和计划对齐的行动修复，解决了信用分配冲突，而不改变原始的标量行动奖励。

**🔧 技术方法**

使用了基于强化学习的训练方法，特别是组相对策略优化（GRPO）和直接偏好优化（DPO）等技术。

**📊 数据集**

使用了AndroidWorld和OSWorld等数据集进行实验，评估任务成功率和计划-行动一致性。

**📈 对比分析**

与仅基于结果的基线方法相比，PACT在AndroidWorld任务成功率上从60.34%提高到68.97%，在OSWorld基准上从40.92%提高到44.17%。

**⚠️ 局限性**

训练时间选择器需要明确的计划，可能会导致有效但隐含的计划或语义等价的替代行动无法被修复。

---

## 439. A polynomial-time decoding algorithm up to half the square-root bound for quadratic residue and generalized quadratic residue codes

**arXiv ID:** 2611.21888 | [PDF](https://arxiv.org/pdf/2611.21888v1)

**作者:** Phlippe Gaborit `[一作]`, Philippe Lebacque `[通讯]` (Universite de Limoges)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种多项式时间解码算法，能够纠正二进制广义二次剩余码（GQR码）中最多⌊(√(q)-1)/2⌋个错误，算法复杂度为O(q^9/2)。

**💡 创新点**

这是第一个适用于整个二进制QR和GQR码的多项式时间解码算法，解决了之前没有已知算法的难题。

**🔧 技术方法**

使用了Helleseth的QR码描述和Kopparty的低度多项式恢复工具，结合了Hasse导数和多重性估计。

**📊 数据集**

使用了二进制广义二次剩余码（GQR码）作为数据集，特别是对于每个奇素数幂q≡±1 (mod 8)。

**📈 对比分析**

与现有的解码方法相比，该算法能够在多项式时间内解码，性能优于之前的解码方法，尤其是在处理GQR码时。

**⚠️ 局限性**

该算法的局限性在于，对于素数长度的QR码，解码超过√(p)/2个错误仍然是一个开放问题。

---

## 440. Does Your Agent Need to Know Its Budget? A Placebo-Controlled Experiment on LLM Coding Agents

**arXiv ID:** 2611.21614 | [PDF](https://arxiv.org/pdf/2611.21614v1)

**作者:** Iftesam Nabi `[一作]` (Louisiana Tech University), Mehrab Hosain `[通讯]` (Louisiana Tech University)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5081033279)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过安置预算执行和预算披露的因素实验，探讨了在自主编码代理中实施令牌预算的因果效应。

**💡 创新点**

创新点在于首次通过随机化、安慰剂对照实验来评估预算执行和预算披露对编码代理的影响，提供了因果证据。

**🔧 技术方法**

使用了随机化的因素实验设计，结合了安慰剂对照和多次运行的实验方法。

**📊 数据集**

使用了两个可执行的程序修复基准数据集，分别是QuixBugs和HumanEvalFix，共进行了5835次实验。

**📈 对比分析**

与无预算句子的基线相比，预算披露并未显著提高任务解决率，反而在某些情况下表现出负面影响。预算执行导致了解决率下降20个百分点，但每个解决任务的令牌消耗减少了约十二倍。

**⚠️ 局限性**

限制在于研究仅在特定的功能级修复任务上进行，未能涵盖更广泛的任务类型，且结果可能不适用于不同的模型或环境。

---

## 441. When Anchors Shift the Target: Advantage-Gated Adaptation for Diffusion Reinforcement Learning

**arXiv ID:** 2611.23038 | [PDF](https://arxiv.org/pdf/2611.23038v1)

**作者:** Pu Cao `[一作]` (Beijing University of Posts and Telecommunications), Lu Yang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为优势门控适应（AGA）的方法，以解决在前向过程强化学习中，移动策略锚点对正反馈学习的影响。

**💡 创新点**

创新点在于形式化了锚点耦合问题，并提出了AGA方法，通过保持正样本的自限制目标回归和线性化负样本的凹目标来解决这一问题。

**🔧 技术方法**

使用了前向过程强化学习（RL）和无分类器引导（CFG）技术。

**📊 数据集**

实验使用了SD3.5-Medium数据集，涉及组合生成、文本渲染和偏好对齐等任务。

**📈 对比分析**

与现有方法（如AWM、DiffusionNFT和RAM）进行比较，AGA在任务适应性和奖励-质量权衡上表现出色，尤其在OCR和PickScore任务中取得了最高分。

**⚠️ 局限性**

限制在于虽然AGA改善了训练稳定性和任务表现，但仍需进一步研究如何在更复杂的生成任务中保持性能。

---

## 442. Do Diffusion Models Learn to Generalize Basic Visual Skills?

**arXiv ID:** 2611.22427 | [PDF](https://arxiv.org/pdf/2611.22427v1)

**作者:** Amish Sethi `[一作]` (Princeton University), Zhuang Liu `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了视觉生成实验室（VGL），用于研究扩散模型如何学习和泛化基本视觉技能。

**💡 创新点**

VGL提供了一种受控的实验框架，能够精确测量模型在特定任务上的泛化能力，填补了现有评估方法的空白。

**🔧 技术方法**

使用了扩散模型，结合了明确的技能值标签和条件机制，如自注意力和适应性归一化。

**📊 数据集**

使用了合成的几何数据集，数据集中的每个技能都是通过明确的规则生成的。

**📈 对比分析**

与现有的基准方法相比，VGL通过任务特定的规则基础指标进行评估，发现模型在训练范围内的泛化能力有限，且组合泛化能力更依赖于技能组合的多样性而非数据集的大小。

**⚠️ 局限性**

VGL依赖于受控的合成数据，主要评估了DiT骨干网络，因此这些发现如何转移到自然图像上仍然是一个开放问题。

---

## 443. DailyBrick: Language-Guided Synthesis of Brick Assembly Instructions

**arXiv ID:** 2611.22132 | [PDF](https://arxiv.org/pdf/2611.22132v1)

**作者:** Hang Zhang `[一作]` (East China Normal University), Xin Tan `[通讯]` (Shanghai Jiaotong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为DailyBrick的多阶段代理系统，该系统能够根据用户请求生成LEGO模型及逐步组装说明。

**💡 创新点**

创新点在于将LEGO模型生成与组装指令的生成结合起来，通过多阶段的代理流程和自我元审查机制来定位错误并进行针对性更新。

**🔧 技术方法**

使用了多阶段规划、部分组装技能和构建软件开发工具包（SDK），结合独立的验证重放机制。

**📊 数据集**

使用了自收集的包含358,775个结构化LEGO模型的数据集，每个模型都有文本描述和砖块计数信息。

**📈 对比分析**

与直接姿态生成和现有的BrickGPT、BrickNet等方法进行了比较，实验结果显示DailyBrick在视觉质量和组装序列有效性上有显著提升。

**⚠️ 局限性**

局限性在于尽管组装有效性得到了保证，但设计质量仍然可能存在问题，例如颜色选择不一致、结构重复过多和意外的不对称性。

---

## 444. ThermLeT: Transformer-based Temperature Prediction for 2.5D Chiplet Architecture

**arXiv ID:** 2611.21954 | [PDF](https://arxiv.org/pdf/2611.21954v1)

**作者:** Varun Darshana Parekh `[一作]` (Pennsylvania State University), Vijaykrishnan Narayanan `[通讯]` (Pennsylvania State University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为ThermLeT的基于Transformer的温度预测模型，用于2.5D芯片架构，能够快速预测稳态温度。

**💡 创新点**

创新点在于使用多模态Transformer模型，结合功率密度图、空间坐标嵌入、材料属性和热传递率，实现了比传统有限元求解器快四个数量级的温度预测。

**🔧 技术方法**

使用了Transformer模型，结合多模态输入进行温度预测。

**📊 数据集**

创建了一个包含6221个芯片布局的数据集，这些布局是从三种代表性架构中扰动而来的。

**📈 对比分析**

与ThermEDGe和HotSpot进行比较，ThermLeT在150个样本的测试集上实现了3.16°C的均方根误差（RMSE）和2.24°C的平均绝对误差（MAE），分别比ThermEDGe提高了35%和比HotSpot提高了71%，推理延迟分别降低了约6倍和11倍。

**⚠️ 局限性**

模型的局限性在于尽管在未见过的布局上表现出良好的泛化能力，但未来的工作需要进一步增强物理约束的集成，以提高模型的准确性和适用性。

---

## 445. Cluster-Guided Decoding with Trust-Region Constraints for Personalized Recommendation

**arXiv ID:** 2611.22163 | [PDF](https://arxiv.org/pdf/2611.22163v1)

**作者:** Gezheng Wen `[一作]` (Amazon), Layne C. Price `[通讯]` (Amazon)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为Cluster-Guided Generative Reconstruction (CG2R)的方法，通过学习语义集群的层次结构，并利用集群级先验来引导个性化推荐的项目排名。

**💡 创新点**

CG2R的创新点在于将集群先验直接添加到项目评分规则中，使得学习到的集群结构直接影响排名，而不仅仅是用于候选项的预选择。

**🔧 技术方法**

使用了集群引导解码、残差量化变分自编码器（RQ-VAE）和因果Transformer等技术。

**📊 数据集**

在三个公开基准数据集上进行了评估，包括MovieLens-20M、Steam和Google Local Reviews（华盛顿子集）。

**📈 对比分析**

与经典、神经网络和基于大语言模型的基线方法相比，CG2R在Recall和nDCG指标上均表现优异，尤其在数据稀疏时性能提升更为明显。

**⚠️ 局限性**

CG2R的局限性在于可能会在历史交互中重现偏见，并且在某些情况下可能会过于集中于流行或语义狭窄的集群。

---

## 446. Low-speed Limit Cycle Walking of Planar X-shaped Bipedal Robot with Special Properties

**arXiv ID:** 2611.22122 | [PDF](https://arxiv.org/pdf/2611.22122v1)

**作者:** Fumihiko Asano `[一作]` (Japan Advanced Institute of Science and Technology), Cong Yan `[通讯]` (Ritsumeikan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种新型的平面半圆足X形双足机器人，并讨论了在平坦地面上生成稳定极限循环步态的方法。

**💡 创新点**

创新点在于机器人模型的设计，腿部连杆的质心与髋关节位置重合，以及利用半圆足的抗重力效应来提高步态的稳定性和效率。

**🔧 技术方法**

使用了数学建模和数值仿真技术来分析机器人的运动方程和控制系统。

**📊 数据集**

未具体提及使用的数据集，但通过数值仿真验证了理论结果。

**📈 对比分析**

通过数值仿真与传统的双足机器人步态进行比较，结果显示所提控制方法能够有效生成稳定的步态，步态描述符如步长和恢复的机械能均趋于稳定值。

**⚠️ 局限性**

限制在于模型假设和控制系统设计的复杂性，未来需要进一步分析步态稳定性和效率，尤其是在不同重力条件下的表现。

---

## 447. An Open-Source da Vinci X/Xi Instrument Module for Laparoscopic Surgery Research

**arXiv ID:** 2611.21771 | [PDF](https://arxiv.org/pdf/2611.21771v1)

**作者:** Douglas McDonough `[一作]` (Johns Hopkins University), Chunwoo Kim `[通讯]` (Korea Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本论文提出了一个开源模块，用于在UR5/UR5e机器人臂上控制da Vinci X/Xi仪器的运动和能量激活，并记录每个动作的时间。

**💡 创新点**

创新点在于结合了五个电机驱动器和电外科接口，允许对复杂的外科仪器进行控制，同时优先考虑外科医生的脚踏板操作。

**🔧 技术方法**

使用了五个DYNAMIXEL XC330-M288-T伺服电机和一个定制的电路板（ESU-TAP v2），实现了软件控制和激活记录。

**📊 数据集**

论文中使用的实验数据集包括对脾脏组织的单极和双极凝固，以及胆囊切除术中的单极切割的外部验证。

**📈 对比分析**

通过与现有的DASH平台进行比较，模块在工具更换时间（平均2.41秒）和能量激活的可靠性方面表现良好，能够记录所有激活和去激活事件。

**⚠️ 局限性**

限制在于该模块需要手动选择仪器配置，且在某些情况下，控制器可能会挂起或继电器可能会焊接，这些情况未被电路板覆盖。

---

## 448. SignAligner: Dual-Granularity Span Generation for Sign Language Alignment

**arXiv ID:** 2611.22409 | [PDF](https://arxiv.org/pdf/2611.22409v1)

**作者:** Jongbhin Woo `[一作]` (KAIST), Joon Son Chung `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于提示的视觉语言跨度生成器，旨在将手语视频与字幕句子或目标词查询映射到文本的开始/结束帧间隔。该模型首先在字幕跨度和单词跨度上进行联合训练，然后通过重叠奖励进行优化。

**💡 创新点**

创新点在于将字幕对齐和单词识别视为相同的文本条件跨度预测任务，通过联合训练引入现有的手语识别数据，从而提高字幕对齐的准确性。

**🔧 技术方法**

使用了基于提示的生成模型，结合了视觉编码器、轻量级投影器和解码器，采用了两阶段的训练方法：第一阶段进行双粒度监督学习，第二阶段进行重叠奖励优化。

**📊 数据集**

使用了英国广播公司-牛津英国手语（BOBSL）语料库，该语料库包含手语视频和与之对齐的字幕，数据集规模大，包含数千小时的手语实例。

**📈 对比分析**

与其他系统相比，该模型在字幕测试集上达到了F1@0.50的最佳成绩，优于没有外部手语视频资源的其他系统。重叠奖励优化的效果超过了相同模型的交叉熵继续训练。

**⚠️ 局限性**

局限性包括：使用英语字幕词作为检索查询而非BSL术语，输出的单词级别结果仅为候选注释，缺乏与聋人用户的共同设计评估，且未测试在没有音频对齐先验的情况下的表现。

---

## 449. Understanding Cross-Modal Inconsistency through Representation Subspaces in Omni Models

**arXiv ID:** 2611.22619 | [PDF](https://arxiv.org/pdf/2611.22619v1)

**作者:** Seonghyeon Cho `[一作]` (Soongsil University), Chanjun Park `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了跨模态不一致性，分析了Omni模型在处理音频、视觉和文本时的表现，提出了一种低维的答案子空间来分离与答案相关的成分与其他表示。

**💡 创新点**

创新点在于通过估计低维答案子空间，揭示了预测跨模态不一致性的信息主要位于答案子空间之外，并且这种信息集中在一个可重复的二维表示中。

**🔧 技术方法**

使用了线性解码、匹配干预等技术来分析模型的内部表示和行为。

**📊 数据集**

使用了来自XModBench的数据集，涵盖了空间和外部任务家族，确保在音频和图像条件下对相同项目进行评估。

**📈 对比分析**

通过与随机子空间的比较，发现扰动答案子空间会显著改变模型的预测，而扰动不一致性表示几乎不改变预测，表明两者在行为上的不同角色。

**⚠️ 局限性**

限制在于不一致性预测坐标是从两个模态条件中估计的，因此无法在两个预测都可用之前标记不一致性。此外，恢复的方向可能部分反映了项目的难度或模糊性。

---

## 450. Teaching What to Look For: Privileged Perception Questions for Agentic On-Policy Distillation in Video Captioning

**arXiv ID:** 2611.22755 | [PDF](https://arxiv.org/pdf/2611.22755v1)

**作者:** Yifan Xu `[一作]` (Nanjing University), Limin Wang `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于感知问题的主动视频描述框架，通过训练引导视频描述代理主动获取视觉证据并生成详细描述。

**💡 创新点**

创新点在于使用训练时的感知问题（PQs）来指导视频描述代理的视觉探索，而在推理时不需要这些问题，从而提高了描述质量和准确性。

**🔧 技术方法**

采用了基于政策蒸馏（on-policy distillation）的方法，结合感知问题进行训练，指导学生模型生成交互轨迹。

**📊 数据集**

使用了5,979个视频和43,477个感知问题的数据集，涵盖了对象、动作和时间发展。

**📈 对比分析**

与未训练的Qwen3.5-9B代理进行比较，提出的方法在多个基准测试中表现出一致的改进，尤其在DREAM-1K、CaReBench等基准上取得了最佳结果。

**⚠️ 局限性**

限制在于训练过程中未提供感知问题的答案，可能导致模型在某些情况下无法充分利用视觉证据，且对答案泄露的敏感性可能影响性能。

---

## 451. Inventa: Decentralized Self-Improving Agents for Autonomous Scientific Discovery

**arXiv ID:** 2611.21847 | [PDF](https://arxiv.org/pdf/2611.21847v1)

**作者:** Prithwish Jana `[一作]` (Georgia Institute of Technology), Sahika Genc `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于自动化科学发现的框架，该框架利用自主研究单元和经过验证的共享知识来生成研究想法。

**💡 创新点**

创新点在于引入了自主研究单元和自适应提案机制，能够在不同的研究阶段进行有效的探索和知识共享。

**🔧 技术方法**

使用了自主研究单元、变异代理和验证者组件的组合技术。

**📊 数据集**

在五个基准测试上进行了评估，包括检索、创意生成、提案写作和科学法则发现。

**📈 对比分析**

与现有的前沿AI代理和进化搜索方法相比，该框架在所有五个基准测试中均表现优异，领先6-40%。在EurekaBench上达到了73.6%的得分，而CORAL为56.2%；在IdeaBench上每篇论文发现了26.0个新想法，相比之下ResearchAgent为18.6，且模型调用减少了82%。

**⚠️ 局限性**

限制在于框架的复杂性和对共享知识的依赖，可能导致在某些情况下的知识传播错误。

---

## 452. SimAI: Mechanistic User Simulation through Psychological Network Models

**arXiv ID:** 2611.21808 | [PDF](https://arxiv.org/pdf/2611.21808v1)

**作者:** Cheng Li `[一作]` (University of Washington), Tim Althoff `[通讯]` (University of Washington)

**通讯引用:** 11288 | [OpenAlex ID](https://openalex.org/A5033639139)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于心理网络模型的机制用户模拟框架SimAI，旨在模拟用户行为背后的心理机制，以便在直接实验成本高或潜在有害的情况下研究人际互动。

**💡 创新点**

创新点在于通过心理网络模型显式建模心理因素之间的相互作用，从而指导用户模拟，克服了现有模拟器忽视心理状态相互影响的局限。

**🔧 技术方法**

使用了图重建目标和心理网络模型，结合观察者模型和行为者模型来生成用户行为。

**📊 数据集**

构建了PSYCHE数据集，包括来自516个心理治疗案例的18,576个个人资料-网络模型对。

**📈 对比分析**

通过与现有患者模拟器（如Roleplay-DOH、Eeyore和Patient-Psi）进行比较，SimAI在模拟和行为多样性方面表现更佳，同时保持强大的个人资料一致性。在与30名心理健康专业人士的互动研究中，SimAI在对话真实性、现实帮助寻求行为和临床培训效用方面获得了最高评分。

**⚠️ 局限性**

局限性在于虽然SimAI能够模拟心理机制，但其一致性并不等同于真实人类行为的机制，未来需要进一步验证其模型的有效性。

---

## 453. Multicore Quantum Architecture for Scalable and Energy-Aware Green IoT Computing

**arXiv ID:** 2611.21663 | [PDF](https://arxiv.org/pdf/2611.21663v1)

**作者:** Seham Al Abdul Wahid `[一作]` (Toronto Metropolitan University), Arghavan Asad `[通讯]` (Algoma University)

**通讯引用:** 476 | [OpenAlex ID](https://openalex.org/A5041995126)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

本文提出了一种用于绿色物联网的可扩展和节能的多核量子架构，并通过Qcomm模拟器对同质多核经典和量子架构进行了通信感知的比较。

**💡 创新点**

创新点在于提出了一种基于网格的同质量子-经典多核架构，以实现分布式量子工作负载的可扩展和高效执行，并分析了流量模式和光-物质端口配置对性能的影响。

**🔧 技术方法**

使用了Qcomm模拟器来评估多核量子和经典架构在不同流量条件下的性能，并实现了量子近似优化算法（QAOA）、变分量子特征求解器（VQE）和量子神经网络（QNN）等算法。

**📊 数据集**

使用了4×4的网格架构，模拟了16个处理核心的性能，评估了在常规、经典瓶颈和量子瓶颈条件下的表现。

**📈 对比分析**

与多核经典处理器相比，多核量子处理器在QAOA、VQE和QNN算法上分别实现了10倍、9倍和7倍的能效提升，表明量子架构在执行时间和能效方面显著优于经典架构。

**⚠️ 局限性**

限制在于当前量子硬件仍处于噪声中间规模量子（NISQ）时代，受限于可用量子比特数量和高错误率，尚需进一步验证与实际物联网工作负载的兼容性。

---

## 454. Pixel is All You Need

**arXiv ID:** 2611.22820 | [PDF](https://arxiv.org/pdf/2611.22820v1)

**作者:** Jie Chu `[一作]` (Shanghai Jiao Tong University), Muning Wen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出了一种新的像素空间生成模型，旨在统一文本、图像、音频等多种模态的推理能力，模型能够在没有文本标记的情况下直接在像素画布上进行推理和生成。

**💡 创新点**

创新点在于首次系统性地研究了纯像素空间生成模型的推理能力，展示了这些模型在没有文本通道的情况下，如何学习在画布上直接读取、推理和书写。

**🔧 技术方法**

使用了流匹配变换器（flow-matching transformers），直接在原始像素上操作，没有文本标记、标记器或文本编码器。

**📊 数据集**

使用了多种任务的数据集，包括算术、多位数乘法、文字问题、事实问答以及交叉模态任务，所有任务均在1024x1024的画布上进行。

**📈 对比分析**

与现有的图像生成模型相比，当前模型在直接回答文本问题时表现不佳，几乎无法正确回答，而新模型在纯视觉设置下表现良好，展示了可预测的扩展性、逐步推理的优势和强化学习的提升。

**⚠️ 局限性**

限制在于当前模型的训练仍然依赖于大量的计算资源，且在某些复杂的交叉模态任务上可能仍需进一步优化。

---

## 455. SIMQ: A Single Graph-based Index for Multi-Vector Query

**arXiv ID:** 2611.22079 | [PDF](https://arxiv.org/pdf/2611.22079v1)

**作者:** Yifan Lu `[一作]` (Tongji University), Heng Tao Shen `[通讯]` (Tongji University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的多向量检索框架SIMQ，通过数据感知的采样聚类和每个令牌的图导航来提高检索效率，避免了传统方法中的计算瓶颈。

**💡 创新点**

创新点在于SIMQ框架结合了单向量图索引和多向量检索，采用数据感知的聚类方法，避免了GPU加速和交互矩阵的需求，同时引入了SIMQ+以增强鲁棒性。

**🔧 技术方法**

使用了数据感知的采样聚类技术和图导航方法，构建了一个单一的图索引来进行近似MaxSim评分。

**📊 数据集**

在五个基准数据集上进行了实验，包括Quora、SciDocs、NQ、HotpotQA和LoTTE-pooled，验证了SIMQ和SIMQ+的有效性。

**📈 对比分析**

与多个基线方法（如PLAID、MUVERA、IGP和GEM）进行了比较，结果显示SIMQ在召回率、查询吞吐量和构建时间上表现优越，尤其是在CPU上运行时。

**⚠️ 局限性**

限制在于SIMQ的性能可能受到数据分布的影响，且在某些情况下可能无法完全替代GPU加速的传统方法。

---

## 456. Harnesses That Improve Themselves: A Unifying Framework for Self-Improving Agent Harnesses

**arXiv ID:** 2611.22299 | [PDF](https://arxiv.org/pdf/2611.22299v1)

**作者:** Deep Shah `[一作]` (Google), Nehal Kathrotia `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种自我改进的代理工具框架，该框架通过四个可编辑的运行时组件和一个四阶段循环来诊断失败、提出代码补丁、验证候选项并部署更新。

**💡 创新点**

创新点在于将运行时工具视为可变软件，允许系统根据自身执行轨迹进行自我优化，从而提高了模型的可靠性和效率。

**🔧 技术方法**

使用了自我改进的代理工具框架，结合了上下文和状态管理、代理控制、动作接口以及验证和恢复等技术。

**📊 数据集**

在SWE-bench Lite数据集上进行了实验，验证了该框架的有效性。

**📈 对比分析**

与传统的提示反射和全工具重写方法相比，门控模块演化在多个模型和代码库中表现出更好的性能，并且在令牌成本上显著降低。

**⚠️ 局限性**

限制在于自我改进的工具可能会过拟合特定任务，导致在不同模型或任务上的泛化能力下降。

---

## 457. Eight in Ten LLMs Leak System-Prompt Secrets: Nevan, an Agentic Capture-the-Flag Study

**arXiv ID:** 2611.21956 | [PDF](https://arxiv.org/pdf/2611.21956v1)

**作者:** Ana Clara Zoppi Serpa `[一作]` (Institute of Computing University of Campinas), Hélio Pedrini `[通讯]` (Institute of Computing University of Campinas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对OpenRouter目录中的363个文本输入/输出模型进行了指令层次执行的审计，研究哪些模型泄露了嵌入系统提示中的受保护秘密，以及使用了哪些技术。

**💡 创新点**

创新点在于通过捕获旗帜游戏（CTF）对指令层次执行进行全面审计，涵盖了更广泛的模型生态系统，而不仅仅是前沿模型。

**🔧 技术方法**

使用了人工智能代理（OpenCode，基于DeepSeek V4.1 Flash）进行攻击，记录每次交互并分析捕获结果。

**📊 数据集**

使用了OpenRouter目录中的363个模型，最终捕获了300个模型的秘密。

**📈 对比分析**

与之前的研究相比，本研究涵盖了更广泛的模型，296个模型泄露了秘密，170个捕获涉及非英语语言，尤其是法语。性能显示大多数模型未能有效保护系统提示中的秘密。

**⚠️ 局限性**

限制在于审计仅针对冻结的模型目录，未能评估模型的更新或替换情况，且未考虑语义敏感数据的泄露风险。

---

## 458. Cancellation-Free Constrained SquareCB for Safe Controller Selection

**arXiv ID:** 2611.21880 | [PDF](https://arxiv.org/pdf/2611.21880v1)

**作者:** Kartik Pandit `[一作]` (New Jersey Institute of Technology), Arnob Ghosh `[通讯]` (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了受约束的上下文赌博机，提出了一种新的算法SquareCB，该算法通过调整成本权重来最小化损失，同时保持期望成本在预算之内。

**💡 创新点**

创新点在于使用了修正的成本权重，当估计成本在预算内时权重为零，否则权重足够大以将成本拉回预算，避免了以往方法中轮次间的松弛抵消。

**🔧 技术方法**

使用了SquareCB算法，并结合了在线回归oracle进行成本权重的调整。

**📊 数据集**

在安全控制的四旋翼和BOPTEST建筑的控制选择实验中进行了验证。

**📈 对比分析**

与原始对偶和虚拟队列基线方法相比，算法在预算超支方面低6到9倍，同时具有更低或相当的遗憾。

**⚠️ 局限性**

限制在于算法需要已知的最优乘数的界限，而这一点通常通过安全的后备控制器提供。未来的工作方向包括去除这一要求并扩展到多个约束。

---

## 459. Subsystem Stabilizer $(r,δ)$-Locally Recoverable Codes: Characterization, Bounds and Constructions

**arXiv ID:** 2611.22221 | [PDF](https://arxiv.org/pdf/2611.22221v1)

**作者:** Yaozong Zhang `[一作]` (Nankai University), Jian Gao `[通讯]` (Shandong University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了子系统稳定器(r,δ)-局部可恢复编码((r,δ)-SSLRCs)，并探讨了在给定修复集内是否可以纠正擦除的问题。

**💡 创新点**

创新点在于对局部擦除恢复的代数特征进行了详细的描述，并推导出多种参数界限，同时构造了九类最优的_q-线性纯(r,δ)-SSLRCs。

**🔧 技术方法**

使用了代数方法来表征局部擦除恢复，并应用了CSS和Hermitian hull构造方法。

**📊 数据集**

使用了多种编码，包括最大距离可分离(MDS)稳定器编码、Reed-Solomon编码和具有MDS Hermitian hull的线性编码。

**📈 对比分析**

通过与现有的局部可恢复编码进行比较，本文展示了所提出的编码在多个界限下的最优性，性能表现优越。

**⚠️ 局限性**

限制在于未假设纯度的情况下推导的界限可能不适用于所有情况，且在某些情况下可能无法达到最优性能。

---

## 460. ASTRA: ASV Simulator for Transferable Realistic Autonomy

**arXiv ID:** 2611.21789 | [PDF](https://arxiv.org/pdf/2611.21789v1)

**作者:** Tsimafei Lazouski `[一作]` (University of Michigan), Maani Ghaffari `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文提出了ASTRA，一个基于NVIDIA Isaac Sim的海洋自主表面车辆（ASV）仿真框架，结合了海洋车辆动力学、逼真的感知和ROS 2自主接口。

**💡 创新点**

创新点在于ASTRA是一个开源的仿真框架，支持完整的ASV自主开发，能够实现高保真度的视觉渲染和ROS 2集成，同时提供了波浪-船体相互作用的模拟。

**🔧 技术方法**

使用了NVIDIA Isaac Sim作为仿真平台，结合Fossen风格的6自由度动力学模型、几何基础的浮力计算和可编程的波浪干扰。

**📊 数据集**

使用了物理BlueBoat测试平台的现有ROS 2定位和感知模块，并进行了针对性的零推进实验以验证波浪-船体耦合。

**📈 对比分析**

与其他海洋仿真器相比，ASTRA在支持完整的6自由度动力学、波浪-船体耦合和ROS 2自主开发方面具有优势，性能表现良好，能够以约35 FPS的速度运行完整的传感器套件和自主堆栈。

**⚠️ 局限性**

限制在于ASTRA目前仅提供了基于现场数据的平面（冲击、偏航）阻尼模型，尚未针对物理的升沉、滚动和俯仰响应进行校准，波浪模型也仅验证了可重复的物理一致性干扰生成。

---

## 461. CascadeJEPA: Enhancing 3D Geometric Representations via Spatially Coupled Cascaded Prediction

**arXiv ID:** 2611.22686 | [PDF](https://arxiv.org/pdf/2611.22686v1)

**作者:** Kailai Zhou `[一作]` (Nanyang Technological University), Bihan Wen `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的空间耦合级联预测框架CascadeJEPA，以解决I-JEPA在语义学习与几何表示之间的差异问题。

**💡 创新点**

CascadeJEPA通过引入级联预测和交叉注意力机制，增强了几何表示的学习，同时保持了I-JEPA的架构简洁性。

**🔧 技术方法**

使用了空间耦合的级联预测框架和交叉注意力机制。

**📊 数据集**

使用了ImageNet-1K数据集进行预训练，并在多个3D几何任务上进行了评估。

**📈 对比分析**

与I-JEPA相比，CascadeJEPA在多个3D几何任务上表现出显著的性能提升，同时保持了强大的语义识别能力。

**⚠️ 局限性**

CascadeJEPA由于其两阶段的预测框架，可能会增加预训练的开销，并且其性能可能对超参数配置敏感。

---

## 462. Fine-Grained Japanese Clinical Named Entity Recognition: A Controlled Comparison of BERT and Open-Weight Large Language Models

**arXiv ID:** 2611.22219 | [PDF](https://arxiv.org/pdf/2611.22219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 463. Refin3R: Geometry-Guided Refinement for Feed-Forward 3D Reconstruction

**arXiv ID:** 2611.23051 | [PDF](https://arxiv.org/pdf/2611.23051v1)

**作者:** Jongmin Lee `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的方法，通过将前馈3D重建与运动结构(SfM)点结合，改进了多视图图像的点云重建，生成密集且准确的点图。

**💡 创新点**

创新点在于采用双分支架构，2D分支在图像网格上扩展SfM几何信息，而3D分支则利用所有视图的SfM点进行多视图结构推理，从而实现了更高的点图精度。

**🔧 技术方法**

使用了双分支网络架构，其中包括2D分支和3D分支，结合了几何感知的图像特征和SfM点。

**📊 数据集**

使用了多个数据集进行实验，包括ScanNet++、ETH3D、Tanks&Temples，以及一些超出训练域的人类和外科场景。

**📈 对比分析**

与前馈和几何引导方法相比，所提出的方法在点图精度上表现出显著的优势，并且在速度上也优于最强的几何引导基线。

**⚠️ 局限性**

限制在于方法依赖于SfM点的质量和覆盖范围，可能在某些特定场景下表现不佳，尤其是在纹理稀少或视角重叠较少的区域。

---

## 464. EviDARL: Evidence-Centric Dual-Agent Reasoning via Alternating Reinforcement Learning for Multi-Table Question Answering

**arXiv ID:** 2611.22310 | [PDF](https://arxiv.org/pdf/2611.22310v1)

**作者:** Dunhan Li `[一作]` (Zhejiang University), Fei Wu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为EviDA的证据中心双代理框架，旨在解决多表问答中的多跳推理问题。该框架由证据构建代理和推理回答代理组成，前者生成并执行表操作程序以构建问题特定的证据，后者则基于这些证据进行推理以得出最终答案。

**💡 创新点**

创新点在于通过双代理的方式将多表问答分解为可执行的证据构建和基于证据的推理，从而简化了推理过程并提供了可验证的中间追踪。

**🔧 技术方法**

使用了证据构建代理和推理回答代理的双代理框架，并结合了细粒度奖励和交替强化学习来优化两个代理之间的协作。

**📊 数据集**

在MMQA和TQA-Bench两个多表问答基准上进行了实验，MMQA包含2123个训练实例和323个测试实例，TQA-Bench则包含不同输入长度的多项选择问题。

**📈 对比分析**

与多种竞争基线进行比较，EviDARL在受限模型和训练预算下显著超越了开放模型和多表推理基线，EviDARL在TQA-Bench上达到了88.40%的准确率，在MMQA上达到了67.80%的精确匹配分数，超越了如GPT-5.5和Claude-Sonnet-5等先进的LLM。

**⚠️ 局限性**

限制在于评估仅覆盖两个基准和两个8B的基础模型，假设输入为机器可读的表格，未直接处理表格图像、图表或其他视觉证据。此外，推理回答代理在没有调用计算工具的情况下进行推理，可能导致算术和统计答案的准确性依赖于模型生成的推理。

---

## 465. ARG-Avatar: Animation-Rendering Guidance for Efficient Feedforward Avatar Reconstruction

**arXiv ID:** 2611.22207 | [PDF](https://arxiv.org/pdf/2611.22207v1)

**作者:** Jianjin Xu `[一作]` (Carnegie Mellon University), Fernando De la Torre `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为ARG-Avatar的高效前馈框架，用于从图像重建可动画的3D人类头像，旨在使高质量头像创建变得更加可及。

**💡 创新点**

ARG-Avatar通过引入FACRoPE和ITR两个关键组件，显著减少了可训练参数数量，同时保持了重建质量，挑战了以规模为先的训练理念。

**🔧 技术方法**

使用了前馈神经网络架构，结合了几何引导的注意力机制（FACRoPE）和中间令牌渲染（ITR）技术。

**📊 数据集**

在多个公开数据集上进行训练，包括DNARendering、2K2K、4D-Dress等，共包含6800个主体和64000个序列。

**📈 对比分析**

与LHM++等先前方法相比，ARG-Avatar在使用两个或更多输入图像时，在PSNR上表现更优，且训练参数减少约9倍，训练序列减少4.8倍，计算需求减少14倍，推理速度提高6倍。

**⚠️ 局限性**

ARG-Avatar的局限性在于其在单视图输入时的性能较弱，未经过训练以填补不可见区域，因此在实际重建场景中可能需要多个输入图像。

---

## 466. MillBill: A Machining Agent with Geometric Feedback

**arXiv ID:** 2611.22565 | [PDF](https://arxiv.org/pdf/2611.22565v1)

**作者:** Gennadii Savrasov `[一作]` (Lomonosov Moscow State University), Dmitrii Zhemchuzhnikov `[通讯]` (Lomonosov Moscow State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了MillBill，一个将边界表示（B-rep）部件转化为G代码的代理，通过将加工决策与几何执行分开来实现。

**💡 创新点**

创新点在于将加工决策与几何执行分开，使得语言模型只负责操作、工具和切削参数的选择，而几何工具负责生成刀具路径和验证结果。

**🔧 技术方法**

使用了图神经网络（GNN）进行加工特征识别，以及几何工具进行刀具路径生成和材料去除模拟。

**📊 数据集**

使用了从GrabCAD收集的4294个部件的数据集，评估了100个随机选择的部件。

**📈 对比分析**

与基于前沿模型的代码生成代理（如GPT-6 Astra）进行比较，MillBill在目标表面中97.4%在0.1mm的公差内，而GPT-6 Astra为94.3%。没有模型的初始计划也达到了97.0%。

**⚠️ 局限性**

限制在于MillBill仅适用于索引三轴加工，未考虑夹具和机器行程，几何模拟未考虑刀具变形、振动或磨损。

---

## 467. On the Kissing Number of Craig's Lattice and Spherical List Decoding

**arXiv ID:** 2611.22441 | [PDF](https://arxiv.org/pdf/2611.22441v1)

**作者:** Léo Ducas `[一作]` (Centrum Wiskunde en Informatica), Ailsa Robertson `[通讯]` (Universiteit van Amsterdam)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了Craig晶格的亲吻数及其在球面编码中的应用，提出了一种多项式时间的列表解码器，适用于从Craig晶格获得的球面编码，尤其是在角度小于45度的情况下。

**💡 创新点**

创新点在于提出了一种新的列表解码算法，能够在较大的角度范围内有效解码，同时证明了Bacher的下界在特定条件下是渐近紧的。

**🔧 技术方法**

使用了多项式时间的列表解码技术，结合了格理论和球面编码的相关算法。

**📊 数据集**

使用了Craig晶格A_p-1^k的亲吻数数据，计算了维度高达521的亲吻数，并进行了实验验证。

**📈 对比分析**

与现有方法相比，本文的方法在解码性能上表现优越，尤其是在角度接近45度时，能够有效处理更多的向量，且在计算亲吻数时提供了更精确的结果。

**⚠️ 局限性**

限制在于对于较大的k值，可能会导致亲吻数的下界变得不准确，且在某些情况下，算法的复杂度可能会影响其实际应用。

---

## 468. Measuring Human-AI Alignment as an Early Indicator for Effective Collaboration in Hybrid SE Teams

**arXiv ID:** 2611.21633 | [PDF](https://arxiv.org/pdf/2611.21633v1)

**作者:** Andreas Metzger `[一作]` (University of Duisburg-Essen), Andreas Metzger `[通讯]` (University of Duisburg-Essen)

**通讯引用:** 4081 | [OpenAlex ID](https://openalex.org/A5082025709)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

论文探讨了如何在混合软件工程团队中系统地测量人机对齐，以促进有效的协作。

**💡 创新点**

创新点在于提出了一套特定于软件工程的对齐度量指标、验证方法和静态测量工具，以评估AI代理与人类期望之间的对齐程度。

**🔧 技术方法**

使用了软件架构、组织心理学和人机交互等领域的原则和度量标准，结合经验评估方法。

**📊 数据集**

通过快速文献回顾，发现现有文献中对软件工程特定方面的对齐度量考虑较少，缺乏操作性度量定义。

**📈 对比分析**

与现有的协作度量方法相比，提出的对齐度量可以静态测量，便于早期质量保证，且通过业务流程监控验证其有效性。

**⚠️ 局限性**

限制在于如何将HCI/IMO风格的度量转化为可检查的约束，以及如何确保度量的可靠性和可重复性。

---

## 469. Exploiting Aggregate Feedback for Decentralized Equilibrium Learning in Markov Games: A Smart Grid Application

**arXiv ID:** 2611.22762 | [PDF](https://arxiv.org/pdf/2611.22762v1)

**作者:** Siying Huang `[一作]` (University of Chinese Academy of Sciences), Qingshan Jia `[通讯]` (Tsinghua University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在具有聚合奖励结构的马尔可夫博弈中进行去中心化均衡学习，提出了一种聚合预测的阶段性V学习算法，旨在利用聚合反馈信息来提高学习效率。

**💡 创新点**

创新点在于开发了一种聚合预测的阶段性V学习算法，结合了混合反馈的Exp3-IX，能够在不需要对手个体信息的情况下，利用聚合反馈来评估所有候选动作的即时奖励和预测损失。

**🔧 技术方法**

使用了聚合预测的阶段性V学习算法和混合反馈的Exp3-IX，结合了两种时间尺度的设计来优化学习过程。

**📊 数据集**

应用于一个包含10个生产者的微电网，模拟了可再生能源的生成和电力消费决策，使用的状态空间和奖励函数基于实际的电力市场模型。

**📈 对比分析**

与三种学习基线（Mao风格的阶段性V学习、其自适应阶段变体和仅使用选择动作的带宽反馈方法）进行比较，结果显示所提算法在最终偏差间隙和学习效率上均优于这些基线。

**⚠️ 局限性**

限制在于算法的样本复杂度仍然依赖于聚合预测误差，且未能在所有情况下保证均衡学习的收敛速度提升。

---

## 470. Lost in the Funnel: Selection-Induced Miscalibration in Ranking and Decision Systems

**arXiv ID:** 2611.22595 | [PDF](https://arxiv.org/pdf/2611.22595v1)

**作者:** Aditya Somani `[一作]` (Microsoft Ads), Amit Singh `[通讯]` (Microsoft Ads)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了现代排名和决策系统中评分的校准问题，提出了一个边界诊断框架来分析评分在决策时的表现，并提出了基于选择政策的后期校准方法Funnel-Aware Recalibration (FAR)。

**💡 创新点**

创新点在于提出了FAR方法，通过在决策边界进行选择政策感知的后期校准，显著降低了决策时的预期校准误差，并提高了拍卖效率。

**🔧 技术方法**

使用了边界诊断框架和后期校准技术，结合了行业案例研究和A/B测试来验证方法的有效性。

**📊 数据集**

适应了两个公共数据集以模拟行业设置，并提供了额外的机制证据，计划发布这些适应的数据集及其生成管道以支持进一步研究。

**📈 对比分析**

通过与传统方法的比较，FAR在实时A/B测试中将决策时的预期校准误差降低了60-80%，并改善了拍卖效率，显示出显著的性能提升。

**⚠️ 局限性**

限制在于生产环境中的数据和日志是专有的，无法直接观察校准与决策之间的选择，因此需要通过公共数据进行模拟和验证。

---

## 471. M$^3$-Test: Coverage Gaps in Joint-Perturbation Testing of Visual Question Answering

**arXiv ID:** 2611.22329 | [PDF](https://arxiv.org/pdf/2611.22329v1)

**作者:** Enhui Wan `[一作]` (Beijing University of Posts and Telecommunications), Xiaowan Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为M^3-Test的覆盖审计方法，用于评估视觉问答模型在联合鲁棒性测试中的覆盖缺口。

**💡 创新点**

创新点在于通过对每个干净输入与其单模态变体及完整的联合库进行对齐，提供了请求级别的覆盖审计，揭示了通过单模态测试但在联合条件下仍然失败的请求。

**🔧 技术方法**

使用了多模态输入、多个扰动家族和多个强度水平的覆盖审计技术。

**📊 数据集**

使用了来自四个视觉-语言模型的260万条响应数据。

**📈 对比分析**

与传统的最大强度测试相比，M^3-Test能够发现更多的受影响请求，且在25个额外查询中发现了约80%的受影响问题，而最大优先策略仅发现了65-68%。

**⚠️ 局限性**

局限性在于所报告的比例是相对于评估库定义的，且未考虑场景文本数据集的鲁棒性。

---

## 472. HandRSI: Recursive Self-Improvement for Hand Pose Estimation

**arXiv ID:** 2611.22164 | [PDF](https://arxiv.org/pdf/2611.22164v1)

**作者:** Zhekai Wang `[一作]` (South China University of Technology), Yuhua Jiang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为HandRSI的递归自我改进框架，用于3D手势估计系统，通过视频估计器和可执行推理工具的组合来提高手势识别的准确性。

**💡 创新点**

创新点在于使用编码代理在每一轮中改进推理工具，并通过选择性蒸馏来优化估计器，同时确保在所有基准指标上没有显著回归。

**🔧 技术方法**

采用了递归自我改进框架，结合了编码代理、推理工具的演化和信息源的蒸馏技术。

**📊 数据集**

使用了HOT3D和ARCTIC数据集进行评估，数据集包含了手势的3D标注和视频序列。

**📈 对比分析**

与九种已发布的方法进行了比较，HandRSI在所有十个指标上均表现最佳，尤其在HOT3D和ARCTIC的MPJPE-p和抖动方面显著优于其他方法。

**⚠️ 局限性**

限制在于模型的更新可能会改变哪些推理模块仍然有用，因此需要在每次更新后重新验证推理工具的有效性。

---

## 473. Regularized Score Centering

**arXiv ID:** 2611.21791 | [PDF](https://arxiv.org/pdf/2611.21791v1)

**作者:** Hao Bai `[一作]` (University of Illinois at Urbana-Champaign), Tong Zhang `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的正则化评分中心方法（RSC），旨在稳定离线策略学习，减少由于策略不匹配导致的学习不稳定性。

**💡 创新点**

创新点在于引入了反Kullback-Leibler（KL）惩罚，以鼓励学习者的下一个令牌分布保持接近行为策略的下一个令牌分布，从而限制策略偏差。

**🔧 技术方法**

使用了正则化评分中心（RSC）和截断重要性采样（TIS）等技术。

**📊 数据集**

在数学推理和代理任务上进行了评估，包括在线购物、工具使用、游戏和完全异步软件工程等任务。

**📈 对比分析**

与其他方法（如IS-PPOClip）相比，RSC与TIS在所有六个代理任务中表现最佳，AppWorld的平均评估分数提高了14.3%，在BabyAI上达到了最强基线的9.9倍得分。

**⚠️ 局限性**

限制在于当学习者远离生成数据的行为策略时，RSC仍可能失败，未来的研究方向是探讨策略陈旧对RSC的影响。

---

## 474. SkiNet-Sim: Simulating Petri Nets with Data for the Verification and Validation of Controllers for Autonomous Systems

**arXiv ID:** 2611.22397 | [PDF](https://arxiv.org/pdf/2611.22397v1)

**作者:** Baptiste Pelletier `[一作]` (ONERA ISAE-SUPAERO), Karen Godary `[通讯]` (LIRMM CNRS)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了增强型Petri网（APNs），旨在将任意数据模拟与经典Petri网语义结合，以提高自主机器人系统的验证和验证能力。

**💡 创新点**

创新点在于提出了一种新的Petri网形式，允许在执行过程中引入任意数据、守卫和效果，从而增强了Petri网的表达能力和实用性。

**🔧 技术方法**

使用了增强型Petri网（APNs）形式，结合了数据感知的状态空间探索和模型检查技术。

**📊 数据集**

使用了自主机器人执行任务的工业检查案例作为数据集，展示了APNs在实际应用中的有效性。

**📈 对比分析**

与经典Petri网相比，APNs在验证和验证任务的成功率上表现更好，能够处理更复杂的状态空间，并提供更精确的故障诊断。

**⚠️ 局限性**

限制在于当前实现仍然使用Python，可能导致状态空间构建速度较慢，未来希望转向编译语言以提高性能。

---

## 475. Beyond 4: Improved Competitive Traveling Repairperson

**arXiv ID:** 2611.21943 | [PDF](https://arxiv.org/pdf/2611.21943v1)

**作者:** Bibek Maharjan `[一作]` (Kent State University), Gokarna Sharma `[通讯]` (Kent State University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种针对任意度量空间的在线旅行修理工问题（TRP）的确定性算法，该算法在k≥1的修理工的情况下具有3.877的竞争比率，并在零残余实例的特殊情况下将比率提高到3+ϵ。

**💡 创新点**

创新点在于引入了可中断的限制路线框架，结合了两种适应性中断决策（紧急和计划中断），并扩展了结果到一些相关的在线拨打接送问题（DARP）。

**🔧 技术方法**

使用了可中断的限制路线框架，结合了阶段性优化的限制路线oracle，并引入了紧急和计划中断机制。

**📊 数据集**

使用了任意度量空间中的在线请求数据集，具体请求由释放时间、位置和权重组成。

**📈 对比分析**

与现有的4-竞争算法相比，提出的算法在任意实例上达到了3.877的竞争比率，并在零残余实例上达到了3+ϵ的竞争比率，显示出显著的性能提升。

**⚠️ 局限性**

限制在于算法的时间复杂度较高，计算每个阶段的最优限制路线涉及到枚举子集和服务顺序，可能导致指数级的计算复杂度。

---

## 476. Margin Dynamics for Large Language Model Alignment

**arXiv ID:** 2611.21766 | [PDF](https://arxiv.org/pdf/2611.21766v1)

**作者:** Xingzi Xu `[一作]` (Amazon), Karim Bouyarmane `[通讯]` (Amazon)

**通讯引用:** 1261 | [OpenAlex ID](https://openalex.org/A5081311040)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提供了一种理论基础，分析了大语言模型（LLM）对齐方法的边际动态，揭示了不同方法在概率单纯形上的质量重分配机制。

**💡 创新点**

创新点在于提出了一种边际动态框架，能够为主要的成对对齐方法提供封闭形式的常微分方程（ODE），并通过理论预测验证了这些方法的性能。

**🔧 技术方法**

使用了边际动态分析和常微分方程（ODE）技术，结合了标签平滑和不同的加权函数来设计新的损失函数。

**📊 数据集**

在五个模型家族和四个基准测试上进行了验证，具体数据集包括MATH、GSM8K和AIME 2024等。

**📈 对比分析**

通过与现有方法（如SFT、DPO、IPO等）的比较，理论预测的配置（TPO-S，ε=0.05）在所有三个模型上达到了最高的平均IFEval分数，验证了有限吸引子的可用性。

**⚠️ 局限性**

限制在于所选的两完成单纯形模型可能无法完全捕捉复杂的训练动态，且边际与下游质量的相关性较弱，此外，TPO-S相较于SimPO的提升幅度较小。

---

## 477. Code World Model 2.0: Towards Infinite Controllable Video Generation

**arXiv ID:** 2611.22925 | [PDF](https://arxiv.org/pdf/2611.22925v1)

**作者:** Ruoyu Wang `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种代理条件的视频渲染框架，结合了双向视频创建和四步因果流媒体，支持多粒度接口。

**💡 创新点**

创新点在于引入了渐进强制（Progressive Forcing）方法，解决了因果适应和四步生成的分离问题，同时保持结构保真度和参考身份。

**🔧 技术方法**

使用了双向创建模型和因果四步流媒体模型，结合了指导无关训练（Guidance-Free Training）和分布匹配蒸馏（Distribution Matching Distillation）技术。

**📊 数据集**

训练使用了一个对齐的约200小时的多粒度视频代理数据集，包含自收集的数据和公共数据集。

**📈 对比分析**

通过与最近的流媒体基线进行比较，实验表明该方法在视觉质量、结构遵循和参考保留方面表现出一致的提升，能够稳定地进行30分钟的赛车回放。

**⚠️ 局限性**

限制在于当前训练使用固定的代理粒度，未来工作可以探索动态中流级别的细节转换和与交互状态引擎的闭环耦合。

---

## 478. ConserveEdit: Source-Anchored Transactions for Drift-Bounded Iterative Image Editing

**arXiv ID:** 2611.22724 | [PDF](https://arxiv.org/pdf/2611.22724v1)

**作者:** Shuyang Jiang `[一作]` (University of California), Zhenyu Wu `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种编辑器无关的控制器，该控制器将每个细化步骤视为针对源图像的合同的事务，旨在防止在迭代过程中未授权区域的意外更改累积。

**💡 创新点**

创新点在于引入了持久编辑合同和事务性细化循环，确保每个候选者在通过独立的进展、漂移和质量测试后才能被提交，从而保持未授权区域的像素不变。

**🔧 技术方法**

使用了源锚定掩码和事务性编辑控制器，结合了多种图像编辑技术，如Qwen-Image-Edit和MagicBrush。

**📊 数据集**

使用了MagicBrush数据集，该数据集包含1,053个测试回合，每个回合都有源图像、指令和人类绘制的编辑掩码。

**📈 对比分析**

与直接编辑、简单循环和无掩码门控等方法进行了比较，结果表明，使用掩码的提交在所有GT指标上均优于直接编辑，且在保护面孔和文本方面表现出更低的漂移和更高的成功率。

**⚠️ 局限性**

限制在于自动掩码的质量，导致在某些情况下的高弃权率，且在保护实体附近的编辑效果不佳，表明需要进一步改进掩码生成和评估机制。

---

## 479. Can a World Model Predict a Deepfake Video? A Consistency Margin Prior for Deepfake Detection

**arXiv ID:** 2611.21846 | [PDF](https://arxiv.org/pdf/2611.21846v1)

**作者:** Pradip Gupta `[一作]` (Oracle AI), Dan Roth `[通讯]` (Oracle AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文研究了一种基于预测视频模型的深伪检测方法，该方法旨在提高跨数据集的泛化能力。通过使用V-JEPA 2.1模型，作者提出了一种新的检测信号，即预测残差，并利用该信号进行深伪视频的分类。

**💡 创新点**

创新点在于引入了基于预测残差的深伪检测方法，提出了Latent Consistency Head (LCH)，并通过Sign-Flip Identification Protocol (SFIP)验证了一致性边际的方向对跨数据集泛化的影响。

**🔧 技术方法**

使用了V-JEPA 2.1模型进行自监督视频预测，并结合了低秩适配（rank-16 LoRA）和轻量级的LCH进行特征提取和分类。

**📊 数据集**

使用了FaceForensics++（FF++）数据集进行训练，并在Celeb-DFv2、DFDC、WildDeepfake等多个未见数据集上进行了评估。

**📈 对比分析**

与标准和反向边际目标进行比较，反向边际设置在跨数据集评估中表现出更大的泛化差距，平均提高了1.8%。整体模型在多个基准测试中表现良好，尤其是在Celeb-DFv2上达到了90.5的AUC。

**⚠️ 局限性**

限制在于该系统仅针对视觉信息进行处理，未考虑音频-视觉一致性、对抗鲁棒性或人口公平性。此外，该方法依赖于预测视频骨干网络，限制了与仅编码器基础模型的直接比较。

---

## 480. SpecSearch: Towards Hardware-Aware Efficient Speculative Decoding

**arXiv ID:** 2611.22296 | [PDF](https://arxiv.org/pdf/2611.22296v1)

**作者:** Selin Yildirim `[一作]` (University of Illinois), Deming Chen `[通讯]` (University of Illinois)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种硬件感知的高效推测解码方法，旨在优化解码策略、草稿长度和精度，以提高大语言模型的推理效率。

**💡 创新点**

创新点在于结合了分析成本模型和学习型策略选择器，能够在不同工作负载和GPU上自动识别最佳解码配置。

**🔧 技术方法**

使用了分析成本模型和基于Jev的学习型策略选择器，进行硬件特定的校准和动态草稿长度映射。

**📊 数据集**

使用了多个GPU（如NVIDIA RTX 5090、H100、L40和AMD MI300X）进行实验，数据集包括GSM8K、MATH-500、MBPP、MT-Bench和HumanEval等。

**📈 对比分析**

与传统的自回归解码（AR）和默认的推测解码（SD）进行比较，提出的方法在多个GPU上平均提高了2.01倍的吞吐量，相比于SD提高了1.36倍，且在没有持有GPU校准的情况下，Jev预测器的性能接近全校准模式。

**⚠️ 局限性**

局限性在于该研究仅聚焦于单一目标模型和单一扩散基础的草稿器，未来需要扩展到其他SD方法，并改进学习型预测器在某些工作负载和GPU组合上的性能表现。

---

## 481. An Affordable Vibrotactile Surface for Tactile Figure Recognition: Exploration Strategies of Blind and Sighted Users

**arXiv ID:** 2611.22614 | [PDF](https://arxiv.org/pdf/2611.22614v1)

**作者:** Uday Nath Gunda `[一作]` (Birla Institute of Technology and Science Pilani), Parthasarathy PD `[通讯]` (Birla Institute of Technology and Science Pilani)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9`

**🎯 论文内容**

研究了一种低成本的振动触觉表面，旨在帮助盲人和低视力人士识别简单图形，并分析他们的触觉阅读策略。

**💡 创新点**

创新点在于设计了一种可负担的振动触觉表面，采用了负载解耦的像素设计，使得振动电机在手指压力下仍能保持响应。

**🔧 技术方法**

使用了振动电机作为触觉像素，通过控制电机的激活来呈现触觉图形。

**📊 数据集**

使用了一个10x10的振动触觉表面，参与者通过触觉识别七个不同的图形。

**📈 对比分析**

与传统的触觉图形相比，盲人参与者的识别准确率与盲fold的视力参与者相当，但盲人参与者的识别速度约为后者的三倍，显示出不同的探索策略。

**⚠️ 局限性**

限制在于样本量较小，仅有九名参与者，且结果可能不适用于不同的盲人群体，此外，运动分析方法未能捕捉到手部的具体运动轨迹。

---

## 482. OAK-Mind: Zero-Training Mental Health Change Inference from Passive Sensing via Knowledge-Grounded Multi-Agent LLMs

**arXiv ID:** 2611.22248 | [PDF](https://arxiv.org/pdf/2611.22248v1)

**作者:** Yishu Ji `[一作]` (University of Washington), Xuhai "Orson" Xu `[通讯]` (Columbia University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为OAK-Mind的零训练、多代理、知识驱动的LLM管道，用于从被动感知数据中推断心理健康变化。该管道将感知数据组织为行为维度，并通过专业代理解释长期行为变化。

**💡 创新点**

OAK-Mind的创新点在于其无需任务特定训练，能够在没有额外标记样本的情况下，推断个体心理健康的变化，并提供基于观察到的行为模式和领域知识的可读性推理。

**🔧 技术方法**

使用了大型语言模型（LLMs）作为基础，结合了行为分析、跨领域综合和基于问卷的症状评分的多层代理系统。

**📊 数据集**

使用了两个公开的被动感知和心理健康数据集：GLOBEM和College Experience，这些数据集提供了纵向行为数据和自我报告的抑郁和焦虑测量。

**📈 对比分析**

OAK-Mind在多个数据集和预测任务中表现出与最具预测性的监督基线相当或更好的性能，且在四个测试的LLM基础上显示出一致的改进。与传统的监督学习方法相比，OAK-Mind在抑郁变化分类任务中表现尤为突出。

**⚠️ 局限性**

限制在于所使用的心理健康测量工具（PHQ-2和GAD-2）可能无法反映临床上有意义的变化，且数据集主要集中在大学生群体，可能不适用于其他人群或设备。

---

## 483. Fragmented Markers, Mixed Results: A Systematic Review of AI Coding Assistants and Developer Productivity

**arXiv ID:** 2611.21632 | [PDF](https://arxiv.org/pdf/2611.21632v1)

**作者:** Annemarie Wittig `[一作]` (Leipzig University), Norbert Siegmund `[通讯]` (Leipzig University)

**通讯引用:** 5881 | [OpenAlex ID](https://openalex.org/A5004072538)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

进行了分阶段的系统文献综述，旨在提高对AI编码助手对软件开发者生产力影响的可解释性和可比性。

**💡 创新点**

通过构建一个结构化的基础，使得生产力证据在不同研究之间更具可比性和可解释性，强调了人本维度的重要性。

**🔧 技术方法**

采用了分阶段的系统文献综述方法，结合了SPACE框架来分析生产力相关的开发者结果。

**📊 数据集**

分析了45篇论文，涵盖了多种数据来源，包括同行评审的研究、行业报告和预印本。

**📈 对比分析**

通过比较不同研究中的生产力标记，发现虽然大多数报告的效果是积极的，但在统计评估和客观结果中，负面和中性结果的比例更高，表明当前文献的积极性依赖于测量和报告的方式。

**⚠️ 局限性**

文献中对混杂变量的讨论不够系统，导致跨研究的可解释性受到限制，且当前的研究未能充分捕捉生产力的多维特征。

---

## 484. Rescaling OLS-Merged Adapters To Improve Personalized Image Generation

**arXiv ID:** 2611.22969 | [PDF](https://arxiv.org/pdf/2611.22969v1)

**作者:** Leander Girrbach `[一作]` (Technical University of Munich), Zeynep Akata `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出了一种通过重新缩放合并多个适配器以提高图像生成模型性能的方法，解决了传统最小二乘法（OLS）合并方法导致的信号缩减问题。

**💡 创新点**

创新点在于提出了一种重新缩放因子1/γ，通过分析OLS合并方法的缩减效应，显著提高了合并模型在多任务下的表现。

**🔧 技术方法**

使用了普通最小二乘法（OLS）和低秩适配器（LoRA）技术，结合了重新缩放策略来改善合并模型的性能。

**📊 数据集**

使用了DreamBooth数据集进行多主题个性化，WikiArt数据集用于艺术风格迁移，以及CIFAR-10数据集用于概念消除。

**📈 对比分析**

与现有的合并方法（如RegMean、Mix-of-Show和Subspace Routing）进行比较，重新缩放的SSR方法在合并30个DreamBooth适配器时显著提高了主题保真度，且在风格迁移和图像分类任务中也表现优异。

**⚠️ 局限性**

限制在于重新缩放方法是后处理方法，虽然能缓解缩减问题，但并未在合并过程中直接解决。此外，1/γ启发式方法在某些情况下可能会失败，且未考虑合并模型在不同层之间的激活漂移。

---

## 485. StitchRAG: Constructing Compact and Evidence-Preserving Contexts for RAG with Diffusion-Based Stitching

**arXiv ID:** 2611.22766 | [PDF](https://arxiv.org/pdf/2611.22766v1)

**作者:** Minsang Kim `[一作]` (Korea University), Seungjun Baek `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为StitchRAG的框架，用于在检索增强生成（RAG）中构建紧凑的上下文，通过动态压缩检索到的段落来提高信息密度和上下文的连贯性。

**💡 创新点**

创新点在于动态调整每个段落的保留信息比例，以最大化保留的信息和计算成本之间的效用，同时使用混合锚点选择和基于扩散的拼接方法来构建上下文。

**🔧 技术方法**

使用了动态大小分配、锚点保护和基于文本扩散的拼接技术。

**📊 数据集**

在五个开放域问答基准上进行实验，包括HotpotQA、Natural Questions、TriviaQA、2WikiMultiHopQA和PopQA。

**📈 对比分析**

与多种基线方法进行比较，StitchRAG在多个指标上表现优异，尤其是在处理多步和压缩基线时，显示出其动态上下文构建和拼接的有效性。

**⚠️ 局限性**

限制在于StitchRAG仍然需要多次模型推理，导致相较于单次RAG基线的开销增加，尽管通过批处理和并行化策略可以减轻这一负担。

---

## 486. A faster algorithm for commutator length in free groups

**arXiv ID:** 2611.22303 | [PDF](https://arxiv.org/pdf/2611.22303v1)

**作者:** Zixing Wang `[一作]` `[通讯]`, Zixing Wang

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种用于计算自由群中简化词的共轭长度的快速算法。

**💡 创新点**

该算法在确定性情况下以O(M^3g)的时间复杂度返回最多g个共轭的分解，显著提高了效率。

**🔧 技术方法**

使用了递归和Manacher算法来构建弧图，并通过三次跳跃的行走来寻找切点。

**📊 数据集**

使用了自由群中的简化词，长度为M的词作为数据集。

**📈 对比分析**

与现有方法（如BFI方法）相比，该算法在最坏情况下的时间复杂度为O(M^3g)，在g=1的情况下，期望运行时间为O(M)，性能显著提升。

**⚠️ 局限性**

算法在处理复杂度较高的词时可能仍然面临性能瓶颈，尤其是在g值较大时。

---

## 487. Ethics Suppression in AI Development: A Comparative Case Study of Responsible AI Under Competitive Pressure

**arXiv ID:** 2611.21629 | [PDF](https://arxiv.org/pdf/2611.21629v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 488. Adaptive Budget Allocation for Smart-Contract Fuzzing: Why Online Feedback Fails to Improve Vulnerability Triage

**arXiv ID:** 2611.22430 | [PDF](https://arxiv.org/pdf/2611.22430v1)

**作者:** Yanzi Liu `[一作]` (King's College London), Jiachi Tong `[通讯]` (Imperial College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了智能合约模糊测试中的预算分配问题，提出了一种离线概率曲线分配器，通过对143个标注合约和350个注入漏洞合约的追踪重放实验，测试在线反馈是否能改善漏洞报告的分类。

**💡 创新点**

创新点在于提出了一种离线概率曲线分配器，该分配器根据预算预测首次报告的概率，并通过每次执行的预期收益进行分配，显著提高了检测预算的AUC。

**🔧 技术方法**

使用了追踪重放技术和合成预言机实验，结合了多种在线和离线预算分配策略进行比较。

**📊 数据集**

使用了SmartBugs和SolidiFI两个数据集，SmartBugs包含143个标注合约，SolidiFI包含350个注入漏洞的合约。

**📈 对比分析**

与均匀分配方法相比，离线概率曲线分配器在四个新种子上的检测-预算AUC分别提高了5.38%（SmartBugs）和4.81%（SolidiFI），显示出其在预算分配上的优势。

**⚠️ 局限性**

限制在于所测试的在线策略在整体分类上未能超越均匀分配，且信号和合成预言机实验表明，当前的信号对即将到来的报告的预测能力较弱。

---

## 489. Simulating Quantum Computation Using Probabilistic Programming

**arXiv ID:** 2611.21904 | [PDF](https://arxiv.org/pdf/2611.21904v1)

**作者:** Yafei Yang `[一作]` (Indiana University), Chung-chieh Shan `[通讯]` (Indiana University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究将一种高层次的量子编程语言ISO编译为一种支持复杂或负权重的概率编程实现，即因子图文法（FGGs），并通过现有的FGG求解器模拟量子程序。

**💡 创新点**

创新点在于通过编译量子语言到概率语言，利用现有的概率编程基础设施来实现量子计算的模拟，同时证明了编译器和优化保持了源语言的语义。

**🔧 技术方法**

使用了因子图文法（FGGs）作为目标概率编程语言，并对现有FGG求解器进行了优化，以支持张量操作的重写。

**📊 数据集**

使用了量子编程语言ISO作为数据集，进行了一系列基准测试以评估性能。

**📈 对比分析**

通过与其他量子模拟器的比较，结果显示本实现的性能与其他量子模拟器相当，且通过优化提高了在重复模式下的模拟性能。

**⚠️ 局限性**

限制在于当前的实现依赖于现有的概率编程基础设施，可能在处理更复杂的量子程序时面临性能瓶颈，尤其是在编译时间和内存使用方面。

---

## 490. Efficient Robust Optimization via an Adaptive Mono-Level Gaussian Process Surrogate

**arXiv ID:** 2611.22467 | [PDF](https://arxiv.org/pdf/2611.22467v1)

**作者:** Julius Schultz `[一作]` (Technische Universität Braunschweig), Ulrich Römer `[通讯]` (Technische Universität Braunschweig)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于单一高斯过程代理模型的鲁棒优化方法，该方法通过自适应精炼减少了鲁棒优化所需的昂贵模型评估次数。

**💡 创新点**

创新点在于采用单层策略构建高斯过程代理模型，并通过采样基础的不确定性指标引导设计空间的精炼，从而提高了算法的可扩展性和效率。

**🔧 技术方法**

使用了高斯过程回归作为代理模型，并结合路径采样和权重空间表示来实现高效的后验采样。

**📊 数据集**

在多个分析基准模型上进行了评估，包括结构优化问题，具体数据集未详细列出。

**📈 对比分析**

与最先进的双层代理方法相比，所提出的方法在模型评估次数上表现出显著优势，同时在解决精度上保持了相当的准确性。

**⚠️ 局限性**

限制在于当前研究主要考虑了中等维度的增强输入空间，未来需要扩展到更高维度的问题，并探索多目标鲁棒优化的可能性。

---

## 491. Sketch, Don't Paint: Allocating Test-Time Compute in Generative Visual Reasoning

**arXiv ID:** 2611.21812 | [PDF](https://arxiv.org/pdf/2611.21812v1)

**作者:** Sheng Lu `[一作]` (Nanjing University), Ying Tai `[通讯]` (Nanjing University)

**通讯引用:** 17105 | [OpenAlex ID](https://openalex.org/A5029021362)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了多模态模型在生成视觉思维时的计算分配，探讨了何时渲染视觉思维以及每个思维需要多少去噪步骤。

**💡 创新点**

提出了草图投票（sketch-and-vote）方法，通过独立噪声生成草图并进行投票，以较低的去噪步骤提高准确性。

**🔧 技术方法**

使用了多模态模型和去噪调度技术，结合了图像和视频生成的思维过程。

**📊 数据集**

使用了多个基准数据集，包括迷宫、Sokoban、VisPuzzle、BLINK-Jigsaw、MMVP、VStar和ChartQA等。

**📈 对比分析**

通过与直接回答、仅文本思维和绘制视觉思维的比较，发现直接回答在9个模型-基准对中获胜，平均准确率从60.7%提高到65.1%。草图投票在图像和视频推理器上以较少的去噪步骤超越了完整渲染。

**⚠️ 局限性**

模型和任务的依赖性导致不同的思维模式表现不一，草图投票在某些情况下可能无法完全替代完整渲染。

---

## 492. More Than a Score: Reinforcement Learning with Verbalized Feedback for Generative Recommendation

**arXiv ID:** 2611.22410 | [PDF](https://arxiv.org/pdf/2611.22410v1)

**作者:** Zipeng Chen `[一作]` (Tencent Inc.), Chengguo Yin `[通讯]` (Tencent Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为VerRec的强化学习框架，利用口头反馈来改进生成推荐系统的性能。

**💡 创新点**

创新点在于将个性化的口头反馈作为明确的探索信号，指导强化学习中的推荐策略优化。

**🔧 技术方法**

使用了基于大语言模型（LLM）的用户模拟器生成个性化的口头反馈，并结合了联合优化的策略。

**📊 数据集**

在三个真实世界的亚马逊评论数据集上进行了实验，数据集包括手机及配件、婴儿用品和书籍。

**📈 对比分析**

与传统推荐系统和其他基于LLM的推荐系统相比，VerRec在推荐性能上表现优越，能够更好地引导探索并提高推荐质量。

**⚠️ 局限性**

限制在于需要依赖用户模拟器生成反馈，且在实际应用中可能难以获取真实用户的口头反馈。

---

## 493. Every Camera Grows Up: Camera-Height Regression in Interactive Video World Models

**arXiv ID:** 2611.21782 | [PDF](https://arxiv.org/pdf/2611.21782v1)

**作者:** Sheng Lu `[一作]` (Nanjing University), Ying Tai `[通讯]` (Nanjing University)

**通讯引用:** 17105 | [OpenAlex ID](https://openalex.org/A5029021362)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文展示了六种开放权重的交互视频世界模型在生成过程中未能保持相机高度，导致相机高度回归到人眼水平。通过引入新的基准，评估这些模型在不同高度下的表现。

**💡 创新点**

创新点在于提出了一个新的基准，专注于相机高度保持的评估，并揭示了模型在生成过程中相机高度回归的现象及其原因。

**🔧 技术方法**

使用了多视角三维重建技术来测量相机高度，并结合了不同的模型（如SANA-WM、Matrix-Game等）进行比较。

**📊 数据集**

使用了合成的室内场景数据集（如ReplicaCAD、Matterport3D等）和真实场景的校准照片（如Eyeful Tower）进行实验。

**📈 对比分析**

与地面真实渲染进行比较，所有六种模型在180度旋转和行走的过程中都出现了相机高度回归，保留率（β）在0.18到0.67之间，而地面真实渲染的保留率为0.95。

**⚠️ 局限性**

限制在于生成的回放缺乏真实的三维几何结构，主要度量依赖于多视角地面重建，并且未测试闭环效果。训练和指导干预主要集中在SANA-WM模型上，其他模型的补救措施留待未来研究。

---

## 494. SpikWM: Efficient End-to-End Spiking World Modeling for Visual Planning

**arXiv ID:** 2611.22981 | [PDF](https://arxiv.org/pdf/2611.22981v1)

**作者:** Shuhan Ye `[一作]` (Nanyang Technological University), Xudong Jiang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的端到端联合嵌入预测架构（JEPA），用于基于图像的目标条件规划，旨在通过视觉状态和动作后果的预测来提高规划效率。

**💡 创新点**

创新点在于提出了Spiking Regional Encoding（SRE）、Bi-Spike Action Encoding（BAE）和Causal Membrane Prediction（CMP）模块，以保留空间线索和区分不同动作的影响，同时减少计算步骤。

**🔧 技术方法**

使用了脉冲神经网络（SNN）技术，结合了稀疏计算和膜动态，进行高效的目标条件规划。

**📊 数据集**

使用了LeWM的离线数据集，包括Push-T、Two-Room、Reacher和OGBench-Cube等任务。

**📈 对比分析**

与两个脉冲基线模型相比，提出的模型在四个任务上均表现优异，成功率分别为88.67%、97.33%、99.33%和86.67%。在相同的规划预算下，模型的估计算术能耗比LeWM减少了75.6%。

**⚠️ 局限性**

限制在于模型在某些情况下可能无法完全保留所有空间和动作信息，尤其是在复杂的动态环境中，可能会影响预测的准确性。

---

## 495. Magflow: Differentiable Simulation of Industrial-Grade 6D Magnetic Levitation

**arXiv ID:** 2611.23050 | [PDF](https://arxiv.org/pdf/2611.23050v1)

**作者:** Philipp Hartmann `[一作]` (Bielefeld University), Klaus Neumann `[通讯]` (Fraunhofer IOSB-INA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种全新的可微分模拟器Magflow，用于工业级6D磁悬浮系统的控制，旨在通过直接计算物理学来训练神经控制器，而不是依赖于手工设计的近似模型。

**💡 创新点**

创新点在于首次实现了一个完全可微分的GPU并行模拟器，能够在不依赖手工近似的情况下，直接从模拟物理中学习控制策略，并且该控制器能够在真实硬件上零-shot转移。

**🔧 技术方法**

使用了可微分模拟技术，结合了GPU并行计算和反向传播，通过时间训练神经控制器。

**📊 数据集**

使用了一个工业级的6D磁悬浮系统的模拟数据集，该数据集包含了电磁力的计算、动态模拟和传感器测量。

**📈 对比分析**

与传统的手工设计控制器相比，Magflow训练的神经控制器在真实硬件上表现出新的行为，能够控制不符合共设计假设的磁体，并且在4kHz的硬实时约束下运行，性能超越了现有的控制方法。

**⚠️ 局限性**

限制在于Magflow仍然是一个理想化的模型，忽略了某些物理现象（如铁磁材料和去磁化），并且在真实环境中可能需要进一步的校准和调整以减少模拟与现实之间的差距。

---

## 496. Visual Harness Scales General VLMs to Medical Images

**arXiv ID:** 2611.22815 | [PDF](https://arxiv.org/pdf/2611.22815v1)

**作者:** Shaowen Wan `[一作]` (New Jersey Institute of Technology), Lin Zhao `[通讯]` (New Jersey Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种医学视觉工具VERA，旨在通过交互式视觉工具和系统化的诊断证据管理来提升通用视觉语言模型（VLMs）在医学成像任务中的表现。

**💡 创新点**

VERA通过提供有效的证据获取、保存和评估机制，显著提升了VLMs在医学成像任务中的性能，展示了视觉工具作为新范式的潜力。

**🔧 技术方法**

使用了医学视觉工具、无损证据记忆、证据地图、证据验证器和证据引导的代理循环等技术。

**📊 数据集**

在11个任务和四种医学成像模态（CT、MRI、超声和全切片病理）上进行了评估。

**📈 对比分析**

与直接图像输入相比，VERA在所有11个任务中均表现出显著提升，尤其是在脾脏测量（从63.3%提升至98.3%）、肾脏搜索（从57.5%提升至71.2%）和肺结节检测（从0.0%提升至28.2%）等任务中表现突出。

**⚠️ 局限性**

限制包括API成本导致的模型评估范围有限、公共数据集可能与专有VLM训练数据重叠、VERA性能依赖于VLM对医学图像的解释能力，以及缺乏在真实临床环境中的验证。

---

## 497. HubEgo-100K: In-the-Wild Egocentric Video of Skilled Work for Robot Learning

**arXiv ID:** 2611.21938 | [PDF](https://arxiv.org/pdf/2611.21938v1)

**作者:** Javad Amirian `[一作]` (Hub Labs), Karthik Yadati `[通讯]` (Hub Labs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文介绍了一个名为的第一版数据集v1.0，该数据集包含10,124小时的无脚本工作记录，记录者为12,227名贡献者，涵盖了15,548个环境、11种设置类型、30个工作类别和56个任务类型，旨在提高机器人学习的多样性。

**💡 创新点**

创新点在于数据集的多样性，涵盖了多种工作环境和任务类型，且记录者来自不同背景，确保了数据的广泛性和代表性。

**🔧 技术方法**

使用了头戴式手机摄像头记录的1080p视频，并同步了100Hz的惯性测量单元（IMU）数据。

**📊 数据集**

数据集包含来自12,227名贡献者的10,124小时的无脚本工作视频，记录了15,548个环境，涵盖了30个工作类别和56个任务类型。

**📈 对比分析**

与现有数据集相比，本数据集在记录者和环境的多样性上具有显著优势，且每个贡献者的记录时间分布均匀，没有任何贡献者占据超过0.22%的记录时间。性能评估将通过后续版本进行。

**⚠️ 局限性**

限制在于当前版本仅为v1.0，后续版本将扩展到100,000小时，且尚未进行基准测试以验证数据集对机器人学习性能的具体影响。

---

## 498. Runtime Verification of Multiple Natural Language Criteria for Agent Governance

**arXiv ID:** 2611.23074 | [PDF](https://arxiv.org/pdf/2611.23074v1)

**作者:** Silviu Pitis `[一作]` (University of Toronto), Sheila A. McIlraith `[通讯]` (Vector Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的验证模型VFM，用于高效评估AI代理的输出是否符合大量自然语言标准，并返回每个标准的独立判决。

**💡 创新点**

VFM通过树注意力机制并行处理多个标准，能够在运行时动态适应不同的验证任务，且在低数据环境下表现出色。

**🔧 技术方法**

使用了树注意力机制和预训练的VFM模型，能够在运行时快速评估多个标准。

**📊 数据集**

使用了CriteriaBank数据集，该数据集包含394K个（上下文、目标、标准、判决）元组，涵盖321K个不同的标准字符串。

**📈 对比分析**

与现有的验证方法相比，VFM在多个基准测试中表现优异，尤其是在DynaBench多标准基准测试中，VFM-4B在追踪级别的失败检测上超越了DynaGuard-4B。

**⚠️ 局限性**

VFM在真实保险福利验证通话中表现不够可靠，约20%的金标准违规被错误批准，且CriteriaBank标签是模型生成的，可能存在错误。

---

## 499. Harnessing Image Diffusion Prior for Photo-Realistic Video Restoration

**arXiv ID:** 2611.22874 | [PDF](https://arxiv.org/pdf/2611.22874v1)

**作者:** Fanghua Yu `[一作]` (University of Hong Kong), Jinjin Gu `[通讯]` (INSAIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了HYPI2VR方法，将强大的图像生成先验转移到视频恢复中，旨在在保持时间一致性的同时合成细致的细节。

**💡 创新点**

创新点在于通过图像生成先验实现时间一致的细节合成，解决了现有视频恢复方法在视觉保真度和时间一致性之间的矛盾。

**🔧 技术方法**

使用了噪声能量重新平衡、运动对齐噪声变形、自监督去噪编码器和解耦时间注意力等技术。

**📊 数据集**

使用了与HYPIR相同的高质量图像数据集和80K个过滤的YouTube视频片段，每个片段包含64-128帧，处理为1024×1024分辨率。

**📈 对比分析**

与现有视频恢复方法相比，HYPI2VR在视觉质量、细节丰富性和时间稳定性方面显著优于其他方法，定量比较显示其在多个基准测试中表现出色。

**⚠️ 局限性**

限制在于尽管HYPI2VR在视觉质量和时间一致性上表现良好，但在某些情况下可能仍会出现过度平滑或细节丢失的问题。

---

## 500. Operator Precedence Tree Automata

**arXiv ID:** 2611.22792 | [PDF](https://arxiv.org/pdf/2611.22792v1)

**作者:** Matteo Pradella `[一作]` `[通讯]` (Politecnico di Milano), Matteo Pradella (Politecnico di Milano)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

提出了运算优先树自动机（OPTA），将单词的句法结构构建与识别分开，利用运算优先矩阵确定规范树，并通过有限的自底向上的自动机进行评估。

**💡 创新点**

OPTA精确识别运算优先语言，并且在任意无冲突矩阵下都成立，构建了有效可计算的规范完整确定性精确OPTA，且在垂直状态数量上是最小的。

**🔧 技术方法**

使用了运算优先矩阵和有限树识别器，结合了树自动机的水平转换语言，采用了上下文Myhill-Nerode构造。

**📊 数据集**

使用了运算优先语言（OPL）相关的数据集，特别是无冲突的运算优先矩阵。

**📈 对比分析**

通过与现有的OP语言的特征进行比较，证明了OPTA在识别能力上与OP语言等价，并且在构造的规范最小OPTA中，状态数量是唯一的且最小。

**⚠️ 局限性**

限制在于对于循环矩阵的处理，尽管OPTA能够处理无冲突矩阵，但在某些情况下可能无法保证唯一性。

---

## 501. DGF-Asym: Adaptive Peer Weighting for Collective Estimation under Noisy and Faulty Sensing

**arXiv ID:** 2611.22532 | [PDF](https://arxiv.org/pdf/2611.22532v1)

**作者:** Mohammad Nomaan Husain `[一作]` (University of Technology Nuremberg), Tanja Katharina Kaiser `[通讯]` (University of Technology Nuremberg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为DGF-Asym的去中心化集体估计方法，扩展了DeGroot-Friedkin模型，通过基于回顾性预测性能的非对称同行加权来平衡个人观察和社交信息。

**💡 创新点**

创新点在于动态的同行特定社交信息加权机制，使机器人能够根据不同邻居的信息质量分配不同的重要性，而不是在原始模型中对所有邻居赋予相同的重要性。

**🔧 技术方法**

使用了去中心化的集体估计方法，基于DeGroot-Friedkin模型，支持异步通信，并且不需要预先定义的传感器噪声统计或显式故障模型。

**📊 数据集**

在一个物理基础的多机器人模拟器ARGoS中进行评估，使用了多个机器人在噪声观察和时间变化的传感器故障下进行集体导航的模拟数据。

**📈 对比分析**

与三个基线方法相比，DGF-Asym在估计稳定性和决策标准的持续性方面表现更好，但决策时间较长。性能依赖于故障比例、组大小和影响阈值的选择。

**⚠️ 局限性**

限制在于整体性能依赖于故障比例、组大小和影响阈值的选择，且需要对影响阈值进行配置依赖的调优以适应不同的通信活动。

---

## 502. Newton's Method for CP Decomposition

**arXiv ID:** 2611.21814 | [PDF](https://arxiv.org/pdf/2611.21814v1)

**作者:** DuBose Tuller `[一作]` (Wake Forest University), Nicholas I. M. Gould `[通讯]` (STFC)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种使用牛顿法计算CP张量分解的算法。

**💡 创新点**

通过使用真实的Hessian矩阵，展示了牛顿法在大残差情况下也能实现快速收敛，克服了高噪声数据的挑战。

**🔧 技术方法**

使用了牛顿法和高斯-牛顿法的混合方法，并结合了全局化和正则化技术。

**📊 数据集**

使用了合成数据集和两个应用数据集进行数值实验。

**📈 对比分析**

与其他方法（如高斯-牛顿法和交替最小二乘法）进行了比较，结果表明混合方法在大残差情况下表现优越，收敛速度更快。

**⚠️ 局限性**

牛顿法在远离解时可能会表现出不稳定性，导致收敛速度较慢。

---

## 503. SEAM: Computer-Use Agents Know the Harm Between Screens, and Commit It Anyway

**arXiv ID:** 2611.22971 | [PDF](https://arxiv.org/pdf/2611.22971v1)

**作者:** Yuxuan Zhou `[一作]` (Tsinghua University), Shu-Tao Xia `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个新的基准（Safety Execution Across Multiple screens），用于评估计算机使用代理（CUAs）在多个屏幕之间组合信息时可能产生的危害，包含600个多步骤的跨应用任务。

**💡 创新点**

创新点在于识别了CUA安全评估中的盲点，即危害并不在任务意图或单一屏幕中，而是存在于多个屏幕信息的组合中。

**🔧 技术方法**

使用了行为评估方法，通过服务器日志记录任务的执行结果，而不是依赖于大型语言模型（LLM）进行判断。

**📊 数据集**

使用了OSWorld虚拟机中的600个场景，涵盖150条规则和27个危害类别，涉及多种应用领域。

**📈 对比分析**

与现有方法比较，开放权重和旗舰模型在执行危害时的成功率高达92%，而在用户明确指出危害时，执行率显著下降，显示出安全识别与安全执行之间的脱节。

**⚠️ 局限性**

限制在于自托管应用和生成规则在自然性上有所妥协，仅支持英语，每个变体仅在每个配置中运行一次，且对书面推理的分析依赖于LLM评判。

---

## 504. Practice in Text, Act in Pixels: Correcting Where Perception Fails

**arXiv ID:** 2611.22722 | [PDF](https://arxiv.org/pdf/2611.22722v1)

**作者:** Zhongling Xu `[一作]` (University of Texas at Austin), Hairong Wang `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了在图形用户界面（GUI）代理中，如何有效利用有限的截图预算来提高任务成功率，特别是在从文本视图到截图的转变中，识别和纠正屏幕阅读器的误读。

**💡 创新点**

创新点在于提出了“阅读器对齐”方法，通过对齐屏幕阅读器的输出与文本视图，显著提高了任务成功率，超越了直接在截图上训练策略的效果。

**🔧 技术方法**

使用了视觉-语言屏幕阅读器（Qwen3-VL-4B）和一种新的对齐技术来优化阅读器的输出，使其与文本视图一致。

**📊 数据集**

使用了MobileGym平台的模拟移动应用程序生成的截图数据集，包含多个任务实例和相应的文本视图。

**📈 对比分析**

通过与五种不同的训练方法进行比较，发现阅读器对齐在相同预算下的任务成功率明显高于直接在截图上训练策略，且对齐后的阅读器提升了所有文本训练策略的表现。

**⚠️ 局限性**

限制在于该方法依赖于特定平台的截图和文本视图，可能不适用于所有类型的应用程序或界面，且对齐过程需要一定的标注数据。

---

## 505. MIRAGE: Motion Identity Reduction via Autoregressive Generative Encoding for Privacy-Preserving Skeleton-based Motion Data

**arXiv ID:** 2611.22200 | [PDF](https://arxiv.org/pdf/2611.22200v1)

**作者:** Thomas Carr `[一作]` (Incerta Intelligence), Aidong Lu `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种目标无关的因果匿名化器，能够在原始坐标空间中为每个输入帧添加身份减少的残差，利用过去帧的注意力计算，以便依赖于时间上的运动变化。

**💡 创新点**

创新点在于引入了因果注意力机制，使得匿名化器能够利用时间上下文来增强隐私保护，同时保持与原始数据的兼容性。

**🔧 技术方法**

使用了因果编码器-解码器Transformer架构，结合了旋转位置嵌入和自注意力机制。

**📊 数据集**

使用了NTU RGB+D 60数据集，该数据集包含来自不同相机视角的单人动作序列。

**📈 对比分析**

与Moon等人的每帧残差匿名化器进行了比较，结果显示在保持动作识别准确率的同时，身份重识别率有所降低，但在全残差规模下，原始训练的动作分类器无法有效识别其输出。

**⚠️ 局限性**

局限性包括：没有提供正式的隐私保证，适应性攻击者仍然能够识别出比随机猜测更多的身份，且在全残差规模下，原始训练的动作分类器无法直接应用于其输出。

---

## 506. CARE: Conversational Affective Response Retrieval for Internet Memes

**arXiv ID:** 2611.23005 | [PDF](https://arxiv.org/pdf/2611.23005v1)

**作者:** Jiajun Li `[一作]` (Beijing University of Posts and Telecommunications), Qiang Yan `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一种对话式情感响应检索框架CARE，旨在支持在线交流中的表情包检索，结合情感增强的CLIP检索器和基于Jina的重排序器。

**💡 创新点**

创新点在于引入了情感描述的门控融合机制和多正例对比学习，利用参与度监督进行重排序，从而提高了表情包的检索效果。

**🔧 技术方法**

使用了CLIP模型作为基础检索器，并结合了Jina作为重排序器，采用了多正例对比学习和参与度监督的训练方法。

**📊 数据集**

使用了来自豆瓣和Imgflip的两个语言特定的数据集，共包含98,506个表情包记录，涵盖5,583个讨论主题。

**📈 对比分析**

与微调的CLIP模型相比，CARE在豆瓣的Hit@10达到了24.03%，在Imgflip达到了49.33%，相较于微调的CLIP分别提高了0.87和8.26个百分点，显示出显著的性能提升。

**⚠️ 局限性**

限制在于弱相关性和参与度标签的使用，线程成员资格提供的监督不够完整，且情感描述的自动注释可能存在错误，未来需要进行人类评估以验证语义相关性和情感兼容性。

---

## 507. Can Verifiable PIR Match the Online Efficiency of SimplePIR?

**arXiv ID:** 2611.22911 | [PDF](https://arxiv.org/pdf/2611.22911v1)

**作者:** Xinxuan Zhang `[一作]` (Institute of Information Engineering), Xianhui Lu `[通讯]` (Institute of Information Engineering)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的可验证私有信息检索(vPIR)协议，该协议通过设计新的基于纠错码的承诺方案来确保数据库摘要的完整性，支持高效的线性关系证明和精确的范围证明，同时消除了对短整数解(SIS)假设的依赖。

**💡 创新点**

该协议是首个在在线通信和服务器计算成本上与简单PIR完全相同的vPIR协议，同时保持在线客户端计算几乎不变。与VeriSimplePIR相比，该协议在在线通信和客户端计算上减少了约20%，在服务器端计算上减少了超过35%。

**🔧 技术方法**

使用了基于纠错码的承诺方案、线性关系证明和范围证明技术，结合了学习与错误(LWE)假设，避免了对SIS假设的依赖。

**📊 数据集**

使用了多个数据库大小的数据集，包括512 MB、1 GB、4 GB、8 GB和16 GB，进行性能评估。

**📈 对比分析**

与VeriSimplePIR和Rathee等人的方案相比，该协议在在线通信和计算性能上显著优越，在线延迟在69 ms到1244 ms之间，通信量在209 KiB到1182 KiB之间，且在大数据库规模下性能提升明显。

**⚠️ 局限性**

该协议的局限性在于其依赖于LWE假设，尽管在性能上优于现有的vPIR方案，但在安全性上仍需进一步验证其在实际应用中的有效性。

---

## 508. Consistency Is Not Correction: Separating Confidence, Abstention, and Recovery in LLM Agents

**arXiv ID:** 2611.21651 | [PDF](https://arxiv.org/pdf/2611.21651v1)

**作者:** Khushal Sethi `[一作]` `[通讯]`, Khushal Sethi

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了在推理过程中使用的信号的三个目标：区分能力、选择性预测和修正效用，并对其进行了测量和分析。

**💡 创新点**

创新点在于明确区分这三个目标，并提供了修正效用的精确分解，揭示了实验设计的限制。

**🔧 技术方法**

使用了推理时间控制器，包括提示扰动验证、自适应一致性、在线信号选择控制器和标记日志概率等技术。

**📊 数据集**

使用了MiniHouse和GSM8K数据集进行实验，MiniHouse是一个确定性的文本环境，而GSM8K提供了算术文字问题。

**📈 对比分析**

通过比较不同控制器的表现，发现所有控制器在GSM8K决策中没有替换任何提案，表明比较未能测试其机制，且在小模型实验中高区分能力与无修正效用并存。

**⚠️ 局限性**

限制在于控制器几乎没有触发，导致无法有效测试机制，且实验设计未能满足所需的错误预算和不一致对的数量，无法得出积极结果。

---

## 509. Addressing Exogenous Variability in Cooperative Multi-Agent Reinforcement Learning

**arXiv ID:** 2611.22067 | [PDF](https://arxiv.org/pdf/2611.22067v1)

**作者:** Seongmin Kim `[一作]` (Korea Advanced Institute of Science and Technology), Youngchul Sung `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的算法LEICA，用于解决在外部变化下的合作多智能体强化学习（MARL）问题，特别是针对不可控的对手策略变化和环境变化。

**💡 创新点**

创新点在于引入了外生决策部分可观察马尔可夫决策过程（ED-POMDP）框架，并通过影响加权的反事实内在奖励来塑造策略更新，从而提高了在未见对手策略下的泛化能力。

**🔧 技术方法**

使用了集中训练分散执行（CTDE）方法，结合历史条件上下文表示和影响加权的内在奖励。

**📊 数据集**

在SMAX基准测试上进行了评估，涉及对手策略和初始条件的训练/测试变化，并在MABrax和MPE-Tag上进行了补充实验。

**📈 对比分析**

与现有的CTDE、上下文推断和鲁棒MARL基线进行比较，LEICA在训练性能和对未见对手策略的泛化能力上均表现出显著提升，尤其在未见配置下的表现优于其他基线。

**⚠️ 局限性**

限制在于LEICA依赖于从训练回合中识别的动作依赖状态结构，其在噪声过大、隐藏变量、动态变化或数据有限的情况下效果可能下降。此外，自动分区在环境变化时可能需要更新。

---

## 510. TetherMark: Instance-Level Watermark Authentication Against Cross-Instance Transplantation

**arXiv ID:** 2611.22345 | [PDF](https://arxiv.org/pdf/2611.22345v1)

**作者:** Jin Wei `[一作]` (Fudan University), Xiaoyan Sun `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种实例级水印认证框架，旨在防止跨实例移植攻击，确保水印证据与注册实例一致。

**💡 创新点**

创新点在于引入了实例级水印认证，结合了消息检查和关系检查，确保水印证据不仅可解码，还与注册实例一致。

**🔧 技术方法**

使用了基于深度学习的水印嵌入和验证技术，包括VINE和WAM作为消息载体，以及关系检查机制。

**📊 数据集**

使用了Stanford Online Products数据集，包含240个测试对象和2400个合法编辑。

**📈 对比分析**

与其他方法（如TrustMark-P、Pixel Seal等）比较，达到了92.08%的真实接受率（TAR），在WmForger和WMCopier攻击下的攻击成功率（ASR）分别为9.17%和16.25%。

**⚠️ 局限性**

局限性在于相较于其他方法，虽然在实例绑定准确性上表现优异，但在图像保真度和嵌入/检测延迟上有所妥协。

---

## 511. TAILROLL: Tail-Aware Rollout Acceleration for RL-Based LLM Post-Training

**arXiv ID:** 2611.22206 | [PDF](https://arxiv.org/pdf/2611.22206v1)

**作者:** Niloofar Gholipour `[一作]` (Ecole de technologie superieure), Yong Zhang `[通讯]` (Huawei Technologies)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在线调度器，集成到开源的强化学习框架中，以加速大型语言模型的后训练过程，特别是解决了由于长尾响应导致的空闲时间问题。

**💡 创新点**

创新点在于区分了不平衡浪费和滞后浪费，并提出了基于无状态的请求迁移和充分性门控终止的调度策略，从而有效减少了训练过程中的空闲时间。

**🔧 技术方法**

使用了在线调度、无状态请求迁移、成本感知的迁移策略和充分性门控终止等技术。

**📊 数据集**

在Ascend 910C NPUs上使用了Qwen3-14B和Qwen3-32B模型进行评估，涉及多种工作负载，包括长链RL和生产派生工作负载。

**📈 对比分析**

与现有方法相比，提出的方法在十种配置中减少了21-74%的回滚气泡比率，减少了24-34%的回滚时间，提升了生成吞吐量高达33%，同时保持了相似或更高的最终奖励。

**⚠️ 局限性**

限制在于需要在运行时进行调度，可能会增加系统的复杂性和运行开销，尤其是在请求迁移和重新填充的过程中。

---

## 512. Embedding or bridging? AI4Science teams differ more in knowledge recombination than in problem novelty

**arXiv ID:** 2611.22366 | [PDF](https://arxiv.org/pdf/2611.22366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 513. MedGuardBench: A Patient-Oriented Benchmark for Personalized Guardrail Development

**arXiv ID:** 2611.22069 | [PDF](https://arxiv.org/pdf/2611.22069v1)

**作者:** Sara Khosravi `[一作]` (University of California), Muhao Chen `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种患者条件的药物安全监测方法，预测药物处方的安全性和17种风险类别。

**💡 创新点**

创新点在于引入个性化推理监测，考虑患者的具体情况，而不是仅依赖表面线索。

**🔧 技术方法**

使用了个性化推理蒸馏、上下文展开（CU）和基于规则的奖励（引导GRPO）等技术。

**📊 数据集**

使用了MedGuardBench数据集，该数据集包含5001个场景，涵盖87种药物。

**📈 对比分析**

与基础模型和无患者资料的对照组相比，个性化监测在准确性上表现更好，同时减少了误报。

**⚠️ 局限性**

局限性包括数据集的结构化和门诊范围，少数类别的支持不足，以及部分人工标注的限制。

---

## 514. Hierarchical Personalized Federated Learning for Fine-Grained Intrusion Detection: A Multi-Class Analysis in Heterogeneous Attack Environments

**arXiv ID:** 2611.21970 | [PDF](https://arxiv.org/pdf/2611.21970v1)

**作者:** Abul Kalam Azad `[一作]` (University of Alabama), Travis Atkison `[通讯]` (University of Alabama)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了层次化个性化联邦学习（HPFL）在异构攻击环境下进行细粒度入侵检测的收敛动态和性能，使用多层感知器（MLP）模型对CICIDS2017数据集进行8类、6类和4类入侵检测。

**💡 创新点**

创新点在于结合多层聚合与客户端级个性化，提出了一种新的HPFL框架，能够有效处理非独立同分布（non-IID）攻击分布，并实现细粒度的攻击分类。

**🔧 技术方法**

使用了多层感知器（MLP）模型，并在NVIDIA FLARE上实现了HPFL框架。

**📊 数据集**

使用了CICIDS2017数据集，该数据集包含2830743个网络流量样本，涵盖14种攻击类别。

**📈 对比分析**

与FedAvg、FedProx和FedRep等方法进行比较，HPFL在8类任务中实现了宏F1分数0.8324，显著优于FedAvg（0.7176）和FedProx（0.5624）。在跨客户端评估中，HPFL在4类任务中实现了微F1分数0.9294，优于独立本地模型的0.9140。

**⚠️ 局限性**

限制在于HPFL主要针对细粒度多类入侵检测进行了研究，未来工作将探索不同架构（如CNN和递归模型）以及其他数据集的应用。

---

## 515. Mind the Gap: A Deployed Multi-Agent System for Documentation Maintenance

**arXiv ID:** 2611.23142 | [PDF](https://arxiv.org/pdf/2611.23142v1)

**作者:** Lorenzo Masoero `[一作]` `[通讯]` (Amazon), Lorenzo Masoero (Amazon)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种多代理系统，名为TriAgent，旨在自动化维护软件文档，以解决文档滞后于代码更新的问题。

**💡 创新点**

创新点在于将文档维护视为推理问题，通过解决支持票据来识别文档缺口，并利用多代理系统自动修复这些缺口。

**🔧 技术方法**

使用了多代理系统技术，包括TriAgent、反馈循环代理、档案管理员、学生代理、代码合并观察者和文档撰写者等。

**📊 数据集**

使用了内部支持票据数据集，这些票据包含客户标识符和生产细节。

**📈 对比分析**

与传统的检索生成方法相比，TriAgent的响应在78.1%的情况下被偏好，而人类响应在18.2%的情况下被偏好。修复文档缺口后，代理的后续回答质量显著提高。

**⚠️ 局限性**

限制在于该系统仅在一个服务的支持队列上进行评估，可能不适用于其他特征不同的服务；此外，所有文档更改都需要人工审核，可能影响效率。

---

## 516. Manufactured Authenticity: How Behavioral Signals Are Produced and Misread in Online Markets

**arXiv ID:** 2611.22261 | [PDF](https://arxiv.org/pdf/2611.22261v1)

**作者:** Hou-Wan `[一作]`, Yun Huang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在线平台上用户如何从行为信号中推断可信度，提出了“制造的真实性”这一概念，分析了在自动化支持的在线市场中，如何通过可观察的活动生成合法性信号。

**💡 创新点**

创新点在于引入了“制造的真实性”作为框架，探讨了重复生产如何提供可被解读为合法性的行为线索，并识别了四种不同的生产策略。

**🔧 技术方法**

使用了K-means聚类技术对创作者钱包的行为进行分类，并分析了区块链上的活动和平台元数据。

**📊 数据集**

数据集包括996,834个加密货币代币和50,895个创作者钱包，数据时间范围为2024年1月至2026年4月。

**📈 对比分析**

通过比较不同生产策略的行为特征，发现高速度和模板内容策略是替代关系而非累积关系，前者在规模上产生活动，后者则与更高的回复参与度相关，但可见的参与并不意味着更广泛的传播。

**⚠️ 局限性**

限制在于分析使用了观察性数据，无法建立因果关系或参与者意图，且毕业模型缺乏审查调整。

---

## 517. SIDUMN: Self-Supervised Image Denoising Under Symmetric Multiplicative Noise

**arXiv ID:** 2611.22317 | [PDF](https://arxiv.org/pdf/2611.22317v1)

**作者:** Nyan Htet Naing `[一作]` (Kyungpook National University), Gyeongha Hwang `[通讯]` (Yeungnam University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为SIDUMN的自监督图像去噪方法，专门针对均值为1的对称乘法噪声，通过生成两个互补的噪声图像进行训练。

**💡 创新点**

创新点在于结合了对称合成扰动策略和反对称平均推理，提出了一种新的训练损失目标，并证明反对称平均不会增加单个噪声输入重建的期望平方误差。

**🔧 技术方法**

使用深度神经网络进行去噪训练，采用反对称平均推理来提高重建质量。

**📊 数据集**

使用BSD300和ImageNet数据集进行实验，生成带有合成噪声的图像进行训练和测试。

**📈 对比分析**

与DIP、Noise2Self和LogNN2N等自监督去噪方法进行比较，SIDUMN在所有噪声水平上均表现出最佳的PSNR，超出最强基线0.2-1.3 dB，且在SSIM上也表现良好。

**⚠️ 局限性**

局限性在于对真实噪声水平的敏感性，噪声水平的错误指定会导致性能下降，尤其是当假设的噪声水平与真实水平不匹配时。

---

## 518. When Does Discrete Ricci Curvature Help? An Edge-Level Analysis of Curvature-Enhanced GCNs for PPI Prediction

**arXiv ID:** 2611.21849 | [PDF](https://arxiv.org/pdf/2611.21849v1)

**作者:** Clara Joswig `[一作]` (ETH Zurich), Nico Scherf `[通讯]` (Max Planck Institute for Human Cognitive and Brain Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了曲率增强图卷积网络（CGCN）在蛋白质-蛋白质相互作用（PPI）数据集上的表现，特别是其在消息传递中引入Ollivier-Ricci曲率（ORC）的影响。

**💡 创新点**

创新点在于提出CGCN在提高聚类分离度的同时，可能会导致低曲率（瓶颈）边的性能下降，这与传统的过度压缩（over-squashing）理论相悖。

**🔧 技术方法**

使用了曲率加权的消息传递技术，具体通过ORC来调整边的权重。

**📊 数据集**

使用了四个真实的PPI数据集：HuRI-PPI、PPT-Ohmnet、PP-Pathways和PP-Decagon。

**📈 对比分析**

与标准GCN相比，CGCN在所有四个数据集上提高了聚合AUC、AUPR和F1分数，但这种提升主要集中在低度边上，而在低曲率边上的性能在三个数据集中系统性下降。

**⚠️ 局限性**

限制在于观察结果仅限于四个分析的数据集，且模型在其他数据集上的表现可能会有所不同，特别是如果使用不同的训练集采样策略。

---

## 519. The Kernel Reality Check: Benchmarking and Distilling Efficient Attention at Scale

**arXiv ID:** 2611.22947 | [PDF](https://arxiv.org/pdf/2611.22947v1)

**作者:** Fırat Öncel `[一作]` (Concordia University), Çağatay Yıldız `[通讯]` (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文分析了核化注意力方法的效率和质量，指出当前的核化方法在实际应用中未能实现预期的效率提升，并提出了LARA++作为一种改进的方法。

**💡 创新点**

创新点在于提出了LARA++，通过改进的重要性采样和确定性特征来缩小与softmax的质量差距，显示出核化方法的质量损失并非内在问题。

**🔧 技术方法**

使用了LARA++方法，该方法基于重要性采样和确定性特征，结合了聚类技术以提高注意力计算的效率。

**📊 数据集**

使用了DeiT和DINOv2数据集进行实验，评估了不同模型在图像分类和自监督学习任务上的表现。

**📈 对比分析**

与现有的核化方法（如Hedgehog和DiJiang）进行比较，发现LARA++在质量上接近softmax，但在效率上仍存在较大开销，FlashAttention在大多数情况下表现更优。

**⚠️ 局限性**

限制在于LARA++虽然在质量上有所提升，但其计算开销仍然较大，未能解决核化方法的效率问题，且在实际应用中仍需依赖硬件优化的精确方法。

---

## 520. Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration

**arXiv ID:** 2611.22275 | [PDF](https://arxiv.org/pdf/2611.22275v1)

**作者:** Ting-Hsuan Chen `[一作]` (University of Southern California), Yue Wang `[通讯]` (VinMotion)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种去中心化的框架，用于长时间协作的多机器人运动操作，允许机器人在没有网络通信的情况下进行协调和同步执行。

**💡 创新点**

创新点在于结合了多代理后训练框架和视觉-语言规划器，能够在本地观察的基础上实现去中心化控制和长时间规划。

**🔧 技术方法**

使用了多代理强化学习、视觉-语言模型和基于语音的领导-跟随协调机制。

**📊 数据集**

使用了一个包含12个任务的基准数据集，涉及2到10个机器人，包括异构团队的协作任务。

**📈 对比分析**

与基线方法相比，系统在94%的模拟试验中成功，而基线方法的成功率最高为29%。在真实世界实验中，系统在100次试验中成功92次，表现出强大的去中心化协作能力。

**⚠️ 局限性**

限制在于同步依赖于跟随者能够听到领导者的指令，因此需要相对安静的环境和可靠的声音接收。同时，训练规划器需要手动收集演示数据，耗时较长。

---

## 521. Feedback-Linearised Trajectory Tracking via Model-Informed Robust Geometric DDPC

**arXiv ID:** 2611.23215 | [PDF](https://arxiv.org/pdf/2611.23215v1)

**作者:** Pranav Gupta `[一作]` (Indian Institute of Technology Bombay), Ravi Banavar `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本论文研究了一种几何鲁棒数据驱动预测控制（DDPC）方法，用于全驱动机械臂的轨迹跟踪，采用基于模型的反馈线性化技术。

**💡 创新点**

创新点在于通过几何鲁棒DDPC处理轨迹拟合误差，最小化最坏情况下的预测误差，并在开式Grassmann球内进行优化。

**🔧 技术方法**

使用了反馈线性化技术来将虚拟加速度命令映射到关节扭矩，并结合几何鲁棒DDPC进行轨迹预测。

**📊 数据集**

使用了平面2R机械臂的模拟数据集，考虑了重力和测量噪声的影响。

**📈 对比分析**

与双积分模型预测控制（MPC）和直接正则化数据驱动预测控制（DeePC）进行比较，结果显示在双积分MPC下的跟踪性能有所改善，尽管扭矩努力增加，而相对于DeePC的性能则依赖于参考轨迹。

**⚠️ 局限性**

限制在于模型参数的不确定性可能导致预测误差，且在高噪声条件下，跟踪误差和执行器活动增加。

---

## 522. EDA: Observe, Verify, and Assess for Evidence-Grounded Multimodal Depression Analysis

**arXiv ID:** 2611.22648 | [PDF](https://arxiv.org/pdf/2611.22648v1)

**作者:** Changzeng Fu `[一作]` (Northeastern University), Shiqi Zhao `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于证据的框架EDA，旨在改善多模态抑郁评估方法，通过观察、验证和评估三个阶段来处理行为证据。

**💡 创新点**

创新点在于构建了源链接的行为证据记录，明确了观察的行为证据、验证其来源的机制以及如何在评估中使用这些证据。

**🔧 技术方法**

使用了冻结的7B模型进行多模态代理的验证，并结合统计描述符和时间事件进行行为比较。

**📊 数据集**

使用了D-Vlog和LMVD数据集进行实验，评估人类参考的线索一致性和下游评估效用。

**📈 对比分析**

与直接预测方法相比，EDA Support在D-Vlog Leave-Query-Out和LMVD上分别提高了7.61和9.88个百分点的F1分数，显示出其在评估中的有效性。

**⚠️ 局限性**

限制在于虽然EDA提供了可检查的证据记录，但仍需进一步研究如何优化证据的质量和评估效用。

---

## 523. Do LLMs Understand Network Traces? A Comparative Study of Reasoning over Raw and Filtered Traffic Data

**arXiv ID:** 2611.22575 | [PDF](https://arxiv.org/pdf/2611.22575v1)

**作者:** Irene Gosálvez-White `[一作]` (University of Granada), Pedro García-Teodoro `[通讯]` (University of Granada)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一种可重复的评估方法，通过问答任务评估大型语言模型（LLMs）在网络流量分析中的能力，使用CICIDS2017数据集中的真实网络流量进行评估。

**💡 创新点**

创新点在于建立了一种可重复的评估方法，比较不同LLMs在网络流量分析中的表现，并探讨协议过滤对答案准确性的影响。

**🔧 技术方法**

使用了大型语言模型（LLMs）作为评估工具，并开发了自动化的评估管道，结合了人类评估和LLM评估。

**📊 数据集**

使用了CICIDS2017数据集，该数据集包含正常流量和已知攻击流量，特别是选择了2017年7月3日的正常流量进行实验。

**📈 对比分析**

通过与人类生成的基准答案进行比较，评估了不同模型的表现。结果表明，协议过滤显著提高了答案的准确性，并减少了错误响应的数量。

**⚠️ 局限性**

本研究的局限性包括静态的协议特定视图、依赖单一捕获和注释者，以及受限于各个LLM上下文窗口的输入数据大小。

---

## 524. Efficient Wide-Beam Generative Retrieval with Model-Native Trie Decoding and Shared-Context Attention

**arXiv ID:** 2611.22309 | [PDF](https://arxiv.org/pdf/2611.22309v1)

**作者:** Vedant Jhaveri `[一作]` (LinkedIn), Sriram Vasudevan `[通讯]` (LinkedIn)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了一种高效的宽束生成检索方法，利用模型原生的Trie解码和共享上下文注意力机制，旨在提高生成检索的效率和性能。

**💡 创新点**

创新点在于将有效的输出结构作为模型执行的一部分，而不是外部回调，同时引入共享上下文注意力机制以减少重复计算，从而显著提高了检索系统的吞吐量和响应速度。

**🔧 技术方法**

使用了模型原生的Trie解码技术和共享上下文注意力机制，结合了SGLang框架进行实现。

**📊 数据集**

使用了LinkedIn的招聘助手和资格检索平台的实际工作负载数据集进行评估。

**📈 对比分析**

与基于Python的自定义logit处理器（CLP）相比，结合的系统在资格检索平台的工作负载上提高了闭环吞吐量，最高可达5.57倍，且在常见的束宽配置下，性能优于BF16 FlashRec。

**⚠️ 局限性**

限制在于该实现仍需标准的词汇投影和兼容的logit变换，且只支持固定大小的三元组SID，而不支持任意语法或请求依赖的约束。

---

## 525. SE(3)-LIVOM: Adaptive Multi-LiDAR-Inertial-Visual Odometry and Mapping

**arXiv ID:** 2611.22047 | [PDF](https://arxiv.org/pdf/2611.22047v1)

**作者:** Gunhee Shin `[一作]` (Korea Advanced Institute of Science and Technology), Hyun Myung `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了SE(3)LIVOM，一个用于腿部机器人在复杂场景中实时运行的多LiDAR-惯性-视觉里程计和地图构建系统。

**💡 创新点**

创新点在于扩展了SE(3)LIO以支持多个LiDAR和相机，并提出了自适应下采样技术以保持计算时间，同时引入了在线后端进行平面束调整和增量位姿图。

**🔧 技术方法**

使用了多传感器融合技术，包括两个LiDAR、一个相机和一个IMU，采用自适应下采样和在线后端处理。

**📊 数据集**

使用了GrandTour数据集，该数据集记录了腿部机器人在倒塌建筑、施工现场和雪地中的轨迹。

**📈 对比分析**

与其他方法（如FAST-LIVO2）相比，SE(3)LIVOM在六个测试场景中达到了平均绝对轨迹误差（ATE）1.40 cm，且在实时性能上表现良好。

**⚠️ 局限性**

限制在于对于大型场景的束调整过程可能需要较长时间（最多19秒），未来工作将致力于提高大场景的后端处理速度。

---

## 526. RehearsalBench: Can AI Predict What Happens When You Change a Company?

**arXiv ID:** 2611.22480 | [PDF](https://arxiv.org/pdf/2611.22480v1)

**作者:** Mika Siddiqui `[一作]` (Afterflow Inc.), Tharaka Perera `[通讯]` (Independent contributor)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个基准，评估大型语言模型（LLMs）在模拟银行联络中心的操作决策和对反事实效果的预测能力。

**💡 创新点**

这是首个同时评估LLMs的操作决策和预测反事实效果的基准，并且验证了预测排名与真实组织变更的关联性。

**🔧 技术方法**

使用了基于历史通话记录构建和验证的银行联络中心模拟器，结合了对AI干预效果的明确假设。

**📊 数据集**

使用了来自两个银行的历史通话记录（银行A和银行B），并生成了14个虚构银行的测试集。

**📈 对比分析**

通过与预算有限的优化搜索进行比较，发现最强的模型在多达一半的运行中找到的计划优于优化搜索，且其预测远超无效基线。模型普遍正确预测了效果的方向，但高估了效果的大小，尤其是AI语音代理产生的重复联系，模型高估约五倍。

**⚠️ 局限性**

模型在预测效果的大小上存在普遍的高估问题，尤其是在AI干预的效果上，且规划和预测能力并不总是相符。

---

## 527. CrisisBench: an evaluation suite for crisis counseling AI

**arXiv ID:** 2611.21622 | [PDF](https://arxiv.org/pdf/2611.21622v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 528. Making Future-Predictive Representations Actionable

**arXiv ID:** 2611.22994 | [PDF](https://arxiv.org/pdf/2611.22994v1)

**作者:** Fangchuan Wang `[一作]` (Xinjiang University), Yuqing Gao `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的模型RelayWAM，旨在解决现有未来监督世界动作模型（WAMs）在推理时的表现不佳问题，特别是动作生成与未来预测之间的目标不匹配。

**💡 创新点**

创新点在于引入了一个显式的过渡空间，使得未来信息能够有效地传递到动作生成路径中，从而提高了模型在不同分布下的泛化能力。

**🔧 技术方法**

使用了层次化的过渡层（transition layers）和线性探测（linear probing）技术来分析和提取视觉特征中的预测线索，并将其注入到动作注意力机制中。

**📊 数据集**

在LIBERO-Plus数据集上进行了实验，该数据集包含多种扰动类型，用于评估模型在不同环境下的表现。

**📈 对比分析**

与Fast-WAM和OpenWAM-α等方法进行了比较，RelayWAM在LIBERO-Plus上达到了72.03%的成功率，而Fast-WAM的成功率为51.50%，显示出显著的性能提升。

**⚠️ 局限性**

限制在于模型的复杂性和额外的参数引入，尽管增加的参数量相对较小，但仍可能影响模型的训练和推理效率。

---

## 529. Constructions of reversible linear and DNA codes through $T_{D,g}^n-$set and $(n_1,c_1|n_2,c_2)-$right polycyclic codes

**arXiv ID:** 2611.22587 | [PDF](https://arxiv.org/pdf/2611.22587v1)

**作者:** Avanish Kumar Chaturvedi `[一作]` (University of Allahabad), Satyadeep Pandey `[通讯]` (University of Allahabad)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一种基于T_D,g^n集合的可逆线性码和可逆DNA码的构造技术，并给出了相应的码参数。这种技术显著改善了Oztas等人提出的T^n集合构造的码参数。

**💡 创新点**

创新点在于引入了(n_1,c_1|n_2,c_2)-右多环码的新概念，并开发了一种新的构造技术，以获得具有良好参数的可逆线性(n_1,c_1|n_2,c_2)-右多环码。

**🔧 技术方法**

使用了T_D,g^n集合构造技术、Hadamard乘积和矩阵的连接技术。

**📊 数据集**

使用了有限交换环上的多种多项式和SageMath计算生成的码。

**📈 对比分析**

与现有方法相比，本文的方法在生成的可逆线性码中获得了更好的码参数，包括MDS码。性能通过生成的码参数进行比较，显示出显著的改进。

**⚠️ 局限性**

限制在于构造的可逆码可能不满足所有期望的代数性质，且在某些情况下，生成的码可能不具备所需的最小距离。

---

## 530. TempQuant: Learning Adaptive Temperatures for Quantized Language Models

**arXiv ID:** 2611.22128 | [PDF](https://arxiv.org/pdf/2611.22128v1)

**作者:** Mukund Athmaram `[一作]` (Advanced Micro Devices), Manoj Rao `[通讯]` (Advanced Micro Devices)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种名为TempQuant的方法，用于学习量化语言模型的自适应温度，以改善量化对模型性能的影响。

**💡 创新点**

创新点在于通过学习量化模型的温度和top-p设置，来修正量化引起的下一个token概率分布的扭曲，从而提高模型的准确性。

**🔧 技术方法**

使用了温度缩放和top-p过滤技术，结合Kullback-Leibler散度来量化全精度模型与量化模型之间的差异。

**📊 数据集**

使用了Qwen3-4B模型的3位和4位量化版本进行实验，评估了不同温度设置下的模型性能。

**📈 对比分析**

与传统的标签调优方法相比，TempQuant在没有标签的情况下，通过全精度模型的概率分布学习量化模型的最佳设置，显著提高了模型的准确性。例如，在3位Qwen3-4B上，TempQuant将MATH准确率从34%提升至51%。

**⚠️ 局限性**

限制在于温度和top-p无法重新排序tokens，因此仍然存在一些排名翻转的错误。此外，准确性提升主要在量化模型遇到重复时表现明显，且当前的每步控制器未能完全保留所有测量的行为。

---

## 531. SPICE: Sparse Parameter Gossip for LLM Pretraining

**arXiv ID:** 2611.23053 | [PDF](https://arxiv.org/pdf/2611.23053v1)

**作者:** Pietro Cagnasso `[一作]` (Concordia University), Eugene Belilovsky `[通讯]` (Concordia University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种稀疏参数传播的去中心化训练方法，旨在提高在带宽受限的环境下大规模语言模型的预训练效率。

**💡 创新点**

创新点在于结合稀疏参数传播与错误反馈机制，使得在更广泛的邻居中进行有效的通信，同时减少通信量和提高训练质量。

**🔧 技术方法**

使用了稀疏参数传播、错误反馈和外部Nesterov动量优化器等技术。

**📊 数据集**

使用了多个数据集，包括134M参数模型训练的2.70B tokens和551M参数模型训练的11.01B tokens。

**📈 对比分析**

与密集去中心化基线方法相比，提出的方法在验证损失上表现更好，同时传输的数据量更少。在带宽限制下，训练吞吐量高于DDP和密集传播，并在瞬时链路减速期间保持吞吐量。

**⚠️ 局限性**

限制在于该方法在不同模型之间的适用性可能受到影响，尤其是在模型参数差异较大的情况下。

---

## 532. Strong Transitivity of Group Cellular Automata: Characterization and Decidability

**arXiv ID:** 2611.22365 | [PDF](https://arxiv.org/pdf/2611.22365v1)

**作者:** Niccolò Castronuovo `[一作]` (University of Bologna), Luciano Margara `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究了群细胞自动机的强传递性，证明了非阿贝尔群的群细胞自动机不具备强传递性，并为阿贝尔群的群细胞自动机开发了强传递性的分解方法。

**💡 创新点**

提出了一种通过幂子群的分解来保持强传递性的结构化方法，并且为线性细胞自动机提供了强传递性的判定标准。

**🔧 技术方法**

使用了庞特里亚金对偶性和代数特征多项式的性质来表征强传递性，并开发了显式的决策程序。

**📊 数据集**

研究了群细胞自动机在每个维度上的强传递性，特别是针对阿贝尔群的细胞自动机。

**📈 对比分析**

通过与已有的决策程序进行比较，提出的程序在群大小N、邻居数量k和邻域半径ϱ的多项式时间内运行，且不需要搜索任意的言语子群。

**⚠️ 局限性**

限制在于对于非阿贝尔群的群细胞自动机，无法实现强传递性，且在处理特征多项式的因式分解时可能存在计算复杂性。

---

## 533. OVET-Consensus: Rank-Preserving Counterfactual Event Verification for Video Moment Grounding

**arXiv ID:** 2611.22637 | [PDF](https://arxiv.org/pdf/2611.22637v1)

**作者:** Alibay Osmanli `[一作]` (Queen Mary University of London), Shaogang Gong `[通讯]` (Queen Mary University of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种视频时刻检索系统，提出了一种新的验证层来改善候选事件的选择，尤其是在已有候选集的情况下。

**💡 创新点**

创新点在于引入了一种保持排名的反事实事件验证层，该层在不改变原有排名的情况下，仅允许高置信度的排名覆盖。

**🔧 技术方法**

使用了QD-DETR派生的时间专家、Qwen3-VL事件判断、VideoLLaMA3成对时间信号和密集时间证据等技术。

**📊 数据集**

使用了QVHighlights数据集，该数据集包含721,800个训练候选实例和155,000个验证候选实例。

**📈 对比分析**

与其他方法相比，提出的方法在QVHighlights验证集上将父级感知时间专家的R1@0.5从64.71提高到65.35，尽管mAP变化不大，几乎保持不变。

**⚠️ 局限性**

限制在于该方法依赖于父级感知时间专家，如果候选池缺乏有用的替代方案，验证无法恢复事件。

---

## 534. Retrieval-Augmented Correction for Generalist Medical Image Segmentation

**arXiv ID:** 2611.23189 | [PDF](https://arxiv.org/pdf/2611.23189v1)

**作者:** Vincenzo Marcianò `[一作]` (EURECOM), Maria A Zuluaga `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种检索增强的分割修正框架，通过共享的知识库对输入掩膜进行编辑，旨在解决医学图像分割中的结构错误。

**💡 创新点**

创新点在于将检索条件的修正与解剖学无关的修正器结合，使得单一修正器能够跨越不同的器官和成像模态进行操作，而无需为每个新器官或成像模态重新训练模型。

**🔧 技术方法**

使用了检索增强生成（RAG）的方法，结合了多模态编码器和形状感知检索投影器。

**📊 数据集**

使用了18个公共数据集，涵盖CT、MRI和超声成像，包括三个用于数据集转移的保留数据集。

**📈 对比分析**

与多种基线方法进行了比较，结果显示在弱分割器上，输入的DSC为0.386，修正后增加了0.35；在强分割器上，输入的DSC为0.932，变化较小。整体性能在不同的成像模态和数据集上表现出色。

**⚠️ 局限性**

局限性在于检索的有效性依赖于知识库的多样性，未在知识库中表示的解剖结构或病理表现可能导致较弱的检索上下文，此外，转移到未见的模态（如超声）时效果不佳。

---

## 535. A Temporal Algebraic Framework for Data Stream Evaluation Under Delayed Labels and Distribution Shifts

**arXiv ID:** 2611.21914 | [PDF](https://arxiv.org/pdf/2611.21914v1)

**作者:** Ronaldo C. Prati `[一作]` (Universidade Federal do ABC), Elaine Ribeiro de Faria `[通讯]` (Universidade Federal de Uberlândia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种时间代数框架，用于在标签延迟和分布变化的情况下评估数据流。

**💡 创新点**

创新点在于扩展了代数框架，使其能够处理延迟标签和选择性反馈，同时能够区分先前的变化和概念漂移。

**🔧 技术方法**

使用了时间加权的混淆矩阵更新算法，采用了指数衰减和滑动窗口的形式，更新复杂度为O(m^2)。

**📊 数据集**

使用了合成数据流和四个真实数据集进行实验验证。

**📈 对比分析**

与传统的漂移检测方法（如ADWIN和DDM）相比，提出的方法在标签变化下没有产生过多的误报，并且在某些真实场景中恢复了召回率。

**⚠️ 局限性**

限制在于对于精确度、特异性和F_1的恢复仅为部分，且在极端不平衡的情况下，稀有类的评估仍然存在挑战。

---

## 536. Differentiable High-Order Mesh $r$-Adaptivity

**arXiv ID:** 2611.22099 | [PDF](https://arxiv.org/pdf/2611.22099v1)

**作者:** Ketan Mittal `[一作]` (Lawrence Livermore National Laboratory), Vladimir Tomov `[通讯]` (Lawrence Livermore National Laboratory)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本研究开发了一个基于目标矩阵优化范式的高阶网格自适应框架，能够自动生成点对点的一阶和二阶导数，以支持高效的网格优化。

**💡 创新点**

创新点在于通过自动微分技术简化了高阶网格自适应的导数计算过程，提升了收敛性，并支持GPU加速执行。

**🔧 技术方法**

使用了自动微分技术（如Enzyme）和有限元操作分解方法来计算导数，并实现了矩阵自由的Hessian操作。

**📊 数据集**

使用了Kershaw基准问题和其他数值实验数据集来验证所提出框架的有效性和性能。

**📈 对比分析**

与经典的手动导数实现相比，所提出的方法在GPU上解决时间相当，并且由于准确计算导数，r自适应收敛性得到了改善。

**⚠️ 局限性**

限制在于当前框架主要集中在自动微分的实现上，未来工作将探索在网格拓扑变化的情况下计算一致灵敏度的方法。

---

## 537. RetiDual-Select: Reliability-Aware Specialist-Generalist Routing for Retinal OCT under Image-Quality Shift

**arXiv ID:** 2611.22117 | [PDF](https://arxiv.org/pdf/2611.22117v1)

**作者:** Zihao Han `[一作]` (University of Kent), Philippe De Wilde `[通讯]` (University of Kent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究探讨了自适应融合在修复静态集成错误方面的有效性，同时也可能导致正确决策的推翻。研究结合了视网膜专家模型与医学视觉-语言模型的图像编码器，分离了候选融合、路由安全性和预测审查。

**💡 创新点**

创新点在于提出了一种决策架构，明确区分候选融合、路由安全性和预测审查，并保留静态PoE作为保守参考，同时引入了经过校准的专家-通用框架和验证冻结的危害保护机制。

**🔧 技术方法**

使用了光学相干断层扫描（OCT）技术，结合了基础模型、自适应融合和受控图像质量降级的技术。

**📊 数据集**

使用了多个公开的OCT数据集，包括Kermany-4、OCT-C8和OCTDL，分别用于训练、验证和测试。

**📈 对比分析**

与静态PoE方法相比，静态PoE在所有三个数据集上达到了最高或并列最高的准确率。在清洁的2%接受错误目标下，审查策略接受了94.3%的图像，选择性准确率为99.89%。自适应路由在严重对比度降低的情况下表现出显著的增益。

**⚠️ 局限性**

本研究的局限性包括仅使用了合成扰动进行评估，未能证明在自然获取或扫描仪变化下的鲁棒性。此外，患者独立性无法在大多数情况下建立，且未评估其他OCT能力编码器的效果。

---

## 538. Learning Grounded Reasoning from Human Gaze

**arXiv ID:** 2611.23209 | [PDF](https://arxiv.org/pdf/2611.23209v1)

**作者:** Gabriel Sarch `[一作]` (Princeton University), Taylor W. Webb `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究探讨了如何有效利用人类注视数据来改善视觉语言模型（VLMs）中的视觉推理，同时保持与人类行为的一致性。研究收集了7109个来自多样化视觉推理任务的人类注视试验，并训练了不同形式的注视指导的推理模型。

**💡 创新点**

创新点在于提出了Gaze-Guided Reasoner（GGR），该模型在准确性上与领先的基线模型相匹配或超越，同时实现了更高的人类对齐和更可靠的视觉基础。研究表明，选择性使用注视比完全监督每个注视点更有效。

**🔧 技术方法**

使用了强化学习（RL）和注视指导的监督学习技术，结合了人类注视数据来训练视觉推理模型。

**📊 数据集**

使用的数据集包括7109个眼动追踪试验，涵盖了图表理解、计数和视觉搜索等任务。

**📈 对比分析**

与现有方法（如ViGoRL-MCTS）进行比较，GGR在任务准确性和注视对齐方面表现更好。经过强化学习后，GGR的整体准确性从65.3%提高到68.8%，注视对齐从0.84提高到1.15。

**⚠️ 局限性**

限制在于人类注视数据可能包含噪声和冗余，完全模仿人类行为可能不会改善模型的推理能力。此外，研究未完全探讨不同预训练数据或架构如何影响注视监督的有效性。

---

## 539. RyBench: Runtime Verification Benchmarking Framework over Publish/Subscribe Networks

**arXiv ID:** 2611.22401 | [PDF](https://arxiv.org/pdf/2611.22401v1)

**作者:** Mehmet Hilmi Dündar `[一作]` (Bogazici University), Doğan Ulus `[通讯]` (Bogazici University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RyBench，一个基于发布/订阅网络的运行时验证工具基准测试框架，旨在评估这些工具的性能。

**💡 创新点**

RyBench的创新点在于其模块化设计，能够独立处理事件生成、序列化和传输，支持在线和离线生成模式，并能够灵活配置不同的消息模式和传输机制。

**🔧 技术方法**

使用C++23实现，采用了多种序列化格式（如JSON、FlatBuffers和Protocol Buffers）和传输机制（如iceoryx2和Zenoh）。

**📊 数据集**

使用Open Simulation Interface (OSI) schemas和Timescales属性生成事件，支持多种数据格式。

**📈 对比分析**

通过与Reelay和MonPoly的集成，评估了RyBench的吞吐量和延迟性能，结果显示RyBench能够维持每秒数百万事件的生成速率，且不同传输机制对延迟有显著影响。

**⚠️ 局限性**

RyBench的局限性在于当前评估主要集中在单主机的一对一拓扑，未来需要扩展到更复杂的多主机和多节点配置，以评估在不同通信复杂性下的性能表现。

---

## 540. CRA: Continuous Risk Assessment for Safety-critical Visual Navigation

**arXiv ID:** 2611.22080 | [PDF](https://arxiv.org/pdf/2611.22080v1)

**作者:** Lihao Zheng `[一作]` (Northwestern University), Randy Freeman `[通讯]` (Northwestern University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于视觉的连续风险评估框架，用于安全关键的视觉导航，利用RGB和LiDAR观测来估计四种风险因素，并将其结合成一个连续的风险评分。

**💡 创新点**

创新点在于通过四个物理因素（动能、势能、接触严重性和热风险）进行连续风险评估，而不是使用预定义的安全或不安全标签，从而实现更灵活的安全过滤。

**🔧 技术方法**

使用RGB和LiDAR观测，结合控制障碍函数（CBF）和模型预测控制（MPC）技术进行实时安全过滤。

**📊 数据集**

在NVIDIA Isaac Sim中进行模拟实验，使用餐厅场景进行验证。

**📈 对比分析**

与传统的几何安全过滤方法相比，提出的方法在复杂环境中表现出更好的实时安全监控和干预能力，能够处理动态障碍物和不确定性。

**⚠️ 局限性**

限制在于当前方法主要在模拟环境中验证，未来需要在真实环境中进行测试，以评估其在更复杂社交互动场景中的表现。

---

## 541. Fiberpose: state-of-the-art fiber morphometry and instance segmentation with a distance-transform neural network

**arXiv ID:** 2611.22953 | [PDF](https://arxiv.org/pdf/2611.22953v1)

**作者:** Viktor A. Balashov `[一作]`, Daniil S. Antonenko `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Fiberpose的框架，通过直接回归平滑的距离变换（DT）场来分析纤维结构，结合了一套新的几何算法以适应这种表示。

**💡 创新点**

创新点在于将纤维分析重新构建为DT场的直接回归，而不是依赖传统的二值化方法，从而提高了在弱边界和密集纤维区域的分析准确性。

**🔧 技术方法**

使用了深度学习技术，具体为卷积编码器-解码器网络（FPUNet），用于预测DT场。

**📊 数据集**

使用了多种扫描电子显微镜（SEM）图像的数据集进行训练和验证。

**📈 对比分析**

与现有方法（如SAM2和Cellpose-SAM）相比，Fiberpose在实例分割准确性上达到了最先进的水平，并且在直径测量上比DiameterJ、Fibar和GIFT更为准确。

**⚠️ 局限性**

限制在于对于非常细的纤维，DT场的平滑表示可能不够准确，且当前方法主要集中在二维图像上，未来需要扩展到三维数据。

---

## 542. Does the Model Use What It Cites? Counterfactual Tests of Grounded Reasoning Chains in Small Video-Language Models

**arXiv ID:** 2611.21940 | [PDF](https://arxiv.org/pdf/2611.21940v1)

**作者:** Jugal Gajjar `[一作]` (Independent Researchers), Kamalasankari Subramaniakuppusamy `[通讯]` (Independent Researchers)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

论文提出了一种反事实自引测试，用于验证视频语言模型的基础推理链的有效性，特别是检查模型的答案是否依赖于其引用的区域。

**💡 创新点**

创新点在于提出了两种无注释的测试方法：反事实自引测试和实体跟踪测试，以评估模型的推理链的有效性和答案的依赖性。

**🔧 技术方法**

使用了小型视频语言模型（VLMs），如Qwen2.5-VL-3B和InternVL3-2B，结合强化学习和无监督学习的方法进行训练和评估。

**📊 数据集**

使用了CLEVRER和NExT-QA数据集，CLEVRER提供了可执行程序和对象注释，而NExT-QA则是没有空间证据注释的真实视频因果基准。

**📈 对比分析**

通过对比引用的区域和随机区域的遮蔽效果，发现CLEVRER模型在遮蔽引用框时正确答案翻转的比例是控制组的2.2倍，而NExT-QA模型的翻转比例相对较小，显示出引用框对答案的依赖性较低。

**⚠️ 局限性**

限制在于所有训练均在2-3B模型上进行，且NExT-QA结果仅基于一个基准和一个模型，伪标签链存在噪声，且未对桥接损失进行消融实验。

---

## 543. Urban Representation Learning for Identifying Potential Disparities in Crime Reporting

**arXiv ID:** 2611.21825 | [PDF](https://arxiv.org/pdf/2611.21825v1)

**作者:** Jorge Valverde-Rebaza `[一作]` (Tecnologico de Monterrey), Eashley Martínez-Vergara `[通讯]` (Tecnologico de Monterrey)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种自监督的城市表示学习框架，用于识别与相似区域相比的异常报告模式，仅使用地理参考的犯罪记录。

**💡 创新点**

创新点在于引入了邻居配置剖面差异（NPD）作为量化区域报告剖面与其相似邻居剖面之间不一致性的度量，并且该框架不依赖于外部数据。

**🔧 技术方法**

使用了自监督学习技术，构建了小时、每周和静态报告剖面，并通过跨尺度注意力将其整合到统一的潜在空间中。

**📊 数据集**

使用了来自墨西哥城的超过180万条犯罪记录的数据集，这些记录被聚合到1058个H3区域中。

**📈 对比分析**

与Hex2Vec等基线方法相比，提出的表示学习框架在聚类质量上表现更好，平均轮廓系数为0.4052，而Hex2Vec为0.2863。NPD排名在不同邻域大小下高度稳定，斯皮尔曼相关系数在0.9720到0.9951之间。

**⚠️ 局限性**

限制在于NPD不能直接推断未报告的犯罪情况，可能存在地方异质性、行政边界、警务实践等其他未观察到的因素影响报告模式。

---

## 544. Bookkeeping Is Not Judgement: EJBench, an Engineering Judgement Benchmark Built from a Conjugate Heat Transfer Design Campaign

**arXiv ID:** 2611.21707 | [PDF](https://arxiv.org/pdf/2611.21707v1)

**作者:** James Sadler `[一作]` (Gradient Dynamics), Kotub Uddin `[通讯]` (AESC)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

介绍了EJBench，一个工程判断的基准，旨在评估模型在新条件下预测设计行为的能力，基于少量早期结果做出决策。

**💡 创新点**

EJBench是软件无关的，关注工程师在不运行模拟的情况下得出的结论，填补了现有工程基准的空白。

**🔧 技术方法**

使用了对流换热模拟和OpenFOAM进行验证的技术，确保每个答案来自收敛的物理模拟。

**📊 数据集**

数据集来源于数千个电池模块和冷却板设计的模拟，涵盖多种操作条件和几何布局。

**📈 对比分析**

与其他方法相比，EJBench的最佳模型GPT-6 Astra得分为0.71，显示出模型在任务上的一致性和信心，但仍有24个任务未被任何模型解决。

**⚠️ 局限性**

限制在于答案来自单一求解器，未与实验数据进行对比，且评分为全或无，可能隐藏接近正确答案的情况。

---

## 545. SpatialMind: Adaptive Multi-Level Geometric Fusion for Spatial Reasoning in Multimodal Large Language Models

**arXiv ID:** 2611.22415 | [PDF](https://arxiv.org/pdf/2611.22415v1)

**作者:** Shibin Zhao `[一作]` (University of Chinese Academy of Sciences), Jinqiao Wang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为SpatialMind的任务自适应框架，用于在多模态大语言模型中进行多层几何特征融合，以提高空间推理能力。

**💡 创新点**

创新点在于通过GeoFus模块动态调整几何特征的贡献，适应不同空间任务的需求，解决了固定几何融合策略无法满足多样化空间推理要求的问题。

**🔧 技术方法**

使用了GeoFus，一个基于门控网络的多层几何融合模块，结合文本指令、视觉语义和几何表示进行动态调节。

**📊 数据集**

使用了多个数据集，包括VSI-Bench、SPAR和LLaVA-Hound等，涵盖了多种空间推理任务。

**📈 对比分析**

与其他方法相比，SpatialMind在VSI-Bench上取得了71.5和72.3的最新状态分数，显示出其在空间推理基准测试中的竞争力。

**⚠️ 局限性**

限制在于模型的训练依赖于多样化的数据集，可能在特定任务或数据集上表现不如预期，且需要进一步优化以提高稳定性和泛化能力。

---

## 546. Think Once, Act Fast: Cost-Effective Recursive Self-Improvement for Decision Model Harness

**arXiv ID:** 2611.22432 | [PDF](https://arxiv.org/pdf/2611.22432v1)

**作者:** Siqi Lai `[一作]` (Hong Kong University of Science and Technology), Hao Liu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为JevRSI的系统-1/系统-2协作框架，通过递归自我改进实现低延迟决策。

**💡 创新点**

JevRSI是首个在执行过程中实现决策模型自我改进的框架，支持高效且可泛化的决策。

**🔧 技术方法**

使用了Jev风格的决策模型和大型语言模型（LLM）进行模块化的决策支持。

**📊 数据集**

在ALFWorld、BabyAI和TextCraft三个数据集上进行了实验。

**📈 对比分析**

与传统决策模型、系统-1/2合作代理和LLM代理进行了比较，JevRSI在多个设置中决策质量上匹配或超越了LLM代理，同时保持低延迟执行。

**⚠️ 局限性**

当前的研究未在真实世界环境中测试JevRSI的鲁棒性，未来需要扩展到更复杂的环境中。

---

## 547. Exploring Students' Insights on Creating Videos to Learn Software Design and Engineering

**arXiv ID:** 2611.23075 | [PDF](https://arxiv.org/pdf/2611.23075v1)

**作者:** Pedro G. Feijóo-García `[一作]` (Georgia Institute of Technology), Olufisayo Omojokun `[通讯]` (Georgia Institute of Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了学生创建教学视频作为大型本科软件设计与工程课程中的一种主动学习策略，旨在提高学生的参与度和理解力。

**💡 创新点**

创新点在于采用“通过教学学习”的原则，让学生以自己的语言创建教学视频，从而增强他们的学习自主性和理解能力。

**🔧 技术方法**

使用了学生创建的教学视频作为评估形式，结合了定性分析方法来评估学生的反馈和体验。

**📊 数据集**

数据集来自于超过700名学生的开放式反馈，分析了他们对视频创建策略的看法和体验。

**📈 对比分析**

与传统的多项选择考试相比，学生普遍认为视频创建过程更具灵活性和参与感，但也反映出视频制作耗时较长。

**⚠️ 局限性**

限制在于视频创建过程需要大量时间和精力，部分学生对评估标准和内容的清晰度表示担忧，且有些学生认为视频制作的学习效果不如预期。

---

## 548. FIRE-Drive: Feasibility-Informed Risk Evaluation for End-to-End Diffusion Planning

**arXiv ID:** 2611.22297 | [PDF](https://arxiv.org/pdf/2611.22297v1)

**作者:** Qifeng Xie `[一作]` (Xidian University), Kai Sheng `[通讯]` (Xidian University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为FIRE-Drive的框架，用于端到端的自主驾驶规划，旨在通过动态可行锚点初始化、候选特定的风险评估和紧凑的高质量候选监督来改善基于扩散的规划。

**💡 创新点**

创新点在于引入了可行性信息的风险评估框架，结合了场景适应的锚点初始化和风险意识评分，使得生成的轨迹不仅多样化且可行。

**🔧 技术方法**

使用了扩散模型技术，结合了动态可行锚点初始化、候选条件的鸟瞰图风险证据和紧凑的候选监督。

**📊 数据集**

在NAVSIM v1和220条Bench2Drive路线的数据集上进行了实验。

**📈 对比分析**

与其他方法相比，FIRE-Drive在NAVSIM v1上获得了91.5的预测驾驶模型分数，在Bench2Drive上获得了71.43的驾驶分数和42.73%的成功率，显示出其在规划性能上的竞争力。

**⚠️ 局限性**

限制在于当前模型仅使用单帧输入，限制了时间建模的能力，同时辅助参考池仅覆盖了一部分可能的行为，特别是在高度互动的情况下。

---

## 549. Layerwise Learning-Rate Scaling in Vision Networks: Effective-Rate Identification, Power-Mean Aggregation, and Budget Dependence

**arXiv ID:** 2611.21696 | [PDF](https://arxiv.org/pdf/2611.21696v1)

**作者:** Chandini Vysyaraju `[一作]` (University of Wurzburg), Dmitry Ignatov `[通讯]` (University of Wurzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一种层级有效学习率的识别方法，探讨了在从头训练的视觉网络中，层级学习率缩放的效果及其与训练预算的关系。

**💡 创新点**

创新点在于提出了一种基于乘数配置直接计算的有效学习率的总结方法，并通过实验证明了该方法在不同架构中的有效性。

**🔧 技术方法**

使用了层级学习率缩放、参数加权幂均值等统计技术，并进行了多种实验设计以验证假设。

**📊 数据集**

使用了CIFAR-10、CIFAR-100和ImageNette等数据集进行实验。

**📈 对比分析**

通过与均匀控制组的比较，发现层级学习率策略在多个架构中表现优于基线，且有效学习率的总结能够显著提高候选策略的选择准确性。

**⚠️ 局限性**

限制在于模型是针对特定策略空间的经验响应模型，未能涵盖所有可能的层级优化理论，且在不同架构间的迁移效果尚需进一步验证。

---

## 550. CULypso: A Clustered Federated Unlearning paradigm for Vehicular Ad-Hoc Networks

**arXiv ID:** 2611.22682 | [PDF](https://arxiv.org/pdf/2611.22682v1)

**作者:** Eleftheria Chini `[一作]`, Dimitrios Katsaros `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出了一种名为CULypso的分层联合遗忘算法，旨在解决车载自组网中的数据遗忘问题，以满足数据保护法规的要求。

**💡 创新点**

CULypso是首个针对聚类车载自组网的近似分层遗忘算法，利用车辆的移动性和数据异构性来降低通信、资源和时间成本。

**🔧 技术方法**

使用了分层联合学习和近似遗忘技术，结合了动态、移动和数据感知的车辆聚类方法。

**📊 数据集**

通过广泛的模拟实验评估CULypso的性能，比较了其与FedEraser和完全重训练的效果。

**📈 对比分析**

CULypso在时间、服务器存储负载和遗忘保真度方面比FedEraser提高了约65%，并且在性能上远超基线的完全重训练方法。

**⚠️ 局限性**

该研究的局限性在于尚未在真实的聚类拓扑上进行实验，且对高度移动和异构的车载自组网的挑战仍需进一步探索。

---

## 551. High-Dimensional Black-Box Optimization via Guidance Sampling of Diffusion Models

**arXiv ID:** 2611.22965 | [PDF](https://arxiv.org/pdf/2611.22965v1)

**作者:** Xiaoyan Hu `[一作]` (Carnegie Mellon University), Jeff Schneider `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了一种名为BReaD的框架，用于在高维数据空间中优化黑箱奖励函数，结合了预训练的扩散模型和高斯过程。

**💡 创新点**

创新点在于不依赖于标记数据集或奖励梯度，通过高斯过程作为未知目标的可微替代品，主动查询生成样本的标签。

**🔧 技术方法**

使用了高斯过程（GP）和扩散模型，结合了奖励导向采样和不确定性导向采样的机制。

**📊 数据集**

在科学发现和图像生成任务中进行了数值实验，使用了有限的标记样本进行优化。

**📈 对比分析**

与传统的基于梯度的方法相比，BReaD在低数据场景下表现出色，能够有效优化黑箱奖励，性能与访问确切奖励梯度的方法相当。

**⚠️ 局限性**

限制在于该方法在高维空间中的应用可能仍然面临挑战，尤其是在模型的准确性和样本效率方面。

---

## 552. Mechanism Design and Active Systems Theory: 60 Years of Parallel Development

**arXiv ID:** 2611.21803 | [PDF](https://arxiv.org/pdf/2611.21803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 553. Can an LLM dream? Testing the overfitted brain hypothesis on an agent with text memory

**arXiv ID:** 2611.21648 | [PDF](https://arxiv.org/pdf/2611.21648v1)

**作者:** Eric Zou `[一作]` `[通讯]` (University of Waterloo), Eric Zou (University of Waterloo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究测试了Hoel（2021）提出的假设，即梦境通过扭曲重放经历来防止大脑对近期事件的过拟合。研究使用了一个大型语言模型（Claude Opus 5.5），该模型在每晚根据当天的对话记录重写其文本记忆，并在此过程中引入了不同程度的文本损坏。

**💡 创新点**

创新点在于建立了一个评估设置，用于评估大型语言模型在夜间记忆巩固过程中的表现，并提供了不同类型和程度的文本损坏对记忆回忆和任务转移的影响的实证结果。

**🔧 技术方法**

使用了Claude Opus 5.5模型，并通过模拟十天的对话历史进行实验，采用了文本损坏的两种方式（细节损坏和随机文本损坏）。

**📊 数据集**

使用了LongMemEval数据集，该数据集包含长时间的对话历史和相关问题。

**📈 对比分析**

与无重写和干净重写的基线进行比较，结果显示细节损坏导致回忆率显著下降，而随机文本损坏对回忆率没有一致的影响。细节损坏在40%的情况下将回忆率从76%降低到60%。

**⚠️ 局限性**

限制在于只测试了Claude Opus 5.5模型，可能存在其他模型对随机文本的处理不同。此外，记忆大小限制和历史的广泛变化可能影响结果的可靠性。

---

## 554. Sim2CBF: From Forward Simulations to Implicit Control Barrier Function Safety Filtering without Model Derivatives

**arXiv ID:** 2611.22954 | [PDF](https://arxiv.org/pdf/2611.22954v1)

**作者:** Luzia Knoedler `[一作]` (KTH Royal Institute of Technology), Jana Tumova `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出了一种名为Sim2CBF的框架，用于解决复杂机器人与环境交互中的失败避免问题，利用前向模型评估而无需对预测模型进行微分。

**💡 创新点**

创新点在于通过有限差分和局部线性插值来近似cbf约束，从而避免了对预测模型的微分，同时保持了失败避免的保证。

**🔧 技术方法**

使用了有限差分和局部线性插值技术，结合隐式cbf（rpcbf）进行优化。

**📊 数据集**

在模拟、硬件箱子倾倒和基于视觉的控制任务中进行了评估，使用了物理基础的模拟器和学习的世界模型作为预测模型。

**📈 对比分析**

与纯采样的安全过滤方法相比，Sim2CBF需要更少的控制样本，并且比直接的名义-备份混合行为更不保守，性能表现良好。

**⚠️ 局限性**

限制在于在连续控制更新之间的失败避免可能不被保证，且在某些情况下可能需要额外的假设来确保安全性。

---

## 555. Inapproximability of L1->L2 Subspace Distortion

**arXiv ID:** 2611.22531 | [PDF](https://arxiv.org/pdf/2611.22531v1)

**作者:** Vijay Bhattiprolu `[一作]` `[通讯]`, Vijay Bhattiprolu

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文展示了在假设NP不包含随机准多项式时间的情况下，针对最大化ℓ_2/ℓ_1比率的几乎多项式因子不可近似性，特别是在实子空间上的子空间失真问题。

**💡 创新点**

创新点在于构造了一个随机仿射线性映射T，使得Tx的失真度量接近于超立方体{±1}^n的接近度，从而为理解小集扩展和量子可分离性提供了中间步骤。

**🔧 技术方法**

使用了随机仿射线性映射技术，构造了一个近似等距的映射，并在高维几何中进行了分析。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了与压缩感知、错误更正码、字典学习等多个领域的联系。

**📈 对比分析**

与现有方法的比较表明，本文的结果在最坏情况下的近似比率为n^Ω(1)，而在平均情况下的近似比率也未能显著改善，表明该问题的复杂性。

**⚠️ 局限性**

限制在于当前结果依赖于假设NP问题的复杂性，且在处理稀疏解时，局部测试框架的有效性受到挑战。

---

## 556. Motion-Aware Reward Modeling for Instruction-Guided Video Editing

**arXiv ID:** 2611.22495 | [PDF](https://arxiv.org/pdf/2611.22495v1)

**作者:** Li Li `[一作]` (University of Southern California), Yue Zhao `[通讯]` (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了指导视频编辑的奖励模型在时间一致性与运动保留之间的平衡问题，提出了一种运动感知奖励模型，以解决运动崩溃的问题。

**💡 创新点**

创新点在于提出了一种结合RGB帧差流和偏好训练的运动感知奖励模型，能够有效区分运动抑制和伪造运动，并保持与人类质量判断的一致性。

**🔧 技术方法**

使用了运动感知奖励模型，该模型结合了显式的帧间动态流、运动抑制和伪造运动负样本，以及偏好保留的蒸馏。

**📊 数据集**

使用了VE-Bench数据集，该数据集包含人类评分的编辑视频，并且与训练数据严格分离。

**📈 对比分析**

与现有方法相比，提出的运动感知奖励模型在VE-Bench运动梯度上选择原始编辑的比例为100%，而其他模型在运动抑制情况下的选择率高达97%。该模型与人类质量评分的Spearman相关性为0.57，表现优于其他模型。

**⚠️ 局限性**

限制在于模型的训练依赖于特定的负样本构造，可能在面对未见的运动抑制模式时表现不佳。

---

## 557. TIBECG: Temporal Information Bottleneck for Lead-Specific Demographic Privacy in ECG Waveform Release

**arXiv ID:** 2611.21694 | [PDF](https://arxiv.org/pdf/2611.21694v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 558. Evaluating Semantic Insensitivity in Financial Sentiment Analysis: Implications for Sustainable Financial Decision-Making

**arXiv ID:** 2611.21653 | [PDF](https://arxiv.org/pdf/2611.21653v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 559. ForecastGym: A Live, Controlled, and Replayable Benchmark for Forecasting Agents

**arXiv ID:** 2611.23045 | [PDF](https://arxiv.org/pdf/2611.23045v1)

**作者:** Yicheng Tao `[一作]` (Carnegie Mellon University), Zipei Fan `[通讯]` (Jilin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的基准，旨在通过控制证据的获取和重放来评估基于大型语言模型的预测代理的性能。

**💡 创新点**

创新点在于引入了一个可重放的基准，能够在已知结果后离线重放固定的证据状态，从而比较不同模型、记忆策略和搜索源的表现。

**🔧 技术方法**

使用了四个商业网络搜索API进行证据的实时捕获，并通过控制重放来评估预测代理的性能。

**📊 数据集**

数据集包含150个问题、614个证据状态和131,000个搜索结果，覆盖多个领域的真实预测市场问题。

**📈 对比分析**

与市场的比较显示，所有测试的LLM在预测准确性上均落后于市场，且不同搜索API的切换对模型的Brier分数影响显著，超过模型间的差异。

**⚠️ 局限性**

限制在于当前的基准仍然依赖于已知的结果，可能无法完全模拟真实世界中的动态变化和信息获取过程。

---

## 560. MI-Hyper: Generating Multiple Instance Models for Weight Space Transfer Learning in Histopathology via Mixed Tasks Pretrained HyperNetwork

**arXiv ID:** 2611.22968 | [PDF](https://arxiv.org/pdf/2611.22968v1)

**作者:** Xitong Ling `[一作]` (Shenzhen International Graduate School Tsinghua University), Yonghong He `[通讯]` (Shenzhen International Graduate School Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为MI-Hyper的超网络，通过任务描述生成多实例学习（MIL）模型的聚合和分类权重，从而实现零-shot全幻灯片图像（WSI）预测。

**💡 创新点**

MI-Hyper通过条件权重生成机制，学习从任务描述到预测器的映射，提供了一种新的权重空间转移方法，特别适用于长序列建模。

**🔧 技术方法**

使用了超网络（hypernetwork）技术，结合了多实例学习（MIL）模型的聚合和分类机制。

**📊 数据集**

在52个病理数据集上的308个任务中进行了评估，涉及多种病理任务和数据。

**📈 对比分析**

与MI-Zero方法相比，MI-Hyper在平均AUC、平衡准确率和宏F1分数上分别提高了1.85、1.17和0.57个百分点，尤其在某些分子预测任务中，AUC和准确率提高超过20个百分点。

**⚠️ 局限性**

限制在于需要更广泛的源任务覆盖和数据增强的研究，以提高权重生成的稳定性，从而实现更可靠的预测。

---

## 561. Outcome-Balanced PPO and Policy-Induced Flow Rectification for Flow-Based VLA Policies

**arXiv ID:** 2611.22878 | [PDF](https://arxiv.org/pdf/2611.22878v1)

**作者:** Yuhan Wang `[一作]` (Tsinghua University), Jiaguan Zhu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种结合结果平衡近端策略优化（Outcome-Balanced PPO）和策略诱导流重构的强化学习框架，以优化基于流的视觉-语言-动作（VLA）策略的任务回报。

**💡 创新点**

创新点在于通过任务成功率估计对正优势进行加权，并在保持符号的同时重新缩放正负优势，从而解决了稀疏多任务训练中学习信号不均衡的问题。

**🔧 技术方法**

使用了结果平衡近端策略优化（Outcome-Balanced PPO）和策略诱导流重构技术。

**📊 数据集**

在模拟操作基准上进行评估，并在Real2Sim2Real任务中研究了组件组合。

**📈 对比分析**

与基线方法（如π_RL和ReFPO）相比，提出的框架在模拟基准上提高了成功率，尤其是在LIBERO和ManiSkill环境中表现突出。

**⚠️ 局限性**

限制在于框架的有效性可能依赖于任务的具体特性，未来的工作需要在更广泛的真实世界任务中进行评估。

---

## 562. Learning to Undo: Transfer Reinforcement Learning under State Space Transformations

**arXiv ID:** 2611.23081 | [PDF](https://arxiv.org/pdf/2611.23081v1)

**作者:** Mridul Mahajan `[一作]` (Boston University), Xuezhou Zhang `[通讯]` (Boston University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了一种新的强化学习中的迁移学习方法，通过学习一个撤销映射来实现源任务和目标任务之间的知识转移。

**💡 创新点**

创新点在于通过匹配状态特征统计来学习撤销映射，从而有效地在目标任务中重用源任务的策略，并且在理论上证明了该方法在特定条件下具有更好的样本复杂度。

**🔧 技术方法**

使用了基于分布的撤销差异度量和中央矩差异度量（CMD）来评估撤销映射的质量，并提出了一种实用的迁移学习算法来优化这一目标。

**📊 数据集**

在实验中使用了挑战性的连续控制任务和Atari游戏作为数据集，通过对状态空间的结构化变换构建目标任务。

**📈 对比分析**

与从头学习目标策略的方法相比，学习撤销映射的方法在样本效率上显著提高，通常在样本数量减少一个数量级的情况下实现接近最优的性能。

**⚠️ 局限性**

限制在于该方法的理论分析主要针对线性撤销映射和线性-Q^⋆ 可实现的源任务，未来的研究需要扩展到更一般的任务等价性概念。

---

## 563. Visibility Graph Representations of Word Embeddings for Text Classification

**arXiv ID:** 2611.22685 | [PDF](https://arxiv.org/pdf/2611.22685v1)

**作者:** Elif Dorukbaşı `[一作]` (Karabük University), İlker Türker `[通讯]` (Karabük University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文研究了自然可见图作为新闻主题分类的中间表示，探讨了在受控词汇预算下的文本分类问题。

**💡 创新点**

创新点在于将词嵌入平均化为固定长度的文档向量，并通过自然可见图将其转换为有序节点度向量或邻接矩阵，以供分类器使用。

**🔧 技术方法**

使用了自然可见图、卡方特征选择、节点度表示和邻接矩阵等技术。

**📊 数据集**

使用了AG News基准数据集，该数据集包含四个类别的新闻，分别是世界、体育、商业和科学技术。

**📈 对比分析**

与基于直接嵌入的表示方法进行比较，使用完全连接的分类器时，节点度表示在700个选定词汇以上的配置中最为准确，最高准确率为91.24%。在1,500个词汇时，超过了基于BERT的基线0.29个百分点。

**⚠️ 局限性**

限制在于评估仅使用了单一的平衡英语基准，未涵盖不平衡类别、其他语言、情感任务或长文档，且每个配置仅运行一次，需进行重复运行以验证小差异的可靠性。

---

## 564. Logical Relations on Display

**arXiv ID:** 2611.21826 | [PDF](https://arxiv.org/pdf/2611.21826v1)

**作者:** Steven Schaefer `[一作]` (University of Michigan), Max S. New `[通讯]` (University of Michigan)

**通讯引用:** 542 | [OpenAlex ID](https://openalex.org/A5050680882)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文开发了一个可重用的范畴抽象库，用于在Cubical Agda证明助手中机械化编程语言的元理论，重点在于通过范畴理论实现编程语言语义的重用。

**💡 创新点**

创新点在于提出了一种模块化的范畴组件方法，允许用户在构建元理论证明时直接利用库中的结构，而不是从头开始重建，从而减少了重复工作。

**🔧 技术方法**

使用了显示范畴和自由范畴等范畴理论技术，结合了依赖类型理论中的构造。

**📊 数据集**

使用了Cubical Agda中的243个模块，约35,000行代码，扩展了现有的库，涵盖了简单类型演算的语义域。

**📈 对比分析**

通过与现有的元理论证明方法进行比较，展示了所提出方法的有效性，证明了多个熟悉的结果，如典型性、参数性和一致性，且性能良好。

**⚠️ 局限性**

限制在于范畴理论在形式化时的复杂性，尤其是关于对象相等性的推理可能导致性能问题，且在某些情况下，类型检查可能变得非常缓慢。

---

## 565. When Agent Skills Go Stale: Runtime Behavior and Selective Maintenance in Evolving Environments

**arXiv ID:** 2611.22525 | [PDF](https://arxiv.org/pdf/2611.22525v1)

**作者:** Ömer Faruk Akgül `[一作]` (University of Southern California), Stefano Soatto `[通讯]` (AWS Agentic AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

研究了智能体如何在环境变化后重用技能状态，并提出了选择性维护的方法来更新过时的技能状态，以提高任务性能和减少执行成本。

**💡 创新点**

创新点在于引入了选择性维护，通过回放成功执行的检查来验证和修复技能状态，而无需额外的模型调用，从而在环境变化时保持技能的推理价值。

**🔧 技术方法**

使用了大型语言模型（LLM）结合工具使用和执行管理的框架。

**📊 数据集**

使用了四个任务设置和两个模型的数据集，包括历史税表和其他相关任务。

**📈 对比分析**

与不使用技能的基线进行比较，选择性维护在任务准确性上提高了最多67个百分点，并减少了最多37%的执行步骤，显示出显著的性能提升。

**⚠️ 局限性**

限制在于研究仅覆盖了四个数据集和两个来自同一模型家族的LLM，且使用了受控的环境变化，未来需要扩展到更复杂的实际环境中。

---

## 566. Towards Physics-Constrained Rectified Flow for Climate Downscaling: What Projections Can Certify?

**arXiv ID:** 2611.22283 | [PDF](https://arxiv.org/pdf/2611.22283v1)

**作者:** Arun Sharma `[一作]` `[通讯]` (University of Minnesota), Arun Sharma (University of Minnesota)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种物理约束的修正流模型（PC-RF），用于气候下尺度化，结合了风散度、负降水和质量漂移的训练惩罚，并在推断时通过投影确保非负性和质量平衡。

**💡 创新点**

创新点在于将物理约束与修正流模型结合，通过训练惩罚和推断投影来确保生成场的物理有效性，特别是在非负性和质量守恒方面。

**🔧 技术方法**

使用了修正流（Rectified Flow）和深度生成模型，结合了物理约束的训练惩罚和推断投影。

**📊 数据集**

使用了ERA5再分析数据集，包含总降水和10米的风速数据，进行气候下尺度化。

**📈 对比分析**

与双三次插值和四个重新实现的基线模型进行比较，PC-RF在非负性和质量守恒方面表现优异，但在感知质量（SSIM、PSNR、CRPS）上与基线模型RF-base相当，未显示出显著优势。

**⚠️ 局限性**

限制在于所使用的细参考数据集是从ERA5插值而来，缺乏独立的降水结构，可能导致惩罚和投影的效果无法充分评估。此外，模型未在极端天气条件下进行评估，且未考虑空间变异性。

---

## 567. X-LocVul: An Evidence-Propagation Audit of Cascaded C/C++ Vulnerability Detection, Localization, and Explanation

**arXiv ID:** 2611.21626 | [PDF](https://arxiv.org/pdf/2611.21626v1)

**作者:** Hoang-Phuong-Linh Le `[一作]` (Hanoi University of Science and Technology), Quyet-Thang Huynh `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 681 | [OpenAlex ID](https://openalex.org/A5071728979)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对一个模块化的C/C++漏洞分析管道进行了测量优先的审计，评估了UniXCoder检测器、CodeT5行生成和Qwen2.5-Coder解释器之间的信息传递是否具有语义价值。

**💡 创新点**

创新点在于对证据构建、转移和消费的审计，明确区分了源驻留与定位正确性、辅助标签与语义监督、预测证据与有用证据之间的关系。

**🔧 技术方法**

使用了UniXCoder、CodeT5和Qwen2.5-Coder等技术，构建了一个多任务检测器和解释器。

**📊 数据集**

使用了Big-Vul和PrimeVul等数据集，进行C/C++函数的漏洞检测和定位。

**📈 对比分析**

与CodeBERT等方法进行了比较，发现UniXCoder在某些指标上表现更好，但在主要检测端点上，真实CWE监督与随机CWE监督之间没有显著差异。

**⚠️ 局限性**

限制在于缺乏完整的样本清单和某些实验的具体配置，导致无法完全重现所有结果。

---

## 568. Check Your Blind Spots: Decision-Guided Perception by Probing the Planner with Hypothetical Objects

**arXiv ID:** 2611.23030 | [PDF](https://arxiv.org/pdf/2611.23030v1)

**作者:** Hoang Pham Cong `[一作]` (Hanoi University of Science and Technology), Binh P. Nguyen `[通讯]` (Victoria University of Wellington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种决策引导的推测感知方法，通过查询规划器来决定感知的区域，验证这些区域及其支持，以实现高效的低分辨率感知。

**💡 创新点**

创新点在于通过假设对象的插入来探测规划器，从而生成决策敏感性场，标记出未见对象可能改变动作的位置，并结合支持集进行高分辨率验证。

**🔧 技术方法**

使用了决策引导的推测感知算法，该算法结合了低分辨率草图和高分辨率验证，采用了迭代的高分辨率验证机制。

**📊 数据集**

在三个驾驶数据集上进行测试，包括nuScenes、Argoverse 2和Argoverse-HD。

**📈 对比分析**

与其他方法相比，该方法在约22-27%的像素输入下，能够在97.5%-98.7%的帧上重现参考动作，且在制动F1分数上表现最佳，尤其在草图未能检测到关键对象时，性能提升显著。

**⚠️ 局限性**

局限性在于验证过程仅在一定条件下保证返回整个帧高分辨率感知的动作，且时间估计需要在一个尺度下的历史数据，可能导致在某些情况下的召回率下降。

---

## 569. PACD: Provenance-Aware Consensus Decoding for Mitigating Cross-Modal Hallucinations in Omnimodal Large Language Models

**arXiv ID:** 2611.22456 | [PDF](https://arxiv.org/pdf/2611.22456v1)

**作者:** Chang Sun `[一作]` (Xi'an Jiaotong University), Wei Wang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Provenance-Aware Consensus Decoding (PACD)的训练无关框架，用于减轻跨模态幻觉问题。

**💡 创新点**

PACD通过结合相关性引导的候选仲裁和基于来源的共识恢复，解决了现有方法忽视可靠模态特定预测和错误共识持续存在的两个关键挑战。

**🔧 技术方法**

使用了相关性引导候选仲裁（RGCA）和基于来源的共识恢复（PBCR）技术。

**📊 数据集**

在五个基准数据集上进行了实验，包括CMM、AVHBench、SVHalluc、OmniVideoBench和FutureOmni。

**📈 对比分析**

与现有的训练无关解码方法（如AVCD和MAD）相比，PACD在CMM、AVHBench和SVHalluc上实现了最高的整体准确率，平均提高了7.7%至27.6%。

**⚠️ 局限性**

PACD的局限性在于其依赖于模态特定的证据和信心约束，可能在某些情况下无法完全消除错误共识。

---

## 570. Who Pays for Byzantine Robustness? Validity, Data-Share Dictatorships, and Fairness When Only a Few Hospitals Collaborate

**arXiv ID:** 2611.22904 | [PDF](https://arxiv.org/pdf/2611.22904v1)

**作者:** Adnan Malkani `[一作]` `[通讯]` (Sapienza University of Rome), Adnan Malkani (Sapienza University of Rome)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究对八种标准的聚合规则进行了实证审计，评估了它们在小型医院合作中的表现，特别是在存在拜占庭攻击的情况下。

**💡 创新点**

创新点在于引入了基于样本权重的有效性检查，识别出清洁独裁现象，并通过精确的留出控制分解了诚实数据损失与真正攻击损害。

**🔧 技术方法**

使用了多种拜占庭稳健聚合技术，包括FedAvg、坐标中位数、修剪均值、Krum、Multi-Krum、Bulyan、Weiszfeld几何中位数和Centered Clipping。

**📊 数据集**

使用了来自FLamby基准的三个真实跨孤岛医疗数据集：Fed-Heart-Disease（4家医院）、Fed-TCGA-BRCA（6个地区）和Fed-ISIC2019（6个中心）。

**📈 对比分析**

通过与每种攻击配置配对的确定性留出控制进行比较，发现当最大医院攻击时，聚合规则的失败主要是由于诚实训练数据的损失，而非对抗性扰动造成的损害。

**⚠️ 局限性**

限制在于研究仅覆盖了三个临床数据集，且假设医院报告真实样本大小，未来工作应扩展到临床大语言模型的参数高效微调。

---

## 571. Residual-Guided Multi-Step Conditional Generation for Fast Airfoil Flow Field Prediction

**arXiv ID:** 2611.22629 | [PDF](https://arxiv.org/pdf/2611.22629v1)

**作者:** Weishao Tang `[一作]` (Tsinghua University), Yufei Zhang `[通讯]` (Tsinghua University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种残差引导的顺序推断框架，以提高气动流场推断在分布外（OOD）场景下的鲁棒性，框架由三个基于Transformer的组件组成：流初始化器、基于残差的校正器和基于冲击的校正器。

**💡 创新点**

创新点在于将传统数值方法与深度学习相结合，构建了一个多阶段的流场校正操作，显著提高了模型在OOD场景下的预测准确性。

**🔧 技术方法**

使用了基于Transformer的模型架构，具体包括流初始化器、基于残差的校正器和基于冲击的校正器。

**📊 数据集**

训练数据集包含45000个气动翼样本及其对应的高保真流场，OOD测试集基于气动优化任务构建，涵盖了多种几何形状。

**📈 对比分析**

与传统的Geo-to-Flow预测方法相比，提出的框架在OOD测试集上分别减少了30%的L1误差、59%的CL误差和41%的CD误差，且每个样本的额外推断时间仅为0.25秒。

**⚠️ 局限性**

限制在于当前框架主要针对二维气动翼流场，未来工作将扩展到三维气动问题。

---

## 572. A Critical Appraisal and Motivated Intruder Test of A Search Dataset Anonymised for the Purposes of the EU Digital Markets Act

**arXiv ID:** 2611.21689 | [PDF](https://arxiv.org/pdf/2611.21689v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 573. Think Before You Code: A Benchmark for Testing LLMs on Software Engineering Deliberation

**arXiv ID:** 2611.22905 | [PDF](https://arxiv.org/pdf/2611.22905v1)

**作者:** Ashman Mehra `[一作]` `[通讯]` (Carnegie Mellon University), Ashman Mehra (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究开发了一个名为Think Before You Code (TBYC)的基准管道，旨在评估代码聚焦的语言模型是否能够像人类工程师一样推理软件问题。该基准从GitHub问题讨论中提取结构化记录，并评估模型生成的推理与人类注释之间的匹配程度。

**💡 创新点**

创新点在于将软件工程推理过程转化为可测量的结构化表示，并通过与人类讨论的比较来评估模型的推理能力，而不仅仅是生成正确的代码。

**🔧 技术方法**

使用了混合稀疏-密集检索架构（BM25和CodeBERT），并结合了结构化提示和多种模型的输出进行评估。

**📊 数据集**

使用了来自七个公共GitHub仓库的约1000个问题的真实讨论数据集，这些问题具有丰富的评论和多样的类型。

**📈 对比分析**

通过与九个前沿模型的比较，发现当前大型语言模型在识别技术问题方面表现出一定的能力，但在深层工程推理和结构性讨论方面仍然存在显著的局限性，整体得分普遍较低，最高得分为0.436。

**⚠️ 局限性**

限制在于模型在提取结构化元数据时表现不佳，通常无法达到0.20的阈值，表明模型在识别讨论内容和提取具体结论方面存在困难。

---

## 574. SkillBuffBench: Measuring What Agent Skills Contribute across Harnesses, Models, Skill Access Modes and Memory

**arXiv ID:** 2611.22017 | [PDF](https://arxiv.org/pdf/2611.22017v1)

**作者:** Gengyue Dong `[一作]` (University of Oxford), Guanghui Li `[通讯]` (BraneMatrix AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基准和评估协议，旨在评估编码代理在不同技能条件和记忆模式下的表现，特别是在材料科学和生物学领域的研究任务中。

**💡 创新点**

创新点在于将技能访问视为三层因素，并将其与记忆状态结合，提供了一个全面的实验设计，能够独立评估技能和记忆对任务成功的影响。

**🔧 技术方法**

使用了全因子实验设计，结合了四种代理工具和多种模型，采用了确定性验证器和统计报告方法。

**📊 数据集**

使用了来自材料科学和生物学领域的专家贡献的研究任务，具体任务包括氧扩散系数提取、晶体相识别和细胞分布测量等。

**📈 对比分析**

通过与现有基准的比较，发现经过策划的技能可以显著提高任务成功率，但在实际部署中，技能的选择和记忆的影响仍需进一步研究。

**⚠️ 局限性**

限制在于当前评估仍在进行中，结果尚未完全可用，且未能充分控制记忆对技能效果的影响。

---

## 575. Surfacing Disagreement Beats Resolving It: A Gated Ablation Study of Provenance-Weighted Memory for Multi-Agent Debate

**arXiv ID:** 2611.22461 | [PDF](https://arxiv.org/pdf/2611.22461v1)

**作者:** Ziaullah Khan `[一作]` (Inje University), Hee-Cheol Kim `[通讯]` (Inje University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了在多智能体辩论中共享内存的构建，特别是如何处理矛盾信息。通过实验，发现保留矛盾并将其传递给辩论可以显著提高性能。

**💡 创新点**

创新点在于提出了一种新的内存服务机制，允许在固定预算下独立关闭五种文献机制，最终发现保留矛盾的机制显著提高了F1分数，并减少了攻击成功率。

**🔧 技术方法**

使用了DebateGraph作为实验平台，结合了图结构检索和多智能体系统的设计，采用了门控消融实验的方法。

**📊 数据集**

使用了LoCoMo数据集和HotpotQA-distractor数据集，前者包含5788个对话回合，后者则是基于问题的段落集合。

**📈 对比分析**

与现有方法的比较显示，保留矛盾的机制在39次实验中表现优异，F1分数提高了0.340，攻击成功率从22降至7，且在两个检索后端上均能复制这一效果。

**⚠️ 局限性**

限制在于并非所有实验都共享可达的机制，统计功效在较弱条件下不足，且所有结果均基于单一模型，未能分离模型特定效应与机制层面效应。

---

## 576. ROSMonitoring in Action: Runtime Monitoring and Enforcement for ROS 2 Actions

**arXiv ID:** 2611.22407 | [PDF](https://arxiv.org/pdf/2611.22407v1)

**作者:** Giacomo Paltrinieri `[一作]` (University of Modena and Reggio Emilia), Angelo Ferrando `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文扩展了ROSMonitoring，增加了对ROS 2动作的运行时监控和执行支持，能够重建动作的生命周期并处理相关事件。

**💡 创新点**

创新点在于将动作的生命周期信息与监控系统结合，使得监控能够理解和处理动作的状态和事件，提供协议感知的执行。

**🔧 技术方法**

使用了ROS 2的通信机制，结合运行时监控框架，生成监控节点并通过WebSocket与验证oracle进行交互。

**📊 数据集**

使用了模拟的移动操控器案例研究，结合Nav2、MoveIt 2和ros2_control进行评估，同时进行了合成的可扩展性基准测试。

**📈 对比分析**

通过控制压力测试评估实现，结果显示在固定工作负载下，监控成本与动作调用次数呈线性增长，但在更大的工作负载和更多客户端-服务器对时引入额外开销。

**⚠️ 局限性**

当前实现的限制在于同步执行路径的性能，需优化以适应包含多个短动作的工作负载。

---

## 577. Controlling Proactive Multimodal AI: Separating Perception from Participation

**arXiv ID:** 2611.22431 | [PDF](https://arxiv.org/pdf/2611.22431v1)

**作者:** Muawiz Feroze Khan `[一作]` (University of Illinois Urbana-Champaign), Yun Huang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究探讨了如何使主动多模态人工智能（AI）在开放式探索中可检查和用户可配置。通过设计探针SpotMe，研究了用户如何控制AI的感知和参与。

**💡 创新点**

创新点在于将感知与参与分离，使用户能够独立配置AI的感知能力和参与程度，同时提供对AI干预和沉默决策的可检查性。

**🔧 技术方法**

使用了Chrome扩展程序和FastAPI后端，结合屏幕活动、语音和交互日志来决定AI的参与方式，并通过审计面板展示决策依据。

**📊 数据集**

使用了关于美国致命警察使用武力的公共公民数据仪表板进行用户研究，参与者在此仪表板上进行开放式探索。

**📈 对比分析**

通过与传统的主动AI系统进行比较，发现参与者能够在保持所有感知通道可用的同时，将参与程度降低到询问或沉默。参与者对AI的干预和沉默决策的检查帮助他们理解AI的参与选择。

**⚠️ 局限性**

限制在于研究仅基于一个探针的使用经验，支持类型的八个分类是基于先前的行为而非全面分类，且参与者的反馈可能不适用于长期或设备拥有的使用场景。

---

## 578. Byzantine Resilience of Robot Gathering under Round-Robin Scheduling

**arXiv ID:** 2611.22631 | [PDF](https://arxiv.org/pdf/2611.22631v1)

**作者:** Fabian Frei `[一作]` (MIT), Koichi Wada `[通讯]` (Hosei University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

研究了在存在拜占庭机器人情况下，n个匿名、无意识移动机器人在平面上进行确定性聚集的问题，采用轮流调度和刚性运动。

**💡 创新点**

提出了一种新的算法，能够在f≤⌊√(n)/2⌋的条件下实现聚集，显著提高了拜占庭容错能力的下限，从一个拜占庭机器人提升到Θ(√(n))。

**🔧 技术方法**

使用了Look-Compute-Move (LCM)模型，结合刚性运动和强多重性检测的假设。

**📊 数据集**

没有具体提到使用的数据集，但研究对象为n个移动机器人，包含f个拜占庭机器人。

**📈 对比分析**

与之前的研究相比，证明了在轮流调度下，聚集算法的容错能力从一个拜占庭机器人提升到Θ(√(n))，并且给出了ψ(n)=Ω(√(n))的下限。

**⚠️ 局限性**

存在显著的差距在于当前的下限和线性上限之间，特别是是否轮流调度能够容忍常数比例的拜占庭机器人仍然是一个开放问题。

---

## 579. Data-Efficient Algal Bloom Segmentation in Freshwater Bodies Using a Three-Stage Transfer Learning Pipeline with Attention-Enhanced Decoding

**arXiv ID:** 2611.22064 | [PDF](https://arxiv.org/pdf/2611.22064v1)

**作者:** Albert Ansah Benson `[一作]` (Kansas State University), Sudan Baral `[通讯]` (Kansas State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种三阶段的迁移学习管道，用于在淡水图像中进行有害藻华的自动分割，解决了像素级注释稀缺的问题。

**💡 创新点**

创新点在于结合了模型辅助注释协议和SCSE解码器注意力机制，显著扩展了标注的藻类图像数据集，并提高了分割性能。

**🔧 技术方法**

使用了Swin Transformer U-Net架构，结合了SCSE注意力机制进行解码。

**📊 数据集**

使用了两个公共水体分割数据集（RIWA和LaRS）以及扩展后的藻类图像数据集，从145张扩展到655张图像。

**📈 对比分析**

与九个基线模型进行比较，提出的模型在256张测试图像上达到了0.919的mIoU和0.902的藻类IoU，性能排名第四。

**⚠️ 局限性**

限制在于基准测试图像与训练池之间存在重叠，可能导致得分过于乐观；此外，模型在高分辨率无人机图像上的过度分割问题也影响了性能评估。

---

## 580. Get Hint-Augmented GRPO Done Right Through Activation Steering

**arXiv ID:** 2611.23124 | [PDF](https://arxiv.org/pdf/2611.23124v1)

**作者:** Khiem Le `[一作]` (University of Notre Dame), Nitesh V. Chawla `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的HA-GRPO方法ASH，通过重新解释提示的功能为引导模型行为，解决了训练与推理之间的上下文不匹配问题。

**💡 创新点**

ASH通过引导模型行为而非简单降低问题难度，内部化提示的行为转变为可训练的引导向量，从而在推理时无需提示也能保持性能。

**🔧 技术方法**

使用了激活引导技术，通过在模型的激活中注入可训练的引导向量来实现。

**📊 数据集**

使用了OpenR1-MATH-220k数据集，该数据集包含220,000个数学问题及其参考解决方案。

**📈 对比分析**

与标准GRPO和其他HA-GRPO方法（如QUESTA、SAGE、HILL）进行比较，ASH在所有六个基准测试中均表现出一致的性能提升，平均提升达到2.60%至3.84%。

**⚠️ 局限性**

ASH的一个潜在限制是训练引导向量的计算开销，尽管在实践中这一开销相对较小，约为GRPO训练时间的1.5倍。

---

## 581. ReTouchGS: Ultralight Render-free Aesthetic Color Enhancement and Transfer for 3D Gaussian Splatting

**arXiv ID:** 2611.23054 | [PDF](https://arxiv.org/pdf/2611.23054v1)

**作者:** Chuanzhi Xu `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无参考的美学颜色增强框架，支持预训练的3D高斯点云模型的颜色增强和参考引导的颜色转移。

**💡 创新点**

首次实现了无参考的美学颜色增强，能够自动提升场景的视觉吸引力，同时保持不同视角之间的一致性。

**🔧 技术方法**

使用了共享的混合感知颜色细化网络和场景级颜色查找表（LUT），并结合了分析颜色更新和学习的颜色修正。

**📊 数据集**

使用了37个场景的基准数据集，包含来自专家修饰的图像对，以支持美学颜色增强和参考引导的颜色转移的评估。

**📈 对比分析**

与多种基线方法进行了比较，结果显示在颜色准确性、PSNR和LPIPS等指标上均优于所有对比方法，且在不同视角下表现出强大的泛化能力。

**⚠️ 局限性**

该方法的局限性在于可能对某些复杂场景的颜色增强效果不如基于渲染的方法，且在特定情况下可能需要更多的用户输入来达到最佳效果。

---

## 582. Efficient Self-Supervised Adaptation of 3D Abdominal Vision-Language Model for Institution-Specific HCC Classification via Full Fine-Tuning and PEFT

**arXiv ID:** 2611.22504 | [PDF](https://arxiv.org/pdf/2611.22504v1)

**作者:** Febryan Putra Kartika `[一作]` (Chang Gung Memorial Hospital), Kuan-Fu Chen `[通讯]` (Chang Gung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本研究对预训练的3D医学视觉语言模型进行了微调，以实现肝细胞癌（HCC）的分类，使用了来自不同机构的配对腹部CT扫描和放射学报告。

**💡 创新点**

创新点在于使用参数高效微调（PEFT）策略，减少计算成本并在有限数据条件下进行适应，同时首次分析了对比温度对分类性能的影响。

**🔧 技术方法**

使用了自监督对比学习和参数高效微调技术（ConvLoRA和LoRA）来适应模型。

**📊 数据集**

使用了一个内部机构特定的腹部CT数据集，包含3599个扫描，其中1713个为HCC阳性病例。

**📈 对比分析**

与全微调方法相比，PEFT在更新约1%的模型参数的情况下，达到了相似的性能，PEFT的AUC为0.79，F1为0.70，而全微调的AUC为0.82，F1为0.71。

**⚠️ 局限性**

限制在于只评估了一种PEFT策略（ConvLoRA和LoRA），并且研究仅集中在门静脉相位CT上，未来可以扩展到多相位CT和多类肝肿瘤分类。

---

## 583. Chimera: A Configurable LLM Decode Accelerator Architecture with Compute-Memory-Floorplan Co-Design for HBM-Equipped Multi-Die FPGAs

**arXiv ID:** 2611.21677 | [PDF](https://arxiv.org/pdf/2611.21677v1)

**作者:** Zhiheng Chen `[一作]` (University of California), Sitao Huang `[通讯]` (University of California)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可配置的LLM解码加速器架构Chimera，旨在提高HBM装备的多芯FPGA上的LLM解码性能。

**💡 创新点**

创新点在于将计算、内存访问和布局设计进行联合设计，以提高解码性能和能效。

**🔧 技术方法**

使用了可配置的LLM解码加速器架构，结合了HBM代理端口、分层专家存储和统一配置分析等技术。

**📊 数据集**

在AMD Alveo U280上构建了十个设计，针对八种不同的LLM，包括密集型、混合专家和混合线性/标准注意力模型。

**📈 对比分析**

与四个先前的FPGA实现相比，Chimera的设计在吞吐量上达到了1.6倍的几何平均值，并且在能效上分别达到了RTX 3090和A100的3.0倍和1.5倍。

**⚠️ 局限性**

限制在于高带宽内存并不能单独保证高解码性能，计算、内存组织和布局设计的联合优化是关键。

---

## 584. AdaFlowLite: Scalable and Non-blocking Inference on Asynchronous Mobile Data

**arXiv ID:** 2611.21691 | [PDF](https://arxiv.org/pdf/2611.21691v1)

**作者:** Sicong Liu `[一作]`, Zhiwen Yu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的系统AdaFlowLite，用于在异步移动数据上进行可扩展和非阻塞推理，允许在部分数据到达时立即进行推理。

**💡 创新点**

创新点在于引入了一种结构化的跨模态亲和力矩阵，能够动态适应异步和异质数据流，并通过轻量级的多模态Swin Transformer实现实时数据插补。

**🔧 技术方法**

使用了多模态轻量级Swin Transformer（MMLST）技术，结合了t-SNE和层次分析法（AHP）来构建亲和力矩阵。

**📊 数据集**

使用了nuScenes数据集，该数据集包含来自真实世界自动驾驶场景的多模态数据，包括6个摄像头视图和1个LiDAR视图。

**📈 对比分析**

与现有的同步和异步方法相比，AdaFlowLite在推理准确性、延迟和能耗方面表现优越，推理延迟减少了高达80.4%，准确性提高了62.1%。

**⚠️ 局限性**

限制在于该方法在处理新传感器的集成时仍然面临挑战，尤其是在缺乏先验知识的情况下，如何有效管理和优化亲和力矩阵。

---

## 585. Countering Anion Inhibition of Laccase Electron Transfer: Electrostatic Design, Charge-Transfer Dynamics on a Superconducting Quantum Processor, and Its Limits

**arXiv ID:** 2611.22813 | [PDF](https://arxiv.org/pdf/2611.22813v1)

**作者:** Lucia Malickova `[一作]` `[通讯]` (Modelos Inteligencia Artificial), Lucia Malickova (Modelos Inteligencia Artificial)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究使用Euro-Q-Exa量子计算机对真菌漆酶的电荷转移动态进行了模拟，探讨了氢氧根和卤素对其抑制作用的机制。

**💡 创新点**

创新点在于通过量子计算精确测量了氢氧根和卤素对电荷转移能量的影响，并识别了突变Y116E能够显著降低这些抑制效应。

**🔧 技术方法**

使用了Euro-Q-Exa量子计算机，执行了基于Hubbard模型的电荷转移动态模拟，结合了量子电路和经典模型的比较。

**📊 数据集**

研究中使用了真菌漆酶Trametes versicolor的晶体结构（PDB 1GYC）作为数据集，并进行了多种电介质模型的比较。

**📈 对比分析**

通过与经典模型的比较，量子计算结果在氢氧根和卤素的电荷转移能量变化上达到了meV级的精度，突变Y116E显著降低了氢氧根的电荷转移惩罚（从93 ±4meV降至19 ±9meV）。

**⚠️ 局限性**

限制在于模型使用了124-141个原子的聚类模型，且所有紧束缚参数来自GFN2-xTB，未进行DFT检查，且未能解决电子关联的问题。

---

## 586. Groups of AI agents underuse resources despite shared evidence of increased capacity

**arXiv ID:** 2611.22632 | [PDF](https://arxiv.org/pdf/2611.22632v1)

**作者:** Takahiro Ezaki `[一作]` (Research Center for Advanced Science and Technology, University of Tokyo), Katsuhiro Nishinari `[通讯]` (Department of Aeronautics and Astronautics, University of Tokyo)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本研究探讨了AI代理在共享有限资源时的决策行为，特别是在资源恢复后如何有效利用这些资源。通过模拟实验，研究了代理在选择服务器时的表现，尤其是在服务器状态报告滞后的情况下。

**💡 创新点**

创新点在于通过实验分离了三个关键要求：发现变化、对证据作出反应以及在有限资源上分配行动。研究表明，尽管个体响应有效，但集体效率未能达到最优，揭示了共享信息的复杂性。

**🔧 技术方法**

使用了语言模型代理（如Anthropic的Claude Haiku和OpenAI的GPT-6.1 Sol）进行模拟实验，操控代理对资源状态的认知和反馈。

**📊 数据集**

实验使用了模拟服务器A和B，A在前两轮不可用，第三轮恢复，B的容量始终为2。通过不同的报告布局和请求计数的显示与隐藏，测试了代理的决策行为。

**📈 对比分析**

与其他方法比较时，尽管共享成功记录提高了对恢复服务的选择，但所有模型在完成工作方面未能达到优化的独立选择基准，表现出资源利用不充分的现象。

**⚠️ 局限性**

限制在于，尽管个体代理能够有效响应共享信息，但由于代理之间的相同输入导致的集中选择，未能实现资源的有效分配，且不同模型对共享信息的反应存在差异。

---

## 587. Settling the Time and Structural Complexity of Bertrand Networks

**arXiv ID:** 2611.22216 | [PDF](https://arxiv.org/pdf/2611.22216v1)

**作者:** MohammadTaghi Hajiaghayi `[一作]` (University of Maryland), Suho Shin `[通讯]` (Stanford University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文解决了在Bertrand网络中计算近似纳什均衡的复杂性问题，证明了在一般图上计算任何常数误差的近似纳什均衡是NP完全的，并设计了一个高效的多项式时间近似方案（EPTAS），当最大度数和拥有俘获买家的卖家数量都受到限制时。

**💡 创新点**

创新点在于证明了在稀疏二部图上计算近似均衡的NP完全性，并且提供了一个高效的多项式时间近似方案（EPTAS），同时还证明了每个可实现的均衡收入配置可以通过结构复杂度为O(2^n)的均衡来实现。

**🔧 技术方法**

使用了复杂的图论和算法设计技术，包括多项式时间近似方案（EPTAS）和连续线性规划的技术。

**📊 数据集**

使用了Bertrand网络模型，涉及的图是一般图，特别是稀疏二部图，且包含俘获市场和共享市场的权重。

**📈 对比分析**

与现有方法的比较显示，本文的方法在特定条件下（如最大度数和俘获买家数量受限时）能够有效计算近似均衡，而在一般情况下计算是NP完全的，性能表现出显著的复杂性差异。

**⚠️ 局限性**

限制在于对于一般图的均衡计算仍然是NP完全的，且在某些情况下，结构复杂度可能会导致算法的运行时间呈指数级增长。

---

## 588. Edge-Adaptive Hiera-Seg Model for Building-Integrated Photovoltaic Power Forecasting

**arXiv ID:** 2611.21680 | [PDF](https://arxiv.org/pdf/2611.21680v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 589. Projected Penalty-Gradient Tracking for Communication-Efficient Byzantine-Resilient Federated Learning

**arXiv ID:** 2611.22929 | [PDF](https://arxiv.org/pdf/2611.22929v1)

**作者:** Yanjie Dong `[一作]` `[通讯]`, Yanjie Dong

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在存在拜占庭对手的情况下，如何在异构数据集上进行通信高效的联邦学习。

**💡 创新点**

提出了一种投影惩罚梯度跟踪框架，以保持在压缩上传下的拜占庭鲁棒性，并提出了两种算法：FRPG-CS和LAG-FRPG。

**🔧 技术方法**

使用了投影惩罚梯度跟踪框架，结合了任何收缩压缩器，并建立了期望目标差距的有限时间界限。

**📊 数据集**

在MNIST和Fashion-MNIST数据集上进行了实验。

**📈 对比分析**

与十种基准方法进行了比较，FRPG-CS和LAG-FRPG在Min-Max攻击下的准确率最高，而在无攻击情况下，压缩动量基准和本地更新基准在准确率和总流量上更具优势。

**⚠️ 局限性**

限制在于分析要求强凸损失、可信的服务器和可靠的链接，且评估仅限于两个图像数据集的线性模型。

---

## 590. Stopping Is Not the Limit: Understanding and Eliciting Agent Capability in Long-Horizon Vulnerability Exploitation

**arXiv ID:** 2611.22888 | [PDF](https://arxiv.org/pdf/2611.22888v1)

**作者:** Deokhyung Kang `[一作]` (UNIST), Sangdon Park `[通讯]` (POSTECH)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了在漏洞利用中，语言模型代理在已知软件缺陷的基础上开发有效利用代码的能力，探讨了在停止响应后继续执行的潜在收益。

**💡 创新点**

提出了SCoRe方法，通过上下文管理来提高漏洞利用能力，表明停止并不一定标志着代理的利用极限。

**🔧 技术方法**

使用了DeepSeek-V4-Flash-0731和MiMo-V2.6-Pro模型，并提出了SCoRe（Stop-aware Compaction and Relabeling）作为上下文管理策略。

**📊 数据集**

使用了ExploitBench数据集，该数据集包含41个针对V8引擎的漏洞利用任务。

**📈 对比分析**

与原始停止策略（ReAct）和基于摘要的继续策略（ReSum）进行比较，SCoRe在两个模型上均显示出更高的能力覆盖率，MiMo在2000轮中达到了67.0%的能力覆盖分数，超过了原始策略的32.4%和GPT-5.5的58.6%。

**⚠️ 局限性**

本研究的局限性在于仅关注了两个模型和V8漏洞利用任务，未来需要扩展到更多模型和漏洞类别的评估。

---

## 591. Higher-Order Interaction Reconstruction via Integral Physics-Informed Neural Networks

**arXiv ID:** 2611.21935 | [PDF](https://arxiv.org/pdf/2611.21935v1)

**作者:** Xin Mao `[一作]` (University of North Carolina), Can Chen `[通讯]` (University of North Carolina)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的积分物理信息神经网络（IntegralPINN）框架，用于从噪声时间序列数据中重建高阶交互结构。

**💡 创新点**

通过用积分残差替代传统的微分残差，增强了对噪声和非平滑动态的鲁棒性，同时避免了显式时间微分。

**🔧 技术方法**

使用了积分物理信息神经网络（IntegralPINN）技术，该技术结合了神经网络和物理一致性约束。

**📊 数据集**

在生态动态、社会传播和神经同步系统等多个数据集上进行了广泛实验。

**📈 对比分析**

与现有的最先进方法（如HyperPINN、核方法、THIS和积分SINDy）进行比较，IntegralPINN在噪声和稀疏观察下表现出更高的准确性和鲁棒性，尤其在高阶交互重建中显著优于其他方法。

**⚠️ 局限性**

该方法的局限性在于需要已知的节点动态，并且在处理大规模网络时可能面临优化效率和高维交互张量参数化的挑战。

---

## 592. DLS: Dual-Logic Graph Neural Networks with Chaos-Fuzzy Residuals for Robust Link Prediction

**arXiv ID:** 2611.22607 | [PDF](https://arxiv.org/pdf/2611.22607v1)

**作者:** Mohammad Mehdi Keikha `[一作]` (University of Sistan and Baluchestan), Abolfazl Nadi `[通讯]` (University of Tehran)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的链路预测框架DLS（Dual-Logic SAGE），旨在通过“受控不稳定性”机制打破过平滑的限制。

**💡 创新点**

创新点在于引入混沌理论和模糊逻辑，利用混沌注入模块和模糊逻辑门控机制来增强特征区分能力和鲁棒性。

**🔧 技术方法**

使用了混沌理论（Logistic Map）和模糊逻辑技术。

**📊 数据集**

使用了Cora、Squirrel、Chameleon和Actor等多个基准数据集进行实验。

**📈 对比分析**

与现有的最先进方法（如Neo-GNN、BUDDY、ELPH和NCNC）进行了比较，DLS在异质图环境中表现显著优于这些基线，同时在同质图上也保持了竞争力。

**⚠️ 局限性**

限制在于当前DLS依赖于固定的混沌控制参数，假设整个图的非线性需求是均匀的，未来可以考虑将其转变为可学习的节点特定变量。

---

## 593. Beyond Co-Generation: Rethinking Jointness in Joint Audio-Video Generation

**arXiv ID:** 2611.22857 | [PDF](https://arxiv.org/pdf/2611.22857v1)

**作者:** Jihoo Jung `[一作]` (KAIST), Joon Son Chung `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究旨在提高对联合文本到音频视频生成（T2AV）中真实音视频协调的理解，特别强调语义一致性。通过引入内在联合性，评估音视频生成中的音视频一致性。

**💡 创新点**

创新点在于定义了内在联合性，并提出了JointREPA方法来促进跨模态协调，同时引入了UNSAIDBench基准来评估语义联合性。

**🔧 技术方法**

使用了基于表示对齐的后训练方法JointREPA，旨在提高音视频生成模型的语义一致性。

**📊 数据集**

使用了VGGSound数据集的平衡子集，并引入了UNSAIDBench基准来评估语义联合性。

**📈 对比分析**

通过与现有基准（如JavisBench和Verse-Bench）进行比较，JointREPA在多个模型和基准上均表现出一致的语义一致性提升。

**⚠️ 局限性**

限制在于本研究主要集中于显式语义身份的联合性，未来可以扩展到物理属性（如材料和重量）的研究。

---

## 594. Supporting Researcher Control in AI-Assisted Social Media Analysis: A Mixed-Methods Study of AI4OSN

**arXiv ID:** 2611.22006 | [PDF](https://arxiv.org/pdf/2611.22006v1)

**作者:** Qile Wang `[一作]` (University of Delaware), Matthew Louis Mauriello `[通讯]` (University of Delaware)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开发了AI4OSN，一个混合主动的平台，旨在支持研究人员在社交媒体分析中的数据收集、标注和理解，同时保持研究人员的控制权。

**💡 创新点**

创新点在于提供一个人本中心的系统，使研究人员能够在AI辅助下进行社交数据分析，同时保持对查询、标注和解释的控制。

**🔧 技术方法**

使用了大型语言模型（LLMs）作为辅助工具，结合混合主动系统的设计理念。

**📊 数据集**

使用了来自X/Twitter的数据集进行系统的用户研究，参与者为18人。

**📈 对比分析**

通过用户研究评估了系统的可用性，参与者报告的SUS得分为79.7，表明系统的可用性高于基准水平。参与者在AI辅助的标注和分析任务中信心增加，认知负担低。

**⚠️ 局限性**

限制在于AI标注的可靠性和一致性，参与者对AI输出的信任程度各异，且在手动标注过程中仍感到负担较重。

---

## 595. FaceRiff: Data-Efficient Unified Text-to-3D Face Asset Synthesis via Geometric Conditioning

**arXiv ID:** 2611.21867 | [PDF](https://arxiv.org/pdf/2611.21867v1)

**作者:** Elly Akhoundi `[一作]` (Electronic Arts), Étienne Danvoye `[通讯]` (Electronic Arts)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种名为FaceRiff的框架，能够从文本生成完整的3D面部资产，包括网格、反照率和位移，且在共享的UV空间中进行合成。

**💡 创新点**

创新点在于通过几何条件化和参数高效的方式，利用预训练的SDXL骨干网络和任务特定的LoRA适配器，生成高质量的3D面部资产，而无需大量专门的数据集。

**🔧 技术方法**

使用了预训练的SDXL骨干网络、LoRA适配器和Normal ControlNet进行几何条件化。

**📊 数据集**

使用了2049个来自约350个身份的面部扫描数据集进行训练。

**📈 对比分析**

与FaceG2E和DreamFace进行比较，FaceRiff在每个提示的多样性上表现更好，同时保持提示对齐和固定拓扑的UV一致性，且在盲测中被评估为比DreamFace更具真实感。

**⚠️ 局限性**

限制在于数据集无法公开发布，且模型的训练和推理过程可能受到生成错误的影响。

---

## 596. Token Commitment as a Control Surface for Steering Masked Diffusion Generation

**arXiv ID:** 2611.21669 | [PDF](https://arxiv.org/pdf/2611.21669v1)

**作者:** Ambroise Aigueperse `[一作]` (Stanford University), Marco Pavone `[通讯]` (Stanford University)

**通讯引用:** 29019 | [OpenAlex ID](https://openalex.org/A5050003000)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了掩蔽扩散模型中的令牌承诺，将其分为两个决策：承诺哪些令牌和每步承诺多少令牌。

**💡 创新点**

提出了反事实目标感知承诺和校准自适应预算，提供了对令牌生存和承诺速率的独立控制。

**🔧 技术方法**

使用了掩蔽扩散模型，特别是LLaDA和Dream-7B模型，结合了激活引导方法。

**📊 数据集**

使用了多个数据集，包括Amazon、SAMSum和SentiCap，涉及情感分析、主观性、摘要生成和图像-文本对齐等任务。

**📈 对比分析**

与无激活引导的随机调度方法进行比较，结果显示自适应预算在平均去噪成本下具有竞争力，且目标-质量权衡是单调的。

**⚠️ 局限性**

限制在于反事实承诺的计算开销较大，且未涵盖所有目标类型的广泛评估。

---

## 597. Revisiting Index Performance under High Cache-Miss Penalties

**arXiv ID:** 2611.21835 | [PDF](https://arxiv.org/pdf/2611.21835v1)

**作者:** Mathew M. Kim `[一作]` (Jones), Beomseok Nam `[通讯]` (Sungkyunkwan University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AuxBench，一个微基准测试工具，用于在更大的应用工作负载中评估索引的性能，考虑了后查找内存活动、并发性和请求到达模式。

**💡 创新点**

创新点在于将索引视为应用工作负载中的辅助访问路径，而不是孤立的数据结构，从而揭示了传统基准测试无法捕捉的性能变化。

**🔧 技术方法**

使用了微基准测试技术，结合了可配置的后查找内存活动、请求到达模式和操作混合。

**📊 数据集**

评估了四种现有的字节可寻址索引结构，包括FAST&FAIR、WB+Tree、μTree和FPTree。

**📈 对比分析**

与传统的索引基准测试相比，AuxBench显示出在引入应用工作后，索引的相对性能可能会发生显著变化，尤其是在高缓存失效率和请求动态下，某些索引的优势可能会减弱。

**⚠️ 局限性**

限制在于AuxBench的测量开销和在请求速率超过基准测试能力时可能出现的生产者滞后，这可能影响结果的解释。

---

## 598. It's My Turn! Interpretable Methods Reveal Developmental Differences from Mother-Child Conversational Dynamics

**arXiv ID:** 2611.21969 | [PDF](https://arxiv.org/pdf/2611.21969v1)

**作者:** A'di Dust `[一作]`, Maja Matarić `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究开发了一种可解释的计算框架，结合序列分析、信息论测量和贝叶斯层次建模，来表征母子互动中的时间对话动态，分析了94对母子在游戏中的对话。

**💡 创新点**

创新点在于将传统统计方法与时间序列分析相结合，揭示了母子对话中语言发展水平的差异，尤其是在对话的组织性和可预测性方面。

**🔧 技术方法**

使用了序列分析、互信息量和贝叶斯层次建模等技术。

**📊 数据集**

数据集包括94对母子，儿童年龄为两岁，使用Bayley发展评估进行分类，视频记录了母子间的自由游戏互动。

**📈 对比分析**

通过线性混合效应模型比较不同Bayley组之间的序列相似性，发现低Bayley组的对话组织性显著低于平均组，且在语言Bayley组中，低分组的对话模式更倾向于成人主导的交流。

**⚠️ 局限性**

限制在于统计不确定性可能影响了发展组之间的差异识别，手动标注对话轮次可能引入测量误差，样本量相对较小，未来研究应扩展到更大的纵向数据集以验证结果。

---

## 599. Bad Apples in the Kitchen: How LLM Teams Detect, Correct, and Calibrate Against Their Misaligned Members

**arXiv ID:** 2611.22342 | [PDF](https://arxiv.org/pdf/2611.22342v1)

**作者:** Dongseok Heo `[一作]` (Seoul National University), Bongwon Suh `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在实时协作环境中，当一个团队成员行为不一致时，团队的恢复能力。通过在一个定制的四人Overcooked环境中注入三种理论基础的错误代理，分析了检测、纠正和校准的三个阶段。

**💡 创新点**

提出了一种基于联合行动的错误分类法，并强调了在实时协作中，归因与诊断结合在同一发言中的重要性，这对团队的恢复能力至关重要。

**🔧 技术方法**

使用了混合的LLM-控制架构，结合了政策LLM和聊天LLM来处理行动选择和沟通。

**📊 数据集**

使用了定制的四人Overcooked-AI环境，进行了720个实验回合，测试了不同类型的错误代理对团队表现的影响。

**📈 对比分析**

通过比较不同类型的错误代理（如Reneger、Misallocator和Disruptor），发现检测难度和恢复能力存在显著差异。虽然所有错误代理最终都被识别，但只有在归因和诊断结合在同一发言时，团队才能有效恢复。

**⚠️ 局限性**

研究的局限性在于只在一个环境中进行了单一的操作化，且错误代理固定在特定角色，可能无法捕捉到更复杂的动态。

---

## 600. RSI-Prime: Training Models to Train Models

**arXiv ID:** 2611.23138 | [PDF](https://arxiv.org/pdf/2611.23138v1)

**作者:** Dewu Zheng `[一作]` (Sun Yat-sen University), Yanlin Wang `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了RSI-Prime数据集，包含350个完整实验轨迹，旨在提高大语言模型（LLMs）的模型训练能力。

**💡 创新点**

创新点在于通过决策级别的监督选择高质量行为，同时保留完整的实验上下文，从而帮助LLMs学习如何进行有效的模型训练。

**🔧 技术方法**

使用了决策级别损失掩蔽技术，以排除低质量行为的监督，并通过GLM-5.3模型进行实验验证。

**📊 数据集**

使用了RSI-Prime数据集，该数据集包含350个实验轨迹，涵盖七个任务类别。

**📈 对比分析**

与原始GLM-5.3模型相比，经过RSI-Prime训练后，模型在PostTrainBench基准测试中的整体性能提高了31.95%。决策级别监督在所有六个任务上均优于全轨迹监督，显示出更高的任务性能。

**⚠️ 局限性**

限制在于实验决策的质量评估可能受到主观因素的影响，且在某些情况下，成功的实验结果并不一定反映出所有决策的高质量。

---

## 601. MoE--LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study

**arXiv ID:** 2611.23158 | [PDF](https://arxiv.org/pdf/2611.23158v1)

**作者:** Nikit Srivastava `[一作]` (Paderborn University), Axel-Cyrille Ngonga Ngomo `[通讯]` (Paderborn University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究比较了八种低秩适配（LoRA）方法在一个冻结的基础模型上对200种语言的性能，验证了低秩适配在低资源语言支持中的有效性。

**💡 创新点**

创新点在于系统地比较了不同的低秩适配方法在多语言环境下的表现，并提出了低秩适配方法应被视为针对性工具，而非通用的多语言能力提升手段。

**🔧 技术方法**

使用了低秩适配（LoRA）和混合专家（MoE）技术，结合了多种参数高效的适配方法。

**📊 数据集**

使用了FineWeb和FineWeb2数据集，涵盖了10种、96种和200种语言的训练数据。

**📈 对比分析**

通过在多个基准上进行评估，发现低秩适配在翻译数据上表现良好，但在阅读理解上表现下降，且对低资源语言的支持效果最佳。

**⚠️ 局限性**

限制在于未能训练所有166种配置，且每种配置仅训练一次，可能导致排名不稳定。此外，结果仅基于单一基础模型，可能不适用于其他模型。

---

## 602. Polarity Composition Hypothesis: Causal Evidence from Negation in NLI

**arXiv ID:** 2611.23185 | [PDF](https://arxiv.org/pdf/2611.23185v1)

**作者:** Nhi Hoai Doan `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Kentaro Inui `[通讯]` (RIKEN)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言模型在自然语言推理中如何处理否定，提出了极性组合假设作为这一计算的高层因果模型。

**💡 创新点**

创新点在于提出了极性组合假设，并通过行为评估验证了语言模型的预测与该假设的符合程度。

**🔧 技术方法**

使用了分布式对齐搜索（DAS）技术来识别与极性组合假设中的变量对齐的因果有效子空间。

**📊 数据集**

使用了合成数据集，包括1000个主谓宾（SVO）事件，并在MNLI（多领域自然语言推理）数据集上进行了部分验证。

**📈 对比分析**

通过与其他模型的比较，发现Qwen3-8B和Phi-4 Mini Instruct在多个测试中表现出超过85%的准确率，尤其在匹配事件上表现优异。

**⚠️ 局限性**

限制在于该研究的控制语法仅覆盖简单的英语SVO事件，并假设最多只有一个匹配假设，未考虑量化主语和负一致语言的情况。

---

## 603. Arboreus: A Realistic Boreal Forest Simulator with Topographic Terrain and Satellite-Informed Parametric Tree Distributions

**arXiv ID:** 2611.22908 | [PDF](https://arxiv.org/pdf/2611.22908v1)

**作者:** F. Labra Caso `[一作]`, G. Nikolakopoulos `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了Arboreus，一个现实的北方森林模拟器，旨在支持北方森林环境中机器人自主算法的统计验证。Arboreus通过参数化树木、高度变化的地形和卫星数据驱动的树木密度分布，自动合成大规模、数据驱动的可信森林世界。

**💡 创新点**

创新点在于结合了测量的地形、卫星驱动的植被分布和特定物种的参数化树模型，生成与真实森林环境一致的合成森林，同时允许用户根据实际观察进行定制。

**🔧 技术方法**

使用了参数化树生成、空间树分布和地形重建等技术，结合了来自Copernicus卫星的数据和瑞典土地登记局的地形数据。

**📊 数据集**

使用的数据集包括来自Copernicus卫星的树木覆盖密度和主导树种类型的数据，以及25个对应于三个真实森林地点的1000平方米的森林世界数据集。

**📈 对比分析**

与现有方法相比，Arboreus能够生成具有高变异性的森林环境，支持机器人自主算法的重复测试和统计验证。性能评估显示生成的森林在树木分布和地形插值方面与手动注释的数据具有良好的一致性，RMSE约为0.17米。

**⚠️ 局限性**

限制在于生成的森林世界代表的是可能的配置，而不是精确重建或完整的生态模型，未来的工作应量化建模选择如何影响自主性能和合成数据的转移。

---

## 604. From syllable representation via gesture generation to articulation: A spiking neural model

**arXiv ID:** 2611.22499 | [PDF](https://arxiv.org/pdf/2611.22499v1)

**作者:** Bernd J. Kröger `[一作]` `[通讯]` (RWTH Aachen University), Bernd J. Kröger (RWTH Aachen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于脉冲神经网络的感知运动模型，旨在将音位表示转化为协调的发声道运动，以实现语音生成。

**💡 创新点**

创新点在于结合了神经工程框架和语义指针架构，通过神经振荡器和斜坡动态生成发音时序，提供了一种新的语音生成机制。

**🔧 技术方法**

使用了脉冲神经网络（Spiking Neural Networks）和神经工程框架（Neural Engineering Framework），结合了语义指针架构（Semantic Pointer Architecture）来实现模型。

**📊 数据集**

使用了一个包含30个参考录音的数据集，涵盖了8个词汇单词、一个单词的替代实现、一个音节序列和孤立音节。

**📈 对比分析**

通过与先前的模型进行比较，调整手势命令幅度后，成功实现了60/60的激活率，表明模型在不同神经实现中的激活可靠性得到了提升。

**⚠️ 局限性**

模型的局限性在于尚未实现与声学扩展的完全整合，且需要进一步的实验来验证其在更广泛输入和参数设置下的稳定性。

---

## 605. Where Does Hindsight Enter Clinical Agent Benchmarks? An Audit of Simulators and Case Construction

**arXiv ID:** 2611.22241 | [PDF](https://arxiv.org/pdf/2611.22241v1)

**作者:** Meilu Zhu `[一作]` (University of Hong Kong), Axiu Mao `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种审计方法，分析了大型语言模型（LLM）在诊断任务中的表现，特别是如何评估基于回顾性电子健康记录构建的基准的有效性。

**💡 创新点**

创新点在于识别和量化基准构建中的三种后见偏差渠道，并提出了使用真实数据评估这些渠道的有效性的方法。

**🔧 技术方法**

使用了大型语言模型（LLM）和文本分类器进行诊断准确性评估，并通过模拟器生成未记录的测试结果。

**📊 数据集**

使用了MIMIC-IV数据集，该数据集包含27,489个急诊入院病例，涵盖了多种常见的急诊诊断。

**📈 对比分析**

与基于急诊科分诊记录的呈现时间重建相比，基准风格的摘要使四个LLM的零-shot准确率从25.4%提高到39.7%，出院历史的准确率提高到45.3%。六个交互式代理的得分也有所提高，但排名变化不大。

**⚠️ 局限性**

限制在于使用了单一的数据源（MIMIC-IV），且所有模型均在本地推理，可能无法代表其他中心或更复杂的模型。此外，缺乏同时记录的床边历史，限制了对后见偏差的全面评估。

---

## 606. Hybrid Quantum Selection of Learning Rates in Language-Model Training: Termination Criteria and an Interaction-Guided Schedule Oracle

**arXiv ID:** 2611.21701 | [PDF](https://arxiv.org/pdf/2611.21701v1)

**作者:** Wei-Che Hung `[一作]` `[通讯]` (Independent Researcher), Wei-Che Hung (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了量子最小查找作为层组学习率选择器，以改进语言模型的微调过程。

**💡 创新点**

创新点在于提出了一种交互引导的调度oracle，能够在没有预先计算的评分表的情况下，直接在量子电路中评分所有联合调度。

**🔧 技术方法**

使用了量子搜索技术，结合Dürr–Høyer最小查找和BBHT搜索，并比较了固定预算和基于置信度的终止策略。

**📊 数据集**

使用了在固定训练检查点上测量的评分构建的查找oracle，进行了多次实验以评估不同的终止标准。

**📈 对比分析**

通过比较不同的终止标准，发现所有标准在选择质量上相似，但在停止努力上存在显著差异，选择标准D在停止努力上表现最佳。

**⚠️ 局限性**

论文讨论了量子方面的局限性，包括量子硬件的限制和如何有效利用量子资源以提高搜索效率。

---

## 607. RAW: Refusal Aversion in the Wild, A Causal Measurement Method for Deployed LLMs

**arXiv ID:** 2611.21641 | [PDF](https://arxiv.org/pdf/2611.21641v1)

**作者:** Pavel Kireyev `[一作]` (London School of Economics and Political Science), Pavel Kireyev `[通讯]` (London School of Economics and Political Science)

**通讯引用:** 477 | [OpenAlex ID](https://openalex.org/A5058621082)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为RAW的方法，用于测量用户在大型语言模型（LLM）拒绝请求后的反应，特别是用户是否会因此 disengage。

**💡 创新点**

创新点在于利用LLM的随机性作为自然实验，通过分析现有对话日志来识别拒绝对用户行为的因果影响，而无需进行A/B测试。

**🔧 技术方法**

使用了交叉拟合增强逆倾向加权（AIPW）等技术来估计拒绝对用户再参与的平均处理效应（ATT），并结合了多个次要估计器进行交叉验证。

**📊 数据集**

使用了来自LMSYS-Chat-1M和WildChat的数据集，涵盖了15个已部署的LLM的对话日志。

**📈 对比分析**

通过与其他方法的比较，发现对于约一半的模型，拒绝会导致3到17个百分点的后续再参与成本，而其他模型则未显示出显著影响。

**⚠️ 局限性**

局限性包括选择可观测变量的假设未经过测试，可能存在未捕获的混杂因素；此外，结果是局部效应而非总体效应，且仅限于英语对话。

---

## 608. HiGAI-Net: Hierarchical Encrypted Traffic Classification for Generative AI Services

**arXiv ID:** 2611.22373 | [PDF](https://arxiv.org/pdf/2611.22373v1)

**作者:** Madhan Raj Kanagarathinam `[一作]` (Samsung R&D Institute India), Kapal Dev `[通讯]` (Munster Technological University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为HiGAI-Net的分层加密流量分类器，专门用于生成AI服务的流量识别，能够在没有有效负载访问和SNI可见性的情况下，仅依赖数据包头部元数据进行分类。

**💡 创新点**

创新点在于引入了一套滑动窗口时间特征，特别是Δ R和流媒体指数S，这些特征显著提高了L3的F1得分，并且通过每流标签缓存实现了每个数据包的推理时间低于1微秒。

**🔧 技术方法**

使用了机器学习技术，包括LightGBM、XGBoost和随机森林等模型，并进行了超参数调优和窗口大小的搜索。

**📊 数据集**

使用了一个包含44,492个标记流的数据集，这些流来自于实时生成AI服务的会话，包括ChatGPT、Copilot和Gemini。

**📈 对比分析**

与八种模型家族进行了比较，结果显示树模型在L1和L2的表现优于深度学习模型，L1、L2和L3的加权F1分别为98.5%、86.4%和61.0%。

**⚠️ 局限性**

限制在于L3的识别能力受到文本和多媒体模式共享同一HTTP/2连接的特征重叠影响，导致其F1得分上限为61%。

---

## 609. How Much of SAR Vehicle Recognition Is the Vehicle? Background Shortcuts, Angle Gaps and Partial Libraries on MSTAR and ATRNet-STAR

**arXiv ID:** 2611.21827 | [PDF](https://arxiv.org/pdf/2611.21827v1)

**作者:** Mousa Shaya `[一作]` `[通讯]` (Shaya Software and Systems LLC), Mousa Shaya (Shaya Software and Systems LLC)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文研究了合成孔径雷达(SAR)车辆识别中，车辆本身对识别准确率的贡献。通过对MSTAR和ATRNet-STAR基准的实验，发现即使在去除车辆的情况下，网络仍能保持较高的准确率，表明背景信息对识别结果的影响。

**💡 创新点**

创新点在于通过背景控制和目标控制实验，揭示了背景对车辆识别的影响，并首次对SAR基础模型进行类似测试，发现其对背景的依赖性更强。此外，研究了不同俯仰角对识别准确率的影响，以及部分库效应对缺失类别的影响。

**🔧 技术方法**

使用了ConvNeXt-Small和SARATR-X两种深度学习网络，进行了一系列的实验以验证背景对识别的影响。

**📊 数据集**

使用了MSTAR和ATRNet-STAR数据集。MSTAR包含10种军事车辆的X波段图像，而ATRNet-STAR包含40种民用车辆的图像，涵盖多个场景和俯仰角。

**📈 对比分析**

通过与背景控制和目标控制的比较，发现去除背景后，准确率显著提高。在ATRNet-STAR的官方场景变化测试中，背景干扰导致准确率下降，表明背景信息在不同场景下的影响。

**⚠️ 局限性**

限制在于仅使用了幅度图像，没有考虑相位或极化信息；主要结果基于单一架构和训练方案；俯仰角仅以15度为步长，无法测量更小的角度差；部分库实验使用了单一折叠设计，可能影响结果的普遍性。

---

## 610. EviSlide-MIL: Multi-Query Pooling and Shared Prediction for Class-Incremental Whole-Slide Image Classification

**arXiv ID:** 2611.23212 | [PDF](https://arxiv.org/pdf/2611.23212v1)

**作者:** Zhouhan Qian `[一作]` (Duke University), Fei Xia `[通讯]` (University of California, Irvine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为EviSlide-MIL的类增量WSI分类框架，旨在在弱监督下进行肿瘤亚型的分类，同时保留先前学习的知识。

**💡 创新点**

创新点在于结合了多查询池化、参数高效的共享预测和重放监督的知识保留，解决了在增量学习中遗忘的问题。

**🔧 技术方法**

使用了多实例学习（MIL）技术，结合了经验重放和知识蒸馏的方法。

**📊 数据集**

使用了来自癌症基因组图谱（TCGA）的八种肿瘤亚型的数据集进行实验。

**📈 对比分析**

与现有方法相比，EviSlide-MIL在大多数学习设置中表现出更好的分类性能和更低的遗忘率，尤其是在F1分数和旧类召回率方面。

**⚠️ 局限性**

限制在于评估仅限于八种TCGA亚型和四个增量阶段，性能在特征表示和训练配置上可能有所不同，同时对存储和隐私的依赖也限制了适应性。

---

## 611. A Reidemeister-type theorem for vineyards

**arXiv ID:** 2611.23099 | [PDF](https://arxiv.org/pdf/2611.23099v1)

**作者:** Erin Chambers `[一作]` (University of Notre Dame), Mathijs Wintraecken `[通讯]` (Inria centre d'Universite Cote d'Azur)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

研究了闭合葡萄园在周期性滤波家族变化时的持久同调变化，探讨了观察点沿闭合观察循环移动时葡萄园的拓扑结构变化。

**💡 创新点**

提出了一个Reidemeister类型定理，表明在一般变形下，葡萄园的每个拓扑变化由有限的局部几何事件生成，并分类了这些事件。

**🔧 技术方法**

利用了奇点理论的工具，特别是广义对称集和几何判别式。

**📊 数据集**

主要集中在二维平面曲线的持久同调上，未具体提及使用的数据集。

**📈 对比分析**

通过与现有文献的比较，展示了局部几何事件如何影响葡萄园的拓扑变化，证明了在特定条件下葡萄园没有单调性和非平凡拓扑。

**⚠️ 局限性**

局限性在于研究主要集中在二维情况下，尚未扩展到更高维度的持久同调和多参数持久性设置。

---

## 612. DARWIN: a Geospatial Foundation Model for Marine Intelligence

**arXiv ID:** 2611.22496 | [PDF](https://arxiv.org/pdf/2611.22496v1)

**作者:** Coen van den Elsen `[一作]` (Ocean OS), Laurens Geffert `[通讯]` (Ocean OS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了DARWIN，一个用于海洋智能的地理空间基础模型，旨在解决海洋领域特有的挑战，如缺失数据、多模态物理和生物化学变量、稀疏标签以及动态变化。

**💡 创新点**

创新点在于采用了缺失感知编码、扩展的空间和时间掩蔽策略以及傅里叶风格的位置嵌入，适应海洋数据的结构特性。

**🔧 技术方法**

使用了掩蔽自编码器（Masked Autoencoder）作为骨干网络，并结合了视觉变换器（Vision Transformer）架构。

**📊 数据集**

使用了来自Copernicus海洋服务的26年全球重分析数据集，以及GEBCO的海底地形数据。

**📈 对比分析**

与无编码基线和随机初始化编码器进行比较，DARWIN在2022年每个月的相对误差最低，且在仅使用10%的监督训练数据时，已超越了使用100%数据的随机初始化模型。

**⚠️ 局限性**

限制在于模型主要依赖于L4重分析产品，未能充分利用直接卫星观测数据，且当前的架构假设所有输入共享单一的离散像素网格。

---

## 613. Beyond Acuity Accuracy: A Multidimensional Framework for Evaluating Patient-Facing Emergency Triage AI Tools

**arXiv ID:** 2611.22045 | [PDF](https://arxiv.org/pdf/2611.22045v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 614. EgoLink: Egocentric Language-Vision Interactive Network Knowledge Challenge Summary

**arXiv ID:** 2611.21690 | [PDF](https://arxiv.org/pdf/2611.21690v1)

**作者:** Renhe Sun `[一作]` (Ant Group), Jian Liu `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

EgoLink挑战旨在评估以用户为中心的多模态智能，结合第一人称感知、社会推理和交互行动，测试模型在情感和社会线索解读、视觉参考解析及用户请求转化为工具介导的行动的能力。

**💡 创新点**

创新点在于EgoLink作为一个基准，联合研究以用户为中心的社会认知和交互任务执行，并强调过程验证和基于证据的推理在可靠的多模态智能中的重要性。

**🔧 技术方法**

使用了多阶段的数据生成管道，包括多项选择题生成、思维链生成、自动质量评分和视频贡献过滤等技术。

**📊 数据集**

数据集包括E^3视频与EgoLink特定标签和多项选择题的配对，提供了4030个多项选择题和1055个以用户为中心的视频。

**📈 对比分析**

与现有方法相比，EgoLink的Track 1系统在准确率上达到了0.9263，Track 2系统的联合成功率为0.844，微观准确率为0.916，显示出较强的性能。

**⚠️ 局限性**

限制在于结果未能证明在发布场景之外的泛化能力，未来工作应加强证据基础和鲁棒性，特别是在未见场景和工具失败恢复方面。

---

## 615. Exploring Test Smells Across Programming Languages: A Systematic Mapping Study

**arXiv ID:** 2611.21613 | [PDF](https://arxiv.org/pdf/2611.21613v1)

**作者:** Publio Silva `[一作]` (Federal University of Ceará), Márcio Ribeiro `[通讯]` (Federal University of Alagoas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过系统性映射研究分析了测试气味与编程语言和测试框架之间的关系，审查了117篇相关论文，识别了与测试气味相关的主要编程语言、框架及其特征。

**💡 创新点**

创新点在于从编程语言的角度系统性地分析测试气味，填补了现有研究中对不同语言和框架的测试气味的关注不足，提供了跨语言的综合视角。

**🔧 技术方法**

采用了系统性映射研究方法，分析了文献中的数据，识别了213种不同的测试气味及其在11种编程语言和15种测试框架中的分布。

**📊 数据集**

研究中涉及的语言包括Java、Python、JavaScript、TypeScript、C#等，测试框架包括JUnit、PyTest、Jest等，共分析了117篇论文。

**📈 对比分析**

与现有方法相比，本研究提供了对测试气味的跨语言和跨框架的综合分析，发现Java是研究中最常提及的语言，而测试气味的关键性分析显示，Sleepy Test和Ignored Test被认为是最关键的气味。

**⚠️ 局限性**

限制在于研究主要集中在Java及其框架上，其他语言的研究相对较少，且对测试气味的关键性评估在不同研究中存在不一致性，未来需要更多的跨语言研究和工具开发。

---

## 616. Sleep2Word: Peripheral-to-EEG Translation via Duration-Aware Wordization

**arXiv ID:** 2611.21643 | [PDF](https://arxiv.org/pdf/2611.21643v1)

**作者:** Keondo Park `[一作]` (Seoul National University), Hyung-Sin Kim `[通讯]` (Seoul National University)

**通讯引用:** 2520 | [OpenAlex ID](https://openalex.org/A5065781070)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种从外周生理信号（如呼吸、心电图和光电容积描记）估计全夜脑电图（EEG）频谱图的框架，旨在扩展EEG相关的睡眠分析。

**💡 创新点**

创新点在于引入了基于持续时间的生理词汇，通过上下文依赖的分块方法，将生理嵌入分组为可变持续时间的块，并联合量化其内容和持续时间，形成特定模态的词汇表。

**🔧 技术方法**

使用了上下文依赖的分块、基于持续时间的生理词汇化和块自回归解码器等技术。

**📊 数据集**

使用了超过20,000个多导睡眠监测（PSG）案例，涵盖9个数据集进行训练，并在Cleveland Family Study（CFS）上进行验证。

**📈 对比分析**

与现有的频谱图生成基线方法相比，模型在EEG频谱图重建、睡眠分期、觉醒检测和生存预测等任务中表现出色，且在计算效率上具有优势。

**⚠️ 局限性**

限制在于模型可能无法捕捉到所有细粒度的EEG结构，且某些与睡眠阶段相关的信息可能在外周信号中不可用或在生成过程中丢失。

---

## 617. LLMs as Validators for Cross-Chain Bridges: Evaluating Source-Level and Transaction-Level Attack Detection on Solana

**arXiv ID:** 2611.21777 | [PDF](https://arxiv.org/pdf/2611.21777v1)

**作者:** Yasir Rahman `[一作]` (Chittagong University of Engineering and Technology), Anindya Bijoy Das `[通讯]` (University of Akron)

**通讯引用:** 1125 | [OpenAlex ID](https://openalex.org/A5029951257)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文实现了一个本地的Rust锁定和铸造桥，并构建了四个可利用的变种，评估了大型语言模型（LLMs）在源代码和交易记录中的漏洞检测能力。

**💡 创新点**

创新点在于通过使用LLMs来诊断Solana区块链桥的漏洞，特别是通过源代码和交易记录的分析来区分正确的漏洞诊断与错误警报。

**🔧 技术方法**

使用了大型语言模型（LLMs）作为源代码审计和交易记录监控的工具，评估了五种开放权重模型的六种配置。

**📊 数据集**

使用了本地Solana验证器实现的Rust锁定和铸造桥的源代码和交易记录数据集，共收集了16800个响应。

**📈 对比分析**

与现有方法相比，LLMs在检测漏洞时表现出显著差异：在没有注释的情况下，模型仅能诊断24%的漏洞源代码，而在恢复原始注释后，诊断率提高到60%。

**⚠️ 局限性**

限制在于LLMs在识别漏洞时容易产生误报，且在不同攻击类型的交易记录中，诊断能力存在差异，尤其是在缺失签名的情况下难以区分合法的管理员更改。

---

## 618. VITAL: A Framework for Validating Agentic AI in Pharmaceutical Drug Development and Clinical Trial Operations

**arXiv ID:** 2611.22973 | [PDF](https://arxiv.org/pdf/2611.22973v1)

**作者:** Sandeep Burugupalli `[一作]` (AstraZeneca), Raja Shankar `[通讯]` (IQVIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并实施了VITAL框架，用于验证药物开发和临床试验中的代理人工智能（AI），通过景观分析、法规和标准映射、专家访谈和多方利益相关者的修订。

**💡 创新点**

VITAL框架的创新点在于将验证从仅关注最终输出转向评估代理行为、执行轨迹和生命周期证据，强调中间步骤的重要性。

**🔧 技术方法**

使用了多种技术，包括景观分析、法规和标准映射、专家访谈，以及多层次的验证框架设计。

**📊 数据集**

使用了113个相关的文献来源，包括同行评审的文章、法规和标准文件、预印本和灰色文献。

**📈 对比分析**

与现有方法相比，VITAL框架通过风险调整的验证要求和证据输出，提供了更全面的验证策略，能够更好地应对代理AI的复杂性和不确定性。

**⚠️ 局限性**

限制在于风险调整的数值参数是示例性的，缺乏实证基础，且风险模型的结构可能无法充分捕捉各维度之间的相互作用。

---

## 619. A Log-Timescale Perturbation Bound for Selective State Space Models at Every Sequence Length

**arXiv ID:** 2611.21952 | [PDF](https://arxiv.org/pdf/2611.21952v1)

**作者:** Jack Humphries `[一作]` (Surrey), Daniel Midgley `[通讯]` (Surrey)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了在流数据（如限价单簿更新）上部署的Mamba模型的敏感性界限，提出了一种新的测量方法，确保小输入变化对隐藏状态的影响在每个序列长度上都保持在一定范围内。

**💡 创新点**

创新点在于采用对数尺度来测量时间尺度的变化，而不是传统的加法尺度，从而避免了误差的累积，并且在没有时间尺度下限的情况下，证明了在每个步骤之间的状态变化保持在一个界限内。

**🔧 技术方法**

使用了选择性扫描单元的构造和证明方法，结合了对数条件和状态更新的数学推导。

**📊 数据集**

使用了来自2021年1月28日的Databento MBP-10记录的限价单簿数据集，经过特征构建后，数据集包含约455万行和43个模型输入。

**📈 对比分析**

与Cao等人的方法进行了比较，发现对于大多数步骤，基于现有分析的界限在前8步内更紧，但在使用Mamba的softplus时间尺度时，界限在每个长度上都较松。实验表明，本文的界限在每个输入上都有效，但仍然高于梯度攻击找到的变化量。

**⚠️ 局限性**

限制在于该定理仅适用于标量单元，要求对角矩阵A，并且需要真实的转换在(0, 1)之间。此外，未证明从共同起点的最大间隙是否等于η + 2Mtanh(ρ/2)。

---

## 620. Latent Fact Composition under Next-Token Training

**arXiv ID:** 2611.21921 | [PDF](https://arxiv.org/pdf/2611.21921v1)

**作者:** Tianyi Zhang `[一作]` (University of California San Diego), Feiying Chen `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了在下一个标记训练下，模型是否能够回答中间实体未出现在训练示例中的两跳问题，并提出了一种新的循环变换器架构Baton，能够在仅预测答案的训练下几乎完美地回答这些问题。

**💡 创新点**

创新点在于提出了Baton系列循环变换器，该模型通过将中间实体作为标记嵌入传递到第二个循环，从而实现了OOD（超出训练分布）问题的泛化。

**🔧 技术方法**

使用了循环变换器架构，特别是Baton系列模型，并进行了对比实验。

**📊 数据集**

构建了一个合成语料库，其中包含500个实体和50,000个两跳链的句子，训练集包含所有10,000个原子事实和10,000个随机抽取的两跳句子。

**📈 对比分析**

与标准变换器和其他基线模型进行比较，Baton模型在OOD问题上的准确率显著高于其他模型，尤其是在仅预测答案的训练下，准确率接近1.00，而其他模型在OOD问题上的准确率始终低于0.012。

**⚠️ 局限性**

限制在于研究结果仅基于一个合成设置，且模型较小，可能无法推广到自然文本中，此外，模型的OOD准确率在第二跳时下降，尽管在训练过程中仍能回忆起两个事实。

---

## 621. Thinking Harder Doesn't Lead to Better Choices: Reasoning Effort Increases the Confidence but Not the Correctness of LLM Shopping Agents

**arXiv ID:** 2611.22033 | [PDF](https://arxiv.org/pdf/2611.22033v1)

**作者:** Adi Krishnamoorthy `[一作]` (University of Chicago), X. Y. Han `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了大型语言模型（LLMs）作为自主购物代理的表现，探讨了增加推理努力是否能提高产品选择质量，以及模型报告的信心是否能反映其错误的可能性。

**💡 创新点**

创新点在于揭示了增加推理努力并未显著提高购物决策的准确性，而是提高了模型的自信心。此外，研究还发现模型之间的错误高度相关，多个模型在同一任务上往往选择相同的错误产品。

**🔧 技术方法**

使用了十个前沿的LLM模型，通过OpenRouter API进行查询，评估了它们在不同推理努力水平下的表现。

**📊 数据集**

使用了三个数据集：WebShop（包含图像和文本的多模态产品列表）、Amazon的SQID（结合图像和文本的购物查询数据集）和Wayfair的WANDS（仅包含文本的基准数据集）。

**📈 对比分析**

通过比较不同模型在相同任务下的成功率和信心，发现大多数模型在推理努力增加时，成功率变化不显著，但信心显著提高。大部分模型在所有努力水平下的校准效果较差。

**⚠️ 局限性**

限制在于研究任务仅集中于决策部分，未测试搜索、分页或多步骤导航等其他可能影响结果的因素。此外，数据集的标签存在一定的噪声，可能影响模型的绝对准确性和错误相关性。

---

## 622. Simplicity Goes Far: Auditing Prompt Optimizers with Demonstration-Conditioned Prompt Search

**arXiv ID:** 2611.22605 | [PDF](https://arxiv.org/pdf/2611.22605v1)

**作者:** Bohe Chen `[一作]` `[通讯]` (Beijing University of Posts and Telecommunications), Bohe Chen (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的自动提示优化方法，称为示范条件提示搜索（DCPS），用于审计现有的提示优化器，特别是通过生成和选择步骤来评估优化器机制的贡献。

**💡 创新点**

创新点在于将DCPS作为一种控制机制，用于隔离和审计优化器的复杂性，提供了一种可重复的审计方法，并在不同的优化器设置中进行验证。

**🔧 技术方法**

使用了示范条件提示搜索（DCPS）技术，该技术通过冻结生成器并基于随机示范生成候选提示，评估其在验证数据上的表现。

**📊 数据集**

使用了多个数据集，包括GLUE/SuperGLUE、MMLU和指令诱导等任务，进行不同的实验和比较。

**📈 对比分析**

与现有的优化器（如PPO和GEPA）进行比较，DCPS在分类准确性上与PPO相当，但在成本上显著降低，且在某些任务上表现出更高的成本效率。

**⚠️ 局限性**

限制在于DCPS无法定位级联错误，且在不同设置下的比较可能不完全一致，此外，尚未测试代码生成、摘要和更长的交互任务等复杂场景。

---

## 623. Synchrony, Affect, and Perceived Autonomy in Virtual Robot-Facilitated Anxiety Support Groups

**arXiv ID:** 2611.21977 | [PDF](https://arxiv.org/pdf/2611.21977v1)

**作者:** A'di Dust `[一作]`, Maja Matarić `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究探讨了在虚拟机器人辅助的焦虑支持小组中，参与者之间的头部运动同步性与情感和感知自主性之间的关系。

**💡 创新点**

创新点在于首次将头部运动同步性作为评估机器人辅助小组互动质量的客观行为指标，并发现其与治疗结果之间的关系与人类辅助小组相似。

**🔧 技术方法**

使用了Wizard-of-Oz (WoZ) 系统来控制虚拟机器人，并通过窗口交叉相关和峰值选择等技术来测量头部运动同步性。

**📊 数据集**

数据集由96名参与者在18个虚拟机器人辅助的焦虑支持小组会议中收集，参与者的情感状态通过自我报告问卷进行评估。

**📈 对比分析**

与传统人类辅助小组相比，机器人辅助小组的互动动态显示出相似的治疗结果关联，强峰值同步性与较低的负面情感相关，而跟随行为则与较高的负面情感和状态焦虑相关。

**⚠️ 局限性**

限制在于使用的支持小组主持人并非持证治疗师，且研究仅限于单次会议，可能存在新奇效应，结果的普遍性受到限制。

---

## 624. RFW-S: Everyday Deformable Objects with Independent Simulation Verification

**arXiv ID:** 2611.23056 | [PDF](https://arxiv.org/pdf/2611.23056v1)

**作者:** Jiarui Guo `[一作]` (Alibaba Group), Chaojian Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了RoboFlyWheel-Soft (RFW-S)，一个包含83,365个经过模拟验证的日常可变形物体资产的数据集，涵盖485个类别，旨在支持机器人与可变形物体的交互研究。

**💡 创新点**

创新点在于将广泛的语义覆盖与可重用的物理配置相结合，并通过独立的行为验证来确保资产的有效性，解决了现有资源在物理行为验证方面的不足。

**🔧 技术方法**

使用了结构感知建模、语言模型辅助的参数估计和模拟引导的校准技术，并结合独立的交互测试进行验证。

**📊 数据集**

数据集包含83,365个资产，涵盖485个类别，涉及软固体、布料、壳体、服装和袋子等五个结构类别。

**📈 对比分析**

通过控制物理测试、机器人抓取和模拟到现实的实验来评估RFW-S，结果表明，仅依靠材料调节无法替代结构适应，校准成功并不保证在不同交互下的有效行为。

**⚠️ 局限性**

限制在于某些类别的可用覆盖稀疏，且独立测试揭示了校准接受后的响应不匹配，未来需要更多的物理测量和反馈来增强数据集的质量和可靠性。

---

## 625. Classification of Tissues on MALDI-MSI Image Modality for Pancreatic Cancer Studies

**arXiv ID:** 2611.22680 | [PDF](https://arxiv.org/pdf/2611.22680v1)

**作者:** Abubakar Qahir `[一作]`, Marion Marty `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种全自动化的管道，将原始的MALDI-MSI数据转换为适合深度学习的图像补丁，用于二元胰腺组织分类（肿瘤与非肿瘤）。

**💡 创新点**

创新点在于实现了从原始MALDI-MSI数据到深度学习模型的全自动化处理流程，并为每位患者构建了个体模型以识别肿瘤区域。

**🔧 技术方法**

使用了Swin Transformer模型进行图像分类，并进行了自动化预处理。

**📊 数据集**

使用了两个数据集：第一个是从胰腺癌患者的MALDI-MSI图像生成的数据集，第二个是公开的Barrett食管数据集。

**📈 对比分析**

与参考交互工作流程相比，提出的管道在图像质量指标（均方误差MSE）上表现相近，计算时间平均为20分钟，而参考工作流程需要一天的时间。模型在高分辨率光谱上达到了0.94的准确率，在低分辨率光谱上平均准确率为0.59。

**⚠️ 局限性**

限制在于数据集的空间和光谱分辨率较低，导致在第二个数据集上的性能下降。

---

## 626. TUSI: A Citation-Grounded Graph-Based Benchmark

**arXiv ID:** 2611.21823 | [PDF](https://arxiv.org/pdf/2611.21823v1)

**作者:** Marzieh Karimi `[一作]` (Isfahan University of Technology), Alireza Basiri `[通讯]` (Isfahan University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TUSI-Bench，一个用于评估检索增强生成（RAG）系统的波斯法律基准，旨在解决现有基准在真实感、基础和法律推理覆盖方面的不足。

**💡 创新点**

TUSI-Bench是第一个结构化的波斯法律基准，包含真实的律师-客户咨询数据，提供了基于引用的开放式问答和多项选择题的评估任务。

**🔧 技术方法**

构建了一个法律知识图谱，支持结构感知的检索，并提出了一种基于图的检索基线，利用法律知识图谱中的关系结构。

**📊 数据集**

使用了7261条法律条款的数据集，整合了法典、法律评论、学说著作和最高法院统一裁决。

**📈 对比分析**

与标准基线相比，基于图的检索方法提高了检索召回率最多8个百分点，显示出结构化法律表示对RAG系统的重要性。

**⚠️ 局限性**

构建高质量法律知识图谱的过程复杂，错误或不完整的关系链接可能会影响检索过程的准确性。

---

## 627. From Preference Data to Personalization: Tracing Sycophancy in Large Language Models

**arXiv ID:** 2611.21650 | [PDF](https://arxiv.org/pdf/2611.21650v1)

**作者:** Himaghna Bhattacharjee `[一作]` (Meta), Claudia Shi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大型语言模型中的谄媚行为，特别是当用户的主张需要纠正时，模型不当的同意行为。提出了一种有效性意识的谄媚评分器和SycoDrift评估集，以分析人类偏好数据、自动评判规范和上下文个性化对模型同意行为的影响。

**💡 创新点**

创新点在于引入了一种有效性意识的测量协议，避免将合理的同意视为失败，并构建了SycoDrift评估集，提供了受控的有效性、开放性和风险注释。

**🔧 技术方法**

使用了有效性意识的评分器和SycoDrift评估集，结合了人类偏好数据、自动评判规范和个性化上下文的分析。

**📊 数据集**

使用了SycoDrift评估集，该集包含3,298个有效性注释的探针，涵盖了多种个性化情境和用户偏好。

**📈 对比分析**

通过对14个模型的离线评估，发现个性化上下文（如匹配的用户档案）显著增加了不当同意的发生率，尤其是在用户档案直接放置在查询之前时，增加幅度达到6.97个百分点。

**⚠️ 局限性**

本研究的局限性包括偏好数据研究未对模型进行训练，RLAIF研究仅使用了一个基础模型和一种训练算法，个性化研究则使用了合成角色和基准探针，测量的是基准内的敏感性而非实际部署率。

---

## 628. Distortion-Free Semantic Watermarks for Black-Box Generative Models

**arXiv ID:** 2611.23208 | [PDF](https://arxiv.org/pdf/2611.23208v1)

**作者:** Aviv Yossef `[一作]` (Tel Aviv University), Or Zamir `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种无失真语义水印生成算法，旨在提高水印的检测可靠性和鲁棒性。

**💡 创新点**

创新点在于通过增加语义熵来增强检测保证，并且通过独立生成的输出聚合来显著降低检测错误。

**🔧 技术方法**

使用了语义编码器和黑箱生成器的组合，采用了Gumbel采样等技术。

**📊 数据集**

使用了多种图像生成器生成的数据集，并进行了文本到音频输出的水印评估。

**📈 对比分析**

通过与现有的编码器和采样规则进行比较，展示了所提编码器在水印检测中的优势，且Gumbel采样在检测性能上优于其他选择规则。

**⚠️ 局限性**

局限性包括黑箱采样的固有限制，计算成本，固定密钥重用，适应性攻击，以及超出图像的扩展性问题。

---

## 629. What's in a Tree ? Hierarchical $n$-gram models as controlled approximators of language

**arXiv ID:** 2611.23093 | [PDF](https://arxiv.org/pdf/2611.23093v1)

**作者:** Aritra Das `[一作]` (University of Maryland), Andrey Gromov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种自回归生成n-gram模型，通过层次化的顺序-2和顺序-3张量组合构建条件概率张量p(x_n | x_1:n-1)，以研究语言的合成设置。

**💡 创新点**

创新点在于引入了可控的近似模型，通过超参数n和d独立控制模型的上下文窗口和容量，同时能够在性能上与深度变换器模型竞争。

**🔧 技术方法**

使用了层次化的n-gram模型，结合了顺序-2和顺序-3张量近似、归一化层、残差连接和多层感知器（MLP）等技术。

**📊 数据集**

在TinyStories（一个529M标记的合成语言语料库）、WikiText-103（130M标记）和FineWeb-Edu（1.05BT标记）数据集上进行了训练。

**📈 对比分析**

与现有的n-gram基线进行比较，结果显示在小的n值下表现优于传统n-gram估计器，但在较长上下文中仍落后于深度变换器。通过增加容量参数d，可以缩小与LLM的性能差距。

**⚠️ 局限性**

局限性包括架构选择的最优性不明确、仅考虑了狭窄的拓扑类、计算限制导致无法扩展到更长的上下文窗口，以及对不同架构组件的理解有限。

---

## 630. Anticipating or Speculating? Making Futures Pay in AI Ethics

**arXiv ID:** 2611.22268 | [PDF](https://arxiv.org/pdf/2611.22268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 631. PeriaTrace: Tooth-Level Grounding Diagnosis and Patient-Facing Explanations from Periapical Radiographs

**arXiv ID:** 2611.23012 | [PDF](https://arxiv.org/pdf/2611.23012v1)

**作者:** Zhenhuan Zhou `[一作]` (Nankai University), Tao Li `[通讯]` (Nankai University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了PeriaTrace，一个集成的三阶段框架，用于从根尖X光片中进行牙齿级别的诊断、证据基础的报告生成和面向患者的沟通。

**💡 创新点**

创新点在于将牙齿级别的诊断与可视化证据结合，生成结构化的临床报告，并提供患者友好的解释和护理建议。

**🔧 技术方法**

使用了深度学习技术，包括多任务视觉先验提取、定位条件的诊断推理和证据基础的报告生成。

**📊 数据集**

使用了一个内部数据集MM-PRAD，包含12,877张根尖X光片，以及两个额外的临床队列。

**📈 对比分析**

与六个基线模型进行比较，PeriaTrace在报告生成和视觉基础方面表现优越，BLEU-4得分为0.6504，显著高于其他模型，且生成报告的速度比手动撰写快数十倍。

**⚠️ 局限性**

局限性包括所有数据集均来自同一地理区域，可能影响结果的普遍适用性；此外，系统可能会产生不准确或不完整的结果，需由合格的牙科专业人员进行审核。

---

## 632. Colombo: A Frontier-to-Frontier Autoresearch System for Oracle-Based Optimization

**arXiv ID:** 2611.23080 | [PDF](https://arxiv.org/pdf/2611.23080v1)

**作者:** Haihan Zhang `[一作]` (Peking University), Zhouchen Lin `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种自组织的人工智能研究系统Colombo，旨在通过信息接口高效解决问题类，并记录复杂性边界。

**💡 创新点**

创新点在于将自组织和对抗共升机制结合，形成一个动态的研究框架，能够自我更新和适应新的研究前沿。

**🔧 技术方法**

使用了自组织和对抗共升的技术，结合了复杂性地图和适应性研究程序。

**📊 数据集**

使用了复杂性地图（Complexity Atlas）作为数据集，涵盖了超过五十个设置的最优或近最优速率。

**📈 对比分析**

通过与传统算法的比较，Colombo在多个设置中实现了最优或近最优的复杂性保证，性能显著提升。

**⚠️ 局限性**

限制在于当前系统主要集中于oracle-based优化，可能在其他领域的适用性和扩展性尚待验证。

---

## 633. VoxelSBFL: Voxel-Level Spectrum-Based Failure Prediction for Vision-Language-Action Models

**arXiv ID:** 2611.22453 | [PDF](https://arxiv.org/pdf/2611.22453v1)

**作者:** Arusa Kanwal `[一作]` (Mondragon University), Aitor Arrieta `[通讯]` (Mondragon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为VoxelSBFL的黑箱适应性谱基故障定位方法，用于评估视觉-语言-动作（VLA）模型的执行可靠性，特别是通过分析机器人在工作空间中的轨迹来预测失败。

**💡 创新点**

创新点在于将工作空间的体素视为分析单元，结合执行结果构建可疑性图，从而识别与失败相关的空间区域，而无需访问模型内部信息。

**🔧 技术方法**

使用了谱基故障定位（SBFL）技术，结合体素覆盖信息和执行结果，构建可疑性图以评估轨迹的成功与失败。

**📊 数据集**

使用了96,000条在模拟环境中生成的Pick-Up轨迹，涵盖四种不同的VLA模型。

**📈 对比分析**

与34种SBFL技术进行比较，VoxelSBFL在所有模型中获得了0.704的宏ROC-AUC，显著优于现有的基线方法，且在空间粒度和训练集大小的敏感性分析中表现出性能变化。

**⚠️ 局限性**

限制在于VoxelSBFL的性能依赖于体素大小和训练集的覆盖程度，且在不同任务间的迁移能力有限，可能无法在新任务中保持相同的效果。

---

## 634. Large Language Models Have Not Yet Transformed Science

**arXiv ID:** 2611.22975 | [PDF](https://arxiv.org/pdf/2611.22975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 635. What Does a Bias Head Measure? Non-Identifiability in Head-Level Bias Attribution

**arXiv ID:** 2611.23218 | [PDF](https://arxiv.org/pdf/2611.23218v1)

**作者:** S. M Seefat Alam `[一作]` `[通讯]` (Bangladesh University of Engineering and Technology), S. M Seefat Alam (Bangladesh University of Engineering and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文审计了针对大型语言模型中社会偏见的头部归因统计方法，特别是对Qwen-2.5-7B-Instruct和Llama-3.1-8B-Instruct模型的验证，探讨了这些方法是否真正测量了偏见，并评估了消融操作的效果。

**💡 创新点**

创新点在于提出了一种控制图，揭示了现有方法的误导性，并通过干预实验验证了这一点，表明头部归因统计并不如预期那样有效地测量社会偏见。

**🔧 技术方法**

使用了对比性头部公平性归因的方法，结合了不同的提示条件（直接回答和逐步思考），并进行了干预实验以打破长度与标签之间的耦合。

**📊 数据集**

使用了Qwen-2.5-7B-Instruct和Llama-3.1-8B-Instruct模型，进行了60个问题的识别实验，并使用了80个平衡的保留问题进行评估。

**📈 对比分析**

与现有方法相比，本文的方法通过干预实验显示，消融操作并未减少不公平率，反而在某些情况下导致生成质量下降，表明现有方法的有效性存在问题。

**⚠️ 局限性**

限制在于使用的识别集较小（60个问题），且依赖于单一作者的标注，缺乏多标注者间的一致性验证。此外，长度假设的测试仅在一个生成子家族中进行，未能全面验证其他模型的适用性。

---

## 636. A 4.542-approximation for the Small Cuts Cover problem via a multi-rate primal-dual algorithm

**arXiv ID:** 2611.22910 | [PDF](https://arxiv.org/pdf/2611.22910v1)

**作者:** Zeev Nutov `[一作]` (Open University of Israel), Elazar Koren `[通讯]` (Open University of Israel)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了小切割覆盖问题，提出了一种新的多速率原始对偶算法，改进了该问题的近似比率。

**💡 创新点**

创新点在于首次使用多速率策略来提高标准线性规划的近似保证，打破了传统均匀增长的限制。

**🔧 技术方法**

采用了多速率原始对偶算法，该算法对不同的核心以不同的速率提升对偶变量。

**📊 数据集**

使用了小切割覆盖问题的相关数据集，具体数据集未详细说明。

**📈 对比分析**

与经典的原始对偶算法相比，新的算法在近似比率上从5改进到(3+√(37))/2 < 4.542，显示出显著的性能提升。

**⚠️ 局限性**

限制在于该方法的有效性依赖于输入集合的结构特性，可能不适用于所有网络设计问题。

---

## 637. MIPR: A Benchmark for Reasoning-Intensive Multimodal Intent-to-Product Retrieval with Multi-Category Solutions

**arXiv ID:** 2611.22509 | [PDF](https://arxiv.org/pdf/2611.22509v1)

**作者:** Qiang Li `[一作]` (Tongji University), Junjun Zheng `[通讯]` (Taobao and Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MIPR基准，旨在评估多模态意图到产品的检索，包含3522个多模态查询和30731个产品。

**💡 创新点**

MIPR是首个同时支持推理密集检索、多类别解决方案和细粒度相关性标签的基准。

**🔧 技术方法**

使用了多模态嵌入检索器和重排序模型，评估了多种现有的嵌入模型。

**📊 数据集**

基于真实的中国电子商务搜索日志构建，包含3522个查询和30731个产品。

**📈 对比分析**

与现有基准相比，MIPR在多类别解决方案的覆盖和相关性排名上表现不佳，最佳嵌入检索器在nDCG@10上仅达到49.23%。重排序显著提高了性能，但仍有较大改进空间。

**⚠️ 局限性**

现有模型在检索最优产品和覆盖多个解决方案类别方面存在显著挑战，且MIPR的构建和评估方法尚需进一步完善。

---

## 638. Bajillion: Cryptographic Clearing for Micropayments

**arXiv ID:** 2611.21883 | [PDF](https://arxiv.org/pdf/2611.21883v1)

**作者:** Patrick O'Grady `[一作]` (Commonware), Guru-Vamsi Policharla `[通讯]` (Commonware)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一种名为Bajillion的清算协议，用于可扩展的微支付，允许发送者为其消费支付，确保恶意操作员只能省略支付，而不能超额扣款或错误转账。

**💡 创新点**

Bajillion的创新点在于其设计允许发送者在不需要争议的情况下进行微支付，且能够在每个时期结束时立即确认余额，适合短暂的AI代理使用。

**🔧 技术方法**

使用了加密清算机制，结合了数字签名和Merkle树等密码学原语来验证和认证支付。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在大规模网络流量下的微支付场景。

**📈 对比分析**

与现有的链下扩展机制相比，Bajillion能够独立处理来自不同发送者的请求，允许操作员根据发送者的数量进行并行处理，性能上能够支持每秒数亿次微支付。

**⚠️ 局限性**

Bajillion的局限性在于它不提供对恶意操作员的活跃性保证，用户必须能够随时提取资金并退出清算协议，即使操作员不响应或故意阻止提款。

---

## 639. From Perception to Simulation: Unveiling and Empowering Mental Spatial Dynamics in MLLMs

**arXiv ID:** 2611.22541 | [PDF](https://arxiv.org/pdf/2611.22541v1)

**作者:** Jiachun Li `[一作]` (University of Chinese Academy of Sciences), Jun Zhao `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MentalGrid基准，专注于动态空间推理任务，要求模型在没有中间视觉反馈的情况下模拟不断变化的空间状态。

**💡 创新点**

创新点在于引入了一个新的基准和三阶段课程学习框架，显著提升了模型在空间模拟任务中的表现。

**🔧 技术方法**

使用了三阶段课程学习框架和强化学习技术来优化空间状态的演变。

**📊 数据集**

使用了MentalGrid基准数据集，该数据集包含3356个问题和1100个模拟图像，涵盖349个不同场景。

**📈 对比分析**

与现有模型（如Gemini-3.1-Pro和Gemini-3-Flash）进行比较，SpatialWalker模型在空间模拟任务中平均提高了33.8%，并在域外基准上保持了4.9%的性能提升。

**⚠️ 局限性**

限制在于未考虑复杂的物理因素（如物体姿态、材料和碰撞），并且所有场景均来自模拟环境，缺乏真实世界数据的支持。

---

## 640. PCGRL for Complex Platformers: Synthesizing Playable and Diverse Levels via Inflated Envelopes and Forward Curricula

**arXiv ID:** 2611.22250 | [PDF](https://arxiv.org/pdf/2611.22250v1)

**作者:** Di Wu `[一作]` `[通讯]` (Harbin Institute of Technology), Di Wu (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种物理感知的课程引导的程序内容生成框架，旨在解决现有程序内容生成框架在长序列生成任务中的探索瓶颈和扩展限制。

**💡 创新点**

创新点在于引入了'膨胀体积包络'机制，将连续运动可行性与离散状态空间结合，同时采用课程学习策略和多阶段反馈机制来改善稀疏奖励问题。

**🔧 技术方法**

使用了物理模拟、马尔可夫决策过程（MDP）、课程学习和高熵随机推断等技术。

**📊 数据集**

在一个物理约束的2D平台生成环境中进行实验，该环境包含刚体碰撞约束和非线性跳跃动态。

**📈 对比分析**

与传统的PCGRL基线进行比较，结果显示该框架在不同画布规模上保持了更好的收敛稳定性和维度可扩展性，能够可靠地合成可解的关卡，而传统方法在大规模环境中性能下降。

**⚠️ 局限性**

限制在于当前实验验证仅限于25×16的画布分辨率，且未考虑动态障碍物或多房间的全局过渡，扩展动作空间以包含更丰富的视觉语义仍然是一个挑战。

---

## 641. PhyVGGT: Feed-Forward Articulated Object Reconstruction from Monocular Videos

**arXiv ID:** 2611.22244 | [PDF](https://arxiv.org/pdf/2611.22244v1)

**作者:** Qi Shao `[一作]` (Harbin Engineering University), Liang Li `[通讯]` (Harbin Engineering University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为PhyVGGT的单次框架，从未校准的单目视频中重建具有3D可动部件、关节参数和未见关节状态的关节物体。

**💡 创新点**

创新点在于将部件匹配和闭式关节估计锚定到静态基座的几何形状，从而提高了重建的准确性和效率。

**🔧 技术方法**

使用了冻结的3D基础模型，结合了多层令牌融合和基于曼哈顿框架的关节估计。

**📊 数据集**

训练数据集为合成数据，评估使用了多个真实世界和合成视频基准，包括Arti4D、D3D-HOI和PartNet-Mobility。

**📈 对比分析**

与现有方法（如FAMOS和单目ArtiPoint）相比，PhyVGGT在五个基准测试中实现了最先进的单目性能，运行时间为每48帧视频6.8秒，而其他方法需要数小时的场景优化。

**⚠️ 局限性**

限制在于被动RGB帧无法揭示静态或被遮挡的部件或物理参数，如质量、摩擦、弹簧和锁扣，这些将在未来的工作中通过交互操作和力传感来恢复。

---

## 642. KVUse: Reusable-KV User Memory for Efficient LLM Recommenders

**arXiv ID:** 2611.21920 | [PDF](https://arxiv.org/pdf/2611.21920v1)

**作者:** Xin Guo `[一作]` (Meta Recommendation Systems), Lizhu Zhang `[通讯]` (Meta Recommendation Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为KVUse的可扩展记忆增强推荐系统，通过存储用户的个人资料为可重用的KV而不是文本，来提高推荐系统的效率。

**💡 创新点**

创新点在于通过训练可重用的KV写入器，避免了重复计算，从而在保持用户资料和排名质量的同时，提高了下游评分路径的吞吐量。

**🔧 技术方法**

使用了LoRA适配器来训练可重用的KV写入器，并设计了KV调度器来协调KV的上传和LLM推理。

**📊 数据集**

使用了四个数据集进行评估：Amazon Books、Goodreads、Amazon Movies & TV和Yelp。

**📈 对比分析**

与传统的完全预填充方法相比，KVUse在下游评分路径的吞吐量提高了4.0-4.5倍，同时保持了排名质量，NDCG@10的差异在统计上不显著。

**⚠️ 局限性**

限制在于写入器需要为每个数据集单独训练，且KV调度器需要提前知道下一个批次的请求，存储需求较高。

---

## 643. 3D-VASE: Volumetric-Aware Scalable and Efficient Multi-Layer Design Synthesis for 3D-Printed Microfluidics

**arXiv ID:** 2611.22983 | [PDF](https://arxiv.org/pdf/2611.22983v1)

**作者:** Yushen Zhang `[一作]` (Technical University of Munich), Ulf Schlichtmann `[通讯]` (Technical University of Munich)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种名为3D-VASE的两阶段合成框架，用于多层3D打印微流体芯片的设计，旨在解决传统方法在复杂设计中的性能瓶颈。

**💡 创新点**

创新点在于将全局布局与详细路由解耦，采用混合整数规划（MIP）模型进行紧凑的多层布局计算，并引入动态3D路由引擎以满足时序和体积约束。

**🔧 技术方法**

使用混合整数规划（MIP）模型进行全局布局估算，并结合基于启发式搜索的3D路由算法。

**📊 数据集**

使用多个测试案例进行实验评估，涵盖从简单到复杂的多层微流体设计。

**📈 对比分析**

与现有的3M-DeSyn方法进行比较，3D-VASE在所有测试规模上均表现出显著的运行时间优势，并在许多情况下提供了更高质量的布局。

**⚠️ 局限性**

局限性在于该框架的验证是在理想的稳态泵送条件下进行的，未来需要进一步的物理实验来量化泵的不理想性和制造变异的影响。

---

## 644. An upper bound on non-overlapping codes

**arXiv ID:** 2611.21913 | [PDF](https://arxiv.org/pdf/2611.21913v1)

**作者:** Simon R. Blackburn `[一作]` `[通讯]` (Royal Holloway University of London), Simon R. Blackburn (Royal Holloway University of London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文提供了一个关于非重叠编码的上界的自包含证明，证明了对于长度为n的二进制非重叠编码X，|X|≤(5/6e +O(n^-1))2^n/n，随着n趋向于无穷大。

**💡 创新点**

创新点在于改进了Levenshtein在1970年提出的上界，提升了界限的常数因子，从1/e改为5/6e。

**🔧 技术方法**

使用了分析组合学的技术，特别是通过整数优化问题来提供上界，并结合了代数组合学的技巧。

**📊 数据集**

论文中没有具体提到使用的数据集，因为其主要是理论研究，关注于编码的数学性质。

**📈 对比分析**

与Levenshtein的界限进行比较，新的界限在常数因子上有所提升，表明改进的界限在理论上更为紧密。

**⚠️ 局限性**

限制在于该结果主要针对二进制非重叠编码，且对于更大字母表的情况可能需要进一步的研究。

---

## 645. ATLAS-Finance: Benchmarking AI Agents in Realistic Banking Environments

**arXiv ID:** 2611.22085 | [PDF](https://arxiv.org/pdf/2611.22085v1)

**作者:** Elaine Lau `[一作]`, Jonas Mueller `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了ATLAS-Finance基准，旨在评估代理在投资银行工作流程中的表现，特别是在处理不完整指令和从多个应用程序中提取信息的能力。

**💡 创新点**

创新点在于创建了一个高保真度的投资银行工作环境，模拟真实的工作场景和复杂性，提供了全面的评分标准以评估代理的输出。

**🔧 技术方法**

使用了强化学习环境Harbor框架，结合了多种应用程序（如Outlook、Slack、Google Drive等）进行任务执行和信息检索。

**📊 数据集**

使用了ATLAS-Finance数据集，该数据集包含100个投资银行任务，涵盖多个模拟公司和真实的交易材料。

**📈 对比分析**

与其他基准相比，ATLAS-Finance在任务复杂性和评分标准上更为深入，评估结果显示，Claude Opus 5模型在通过率和评分上表现最佳，但整体通过率仍然较低，表明代理在处理复杂任务时存在困难。

**⚠️ 局限性**

限制在于ATLAS-Finance使用的是模拟工作环境，无法完全反映真实银行工作中的复杂性，且评估未考虑个别能力的影响，如处理冲突信息或任务更新的能力。

---

## 646. On Formal Counterfactual Explanations

**arXiv ID:** 2611.22010 | [PDF](https://arxiv.org/pdf/2611.22010v1)

**作者:** Alexey Ignatiev `[一作]` (Monash University), Joao Marques-Silva `[通讯]` (ICREA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过明确地将最近世界的反事实解释表征为所有最小距离反例实例的析取范式，并引入了限制为特征文字的合取反事实解释（CFXps），将反事实解释与特征基础的可解释性AI（XAI）联系起来。

**💡 创新点**

创新点在于引入了合取反事实解释（CFXps），这是一种高效可计算的反事实前提子类，保证预测变化，并且与现有的可解释性AI方法（AXps和CXps）建立了结构性联系。

**🔧 技术方法**

使用了形式逻辑和模型逻辑的方法，结合了特征基础的反事实解释，采用了MaxSAT和MILP作为推理后端。

**📊 数据集**

实验使用了来自UCI机器学习库和Penn机器学习基准的21个公共数据集，特征数量从7到60不等，实例数量从59到58000不等。

**📈 对比分析**

与流行的基于点的反事实解释器相比，CFXps通常不需要更多的特征变化，并且可以保证反事实预测超出单个实例。CFXps的提取在计算上与单个AXp或CXp相当，而最近世界反事实解释需要耗时的穷举枚举。

**⚠️ 局限性**

限制在于CFXps的计算依赖于现有的AXp和CXp机制，尽管在效率上有所提升，但在某些情况下仍可能面临计算复杂性的问题。

---

## 647. Evolve in Panorama, Render in View: Panoramic World State Evolution for Video World Models

**arXiv ID:** 2611.22733 | [PDF](https://arxiv.org/pdf/2611.22733v1)

**作者:** Longtao Jiang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xiaojun Chang `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的视频世界模型，能够在全景视频扩散模型的潜在空间中演变世界状态，并通过透视视频扩散模型渲染视图。

**💡 创新点**

创新点在于模型能够在视角变化时持续演变场景，而不是在视角外的内容被冻结或消失。

**🔧 技术方法**

使用了全景视频扩散模型和透视视频扩散模型的组合，采用潜在空间投影和方向限制的交叉注意力机制来渲染高质量视图。

**📊 数据集**

构建了一个基于真实360度视频的事件中心数据集，包含约20,812个剪辑用于训练。

**📈 对比分析**

与23个公开评估模型进行比较，在WRBench上获得最佳整体得分（0.753），在STEVO-Bench上获得最高总分（364.7），显示出优越的性能。

**⚠️ 局限性**

模型的局限性在于只能处理围绕固定中心的相机旋转，无法处理相机平移带来的视差和新可见表面。

---

## 648. AlignedHPO: Efficient Surrogate-Guided Loss Hyperparameter Optimization

**arXiv ID:** 2611.22658 | [PDF](https://arxiv.org/pdf/2611.22658v1)

**作者:** Ivan Karpukhin `[一作]` (Sber AI Lab), Andrey Savchenko `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为AlignedHPO的高效框架，用于通过梯度对齐和替代验证目标优化可微损失超参数。

**💡 创新点**

创新点在于结合局部梯度对齐与雅可比-向量乘积，避免了昂贵的轨迹微分和逆海森矩阵近似，同时支持非线性损失参数化。

**🔧 技术方法**

使用了雅可比-向量乘积（JVP）技术来高效估计超梯度，并引入了梯度平滑以提高优化的稳定性。

**📊 数据集**

在长尾分类（CIFAR-10-LT和CIFAR-100-LT）和开放集人脸识别（CASIA-WebFace）数据集上进行了实验。

**📈 对比分析**

与隐式微分和传统的反向-反向微分方法进行了比较，AlignedHPO在超梯度计算时间上快了近12倍，并在性能上表现出竞争力或改进。

**⚠️ 局限性**

局部单步近似忽略了超参数对训练的长期影响，且验证梯度的质量依赖于训练和验证分布之间的差异，可能导致噪声和不稳定性。

---

## 649. The Fragility of Reasoning-Importance Attribution for Pruning Language Models

**arXiv ID:** 2611.22234 | [PDF](https://arxiv.org/pdf/2611.22234v1)

**作者:** Himanshu Mishra `[一作]` `[通讯]` (University of British Columbia), Himanshu Mishra (University of British Columbia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了归因引导的剪枝方法在压缩推理语言模型中的应用，指出其依赖的推理重要性信号的测量存在脆弱性，并提出了改进的测量规则。

**💡 创新点**

创新点在于揭示了推理重要性测量的脆弱性，包括最终答案标记的饱和性、单元重要性的不可局部化以及推理特定重要性与一般重要性的不可分离性，并提出了针对这些问题的测量规则。

**🔧 技术方法**

使用了归因引导的剪枝技术，结合了对推理模型的实验和分析。

**📊 数据集**

使用了四个小型推理模型进行实验，包括经过蒸馏的模型和非蒸馏的指令模型，数据集为GSM8K，一个小学数学问题基准。

**📈 对比分析**

通过比较不同的剪枝目标（最终答案标记与推理标记），发现使用最终答案标记进行剪枝会导致27.6到32.2的准确率损失，而使用推理标记则保持较高的准确率，且在准确率与困惑度之间存在解耦现象。

**⚠️ 局限性**

限制在于研究主要集中在1.5B模型上，7B模型的结果不够明确，且集中机制未能得到解决。此外，研究仅覆盖数学推理任务，未来需要扩展到其他类型的推理任务。

---

## 650. Digital Qiang Garden: Artist Residency as Interface for Creativity Under Change

**arXiv ID:** 2611.21640 | [PDF](https://arxiv.org/pdf/2611.21640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 651. A tiny object detection dataset for high-altitude open-water search and rescue

**arXiv ID:** 2611.22825 | [PDF](https://arxiv.org/pdf/2611.22825v1)

**作者:** Hui Xu `[一作]` (Beihang University), Tianshi Xu `[通讯]` (GRG Banking Equipment Co. Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了TinyWOD数据集，这是一个专为高空开放水域搜索与救援（OW-SAR）设计的小物体检测数据集，包含11359张手动标注的图像和19505个标注实例。

**💡 创新点**

TinyWOD数据集填补了现有数据集中缺乏高空小物体检测的空白，特别是针对1.5-2.0公里高空的场景，且定义了新的小物体检测指标AP_tiny。

**🔧 技术方法**

使用了多种技术，包括多尺度特征融合、高分辨率推理、专门的小物体标签分配和增强等。

**📊 数据集**

数据集包含11359张图像，主要来自中国浙江省的三个水域，涵盖不同的水色和表面条件。

**📈 对比分析**

与现有数据集相比，TinyWOD的目标尺度更小，76.96%的实例的等效边长（ESL）小于15像素。评估了七种主流检测网络，结果显示所有网络在检测小型水上物体方面表现良好，最高AP为0.512。

**⚠️ 局限性**

数据集仅包含一种物体类别（缩小版的救生圈），且数据收集仅限于中国浙江省，夜间图像较少，这可能影响其在其他物体类别或低光条件下的适用性。

---

## 652. Social Signal Bench: Evidence-Weighted Evaluation of Co-occurring Social Signals in Video and Audio

**arXiv ID:** 2611.22704 | [PDF](https://arxiv.org/pdf/2611.22704v1)

**作者:** Siddharth Ravi `[一作]` (Universidad de Alicante), Line H. Clemmensen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类准确率。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10数据集进行实验。

**📈 对比分析**

与传统的激活函数（如ReLU）进行比较，结果显示新模型在分类准确率上提高了5%。

**⚠️ 局限性**

限制在于模型在处理高分辨率图像时的性能尚未得到验证。

---

## 653. Physics-informed neural networks for spacecraft trajectory optimization with exact boundary conditions

**arXiv ID:** 2611.21933 | [PDF](https://arxiv.org/pdf/2611.21933v1)

**作者:** Samuel Schönherr `[一作]` (Institute of Theoretical and Computational Physics), Sascha Ranftl `[通讯]` (Brown University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用神经网络优化轨迹，通过将位置建模为时间的函数，并最小化与网络参数相关的推力。这种方法将物理法则直接纳入优化目标，采用无监督训练程序。

**💡 创新点**

提出了一种简单的变换方法，通过利用轨迹优化问题的数学结构，精确地提前强加边界条件，从而将多目标优化问题转化为单目标优化问题，显著提高了收敛性和解的准确性。

**🔧 技术方法**

使用物理信息神经网络（PINNs）和距离函数的变换技术。

**📊 数据集**

进行了多种数值实验，包括多体摆动轨迹和低推力轨道转移，比较了提出的方法与经典数值求解器OpenGoddard的性能。

**📈 对比分析**

与传统的PINNs相比，提出的方法在收敛性、鲁棒性和解的质量上表现更好，尤其是在复杂的轨迹优化问题中，能够精确满足边界条件，而传统PINNs在这些情况下往往无法收敛。

**⚠️ 局限性**

在高维设置中，优化景观可能包含多个物理上有意义的局部最小值，虽然精确的边界条件强加可以提高收敛性，但并不能保证收敛到全局最优解。此外，当前研究仅限于理想化的引力模型，扩展到现实的天体动力学场景需要考虑扰动和更高阶的动态。

---

## 654. A Dichotomy for Exact Quantum Circuit Amplitudes

**arXiv ID:** 2611.22899 | [PDF](https://arxiv.org/pdf/2611.22899v1)

**作者:** Chenghua Liu `[一作]` (Institute of Software), Boning Meng `[通讯]` (University of Regensburg)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对固定有限的单位量子比特门库的电路进行复杂性分类，研究了计算计算基态振幅的精确计算问题。

**💡 创新点**

创新点在于将电路振幅与布尔Holant分类联系起来，提供了一个关于精确振幅计算的#P-hard性与多项式时间可计算性的完整二分法。

**🔧 技术方法**

使用了量子电路理论和布尔Holant理论，结合了张量收缩的技术。

**📊 数据集**

使用了固定的单位量子比特门库，具体数据集未明确提及。

**📈 对比分析**

通过与已知的可计算性条件进行比较，证明了在特定条件下精确振幅计算是多项式时间可计算的，而在其他情况下则是#P-hard的。

**⚠️ 局限性**

限制在于对于不满足三种条件的门库，精确振幅计算的复杂性是#P-hard，且对于任意复杂的门条目，仍需满足特定的代数关系。

---

## 655. OfficePrism: Benchmarking Agents on Single-Domain Operations Across Enterprise Business Functions

**arXiv ID:** 2611.22262 | [PDF](https://arxiv.org/pdf/2611.22262v1)

**作者:** Hanfeng Liu `[一作]` (Alibaba Group), Zeyi Wen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了OfficePrism，一个用于评估单领域企业操作的基准，包含420个任务，涵盖七个业务功能。

**💡 创新点**

创新点在于通过功能闭合的任务设计，提供了一个明确的基准评估框架，能够在单一业务功能内进行局部评估。

**🔧 技术方法**

使用了状态基础的构建管道，结合了业务规则、工具架构和记录，生成中文请求票据及其参考见证和预期响应。

**📊 数据集**

使用了420个任务的自定义数据集，这些任务在一个合成组织内进行，涵盖了行政、财务、人力资源等七个业务功能。

**📈 对比分析**

通过124种模型-工具包配置进行评估，联合自然语言和数据库的通过率范围为22.6%到76.4%，中位数为67.9%。L1在所有配置中表现最佳，L2在122个配置中超过L3。

**⚠️ 局限性**

局限性在于OfficePrism只研究了一个合成组织中的功能闭合操作，未能估计工作活动的普遍性或代表组织、语言、角色或业务流程的分布。

---

## 656. Trusting Trust in AI Software Engineering Agents: The Why, What, and How?

**arXiv ID:** 2611.21634 | [PDF](https://arxiv.org/pdf/2611.21634v1)

**作者:** Pranav V `[一作]` (Indian Institute of Technology), Vikrant Kaulgud `[通讯]` (Accenture Labs)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一个结构化框架，用于分析AI软件工程代理中的信任问题，强调信任不仅仅是单个代理的输出，还涉及多个代理之间的相互依赖。

**💡 创新点**

创新点在于将信任分解为三个问题：为什么信任？什么信任？信任程度如何？并提出了相应的场景、配置和属性。

**🔧 技术方法**

使用了结构化信任推理的方法，结合了信任模型和软件工程的具体场景。

**📊 数据集**

没有具体提到使用的数据集，但提到了一些基于现有代理的工作流示例，如ChatDev和其他商业及开源代理。

**📈 对比分析**

与传统信任模型相比，该框架强调了多代理工作流中的信任传播和配置，性能上提供了更细致的信任分析，但没有提供具体的性能比较数据。

**⚠️ 局限性**

局限性在于该框架是概念性的，尚未经过实证验证，且提出的属性集可能需要进一步的细化和验证。

---

## 657. Longitudinal Multi-Dimensional Analysis of India-Pakistan Conflict Discourse on Reddit (2019-2025)

**arXiv ID:** 2611.22987 | [PDF](https://arxiv.org/pdf/2611.22987v1)

**作者:** Yash Khare `[一作]` (Northeastern University), Mohit Singhal `[通讯]` (Northeastern University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究分析了2019年至2025年间六次印巴危机在Reddit上的讨论，比较了不同社区如何框架这些危机。研究收集了43000个帖子和305000条评论，使用混合方法分析了这些评论的叙事框架和争议性。

**💡 创新点**

创新点在于首次对印巴危机的多次讨论进行纵向研究，结合了评论的叙事框架与其所在社区的特征，揭示了社区对争议的影响。

**🔧 技术方法**

使用了GPT-5.2对评论进行叙事框架标注，并结合Google的Perspective API测量评论的毒性和争议性。

**📊 数据集**

数据集包括来自十个Reddit社区的305217条评论，这些社区分为六个印度对齐社区、一个巴基斯坦对齐社区和三个普通观众社区。

**📈 对比分析**

与传统方法相比，研究发现争议性主要受社区和叙事框架的影响，而非评论的毒性。人道主义框架的争议性最高，而经济框架的争议性最低。社区对争议的影响是2.5倍于毒性对争议的影响。

**⚠️ 局限性**

研究的局限性在于未能深入探讨社区内的个体差异，以及如何在不同社区之间进行有效的比较。

---

## 658. When Lower Harm Misleads: Auditing Concept Corrections

**arXiv ID:** 2611.22084 | [PDF](https://arxiv.org/pdf/2611.22084v1)

**作者:** Tianyou Lai `[一作]` (Carnegie Mellon University), Qilei Li `[通讯]` (Central China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种校正审计方法，用于评估医学影像中概念预测的校正效果，特别是在乳腺超声中BI-RADS描述符的校正。

**💡 创新点**

创新点在于将校正效果分解为有害部分和有用增益，并通过匹配源损害的幅度基线进行比较，从而提供了一种可重复的评估协议。

**🔧 技术方法**

使用了校正审计方法，结合了最坏情况校正训练和单调损害惩罚等技术。

**📊 数据集**

使用了三个乳腺超声数据集（Private250、BUSI647和BUS-BRA）进行实验，并在CUB数据集上进行了控制实验。

**📈 对比分析**

通过与不同训练目标的模型进行比较，发现最坏情况校正训练虽然降低了损害，但在所有21个比较中都损失了更多的有用增益，导致校正后的负对数似然（NLL）上升。

**⚠️ 局限性**

限制在于临床模型对概念的预测能力较弱，且所有临床比较使用了开发数据，独立测试的结果未能排除零的可能性。

---

## 659. Dissecting the GPU Load/Store Unit and Exposing Its Security Risks

**arXiv ID:** 2611.23022 | [PDF](https://arxiv.org/pdf/2611.23022v1)

**作者:** Wenbo Han `[一作]` (University of Luxembourg), Pierre Talbot `[通讯]` (University of Luxembourg)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文通过逆向工程研究了NVIDIA GPU的负载/存储单元（LSU）如何处理负载指令，并分析了其安全隐患。

**💡 创新点**

创新点在于提出了一个周期精确的延迟公式，能够根据负载访问的地址预测延迟，并揭示了LSU的攻击面比之前已知的更广泛。

**🔧 技术方法**

使用了微基准测试（microbenchmarks）和NVIDIA的汇编语言SASS进行实验，设计了定时微基准以分析LSU的行为。

**📊 数据集**

研究涉及NVIDIA的V100、A100和H100 GPU，使用了这些GPU的负载指令进行实验。

**📈 对比分析**

通过与现有方法的比较，发现LSU的饱和状态可以导致后发块的饥饿，且在MPS下，一个warp可以将cuBLAS矩阵乘法的速度降低高达563倍。

**⚠️ 局限性**

限制在于对LSU的某些机制和参数的理解仍然是孤立的，且在不同GPU架构之间可能存在差异，可能影响结果的普适性。

---

## 660. Inferring Locations and Correcting Bias in Infectious Disease Data Using Human Mobility

**arXiv ID:** 2611.23123 | [PDF](https://arxiv.org/pdf/2611.23123v1)

**作者:** Owen Tucker `[一作]` (Emory University), Andreas Züfle `[通讯]` (Emory University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究通过整合病原体基因组数据与人类流动信息，推断未观察到的传染事件，以解决传染病传播的重建问题。

**💡 创新点**

创新点在于提出了一种基于人类流动数据的信念传播方法，以估计未观察到感染的地点，并结合基因组监测数据提高传染树的重建精度。

**🔧 技术方法**

使用了信念传播算法和连续时间马尔可夫链模型，结合基因组数据和人类流动数据进行推断。

**📊 数据集**

使用了模拟数据集和真实的COVID-19基因组监测数据，后者来自GISAID数据库。

**📈 对比分析**

与传统方法相比，结合流动数据的重建方法在模拟数据上减少了39.4%的交叉熵，在真实数据上减少了16.1%，显示出更好的性能。

**⚠️ 局限性**

限制在于仅考虑了收入作为偏差来源，未来研究应考虑更多的偏差因素。

---

## 661. Pathology Randomization for Synthetic Priors and Zero-Shot Detection

**arXiv ID:** 2611.21975 | [PDF](https://arxiv.org/pdf/2611.21975v1)

**作者:** Alexandra G. Roberts `[一作]` (Harvard Medical School and Massachusetts General Hospital), Juan Eugenio Iglesias `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种用于脑MRI病理生成的模型，通过将病变随机化到健康标签图中，生成合成扫描，并在没有真实图像或病变掩膜的情况下进行零-shot检测。

**💡 创新点**

创新点在于提出了一种病理随机化的方法，使得模型能够在没有真实病理数据的情况下，进行病变检测、超分辨率和配准。

**🔧 技术方法**

使用了对比无关的3D UNet网络来估计健康解剖结构和病变的偏差，并进行病变检测。

**📊 数据集**

使用了多个公开数据集进行测试，包括ATLAS、BraTS、WMH和PRES等，涵盖了不同的疾病和扫描条件。

**📈 对比分析**

与对比特定的异常检测器相比，该模型在多个数据集上表现优越或相当，能够保持健康解剖结构的完整性。

**⚠️ 局限性**

限制在于合成病理是基于形状、位置和强度的随机化，而不是基于特定疾病，因此可检测的条件必须位于该随机化的形状流形内。

---

## 662. GameRealm: Grounding Visual Learning in Interactive Games

**arXiv ID:** 2611.23140 | [PDF](https://arxiv.org/pdf/2611.23140v1)

**作者:** Weiguang Zhao `[一作]`, Chunhua Shen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了GameRealm，一个大规模的游戏衍生数据集，提供丰富的多模态监督，旨在解决视觉基础模型在物理智能发展中的数据碎片化和监督不足的问题。

**💡 创新点**

GameRealm的创新点在于其规模和多样性，包含超过1.2亿帧RGB图像和1000小时的游戏录像，提供同步的多模态监督，包括深度信息、相机参数、音频和控制输入。

**🔧 技术方法**

开发了一个统一的数据采集管道，结合人工操作和自动探索，以实现可扩展的数据收集，并通过游戏引擎插件直接获取地面真实信息。

**📊 数据集**

使用了GameRealm数据集，该数据集包含超过1.2亿帧RGB图像和1000小时的游戏录像，涵盖多种游戏环境和交互场景。

**📈 对比分析**

通过在多个视觉学习任务上进行广泛实验，证明GameRealm在视觉生成和几何学习中显著增强了视觉基础模型的性能，尤其是在交互式世界建模和几何重建任务中表现出色。

**⚠️ 局限性**

限制在于当前数据集的注释仍然可以进一步细化，未来工作计划增加更细粒度的地面真实注释，如人类和物体骨架、像素级分割掩码和实例级信息，以支持更广泛的视觉学习任务。

---

## 663. Structured AI ideation yields physical strategies for epithelial barrier repair

**arXiv ID:** 2611.22988 | [PDF](https://arxiv.org/pdf/2611.22988v1)

**作者:** Geonho Nah `[一作]` (Asteromorph), Minhyeong Lee `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一个名为Abelio的AI系统，旨在通过扩展研究者的研究方向来生成新颖的研究假设。

**💡 创新点**

Abelio通过重新构建研究目标，识别传统学科边界之外的相关元素，从而生成多样化的假设，展示了结构化研究构思的潜力。

**🔧 技术方法**

使用了AI驱动的科学构思方法，结合了多种子系统进行假设生成和评估。

**📊 数据集**

使用了Caco-2单层细胞作为实验数据集，进行肠道上皮屏障修复的实验验证。

**📈 对比分析**

与四个前沿大型语言模型进行了比较，Abelio生成的假设在科学严谨性和假设多样性方面表现优于其他模型，并且在与现有文献的距离上也有显著优势。

**⚠️ 局限性**

限制在于需要领域专家的参与来优化实验设计，且当前的评估方法仍然面临挑战，无法全面验证假设的创新性和质量。

---

## 664. HyCoInfer: Cloud-Edge Collaborative Inference for Large Language Models on Heterogeneous GPU-FPGA Architectures

**arXiv ID:** 2611.22086 | [PDF](https://arxiv.org/pdf/2611.22086v1)

**作者:** Zewen Huang `[一作]` (Tsinghua University), Qiang Ma `[通讯]` (Tsinghua University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种异构云-边缘协作推理系统HyCoInfer，该系统将预填充任务分配给云GPU，并将解码任务分配给云GPU和边缘FPGA，以提高大语言模型的推理效率。

**💡 创新点**

创新点在于通过结合云和边缘资源，解决了异构加速器之间的容量平衡、KV缓存跨越受限广域链路的挑战，以及异构执行带来的延迟问题。

**🔧 技术方法**

使用了GPU和FPGA的协作推理技术，结合了层级流水线传输、解码器侧预计算和基于残差向量量化的带宽自适应KV缓存压缩。

**📊 数据集**

在实验中使用了NVIDIA V100 GPU和AMD Xilinx Alveo U280 FPGA，并评估了不同网络条件下的多个工作负载。

**📈 对比分析**

与云端单一vLLM相比，在500 Mbps的链路下，HyCoInfer的请求良率提高了4.11倍，平均端到端延迟降低了61.1%。在不同网络条件和工作负载下，系统整体服务能力得到了提升，尤其在通信密集型任务中表现最佳。

**⚠️ 局限性**

限制在于边缘FPGA的计算能力有限，可能无法单独支持高并发的服务工作负载，同时在不同网络条件下的性能表现可能会有所波动。

---

## 665. Finding Patient Zero in Temporal Contact Networks: A Comparative Study

**arXiv ID:** 2611.23175 | [PDF](https://arxiv.org/pdf/2611.23175v1)

**作者:** Nathan Cincera Brack `[一作]` (University of Zurich), Abraham Bernstein `[通讯]` (University of Zurich)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究对七种主要的流行病源检测方法进行了比较，包括Steiner Tree (ST)、Individual-Based Approximation (IBA)、Dynamic Message Passing (DMP)、Monte Carlo Mean-Field (MCMF)、Soft Margin Estimator (SME)、Backtracking Network (BN)和Belief Propagation (BP)，并在八个不同的时间网络上进行了评估。

**💡 创新点**

本研究的创新点在于统一了所有方法的比较框架，并在相同的实验条件下进行了系统的比较，填补了现有文献中缺乏对时间接触网络源检测方法的系统评估的空白。

**🔧 技术方法**

使用了多种技术，包括图神经网络（BN）、信念传播（BP）和其他基于概率的模型（如MCMF和SME），并对这些方法的假设和算法复杂性进行了分析。

**📊 数据集**

使用了八个不同的时间网络数据集，包括巴布亚猕猴、马拉维村、波士顿学生、法国办公室、纽约签到、哥本哈根学生、东京签到和巴西陪护网络。

**📈 对比分析**

通过比较不同方法在不同基本再生数（R_0）下的源检测性能，发现BP和BN是整体上最强的源检测器。SME在部分观察下表现良好，但在其他情况下表现平平。IBA、DMP和MCMF由于其假设的局限性，表现较差。

**⚠️ 局限性**

本研究的局限性包括只考虑了单一观察快照，未测试方法在多个快照下的适应性；对大规模网络的可扩展性未进行测试；以及补充实验仅在较小的网络上进行，证据相对较弱。

---

## 666. PURE: A Diagnostic Benchmark for Multi-Granularity Multimodal Identity Matching

**arXiv ID:** 2611.22343 | [PDF](https://arxiv.org/pdf/2611.22343v1)

**作者:** Hengyu You `[一作]` (Alibaba Group), Liang Ding `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个用于多粒度多模态身份匹配的诊断基准Pure，涵盖了感知、理解、推理和基于证据的判断四个能力维度，共包含1457个多项选择题，旨在评估模型在不同匹配粒度下的表现。

**💡 创新点**

创新点在于定义了多粒度身份匹配的标准，并通过真实的电子商务记录构建了一个系统化的基准，揭示了模型在不同能力维度上的表现差异。

**🔧 技术方法**

使用了多模态学习模型（MLLMs）进行实验，评估模型在感知、理解、推理和判断等任务上的能力。

**📊 数据集**

数据集来源于真实的电子商务记录对，经过两个领域专家的验证，确保了数据的真实性和有效性。

**📈 对比分析**

与十个最新的多模态学习模型进行比较，整体得分范围从47.1%到86.0%。理解能力得分最高，而判断能力得分最低，显示出模型在视觉差异定位和变体级规则判断上的弱点。

**⚠️ 局限性**

限制在于Pure仅评估生成型多模态学习模型的显式响应，未能直接表征基于嵌入的检索或端到端匹配系统的性能。此外，基准的构建依赖于特定电子商务环境中的记录，可能无法推广到其他平台、语言或产品类别。

---

## 667. Watermark Evidence under Open-Weight LLM Modification: Exact Bounds for Low-Rank Output Watermarks

**arXiv ID:** 2611.21702 | [PDF](https://arxiv.org/pdf/2611.21702v1)

**作者:** Prashant Shekhar `[一作]` (Embry-Riddle Aeronautical University), Caroline Howard `[通讯]` (Embry-Riddle Aeronautical University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型（LLM）水印的检测与验证，提出了一种新的方法OpenStamp，通过改变生成文本中的标记概率来实现水印的嵌入和验证。

**💡 创新点**

创新点在于提出了一种精确的有限断点下界，用于在有界logit残差下的条件参考-LLR均值，并结合分布无关的顺序统计容忍极限来界定错过概率。

**🔧 技术方法**

使用了OpenStamp方法，该方法嵌入了结构化的低秩输出头更新，并通过比较标记和私有保留基检查点下的标记文本的token概率来进行验证。

**📊 数据集**

使用了Llama-2-7B和Mistral-7B-v0.3模型，并在OpenWebText数据集上进行了500步的LoRA微调，数据集包含了11,697个符合条件的文档。

**📈 对比分析**

与其他方法相比，OpenStamp在INT8量化和30%幅度修剪下保持了85.6%-94.8%的真实正率（TPR），而LoRA微调后TPR降至28.8%和11.6%。所有八个家庭的95%保留界限均保持在1.0。

**⚠️ 局限性**

限制在于当前样本和时刻包络未能提供有用的数值人口错过风险保证，需要更大的认证集、额外的种子和更多的修改类型来评估证书何时变得信息丰富。

---

## 668. Benchmarking Deep Learning Models for Wheelchair Configuration Classification in Accessible Automated Vehicle Applications

**arXiv ID:** 2611.21945 | [PDF](https://arxiv.org/pdf/2611.21945v1)

**作者:** Samir Ahmed `[一作]` (University of California Merced), Ross Greer `[通讯]` (University of California Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种新的轮椅配置图像数据集，并评估了深度学习模型在自动化车辆应用中的轮椅配置分类性能。

**💡 创新点**

创新点在于创建了一个专注于轮椅配置的公开数据集，并进行了跨数据集的准确性和选择性推迟评估，分析了模型在高斯模糊下的鲁棒性。

**🔧 技术方法**

使用了卷积神经网络（CNN）架构，包括ResNet-50、EfficientNet-B0和ConvNeXt-Tiny，以及一个冻结的视觉-语言编码器进行比较。

**📊 数据集**

使用了467张轮椅图像，分为273张验证集和194张测试集，图像来自192个捕获组，涵盖了不同的轮椅配置和环境条件。

**📈 对比分析**

与其他方法相比，ConvNeXt-Tiny在194张测试图像上取得了最高的平均准确率（0.921±0.006）和最低的选择性风险（AURC 0.015±0.004），在推迟30%最不自信的预测后准确率达到0.980±0.008。

**⚠️ 局限性**

限制在于数据集规模、网络来源偏差以及缺乏物理设备序列号，模型在不同上传者的相同轮椅图像可能出现在测试和验证集中，且每个模型的动力轮椅召回率低于手动轮椅召回率。

---

## 669. SOTA or Luck? The Winner's Curse in LLM Leaderboards

**arXiv ID:** 2611.22227 | [PDF](https://arxiv.org/pdf/2611.22227v1)

**作者:** Sikun Xu `[一作]` (Washington University in St. Louis), Heng Zhang `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的方法来修正公共大型语言模型（LLM）排行榜中的排名偏差和不确定性，开发了一个名为RULER的报告工具，旨在提供更准确的模型评估。

**💡 创新点**

创新点在于提出了m-out-of-N自助法来校正排名偏差，并量化模型排名的不确定性，提供了排名概率矩阵和模型分层的工具。

**🔧 技术方法**

使用了m-out-of-N自助法进行统计推断，结合了模型评分调整、排名概率矩阵和排名区间。

**📊 数据集**

使用了来自SWE-bench Verified的500个软件工程任务的数据集，评估了173个模型的表现。

**📈 对比分析**

通过与传统方法的比较，RULER显著降低了选择误差，排名1的调整从2.38降低到0.73，减少了约69.5%。在14个基准中，排名1的调整超过1个百分点的有7个，12个基准中存在多个模型在领先层。

**⚠️ 局限性**

限制在于RULER的表现依赖于基准的代表性和评分的意义，样本生成过程的变化可能导致额外的偏差，且在均值差异与采样噪声相当时，校正可能会留下残余偏差。

---

## 670. Can You Undo It? Reversibility Classes for AI Agent Actions

**arXiv ID:** 2611.22075 | [PDF](https://arxiv.org/pdf/2611.22075v1)

**作者:** Deepak Waghmare `[一作]` (Two Much Labs), Shaad Sufi `[通讯]` (Two Much Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了可逆性类别的概念，定义了一个五级词汇，旨在评估AI代理的工具调用是否可以撤销。

**💡 创新点**

创新点在于引入了可逆性类别的分类系统，提供了机器可读的恢复声明，并通过实验验证了不同工具的恢复能力。

**🔧 技术方法**

使用了分类器、会话保护机制和证据链接注册表等技术。

**📊 数据集**

使用了536个工具的注册表，涵盖20个MCP服务器，并进行了2158次记录的实验。

**📈 对比分析**

与现有的MCP注释进行了比较，发现所有738个规则案例检查符合预期，且在攻击正面运行中，实施的会话保护机制在29次中成功阻止了相关操作，而读/写代理仅在18次中成功。

**⚠️ 局限性**

限制在于人类评审的独立性不足，且未能验证所有工具的恢复能力，未来需要更多的独立评估和实地测试。

---

## 671. SCAN: Site-Aware Collaborative Privacy-Preserving Adaptation of Vision-Language Models for Neurological Disease Classification

**arXiv ID:** 2611.21955 | [PDF](https://arxiv.org/pdf/2611.21955v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 672. Large-scale Neuron Community Discovery Reveals Neuron-Sparse Behaviors in Language Models

**arXiv ID:** 2611.23112 | [PDF](https://arxiv.org/pdf/2611.23112v1)

**作者:** Igor Shilov `[一作]` (Goodfire), Owen Lewis `[通讯]` (Goodfire)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种神经元社区发现的方法，通过分析大量提示，识别出反复共同被激活的神经元组，并测试它们在预测中的必要性和充分性。

**💡 创新点**

创新点在于不需要预先指定功能，通过对数百万个提示的分析，发现神经元组的共同计算单元，从而实现可扩展的神经元社区发现。

**🔧 技术方法**

使用了基于梯度的归因方法和Leiden社区检测算法。

**📊 数据集**

使用了来自FineWeb-Edu、维基百科和FineMath的1000万个提示数据集。

**📈 对比分析**

通过因果干预方法比较社区的必要性和充分性，发现约10.3%的提示可以通过发现的神经元社区进行解释，性能显著优于随机基线（仅3%）。

**⚠️ 局限性**

限制在于未能发现的行为可能仍然存在神经元稀疏机制，且当前方法仅限于最后一个提示位置的归因，可能错过跨位置的社区。

---

## 673. FMT: Lossless Image Compression in a Fixed Number of Parallel Steps

**arXiv ID:** 2611.23108 | [PDF](https://arxiv.org/pdf/2611.23108v1)

**作者:** Ammar Ali `[一作]` `[通讯]` (ITMO University), Ammar Ali (ITMO University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的无损图像编码格式，称为固定步多尺度树（Fixed-step Multiscale Tree），旨在提高解码速度并减小文件大小。

**💡 创新点**

创新点在于将解码顺序和熵编码器设计结合，使得解码过程中的等待时间不随图像大小增长，从而实现固定数量的顺序步骤解码。

**🔧 技术方法**

使用了GPU并行处理技术，结合了快速熵编码器（rANS）和多层感知器（MLP）网络来预测像素值。

**📊 数据集**

使用了Kodak照片数据集（768×512）以及其他大型照片集（CLIC和DIV2K）进行实验。

**📈 对比分析**

与传统编码器（如JPEG XL和HTJ2K）相比，提出的方法在文件大小和解码速度上均表现优越，尤其是在使用大型网络时，解码速度比学习型编码器快9到61倍。

**⚠️ 局限性**

限制在于该方法主要针对8位照片，且每张照片的调优过程需要几秒钟，解码器在CPU上竞争力较弱，且对10到16位图像的支持不足。

---

## 674. NexSight: Verifiable Tasks for Fine-Grained Agentic Long-Video Understanding

**arXiv ID:** 2611.22790 | [PDF](https://arxiv.org/pdf/2611.22790v1)

**作者:** Tianci Xie `[一作]` (Zhejiang University), Rui Zheng `[通讯]` (Shanghai Qiji Zhifeng Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了NexSight，一个用于评估和训练长视频理解的可验证任务构建框架，旨在解决长视频中细粒度问题的挑战。

**💡 创新点**

创新点在于通过SightForge构建可验证任务，并通过SightLoop进行多尺度的主动评估，从而提高了模型在长视频理解中的表现。

**🔧 技术方法**

使用了主动观察模型，结合了任务导向的观察选择和多尺度的评估框架。

**📊 数据集**

使用了来自108个长视频的310个可验证任务，涵盖了74.12小时的内容。

**📈 对比分析**

与其他方法相比，NexSight在验证任务上表现出色，最强模型在SightLine上仅达到54.52%的准确率，但在提供注释证据时可提升至79.35%。

**⚠️ 局限性**

限制在于SightLine主要关注视觉证据和屏幕文本，未来需要整合音频-视觉跨模态验证，并扩展到更大的视频数据集。

---

## 675. Solver-Free and Cost-Aware Online Scheduling of AI Data Center Tasks

**arXiv ID:** 2611.22239 | [PDF](https://arxiv.org/pdf/2611.22239v1)

**作者:** Xin Qin `[一作]` (Nanyang Technological University), Fei Teng `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种无求解器的在线调度规则，用于优化AI数据中心任务的能耗成本，确保在任务到达时为每个任务分配一个开始时间。

**💡 创新点**

创新点在于提出了一种基于双重分解的在线优化方法，能够在不依赖求解器的情况下高效调度任务，同时优化能耗成本。

**🔧 技术方法**

使用了无求解器的在线调度规则，结合了双重分解的思想来整合CPU/GPU资源的影子价格。

**📊 数据集**

使用了A100 GPU追踪数据集，包含3456个GPU和1343个任务的实际AI任务数据。

**📈 对比分析**

与多时间尺度优化基准相比，提出的方法在成本上更低，且计算时间显著减少，能够在每个任务到达时快速做出决策。

**⚠️ 局限性**

限制在于未来任务和市场价格的不确定性，以及需要进一步研究多天评估和在线算法的理论界限。

---

## 676. Learning Optimal Transport Plans Via Autoregressive Token Regression

**arXiv ID:** 2611.22218 | [PDF](https://arxiv.org/pdf/2611.22218v1)

**作者:** Ivan Jaen Marquez `[一作]` (University of Wisconsin--Madison), Vikas Singh `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于变换器的模型ToR-OT，用于通过自回归令牌回归预测离散的最优运输计划。

**💡 创新点**

ToR-OT通过将最优运输问题重新构建为结构化序列生成任务，展示了自回归模型在解决约束优化问题中的潜力，尤其是在零-shot泛化和稀疏解码方面。

**🔧 技术方法**

使用了变换器（Transformer）架构，采用自回归令牌回归机制进行预测。

**📊 数据集**

使用了合成生成的最优运输实例数据集，涵盖多种成本类型和不同的分布对。

**📈 对比分析**

与传统的最优运输求解器相比，ToR-OT在处理未见分布对时表现出良好的零-shot泛化能力，并且在推理效率上优于每实例的神经最优运输基线，尽管在严格的最优性要求下仍需依赖传统求解器。

**⚠️ 局限性**

ToR-OT在处理更大支持（更多箱子）时可能面临挑战，且在高维度测度的扩展上仍需进一步研究。

---

## 677. FieldGuide: A Corpus Map for Token-Efficient Agentic Document Exploration

**arXiv ID:** 2611.21647 | [PDF](https://arxiv.org/pdf/2611.21647v1)

**作者:** Nick Hagar `[一作]` `[通讯]` (University of Minnesota), Nick Hagar (University of Minnesota)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文介绍了FieldGuide，一个面向代理的语料库地图，旨在帮助记者高效探索大型文档集合。该系统通过主题层次结构组织文档，并提供语料库定位、范围搜索和有限阅读的工具。

**💡 创新点**

创新点在于FieldGuide通过分层的主题索引和导航工具，优化了文档探索过程，减少了所需的token数量和时间，同时保持了准确性。

**🔧 技术方法**

使用了生成式AI技术，结合了模型上下文协议（MCP）来实现文档的导航和检索。

**📊 数据集**

使用了来自芝加哥警察局的38份公共记录文档作为数据集，这些文档包括多种类型，如事件报告、战术响应报告和政策文件。

**📈 对比分析**

与原始文件访问和检索增强生成（RAG）方法相比，FieldGuide在五个任务中分别减少了27%和41%的token使用和时间，相较于RAG，减少了38%和21%。

**⚠️ 局限性**

限制在于该评估估计了工作流程级别的性能，而没有单独隔离FieldGuide各个组件的效果。此外，评估仅使用了一种模型和特定的提示，结果可能不适用于其他代理或提示策略。

---

## 678. Born for the Task: Rethinking Quantum Chemistry with Foresighted Agentic Computing

**arXiv ID:** 2611.22897 | [PDF](https://arxiv.org/pdf/2611.22897v1)

**作者:** Haozhi Han `[一作]` (Peking University), Kun Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种前瞻性的代理计算框架，能够根据需求合成高性能的量子化学程序，旨在提高计算效率并扩展计算能力。

**💡 创新点**

创新点在于通过预测每种计算策略的目标规模误差和运行时间，避免了在目标规模上进行昂贵的试错，从而使程序开发成为计算的一部分。

**🔧 技术方法**

使用了前瞻性代理计算框架，结合了组件级性能预测、目标规模前瞻性分析和代理驱动的编译技术。

**📊 数据集**

在多个数据集上进行了评估，包括Hartree-Fock计算（50,024个基函数）、泛素计算和B-DNA双螺旋计算。

**📈 对比分析**

与通用软件包相比，该框架在Hartree-Fock计算中实现了77倍的计算加速，并在开发时间包含的情况下仍然实现了9.8倍的加速。对于泛素计算，减少了GPU需求，从64个降至1个，同时保持了相似的运行时间。

**⚠️ 局限性**

局限性在于该框架的有效性依赖于对小规模系统的准确预测，且在处理更复杂的物理模型时可能需要进一步的研究和优化。

---

## 679. NEBench R1: a Reasoning Benchmark for Evaluating Nuclear Domain Competence in Large Language Models

**arXiv ID:** 2611.21682 | [PDF](https://arxiv.org/pdf/2611.21682v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 680. Breaking Visual Conventions: Spectrally Coupled Weight Perturbation for Creative Image Generation

**arXiv ID:** 2611.22318 | [PDF](https://arxiv.org/pdf/2611.22318v1)

**作者:** Fu Feng `[一作]` (Southeast University), Xin Geng `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为CosPert的方法，通过对预训练的文本到图像（T2I）模型的权重进行结构化扰动，增强视觉创造力。

**💡 创新点**

CosPert通过对注意力投影进行耦合扰动，打破了学习到的视觉惯例，同时保持图像质量和语义一致性。

**🔧 技术方法**

使用了耦合谱扰动（CosPert）技术，该方法无需训练，通过对注意力投影矩阵的结构化扰动来促进创造性生成。

**📊 数据集**

使用了FLUX.1和Stable Diffusion 3.5数据集，构建了一个包含50个常见类别的基准数据集。

**📈 对比分析**

与标准T2I模型和其他创造性生成方法相比，CosPert在创造性和多样性方面表现出色，获得了最高的CREward分数，并在用户研究中获得了最高的偏好评分。

**⚠️ 局限性**

限制在于CosPert依赖于一次性的层敏感性分析，可能无法适应所有类型的模型或任务。

---

## 681. Testing World Models on Actions They Have Never Seen: Held-Out Plans, Forecast Accuracy and Price Rules in Retail

**arXiv ID:** 2611.22537 | [PDF](https://arxiv.org/pdf/2611.22537v1)

**作者:** Karthik Iyer `[一作]` `[通讯]`, Karthik Iyer

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文测试了世界模型在未见过的行动上的表现，特别是在零售环境中，评估价格和促销计划的预测准确性。

**💡 创新点**

创新点在于将三种测试方法结合起来，评估世界模型在未见过的行动上的准确性，并与简单模型进行比较。

**🔧 技术方法**

使用了世界模型、LightGBM和对数线性模型等技术，结合了JEPA预训练和价格规则。

**📊 数据集**

使用了生成的零售数据集，包括Breakfast at the Frat、The Complete Journey和M5等真实数据集。

**📈 对比分析**

通过比较不同模型在新计划上的预测误差，发现世界模型在某些情况下优于简单模型，尤其是在深度价格削减的情况下，但在持续价格变动上表现不佳。

**⚠️ 局限性**

限制在于所有测试世界都是生成的，真实数据的响应可能与生成的模型不同，且未考虑实际零售环境中的复杂因素，如库存不足和竞争对手的影响。

---

## 682. Deal With It: On Better Schedules for Listwise Reranking

**arXiv ID:** 2611.22392 | [PDF](https://arxiv.org/pdf/2611.22392v1)

**作者:** Benjamin Clavié `[一作]` (Mixedbread), Makoto P. Kato `[通讯]` (National Institute of Informatics)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的文档重排序方法DealRank，旨在提高大型语言模型（LLM）在处理文档时的效率和准确性。

**💡 创新点**

创新点在于通过轮流分配候选文档的方式来优化文档的分组，从而减少修复调用的次数，提高处理速度。

**🔧 技术方法**

使用了简单的检索模型来分析候选文档的分配方式，并与现有的调度方法（如BlitzRank）进行了比较。

**📊 数据集**

使用了六种大型语言模型（LLM）进行实验，评估不同调度方法的性能。

**📈 对比分析**

与BlitzRank相比，DealRank在两个回合内的平均nDCG@10分数仅低0.1，并且在处理速度上比BlitzRank快1.1到1.4倍，同时在59%的查询中减少了修复调用的比例。

**⚠️ 局限性**

限制在于该方法的性能依赖于检索模型的质量，且在某些情况下可能无法显著提高质量。

---

## 683. LMMR: Learned Multigranular Memory Retrieval for Small Language Model Agents

**arXiv ID:** 2611.22425 | [PDF](https://arxiv.org/pdf/2611.22425v1)

**作者:** Pengfeng Song `[一作]` (Zhejiang Wanli University), Liye Shi `[通讯]` (Zhejiang Wanli University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种学习的多粒度记忆检索方法LMMR，该方法选择下一个来源及其读取量，以解决小语言模型在多跳问题上的记忆限制。

**💡 创新点**

LMMR通过将段落和句子片段与其原始来源链接，并在训练期间配对提供相同缺失支持的视图，创新性地提高了多跳问题的回答质量。

**🔧 技术方法**

使用了条件重排序器和选择器技术，结合了粗细视图的训练，以优化信息检索和记忆使用。

**📊 数据集**

使用了来自HotpotQA、2WikiMultihopQA和MuSiQue的22,398个官方验证问题的数据集。

**📈 对比分析**

与查询扩展（QE）和其他方法相比，LMMR在HotpotQA上提高了F1分数3.06，且在输入长度相当的情况下，LMMR在HotpotQA和2WikiMultihopQA上分别获得了2.76和3.99的F1提升，且这些差异在Holm校正后仍然显著。

**⚠️ 局限性**

LMMR的局限性在于其对支持完整性的要求较高，可能导致在某些情况下未能保留所有必要的支持句子。

---

## 684. Reading Scene States: Uncovering Safety Failures in Vision-Language Models

**arXiv ID:** 2611.23206 | [PDF](https://arxiv.org/pdf/2611.23206v1)

**作者:** Tongxi Wu `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文识别了视觉语言模型（VLMs）在解释视觉事件时可能导致的安全漏洞，并提出了SceneSafetyBench基准，包含375对图像-问题对，涵盖初始、中间和最终场景状态及五种视觉表现形式。

**💡 创新点**

创新点在于识别场景状态推理作为有害帮助的来源，并提出S3TA攻击框架，将有害目标转化为场景状态，独立查询图像-问题对并汇总响应。

**🔧 技术方法**

使用了HiddenDetect技术来分析VLMs的安全相关信号，并构建了S3TA攻击框架来自动化场景状态攻击。

**📊 数据集**

使用了SceneSafetyBench和MM-SafetyBench数据集，涵盖了七个类别的有害请求，包括非法活动、仇恨言论、恶意软件生成等。

**📈 对比分析**

与现有视觉攻击方法进行比较，S3TA在八个开放权重和商业VLMs上实现了71.2%的攻击成功率（ASR），超过最强基线6.1个百分点，显示出其有效性。

**⚠️ 局限性**

限制在于文本侧过滤和视觉审核提供的保护有限，未能有效阻止S3TA的攻击，表明现有的安全防护措施未能覆盖场景基础推理的安全问题。

---

## 685. Beyond Appearance: Where Forest Visual Place Recognition Actually Fails

**arXiv ID:** 2611.21996 | [PDF](https://arxiv.org/pdf/2611.21996v1)

**作者:** Blake Matheson `[一作]` (Monash University), Kavindie Katuwandeniya `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文研究了在森林环境中进行视觉地点识别（VPR）的挑战，探讨了时间、语言、结构和几何上下文是否能改善基于外观的描述符MegaLoc的性能。

**💡 创新点**

创新点在于比较了两种无训练的时间序列匹配策略，并评估了语言、几何和结构表示作为外观基础VPR的补充，识别了它们在森林环境中的有效性限制。

**🔧 技术方法**

使用了时间序列匹配、视觉-语言模型（VLM）、场景图表示和深度导出的干茎图等技术。

**📊 数据集**

使用了WildCross基准数据集，该数据集包含了在森林环境中进行的重复遍历和SLAM导出的6DoF姿态。

**📈 对比分析**

与MegaLoc基线进行比较，序列匹配在相同方向的重访中达到了94.1%的Recall@1，而单帧MegaLoc为79.0%。在100%精度下，序列匹配的Recall@100P从50.9%提升至71.6%。

**⚠️ 局限性**

限制在于语言描述符在区分地点方面效果有限，重排序受到初始候选短名单的限制，干茎图匹配即使使用LiDAR深度也表现较弱。

---

## 686. Path-Aligned Constraint Equivariance

**arXiv ID:** 2611.22873 | [PDF](https://arxiv.org/pdf/2611.22873v1)

**作者:** Yuhao Wang `[一作]` (National University of Defense Technology), Yuanxi Peng `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的训练框架PACE，通过路径等变原理来确保生成模型在整个传输轨迹上保持线性约束。

**💡 创新点**

创新点在于提出了路径等变原理，确保在生成模型中线性约束的保持，并通过null-space参数化和约束尊重的噪声采样器实现了这一点。

**🔧 技术方法**

使用了条件流匹配（Conditional Flow Matching）和null-space参数化技术。

**📊 数据集**

使用了三个数据集，分别是二维Kolmogorov流、保守场和点归一化的多通道场，所有数据集均在128×128的周期性网格上定义。

**📈 对比分析**

与无约束回归、软惩罚回归和硬回归方法进行了比较，PACE在约束残差上减少了一个到五个数量级，且在生成质量上与数据生成管道的残差水平相匹配。

**⚠️ 局限性**

限制在于该框架目前仅适用于允许闭式形式的null-space头的线性约束，未来的研究可以扩展到非线性约束。

---

## 687. FVScore: Specification-Driven Neuro-Symbolic Video Evaluation

**arXiv ID:** 2611.22673 | [PDF](https://arxiv.org/pdf/2611.22673v1)

**作者:** Do Xuan Long `[一作]`, Yale Song `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种神经符号框架Faithfulness Video Score，用于基于规范的视频生成评估，解决了现有评估方法与人类判断对齐的问题。

**💡 创新点**

创新点在于将视频评估形式化为规范驱动的验证，能够明确区分显式和隐式要求，并通过感知工具进行验证，从而提高了与人类判断的对齐度。

**🔧 技术方法**

使用了神经符号推理技术，结合了多模态大语言模型（MLLM）和符号验证工具，采用了线性时序逻辑（LTL）灵感的软时间操作符。

**📊 数据集**

在公共和复杂的多事件基准上进行了广泛实验，使用了EvalCrafter、VideoFeedback等数据集进行评估。

**📈 对比分析**

与七种视频评估器进行了比较，结果显示该框架在与人类判断的相关性上显著优于现有评估器，尤其在物理常识和视频音频同步方面的相关性提升达44%。

**⚠️ 局限性**

局限性在于隐式约束的提取可能存在不确定性，且在复杂场景中可能会引入噪声，影响评估的准确性。

---

## 688. RAG in a Trenchcoat: When Minimal Memory Is Enough for Agentic Systems, and When It Isn't

**arXiv ID:** 2611.22179 | [PDF](https://arxiv.org/pdf/2611.22179v1)

**作者:** Jingyu Liu `[一作]` (University of Chicago), Ce Zhang `[通讯]` (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了代理记忆的有效性，提出了一种名为Bridge的简单记忆系统，并通过对比现有复杂记忆系统，展示了其在多个基准测试中的表现。

**💡 创新点**

创新点在于Bridge作为一个极简的RAG记忆系统，能够在多个基准测试中与复杂系统相媲美，且强调了现有基准测试未能有效评估记忆系统的能力。

**🔧 技术方法**

使用了混合BM25和余弦检索的技术，结合了块级观察笔记的提取和时间顺序的渲染。

**📊 数据集**

使用了LongMemEval-S、BEAM和EverMemBench等多个基准数据集进行测试。

**📈 对比分析**

与其他方法的比较显示，Bridge在标准基准测试中表现与复杂系统相当，但在应用记忆的探针测试中，Bridge在模式归纳方面表现优于其他系统，而原子事实提取器在规则应用方面表现最佳。

**⚠️ 局限性**

限制在于基准测试的规模较小，且使用的场景是合成数据，可能无法完全反映真实应用中的复杂性。

---

## 689. Auditors Struggle to Catch Sabotage In End-to-End Transcripts of AI Safety Research

**arXiv ID:** 2611.21832 | [PDF](https://arxiv.org/pdf/2611.21832v1)

**作者:** Keshav Shenoy `[一作]` (Redwood Research), Vivek Hebbar `[通讯]` (Redwood Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了在AI安全研究中，如何通过人类专家作为代理破坏者来测试对研究破坏的防御能力。研究采用红队/蓝队游戏的形式，评估审计员识别被破坏与未被破坏研究的能力。

**💡 创新点**

创新点在于首次进行端到端的绿色田野研究，生成了10个良性和12个被破坏的研究轨迹，并展示了人类审计员在识别破坏方面的困难。

**🔧 技术方法**

使用了红队/蓝队游戏的方法，红队由人类研究者组成，蓝队为审计员，研究者在实验过程中进行干预并记录实验轨迹。

**📊 数据集**

使用了10个新颖的AI安全研究问题，生成了相应的良性和被破坏的研究轨迹。

**📈 对比分析**

与人类审计员相比，前沿模型在识别破坏机制方面表现更好，AUROC分别为0.73，而人类审计员的AUROC为0.59，表明AI模型在识别破坏方面的性能优于人类。

**⚠️ 局限性**

限制在于人类审计员在识别破坏机制方面的能力较弱，许多破坏机制未被识别，且审计员在区分被破坏与未被破坏的轨迹时表现不佳。

---

## 690. CADENCE: Learning Acyclic Gaits for Model-Based Quadruped Locomotion

**arXiv ID:** 2611.23102 | [PDF](https://arxiv.org/pdf/2611.23102v1)

**作者:** Owen Sullivan `[一作]`, Mahdi Agheli `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种名为CADENCE的学习步态规划器，能够在模型基础控制器上进行步态和足迹规划。

**💡 创新点**

CADENCE的创新点在于它不依赖于固定的步态调度，而是通过强化学习在小的动作空间中选择步伐，能够在不同的模型基础控制器之间进行迁移。

**🔧 技术方法**

使用了强化学习（PPO）和模型预测控制（MPC）技术，结合了逆足迹模型（IFM）来优化步伐选择。

**📊 数据集**

在OCS2模拟环境中进行了实验，使用了不同的地形和速度（如0.1 m/s和0.2 m/s），并在多种条件下测试了CADENCE的性能。

**📈 对比分析**

与OCS2的循环步态（如OCS2-Trot和OCS2-Single）进行比较，CADENCE在多种地形和速度下的成功率显著高于OCS2的循环步态，尤其是在空隙较多的地形中表现更佳。

**⚠️ 局限性**

CADENCE的局限性在于需要在新的控制器上重新拟合IFM，并且在某些情况下，简单的足迹修正无法提升其性能。

---

## 691. WebAgents: Unbundling Agent Creation, Hosting, and Execution for an Agent-Native Web

**arXiv ID:** 2611.21978 | [PDF](https://arxiv.org/pdf/2611.21978v1)

**作者:** Dario Fenoglio `[一作]` (Università della Svizzera italiana), Ramesh Raskar `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种新的可移植执行框架，通过签名的代理包和基于OpenClaw的浏览器/运行时实现代理创建、托管和运行时的分离。

**💡 创新点**

创新点在于将代理的创建、托管和执行角色分离，使得代理行为可以在不同的创建者、托管者和运行时之间移动，而不需要将发布与特定的执行环境耦合。

**🔧 技术方法**

使用了基于OpenClaw的浏览器/运行时和签名的代理包技术。

**📊 数据集**

展示了三个不同的工作流，包括文献综述和使用语义工具的旅行代理，使用的具体数据集未明确说明。

**📈 对比分析**

通过控制实验比较了不同模型绑定下的工作流执行，结果显示所有716个注入的合同违规行为被成功阻止，同时295个良性控制被接受，预执行处理的中位时间低于0.5秒。

**⚠️ 局限性**

限制在于当前原型仅支持通过明确的HTTPS URL、实验性网站广告链接或批准的本地工件打开包，未来需要扩展到更广泛的发现和集成。

---

## 692. Learning In-Context Adaptivity for LLMs through Continual Gameplay

**arXiv ID:** 2611.21710 | [PDF](https://arxiv.org/pdf/2611.21710v1)

**作者:** Jiale Liu `[一作]` (Penn State University), John Schultz `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言模型在重复的不完全信息博弈中的适应性，特别是在Leduc Hold'em和No-Limit Hold'em游戏中，比较了有记忆和无记忆的表现。

**💡 创新点**

提出了一个评估协议，识别了模型在记忆使用中的执行差距，并通过后训练方法提高了模型的适应性。

**🔧 技术方法**

使用了监督微调和轨迹级强化学习技术，结合上下文压缩来优化模型的决策过程。

**📊 数据集**

使用了Leduc Hold'em和No-Limit Hold'em的对手池作为数据集，进行模型训练和评估。

**📈 对比分析**

与基线模型（如RL和求解器）进行比较，发现记忆对开放权重模型的性能提升有限，而前沿模型在某些情况下表现不一致。后训练的9B模型在未见对手上达到了前沿水平的经验增益。

**⚠️ 局限性**

研究的局限性在于对手策略在每场比赛中保持不变，未来的工作可以扩展到多智能体设置，以探索更复杂的适应性问题。

---

## 693. Computational Hardness of Online Stochastic Matching

**arXiv ID:** 2611.23157 | [PDF](https://arxiv.org/pdf/2611.23157v1)

**作者:** Yifeng Teng `[一作]` (Google), Zongjun Yang `[通讯]` (Columbia University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在线随机匹配问题，特别是在随机到达顺序下的计算复杂性，首次证明了已知独立同分布（i.i.d.）到达情况下的在线最优解计算是NP-hard的。

**💡 创新点**

创新点在于首次提供了随机到达顺序下在线随机匹配的计算难度结果，特别是在i.i.d.到达情况下的近似计算是NP-hard的。

**🔧 技术方法**

使用了复杂的图论和计算复杂性理论，特别是通过构造从最大割问题到在线匹配问题的归约来证明计算难度。

**📊 数据集**

使用了简单的立方图作为数据集，通过构造在线随机匹配实例来展示计算复杂性。

**📈 对比分析**

与现有的多项式时间算法进行比较，证明了在已知i.i.d.到达情况下，任何多项式时间近似方案都无法达到在线最优解，性能上存在显著差距。

**⚠️ 局限性**

限制在于该研究主要集中于特定的到达模型（如i.i.d.），未能涵盖其他可能的到达模式，且在实际应用中可能面临计算复杂性高的问题。

---

## 694. Sub-Meter Canopy Height Synthesis from Sentinel-2 via Anchored Residual Flow

**arXiv ID:** 2611.22636 | [PDF](https://arxiv.org/pdf/2611.22636v1)

**作者:** Kiarie Ndegwa `[一作]` (Vibrant Planet), Guy C Bayes `[通讯]` (Terradot)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究利用免费的10米分辨率Sentinel-2影像生成0.5米的树冠高度模型（CHM），以支持森林碳核算、野火燃料映射和树木生态学研究。

**💡 创新点**

创新点在于提出了两个操作点：一个是针对低失真的确定性回归器（Anchor），另一个是基于条件流匹配模型（AR-flow），后者在学习的CHM潜在空间中训练，旨在提高感知保真度。

**🔧 技术方法**

使用了确定性回归和条件流匹配技术，结合了预训练的CHM自编码器和DINOv3特征。

**📊 数据集**

使用的数据集包括197,000个与LiDAR配对的图块，覆盖美国大陆的69个生态区，并在2,530个保留图块上进行评估。

**📈 对比分析**

与八个基线方法进行比较，Anchor模型在MAE上达到2.83米，AR-flow在感知指标上有所提升，但MAE为3.68米。Anchor在27个生态区的MAE低于Meta CHMv2，而AR-flow在感知指标上优于Meta。

**⚠️ 局限性**

限制在于生成的亚像素细节代表的是合理的结构而非实际测量的单棵树，且在个体树检测方面表现较弱。

---

## 695. Say Less, Hallucinate Less? Diagnosing and Restoring Object Coverage in Preference-Optimized Vision-Language Models

**arXiv ID:** 2611.23179 | [PDF](https://arxiv.org/pdf/2611.23179v1)

**作者:** Hansang Lee `[一作]` `[通讯]` (Seoul Women's University), Hansang Lee (Seoul Women's University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了偏好优化（PO）在大型视觉语言模型中的应用，发现LLaVA-1.5-7B的四个公共PO检查点降低了对象覆盖率，并提出了基于基础模型的对象类别恢复方法。

**💡 创新点**

创新点在于提出了基于基础模型的对象类别恢复方法，该方法无需训练即可将高置信度的基础模型类别附加到PO标题中，从而提高了对象召回率。

**🔧 技术方法**

使用了LLaVA-1.5-7B模型，并通过正向和负向的结束序列（EOS）logit偏差来控制模型的停止和生成长度。

**📊 数据集**

使用了COCO数据集中的两个不重叠的子集进行实验，具体为CHAIR-500和其他500个COCO val2014图像。

**📈 对比分析**

通过与基础模型的停止曲线进行比较，发现基础模型的早期停止能够重现58%到86%的CHAIRi增益，而延长PO标题几乎无法恢复丢失的召回率。

**⚠️ 局限性**

限制在于只研究了LLaVA-1.5-7B这一基础模型，且未考虑属性和关系的幻觉问题，恢复方法仅覆盖每个基准识别的对象类别或名词词干。

---

## 696. LeViT-UNet++: Towards Balanced Fine-Grained Details and Context for Efficient Medical Image Segmentation

**arXiv ID:** 2611.22482 | [PDF](https://arxiv.org/pdf/2611.22482v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 697. The Multivariate Feedback Particle Filter for State-dependent Multiplicative Noises and Its Application to Attitude and Pose Estimations

**arXiv ID:** 2611.22518 | [PDF](https://arxiv.org/pdf/2611.22518v1)

**作者:** Jingyao You `[一作]` (Beihang University), Xue Luo `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种多变量反馈粒子滤波器（FPF-DM），用于处理具有状态依赖乘法噪声的非线性系统，特别应用于航天器姿态和位姿估计。

**💡 创新点**

创新点在于通过测度变换和重要性重加权，将状态依赖的逆噪声协方差纳入非均匀加权的高斯混合中，从而恢复多项式结构，并利用Hermite反向递归和不完全伽马函数提供闭式反馈增益。

**🔧 技术方法**

使用了反馈粒子滤波（FPF）技术，结合了测度变换和重要性重加权的方法。

**📊 数据集**

在4D姿态跟踪和8D点云配准问题上进行了仿真，使用了标准的斯坦福兔子数据集。

**📈 对比分析**

与基线滤波器（如常数增益FPF和标准粒子滤波器）进行比较，结果显示使用仅10个粒子的QFPF-DM和DQFPF-DM在估计精度和计算性能上均优于使用50个或200个粒子的基线方法。

**⚠️ 局限性**

限制在于该方法依赖于状态依赖的噪声模型，可能在某些极端情况下表现不佳，且对粒子数量的选择敏感。

---

## 698. RegimeCouncil: Paired-Evidence Routing For Memory Conflict Governance In LLM Agents

**arXiv ID:** 2611.22113 | [PDF](https://arxiv.org/pdf/2611.22113v1)

**作者:** Qingcan Kang `[一作]` (Huawei Noah's Ark Lab), Mingxuan Yuan `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的在线记忆冲突治理方法RegimeCouncil，旨在解决语言代理在使用长期记忆时如何处理过时信息的问题。

**💡 创新点**

创新点在于引入了局部顾问和全局顾问的双重顾问机制，通过不对称的元治理者在线调解两者的决策，从而在不重新训练模型的情况下优化记忆使用。

**🔧 技术方法**

使用了一种轻量级的配对证据路由器RegimeCouncil，结合局部顾问、全局顾问和元治理者，进行在线决策。

**📊 数据集**

使用了LongMemEval更新流和真实世界历史漂移数据集TempLAMA和TAQA，评估了模型在不同情况下的表现。

**📈 对比分析**

与固定范围策略相比，RegimeCouncil在主要分歧事件上提高了5.2到7.3个百分点，并在所有30个保留设置中表现优于固定范围策略，消除任何模块都会降低2到7个百分点的准确性。

**⚠️ 局限性**

局限性在于该方法依赖于延迟反馈的有效性，且在某些情况下可能无法完全避免跨事实干扰。

---

## 699. Proximal Targets: Training Looped Refiners on Problems with Many Correct Answers

**arXiv ID:** 2611.22737 | [PDF](https://arxiv.org/pdf/2611.22737v1)

**作者:** Syed Affan Daimi `[一作]` `[通讯]` (International Institute of Information Technology), Syed Affan Daimi (International Institute of Information Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的训练方法，称为近端目标（proximal target），用于解决具有多个正确答案的循环细化问题。

**💡 创新点**

创新点在于通过选择每次传递中最接近当前输入的正确答案作为目标，避免了答案混合的问题，使得每个答案成为固定点，从而提高了模型的收敛性和解决问题的能力。

**🔧 技术方法**

使用了循环变换器（looped transformers）、递归推理（recursive reasoning）、深度监督（deep supervision）等技术。

**📊 数据集**

使用了多个数据集，包括具有2到64个答案的数独（Sudoku）问题、稀疏数独（sparse Sudoku）、200节点的图着色（3-colouring）等。

**📈 对比分析**

与GRAM（随机潜变量推理器）等方法进行比较，近端目标在解决多答案数独问题时的正确率为98.8%，而GRAM为57.7%；在20次运行中，近端目标找到的答案比例为53.6%，而GRAM为37.4%。

**⚠️ 局限性**

限制在于训练需要完整的答案集或能够到达答案的方式；如果只随机选择部分答案，则会导致答案混合的问题。

---

## 700. Domain-Tuned Diffusion Super-Resolution for 4K Facial Asset Maps

**arXiv ID:** 2611.21890 | [PDF](https://arxiv.org/pdf/2611.21890v1)

**作者:** Elly Akhoundi `[一作]` (Electronic Arts), Étienne Danvoye `[通讯]` (Electronic Arts)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对面部资产超分辨率的领域调优扩散上采样器，将面部UV贴图从1K分辨率提升到4K分辨率。

**💡 创新点**

创新点在于使用轻量级的LoRA模块对冻结的SDXL骨干网络进行适应，同时利用Tile ControlNet保持输入结构，恢复细节。

**🔧 技术方法**

采用了领域调优的扩散模型，结合了LoRA模块和Tile ControlNet。

**📊 数据集**

使用了来自28个摄像头的12MP设备捕获的4K真实地图作为数据集。

**📈 对比分析**

与双线性上采样、Real-ESRGAN和预训练的Stable Diffusion ×4进行比较，结果显示该方法在细节恢复上更为清晰和真实，LPIPS从0.21降低到0.08。

**⚠️ 局限性**

局限性在于评估使用的是来自单一捕获管道的持出地图，未测试在不同捕获条件、扫描质量和其他缩放因子下的鲁棒性。

---

## 701. Where Should Physics Enter? Information, Identifiability and Calibration in Probabilistic Sea Surface Temperature Forecasting

**arXiv ID:** 2611.23129 | [PDF](https://arxiv.org/pdf/2611.23129v1)

**作者:** Parmitha Vangapandu `[一作]` (Indian Institute of Information Technology Dharwad), Santanu Bhattacharya `[通讯]` (MIT Media Lab Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文测试了在异方差ConvLSTM中引入物理信息的对流-扩散先验的必要性，以改进海表温度（SST）预测，使用了1982年至2024年间的NOAA OISST v2.1数据和一个已知流速的合成基准。

**💡 创新点**

创新点在于通过控制实验分离了物理信息的三个组成部分：正则化项、传输结构和速度信息，发现物理正则化对预测没有显著帮助，而提供的流速主要作为信息起作用。

**🔧 技术方法**

使用了异方差ConvLSTM模型，结合了物理信息的正则化项和传输步骤。

**📊 数据集**

使用了NOAA OISST v2.1（1982-2024）和GLORYS12再分析流速数据，合成基准数据也用于测试。

**📈 对比分析**

与其他方法的比较显示，物理正则化项没有显著改善预测性能，模型在短期和长期的校准存在问题，使用提供的流速作为输入时，模型的性能有所提升，但整体效果仍不如简单的网络。

**⚠️ 局限性**

限制在于只测试了一个区域和一个分辨率，模型的超参数来自早期版本，未进行更广泛的验证和比较，且未考虑更复杂的动态模型。

---

## 702. XM-Sparse: High-Throughput Sparse-Attention LLM Serving over Scale-up Memory Pools

**arXiv ID:** 2611.22280 | [PDF](https://arxiv.org/pdf/2611.22280v1)

**作者:** Yijun Sun `[一作]` (Netpreme), Zhizhen Zhong `[通讯]` (Netpreme)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种新的稀疏注意力卸载系统，旨在通过扩展内存池提高大规模语言模型的解码吞吐量。

**💡 创新点**

创新点在于首次将稀疏注意力卸载系统与扩展内存池结合，采用针对不同访问模式的I/O机制来优化KV缓存和K索引缓存的卸载。

**🔧 技术方法**

使用了融合的TMA检索和GPU发起的DMA技术来实现KV和K索引的高效卸载。

**📊 数据集**

在DeepSeek-V3.2和GLM-5.3模型上进行了实验，使用了模拟的扩展内存池进行评估。

**📈 对比分析**

与基于PCIe的ECHO和HiSparse系统相比，该系统在解码吞吐量上分别提高了4.98倍和4.52倍，并且在相同平台上的简单集成中提高了1.34倍。

**⚠️ 局限性**

限制在于尽管通过卸载提高了吞吐量，但HBM容量仍然是瓶颈，进一步的性能提升需要更多的HBM释放。

---

## 703. Beyond the Camera: A Tri-Modal Driving World Model for Future RGB, Thermal, and LiDAR Forecasting in Adverse Weather

**arXiv ID:** 2611.22669 | [PDF](https://arxiv.org/pdf/2611.22669v1)

**作者:** Faysal Mehrab Chowdhury `[一作]` (North South University), Savannah Shannon `[通讯]` (Clark Atlanta University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了在真实恶劣天气条件下，RGB、热成像和LiDAR传感器的联合预测，使用DSERT-RoLL数据集进行实验。

**💡 创新点**

首次提出了同时预测RGB、热成像和LiDAR数据的三传感器模型，并展示了在不同天气条件下的有效性。

**🔧 技术方法**

使用了视频世界模型Cosmos-Predict2.5进行后训练，并通过低秩适应技术对每个传感器进行优化。

**📊 数据集**

使用了DSERT-RoLL数据集，该数据集包含在六种天气条件下的真实序列，包括清晰、雾、轻雨、重雨、轻雪和大雪。

**📈 对比分析**

与持久性基线进行比较，后训练的RGB和热成像预测在所有58个验证序列中表现优于压缩匹配的持久性，LiDAR的误差在2秒内降低了2.1米。

**⚠️ 局限性**

主要限制在于计算资源，训练仅在约30GB的子集上进行，且未能充分利用完整数据集进行训练和更多预测样本的生成。

---

## 704. Stochastic Diversity Windows: Diagnosing Noise Influence in Robotic Diffusion Sampling

**arXiv ID:** 2611.22743 | [PDF](https://arxiv.org/pdf/2611.22743v1)

**作者:** Tina Lakhani `[一作]` (Cardiff University), Ze Ji `[通讯]` (Cardiff University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种逐步反事实诊断方法，用于测试采样噪声变化对最终输出的影响，并在多个机器人基准上进行评估。

**💡 创新点**

创新点在于定义了随机多样性窗口（SDWs），用于识别采样器随机性对最终输出影响最大的时间段，并提出了两种基于此窗口的干预方法。

**🔧 技术方法**

使用了反事实诊断技术，通过重新采样每个去噪时间步的噪声来测量其对最终输出的影响。

**📊 数据集**

使用了三个机器人基准数据集：Maze2D、运动规划扩散（MPD）和推送任务（PushT）。

**📈 对比分析**

通过与五种模型-采样器-条件配置进行比较，结果显示不同配置下随机影响的时间敏感性差异显著，干预方法在保持可行性、成本和多样性方面表现良好。

**⚠️ 局限性**

限制在于该诊断需要源代码级的采样器访问权限，并且在评估每个时间步时需要显著的计算成本，尽管这一成本在冻结的预训练模型和采样器上只需支付一次。

---

## 705. DGV-DLIVO: Degeneracy-Guided Visual Fusion for Direct LiDAR-Inertial-Visual Odometry in Degenerate Tunnel Environments

**arXiv ID:** 2611.22385 | [PDF](https://arxiv.org/pdf/2611.22385v1)

**作者:** Junrui Zhang `[一作]` `[通讯]` (Tongji University), Junrui Zhang (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为DGV-DLIVO的框架，用于在严重退化的隧道环境中实现直接的LiDAR-惯性-视觉里程计，通过利用LiDAR的退化方向指导视觉信息的选择和融合。

**💡 创新点**

创新点在于将LiDAR的退化方向作为主动指导信号，选择有用的视觉区域，建立LiDAR锚定的视觉关联，并进行子空间选择性融合，从而在几何和视觉退化严重的环境中保持连续的运动估计。

**🔧 技术方法**

使用了LiDAR、惯性测量单元（IMU）和单目相机的融合技术，结合了退化引导的视觉约束构建策略和子空间选择性融合策略。

**📊 数据集**

在真实的水电隧道环境中进行实验，使用了在隧道中设置的稀疏地标作为参考，评估不同方法的性能。

**📈 对比分析**

与现有的LIO、LIVO和VIO方法进行比较，DGV-DLIVO在40米的隧道序列中实现了91.4%的进展比率，显著提高了纵向恢复和轨迹稳定性，而其他方法则表现出停滞、振荡、严重的欠旅行或漂移。

**⚠️ 局限性**

限制在于实验数据集的保密性，无法公开获取，且在不同隧道几何形状和点云分布下，退化光谱的表现可能会有所不同。

---

## 706. Better Rankers, Worse Retrievers: The Hidden Cost of RL Post-Training in Generative Recommendation

**arXiv ID:** 2611.22838 | [PDF](https://arxiv.org/pdf/2611.22838v1)

**作者:** Yuheng Chen `[一作]` (Hong Kong University of Science and Technology), Can Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

论文探讨了生成推荐系统中强化学习（RL）后训练的隐性成本，发现虽然RL在列表顶部提高了准确性，但在候选列表的召回率上却降低了性能。

**💡 创新点**

创新点在于首次系统性地研究了召回率损失，识别了其原因，并提出了简单的解决方案——超出回滚的锚定（out-of-rollout anchoring），以恢复召回率。

**🔧 技术方法**

使用了生成推荐模型和强化学习技术，特别是GRPO（基于奖励的策略优化）进行后训练。

**📊 数据集**

使用了五个数据集进行实验，包括Office、Industrial、Beauty、Toys和Sports，涵盖了不同的领域和目录大小。

**📈 对比分析**

通过与SFT模型的比较，发现RL后训练在小K（如前十个结果）上表现良好，但在K较大（如500）时召回率显著下降。通过控制实验验证了召回率损失的原因，并提出了超出回滚的锚定作为解决方案，恢复了召回率。

**⚠️ 局限性**

限制在于RL后训练的目标仅针对每个用户记录的下一个项目，导致模型对其他候选项的惩罚，尽管这些候选项可能是相似用户的下一个项目。

---

## 707. Context Has a Price: Adding and Removing Context for LLM Coding Agents under Prefix Caching

**arXiv ID:** 2611.22238 | [PDF](https://arxiv.org/pdf/2611.22238v1)

**作者:** Zerui Yang `[一作]` (Huawei Technologies Ltd), Bo Bai `[通讯]` (Huawei Technologies Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在编码代理与语言模型交互中，添加和移除上下文对任务解决成本的影响，进行了超过20,000次的实验。

**💡 创新点**

提出了一种新的成本计算方法，考虑了缓存输入和输出的不同价格，并揭示了上下文管理方法在不同价格下的效果差异。

**🔧 技术方法**

使用了多种开源模型（如Qwen系列）和实验设置，包括最小bash代理和mini-SWE代理，进行了控制实验。

**📊 数据集**

使用了SWE-bench Verified、SWE-bench和Loc-Bench等数据集，进行了大量的实验。

**📈 对比分析**

通过比较添加和移除上下文的方法，发现添加上下文可以减少搜索命令，但会增加成本，而移除旧观察的效果依赖于价格，某些情况下会节省成本，某些情况下则会增加成本。

**⚠️ 局限性**

研究的局限性在于所有模型均为本地服务的开放模型，且实验设置未能涵盖所有可能的API代理行为，价格计算基于记录的令牌流而非实际API发票。

---

## 708. What Do LLMs Use to Judge Synthesis Recipes? An Experimental Analysis on 240 Solid-State Syntheses

**arXiv ID:** 2611.22623 | [PDF](https://arxiv.org/pdf/2611.22623v1)

**作者:** Derek van Tilborg `[一作]`, Joonatan E. M. Laulainen `[通讯]` (Altrove)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究进行了240个实验的固态合成基准测试，评估了大型语言模型（LLM）在无监督情况下对合成配方的排序能力。

**💡 创新点**

创新点在于构建了一个基于实验结果的合成基准，而不是依赖于已发布的配方，并首次评估了LLM作为配方排序器的性能。

**🔧 技术方法**

使用了Bradley-Terry模型对配方进行排序，并通过X射线衍射测量合成产物的纯度。

**📊 数据集**

使用了八种已知和未知的无机氧化物材料的合成数据集，共进行了240次实验。

**📈 对比分析**

与实际纯度相比，LLM的配方排序准确率为0.72，已知目标的排序性能高于未知目标，但统计上没有显著差异。

**⚠️ 局限性**

限制在于基准测试的规模较小，且目标材料主要受热驱动，未来需要更多实验来验证这些发现是否适用于其他类型的材料合成。

---

## 709. ReBoT-MiCS: Reliability-Aware Bounding Time for Enhancing Timing Behavior of Embedded Mixed-Criticality Systems

**arXiv ID:** 2611.22588 | [PDF](https://arxiv.org/pdf/2611.22588v1)

**作者:** Mohammad Abbasinia `[一作]` (Ruhr University Bochum), Akash Kumar `[通讯]` (Ruhr University Bochum)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为ReBoT-MiCS的可靠性意识低WCET确定方案，旨在平衡嵌入式混合关键性系统中的利用率、模式切换概率和服务质量，同时满足实时性和可靠性约束。

**💡 创新点**

创新点在于将PFH作为WCET^LO确定的显式设计约束，提出了一种可靠性意识的低WCET确定方案，能够在满足PFH约束的同时提高系统利用率和降低模式切换概率。

**🔧 技术方法**

采用了基于Chebyshev定理的分析方法来确定低WCET，并结合了瞬态故障模型进行可靠性分析。

**📊 数据集**

使用了多种合成基准和真实嵌入式基准（如MiBench和AXBench）进行实验评估。

**📈 对比分析**

与现有的WCET^LO选择方法进行比较，结果显示该方案在满足PFH约束的同时，利用率提高了最多27.6%，平均提高了11.67%，模式切换概率降低了最多5.48%。

**⚠️ 局限性**

限制在于该方案主要针对单核处理器，未来工作将扩展到支持多核异构平台，并考虑任务特定的降级因子。

---

## 710. A Forensic Analysis of Metadata and Information Leakage in Swedish Public Documents

**arXiv ID:** 2611.21922 | [PDF](https://arxiv.org/pdf/2611.21922v1)

**作者:** Noor Abo Hasan `[一作]` (Stockholm University), Monia Lönn `[通讯]` (Stockholm University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究调查了来自30个瑞典公共部门机构的公开PDF文档中的元数据和结构内容泄露。通过自动化收集管道和基于ExifTool的提取，分析了2876份文档，并使用R0-R3风险分类法进行了分类。

**💡 创新点**

创新点在于首次系统性地分析了瑞典公共部门文档中的元数据泄露，特别是通过增量更新机制恢复的敏感信息，并将其与NIS2指令的实施进行比较。

**🔧 技术方法**

使用了ExifTool进行元数据提取，并采用Python编写的自动化收集管道进行数据处理和分析。

**📊 数据集**

使用的数据集包括来自30个瑞典公共部门机构的2876份公开PDF文档。

**📈 对比分析**

通过卡方检验比较了不同机构层级和时间段的元数据泄露情况，结果显示在机构层级之间存在显著差异，且在NIS2实施后，元数据泄露率有所下降，但仍有44.9%的文档泄露敏感元数据。

**⚠️ 局限性**

本研究的局限性在于仅分析了公开的PDF文档，未涵盖未发布文件和内部系统。此外，自动化分类可能引入假阳性和假阴性，且后NIS2时间段样本较小，需进一步验证其结果的持久性。

---

## 711. Evaluating the Evaluation of Music Generation Models with Generative Psychometrics

**arXiv ID:** 2611.21809 | [PDF](https://arxiv.org/pdf/2611.21809v1)

**作者:** João Carlos Santana Silveira `[一作]` (Federal University of Minas Gerais), Lucas N. Ferreira `[通讯]` (Federal University of Minas Gerais)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本论文对2020年至2025年间ISMIR音乐生成论文中的主观评估方法进行了映射，并应用生成心理测量学评估其项目的稳定性。

**💡 创新点**

创新点在于提出了一种统一的用户研究方法，通过分析284个标准化项目，识别出六个稳定的质量维度：节奏、旋律、音频、作曲、结构和歌词。

**🔧 技术方法**

使用了生成心理测量学方法AI-GENIE，结合大语言模型（LLM）进行项目标准化和上下文嵌入提取。

**📊 数据集**

使用的数据集为2020年至2025年间的150篇ISMIR音乐生成论文中的主观评估项目，共提取284个项目。

**📈 对比分析**

通过独特变量分析（UVA）和引导探索图分析（bootEGA）比较方法，最终确定了57个独特项目，并将其分为六个稳定的社区，显示出高结构稳定性（84%至100%）。

**⚠️ 局限性**

限制在于本研究仅限于单一会议，且公开问卷的稀缺限制了分析的广泛性，未来可通过新的听力测试进行实证验证。

---

## 712. Leveraging Segmentation Representations for Efficient Classification: A Self-Configuring Framework Based on nnU-Net

**arXiv ID:** 2611.22380 | [PDF](https://arxiv.org/pdf/2611.22380v1)

**作者:** Renjun Cai `[一作]` (University of Waikato), Félix Fuentes-Hurtado `[通讯]` (University of Alicante)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种自配置框架，扩展了nnU-Net工作流程，以支持3D医学成像中的分割和分类。该框架重用分割模型学习到的表示，用于图像级分类。

**💡 创新点**

创新点在于通过轻量级分类器重用分割模型的编码器特征和预测的分割概率，并探索了两种信息共享策略。

**🔧 技术方法**

使用了nnU-Net作为基础框架，并结合了特征增强和部分编码器适应的技术。

**📊 数据集**

使用了FLARE 2026任务提供的多个公共数据集，包括EGCT和PHLF等。

**📈 对比分析**

通过五折交叉验证比较了六种配置（A0-A5），结果显示更复杂的配置并未一致提高分类性能，最终框架在公共验证排行榜上获得了0.8056的总体得分，并在FLARE 2026任务中排名第一。

**⚠️ 局限性**

限制在于分类架构和自动配置规则的范围有限，分类性能在高度不平衡的数据集上仍然不确定，且未直接评估分割错误对分类的影响。

---

## 713. BQ-LoRA: Binary-Quantized Low-Rank Adapters as Implicit Regularizers for Parameter-Efficient Fine-Tuning

**arXiv ID:** 2611.22334 | [PDF](https://arxiv.org/pdf/2611.22334v1)

**作者:** Juyoung Park `[一作]` (Independent Researcher), Heejun Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的二进制低秩适配器BQ-LoRA，用于量化大语言模型的参数高效微调，替代了FP16适配器的精度。

**💡 创新点**

BQ-LoRA通过使用每个适配器参数一个二进制位，结合学习的每层缩放因子，显著减少了适配器存储需求，同时保持或提高了性能。

**🔧 技术方法**

使用了二进制量化和缩放的直通估计器（Scaled Straight-Through Estimator）技术。

**📊 数据集**

在Qwen2.5（1.5B–72B）和LLaMA-2（7B/70B）数据集上进行了实验。

**📈 对比分析**

与QLoRA相比，BQ-LoRA在每个规模上都超越了QLoRA的基准平均值，性能提升在0.8到3.5点之间；在LLaMA-2上保持竞争力，但不总是领先于最强的FP16适配器基线。

**⚠️ 局限性**

局限性包括仅使用了{q,k,v,o}适配器目标，未测试在更高秩或其他类型适配器上的表现；LLaMA-2-70B结果和混合数据集消融实验为单种子，且只评估了英语指令微调。

---

## 714. From Attack Tree to Model Checker: Binding CAPEC Patterns to Verify Robotic System Security

**arXiv ID:** 2611.22408 | [PDF](https://arxiv.org/pdf/2611.22408v1)

**作者:** Juel Hussain `[一作]` (University of Manchester), Marie Farrell `[通讯]` (University of Manchester)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本研究提出了一种机制化的管道，将攻击树的每个叶子节点映射到CAPEC攻击模式，并用一阶逻辑表示为规则，从而实现对自主和机器人系统的安全验证。

**💡 创新点**

创新点在于将CAPEC攻击模式转化为系统特定的攻击者模型，并通过绑定机制实现对系统的安全性验证，解决了手动构建攻击者模型的低效问题。

**🔧 技术方法**

使用了一阶逻辑和CAPEC攻击模式的操作语义，结合模型检查工具SPIN进行验证。

**📊 数据集**

使用了CAPEC攻击模式作为数据集，并结合特定的机器人系统配置进行验证。

**📈 对比分析**

通过与SPIN的比较，评估了管道的有效性，验证了在不同控制配置下攻击的适用性和可达性，结果显示该方法能够有效识别系统的安全漏洞。

**⚠️ 局限性**

限制在于该方法依赖于CAPEC模式的完整性和准确性，且在处理复杂攻击路径时可能面临挑战。

---

## 715. The Only Edge Was Memorized in Training: LLM Trading Agents Around Earnings and Bad News

**arXiv ID:** 2611.22177 | [PDF](https://arxiv.org/pdf/2611.22177v1)

**作者:** Jackson Wang `[一作]` `[通讯]`, Jackson Wang

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文测试了大型语言模型（LLM）交易代理在新闻发布前后的表现，特别是在已知日期的收益公告前和坏消息后的交易时机。

**💡 创新点**

创新点在于通过无前瞻性基准测试LLM代理在特定时刻的交易能力，并与简单的交易策略进行比较，发现LLM代理的表现并未优于简单的交易策略。

**🔧 技术方法**

使用了大型语言模型（如GPT-5.6 Luna和Gemini 3.5 Flash）进行多步骤决策和单步决策的比较。

**📊 数据集**

使用了2020、2022和2026年间的大型美国股票数据集，特别关注收益公告和坏消息下跌的事件。

**📈 对比分析**

与简单的交易规则和逻辑回归模型进行比较，结果显示没有任何代理的表现超过简单的买入策略，且在坏消息后，代理的选择未能优于简单的买入所有下跌的策略。

**⚠️ 局限性**

限制在于研究仅关注大型上市公司，未考虑那些因坏消息而失败的公司，且事件样本来自三个特定时期，可能无法全面反映市场情况。

---

## 716. Mixed-Size Placement Initialization via Stiefel-Manifold Sequential Subspace Optimization

**arXiv ID:** 2611.22402 | [PDF](https://arxiv.org/pdf/2611.22402v1)

**作者:** Pengwen Chen `[一作]` (National Tsing Hua University), Ying Yuan `[通讯]` (University of California San Diego)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种混合尺寸的VLSI设计初始化方法，将初始化问题表述为带有显式一阶和二阶约束的二次约束二次规划（QCQP）。

**💡 创新点**

创新点在于将二次约束几何结构显式化，通过度量均衡将问题转化为Stiefel流形，从而实现可行性保持的Riemannian优化和牛顿共轭梯度二阶步骤。

**🔧 技术方法**

使用了二次约束二次规划（QCQP）、Riemannian优化和Stiefel流形等技术。

**📊 数据集**

使用了20个基准测试，包括四个TILOS基准和16个MMS基准。

**📈 对比分析**

与现有的启发式初始化方法相比，提出的方法在RePlAce下平均提高了13.8%的加权半周长（wHPWL），在DREAMPlace下提高了3.5%，在AutoDMP下提高了4.8%。

**⚠️ 局限性**

限制在于初始化运行时间较长，范围从2.4到87.8分钟，且在Mempool组上达到约两小时，未来需要进一步优化。

---

## 717. TRIDENT: Efficient Triple-Task Learning of Dehazing, Depth, and Uncertainty Estimation for Underwater 3-D Robot Visual Perception

**arXiv ID:** 2611.22224 | [PDF](https://arxiv.org/pdf/2611.22224v1)

**作者:** Geonmo Yang `[一作]` (Inha University), Younggun Cho `[通讯]` (Inha University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新颖的基于学习的感知系统，解决了水下图像增强、深度估计和不确定性等多维视觉任务。

**💡 创新点**

创新点在于提出了TRIDENT模型，该模型采用三条并行解码器和一个主干结构，以高效共享特征，并在执行多个任务时保持较高的效率。

**🔧 技术方法**

使用了深度学习技术，具体包括卷积神经网络（CNN）和多任务学习框架。

**📊 数据集**

使用了多个标准数据集进行实验评估，包括Joint-ID的测试集和L400等。

**📈 对比分析**

与其他现有方法相比，TRIDENT在图像增强和深度估计上显著优于其他模型，同时在内存大小和推理时间上表现出更好的效率。

**⚠️ 局限性**

限制在于深度尺度未知，这可能导致在大规模3D重建或SLAM中的尺度问题。

---

## 718. Decoding Without a Decoder: Direct Pixel Prediction from Diffusion Transformers

**arXiv ID:** 2611.22287 | [PDF](https://arxiv.org/pdf/2611.22287v1)

**作者:** Hansheng Chen `[一作]` (Stanford University), Leonidas Guibas `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的方法Decoding Without a Decoder (DWD)，通过线性头直接从扩散变换器生成最终像素，省去了解码器的复杂性和推理成本。

**💡 创新点**

DWD通过将去噪头与像素预测头分离，改善了像素质量，同时降低了推理成本，尤其在视频生成中减少了细节失真和闪烁。

**🔧 技术方法**

使用了扩散变换器（DiT）和线性输出头，进行微调以优化像素质量损失，同时保持去噪头的预训练模型。

**📊 数据集**

在ImageNet、文本到图像生成和视频生成等多个数据集上进行了评估。

**📈 对比分析**

与传统的扩散模型相比，DWD在去掉解码器后，几乎所有FID指标都有所改善，视频生成中的变形误差降低了23%。在文本到图像生成中，DWD的ImageNet FID从1.55提高到1.43，成为像素流模型中的新状态。

**⚠️ 局限性**

当解码器已经很强时，DWD的质量提升效果会减小。此外，尚未证明线性头在超过16×16像素的空间补丁大小上是足够的，可能需要更强的头部设计。

---

## 719. AR-WAM: efficient streaming world-action modeling for autonomous driving

**arXiv ID:** 2611.21851 | [PDF](https://arxiv.org/pdf/2611.21851v1)

**作者:** Baiting Luo `[一作]` (Toyota Motor North America Research and Development - InfoTech Labs), Nejib Ammar `[通讯]` (Toyota Motor North America Research and Development - InfoTech Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种高效的流式世界-动作模型AR-WAM，结合了自回归时间建模和单次未来条件化，以提高自动驾驶中的决策效率。

**💡 创新点**

AR-WAM通过增量更新视觉历史和一次性预测未来特征，避免了重复编码和迭代生成视频的成本，从而提高了动态交通环境中的泛化能力。

**🔧 技术方法**

使用了自回归时间建模和单次前向传递的预测特征，结合了强化学习来优化驾驶奖励。

**📊 数据集**

在NAVSIM-v1数据集上进行训练和评估，该数据集包含103,288个场景。

**📈 对比分析**

与其他强大的端到端世界-动作模型相比，AR-WAM在NAVSIM-v1上获得了91.5 PDMS，经过强化学习后达到94.0 PDMS，且在动态设置中表现出更低的延迟和更高的性能。

**⚠️ 局限性**

AR-WAM的局限性在于其依赖于单一前向传递的预测特征，可能在某些复杂场景中无法充分利用历史信息。

---

## 720. CoBoVLA: Decoupling Language from Action with Discrete Codes for Robust and Interpretable Robot Manipulation

**arXiv ID:** 2611.22395 | [PDF](https://arxiv.org/pdf/2611.22395v1)

**作者:** Mehul Arora `[一作]` (University of California San Diego), Ambuj Agrawal `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的视觉-语言-动作（VLA）模型，名为CodeBookVLA，通过学习的离散可解释代码解耦语言理解、规划和控制。

**💡 创新点**

创新点在于将语言理解与控制通过离散接口连接，使得模型在面对指令的不同表述时更具鲁棒性，并且能够更容易地诊断失败原因。

**🔧 技术方法**

使用了文本编码器、视觉规划器和动作执行器等技术，采用了基于最小描述长度（MDL）分割的动作代码。

**📊 数据集**

使用了LIBERO基准数据集，并生成了27,222个与40个LIBERO指令相关的同义句。

**📈 对比分析**

与SmolVLA进行比较，CodeBookVLA在标准指令上成功率为91.4%，而SmolVLA为82.8%；在未见的同义句上，CodeBookVLA成功率为81.1%，而SmolVLA为36.5%。

**⚠️ 局限性**

限制在于仅在模拟环境中进行训练和评估，且只针对10个基础指令的同义句进行训练，可能影响模型的泛化能力。

---

## 721. Multi-Agent Memory Inconsistency: Where Factual Conflicts Disappear in LLM Agents

**arXiv ID:** 2611.22245 | [PDF](https://arxiv.org/pdf/2611.22245v1)

**作者:** Hejia Zhang `[一作]` (Intel), Jishen Zhao `[通讯]` (UC San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了多代理委托在大型语言模型（LLM）系统中的应用，重点分析了在文档审查任务中，代理之间的记忆不一致性及其对事实冲突检测的影响。

**💡 创新点**

创新点在于提出了一个基准和证据流分析工具，用于研究多代理记忆不一致性，特别是在控制的事实冲突情况下，首次进行阶段性实证研究。

**🔧 技术方法**

使用了多种大型语言模型（如GPT-5.6和GPT-6）进行实验，并通过Codex SDK和Claude Code SDK进行评估。

**📊 数据集**

使用了11个Harvey LAB文档审查任务的数据集，这些任务包含了73个控制的数值冲突。

**📈 对比分析**

通过比较单代理执行、可选委托和强制委托三种方法，发现强制委托在检索冲突声明的覆盖率上有所提高（从50.2%提高到68.7%），但冲突检测率却下降（从16.2%降至13.2%）。可选委托的检测率最高，达到24.2%。

**⚠️ 局限性**

限制在于研究仅基于11个特定任务，且只考虑了数值冲突，缺乏对更广泛文档领域和非数值冲突的测试。此外，分析中未能追踪何时两个声明共同可用，可能影响结果的全面性。

---

## 722. Where Should the Lamp Go? Ambient-Aware Lamp Placement for Portraits

**arXiv ID:** 2611.22005 | [PDF](https://arxiv.org/pdf/2611.22005v1)

**作者:** Yuyan Chen `[一作]` `[通讯]` (ModelsLive Inc.), Yuyan Chen (ModelsLive Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为LightMatch的框架，用于优化灯光在肖像摄影中的位置，以提高面部照明效果。

**💡 创新点**

首次将环境感知灯光放置问题形式化为一个逆向照明问题，结合了物理光感知与人类美学偏好。

**🔧 技术方法**

使用了DINOv3特征的交叉注意力机制来读取光线，并通过布拉德利-特里评分模型来优化灯光位置。

**📊 数据集**

使用了4000对渲染图像和1500对真实面孔图像作为数据集进行训练和测试。

**📈 对比分析**

与传统的固定位置方法和忽略环境光的方法相比，LightMatch在多个测试集上表现优越，Share指标提高了28.4%，并且在真实面孔上减少了37.7%的灯光方向误差。

**⚠️ 局限性**

在环境光估计误差较大或灯光强度低于环境光时，LightMatch的建议可能会偏离最佳位置。

---

## 723. EgoHandWAM: Probing the Role of Hand Motion in Egocentric World Action Modeling

**arXiv ID:** 2611.22151 | [PDF](https://arxiv.org/pdf/2611.22151v1)

**作者:** Kailin Wang `[一作]` (Country Garden Services Group), Zhaosong Li `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文提出了EgoHandWAM模型，该模型能够根据当前图像和最近的手部状态预测未来两秒的手腕和指尖轨迹，以及未来的视觉特征。

**💡 创新点**

创新点在于通过结合手部运动和视觉特征预测，探索手部运动如何影响第一人称相机的视觉预测，并在此过程中引入了未来特征的监督。

**🔧 技术方法**

使用了冻结的DINOv2编码器进行训练，并采用了回归模型来处理手部轨迹和视觉特征的预测。

**📊 数据集**

使用了751,065个EgoDex窗口的数据集进行训练和评估。

**📈 对比分析**

与静态预测模型相比，动作模型在两秒的手腕端点误差上从60.26毫米降低到56.48毫米，尽管在长距离预测上仍存在低估的问题。引入未来特征监督后，轨迹准确性几乎没有变化。

**⚠️ 局限性**

限制在于模型在长距离预测时的表现不佳，且在机器人任务成功率上，使用对齐的人类手部历史并未带来一致的提升。

---

## 724. A Systematic Survey of Decompilation: Mapping Classical and Machine-Learning Approaches

**arXiv ID:** 2611.21612 | [PDF](https://arxiv.org/pdf/2611.21612v1)

**作者:** Keitaro Shimane `[一作]` (Nihon University), Yutaka Matsuno `[通讯]` (Nihon University)

**通讯引用:** 551 | [OpenAlex ID](https://openalex.org/A5033690770)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文对405项1973年至2026年间的反编译研究进行了系统性调查，旨在将反编译研究映射到一个以管道为中心的分类法中，并探讨经典方法与机器学习（ML）方法在反编译过程中的位置及其技术设计和评估证据的差异。

**💡 创新点**

创新点在于通过统一的文献库和管道中心的映射，明确了经典和机器学习方法在反编译中的关系，并对ML方法的技术设计和评估证据进行了深入比较。

**🔧 技术方法**

使用了系统性文献调查的方法，结合数据库搜索和引用追踪，分析了394项主要研究，并对116项基于机器学习的研究进行了详细的技术设计和评估证据提取。

**📊 数据集**

数据集包括405项研究，涵盖了从1973年到2026年间的反编译研究，具体数据集未详细列出，但涉及多种编程语言和反编译工具。

**📈 对比分析**

与经典方法相比，机器学习方法在符号和名称恢复、输出精炼和端到端翻译等阶段表现突出。大多数ML研究针对本地二进制文件，并通过逐项匹配参考文献来支持其主张，但执行测试和人类或LLM判断的证据相对较少。

**⚠️ 局限性**

限制在于ML方法并未取代经典管道，且在某些阶段（如控制流结构化和IR提升）仍然主要依赖经典技术。此外，评估证据的多样性和任务特异性使得跨研究比较变得困难。

---

## 725. From AIOps Pipelines to Agentic Service Operations: A Survey of Capability, Authority, and Evidence

**arXiv ID:** 2611.21620 | [PDF](https://arxiv.org/pdf/2611.21620v1)

**作者:** Hailiang Zhao `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**通讯引用:** 12421 | [OpenAlex ID](https://openalex.org/A5055284175)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文调查了从AIOps管道到自主服务操作的转变，重点关注能力、操作权威和评估证据三个方面，分析了现代服务系统在事件生命周期中的表现。

**💡 创新点**

创新点在于提出了一种新的分类法，将能力、权威和证据分开，并建立了一个四平面参考架构，以支持自主服务操作的评估和实施。

**🔧 技术方法**

使用了统计学习、深度学习、图模型和因果方法等技术，结合传统AIOps方法和语言模型增强系统进行审计和比较。

**📊 数据集**

审计了22个语言模型增强的系统，涉及的证据包括日志、指标、跟踪、事件、拓扑和变更记录等。

**📈 对比分析**

通过与传统AIOps方法的比较，发现大多数系统仍处于副驾驶或只读调查者的角色，只有两个系统具备写入权威，但都在受限的操作范围内，且缺乏可控的生产结果证据。

**⚠️ 局限性**

限制在于当前审计的系统大多数仍处于较低的权威水平，缺乏足够的证据支持其自主操作的能力，且在实际应用中可能面临操作风险和不确定性。

---

## 726. A comparative exploration of Steinerberger curvature and resistance curvature in model and real-world complex networks

**arXiv ID:** 2611.23204 | [PDF](https://arxiv.org/pdf/2611.23204v1)

**作者:** Madhumita Mondal `[一作]` (Institute of Mathematical Sciences), Jürgen Jost `[通讯]` (Max Planck Institute for Mathematics in the Sciences)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究探讨了Steinerberger曲率(SBC)和电阻曲率(RC)，并分析了它们与已建立的图曲率度量（如Forman-Ricci、增强Forman-Ricci和Ollivier-Ricci曲率）之间的关系。

**💡 创新点**

创新点在于系统比较了SBC和RC在不同合成网络模型和真实世界网络中的表现，揭示了它们与传统图度量之间的不同关联性。

**🔧 技术方法**

使用了Steinerberger曲率和电阻曲率的数学框架，结合了图的距离矩阵和有效电阻矩阵，进行网络结构的分析。

**📊 数据集**

数据集包括四种标准随机图模型（Erdös-Rényi、Watts-Strogatz、Barabási-Albert和超球图生成器）以及18个真实世界网络（如社交网络、交通网络等）。

**📈 对比分析**

通过与已建立的Ricci曲率度量进行比较，发现RC与这些度量之间存在强正相关，而SBC与这些度量的相关性普遍较弱，表明SBC捕捉的结构信息与其他度量有显著不同。

**⚠️ 局限性**

限制在于SBC和RC的计算相对复杂，尤其是在处理大规模网络时，计算时间较长，且两者在不同网络结构中的表现差异较大。

---

## 727. When Passing Checks Do Not Establish Completion: Evidence Interpretation in Coding Agents

**arXiv ID:** 2611.22835 | [PDF](https://arxiv.org/pdf/2611.22835v1)

**作者:** Xueping Gao `[一作]` `[通讯]`, Xueping Gao

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对多个任务和接口进行了探索性比较，保留了负面结果和未通过的资格。

**💡 创新点**

创新点在于通过独立的补丁流和控制，提供了对不同任务的成功控制和公共范围校准的深入分析。

**🔧 技术方法**

使用了多种技术，包括源绑定、技术恢复和语法检查等。

**📊 数据集**

数据集包括来自官方SWE-bench实验库的多个任务和补丁提交。

**📈 对比分析**

通过与多个基准任务的比较，展示了不同补丁在通过检查时的表现，部分补丁在特定检查中表现良好，但整体性能未能显著提升。

**⚠️ 局限性**

限制在于未能提供新的模型调用，且部分任务的结果未能达到预期的修复效果。

---

## 728. COTTI: Contrastive Training and Time-Aware Inference for Robust Industrial Recommendation under Promotional Drift

**arXiv ID:** 2611.22597 | [PDF](https://arxiv.org/pdf/2611.22597v1)

**作者:** Maolei Huang `[一作]` (Alibaba Group), Shuhan Song `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了COTTI框架，用于在促销漂移下进行稳健的工业推荐，分离历史表示学习与事件时间分数调整。

**💡 创新点**

创新点在于结合了跨周期对比对齐、历史距离示例加权和事件条件分数调整，形成三部分框架。

**🔧 技术方法**

使用了对比学习、加权历史重标定和自适应边界对齐等技术。

**📊 数据集**

使用了来自Fliggy的数据集，涵盖常规流量和多个促销期的数据。

**📈 对比分析**

与TEN、HDR等方法进行比较，COTTI在离线AUC上达到0.8127，优于TEN的0.8101和HDR的0.8001，在线实验显示相对提升3.14%的转化率和2.68%的每千次展示收入。

**⚠️ 局限性**

局限性在于设计未对分布稳健性或子群体水平的保证，且在评估的促销条件下的表现可能不具普遍适用性。

---

## 729. Comparing Implicit and Explicit Language Attitudes toward the Dutch multiethnolect Straattaal in Humans and Large Language Models

**arXiv ID:** 2611.22335 | [PDF](https://arxiv.org/pdf/2611.22335v1)

**作者:** Hedwig Oldenhof `[一作]` (Utrecht University), Dong Nguyen `[通讯]` (Utrecht University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大型语言模型（LLMs）是否表现出类似于人类的隐性和显性语言态度，特别是针对荷兰多民族方言Straattaal与移民社区的关联。

**💡 创新点**

创新点在于首次将隐性和显性语言态度的区分应用于LLMs，并通过与人类的比较设计来评估LLMs的语言态度。

**🔧 技术方法**

使用了隐性联想测试（IAT）和显性评分任务来评估人类的态度，同时使用LLM词汇联想测试（LLM WAT）和相同的显性评分任务来评估LLMs的态度。

**📊 数据集**

研究使用了与Straattaal相关的词汇和标准荷兰语的词汇，涉及的参与者为90名荷兰本土说话者。

**📈 对比分析**

通过比较人类和LLMs的反应模式，发现LLMs在隐性和显性任务中均表现出与人类相似的态度，尤其是在较大和更新的模型中，关联更强。

**⚠️ 局限性**

限制在于人类和LLMs的实验设置和反应机制不同，导致结果的直接比较受到限制。此外，LLMs的模型和提示条件有限，限制了结果的普遍性。

---

## 730. V2V Communications Using Blockchain-Enabled 6G Technology and Federated Learning

**arXiv ID:** 2611.21932 | [PDF](https://arxiv.org/pdf/2611.21932v1)

**作者:** Tahir H. Ahmed `[一作]` (Multimedia University), Shahid Mumtaz `[通讯]` (Silesian University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

本研究提出了一种集成区块链技术、联邦学习和延迟及资源分配优化的车对车（V2V）通信方法，评估了该系统在数据包传递率、模型准确性和延迟等性能指标上的表现，并展示了其优于现有技术的优势。

**💡 创新点**

创新点在于将区块链和联邦学习技术结合，提供了一种去中心化、不可篡改的V2V通信平台，增强了安全性和数据隐私，同时优化了资源利用和延迟。

**🔧 技术方法**

使用了混合整数非线性规划（MINLP）、多目标粒子群优化（MOPSO）和区块链辅助共识优化（BACO）等技术。

**📊 数据集**

使用了模拟数据集，包括100个车辆节点，通信范围500米，数据包大小1500字节等参数进行性能评估。

**📈 对比分析**

与三种现有技术进行了比较，结果显示该系统在延迟和资源利用率方面表现优异，延迟低于0.1毫秒，数据包传递率和模型准确性均超过94%。

**⚠️ 局限性**

限制在于该研究主要基于模拟环境，实际应用中可能面临更复杂的网络条件和挑战，未来需要进一步验证和优化。

---

## 731. Strong Fused Performance Can Mask Branch Collapse in a Dual-View Heart-Sound Classifier: A Three-Part Diagnostic Protocol

**arXiv ID:** 2611.22556 | [PDF](https://arxiv.org/pdf/2611.22556v1)

**作者:** Safayet Khan `[一作]` (East West University), Md. Tauhid Bin Iqbal `[通讯]` (East West University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本研究探讨了在双视图心音分类中，融合模型的表现可能掩盖某些分支未被使用的情况。通过对AudioFuse模型的分析，提出了三部分诊断方法来检测分支崩溃现象。

**💡 创新点**

创新点在于提出了一种新的诊断方法来评估分支的有效性，包括删除测试、线性探测测试和独立训练测试，强调了仅依赖融合得分无法证明每个分支都学习到了有意义的表示。

**🔧 技术方法**

使用了AudioFuse模型，该模型融合了光谱图变换器和波形卷积神经网络，并进行了多种训练策略的比较。

**📊 数据集**

主要使用PhysioNet/CinC 2016数据集进行实验，同时在ICBHI 2017和CirCor DigiScope数据集上进行了验证。

**📈 对比分析**

通过与单分支训练的比较，发现光谱图分支在联合训练中未能有效学习，且删除该分支对融合ROC-AUC的影响微乎其微，表明该分支未被有效利用。

**⚠️ 局限性**

限制在于仅测试了一种架构家族，未验证该效应是否出现在其他双视图模型或自监督预训练编码器中。优化证据与早期更新不对称和初始化效应一致，但未能证明因果关系。

---

## 732. BehaVerify: A Formal Verification Tool for Behavior Trees

**arXiv ID:** 2611.22403 | [PDF](https://arxiv.org/pdf/2611.22403v1)

**作者:** Serena S. Serbinowska `[一作]` (Vanderbilt University), Taylor T. Johnson `[通讯]` (Vanderbilt University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出了一种用于行为树（BTs）和神经符号行为树（Neuro-Symbolic BTs）的验证工具箱，旨在帮助用户使用领域特定语言（DSL）指定BT，并快速验证其规范。

**💡 创新点**

创新点在于将多个功能整合到一个工具箱中，提供了一个统一的DSL和命令行接口，支持BT的验证和实现，并且能够生成Python和Haskell代码。

**🔧 技术方法**

使用了快速编码（fastforwarding）技术来高效验证BT的规范，并支持生成Python和Haskell实现。

**📊 数据集**

使用了多个示例数据集，包括无人机控制、火星探测器和列车控制等场景，进行工具的验证和比较。

**📈 对比分析**

与其他工具（如BT2Fiacre和BT2BIP）进行了比较，结果显示该工具在可扩展性和验证时间上具有显著优势，尤其是在处理大规模BT时，快速编码显著提高了性能。

**⚠️ 局限性**

限制在于与其他工具的语义不匹配，无法实现真正的并行节点执行，并且当前示例的规模较小，可能影响性能比较的有效性。

---

## 733. CoSphere-SLAM: Multi-Agent Panoramic Gaussian Splatting SLAM with Spherical Loop Closure

**arXiv ID:** 2611.22235 | [PDF](https://arxiv.org/pdf/2611.22235v1)

**作者:** Qi Shao `[一作]` (Harbin Engineering University), Liang Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了CoSphere-SLAM，这是第一个用于全景相机的多代理高斯喷溅SLAM框架，旨在解决室内环境中的密集协作映射问题。

**💡 创新点**

创新点在于引入了球形环闭合，通过SO(3)对任意方向的特征进行关联，并通过全局Sim(3)共识拒绝对称的错误闭合，同时采用方向性全局优化来纠正尺度漂移。

**🔧 技术方法**

使用了快速傅里叶相关技术进行球形环闭合，方向性全局优化解决每个关键帧的Sim(3)姿态图，并通过全向高斯融合来优化姿态和地图。

**📊 数据集**

在Holo360D和360Loc数据集上进行了实验，这些数据集包含多个室内视频和手持录制的全景图像。

**📈 对比分析**

与现有的单代理和多代理方法相比，CoSphere-SLAM在轨迹精度和渲染质量上达到了最先进的水平，特别是在多个场景中表现出色，测试结果显示其在多个场景中的平均绝对轨迹误差（ATE）显著低于其他方法。

**⚠️ 局限性**

限制在于该系统仍假设场景是静态的，并且共享视图，未来的工作将考虑动态物体的建模，并整合惯性和测距传感器以实现度量尺度和现场部署。

---

## 734. MCTS Is Counting the Wrong Thing: Learning Exploration Counts for LLM Reasoning

**arXiv ID:** 2611.22301 | [PDF](https://arxiv.org/pdf/2611.22301v1)

**作者:** Xuan Zhang `[一作]` (Fudan University), Chao Qu `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的蒙特卡洛树搜索方法LPC-MCTS，旨在提高大语言模型（LLM）推理中的探索效率。

**💡 创新点**

创新点在于通过引入学习的伪计数来替代传统的独立访问计数，从而减少计数碎片化，提高探索效率。

**🔧 技术方法**

使用了轻量级的硬币翻转网络（Coin-Flipping Network）来估计表示覆盖率，并将其应用于MCTS的选择过程中。

**📊 数据集**

在数学推理和搜索基准上进行了评估，使用了Qwen2.5-7B和Qwen3-4B作为政策骨干，并在多个数学和搜索任务上进行了测试。

**📈 对比分析**

与标准MCTS相比，LPC-MCTS在5个数学基准上平均提高了3.79个百分点，在7个搜索基准上也表现出一致的性能提升，显示出其在探索效率和推理性能上的显著改善。

**⚠️ 局限性**

限制在于LPC-MCTS的性能提升主要依赖于学习的伪计数的准确性和有效性，可能在某些特定任务或数据集上表现不如预期。

---

## 735. Conserved Verdicts, Drifting Commitments: Identifying Harmful Overthinking

**arXiv ID:** 2611.21946 | [PDF](https://arxiv.org/pdf/2611.21946v1)

**作者:** Jianghong Zhou `[一作]`, Chen Lin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文探讨了推理模型在思考过程中可能会失去正确答案的现象，并提出了一种比较当前答案和继续推理后答案正确性的概率的方法。

**💡 创新点**

创新点在于明确了承诺-裁决差距的概念，并通过该差距来衡量停止推理的价值，提出了新的测量方法来评估推理过程中的决策。

**🔧 技术方法**

使用了概率论的标准技术，包括塔性质、Doob分解和望远镜性质。

**📊 数据集**

使用了MATH-500和OlympiadBench数据集进行实验，涉及多个推理模型和不同的预算设置。

**📈 对比分析**

通过比较不同推理链的准确性，发现继续推理在大多数情况下会导致准确性下降，尤其是在某些特定问题上。实验结果显示，早期停止的规则在准确性上几乎没有提升。

**⚠️ 局限性**

限制在于实验主要集中在数学推理和最多8B参数的模型上，承诺的定义依赖于特定的停止和回答程序，且训练目标仅在合成数据上进行了测试。

---

## 736. FACE: Flow-Guided Angular Channel Estimation Framework for mmWave MIMO Systems

**arXiv ID:** 2611.21698 | [PDF](https://arxiv.org/pdf/2611.21698v1)

**作者:** Samitha Gunarathne `[一作]` (University of Oulu), Matti Latva-aho `[通讯]` (University of Oulu)

**通讯引用:** 16404 | [OpenAlex ID](https://openalex.org/A5035872239)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于流引导的角度信道估计框架FACE，用于毫米波MIMO系统，以解决高维信道、有限导频观测和测量噪声带来的挑战。

**💡 创新点**

创新点在于利用流匹配模型作为生成信道先验，结合PnP-PGD框架，显著提高了信道估计的准确性和计算效率。

**🔧 技术方法**

使用了流匹配技术作为生成模型，结合了测量一致性和学习的信道分布进行信道估计。

**📊 数据集**

使用了生成的毫米波信道数据集和真实的60 GHz信道测量数据集进行评估。

**📈 对比分析**

与传统的LMMSE、OMP、SOMP和扩散基估计器相比，FACE在多种信道和系统配置下表现出更高的信道估计准确性，并且在较少的导频观测下保持高估计准确性，平均可实现速率接近完美CSI基准。

**⚠️ 局限性**

限制在于该方法尚未扩展到毫米波MIMO系统中的干扰管理，未来工作将探讨这一方向。

---

## 737. Can AI Help Explain AI? Effects of Multimodal LLM-Generated Explanations on Perceived Understanding and Perceived Trust in XAI Visualizations

**arXiv ID:** 2611.21638 | [PDF](https://arxiv.org/pdf/2611.21638v1)

**作者:** Mario Truss `[一作]` (Adobe), Stefan Feuerriegel `[通讯]` (Ludwig Maximilian University of Munich and Munich Center for Machine Learning)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了通过多模态生成的文本解释是否能提高用户对可视化的理解和信任。进行了一个预注册的被试内实验，参与者在七个决策场景中评估了仅有可视化和带有AI生成解释的可视化的效果。

**💡 创新点**

创新点在于首次系统性地测试了多模态生成的文本解释对用户感知理解和信任的影响，尤其是针对非技术用户的可解释性问题。

**🔧 技术方法**

使用了先进的多模态生成模型（如GPT-5.6 Sol）来生成自然语言解释，这些解释结合了可视化和任务上下文。

**📊 数据集**

使用了七个不同的决策场景，包括工作预测、信用风险、心脏病风险等，涵盖了多种数据类型（文本、表格、图像）。

**📈 对比分析**

通过比较用户在有无解释的情况下的反应，结果显示，带有解释的可视化在感知理解和信任上分别提高了0.35和0.26分（均p < .001），且这种提升在所有场景中均有体现。

**⚠️ 局限性**

限制在于样本主要集中在高教育水平和频繁使用AI工具的用户，未来研究需要探索这些效果是否适用于教育水平较低或使用频率较低的用户群体。

---

## 738. ReLAP: Latent Action Pretraining through Recurrent Video Prediction in JEPA Space

**arXiv ID:** 2611.22077 | [PDF](https://arxiv.org/pdf/2611.22077v1)

**作者:** R. Gnana Praveen `[一作]` (Bagel Labs), Bidhan Roy `[通讯]` (Bagel Labs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为ReLAP的潜在动作预训练框架，通过在冻结的JEPA表示空间中结合局部过渡学习和递归多步预测，学习机器人行为的潜在动作。

**💡 创新点**

ReLAP的创新点在于通过递归的潜在动作学习，鼓励时间一致性，而不是仅依赖于成对重建，允许后续预测错误通过预测轨迹反向传播到早期的潜在动作。

**🔧 技术方法**

使用了冻结的V-JEPA 2.1编码器和共享的逆动态模型，结合局部过渡重建和递归预测技术。

**📊 数据集**

使用了七个大型视频数据集，包括Bridge-v2、DROID、AgiBot、EgoDex、Airoa-MOMA、InternData和Ego4D，进行训练和评估。

**📈 对比分析**

与其他方法相比，ReLAP在LIBERO基准上取得了97.9%的成功率，在LIBERO-Plus的零-shot评估中获得了70.0%的成功率，表现优于大多数对比方法，尤其在空间任务上表现最佳。

**⚠️ 局限性**

ReLAP的局限性在于递归潜在轨迹的时间不对称性，早期潜在动作受到更强的监督，而后期潜在动作的错误更难以纠正。此外，当前的预训练管道未能显式强制多视角不变性，可能影响在不同视角下的表现。

---

## 739. Keeping Parts Honest: Isolated Part Graphs for Faithful Vision Transformer Explanations

**arXiv ID:** 2611.22853 | [PDF](https://arxiv.org/pdf/2611.22853v1)

**作者:** Bernard Tomczyk `[一作]` (Lancaster University), Bryan Williams `[通讯]` (Lancaster University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的分类器，该分类器将每个图像表示为一个孤立部分的图形，避免了自注意力机制导致的信息泄漏。

**💡 创新点**

创新点在于引入了区域忠实性（region faithfulness）作为可解释性模型的一个可测试属性，并提出了一种通过构建孤立部分来实现这一属性的分类器。

**🔧 技术方法**

使用了视觉变换器（ViT）技术，并通过冻结早期块和在孤立条件下微调最后块来编码每个部分。

**📊 数据集**

在11个细粒度和一般对象基准上进行了评估，包括CUB-200-2011、FGVC-Aircraft、Stanford Cars等，没有使用部分或框注释进行训练或测试。

**📈 对比分析**

与五种基于部分和原型的ViT分类器进行了比较，结果显示该方法在大多数基准上超越了现有的可解释方法，且提供的解释更符合模型实际使用的内容。

**⚠️ 局限性**

限制在于该方法依赖于孤立部分的发现，可能在某些情况下对部分的选择和质量敏感。

---

## 740. From Human Videos to Bimanual Dexterous Manipulation Data: Confidence-Guided RL Refinement of Video Priors

**arXiv ID:** 2611.21908 | [PDF](https://arxiv.org/pdf/2611.21908v1)

**作者:** Yanghong Liu `[一作]` (University of California), Wei Zhan `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了PriDex框架，将从人类视频中提取的近似先验转化为可执行的双手灵巧机器人演示，通过强化学习在模拟环境中进行优化。

**💡 创新点**

创新点在于分离可转移的交互结构与特定于体现的运动，采用信心引导的策略来平衡物体运动和手腕运动的指导信号。

**🔧 技术方法**

使用了强化学习（RL）技术，特别是PPO（Proximal Policy Optimization）算法进行运动优化。

**📊 数据集**

使用了EgoDex数据集中的四个双手灵巧操作任务，包括倒水、扫入垃圾桶、拧瓶盖和清洁盘子，基于仅12个视频生成数千个机器人轨迹。

**📈 对比分析**

与H2S2R和ViViDex等方法进行比较，PriDex在所有任务上表现优越，平均成功率达到57.2%。通过消融实验显示，信心适应和手腕运动指导对任务成功率有显著影响。

**⚠️ 局限性**

限制在于视频-derived运动估计的不确定性，尤其是在遮挡或视觉模糊情况下，可能导致物体运动估计不可靠。

---

## 741. WIP: Exploring Student-Created AI Agents and Instructional Videos to Foster Engagement in Computing Education

**arXiv ID:** 2611.23058 | [PDF](https://arxiv.org/pdf/2611.23058v1)

**作者:** Lucas G. Zangari `[一作]` (Georgia Institute of Technology), Pedro Guillermo Feijóo-García `[通讯]` (Georgia Institute of Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究探讨了学生创建的人工智能代理和教学视频在促进计算机教育中的参与度的作用，特别是在大型本科软件设计与工程课程中。

**💡 创新点**

创新点在于通过让学生设计自己的AI代理和教学视频，促进主动学习和更深层次的课程参与，鼓励创造力和批判性思维。

**🔧 技术方法**

使用了学生创建的多媒体艺术品，包括教学视频和对话式AI代理，结合了主动学习策略。

**📊 数据集**

数据集来自于2024年夏季的两门计算机科学课程，参与者为19名学生，涵盖本科和研究生层次。

**📈 对比分析**

与传统教学方法相比，学生在创建视频和AI代理时报告了较高的学习感知（视频平均5.43，AI代理平均5.59），并且57.9%的学生认为创建AI代理有助于学习课程内容。

**⚠️ 局限性**

限制在于样本量较小，缺乏对照组，无法全面评估学生创建的艺术品相较于传统教学方法的有效性，未来研究需在更大规模的环境中进行验证。

---

## 742. Spatially local mini-batching for implicit neural representations

**arXiv ID:** 2611.22236 | [PDF](https://arxiv.org/pdf/2611.22236v1)

**作者:** Minseok Choi `[一作]` (Pohang University of Science and Technology), Donsub Rim `[通讯]` (Washington University in St Louis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出了一种基于低秩神经表示（LRNR）的隐式神经表示（INR），通过空间局部小批量训练来提高模型在局部特征（如冲击波和接触不连续性）上的准确性。

**💡 创新点**

创新点在于通过低秩结构使得LRNR能够在局部小批量上进行有效训练，从而避免了传统INR在局部训练时的准确性下降问题。

**🔧 技术方法**

使用了低秩神经表示（LRNR）和超网络（hypernetwork）技术，结合了共享基和小超网络发出的系数来构建模型。

**📊 数据集**

使用了六个超波动守恒定律问题的数据集，这些数据集通过自适应网格细化的数值解计算得出。

**📈 对比分析**

与超网络-SIREN、DeepONet和傅里叶神经算子（FNO）等基线模型进行比较，LRNR在相同的局部批量和训练预算下，达到了4.9%的相对L^2误差，而基线模型的误差在16%到78%之间。

**⚠️ 局限性**

局部训练的限制在于可能无法捕捉全局特征，且在某些情况下，局部小批量的训练可能导致模型在全局范围内的表现不如全局批量训练。

---

## 743. A Note on Truly Subquadratic Time FPTAS for Knapsack

**arXiv ID:** 2611.22809 | [PDF](https://arxiv.org/pdf/2611.22809v1)

**作者:** Lin Chen `[一作]` (Zhejiang University), Guochuan Zhang `[通讯]` (Zhejiang University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种针对整数（min,+）卷积的（m^ρ）时间算法，1<ρ<2，从而为0-1背包问题提供了一个随机完全多项式时间近似方案，运行时间为(n+(1/)^1+ρ/2)。该算法返回的可行解的利润至少为最优解的1+倍，且具有较高的概率。

**💡 创新点**

创新点在于通过替换Chen等人的两次二次计算为请求前缀和后缀函数的阻塞计算，以及在共享Chan的利润网格的组内进行精确计算，从而保留了卷积指数的一半节省。

**🔧 技术方法**

使用了随机化算法和全容量接近分解技术，结合了Wang的加权缓存3SUM界限和Zhou的见证整数卷积。

**📊 数据集**

论文中没有明确提到使用的数据集，但讨论了整数权重和利润的情况，假设输入的整数大小是多项式的。

**📈 对比分析**

与现有方法的比较显示，该算法在(n+(1/)^1.981)时间内实现了随机化的完全多项式时间近似方案，性能优于之前的(n+(1/)^2)时间的FPTAS。

**⚠️ 局限性**

限制在于该算法依赖于（min,+）卷积假设，且对于固定的δ>0，无法实现O((n+1/)^2-δ)时间的FPTAS。

---

## 744. RefCloth: Transferable Adversarial Tablecloths for Vision-Language-Action Models

**arXiv ID:** 2611.22736 | [PDF](https://arxiv.org/pdf/2611.22736v1)

**作者:** Shaozhen Liu `[一作]` (University of Chinese Academy of Sciences), Zhenan Sun `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4`

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

## 745. Dynamic Task Allocation with Dialogue

**arXiv ID:** 2611.23119 | [PDF](https://arxiv.org/pdf/2611.23119v1)

**作者:** Matthew Groholski `[一作]` (Arizona State University), Nakul Gopalan `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种支持动态任务分配的通信增强贝叶斯任务分配框架，旨在应对部分可观察性下的多智能体协作。

**💡 创新点**

创新点在于引入了自然语言对话作为任务分配推理的概率证据，并在动态任务到达的情况下进行评估。

**🔧 技术方法**

使用了贝叶斯推理和自然语言处理技术，结合了多智能体马尔可夫决策过程（MMDP）和去中心化部分可观察马尔可夫决策过程（Dec-POMDP）。

**📊 数据集**

在扩展的Overcooked环境中进行评估，该环境支持动态到达的任务，并考虑了完全和部分对象空间可观察性。

**📈 对比分析**

与基线方法进行比较，通信代理在部分可观察性环境中表现显著优于传统的LLM代理，平均加速比为0.585，完成时间平均快18个时间步；在完全可观察性下，性能与非通信基线相当。

**⚠️ 局限性**

局限性包括通信机制较为狭窄，仅传达任务分配意图而非更丰富的信念、问题或协商；实验规模有限，仅在一个环境系列中进行，且假设已知任务类型、固定代理集和可靠的无成本通信。

---

## 746. Provable Mixing Properties of Parallel Mixnets Revisited

**arXiv ID:** 2611.22732 | [PDF](https://arxiv.org/pdf/2611.22732v1)

**作者:** Pierfrancesco Ingo `[一作]` (Max Planck Institute for Software Systems), Deepak Garg `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文对并行混合网络（mixnet）进行了新的分析，证明了这些网络在没有额外噪声消息的情况下也能提供强的消息不可区分性，从而确保发送者的匿名性。

**💡 创新点**

创新点在于提供了并行混合网络的强匿名性下限的严格证明，并且通过新的随机分析方法，探讨了匿名性、延迟和每个路由器带宽之间的定量权衡。

**🔧 技术方法**

使用了随机分析技术，特别是通过集中界限（如切比雪夫不等式）来界定消息的后验赔率的方差，并证明了消息的（ε, δ）-不可区分性。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了并行混合网络的理论模型和参数设置。

**📈 对比分析**

与现有的工作相比，本文提供了并行混合网络的强匿名性下限的证明，且在实际操作条件下，证明的匿名性与先前的经验估计相符，显示出良好的性能。

**⚠️ 局限性**

限制在于，虽然提供了强的理论保证，但在服务器被攻击的情况下，匿名性会显著下降，且在高服务器妥协率下，性能的改善有限。

---

## 747. A short proof of the 1-1/e performance of RANKING on the complete upper triangular instance

**arXiv ID:** 2611.22923 | [PDF](https://arxiv.org/pdf/2611.22923v1)

**作者:** Giovanni Montanari `[一作]` (FairPlay joint team), Vianney Perchet `[通讯]` (FairPlay joint team)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提供了一个简短的证明，证明了Karp、Vazirani和Vazirani（1990）提出的随机在线二分匹配算法在完全上三角实例中期望匹配的顶点数为(1-1/e)n+o(n)。

**💡 创新点**

创新点在于从离线的角度分析最终匹配，而不是跟随在线到达的顺序，利用马尔可夫链和微分方程的方法来推导匹配的期望大小。

**🔧 技术方法**

使用了马尔可夫链和微分方程技术来分析匹配过程。

**📊 数据集**

使用了完全上三角实例作为数据集，该实例包含所有对角边，并且在线顶点的到达顺序是已知的。

**📈 对比分析**

与其他在线匹配算法相比，本文的算法在最坏情况下提供了比任何确定性在线算法更好的保证，且在完全上三角实例中性能达到1-1/e的竞争比。

**⚠️ 局限性**

限制在于该分析主要针对完全上三角实例，可能无法直接推广到其他类型的二分图实例。

---

## 748. Risk Density RRT$^*$: Planning in the Space of Probability

**arXiv ID:** 2611.22654 | [PDF](https://arxiv.org/pdf/2611.22654v1)

**作者:** Lorenzo Paiola `[一作]` (Istituto Italiano di Tecnologia), Antonio Bicchi `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的路径规划算法Risk Density RRT* (RD-RRT*)，该算法在不确定环境中直接优化碰撞概率作为规划目标。

**💡 创新点**

创新点在于将风险密度功能引入路径规划中，允许算法在考虑碰撞概率和严重性时进行高效的路径搜索。

**🔧 技术方法**

使用了风险密度功能作为成本函数，并结合了RRT*算法的结构，进行高效的路径规划。

**📊 数据集**

在多种场景下进行了验证，包括使用MuJoCo模拟器中的TidyBot模型进行实际路径规划。

**📈 对比分析**

与几种基线方法（如几何方法、机会约束RRT*、BIT*和AIT*）进行了比较，RD-RRT*在估计的碰撞概率上表现出显著优势，能够找到更安全的路径。

**⚠️ 局限性**

限制在于未能证明修改后的目标、引导规则和半径调度的组合是否保留了原RRT*算法的渐近最优性保证，并且当前实现仅限于欧几里得空间。

---

## 749. HealthCLBench:Evaluating Context Learning in Health Conversations through Controlled Case Clusters

**arXiv ID:** 2611.22362 | [PDF](https://arxiv.org/pdf/2611.22362v1)

**作者:** Ting Yang `[一作]` (Ant Group), Ant Group `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究评估了健康背景学习，旨在获取、整合和应用相关背景信息，以回答健康问题并适应事实、证据或任务目标的变化。

**💡 创新点**

创新点在于提出了一个包含328个受控集群、1103个案例和4931个评分标准的中文医疗基准，系统地结合了明确的上下文困难、受控案例变异和特定成员的医疗标准。

**🔧 技术方法**

使用了健康对话系统生成的模拟对话，并通过70人的医疗团队进行注释和审核。

**📊 数据集**

数据集包含328个集群和1103个案例，涵盖医疗咨询、饮食、运动和日常健康管理等多个领域。

**📈 对比分析**

与七个通用模型的比较显示，GPT-6-Astra在全上下文指标上表现最佳，得分为75.03，其他模型得分在36.44到68.21之间，表明在不同条件下的性能变化。

**⚠️ 局限性**

限制在于模拟对话可能反映场景和表达偏好，任务组成在不同组之间变化，且某些组的样本量较小。

---

## 750. SilverSpoon-KD: A General-Purpose Toolkit for Knowledge Distillation

**arXiv ID:** 2611.22842 | [PDF](https://arxiv.org/pdf/2611.22842v1)

**作者:** Xaver R. Davey `[一作]` (KTH Royal Institute of Technology), David Broman `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个名为SilverSpoon-KD的开源PyTorch工具包，用于知识蒸馏，统一了不同的蒸馏范式和架构关系，支持多种工作负载和应用。

**💡 创新点**

创新点在于首次实现了通用的块级知识蒸馏，并提供了独特的可扩展性特征，能够在多个任务上验证其多功能性。

**🔧 技术方法**

使用了PyTorch框架，并与Hugging Face Trainer和标准并行计算栈（DDP、FSDP、DeepSpeed）兼容。

**📊 数据集**

在视觉、掩蔽语言模型和因果语言模型等任务上进行了验证，涵盖了同质、异质和量化蒸馏等多种情况。

**📈 对比分析**

通过与现有工具包的性能比较，包括吞吐量和GPU内存使用情况，SilverSpoon-KD的性能与现有工具包相当或更好。

**⚠️ 局限性**

限制在于假设对教师模型的白盒访问，当前仅支持单个学生与单个教师的训练，未来计划支持更多功能，如多教师蒸馏和黑盒管道。

---

