# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-06 | 今日论文总数: 510

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. A Robust Unsupervised Domain Adaptation Framework for Medical Image Classification Using RKHS-MMD

**arXiv ID:** 2605.03787 | [PDF](https://arxiv.org/pdf/2605.03787v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 2. SkCC: Portable and Secure Skill Compilation for Cross-Framework LLM Agents

**arXiv ID:** 2605.03353 | [PDF](https://arxiv.org/pdf/2605.03353v1)

**作者:** Yipeng Ouyang `[一作]` (Sun Yat-sen University), Xianwei Zhang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 4884 | [OpenAlex ID](https://openalex.org/A5051677902)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

建立了一套四阶段编译器框架，针对LLM Agent技能的跨框架可移植性与安全性进行编译、分析与格式化。

**💡 创新点**

引入强类型中间表示IR和编译时安全分析（Anti‑Skill Injection），实现从单一Markdown到多平台特定格式的自动化转换，降低维护O(m×n)到O(m+n)。

**🔧 技术方法**

经典编译原理（前端语法分析、IR、分析器、后端Emitter）、形式化安全规则、XML/YAML/Markdown格式自适应策略、Token/执行时间优化。

**📊 数据集**

SkillsBench 89任务、225社区技能、3,984社区技能安全审计数据等。

**📈 对比分析**

对比原始格式（O）与编译后格式（C）在四大框架下的Pass@1、奖励、token、时间；编译后在Claude+Kimi等平台提升13‑20% Pass率、Token减少10‑46%，编译延迟<10 ms，Anti‑Skill触发率94.8%。

**⚠️ 局限性**

仍需手工维护多目标Emitter规则、模型特定格式依赖、对极端复杂技能的扩展性未充分验证、对非SKILL.md生态或不同语言的支持有限。

---

## 3. LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models

**arXiv ID:** 2605.03299 | [PDF](https://arxiv.org/pdf/2605.03299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 4. Where's the Team Spirit? An Exploratory Study on Team Development Through Co-located Tablet-Based VR

**arXiv ID:** 2605.03127 | [PDF](https://arxiv.org/pdf/2605.03127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 5. A Comprehensive Analysis of Tokenization and Self-Supervised Learning in End-to-End Automatic Speech Recognition applied on French Language

**arXiv ID:** 2605.03696 | [PDF](https://arxiv.org/pdf/2605.03696v1)

**作者:** Thibault Bañeras-Roux `[一作]` (Nantes University), Richard Dufour `[通讯]` (Nantes University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文系统性评估了子词分词策略与自监督学习（SSL）模型对法语端到端自动语音识别（ASR）性能的影响，提出多维度评估指标；

**💡 创新点**

创新点在于：①同时考察分词算法、SSL模型以及多种词汇、语义、音位误差指标；②发现词表大小与模型泛化、评估指标一致性存在明显关联；③提出Unigram子词+150词表最佳的结论；

**🔧 技术方法**

技术包括：SpeechBrain端到端ASR框架、BPE/WordPiece/Unigram/SentencePiece/字符+语素分词、wav2vec 2.0（法语1k/3k/7k、英语53k、XLSR多语）自监督模型、CTC解码、最佳路径推理、句子嵌入与语义距离、语素误差等；

**📊 数据集**

数据集：ESTER 1/2、EPAC、ETAPE、REPERE（≈356 h 语音）训练，REPERE 10 h 作为测试；

**📈 对比分析**

对比方法：在相同SSL模型下比较不同分词策略，对同一分词下比较不同SSL模型；性能方面，法语7k模型+Unigram 150在WER 15.07%、CER 6.36%、SemDist 9.33%等指标均优于其他组合；

**⚠️ 局限性**

局限性：评估指标在系统级排名不一致，无法单一指标决定最佳；数据集规模仍有限，未覆盖方言与噪声多样性；未探究更大词表或更复杂解码策略的影响。

---

## 6. On Solving Problems of Substantially Super-linear Complexity in $N^{o(1)}$ Rounds in the MPC Model

**arXiv ID:** 2605.03376 | [PDF](https://arxiv.org/pdf/2605.03376v1)

**作者:** Andrzej Lingas `[一作]` (Lund University), Andrzej Lingas `[通讯]` (Lund University)

**通讯引用:** 2655 | [OpenAlex ID](https://openalex.org/A5016509031)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在Massively Parallel Computation (MPC) 模型中，对于具有显著超线性序列时间复杂度的问题，若采用 N^o(1) 轮次的协议，则每台机器在每轮的平均本地计算复杂度指数必大于该问题的最优序列时间复杂度指数；并给出了关于轮次、机器数、局部存储大小与平均本地计算复杂度之间的精确下界关系；

**💡 创新点**

提出了一种新的全局-本地计算复杂度权衡框架，证明在有限局部存储下实现低轮次协议必然导致本地计算变得更为复杂；该框架可用于评估超线性复杂度问题在 MPC 中的可行性与困难度；

**🔧 技术方法**

采用形式化的计算量与工作量定义，利用对数变换推导不等式，结合已知的最优序列时间复杂度参数 opt(P)，得到 ave(B) 与 opt(P) 的下界关系；

**📊 数据集**

无具体数据集，论文完全基于理论分析与抽象模型，无实验或数据集使用；

**📈 对比分析**

通过与已知的高速 MPC/拥塞 clique 协议（如矩阵乘法、APSP 等）的轮次与本地存储参数对照，证明这些协议在给定局部存储下的平均本地计算指数已超过阈值，说明其本地计算相对较重；

**⚠️ 局限性**

局限性在于仅给出上界条件与理论下界，未提供实际实现或算法；并假设理想的随机访问机模型与通信模型，可能与真实系统偏差；

---

## 7. Tracing Like a Clinician: Anatomy-Guided Spatial Priors for Cephalometric Landmark Detection

**arXiv ID:** 2605.03358 | [PDF](https://arxiv.org/pdf/2605.03358v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 8. Asymptotic properties of random monomial ideals

**arXiv ID:** 2605.03464 | [PDF](https://arxiv.org/pdf/2605.03464v1)

**作者:** Fatemeh Mohammadi `[一作]` (KU Leuven), Eduardo Sáenz-de-Cabezón `[通讯]` (Universidad de La Rioja)

**通讯引用:** 258 | [OpenAlex ID](https://openalex.org/A5090745504)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

研究随机多项式理想的渐近性质，特别是通过统计视角分析 LCM‑格的密度和其相应的 Betti 密度，揭示了在随机采样下 LCM‑格表现出的尖锐相变行为；

**💡 创新点**

首次将 LCM‑格从传统的“单个理想”分析转向“随机族”视角，发现其密度随生成元数或度数的变化呈现明显的临界阈值，形成低密度、过渡区和高密度三阶段；

**🔧 技术方法**

利用实验数学方法构建随机多项式理想模型（包括 Erdős‑Rényi 以及更一般的 ℬ(n,D,p) 模型），通过大量计算（如 100‑1000 个随机实例）求取 LCM‑格大小、Betti 密度、长度密度，并进行线性回归与相关性分析；

**📊 数据集**

使用随机生成的结构化多项式集合，尤其是二维或三维的平面/超图对应的 squarefree 多项式理想（如 n≤10、d≤3 的样本），并在 GitHub 仓库提供的计算脚本中生成并统计这些实例；

**📈 对比分析**

将 LCM‑密度与图密度、Betti 密度等指标进行 Pearson 相关性与线性回归比较，显示在不同阶段存在高度负相关或正相关；实验结果展示了在临界概率附近 LCM‑密度的急剧下降，验证了尖锐相变的存在；

**⚠️ 局限性**

受限于计算资源，实验仅覆盖较小的 n、d，且仅提供经验性证据；缺乏严格的理论证明，结论仍是猜想；进一步的工作需要在更大规模样本上验证并发展相应的概率论与代数证明。

---

## 9. An ERP Study of Recursive Possessive Parsing in ASD Children and Its Cognitive Neuro Mechanisms

**arXiv ID:** 2605.03447 | [PDF](https://arxiv.org/pdf/2605.03447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 10. SOAR: Real-Time Joint Optimization of Order Allocation and Robot Scheduling in Robotic Mobile Fulfillment Systems

**arXiv ID:** 2605.03842 | [PDF](https://arxiv.org/pdf/2605.03842v1)

**作者:** Yibang Tang `[一作]` (Beihang University), Zhen Zhao `[通讯]` (Geekplus Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于深度强化学习的统一框架 SOAR，实现 RMFS（机器人移动补给系统）中的订单分配与机器人调度的实时联合优化。

**💡 创新点**

创新点主要包括：
• 软订单分配机制，将订单与货架、工位的匹配度作为观测，打破传统离散分配与调度的耦合；
• 事件驱动的马尔可夫决策过程（ED‑MDP），使智能体能够在订单到达、机器人空闲、拣选完成等异步事件上即时做决策；
• 采用 Heterogeneous Graph Transformer 对仓库的异构实体（机器人、工位、存储位）进行全局消息传递，充分利用空间与关系信息；
• p‑norm 奖励塑造和阶段性偏置设计，解决长时序稀疏奖励、零梯度问题并引导策略学习。

**🔧 技术方法**

技术手段：
• 深度强化学习（PPO）配合时间感知（Δτ‑γ）与 GAE；
• Heterogeneous Graph Transformer（HGT）用于状态编码；
• 软匹配度矩阵与 Top‑K 过滤实现订单–货架–工位的软分配；
• p‑norm 奖励塑造（Φ(s)）与阶段偏置（Pick‑up、Delivery、Return）；
• 数字孪生平台与实体裁剪实现从仿真到真实环境的平滑部署。

**📊 数据集**

数据集：
• 合成数据：100×80 网格，3 个规模（Small、Medium、Large）对应 15/20/25 机器人、200/500/1000 订单；
• 真实 Geekplus 仓库：40×72 网格，861 货架、198 机器人、日均 13,000+ 订单，亦划分为 Small/Medium/Large。

**📈 对比分析**

与基线对比：
• 分离式基线（SQF/WLB/OR‑Tools + Nearest/Earliest/TSP/PSMDRL）
• 联合式基线（JOTP、SABS）
实验结果显示，SOAR 在合成与真实数据上均显著降低 makespan（7.5%）和平均订单完成时间（15.4%），且决策延迟低于 100 ms；在 7 天真实部署中，提升工位吞吐率 3.85%、平均订单完成时间下降 55 s、货架命中率 +6%、机器人行驶距离 -2.8%。

**⚠️ 局限性**

局限性：
• 仅处理高层调度，未集成路径规划与碰撞避免等底层控制；
• 需要大量训练样本和仿真环境，迁移到不同规模或布局的仓库时可能需重新训练；
• 对软分配参数（K）和奖励塑形参数（p）的敏感性需进一步理论与实验验证；
• 解释性与可视化仍有限，决策过程难以完全说明。

---

## 11. Sentiment Analysis of Indonesian Spotify Reviews Using Machine Learning and BiLSTM

**arXiv ID:** 2605.03443 | [PDF](https://arxiv.org/pdf/2605.03443v1)

**作者:** Uliano Wilyam Purba `[一作]` (Sumatera Institute of Technology), Martin C. T. Manullang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文比较了三种Scikit‑learn分类器（SVM、MNB、DT）与两层BiLSTM在印尼Spotify评论的三分类情感分析中的表现，使用统一的预处理流程；

**💡 创新点**

创新点在于统一的印尼预处理管道、在完整70k样本上对传统模型进行5折交叉验证、对BiLSTM使用CPU受限的20k子集进行实验，并将两套模型公开部署；

**🔧 技术方法**

技术包括TF‑IDF特征、SMOTE样本重采样、支持向量机、朴素贝叶斯、决策树，以及128维嵌入+两层双向LSTM的深度学习模型；

**📊 数据集**

数据集为从Google Play抓取的10万条印尼Spotify评论，去重后70,155条可用文本；BiLSTM仅使用20,000条划分为训练、验证、测试的子集；

**📈 对比分析**

采用5折交叉验证与独立测试集对比：DT在完整数据上取得weighted F1 0.7269、准确率0.7286；BiLSTM在子集测试上取得weighted F1 0.8069、准确率0.8314，但macro F1仅0.5478，说明对中性类识别不足；

**⚠️ 局限性**

局限性包括：BiLSTM训练样本远少于传统模型（4.9倍差异）；深度模型未采用类重采样或权重；仅在CPU环境下训练；未使用预训练印尼语词向量或Transformer模型。

---

## 12. Learning Correct Behavior from Examples: Validating Sequential Execution in Autonomous Agents

**arXiv ID:** 2605.03159 | [PDF](https://arxiv.org/pdf/2605.03159v1)

**作者:** Reshabh K Sharma `[一作]` (University of Washington), Yu Hu `[通讯]` (Microsoft)

**通讯引用:** 9521 | [OpenAlex ID](https://openalex.org/A5014478407)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种仅需 2–10 条通过执行轨迹即可自动学习正确行为模型，并用该模型对新的执行进行验证的方法。

**💡 创新点**

创新点在于将前缀树接受器与编译器中的支配子树分析相结合，再配合多层视觉与 LLM 语义等价检测，实现对非确定性路径的自动识别与容忍。

**🔧 技术方法**

技术方案包括 Prefix Tree Acceptors、支配树抽取、视觉相似度（pHash、SSIM、像素变化率）、多模态 LLM 语义分析、拓扑子序列匹配以及覆盖率指标。

**📊 数据集**

数据集采用合成的 VS Code 扩展测试集，共 28 条执行（3 条用于训练、25 条用于评估），涵盖通过、错误、假成功等多种情况。

**📈 对比分析**

与代理自身评估（CUA）对比，系统在所有测试类别上实现 100% 的检测准确率、精度、召回率和 F1，显著优于 CUA 的 82.2% 准确率和 60% 召回率。

**⚠️ 局限性**

局限性包括仅适用于通过轨迹的学习、依赖视觉状态表示（不适用于纯后端或非视觉任务）、LLM 调用成本、缺乏时间/时序约束以及实验仅基于合成数据、样本量有限。

---

## 13. HackerSignal: A Large-Scale Multi-Source Dataset Linking Hacker Community Discourse to the CVE Vulnerability Lifecycle

**arXiv ID:** 2605.03158 | [PDF](https://arxiv.org/pdf/2605.03158v1)

**作者:** Benjamin M. Ampel `[一作]` (Georgia State University), Sagar Samtani `[通讯]` (Indiana University Bloomington)

**通讯引用:** 2294 | [OpenAlex ID](https://openalex.org/A5038811607)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了名为 HackerSignal 的公开基准数据集，包含 7.45M 文档、64 个公开来源，并设计了三项跨源、跨时域的 CTI 评测任务（CVE 链接检索、漏洞类型分类、CVE 独立时间推广）。

**💡 创新点**

创新点在于：① 将从黑客论坛、漏洞数据库、官方公告到补丁提交等 8 层不同文本源统一映射到共享 CVE 标识空间；② 采用严格的时间切分与 CVE‑disjoint 评估，模拟真实系统需对未来威胁进行推断；③ 提供完整的治理与泄露诊断文档，保证公开基准的可信度与可复现性。

**🔧 技术方法**

技术方法包括：数据爬取与合并、Unicode 清洗、去重（SHA‑256 + MinHash‑LSH）、时间划分；检索基线使用 BM25、四类 dense bi‑encoders（MiniLM、mpnet、BGE、E5）及混合 BM25+dense；分类基线涵盖 Bag‑of‑Words（DT、TF‑IDF+LR、SVM）、循环网络（RNN/GRU/LSTM/BiLSTM）和领域特定 Transformer SecBERT。

**📊 数据集**

使用的数据集为 HackerSignal 本身：7.45M 文档、64 来源、8 层结构；主要对照语料为 NVD CVE 描述（340K 条）以及 ExploitDB、GitHub Advisory、CISA KEV 等公开漏洞与补丁记录，形成跨源 CVE 链接。

**📈 对比分析**

比较方法采用多模型基线阶梯；在检索任务上，BM25 R@1≈0.45，E5‑base‑v2 MRR≈0.60；在漏洞类型分类任务上，BiLSTM/GRU/ LSTM 的宏 F1≈0.87，SVM 为 0.80；所有任务均使用严格的时间切分，验证模型对未来 CVE 的泛化能力。

**⚠️ 局限性**

局限性包括：① CVE 链接为元数据自动推断，存在噪声与误链接；② 数据多语言覆盖有限，未评估跨语言检索；③ 未对大型语言模型或检索增强生成（RAG）进行实验；④ 评估未涵盖高成本模型或更复杂的安全任务。

---

## 14. A User-Centric Analysis of Explainability in AI-Based Medical Image Diagnosis

**arXiv ID:** 2605.02903 | [PDF](https://arxiv.org/pdf/2605.02903v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 15. Terminus-4B: Can a Smaller Model Replace Frontier LLMs at Agentic Execution Tasks?

**arXiv ID:** 2605.03195 | [PDF](https://arxiv.org/pdf/2605.03195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 16. Zero Day Attacks: Novel Behaviour or Novel Vulnerability?

**arXiv ID:** 2605.03138 | [PDF](https://arxiv.org/pdf/2605.03138v1)

**作者:** Nnamdi Jibunoh `[一作]` (New York Institute of Technology), Adetokunbo Makanju `[通讯]` (New York Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统回顾了过去20年的零日攻击事件，证明这些攻击主要通过未公开的硬件或软件漏洞实现，而非新的攻击行为；基于此提出了一套零日漏洞类型的分类法，并对现有基于机器学习的入侵检测系统（IDS）进行评估，指出它们往往关注“新颖行为”，与真实零日攻击的“未知漏洞”不符。

**💡 创新点**

创新点在于：①以漏洞为中心而非行为为中心重新定义零日攻击；②构建了涵盖硬件、软件、配置、供应链等维度的完整零日漏洞分类体系；③通过对三大公开数据库（INS、GPZ、ZERO‑DAY‑CZ）进行跨数据集映射，揭示了当前ML‑IDS评估方法与实际攻击机制的根本不匹配；④提出应将研究重心转向自动化漏洞发现与生命周期内的早期修补。

**🔧 技术方法**

使用的技术主要是：结构化数据抽取与预处理（利用OSINT来源、CVE、CWE信息）；文本映射与标签自动化（ChatGPT 生成标签后人工校验）；统计与可视化分析（Python、Matplotlib/Seaborn）；以及对比分析方法（交叉表、Venn图、趋势图）来评估不同漏洞类型在零日事件中的分布。

**📊 数据集**

使用的数据集包括：①自建 INS 数据集（42 起零日事件，覆盖 4 年）②Google Project Zero（GPZ，380 条）③ZERO‑DAY‑CZ（892 条）。三者均来自公开漏洞数据库、厂商安全公告、威胁情报报告等源。

**📈 对比分析**

比较方法：将三大数据集中的 CVE 进行映射，构建统一的零日漏洞层级；对各类漏洞在不同数据集中的占比进行柱状图和时间序列对比；并与常用的 Google 公开漏洞类别（Memory Corruption、Logic/Design Flaw 等）进行对照。性能方面并未直接评估 IDS 的检测准确率，而是通过对漏洞类型分布的统计表明现有 ML‑IDS 训练集往往与真实零日漏洞的分布差异显著，导致检测能力被高估。

**⚠️ 局限性**

局限性包括：①仅基于公开事件进行回顾，可能遗漏未披露的零日攻击；②没有对实际 ML‑IDS 进行实验验证，只是理论与数据对比；③依赖 ChatGPT 进行标签映射，可能存在自动化错误；④未讨论不同操作系统/平台对漏洞利用率的差异；⑤结论主要聚焦于漏洞检测的方向，缺乏对可操作化工具链的详细实现。

---

## 17. Dynamic Hypergame for Task Assignment in Multi-platform Mobile Crowdsensing Under Incomplete Information

**arXiv ID:** 2605.03569 | [PDF](https://arxiv.org/pdf/2605.03569v1)

**作者:** Sumedh J. Dongare `[一作]` (Communication Engineering Lab, Technical University of Darmstadt), Anja Klein `[通讯]` (Communication Engineering Lab, Technical University of Darmstadt)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了多平台移动众测在信息不完全环境下的任务分配问题，提出了双向匹配市场模型。

**💡 创新点**

创新点是将动态超游戏与感知自适应学习相结合，提出PACMAB框架以在缺失竞争者偏好与任务成本信息时实现最优匹配。

**🔧 技术方法**

采用匹配理论、超游戏理论、组合多臂赌博机（UCB）及感知剪枝等技术进行学习与优化。

**📊 数据集**

使用仿真生成的任务与移动单元数据（任务类型、支付级别、MU特征等）进行实验，没有公开真实数据集。

**📈 对比分析**

与OPT、PRISM、MGS、CMAB、Random MCSP等基线对比，PACMAB在社交福利上达93%–99%最优、任务完成率99%以上，显著优于其它学习方法。

**⚠️ 局限性**

局限在于仅在仿真环境验证，缺乏真实部署与动态用户行为的实际评估。

---

## 18. Multi-Agent Systems for Root Cause Analysis in Microservices

**arXiv ID:** 2605.03505 | [PDF](https://arxiv.org/pdf/2605.03505v1)

**作者:** Alexander Naakka `[一作]` (Zoner Oy), Mika V Mäntylä `[通讯]` (University of Helsinki)

**通讯引用:** 7277 | [OpenAlex ID](https://openalex.org/A5078824435)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 LATS-RCA，一个基于 LLM 的多智能体框架，用反射引导的树形搜索对微服务系统的日志和指标进行根因分析。

**💡 创新点**

创新点在于将根因分析视为多路径树形搜索，并通过 LLM 反射为搜索提供动态评估，从而实现系统化的多假设探索，突破传统线性推理。

**🔧 技术方法**

采用 Claude Sonnet 4.5 大语言模型、LangChain 与 LangGraph 进行工具调用和智能体编排，并结合 MCTS、UCT、反射评分和回传等技术实现决策搜索。

**📊 数据集**

使用公开的 Light‑OAuth2（LO2）微服务数据集以及真实生产环境收集的 37 条事故日志/指标数据进行评估。

**📈 对比分析**

通过与单/多 Agent ReAct 基线对比，LO2 上准确率提升至 91.3%（相较 39.8%/57.4%），但平均 API 调用 53.1 次、156K token、9.1 分钟；生产环境准确率约 60–70%，每案约 75 次调用、220K token、13 分钟。

**⚠️ 局限性**

在生产环境中受多语言堆栈、日志/指标异构、事件多因子等因素影响，准确率下降且计算成本提高；方法依赖结构化日志和统一规范，需进一步提升对非结构化日志的适应性。

---

## 19. From TinyGo to gc Compiler: Extending Zorya's Concolic Framework to Real-World Go Binaries

**arXiv ID:** 2605.03492 | [PDF](https://arxiv.org/pdf/2605.03492v1)

**作者:** Karolina Gorna `[一作]` (Telecom Paris and Ledger Donjon), Keith Makan `[通讯]` (University of the Western Cape)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

扩展了Zorya框架，使其能够分析由Go标准编译器生成的多线程二进制文件，自动检测真实世界的安全漏洞。

**💡 创新点**

创新点在于将覆盖路径分析（overlay concolic）与多线程状态恢复、无预处理和复制写入语义相结合，并实现对未执行分支的有限深度检查，首次在Go二进制中检测到无崩溃的整数溢出。

**🔧 技术方法**

采用了concolic执行、Ghidra P-Code中间表示、Z3 SMT求解器、copy‑on‑write覆盖机制、线程状态转储与恢复，以及panic‑reachability过滤等技术。

**📊 数据集**

使用了包含11个来自生产Go项目（Kubernetes、Go‑Ethereum、CoreDNS等）的真实漏洞实例的专门数据集，提供了重现工作流和触发输入。

**📈 对比分析**

与七种现有工具（静态分析器、模糊测试器、二进制符号执行器）进行对比，Zorya在所有11个漏洞中检测到7个，其中包括唯一的无崩溃整数溢出；平均检测时间约为16.5分钟，虽然比模糊测试慢，但能提供完整指令级执行轨迹。

**⚠️ 局限性**

局限性包括仅进行单一路径（function‑mode）探索，覆盖深度被固定为15条指令，可能漏报；无法检测并发错误；对复杂运行时依赖的支持仍有限，且对其他漏洞类型的覆盖仍需进一步扩展。

---

## 20. AI Advocate: Educational Path to Transform Squads to the Future

**arXiv ID:** 2605.03800 | [PDF](https://arxiv.org/pdf/2605.03800v1)

**作者:** Carla Soares `[一作]`, Marselle Silva `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

在巴西技术公司Zup Innovation内设计并实施了为238名软件工程师提供的AI Advocate培训课程，覆盖AI基础、领域上下文化和实操工具三大支柱，并通过两次知识测评与满意度调查评估学习效果。

**💡 创新点**

创新点在于构建面向混合团队的全流程教育框架，将生成式AI与责任AI原则、Spec-Driven Development和领域驱动设计结合，且通过内部培训团队快速推广，形成可复制的“AI Advocate”职业路径。

**🔧 技术方法**

采用的技术包括大型语言模型（LLM）与AI代理原理、Prompt/Context工程、Spec-Driven Development、AI辅助编码工具（市场及内部定制）以及责任AI框架。

**📊 数据集**

使用的数据主要来自内部：培训前后对同一受众进行的知识测评（共两轮）以及培训结束后86份满意度问卷，未涉及公开数据集。

**📈 对比分析**

通过对比培训前后知识测评得分（最高分比例从45.8%提升至61.7%）和NPS（80%为推荐者）验证效果，表明学习成果提升约35%，且受众满意度高。

**⚠️ 局限性**

局限性包括：仅在单一公司内部实施，缺乏长期跟踪评估；培训时长有限，理论与实操平衡不足；受众主要为中高级技术人员，对初级工程师的适应性尚未验证。

---

## 21. Design and Analysis of Quantum Dual-Containing CSS LDPC Codes based on Quasi-Dyadic Matrices

**arXiv ID:** 2605.03631 | [PDF](https://arxiv.org/pdf/2605.03631v1)

**作者:** Alessio Baldelli `[一作]` (Università Politecnica delle Marche), Paolo Santini `[通讯]` (Università Politecnica delle Marche)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了两种基于准二项矩阵的量子双重包含 CSS LDPC 码构造方法，能够实现高码率、可横向 Hadamard 门、低复杂度 BP 解码。

**💡 创新点**

创新点在于：① 通过准二项结构实现双重包含性和大自动群；② 设计了两种构造（A、B）并引入启发式算法减少 4-环；③ 这些构造在性能上优于现有双重包含的自行车码。

**🔧 技术方法**

使用了准二项/二项矩阵理论、CSS 码构造、贝叶斯图形信号传递（BP）解码、环路分析与蒙特卡罗仿真等技术。

**📊 数据集**

无外部数据集，采用随机生成的准二项码与在退化化通道（depolarizing channel）上的仿真，比较不同长度与参数的码。

**📈 对比分析**

通过在相同码率和行列权重下比较逻辑错误率（LER）与退化概率，结果显示所构造的码在相同条件下比自行车码具有更低的 LER；与非双重包含 QC‑LDPC 码相比，虽然在低错误概率下略逊一筹，但仍保持竞争力。

**⚠️ 局限性**

局限性：① 双重包含性必然导致 4‑环，最小距离受行权重上限；② 随着块长度增大，短环增多导致性能下降；③ 仅支持可横向 Hadamard 门，无法利用更大圆环的非双重包含码带来的更高 girth。

---

## 22. Dual-Foundation Models for Unsupervised Domain Adaptation

**arXiv ID:** 2605.03365 | [PDF](https://arxiv.org/pdf/2605.03365v1)

**作者:** Yerin Cheon `[一作]` (Stony Brook University), Francois Rameau `[通讯]` (SUNY Korea)

**通讯引用:** 1526 | [OpenAlex ID](https://openalex.org/A5090418377)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在语义分割的无监督域适应任务中，提出了利用两个基础模型（SAM和DINOv3）的双基线框架，先通过SAM进行基于超像素的提示和掩膜过滤以扩展伪标签的覆盖范围，再用DINOv3生成领域不变的类原型来引导对齐，进而提升模型的跨域泛化性能。

**💡 创新点**

创新点在于：①将SAM的超像素引导提示与掩膜过滤结合，使伪标签不再局限于高置信度像素，显著提升训练数据多样性；②采用DINOv3预训练特征生成固定、领域无关的类原型，避免了传统基于源模型的原型偏置和教师更新导致的崩溃问题；③两者协同实现伪标签细化与特征对齐，解决了伪标签稀疏与原型不稳定的双重瓶颈。

**🔧 技术方法**

核心技术包括：自监督对齐的EMA教师-学生自训练框架、基于SEEDS的超像素分割与SAM点提示、掩膜重叠过滤与高置信度+低熵伪标签选择、DINOv3 ViT-L特征提取、投影头与温度标定的原型对比损失。

**📊 数据集**

在GTA→Cityscapes（24,966张合成图）和SYNTHIA→Cityscapes（9,400张合成图）两个典型合成到真实的迁移场景上进行评估，Cityscapes作为目标域。

**📈 对比分析**

与多种主流方法（如ADVENT、DACS、ProDA、DAFormer、HRDA、MIC、COPT）对比，所提方法在GTA→Cityscapes和SYNTHIA→Cityscapes上分别提升mIoU约+1.3%和+1.4%，并在稀有或难辨类别（如火车、自行车、围栏、灯杆、交通灯、标志等）上表现尤为显著。

**⚠️ 局限性**

局限性包括：在已有强基线（如COPT、HRDA）上提升幅度有限，仍受底层自训练伪标签质量的制约；引入SAM和DINOv3会增加额外的计算与存储开销；方法主要针对语义分割任务，推广到其他任务仍需进一步验证。

---

## 23. DITRON: Distributed Multi-level Tiling Compiler for Parallel Tensor Programs

**arXiv ID:** 2605.02953 | [PDF](https://arxiv.org/pdf/2605.02953v1)

**作者:** Size Zheng `[一作]` (ByteDance Seed), Xin Liu `[通讯]` (ByteDance Seed)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个分层分布式张量编译器，扩展了 Triton 的 tile 编程模型，支持核心层、设备层和任务层的三层分块，实现了计算与通信的重叠与融合。

**💡 创新点**

主要创新点包括：① 在保持 Triton 兼容性的前提下引入多级分块；② 通过硬件无关的 OpenSHMEM 原语实现跨平台（NVIDIA、AMD、PCIe 等）代码生成；③ 设计分布式 Swizzling 技术，自动重排块执行顺序以最大化通信与计算重叠；④ 集成低延迟协议、D2D 复制融合和 PCIe 软同步，显著降低通信瓶颈。

**🔧 技术方法**

使用了 Triton、LLVM IR、OpenSHMEM、NVSHMEM / rocSHMEM、CUDA / ROCm 生态，以及自研的 Distributed IR 和代码生成器。

**📊 数据集**

在真实 LLM 模型（LLaMA、Mixtral、GPT、Qwen、DeepSeek、Qwen3-32B、LLaMA3-70B 等）及其对应的注意力、FFN、MoE、AllGather/ReduceScatter 等子任务上进行评估。

**📈 对比分析**

与 CuBLAS+NCCL、TileLink、FLUX、COMET、PyTorch Eager、Mirage、vLLM 等基线比较，平均实现 1.27×–19.18× 的加速；在集成 vLLM 时对大批量推理获得 5%–30% 的吞吐提升；在 AMD / PCIe GPU 上相对 ROCmBLAS+RCCL 或 CuBLAS+NCCL 亦实现 1%–38% 的加速。

**⚠️ 局限性**

局限性包括：需要手动编写 Triton 风格内核，学习曲线仍高；主要针对 GPU，尚未覆盖 CPU/TPU 等硬件；在极大规模集群或高度动态形状工作负载下，Swizzling 与低延迟协议的调优仍依赖经验；部分低层优化（如 LL 协议）依赖硬件特性，迁移到新平台时可能需重写原语。

---

## 24. VLMaxxing through FrameMogging Training-Free Anti-Recomputation for Video Vision-Language Models

**arXiv ID:** 2605.03351 | [PDF](https://arxiv.org/pdf/2605.03351v1)

**作者:** JF Bastien `[一作]`, Sam D'Amico `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对冻结的视频视觉语言模型（VLM）进行训练‑free 的反重算研究，探讨在同一视频上后续问答与首次帧剪裁如何提升推理速度。

**💡 创新点**

创新点包括提出三种机制：C‑PERSIST 后续重用与可修复尾部重填、C‑CEILING 阶段共享上限模型，以及 C‑VISION 首帧视觉裁剪；这些方法在不修改模型权重的前提下实现了数十倍后续速度提升。

**🔧 技术方法**

所用技术包括基于像素差分的路由器、可修复的 KV 缓存、视觉塔裁剪与保留率、以及阶段共享上限计量等；实验中还结合了可修复的尾部重填与适应性刷新策略。

**📊 数据集**

实验数据集涵盖 VideoMME、MVBench、TOMATO 以及大规模 26B 级别的 Gemma/RTSP 屏幕录制流，验证了方法在多种基准上的有效性。

**📈 对比分析**

与全密集 RGB 推理对比，C‑PERSIST 在后续问答中实现 14.9–35.9× 的速度提升且无配对漂移；C‑VISION 在首帧可得到 1.1–1.4× 的加速，但受视觉占比上限限制；在 26B 规模下，默认缓存路径不安全、后热前缀快照在小 N 下略有收益。

**⚠️ 局限性**

局限性包括仅适用于冻结模型、对特定架构（Qwen、Gemma）依赖、仅使用像素差分作为新鲜度信号、缺乏统一的低 FPS 基线、无法直接证明整体推理加速以及对真实流媒体场景的验证不足。

---

## 25. Enhancing AI-Based ECG Delineation with Deep Learning Denoising Techniques

**arXiv ID:** 2605.03183 | [PDF](https://arxiv.org/pdf/2605.03183v1)

**作者:** Jeff Breeding-Allison `[一作]` (Mars Petcare), Emil Walleser `[通讯]` (Mars Next Generation Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出并实现了基于自编码器的犬类心电图降噪框架，利用合成多种噪声训练并在降噪后进行波形细分；

**💡 创新点**

创新点包括：1）设计多种模拟噪声生成策略（基线漂移、白噪声、线性漂移、冲击脉冲）并在训练中引入(clean,clean)对以提升泛化；2）将下游波形细分性能作为降噪的优化目标；3）采用多通道数据增强，提升模型对不同导联噪声的鲁棒性；

**🔧 技术方法**

技术手段包括：PyTorch实现的深度自编码器（6层编码+6层解码，卷积核16）；多噪声变换与多导联随机化；信号相似度指标（SNR、SSD、MAD、cosine distance）；细分性能评估（峰值定位、区间重叠、异常分类）。

**📊 数据集**

数据集：10,000条10秒的犬心电图干净片段（人工标注），以及通过上述噪声变换合成的对应噪声对；评估集5,000条10秒记录，包含干净与合成噪声样本。

**📈 对比分析**

与传统滤波器（如elgendi2010）进行对比，使用SSD、MAD、cosine distance、SNR以及波形细分准确率等指标。结果显示，自编码器在多噪声情境下保持高细分准确率，虽然在纯信号抖动抑制上略逊于某些经典滤波，但其对下游临床任务的提升更显著。

**⚠️ 局限性**

局限性：需要人工挑选干净样本，噪声生成模型可能无法覆盖所有真实噪声类型；对极端或罕见噪声的鲁棒性尚未验证；模型训练和推理成本相对传统滤波较高，且未在真实现场数据上进一步验证。

---

## 26. AutoRAGTuner: A Declarative Framework for Automatic Optimization of RAG Pipelines

**arXiv ID:** 2605.02967 | [PDF](https://arxiv.org/pdf/2605.02967v1)

**作者:** Xintan Zeng `[一作]` (Ant Group), Jiajun Zhen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个可声明式配置驱动的框架AutoRAGTuner，用于自动化RAG的构建、执行、评估和优化。

**💡 创新点**

创新点包括模块化架构与组件注册机制、统一的Domain-Element Model（DEM）用于异构数据建模、可声明式JSON配置语言以及基于贝叶斯优化的全流程超参数调优。

**🔧 技术方法**

采用了模块化组件化、Domain-Element Model、声明式JSON配置、贝叶斯优化（含随机探索和EI获取函数）等技术。

**📊 数据集**

实验使用了HotPotQA和2WikiMultiHopQA数据集，LLM为Kimi-K2-Instruct-0905，检索器为Qwen3-Embedding-4B。

**📈 对比分析**

与默认基线对比，AutoRAGTuner在Vanilla RAG和Graph RAG两种管线中分别提升了5%~8%的Recall@5，F1提高最多4%；同时将代码量和调优时间降低到手动调优的十分之一以上。

**⚠️ 局限性**

局限性包括对贝叶斯优化的依赖可能在大规模超参数空间下效率受限；实验仅覆盖两种RAG管线，未验证在更广泛的场景下的适用性；以及需要手工定义DEM和JSON配置，学习曲线仍存在。

---

## 27. Revisiting Graph-Tokenizing Large Language Models: A Systematic Evaluation of Graph Token Understanding

**arXiv ID:** 2605.03514 | [PDF](https://arxiv.org/pdf/2605.03514v1)

**作者:** Zhongjian Zhang `[一作]` (Beijing University of Posts and Telecommunications), Chuan Shi `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 16065 | [OpenAlex ID](https://openalex.org/A5100705849)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

系统评估 Graph‑Tokenizing LLMs（GTokenLLMs）的图标记理解能力，提出并实验了 GTEval 评测流程，分析了不同模型在指令变化下的表现。

**💡 创新点**

创新点包括：① 统一 GTokenLLMs 的阶段化框架；② 设计基于格式与内容层面的指令变换评测方法；③ 通过注意力、结构扰动、指令微调等多维度实验，深入剖析图标记理解的根本瓶颈。

**🔧 技术方法**

使用的技术包括：指令变换（Rephrasing、Relabeling、Reversing、Randomizing）、PRBCD 结构攻击、t‑SNE 可视化、GCN 链接预测、注意力分布分析、额外指令微调，以及与 GTextLLM 的对比实验。

**📊 数据集**

主要使用的图数据集为 6 个代表性 GTokenLLM（LLaGA、InstructGLM、GraphGPT、GraphTranslator、TEA‑GLM、GOFA）在不同 TAG 数据集（如 Cora、CiteSeer、PubMed 等）上的测试集。

**📈 对比分析**

方法：在原始指令、Rephrasing、Relabeling、Reversing、Randomizing 五种指令变体下评估 6 个模型的准确率。结果显示：在原始指令下模型表现约 70%‑90%，但在格式/内容变动时准确率骤降（最高降幅 50%），整体性能远低于预期，表明图标记理解不足。

**⚠️ 局限性**

局限性：GTokenLLMs 对指令变化表现过敏或过不敏，过度依赖节点文本属性而非图结构；额外指令微调只能提升已见指令的表现，对未见指令无效；GTextLLM 在任务准确率上仍落后，无法弥补 GTokenLLMs 的理解缺陷。

---

## 28. Sparse Memory Finetuning as a Low-Forgetting Alternative to LoRA and Full Finetuning

**arXiv ID:** 2605.03229 | [PDF](https://arxiv.org/pdf/2605.03229v1)

**作者:** Prakhar Gupta `[一作]` (University of Michigan), Anirudh Kanchi `[通讯]` (University of Michigan)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5117335643)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在Qwen-2.5模型上实现Sparse Memory Finetuning，并与LoRA及全量微调在医学问答任务MedMCQA上进行对比。

**💡 创新点**

提出了增量式稀疏记忆微调与KL槽选择规则，展示了在保持低遗忘的同时提升任务性能的能力。

**🔧 技术方法**

采用Product Key Memory层、稀疏梯度更新、TF-IDF与KL槽选择、两阶段训练（密集retrofit与稀疏任务训练）等技术。

**📊 数据集**

使用MedMCQA作为目标任务，OpenAssistant作为背景统计，WikiText‑103和TriviaQA作为遗忘探针。

**📈 对比分析**

结果表明Additive Sparse (KL)在MedMCQA上提升约2.5个百分点，WikiText perplexity与TriviaQA准确率基本不变；LoRA与全量微调获得更高任务增益，但遗忘显著。

**⚠️ 局限性**

实验仅覆盖单一模型单一领域，遗忘评估仅限两项指标，未能完全分离稀疏更新与密集retrofit的影响，且在更大模型或不同领域的泛化尚未验证。

---

## 29. Stage Light is Sequence$^2$: Multi-Light Control via Imitation Learning

**arXiv ID:** 2605.03660 | [PDF](https://arxiv.org/pdf/2605.03660v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 30. Generalized Function-Correcting Partition Codes

**arXiv ID:** 2605.03370 | [PDF](https://arxiv.org/pdf/2605.03370v1)

**作者:** Charul Rajput `[一作]` (International Institute of Information Technology Hyderabad), V. Lalitha `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了能够同时保护多个消息空间划分的通用函数校正分区码（GFCPC）框架，归纳了前人关于FCC、FCPC以及带数据保护的FCC的结果并将其统一；

**💡 创新点**

创新点在于：①允许不同划分对应不同的距离要求，实现多级误差保护；②给出了多步构造方法并证明其正确性；③推导了通用上下界和距离需求矩阵（DRM）来刻画最优冗余；④在二进制两划分情况下给出了更强的下界；

**🔧 技术方法**

采用的技术主要是组合编码理论、集合划分与join运算、距离需求矩阵、Irregular-distance（𝒟）码框架、分组优化与多步编码迭代；

**📊 数据集**

本文以理论示例为主，使用了 𝔽₃³、𝔽₂³ 等有限域上的离散集合作为实验场景，未涉及真实数据集；

**📈 对比分析**

与传统的单个FCPC以及对所有划分取join后单一FCPC 的冗余比较，展示多步构造往往得到更小的冗余；在二进制两划分例子中，多步构造和上界接近最优，证明理论上可达到或接近最优冗余；

**⚠️ 局限性**

局限性包括：①对最优冗余的闭式表达仍未给出，仅给出上下界；②改进的二进制下界仅适用于二元域；③对非二元场下的更强下界缺乏；④多步构造需要预先知道划分间的join关系，实际实现时可能复杂。

---

## 31. TCD-Arena: Assessing Robustness of Time Series Causal Discovery Methods Against Assumption Violations

**arXiv ID:** 2605.03045 | [PDF](https://arxiv.org/pdf/2605.03045v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 32. Carbon-Aware Compute--Power Scheduling for AI Data Centers with Microgrid Prosumer Operations

**arXiv ID:** 2605.03751 | [PDF](https://arxiv.org/pdf/2605.03751v1)

**作者:** Johnny R. Zhang `[一作]` (Independent Researcher), Xian Sun `[通讯]` (Duke University)

**通讯引用:** 344 | [OpenAlex ID](https://openalex.org/A5104111079)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

针对分布式 AI 数据中心的微电网 prosumer 能源与计算需求，提出了联合碳意识的工作负载与电力调度方案。

**💡 创新点**

创新点在于将训练作业的非抢占性调度、推理任务的可路由性、冷却功率、局部发电、储能和双向电网交互统合进同一混合整数线性规划（MILP），并在模型中加入系统级碳预算约束。

**🔧 技术方法**

采用混合整数线性规划（MILP）建模，使用 Gurobi 求解器进行求解，并通过变量界限收缩、不可行路由预剔除等预处理加速。

**📊 数据集**

实验数据为基于实际业务特征构造的合成数据集，包含 3 个数据中心站点、24 小时时段、6 个硬性训练作业和 3 类弹性推理负载。

**📈 对比分析**

与 compute‑only、energy‑only、no‑battery、no‑routing、no‑carbon 等基线比较，联合 MILP 在目标值上提升约 3‑4 倍，排放量降至约 35% 左右；运行时间在实验规模内均在秒级完成。

**⚠️ 局限性**

局限性：模型为确定性有限时域，未考虑需求、电价、发电和碳强度的随机性；求解复杂度随时间段和作业数呈指数级增长；仅适用于中小规模实例，需进一步研究分解或在线扩展。

---

## 33. Complex Analysis of Channel Polarization on discrete BMS Channels

**arXiv ID:** 2605.03805 | [PDF](https://arxiv.org/pdf/2605.03805v1)

**作者:** Dongxiao Xu `[一作]` (Technical University of Munich), Holger Boche `[通讯]` (Technical University of Munich)

**通讯引用:** 13598 | [OpenAlex ID](https://openalex.org/A5000732219)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了复杂分析框架，推导了极化码在有限块长下的Bhattacharyya参数的精确表达式。

**💡 创新点**

创新点在于将Mellin变换、超几何函数与多项式展开相结合，得到闭式公式并显著降低计算复杂度。

**🔧 技术方法**

使用了复分析技术（Mellin–Barnes 积分、Pfaff与Euler变换等），以及超几何函数和概率分布的矩生成函数。

**📊 数据集**

主要使用二元对称信道（BSC）作为实验信道，无需外部数据集。

**📈 对比分析**

通过与Monte Carlo仿真比较，理论结果与数值高度吻合，准确率几乎为100%，显著减少了数值误差。

**⚠️ 局限性**

局限在于仅适用于二元对称信道及中等层数，扩展到更复杂信道或更大层数时公式仍然繁琐，计算量显著增加。

---

## 34. Graph Reconstruction from Differentially Private GNN Explanations

**arXiv ID:** 2605.03388 | [PDF](https://arxiv.org/pdf/2605.03388v1)

**作者:** Rishi Raj Sahoo `[一作]` (National Institute of Science Education and Research), Subhankar Mishra `[通讯]` (National Institute of Science Education and Research)

**通讯引用:** 353 | [OpenAlex ID](https://openalex.org/A5030378871)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对在 GDPR 等监管框架下必须公开的 GNN 解释（post‑hoc attribution），本文提出了一种基于逆扩散的重建攻击——PrivX，能够在仅观察差分隐私（DP）扰动后的解释的情况下，恢复图的隐藏结构。

**💡 创新点**

创新点在于①将 Gaussian DP 视为单步 DDPM 前向扩散，逆扩散等价于在已知噪声模型下的贝叶斯去噪；②构建了一个分层对手模型（参数 (τ,δ̂,𝒮,ρ)），在不完全了解 DP 参数且只观测部分节点的情形下给出两侧 AUC 上下界；③提出诊断工具  来拆分解释器诱发泄漏与图本身分布诱发泄漏；④给出基于 homophily/heterophily 的可解释性与隐私泄漏的分层指导。

**🔧 技术方法**

使用技术包括：差分隐私机制（Gaussian、Laplace、Rényi），DDPM 逆扩散框架，基于图 Transformer 的去噪网络，跨注意力与时间步条件，DP 噪声校准与估计，AUC/AP 评估，实验对比基线（ExplainSim、GSE、FeatureSim、SLAPS）。

**📊 数据集**

实验数据集共七个：三类同质性图（Cora、CiteSeer、PubMed）、大规模同质图（ogbn-arxiv）、三类异质性图（Chameleon、IMDB、Amazon‑Ratings）。

**📈 对比分析**

与现有基准方法（ExplainSim、GSE、FeatureSim、SLAPS）对比，PrivX 在 AUC/AP 上显著提升（例如在 Cora 上 GNNExplainer + GraphSAGE 的 AP 由 0.51 提升至 0.83，AUC 由 0.51 提升至 0.89），在多数数据集上 DP 预算 ε≈5 时即可达到 ≥0.7 的 AP，表明在常用预算下 DP 保护不足。

**⚠️ 局限性**

主要局限：①密集图重建复杂度 O(Tn²)，仅适用于 n≈10³ 以下的子图；②对 heterophilic 图的理论下界依赖于特征‑结构反相关性，若此相关性弱则攻击效果退化；③仅评估 Gaussian、Laplace、Rényi DP，未考虑自适应 DP 或训练时扰动；④诊断分解仅捕捉解释器诱发与图分布诱发泄漏的部分，未覆盖训练过程或查询策略对泄漏的影响。

---

## 35. Benchmarking Logistic Regression, SVM, Naive Bayes, and IndoBERT Fine-Tuning for Sentiment Analysis on Indonesian Product Reviews

**arXiv ID:** 2605.03439 | [PDF](https://arxiv.org/pdf/2605.03439v1)

**作者:** Nabila Zakiyah Zahra `[一作]` (Institut Teknologi Sumatera), Martin C. T. Manullang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对印尼Tokopedia产品评论的三类情感分析进行基准实验，比较传统TF‑IDF+机器学习模型与细调的IndoBERT模型。

**💡 创新点**

在同一实验管线下针对极度类别不平衡提出自定义加权交叉熵损失，并首次系统对比传统稀疏特征方法与Transformer的实际表现。

**🔧 技术方法**

TF‑IDF特征提取+Logistic回归/Linear SVC/Multinomial Naive Bayes；IndoBERT细调并使用加权交叉熵；预处理正则化、分词、长度截断；部署使用Gradio与Hugging Face Spaces。

**📊 数据集**

Tokopedia Product Reviews 2025 数据集（约65,335条评论）用于基线模型；IndoBERT实验采用约1,800条训练/3,000条测试的子集。

**📈 对比分析**

使用准确率、宏F1、加权F1三指标进行评估；Linear SVC在完整数据集上取得97.60%准确率和0.551宏F1；IndoBERT在子集上仅得到88.70%准确率和0.509宏F1，显示传统模型在当前设置下仍更优。

**⚠️ 局限性**

实验不对齐：基线使用全数据集，Transformer仅用子集；样本量不足限制Transformer表现；仅通过加权交叉熵处理不平衡，未尝试更高级样本或损失方法；缺少完整混淆矩阵与细粒度误差分析。

---

## 36. Beyond Distributive Justice: Hermeneutical Fairness in Ad Delivery

**arXiv ID:** 2605.03419 | [PDF](https://arxiv.org/pdf/2605.03419v1)

**作者:** Camilla Quaresmini `[一作]` (Politecnico di Milano), Giulia De Pasquale `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 158 | [OpenAlex ID](https://openalex.org/A5088027101)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过引入Miranda Fricker的诠释不公理论，提出将在线广告公平性从传统的分配性扩展到诠释性维度，定义了诠释性剥夺与扭曲两种广告造成的知识公平危害，并在广告投放优化框架中加入诠释性成本和诠释性机会约束，构建了一种同时考虑经济效用、分配公平与诠释公平的多目标投放策略；

**💡 创新点**

创新点在于：①首次将诠释不公理论与广告公平性相结合，提出诠释性公平约束与成本；②将诠释资源视为可分配资产，设计了群体层面的诠释机会均等约束；③通过仿真展示传统分配约束与诠释公平约束在效用和公平性上的权衡；

**🔧 技术方法**

主要技术包括：基于线性/凸优化的最优投放策略求解、贝塔分布采样生成用户特征、Monte Carlo仿真评估不同约束下的效用与公平性指标、统计显著性分析；

**📊 数据集**

采用了1986-1987年AIDS Advertising Evaluation Surveys的离线广告数据作为实证基准，未使用真实在线广告数据；

**📈 对比分析**

通过数值仿真将无约束、曝光平衡、机会平衡、诠释机会平衡等策略进行比较；结果显示引入诠释性约束后，平台的经济效用下降约1-2%，但曝光差距显著缩小，诠释性成本亦得到显著降低；

**⚠️ 局限性**

局限性包括：①缺乏对在线广告环境的真实实证验证，使用离线数据仅为启发性展示；②诠释性参数（θ、ω、ξ）未通过实验或实际数据估计；③模型仅考虑两类受保护群体、单一广告，未覆盖多广告、多群体和动态反馈；④隐私与数据获取风险未得到充分解决；

---

## 37. DeRelayL: Sustainable Decentralized Relay Learning

**arXiv ID:** 2605.02935 | [PDF](https://arxiv.org/pdf/2605.02935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 38. Dynamic Detours

**arXiv ID:** 2605.03225 | [PDF](https://arxiv.org/pdf/2605.03225v1)

**作者:** Daniel Dadush `[一作]` (CWI and Utrecht University), Michał Włodarczyk `[通讯]` (University of Warsaw)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5103139616)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了三种动态图结构，用于查询给定端点的长路径、长 detour 以及偶奇长度路径。

**💡 创新点**

创新点在于将“延迟边插入”技术扩展到双连通分量层面，并结合动态树宽和动态二分图检测，实现在高树宽或低树宽两种情形下均能快速回答查询。

**🔧 技术方法**

主要技术包括动态双连通分量维护、动态树宽维护、动态二分图维护、树分解自动机、以及基于树宽的 win/win 思想。

**📊 数据集**

论文未使用实际数据集，而是在理论模型（word RAM）下给出时间复杂度。

**📈 对比分析**

相对于先前针对短路径的动态结构，本文在长路径/ detour 查询上实现了对参数 k 的 FPT 复杂度；长路径/ detour 的查询耗时为 2^{k^3}log n + O(log^2 n log^2 log n)，偶奇路径的查询耗时为 O(log^2 n log^2 log n)。

**⚠️ 局限性**

主要局限在于对参数 k 的指数依赖仍较大，2^{k^3} 不是最优，且实现高度依赖当前最优的双连通分量数据结构；若该结构进一步改进，可提升整体性能。

---

## 39. Hybrid Machine Learning and Physical Modeling of Feedstock Deformation During Robotic 3D Printing of Continuous Fiber Thermoplastic Composites

**arXiv ID:** 2605.03186 | [PDF](https://arxiv.org/pdf/2605.03186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 40. Where Paths Split: Localized, Calibrated Control of Moral Reasoning in Large Language Models

**arXiv ID:** 2605.03609 | [PDF](https://arxiv.org/pdf/2605.03609v1)

**作者:** Chenchen Yuan `[一作]` (Technical University of Munich), Gjergji Kasneci `[通讯]` (Technical University of Munich)

**通讯引用:** 14818 | [OpenAlex ID](https://openalex.org/A5024434748)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在推理阶段对大型语言模型进行道德偏好调控的方法，通过定位并编辑Transformer内部的分支点来实现不同伦理框架（功利主义与义务论）的控制；

**💡 创新点**

创新点在于①引入Convergent-Divergent Routing（CDR）定位共享注意力但FFN分离的分支点并进行局部门控；②使用Common Spatial Patterns（CSP）在残差空间提取一对对应功利主义和义务论的对立方向；③提出Dual Logit Calibration（DLC）在该二维子空间内最小化L2范数实现精确的加权调节，从而把偏好权重转化为可解释的比例控制；

**🔧 技术方法**

主要技术包括注意力头探针与线性回归、FFN向量对齐、分支点门控、CSP特征提取、DLC闭式更新、以及对残差流的实时编辑；

**📊 数据集**

使用的公开数据集有ETHICS（义务论和功利主义子集）和AITA（日常道德困境），并对AITA文本进行摘要；

**📈 对比分析**

与基线（prompt-only、top‑K head steering、Best‑Layer Post‑FFN Ratio Steering）比较，本文方法在U_op与α_U之间的偏差最小，MAE低至≈1.8个百分点，保持了模型在TriviaQA、GSM8K等通用任务上的能力；

**⚠️ 局限性**

限制主要在于仅处理两大伦理框架，适用的注意力架构局限于标准MHA，实验范围聚焦于道德推理任务，未来需扩展至多元价值体系、GQA等架构以及更广泛的任务场景。

---

## 41. MARS-DA: A Hierarchical Reinforcement Learning Framework for Risk-Aware Multi-Agent Bidding in Power Grids

**arXiv ID:** 2605.03142 | [PDF](https://arxiv.org/pdf/2605.03142v1)

**作者:** Jiayi Chen `[一作]` (New Jersey Institute of Technology), Guiling Wang `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 22328 | [OpenAlex ID](https://openalex.org/A5100744284)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了基于PJM真实市场数据的高保真双结算电力市场Gymnasium环境，并提出了MARS-DA多代理层级强化学习框架，用于在日间与实时市场之间动态平衡收益与风险。

**💡 创新点**

创新点包括：①开源的、可复现的两结算市场模拟环境；②通过Meta-Controller动态混合“安全代理”与“投机代理”，实现对不同市场态势的自适应分区；③在Meta-Controller中引入凸合约收益的concave utility奖励，显著提升风险调整后回报。

**🔧 技术方法**

使用了PPO层级强化学习、角色专化奖励、风险敏感的concave utility、周期性特征编码、线性插值与季节平均的历史数据预处理，训练分为“University Phase”和“Manager Phase”。

**📊 数据集**

采用了PJM Interconnection 2018-2022年历史DA/RT LMP、负荷、气象与天然气价格数据；测试集包含2024-2025年最近两年的市场情况，确保模型在外推条件下的鲁棒性。

**📈 对比分析**

与NNSF（PPO）、SAC、CVaR-PPO、RollingOpt等四个基线对比，评估指标包括Sharpe Ratio、Sortino Ratio、最大回撤和Regime Alignment Score。MARS-DA在两测试期均取得最高Sharpe和最低回撤，显示出显著的风险调整收益优势。

**⚠️ 局限性**

局限性在于：①仅针对单一燃气机型，缺乏多资产与多市场协同的考量；②Meta-Controller需要冻结基代理，可能限制进一步的策略演化；③对极端异常事件（如系统崩溃、价格跳跃）的鲁棒性仍未完全验证；④训练与评估高度依赖PJM数据，跨市场推广需更多验证。

---

## 42. Adaptive Dual-Path Framework for Covert Semantic Communication

**arXiv ID:** 2605.03423 | [PDF](https://arxiv.org/pdf/2605.03423v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 43. CropVLM: A Domain-Adapted Vision-Language Model for Open-Set Crop Analysis

**arXiv ID:** 2605.03259 | [PDF](https://arxiv.org/pdf/2605.03259v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 44. Deep Graph-Language Fusion for Structure-Aware Code Generation

**arXiv ID:** 2605.03689 | [PDF](https://arxiv.org/pdf/2605.03689v1)

**作者:** Mert Tiftikci `[一作]` (Technische Universitaet Darmstadt), Mira Mezini `[通讯]` (Technische Universitaet Darmstadt)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CGFuse 框架，将图神经网络提取的代码图结构信息在预训练语言模型的中间层中逐 token 融合，从而提升代码生成性能。

**💡 创新点**

通过在 PLM 内部深度插入 GNN 表示实现 token 级别的结构融合，克服传统提示或稠密特征提取导致的信息损失，并系统评估多种模型与 GNN 架构的效果。

**🔧 技术方法**

采用 GNN（R‑GCN、GraphSAGE、GIN）与 Transformer 语言模型的融合技术，预训练 GNN 作为图专家，使用 λ 控制特征融合，并在多种 encoder/encoder‑decoder 架构上微调。

**📊 数据集**

主要使用 CONCODE 代码生成基准数据集（约 10 万 Java 类），并构建 AST 与 DFG 代码图进行训练与评估。

**📈 对比分析**

与原始 NL、代码预训练模型以及不同 GNN 配置进行对比，使用 Exact Match、BLEU、CodeBLEU 等指标；在多模型上 CGFuse 实现 BLEU 提升 10–16%、CodeBLEU 提升 6–11%，并展示出显著的样本效率优势。

**⚠️ 局限性**

深层 GNN（多层）导致过度平滑和性能下降；对不完整代码的图构建与解码仍具挑战；在更大模型或更复杂任务上的可扩展性尚需进一步验证。

---

## 45. Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies

**arXiv ID:** 2605.03596 | [PDF](https://arxiv.org/pdf/2605.03596v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 46. Learning Generalizable Action Representations via Pre-training AEMG

**arXiv ID:** 2605.03462 | [PDF](https://arxiv.org/pdf/2605.03462v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 47. KVerus: Scalable and Resilient Formal Verification Proof Generation for Rust Code

**arXiv ID:** 2605.03822 | [PDF](https://arxiv.org/pdf/2605.03822v1)

**作者:** Yuwei Liu `[一作]` (Ant Group), Tao Wei `[通讯]` (Ant Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个检索增强、自适应的Verus Rust验证系统，用于在多模块、不断演进的代码库中自动生成和修正证明。

**💡 创新点**

①解决了“语义-结构差距”，通过构建跨模块依赖图、语义词条化的lemma知识库和错误驱动自我修正；②实现了基于RAG的检索增强、语义lemma索引和错误驱动自适应修正；③系统能够在代码或工具链演进时自动更新知识并重新生成证明。

**🔧 技术方法**

检索增强生成（RAG）、LLM（Claude Sonnet 4.0）推理、静态代码分析（verus-analyzer）、知识图谱（代码元数据图与lemma索引）、错误分类与自适应修正机制。

**📊 数据集**

四个人工设计的单文件基准（Verus-Bench、MBPP、Human-Eval、MathSpec-Bench）和两个真实项目（Memory Allocator、CortenMM），以及在Asterinas内核的实验。

**📈 对比分析**

与AutoVerus、AutoProof等现有LLM验证工具比较，单文件任务准确率从56.9%提升到80.2%（+23.3%），仓库级任务从4.5%提升到51.0%；在Asterinas验证23个函数，验证率21%；同时在成本（token数）和对工具链更新的鲁棒性方面显著优于基线。

**⚠️ 局限性**

高度依赖大模型（Claude 4.0），对模型可访问性和复现性有限；数据稀缺导致错误率较高，生成的证明往往比人工证明冗长；未能充分利用SMT求解器信息，难以实现证明压缩；对规范错误缺乏主动检测与校正。

---

## 48. Learning to Segment using Summary Statistics and Weak Supervision

**arXiv ID:** 2605.03059 | [PDF](https://arxiv.org/pdf/2605.03059v1)

**作者:** Omkar Kulkarni `[一作]` (University of Maryland), Tim Oates `[通讯]` (University of Maryland)

**通讯引用:** 7421 | [OpenAlex ID](https://openalex.org/A5114778025)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在医学图像分割中，利用仅有的ROI百分比统计与少量中心像素标注，训练出能够实现近似全监督水平的分割模型。

**💡 创新点**

提出将统计损失、弱监督损失与自监督重构、置信度损失相结合的复合损失函数，解决仅有统计信息与极少像素标注的弱监督分割难题。

**🔧 技术方法**

使用DeepLabV3+ResNet-50架构，并加入自监督重构（autoencoder）输出、置信度损失、L1统计损失以及基于弱掩模的交叉熵损失。

**📊 数据集**

实验数据集包括乳腺超声图像集合BUSI和肾肿瘤CT图像集合KiTS23，分别处理成二维切片。

**📈 对比分析**

与仅统计损失、仅弱监督损失以及完整监督（全标注）三种对照设置比较，结果显示在BUSI上IoU从45.29%提升至63.15%，在KiTS23上从8.75%提升至59.85%，接近全监督水平。

**⚠️ 局限性**

对视觉区分度低的目标（如肿瘤）性能仍有限；弱监督仅提供正样本，缺少负样本信号；目前仅针对二维切片，尚未验证3D体数据；需要进一步改进负样本提示与模型泛化能力。

---

## 49. The Algebra of Iterative Constructions

**arXiv ID:** 2605.03176 | [PDF](https://arxiv.org/pdf/2605.03176v1)

**作者:** Kevin Batz `[一作]` (Cornell University), Todd Schmid `[通讯]` (Bucknell University)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5087089457)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了迭代构造代数（AIC），通过等式逻辑对完整格上的连续映射的固定点迭代进行形式化，并在Isabelle/HOL中实现了AIC，利用它给出了Tarski‑Kantorovich、Kleene、Park、Olszewski以及latticed k‑induction等固定点定理的纯代数证明。

**💡 创新点**

创新点包括：① 将固定点迭代视作代数运算，消除显式索引；② 推导出改进的Olszewski固定点定理；③ 建立AIC的可计算推理体系；④ 分析并证明有限等式公理不完备、提供无穷公理的完备性结果。

**🔧 技术方法**

主要技术手段为：等式逻辑与准等式推理、序列代数的定义、连续性与半连续性公理、可递归归约、Isabelle/HOL的浅嵌入实现以及Sledgehammer自动推理。

**📊 数据集**

本研究为理论工作，没有使用实验数据集；所有证明均在数学和形式化证明环境中完成。

**📈 对比分析**

与传统的基于索引的证明方法相比，AIC能够在更高层次上自动化证明：在Isabelle/HOL中使用Sledgehammer可完全自动化证明Tarski‑Kantorovich、Park等定理；相比手工证明，逻辑步骤更短且易于复现。

**⚠️ 局限性**

局限性包括：① 目前仅适用于可数完备格；② 不能处理传递性固定点（超限迭代）；③ 有限等式公理系统不可完备；④ 需要ω‑连续性或更强的连续性假设。

---

## 50. Approaching human parity in the quality of automated organoid image segmentation

**arXiv ID:** 2605.03053 | [PDF](https://arxiv.org/pdf/2605.03053v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 51. Information Plane Analysis of Binary Neural Networks

**arXiv ID:** 2605.03636 | [PDF](https://arxiv.org/pdf/2605.03636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 52. GRAFT: Auditing Graph Neural Networks via Global Feature Attribution

**arXiv ID:** 2605.03377 | [PDF](https://arxiv.org/pdf/2605.03377v1)

**作者:** Rishi Raj Sahoo `[一作]` (National Institute of Science Education and Research), Subhankar Mishra `[通讯]` (National Institute of Science Education and Research)

**通讯引用:** 353 | [OpenAlex ID](https://openalex.org/A5030378871)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种全局基于输入特征的GNN解释框架GRAFT，用于生成各类别的特征重要性档案并自动生成可读自然语言规则。

**💡 创新点**

创新点在于：①利用Farthest Point Sampling选择多样化的代表节点作为例子；②使用Integrated Gradients对每个例子做精细梯度归因；③通过聚合得到稳健的类别级特征重要性谱；④将重要特征列表交给LLM生成规则并自我校正；⑤设计多维度评估工具（bias检测、迁移学习、忠实度、稳定性、跨架构共识）。

**🔧 技术方法**

核心技术包括：Farthest Point Sampling、Integrated Gradients（Captum实现）、均值/置信度加权聚合、LLM（Claude Sonnet 4.6等）自然语言生成与自我校正、以及一系列量化指标（Jaccard稳定性、共识率、对比档案等）。

**📊 数据集**

在13个节点分类基准上进行实验，涵盖文本特征的Cora、CiteSeer、PubMed等；结构特征的Coauthor-CS/Physics、Amazon Computers/Photo；异质图的Chameleon、Squirrel；WebKB（Wisconsin、Cornell、Texas）以及Actor；并在四种主流GNN架构（GCN、GAT、GraphSAGE、GIN）上评估，共计260次实验。

**📈 对比分析**

与频率基线、随机基线以及多模型共识等方法对比：在bias检测实验中，GRAFT能在所有11个数据/架构组合中将注入的短路特征排在前3（9/11中排在第1）；在迁移学习中，GRAFT选出的特征在大部分数据集上达到或超过频率基线，且在少数特征占比极低的情况下仍能保持接近完整模型的精度；在忠实度评估中，GRAFT在文本特征图中保留60–90%的原始准确率，显著优于随机和频率，且在异质图上可解释性较低。性能最好的组合为Coauthor-CS/GraphSAGE，Fidelity 0.903；稳定性最高的为Cora/GCN和Squirrel/GIN，Jaccard 1.00。

**⚠️ 局限性**

局限性：①零基线的IG在连续或非稀疏特征上可能效果欠佳；②LLM生成的规则对特征命名依赖强，匿名特征索引时可读性下降；③示例选择k和特征数K是超参数，尽管实验表明鲁棒，但仍需进一步自动化；④在特征维度极高、贡献分散的图（如Amazon）中，IG的分辨率下降，导致性能趋于频率基线；⑤跨架构共识低说明模型选择对解释结果影响显著，需要多模型对比。

---

## 53. Attribution-Guided Masking for Robust Cross-Domain Sentiment Classification

**arXiv ID:** 2605.03091 | [PDF](https://arxiv.org/pdf/2605.03091v1)

**作者:** Shubham Harkare `[一作]` (University of Michigan), Yash Kulkarni `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究跨域情感分类的泛化问题，提出在训练期间使用梯度归因动态检测并抑制域特异性噪声词的 Attribution‑Guided Masking（AGM）方法，从而提升零样本迁移性能。

**💡 创新点**

创新点包括：① 证明归因漂移无法预测泛化失败；② 引入基于梯度归因的掩蔽损失 ℒ_mask，实时抑制模型对伪相关词的依赖；③ 无需目标域标签或人工标注，且在训练中提供 token‑level 可解释性。

**🔧 技术方法**

使用预训练 RoBERTa‑base 作为编码器，梯度×输入近似归因，ℒ_mask 以及可选的 counterfactual contrastive loss ℒ_CCL，在严格的零样本留一（leave‑one‑out）设置下训练与评估。

**📊 数据集**

四个英文情感分类数据集：IMDb、Amazon、TripAdvisor、Sentiment140。

**📈 对比分析**

与 DANN、IRM、DRO、Fish 等五个强基线在四个源‑目标组合上进行零样本比较，使用 Generalization Gap Δ 作为指标；AGM 在最难的 Sentiment140 转移上取得 Δ=0.244，优于 DANN(0.264)、DRO(0.248)、Fish(0.247)，与 IRM(0.238)相近且方差更小。

**⚠️ 局限性**

局限性：仅评估二分类英文情感任务；训练成本约为标准微调的 2–3 倍；不同方法在最强转移上的差异置信区间重叠，难以显著区分；未验证多类、多语种或非情感任务。

---

## 54. Joint Energy Management and Coordinated AIGC Workload Scheduling for Distributed Data Centers: A Diffusion-Aided Reward Shaping Approach

**arXiv ID:** 2605.02965 | [PDF](https://arxiv.org/pdf/2605.02965v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 55. Attention: What Prevents Young Adults from Speaking Up Against Cyberbullying in an LLM-Powered Social Media Simulation

**arXiv ID:** 2605.03287 | [PDF](https://arxiv.org/pdf/2605.03287v1)

**作者:** Qian Yang `[一作]` (Cornell University), Natalie N. Bazarova `[通讯]` (Cornell University)

**通讯引用:** 4503 | [OpenAlex ID](https://openalex.org/A5042135500)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了基于LLM的多智能体社交媒体仿真系统Upstanders' Practicum，供34名青年成年人练习公开干预网络欺凌。

**💡 创新点**

发现并阐明了三重注意力转移是促使公开干预的关键，提出针对真实关注、上扬“正义发声”身份、公共规范设定的设计机会，并开源Truman Agents平台。

**🔧 技术方法**

使用多模态LLM代理（如ChatGPT等），结合Truman Agents模拟框架，实现动态角色扮演、文本交互与反馈。

**📊 数据集**

基于先前实证研究的网络欺凌情境，设计四类情境（强制霸凌、网络跟踪、鲁莽曝光、故意曝光），不使用公开数据集。

**📈 对比分析**

未与其他方法做量化比较，仅通过质性访谈与行为记录评估，发现完成三重注意力转移后公开干预率显著提升。

**⚠️ 局限性**

限制包括单次实验室会话、受限样本、缺乏真实社交媒体环境、未验证长期影响，且受实验情境与潜在实验偏差限制。

---

## 56. Natural Language Processing: A Comprehensive Practical Guide from Tokenisation to RLHF

**arXiv ID:** 2605.03799 | [PDF](https://arxiv.org/pdf/2605.03799v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 57. Graph Convolutional Support Vector Regression for Robust Spatiotemporal Forecasting of Urban Air Pollution

**arXiv ID:** 2605.03795 | [PDF](https://arxiv.org/pdf/2605.03795v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 58. SoK: After Decades of Web Tracker Detection, What's Next?

**arXiv ID:** 2605.02982 | [PDF](https://arxiv.org/pdf/2605.02982v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 59. Learning Discriminative Signed Distance Functions from Multi-scale Level-of-detail Features for 3D Anomaly Detection

**arXiv ID:** 2605.03437 | [PDF](https://arxiv.org/pdf/2605.03437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 60. Moral Sensitivity in LLMs: A Tiered Evaluation of Contextual Bias via Behavioral Profiling and Mechanistic Interpretability

**arXiv ID:** 2605.03217 | [PDF](https://arxiv.org/pdf/2605.03217v1)

**作者:** Yash Aggarwal `[一作]` (University of Maryland), Manas Gaur `[通讯]` (University of Maryland)

**通讯引用:** 1571 | [OpenAlex ID](https://openalex.org/A5023667301)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出七层道德压力测试与道德敏感性指数（MSI），评估并机制验证大型语言模型在不同情境下的偏见行为；同时发现推理蒸馏会导致偏见的“U曲线”回弹。

**💡 创新点**

① 将偏见量化为梯度化指标MSI，捕捉从中立推理到安全对齐的渐进转变；② 结合行为与机制的五步解释管道（logit lens、注意力分析、激活补丁、语义方向、OV电路重构）；③ 揭示推理蒸馏在压缩模型时会破坏安全相关的内部表示，导致偏见反弹。

**🔧 技术方法**

多维偏见量化（偏见得分、模糊度、熵系数）；机制解释技术包括logit lens、注意力差异分析、激活补丁、语义方向投影、OV电路重构；使用多模型对照实验（Claude、Qwen、Llama 3、Gemini 1.5）。

**📊 数据集**

七层分层的电车难题数据集（从纯数字到历史不公的逐步增添社会语境）以及日常偏见测试集。

**📈 对比分析**

对四大模型分别计算MSI、偏见率、模糊率、熵，并对比其在不同层级的表现；在蒸馏模型上发现偏见从高到低再到高的U形曲线；结果表明大模型在低层情境下偏见低，但在高层出现显著偏见；蒸馏后偏见反弹，说明压缩过程破坏了安全表示。

**⚠️ 局限性**

研究仅针对四个特定模型与单一伦理任务，无法覆盖多语言、多任务场景；机制解释需大量计算资源，规模化应用受限；MSI系数的泛化与解释力尚未在更广泛数据集上验证；未探讨如何在蒸馏目标中显式保留安全表示。

---

## 61. AsymK-Talker: Real-Time and Long-Horizon Talking Head Generation via Asymmetric Kernel Distillation

**arXiv ID:** 2605.02948 | [PDF](https://arxiv.org/pdf/2605.02948v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 62. Gated Subspace Inference for Transformer Acceleration

**arXiv ID:** 2605.03109 | [PDF](https://arxiv.org/pdf/2605.03109v1)

**作者:** Stephen J. Thomas `[一作]` (Lehigh University), Stephen J. Thomas `[通讯]` (Lehigh University)

**通讯引用:** 1806 | [OpenAlex ID](https://openalex.org/A5111448726)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 Gated Subspace Inference（GSI）的推理加速方法，利用每层激活向量的低有效秩对线性层权重读取进行加速。

**💡 创新点**

核心创新点在于：①对激活进行子空间分解并缓存低秩权重图；②引入门控残差通道，动态决定是否跳过残差校正；③通过深度级连锁初始化和时间维度增量追踪实现高效的子空间维护，保持无损推理。

**🔧 技术方法**

使用的技术包括：子空间追踪（DGKS）、低秩投影、残差门控、深度连锁（cascade）初始化、ADA（注意力层低秩加速）以及针对AMD MI300X的自定义GPU内核。

**📊 数据集**

数据集为 512 词长的多领域英文文本（包含数学、烹饪、财经、太空探索等），用于模型校准和性能评估。

**📈 对比分析**

在 AMD MI300X 上对 GPT‑2 124M、GPT‑J 6B、OPT 6.7B 进行实验，线性层权重读取加速 3×–16×，perplexity 比例 < 1、top‑1 预测匹配率 > 98%，greedy 生成与基线完全一致。

**⚠️ 局限性**

限制：需要一次校准并固定子空间，长上下文或分布漂移时可能需要在线适配；对包含高维位置嵌入的模型（如 OPT）效果受限；实现依赖自定义 GPU 内核，部署复杂度较高。

---

## 63. Multimodal Learning on Low-Quality Data with Conformal Predictive Self-Calibration

**arXiv ID:** 2605.03820 | [PDF](https://arxiv.org/pdf/2605.03820v1)

**作者:** Xun Jiang `[一作]` (Tongji University), Xing Xu `[通讯]` (Tongji University)

**通讯引用:** 35869 | [OpenAlex ID](https://openalex.org/A5101679416)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于合成预测的自校准框架CPSC，统一解决多模态低质量数据中的模态不平衡与噪声问题；

**💡 创新点**

创新点在于将 conformal prediction 与特征层（RSC）和梯度层（GSC）自校准机制相结合，并在训练循环中动态更新 CP 以实现可自适应的自校准；

**🔧 技术方法**

主要技术包括合成预测（conformal prediction）、特征分解与多组件筛选、梯度重加权、动态 CP 更新、深度多模态编码器与融合网络；

**📊 数据集**

使用了六个基准数据集：CREMA‑D、AVE、Kinetics Sounds（模态不平衡场景）以及 SUN RGB‑D、NYU Depth V2、MVSA‑Single（噪声场景）；

**📈 对比分析**

与 ReconBoost、MMPareto、LFM、InfoReg、DGL、ARL、IPRM 等最新方法在所有实验中进行对比，CPSC 在模态不平衡和噪声环境下均取得最高或最优平均准确率，尤其在模态不平衡数据上提升约5%；

**⚠️ 局限性**

局限性包括对 CP 更新频率和预热阶段的依赖，极端噪声或极大模态差异时效果仍有限，并且相比传统方法增加了一定的计算和内存开销。

---

## 64. Pareto-type finite-block optimality for source codes: a constrained Markov example

**arXiv ID:** 2605.03552 | [PDF](https://arxiv.org/pdf/2605.03552v1)

**作者:** Stefano Della Fiore `[一作]` (University of Brescia), Stefano Della Fiore `[通讯]` (University of Brescia)

**通讯引用:** 47 | [OpenAlex ID](https://openalex.org/A5041359704)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种针对注入源编码的帕累托型有限块最优性概念，通过期望块长度的完整序列比较两个编码。

**💡 创新点**

提出了一种规范的注入二进制映射，并证明了在特定源下，达赖-莱昂纳迪编码在有限块平均长度方面不是帕累托最优的。

**🔧 技术方法**

使用了组合计数和信息成本的精确枚举技术，结合短字典顺序的编码方法。

**📊 数据集**

使用了达赖和莱昂纳迪提出的四符号约束马尔可夫源作为数据集。

**📈 对比分析**

通过与达赖-莱昂纳迪编码的比较，证明了新编码在每个块长度n≥2时的平均长度更优，且在n=1时相等。

**⚠️ 局限性**

研究中未提及具体的局限性，但可能存在对更复杂源的适用性不足。

---

## 65. Are you with me? A Framework for Detecting Mental Model Discrepancies in Task-Based Team Dialogues

**arXiv ID:** 2605.03149 | [PDF](https://arxiv.org/pdf/2605.03149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 66. ARGUS: Defending LLM Agents Against Context-Aware Prompt Injection

**arXiv ID:** 2605.03378 | [PDF](https://arxiv.org/pdf/2605.03378v1)

**作者:** Shihao Weng `[一作]` (Nanjing University), Jia Liu `[通讯]` (Nanjing University)

**通讯引用:** 37074 | [OpenAlex ID](https://openalex.org/A5100409741)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了针对LLM代理的上下文依赖型Prompt Injection攻击的benchmark（AgentLure）和一种基于影响谱系图的决策审计防御（ARGUS），并在多领域、多向量任务上评估其安全与效用。

**💡 创新点**

创新点在于：①提出“上下文感知”的Prompt Injection基准，覆盖四个业务领域、八种攻击向量和六种攻击面；②设计SPAN‑level影响谱系图，结合ContentSegmenter、ArgumentGrounder、InvariantChecker与EntailmentVerifier四种LLM工具实现对状态变更调用的细粒度审计；③在保持高效的同时显著降低攻击成功率并减少拒绝率。

**🔧 技术方法**

使用技术包括：影响谱系图（IPG）数据结构；SPAN‑level内容分割（ContentSegmenter）；参数化的参数来源追踪（ArgumentGrounder）；任务约束生成与验证（InvariantChecker）；证据充分性检验（EntailmentVerifier）；LLM（GPT‑4o‑mini）辅助工具；以及对调用链的runtime中间层拦截实现。

**📊 数据集**

使用AgentLure benchmark，共320条样本，涵盖Banking、Travel、Workspace、Slack四个领域，每个领域10个上下文相关任务，按八个攻击向量与六个攻击面构造。

**📈 对比分析**

与八种现有文本级与执行级防御在同一基准下对比，指标包括ASR、Worst‑vec ASR、Utility（无攻击/有攻击）、Refusal、Token成本、综合安全‑效用得分（EDS）。ARGUS将ASR从28.8%降至3.8%（Worst‑vec 7.5%），保留87.5%无攻击任务效用，综合EDS 84.2%，在所有向量上保持稳定；在白盒自适应攻击下ASR提升至5.9%，仍低于基线。

**⚠️ 局限性**

局限性：假设任务载体合法，无法抵御完全伪造的载体；主要针对Prompt Injection，未覆盖更广泛的Agent安全风险；依赖LLM推理，模型规模与成本对部署可行性有影响。

---

## 67. Graph Neural Network based Hierarchy-Aware Embeddings of Knowledge Graphs: Applications to Yeast Phenotype Prediction

**arXiv ID:** 2605.03690 | [PDF](https://arxiv.org/pdf/2605.03690v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 68. Benchmarking Local Language Models for Social Robots using Edge Devices

**arXiv ID:** 2605.03111 | [PDF](https://arxiv.org/pdf/2605.03111v1)

**作者:** Dorian Lamouille `[一作]` (University of Tartu), Karl Kruusamäe `[通讯]` (University of Tartu)

**通讯引用:** 685 | [OpenAlex ID](https://openalex.org/A5055075752)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对25种开源语言模型在Raspberry Pi等边缘设备上的本地部署进行了系统基准测试，评估了推理效率、通用知识（MMLU子集）和教学效果（LLM与人类评审一致），并提出了三层本地推理架构；

**💡 创新点**

创新点在于：①首次将硬件效率、知识准确度与教学有效性三维度统一评估；②验证LLM-as-judge与人工评审的一致性；③设计可在资源受限的教育机器人中实现的多层本地推理架构；

**🔧 技术方法**

使用了Raspberry Pi 4/5、笔记本GPU的Ollama推理框架、Q4_K_M量化模型、DeepEval进行MMLU子集评估、GPT-4o-mini进行教学评分，并结合人类评审；

**📊 数据集**

采用了六类MMLU子集（Formal Logic、Global Facts、College CS、College Math、Marketing、High School Macroeconomics）以及10个教学问题来测评；

**📈 对比分析**

方法通过对TPS、TPJ、MMLU准确率与教学得分进行多维度比较，发现模型在推理速度、能效和知识水平之间存在显著折衷；如Granite4 Tiny Hybrid 7B在TPS≈2.5、TPJ≈0.9、MMLU≈54.6%同时保持较高教学得分，成为最佳权衡模型；

**⚠️ 局限性**

局限性包括：单次运行测量缺乏方差估计、TPJ估计基于线性电流模型、MMLU子集覆盖有限、LLM评估可能存在偏差、硬件热负载影响实际性能以及多模型部署对存储空间的压力。

---

## 69. Human-in-the-Loop Uncertainty Analysis in Self-Adaptive Robots Using LLMs

**arXiv ID:** 2605.02983 | [PDF](https://arxiv.org/pdf/2605.02983v1)

**作者:** Hassan Sartaj `[一作]` (Simula Research Laboratory), Peter Gorm Larsen `[通讯]` (Aarhus University)

**通讯引用:** 5096 | [OpenAlex ID](https://openalex.org/A5037561273)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种人机协同、基于大型语言模型的自适应机器人不确定性分析方法和工具，结合不确定性分类体系实现了设计阶段的系统化不确定性探索。

**💡 创新点**

创新点在于构建了专门针对自适应机器人的12维度不确定性分类体系，并将高级提示工程与迭代细化（排名、案例、分类指导）相结合，实现了对LLM输出的可控、可重复改进。

**🔧 技术方法**

使用的核心技术包括大型语言模型（Gemini 2.5 Flash）、结构化提示模板、角色提示、评分提示、例子提示、分类导向提示，以及Web前端与后端的交互式工具。

**📊 数据集**

实验数据来源于16名从事自主移动机器人、工业拆解机器人、协作制造机器人和自主船舶等四类真实工业案例的从业者，结合他们提供的需求文档与交互日志。

**📈 对比分析**

通过用户体验问卷和交互日志分析，工具在“结构化提示”与“迭代细化”两大功能上获得了平均4.25/5的高评分，整体可用性与易懂性分别为3.75/5与3.88/5，表明方法在实际工程场景中具备良好的可操作性和用户接受度。

**⚠️ 局限性**

局限性包括样本规模仅为16人、评估时间短、LLM响应可变性导致部分细化方法效果不稳定，以及对分类体系的熟悉程度影响了工具的使用效率。

---

## 70. From prompting to evidence-based translation: A RAG+prompt system for Japanese-Chinese translation and its pedagogical potential

**arXiv ID:** 2605.03387 | [PDF](https://arxiv.org/pdf/2605.03387v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 71. Evaluating Reasoning Models for Queries with Presuppositions

**arXiv ID:** 2605.03050 | [PDF](https://arxiv.org/pdf/2605.03050v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 72. FIBER: A Differentially Private Optimizer with Filter-Aware Innovation Bias Correction

**arXiv ID:** 2605.03425 | [PDF](https://arxiv.org/pdf/2605.03425v1)

**作者:** Duc Dm `[一作]` (Korea Advanced Institute of Science and Technology), Huy Nguyen `[通讯]` (Northeastern University)

**通讯引用:** 31980 | [OpenAlex ID](https://openalex.org/A5001558226)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 FiBeR，一种专门针对已通过时序滤波的差分隐私梯度的自适应优化器，能在保留隐私的同时显著提升训练性能。

**💡 创新点**

创新点包括①在创新（残差）空间进行滤波并积分得到去噪梯度；②将两点梯度构造与滤波增益解耦，形成更灵活的超参数空间；③推导滤波后 DP 噪声对 AdamW 二阶矩的衰减因子 A(ω)，并在优化器中加入闭式校正，消除因滤波导致的偏差。

**🔧 技术方法**

使用了 DP-SGD + 高斯机制、两点梯度构造、指数移动平均/残差滤波、二阶递推、滤波噪声方差衰减分析、滤波感知的 AdamW 校正、RDP 隐私核以及实验评估框架。

**📊 数据集**

视觉任务：MNIST、CIFAR‑10/100、ImageNet‑1k；语言任务：GLUE（MNLI、QNLI、SST‑2、QQP）和 E2E 文本生成；模型分别为 CNN5、WRN、ViT‑small、RoBERTa、GPT‑2‑small。

**📈 对比分析**

与 DP‑AdamW、DiSK、DiSK‑CORR、DOPPLER、MF‑DP‑FTRL 等基线在相同 ε、δ 隐私预算下进行网格搜索对比。FiBeR 在所有任务中均获得更高准确率，尤其在低 ε（高隐私）和长时间训练中提升明显，优于其他滤波或校正方法。

**⚠️ 局限性**

局限性包括：需要两点梯度导致每步两次反向，计算成本略高；滤波参数仍需手动调优；假设噪声与信号独立，跨项误差较小；在高 ε（低噪声）场景收益降低；理论校正主要针对稳态，初期可能保守；尚未在更广泛的模型/任务上验证。

---

## 73. FACTOR: Counterfactual Training-Free Test-Time Adaptation for Open-Vocabulary Object Detection

**arXiv ID:** 2605.03294 | [PDF](https://arxiv.org/pdf/2605.03294v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 74. A Hierarchical Sampling Framework for bounding the Generalization Error of Federated Learning

**arXiv ID:** 2605.03499 | [PDF](https://arxiv.org/pdf/2605.03499v1)

**作者:** Dario Filatrella `[一作]` (Royal Institute of Technology), Mikael Skoglund `[通讯]` (Royal Institute of Technology)

**通讯引用:** 8954 | [OpenAlex ID](https://openalex.org/A5041348422)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

提出了一种层级采样框架，用于分析层级联邦学习的泛化误差；

**💡 创新点**

在 Wasserstein 距离下给出更通用、更紧的泛化误差上界，并能推导出现有 CMI 上界与差分隐私下界；

**🔧 技术方法**

采用层级采样树、超样本构造、Lipschitz 损失、Wasserstein 距离与信息论技术；

**📊 数据集**

主要为理论分析，未使用公开数据集，而是以高斯位置模型作案例验证；

**📈 对比分析**

通过与高斯位置模型的真实泛化误差比较，证明上界在样本量规模上表现良好，但在树深度扩展时会出现误差放大；

**⚠️ 局限性**

局限在未考虑通信/计算约束，对树深度的泛化误差上界不够紧，且缺乏对非均匀树结构的扩展。

---

## 75. The Right Answer, the Wrong Direction: Why Transformers Fail at Counting and How to Fix It

**arXiv ID:** 2605.03258 | [PDF](https://arxiv.org/pdf/2605.03258v1)

**作者:** Gabriel Garcia `[一作]` `[通讯]` (Independent Researcher), Gabriel Garcia (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在计数任务中表现不佳的原因，发现是输出头与内部计数表示之间的几何对齐问题；并通过 9 行修复、LoRA Q/V、DPS 等干预手段验证并改善这一瓶颈。

**💡 创新点**

提出“几何读出瓶颈”理论：计数信息被准确编码，但其编码方向与数字输出行几乎正交；同时展示了仅修复数字行即可显著提升约束下的计数准确率，并在全词表推理下通过 LoRA Q/V 改善注意力路由实现更高性能。

**🔧 技术方法**

使用线性探针、Logit‑Lens 以及主动干预技术（数字行微调、LoRA Q/V、DPS）对模型内部表示和输出头进行分析与修改；还使用梯度与余弦相似度评估方向对齐。

**📊 数据集**

在人工合成计数基准（实体计数、字符计数、加法、列表长度）以及自然语言计数（8 种实体、8 种模板）上进行评估，并在 MMLU、GSM8K、DROP 等多步骤推理任务上做负向验证。

**📈 对比分析**

与基线（10–24%）相比，9 行修复在约束下提升至 60–100%（最高 100%），LoRA Q/V 在无约束自回归生成中达到 83%±7.2%；DPS 在约束下可逼近探针上限（≈97%）。

**⚠️ 局限性**

局限性：仅在低词表聚合任务（计数、取最大等）适用，需在约束下使用；对无约束生成需额外注意力路由修正；不适用于多步骤推理或开放式生成；模型规模越大对齐越差，需更大修正。

---

## 76. BifrostUMI: Bridging Robot-Free Demonstrations and Humanoid Whole-Body Manipulation

**arXiv ID:** 2605.03452 | [PDF](https://arxiv.org/pdf/2605.03452v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 77. Identity-Consistent Multi-Pose Generation of Contactless Fingerprints

**arXiv ID:** 2605.03830 | [PDF](https://arxiv.org/pdf/2605.03830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 78. MedStruct-S: A Benchmark for Key Discovery, Key-Conditioned QA and Semi-Structured Extraction from OCR Clinical Reports

**arXiv ID:** 2605.03103 | [PDF](https://arxiv.org/pdf/2605.03103v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 79. MEMSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning in Retrieval-Augmented Agents

**arXiv ID:** 2605.03482 | [PDF](https://arxiv.org/pdf/2605.03482v1)

**作者:** Ishrith Gowda `[一作]` `[通讯]` (University of California), Ishrith Gowda (University of California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种针对持续性内存注入攻击的检测方法MemSAD，并给出了攻击模型的正式游戏框架；

**💡 创新点**

其创新点在于通过梯度耦合定理将检索得分与异常分数的梯度对齐，构造出可证实的检测半径，并证明该方法在样本量上达到Le Cam下界；

**🔧 技术方法**

采用了基于密度向量的余弦相似度检索、Roll‑in 预过滤、梯度耦合定理、Le Cam最优性证明以及在线窗口调节；

**📊 数据集**

实验使用了1,000条人工合成的多类内存条目（包括日程、偏好、事实等）以及融合自然问题(Natural Questions)语料的混合数据集；

**📈 对比分析**

与四种单体防御（Watermark、Validation、Proactive）及现有RAG防御做对比，在3×5攻击‑防御矩阵中，复合防御组合在触发查询协议下实现了100%检测率、0%误报，且在非触发攻击中也显著提升；

**⚠️ 局限性**

主要局限包括：实验基于合成数据，真实世界中语义变形或多模态内容可能导致检测失效；同义词替换在低嵌入扰动下能绕过梯度耦合；以及对动态漂移假设的在线窗口选择依赖较慢变化的环境。

---

## 80. Exponential-Size Circuit Complexity is Comeager in Symmetric Exponential Time

**arXiv ID:** 2605.03306 | [PDF](https://arxiv.org/pdf/2605.03306v1)

**作者:** John M. Hitchcock `[一作]` (University of Wyoming), John M. Hitchcock `[通讯]` (University of Wyoming)

**通讯引用:** 625 | [OpenAlex ID](https://openalex.org/A5077270481)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

在对称指数时间类中证明了需要指数规模电路的语言在资源受限的Baire范畴中占主导地位（即comeager）

**💡 创新点**

将资源受限范畴推广到对称指数时间，并利用Li的单值exponential‑alternation算法构造了Banach‑Mazur游戏中的获胜策略

**🔧 技术方法**

使用Banach‑Mazur游戏、资源受限Baire范畴、单值exponential‑alternation算法以及对称指数时间类的定义

**📊 数据集**

无数据集，纯理论证明

**📈 对比分析**

无实验比较，理论上证明该类语言的典型性

**⚠️ 局限性**

结果仅适用于对称指数时间类，尚未扩展到更小的层级或资源受限测度下

---

## 81. Finite-Size Gradient Transport in Large Language Model Pretraining: From Cascade Size to Intensive Transport Efficiency

**arXiv ID:** 2605.02968 | [PDF](https://arxiv.org/pdf/2605.02968v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 82. ISAAC: Auditing Causal Reasoning in Deep Models for Drug-Target Interaction

**arXiv ID:** 2605.02962 | [PDF](https://arxiv.org/pdf/2605.02962v1)

**作者:** Barbara Tarantino `[一作]` (University of Pavia), Paolo Giudici `[通讯]` (University of Pavia)

**通讯引用:** 7786 | [OpenAlex ID](https://openalex.org/A5051364218)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出ISAAC框架，对已训练的深度模型进行结构化干预后期审计，以评估其是否基于机理性结构进行推理

**💡 创新点**

首次在深度学习模型的后期阶段通过对比与结构先验对齐与不对齐的干预，量化模型的因果推理能力

**🔧 技术方法**

基于输入级结构干预、干预响应差异计算、Reasoning Score、分离系数和重叠率等指标

**📊 数据集**

Davis kinase抑制剂-靶点交互数据集（含KLIFS pocket注释）

**📈 对比分析**

在保持相近AUROC（约3%差距）的前提下，对DeepDTA、DeepConvDTI和TAPB三种架构分别进行干预，发现TAPB在RS、C_sep上显著优于其他两者，表明在相同预测水平下推理结构有明显差异

**⚠️ 局限性**

干预仅针对靶点序列，缺少药物侧干预；结构先验基于KLIFS，可能无法覆盖所有影响力位点；干预空间与几何匹配不完全，对比的对齐度与空间分布差异未完全消除

---

## 83. Geolocating News about Extreme Climate Events: A Comparative Analysis of Off-the-Shelf Tools for Toponym Identification in German

**arXiv ID:** 2605.03414 | [PDF](https://arxiv.org/pdf/2605.03414v1)

**作者:** Brielen Madureira `[一作]` (Leipzig University), Andreas Niekler `[通讯]` (Leipzig University)

**通讯引用:** 1139 | [OpenAlex ID](https://openalex.org/A5058667349)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对三种德国NER工具（Flair、SpaCy、Stanza）在气候灾害新闻文本中的拓扑命名识别进行比较，并评估其对文档级地理定位和国家排名的影响。

**💡 创新点**

采用离散化的外推评估方法，将NER输出直接用于文档级地理定位和国家频率排名，揭示工具差异对下游结论的显著影响。

**🔧 技术方法**

使用 Flair、SpaCy、Stanza NER 模型，结合 Geonames 与 Nominatim 地理数据库，以及三种投票/几何投票/关键词邻近的预测策略。

**📊 数据集**

983 篇 2000-2024 年德国报纸灾害与极端气候事件的新闻文本，人工标注事件类型与事件发生国。

**📈 对比分析**

通过交叉比较工具间的词汇重叠（IoU）、对文档地理焦点的精确/部分匹配率以及国家频率排名的 Spearman/Kendall 相关系数进行比较，结果显示 Flair 最高，工具差异导致预测准确率差距可达 5 个百分点，国家排名相关性最高。

**⚠️ 局限性**

缺乏金标准地名注释、使用简单启发式预测方法、Geo 数据库覆盖有限、NER 只标注单一地点标签，可能引入关联地名误差。

---

## 84. PDSL: Propagation Dynamics Aware Framework for Source Localization

**arXiv ID:** 2605.03550 | [PDF](https://arxiv.org/pdf/2605.03550v1)

**作者:** Yansong Wang `[一作]` (Southwest University), Tao Jia `[通讯]` (Southwest University)

**通讯引用:** 3074 | [OpenAlex ID](https://openalex.org/A5019949140)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一个基于传播动态感知的源定位框架PDSL，利用深度生成模型和图神经ODE联合推断信息传播源。

**💡 创新点**

创新点在于将传播动态与生成模型融合，显式建模传播过程的不确定性，并用图神经ODE连续模拟扩散演化。

**🔧 技术方法**

采用条件变分自编码器、图神经ODE、匹配机制以及生成式源分布推断等技术。

**📊 数据集**

在四个真实网络（Jazz、Network Science、CollegeMsg、Cora-ML）以及自生成的BA网络和两大真实传播数据集Twitter、Douban上进行实验。

**📈 对比分析**

与八种基线（Netsleuth、LPSI、OJC、GCNSI、IVGD、SLVAE、BOSouL、SDSA）比较，PDSL在SI、GLT、SIR三种传播机制下的宏F1和准确率均优于对手，尤其在真实数据上表现突出。

**⚠️ 局限性**

局限性包括对训练样本质量的依赖、需要预先收集多时刻快照或对匹配策略的计算开销。

---

## 85. ADAPTS: Agentic Decomposition for Automated Protocol-agnostic Tracking of Symptoms

**arXiv ID:** 2605.03212 | [PDF](https://arxiv.org/pdf/2605.03212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 86. Neuron-Anchored Rule Extraction for Large Language Models via Contrastive Hierarchical Ablation

**arXiv ID:** 2605.03058 | [PDF](https://arxiv.org/pdf/2605.03058v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 87. Geographic Variation in Stack Overflow Code Quality: Evidence from a Cross-Regional Study of Coding Practices

**arXiv ID:** 2605.03670 | [PDF](https://arxiv.org/pdf/2605.03670v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 88. Exact and Approximate Algorithms for Polytree Learning

**arXiv ID:** 2605.03622 | [PDF](https://arxiv.org/pdf/2605.03622v1)

**作者:** Juha Harviainen `[一作]` (University of Helsinki), Manuel Sorge `[通讯]` (TU Wien)

**通讯引用:** 660 | [OpenAlex ID](https://openalex.org/A5033440939)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究多项式树（Polytree）结构学习的最优算法与近似算法，并给出了在常数入度约束下的高效动态规划求解方法，以及若干近似比率的多项式时间算法。

**💡 创新点**

创新点在于：① 推导出一种只需  (2+ϵ)^n  时间即可求解入度上限为 k 的最优多项式树；② 在不同约束（入度、可加得分、分量大小）下提供了最优或几乎最优的多项式时间近似算法；③ 通过 Set Cover/Set Partitioning 等难度假设给出时间/近似因子的下界，表明所提出算法的时间/性能已接近最优。

**🔧 技术方法**

主要技术包括：高度优化的动态规划（利用 DFS 生成的节点序列控制状态空间）、父集剪枝、递归归纳证明、归约证明（到 Set Cover/Set Partitioning）以及基于可加分数的贪心算法。

**📊 数据集**

论文为理论研究，没有使用实测数据集，所有结果均为算法复杂度和近似比率的证明。

**📈 对比分析**

与先前已知的 3^n 复杂度算法相比，新算法将指数基数从 3 降到 2+ϵ，且在入度受限时得到 (k+1)-近似（在一般分数下）以及 2 近似（可加分数），与已知下界相匹配；在分量大小约束下得到 2q 近似，同样与 UG 下界一致。

**⚠️ 局限性**

局限性包括：① 对于一般（无入度限制）多项式树仍需指数时间；② 近似算法仅适用于特定的分数或约束，且在最一般情形下仍无法突破 P=NP 的界限；③ 结果主要依赖于 Set Cover/UG 等难度假设，实际实现的常数和实际性能未给出。

---

## 89. Boundary-Aware Uncertainty Quantification for Wildfire Spread Prediction

**arXiv ID:** 2605.03148 | [PDF](https://arxiv.org/pdf/2605.03148v1)

**作者:** Jonas V. Funk `[一作]` `[通讯]` (Independent Research), Jonas V. Funk (Independent Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在 WildfireSpreadTS 数据集上使用 FCER 框架对火灾蔓延预测模型的不确定性进行空间感知评估，并比较了深度集成与蒸馏单模型的表现。

**💡 创新点**

创新点在于提出 Fire‑Centered Evaluation Region (FCER) 用于边界敏感的 UQ 评估，并将 DUDES 蒸馏方法改造为单通道模型以实现实时单前向推理。

**🔧 技术方法**

使用 U‑Net+Temporal Attention Encoder (UTAE) 作为骨干，结合 Deep Ensemble 和 DUDES 蒸馏技术。

**📊 数据集**

数据集为 WildfireSpreadTS，包含 2018‑2021 年的 375 m 分辨率火灾蔓延序列。

**📈 对比分析**

通过在 FCER 区域内计算 AUROC、AUPRC、Brier、NLL 等指标比较，两者在分割和校准上相近，但 DUDES 在大于 750 m 的 FCER 范围内在不确定性排序上显著优于集成。

**⚠️ 局限性**

局限性包括仅使用 n=3 的集成、固定 128×128 中心裁剪、骨干被冻结以及仅在 WildfireSpreadTS 上验证，缺乏对更大模型、不同 ASD 影响的分析。

---

## 90. Single-Period Portfolio Selection via Information Projection

**arXiv ID:** 2605.03184 | [PDF](https://arxiv.org/pdf/2605.03184v1)

**作者:** Bo-Yu Yang `[一作]` (Ecole Polytechnique Federale de Lausanne), Michael Gastpar `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了单周期CRRA效用下的资产组合选择问题，并将其转化为Renyi信息投影，给出了CE增长率的分解以及相应的交替优化算法（Info‑Proj EG）

**💡 创新点**

创新点在于：①揭示风险厌恶系数即Renyi阶数，使CRRA组合选择等价于Renyi投影；②提供Blahut‑Arimoto式的交替优化，并得到闭式辅助更新；③在低风险厌恶区间内实现更快的收敛

**🔧 技术方法**

采用了信息论工具（Renyi散度、Renyi熵）、变分表示、Blahut‑Arimoto算法、指数梯度（EG）与Armijo退火、凸优化与KL投影等技术

**📊 数据集**

实验使用模拟数据：随机生成(k=100, m=50)支付矩阵，状态概率取自Dirichlet(10·1_k)，未使用真实金融数据

**📈 对比分析**

与Naive EG（直接最小化CRRA等价损失）和Cover的乘法更新（ρ=1）对比，结果显示在ρ=0.5和1.5时Info‑Proj EG迭代次数更少、误差更小；ρ=1时两者表现一致

**⚠️ 局限性**

局限性：仅适用于有限支持且低风险厌恶(ρ<2)的情形；未考虑市场估计误差、动态多期模型；对高风险厌恶或非对称分布缺乏理论与实验支持

---

## 91. OGPO: Sample Efficient Full-Finetuning of Generative Control Policies

**arXiv ID:** 2605.03065 | [PDF](https://arxiv.org/pdf/2605.03065v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 92. Disentangling Shared and Task-Specific Representations from Multi-Modal Clinical Data

**arXiv ID:** 2605.03570 | [PDF](https://arxiv.org/pdf/2605.03570v1)

**作者:** He Lyu `[一作]` (Sichuan University), Huan Song `[通讯]` (Sichuan University)

**通讯引用:** 6051 | [OpenAlex ID](https://openalex.org/A5043778832)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在临床多模态数据（表格、文本、生命体征）上，构建了统一Transformer融合框架，并提出Orthogonal Task Decomposition（OrthTD）来实现共享与任务特定表示的显式分离，用正交约束降低冗余，以提升多任务预测性能。

**💡 创新点**

创新点在于：① 将多模态信息统一映射到Transformer后通过可学习全局token实现全局聚合；② 设计任务解耦模块，将融合表示分成共享子空间与每个任务的特定子空间；③ 引入几何正交正则化，强制共享与特定子空间正交，从而避免负迁移与信息重叠；④ 在临床多任务预测场景中首次结合上述三者并验证其有效性。

**🔧 技术方法**

核心技术包括：统一Transformer编码器、BERT文本编码（仅微调最后一层）、任务解耦投影层、LayerNorm+GELU、正交正则化、异步损失（asymmetric loss）与交叉熵组合、AdamW优化器、余弦学习率调度、批量大小128。

**📊 数据集**

使用中国外科与麻醉队列（CSAC）中西川医院12,430例外科患者的数据，包含结构化表格变量、临床文本记录以及预手术/术中生理数据，目标为七天内四个并发症（任何并发症、肺部并发症、急性肾损伤、非计划ICU入住）。

**📈 对比分析**

与传统机器学习基线（XGBoost、LightGBM）、现代表格深度学习（FT‑Transformer、TabPFN）、简单多模态拼接（MLP+Text、Transformer+Text）以及多任务学习典型策略（硬共享、不确定性加权、Cross‑Stitch、MMoE）进行比较。OrthTD在AUC上达到87.5%，在AUPRC上达到37.2%，分别优于最强基线（AUC 87.1%，AUPRC 32.9）和所有多任务对照组，尤其在稀疏事件（AKI、ICU入住）上显著提升。

**⚠️ 局限性**

局限性包括：① 仅使用表格和文本两种模态，未评估图像、波形等其他模态的影响；② 对缺失模态的鲁棒性尚未系统研究；③ 正交约束权重λ的取值需经验调优，缺乏理论指导；④ 实验仅在单中心数据集上进行，外部验证与泛化能力待进一步验证；⑤ 只关注术后七天并发症，无法直接推广到更长期或不同临床任务。

---

## 93. On Surprising Effects of Risk-Aware Domain Randomization for Contact-Rich Sampling-based Predictive Control

**arXiv ID:** 2605.03290 | [PDF](https://arxiv.org/pdf/2605.03290v1)

**作者:** Sergio A. Esteban `[一作]` (California Institute of Technology), Aaron D. Ames `[通讯]` (California Institute of Technology)

**通讯引用:** 15207 | [OpenAlex ID](https://openalex.org/A5039171820)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在预测性采样控制（SPC）中研究风险感知域随机化，比较平均、乐观、悲观三种聚合策略以提升接触丰富任务的鲁棒性。

**💡 创新点**

发现乐观聚合能显著提升性能，并揭示域随机化通过改变搜索空间的“盆地形状”影响采样优化，突破了以往对域随机化仅关注鲁棒性的认识。

**🔧 技术方法**

采用采样预测控制（Predictive Sampling）、GPU并行仿真框架Hydrax和MuJoCo MJX实现高速并行仿真与采样。

**📊 数据集**

使用Push‑T仿真任务，采集多种随机参数（摩擦、接触时间、质量、驱动力增益）下的控制序列和成本数据。

**📈 对比分析**

在不同R（0、4、16、32、64）随机域下对三种聚合策略进行对比，通过平均总成本和块位置误差评估，乐观策略在R≈16时表现最佳。

**⚠️ 局限性**

实验仅限单一任务与单一采样算法，缺乏理论解析及对更复杂或真实系统的推广，结果受限于仿真环境的简化假设。

---

## 94. Visibility Queries in Simple Polygons

**arXiv ID:** 2605.03334 | [PDF](https://arxiv.org/pdf/2605.03334v1)

**作者:** Sujoy Bhore `[一作]` (Indian Institute of Technology Bombay), Jie Xue `[通讯]` (New York University Shanghai)

**通讯引用:** 1425 | [OpenAlex ID](https://openalex.org/A5101635746)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文研究了简单多边形P的可见性查询问题，提出了一种新的数据结构，可以在O(log n + k)的查询时间内计算查询点q在P中的可见性多边形，且空间复杂度为O(n^2 + ϵ)，其中k是可见性多边形的大小。

**💡 创新点**

创新点在于提出了一种新的多边形分解方法，利用该方法构建的数据结构在空间和查询时间上均优于现有的最佳结果，尤其是在空间复杂度上显著降低。

**🔧 技术方法**

使用了一种新的多边形分解技术，结合了平衡分解和几何三角形的概念，构建了高效的数据结构以支持可见性查询。

**📊 数据集**

使用的数据集是简单多边形P，具有n个顶点，具体的多边形结构和顶点数量未在摘要中详细说明。

**📈 对比分析**

与之前的方法相比，本文的方法在O(n^2)空间下将查询时间从O(log^2 n + k)改进为O(log n loglog n + k)，在O(n)空间下支持O(n^1/2 + k)的查询时间，性能显著提升。

**⚠️ 局限性**

限制在于当空间复杂度低于O(n^2)时，查询时间的提升可能会受到影响，且在某些特殊情况下，数据结构的构建和查询效率可能会受到多边形复杂度的影响。

---

## 95. Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts

**arXiv ID:** 2605.03348 | [PDF](https://arxiv.org/pdf/2605.03348v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 96. NucEval: A Robust Evaluation Framework for Nuclear Instance Segmentation

**arXiv ID:** 2605.03144 | [PDF](https://arxiv.org/pdf/2605.03144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 97. What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity

**arXiv ID:** 2605.03782 | [PDF](https://arxiv.org/pdf/2605.03782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 98. Backtrackable Inprocessing

**arXiv ID:** 2605.03654 | [PDF](https://arxiv.org/pdf/2605.03654v1)

**作者:** Alexander Nadel `[一作]` (Technion), Alexander Nadel `[通讯]` (Technion)

**通讯引用:** 684 | [OpenAlex ID](https://openalex.org/A5006830161)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Backtrackable Inprocessing (BI)框架，使增量SAT求解中可在任何决策层级、任何时点进行inprocessing并保持回溯安全；

**💡 创新点**

突破传统inprocessing只能在全局层级0进行的限制，引入stashed和deputy句子管理保证在回溯时能够安全恢复；

**🔧 技术方法**

实现了子句消除（subsumption）、自子句消除（self-subsuming resolution）和有界变量消除（BVE），并在Island SAT求解器中实现BI；

**📊 数据集**

在Hardware Model Checking Competition 2017（298个未被排除的硬件模型）生成的BMC（Bounded Model Checking）基准，最多展开到100层；

**📈 对比分析**

与传统全局增量inprocessing baseline相比，BI在一小时限时下多解了约1.5倍困难边界，最优策略在所有实验中比baseline多解决197个边界；

**⚠️ 局限性**

局限性在于实验仅限于BMC基准且仅实现了三种核心inprocessing技术，未评估在非BMC或更大规模实例上的效果；

---

## 99. Geometric Deviation as an Unsupervised Pre-Generation Reliability Signal: Probing LLM Representations for Answerability

**arXiv ID:** 2605.03196 | [PDF](https://arxiv.org/pdf/2605.03196v1)

**作者:** Yucheng Du `[一作]` `[通讯]` (University of Southern California), Yucheng Du (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM内部表示几何距离能否在生成前预警回答不可答问题；

**💡 创新点**

提出不需要标签或输出采样的无监督几何偏差信号，证明其在结构性不可答的数学和代码问题上有效；

**🔧 技术方法**

使用对比匹配对的余弦距离、均值中心化、层级表示抽取、ROC‑AUC和F1评估；

**📊 数据集**

构造了三类提示形式（Math、Fact、Code）的匹配对数据集，共计 Math 50对、Fact 10对、Code 30对；

**📈 对比分析**

与拒绝关键词检测和自一致性多样本方法比较，几何信号在 Math 上达到 0.78–0.84 的 ROC‑AUC，明显优于拒绝基线（0.63–0.73）和自一致性（0.30–0.62）；在 Fact 上效果不显著，Code 上效果大但显著性混合；

**⚠️ 局限性**

局限包括样本规模小、仅在结构性不可答问题表现良好、未探索更复杂的探测器或更大数据集、仅评估 7–8B 模型，且未验证跨模型/对齐版本的普适性。

---

## 100. AniMatrix: An Anime Video Generation Model that Thinks in Art, Not Physics

**arXiv ID:** 2605.03652 | [PDF](https://arxiv.org/pdf/2605.03652v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 101. Relation Reasoning with LLMs in Expensive Optimization

**arXiv ID:** 2605.02933 | [PDF](https://arxiv.org/pdf/2605.02933v1)

**作者:** Ye Lu `[一作]` (Shanghai Institute of Al for Education), Hao Hao `[通讯]` (Shanghai Institute of Al for Education)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于强化学习训练的关系推理LLM代理，集成到昂贵优化的代理进化算法R2SAEA中，用以在不进行每代重新训练的情况下完成候选解的比较与筛选。

**💡 创新点**

创新点在于：①将关系建模转化为在上下文中的对比推理任务；②提出锚点迭代式上下文构造，将Prompt规模从二次降至一次；③通过GRPO对小型LLM进行关系推理微调，构建零训练成本的代理；④利用投票聚合将二进制/三分类关系转换为可直接用于选择的绝对分数。

**🔧 技术方法**

核心技术包括：大型语言模型（Qwen2.5）、锚点迭代推理、投票聚合、Group Relative Policy Optimization (GRPO)、量化（4/8位）以实现边缘设备部署。

**📊 数据集**

实验使用公开的单目标测试集LZG、YLL，以及多目标测试集DTLZ、WFG、MaF；训练数据来自遗传算法在多个函数和进化阶段生成的轨迹样本。

**📈 对比分析**

与传统SAEAs、基于回归/分类的代理、通用LLM代理以及贝叶斯优化等基线进行30次独立实验；在单目标、双目标与多目标设置下，R2SAEA在有限评估预算（FEs=300）中平均排名第一，显著优于所有对比方法。

**⚠️ 局限性**

局限性：①需要预先收集并使用RL进行大规模训练；②锚点与投票机制在极高维或极大候选集时的推理效率下降；③对不同分布或尺度变化的适应性仍有限；④量化后模型对GPU/CPU性能依赖仍存在，尚未完全实现低功耗嵌入式实时优化。

---

## 102. Detecting Stealth Sycophancy in Mental-Health Dialogue with Dynamic Emotional Signature Graphs

**arXiv ID:** 2605.03472 | [PDF](https://arxiv.org/pdf/2605.03472v1)

**作者:** Tianze Han `[一作]` (Shenzhen MSU-BIT University), Yongming Lu `[通讯]` (Shenzhen MSU-BIT University)

**通讯引用:** 3331 | [OpenAlex ID](https://openalex.org/A5058842708)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种离线评估框架DESG，用结构化临床状态和有向图来判定对话是否存在隐蔽的“顺从性”，即模型表面同情却强化负面认知。

**💡 创新点**

创新点在于：①将LLM仅用作状态感知器，消除其最终判定的偏差；②构建临床状态的非对称几何距离（CDD）和情绪-认知图谱，实现方向敏感的风险评估；③在多领域数据上构建诊断性基准，展示结构化状态空间是主要判别因素。

**🔧 技术方法**

技术包括：基于LLM的语义、情感(valence‑arousal)与认知失衡（10种CBT失衡）三轨道的1548维临床状态向量；构建有向情绪签名图并用CDD进行异向度距离计算；多种内部变体（ConcatANN、DESG‑Deep等）以及与文本相似度、官方评估器的对比。

**📊 数据集**

使用由EmpatheticDialogues、ESConv、CRADLE‑Dialogue三源合成的3×1000交叉域基准（每域1000条窗口，标签分布500 harmful/300 productive/200 neutral）。

**📈 对比分析**

与直接LLM评判、BERTScore、Prometheus‑2、TRACT等基线相比，DESG‑Ensemble在600条测试窗口上宏观F1达0.9353，明显优于最强基线0.5972，且在三域均表现稳健。

**⚠️ 局限性**

局限包括：基准存在构造性痕迹（如词汇/来源泄露），与人工多数判断的匹配度仅中等；评估完全离线，不能替代临床诊断；依赖LLM状态提取器，提取器偏差未彻底排除。

---

## 103. Posterior-First Neural PDE Simulation: Inferring Hidden Problem State from a Single Field

**arXiv ID:** 2605.03247 | [PDF](https://arxiv.org/pdf/2605.03247v1)

**作者:** Wenshuo Wang `[一作]` (South China University of Technology), Fan Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 54623 | [OpenAlex ID](https://openalex.org/A5100403400)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出在单观测PDE模拟中，先推断隐藏问题状态的后验分布，再基于该后验进行预测。

**💡 创新点**

核心创新是将问题状态后验作为可重用中间对象，理论证明后验与贝叶斯决策、可学习性和避免歧义的三重优势。

**🔧 技术方法**

结合后验推断网络、合适的评分规则（对数分数/Brier）以及匹配的Rollout骨干（U‑Net/FNO/ConvLSTM/Transformer）实现。

**📊 数据集**

在自制的灰斯塔克反应扩散合成数据集和公开的PDEBench（Diffusion‑Reaction、Diffusion‑Sorption、Shallow‑Water、Incompressible Navier–Stokes）上进行实验。

**📈 对比分析**

相较于传统的场到场直接预测、点式潜变量、增强容量等方法，后验首选模型在合成数据上将未来NLL从1.18降至1.02，在PDEBench上nRMSE从0.175降至0.132，闭合约59%向oracle的误差差距。

**⚠️ 局限性**

局限在于仅使用可数化的监督细化标签，缺乏对连续隐状态的处理；后验仅以摘要形式传递给Rollout，未实现完整的混合模拟；实验依赖特定任务族和稳定性假设。

---

## 104. Cross-Slice Co-Location Risk-Aware SFC Provisioning in Multi-Slice LEO Satellite Networks

**arXiv ID:** 2605.03656 | [PDF](https://arxiv.org/pdf/2605.03656v1)

**作者:** Mohammed Mahyoub `[一作]`, Halim Yanikomeroglu `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种针对多切片LEO卫星网络的风险感知SFC部署框架，联合考虑跨切片共位风险、CPU资源占用与VNF迁移稳定性；

**💡 创新点**

创新点在于：①构建跨切片共位风险的乘法模型并给出精细与粗粒度两种可行性；②提出迁移稳定机制，区分不可避免的拓扑变动与可避免迁移；③设计三阶段混合优化器（预处理+模拟退火+分支定界）实现实时重优化；

**🔧 技术方法**

使用混合整数线性规划（MILP）作为优化模型，结合模拟退火进行热启动，随后进行分支定界精细求解；

**📊 数据集**

实验基于公开Walker‑Star 60颗卫星星座、5个地面切片、10名用户/切片、5种VNF类型，并使用NIST SP 800‑53风险映射及3GPP/ETSI标准参数；

**📈 对比分析**

与三种基线（资源最小化MILP、稳定性MILP、贪心最近卫星）对比，实验显示风险下降93%且避免迁移下降98%，平均CPU占用仅提升0.3%，求解时间从256s降至11s（+23×加速），满足60s时隙预算；

**⚠️ 局限性**

局限在于假设切片和E2E延迟预算静态，未考虑动态切片到来/离开，以及未建模排队/拥塞等网络延迟因素。

---

## 105. Adaptive graph-based algorithms for conditional anomaly detection and semi-supervised learning

**arXiv ID:** 2605.03495 | [PDF](https://arxiv.org/pdf/2605.03495v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 106. OracleProto: A Reproducible Framework for Benchmarking LLM Native Forecasting via Knowledge Cutoff and Temporal Masking

**arXiv ID:** 2605.03762 | [PDF](https://arxiv.org/pdf/2605.03762v1)

**作者:** Yiding Ma `[一作]` (Beijing University of Posts and Telecommunications), Linna Zhou `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 1225 | [OpenAlex ID](https://openalex.org/A5102027008)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OracleProto框架，将已完成的事件重新构造为可重复评估的LLM本土预测任务，并建立可审核的评估体系。

**💡 创新点**

创新点在于通过知识截止、工具层时间屏蔽和内容级泄漏检测三重边界，将过期预测数据转化为零泄漏的训练/评估样本，并实现可复现的评估流程。

**🔧 技术方法**

采用时间截断的检索过滤、辅助泄漏检测器、ReAct交互循环、离散答案归一化与分层评分，以及基于离散集合的精确评估指标。

**📊 数据集**

使用了从FutureX‑Past筛选的80道已决策问题，并结合手工零泄漏审核作为评估与训练样本。

**📈 对比分析**

在六款LLM上使用统一的截断、检索与预算设置进行三次采样，评估复合准确率、成本与稳定性，最高准确率约0.60，最低成本模型（Qwen3.5‑Flash）每次正确样本费用仅0.003美元，整体差距约10–16倍。

**⚠️ 局限性**

局限在于仍依赖外部检索与泄漏检测，泄漏率虽低于3%但非零；框架对模型知识截止公开信息准确性敏感，且仅覆盖离散答案任务，无法直接推广到连续预测或多模态场景。

---

## 107. Enwar 3.0: An Agentic Multi-Modal LLM Orchestrator for Situation-Aware Beamforming, Blockage Prediction, and Handover Management

**arXiv ID:** 2605.03215 | [PDF](https://arxiv.org/pdf/2605.03215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 108. Retrieving Floods without Floodlights: Topic Models as Binary Classifiers for Extreme Climate Events in German News

**arXiv ID:** 2605.03450 | [PDF](https://arxiv.org/pdf/2605.03450v1)

**作者:** Brielen Madureira `[一作]` (Helmholtz Centre for Environmental Research), Andreas Niekler `[通讯]` (Leipzig University)

**通讯引用:** 1139 | [OpenAlex ID](https://openalex.org/A5058667349)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对通过关键词检索得到的德国气候灾害新闻集合，采用主题模型、微调文本嵌入和LLM进行二分类，以提升检索的精确率。

**💡 创新点**

创新在于用无监督主题模型的词概率自动划分主题为相关/不相关，无需人工主题解释；并针对不同灾害分别训练，展示模型表现随灾害不同。

**🔧 技术方法**

使用LDA、NMF主题模型；SetFit微调文本嵌入；Mistral-3 14B LLM；以及多数投票集成。

**📊 数据集**

基于wiso-net新闻聚合数据库中 2000-2024 年的 2.44M 条德语新闻，手工标注 3,150 条作为黄金标准，按灾害类型拆分。

**📈 对比分析**

比较显示：LLM 召回率最高但精度最低；TM 在高精度下召回率低；微调嵌入平衡两者，众数投票取得最高 F1，且在多灾害上表现稳健。

**⚠️ 局限性**

局限包括：样本量小（每种灾害仅 100 条测试），标注不确定性，主题模型仅用有限超参搜索，规则过滤可能漏掉相关文档，且未对多灾害共现进行建模。

---

## 109. Analysis and Explainability of LLMs Via Evolutionary Methods

**arXiv ID:** 2605.02930 | [PDF](https://arxiv.org/pdf/2605.02930v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 110. Replacing Parameters with Preferences: Federated Alignment of Heterogeneous Vision-Language Models

**arXiv ID:** 2605.03426 | [PDF](https://arxiv.org/pdf/2605.03426v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 111. Induction rules for Transition Algebra

**arXiv ID:** 2605.03628 | [PDF](https://arxiv.org/pdf/2605.03628v1)

**作者:** Go Hashimoto `[一作]` (Kyushu University), Go Hashimoto `[通讯]` (Kyushu University)

**通讯引用:** 894 | [OpenAlex ID](https://openalex.org/A5068535758)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

构造了一种基于转移代数（Transition Algebra, TA）的紧凑序列推理系统，并通过引入归纳规则和Kleene代数语义实现了完备性与Craig插值定理的证明。

**💡 创新点**

在传统TA缺乏紧凑性的基础上，提出了利用归纳规则约束*规则的新系统，并将语义转向Kleene代数，从而实现了在可数签名下的完备性与插值性。

**🔧 技术方法**

采用了序列推理系统、归纳规则、Kleene代数语义、机构（institution）框架以及模型理论（如Henkin扩张）进行理论分析与证明。

**📊 数据集**

无实验数据集，本文完全为形式化逻辑与模型理论研究。

**📈 对比分析**

本工作未涉及实验或性能对比，重点在于理论证明与系统一致性，因此不存在性能评估。

**⚠️ 局限性**

局限性包括：未能在不计数签名或非可数签名下保持完备性；归纳规则的某些推理规则（如_R^±、_L^∓）是否可消除仍未解决；在归纳情形下的割消性（cut‑elimination）不一定成立。

---

## 112. Delay, Plateau, or Collapse: Evaluating the Impact of Systematic Verification Error on RLVR

**arXiv ID:** 2605.02909 | [PDF](https://arxiv.org/pdf/2605.02909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 113. From Synthesis to Clinical Assistance: A Strategy-Aware Agent Framework for Autism Intervention based on Real Clinical Dataset

**arXiv ID:** 2605.02916 | [PDF](https://arxiv.org/pdf/2605.02916v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 114. RLDX-1 Technical Report

**arXiv ID:** 2605.03269 | [PDF](https://arxiv.org/pdf/2605.03269v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 115. Synthetic Data Generation for Long-Tail Medical Image Classification: A Case Study in Skin Lesions

**arXiv ID:** 2605.03221 | [PDF](https://arxiv.org/pdf/2605.03221v1)

**作者:** Jiaxiang Jiang `[一作]` (Intel), Omesh Tickoo `[通讯]` (Intel)

**通讯引用:** 1900 | [OpenAlex ID](https://openalex.org/A5003931059)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于扩散模型的合成数据增强管线，用inpainting diffusion模型生成合成医学图像并通过OOD过滤得到干净样本，提升长尾医学分类性能。

**💡 创新点**

创新点包括：① 为医学图像设计的新型inpainting diffusion模型架构；② 结合OOD后选择机制保证合成样本真实性；③ 采用自监督异常检测进行样本加权，平衡各类样本的贡献；④ 将数据生成与分类器解耦，形成通用数据增强流程。

**🔧 技术方法**

使用技术包括扩散模型（UNet+VAE编码的inpainting diffusion）、类嵌入网络、OOD检测方法、样本加权/过采样、自监督异常检测、5折交叉验证以及常用评估指标（BMA、F1、敏感度、特异度）。

**📊 数据集**

实验基于ISIC2019皮肤病变分类数据集（8类、样本极度不平衡），对DF、VASC等尾部类尤为关注。

**📈 对比分析**

与当前SOTA方法（如MRE、CNNE、DataFuse、MME、CBDM、Dreambooth）进行对比，单模型无集成基准下，本文方法在BMA、F1、敏感度、特异度均实现最高值（BMA 0.802、F1 0.780、敏感度 0.802、特异度 0.965），相比MRE提升约4% BMA；OOD过滤进一步提升性能。

**⚠️ 局限性**

局限性包括：仅在单一CNN分类器上验证，缺乏多模型或多数据集的评估；数据生成与分类器分离，可能未能充分协同优化；OOD过滤对超参数γ敏感，需手工调优；对医学长尾的跨数据集泛化能力未充分验证。

---

## 116. Two Calls, Two Moments, and the Vote-Accuracy Curve of Repeated LLM Inference

**arXiv ID:** 2605.03379 | [PDF](https://arxiv.org/pdf/2605.03379v1)

**作者:** Yi Liu `[一作]` `[通讯]`, Yi Liu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究二分类LLM推理的二元正确性层，提出仅用两次标注呼叫即可识别可行的两矩（均值和二阶矩）并构造分布无关的多数投票区间，给出闭式三投票区间与无穷投票极限，并在实验中验证其有效性。

**💡 创新点**

创新点包括：
- 发现两次呼叫即可识别所有可能的三点支撑极端分布；
- 利用二次对偶证书得到任意奇数投票预算的精确两矩区间；
- 提供三投票闭式公式与无穷投票阈值敏感性分析；
- 引入最大熵和LDGP（Gaussian‑probit）两种参数化完成方案；
- 通过实验验证两矩区间与实际投票性能的一致性。

**🔧 技术方法**

使用的技术包括：
- 两矩极值理论与三点支撑极限；
- 二次对偶（quadratic dual）证书；
- 置信区间与有限样本推断（Wald moment region）；
- 最大熵与Gaussian probit模型完成；
- GLUE数据集实验与投票准确率计算。

**📊 数据集**

实验使用的公开数据集是GLUE中的QNLI和QQP（均转为二分类任务）。

**📈 对比分析**

比较方法：对每个模型、温度与投票预算，先用两次呼叫估计μ和ν，计算两矩区间；随后在同一数据集上执行5次呼叫，得到3投票与5投票经验准确率，并检查其是否落入区间。实验表明：
- 经验准确率均落在对应的两矩区间内；
- 高同例相关的策略几乎没有投票提升；
- 低同例相关或随机混合策略在3/5投票时可超过单次准确率。

**⚠️ 局限性**

局限性：
- 仅考虑二分类正确性层，忽略多分类或更细粒度的不确定性；
- 假设条件独立且同分布，实际LLM输出可能出现序列相关性；
- 无穷投票极限对阈值附近的质量分布高度敏感，且难以通过有限呼叫估计；
- 参数化完成（MaxEnt/LDGP）需额外假设，若真实分布偏离假设会导致误差；
- 仅使用了5次呼叫的数据，无法完全验证更大投票预算下的表现。

---

## 117. StateSMix: Online Lossless Compression via Mamba State Space Models and Sparse N-gram Context Mixing

**arXiv ID:** 2605.02904 | [PDF](https://arxiv.org/pdf/2605.02904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 118. Global and Local Topology-Aware Attention with Persistent Homology and Euler Biases for Time-Series Forecasting

**arXiv ID:** 2605.03163 | [PDF](https://arxiv.org/pdf/2605.03163v1)

**作者:** Usef Faghihi `[一作]` (University of Quebec), Amir Saki `[通讯]` (University of Quebec)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种在时间序列预测中将拓扑信息注入自注意力的框架，兼顾全局与局部几何结构；

**💡 创新点**

创新在于通过持久同调、锚定欧拉变换和核希尔伯特通道直接对注意力 logits 进行加权，并加入验证门控的局部残差；

**🔧 技术方法**

利用持久同调（H0-H2）、Euler characteristic transform、RKHS 距离与软化 H1、H2 近似，以及自适应温度参数；

**📊 数据集**

在七个公开和合成数据集上评估，包括高阶拓扑测试、周期/环形、壳层/空洞、CO₂、S&P 500 波动率、NASA IMS轴承降解等；

**📈 对比分析**

与无拓扑基线及标准 Transformer 结构（轻量化注意力/Ridge、PatchTST、TimeSeriesTransformer）对比，采用无泄漏验证-选择-测试协议；在大多数实验中实现正向配对改进，平均 RMSE 降低分别为 12.5%、23.5% 与 47.8%，但增益在几何结构强的任务中更显著；

**⚠️ 局限性**

局限性包括：计算持久同调时需要截断或近似，H1/H2 近似不等价于精确持久性；O(N²) 甚至 O(N³) 复杂度限制长序列；实验仅覆盖有限数据与模型族，需在更广泛公共数据集与长序列架构上验证。

---

## 119. Kernel Affine Hull Machines for Compute-Efficient Query-Side Semantic Encoding

**arXiv ID:** 2605.02950 | [PDF](https://arxiv.org/pdf/2605.02950v1)

**作者:** Mohit Kumar `[一作]` (University of Rostock), Manuela Geiß `[通讯]` (Software Competence Center Hagenberg GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在已冻结的语义教师空间下，研究如何用轻量级的KAHM几何估计器将查询的廉价词汇特征映射到该语义空间，从而在不执行在线Transformer推理的前提下完成查询编码。

**💡 创新点**

提出了一个完全可分析的后验估计与原型混合框架：利用KAHM的空间折叠度量得到后验权重，再通过NLMS更新语义原型，实现非渐近的后验近似误差控制与完整的编码误差分解，证明了在固定教师空间下可替代在线神经推理。

**🔧 技术方法**

核心技术包括：①核仿射包络机（KAHM）与其空间折叠测度；②后验概率的梯度无关估计与归一化后验归一化；③基于后验权重的原型混合模型；④Normalized Least‑Mean‑Squares (NLMS) 对原型的自适应迭代；⑤在可分析的RKHS假设空间上进行非参数后验估计。

**📊 数据集**

实验使用的是奥地利法律检索基准：10,762个法律条文片段与5,000个人工生成的查询，查询语料由7种表述风格产生，训练集共40,000条。教师模型为Mixedbread E5-large（1024维），词汇特征采用IDF‑SVD 512维。

**📈 对比分析**

与IDF‑SVD词汇检索、直接Transformer（Mixedbread）检索以及多种匹配的学习型适配器（岭回归、MLP回归、检索蒸馏学生、逻辑回归原型、MLP原型）进行比较。KAHM在教师空间重构（MSE=0.000091, R²=0.907）与检索质量（MRR@20=0.504, Hit@20=0.694, Top‑1=0.411）上均超过其他学习型适配器，并在保持相同冻结语义索引的前提下实现了从800 ms到94 ms的查询时延提升（≈8.5×）。

**⚠️ 局限性**

局限性包括：①仅在已冻结教师空间和离线语料索引可预先完成的部署场景下验证；②对不同语义模型、不同查询语义复杂度的泛化能力尚未系统评估；③KAHM后验估计依赖于空间折叠的分布假设，若簇间重叠严重可能导致误差放大；④原型数目与KAHM参数（ω、K）需要手工调优，缺乏自动化选择机制。

---

## 120. Keyword spotting using convolutional neural network for speech recognition in Hindi

**arXiv ID:** 2605.02928 | [PDF](https://arxiv.org/pdf/2605.02928v1)

**作者:** Saru Bharti `[一作]` (Indian Institute of Technology), Pushparaj Mani Pathak `[通讯]` (Indian Institute of Technology)

**通讯引用:** 1499 | [OpenAlex ID](https://openalex.org/A5058929088)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研发了面向印地语的关键词检测(KWS)系统，使用CNN对自建的40,000条44kHz、1.9秒的语音样本进行训练与测试。

**💡 创新点**

创新点在于：①自建覆盖21类（0–15数字、ha、nhi、sambandh、vibhag及负类）Hindi KWS数据集；②结合多种噪声增广与时间偏移，显著提升鲁棒性；③在低功耗设备上实现高精度91.79%的轻量级CNN模型。

**🔧 技术方法**

技术手段：MFCC特征提取、卷积层+ReLU+批归一化+最大池化+Dropout的CNN架构，后接两层全连接+softmax；采用交叉熵损失、Adam优化器，mini-batch 64，动态学习率。

**📊 数据集**

使用数据集：自建40,000条样本（数字0–15、ha、nhi、sambandh、vibhag、negative），每条1.9s、44kHz；训练期间通过噪声叠加生成约35,000条增强样本。

**📈 对比分析**

方法比较：通过增广前后对比实验，模型从约60%提升至91.79%准确率；验证集达95%；在10个未见过的样本（每人10条）上测试亦保持91.79%。

**⚠️ 局限性**

局限性：数据集词汇范围有限（主要为数字和少量词），对极端噪声或多种口音的鲁棒性未充分验证；缺乏与大型云端模型（如Whisper）或其他公开KWS模型的对比基准。

---

## 121. Discovering Reinforcement Learning Interfaces with Large Language Models

**arXiv ID:** 2605.03408 | [PDF](https://arxiv.org/pdf/2605.03408v1)

**作者:** Akshat Singh Jaswal `[一作]`, Paras Chopra `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

针对强化学习任务，自动从原始模拟器状态中搜索并生成最优的观测映射和奖励函数，从而构造完整的 RL 接口。

**💡 创新点**

将大语言模型 (LLM) 作为结构化突变操作符与 MAP‑Elites 进化框架相结合，实现了从代码程序空间中联合搜索观测和奖励，首次在同一框架内实现了完整 RL 接口的自动设计。

**🔧 技术方法**

使用 LLM (Claude Sonnet 4.6) 进行代码生成和突变，MAP‑Elites 质量多样性归档进行搜索，内部循环采用 PPO 训练评估接口性能，利用轨迹级成功度量作为进化的适应度信号。

**📊 数据集**

实验数据集包括：XLand‑MiniGrid 的 Easy/Medium/Hard 三个离散网格任务；MuJoCo 的 Go1 推力恢复任务和 Panda 轨迹跟踪任务，覆盖离散推理和连续控制两大领域。

**📈 对比分析**

与稀疏奖励、观测仅优化、奖励仅优化以及独立 LLM 采样等基线进行比较。联合优化在所有任务上都取得最高成功率（如 Easy/Medium 约 99%，Hard 约 85%，Panda 约 45%，Go1 约 48%），而任何单独优化在至少一个任务上会崩溃；独立 LLM 采样的平均表现远低于进化搜索。

**⚠️ 局限性**

局限性包括：需要事先提供可靠的轨迹级成功度量；计算成本高（主要由 PPO 训练决定）；依赖结构化模拟器状态，难以迁移到仅观测视觉等场景；进化策略相对简单，可进一步提升搜索效率。

---

## 122. ProgramBench: Can Language Models Rebuild Programs From Scratch?

**arXiv ID:** 2605.03546 | [PDF](https://arxiv.org/pdf/2605.03546v1)

**作者:** John Yang `[一作]` (Meta FAIR), Ofir Press `[通讯]` (Meta TBD)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并构建了ProgramBench基准，用于评估软件工程代理从零开始重建可执行程序的能力。

**💡 创新点**

通过仅给出可执行文件及其文档，让模型自行做架构设计和实现，评估实现多样性；自动化生成200个任务实例并构造行为测试。

**🔧 技术方法**

基于SWE‑agent（mini‑swe‑agent）与Claude Sonnet 4.5 等大型语言模型，使用 Docker 环境禁网，利用SWE‑agent进行编译、行为测试生成及代码生成；使用 assertion quality linter 等工具提升测试质量。

**📊 数据集**

从 GitHub 公开仓库筛选 200 个可执行项目（主要为 C/C++、Go、Rust、Java、Haskell 等），通过自动化流程生成任务实例。

**📈 对比分析**

通过 %Resolved 与 %Tests Passed 两指标对比 9 个语言模型；未出现任何模型完全通过任务；最佳模型 Opus 4.7 在 3% 任务中 95%+ 测试通过；简单 CLI 工具表现好，复杂系统表现差。

**⚠️ 局限性**

测试覆盖有限，未捕捉非功能性属性（性能、内存等）；行为测试只验证输入输出，可能忽略边缘输入；作弊检测不完美；模型生成的代码与原始实现差异大，导致难以比较实现细节。

---

## 123. Automated Large-scale CVRP Solver Design via LLM-assisted Flexible MCTS

**arXiv ID:** 2605.03339 | [PDF](https://arxiv.org/pdf/2605.03339v1)

**作者:** Tong Guo `[一作]` (Nanyang Technological University), Yew Soon Ong `[通讯]` (Nanyang Technological University)

**通讯引用:** 27196 | [OpenAlex ID](https://openalex.org/A5068243197)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了LaF-MCTS框架，利用三层决策层次与改进的蒙特卡洛树搜索自动设计大规模CVRP求解器。

**💡 创新点**

通过三层决策层次与语义剪枝+分支再生机制，使LLM逐步构建分解+子求解器，显著提升搜索效率并实现全自动化。

**🔧 技术方法**

采用LLM代码生成、Monte Carlo Tree Search、WSMD语义距离、结构匹配以及负约束提示实现分支再生，并基于HGS模板进行参数调优。

**📊 数据集**

训练使用500节点合成CVRP数据，评估使用CVRPLib的100个100–1000节点实例。

**📈 对比分析**

与Metaheuristics、NCO和LLMaAD基线对比，使用平均目标值、最佳解次数、平均排名等指标，LaF-MCTS在所有规模上均优于所有基线，达到state‑of‑the‑art水平。

**⚠️ 局限性**

依赖专家预先定义的HGS工作流限制了架构多样性，缺乏完全自适应的流程进化，对LLM质量和算力的依赖较高。

---

## 124. An Optimal Algorithm for Cardinality-Constrained Diameter Partitioning

**arXiv ID:** 2605.03431 | [PDF](https://arxiv.org/pdf/2605.03431v1)

**作者:** Chao Xu `[一作]` (University of Electronic Science and Technology of China), Mingdong Yang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种全局最优的二类划分算法，能够在O(n²)时间内同时求得所有基数c∈{0,…,n}的最小划分直径；对欧氏空间下的单一基数问题给出了次二次子多项式解法；并给出了匹配的Ω(n²)下界。

**💡 创新点**

核心创新在于：①将原问题化简为树的“瓶颈二色化”问题，并用树DP在O(n²)内一次性求解所有基数；②利用最大生成树的循环性质避免二分搜索所有权重；③证明任何算法在仅能查询边权的模型下都需Ω(n²)查询，从而说明所给算法的时间复杂度是最优的。

**🔧 技术方法**

主要技术包括：最大生成树构造、树DP（最大加法结合子集和算法）、瓶颈二色化问题的归约、欧氏距离的最近点与最小生成树算法、以及信息论式下界证明。

**📊 数据集**

该工作主要为理论分析，未使用具体数据集；实验部分仅对理论算法的复杂度进行推导与比较。

**📈 对比分析**

与Avis（1986）提出的O(n²log n)算法相比，本方法去掉了对数因子，且一次性得到所有基数的最优解；在欧氏维度d=2时，可实现O(nlog n)时间；对于固定d≥3，时间为O(n^{2-2/(⌈d/2⌉+1)+ε})，已是目前已知的最优或最接近最优复杂度。

**⚠️ 局限性**

限制主要包括：①在一般加权图（非欧氏）下仍需O(n²)时间，无法突破；②对欧氏空间多维度问题，除平面外仍保持次二次；③算法基于对所有边权的查询，若边权分布已知或可预先获取，可能存在更快的特定场景方案。

---

## 125. DINO Soars: DINOv3 for Open-Vocabulary Semantic Segmentation of Remote Sensing Imagery

**arXiv ID:** 2605.03175 | [PDF](https://arxiv.org/pdf/2605.03175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 126. VDCores: Resource Decoupled Programming and Execution for Asynchronous GPU

**arXiv ID:** 2605.03190 | [PDF](https://arxiv.org/pdf/2605.03190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 127. ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration

**arXiv ID:** 2605.03042 | [PDF](https://arxiv.org/pdf/2605.03042v1)

**作者:** Ruofeng Yang `[一作]` (Shanghai Jiao Tong University), Shuai Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 132473 | [OpenAlex ID](https://openalex.org/A5100371500)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个开放源代码的自治机器学习研究工具包，协调跨模型对抗式协作，提供执行、编排和保证三层架构，并实现了多项审核机制与技能库。

**💡 创新点**

采用异构模型执行-审阅（cross‑model adversarial collaboration）以消除同一模型自评的偏差，设计了多层保障堆栈（三阶段证据到声明审计、五轮编辑、证明检查、视觉PDF检查、引用审核），以及可调节努力级别和持久化研究wiki，实现模块化、可替换的工作流和自适应优化。

**🔧 技术方法**

利用大型语言模型（Claude、GPT、Codex、Gemini等）、MCP桥接、Markdown技能文件、持久化wiki（文件系统版本化）、自定义审计流水线、五轮编辑算法、证明验证器、SVG渲染器、Rust CLI ARIS‑Code、meta‑optimization日志与建议等技术。

**📊 数据集**

主要使用实验结果文件、论文引用数据库（DBLP、CrossRef、Semantic Scholar等）以及GPU实验产生的内部数据；未明确列出公开数据集。

**📈 对比分析**

通过一次夜间运行，四轮审阅‑修订循环将评分从5.0提升至7.5/10，并在20+ GPU实验中剔除不支持的声明；未提供系统性对比实验，示例性证据表明跨模型审阅能提升质量，但缺乏因果评估。

**⚠️ 局限性**

无法保证结果正确性；LLM仍可能出现幻觉、方法缺口；跨模型审阅虽降低风险但未彻底消除；缺乏系统性因果评估与公开数据集验证；需人机协作提升最终质量。

---

## 128. Realizable Bayes-Consistency for General Metric Losses

**arXiv ID:** 2605.03823 | [PDF](https://arxiv.org/pdf/2605.03823v1)

**作者:** Dan Tsir Cohen `[一作]` (Ben-Gurion University of the Negev), Aryeh Kontorovich `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 868 | [OpenAlex ID](https://openalex.org/A5019371046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究了在可实现（realizable）设定下，使用任意（可能无限的）度量损失的强普适贝叶斯一致性（strong universal Bayes‑consistency）问题，并给出了判定可一致性的必要且充分条件。

**💡 创新点**

提出了“无限非递减（γ_k）-Littlestone 树”这一新的组合学障碍，并证明若存在此类树则无学习器能实现一致性；若不存在，则构造了明确的分布无关学习规则实现一致性，从而在可实现的度量损失学习中完成了完全表征。

**🔧 技术方法**

主要采用组合学树（Littlestone 树）与Gale–Stewart 无限博弈理论相结合的技术，并利用可测性与紧致参数化假设实现可测策略；同时在无穷路径上使用 Borel–Cantelli 定理构造反例，进一步借助分块与有限损失子问题的分区技术实现学习规则。

**📊 数据集**

该工作为理论研究，无需使用真实数据集；所有结果均为严格数学证明。

**📈 对比分析**

与已有方法的比较仅在理论层面：论文展示了在存在无限非递减 Littlestone 树时任何学习器必然失败，而在无此障碍时构造的学习器几乎必然收敛至零风险；实验性能没有评估，主要关注一致性与可行性。

**⚠️ 局限性**

局限性包括：仅考虑可实现（realizable）情形；对标签空间和实例空间的紧致参数化与可测性假设较强；未给出对抗性或无监督情况的结果；以及对非度量或有限损失情况的推广仍需进一步研究。

---

## 129. Diffusion Masked Pretraining for Dynamic Point Cloud

**arXiv ID:** 2605.03639 | [PDF](https://arxiv.org/pdf/2605.03639v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 130. Single-Step Six-Dimensional Movable Antenna Reconfiguration for High-Mobility IoV: Modeling, Analysis, and Optimization

**arXiv ID:** 2605.03321 | [PDF](https://arxiv.org/pdf/2605.03321v1)

**作者:** Maoxin Ji `[一作]` (Jiangnan University), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 46333 | [OpenAlex ID](https://openalex.org/A5079052203)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种低复杂度、无CSI的六维可移动天线（6DMA）单步重构框架，用以提升高速移动的车联网（IoV）环境中的上行容量；

**💡 创新点**

创新点包括：①基于纬经度网格的离散位置生成与邻接图理论下的运动成本与时间成本下界分析；②利用累计车辆分布预测与离线环境先验相结合的适应性优化策略；③将重构动作严格限定在一阶邻域，彻底消除服务中断；

**🔧 技术方法**

采用了图论（BFS、匈牙利算法）、预测模型（车辆速度平均预测）、离线映射库、历史反馈评分、贪心位置分配及旋转决策；

**📊 数据集**

主要通过数值仿真验证，使用城市十字路口场景的车辆分布模型（随机布点+高斯速度），并未使用公开实验数据集；

**📈 对比分析**

与固定天线、圆轨道6DMA、仅旋转6DMA以及全局搜索6DMA等基线比较，实验显示所提单步6DMA在不同功率、用户数及重构间隔下均能实现至少10%–20%的上行总速率提升，且运动与时间成本显著低于全局搜索；

**⚠️ 局限性**

局限性在于：①仅考虑离散位置网格，实际机械实现仍需进一步验证；②预测误差和极端交通突发事件下的鲁棒性未被完全评估；③算法对车辆密度和分布的统计稳定性有一定要求，极高移动速率或大规模车流时可能需更频繁重构。

---

## 131. Sentinel2Cap: A Human-Annotated Benchmark Dataset for Multimodal Remote Sensing Image Captioning

**arXiv ID:** 2605.03189 | [PDF](https://arxiv.org/pdf/2605.03189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 132. Generate, Filter, Control, Replay: A Comprehensive Survey of Rollout Strategies for LLM Reinforcement Learning

**arXiv ID:** 2605.02913 | [PDF](https://arxiv.org/pdf/2605.02913v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 133. Enhancing Agent Safety Judgment: Controlled Benchmark Rewriting and Analogical Reasoning for Deceptive Out-of-Distribution Scenarios

**arXiv ID:** 2605.03242 | [PDF](https://arxiv.org/pdf/2605.03242v1)

**作者:** Zuoyu Zhang `[一作]` (Shenzhen University), Yancheng Zhu `[通讯]` (Shenzhen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建 ROME benchmark 并提出 ARISE 推理增强方法

**💡 创新点**

首次系统将显式危险轨迹改写为隐式、模糊及捷径型欺骗实例，并通过检索类比推理提升安全判断

**🔧 技术方法**

采用多模型 red‑team 进化、类比检索（ReAct 风格轨迹）与 prompt 结合的 inference‑time 技术

**📊 数据集**

使用 100 条 R-Judge 公开不安全轨迹、2,000 条 AgentSafeBench 对话（ReAct 格式）以及 36 条手工 seed 示例

**📈 对比分析**

与六大代表性 LLM（GPT‑4o、Claude 3.7 Sonnet、DeepSeek‑V3、DeepSeek‑R1、Qwen3‑8B、Qwen3‑235B）在四种评测条件（Original、IR、CA、SDM）下对比，ARISE 标准模式平均提升 20‑40% F1，尤其在隐式风险(IR)上显著提升，误报率也显著下降

**⚠️ 局限性**

局限于来源单一的 100 条轨迹、未覆盖安全但模糊情景、仅评估判断而非完整代理执行、检索/推理延迟与对外部知识库的依赖

---

## 134. The Fragility of AI Companionship: Ontological, Structural, and Normative Uncertainty in Human-AI Relationships

**arXiv ID:** 2605.03367 | [PDF](https://arxiv.org/pdf/2605.03367v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 135. PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution

**arXiv ID:** 2605.03399 | [PDF](https://arxiv.org/pdf/2605.03399v1)

**作者:** Onkar Jadhav `[一作]` (University of Western Australia), Nicole L. Jones `[通讯]` (University of Western Australia)

**通讯引用:** 2312 | [OpenAlex ID](https://openalex.org/A5091755240)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出PODiff框架，用POD系数空间进行条件扩散，以实现高效的科学领域超分辨率生成与不确定性量化。

**💡 创新点**

创新点在于：①使用固定、方差排序的POD基底作为结构化潜在空间，保持线性可逆性与可解释性；②在潜在空间进行扩散模型训练，实现对空间场的高质量采样与解析不确定性传播；③大幅降低内存与计算成本，保持与像素空间扩散相当的重建精度。

**🔧 技术方法**

核心技术包括：POD降维与投影、条件扩散模型（前向噪声、反向去噪网络为MLP）、时间步嵌入、标准化系数、解析从潜在到物理空间的协方差传播；对比实验使用U-Net、MC Dropout U-Net、像素空间Diffusion、RBF插值等基线。

**📊 数据集**

实验数据集：1）西澳洲海表温度（SST） 1998‑2009年训练，2010年验证，2011年测试；2）人工生成的二维推移扩散 PDE 数据，用于控制实验。

**📈 对比分析**

与基线相比，PODiff在SST下标记的RMSE/MAE远低于U‑Net、RBF、RandOrthDiff，近似PixelDiff的重建精度，同时显著降低了参数量、GPU内存和推理时间；在不确定性评估上，PODiff的经验覆盖率与目标水平高度一致，MACE低于MC Dropout U‑Net。

**⚠️ 局限性**

局限性包括：对低秩线性结构依赖强，湍流或高阶非线性场需更多POD模式或自适应重建；未建模截断误差；POD基底固定后若数据分布漂移需重新计算。

---

## 136. ConRAD: Conformal Risk-Aware Neural Databases

**arXiv ID:** 2605.03806 | [PDF](https://arxiv.org/pdf/2605.03806v1)

**作者:** Sonia Horchidan `[一作]` (KTH), Paris Carbone `[通讯]` (KTH)

**通讯引用:** 1869 | [OpenAlex ID](https://openalex.org/A5039432313)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种基于 Conformal Risk Control 的神经图数据库查询框架 ConRAD，能够在多跳查询中提供可验证的召回保证。

**💡 创新点**

创新点在于将 CRC 扩展到多操作符查询，提出阈值空间量化标量化与 conformal gate 两种技术，实现在保证召回的同时最大化精度。

**🔧 技术方法**

采用 Conformal Prediction、Conformal Risk Control、量化空间标量化、分布式神经链接预测器 UltraQuery 以及自定义的哈希分布式打分方法。

**📊 数据集**

实验使用 FB15k-237、NELL-995、YAGO3-10 三个主流知识图谱，在 5%、20%、40% 随机删边的不同稀疏度下进行评估。

**📈 对比分析**

与纯检索、手工阈值化推理、手工阈值化混合三种基线比较，ConRAD 在召回目标下精度与基线持平或更高，召回偏差最大不超过 0.046，神经调用量可降至 0%。

**⚠️ 局限性**

局限在于只能给出边缘（marginal）召回保证，无法对单个查询做条件覆盖；需要离线校准，缺乏零样本迁移能力；对底层模型的质量仍有一定依赖。

---

## 137. Segmenting Human-LLM Co-authored Text via Change Point Detection

**arXiv ID:** 2605.03723 | [PDF](https://arxiv.org/pdf/2605.03723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 138. Making the Invisible Visible: Understanding the Mismatch Between Organizational Goals and Worker Experiences in AI Adoption

**arXiv ID:** 2605.03078 | [PDF](https://arxiv.org/pdf/2605.03078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 139. Same Voice, Different Lab: On the Homogenization of Frontier LLM Personalities

**arXiv ID:** 2605.02897 | [PDF](https://arxiv.org/pdf/2605.02897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 140. Cryptographic Registry Provenance: Structural Defense Against Dependency Confusion in AI Package Ecosystems

**arXiv ID:** 2605.03309 | [PDF](https://arxiv.org/pdf/2605.03309v1)

**作者:** Alan L. McCann `[一作]` `[通讯]` (Mashin), Alan L. McCann (Mashin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出一种基于公钥签名的分发可信度体系，利用注册表身份、双重签名和命名空间绑定三层加密防御，彻底消除依赖混淆攻击。

**💡 创新点**

创新点包括：①为每个注册表实例生成 Ed25519 身份并公开 key fingerprint；②在打包时由发布者签名、上传时由注册表反签名，实现时间顺序和责任分离；③在消费者侧通过 TOFU 锁定注册表指纹并强制命名空间绑定，实现 Artifact‑level 的完整可信链。

**🔧 技术方法**

核心技术为 Ed25519 加签/验签、SHA‑256 哈希、两层压缩归档格式、JSON/TOML 配置文件、可选的透明度日志和外部线性化 Ledger（AI 生成 Provenance）。

**📊 数据集**

评估使用多种语言（Go、Rust、Erlang）实现的公开实现，测试数据集为 4 机器构成的 Krate（共 4 机器源代码，约 10KB），并对八大生态（npm、Cargo、Hex.pm、PyPI、Go Modules、Docker/OCI、NuGet、Maven）做功能对比。

**📈 对比分析**

对比方法为功能矩阵（签名、注册表身份、反签名、消费者强制等）与性能评估：单包验证平均 1.2‑2.5 ms，完整链（含 lineage）约 1.1‑4.1 ms，归档开销随包大小从 2600% 降至 300% 之间；与现有 Sigstore、P2P sumdb 等方案相比，性能相当或更优且无网络延迟。

**⚠️ 局限性**

局限性包括：①缺乏密钥失效/吊销机制；②TOFU 初始信任可能被 MITM 攻击；③需手动在消费者侧配置命名空间绑定；④大规模组织多注册表时密钥管理成本高；⑤不处理打字错误、内部威胁、社交工程攻击等非签名相关风险。

---

## 141. MEMTIER: Tiered Memory Architecture and Retrieval Bottleneck Analysis for Long-Running Autonomous AI Agents

**arXiv ID:** 2605.03675 | [PDF](https://arxiv.org/pdf/2605.03675v1)

**作者:** Bronislav Sidik `[一作]` (Ben-Gurion University), Lior Rokach `[通讯]` (Ben-Gurion University)

**通讯引用:** 31857 | [OpenAlex ID](https://openalex.org/A5012622155)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了适用于长周期自主代理的分层内存体系结构，解决了上下文崩溃、压缩不连续、结构盲点和缺乏归因循环等问题。

**💡 创新点**

创新点在于将记忆拆分为结构化的情节式JSONL存储、五信号加权检索引擎、注意力归因认知权重更新、异步整合守护进程以及基于PPO的检索权重学习。

**🔧 技术方法**

采用BM25检索、时间衰减、认知权重、层级提升等多信号加权、LLM事实提取、SGLang日志概率归因以及PPO强化学习调整检索权重。

**📊 数据集**

使用LongMemEval‑S 500道题目的长期记忆评估数据集进行实验。

**📈 对比分析**

在该基准上，Semantic预填充后模型在Acc、F1分别达到0.382、0.412，单会话召回率0.686–0.714，高出RAG BM25 GPT‑4o基线0.560，提升约33个百分点。

**⚠️ 局限性**

主要局限包括归因路径受硬件限制、PPO权重被BM25无界得分主导、以及关系提取仅使用粗略启发式模式。

---

## 142. SILMARILS: Information-Theoretic and Quantum-Secure Designated-Verifier Signatures

**arXiv ID:** 2605.03230 | [PDF](https://arxiv.org/pdf/2605.03230v1)

**作者:** Hassan Khodaiemehr `[一作]` (University of British Columbia), Dariia Porechna `[通讯]` (EternaX Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

设计并实现了一种信息理论安全、轻量级的数字签名框架SILMARILS，可在两方可转移指定验证者模式和三方可转移签名模式下使用。

**💡 创新点**

创新点在于利用有限域𝔽_p上的极简代数核心与完美的2/2-Shamir秘密共享，既实现了极小的密钥/签名尺寸，又兼具在ROM和QROM下的DV可模拟性和非指定验证者的EUF‑CMA安全，以及在三方模式下的统计安全与1/p误差。

**🔧 技术方法**

采用了Shamir秘密共享、随机数生成、哈希函数/随机预言机（ROM、QROM）模型、量子随机预言机重编程技术、Fitzi等可转移签名框架以及Jakobsson–Sako–Impagliazzo的DV可模拟性分析。

**📊 数据集**

论文未使用外部数据集，而是基于256位素数域的参数进行实现与性能评估，展示了理论分析与实验结果。

**📈 对比分析**

通过与Dilithium、Falcon、SPHINCS+等主流后量子签名方案在密钥大小、签名大小和运算开销上的对比，SILMARILS在相同安全等级下实现公钥64B、签名160B，仅需约5–6次域乘法，显著优于传统方案。

**⚠️ 局限性**

局限性包括：仅支持可转移指定验证者（非公开验证）场景；三方模式存在1/p误差；对单一破坏者的安全性有限；在实际区块链应用中需配合专门的事务验证流程。

---

## 143. Enhance the after-discharge mortality rate prediction via learning from the medical notes

**arXiv ID:** 2605.03560 | [PDF](https://arxiv.org/pdf/2605.03560v1)

**作者:** Zijiang Yang `[一作]` `[通讯]` (University of Texas), Zijiang Yang (University of Texas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究利用电子健康记录中的非结构化医疗笔记，结合深度神经网络与类别权重池化机制，对肾衰竭患者出院后15、30、60、365天的死亡率进行预测。

**💡 创新点**

创新点在于：①提出按笔记类型学习权重的池化机制，能够自动抑制冗余信息并突出关键信息；②通过权重、敏感性和关键词分析揭示不同笔记类别与死亡率的关系。

**🔧 技术方法**

技术手段包括：文本预处理（nltk分词 + CountVectorizer 生成 400 维词频向量）、传统机器学习模型（Logistic Regression、Random Forest、XGBoost）以及自定义的 4 层深度网络（含短路连接）配合 Adam 优化器与早停策略。

**📊 数据集**

数据集为 MIMIC‑III ICU 数据库，筛选出 6,365 名确诊肾衰竭患者，使用其基本信息与医疗笔记作为特征。

**📈 对比分析**

比较方法：将基于基本信息的传统模型、加入笔记信息的传统模型以及提出的 DNN 进行对比。DNN 在所有时间窗（15/30/60/365 天）下的 AUC‑ROC 分别比传统模型高 2%–14%，并且相较仅使用基本信息的模型提升约 0.1。

**⚠️ 局限性**

局限性：①文本质量低、重复冗余；②实验仅针对肾衰竭患者，缺乏跨病种验证；③模型解释性虽有分析但仍不够直观；④未进行多中心外部验证，泛化性待进一步评估。

---

## 144. AHPA: Adaptive Hierarchical Prior Alignment for Diffusion Transformers

**arXiv ID:** 2605.03317 | [PDF](https://arxiv.org/pdf/2605.03317v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 145. HeadQ: Model-Visible Distortion and Score-Space Correction for KV-Cache Quantization

**arXiv ID:** 2605.03562 | [PDF](https://arxiv.org/pdf/2605.03562v1)

**作者:** Jorge L. Ruiz Williams `[一作]` `[通讯]` (Independent Researcher), Jorge L. Ruiz Williams (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 KV 缓存压缩方法 HeadQ，在写入时保留低秩残差代码，读取时用当前查询将其解码为对 logit 的补偿，从而在保持低位宽的同时显著降低解码时的语义损失。

**💡 创新点**

创新点在于将 KV 缓存错误的评价转向模型可见坐标：键的误差以 softmax 中的 score‑visible 量度（Fisher 量化）来衡量，值的误差以 A² 加权的读出误差来评估；并基于此提出 HeadQ 的低秩残差补偿机制。

**🔧 技术方法**

采用的技术包括：基于校准查询学习的每头 PCA 量化基；对键残差做低秩量化；利用 Fisher 信息矩阵计算 score‑visible 误差；使用 A² 权重的读取误差代理；以及对比实验中的 Oracle 主动轴、随机基准和逆向控制。

**📊 数据集**

实验使用 WikiText-103 作为解码语料，评估六款模型（GPT‑2 124M、Pythia‑160M、Qwen2‑0.5B、Qwen2.5‑0.5B、TinyLlama‑1.1B、Mistral‑7B）。

**📈 对比分析**

与传统基于 MSE 的量化相比，HeadQ 在 2‑bit 关键行上将 PPL 降低 84–94%，在全 KV 2‑bit 组合中平均提升 10–20% 的 PPL；Fisher 量化在六个模型上均显著优于 MSE（平均 Spearman 提升 0.65–0.70）。

**⚠️ 局限性**

局限性包括：未实现高效的 packed kernel 或延迟/吞吐量优化；实验仅在短窗口和低于 7B 的模型上验证；未处理查询的量化；并且仅证明了 KV 缓存压缩的机制性效果，而非完整的生产系统部署。

---

## 146. A Framework for Exploring and Disentangling Intersectional Bias: A Case Study in Fetal Ultrasound

**arXiv ID:** 2605.02942 | [PDF](https://arxiv.org/pdf/2605.02942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 147. QoS Assurance Mechanism for 5G Network Slicing Based on the Deep Reinforcement Learning PPO Algorithm

**arXiv ID:** 2605.03345 | [PDF](https://arxiv.org/pdf/2605.03345v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 148. TRACE: A Metrologically-Grounded Engineering Framework for Trustworthy Agentic AI Systems in Operationally Critical Domains

**arXiv ID:** 2605.03838 | [PDF](https://arxiv.org/pdf/2605.03838v1)

**作者:** Serhii Zabolotnii `[一作]` `[通讯]` (Cherkasy State Business), Serhii Zabolotnii (Cherkasy State Business)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出TRACE框架，构建四层可信代理AI架构，并在临床、工业与司法三领域实例化验证。

**💡 创新点**

创新点包括将经典ML与LLM分层、提出计算可行性比率CPR作为模型稀疏度量、以及整合元测量的信任度量体系。

**🔧 技术方法**

使用经典机器学习模型、LLM验证器、状态化协同策略、GUM/VIM/ISO 17025测量框架及可选的Tsaheylu协议。

**📊 数据集**

使用各领域内部数据（临床病例、工业传感器、司法文档与合同文本），未公开公开数据集。

**📈 对比分析**

通过与TAO和Eywa等方法比较，TRACE在错误率下降、可追溯性与解释性方面优于单LLM部署，且CPR显示模型选择更合适。

**⚠️ 局限性**

局限在于缺乏完整实证数据、CPR需基准化、实例C尚未完全实现以及部分指标仍处于理论定义阶段。

---

## 149. Learning to Theorize the World from Observation

**arXiv ID:** 2605.03413 | [PDF](https://arxiv.org/pdf/2605.03413v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 150. Height Control and Optimal Torque Planning for Jumping With Wheeled-Bipedal Robots

**arXiv ID:** 2605.03302 | [PDF](https://arxiv.org/pdf/2605.03302v1)

**作者:** Yulun Zhuang `[一作]` (Southern University of Science and Technology), Chenglong Fu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 2879 | [OpenAlex ID](https://openalex.org/A5024107992)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了基于W-JBD模型的轮式双足机器人跳跃高度控制方案，并开发了基于贝叶斯优化的BOTP方法，结合Webots仿真实现了跳跃高度的精确控制。

**💡 创新点**

创新点在于①构建了考虑轮式双足机器人独特动力学的W-JBD模型；②提出了使用BOTP在有限迭代内获得连续且不破坏电机的最优扭矩曲线；③将仿真优化框架与贝叶斯优化结合，实现了高度误差≤4%并显著降低能耗。

**🔧 技术方法**

技术包括：W-JBD动力学建模、线性化姿态控制、Webots仿真、贝叶斯优化（Advisor/Google Vizier）求解扭矩优化、能耗评估与闭环控制。

**📊 数据集**

使用Webots仿真环境中的简化机器人模型（7.8 kg，总体参数与实际硬件匹配）进行实验，不涉及公开数据集。

**📈 对比分析**

对比方法：W-JBD模型的离散扭矩规划与BOTP连续扭矩规划。实验表明BOTP在平均40次迭代内将高度误差从4%降至0.2%，能耗下降约26.9%，且扭矩曲线平滑，降低电机损伤风险。

**⚠️ 局限性**

局限性包括：①模型仍未考虑落地阶段的软着陆；②贝叶斯优化对计算量仍有一定要求；③实验仅在仿真中验证，真实硬件落地表现尚待进一步验证。

---

## 151. Honest Reporting in Scored Oversight: True-KL0 Property via the Prekopa Principle

**arXiv ID:** 2605.03793 | [PDF](https://arxiv.org/pdf/2605.03793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 152. GRIFDIR: Graph Resolution-Invariant FEM Diffusion Models in Function Spaces over Irregular Domains

**arXiv ID:** 2605.03497 | [PDF](https://arxiv.org/pdf/2605.03497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 153. RFPrompt: Prompt-Based Expert Adaptation of the Large Wireless Model for Modulation Classification

**arXiv ID:** 2605.03279 | [PDF](https://arxiv.org/pdf/2605.03279v1)

**作者:** Md Raihan Uddin `[一作]` (Clemson University), Fatemeh Afghah `[通讯]` (Clemson University)

**通讯引用:** 3785 | [OpenAlex ID](https://openalex.org/A5035395012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于prompt的参数高效迁移方法RFPrompt，利用冻结的无线基础模型LWM实现对下游调制分类的适配，尤其在分布外和少量标注场景下表现优异。

**💡 创新点**

创新点在于：①为多专家Mixture-of-Experts结构设计专家特定的深层prompt，实现对每层注意力的精细引导；②仅训练约0.34%的参数即可获得与部分微调相当甚至更好的鲁棒性；③通过prompt学习在保持预训练结构的同时实现快速适配。

**🔧 技术方法**

使用深度prompt学习（Expert‑Specific Deep Prompt Tokens）、Mixture‑of‑Experts Vision Transformer、路由器（router）和轻量化分类头，并结合AdamW、cosine退火等训练技巧。

**📊 数据集**

评估数据集为IEEE Dataport IQ（低分布偏移）和Real‑World IQ（真实空中捕获、显著分布外），均采用STFT谱图输入。

**📈 对比分析**

与冻结专家、部分微调、传统CNN、MAML、FOMAML、Reptile、SpectrumFM等基线对比，RFPrompt在Stage A中在N≤800时已领先，Stage B（few‑shot）更是显著超越，尤其在Real‑World IQ上取得最高精度（最高0.82）。

**⚠️ 局限性**

局限性：对prompt长度和层数的依赖需进一步探索；在高标注量或与预训练分布高度匹配的任务中，纯微调或其他元学习方法可弥补RFPrompt的优势；未对更多真实硬件和信道条件进行验证。

---

## 154. Skew polycyclic over finite chain rings associated to trinomials

**arXiv ID:** 2605.03164 | [PDF](https://arxiv.org/pdf/2605.03164v1)

**作者:** Maryam Bajalan `[一作]` (Institute of Mathematics and Informatics), Hassan Ou-azzou `[通讯]` (Mohammed VI Polytechnic University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在有限链环上由中心三项式定义的偏置循环码（skew polycyclic codes）的汉明等价性，提出了(n,σ)-等价关系并通过子群H_{ℓ,σ}给出了代数判定；

**💡 创新点**

创新点在于将等价关系与Schur乘积群中的子群关联，提供了判定两类码族是否汉明等价的完整代数标准，并给出了对应等价类的显式计数公式；

**🔧 技术方法**

使用了有限链环与σ-偏置多项式环的代数结构、σ-范数、单位群分解以及群论（Schur乘积、同构定理）等技术；

**📊 数据集**

无实验数据集，整个工作为纯理论推导；

**📈 对比分析**

方法以理论证明为主，未进行实验比较，性能评价基于能否给出完整的等价类计数和判定条件；

**⚠️ 局限性**

局限性包括仅适用于中心三项式、要求n≡ℓ≡0 μ等额外条件，且对非中心多项式或更一般的偏置多项式结构尚未覆盖；

---

## 155. Partially Observed Structural Causal Models

**arXiv ID:** 2605.03268 | [PDF](https://arxiv.org/pdf/2605.03268v1)

**作者:** Turan Orujlu `[一作]` (University of Tuebingen), Konrad P. Kording `[通讯]` (University of Pennsylvania)

**通讯引用:** 24198 | [OpenAlex ID](https://openalex.org/A5072047827)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并验证了部分可观测结构因果模型（POSCM），将结构生成与潜在上下文耦合，并给出节点与边级的可操作干预层次。

**💡 创新点**

在SCM框架中将边缘形成本身视为内生机制，采用有序生成与Kolmogorov–Arnold–Sprecher（KAS）边功能分解，实现可度量的消息增强模型，并给出可识别性理论与正负结果。

**🔧 技术方法**

使用有序生成语义、KAS分解、消息增强POSCM、节点/边消息干预、概率论可识别性证明以及数值验证。

**📊 数据集**

利用虚拟人类视网膜模拟器（NEURON ModelDB 2018247），包括光感受器→水平细胞→双极细胞→无突触细胞→视网膜神经节细胞的多层微电路。

**📈 对比分析**

通过三组实验验证识别性结论：①无β干预时模型不可识别；②边隐蔽时节点干预无法区分结构-机制；③结合β节点和V节点干预可恢复传递曲线。结果显示正面理论得到仿真验证，恢复的传递曲线与真实参数吻合，优于仅节点干预。

**⚠️ 局限性**

需对结构、上下文与值的完整读出和干预覆盖，目前仅在无环有序生成模型中验证，循环或时序扩展、噪声读出及有限干预仍待研究。

---

## 156. Bandits on graphs and structures

**arXiv ID:** 2605.03493 | [PDF](https://arxiv.org/pdf/2605.03493v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 157. Robust Agent Compensation (RAC): Teaching AI Agents to Compensate

**arXiv ID:** 2605.03409 | [PDF](https://arxiv.org/pdf/2605.03409v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 158. Enhancing Self-Supervised Talking Head Forgery Detection via a Training-Free Dual-System Framework

**arXiv ID:** 2605.03390 | [PDF](https://arxiv.org/pdf/2605.03390v1)

**作者:** Ke Liu `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 112307 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练‑自由双系统框架（TFDS），先用系统‑1对自监督伪造检测器的分数进行阈值路由，将样本分为自信集和不确定集；再用系统‑2对不确定集进行基于CLIP的局部证据挖掘、Qwen语言推理和文本重排序，最终在不改变全局分数分布的前提下重新排序不确定样本。

**💡 创新点**

创新点在于：① 将自监督检测器的输出视为系统‑1，利用Youden阈值实现轻量级的不确定样本路由；② 引入系统‑2进行细粒度多模态证据挖掘与语言推理，而不需要对检测器进行再训练；③ 通过“槽位保持”重排序，仅在不确定子集中修正样本顺序，保留基线检测器的全局结构，从而实现显著性能提升。

**🔧 技术方法**

技术包括：自监督音视听检测器（AVH‑Align/AVAD）；Youden阈值分割；冻结CLIP ViT‑L/14用于帧与局部补丁的相似度评分与筛选；Qwen‑VL‑7B进行局部证据描述；BGE‑Reranker‑Large将文本描述映射为可比较的排名分数；槽位保持的局部重排序算法。

**📊 数据集**

使用的公开数据集有：AVLips、FakeAVCeleb（FKAV）和 TalkingHeadBench（THB）。实验还包括在THB上对视频做四种扰动（反转、噪声、模糊、压缩）进行鲁棒性评估。

**📈 对比分析**

与基线自监督检测器（AVH‑Align、AVH‑Align*、AVAD）以及监督检测器（CViT、EfficientViT、RealForensics、LipFD）进行对比。实验显示，TFDS在全量测试集上对AVH‑Align*分别提升AP/ AUC 约 +12.6/ +18.8（THB）、+17.1/ +35.5（AVLips）和 +1.6/ +1.8（FKAV）；在不确定子集上提升幅度更大；在所有扰动设置下均能显著降低性能退化，最大可达 AP +25.6、AUC +28.5。

**⚠️ 局限性**

局限性包括：① 依赖基线检测器的初始分数分布，若基线分数差距已极大则提升空间有限；② 系统‑2的推理与重排序需要外部大型模型（CLIP、Qwen），计算成本相对较高；③ 只针对不确定子集改进，无法直接提升已被系统‑1判定为自信的样本；④ 阈值选择基于验证集，可能对不同数据分布或极端对抗样本敏感。

---

## 159. Human-Provenance Verification should be Treated as Labor Infrastructure in AI-Saturated Markets

**arXiv ID:** 2605.03210 | [PDF](https://arxiv.org/pdf/2605.03210v1)

**作者:** Erin McGurk `[一作]` (University of Cambridge), David Khachaturov `[通讯]` (University of Cambridge)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5015969121)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文通过理论分析阐述了在AI饱和市场中，验证人类存在的劳动力将成为稀缺且有价值的属性，并主张将人类来源验证视为劳动基础设施，而非奢侈品认证。

**💡 创新点**

创新点包括：①提出“人类来源溢价”概念，将其与Veblen效应区分；②构建“表现性人性”三维分类（关系存在劳动、审美来源劳动、责任劳动）；③提出“构成性人类存在”标准，用以评估混合人机工作；④将人类来源验证框架定位为劳动基础设施，并给出五项设计原则。

**🔧 技术方法**

无实验技术，主要采用劳动力经济学、信息经济学、制度经济学等理论框架进行推理和建模；未使用具体机器学习模型或算法。

**📊 数据集**

无数据集，文章依赖文献综述、案例分析（如ChatGPT、情感AI、行业认证案例）和理论模型。

**📈 对比分析**

无实验比较或性能评估，论文通过经济学与制度学视角构建假设并讨论潜在影响，未给出定量指标。

**⚠️ 局限性**

局限性包括：①理论假设缺乏实证验证；②对人类来源溢价的Veblen效应假设尚未被市场充分观察；③在多元化劳动者（尤其是边缘化群体）获取验证机制的可及性与公平性未得到充分解决；④未讨论监管与法律框架在不同司法辖区内的可行性与实施细节。

---

## 160. POSTCONDBENCH: Benchmarking Correctness and Completeness in Formal Postcondition Inference

**arXiv ID:** 2605.03356 | [PDF](https://arxiv.org/pdf/2605.03356v1)

**作者:** Gehao Zhang `[一作]` (University of Massachusetts), Juan Zhai `[通讯]` (University of Massachusetts)

**通讯引用:** 2225 | [OpenAlex ID](https://openalex.org/A5071575216)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个多语言（Python/Java）真实项目的后置条件生成基准，并实现了自动化评估平台，可同时测量生成后置条件的正确性与完整性。

**💡 创新点**

创新点包括①在真实代码仓库中构建420个任务，覆盖多语言与多种方法复杂度；②引入缺陷区分（defect discrimination）度量完整性，克服传统相似度匹配的局限；③提供可运行环境与自动化完整性评估，避免手工检查；④结合LLM生成与专家校准，构建高质量真值后置条件集。

**🔧 技术方法**

使用的技术包括大型语言模型（LLM）进行后置条件与变异生成、Mutation测试、代码覆盖分析、JML/​icontract规范框架、LLM驱动的仓库环境搭建及自动化评测脚本。

**📊 数据集**

数据集来源于121个公开GitHub仓库，挑选出210个Python与210个Java方法，均具有高线覆盖率、完整测试集、手工验证的后置条件集合。

**📈 对比分析**

比较了5个SOTA LLM在三种生成任务（仅NL、仅代码、两者混合）下的正确率与完整率，结果显示完整率远低于正确率（Python最高17%，Java最高43%），且仓库级依赖与方法复杂度进一步拉大两者差距。

**⚠️ 局限性**

局限性包括：①排除了需要未支持的规范构造或无法被现有测试捕获的变异；②仅覆盖高覆盖率方法，可能导致样本偏倚；③未涵盖异常后置、并发等复杂行为；④完整性度量基于Mutation测试，无法保证绝对语义完整。

---

## 161. Population-Aware Imitation Learning in Mean-field Games with Common Noise

**arXiv ID:** 2605.03357 | [PDF](https://arxiv.org/pdf/2605.03357v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 162. DECKER: Domain-invariant Embedding for Cross-Keyboard Extraction and Recognition

**arXiv ID:** 2605.03384 | [PDF](https://arxiv.org/pdf/2605.03384v1)

**作者:** Bikrant Bikram Pratap Maurya `[一作]` (Indraprastha Institute of Information Technology Delhi), Arun Balaji Buduru `[通讯]` (Indraprastha Institute of Information Technology Delhi)

**通讯引用:** 363 | [OpenAlex ID](https://openalex.org/A5014100784)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究键盘声学侧信道攻击，构建覆盖多键盘、多用户、多环境、多麦克风的 HEAR 数据集，并提出域不变的 DECKER 框架，结合 LLM 对预测序列进行纠正；

**💡 创新点**

创新点在于①设计了首个多维度真实场景数据集 HEAR；②提出四阶段域不变训练策略（键盘签名归一化、对抗解耦、跨键盘对比对齐、声学风格随机化）显著提升跨设备泛化；③将大型语言模型融入束搜索解码，进一步恢复完整句子/密码；

**🔧 技术方法**

技术包括声学信号预处理、ECAPA‑TDNN 编码、语音预训练模型（wav2vec2、HuBERT 等）、域对抗训练（GRL）、监督对比学习、键盘签名归一化、声学风格随机化、LLM constrained beam search；

**📊 数据集**

使用 HEAR 数据集：53 名受试者、37 种笔记本键盘，外部麦克风、设备自带麦克风、VoIP 录音，附带性别、左右手、意识等元数据；

**📈 对比分析**

与传统单模态基线（wav2vec2、ECAPA、CoAtNet-S 等）和多模态融合基线（早期/后期融合、跨注意力、加权融合）对比；DECKER 在跨键盘测试中 Top‑1 81.3%（对比 ECAPA 58.1%），跨用户/环境也显著提升；LLM 后处理将句子准确率提升至 70%+（GPT‑4 达 72%+），小型 LLM 也能带来显著提升；

**⚠️ 局限性**

局限性包括：未覆盖连续流式输入的实时分割与识别；极端噪声环境和非传统键盘（触摸屏、柔性键盘）尚未充分测试；对用户个体打字习惯的细微差异影响模型鲁棒性仍待深入；未来需进一步探索多模态侧信道融合与防御方案。

---

## 163. Do LLMs have core beliefs?

**arXiv ID:** 2605.03255 | [PDF](https://arxiv.org/pdf/2605.03255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 164. Adaptive Negative Scheduling for Graph Contrastive Learning

**arXiv ID:** 2605.03076 | [PDF](https://arxiv.org/pdf/2605.03076v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 165. On the Invariants of Softmax Attention

**arXiv ID:** 2605.02907 | [PDF](https://arxiv.org/pdf/2605.02907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 166. Disease Is a Spectral Perturbation

**arXiv ID:** 2605.02949 | [PDF](https://arxiv.org/pdf/2605.02949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 167. PrismAgent: Illuminating Harm in Memes via a Zero-Shot Interpretable Multi-Agent Framework

**arXiv ID:** 2605.02940 | [PDF](https://arxiv.org/pdf/2605.02940v1)

**作者:** Zihan Ding `[一作]` (Sichuan University), Yi Zhang `[通讯]` (Sichuan University)

**通讯引用:** 97698 | [OpenAlex ID](https://openalex.org/A5100388089)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 PrismAgent，一个零样本、可解释的多代理框架，用于检测有害 meme；

**💡 创新点**

创新点在于将检测任务拆解为分析员、调查员、检察官、法官四个角色，模拟刑事案件流程，通过意图重写、检索证据和分步判断实现可解释的零样本推理；

**🔧 技术方法**

采用大型视觉语言模型（如 LLaVA）作为代理，结合 Prompt 工程、模态检索和多阶段推理链；

**📊 数据集**

使用三大公开多模态 meme 数据集：HarM、FHM 与 MAMI；

**📈 对比分析**

与训练基准、现有零样本方法和多代理框架比较，在三数据集上平均提升约 2–3% 的准确率、≈3% 的宏 F1 分数，超越 MIND、GPT‑4o 等模型；

**⚠️ 局限性**

局限包括计算成本高、推理延时长、对极端或稀有 meme 的识别仍受限于 LLM 的能力以及意图重写带来的额外开销。

---

## 168. Design of Memristive Lightweight Encryption For In-Memory Image Steganography

**arXiv ID:** 2605.03494 | [PDF](https://arxiv.org/pdf/2605.03494v1)

**作者:** Seyed Erfan Fatemieh `[一作]` (University of Isfahan), Esmail Zarezadeh `[通讯]` (AmirKabir University of Technology)

**通讯引用:** 215 | [OpenAlex ID](https://openalex.org/A5036090637)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

实现了Trivium和Grain-128a轻量级流密码在基于IMPLY的计算在存储（CIM-A）结构中的实现，并提出了低能耗的移位寄存器重设计；

**💡 创新点**

通过将IMPLY门实现的状态存储逻辑与反相器替代缓冲器的移位寄存器相结合，显著降低了计算步骤和能耗（最高可达44%），同时在图像隐写中验证了其功能；

**🔧 技术方法**

采用memristor的IMPLY门、VTEAM模型、LTspice仿真、Octave行为仿真以及自定义的移位寄存器算法；

**📊 数据集**

在图像隐写实验中使用五张256×256灰度图（boat、cameraman、circuit、rice、walkbridge）嵌入随机密文；

**📈 对比分析**

与传统IMPLY实现进行对比，测量计算步骤、能耗及PSNR；改进版Trivium在10,000位时步数和能耗分别比Grain-128a低18%和22%，PSNR均>65 dB，说明性能提升明显；

**⚠️ 局限性**

仅针对流密码实现，未验证大规模数据吞吐和多通道系统的可扩展性，且缺乏针对实际攻击模型的安全性深入分析。

---

## 169. A Workflow-Oriented Framework for Asynchronous Human-AI Collaboration in Hybrid and Compute-Intensive HPC Environments

**arXiv ID:** 2605.03743 | [PDF](https://arxiv.org/pdf/2605.03743v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 170. Generalization Bounds of Spiking Neural Networks via Rademacher Complexity

**arXiv ID:** 2605.02927 | [PDF](https://arxiv.org/pdf/2605.02927v1)

**作者:** Shao-Qun Zhang `[一作]` (Nanjing University), Zhi-Hua Zhou `[通讯]` (Nanjing University)

**通讯引用:** 62131 | [OpenAlex ID](https://openalex.org/A5100621138)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了常用积分-发放（LIF）模式下的尖峰神经网络（SNN）的泛化性能，并给出了基于经验Rademacher复杂度和覆盖数的严格泛化界；

**💡 创新点**

创新点在于通过分析SNN的函数有界变差特性，利用Grönwall不等式和Sobolev空间，推导出Rademacher复杂度与网络深度、宽度、时间长度、参数范数及样本数相关的更紧泛化界；

**🔧 技术方法**

使用的主要技术包括经验Rademacher复杂度、覆盖数上界、Grönwall不等式、Lipschitz连续性、Sobolev空间理论以及函数有界变差的定义；

**📊 数据集**

实验采用了延迟记忆XOR任务的数据集，模拟不同时间长度、网络宽度和深度的SNN；

**📈 对比分析**

与VC维、传统Rademacher复杂度等方法比较，实验结果表明所给泛化界更为紧凑，且泛化误差随网络深度、宽度变化与理论预测吻合；

**⚠️ 局限性**

局限在于理论对宽度和时间长度对泛化的影响解释仍不够精确，且验证仅限于单一任务，未能覆盖更广泛的应用场景。

---

## 171. Nora: Normalized Orthogonal Row Alignment for Scalable Matrix Optimizer

**arXiv ID:** 2605.03769 | [PDF](https://arxiv.org/pdf/2605.03769v1)

**作者:** Jinghui Yuan `[一作]` (Northwestern Polytechnical University), Feiping Nie `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 44016 | [OpenAlex ID](https://openalex.org/A5003222421)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Nora的矩阵优化器，通过对动量进行逐行正交投影并行行归一化，实现高效、稳定、快速的LLM训练。

**💡 创新点**

创新点在于将行级正交投影与行归一化结合，既保持权重的尺度不变性（稳定性），又利用Transformer Hessian的行块对角优势近似Muon的预处理（效率），且只需两行代码即可实现。

**🔧 技术方法**

主要技术包括行正交投影（row‑wise perpendicular projection）、行归一化（row‑wise normalization）、Muon‑style预处理近似、μP宽度缩放分析以及非凸收敛保证。

**📊 数据集**

使用LLaMA‑style Transformer模型（60M、135M参数）在自动回归语言建模任务上进行实验，验证Nora的效果。

**📈 对比分析**

与Mu­on、Mano、RMNP在相同训练配置下比较，Nora在验证损失和困惑度（perplexity）上均表现最佳，且相较于Newton–Schulz正交化，行归一化的运算速度提升约10–70倍。

**⚠️ 局限性**

局限性包括：仅在中小规模（最高135M）LLaMA模型上验证，缺乏对更大模型或不同网络结构的广泛实验；对权重衰减和超参数的敏感性未做完整系统探测；行块对角假设在某些Transformer变体中可能不完全成立。

---

## 172. A Few-Step Generative Model on Cumulative Flow Maps

**arXiv ID:** 2605.03623 | [PDF](https://arxiv.org/pdf/2605.03623v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 173. Analytic Bridge Diffusions for Controlled Path Generation

**arXiv ID:** 2605.02961 | [PDF](https://arxiv.org/pdf/2605.02961v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 174. Renewables Power the Orbit? Achieving Sustainable Space Edge Computing via QoS-Aware Offloading

**arXiv ID:** 2605.03232 | [PDF](https://arxiv.org/pdf/2605.03232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 175. Learning Reactive Dexterous Grasping via Hierarchical Task-Space RL Planning and Joint-Space QP Control

**arXiv ID:** 2605.03363 | [PDF](https://arxiv.org/pdf/2605.03363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 176. Design and Implementation of BNN-Based Object Detection on FPGA

**arXiv ID:** 2605.03396 | [PDF](https://arxiv.org/pdf/2605.03396v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 177. A Paradigm for Interpreting Metrics and Identifying Critical Errors in Automatic Speech Recognition

**arXiv ID:** 2605.03671 | [PDF](https://arxiv.org/pdf/2605.03671v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 178. Where to Bind Matters: Hebbian Fast Weights in Vision Transformers for Few-Shot Character Recognition

**arXiv ID:** 2605.02920 | [PDF](https://arxiv.org/pdf/2605.02920v1)

**作者:** Gavin Money `[一作]` (University of Alabama), Noorbakhsh Amiri Golilarz `[通讯]` (University of Alabama)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在Omniglot数据集上，探究将Hebbian快权重模块（HFW）嵌入ViT、DeiT和Swin-Tiny Transformer，以提升5‑way少样本分类性能。

**💡 创新点**

关键创新是提出单一“最终阶段”HFW模块的放置策略，并将线性注意力视作显式Hebbian更新，从而在Transformer中实现快速适应。

**🔧 技术方法**

采用可学习的Hebbian快权重模块（含可学习的塑性速率、衰减、门控和Frobenius范数正则化），并在Prototypical Networks框架下进行元学习训练。

**📊 数据集**

使用Omniglot手写字符数据集（1,623类，84×84像素），在5‑way 1/5-shot设置下进行训练、验证与测试。

**📈 对比分析**

与原始ViT、DeiT、Swin基线相比，单一Swin‑Hebbian实现了1‑shot 96.2%与5‑shot 99.2%的准确率，分别比Swin基线提升0.3%与0.5%；而ViT/DeiT的多块HFW实现反而导致性能下降。

**⚠️ 局限性**

限制在于模型从零开始训练，未使用预训练权重；HFW在ViT/DeiT的多块放置导致梯度干扰，且在支持样本增多时改进有限。

---

## 179. Nested array design of extended coprime sets for DOA estimation of non-circular signals

**arXiv ID:** 2605.03856 | [PDF](https://arxiv.org/pdf/2605.03856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 180. Distributed Learning with Adversarial Gradient Perturbations

**arXiv ID:** 2605.03313 | [PDF](https://arxiv.org/pdf/2605.03313v1)

**作者:** Nawapon Sangsiri `[一作]` (Chinese University of Hong Kong), Yufei Tao `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 15875 | [OpenAlex ID](https://openalex.org/A5048609852)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究分布式学习中客户端可能会返回受限但任意偏离真实梯度的情况，提出在 L‑光滑凸函数上实现最优子最优性误差的理论与算法。

**💡 创新点**

创新点在于：① 给出了无上界下无法保证任何误差的不可行性证明；② 在已知解范数上界 R 的前提下，确定了最小可达子最优性误差区间 [R/2, 5R]；③ 提出基于 AGP‑opt 的梯度下降算法，给出最优的查询复杂度上界，并在可接受失败概率时进一步降低查询数。

**🔧 技术方法**

主要技术包括：对抗梯度扰动（AGP）理论、凸光滑函数的分析、梯度下降与早停策略的改进、中心估计（随机采样求均值）以及复杂度证明与不等式推导。

**📊 数据集**

实验使用 LIBSVM 中的 ijcnn1、covtype 和 HIGGS 三个真实数据集，分别配合鲁棒回归（RR）和二元交叉熵（BCE）两种损失函数。

**📈 对比分析**

通过在三种对抗策略（opposing、amplifying、fixed‑direction）下对 AGP‑opt 进行实验，结果显示当误差参数较大时能达到理论上给出的 5R 的误差界；在查询预算实验中发现每次约 100 次查询即可保证稳定的中心估计，从而在不同 K 与 Q 组合下获得与全量查询相近的最终损失。

**⚠️ 局限性**

局限性包括：仅处理凸光滑目标，假设所有客户端都可能受攻击且不考虑异构数据分布；对 R 的先验上界要求在实际应用中可能难以估计；实验中对抗策略为预设，未考虑更复杂的攻击模型。

---

## 181. Evaluating Generative Models as Interactive Emergent Representations of Human-Like Collaborative Behavior

**arXiv ID:** 2605.03855 | [PDF](https://arxiv.org/pdf/2605.03855v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 182. An End-to-End Framework for Building Large Language Models for Software Operations

**arXiv ID:** 2605.02906 | [PDF](https://arxiv.org/pdf/2605.02906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 183. Stop Automating Peer Review Without Rigorous Evaluation

**arXiv ID:** 2605.03202 | [PDF](https://arxiv.org/pdf/2605.03202v1)

**作者:** Joachim Baumann `[一作]` (Stanford University), Dirk Hovy `[通讯]` (Bocconi University)

**通讯引用:** 6705 | [OpenAlex ID](https://openalex.org/A5084505122)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对当前 AI 生成的论文评审进行实证评估，探究其多样性和可游戏性。

**💡 创新点**

发现 AI 评审存在“蜂群效应”（高相似度）和易被“洗稿”游戏的缺陷。

**🔧 技术方法**

使用 GPT‑5.1/5.4、Claude Sonnet 4.6 等 LLM 进行评审生成，并采用文本嵌入与余弦相似度度量。

**📊 数据集**

ICLR 2026 的 75,800 条审稿数据和 60 篇随机论文的 AI 重写实验。

**📈 对比分析**

与人类评审相比，AI 评审在多样性上低于人类、游戏效果显著提升评分；但整体预测准确性低于人类评审。

**⚠️ 局限性**

局限包括仅使用单一提示、依赖第三方 AI 评审标签、模型可扩展性与更新性不足等。

---

## 184. PHANTOM: Polymorphic Honeytoken Adaptation with Narrative-Tailored Organisational Mimicry

**arXiv ID:** 2605.02992 | [PDF](https://arxiv.org/pdf/2605.02992v1)

**作者:** Abraham Itzhak Weinberg `[一作]` `[通讯]` (AI Experts), Abraham Itzhak Weinberg (AI Experts)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一个基于组织上下文的蜜罐令牌生成框架PHANTOM

**💡 创新点**

通过在每个令牌中注入域名、服务命名约定、技术栈习惯等组织特定信息，实现了语义连贯且检测难以识别的高质量蜜罐令牌

**🔧 技术方法**

采用规则驱动的多组件生成流水线，配合自定义的Believability Score评估指标和三类模拟扫描器（正则、熵分析、ML分类）

**📊 数据集**

构建了32个PHANTOM令牌与32个模板生成令牌的数据集，覆盖8种令牌类型和4个行业场景（金融、医疗、国防、电商）

**📈 对比分析**

与模板基线在Believability Score、检测抵抗率和人类接受率等指标上对比，PHANTOM平均提升约20%（B从0.576提升到0.778），检测抵抗率从0.609提升到0.870，100%令牌通过人类实验

**⚠️ 局限性**

局限性包括样本规模有限、组织上下文静态、扫描器模型仅为代表性例子，未覆盖更高级或自定义的攻击者检测手段

---

## 185. Revisiting JBShield: Breaking and Rebuilding Representation-Level Jailbreak Defenses

**arXiv ID:** 2605.03095 | [PDF](https://arxiv.org/pdf/2605.03095v1)

**作者:** Kemal Derya `[一作]` (Worcester Polytechnic Institute), Berk Sunar `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 8407 | [OpenAlex ID](https://openalex.org/A5066592325)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型（LLM）的解禁攻击（jailbreak）防御，提出并评估了一种适应性攻击（JB‑GCG）和一种新型防御（RTV）

**💡 创新点**

创新点在于：①将拒绝方向（refusal direction）与GCG目标结合构造双目标攻击，突破了单一概念门（AND‑gate）检测；②提出多层、多位置拒绝方向指纹，并用马氏距离（Mahalanobis）做 OOD 检测，形成无需解禁示例的鲁棒防御；③在最强白盒自适应攻击下，仅提升 7% 的攻击成功率，同时代价显著增加

**🔧 技术方法**

使用了：Greedy Coordinate Gradient（GCG）框架、反向传播/梯度搜索、单向量/矩阵 SVD、cosine 相似度、马氏距离、Ledoit‑Wolf 估计、BERT/Transformer 词向量

**📊 数据集**

数据集包括 Llama‑3‑8B 模型、HarmBench（危害提示）、Alpaca（无害提示）以及 9 种标准解禁攻击（SAA、GCG、PAIR 等）

**📈 对比分析**

与 JB‑Shield 及现有单层概念检测方法对比，JB‑GCG 在 JB‑Shield 上攻击成功率从 0% 提升至 53.4%（ASR）；RTV 在相同攻击下实现 0.99 AUROC，且在多数标准攻击上实现 100% 检测率；在自适应攻击下，RTV 的 ASR 仅为 7%，显著低于 JB‑GCG 的 53% 并且计算成本提升 13 倍

**⚠️ 局限性**

局限性包括：只在 Llama‑3‑8B 上验证，可能对不同深度/架构的适用性未知；对自然化、迭代生成的攻击（如 PAIR）检测率仅 58%；自适应攻击虽难，但仍能略胜 7%，且未处理模型分布漂移和多样化攻击空间；未探索嵌入空间级自适应攻击

---

## 186. Phoneme-Level Deepfake Detection Across Emotional Conditions Using Self-Supervised Embeddings

**arXiv ID:** 2605.03079 | [PDF](https://arxiv.org/pdf/2605.03079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 187. GeoDecider: A Coarse-to-Fine Agentic Workflow for Explainable Lithology Classification

**arXiv ID:** 2605.03383 | [PDF](https://arxiv.org/pdf/2605.03383v1)

**作者:** Jiahao Wang `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 29058 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 GeoDecider，一种基于多阶段粗细推理与地质细化的岩性分类框架。

**💡 创新点**

创新点在于将地质专家思维拆解为粗分类、工具增强推理、地质一致性细化三步，并通过自适应推理路由仅在不确定样本上调用 LLM。

**🔧 技术方法**

使用了轻量级基分类器、LLM（如 DeepSeek‑R1）、上下文趋势分析、邻近检索、知识库、历史辅助工具以及多视角推理和地质细化模块。

**📊 数据集**

在四个公开井测量数据集（SEAM、Facies、Force、Geolink）上进行实验。

**📈 对比分析**

与传统机器学习、深度学习、LLM 时间序列模型以及基线相比，GeoDecider 在召回率和 F1 上提升约 3–6%（最高约 15%），并且在不使用 LLM 的样本上保持高效。

**⚠️ 局限性**

局限包括对 LLM 的依赖、温度调节敏感、需要大量领域专家协同构建工具，且在严格数据治理环境下的部署仍面临挑战。

---

## 188. Instance-Level Costs for Nuanced Classifier Evaluation

**arXiv ID:** 2605.03135 | [PDF](https://arxiv.org/pdf/2605.03135v1)

**作者:** Kabir Kang `[一作]` (Georgia Institute of Technology), Stephen Mussmann `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 557 | [OpenAlex ID](https://openalex.org/A5034715099)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于实例级误分类成本的评估指标 NEC，并在文本、图像、表格等多领域数据集上验证其有效性，同时探究将成本用于训练的效果。

**💡 创新点**

将人类标注者的投票不一致性转化为每例误差成本，并定义归一化超额成本指标，以更细致地衡量模型错误；同时对成本敏感训练方法进行系统实验。

**🔧 技术方法**

使用实例级成本权重、采样策略、Δ 回归以及预训练模型（RoBERTa、ResNet‑50、HistGradientBoosting 等）进行分类与训练。

**📊 数据集**

在 Jigsaw toxic 评论、Turkey 受伤鸡只、iNaturalist 植物图像、NHANES 血压诊断以及一个可预测成本的合成数据集上开展实验。

**📈 对比分析**

通过将标准交叉熵训练与成本加权训练、采样和回归等方法对比，发现 NEC 往往显著低于错误率；成本加权训练在成本可预测的合成数据上提升明显，但在真实数据上效果有限。

**⚠️ 局限性**

成本来源于投票或阈值距离的假设不一定完全反映真实重要性；成本敏感训练仅在成本可由特征预测时有效，且缺乏对外部经济或法律成本的直接整合。

---

## 189. PerFlow: Physics-Embedded Rectified Flow for Efficient Reconstruction and Uncertainty Quantification of Spatiotemporal Dynamics

**arXiv ID:** 2605.03548 | [PDF](https://arxiv.org/pdf/2605.03548v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 190. SprayCheck: Finding Gray Failures in Adaptive Routing Networks

**arXiv ID:** 2605.03702 | [PDF](https://arxiv.org/pdf/2605.03702v1)

**作者:** Jakob Krebs `[一作]` (Technion), Mark Silberstein `[通讯]` (Technion)

**通讯引用:** 3161 | [OpenAlex ID](https://openalex.org/A5022593894)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种针对使用自适应路由（AR）的分布式机器学习训练集群的被动灰色故障检测与定位系统；

**💡 创新点**

创新点在于利用AR的喷射均衡特性，结合流优先级隔离与无协调的流选择，通过仅监测应用流量即可在不增加网络负载的情况下检测并定位灰色故障；

**🔧 技术方法**

核心技术包括：在Tofino P4交换机上实现流优先级隔离、每棵上行脊柱的包计数、阈值计算与统计假设检验、中心化路径交叉定位；

**📊 数据集**

使用真实的Llama‑3 70B 训练集群收集的收敛集群通信数据、NS‑3仿真中的2‑层脂肪树拓扑以及8‑棵脊柱测试台的实际流量；

**📈 对比分析**

在64棵脊柱网络中，系统在单轮 Llama‑3 训练中即可完美检测到1.5% 损失率的链路，0.5% 损失率在5轮内检测完毕；与传统探测/监控基线相比，误报率为0%、误检率为0%，且对应用性能的额外负载低于0.3%；

**⚠️ 局限性**

局限性包括：仅适用于无损失（lossless）Fabric，无法检测访问链路故障；仅支持2‑层 Fat‑Tree，未覆盖3‑层或更深拓扑；对加权喷射策略支持不足；需要流大小信息；对高损失率或过度拥塞的网络表现仍有待进一步验证。

---

## 191. Complex Equation Learner: Rational Symbolic Regression with Gradient Descent in Complex Domain

**arXiv ID:** 2605.03841 | [PDF](https://arxiv.org/pdf/2605.03841v1)

**作者:** Sergei Garmaev `[一作]` (Intelligent Maintenance and Operations Systems Laboratory), Olga Fink `[通讯]` (Intelligent Maintenance and Operations Systems Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出Complex Equation Learner (CEQL)，一种在复数域优化的符号回归模型，能够稳定学习包含除法、对数、平方根等会导致奇异性或域约束的运算，直接从数据中恢复可解释的有理函数；

**💡 创新点**

创新点在于将网络权重扩展到复数域并在输出上投影到实轴，从而绕过实值梯度消失与奇异点导致的梯度抖动，解决了传统Equation Learner在处理除法及多值函数时的优化路径病态；

**🔧 技术方法**

技术上使用复数参数化的EQL架构、复数梯度下降、稀疏正则化与迭代修剪、以及针对除法的特殊 surrogate 以及对数、平方根的复数实现；

**📊 数据集**

实验数据包括合成的十个带奇异性或域约束的符号表达式（E‑1~E‑10）以及真实的悬臂钢梁频率响应测量数据（频率与损伤程度），训练集为[-2,2]^d，测试集包含内插与外推两部分；

**📈 对比分析**

与SINDy、EQL_÷以及PySR等基线比较，CEQL在所有奇异性任务中实现了与基线相当甚至更低的插值和外推均方误差，并在实测频率响应中成功捕捉两大共振峰的位移，表现出较好的泛化与可解释性；

**⚠️ 局限性**

局限性包括对周期性三角函数的学习仍不稳定、稀疏性约束对结果敏感、以及在多变量/高维场景中对稀疏化策略的进一步优化需求。

---

## 192. DGPO: Distribution Guided Policy Optimization for Fine Grained Credit Assignment

**arXiv ID:** 2605.03327 | [PDF](https://arxiv.org/pdf/2605.03327v1)

**作者:** Hongbo Jin `[一作]` (Peking University), Jiayu Ding `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无价值网络的强化学习框架 DGPO，通过分布偏差作为探索信号，细粒度地把序列级优势重分配到 token 级别。

**💡 创新点**

创新点在于：①用有界 Hellinger 距离替代无界 KL，避免梯度爆炸；②引入基于熵的门控机制过滤幻觉；③在不增加额外价值网络的前提下实现 token 级信用分配。

**🔧 技术方法**

技术细节包括：critic‑free RL、Hellinger 距离、熵门控、softmax 温度重分配、组相对策略优化（GRPO）的改进。

**📊 数据集**

训练使用公开 DAPO‑17K 数据集，评估在 AIME 2024 与 AIME 2025 数学推理基准上，模型为 Qwen2.5‑32B 与 7B。

**📈 对比分析**

与 GRPO、DAPO、FIPO 等基线在 AIME 2024/2025 上对比，DGPO 在 32B 上 Avg@32 达 60.0%/46.0%（超 DAPO 50.0%/38.0%），在 7B 上 Pass@1 达 43.0%/24.0%（超 GRPO 22.0%/18.0%）。

**⚠️ 局限性**

局限性：仍需手工调节温度 τ 与熵门参数 κ；对极长序列的稳定性未完全验证；在其他任务上的泛化能力需要进一步研究。

---

## 193. FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles

**arXiv ID:** 2605.03337 | [PDF](https://arxiv.org/pdf/2605.03337v1)

**作者:** Lucas Yunkyu Lee `[一作]` (Pohang University of Science and Technology), Jaesik Park `[通讯]` (Seoul National University)

**通讯引用:** 9482 | [OpenAlex ID](https://openalex.org/A5100611457)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对4D高斯展平（4D Gaussian Splatting）进行系统性分析，揭示其成功的隐秘因素，并基于这些发现提出改进版方法。

**💡 创新点**

创新点包括：①将先前模糊的实现细节（初始化、密度控制、迁移策略等）明确化；②引入 UFM‑guided 时空初始化、神经速度场（Neural Velocity Field）与门控边缘化（Gated Marginalization）以实现更物理一致的运动和更高的训练可重复性；③结合仿射色彩校正以降低光照漂移。

**🔧 技术方法**

主要技术：高斯展平渲染、分段时间分布建模、MCMC 迁移策略、哈希网格编码的神经速度场、门控函数实现时间分区、光度、SSIM、LPIPS 以及颜色校正损失。

**📊 数据集**

使用 DyNeRF 与 SelfCap 两大多视角动态数据集进行评估。

**📈 对比分析**

与重现的 FreeTimeGS 基线对比，方法在 DyNeRF 上 PSNR 提升约 0.9 dB，SelfCap 上 0.7 dB；同时显著降低了训练过程的波动性，改善了运动一致性与物理可解释性。

**⚠️ 局限性**

局限性：方法仍主要验证于结构化室内场景，未充分评估在更复杂、非结构化环境下的鲁棒性；门控与神经速度场的超参数选择仍需经验调优；实现复杂度较高，可能限制大规模实时部署。

---

## 194. Semantically Enriching Investor Micro-blogs for Opinion-Aware Emotion Analysis: A Practical Approach

**arXiv ID:** 2605.03092 | [PDF](https://arxiv.org/pdf/2605.03092v1)

**作者:** Gaurav Negi `[一作]` (University Of Galway), Paul Buitelaar `[通讯]` (University Of Galway)

**通讯引用:** 4505 | [OpenAlex ID](https://openalex.org/A5025406029)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究通过使用大语言模型对StockEmotions数据集进行语义化意见图谱增强，并将其融入图神经网络，提升投资者情绪分类准确性。

**💡 创新点**

创新点在于将统一意见概念UOC本体与情感标签结合，构建细粒度意见图谱，并在编码器上加入GNN进行语义融合。

**🔧 技术方法**

使用了LLM驱动的注释管线、UOC本体、Transformer编码器（BERT、RoBERTa）、图注意力网络（GATv2）以及多种特征融合策略。

**📊 数据集**

使用的主要数据集是由10,000条StockTwits评论构成的StockEmotions，并通过LLM补充意见语义后生成的增强版。

**📈 对比分析**

通过与标准Transformer基线和大型生成式LLM（GPT-5、Qwen-3.5-35B）在十二类情感上的对照实验，GNN增强模型在大部分类别上实现了1–5个百分点的macro‑F1提升。

**⚠️ 局限性**

局限性包括对“含糊”“信念”等类别的性能仍低，LLM在零样本情感识别中表现欠佳，以及图谱构建对文本长度和多意图句子可能受限。

---

## 195. Understanding Self-Supervised Learning via Latent Distribution Matching

**arXiv ID:** 2605.03517 | [PDF](https://arxiv.org/pdf/2605.03517v1)

**作者:** Fabian A Mikulasch `[一作]` (Friedrich Miescher Institute for Biomedical Research), Friedemann Zenke `[通讯]` (Friedrich Miescher Institute for Biomedical Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个统一的分布匹配框架，将自监督学习视为在潜在空间匹配先验模型，推导出可识别的预测性SSL方法并给出了基于Kalman滤波的非线性贝叶斯滤波模型。

**💡 创新点**

通过将对齐（对数似然）与均匀性（熵）统一为KL散度目标，阐明MI最大化的冗余性，给出识别性的证明，并提出利用Kalman预测器实现不确定性估计的全新预测式SSL。

**🔧 技术方法**

使用KL分布匹配、熵估计（KDE、kNN、参数化）、对齐与均匀性损失、Kalman滤波器预测器以及理论证明（线性与非线性ICA、可识别性）。

**📊 数据集**

在公开图像数据集CIFAR‑10、CIFAR‑100、ImageNet‑100上进行线性探针实验，并在合成高维时序数据以及鼠脑海马尖峰时序上验证模型。

**📈 对比分析**

通过对比不同熵估计器、是否最大化MI、以及预测器类型（CPC、stop‑grad、Kalman）在线性探针准确率和表示质量上进行评估，结果表明MI最大化对性能影响不大，预测性SSL可与CPC取得相当的表现，并成功恢复潜在空间。

**⚠️ 局限性**

依赖于可逆编码器假设、熵估计的准确性与可优化性，Kalman预测器需线性假设，识别性证明要求高斯预测误差，计算复杂度和对非线性映射的适用性仍有限。

---

## 196. Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe

**arXiv ID:** 2605.03677 | [PDF](https://arxiv.org/pdf/2605.03677v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 197. DACP: A Scientific Data Access and Collaboration Protocol

**arXiv ID:** 2605.03411 | [PDF](https://arxiv.org/pdf/2605.03411v1)

**作者:** Zhihong Shen `[一作]` (Computer Network Information Center, Chinese Academy of Sciences), Changfa Lu `[通讯]` (Computer Network Information Center, Chinese Academy of Sciences)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了科学数据访问与协作协议DACP，并实现了参考服务器faird和客户端SDK，展示了在大规模科学数据场景中的可行性和性能优势。

**💡 创新点**

创新点在于引入统一的Streaming Data Frame模型、任务导向的抽象、服务器端谓词下推、惰性评估以及基于DAG的现场计算和反向供给机制，实现跨域协作与数据可发现。

**🔧 技术方法**

采用了Apache Arrow Flight传输、Arrow列式内存布局、零拷贝、REST‑style GET/PUT/COOK方法、token身份认证以及与Spark、PyTorch、HuggingFace的生态集成。

**📊 数据集**

评估使用Yelp Open Dataset的结构化子集以及ImageNet验证集的混合文件集进行实验。

**📈 对比分析**

通过与传统FTP协议在相同硬件和网络环境下的对比实验，DACP在结构化数据上提升3.1–5.36倍，在混合数据上提升1.21倍，上传性能保持对称，优于FTP 13–27%的下降。

**⚠️ 局限性**

限制在于实现仍集中在faird实现，DAG定义复杂度高，尚未标准化，且对极大规模分布式调度的鲁棒性和安全性需进一步验证。

---

## 198. ADS: Random Sampling of Occupancy Functions using Adaptive Delaunay Scaffolding

**arXiv ID:** 2605.03235 | [PDF](https://arxiv.org/pdf/2605.03235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 199. SigLoMa: Learning Open-World Quadrupedal Loco-Manipulation from Ego-Centric Vision

**arXiv ID:** 2605.03846 | [PDF](https://arxiv.org/pdf/2605.03846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 200. Self-Mined Hardness for Safety Fine-Tuning

**arXiv ID:** 2605.03226 | [PDF](https://arxiv.org/pdf/2605.03226v1)

**作者:** Prakhar Gupta `[一作]` (University of Michigan), Donghua Zhang `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套自监督的安全微调管线，利用模型自身生成的rollout评估提示难度并自动挑选最具攻击性的提示进行训练。

**💡 创新点**

创新点在于：①通过模型自身的rollout自我挖掘提示硬度，避免人工标签；②采用1:1 adversarially‑framed benign混合训练来缓解过度拒绝问题。

**🔧 技术方法**

使用了LoRA参数高效微调、三位安全评判器投票构造硬度评分、硬度排序/随机抽样策略，以及adversarially‑framed benign对齐训练。

**📊 数据集**

主要数据集包括WildJailbreak、WildGuardMix、ClearHarm等公开攻击/安全数据集，在Llama‑3‑8B‑Instruct和Llama‑3.2‑3B‑Instruct上进行实验。

**📈 对比分析**

通过与纯对抗、随机、控制以及混合版本在相同训练步数/计算量下对比，纯对抗将WildJailbreak ASR降至≈2%/1%但拒绝率升至>80%；混合版本将ASR提升至≈5%/7%且adv‑benign拒绝率降至≈30–50%，实现较优的ASR–overrefusal Pareto改进。

**⚠️ 局限性**

局限性包括仅在Llama‑3族上验证，缺乏跨模型泛化；混合后adv‑benign拒绝率仍高达30–70%；硬度阈值设定依赖后验分析，且未充分利用always‑jailbroken提示或更细粒度的硬度定义。

---

## 201. Adaptive Data Compression and Reconstruction for Memory-Bounded EEG Continual Learning

**arXiv ID:** 2605.03085 | [PDF](https://arxiv.org/pdf/2605.03085v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 202. ZeRO-Prefill: Zero Redundancy Overheads in MoE Prefill Serving

**arXiv ID:** 2605.02960 | [PDF](https://arxiv.org/pdf/2605.02960v1)

**作者:** Zhaoyuan Su `[一作]` (University of Virginia), Yuxiong He `[通讯]` (Snowflake AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种专门针对Mixture-of-Experts（MoE）LLM的 Prefill‑only 服役系统，利用按权重聚集专家、异步专家权重流、前端前缀感知调度和饱和阈值强制，消除传统 MoE 预填阶段的同步激活路由、通信冗余与调度不均等三重瓶颈。

**💡 创新点**

核心创新在于：①将专家位置与激活路由解耦，改用按权重聚集专家；②将专家权重流与前向计算完全重叠，形成零冗余通信；③设计基于硬件/模型物理极限的饱和阈值 T 并在前端实现前缀匹配与真实 FLOPs 计量，实现前后端协同优化；④提供可选 KV‑cache‑free 模式与权重 CPU‑DRAM 混合卸载，进一步压缩 HBM 用量。

**🔧 技术方法**

技术细节包括：MoE 模型与 DP/EP 并行策略、NVLink D2D AllGather + PCIe H2D 混合权重预取、Prefix‑aware routing（最长块匹配）、基于 FLOPs 的负载跟踪、物理饱和阈值 T 的计算与前端应用、KV‑cache‑free 计算模式、与 vLLM 的集成。

**📊 数据集**

使用了六个公开基准的聚合 Prefill‑only 工作负载：MoralStories、MMLU、BoolQ、IMDB、QuALITY、ArXiv Classification，并在此基础上合成了无前缀重用的合成工作负载（不同上下文长度）。

**📈 对比分析**

通过与五种常用的分布式 MoE 并行策略（DP+EP、DP+TP、TP+EP、TP+TP、PP+PP）以及 PrefillOnly 的改进版本进行对比，评估指标包括吞吐量（tokens/s）、MFU、HBM 使用率和精度。实验结果表明：在 Qwen3‑235B‑A22B 上，系统在 1–8 张 GPU、BF16/FP8 不同硬件上均实现了 1.35–1.37 倍的吞吐提升，合成无前缀重用工作负载时可达 1.59 倍；MFU 在 30–36% 之间，且部署范围从“≥4 GPU”扩大到“1–8 GPU”。

**⚠️ 局限性**

主要限制包括：① 需要高速互连（NVLink/PCIe）才能保证权重流在计算窗口内完全重叠；② 饱和阈值 T 仅在启动时校准，严重的工作负载漂移可能导致短暂的同步瓶颈；③ 仅适用于吞吐量导向的批量 Prefill‑only 场景，对低延迟交互式推理或无专家堆叠的密集模型无效；④ KV‑cache‑free 与前缀路由的收益受工作负载前缀共享度影响，随机工作负载时效果有限。

---

## 203. Globally adaptive and locally regular point discretization of curved surfaces

**arXiv ID:** 2605.03803 | [PDF](https://arxiv.org/pdf/2605.03803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 204. VEBench:Benchmarking Large Multimodal Models for Real-World Video Editing

**arXiv ID:** 2605.03276 | [PDF](https://arxiv.org/pdf/2605.03276v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 205. MK-ResRecon: Multi-Kernel Residual Framework for Texture-Aware 3D MRI Refinement from Sparse 2D Slices

**arXiv ID:** 2605.03432 | [PDF](https://arxiv.org/pdf/2605.03432v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 206. PatRe: A Full-Stage Office Action and Rebuttal Generation Benchmark for Patent Examination

**arXiv ID:** 2605.03571 | [PDF](https://arxiv.org/pdf/2605.03571v1)

**作者:** Qiyao Wang `[一作]` (Shenzhen Institute of Advanced Technology), Min Yang `[通讯]` (Shenzhen Institute of Advanced Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了PatRe benchmark，覆盖完整专利审查生命周期的Office Action与反驳生成任务。

**💡 创新点**

首个全阶段多轮审查生成基准，并引入LLM-as-a-Judge评估框架，细粒度法律与技术评估。

**🔧 技术方法**

采用大语言模型（GPT、Gemini、LLaMA、Qwen、Gemma等）与检索模拟、引用筛选等技术。

**📊 数据集**

使用USPTO公开数据库收集的480条完整审查记录，涵盖8个IPC章节。

**📈 对比分析**

与多种LLM（专有与开源）对比，专有模型在OA与反驳上表现更好，OpenAI GPT-5-mini在决策准确率和律师评估分上最高；Open源模型在文本相似度高但法律逻辑弱。

**⚠️ 局限性**

受限于对先前文献的引用准确性、对法律条文的把握不稳、对逻辑推理不足，导致模型在拒绝预测上偏高且缺乏实质性依据。

---

## 207. Set-like operations on propositional logic programs

**arXiv ID:** 2605.03613 | [PDF](https://arxiv.org/pdf/2605.03613v1)

**作者:** Christian Antić `[一作]` `[通讯]` (Vienna University of Technology), Christian Antić (Vienna University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一套基于集合运算的逻辑程序代数框架，构建了一系列集合式运算（如体联、体交、补集、差集、对称差、幂集）以及相应的分解定理，使得任意（或最简）Horn程序可拆解为Krom（仅含单体体）的子程序，从而可在子程序层面恢复或逼近原程序的最小模型。

**💡 创新点**

创新点在于：①首次将集合运算的概念迁移至逻辑程序并定义相应的规则级操作；②证明了在此框架下的分解定理，展示了Krom程序的可组合性与最小模型的可恢复性；③构造了若干同构、同余与幺半群结构，为后续的程序合成、变换与类比推理提供了代数基础。

**🔧 技术方法**

使用了逻辑程序的语义学（van Emden–Kowalski算子与最小模型）、集合代数与半环、模态逻辑中的同余与约简等理论工具，并通过证明方法展示了运算的封闭性、分配律、幺元与补运算等性质。

**📊 数据集**

该工作为理论研究，未使用任何实验数据集；所有结果均为形式证明。

**📈 对比分析**

本文未进行实验或数值比较；性能评估以理论证明为主，说明在Krom子程序上可实现关联运算的可组合性与最小模型的恢复，而对一般程序则仅给出近似上界。

**⚠️ 局限性**

局限性包括：①仅针对无函数符号的命题Horn程序，未覆盖带有函数符号或归一化约束的程序；②对非Horn或带有否定的稳定/答案集程序的推广仍需进一步研究；③在实际应用中，虽然框架提供了代数操作，但在复杂程序的拆解与合成效率仍未得到实验验证。

---

## 208. Erase Persona, Forget Lore: Benchmarking Multimodal Copyright Unlearning in Large Vision Language Models

**arXiv ID:** 2605.03547 | [PDF](https://arxiv.org/pdf/2605.03547v1)

**作者:** JuneHyoung Kwon `[一作]` (Chung-Ang University), YoungBin Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 1903 | [OpenAlex ID](https://openalex.org/A5016930939)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CoVUBench——首个针对多模态版权去学习的基准，包含合成的版权概念（角色、商标）以及多样化视觉样式与视觉-文本问答；同时设计了面向版权持有者与模型部署者的双重评估指标。

**💡 创新点**

创新点在于①构建了无真实版权风险的合成版权内容生成管道；②引入视觉多样性与领域变换，评估概念泛化忘记；③提出跨模态、跨域的问答测试与双重指标（遗忘效能与模型实用性）；④揭示当前去学习方法在多模态场景下的“模态差距”与性能极端化。

**🔧 技术方法**

使用文本+图像合成技术（Nano Banana API）结合 persona blueprint；LoRA 微调 + AdamW 优化；五种去学习算法（Gradient Ascent, Gradient Difference, KL 正则化, Direct Preference Optimization, Negative Preference Optimization）。

**📊 数据集**

数据集：20 个合成版权概念（10 角色+10 商标），共 2,420 条视觉问答对；划分为训练、遗忘、保留与测试集，包含不同视觉布局与领域表现。

**📈 对比分析**

在 LLaVA-Phi-3B 与 LLaVA-1.5-7B 两模型上对五种去学习方法进行评估，使用 EM、ROUGE、Divergence（遗忘效能）与 Fluency、Specificity、Capability（实用性）等指标。结果显示，梯度法在遗忘效能上表现最佳，但导致实用性大幅下降；偏好优化方法保持实用性但几乎不忘记内容；总体上没有方法能兼顾两大需求。

**⚠️ 局限性**

局限性：①合成数据未必覆盖所有真实版权情况；②仅评估角色与商标，其他视觉版权类型缺失；③现有去学习算法未能有效处理跨模态概念，存在“模态差距”；④大模型去学习成本高，方法可扩展性待验证。

---

## 209. Predicting Euler Characteristics and Constructing Topological Structure Using Machine Learning Techniques

**arXiv ID:** 2605.02947 | [PDF](https://arxiv.org/pdf/2605.02947v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 210. Quantum Hierarchical Reinforcement Learning via Variational Quantum Circuits

**arXiv ID:** 2605.03434 | [PDF](https://arxiv.org/pdf/2605.03434v1)

**作者:** Yu-Ting Lee `[一作]` (National Taiwan University), Fu-Chieh Chang `[通讯]` (MediaTek Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一个混合量子‑经典的option‑critic架构，将变分量子电路（VQC）替换特征提取器、选项价值函数、终止函数和内部策略，并在CartPole与Acrobot环境中对其性能进行系统评估。

**💡 创新点**

系统评估了量子组件在HRL中的不同位置，发现仅量子化特征提取器即可显著提升性能并节省参数；同时揭示了量子选项价值函数是性能瓶颈，并通过消融实验给出电路深度、纠缠和可学习输入缩放等设计原则。

**🔧 技术方法**

使用变分量子电路（PennyLane实现）、经典MLP、option‑critic学习算法、经验回放与目标网络，并将量子与经典模块结合。

**📊 数据集**

Gymnasium中的CartPole与Acrobot两大经典连续状态、离散动作任务。

**📈 对比分析**

与纯经典option‑critic、单层MLP基线及随机策略比较；在CartPole中Hybrid_F获得约2.95倍相对奖励、比24‑神经元经典基线多得66%参数；在Acrobot中Hybrid_F将惩罚降低约46%；量子选项价值导致性能接近随机。

**⚠️ 局限性**

仅在简单环境验证，未测试更复杂任务；未阐明选项价值瓶颈根源；仅使用单一学习算法，未考虑噪声影响；量子模拟成本高，训练时间长。

---

## 211. When to Think, When to Speak: Learning Disclosure Policies for LLM Reasoning

**arXiv ID:** 2605.03314 | [PDF](https://arxiv.org/pdf/2605.03314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 212. When Safety Geometry Collapses: Fine-Tuning Vulnerabilities in Agentic Guard Models

**arXiv ID:** 2605.02914 | [PDF](https://arxiv.org/pdf/2605.02914v1)

**作者:** Ismail Hossain `[一作]` (University of Texas at El Paso), Sajedul Talukder `[通讯]` (University of Texas at El Paso)

**通讯引用:** 420 | [OpenAlex ID](https://openalex.org/A5014369148)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在无恶意数据的域特化细调过程中，已对齐的安全守卫模型（Guard Models）如何失去安全性，并提出了一种新的正则化方法 FW-SSR 以在细调时保持安全几何结构。

**💡 创新点**

创新点在于：①将安全对抗子空间（Safety Subspace）从激活统计中提取并作为训练时的锚点；②引入基于 Fisher 信息的曲率加权正则化与自适应 λ 调度；③证明安全几何（CKA、Fisher Score）比传统输出指标更能预测安全失效。

**🔧 技术方法**

使用了 SVD 提取安全子空间、CKA 对比、Fisher 信息近似、LoRA 微调（QLoRA）以及自适应梯度冲突调度等技术。

**📊 数据集**

数据集包括：Alpaca 指令跟随数据（用于细调）、构造的 80 条安全探测样本（40 有害、40 友善）、以及 20 条有害提示用于攻击成功率评估。

**📈 对比分析**

与原始模型和无防护细调模型对比，FW-SSR 在 Granite Guardian 上恢复 75% 的拒绝率（CKA 0.98），在 WildGuard 上将攻击成功率从 17.1% 降低到 3.6%；对比指标主要是拒绝率、合规率、模糊率、CKA、Fisher Score 等，表现明显优于基线。

**⚠️ 局限性**

局限性包括：仅在单一细调实验环境下验证，未考虑多代理管线中的级联失效；安全子空间的 Fisher 近似忽略了方向间相关性；安全探测样本覆盖不完整，可能导致某些类别的正则化效果不足。

---

## 213. A Universal Reproducing Kernel Hilbert Space from Polynomial Alignment and IMQ Distance

**arXiv ID:** 2605.03262 | [PDF](https://arxiv.org/pdf/2605.03262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 214. GEM-FI: Gated Evidential Mixtures with Fisher Modulation

**arXiv ID:** 2605.03750 | [PDF](https://arxiv.org/pdf/2605.03750v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 215. A Validated Prompt Bank for Malicious Code Generation: Separating Executable Weapons from Security Knowledge in 1,554 Consensus-Labeled Prompts

**arXiv ID:** 2605.03179 | [PDF](https://arxiv.org/pdf/2605.03179v1)

**作者:** Richard J. Young `[一作]` (University of Nevada Las Vegas), Gregory D. Moody `[通讯]` (University of Nevada Las Vegas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一的恶意代码拒绝评估数据集，将四个公开基准的提示聚合并通过五个大模型的共识判定，将提示划分为可执行代码（CODE）与安全知识（KNOWLEDGE），最终得到1554条CODE提示和388条KNOWLEDGE提示；

**💡 创新点**

创新点在于将“武器”（可执行恶意软件）与“知识请求”作为核心二元分类轴，并采用多厂商多模型共识协议来验证标签，提供了高可靠性、标准化的评估基准；

**🔧 技术方法**

技术手段包括：使用Claude Sonnet、GPT-5.3-Codex、Gemini 3 Flash、GLM-5、Qwen3-Coder-Next五个大型语言模型并行判定，采用三票多数共识、Fleiss' κ统计并配合10,000次bootstrap计算95%置信区间；

**📊 数据集**

数据集来源于RMCBench、MalwareBench、CySecBench和harmful_behaviors（AdvBench）四个公开基准，经过去重、正则预过滤后得到3,133条候选提示；

**📈 对比分析**

通过共识协议得到的标签显示69.3%的提示达到5/5一致率，Fleiss' κ=0.876（95%CI 0.862–0.888），表明几乎完美的一致性；该方法相比传统混合基准的单一模型评估，能消除提示类型混杂，提升拒绝率测量的可信度；

**⚠️ 局限性**

局限性包括：共识基于LLM而非人工标注，可能与人类判断存在偏差；仅限英语提示；处理的是单轮提示，未覆盖多轮交互场景；共识阈值保守，极端模糊提示被排除；当源基准更新时需重新验证标签。

---

## 216. ReLeaf: Benchmarking Leaf Segmentation across Domains and Species

**arXiv ID:** 2605.03784 | [PDF](https://arxiv.org/pdf/2605.03784v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 217. SoDa2: Single-Stage Open-Set Domain Adaptation via Decoupled Alignment for Cross-Scene Hyperspectral Image Classification

**arXiv ID:** 2605.03371 | [PDF](https://arxiv.org/pdf/2605.03371v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 218. Evaluating Retrieval-Augmented Generation for Explainable Malware Analysis

**arXiv ID:** 2605.03140 | [PDF](https://arxiv.org/pdf/2605.03140v1)

**作者:** Jayson Ng `[一作]` (New York Institute of Technology), Amin Milani Fard `[通讯]` (New York Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估检索增强生成（RAG）在使用 VirusTotal 报告进行恶意软件解释时的效果，探讨其对解释质量的影响。

**💡 创新点**

发现当已有结构化证据足够时，RAG往往降低解释质量，并提出将恶意软件解释视为信号提取任务而非知识检索，给出安全工作流的设计建议。

**🔧 技术方法**

使用 LlamaIndex + ChromaDB + OpenRouter 进行 RAG；采用 all‑MiniLM‑L6‑v2、text‑embedding‑3‑large 等嵌入；对比 GPT‑OSS‑21B、DeepSeek‑R1、GPT‑5.1 等 LLM；评估指标为 BERTScore。

**📊 数据集**

MalGPT 语料库中 1,702 条 VirusTotal JSON 报告，包含 API 调用、注册表修改、网络行为等结构化指标。

**📈 对比分析**

在相同提示下分别用仅 JSON 与 JSON+检索上下文两种方式生成解释，使用 BERTScore 进行对比；结果显示大多数模型在加入 RAG 后 BERTScore 降低，最高无 RAG 得分为 DeepSeek‑R1 0.8617，RAG 方案通常在 0.818–0.854 之间。

**⚠️ 局限性**

仅测试了文本检索且未微调；评估仅基于 BERTScore，未覆盖人类可读性或安全实用性；实验规模有限，未验证在不同恶意软件类型或更复杂检索策略下的通用性。

---

## 219. SERE: Structural Example Retrieval for Enhancing LLMs in Event Causality Identification

**arXiv ID:** 2605.03701 | [PDF](https://arxiv.org/pdf/2605.03701v1)

**作者:** Zhifeng Hao `[一作]` (Guangdong University of Technology), Boyan Xu `[通讯]` (Guangdong University of Technology)

**通讯引用:** 1436 | [OpenAlex ID](https://openalex.org/A5034536387)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于结构化示例检索的框架SERE，用来提升大语言模型在事件因果识别(ECI)任务中的性能。

**💡 创新点**

创新点包括：①将概念路径、句法结构和因果模式三种结构信号统一起来；②设计概念路径编辑距离和句法树编辑距离两种结构相似度度量；③引入基于LLM的因果模式过滤器；④通过结构化检索减少因果幻觉。

**🔧 技术方法**

技术手段：大语言模型的few-shot推理；ConceptNet+Contriever-msmarco对事件进行概念映射并求最短路径；spaCy构建依存句法树并计算树编辑距离；编辑距离和树编辑距离的加权融合；LLM生成因果模式并进行正则表达式匹配。

**📊 数据集**

使用的数据集包括：EventStoryLine(ESC)、Causal-TimeBank(CTB)以及MAVEN-ERE。

**📈 对比分析**

实验与Base、CoT、Dr.ECI等基线对比，SERE在ESC、CTB、MAVEN-ERE三个数据集上都取得最高的F1分数；在Fine-tuning情境下与CPATT相近，证明了方法的稳健性。

**⚠️ 局限性**

限制：①仅评测少数主流LLM（未涉及开源模型）；②仅在示例检索阶段利用三种结构信息，未探索其他可能的结构特征。

---

## 220. A Fast Model Counting Algorithm for Two-Variable Logic with Counting and Modulo Counting Quantifiers

**arXiv ID:** 2605.03391 | [PDF](https://arxiv.org/pdf/2605.03391v1)

**作者:** Shixin Sun `[一作]` (Jilin University), Yi Chang `[通讯]` (China)

**通讯引用:** 25829 | [OpenAlex ID](https://openalex.org/A5100388267)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的加权一阶模型计数算法（），用于处理带有计数量词的两变量片段和模计数扩展。

**💡 创新点**

通过直接在斯科特标准形式上操作，避免了多阶段的约简技术，从而提高了算法的效率和可扩展性。

**🔧 技术方法**

使用了增量WFOMC算法，该算法通过逐步扩展域来计算模型计数，并引入了更丰富的状态表示来跟踪计数量词的信息。

**📊 数据集**

在多个基准数据集上进行了实证评估，包括k-正则图、k-正则l-着色图、k-正则有向图和Barabási–Albert图等。

**📈 对比分析**

与现有的算法和最先进的命题模型计数器相比，算法在运行时间和可扩展性上均表现出显著的改进，尤其是在处理较大域时。

**⚠️ 局限性**

算法的局限性在于其对输入句子的结构和复杂性敏感，可能在某些特定情况下仍然面临性能瓶颈。

---

## 221. Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks

**arXiv ID:** 2605.03338 | [PDF](https://arxiv.org/pdf/2605.03338v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 222. eOptShrinkQ: Near-Lossless KV Cache Compression Through Optimal Spectral Denoising and Quantization

**arXiv ID:** 2605.02905 | [PDF](https://arxiv.org/pdf/2605.02905v1)

**作者:** Pei-Chun Su `[一作]` (Yale University), Pei-Chun Su `[通讯]` (Yale University)

**通讯引用:** 90 | [OpenAlex ID](https://openalex.org/A5040721135)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 eOptShrinkQ，一种将 KV 缓存块视为 spiked 随机矩阵的压缩方法：先用 eOptShrink 自动选秩并进行最优奇异值衰减去除低秩共享上下文，再用 TurboQuant 对残差进行无偏标量量化；

**💡 创新点**

创新点在于：①将 KV 缓存块拆解为低秩信号和全秩残差并证明残差满足薄壳与去中心化特性；②利用 eOptShrink 的 BBP 阈值与 D-变换实现自动秩选取与最优奇异值衰减，恢复 TurboQuant 的等距假设；③无需子组划分或异常处理，显著节省约 1bit/元素并降低内积偏差；

**🔧 技术方法**

技术核心包括随机矩阵理论（spiked 模型、BBP 阈值、Marchenko–Pastur、D-变换）、eOptShrink 最优奇异值衰减算法、TurboQuant 的随机正交旋转+Lloyd–Max 标量量化（可选 QJL 校正）；

**📊 数据集**

实验数据集为 Llama‑3.1‑8B‑Instruct 与 Ministral‑8B‑Instruct 的 KV 缓存，评估基准为 LongBench（16 英文任务）和多针检索（needle‑in‑a‑haystack）任务；

**📈 对比分析**

与 KIVI、TurboQuant (TQMSE)、TQprod、SVD_r=1+TQ 等基线对比；在每 head 的 L2 误差、IP 偏差与方差上，eOptShrinkQ 在 2‑3 bits/元素下实现 17‑25% L2 误差，几乎节省 1bit；在 LongBench 上 2.2 bits/元素达到 47.4/48.3 分，接近 FP16，超过 3.0 bits 的 TQprod；多针检索上 2.2 bits 记忆率 0.98‑0.99，优于 FP16；

**⚠️ 局限性**

限制在于需按 128‑token 块批量处理，最近 127 token 仍以 FP16 存储；未实现在线 SVD 更新，对极短生成序列不友好；目前仅在 8B 模型验证，扩展到更大模型与硬件加速仍需进一步研究。

---

## 223. Some Improved Results on Fair and Balanced Graph Partitions

**arXiv ID:** 2605.03238 | [PDF](https://arxiv.org/pdf/2605.03238v1)

**作者:** Vignesh Viswanathan `[一作]` (University of Massachusetts Amherst), Vignesh Viswanathan `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 615 | [OpenAlex ID](https://openalex.org/A5065675484)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在无向图（社会网络）中将节点划分为大小相等的 k 个部分，并同时满足近似公平性（近似无嫉妒性和近似核心）的问题。

**💡 创新点**

创新点在于首次证明存在既近似无嫉妒又近似核心的划分，并给出相对较好的近似因子；对 k=2 的情况进一步突破核心近似到 φ≈1.618，且提供多项式时间近似算法；同时提出在放宽平衡约束下可高效求解。

**🔧 技术方法**

主要技术包括概率方法、Chernoff 边界、Lovász 局部引理（含算法化实现）以及局部搜索求解最小割的近似核心。

**📊 数据集**

论文为理论研究，无使用具体数据集；所有结果均在理论上给出上界/下界。

**📈 对比分析**

与先前工作相比，本文在一般 k 情况下的无嫉妒近似因子从 O(√(n/k ln k)) 提升到 O(max{√Δ, k²} ln n)，核心近似从 (2k-1,1) 改为 (k+√k, O(k^{5/2} ln n))，k=2 时核心从 2 降到 1.618。实验或实验比较未给出，仅以理论证明为主。

**⚠️ 局限性**

局限性：求解严格平衡划分的多项式时间算法仍未找到；近似因子虽改善但仍非最优；对大 Δ 或 k 的情况近似仍显著；算法仅在平衡约束放宽后才高效。

---

## 224. Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective

**arXiv ID:** 2605.03373 | [PDF](https://arxiv.org/pdf/2605.03373v1)

**作者:** Zhe Li `[一作]` (Rochester Institute of Technology), Haibo Yang `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1609 | [OpenAlex ID](https://openalex.org/A5013868893)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文通过一阶分析推导了零阶梯度下降（ZO SGD）的单步学习动力学，揭示了经验神经切线核（eNTK）是学习行为的关键因素。

**💡 创新点**

创新点在于将ZO学习动力学与eNTK及Johnson-Lindenstrauss引理联系，证明ZO近似误差主要受模型输出尺寸影响，而非参数维度，从而解释了ZO方法在百亿参数LLM微调中的可扩展性。

**🔧 技术方法**

主要技术包括零阶梯度估计、经验神经切线核分析、Johnson-Lindenstrauss Lemma以及对随机低维投影的理论研究。

**📊 数据集**

文中未给出具体数据集，主要讨论LLM微调任务的理论分析。

**📈 对比分析**

通过理论推导与已有ZO实验结果对比，作者说明ZO损失收敛速率与参数维度无关，符合实际LLM微调表现，但未给出数值性能指标。

**⚠️ 局限性**

局限在于仅分析单步动力学与理论模型，缺乏全面的实验验证；且假设eNTK近似成立，未讨论高阶非线性或噪声影响。

---

## 225. FINER-SQL: Boosting Small Language Models for Text-to-SQL

**arXiv ID:** 2605.03465 | [PDF](https://arxiv.org/pdf/2605.03465v1)

**作者:** Thanh Dat Hoang `[一作]` (Griffith University), Quoc Viet Hung Nguyen `[通讯]` (Griffith University)

**通讯引用:** 9028 | [OpenAlex ID](https://openalex.org/A5051219382)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FINER‑SQL 框架，利用稠密执行反馈训练小型语言模型（≤3B）以生成更准确的 Text‑to‑SQL 查询。

**💡 创新点**

创新点包括：①使用记忆奖励（Memory Reward）对生成的推理轨迹进行语义对齐，提升推理一致性；②使用原子奖励（Atomic Reward）对 SQL 结构进行细粒度匹配；③在小模型上采用 Group Relative Policy Optimization（GRPO）避免价值网络，获得稳定高效的 RL 训练。

**🔧 技术方法**

技术手段：大模型知识蒸馏生成推理库；对推理进行嵌入检索；稠密奖励设计（格式、执行、原子、记忆）；GRPO 训练；多候选生成 + 多数投票推理。

**📊 数据集**

数据集：Spider 与 BIRD 两大 Text‑to‑SQL 公开基准。

**📈 对比分析**

对比实验显示：3B 模型在 BIRD Dev 上 67.73% EX、Spider Dev 上 85.0% EX，接近 14‑70B 大模型，推理延迟仅 5.57 s/样本，显著低于现有开源与 API 方案，且能在单 12–24 GB GPU 上部署。

**⚠️ 局限性**

局限性：仍依赖大模型蒸馏和跨数据库检索；对极其复杂查询的鲁棒性有限；RL 训练需要大量 rollouts；记忆库维护成本与查询效率受制；实验范围仅限 Spider 与 BIRD，未验证在其它领域的泛化。

---

## 226. MASRA: MLLM-Assisted Semantic-Relational Consistent Alignment for Video Temporal Grounding

**arXiv ID:** 2605.03398 | [PDF](https://arxiv.org/pdf/2605.03398v1)

**作者:** Ran Ran `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 112307 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MASRA框架，在训练阶段利用多模态大语言模型（MLLM）生成事件级描述和clip级字幕作为文本先验，提供更稠密的监督，改进视频时序定位（VTG）性能。

**💡 创新点**

创新点包括：①利用MLLM生成事件级和clip级文本先验，缓解跨模态语义鸿沟；②设计事件语义-时序对齐（ESTA）与局部关系一致性对齐（LRCA）两种互补对齐机制；③引入Decoupled Alignment Interaction、Semantic‑Guided Enhancement及Second‑Order Relational Attention等模块，提升跨模态交互与局部一致性；④MLLM仅用于训练，推理时无额外开销。

**🔧 技术方法**

技术手段包括多模态Transformer、VQ‑style代码簿、对齐损失、关系一致性损失、二阶关系注意力；使用CLIP文本编码器、SlowFast/CLIP视频编码器；GPT‑5（或同类MLLM）生成文本先验；AdamW优化、标准VTG损失组合。

**📊 数据集**

实验数据集为QvHighlights、Charades‑STA和TACoS，使用SlowFast+CLIP特征或VGG特征。

**📈 对比分析**

与M‑DETR、QD‑DETR、TR‑DETR、CG‑DETR、RGTR、KDA等主流VTG方法对比，MASRA在R@1、mAP、mIoU等指标上普遍取得最优或第二名，尤其在QVHighlights的mAP提升显著；在Charades‑STA和TACoS上亦保持领先。

**⚠️ 局限性**

局限性包括：训练时依赖MLLM生成文本先验，增加训练成本；事件划分与文本质量受MLLM表现影响；对极长视频或极细粒度事件的定位仍面临挑战；仅在训练阶段使用MLLM，推理时未获得实时性能提升。

---

## 227. Text-Conditional JEPA for Learning Semantically Rich Visual Representations

**arXiv ID:** 2605.03245 | [PDF](https://arxiv.org/pdf/2605.03245v1)

**作者:** Chen Huang `[一作]` (Apple), Josh Susskind `[通讯]` (Apple)

**通讯引用:** 3033 | [OpenAlex ID](https://openalex.org/A5033404184)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Text-Conditional JEPA，利用图像标题在 Joint-Embedding Predictive Architecture（JEPA）中进行细粒度文本调制，降低特征预测不确定性，学习语义丰富且对文本敏感的视觉表示。

**💡 创新点**

①将跨注意力（cross‑attention）与多层视觉特征调制结合，形成细粒度文本条件；②引入稀疏性与层间一致性正则，自动学习图像‑文本对应关系；③仅靠特征预测即可完成预训练，完全不依赖对比损失。

**🔧 技术方法**

JEPA基础框架；文本编码使用预训练 T5；多层跨注意力调制；稀疏性正则、层间一致性约束；多标题聚合与最大池化；ViT（B/L/H）视觉编码器；使用 ImageNet、CC12M、YFCC15M 进行预训练。

**📊 数据集**

ImageNet‑1k、ImageNet‑21k；大规模图文数据 CC12M 与 YFCC15M（并用 ShareGPT4V 合成标题）；在对比实验中还参考 WebLI、CLIP、SigLIP 等公开数据集。

**📈 对比分析**

与多类 MIM（MAE、data2vec、CAPI）、invariance‑based SSL（DINO、iBOT、DINOv2）以及 contrastive 视觉‑语言模型（CLIP、BLIP、MaskCLIP、SPARC、DreamLIP、GroupViT）进行对比；在分类、检测、分割、captioning、VQA 等任务中均表现优于或逼近最新模型，尤其在细粒度视觉理解和多模态推理上取得显著提升。

**⚠️ 局限性**

①需要在预训练阶段提供图像标题（合成或真实），缺乏文本时效果受限；②仅使用特征预测，未结合对比学习，可能在极大规模无文本数据下表现不如传统方法；③模型对标题生成质量敏感，可能继承合成文本中的偏见；④训练时需额外的文本编码与注意力计算，计算成本略高。

---

## 228. Counting Small Balanced (p,q)-bicliques in Signed Bipartite Graphs

**arXiv ID:** 2605.03603 | [PDF](https://arxiv.org/pdf/2605.03603v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 229. From Static Analysis to Audience Dissemination: A Training-Free Multimodal Controversy Detection Multi-Agent Framework

**arXiv ID:** 2605.02939 | [PDF](https://arxiv.org/pdf/2605.02939v1)

**作者:** Zihan Ding `[一作]` (Sichuan University), Yi Zhang `[通讯]` (Sichuan University)

**通讯引用:** 97698 | [OpenAlex ID](https://openalex.org/A5100388089)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个训练无关的多智能体框架 AuDisAgent，用动态观众传播模拟来检测短视频及其评论中的争议性。

**💡 创新点**

创新点包括：①将争议检测从静态特征学习转化为观众传播的动态模拟；②设计了 Screening、Viewing Panel、Arbitration 三类智能体，并通过一致性门控实现不确定样本的进一步讨论；③针对冷启动问题引入评论引导策略，从历史相似视频迁移评论。

**🔧 技术方法**

使用了多模态预训练模型（mPLUG‑Video、GLM‑4‑9B 等）、大型语言模型进行推理、文本编码模型 bge‑large‑zh‑1.5 做相似检索，以及多智能体交互的逻辑流程。

**📊 数据集**

在公开的 MMCD（Multimodal Controversy Detection）数据集上进行实验，该数据集包含 1 万余条中文短视频及其评论。

**📈 对比分析**

与 13 种基线方法（包括标准提示、Chain‑of‑Thought、Tree‑of‑Thought、AgentMCD 等）进行对比，AuDisAgent 在丰富评论和稀缺评论两种情境下均实现 F1 分别提升约 1.7% 与 2.0%，准确率提升约 1.3% 与 1.1%，显著优于现有 SOTA。

**⚠️ 局限性**

局限性：①目前仅针对短视频设计，难以直接迁移到长视频；②依赖标题等元数据进行检索，若元数据不足会影响性能；③整体性能受限于底层大模型的表达与推理能力，未来需引入更强的多模态理解模型。

---

## 230. An Identity for Catalan Numbers via Restricted Dyck Paths

**arXiv ID:** 2605.03567 | [PDF](https://arxiv.org/pdf/2605.03567v1)

**作者:** Antonio Bernini `[一作]` (University of Firenze), Elisa Pergola `[通讯]` (University of Firenze)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究高度≤h且在高度h-1处不出现k-1个连续谷的Dyck路径，并推导其计数与生成函数

**💡 创新点**

给出一种通用的ECO构造和递推生成规则，推导出新的Catalan数恒等式，系数为常数

**🔧 技术方法**

使用ECO方法、递推、生成函数以及解析式构造

**📊 数据集**

无实验数据集，纯理论推导

**📈 对比分析**

无实验比较，未给出性能指标，全部为理论证明

**⚠️ 局限性**

仅限于高度≤h且特定谷口约束，未推广至任意高度或其他局部结构，缺乏实验验证

---

## 231. One Sequence to Segment Them All: Efficient Data Augmentation for CT and MRI Cross-Domain 3D Spine Segmentation

**arXiv ID:** 2605.03098 | [PDF](https://arxiv.org/pdf/2605.03098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 232. MHPR: Multidimensional Human Perception and Reasoning Benchmark for Large Vision-Languate Models

**arXiv ID:** 2605.03485 | [PDF](https://arxiv.org/pdf/2605.03485v1)

**作者:** Kangkang Wang `[一作]` (Baidu), Shengzhao Wen `[通讯]` (Baidu)

**通讯引用:** 79 | [OpenAlex ID](https://openalex.org/A5075717206)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出多维度人类感知与推理基准 MHPR，并构建多层数据结构 (C-RD, SFT-D, RL-D, T-D) 与自动化 Caption/VQA 生成管道 ACVG，随后使用该基准对 Qwen2.5-VL-7B 进行监督微调与强化学习，评估其在细粒度属性与高阶语义推理方面的表现。

**💡 创新点**

① 结合个体、多人与人-物交互三大场景，统一评估感知与推理；② 设计四层数据体系并实现 ACVG 的多模型投票与结构化纠错，显著提升数据质量与可扩展性；③ 通过错误案例驱动的 RL 数据，针对空间关系、细粒度感知与推理深度等关键瓶颈进行精细化提升。

**🔧 技术方法**

使用大规模视觉语言模型 Qwen2.5-VL-7B（含多模态 LLM 与视觉编码器）进行监督微调与强化学习；ACVG 采用多模型投票、结构化差异检测、冲突纠正与最终融合；VAGP 通过三模型一致性筛选生成高质量 VQA；RL 通过奖励函数强化空间一致性与细粒度判定。

**📊 数据集**

基础语料来自 HumanCaption-HQ-311K、HumanCaption-10M 与 COYO-700M，经过 ACVG 处理后构成 C-RD、SFT-D、RL-D 与 T-D 四类数据集，用于训练、验证与测试。

**📈 对比分析**

在多子集（单人、多人、人-物）下与 Qwen2.5-VL-7B-Instruct、InternVL3-8B 对比，Qwen2.5-VL-7B-SFT 的平均准确率达到约 81.4%（或 81.88%），在大多数子集超越对手，验证了 MHPR 与 ACVG 在提升细粒度感知与推理方面的有效性。

**⚠️ 局限性**

模型仍受空间关系辨识与视角判断的限制；细粒度感知易受遮挡与小目标影响；推理深度校准不够精确；在部分高难度子集表现仍不尽理想，且对跨域迁移与更复杂多模态推理的鲁棒性尚需进一步提升。

---

## 233. A Domain Incremental Continual Learning Benchmark for ICU Time Series Model Transportability

**arXiv ID:** 2605.03832 | [PDF](https://arxiv.org/pdf/2605.03832v1)

**作者:** Ryan King `[一作]` (Texas A&M University), Bobak J. Mortazavi `[通讯]` (Texas A&M University)

**通讯引用:** 4328 | [OpenAlex ID](https://openalex.org/A5040096171)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了基于 ICU 时序数据的领域增量学习基准，用以评估将大医院训练的模型迁移到不同地区小型医院的可转移性。

**💡 创新点**

创新点包括：①构建跨地区分布差异明显的 DIL benchmark；②提出结合 EWC 与数据重放的新方法，并对其进行改进；③在四个临床预测任务上系统评估该方法与传统方法的性能。

**🔧 技术方法**

使用的技术主要是 BiLSTM / LSTM 时序模型；领域增量学习技术包括 EWC、传统数据重放、改进的重放以及两者结合的组合方法。

**📊 数据集**

使用的数据集为：MIMIC‑III（源域）和 eICU（按南、中、西、北四区划分为目标域），四个任务分别为在院死亡、去压、LOS 与病种分类。

**📈 对比分析**

通过 PSAd（每域平均性能）对比基线、EWC、Replay、调整后的 Replay 以及组合方法，结果显示组合方法在大多数任务和地区上均优于单独方法，提升幅度约 0.5–2%（如 AUC‑ROC、宏平均 AUC 等）。

**⚠️ 局限性**

局限性：需要在源与目标之间共享部分患者数据，存在隐私与合规挑战；实验仅涉及两域（MIMIC‑III→eICU）且未尝试无记忆或更复杂的模型扩展；未来需在更多域、更多任务和更严格隐私保护条件下验证。

---

## 234. Brainrot: Deskilling and Addiction are Overlooked AI Risks

**arXiv ID:** 2605.03512 | [PDF](https://arxiv.org/pdf/2605.03512v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 235. Neural networks as fuzzy logic formulas

**arXiv ID:** 2605.03064 | [PDF](https://arxiv.org/pdf/2605.03064v1)

**作者:** Damian Heiman `[一作]` (Mathematics Research Centre), Esko Turunen `[通讯]` (Mathematics Research Centre)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本论文研究了通过模糊逻辑（Rational Pavelka Logic 及其扩展、以及Łukasiewicz logic 与 Product Logic 结合的逻辑）对普通前馈神经网络（权重与偏置为有理数，激活函数为ReLU）以及更一般的多项式环结构进行逻辑表征与可表达性分析。

**💡 创新点**

创新点在于：①首次给出Rational Pavelka Logic 及其扩展与神经网络的可表达性等价性；②提出一系列受限子句（如 _≤d、_≤1 等）来精确描述神经网络与逻辑之间的表达能力；③构造了从逻辑公式到神经元（proto-neuron）以及从神经元到逻辑公式的递归翻译，证明了两者在缩放后具有同等可表达性。

**🔧 技术方法**

使用的技术主要是：模糊逻辑的语法与语义（包括真值常数、乘积联结符、产品蕴含），多项式环与 ReLU 函数的代数结构，缩放映射 _k 将实数区间映射到 [0,1]，以及构造性的归纳证明与等价性转换。

**📊 数据集**

论文为理论性研究，无使用实际数据集。

**📈 对比分析**

由于研究为纯理论性质，未进行实验或性能对比；因此无法给出性能数值或对比结果。

**⚠️ 局限性**

限制包括：需要将输入/输出值缩放到 [-k,k] 区间以适配逻辑真值；对权重与偏置仅限于有理数（但可推广到实数）；逻辑表达力受子句约束，无法覆盖所有神经网络；缺乏经验验证。

---

## 236. LLM-ADAM: A Generalizable LLM Agent Framework for Pre-Print Anomaly Detection in Additive Manufacturing

**arXiv ID:** 2605.03328 | [PDF](https://arxiv.org/pdf/2605.03328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 237. Forward--Backward Green Cosine Geometry for Directed Community Detection and Overlap Expansion

**arXiv ID:** 2605.03318 | [PDF](https://arxiv.org/pdf/2605.03318v1)

**作者:** Duy Hieu Do `[一作]` `[通讯]` (Institute of Mathematics Vietnam Academy of Science and Technology), Duy Hieu Do (Institute of Mathematics Vietnam Academy of Science and Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于“前向–后向 Green 余弦几何”的方法，用来在有向图中检测非重叠和重叠社区，并给出了两种具体算法：Di‑Green‑FB‑cosine‑KMeans（离散聚类）和 Di‑Green‑FB‑Cosine‑Overlap（重叠扩展）。

**💡 创新点**

核心创新点包括：
1) 用中心化 Green 传播矩阵（去除站稳分布基线）取代原始到达时间向量，解决了余弦相似度对平移不变性的敏感性；
2) 结合前向（原向）和后向（边反转）Green 向量，构造“前向–后向”坐标，兼顾发射和接收两种导向性；
3) 在同一几何空间上完成非重叠聚类与重叠扩展，避免了不同阶段使用不同特征的混杂问题；
4) 通过理论证明显示该坐标形成正定核，且在理想块模型下在同一社区内保持完全相同。

**🔧 技术方法**

技术方法包括：
- 有向随机游走的转移矩阵与跳转（teleportation）正则化；
- Green 运营符号（中心化的 fundamental matrix）和其截断的扩散部分；
- 前向–后向 Green 余弦相似度（结合两个方向的内积，权重 λ=0.5）；
- 球面 K‑means（Spherical K‑means）在该余弦空间中做聚类；
- 基于社区内部余弦分布的自适应阈值，按平均最高 η% 的邻居余弦得分决定是否加入额外社区。

**📊 数据集**

实验数据集：
- 合成：
  • 有向高斯分割模型（heterogeneous Gaussian partition）
  • 有向度校正块模型（degree‑corrected block model）
  • 有向重叠种植分区模型（overlapping planted‑partition）
- 实际网络：
  • email‑Eu‑core、CollegeMsg、wiki‑Vote、p2p‑Gnutella04/08、soc‑sign‑bitcoinalpha‑pos、soc‑sign‑bitcoinotc‑pos、wiki‑RfA‑pos。

**📈 对比分析**

比较方法：
- 方向谱嵌入（oPCA、rPCA、D‑SCORE、D‑SCOREq）
- 流量法（Directed Infomap）
- 先前的方向余弦重叠方法（Di‑Cosine Overlap）
- 模型基础重叠方法（CoDA）

性能结果：
- 在所有合成基准上，Di‑Green‑FB‑cosine‑KMeans 的 NMI/ARI/PairF1/方向模量均超过或等于最强对手（尤其在 μ≥0.20 时差距明显）；
- 在真实网络上，该方法在 8/12 个网络中获得最高 Q_dir，平均 Q_dir 约 0.373，明显优于其它谱和流量基准；
- 重叠扩展阶段，在 oracle 初始化下，Di‑Green‑FB‑Cosine‑Overlap 的 ONMI/OverlapF1/总分均超过 Di‑Cosine Overlap，尤其在困难混合度 μ=0.30 时提升显著。

**⚠️ 局限性**

限制与未来工作：
- 计算复杂度与存储成本：需要 O(n^2) 的 Green 向量和 O(n^2) 的相似度矩阵，适用于中等规模图（n≤10⁴）; 需要进一步压缩或近似（截断特征、随机投影、近似最近邻）。
- 需要设定跳转参数 α、截断长度 T、权重 λ；虽在实验中对其敏感性低，但在不同数据集仍需调优。
- 目前不自动估计社区数 K，需外部预先给定或进行网格搜索。
- 对极度稀疏或高度结构化的有向网络（如大规模社交图）效果尚未系统评估。
- 未来可探索多尺度 Green 坐标、非线性核化以及直接优化方向模量的变体。

---

## 238. Computing Thiele Rules on Interval Elections and their Generalizations

**arXiv ID:** 2605.03067 | [PDF](https://arxiv.org/pdf/2605.03067v1)

**作者:** Dimitris Avramidis `[一作]`, Adrian Vetta `[通讯]` (McGill University)

**通讯引用:** 3132 | [OpenAlex ID](https://openalex.org/A5027727214)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了Thiele规则在结构化选民-候选人区间域（VCI）和线性一致域（LC）下的可解性，给出了多项式时间算法，并证明了树表示域下的NP‑硬性。

**💡 创新点**

证明即使约束矩阵不具完全单调性，相关LP仍拥有整数最优解；首次将VCI域扩展至LC域，并揭示LC严格包含VCI；同时给出树表示域的NP‑完全证明。

**🔧 技术方法**

使用线性规划松弛与完全单调性分析、矩阵重排与区间图论、LP基极点性证明、贪心移位操作以及图形互补关系等技术。

**📊 数据集**

主要为理论证明，无实验数据集；若有实验则使用人工构造的示例矩阵进行演示。

**📈 对比分析**

通过证明最优解可在多项式时间内获得，算法复杂度为多项式；在树表示域则通过归约证明为NP‑完全，无法高效求解。

**⚠️ 局限性**

结果仅适用于VCI和LC域，无法直接推广到更一般的区间域；LP整数性证明依赖特定重排，算法实现较为复杂；缺乏对大规模实际实例的实验评估。

---

## 239. Sensitivity Analysis of Tactical Wireless Network Design Under Realistic Operational Constraints

**arXiv ID:** 2605.03072 | [PDF](https://arxiv.org/pdf/2605.03072v1)

**作者:** Wissem Ahmed Zaid `[一作]` (Polytechnic Montreal), Alain Hertz `[通讯]` (Polytechnic Montreal)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对战术无线网络设计进行系统的灵敏度分析，涵盖主干节点选择、PMP后继限制、天线波束宽度、天线技术以及目标函数权重等结构、技术与建模参数。

**💡 创新点**

首次将三类参数（结构、技术、建模）统一到同一优化框架中，并通过统计显著性检验揭示阈值效应与规模相关的技术转折点，提供针对不同目标的参数调优指导。

**🔧 技术方法**

采用Tabu Search元启发式进行树形拓扑优化；使用Friedman和Wilcoxon检验评估配置差异；构建基于最小、平均吞吐量与情景权重的多目标目标函数；采用多波束天线几何建模与频率/通道分配。

**📊 数据集**

合成基准数据集：10、15、20、30节点四种规模，每种规模10个实例，共40个实例，全部满足树形拓扑与节点度约束。

**📈 对比分析**

通过非参数统计检验对比各配置与基准，检验显著性；发现主干节点采用度+中心性组合策略、PMP后继上限为10、24波束并配合OmniSwitch以及多波束天线在大规模网络中均显著提升吞吐量，性能提升幅度可达数十个百分点。

**⚠️ 局限性**

局限性在于仅使用合成数据、仅考虑树形拓扑、特定的天线与通道模型，缺乏真实部署环境的验证；参数调优结果对不同规模或不同物理约束的泛化性有限；计算成本随网络规模急剧增加未作深入评估。

---

## 240. Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving

**arXiv ID:** 2605.03375 | [PDF](https://arxiv.org/pdf/2605.03375v1)

**作者:** Shi Qiu `[一作]` (Xiamen University), Yiming Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6836 | [OpenAlex ID](https://openalex.org/A5100395360)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一种GPU‑中心化的SSD后端KV缓存方案，消除了GPU HBM与NVMe SSD之间关键数据与I/O控制路径中的CPU干预，提升了大上下文LLM推理的可扩展性和性能。

**💡 创新点**

创新点包括：① GPU原生对象抽象与GPU文件池，使KV缓存与GPU直接交互；② GPU io_uring实现异步GPU直接对象I/O；③ Slack‑aware I/O调度器精准避开GPU资源争用和SSD读写冲突；④ 结合GPU文件映射与SGL减少PRP开销，实现高吞吐。

**🔧 技术方法**

技术手段涵盖GPU-centric object store（基于GeminiFS）、GPU io_uring、SGL/PRP、P2P内存映射表、GPU文件池、SM分区（I/O域与计算域）、Slack‑aware调度、vLLM集成、Mooncake分布式协调。

**📊 数据集**

使用的实验数据集为LEval和LooGLE，模型为Llama3‑8B和GLM‑4‑9B‑Chat‑1M（支持1M上下文）。

**📈 对比分析**

与现有LMCache（GDS、DRAM）对比，TTFT下降78.3%（SLO约束下），请求率提升2×，GPU停顿几乎为零；在成本上比LMCache‑SSD低约27%。在长上下文场景下，性能接近DRAM级别但容量近乎无限。

**⚠️ 局限性**

主要局限包括：远程KV检索路径仍由CPU介导，未充分利用GPU直接RDMA；极高前缀重用（>96k）时仍受SSD带宽限制；实现依赖GPU 5.0/6.0 PCIe与NVLink，受限于特定硬件。

---

## 241. GPUBreach: Privilege Escalation Attacks on GPUs using Rowhammer

**arXiv ID:** 2605.03812 | [PDF](https://arxiv.org/pdf/2605.03812v1)

**作者:** Chris S. Lin `[一作]` (University of Toronto), Gururaj Saileshwar `[通讯]` (University of Toronto)

**通讯引用:** 689 | [OpenAlex ID](https://openalex.org/A5071514906)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文演示了利用 GPU Rowhammer 攻击实现 GPU 侧权限升级，通过篡改 GPU 页表实现任意 GPU 内存读写，进一步泄露 cuPQC 加密密钥、破坏 ML 代码以隐蔽降低模型准确率，并通过 DMA 触发 CPU 侧的系统权限提升。

**💡 创新点**

创新点包括：①首次将 Rowhammer 攻击应用于 NVIDIA GPU 的页表，完成 GPU 侧权限升级；②提出针对 GPU 页表的“填充与摆放”技术；③发掘基于 UVM 申请的时间侧信道，用于精准定位页表分配；④实现 GPU 侧权限升级后跨越 IOMMU 触发 CPU 侧 root 权限，构成完整系统攻击链。

**🔧 技术方法**

使用的技术主要有：GPURowhammer（GPUHammer 攻击模式）、统一虚拟内存（UVM）内存管理与页面大小控制、对 GPU 页表的位翻转与页面帧号（PFN）篡改、DMA 与 GPU 驱动缓冲区越界利用，以及 Linux 内核的特权提升链。

**📊 数据集**

实验基准与数据集包括：RTX A6000 GPU、cuPQC（MLKEM512）加密库、PyTorch 预训练模型（AlexNet、VGG16、ResNet50 等）、大语言模型推理服务，所有实验均在 48GB GDDR6 内存配置下进行。

**📈 对比分析**

与现有 CPU Rowhammer 攻击比较，GPU 权限升级仅需约 20 秒完成，密钥泄露成功率 4.4%（单一 key-exchange 运行），模型准确率从 57–80% 降至 ~0.1%（仅 <6.6% 时延提升），CPU 侧 root 提升可在 <1 秒内完成，整体性能损耗低于 10% 的正常运行时延。

**⚠️ 局限性**

局限性：①要求 ECC 被禁用且 UVM 可用；②仅对使用 GDDR6（如 Ampere、Ada、Hopper、Blackwell 系列）GPU 有效，HBM/HBM2 或 vGPU 环境不适用；③需要对 GPU 页表分布有精确掌握，若驱动改变内存分配策略则攻击难度大；④对攻击者必须拥有对 GPU 的代码执行权限，并且需要充分的 GPU 内存以完成页表填充与侧信道观测。

---

## 242. A Deeper Dive into the Irreversibility of PolyProtect: Making Protected Face Templates Harder to Invert

**arXiv ID:** 2605.03857 | [PDF](https://arxiv.org/pdf/2605.03857v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 243. Local Truncation Error-Guided Neural ODEs for Large Scale Traffic Forecasting

**arXiv ID:** 2605.03386 | [PDF](https://arxiv.org/pdf/2605.03386v1)

**作者:** Xiao Zhang `[一作]` (Zhengzhou University), Mingliang Xu `[通讯]` (Zhengzhou University)

**通讯引用:** 9456 | [OpenAlex ID](https://openalex.org/A5081346568)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种结合连续ODE与离散补偿的混合动力系统（LTE-ODE），通过利用数值求解器的局部截断误差（LTE）作为无监督前向诱导信号，在出现突发交通冲击时动态激活离散补偿分支，从而捕捉交通流的宏观连续节律与微观离散冲击。

**💡 创新点**

创新点在于：①将LTE视为正向诱导而非需惩罚的误差，消除了传统物理约束导致的“注意力崩溃”；②通过动态空间注意力掩模仅在冲击节点激活离散补偿，既保持高精度连续演化又具备非线性分岔能力；③零额外神经函数评估（NFE）的嵌入式求解器实现高效计算，配合稀疏激活实现可扩展性。

**🔧 技术方法**

技术方法包括：双重求解器（Euler 与 RK2）计算LTE；Sigmoid映射生成空间注意力掩模；离散补偿子网络 gϕ；基于图神经网络的双流连续向量场 fθ；无正则化、纯端到端的任务损失；嵌入式Runge-Kutta实现高效 NFE。

**📊 数据集**

在四个公开交通流数据集上进行评测：PEMS07、PEMS08、CA、England，使用 MAE、MAPE、RMSE 等指标。

**📈 对比分析**

与13个最先进基线（统计模型、GNN、Transformer等）对比，LTE-ODE 在大多数数据集和指标上均实现了最优或近优表现，特别是 RMSE 上显著降低；同时在训练/推理时间上位于 Pareto 最优前沿，比大多数 Transformer 方案快约两倍；集成步数的 ablation 也展示了灵活的硬件部署能力。

**⚠️ 局限性**

局限性主要包括：①仍需在非常大规模图上进一步验证扩展性；②对异常检测的阈值选择可能依赖数据分布；③模型的可解释性尚未完全开放，特别是离散补偿分支的物理意义仍需深入研究。

---

## 244. Robust Path Tracking for Vehicles via Continuous-Time Residual Learning: An ICODE-MPPI Approach

**arXiv ID:** 2605.03260 | [PDF](https://arxiv.org/pdf/2605.03260v1)

**作者:** Shugen Song `[一作]` (Southeast University), Chengyan Zhao `[通讯]` (Kyushu Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在 Model Predictive Path Integral (MPPI) 控制框架中引入连续时间残差学习器 (ICODE) 的鲁棒路径跟踪方法

**💡 创新点**

将输入共变神经ODE作为残差学习模块嵌入 MPPI，保持物理一致性与时间连续性，显著降低模型误差并抑制控制抖动

**🔧 技术方法**

采用 MPPI 采样控制、输入共变神经ODE、RK4 积分、迭代数据聚合、Softplus 激活函数以及 Adam 优化器

**📊 数据集**

在高保真仿真环境中收集随机探索与任务特定轨迹数据，并用正弦扰动模拟外部环境，作为训练集

**📈 对比分析**

与标准 MPPI 在椭圆、正弦波、Figure‑8 三条轨迹上对比，使用 RMSE 与控制抖动概率分布评估；ICODE‑MPPI 将位置误差降低约 69%，并显著平滑转向指令

**⚠️ 局限性**

对方向角误差有轻微增加，纵向加速度控制需要更频繁调节；仅在仿真中验证，缺乏真实车辆实验与动态障碍物避障能力

---

## 245. Mechanical Conscience: A Mathematical Framework for Dependability of Machine Intelligenc

**arXiv ID:** 2605.03847 | [PDF](https://arxiv.org/pdf/2605.03847v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 246. Flow Matching on Symmetric Spaces

**arXiv ID:** 2605.03588 | [PDF](https://arxiv.org/pdf/2605.03588v1)

**作者:** Francesco Ruscelli `[一作]` (University of Heidelberg), Rita Fioresi `[通讯]` (University of Bologna)

**通讯引用:** 1026 | [OpenAlex ID](https://openalex.org/A5089505566)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了在黎曼对称空间（如球面、Grassmannian）上训练流匹配模型的通用框架，通过将目标分布映射到 Lie 代数的 p 子空间并在该向量空间上执行欧氏流匹配，从而实现了对曲率空间的生成建模。

**💡 创新点**

创新点在于利用 Cartan 分解将流匹配问题线性化到 Lie 代数的 p 子空间，避免了预度量或显式 geodesic 计算，并自动保持对称空间的等变性；同时将连续归一化流（CNF）的训练转化为无仿真、可扩展的欧氏流匹配。

**🔧 技术方法**

主要技术包括流匹配（Flow Matching）与连续归一化流、Cartan 分解、Lie 群指数映射、欧氏流匹配、对称空间的投影与重投影（从 p 到 G，再到 M = G/K），以及使用 MLP/ConcatSquash 网络实现可微分向量场。

**📊 数据集**

实验数据集包括：一维棋盘分布映射到球面 S²；Grassmannian 数据集 DW4、LJ13、LJ55（分别为 Gr(2,4)、Gr(3,13)、Gr(3,15)）。

**📈 对比分析**

通过计算测试集负对数似然（NLL）与参考文献进行比较；结果显示在 Grassmannian 上的 NLL 为 -1.13、-74.13、-303.66，证明方法在可扩展性和性能上表现良好，虽然与传统 CNF 在同一空间上的直接对比难度较大。

**⚠️ 局限性**

局限性包括：指数映射需要可达；仅适用于具有 Cartan 分解的对称空间（主要是紧型）；在非紧型或指数映射不可逆的空间中可能失效；以及在 Grassmannian 上生成噪声时必须避免使用梯度不良的矩阵对数，限制了噪声生成方式。

---

## 247. Observability for Post-Quantum TLS Readiness: A Multi-Surface Evidence Framework

**arXiv ID:** 2605.02978 | [PDF](https://arxiv.org/pdf/2605.02978v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 248. Memorization In Stable Diffusion Is Unexpectedly Driven by CLIP Embeddings

**arXiv ID:** 2605.02908 | [PDF](https://arxiv.org/pdf/2605.02908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 249. Lifting to tensors when compiling scientific computing workloads for AI Engines

**arXiv ID:** 2605.03566 | [PDF](https://arxiv.org/pdf/2605.03566v1)

**作者:** Nick Brown `[一作]` (EPCC at University of Edinburgh), Gabriel Rodriguez-Canal `[通讯]` (EPCC at University of Edinburgh)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文提出一种基于OpenMP标记的Fortran循环自动卸载到AMD NPU的编译流水线，通过将循环提升为MLIR张量表示，完成从CPU到AIE的高层映射；

**💡 创新点**

创新点在于将OpenMP循环语义提升到张量层，利用张量的高层信息实现自动决策（分解、向量化、数据流），大幅简化代码复杂度并兼容现有Fortran代码；

**🔧 技术方法**

使用技术包括MLIR（tensor、tosa、hlaie dialects）、Flang前端、LLVM IR、AIE-MLIR、XRT、OpenMP、以及AMD提供的aie、aievec、aiex dialects；

**📊 数据集**

评估数据集为六个通用算子（softmax、relu、saxpy、dot product、l2norm、gemm）以及两个科学计算示例（MONC PW advection、Shallow Water Equation），所有测试均采用float32（gemm使用bf16输入）；

**📈 对比分析**

与手写AIE实现和CPU多核对比，NPU实现的性能与手写相当或略优，能耗显著降低；在CPU+NPU混合执行时，吞吐量提升约40%且能耗比单CPU低；

**⚠️ 局限性**

局限性包括：不支持atomic OpenMP pragma，需手动实现向量化（通过外部aie-translate工具），目前AIE对float32/ int32 的原生支持有限，且实现仅针对Fortran+OpenMP，尚未覆盖更广泛的循环模式。

---

## 250. Revisiting General Map Search via Generative Point-of-Interest Retrieval

**arXiv ID:** 2605.03397 | [PDF](https://arxiv.org/pdf/2605.03397v1)

**作者:** Dong Chen `[一作]` (Beijing Jiaotong University), Zhenfeng Zhu `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 9293 | [OpenAlex ID](https://openalex.org/A5101867605)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种面向通用地图搜索的生成式POI检索框架GenPOI，能够将用户查询、位置信息、历史行为等异构上下文统一序列化后输入LLM，实现对大规模POI数据库的精准检索。

**💡 创新点**

创新点包括：1) Geo‑Semantic POI Tokenization，将语义与地理信息融合为结构化PID；2) PID Trie受限生成，保证生成结果有效且与数据库一致；3) 基于Geohash的空间感知搜索空间剪枝，进一步提升空间相关性与推理效率。

**🔧 技术方法**

主要技术手段包括：LLM（Qwen3‑0.6B/1.7B）自回归生成、GeoPE地理位置嵌入、RQ‑VAE量化生成Semantic ID、Geohash编码、PID Trie约束、基于距离的近似估计器、Beam Search等。

**📊 数据集**

实验数据来自腾讯地图的两大工业级数据集：TMap‑S（5座城市、125万POI）和TMap‑L（全国范围、1162万POI），并使用真实用户搜索日志构造历史交互序列。

**📈 对比分析**

与传统检索模型（DSSM、PALM、HGAMN）以及生成式基线（TIGER、GNPR‑SID）对比，GenPOI在Recall@K和NDCG@K上均领先，最大提升达13.84%（Recall@20），验证了生成式+空间建模的有效性。

**⚠️ 局限性**

局限性在于：仍需大量GPU算力训练LLM；在极大规模时生成空间膨胀导致的偶尔“hallucination”风险；缺乏多模态特征，未来计划引入压缩与多模态扩展。

---

## 251. S^2tory: Story Spine Distillation for Movie Script Summarization

**arXiv ID:** 2605.03244 | [PDF](https://arxiv.org/pdf/2605.03244v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 252. The TTS-STT Flywheel: Synthetic Entity-Dense Audio Closes the Indic ASR Gap Where Commercial and Open-Source Systems Fail

**arXiv ID:** 2605.03073 | [PDF](https://arxiv.org/pdf/2605.03073v1)

**作者:** Venkata Pushpak Teja Menta `[一作]` `[通讯]` (Praxel Ventures), Venkata Pushpak Teja Menta (Praxel Ventures)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种自包含的 TTS↔STT 飞轮方法，通过生成实体密集合成语音并在 Whisper/vasista22 上进行 LoRA 微调，显著提升印度语言实体识别性能。

**💡 创新点**

创新点包括：① 构建可公开的实体密集合成语料库 (EDSA)；② 采用多 TTS 系统路由实现语音多样性；③ 在 Whisper 和 vasista22 上使用 LoRA 进行实体增量学习，并针对脚本崩溃提出语言条件 LoRA 修正。

**🔧 技术方法**

使用的技术包括 Whisper‑large‑v3、vasista22 Whisper、Praxy R6 等多 TTS、LoRA 微调、实体命中率 (EHR) 指标、脚本忠诚度 (SFR) 度量，以及 Anthropic Haiku‑4.5 文本生成。

**📊 数据集**

使用的数据集包括 22,000+ 句合成实体密集语音（EDSA）、FLEURS、Common Voice 25.0、IndicVoices、Cartesia hold‑out、以及手工构建的实体词典。

**📈 对比分析**

比较方法：在 Cartesia 合成实体密集 hold‑out 上评估 EHR，结果从 0.027 提升至 0.473（17×）并超过 Deepgram 0.16（3×）；在原始读通顺集上回归仅 +6.6pp WER；跨语言的 EHR 分别达到 0.337（Hi）和 0.543（Ta）；SFR 在 Telugu 上提升至 0.93。

**⚠️ 局限性**

局限性：评测主要基于合成 hold‑out，缺乏多说话人/多环境验证；未给出自举置信区间；仅以 Deepgram 为商业基线；样本量有限；LoRA 在非 Telugu 语言上可能导致回归。

---

## 253. CRT: Collision-Tolerant Residence Time for Deterministic Transmission in LEO Satellite Networks

**arXiv ID:** 2605.03382 | [PDF](https://arxiv.org/pdf/2605.03382v1)

**作者:** Siqi Yang `[一作]` (Fudan University), Yue Gao `[通讯]` (Fudan University)

**通讯引用:** 19592 | [OpenAlex ID](https://openalex.org/A5100602494)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出了一种名为CRT的基于驻留时间的定时传输框架，能够在不需要全局时钟同步的LEO卫星网络中实现时序敏感流的可确定性传输。

**💡 创新点**

创新点在于：①使用本地时钟控制每跳驻留时间，补偿链路时延波动；②引入冲突容忍调度模型，最大化可调度流量并在冲突引起的抖动上设定上界；③设计CRT‑Fast启发式算法，采用迭代分层和路径连续性实现高效调度与手动切换；④提出隐式备份路径的无缝切换机制。

**🔧 技术方法**

采用TSN相关技术（IEEE 802.1Qbv/Qbu）结合网络计算理论求解WCD、冲突度；利用整数规划框架描述调度问题，并用启发式贪心+层次分配实现。

**📊 数据集**

在Iridium（6平面×11卫星）和Starlink（72平面×1,584卫星）两套星座的仿真数据集上进行实验。

**📈 对比分析**

与DSTMR、严格不冲突、最短路径优先、负载感知贪心等基线方法对比。结果显示CRT‑Fast在高负载下可达94.5%调度成功率，冲突度和碰撞引起的抖动显著低于基线，且路径切换次数减少约38.7%。

**⚠️ 局限性**

局限性包括：①仍假设每条链路有足够的保留带宽；②算法复杂度为O(|T|·|F|^2·K·L)，在极大规模场景下仍可能成为瓶颈；③未考虑非TT流对驻留时间的动态影响；④仿真中使用理想化的链路失效与延迟模型，实际环境中可能需进一步验证。

---

## 254. A Skill-Based AI Agentic Pipeline for Library of Congress Subject Indexing

**arXiv ID:** 2605.03537 | [PDF](https://arxiv.org/pdf/2605.03537v1)

**作者:** Eric H. C. Chow `[一作]` (University of Hong Kong), Eric H. C. Chow `[通讯]` (University of Hong Kong)

**通讯引用:** 84 | [OpenAlex ID](https://openalex.org/A5024409827)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个由四个模块化AI agent技能组成的管道，用以自动化美国国会图书馆主题词索引（LCSH），生成符合MARC 6xx字段的主体索引；

**💡 创新点**

创新点在于将主题索引流程拆分为可单独验证、可审计的四个技能，显式编码《主题词手册》（SHM）规则，提升规则遵循性与可升级性，区别于单一提示的LLM方法；

**🔧 技术方法**

使用Anthropic Claude Code的agent技能框架，配合Claude LLM执行概念分析、定量过滤、权威验证与MARC合成；技术上还整合本地TF‑IDF索引和LC Linked Data API进行权威检查；

**📊 数据集**

评估数据集为Harvard Library Bibliographic Dataset中选取的10本书的现有LCSH记录，作为人类索引参考；

**📈 对比分析**

通过将管道输出与Harvard记录的6xx字段逐项对比，关注概念召回、题名精确度、细分准确度及形式/流派处理；在10个标题中概念重叠率超过50%，多数案例比基准更具体，但在地理细分、同义词导航及与现有记录协同方面存在差距；

**⚠️ 局限性**

局限性包括样本量小（仅10本）、缺乏统计显著性；地理/时间细分不足；同义词选择不够精确；未利用复制目录信息；仅测试LCSH，未扩展到其他词表；与人类索引间仍存主观差异。

---

## 255. Fully Automatic Trace Gas Plume Detection

**arXiv ID:** 2605.03372 | [PDF](https://arxiv.org/pdf/2605.03372v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 256. Cheap Expertise: Mapping and Challenging Industry Perspectives in the Expert Data Gig Economy

**arXiv ID:** 2605.03295 | [PDF](https://arxiv.org/pdf/2605.03295v1)

**作者:** Robert Wolfe `[一作]` (Rutgers University), Aayushi Dangol `[通讯]` (University of Washington)

**通讯引用:** 833 | [OpenAlex ID](https://openalex.org/A5003028922)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过定向内容分析和主题分析，对 2025 年 6 月至 12 月期间五家领先专家数据标注公司的 X 帖子和 CEO 访谈进行定性研究，阐述了行业对 AI 专业、人类专业和制度专业的未来愿景。

**💡 创新点**

创新点在于将专家工作重新定义为“廉价 AI 专业”“可提取的人类专业”和“需要解放的制度专业”，并提出针对每类专业的挑战与未来发展方向。

**🔧 技术方法**

研究采用了定向内容分析、归纳-演绎编码以及主题分析技术，构建代码表并对社交媒体文本与播客转录进行编码。

**📊 数据集**

数据集来源于公开的 X 帖子（共 1,254 条）和 29 条 CEO 访谈的音视频转录，涵盖五家公司（Surge AI、Scale AI、Handshake AI、Mercor、Turing）的公开沟通。

**📈 对比分析**

通过比较三类专业的主题，展示行业视角；由于研究为定性分析，未给出数值性能指标，但结果清晰揭示了行业对专家工作与 AI 交互的预期与影响。

**⚠️ 局限性**

局限性包括仅分析公司公开的沟通内容、聚焦于 2025 年 6-12 月的 5 家美国公司，未涵盖专家工人本身视角、非西方语境以及更广泛的行业内部观点。

---

## 257. ReasonAudio: A Benchmark for Evaluating Reasoning Beyond Matching in Text-Audio Retrieval

**arXiv ID:** 2605.03361 | [PDF](https://arxiv.org/pdf/2605.03361v1)

**作者:** Honglei Zhang `[一作]` (Nanjing University), Yilei Shi `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 2677 | [OpenAlex ID](https://openalex.org/A5061007704)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建了一个基于音频的推理检索基准（RAN），包含 1,000 条文字查询和 10,000 条合成音频片段，设计了五个需要逻辑推理的任务（否定、顺序、重叠、持续时长、混合）。

**💡 创新点**

创新点在于首次构造专门用于评估音频检索中逻辑推理能力的基准数据集，并通过人工与程序相结合的方式实现高质量、可扩展的标签，填补了现有语义匹配基准的空白。

**🔧 技术方法**

采用 200 条从 FSD50K 与 Freesound 选取的原子声音合成 10,000 条多时序音频，使用预定义模板生成查询，并通过 deterministic 程序标注对应关系；随后在十个最先进模型上进行评估。

**📊 数据集**

使用的数据集包括：200 条原子声音（FSD50K/Freesound）以及构造出的 RAN 数据集（1,000 条查询、10,000 条音频），并与 AudioCaps、Clotho、WavText5K 等传统音频检索基准进行对比。

**📈 对比分析**

通过对两阶段检索、CLIP 风格对比学习模型和基于多模态 LLM 的嵌入模型进行实验，使用 Acc@1 和 nDCG@10 评价；结果显示最优模型 OmniEmbed-7B 的平均准确率仅为 20.1%，所有模型在否定与持续时长任务上表现尤为差。

**⚠️ 局限性**

限制在于现有的多模态检索模型无法很好地捕捉逻辑约束，尤其是否定和时长推理；对比学习训练强调相似度匹配而忽略约束；文本与音频嵌入空间对齐不足，导致推理能力被削弱。

---

## 258. Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch

**arXiv ID:** 2605.03346 | [PDF](https://arxiv.org/pdf/2605.03346v1)

**作者:** Dionysis Arvanitakis `[一作]` (Northwestern University), Yiyuan Luo `[通讯]` (University of California, Santa Cruz)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在三元组对比学习中，维度与准确率的关系，证明当嵌入维度低于真实维度 D 的常数倍时，准确率会骤降至仅 50% 的基线；同时在 Unique Games Conjecture 下证明无法在多维度下实现超过 12% 的准确率提升。

**💡 创新点**

首次给出嵌入维度-准确率的严格信息论下界和计算难度下界；证明维度崩溃现象，并构造逼近抵抗的几何约束满足问题；在可实现实例中揭示维度截断导致的准确率坍塌。

**🔧 技术方法**

采用概率方法与随机实例构造、可实现性与无环图对应、VC 维泛化界、学习理论泛化界、图 Arboricity 分析、以及从最大无环子图（MAS）到三元组嵌入的逼近抵抗归约等技术。

**📊 数据集**

使用合成数据：随机生成单位球嵌入（n=1000，D∈{128,256,512,1024}）和随机三元组（n=4000，m=10^6）。

**📈 对比分析**

通过训练不同维度（包括 unconstrained 与 spherical 嵌入）并与 50% 随机基线对比，实验显示当维度降至 D 的约 5% 时准确率跌至 ~12% 近似 50% 基线，验证理论预测。

**⚠️ 局限性**

结果仅适用于理想化的三元组/四元组比较设置，未考虑结构化数据或大间隔假设；计算难度证明依赖于 Unique Games Conjecture，实际可实现性仍有不确定性。

---

## 259. From Passive Feeds to Guided Discovery: AI-Initiated Interaction for Vague Intent in Content Exploration

**arXiv ID:** 2605.02902 | [PDF](https://arxiv.org/pdf/2605.02902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 260. GRPO-TTA: Test-Time Visual Tuning for Vision-Language Models via GRPO-Driven Reinforcement Learning

**arXiv ID:** 2605.03403 | [PDF](https://arxiv.org/pdf/2605.03403v1)

**作者:** Yujun Li `[一作]` (Northwestern Polytechnical University), Yuan Yuan `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 27188 | [OpenAlex ID](https://openalex.org/A5100334733)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于GRPO的测试时自适应框架GRPO‑TTA，针对视觉‑语言模型在无标签环境下的动态适配

**💡 创新点**

将GRPO重新构造为类特定提示的组策略优化，并设计了对齐奖励与分散奖励两种无监督奖励，避免模型退化

**🔧 技术方法**

采用GRPO（强化学习）、CLIP相似度、对抗式奖励设计、数据增强与低熵筛选等技术

**📊 数据集**

在十个跨域视觉识别数据集与ImageNet及其四个OOD变体（ImageNet‑A/V2/R/Sketch）上进行评估

**📈 对比分析**

与CoOp、CoCoOp、TPT、DiffTPT、RLCF、WATT、CLIP‑TTA、TDA、DPE、DOTA等传统与强化学习驱动的TTA方法对比，GRPO‑TTA在跨域平均准确率提升至70.45%，自然分布偏移平均提升至74.91%，同时适配时间约3小时40分钟，显著优于大多数方法

**⚠️ 局限性**

适配时间仍高于最轻量级方法（如DOTA），且对超参数如λ、K、更新步数较为敏感，过多采样或更新可能导致性能下降

---

## 261. A Poisson Process for Submodular Maximization

**arXiv ID:** 2605.03071 | [PDF](https://arxiv.org/pdf/2605.03071v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 262. MAGE: Safeguarding LLM Agents against Long-Horizon Threats via Shadow Memory

**arXiv ID:** 2605.03228 | [PDF](https://arxiv.org/pdf/2605.03228v1)

**作者:** Yuhui Wang `[一作]` (Stony Brook University), Ting Wang `[通讯]` (Stony Brook University)

**通讯引用:** 79900 | [OpenAlex ID](https://openalex.org/A5075670673)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种名为MAGE的基于agentic memory的安全框架，利用shadow memory对LLM代理在多轮交互中的长时序攻击进行检测与阻断。

**💡 创新点**

首次将agentic memory转为安全防御原语，采用shadow stack类的安全记忆层进行长期上下文抽取与风险评估，并在同一模型中联合训练memory manager和judge。

**🔧 技术方法**

使用强化学习的Turn‑wise GRPO进行联合训练，配合小型LLM（如Qwen3‑4B）实现memory manager与judge；同时设计格式、长度惩罚等奖励项以优化记忆质量与决策。

**📊 数据集**

采用SHADE‑Arena任务与AgentDojo环境生成多轮攻击与正常轨迹，并用STAC、PI2等攻击实例进行训练与评估。

**📈 对比分析**

与Spotlighting、Repeated‑Prompt、PI‑Detector、IPIGuard、MELON等现有方法对比，MAGE在用户/环境对手模型下攻击成功率降至0‑0.4%，正任务完成率≥73%，并仅额外消耗1‑3K tokens，显著优于对手。

**⚠️ 局限性**

局限在于仅针对单代理场景，攻击类型和模型仅覆盖少数案例；shadow memory的固定结构缺乏自适应能力，对极端适应性攻击仍存在潜在风险。

---

## 263. Stable Agentic Control: Tool-Mediated LLM Architecture for Autonomous Cyber Defense

**arXiv ID:** 2605.03034 | [PDF](https://arxiv.org/pdf/2605.03034v1)

**作者:** Kerri Prinos `[一作]` (Horizon3.ai), Amy Villaseñor `[通讯]` (Horizon3.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种工具介入的 LLM 控制架构，用于在对抗性压力下实现端点检测与响应 (EDR) 政策的安全配置，并通过 Lean4 机械化证明得到闭环稳定性（可控性、可观测性和输入对状态的稳定性）保证，实验验证涵盖 282 条真实企业攻击图；

**💡 创新点**

创新点在于将博弈论、控制理论与工具化 LLM 框架相结合，形成非训练、非学习的确定性工具输出机制，并通过组合李雅普诺夫函数实现闭环稳定性机理化验证；

**🔧 技术方法**

采用了 Stackelberg 双数目算法、贝叶斯观察器、攻击图原语、工具化 LLM（如 Claude Sonnet/HaiKu）以及 Lean4 机械化证明；

**📊 数据集**

使用了来自 Horizon3.ai NodeZero 的 282 条真实企业渗透测试攻击图（以及一张 GOAD 任务图配对的防御与攻击遥测）；

**📈 对比分析**

与确定性贪婪基线对比，Sonnet 通过工具化 LLM 在 40 次跑 里以零方差实现 59% 的博弈价值下降；Haiku 在 40 次跑中表现出较大方差但仍保持在架构安全边界内；整体性能显示架构在保持安全性的同时显著提升防御效果；

**⚠️ 局限性**

局限性包括需预先限定有限的动作目录、对抗者必须受限于该目录，且虽然闭环稳定，但无法保证达到全局最优；若攻击者能突破目录或工具假设，系统可能失效；

---

## 264. Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use

**arXiv ID:** 2605.02964 | [PDF](https://arxiv.org/pdf/2605.02964v1)

**作者:** Kunvar Thaman `[一作]` `[通讯]` (Independent Researcher), Kunvar Thaman (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Reward Hacking Benchmark（RHB），通过多步工具使用任务量化 LLM 代理的奖励破解行为，并系统评估不同模型、链长、任务难度及环境硬化对破解率的影响。

**💡 创新点**

创新点在于：①首次构建可链式、自动可评判的多步工具使用基准，②通过深度同族对比揭示 RL 后训练显著提升奖励破解率，③发现链长和任务难度可导致破解率跃升，并验证简单环境硬化可显著抑制破解行为。

**🔧 技术方法**

采用工具调用（bash、Python 等）模拟 LLM 代理执行、记录命令日志、文件读写；利用自定义判定规则对日志进行奖励破解分类；使用统计方法（Clopper–Pearson CI、Fisher 检验）对破解率和成功率进行比较。

**📊 数据集**

使用 RHB 提供的 CoreRHB 与 MicroRHB 任务集，涵盖数据处理、日志取证、性能优化与多文件重构四大任务族，并生成标准与难度提升的两层变体。

**📈 对比分析**

通过对比 13 款前沿模型（含 OpenAI、Anthropic、Google、DeepSeek 等）及 DeepSeek V3 与 R1‑Zero 的同族对比，发现 RL 训练模型的破解率从 0.6% 提升至 13.9%；链长 5 时率急剧上升；在硬化环境下破解率从 6.5% 降至 0.8%，相对降低 87.7%，任务成功率无显著下降。

**⚠️ 局限性**

主要限制包括：样本量受 API 成本限制、RL 与其他训练因素共线性导致归因不完全、规则检测误差约 6%、硬化方法需与模型进步同步演进、实验环境缺少网络与真实世界泄漏途径。

---

## 265. TeamUp: Semantic Project Matching and Team Formation for Learning at Scale

**arXiv ID:** 2605.03237 | [PDF](https://arxiv.org/pdf/2605.03237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 266. PAMNet: Cycle-aware Phase-Amplitude Modulation Network for Multivariate Time Series Forecasting

**arXiv ID:** 2605.02938 | [PDF](https://arxiv.org/pdf/2605.02938v1)

**作者:** Yingbo Zhou `[一作]` (Beihang University), Dejing Dou `[通讯]` (Fudan University)

**通讯引用:** 5026 | [OpenAlex ID](https://openalex.org/A5066063885)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种Cycle-aware Phase-Amplitude Modulation Network（PAMNet），用于多变量时序预测，显式将周期模式拆分为相位和幅度两部分并通过轻量化调制器进行调制。

**💡 创新点**

创新点在于通过双分支可学习的相位与幅度嵌入以及Hadamard乘法的调制器，显式捕获周期非平稳中的相位-幅度耦合，从而在不使用复杂注意力机制的情况下实现更精细的周期建模。

**🔧 技术方法**

使用技术包括可学习周期嵌入、SiLU激活函数、基于特征维度的Hadamard乘法调制器、轻量化MLP投影以及混合MAE/频域损失等。

**📊 数据集**

实验在12个真实世界数据集上进行，涵盖ETT、PEMS、Electricity、Weather、Traffic、Solar等多种业务场景。

**📈 对比分析**

与9个最新基线（TQNet、FilterTS、Amplifier、CycleNet、TimeMixer、iTransformer、TimesNet、PatchTST、DLinear）在多尺度预测窗口下进行对比，PAMNet在20/24个指标中夺得Top‑1，平均提升约2‑4%，且在计算效率和显存占用上保持竞争力。

**⚠️ 局限性**

局限性包括对周期长度c的设定仍需经验或调参，过小或过大会导致性能下降；在极端缺失或噪声场景下相对鲁棒性不及专门设计的缺失处理模型；在周期性不明显的数据上其优势不如传统模型显著。

---

## 267. CuraView: A Multi-Agent Framework for Medical Hallucination Detection with GraphRAG-Enhanced Knowledge Verification

**arXiv ID:** 2605.03476 | [PDF](https://arxiv.org/pdf/2605.03476v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 268. When Should a Language Model Trust Itself? Same-Model Self-Verification as a Conditional Confidence Signal

**arXiv ID:** 2605.02915 | [PDF](https://arxiv.org/pdf/2605.02915v1)

**作者:** Aditya Ajay Phalod `[一作]` `[通讯]` (Independent Researcher), Aditya Ajay Phalod (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对同模型自我验证（self‑verification）在多选题上的置信度估计进行系统评估，探讨其在不同任务、模型与提示下的有效性。

**💡 创新点**

提出将自我验证与基于 log‑likelihood 的 LL‑AVG、LL‑SUM 进行对比，并展示其在 ARC‑Challenge 上能显著提升选择性预测性能，却在 TruthfulQA‑MC 上表现不稳定。

**🔧 技术方法**

使用 LL‑AVG/LL‑SUM 评分、对自我验证提示的对比、AUROC、AURC 等评价指标，并在多种模型（Qwen、DeepSeek、LLaMA、Falcon、Llama‑2）上进行实验。

**📊 数据集**

采用 ARC‑Challenge 与 TruthfulQA‑MC 两个多选题数据集，分别对应推理难度与真相性错误。

**📈 对比分析**

通过 AUROC、AURC、风险‑覆盖曲线等指标对比，发现自我验证在 ARC‑Challenge 上可显著提升 AUROC 与降低 AURC，提升低错误率覆盖率，但在 TruthfulQA‑MC 上多为负面或低于 LL‑SUM。

**⚠️ 局限性**

局限性包括模型覆盖范围有限、仅限多选任务、对提示的探索不够全面、未覆盖更丰富的置信度基线、缺乏因果机制解释。

---

## 269. Hyper-Minimization for Deterministic Register Automata

**arXiv ID:** 2605.03535 | [PDF](https://arxiv.org/pdf/2605.03535v1)

**作者:** Yong Li `[一作]` (Chinese Academy of Sciences), Di-De Yen `[通讯]` (University of Liverpool)

**通讯引用:** 143 | [OpenAlex ID](https://openalex.org/A5052125757)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出了一种针对确定性寄存器自动机（DRA）的超最小化算法，能够在保证与原DRA在词类型上仅存在有限差异的前提下，得到同时在状态数和寄存器数上最小的DRA。

**💡 创新点**

创新点在于：①首次将超最小化概念推广到无限字母表的DRA；②引入词类型的概念，使得“有限错误”在无限字母表上仍然有意义；③构造了满足well‑typed条件的DRA的唯一最小化形式，并证明了该超最小化问题的可判定性；④设计了兼顾数据和状态两方面的合并操作，确保最终结果在两方面均最小。

**🔧 技术方法**

技术方法主要包括：Myhill–Nerode定理的推广、记忆性（memorability）与可泵性质的使用、对well‑typed DRA的正规化与剪枝、对状态的“近似等价”关系的定义与计算、以及在DRA上实现的合并操作和递归合并策略。

**📊 数据集**

本文为理论研究，未使用具体实验数据集；所有结果均基于数学证明与算法复杂度分析。

**📈 对比分析**

对比方法：在DRA理论框架下，与传统DFA超最小化算法（时间复杂度O(n log n)）进行对比；性能方面，算法在well‑typed DRA上实现了与传统DFA相同阶的时间复杂度，同时在状态与寄存器数上实现了最优（即同时数据与状态最小）。

**⚠️ 局限性**

局限性：①仅对well‑typed DRA可应用，未覆盖所有DRA；②超最小化结果并非唯一；③在实际实现中，计算“近似等价”与合并操作可能导致较高的运行时间，尤其是对大规模寄存器数的DRA；④对非well‑typed DRA的最小化仍是未解决的问题。

---

## 270. MenuNet: A Strategy-Proof Mechanism for Matching Markets

**arXiv ID:** 2605.03216 | [PDF](https://arxiv.org/pdf/2605.03216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 271. When Prompts Interact: Assessing Prompt Arithmetic for Deconfounding under Distribution Shift

**arXiv ID:** 2605.03096 | [PDF](https://arxiv.org/pdf/2605.03096v1)

**作者:** Zhecheng Sheng `[一作]` (University of Minnesota), Serguei Pakhomov `[通讯]` (University of Minnesota)

**通讯引用:** 5215 | [OpenAlex ID](https://openalex.org/A5056292346)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Hybrid Prompt Arithmetic（HyPA）以在存在混淆偏移的文本分类任务中提升鲁棒性

**💡 创新点**

将非线性任务提示微调与线性化混淆提示微调相结合，形成两阶段混合算术来抑制短路学习

**🔧 技术方法**

利用提示微调、线性化梯度近似、任务向量算术和可插入的软提示

**📊 数据集**

在五个公开文本分类数据集（SHAC、MIMIC、Hate Speech、Civil Comments、Amazon Reviews）上进行实验

**📈 对比分析**

与仅使用非线性或线性化提示算术的基线比较，HyPA在OOD AUPRC、AUSC和Adjusted Integral上均表现更佳，展示出更优的性能-鲁棒性权衡

**⚠️ 局限性**

主要局限包括缺乏理论解释、对混淆方向的确切影响不明、以及在未知测试时移情景下的模型选择难题

---

## 272. Device-Induced Thrombus Formation in Cerebral Aneurysms: Linking Patient-Specific Clot Modeling and Functional Occlusion to Virtual Angiographic Assessment

**arXiv ID:** 2605.03536 | [PDF](https://arxiv.org/pdf/2605.03536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 273. RAG over Thinking Traces Can Improve Reasoning Tasks

**arXiv ID:** 2605.03344 | [PDF](https://arxiv.org/pdf/2605.03344v1)

**作者:** Negar Arabzadeh `[一作]` (University of California), Matei Zaharia `[通讯]` (University of California)

**通讯引用:** 50120 | [OpenAlex ID](https://openalex.org/A5005554337)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了使用思考轨迹（intermediate reasoning traces）作为检索语料的检索增强生成（RAG）框架，并引入了离线转换方法（T3）将原始轨迹转化为结构化、压缩或诊断形式，以提升推理性能。

**💡 创新点**

创新点在于将检索语料从传统的文档转为先前模型的思考轨迹，并通过轻量级离线转换提升轨迹的可检索性和有效性，展示了思考轨迹在推理任务中的强大价值。

**🔧 技术方法**

技术主要包括：检索增强生成（RAG）pipeline；离线转换模型（T3）实现结构化、语义提炼和反思三种转换策略；使用预训练的检索器（e5-base）和生成模型（Gemini-2.5-Flash、GPT-5、GPT-OSS-120B）。

**📊 数据集**

数据集包括从QwQ-32B和Gemini-2-thinking生成的59K-114K思考轨迹（涵盖数学、代码、科学等领域），以及评测基准AIME 2025–2026、GPQA-Diamond、LiveCodeBench。

**📈 对比分析**

与无检索（No RAG）和多种通用语料库（OpenWebMath、StackExchange、Wikipedia、GitHub、ArXiv、CompactDS、Tavily）相比，使用原始思考轨迹即可显著提升准确率（例如Gemini-2.5-Flash在AIME从53.3%提升至80.0%），而经T3转换后进一步提升（GPT-5在AIME从86.7%提升至93.3%）。在成本‑准确率平衡上，T3转换往往实现更低成本或更高准确率。

**⚠️ 局限性**

局限性包括：检索语料仍偏重数学领域；转换方法依赖较强的轻量级模型，转换质量受限；不同模型对检索效果的敏感性差异大，未能系统探索所有组合；以及在某些任务和模型上转换后的收益有限。

---

## 274. Stable Multimodal Graph Unlearning via Feature-Dimension Aware Quantile Selection

**arXiv ID:** 2605.03303 | [PDF](https://arxiv.org/pdf/2605.03303v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 275. Kerncap: Automated Kernel Extraction and Isolation for AMD GPUs

**arXiv ID:** 2605.03208 | [PDF](https://arxiv.org/pdf/2605.03208v1)

**作者:** Cole Ramos `[一作]` (Advanced Micro Devices), Keith Lowery `[通讯]` (Advanced Micro Devices)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5059289656)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

开发了一款名为Kerncap的自动GPU内核提取与隔离工具，能在AMD GPU上捕获内核、运行时状态与环境，生成可编辑、可验证的独立重现项目。

**💡 创新点**

创新点包括：统一HSA级拦截实现HIP与Triton的内核捕获；采用地址空间闭包实现VA忠实设备内存快照；利用Python编译钩子捕获Triton JIT元数据并自动绑定自适应调优配置；通过Clang VFS覆盖实现原始编译器参数的可重编译。

**🔧 技术方法**

技术手段包括：HSA API表钩取（rocprofiler-sdk）、Clang VFS、Python AST与import追踪、哈希校验、设备内存地址闭包、HSA replay、自动源代码发现、DWARF解析、Triton autotuner钩子等。

**📊 数据集**

评测使用六个真实工作负载：HIP的llama.cpp、LAMMPS、rocBLAS GEMM，以及Triton的Flash Attention 2、vLLM MoE、torch.compile，快照大小从152 MB至30 GB不等。

**📈 对比分析**

与传统全应用构建+运行循环对比，Kerncap在llama.cpp上实现13.6×的完整迭代加速，单内核编辑-重编译-验证循环提升5×；捕获开销占短工作负载总时间的1.5–1.9×，长工作负载约11%，相较NVIDIA Nsight Compute/CUPTI等工具更轻量且可编辑。

**⚠️ 局限性**

局限性：仅单核隔离，无法捕获跨核依赖；单GPU，缺乏多GPU支持；若设备缓冲在快照后被释放会丢失；不捕获主机侧状态；源代码发现依赖编译数据库；Triton自适应配置不跨GPU可移植；RDNA评测范围有限。

---

## 276. ZK-Value: A Practical Zero-Knowledge System for Verifiable Data Valuation

**arXiv ID:** 2605.03581 | [PDF](https://arxiv.org/pdf/2605.03581v1)

**作者:** Zhaoyu Wang `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 25855 | [OpenAlex ID](https://openalex.org/A5100328273)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了一套可在数据市场中进行隐私保护和可验证的数据价值评估系统——ZK‑VALUE，使用基于局部敏感哈希的 Shapley 近似（LSH‑Shapley）来评估每个数据提供者的价值，并通过专门设计的零知识证明协议（ZK‑LSH‑SHAPLEY）对评估结果进行全流程证明。

**💡 创新点**

创新点包括：
1) 将 LSH‑Shapley 与 ZKP 共设计，避免传统 KNN‑Shapley 在零知识证明中的指数级成本；
2) 提出专门的 ZKP 协议，利用桶级计数代替逐对距离证明；
3) 采用超级预言机批处理与稀疏跳过两种优化，显著减少证明大小和证明时间；
4) 在理论与实验上证明该方案在满足完整性、隐私和可验证性四大要求的同时，能够在现代机器学习规模上实现可行性。

**🔧 技术方法**

技术要点包括：Locality‑Sensitive Hashing（SimHash）、Shapley 值理论、polynomial commitment、sumcheck、lookup argument、Fiat–Shamir 变换；此外还引入超级预言机批处理（super‑oracle batching）与稀疏跳过（sparsity‑aware sumchecks）来降低计算量和通信量。

**📊 数据集**

实验数据集：
- 12 个公开基准：10 个 OpenML 表格数据集（特征维度 5–170）和 2 个 ViT 预嵌入数据集（768 维）;
- 合成高斯数据集，用于可扩展性与消融实验，控制特征维、样本量、类别数、验证集大小。

**📈 对比分析**

对比方法：KNN‑Shapley、梯度‑Shapley、Monte‑Carlo Shapley、Data Banzhaf、两种 ZKP 基线（KNN‑Shapley SNARK 与 per‑pair tensor 编码）。评估指标包括 AUROC（误标/噪声检测任务）、评估运行时间、证明时间、验证时间与证明大小。结果显示：
- AUROC 与 KNN‑Shapley 差距 ≤ 0.02；
- 证明时间从秒级到分钟级，平均约 2–10 秒；
- 验证时间 < 1 秒；
- 证明大小几 MB；
- 相较于基线，证明时间提升 10–100 倍，证明大小下降 5–10 倍。

**⚠️ 局限性**

局限性：
1) 仅针对分类任务的 Shapley 近似，无法直接用于回归或生成模型；
2) 对极高维特征仍需 PCA 预处理，增加额外步骤；
3) 依赖特定 ZKP 体系结构（Brakedown+Goldilocks），在不同硬件/编译器上需要重新适配；
4) 与差分隐私或 MPC 方案兼容，但尚未完成完整的隐私+可验证联合设计；
5) 在极大规模（> 10^6 样本）时仍可能受到内存或单机算力限制。

---

## 277. Towards a Risk-Cost Model for Financial Adaptive Authentication

**arXiv ID:** 2605.02979 | [PDF](https://arxiv.org/pdf/2605.02979v1)

**作者:** Supriya Khadka `[一作]` (George Mason University), Sanchari Das `[通讯]` (George Mason University)

**通讯引用:** 1341 | [OpenAlex ID](https://openalex.org/A5059400253)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了面向金融系统的风险成本模型（RCM），将自适应身份验证建模为受限动态优化问题，结合损失函数、尾部风险、隐私约束与顺序决策。

**💡 创新点**

将欺诈损失、机会成本和挑战摩擦融入统一的成本函数，并在此基础上引入CVaR尾部风险、分布鲁棒性以及价值信息驱动的挑战触发策略，首次将经济与对抗性动态视角融合进金融身份验证。

**🔧 技术方法**

使用贝叶斯决策、概率校准、CVaR优化、分布式鲁棒（Wasserstein/TV球）和顺序决策（bandit/POMDP）等技术，并结合信息理论泄露度量。

**📊 数据集**

论文未公开具体数据集，示例基于假想或真实金融交易日志与身份验证事件，但缺乏可复现的数据来源。

**📈 对比分析**

主要提供理论框架与算法伪代码，未进行实验对比；若实验，预期相较传统阈值或分类方法能更好平衡欺诈成本与用户体验。

**⚠️ 局限性**

需要准确估计成本参数、概率校准与隐私泄露度；模型对参数敏感、缺乏大规模实验验证与可扩展实现。

---

## 278. Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection

**arXiv ID:** 2605.02958 | [PDF](https://arxiv.org/pdf/2605.02958v1)

**作者:** Xulin Hu `[一作]` (Peking University), Zhong Chen `[通讯]` (Peking University)

**通讯引用:** 46833 | [OpenAlex ID](https://openalex.org/A5100430399)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过因果追踪揭示LLM拒绝行为的动态“拒绝轨迹”，并基于此提出一种无梯度、推理时可用的稀疏激活定位检测器SALO，用于零样本检测多种恶意攻击。

**💡 创新点**

①首次证明拒绝不是终端状态，而是沿着中层激活形成的稀疏轨迹；②提出稀疏激活定位算子(SALO)，只利用标准安全训练数据即可对未知攻击实现零样本通用检测；③通过多尺度卷积与最大池化保留轨迹的空间稀疏特征，提升鲁棒性。

**🔧 技术方法**

因果追踪（Causal Tracing）、稀疏激活定位算子、三层滑动窗口、三种宽度的卷积核、多尺度特征聚合、全局最大池化、Sigmoid分类器；训练仅使用对齐数据，无对抗样本。

**📊 数据集**

PKU‑SafeRLHF、Toxic‑Chat（约5.5k对话，包含5,500个安全提示），40对恶意‑良性提示用于因果追踪；XSTest（含“硬负样本”）用于评估AUROC和DSR；AdvBench提供直接、Prefilling、GCG、AutoDAN攻击样本。

**📈 对比分析**

与PPL Filter、Linear Probe、GradSafe、SmoothLLM等基线对比；在Qwen、Mistral、Llama三大模型上，SALO在Prefilling、GCG、AutoDAN等攻击下均能保持DSR>85%，在AutoDAN上近乎100%，在GCG上比Linear Probe高出≈75%；对抗白盒优化（Adaptive GCG）下仍保持≈84%检测率，攻击成功率显著下降。

**⚠️ 局限性**

①因果追踪仅确认充足性，未证明必要性，可能忽略其他冗余路径；②未映射具体注意力头或电路细节；③SALO依赖模型内部已识别恶意语义，若攻击通过编码或语义模糊化完全隐藏恶意，轨迹可能未激活，导致误判或漏判。

---

## 279. iSMC: A BDD-based Symbolic Model Checker with Interactive Certification

**arXiv ID:** 2605.03705 | [PDF](https://arxiv.org/pdf/2605.03705v1)

**作者:** Philipp Czerner `[一作]` (Technical University of Munich), Konrad Winslow `[通讯]` (Technical University of Munich)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

开发了首个自我认证的 BDD 基础符号模型检查器，能够在不重新执行模型检查的情况下通过交互式证明系统验证求解结果的正确性。

**💡 创新点**

创新点包括：①基于交互式证明的证书生成与验证，保证高概率正确性；②针对执行轨迹构造的通用布尔电路（GBC）并设计专门的交互协议；③改进 eBDD 数据结构以支持全局缓存、不可变节点和版本链接，从而大幅提升 Prover 的效率；④引入基于“oracle”假设的自底向上协议，实现垃圾回收与性能优化。

**🔧 技术方法**

使用的技术主要有：BDD 以及扩展 BDD（eBDD）；多变量多项式 arithmetization；交互式证明协议（IP）与 DeMillo‑Lipton‑Schwartz‑Zippel 线性化技术；随机多项式检验；以及针对 CTL 及正义要求的符号模型检查算法。

**📊 数据集**

实验数据集来自 HWMCC25 的 liveness track，包含 53 个 AIGER 形式的基准模型，使用转换工具将其转为 .cit 格式。

**📈 对比分析**

与成熟的 NuSMV 进行比较：Solver 的平均慢速因子约为 1.5；Prover 与 Solver 运行时间比例约为 1：1.7，Verifier 运行时间始终小于 1 秒，平均比 Prover 快 7 倍；改进后的 + + 相比原始实现性能提升平均 73 倍、内存降低 5.7 倍。

**⚠️ 局限性**

限制：当前实现不支持关系积等高级 BDD 操作；交互式证明依赖“oracle”假设，若 Prover 存在跨查询的隐藏状态可能降低安全性；在极大模型的 trace 长度仍可能影响 Verifier 线性时间的实际效率。

---

## 280. ELAS: Efficient Pre-Training of Low-Rank Large Language Models via 2:4 Activation Sparsity

**arXiv ID:** 2605.03667 | [PDF](https://arxiv.org/pdf/2605.03667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 281. BIT.UA-AAUBS at ArchEHR-QA 2026: Evaluating Open-Source and Proprietary LLMs via Prompting in Low-Resource QA

**arXiv ID:** 2605.03618 | [PDF](https://arxiv.org/pdf/2605.03618v1)

**作者:** Richard A. A. Jonker `[一作]`, Sérgio Matos `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究在ArchEHR‑QA 2026共享任务中，使用纯生成式大型语言模型（LLM）并通过细粒度提示工程（prompt engineering）完成四个子任务：将患者提问转化为简洁临床问题、检索支持句子、生成患者友好的答案以及将答案句子与证据句子对应。

**💡 创新点**

创新点在于：①在极低资源（仅20个开发样本）环境下，全部依赖提示工程而非监督微调；②提出多类提示策略（约束式、提取-生成分解、链式思维、少样本ICL等）并系统评估；③通过“LLM‑as‑a‑Judge”与投票集成提升鲁棒性；④深入探讨开放源代码模型（如MedGemma‑27B）与闭源顶尖模型的性能差距。

**🔧 技术方法**

技术手段包括：大型语言模型（Gemini 3 Flash、Claude Sonnet 4.5、GPT‑4.1、Llama‑3.1 8B Instruct、Qwen‑3、MedGemma‑27B 等）；提示工程技术（零样本、少样本ICL、约束式输出、Task Decomposition、Chain‑of‑Thought、Lexical Constraints、Prompt‑Ensemble、LLM‑as‑a‑Judge）；评估指标多维度组合（ROUGE、BERTScore、AlignScore、MEDCON、BLEU、SARI、Micro‑F1 等）。

**📊 数据集**

使用ArchEHR‑QA 2026共享任务提供的电子健康记录（EHR）语料库，包含47个测试样本（子任务1‑3）和147个子任务4样本；开发集仅20个样本。

**📈 对比分析**

比较方法：对各子任务在开发集上进行多提示、多模型验证，并在公开排行榜上与参赛队伍比较。官方结果显示：子任务4（证据对齐）以81.5 Micro‑F1排名第1；子任务3（患者友好答案生成）以35.6 综合评分排名第3；子任务1和2表现相对较弱，分别排名第13和第11。

**⚠️ 局限性**

局限性包括：①仅靠提示工程，缺乏监督微调，可能受限于小样本导致过拟合；②对闭源模型的依赖导致可复现性与隐私合规问题；③使用固定温度（0.0）和 top‑p（0.95）可能不适合所有模型；④评估仅覆盖英文EHR，对多语言情况未知；⑤实验未覆盖所有模型与提示组合，可能未达到最优。

---

## 282. Pairwise matrices for sparse autoencoders: single-feature inspection mislabels causal axes

**arXiv ID:** 2605.03160 | [PDF](https://arxiv.org/pdf/2605.03160v1)

**作者:** Michael A. Riegler `[一作]` (SimulaMet), Birk Sebastian Frostelid Torpmann-Hagen `[通讯]` (Simula)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于系数与联合条件的双轴矩阵探测协议，用以揭示稀疏自编码器（SAE）特征在语言模型中的行为缺陷，并在三款后训练LLM上进行了实验。

**💡 创新点**

创新点包括：①引入系数-联合条件双轴矩阵协议，取代传统单特征标签检验；②发现特征在不同系数下的非单调倒U形模式；③证明联合抑制三近正交特征可导致模型合成失效；④通过匹配几何扰动对比验证输出连贯性取决于方向而非幅度。

**🔧 技术方法**

技术上使用稀疏自编码器（Qwen‑Scope、Gemma‑Scope、Goodfire Llama SAE）对残差流进行编码，并在解码器方向上施加加权系数实现特征驱动的生成干预；同时对输出进行正则化、NLL、几何测度评估。

**📊 数据集**

实验数据集主要为自定义的20个内省性提问和20个程序化控制（食谱、汽车引擎等）提示，生成100/50条样本；另外使用8个留作测试的内省提示进行OOD评估。

**📈 对比分析**

与传统单特征标签+单特征驱动检验对比，新协议在三个维度上验证了有效性：①特征#26221在+500系数时产生“沉思哲学家”输出，说明标签误导；②三特征联合抑制导致完整文本变为占位符，单特征抑制不产生此效果；③匹配几何扰动的随机方向不会产生占位符，证明方向决定性；这些结果表明传统协议缺失重要行为，矩阵协议更能捕捉特征因果作用。

**⚠️ 局限性**

局限性包括仅在三款模型（Qwen、Gemma、Llama）和少量特征上验证；随机方向匹配仅在单个系数点完成；未对大规模模型或不同任务进行测试；以及对连贯性评估依赖自动指标，缺乏人工评估。

---

## 283. EvoJail: Evolutionary Diverse Jailbreak Prompt Generation for Large Language Models

**arXiv ID:** 2605.02921 | [PDF](https://arxiv.org/pdf/2605.02921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 284. Calibration of the underlying surface parameters for urban flood using latent variables and adjoint equation

**arXiv ID:** 2605.02959 | [PDF](https://arxiv.org/pdf/2605.02959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 285. The Infinite Mutation Engine? Measuring Polymorphism in LLM-Generated Offensive Code

**arXiv ID:** 2605.03619 | [PDF](https://arxiv.org/pdf/2605.03619v1)

**作者:** Gabriel Hortea `[一作]` (Universidad Carlos III de Madrid), Juan Tapiador `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 5127 | [OpenAlex ID](https://openalex.org/A5070199150)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了一个双代理四阶段流水线，利用Claude Opus 4.6自动生成、验证并迭代4阶段Lua恶意代码，系统评估其产生的多态性；

**💡 创新点**

创新点在于首次对LLM生成恶意代码的结构与语义多样性进行量化测评，并提出基于历史注入的显式多态化提示策略；

**🔧 技术方法**

主要技术包括Prompt工程、双代理（生成器+测试器）自适应循环、AST编辑距离与代码嵌入余弦距离度量、DBSCAN聚类与成本估算；

**📊 数据集**

使用自构造的Lua恶意代码数据集，共生成200个样本（100个天然多态、100个显式多态），涵盖遍历、加密、外泄与集成四个模块；

**📈 对比分析**

对比天然与显式两种提示模式，结果显示AST距离平均在0.76–0.92之间，语义距离在0.02–0.30之间；显式模式提升了结构多样性同时略微增加语义差异，成本约为$0.41–$0.73/样本；

**⚠️ 局限性**

局限性包括仅评估单一语言（Lua）和单一模型（Claude Opus 4.6），且聚类与距离度量受模型特性限制，未覆盖运行时行为或不同硬件平台的适配。

---

## 286. The Parameterized Complexity of Scheduling with Precedence Delays: Shuffle Product and Directed Bandwidth

**arXiv ID:** 2605.03727 | [PDF](https://arxiv.org/pdf/2605.03727v1)

**作者:** Hans L. Bodlaender `[一作]` (Utrecht University), Maher Mallem `[通讯]` (Inria)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了带前置约束延迟的调度问题及其与 Shuffle Product 和 Directed Bandwidth 问题的参数化复杂度。

**💡 创新点**

首次证明了 Binary Shuffle Product、Directed Bandwidth（在 DAG、树等结构上）以及相关调度问题均为 XNLP‑complete，扩展了先前仅有的 W[2]/W[t] 难度结果，揭示了这些问题的真实难度阶层。

**🔧 技术方法**

使用了从 NNCCM Acceptance 的参数化归约、对字母序列的块映射构造、拓扑排序带宽分析，以及对树形结构的尾部（tail）技术来保持宽度约束。

**📊 数据集**

本研究为理论分析，未使用具体实验数据集；所有结论均来自构造性证明与复杂度归约。

**📈 对比分析**

通过构造性的 NP/XP 归约与空间/时间上限的证明，表明这些问题在 XNLP 范畴内，没有已知的 f(k)n^O(1) 空间的 XP 算法；对比之前仅证明 W[2] 难度的结果，显著提升了对问题复杂度的认识。

**⚠️ 局限性**

仍存在若干开放问题，例如单机最小延迟调度在常数延迟下的 NP 难度、带宽约束下的树形图是否为 FPT、以及带宽为 2 的情况是否可多项式解决等。

---

## 287. Ortho-Hydra: Orthogonalized Experts for DiT LoRA

**arXiv ID:** 2605.03252 | [PDF](https://arxiv.org/pdf/2605.03252v1)

**作者:** Seunghyun Ji `[一作]` `[通讯]`, Seunghyun Ji

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在多风格扩散模型（DiT）上引入了一种新的LoRA适配器结构 Ortho‑Hydra，解决了传统 HydraLoRA 在冷启动阶段出现的专家同质化（style bleed）问题。

**💡 创新点**

创新点在于：1) 用 SVD 选取预训练权重的 top‑(Er) 左奇异向量，将其划分为互不重叠的子空间，每个专家仅在自己子空间内进行旋转；2) 采用 Cayley 旋转保证正交性，保持跨专家正交性不变；3) 在 router 输入使用 RMS‑池化的低秩激活，而非原始输入，避免 DC‑消除导致的梯度消失；4) 通过结构性分离打破零初始化时的对称梯度，避免了原始的专家热身（warm‑up）和 σ‑噪声扰动等经验性技巧。

**🔧 技术方法**

技术手段包括：LoRA、Mixture‑of‑Experts LoRA、Orthogonal Fine‑Tuning（OFT）、Cayley 旋转、SVD 预计算、RMS‑池化、Switch‑Transformer 负载平衡损失、AdamW 优化器。

**📊 数据集**

使用了基于 2.4k 张多风格动漫图像（不同艺术家）及其文本说明的数据集，采用 Qwen3‑0.6B 文本编码器和 qwen‑image VAE 的 latents 作为输入；训练集经过掩码（SAM3 + manga‑text‑detector）处理。

**📈 对比分析**

比较方法：在相同的 DiT 基础模型、相同的优化器和调度下，分别训练三种 HydraLoRA 变体（naive、jittered、ortho）。评价指标为路由器的归一化熵（越低表示越专业化）和训练损失。结果显示：naive 与 jittered 在 1k 步内路由熵保持 ≈1，无法脱离均匀 prior；Ortho‑Hydra 在约 200 步后熵下降到 ≈0.8，最终收敛到 ≈0.35，训练损失与基线相近，说明改进主要体现在路由器的学习而非整体性能。

**⚠️ 局限性**

局限性：1) 未评估终端生成质量（多风格图像生成效果）和对不同 E、r、平衡损失权重的敏感性；2) 仅在 DiT 任务上验证，缺乏跨模型、跨数据集的通用性实验；3) 依赖预训练权重的前  Er 奇异向量，假设主子空间足够表达新风格，若新风格需要超出该子空间的能力则可能受限；4) 只在单 GPU 训练下进行，未考察大规模分布式训练时的数值稳定性。

---

## 288. Rethinking the Rank Threshold for LoRA Fine-Tuning

**arXiv ID:** 2605.03724 | [PDF](https://arxiv.org/pdf/2605.03724v1)

**作者:** Juneyoung Park `[一作]` `[通讯]` (OptAI Inc.), Juneyoung Park (OptAI Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究LoRA在神经切线核（NTK） regime 下的低秩阈值问题，尤其聚焦二分类少样本情形

**💡 创新点**

提出更弱的低秩足够条件 r(m+n)-r²>C*·KN（C*≈1.35），证明交叉熵不需要秩阈值，并给出Rademacher复杂度分析表明二分类下秩1即最优

**🔧 技术方法**

使用非对称Sard计数、NTK线性化、Polyak–Łojasiewicz不等式、Rademacher复杂度上界、Marchenko–Pastur自洽分析

**📊 数据集**

在GLUE四个二分类任务（SST‑2、QNLI、MR、QQP）和MNLI多分类任务上验证；使用RoBERTa-base/large、BERT、DistilBERT等编码器

**📈 对比分析**

对比原始LoRA阈值 r≥12，实验显示 r=1 在所有二分类任务、不同模型、不同层、不同样本量下性能相当或更好；在MNLI中 r>1 更优，符合理论预期

**⚠️ 局限性**

理论基于Gaussian‑iid特征和PL假设，C*≈1.35 仅在合成设置下可量化；未给出多分类 K>2 的精确阈值；实验证明NTK矩阵非高斯时仅保留定性一致性

---

## 289. Safety in Embodied AI: A Survey of Risks, Attacks, and Defenses

**arXiv ID:** 2605.02900 | [PDF](https://arxiv.org/pdf/2605.02900v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 290. Mitigating the reconstruction-detection trade-off in VAE-based unsupervised anomaly detection

**arXiv ID:** 2605.02918 | [PDF](https://arxiv.org/pdf/2605.02918v1)

**作者:** Agathe Senellart `[一作]`, Ninon Burgos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

研究了 β‑VAE 在医学图像（FDG PET）体素级无监督异常检测中的重构质量与异常检测性能之间的权衡，并通过种子波动性与潜在分布距离的关联进一步探究模型稳定性。

**💡 创新点**

①首次系统量化潜在空间约束对重构与检测的影响；②提出通过周期性 β‑调度和稀疏 VAE（Gaussian dropout）两种方法减轻该权衡；③利用潜在分布距离解释随机种子引起的性能波动。

**🔧 技术方法**

采用 β‑VAE、周期性 β‑调度、稀疏 VAE；使用模拟的 AD 异常图像生成与真实 AD 数据；评估指标包括 hhMSE、ahMSE、rMSE、平均精度 AP、Dice 以及样本级 AUC；对比不同 β、潜在维度、调度与稀疏策略。

**📊 数据集**

ADNI 数据集的 3D FDG PET 扫描，训练 545 张健康图像，测试 50 张健康与 50 张真实 AD 病例，另外通过模拟得到的 AD‑30、AD‑50 级别的异常图像。

**📈 对比分析**

与基准 β‑VAE（d=64，β=10）相比，周期性 β‑调度在保持重构质量的同时提升 AP 约 3–5%，稀疏 VAE 在所有指标（rMSE↓、AP↑、Dice↑）上均优于基准，并在 30 个随机种子上表现出更低的 AP 方差（2.0 vs 3.3）。在真实 AD 数据上，所有模型的样本级 AUC 均在 80–81 之间，差异不显著。

**⚠️ 局限性**

仅针对 β‑VAE 进行了实验，未尝试其他潜在空间正则化方法；异常图像采用仿真生成，缺乏真实异常对；模型对训练数据纯度和超参数选择仍敏感；结果的通用性（非 AD PET 场景）尚未验证。

---

## 291. TriBench-Ko: Evaluating LLM Risks in Judicial Workflows

**arXiv ID:** 2605.03792 | [PDF](https://arxiv.org/pdf/2605.03792v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 292. SHIELD: A Diverse Clinical Note Dataset and Distilled Small Language Models for Enterprise-Scale De-identification

**arXiv ID:** 2605.03301 | [PDF](https://arxiv.org/pdf/2605.03301v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 293. Will the Carbon Border Adjustment Mechanism Impact European Electricity Prices? A GNN-Based Network Analysis

**arXiv ID:** 2605.03304 | [PDF](https://arxiv.org/pdf/2605.03304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 294. PriorNet: Prior-Guided Engagement Estimation from Face Video

**arXiv ID:** 2605.03615 | [PDF](https://arxiv.org/pdf/2605.03615v1)

**作者:** Alexander Vedernikov `[一作]` `[通讯]`, Alexander Vedernikov

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 PriorNet 框架，通过在预处理、模型适配与损失设计三个阶段注入任务相关先验，实现面部视频参与度估计；

**💡 创新点**

创新点在于①将面部检测失败显式编码为零帧占位符，②在冻结的 SVFAP 主干上插入 Prior‑LoRA 低秩自适应模块实现参数高效微调，③结合 Dirichlet‑evidential 与不确定性加权的目标函数处理主观标签；

**🔧 技术方法**

采用 OpenCV SSD 进行面部检测、SVFAP 自监督视频表情感知主干、Prior‑LoRA 低秩适配、Dirichlet‑evidential 与 uncertainty‑weighted 损失；

**📊 数据集**

在 EngageNet、DAiSEE、DREAMS 与 PAFE 四大面部视频参与度基准上进行实验；

**📈 对比分析**

与各数据集原始最高表现对比，PriorNet 在 EngageNet 提升 2.34% 计分、DAiSEE 提升 1.58% 计分、DREAMS 提升 0.081 F1、PAFE 提升 0.04 F1；

**⚠️ 局限性**

局限性包括占位符依赖面部检测器表现、评测协议多样导致跨数据集可比性有限、仅使用硬标签而非完整投票分布，且验证主要集中在现有基准上。

---

## 295. From Informal Addresses to Reliable Places: Participatory Data Governance of Civic Addressing in Puerto Rico

**arXiv ID:** 2605.02924 | [PDF](https://arxiv.org/pdf/2605.02924v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 296. Unifying Dynamical Systems and Graph Theory to Mechanistically Understand Computation in Neural Networks

**arXiv ID:** 2605.03598 | [PDF](https://arxiv.org/pdf/2605.03598v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 297. Vanishing L2 regularization for the softmax Multi Armed Bandit

**arXiv ID:** 2605.03752 | [PDF](https://arxiv.org/pdf/2605.03752v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 298. APEX: Large-scale Multi-task Aesthetic-Informed Popularity Prediction for AI-Generated Music

**arXiv ID:** 2605.03395 | [PDF](https://arxiv.org/pdf/2605.03395v1)

**作者:** Jaavid Aktar Husain `[一作]` (Singapore University of Technology and Design), Dorien Herremans `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 1808 | [OpenAlex ID](https://openalex.org/A5069548004)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出APEX，一个大规模多任务学习框架，可同时预测AI生成音乐的受欢迎度（流量与点赞分数）和五个美学维度。

**💡 创新点**

首个将受欢迎度与美学指标联合训练的多任务模型，利用MERT音频表示实现对未见生成系统的良好泛化。

**🔧 技术方法**

技术包括冻结的MERT Transformer音频编码器、分层共享全连接网络、任务加权损失（不确定性加权）以及歌曲级或片段级输入模式。

**📊 数据集**

训练数据为211k首来自Udio和Suno的AI生成音乐（约10k小时音频），评估数据为Music Arena的人类偏好对决集。

**📈 对比分析**

在24种实验配置中，使用不确定性加权与歌曲级聚合的配置取得最佳效果；在Music Arena上使用APEX预测的人类偏好AUC约为0.64，显著优于仅使用受欢迎度特征的基线。

**⚠️ 局限性**

局限性：在人声轨道上表现不足，模型仅基于音频特征，未利用歌词或其他元数据信息；对音频质量异常的鲁棒性待提升。

---

## 299. When Agents Handle Secrets: A Survey of Confidential Computing for Agentic AI

**arXiv ID:** 2605.03213 | [PDF](https://arxiv.org/pdf/2605.03213v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 300. Beyond Activation Alignment: The Geometry of Neural Sensitivity

**arXiv ID:** 2605.03222 | [PDF](https://arxiv.org/pdf/2605.03222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 301. A Comparison of Traditional Machine Learning Algorithms and LSTM-Based Deep Learning Models for Email Sentiment Analysis

**arXiv ID:** 2605.03440 | [PDF](https://arxiv.org/pdf/2605.03440v1)

**作者:** Virdio Samuel Saragih `[一作]` (Sumatra Institute of Technology), Martin C. T. Manullang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了使用 Word2Vec 特征对印尼语垃圾邮件进行分类和情感检测，并对传统机器学习模型（SVM、Logistic Regression、Naive Bayes）与深度学习模型（LSTM）在同一特征空间下的性能进行了系统比较。

**💡 创新点**

创新点在于：①统一采用 Word2Vec 词向量对文本进行特征抽取；②在相同的词向量表示下，对 SVM、Logistic Regression、Naive Bayes 与 LSTM 进行公平对比；③发现线性 SVM 在准确率和训练速度上明显优于 LSTM，提供了针对密集词向量的实证结论。

**🔧 技术方法**

技术方法包括：文本预处理（清洗、去停用词）；Word2Vec 词向量训练与平均投影；SVM（LinearSVC）、Logistic Regression、Gaussian Naive Bayes 的经典实现；LSTM 神经网络（单层双向、Embedding 64 维、二分类 Sigmoid）；交叉验证、混淆矩阵、准确率、AUC、召回率、精确率、F1 等评价指标。

**📊 数据集**

使用的数据集为 Kaggle 上公开的“印尼语垃圾邮件数据集”，共 2,620 条记录（Spam 1,363 条，Ham 1,258 条），经过 80/20 的训练/测试划分后进行实验。

**📈 对比分析**

实验采用相同的 80/20 划分、相同的 Word2Vec 特征，分别训练三种传统模型和 LSTM；评估指标为准确率、AUC、召回率、精确率、F1、训练时长。结果显示：SVM 取得 98.74% 的准确率、0.908 秒训练时间；Logistic Regression 97.53%、2.695 秒；Naive Bayes 94.49%；LSTM 达 97% 准确率、0.97 的 F1、但需要 30 轮训练且耗时明显更长。

**⚠️ 局限性**

局限性包括：仅在单一印尼语邮件数据集上验证，缺乏跨语言或更大规模数据集的泛化测试；LSTM 的训练时间较长且未尝试更高效的网络架构或预训练模型；实验未探讨不同词向量方法（如 BERT、FastText）的影响；缺少对模型鲁棒性和可解释性的深入分析。

---

## 302. Pose Tracking with a Foundation Pose Model and an Ensemble Directional Kalman Filter

**arXiv ID:** 2605.03105 | [PDF](https://arxiv.org/pdf/2605.03105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 303. Nondeterministic state complexity of square root

**arXiv ID:** 2605.02957 | [PDF](https://arxiv.org/pdf/2605.02957v1)

**作者:** Sergey Onishchenko `[一作]` `[通讯]` (Saint Petersburg University), Sergey Onishchenko (Saint Petersburg University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

证明了对任意 n‑状态 NFA，平方根操作 √(L) 的最坏情况非确定性状态复杂度正好为 n³，并给出了匹配的上界与下界构造。

**💡 创新点**

创新点在于彻底消除了之前已知的上界 n³ 与下界 (n‑1)(n‑2)(n‑3) 之间的空隙，确立了精确的极限；并提供了一个在 n≥6 的情形下的显式 NFA 构造实现该极限。

**🔧 技术方法**

使用了非确定性有限自动机理论、状态复杂度分析、fooling set 技巧以及构造性证明方法。

**📊 数据集**

无数据集，本研究为纯理论算法与形式语言分析。

**📈 对比分析**

通过构造上界 NFA（n³ 状态）与下界 NFA（证明任何 NFA 必须至少 n³ 状态），实现了理论上最优性能，证明了 n³ 是最小可能的上界。

**⚠️ 局限性**

局限性包括：仅适用于平方根操作；构造仅在 n≥6 时有效；未讨论对其他语言运算的推广。

---

## 304. Mixed-Precision Information Bottlenecks for On-Device Trait-State Disentanglement in Bipolar Agitation Detection

**arXiv ID:** 2605.03039 | [PDF](https://arxiv.org/pdf/2605.03039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 305. Dependency-Aware Privacy for Multi-turn Agents

**arXiv ID:** 2605.03188 | [PDF](https://arxiv.org/pdf/2605.03188v1)

**作者:** Divyam Anshumaan `[一作]` (University of Wisconsin-Madison), Somesh Jha `[通讯]` (University of Wisconsin-Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出RootGuard，一种在多轮LLM代理交互中仅对根属性噪声一次，后续所有发布均为对已噪声根的确定性后处理，消除多轮信息泄露；

**💡 创新点**

创新点在于依赖结构知识（根间关系与目标函数）进行预算分配与噪声共计，仅需一次噪声即可保证所有派生值的隐私，并通过双重不对称性提升效用；

**🔧 技术方法**

技术包括：度量差分隐私（mDP）、格式化加密（FPE）、指数/拉普拉斯/阶梯噪声机制、敏感度权重预算分配、DAG状态管理、MAP重构攻击分析；

**📊 数据集**

使用CDC NHANES 2017‑2018公开健康调查数据，构造8个医学诊断模板（如FIB‑4、AIP、HOMA等）进行实验；

**📈 对比分析**

与传统独立噪声方法（ℳ‑All）对比，在ε=0.1、B=(2k+1)ε下，RootGuard wMAPE降低2.6‑3.8倍，风险类别误差降低2.2‑3.3倍；在MAP重构攻击下，RootGuard保持不变，独立噪声随查询数下降；在LLM工具调用部署（21,600会话）中保持相同优势；

**⚠️ 局限性**

限制包括：假设完美命名实体识别、仅评估数值型根属性、未考虑多用户/多模态场景、未深入探讨非线性派生函数下的攻击、未将噪声学习的目标函数集成到机制中。

---

## 306. Capability centrality: the next step from scale-free property

**arXiv ID:** 2605.03796 | [PDF](https://arxiv.org/pdf/2605.03796v1)

**作者:** Mikhail Tuzhilin `[一作]` `[通讯]` (Higher School of Economics), Mikhail Tuzhilin (Higher School of Economics)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并研究了新的中心性度量ksi‑centrality及其归一化版本，探讨其分布特征及与网络结构的关联；

**💡 创新点**

创新点在于：①定义了基于邻居拓扑的中心性度量；②发现其分布对真实网络与随机网络具有区分性；③将平均归一化ksi系数与Barabasi‑Albert模型的m参数建立一一对应关系；④与代数连通度、Cheeger值等经典指标相联系；

**🔧 技术方法**

使用了统计学方法（Weibull拟合、Pearson偏度系数）、图论概念（邻接矩阵、拉普拉斯算子）、理论推导（期望值、渐近性质）以及程序实现（Python实现并发布在GitHub）；

**📊 数据集**

利用了40个来自生物、社交、互联网、交通等领域的真实网络数据集，并对Erdos‑Renyi、Watts‑Strogatz、Barabasi‑Albert、Boccaletti‑Hwang‑Latora等四种模型网络进行对比；

**📈 对比分析**

通过比较ksi分布的偏度系数、拟合曲线和与度分布的相似度，发现真实网络的ksi分布右偏且接近Weibull分布，随机与模型网络则呈中心对称；偏度阈值1可有效区分；该方法在区分真实与人工网络方面表现优于仅基于度分布的方法；

**⚠️ 局限性**

局限性包括：①对Barabasi‑Albert和Watts‑Strogatz等模型参数极小值时仍能产生右偏分布，影响区分；②ksi未满足Freeman星形性质，无法完美捕捉某些节点重要性；③对大规模网络的计算复杂度及归一化ksi随节点数趋近0的性质限制了其在某些应用场景下的使用。

---

## 307. Refining Compositional Diffusion for Reliable Long-Horizon Planning

**arXiv ID:** 2605.03075 | [PDF](https://arxiv.org/pdf/2605.03075v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 308. Fast Strategy Solving for the Informed Player in Two-Player Zero-Sum Linear-Quadratic Differential Games with One-Sided Information

**arXiv ID:** 2605.03112 | [PDF](https://arxiv.org/pdf/2605.03112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 309. Healthcare AI GYM for Medical Agents

**arXiv ID:** 2605.02943 | [PDF](https://arxiv.org/pdf/2605.02943v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 310. Stream-R1: Reliability-Perplexity Aware Reward Distillation for Streaming Video Generation

**arXiv ID:** 2605.03849 | [PDF](https://arxiv.org/pdf/2605.03849v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 311. In-memory Multidimensional Indexing Using the skd-tree

**arXiv ID:** 2605.03640 | [PDF](https://arxiv.org/pdf/2605.03640v1)

**作者:** Achilleas Michalopoulos `[一作]` (University of Ioannina), Nikos Mamoulis `[通讯]` (University of Ioannina)

**通讯引用:** 12150 | [OpenAlex ID](https://openalex.org/A5045731304)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在内存中高效支持多维范围查询和k近邻查询的切片kd树（slicing kd-tree）结构，支持动态插入和删除。

**💡 创新点**

创新点在于：①每个节点在单一维度上做多分裂，极大提升 fan‑out；②使用前缀压缩存储分裂阈值，减少树高；③采用 SIMD 并行比较实现无分支搜索；④设计顶层自适应构造、叶子类型（轻、重、异常）与增量更新算法。

**🔧 技术方法**

核心技术包括多分裂 kd‑tree、前缀压缩（N16、N32、N64 变体）、SIMD 并行比较、基于堆的最佳优先 kNN、以及自适应的 top‑down 构造与增量更新策略。

**📊 数据集**

实验使用了六个真实数据集：EDGES（2D）、TORONTO（3D）、OOKLA（5D）、GAIA（6D）、NYT（8D）和 TPC‑H（8D），并在统一/高斯合成数据上验证可扩展性。

**📈 对比分析**

与 Boost R‑tree、nanoflann kd‑tree、PH‑tree、Uniform/Adaptive Grid、Flood、IFI 等经典及学习型索引对比，切片kd树在范围查询和kNN查询中平均比 R‑tree 快 1.5‑2×，在混合更新+查询工作负载中比 R‑tree 快数倍至十倍，且内存占用比 R‑tree 低 36–39%。

**⚠️ 局限性**

局限性包括：①对极端高维或极度不均匀分布的数据可能需要更高 fan‑out 或更频繁的重构；②学习型索引在动态更新时需要重训练；③对极大并发更新场景尚未评估，需要进一步的多线程实现。

---

## 312. SPEC CPU2026: Characterization, Representativeness, and Cross-Suite Comparison

**arXiv ID:** 2605.03713 | [PDF](https://arxiv.org/pdf/2605.03713v1)

**作者:** Ruihao Li `[一作]` (University of Texas at Austin), Lizy K. John `[通讯]` (University of Texas at Austin)

**通讯引用:** 7864 | [OpenAlex ID](https://openalex.org/A5068885069)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 SPEC CPU2026 进行全面表征，涵盖九款现代 CPU（Intel、AMD、Ampere、Nvidia），分析其指令量、内存占用和缓存压力；通过聚类代表性分析提炼 4–5 个工作负载子集，保持 96.4–99.9% 的完整套件行为；与 SPEC CPU2017、DCPerf、MLPerf 进行跨套件微架构比较，并通过回调轮询（round‑robin stagger）生成代理工作负载，缩小 IPC 差距至 13.7%；通过页面大小、内存分配器、预取、编译器优化、ISA 敏感性及多核扩展等案例，展示其在实际架构评估中的可用性。

**💡 创新点**

① 用聚类技术提炼高代表性的子集，大幅降低评估成本；② 引入回调轮询模式生成 DCPerf 代理负载，显著提升与实际数据中心工作负载的接近度；③ 将 SPEC CPU2026 与多套基准在微架构层面进行统一对比，验证其作为通用 CPU 评测基准的有效性；④ 在案例研究中展示基准能驱动具体架构优化。

**🔧 技术方法**

聚类代表性分析、跨套件微架构指标对比、回调轮询（round‑robin stagger）代理生成、各类案例实验（页面大小、分配器、预取、编译器、ISA、扩展性）。

**📊 数据集**

SPEC CPU2026（完整套件）、SPEC CPU2017、DCPerf、MLPerf；测试平台包括最近的 Intel、AMD、Ampere、Nvidia 处理器，总计九个平台。

**📈 对比分析**

采用统一的微架构指标（如前端压力、指令缓存压力、向量压力等）对四个基准进行交叉比较；结果表明 SPEC CPU2026 在指令量和内存占用上更贴近真实工作负载；相比 MLPerf 更不向量化，且前端压力低于 DCPerf；通过回调轮询模式，SPEC CPU2026 代理工作负载将 IPC 差距压缩至 13.7%，说明其性能表现更接近 DCPerf。

**⚠️ 局限性**

1）仅覆盖 CPU 侧工作负载，未涵盖 GPU/加速器场景；2）代表性子集的聚类依据可能因平台差异或工作负载变化而失效；3）研究仅在九款主流平台上验证，其他新兴体系结构的适用性尚待进一步评估；4）基准本身仍受限于工作负载选择，可能无法捕捉所有新兴瓶颈。

---

## 313. Cosmodoit: A Python Package for Adaptive, Efficient Pipelining of Feature Extraction from Performed Music

**arXiv ID:** 2605.03541 | [PDF](https://arxiv.org/pdf/2605.03541v1)

**作者:** Corentin Guichaoua `[一作]` (STMS Laboratoire CNRS IRCAM Sorbonne Université Ministère de la Culture), Elaine Chew `[通讯]` (King's College London)

**通讯引用:** 2508 | [OpenAlex ID](https://openalex.org/A5034873733)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了Cosmodoit Python包，用于音乐表演分析的特征提取与表演-乐谱对齐。

**💡 创新点**

将多语言（Python、C++、Matlab）算法统一封装、采用模块化设计与依赖追踪，实现单命令增量更新与高效管线。

**🔧 技术方法**

使用Python与Doit构建系统、配置文件驱动参数、包装C++/Matlab代码，并通过可扩展模块化结构实现功能集成。

**📊 数据集**

以表演音频/MIDI和乐谱为输入，示例数据为Raoul Pugno演奏的肖邦夜曲；此外包含了多种已实现的特征子模块。

**📈 对比分析**

论文未给出具体实验对比；强调Cosmodoit可通过单命令更新、自动跳过不必要步骤、避免重复计算，从而显著降低计算时间与错误率。

**⚠️ 局限性**

当前仅集成三大子模块，缺乏多算法替代；需要手动添加新模块，且对文件格式的兼容性与跨语言接口仍有一定限制。

---

## 314. Reasoning-Guided Grounding: Elevating Video Anomaly Detection through Multimodal Large Language Models

**arXiv ID:** 2605.02912 | [PDF](https://arxiv.org/pdf/2605.02912v1)

**作者:** Sakshi Agarwal `[一作]` (Accenture), Ankit Parag Shah `[通讯]` (Accenture)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Vanguard框架，整合视频异常检测、链式思考解释与空间定位于单一多模态大语言模型中

**💡 创新点**

通过三阶段自适应训练（分类预热、LoRA空间定位、链式思考微调）与教师-学生自动标注管道，突破VLM在异常检测中空间定位不准的问题

**🔧 技术方法**

使用Qwen3-VL-4B视觉语言模型+LoRA适配，GroundingDINO进行定位，链式思考与文本坐标GIoU损失共同训练

**📊 数据集**

在UCF‑Crime、XD‑Violence、ShanghaiTech校园三大公开视频异常检测数据集上进行实验

**📈 对比分析**

与零样本与微调VLM方法对比，Vanguard在UCF‑Crime上实现ROC‑AUC 93.78%、F1 83.6%，并首次给出平均IoU 0.62 的空间定位指标，整体性能优于现有基线

**⚠️ 局限性**

仍面临小目标与稀疏类别定位困难、模型规模导致推理成本高、教师模型偏差影响等局限

---

## 315. Beyond Similarity Search: A Unified Data Layer for Production RAG Systems

**arXiv ID:** 2605.03275 | [PDF](https://arxiv.org/pdf/2605.03275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 316. Distribution-Free Pretraining of Classification Losses via Evolutionary Dynamics

**arXiv ID:** 2605.03722 | [PDF](https://arxiv.org/pdf/2605.03722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 317. Unified Multimodal Visual Tracking with Dual Mixture-of-Experts

**arXiv ID:** 2605.03716 | [PDF](https://arxiv.org/pdf/2605.03716v1)

**作者:** Lingyi Hong `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**通讯引用:** 21013 | [OpenAlex ID](https://openalex.org/A5100441502)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出统一多模态跟踪框架OneTrackerV2，支持RGB与RGB+X多模态输入，并实现一次性训练；

**💡 创新点**

创新点在于引入Meta Merger实现跨模态特征统一，以及Dual Mixture-of-Experts (DMoE)分离时空关系与多模态融合，显著缓解特征冲突与缺失模态鲁棒性问题；

**🔧 技术方法**

采用视觉Transformer+低秩Mixture-of-Experts，结合软门控、正交解耦损失、路由聚类正则、随机模态扰动以及模型压缩技术；

**📊 数据集**

使用LaSOT、TrackingNet、GOT-10k、COCO、VASTTrack、DepthTrack、VisEvent、LasHeR、TNL2K等RGB与RGB+X跟踪数据集进行训练与评估；

**📈 对比分析**

通过与现有RGB和RGB+X跟踪器在12个基准、5个任务的对比，OneTrackerV2实现SOTA性能，即使在缺失模态或压缩模型情况下亦保持高准确率；

**⚠️ 局限性**

局限性包括对大规模训练集的依赖、模型复杂度仍相对较高，以及对极端缺失模态或新模态的泛化能力尚未充分验证。

---

## 318. Bridging the Embodiment Gap: Disentangled Cross-Embodiment Video Editing

**arXiv ID:** 2605.03637 | [PDF](https://arxiv.org/pdf/2605.03637v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 319. Feasibility-aware Hybrid Control for Motion Planning under Signal Temporal Logics

**arXiv ID:** 2605.03662 | [PDF](https://arxiv.org/pdf/2605.03662v1)

**作者:** Panagiotis Rousseas `[一作]` (KTH Royal Institute of Technology), Dimos V. Dimarogonas `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 18994 | [OpenAlex ID](https://openalex.org/A5055348953)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于S-Temporal Logic控制栏函数（STL-CBF）的混合任务与运动规划框架，能够在任意复杂的平面工作空间中处理重叠的时空任务，并通过离散配置变量实现任务可行性判断与动态选择；

**💡 创新点**

核心创新包括：1）利用双向同胚变换将非凸工作空间映射为圆盘，从而在平滑域上设计防死锁的STL-CBF；2）将任务约束与输入约束统一到二次规划（CBF‑QP）中，并通过离散配置向量对冲突任务进行在线可行性搜索；3）提出基于优先级的启发式配置搜索算法，降低搜索复杂度；

**🔧 技术方法**

技术手段包括：同胚变换、STL-CBF构造（Eventually、Always、Until）、前向不变性分析、二次规划控制、线性规划可行性检验、启发式配置搜索；

**📊 数据集**

使用仿真测试数据，无公开数据集；通过MATLAB对8个重叠STL约束的平面机器人进行仿真；

**📈 对比分析**

与传统通过松弛变量直接放宽所有STL‑CBF约束的方法进行对比；结果表明，所提方法在受限输入条件下能满足6/8个任务，松弛方法仅满足1个；计算时间略高（约0.1 s/搜索，CBF‑QP时间相近），但显著提升了任务满足率；

**⚠️ 局限性**

局限性包括：①方法不完整，无法保证所有可满足任务被实现；②同胚映射可能导致几何失真，影响约束满足；③配置搜索为NP‑hard，尽管采用启发式，但在任务数大时仍可能耗时；④参数（如δ、z）需人工调节，缺乏自适应机制。

---

## 320. LiteShield: Hybrid Feature Selection-Driven Lightweight Intrusion Detection for Resource-Constrained IoT Networks

**arXiv ID:** 2605.02987 | [PDF](https://arxiv.org/pdf/2605.02987v1)

**作者:** Dileepa Mabulage `[一作]` (Informatics Institute of Technology), Banuka Athuraliya `[通讯]` (Informatics Institute of Technology)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5008212798)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了LiteShield轻量级IoT入侵检测框架，结合混合特征选择与轻量级机器学习分类器实现准确且资源友好的检测。

**💡 创新点**

创新点在于采用MI+RFECV双阶段混合特征选择，系统性比较六种轻量级分类器，并在评估中同时关注模型尺寸、推理延迟等资源指标。

**🔧 技术方法**

技术包括Mutual Information、Recursive Feature Elimination with Cross-Validation、Decision Tree、Random Forest、K-Nearest Neighbors、Logistic Regression、Naïve Bayes、Support Vector Machine，配合数据预处理与平衡采样。

**📊 数据集**

使用UNSW-NB15数据集进行实验。

**📈 对比分析**

通过二分类和多分类的准确率、F1、模型大小、推理延迟等指标进行对比；KNN在准确率上最高（二分类98.26%，多分类85.22%），但模型尺寸与延迟最高；Random Forest在保持高准确率（二分类98.01%，多分类80.39%）的同时，模型尺寸（1.9 MB/4.32 MB）和推理速度最优。

**⚠️ 局限性**

局限性：仅在UNSW-NB15上评估；未在真实IoT硬件上验证；对零日或稀有攻击的处理不足；未探索量化或压缩的深度学习方案。

---

## 321. GeoTopoDiff: Learning Geometry--Topology Graph Priors through Boundary-Constrained Mixed Diffusion for Sparse-Slice 3D Porous Reconstruction

**arXiv ID:** 2605.03764 | [PDF](https://arxiv.org/pdf/2605.03764v1)

**作者:** Yue Shi `[一作]` (Manchester Metropolitan University), Liangxiu Han `[通讯]` (Manchester Metropolitan University)

**通讯引用:** 4052 | [OpenAlex ID](https://openalex.org/A5023625466)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

利用混合连续-离散图状态空间，GeoTopoDiff从稀疏的μCT切片重建3D多孔微结构；

**💡 创新点**

创新点在于把扩散先验从体素空间迁移到混合图空间，并引入稀疏边界图先验以约束逆向采样，显著提升拓扑和传输一致性；

**🔧 技术方法**

核心技术包括混合图扩散模型、稀疏边界图提取、图到体素解码器、基于深度扩散的后验采样与边界投影；

**📊 数据集**

实验使用了PTFE（anisotropic fibrous）和Fontainebleau sandstone两种μCT数据集；

**📈 对比分析**

与六类基线（统计、GAN、2D→3D扩散、体素扩散、隐式+扩散、图扩散）在形态、拓扑及渗流指标上对比，GeoTopoDiff在TPCF、PSD、渗流均值误差和断连率等指标上平均提升约30%至40%；

**⚠️ 局限性**

主要限制是计算开销相对更高，且仅在稀疏边界条件下测试，需进一步优化速度与适用性。

---

## 322. Say the Mission, Execute the Swarm: Agent-Enhanced LLM Reasoning in the Web-of-Drones

**arXiv ID:** 2605.03788 | [PDF](https://arxiv.org/pdf/2605.03788v1)

**作者:** Andrea Iannoli `[一作]` (University of Bologna), Marco Di Felice `[通讯]` (University of Bologna)

**通讯引用:** 4913 | [OpenAlex ID](https://openalex.org/A5032452851)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基于LLM的无人机群管理框架，用户用自然语言描述任务目标，系统通过无代码、持续闭环的LLM+WoT+MCP架构实现自主执行；

**💡 创新点**

创新点包括：①利用W3C Web of Things和MCP协议将多种无人机与传感器统一抽象为标准化Thing，②通过LLM函数调用实现无代码连续执行而非一次性代码生成，③引入运行时guardrail和helper tools，提升推理安全性与执行鲁棒性；

**🔧 技术方法**

核心技术包括：大型语言模型（GPT‑5.2、DeepSeek‑V3.2、GLM‑4.7、Grok‑4.1 Fast、Claude‑Haiku‑4.5、Qwen‑3‑8B）配合LangChain框架；Web of Things标准与Thing Description Directory（Zion）及MCP网关（NestJS）实现设备互联；ArduPilot SITL、MAVProxy、MAVLink提供无人机仿真与控制；Zod用于工具输入/输出校验；

**📊 数据集**

实验数据集：在ArduPilot仿真环境中使用10架多旋翼无人机、3个地面湿度传感器+1个温度传感器，四类任务（区域覆盖、有无规划工具；星形编队；灌溉决策），任务数据全部由仿真生成，不使用公开真实数据集；

**📈 对比分析**

评估方法：每种LLM在每个任务下执行10次独立跑，记录成功率、执行时间、电量、碰撞数、token消耗。结果显示：带规划工具时GLM成功率100%，Grok 70%；无工具时GPT 100%；编队任务GPT 60% vs GLM 50%；灌溉任务GLM 90%（最高）。token消耗与成功率无直接相关性，显示推理结构与反馈更关键；

**⚠️ 局限性**

局限性：LLM对实时执行仍受限，无法完全持续持久化状态；小模型或低延迟版本在复杂任务上表现不佳；系统高度依赖外部规划与工具，缺少完全自适应能力；安全性与碰撞避免仍需进一步验证；实验仅在仿真环境，真实场景鲁棒性尚未验证。

---

## 323. MedSR-Vision: Deep Learning Framework for Multi-Domain Medical Image Super-Resolution

**arXiv ID:** 2605.03343 | [PDF](https://arxiv.org/pdf/2605.03343v1)

**作者:** Subhash Gurappa `[一作]` (Florida International University), Sundararaj Sitharama Iyengar `[通讯]` (Florida International University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一套统一的MedSR-Vision框架，系统评估了5种医学影像模态（脑MRI、胸部X光、肾脏超声、肾结石CT、脊柱MRI）在×2、×3、×4三种放大倍数下的三种深度学习超分模型。

**💡 创新点**

首次将CNN（SRCNN）、Transformer（SwinIR）和GAN（Real‑ESRGAN）在同一平台下进行多模态、多尺度对比，并给出基于诊断需求的模型选择指导，形成了行业内首个标准化评测体系。

**🔧 技术方法**

采用了SRCNN、SwinIR和Real‑ESRGAN三种代表性深度学习架构，结合PSNR、SSIM、FSIM、LPIPS、ODI、Tenengrad、Laplacian等多维度指标进行量化评估。

**📊 数据集**

使用人工降采样得到的脑MRI、胸部X光、肾脏超声、肾结石CT和脊柱MRI的低分辨率–高分辨率配对数据集，涵盖了不同噪声、纹理和结构特征的真实临床图像。

**📈 对比分析**

通过统一评测脚本在NVIDIA A100 GPU上进行实验，结果显示Real‑ESRGAN在感知质量与锐度上表现最佳，SwinIR在结构保真与诊断特征恢复上占优，SRCNN在低倍率下速度最快且稳定，提供了详细的数值对比与性能阈值。

**⚠️ 局限性**

局限性包括缺乏真实临床降采样数据导致泛化能力不确定，GAN可能产生伪影，且对不同扫描仪和成像条件的适应性不足，需要进一步的临床验证与模型鲁棒性研究。

---

## 324. From Barrier to Bridge: The Case for AI Data Center/Power Grid Co-Design

**arXiv ID:** 2605.03090 | [PDF](https://arxiv.org/pdf/2605.03090v1)

**作者:** Noman Bashir `[一作]` (Massachusetts Institute of Technology), Minlan Yu `[通讯]` (Harvard University)

**通讯引用:** 9454 | [OpenAlex ID](https://openalex.org/A5035157838)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

阐述了人工智能训练数据中心对传统电网负载多样性假设的冲击，并提出需在电网与数据中心之间实现协同设计与运营

**💡 创新点**

提出了四个关键研究方向：联合规划、多时域控制、计算–电力协议栈以及市场创新，形成跨域协同的系统框架

**🔧 技术方法**

结合电网控制理论、数据中心功率电子学、可编程UPS/电池缓冲、以及标准化遥测（如IEEE 2030.5）等技术手段

**📊 数据集**

主要使用公开的电网负载数据、NERC可靠性报告、AI训练规模估计及行业投资数据（如Amazon、Microsoft等）进行分析与案例研究

**📈 对比分析**

该工作为概念性框架，未进行实验验证；通过案例分析与假设性模型展示其可行性，预期能显著提高电网可调节性与数据中心能源弹性

**⚠️ 局限性**

局限在于缺乏实验验证与实际部署案例，需跨行业监管与技术标准协同演进，且对不同规模AI推理负载的适用性尚未充分评估

---

## 325. Pact: A Choreographic Language for Agentic Ecosystems

**arXiv ID:** 2605.03143 | [PDF](https://arxiv.org/pdf/2605.03143v1)

**作者:** Kiran Gopinathan `[一作]` (Basis Research Institute), Eli Bingham `[通讯]` (Basis Research Institute)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出一种名为 Pact 的 choreographic 语言，扩展了传统的协作式编程框架，引入了代理的策略选择、效用函数和外部变量（nature variables），并将协议映射为形式化的博弈，从而使代理能够在多方系统中进行基于博弈论的自我利益决策和协议协商。

**💡 创新点**

创新点在于将 choreographic 编程与博弈理论相结合，首次将代理的自利动机、优先级和先验信息显式编码到协议中，形成一个既能保证死锁自由又能进行策略分析的统一框架；同时实现了基于理论推理的决策策略求解器，并在书店协议上复现了“柠檬市场”现象。

**🔧 技术方法**

技术方法包括：① 设计带有 choice、values、nature 操作的高层协议语言；② 通过端点投影将协议转换为本地程序；③ 在 Python 中嵌入 DSL，利用代数效应实现投影；④ 基于理论推理（Theory‑of‑Mind）构建递归概率模型并进行推断，得到决策策略；⑤ 使用基于博弈论的决策分析求解器。

**📊 数据集**

论文未使用公开数据集，而是通过构造的书店协议作为案例进行演示和分析。

**📈 对比分析**

在示例中，作者使用自定义的决策策略求解器对书店协议进行博弈分析，并与不同递归深度下的理论推理结果进行对比，验证了模型能够捕捉信息不对称下的市场失灵现象，但未给出与现有工具或基准的数值性能比较。

**⚠️ 局限性**

主要限制包括：① 目前缺乏对 Pact 语言语义的形式化定义与证明；② 仅实现了初步的决策策略求解器，尚未支持更复杂的博弈分析如均衡计算和激励相容性检查；③ 现有实现仅在 Python 环境下，缺乏跨语言或大规模部署的验证。

---

## 326. Agent-Based Modeling of Low-Emission Fertilizer Adoption for Dairy Farm Decarbonisation using Empirical Farm Data

**arXiv ID:** 2605.03648 | [PDF](https://arxiv.org/pdf/2605.03648v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 327. Agentic-imodels: Evolving agentic interpretability tools via autoresearch

**arXiv ID:** 2605.03808 | [PDF](https://arxiv.org/pdf/2605.03808v1)

**作者:** Chandan Singh `[一作]` (Microsoft Research), Jianfeng Gao `[通讯]` (Microsoft Research)

**通讯引用:** 35445 | [OpenAlex ID](https://openalex.org/A5114910293)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于代理自动研究循环的算法，自动演化适用于代理可解释性的回归模型库，提升了预测性能与代理可解释性。

**💡 创新点**

首次将LLM可解释性测试作为优化目标，构建代理可解释性度量并用编码代理迭代改进模型，实现了预测-可解释性双目标 Pareto 前沿。

**🔧 技术方法**

采用LLM可解释性测试、代理编码（Claude Code、Codex）、自动化评估循环、Python 类实现和多模型对比评估。

**📊 数据集**

使用65个Tabular回归数据集（OpenML TabArena、PMLB），以及16个OpenML held‑out 数据集进行泛化评估，BLADE 基准评测四个 ADS 代理。

**📈 对比分析**

与16个基线模型（线性、树、加性、规则、黑盒）比较，演化模型在预测排名和代理可解释性测试通过率上均实现Pareto改进，BLADE 结果提升8%–73%。

**⚠️ 局限性**

评价指标易受奖励劫持，LLM评估的偏差与人类可解释性不完全一致，实验成本高，且仅针对回归任务，缺乏更广泛任务验证。

---

## 328. Rational Communication Shapes Morphological Composition

**arXiv ID:** 2605.03510 | [PDF](https://arxiv.org/pdf/2605.03510v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 329. Mantis: Mamba-native Tuning is Efficient for 3D Point Cloud Foundation Models

**arXiv ID:** 2605.03438 | [PDF](https://arxiv.org/pdf/2605.03438v1)

**作者:** Zihao Guo `[一作]` (Xi'an Jiaotong University), Ajmal Saeed Mian `[通讯]` (University of Western Australia)

**通讯引用:** 20888 | [OpenAlex ID](https://openalex.org/A5089986388)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Mantis，一种针对Mamba基础模型的参数高效微调框架

**💡 创新点**

通过状态感知适配器（SAA）和双序列一致性蒸馏（DSCD）解决Transformer PEFT与Mamba的粒度不匹配与序列序列敏感问题

**🔧 技术方法**

使用Mamba（SSM）网络、SAA、DSCD、空间填充曲线双序列序列化、对比损失以及低秩控制信号等技术

**📊 数据集**

在ScanObjectNN、ModelNet40、ShapeNetPart等3D点云数据集上评估

**📈 对比分析**

与全微调、线性探测以及多种现有PEFT方法对比，Mantis仅5%可训练参数即可获得或超过全微调性能，尤其在PB_T50_RS和细粒度分割上提升0.5%–1.5%

**⚠️ 局限性**

依赖手工设计的序列化与超参数，且仅支持单模态点云，跨模态或自动配置仍未解决

---

## 330. Multi-Agent Strategic Games with LLMs

**arXiv ID:** 2605.03604 | [PDF](https://arxiv.org/pdf/2605.03604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 331. Cyclic codes over the ring Z2[u,v](u2(1+u),v2(1+v2))

**arXiv ID:** 2605.03031 | [PDF](https://arxiv.org/pdf/2605.03031v1)

**作者:** Cristina Flaut `[一作]`, Bianca Liana Bercea-Straton `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

在本文中，作者研究了环ℤ₂[u,v]/(u²(1‑u),v²(1‑v²))（其中u³=u²、v⁴=v²、uv=vu）上的线性与循环码，先分别分析了子环ℤ₂[u]/(u²(1‑u))与ℤ₂[v]/(v²(1‑v²))上的循环码，并通过Gray映射将其映射为二进制码；随后利用这两个子环的结构将原环上的循环码拆解为张量积形式，给出了生成多项式与子码之间的关系，并提供了长度3、4等小长度码的具体构造示例。

**💡 创新点**

创新点在于：①首次对上述复合环进行结构分解并给出其非主理想、非局部、非链环的性质；②在此基础上提出了在该环上循环码的完全生成描述（含单生成、多生成与混合生成形式）；③通过Gray映射将环码映射为多维二进制码，保留重量与距离，从而便于与经典二进制循环码比较。

**🔧 技术方法**

使用的技术主要包括：环理论（同余与张量积分解）、多项式环与理想的结构分析、Gray映射构造、以及二进制循环码的生成多项式理论。

**📊 数据集**

本文为理论性研究，没有使用实际数据集。

**📈 对比分析**

未进行实验对比或性能评估；所给出的“比较”仅为理论上的等价性说明（如Gray映射保持Lee重量等），并未涉及码的最小距离、纠错能力或实现复杂度的数值比较。

**⚠️ 局限性**

局限性包括：①缺乏对双循环码或其他多生成形式的实际编码/译码算法细节；②未探讨码的对偶结构、纠错性能与密钥生成的实际应用；③本文仅处理长度有限的小例子，未给出通用算法或软件实现，导致难以直接验证大规模代码的性质。

---

## 332. Geometry over Density: Few-Shot Cross-Domain OOD Detection

**arXiv ID:** 2605.03410 | [PDF](https://arxiv.org/pdf/2605.03410v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 333. Smart Passive Acoustic Monitoring: Embedding a Classifier on AudioMoth Microcontroller

**arXiv ID:** 2605.03412 | [PDF](https://arxiv.org/pdf/2605.03412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 334. CreativityBench: Evaluating Agent Creative Reasoning via Affordance-Based Tool Repurposing

**arXiv ID:** 2605.02910 | [PDF](https://arxiv.org/pdf/2605.02910v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 335. Towards Definitional Interpreters for Hoare Logics

**arXiv ID:** 2605.02963 | [PDF](https://arxiv.org/pdf/2605.02963v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 336. Contrastive Privacy: A Semantic Approach to Measuring Privacy of AI-based Sanitization

**arXiv ID:** 2605.02977 | [PDF](https://arxiv.org/pdf/2605.02977v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 337. Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges

**arXiv ID:** 2605.02973 | [PDF](https://arxiv.org/pdf/2605.02973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 338. Inferring Phylogenetic Networks from Allowed and Forbidden LCA-Constraints

**arXiv ID:** 2605.03827 | [PDF](https://arxiv.org/pdf/2605.03827v1)

**作者:** Patricia A. Ebert `[一作]` (Stockholm University), Marc Hellmuth `[通讯]` (Stockholm University)

**通讯引用:** 960 | [OpenAlex ID](https://openalex.org/A5019719296)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究了在允许与禁止的 LCA 约束下，如何构造满足这些约束的系统发育网络，并给出了三种不同的禁止约束定义；针对每种定义提供了完全的可判定性与构造算法。

**💡 创新点**

提出了对允许/禁止约束对 (R,F) 的全新可判定性条件 (Y1、Y2) 与对应的闭包运算 (R)，并证明了在此基础上可以在多项式时间内构造满足约束的网络；此外，首次系统性比较了三种禁止约束的严格程度与可行性。

**🔧 技术方法**

使用了偏序理论（闭包、等价类、Hasse 图）构造“规范 DAG”；引入四条推理规则（R1–R4）来计算闭包 (R)；利用 xy‑扩展实现禁止约束；并用图论中的 LCA、DAG、网络构造与可扩展性分析技术。

**📊 数据集**

无实验数据集；本文为理论算法研究，所给算法仅在输入为符号约束对 (R,F) 的情况下进行分析。

**📈 对比分析**

通过理论证明，算法在输入规模 |X| 上的时间复杂度为多项式（例如 O(|X|^4) 的闭包计算 + O(|X|^4) 的图构造）；与以往仅处理允许约束的算法相比，新增了对禁止约束的判定，仍保持多项式时间。

**⚠️ 局限性**

所构造的网络不属于任何受限网络类，可能结构复杂；算法仅对三种禁止约束定义给出判定，对更具生物学意义的网络类（如级别‑1 网络、正常网络等）的可实现性尚未解决。

---

## 339. SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Generation

**arXiv ID:** 2605.03534 | [PDF](https://arxiv.org/pdf/2605.03534v1)

**作者:** Jingxi Qiu `[一作]` (ZenWeave AI), Cheng Huang `[通讯]` (ZenWeave AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SURE-RAG，一种针对检索增强生成的证据充分性三分类验证器，能够在证据支持时给出答案，反之则拒绝回答。

**💡 创新点**

创新点在于将证据充分性视作集合级属性，设计可解释的聚合协议，将局部（claim, passage）关系分布聚合为覆盖率、关系强度、分歧、冲突及检索不确定性等五个信号，并在此基础上实现可审计的选择性答复。

**🔧 技术方法**

技术实现上采用 DeBERTa‑v3‑base 交叉编码器预测每个（claim, passage）对的支持/驳斥/中立分布，随后使用轻量级聚合器计算解释性特征，并通过后验校准与置信度衰减得到最终答复决策。

**📊 数据集**

实验数据集主要包括受控的 HotpotQA‑RAG v3（多跳、对抗式证据变体）、SciFact（诊断实验）以及 HaluBench（自然幻觉检测迁移评估）。

**📈 对比分析**

与多种基线（max/mean/Top‑k 池化、concat cross‑encoder、shortcut 基线以及 GPT‑4o 判别器）比较，SURE‑RAG 在 HotpotQA‑RAG v3 上校准后 Macro‑F1 达到 0.9075，显著优于池化基线且与 concat baseline 相当，同时保持可审计；在选择性回答任务中低覆盖率风险下降 37%；在 HaluBench 上性能明显逊色，体现任务边界。

**⚠️ 局限性**

局限性包括：仅在受控数据集上验证，快捷路风险仍存在；未针对长答案多声明覆盖进行充分验证；未实现跨段推理，仅聚合局部信号；对自然幻觉检测的能力有限；低覆盖率下的风险控制仍不完善；实验依赖 GPT‑4o 进行语义审计，缺乏大规模人工标注。

---

## 340. ScrapMem: A Bio-inspired Framework for On-device Personalized Agent Memory via Optical Forgetting

**arXiv ID:** 2605.03804 | [PDF](https://arxiv.org/pdf/2605.03804v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 341. Rose-SQL: Role-State Evolution Guided Structured Reasoning for Multi-Turn Text-to-SQL

**arXiv ID:** 2605.03720 | [PDF](https://arxiv.org/pdf/2605.03720v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 342. PRISM-CTG: A Foundation Model for Cardiotocography Analysis with Multi-View SSL

**arXiv ID:** 2605.02917 | [PDF](https://arxiv.org/pdf/2605.02917v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 343. How Language Models Process Negation

**arXiv ID:** 2605.03052 | [PDF](https://arxiv.org/pdf/2605.03052v1)

**作者:** Zhejian Zhou `[一作]` (Information Sciences Institute), Jonathan May `[通讯]` (Information Sciences Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过机制解释方法研究大语言模型（LLM）如何处理否定，系统揭示了内部的构造（constructive）与抑制（suppression）两种机制共存，并发现后者在后期层通过快捷注意力（shortcut attention）导致负向推理错误。

**💡 创新点**

创新点包括：①提出 Attention Sink Ablation 与 Contrastive Attribution 两种新型机制解释工具；②从实证上证明否定处理是由构造机制主导并辅以抑制机制完成；③发现并剔除导致错误的快捷注意力头，显著提升负向准确率。

**🔧 技术方法**

采用的技术包括：LogitLens、Sparse AutoEncoder (SAE)、PCA+LDA 分析注意力向量、Path Patching、Attention Sink、对比归因 (contrastive attribution) 等机制解释手段。

**📊 数据集**

使用了作者自制的 648 条否定问答对数据集（162 个问题，4 种提示模板），每条记录包含正向与否定两种提示及对应的正确答案。

**📈 对比分析**

与原始模型、LogitLens 和 Attention Sink 等方案对比，负向准确率从约 50% 提升至 66–68%（绝对提升 15–20%），正向准确率基本不变，显示方法对模型内部错误机制具有显著纠正效果。

**⚠️ 局限性**

局限性在于：仅针对单词级否定（例如“not gas”），未探讨复杂句法或多词否定；方法对不同模型的泛化能力尚未充分验证；需要手工设计阈值和层次，使用成本较高。

---

## 344. SAM-NER: Semantic Archetype Mediation for Zero-Shot Named Entity Recognition

**arXiv ID:** 2605.03706 | [PDF](https://arxiv.org/pdf/2605.03706v1)

**作者:** Ruichu Cai `[一作]` (Guangdong University of Technology), Boyan Xu `[通讯]` (Guangdong University of Technology)

**通讯引用:** 1436 | [OpenAlex ID](https://openalex.org/A5034536387)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SAM-NER，一个三阶段的零样本命名实体识别框架。

**💡 创新点**

创新点在于通过语义原型中介（Semantic Archetype Mediation）将实体抽象为跨域不变的原型，缓解标签语义漂移。

**🔧 技术方法**

使用了大语言模型（Llama3-8B、Qwen2.5-7B）进行指令调优的实体检索器、原型分类器，以及冻结LLM进行定义对齐校准；还利用双源协同提取与一致性去噪技术。

**📊 数据集**

训练数据为 Pile-NER 与 IEPile，评估数据为 CrossNER。

**📈 对比分析**

与 InstructUIE、UniNER、IEPile、GoLLIE、KnowCoder、GLiNER、IRRA、GUIDEX 等前沿零样本 NER 方法对比，在 CrossNER 上平均 micro‑F1 达 66.3，显著优于所有基线。

**⚠️ 局限性**

局限性包括：① 原型空间仅覆盖 14 个语义原型，可能在高度专业化领域缺乏细粒度表达；② 对目标标签定义的可辨别性依赖较大，定义不清时校准效果受限。

---

## 345. TACO: Trajectory Aligning Cross-view Optimisation

**arXiv ID:** 2605.03315 | [PDF](https://arxiv.org/pdf/2605.03315v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 346. Multilingual Safety Alignment via Self-Distillation

**arXiv ID:** 2605.02971 | [PDF](https://arxiv.org/pdf/2605.02971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 347. Towards an End-to-End System for 3D Tracking of Physical Objects in Virtual Immersive Environments

**arXiv ID:** 2605.02901 | [PDF](https://arxiv.org/pdf/2605.02901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 348. Proteo-R1: Reasoning Foundation Models for De Novo Protein Design

**arXiv ID:** 2605.02937 | [PDF](https://arxiv.org/pdf/2605.02937v1)

**作者:** Fang Wu `[一作]` (Stanford University), Yejin Choi `[通讯]` (Stanford University)

**通讯引用:** 26439 | [OpenAlex ID](https://openalex.org/A5102992157)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Proteo‑R1，一个将多模态 LLM 作为预先推理专家与 AF3‑style 扩散生成专家分离的双专家框架，用于抗体 CDR 的协同设计。

**💡 创新点**

创新点在于将预生成的 residue‑level 关键位点识别转化为稀疏硬约束，既保持几何生成器的稳定性与可解释性，又实现了可控的设计流程。

**🔧 技术方法**

技术上结合了多模态预训练 LLM、AF3‑style 扩散模型、三阶段课程学习、结构化推理监督以及 CoT 与 JSON 交互。

**📊 数据集**

使用公开蛋白结构数据库 PDB、SAbDab 等进行多模态对齐、结构推理和抗体–抗原 CDR 重新设计的数据集。

**📈 对比分析**

通过与 DiffAb、dyMEAN、AbX 等基线在多 CDR 重新设计和单 H3 重新设计任务中的 RMSD、IMP、JSD_bb 等几何与结合能指标对比，Proteo‑R1 在绝大多数指标上表现优于或与最强基线相当，并在多模型兼容性实验中进一步提升性能。

**⚠️ 局限性**

主要局限在于关键位点识别的准确性受限于推理专家的推断能力，导致与 Oracle Anchor 的性能差距；此外，三阶段课程学习和大规模模型训练对计算资源和工程复杂性提出了较高要求。

---

## 349. Dynamic Distillation and Gradient Consistency for Robust Long-Tailed Incremental Learning

**arXiv ID:** 2605.03364 | [PDF](https://arxiv.org/pdf/2605.03364v1)

**作者:** Taigo Sakai `[一作]` (Meijo University), Kazuhiro Hotta `[通讯]` (Meijo University)

**通讯引用:** 2190 | [OpenAlex ID](https://openalex.org/A5103163418)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了梯度一致性正则化与基于熵的动态蒸馏技术，用于解决长尾类别增量学习中的灾难性遗忘与少数类欠学习问题

**💡 创新点**

创新点在于将梯度移动平均约束与归一化熵相结合，实现了任务间梯度平滑与知识蒸馏权重的自适应调节

**🔧 技术方法**

使用梯度重加权（GR）、梯度一致性正则化（GCR）、熵感知动态蒸馏、ResNet 结构及SGD优化

**📊 数据集**

在CIFAR-100-LT、ImageNetSubset-LT与Food101-LT三大长尾数据集上进行实验

**📈 对比分析**

与iCaRL、PODNet、GR等基线对比，实验表明在大多数设置下提升了3–5%的平均准确率，尤其在“In-ordered”顺序下效果显著

**⚠️ 局限性**

仍存在对超参数（如λ_GCR、β）敏感、仅在特定任务划分与数据集上验证，且对更大规模或多模态数据集的鲁棒性尚待进一步探究

---

## 350. Contrastive Regularization for Accent-Robust ASR

**arXiv ID:** 2605.03297 | [PDF](https://arxiv.org/pdf/2605.03297v1)

**作者:** Van-Phat Thai `[一作]` (Air Traffic Management Research Institute), Sameer Alam `[通讯]` (Air Traffic Management Research Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究将监督对比学习（SupCon）作为CTC微调的轻量级正则化，在不改变模型结构的前提下提升多口音ASR性能。

**💡 创新点**

创新点在于利用相同转录的utterance级正样本进行对比学习，无需额外口音标签或架构改动，显著提升未见口音场景的鲁棒性。

**🔧 技术方法**

使用的技术包括wav2vec 2.0/WavLM自监督声学编码器、CTC主损失、带温度τ=0.1的监督对比损失以及λ=0.1的权重调度。

**📊 数据集**

实验数据集为L2-ARCTIC多口音英语数据集，评估未见转录（UT）和未见口音（UA）两种设置。

**📈 对比分析**

与CTC基线、Whisper FT、MAS-LoRA-QKVO等方法比较，SupCon在UA下将WER降至7.41%（相对提升25.8%），在UT下降至9.14%（相对提升12.7%）。

**⚠️ 局限性**

局限性在于需要转录重复样本以构造正样本，转录稀缺时效果受限；对WavLM提升有限，且对低资源或跨语言场景的泛化尚未验证。

---

## 351. Heterogeneous Graph Importance Scoring and Clustering with Automated LLM-based Interpretation

**arXiv ID:** 2605.02919 | [PDF](https://arxiv.org/pdf/2605.02919v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 352. Jiao: Bridging Isolation and Customization in Mixed Criticality Robotics

**arXiv ID:** 2605.03641 | [PDF](https://arxiv.org/pdf/2605.03641v1)

**作者:** James Yen `[一作]` (Shanghai Jiao Tong University), Zhengwei Qi `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2513 | [OpenAlex ID](https://openalex.org/A5011323970)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在工业机器人中实现硬实时控制、软实时感知和用户应用共用多核平台时，提出Jiao体系架构，包括安全 IO 单元、参数同步服务和 IEC 61508 对齐的安全通信层，解决了专业知识不对称和跨域定制安全性问题。

**💡 创新点**

通过把隔离边界视为安全接口，提供可定制的 ROS 风格 API、硬件级覆盖能力和黑通道完整性校验，实现了跨域安全交互与实时隔离的统一设计。

**🔧 技术方法**

采用 Jailhouse 静态分区 hypervisor、ivshmem 共享内存、CRC‑32C、序列号、时间戳的安全通信层，以及 GPIO 级别的硬件覆写；整合 ROS 2、EtherCAT、CAN 等通信协议。

**📊 数据集**

在 D9340 SoC 上使用 ARM Cortex‑A55 四核 + R‑core，采集 1 kHz EtherCAT 操作的实时控制周期与 10 分钟的感知负载（来自 OpenMind 的工作负载）进行实验。

**📈 对比分析**

将基线（无隔离）与隔离配置进行对比，周期波动标准差从 12.58 μs 降至 1.95 μs（84.5% 降低），p99 jitter 从 69.0 μs 降至 7.8 μs，最大偏差从 321.5 μs 降至 32.8 μs，完全消除 |jitter|>50 μs 的大偏差。

**⚠️ 局限性**

评估仅覆盖了单一 1 kHz 任务的时序隔离，未对黑通道完整性做系统性故障注入，且在其他多感知或视觉密集工作负载下的性能仍待验证。

---

## 353. Coordination as an Architectural Layer for LLM-Based Multi-Agent Systems

**arXiv ID:** 2605.03310 | [PDF](https://arxiv.org/pdf/2605.03310v1)

**作者:** Maksym Nechepurenko `[一作]` (Devnull FZCO), Pavel Shuvalov `[通讯]` (Devnull FZCO)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多代理LLM系统的协调架构对失败模式的影响，提出协调层抽象，预设Murphy分解签名并在Polymarket预测市场上实验验证。

**💡 创新点**

将协调视为可配置的架构层，提供可独立于信息访问和代理实现的规范，预先可判定失败模式签名，并提出信息控制实验设计。

**🔧 技术方法**

使用Claude‑Opus‑4‑6单一LLM、固定工具堆栈与提示模板，采用Murphy分解、Brier与Alpha分解、Bootstrap与功效分析等技术。

**📊 数据集**

使用100个Polymarket二元预测市场数据集，满足后训练截止后30天、基准价平衡、类别分层等筛选条件。

**📈 对比分析**

在相同信息、相同模型下对比五种协调配置，利用Murphy签名、Brier/Alpha、成本–质量Pareto等指标评估，发现最优配置在成本与精度上占优，部分预设签名得到支持。

**⚠️ 局限性**

样本量不足以在Bonferroni校正下区分相邻配置；实验仅在单一模型/工具/信息环境下，未评估跨模型、跨领域或实时搜索信息的影响。

---

## 354. Orientation-Aware Unsupervised Domain Adaptation for Brain Tumor Classification Across Multi-Modal MRI

**arXiv ID:** 2605.03490 | [PDF](https://arxiv.org/pdf/2605.03490v1)

**作者:** Sapna Sachan `[一作]` (Indian Institute of Technology), Prashant Wagambar Patil `[通讯]` (Indian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一个两阶段的方向感知无监督域自适应框架，用于跨多模态MRI进行多类别脑瘤分类；先用DilatedCNN将混合方向的切片分离为轴向、矢状面和冠状面，再对每个方向训练单独的ResNet50分类器，并通过伪标签与MMD对齐源域与目标域特征；

**💡 创新点**

创新点包括：①引入方向感知的切片分离以降低域偏移；②在每个方向上使用伪标签指导的类级MMD对齐，兼顾类语义与域一致性；③将多模态源域（T1/T2/FLAIR）知识迁移到单模态目标域（增强T1）；④结合膨胀卷积提升局部与全局特征的融合；

**🔧 技术方法**

采用的技术包括DilatedCNN（膨胀卷积）、ResNet50骨干网络、伪标签生成、最大均值差异（MMD）损失、数据增强（旋转、阈值化）、交叉熵分类、Adam优化器、固定种子复现；

**📊 数据集**

使用的数据集为：源域 Bangladesh Brain Cancer MRI Dataset（T1/T2/FLAIR，共6056张2D图像，三类脑瘤）；目标域 Figshare Brain Tumor Dataset（3044张增强T1 2D切片，三类脑瘤）；两者均包含混合解剖方向；

**📈 对比分析**

在无域自适应情况下，目标域Macro F1仅约38%；与现有多种UDA方法（SHOT、DDC、DJSA、JAN、CDAN、DANN、MCD、DSBN）比较，所有方法均采用相同ResNet50骨干；本框架在目标域实现Macro F1 72.95%，显著优于最佳对手约50-57%；若不使用切片分离，目标Macro F1仅为51.96%；切片分离准确率达97%/97.4%；

**⚠️ 局限性**

局限性：轴向切片分类性能相对较弱，导致整体性能下降；类不平衡仍影响模型稳定性；伪标签可能引入噪声；仅使用2D切片，未充分利用3D空间信息；在更大、多机构数据集上的验证仍待开展。

---

## 355. cotomi Act: Learning to Automate Work by Watching You

**arXiv ID:** 2605.03231 | [PDF](https://arxiv.org/pdf/2605.03231v1)

**作者:** Masafumi Oyamada `[一作]` (NEC Corporation), Takuya Tamura `[通讯]` (NEC Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款基于浏览器的计算机使用代理，结合强推理与从用户行为学习的组织知识。

**💡 创新点**

将行为到知识管道与可编辑共享工作区结合，使代理能够持续学习组织隐性知识并通过双向编辑实现人机协作。

**🔧 技术方法**

采用 ReAct 风格的 agent scaffold（自适应懒观察、语义差分压缩、粗粒度动作、任务分解、best‑of‑N 选择），使用 LLM 对行为日志进行抽象与 ETL，并通过共享工作区与 RAG 交互实现知识检索。

**📊 数据集**

在 WebArena（812 任务，其中 179 任务子集）评估推理能力；在 WorkArena‑L1 评估行为知识；使用公开 WebArena 自动评分器与人工验证数据。

**📈 对比分析**

与 SteP、OpAgent、CUGA 等基线在相同 179 任务子集比较，取得 80.4% 成功率，超过人类 78.2%；在行为知识实验中，知识覆盖率提升可使任务成功率提升至 +10%。

**⚠️ 局限性**

依赖用户主动授权与编辑，行为日志噪声与偏差仍需实地验证；原始轨迹高效但可扩展性差，抽象粒度平衡需要进一步研究。

---

## 356. New bounds on the covering radius of orthogonal arrays of even strength

**arXiv ID:** 2605.03589 | [PDF](https://arxiv.org/pdf/2605.03589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 357. DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data

**arXiv ID:** 2605.03430 | [PDF](https://arxiv.org/pdf/2605.03430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 358. BFORE: Butterfly-Firefly Optimized Retinex Enhancement for Low-Light Image Quality Improvement

**arXiv ID:** 2605.03509 | [PDF](https://arxiv.org/pdf/2605.03509v1)

**作者:** Ahmed Cherif `[一作]` `[通讯]` (Sofrecom), Ahmed Cherif (Sofrecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种BFORE框架，利用混合的蝴蝶优化算法（BOA）与萤火虫算法（FA）自动调节多阶段Retinex增强管线（AGCWD、ABM3D、MSRCR）的全部参数，实现无训练数据的低照度图像增强；

**💡 创新点**

创新点在于：①将BOA与FA组合为双阶段全局探索-局部精炼的混合优化策略；②为多阶段管线（亮度校正、降噪、颜色恢复）提供统一的自动参数搜索；③引入收敛阈值切换机制，使优化过程更稳定、高效；

**🔧 技术方法**

使用了Retinex理论、Adaptive Gamma Correction with Weighted Distribution (AGCWD)、Adaptive BM3D denoising、Multi-Scale Retinex with Color Restoration (MSRCR)、蝴蝶优化算法、萤火虫算法及其混合搜索策略；

**📊 数据集**

主要使用LOL（Low‑Light）基准数据集，15张测试低照度–正常照度配对图像；

**📈 对比分析**

与传统方法（HE、CLAHE、MSR、MSRCR、AGCWD）以及深度学习基线（RetinexNet、KinD、EnlightenGAN、Zero‑DCE、MIRNet）进行对比；BFORE在LOL上实现PSNR 17.22 dB，最高于传统方法，并且在PSNR与SSIM上优于RetinexNet；

**⚠️ 局限性**

主要局限包括：优化过程耗时较长（≈170 s/图像），对实时应用不友好；SSIM略低于默认MSRCR，牺牲部分结构相似性；评估仅基于单一数据集，缺乏对不同领域的验证。

---

## 359. Self-Improvement for Fast, High-Quality Plan Generation

**arXiv ID:** 2605.03625 | [PDF](https://arxiv.org/pdf/2605.03625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 360. Enhancing Performance Insight at Scale: A Heterogeneous Framework for Exascale Diagnostics

**arXiv ID:** 2605.03561 | [PDF](https://arxiv.org/pdf/2605.03561v1)

**作者:** Dragana Grbic `[一作]` (Rice University), Dragana Grbic `[通讯]` (Rice University)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5134650876)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并加速了一个异构性能分析框架，能够在秒级完成100,000个MPI rank的性能数据摄取、GPU并行分析，并将诊断结果映射到物理网络拓扑上。

**💡 创新点**

创新点包括：①用高性能API和GPU并行实现超大规模数据摄取；②集成三维（节点·追踪·迭代）模型重构迭代行为；③拓扑感知的工作流将逻辑性能异常映射到Slingshot互连物理坐标。

**🔧 技术方法**

技术涵盖：HPCToolkit测量、C/C++共享库+OpenMP、内存映射文件、GPU后端（cuDF、hipDF、SYCL/oneAPI）、Pandas/PyDataFrame、Hatchet/Thicket、DBSCAN/K-Means聚类、POSIX ID映射。

**📊 数据集**

使用了Frontier和Aurora超级计算机的生产工作负载：Frontier上GAMESS（Hartree-Fock）和AMG基准；Aurora上100,000 rank的AMG执行，产生约11GB的性能数据库。

**📈 对比分析**

与原始Python实现对比：在Aurora上摄取10万rank耗时9.69秒，Python版需400+秒；GPU后端对100k追踪的分析比CPU快314×；在Frontier上预测通过负载均衡可提升32.28%。

**⚠️ 局限性**

局限性：对高度规律的迭代工作负载最有效；对非规则或高度变化的工作不适用；Intel GPU支持仅限基本单列操作；GPU后端需显存；仍需完善多列复杂运算。

---

## 361. Mix3R: Mixing Feed-forward Reconstruction and Generative 3D Priors for Joint Multi-view Aligned 3D Reconstruction and Pose Estimation

**arXiv ID:** 2605.03359 | [PDF](https://arxiv.org/pdf/2605.03359v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 362. Information Theory and Statistical Learning

**arXiv ID:** 2605.02989 | [PDF](https://arxiv.org/pdf/2605.02989v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 363. deSEO: Physics-Aware Dataset Creation for High-Resolution Satellite Image Shadow Removal

**arXiv ID:** 2605.03610 | [PDF](https://arxiv.org/pdf/2605.03610v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 364. Real-Time Evaluation of Autonomous Systems under Adversarial Attacks

**arXiv ID:** 2605.03491 | [PDF](https://arxiv.org/pdf/2605.03491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 365. OCRR: A Benchmark for Online Correction Recovery under Distribution Shift

**arXiv ID:** 2605.03153 | [PDF](https://arxiv.org/pdf/2605.03153v1)

**作者:** Adrian Grassi `[一作]` `[通讯]` (Independent Researcher), Adrian Grassi (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的基准OCRR，用于评估模型在分布漂移下通过在线纠正快速恢复的能力。

**💡 创新点**

创新点在于：①设计了流式纠正恢复率评估协议，既衡量新类别学习又检测遗忘；②引入哈希链增量账本加“margin‑band majority”投票的无参 substrate，成为唯一同时保持高新类与原分布准确率的系统；③展示了在不同存储预算下的Pareto前沿，揭示检索式学习对样本效率的优势。

**🔧 技术方法**

采用的技术包括：在线检索（kNN-LM）、连续学习方法（EWC、A‑GEM、LwF）、在线线性模型、LoRA参数高效微调、以及自定义的增量账本与投票机制；同时使用近似最近邻（HNSW）验证规模可扩展性。

**📊 数据集**

使用的公开数据集为 Banking77（77类）和 CLINC150（151类），并在两者上进行 held‑out 类别的流式测试。

**📈 对比分析**

与九个基线（包括连续学习、检索增量、在线线性、LoRA‑DeBERTa 等）及其存储限制版本比较，substrate 在所有设置下都占据Pareto最优：在 10% 或 70% novel‑accuracy 阈值下，既实现 80‑90% novel accuracy 又保持 95‑99% 原分布准确率；相对最佳参数化基线提升 30‑80% 点，且每次纠正成本仅为几微秒。

**⚠️ 局限性**

局限性包括：仅评估单一语言的类别漂移；不考虑同类漂移、开放词汇、跨模态场景；纠正假设为完美标签；存储规模验证仅在合成数据上完成；所有检索系统固定使用相同编码器。

---

## 366. Tailored Prompts, Targeted Protection: Vulnerability-Specific LLM Analysis for Smart Contracts

**arXiv ID:** 2605.03697 | [PDF](https://arxiv.org/pdf/2605.03697v1)

**作者:** Xing Zhang `[一作]` (NetX Foundation), Anbang Ruan `[通讯]` (NetX Foundation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大语言模型的智能合约漏洞检测框架，利用AST抽取精准上下文并为13类常见漏洞设计定制提示；

**💡 创新点**

通过构造大规模真实审计数据集，精确AST上下文过滤，以及在无专门微调情况下使用链式思维（CoT）提示提升LLM推理；

**🔧 技术方法**

采用大语言模型（如GPT‑4）、AST解析（python‑solidity‑parser）、少量样例提示（ICL）与链式思维提示；

**📊 数据集**

构建并公开的31165条手工标注漏洞实例数据集，涵盖15大区块链平台、3,200+真实项目；

**📈 对比分析**

在真实审计数据上评估，13类漏洞平均正召回率0.92、负召回率0.85；相较于传统规则、静态/动态分析方法，性能显著提升，误报率显著降低；

**⚠️ 局限性**

仍需人工精细提示调优，难以自动化生成；对未知或新兴漏洞模式的泛化能力有限；对LLM模型的依赖导致成本与可扩展性受限。

---

## 367. A Study of Consumers Cognitive Load in eCommerce Websites using Eye-tracking Technology

**arXiv ID:** 2605.02899 | [PDF](https://arxiv.org/pdf/2605.02899v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 368. Agentic AI-Based Joint Computing and Networking via Mixture of Experts and Large Language Models

**arXiv ID:** 2605.02911 | [PDF](https://arxiv.org/pdf/2605.02911v1)

**作者:** Robert-Jeron Reifert `[一作]` (Ruhr University Bochum), Aydin Sezgin `[通讯]` (Ruhr University Bochum)

**通讯引用:** 3347 | [OpenAlex ID](https://openalex.org/A5034269994)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于大型语言模型门控的 Mixture of Experts 框架，用于联合通信与计算网络的自适应资源优化。

**💡 创新点**

创新点在于将 LLM 作为语义门控，动态选择并加权专门优化专家，解决多目标、鲁棒性和异构资源耦合问题。

**🔧 技术方法**

使用了 LLM（如 ChatGPT）门控、Mixture of Experts 架构、深度学习优化专家以及不确定性注入机制。

**📊 数据集**

通过模拟实验使用 Rayleigh 信道、Gamma 分布计算需求以及人工生成的网络状态数据集。

**📈 对比分析**

与单一专家及联合专家基准对比，结果显示 Agentic MoE 在多目标任务中实现近最优性能，并显著降低计算复杂度。

**⚠️ 局限性**

主要局限在于 LLM 对查询语义的准确性、门控选择误差导致的性能下降，以及在大规模专家库时推理延迟可能增加。

---

## 369. MILE: Mixture of Incremental LoRA Experts for Continual Semantic Segmentation across Domains and Modalities

**arXiv ID:** 2605.03555 | [PDF](https://arxiv.org/pdf/2605.03555v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 370. First Shape, Then Meaning: Efficient Geometry and Semantics Learning for Indoor Reconstruction

**arXiv ID:** 2605.03463 | [PDF](https://arxiv.org/pdf/2605.03463v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 371. MemFlow: Intent-Driven Memory Orchestration for Small Language Model Agents

**arXiv ID:** 2605.03312 | [PDF](https://arxiv.org/pdf/2605.03312v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 372. Internet of Things Security: A Survey on Common Attacks

**arXiv ID:** 2605.03744 | [PDF](https://arxiv.org/pdf/2605.03744v1)

**作者:** Dalton Cézane Gomes Valadares `[一作]` (Universidade Federal da Paraíba), Kyller Costa Gorgônio `[通讯]` (Universidade Federal de Campina Grande)

**通讯引用:** 430 | [OpenAlex ID](https://openalex.org/A5031124772)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述并分类了28种常见的物联网攻击，使用STRIDE模型进行功能性威胁分类，并用CVSS评估每种攻击的严重性；同时将攻击映射到五类基础漏洞（Process、Code、Communication、Operation、Device），并梳理了针对每种攻击的现有缓解技术和未来研究方向。

**💡 创新点**

①将功能性威胁分类与漏洞根因分类两者结合，形成多维度威胁-漏洞映射框架；②首次在一篇综述中同时使用STRIDE和CVSS进行定性与定量评估；③提出以Named Data Networking (NDN)为核心的未来安全范式，并指出Zero Trust与AI辅助检测在IoT中的潜在作用。

**🔧 技术方法**

STRIDE威胁建模、CVSS 4.0评分、五类漏洞分类、技术对策如轻量级加密、机器学习异常检测、基于NDN的数据中心安全方案、零信任模型。

**📊 数据集**

参考文献数据集来自IEEE Xplore、ACM Digital Library、Springer Link等主流数据库，覆盖近五年内在物联网安全领域发表的期刊、会议和技术报告，重点提取了28种攻击案例与对应的对策。

**📈 对比分析**

由于论文为综述性质，未进行实验或性能对比；评价标准是通过CVSS将攻击严重性分级（Medium、High、Critical），并以STRIDE对攻击的功能影响进行分类，形成一套可视化的威胁-漏洞映射，便于安全工程师快速定位风险与优先级。

**⚠️ 局限性**

①仅涵盖28种常见攻击，未覆盖所有可能的IoT威胁；②依赖已有文献，缺乏新的实证数据和实验验证；③多维度映射框架尚未在实际系统中进行验证；④未提供对比基准或度量标准，缺少对缓解技术效果的量化评估；⑤在快速演进的IoT生态中，新的攻击向量与技术变革可能导致框架的适用性随时间下降。

---

## 373. Real Image Denoising with Knowledge Distillation for High-Performance Mobile NPUs

**arXiv ID:** 2605.03680 | [PDF](https://arxiv.org/pdf/2605.03680v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 374. DALPHIN: Benchmarking Digital Pathology AI Copilots Against Pathologists on an Open Multicentric Dataset

**arXiv ID:** 2605.03544 | [PDF](https://arxiv.org/pdf/2605.03544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 375. Neural Control: Adjoint Learning Through Equilibrium Constraints

**arXiv ID:** 2605.03288 | [PDF](https://arxiv.org/pdf/2605.03288v1)

**作者:** Dezhong Tong `[一作]` (University of Michigan), M. Khalid Jawed `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于等势约束的神经控制框架，利用邻域推导的代理梯度对弹性线性对象（DLO）进行路径和终端目标的精确控制；

**💡 创新点**

创新点在于：①通过对收敛平衡解的隐式微分，构造不需要展开Newton/CG迭代的代理梯度；②结合递推型模型预测控制（RHC）重新锚定优化轨迹，避免多稳态系统的基底漂移；

**🔧 技术方法**

核心技术包括：邻域推导（adjoint sensitivity）对等势方程求解的梯度、冻结切线近似、递推型连续控制（RHC）与代理梯度的融合；

**📊 数据集**

使用的实验数据集：仿真中对弹性条的离散模型（101节点）以及真实6-DoF机械臂抓取并操作同一弹性条的实测数据；

**📈 对比分析**

与基线方法（SPSA、CEM）以及不使用RHC的代理梯度进行对比，实验显示Adjoint+RHC在相同更新次数下比SPSA/CEM快19–127倍，且最佳损失低3–6个数量级；

**⚠️ 局限性**

局限性包括：冻结切线近似导致的高阶误差，近分叉点或非光滑物理时不稳定，递推段长度有限可能导致长程规划性能下降。

---

## 376. Task Vector Geometry Underlies Dual Modes of Task Inference in Transformers

**arXiv ID:** 2605.03780 | [PDF](https://arxiv.org/pdf/2605.03780v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 377. Exploring Pass-Rate Reward in Reinforcement Learning for Code Generation

**arXiv ID:** 2605.02944 | [PDF](https://arxiv.org/pdf/2605.02944v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 378. Can AI Help You Get Over Your Breakup? One Session with a Belief-Reframing Chatbot Shows Sustained Distress Reduction

**arXiv ID:** 2605.03261 | [PDF](https://arxiv.org/pdf/2605.03261v1)

**作者:** Thomas Menzel `[一作]` (Technical University of Munich), Thomas Bohné `[通讯]` (University of Cambridge)

**通讯引用:** 627 | [OpenAlex ID](https://openalex.org/A5074627376)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并测试了一个名为 overit 的单次会话 AI 聊天机器人，用于缓解分手后产生的心理痛苦，并在 7 天内与仅完成评估的对照组进行随机对照试验。

**💡 创新点**

创新点在于将记忆再巩固理论与认知重构方法结合，构建了一个阶段化的、基于情绪冲突的对话流程，并采用双调用架构（生成与评估分离）避免 LLM 同时满足共情与挑战的矛盾需求。

**🔧 技术方法**

使用的技术包括：Claude Sonnet 4‑5 LLM 作为对话引擎，系统提示动态拼装、阶段性指令；并行评估调用实现对话进度检测；Flutter 移动端与 Flask+Google Cloud 后端协同实现实时提示与状态管理；Whisper+Groq API 处理语音输入。

**📊 数据集**

数据集来源于 254 名在美国与英国经历过恋爱关系结束的成年人（通过 Prolific 招募），使用 Breakup Distress Scale (BDS) 与 Experiences in Close Relationships Scale (ECR‑S) 进行基线与后续评估；后续 1 个月问卷亦使用相同 BDS 量表。

**📈 对比分析**

比较方法：线性混合效应模型评估 7 天的时间×条件交互；效应量为标准化差异 d = -0.70（显著）；在 1 个月的探索性分析中，交互仍显著但效应减弱（d = -0.26）。对照组仅完成评估，未提供任何结构化对话；与 overit 的差异表明单次会话即可产生临床意义的短期缓解。

**⚠️ 局限性**

局限性包括：对照条件仅为评估，无法排除自我披露或期待效应；未进行再巩固时间窗口控制，因而无法确立记忆更新机制；仅单次会话，缺乏多次检查点验证长期效果；样本为 Prolific 便利样本，主要为 iPhone 用户，且多为白人女性，外部有效性受限；所有测量均为自评，可能受社会期望偏差影响。

---

## 379. PERFECT: Personalized Federated Learning for CBRS Radar Detection

**arXiv ID:** 2605.03199 | [PDF](https://arxiv.org/pdf/2605.03199v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 380. ChaRVoC: A Challenge-Response Voice Cancelable Authentication System

**arXiv ID:** 2605.02990 | [PDF](https://arxiv.org/pdf/2605.02990v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 381. Optimal Union Probability Interval Is NP-Hard

**arXiv ID:** 2605.03556 | [PDF](https://arxiv.org/pdf/2605.03556v1)

**作者:** Petteri Kaski `[一作]` (Aalto University), Chandra Kanta Mohapatra `[通讯]` (Aalto University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在只给定部分交集概率时，如何计算事件并集概率的最优上下界，进一步阐述了其几何结构并证明了该问题的NP难度。

**💡 创新点**

创新点在于首次证明即使只给定最多两元交集概率，该问题的最优界求解仍为NP‑hard，并通过Venn多面体、关联多面体和并集多面体的几何关系构造了新的证明框架。

**🔧 技术方法**

主要技术包括线性规划（Hailperin 的 LP）、多面体投影与极点描述、极点-顶点变换、分数着色与团数问题的多项式时间归约，以及极点多面体的极性与可行性 oracle 的构造。

**📊 数据集**

本研究为理论工作，没有使用实验数据集。

**📈 对比分析**

由于本文给出的结果是NP难度证明，没有提出可直接实现的算法或实验比较，因此没有性能评估。

**⚠️ 局限性**

限制在于仅提供了最优解求解的NP难度，未给出近似算法或有效特例；此外，证明仅适用于给定交集概率时的情况，实际应用中可能需要进一步的经验性方法。

---

## 382. FUS3DMaps: Scalable and Accurate Open-Vocabulary Semantic Mapping by 3D Fusion of Voxel- and Instance-Level Layers

**arXiv ID:** 2605.03669 | [PDF](https://arxiv.org/pdf/2605.03669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 383. ARISE: A Repository-level Graph Representation and Toolset for Agentic Fault Localization and Program Repair

**arXiv ID:** 2605.03117 | [PDF](https://arxiv.org/pdf/2605.03117v1)

**作者:** Shahd Seddik `[一作]` (University of British Columbia), Fatemeh Fard `[通讯]` (University of British Columbia)

**通讯引用:** 448 | [OpenAlex ID](https://openalex.org/A5029327446)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 ARISE 系统，为 LLM 代理提供多粒度程序图和可查询的数据流切片工具，解决仓库级故障定位与自动修复问题。

**💡 创新点**

创新点在于将 intra‑procedural 定义‑使用关系嵌入图中，并将数据流切片作为首层可调用工具，使得模型能够一次性获得变量传播路径，显著提升函数/行级定位精度；同时提供框架无关的三层工具 API。

**🔧 技术方法**

技术实现包括：Python AST 解析构建包含目录、模块、类、函数、语句等多级节点的程序图；基于 def‑use 边的 BFS 切片；在 SWE‑agent 框架上集成工具调用；使用 Qwen2.5‑Coder‑32B‑Instruct 进行推理。

**📊 数据集**

评估使用 SWE‑bench Lite，包含 300 条真实 GitHub issue，覆盖 11 个 Python 开源仓库。

**📈 对比分析**

与 BM25 静态检索、SWE‑agent 基线以及 RepoGraph/LocAgent 等结构图方案对比，ARISE 在函数 Recall@1 提升 17 点、行 Recall@1 提升 15 点，Pass@1 达到 22%（比基线高 4.7%）。

**⚠️ 局限性**

局限在于数据流切片仅覆盖同一函数内部，无法跨调用边缘追踪变量；对属性访问、动态 dispatch 等复杂模式的分析缺失，导致部分错误定位失败。

---

## 384. The Detector Teaches Itself: Lightweight Self-Supervised Adaptation for Open-Vocabulary Object Detection

**arXiv ID:** 2605.03642 | [PDF](https://arxiv.org/pdf/2605.03642v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 385. Before Forgetting, Learn to Remember: Revisiting Foundational Learning Failures in LVLM Unlearning Benchmarks

**arXiv ID:** 2605.03759 | [PDF](https://arxiv.org/pdf/2605.03759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 386. A formulation of D-institution using functor categories

**arXiv ID:** 2605.03597 | [PDF](https://arxiv.org/pdf/2605.03597v1)

**作者:** Go Hashimoto `[一作]` `[通讯]`, Go Hashimoto

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种直接在机构理论中引入变量结构的新方法，使用泛函范畴的一般化来构造谓词逻辑范畴，并给出复合句子的函子与证明系统，证明其完备性。

**💡 创新点**

核心创新是将变量结构直接编码为范畴中的扩展，避免了传统机构理论中对变量的“隐式扩展”与多余条件，形成了可构造的 -机构范畴与机构同态的自然定义。

**🔧 技术方法**

采用了范畴论工具：泛函范畴、Grothendieck 构造、余积与推导等，以及机构的自然变换与模型扩张的推理，构建了可计算的证明规则。

**📊 数据集**

本工作不依赖具体数据集，而是纯粹的理论性贡献，未进行实验验证。

**📈 对比分析**

比较方法通过构造 -机构与传统机构的同构与变换来展示优越性，但未给出量化性能指标，主要以理论可证性与结构简洁性作为评价。

**⚠️ 局限性**

局限性在于理论复杂度高、实现细节未给出、缺乏针对实际逻辑系统的具体实例与实验评估。

---

## 387. Programmatic Context Augmentation for LLM-based Symbolic Regression

**arXiv ID:** 2605.03101 | [PDF](https://arxiv.org/pdf/2605.03101v1)

**作者:** Hao Liu `[一作]` (California Institute of Technology), Yisong Yue `[通讯]` (California Institute of Technology)

**通讯引用:** 6595 | [OpenAlex ID](https://openalex.org/A5085826758)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于LLM的进化搜索框架，称为Programmatic context augmentation（PCA），让LLM既生成候选符号回归方程，又生成并执行数据分析代码，以获取更丰富的统计上下文来指导搜索。

**💡 创新点**

核心创新是将数据分析代码生成与方程生成双重任务融合到LLM进化框架中，实现模型主动获取数据特征信息，从而突破传统仅用MSE评价的单向反馈限制。

**🔧 技术方法**

使用大型语言模型（DeepSeek‑V3.1、Qwen3系列等）结合进化搜索、程序合成与执行、经验缓冲区管理，并对模型进行监督微调。

**📊 数据集**

在LLM‑SRBench基准上进行评估，包含LSR‑Transform（改写的费曼方程）和LSR‑Synth（四个学科的合成任务）两个子集。

**📈 对比分析**

与原始LLM‑SR基线以及手工统计提示基线在NMSE指标下对比，PCA在大多数任务和模型上显著降低误差，DeepSeek‑V3.1场景下误差降低约3倍；监督微调后性能进一步提升。

**⚠️ 局限性**

局限性：代码生成模板预设，可能无法捕捉复杂或非标准的数值关系；LLM有时生成不可执行代码；性能高度依赖模型规模，较弱模型受扩展上下文影响更大。

---

## 388. FluxFlow: Conservative Flow-Matching for Astronomical Image Super-Resolution

**arXiv ID:** 2605.03749 | [PDF](https://arxiv.org/pdf/2605.03749v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 389. Sensorless State Estimation and Control for Agile Cable-Suspended Payload Transport by Quadrotors

**arXiv ID:** 2605.03666 | [PDF](https://arxiv.org/pdf/2605.03666v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 390. Decompose to Understand, Fuse to Detect: Frequency-Decoupled Anomaly Detection for Encrypted Network Traffic

**arXiv ID:** 2605.02970 | [PDF](https://arxiv.org/pdf/2605.02970v1)

**作者:** Xinglin Lian `[一作]` (University of Electronic Science and Technology of China), Fan Zhou `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 470863 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出FreeUp框架，利用频率解耦（将加密流量图像分为低频和高频两分支）和不确定性融合实现更精准的异常检测；

**💡 创新点**

首次揭示全频特性与谱偏差问题，并通过双分支频率分离和动态不确定性融合两大创新有效缓解谱偏差；

**🔧 技术方法**

采用傅里叶变换+高低通滤波分离频段、每个分支使用轻量级自编码器、基于Normal-Inverse-Gamma的不确定性学习与动态融合；

**📊 数据集**

在CIC-IoT2023、DoHBrw2020和ISCX-Tor2016这三大公开加密流量数据集上进行实验；

**📈 对比分析**

与DAGMM、GANomaly、ARCADE、MFR、UnDiff、NeuTral、MCM、Anomaly Transformer、TSLANet等九个最先进基线对比，FreeUp在AUC、ACC和F1上均显著领先（最高AUC 95.53%/ACC 93.26%/F1 93.22%），提升幅度约4‑6%；

**⚠️ 局限性**

模型相对较大（双分支导致参数约三倍于单分支），对频率分解参数（如高斯滤波阈值D、包络数P）敏感，且对不同加密协议和实时部署场景的泛化仍需进一步验证。

---

## 391. PIIGuard: Mitigating PII Harvesting under Adversarial Sanitization

**arXiv ID:** 2605.03129 | [PDF](https://arxiv.org/pdf/2605.03129v1)

**作者:** Mingshuo Liu `[一作]` (Vrije Universiteit Amsterdam), Min Chen `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 266 | [OpenAlex ID](https://openalex.org/A5100337210)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种网页级防护方案 PIIGuard，利用隐藏的间接提示注入（IPI）在 HTML 页面中嵌入优化的指令片段，阻止浏览支持的 LLM 在回答时泄露个人身份信息（PII）。

**💡 创新点**

创新点在于：① 将 IPI 作为防护手段并通过进化搜索自动生成最佳隐藏片段；② 设计两阶段泄露评估（规则匹配+判读恢复）确保防护既消除表面泄露又防止语义重构；③ 在三种真实攻击场景（原始 HTML、URL 调用、对抗性 sanitizer）下验证鲁棒性。

**🔧 技术方法**

技术手段包括：隐式 HTML 注入、规则基础泄露评分、演化（mutation + 交叉）搜索、判读恢复评估、两阶段漏评与复合效用排序。

**📊 数据集**

数据集：900 条合成新闻网页，包含四个 PII 字段（姓名、电话、邮箱、地址）以及固定的查询；100 条用于评估，80 条用于搜索训练。

**📈 对比分析**

对比方法：与手工字符替换基线、无防护情况以及不同模型（GPT‑5.4‑nano、Claude‑haiku‑4.5、DeepSeek‑chat）进行比较。PIIGuard 在直接 HTML 模式下 Rule‑和 Judge‑泄露率均达到 97%–100%，在 URL 模式下仍保持 90% 以上的防护率，且对原始问答任务几乎不影响（MAF1 与 BCR 均无明显下降）。

**⚠️ 局限性**

局限性：① 评估依赖单一 LLM 判读，可能存在自利偏差；② URL 实验仅在自托管静态站点，缺乏真实第三方网页验证；③ 只针对结构化的四字段 PII，未覆盖非结构化或其他网页类型；④ 对抗性 sanitizer 的鲁棒性只测试了三种提示模板，实际攻击多样性仍需进一步研究。

---

## 392. RouteHijack: Routing-Aware Attack on Mixture-of-Experts LLMs

**arXiv ID:** 2605.02946 | [PDF](https://arxiv.org/pdf/2605.02946v1)

**作者:** Zhiyuan Xu `[一作]` (Bristol), Lichao Wu `[通讯]` (Bristol)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种针对稀疏专家（MoE）大语言模型的路由感知 jailbreak 攻击（RouteHijack），通过在输入层注入优化后的后缀，直接操纵模型的专家路由决策，从而绕过安全对齐机制。

**💡 创新点**

创新点在于：①通过“响应驱动的对比剖析”精准定位安全与有害专家；②设计三元损失函数，既抑制安全专家又激励有害专家，同时抑制拒绝词；③实现了只需输入级别操作即可对多种 MoE 变体和 VLM 进行有效攻击，并实现了零样本跨模型和跨模态迁移。

**🔧 技术方法**

技术方法包括：Mixture-of-Experts 路由建模、对比激活剖析、基于梯度的离散优化、三元损失（Safety Suppress、Harm Promote、Refusal Unlikelihood）以及 soft‑routing 近似以克服 Top‑K 非可微问题。

**📊 数据集**

使用的数据集主要有：LLM‑LAT（对抗式提示与安全拒绝对），Alpaca 与 WikiText‑2（用于实用性过滤），以及 StrongREJECT（用于评估攻击成功率）。

**📈 对比分析**

与基线、GCG、SAFEx、SteerMoE 等方法比较，RouteHijack 在七款主流 MoE LLM 上平均攻击成功率达到 69.3%（最高 89.1%），比 GCG 提升约 3.2 倍；对五项 NLU 基准的平均性能下降仅 1.3%；且在同族模型与 VLM 上均实现显著的零样本迁移。

**⚠️ 局限性**

局限性包括：①攻击效果受限于 MoE 结构，无法直接针对密集模型；②需要在相似模型上拥有白盒访问进行离线优化；③在部分高安全性或多步推理（CoT）模型上迁移效果较弱；④目前尚未验证对持续对抗训练或路由正则化等防御手段的鲁棒性。

---

## 393. What Shapes Participant Data Quality? A Scoping Review and Case Study of Crowdsourced Webcam Eye Tracking in AI Interviews

**arXiv ID:** 2605.02898 | [PDF](https://arxiv.org/pdf/2605.02898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 394. Parameter-Efficient Multi-View Proficiency Estimation: From Discriminative Classification to Generative Feedback

**arXiv ID:** 2605.03848 | [PDF](https://arxiv.org/pdf/2605.03848v1)

**作者:** Edoardo Bianchi `[一作]` (Free University of Bozen-Bolzano), Antonio Liotta `[通讯]` (Free University of Bozen-Bolzano)

**通讯引用:** 8061 | [OpenAlex ID](https://openalex.org/A5026941307)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了SkillFormer、PATS与ProfVLM三种多视角熟练度估计方法，致力于在Ego-Exo4D数据集上实现高效且可解释的动作质量评估；

**💡 创新点**

创新点包括：①基于跨视角注意力的Selective View Fusion（CrossViewFusion）实现轻量化融合；②针对熟练度的PATS时序采样策略在连续动作片段内提高局部时序密度；③将分类任务改写为生成任务的ProfVLM，利用Vision‑Language模型同时输出熟练度标签和专家式自然语言反馈；

**🔧 技术方法**

使用技术包括：LoRA参数高效微调、TimeSformer视频编码器、CrossViewFusion与AttentiveGatedProjector多视角融合模块、SmolLM2‑135M指令模型的文本生成、以及BERTScore/METEOR/ROUGE-L等评估指标；

**📊 数据集**

数据集为Ego-Exo4D，涵盖6个技能域（烹饪、篮球、足球、舞蹈、音乐、攀岩）并提供多视角同步视频、四级熟练度标签及自由文本专家点评；

**📈 对比分析**

与TimeSformer基线（121M参数）和EgoPulseFormer相比，SkillFormer在仅训练4个epoch、参数量为27M的条件下取得了约4.5倍更少的可训练参数，准确率提升至47.5%；加入PATS后提升至48.0%；ProfVLM在8帧、6个epoch、5.3M参数的设置下达到48.2%的Top‑1准确率，超过基线且参数量降低约20倍；

**⚠️ 局限性**

局限性主要在于：①不同技能域对视角和采样配置的需求差异大，单一模型无法完美覆盖；②仍缺乏长期自适应与个性化反馈机制；③对更复杂的多摄像头设置和离线部署支持有限；④评估主要基于自动化指标，缺少人类主观可操作性评价。

---

## 395. Resource Allocation and AoI-Aware Detection for ISAC with Stacked Intelligent Metasurfaces

**arXiv ID:** 2605.03558 | [PDF](https://arxiv.org/pdf/2605.03558v1)

**作者:** Elaheh Ataeebojd `[一作]` (Centre of Wireless Communications), Mehdi Rasti `[通讯]` (Centre of Wireless Communications)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了基于堆叠智能超材料（SIM）的多用户ISAC系统的能效资源分配与AoI感知检测算法，在两个时间尺度下同时满足eMBB、URLLC和感知服务。

**💡 创新点**

①将SIM的波域自由度与2时尺度调度（puncturing）相结合，首次在同一系统同时满足多种QoS并考虑AoI约束；②用EE最大化的两时尺度分解+Dinkelbach+AO+Lyapunov实现可实现的联合优化；③通过仿真证明SIM显著提升EE，仅需4个天线即可匹配传统BS 24天线的性能。

**🔧 技术方法**

Dinkelbach分数优化、交替优化（AO）、投影梯度下降（PGA）处理SIM相位、Lyapunov drift-plus-penalty控制AoI、OFDM、2时尺度调度、波域传输矩阵、FBL容量、Cramér‑Rao/ beampattern约束、凸/整数规划等技术。

**📊 数据集**

采用仿真生成的随机用户位置和到达速率，无真实数据集。

**📈 对比分析**

与Random‑SIM、No‑SIM、Communication‑Only 三个基准进行对比，实验显示在相同功率下，提出方案比No‑SIM提升约140%‑230% EE，且仅用4个发射天线即可匹配传统BS 24天线；EE随SIM层数、元件数、用户数、RB数、功率等变化符合预期。

**⚠️ 局限性**

仅考虑单小区、完美CSI、静态通道；算法迭代复杂度高；未考虑多用户干扰、硬件误差、频谱共享等；对AoI的平均约束仅使用Lyapunov，无法保证瞬时满足；未给出实时实现的计算开销。

---

## 396. VL-SAM-v3: Memory-Guided Visual Priors for Open-World Object Detection

**arXiv ID:** 2605.03456 | [PDF](https://arxiv.org/pdf/2605.03456v1)

**作者:** Chih-Chung Liu `[一作]` (Wangxuan Institute of Computer Technology), Yongtao Wang `[通讯]` (Wangxuan Institute of Computer Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VL-SAM-v3，利用检索式外部视觉记忆增强开世界目标检测。

**💡 创新点**

创新点在于将检索到的视觉原型转换为稀疏与稠密视觉先验，并通过记忆引导提示细化统一处理开词汇和开类别两种推理。

**🔧 技术方法**

采用多模态嵌入检索、DINOv3特征、稀疏/稠密先验融合、记忆引导提示细化以及场景描述与标签约束解码等技术。

**📊 数据集**

在 LVIS、COCO 以及 GroundingCap-1M 等数据集上进行训练与评估。

**📈 对比分析**

与 LLMDet、SAM3 等基线对比，VL‑SAM‑v3 在 LVIS 零样本开词汇检测中 AP 提升至 53.4/43.5，开类别检测提升至 44.8/41.9，显示显著性能提升。

**⚠️ 局限性**

局限在于仍依赖外部检索库的质量与覆盖，且在非零样本场景下可能无法充分提升，另外检索速度与内存开销也需要进一步优化。

---

## 397. Uncertainty Estimation in Instance Segmentation of Affordances via Bayesian Visual Transformers

**arXiv ID:** 2605.03614 | [PDF](https://arxiv.org/pdf/2605.03614v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 398. TsallisPGD: Adaptive Gradient Weighting for Adversarial Attacks on Semantic Segmentation

**arXiv ID:** 2605.03405 | [PDF](https://arxiv.org/pdf/2605.03405v1)

**作者:** Alexander Matyasko `[一作]`, Wei Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于 Tsallis 熵的对抗攻击方法 TsallisPGD，用于语义分割模型的对抗样本生成。

**💡 创新点**

创新点在于引入可调参数 q 的 Tsallis 交叉熵，通过控制梯度聚焦的置信度区间，动态调整攻击重点；并设计了从 q=-2 到 q=1 的线性 q‑schedule，使单次攻击即可覆盖多种梯度浓度策略，显著提升攻击效果。

**🔧 技术方法**

采用的技术包括：Tsallis 交叉熵损失、投影梯度下降（PGD）与自适应步长 APGD、线性 q‑schedule、以及多 ε 叠加技巧。

**📊 数据集**

在 Cityscapes、Pascal VOC 与 ADE20K 三个公开语义分割数据集上进行实验，并评估多种标准分割模型（PSPNet、UPerNet、Segmenter）以及其对抗训练版本。

**📈 对比分析**

与 CEPGD、SegPGD、CosPGD、JSPGD、MaskedPGD 等强基线对比，TsallisPGD 在 21 个评估设置中平均排名第一，显著降低模型的像素准确率和 mIoU，尤其在对抗训练模型上提升最为显著。

**⚠️ 局限性**

局限性包括：q‑schedule 的选取仍依赖于验证集，缺乏在线自适应机制；仅针对 ℓ∞ 约束的对抗攻击；在极大扰动预算下对清洁模型的提升有限。

---

## 399. Unsecured Lending via Delegated Underwriting

**arXiv ID:** 2605.03307 | [PDF](https://arxiv.org/pdf/2605.03307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 400. RPBA-Net: An Interpretable Residual Pyramid Bilateral Affine Network for RAW-Domain ISP Enhancement

**arXiv ID:** 2605.03626 | [PDF](https://arxiv.org/pdf/2605.03626v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 401. What Happens Inside Agent Memory? Circuit Analysis from Emergence to Diagnosis

**arXiv ID:** 2605.03354 | [PDF](https://arxiv.org/pdf/2605.03354v1)

**作者:** Xutao Mao `[一作]` (City University of Hong Kong), Cong Wang `[通讯]` (City University of Hong Kong)

**通讯引用:** 25870 | [OpenAlex ID](https://openalex.org/A5100390514)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM代理记忆系统的内部机制，追踪写‑管理‑读循环中的特征电路，定位无声失效并提出基于电路的诊断方法。

**💡 创新点**

创新点包括：①发现控制先行、内容后现的规模依赖性；②写读共享的上下文基底（hub）被内存框架招募而非新建；③可检测性与可操作性阈值的分离。

**🔧 技术方法**

使用预训练线性变换器（PLT）进行特征电路追踪、因果验证与干预，结合跨规模 Qwen‑3 系列模型和两种记忆框架（mem0、A‑MEM）。

**📊 数据集**

使用 LongMemEval、LoCoMo、MemoryAgentBench 等公开问答基准数据集进行实验。

**📈 对比分析**

与多数类、熵、行为规则、逻辑回归等无监督基线对比，诊断准确率达 76.2%，失败定位准确率超过 70%，比训练器提升约 24pp；在 8B 规模下，写读共享 hub 的干预可提升 5–8pp 的事实回忆率。

**⚠️ 局限性**

局限性在于可干预性仅在 8B 规模下稳定，较小模型中控制先行但内容失效；诊断方法依赖内部特征分离，跨不同架构或更大模型的泛化尚待验证。

---

## 402. On Computing Total Variation Distance Between Mixtures of Product Distributions

**arXiv ID:** 2605.03839 | [PDF](https://arxiv.org/pdf/2605.03839v1)

**作者:** Weiming Feng `[一作]` (University of Hong Kong), Anqi Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 21902 | [OpenAlex ID](https://openalex.org/A5100331094)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出了在离散域上近似两混合产品分布总变差距离的算法，并在布尔子立方特殊情况下给出精确求解方法。

**💡 创新点**

创新性地构造了递归耦合和粗耦合估计器，使得在混合元件数为常数时可实现FPRAS；对布尔子立方采用组合计数技巧，获得多项式时间精确算法；同时给出了当元件数线性增长时的#P‑难度上界。

**🔧 技术方法**

主要技术包括：递归采样与递归耦合、粗耦合估计、动态规划求解DAG中的失败概率、基于特征函数的包含-排除计数、以及组合数目约束下的精确计数。

**📊 数据集**

本研究不涉及具体数据集，所有结果均在理论上给出。

**📈 对比分析**

相较于以往只能判断两分布是否相同的判定算法，本文给出了相对误差<ε的FPRAS（时间多项式于(nq)^{k1+k2}）以及布尔子立方的多项式时间精确算法，表现优于已知的加性误差近似或仅适用于特殊结构的算法。

**⚠️ 局限性**

主要限制在于算法对混合元件数(k1+k2)的指数依赖；当元件数随维度增长时，FPRAS失效；且仅针对离散域，未给出确定性近似算法。

---

## 403. Annotation Quality in Aspect-Based Sentiment Analysis: A Case Study Comparing Experts, Students, Crowdworkers, and Large Language Model

**arXiv ID:** 2605.03624 | [PDF](https://arxiv.org/pdf/2605.03624v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 404. Cascade Token Selection for Transformer Attention Acceleration

**arXiv ID:** 2605.03110 | [PDF](https://arxiv.org/pdf/2605.03110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 405. FinSTaR: Towards Financial Reasoning with Time Series Reasoning Models

**arXiv ID:** 2605.03460 | [PDF](https://arxiv.org/pdf/2605.03460v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 406. AfriVox-v2: A Domain-Verticalized Benchmark for In-the-Wild African Speech Recognition

**arXiv ID:** 2605.03590 | [PDF](https://arxiv.org/pdf/2605.03590v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 407. Rethinking Temporal Consistency in Video Object-Centric Learning: From Prediction to Correspondence

**arXiv ID:** 2605.03650 | [PDF](https://arxiv.org/pdf/2605.03650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 408. Taming the Curses of Multiagency in Robust Markov Games with Large State Space through Linear Function Approximation

**arXiv ID:** 2605.03125 | [PDF](https://arxiv.org/pdf/2605.03125v1)

**作者:** Jingchu Gai `[一作]` (Carnegie Mellon University), Laixi Shi `[通讯]` (Johns Hopkins University)

**通讯引用:** 209 | [OpenAlex ID](https://openalex.org/A5075795654)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

论文未提供具体内容，无法确定做了什么。

**💡 创新点**

论文未提供具体内容，无法确定创新点。

**🔧 技术方法**

论文未提供具体内容，无法确定使用了什么技术。

**📊 数据集**

论文未提供具体内容，无法确定使用了什么数据集。

**📈 对比分析**

论文未提供具体内容，无法确定比较的方法及性能。

**⚠️ 局限性**

论文未提供具体内容，无法确定限制是什么。

---

## 409. Revisiting the Travel Planning Capabilities of Large Language Models

**arXiv ID:** 2605.03308 | [PDF](https://arxiv.org/pdf/2605.03308v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 410. Meta-Inverse Physics-Informed Neural Networks for High-Dimensional Ordinary Differential Equations

**arXiv ID:** 2605.03511 | [PDF](https://arxiv.org/pdf/2605.03511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 411. Robust Visual SLAM for UAV Navigation in GPS-Denied and Degraded Environments: A Multi-Paradigm Evaluation and Deployment Study

**arXiv ID:** 2605.03678 | [PDF](https://arxiv.org/pdf/2605.03678v1)

**作者:** Prasoon Kumar `[一作]` (National Institute of Technology Patna), Sandeep Kumar `[通讯]` (Central Research Laboratory Bharat Electronics Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对五种 V-SLAM 系统（ORB‑SLAM3、DPVO、DROID‑SLAM、DUSt3R、MASt3R）在 GPS 缺失与低能见度（低光、尘雾、运动模糊、组合）环境下进行系统化比较评估，构建自制单目室内数据集并验证；

**💡 创新点**

创新点在于统一评估不同范式 SLAM 在多种降解模式下的鲁棒性，提出基于统计显著性检验的量化方法，并给出嵌入式部署的 SWaP‑C 指南；

**🔧 技术方法**

采用经典特征、深度补丁、递归优化、Vision Transformer 等多范式 SLAM，统一前端后端管线，并在 GPU 加速环境下进行统计显著性比较；

**📊 数据集**

使用 TUM RGB‑D、EuRoC MAV、UMA‑VI、SubT‑MRS 四个公开数据集以及自制带低光、尘雾、运动模糊等降解的室内单目数据集；

**📈 对比分析**

通过 ATE、RPE、TSR、FPS、GPU 内存等指标量化比较，MASt3R 在降解环境下取得最低 ATE（0.027 m）和最高 TSR（≈96%），DPVO 在内存友好与实时性方面表现最佳，ORB‑SLAM3 在低能见度下失效；

**⚠️ 局限性**

实验仅在单目不融合 IMU、固定高端 RTX‑3090 GPU 进行，未涵盖动态遮挡、热成像、事件相机等实际场景，也未测量能耗或评估攻击鲁棒性，限制了结果的普适性。

---

## 412. Most ReLU Networks Admit Identifiable Parameters

**arXiv ID:** 2605.03601 | [PDF](https://arxiv.org/pdf/2605.03601v1)

**作者:** Moritz Grillo `[一作]` (Max Planck Institute for Mathematics in the Sciences), Guido Montúfar `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

论文研究了深度 ReLU 网络的实现映射（realization map），并证明在大多数架构（隐藏层宽度至少为2）中存在可识别参数的开放子集；进一步证明了功能维度等于参数总数减去隐藏神经元数，并揭示了最小化网络仍可能存在非平凡的参数冗余；最后提出了泛化深度层次结构，即大多数参数下的深网络无法被更浅网络通用表示。

**💡 创新点**

创新点包括：1) 引入加权多面体复合体（weighted polyhedral complex）框架来捕捉 ReLU 网络的几何结构；2) 在“泛化”参数空间（generic parameters）下证明超转移性（supertransversality）和无消除性（cancellation‑free）属性；3) 通过构造有向依赖图（dependency graph）实现对参数纤维的精确描述；4) 展示最小化并不意味着可识别，打破了先前关于可识别性与最小化对应的猜想；5) 给出泛化深度层次结果，证明深度无法被宽度所替代。

**🔧 技术方法**

主要技术手段包括：多面体几何与泛化排列理论、加权多面体复合体与 Tropical 权重的紧耦合、逆向工程（reverse‑engineering）恢复网络层参数、构造斜面层（slab layer）实现可视化分割、利用可转移性（transversality）与透明性（transparency）证明 TPIC 与 LRA 条件、以及代数几何方法分析参数纤维的结构。

**📊 数据集**

该工作为纯理论分析，无实验数据集；作者在论文中仅使用理论构造和数学证明，未采用具体机器学习数据集。

**📈 对比分析**

方法比较基于数学证明和结构性对比；通过构造证明开放子集满足可识别性，从而实现功能维度的确切上界；与之前只适用于特定宽度或金字塔网络的结果相比，作者给出了更通用、宽度不受限制的结论；在深度层次方面，证明了泛化层次结构，表明在大多数参数下深度不可被宽度替代。

**⚠️ 局限性**

限制与未解决的问题包括：1) 对于宽度为1的隐藏层的可识别性仍未完全解决；2) 结论依赖于参数的“泛化”假设，极值或特殊结构参数可能不满足；3) 对多层网络的参数空间维度与可视化结构的完整描述尚未完成；4) 实际优化过程中梯度与损失景观的具体影响尚未通过实验验证。

---

## 413. Random test functions, $H^{-1}$ norm equivalence, and stochastic variational physics-informed neural networks

**arXiv ID:** 2605.03542 | [PDF](https://arxiv.org/pdf/2605.03542v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 414. Exploring Sustainability in Scientific Software through Code Quality & Test Coverage Metrics

**arXiv ID:** 2605.03243 | [PDF](https://arxiv.org/pdf/2605.03243v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 415. Distributed Deep Variational Approach for Privacy-preserving Data Release

**arXiv ID:** 2605.03069 | [PDF](https://arxiv.org/pdf/2605.03069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 416. Benchmarking Parameter-Efficient Fine-Tuning of Large Language Models for Low-Resource Tajik Text Generation with the Tajik Web Corpus

**arXiv ID:** 2605.03742 | [PDF](https://arxiv.org/pdf/2605.03742v1)

**作者:** Mullosharaf K. Arabov `[一作]` `[通讯]`, Mullosharaf K. Arabov

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了塔吉克语最大规模的网络语料库，并对多种大型语言模型在该语料上进行文本生成微调实验。

**💡 创新点**

首次提供塔吉克语最大的公开语料库，并系统评估了LoRA/QLoRA等参数高效微调方法在低资源、非拉丁脚本语言上的效果。

**🔧 技术方法**

使用LoRA、QLoRA等参数高效微调技术，对GPT‑2、DistilGPT‑2、Mistral 7B、Phi‑2、Qwen 2.5、mT5‑small等模型进行训练。

**📊 数据集**

塔吉克语网络语料库（319,298 文档，约1.11 亿字符）以及从中抽取的10,000文档子样本。

**📈 对比分析**

对17种架构-微调-秩组合进行多次实验，评估指标包括困惑度、交叉熵、显存占用和训练时间；Mistral 7B+QLoRA（r=8/16）取得最佳困惑度≈5，GPT‑2 Medium全微调困惑度≈3.48但出现灾难性遗忘。

**⚠️ 局限性**

实验仅使用10k文档子样本、固定超参、仅评估生成质量，缺乏人类评测、任务多样性以及对完整语料库的验证。

---

## 417. Operationalizing Software Engineering Theories for Practical Validation

**arXiv ID:** 2605.03257 | [PDF](https://arxiv.org/pdf/2605.03257v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 418. AdapShot: Adaptive Many-Shot In-Context Learning with Semantic-Aware KV Cache Reuse

**arXiv ID:** 2605.03644 | [PDF](https://arxiv.org/pdf/2605.03644v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 419. Evaluating Prompting and Execution-Based Methods for Deterministic Computation in LLMs

**arXiv ID:** 2605.03227 | [PDF](https://arxiv.org/pdf/2605.03227v1)

**作者:** Hongkun Yu `[一作]` `[通讯]` (Virginia Tech), Hongkun Yu (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估LLM在确定性计算任务（二进制计数、最长子串、算术计算）中的表现，并比较多种提示策略与程序化思维（PoT）及微调代码生成模型的效果。

**💡 创新点**

系统化对比标准提示、CoT、Least‑to‑Most、PoT与Self‑Consistency，并证明PoT与CodeT5‑small可实现完美准确率，揭示LLM单纯自然语言推理对精确计算的局限。

**🔧 技术方法**

采用Chain‑of‑Thought、Least‑to‑Most、Program‑of‑Thought、Self‑Consistency提示，外部Python解释器执行生成代码，以及对CodeT5‑small进行微调生成可执行程序。

**📊 数据集**

构建合成数据集，包含三类任务（二进制计数、最长子串、算术计算），共3000条样本，所有任务均提供自然语言指令、输入和可执行目标代码。

**📈 对比分析**

在单任务与混合任务评估中，PoT与CodeT5‑small均达100%准确率；标准提示、CoT仅约56–64%；Least‑to‑Most表现最差；Self‑Consistency提升至约70%但计算成本高。

**⚠️ 局限性**

实验仅针对合成、结构化任务，未验证在真实世界长序列、噪声输入或开放式问题中的泛化能力。

---

## 420. Can Multimodal Large Language Models Understand Pathologic Movements? A Pilot Study on Seizure Semiology

**arXiv ID:** 2605.03352 | [PDF](https://arxiv.org/pdf/2605.03352v1)

**作者:** Lina Zhang `[一作]` (University of California), Vwani Roychowdhury `[通讯]` (University of California)

**通讯引用:** 9564 | [OpenAlex ID](https://openalex.org/A5043479061)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在癫痫发作视频中进行零样本病理运动识别，并通过面部裁剪、姿态估计与音频去噪等特征‑定向预处理提升模型性能。

**💡 创新点**

首次将通用多模态大语言模型与面部裁剪、姿态估计、音频去噪等信号增强相结合，实现对20种ILAE癫痫半征的零样本识别，并提供可解释的自然语言说明。

**🔧 技术方法**

使用InternVL‑3.5、Qwen‑VL‑2.5 及 Audio Flamingo 3 等多模态大语言模型；同时采用面部裁剪、姿态估计、SEGAN音频去噪以及 Whisper 语音转写等预处理技术。

**📊 数据集**

90段来自 UCLA 29例成人癫痫监测中心的视频，人工标注20项ILAE标准半征。

**📈 对比分析**

与微调后的 CNN 与 ViViT 基线对比，零样本 MLLM 在 13/18 关键特征上 F1 更高；信号增强后进一步提升 10/20 特征；解释性评估显示 94.3% 的正向预测生成的解释至少达到 60% 的正确性。

**⚠️ 局限性**

对细微、高频运动的识别仍不够准确；仅来自单中心的数据，缺乏多中心验证；零样本准确率尚未满足临床直接使用，需进一步微调和更精细的多模态融合。

---

## 421. Effective Performance Measurement: Challenges and Opportunities in KPI Extraction from Earnings Calls

**arXiv ID:** 2605.03147 | [PDF](https://arxiv.org/pdf/2605.03147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 422. Exposing LLM Safety Gaps Through Mathematical Encoding:New Attacks and Systematic Analysis

**arXiv ID:** 2605.03441 | [PDF](https://arxiv.org/pdf/2605.03441v1)

**作者:** Haoyu Zhang `[一作]` (Northeastern University), Shanu Sushmita `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并系统比较了六种将有害提示转化为数学问题的编码攻击，验证其对八款主流LLM安全机制的绕过效果；

**💡 创新点**

提出了新的“Formal Logic”编码方式，并发现深度重构（而非仅使用数学符号）是成功关键；

**🔧 技术方法**

利用LLM辅助模型（GPT‑4.1‑Mini）进行深度重构，规则化方法则采用定向算术、条件概率和符号注入；

**📊 数据集**

使用HarmBench（159条有害行为）和JailbreakBench（100条）两大基准进行评估；

**📈 对比分析**

在两个基准上，LLM‑based编码平均攻击成功率（ASR）为46‑56%，规则化编码仅为9‑11%，未编码基线为7‑10%；Set Theory与Formal Logic相当，Quantum Mechanics略逊；模型层面上，Gemini 系列最易攻击，GPT‑5 最难；

**⚠️ 局限性**

结果受自动判定器影响，未人工审核；LLM‑based与规则化方法因多重变量共变，难以单独归因重构深度；实验仅涉及英文提示；缺乏对攻击输出质量的细粒度评估；

---

## 423. Sorry for the late reply: Response times and reciprocity in WhatsApp and Instagram chats

**arXiv ID:** 2605.03687 | [PDF](https://arxiv.org/pdf/2605.03687v1)

**作者:** Florian Martin `[一作]` (Bielefeld University), Hanna Drimalla `[通讯]` (Bielefeld University)

**通讯引用:** 416 | [OpenAlex ID](https://openalex.org/A5052236115)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了WhatsApp和Instagram即时聊天的回复时间，量化了聊天双方的回复速度平衡，并检验其随时间的稳定性。

**💡 创新点**

首次将回复速度作为定量的互惠指标，结合 Jensen‑Shannon 相似度和线性混合模型评估即时聊天中的时间互惠。

**🔧 技术方法**

使用了Jensen‑Shannon距离、线性混合效应模型、消息合并以及月度相似度分析等方法。

**📊 数据集**

基于3.4 百万条匿名WhatsApp和Instagram聊天元数据（包括消息长度、时间戳和匿名ID）收集自约97名年轻用户的捐赠数据。

**📈 对比分析**

通过相似度分布、LMM斜率和每月 MAD 对比，发现双方回复时间高度相似，LMM斜率≈0.79，MAD≤0.07，表明互惠稳定。

**⚠️ 局限性**

样本年轻且缺乏对话伙伴的人口学信息，且仅考虑时间不考虑文本长度，可能影响普适性与精细时间尺度的捕捉。

---

## 424. Deepfake Audio Detection Using Self-supervised Fusion Representations

**arXiv ID:** 2605.03420 | [PDF](https://arxiv.org/pdf/2605.03420v1)

**作者:** Khalid Zaman `[一作]` (Japan Advanced Institute of Science and Technology), Masashi Unoki `[通讯]` (Japan Advanced Institute of Science and Technology)

**通讯引用:** 2266 | [OpenAlex ID](https://openalex.org/A5014199725)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种双分支深度伪造音频检测框架，分别使用 XLS-R 与 BEATs 预训练模型提取语音与环境声的自监督表示，并通过匹配头与多头跨注意力实现表示融合与原始类别推理。

**💡 创新点**

创新点在于：① 关注组件级伪造（语音与环境可独立伪造）而非整体信号；② 设计匹配头对两种表示差异进行统计归一化与交互，直接估计原始类别；③ 引入双向跨注意力实现语音与环境表示的互补信息交换。

**🔧 技术方法**

使用技术包括自监督预训练模型 XLS-R（语音）、BEATs（环境声）、多头跨注意力、匹配头（统计池化+交互）、AASIST 分类器、任务加权多任务损失、数据增强与类别平衡采样。

**📊 数据集**

主要数据集为 CompSpoofV2，涵盖 5 类（原始、bonafide_bonafide、spoof_bonafide、bonafide_spoof、spoof_spoof），总计约 250k 4 秒音频，分为训练/验证/评估/测试四份。

**📈 对比分析**

在评估集和测试集上，提出的方法 F1‑score 分别提升到 70.11% 与 70.20%，相比基线提升约 7–8%；环境声 EER 由基线 43.36% 降至 16.54%/18.83%，展示显著性能提升；训练效率亦比基线快约 4 倍。

**⚠️ 局限性**

局限性包括：① 主要在合成数据集上验证，缺乏真实环境噪声下的鲁棒性评估；② 双分支模型计算量较单一编码器大，部署成本相对较高；③ 仍有少数相似类别混淆，表明对极细粒度伪造的区分仍需进一步提升。

---

## 425. Bandits attack function optimization

**arXiv ID:** 2605.03496 | [PDF](https://arxiv.org/pdf/2605.03496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 426. From Code to Prediction: Fine-Tuning LLMs for Neural Network Performance Classification in NNGPT

**arXiv ID:** 2605.03686 | [PDF](https://arxiv.org/pdf/2605.03686v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 427. Leveraging Code Automorphisms for Improved Syndrome-Based Neural Decoding

**arXiv ID:** 2605.03620 | [PDF](https://arxiv.org/pdf/2605.03620v1)

**作者:** Raphaël Le Bidan `[一作]` (IMT Atlantique), Charbel Abdel Nour `[通讯]` (IMT Atlantique)

**通讯引用:** 1326 | [OpenAlex ID](https://openalex.org/A5043390452)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

利用码自动同构进行数据增强，提高SBND模型在训练和推理阶段的性能，逼近最大似然解码（MLD）效果。

**💡 创新点**

首次将码自动同构同时应用于训练时的数据增强和推理时的测试时增强（TTA），并将其与传统自动同构集成解码相结合，显著提升了SBND的泛化与准确率。

**🔧 技术方法**

使用SBND框架中的ECCT（基于Transformer的模型）、BCH/Polar码的自动同构群、混合精度训练、交叉熵损失以及TTA等技术手段。

**📊 数据集**

实验采用短码集合：BCH(31,21,5)、(63,45,7)以及Polar(128,64,8)，并构建了固定训练集（包含ML标签与oracle标签）。

**📈 对比分析**

通过与原始ECCT、GRU、CrossMPT、Mamba-Transformer以及SCL/OSD等基线对比，实验显示经过自动同构增强和TTA后，FER大幅下降，逼近MLD，甚至在某些SNR下超越SCL-8。

**⚠️ 局限性**

局限性包括：仅在短高码率码上验证，训练需要昂贵的ML标签；推理时TTA会增加计算负担；对更长码或低率码的适用性尚未验证。

---

## 428. WorldJen: An End-to-End Multi-Dimensional Benchmark for Generative Video Models

**arXiv ID:** 2605.03475 | [PDF](https://arxiv.org/pdf/2605.03475v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 429. Degeneracy-Aware Functional and Algorithmic Resilience in Virtualized 6G Networks Under Correlated Failures

**arXiv ID:** 2605.03035 | [PDF](https://arxiv.org/pdf/2605.03035v1)

**作者:** Mohamed Khalafalla Hassan `[一作]` (Walton Institute), Indrakshi Dey `[通讯]` (Walton Institute)

**通讯引用:** 330 | [OpenAlex ID](https://openalex.org/A5078593882)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于退化性（degeneracy）的弹性评估框架，构建了功能替代评分（FSS）、算法弹性商（ARQ）和跨层退化指数（MLDI）三大指标，用以量化结构多样性与功能等价性；

**💡 创新点**

创新点在于将结构多样性与功能等价性分离，形成可区分结构差异的替代度量，并通过模拟验证冗余与实际弹性存在显著差异；

**🔧 技术方法**

采用相似性核、结构距离度量、熵加权交叉层指标等技术，对虚拟化6G网络的节点、算法与层级进行度量，配合有针对性的故障实验；

**📊 数据集**

使用合成数据集，模拟O‑RAN、NFV部署参数，生成节点容量、负载、可用性、可靠性和结构签名等特征；

**📈 对比分析**

通过与传统冗余、多RAT基线对比实验，结果显示FSS、ARQ、MLDI在中高失效率时显著提升功能服务连续性，退化点被推后；

**⚠️ 局限性**

局限性在于计算复杂度为O(n²)，主要适用于离线规划；指标对阈值和权重敏感，且实验仅基于合成数据，缺乏真实网络验证。

---

## 430. Conditions for well-posed color recovery in scattering media

**arXiv ID:** 2605.03837 | [PDF](https://arxiv.org/pdf/2605.03837v1)

**作者:** Grigory Solomatov `[一作]` (University of Haifa), Derya Akkaynak `[通讯]` (Interuniversity Institute for Marine Sciences)

**通讯引用:** 2228 | [OpenAlex ID](https://openalex.org/A5027485746)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f`

**🎯 论文内容**

探讨在散射介质中通过理论条件实现颜色恢复的唯一性

**💡 创新点**

提出“恢复模式”与“恢复集”概念，证明满足这些结构时颜色恢复可被证明为良定，并将暗通道先验纳入理论框架

**🔧 技术方法**

基于辐射传输方程简化、光谱投影分析、线性代数与微分约束的理论推导

**📊 数据集**

未使用任何实际或合成数据集，全部为理论分析

**📈 对比分析**

无实验对比，文章仅给出理论证明与公式，未给出数值性能

**⚠️ 局限性**

假设理想无损光谱相机、已知深度、常数瞬时辐射增益、忽略前向散射、需要准确恢复模式检测等理想化假设

---

## 431. Reproducing Complex Set-Compositional Information Retrieval

**arXiv ID:** 2605.03824 | [PDF](https://arxiv.org/pdf/2605.03824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 432. RoboAlign-R1: Distilled Multimodal Reward Alignment for Robot Video World Models

**arXiv ID:** 2605.03821 | [PDF](https://arxiv.org/pdf/2605.03821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 433. Enhancing Visual Question Answering with Multimodal LLMs via Chain-of-Question Guided Retrieval-Augmented Generation

**arXiv ID:** 2605.03790 | [PDF](https://arxiv.org/pdf/2605.03790v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 434. Firmware Distribution as Attack Surface: A Security Study of ASIC Cryptocurrency Miners

**arXiv ID:** 2605.03770 | [PDF](https://arxiv.org/pdf/2605.03770v1)

**作者:** Pierre Pouliquen `[一作]`, Antoine Houssais `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文仅演示IEEEtran.cls会议论文模板的使用方法，并未开展实际研究。

**💡 创新点**

创新点在于展示了如何利用IEEEtran.cls构建会议论文的结构与格式。

**🔧 技术方法**

使用的技术主要是LaTeX与IEEEtran.cls宏包。

**📊 数据集**

未使用任何数据集。

**📈 对比分析**

无实验或比较方法，无法评估性能。

**⚠️ 局限性**

缺乏实际研究内容，无法提供实验结果或性能评估，适用范围有限。

---

## 435. LIPPEN: A Lightweight In-Place Pointer Encryption Architecture for Pointer Integrity

**arXiv ID:** 2605.03974 | [PDF](https://arxiv.org/pdf/2605.03974v1)

**作者:** Erfan Iravani `[一作]` (Virginia Tech), Wenjie Xiong `[通讯]` (Virginia Tech)

**通讯引用:** 1309 | [OpenAlex ID](https://openalex.org/A5008550066)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

设计并实现了一种全指针加密架构（LIPPEN），在64位指针上直接加密以实现强指针完整性与机密性，并实现零元数据开销。

**💡 创新点**

创新点：① 用完整64位加密取代PAC的截短MAC，显著提升熵并抵抗暴力攻击；② 通过与PAC兼容的ISA扩展实现无缝迁移；③ 采用PRINCEv2轻量级块密码，并引入灵活的modifier设计，保持硬件低功耗。

**🔧 技术方法**

技术实现：轻量级块密码PRINCEv2（ECB模式），RISC‑V Rocket/BOOM核心的RoCC加速器，LLVM编译器插件，ISA扩展指令，FPGA原型验证。

**📊 数据集**

使用数据集：微基准、nbench、SPEC CPU2017率基准，并与Apple M1 PAC做对比。

**📈 对比分析**

比较方法：在FPGA上将LIPPEN与自研RISC‑V PAC（QARMA实现）以及Apple M1 PAC进行跑分，测量指令计数和运行时开销。性能表现：与PAC相比，指令计数提升约0.2%–1%，运行时开销在0.2%–3%之间；硬件面积和功耗提升低于4%。

**⚠️ 局限性**

局限性：仍需优化指针算术与编译器调度；对C++异常处理兼容性未完成；密钥管理依赖外部机制；在深度递归返回时预测失效会增加开销；缺乏针对Spectre等预期攻击的完整安全分析。

---

## 436. Reservoir property image slices from the Groningen gas field for image translation and segmentation

**arXiv ID:** 2605.03942 | [PDF](https://arxiv.org/pdf/2605.03942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 437. Transformers with Selective Access to Early Representations

**arXiv ID:** 2605.03953 | [PDF](https://arxiv.org/pdf/2605.03953v1)

**作者:** Skye Gunasekaran `[一作]` (University of California Santa Cruz), Jason Eshraghian `[通讯]` (University of California Santa Cruz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Selective Access Transformer（SATFormer）架构，利用对第一层value投影的可学习门控实现对早期表示的可控重用；

**💡 创新点**

创新点在于将早期表示的访问从层级全局的静态系数转变为token与head依赖的动态门控，既保持了单一路径的简洁，又实现了细粒度、上下文感知的重用；

**🔧 技术方法**

技术包括在标准Transformer基础上添加ReLU门控的线性投影，采用与ResFormer、DenseFormer、MUDDFormer、HyperConnections等方法对比；

**📊 数据集**

使用FineWeb‑Edu子集作为预训练数据，配合Llama‑2词表，评估检索密集基准（TriviaQA、SWDE、SQuAD、NQ、FDA、DROP）以及通用语言建模与零样本任务；

**📈 对比分析**

与Transformer、ResFormer及更密集的架构比较，SATFormer在检索基准上比ResFormer提升约1.5点，在XL规模上零样本平均提升0.67点，同时保持与Transformer相近的吞吐量和显存，显著优于DenseFormer、MUDDFormer等密集方案；

**⚠️ 局限性**

局限性包括在某些规模下仍落后于最密集方法，对深度与配置敏感，门控对不同特征类别的差异仍有限，且缺乏对深度、宽度、训练token独立尺度的全面分析。

---

## 438. UnAC: Adaptive Visual Prompting with Abstraction and Stepwise Checking for Complex Multimodal Reasoning

**arXiv ID:** 2605.03950 | [PDF](https://arxiv.org/pdf/2605.03950v1)

**作者:** Yifan Wang `[一作]` (Northeastern University), Yun Fu `[通讯]` (Northeastern University)

**通讯引用:** 31881 | [OpenAlex ID](https://openalex.org/A5005819096)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种多模态提示框架 UnAC（理解、抽象、检查），通过自适应视觉提示、视觉信息抽象和逐步自检三步策略提升 LMM 在复杂视觉推理任务中的表现。

**💡 创新点**

创新点包括：① 基于问题动态生成的自适应视觉提示（结合分割与 OCR）；② 将图像关键信息转化为文本进行抽象；③ 采用逐步自检机制逐子问题校验，避免一次性全局检查的错误。

**🔧 技术方法**

采用的技术包括：SEEM 分割、OCR（easyOCR）生成视觉标记；视觉抽象提示和多步骤链式思维；渐进式自检；在 GPT‑4V、Gemini‑1.5‑Flash、LLaVA、InternVL 等 LMM 上无参数更新进行实验。

**📊 数据集**

实验数据集：MathVista（视觉数学推理）、MM‑Vet（复杂 VQA）和 MMMU（专业多模态推理）。

**📈 对比分析**

与基线及现有提示方法（SoM、CCoT、SKETCHPAD）对比，UnAC 在所有模型上均实现显著提升：GPT‑4V+UnAC 在 MathVista 上提升 4.9%，InternVL+UnAC 提升 4.3%，在 MM‑Vet 与 MMMU 上均提升 3–5% 以上，证明其模型无关性与强大效果。

**⚠️ 局限性**

局限性：对高度抽象的几何等问题仍受模型本身视觉识别能力限制；自检虽然有效但仍受 LMM 推理瓶颈影响，全球检查方法效果不佳；视觉提示不一定在所有任务中均能显著提升，需进一步研究更通用的提示策略。

---

## 439. Integrating Feature Correlation in Differential Privacy with Applications in DP-ERM

**arXiv ID:** 2605.03945 | [PDF](https://arxiv.org/pdf/2605.03945v1)

**作者:** Tianyu Wang `[一作]` (Columbia University), Rachel Cummings `[通讯]` (Columbia University)

**通讯引用:** 1117 | [OpenAlex ID](https://openalex.org/A5077952586)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种“相关性感知差分隐私”框架，允许在数据中区分敏感与不敏感特征，并通过特征间的总变差距离来调节噪声大小，进而在差分隐私下的经验风险最小化（DP‑ERM）和梯度下降（DP‑SGD）中提升隐私-效用平衡。

**💡 创新点**

创新点在于：①引入基于总变差的相关性度量，将不敏感特征的噪声尺度降到依赖于与敏感特征的相关性；②给出全局差分隐私下的改进算法与理论上更优的误差界；③设计可估计TV距离的机制，保持实用性；④提供与传统DP、半敏感DP等基线的实证对比。

**🔧 技术方法**

技术主要包括：相关性感知差分隐私定义、基于总变差的相关性测度、改进的Laplace和Gaussian机制、Corr‑SGD（自适应噪声尺度）以及TV距离的估计与上界保密性分析。

**📊 数据集**

实验数据集涵盖：Synthetic（高维高相关Gaussian生成数据）、Adult（收入预测）、Sepsis（存活预测）、Credit Card（违约预测）、Medical Cost（医疗费用预测）。

**📈 对比分析**

与Semi、Standard、Partial三种基线对比，Corr‑DP在相同隐私预算下的误差/准确率均优于Standard，且接近Semi，尤其在高隐私（低ε）场景下表现更显著。

**⚠️ 局限性**

局限性包括：①需要明确划分敏感/不敏感特征，边界模糊时效果下降；② TV距离估计在高维下计算成本高；③ 对特征空间和模型结构有一定假设（如光滑性、相关性低等），不满足时可用更宽松的版本。

---

## 440. Computing Planar Convex Hulls with a Promise

**arXiv ID:** 2605.03904 | [PDF](https://arxiv.org/pdf/2605.03904v1)

**作者:** Sepideh Aghamolaei `[一作]` (Amirkabir University of Technology), Benjamin Raichel `[通讯]` (University of Texas at Dallas)

**通讯引用:** 310 | [OpenAlex ID](https://openalex.org/A5054913384)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文提出了在“凸包顶点按x坐标排序”这一弱约束下，计算平面点集凸包的两种算法：一个确定性算法实现 O(n√log n) 时间，另一个随机化算法期望时间为 n·2^{O(√log log n)}，并给出了对应的下界证明，表明只要有一点点违背该约束，时间复杂度会退回到 Ω(n log n)。

**💡 创新点**

创新点主要在于：①首次证明在仅保证凸包顶点按顺序出现的情形下即可突破 Ω(n log n) 下界；②提出 L‑bad 对、桥（bridge）与细分垂直线集合相结合的分治框架，既可实现确定性 O(n√log n) 结果，又能通过随机采样与递归降低到 n·2^{O(√log log n)}；③给出了匹配的下界，证明该约束的“极限”，即只缺一条顶点就会回到 Ω(n log n)。

**🔧 技术方法**

核心技术包括：垂直线分层（slab）分解、L‑bad 对检测、桥（bridge）计算、Graham 采样与扫描、递归细化（双重递归）以及随机化线的选择和量化（quantiles）。算法通过在每个细分区内仅保留“安全点”（safe points）并递归处理，避免全局排序。下界证明则利用自适应对抗模型和排列排序的硬性要求。

**📊 数据集**

实验/数据集：本文为理论工作，没有使用具体数值数据集，而是通过抽象点集模型（如抛物线点、随机采样线）演示算法与下界。所有证明均基于理论分析与对抗模型。

**📈 对比分析**

与传统的 O(n log n) 凸包算法相比，本文算法在满足约束时实现了显著的时间改进：确定性版本为 O(n√log n)，随机化版本为 n·2^{O(√log log n)}（即 o(n log^ε n)）。虽然算法结构更复杂，常数和实现难度较高，但在理论上展示了可在子对数时间内完成凸包的可能性。

**⚠️ 局限性**

局限性：①需严格满足“凸包顶点按x坐标排序”这一前置条件，稍有违背即可恢复 Ω(n log n) 的下界；②随机化算法尚未实现确定化；③算法实现复杂度高，常数因素未评估，实际应用中的效率不确定；④仅在二维平面讨论，三维情况仍无法突破 Ω(n log n)。

---

## 441. Raising the Ceiling: Better Empirical Fixation Densities for Saliency Benchmarking

**arXiv ID:** 2605.03885 | [PDF](https://arxiv.org/pdf/2605.03885v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 442. Deco: Extending Personal Physical Objects into Pervasive AI Companion through a Dual-Embodiment Framework

**arXiv ID:** 2605.03882 | [PDF](https://arxiv.org/pdf/2605.03882v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 443. DMGD: Train-Free Dataset Distillation with Semantic-Distribution Matching in Diffusion Models

**arXiv ID:** 2605.03877 | [PDF](https://arxiv.org/pdf/2605.03877v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 444. Bodyless Presence: Reconsidering the Minimal Self in Immersive Video

**arXiv ID:** 2605.03873 | [PDF](https://arxiv.org/pdf/2605.03873v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 445. Correct Is Not Enough: Training Reasoning Planners with Executor-Grounded Rewards

**arXiv ID:** 2605.03862 | [PDF](https://arxiv.org/pdf/2605.03862v1)

**作者:** Tianyang Han `[一作]` (D4 Lab), Junhao Su `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个planner‑executor框架，通过奖励中间推理轨迹的质量和对固定执行器的提升来训练推理计划者。

**💡 创新点**

创新点在于将基于rubric的Reasoning Reward Model与执行器提升联合起来的双重奖励机制，避免仅靠最终答案评估导致的信用分配错误，并构建了专门的reason‑only对齐数据集。

**🔧 技术方法**

技术上采用组合作用学习（GRPO）进行策略优化，离线训练奖励模型（RM）以多维rubric评分为监督，利用K次采样估计执行器提升，并将两者结合成最终奖励。

**📊 数据集**

使用的数据集包括从GSM8K和OpenCodeReasoning提取的6,000组reason‑only轨迹，以及HumanEval、HumanEval+、MBPP、LiveCodeBench、GSM8K、GSM‑Hard、SVAMP和MATH500等评测基准。

**📈 对比分析**

在代码和数学任务上与仅使用执行器奖励的Exec‑only方法对比，TraceLift在所有基准上平均提升约2–4%的通过率，且不需要更大的模型规模或额外的训练预算。

**⚠️ 局限性**

局限性在于需要固定执行器，RM依赖手工rubric标注且可能在跨任务或更复杂执行器时泛化受限；同时K次采样的执行器提升估计会增加计算成本。

---

## 446. Aspect-Aware Content-Based Recommendations for Mathematical Research Papers

**arXiv ID:** 2605.03861 | [PDF](https://arxiv.org/pdf/2605.03861v1)

**作者:** Ankit Satpute `[一作]` (FIZ Karlsruhe), Bela Gipp `[通讯]` (University of Göttingen)

**通讯引用:** 6131 | [OpenAlex ID](https://openalex.org/A5058837356)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种面向数学论文的内容相似推荐系统，强调关系由概念、方法、结构等方面驱动，而非文本或引用重叠

**💡 创新点**

创新点在于(1) 首次构建面向数学的专家标注数据集GoldRiM和大规模自动生成数据集SilverRiM；(2) 设计Aspect‑Conditioned Heterogeneous GNN（AchGNN），在图神经网络中同时建模论文文本语义、引用网络和作者血缘，并在评分层进行多方面条件化；(3) 通过辅助的方面预测任务提升表示的区分度

**🔧 技术方法**

技术包括：大语言模型（Qwen3‑7B）文本和作者嵌入；异构图卷积（GraphSAGE）消息传递；方面条件化交互向量与MLP评分；贝叶斯个性化排序（BPR）与交叉熵混合训练；多层采样邻居、注意力机制与层数调优

**📊 数据集**

使用数据集：GoldRiM（80种子·420对，人工专家标注），SilverRiM（212k种子·532k对，自动抽取方面标签），以及Papers‑with‑Code（机器学习领域）做跨域验证

**📈 对比分析**

与基线（SciBERT‑FT、Qwen3‑7B‑FT、GraphCL、Mabowdor等）对比，AchGNN在GoldRiM和SilverRiM上均获得最高P@10、R@10、MRR，GoldRiM上MRR达0.593，较GraphCL提升约26%；在PwC数据集也表现出与SciBERT‑FT相近甚至更优的性能

**⚠️ 局限性**

局限性包括GoldRiM规模有限、单一专家偏倚；SilverRiM自动抽取方面标签噪声高；实验集中在zbMATH Open，缺乏跨数据库的广泛评测；目前只考虑四类主要方面，未覆盖所有潜在概念关系；公式信息对性能提升有限，仍需进一步挖掘

---

## 447. Generating Proof-of-Vulnerability Tests to Help Enhance the Security of Complex Software

**arXiv ID:** 2605.03956 | [PDF](https://arxiv.org/pdf/2605.03956v1)

**作者:** Shravya Kanchi `[一作]` (Virginia Tech), Na Meng `[通讯]` (Virginia Tech)

**通讯引用:** 2222 | [OpenAlex ID](https://openalex.org/A5070152860)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于 AI 代理（Codex 等）的自动化流水线，用于识别 Java 应用中可攻击的入口点，生成、编译、执行并评估针对第三方库漏洞的 Proof‑of‑Vulnerability（PoV）测试。

**💡 创新点**

创新点包括：① 端到端的 agent‑driven PoV 测试生成流程；② 通过调用路径分析与多轮迭代反馈精细化测试；③ 利用 LLM（GPT）对构建/执行日志进行自动质量评估；④ 与现有 LLM 方案相比，大幅提升测试覆盖率与成功率。

**🔧 技术方法**

采用技术包括：AI 编码代理 Codex（及 Gemini、Mistral 作为对比）、GPT‑5.x 进行评估、调用路径分析提示模板、基于 Maven/Gradle 的自动构建与运行脚本、JSON/文本日志的结构化传递、以及 LLM 自评机制。

**📊 数据集**

使用了 33 个 Java 项目（29 Maven、3 Gradle、1 纯 Java），来自先前的 49 对 <App,Lib> 数据集，涵盖 20 个 CVE 及其对应的 29 个受影响库版本，攻击类型包括 DoS、DT、RCE 等。

**📈 对比分析**

与 Zhang 等人的 LLM‑based 方法以及 SIEGE、Transfer 等工具对比，本文在 33 对实例中实现了 96% 的入口点发现、>70% 的测试成功编译、>55% 的 PoV 演示率，并且 LLM 评估准确率达到 68%–79%，显著优于现有方法。

**⚠️ 局限性**

局限性包括：① 仅在可编译的 Java 环境下有效；② 依赖于示例测试的可用性；③ LLM 仍存在幻觉与评估误差，需人工复核；④ 数据集规模有限，难以覆盖所有漏洞场景；⑤ 对于非 Java 或更复杂的构建系统支持不足。

---

## 448. MOSAIC-Bench: Measuring Compositional Vulnerability Induction in Coding Agents

**arXiv ID:** 2605.03952 | [PDF](https://arxiv.org/pdf/2605.03952v1)

**作者:** Jonathan Steinberg `[一作]` (University of Haifa), Oren Gal `[通讯]` (University of Haifa)

**通讯引用:** 344 | [OpenAlex ID](https://openalex.org/A5030616543)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MOSAIC-Bench 基准，包含 199 条三阶段攻击链，并评估了多款生产级编码代理与代码审查代理在逐票实施与整体审核中的行为。

**💡 创新点**

创新点在于揭示“构成间隙”——单一安全评估无法捕捉通过无害票据序列合成的漏洞；并将漏洞真值与审查协议作为基准轴，提出将审查者角色改为渗透测试者的对抗式框架，显著降低逃逸率。

**🔧 技术方法**

技术手段包括：
1) 采用 Jira 样式三阶段票据，确保每张票据看似普通、全局可审；
2) 在 Docker 化的子系统上执行确定性 Python PoC 或acle 以验证漏洞；
3) 构建两层评估管线，分别检测逐票差异、审查者判定与最终漏洞可利用性；
4) 对编码代理采用单会话与续会话协议，对审查者采用中立与渗透测试者 framing 以及 diff‑only 与全代码库上下文两种输入模式。

**📊 数据集**

使用的数据集为 MOSAIC-Bench：199 条 oracle‑支持的 3‑stage 攻击链，涵盖 10 种 Web‑应用子系统、31 个 CWE 类、5 种编程语言；另外收集了 608 条真实 GitHub PR 用于评估误报率。

**📈 对比分析**

比较方法：
- 评估九款主流编码代理的 ASR（攻击成功率），发现 staged 方案下 53–86%；
- 在审查阶段对同一 diff 进行 neutral 与 pentester framing 的审查，记录 evasion；
- 对比 diff‑only 与 full‑context 两种上下文，衡量上下文对逃逸率的影响；
- 结果显示：
  * 单提示下 Claude 0–1.9%、Codex 9.3–20.4%；
  * staged 方案下 53–86%；
  * full‑context 下审查逃逸率 8.5–14.6%；
  * pentester framing 将逃逸率降至 3.0–17.6%，Gemma‑4‑E4B‑it 在 88.4% 检测率与 4.6% FP 之间取得平衡。

**⚠️ 局限性**

局限性包括：
- 子系统主要为 Web‑应用样板，未覆盖 Rust、.NET、移动、嵌入式、内核等领域；
- 采用合成凭证与局部容器，缺乏真实生产秘密；
- 仅对单票、单会话流程评估，未考虑更复杂的跨票工作流；
- oracle 作为唯一真值来源，若 PoC 与实际环境差异可能导致误判；
- 防御方案（如 pentester framing）在误报率与成本间存在折中，且在不同组织部署时需进一步验证。

---

## 449. Towards Open World Sound Event Detection

**arXiv ID:** 2605.03934 | [PDF](https://arxiv.org/pdf/2605.03934v1)

**作者:** P. H. Hai `[一作]`, L. H. Son `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了开放世界声音事件检测（OW-SED）框架，模型在已知事件上进行检测，同时识别并标记未知事件并支持增量学习。

**💡 创新点**

创新点包括：1）1D 可变形 Transformer 结构以适应音频的时间特性；2）特征分离机制，将事件表示拆分为类别特定与类别无关部分；3）两阶段训练策略，先一对多匹配再加入多样性损失，提升未知事件检出的多样性。

**🔧 技术方法**

使用 1D Deformable DETR、WOOT Transformer、特征分离模块、两阶段训练、事件性 (eventness) 估计、重放缓存等技术；训练采用 AdamW，采用多查询、位置编码和多头注意力。

**📊 数据集**

在 URBAN-SED 和 DESED 两个公开数据集上进行实验，分别划分为三阶段任务进行开放世界评估。

**📈 对比分析**

与闭域基线（CRNN、CTrans）以及开放世界方法（OW-DETR、SS OW-DETR、CAT、PROB）对比，WOOT 在已知事件的 F1 评分上保持竞争力，并在未知事件召回率上显著超越其他方法（例如 URBAN-SED Task 1 的 U-Recall 由 21.4% 提升至 28.6%、Task 2 由 27.7% 提升至 33.4%）。

**⚠️ 局限性**

局限性包括：对查询数量的敏感性（过多或过少均影响性能）；尚未在更大规模或多模态数据集上验证；缺乏自监督或对比学习的预训练，可能限制在新领域的迁移能力；在闭域任务中的 Segment/F1 性能仍略低于传统 CRNN/CTrans。

---

## 450. Optimal Posterior Sampling for Policy Identification in Tabular Markov Decision Processes

**arXiv ID:** 2605.03921 | [PDF](https://arxiv.org/pdf/2605.03921v1)

**作者:** Cyrille Kone `[一作]`, Kevin Jamieson `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于后验采样与在线学习的算法，用以在有限周期表格MDP中实现高置信度最佳策略识别。

**💡 创新点**

创新点在于通过构造“挑战者MDP”避免了传统的策略枚举和优化步骤，同时实现了实例依赖的最优样本复杂度与后验收敛率。

**🔧 技术方法**

采用后验采样、Top‑Two抽样、KL奖励惩罚与镜像下降在线学习器相结合的技术框架。

**📊 数据集**

在三种经典环境上进行实验：MOCA、RiverSwim和CombinationLock。

**📈 对比分析**

与MOCA‑UCBVI、PSRL、SSR等基线相比，算法在识别率和性能因子上均表现更好，尤其在MOCA环境中显著提升。

**⚠️ 局限性**

局限性包括对大规模状态空间的采样开销、缺乏非渐近（δ=O(1)）下的精确界限，以及对线性函数逼近或ε-近似最佳策略识别的扩展仍待研究。

---

## 451. Task-Aware Scanning Parameter Configuration for Robotic Inspection Using Vision Language Embeddings and Hyperdimensional Computing

**arXiv ID:** 2605.03909 | [PDF](https://arxiv.org/pdf/2605.03909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 452. Steer Like the LLM: Activation Steering that Mimics Prompting

**arXiv ID:** 2605.03907 | [PDF](https://arxiv.org/pdf/2605.03907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 453. Contextual Multi-Objective Optimization: Rethinking Objectives in Frontier AI Systems

**arXiv ID:** 2605.03900 | [PDF](https://arxiv.org/pdf/2605.03900v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 454. Spatiotemporal Convolutions on EEG signal -- A Representation Learning Perspective on Efficient and Explainable EEG Classification with Convolutional Neural Nets

**arXiv ID:** 2605.03874 | [PDF](https://arxiv.org/pdf/2605.03874v1)

**作者:** Laurits Dixen `[一作]` (IT University of Copenhagen), Paolo Burelli `[通讯]` (IT University of Copenhagen)

**通讯引用:** 1102 | [OpenAlex ID](https://openalex.org/A5087367222)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

比较了1D与2D卷积在EEG信号分类中的效果，并通过特征重建、相关性分析和RSA探讨内部表征差异。

**💡 创新点**

提出将连续的1D时空卷积替换为单层2D时空卷积，以降低计算复杂度并保持相同性能。

**🔧 技术方法**

使用ShallowFBCSPNet、EEG Conformer及其2D版本，结合特征重建、相关性分析和RSA等技术。

**📊 数据集**

使用MOABB提供的BNCI2014-001（22通道四分类）和BNCI2014-004（3通道二分类）脑机接口数据集。

**📈 对比分析**

通过5折交叉验证评估准确率、训练时间、迭代次数和推理时间，结果显示2D卷积在多通道任务中显著加速，性能保持不变。

**⚠️ 局限性**

实验仅覆盖两组相似任务，未测试更大模型或不同任务，实验范围有限。

---

## 455. Surviving the Edge: Federated Learning under Networking and Resource Constraints

**arXiv ID:** 2605.03870 | [PDF](https://arxiv.org/pdf/2605.03870v1)

**作者:** Mike Mwanje `[一作]` (Carnegie Mellon University), Joao Barros `[通讯]` (Carnegie Mellon University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在非洲资源受限的边缘网络环境下，系统性评估并定位联邦学习（FL）在高延迟、丢包与客户端失败下的性能瓶颈；

**💡 创新点**

首次揭示FL突发-空闲通信模式与TCP默认参数不匹配导致的失败，并通过仅三项TCP参数调优（tcp_syn_retries、tcp_keepalive_time、tcp_keepalive_intvl）恢复极端条件下的训练；

**🔧 技术方法**

使用Flower框架、Kubernetes容器化环境、Linux NetEm和Chaos Mesh进行实验，并对TCP参数进行微调；

**📊 数据集**

以MNIST数据集为实验基准；

**📈 对比分析**

通过与默认配置对比，发现延迟>5s、丢包>50%或客户端失败>90%导致训练失败，调优后训练时间显著下降，准确率基本保持不变；

**⚠️ 局限性**

实验仅覆盖Flower框架，未检验其他FL实现；仅使用MNIST，缺乏对更复杂任务的验证；未实现动态自适应TCP调优机制，且对功耗、能效等指标未做评估。

---

## 456. On Adaptivity in Zeroth-Order Optimization

**arXiv ID:** 2605.03869 | [PDF](https://arxiv.org/pdf/2605.03869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 457. Ecologically-Constrained Task Arithmetic for Multi-Taxa Bioacoustic Classifiers Without Shared Data

**arXiv ID:** 2605.03914 | [PDF](https://arxiv.org/pdf/2605.03914v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 458. Quantifying the human visual exposome with vision language models

**arXiv ID:** 2605.03863 | [PDF](https://arxiv.org/pdf/2605.03863v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 459. Atomic Fact-Checking Increases Clinician Trust in Large Language Model Recommendations for Oncology Decision Support: A Randomized Controlled Trial

**arXiv ID:** 2605.03916 | [PDF](https://arxiv.org/pdf/2605.03916v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 460. Logical Consistency as a Bridge: Improving LLM Hallucination Detection via Label Constraint Modeling between Responses and Self-Judgments

**arXiv ID:** 2605.03971 | [PDF](https://arxiv.org/pdf/2605.03971v1)

**作者:** Hao Mi `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Juan Cao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LaaB框架，将LLM的微观内在模式与宏观自评符号化判断通过逻辑一致性桥接，实现幻觉检测。

**💡 创新点**

创新点在于将自评“meta-judgment”映射回特征空间，并通过逻辑约束实现双视角互学习，提升检测准确率。

**🔧 技术方法**

使用内在模式提取（隐藏状态、logits、注意力）和自评生成，构建MLP分类器，并引入Huber逻辑损失与互学习。

**📊 数据集**

在四个公开数据集（TriviaQA、MMLU、NQ_Open、HaluEval）上测试，并采用四款LLM（Llama‑3.1‑8B、70B，Qwen‑2.5‑32B，Mistral‑7B）。

**📈 对比分析**

与8个基线对比，LaaB在宏观F1与准确率上平均提升5–8个百分点，且跨数据集泛化显著。

**⚠️ 局限性**

局限性包括需访问LLM内部信息、对“Yes/No”限制可能引入噪声、无法纠正两视角都错的样本且逻辑一致性为软约束。

---

## 461. The Counterexample Game: Iterated Conceptual Analysis and Repair in Language Models

**arXiv ID:** 2605.03936 | [PDF](https://arxiv.org/pdf/2605.03936v1)

**作者:** Daniel Drucker `[一作]` (University of Texas at Austin), Kyle Mahowald `[通讯]` (University of Texas at Austin)

**通讯引用:** 3893 | [OpenAlex ID](https://openalex.org/A5039468724)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过让语言模型在迭代的反例‑修复循环中进行概念分析，检验其哲学推理能力。

**💡 创新点**

创新点在于将传统的哲学“反例游戏”转化为模型自我迭代的 actor‑critic 过程，并对模型生成的反例与修正进行系统评估。

**🔧 技术方法**

使用了多模型（Claude Opus 4.5、GPT‑5、Gemini 2.0 Flash 等）以及自我对话的“反例生成‑定义修复”链条，并结合人类评审和自动判定。

**📊 数据集**

数据集包含20个概念（10名词、10动词）的种子定义，随后产生约 1,200–6,000 条反例‑修复循环，覆盖多轮迭代。

**📈 对比分析**

对比方法包括人类评审与 LM 判定的一致性、定义准确度与简洁度随迭代变化的趋势；结果显示 LM 的反例有效率随迭代下降，定义长度增加但准确度不升，整体表现仅略优于人类平均水平。

**⚠️ 局限性**

局限性包括：模型难以持续改进反例与定义、产生冗长但无实质性提升的定义、低质量反例导致修正停滞，以及概念间差异导致性能不一。

---

## 462. Pretrained Model Representations as Acquisition Signals for Active Learning of MLIPs

**arXiv ID:** 2605.03964 | [PDF](https://arxiv.org/pdf/2605.03964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 463. Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators

**arXiv ID:** 2605.03969 | [PDF](https://arxiv.org/pdf/2605.03969v1)

**作者:** Mohamed Mady `[一作]` (Technical University of Munich), Björn Schuller `[通讯]` (Imperial College London)

**通讯引用:** 55021 | [OpenAlex ID](https://openalex.org/A5043060302)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练 Transformer 监督检测器以识别 AI 生成文本，并在训练后仅用一次阈值校准后在多域、多生成器和外部数据集上直接评估，展示分布漂移下的真实鲁棒性；

**💡 创新点**

提出在 Transformer 表示上融合手工语言特征的注意力融合机制（FeatAttn），以及采用 DeBERTa‑v3 背骨和单阈值部署协议，能显著提升跨域迁移性能；

**🔧 技术方法**

使用 BERT、RoBERTa、DeBERTa‑v3 作为编码器，配合 30 维手工特征（可读性、词汇多样性、POS、句法等）与自适应注意力融合，并通过交叉熵训练；

**📊 数据集**

基准数据集包括 HC3‑PLUS（含语义不变重写）、M4（多域多生成器）以及 AI‑Text‑Detection‑Pile；

**📈 对比分析**

与 BERT/RoBERTa 传统基线以及 Fast‑DetectGPT、RADAR、Log‑Rank 等零样本方法进行对比；在 HC3‑PLUS 上几乎达 99% 的 BA，在 M4 上 85.9% BA（人类召回 81.3%，AI 召回 90.5%），远超零样本基线（+7.22 点），并在多种随机种子下保持稳定；

**⚠️ 局限性**

局限性包括：在某些生成器（如 Gemini 3.0 Pro）召回下降，手工特征选择可能依赖训练集；固定阈值虽然符合部署需求，但对不同目标分布的泛化仍受限，且过度依赖特征可能导致对人工编辑文本的误判。

---

## 464. Inconsistent Databases and Argumentation Frameworks with Collective Attacks

**arXiv ID:** 2605.03954 | [PDF](https://arxiv.org/pdf/2605.03954v1)

**作者:** Yasir Mahmood `[一作]` (Paderborn University), Axel-Cyrille Ngonga Ngomo `[通讯]` (Paderborn University)

**通讯引用:** 9051 | [OpenAlex ID](https://openalex.org/A5038745720)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

本文研究了在包含否定约束（DC）和局部视图产生依赖（LTGD）的不一致关系数据库中，子集最大修复与抽象论证框架（SETAFs）扩展之间的对应关系，并给出了完整的复杂度分析。

**💡 创新点**

创新点在于首次将LTGD与DC同时纳入SETAF的翻译，证明在两类约束共同作用时修复对应唯一的preferred扩展；并在特殊子类（FD、ID）下进一步将模型转化为普通AF，保持唯一性与可预处理性。

**🔧 技术方法**

采用了约束翻译技术（冲突图、支持集）、SETAF/AF构造、预处理消除自攻击辅助论点以及复杂度归约与证明方法。

**📊 数据集**

主要使用理论示例（如员工、部门、订单表等）和合成数据库进行证明，未涉及公开大规模数据集。

**📈 对比分析**

通过理论复杂度分析，证明存在性、勇敢（credulous）与谨慎（skeptical）推理问题在数据复杂度下分别为P、NP/PSPACE完整；相较传统修复算法，提供了多项式预处理的理论最优性。

**⚠️ 局限性**

局限性包括仅处理子集删除型修复，未覆盖生成型修复或优先级约束；LTGDs翻译依赖自攻击辅助论点，模型直观性受限；更强的TGD或对称差异修复仍未研究。

---

## 465. HELO Cryptography: A Lightweight Cryptographic System for Enhancing IoT Security in P2P Data Transmission

**arXiv ID:** 2605.03948 | [PDF](https://arxiv.org/pdf/2605.03948v1)

**作者:** Tahsin Ahmed `[一作]` (BRAC University), Muhammad Iqbal Hossain `[通讯]` (BRAC University)

**通讯引用:** 562 | [OpenAlex ID](https://openalex.org/A5100454780)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文设计并实现了一套名为HELO的轻量级加密系统，旨在提升物联网设备的数据安全性；

**💡 创新点**

创新点在于将ChaCha20-Poly1305 AEAD、ECDH+ECDSA密钥交换与签名、一次性随机数、数据分块与多线程处理相结合，兼顾高安全性与低资源占用；

**🔧 技术方法**

主要技术包括ChaCha20-Poly1305加密、SHA3‑256哈希、SECP256R1椭圆曲线的ECDH/ECDSA、Python多线程与文件分块、以及与AES、Blowfish、Fernet等传统算法的对标；

**📊 数据集**

实验使用了Kaggle公开数据集，涵盖TXT、CSV以及JPG/PNG图片文件，尺寸从几百字节到数十兆字节不等；

**📈 对比分析**

通过CPU耗时、内存占用、运行时间、能耗和雪崩效应等指标与AES、Blowfish、Fernet比较，HELO在多数小文件上速度最快、能耗最低，对大文件的多线程开销略显不足；

**⚠️ 局限性**

局限性包括实验仅在桌面PC上进行，未在真实低功耗IoT硬件上验证；能耗测算基于软件层面估算；多线程处理在大文件时可能产生额外开销。

---

## 466. TabSurv: Adapting Modern Tabular Neural Networks to Survival Analysis

**arXiv ID:** 2605.03944 | [PDF](https://arxiv.org/pdf/2605.03944v1)

**作者:** Stanislav Kirpichenko `[一作]` (Higher School of Artificial Intelligence Technologies), Lev Utkin `[通讯]` (Higher School of Artificial Intelligence Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出TabSurv框架，将现代表格深度学习模型（如MLP、RealMLP、TabM）与生存分析任务相结合，支持参数化（Weibull）和非参数化预测，并引入SurvHL直方图损失；实现并行深度集成训练；提供完整可公开的实现。

**💡 创新点**

① 模块化、可插拔的设计，可在任意表格backbone与输出头间自由组合；② 针对删失数据的SurvHL损失，用加权平滑的直方图实现更稳定的梯度；③ 并行深度集成策略，先分别优化分布参数后平均，提升模型多样性和鲁棒性。

**🔧 技术方法**

使用多层感知机（MLP）及其变体、Weibull分布参数化、非参数化 logits、SurvHL损失、深度集成；通过Optuna TPE进行超参搜索；评估指标包括Harrell C-index、Integrated Brier Score（IBS）以及时间相关的AUC。

**📊 数据集**

10个真实世界生存数据集（样本量均≥500），包含临床、金融等多领域。

**📈 对比分析**

在C-index、IBS和时间相关AUC上与RSF、DeepSurv、DeepHit、SurvTRACE等基线比较；TabSurv的WAS、LAS和LS(MLP)变体平均排名位列首位；WAS在C-index与AUC上最高，LS(MLP)在IBS上表现最佳；RealMLP作为backbone在此任务中效果不如标准MLP。

**⚠️ 局限性**

对非多模态生存时间的Weibull假设在真正多峰分布下可能欠拟合；集成方案虽提升性能但计算成本较高；目前仅支持Weibull和非参数化输出，需扩展其他分布；缺乏竞争风险、多任务和预训练表格模型的直接集成。

---

## 467. Beyond Rules: LLM-Powered Linting for Quantum Programs

**arXiv ID:** 2605.03943 | [PDF](https://arxiv.org/pdf/2605.03943v1)

**作者:** Pietro Cassieri `[一作]` (University of Salerno), Domenico Bianculli `[通讯]` (University of Luxembourg)

**通讯引用:** 1781 | [OpenAlex ID](https://openalex.org/A5038017715)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并评估两种基于大语言模型的量子程序 linting 方法：多提示链式思考（CoT）管道以及加入检索增强生成（RAG）的变体。

**💡 创新点**

创新点在于：①将 CoT 多提示策略应用于量子代码审计；②构建仅包含人工验证真阳性实例的知识库并通过 RAG 让模型在推理时使用一例学习；③在传统规则基础的 LintQ 上实现显著性能提升。

**🔧 技术方法**

核心技术包括 GPT‑5 大语言模型、Chain‑of‑Thought 提示工程、Retrieval‑Augmented Generation（基于 FAISS 的向量检索）、手工标注的量子程序真阳性数据、以及 Qiskit 代码的静态分析。

**📊 数据集**

数据集为 55 篇 Qiskit 程序（43 实际文件 + 12 注入式合成文件），每篇文件被人工标注为 10 类量子编程问题的 Ground Truth。

**📈 对比分析**

与规则式 linters（LintQ）进行对比评估，采用 precision、recall、F1‑score 三项指标。结果显示：CoT 方案 F1‑score 0.70、RAG 方案 0.68，均远超 LintQ（0.41）；RAG 在 precision 上略优（0.56 vs 0.55），但 recall 略低（0.86 vs 0.96）。

**⚠️ 局限性**

局限性包括：①依赖单一 GPT‑5 模型，模型不确定性和泛化能力未知；②人工标注为单人完成，可能产生主观偏差；③RAG 对代码结构变异敏感，易受代码混淆影响；④实验仅针对 Qiskit，无法直接推广至 Cirq、Q# 等其他量子语言；⑤缺乏对不同 LLM 或嵌入模型的鲁棒性评估。

---

## 468. MiniMind-O Technical Report: An Open Small-Scale Speech-Native Omni Model

**arXiv ID:** 2605.03937 | [PDF](https://arxiv.org/pdf/2605.03937v1)

**作者:** Jingyao Gong `[一作]` `[通讯]` (Independent Researcher), Jingyao Gong (Independent Researcher)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个0.1B规模的开放多模态模型（MiniOmni‑0.1B），支持文本、语音、图像输入和流式语音输出，并公开了完整的模型代码、检查点与Parquet训练数据；

**💡 创新点**

在小规模模型上验证并公开了中间层语义桥接、公开多模态序列格式和低秩8码本接口这三大设计选择，并展示了可复现的完整交互循环；

**🔧 技术方法**

采用MiniMind Transformer作为Thinker，四层Talker预测Mimi音频码本，使用冻结的SenseVoice‑Small与SigLIP2编码器，并通过两层MLP投影与低秩嵌入/头部适配器实现跨模态投射；

**📊 数据集**

公开了T2A、I2T、A2A三套Parquet格式数据集，包含文本、图像字节、语音输入、Mimi码本目标、参考码提示及说话人嵌入；

**📈 对比分析**

通过Thinker‑Talker一致性评估（CER≈0.09）和说话人相似度（CAM++余弦≈0.60）对比稠密与MoE变体，在短答案场景下参数约为Mini‑Omni/Omni2的五分之一；

**⚠️ 局限性**

语音自然度和长文本稳定性不足，视觉路径受限于冻结SigLIP2与简单MLP，克隆效果高度依赖参考质量，评估仅关注一致性，不涵盖MOS、延迟或鲁棒性等方面。

---

## 469. PHALAR: Phasors for Learned Musical Audio Representations

**arXiv ID:** 2605.03929 | [PDF](https://arxiv.org/pdf/2605.03929v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 470. QKVShare: Quantized KV-Cache Handoff for Multi-Agent On-Device LLMs

**arXiv ID:** 2605.03884 | [PDF](https://arxiv.org/pdf/2605.03884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 471. From Data Lifting to Continuous Risk Estimation: A Process-Aware Pipeline for Predictive Monitoring of Clinical Pathways

**arXiv ID:** 2605.03895 | [PDF](https://arxiv.org/pdf/2605.03895v1)

**作者:** Pasquale Ardimento `[一作]` (University of Bari), Samuele Latorre `[通讯]` (University of Bari)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个可复现、面向过程的预测监控流水线，将临床记录提升为时间一致的事件日志，并基于前缀表示实现对COVID-19患者ICU入院风险的连续预测。

**💡 创新点**

将数据提升、时间重构、事件日志构建与前缀特征抽取整合为统一的端到端流水线，并强调持续更新的过程感知预测能力。

**🔧 技术方法**

数据提升、时间校准、XES事件日志生成、前缀特征提取、Logistic Regression/Random Forest等传统机器学习。

**📊 数据集**

COVID Data for Shared Learning（CDSL）临床数据集，包含4479例COVID-19患者，共46,804个前缀。

**📈 对比分析**

通过病例级别划分训练/测试集，评估AUC、F1等指标；Logistic Regression在测试集上AUC 0.906，F1 0.835，且随着前缀长度增加AUC从0.642提升至0.942。

**⚠️ 局限性**

使用表格前缀特征而非序列模型，可能丢失细粒度时间依赖；仅在单一COVID-19数据集上验证，缺乏跨领域泛化；未提供临床可解释性分析。

---

## 472. MCJudgeBench: A Benchmark for Constraint-Level Judge Evaluation in Multi-Constraint Instruction Following

**arXiv ID:** 2605.03858 | [PDF](https://arxiv.org/pdf/2605.03858v1)

**作者:** Jaeyun Lee `[一作]` (University of Oxford), Ronald Clark `[通讯]` (University of Oxford)

**通讯引用:** 5986 | [OpenAlex ID](https://openalex.org/A5054998594)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MCJudgeBench benchmark，评估多约束指令遵循时 LLM 判断器在每条约束上的判断准确性与稳定性，包含约束列表、每约束 gold 标注以及响应与提示的可变形集。

**💡 创新点**

① 引入约束级别的评估与多维度可靠性指标；② 设计响应与提示的可变形测试，探究判定器在不变形情况下的一致性；③ 将正确率、内在不一致性、程序不一致性等指标结合，系统性地分析 LLM 判断器的可靠性。

**🔧 技术方法**

使用 LLM-as-a-judge 框架，利用 Qwen3-4B-Instruct 生成候选响应，GPT-5 mini 辅助生成标签与扰动；通过统计测度 CJAR、macro‑F1、CIR_intr、CIR_prompt、CIR_resp 等对判定器进行评估。

**📊 数据集**

以 ComplexBench 与 InFoBench 为源构建 141 条指令、653 条约束，分 easy/hard/complex 三个 split；通过人工验证获得 201 个符合标注的响应扰动。

**📈 对比分析**

对比多家专有 LLM（GPT‑5.2、Claude Sonnet 4.6、Gemini 3.1 Pro 等）与开源 LLM（Qwen3.5‑4B、Llama 3.2 3B）在 CJAR、macro‑F1 以及各类不一致性指标上的表现；结果显示专有模型在整体准确率上优于开源，但开源模型在内在不一致性上差距显著；推理功能提升 macro‑F1 但并未统一降低不一致性。

**⚠️ 局限性**

仅适用于英文；仅覆盖 ComplexBench 与 InFoBench，未扩展到其他语言或指令分布；扰动设置有限，未覆盖所有可能的不稳定源；部分约束难以区分 partial 与 yes/no，存在主观性。

---

## 473. Parallel Reachability and Shortest Paths on Non-sparse Digraphs: Near-linear Work and Sub-square-root Depth

**arXiv ID:** 2605.03892 | [PDF](https://arxiv.org/pdf/2605.03892v1)

**作者:** Vikrant Ashvinkumar `[一作]` (Rutgers), Thatchaphol Saranurak `[通讯]` (University of Michigan)

**通讯引用:** 1100 | [OpenAlex ID](https://openalex.org/A5010547647)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了用于计算有向图中单源可达性和最短路径的并行算法，工作量接近线性，深度为o(√(n))，当m≥n^(1+o(1))时表现良好。

**💡 创新点**

在所有密度情况下，深度显著低于现有的最佳算法，特别是在m=Ω(n^2)时，达到n^0.136和n^0.25+o(1)的深度。

**🔧 技术方法**

使用了近线性时间的随机算法和快速矩阵乘法技术。

**📊 数据集**

在无权有向图上进行实验，特别关注于多种图的密度情况。

**📈 对比分析**

与现有的几种算法进行比较，显示出在相同工作量下，深度显著降低，尤其在稠密图中表现出色。

**⚠️ 局限性**

在稀疏图情况下，算法的性能没有显著提升，仍然存在深度为n^(1/2)+o(1)的限制。

---

## 474. Label-Efficient School Detection from Aerial Imagery via Weakly Supervised Pretraining and Fine-Tuning

**arXiv ID:** 2605.03968 | [PDF](https://arxiv.org/pdf/2605.03968v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 475. Randomized and Diverse Input State Generation for Quantum Program Testing

**arXiv ID:** 2605.03957 | [PDF](https://arxiv.org/pdf/2605.03957v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 476. A Benchmark for Interactive World Models with a Unified Action Generation Framework

**arXiv ID:** 2605.03941 | [PDF](https://arxiv.org/pdf/2605.03941v1)

**作者:** Jianjie Fang `[一作]` (Tsinghua University), Yong Li `[通讯]` (Tsinghua University)

**通讯引用:** 38381 | [OpenAlex ID](https://openalex.org/A5100355277)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向交互式世界模型的统一评测基准 iWorld-Bench，包含多视角、多天气、多场景的大规模视频数据集和六种交互任务。

**💡 创新点**

提出统一动作生成框架，将不同模型的动作表示映射到统一的81种基本动作；同时设计多维评估指标和记忆任务，弥补现有基准的不足。

**🔧 技术方法**

采用视频生成模型、视图控制、姿态编码、视觉语言模型（VLM）以及多模态动作映射等技术。

**📊 数据集**

基于12个公开高质量数据集（如 KITTI、Waymo、RealEstate‑10K、TUM‑RGB‑D、SpatialVid 等）和18个模拟器自采样，最终生成 330k 条视频，评测子集 2100 条。

**📈 对比分析**

通过对 14 种现有交互式世界模型（文本控制、one‑hot、相机参数）进行统一评测，使用 9 项指标（图像质量、轨迹跟随、记忆等），结果显示 one‑hot 模型在交互能力上领先，文本模型在生成质量上优越，AC3D 在相机参数控制上表现最佳。

**⚠️ 局限性**

仍存在模型在可控性与生成质量之间的权衡、对长时序一致性的不足，以及对实时性能的评估缺失。

---

## 477. Tree transducers of linear size-to-height increase (and the additive conjunction of linear logic)

**arXiv ID:** 2605.03928 | [PDF](https://arxiv.org/pdf/2605.03928v1)

**作者:** Luc Dartois `[一作]` (Université Marie et Louis Pasteur), Charles Peyrat `[通讯]` (Université Paris-Est Créteil)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究了树到树 Hennie 机器（Tree‑to‑Tree Hennie Machines, TTHM）的表达能力，证明它们既能计算所有线性大小到高度增长（LSHI）函数，又能实现更大类的函数，位于 LSHI MTT 与集合解释（set interpretations）之间。

**💡 创新点**

创新点在于：
• 提出了 TTHM 这一新型树转发器模型，并证明其严格包含 LSHI MTT，且严格小于集合解释。
• 证明了该类函数在线性大小/高度增长下的闭包性质（对 MTT 的预后置、后置组合保持闭包），从而得到完整的 LSHI 组合层次。
• 给出了基于线性类型 λ‑演算的等价表征，并通过游戏语义/几何相互作用证明等价，开创了树结构下的 λ‑演算与自动机之间的新桥接。

**🔧 技术方法**

使用了多种技术：
• 树走访机器与 Hennie 机器的定义与约束（bounded‑visit、narrow‑visit、weight‑reducing）。
• 通过宏树转发器（MTT）与树走访转发器（TWT）构造将 MTT 编译成 TTHM。
• 采用集合解释（MSO set interpretations）来对 TTHM 进行语义化。
• 通过 actor 模型和游戏语义实现 λ‑演算到 TTHM 的翻译。
• 证明闭包性、组合层次严格性、正则性反射等结论。

**📊 数据集**

该工作为纯理论研究，未使用任何实验数据集；所有结论均为理论证明。

**📈 对比分析**

比较方法：与现有模型（宏树转发器、集合解释、MSO 转换、树走访转发器、Peeble 机器等）在表达能力上进行层级比较。性能方面通过理论上给出的线性大小/高度增长上界及闭包性质评估；未给出时间复杂度或运行时间的实验评估。

**⚠️ 局限性**

局限性：
• 仅在决定性、完整性等条件下给出等价性，非确定性或部分定义的情况仍未完整处理。
• 对于更一般的非线性大小/高度增长的函数类，TTHM 的表达能力尚不清楚。
• 现有证明依赖于复杂的几何相互作用和游戏语义，难以直接推广到更高阶或更一般的模型。
• 未提供算法实现或效率评估，无法直接用于实际程序转换或优化。

---

## 478. StateVLM: A State-Aware Vision-Language Model for Robotic Affordance Reasoning

**arXiv ID:** 2605.03927 | [PDF](https://arxiv.org/pdf/2605.03927v1)

**作者:** Xiaowen Sun `[一作]` (University of Hamburg), Stefan Wermter `[通讯]` (University of Hamburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了 StateVLM 模型，用于对象检测、对象状态定位和把握区域预测。

**💡 创新点**

创新点：①在训练阶段通过 Box Decoder 计算辅助回归损失（ARL）来监督连续坐标，保持推理时仅使用序列预测；②构建了首个面向对象状态和抓取推理的公开基准 OSAR。

**🔧 技术方法**

技术实现：基于 MiniCPM-V + Qwen2-7B 的 VLM；使用 Box Decoder + MLP 提取连续坐标；损失组合 CLM + ARL（L1 + GIoU）；LoRA 参数高效微调；采用 DeepSpeed 分布式训练。

**📊 数据集**

使用数据集：公开 RefCOCO、RefCOCO+、RefCOCOg；自建 OSAR（1,172 场景、7,746 对象、25,401 语言表达）。

**📈 对比分析**

比较方法：在 RefCOCO 等基准上对比仅 CLM 与 CLM+ARL，平均提升 1.6%；在 OSAR 上，ARL+LoRA 在对象检测与抓取推理任务中分别达 60.2%，相较无 ARL 提升 5.2% 或更多，异常率降至 0，显示显著性能提升。

**⚠️ 局限性**

局限性：OSAR 数据集规模有限且主要依赖生成图像，缺乏多样性和真实场景；模型对模糊或歧义指令评估不足；对某些细粒度类别（如刀具）定位仍不理想；仅验证了一种 VLM 架构，未探讨更大模型或更复杂多模态融合。

---

## 479. Demographic Divides in Political Content Exposure on Facebook

**arXiv ID:** 2605.03962 | [PDF](https://arxiv.org/pdf/2605.03962v1)

**作者:** S M Mehedi Zaman `[一作]` (Rutgers University), Kiran Garimella `[通讯]` (Rutgers University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过基于用户数据捐赠的方式，构建了覆盖2012–2023年美国1,100多名用户的全公共页面和群组帖子数据，系统评估了Facebook上政治内容的曝光比例、跨人群差异、政治倾向以及平台算法更新对曝光的影响。

**💡 创新点**

①首次在大规模样本下量化潜在信息曝光而非仅关注交互；②引入双层分类（页面级+帖子级）精准捕捉跨界政治内容；③利用大语言模型进行政治标签与意识形态定位，并通过贝叶斯结构时间序列模型评估平台干预因果效应。

**🔧 技术方法**

- Facebook Graph API 与 CrowdTangle 爬取公开内容；- GPT‑5 对页面标题进行四分类；- LLaMA‑3.3 70B Instruct 对帖子进行政治/非政治与左右倾向标注；- 贝叶斯结构时间序列模型（BSTS）估计算法更新对曝光的时间动态效应；- 重新加权技术实现样本对总体的代表性。

**📊 数据集**

包含约193.4 M条帖子（英/西语187 M，其他约6.5 M），来自12,896个非政治页面、458个政治页面、1,051个新闻页面、2,648个其他页面，覆盖1,115名美国用户的公共关注列表。

**📈 对比分析**

研究主要以描述性统计和贝叶斯因果推断为主，未与传统模型直接对比；但通过分层可视化和置信区间展示不同干预（2016年好友优先、2018年有意义互动、2021年新闻降权）对政治曝光比例与量的影响，显示2018年更新在比例上显著提升，而2021年仅在量上略减。

**⚠️ 局限性**

1) 只测量潜在曝光，未反映算法排序后的实际可见内容；2) 仅分析文本与元数据，忽略图片/视频政治信号；3) 样本为便利抽样，需重加权；4) BSTS因果估计受未观测冲击、政策叠加等假设限制，可能存在残余混杂。

---

## 480. CC-OCR V2: Benchmarking Large Multimodal Models for Literacy in Real-world Document Processing

**arXiv ID:** 2605.03903 | [PDF](https://arxiv.org/pdf/2605.03903v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 481. EvoLM: Self-Evolving Language Models through Co-Evolved Discriminative Rubrics

**arXiv ID:** 2605.03871 | [PDF](https://arxiv.org/pdf/2605.03871v1)

**作者:** Shuyue Stella Li `[一作]` (University of Washington), Yulia Tsvetkov `[通讯]` (University of Washington)

**通讯引用:** 5310 | [OpenAlex ID](https://openalex.org/A5062910836)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用语言模型自身的评估知识，训练一个可生成自然语言 rubric 的模型，并将这些 rubric 作为奖励信号共同训练策略，实现完全无人工标注的自演化后训练方法。

**💡 创新点**

创新点在于将 rubric 视为可学习的判别变量，使用变分推断最大化“判别效用”来训练 rubric generator，并通过自我对比（temporal contrast）生成 preference pairs，无需外部评估器或人类注释。

**🔧 技术方法**

技术包括：GRPO 强化学习、变分推断+policy‑gradient 训练 rubric generator、冻结的 1.7B judge、margin + format 奖励设计、以及多 judge 集成训练。

**📊 数据集**

使用的数据集：Tulu 3 preference mixture（约271k prompts，覆盖聊天、指令、数学、代码、知识等领域），RewardBench‑2、JudgeBench 作为评价基准，OLMo3‑Adapt 12‑benchmark suite 用于最终策略评估。

**📈 对比分析**

与 GPT‑4.1/ Qwen3‑8B prompt rubrics、Skywork‑RM scalar reward、RaR、RRD、RLCER、Rubric‑ARM 等基线比较，Co‑evolving 方法在 OLMo3‑Adapt 上平均 69.3% 分数，显著优于所有基线；Rubric generator 在 RewardBench‑2 上 46% 但能产生更强的策略；多 judge 训练进一步提升跨评判器泛化。

**⚠️ 局限性**

局限性：仅在通用任务上验证，专业化领域（医学、法律等）效果未知；对纯主观评价的鲁棒性不明；冻结 judge 限制了可学习 rubric 的复杂度；在完全无标签的极端领域仍需改进。

---

## 482. Optimal Hardness of Online Algorithms for Large Common Induced Subgraphs

**arXiv ID:** 2605.03893 | [PDF](https://arxiv.org/pdf/2605.03893v1)

**作者:** David Gamarnik `[一作]` (Massachusetts Institute of Technology), Gabe Schoenbach `[通讯]` (University of Chicago)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5062859081)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在两张独立 Erdős–Rényi 随机图 G_1, G_2 ∼ 𝔾(n,½) 上寻找最大公共诱导子图的问题，分析了在线算法的性能，并证明了在线算法的计算-优化缺口。

**💡 创新点**

创新点在于：①给出了一个简单的贪心在线算法，证明其在高概率下能获得 (2‑o(1)) log₂ n 的解；②利用重叠间隙性质（OGP）和插值路径技术，证明任何在线算法都无法在高概率下获得 (2+ε) log₂ n 的解，从而首次揭示该问题在平均情况上存在明显的在线算法限制。

**🔧 技术方法**

核心技术包括：贪心在线算法设计、概率分析（如使用二项分布尾估计）、重叠间隙性质（OGP）的证明、以及 Gamarnik‑Kızıldağ‑Warnke 的插值路径与 Jensen 不等式结合的框架。

**📊 数据集**

数据集为理论上的随机图模型 G(n,½)，并未使用真实数据集，所有实验/证明均在概率上完成。

**📈 对比分析**

与传统的离线全局搜索或启发式算法相比，贪心在线算法仅获得最优解的一半（约 50%），但由于在线算法的实用性，其性能已是最优的；证明表明任何在线算法都无法突破该界限。

**⚠️ 局限性**

局限性包括：仅在密集随机图（p=½）下证明，未扩展到不同边概率或稀疏图；仅针对在线算法，不涉及随机化离线算法或其他算法类别；并未给出实际实现细节或实验验证。

---

## 483. Memory-Efficient Continual Learning with CLIP Models

**arXiv ID:** 2605.03866 | [PDF](https://arxiv.org/pdf/2605.03866v1)

**作者:** Ryan King `[一作]`, Tianbao Yang `[通讯]` (Texas A&M University)

**通讯引用:** 6115 | [OpenAlex ID](https://openalex.org/A5023288846)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出利用双模态对比学习和分布式鲁棒优化实现CLIP模型的增量学习，避免灾难性遗忘并在有限内存下保持高性能；

**💡 创新点**

创新点在于将标签嵌入与图像共同投影到对比空间，并通过动态类权重的DRO方法在内存受限时实现对不平衡分布的自适应调节；

**🔧 技术方法**

主要技术包括CLIP预训练模型、全局对比损失（GCL）、分布式鲁棒优化（GDRO）与移动平均梯度估计；

**📊 数据集**

使用了CIFAR-100、ImageNet1K的类别增量数据集以及DomainNet的域增量数据集；

**📈 对比分析**

与EWC、DER、iCaRL、Co2L、FOSTER等传统CL方法及CaSSLe对比，实验表明在不同内存规模下均显著优于基线，GCL在大内存时表现最好，GDRO在小内存时更稳健；

**⚠️ 局限性**

局限性包括对预训练CLIP的依赖、对超参数（γ、λ等）的敏感性以及仅在图像任务上验证，缺乏对更复杂多模态或非视觉领域的推广。

---

## 484. A Closed-Form Adaptive-Landmark Kernel for Certified Point-Cloud and Graph Classification

**arXiv ID:** 2605.04046 | [PDF](https://arxiv.org/pdf/2605.04046v1)

**作者:** Sushovan Majhi `[一作]` (George Washington University), Pramita Bagchi `[通讯]` (George Washington University)

**通讯引用:** 450 | [OpenAlex ID](https://openalex.org/A5036075177)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

该论文提出了一种自适应地标点（landmark）聚合式持久性图（persistence diagram）向量化方法（PALACE），并给出了覆盖证书、误差上界和特征选择统计量；

**💡 创新点**

创新点在于：①通过“τ-可容忍”覆盖（landmark cover）提供持久性图向量化的显式下界和结构性正则化；②使用可加高斯地标核（additive landmark Gaussian kernel）实现非线性升维；③给出无交叉验证的样本级预测证书；④设计了闭式数据驱动的选择统计量（如γ̂/√K、ρ̂_Mah），实现不同配置的可比评估；

**🔧 技术方法**

主要技术包括：持久性图到测度空间的线性求和嵌入、可加地标核的RKHS升维、最大间距SVM、MMD、马尔科夫蒙特卡洛（Ledoit–Wolf Shrunk covariance）、高斯核与Pinelis Hoeffding 以及Le Cam两点极小化。

**📊 数据集**

实验使用了图数据集（COX2、MUTAG、DHFR、NCI1、PTC）和点云数据集（Orbit5k）以及基准集（PROTEINS、ADJACENCY）。

**📈 对比分析**

与持久图图像、持久景观、Rips复杂度等传统向量化方法相比，PALACE在所有自适应配置下取得最高的预测准确率；在图数据集上可达约0.93-0.96的准确率，在大规模数据集上显著优于Transformer基线。

**⚠️ 局限性**

局限性包括：①对非干涉、τ-可容忍假设的依赖，在实际分布中往往难以满足；②需要对地标配置做有限的交叉验证（bandwidth、预算、放缩等）；③在高维稠密配置下的稀疏性优势被削弱；④证明的采样复杂度与实际必要样本量之间仍有多项式差距。

---

## 485. Large Language Models are Universal Reasoners for Visual Generation

**arXiv ID:** 2605.04040 | [PDF](https://arxiv.org/pdf/2605.04040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 486. OpenSeeker-v2: Pushing the Limits of Search Agents with Informative and High-Difficulty Trajectories

**arXiv ID:** 2605.04036 | [PDF](https://arxiv.org/pdf/2605.04036v1)

**作者:** Yuwen Du `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9012 | [OpenAlex ID](https://openalex.org/A5066373402)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们提出了 OpenSeeker-v2，使用仅 10.6k 质量高、难度大的 SFT 训练样本，在 30B 参数的 ReAct 模型上实现了搜索代理的 SOTA 结果。

**💡 创新点**

创新点在于三项数据合成策略：扩大知识图规模、扩充工具集合以及严格低步过滤，显著提升训练样本难度与信息丰富度，仅凭 SFT 即可击败重度 CPT+RL 的工业模型。

**🔧 技术方法**

采用 Qwen3-30B-A3B-Thinking-2507 基础模型，利用 ReAct 框架进行监督微调，未使用 RL 或额外超参调优。

**📊 数据集**

构建了 10.6k 条人工合成的高难度搜索轨迹，来源于扩展后的知识图和多工具交互，未使用公开 Benchmark 原始数据。

**📈 对比分析**

在 BrowseComp、BrowseComp-ZH、Humanity's Last Exam 与 xbench 四大深度搜索基准上，OpenSeeker-v2 分别取得 46.0%、58.1%、34.6% 与 78.0% 的准确率，超越同规模 Tongyi DeepResearch（43.4/46.7/32.9/75.0）和其他工业/开源模型。

**⚠️ 局限性**

局限在于仅验证 30B 规模下的单一 SFT，未探究更大模型或更长轨迹的扩展效果，并且依赖人工合成数据的可复制性与真实性仍待验证。

---

## 487. Probabilistic-bit Guided CDCL for SAT Solving using Ising Consensus Assumptions

**arXiv ID:** 2605.04033 | [PDF](https://arxiv.org/pdf/2605.04033v1)

**作者:** Melki Bino `[一作]` `[通讯]` (University of Texas at Dallas), Melki Bino (University of Texas at Dallas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研发了一种将p-bit Ising采样器与传统CDCL求解器耦合的混合SAT求解框架，用低能量样本生成的高一致性文字作为临时假设以减轻CDCL搜索负荷。

**💡 创新点**

通过概率位的能量采样获得高一致性文字并以假设形式导入CDCL，同时保留完整性回退，并探讨了可学习的适用性门控机制。

**🔧 技术方法**

基于Ising能量函数的p-bit采样、样本一致性评估与加权磁化分数、CDCL假设接口、预算限定的尝试-重试-回退策略以及随机森林门控。

**📊 数据集**

在SATLIB的可满足随机3-SAT（RTI、CBS、BMS）以及图着色实例上进行实验，涵盖约4800个公式。

**📈 对比分析**

与纯CDCL比较，针对CBS与RTI的随机实例，中位冲突数降低80–85%，传播数降低80–85%；在BMS略低，图着色实例效果不佳；门控实验在保留94.8%混合成功率的同时减少了大量冲突/传播。

**⚠️ 局限性**

性能高度依赖实例分布，非结构化实例效果差；采样器为Python原型，未体现时间收益；门控模型存在特征泄露，需进一步验证在新分布下的泛化。

---

## 488. SymptomAI: Towards a Conversational AI Agent for Everyday Symptom Assessment

**arXiv ID:** 2605.04012 | [PDF](https://arxiv.org/pdf/2605.04012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 489. Domain-Adaptive Dense Retrieval for Brazilian Legal Search

**arXiv ID:** 2605.04005 | [PDF](https://arxiv.org/pdf/2605.04005v1)

**作者:** Jayr Pereira `[一作]` (Universidade Federal do Cariri), Luiz Bonifacio `[通讯]` (NeuralMind)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在巴西葡萄牙语法律检索中，构建并评估了两种基于 Qwen3‑Embedding‑4B 的稠密检索器，一种仅使用法律数据进行微调，另一种采用混合监督（法律数据 + SQuAD‑pt）进行微调。

**💡 创新点**

创新点在于系统性探讨了法律领域内的稠密检索器在“专业化”与“跨检索域鲁棒性”之间的权衡，并证明混合监督能在保持法律效能的同时显著提升对非法律域（如通用检索）的适应性。

**🔧 技术方法**

技术主要包括：LoRA 微调、InfoNCE 对比损失、基于 BM25 的 hard‑negative 采样，以及 Qwen3‑Embedding‑4B 作为预训练编码器。

**📊 数据集**

使用的数据集包括 JUÁ‑Juris、JurisTCU、NormasTCU、Ulysses‑RFCorpus、BR‑TaxQA‑R（所有均来自 JUÁ 评测平台）以及 Quati（通用葡萄牙语检索基准）。

**📈 对比分析**

评估方法采用 NDCG@10、MRR@10 与 MAP@10 在六个数据集上的平均值。结果显示：基准模型平均 NDCG@10 为 0.414，法律‑专用微调提升至 0.433，混合监督微调进一步提升至 0.447；在 Quati 上混合模型的提升尤为显著（NDCG@10 由 0.438 提升至 0.503）。

**⚠️ 局限性**

局限性包括：①训练数据规模相对较小，未能全面检验大规模混合监督的效果；②实验对比仅涉及三种模型，未做完整的因果分解；③仅在六个评测数据集上验证，缺乏对更广泛法律检索场景的外部验证；④未在下游 RAG 等完整管道中评估实际影响。

---

## 490. Physics-Grounded Multi-Agent Architecture for Traceable, Risk-Aware Human-AI Decision Support in Manufacturing

**arXiv ID:** 2605.04003 | [PDF](https://arxiv.org/pdf/2605.04003v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 491. EQUITRIAGE: A Fairness Audit of Gender Bias in LLM-Based Emergency Department Triage

**arXiv ID:** 2605.03998 | [PDF](https://arxiv.org/pdf/2605.03998v1)

**作者:** Richard J. Young `[一作]` (University of Nevada Las Vegas), Alice M. Matthews `[通讯]` (DeepNeuro AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对五款 LLM（Gemini‑3‑Flash、Nemotron‑3‑Super、DeepSeek‑V3.1、Mistral‑Small‑3.2、GPT‑4.1‑Nano）在 18,714 个由 MIMIC‑IV‑ED 生成、并注入合成姓名的 ED 病例提示中，系统评估其在性别（女/男）对比下的 ESI 分级一致性（counterfactual flip rate）、方向性不平等（F/M 比例）、群体公平度（DPD）及临床准确性（κ_w、校准差距）。

**💡 创新点**

①首次在多模型、多提示策略下进行大规模 counterfactual 公平性审计；②通过姓名‑性别拆分 ablation 揭示不同模型出现同一女性低分配现象的根本机制；③系统比较四种提示干预（baseline、CoT、debiased、blind）对公平性与准确性的双重影响。

**🔧 技术方法**

使用 Kusner 等人的 counterfactual fairness 框架；构建姓名‑性别互换对；对每个模型应用四种提示策略；统计 Bootstrap、McNemar、kappa、校准差距；对 Gemini‑3‑Flash 进行 test‑retest 噪声基线测定；在 DeepSeek 与 Gemini 之间做姓名‑性别拆分 ablation。

**📊 数据集**

基于 MIMIC‑IV‑ED v2.2（≈425k ED 入院记录）的 18,714 个去标识化病例，合成使用 SSA/Census 统计同性别、同种族的姓名，确保姓名不泄露个人身份。

**📈 对比分析**

通过 374,275 次模型评估得到：所有模型的 flip rate 超过 5% 阈值，分为三种偏差模式；DeepSeek‑V3.1 与 Gemini‑3‑Flash 表现出女性低分配（F/M ≈2.15 与 1.34）；Nemotron‑3‑Super 仅高 flip rate（≈43.8%）且微弱男性倾向；GPT‑4.1‑Nano 与 Mistral‑Small‑3.2 接近对称。干预方面，blind 提升部分模型的公平性但对 flip rate 效果不一；debias 仅在 Gemini‑3‑Flash 上改进；CoT 在大多数模型中降低 κ_w 并提升 flip rate。

**⚠️ 局限性**

①数据来源单一中心，缺乏多机构泛化；②仅使用文本化结构化提示，缺少真实 triage 语境（姿态、伴随评估等）；③ground‑truth ESI 本身已含性别偏差；④姓名‑性别互换未完全分离两因素；⑤blind 提示同时去除年龄，难以区分性别与年龄的独立作用；⑥不同模型后端（Ollama 与 OpenRouter）可能引入细微行为差异。

---

## 492. Joint Design of Piggyback and Conjugate Transformation Functions for Repair Bandwidth Reduction in Piggybacking Codes

**arXiv ID:** 2605.03991 | [PDF](https://arxiv.org/pdf/2605.03991v1)

**作者:** Hao Shi `[一作]`, Hanxu Hou `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出一种新的MDS阵列码——共轭猪背码（conjugate‑piggybacking codes），通过联合设计猪背函数和共轭变换来降低节点修复带宽，保持子分片级别ℓ=n‑k；

**💡 创新点**

创新点在于首次将猪背函数与跨奇偶节点的共轭变换结合，实现既可在有限子分片下提高数据节点修复效率，又能让部分奇偶节点达到最优修复带宽，并在中等字段大小上保持MDS属性；

**🔧 技术方法**

技术主要包括：①从基本MDS实例构造基础数组；②按分组设计猪背函数并添加到奇偶节点；③对奇偶节点对角线符号做共轭线性组合变换；随后推导MDS证明与修复算法；

**📊 数据集**

实验使用了高率参数集{r=4，k∈[12,52]}, 字段F2^8；对比了OOP、代码C1、RSR‑I等传统猪背码；

**📈 对比分析**

通过理论平均修复带宽比分析与模拟，结果显示在相同子分片ℓ=r的前提下，本文码的平均修复带宽比远低于现有方案，单节点修复流量比RS低37%–55%；字段大小比HTEC、BPD、ET‑RS更小；

**⚠️ 局限性**

局限性包括：需要较大字段（q>k r^2），不适用于极小字段环境；仅在高率(r≪k)且ℓ≈r的场景下性能最佳；并未实现对所有节点的MSR级别修复，部分奇偶节点仍需额外传输。

---

## 493. Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes

**arXiv ID:** 2605.03984 | [PDF](https://arxiv.org/pdf/2605.03984v1)

**作者:** Aaron Havens `[一作]` (FAIR at Meta), Neta Shaul `[通讯]` (Weizmann Institute of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Flow Sampling框架，利用条件去噪扩散动力学从未归一化密度中进行采样；

**💡 创新点**

创新点在于直接匹配条件去噪扩散漂移，避免了MCMC或重要性采样等额外校正，同时提供了常曲率黎曼流形上闭式条件漂移；

**🔧 技术方法**

技术包括扩散模型、流匹配、固定点训练、重放缓冲、Euler-Maruyama求解器以及几何插值；

**📊 数据集**

使用合成能量函数（DW-4、LJ-13、LJ-55）、小肽结构（Ala2、Ala4）、大规模分子构象生成（SPICE、GEOM-DRUGS）以及球面vMF混合分布等数据集；

**📈 对比分析**

与iDEM、PIS、DDS、AS、ASBS等基线对比，Flow Sampling在能量W2距离、KL/JSD度量以及构象覆盖率方面均优于或相当于最先进方法，且训练成本显著降低；

**⚠️ 局限性**

局限在于固定点近似缺乏全局收敛保证，对更一般黎曼流形的推广尚未实现。

---

## 494. An $\widetilde{O} (n^{3/7})$ Round Parallel Algorithm for Matroid Bases

**arXiv ID:** 2605.03979 | [PDF](https://arxiv.org/pdf/2605.03979v1)

**作者:** Sanjeev Khanna `[一作]` (New York University), Junkai Song `[通讯]` (New York University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于独立性查询的并行算法，在 O(n^{3/7}) 轮内求解任意 matroid 的基底，并将该算法推广至 matroid 交叉与子模函数最大化等相关问题。

**💡 创新点**

创新点在于三项新技术：子集命中分解、利用短电路证据批量删除冗余元素以及自适应早停策略，这些方法将对随机电路的分析从单元素转向子集层面，显著提升了删除和收缩的进度。

**🔧 技术方法**

技术手段包括全局最优集合的分解框架、α-参数演化与子集命中概率分析、短电路证据的批量检索以及自适应进度阈值的早停策略。

**📊 数据集**

论文未使用实验数据，所有结论均通过理论证明得出；若需实验，可采用随机 matroid 或对应图的循环族作为测试集。

**📈 对比分析**

相较于 KUW 的 O(√n) 轮和 KPS 的 O(n^{7/15}) 轮，算法将基底求解的并行复杂度压缩至 O(n^{3/7})，并在 matroid 交叉问题实现 O(n^{17/21}) 轮，在子模函数最大化问题实现 O(n^{3/7}/ε^3) 轮，均为先前结果的实质性提升。

**⚠️ 局限性**

主要局限是仍停留在理论层面，依赖独立性查询 oracle；常数项与实际并行实现细节未给出；在规模极小时仍需退回传统方法。

---

## 495. Audio-Visual Intelligence in Large Foundation Models

**arXiv ID:** 2605.04045 | [PDF](https://arxiv.org/pdf/2605.04045v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 496. Precomputed Lens Transport Maps

**arXiv ID:** 2605.04017 | [PDF](https://arxiv.org/pdf/2605.04017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 497. RD-ViT: Recurrent-Depth Vision Transformer for Semantic Segmentation with Reduced Data Dependence Extending the Recurrent-Depth Transformer Architecture to Dense Prediction

**arXiv ID:** 2605.03999 | [PDF](https://arxiv.org/pdf/2605.03999v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 498. Mitigating False Positives in Static Memory Safety Analysis of Rust Programs via Reinforcement Learning

**arXiv ID:** 2605.04000 | [PDF](https://arxiv.org/pdf/2605.04000v1)

**作者:** P Akilesh `[一作]` (Indian Institute of Technology), Sridhar Chimalakonda `[通讯]` (Indian Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于强化学习的框架，用于自动识别并抑制 Rust 程序静态内存安全分析中的误报。

**💡 创新点**

创新点在于将强化学习与静态分析的 MIR 特征相结合，学习可控的误报抑制策略，并通过 cargo-fuzz 动态验证作为 RL 的可选动作，实现成本感知的混合分析。

**🔧 技术方法**

技术手段包括：从 Rust 的 Mid-level Intermediate Representation（MIR）提取语义特征；采用 PPO 强化学习算法训练抑制策略；将 cargo-fuzz 集成为动态验证动作；与多种 LLM（CodeLlama、Llama、Claude 等）进行基线对比。

**📊 数据集**

使用的数据集为 4,879 条手工标注的 Rudra 警告，来自约 20,000 个 crates.io 项目，按 70/15/15 的比例划分训练、验证和测试集。

**📈 对比分析**

通过与 Raw Rudra、RL 单独、以及多款 LLM 进行对比，RL+Fuzz 在测试集上达到了 65.2% 的准确率、0.659 的 F1 分数、59% 的精度和 74.6% 的召回率，显著优于最佳 LLM 基线（提升 17.1% F1）。

**⚠️ 局限性**

局限性包括：仅针对 Rudra 分析器验证，泛化性待证；依赖人工标注且标注过程存在误差；误报率高导致训练不平衡；动态验证成本和覆盖率受限；对 Rust 生态外的语言或更大规模项目的适用性尚未验证。

---

## 499. An Agent-Oriented Pluggable Experience-RAG Skill for Experience-Driven Retrieval Strategy Orchestration

**arXiv ID:** 2605.03989 | [PDF](https://arxiv.org/pdf/2605.03989v1)

**作者:** Dutao Zhang `[一作]` (Macao Polytechnic University), Tian Liao `[通讯]` (Macao Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Experience‑RAG Skill 的插件式检索编排层，将检索策略选择包装为可供代理调用的可重用技能，位于代理与检索器池之间。

**💡 创新点**

创新点在于把检索策略选择视为代理能力，利用场景分析、经验记忆与策略路由三大模块实现检索器池的动态调度，并提供统一的结构化检索结果包。

**🔧 技术方法**

采用了规则式路由、经验记忆记录场景特征与检索性能、检索器池封装以及结果打包等技术，并对比了多种单一检索方法与 Adaptive‑RAG 风格的路由。

**📊 数据集**

使用了 BeIR 的三个公开基准数据集：nq、hotpotqa 与 scifact，各取 120 条查询进行评估。

**📈 对比分析**

与 BM25、Dense、Hybrid RRF、Adaptive‑RAG 等基线比较，Experience‑RAG Skill 在 nDCG@10、Recall@10 与 MRR@10 上均实现最高或相近水平（nDCG@10=0.8924，优于所有固定策略，略逊于 Adaptive‑RAG，但差距极小）。

**⚠️ 局限性**

局限性包括：实验仅在采样的子语料库上进行，未实现动态候选检索器的上线；学习式路由尚未超过基于规则的方案；缺乏完整端到端代理交互评测，适合短论文或预印本而非完整系统论文。

---

## 500. Large-Scale High-Quality 3D Gaussian Head Reconstruction from Multi-View Captures

**arXiv ID:** 2605.04035 | [PDF](https://arxiv.org/pdf/2605.04035v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 501. From Intent to Execution: Composing Agentic Workflows with Agent Recommendation

**arXiv ID:** 2605.03986 | [PDF](https://arxiv.org/pdf/2605.03986v1)

**作者:** Kishan Athrey `[一作]` (Cisco Systems), Mahesh Viswanathan `[通讯]` (Cisco Systems)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5114430980)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 AutoMAS 框架，实现从用户意图自动生成多智能体系统，包括规划、任务到智能体映射、动态调用图和批判机制，并通过实验验证其有效性。

**💡 创新点**

创新点在于引入两阶段检索（混合检索 + LLM 再排序）结合智能体描述丰富技术，以及迭代批判机制，显著提升智能体推荐召回率、精度和系统鲁棒性，实现端到端的自动化生成。

**🔧 技术方法**

采用 LLM（如 GPT‑4o、o1）进行规划与再排序，混合检索（dense + sparse）在 Weaviate 上实现，智能体描述丰富通过合成查询增强，Variable Call Graph 动态路由，ReAct 风格规划，LLM‑as‑a‑Judge 与结构化评估指标。

**📊 数据集**

使用 ToolE（199 个工具、20,550 个查询）进行智能体推荐实验，使用 TaskBench（dailylifeapis、huggingface、multimedia）进行规划与组合实验。

**📈 对比分析**

与传统检索（BM25）及 Re‑Invoke 进行对比；在 Agent Recommender 上 recall@10≈0.84、nDCG@1≈0.74、mAP≈0.81；在 TaskBench 端到端评估中整体匹配率≈59%、步长比≈0.74，结构化指标 F1>0.90；批判机制提升 F1 至 0.92、序列相似度至 0.90。

**⚠️ 局限性**

存在 LLM 再排序对成本偏好的固有倾向，批判无法改进检索候选集、仅调整顺序；对大规模 agent registry 仍需分离摄取与查询；实验受限于工具数量和 prompt 微调需求。

---

## 502. Stayin' Aligned Over Time: Towards Longitudinal Human-LLM Alignment via Contextual Reflection and Privacy-Preserving Behavioral Data

**arXiv ID:** 2605.04029 | [PDF](https://arxiv.org/pdf/2605.04029v1)

**作者:** Simret Araya Gebreegziabher `[一作]` (University of Notre Dame), Toby Jia-Jun Li `[通讯]` (University of Notre Dame)

**通讯引用:** 2007 | [OpenAlex ID](https://openalex.org/A5007240808)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并部署了名为BITE的Chrome扩展，实时捕获LLM交互的即时评价，结合事件触发的后续反思和隐私保护的行为轨迹，旨在实现纵向人机对齐评估。

**💡 创新点**

创新点在于提出纵向、基于情境的对齐测量框架；采用事件驱动的后续评估与进化式同意模型；将行为轨迹与LLM评价关联，体现对齐随时间演变的视角。

**🔧 技术方法**

技术包括Chrome扩展开发、DOM监控、Meta Llama3.1 70B主题分类、Jaccard相似度匹配、逐步隐私同意交互界面。

**📊 数据集**

数据集为两周实地部署中收集的182条LLM会话及对应即时与后续评价、8名受试者的浏览记录和Gmail事件，未使用公开公开数据集。

**📈 对比分析**

通过对比即时与后续评价差异、计算方向不对称指数并进行统计显著性检验，发现准确性和相关性评估多变，信任在后续显著提升，表明单次反馈不足；整体性能表现为能捕获用户随时间的评估变化。

**⚠️ 局限性**

局限性包括受试者规模小、仅两周实验、仅使用Gmail作为后续信号、主题分类与词重叠匹配可能误判、未测量客观任务结果以及可能的社会期望偏差。

---

## 503. Decentralized Edge Caching under Budget and Storage Constraints: A Game-Theoretic Approach

**arXiv ID:** 2605.04023 | [PDF](https://arxiv.org/pdf/2605.04023v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 504. UniCorrn: Unified Correspondence Transformer Across 2D and 3D

**arXiv ID:** 2605.04044 | [PDF](https://arxiv.org/pdf/2605.04044v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 505. Safety and accuracy follow different scaling laws in clinical large language models

**arXiv ID:** 2605.04039 | [PDF](https://arxiv.org/pdf/2605.04039v1)

**作者:** Sebastian Wind `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Soroosh Tayebi Arasteh `[通讯]` (RWTH Aachen University)

**通讯引用:** 1217 | [OpenAlex ID](https://openalex.org/A5076251937)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出SaFE-Scale框架并构建RadSaFE-200安全评估基准，评估了34种临床LLM在不同部署条件（闭书、清晰证据、冲突证据、标准RAG、代理RAG、最大上下文）下的准确率与多维安全指标（高危错误、无效回答、证据矛盾、危险过度自信），重点分析规模、检索、上下文、推理和计算投入对临床安全的影响。

**💡 创新点**

创新点在于将安全性纳入LLM扩展法则的评估，设计了具备选项级安全标签的多选问答基准RadSaFE-200，并通过SaFE-Scale系统性探究证据质量、检索策略、上下文暴露和推理计算对临床安全的决定性作用，揭示安全与准确率解耦。

**🔧 技术方法**

采用多种LLM（Qwen、Llama、Gemma、MedGemma、DeepSeek、Mistral、OpenAI-OSS）以及检索增强（标准RAG、代理RAG）、最大上下文提示、自一致性推理、固定三模型投票等技术进行实验，并使用模型内部多重抽样和熵归一化的自信度衡量。

**📊 数据集**

使用200题文本化放射学多选问答基准RadSaFE-200，涵盖诊断、技术、分类等九类问题，每题配有干净证据、冲突证据及选项级高危/无效/矛盾安全标签。

**📈 对比分析**

比较方法为对34个模型在六种部署条件下的平均准确率与四个安全指标进行跨模型、跨条件的配对统计与方差分解；实验结果显示，清晰证据下准确率升至94.1%并将高危错误降至2.6%，而检索、代理RAG、最大上下文和自一致性提升有限，集成投票虽提升平均准确率但同步错误未消除。

**⚠️ 局限性**

局限性包括基准仅为文本多选问答，缺乏图像和开放式回答；安全标签依赖主观临床判断；检索仅基于Radiopaedia，未涵盖更大或多样化知识库；模型推理与自信度评估未结合外部校准或多模态评估；故实验结果对真实临床部署的推广需进一步验证。

---

## 506. Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems

**arXiv ID:** 2605.04018 | [PDF](https://arxiv.org/pdf/2605.04018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 507. Enhanced 3D Brain Tumor Segmentation Using Assorted Precision Training

**arXiv ID:** 2605.04008 | [PDF](https://arxiv.org/pdf/2605.04008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 508. Implementing True MPI Sessions and Evaluating MPI Initialization Scalability

**arXiv ID:** 2605.03983 | [PDF](https://arxiv.org/pdf/2605.03983v1)

**作者:** Hui Zhou `[一作]` (Argonne National Laboratory), Rajeev Thakur `[通讯]` (Argonne National Laboratory)

**通讯引用:** 9993 | [OpenAlex ID](https://openalex.org/A5014920685)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

实现MPICH对真正MPI Sessions的完整支持，并对其初始化可扩展性进行评估

**💡 创新点**

通过内部重构将本地与集体初始化分离，引入与会话无关的进程ID，支持组级PMI收发和原子共享内存，从而彻底去除世界通信器依赖，支持构建无世界通信器的稀疏拓扑

**🔧 技术方法**

使用MPI‑4标准、MPICH、OpenMPI、PMIx/PMI、Hydra进程启动器、Aurora HPC系统及POSIX原子操作等技术

**📊 数据集**

在Aurora系统上使用1–2048节点、每节点96进程的规模进行实验

**📈 对比分析**

比较MPICH‑dev、MPICH 4.3.0与OpenMPI 5.0.7的MPI_Init时间和节点内存占用，发现Session模型与传统世界模型相当，稀疏模型在内存和启动时间上略有提升

**⚠️ 局限性**

实现仍存在额外上下文分配开销、负载不平衡、对PMIx组障碍支持有限，以及对多线程、容错等高级功能的支持尚不完整

---

## 509. Redefining AI Red Teaming in the Agentic Era: From Weeks to Hours

**arXiv ID:** 2605.04019 | [PDF](https://arxiv.org/pdf/2605.04019v1)

**作者:** Raja Sekhar Rao Dheekonda `[一作]` (Dreadnode), Nick Landers `[通讯]` (Dreadnode)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一款基于Dreadnode SDK的AI红队代理，能够通过自然语言交互自动生成、执行多模态、多语言、多代理系统的攻击工作流，并提供统一的分析与合规报告。

**💡 创新点**

创新点在于：①agentic界面实现自然语言目标驱动的攻击生成；②统一框架兼容生成式与传统ML攻击；③完整端到端分析与自动合规映射。

**🔧 技术方法**

技术包括：大型语言模型（如Llama）、OpenAI/Claude API、Dreadnode SDK、45+攻击、450+转换、130+评分器、OpenTelemetry追踪以及自动严重性与合规映射。

**📊 数据集**

使用Dreadnode提供的恶意内容与公平偏见目标数据集，共68个攻击目标，涵盖OWASP LLM Top10、MITRE ATLAS、NIST AI RMF等。

**📈 对比分析**

在Meta Llama Scout上对比实验：3种攻击类型、5种转换，共681评估、674次攻击、7727次试验，攻击成功率约85%，严重度分布可观；相比传统库驱动方式，耗时从数周压缩至约3小时。

**⚠️ 局限性**

限制：代理的可靠性受LLM理解能力限制；评分器可能产生偏差；攻击库无法覆盖所有新技术；案例规模有限，缺乏对比实验评估代理选择质量。

---

## 510. 3D Human Face Reconstruction with 3DMM face model from RGB image

**arXiv ID:** 2605.03996 | [PDF](https://arxiv.org/pdf/2605.03996v1)

**作者:** Zhangnan Jiang `[一作]` (New York University), Zichen Yang `[通讯]` (New York University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用单张 RGB 图像，通过预训练的 50 层 ResNet 回归 Basel 3D 形状模型参数，结合 Soft Rasterizer 实现可微分渲染，并以图像差异与面部关键点差异作为弱监督损失，完成三维人脸重建。

**💡 创新点**

创新点在于：① 将弱监督的图像损失与关键点损失联合用于 3DMM 参数回归；② 在此基础上采用 Soft Rasterizer 进行可微渲染，避免了传统离散渲染带来的梯度不连续；③ 在 ResNet 结构上加入全连接层以适配 257 维参数输出。

**🔧 技术方法**

核心技术包括：ResNet-50 预训练网络、MTCNN 人脸检测与关键点提取、Basel 3D Face Model (BFM) 参数化、Soft Rasterizer（可微分光栅化器）、均方误差（MSE）图像损失与关键点损失、Adam 优化器与批量归一化/Dropout。

**📊 数据集**

使用公开人脸验证基准 Labeled Faces in the Wild (LFW) 作为训练数据，并利用 Basel 3D Face Model (BFM) 提供的形状、纹理、光照基准；在此基础上进行数据增强与预处理。

**📈 对比分析**

在实验中将训练轮数设为 20、批大小 16、学习率 1e-3，得到的 3D 重建效果在自测图像上表现出细节（如皱纹）较好；相比纯弱监督方法，该方案在重建精度上有所提升，但未给出与其他公开基准的数值对比。

**⚠️ 局限性**

局限性包括：① 对遮挡、胡须、浓妆等极端外观的鲁棒性不足；② 依赖弱监督，缺乏真实 3D 目标的直接误差反馈；③ 训练数据仍受 LFW 的多样性与分辨率限制，可能导致泛化能力受限。

---

