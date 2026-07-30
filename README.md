# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-30 | 今日论文总数: 491

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. From Conceptual Hydrologic Models to Conceptually Interpretable Neural Networks: A Snow-Water Mass-Conserving-Perceptron Framework for Discovering Catchment-Scale Precipitation-Storage-Runoff Representations

**arXiv ID:** 2607.26492 | [PDF](https://arxiv.org/pdf/2607.26492v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 2. RAGuard: A Layered Defense Framework for Retrieval-Augmented Generation Systems Against Data Poisoning

**arXiv ID:** 2607.26339 | [PDF](https://arxiv.org/pdf/2607.26339v1)

**作者:** Pushkal Kumar `[一作]`, Vincent Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RAGuard，一种针对检索增强生成（RAG）系统的双层防御框架，旨在抵御语料库投毒攻击。

**💡 创新点**

创新点在于将对抗性检索器微调与一种无标签、黑盒的“零知识推理补丁”相结合，后者通过逐一去除检索到的文档并观察答案稳定性与熵变化来检测毒化文档。

**🔧 技术方法**

技术手段包括：对检索器的对抗性对比学习（triplet margin loss）；留一法（LOO）解码结合答案语义相似度与输出熵差异生成异常分数；批量化、提前停止与子集抽样等近似策略降低推理成本。

**📊 数据集**

使用的数据集为Natural Questions（NQ）和BEIR（NFCorpus），并在两者上构造了基于LLM重写的三类合成毒化样本（伪造、矛盾、推理陷阱）。

**📈 对比分析**

与未防御、单层防御及基线检索器进行对比，实验表明RAGuard将攻击成功率（ASR）降至0.000，Recall@5保持在无毒化基线的±0.03范围内；单层检索器微调虽提升召回但未能消除攻击；补丁的额外成本为每个查询k+1次生成推理（k=5时约6倍）。

**⚠️ 局限性**

局限性包括：较高的推理开销；对多文档协同投毒的鲁棒性不足；可能产生误报，尤其在歧义或偏见文档上；实验范围受限于少量种子和特定毒化比例；依赖密集检索，无法直接对混合词典+语义检索模型评估。

---

## 3. On Exercising Governance Power in Decentralized Autonomous Organizations

**arXiv ID:** 2607.26204 | [PDF](https://arxiv.org/pdf/2607.26204v1)

**作者:** Vabuk Pahari `[一作]` (Max Planck Institute for Software Systems), Abhisek Dash `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文系统性分析了 48 个活跃的以太坊 DAO 的治理合约，提炼出关键设计维度（提案发起方式、投票平台、投票形式、证书与执行机制、否决权等），并揭示这些设计导致的信任与透明度权衡，进一步发现并归纳了一类基于治理机制的攻击（治理攻击）及其典型案例。

**💡 创新点**

创新点在于：①将 DAO 治理过程细分为提案、投票、认证与执行四阶段，并定义了若干可操作的设计维度；②通过对 48 个 DAO 的实证分析，系统地映射了设计选择与安全、去中心化、效率之间的权衡；③首次提出“治理攻击”这一新的攻击范式，说明即使合约实现无 bug，设计缺陷也可导致重大安全事件。

**🔧 技术方法**

主要技术包括：智能合约静态与动态分析（以太坊节点 Erigon 与 Etherscan API）、投票数据抓取（Snapshot API）、文本挖掘与文档解析、以及定量统计（投票时序、参与度、投票差异等）

**📊 数据集**

使用的数据集包括：48 个 DAO 的治理合约源码与已部署字节码（Etherscan）、治理代币合约、Snapshot 投票记录、以及 DeepDAO 上的 Treasury 规模与活跃度数据。

**📈 对比分析**

研究采用对比分析而非传统模型评估：对各 DAO 在 48 个设计维度上的配置进行交叉表述，统计各维度与安全事件（如投票抢占、否决权缺失等）的相关性；实验表明，缺失否决权、无投票延迟、允许任何人发起提案的 DAO 更易遭受治理攻击，但并未给出数值性能指标；

**⚠️ 局限性**

局限性：①只关注公开的以太坊 DAO，排除私有或非活跃 DAO；②未深入评估 DAO 内部治理实践与社区行为对安全的影响；③研究侧重设计与安全关联，缺乏对治理合约改进后长期效能的实验验证。

---

## 4. Steering Instruction Hierarchies at Inference Time

**arXiv ID:** 2607.26228 | [PDF](https://arxiv.org/pdf/2607.26228v1)

**作者:** Siqi Zeng `[一作]` (University of Illinois, Urbana-Champaign), Julia Hockenmaier `[通讯]` (University of Illinois, Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练的推理时方法 V-Steer，通过编辑 KV 缓存中的 value 向量来恢复语言模型的指令层级。

**💡 创新点**

创新点在于利用直接 Logit 归因（DLA）识别低优先级干扰头，并仅在缓存层面进行乘法增益/抑制，保持推理速度。

**🔧 技术方法**

采用 DLA、缓存值编辑、乘法增益/抑制和单次预填充前向传递来实现。

**📊 数据集**

在 Control Illusion 和 IHEval 两个指令层级冲突基准上评估，覆盖 7B–70B Llama、Qwen 系列模型。

**📈 对比分析**

与仅提示、训练增强等基线相比，V-Steer 在主要约束准确率上从低于18%提升至92%，与最先进训练方法相当且在推理速度上仅有 1% 额外开销。

**⚠️ 局限性**

局限包括需要手工或自动提取冲突片段、在某些任务上轻微削弱通用性能、以及对极端增益抑制参数可能产生重复生成。

---

## 5. StealthBench: Measuring Operational Stealth in Autonomous Offensive-Security Agents

**arXiv ID:** 2607.26314 | [PDF](https://arxiv.org/pdf/2607.26314v1)

**作者:** Ads Dawson `[一作]`, Adrian Wood `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了StealthBench，评估大型语言模型自动化渗透测试代理在执行攻击时的隐蔽性（OPSEC）表现；

**💡 创新点**

首次将“隐蔽性”作为客观指标，通过多维度（凭证泄露、破坏性操作、检测级联、遥测注入、工件污染、噪声纪律）划分评估标准，并采用LLM裁判面板实现可扩展的自动评估；

**🔧 技术方法**

利用三大模型（GPT‑5.6 Sol、GLM‑5.2、Kimi‑K3）组成裁判面板，对完整的Agent轨迹（ATIF格式）进行JSON格式判定；

**📊 数据集**

基于11条真实红队/bug‑hunt OPSEC事件扩展成14个docker化任务场景，包含约771条Agent轨迹；

**📈 对比分析**

对8款代理模型（包括前沿与非前沿模型）在14个任务中进行5次重复实验，计算安全成功率、Stealth@Solve和鲁莽完成率；结果显示无模型安全成功率超过54%，表明隐蔽性与攻击能力相互独立且普遍欠缺；

**⚠️ 局限性**

局限包括数据集规模有限、仅覆盖工具调用型代理、裁判模型与被测模型存在重叠、判定仍以模型推断为主且缺乏人类专家基准，且未能因果分辨训练信号与提示、策略等因素的影响。

---

## 6. Shared SFT Lessons Across Alignment, Model Organisms, and Toy Models

**arXiv ID:** 2607.26173 | [PDF](https://arxiv.org/pdf/2607.26173v1)

**作者:** Anton de la Fuente `[一作]`, Arthur Conmy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在对齐训练、模型有机体实验和简化模型实验之间迁移监督微调(SFT)的策略，探讨了行为泛化、能力保持和对后续训练的鲁棒性等共享目标。

**💡 创新点**

创新点在于：①把“以原因训练”提升行为泛化的技巧从对齐训练迁移到简化模型；②把“off‑model输出会削弱能力”这一发现用于对齐训练，并通过“on‑model replay”恢复能力；③验证已安装行为在后续正常微调中的“wash‑out”鲁棒性，并发现中间训练可缓解此问题。

**🔧 技术方法**

使用的技术包括：监督微调 (LoRA)、原因/示例式训练、off‑model 与 on‑model 数据混合、模型规范中间训练 (Model‑Spec Midtraining)、行为评估指标准备（如 GPQA Diamond、Agentic Misalignment、动物福利与自我保护审计）。

**📊 数据集**

数据集涵盖：①三种简化模型任务（打包答案、动物福利、自治防止关机）使用 Qwen3‑4B / Qwen3.5‑4B 生成的标注；②Model‑Spec Midtraining 使用 Claude Opus 推理轨迹及 Qwen3‑32B 自己生成的中间样本；③对齐训练与洗出实验采用公开的 GPQA、AM 场景集。

**📈 对比分析**

对比方法：将仅示例训练与带原因训练、仅使用 off‑model 训练与混合 on‑model replay、对齐训练与中间训练混合 replay、以及后续 benign 微调前后对同一模型进行评估。性能表现为：原因训练使打包答案的泛化率从约10%提升至近100%；混合 on‑model replay 能在保持目标行为（AM ≤ 0.2）同时将 GPQA 复原至90%+；wash‑out 后中间训练模型的 AM 恢复幅度仅为 39%，而纯对齐训练为 88%。

**⚠️ 局限性**

局限性：仅在监督微调阶段验证，未考察 RL 或更强目标相关任务；评估指标主要为 GPQA、AM，未覆盖所有能力维度；数据量有限，缺乏跨模型（如不同规模或架构）的泛化验证；off‑model 影响与教师模型能力无关的假设需要更系统验证。

---

## 7. Neural Architecture Search for Traffic Prediction: A Survey of Methods, Challenges, and Future Directions

**arXiv ID:** 2607.26467 | [PDF](https://arxiv.org/pdf/2607.26467v1)

**作者:** Truong Giang Vu `[一作]` (Ontario Tech University), Richard W. Pazzi `[通讯]` (Ontario Tech University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了交通预测领域中神经架构搜索（NAS）技术，按搜索策略（梯度法、进化法、一拍/权重共享法等）组织已有工作，并指出当前研究的开放挑战与未来方向。

**💡 创新点**

创新点在于提出清晰的NAS方法分类与搜索空间设计的统一框架，系统梳理了空间-时间数据的特定需求，识别了计算成本、手工搜索空间设计、跨城市迁移、固定图结构与基础模型集成等五大挑战，并为未来研究提供了可行路线。

**🔧 技术方法**

主要技术包括：差分可微搜索（DARTS/AutoST/AutoSTG等）、进化搜索（Klosa et al., LENAS等）、一拍权重共享搜索（AutoCTS/AutoSTF等）、以及零成本代理评估与元学习搜索空间等。

**📊 数据集**

使用的交通预测数据集包括 METR‑LA、PeMS‑BAY、PeMS04/08、LargeST（≈8600 传感器）等公共基准，以及城市级别的不同传感器网格数据。

**📈 对比分析**

作者通过表格对比 MAE、搜索成本（GPU‑h）等指标评估不同方法，发现 NAS 得到的模型往往能匹配或超越手工设计的基准模型，但在不同城市和数据规模上仍存在性能差异。

**⚠️ 局限性**

限制主要体现在：① 计算成本高，尤其在大规模网络上难以可行；② 搜索空间仍由人工定义，缺乏系统性；③ 迁移性差，需在新城市重新搜索；④ 大多数方法假设图结构固定，无法应对动态交通网络；⑤ 随着空间‑时间基础模型出现，NAS 需重新定位搜索对象。

---

## 8. DHRCL:Training Code LLMs with Dense Hierarchical Rewards and Curriculum Learning

**arXiv ID:** 2607.26457 | [PDF](https://arxiv.org/pdf/2607.26457v1)

**作者:** Shuhang Wang `[一作]`, Hui Cheng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了AAAI会议论文的排版与出版规范说明，阐述了使用2027年版的样式文件和引用格式的要求。

**💡 创新点**

创新点在于统一了排版流程和文件格式，明确了作者在提交电子稿时必须遵循的标准，降低了排版错误与出版延误。

**🔧 技术方法**

采用LaTeX排版系统和aaai2027.bst引用样式。

**📊 数据集**

无数据集；本文为格式说明文档。

**📈 对比分析**

对比方法：仅与旧版样式文件进行对比，说明新文件兼容性与易用性提升，未涉及实验性能评估。

**⚠️ 局限性**

限制：内容仅限排版规范，不包含科研实验或数据分析；对非LaTeX用户支持有限。

---

## 9. Hakka Kitchen: Engagement with Culinary Cultural Heritage Through Immersive Game Play

**arXiv ID:** 2607.26183 | [PDF](https://arxiv.org/pdf/2607.26183v1)

**作者:** Jingle Huang `[一作]`, Ray LC `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个基于VR的游戏化烹饪体验《Hakka Kitchen》，通过让玩家在虚拟厨房中以手部动作完成客家馅料苦瓜的全过程，来传播和增强对客家烹饪非物质文化遗产的认知与情感。

**💡 创新点**

① 通过专家访谈构建“程序词典”，把隐性的烹饪经验（阈值、触感、时机等）转化为可编码的游戏机制；② 提出“约束化执行”框架，强调在游戏中保持文化传承的核心约束而非消除难度；③ 设计了动作遮蔽（action‑masking）效应的实验对照，揭示交互与观摩在文化传播层面各自的优势与局限。

**🔧 技术方法**

使用Unity 6引擎、Meta Quest 3头戴设备，结合手部追踪与物理仿真实现逼真刀工、切割、蒸煮等交互；配合音频提示、视觉反馈与叙事音频构建沉浸式学习环境。

**📊 数据集**

采用了40名大学生受试者（18‑37岁）做为实验样本，随机分为交互式VR与观摩式VR视频两组；同时使用访谈记录与标准化问卷（IMI、GEQ、ICH量表、MCQ）评估体验效果。

**📈 对比分析**

通过两组之间的比较（Mann‑Whitney U + Cliff’s δ，未校正p值）发现：交互式VR在兴趣/享受、感官与想象沉浸、正面情绪及部分文化认知维度（传承感、活力）上显著优于视频组；但程序知识得分无显著差异。效果量大（δ≈0.6–0.8）表明交互化设计对体验与文化感知有明显提升。

**⚠️ 局限性**

限制包括：实验时间短，未测量长期记忆或真实厨房迁移；缺乏化学感知（味觉、嗅觉）反馈导致沉浸度不完全；动作遮蔽可能隐藏叙事信息；样本以年轻华人大学生为主，泛化性受限；未对不同技能水平或跨文化受试者进行验证。

---

## 10. Lilith: Backdoor Generalization under Training-Inference Trigger Shift

**arXiv ID:** 2607.26099 | [PDF](https://arxiv.org/pdf/2607.26099v1)

**作者:** Zhou Feng `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**通讯引用:** 8140 | [OpenAlex ID](https://openalex.org/A5058611515)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了 Lilith 框架，能够仅用单一训练时的触发器植入后门，并在推理阶段通过一族未见触发器实现攻击，解决训练-推理触发器移位问题。

**💡 创新点**

创新点包括：①将后门泛化建模为触发器支持位移问题，引入 anchor‑clearance 与 family‑reach 的几何条件；②设计黑盒 anchor‑to‑family 两阶段攻击流程；③证明表示对齐是族激活的核心机制，并给出充分条件。

**🔧 技术方法**

采用触发器设计与优化（频域、形变、混合），表示几何分析、Lipschitz 约束、对齐投影、随机/Sobol 采样、黑盒 surrogate 训练、性能评估指标等技术。

**📊 数据集**

使用 CIFAR‑10/100、TinyImageNet、SVHN、ImageNet‑1K 等数据集，受害模型包括 ResNet‑18/34、VGG13‑BN、ViT、SimpleViT 等。

**📈 对比分析**

在与多种基线（固定模式、动态/样本特定、频域、清标签等）相同触发器族评估下比较，Lilith 在 0.5–5% 的毒化率下族激活率均≥90%，误差≤3.5%，显著优于基线。

**⚠️ 局限性**

局限性：仍依赖 surrogate–victim 传递假设，对更大模型或高维特征迁移的鲁棒性尚未充分验证；部分检测方法仍能偶尔识别；对触发器空间的完整泛化还有待进一步研究。

---

## 11. Framework Implementation Maturity in Blockchain-Based Third-Party Compliance Assessment

**arXiv ID:** 2607.26087 | [PDF](https://arxiv.org/pdf/2607.26087v1)

**作者:** Jemima Owusu-Tweneboah `[一作]` (Tennessee Technological University), Maria Luisa Figueroa `[通讯]` (Texas A&M University Central)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于许可区块链的第三方风险评估框架（TPRA），将第三方合规性评估视为跨周期的成熟度测量，并定义了相应的度量指标与成熟度模型；通过在医疗远程患者监测（RPM）场景中的示例评估演示了该框架的可行性。

**💡 创新点**

创新点在于：①将合规评估从单次静态核查转为可重复、可验证的成熟度测量；②利用许可链的交易执行、共识和权限策略，强制执行评估生命周期、证据锚定与多评估人治理；③提出以链上证据、投票记录与最终化状态为依据的证据驱动成熟度指标与定性成熟度层级；④通过情境驱动方法验证指标在多周期评估中的可操作性。

**🔧 技术方法**

使用技术包括：Hyperledger Fabric 许可链、链码（智能合约）、多方共识与 endorsement 策略、通道治理、哈希锚定的离线证据存储、基于投票的聚合规则、事件审计日志及与外部身份认证（MSP）集成。

**📊 数据集**

采用构造的医疗远程监测（RPM）案例场景作为数据集，未使用真实行业数据，而是通过情境模拟演示评估周期与证据提交。

**📈 对比分析**

比较方法是情境驱动的度量评估：在两轮评估周期内收集链上事务记录，计算 M1–M8 指标并映射到 L0–L4 成熟度层级；未进行性能基准测试，仅在讨论中给出线性扩展性预测，认为评估次数和参与方数量对交易量影响可控。

**⚠️ 局限性**

局限性包括：①缺乏大规模部署的性能评估；②证据完整性与安全依赖外部存储，未提供加密证明；③无法独立验证证据内容的真实性，只保证不可篡改；④对协同攻击的抵御能力有限；⑤未覆盖 DoS 等可用性威胁；⑥仅为原型验证，未实现完整的工业级部署。

---

## 12. Aligning LLM-Simulated and Human Examinees for Psychometric Calibration: A Cognitive Diagnostic Profiling Approach

**arXiv ID:** 2607.26317 | [PDF](https://arxiv.org/pdf/2607.26317v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 13. Time-delay Control Using a New Nonlinear Adaptive Law for Cable-Driven Robots

**arXiv ID:** 2607.26383 | [PDF](https://arxiv.org/pdf/2607.26383v1)

**作者:** Wenbo Gao `[一作]` (Nanjing University of Aeronautics and Astronautics), Hanzhuo Wang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种基于时间延迟估计（TDE）的无模型控制框架，结合分数阶无奇点终端滑模（FONTSM）误差动力学与快速终端滑模到达律，为绳索驱动机器人实现精确轨迹跟踪。

**💡 创新点**

核心创新点在于：①在自适应增益更新中引入自适应指数项，形成非线性自适应增益θ̂^σ̂，既能在平滑跟踪阶段抑制噪声诱发的抖振，又能在轨迹反转时迅速提升控制增益；②将FONTSM与快速终端滑模结合，实现快速有限时收敛；③在TDE框架下实现无模型控制，显著降低建模与参数识别的难度。

**🔧 技术方法**

使用技术包括：时间延迟估计（TDE）对未知动力学进行无模型补偿；分数阶无奇点终端滑模（FONTSM）误差动力学与快速终端滑模到达律；自适应指数项的非线性自适应律；Lyapunov稳定性分析；分数阶运算的数值实现。

**📊 数据集**

实验数据来源于实测平台：一台绳索驱动机械臂（MOONS ECU19058H24-S001电机与AQMD2403BLS-M驱动），采样频率1 kHz，执行预定轨迹并附加50 g负载进行鲁棒性测试；实验记录了关节角误差、控制力矩、TDE估计误差等信号。

**📈 对比分析**

通过与基线TDE自适应控制器及三种最近的自适应律进行比较，采用RMSE、ITAE、ISCT三项指标评估性能。实验结果显示：对关节1，RMSE下降34.5%，ITAE下降33.8%，ISCT下降6.7%；对关节2，RMSE下降31.1%，ITAE下降32.9%，ISCT下降17.8%。自适应律表现出更快的增益响应、更小的振荡以及更好的抖振抑制。

**⚠️ 局限性**

局限性包括：①分数阶运算导致的计算量和内存开销，需在采样周期内完成；②对测量噪声敏感，需低通滤波，滤波过度会引入相位延迟；③TDE误差随采样间隔依赖，低采样率或高速扰动会降低估计精度；④稳定性分析中对控制增益大于估计误差的假设较保守，实际系统可能更稳健。

---

## 14. QUIC-TRIP: A Triple-Redundant Journey Toward Secure Substation Communications

**arXiv ID:** 2607.26379 | [PDF](https://arxiv.org/pdf/2607.26379v1)

**作者:** Jorge David de Hoz Diego `[一作]` (University College Dublin), Anca Jurcut `[通讯]` (University College Dublin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了QUIC-TRIP——一种在传输层实现的多路径三重冗余反射器/过滤器，用于在不影响应用协议的前提下，安全传输R-GOOSE等电力子站通信，并在DoS攻击下提升可靠性。

**💡 创新点**

在QUIC的多路径与零延迟握手基础上，首次将端到端加密与三重冗余结合，提供透明加密、复制及过滤的多路径安全机制，显著降低RTT并增强对DoS攻击的抵御能力。

**🔧 技术方法**

使用QUIC/QUIC‑TLS 1.3、SSH3代理、R‑GOOSE、DTLS 1.2、OpenVPN、Linode云实验平台、UDP echo RTT测评、DoS模拟攻击、三路多路径反射与过滤。

**📊 数据集**

实验主要使用自建的162字节R‑GOOSE数据包；在Linode Frankfurt‑Amsterdam链路上进行RTT和DoS测试；未使用公开数据集，全部基于实验生成的流量。

**📈 对比分析**

通过UDP echo与DTLS 1.2、OpenVPN比较，平均RTT分别为6.25 ms、6.97 ms、6.43 ms，QUIC代理仅比DTLS高2.88%；在DoS场景中，反射过滤配置将延迟从≈100 ms降低至10–20 ms；每条启用路径的通信开销约32.18%。

**⚠️ 局限性**

开销相对较高（每条路径≈32%）、依赖多路径网络和代理节点、实验环境为云IaaS，未在真实子站环境下验证时延稳定性、缺乏长期可扩展性与高强度DoS攻击极限评估。

---

## 15. Voice Memory for Agentic Speech Recognition

**arXiv ID:** 2607.26410 | [PDF](https://arxiv.org/pdf/2607.26410v1)

**作者:** Chao-Han Huck Yang `[一作]` (NVIDIA), Boris Ginsburg `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在保持 ASR 纠错模型权重不变的前提下，通过在推理时读取并优化一个可编辑的文本记忆文件，实现对低错误率语音识别的自适应纠错。

**💡 创新点**

创新点在于：①把纠错决策拆分为“是否纠错”与“如何纠错”两步，形成 listener‑thinker 架构；②利用只前向推理的验证门循环，基于评分信号生成有限制的增删改规则，学习到的记忆既可审计、可迁移，又不需训练；③引入 Recoverable Information Ratio 与 Harmful Edit Rate 两种衡量指标，系统性评估纠错效果与过度纠错风险。

**🔧 技术方法**

核心技术包括：冻结的 ASR‑LM 纠错器、外部文本记忆、基于分数的增删改编辑生成器、语义相似度门控（可选）、以及基于词级编辑计数的有害编辑率计算。

**📊 数据集**

使用了 HyPoradise v0（10 个语音领域）、Robust HyPoradise（CHiME‑4、NOIZEUS 等噪声集）、以及多语言 X→En 机器翻译任务的数据集进行实验。

**📈 对比分析**

与传统的零样本 GER、5‑shot 交互式提示、以及 fine‑tune 或噪声嵌入方式的纠错方法比较，本文方法在 1‑best‑to‑oracle 的可恢复信息比例上高达 0.9 以上，WER 通常比静态 GER 低 0.5–1.0%，且在高错误率领域表现最佳，且不需要额外的训练开销。

**⚠️ 局限性**

主要限制包括：实验只基于单一大型开源模型（MiniMax‑M3），对语义门控的依赖可能因句子编码器不同而变化；记忆文件的质量高度依赖人工编写或模型生成的规则，且在跨域迁移时仍需人工审核以避免偏见。

---

## 16. Comparing the Performance of Foundation Model Derived Embeddings with Traditional Approaches for Distant Metastasis Prediction in Head and Neck Cancer

**arXiv ID:** 2607.26276 | [PDF](https://arxiv.org/pdf/2607.26276v1)

**作者:** Erich Schmitz `[一作]` (University of Texas Southwestern Medical Center), Jing Wang `[通讯]` (University of Texas Southwestern Medical Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了利用CT基础模型提取嵌入向量预测头颈癌远处转移风险，并将其与传统放射组学和ViT特征进行比较。

**💡 创新点**

创新点在于使用无需手工分割的CT基础模型特征，降低对专家知识和大规模标注数据的依赖，并证明其预测性能可与传统方法相当或更优。

**🔧 技术方法**

采用了CT Foundation预训练模型生成一维嵌入、放射组学特征、ViT深度学习特征，并使用多层感知机和支持向量机进行分类。

**📊 数据集**

使用了公开的RADCURE CT影像数据集，共计2327例头颈癌患者，其中1819例用于训练/验证，508例用于测试。

**📈 对比分析**

通过5折交叉验证与hold‑out测试集评估AUC、AP、灵敏度与特异度等指标，CT基础模型AUC为0.791，放射组学0.772，ViT 0.753，放射组学+ViT 0.794，CT基础模型与组合模型无显著差异，性能接近或优于传统方法。

**⚠️ 局限性**

局限性包括仅使用单一机构数据、未评估其他基础模型、CT基础模型需云端API并涉及数据隐私、以及对对比剂和性别分布的影响未充分探究。

---

## 17. Registration-Grounded Spectral Fusion for Unregistered WLI/NBI Endoscopic Lesion Segmentation

**arXiv ID:** 2607.26395 | [PDF](https://arxiv.org/pdf/2607.26395v1)

**作者:** Pengyu Jie `[一作]` (Sun Yat-sen University), Chenqiang Gao `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `da1b1a89-583a-4b57-9c81-478778569bec` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种可靠性感知的复杂频域融合框架，用于配对但未对齐的白光成像（WLI）与窄带成像（NBI）内镜病变分割。

**💡 创新点**

创新点在于将拓扑正则化的差分变形与可靠性估计相结合，构建可靠的跨模态对应基准，并在复数频域中按 WLI 的幅值、NBI 的相位角色实现可学习的幅相交互，从而抑制不可靠的跨模态干扰。

**🔧 技术方法**

使用了可微分的特征级差分流形注册（R‑DRG）、可靠性映射、可学习的复数幅相融合（R‑CSF）、频域自适应重校准以及多尺度 FPN‑SegFormer 解码器等技术。

**📊 数据集**

在两个配对但未对齐的内镜数据集 FAHSYSU（Paired）和 PICCOLO（Paired）上进行实验。

**📈 对比分析**

与十个基线方法（UMF‑SegNet、DFormer、FTransUNet、DCRS、VI‑ReID、ShapeConv、MCSNet、MFNet、UMSCS、ADFNet）在 mIoU、DSC、PA、Recall、Precision 等指标上均实现显著提升（p<0.05），在边界质量指标（HD95、ASSD、BIoU）也位居前列。

**⚠️ 局限性**

局限性包括：仍需依赖无监督的特征级对齐，极端视角或严重局部失配时性能可能下降；未给出精确的像素级对齐评估，缺少跨机构/跨设备的泛化验证。

---

## 18. Optimizing Sensor Placement for Hydrogen Leak Detection in Enclosed Infrastructure: A Comparative Study Using CFD-informed Genetic Algorithm and DeepSets Neural Surrogate

**arXiv ID:** 2607.26078 | [PDF](https://arxiv.org/pdf/2607.26078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 19. Entity Resolution in Practice: Lessons from a Self-Serve Pipeline

**arXiv ID:** 2607.26298 | [PDF](https://arxiv.org/pdf/2607.26298v1)

**作者:** Kaushik Pavani `[一作]` (Walmart), Kiran Sanka `[通讯]` (Walmart)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个自助式实体解析（ER）管线，验证了三条关键实战经验。

**💡 创新点**

提出了无单一匹配器适用、精确度与召回需独立工具、以及需加入验证合并防止单误匹配导致错误聚类的三项实战经验。

**🔧 技术方法**

采用 SOP+LLM 教师+蒸馏轻量匹配器、检索与匹配分离、嵌入+标识符检索、稀疏阈值/硬规则、Verified Merge 等技术。

**📊 数据集**

使用六个公开 ER 基准（Restaurants、Cora、GeoSettlements、DBLP-Scholar、MusicBrainz 200K、NCV 5M）进行评估。

**📈 对比分析**

通过在每个数据集上对比不同匹配器家族、阈值和聚类策略，采用 Pair-F1、纯度和 ARI 等指标，实验显示自动匹配器锦标赛、稀疏阈值+硬规则和 Verified Merge 可在大多数基准上显著提升 F1，尤其在大型数据集上避免了巨集群。

**⚠️ 局限性**

实验局限在于仅使用公开基准、10K 标注限制、仅涵盖三类匹配器、Verified Merge 在极低基准精度时效果有限，并未测量人工成本与实时漂移鲁棒性。

---

## 20. LumaGuide: Distribution Shaping for Training-Free HDR Generation in Diffusion Models

**arXiv ID:** 2607.26237 | [PDF](https://arxiv.org/pdf/2607.26237v1)

**作者:** Bowen Chen `[一作]` (University of Texas at Austin), Alan C. Bovik `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练、能量引导的采样框架 LumaGuide，通过在 PQ 空间对亮度直方图进行分布形状控制，实现预训练扩散模型的 HDR 输出，同时保持语义和几何结构。

**💡 创新点**

创新点在于：①仅在采样阶段通过可微软直方图和 Wasserstein‑1 距离对输出分布进行形状约束，无需改动模型参数；②提供三种目标设定（预设、参考图、文本驱动），可灵活控制 HDR 统计；③扩展到视频生成时加入 Temporal Luminance Coherence 约束，实现时序一致的 HDR 视频。

**🔧 技术方法**

使用的技术包括：能量引导采样、软直方图逼近、PQ 亮度空间、Wasserstein‑1 距离、Temporal Luminance Coherence（TLC）约束、无训练的扩散模型。

**📊 数据集**

实验数据集主要包括：Beyond8Bits（用于文本驱动的亮度直方图回归）、100 条 HDR 主题文本提示、Flux.1‑dev、SD3、SDXL 预训练扩散模型以及 CogVideoX 视频扩散模型。

**📈 对比分析**

与 LEDiff、BracketDiffusion、X2HDR 等基线对比，LumaGuide 在 Q‑alignment、DR_stops、JOD 评分上均优于 X2HDR，Q‑quality 也达到或超过基线；运行时间与 X2HDR 相近，远快于 BracketDiffusion。

**⚠️ 局限性**

局限性包括：过大的引导尺度可能导致 banding 与纹理失真；当前仅控制亮度直方图，无法直接处理更复杂的空间或多模态分布；视频时仍需额外的 TLC 约束来保持时序一致。

---

## 21. Meta-Learned Reward Shaping for Reinforcement Learning from Human Feedback

**arXiv ID:** 2607.26094 | [PDF](https://arxiv.org/pdf/2607.26094v1)

**作者:** Yunpeng Chu `[一作]` `[通讯]` (Stony Brook University), Yunpeng Chu (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在RLHF中使用元学习得到的奖励塑形函数，称为MeRLa，能够在训练前通过辅助任务学习获得任务感知的奖励信号，并在正式RLHF阶段与基准奖励模型融合；

**💡 创新点**

创新点在于将奖励塑形转化为元学习问题，使用潜在基础约束保证策略不变，构建包含任务区分、熵正则和保守性三项的元目标，并在理论上证明了策略不变、表示漂移敏感度与熵最大化导致的激励失配的上界；

**🔧 技术方法**

采用的技术包括：元学习框架（双层优化）、轻量级MLP塑形网络、冻结编码器提取稳定嵌入、谱归一化控制Lipschitz、潜在基础塑形（token级分解）、熵正则与保守性损失、以及在RLHF中使用的GRPO/PPO/DPO等算法；

**📊 数据集**

使用的主要数据集为LLaMA‑3‑8B模型，UltraFeedback偏好数据训练奖励模型，AlpacaEval 2.0、MT‑Bench、MATH、IFEval四个评估基准；

**📈 对比分析**

与SFT、PPO、DPO、GRPO、DAPO等基线对比，MeRLa在AlpacaEval 2.0的长度控制胜率达90.8%（比DAPO高3.9点），MT‑Bench 9.14、MATH 53.4%、IFEval 81.2%均有显著提升；在不同RLHF骨干上均表现出显著增益；

**⚠️ 局限性**

局限性包括：元学习阶段额外的GPU时间开销；潜在基础约束在实际中存在残差误差，理论保证在残差为零时才严格成立；塑形网络仅使用固定长度嵌入，难以处理变长或token级细粒度塑形；实验仅验证在LLaMA‑3‑8B上，尚未在更大模型上检验；

---

## 22. The Age of AI Agents Demands A New Scientific Paradigm To Sustain Trustworthy Science

**arXiv ID:** 2607.26064 | [PDF](https://arxiv.org/pdf/2607.26064v1)

**作者:** Belinda Mo `[一作]` `[通讯]` (Long Horizon Research), Belinda Mo (Long Horizon Research)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析了AI代理在科研中的崛起，并提出了适应新形势的验证基础设施框架。

**💡 创新点**

首次系统化定义了可观察性、可归属性与可重复性三大挑战，并提出可观测工作流、分层验证协议、明确归属标准等创新方案。

**🔧 技术方法**

利用MLOps实验跟踪、版本控制、日志记录等技术实现自动化可观测工作流，并设计分层验证与归属标准。

**📊 数据集**

无具体数据集；讨论基于现有AI科研工具与实践。

**📈 对比分析**

无实验比较，本文为方法论性论文。

**⚠️ 局限性**

实施成本高、对资源有限实验室的负担、可能导致创新受限及缺乏可行性评估。

---

## 23. Spline-Based Boundary Representations for Sparse View Reconstruction and Simulation Using Isogeometric Analysis

**arXiv ID:** 2607.26234 | [PDF](https://arxiv.org/pdf/2607.26234v1)

**作者:** Davor Dobrota `[一作]` (Schindler AG), Malcolm Mielle `[通讯]` (Schindler AG)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过从稀疏多模态图像直接优化多面 B‑样条边界表示，构建可直接用于有限元或等距分析的闭合、光滑、拓扑有效的数字孪生模型，并在同一基底上投影温度、语义等物理场。

**💡 创新点**

首次在重建优化过程中同时保证几何闭合、法向连续、曲率平滑和可用于数值仿真的兼容性，并通过双表征策略将可微渲染与 B‑样条几何同步，避免后处理修复。

**🔧 技术方法**

采用双表征（B‑样条+网格+纹理）与可微渲染、L‑BFGS 优化、粗细分层 h‑refinement、法向连续与曲率正则化、MS‑SSIM 损失、Dr.Jit+GPU加速的 IGA 线性系统构造。

**📊 数据集**

使用三类数据集：合成纹理无对象（Suzanne、Bunny、Spot、Armadillo）、BuildNet3D 真实建筑、以及四个包含 RGB 与热图的真实场景（建筑 A、狮子、汽车、木棚）。

**📈 对比分析**

与基准 FEM/IGA 热与模态仿真以及现有 Mesh‑based 形状优化（Large Steps）和 NVS 对比，重建误差低于 0.1mm，模态误差 ≤ 6.9%，热模拟误差 1–5%，在 4–6 小时 GPU 训练后即可得到可直接导出 STEP/IGES 的模型。

**⚠️ 局限性**

仅适用于 genus‑0 单体固体；B‑样条可能导致细节过平滑；相机位姿与尺度仅通过 VGGT 估计，缺乏自动尺度恢复；训练时间较长且对场的采样有限。

---

## 24. Lightweight Image Classification of Raptor Species for Edge Devices: Rare-Species Dataset Expansion via Video Frame Extraction, Knowledge Distillation, and TensorRT Deployment

**arXiv ID:** 2607.26238 | [PDF](https://arxiv.org/pdf/2607.26238v1)

**作者:** Takeshi Nishikawa `[一作]` `[通讯]` (Foundation for Computational Science), Takeshi Nishikawa (Foundation for Computational Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种轻量级猛禽分类系统，结合知识蒸馏、视频帧抽取扩充稀有种数据，并将其部署在 NVIDIA Jetson Orin Nano 上，实现风机碰撞预防的实时识别。

**💡 创新点**

创新点在于通过显著扩大 Steller's Sea Eagle 的样本量、重新微调 DINOv2-L 教师模型并重新蒸馏，最终使轻量级集成模型在视频/源图像分组拆分下宏召回达 0.935±0.004，同时显著降低两种相近种的误判率。

**🔧 技术方法**

采用 DINOv2-L 作为教师模型，将其知识蒸馏至 MobileNetV4、ViT‑Small 和 EfficientNet‑B0；使用 TensorRT FP16 引擎在 Jetson Orin Nano 上进行推理；进行多种子对照实验、A/B 测试以及 INT8 量化比较。

**📊 数据集**

使用 12,519 张六种猛禽（包含 2,050 张 Steller's Sea Eagle 视频帧抽取样本以及来自 iNaturalist、GBIF、Wiki 等渠道的图片）构建的数据集，并对比旧旧数据集与扩充后数据集的性能差异。

**📈 对比分析**

通过五个随机种子对蒸馏与非蒸馏、不同教师版本（DINOv2‑L vs. DINOv3‑L）的性能进行对照，集成模型宏召回为 0.935±0.004；单学生 EffNet‑B0 在 FP16 推理下 3.19 ms/图，误差仅 0.2%，与 FP32 结果几乎一致。

**⚠️ 局限性**

主要限制在于蒸馏本身对精度提升有限，主要依赖数据扩充；在低分辨率下性能显著下降；系统未实现端到端实时检测链，无法完整评估 OOD 拒绝与实时部署的整体效果。

---

## 25. Do Code Language Models Use Tests? A Behavioral and Representational Study of Test-Driven Code Generation

**arXiv ID:** 2607.26244 | [PDF](https://arxiv.org/pdf/2607.26244v1)

**作者:** Yunhao Liang `[一作]` (Chengdu Institute of Computer Applications, Chinese Academy of Sciences), Shiwen Ni `[通讯]` (Artificial Intelligence Research Institute, Shenzhen University of Advanced Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究公共测试在代码生成中是否真正被模型利用，采用行为 flip 和隐藏状态分析，对不同测试类型（相关、混排、无关、断言、合成）在 Qwen2.5‑Coder‑7B 与 Qwen3.6‑27B 上进行评估。

**💡 创新点**

创新点在于将测试驱动提示与表示层 shift 相结合，区分模型是否真正利用测试语义，而非仅作提示扰动；同时通过合成测试实验探讨测试质量与数量对性能的影响。

**🔧 技术方法**

使用了贪心解码的 Qwen 语言模型，行为 flip 分析、McNemar 检验，线性探针预测隐藏状态是否编码提示条件，以及计算隐藏状态余弦距离来度量表示变化。

**📊 数据集**

实验数据集包括 HumanEval+、MBPP+（EvalPlus 隐藏测试）和 LiveCodeBench v6‑new（官方跑器），并在这些任务上对不同提示条件进行比较。

**📈 对比分析**

通过 pass@1/隐藏测试通过率、行为 flip 计数和线性探针准确率等指标比较不同测试条件；MBPP+ 在 Qwen2.5‑Coder 上可显著提升约7个百分点，HumanEval+ 几乎无改进，LiveCodeBench 上提升有限；Qwen3.6‑27B 在 LiveCodeBench 上显著提升基础通过率，但相关测试对性能的提升仍不稳定。

**⚠️ 局限性**

局限性包括仅评估 Qwen 家族模型；synthetic 测试缺乏参考解，可能引入噪声；隐藏状态分析为相关性，未揭示因果机制；仅使用一次贪心生成，未覆盖多样化生成；合成测试质量启发式简单，难以精准预测测试效果。

---

## 26. TraceCLIP: Recovering Local Semantics from Patch-to-CLS Contributions

**arXiv ID:** 2607.26107 | [PDF](https://arxiv.org/pdf/2607.26107v1)

**作者:** Xinran Liu `[一作]`, Sheng Zhong `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究冻结的CLIP模型内部的贡献路径，提出了一种训练‑free方法TraceCLIP，用于实现零样本语义分割。

**💡 创新点**

创新点在于发现patch‑to‑CLS贡献特征携带更强的局部语义信息，并将其转化为语义‑地理拓扑门来校准最终层注意力，从而显著提升分割效果。

**🔧 技术方法**

技术手段包括CLIP ViT编码器、贡献特征提取、语义‑地理拓扑门构造、注意力校准与融合、滑动窗口推理以及文本嵌入对齐。

**📊 数据集**

实验使用了八个零样本语义分割基准：VOC20/21、Context59/60、COCO‑Stuff/Object、Cityscapes 与 ADE20K。

**📈 对比分析**

与 GroupViT、TCL、MaskCLIP、ReCo、SCLIP、ClearCLIP、NACLIP 等训练‑free 对手比对，TraceCLIP 在无背景和有背景两种设置下分别平均提升 1.3–4.5 个点的 mIoU。

**⚠️ 局限性**

局限性包括：仍依赖 CLIP 内部贡献特征，无法直接处理更高分辨率图像；对多尺度和长程依赖建模有限；以及对模型内部的解释性仅停留在贡献路径层面，缺乏更深层次的可解释机制。

---

## 27. Choosing Where and How to Moderate: End-to-End Trade-offs in Filter Placement and Response Rewriting

**arXiv ID:** 2607.26200 | [PDF](https://arxiv.org/pdf/2607.26200v1)

**作者:** Mengya Hu `[一作]`, Curt Tigges `[通讯]` (Goodfire)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于最终用户体验的内容审核决策框架，比较在聊天系统中不同干预位置（输入、输出、两者）与干预动作（直接屏蔽 vs. 重写）的组合，以端到端的“有用性”和“有害暴露”指标评估整体效果；

**💡 创新点**

创新点在于：①从整体客户结果而非单独分类器准确率出发，量化不同审核配置的安全性与实用性；②在同一实验环境下对干预位置与动作进行系统比较；③结合重写技术实现对被屏蔽内容的恢复，同时保持安全门槛；④使用探测路由显著降低重写阶段延迟；

**🔧 技术方法**

技术手段包括：Azure AI Content Safety文本过滤器、T5 细化过滤器、Qwen3‑4B 与 GPT‑5 重写器、轻量级探针路由、基于 LLM 的判定器、自动化安全与相关性评测器；

**📊 数据集**

数据集：①内部手工构造、以安全等级分层的 1,250 条单轮对话（包含性、仇恨、暴力、自伤）；②公开 ToxicChat 数据集 5,654 条人类标注的单轮对话，采用 T5 过滤器与 GPT‑5 生成；

**📈 对比分析**

比较方法：在内部与公共基准上对四种配置（输入、输出、两者、输出+重写）进行端到端评估，主要指标为 Usefulness（展示且无害且相关的比例）和 Harmful Exposure（展示有害内容的比例）；结果显示：响应侧屏蔽在两种设置下实现最高 Usefulness，输入+响应能进一步降低 Harmful Exposure；引入重写后，可将响应侧被屏蔽的流量恢复至约 95% 的 Usefulness，同时保持与响应侧屏蔽相同的有害暴露率；探针路由使重写条件时间从 13.8 秒下降到 0.47 秒；

**⚠️ 局限性**

局限性：①内部基准为人工构造、伤害富集且非自然流量，无法直接估计生产环境比例；②实验仅在单轮英文对话与特定生成器、过滤器上验证，缺乏对多轮、跨语言或多代理系统的通用性；③重写效果评估多依赖自动评测器，未覆盖所有可能的安全细节丢失；④仅测量条件重写延迟，未提供完整用户可见延迟；⑤所有实验使用固定过滤器，未探究输入与输出独立过滤的影响；⑥公开与内部实现差异较大，结果仅为方向性验证而非精确量化。

---

## 28. NISPO: Open-source IUPAC name generation tool

**arXiv ID:** 2607.26113 | [PDF](https://arxiv.org/pdf/2607.26113v1)

**作者:** Nicholas T. Runcie `[一作]` (University of Oxford), Charlotte M. Deane `[通讯]` (University of Oxford)

**通讯引用:** 17680 | [OpenAlex ID](https://openalex.org/A5015572211)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基于 RDKit 的开源 Python 包 NISPO，用于自动生成 IUPAC 名称，并通过自我改进循环实现高精度命名；

**💡 创新点**

创新点在于利用 Codex（GPT‑5.5）驱动的编码代理和 OPSIN 的 round‑trip 验证，自动化完成从分子结构到规范命名的闭环学习；

**🔧 技术方法**

采用 Codex 作为编码代理、RDKit 进行分子处理、OPSIN 进行名称解析与验证，并在 Python 环境中实现整个工作流；

**📊 数据集**

训练集为 SureChEMBL 约 2.68 M 分子（已预筛选），测试集为 PubChem 约 103 M 分子；同时在 9,143 M MolJSON 分子上进行初始 bootstrap；

**📈 对比分析**

与商业工具 Lexichem 在 PubChem 上比较，NISPO 的 round‑trip 准确率为 98.1%，略低于 Lexichem 的 99.0%，并且在名称长度和运行时间上保持可接受水平；

**⚠️ 局限性**

生成的名称虽能通过 OPSIN round‑trip 验证，但不一定符合 IUPAC Blue Book 的首选命名规范，且代码质量较低，缺乏可维护性。

---

## 29. Guess Where You Go: Generative Next Point-of-Interest Recommendation in Amap

**arXiv ID:** 2607.26073 | [PDF](https://arxiv.org/pdf/2607.26073v1)

**作者:** Penglong Zhai `[一作]` (AMAP, Alibaba Group), Xin Li `[通讯]` (AMAP, Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种端到端的生成式下一地点推荐框架Gwhere，将语义标识（SID）与LLM结合；

**💡 创新点**

创新点在于使用对比学习的多模态SID生成器、在持续预训练中构造SID-位置-描述与行为序列语料，并引入曝光感知Kahneman‑Tversky优化以提升用户偏好对齐；

**🔧 技术方法**

技术包括残差量化对比学习的tokenizer、LLM（Qwen）持续预训练、SFT、RL（EAKTO）、多模态融合与注意力机制、PagedAttention等；

**📊 数据集**

使用公开的Foursquare-NYC/TKY/Gowalla-CA数据集以及大规模脱敏的Amap行业数据；

**📈 对比分析**

与多种基线（PRME、STAN、GETNext、LLM4POI、GNPR‑SID等）比较，Gwhere在Acc@1、CTR等指标上提升超过10%，在线A/B实验提升P‑CTR 5.83%、U‑CTR 6.20%；

**⚠️ 局限性**

局限包括对高阶模型规模导致推理延迟、对冷启动场景的支持有限，以及对多样化用户情境的深度自适应仍待完善。

---

## 30. Cross-Model Cross-Language AI Coding Agent Performance: Accuracy and Speed of Parallel CLRS Algorithms

**arXiv ID:** 2607.26083 | [PDF](https://arxiv.org/pdf/2607.26083v1)

**作者:** Shiqi Cheng `[一作]` (Massachusetts Institute of Technology), Alan Edelman `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 18903 | [OpenAlex ID](https://openalex.org/A5029673947)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对三种AI编码代理（Composer 2.0、GPT 5.4、Claude Sonnet 4.6）在C++、Python、Julia三种语言上生成并评测了12个CLRS经典算法（排序、图遍历、搜索）的并行实现。

**💡 创新点**

创新点在于将多模型、多语言的并行代码生成与性能评测结合，采用迭代式提示、功能正确性与运行时加速双重指标，揭示了不同模型、语言和算法间的性能差异。

**🔧 技术方法**

使用技术包括大语言模型代理、并行库（ProcessPoolExecutor、Polyester、OpenMP）、迭代提示流程、基准测评脚本和日志分析。

**📊 数据集**

数据集为CLRS标准算法实现与相应输入规模（排序/搜索N=10^4~10^6，图算法n=10^5），以及自定义与第三方库的基准实现。

**📈 对比分析**

比较方法是记录每个模型-语言-算法组合的首次正确性迭代次数、并行实现的最优运行时间及相对速度提升；结果显示Sonnet 4.6在7/15组合上获得加速，GPT 5.4仅保证正确性而无速度提升，Composer 2.0介于两者之间。C++在图算法上最稳健，Julia在某些排序与Bellman–Ford上表现突出。

**⚠️ 局限性**

局限性包括：正确性与性能之间的显著差距，模型对类型系统与并行细节的误判导致多轮迭代；评测仅覆盖CPU多线程、问题规模有限；未涉及GPU、异构计算；以及模型在复杂同步与内存访问优化上的推理不足。

---

## 31. Mergeable Model-Side Aggregation States for Long-Context Language Models

**arXiv ID:** 2607.26448 | [PDF](https://arxiv.org/pdf/2607.26448v1)

**作者:** Dachuan Song `[一作]` (George Mason University), Xuan Wang `[通讯]` (George Mason University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种模型侧聚合接口，利用固定预算的 HyperLogLog（HLL）状态在冻结LLM的前向过程中实时跟踪长文本中的集合信息，并直接将聚合结果提供给模型进行下游推理。

**💡 创新点**

创新点在于将可合并、固定大小的草图状态与LLM并行更新，消除了外部代码执行循环，保证了聚合结果的可预测误差和可合并性，同时在不扩展模型显存的前提下实现了长上下文的集合推理。

**🔧 技术方法**

核心技术包括 HyperLogLog 估计基数、JMLE 联合最大似然估计用于集合交并、相似度和包含度的读出；Count‑Min Sketch 用于点频率查询；以及在模型前向过程中并行更新与状态合并的实现。

**📊 数据集**

实验使用 Oolong‑Synth 数据集，对其中 174 条长记录窗口构造 3,969 个聚合‑推理任务。

**📈 对比分析**

在 Qwen‑3.6B 与 Gemma‑4B 上，SketchOps 在 3,969 个任务上达 89.0%（Qwen）/99.2%（Gemma）准确率，显著高于完整上下文、CoT、guided‑choice 等基线，且相较外部代码执行误差仅低 0.8–8.9个百分点，但生成 token 仅为 2.2，极大降低生成开销。

**⚠️ 局限性**

局限性：固定预算草图导致估计误差；总内存随维护的状态数量线性增长；目前仅支持基数、并集、Jaccard、包含度、分组基数、点频率等聚合目标；对需要绝对精确或更复杂算子的场景仍需外部执行或更大预算。

---

## 32. Multi-Agent Debate Strategies: Survey, Taxonomy, and Challenges

**arXiv ID:** 2607.26212 | [PDF](https://arxiv.org/pdf/2607.26212v1)

**作者:** Quim Motger `[一作]` (Universitat Politècnica de Catalunya), Xavier Franch `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过系统综述141篇论文，构建了三维(MAD参与者、交互、协议)的多代理辩论体系结构分类与正式表述；

**💡 创新点**

提出统一且可机器可读的三维分类法，解决术语混乱、设计决定隐式、跨研究比较困难的问题；

**🔧 技术方法**

采用系统文献综述方法、结构化搜索、向后向前雪球法，并对抽取的设计维度使用正式图论与时序表示；

**📊 数据集**

主要基于公开大语言模型公开数据集（如MMLU、BoolQ、HotpotQA、GSM8K、MATH、CLUTRR等）与对应任务的benchmark；

**📈 对比分析**

对比分析不同设计配置对准确率、成本、token使用等指标的影响，发现主流配置（静态全连、verbatim、短期记忆）效果最优，但未系统比较；

**⚠️ 局限性**

局限在仅覆盖文本任务、对新兴设计维度（如嵌入式通信、长程记忆）探讨不足，且随技术快速迭代可能需不断更新分类与评测标准。

---

## 33. Randomizing the Number of Centers in k-means++

**arXiv ID:** 2607.26202 | [PDF](https://arxiv.org/pdf/2607.26202v1)

**作者:** Vaclav Rozhon `[一作]` `[通讯]`, Vaclav Rozhon

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在预算平滑（k 从 K 到 2K-1 均匀取）模型下，证明 k-means++ 在常数概率下能获得 O(1) 的近似比。

**💡 创新点**

创新点在于将 k 范围按最优成本分块，结合 oversampling 与“浪费中心”分析，突破了传统 worst‑case O(log k) 的上界，得到在大多数 k 上常数近似的概率保证。

**🔧 技术方法**

使用的技术包括：块划分 Lemma、k-means++ 的概率采样分析、对浪费中心的期望上界、二项分布与马尔可夫不等式、以及 oversampling 结果的巧妙套用。

**📊 数据集**

本文为理论研究，没有使用具体数据集；所有证明均基于抽象的点集与成本函数。

**📈 对比分析**

与传统 k-means++ worst‑case 分析（O(log k) 近似）相比，该方法在预算平滑模型下可达到常数近似，概率至少 1/2；未给出实验性能数据，只给出了理论概率上界。

**⚠️ 局限性**

局限性：仅在常数概率下保证常数近似，未提供期望近似的结果；适用于均匀或类似分布的 k，未检验对其他分布或实际数据的鲁棒性；并未给出实际实验验证。

---

## 34. Probing the Origins of Reasoning Performance: Representational Quality for Mathematical Problem-Solving in RL vs. SFT Fine-Tuned Models

**arXiv ID:** 2607.26119 | [PDF](https://arxiv.org/pdf/2607.26119v1)

**作者:** Antyabha Rahman `[一作]` (University of New South Wales), Aishwarya Balwani `[通讯]` (St Jude Children's Research Hospital)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较RL训练和SFT训练的数学推理模型，分析其内部表示、层级重要性以及输出 token 的变异性。

**💡 创新点**

首次将线性探针、均值消融和 token CV 三种方法结合，揭示 RL 能产生更早、更清晰、更层级化的内部表示，并指出 token 变异受训练管线影响而非单纯训练方式决定。

**🔧 技术方法**

使用线性探针、均值消融（activation patching）、多样本 token 方差（CV）分析、synthetic problem 生成与采样、统计检验（Pearson r）等技术。

**📊 数据集**

基于 GSM8K、GSM8K‑Platinum、以及 1,000 条模板生成的 synthetic 数学题集；对比 DeepSeek‑Math 与 Olmo‑3 家族的 RL 与 SFT 版本。

**📈 对比分析**

通过层级探针准确率、AD 下降量与层深相关系数 r 以及 token CV 分箱进行对比；RL 版在探针准确率上平均高 8% 以上，深层重要性呈正相关；token CV 维持在 0.1–0.125 之间的 Olmo‑3 与 DeepSeek‑Math‑RL 在难度边缘表现出显著差异。

**⚠️ 局限性**

探针依赖样本平衡，线性可分假设可能限制解释；仅在数学推理任务上验证，未必适用于其他推理领域；token CV 受具体训练管线与奖励设计影响，结果难以泛化。

---

## 35. Incast-Free MoE Rate-Based Scheduling

**arXiv ID:** 2607.26340 | [PDF](https://arxiv.org/pdf/2607.26340v1)

**作者:** Evyatar Cohen `[一作]` (Technion), Isaac Keslassy `[通讯]` (Technion)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对大规模MoE模型在数据中心网络中出现的指数级in‑cast问题，提出了一种基于比例率归一化的全局公平调度框架，并在NIC层实现了子微秒级的速率更新。

**💡 创新点**

创新点在于将动态MoE流量矩阵通过最大行/列需求进行归一化，保证任何发送或接收端的累计到达率不超过线速，从而天然消除in‑cast；并结合NVIDIA DOCA PCC硬件速率控制实现高速、无通信开销的调度。

**🔧 技术方法**

使用技术包括Birkhoff‑von Neumann分解思想、流量矩阵归一化、HTSim仿真、非阻塞Fat‑tree网络、NVIDIA DOCA PCC硬件速率限制，以及AR/OPS/REPS负载均衡策略。

**📊 数据集**

实验数据集包括真实生产MoE路由记录（Qwen2/3的64×64与128×128矩阵）和基于DeepSeek‑V3参数的Zipfian合成流量矩阵。

**📈 对比分析**

通过与Round‑Robin+NSCC基线对比，利用CCT和链路利用率为评估指标；结果显示在所有网络规模与Skew度下，提出的比例率调度显著降低CCT（接近理想100%利用率）并避免指数级in‑cast。

**⚠️ 局限性**

局限性在于当前仅在仿真和理论层面验证，实际NIC实现仍需进一步落地；对极端Skew可能仍有限制；且128×128规模仅有单个样本，未覆盖多租户或动态拓扑变化场景。

---

## 36. A Reference-Free Score for Detecting Silent Reasoning Failures in Large Language Models

**arXiv ID:** 2607.26102 | [PDF](https://arxiv.org/pdf/2607.26102v1)

**作者:** Vivek Shukla `[一作]` (Allenhouse Institute of Technology), Mehul Kumar Das `[通讯]` (Allenhouse Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了无参考的 Reasoning–Answer Faithfulness Score（RAFS）来评估数学推理链的可信度，能检测推理错误与答案不一致的“silent failures”。

**💡 创新点**

创新点在于将步骤有效性、推理‑答案支持、反事实敏感性、答案共识与条件推理稳定性四个维度整合成非补偿的参考自由评分，解决传统仅看答案精度的缺陷。

**🔧 技术方法**

使用了 LLM 批判器、符号算术检查、NLI 归纳、可解释的反事实干预、采样共识与条件稳定性评估，并通过几何平均聚合与校准得到 RAFS。

**📊 数据集**

实验数据集为 GSM8K 与 MATH 两大数学推理基准，涵盖多步算术、代数、几何等题型。

**📈 对比分析**

通过与准确率、答案自一致性、步骤评分、ROSCOE 等基线比较，预计在检测 silent failures 的 AUPRC 和召回率上优于现有方法；具体性能待实验验证。

**⚠️ 局限性**

局限包括对步骤分割与验证器的手工设计依赖、干预覆盖率有限、仅针对数学推理场景、对模型偏差和生成策略的鲁棒性尚未充分评估。

---

## 37. Do Methods Support the Claims? Intra-Paper Verification for Peer Review

**arXiv ID:** 2607.26066 | [PDF](https://arxiv.org/pdf/2607.26066v1)

**作者:** Ranjitha Shivaprasad Ballakuraya `[一作]` (University of West Florida), Ashok Srinivasan `[通讯]` (University of West Florida)

**通讯引用:** 11631 | [OpenAlex ID](https://openalex.org/A5091299001)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个内部论文声明验证框架，用LLM提取创新声明、检索方法证据并评估方法是否支持声明，从而辅助同行评审。

**💡 创新点**

创新点在于从论文内部而非仅比较文献来验证声明与方法的匹配，构建基于人类评审的评价标准并生成结构化评审反馈。

**🔧 技术方法**

采用大规模语言模型（LLM）进行信息抽取与评估，使用BERTScore进行文本相似度评估，并利用提示工程设计评价流程。

**📊 数据集**

使用了182份ICLR 2025会议论文及其人类评审文本，构建了一个平衡的接受/拒绝子集进行实验。

**📈 对比分析**

通过统计检验（p < 0.001, Cohen's d = 1.17）和BERTScore对比，发现LLM生成的评审与人类评审高度一致，尤其在创新性问题上表现最佳。

**⚠️ 局限性**

局限性包括：依赖LLM的抽取和推理质量，可能忽略细微技术细节；仅在ICLR 2025数据集上验证，缺乏跨会议或跨学科的泛化；评估指标主要是文本相似度，未深入探讨实际审稿质量或决策影响。

---

## 38. Towards Trustworthy Embodied Intelligence: A Systems Framework and Graded Trustworthiness Levels

**arXiv ID:** 2607.26121 | [PDF](https://arxiv.org/pdf/2607.26121v1)

**作者:** Xinyu Yang `[一作]` (Xspark AI), Wenbo Ding `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对可信的具身智能进行系统性综述，提出四层（模型、系统、证据、部署）框架和 T0–T5 可信等级，并归纳实现这些层的关键机制与评价方法。

**💡 创新点**

创新点在于把模型、系统、证据与部署四层互相依赖的关系与交互进行系统化梳理，提出可量化的可信等级层次（T0–T5），并给出跨层次的证明与治理策略。

**🔧 技术方法**

主要采用理论分析、案例对比、跨学科综述以及对现有安全/可信AI标准、自动驾驶安全框架等的借鉴，构建框架与等级定义。

**📊 数据集**

本工作为综述性论文，无实验数据集；所引用的研究基于公开的 benchmark、仿真平台和工业案例。

**📈 对比分析**

通过与传统可信AI标准、自动驾驶 SAE L0–L5、机器人安全规范等的对比，展示了该框架在覆盖范围、系统治理与评估维度上的全面性；未给出具体数值性能，而是提出了评估维度与级别定义。

**⚠️ 局限性**

限制：缺乏定量可信度指标与统一评估流程，需要根据具体领域制定详细的阈值与验证步骤；层级划分仍需与行业标准对齐，当前更侧重理论框架与概念性说明。

---

## 39. ProFlow: RL-Driven and Performance-Aware Proactive Flow Placement in Datacenter Networks

**arXiv ID:** 2607.26231 | [PDF](https://arxiv.org/pdf/2607.26231v1)

**作者:** Sourya Saha `[一作]` (City University of New York), Saptarshi Debroy `[通讯]` (City University of New York)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

ProFlow 是一个基于离线强化学习的主动流放置框架，在多租户数据中心网络中利用分布式遥测的前驱信号，提前检测并迁移性能敏感流，防止拥塞导致吞吐下降。

**💡 创新点**

创新点在于：① 通过离线训练的 RL 模型学习拥塞前驱信号，实现在拥塞发生前做出迁移决策；② 设计了稳定性门控制切换频率，避免过度迁移；③ 将主动放置与传统被动迁移在同一实验平台上对比，展示“预防式”拥塞管理的优势。

**🔧 技术方法**

技术手段包括：离线强化学习（模型学习 + 拟合 Q 迭代 + Dyna 规划）、分布式遥测收集、SDN 控制器通过 OpenFlow 安装流表规则，以及基于 Python + PyTorch 的实现。

**📊 数据集**

使用了 NSF FABRIC 物理实验室的叶-聚合两层拓扑，构造了 10 个多租户拥塞场景与 2 个蒸汽测试场景的实验数据集，用于训练和评估。

**📈 对比分析**

与静态放置和基于令牌桶溢出阈值的被动迁移基线对比，ProFlow 在 10 个场景中平均吞吐量提升约 40%，且提前约 34 秒做出迁移决策，显示出显著性能优势。

**⚠️ 局限性**

主要局限包括：目前仅保护单条性能敏感流，未处理多流协同放置；依赖离线训练数据，需保证场景覆盖；在全拥塞或极端流量模式下，目标切换选择仍可能不最优。

---

## 40. Try Again, Don't Look Back: Blind Resampling Outperforms Self-Repair in Small Code Models

**arXiv ID:** 2607.26117 | [PDF](https://arxiv.org/pdf/2607.26117v1)

**作者:** Yuvraj Verma `[一作]` `[通讯]` (Independent Researcher), Yuvraj Verma (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了小规模代码生成模型（1.5B、3B、7B）在回显自我修复与盲重采样条件下的性能，通过placebo‑controlled实验比较四种重试策略。

**💡 创新点**

创新点在于引入盲重采样作为对照，揭示自我修复中的“锚定”偏差，证明执行反馈的内容对提升效果几乎无贡献，并说明模型规模对锚定成本的影响。

**🔧 技术方法**

采用 Qwen2.5‑Coder 在温度0.8下进行重采样，使用嵌套评估、McNemar检验、Bootstrap 区间、Holm 校正等统计方法，对 MBPP+ 与 HumanEval+ 基准进行系统对比。

**📊 数据集**

主要使用 MBPP+（378 任务）和 HumanEval+（164 任务）作为代码生成与测试数据集。

**📈 对比分析**

通过将四种条件与盲重采样在相同 token 预算下对比，并在 k≤8 的迭代中记录通过率，发现盲重采样在 1.5B、3B 模型下准确率最高且 token 成本最低，7B 时与最佳条件持平，表明自我修复对性能提升不显著。

**⚠️ 局限性**

研究局限在于仅关注函数级代码合成、单一 benchmark 族、词法检索与固定反思提示，且跨任务检索无效，无法确认结论是否适用于更大规模或不同架构的模型。

---

## 41. Self-Adaptive Learning and Model Predictive Control for Tracking Unknown Dynamics with No Regret

**arXiv ID:** 2607.26370 | [PDF](https://arxiv.org/pdf/2607.26370v1)

**作者:** Atharva Navsalkar `[一作]` (University of Michigan), Vasileios Tzoumas `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种自适应在线学习控制方法，实时学习多种预测模型并自适应选择以跟踪未知动态目标。

**💡 创新点**

将专家学习算法与模型预测控制结合，利用自监督一次性学习、多模型预测和基于bandit的自适应选择，实现零均方回报并对动态切换和学习误差鲁棒。

**🔧 技术方法**

采用随机傅里叶特征逼近目标动态、在线最小二乘/梯度下降更新、Bandit（EXP3等）选择器以及模型预测控制（MPC）规划。

**📊 数据集**

在Gazebo仿真和Crazyflie硬件实验中，使用勒姆尼斯卡、正弦、星形、随机步行、随机对抗等多种目标轨迹作为测试数据。

**📈 对比分析**

与基于随机梯度、核回归、神经网络等现有在线学习控制方法比较，实验显示在所有目标类型下平均跟踪误差和预测误差均低于基线，平均回报接近无遗憾。

**⚠️ 局限性**

对非切换误差仍有残差，依赖预设预测窗口和特征参数，无法自适应新类型预测器的生成，且在极快切换或高噪声环境下性能可能下降。

---

## 42. Exploring Structures in Physics Problems: Can AI Agents Discover Statistical Mechanical Mappings?

**arXiv ID:** 2607.26367 | [PDF](https://arxiv.org/pdf/2607.26367v1)

**作者:** Wanyu Zhao `[一作]` (University of Illinois), Wanbing Zhao `[通讯]` (Rice University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了使用大语言模型（LLM）驱动的智能体在统计物理中发现映射（将原始配分函数转化为已知可解析模型）的可行性，并设计了一个基于提议‑验证‑修订循环的简易映射智能体。

**💡 创新点**

创新点在于：①首次将统计物理映射任务正式定义为AI代理问题；②构建了包含六个 1D/2D Ising 任务的 Tier 1 基准（BenchMark），涵盖传递矩阵、可消去杂质、平面/Pfaffian 等四种可解析结构；③系统评估了多款 LLM 在该任务上的分类、数值验证与复杂度自我报告能力，并揭示了数值验证不足以保证映射正确性的失败模式。

**🔧 技术方法**

主要技术包括：LLM 生成代码的提议；基于 Python 的全枚举数值验证；对代码做 AST 语法检查；对复杂度标注进行正则匹配的诚实性审核；以及通过多轮反馈实现的迭代修订。

**📊 数据集**

使用的“数据集”是自定义的六个配分函数任务（P01–P06），每个任务有两种自然语言描述（原版与改写），配合多达八种商业/开源 LLM（Claude Haiku 4.5、Sonnet 4.6、Opus 4.7、DeepSeek V4‑Flash、DeepSeek V4‑Pro、Gemini 3.1‑Pro、GPT‑5.4‑mini（低/高推理））。

**📈 对比分析**

比较方法：在每个 LLM 上执行至多 5 轮修订，记录首轮可解析类别准确率（TC₀）、首轮数值通过率（NM₀）、最终诚实度评分（O）、迭代完成轮数（k*）、总耗时与成本。结果显示 Opus 4.7 在分类与数值通过率上领先，且大多数模型在首轮即可通过；但数值通过并不能完全纠正错误的复杂度标注，表明仅靠数值验证无法评估映射的真正可解析性。

**⚠️ 局限性**

局限性包括：①数值验证仅能修复实现错误，无法纠正错误的映射类别或复杂度断言；②当前基准规模仅到 n ≤ 16，缺乏对大规模问题的评估；③依赖手工编写的任务描述与基准，难以自动化扩展到更广泛的 Tier 2/3 任务；④缺乏符号或图结构级别的多层验证栈，导致无法完全捕捉映射的物理正确性。

---

## 43. Formally certifying number field invariants

**arXiv ID:** 2607.26230 | [PDF](https://arxiv.org/pdf/2607.26230v1)

**作者:** Alain Chavarri Villarello `[一作]` (VU Amsterdam), Sander R. Dahmen `[通讯]` (VU Amsterdam)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在Lean 4中实现了对数域基本不变量（判别式、签名、单位群模 p 次幂、类群及其阶）的形式化验证；通过外部CAS生成证书并在Lean中检查以保证计算正确性。

**💡 创新点**

提出了基于可计算伪余数序列的判别式求法、在实闭域上证明Sturm定理并用整数算术计数实根、以及在Lean中构造完整的类群结构证明框架；同时提供了高效的理想算术、素理想判别及素因子基的证明体系。

**🔧 技术方法**

使用了Lean 4、Mathlib 4、可计算多项式表示、伪余数序列、Resultant、Sturm序列、实闭域理论、PrimeSieve、Ordnode二叉树、以及SageMath脚本生成证书。

**📊 数据集**

在LMFDB数据库的数域条目（包含判别式、签名、类数和类群结构）上验证了数百个实例。

**📈 对比分析**

相较于传统CAS，验证过程耗时显著但可保证正确性；对高阶数域（如10次）判别式的验证曾在之前不可行；数值检验显示在现代硬件上可在数秒内完成。

**⚠️ 局限性**

限制包括：对大阶数域和高判别式的计算仍受计算资源限制；类群结构证明依赖于已知的素基与关系，需大量CAS计算；仅在未假设GRH的情况下使用Minkowski界，因而无法在更大范围内高效完成。

---

## 44. AI Security Priorities: A Field-Wide Agenda

**arXiv ID:** 2607.26069 | [PDF](https://arxiv.org/pdf/2607.26069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 45. Misalignment Has a Personality: A Big Five Account of Emergent Misalignment

**arXiv ID:** 2607.26389 | [PDF](https://arxiv.org/pdf/2607.26389v1)

**作者:** Hasibur Rahman `[一作]` (Northeastern University), Smit Desai `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过从模型激活中提取可校准的 Big Five 人格向量，测量并揭示细化训练导致的模型误导现象。

**💡 创新点**

将“误导”重新表述为多维人格偏移，用可解释的梯度向量替代单一标量方向，并验证其跨模型、跨语料的可转移性。

**🔧 技术方法**

采用三层级 Trait Modulation Keys 的对比提取、均值差分向量、投影评分、Spearman 相关、AUC 与主成分分析等技术进行分析。

**📊 数据集**

主要使用 Qwen2.5-7B‑Instruct、Llama‑3.1‑Nemotron‑Nano‑8B 两款公开模型；Trait Modulation Keys 生成的梯度对比数据；BIG5‑CHAT 评估集；以及八类误导数据集（evil、sycophancy、hallucination、insecure code、math、medical、opinion、GSM8K）。

**📈 对比分析**

通过零样本 AUC、交叉验证准确率、分层稳定性、相关系数等指标进行比较，结果显示每条向量的 AUC ≥0.86，误导签名在不同模型和任务上相关系数 ≥0.94，微调后表现与数据签名高度一致。

**⚠️ 局限性**

仅在两款 7–8B 英文模型上验证，缺乏对更大模型或多语种的评估，且人格向量仍受文本表面词汇影响，未能完全证明与人类人格结构的对应关系。

---

## 46. ForgetBench: Benchmarking Forgetting Dynamics of Long-Term Parametric Memory in Language Models

**arXiv ID:** 2607.26455 | [PDF](https://arxiv.org/pdf/2607.26455v1)

**作者:** Ruxi Gu `[一作]` (University of Science and Technology of China), Wei Wang `[通讯]` (University of Science and Technology Beijing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 ForgetBench 基准，用以系统评估大型语言模型在持续知识编辑过程中的长期记忆保持与遗忘动态。

**💡 创新点**

创新点在于：①将记忆评估从单步编辑扩展为时间序列的连续编辑；②设计两种互补的 QA 构造范式（概念式 QA 与情景式 QA），区分原子事实与结构化关系知识；③构建统一的时空评估框架与一组量化指标（编辑成功率、保持率、遗忘曲线、半衰期、AUF、速度、跨实例稳定性等），可直观描绘忘记曲线。

**🔧 技术方法**

使用技术包括：持续编辑框架（逐步插入知识）、基于大模型的 QA 生成（多轮互动图构造+自然语言转化）、遗忘曲线建模与统计分析（平滑、插值、半衰期计算）、跨实例一致性度量。

**📊 数据集**

数据集：概念式 QA 采用 2500 条随机生成的属性对；情景式 QA 通过模拟 20 名 Agent 进行 50 轮互动生成 56095 节点/96082 边的知识图，随后抽样 2500 子图（约 20 节点）生成 6431 个 QA 对。

**📈 对比分析**

与四种主流编辑方法（MEMIT、WISE、AlphaEdit、UltraEdit）在四大 LLM 上（Llama-3/3.1 8B、Qwen-2.5 7B、DeepSeek-R1 7B）对比实验，结果显示：AlphaEdit 在保持率上表现最好，但泛化与流畅度大幅下降；MEMIT 记忆消失严重；WISE 具备流畅与泛化但记忆保持差；UltraEdit 记忆保持极低。情景式 QA 明显缓解了保持-泛化折衷，提升了所有方法的可观召回，但并未根本解决参数层面的遗忘。

**⚠️ 局限性**

局限性：①基准仅使用合成/半合成数据，缺乏真实世界知识演化的复杂性；②只评估连续编辑方式，未覆盖其他记忆更新机制（如持续预训练、强化学习等）。

---

## 47. Rad-JEPA 3D: Radiology Joint-Embedding Predictive Model for 3D Computed Tomography

**arXiv ID:** 2607.26196 | [PDF](https://arxiv.org/pdf/2607.26196v1)

**作者:** Quoc-Huy Trinh `[一作]` (Northwestern University), Ulas Bagci `[通讯]` (Northwestern University)

**通讯引用:** 10326 | [OpenAlex ID](https://openalex.org/A5030188696)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了 Rad-JEPA 3D，一种针对 3D CT 的自监督联合嵌入预测框架，用于学习可迁移的体积特征。

**💡 创新点**

创新点在于：① 设计 H‑Mamba 混合编码器，融合 Mamba 状态空间模型与 Grouped‑Query Attention 并通过轻量级路由器动态选择分支；② 引入 Hidden States Orthogonal Regularization（HSOR）在层级上对齐学生–教师隐藏状态、减少特征冗余并施加权重正交约束。

**🔧 技术方法**

技术实现包括：自监督联合嵌入预测（JEPA）、EMA teacher、Mamba 状态空间模型、Grouped‑Query Attention、轻量级路由器、Cube Tokenizer、75% 随机遮挡、HSOR、以及 1‑层梯度正则化与 load‑balancing。

**📊 数据集**

使用公开 M3D 数据集进行预训练，约 120,000 张 CT 扫描；下游任务采用 M3D‑CAP（图像‑文本对）与 M3D‑VQA（VQA 语料）进行评估。

**📈 对比分析**

通过冻结‑encoder k‑NN 探测、闭式与开放式 VQA、以及 SpatialMed 空间推理基准进行对比；在 Plane/Phase/Organ 任务上分别取得 98.1%/48.3%/19.6% 的 k‑NN 准确率，VQA 平均准确率达 81.66%，SpatialMed 平均得分 58.16%，在仅 4B 参数规模下与更大模型保持同等或更优性能。

**⚠️ 局限性**

局限性包括：对 Direction（DIR）子任务的性能略逊于 Med3DVLM，极端遮挡比例（85%）导致性能下降；模型对跨轴细节的捕捉仍不完美，且主要验证于 CT 数据，跨模态泛化尚待进一步探索。

---

## 48. Global Pass Barriers Without Per-Resource RHI Tracking: A Cross-Vendor Study with Blade

**arXiv ID:** 2607.26506 | [PDF](https://arxiv.org/pdf/2607.26506v1)

**作者:** Dzmitry Malyshau `[一作]` `[通讯]`, Dzmitry Malyshau

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本研究分析了在Vulkan、WebGPU与Metal三种显式GPU API中，如何通过自动或显式的全局Barrier来管理资源同步，探讨是否可以在不维护每个资源状态的前提下完成同步。

**💡 创新点**

创新点在于提出仅依据pass‑kind（渲染/计算/传输）推导Barrier的源/目标阶段与访问掩码，从而完全消除资源跟踪，同时通过统一图像布局减少驱动层的冗余缓存操作。

**🔧 技术方法**

技术手段包括：Vulkan API的Barrier声明与记录；对RADV/Anv驱动源码的分析以推断Barrier的实际命令与缓存操作；RenderDoc捕获对比不同策略下的Barrier请求；对GPU时间和主机记录+提交时间进行统计；对不同Barrier粒度与Scope进行实验对比。

**📊 数据集**

实验使用六种自定义工作负载（单独compute、compute链、单独render、render链、compute/graphics交替以及混合链），并在六台GPU上运行：RTX 5070、RX 7900 XT、Raphael iGPU、Radeon 780M、Intel Xe (RPL‑U)以及Apple M3（Metal）。

**📈 对比分析**

比较方法为：对齐不同Barrier策略下的Barrier请求和最终输出哈希，测量GPU span与主机录制+提交时间的百分比差异。结果显示：在绝大多数设备上，移除冗余全局Barrier可将GPU span降低5%–30%；将Scope缩小为pass‑kind可在某些设备上进一步提升1%–10%；但在部分设备或特定工作负载上无显著改进。

**⚠️ 局限性**

局限性包括：只评估单队列Vulkan和Metal；未针对多队列或Tile‑based GPU；未单独测量资源级Barrier的成本；驱动实现差异导致结果在不同硬件间不一致；以及未分析深层驱动内部缓存刷新与命令排程细节。

---

## 49. FinCacheServe: Dependency-Consistent Answer Reuse for Cost-Efficient RAG Serving over Mutable Enterprise Documents

**arXiv ID:** 2607.26076 | [PDF](https://arxiv.org/pdf/2607.26076v1)

**作者:** Lingteng Zeng `[一作]` (Chinese University of Hong Kong), Yifan Jin `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 80 | [OpenAlex ID](https://openalex.org/A5108527831)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一个基于依赖一致性的答案重用系统，缓存已经生成的检索增强生成（RAG）答案，并在 GPU 调用前对财务文件版本、证据指纹、工具指纹、模型和解码配置等依赖进行严格检查，从而显著减少 LLM 调用次数。

**💡 创新点**

创新点在于：① 引入完整的财务签名+证据+工具+模型+版本依赖合同，实现对可变企业文件的安全答案重用；② 结合容量感知的 admission/eviction 策略和逆向索引失效机制，实现高命中率且无陈旧输出；③ 在多模型、多规模实验中验证该机制在节能、延迟和 SLO 方面的显著优势。

**🔧 技术方法**

采用 vLLM+Qwen2.5 LLM、FAISS 向量检索、检索缓存、答案缓存、metadata plane 与逆向索引、SQLite/WAL 事务后端、在线 CacheOpt、LRU/LFU/Utility‑aware 等技术栈。

**📊 数据集**

使用 SEC EDGAR 公开财务文件生成的问答工作负载，包括公司、期间、文档范围、查询家族等多维度信息，覆盖流动性、收入、利润率、负债、风险因素、现金流、比例分析等。

**📈 对比分析**

与无缓存、检索缓存、TTL、语义缓存、精确答案重用、版本语义缓存、实体周期语义缓存、Grounded‑style、财务签名等基线进行对比。7B 全轨迹中跳过 53.27% LLM 调用，32B 3 个种子跳过 53.31%；容量 64 条目时 Utility‑aware 跳过 36.66%，与离线 Belady Oracle 差距 0.56%；在 2s SLO 下获得 53.31% 依赖新鲜 goodput，Wh 成本 2.80/1000 次，较版本语义缓存下降 44.3%。

**⚠️ 局限性**

局限性包括：仅在公开 SEC 文档和 Qwen2.5 模型上验证；缺乏多地域、分布式缓存一致性和真实业务流量多样性的评估；能源估算基于板级功耗假设；未验证答案事实准确性；对其他行业文件或更复杂工具链的泛化需要进一步研究。

---

## 50. (EC)2: Event-Centric Explainability for Cybersecurity Through Multi-Agent LLM Investigations

**arXiv ID:** 2607.26201 | [PDF](https://arxiv.org/pdf/2607.26201v1)

**作者:** Neta Kirmayer `[一作]` (Ben-Gurion University of Negev), Rami Puzis `[通讯]` (Ben-Gurion University of Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种事件中心的可解释性框架（EC²），通过多代理、基于假设的调查自动生成基于可验证证据的网络事件解释。

**💡 创新点**

创新点在于将可解释性从模型决策转向事件本身，采用多代理协同、检索增强生成（RAG）和迭代假设检验，实现结构化、可验证的说明。

**🔧 技术方法**

核心技术包括大型语言模型（GPT‑5‑mini）驱动的多代理架构、检索增强生成（RAG）与结构化知识图谱、关系型数据库查询，以及AutoGen框架的工作流调度。

**📊 数据集**

使用CSE‑CIC‑IDS2018网络安全数据集，构建网络拓扑知识图谱与流量数据库，以模拟企业网络环境进行评估。

**📈 对比分析**

通过与现有LLM解释框架eX‑NIDS的对比以及消融实验评估，EC²在攻击类型识别、实体识别与解释完整度三项指标上均显著优于基线（平均提升约1.7–2.0分），且在边界区域重新分类时能显著提高检测召回率与精确率。

**⚠️ 局限性**

主要限制是每条事件调查耗时约17分钟，且依赖完整的多源数据检索，限制了大规模并发警报的实时处理能力。

---

## 51. GPT-Red: Automated Red Teaming via Self-Play at Scale

**arXiv ID:** 2607.26115 | [PDF](https://arxiv.org/pdf/2607.26115v1)

**作者:** Eric Wallace `[一作]` (OpenAI), Kai Chen `[通讯]` (OpenAI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出并训练了一款名为GPT‑Red的自动红队代理，通过自我对弈与多样化防御模型的对抗，旨在发现并训练LLM对提示注入与内容政策越狱的鲁棒性。

**💡 创新点**

创新点在于：①将红队任务规模化为海量、异质的RL环境；②设计可扩展的agentic harness，使攻击者能在推理时有效搜索与迭代；③采用多防御模型的自我对弈，使攻击者不断演化，形成“自我提升”的安全训练闭环。

**🔧 技术方法**

技术方法包括：大规模强化学习（RL‑Post‑Training）、自我对弈算法、状态化攻击 harness、工具调用与文件编辑接口、程序化奖励与违规检测、以及多模态（文本+图像）注入策略。

**📊 数据集**

数据集主要是：从现有能力任务（如GPT‑5.6后训练任务）转化的红队RL环境；直接与间接提示注入、跨模态注入、内容政策越狱等；此外使用I​PI 2025挑战集和人类手工注入数据作为验证。

**📈 对比分析**

与人类红队者、传统冻结LLM红队方法对比，GPT‑Red在自我对弈生成的攻击上攻击成功率提高约2倍，且在I​PI 2025数据集上达成近100%鲁棒率；在现实系统（AI售货机）中成功实现多目标攻击。

**⚠️ 局限性**

局限性包括：多模态与多轮攻击的训练仍不足；对极端或未知攻击模式的泛化尚未充分验证；攻击者的探索空间受限于预设的注入点与威胁模型，可能无法覆盖所有真实场景。

---

## 52. Multi-Objective Compliance-Integrated Coevolution For Simulated And Real-World Deployment Of Multi-Robot Marine Autonomy

**arXiv ID:** 2607.26279 | [PDF](https://arxiv.org/pdf/2607.26279v1)

**作者:** Everardo Gonzalez `[一作]` (Oregon State University), Kagan Tumer `[通讯]` (Oregon State University)

**通讯引用:** 4318 | [OpenAlex ID](https://openalex.org/A5084748531)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 MMOCIC 框架，将协同进化产生的协作行为与预定义的合规行为通过可调权重混合，实现海上多机器人任务进度与法规遵从的平衡。

**💡 创新点**

创新点在于将学习与合规分离，允许在训练时仅优化协作行为，再通过零训练调整合规权重实现多目标权衡，并结合控制障碍函数保障硬件安全。

**🔧 技术方法**

采用协同进化算法（CCEAs）训练神经网络行为、行为基自治（行为混合与标量化）、控制障碍函数（CBFs）过滤冲突控制，以及可调权重的行为混合层。

**📊 数据集**

使用随机生成的水手位置数据（20 次随机配置）进行模拟训练，硬件实验采用真实海域中的 8 车/4 车/12 车部署，实验数据包含救援次数、时间、距离、碰撞与近距离接触记录。

**📈 对比分析**

通过与手工设计的基线行为进行对比，评价指标包括救援数量、完成时间、总行驶距离、碰撞/近距离接触次数；实验显示 MMOCIC 与基线性能相当，且在 8 车实验中行驶距离更短，体现更高效率。

**⚠️ 局限性**

局限性包括：权重设置在训练后固定，缺乏在线自适应；硬件安全仍依赖 CBFs 处理极端冲突；对环境扰动（如河流流速）的鲁棒性有限；训练未包含破坏性代理，导致对未知冲突情境的泛化受限。

---

## 53. When Synthetic Users Fail: A Cross-Domain Benchmark of LLM-Simulated Human Survey Responses

**arXiv ID:** 2607.26348 | [PDF](https://arxiv.org/pdf/2607.26348v1)

**作者:** Zihan Chen `[一作]` (Stevens Institute of Technology), Lei Nico Zheng `[通讯]` (University of Massachusetts Boston)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在两域（美国社会态度与63国跨文化价值）中，用两种提示格式对四个大型语言模型（Claude Haiku、Claude Sonnet、Llama‑3.1‑8B、Llama‑3.1‑70B）进行合成用户模拟，评估其在个体预测、聚合分布以及群体结构上的表现，并提出统一的验证框架。

**💡 创新点**

创新点包括：① 构建跨域、跨模型的标准化评测协议；② 引入基准对齐的个体级度量（与人口查找、逻辑回归、随机森林对比）和“刻板化指数”（η²差值与Cramér’s V），量化模型对身份的过度决定性；③ 结合决策影响分析，说明过度刻板化如何导致错误的分段决策。

**🔧 技术方法**

技术手段：使用两种提示格式（单答案、概率分布）在Claude和Llama两大模型家族上进行推理；计算准确率、MAE、EMD、JS、η²、Cramér’s V、log‑loss、Brier分数等指标；采用配对bootstrap 95%置信区间对比非LLM基准；并分析无效/拒绝率、格式稳定性与模型规模对结果的影响。

**📊 数据集**

数据集：美国General Social Survey（GSS）2016–2024波次（10个态度题，约1.5万受访者）和World Values Survey（WVS）第7波（63国，16个价值题，约9.1万受访者）。

**📈 对比分析**

比较方法：将LLM输出与四类非LLM基准（人口边际、人口查找、逻辑回归、随机森林）在同一受试者上配对比较；对个体层面使用准确率、MAE、EMD和log‑loss；对聚合层面使用JS；对群体层面使用η²差值和Cramér’s V；对决策层面计算间段差距放大因子、错误目标率和伪分段率。性能结果表明：LLM在个体层面不胜基准（尤其在价值题上落后20+个百分点），聚合层面表现相当；刻板化指数普遍为正，说明模型过度刻画身份与态度的关系。

**⚠️ 局限性**

局限性：① 仅评估两种提示方式和四个模型，未涉及更丰富的persona构建、少量示例或微调；② 数据为近年GSS/WVS且未加权，可能对罕见子群不够稳健；③ 仅在两大调查问卷上验证，未检验其他文化或行为场景；④ 结果对模型版本更新、提示变体或训练数据变化可能有差异。

---

## 54. Large Language Models for Software Engineering Diagrams: A Systematic Review of UML and ER modelling

**arXiv ID:** 2607.26100 | [PDF](https://arxiv.org/pdf/2607.26100v1)

**作者:** Mojdeh Rahmanian `[一作]` (Edinburgh Napier University), Yanchao Yu `[通讯]` (Edinburgh Napier University)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5086863890)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统综述了2023‑2025年间64篇论文，聚焦大语言模型（LLM）在 UML 与 ER 图建模中的应用与评估。

**💡 创新点**

创新点在于首次将 LLM 在软件工程图形建模的研究进行系统归纳，揭示任务覆盖、技术方法、评估缺口以及多视图一致性和语义可靠性等未被充分讨论的关键议题。

**🔧 技术方法**

主要技术包括 GPT 系列（GPT‑3.5/4）、大语言模型的零/少/多步提示、参数高效微调（LoRA/QLoRA）、检索增强生成（RAG）、多代理架构以及与传统验证工具的组合。

**📊 数据集**

数据集多为研究自制或公开的 UML/ER 语料库，覆盖自然语言说明、PlantUML/Mermaid 代码、XMI、JSON 等格式，但跨论文共享的基准极少，缺乏统一的数据集。

**📈 对比分析**

比较方法呈现多样化：语法有效性、精确率/召回率/F1、语义正确性、人类评审、图编辑距离等指标。总体表现相对有潜力，但因评估不统一、实验规模小，难以形成客观的性能对比。

**⚠️ 局限性**

主要局限包括：语义错误/幻觉、对提示敏感、模型可重复性差、缺乏统一基准和评估标准、行为/数据建模覆盖不足、缺乏多视图一致性检查，以及工业级可扩展性与集成度低。

---

## 55. Learning the Word Problem: Geodesic Lengths and Cryptographic Applications

**arXiv ID:** 2607.26241 | [PDF](https://arxiv.org/pdf/2607.26241v1)

**作者:** Elisabeth Fink `[一作]` `[通讯]` (Middlesex University), Elisabeth Fink (Middlesex University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 WPNet，一种将未约简单词映射为动态图并通过图神经网络学习等价性嵌入的架构，能够快速近似求解 Baumslag–Solitar 群 BS(1,2) 和随机 Artin 群的单词问题与测地长度，并在此基础上实现对 Wagner‑Magyarik 公钥加密系统的神经网络攻击。

**💡 创新点**

创新点在于：① 使用动态图构造和 GINE 消息传递，使网络能够在连续嵌入空间聚类代数等价元素；② 采用对比学习（InfoNCE）与 Triplet Margin Loss，逼近测地长度与等价判别；③ 在高词长（达 20,000）下仍保持高准确率；④ 将该模型直接用于破解实际加密协议，展示了对后量子密码学的潜在威胁。

**🔧 技术方法**

技术方法包括：Bi‑LSTM 对子词编码；Gated Graph Isomorphism Network (GINE) 消息传递；注意力聚合；对比学习（InfoNCE）与三元组损失；交叉熵分类用于测地长度预测；以及自适应数据生成和动态 triplet 采样。

**📊 数据集**

数据集为自生成的合成单词，对 BS(1,2) 与随机 Artin 群（生成 101 个生成元，m_{i,j} 为 2/3 或 ∞ 的随机 Coxeter 图）产生的 8192 个 triplet；词长范围 20–100（训练）到 5,000（测试），并使用针对 Wagner‑Magyarik 的随机加密/解密实例进行攻击评估。

**📈 对比分析**

评估通过三项任务：① 单词嵌入区分度（正负相似度阈值 0.2），在 BS(1,2) 上 90%+准确率到 5,000 词长；② 测地长度预测分类精度，BS(1,2) 约 80% 正确率（MAE < 0.3），Artin 群 约 73%；③ Wagner‑Magyarik 直接解密成功率，BS(1,2) 与 Artin 群均 >97%，并且在 1,000 词长噪声下仍保持高置信度。

**⚠️ 局限性**

限制包括：仅为启发式近似，无法提供严格数学证明；最大词长受数据生成瓶颈限制（≈20,000）；对极短词误差较大；模型仅在 BS(1,2) 与特定随机 Artin 群上验证，未普适到所有无限非阿贝尔群；高词长下嵌入压缩导致分辨率下降。

---

## 56. Sensor-Placement-Agnostic Sonomyography: Toward Continuous High-Dimensional Control by Users with Tetraplegia

**arXiv ID:** 2607.26401 | [PDF](https://arxiv.org/pdf/2607.26401v1)

**作者:** Gavin Sueltz `[一作]` (University of Utah), Laura A. Hallock `[通讯]` (University of Utah)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了一种基于稀疏光流的实时、传感器位置无关的超声肌电（SMG）控制系统，支持单自由度（1-DOF）连续控制并提出了可扩展的二维（2-DOF）原型；

**💡 创新点**

创新点在于使用光流追踪肌肉和软组织变形，实现只需三点校准即可在任意身体部位获得连续控制信号，并首次展示了位置无关的多自由度连续SMG控制；

**🔧 技术方法**

采用稀疏Lucas–Kanade光流算法对B‑mode超声图像进行特征点跟踪，并利用线性插值映射到控制坐标；

**📊 数据集**

使用来自9名受试者（3名颈椎SCI受伤者和6名健康对照）的实时超声数据，覆盖6个传感器位置（肱二头肌、颈部、斜方肌、三角肌、伸展肌、屈肌）；

**📈 对比分析**

通过轨迹跟踪任务评估，所有受试者在任一位置均能实现1-DOF控制，平均跟踪误差低于5.5%，多数受试者低于4%；2-DOF系统在部分受试者中能实现全二维工作空间遍历，展示了可行性；

**⚠️ 局限性**

局限性包括样本量小、实验时间短、光流点漂移导致误差累积、2-DOF算法仍存在信号耦合和非线性问题，以及硬件传感器固定方式不够舒适，需进一步优化和更大规模验证。

---

## 57. Weight and Height Estimation from a Single Human Image Captured in the Wild

**arXiv ID:** 2607.26104 | [PDF](https://arxiv.org/pdf/2607.26104v1)

**作者:** Hira Yaseen `[一作]` (Information Technology University), Waqas Sultani `[通讯]` (Information Technology University)

**通讯引用:** 914 | [OpenAlex ID](https://openalex.org/A5082613558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究利用单张野外图片自动估计人体身高、体重和BMI，并构建了包含6105张多姿态、多民族、不同性别的全身与上身图片数据集

**💡 创新点**

创新点包括：①公开大型Body2BMI-ITU数据集；②在多任务学习框架下融合RGB、姿态亲和图、深度图和前景遮罩等多模态输入；③证明多任务与多模态相结合可显著提升估计精度

**🔧 技术方法**

采用的技术包括：深度卷积网络（ResNet‑50、DenseNet‑121、VGG‑16）与单/多任务回归/分类；姿态检测（OpenPose）生成亲和图；深度图生成（NYU‑v2预训练网络）；前景遮罩（Mask‑RCNN）；多模态拼接（GAD、GAM）

**📊 数据集**

使用自采的Body2BMI‑ITU数据集（6105张图片，包含全身、上身和面部子集），每张图片配有体重、身高与BMI标签

**📈 对比分析**

与现有SOTA方法（如Dantcheva、Jiang等）比较，MTR模型在全身图像上的MAE最低（BMI≈3.32，体重≈12.6kg，身高≈0.08m），BMI分类准确率约64%（AUC≈81%），多任务学习优于单任务；多模态GAM优于GAD

**⚠️ 局限性**

限制：极端BMI（肥胖、偏瘦）样本稀缺导致误差增大；深度图质量受限，姿态检测误差影响精度；数据集与代码尚未完全公开，实验可重复性受限

---

## 58. Learning Dynamic User Personas from Implicit Interaction Streams via Iterative Refinement

**arXiv ID:** 2607.26473 | [PDF](https://arxiv.org/pdf/2607.26473v1)

**作者:** Haifeng Wu `[一作]` `[通讯]`, Haifeng Wu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 IRIS 框架，利用隐式交互流持续学习动态用户人物模型。

**💡 创新点**

创新点在于无监督、闭环预测误差驱动的人物迭代更新与稳定正则。

**🔧 技术方法**

使用 LLM 提取行为记忆、合成自然语言人物、行为预测并进行误差反馈。

**📊 数据集**

评估使用公开自传文本作为合成流、Reddit r/AmItheAsshole 真实决策数据，计划 PRISM 与 Chatbot Arena。

**📈 对比分析**

与无个性化、静态人物、仅记忆回放、SynthesizeMe 四种基线比较，IRIS 在真实决策 DPA 上取得最高 61% 准确率。

**⚠️ 局限性**

局限包括隐式信号歧义、冷启动不足、评估指标受词汇记忆影响、计算成本和单模型单种子结果的稳健性。

---

## 59. Bias at the Borderline: Who Gets the Benefit of the Doubt in Peer Review?

**arXiv ID:** 2607.26280 | [PDF](https://arxiv.org/pdf/2607.26280v1)

**作者:** Hazem Ibrahim `[一作]` (New York University Abu Dhabi), Yasir Zaki `[通讯]` (New York University Abu Dhabi)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对ICLR会议的两阶段同行评审过程进行了审计，重点考察在边缘分数区间内，作者所在机构声望、国家背景（WEIRD与非WEIRD）以及作者团队性别是否在评审官的裁决中遭遇更高门槛。

**💡 创新点**

创新点在于首次将“稳健结果检验”方法应用于同行评审，并将拒稿方的后续学术表现纳入考量，实现了对决策偏差与信息差异的双重检验；此外，本研究构建了一个预先注册的、包含多重检验校正的完整结果体系。

**🔧 技术方法**

方法上结合了边缘区间限制、决策率差异分解、Gaebler‑Goel 的稳健结果检验、贝叶斯/频率检验以及置换检验等统计技术，并对拒稿后发表路径进行了文本相似度、时间延迟和期刊层级转移等后续指标的分析。

**📊 数据集**

数据集为2019‑2025年ICLR会议公开的OpenReview记录（约3.17万篇提交，其中1.04万篇位于边缘区间），以及2026年的公开数据用于无样本外复制；随后对拒稿论文的后续发表记录（ArXiv、期刊等）进行追踪。

**📈 对比分析**

比较方法：以边缘区间内等分数组的接受率为基准，拆解为评分差异和决策阶段差异；随后使用稳健结果检验检验是否存在“更高门槛”。实验结果显示：仅在声望轴上存在显著的决策阶段接受率差距（约‑1.6个百分点），但所有后续指标均未出现与此差距一致的结果，最终结论为无“更高门槛”证据。

**⚠️ 局限性**

局限性包括：1）后续指标仅为代理质量，可能受声望、可见度等因素影响；2）性别判定基于姓名推断，误差与偏差难以完全校正；3）审计仅覆盖ICLR的特定年份与边缘区间，未能检验决策前的评分或初始提交过程；4）稳健结果检验仅能给出充分但非必要条件，未排除所有潜在偏差。

---

## 60. Characterizing Human-Likeness in AI Generated Poetry: A Zero-shot Classification Study

**arXiv ID:** 2607.26221 | [PDF](https://arxiv.org/pdf/2607.26221v1)

**作者:** A. N. Biswas `[一作]` (BRAC University), A. Ahmed `[通讯]` (BRAC University)

**通讯引用:** 1999 | [OpenAlex ID](https://openalex.org/A5052493207)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一套零射击检测管线，能够区分人类与生成式 AI 诗歌。

**💡 创新点**

首次系统化挖掘人类与 AI 诗歌的判别属性，并验证 Gemma 4 在零射击场景下优于现有检测器。

**🔧 技术方法**

使用 Gemma 4‑31B、传统统计检测器（Log‑Likelihood、LRR、Binoculars、Fast‑DetectGPT）以及人工问卷进行对比。

**📊 数据集**

基于 1,513 篇人类诗歌与 4,539 篇 LLM（GPT‑OSS‑120B、Llama‑3.3‑70B、Qwen‑3‑32B）生成的诗歌，合计 5,970 篇。

**📈 对比分析**

在零射击下 Gemma 4 的加权 F1 为 0.900，超过 Log‑Likelihood 的 0.865，传统检测器性能普遍落后，人工判断准确率仅 44.7%。

**⚠️ 局限性**

数据集清洗不完善、缺乏英文学者评价、模型偏好少样本导致主题不一致、未对诗歌进行人类化处理。

---

## 61. Designing Needs- and Attention-Aware AI Learning Tools for Engineering Education: Insights from Psychological Outcomes

**arXiv ID:** 2607.26338 | [PDF](https://arxiv.org/pdf/2607.26338v1)

**作者:** Kevin Zhongyang Shao `[一作]` (University of Washington), Sep Makhsous `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对美国某大型公立研究型大学206名工程本科生的问卷调查，结合结构方程模型与潜变量交互效应，研究了AI聊天机器人对学生自主、能力、归属等心理需求的影响，并探讨了基础心理状态与注意力差异对这些影响的调节作用。

**💡 创新点**

①提出AI支持是需求特定而非均一的，①基于自我决定理论识别不同需求的差异化效果；②将注意力缺失视为调节变量，揭示其在AI效果中的作用；③基于测量结构提供可复用的HCI设计与评估指导，填补了现有研究中对心理需求和注意力差异化效应的空白。

**🔧 技术方法**

采用问卷设计、EFA/CFA评估测量模型、结构方程模型（SEM）估计路径关系、modsem包实现潜变量交互效应、Cronbach/omega计算内部一致性；使用R语言及相关统计包完成分析。

**📊 数据集**

使用2025年春季某大型公立研究型大学的工程专业本科生（n=206）在使用AI聊天机器人的问卷数据，涵盖自我决定理论的自主、能力、归属需求、个人代理、自我调节学习、注意力等测量。

**📈 对比分析**

SEM模型解释了52.3%的认知能力满足感变化、37.5%的自主满足感变化、9.7%的归属满足感变化，并通过交互效应显示注意力对能力满足和自主满足的调节作用。与仅考虑基线需求的模型相比，加入注意力变量提升了模型拟合优度（CFI、RMSEA、SRMR均改善），表明注意力是重要的调节因素。

**⚠️ 局限性**

研究仅在单一高校、工程学科内进行，样本以男性为主，跨机构和跨文化推广有限；采用自我报告问卷，受限于响应偏差与横断面设计，无法确立因果关系；量表在本研究中进行了删改，导致与原始验证工具的可比性受限；未收集客观学习成绩或纵向跟踪数据，限制了对实际学习成效的评估。

---

## 62. CMT-RAG: Complementary Memory Traces for Multi-turn Multi-hop RAG

**arXiv ID:** 2607.26470 | [PDF](https://arxiv.org/pdf/2607.26470v1)

**作者:** Lang Zhou `[一作]` (Sun Yat-sen University), Zhilin Zhao `[通讯]` (Sun Yat-sen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了CMT‑RAG框架，将对话上下文表示为子问题级别的检索痕迹，并提出MuMu‑QA基准来评估多轮多跳问答；

**💡 创新点**

创新点在于双通道记忆——状态空间模型（SSM）保持局部对话状态，持久Trace DAG记录子问题、依赖与证据，显式建模跨轮子问题依赖并以结构化痕迹为检索单元；

**🔧 技术方法**

使用Mamba‑2基的SSM作为子问题生成器，并通过LoRA+Direct Preference Optimization（DPO）进行训练；Trace DAG存储关键词、依赖与段落ID；检索采用DRAGON；评估指标包括EM、F1、检索段落数与延迟；

**📊 数据集**

主要使用MuMu‑QA（由MuSiQue合成的多轮多跳问答数据），并在RECOR、HotpotQA、2WikiMultiHopQA上进行跨基准迁移评测；

**📈 对比分析**

与五类RAG基线（Direct C‑RAG、ReAct、Self‑Ask、ChatQA、LogicRAG等）进行对比，CMT‑RAG在Qwen3‑32B和Llama‑3.3‑70B‑Instruct上分别提升EM/F1约6/4点，同时平均检索段落数从20降至约14；在跨基准上表现最优或具有竞争力；

**⚠️ 局限性**

主要局限是对子问题分解和依赖预测的精度高度依赖，生成误差会削弱性能；MuMu‑QA为合成数据，可能带来领域偏差；长轮会话的记忆机制仍需进一步鲁棒性验证。

---

## 63. Collusion with Competitive Marginals: Price-Level Audits Are Blind by Construction

**arXiv ID:** 2607.26385 | [PDF](https://arxiv.org/pdf/2607.26385v1)

**作者:** Xin Xu `[一作]` (Carnegie Mellon University), Hanzhe Hong `[通讯]` (Carnegie Mellon University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了一种仅通过共享公共随机源实现的无通信竞价协同机制，并证明传统基于单边际价格水平的检测方法对其无法检出；

**💡 创新点**

提出通过仅改变协同方之间的copula、保持边际分布不变的“无边际价格提升”协同方案，并在语言模型代理与以太坊MEV-Boost市场中验证其存在与收益；

**🔧 技术方法**

使用概率论中的极值理论、copula理论、蒙特卡洛模拟、残差相关性检验及多层审计算法来构建与检验协同模型；

**📊 数据集**

利用20个语言模型代理（19位开发者）在不同采样温度与提示下的模拟竞价数据，以及24天的以太坊MEV-Boost区块生成竞价日志；

**📈 对比分析**

与传统单边际检验、成对相关性检验及贝叶斯校正的假设检验对比；结果显示单边际检验始终等于显著性水平，成对检验在足够样本下可达到1的检验力，温度提升可显著削弱协同；

**⚠️ 局限性**

局限性包括仅在单日数据上评估实体计数，对模型代理与真实区块生成者的差异缺乏实证；依赖于残差拆分和审计强度，无法直接量化租金大小。

---

## 64. A Controlled Candidate-Set Benchmark for Offline Satellite-Security Plan Decomposition

**arXiv ID:** 2607.26371 | [PDF](https://arxiv.org/pdf/2607.26371v1)

**作者:** João Paolo Cavalcante Martins Oliveira `[一作]` (Universidade Federal do Rio Grande do Norte), Paulo Matias `[通讯]` (Universidade Federal de São Carlos)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了低秩适配器在卫星安全计划分解任务上的应用，使用控制候选集进行离线评估。

**💡 创新点**

创新点在于构建了可泄露控制的候选集基准，结合低秩适配器和结构化输出，明确区分检索与选择。

**🔧 技术方法**

使用低秩LoRA适配器、Qwen2.5基础模型、top-p采样、结构化生成与评估脚本。

**📊 数据集**

使用24个人工编写的卫星安全分解案例，生成83个下一步示例，构成SatSec Corpus。

**📈 对比分析**

与无适配器、模式提示、两示例提示等基线对比；在0.5B、1.5B、7B规模上，适配器在精确率上略优但召回率低，整体表现并未显著提升。

**⚠️ 局限性**

局限在于仅检验已知技术的候选集选择，未评估检索、语义验证、对新技术的泛化；数据量小，案例不代表总体，性能受模型规模限制。

---

## 65. How Wrangling Tools Shape Wrangling: A Technical Dimensions Analysis

**arXiv ID:** 2607.26198 | [PDF](https://arxiv.org/pdf/2607.26198v1)

**作者:** Shiyi He `[一作]` (University of Utah), Andrew M. McNutt `[通讯]` (University of Utah)

**通讯引用:** 576 | [OpenAlex ID](https://openalex.org/A5058039138)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过40名受试者在Jupyter、Excel、ChatGPT与OpenRefine四种接口范式下完成数据清洗任务，探讨工具特性如何影响用户的清洗策略与结果。

**💡 创新点**

创新点在于首次将技术维度编程系统（TDoPS）框架用于比较不同类型的wrangling工具，并揭示工具可用性与用户行为之间的相互作用与权衡，而非单纯比较性能。

**🔧 技术方法**

主要技术手段包括观察性实验、think‑aloud 录音与屏幕捕捉（ReVISit平台），使用TDoPS进行定性编码，并采用Jaccard相似度计算与聚类对比受试者输出与预设银表。

**📊 数据集**

使用了两个公开数据集：Game Rating Dataset（约10分钟限制）和Used Cars Dataset（无时间限制），用于评估工具在约束与开放任务中的表现。

**📈 对比分析**

比较方法是通过生成银表作为合理解空间的基准，对受试者输出与银表以及原始数据进行Jaccard相似度计算，随后聚类分析；结果显示各工具在不同任务下无显著性能优势，表现取决于工具特性与任务约束。

**⚠️ 局限性**

限制包括实验在远程实验室环境下进行，任务与数据规模有限，受试者可使用外部资源影响结果，缺乏人口统计信息，工具部署方式（VM、JupyterLite）可能产生差异，且未采用纵向对照设计。

---

## 66. When a positive SIMP density floor is not enough: solver admissibility and guarded floor selection in matrix-free 3D topology optimization

**arXiv ID:** 2607.26382 | [PDF](https://arxiv.org/pdf/2607.26382v1)

**作者:** Shaoliang Yang `[一作]` (Santa Clara University), Yunsheng Wang `[通讯]` (Santa Clara University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了三维SIMP拓扑优化中，矩阵自由几何多重网格FGMRES求解器在使用低密度地板（density floor）时，投影残差判据与实际残差不一致导致求解误判的问题，并提出一种基于投影残差特征的探测规则与重新计算真实残差的安全门控策略来确保求解可接受性。

**💡 创新点**

创新点在于：①证明投影残差判据在高对比度、迭代依赖预条件器下会误判收敛；②设计了两特征（r₅₀、r₁₀₀）驱动的地板提升规则；③引入真实残差校验门控，避免误接受；④通过经验评估证明在保持原始地板的同时可显著降低合规性与梯度偏差。

**🔧 技术方法**

使用的技术包括：矩阵自由有限元三维弹性求解器、四级几何多重网格预条件器（Chebyshev平滑、Galerkin粗网格、可调节点块Jacobi细层修正）、右预条件的灵活GMRES、投影残差特征提取、真实残差重算与阈值判定。

**📊 数据集**

数据集为：两类结构（cantilever、bridge）在不同分辨率（64k–514.5k单元）下的102个随机Bernoulli场和7个优化后设计；还包括对1百万单元模型的单次验证。

**📈 对比分析**

与固定地板（10⁻³、10⁻²）以及完整尝试（先10⁻¹²再回退）的策略比较，安全门控策略在保持原地板时可避免平均31%合规误差、0.34梯度偏差，成本约为固定10⁻³策略的2.5倍；固定10⁻³策略最快，但会导致较大模型偏差。

**⚠️ 局限性**

局限性包括：仅验证单一多重网格/FGMRES实现；测试范围局限于结构化八节点八面体网格、线性弹性、特定预条件器；对更大规模、非结构化网格、非线性/接触、不同精度（FP32/BF16）缺乏全面评估；门控策略的运行成本高，且在优化迭代中未探索状态重用或更细粒度的预条件器调整。

---

## 67. High-Order Markov Blanket Discovery via a k-Order Relaxation of the Faithfulness Assumption

**arXiv ID:** 2607.26357 | [PDF](https://arxiv.org/pdf/2607.26357v1)

**作者:** Loong Kuan Lee `[一作]` (Fraunhofer IAIS), Nico Piatkowski `[通讯]` (Fraunhofer IAIS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种 k-order 失真容忍的忠实性假设，并基于此设计了 kOMB（k-order Markov Blanket）算法，用以在存在 XOR/Parity 等高阶依赖时学习变量的马尔可夫毯。

**💡 创新点**

创新点在于：①从高阶（k+2）变量的相互作用角度重新定义“k-order 关联/依赖”；②引入“k-order 失真容忍”假设和 l‑bounded separator 假设，扩展了传统的 2-adjacency 失真容忍；③在此理论框架下给出了完整的证明，证明 kOMB 在满足假设时能正确恢复图形马尔可夫毯。

**🔧 技术方法**

技术包括：约束型（constraint‑based）MB 搜索（类似 GS 算法）、基于 G‑test/SCI 的条件独立性检验、k-order 关联/依赖检测、l‑bounded separator 检测、以及一系列子算法（find_inter_dep、find_cond、make_strict）。

**📊 数据集**

实验数据集：①5个 4 变量的合成 Bayesian 网络（包含 XOR/Parity 等不同 Boolean 逻辑）；②若干公开基准 Bayesian 网络（Alarm, Insurance, Barley, Mildew 等），在每个网络中随机采样 5000 条样本，并聚焦最大邻域变量。

**📈 对比分析**

与 GS、IAMB、HITON、MMMB、PCMB、LRH、STMB、BAMB 等 8 种主流 MB 发现方法比较。指标为 F1 分数（结合精确率与召回率），kOMB 在 2‑order 设定下在合成数据上能完全恢复马尔可夫毯，且在基准数据上通常取得更高 F1；同时在高阶依赖场景（Parity）下 GS 无法恢复。kOMB 的运行时间随 k 与 l 增大呈指数级增长，但在 k≤2 时仍保持可接受。

**⚠️ 局限性**

局限性：①对 k 与 l 的选择需要经验性调参，理论上 k 与 l 越大算法复杂度越高；②假设（k‑order 失真容忍、l‑bounded separator）在真实数据中的满足程度未知；③实验仅使用 G‑test/SCI 作为 CI 检验，未来可尝试更鲁棒的检验方法；④目前实现为 proof‑of‑concept，缺乏大规模数据下的高效近似搜索策略。

---

## 68. Emergent Sparsity in Frozen Random CNN Feature Extractors for Deep Reinforcement Learning

**arXiv ID:** 2607.26059 | [PDF](https://arxiv.org/pdf/2607.26059v1)

**作者:** Scott M. Norton `[一作]` `[通讯]` (Independent Researcher), Scott M. Norton (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究冻结随机CNN特征提取器的深度强化学习Agent，在不加任何稀疏性约束的情况下自发产生极度稀疏的全连接层表示，并探究其与任务复杂度的关系。

**💡 创新点**

发现稀疏性可在冻结随机投影下自发出现，证明任务内在维度决定有效特征数量；首次揭示激活神经元早期锁定并与奖励提升存在滞后；将冻结CNN视为测量任务内在维度的工具。

**🔧 技术方法**

使用PPO（Stable-Baselines3与Sample-Factory）、冻结CNN特征提取器、激活分析与消融实验、PCA/参与率、长期检查点追踪等技术。

**📊 数据集**

以Atari 2600四个游戏（Pong、Breakout、Space Invaders、Freeway）为实验数据集，采用标准ALE预处理。

**📈 对比分析**

与可训练CNN对照，比较激活数量、参与率、奖励；冻结CNN在简单游戏仅需1–3神经元即可达到专家水平，复杂游戏需要19–26/≈42神经元；可训练CNN在所有游戏均占满64个，表现相对稳定。

**⚠️ 局限性**

局限性：仅验证了Nature-DQN架构和离散动作游戏；缺乏理论证明和对连续控制或不同网络结构的泛化；sticky-action与种子随机性对结果影响尚未完全消除。

---

## 69. Route-Block Membership Selects Packed-AWQ Arithmetic: A Controlled Single-Fixture Mechanism Study

**arXiv ID:** 2607.26316 | [PDF](https://arxiv.org/pdf/2607.26316v1)

**作者:** Lukas Stepanek `[一作]` `[通讯]`, Lukas Stepanek

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在 Qwen3-Coder 的第 6 层 MoE 模型上对路由块对齐的细节进行实验，证明了路由块成员关系可以改变已量化的矩阵乘法的精确算术轨迹。

**💡 创新点**

创新点在于首次将 MoE 对齐过程视为算术控制流，展示了跨块边界的路由交换、同块内部交换、单切片网格控制以及稳定构造三种操作如何分别影响、保持或统一精确算术轨迹。

**🔧 技术方法**

使用技术包括：AWQ 4‑bit 权重量化、Marlin 的打包混合精度 GEMM、vLLM 的路由实现、CUDA 13.0/RTX 3090 GPU、对齐块与算术调度类的映射推导以及自定义的“路由‑块”抽象。

**📊 数据集**

数据集为 Qwen3-Coder 30B-A3B‑Instruct‑AWQ 的单层（层6）量化权重和对应的路由/输入张量，全部来自公开的 Hugging‑Face checkpoint，实验在 70 条独立的冷启动进程中重复验证。

**📈 对比分析**

比较方法是对相同输入/权重在不同对齐策略下记录四个算术表面（激活、路由、输出和最终结果）的位级哈希，验证精确等价与转移。性能指标仅提供单点延迟（0.241 ms）与吞吐（28.4 tokens/s）对比，未做大规模性能声明。

**⚠️ 局限性**

限制主要包括：只测试了单个层、单个 checkpoint、单个 GPU 架构；实验依赖预设的复合预构造处理，未分离分配器/内存碎片等因素；结果仅展示算术轨迹差异，未评估实际推理质量或服务层面的影响；并且缺乏跨平台或跨模型的泛化验证。

---

## 70. Continuous Online Evaluation of Recommendation Strategies in Social Science Academic Search

**arXiv ID:** 2607.26380 | [PDF](https://arxiv.org/pdf/2607.26380v1)

**作者:** Mehmet Deniz Türkmen `[一作]` (Leibniz Institute for Social Sciences), Daniel Hienert `[通讯]` (Leibniz Institute for Social Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在 GESIS Search 上通过 STELLA 框架进行连续在线评估，比较了基于词项、语义嵌入和会话路径的五种推荐器。

**💡 创新点**

首次在学术搜索领域实现持续在线实验与评估，并结合类别特定分析揭示不同信息类型下推荐效果差异。

**🔧 技术方法**

采用 Elasticsearch MLT、Transformer 嵌入（para‑phrase‑multilingual‑mpnet‑base‑v2、nomic‑embed‑text‑v2‑moe、all‑MiniLM‑L6‑v2）与 KNN、GRU RNN 进行会话预测，并用 STELLA 进行团队草稿互插。

**📊 数据集**

使用 GESIS Search 的六大信息类别（出版物、研究数据、变量、仪器工具、图书馆文献等）以及 8 年累计 16M 次用户交互日志进行训练与评估。

**📈 对比分析**

采用团队草稿互插、两侧二项符号检验和 Holm‑Bonferroni 校正进行配对比较；结果显示语义嵌入（nomic）最佳，其次是 MiniLM≈mpnet‑base，随后是 CPR，词项最差；不同类别表现存在差异。

**⚠️ 局限性**

主要限制在于仅以点击作为相关性指标，忽略停留时长等信号；评估周期长；少量点击的类别数据不充分；会话推荐受训练数据偏倚影响。

---

## 71. A Fresh Look at Best Inductive Loop Invariant Synthesis for Bit-Vector Relations

**arXiv ID:** 2607.26386 | [PDF](https://arxiv.org/pdf/2607.26386v1)

**作者:** Hanrui Zuo `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对位向量程序，重新定义并实现最佳归纳不变量（BII）的求解，提供线性搜索和位级贪心两种高效算法。

**💡 创新点**

创新点：①把 BII 问题转化为基于约束的优化形式；②设计 propose‑and‑refine 框架；③提出位级贪心（binary‑lifting）策略与下界上限、bounded‑leap 等优化，使求解器调用数线性于总位宽；④在 k‑induction 上直接使用 BII，显著提升验证效果。

**🔧 技术方法**

核心技术包括 SMT/OMT 约束求解、位向量模板域（区间、八角形、稀疏模板多面体）、binary‑lifting、under‑approximation（boundary limits）和 bounded‑leap。实现基于 Z3 的求解器，利用约束模板推导最优不变量。

**📊 数据集**

使用多源基准集：LoopInvGen（结合 SV‑COMP、SyGuS‑COMP 等）、自定义的多阶段基准，包含 32/64/128 位的位向量实例，覆盖区间、八角形和稀疏模板多面体三种抽象域。

**📈 对比分析**

与传统符号抽象（CIBII(BS)、CIBII(Bi)）进行对比：在区间域中，EFBII(G) 解决 162 个实例，比基线多 86%；在八角形域中，EFBII(G) 解决 55/56 个实例，速度提升至 17.9×；在稀疏模板多面体域中，速度提升约 3.4×。solver‑call 仅为 1.7–1.8 倍，而基线求解器调用数多达 10⁶。对 k‑induction 的辅助验证实验表明，使用最佳不变量可提升 17% 的证明率，缩短 1.2× 的证明深度。

**⚠️ 局限性**

局限性：①仅适用于有限高度的抽象域；②对大规模关系模板（如完整多面体）求解器调用数仍高；③对位向量域的性能提升主要体现在位宽不太大时，极大位宽仍受 solver 量化约束限制；④在某些非线性循环中，求解器可能难以快速判定，导致时间溢出。

---

## 72. BG-REAL: A Public Real-Data Anchored Benchmark for Background Manipulation Detection and Localization

**arXiv ID:** 2607.26232 | [PDF](https://arxiv.org/pdf/2607.26232v1)

**作者:** Bugra Alperen Uluirmak `[一作]` (Erciyes University), Rifat Kurban `[通讯]` (Abdullah Gul University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个公开的背景编辑检测与定位基准包BG‑REAL，并提供完整的数据准备、评估和复现脚本。

**💡 创新点**

创新点在于：① 使用真实数据(anchor)与匹配的再编码对照样本，排除重编码伪影的误导；② 设计了六种编辑族与源无关的 train/val/test 切分；③ 结合人类辅助质量控制与“matched‑authentic”诊断揭示基线普遍存在的误判；④ 提供多基线（TruFor、MVSS‑Net、HiFi‑Net、BG‑RIFT）和五种随机种子评估。

**🔧 技术方法**

采用的数据集来源为 Open Images V7 实例分割；通过自定义管道生成合成、和谐化、公共背景替换、JPEG/resize 等六大编辑族；评估使用 AUROC、AUPRC、F1、准确率、ECR 以及像素 IoU、Pixel‑F1、Mask‑AP、Boundary‑F1 等定位指标；基线实现包括内部训练的 BG‑RIFT 与 RGB/Artifact 传统方法，外部的 TruFor、MVSS‑Net、HiFi‑Net 通过官方适配器调用。

**📊 数据集**

使用 Open Images V7 公开图片共 6,000 张（真实源）+ 1,000 张合成对照，生成 7,000 样本，覆盖 1,200 个源组，包含 6 种编辑族。

**📈 对比分析**

在 ID 与 source‑OOD 两个主要切分上，TruFor 的 AUROC 最高（≈0.84），BG‑RIFT 次之（≈0.71），MVSS‑Net 与 HiFi‑Net 低于 0.60；定位指标 Pixel‑F1 在 ID 约 0.36，source‑OOD 约 0.36；匹配对照测试显示大多数基线的误报率高达 0.99，TruFor 最低为 0.57；基线间通过 5 种种子计算均值±方差，显著性检验表明排名稳健。

**⚠️ 局限性**

局限包括：① 仍包含 1,000 张合成对照，未完全为“真实‑only”；② tool‑OOD 与 generator‑OOD 等 OOD 子集仅为标签过滤而非真正零泄漏；③ 缺少扩散式背景编辑族；④ 评估基线多为零训练、无 fine‑tuning，未覆盖所有常用模型；⑤ 随机种子数量有限，可能影响统计稳健性；⑥ 人类质量控制仅覆盖 599 行，未覆盖全部样本。

---

## 73. User-Reported Misinformation Exposure Across Social Media Platforms

**arXiv ID:** 2607.26218 | [PDF](https://arxiv.org/pdf/2607.26218v1)

**作者:** Catherine King `[一作]` (Carnegie Mellon University), Kathleen M. Carley `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 1,010 名美国社交媒体用户进行调查，收集其对不同平台误导信息曝光和发布的主观感知，并使用混合效应逻辑回归模型分析平台使用频率与误导信息曝光和发布之间的关系。

**💡 创新点**

首次从用户主观感知角度系统比较 11 种社交平台的误导信息曝光差异，并提出针对交互型、讨论型和发现型三类平台的差异化治理策略。

**🔧 技术方法**

采用混合效应逻辑回归（R 语言的 `lme4` 包）分析平台使用频率、平台类别、性别、政治意识形态等变量对曝光和发布的影响。

**📊 数据集**

利用 Qualtrics 采集的 1,010 名美国成年人社交媒体使用和误导信息相关问卷数据，包含平台使用频率、曝光来源、发布意图等字段。

**📈 对比分析**

与传统内容传播度量（如帖文点击率、分享次数）不同，本研究基于自报感知进行统计比较；模型结果显示平台使用频率与曝光/发布的相关系数均在 0.18-0.56 之间，曝光对发布的影响显著（OR ≈ 5.6）。

**⚠️ 局限性**

样本仅覆盖活跃的美国用户，误导信息的定义依赖于受访者主观判断，缺乏客观事实核查，且只关注自报数据，可能导致偏差。

---

## 74. Weak-to-Strong On-Policy Distillation

**arXiv ID:** 2607.26246 | [PDF](https://arxiv.org/pdf/2607.26246v1)

**作者:** Fangxu Yu `[一作]` (University of Maryland), Jianfeng Gao `[通讯]` (Microsoft Research)

**通讯引用:** 39291 | [OpenAlex ID](https://openalex.org/A5114910293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种弱到强的 On‑Policy Distillation（W2S‑OPD）框架，利用弱模型的对比（logit 差异）提取能力方向，并将其注入强学生的基准模型，从而提升学生的推理与代码生成性能。

**💡 创新点**

创新点在于：①通过对弱模型的正负对比获得能力方向，避免直接仿真弱教师导致的能力瓶颈；②构造分布相近的代理教师，将该方向“注入”学生基准模型；③支持多种对比方式（RL、规模差异、提示差异），无需更大教师即可提升强模型。

**🔧 技术方法**

技术包括：On‑Policy Distillation、逆 KL 损失、logit 差异合成代理教师、放大系数 α 的调节、Top‑k OPD、以及多教师聚合。

**📊 数据集**

使用的数据集包括数学推理基准 AIME24/25、HMMT25；代码生成基准 HumanEval+、MBPP+、LiveCodeBench‑V6；以及 OOD 评测 GPQA‑Diamond、IFBench。

**📈 对比分析**

与传统 OPD、SFT 对比，W2S‑OPD 在单教师和多教师设置下均显著优于 OPD，平均提升约 11% 的数学推理准确率、3.7% 的代码生成准确率，并能超越所使用的弱教师；在 OOD 任务上也表现出更好的迁移。

**⚠️ 局限性**

局限性包括：对比模型的选择与 α 参数调节仍需经验；实验仅涵盖数学与代码两大任务，未验证更大规模或其它领域的适用性；代理教师的构造可能在某些任务中效果有限，且多对比的组合仍需要进一步优化。

---

## 75. An ER-Model-Based Framework for Case Notion Selection in Object-Centric Processes

**arXiv ID:** 2607.26384 | [PDF](https://arxiv.org/pdf/2607.26384v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 76. Robostreet Flow: A Lightweight, Ultra-Low-Drag Electric Tractor and Four-Truck Hybrid Convoy Architecture for Minimum-Cost Point-to-Point Freight

**arXiv ID:** 2607.26250 | [PDF](https://arxiv.org/pdf/2607.26250v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 77. Sim2Win: A Team-Agnostic, Event-Based Pre-Match Outcome Prediction and Tactical Profiling System for Football

**arXiv ID:** 2607.26061 | [PDF](https://arxiv.org/pdf/2607.26061v1)

**作者:** Mouad Zemzoumi `[一作]` (Al Akhawayn University in Ifrane), Amine Abouaomar `[通讯]` (Al Akhawayn University in Ifrane)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了Sim2Win，一个团队无关、基于事件的赛前结果预测与战术分析系统，利用滚动5场比赛的战术特征构建球队行为向量；

**💡 创新点**

创新点在于：①使用可解释的四个战术比率和K‑Means聚类得到八种战术风格，完全不依赖球队身份；②在Leave‑One‑Competition‑Out（LOCO）框架下验证了跨赛制的泛化能力；③通过团队无关的行为表示实现了对未见球队的可迁移预测；

**🔧 技术方法**

技术包括：StatsBomb事件数据预处理、滚动窗口特征工程、四个可解释战术比率、标准化+K‑Means聚类、CatBoost分类器、10折交叉验证、McNemar统计检验、LOCO评估、SHAP解释；

**📊 数据集**

使用了StatsBomb公开事件数据集，涵盖11大赛，178支球队，1,411条队-赛记录（约706场比赛）；

**📈 对比分析**

通过与ELO、Pi‑Rating和GAP三种身份依赖的基准在LOCO协议下对比，Sim2Win平均ROC‑AUC 0.704、准确率55.4%，在所有比较中均优于基准；在分布内验证中CatBoost实现最高60.9%准确率；

**⚠️ 局限性**

局限性包括：数据量有限且以欧洲联赛为主，导致在AFCON、欧锦赛等赛事的泛化性能较弱；无法可靠预测平局；模型对主客场优势存在偏倚；8类战术聚类未通过专家验证；缺少球员层面、伤病、天气等情境特征。

---

## 78. The Code Distortion Problem

**arXiv ID:** 2607.26261 | [PDF](https://arxiv.org/pdf/2607.26261v1)

**作者:** Huck Bennett `[一作]` (University of Colorado Boulder), Bryant Morrell `[通讯]` (University of Colorado Boulder)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文提出并研究了代码失配度（Code Distortion）概念，定义了测量两个线性码相似度的最小失配度，并给出了对应的搜索与决策问题（Code Distortion Problem, CDP）

**💡 创新点**

创新点在于将传统的线性码等价问题推广为可测距的最小失配度问题，证明了该问题在常数逼近下是 NP‑难的，并给出了单指数时间的近似算法与上界分析；同时引入了码的连续最小值基（successive minima basis）和对码的 0→0 矩阵范数的研究

**🔧 技术方法**

主要技术包括：对码连续最小值基的构造与利用、对 0→0 范数的归约与复杂度分析、基于线性映射的失配度上界证明、以及从最短向量/最近向量问题的线性化归约方法

**📊 数据集**

文章未使用具体实验数据集，而是通过理论构造与抽象证明来验证算法与硬件复杂度的有效性

**📈 对比分析**

与传统线性码等价（等距）判定比较，本文的失配度算法在最坏情况下给出 k² 近似（二进制码连续最小值相同可提升到 (2k+1/3)² 近似），与全枚举的 q^k² 复杂度相比，取得显著加速；但在常数因子上仍不如理想等距判定精度

**⚠️ 局限性**

主要局限包括：1) 失配度问题的精确计算仍需指数时间；2) 近似因子仍较大，尚未达到多项式近似；3) 对于 q>2 的二进制特例，改进分析更复杂，且 0→0 范数子空间计算的 NP‑难度限制了进一步的多项式级别优化。

---

## 79. GuidedRAG: Semantic Steering of Retrieval-Augmented Generation

**arXiv ID:** 2607.26071 | [PDF](https://arxiv.org/pdf/2607.26071v1)

**作者:** Matthijs Jansen op de Haar `[一作]` (ETH Zürich), Lorenzo Gatti `[通讯]` (University of Twente)

**通讯引用:** 622 | [OpenAlex ID](https://openalex.org/A5079215275)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出GuidedRAG框架，在传统RAG中加入选择阶段和语义引导，使检索过程先根据预先指定的语义限制搜索空间，再进行检索、增量与生成。

**💡 创新点**

创新点是把语义驱动的选择阶段嵌入RAG，实现检索前语义约束，显著提升检索相关性、精度并降低计算开销。

**🔧 技术方法**

使用的技术包括向量检索（余弦相似度）、语义选择（实体、关系、集合操作）、以及LLM增量生成。

**📊 数据集**

使用All Relations Lead to Rome知识图谱（约300篇维基百科文本，6000问答+2400不可答问答）。

**📈 对比分析**

与传统RAG和GraphRAG对比，GuidedRAG在检索相关性提高14-15.8%，精度提升19-27%并将检索空间压缩至原来的0.01-11.9%，整体性能明显优于现有方法。

**⚠️ 局限性**

主要局限是依赖高质量的语义标签，若标签错误或缺失会影响效果，并且评测仅在单一知识库上，未验证跨域泛化。

---

## 80. Constrained minimization problems and FFT-based solvers: application to local Dirichlet boundary conditions and contact mechanics

**arXiv ID:** 2607.26082 | [PDF](https://arxiv.org/pdf/2607.26082v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 81. Progress in Benchmarking Generics for Mathematical Computation

**arXiv ID:** 2607.26206 | [PDF](https://arxiv.org/pdf/2607.26206v1)

**作者:** Daniel Pang `[一作]` (University of Waterloo), Stephen M. Watt `[通讯]` (University of Waterloo)

**通讯引用:** 3853 | [OpenAlex ID](https://openalex.org/A5082854012)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对 SciGMark 1.5 基准集进行实验，比较现代语言（Rust、Java、Go、TypeScript、Julia）在数值和符号计算中的泛型实现性能。

**💡 创新点**

将原始 SciGMark 扩展到有限域线性代数、有限域 FFT 和 naïve Gröbner 基础，并系统评估多种泛型实现策略（前端 monomorphization、类型擦除、运行时专化）在实际算法中的开销。

**🔧 技术方法**

采用前端多态化、类型擦除、运行时多分派等泛型技术，配合手写的 Monte Carlo、SOR、LU、FFT、Gröbner 基础五个基准核。

**📊 数据集**

使用固定规模的浮点、有限域和多项式输入，包括 n=4,5,6 的循环多项式系统、标准矩阵和 FFT 数据集。

**📈 对比分析**

通过同一语言下专用实现与泛型实现的速度比进行比较，结果显示 Rust 近乎无开销，Java/Go/TS 在循环密集型核中存在显著延迟，符号工作负载对所有语言均增加额外开销。

**⚠️ 局限性**

仅覆盖五个核，未涉及稀疏矩阵、并行或 GPU 场景，且仅测量整体时间，未细分分配、分派、缓存等细粒度瓶颈。

---

## 82. (Im)Paired Programming: Coding Agents Improve Productivity but Harm Understanding

**arXiv ID:** 2607.26375 | [PDF](https://arxiv.org/pdf/2607.26375v1)

**作者:** Nishant Balepur `[一作]` (University of Maryland), Jordan Lee Boyd-Graber `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对54名计算机科学学生进行实验，评估使用编码代理（直接编辑代码）和聊天机器人（需要用户自行编写或改写代码）在网页开发任务中的生产力与代码理解能力，结果表明代理虽能提升任务完成度，却显著降低用户对自身代码的理解并不提升后续代码扩展性能；

**💡 创新点**

首次将编码代理的使用与代码理解（回忆、推理、扩展）系统性关联，揭示低付出提示与自动接受编辑会削弱理解，并提出针对提示策略、审查交互、代码可读性等方面的改进方向；

**🔧 技术方法**

采用GPT-4.1驱动的Aider代理、ChatGPT聊天机器人、Gemini-3.1 Pro评判器、基于ESprima的代码可读性度量，以及自定义的评估 Rubric 与自适应多选题生成器；

**📊 数据集**

使用54份学生完成的网页（zic-zac-zoe）代码、提示记录、背景测验、理解问卷与后续扩展任务数据，公开数据集供后续研究；

**📈 对比分析**

对比方法为：随机分组实验，测量任务完成率、时间、回忆/推理题得分及扩展任务准确率；实验结果显示代理组初始任务完成率高但理解得分低，且在无代理的扩展任务中表现与聊天机器人组相当，表明任务完成度并未带来理解提升；

**⚠️ 局限性**

局限性包括仅在学生群体与单一网页开发任务上验证，实验环境与真实工作流程差异较大，评估工具与自动化生成问题可能存在误差，且仅针对短期任务，未检验长期学习与技能形成的影响。

---

## 83. Model-Driven Requirements Configuration with Three-Valued Uncertainty Scoring

**arXiv ID:** 2607.26220 | [PDF](https://arxiv.org/pdf/2607.26220v1)

**作者:** Ahmed Ibrahim `[一作]` `[通讯]` (Western University), Ahmed Ibrahim (Western University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个神经-符号多智能体系统，将大型语言模型（LLM）与符号验证器结合，自动从自然语言项目愿景中选择满足OOMRAM需求格子的结构正确的功能；

**💡 创新点**

将结构约束交给确定性符号验证器，保证消除逻辑不一致，并引入三值（T,I,F）不确定性框架量化LLM决策的可选性；

**🔧 技术方法**

使用LangGraph实现多智能体架构，LLM（Llama 3.1 8B及NVIDIA Nemotron 3 Ultra）、基于OOMRAM格子约束的符号验证器、前沿导航策略以及三值评分；

**📊 数据集**

37份自然语言项目愿景，覆盖11个应用族（如电子记录、智能家居、汽车信息娱乐等），配备手工构建的OOMRAM需求格子；

**📈 对比分析**

与无验证器的LLM基线（普通LLM、RAG、Self‑consistency、多智能体辩论）对比；验证器使结构错误率从约60–95%降至Llama 3.1 5.4%，Nemotron 3 Ultra 0%；平均每个节点需约1.16次LLM调用，验证器耗时低于0.4%；

**⚠️ 局限性**

验证器只能捕获结构约束，无法检测语义错误；残留失败取决于LLM的指令遵循能力；需要预先构建OOMRAM格子；迭代循环可能耗费大量标记；评估仅基于轻量级LLM和前沿模型，缺乏对其他模型的泛化验证。

---

## 84. Position: Evaluation Scores Are Perishable Knowledge Claims

**arXiv ID:** 2607.26191 | [PDF](https://arxiv.org/pdf/2607.26191v1)

**作者:** Sankalp Gilda `[一作]` (DeepThought Solutions), Shlok Gilda `[通讯]` (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出“评估信任膨胀”概念，指出把多种评估信号平均化会让整体置信度超过最弱信号的可靠性；并在此基础上提出评估结果应包含形式化层级、适用范围和有效期等元数据，保证评估知识主张的透明与可审计。

**💡 创新点**

核心创新在于将评估视为知识论断，提出三大元数据属性（形式化层级、范围声明、有效窗口），引入最弱链原则和参数化的OWA聚合操作（可调惰性参数ρ），并通过公开榜单实验展示平均聚合与最弱链聚合导致的模型排名分离。

**🔧 技术方法**

采用形式化层级阈值表、OWA聚合框架（min、mean、max可调）、基于实例的评估基线、与模型评估工具（如HELM）结合的评估Harness。

**📊 数据集**

使用HELM Lite v1.13.0、HELM Capabilities v1.0.0、MMLU、OpenBookQA、GSM8K、Math-CoT、MedQA、WMT等多维度公开评测数据集。

**📈 对比分析**

与传统平均聚合方法对比，使用最弱链聚合（WLNK）后，榜单排名产生显著偏移（Spearman ρ≈0.89，最大位移达21位，Top‑5重合度为0），表明平均聚合掩盖了关键维度的弱点。

**⚠️ 局限性**

局限性包括缺乏大规模实证验证、最弱链过度保守可能导致过度低估性能、有效窗口和形式化层级可能产生激励失衡或过度官僚化等问题。

---

## 85. From Interface to Inference: Eliciting Any-Order Inference from Any-Order Models

**arXiv ID:** 2607.26504 | [PDF](https://arxiv.org/pdf/2607.26504v1)

**作者:** Seunggeun Kim `[一作]` (University of Texas at Austin), Sitan Chen `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两种解决 Masked Diffusion Model (MDM) 在实现任意顺序推理时的“位置不确定性”瓶颈的方法，分别为插入式 FlexMDM 和潜在空间 LatentMDM，并在代码生成与数理推理任务上验证其效果。

**💡 创新点**

创新点在于：①首次从模型架构角度系统识别并量化位置不确定性，阐明其导致的因果崩塌；②设计插入式 MDM 通过动态插入掩码突破固定画布限制；③提出潜在空间 MDM，将任意顺序推理迁移到语义段级别，获得更高层次的生成顺序搜索。

**🔧 技术方法**

核心技术包括 Masked Diffusion Modeling、插入式训练与推理机制、潜在编码-解码架构、基于 AST 的结构相似度评估、位置不确定性量化指标（aggregate mass 与 localization score）、confidence-based 与策略学习的解码策略。

**📊 数据集**

使用了 Python 代码生成数据集（HumanEval、HumanEval+、MBPP、MBPP+）以及 TinyGSM（GSM8K 结构化解法）进行预训练与评估，FlexMDM 在 Dream-Coder 7B 基础上微调；LatentMDM 在 125M 参数规模上从零训练。

**📈 对比分析**

与传统 MDM、Autoregressive、流模型（Spherical Flow、DUO）以及基于不同解码策略（随机、L2R）的对照实验。结果显示：FlexMDM 在 HumanEval@16 及 HumanEval+@16 上分别提升约1-2个百分点；LatentMDM 在 GSM8K 上在匹配壁钟时间下超过 5.5% 的准确率，并在 8–16 步采样中实现更高效的采样速度。

**⚠️ 局限性**

局限性包括：①插入式方法依赖于预训练 MDM，参数规模大；②潜在空间方法需从头训练，训练成本高；③两种方法在非代码、非结构化文本等更广泛任务中的通用性尚未验证；④对位置不确定性的量化仍是经验性指标，可能未覆盖所有场景。

---

## 86. Machine-Checked Certificates for the Geometric Half of the Minimum Kochen-Specker Bound

**arXiv ID:** 2607.26413 | [PDF](https://arxiv.org/pdf/2607.26413v1)

**作者:** Shayaan Siddique `[一作]` (Millennium Research), Ibrahim Mian `[通讯]` (Millennium Research)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对Kochen–Specker向量系统最小下界的几何部分进行形式化验证，给出可重放的分式案例树证书，并用两种独立检查器验证所有阻塞数据库中的图。

**💡 创新点**

提出了精确有理数案例树证书格式，支持因子分解、SOS、理想与正性分支，并在Lean 4中实现了核可验证的完美性定理，填补了原先缺失的证明对象。

**🔧 技术方法**

利用精确分式运算、符号SOS分解、正定性（Positivstellensatz）证明、Lean 4反演式检查器和纯Python重放器，以及Z3非线性实数解的可追溯性。

**📊 数据集**

验证了原论文的阻塞数据库（order ≤12）以及七个order-13最小不可嵌入图，共计266条源行；此外还检验了约43个order-23候选图的子图包含关系。

**📈 对比分析**

通过两套检查器（Python+分式、Lean 4）对每个图的非可嵌入性进行核检查，单图平均约0.5-1.8秒的kernel时间；对完整数据库的CI集成实现无缝重放，并通过对抗性变异测试保证鲁棒性。

**⚠️ 局限性**

仍未完成对全部order-23候选图的直接证书，且依赖原始阻塞列表与子图包含关系；此外，原论文的计数与文档不一致，需进一步同步。

---

## 87. Lag-aware cross-hand alignment for dual-hand action segmentation

**arXiv ID:** 2607.26215 | [PDF](https://arxiv.org/pdf/2607.26215v1)

**作者:** Fatemeh Ziaeetabar `[一作]` `[通讯]` (University of Tehran), Fatemeh Ziaeetabar (University of Tehran)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种轻量级的跨手臂时延对齐模块LACA，用于双手动作分割；

**💡 创新点**

创新点在于显式学习双手之间可变时延的方向性分布，并引入可学习的空状态来抑制无效跨手信息传递，同时用兼容性自适应的目标对齐监督；

**🔧 技术方法**

采用多头注意力机制进行时延搜索、软注意力归一化、空状态学习以及兼容性矩阵构造与软标签监督；

**📊 数据集**

在HA‑ViD和ATTACH这两个基于组装任务的双手动作数据集上进行评估；

**📈 对比分析**

与原Polyphony及其他对照方法比较，LACA在两手F1@50和B‑F1上分别提升约1.9–2.1点和3.1–3.2点；未来无缝版本LACA‑C在无未来观测下仍保持高召回率（83.6%）并大幅降低延迟与误报；

**⚠️ 局限性**

局限在于只验证了两套组装类数据集，未来无缝评测仅在ATTACH上；方法依赖帧级标注、固定搜索窗口，且吞吐量评估未包含特征提取时间。

---

## 88. MetaKoopman: Bayesian Meta-Learning of Koopman Operators for Modeling Structured Dynamics under Distribution Shifts

**arXiv ID:** 2607.26345 | [PDF](https://arxiv.org/pdf/2607.26345v1)

**作者:** Mahmoud Selim `[一作]` (TRATON), Karl H. Johansson `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出MetaKoopman框架，利用贝叶斯元学习在分布偏移场景下对非线性动力学进行在线自适应和不确定性估计。

**💡 创新点**

1）将Koopman算子与贝叶斯元学习结合，得到可闭式更新的MNIW先验；2）引入可学习的温度因子β实现先验调节；3）采用变分动作编码器加速采样规划。

**🔧 技术方法**

矩阵正态-逆Wishart先验、闭式贝叶斯推断、元学习、Koopman算子、变分自编码器、采样式运动规划。

**📊 数据集**

仿真数据：HalfCheetah‑Slope、Hopper‑Gravity、Walker‑Friction、Ant‑DisabledJoints、Panda‑Damping等；真实数据：37.5吨自动卡车在雪、冰、混沌摩擦等冬季道路条件下收集的驾驶轨迹。

**📈 对比分析**

与DKO、GrBAL、BayesianMAML、BNN、EMLP、NeuralODE等基线比较，MetaKoopman在多步预测MSE、分布偏移鲁棒性、误差校准和真实场景安全性上均优于对手。

**⚠️ 局限性**

仅对Koopman算子建模不确定性，未对嵌入表示建模；依赖于先验的质量，极端非线性或高维嵌入空间时性能可能下降；对连续时间控制的适用性仍待验证。

---

## 89. A large-scale corpus of religious radio broadcast transcripts from webstream recordings in the United States

**arXiv ID:** 2607.26249 | [PDF](https://arxiv.org/pdf/2607.26249v1)

**作者:** Samuel Bestvater `[一作]`, Aaron Smith `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了美国宗教广播的大规模转录语料库，采集了785个网络流在2025年7月的15分钟录音，共715,688条，转录并进行说话人分离、音乐/语音划分以及使用GPT‑4.1进行节目格式与主题标签；为宗教广播内容分析提供统一、可查询的数据平台。

**💡 创新点**

创新点在于：①首次覆盖所有可访问网络流的宗教广播，规模超过之前的RadioTalk；②将自动转录、说话人分离与LLM驱动的段落划分与多标签主题标注集成到一个统一的三层表结构；③为多语种、跨地区研究提供可扩展的元数据映射（流↔站点）。

**🔧 技术方法**

技术手段包括：FFmpeg+容器化监听实现大规模录音；Audio Spectrogram Transformer 进行语音/音乐分类；WhisperX（whisper‑large‑v3‑turbo + pyannote diarization）完成自动转录与说话人分离；OpenAI GPT‑4.1完成段落划分与格式/主题双重标注；数据以Parquet格式发布，支持快速流式查询。

**📊 数据集**

数据集为：785条直播流、3,904个主站与2,886个翻译/放大站的15分钟录音，715,688条记录，超过60,000,000行转录文本；包含录音时间、流元数据、节目格式与主题标签。

**📈 对比分析**

评估指标：ASR WER 5.14%（对照人类对比 2.57%），按频段、地区、拥有者等维度的细分WER均低于10%；LLM格式宏F1 0.77、微F1 0.74、整体一致率74%；主题宏F1 0.71、微F1 0.79、至少一标签一致率83%。音质差的录音W​ER显著升高至7.30%，表明模型对低质量音频更易失误。

**⚠️ 局限性**

限制包括：自动转录与LLM标签仍存在错误，音乐区块仅为占位符且不转录；跨录音说话人身份不可追踪；仅包含英语广播，非英语站点被排除；未提供原始音频，仅发布转录与衍生数据。

---

## 90. How Do Researchers Manage Visualization Experiment Stimuli?

**arXiv ID:** 2607.26443 | [PDF](https://arxiv.org/pdf/2607.26443v1)

**作者:** Hyeok Kim `[一作]` (Korea Advanced Institute of Science & Technology), Jeffrey Heer `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对19位可视化实验研究者的半结构访谈，系统梳理了实验刺激设计与管理中的挑战与策略，并探讨了AI支持的机会与风险。

**💡 创新点**

首次将实验刺激生命周期的设计、探索、选择、发放、分析等环节与研究者面临的约束（劳动力、技能、不确定性、不可修复性）结合，提出了针对性研究机会。

**🔧 技术方法**

采用访谈、录音、转写、主题分析等质性研究方法。

**📊 数据集**

基于19名研究者的访谈记录构成数据集。

**📈 对比分析**

未进行定量比较，主要是定性描述与分析。

**⚠️ 局限性**

样本偏向在线众包实验，研究者多为初级/中级研究者，且多为正定研究范式，未来需扩展到更广泛实验类型与研究者层级。

---

## 91. Reconstructing Backpropagation from Forward Fluctuations in Noise-modulated Neural Networks

**arXiv ID:** 2607.26483 | [PDF](https://arxiv.org/pdf/2607.26483v1)

**作者:** Shuhei Ikemoto `[一作]` `[通讯]` (Kyushu Institute of Technology), Shuhei Ikemoto (Kyushu Institute of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在噪声调制神经网络（NNN）中，通过仅利用前向传播时的噪声波动统计来估计权重的转置并递归地重构反向传播，实现无权重传输的学习；

**💡 创新点**

创新点在于提出了基于协方差权重镜像（covariance weight mirror）和局部梯度估计的前向统计学习规则，可在不使用后向路径或转置权重的情况下近似真实梯度，并在数字硬件上实现高效稀疏性；

**🔧 技术方法**

主要技术包括：噪声调制的交叉激活函数、前向协方差估计、权重镜像回归、局部导数的分布无关估计、梯度重构递归、Adam优化器与三阶中心矩校正等；

**📊 数据集**

使用的任务包括一维正弦回归、Friedman #1 回归、Two Moons 二分类、Concentric Circles 二分类等自生成的低维数据集；

**📈 对比分析**

与标准反向传播（backprop）对比，采用结构化协方差信用规则（Cov‐Mirror）和完整的协方差信用规则（Cov‐Full）在MSE、分类准确率上与backprop几乎无差距；相比简化的协方差规则（Cov‐Scal）性能明显逊色；

**⚠️ 局限性**

局限性在于实验规模较小，仅验证了浅层单输出网络，深层网络和多分类任务尚未评估；方法需要多次前向采样和Adam优化，计算与内存成本相对较高；

---

## 92. Polynomially Improved Lower Bounds for Trifferent Codes via Locally Sparse $3$-Uniform Hypergraphs

**arXiv ID:** 2607.26376 | [PDF](https://arxiv.org/pdf/2607.26376v1)

**作者:** Xuejiao Han `[一作]` (Capital Normal University), Gennian Ge `[通讯]` (Capital Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

论文证明了三元码的最大大小 T(n) 的多项式增强，具体为 T(n)≥ c√(n)(9/5)^n/4。

**💡 创新点**

创新点在于首次对经典的 Körner–Marton 下界进行了无界因子的改进，并通过改进外部编码构造来实现。

**🔧 技术方法**

使用了三元超图的随机稀疏化技术，并结合了 Verstraete 和 Wilson 的独立集定理。

**📊 数据集**

使用了长度为 n 的三元码，特别是 Tetra 码作为内部码。

**📈 对比分析**

与之前的上界和下界进行比较，证明了新的下界 T(n) 的增长速度优于之前的结果，且在大 n 的情况下表现出更好的性能。

**⚠️ 局限性**

限制在于该证明是非构造性的，尚未找到具有可比多项式改进的显式构造。

---

## 93. Evaluating Prompt Scope and Demonstration Similarity in Local LLM Machine Translation

**arXiv ID:** 2607.26286 | [PDF](https://arxiv.org/pdf/2607.26286v1)

**作者:** Mihael Arcan `[一作]` `[通讯]` (Home Lab), Mihael Arcan (Home Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究本地LLM在机器翻译中的提示范围（单目标vs.家族范围）和示例检索策略（随机、词汇相似、嵌入相似）对翻译质量和结构化多目标输出合规性的影响。

**💡 创新点**

创新点在于将提示范围和检索策略纳入评价变量，系统比较三种LLM与专用MT基线，并评估多目标JSON输出的完整性与合规性，揭示小模型在家族范围提示下易失效。

**🔧 技术方法**

使用本地指令调优LLM（如Llama、Claude等）进行零/少样本提示；采用词汇和嵌入相似度检索挑选示例；通过BLEU/chrF++/COMET自动评估翻译质量，并统计覆盖率、完整输出率等合规性指标。

**📊 数据集**

实验数据集为FLORES‑200的devtest（1012句）及dev示例池，覆盖9个欧盟语言（法语、西班牙语、意大利语、葡萄牙语、罗马尼亚语、德语、荷兰语、丹麦语、瑞典语）。

**📈 对比分析**

与OPUS‑MT、NLLB‑200等专用MT基线按配对bootstrap CI比较；结果显示专用MT在chrF++上始终领先，LLM在部分罗曼语种（西班牙语、葡萄牙语）相近；少样本对强模型有益但对最小模型有害；嵌入检索平均最佳但提升幅度有限；家族范围提示在大模型上可接受，小模型合规率显著下降。

**⚠️ 局限性**

局限性包括仅评估英语源语、两族语；仅使用自动指标未结合人工评估；模型固定、温度为0、单轮提示；未考察非拉丁文字、低资源语言或领域专用语料；对多目标翻译安全性缺乏实地验证。

---

## 94. MEDA: Measurement-Efficient Disorder-Aware Majorana Zero Mode Detection in Realistic Devices

**arXiv ID:** 2607.26208 | [PDF](https://arxiv.org/pdf/2607.26208v1)

**作者:** Nathan Jones `[一作]` (Clemson University), Rong Ge `[通讯]` (Clemson University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出MEDA框架，将稀疏测量的微分电导直接映射至无偏的周期性无序不变量以检测Majorana零模。

**💡 创新点**

创新点在于同时克服边界偏差与测量瓶颈，使用注意力驱动的多实例学习实现10倍测量量减少，并在真实无序条件下保持高精度。

**🔧 技术方法**

采用ResNet‑18编码器、注意力池化、CNN解码器以及多损失（BCE、Dice、Focal、TV）优化的深度学习模型。

**📊 数据集**

使用大规模模拟的1D半导体-超导纳米线传输数据，覆盖广泛的化学势、磁场、无序强度与相关长度。

**📈 对比分析**

与理论SMI上限和ViT基准对比，MEDA在10%测量数据下取得与SMI相近的F1/精度，显著优于ViT，尤其在中高无序相关长度下表现突出。

**⚠️ 局限性**

局限在于对高度碎裂的拓扑区域过度平滑，导致在极强无序下难以捕捉细小的拓扑岛，且需进一步调节正则化以平衡泛化与细节保留。

---

## 95. Reading Between the Curly Braces: On Textual Data Serialization Format Usability

**arXiv ID:** 2607.26211 | [PDF](https://arxiv.org/pdf/2607.26211v1)

**作者:** Shiyi He `[一作]` (University of Utah), Andrew M. McNutt `[通讯]` (University of Utah)

**通讯引用:** 576 | [OpenAlex ID](https://openalex.org/A5058039138)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过在众包平台进行的文本序列化格式可用性实验和对专业开发者的访谈，研究了格式语法差异与可用性之间的关系。

**💡 创新点**

发现语法差异对可用性影响有限，且可用性更受生态系统、工具、规范等社会技术因素主导，从而为“最佳格式”讨论提供了经验性视角。

**🔧 技术方法**

使用了Ace编辑器进行无工具干预的众包实验、手工评分和自动校验器，以及半结构化访谈结合affinity diagramming进行定性分析。

**📊 数据集**

采用了来自GitHub公开项目的真实数据集，包括JSON、YAML、XML、CSV、TOML等常见文本格式文件，用于阅读、生成和编辑任务。

**📈 对比分析**

采用两因素ANOVA和配对t检验对格式、任务和数据情境进行比较；结果显示无单一最佳格式，语法宽松格式（如HJSON、YAML）在特定编辑任务中略有优势，但总体差异不显著。

**⚠️ 局限性**

局限性包括任务过度简化、受试者主要为新手、对工具支持的控制有限、样本量不足以捕捉专家级差异，以及未能充分评估更复杂编辑环境（如IDE+linters）对可用性的影响。

---

## 96. Symphony of Bias: Exploring Gender Associations with Musical Instruments in Multimodal LLMs

**arXiv ID:** 2607.26355 | [PDF](https://arxiv.org/pdf/2607.26355v1)

**作者:** Farhan Farsi `[一作]`, Donya Rooein `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提供了如何使用ACL样式文件的示例，演示LuaLaTeX和XeLaTeX的使用方法。

**💡 创新点**

主要是模板演示，没有创新点。

**🔧 技术方法**

LuaLaTeX或XeLaTeX。

**📊 数据集**

无数据集。

**📈 对比分析**

无比较方法与性能评估。

**⚠️ 局限性**

仅为示例，缺少实际实验和评估。

---

## 97. Safety-Gated Autoscaling: A Multi-Layered Defense Architecture for Kubernetes Vertical Resource Optimization

**arXiv ID:** 2607.26503 | [PDF](https://arxiv.org/pdf/2607.26503v1)

**作者:** Azra Karakaya `[一作]` (Istanbul Medipol University), Ahmet Kaplan `[通讯]` (Istanbul Medipol University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个安全门控的 Kubernetes 垂直资源自动伸缩器（Intelligent Cluster Optimizer），整合多层安全管线、内存泄漏检测门、Holt‑Winters 预测、Pareto 多目标优化、干运行审批等功能，能在生产集群中安全地自动调整容器 CPU/内存请求。

**💡 创新点**

创新点包括：① 将异常检测作为伸缩门控（而非仅报警），实现内存泄漏检测门；② 设计五层安全管线（泄漏门、SLA 监控、电路断路器、HPA/PDB 冲突检测、策略引擎）；③ 对每个容器执行 Pareto 多目标优化并使用拥挤距离保持多样性；④ 提供干运行（dry‑run）模式让人工审批推荐；⑤ 采用开源、云无关的 Go Operator 与 REST/Web Dashboard 进行治理。

**🔧 技术方法**

采用技术包括：Go 语言实现的 Kubernetes Operator；Metrics API 采集 CPU/内存 60 s 采样；Holt‑Winters 三重指数平滑预测；线性回归 + R² 判断内存泄漏；统计百分位、Z‑score、IQR、移动平均做异常检测；Pareto 优化 + 拥挤距离；策略引擎、滚动更新、回滚机制、REST 及 Web Dashboard。

**📊 数据集**

使用数据集：1118 条自动化测试（覆盖率 80.3%）；在 Google Kubernetes Engine (GKE) 上部署的两种合成工作负载（stress‑master、stress‑cyclic）；Metrics API 收集的 CPU/内存历史（默认 24 h 保留）；以及 GCP 计费模型用于成本估算。

**📈 对比分析**

通过在 GKE 上与默认 VPA、KEDA、Goldilocks 等工具对比，评估维度包括：成本节约（预估 20–40%）、内存泄漏检测准确率（83%）、推荐准确率（90%）、安全层完整通过率、循环时延 ≤30 s、回滚时延 ≤60 s。实验显示垂直缩放后实际节约可达约 $2.82/月（基于预估），并在干运行模式下展示推荐与成本收益，整体性能符合设计预期。

**⚠️ 局限性**

局限性：① 评估仅在两台 e2‑small 节点、两种合成工作负载，未覆盖大规模异构生产集群；② Holt‑Winters 预测在长期稳定性上未充分验证；③ 内存泄漏检测准确率未达到 85% 目标，缺乏精度/召回等完整评价；④ 安全性未对抗性攻击进行测试；⑤ 缺乏消融实验验证各组件贡献；⑥ 与其他工具对比仅基于文档/what‑if 评估，未进行同步跑测；⑦ 仅在 GCP 上测试，缺乏跨云平台验证。

---

## 98. ClockRoPE: Random Fourier Rotations for Temporal Routine Modeling

**arXiv ID:** 2607.26369 | [PDF](https://arxiv.org/pdf/2607.26369v1)

**作者:** Yiwen Chen `[一作]` (YouTube), Qian Sun `[通讯]` (YouTube)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 ClockRoPE——一种基于随机 Fourier 旋转的周期性位置编码，用于提升基于 Transformer 的序列推荐模型对用户日常/周期性行为的建模。

**💡 创新点**

创新点在于把 Bochner 定理与 RoPE 结合，证明任意正定归一化的时间相关函数都可由其 Fourier 变换采样得到的随机旋转近似，从而构造出能捕捉周期性注意力调制的 ClockRoPE，并与传统 RoPE 成功协同。

**🔧 技术方法**

核心技术包括：随机 Fourier 旋转（Random Fourier Rotations）、周期性正定核的 Fourier 系列、对称与折叠频率分布、以及多周期（每日、每周）分层采样；模型实现基于标准 Transformer 与 RoPE 的多头注意力。

**📊 数据集**

在大型视频推荐平台的生产环境中，使用该平台的历史用户交互序列和绝对时间戳数据进行训练与评估（无公开数据集，属于内部数据）。

**📈 对比分析**

离线实验：在无 RoPE 基线上，ClockRoPE-高斯折叠版本在 MAP@1 提升 3.61%、MAP@50 提升 2.25%；与 RoPE 组合后进一步提升 2.39%/2.00%；在线 A/B 测试（14 天）显示与 RoPE 组合后，总体价值互动提升 0.08%。

**⚠️ 局限性**

局限性：需要针对不同周期调节截断大小和方差；目前只针对周期性行为有效，非周期性长程依赖需仍靠 RoPE；模型训练时随机旋转采样可能导致训练不稳定，且在多周期混合时的参数选择仍需经验。

---

## 99. ExplainBench: Evaluating Code Explanations from Agents

**arXiv ID:** 2607.26451 | [PDF](https://arxiv.org/pdf/2607.26451v1)

**作者:** Zhiyuan Pan `[一作]` (Zhejiang University), Abhik Roychoudhury `[通讯]` (National University of Singapore)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ExplainBench，一个自动评估 LLM 代理补丁说明质量的基准，并对五个开源代理进行了评估。

**💡 创新点**

创新点在于①构建了基于 LLM 问答的多选题评估框架；②利用执行跟踪与差分测试生成可验证的问题；③提出了解释审计代理（Audit）来自动改进说明的可信度。

**🔧 技术方法**

使用技术包括 GPT‑5‑mini 进行问题回答与表达式生成、GPT‑5‑mini 生成 PBT 与候选表达式、Python 代码执行跟踪与差分执行、以及差分测试与调用图分析。

**📊 数据集**

数据集基于 SWE‑bench Verified，经过筛选后得到 297 个可测试的真实世界 bug 案例。

**📈 对比分析**

通过计算解释得分（正确回答比例）与 SWE‑bench 修复率对比，发现两者无相关性；在所有代理中，OpenHands 的解释得分最高；应用 Audit 后，平均提升约 10.9%，说明了审计机制的有效性。

**⚠️ 局限性**

局限性包括：仅评估行为一致性与覆盖率，未考虑可读性、可维护性等质量维度；依赖 SWE‑bench Verified，可能不适用于无测试或极端复杂的项目；评估过程受 LLM 随机性与误差影响，需要人工核对。

---

## 100. ClinLens: Towards Long-Horizon Coding Agents for Longitudinal Multimodal Clinical Data Science

**arXiv ID:** 2607.26155 | [PDF](https://arxiv.org/pdf/2607.26155v1)

**作者:** Yuan Zhu `[一作]` (Shandong University), Jindong Han `[通讯]` (Shandong University)

**通讯引用:** 361 | [OpenAlex ID](https://openalex.org/A5027644323)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一个新的可执行临床数据科学评测基准，涵盖200个基于MIMIC五大数据源的任务，涵盖患者、入院、ICU、事件四个时间维度与五类分析能力；并生成了可验证的参考工作流和 artifact 检查。

**💡 创新点**

创新点在于：①将多模态纵向临床数据（结构化记录、文本、心电图、胸片、超声）按患者层级链接；②设计4×5的患者‑时间与分析能力维度；③使用“程序优先逆向合成”方法自动构造任务与参考流程；③引入可追踪的 artifact 检查，量化执行‑正确性差距。

**🔧 技术方法**

技术手段包括：程序优先逆向合成、LLM 代理框架（ReAct、DataVoyager、Reflexion、SelfDebug、RAG‑ReAct）、基线模型（DeepSeek‑V4‑Pro、GLM‑5.2、GPT‑5.5、Claude‑Opus‑4.8）、编程评估指标（执行成功、宏观准确率）以及对比实验设计。

**📊 数据集**

使用的数据集为：MIMIC‑IV 结构化记录、MIMIC‑IV‑Note 文本、MIMIC‑IV‑ECG 心电图、MIMIC‑CXR‑JPG 胸部 X 光图像、MIMIC‑IV‑ECHO 超声心动图，全部保持原始标识与时间戳。

**📈 对比分析**

对比方法：在固定的126题实验套件上，评估24种模型‑脚手架组合，报告各时间维度与总体宏观准确率；最优组合 GPT‑5.5+SelfDebug 的宏观准确率为56.3%，而单一编程代理可解决83题；GPT‑4o‑mini 的适配医学代理仅达2.9% 宏观准确率，凸显执行‑正确性差距。

**⚠️ 局限性**

局限性包括：仅来自单一医院的 MIMIC 数据，缺乏专家审计与多轮实验；实验仅限一次运行，未考虑模型泛化与鲁棒性；参考工作流与检查严格，可能难以迁移至更大或不同的数据集；以及对某些任务的时间与内存限制。

---

## 101. DuplexGen: Adaptive Synthesis of Human-AI Turn-Taking Dialogues

**arXiv ID:** 2607.26178 | [PDF](https://arxiv.org/pdf/2607.26178v1)

**作者:** Takyoung Kim `[一作]` (University of Illinois Urbana Champaign), Dilek Hakkani-Tür `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个针对不同场景的自适应对话生成框架，能够在生成的双向语音对话中插入符合人类偏好的回声或占位行为。

**💡 创新点**

创新点在于通过少量（每个场景约 10–20 条）槽位级人类偏好标注对大模型进行情境感知校准，从而实现场景特定的交互节奏，而非单一通用模式。

**🔧 技术方法**

技术包括：文本转语音风格的转换、基于语言模型的槽位识别、以 KL 散度为目标的软标签校准（fine‑tune Qwen3/ GPT‑4 生成分布），以及在后续对话合成中根据校准模型插入回声/占位标记。

**📊 数据集**

数据集：使用六种任务（教学、规划、访谈、谈判、劝说、社交聊天）中的公开文本数据集（如 SODA、OpenAI 预训练对话），并通过本框架合成语音风格对话；人类标注由 248 名英语使用者完成。

**📈 对比分析**

比较方法：将校准模型与 Prompt‑Only、Switchboard‑only、以及 Switchboard‑+校准模型进行 KL 散度对比；在全双工模型上与 Moshi、PersonaPlex 基线进行单轮和多轮对话评估。结果表明校准模型在预测分布上显著更贴近人类偏好，PP‑DG（使用本框架合成数据训练的 PersonaPlex）在多轮评估中在指令遵循和自然度上均显著优于基线。

**⚠️ 局限性**

局限性：合成数据覆盖的场景和交互类型有限；对训练超参数（延迟、停顿长度、学习率）高度敏感；当前校准仅基于文本，未考虑语音韵律或视觉线索，未来需要多模态标注与校准。

---

## 102. Archetypes or ability? Clustering for modelling student mathematical competence

**arXiv ID:** 2607.26063 | [PDF](https://arxiv.org/pdf/2607.26063v1)

**作者:** Benjamin Mawdsley `[一作]` (Stfc Hartree Centre), Paul Edwards `[通讯]` (Stfc Hartree Centre)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对英国13场全国性考试的119,034名学生模拟考试结果进行聚类分析，探讨学生数学能力是否可分解为离散的能力集合。

**💡 创新点**

首次在如此大规模数据上使用可解释的贝叶斯混合模型（BMM）挖掘学生能力结构，并将其与传统逻辑回归、kNN 等基线模型做对比，提供了新的教育机器学习基准。

**🔧 技术方法**

使用Bernoulli混合模型、k-最近邻、单能力与多能力逻辑回归、EM算法、BIC模型选择、交叉验证、概率校准等技术。

**📊 数据集**

数据来自英国学校的13场GCSE模拟考试，共119,034名学生的问答记录，采用通过/失败二值化处理。

**📈 对比分析**

与单能力逻辑回归、全问答多能力逻辑回归、kNN和BMM等模型比较，准确率约为78%，与更复杂的深度学习模型相当；模型的提升主要来源于整体能力，而个别题目级别的差异提升有限。

**⚠️ 局限性**

局限包括缺乏学生人口统计信息导致无法检测性别/背景偏差；模型在中等能力学生上的预测准确率较低；聚类结果大多是整体能力的线性缩放，未发现明显的离散能力分组；未考虑额外特征如学习视频等。

---

## 103. Diagnosing Fine-Grained Inconsistency Classification in Financial Disclosure Text

**arXiv ID:** 2607.26368 | [PDF](https://arxiv.org/pdf/2607.26368v1)

**作者:** Aman Kumar `[一作]` (Hitachi America, Ltd.), Ahmed K Farahat `[通讯]` (Hitachi America, Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在金融披露文本中研究细粒度不一致类型分类，使用合成数据集评估多种模型；

**💡 创新点**

对比冻结编码器、微调编码器、证据增强分类器、提示式LLM和LoRA适配的生成模型，并揭示定位与类型区分的瓶颈；

**🔧 技术方法**

采用EmbeddingGemma、QwenEmbedding、DeBERTa等句子编码器、问答抽取器、MLP分类头、指令调优LLM和LoRA微调；

**📊 数据集**

使用SBID‑FD合成金融披露基准（5,940条实例，11类标签，包含参考证据跨度）；

**📈 对比分析**

在统一评估协议下，微调的300M编码器在精度上与规模更大的提示LLM和LoRA模型相当（≈61–62%），并显示证据提供可提升约3个百分点；

**⚠️ 局限性**

局限在于数据为合成短段，缺乏真实披露中的表格、跨文档冲突，且假设两条精准语句可定位，定位器的精度仍待提升，未评估误检与误分类的实际影响。

---

## 104. GuideSkill: Evolving Executable LLM Agent Skills for Guideline-Grounded Clinical Reasoning

**arXiv ID:** 2607.26160 | [PDF](https://arxiv.org/pdf/2607.26160v1)

**作者:** Lang Cao `[一作]` (University of Illinois Urbana Champaign), Yue Guo `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出GuideSkill，一种外部可执行的诊断技能库，将临床实践指南中的诊断规则编译为可调用的函数，并通过病例标注进一步演化和扩充诊断覆盖；在推理时将LLM生成的候选诊断与技能执行结果融合以提高诊断准确率。

**💡 创新点**

创新点在于将指南知识抽象为可执行的病种特异性技能，并在不更新LLM参数的情况下，通过案例演化自动扩展诊断覆盖率，从而实现可复用、可解释且可迁移的诊断决策层。

**🔧 技术方法**

技术包括指南文本解析与映射、LLM编译为可执行Python函数、基于案例的规则提炼与技能优化、候选诊断生成与技能评分融合（α融合系数）以及批量特征提取提升推理效率。

**📊 数据集**

使用了四个诊断推理基准：MedCaseReasoning、ER-Reason、MIMIC-CDM-FI、MedThink-Bench；以及多种LLM后端（GPT‑5.4、Claude‑Sonnet‑4.6、MedGemma‑27B、Qwen3.5‑9B）。

**📈 对比分析**

与直接推理、Chain‑of‑Thought、3‑shot ICL、RAG、LLM‑DDx及其结合等基线相比，GuideSkill‑Zero在未演化阶段已显著优于RAG；GuideSkill‑Evo在所有四个模型和四个基准上实现宏平均准确率提升6.98–8.39个百分点，且在Qwen3.5‑9B上超过所有参数更新或结构化指南基准（+5.12个百分点）。

**⚠️ 局限性**

局限性包括：仍依赖LLM生成的候选诊断，导致候选召回是主要错误来源；技能演化依赖人工评估以防止规则过度泛化；以及对罕见病和未覆盖指南的诊断依赖案例演化，可能需要大量标注数据。

---

## 105. Graphene-based Hemispherical Transmitarray Antenna for Wide-Angle Beam Steering and Ultrafast Moving Target Tracking

**arXiv ID:** 2607.26437 | [PDF](https://arxiv.org/pdf/2607.26437v1)

**作者:** Somayeh Komeylian `[一作]` (University of California San Diego), Christopher Paolini `[通讯]` (San Diego State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种基于石墨烯可调谐的半球形传输阵天线，可在250 GHz实现–87°至87°的仰角、360°方位全覆盖的宽角度电子波束扫描和超快移动目标跟踪。

**💡 创新点**

创新点包括①利用石墨烯电压可调表面电导实现纳秒至微秒级波束切换；②采用半球形多层复合结构（金属FCC贴片–hBN介质–石墨烯）实现高效、宽带的波前控制；③构建完整的解析模型（Kubo–Drude、等效RLC、GSTC/Huygens、ABCD矩阵、球面阵列因子），并实现从期望波束方向到石墨烯偏置电压的直接映射；④可扩展的11×11电压控制矩阵与FPGA实现全镜面实时重配置。

**🔧 技术方法**

采用石墨烯表面电导的Kubo–Drude模型、等效RLC电路、Generalized Sheet Transition Condition (GSTC) 与Huygens表面概念、ABCD矩阵分析、球面阵列因子计算；仿真使用CST Microwave Studio；偏置控制通过Xilinx Zynq UltraScale+ RFSoC ZCU670平台的多通道高速DAC与FPGA实现。

**📊 数据集**

论文未使用公开数据集，而是基于自定义仿真参数（频率250 GHz、化学势0.1–1 eV、半球半径等）和实验设置进行性能验证。

**📈 对比分析**

通过与传统反射阵、平面元胞、频率依赖、机械扫描等技术的对比表格，显示本设计在扫描速度(ns–μs)比机械扫描(ms–s)快、扫描范围(±87°)远超传统±30°–±60°，以及效率(68%–80%)显著高于常规石墨烯天线(16%–66%)，证明了在宽角度、效率与高速三维指标上的优越性。

**⚠️ 局限性**

主要限制包括：①材料损耗（金属与hBN）对效率仍有限制；②多通道高速DAC与FPGA的硬件复杂度和功耗；③热管理与大规模阵列的可扩展性尚未深入探讨；④目前仅在250 GHz验证，其他频段的可行性和性能尚未评估。

---

## 106. Learning Implicit Causal World Models from Multi-Agent Demonstrations

**arXiv ID:** 2607.26336 | [PDF](https://arxiv.org/pdf/2607.26336v1)

**作者:** Jasorsi Ghosh `[一作]` `[通讯]` (Purdue University), Jasorsi Ghosh (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出隐式因果世界模型（ICWM），通过在多智能体离线演示中加入软干预（策略方差）学习可解释且对分布外更鲁棒的环境动力学。

**💡 创新点**

创新点在于：① 利用软干预的“介入强度”σ满足多智能体顺序后门条件，使得在没有先验因果图的情况下可辨识真实因果结构；② 将PCMCI+与可微因果掩码结合，直接在神经网络中编码因果骨架；③ 证明介入强度对图结构质量和OOD性能呈倒数关系。

**🔧 技术方法**

技术手段包括：神经网络构建因果掩码、PCMCI+时序因果发现、软干预策略（σ-混合分布）、可微掩码门控、对比实验中的损失函数与谱距离评估。

**📊 数据集**

使用三类仿真多智能体协调任务（Two‑Door、Navigation、Giveway），覆盖完整与部分可观测、2/3/4 代理，训练集为不同σ的离线演示。

**📈 对比分析**

与关联性世界模型、FOCUS、BC‑WM、R‑ICWM、C‑VAE等基线对比，ICWM 在 OOD 随机/对抗策略下 MSE/Var 下降两位数；图结构误差随 σ 降低，且在部分可观测和布局外推时表现优于无图模型；在多智能体规模增长时误差保持 ~10⁻³。

**⚠️ 局限性**

局限性在于：需要在训练中注入随机干预，可能破坏高度协作的演示；介入强度对几何内动力学识别有效，但对空间布局变更的泛化无效；缺乏对真实机器人安全部署的验证。

---

## 107. Sensitivity and Differential Privacy in Metric Voting with Distortion below Three

**arXiv ID:** 2607.26388 | [PDF](https://arxiv.org/pdf/2607.26388v1)

**作者:** Shinsaku Sakaue `[一作]` (CyberAgent), Yuichi Yoshida `[通讯]` (National Institute of Informatics)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于 Gibbs 分布的随机化选举规则，能够在未知度量空间下以比 3 更小的度量失真（即 3-ε）同时保持低极端敏感性和近似差分隐私。

**💡 创新点**

创新点在于：① 将有限随机性（bounded‑randomness）与偏度量（biased‑metric）框架结合，证明存在常数大小的列表可实现 3-ε 失真；② 通过在这些列表上构造 Gibbs 分布，并通过温度参数调节，得到在任意候选人数量 m 与选民数量 n 下的最坏情况敏感性 O((log m+1)/n) 以及 (ε,δ)‑差分隐私 ε≈O((log m+log(1/δ)+1)/n) 的保障；③ 采用了对低能级子集的稳定性分析与 Gibbs 尾部界，完成了敏感性和隐私的全局证明。

**🔧 技术方法**

主要技术包括：偏度量比值（λ_σ(L)）的定义与性质、bounded‑randomness 定理提供低失真列表、Gibbs 分布（指数机制）在列表空间上的构造、对 λ 的局部漂移（ratio change bound）的证明、Gibbs 分布的权重稳定性与尾部界、以及在这些界定下的 Wasserstein 敏感性和差分隐私分析。

**📊 数据集**

本工作完全为理论研究，无实验数据集；所有结果均为数学证明，依赖于假设的度量空间与排名信息。

**📈 对比分析**

与传统的随机独裁（Random Dictatorship）以及其它已知随机规则（如随机化的 Kempe 规则）相比，本规则在失真方面保持 3‑ε（小于 3），而传统规则的失真会随 n 或 m 趋向 3。敏感性方面，本规则的最坏情况敏感性为 O((log m+1)/n)，远优于传统规则的 Ω(1)；差分隐私方面，给出了 (ε,δ)‑DP 的上界 ε≈O((log m+log(1/δ)+1)/n)，这是前所未有的结果。整体性能表现显示：在任何规模的选举中都能在 1/n 的尺度上实现高效、隐私兼顾的决策。

**⚠️ 局限性**

限制与开源问题：① 规则的实现需要枚举所有 m^K 个列表，计算复杂度指数级；② 常数 ε 与温度参数的具体数值未给出，实际应用需进一步实验；③ log m 因子是否必要仍是未解决的开放问题；④ 规则仅满足低失真、敏感性和差分隐私，未探讨策略性、Condorcet 兼容性或参与性等其他社会选择性质。

---

## 108. Identifying Implicit Bias in LLM-based Chat AI Toward People with Intellectual Disabilities

**arXiv ID:** 2607.26062 | [PDF](https://arxiv.org/pdf/2607.26062v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 109. Strong imposition of Dirichlet boundary velocities in structure-preserving discretizations of elastodynamics

**arXiv ID:** 2607.26248 | [PDF](https://arxiv.org/pdf/2607.26248v1)

**作者:** Cristobal Ponce `[一作]` (Universidad Tecnica Federico Santa Maria), Yann Le Gorrec `[通讯]` (Universite Marie et Louis Pasteur)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出一种连续-离散的运动学升维框架，利用端口哈密顿体系理论实现对Dirichlet边界速度的强制施加，得到仅含ODE的结构保持有限元模型；

**💡 创新点**

创新在于将边界施加内化为分布式端口，借助运动学升维分解避免代数约束，且离散后可与传统有限元矩阵分块等价；

**🔧 技术方法**

使用端口哈密顿体系、Hamilton‑Pontryagin与Hu‑Washizu虚功变分原理、有限元离散化、矩阵分块、能量保持积分法等技术；

**📊 数据集**

通过自定义的1D Timoshenko梁、几何非线性二维Neo‑Hookean框架以及几何精确字符串等仿真模型（无公开数据集）验证方法；

**📈 对比分析**

与传统有限元、混合有限元以及DAE方案对比，结果显示锁定消除、能量守恒良好、能量平衡保持，计算复杂度与标准方法相当或更低；

**⚠️ 局限性**

局限在于仅考虑弹性无粘性系统，未涵盖多物理耦合、非线性控制或观测器设计，且对大型3D大尺度问题的可扩展性尚待验证。

---

## 110. Knowledge-guided Disentanglement with Atomic Actions for Action Recognition

**arXiv ID:** 2607.26097 | [PDF](https://arxiv.org/pdf/2607.26097v1)

**作者:** Tianci Wu `[一作]` (Xidian University), Liang Zhang `[通讯]` (Xidian University)

**通讯引用:** 31935 | [OpenAlex ID](https://openalex.org/A5100425201)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大语言模型将动作标签拆解为原子动作，结合结构化视频表示（SSG）进行知识注入与分离，实现多动作情境下的精细动作识别。

**💡 创新点**

提出分层知识分离框架（KIM、KDM、KD Loss），在特征、结构和语义层面引入原子动作知识，显著提升特征判别力与分离效果。

**🔧 技术方法**

大语言模型（LLM）生成原子动作、场景图（SSG）编码、知识注入模块、知识分离模块、知识分离损失、CLIP文本编码器、ViT/ResNet-101等视觉网络。

**📊 数据集**

在Charades（多标签动作识别）和SportsHHI（人际交互分类）两个基准数据集上进行实验。

**📈 对比分析**

与现有方法（ProDA、OR2G、LaIAR、VicTR等）对比，Oracle设置下mAP 73.2% 超越ProDA 71.1%，标准设置下性能与ProDA相近；在SportsHHI上加入KIM/KDM后mAP提升至11.70%（相较基线10.16%）。

**⚠️ 局限性**

对场景图质量敏感；视觉与文本特征跨模态对齐仍存在问题；需要LLM生成原子动作及维护大规模知识库，计算成本较高。

---

## 111. Contextualized Counterspeech Can Be More Persuasive Than Generic Counterspeech

**arXiv ID:** 2607.26236 | [PDF](https://arxiv.org/pdf/2607.26236v1)

**作者:** Lorenzo Cima `[一作]` (University of Pisa and IIT-CNR), Stefano Cresci `[通讯]` (IIT-CNR)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何通过在生成对抗性言论（counter‑speech）时加入会话上下文和用户个性化信息，以提升对抗性言论的适当性、说服力与自然度。

**💡 创新点**

提出多种上下文化与个性化生成策略，并系统评估其对生成对抗性言论质量的影响，揭示轻量级上下文提示能有效提升效果，而过度微调则可能适得其反。

**🔧 技术方法**

使用基于 LLaMA‑2‑13B 的指令微调模型，并结合 ROUGE、BLEU、BERTScore、Perspective API 等自动指标以及混合设计的 MTurk 人工评测。

**📊 数据集**

数据来源于美国政治议题的 5 个 Reddit 子版块，筛选 128 条毒性评论及其 2,912 条人工对话，用于生成 4,608 条对抗性言论。

**📈 对比分析**

通过算法指标与人工评估两阶段筛选，最终比较 7 个关键配置；结果显示配置 [Base+Context+History] 与 [Base+Context] 在人工评估中最具说服力和适当性，但与自动指标存在负相关。

**⚠️ 局限性**

局限性包括：仅针对美国政治 Reddit 语料，毒性类型偏向粗俗/侮辱；算法指标与人类评估不一致；未测量实际行为改变；微调可能引入毒性或缺乏调解框架。

---

## 112. An Efficient Algorithm for Computing Mountain Prominence in Almost Linear Time

**arXiv ID:** 2607.26496 | [PDF](https://arxiv.org/pdf/2607.26496v1)

**作者:** George Alex Dumitrescu `[一作]` (Alexandru Ioan Cuza University of Iasi), Paul Flavian Diac `[通讯]` (Alexandru Ioan Cuza University of Iasi)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种近线性时间的水浸算法，利用分块（瓦片）处理DEM，计算地球上所有山峰的 prominence，并通过只保留每块瓦片的外部峰和边缘信息来大幅减少存储与计算量。

**💡 创新点**

创新点在于证明只需记录每块瓦片的外部峰及其边缘即可确定所有峰的真正 prominence，从而避免构建全局树；采用分治合并树的方式，降低了复杂度至几乎线性；同时在瓦片内部预处理使外部峰数目极低。

**🔧 技术方法**

使用技术包括水浸（water sweep）算法、并查集（union‑find）、计数排序、分治（divide and conquer）合并策略、树结构存储外部峰与边缘，并在瓦片合并时再次进行水浸。

**📊 数据集**

使用的数据集为 3 arcsecond（约 90 m）SRTM 数字高程模型，共 576 个 1°×1° 瓦片，涵盖喜马拉雅等高山地区。

**📈 对比分析**

实验与 Kirmse & de Ferranti 2017 年的算法对比，基准测试在 24–39°N, 72–107°E 区域完成于 11 min 7 s，水浸单瓦片耗时最多；算法在不剪枝的情况下即可一次性得到所有峰值，整体复杂度为 O(T log T·H + log T·T·n)，显著优于未剪枝的全局树方法。

**⚠️ 局限性**

局限性包括尚未充分利用多线程与剪枝技术；对更高分辨率 DEM 的适配仍需改进；外部峰虽少，但在更大范围时可能仍显高；SRTM 数据缺陷（空洞、误差）仍会影响结果。

---

## 113. Automorphism-Induced Non-Canonicity in Top-k Explanations of Graph Neural Networks

**arXiv ID:** 2607.26344 | [PDF](https://arxiv.org/pdf/2607.26344v1)

**作者:** Xin Xu `[一作]` (Carnegie Mellon University), Kaizhen Tan `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统研究了图神经网络解释器在面对图中对称结构时产生的不确定性，证明了基于梯度或掩码的解释方法在对称输入下必然产生相同的归因，却在报告子图时因排序等实现细节而任选其中一个对称部件；进一步提出了判定是否必然需要分裂轨道的无参数判据，并在 Lean4 中实现了形式化证明；最后给出轨道感知的报告策略，消除这种随意性并改进可解释性评估指标。

**💡 创新点**

主要创新点包括①对称性导致的三元困境（单值、最小化、对称性不可兼得）的正式阐述；②构造了可从图本身判断是否必须分裂轨道的参数无关判据，并在 Lean4 形式化验证；③提出并验证了轨道感知的 top‑k 报告方法，可消除实现层面的任意性；④揭示了现有稳定性指标无法区分可解释性与实现差异。

**🔧 技术方法**

技术上结合了图论中的自同构群与消息传递等价性理论、归因方法（Integrated Gradients、Saliency、GNNExplainer 等）、Lean4 形式化工具、以及机械模型等价性检查；实验采用了多种解释器实现、不同的 GNN 架构（GCN、GINE、GAT、GINet）和多种数据集。

**📊 数据集**

使用了公开化学分子数据集 MUTAG、Mutagenicity、PROTEINS、BBBP；以及人工合成图数据集 Tree‑Grid、Tree‑Cycles；还在这些数据上训练了 3 层 GCN、GINE、GAT 等网络进行评估。

**📈 对比分析**

通过将解释结果与机械等价性检查（即将报告的子图及其自同构映射送入冻结模型比对输出）进行对比，验证判据的准确性（在所有测试决策上 100% 匹配）；评估了不同解释器在 top‑k 预算下分裂轨道的比例、稳定性指标（交叉跑 Jaccard）以及可解释性质量；结果显示约 10‑30% 的实例在报告时会任意挑选对称部件，轨道感知报告将此比例降至 0%。

**⚠️ 局限性**

局限性包括：只针对 3 层或更浅的 GNN 进行实验，未考虑更深或 Transformer‑style 网络；只关注顶点/边的 top‑k 报告，未扩展到连续或多分辨率解释；使用的解释器主要为公开实现，未设计新的可解释性算法；synthetic benchmark 的生成方式对对称性影响有限，可能低估了实际情况。

---

## 114. Large-Scale ChatBot Validation Through Customer Digital Twin Simulations

**arXiv ID:** 2607.26060 | [PDF](https://arxiv.org/pdf/2607.26060v1)

**作者:** Cristovao Iglesias `[一作]` (NatWest AI Research), Raad Khraishi `[通讯]` (NatWest AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于真实交易与对话数据的高保真合成客户代理（SCA）方法，并开发了利用SCA进行大规模聊天机器人验证的框架，结合自动LLM-as-a-Judge评估、人类专家测试和对抗性探测，验证了银行业务聊天机器人的安全性与合规性。

**💡 创新点**

创新点包括：①将合成客户代理视为数字双胞胎，在保持语义一致性的同时可通过情感或行为干预生成多样化的客户对话；②设计了可扩展的验证框架，将LLM评估与人工审核和对抗测试融合，实现快速、可重复的合规性评估；③在监管金融场景下提供了可量化的多维指标（语义相似度、误报率、对抗鲁棒性等），首次实现了对聊天机器人在不同情绪、人口统计与语言复杂度下的系统性验证。

**🔧 技术方法**

主要技术包括：大语言模型（GPT‑4.1）作为合成客户代理与评估者；结构化系统提示工程，支持历史上下文与人格干预；LLM-as-a-Judge自动评估九维度指标；人工专家测试与对抗性红蓝团队实验；对话模拟与指标统计平台。

**📊 数据集**

使用了两批匿名客户对话数据集：600条用于语义相似度与词汇重叠评估，750条用于事实忠实性与误报率评估；此外还对真实客户的人格特征（Big Five）进行了基准对照。数据均来自英国一家大型银行的历史交互记录，已去标识化。

**📈 对比分析**

对比方法：将SCA生成的对话与真实对话进行语义相似度（余弦相似度≈0.85）和BLEU分数比较，事实忠实性由LLM-as-a-Judge评估并计数误报，个性化对齐通过IPIP‑NEO‑300评分测量。结果显示：误报率仅3.2%，语义相似度高但词汇差异显著；在情绪与人口统计差异下，聊天机器人保持了高达F1≈0.92的任务性能，且无显著公平性偏差。

**⚠️ 局限性**

局限性包括：①评估仍依赖LLM的判断，存在主观误差；②对极端稀缺场景（如极低语种、极端情绪）测试不足；③数据集规模相对有限，未覆盖所有监管要求的边缘案例；④合成客户在细粒度行为细节上仍可能出现微小缺失或少量幻觉；⑤在真实业务部署前仍需进一步的跨机构验证与成本评估。

---

## 115. Two2Four: Generative Quadruped Puppeteering from Human Motion

**arXiv ID:** 2607.26108 | [PDF](https://arxiv.org/pdf/2607.26108v1)

**作者:** Fatemeh Zargarbashi `[一作]` (Disney Research Studios), Jakob Buhmann `[通讯]` (Disney Research Studios)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出两阶段扩散模型（Trajectory + Full-body）实现人类运动驱动的四足动物动画，并提供精细的头部、腿部和姿态控制。

**💡 创新点**

创新点：① 用稀疏轨迹先导再映射全身的两阶段扩散分解；② 结合结构化条件与inpainting，使得精确与模糊的运动特征都可在生成时实时控制；③ 在仅有约40分钟四足数据的情况下实现鲁棒的跨形态迁移。

**🔧 技术方法**

使用条件扩散（U‑Net + DDIM）、轨迹与全身两阶段设计、inpainting控制、姿态归一化、逆运动学约束、自动回归生成。

**📊 数据集**

训练数据：约40分钟四足动物运动捕捉数据（走、跑、跳、坐、躺等）和 Human‑Loco 等人类动作数据。

**📈 对比分析**

与 Dog Code、Walk‑the‑Dog、Motion2Motion 等基线对比；艺术家定性评分、FID 与脚滑指标；结果显示：两阶段模型在人-四足转化中的 FID≈8（比 Walk‑the‑Dog 的 14.7 更好），脚滑率也显著降低。

**⚠️ 局限性**

局限：对人类倒退动作的支持不足，导致滑动；受限于小规模四足数据，未完成自动模式切换；对非犬类四足的泛化仍需更多数据。

---

## 116. Reeling It In: Flexible Needle Pick Up via Thread Manipulation for Autonomous Suturing

**arXiv ID:** 2607.26337 | [PDF](https://arxiv.org/pdf/2607.26337v1)

**作者:** Emma Huang `[一作]` (University of California San Diego), Michael C. Yip `[通讯]` (University of California San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一套完整的自主缝合过程中针头拾取框架，通过利用缝线的几何特征和不确定性信息，先对缝线与组织进行三维重建，再选取可靠的抓取点，采用圆形提拉、拉长及双手捕鱼的机械动作，最终实现针头的安全拾取。

**💡 创新点**

创新点在于：①利用缝线重建的可靠性指标实现安全抓取点选择，显著降低组织损伤；②引入圆形提拉与拉长运动抑制针头拖拽，提升操作安全性；③设计双手协作的捕鱼策略，克服针头不可直接接近导致的拾取难题。

**🔧 技术方法**

主要技术包括：双目立体视觉与缝线分割、三维点云重建、基于约束优化的缝线样条拟合与可靠性评估、距离约束下的抓取点优化、基于重心和重力的圆形提拉与拉长运动规划、双手协作的基于不确定性锥的捕鱼路径规划。

**📊 数据集**

实验使用了真实的 dVRK（Da Vinci Research Kit）平台，搭载 36 mm/2 圆弧针头与 45 cm 缝线，设置四种实验情景（Easy、Medium、Hard、Occlusion）进行评估。

**📈 对比分析**

与直接提拉和传统直接抓取方法对比，本文实现了 95% 的成功率（Easy/Medium）、70%（Hard）和 65%（Occlusion）拾取成功率；在可靠性评估消除误差的 ablation 试验中，使用可靠性指标的抓取点成功率提升了 53.33%。

**⚠️ 局限性**

主要限制包括：在高遮挡或缝线重叠复杂的情境下重建不准确导致抓取点不足；线缆松弛或环节导致捕鱼过程中的不确定性过大，易错；双手协作时需要更精细的姿态估计与实时反馈，现有方法对实时性和鲁棒性仍有提升空间。

---

## 117. WildShadowRemover: In-the-Wild Video Shadow Removal via Detail-Preserving Video Diffusion Models

**arXiv ID:** 2607.26203 | [PDF](https://arxiv.org/pdf/2607.26203v1)

**作者:** Jiamin Xu `[一作]` (Hangzhou Dianzi University), Gang Xu `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 83256 | [OpenAlex ID](https://openalex.org/A5055010081)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了WildShadowRemover框架，实现了在真实环境下的视频影子去除，结合预训练视频扩散模型、LoRA微调、细节注入、频率分解调制以及深度几何引导；

**💡 创新点**

首次将预训练的视频扩散模型用于影子去除任务，并提出了基于影子掩码的频率分解调制与细节注入机制，利用单目深度先验实现几何感知；

**🔧 技术方法**

采用视频扩散模型Wan‑Control、LoRA参数高效微调、VAE编码解码、流匹配训练、频率分解调制（FDM）、细节注入模块、RRDB融合、DA3深度先验等技术；

**📊 数据集**

使用自行构建的大规模合成影子视频数据集WildShadow（包含3D‑FRONT、iCity、Infinigen等场景）及其图像版WildShadow‑I，同时在ISTD+、SRD、INS等公开基准上进行评估；

**📈 对比分析**

与多种基准方法（如ShadowFormer、DMTN、StableShadowRemoval、OmniSR等）在图像和视频影子去除任务中对比，实验证明在INS和WildShadow‑I上取得最高PSNR/SSIM，视频版本相比OmniSR帧级实现具有更佳的重建质量和更低的时间一致性误差；

**⚠️ 局限性**

在高度复杂几何结构（如浓密叶片、细小结构）场景下仍易出现残留阴影或细节丢失，并且模型主要依赖合成对照数据，缺乏对大规模真实视频的自监督或弱监督训练能力。

---

## 118. CaM-Wolf: Causal-Aware Multimodal Agents for Social Deduction Games

**arXiv ID:** 2607.26393 | [PDF](https://arxiv.org/pdf/2607.26393v1)

**作者:** Zheng Zhang `[一作]` (Hong Kong University of Science and Technology), Hao Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并评估了CaM-Wolf——一种集成视频感知、因果推理与动画头像生成的社交推理游戏（狼人杀）AI代理。

**💡 创新点**

①首次在SDG代理中实现双向多模态交互；②采用因果意识Reasoner并通过强化学习与反事实干预来生成可信的推理链。

**🔧 技术方法**

使用Qwen2.5-Omni-7B/72B-Instruct做语音/视频感知与推理基座，Qwen2.5-14B-Instruct进行干预评估；利用OmniAvatar与EmotiVoice生成动画视频；通过Group Relative Policy Optimization（GRPO）进行强化学习。

**📊 数据集**

自生成的500场对局日志（3000个说话回合）用于训练；16名人类玩家的视频与语音数据用于人机混合评测；并引用公开的狼人杀规则文档。

**📈 对比分析**

与ReAct、ReCon、SLA、LSPO等基线在Agent-Agent和人机混合游戏中对比；Agent-Agent中获胜率60%（对照57%），角色识别准确率32.6%（对照27%），人机评测中平均被投票次数0.21，胜率46.8%最高。

**⚠️ 局限性**

在高熵环境下仍表现不足，尤其团队狼人占优势；模型对人类高级策略的适应性有限；多模态数据获取成本高且训练集规模有限，可能限制泛化能力。

---

## 119. Where Physics Meets Privacy: Federated PINNs for Privacy-Preserving Brain Tumor Biomechanical Modeling

**arXiv ID:** 2607.26207 | [PDF](https://arxiv.org/pdf/2607.26207v1)

**作者:** Mahmuda Akter Sristy `[一作]` (United International University), Kazi Irfan Subhan `[通讯]` (United International University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种联邦物理信息神经网络（FedPINN），通过联邦学习在三家模拟医院间协作训练，利用线性弹性物理约束实现脑肿瘤分类和位移场预测。

**💡 创新点**

创新点在于将物理信息损失（Navier–Cauchy 平衡方程）嵌入联邦学习框架，既保证了患者数据隐私，又让模型在无原始数据共享的情况下学习可解释的组织位移和肿瘤分类。

**🔧 技术方法**

使用技术包括：联邦平均（FedAvg）协议、物理信息神经网络（PINN）与自动微分、U‑Net + ResNet34 编码器、交叉熵/Dice 损失的分类头，以及基于线性弹性方程的 PDE 残差正则化。

**📊 数据集**

数据集为公开的 Kaggle Brain Tumor MRI 数据集（包括 glioma、meningioma、pituitary、无肿瘤四类），在三家模拟站点按类分布划分，并在统一测试集上评估。

**📈 对比分析**

通过在同一测试集上对比联邦模型和集中式基线模型，联邦模型整体准确率 91.4%（集中式 90.0%），平均 AUC ≈0.985，pituitary 类准确率从 85.6% 提升至 94.5%，但 glioma 类的精确度下降。

**⚠️ 局限性**

局限性包括：仅进行一次实验未给出方差与显著性检验；使用软损失约束物理一致性，偶尔会偏离精确物理；材料模型仅区分肿瘤与健康两种，未考虑不同肿瘤亚型；缺乏时间序列与 MRE 数据；计算成本较高，对噪声敏感；未实现加密/安全聚合，隐私保障仅为技术层面。

---

## 120. AgentGUI: An Interface for Observing and Steering Long-Running AI Agents

**arXiv ID:** 2607.26300 | [PDF](https://arxiv.org/pdf/2607.26300v1)

**作者:** Xuan Zhao `[一作]` (ETH Zürich), Michael Moor `[通讯]` (ETH Zürich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出并实现了AgentGUI，一款本地托管的可视化界面，用于实时观察、手动和自动调度多任务长时间运行的AI代理。

**💡 创新点**

创新点在于将轨迹可视化、人工干预与自动审计三大功能集成在一个界面，并通过可插拔的自动化管理器实现对代理漂移的即时纠正。

**🔧 技术方法**

技术包括React前端+FastAPI后端+WebSocket实时推送，Hermes/Claude Agent SDK接口，Docker沙箱隔离，LLM驱动的自动审计模型（Qwen3.5）等。

**📊 数据集**

数据集主要使用OrganSMNIST进行CNN训练、MedXpertQA进行问答提示迭代，以及合成的98文件病历用于验证自动审计。

**📈 对比分析**

与Hermes Dashboard对比，用户研究显示平均答题时间缩短38%（p=0.023），准确率提升至93%（p=0.031），自动审计在不同模型规模上将完成率提升10-54个百分点，总调试token占比低于1%。

**⚠️ 局限性**

局限性包括样本量仅8人的同质化用户研究、仅在单代理模型上验证自动审计、未覆盖多代理协作的质性评估。

---

## 121. "Nobody Did This": Contribution, Originality, and Accountability in Agent-Mediated Collaboration

**arXiv ID:** 2607.26387 | [PDF](https://arxiv.org/pdf/2607.26387v1)

**作者:** Kashif Imteyaz `[一作]` (Northeastern University), Saiph Savage `[通讯]` (Northeastern University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出“贡献消解”概念，探讨LLM代理在协作中如何模糊贡献边界，并设计一整天的工作坊（包括映射、设计未来等活动）来聚焦该问题、评估文档化技术的局限，并推动责任基础设施的研究。

**💡 创新点**

创新点在于将贡献消解视为协作知识工作中的结构性变革，强调可观测贡献而非单纯的记录；同时提出“文档陷阱”框架，批判传统的归因与记录方法，并为责任基础设施提供新视角。

**🔧 技术方法**

技术主要是以工作坊设计与实验活动为手段，包括映射活动、案例书写、责任基础设施草图、文档化技术演示（AI使用声明、水印、溯源日志）等；并使用与LLM交互的原始草稿与协作日志作为材料。

**📊 数据集**

使用的“数据集”是由组织者准备的模拟协作项目材料——最终报告、各阶段草稿以及完整的代理交互日志；这些材料用于工作坊中的案例分析与讨论。

**📈 对比分析**

方法是对比不同文档化技术（如AI使用声明、水印、溯源日志）在捕捉归因信息上的有效性；实验结果显示，这些技术往往无法提供可辨别人类与代理贡献的足够信息，无法完全恢复贡献的可观测性。

**⚠️ 局限性**

局限性包括缺乏大规模实证评估、依赖模拟案例而非真实生产环境、工作坊的结果主要为定性洞察，且对责任基础设施的具体实现仍未给出可验证的技术方案。

---

## 122. Dissecting Sensitivity to Training Language in Self-Supervised Speech Learning Using Neural Audio Codec Tokens

**arXiv ID:** 2607.26350 | [PDF](https://arxiv.org/pdf/2607.26350v1)

**作者:** Daigo Takizawa `[一作]` (National Institute of Advanced Industrial Science and Technology), Shinji Watanabe `[通讯]` (Carnegie Mellon University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过系统实验分析了神经音频码器（NAC）与基于离散码序列的自监督学习（SSL）在跨语言下的语言敏感性，评估了NAC训练语言与SSL预训练语言对下游语音识别（ASR）和情感识别（SER）性能的影响。

**💡 创新点**

创新点在于将NAC与SSL预训练阶段解耦，使用系数变异（CoV）定量评估语言敏感性，发现NAC训练语言对性能影响有限，而SSL预训练语言决定跨语言性能，从而证明可复用单一NAC跨多语言。

**🔧 技术方法**

采用多种公开NAC（DAC、EnCodec、SpeechTokenizer等）及HuBERT框架进行codec‑based SSL，利用离散码序列作为输入；下游使用Conformer ASR模型与SUPERB情感识别设置。

**📊 数据集**

NAC训练数据来源于Libri-Light、LaboroTVSpeech、WenetSpeech、AISHELL‑1等；SSL预训练使用LibriSpeech、日语广播电视、中文WenetSpeech；下游任务使用LL‑10h、LTVS‑100h、CSJ、COJADS、WS‑100h、AS1、IEMOCAP、JTES、EmotionTalk等数据集。

**📈 对比分析**

比较方法包括：①使用波形输入与NAC重建波形对比；②不同NAC训练语言对下游性能的影响；③固定NAC变更SSL预训练语言；④固定SSL预训练语言变更NAC训练语言。性能通过WERR/CER和AR评估，并用CoV衡量跨语言差异。结果显示SSL预训练语言对性能高度敏感，NAC训练语言影响较小，CoV值较低。

**⚠️ 局限性**

局限性：仅评估了三种语言（英语、日语、汉语）和两类下游任务（ASR、SER），未涵盖更多语言或复杂任务；实验采用固定的超参数和单一NAC模型，未探索更大规模或多域场景的跨语言适应性。

---

## 123. MoMo: Dial Motion Mode in Robot Manipulation with Spatiotemporal Action Tokenization

**arXiv ID:** 2607.26315 | [PDF](https://arxiv.org/pdf/2607.26315v1)

**作者:** Yuhan Hu `[一作]` (Apple), Arto Kivila `[通讯]` (Apple)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了两阶段的模仿学习框架 MoMo，能够在机器人抓取、倒料、观测等操作中通过一个连续的“运动模式”标量动态调节执行风格（如平稳或动态），同时保持任务完成率。

**💡 创新点**

创新点在于将动作分为空间分支（负责路径几何、接触）和时间分支（负责节奏、平滑度）两条残差向量量化（RVQ）通道，并把运动模式映射到时间通道的连续标量，使得该行为因子可跨任务复用、可插值、可迁移；此外引入 swap 损失、对抗互信息和对比学习以强化空间–时间的解耦。

**🔧 技术方法**

使用技术包括：残差向量量化（RVQ）编码器/解码器、离散余弦变换（DCT）提取时间频域特征、Transformer 作为行为克隆器、对抗 MI 损失、对比损失、swap 损失、离散化动作分词、连续偏移补偿。

**📊 数据集**

数据集：基于 Interbotix X‑Series 双臂实验平台，在桌面上收集了 6 类操作任务（pick, pour, look, avoid, highfive, push）共约 360 条演示，每类约 60 条，演示覆盖两种运动模式（Mode A 慢稳、Mode B 快动），并包含单模式演示的转移任务。

**📈 对比分析**

评估方法：通过关节速度、加速度、末端执行器接近角度等定量指标、Spearman 相关性、拉普拉斯人类评估（风格一致性、任务完成度）来比较不同模式、插值以及跨任务迁移的表现。结果显示：模式标量能够实现单调、可辨别的风格变化；在训练任务中成功率保持在 90%+，在未演示模式的转移任务中仍保持约 80% 以上成功率，且人类评估能正确区分三种模式。

**⚠️ 局限性**

局限性：仅使用两个原型模式且通过一维标量控制，限制了可表达的行为多样性；空间分支仍吸收部分运动特征，导致迁移时速度变化受限；实验场景单一机器人、有限任务，未覆盖更复杂的环境或人机交互；评估指标聚焦于速度/加速度，未充分反映所有感知的运动特征。

---

## 124. Can We Trust AI in 6G? Verifiable and Auditable AI-Driven Trustworthy Wireless Networks

**arXiv ID:** 2607.26409 | [PDF](https://arxiv.org/pdf/2607.26409v1)

**作者:** Genze Jiang `[一作]` (Brunel University London), Kezhi Wang `[通讯]` (Brunel University London)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出机械审计方法与审计原生网络架构，利用三步审计（定位协议相关特征、因果验证、适应诊断）检查AI网络功能内部是否符合3GPP规范，以实现6G无线网络的可验证可信性。

**💡 创新点**

将机制性审计嵌入AI网络功能，创建离线与在线两种验证模式，构建机器可验证的3GPP谓词库，并通过稀疏自编码器、因果干预等技术实现对AI内部表示的定位、因果验证与适应诊断，填补传统输出测试无法覆盖的可信性缺口。

**🔧 技术方法**

采用稀疏自编码器对内部表示进行分解，利用可执行的3GPP谓词库进行特征定位，实施因果干预/消融检测因果使用，使用代表性相似性分析诊断适应，结合O‑RAN架构中的audit xApp 与 rApp，示例中使用Qwen3等大型语言模型的内部表示。

**📊 数据集**

使用自建的大规模城市实测无线资源控制（RRC）信令数据集，数据同步位置与物理层测量，可用于评估谓词与实际网络状态；未引用公开公共数据集。

**📈 对比分析**

对比两名具有相同handover精度的AI代理，审计通过的代理在消除对应协议特征后准确率从63.7%骤降至6%，而随机消除特征时准确率仅略降至49.3%，说明因果验证有效；通过审计的代理被纳入网络，未通过者被拒绝。

**⚠️ 局限性**

限制包括谓词库覆盖不全、验证代理对分布漂移与对抗攻击的鲁棒性不足、运行时审计开销与策略仍需量化、轻量级再认证方法缺失、缺乏标准化的审计报告格式等。

---

## 125. Examining the Efficacy of Graph Neural Network Message-Passing in Regression Contexts

**arXiv ID:** 2607.26404 | [PDF](https://arxiv.org/pdf/2607.26404v1)

**作者:** Keith G. Mills `[一作]` (Louisiana State University), Joong Ho Kim `[通讯]` (Louisiana State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了多种图神经网络（GNN）消息传递机制在图级回归任务中的表现，通过在公开仓库（FlowerFormer、PINAT、AutoBuild、Qua^2SeDiMo）中仅替换 GNN 层，保持其它配置不变，比较不同 MP 机制在多项回归指标（Kendall Tau、MAE、SRCC、NDCG）和硬件成本（推理延迟、显存）上的优劣；

**💡 创新点**

首次将 GNN 性能评测从传统分类迁移到回归，系统化对比卷积、同构、注意力等多种 MP 机制在回归任务中的效果，并指出 Generalized Graph Convolution（GEN）在多任务上最稳健，同时结合硬件成本提供更全面的视角；

**🔧 技术方法**

使用标准 GNN 变体（GCN、GraphSAGE、GIN、GEN、k‑GNN、PNA、GATv2、Graph Transformer）以及相应的回归损失（MSE、pairwise ranking、SRCC、LambdaRank），并采用多种评估指标（KT、MAE、SRCC、NDCG）进行实验；

**📊 数据集**

主要使用 NAS 领域数据集（NAS‑Bench‑101/201/301/ASR/Graph）、网络压缩与图像生成模型（OFA‑MBv3/PNet、ONNX‑IR、PixArt‑α、Hunyuan、SDXL）以及 PTQ 量化数据集（Qua^2SeDiMo），覆盖从小到大、从 DAG 到序列以及高复杂度图的多种场景；

**📈 对比分析**

通过五次随机种子平均，只替换 GNN 层进行 ablation，得到各指标的排名。实验显示 GEN 在大多数基准上取得最优或第二名；k‑GNN、PNA、GT 在部分场景竞争力强；注意力机制并非始终优于卷积；在低样本或大图场景下多头 GATv2 可能更具优势。硬件评测表明 DAGFormer 的精度最高但延迟/显存最高，而 k‑GNN 在保持性能的同时显著降低延迟；

**⚠️ 局限性**

研究仅评估了公开仓库的默认训练配置，未对超参数或模型规模进行全量搜索；极大图（>1k 节点）实验有限；聚焦回归指标，未深入解释性或可解释性；结果受 GPU 资源限制（OOM）影响，部分配置可能无法完成。

---

## 126. Parameterized Fair Resource Allocation under Diversity Constraints

**arXiv ID:** 2607.26485 | [PDF](https://arxiv.org/pdf/2607.26485v1)

**作者:** Keke Huang `[一作]` (Huazhong University of Science and Technology), Xiaokui Xiao `[通讯]` (National University of Singapore)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可参数化的公平资源分配框架，通过不等式厌恶参数软约束实现多组资源分配的公平与效率平衡。

**💡 创新点**

核心创新在于引入可调的不等式厌恶参数，使得在任意公平度量（如Atkinson、Nash等）下均可实现最优公平分配，并在离散资源情况下给出误差上界。

**🔧 技术方法**

主要技术包括贪心分配算法、连续可分解模型、KKT条件分析以及风险厌恶理论的应用。

**📊 数据集**

实验基于三大真实数据集：新加坡HDB公共住房、大学课程分配和招聘推荐。

**📈 对比分析**

与两种基于硬约束的基线方法对比，实验显示该方法在公平性（Atkinson）和效用（Nash）上均优于基线，且对参数和数据规模具有较强鲁棒性。

**⚠️ 局限性**

局限性包括需要手动设定α的初始值，对极小规模或高度不平衡场景下的离散误差上界较宽，可能导致公平性下降。

---

## 127. SimpleWikiSearch: A Clean Offline Wikipedia Environment for Agentic Search

**arXiv ID:** 2607.26070 | [PDF](https://arxiv.org/pdf/2607.26070v1)

**作者:** Guanming Xiong `[一作]` (Peking University), Penghui Zhang `[通讯]` (Peking University)

**通讯引用:** 327 | [OpenAlex ID](https://openalex.org/A5100752918)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了一个完整、可复现的离线维基百科检索环境 SimpleWikiSearch，并在该环境下对多种 LLM 进行基线评估

**💡 创新点**

创新点在于：① 明确规范了 agentic‑search 的实验环境，包括语料构建、检索堆栈、工具协议、交互协议和评估流程；② 提供了完整代码、数据、结果文件和交互统计，消除了传统研究中对检索细节的隐蔽性；③ 通过对比开放源 LLM 与闭源商业模型，展示了检索环境对最终 QA 性能的显著影响

**🔧 技术方法**

主要技术手段：使用官方英文 Wikipedia dump 进行 wikitext 清洗和 token‑aware 章节分块；构建 Tantivy 关键字索引和 FAISS HNSW 向量索引，采用 RRF 混合检索；实现了 OpenAI‑style 三个函数工具（search、open_url、submit_answer）；使用 SGLang 在 A100 GPU 上跑 LLM 推理；在评估时采用 token‑level F1 和 LLM‑judged accuracy 两个指标

**📊 数据集**

评估数据集：2WikiMultiHopQA、HotpotQA、MuSiQue、FRAMES、PopQA、Bamboogle；同时提供了 full‑test 和随机 300‑样本子集，用于对比不同模型

**📈 对比分析**

比较方法：在同一检索环境下对开源 LLM（Qwen3.5‑4B、Qwen3.5‑9B）和闭源商业模型（deepseek‑v4‑pro、gpt‑5.4‑2026‑03‑05）进行评测；使用 token‑level F1 和 LLM‑judged accuracy 两个指标；结果显示：在多跳/实体中心任务上开源 LLM 表现已具竞争力；闭源模型在某些数据集（如 FRAMES、Bamboogle）上取得更高 judge accuracy，但在 lexical F1 上略逊

**⚠️ 局限性**

局限性：① 依赖离线 Wikipedia 版本，无法实时获取最新知识；② 工具接口仅包含 3 种简化操作，缺乏更丰富的工具组合；③ 对于多跳难度较高的数据集，成功提交率仍低于 70%，表明当前检索+推理策略仍不足；④ 评估仅覆盖特定 QA 数据集，未检验更广泛的通用性

---

## 128. GoGoTB: Agentic RTL Verification with Specification-Grounded Coverage Closure

**arXiv ID:** 2607.26181 | [PDF](https://arxiv.org/pdf/2607.26181v1)

**作者:** Xin Xin `[一作]` (Tencent), Yibo Lin `[通讯]` (Beijing Advanced Innovation Center for Integrated Circuits)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个名为GoGoTB的端到端RTL验证框架，自动生成并迭代优化验证环境，最终实现完整覆盖闭合；

**💡 创新点**

创新点包括三层执行控制架构、按需知识注入的可演化知识系统以及基于规范的功能覆盖闭合方法，三者协同实现了无人工干预的验证闭合；

**🔧 技术方法**

采用LLM驱动的Agentic系统、Deterministic Enforcement、Skill Library与Reference Library知识调度、覆盖模型绑定规范行为、Stochastic Simulation与根因诊断循环；

**📊 数据集**

使用了8个开源RTL设计（AES、ALU、UART、SPI、I2C、SHA256、SM4、SDRAM）作为验证基准；

**📈 对比分析**

与UVM^2基线对比，GoGoTB在所有设计上实现了100%环境生成成功率（SRG），功能覆盖平均达83.2%，而UVM^2仅0% SRG且无功能覆盖；

**⚠️ 局限性**

主要限制在于LLM对复杂多步协议的定向测试生成能力不足，导致Stimulus Gap和Sequence Gap仍占残差覆盖的主要比例。

---

## 129. Validating ETCS Data with the B Mathematical Language: An Industrial Pipeline and a Blueprint for LLM Integration

**arXiv ID:** 2607.26111 | [PDF](https://arxiv.org/pdf/2607.26111v1)

**作者:** Lecomte Thierry `[一作]` (CLEARSY), Germain Vincent `[通讯]` (CLEARSY)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将OP SIMU模拟器数据库与CLEARSY Data Solver连接，使用B数学语言规则对ETCS轨道侧数据进行验证，并在此过程中构建LLM（Claude）辅助的规则生成与审查管道。

**💡 创新点**

提出LLM作为“围栏助手”而非替代正式方法：所有LLM产生的规则、解析器和场景都需经过B数学语言验证器和系统化人力审查，形成工业级验证流程与LLM协作蓝图。

**🔧 技术方法**

技术栈包括CLEARSY Data Solver（Caval + ProB）、OP SIMU SQL数据库、Python CLI桥接、FastMCP协议、Claude LLM、B数学语言、Lark解析器与JSON‑Schema。

**📊 数据集**

使用公开的UNISIG Subsets（026、036、044、085）规则集和CLEARSY内部的OP SIMU示例数据库（三份生产数据库与一份种子KO数据库）进行实验。

**📈 对比分析**

在控制数据集上完成42条规则的验证（约5 min，总计约7–11 s/规则），在生产数据库上执行3条电报内容规则（约8 s/规则）。验证结果显示OK/KO/NONE分布与预期一致，性能足以支持交互式工程师使用。

**⚠️ 局限性**

局限性：仅覆盖可从JSON模型推导的43条规则，剩余104条（需位码解析、OBU仿真等）不在范畴；LLM生成错误仍需人工审查；未评估情景意图符合度；LLM部署与安全性（托管或本地）仍需进一步研究。

---

## 130. Every string has probabilistic automatic complexity at most three

**arXiv ID:** 2607.26275 | [PDF](https://arxiv.org/pdf/2607.26275v1)

**作者:** Bjørn Kjos-Hanssen `[一作]` `[通讯]` (University of Hawaii at Manoa), Bjørn Kjos-Hanssen (University of Hawaii at Manoa)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

证明任意有限字母表上任意非空字符串的概率自动复杂度不超过3，并给出显式的三状态PFA构造

**💡 创新点**

用仿射映射与三角形的仿射自映射构造，使得PFA可以唯一最大化给定字符串的接受概率，解决了Gill提出的最大值是否无界的问题

**🔧 技术方法**

仿射自映射、三角形 barycentric 坐标、基数展开动态、线性代数（矩阵乘法）以及 Lean 形式化验证

**📊 数据集**

对所有二进制字符串长度≤8进行穷举验证，对随机选取的字符串长度≤14进行数值检验

**📈 对比分析**

与Gill等人先前的经验结论比较，证明最大值为3，且对二进制字符串可计算；实验结果显示构造的PFA在所有测试字符串上唯一最大化

**⚠️ 局限性**

对于字母表大小≥3，仍未完成 (w)=2 字符串的完整分类，因而对该情况的可计算性仍是未解；构造给出的下界很小（gap ~ b^{-2n})，不适用于更精细的复杂度度量

---

## 131. Round Trip Time: A Benign Signal or an Indirect Window into Datacenter Workloads?

**arXiv ID:** 2607.26239 | [PDF](https://arxiv.org/pdf/2607.26239v1)

**作者:** Sourya Saha `[一作]` (City University of New York), Saptarshi Debroy `[通讯]` (City University of New York)

**通讯引用:** 720 | [OpenAlex ID](https://openalex.org/A5015917097)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过仅利用RTT测量，构造了叶-脊架构中共享路径的网络侧信道，实现了对多租户工作负载类型的推断。

**💡 创新点**

创新点在于提出了路径不变的时域统计特征、基于PCA与CORAL的域适配、以及窗口级后验累积的增量推断机制。

**🔧 技术方法**

使用了RTT归一化、幅度/时域/频域/长程相关等多模特征、PCA降维、CORAL协方差对齐、随机森林、LightGBM、LSTM、Window Transformer等机器学习模型。

**📊 数据集**

实验数据来自FABRIC测试平台，包含15类典型云端工作负载（通信、客户端-服务器、宽带），每类多次运行，形成了带标签的RTT样本。

**📈 对比分析**

在跨路径评估下，所有模型均达73.7%运行级别准确率，合并低信号类别后最高可达97.3%；序列模型优于单窗口模型，推断延迟约87 ms。

**⚠️ 局限性**

局限性包括需要共享路径、低负载导致RTT信号低于噪声阈值、仅在单路径STP环境实验、未考察ECMP或多路径路由、对工作负载强度变化的泛化有限。

---

## 132. Impossible to hide secret ...: Uncovering Security and Privacy Issues in LLM-native IDEs

**arXiv ID:** 2607.26390 | [PDF](https://arxiv.org/pdf/2607.26390v1)

**作者:** Mostafijur Rahman Akhond `[一作]` (York University), Song Wang `[通讯]` (York University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析1.1M条 Reddit 讨论，系统识别并归纳 LLM‑原生 IDE（LIDEs）中的安全与隐私问题，构建了32类问题的分类体系，并梳理了13种开发者采用的缓解策略；

**💡 创新点**

创新点在于首次大规模、基于真实用户讨论的 LIDEs 安全隐私研究，发现问题主要源于系统设计而非 LLM 本身，提出针对性改进建议；

**🔧 技术方法**

采用混合方法：LLM 二分类器筛选安全隐私帖子、人工编码构建分类体系、LLM 辅助提取评论中的缓解建议，并使用 Cohen κ 评估标注一致性；

**📊 数据集**

使用数据集为 1.1M 条来自 29 个 LIDEs 相关子版块的 Reddit 帖子，最终提炼出 446 条相关帖子及 6,280 条评论；

**📈 对比分析**

通过对提炼出的问题与缓解策略进行统计与多标签分析，评估各类问题的流行程度及开发者应对频率；不涉及模型性能对比；

**⚠️ 局限性**

局限性包括：仅基于自述式 Reddit 讨论，无法验证所有漏洞；可能存在筛选误差和多标签冲突；外部有效性受限于公开讨论的偏差，且时间上随社区活跃度变化。

---

## 133. TraceCoder: Explainable and Auditable Code Generation with Position-Key Snippet Versioning

**arXiv ID:** 2607.26307 | [PDF](https://arxiv.org/pdf/2607.26307v1)

**作者:** Rwaida Alssadi `[一作]` (Florida Institute of Technology), Marius Silaghi `[通讯]` (Florida Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 TraceCoder，一个能够在 LLM 代码生成过程中记录每一次修复事件、保持代码片段级历史的系统；

**💡 创新点**

通过位置键版本化（Fractional Indexing Scheme）与递归修复事件表实现代码片段可追溯、可审计；同时提供基于浏览器的热图可视化查看历史；

**🔧 技术方法**

使用 SQLite 关系数据库存储代码片段与历史，Fractional Indexing Scheme（FIS）实现无冲突插入，调用 Gemini、Grok、DeepSeek 等多 LLM API，使用 diff/patch 记录变更，HTML/JS 生成交互式查看器；

**📊 数据集**

对 30 个单文件 Python 算法任务（字符串处理、数学计算、数据结构等）进行实验，使用 LLM 生成的基准测试（benchmarks）驱动迭代修复；

**📈 对比分析**

与 Gemini 单一模型对比：平均迭代 5.45 次、修复比例 20.7%、数据库约 46.8 KB；与 Grok+DeepSeek 组合对比：平均迭代 2.07 次、修复比例 30.0%、数据库约 54.2 KB，整体执行时间约 600 s；

**⚠️ 局限性**

局限在于仅支持单文件 Python、依赖 LLM JSON 输出的可靠性、API 限流影响、基准生成质量不一、对多文件或编译语言的适用性未知。

---

## 134. Do Unified Multimodal Models Think in One Space? A Lens Through Cross-Branch Steering

**arXiv ID:** 2607.26411 | [PDF](https://arxiv.org/pdf/2607.26411v1)

**作者:** Yu Wang `[一作]` (University of Wisconsin-Madison), Sharon Li `[通讯]` (University of Wisconsin-Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并验证了跨分支语义导向（cross‑branch semantic steering）框架，用来探究统一多模态模型（UMM）中理解分支与生成分支是否共享可迁移的语义空间，并通过对“理解→生成”和“生成→理解”两种方向的向量迁移实验揭示了语义对齐的差异。

**💡 创新点**

创新点在于：①首次把steering向量从理解分支迁移到生成分支，作为诊断工具探测语义统一；②构建了针对51对概念（颜色、文本、计数、位置、风格、外观、人物）的对比数据集UMMSteer；③揭示了理解向量可有效控制生成，而生成向量却无法影响理解，归因于表征层级不匹配，并给出对模型架构（AR+diffusion）设计的启示。

**🔧 技术方法**

技术上采用了对比学习式steering向量提取方法（CAA、RepE、ITI）从理解分支获得语义方向；在生成分支的VAE潜在层注入这些向量；使用Qwen‑VL‑3 30B 作为自动评估器计算Steering Success Rate（SSR）；利用PCA Subspace Projection（PSP）度量向量与目标分支语义子空间的对齐程度。

**📊 数据集**

使用了自建的UMMSteer数据集（51个概念对，覆盖颜色、文本、计数、位置、风格、外观、人物七类），并在BAGEL、Janus‑Pro、UniPic‑1、Show‑o2等公开模型上进行实验。

**📈 对比分析**

实验通过SSR评估迁移效果。理解→生成方向在BAGEL上，CAA方法的SSR最高可达96%（颜色）及高于80%（其他概念）；其他方法也能实现70–80%级别成功率；相反，生成→理解方向SSR接近0。对不同架构的对比显示，纯AR模型失效，而Hybrid AR+diffusion（Show‑o2）能成功迁移，说明性能与模型架构密切相关。

**⚠️ 局限性**

局限性包括：①跨分支迁移仅在AR+diffusion模型上表现良好，纯AR模型无法实现；②生成分支提取的steering向量主要捕获低层外观特征，难以跨越到理解分支；③实验范围受限于51对概念，未覆盖更广泛的语义或更大规模模型；④评估依赖自动问答评测器，尽管与人工评测高度一致，但仍可能存在偏差。

---

## 135. Two-sided RDMA Striking Back for Disaggregated Memory Databaases

**arXiv ID:** 2607.26227 | [PDF](https://arxiv.org/pdf/2607.26227v1)

**作者:** Hokeun Cha `[一作]` (University of Wisconsin-Madison), Xiangyao Yu `[通讯]` (University of Wisconsin-Madison)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于两端 RDMA 的分离内存 OLTP 系统 Lotus，支持 B+树索引和两阶段锁，并通过记录级缓存和事务内/事务间批处理降低网络往返次数。

**💡 创新点**

证明两端 RDMA 在分离内存环境中能弥补 CPU 受限问题，提出轻量级记录级缓存、优先级调度、抢占以及自适应批处理等创新机制，实现无饥饿并显著提升吞吐和尾延迟。

**🔧 技术方法**

使用两端 RDMA（SEND/RECV）和一端 RDMA（READ/WRITE/ATOMIC）对比，B+树索引，2PL 并发控制，记录级缓存，事务内/事务间批处理，RC 与 UD 传输实验，以及自适应领导选举与批处理超时机制。

**📊 数据集**

采用 Yahoo! Cloud Serving Benchmark (YCSB) 100 GB 数据集（100 M 1 KB 记录，Zipf θ=0.9）并构造三种工作负载（W0：50%读/50%写，W1：95%读/5%写，W2：只读）。

**📈 对比分析**

在 CloudLab 四台机器（AMD EPYC 7402P、128 GB DRAM、100 Gbps Mellanox ConnectX‑5 NIC）上与 Sherman/Deft（和 DEX）对比，测量吞吐、p999 延迟、网络流量等；Lotus 在所有负载下实现 8.2× 吞吐提升、42.9× p999 延迟降低，并在事务长度、缓存大小、线程数等敏感性实验中保持优异表现。

**⚠️ 局限性**

未实现容错/日志恢复；实验仅针对 RC 传输，未覆盖 UD 的大消息处理；系统依赖 RDMA NIC；高写冲突下批处理可能增加尾部延迟；对 CXL 等新型远程内存技术的适配未探讨。

---

## 136. Reproducibility in Recommender Systems: A Survey

**arXiv ID:** 2607.26074 | [PDF](https://arxiv.org/pdf/2607.26074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 137. Embodied Agents Take Control: Minimal-Interface Zero-Shot Agents Rival Industrial-Scale Policies in Vision-and-Language Navigation

**arXiv ID:** 2607.26148 | [PDF](https://arxiv.org/pdf/2607.26148v1)

**作者:** Jian Zhou `[一作]` (Australian Institute for Machine Learning, Adelaide University), Qi Wu `[通讯]` (Australian Institute for Machine Learning, Adelaide University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种“agentic embodied control”，即让通用大语言/视觉模型在仅提供单目RGB和四个离散原语动作的最小接口下自行完成从观察、决策到执行和终止的完整导航循环，并评估其在R2R-CE短程和RxR-CE长程任务上的性能。

**💡 创新点**

创新点在于：①将导航任务从传统的“policy”或“workflow”框架转变为“agentic”框架，让模型自己掌握高层交互循环；②在极简接口下证明了通用模型可实现高达78%成功率；③通过可选的waypoint工具展示了把训练模块作为可选能力而非强制控制路径可显著提升效率和准确性。

**🔧 技术方法**

技术包括：通用视觉-语言模型（如 Claude、Qwen、Codex 等）、软件工程层面的 agent harness（mini-swe-agent、Claude Agent SDK、Codex CLI）、仅含RGB感知与四个原语动作的最小接口、以及可选的深度预测 waypoint 工具；实验通过多次跑、不同模型/工具/努力级别交叉验证。

**📊 数据集**

使用的主要数据集是 R2R-CE（固定 subset，100 轨迹）作为短程基准，RxR-CE（更长、蜿蜒轨迹）作为长程挑战；并在真实 Unitree Go2 机器人上进行物理部署验证。

**📈 对比分析**

比较方法：与现有训练式策略（NaVid、NaVILA、StreamVLN 等）、零射流程（NavGPT、MapGPT、AgenticNav 等）以及工业级双脑系统（ABot-N1、InternVLA-N1 等）在同一测试集上对比 SR/SPL。实验结果显示，最小接口下的 agentic 系统（Claude+默认 effort）可达 68.3% SR，最高 78% SR；在与强训练式策略相比时，表现已逼近甚至优于部分工业级系统；可选 waypoint 接口可将 SR 提升至 76.7% 并将步骤/时间显著降低。

**⚠️ 局限性**

limitations：①在更长距离任务（RxR-CE）成功率显著下降（仅 26–39%）；②极简接口导致上下文随时间无限增长，导致推理延迟和 token 过载；③缺乏持久地图、姿态估计和碰撞反馈，导致机器人在真实硬件上出现行进误差和物理碰撞；④模型对任务细节（如指代消解、几何判断）的自我诊断能力不足，导致错误的自我确认。

---

## 138. Data Fusion and Contrastive Alignment for Unconstrained IR Molecular Structure Elucidation

**arXiv ID:** 2607.26164 | [PDF](https://arxiv.org/pdf/2607.26164v1)

**作者:** Ethan J. Mick `[一作]` (University of Missouri), Derek T. Anderson `[通讯]` (University of Missouri)

**通讯引用:** 3092 | [OpenAlex ID](https://openalex.org/A5076164452)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种无公式输入的Transformer架构，直接从IR光谱预测SMILES，实现分子结构的自动解析。

**💡 创新点**

创新点包括：在解码器中引入混合专家（MoE）并使用非加性聚合（LOSN与Choquet Integral）提升表示能力；在编码器层融合与解码器聚合中应用模糊聚合；通过对比学习实现IR与结构嵌入的对齐；以及基于子结构的误差分析。

**🔧 技术方法**

采用的技术有：Transformer编码器‑解码器、Mixture‑of‑Experts、Linear Order Statistic Neuron (LOSN)、Choquet Integral MLP (ChIMP)、CoCa式对比损失、Beam search生成、SMILES枚举增强等。

**📊 数据集**

使用的数据集包括：IBM‑MD模拟IR与结构数据、QM9S量子化学模拟数据、NIST实验IR与结构数据；SMILES词表为47个符号。

**📈 对比分析**

与现有IR‑only基线（如Alberts、Wu等）在Top‑K准确率上对比；在NIST实验集无公式输入时取得Top‑1 31.8%，Top‑5 50.1%，Top‑10 57.2%；相较于传统Dense FFN和MoE‑LCS，MoE‑LOSN在Top‑1上提升约+7个百分点，且对比损失与层融合进一步提升性能。

**⚠️ 局限性**

局限性在于：仍低于公式约束下的等价物排名模型；对功能团识别存在错误，尤其是重叠或弱吸收峰；对更大、更复杂分子结构的辨识尚不足；受限于模拟与实验数据差异，模型泛化能力需进一步验证。

---

## 139. Revisiting the Algebraic Foundation of Relational Data

**arXiv ID:** 2607.26356 | [PDF](https://arxiv.org/pdf/2607.26356v1)

**作者:** Yisu Remy Wang `[一作]` (University of California, Los Angeles), Paul Talma `[通讯]` (University of California, Los Angeles)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现基于Tarski关系代数（TAR）的新型查询语言Prela，旨在为现代数据库提供统一的抽象层；

**💡 创新点**

创新点在于将TAR直接用作查询语法和执行层的核心，将二元关系映射到列式存储，并通过嵌入式Rust实现可组合、可控的查询；

**🔧 技术方法**

采用TAR操作符、Rust的连续调用式(CPS)实现、列式向量存储、函数式编译技术以及手工调优的查询计划；

**📊 数据集**

在Join Order Benchmark（JOB）和TPC‑H两个主流基准上进行实验；

**📈 对比分析**

对比方法是将Prela单线程实现与DuckDB单线程执行相同查询进行基准测试，结果显示Prela在JOB上约4.5×、在TPC‑H上1.1×至2.9×的性能提升；

**⚠️ 局限性**

主要局限在于缺乏完整的查询优化器、并行执行能力有限以及编译开销较高。

---

## 140. A Picture Says Thousands of Words - Harnessing Dermal Exposure Data from Images through Hybrid Deep Learning for Enhanced Safety Assessment

**arXiv ID:** 2607.26170 | [PDF](https://arxiv.org/pdf/2607.26170v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 141. Q-Steer: Action-Value Guidance for Molecular Policy Optimization

**arXiv ID:** 2607.26391 | [PDF](https://arxiv.org/pdf/2607.26391v1)

**作者:** Xinyu Wang `[一作]` (University of Connecticut), Minghu Song `[通讯]` (Hefei Comprehensive National Science Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在分子生成任务中通过在生成过程的每一步使用冻结的前缀动作价值模型来调整采样对数的“Q-Steer”方法；

**💡 创新点**

创新点在于把动作级未来价值估计作为一种无须修改优化器更新规则、且不占用在线oracle调用次数的、可复用的生成时刻采样偏置；

**🔧 技术方法**

技术上采用离线训练的前缀动作价值模型Q_ϕ、对采样对数加上归一化的价值奖励增量、β调节强度，以及在LSTM和GPT‑2分子语言模型上实现的采样时间控制；

**📊 数据集**

使用了PMO23基准数据集（包括相似性、MPO、重发现、骨架、异构体、SMARTS、QED、QSAR等8项任务）；

**📈 对比分析**

通过在10,000次在线oracle调用预算下，对4种优化器（PPO、REINVENT、AHC、hill‑climbing）和2种backbone（LSTM、GPT‑2）进行完整因子化实验，结果显示所有8个backbone–optimizer组合均获得正向平均有效-唯一分数提升，宏观平均提升区间+0.034至+0.049，且每个组合18–20个任务表现优于基线，但唯一性下降、上尾指标（Top‑10、Best‑Score）提升不一；

**⚠️ 局限性**

局限性包括：需要额外的离线模型训练和在线推理成本；对分子多样性产生负面影响；上尾发现能力不稳定；仅在LSTM和GPT‑2上验证，未证明对更大规模或不同架构的泛化；

---

## 142. A literature review of recent advances in software design and architecture

**arXiv ID:** 2607.26110 | [PDF](https://arxiv.org/pdf/2607.26110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 143. SARC-DQ: Runtime Data-Quality Gating for Agentic AI: Silent Evidence Defects, the Incompetence Shield, and Downstream-Only Remediation

**arXiv ID:** 2607.26313 | [PDF](https://arxiv.org/pdf/2607.26313v1)

**作者:** Gaston Besanson `[一作]` `[通讯]` (Universidad Torcuato Di Tella), Gaston Besanson (Universidad Torcuato Di Tella)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了元数据驱动的缺陷在代理系统中导致的经济损失，并提出在行动前对元数据进行质量检查的门控机制。

**💡 创新点**

创新点在于将证据完整性视为与模型能力并列的系统维度，证明仅提升模型规模无法检出元数据缺陷，且通过在执行点插入轻量级元数据门控即可显著降低损失。

**🔧 技术方法**

使用了基于SARC-DQ的多门控架构、数据质量谓词、内容地址化元数据、决策几何的模型无关解析预言器，以及对不同规模LLM（haiku→fable）的推理成本控制。

**📊 数据集**

采用了GIGO-Bench仿真数据，覆盖八类数据质量腐败（如陈旧、版本冲突、单位不一致等），在单周期新闻订货任务中注入2%、5%、10%、20%的错误率。

**📈 对比分析**

将门控（D）与无门控、提示式、payload-only 评审、oracle及上游清洗等六种策略对比。门控在覆盖的元数据通道（新鲜度、模式一致性）上检测率提升约2倍，ADR下降从约0.18降至0.07，单次损失平均下降≈20%（置信区间零交叉），但未能覆盖所有缺陷导致整体恢复率仅约0.48。

**⚠️ 局限性**

局限性包括：①元数据缺陷需已被标注并可访问，门控无法补救缺失元数据或无谓词的缺陷；②实验仅在单一决策任务和一组LLM模型上验证，泛化性待验证；③对元数据质量的假设（如正确性、完整性）可能在真实系统中不成立；④门控逻辑的谓词覆盖仍有限，需进一步扩展。

---

## 144. Explicit Note-Event Tokenization and Pitch-Validity Constrained Decoding for MIDI-to-Tablature Transcription

**arXiv ID:** 2607.26440 | [PDF](https://arxiv.org/pdf/2607.26440v1)

**作者:** Ting-Kai Hsu `[一作]` (National Taiwan University), Yu-Hua Chen `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种从MIDI事件到吉他tablature的序列到序列框架，采用显式的note‑event分词与基于音高有效性的约束解码；

**💡 创新点**

创新点包括在解码器输出中加入NOTE_ON/OFF、REST等事件令符，使音符边界和音高信息显式化，并在解码时使用音高有效性掩蔽限制TAB候选，从而避免生成错误音高；

**🔧 技术方法**

使用T5编码解码器进行端到端训练，采用权重衰减、Dropout等正则化，利用教师强迫交叉熵损失，并在解码时通过logit掩蔽实现音高有效性约束；

**📊 数据集**

在大型DadaGP数据集和小型François Leduc数据集上进行实验；

**📈 对比分析**

与重新训练的Fretting Transformer进行对比，结果显示在DadaGP上Token、Pitch、Tab准确率均提升；在Leduc上即使单独训练亦能达到约98% Tab准确率，约束解码进一步提升准确率；

**⚠️ 局限性**

主要限制是仍难以消除同一音高对应多种字符串‑品位选择的歧义，未充分考虑手指位置、可演奏性约束，导致在精确位置选择上仍存在误差。

---

## 145. FloDR: An invertible dimensionality reduction method based on a normalising flow

**arXiv ID:** 2607.26278 | [PDF](https://arxiv.org/pdf/2607.26278v1)

**作者:** Abdallah Baraka `[一作]` (Wageningen University & Research), Daniel Probst `[通讯]` (Wageningen University & Research)

**通讯引用:** 3245 | [OpenAlex ID](https://openalex.org/A5046410250)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种基于可逆正态流的二维嵌入方法FloDR，能够在保持局部与全局结构的同时保留残差信息并提供可逆映射与密度。

**💡 创新点**

通过在可逆流中将前两维用于可视化、剩余维度作为残差，并利用可逆性构造可检验的诊断场（条件扩散与隐藏对比），提供可证实的嵌入质量评估。

**🔧 技术方法**

可逆正态流（Affine Coupling/RealNVP）、邻居嵌入目标（Umap/ordinal loss）、密度估计与判别器、bootstrap置信度、校准的置换检验。

**📊 数据集**

MNIST、Fashion‑MNIST、paul15 myeloid progenitor、Schneider reaction dataset、三种单细胞图谱（人类胎儿骨髓、发育小鼠小脑、胚芽Arabidopsis种子）。

**📈 对比分析**

与t‑SNE、UMAP、openTSNE、PHATE、TriMap、PaCMAP、LocalMAP等基线在局部邻域召回(k=15)与全局Spearman相关(CPD)进行比较，FloDR在保持局部召回的同时显著提升CPD，在大多数数据集上接近最优或更好。

**⚠️ 局限性**

仍在CPU上表现不佳，局部邻域精度在单前向推断时较差，需要额外优化步骤；重建质量未被证明，诊断依赖于残差/标签后验估计，参数选择与随机性导致可重复性问题。

---

## 146. The Fabric Is the Cluster Driver: Cross-Layer eBPF Policies for GPU-CXL Fabrics

**arXiv ID:** 2607.26335 | [PDF](https://arxiv.org/pdf/2607.26335v1)

**作者:** Yiwei Yang `[一作]` (University of California Santa Cruz), Andi Quinn `[通讯]` (University of California Santa Cruz)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

实现了一套基于eBPF的中间件编译器和运行时，能够在GPU、DPU/NIC和CXL fabric等多设备间统一执行跨层的资源调度与数据搬迁策略。

**💡 创新点**

创新点在于提出了语义移动图（semantic movement graph）抽象，能够将跨设备的数据流与变换统一描述，并通过eBPF将其动态编译为可在各设备上直接执行的策略程序，实现真正的 fabric‑level 集群驱动。

**🔧 技术方法**

技术上结合了eBPF编译器、可验证的eBPF程序、跨设备一致性映射（如一致性类BPF map）、Near‑Type‑2小核心的硬件JIT以及复制路径优化器，支持Move、Quantize、Compress等多种数据变换。

**📊 数据集**

使用了1,000条生成的微基准移动测试，以及14个LLM相关工作负载（Qwen 27B解码、长上下文CXL KV缓存、管线/张量并行、LoRA、Qwen 2→32GPU扩展、MoE all‑to‑all、prefill/decode共定位、远程KV缓存、训练/检查点流量）。

**📈 对比分析**

与传统的主机/网络驱动流水线相比，在所有测试工作负载中获得了2.34×的端到端数据移动加速，使用复制路径优化后提升至3.47×，验证了 fabric 驱动模型的显著优势。

**⚠️ 局限性**

局限性包括：仅在实验平台（BlueField DPU + GPU + CXL）上验证，未覆盖所有硬件/软件栈；eBPF编译及验证开销仍需进一步压缩；对极大规模模型的扩展性与多租户隔离机制仍待完善。

---

## 147. Eddeep: a deep-learning framework for fast eddy-current distortion correction in diffusion MRI

**arXiv ID:** 2607.26292 | [PDF](https://arxiv.org/pdf/2607.26292v1)

**作者:** Antoine Legouhy `[一作]` (University College London), Hui Zhang `[通讯]` (University College London)

**通讯引用:** 23070 | [OpenAlex ID](https://openalex.org/A5100323374)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `70e40602-aae3-44bd-80ec-4a7f2674330f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一种名为 Eddeep 的两阶段深度学习框架，用于快速校正扩散 MRI 中的 eddy‑current 失真和头部运动；

**💡 创新点**

创新点在于先通过图像翻译网络标准化不同 b‑value 与梯度方向的对比度差异，再使用物理约束的全局参数回归网络一次性完成失真与运动校正，避免传统的迭代预测‑校正流程；

**🔧 技术方法**

采用 3D U‑Net 进行对比度翻译，CNN‑MLP 结构回归 uni‑directional quadratic 失真模型与刚性运动参数，使用 Jacobian 强度调制与无监督训练技术；

**📊 数据集**

使用 UK Biobank 数据集进行训练、验证与测试，并在未见过的 Memodyn 数据集上进行跨域评估；

**📈 对比分析**

与 FSL Eddy 在多种指标（between‑volume jitter、FA halo、DKI 残差、信号不规则性、NMI）以及计算时间进行对比，Eddeep 在大多数指标上与 FSL Eddy 相当或略优，但推理时间仅为 2 min，显著快于 FSL Eddy 的 1 h 处理；

**⚠️ 局限性**

主要局限包括：需要先用传统工具生成训练标签；不支持 FSL Eddy 的内体运动与 outlier replacement；二次失真模型可能与真实物理不完全匹配；缺乏真实 ground‑truth 评估；对极端梯度采样方案的鲁棒性尚未验证。

---

## 148. Even More Deception: Objective Misalignment in Mixed-Motive LLM Multi-Agent Systems

**arXiv ID:** 2607.26120 | [PDF](https://arxiv.org/pdf/2607.26120v1)

**作者:** Marylou Fauchard `[一作]` (Université de Montréal), Golnoosh Farnadi `[通讯]` (Université de Montréal)

**通讯引用:** 862 | [OpenAlex ID](https://openalex.org/A5053667504)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了LLM驱动的多代理系统在混合动机环境（Werewolf社交推理游戏）下的目标不一致问题，评估其对决策、推理与游戏结果的影响，并通过内部推理与公共交流的对比分析揭示隐藏目标的策略变化。

**💡 创新点**

①提出三种目标不一致框架（善意、自利、恶意）；②在社交推理游戏中同时评估内部推理与公共交流，发现目标不一致导致显著策略差异但难以从公开沟通中识别；③阐明信息与角色优势会放大目标不一致的负面影响。

**🔧 技术方法**

使用多种LLM（Qwen、Gemma、Llama、GPT‑5 mini）与embedding技术（Qwen3‑Embedding‑8B），利用t‑SNE、谱聚类、Wilson置信区间和 Fisher 检验等统计方法分析推理与交流的潜在空间。

**📊 数据集**

采用Werewolf Arena模拟生成的游戏数据，针对每个模型、角色和目标设置进行30局游戏，总计数千局实验。

**📈 对比分析**

通过比较不同目标下村庄获胜率、角色行为比例以及t‑SNE/聚类可视化，发现自利与恶意目标显著降低村庄获胜率，尤其是信息优势角色（如预言家）受影响最大，实验结果与传统协作场景相比更能揭示目标不一致的危害。

**⚠️ 局限性**

仅测试单个失效代理，未考虑多代理协作或联盟；样本量有限导致置信区间较宽；缺乏对交流内容细粒度的定性分析，未探索更复杂的攻击或防御机制。

---

## 149. The Easy Trap: Why LLMs Underestimate Misconception-Driven Difficulty

**arXiv ID:** 2607.26067 | [PDF](https://arxiv.org/pdf/2607.26067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 150. Embedding Items at Scale: Comparing GNN-Based and ID-Based Item Embeddings in the Yandex Ecosystem

**arXiv ID:** 2607.26365 | [PDF](https://arxiv.org/pdf/2607.26365v1)

**作者:** Sergei Makeev `[一作]` (Yandex), Kirill Khrylchenko `[通讯]` (Yandex)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在Yandex生态系统中对比了基于图神经网络预训练的物品嵌入与直接在Transformer中端到端训练的ID嵌入，评估其在大规模和低资源环境下的效果与成本。

**💡 创新点**

创新点在于首次在工业级推荐系统中系统比较两种嵌入策略，并公开了低资源场景的数据与代码，证明预训练在数据稀缺时能显著提升性能。

**🔧 技术方法**

主要技术包括TwHIN、MultiBiSage GNN预训练、multihash ID嵌入、两塔Transformer架构、CatBoost排名器、采样负样本以及Wilcoxon检验。

**📊 数据集**

实验数据来自Yandex Market、Yandex Music（数十亿交互）以及公开的Yandex Lavka低资源数据（约15万条交互）。

**📈 对比分析**

通过离线nDCG、准确率、加权准确率等指标进行比较，发现大规模场景下ID嵌入优于预训练GNN，而低资源场景下预训练GNN能获得更高的nDCG和准确率。

**⚠️ 局限性**

局限性包括预训练GNN在大规模场景下成本高且收益有限、实验仅覆盖Yandex内部系统缺乏跨平台验证、以及对稀疏物品的处理仍不充分。

---

## 151. Interpretable Image-Level Acne Severity Grading via EfficientNet-B0 Transfer Learning and Grad-CAM

**arXiv ID:** 2607.26461 | [PDF](https://arxiv.org/pdf/2607.26461v1)

**作者:** Sophie Zeng `[一作]` (Dr Robot Inc), Haipeng Xie `[通讯]` (Dr Robot Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发并评估了一个可解释的图像级痤疮严重度分类器，并发布跨平台（Python+MATLAB）实现；

**💡 创新点**

创新点在于将迁移学习与Grad‑CAM可解释性结合，提供双框架实现和临床友好接口，解决可重现性与后端适配问题；

**🔧 技术方法**

采用EfficientNet‑B0迁移学习、AdamW优化、数据增强、Grad‑CAM可解释、分层后端回退等技术；

**📊 数据集**

使用公开的Acne‑LDL基准数据集（2,983张图像，四级Hayashi分级）；

**📈 对比分析**

通过内部70%/15%/15%拆分评估，测试集准确率93.5%、宏F1 94.4%，Cohen κ 0.956，显示“几乎完美”的性能；

**⚠️ 局限性**

局限性包括仅内部拆分，无跨设备或外部验证，单一标注者，皮肤色彩覆盖不足，缺乏细粒度检测与可解释性量化等。

---

## 152. Pragmatic Reasoning in Design

**arXiv ID:** 2607.26322 | [PDF](https://arxiv.org/pdf/2607.26322v1)

**作者:** Lance Ying `[一作]` (Harvard University), Samuel J. Gershman `[通讯]` (Harvard University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证一种基于合作博弈与递归心智化的设计框架，解释设计者如何通过布置关键物件来向用户传递隐藏的因果结构；

**💡 创新点**

将设计过程建模为设计者与用户之间的合作博弈，利用Rational Speech Act思路将设计决策视为信息信号，并通过递归心智化推断用户对隐藏结构的理解，从而量化设计者在效率与信息性的权衡；

**🔧 技术方法**

采用RSA式的递归贝叶斯推理，构建了字面与冗余（pragmatic）设计者/用户模型，并使用softmax策略与KL散度衡量信息传递；

**📊 数据集**

在30个基于网格世界的“房间设计游戏”场景上进行实验，数据来源于10个地图、每张地图3种钥匙-门映射的人工设计与用户判断；

**📈 对比分析**

通过与人类参与者的设计和评估实验比较，pragmatic设计者模型与用户模型分别在相关系数上达到0.954和0.949，明显优于字面模型的0.316与0.906，显示出模型与人类判断高度一致；

**⚠️ 局限性**

局限在于实验环境过于简化（仅网格世界、视觉相同钥匙、单一隐藏结构），未考虑用户先验多样性、学习迭代、物理约束和更丰富的设计信号。

---

## 153. Early Verdicts, Better Budgets: Sequential Adaptive Rollout Allocation for Compute-Efficient RLVR

**arXiv ID:** 2607.26253 | [PDF](https://arxiv.org/pdf/2607.26253v1)

**作者:** Pixel Nomand `[一作]` (University of Wisconsin--Madison), Sofia Reyes `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种顺序自适应抽样算法（Sequential Adaptive Rollout Allocation），在强化学习可验证奖励（RLVR）中通过在生成抽样时提前决定是否继续收集，从而显著减少rollout数量；

**💡 创新点**

创新点在于把每一步的rollout收集视为有限时序分配问题，利用Beta–Binomial后验预测组是否会产生有效梯度，并使用两个阈值的SPRT式停止规则实现无额外预测rollout的高效分配；

**🔧 技术方法**

核心技术包括：贝塔先验更新、闭式后验预测有效性的Beta–Binomial公式、两阈值顺序停止策略、同步批处理分配与重新分配、以及对预算的精细控制；

**📊 数据集**

实验数据集涵盖数学与规划任务：MATH、Countdown、AIME24、AMC23、MATH500、Minerva、OlympiadBench；

**📈 对比分析**

与Uniform、History Resampling、DPS、动态抽样（DS）等基线对比，SARA在保持相同有效批大小的前提下，匹配DPS的准确度但花费更少的rollout；与DS相比，SARA在更少rollout下达到相近或更优的性能，+DPS组合在准确率与成本上均优于DS；

**⚠️ 局限性**

局限性包括仅适用于二值可验证奖励、假设同组rollout独立同分布；对连续奖励或树形rollout的推广尚未验证。

---

## 154. Post-Training at the Edge of Detectability: A Game-Theoretic Approach to Fine-Tuning

**arXiv ID:** 2607.26358 | [PDF](https://arxiv.org/pdf/2607.26358v1)

**作者:** Keegan Harris `[一作]` (University of California, Berkeley), Michael I. Jordan `[通讯]` (University of California, Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于序列检测博弈的 RL 细调框架，将奖励最大化与保持与参考策略相似性问题统一为博弈形式。

**💡 创新点**

创新点在于通过博弈论解析得到 KL 正则化 RL 的最优系数为平衡奖励与统计可辨识度的“阴影价格”，并提出随机二分法无超参数搜索求解该系数。

**🔧 技术方法**

使用博弈论、分数规划、KL 正则化 RL、序列假设检验与随机二分根求算法。

**📊 数据集**

实验使用简单 prompt “Once upon a time” 产生的生成文本，评估 token 长度奖励，并通过人类评测得到叙事和语法一致性。

**📈 对比分析**

与传统在 β 上做指数格点搜索对比，算法自动定位“elbow”点，获得更优的奖励-保留平衡；在模型审计任务中，使用基于 equilibrium 的 SPRT 在约 1-5 次样本内即可检测细调模型，误报率低于基准。

**⚠️ 局限性**

局限在于需要已知奖励函数和参考策略，理论依赖高置信区间与离散时间假设，实际 RL 近似求解和数据量需求较高，实验规模较小。

---

## 155. Between Gradient and Natural Gradient: A Continuum of LoRA Initializations

**arXiv ID:** 2607.26247 | [PDF](https://arxiv.org/pdf/2607.26247v1)

**作者:** Dianze Liu `[一作]` (Georgia Institute of Technology), Farshid Ghezelbash `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 810 | [OpenAlex ID](https://openalex.org/A5043020432)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一的LoRA初始化框架 ULoRA，阐明了现有梯度基初始化方法的统一性，并给出了可调参数 α（谱白化指数）和 β（Adam式对角预处理指数），从而在初始化时利用损失梯度与曲率信息。

**💡 创新点**

创新点在于：①把梯度预处理、谱白化与对角归一化融为一条两参数连续谱，使得 LoRA-GA、LoRA-One、CG-LoRA 等方法成为特例；②系统探究该谱中不同点的性能，发现最佳点往往位于两端之外；③设计了无搜索的 ULoRA‑Auto，根据每层谱统计自适应地选择 α、β，几乎匹配全局网格搜索的上界。

**🔧 技术方法**

核心技术包括：K‑FAC 曲率估计、Rayleigh‑Ritz 投影、谱白化、Adam式对角归一化、交叉熵 Fisher 校正、低秩 SVD 与平衡实现、以及跨层自适应指数策略。

**📊 数据集**

使用数据集：RoBERTa‑base 与 T5‑base 在 GLUE（CoLA、MNLI、QNLI、SST‑2、MRPC）上的微调；LLaMA‑2‑7B 在 GSM8K、HumanEval、MMLU 上的微调。

**📈 对比分析**

与 LoRA、LoRA‑+、rsLoRA、PiSSA、LoRA‑GA、LoRA‑One、CG‑LoRA 以及完整微调进行对比。ULoRA（oracle）在所有 GLUE 任务上匹配或超过完整微调；ULoRA‑Auto 在无需搜索的情况下，几乎达到上界，在大多数任务中居于可部署方法之首。

**⚠️ 局限性**

局限性包括：①上界实验使用了全局（α,β）网格搜索，未探索每层、每侧独立搜索的可能性；②只在 rank = 8、单轮训练、7B 模型上评估，缺乏更高秩、长周期或其他模态的验证；③初始化需要一次前向后向推理和本征分解，带来额外计算开销；④对最优指数的理论解释仍未给出。

---

## 156. Top-$k$ Pareto Bandits: Hypervolume Regret for Multi-Objective Slate Selection

**arXiv ID:** 2607.26273 | [PDF](https://arxiv.org/pdf/2607.26273v1)

**作者:** Nicolas Gutowski `[一作]` (Université d'Angers), Sylvain Lamprier `[通讯]` (Université d'Angers)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对多目标多臂赌博机（top‑k Pareto Bandits）的懈怠算法 THV‑UCB，用来在半博弈反馈下挑选 k 个臂，使其在预设参考点上的被支配体积（dominated hypervolume）最大化。

**💡 创新点**

创新点包括：① 将被支配体积作为集体奖励函数并证明其单调子模性质；② 设计了基于坐标上界置信盒的乐观估计，并用贪婪子模最大化构造最佳 slates；③ 给出无间隙（gap‑free）和有间隙（gap‑dependent）理论上界，证明在任意实例下实现亚线性 α‑近似超体积损失；④ 在实验中对比多种基线，展示了 THV‑UCB 在所有前沿几何形状和维度下均优于现有方法。

**🔧 技术方法**

主要技术包括：坐标上界置信盒 UCB、子模函数的贪婪最大化、被支配体积的 Lipschitz 连续性、集中不等式和安全剪枝机制，以及理论分析中的自适应偏差与方差平衡。

**📊 数据集**

使用的“数据集”为多维合成 Pareto 前沿，基于 DTLZ 参数化的线性、凸、凹和聚类四种几何形状，维度 d ∈ {2,3,4,5}，每个维度对应的臂数 n 及时间步 T 在实验中固定为 n=36、T=2000（k=3/4/5/6）。

**📈 对比分析**

与 ParetoUCB、ParetoTS、ChebyshevUCB、ScalarUCB 及其随机权重、随机选择等基线在同一实验环境下进行对比。结果显示 THV‑UCB 在累计 α‑超体积损失上显著更低，在最终被支配体积上最高，尤其在高维和聚类前沿场景下优势最为突出。

**⚠️ 局限性**

局限性：目前仅给出了上界，缺乏匹配下界；理论分析中假设臂间距足够分离以获得 gap‑dependent 结果；实验仅覆盖合成数据，尚未验证在真实业务场景（如推荐、投资组合）的泛化效果；对高 k 与大 n 的计算复杂度仍需进一步优化。

---

## 157. HeteroPROPMT: A Real-time and Privacy-Preserving Heterogeneous Collaborative Perception Framework

**arXiv ID:** 2607.26283 | [PDF](https://arxiv.org/pdf/2607.26283v1)

**作者:** Armin Maleki `[一作]` (Michigan State University), Hayder Radha `[通讯]` (Michigan State University)

**通讯引用:** 5527 | [OpenAlex ID](https://openalex.org/A5015107642)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量级、实时且隐私保护的异构协同感知框架（HeteroPROMPT），通过视觉提示（prompt）和FiLM调制将来自不同传感器和模型的BEV特征对齐到自我中心统一空间；

**💡 创新点**

创新点包括：①只使用两个低秩（PARAFAC）提示调制模块即可对齐特征；②通过自动编码器学习紧凑特征空间，实现无需元数据的模态推理；③冻结主感知与融合网络，仅对提示模块进行微调，极大降低参数量与计算成本；

**🔧 技术方法**

技术手段包括：视觉提示（prompt）+Feature-wise Linear Modulation（FiLM）；PARAFAC低秩分解提示；AutoEncoder（AE）用于压缩特征并提取模态指纹；多尺度金字塔融合与前景估计；轻量级MLP/ CNN分类器；

**📊 数据集**

使用OPV2V-H和V2XSet两个大型异构协同感知数据集，分别包含多种LiDAR和摄像头传感器及不同编码器（PointPillar、SECOND、EfficientNet、ResNet-50）等模态；

**📈 对比分析**

与HEAL、CoBEVT、HM‑ViT等基线在AP50/AP70上对比，HeteroPROMPT在所有异构场景下均提高1–2% AP50，同时参数量从1.1M降至0.064M（下降≈95%），显著提升训练吞吐量（↑2.5×）并降低显存占用（↓45%），在隐私与实时性方面表现突出；

**⚠️ 局限性**

局限性在于：仅针对已知模态进行训练，未评估对未知传感器/模型的泛化；未考虑高延迟、丢包、定位误差等更严苛网络环境；模型对极端光照或天气变化的鲁棒性尚待验证。

---

## 158. Dynamic Parameterization Is Not Dynamic Inference

**arXiv ID:** 2607.26192 | [PDF](https://arxiv.org/pdf/2607.26192v1)

**作者:** Zongfei Li `[一作]`, Guozhong Luo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了 Frozen‑Controller Auditing (FCA)，一种在冻结模型后通过缓存和重放控制器系数来评估输入依赖控制器的功能性与执行动态性的框架；同时对 FeatureGate 和 MUDDPythia 两类动态 Transformer 进行了系统实验。

**💡 创新点**

创新点在于将动态模型的评估拆分为三类：系数变化、功能依赖性（冻结模型对系数分配的敏感度）以及实际执行（层/子模块是否被跳过），并提供了可复现的 FCA 实现；通过交叉输入重分配、token 置换和静态配置等干预手段直接量化功能依赖。

**🔧 技术方法**

技术手段包括：
- 两阶段缓存与重放（先记录完整系数张量，后在冻结模型中重放）；
- 系数保持的干预（Cross‑input、Token‑shuffle、Cross+shuffle、Layer×PositionMean、StaticProfile 等）；
- 交互效应与静态保留度量；
- 对比实验中对 NLL、logit 重放一致性、层执行计数和系统延迟的测量；
- 对 FeatureGate 与 MUDDPythia 的细粒度对比分析。

**📊 数据集**

使用的数据集：WikiText‑103（验证/测试集）、OpenWebText（训练集）以及独立的校准集用于估计静态配置；实验也在 76M、504M 和 1.4B 参数规模上进行。

**📈 对比分析**

比较方法：
- 计算在不同干预下的 NLL 变化（ΔNLL）和静态保留率；
- 评估不同模型在相同输入下的层执行次数、延迟和吞吐量；
- 对 FeatureGate，交叉输入和 token 置换导致的 NLL 仅约 10⁻³，静态层平均配置几乎保持 99% 的性能；
- 对 MUDDPythia，交叉输入和 token 置换导致 NLL 上升 1–3，显示强功能依赖；
- 但两类模型均未实现任何层跳过，实际执行与 Dense 基线相当或更慢。

**⚠️ 局限性**

局限性：
- 评估仅覆盖 FeatureGate（76M、504M）和单一 MUDDPythia 1.4B 检查点；
- 训练规模和数据分布（OpenWebText、WikiText‑103）有限，无法推广至更大或不同任务的动态模型；
- 静态配置仅匹配系数均值或签名和，未能完全复制残差更新的实际效果；
- 结构移除参考在两类模型中定义不同，跨模型比较存在偏差；
- FCA 只评估功能依赖性，未能直接证明对推理时间或 FLOPs 的节省。

---

## 159. CG-World: A Large-Scale World-State Dataset and Protocol for World Models

**arXiv ID:** 2607.26452 | [PDF](https://arxiv.org/pdf/2607.26452v1)

**作者:** Yiming Cai `[一作]` (CGyear World Model Lab), Yong Guo `[通讯]` (CGyear World Model Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出了CG‑World世界状态数据协议及其大规模数据集，并在视频生成、动作预测和视语动作迁移三项任务中进行评估。

**💡 创新点**

创新点在于将工业计算机图形生产过程中的中间物理状态、渲染参数、动作与观测同步记录为统一的多模态时间序列；构建了包含事实轨迹、观测、动作、机制干预以及严格反事实分支的完整干预树；并在此基础上发布了约850k条长度1–5秒的世界状态段。

**🔧 技术方法**

技术方法包括：基于OpenUSD/RLDS/MaterialX的协议化数据抽取与重组；对LTX‑2.3、Cosmos3‑Nano、OpenVLA等大型模型进行LoRA微调；使用LPIPS、FVD、MSE、RMSE、Rotation Error、Gripper F1等多维度指标评估。

**📊 数据集**

数据集主要为CG‑World v1；实验中对LTX‑2.3（视频生成）、Cosmos3‑Nano（动作预测）和OpenVLA（视觉‑语言‑动作迁移）分别使用CG‑World进行后训练；对比基线使用LIBERO、ManiSkill3等公开数据。

**📈 对比分析**

评估采用相同初始帧、提示与随机种子，比较LoRA微调前后的指标。结果显示：在Depth分支LPIPS下降37.6%、FVD下降38.8%；在Canny分支LPIPS下降33.1%、FVD下降43.6%；动作预测MSE、Translation RMSE、Rotation Error分别下降约34%、20%、53%；OpenVLA在PickCube任务成功率从4%提升至39%，在StackCube OOD成功率提升至25%。

**⚠️ 局限性**

局限性包括：来源于工业CG的生产偏差（叙事选择、风格化运动、引擎/求解器特定行为），缺乏真实物理校准；观测与状态之间的对应可能因合成、透明材质等因素不完全对齐；因此数据更适合作为明确监督而非真实物理真相。

---

## 160. FAS-R1: A Unified Multi-Task MLLM for Reasoning Face Anti-Spoofing

**arXiv ID:** 2607.26432 | [PDF](https://arxiv.org/pdf/2607.26432v1)

**作者:** Hongyang Wang `[一作]` (Shijiazhuang Tiedao University), Zitong Yu `[通讯]` (Great Bay University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种两阶段的多模态语言模型框架FAS‑R1，能够统一完成真实性判断、攻击类型识别、欺骗区域定位及可解释理由生成。

**💡 创新点**

创新点包括：① 构建高质量长链式思考数据集FAS‑R1‑23K；② 提出针对面部欺骗的专属强化学习GRPO，并在其基础上设计降质模拟增强（DSA）和困难感知重加权（DA‑GRPO），显著提升鲁棒性与难样本学习；③ 将链式思考与强化学习相结合，实现稳定的证据导向推理。

**🔧 技术方法**

使用技术包括：Qwen2.5‑VL‑3B/7B全参数SFT；链式思考监督；阶段二的GRPO、DSA、DA‑GRPO；奖励函数基于答案格式、正确性、定位IoU及GPT‑5一致性验证。

**📊 数据集**

主要使用WMCA、PADISI‑Face、SiW‑Mv2三大公开面部欺骗数据集，构建出22,996样本的FAS‑R1‑23K长链式思考数据集。

**📈 对比分析**

与传统判别模型、FaceShield、PA‑FAS等方法进行对比；在WMCA、PADISI‑Face、SiW‑Mv2的三任务上，真实性分类ACC达98.75%，攻击类型ACC 93.33%，定位AP@40/50分别96.30%/94.73%；跨域验证显示在所有协议下均取得最优或接近最优；理由质量评估中Elo得分1803，明显优于基线。

**⚠️ 局限性**

局限性包括：对大量标注长链式思考数据的依赖；对极细微或新型攻击的推理深度有限；模型规模大、推理成本高；缺乏实时性评估与对抗鲁棒性分析。

---

## 161. IFCMemoryBench: Evaluating Long-Term Memory of LLM-Based Agents in BIM Information Retrieval

**arXiv ID:** 2607.26072 | [PDF](https://arxiv.org/pdf/2607.26072v1)

**作者:** Changyu Du `[一作]` (Technical University of Munich), André Borrmann `[通讯]` (Technical University of Munich)

**通讯引用:** 7080 | [OpenAlex ID](https://openalex.org/A5079755356)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了IFCMemoryBench，一个专为建筑信息建模（BIM）信息检索设计的长期记忆评测基准；

**💡 创新点**

创新点在于将原本无状态的BIM QA问题转换为多会话记忆任务，并通过人工验证构建跨项目、多会话的真实记忆依赖场景；

**🔧 技术方法**

采用了多种通用记忆体系（向量存储Mem0、图数据库Graphiti、Markdown文件管理），以及ReAct式检索代理和LLM评判器；

**📊 数据集**

数据集来源于IFC-Bench v2的类别4问题，扩展后生成143个任务，涵盖19个项目、23个IFC模型和4016个先前会话；

**📈 对比分析**

对比方法是固定probe LLM，分别测量记忆写入、检索和利用三个阶段的质量，结果显示最佳系统在部署真实写入范围下仅达32.4%答复准确率，记忆覆盖率不足一半；

**⚠️ 局限性**

主要限制包括：记忆仅从用户发言中提取，未覆盖助理/工具输出的持久信息；基准为人工合成，缺乏真实项目多模态输入；评判器与probe LLM同属一体系，可能存在自我偏好；

---

## 162. IDP AutoOpt: Agent-Driven Optimization of Document Processing Pipeline Configurations

**arXiv ID:** 2607.26075 | [PDF](https://arxiv.org/pdf/2607.26075v1)

**作者:** David Kaleko `[一作]` (Amazon Web Services), Md Mofijul Islam `[通讯]` (Amazon Web Services)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一个自主LLM代理，利用闭环评估迭代自动调优智能文档处理（IDP）流水线配置，显著降低人工配置时间。

**💡 创新点**

创新点在于将多模态LLM与手工编写的领域技能相结合，能够在混合配置空间（提示、模型、OCR、架构、模式等）上实现全局优化；发现LLM能力阈值及手工技能优于原始源代码访问；并提出上下文窗口压缩与方差缓解技术。

**🔧 技术方法**

使用技术包括：多模态LLM（Claude Sonnet/Opus 等）+工具访问；领域技能库；闭环评估与版本化配置；上下文窗口压缩；自动化部署（Bedrock AgentCore、CloudFormation、DynamoDB）。

**📊 数据集**

实验数据集包括 RealKIE FCC-Verified 广告发票数据集；OCR-Benchmark 分类任务集；DocSplit-Poly-Seq 拆包任务集。

**📈 对比分析**

通过与人类专家在提取、分类、拆包三项IDP任务上进行成本约束对比，LLM代理在提取任务上达到 90.2% 准确率、$0.022/页成本，较专家提升 8.6pp、成本降低 4.6×；在分类任务匹配专家且成本更低；在拆包任务提升 11% 同成本。

**⚠️ 局限性**

局限性包括：运行间方差大，需要多次执行；领域技能需人工编写，扩展成本高；评估集规模有限，未系统研究更大集对优化效果的影响；迭代未收敛，可能进一步提升但需更长时间。

---

## 163. Pramana: A Composable, Domain-Specific Backend for Empirical Networking Research

**arXiv ID:** 2607.26352 | [PDF](https://arxiv.org/pdf/2607.26352v1)

**作者:** Jaber Daneshamooz `[一作]` (University of California Santa Barbara), Arpit Gupta `[通讯]` (University of California Santa Barbara)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可组合的“薄腰”后端系统（Pramana），实现了意图规范语言，让研究者仅需描述实验意图与子系统，后端通过组合Mininet、NetReplica、NetUnicorn、NetGent等已有工具自动生成实验数据；

**💡 创新点**

创新点在于将实验的意图（what）、子系统（where）与实现机制（how）三维分离，提出统一的意图规范语言与能力协议，并通过可组合的底层架构实现跨工具、跨平台的实验复现；

**🔧 技术方法**

技术包括领域特定语言（DSL）定义意图语法、AI模型用于自然语言到DSL的编译、语义检查与能力匹配、可组合的插件化体系（NetGent/NetReplica/NetUnicorn/Mininet）以及分布式执行与遥测收集；

**📊 数据集**

使用了从过去十年测量论文中提取的意图语料库（覆盖约200+实验）以及CCAnalyzer等基准实验的数据；

**📈 对比分析**

通过对比现有专用工具在该语料库上的覆盖率，证明Pramana实现了原始目标的约55%（相当于现有工具的2倍以上）并在CCAnalyzer例子中将实验从数周缩短至约2.7秒编译+71秒数据传输；

**⚠️ 局限性**

局限性包括：目前实现仅覆盖语料库的一部分（约55%），仍需社区补齐；在高并发时遥测传输瓶颈尚未优化；AI仅用于编译步骤，可能限制对更复杂意图的支持；

---

## 164. Seeing or Knowing? Visual Context Sensitivity in Multimodal Large Language Models

**arXiv ID:** 2607.26326 | [PDF](https://arxiv.org/pdf/2607.26326v1)

**作者:** Jiaang Li `[一作]` (University of Copenhagen), Vésteinn Snæbjarnarson `[通讯]` (University of Copenhagen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多模态大语言模型在视觉任务中的失效机制，通过分离感知与利用两阶段并提出可控视觉上下文敏感性框架。

**💡 创新点**

提出了对抗性视觉对比基准和子空间控制键，能够定位并调节视觉与先验知识的权衡，证明失效主要源于利用而非感知。

**🔧 技术方法**

使用图像重建、对抗性基准、激活补丁定位关键层、以及残差流子空间干预等技术。

**📊 数据集**

采用Qwen2.5‑VL、Qwen3.5、Gemma‑4等模型，并使用VLMs‑are‑Biased、What‑If‑Visual‑Counterfactual基准及ImageNet‑1K训练解码器。

**📈 对比分析**

对比了原始模型、监督微调(SFT)和子空间控制的效果，原始模型对比率约14.7%，SFT提升至约52.7%，子空间控制提升至约37.7%；文本通道控制更为容易。

**⚠️ 局限性**

仅针对粗粒度属性，难以推广至需要推理的属性（如重量）；子空间控制需在训练集上学习，可能对不同架构的普适性有限。

---

## 165. Zero-Fi: Zero-Shot Wi-Fi-Based Human Activity Recognition via Contrastive Signal-Language Alignment

**arXiv ID:** 2607.26381 | [PDF](https://arxiv.org/pdf/2607.26381v1)

**作者:** Yitong Shen `[一作]` (University of South Florida), Yili Ren `[通讯]` (University of South Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种零样本Wi‑Fi 人体活动识别框架Zero-Fi，通过对Wi‑Fi信号与自然语言描述进行对比学习对齐，实现对未见活动类别的识别。

**💡 创新点**

创新点在于：①利用LLM生成的丰富动作描述将活动语义细化为身体部位运动属性；②将多模态（幅度、相位、DFS）Wi‑Fi特征融合并通过域判别器去除环境/硬件噪声；③在共享嵌入空间中进行双向对比学习，提升跨模态对齐效果。

**🔧 技术方法**

技术手段包括：Wi‑Fi信号处理（CSI、共轭乘法、STFT）、Transformer‑based 多模态编码器、域判别器与梯度反转、LLM（ChatGPT）生成动作描述、CLIP文本编码、InfoNCE对比损失。

**📊 数据集**

使用公开的大型Wi‑Fi 数据集 WiDAR 3.0（22类）和 XRF55（55类），合并后进行 75 类划分，其中 7 类为零样本测试。

**📈 对比分析**

与 THA​T、OneFi、CLAR、FM‑ZSL‑IoT、Wi‑CLIP 等基线在相同零样本协议下比较，Zero-Fi 在 trimmed‑mean accuracy 上取得 69.58%，显著优于 Wi‑CLIP（26.79%）和 FM‑ZSL‑IoT（23.86%），以及传统监督模型。

**⚠️ 局限性**

局限性包括：①对LLM生成描述的依赖，可能受模型偏差影响；②仍需人工定义结构化提示；③仅在活动类别层面验证，未充分评估跨人群或跨环境的泛化；④训练复杂度较高，需大规模算力。

---

## 166. DVPSFormer: Efficient Online Depth-aware Video Panoptic Segmentation for Autonomous Driving

**arXiv ID:** 2607.26165 | [PDF](https://arxiv.org/pdf/2607.26165v1)

**作者:** Yung-Hsu Yang `[一作]` (ETH Zürich), Marc Pollefeys `[通讯]` (ETH Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种统一在线的深度感知视频全景分割（DVPS）框架，能够同时完成语义分割、度量深度估计和实例追踪；

**💡 创新点**

创新点在于将分割过程视为显式场景离散化（ESD），通过单通量的离散到连续（D2C）深度头实现高效深度估计；以及在线多数投票（OMV）机制利用时间一致性提升实例分类；

**🔧 技术方法**

使用Transformer解码器与Mask2Former的分割头、基于查询的轻量追踪头、点采样与滤波的深度监督、以及相似度学习的轨迹关联；

**📊 数据集**

在Cityscapes-DVPS与SemKITTI-DVPS两个基准数据集上进行训练与评测；

**📈 对比分析**

与Uni-DVPS、MultiFormer等现有方法对比，取得了全新SOTA：Cityscapes-DVPS上DVPQ提升约3.9分，SemKITTI-DVPS上DVPQ提升至7.3分，且推理速度提升约18倍；

**⚠️ 局限性**

局限性包括：仍需高性能GPU（如RTX 4090）进行训练；对LiDAR投影深度的依赖可能在深度稀疏或噪声较大场景下表现受限；以及在线多数投票仅使用最近5帧，可能无法充分利用更长时间上下文。

---

## 167. MoSAIC: Aligned Intervention Supervision for Part-Local Motion Style Transfer

**arXiv ID:** 2607.26304 | [PDF](https://arxiv.org/pdf/2607.26304v1)

**作者:** Nazanin Amini `[一作]`, Kevin Desai `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了多源局部姿势风格迁移，并提出MoSAIC框架，支持用户在动画序列中为每个骨架部位指定不同的参考源，实现局部风格编辑。

**💡 创新点**

创新点在于提出对齐干预监督（aligned intervention supervision）解决缺失反事实监督问题，能够同时学习局部响应与保留不变部分；并通过内容、根轨迹、部位风格的分层潜在扩散网络实现可控路由。

**🔧 技术方法**

技术方法包括：冻结的六部分VAE与MotionCLIP进行内容与风格分解；带解耦注入的DiT式潜在扩散模型；对齐干预生成同步参考与对照；DDIM采样与分类器自由引导。

**📊 数据集**

主要使用HumanML3D数据集进行训练和评估；在CMU MoCap数据集上验证整体内容‑风格迁移性能。

**📈 对比分析**

与全身路由对比，掩码路由在选定部位响应提升约0.198，保留区误差降低4.19mm，离目标泄漏降低8.20mm；在HumanML3D上保持FID≈0.308，CMU指标SRA≈50%，CRA≈38%，说明保持了生成质量并提升了局部控制效果。

**⚠️ 局限性**

局限性包括：评估仅基于合成的对齐干预，缺少自然局部配对数据；上肢响应相对弱，跨部位隔离不完全；实验只使用单一训练种子，未开展多模型或人类主观评估。

---

## 168. Foundational Refinement Proofs for Deployed Bytecode, at the Price of Tokens

**arXiv ID:** 2607.26306 | [PDF](https://arxiv.org/pdf/2607.26306v1)

**作者:** Lefteris Lazaropoulos `[一作]` (Argot Collective), Zoe Paraskevopoulou `[通讯]` (Argot Collective)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

使用大型语言模型自动生成针对已部署以太坊字节码的基础层可重放证明，证明字节码与其高层规格的 Refinement 关系。

**💡 创新点**

创新点在于：① 为任意来源的 EVM 字节码提供完整的 Refinement 证明框架；② 将证明成本转化为 token 费用，几乎不需要人工干预；③ 设计了 Sol‑minor 规格语言和可组合的证明库，支持跨编译器、跨语言的交互。

**🔧 技术方法**

主要技术包括：Lean4 证明助手与可组合证明库（reach invariant、opcode combinator、loop induction等）、OpenAI GPT‑5.5/Anthropic Opus LLM 代理、可执行 EVM 语义模型，以及自定义的规格语言与 ABI 解析。

**📊 数据集**

使用的实验数据集为 23 个真实部署的以太坊智能合约（包括 MakerDAO、WETH9、Nouns、UniswapV2Pair 等），字节码长度最高约 10 KB。

**📈 对比分析**

通过对每个合约的 token 花费、证明时间、生成证明文件大小等指标进行评估，平均每个合约需约 10⁸ token、约 100 小时、约 7.6 万行证明代码，表明在成本与可扩展性方面显著优于传统手工证明。

**⚠️ 局限性**

主要局限包括：未对 gas 消耗和子状态做完整建模；LLM 需要高 token 费用和长时间运行；对外部函数（如 Keccak）和 native_decide 产生额外可信面；复杂合约仍可能因资源限制导致证明超时。

---

## 169. When benchmark inferences do not compose: Projectibility in AI evaluation

**arXiv ID:** 2607.26159 | [PDF](https://arxiv.org/pdf/2607.26159v1)

**作者:** Brett Reynolds `[一作]` (Humber Polytechnic), Brett Reynolds `[通讯]` (Humber Polytechnic)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5055886085)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出非组合原理，阐明基准结果与实际使用之间的推理链不一定能自动组成，并开发了项目可行性审计框架，用来检查链中各环节是否符合对象、样本、条件、结果与时间的对齐与依赖传递。

**💡 创新点**

创新点在于：①引入非组合原则，揭示单独被证实的链条并不能自动合成；②将项目可行性审计具体化为一套完整的声明与检查流程；③将有效性评估与项目可行性结合，明确各环节的证据需求。

**🔧 技术方法**

技术手段包括：论证基础的有效性框架、类型理论描述的节点与边、统计方法（置信区间、贝叶斯估计、交叉验证）、仿真与重分析、以及案例模拟（法律研究与对照实验）。

**📊 数据集**

主要数据集为公开的法律问答基准（如 MMLU‑Pro）与作者重新分析的 zhang2026illusionRobustness 公开数据；此外通过模拟生成的法律研究案例、检索扰动实验和律师审核记录作为验证案例。

**📈 对比分析**

对比方法：用传统基准评估与非组合审计两者在同一法律案例链上做对照，演示传统评估忽视接口缺口导致的推理失效；性能指标主要是证据链完整性与是否能通过项目可行性声明而非数值准确度。

**⚠️ 局限性**

局限性包括：需要开发者与部署方共同提供完整信息，单方难以覆盖全部链条；对因果、测量、预测等子问题仍需专业工具；非组合原则本身不提供完整归纳理论，仍需结合具体领域知识；实施成本高，且在商业化环境下可能被策略性框架规避；数据可访问性与隐私限制也会影响完整审计。

---

## 170. EvoPINN: Agentic Discovery of Executable Algorithms for Physics-Informed Neural Networks

**arXiv ID:** 2607.26490 | [PDF](https://arxiv.org/pdf/2607.26490v1)

**作者:** Peng Yin `[一作]` (University of Chinese Academy of Sciences), Jian Cheng `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 EvoPINN，一个执行驱动的智能体框架，通过 LLM 迭代生成、验证并演化 PINN 的神经表示与训练程序，自动发现高效的学习算法；其中首次演化到 SLRC‑PINN 这一自局部化纠正的全局‑局部网络结构。

**💡 创新点**

核心创新在于（1）将 PINN 设计拆分为两块（表示层 M 与训练程序 T），实现模块化演化并使用 UCB 调度；（2）利用训练诊断、演化记忆与搜索焦点等信息引导 LLM 生成可执行且结构新颖的程序；（3）构建严格的结构验证、可执行测试与预算匹配评估流程，保证科学有效性；（4）在多种 PDE 领域中从零开始发现与专家设计同级甚至更优的算法。

**🔧 技术方法**

技术包括：大语言模型（gpt‑5.4‑mini）生成代码、抽象语法树 (AST) 结构对比、可执行性修复、基于 UCB 的模块调度、诊断摘要 D_g、演化记忆与多模搜索焦点、共享训练预算与基准评估、DeepXDE 与 PyTorch 实现。

**📊 数据集**

使用四类 PDE 数据集：Poisson2D（COMSOL 参考）、Burgers1D（数值参考）、Wave1D（解析解）和 Heat2D（解析解），在每个领域分别评估算法性能。

**📈 对比分析**

与种子 PINN、16 种专家设计的 PINN（含 Fourier、SIREN、NTK 等）以及其他自动搜索框架（DPSTE、PINNsAgent、无反馈 56 试验）进行对比；EvoPINN 在 Poisson2D、Burgers1D、Wave1D 的相对 L₂误差均最低，Heat2D 虽略逊但仍具竞争力；在不同搜索策略下的平均误差显著低于传统方法，并在邻近参数下保持优良的零射转移性能。

**⚠️ 局限性**

局限包括：对 LLM 代码合成与科学推理能力的依赖，候选算法全预算训练导致搜索成本高，未引入代理评估或科学微调模型以加速筛选。

---

## 171. Global Exponential Stabilization of the Kinematic Bicycle Model of a Car in Polar Coordinates

**arXiv ID:** 2607.26442 | [PDF](https://arxiv.org/pdf/2607.26442v1)

**作者:** Velimir Todorovski `[一作]` (University of California San Diego), Miroslav Krstic `[通讯]` (University of California San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种基于极坐标与范围归一化坐标的后向嵌入控制方法，实现了汽车式双轮车的并排泊车；

**💡 创新点**

创新点在于利用极坐标与范围归一化坐标绕过Brockett不可能性，构造严格反馈形式并设计光滑全局指数输出稳定的反馈律，首次在极坐标下实现全局指数泊车控制；

**🔧 技术方法**

采用极坐标变换、范围归一化坐标、后向嵌入设计、Lyapunov 夹逼与控制Lyapunov函数（CLF）等控制理论技术；

**📊 数据集**

未使用真实数据集，仅通过仿真验证；

**📈 对比分析**

通过对比前进与后退泊车仿真轨迹、输入平滑性以及理论证明的指数收敛，展示了控制律的全局指数输出稳定性和人类式泊车轨迹；未给出数值性能指标；

**⚠️ 局限性**

局限在于在笛卡尔坐标下仅具有吸引性而非稳定性，靠近目标时需要大幅控制，且理论验证未在实验平台上验证。

---

## 172. When Fish Look Alike: Tracking Identities with Dual-branch Elasticity

**arXiv ID:** 2607.26412 | [PDF](https://arxiv.org/pdf/2607.26412v1)

**作者:** Vran Lee `[一作]` (China Agricultural University), Zhenbo Li `[通讯]` (China Agricultural University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为TIDE的多目标跟踪框架，针对密集同质的鱼群进行实时跟踪。

**💡 创新点**

创新点在于双分支弹性设计（轻量化与可扩展化）以及自适应几何对应IoU（AGCIoU）机制，完全去除昂贵的Re-ID网络。

**🔧 技术方法**

采用JDE联合检测与嵌入、Conv-C3/Conv-Up-C3、C3TR、CSConv等轻量模块，并结合密度图回归和动态任务不确定性权重。

**📊 数据集**

使用专门构建的MFT‑Edge边缘测试数据集以及公开的MFT25大规模数据集进行评估。

**📈 对比分析**

与SDE主流方法（如SU‑T、ByteTrack、TFMFT）以及JDE/Transformer框架对比，轻量分支仅消耗20.47G FLOPs、5.79M参数，却获得28.43 HOTA，轻量化约38.7倍算力节省；可扩展分支进一步提升至29.98 HOTA。

**⚠️ 局限性**

在极端遮挡或极密集聚集时仍会出现ID切换，且仍依赖线性卡尔曼预测，对大幅形变或运动漂移的鲁棒性有限。

---

## 173. Multi-Decoder OneRec: Controllable Generative Retrieval for Multi-Objective Industrial Recommendation

**arXiv ID:** 2607.26500 | [PDF](https://arxiv.org/pdf/2607.26500v1)

**作者:** You Wang `[一作]` (Kuaishou Technology), Wenwu Ou `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多解码器OneRec框架，通过共享用户上下文、通用解码器与目标特定LoRA专家，在固定候选预算下实现多目标可控生成检索。

**💡 创新点**

引入梯度隔离的LoRA专家、基于相对奖励的KL正则化策略优化以及多解码器约束束搜索，既保持共享建模又实现目标特定控制与候选互补。

**🔧 技术方法**

使用Semantic ID生成、低秩适配LoRA、下一词预测、强化学习奖励优化、KL正则化与多路束搜索等技术。

**📊 数据集**

公开的Kwai26亿级多目标检索基准，包含1.31亿原始交互记录、25.03M有效Semantic ID项。

**📈 对比分析**

与DSSM、SASRec、HSTU、TIGER、单解码OneRec等基线在512候选预算下对比，Recall@512提升1.69–5.62%，A/B测试提升使用时长0.37%和冷启动2.09%。

**⚠️ 局限性**

仍受限于固定候选预算下的搜索开销、LoRA容量限制以及对连续目标仅适用的相对奖励设计，未解决多路解码器的计算与存储扩展。

---

## 174. Which RAG Paradigm Wins at Scale? A Scaling Study of Retrieval-Augmented Generation Paradigms

**arXiv ID:** 2607.26497 | [PDF](https://arxiv.org/pdf/2607.26497v1)

**作者:** Pengyu Wang `[一作]` (University of Science and Technology of China), Licheng Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究在企业级大规模语料上构建了 28 层嵌套规模阶梯，统一对 BM25、DenseRAG、图式 RAG（MS‑GraphRAG、LightRAG、HippoRAG2、LinearRAG）以及基于文件系统的 Agent 进行对比实验。

**💡 创新点**

创新点在于：①首次在同一实验框架下同时测量构建成本、查询成本与准确性；②引入“代理访问层”将检索结构与代理策略解耦，评估两者对性能的独立影响；③利用 28 层嵌套阶梯揭示不同 RAG 方案随语料规模扩展的真实壁垒。

**🔧 技术方法**

使用的技术包括 Qwen3.6-27B 生成式读者、Qwen3-Embedding-0.6B 嵌入模型、BM25 倒排索引、稠密向量检索、图结构构建（实体、关系、社区、PageRank）以及 LLM 代理工具调用循环；同时采用 token‑级计量与 VLLM 并行执行。

**📊 数据集**

数据集为 EnterpriseRAG‑Bench，包含 511,959 篇多源文档（约 600M token）以及 500 题目、722 份 gold 文档，加入 326 份陷阱与 99 份误导文档，形成真实的企业知识库环境。

**📈 对比分析**

实验对比方式：在所有 28 层阶梯上对同一 150 题集进行一次性与迭代检索，记录构建 token、查询 token、延迟及官方综合分数；结果显示 BM25 在所有规模下保持 Pareto‑optimal，文件系统 Agent 在最小规模竞争力强但随规模放大退化；图式 RAG 受构建成本壁垒限制；Agent+BM25 在全规模时将分数从 36.9 提升至 69.4，且查询 token 仅为原 Agent 的 1/9。

**⚠️ 局限性**

局限性包括：①图式索引构建成本随规模呈超线性，导致 10^5–10^6 文档难以扩展；②实体提取噪声与谓词丢失削弱图结构信息；③实验仅覆盖单一企业式语料，缺乏跨域验证；④代理检索受调用预算与模型能力限制，未探索自适应预算分配。

---

## 175. LLMET: Enabling Cross-Layer Evaluation of Emerging M3D Memories for Energy-Efficient LLM Serving

**arXiv ID:** 2607.26491 | [PDF](https://arxiv.org/pdf/2607.26491v1)

**作者:** Ming-Yen Lee `[一作]` (Georgia Institute of Technology), Shimeng Yu `[通讯]` (Georgia Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一套名为LLMET的跨层仿真框架，用来评估大规模LLM推理中使用M3D 2T-GC内存时的能耗与性能变化。

**💡 创新点**

创新点在于：① 结合前端算子映射与后端设备级能耗/面积模型，实现从电路层到系统层的完整能耗评估；② 引入容量感知映射与算子融合策略，精确捕捉大容量缓存对访存流量的影响；③ 在框架中首次支持M3D等新兴3D内存技术，提供可扩展的跨层分析。

**🔧 技术方法**

使用的技术包括：M3D 2T-GC 3D堆叠内存、NS-Cache、ASAP7 RTL合成、NeuroSim、Systolic Array和向量单元等硬件模块；前端采用算子级跟踪与容量感知映射；后端实现设备级功耗/面积模型；在不同节点（7nm、3nm）上进行技术投影。

**📊 数据集**

主要评估了Llama 3.1 70B、Llama 3.1 405B、Llama 3.2 1B等模型，并在服务器平台（A100、B200-like）和边缘平台（Jetson Orin NX）上通过多种工作负载（长上下文QA、代码生成、对话、文档合成、语音指令等）进行实验。

**📈 对比分析**

通过与公开的A100数据对比验证仿真精度（面积误差<7%），实验显示在服务器端将L2缓存从40MB扩展到1GB可将预填充阶段能耗降低44%，在B200-like平台缓存从128MB扩展到4GB能降低24%；在边缘平台将缓存从8MB提升到256MB时解码能耗可降低30%。

**⚠️ 局限性**

局限性包括：仿真仍基于模型与投影，缺乏真实硬件实现验证；主要关注能耗与面积，不同工作负载的实际吞吐量影响未完整评估；对M3D技术的制造成本和热管理仍未深入探讨。

---

## 176. DualDecoder: Accelerate Long Context LLM Inference by Predictive Prefetch

**arXiv ID:** 2607.26475 | [PDF](https://arxiv.org/pdf/2607.26475v1)

**作者:** Zuning Liang `[一作]` (Fudan University Shanghai Innovation Institute), Yuan Cheng `[通讯]` (Fudan University Shanghai Innovation Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对长上下文的大模型推理，提出了一种轻量级的KV缓存预取系统 DualDecoder，利用对下一步关键-值（KV）索引的可预测性，在推理期间提前从主机内存拉取所需的KV条目，从而消除传统稀疏KV缓存中 GPU 上占用大量空间的辅助状态。

**💡 创新点**

创新点包括：1）利用在每一步解码中同时生成的“推测”token，精确预测下一步的KV检索索引；2）设计双token解码管线，在单个模型核内同时计算真实token和推测token，几乎不增加计算开销；3）基于层级的KV预取调度，提前将预取的KV与计算重叠，同时优先处理预测错误导致的缺失KV；4）在需要时自适应恢复低秩key缓存，仅对预测精度低的层使用；5）采用两层 ping‑pong 缓冲和 GPU 侧 KV 组装，显著降低 GPU 上的 KV 存储量。

**🔧 技术方法**

技术手段包括：稀疏KV缓存（如 ShadowKV 方案）、双token解码、KV索引预测、层级预取调度、优先级流调度、关键-值重构、层级KV内存管理、GPU 侧 KV 组装与重排、CUDA 流与事件同步。

**📊 数据集**

使用的主要数据集有：RULER（长上下文基准），NIAH‑1、NIAH‑2、NIAH‑3、QA‑2 等；模型为 Llama‑3‑8B‑Instruct‑Gradient‑1048k、Qwen‑2.5‑8B/14B/32B 等；实验上下文长度从 64K 到 512K tokens。

**📈 对比分析**

与 vLLM、SpeCache、ShadowKV 三个主流稀疏KV缓存系统对比。实验结果显示：1）在 128K–512K 上下文长度下，DualDecoder 的解码吞吐量比 ShadowKV 高 2.62×，比 SpeCache 高 1.32×，比 vLLM 高 9.02×；2）GPU 内存占用相比 ShadowKV 与 SpeCache 分别降低 36–62%；3）延迟（TPOT/TTFT）与基线基本一致；4）请求完成时间（P99）在高请求速率下提升 2.5×，满足更严格的 QoS 需求。

**⚠️ 局限性**

限制与挑战：1）对KV索引预测的准确性高度依赖初始推测token，随机初始化会显著降低准确率；2）预测误差仍会导致缺失KV，需要优先级调度，误差率过高时仍会出现延迟波动；3）自适应关键重构仅在预测精度低的层使用，仍需额外的低秩key缓存，无法完全消除所有辅助状态；4）该方案假设模型结构符合典型 Transformer 设计，特殊模型或大型多模态模型的迁移性需进一步验证。

---

## 177. MultivationBench: A Benchmark for Multimodal Sequential Motivation Reasoning

**arXiv ID:** 2607.26465 | [PDF](https://arxiv.org/pdf/2607.26465v1)

**作者:** Kawai Chung `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 MulTivationBench，一套基于多模态视觉叙事的顺序动机推理基准，用以评估大型语言模型在累计上下文中推断人物动机的能力。

**💡 创新点**

创新点在于将 Maslow 层次与 Reiss 基本欲望两大心理学框架并行应用，构造多标签推理任务；通过 AI‑人类双轮管道自动生成并人工验证行为与动机标签；以及引入故事级一致性度量，评估模型在整条叙事线上的持续推理表现。

**🔧 技术方法**

采用多模态大模型（如 Grok‑4.1‑Fast、Gemini‑3‑Flash、Llama‑4 系列、Phi‑4‑multimodal‑instruct 等）进行零样本推理；使用 EM 与 F1 两种评价指标；在三种输入模式（多模态、文本仅、图像仅）以及故事长度分段（短/中/长）下进行实验。

**📊 数据集**

数据来源融合 MovieBench、StoryReasoning 与 SSID，构成 1,000 条视觉叙事，4,023 条行为实例，共 16,092 个评估问题，涵盖 Maslow（8 类）与 Reiss（16 类）四个多标签任务。

**📈 对比分析**

实验结果显示最优 EM 仅达 39%（Phi‑4‑multimodal‑instruct），最佳 F1 约 55%（Grok‑4.1‑Fast）。模型在多模态输入下明显优于文本或图像单独输入，但在故事级一致性上均低于 1%，表明模型难以在完整叙事中保持一致推理；与人类对比，模型仍存在显著性能差距。

**⚠️ 局限性**

限制主要包括：评估仅为被动观察，未涉及主动预测或干预；对长序列的记忆与更新能力不足，易产生过度解释；依赖现有公开数据集，存在版权与样本多样性限制；目前模型在细粒度 Reiss 任务上的性能仍不理想。

---

## 178. From Micro-Cognition to Self-Construction: A Four-Layer Integrative Review of Psychological Theories in HCI

**arXiv ID:** 2607.26402 | [PDF](https://arxiv.org/pdf/2607.26402v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 179. Existence-Field Diffusion Model for Spatial Point Processes with Variable Cardinality

**arXiv ID:** 2607.26428 | [PDF](https://arxiv.org/pdf/2607.26428v1)

**作者:** Xiaoyin Pan `[一作]` (University of California, Riverside), Chengkuan Hong `[通讯]` (Zhongguancun Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种连续变量存在场扩散模型 (EFDM)，统一在单一扩散过程中同时建模点的空间位置和集合大小。

**💡 创新点**

创新点在于为每个潜在点引入存在变量并对其进行对数几率变换，使得点的出现与否在扩散过程中可连续变化，避免了先前方法中离散跳跃和不对称的维度变换。

**🔧 技术方法**

使用扩散模型框架，对位置和存在变量一起进行高斯噪声扩散和反向去噪；对存在变量做对数几率变换；训练时使用噪声预测损失；采样时通过 Sigmoid 转为概率再伯努利采样得到实际点集合。

**📊 数据集**

实验数据集包括：1) 合成数据集（多种卡尔丹性和对应空间分布）；2) 路易斯维尔 e‑scooter 日常上车位置数据；3) QM9 分子数据集（3D 原子坐标与种类）。

**📈 对比分析**

与三类基线（独立点扩散 IDM、固定维度扩散 FDDM、转维度扩散 TDDM）比较。EFDM 在条件评价下对空间统计（均值、方差、偏度、峰度、最近邻距离）的 Wasserstein‑1 距离最小，说明能更好捕获卡尔丹性与空间结构的联合分布；在无条件评价中表现仍保持竞争力。对于分子生成，EFDM 在原子稳定性、分子稳定性与有效性等指标上均优于 FDDM，且在原子计数分布的 Wasserstein‑1 距离上也最小。相比之下，TDDM 的性能受限于离散跳跃，且在高卡尔丹性范围内表现不佳。

**⚠️ 局限性**

局限性包括：当卡尔丹性与空间分布无关时可能出现过拟合；模型需要预先设定最大点数 N，导致额外的计算和内存开销；目前仅适用于静态空间点过程，尚未扩展到时空点过程。

---

## 180. PSG: Pair-Space Generation for Efficient Generative Reranking

**arXiv ID:** 2607.26427 | [PDF](https://arxiv.org/pdf/2607.26427v1)

**作者:** Chao Feng `[一作]` (Kuaishou Tech), Xiang Li `[通讯]` (Kuaishou Tech)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Pair-Space Generation（PSG），将生成单元从单个候选物品提升为有序物品对，减少生成长度并保持表达力；

**💡 创新点**

创新点在于：①用物品对作为生成单元实现生成长度减半，理论上可达2~4倍速度提升；②通过预训练的Pair-Token Representation模块解决n²词表稀疏问题；③证明PSG与传统单项生成等价且在worst-case suboptimality上提高约4倍；

**🔧 技术方法**

使用的技术包括：Transformer Encoder-Decoder结构、动态词表解码、Pair-Token Representation（基于MLP的组合编码）、GRPO强化学习（组相对策略优化）以及基于经验的pretrain与next-token prediction；

**📊 数据集**

实验数据集包括公开的ML-1M、Amazon-Books以及工业短视频平台RecFlow，候选数分别为50、50和60；

**📈 对比分析**

与多种Generator-Only与Generator-Evaluator基线（如Seq2Slate、SetRank、NAR4Rec、GoalRank等）对比，PSG在NDCG@6、Precision@6、Recall@6、F1@6上分别提升约2-8%，在线A/B测试中提升0.178%停留时间、生成器延迟1.83×，可支持约80%更多QPS；

**⚠️ 局限性**

限制在于：k>2时词表规模爆炸导致内存和延迟难以控制，当前仅在k=2下表现可接受；此外，Pair-Token Representation的预训练依赖大规模曝光日志，若日志稀缺可能影响性能；

---

## 181. FleetScape: A Mixed Reality Sandtable for Spatial Supervision and Control of Scalable Drone Fleets

**arXiv ID:** 2607.26423 | [PDF](https://arxiv.org/pdf/2607.26423v1)

**作者:** Peisen Xu `[一作]` (National University of Singapore), Christophe Jouffrais `[通讯]` (CNRS)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个混合现实沙盘接口 FleetScape，用于多无人机舰队的空间监督与控制，并在模拟建筑检查任务中验证其可行性。

**💡 创新点**

将舰队监督重新定义为空间交互，利用3D 沙盘与多层实时数据融合实现从全局监督到局部手动控制的无缝切换；提出可扩展的设计空间并通过实验验证可管理的舰队规模上限。

**🔧 技术方法**

使用 Meta Quest 3 混合现实硬件、Unity3D 引擎、GPU 计算着色器点云渲染、实时无线数据流、Meta Touch Pro 控制器交互以及 SAGAT、SART、Bedford 工作负荷、TAM 等评估方法。

**📊 数据集**

采用 City Pack 城市资产、真实建筑边界与占用图（可来自 Google Maps、BIM 等）以及通过模拟生成的多无人机轨迹、环境点云和安全事件数据。

**📈 对比分析**

在六名专业操作者的实地实验中，对比不同舰队规模（5、10、15架）下的情境意识、认知负荷、任务完成度，发现情境意识下降、认知负荷上升、每架无人机的工作效率随规模增大而降低；实验证实可管理舰队上限约为10架。

**⚠️ 局限性**

样本量有限、仅在模拟环境中评估、缺乏真实世界通信延迟与动态环境挑战、实验时间短且不涵盖长期或高危任务，限制了结果的普适性。

---

## 182. Flow Map Learning via Nongradient Vector Flow

**arXiv ID:** 2607.26398 | [PDF](https://arxiv.org/pdf/2607.26398v1)

**作者:** Mark Goldstein `[一作]` (Flatiron Institute), Rajesh Ranganath `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种新的流映射学习方法——StopGrad Flow，能够在不需要可逆网络或模型逆的前提下学习概率流ODE的流图。

**💡 创新点**

创新点在于使用停梯度（stopgrad）与非保守动力学结合，通过PDE残差损失保证真实流图为唯一稳态点，同时消除了对模型逆和嵌套梯度的需求。

**🔧 技术方法**

采用了JVP（Jacobian-Vector Product）计算、停梯度、模拟无损的损失函数以及对ODE解与速度的联合训练。

**📊 数据集**

在CIFAR-10无条件图像生成任务上进行实验。

**📈 对比分析**

与流匹配（Flow Matching）、Meanflow以及Lagrangian map matching比较，SG Flow在10步采样时获得最低FID（3.85），在其他步数仍保持竞争力，并且是唯一拥有停梯度稳态点证明的方法。

**⚠️ 局限性**

局限性包括对网络架构未做针对性优化、仅在CIFAR-10上验证、未与模型蒸馏方法直接对比，以及在极少步数下性能略逊于专门针对1步的Meanflow。

---

## 183. NMKFR: A Robust Framework for Time-Aware Cold-Start Recommendation

**arXiv ID:** 2607.26429 | [PDF](https://arxiv.org/pdf/2607.26429v1)

**作者:** Chengzhi Liu `[一作]` (Southwest University), Zehui Qu `[通讯]` (Southwest University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为NMKFR的神经记忆卡尔曼融合推荐器，专门解决在时间变化环境下的物品冷启动问题。

**💡 创新点**

创新点在于：①将Titans式记忆增强的语义编码与时间感知卡尔曼状态追踪相结合；②利用后验协方差作为不确定性信号，动态调节语义记忆检索和静态-动态特征融合；③设计了自适应比较融合模块（ACFM），在融合过程中显式比较静态语义与时序状态的相似度与差异；④在多种噪声与稀缺历史下证明了自适应融合的鲁棒性。

**🔧 技术方法**

主要技术包括：Titans记忆网络、时间归一化与矩阵指数式时间转移的卡尔曼滤波、后验协方差驱动的反馈机制、交互特征融合、混合列表/点对/点式损失的联合训练。

**📊 数据集**

使用的公开数据集为Amazon Video Games（文本简短、互动稀疏）和MovieLens‑32M（标题+标签、较长文本）。

**📈 对比分析**

与12种基线（包括BERT4Rec、SASRec、Mamba4Rec等）在Time‑aware Cold‑Start和Item Cold‑Start两种协议下进行比较，使用Recall@5/10/20、NDCG@5/10/20、MRR指标。NMKFR在两大数据集的所有指标均显著优于基线，最大提升可达3.4%（Recall）和3.6%（NDCG）。Ablation实验进一步验证每一子模块的贡献，鲁棒性实验表明在噪声或历史不足的情况下仍保持领先。

**⚠️ 局限性**

局限性包括：①评估仅在离线采样排名协议下，未验证在线实时推荐效果；②仅使用文本信息，未利用多模态特征；③模型结构相对复杂，训练成本和推理时延较高；④对超参数（如卡尔曼过程噪声、时间归一化等）仍需手工调优。

---

## 184. HERMES: A Hybrid Ensemble for Head-and-Neck Tumor Segmentation, TN Staging, and Recurrence-Free Survival on PET/CT

**arXiv ID:** 2607.26498 | [PDF](https://arxiv.org/pdf/2607.26498v1)

**作者:** Kai Wang `[一作]` (University of Colorado School of Medicine), Moyed Miften `[通讯]` (University of Colorado School of Medicine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了HERMES算法，实现头颈部PET/CT一次性完成肿瘤分割、T/N分期与复发无病生存预测。

**💡 创新点**

创新点包括用分割掩码的几何特征替代通用放射组学以提升N分期准确度，以及使用一致权重融合与对齐损失来实现更稳健的生存预测。

**🔧 技术方法**

采用STU-Net小型网络10折集成进行分割，3D ResNet-18深度专家与L1正则化逻辑回归进行分期，使用对齐损失训练生存模型，并通过等权重z平均实现多专家融合。

**📊 数据集**

使用HECKTOR 2026公开训练集782例多中心PET/CT与18维临床变量，验证集约50例。

**📈 对比分析**

通过内部10/5折交叉验证与公开验证集比较，分割Mean Dice为0.641，T、N分期平衡准确度分别为0.580/0.642，复发无病生存C指数为0.679；几何特征在N分期上相较放射组学提升约0.03。

**⚠️ 局限性**

局限性包括受限于分割精度导致几何特征受噪声影响，缺失侧向性信息限制N分期完整性，T分期受解剖关系影响难以完全捕获，生存评估未考虑竞争风险与校准。

---

## 185. Conformal Changepoint Localization and Root Cause Analysis with Corrupted Observations

**arXiv ID:** 2607.26481 | [PDF](https://arxiv.org/pdf/2607.26481v1)

**作者:** Seunghun Yu `[一作]` (Korea Advanced Institute of Science and Technology), Osvaldo Simeone `[通讯]` (Northeastern University London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出在受Huber型污染的情况下，对离散事件序列进行置信集式的变更点定位和根因分析，并通过加权合成实现对污染观测的鲁棒性；

**💡 创新点**

创新点在于：①设计基于权重的CONCH和CROC（W-CONCH/W-CROC），利用观测不确定性信号下采样以缩小置信集；②提出元学习机制(MW-CONCH/MW-CROC)，无需预知污染率即可学习最优权重；③证明在污染模型下保持分布无关的覆盖保证；

**🔧 技术方法**

核心技术包括：分布式非参数合成预测、Huber污染模型、基于分类器的不确定性估计（EDL、MC Dropout等）、权重自适应和元学习的梯度优化；

**📊 数据集**

实验使用DomainNet、CIFAR-100、Milan电信活动、以及自构的多流CIFAR-100和MNIST根因基准，覆盖多种污染程度；

**📈 对比分析**

与原始CONCH/CROC及oracle加权方法对比，W-CONCH/W-CROC在高污染下显著降低置信集大小（如DomainNet从137降至18），且覆盖率保持≈95%；元学习版本在不知污染率时进一步提升效率，甚至超过oracle；

**⚠️ 局限性**

局限在于：权重阈值对污染率的依赖仍需经验选择；元学习需要大量多任务训练；方法仍是离线批处理，无法直接应用于实时在线检测；

---

## 186. Audio-Anchored Fusion of Multi-Ratio DiT Reconstruction Residuals for Cross-Domain Audio Deepfake Detection

**arXiv ID:** 2607.26472 | [PDF](https://arxiv.org/pdf/2607.26472v1)

**作者:** Haotian Mo `[一作]` (National University of Defense Technology), Qinglin Wang `[通讯]` (National University of Defense Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了基于大规模真实语音训练的Diffusion Transformer（DiT）重建探针，提取多比例残差并与冻结的WavLM-Large音频特征通过音频锚定的加性融合进行深度伪造检测

**💡 创新点**

创新点在于：①使用仅训练真实语音的DiT生成多比例残差，保留局部不匹配信息；②提出音频锚定的加性融合方式，防止残差域移位抑制音频分支；③证明辅助监督对动态竞争融合有负面影响，支持残差补充证据而非竞争

**🔧 技术方法**

使用的技术包括Diffusion Transformer（F5-TTS变体）、Masked Spectrogram Reconstruction、WavLM-Large自监督声学编码、ResNet-18残差编码、三比例残差合成、加性融合、三元组损失与攻击ID辅助损失、固定原型评分

**📊 数据集**

训练与评估数据集：ASVspoof 5 Train/Dev/Eval（攻击A01–A32）、ITW Full；DiT预训练使用ASVspoof 5 Train、ASVspoof 2019 Train、Common Voice、VoxCeleb2、LibriSpeech共255万真实语音；评估还对单流WavLM–ResNet18基准进行对比

**📈 对比分析**

与单流WavLM–ResNet18基准以及公开系统对比，音频锚定残差融合在ASVspoof 5评估EER约6.5%/min-DCF 0.18，在ITW Full评估EER降至13.8%（三种随机种子平均15.3%±2.1%），显著优于单流基准（18.3%±4.98%）且比动态竞争融合表现更稳健

**⚠️ 局限性**

局限性包括：残差不具域鲁棒性；残差与音频分支时长不匹配；仅使用单向冻结的DiT探针，未探索多次重建或不确定性；评估仅覆盖ASVspoof 5→ITW迁移，缺乏更广泛跨域验证；辅助监督对动态竞争融合的负面影响仅在本实验中观察到，未系统分析原因

---

## 187. RLMM-Flow: A Flow-based Mobile Manipulation Framework with Latent-Space Reinforcement Learning

**arXiv ID:** 2607.26460 | [PDF](https://arxiv.org/pdf/2607.26460v1)

**作者:** Shuhang Wang `[一作]`, Hui Cheng `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于流匹配的移动机械臂整体运动生成框架 RLMM-Flow，通过在预训练的流匹配网络上进行潜在空间强化学习，对噪声进行可控 steering，从而提升任务成功率、碰撞避免和轨迹平滑度。

**💡 创新点**

创新点包括：① 采用级联 transformer 先生成底盘动作再生成机械臂动作，显式建模底盘-臂结构；② 通过冻结预训练流匹配网络，仅在潜在噪声空间进行强化学习；③ 引入 Q^A 预热与粗细层级潜在 steering，既提升价值估计稳定性，又降低高维潜在搜索难度。

**🔧 技术方法**

使用的核心技术包括流匹配(flow matching)与点云条件 transformer、潜在空间强化学习、离线 RL（IQL、AWR、DSRL 风格的价值蒸馏）、Q^A 预热、粗细层级潜在 steering、以及基于离线数据的奖励设计。

**📊 数据集**

数据集来源于：① 9000 条基于 REMANI 规划器的高质量专家轨迹；② 通过放松 REMANI 条件生成 1500 条低质量轨迹形成离线 RL 数据集；③ 场景包括 PhyScene 的模拟室内环境与 FastLIO2 捕获的真实点云，分别用于训练、验证与真实部署。

**📈 对比分析**

与 M2Diffuser（无/有优化）、Base Flow、动作空间 AWR/IQL、DSRL‑NA(Flow) 等方法进行比较。RLMM-Flow 在见过场景、未见场景以及真实环境的实验中均表现出最高的成功率、最低的碰撞率、关节违规率和轨迹不连续性，同时保持与 Base Flow 相近的推理时间；在真实部署中同样保持最佳性能。

**⚠️ 局限性**

局限性包括：① 仍基于离线数据，无法自适应实时环境变化；② 对极端动态或极少见的几何布局的泛化仍有限；③ 真实环境下因感知噪声与执行误差导致成功率下降；④ 需要预训练与精细超参数调节，部署成本相对较高。

---

## 188. StrataCL: Fabric-Native Communication Library for Production Supernodes

**arXiv ID:** 2607.26444 | [PDF](https://arxiv.org/pdf/2607.26444v1)

**作者:** Tiancheng Hu `[一作]` (Peking University), Chenxi Wang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种零冗余、面向超节点的通信库，在生产环境下显著提升分布式 AI 的通信效率。

**💡 创新点**

创新点包括：注册即分配机制、全网格编程抽象、工作负载均衡的 NPU 核划分以及基于 SDMA 的 NPU 驱动通信卸载。

**🔧 技术方法**

采用了异步注册、影子虚拟地址、全网格收发、负载均衡调度、SDMA offload 以及 NPU 侧性能剖析等技术。

**📊 数据集**

使用了 DeepSeek V4 Flash LLM 推理、DeepSeek V3.2 671B 训练、DLRM 推荐系统训练等真实生产工作负载；在数据集方面采用了 Criteo Terabyte、Splitwise 对话等。

**📈 对比分析**

与 NCCL/HCCL、HCCL-zerocopy、DeepEP 等基线对比，AllGather、AllReduce 及 MoE dispatch/combine 的总线带宽提升高达 1.6×/1.4×，LLM 推理吞吐提升 1.9×，P99 TTFT 降低 2.2×，训练迭代时间分别缩短 1.4×/1.3×。

**⚠️ 局限性**

局限包括：在极大规模或高延迟网络环境下，全网格执行的拥塞与网络拓扑敏感；SDMA 占用仍有 9% 的延迟开销；对动态内存分配模式的处理仍需额外元数据广播。

---

## 189. Reinforcement Learning on Cost-Constrained Quadrupedal Hardware

**arXiv ID:** 2607.26434 | [PDF](https://arxiv.org/pdf/2607.26434v1)

**作者:** Javier C. Weddington `[一作]` (Stanford University), Stephen A. Baccus `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在低成本四足机器人 Mini Pupper 2 上，通过强化学习训练并部署时间感知网络，实现了在 76 ms 延迟环境下的稳健步态，并展现了自发产生的中央模式产生器（CPG）；

**💡 创新点**

创新点在于证明在高传输延迟的低成本硬件上，LSTM 能自发学习 CPG 解决方案，且只需单一调参即可实现无反馈开环部署，从而大幅降低硬件成本与部署工程复杂度；

**🔧 技术方法**

采用了 Isaac Lab PPO 强化学习框架、对比 LSTM/GRU/Transformer/MLP 四种网络架构、前馈 PD 观察桥、仿真延迟模型、POMDP 框架以及两层部署架构（LSTM + MLP 伺服预测器）；

**📊 数据集**

使用了 IsaacSim 生成的自定义四足仿真环境（约 4098 并行环境、500 Hz 物理步长）以及 Mini Pupper 2 真实实验数据（传感器噪声、延迟特性等）；

**📈 对比分析**

通过在相同奖励、环境及课程设置下对四种网络进行收敛、部署复杂度和延迟鲁棒性比较；LSTM 在 76 ms 延迟下获得 271 奖励且步频稳健；MLP 需要课程和至少六个手调参数；GRU 与 Transformer 无法收敛；LSTM 在 +320 ms 延迟时仍保持 5.3 Hz 步频；

**⚠️ 局限性**

限制包括对仿真延迟与硬件匹配的高度依赖、LSTM 需要手动调节硬件比例、训练对算力要求高、模型对高阶噪声与硬件偏差敏感，以及两层架构在极低数据量下可能受限。

---

## 190. DIRECTOR: Dynamic Index-based Recommendation with Transport-Optimized Retrieval

**arXiv ID:** 2607.26418 | [PDF](https://arxiv.org/pdf/2607.26418v1)

**作者:** Yuanhao Pu `[一作]` (University of Science & Technology of China), Defu Lian `[通讯]` (University of Science & Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种全局协同的非自回归重排序框架 DIRECTOR，利用动态检索索引生成、容量约束的最优传输训练和全局硬匹配推断，同时通过前缀锚定信用分配实现对不可微评估器的学习。

**💡 创新点**

创新点包括：①在所有位置同时生成连续检索索引而非逐步生成，①使用熵正则化的容量约束最优传输实现位置间冲突感知的软匹配训练，②在推断时直接做全局硬匹配得到无重复的最终 slate，③通过前缀锚定路径实现对仅返回标量奖励的评估器的细粒度信用分配。

**🔧 技术方法**

技术包括：动态检索索引生成（Conditional VAE 或条件扩散模型）、容量约束熵正则化最优传输（软传输）以及全局硬匹配（最短增广路径算法），前缀锚定信用分配与策略梯度相结合。

**📊 数据集**

离线数据集：ML‑1M、Amazon‑Books、RecFlow；线上实验：Kuaishou 短视频推荐平台（10% 流量对照组 vs 10% 处理组）。

**📈 对比分析**

与多种基准（点式排序器、上下文感知重排序器、AR 生成器、Generator‑Evaluator 方法）进行对比。DIRECTOR 在 ML‑1M、Amazon‑Books、RecFlow 上 NDCG@6 提升约 2.8%‑3.7%，在线 A/B 测试中 VV 提升 0.52%（p<0.05），CPU 消耗下降 66.7% 同时保持相同吞吐量与 30 ms P99 延迟。

**⚠️ 局限性**

局限性：①对评估器的依赖较强，若评估器不可训练或不稳定会影响生成；②在极大候选集（M≳10⁴）时全局硬匹配的求解成本仍然是瓶颈；③当前实现主要在固定 slate 长度下验证，动态长度或多层级 slate 的适配尚未探讨。

---

## 191. SCOUT: Per-Context Reset Curricula for Sparse-Reward Reinforcement Learning

**arXiv ID:** 2607.26417 | [PDF](https://arxiv.org/pdf/2607.26417v1)

**作者:** Siddharth Aphale `[一作]` (Stanford University), Ayushman Singh `[通讯]` (Sesame AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了SCOUT，一种在线、无监督的逐个任务实例（context）级别的重置课程调度器，用来解决稀疏奖励强化学习中的起始障碍。

**💡 创新点**

创新点在于仅使用二值成功反馈即可动态地为每个任务实例分配和撤销中间起始状态，解决了传统全局或基于组的课程调度在不同学习进度下的失效问题。

**🔧 技术方法**

使用了基于二值成功的前沿控制算法，结合SAC+HER、PPO等主流学习器，并利用轨迹生成的重置梯子来自动构造起始状态。

**📊 数据集**

在六种导航与操控环境（PointMaze、MiniGrid MultiRoom、MiniGrid DoorKey、FetchPickAndPlace、Meta-World PickPlace）以及构造的“PickAndPlace”冲突组上进行实验。

**📈 对比分析**

与目标起始训练、随机重置、全局退化、RFCL/ACED等基线相比，SCOUT在大多数情形下实现了更高的AUC和最终成功率，特别是在学习进度差异显著的冲突场景中表现突出。

**⚠️ 局限性**

局限性包括需要环境支持精确状态重置、对构造的起始梯子质量依赖、在极大任务库时可能需要额外分组或更精细的证据统计，且在物理机器人无标记重置的情况下难以直接应用。

---

## 192. Knowledge before Reasoning: EC-Reason-Bench, a Training-Free Diagnostic Benchmark for LLM Enzyme Classification

**arXiv ID:** 2607.26397 | [PDF](https://arxiv.org/pdf/2607.26397v1)

**作者:** Linyu Li `[一作]` (Peking University), Nyima Tashi `[通讯]` (Tibet University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 EC-Reason-Bench，一个无训练、诊断式的酶 EC 预测基准，拆分出格式化失败与知识缺失两大问题，并通过四个互斥支路（输出结构、外部知识、推理结构、推理鲁棒）进行评估。

**💡 创新点**

创新点在于将酶功能预测的全流程拆解为可单独度量的四个支路，形成可复现的、无训练的诊断框架；并首次量化通用 LLM 在外部知识缺失时的性能崩塌及其在检索辅助下的“重建”能力。

**🔧 技术方法**

使用的技术包括：零样本直接生成基线、层级级联（多选）提示、基于检索的 agentic search（ReAct），基于机制的链式思考（CoT）和自一致性（Self‑Consistency）推理。

**📊 数据集**

数据集为 1349 个酶样本，划分为低同源、介质、对抗性和多标签四个子集，配合 4963 条 EC 树结构与 5 条离线检索通道（BLAST、HMMER、ESM‑kNN、Foldseek 结构、活性位点）。

**📈 对比分析**

对比结果显示：闭书设置 L4 准确率仅 0.16 左右，开启检索后提升至 0.70‑0.76，接近或略优于专门模型；与单纯邻居投票基线 0.723 相比，LLM 的增益整体接近零，但在对抗性与多标签子集表现出显著（正或负）差异；自一致性提升模型校准和深度。

**⚠️ 局限性**

局限性包括：仍高度依赖检索数据库的同源覆盖；推理改进有限，仅在特定子集带来明显收益；未对 LLM 进行微调，评估范围受限于离线缓存；未能在实际在线环境验证检索与推理的协同效果。

---

## 193. Inapproximability of Unique-Machine Precedence Scheduling for Unit-Length Jobs

**arXiv ID:** 2607.26590 | [PDF](https://arxiv.org/pdf/2607.26590v1)

**作者:** Venkatesan Guruswami `[一作]` (University of California Berkeley), Shaoxuan Tang `[通讯]` (Tsinghua University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了唯一机优先级调度（UMPS）问题，证明了单元长度版本在任何常数因子上都是近似不可解的，并且在多项式时间内无法获得(log n)^γ 的近似；并将该结果推广到两种通信延迟调度模型；同时给出一个简单的4/3近似不可解证明。

**💡 创新点**

创新点在于构造了一种从超图着色promise问题到UMPS的多项式时间gap reduction，并利用超图组合技术对参数进行放大，从而实现常数因子及多项式时间内(log n)^γ 的下界；此外首次将UMPS的近似不可解结果与通信延迟模型的约简联系起来。

**🔧 技术方法**

主要技术包括：1) 超图着色的硬度结果（Guruswami&Lee 2018、Guruswami等 2017）；2) 通过vertex和edge机器gadget设计的时间窗口编码；3) 超图组合（composition）放大参数；4) 约简传递到通信延迟调度模型。

**📊 数据集**

作为理论论文，作者未使用实际数据集，所有结果均基于理论构造和已知的超图着色难度假设。

**📈 对比分析**

与以往仅有的5/4近似不可解结果相比，本文将下界提升至任意常数因子，并进一步得到多项式时间内(log n)^γ 的不可能近似；论文未给出算法，只提供了硬度证明。

**⚠️ 局限性**

限制：仅给出了硬度证明，未提出可行的近似算法；结果依赖于假设NP不属于准多项式时间；对实际调度实例的指导意义有限。

---

## 194. CDN Tsunami: Exploiting HTTP/3-HTTP/1.1 Conversion for DoS Attacks

**arXiv ID:** 2607.26589 | [PDF](https://arxiv.org/pdf/2607.26589v1)

**作者:** Ziyu Lin `[一作]` (National University of Singapore), Biplab Sikdar `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并实现了针对CDN服务的两种基于HTTP/3到HTTP/1.1转换的放大攻击（HBA和HCA），并在六大主流CDN上验证其可行性。

**💡 创新点**

创新点在于揭示HTTP/3与CDN混合部署导致的协议转换漏洞，提出利用QPACK动态表和多路复用实现高放大因子的攻击手段。

**🔧 技术方法**

使用的技术包括QUIC/HTTP3协议分析、QPACK压缩机制、HTTP/3多路复用、基于Apache的实验平台、以及自动化测量框架。

**📊 数据集**

数据集采用Tranco Top 1M域名列表进行子域枚举，筛选出151,685个托管于易受攻击CDN的子域，其中42,330个已启用HTTP/3。

**📈 对比分析**

与现有放大攻击对比，HBA在静态表下放大因子≈66×，动态表下≈350×；HCA可在300秒内消耗主机所有连接资源，实验中五家CDN出现明显延时或超时。

**⚠️ 局限性**

局限性包括实验受控带宽与连接数，未在大规模真实网络环境中验证；且仅适用于CDN仅支持HTTP/1.1到主站的混合部署。

---

## 195. R-SLPR: Region-based Small-to-Large Point-cloud Registration with Contrastive Learning

**arXiv ID:** 2607.26583 | [PDF](https://arxiv.org/pdf/2607.26583v1)

**作者:** Yusen Wan `[一作]` (University of Washington), Xu Chen `[通讯]` (University of Washington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出R‑SLPR框架，用区域提议–匹配–迭代细化的三阶段流程实现小到大点云配准。

**💡 创新点**

创新点：①采用Fibonacci Grid Segmentation生成均匀方向锚点；②在区域级别引入对比学习提升特征判别性；③设计Cascade Anchor Selection and Refinement (CASR) 迭代更新锚点，显著提高定位精度。

**🔧 技术方法**

技术手段：基于深度配准网络（以RPMNet为骨干）结合对比学习损失；Fibonacci Grid Segmentation生成锚点；CASR迭代算法；评价使用平均距离与SE(3)误差。

**📊 数据集**

使用数据集：ModelNet40（工业级合成数据）和ISOPR（工业仿真点云）。

**📈 对比分析**

与ICP、Small_GICP、PointNetLK、PointNetLK‑revisited、RPMNet、MFGNet、RORNet、MCLNet等方法在clean、unseen、noise、noise+unseen等四种实验设置下对比，R‑SLPR在旋转MAE约1.1°、平移MAE 0.009、RMSE均显著低于所有基线。

**⚠️ 局限性**

局限性：对极稀疏或极小片段的配准仍可能失败；对噪声的鲁棒性虽优于部分基线，但在高噪声环境下仍易受影响；当前实现为分块+网络+后处理，未实现完全端到端统一学习。

---

## 196. Where Detectors Fail: Closing the Tail-Domain Gap with Expert-Guided Mutual Distillation

**arXiv ID:** 2607.26555 | [PDF](https://arxiv.org/pdf/2607.26555v1)

**作者:** Xuan Feng `[一作]` (Jinan University), Bo An `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种专家引导的互相蒸馏（EGMD）框架，用于解决多模态假新闻检测中的领域偏差和语义不匹配问题。

**💡 创新点**

创新点包括：① 在输入层引入基于CLIP相似度的共振增益校准；② 通过动态领域特定归一化（DDN）和条件Mixture-of-Experts实现表示层去偏；③ 在决策层使用领域原型锚定和双通道互相蒸馏，既保留特征几何，又抑制局部领域先验。

**🔧 技术方法**

使用BERT、MAE、CLIP、DDN、MoE、互相蒸馏（feature‑level InfoNCE + logit‑level KL）以及领域原型记忆等技术。

**📊 数据集**

在四个数据集上评估：中文的 Weibo、Weibo21 以及自制的 Weibo_Balanced（平衡版）和英文的 FineFake。

**📈 对比分析**

与 MFAN、MRML、COOLANT、FSRU、MMDFND、MiMOE‑FND、DAMMFND 等基线对比，EGMD 在四个数据集的平均准确率、最低域准确率和总误差差距（Total）均优于或接近最优，显著降低领域间误差差距（最多可低至 57.3%）。

**⚠️ 局限性**

局限性：① 在某些高资源领域（如军事）仍略逊；② 需要在训练时获取领域标签；③ 复杂的教师网络在推理时被裁剪为轻量化学生，仍存在训练成本和模型复杂度；④ 仍未完全解决所有语义冲突导致的误判。

---

## 197. MedARC: Training-Free Adaptive Redundancy Compression of Visual Tokens for 3D Medical Vision-Language Models

**arXiv ID:** 2607.26554 | [PDF](https://arxiv.org/pdf/2607.26554v1)

**作者:** Yitao Zhu `[一作]` (Hong Kong Polytechnic University), Anqi Qiu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了MedARC，一种无训练、可自适应压缩3D医学视觉-语言模型视觉标记的方法。

**💡 创新点**

将视觉注意力、视觉-文本相似度和冻结3D视觉基础模型的结构显著性融合成多信号重要性评分，并通过保留主标记与合并冗余标记实现压缩。

**🔧 技术方法**

利用多路重要性分数加权、基于相似度的上下文锚点合并、以及冻结的3D VFM 结构特征等技术。

**📊 数据集**

在CT-RATE和MR-RATE这两个大规模医学影像数据集上进行实验。

**📈 对比分析**

与无压缩模型及四种无训练压缩基线（Hulu-L1、VisionZip、HiPrune、MedPruner）对比，MedARC在保持约80%标记保留率下在报告生成和VQA指标上超过多数基线，并在约50%保留率下仍保持竞争力。

**⚠️ 局限性**

对MRI任务的压缩敏感度较高，最佳标记保留率受模态影响，且未对稀有病灶的鲁棒性进行深入评估。

---

## 198. ThinkOmni: A Reasoning-Driven Omni-Modal LLM Framework for Audio Forgery Detection and Localization

**arXiv ID:** 2607.26553 | [PDF](https://arxiv.org/pdf/2607.26553v1)

**作者:** Yuxiong Xu `[一作]` (Shenzhen University), Sheng Li `[通讯]` (Afirstsoft Technology Group Co., Ltd.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于大语言模型的推理驱动全模态框架 ThinkOmni，用于音频伪造检测与局部时间定位，并通过结构化推理链实现显式证据分析。

**💡 创新点**

创新点在于：①构建了 100K 样本的 Forensic‑Aware Chain‑of‑Thought（FACoT）数据集，提供语义、声学与视觉层面细粒度的推理标注；②提出 Forensic‑Aware Modality‑Incremental Learning（FMIL）逐步对齐语义、声学和谱图视觉特征；③设计 Forensic‑Consistent Multi‑task Loss（FCML）将推理生成、伪造判别与时间边界回归联合优化。

**🔧 技术方法**

核心技术包括：大语言模型 Qwen2.5‑Omni、SAFE 交叉注意力模块、CLAP 语音‑文本一致性过滤、Weighted Cross‑Entropy 与自适应定位损失等；训练采用 LoRA 微调、逐步模态融合和多任务损失。

**📊 数据集**

使用来自 8 个公开数据集（ASVspoof‑2019 LA、HAD、PartialSpoof、LAV‑DF、ArEnAV、LlamaPartialSpoof、SINE、AV‑Deepfake1M++）的 100K 样本进行训练，并在原始数据集以及 ADD、Speech‑Forensics 等交叉数据集上评估。

**📈 对比分析**

与传统 SSL‑和 ALLM‑基准相比，ThinkOmni 在检测上实现了 93.70% ACC / 93.72% F1，跨数据集提升 34% 左右；在时间定位上达 88.05% mAP，跨数据集 mAP 达 74.67%，显著优于同类方法（最大提升约 15‑30%）。

**⚠️ 局限性**

局限性包括：推理与定位过程对计算资源消耗较大，推理时长较长；对极端噪声或未知生成器的鲁棒性尚待进一步验证；缺乏对不同语言、语音环境的广泛适配与误差可解释性。

---

## 199. A Persona-based Rate Action Index

**arXiv ID:** 2607.26545 | [PDF](https://arxiv.org/pdf/2607.26545v1)

**作者:** Hayden Helm `[一作]` (Helivan), Andrew Dassori `[通讯]` (Wavelength Capital)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个由公开声明组成的时间序列FOMC成员语料库，并利用检索增强生成模型为每位成员创建数字化人格，进而构建基于人格的利率行动指数。

**💡 创新点**

创新点在于首次证明可通过集合数字化人格来追踪并预测时间变化的系统层面协方差——即FOMC的政策利率走向。

**🔧 技术方法**

技术上采用检索增强生成、语言模型生成人格、嵌入投影、Kendall τ评估以及线性分类器。

**📊 数据集**

使用了包含2006-2026年24,333块公共声明的FOMC成员语料库，覆盖17名成员的演讲、证词、会议记录和新闻发布会。

**📈 对比分析**

与随机、持久性、泰勒规则、宏观回归、检索文本、静态查询集以及直接投票等基线对比，PBI在2022-2025窗口的三类预测准确率为0.69，领先基线；在时间序列上与利率级联的Kendall τ从0.42提升至0.84，约提前三季度。

**⚠️ 局限性**

局限包括成员语料量不均、仅使用官方公开文本缺乏媒体/社交信息、训练数据截止2023年导致早期预测受模型权重影响、对新主席/成员变化的适应性未知，以及对切除动作的预测不足。

---

## 200. The Art of Not Forgetting A Local Learning Architecture for Continual Learning

**arXiv ID:** 2607.26523 | [PDF](https://arxiv.org/pdf/2607.26523v1)

**作者:** Ashmith Atmuri `[一作]` (Arkadhi Research), Yashaswini Rao Bhogarajula `[通讯]` (Arkadhi Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实验了 CMP（Cognitive Memory Primitive）这一持续学习架构，用稀疏关系编码、竞争式记忆与预测编码等模块在字节级语言建模任务上评估其对灾难性遗忘的抵抗能力。

**💡 创新点**

通过将稀疏关系绑定、两层竞争式记忆、层级预测编码和基于参数运动的可塑性调节机制集成到一个局部学习框架中，首次展示了无需端到端反向传播即可显著降低灾难性遗忘的可能性。

**🔧 技术方法**

稀疏关系绑定（k‑wins），两层竞争记忆，层级预测编码，线性读取器的局部 delta 规则，以及基于累计参数运动的可塑性调节。

**📊 数据集**

自制的 15 领域字节级文本语料库（维基百科、Python 源码、莎士比亚、圣经、新闻等）以及 Split‑MNIST 作为对照实验。

**📈 对比分析**

与参数匹配的 Transformer（Naive + Online‑EWC）在相同数据预算和训练协议下对比，使用 BWT/FWT 指标。CMP 在 BWT 上相较 Naive 下降 94%，相较 Online‑EWC 降低 93%，但单域 BPB 仍显著高于 Transformer；Split‑MNIST 上无显著提升。

**⚠️ 局限性**

单域性能低于 Transformer，模型对域序敏感，基准数据集为自制且未覆盖公开 benchmark，Split‑MNIST 上未显示优势，缺乏更广泛的泛化验证。

---

## 201. HiFloat4 Format for End-To-End Reinforcement Learning Post-Training of Large Language Models

**arXiv ID:** 2607.26515 | [PDF](https://arxiv.org/pdf/2607.26515v1)

**作者:** Hei Yi Mak `[一作]` (Huawei), Anandharaju Durai Raju `[通讯]` (Huawei)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了首个全FP4强化学习后训练框架，在推理和训练两阶段均使用4位量化，并提出Rollout Residual Quantization (RRQ) 解决激活量化导致的零化问题。

**💡 创新点**

①证明FP4 RL的主要瓶颈是rollout激活量化；②提出RRQ——在FP4矩阵乘中加入稀疏残差校正，兼顾精度与计算效率；③验证HiFloat4三层层次缩放是最优的FP4格式。

**🔧 技术方法**

采用HiFloat4（HiF4）和MXFP4量化格式，使用GRPO强化学习算法，结合S_2:4半结构稀疏残差，并与SmoothQuant、RHT、OCC等方法对比。

**📊 数据集**

在Qwen2.5-3B上使用GSM8K和Math-500；在Qwen2.5-Math-7B上使用DAPO-Math-17K、AIME2024/2025、AMC-2023等数学推理基准。

**📈 对比分析**

与BF16全精度、无修正FP4、SmoothQuant、RHT、OCC等基线在同一仿真环境下对比，RRQ在HiF4下将BF16误差从4.9%降至1.1%，在MXFP4下从13.6%降至5.3%，显著提升平均奖励并加速收敛。

**⚠️ 局限性**

实验依赖FP4仿真，缺乏原生硬件支持导致训练时间长；仅在50%稀疏度下测试；未评估更大模型或多任务数据；无法测量实际速度提升。

---

## 202. ASARL: Autonomous Social-Aware Relevance Learning for QQ Search

**arXiv ID:** 2607.26593 | [PDF](https://arxiv.org/pdf/2607.26593v1)

**作者:** Tao Su `[一作]` (Tencent PCG), Hui Wang `[通讯]` (Tencent PCG)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在社交搜索领域提出了ASARL框架，自动化生成社交语义相关数据并训练符合用户偏好的模型，提升搜索质量与用户参与度。

**💡 创新点**

创新点包括：①多智能体协作生成和校验可解释的相关性标签（ReasonAgent、CriticAgent、GenAgent）以解决数据稀缺与逻辑不一致问题；②三阶段训练（SCT、PGO、SD）融合社交语境、用户行为偏好与模型蒸馏，实现高质量、可部署的社交搜索模型。

**🔧 技术方法**

使用大语言模型驱动的智能体、链式推理（Social‑Aware CoT）、偏好导向优化（DPO）、知识蒸馏以及多阶段监督学习。

**📊 数据集**

基于腾讯QQ搜索日志构建的约110万条查询‑标题对数据集，覆盖群组搜索与频道搜索，按教育、游戏、健康等多领域统一抽样。

**📈 对比分析**

与BERT、RoBERTa、LLM Base等基线对比，ASARL_8B在离线Macro‑F1、NDCG、Accuracy上分别提升至约83/78/85；在线A/B测试中CTR+JR+GSB提升2–3%，显著提升用户体验和平台社交增长。

**⚠️ 局限性**

主要局限：高度依赖大模型和计算资源，可能在多模态或非中文场景下效果不佳；生成标签与人工审核对齐仍需进一步验证；对特定社群语境的适应性需持续评估。

---

## 203. MultiFixer: A Coordinator-Proposer Based Multi-Agent Framework For Fixing Multi-Hunk Bugs

**arXiv ID:** 2607.26591 | [PDF](https://arxiv.org/pdf/2607.26591v1)

**作者:** Haichuan Hu `[一作]` (Nanjing University of Science and Technology), Quanjun Zhang `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于多代理的框架，用来解决多块（multi‑hunk）程序修复问题。

**💡 创新点**

创新点在于引入生成‑识别不对称机制，结合动态 hunk 调度、候选生成与选择，实现分步、协同的多块修复流程。

**🔧 技术方法**

采用 LLM 驱动的工具增强式 Bug 分析、细粒度上下文构建、hunk 级候选生成、hunk 调度代理以及两阶段语法与语义精炼技术。

**📊 数据集**

在 Defects4J、VUL4J、SEC‑bench 和 PatchEval 四个基准数据集上进行评估。

**📈 对比分析**

与 11 个 SOTA APR 基线对比，GPT‑3.5 版在 Defects4J 取得 326 个正确修复（其中 62 为多方法、27 为多文件），VUL4J 修复 24/79 病毒，SEC‑bench 11/65，PatchEval 19/120；在多块修复场景中显著优于现有方法。

**⚠️ 局限性**

对 10+ 块或具有复杂多级依赖的 bug，调度与循环可能耗尽预算，导致修复效果下降；高 hunk 数量时性能进一步受限。

---

## 204. One Run Is Not an Idea: The Implementation Lottery in Automated Research

**arXiv ID:** 2607.26587 | [PDF](https://arxiv.org/pdf/2607.26587v1)

**作者:** Jingjie Ning `[一作]` (Carnegie Mellon University), Chenyan Xiong `[通讯]` (Carnegie Mellon University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对自动化研究系统中的“实现彩票”进行量化评估，提出并验证Idea Reliability Audit，测量想法可靠性（ICC）和实现选择导致的赢家反转率。

**💡 创新点**

创新点在于：①把想法抽象为机制卡并对其进行语义门控；②在实现层面做三次独立实现并进行盲评；③用ICClike指标与留一实现反转来衡量想法可靠性，区别于传统的best‑of‑N artifact评估。

**🔧 技术方法**

使用：多层嵌套方差分析、ICC估计、留一实现反转（LOO）、LLM驱动的盲评、相同代码重复运行、Python/NumPy/Scikit‑learn实现，两个编码代理（Bounded与Agentic）。

**📊 数据集**

数据集：13个公开OpenML分类任务（约500–6400样本、4–60特征）和3个Matbench衍生回归任务（钢强度、带隙、声子频率）。

**📈 对比分析**

比较方法：对比实现方差与同一实现重复方差；对比两种实现流程的ICC和反转率；对比在卡级别过滤前后结果。性能表现为实现方差占ITT方差的33–45%，ICC在0.5–0.6区间，单次实现赢家反转率在25.6–43.6%。

**⚠️ 局限性**

局限性：仅在限定任务池、每任务四张卡、两种实现流程、单一评估器下测试；卡级审查后置，未实现前置门控；不评估最佳‑N稳定性和更深层次深度学习流水线；材料实验为探索性且样本量小，结果不易推广。

---

## 205. Level, Sharpness, and Corpus: Why Zero-Shot OOD Detector Rankings Do Not Transfer

**arXiv ID:** 2607.26582 | [PDF](https://arxiv.org/pdf/2607.26582v1)

**作者:** Ignacio M. De la Jara `[一作]` (University of Adelaide), Damith Ranasinghe `[通讯]` (University of Adelaide)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对17个ID数据集和三种VLM进行零样本OOD检测器可迁移性审计，并提出了不需要训练的Complementary Evidence Guard（CEG）框架。

**💡 创新点**

创新点在于揭示检测器排名随ID域和VLM变化会反转，发现level与sharpness等证据通道互补性，并设计CEG通过保留多证据来提高部署鲁棒性。

**🔧 技术方法**

使用CLIP等VLM的logit特征，提取level、sharpness和WordNet外部语义证据，利用ID样本的经验分位数进行校准，采用非补偿式最小化融合的CEG方法。

**📊 数据集**

实验使用17个ID数据集（如ImageNet、CUB-200、EuroSAT、MNIST等）、三种VLM（CLIP、SigLIP2、PE-Core）及其对应的标准OOD集（iNaturalist、SUN、Places、Textures等）。

**📈 对比分析**

与七类零样本OOD检测器（MCM、GL-MCM、MaxLogit、Energy、Mahalanobis、NegLabel、CSP等）对比，使用FPR95和family-balanced FPR95指标，CEG显著降低9–36点误差并将最佳与最差差距压缩到3–4点。

**⚠️ 局限性**

局限性在于CEG仅基于ID统计校准，缺乏严格的覆盖保证；对某些ID域与外部语义覆盖度不足的组合仍易失效；未在动态或多模态部署场景下验证。

---

## 206. EgoSafe: A First-Person Mobile-Captured Benchmark for Visual Safety Understanding

**arXiv ID:** 2607.26518 | [PDF](https://arxiv.org/pdf/2607.26518v1)

**作者:** Yuyun Chen `[一作]` (South China University Of Technology), Ziqian Zeng `[通讯]` (South China University Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EgoSafe-Bench——一个专为第一人称视角安全理解设计的基准，包含 3,000 条高分辨率移动摄像机拍摄的视频和 12,000 条按四层层级推理（HRE）设计的 QA 样本，并对现有大规模视觉-语言模型（LVLM）进行零样本评估。

**💡 创新点**

① 构建了面向安全场景的第一人称数据集；② 设计了层级推理评估框架 HRE，要求模型在盲区推断与因果推理中保持一致；③ 在 11 种噪声变形下系统评估鲁棒性，揭示“感知‑推理脱耦”与过度自信等缺陷。

**🔧 技术方法**

使用 4K 高帧率视频捕获、人工标注与人工审核相结合的 HRE 链接生成，采用 GPT‑5 作为评判器计算 SemSim、KFC、Reasoning Score 和 ECE 等指标；对 Qwen3‑VL、VideoLLaMA‑3、InternVL‑3.5、MiniCPM‑V‑4.5、LLava‑Video 等 LVLM 进行零样本推理。

**📊 数据集**

EgoSafe-Bench 数据集（3,000 片视频、12,000 QA 链）以及 11 种降噪变体；对比基准 UCF‑Crime、RWF‑2000、XD‑Violence、VVR、DVD 等传统安全/暴力检测数据集。

**📈 对比分析**

实验显示：LLM 在 SemSim/KFC 上表现优异，但 Reasoning Score 明显下降（最高 74.60 vs 人类 89.93）；在噪声条件下 Reasoning Score 均下降 30%–45%；MiniCPM‑V‑4.5 ECE 高达 92%，显示严重过度自信；与人类相比，现有 LVLM 在推理层面存在明显劣势。

**⚠️ 局限性**

受限于对盲区推断和因果链条的缺失，模型容易产生幻觉与自信过高；当前训练数据缺少真实安全场景的盲区与复杂噪声，导致模型在极端视觉干扰下鲁棒性差；评估仍主要基于人工标注，难以覆盖更广泛的安全事件和文化差异。

---

## 207. Explicit Kinematic Guidance from Analytic Concepts for Vision-Language-Action Models

**arXiv ID:** 2607.26513 | [PDF](https://arxiv.org/pdf/2607.26513v1)

**作者:** Mingyang Sun `[一作]` (Zhejiang University), Jianhua Sun `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为视觉语言动作模型提供结构化3D先验，构建SAGE框架实现精细化对齐与学习

**💡 创新点**

引入可执行的Analytic Concepts和动态参数追踪，提供稠密奖励与动作约束，显著提升样本效率

**🔧 技术方法**

使用VLM、3D基础模型（VGGT/SAM）、概念专家、对齐损失、运动约束监督和概念导向奖励进行SFT/RL微调

**📊 数据集**

Simulenv、ShapeNet‑Mobility、真实AGILE PiPER机器人及Orbbec摄像头数据集

**📈 对比分析**

与OpenVLA、RoboVLM、SpatialVLA及PPO/GRPO等基线比较，SAGE在模拟和真实任务中成功率提升10‑20%（SAGE‑CQL 71.7%，SAGE‑SFT 69%）

**⚠️ 局限性**

对概念参数估计的依赖、3D感知误差、计算开销及在极端复杂场景下的泛化受限

---

## 208. Evidence-Ledger Adjudication for Claim-Evidence Traceability

**arXiv ID:** 2607.26512 | [PDF](https://arxiv.org/pdf/2607.26512v1)

**作者:** Gengyu Chen `[一作]` (Carnegie Mellon University), Weiling Wang `[通讯]` (Syracuse University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了AI辅助写作中的证据账本裁决流程，将主张与证据包配对，预测支持关系并决定是否需要作者复核。

**💡 创新点**

创新点在于将主张-证据关系显式化为可审计轨迹，并引入“路线”指令让作者即时调整，同时构建跨来源的盲测基准。

**🔧 技术方法**

使用语言模型代理结合规则式关系预测、置信度和简短理由生成，对比无代理基线（always-supported、lexical、TF‑IDF logistic）。

**📊 数据集**

数据集包括AVeriTeC、CLIMATE‑FEVER、SciFact三源共2335条盲测样本（500、1535、300条），涵盖多种证据格式。

**📈 对比分析**

在该盲测上比较四个条件，代理证据账本在关系准确率0.676、macro‑F1 0.601、复核召回0.885上显著优于最佳基线（准确率0.383、macro‑F1 0.303、召回0.723）。

**⚠️ 局限性**

限制：对混合证据的识别仍较弱（F1仅0.289），在CLIMATE‑FEVER上的性能相对较低；依赖预先标注的外部标签，且未评估模型在真实写作场景中的实时交互效果。

---

## 209. Collaborative Weighting with Pessimistic Critic for Mitigating Overestimation in Off-Policy Reinforcement Learning

**arXiv ID:** 2607.26509 | [PDF](https://arxiv.org/pdf/2607.26509v1)

**作者:** Gong Gao `[一作]` (Tongji University), Weidong Zhao `[通讯]` (Tongji University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种名为 CWAC 的协同加权演员-批评家框架，用于解决连续控制任务中离线强化学习的价值过估计与训练不稳定问题。

**💡 创新点**

创新点在于将分布式批评家产生的收益不确定性与 TD 误差通过协同加权机制结合，并通过随机悲观值估计（stochastic pessimistic sampling）抑制过估计，从而实现对不可靠样本的自适应忽略与对可靠样本的强化学习。

**🔧 技术方法**

主要技术包括分布式 Q 网络、Huber 损失、随机悲观值抽样、协同加权（对 TD 误差与不确定性分别赋权），以及在 SAC/Td3/DDPG 等主流离线算法中的无缝集成。

**📊 数据集**

实验使用了 OpenAI Gym（HalfCheetah、Ant、Hopper 等）、PyBullet（HalfCheetahBulletEnv、AntBulletEnv 等）和 DeepMind Control（reacher‑hard、hopper‑hop 等）共 22 个连续控制任务，全部采用统一超参数设置。

**📈 对比分析**

与 VIAC、ALH、LAP、SAC、TD3 等基线对比，CWAC 在所有 22 个任务上均取得平均 15–40% 的性能提升，且在训练过程中的 TD 误差更平滑、收敛更稳定，且与原 SAC 相比仅增加 7% 的计算开销。

**⚠️ 局限性**

局限性主要是随机悲观值抽样在一定程度上引入过度保守性，可能抑制探索；对关键超参数（μ、β_ω、β_ξ）的选择仍需经验，且在极端高噪声或奖励不确定性环境下的鲁棒性尚待进一步验证。

---

## 210. SpatialQ: Understanding 3D Gaussian Splatting Scene Quality via Visual-based MLLM

**arXiv ID:** 2607.26595 | [PDF](https://arxiv.org/pdf/2607.26595v1)

**作者:** Jingxuan Su `[一作]` (Peking University), Wei Gao `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于多模态的3D高斯散射（3DGS）场景质量评估框架SpatialQ，结合3D感知编码器与多模态大型语言模型（MLLM）进行结构化降解推理与分数校正

**💡 创新点**

创新点在于：①通过VGGT+质量头学习跨视角一致的3D空间质量表示；②构建基于MLLM的“根源推理”机制，将深度图、点云渲染与相机参数作为证据进行降解类型与严重度诊断；③将基准分数与推理校正分离，形成可解释的分数修正流程

**🔧 技术方法**

核心技术包括：3D-aware视觉编码器（VGGT）+质量头；跨视角特征融合；Qwen2.5-VL大语言模型的多模态推理；基于降解诊断的分数校正映射函数

**📊 数据集**

使用公开的3DGS-IEval-15K基准（包含15,200帧、10个真实场景、6种压缩方法）以及MipNeRF 360、Tanks & Temples、Deep Blending等子数据集进行评测

**📈 对比分析**

与传统手工IQA、深度学习IQA以及零样本MLLM方法对比，SpatialQ在SRCC、PLCC、KRCC上均取得显著提升，尤其在几何-only与色彩+几何混合场景中领先；实验还展示了推理校正可显著减小绝对误差

**⚠️ 局限性**

主要限制在于：MLLM推理受提示设计与推理稳定性影响；对极其细微或非结构化降解（如Light压缩）仍难以精准区分；模型对不同压缩算法的泛化能力待进一步验证

---

## 211. Unified Shared Memory in OpenMP: Implementation, Programmability, and Performance on Intel Accelerators

**arXiv ID:** 2607.26584 | [PDF](https://arxiv.org/pdf/2607.26584v1)

**作者:** Harald Servat `[一作]` (Intel Corporation), Rakesh Krishnaiyer `[通讯]` (Intel Corporation)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 OpenMP 5.0 的框架下，提出并实现了统一共享内存（USM）功能，使得主机与加速器共享同一地址空间，从而简化数据迁移与一致性维护；随后评估了该功能在 Intel Arc GPU 上对现有 OpenMP 加速器程序的可编程性与性能影响。

**💡 创新点**

首次将 USM 集成到 Intel 的 OpenMP 运行时与驱动栈中，并通过对比显式数据迁移与 USM 的实验，挑战了“USM 只会带来开销”的假设，证明其在部分应用中能实现性能提升。

**🔧 技术方法**

实现依托于 Linux 内核模块（Xe driver）、用户空间驱动、Intel OneAPI 编译器（C/C++/Fortran）以及 OpenMP runtime；实验中使用 Intel Arc Pro B70（Battlemage）GPU、Intel Core Ultra 245K CPU 与 oneAPI Level Zero API。

**📊 数据集**

采用一系列主流 HPC 基准与应用：SPECaccel 2023（403.stencil、404.lbm）、Mantevo（miniFE、miniMD）、LULESH、HipFT、miniGhost、ilbdc、swim 等，覆盖流体动力、分子动力学、有限差分、有限元等领域。

**📈 对比分析**

方法是把每个基准分别以原始显式数据迁移实现、自动 USM（在所有编译单元添加 `#pragma omp requires unified_shared_memory`）和手动 USM（手动改写代码）三种方式编译并跑测，比较执行时间与传输量；结果显示总体几何平均开销约 1.2 倍，经过手动优化后降至 1.03 倍，其中 LULESH 与 HipFT 在 USM 下甚至出现 2.7x、1.6x 的加速。

**⚠️ 局限性**

局限性包括：1）在某些应用（如 miniFE、403.stencil）仍有显著开销，主要源于页面错误与大块传输；2）需要编译器与驱动支持 USM 相关指令，且对大型代码库而言添加指令仍可能繁琐；3）目前仅在 Intel Arc GPU 上验证，缺乏跨平台与更大规模实验；4）部分性能提升依赖手动改写或新驱动优化，未完全标准化。

---

## 212. Recover, Decode, Reguard: Guard-Agnostic Defense Amplification againstEncoded VLM Jailbreaks

**arXiv ID:** 2607.26574 | [PDF](https://arxiv.org/pdf/2607.26574v1)

**作者:** Haoyu Zhang `[一作]`, Shanu Sushmita `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了黑盒视觉–语言模型（VLM）防御中的 decode gap，并提出了一种无关防御器的 recover‑and‑decode 放大器（amplifier）以及可选的 reguard 层来对抗编码和跨模态 jailbreak 攻击。

**💡 创新点**

创新点在于：①提出 guard‑agnostic 的 recover‑and‑decode 放大器，利用目标 VLM 的恢复与解码能力在任何已存在的安全分类器之前重新呈现“真”请求；②引入 reguard 层，对恢复前的文本再次进行安全检测；③使用 AutoAttack 风格的最佳‑套件（best‑of‑suite）评估指标，揭示了传统 per‑attack 平均评估低估攻击者能力的 3.5 倍误差；④发现了非迭代恢复防御在安全–实用性上存在的经验性“安全‑效用天花板”。

**🔧 技术方法**

核心技术包括：VLM 文本恢复（image‑to‑text）、语义解码（恢复编码为普通文本）、可选的 reguard 层（对恢复前文本再次分类）、AutoAttack 风格的全集合评估（ensemble ASR）以及基于 gpt‑5‑mini 的人工裁判对有害与无害输出的标注。

**📊 数据集**

使用的数据集：HarmBench（100 条有害行为）与 OR‑Bench‑Hard（100 条无害行为）来分别评估攻击成功率（ASR）和过度拒绝率；攻防测试共覆盖 11 种编码/跨模态攻击，并在 Qwen2.5‑VL‑7B 与 InternVL3‑8B 两个目标 VLM 上进行。

**📈 对比分析**

对比方法：与未防御、单一防御器、amplifier、amplifier+reguard 的四种配置进行比较。结果显示：amplifier 在多数防御器上能显著降低 per‑attack ASR，但在 11 攻击集合的最佳‑套件 ASR 上仅降低 1–4%；reguard 在所有防御器上进一步降低最佳‑套件 ASR（最高至 48%）但导致无害过度拒绝率飙升至 81–92%，在安全‑效用平面上形成了不可逾越的天花板。

**⚠️ 局限性**

局限性：①实验仅涵盖单视图、非迭代恢复，未考虑隐状态或表示空间的防御；②安全‑效用天花板是经验性结论，可能受恢复/解码提示、阈值设定等因素影响；③仅对已公开的编码/跨模态攻击进行评估，未覆盖像素空间或嵌入空间攻击；④适用的 VLM 与防御器规模有限，可能对更大或更专用模型效果不同；⑤评估中使用的 gpt‑5‑mini 进行裁判，虽然交叉验证了一致性，但仍是人工判定的主观评估。

---

## 213. From Tokens to Watt-hours: Analytical Energy Estimation for LLM Inference on Modern GPUs

**arXiv ID:** 2607.26571 | [PDF](https://arxiv.org/pdf/2607.26571v1)

**作者:** Tina Vartziotis `[一作]` (National Technical University of Athens), Francesca Dominici `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于GPU张量核计算和HBM内存流量的半解析能耗估计器，能够在没有实时功率测量的情况下估算NVIDIA H100上LLM推理的GPU能耗。

**💡 创新点**

创新点在于将参数缩放的Transformer FLOPs与校准的内存流量因子（γ、s_attn、η）相结合，提供了可分解的推理阶段能耗模型，并支持token级能耗估计。

**🔧 技术方法**

使用了Transformer FLOPs计数、HBM数据传输模型、FP16/BF16张量核能耗系数，并通过已有测量结果进行参数校准。

**📊 数据集**

校准过程基于Caravaca等人报告的H100推理能耗数据，模型集覆盖从数亿到百亿参数的多种decoder-only与视语言LLM。

**📈 对比分析**

与测量结果对比，误差在4%–27%之间，能够准确捕捉模型规模线性增长和输出长度超线性内存开销的趋势。

**⚠️ 局限性**

局限于GPU侧能耗，未包含CPU、网络、散热及PUE等系统级耗能，且对量化、MoE或其他注意力变体需重新校准。

---

## 214. Repair as Representational Work: Integration Bottlenecks in AI-Assisted Development

**arXiv ID:** 2607.26517 | [PDF](https://arxiv.org/pdf/2607.26517v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 215. Representation Trajectories Matters: Complementary Evidence for OOD Detection and Image Classification

**arXiv ID:** 2607.26565 | [PDF](https://arxiv.org/pdf/2607.26565v1)

**作者:** Ignacio M. De la Jara `[一作]` (University of Adelaide), Damith Ranasinghe `[通讯]` (University of Adelaide)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了视觉模型在前向传播过程中形成的表示轨迹，并将轨迹视为一种独立的可靠性信号，探索其在 OOD 检测和图像分类（包括平移和自然/合成扰动）中的补充作用。

**💡 创新点**

创新点在于：① 将每个样本在网络中的连续状态连成轨迹，分离类共识运动与样本特定创新；② 设计基于轨迹的 ID‑only 过渡惊讶分数，并与最终状态得分融合；③ 通过线性探针融合更新信息提升分类精度；④ 通过多种对照实验（匹配、洗牌、历史、方向）验证轨迹的持续性与结构性。

**🔧 技术方法**

主要技术包括：表示轨迹建模、残差分解、MLP 过渡预测、标准化与回归融合、线性探针学习与温度加权融合、基于 Mahalanobis++、Relative Mahalanobis、X‑Mahalanobis、kNN 等的 OOD 检测方法。

**📊 数据集**

使用了多种公开数据集：ImageNet‑200/1K、CIFAR‑10/100、MNIST、CIFAR‑100‑C、PACS、Office‑Home、以及 OpenOOD v1.5 的多 OOD 评估集，并在 38 个监督/自监督/多模态/层次/卷积模型上进行实验。

**📈 对比分析**

与传统仅依赖最终状态的 OOD 检测器和分类器比较，轨迹信息在 152 个非饱和基准对中 FPR95 平均下降 4.33 点（Mahalanobis++）至 8.75 点（kNN），提升 131/152 案例；在 72 个分类任务中，精度平均提升 4.41 点，71/72 案例获得正增益；在 CIFAR‑100‑C、PACS、Office‑Home 的平移任务中，平均提升 2.24–3.11 点。

**⚠️ 局限性**

局限性包括：① 轨迹信息的好坏高度依赖于模型架构、预训练方式和数据分布；② 仅在冻结模型下可用，无法在训练过程中动态调整；③ 需要额外的计算和显存开销（约 0.3–0.4 ms 与 35–82 MiB），虽然相对轻量，但在大规模部署时仍需考虑；④ 轨迹仅能补充已有信息，不能生成缺失的分布信息，最终仍需结合最终状态与置信度共同判断。

---

## 216. Prosody-driven Jailbreaks in Audio LLMs: A Controlled Study and Mechanistic Analysis

**arXiv ID:** 2607.26541 | [PDF](https://arxiv.org/pdf/2607.26541v1)

**作者:** Jiachen Qian `[一作]` (City University of Hong Kong), Junyu Li `[通讯]` (City University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本文中，作者通过固定文字稿，仅改变语音的情绪、权威性和语速等发音特征，构建了一个基准（AdvAudio‑Prosody），并提出了PJ‑Break评估流程，来研究语音交付的匹配文本攻击效果。

**💡 创新点**

创新点在于：①首次将语音发音（prosody）作为独立因素来评估音频LLM的安全性；②设计了六种可控的语音预设（Panic、Anger、Commanding、Fast、Whisper、Neutral），并用TTS系统生成；③通过固定查询预算的best‑of‑six策略，客观量化了语音特征对攻击成功率的影响。

**🔧 技术方法**

使用的技术包括：Azure Neural TTS合成、语音属性的声学验证（F0、响度、语速等）、黑盒音频LLM评估（Qwen2‑Audio、GPT‑4o等）、多评审判定（Claude、Llama‑Guard、关键词分类器）以及对比基线（StyleBreak、BoN、AJailBench等）。

**📊 数据集**

数据集主要是AdvAudio‑Prosody（600条样本，100条语义种子×6种语音预设），以及RealSpeech‑20（20条语义种子×3位真人录音）和OTA‑Replay（手机播放实验）。

**📈 对比分析**

在保持相同查询预算（Q=6）的对比实验中，PJ‑Break在Qwen2‑Audio上的成功率为44/95（46.3%），显著高于匹配预算的StyleBreak（27/95，28.4%）和其他基线；同声同语音的5个预设平均能覆盖40/95（42.1%）。此外，情绪语音单独比情绪文字更具攻击效果（44/95 vs 11/95）。

**⚠️ 局限性**

局限性包括：仅在单轮交互和固定查询预算下评估；语音样本主要来自TTS且以英语为主；Commanding预设使用不同声源导致语音身份混淆；RealSpeech‑20和OTA实验样本有限，无法完全验证真实语音迁移；内部诊断仅基于一个开源模型，未能完全揭示机制。

---

## 217. Classification of Disease from Lungs X-ray Images using VGG16, VGG19 and ResNet50 Models

**arXiv ID:** 2607.26580 | [PDF](https://arxiv.org/pdf/2607.26580v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 218. When to Treeify Hash Table Buckets: A Reproducible C Study of List, Hybrid, and Red-Black Tree Chaining

**arXiv ID:** 2607.26530 | [PDF](https://arxiv.org/pdf/2607.26530v1)

**作者:** Georgii Kashintsev `[一作]` `[通讯]` (Rambler&Co), Georgii Kashintsev (Rambler&Co)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估 C 语言实现的链式哈希表中列表、混合树和红黑树分支策略的性能

**💡 创新点**

提供可复现的 API，区分批量与增量树化，并在真实词典倒排索引长度上验证策略

**🔧 技术方法**

使用 FNV 哈希、红黑树、链表以及自定义实验框架

**📊 数据集**

英文词典 370,105 个单词的 9,165 个三元组倒排列表

**📈 对比分析**

通过计数比较、内存使用和插入/查询耗时评估，发现增量树化和全树化在长链条下明显优于列表

**⚠️ 局限性**

仅在高负载下不自动重哈希，且批量树化在中间阶段仍需遍历列表

---

## 219. ContactFlow: A video action conditioning that transfers across embodiments

**arXiv ID:** 2607.26579 | [PDF](https://arxiv.org/pdf/2607.26579v1)

**作者:** Sami Azirar `[一作]` (University of Bonn), Hermann Blum `[通讯]` (University of Bonn)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种与实现无关的接触流动作表示，结合大规模视频生成世界模型，用于机器人规划与验证。

**💡 创新点**

关键创新在于提出只基于物体接触点轨迹的“Contact Flow”，实现了人机共享、跨体态的动作条件。

**🔧 技术方法**

采用大规模视频扩散生成器（DiT + ControlNet/VACE）、Contact Flow提取管道、Vision‑Language Model 进行验证，并结合多模态感知（SAM、HaMeR、FoundationStereo等）。

**📊 数据集**

训练数据来自人类交互视频（TACO、TasteRob、OakInk）、机器人演示（DROID、LIBERO）以及混合数据集。

**📈 对比分析**

在DROID和跨数据集评估中，Contact Flow条件下的模型在DreamSim、SSIM、LPIPS等指标上显著优于对比基线，实测机器人验证成功率达80%（8/10）。

**⚠️ 局限性**

限制包括对高质量外摄深度与标定的依赖、推理速度不适合实时部署以及对接触估计误差敏感。

---

## 220. Eco3S: Complex Socio-Economic System Simulation via Agent-Based Models

**arXiv ID:** 2607.26588 | [PDF](https://arxiv.org/pdf/2607.26588v1)

**作者:** Shaopeng Wei `[一作]` (Guangxi University), Gang Kou `[通讯]` (Southwestern University of Finance and Economics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一个基于大语言模型的经济学社会模拟框架 Eco3S，能够通过共演环境设计、结构因果模拟以及自动化实验循环（SAR）实现从自然语言需求到可执行模拟的全流程。

**💡 创新点**

创新点在于：① 通过双向反馈的共演环境机制，使代理与物理、社会、经济环境同步演化；② 将结构因果模型与 ABM 结合，提供可操作的因果干预和反事实推理；③ 设计 SAR 并行化的自动化实验循环，自动化生成、执行、分析并优化实验，显著降低手工配置成本。

**🔧 技术方法**

技术方面包括：大语言模型（多家供应商的 LLM）驱动的代理决策；双向转换函数 ℱ_a, ℱ_e；结构因果模拟 SCS；记忆摘要机制；多智能体框架与 HIN；并行计算与 O( N × T/P ) 的时间复杂度；SAR 中的四类 LLM 代理协同工作。

**📊 数据集**

使用的“数据集”主要是基于历史案例与实验设计的模拟场景：
- Grand Canal decay 与 rebellion（历史时间序列与空间分布），
- Origins of Governance（河流迁移与国家形成），
- Information Propagation（2×2 factorial 传播策略），
- 以及四类自动化实验（金融羊群、客户忠诚度、资产泡沫、Schelling 隔离）。这些案例提供了已验证的实验数据和空间/时间标注，用于评估模型的重现性和泛化。

**📈 对比分析**

比较方法：在相同情境下与 YuLan‑OneSim、GenSim、Vensim PLE 进行对比；对抗性和基准模拟评估；ABM 与传统宏观模型的重现性对照；对比不同 LLM 提供商的输出；Ablation 研究验证关键模块。性能表现：能够在 10,000 代理、20 步时间范围内稳定运行，误差低于 8% 的重现率；自动化 SAR 在平均 4~5 次迭代内即可收敛；在信息传播实验中对照实验差异与真实实验一致性均超过 90%。

**⚠️ 局限性**

局限性包括：① LLM 的黑盒推理缺乏可解释性；② 物理环境仍以简化的河流/基础设施为主，缺乏城市交通、地形等更细粒度因素；③ 评价与验证依赖历史数据或多重实验设计，缺乏统一的客观指标；④ 对极端情形的鲁棒性评估不足；⑤ 计算资源依赖 LLM 供应商，跨平台性能波动。

---

## 221. 3DGBGS: 3D Granular Ball Gaussian Splatting for Compact Novel View Synthesis

**arXiv ID:** 2607.26578 | [PDF](https://arxiv.org/pdf/2607.26578v1)

**作者:** Meng Yang `[一作]` (Chongqing University of Posts and Telecommunications), YiWang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

基于三维高斯散射的原始点云通过可变尺度的三维颗粒球分割，构建紧凑的锚点布局并以颗粒球半径为尺度先验，形成一种更高效的锚点式高斯散射模型。

**💡 创新点**

创新点在于将颗粒球计算（Granular‑Ball Computing）的自适应粗到细分割机制引入锚点构建与尺度初始化，既减少冗余锚点，又保持对复杂几何与纹理细节的捕捉，显著提升表示紧凑度。

**🔧 技术方法**

主要技术包括：颗粒球分割（基于PCA二分递归）、颗粒球锚点初始化（GBAI）、颗粒球尺度先验（GBSP）、与Scaffold‑GS相同的高斯预测、可微分绘制和动态锚点增长/修剪。

**📊 数据集**

使用了19个真实场景，覆盖四个公开数据集：Mip‑NeRF 360、Tanks&Temples、Deep Blending 和 BungeeNeRF。

**📈 对比分析**

与基准Scaffold‑GS直接对比，3DGBGS在保持类似PSNR/SSIM/L‑PIPS的前提下，初始锚点减少约37.1%、最终锚点减少约10%、模型存储减少约9.8%；在所有数据集上均实现了更紧凑的表示与可接受的渲染质量。

**⚠️ 局限性**

局限性包括：仍需在复杂场景中进一步优化颗粒球分割精度；模型训练与渲染仍相对耗时；对动态场景或非结构化点云的适应性未充分验证；尚未探索可学习的颗粒球形状（如椭圆形）以进一步提升紧凑度与细节保留。

---

## 222. Simultaneous Coverage and Efficiency Guarantee in Online Conformal Prediction

**arXiv ID:** 2607.26577 | [PDF](https://arxiv.org/pdf/2607.26577v1)

**作者:** Rahul Vaze `[一作]` `[通讯]` (Tata Institute of Fundamental Research), Rahul Vaze (Tata Institute of Fundamental Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种统一的在线学习框架，旨在同时控制绝对覆盖违规和预测集效率，以应对动态变化的基准，解决了现有自适应符合推断方法的三大基本局限性。

**💡 创新点**

创新点在于首次提供了同时控制覆盖和效率的保证，特别是在非平稳环境下的在线符合预测中，明确了反馈变得粗糙和目标依赖协变量时性能的退化特征。

**🔧 技术方法**

使用了在线优化技术，特别是投影在线梯度下降和滑动窗口量化跟踪算法。

**📊 数据集**

使用了真实的金融市场数据集，特别是DAX指数的日收盘价格数据，进行波动率预测的实验。

**📈 对比分析**

与现有方法相比，提出的框架在动态基准下提供了更好的覆盖和效率保证，性能表现优于传统的自适应符合推断方法，尤其是在处理动态变化的分布时。

**⚠️ 局限性**

局限性在于该方法依赖于对动态基准的准确建模，且在极端情况下可能会受到影响，此外，算法的复杂性和计算成本在实际应用中可能会增加。

---

## 223. From Spatial Semantics to Temporal Context: Leveraging Gaze Trajectory for Weakly Supervised Medical Image Segmentation

**arXiv ID:** 2607.26542 | [PDF](https://arxiv.org/pdf/2607.26542v1)

**作者:** Shaoxuan Wu `[一作]` (Northwest University), Jun Feng `[通讯]` (Northwest University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于视线轨迹的医学影像分割网络TrailNet；

**💡 创新点**

创新点在于：①显式建模视线轨迹的时序上下文并与图像空间语义进行互补交互；②采用多尺度不确定性解码器通过类别互斥约束提升确定性预测；③利用循环蒸馏策略实现视线无监督推理；

**🔧 技术方法**

主要技术包括：Trajectory‑Guided Spatio‑Temporal Encoder（TSE）、Multi‑Scale Uncertainty Decoder（MUD）、Cycle Distillation Strategy（CDS）以及Mamba状态空间模型、交叉注意力、根均方归一化等；

**📊 数据集**

使用公开的Kvasir‑SEG（结肠息肉分割）和NCI‑ISBI（前列腺T2‑MRI分割）两个数据集；

**📈 对比分析**

与全监督、框盒、点、涂鸦及视线监督的多种弱监督方法对比，TrailNet在两个数据集上分别取得Dice 81.25%与81.85%，明显优于最新SOTA（如GradTrack、GNAN等），提升幅度约1%–2%；

**⚠️ 局限性**

局限性包括：①训练阶段仍需视线轨迹数据；②对轨迹噪声的处理仍不完全；③仅在二维CT/内镜影像上验证，缺乏跨模态通用性验证；

---

## 224. A Physics-Informed Framework for PID Tuning of Chemical Processes Using Large Language Model Agents

**arXiv ID:** 2607.26594 | [PDF](https://arxiv.org/pdf/2607.26594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 225. TPCD: Tone-Pressure Contrastive Decoding and the Label-Free Gating Bottleneck in Vision-Language Models

**arXiv ID:** 2607.26536 | [PDF](https://arxiv.org/pdf/2607.26536v1)

**作者:** Jinkun Zhao `[一作]` (Beihang University), Wenjun Wu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种新的对比解码方法，称为音调压力对比解码（TPCD），用于减轻视觉语言模型（VLM）在高压提示下产生的幻觉现象。

**💡 创新点**

创新点在于将压力诱导的幻觉视为承诺偏差，并通过TPCD方法在推理时利用这种压力响应来减轻幻觉。

**🔧 技术方法**

使用了音调压力对比解码（TPCD）技术，该技术通过在安全中性指令和高压指令下运行同一VLM来计算对比损失。

**📊 数据集**

使用了800个示例的音调重要性基准数据集，涵盖了八个视觉上不确定的类别。

**📈 对比分析**

与安全中性化、ICD、VCD等方法进行比较，TPCD在减轻幻觉方面表现出色，ASR从66.75%降至0.50%，但正向准确率有所下降。

**⚠️ 局限性**

限制在于当前的门控机制尚未经过独立验证，且依赖于表面形式，无法保证在不同压力提示下的普适性。

---

## 226. Shared Symbolic Backbones for Physically Consistent Multi-Output Symbolic Regression

**arXiv ID:** 2607.26528 | [PDF](https://arxiv.org/pdf/2607.26528v1)

**作者:** Manuel Rodriguez `[一作]` `[通讯]` (Universidad Politecnica de Madrid), Manuel Rodriguez (Universidad Politecnica de Madrid)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `14d48e9d-0069-4ad9-996a-1d5968216998` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种用于多输出系统的共享符号骨干神经进化符号回归方法（MO‑SB‑NESR），能够同时搜索一组可重用的符号潜在表达式，并让不同输出通过稀疏加法或乘法读取器选择并组合这些潜在单元；

**💡 创新点**

创新点在于：① 在多输出场景下引入共享骨干结构，实现物理因子一次发现、全局共享；② 采用演化搜索离散结构（运算符、输入掩码、读取器）与梯度下降参数调优相结合的Lamarckian进化框架；③ 通过三臂先验协议（无先验、软种子、硬冻结）可辨别模型发现、可达性与可识别性；④ 结构稀疏机制促使共享自发产生，避免统一混合；

**🔧 技术方法**

使用的技术包括：进化算法（变异、交叉）、梯度下降（Adam）、符号表达式库（如exp、log、sin等）、稀疏正则化、三臂先验协议与结构评估（复杂度、一致性间隙）；

**📊 数据集**

数据集包括多种合成基准（如B1-B7，其中涉及朗缪尔–辛格定律、竞争酶动力学、MeOH/RWGS等），以及真实的水热裂解（HTL）产率实验数据；

**📈 对比分析**

与基线的比较采用预测误差（验证MSE）、模型复杂度、结构恢复度、跨输出一致性间隙等指标。实验表明：在共享因子难以单独辨识的情形下（如朗缪尔分母、MeOH/RWGS站点占有率），共享骨干显著降低一致性间隙并实现更紧凑、可解释的模型；而在每个输出已可识别的系统（如Van de Vusse、竞争酶动力学）时，独立PySR可匹配或优于共享模型；整体来看，性能优势主要体现在结构一致性与解释性，而非纯粹的预测误差提升；

**⚠️ 局限性**

局限性包括：① 需要事先对特征与运算符库进行人工调优，搜索空间扩大易导致计算成本上升；② 对弱可识别结构的发现仍依赖先验或特定实验设计，完全无先验时难以突破；③ 目前主要针对静态数据，动态系统或时序数据尚未扩展；④ 在高维多变量（如完整的反应机理）时，潜在单元数目可能急剧增加，模型复杂度可能失控。

---

## 227. A Design Study on Voice-based Interaction for Immersive Network Visualization and Analysis

**arXiv ID:** 2607.26526 | [PDF](https://arxiv.org/pdf/2607.26526v1)

**作者:** Sam Yu-Te Lee `[一作]` (University of California, Davis), Kwan-Liu Ma `[通讯]` (University of California, Davis)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并评估了一套以语音为主的沉浸式网络可视化系统，支持通过口头命令在VR环境中探索、编辑和分析网络图。

**💡 创新点**

创新点在于：①将语音作为主要交互模式，配合LLM驱动的多阶段命令分类管线实现自然语言到图可视化操作的闭环；②在沉浸式环境中融合少量控制器，实现高效流畅的网络作者操作；③通过技术评估与用户研究揭示语音交互显著提升可用性、流畅性与沉浸感。

**🔧 技术方法**

技术栈包括 Unity XR 与 XR Interaction Toolkit、自定义计算着色器渲染网络、Whisper 语音识别、LangGraph+LLM 的命令分类管线、Neo4j 图数据库、以及多模态交互（语音+控制器）界面。

**📊 数据集**

使用了社交学领域的 bully‑friendship 网络数据集（788 个节点、10,430 条关系），并声明系统可兼容任何结构相似的静态网络。

**📈 对比分析**

对 175 条标注语句进行技术评估：核心波段 100% 通过，扩展波段 89.8% 通过，边界波段澄清率 50%；动作序列准确率 93.6%，Cypher 正确率 99.0%，稳定性 99.2%；模型+数据库平均延迟 1.73 s，含识别约 2.73 s。用户研究显示语音交互在可用性、直观性、以及沉浸感上优于传统控制器，且接受度较高。

**⚠️ 局限性**

局限性包括：①语音识别误差导致命令歧义与误执行；②对高级网络分析（路径、中心性、聚类等）支持不足，容易产生不恰当的查询；③对语义边界的识别不完整；④用户样本有限、缺乏量化任务完成时间的数据；⑤命令可发现性与学习曲线仍需改进；⑥对公共环境下的隐私和声音使用产生顾虑。

---

## 228. Graph k-Coloring in Average Sublinear Time

**arXiv ID:** 2607.26592 | [PDF](https://arxiv.org/pdf/2607.26592v1)

**作者:** Cassandra Marcussen `[一作]` (Harvard University), Shlomo Tauber `[通讯]` (Tel Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了图的k‑着色问题在平均输入分布下的平均时间复杂度，证明当k≤n^c（c≈1/37）时，平均时间复杂度为Θ(nk)，从而打破了长期存在的二次上界，并给出了匹配的下界；

**💡 创新点**

创新点在于：①首次用子线性和局部算法思想结合图正则化理论，构造出可唯一着色的随机子图并进行有效证实；②利用该子图的唯一性实现全图的快速传播，从而将平均时间降到线性阶；③给出最优的Ω(nk)下界，说明该复杂度是不可避免的；

**🔧 技术方法**

主要技术包括：子线性/局部算法框架、图正则化与度/共度数的近似正则性判定、McDiarmid不等式与大偏差分析、以及随机子图的唯一着色证明；

**📊 数据集**

实验/分析所用的数据集为在所有k‑可着色图（节点集为[1…n]）上均匀分布的随机图；

**📈 对比分析**

与之前的Dyer–Frieze（O(n²)）及Kučera（O(n²/k)）等算法相比，本文的算法在常数k时实现了线性时间，整体性能提升显著；并通过匹配下界证明了该结果的最优性；

**⚠️ 局限性**

限制包括：算法仅适用于k≤n^c（c≈1/37）范围；在确定性版本中仍需额外的n因子；对完整k∈[1,n]范围的实现及对其他NP‑难问题的推广尚未完成。

---

## 229. Benchmarking ConvLSTM for One-Day-Ahead IMDAA Rainfall-Field Prediction across Four Indian Cities

**arXiv ID:** 2607.26581 | [PDF](https://arxiv.org/pdf/2607.26581v1)

**作者:** Tanmay Ghosh `[一作]` (National Institute of Advanced Studies), Nithin Nagaraj `[通讯]` (National Institute of Advanced Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

该研究对印度四大城市的IMDAA日常降雨场进行一日预测，比较了ConvLSTM、FC‑LSTM、CNN、ExtraTrees以及基准模型的表现。

**💡 创新点**

创新点在于将降雨历史与大气变量同时作为空间序列输入，系统评估ConvLSTM在小域日常预测中的优势是否成立，并使用可解释AI方法探究模型敏感度。

**🔧 技术方法**

使用卷积长短期记忆网络、全连接LSTM、卷积CNN、岭回归和ExtraTrees等机器学习与深度学习架构，并通过组装置换、时间遮蔽、空间单元遮蔽和Grad‑CAM等方法进行后置解释。

**📊 数据集**

采用印度季风地区重分析IMDAA（1998–2020年6–9月），城市域为9–42格，日降雨与七个大气预测变量作为输入与目标。

**📈 对比分析**

通过域均值RMSE、完整场RMSE、空间异常RMSE以及高降雨天检测指标进行比较，ConvLSTM在孟买的空间异常RMSE最低，但整体性能与FC‑LSTM或简单基准无显著优势。

**⚠️ 局限性**

限制包括仅预测IMDAA降雨而非观测，样本量小、网格分辨率低，模型对高降雨天表现差，未评估更复杂的概率或强度敏感目标。

---

## 230. Semi-Decentralized Multi-Spacecraft Collision Avoidance under Communication Constraints

**arXiv ID:** 2607.26570 | [PDF](https://arxiv.org/pdf/2607.26570v1)

**作者:** Grace Ra Kim `[一作]`, Mykel J. Kochenderfer `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种半去中心化的多航天器碰撞规避框架，在地面站间断通信条件下通过SDec-POMDP模型实现协同决策。

**💡 创新点**

创新点在于将真实的地面站可视窗口作为同步约束直接嵌入决策模型，并证明仅需28.5%较少同步即可恢复中心化性能。

**🔧 技术方法**

采用RS‑SDA*近似递归小步半去中心化A*算法求解SDec-POMDP，并使用非均匀离散化的RTN状态与Gaussian噪声转移模型。

**📊 数据集**

实验基于52个合成LEO接近碰撞场景，使用Brahe轨道仿真和六站KSAT网络生成的可视窗口。

**📈 对比分析**

与中心化、去中心化及手工规则启发式对比，半去中心化方案在保卫碰撞安全的同时，平均返回值与中心化相当，且在轨道偏移上显著低于规则策略。

**⚠️ 局限性**

局限在于未考虑地面通信延迟、非直连互联以及更高阶轨道动力学，且仅聚焦沿轨道冲量燃烧和低轨道情境。

---

## 231. Speech2Grasp: Data-Efficient Transfer of Text-Conditioned Grasp Detection to Speech in Humanoid Robots

**arXiv ID:** 2607.26567 | [PDF](https://arxiv.org/pdf/2607.26567v1)

**作者:** Hung Nguyen `[一作]` (University of California San Diego), Quan Nguyen `[通讯]` (VinMotion)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将基于文本的抓取检测模型 ALBEF 通过轻量级 MLP 投影器迁移到直接接受语音输入的框架 Speech2Grasp。

**💡 创新点**

创新点在于证明仅用少量（≈15K）带有语音的标注数据即可通过知识蒸馏实现文本与语音嵌入空间的高度对齐，并在此基础上实现语音驱动抓取，避免了 ASR 步骤导致的误检和延迟。

**🔧 技术方法**

使用 Whisper 作为语音编码器，MLP 投影器进行模态对齐，ALBEF 作为教师模型，离散知识蒸馏损失；在视觉侧加入 DWT 频域分解以提升噪声鲁棒性；整体采用轻量级 LGD 抓取头。

**📊 数据集**

主要数据集包括 Grasp-Anything++（图像+文本），利用 Kitten-TTS 生成对应语音（约 15K 条），并在实验中扩充为 84K 受扰动语音；真实测试使用 Unitree 人形机器人环境录制的语音命令。

**📈 对比分析**

与传统 ASR+LGD 脚本进行对比：在模拟实验中 Speech2Grasp 的抓取成功率与文本教师相当（S≈0.36, U≈0.33, H≈0.35），而 ASR+LGD 仅 0.34；推理延迟从 102.2 ms 降至 36.6 ms；在真实机器人实验中，Speech2Grasp 的单物体/多物体成功率分别提升约 0.11/0.07。

**⚠️ 局限性**

限制包括：1）需要在训练阶段同时计算教师与学生的前向传播，显著增加显存；2）仅在 ALBEF 这种融合架构上验证，是否适用于其他 VLM 未证实；3）对极端噪声/口音的鲁棒性仍有限，且训练仍需至少数千条标注语音。

---

## 232. ServerlessT2I: Efficient Text-to-Image Workflow Serving on a Serverless Platform

**arXiv ID:** 2607.26566 | [PDF](https://arxiv.org/pdf/2607.26566v1)

**作者:** Xiaoxiao Jiang `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

实现了一个服务器无关的文本到图像（T2I）工作流推理系统，将工作流拆解为模型DAG并在GPU共享内存上无缝执行。

**💡 创新点**

创新点在于通过模型层级缓存、GPU内存虚拟化和基于vTime的公平调度，实现了对稀缺GPU资源的细粒度管理与高效利用。

**🔧 技术方法**

采用了NVSHMEM一侧通信、模型权重量化缓存、CUDA-graph、Python+PyTorch、ZeroMQ调度、分布式执行等技术。

**📊 数据集**

使用了阿里巴巴生产轨迹中的3万多条定制工作流、6个主流扩散模型及其ControlNet/LoRA适配器。

**📈 对比分析**

与基线vLLM-Omni/Diffusers在相同GPU预算下对比，系统可实现2×请求率、3×GPU节省、7×更严格SLO，且SLO达成率≥90%。

**⚠️ 局限性**

局限在于不支持视频生成模型、对非自定义工作流不友好，且在极端高并发或不同硬件环境下的评估尚有限。

---

## 233. AgentGFM: A Graph Foundation Model with Node-Agent Information-Flow Control

**arXiv ID:** 2607.26533 | [PDF](https://arxiv.org/pdf/2607.26533v1)

**作者:** Jingbo Cui `[一作]` (Tianjin University), Dongxiao He `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的图基础模型AgentGFM，能够通过节点级的智能代理实现自适应信息流控制，实现跨图谱的迁移学习。

**💡 创新点**

创新点在于将每个节点视为共享策略的智能代理，在预测-行动-观察-纠正循环中动态决定源接收、信号通道和停止，显著提升在多种拓扑结构下的适应性。

**🔧 技术方法**

使用了基于agent决策的框架、低/高频信道选择、增益感知停止、反馈门控状态纠正、前向-后向的预测-观察一致性自监督目标、straight‑through估计器等技术。

**📊 数据集**

实验涵盖10个十字域1-shot节点分类数据集（Cora、CiteSeer、PubMed、Computers、Photo、Texas、Wisconsin、Cornell、Chameleon、Squirrel）、5个图分类数据集（MUTAG、DD、IMDB-Binary、ENZYMES、PROTEINS）以及3个大规模节点分类数据集（Physics、Ogbn-Products、Ogbn-Arxiv）。

**📈 对比分析**

与传统GNN、无监督预训练方法以及现有图基础模型（SAMGPT、BRIDGE、R-GFM、GCOPE、GraphAny）进行对比，AgentGFM在1-shot节点分类上平均排名第一，在大多数数据集上取得最佳或竞争性表现，在图分类与大规模节点分类任务中同样保持领先或相近的优势。

**⚠️ 局限性**

主要局限是因重复信息流滚动导致额外的计算和内存开销，且在极大图规模下仍不如某些方法最快或最轻量，未来需进一步优化交互和停止机制。

---

## 234. CineWeaver: Training-Free Reference-Controllable Multi-Shot Long Video Generation for Cinematic Storytelling

**arXiv ID:** 2607.26529 | [PDF](https://arxiv.org/pdf/2607.26529v1)

**作者:** Yuyang Huang `[一作]` (Shanghai Jiao Tong University), Xuelong Li `[通讯]` (China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在不进行任何训练的情况下，利用预训练的视频扩散模型，通过在推理时操纵时间编码、注意力模式以及VAE解码，构造出可参考控制、支持多镜头切换与长时序的电影级视频生成框架 CineWeaver。

**💡 创新点**

创新点主要包括：① 通过插入“gap frames”与“transition frames”并对 RoPE 进行动态调节，打破模型对时间连续性的天然偏置，实现多镜头生成；② 采用镜头级别的参考路由（shot‑wise reference routing）在推理时将参考图像精确注入对应镜头；③ 引入 Anchor Memory 机制，在不同独立生成片段之间传递全局外观信息，从而保持长视频的一致性。

**🔧 技术方法**

技术手段：RoPE（旋转位置编码）与自注意力掩码、镜头级跨注意力与前馈网络、独立 VAE 解码、Anchor Memory 的时间不变参考 token、基于预训练模型（Wan2.1‑14B、Phantom‑14B）进行推理。

**📊 数据集**

主要使用自构造的基准：基于 Gemini‑3 生成的多镜头文本提示，Nano Banana 生成的参考图像；测试数据覆盖 200 组多镜头提示、50 组参考控制提示及 15 组长视频提示。

**📈 对比分析**

与多镜头生成方法（CineTrans、EchoShot、HoloCine、MultiShotMaster）以及参考控制方法（Phantom、VACE）比较，CineWeaver 在文本一致性、镜头间一致性、过渡清晰度、叙事连贯性等自动评测指标上均优于对照方法；在人类评估中在过渡清晰度、参考一致性、镜头连贯性及叙事连贯性上占优。

**⚠️ 局限性**

局限性：需人工指定镜头数量、时长及参考图像；对长视频的推理仍受 GPU 计算资源限制；在极长序列或高度动态场景下，Anchor Memory 可能不足以完全消除全局漂移；模型对某些视觉风格的适应性受限于预训练模型的覆盖范围。

---

## 235. AtlasLC: Fast Codec-Ready Compression of Object-Centric 3D Gaussian Splatting

**arXiv ID:** 2607.26525 | [PDF](https://arxiv.org/pdf/2607.26525v1)

**作者:** ByungHyun Kim `[一作]` (KAIST), Woontack Woo `[通讯]` (KAIST)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 AtlasLC，一种源自由、无训练的3D Gaussian Splatting 对象资产压缩管线，专为 XR 可部署场景设计。

**💡 创新点**

创新点在于结合局部竞争裁剪和确定性 Atlas 打包，消除传统映射/重映射的高成本，同时保持几何一致性和前景支持。

**🔧 技术方法**

使用轻量级排序共享坐标、局部竞争裁剪、确定性 Atlas 打包、无训练量化等技术实现压缩与编码。

**📊 数据集**

评估基于 Objaverse 与 Stanford‑ORB 两大公开数据集。

**📈 对比分析**

与 PLAS、LGSCV、UVGS、FlexGaussian 等基线对比，AtlasLC 在相同码率下实现更低的 Payload、总流程时间与解码延迟，并在 3D F1 与 FPS 上优于其他方法。

**⚠️ 局限性**

局限性包括仅适用于静态对象资产、未覆盖动态/可变形场景，且单一 Opacity 评分可能忽略薄透明结构；方法依赖固定标准编解码器。

---

## 236. From Unsupervised Subgroups to Hypothetical State-Intervention Policies: An Evaluation of Selected Subgrouping Methods in Observational Health Data

**arXiv ID:** 2607.26521 | [PDF](https://arxiv.org/pdf/2607.26521v1)

**作者:** Vasundhara Acharya `[一作]` (Rensselaer Polytechnic Institute), Bulent Yener `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文评估了在观测性健康数据中，用无监督子群发现方法构建预算约束的假设状态转移干预策略，并与基于治疗效应的有监督方法做对比。

**💡 创新点**

创新点在于提出了一个以先验因果结构为导向、对分群结果和效益估计做不确定性门控的“表型优先”框架，并通过经验贝塞尔和贝叶斯层级池化门控实现安全性控制。

**🔧 技术方法**

使用技术包括因果发现（PC、FCI、NOTEARS等）、多元对应分析（MCA）处理分类变量、K‑means、FCM、贝叶斯高斯混合模型等聚类，双重稳健（DR）离线评估，经验贝塞尔与贝叶斯门控，以及引导自助法进行不确定性与多重性校正。

**📊 数据集**

采用了两大公开数据集：PIMA Indians Diabetes 数据集（BMI、葡萄糖干预）和 NHANES（吸烟状态与睡眠障碍干预），用于构建和验证干预策略。

**📈 对比分析**

通过在学习集和评估集分离、固定子群结构与排序、并使用重抽样与双重稳健估计来比较不同聚类与有监督方法的政策风险与效用，结果显示所有无监督方法的效用相近，贝叶斯GMM在某些实验中略优，但统计显著性未显著；不确定性（MC与Bootstrap）相对较小。

**⚠️ 局限性**

局限性包括：仅在观测数据下工作，无法验证真实个体效应；依赖先验因果图与固定子群数；处理缺失值采用完整案例，可能导致选择偏差；未加入 NHANES 设计权重；聚类前处理和变量选择对结果的影响未充分评估。

---

## 237. A Graph-Native Bitemporal Memory Store for Conversational AI Agents

**arXiv ID:** 2607.26520 | [PDF](https://arxiv.org/pdf/2607.26520v1)

**作者:** Alp Niksarli `[一作]` (Davidson), Gopesh Baheti `[通讯]` (Davidson)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于Neo4j的双时间点记忆存储，支持会话间持久化和语义检索。

**💡 创新点**

创新点在于将身份节点与版本化内容节点分离，使用HNSW向量索引和双重索引实现即时与历史查询，并在写入时自动构建语义相似边，解决传统向量存储无法区分历史版本的问题。

**🔧 技术方法**

采用Neo4j 5.x（Bolt、HNSW向量索引、B-tree组合索引）、Amazon Titan Embed Text v2 生成1024维向量、Anthropic Claude 用于代理工具调用以及Python驱动实现工具接口。

**📊 数据集**

使用LongMemEval 500问答基准（涵盖六种问答类型）进行评估。

**📈 对比分析**

通过对比当前状态检索（Strategy 2）和时间旅行检索（Strategy 1），在单会话用户声明上R@10达到90%，知识更新上达到80%，时间推理上仅为37.5%；整体R@10为46.7%，显示了在长时记忆检索中的可行性。

**⚠️ 局限性**

限制包括仅索引用户消息导致无法回忆助手生成内容；时间旅行检索因过度获取导致噪声；多会话计数与偏好推断等任务仍需额外事件聚合或重排序；整体检索精度受向量相似度与时间过滤冲突影响。

---

## 238. Semantic-Aware Temporal Adaptation for UAV Anti-UAV Tracking

**arXiv ID:** 2607.26511 | [PDF](https://arxiv.org/pdf/2607.26511v1)

**作者:** Xiaozhen Qiao `[一作]` (University of Science and Technology of China), Xuelong Li `[通讯]` (China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出SATATrack，结合语义引导的时间上下文传播与测试时分布对齐的无人机对无人机跟踪框架。

**💡 创新点**

创新点在于将目标语言描述嵌入到状态空间模型的写入门控与保持矩阵，实现语义感知的时间记忆；以及引入轻量级的TADA做无梯度的分布对齐。

**🔧 技术方法**

采用Fast-iTPN视觉骨干、CLIP文本编码器、SACP模块和TADA，配合语义对比正则化。

**📊 数据集**

使用UAV-Anti-UAV基准以及Anti-UAV318、DUT Anti-UAV、UAV123、UAVDT等数据集进行训练与评测。

**📈 对比分析**

在UAV-Anti-UAV上实现46.4% AUC，显著高于MambaSTS等同类方法；在其它基准上也保持竞争力，表现优于绝大多数纯视觉或视觉语言跟踪器。

**⚠️ 局限性**

局限在于对极小目标、完全离场以及极端模糊的处理仍有限，缺乏长时期重检机制。

---

## 239. Few-Shot Open-Set Audio Classification via Transductive Prototype Refinement and Class Logit Enhancement

**arXiv ID:** 2607.26607 | [PDF](https://arxiv.org/pdf/2607.26607v1)

**作者:** Tianyan Deng `[一作]` (South China University of Technology), Jiahao Du `[通讯]` (South China University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种用于少样本开放集音频分类的转导推理方法ROLE，结合了潜在内点性加权的原型细化和基于自由能的先验自适应拒绝分数；

**💡 创新点**

创新点在于在转导阶段引入查询样本的潜在内点性评分来抑制未知类别对原型的污染，并将分类与拒绝解耦，利用先验调整阈值实现对未知比例的自适应；

**🔧 技术方法**

技术主要包括：冻结的预训练AST音频编码器、两阶段原型细化（基于软赋值和内点性门控）、自由能（negative log-mean-exp）得分、转导损失（支持交叉熵、条件熵最小化与边缘熵最大化）以及温度缩放的余弦相似度；

**📊 数据集**

实验使用ESC‑50、FSD‑Kaggle2018和UrbanSound8K三大音频数据集，分别构造5‑way 1/5‑shot开放集情形，探讨不同未知比例（20%、50%、80%）；

**📈 对比分析**

与八个基线（四个转导、四个归纳）以及AISP相比，ROLE在宏观平均AUROC上取得最高分（1‑shot 85.88%，5‑shot 92.22%），在大多数数据集/未知比例组合下均优于对手，尤其在中高未知比例下表现突出；

**⚠️ 局限性**

局限包括：仅使用冻结编码器，未探索更深层的编码器微调；潜在内点性估计对相似未知声音仍易受干扰；方法对查询集规模和超参敏感，需要进一步鲁棒性验证。

---

## 240. JEPADepth: Masked Predictive Representation Learning for Self-Supervised Monocular Depth Estimation

**arXiv ID:** 2607.26600 | [PDF](https://arxiv.org/pdf/2607.26600v1)

**作者:** Ionuţ Grigore `[一作]`, Călin-Adrian Popa `[通讯]` (Politehnica University of Timișoara)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种自监督单目深度估计框架JEPADepth，将I-JEPA的掩码预测任务与传统光度重建损失结合。

**💡 创新点**

创新点在于在表示空间使用预训练的DINOv3 ViT进行掩码预测，作为辅助训练目标，提升模型对几何结构的理解和跨域泛化，且推理时无需额外网络。

**🔧 技术方法**

使用的技术包括：预训练的DINOv3 Vision Transformer、I-JEPA掩码预测、光度重建与平滑约束、FPN卷积解码器、EMA更新的目标编码器以及自动遮挡掩码。

**📊 数据集**

主要数据集为KITTI（Eigen split），并在Cityscapes与Make3D上做无监督零样本迁移评估。

**📈 对比分析**

与现有自监督方法（Monodepth2、HR-Depth、MonoViT等）相比，JEPADepth在KITTI上取得与Transformer基准相当、优于CNN基准的精度；在Cityscapes和Make3D零样本评估中取得最佳或接近最佳的AbsRel等指标，显示出更强的跨域性能。

**⚠️ 局限性**

主要局限是轻量化卷积解码器可能无法充分利用DINOv3的丰富特征；实验表明简单替换为DPT解码器在自监督场景下并未提升效果，未来需要设计更合适的解码器。

---

## 241. Tournaments determined by three and five voters

**arXiv ID:** 2607.26690 | [PDF](https://arxiv.org/pdf/2607.26690v1)

**作者:** Leonid Chindelevitch `[一作]` (Imperial College London), Ararat Harutyunyan `[通讯]` (University of Paris-Dauphine)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在仅有3或5名选民的投票设置下，如何从投票者的偏好中构造一致的排序，并检验相关的可诱导投票论断与预测性阈值的正确性。

**💡 创新点**

主要创新包括：1) 证明在任何锦标赛中最小回馈弧集必为3环的最小打击集，扩展了Milosz‑Hamel‑Pierrot的定理；2) 构造了3或5名选民的反例，否定了该定理的两项猜想；3) 对3名选民阈值猜想给出完整的反例，表明仅凭预测性不一定能保证可诱导；4) 证明存在仅用5名选民无法诱导的最小锦标赛，并给出Paley(43)的显式例子；5) 通过计数与ILP提升了5名选民可诱导性的上界。

**🔧 技术方法**

采用整数线性规划（ILP）、线性规划（LP）与动态规划相结合的技术，利用自动对称性简化搜索；对Paley锦标赛采用Orbit Subset‑DP与meet‑in‑the‑middle算法求MAS；使用最小覆盖与障碍枚举技术构建完整障碍分类。

**📊 数据集**

主要使用了从McKay档案获取的所有10至13节点锦标赛集，Paley族的43节点锦标赛，以及自定义的自对称锦标赛集合；还使用了完整的3、5选民投票配置集合进行可诱导性检验。

**📈 对比分析**

与以往仅给出非构造性上界的结果相比，本文通过精确枚举和ILP给出了实际可构造的反例（如Paley(43)），并将5名选民可诱导性的最大尺寸从原来的41下界提高到38；对于3名选民阈值猜想，完整地列出了所有违例。

**⚠️ 局限性**

局限性包括：仅对奇数选民3与5给出了实证结果，尚未给出更一般的多选民阈值判定；Paley(43)仅是最小已知的5选民不可诱导例子，尚未构造更小或更一般的反例；计数与ILP方法对大规模锦标赛仍不可扩展，需进一步改进算法。

---

## 242. OpenCSI: Self-Calibration Layer for Heterogeneous Mesh Wireless Sensor Networks

**arXiv ID:** 2607.26665 | [PDF](https://arxiv.org/pdf/2607.26665v1)

**作者:** Karim Khamaisi `[一作]` (University of St. Gallen), Bruno Rodrigues `[通讯]` (University of St. Gallen)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 OpenCSI，一个自校准层，将 WiFi CSI 转化为无量纲 Z-score，支持跨房间、跨硬件、跨 WiFi 代号的零样本二元占用检测；

**💡 创新点**

通过在线学习静默期均值和标准差并使用 Z-score 归一化，抑制芯片与房间特定工件，实现零样本迁移；同时提供可靠性与成熟度监测，以及开放源代码与数据集；

**🔧 技术方法**

使用 ESP32 CSI 解析、硬件抽象层、Welford 累计器、静默期检测、Z-score 归一化、随机森林分类器等技术；

**📊 数据集**

使用三个物理房间（A、B、C）内不同 ESP32 代（S3、C3、C6）采集的 CSI 时序数据，公开数据集与代码托管在 GitHub/Zenodo；

**📈 对比分析**

对比标准会话归一化和跨子载波标准差消融实验，使用随机森林评估二元 F1；在跨房间、跨硬件、跨代迁移中 OpenCSI 取得 F1≥0.98，标准方法仅 0.87-0.87；跨代迁移 F1>0.89；消融实验表明时序标准差是核心；

**⚠️ 局限性**

仅验证单天线 2.4 GHz ESP32，无法区分静态与移动，未验证 5 GHz 或多天线、深度定位、多人数计数等；校准需空房间 120 秒启动，噪声底差导致压缩比率不完全；

---

## 243. Contrastive ESA: Human Evaluation of Multiple Translations at Once

**arXiv ID:** 2607.26640 | [PDF](https://arxiv.org/pdf/2607.26640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 244. Enfold: Folding World-Generator Computation into Predictive Representations for Efficient Embodied Control

**arXiv ID:** 2607.26657 | [PDF](https://arxiv.org/pdf/2607.26657v1)

**作者:** Weili Zeng `[一作]` (Shanghai Jiao Tong University), Yichao Yan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Enfold，利用世界生成模型内部的多层状态作为监督信号，从当前视觉上下文和语言指令中预测一个可用于未来生成和动作控制的表征。

**💡 创新点**

创新点在于：①将生成器的内部计算“内卷”到一个仅基于当前输入的表征；②通过双向耦合（G2R 与 R2G）让该表征既能更好地捕捉未来动力学，又能提升生成器的未来预测；③实现不需要在推理阶段执行生成器的“无滚动”控制，显著降低延迟。

**🔧 技术方法**

核心技术包括：多层生成器（Cosmos‑Predict 2.5 2B）作为教师，当前输入编码器（DINOv3 ViT‑H+/16）作为学生；使用生成器到表征（G2R）和表征到生成器（R2G）的监督；基于 Flow‑Matching 的动作生成器；stop‑gradient 机制保证表征不被任务梯度污染；TensorRT 加速实现 Enfold‑Flash。

**📊 数据集**

使用的公开数据集：LIBERO（40 语言控制任务）、RoboTwin2.0（50 双臂任务，含随机化测试）以及四个真实双臂机器人任务（Fold Towel、Organize Desktop、Spoon Powder、Store Plate）。

**📈 对比分析**

与 Fast‑WAM、Mantis、LingBot‑VA 等现有方法比较：在 LIBERO 上平均成功率 97.8%（与 Fast‑WAM 97.6% 相近）且推理延迟 134 ms，约 3.7× 其速度；在 RoboTwin2.0 上平均 91.77%（Fast‑WAM 91.83%），在真实机器人任务上平均 89.7%（Fast‑WAM 73.3%），展示了显著的速度与鲁棒性提升；Enfold‑Flash 通过 TensorRT 进一步将延迟降至 49 ms，速度提升至 10.1×。

**⚠️ 局限性**

局限性：①仍需在训练阶段使用完整生成器，训练成本高；②在极端分布外或对完整因果推理要求高的场景下性能仍不如顶尖方法（如 LingBot‑VA）；③当前对人类干预的适应性验证仅为定性展示，缺乏量化因果推理指标；④对高噪声或极端遮挡下的生成器状态鲁棒性尚未充分评估。

---

## 245. Graph Is the Verifier: Agentic Reinforcement Learning for Interprocedural Vulnerability Detection

**arXiv ID:** 2607.26656 | [PDF](https://arxiv.org/pdf/2607.26656v1)

**作者:** Yikun Li `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于Code Property Graph（CPG）的Agentic RL框架，用于跨函数（interprocedural）漏洞检测；模型在推理时通过查询CPG获取证据，在训练时利用同一CPG对证据进行精确验证，从而获得可检验的奖励；并通过教师演示的hint-guided轨迹进行SFT预热，随后使用Group Relative Policy Optimization（GRPO）进行强化学习。

**💡 创新点**

创新点在于：①将CPG同时作为工具接口和奖励验证器，实现查询与验证双向耦合；②设计了包含判定、证据、CWE三部分的复合奖励，利用CPG节点ID实现精确的证据匹配；③引入hint-guided教师蒸馏与两阶段训练（SFT+GRPO），证明SFT预热是实现工具使用的必要条件。

**🔧 技术方法**

技术包括：大规模语言模型（Qwen2.5-Coder-7B-Instruct）; CPG构建与查询（Joern/CodeQL）; 反应式工具调用（ReAct）; Group Relative Policy Optimization（GRPO）; 结构化奖励计算与证据验证。

**📊 数据集**

使用PrimeVul数据集（8:1:1 代码仓库级拆分，1,122函数）作为主要训练/测试；外部评测使用TitanVulOOD（648函数、217仓库）；在实验中还进行了类不平衡与out-of-distribution评估。

**📈 对比分析**

与前沿基线（Claude Opus 4.7、GPT-5.5、PrimeVul CoT、VulTrial、JitVul）相比，本文模型在pair-wise-correct（P-C）指标上从0.378提升到对比基线0.273，准确率0.633提升至0.621；在JitVul上提升32.7个百分点；在out-of-distribution和类不平衡场景同样保持优势，F1从0.275提升到0.197。

**⚠️ 局限性**

局限性包括：①对CPG构建的依赖，CPG生成失败或不完整会影响性能；②奖励主要基于教师轨迹，若教师误判或不足，可能导致误导；③模型规模相对较小（7B），对更大规模模型的可扩展性尚未验证；④仅针对C/C++，跨语言泛化需要进一步实验。

---

## 246. Physically Real-time Infrared Attack against Optical Flow Estimation Networks

**arXiv ID:** 2607.26651 | [PDF](https://arxiv.org/pdf/2607.26651v1)

**作者:** Shen You `[一作]` (City University of Hong Kong), Ka-Chun Wong `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种实时红外光物理攻击方法，能够在不改动目标系统的前提下通过在目标物体上投射红外扰动来干扰光流估计网络的输出；

**💡 创新点**

创新点在于：①使用不可见红外灯实现人眼不可察觉的攻击；②通过遗传算法在真实环境中生成训练样本，再训练生成网络实现毫秒级实时攻击；③设计三种针对光流特性的损失函数（不可见性、不可感知性、闪烁损失）以提升攻击效果；

**🔧 技术方法**

技术包括遗传算法（GA）用于离线AEs生成，生成式网络（AGN）用于在线AEs生成，红外光投射硬件，三种自定义损失函数以及光流估计网络（RAFT、PWC‑Net）作为受害模型；

**📊 数据集**

使用自建物理实验环境中的视频序列（不使用公开数据集），通过在轨道上移动目标物体并记录不同光照、速度、距离条件下的光流输出进行评估；

**📈 对比分析**

与基准光流模型（RAFT、PWC‑Net）对比，使用平均不可见度（AIV）和平均不可感知度（AIP）指标；实验表明攻击能够使AIV、AIP保持在90%以上，在多种光照、速度和距离下均表现出色，且相较于无攻击情形下降幅度仅为几百分点；

**⚠️ 局限性**

局限性包括：需要硬件支持（红外灯和控制器）；对极端强光或极快运动的鲁棒性有限；仅针对光流估计任务，未验证对检测、跟踪等下游任务的效果；实验环境受限，缺乏公开数据验证。

---

## 247. The Sparsity Ceiling: Where Spiking Networks Can and Cannot Trade Activity for Energy

**arXiv ID:** 2607.26648 | [PDF](https://arxiv.org/pdf/2607.26648v1)

**作者:** Zeyu Wang `[一作]` `[通讯]` (Georgia Institute of Technology), Zeyu Wang (Georgia Institute of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过匹配架构只更换隐藏层神经元（传统ReLU/tanh vs. LIF脉冲单元），探测不同任务对稀疏性的上限；

**💡 创新点**

创新点在于发现稀疏度上限取决于任务结构而非网络本身，给出信息论的稀疏度下限公式并验证，并区分递归压缩与注意力在神经形态硬件上的能耗代价；

**🔧 技术方法**

采用两侧目标发射率正则化、匹配架构对照、信息论稀疏度下限推导，并通过卷积感知、字符级语言模型和Spiking Transformer实验验证；

**📊 数据集**

使用Rate‑encoded FashionMNIST、WikiText‑103字符级数据集以及合成复制任务；

**📈 对比分析**

在保持网络参数不变的前提下对比CNN/ANN、RNN/ANN、Transformer/ANN，结果显示感知任务可压至5%发射率且准确率无损，递归LM受限于≈50%发射率，Spiking Transformer可压至2%且保持质量；

**⚠️ 局限性**

局限包括感知任务仅使用率编码而非原生事件数据，Transformer仅为单层字符级模型未测KV缓存内存成本，能耗估计基于45nm工艺未在Loihi2/SpiNNaker2上测量，实验规模受限于小N≤8。

---

## 248. Rethinking Self-Evolution: A Constrained Exploration-Exploitation Process for Mitigating Skill Overfitting

**arXiv ID:** 2607.26643 | [PDF](https://arxiv.org/pdf/2607.26643v1)

**作者:** Hongqiang Lin `[一作]` (Zhejiang University), Xipeng Cao `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于约束的探索-利用框架SkillBoost，用以在LLM代理中迭代改进外部技能，避免技能过拟合。

**💡 创新点**

创新点在于将技能自演化视为马尔可夫决策过程，结合结构化错误归因、基于先验的多样化候选生成以及验证接受门控，形成Best‑of‑N探索与回归控制的闭环。

**🔧 技术方法**

使用技术包括：1）技能结构化表示与错误归因（工作流符合度、推理链重构、根因聚类）；2）LLM先验驱动的多策略编辑生成；3）Best‑of‑N候选搜索与验证接受门控；4）理论分析证明候选数对进展的边界；5）成本分层评估。

**📊 数据集**

数据集：SpreadsheetBench、BFCL‑v4、LiveMathematicianBench、ALFWorld、DocVQA；模型：Claude‑opus‑4‑6、Qwen‑3.7‑max、Qwen‑3.6‑plus、Kimi‑k2.6、DeepSeek‑v4‑pro。

**📈 对比分析**

与无技能、人工技能、一次性LLM技能、Trace2Skill、SkillOpt等基线比较，SkillBoost在23个模型‑基准组合中平均提升10.6–28.4个百分点，显著降低测试‑训练差距，达到或超过所有基线的最佳性能。

**⚠️ 局限性**

局限性包括：1）对基础LLM的失败归因、编辑生成与评估能力高度依赖，模型弱时效果受限；2）候选池规模与评估成本存在折中；3）整体评估成本仍以推理 token 为主，部署时需进一步压缩；4）目前仅针对文本与多模态任务，跨领域推广待验证。

---

## 249. Borrowed Strength: Best-of-N Search over a Code EncodingBreaks Self-Check Jailbreak Defenses

**arXiv ID:** 2607.26639 | [PDF](https://arxiv.org/pdf/2607.26639v1)

**作者:** Haoyu Zhang `[一作]` (Northeastern University), Shanu Sushmita `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究将弱攻击（代码编码 + BoN搜索）组合后突破SAGE自检查防御，揭示组合能量的机制。

**💡 创新点**

创新点包括：①说明自检查防御的强度取决于目标模型的拒绝倾向；②发现攻击顺序随防御类型翻转；③提出 probe‑count 解释并给出一行诊断方法。

**🔧 技术方法**

使用技术包括 best‑of‑N 搜索、CodeAttack 编码、SAGE 自检查、统计评估（ASR、QtFS）、人类验证判定与多模型对比实验。

**📊 数据集**

实验数据来源于 HarmBench（100 行为 × 100 抽样）以及 4 个开源指令调优模型（Llama‑3.1‑8B、Qwen2.5‑7B、Gemma‑2‑9B、Llama‑3.3‑70B）。

**📈 对比分析**

通过在每个防御/攻击组合下计算 ASR(N=100) 与 QtFS 进行比较，组合攻击在 SAGE 下实现 67/22/15% 的行为覆盖，远超单一攻击仅 4.7%；在 70B 模型上亦达到 22%，表明防御被突破。

**⚠️ 局限性**

局限性包括：仅测试 8–70B 开源模型；使用单一判定器；未评估闭源/视觉‑语言目标；诊断假设与泛化性待进一步验证。

---

## 250. Filesystem-Based Memory for LLM Agents: Organization, Evolution, and Sustainability

**arXiv ID:** 2607.26637 | [PDF](https://arxiv.org/pdf/2607.26637v1)

**作者:** Sizhe Zhou `[一作]` (University of Illinois Urbana-Champaign), Jiawei Han `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在LLM代理中使用文件系统作为长期记忆的可行性，并在对话与技能两类任务中系统评估了管理、检索与执行角色的表现。

**💡 创新点**

首次将文件系统记忆统一为单一存储类，提出管理/检索/执行三角色分解，并构建了可对比的实验框架；发现组织结构能降低检索成本，但并不直接提升答案质量。

**🔧 技术方法**

使用LLM（gpt‑4o‑like）与行业级内存工具的文件操作、检索工具（BM25、Shell）、基于markdown的存储格式，并实现管理/检索/执行的多模型策略。

**📊 数据集**

采用长会话对话数据集（DialoGPT、OpenBookQA等）以及真实人机对话和家庭任务数据集（OpenAI/Anthropic等）进行评测。

**📈 对比分析**

通过对不同记忆形状（整理、原始、分块）、工具包、模型强度与规模进行交叉实验，衡量回答正确率、检索成本和存储健康；结果表明整理可将检索成本约减50%，但在回答准确率上无明显优势。

**⚠️ 局限性**

局限性包括实验仅覆盖单次会话或短链任务，未检验长期累积的可持续性；不同基准对形状敏感度低，难以捕捉组织对质量的潜在影响；模型与工具的组合可能导致结果不易泛化。

---

## 251. WhisperRec: Latent Reasoning for Efficient Foundation Recommendation Models

**arXiv ID:** 2607.26621 | [PDF](https://arxiv.org/pdf/2607.26621v1)

**作者:** Hao Jiang `[一作]` (Kuaishou Technology), Wenwu Ou `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 WhisperRec，一种在基础推荐模型上实现高效隐式推理的框架；

**💡 创新点**

创新点包括：①多视角自适应 Chain‑of‑Thought（MV‑ACoT）生成多样化、复杂度自适应的推理轨迹；②三阶段隐式推理对齐把教师 CoT 内化为可学习的潜在标记；③基于课程化的后期微调激活潜在推理并保持推荐性能；

**🔧 技术方法**

采用 Qwen3.5‑0.8B 作为骨干，扩展 Semantic ID 词表，使用三层 SID 表示，3 个潜在标记；对 MV‑ACoT 进行多视角监督；训练中使用三阶段对齐、后期课程化微调；

**📊 数据集**

在公开的 Kuaishou LLM‑Rec 基准和工业级 Kuaishou 本地服务（7 天真实交互）两大数据集上进行实验；

**📈 对比分析**

与 ID‑based、SID‑based 传统推荐器以及 OneReason 及其 Think/No‑Think 变体对比；WhisperRec 在 SID@64 上提升 17.44%（对比 Think）且在线推理吞吐量提升约 10 倍，整体推荐效果优于所有基线；

**⚠️ 局限性**

局限性：①仍需依赖大规模预训练 LLM，模型参数和训练成本高；②潜在标记数量有限，可能难以完全捕获极细粒度的推理细节；③多阶段对齐和课程化训练流程复杂，需额外工程成本；

---

## 252. First- and Second-Order Phase Transformation Modeling Based on the Hamilton Principle: A Coupled Thermo-Mechanical Approach for Glass Additive Manufacturing

**arXiv ID:** 2607.26610 | [PDF](https://arxiv.org/pdf/2607.26610v1)

**作者:** Tobias Rudolf `[一作]` (Leibniz University Hanover), Philipp Junker `[通讯]` (Leibniz University Hanover)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

基于扩展Hamilton原理构建了玻璃增材制造过程中热-机械-相变耦合的多物理模型；

**💡 创新点**

创新点在于将第一阶熔融与第二阶玻璃转变统一在变分框架内，兼顾热膨胀、相变体积变化及可塑性变形；

**🔧 技术方法**

采用有限应变乘法分解、温度相关黏弹粘度模型、NEM热传递求解与单步 Backward‑Euler 内部变量积分；

**📊 数据集**

使用文献中给出的玻璃晶体、玻璃态和液态的力学与热学参数作为输入，未涉及实验数据集；

**📈 对比分析**

通过材料点的 TTT 验证与 3D ANSYS FEM 计算，模型能准确再现热梯度下的相变动力学与残余应力，计算效率较传统单独耦合方法提升；

**⚠️ 局限性**

局限性包括参数校准依赖文献，模型对极高黏度差异的数值稳定性要求高，缺乏实验验证，且对非均匀几何形状和多次循环加热的精确度待进一步检验。

---

## 253. FedWeave: Rethinking the Unit of Specialization in Heterogeneous Federated MoE-LoRA

**arXiv ID:** 2607.26618 | [PDF](https://arxiv.org/pdf/2607.26618v1)

**作者:** Donghang Duan `[一作]` (University of Electronic Science and Technology of China), Meng Han `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Federated MoE-LoRA 的异步聚合框架 FedWeave，解决客户端内任务异质性导致的交叉任务干扰与梯度冲突。

**💡 创新点**

创新点在于将专家聚合与路由器优化异步分离，利用无监督原型发现与跨客户端对齐实现专家纯度与路由对比，从而在多任务场景中实现高效稀疏推理。

**🔧 技术方法**

采用 LoRA 参数化、Mixture-of-Experts（MoE）结构、无监督原型发现与聚类、soft routing、稀疏推理以及理论分析等技术。

**📊 数据集**

在包含 CoEdIT、GSM8K、TweetEval、ARC-C 的四任务基准上，使用 Llama3.2‑3B 与 Gemma‑2‑2B 两大模型进行评测。

**📈 对比分析**

与 FedIT、FFA‑LoRA、FedLEASE 等基线对比，FedWeave 在宏平均分数和教师强迫损失上提升约 0.02‑0.03，并在稀疏推理模式下几乎不降低性能。

**⚠️ 局限性**

局限包括模型规模受限（仅测试 2B‑3B 级模型）、对原型可分离性的假设（需明显任务模式差异）、未加入差分隐私或安全聚合机制，以及原型发现阶段的额外计算与通信开销。

---

## 254. Fewer Clarifications, Better Code: Benchmarking Cross-Session Personalized Ambiguity Adaptation in Coding Assistants

**arXiv ID:** 2607.26611 | [PDF](https://arxiv.org/pdf/2607.26611v1)

**作者:** Zijian Xu `[一作]` (Southeast University), Chuhan Shi `[通讯]` (Southeast University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了跨会话个人化歧义适应任务，让编程助手利用同一用户已完成的会话来消解新的歧义请求并最小化澄清步骤。

**💡 创新点**

创新点在于将用户特定的歧义模式视为可迁移的知识，并构建了包含六种歧义机制的可执行编程对话基准。

**🔧 技术方法**

使用大型语言模型（GPT‑5.5、Claude Opus 等）结合外部执行评判器、基于对话的检索和门控机制进行实验。

**📊 数据集**

基准数据集为自研的“Cross‑Session Adaptation to Personalized Ambiguity” (CSPA)，包含 600 个多轮编程会话，分为同一用户历史与 held‑out 评估。

**📈 对比分析**

在可执行成功率、首次成功率与完成轮次三指标上，与无历史基线相比，同一用户历史提升 6–15pp 的首次成功率并减少 0.8 轮完成时间；门控策略进一步提高首次成功率并降低完成轮次。

**⚠️ 局限性**

局限在于仍难以完全解决复杂任务的歧义，且对模型的 prompt 设计和内存检索策略依赖较高，未来需改进更稳健的跨会话记忆与歧义推理框架。

---

## 255. WikiLoop: Jointly Learning to Build and Navigate Agent-Native Wikis with Downstream Feedback

**arXiv ID:** 2607.26604 | [PDF](https://arxiv.org/pdf/2607.26604v1)

**作者:** Haoliang Ming `[一作]` (Tencent Inc.), Wenhui Que `[通讯]` (Tencent Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 256. Visko Orbis 1.0: A Live Model for Real-Time Interactive Long Video Generation

**arXiv ID:** 2607.26694 | [PDF](https://arxiv.org/pdf/2607.26694v1)

**作者:** Xiangbo Gao `[一作]` (Team Visko), Zhengzhong Tu `[通讯]` (Team Visko)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一款实时交互式长视频生成Live Model Orbis，支持文本→视频、图像→视频、视频续写、多语言提示及在生成过程中即时切换提示，能够以4K 24FPS实时输出且用户在生成过程中更新提示时平均可见响应低于1 s。

**💡 创新点**

提出了统一的Live Video框架——基于分块自回归的潜在视频扩散模型，在生成过程中保留可持续的多尺度记忆；利用事件对齐数据和分段训练、少步流化后训练与物理感知奖励对齐、漂移抑制控制、参考感知超分辨率等创新技术，实现了长时序的一致性与实时交互。

**🔧 技术方法**

核心技术包括：潜在视频扩散模型+分块自回归；分层多尺度记忆与递推式历史压缩；少步流化后训练（Guided Distillation + Self‑Forcing DMD）；GRPO式强化学习对视觉、运动与文本对齐奖励进行对齐；物理感知的潜在世界模型用于推理时的运动一致性；参考感知单步超分辨率模型；编译Transformer、全序列多GPU并行推理、进阶解码与流式交付。

**📊 数据集**

使用大规模原始视频集合（涵盖自然、动物、人物、工业等多领域），通过shot‑aware剪辑、分层安全与质量过滤、分布重平衡、人工优先级校正与多阶段事件对齐字幕，构成覆盖丰富、分层质量的训练数据。

**📈 对比分析**

对74个长时序案例（1–3 min）采用DOVER、VideoAlign、HeliosBench、HPSv3等自动指标以及双盲Arena人类评测进行比较。Orbis在DOVER美学与技术评分、VideoAlign视觉/运动质量、HeliosBench美学、以及人类评测中的整体偏好与时间稳定性上均位居榜首，显示出优异的视觉质量与交互一致性。

**⚠️ 局限性**

主要限制包括：对高性能GPU和多机推理架构的依赖，导致部署成本高；长时序生成仍可能出现累计漂移与物理不一致；对极端场景、低光或高速运动的处理尚不成熟；多语言提示的泛化能力需进一步验证。

---

## 257. Scientific Knowledge Discovery in the Age of Large Language Models

**arXiv ID:** 2607.26670 | [PDF](https://arxiv.org/pdf/2607.26670v1)

**作者:** Eleni Adamidi `[一作]` (IMSI, ATHENA RC), Thanasis Vergoulis `[通讯]` (IMSI, ATHENA RC)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对2024-2026年发表的34篇同行评审论文进行系统综述，聚焦生成式LLM在文献检索和筛选中的应用，并对其方法、模型、评估等维度进行分类与总结。

**💡 创新点**

创新点在于首次从检索与筛选两大核心任务两侧系统性梳理LLM技术路径，识别多代理/多模型组合、开放权重模型崛起、以及在生物医学领域的高聚焦使用等共性趋势，为后续研究提供方法学地图。

**🔧 技术方法**

使用系统文献检索、布尔查询、OpenAIRE图数据库索引、论文主题与方法归类、模型与访问方式记录、评估指标对比、以及定量与定性指标的交叉分析。

**📊 数据集**

核心数据源为OpenAIRE图数据库的1,589篇候选文献，筛选后得到34篇；被综述的论文涉及多种生物医学数据集、系统综述数据集、以及公开的检索/筛选基准。

**📈 对比分析**

通过对比所使用的LLM（如GPT-4、Claude、Gemini、开源大模型等）、访问模式（API、本地部署、Web UI）以及评估指标（召回率、精确率、NDCG、F1、AUC等）进行性能汇总；检索部分多使用IR指标但缺乏统一评测；筛选部分多采用诊断准确性工具，结果显示性能提升明显，但各系统间差异仍大。

**⚠️ 局限性**

局限性包括：仅使用单一检索引擎且仅限英文文献，样本时间窗口短，检索评测不足；综述聚焦生物医学领域，跨学科适用性有限；对LLM细粒度适配（微调、提示工程）细节报道不均，难以完整重现。

---

## 258. AgenticCANN: Automated Ascend C Operator Generation via Knowledge-Augmented Agentic Evolution

**arXiv ID:** 2607.26661 | [PDF](https://arxiv.org/pdf/2607.26661v1)

**作者:** Junhao Qiu `[一作]` (City University of Hong Kong), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AgenticCANN 框架，通过知识驱动的生成与阶段性代理进化，实现 Ascend C 低语料库 NPU 环境下的自动算子合成。

**💡 创新点**

创新点在于：①将六层域知识体系与阶段分化组装相结合，突破上游可行性瓶颈；②引入阶段自适应代理调度，兼顾探索与收敛；③在硬件缺失知识的环境中实现高达 90% 的可行率和 6.65× 的速度提升。

**🔧 技术方法**

技术包括：大语言模型（DeepSeek‑V4）、多层知识体系与模式路由、工具使用代理（如 Smolagent）与固定循环修复、沙箱编译验证与硬件性能测评。

**📊 数据集**

使用 54 个 Ascend C 算子（覆盖 5 种模式）以及 14 个 elementwise 算子进行实验，并在 Huawei Ascend 910B NPU 上执行。

**📈 对比分析**

与闭环进化基线（EoH）比较，AgenticCANN 在可行率上从 0% 提升至 90%–100%，在 Pangu 1B 模型上实现 6.65× 的加速；同时在 Token 消耗上比全工具代理模式低约 90%。

**⚠️ 局限性**

局限性包括：广播模式仍无法生成可行代码、对手工知识库的依赖、在极低语料环境下仍需大量人工策划、工具代理在后期易破坏结构导致收敛不佳。

---

## 259. Galvanic Vestibular Stimulation in Latent Space

**arXiv ID:** 2607.26659 | [PDF](https://arxiv.org/pdf/2607.26659v1)

**作者:** Zhi Liu `[一作]` (University of Tsukuba), Yoichi Ochiai `[通讯]` (University of Tsukuba)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

建立了GVS波形与自由文本描述的配对数据集，并开发检索增强的1D‑CNN VAE模型，实现基于文本生成GVS波形，验证其与视觉情景的语义一致性。

**💡 创新点**

首次公开GVS波形‑感受文本配对数据集；将检索增强与变分自编码器结合，完成多标签模态生成，并证明生成波形可与叙事情景实现语义匹配。

**🔧 技术方法**

1D‑CNN VAE、检索增强生成（RAG）、E5‑large‑v2语义嵌入、k‑means聚类、Savitzky‑Golay滤波、信号检测理论评估。

**📊 数据集**

100条随机生成的GVS波形与16名参与者产生的1,526条自由文本描述；另外10名新参与者用于行为验证，构成50对视觉‑波形配对。

**📈 对比分析**

采用参与者置换检验验证波形‑感受关联非随机；Latent空间与波形距离相关性 r=0.945；行为验证中对齐与不对齐配对准确率63.3%（d′=0.70，AUC=0.689），均显著优于随机。

**⚠️ 局限性**

数据集规模和波形库多样性有限，缺乏个体校准与闭环优化，生成波形在跨个体一致性和感知细节上仍存在局限。

---

## 260. Fingerprint-Driven Automation: Coupling Reconnaissance with POC Verification

**arXiv ID:** 2607.26655 | [PDF](https://arxiv.org/pdf/2607.26655v1)

**作者:** Hongping Wang `[一作]` (Hainan University), Xiaoqi Li `[通讯]` (Hainan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个轻量化、全自动的网络安全侦察与漏洞验证工具DeepScan。

**💡 创新点**

通过指纹驱动的闭环管道将侦察与验证无缝连接，实现精准PoC触发与高效扫描。

**🔧 技术方法**

利用Python多线程/异步I/O、nmap、Webeyes、Wappalyzer、geoip2、SQLite、Flask、pyfiglet等技术栈。

**📊 数据集**

在BUUCTF在线CTF平台与Vulhub Docker镜像提供的真实漏洞环境中进行验证。

**📈 对比分析**

在这些环境下与传统扫描工具对比，表现出更低网络开销、更高命中率和更快扫描速度。

**⚠️ 局限性**

受限于Windows异步库支持不足、第三方依赖维护风险及对非公开服务检测覆盖不足。

---

## 261. FakeIDet3-DB: Refining Digital Attacks and Patch Extraction for Secure ID Benchmarking

**arXiv ID:** 2607.26641 | [PDF](https://arxiv.org/pdf/2607.26641v1)

**作者:** Muñoz-Haro Javier `[一作]` (Universidad Autónoma de Madrid), Fierrez Julian `[通讯]` (Universidad Autónoma de Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 FakeIDet3-DB 数据库，包含 250 张真实政府身份证的 6.4K 张数字篡改样本（经典与 GenAI 攻击），并提出了 PACE 伪匿名上下文补丁提取算法，实现安全隐私合规的补丁级评测基准。

**💡 创新点**

创新点包括：① 首个将真实身份证与高级 Generative AI 生成的细粒度攻击结合的公开数据库；② 通过精细化后处理（重绘、羽化、光照校正）提升攻击逼真度；③ 设计了面向隐私的 PACE 算法，利用积分图、距离变换与贪婪 NMS，高效提取贴近遮盖区的高语义密度补丁；④ 在补丁级别上对现有图像取证模型进行评测，揭示全图评估与补丁评估的性能差距。

**🔧 技术方法**

使用的技术包括：生成式模型（LDM、GAN）、经典图像处理（copy‑move、splicing、face‑swap、inpainting）、多尺度积分图和距离变换、距离驱动的贪婪 NMS、语义分割（SAM‑2、Hi‑SAM）以及现有取证模型（TruFor、Re‑MTKD、SparseViT、FakeIDet2）。

**📊 数据集**

数据集为 FakeIDet3-DB：250 张真实身份证，生成 6.4K 张带 8 类攻击（经典 3 类 + GenAI 5 类）的图像，采用伪匿名裁剪后抽取 5.2M 个 64×128 像素补丁，公开 GitHub 链接 https://github.com/BiometricsAI/FakeIDet3-DB。

**📈 对比分析**

通过与 FantasyID 等现有数据库对比，FakeIDet3-DB 在全图检测任务中 EER 从 16.02% 提升至 32.45%（更难），在定位任务中 AUC 仍保持 83.48%（与 FantasyID 相近）。在补丁级别评估时，所有基线模型的 EER 接近 50%，AUC 降至 50% 左右，显著暴露模型对全图上下文依赖的缺陷。

**⚠️ 局限性**

局限性包括：① 受限于 250 张身份证样本，可能无法覆盖所有国家/地区身份证特征；② 补丁级评估需手工裁剪与合并，未实现端到端训练；③ 现有取证模型主要针对全图设计，补丁级性能低，需开发专门的补丁级网络；④ 精细化后处理可能对部分攻击类型产生过度平滑，导致误检率波动。

---

## 262. Guarding Organizations Against Malware Risk: A Novel Graph-Based Malware Detection Method

**arXiv ID:** 2607.26634 | [PDF](https://arxiv.org/pdf/2607.26634v1)

**作者:** Yinan Gao `[一作]`, Xiao Fang `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于程序图的恶意软件检测方法（MalGuard），通过识别基本块的操作角色并学习程序图表示，提升对规避行为的检测能力。

**💡 创新点**

创新点：①使用潜在变量模型自动识别和聚合基本块的操作角色；②通过图粗化、注意力消息传递、门控池化以及层次学习来捕捉角色交互、保留稀疏恶意信号并体现程序图的层次结构。

**🔧 技术方法**

技术手段：变分推理+潜在变量建模、图注意力网络（GAT）、消息传递与注意力聚合、门控池化、层次图卷积、二分类交叉熵训练；利用angr框架提取CFG与CG图。

**📊 数据集**

数据集：手工构建的实时PE恶意软件与正常软件集合，来自VirusShare、VirusTotal、Wiki DLL、NirSoft、CNET，最终1428恶意、7342正常，1:5比例。

**📈 对比分析**

与字节级方法（MalConv、NonNeg、DRSM）和图级方法（GCN、GAT、GIN、MAGIC、DeepCall、Mal2GCN、MalGraph）比较；在F1、AUC、AUPRC三个指标上均显著优于最佳基线，提升约5% F1、3% AUC、4% AUPRC，且在引入规避攻击的测试集上表现更为稳健。

**⚠️ 局限性**

局限性：需手工提取程序图导致计算成本高；对新规避策略的泛化需持续更新；在大规模企业部署时可扩展性和实时性待进一步提升；模型对CFG抽取误差敏感，且未评估跨平台或其他文件格式的适用性。

---

## 263. NELSSA: A GPU-PNM Heterogeneous System for Mixed-Length LLM Serving via Length-based Request Placement

**arXiv ID:** 2607.26633 | [PDF](https://arxiv.org/pdf/2607.26633v1)

**作者:** Sookyung Choi `[一作]` (SK hynix), Jongse Park `[通讯]` (Korea Advanced Institute Of Science And Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个混合长度LLM推理系统NELSSA，将GPU与近内存加速器（PNM）结合，支持短长请求的长度感知调度和动态迁移。

**💡 创新点**

创新点在于：①基于硬件性能导出的长度阈值实现动态请求分配；②一次性迁移机制使长请求在GPU失去内存后无重计算直接在PNM继续推理；③完整端到端系统集成，使用CXL、RDMA实现低延迟跨层内存移动。

**🔧 技术方法**

使用的技术包括GPU FlashAttention、PNM稀疏注意力、KV缓存簇化与头部分区、CXL互连、RDMA、gRPC、RoCEv2、ARM Neoverse V2、NVIDIA H100 NVL等。

**📊 数据集**

使用的数据集有Llama3-8B-1048K模型、Mooncake对话跟踪合成的混合长度工作负载以及RULER基准用于准确性评估。

**📈 对比分析**

与单GPU、双GPU FlashAttention基线及RetroInfer CPU offload进行对比，在混合长度推理下提升至5.5×吞吐量，P99延迟降低15×；在动态增长场景下提升29%，且能处理GPU无法运行的大序列长度请求。

**⚠️ 局限性**

局限性包括：当前PNM设备容量32GB受限，需扩展至512GB；极高选择率或高频长请求仍可能受限；跨节点网络仍是潜在瓶颈；系统对多租户动态负载波动的自适应调度尚未完善。

---

## 264. Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes

**arXiv ID:** 2607.26627 | [PDF](https://arxiv.org/pdf/2607.26627v1)

**作者:** Tianyu Wang `[一作]` (Independent Researcher), Junyuan Shang `[通讯]` (Baidu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析并归纳了“低精度验证（lossy verification）”在投机式解码（speculative decoding）中的分布特性，并将其划分为截断式验证和协作式验证两大范式。

**💡 创新点**

创新点在于：①提出了对低精度验证分布的系统性分类；②揭示截断式验证会引入显著的分布失真，导致性能下降；③发现协作式验证中控制“过度估计”溢出概率（overshoot ceiling）是提升速度与质量平衡的关键。

**🔧 技术方法**

技术手段包括：投机式解码框架、损失精度验证、截断采样（min‑p、η‑sampling）、协作式解码（加权融合、对比解码）、误差分布分析、实验对照与消融研究。

**📊 数据集**

使用的评估数据集有：MATH、MBPP+、INCLUDE、BFCL；模型体系为 Qwen2.5-72B/0.5B、Llama‑3.1‑8B，配合 EAGLE‑3 树结构验证。

**📈 对比分析**

与匹配的截断采样基线（min‑p/η‑sampling）进行对照实验，衡量块效率（BE）、解码速度（DS）与任务准确率；结果表明：截断式验证相较于基线存在 1–8 分的准确率下降；协作式验证通过溢出控制可保持与无失真验证相近的质量，同时提升速度。

**⚠️ 局限性**

局限性包括：实验仅覆盖 Qwen2.5 与 Llama‑3.1 两族模型，未验证在其他架构或规模上的普适性；评估聚焦于结构化任务（推理、代码、语言、多模态），不涵盖开放式生成；理论分析假设了特定截断与协作规则，未来变体可能不完全适用；硬件与服务栈差异会影响实际时间加速。

---

## 265. Understanding Knowledge Transfer Mechanism in Heterogeneous MLLM Fusion: A Simple Linear Approach

**arXiv ID:** 2607.26608 | [PDF](https://arxiv.org/pdf/2607.26608v1)

**作者:** Yinghao Hou `[一作]` (University of Science and Technology of China), Hong Xie `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练的交叉尺度方向参数注入(CDPI)方法，用于分析异构多模态大语言模型的知识转移；

**💡 创新点**

创新点在于通过投影源模型参数方向并线性注入到目标模型，结合二阶理论分析揭示了可转移性选择性以及只在小注入比例下有效的机制；

**🔧 技术方法**

采用参数对齐、注意力头映射、层映射、投影方向注入等技术，并结合本地二阶梯度分析；

**📊 数据集**

使用Qwen3-VL系列模型（2B、4B、8B、32B）以及12个多模态基准（MMMU-Pro、MathVista、MATH-Vision、VisuLogic、VisualPuzzles、MMMU、MMVU、MME、MMStar、BLINK、OCRBench、ChartQA）；

**📈 对比分析**

通过与目标模型的平均分数提升比较，结果显示高层推理能力显著提升，感知能力基本不变；最佳效果出现在注入比例α≈0.02–0.04；

**⚠️ 局限性**

局限性包括只在小注入比例下有效、未在更大比例或不同模型架构上验证、实验仅覆盖Qwen3-VL系列，缺乏跨模型通用性验证。

---

## 266. Harnessing Large Language Models for Intelligent Resource Allocation in the Internet of Everything

**arXiv ID:** 2607.26602 | [PDF](https://arxiv.org/pdf/2607.26602v1)

**作者:** Haijun Zhang `[一作]` (University of Science and Technology Beijing), Yuzheng Ren `[通讯]` (University of Science and Technology Beijing)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了基于大型语言模型（LLM）驱动的任务导向资源调度框架（LLM-PRO），通过语义提示生成、强化学习决策与外部评估闭环实现IoE多任务资源分配。

**💡 创新点**

创新点在于将LLM的语义提示与强化学习融合，形成闭环的提示-评估-反馈机制，显著提升收敛速度与适应性，并引入外部评估模块实时校验约束与奖励。

**🔧 技术方法**

主要技术包括LLM（ChatGPT‑4o）语义提示与奖励塑造、Q‑Learning强化学习、外部评估反馈、边缘节点RL代理以及多层协同架构。

**📊 数据集**

实验使用模拟IoE环境，包含50个终端设备与5个边缘节点，任务按泊松过程生成，节点资源随机化。

**📈 对比分析**

与传统QL、PSO、DQN对比，LLM‑PRO在收敛速度、平均任务延迟与能耗上均优于对照组，收敛更快、延迟与能耗更低。

**⚠️ 局限性**

局限在于LLM仅部署在云端，推理延迟与通信开销；未使用更先进的DRL算法；缺乏真实物理网络实验；对低功耗边缘部署的可行性待验证。

---

## 267. Uncertainty-Guided LLM Semantic Augmentation for Heterogeneous Treatment Effect Estimation

**arXiv ID:** 2607.26599 | [PDF](https://arxiv.org/pdf/2607.26599v1)

**作者:** Jialu Xu `[一作]` (Beihang University), Junjie Wu `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个利用不确定性引导的LLM语义增强插件CURL，用来改进有限样本下的异质性处理效应估计。

**💡 创新点**

创新点在于通过估计器的不确定性来识别需要语义增强的样本，并构造分离的分配和异质性通道，再通过角色感知路由将这些LLM生成的语义表示注入主学习器。

**🔧 技术方法**

主要技术包括MC Dropout估计不确定性、冻结的LLM文本生成与嵌入、双通道语义构造、门控融合、渐进式细化以及测试时基于分位数的阈值决策。

**📊 数据集**

在IHDP、Adult、Jobs和Hillstrom四个基准数据集上进行了评估。

**📈 对比分析**

与十种主机学习器以及LLM原生基线对比，CURL在大多数设置下显著提升了PEHE、AUC等性能，且优于单纯使用LLM或GATE方法。

**⚠️ 局限性**

限制在于仍依赖已知可观测协变量，无法解决未测混杂或重叠违规，且在域漂移下LLM语义可靠性与不确定性校准尚待研究。

---

## 268. Living-Harness Is an Interactive-Agent Evolver

**arXiv ID:** 2607.26598 | [PDF](https://arxiv.org/pdf/2607.26598v1)

**作者:** Yuetian Du `[一作]` (Zhejiang University), Qiang Zhu `[通讯]` (Zhejiang University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Living‑Harness，一种自适应的LLM代理外部工具箱，能将每个完成轨迹及其评估反馈转化为持久的程序化修复，提升后续任务性能。

**💡 创新点**

创新点在于将跨回合的评估证据映射到可检索的程序状态——事件记忆与状态图——并通过固定工具与上下文保持程序边界，避免模型内部单步修正导致的失效循环。

**🔧 技术方法**

核心技术包括：演化SOP（Evolution‑SOP）规则将轨迹抽象为修复证据，程序状态POMDP框架建模跨回合更新，episodic memory与state‑graph双结构存储经验与流程修正，并通过检索-渲染机制在回合中提供上下文。

**📊 数据集**

在两大交互基准上验证：τ²‑Bench（Retail、Airline、Telecom）和MultiWOZ‑2.4（Restaurant、Hotel、Train、Attraction、Taxi）。

**📈 对比分析**

与旗舰模型（Gemini‑3 Pro、GLM‑5、Qwen3‑max、Kimi‑k2）及交互式基线（ReAct、Reflexion、AWM、ReasoningBank、EvoTest）对比，Living‑Harness在τ²‑Bench平均Pass@1提升约10个百分点，在MultiWOZ‑2.4提升约9.9个百分点；且通过多周期演化实现逐步性能累积，并能冻结状态在不同模型上迁移使用。

**⚠️ 局限性**

局限性包括：需手工定义演化SOP的域规则，更新过程受限于评估信号质量；状态图与记忆规模随经验增长可能膨胀，检索与更新成本上升；以及对极端多样化任务的泛化能力仍有待进一步验证。

---

## 269. Decoupled Visual Processing: Efficient Multimodal Adaptation via Modality-Specific Transformer Substitution

**arXiv ID:** 2607.26596 | [PDF](https://arxiv.org/pdf/2607.26596v1)

**作者:** Mingkuan Feng `[一作]` (Tsinghua University), Jianhua Tao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Decoupled Visual Processing（DVP）方案，在预训练的LLM解码器中将视觉token的上层处理拆分，使用单个轻量化Transformer块专门处理视觉token，同时冻结原始LLM参数。

**💡 创新点**

创新点在于极端参数效率：仅更新单个Transformer块而无需微调全模型，验证视觉token不必经过完整解码器深度，并通过拆分-处理-拼接结构实现高效多模态适配。

**🔧 技术方法**

采用了LLaVA框架、CLIP ViT视觉编码器、MLP投影器、单Transformer块、标准语言建模损失、AdamW优化器等技术。

**📊 数据集**

在LLaVA-1.5-665K视觉指令调优数据集上训练，并在MME、POPE、ChartQA三大基准上进行评测。

**📈 对比分析**

与全参数微调的LLaVA-1.5-7B和同架构但全参数训练的Normal Training进行对比；DVP仅训练约3.1%参数，MME感知得分1440（≈96%基线），POPE 0.8619（最高），ChartQA 0.2356，整体性能与全参数模型相当或略优，显著降低训练参数量。

**⚠️ 局限性**

限制包括评测范围有限（仅MME、POPE、ChartQA），未探索不同拆分点、视觉子网络数量、模型规模或多模态任务；拆分后失去后续跨模态交互，可能对细粒度视觉推理产生影响。

---

## 270. Not All Reads Are Conflicts: A Write-Only Analysis of the Sui Blockchain

**arXiv ID:** 2607.26691 | [PDF](https://arxiv.org/pdf/2607.26691v1)

**作者:** Haygen Tsoi `[一作]` (University College London), Philipp Jovanovic `[通讯]` (Mysten Labs)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对Sui主网交易使用写集只冲突模型构建冲突图，评估真实可实现的并行性下限；

**💡 创新点**

提出写集仅冲突模型，消除读侧噪声并提供更精确的并行性边界；同时用并查集对DeepBook等应用进行细粒度划分；

**🔧 技术方法**

采用图论指标（LSP/χ、密度、度协同、团数等）、贪心着色、DFS、并查集、gas加权分析以及区块级写集抽取；

**📊 数据集**

采样Sui 2025年主网checkpoint（约每日100个），利用PostgreSQL索引抽取写集数据；

**📈 对比分析**

与以往读+写冲突分析对比，写集模型下可实现的并行性提高30–40%，DeepBook虽占大部分冲突，但并未产生额外串行瓶颈；经济价值中10–50%仍受顺序路径限制；

**⚠️ 局限性**

限制包括：固定步长采样可能导致时间偏差；贪心算法估计可能偏离真实值；写→读转发不计入；仅基于checkpoint轨迹，无法捕获运行时调度细节。

---

## 271. Upper bounds on the length of quasi-MDS codes

**arXiv ID:** 2607.26684 | [PDF](https://arxiv.org/pdf/2607.26684v1)

**作者:** Umberto Martínez-Peñas `[一作]` (University of Valladolid), Rubén Rodríguez-Ballesteros `[通讯]` (University of Valladolid)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了𝔽_q线性QMDS码在折叠汉明距离下的长度上界，特别是与字段大小q的关系。

**💡 创新点**

通过将QMDS码与子空间族的对应关系，提出了一种从这些族到部分扩展的简化方法，从而引入了有限几何中的尖锐界限。

**🔧 技术方法**

采用几何方法，利用有限几何中的结果，如Drake–Freeman、Năstase–Sissokho和Honold–Kiermaier–Kurz的结果。

**📊 数据集**

使用了𝔽_q的线性码，特别是折叠汉明距离下的线性码。

**📈 对比分析**

通过与已知的部分扩展界限进行比较，恢复了Ball等人的Griesmer型上界，并在多个参数范围内获得了更紧的上界。

**⚠️ 局限性**

限制在于只考虑了特定的参数范围，可能无法涵盖所有情况，特别是在r与k的关系上。

---

## 272. Anchoring and Steering Diffusion: Enhancing the Faithfulness of Text-to-Image Generation at Inference Time

**arXiv ID:** 2607.26647 | [PDF](https://arxiv.org/pdf/2607.26647v1)

**作者:** Xinyi Wang `[一作]` (Shanghai Jiao Tong University), Wenxian Yu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AnchorSteer，一个训练无关的框架，通过细粒度控制初始噪声和去噪轨迹，显著提升文本与图像在复杂组合提示下的语义一致性。

**💡 创新点**

创新点在于：① 利用 CLIP 的语义先验与 Latent‑Prior Score Distillation Sampling (LP‑SDS) 构造与文本对齐且符合分布的初始噪声；② 引入 Reflective Steering，包含 Think–Erase–Retouch 三阶段循环，实时诊断并纠正去噪过程中的语义漂移；③ 这两者结合实现了无训练、全流程可解释的对齐提升。

**🔧 技术方法**

核心技术包括 CLIP 语义提取、DAS（Direct Ascent Synthesis）、LP‑SDS（Score Distillation Sampling 的变体）、DDIM/DDPM 前向/逆向过程、VLM（如 Qwen‑VL‑Chat）进行语义诊断、CFG 指导与双向纠错（Erase/Retouch）。

**📊 数据集**

在 GenEval 与 T2I‑CompBench++ 两大基准上进行评测，使用 SDXL 与 HunyuanDiT 两种不同架构的扩散模型。

**📈 对比分析**

与多种基线（AaE、InitNO、DNO、Zigzag、Diffusion‑DPO、SPO、NPNet 等）比较，AnchorSteer 在 GenEval 上整体得分提升约 7.5 分（从 55.93 到 63.39），在 T2I‑CompBench++ 上多数语义子指标均名列第一；同时保持甚至提升了图像质量与多样性指标。

**⚠️ 局限性**

主要限制是推理时的计算开销显著增加，尤其 Reflective Steering 的每一步循环导致每张图平均 68.5 秒的生成时间；若需更快速度，需要进一步优化或压缩模型。

---

## 273. Genie Sim PanoWorld: An Infinite Indoor 3D World Generation Pipeline via Panoramic Scene Modeling and Simulation

**arXiv ID:** 2607.26646 | [PDF](https://arxiv.org/pdf/2607.26646v1)

**作者:** Yongxin Su `[一作]`, Maoqing Yao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本工作提出了一条完整的无优化、单视角全景图到可自由漫游的3D高斯场景的端到端流程，先通过可控轨迹生成全景视频，再用无优化的全景重建器将视频映射为实时可渲染的3D Gaussian 模型。

**💡 创新点**

创新点主要包括：① 通过 NavMesh 规划的 SE(3) 轨迹作为显式条件注入潜在视频扩散模型，实现轨迹可控的全景视频生成；② 引入长短轨迹混合训练与自一致性 shortcut 模型，四步无 CFG 采样即可获得高质量视频；③ 设计了 voxel‑aligned Gaussian 解码器，利用多视角与 KNN attention 将视频帧直接升维为高质量的 3D Gaussian 表示。

**🔧 技术方法**

核心技术包含：潜在视频扩散模型（Wan2.2‑TI2V‑5B + LoRA）、shortcut self‑consistency 训练、Depth Anything 360 作为深度估计、3D VAE、NavMesh 轨迹规划、PanoVGGT Transformer 进行全景姿态与深度估计、voxel‑aligned Gaussian head、KNN 与多视角注意力、RealESRGAN 上采样，以及 3D Gaussian Splatting 渲染。

**📊 数据集**

训练与评测主要使用：RealSee3D、InteriorGS、Structured3D、以及作者自制的 Infinigen‑Indoor 360 全景数据集；在 InteriorGS 测试集、RealSee3D 零射击集上进行评测，并与 Matrix‑3D、OmniRoam、PanoWorld 等基线进行对比。

**📈 对比分析**

采用帧级 PSNR/SSIM/LPIPS/WS‑PSNR、视频级 FVD/FID、轨迹误差 ATE/RRE/RTE 等指标进行比较；四步无 CFG 生成在 PSNR 23.5–23.9、Fidelity 细节上优于 10 步的 Matrix‑3D；在 3D 重建上 PSNR 26.7（context）/23.9（novel）显著高于 PanoWorld、HY‑WM2.0 等基线；整体推理仅需约 147 s 在 RTX4090 上完成全流程。

**⚠️ 局限性**

主要局限：假设场景静态，无法处理移动物体导致的时间不一致；依赖单视角深度估计，深度误差会影响 NavMesh 与轨迹规划，导致潜在的墙壁穿透；当轨迹步长过大时，几何‑warp 条件失效导致失真；目前仅在室内数据集上验证，未评估户外或大规模场景。

---

## 274. FPSGen: Flexible Point Cloud Scene Generation with BEV-Supported Transport Flows

**arXiv ID:** 2607.26645 | [PDF](https://arxiv.org/pdf/2607.26645v1)

**作者:** Wenzhe He `[一作]` (Hunan University), Ruihui Li `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种名为 FPSGen 的两阶段点云生成框架，先在鸟瞰图上预测稠密分布、最大高度和占据掩码作为场景先验，再从该先验中采样点源并通过教师‑学生近似最优传输模型进行点级流式生成，从而实现无扫描或扫描条件下的户外 LiDAR 场景补全与生成。

**💡 创新点**

创新点在于：① 用 BEV 先验独立于扫描构建点源，消除训练‑推理不匹配；② 采用教师估计的源‑目标端点实现近似最优传输，避免大规模点集匹配；③ 通过贝叶斯自由条件化实现同时支持无条件、道路/车辆/LiDAR 条件生成。

**🔧 技术方法**

技术手段包括：条件流匹配（flow matching）与时间条件化网络；BEV 生成网络（U‑Net + Transformer）预测密度/高度/掩码；BEV 源采样器将密度映射为点源；教师‑学生结构对齐源到目标；分类器无监督引导（CFG）；多尺度稀疏 U‑Net 进行点级速度预测。

**📊 数据集**

在 SemanticKITTI（训练 00–07、09–10；评测 08）和 KITTI‑360（评测 00）两个公开无人驾驶 LiDAR 数据集上进行实验，使用完整点云作为监督。

**📈 对比分析**

与 LiDiff、LiDPM、ScoreLiDAR、Distillation‑DPO 等基线比较，FPSGen 在 CD、JSD、IoU 等补全指标上均达到或超过最优；在无条件生成任务中在 COV、EMD、1‑NNA 等分布指标上亦表现最佳；并且单步点流即可获得与多步相近质量，推理速度快、显存占用低。

**⚠️ 局限性**

局限性包括：仍需完整点云监督以构建 BEV 先验；生成受限于单帧静态场景，未处理时序；近似最优传输依赖教师网络，未实现真正全局最优匹配；BEV 先验分辨率有限，细节层次可能不足。

---

## 275. Nix to the Rescue for a Reproducible HPC-AI Software Stack

**arXiv ID:** 2607.26688 | [PDF](https://arxiv.org/pdf/2607.26688v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 276. Cut Query Reachability for DAGs with Subquadratic Queries

**arXiv ID:** 2607.26630 | [PDF](https://arxiv.org/pdf/2607.26630v1)

**作者:** Ben Bals `[一作]` (CWI), Yasamin Nazari `[通讯]` (Vrije Universiteit)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在切割查询模型中，研究了有向图的单源可达性问题，提出了一种确定性算法，使用O(n√(n log n))次查询来判断从源点s到目标点t的可达性。

**💡 创新点**

创新点在于针对有向无环图（DAG）提出了一种新的算法，能够在低于二次查询复杂度的情况下解决单源可达性问题，并且可以扩展到计算单源最短路径。

**🔧 技术方法**

使用了拓扑排序算法，并在此基础上进行了改进，以适应切割查询模型的需求。

**📊 数据集**

研究中使用的图是有向无环图（DAG），具体数据集未明确提及。

**📈 对比分析**

与现有的随机算法相比，提出的算法在查询复杂度上有显著改进，能够在O(n√(n log n))的查询次数内完成任务，优于重建整个图所需的O(n^2)查询。

**⚠️ 局限性**

限制在于该算法主要针对有向无环图，对于一般有向图的可达性问题仍然存在挑战，特别是在确定强连通分量方面。

---

## 277. Understanding Context Sampling in TabPFN on Small Tabular Datasets

**arXiv ID:** 2607.26628 | [PDF](https://arxiv.org/pdf/2607.26628v1)

**作者:** Mohammed Abdullah `[一作]` `[通讯]` (Anna University), Mohammed Abdullah (Anna University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了TabPFN在小型表格数据集上上下文样本选择的影响，探讨上下文大小、代表性与多样性对模型性能的作用，并比较了随机抽样、K-Means与Farthest-Point Sampling的效果。

**💡 创新点**

创新点在于证明上下文多样性而非分布匹配是决定TabPFN准确性的关键因素，展示随机抽样即可提供足够多样性，同时对不同选择方法的成本效益进行实证评估。

**🔧 技术方法**

使用了TabPFN v3、随机抽样、K-Means、Farthest-Point Sampling、受控构造实验和混合效应模型等技术。

**📊 数据集**

采用15个OpenML公开小型表格数据集（行数约200–5000），覆盖多类别与二分类任务。

**📈 对比分析**

在k=128时，随机抽样、K-Means和FPS的AUC差异小于1个百分点，但随机抽样的选择时间低至0.0003 s，远低于K-Means（0.22 s）和FPS（0.04 s），表明额外成本几乎无性能提升。

**⚠️ 局限性**

局限性包括仅评估小型表格数据、单一模型（TabPFN v3）、有限的选择方法（仅随机、K-Means、FPS）以及对多样性度量的几何近似，结果可能不适用于大规模或非表格数据。

---

## 278. Enhancing Automated Machine Learning via Homogeneous Train-Test Splitting Methods

**arXiv ID:** 2607.26625 | [PDF](https://arxiv.org/pdf/2607.26625v1)

**作者:** Yearn Tan Yin Tze `[一作]` (University of Sheffield), Charles Grellois `[通讯]` (University of Sheffield)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究并比较了五种传统和一种改进的训练-测试划分方法在自动机器学习中的分布相似性与模型评估可靠性。

**💡 创新点**

创新点是提出Optimised-Distribution方法，将分布相似性作为显式优化目标，并在多数据集上实现最高MMD相似度。

**🔧 技术方法**

采用统计相似性检验（卡方、KS、MMD）与PyCaret自动ML框架，结合Kennard–Stone、SPXY、Duplex等空间填充算法。

**📊 数据集**

使用了十五个UCI公开分类数据集，涵盖从几百到二十五万样本、4-169维的混合特征。

**📈 对比分析**

通过比较相似度得分和下游模型准确率/加权F1，发现Optimised-Distribution与随机/分层划分在高相似度下性能相近，空间填充方法在某些难题上表现更差。

**⚠️ 局限性**

限制包括仅使用单一随机种子、仅评估分类任务、未充分考虑特征冗余与有效信息数、以及对大规模数据的计算成本。

---

## 279. Active Movable-Element RIS Assisted Vehicular Semantic Communications: Modeling and Optimization

**arXiv ID:** 2607.26658 | [PDF](https://arxiv.org/pdf/2607.26658v1)

**作者:** Maoxin Ji `[一作]` (Jiangnan University), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种行级可移动主动可重构智能表面（RM‑A‑RIS）辅助的车载语义通信系统，并联合优化 RIS 元素位置、主动反射系数和语义符号长度以最大化语义频谱效率。

**💡 创新点**

创新点在于将主动放大与可移动 RIS 元素相结合，利用位置调节重构信道几何并补偿乘法衰落，同时提出两时隙协同优化框架和基于投影梯度与 SCA 的高效算法。

**🔧 技术方法**

使用了主动 RIS、可移动 RIS、语义通信框架、三次样条插值、SCA、二次变换、投影梯度（PGA）、PAVA 及贪心搜索等技术。

**📊 数据集**

通过 DeepSC 预训练模型生成的语义相似度查找表（基于 Sentence‑BERT 语料库）进行评估，并在仿真车道场景中随机生成车辆位置与参数。

**📈 对比分析**

与被动 RIS、固定位置主动 RIS、QPSO 等基线比较，实验证明 RM‑A‑RIS 在 Sum‑SSE 上提升 132.9%（对被动 RIS）和 9.2%（对固定位置主动 RIS），平均 SINR 提升 11.27 dB。

**⚠️ 局限性**

主要局限包括机械移动速度与精度限制、对实时 CSI 的高度依赖、非凸优化可能陷入局部最优以及在大规模车辆密度下的计算复杂度挑战。

---

## 280. AIGen: Automating AI Bill of Materials Generation Through Hybrid MLOps Integration

**arXiv ID:** 2607.26652 | [PDF](https://arxiv.org/pdf/2607.26652v1)

**作者:** Federica Pepe `[一作]` (University of Sannio), Massimiliano Di Penta `[通讯]` (University of Sannio)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为AIGen的模块化工具，用于自动生成符合SPDX 3.0 AI Bill of Materials（AIBoM）的机器可读文档，结合MLflow的结构化元数据提取与大语言模型（LLM）的自然语言合成；

**💡 创新点**

核心创新在于将MLOps平台的直接数据抽取与LLM生成的文本信息通过插件化、Builder/Team/Goal架构实现无缝集成，既支持多种AI框架（Hugging Face、PyTorch、TensorFlow）又保持可扩展性，并提供可验证的合规输出；

**🔧 技术方法**

技术主要包括Java实现的PipeManager、Builder/Team/Goal插件体系、LLMClient接口（可绑定Ollama或OpenAI），SPDX 3.0 Serializer与验证器，以及多阶段流水线（配置、检索、序列化、验证）；

**📊 数据集**

在评估阶段使用了八个GitHub/Kaggle开源机器学习项目，覆盖计算机视觉（COVID‑xray）、表格预测（AQI）、推荐系统等，利用MLflow记录指标和工件，并通过Ollama部署的DeepSeek‑r1 8B LLM进行文本生成；

**📈 对比分析**

通过四分制定性评估（Perfect、Good、Lackluster、Bad）对AIBoM字段进行质量打分，结果显示在结构化字段上91%为Perfect或Good，LLM生成的描述性字段则出现部分幻觉和上下文丢失，但整体性能优于手工编写；

**⚠️ 局限性**

局限性包括LLM的幻觉风险、滑动窗口策略导致的上下文缺失、对域特定信息依赖度高（如模型局限性字段需额外文档支持），以及在缺失元数据时仍需人工干预来完善文档。

---

## 281. AlphaSchema: Exploring the Space of Trading Semantics for LLM-Based Alpha Mining

**arXiv ID:** 2607.26642 | [PDF](https://arxiv.org/pdf/2607.26642v1)

**作者:** Jingyang Yi `[一作]` (X-Tech), Jian Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种将 Alpha 矿业从代码实现搜索转向语义计划搜索的框架，使用结构化的交易语义计划（事件、上下文、质量、方向、输出）生成候选因子，通过 LLM 将计划转化为可执行代码并评估，再利用奖励回馈学习语义空间的代理模型，以此引导后续搜索。

**💡 创新点**

创新点在于：① 明确划分搜索与实现，构建可解释的语义空间；② 采用奖励导向的分配式计划选择器，平衡全局探索、代理引导的利用和局部突变；③ 通过计划层面的奖励建模，学习可重用的先验，使得搜索更具可控性与可扩展性；④ 证明不同 LLM 在实现成功率上差异较大，但因子质量基本相同，提升了系统的鲁棒性。

**🔧 技术方法**

核心技术包括：大语言模型（DeepSeek‑V4‑Flash 等）做代码生成；LightGBM 训练语义奖励代理；基于分配式预算的迭代搜索算法；执行与回测的安全检查；PCA、K‑means 等分析手段。

**📊 数据集**

使用中国 A 股 CSI300 交易 Universe，训练集 2016‑2020，验证集 2021‑2022，测试集 2023‑2025；同时在 CSI500 上做独立回测验证。

**📈 对比分析**

与传统机器学习预测器（MLP、XGBoost）、深度序列模型（Transformer、GRU、LSTM）、公开因子库（Alpha158、Alpha360）以及现有代理式挖掘系统（RD‑Agent、QuantaAlpha）对比。实验显示，语义搜索产生的因子池在 IC、ICIR、RankIC 等预测指标上均优于对照组，在 IR、AER 等组合性能上也取得最高或次高记录，净值曲线表现稳健增长。

**⚠️ 局限性**

局限性包括：① 搜索空间受预定义词表限制，难以覆盖所有潜在交易语义；② 需要多次实现评估以减小噪声，导致计算成本高；③ 对 LLM 的实现成功率仍敏感，若模型表现差会影响搜索效率；④ 主要在中国市场验证，跨市场泛化尚待验证；⑤ 未对计划间的兼容性进行正式约束，可能出现逻辑冲突。

---

## 282. RAG-HAR+: Towards Cost-Efficient LLM-Based Human Activity Recognition for Edge Deployment

**arXiv ID:** 2607.26631 | [PDF](https://arxiv.org/pdf/2607.26631v1)

**作者:** Hansi Karunarathna `[一作]` (University of Sydney), Kanchana Thilakarathna `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 RAG‑HAR+：一种基于检索增强生成（RAG）的、训练无关且成本可控的运动识别框架，将大模型的使用限制在离线特征组设计和仅在检索不确定时的在线推理。

**💡 创新点**

创新点包括：①离线 Retrieval Designer Agent 利用 LLM 自动为每个数据集挑选并分组 3 组多维特征，从而构建更具区分性的检索表示；②在线阶段使用多数投票进行检索直接分类，仅当投票不确定才调用 LLM 作为 Ambiguity Resolver，从而把推理成本与样本数解耦；③在移动端实现原型，展示了检索优先、LLM 辅助能在边缘设备上实现低延迟、低费用的实时运动识别。

**🔧 技术方法**

技术包括：多向量检索（多特征组向量化+加权相似度）、LLM Prompt Engineering（离线特征组设计和在线不确定性推理）、统计/时间/频谱/形状特征提取、近似最近邻索引、基于多数投票的决策路由、移动端 Flutter+Milvus 部署。

**📊 数据集**

在六个公开 HAR 基准上验证：USC‑HAD、PAMAP2、MHEALTH、GOTOV、HHAR、Skoda。

**📈 对比分析**

与原始 RAG‑HAR、最强非 LLM 和 LLM 相关基线对比：RAG‑HAR+ 在四个数据集上提升或保持性能，且在线 LLM 调用量下降 89.3%–99.9%，平均延迟降低 5–45 倍；在移动原型中实现了约 15 倍速度提升，宏 F1 与 RAG‑HAR 差距 ≤1%。

**⚠️ 局限性**

局限性包括：仍需离线 LLM 设计阶段（一次性成本），在极端多通道或非常动态活动下检索仍可能产生模糊邻居导致 LLM 调用增多；目前只采用多数投票路由，未探索更细粒度的不确定性度量；对隐私与安全的影响尚未深入评估。

---

## 283. Constitutional Midtraining: Content Presence Drives Alignment Gains

**arXiv ID:** 2607.26654 | [PDF](https://arxiv.org/pdf/2607.26654v1)

**作者:** Desiree Cho `[一作]` (University of Oxford), Nigel Shadbolt `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在120B模型的中训练阶段插入基于Anthropic Constitution的约260-500M token合成内容，并评估其对对齐性、跨分布泛化及在后续SFT与benign fine‑tune中的耐久性。

**💡 创新点**

首次在大规模（120B）中训练阶段进行宪法式对齐干预，验证其对跨分布泛化和耐久对齐的显著提升，并系统探索课程顺序与推理结构的二级影响。

**🔧 技术方法**

采用2×2因子设计（课程顺序×推理）合成对齐语料，混合重放预训练数据，随后进行无偏见SFT与benign fine‑tune；评估使用GPT‑4o判定、ACR指标及多种对齐/能力基准。

**📊 数据集**

使用Anthropic Constitution提取的40条价值观生成257.6M token的DR（含推理）与136.8M token的noDR语料，配合200K示例SFT混合，评估基准包括Blackmail、Emergent Misalignment、MASK、MMLU、ARC‑Easy、piqa、GSM8K。

**📈 对比分析**

在post‑MT、post‑SFT、post‑BFT三阶段对照CMT与对照组，CMT在ID/OOD安全问答、Blackmail等指标均显著优于对照，优势在benign fine‑tune后仍保持；在能力基准上无显著下滑，偶有轻微提升。

**⚠️ 局限性**

限制在于仅使用混合Mamba‑attention架构与单一英镑宪法，未检验更大规模或DPO后效；课程与推理结构的二级效应不显著，可能受数据/评测偏差或模型内部机制差异影响。

---

## 284. Efficient Heteroscedastic Bayesian Optimization for Risk-Aware AutoRL

**arXiv ID:** 2607.26680 | [PDF](https://arxiv.org/pdf/2607.26680v1)

**作者:** Mingxuan Che `[一作]` (Leibniz University Hannover), Alexander von Rohr `[通讯]` (University of Technology Nuremberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自适应采样的风险敏感贝叶斯优化方法（ERAHBO），用于在强化学习中高效地寻找既能提升平均收益又能降低训练波动的超参数配置。

**💡 创新点**

创新点在于将RHA​BO的固定复测预算替换为基于置信界的自适应复测策略，使采样资源自动聚焦于不确定性高、潜在优异的配置，从而显著提升样本效率。

**🔧 技术方法**

采用了异方差高斯过程建模平均回报和方差的贝叶斯优化框架，配合基于置信区间的停止规则和上界/下界采样策略，并提供了子线性累计回退保证。

**📊 数据集**

使用了在ARLBench平台收集的离线强化学习实验数据，包括DQN、SAC和PPO在Brax、XLand‑Minigrid以及Classic Control环境下的512个配置与50个随机种子结果。

**📈 对比分析**

与风险中性GP‑UCB和固定复测的RAHBO（k=2与k=20）进行对比，ERAHBO在19个算法-环境组合中在平均风险回退和样本效率上均显著优于对照组，尤其在较大预算时能更快达到设定的回退阈值。

**⚠️ 局限性**

局限性包括仅使用均值-方差的风险度量，α参数对尺度敏感且未能直接控制尾部失败概率；理论回退界为保守上界；实验基于离线数据，未覆盖在线HPO场景；在某些XLand环境中效果不如预期。

---

## 285. PRISM-Net: Patient-specific reference-guided inter-breast symmetry matching for three-class breast DCE-MRI classification

**arXiv ID:** 2607.26799 | [PDF](https://arxiv.org/pdf/2607.26799v1)

**作者:** Boya Zhang `[一作]` (Nankai University), Xiru Li `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种无配准、基于对侧乳腺的患者特异性参考的双侧学习框架 PRISM‑Net，用于三分类乳腺 DCE‑MRI（无病变/良性/恶性）诊断。

**💡 创新点**

创新点在于将对侧乳腺视为个体背景参考，通过自适应 patch‑级特征匹配与焦点异构重加权实现患者特异性背景归一化，避免了复杂的空间配准并显著提升异构性辨识。

**🔧 技术方法**

使用共享权重 DINOv3 ViT 对切片编码、跨乳腺 patch 匹配构造关系特征、焦点异构重加权、Slice‑level Transformer 融合以及 MLP 分类器，整体为端到端的深度学习框架。

**📊 数据集**

训练与评估使用公开的多中心 ODELIA DCE‑MRI 数据集，并在独立的 2025 年机构外部数据集（含 BPE 与 FGT 复杂子集）进行外部验证。

**📈 对比分析**

与多种基线（BreastMRI‑FCDD、PBPK DAE‑CNN、DisAsymNet、BiGAM‑Net、ODELIA baseline）在内分布与外部样本上比较，PRISM‑Net 在 Macro AUC（84.1%）、Micro AUC（90.6%）和 QWK（60.9%）上均领先，并在 BPE / FGT 复杂集上保持显著优势。

**⚠️ 局限性**

局限性包括：仅在单机构外部数据验证、对侧乳腺异常（如双侧病变、严重 BPE、植入物）时性能可能下降、未实现病灶定位、未整合多参数 MRI 与临床风险因素。

---

## 286. Property-driven Causal Abstractions for Markov Decision Processes

**arXiv ID:** 2607.26787 | [PDF](https://arxiv.org/pdf/2607.26787v1)

**作者:** Jule Schmidt `[一作]` (Ruhr-University Bochum), Nils Jansen `[通讯]` (Ruhr-University Bochum)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于因果关系的属性驱动式分解方法，用于在分量化的MDP模型中生成高效的抽象模型；

**💡 创新点**

创新点在于将特征因果性与MDP抽象相结合，构造了多种因果分区并在理论与实验中证明其优越性；

**🔧 技术方法**

使用了因果推理（特征因果性）、BDD求解、模型检查器PRISM/Storm等技术实现抽象与价值分析；

**📊 数据集**

评估采用了标准PRISM格式的MDP基准模型，涵盖10^4–10^6状态规模；

**📈 对比分析**

通过比较抽象大小、值区间宽度和策略性能，发现SG抽象在保证性与策略效果上优于WA和IMDP，且因果分区能显著减小模型规模；

**⚠️ 局限性**

主要限制是需对完整模型进行昂贵的因果计算，导致在大规模模型上的计算时间高昂，并依赖手工设置阈值与迭代次数。

---

## 287. SkillRise: Agentic Reinforcement Learning for Cross-Task Skill Evolution

**arXiv ID:** 2607.26784 | [PDF](https://arxiv.org/pdf/2607.26784v1)

**作者:** Zhiyuan Yao `[一作]` (Zhejiang University), Yongliang Shen `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种端到端的跨任务技能学习框架，在任务序列中使用同一策略交替解决任务和更新技能文档。

**💡 创新点**

创新点在于将相关但不同任务组织成递增难度的序列，采用分离任务解决与技能策划的信用分配，并通过后续任务奖励指导技能文档的演进，实现技能的可迁移与自我改进。

**🔧 技术方法**

使用强化学习（PPO风格）与大型语言模型（Qwen3）作为策略，通过跨任务奖励分配、分组相对优势计算以及文本生成技术实现任务执行与技能更新。

**📊 数据集**

实验数据集包括 ALFWorld、WebShop 和 ScienceWorld 三个文本交互基准。

**📈 对比分析**

与零射、ReAct、Reflexion、PPO、RLOO、GRPO、GiGPO、LaMer 等基线对比，Pass@1 最高，提升 2.3~8.5 个百分点；在跨任务测试时随序列长度延长，性能持续提升。

**⚠️ 局限性**

局限性：需要任务族元数据来构造序列；实验仅在 4B 规模模型上验证；主要针对文本基准，尚未测试多模态或更大规模模型。

---

## 288. CodeSpec: Dual Executable Specifications for Agentic Long-Horizon Feature Development

**arXiv ID:** 2607.26777 | [PDF](https://arxiv.org/pdf/2607.26777v1)

**作者:** Peiding Wang `[一作]` (Beihang University), Yinghao Zhu `[通讯]` (University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于证据的功能链和双可执行规范，用于在代码库层面实现新功能。

**💡 创新点**

创新点在于将需求语义与仓库证据配对，构建可靠功能链，并将链编译为可执行的架构与行为规范，持续在多轮实现中校验完整性与正确性。

**🔧 技术方法**

主要技术包括大语言模型（DeepSeek‑V4‑Pro、GPT‑5.4‑mini）驱动的功能链推理、规范编译以及执行反馈；以及对代码库的静态/动态检查。

**📊 数据集**

使用 FeatureBench（Lite/Fast/Full）和 NL2Repo‑Bench 进行实验。

**📈 对比分析**

与 OpenHands、Mini‑SWE‑Agent、Claude Code、Codex、RTADev 等基线对比，单模型在 FeatureBench Lite 上通过率最高达 70.7%，比 RTADev 高 5.4%，在全量任务上也保持优势；在 NL2Repo‑Bench 上同样领先，说明方法具有通用性。

**⚠️ 局限性**

局限性包括：额外的规范生成与执行带来微量成本；在任务复杂度极高或需求规模庞大时仍有提升空间；依赖大语言模型的推理，易受模型质量和上下文长度限制。

---

## 289. Journey Operators for Structured Multi-Axis Composition

**arXiv ID:** 2607.26775 | [PDF](https://arxiv.org/pdf/2607.26775v1)

**作者:** Mahesh Godavarti `[一作]` `[通讯]` (A Carrot Inc), Mahesh Godavarti (A Carrot Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多轴结构的代数框架，构建数据点携带内容与轴步变换，定义路径运输和价值路径公式，并将其应用于新的Transformer架构JoFormer；

**💡 创新点**

核心创新在于把RoPE等位置编码推广为多轴可组合、可交换的正交变换；揭示路径无关性仅当轴步变换可交换时成立；引入“V旋转”将同一旋转作用于值侧；提出数据相关角度（JoFormer-projected）实现内容自适应的定位；

**🔧 技术方法**

使用正交旋转矩阵（block-diagonal (2)^{d/2}）、旅程算子、可学习或自适应角度投影、以及对称性与共形性分析；

**📊 数据集**

在视觉上使用CIFAR-100、ImageNet；在语言上使用全英文维基百科；在长度泛化上使用OpenWebText；

**📈 对比分析**

与RoPE、2D-RoPE等基线进行单种子对比，CIFAR-100提升约2%，ImageNet提升0.4%，维基百科语言模型PPL下降至0.3-0.5分，长度泛化比率降至1.02×，显示改进效果；

**⚠️ 局限性**

局限性包括单种子实验、未在大规模预训练上验证、理论仅证明可交换条件下的路径无关性，数据相关角度和V旋转的效果未被严格理论证明。

---

## 290. See2Think: Do Multimodal Models Really Use Intermediate Visual States?

**arXiv ID:** 2607.26769 | [PDF](https://arxiv.org/pdf/2607.26769v1)

**作者:** Siyu Yan `[一作]` (Hong Kong University of Science and Technology), Alex Jinpeng Wang `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一框架See2Think，用于评估多模态模型在推理过程中是否真正使用中间视觉状态，包含1.2K样本的Benchmark和可观测、可干预的视觉动作记录协议VAoT。

**💡 创新点**

创新点在于同时诊断视觉动作选择、渲染真实性和反馈利用三阶段，并通过受控干预区分视觉状态的有用性与模型行为的依赖性。

**🔧 技术方法**

使用了多模态链式思维(CoT)、视觉动作记录(VAoT)、外部渲染器、文本与图像交互以及自动与人工评估技术。

**📊 数据集**

使用了1.2K样本的See2ThinkBench，涵盖12个任务类别，分为2D结构、3D场景和真实世界三类视觉推理，来源包括Geometry、Spatial Puzzle、Physics、Chemistry、Science QA、Abstract Pattern、Object Attributes、Compositional 3D、Robot Manipulation、Robot State Change、Visual Commonsense、Intuitive Physics等。

**📈 对比分析**

通过对GPT‑5.5、GPT‑o3、Gemini 3.5 Flash、Qwen3‑VL‑32B‑Instruct四模型，在CoT、VAoT‑NoRender、VAoT和VAoT‑WrongRender四种设置下对比准确率，发现不同模型和任务域表现差异，未出现统一最优策略；进一步的过程级诊断揭示渲染真实性是主要瓶颈。

**⚠️ 局限性**

局限性包括仅评估四种代表模型，VAoT依赖外部渲染器和自动评估，干预质量对WrongRender效果影响较大，缺乏更广泛的模型覆盖和更细粒度的过程监督。

---

## 291. MediaWiki Code2Code Search: Neural Retrieval for the Semantic Discovery of Open-Source Software Entities

**arXiv ID:** 2607.26766 | [PDF](https://arxiv.org/pdf/2607.26766v1)

**作者:** Francesco Tosoni `[一作]` `[通讯]` (Sant'Anna School of Advanced Studies), Francesco Tosoni (Sant'Anna School of Advanced Studies)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款基于深度学习的代码检索系统 MediaWiki Code2Code Search，能够在超过2,500个 MediaWiki 仓库中根据语义意图检索函数、类型和模板。

**💡 创新点**

创新点包括：1）使用 Qwen3 Embedding 生成 1,024 维向量实现语义检索；2）将所有仓库统一归档到 Software Heritage 并为每个实体生成 SWHID，保证可追溯性；3）采用 split-build 架构，在 GPU 训练后在 CPU 资源受限的 Toolforge 上部署，显著压缩索引体积（168.6 MB，96.6% 以内存减少）。

**🔧 技术方法**

技术涵盖：Python、JavaScript、SQLite、FAISS（IVF‑PQ）、Qwen3 Embedding、Tree‑Sitter 语法分析、FastAPI、Codex 前端、Swagger API、Software Heritage 接口。

**📊 数据集**

数据集为 1,289,452 条结构化代码片段（1,050,748 函数、237,653 类型、1,051 模板），来自 2,242 个活跃的 MediaWiki 仓库，全部通过 Software Heritage 归档。

**📈 对比分析**

通过 27 条手工设计的代码片段基准（四类：算法、命名模糊、跨语言、领域特定）与 BM25 传统检索做对比，Code2Code 的 P@10 达到 0.87（相比 BM25 的 0.64），严格匹配下为 0.52 vs 0.34；在命名模糊类 B 上表现最突出，P@10 0.91 vs 0.54。整体查询延迟在本地 CPU 上为 1.85 s，Toolforge 上约 31 s。

**⚠️ 局限性**

局限包括：1）评估使用单一 LLM 判断，缺乏多评注者一致性；2）基准样本规模小，且仅英文；3）索引存在较高重复率，影响精准度评估；4）对高度专有词汇的检索仍受限，需混合词法-语义排名；5）Toolforge 上 CPU 资源受限导致查询延迟高，需进一步优化或加入 GPU。

---

## 292. Searching for Robust Augmentations to Improve Out-of-Domain Generalization in Dermoscopic Skin Cancer Classification

**arXiv ID:** 2607.26765 | [PDF](https://arxiv.org/pdf/2607.26765v1)

**作者:** Alexander Kozachok `[一作]` (Trusted AI Research Center, Russian Academy of Sciences), Oleg Samovarov `[通讯]` (Trusted AI Research Center, Russian Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

系统搜索并评估了多种图像增广策略（光度、几何、伪色等），以提升基于 ConvNeXt-Large 的皮肤镜图像二分类模型在源级 OOD 情况下的鲁棒性。

**💡 创新点**

创新点在于：①采用源级留出法评估域泛化；②对光度增广进行细粒度组合探索，发现包含 HEStain 等非传统光度操作的混合策略 mix 在 OOD 上显著提升 ROC‑AUC；③通过梯度显著性图和 t‑SNE 等解释手段验证模型对源特定特征的依赖减少。

**🔧 技术方法**

技术方法包括：ConvNeXt‑Large 主干网络、AdamW + focal loss、多种光度/几何增广（ColorJitter、PlanckianJitter、HEStain 等），以及 ROC‑AUC、DeLong 统计检验、bootstrap 置信区间、Grad‑CAM、t‑SNE、域混合指标等评估工具。

**📊 数据集**

使用了公共 ISIC Archive（BCN20000、Derm12345、HAM10000、ISIC 2016‑2020、HIBA、BALD、MILK10k）与 Derm7pt 的训练与内部/外部测试集，并在完全独立的临床数据集 Melanoscope 上做最终外部验证。

**📈 对比分析**

对比方法：在内部测试集、OOS（HAM10000 + ISIC 2019‑2020）以及外部 Melanoscope 上分别计算 ROC‑AUC、AUPRC 等指标。mix 策略在内部测试上 +0.006 ROC‑AUC，在 OOS 上 +0.053 ROC‑AUC（p<0.001），并在多种随机种子下保持一致；在 Melanoscope 上仅观察到趋势性提升，未达到显著性。

**⚠️ 局限性**

主要限制：①策略选择与评估在同一 OOD 源上完成，存在选择偏差；②外部验证样本量不足，未能充分证明跨机构泛化；③提升主要来自降低误报而非阳性召回，且二分类框架掩盖了各亚型差异；④仅使用 ConvNeXt‑Large，未探讨其他骨干；⑤训练方差仍需更大规模验证。

---

## 293. Relation Geometry in Semantic Space of Language Models

**arXiv ID:** 2607.26762 | [PDF](https://arxiv.org/pdf/2607.26762v1)

**作者:** Zhihan Cao `[一作]` (University of Tokyo), Takenobu Tokunaga `[通讯]` (Institute of Science Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言模型语义空间中的关系几何，探索不同语义关系在向量空间中的可分性、方向性和传递性，并量化词形与上下文信息对关系几何的影响。

**💡 创新点**

提出一种基于双线性探针的解释性评估框架，能够同时衡量关系可预测性、方向性与传递性；首次对六种关系（上位/下位、整体/部分、同义、反义）以及无关关系进行统一实验；量化词形与上下文信息在关系几何中的贡献。

**🔧 技术方法**

双线性探针（Bilinear probe）+交叉熵损失；随机对照探针；词形消除与上下文消除（degenerate/egalitarian decontextualization）；统计显著性检验（配对置换检验）。

**📊 数据集**

SemCor语料（带有WordNet词义标注的句子）构造三十万以上三类关系的三元组；使用fastText静态词向量作为基线；同时使用ModernBERT（MLM）、LLaDA-8B-Base（DLM）和LLaMA-3.1-8B（CLM）生成上下文敏感的词向量。

**📈 对比分析**

采用宏观F1（关系可预测性）、方向性得分和传递性得分进行比较，并通过随机对照探针控制模型自身拟合能力；实验显示LLaMA在大多数指标上优于其他两种LM，尤其在异构关系（上位/下位、整体/部分）上表现更好；对称关系（同义/反义）整体表现较差；传递性随距离增加显著下降。

**⚠️ 局限性**

仅使用线性探针限制了对非线性关系几何的探测；实验仅涉及名词及其词义，未覆盖多词表达、形态变化等；使用的SemCor数据来自20世纪70年代的Brown语料，可能不代表现代用法；模型对比未匹配参数量或训练数据，LLaMA优势可能受此影响。

---

## 294. Phoneme- vs. Character-Level Targets and Selective State-Space Models for Intracortical Brain-to-Text

**arXiv ID:** 2607.26751 | [PDF](https://arxiv.org/pdf/2607.26751v1)

**作者:** Lucas Zamora Vera `[一作]` (Universitat Oberta de Catalunya), Jose A. Gonzalez-Lopez `[通讯]` (University of Granada)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在 Brain‑to‑Text '25 benchmark 上，对 intracortical 脑‑到‑文字系统的两大设计轴进行系统性对比：一是比较传统 GRU 与新型选择性 state‑space 模型（Mamba 及其 ConvMambaGRU 变体）作为神经解码器；二是比较 phonetic（音素）与 character（字符）两种目标表示，全部采用统一的 CTC 训练框架并使用 LM 重评分；

**💡 创新点**

首次在 intracortical 解码中实现并评估 ConvMambaGRU hybrid 结构，提供了基于目标表示的错误分析，展示了音素目标对神经信号的更高可直接性以及字符目标在去除发音词典依赖方面的优势；

**🔧 技术方法**

使用的核心技术包括 CTC 损失、GRU、Mamba、ConvMamba、ConvMambaGRU 结构；SpecAugment 风格的数据增强；按 session 归一化、时间补丁化预处理；GPT‑2 XL 与 KenLM 语言模型进行 beam search 及 LM 重评分；

**📊 数据集**

使用公开的 Brain‑to‑Text '25 benchmark 数据集，单位 ALS 受试者在 256 维微电极阵列上收集的 8,072 训练、1,426 验证试验的神经信号与对齐文本；

**📈 对比分析**

通过多随机种子平均 ± 标准差评估 PER，使用 Wilcoxon 检验比较；结果显示 GRU 在 phonetic 目标上最低 PER 为 12.62% 与 WER 21.19%；ConvMambaGRU 在 PER 仅略高（13.44%）但 WER 更高（24.11%）；字符目标整体性能不及 phonetic，最佳 WER 26.28%；

**⚠️ 局限性**

受限于单个受试者和极少的训练样本，SSM 仅在加入卷积前置层和 GRU 后处理后才接近 GRU；语言模型对最终 WER 影响显著；缺乏跨受试者泛化验证；未探索更丰富的目标表示或更强的训练/重评分策略。

---

## 295. Dual Inversion for Text-to-Image Diffusion Models: From Both Prompt and Noise Perspectives

**arXiv ID:** 2607.26735 | [PDF](https://arxiv.org/pdf/2607.26735v1)

**作者:** Xiaolong Liu `[一作]` (University of Technology Sydney), Huan Huo `[通讯]` (University of Technology Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种双重逆向方法 Dualin，先利用 VLM、CLIP 与 LLM 三阶段推断可解释的 hard prompt，再通过无条件 DDIM 逆向恢复目标图像的 latent 噪声，从而实现高保真、可编辑的图像重建。

**💡 创新点**

创新点在于将 prompt 逆向与噪声逆向双重并行进行：①使用 VLM 生成语义草稿，CLIP 进行概念检索提升细节捕捉，LLM 生成最终可读 prompt；②通过无条件 DDIM 恢复噪声保证结构一致性，理论证明可实现无重优化的可控编辑。

**🔧 技术方法**

技术细节包括：Vision‑Language Model（如 BLIP‑2）提取语义；CLIP（ViT‑Large）检索风格与质量概念；LLM（Llama‑3‑8B‑Instruct）融合语义与概念生成 Prompt；无条件 DDIM 逆向恢复噪声；评估使用 CLIP‑T/CLIP‑I、LPIPS、SSIM、GPT‑4o 与 Gemini 3.0 评判。

**📊 数据集**

实验使用三大数据集：MS COCO、LAION 以及 DiffusionDB，以不同 T2I 模型（Stable Diffusion V1.5、Stable Diffusion XL、Pixart‑α）检验方法泛化。

**📈 对比分析**

与 PEZ、PH2P、BLIP、VGD 等基线比较，Dualin 在 CLIP‑T、CLIP‑I、LPIPS、SSIM 等指标上显著领先，GPT‑4o/Gemini 评估分数最高；速度上仅比无梯度方法略慢，明显快于梯度优化方法。

**⚠️ 局限性**

局限性在于额外的噪声逆向步骤导致计算成本略高；对极其复杂或高分辨率场景的细节重建仍存在挑战；需要进一步研究噪声估计的鲁棒性和跨模型泛化。

---

## 296. CASIAL: Geometric Distortion Robust Image Watermarking

**arXiv ID:** 2607.26729 | [PDF](https://arxiv.org/pdf/2607.26729v1)

**作者:** Yupeng Qiu `[一作]` (National University of Singapore), Ee-Chien Chang `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种针对几何变形鲁棒的深度图像水印框架CASIAL，能够在区域移除和失同步攻击下可靠解码并保持高视觉质量；

**💡 创新点**

创新点在于两大机制：①基于覆盖图像的全局水印扩散（CAS），将信息以覆盖条件方式分布到整幅图像；②空间注意力对齐学习（IAL），在几何失配时对特征进行自适应聚合，实现几何不变表示；

**🔧 技术方法**

采用变分自编码器（VAE）进行特征编码/解码，结合双通道注意力（IAL）与JND感知残差衰减，整体构建端到端可训练的水印嵌入与提取网络；

**📊 数据集**

实验基于公开图像数据集（如COCO、DIV2K等常用高分辨率图像集）进行训练与评估；

**📈 对比分析**

与11种最新基线在16种白盒与黑盒失真（几何、信号、光照）下对比，CASIAL在几何变形上实现至少98.44%准确率，白盒平均鲁棒率99.40%，并在黑盒平均鲁棒率99.55%，同时保持40.82 dB PSNR、0.9779 SSIM、0.0074 LPIPS的最高视觉质量；

**⚠️ 局限性**

主要局限包括：①对超大尺寸或非标准尺寸图像的鲁棒性尚未全面验证；②模型训练与推理的计算开销较大，适配实时或嵌入式系统需要进一步优化。

---

## 297. From Individual to Shared Ownership: A Coalitional Game Approach to Sustainable Co-investment

**arXiv ID:** 2607.26725 | [PDF](https://arxiv.org/pdf/2607.26725v1)

**作者:** Yue Yu `[一作]` (Telecom SudParis), Rosario Patanè `[通讯]` (Telecom SudParis)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于合作博弈的可持续共建共享基础设施框架，并通过监管激励实现投资协调。

**💡 创新点**

创新点在于将共建问题建模为可转移效用的协作博弈，利用线性生产博弈（LPG）实现核心非空且直接得到双价格核心分配。

**🔧 技术方法**

采用了合作博弈理论、线性生产博弈（LPG）与线性规划、双重价格核心分配方法，并在案例研究中通过数值仿真验证。

**📊 数据集**

使用了公开的美国基准参数（电价、补贴、太阳能发电成本）和三家数据中心的异质负荷曲线进行仿真。

**📈 对比分析**

通过对比单独投资与集体投资（有无补贴）来评估模型，结果显示：无补贴时集体投资降低总装机容量且产值更高；有补贴时则促使集体装机容量上升并实现价值叠加，双价格分配保证核心稳定。

**⚠️ 局限性**

局限性包括假设需求、供给和价格确定性、仅考虑单一共享设施、激励参数为外生且未考虑网络约束，未来可扩展为随机/鲁棒模型和多设施、多层级协作。

---

## 298. UrbanDS: A Graph-Guided LLM Multi-Agent System for Data-Intensive Urban Tasks

**arXiv ID:** 2607.26724 | [PDF](https://arxiv.org/pdf/2607.26724v1)

**作者:** Zhilun Zhou `[一作]` (Tsinghua University), Yong Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建一个面向大规模、异构城市数据仓库的图结构（包含数据集技能与空间/时间/语义关系），并基于此设计多代理LLM系统（规划、执行、报告）自动完成城市数据科学任务。

**💡 创新点**

创新点：①利用数据集技能（一次性生成可重用的描述性知识）与图关系来显式表示数据集内容和互联；②图引导式数据检索与递进式规划，解决大规模数据发现与组合难题；③多代理协同执行与共享进度内存，提升可解释性与可复用性。

**🔧 技术方法**

技术：大型语言模型（DeepSeek‑V4‑Pro），多代理架构（Planner、Execution、Report等），数据集特征提取与关系识别（空间覆盖、时间区间、语义代码本），图数据库（dataset graph），Python 代码生成与执行。

**📊 数据集**

数据集：UrbanDS‑Bench 94 个城市数据（10 个中国大城市）涵盖地理、移动、社会经济等 450 条问答任务与 8 个建模任务；部署环境中 103 个真实城市治理与运营数据。

**📈 对比分析**

与 DS‑Agent、Data Interpreter、DeepAnalyze、AutoGen、Claude Code 等基线在 UrbanDS‑Bench 与 CoDA‑Bench 进行对比。UrbanDS 在空间/时间/时空任务中分别取得 65.7%/83.6%/73.9% 的准确率，总体 70.0%，比最佳基线 Claude Code 提升 11.2%；在 CoDA‑Bench 上 46.2% 也高于 36.1%。在建模任务上均获得最高指标，尤其在需要辅助数据的任务上表现显著。

**⚠️ 局限性**

局限性：①对需要 4 及以上数据集的任务准确率仍低于 53%；②系统侧重于已知问题的查询，缺乏主动探索与新见解挖掘能力；③依赖 8B 级 LLM，可能受模型容量限制；④在极大规模仓库中，关系识别与技能生成仍需进一步优化。

---

## 299. DREvo: Distilling Recalibrated Historical Experience for Harness Self-Evolution

**arXiv ID:** 2607.26722 | [PDF](https://arxiv.org/pdf/2607.26722v1)

**作者:** Hanghui Guo `[一作]` (Southeast University), Shimin Di `[通讯]` (Southeast University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于历史实验自我进化的 harness 进化方法 DREvo，通过将实验日志分解为函数级证据，动态重新校准证据有效性并将其转化为角色条件搜索意图。

**💡 创新点**

创新点在于：① 将粗粒度日志转换为函数级证据；② 引入状态依赖证据再校准以判定证据是否仍适用；③ 通过角色条件搜索意图将证据明确为具体修改指令。

**🔧 技术方法**

技术手段包括函数级证据锚定、状态依赖证据再校准（利用历史一致性、鲜活度、AST 编辑距离），以及角色条件搜索意图提炼（执行、避免、复测、探索）。

**📊 数据集**

数据集涵盖三项领域推理任务（USPTO、S2D、Law）以及两项 agentic 任务（Terminal‑Bench 2.0、SWE‑Bench Verified）。

**📈 对比分析**

与多种手工设计的 harness 及现有自我进化基线（MCE、ACE、Meta‑harness、AHE 等）比较，DREvo 在五个基准上均取得最高准确率，平均提升 16.2%（领域推理）和 13.8%（agentic）且演化轨迹更平滑。

**⚠️ 局限性**

局限性在于仍依赖单一 LLM 提议器，针对跨组件依赖的证据校准尚不完善，且在更长的进化周期下如何自适应角色选择仍待研究。

---

## 300. ActSWM: Action-Sensitive World Models for Long-Horizon Planning in Open-World Games

**arXiv ID:** 2607.26712 | [PDF](https://arxiv.org/pdf/2607.26712v1)

**作者:** Zhenfeng Gan `[一作]`, Xueqian Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种动作敏感的潜在世界模型 ActSWM，解决了长时序预测中的上下文崩溃问题，并支持闭环控制规划。

**💡 创新点**

通过引入基于rollout的动作对比损失和固定动作读取器的转移分离原则，使模型在不同动作条件下保持可分辨的潜在未来，显著提升规划可靠性。

**🔧 技术方法**

采用JEPA潜在预测架构，结合多步预测、对比hinge loss、固定readout约束、CEM规划以及离线视频动作恢复等技术。

**📊 数据集**

使用了Minecraft VPT离线轨迹、MineStudio任务（Torch放置、Stone Mining、Pillar Building）以及Counter‑Strike 2、GTA V、Apex Legends等跨游戏视频数据集。

**📈 对比分析**

通过step‑drift诊断、闭环规划成功率、CEM恢复的动作gap、关键帧准确率等指标与LeWM、WAM等基线对比，ActSWM在动作间隔保持上提升约380倍，规划成功率提高45%，动作恢复gap提升至16.5×。

**⚠️ 局限性**

仍受限于离线数据的质量、固定读取器可能限制表征灵活性，并未在极大多样动作空间或实时在线学习场景下进行充分验证。

---

## 301. SecRespond: Benchmarking AI Agents for Real-World Post-Compromise Incident Response

**arXiv ID:** 2607.26791 | [PDF](https://arxiv.org/pdf/2607.26791v1)

**作者:** Lehan Wang `[一作]` (Tongyi Lab, Alibaba Group), Xiaomeng Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了首个针对后受损端点事件响应的LLM代理基准，要求代理在已获取磁盘取证快照、实时告警、漏洞扫描和基线检查等输入后，生成入侵报告、基线风险、漏洞风险及完整的补救计划。

**💡 创新点**

创新点在于：①将真实受损云主机的完整取证快照与多源安全分析相结合，构建10个多维度（4入口点、21 ATT&CK技术、5操作系统）的安全范围；②设计层级化的Capability（CAP）评估框架和280个细粒度检查点，实现检测与规划双维度的LLM-as-a-Judge评分；③引入可执行的程序性先验（procedural priors）技能，验证其对模型规划性能的提升。

**🔧 技术方法**

核心技术包括：LLM代理框架OpenCode、基于多判官的LLM-as-a-Judge评测方法、层级化检查点拆分与CAP映射、以及三大专有LLM（Claude Opus 4.7、Gemini 3.1 Pro、GPT-5.4 Pro）共同判分。

**📊 数据集**

使用的数据集为10个构造的安全范围，包含各自的磁盘取证快照、告警、漏洞扫描结果与基线检查报告；共涉及4种入口点、21种MITRE ATT&CK技术、5个操作系统。

**📈 对比分析**

通过将23个大型语言模型（覆盖8系列）在OpenCode环境下执行，采用三判官平均分，计算检测与规划的CHECKPOINT分数，并映射到CAP维度进行聚合。结果显示：所有模型检测分数均高于规划分数；Claude Opus 4.7在检测和规划上表现最优（检测≈79%，规划≈65%），但仍未达到任何范围的完全检测与补救。

**⚠️ 局限性**

局限性包括：①模型在主动探测“无声”入侵和完整补救方面表现不足；②检测与规划之间存在显著差距，规划水平普遍偏低；③评测范围仅为10个实验构造，难以覆盖更广泛的真实攻击场景；④LLM-as-a-Judge的判官主观性和偏差仍是潜在影响因素。

---

## 302. Stable and Budget-Feasible Coalition Formation for Clustered Federated Learning: A Hedonic Potential-Game Approach

**arXiv ID:** 2607.26788 | [PDF](https://arxiv.org/pdf/2607.26788v1)

**作者:** Cengis Hasan `[一作]` `[通讯]` (Cognifinity), Cengis Hasan (Cognifinity)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文设计了基于可转移盈余的联邦学习聚合机制，利用对称对偶分配构造精确势函数，保证在预算可行的前提下产生Nash稳定且可去中心化的协同联盟，从而提升聚合效果。

**💡 创新点**

创新点在于将联邦学习收益与参与成本分离，提出对称对偶对分配实现势函数的等价性，证明存在Nash稳定分区并给出效率与预算松弛的定量保证，以及将全局潜能最大化映射到加权一致性聚类实现近似。

**🔧 技术方法**

使用了组合优化中的势函数游戏理论、子模函数与凸优化、加权一致性聚类算法、以及分布式更好响应动态。

**📊 数据集**

实验使用CIFAR-10图像分类数据集，构建了四方参与者的异构数据分布。

**📈 对比分析**

与FedAvg、IFCA、独立本地训练、等价分配等基线对比，机制在所有随机种子下达到或接近表格最优，价格稳定性为1，平均稳定迁移仅需1.5步，性能优于全局聚合约26%且明显优于等价分配。

**⚠️ 局限性**

局限性包括仅在n=4小规模下验证，需枚举所有子集；对更大参与者规模与多任务的可扩展性未证明；需要准确估计收益/成本，且机制未考虑私有类型的激励兼容。

---

## 303. Enhancing Generative Information Extraction with Two-step Validation: A Product Attribute Use Case

**arXiv ID:** 2607.26780 | [PDF](https://arxiv.org/pdf/2607.26780v1)

**作者:** Yi-Sheng Hsu `[一作]` (Ruhr West University of Applied Sciences), Uwe Handmann `[通讯]` (Ruhr West University of Applied Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两步验证式生成式信息抽取（IE）方法，利用预训练语言模型（PLM）先做粗略抽取，再让大型语言模型（LLM）进行校正，以提升在产品领域的抽取性能；

**💡 创新点**

创新点在于将生成式IE改写为验证任务，通过LLM的校正能力提升低显著性实体的抽取效果，并证明轻量级LLM在本任务上也可取得与大型模型相近的表现；

**🔧 技术方法**

使用了开源LLM（Llama、Mistral、Gemma等）和微调的PLM（RoBERTa、DeBERTa），配合一次性提示（one-shot prompting）和自我校正框架；

**📊 数据集**

采用Amazon产品描述和Kaggle电商文本分类两个公开数据集，人工标注了六类实体（尺寸、重量、产品编号、组件、材料、制造商）；

**📈 对比分析**

与单步生成式抽取任务及PLM基线对比，基于F1分数评估，发现两步验证方法在大多数模型上提升了1–7个百分点，特别是对弱表达实体的召回率显著提高；

**⚠️ 局限性**

限制包括仅有单一标注者导致标注偏差、token化问题导致评估误差、以及在极小LLM上验证任务效果有限。

---

## 304. Long-Tailed 3D Point Cloud Dataset Distillation

**arXiv ID:** 2607.26763 | [PDF](https://arxiv.org/pdf/2607.26763v1)

**作者:** Jiahao You `[一作]` (Huazhong University of Science and Technology), Xianzhi Li `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对点云数据的长尾分布数据蒸馏框架，能够在有限的合成预算下生成高效的合成数据集，保持原始数据集的训练效用。

**💡 创新点**

首次结合自适应合成预算分配与长尾分布匹配，利用全局‑局部特征对齐和基于先验的监督，专门针对训练集与测试集均为长尾分布的点云数据进行蒸馏。

**🔧 技术方法**

采用自适应合成预算 (ASB)、3D 长尾分布匹配 (3D‑LTDM)，其中包含全局‑局部特征对齐 (GLFA) 与先验感知监督 (PAS) 的分布匹配与专家模型监督技术。

**📊 数据集**

在 ModelNet10/40、ShapeNet55、ScanObjectNN、ShapeNetPart 等点云长尾基准，以及 CIFAR‑10/100 长尾版本上进行实验验证。

**📈 对比分析**

与 PCC、SADM、LTDD 等现有蒸馏与长尾蒸馏方法对比，实验表明在 ShapeNet55 上提升 7% 以上，在不同模型（PointNet、PointNet++、DGCNN 等）与任务（分类、分割）中均取得显著性能提升。

**⚠️ 局限性**

方法依赖训练分布进行预算分配与监督，若训练‑测试分布出现显著漂移，性能可能下降；在极端长尾条件下的适应性尚待进一步验证。

---

## 305. Domain adaptation for handwriting trajectory reconstruction from IMU sensors

**arXiv ID:** 2607.26736 | [PDF](https://arxiv.org/pdf/2607.26736v1)

**作者:** Florent Imbert `[一作]` (IRISA, Universite de Rennes, INSA Rennes), Eric Anquetil `[通讯]` (IRISA, Universite de Rennes, INSA Rennes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用域适应技术，将从成人平板写字训练得到的手写轨迹重建模型迁移到儿童平板写字，并与从零训练和微调模型进行对比。

**💡 创新点**

首次将 Domain‑Adversarial Neural Network (DANN) 与 Temporal Convolutional Network (TCN) 结合用于手写轨迹重建，实现跨年龄组（成人→儿童）的共享特征空间，提升两域性能。

**🔧 技术方法**

对抗式域适应（DANN）、TCN（特征提取器和任务预测器）、域判别器、Fréchet 距离评估。

**📊 数据集**

两个数据集：成人平板写字 9629 样本；儿童平板写字 3910 样本，包含字符、词、方程和形状。

**📈 对比分析**

与基线（仅成人训练）、从零训练儿童模型、微调成人模型进行比较。使用 Fréchet 距离评价；在儿童测试集上 DANN 的 Fréchet 距离 0.349，显著低于基线 0.470，且优于从零训练 0.348、微调 0.348；在成人测试集上 DANN 0.364 略高于基线 0.332，但仍优于从零训练和微调。整体来看，DANN 在两域上都表现最佳，兼顾两类用户。

**⚠️ 局限性**

未对 λ 权重进行自适应调优；批次划分与 padding 方式需进一步研究；仅验证了平板→平板迁移，尚未评估平板→纸张的跨域性能。

---

## 306. Online Handwriting Trajectory Reconstruction from Kinematic Sensors using Temporal Convolutional Network

**arXiv ID:** 2607.26733 | [PDF](https://arxiv.org/pdf/2607.26733v1)

**作者:** Wassim Swaileh `[一作]` (IRISA, Université de Rennes, INSA Rennes), Eric Anquetil `[通讯]` (IRISA, Université de Rennes, INSA Rennes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套从数字笔 IMU 传感器信号重建在线书写轨迹的完整处理流水线，包括预处理（DTW 对齐）、基于 Temporal Convolutional Network 的回归模型以及后处理。

**💡 创新点**

创新点在于：① 使用 DTW 进行时序对齐而非线性插值，保留书写动态；② 采用非因果 TCN 结构，能够在短样本且噪声较大的 IMU 数据上获得更好的时空特征；③ 构建并公开了新的 IRISA‑KIHT 轨迹重建基准数据集。

**🔧 技术方法**

技术手段包括：IMU 传感器（加速度计、陀螺仪、磁力计、压力感应）采集；DTW 对齐；Temporal Convolutional Network（3 层膨胀卷积 + 全连接）；Fréchet 距离评估。

**📊 数据集**

使用公开的 FAU‑EINNS 数据集和自建的 IRISA‑KIHT（38/30 名作者）以及其子集 IRISA‑KIHT‑S 进行实验评估。

**📈 对比分析**

与 Wehbi 等人提出的 CNN+线性插值方法进行对比。实验结果显示：在 Fréchet 距离上，TCN+DTW 在 FAU‑EINNS 上从 0.2628 降至 0.077，在 IRISA‑KIHT 上从 0.4691 降至 0.2039，性能提升显著。

**⚠️ 局限性**

主要限制在于无法准确重建未被平板跟踪的悬停（hovering）轨迹，且仅评估了轨迹形状而未考虑速度、加速度等动态特征。

---

## 307. AtmosERC: Modeling Dialogue-Level Affective Atmosphere for Emotion Recognition in Conversation

**arXiv ID:** 2607.26726 | [PDF](https://arxiv.org/pdf/2607.26726v1)

**作者:** Weijie Feng `[一作]` (Hefei University of Technology), Zhiyong Cheng `[通讯]` (Hefei University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

无法获取论文内容，无法概括

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

## 308. FreeShadow: Training-Free Shadow Removal via Illumination Transfer and Selective Content Preservation in Diffusion Models

**arXiv ID:** 2607.26715 | [PDF](https://arxiv.org/pdf/2607.26715v1)

**作者:** Yinan Wang `[一作]` (South China University of Technology), Patrick Le Callet `[通讯]` (Université de Nantes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了FreeShadow，一种无需训练或测试时优化的阴影移除框架，利用预训练扩散模型完成阴影恢复与内容保持。

**💡 创新点**

创新点包括：① 通过照明转移注意力（ITA）实现阴影区域照明信息的直接迁移；② 采用选择性注意力映射与高频注入（SAMI & SHFI）在不破坏结构细节的前提下抑制残影；③ 引入局部纹理保持重光照（LTPR）解决VAE压缩导致的纹理失配；④ 通过阴影重定向引导（SRRG）加强采样路径向阴影去除方向迁移。

**🔧 技术方法**

主要技术：Stable Diffusion 2.1预训练模型；DDIM反演与采样；自注意力权重重调；PCA与自阴影影响指标；离散小波变换（DWT）高频注入；局部均值方差融合；多尺度注意力层选择性注入。

**📊 数据集**

实验使用 ISTD+、UIUC+、SRD 与 PSM 四大阴影去除基准数据集。

**📈 对比分析**

与监督、无监督及零样本方法对比，在跨数据集测试与 PSM 上均实现 MAE、SSIM、PSNR 领先或可与最优监督模型相当，且在 20 步采样下推理速度比大多数零样本方法快约 95% 以上。

**⚠️ 局限性**

局限性：对自阴影处理不如投射阴影彻底；对阴影掩码准确性要求较高，粗略掩码会导致边缘纹理失真；在极硬阴影或自阴影与几何紧密耦合场景中仍易出现残影。

---

## 309. Multimodal fusion of visual and morphometric features for avian bone classification

**arXiv ID:** 2607.26743 | [PDF](https://arxiv.org/pdf/2607.26743v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 310. Mixture-of-experts for handwriting trajectory reconstruction from IMU sensors

**arXiv ID:** 2607.26708 | [PDF](https://arxiv.org/pdf/2607.26708v1)

**作者:** Florent Imbert `[一作]` (IRISA, Universite de Rennes, INSA Rennes), Romain Tavenard `[通讯]` (IRISA, Universite Rennes 2)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用 Digipen 手写笔的惯性测量单元（IMU）信号，通过混合专家（MOE）网络重建在线手写轨迹，区分触笔和抬笔两阶段。

**💡 创新点**

创新点包括：① 将轨迹分为 2D 触笔段和 3D 抬笔段，分别训练专门的专家模型；② 在触笔专家中加入物理动力学的时间上下文；③ 对抬笔专家使用倾斜平面 3D 标注数据进行微调；④ 设计 MOE‑CI 系统实现两专家协同；⑤ 公开构建 KIHT‑Public 数据集作为基准。

**🔧 技术方法**

采用 Temporal Convolutional Network（TCN）作为骨干网络，构建 Touching Expert Model（TEM）和 Hovering Expert Model（HEM）；通过 DTW 对齐、Fréchet 距离评估、以及上下文/高度特征增强；实现 MOE‑CI 的动态专家切换。

**📊 数据集**

使用 KIHT‑Public（130 份录制，约 2761 例子）和 KIHT‑Private（371 份录制，约 11811 例子）两套数据集，分别包含字符、单词、方程等多样化手写样本，并加入倾斜面采集数据。

**📈 对比分析**

与两种主流方法（仅触笔 TCN、CNN+线性插值）在同一数据集上对比，采用 Fréchet 距离进行 label 与 stroke 级别评估。MOE‑CI 在 label 级别 0.312、stroke 级别 0.091 的距离显著低于对手，#wins 188/194，表明性能优越。

**⚠️ 局限性**

局限性：① 训练数据仍以成人平板书写为主，对儿童手写及纸面书写的泛化能力有限；② 需要两模型，导致内存占用较大，难以直接嵌入笔尖硬件；③ 数据集规模相对有限，难以覆盖所有书写变异。

---

## 311. Tangling Pull Requests: Curating a Commit Untangling Dataset from Merged PRs

**arXiv ID:** 2607.26730 | [PDF](https://arxiv.org/pdf/2607.26730v1)

**作者:** Yuki Ueno `[一作]` (Institute of Science Tokyo), Takashi Kobayashi `[通讯]` (Institute of Science Tokyo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过自动筛选合并的Pull Request并应用八条过滤规则，构建了大规模的Composite Commit（CC）与Single‑Task Commit（STC）数据集，显著降低人工标注成本；

**💡 创新点**

创新点在于利用PR的自然拆分作为真实标签，证明此方法生成的数据与传统Herzig启发式数据在结构上显著不同，并且对自动untangling算法评估产生实质影响；

**🔧 技术方法**

技术包括基于GitHub API的爬取、八条精细化过滤规则、人工评估、KS检验与Cliff's delta统计、Herzig的confidence voters算法以及精度（precision）评估；

**📊 数据集**

使用的主要数据集为Java 15,320 PR（对应34,091 STC）与Python 40,303 PR（对应91,995 STC），覆盖1,092 Java与1,262 Python开源项目；

**📈 对比分析**

通过与Herzig基准数据集的KS检验显示文件距离、包距离等维度差异显著；在Herzig confidence voters算法上，本方法得到的精度为0.575，显著高于Herzig基准的0.504；

**⚠️ 局限性**

局限性包括仅覆盖GitHub Java/Python项目，可能忽略非PR或多作者PR产生的CC；数据仍含约20%噪声；方法在其他语言或平台的泛化需进一步验证。

---

## 312. CheckVLA: Execution-Time Verification with Action-Conditioned World Model for Long-Horizon Mobile Manipulation

**arXiv ID:** 2607.26789 | [PDF](https://arxiv.org/pdf/2607.26789v1)

**作者:** Yushan Liu `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种基于动作条件世界模型的执行时验证框架CheckVLA，用于在移动机器人长时程任务中监测并纠正已提交的动作块

**💡 创新点**

创新点在于：①使用冻结的动作条件世界模型将动作块视为可验证的预测；②采用功能式共形校准给出全局概率阈值，控制首次不必要干预的风险；③根据阈值超越程度动态调整被替换动作块的保留强度；④引入时延感知的硬前缀裁剪和事件驱动的关键帧记忆以实现可部署的修复

**🔧 技术方法**

技术要点包括：动作条件滚动预测、共形风险阈值触发、时延感知后缀重写、风险头聚合、以及基于关键帧的情景记忆

**📊 数据集**

使用了公开的RoboCasa365数据集（365个厨房任务，Human300预训练）

**📈 对比分析**

与周期性重规划和其他执行时监测方法对比：在相同调用预算下，CheckVLA平均成功率为36.1%，比周期性重规划提升8.5个百分点；及时召回率为77.9%，远高于仅观察的48.6%和动作打乱的37.9%

**⚠️ 局限性**

局限性包括：验证仅在仿真环境下评估；共形校准仅保证首个不必要干预概率，不保证召回或分布漂移下的鲁棒性；修复受时延和可部署后缀长度限制；需要针对新传感器或硬件重新校准与测试

---

## 313. Global Sensitive-Based Input Shaping for UAV-Payload Precision Motion Control

**arXiv ID:** 2607.26717 | [PDF](https://arxiv.org/pdf/2607.26717v1)

**作者:** Karan Baker `[一作]` (Louisiana State University), Adrian Stein `[通讯]` (Louisiana State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了利用全局敏感性分析和Shapley值约束设计UAV-载荷系统的输入整形器，实现对载荷质量和绳长不确定性的鲁棒控制。

**💡 创新点**

首次将全局敏感性分析与Shapley值理论结合用于输入整形器设计，从而显著降低对不确定参数的敏感性。

**🔧 技术方法**

使用Lagrangian建模、全局敏感性分析(GSA)、Shapley值约束、MATLAB fmincon优化、Monte Carlo仿真。

**📊 数据集**

仿真中使用均匀分布的载荷质量[0.2,1]kg和绳长[0.2,1.8]m，共10,000次蒙特卡罗采样。

**📈 对比分析**

与非鲁棒、鲁棒、minimax及传统输入整形器比较，GSA和Shapley GSA输入整形器在载荷摆幅范围均值和标准差上分别提升约30%和显著降低约70%，实现更小的摆幅波动。

**⚠️ 局限性**

由于系统非线性，Shapley值数值求解不精确且计算量大；改进需采用多项式混沌或更高效的代理模型。

---

## 314. Automated Multilabel Mpox Research Classification with Explainable Transformer Models

**arXiv ID:** 2607.26700 | [PDF](https://arxiv.org/pdf/2607.26700v1)

**作者:** Tanjim Taharat Aurpa `[一作]` `[通讯]` (University of Frontier Technology, Bangladesh), Tanjim Taharat Aurpa (University of Frontier Technology, Bangladesh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究对14590篇关于Mpox（猴痘）的研究论文进行多标签分类，分别标记为“预防相关”“流行病学相关”“疫苗相关”，并通过SHAP解释模型决策。

**💡 创新点**

①使用BERT及其他Transformer模型在大规模Mpox文献上的多标签分类任务中实现高精度；②将SHAP用于文本级别的可解释性，揭示各类标签的关键词语与上下文影响；③在数据集上从四类降至三类以缓解标签不平衡。

**🔧 技术方法**

Transformer模型（BERT、RoBERTa、ALBERT、DistilBERT、ELECTRA）+ SHAP；数据预处理包括清洗、分词、词干化/词形还原；多标签训练采用sigmoid输出与二元交叉熵。

**📊 数据集**

PubMed检索得到14590篇文章，包含PMID、标题、摘要和人工标注的三类标签；通过关键词筛选和人工检查保证标签质量。

**📈 对比分析**

在准确率、Micro‑F1、Macro‑F1等指标上与五种Transformer比较，BERT取得最高准确率97.05%、Micro‑F1 97.67%、Macro‑F1 96.46%；RoBERTa与ALBERT紧随其后，DistilBERT略低，ELECTRA表现最差。AUC均接近0.97，显示模型在三类上的区分能力极佳。

**⚠️ 局限性**

①模型受限于手工标注的三类标签，未覆盖所有可能子主题；②使用的词典和预训练模型是英语，可能忽略非英语文献；③多标签样本仍存在一定不平衡，导致部分类别的召回率略低；④仅使用SHAP解释，未进一步验证解释的可操作性或与专家判定的一致性。

---

## 315. Universality and Approximation Rates of Graph Neural Networks with Random Features

**arXiv ID:** 2607.26699 | [PDF](https://arxiv.org/pdf/2607.26699v1)

**作者:** Lukas Gonon `[一作]` (University of St. Gallen), Niklas Weber `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明了在给定随机节点特征的前提下，PENN（Permutation-Equivariant Neural Network）可以近似任何可测的置换不变或置换等变函数，实现了对图数据的普适逼近；

**💡 创新点**

创新点在于：①提出了对PENN的全新通用逼近定理，覆盖可测、非连续函数；②将随机节点特征与PENN结合，突破1‑WL测试的表达局限；③给出了逼近速率与网络复杂度的上界；

**🔧 技术方法**

主要技术包括：定义PENN架构并加入随机特征；利用测度论、可测函数逼近与深度前馈网络的普适逼近定理；构造随机特征以实现节点唯一性；推导逼近率与层深、权重数量的关系；

**📊 数据集**

文章为理论研究，未使用具体数据集；

**📈 对比分析**

本研究仅提供理论证明与复杂度分析，未进行实验对比；

**⚠️ 局限性**

局限性在于：①需随机特征提供节点唯一性，可能破坏路径级置换等变性；②理论逼近率受维度灾难影响，实际训练效果仍需验证；

---

## 316. When Does Span-Guided Detoxification Help? Human Preferences and Evaluator Diagnostics in a Controlled Comparison

**arXiv ID:** 2607.26795 | [PDF](https://arxiv.org/pdf/2607.26795v1)

**作者:** Kyungwon Park `[一作]` `[通讯]` (Yonsei University), Kyungwon Park (Yonsei University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 span‑guided（局部编辑）与 unguided（全句重写）两种文本去毒策略，在 60 条混合来源的英语样本上进行对比研究，评估它们在意义保留和毒性降低方面的效果。

**💡 创新点**

首次系统化展示两种策略在不同“强度”样本中的互补失败风险，并指出自动评估指标与人类判断之间的分层差异，提出评估指标应分层报告且要单独关注残余危害与过度修改。

**🔧 技术方法**

使用了 Qwen2.5‑7B‑Instruct 生成模型、Perspective API 的毒性评分、BERTScore 语义相似度、Q_λ 线性组合评估，以及 GPT‑4o 与 GPT‑4o‑mini 作为 LLM 评估者。

**📊 数据集**

采用了两类数据集：30 条手工策划的文本（避免明显 slur）和 30 条来自 HateXplain 的测试项（含 gold rationale spans），以及额外的 300 条 HateXplain 负样本用于自动评估。

**📈 对比分析**

通过 1,860 条盲评估、自动代理评分和 LLM 评估三种方法进行比较。人类结果显示总体上 unguided 更受青睐，但在“强”样本中两者几乎持平；在“弱”样本中 unguided 占优势。自动代理和 LLM 评估能复现整体趋势，但无法捕捉人类评估的分层差异。

**⚠️ 局限性**

局限包括：仅使用单一生成模型；span 信息为 gold oracle；样本仅限英语，未跨语言验证；缺乏对预测 span 的误差分析；未构建通用严重度标度，故无法直接用于部署策略路由。

---

## 317. The Price of Meaning: Quantifying Semantic Communication Overheads in Practice

**arXiv ID:** 2607.26764 | [PDF](https://arxiv.org/pdf/2607.26764v1)

**作者:** Xinyi Lin `[一作]` (Toshiba Europe Limited), Adnan Aijaz `[通讯]` (Toshiba Europe Limited)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一套统一的频谱与能耗开销感知框架，用以评估语义通信在不同部署场景（点对点、UE‑gNB 上行、gNB 多用户下行、侧链以及路由 UE‑UE）下的实际效益，并给出了显式的突破点条件。

**💡 创新点**

创新点在于将语义元数据、协议控制、反馈、模型/知识库同步及神经网络推理能耗等所有实用开销纳入同一解析式，并通过闭式表达式揭示了 Payload 大小、压缩因子 ρ、模型重用因子 N 与频谱/能耗之间的权衡。

**🔧 技术方法**

主要使用了信息理论与通信系统模型（NR Uu、PDSCH/PSSCH 等物理层资源描述）、能耗模型（MAC 运算、内存访问、同步能耗）以及仿真框架来计算各场景下的总频谱与能耗成本。

**📊 数据集**

并未使用真实数据集，而是基于文中给出的仿真参数（Payload 长度、频谱效率、固定协议开销、能耗系数等）进行数值评估。

**📈 对比分析**

通过与传统比特级传输在相同任务效用下的频谱/能耗比值对比，结果显示：短包语义通信往往不具备优势，频谱优势需要 Payload 超过一定阈值；能耗优势的阈值更高；多用户下行在共享语义表示与同步开销时尤其显著。

**⚠️ 局限性**

主要局限在于：仅通过仿真验证，缺乏真实网络实验；对模型同步频率、信道条件等参数假设较为理想；并未考虑动态任务切换导致的元数据频繁更新对开销的进一步影响。

---

## 318. StatePlay: State-Aware Game World Models for Mechanics-Consistent Generation

**arXiv ID:** 2607.26754 | [PDF](https://arxiv.org/pdf/2607.26754v1)

**作者:** Zijun Lin `[一作]` (Tencent), Yeying Jin `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了StatePlay，一种联合预测游戏帧和内部状态的游戏世界模型。

**💡 创新点**

首次在游戏世界模型中显式加入状态预测并用混合Transformer架构实现跨模态交互。

**🔧 技术方法**

采用噪声扰动的状态和视觉编码、Mixture-of-Transformers主干、Smooth L1回归损失以及流匹配视觉损失。

**📊 数据集**

构建了包含5秒街头霸王3游戏录像、动作、精确状态以及NPC策略的10,000条平衡样本数据集。

**📈 对比分析**

在零射与微调两种设定下，与多种现有游戏世界模型比较，StatePlay在机制忠诚度上提升18.6%，在状态对齐上平均误差0.06，视觉质量与动作可控性保持相当。

**⚠️ 局限性**

缺乏跨游戏通用的状态标注数据，且在多机制并发时仍出现视觉细节不一致。

---

## 319. CaIRec: Calibrated Modality Imputation for Incomplete Multimodal Recommendation

**arXiv ID:** 2607.26720 | [PDF](https://arxiv.org/pdf/2607.26720v1)

**作者:** Ruiyu Liu `[一作]` (Southern University of Science and Technology), See-Kiong Ng `[通讯]` (National University of Singapore)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段框架（Structural Imputation Calibration 与 Preference-oriented Representation Calibration），解决不完整多模态推荐中缺失模态导致的交叉模态结构扭曲与偏好适配缺口。

**💡 创新点**

创新点包括：① 共享潜在变量重建缺失模态并通过谱正则化和对应校准来纠正跨模态关系；② 通过伪缺失实例对齐恢复路径并构建完整感知图，提升推荐空间中的适配性；③ 先完成结构校准再进行推荐适配的顺序训练策略。

**🔧 技术方法**

采用共享潜在变量模型 + 结构正则化（谱正则 + 对应校准）、伪缺失对齐、完整感知图（融合协同与内容相似）、LightGCN传播与BPR损失优化。

**📊 数据集**

使用 Amazon Clothing、Sports、Beauty 三个数据集，包含视觉CNN 4096 维特征和文本句子 Transformer 384 维特征。

**📈 对比分析**

与 LightGCN、SimGCL（CF基线）、VBPR、BM3、PGL、MIG-GT（通用多模态）、I^3MRec、MILK（鲁棒方法）以及 DGMRec、HEAT（完成方法）进行对比；在 Recall@10/20、NDCG@10/20 上均实现显著提升，平均提升 3.77%，尤其在缺失项上表现尤为突出。

**⚠️ 局限性**

局限性：实验仅在 Amazon 视觉+文本数据上进行，缺失模式人为生成；未考察多于两种模态或真实缺失场景；未来需验证在更广泛域与自然缺失模式下的效果。

---

## 320. Not In My Git Yard: Catching Backdoors at Commit and Release Time

**arXiv ID:** 2607.26719 | [PDF](https://arxiv.org/pdf/2607.26719v1)

**作者:** Dimitri Kokkonis `[一作]` (Université Paris-Saclay), Stefano Zacchiroli `[通讯]` (LTCI, Télécom Paris)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种自动化检测代码级后门的工具Lily，能够在CI流水线和发行包验证阶段通过灰盒fuzzing与运行时系统调用监控并发掘潜在后门。

**💡 创新点**

创新点在于将新旧版本的系统调用行为进行双重比对：既检测新出现的行为，又验证其与旧版本的显著差异，从而显著降低误报并精准定位后门代码。

**🔧 技术方法**

技术手段包括AFL++灰盒fuzzing、strace系统调用跟踪、基于行为图谱的标准行为识别、以及将静态差分与动态行为关联的可疑代码追踪。

**📊 数据集**

实验使用了ICSE'25发布的后门基准集（17个真实项目+合成后门），共构造了545个合法/恶意提交与发行对，评估覆盖了数十万行代码改动。

**📈 对比分析**

与现有的Git历史ML检测和二进制后门扫描工具相比，Lily在20次10分钟fuzzing实验中，提交级别检测率达90%，发行级别83%，误报率仅0.2%/4.3%，并且可在一次运行中将关注范围压缩至十行以内。

**⚠️ 局限性**

局限性包括：依赖于有效的fuzzing harness 与系统调用覆盖，覆盖不足可能导致漏报；高复杂度或多步骤后门仍有被绕过的风险；在极大规模变更时，硬化模式会提升误报率，需要在实战中权衡。

---

## 321. PowerAtlas: Towards Electricity-Computing Co-Scheduling for Power Systems

**arXiv ID:** 2607.26710 | [PDF](https://arxiv.org/pdf/2607.26710v1)

**作者:** Kaiwen Jiang `[一作]` (Beijing University of Posts and Telecommunications), Haoran Luo `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于LLM的电力‑计算协同调度框架PowerAtlas，能够一次性生成既满足电网安全约束又满足计算任务服务等级的联合调度计划。

**💡 创新点**

通过将预训练LLM先在已求解实例上warm‑start，再使用可检验物理约束的FA‑GRPO强化学习对齐，构建了可在约束满足和经济性上与前沿模型竞争的协同调度模型；同时构造了含oracle解的ECCS基准ECBench。

**🔧 技术方法**

使用大语言模型（如Llama、Qwen、GPT等）、链式思维与结构化输出、物理一致性检验器、检索增强以及Feasibility‑Aware Group Relative Policy Optimization（FA‑GRPO）等技术。

**📊 数据集**

基于RTS‑GMLC电网、真实数据中心运维数据合成2,000个ECCS实例，形成ECBench基准，其中每个实例都配备了Gurobi求解得到的最优解和评估器。

**📈 对比分析**

与8个前沿LLM及3个无训练基础模型在ECBench上进行零样本推断对比，PowerAtlas将平均电网违规从数千MWh降至约200MWh，成本提升至最优解的1.1–1.5倍，推理时间仅为一次前向传播，显著优于传统求解器和未训练LLM。

**⚠️ 局限性**

仍存在约200MWh的平均违规主要集中在峰值时段，未能完全消除失负荷；对低优先级任务的舍弃导致部分服务等级下降；对不同地区电网和数据中心规模的泛化尚待进一步验证。

---

## 322. TPD: Temporal Prior Decoupling for Text-to-Video Diffusion Models

**arXiv ID:** 2607.26706 | [PDF](https://arxiv.org/pdf/2607.26706v1)

**作者:** Taewon Kang `[一作]` (University of Maryland at College Park), Matthias Zwicker `[通讯]` (University of Maryland at College Park)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究文本到视频生成中的时间先验抑制问题，并提出一种在推理时恢复被压制的晚期事件的 Temporal Prior Decoupling 方法。

**💡 创新点**

将晚期事件缺失建模为低阈值可行性约束，利用时间反事实构造抑制方向并通过最小能量投影恢复语义，形成与上界抑制、漂移去除互补的下界恢复框架。

**🔧 技术方法**

在预训练的文本到视频扩散模型基础上，使用 LLM 对提示进行早晚段分解，构造时间反事实，计算抑制方向并做最小能量投影，完全在无训练的 classifier‑free guidance 空间内实现。

**📊 数据集**

构建了包含八类出现机制（实体出现、聚集、现象启动、因果转化等）的单句未分段提示评测集，用于量化晚期概念实现和抑制率。

**📈 对比分析**

与 Mochi、HunyuanVideo、CogVideoX 等主流扩散模型基线以及尺度提升、Always‑Add 等对照进行定量和定性对比，TPD 在 CLIPScore、TCS、TVR、EPS 等指标上均优于基线，显著降低晚期抑制并保持早期场景完整。

**⚠️ 局限性**

仍需依赖 LLM 进行提示分解，处理长句或复杂结构时可能不稳定；对极端抑制程度仍有限；仅在无训练的推理阶段改进，无法根本解决模型内在的先验学习偏差。

---

## 323. Vision-TL-Action: Neuro-Symbolic Trajectory Generation from Visual Observations and Temporal Logic

**arXiv ID:** 2607.26770 | [PDF](https://arxiv.org/pdf/2607.26770v1)

**作者:** Zezhi Liu `[一作]` (Nankai University), Yongchun Fang `[通讯]` (Nankai University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 Vision-TL-Action，一种从多视角图像和无量纲 TL 图生成满足时序逻辑约束的机器人轨迹的方法，消除了对精确几何信息的需求。

**💡 创新点**

创新点包括：①将 TL 图与视觉信息通过双向交叉注意力融合；②引入训练时的谓词‑对象对齐损失实现节点级语义绑定；③采用 Flow‑Matching 的 Temporal U‑Net 在无几何条件下生成轨迹。

**🔧 技术方法**

技术包括：对象身份嵌入、图卷积编码、地址化视觉 token、双向交叉注意力、谓词‑对象对齐损失、Flow‑Matching Trajectory Generation（Temporal U‑Net）。

**📊 数据集**

使用了 Panda 机器人抓取/避障任务数据集（71k 轨迹）和 AntMaze 导航数据集（约 50k 轨迹），均包含多视角图像和对应的 STL 规范。

**📈 对比分析**

通过 Success@K（候选集合覆盖率）与 oracle‑state（精确几何）对比。Vision‑TL‑Action 在大候选数下在 Panda 上达到 67.45%（比 oracle‑state 高 8.3%），在 AntMaze 上 96.35%（与 oracle‑state 差距不足 0.5%）。单样本性能低于 oracle‑state，但候选覆盖率优势显著。

**⚠️ 局限性**

局限性：训练时仍需使用对象投影的监督；评估基于仿真和合成图像；未提供闭环重规划或真实机器人验证；缺乏候选轨迹排序器，实际执行效果仍需进一步研究。

---

## 324. Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM

**arXiv ID:** 2607.26773 | [PDF](https://arxiv.org/pdf/2607.26773v1)

**作者:** Huixiang Zhang `[一作]` (Georgia Institute of Technology), Mahzabeen Emu `[通讯]` (Memorial University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种因果审计框架，用于在LLM多智能体系统中检验潜在通信是否真正被接收者使用；

**💡 创新点**

设计四种消息设置（无消息、当前示例、其他示例、自生成）和五个度量（PS、PL、CIC、CAG、SSG）来分解消息影响，并通过组件恢复与等价性检验实现对通信机制的细粒度识别；

**🔧 技术方法**

在发送者–接收者边界执行信息替换干预，利用概率加权嵌入/隐藏状态/KV缓存作为载体；使用Jensen–Shannon散度、互信息下界、任务准确率等统计量；对Qwen3-4B/8B模型进行多步推理实验；

**📊 数据集**

使用GSM8K、ARC‑Challenge（ARC‑C）和MATH‑500三大数据集，涵盖算术推理、多项选择科学问答和中等难度数学题；

**📈 对比分析**

通过对比无消息、当前示例消息、其他示例消息和自生消息，计算PL、CIC、CAG、SSG及整体效果；结果表明整体准确率可能掩盖相反的子效应，且CAG与SSG在不同模型/任务下方向不一，单一准确率不足以评估潜在通信；

**⚠️ 局限性**

审计对计算资源要求高，需在消息维度/长度匹配上做额外处理；互信息估计为下界，可能低估编码信息；仅针对概率加权嵌入/隐藏状态/KV缓存，未覆盖更广泛的潜在通信方式；结果依赖模型实现与任务选择。

---

## 325. Metis: Memory Foundation Model

**arXiv ID:** 2607.26760 | [PDF](https://arxiv.org/pdf/2607.26760v1)

**作者:** Zeyu Zhang `[一作]` (MemTensor (Shanghai) Technology Co., Ltd.), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了内置内存的“memory foundation model”概念，并实现了第一个原型模型 Metis，能够在 Transformer 前向计算中直接维护并操作动态记忆状态。

**💡 创新点**

创新点在于将记忆状态与模型参数融合成可训练的动态参数（native memory state）并通过可学习的记忆操作（native memory procedure）实现记忆的记、忘、改、回溯；同时引入了全新记忆块结构（Metis block）和多任务训练目标，突破了传统外部 RAG/记忆网络的离散化与非端到端优化瓶颈。

**🔧 技术方法**

主要技术包括 Fast Weight Programming 的密集记忆网络、Gated Delta Update (GDU) 的更新机制、可学习的查询投影与归一化、以及记忆重建、操作与正则化三大训练目标的联合优化。

**📊 数据集**

构造了大规模自监督记忆数据集：基于 27 个公开 benchmark（如 LoCoMo、LongMemEval、NeedleInAHaystack 等）合成的主数据（主记忆操作样本）与辅数据（多实体、记忆污染场景），共计约 406M tokens。

**📈 对比分析**

与 Qwen3.5 原始模型、RAG、Temp‑LoRA、δ‑Mem 等基线进行对比，在无上下文（native memory）设置下在 MemOps、LoCoMo (Gold)、NextMem、Metis 自测集上平均提升 10–30%（Metis‑27B 在无上下文条件下常取得最高分），并在 OOD Benchmarks（ATM‑Bench、MemDaily）保持竞争优势。

**⚠️ 局限性**

局限性包括：固定尺寸的记忆参数导致长时序信息衰减与干扰、对语义相近事实的混淆、缺乏对极长交互的可扩展性，以及在一些任务中仍需外部记忆或更强的控制机制。

---

## 326. CalTwin: Towards Calibrated, Shift-Robust Medical World Models via Fisher-Information Regularisation

**arXiv ID:** 2607.26752 | [PDF](https://arxiv.org/pdf/2607.26752v1)

**作者:** Behraj Khan `[一作]` (Institute of Business Administration Karachi), Tahir Qasim Syed `[通讯]` (Institute of Business Administration Karachi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种名为 CalTwin 的正则化方法，用于提升医学世界模型（latent state + transition predictor）的跨医院协变移鲁棒性和多步预测的置信度校准。

**💡 创新点**

创新点在于将 Fisher 信息矩阵正则（FIcsR）与 Confidence Misalignment Penalty（CMP）结合到连续潜在转移预测器中，并通过分片式训练实现对跨片段协变移的在线自适应正则化；首次将这两种正则同时应用于医学世界模型的自回归预测任务。

**🔧 技术方法**

使用技术包括：1）Empirical Fisher Information Matrix 及其指数移动平均累积，用于参数空间的正则化；2）CMP 对离散辅助风险头的概率分布进行校准惩罚；3）GRU‑based 潜在转移预测器与线性辅助头；4）分片式（fragmented）训练框架。

**📊 数据集**

数据集：PhysioNet 2019 Sepsis Challenge（两医院 A/B 的 ICU 生命体征序列）和 eICU‑CRD Demo（20 家医院的 ICU 生命体征），分别用作跨医院协变移的评估。

**📈 对比分析**

比较方法：在同一实验设置下对 Baseline（无惩罚）、FIM‑only、CMP‑only、CalTwin 四种训练策略进行对比。结果显示：在 PhysioNet 上 CalTwin 在 OOD 下的下一步 MSE 降低 9.1%（FIM‑only 7.0%），ECE 降低 0.7%（CMP‑only 1.3%）；在 eICU‑CRD Demo 上 CMP‑only 与 CalTwin 的表现互相转置，说明不同数据集对正则化效果的敏感性。

**⚠️ 局限性**

局限性：仅使用单个随机种子；评估为教师强迫（teacher‑forced）而非闭环自回归预测；CMP 采用离散辅助头，未验证对连续输出的直接校准方法；Empirical Fisher 的近似在不同分片策略下可能不稳定；缺乏多数据集、多种子以及超参调优的系统验证。

---

## 327. Sequence-SOD: Bio-inspired Sequence-aware Spiking ObjectDetection for Event Cameras

**arXiv ID:** 2607.26703 | [PDF](https://arxiv.org/pdf/2607.26703v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 328. Verifiable Random Sampling

**arXiv ID:** 2607.26734 | [PDF](https://arxiv.org/pdf/2607.26734v1)

**作者:** Yeoh Wei Zhu `[一作]` (JPMorganChase), Ruslan Shaydulin `[通讯]` (JPMorganChase)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种可公开验证的随机采样协议（VRS），利用量子随机电路采样（RCS）实现从任意目标分布生成新鲜且可验证的随机样本；

**💡 创新点**

①首次定义并构造VRS，使随机性非确定性且对抗协同攻击；②在量子硬件上实现可公开验证的采样；③将区块链时间戳与挑战生成结合，提升抗预处理和延迟攻击能力；

**🔧 技术方法**

量子随机电路采样（RCS）、交叉熵基准测试（XEB）、随机数提取器（quantum-proof strong two-source extractor）、区块链（批量公告板）、构造密码学（CC）框架；

**📊 数据集**

未使用传统机器学习或图像数据集；主要使用随机生成的量子电路集合（Haar随机电路）和区块链区块哈希作为熵源；

**📈 对比分析**

相较于传统VRF、随机信标、PUF及其他认证随机性方案，VRS在可验证性、抗预处理、分布灵活性和组合性上显著优越；实验表明可在现有噪声量子设备上实现，并通过XEB测试验证真随机性；

**⚠️ 局限性**

依赖当前噪声量子设备的深度与错误率；对抗模型仅覆盖有限的攻击者；需要区块链或公告板来提供时间戳和熵；未来需迁移至容错量子计算机并完善安全分析。

---

## 329. Affective Tools for Thought: Towards Shared Attention and Affective Reorienting in AI-Supported Thinking

**arXiv ID:** 2607.26731 | [PDF](https://arxiv.org/pdf/2607.26731v1)

**作者:** Yifu Liu `[一作]` (UCL Interaction Centre), Nadia Bianchi-Berthouze `[通讯]` (UCL Interaction Centre)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在工具思维（Tools for Thought）框架中，将情感视为认知的构成支架，基于触摸感知聊天机器人在工艺修补实验中验证情感再定向的可行性。

**💡 创新点**

创新点在于：①提出情感为认知的根本构成；②识别共享注意与情感再定向两大障碍；③提出链式情感、情感镜像与提示再定向三种设计策略，改变传统情感作为噪声或可调信号的视角。

**🔧 技术方法**

采用触摸感知聊天机器人（Wizard‑of‑Oz）+情感检测与推理链（Chain of Emotion）+基于情绪轨迹的响应生成技术。

**📊 数据集**

使用单一实验收集的用户‑系统交互数据：一组参与者在修补衣物任务中与触摸感知聊天机器人交互，形成的实验日志与访谈记录。

**📈 对比分析**

未与传统 TfT 进行数值性能对比，而是通过轨迹偏离度、情感转变频率等定性指标评估，实验表明情感镜像与提示再定向能促成新任务轨迹而非单纯提高完成率。

**⚠️ 局限性**

局限包括：单一工艺修补场景；人工操纵（Wizard‑of‑Oz）非自动化实现；缺乏跨文本或其他 TfT 的验证；对情感建模的隐私、可解释性与长期效应未充分探究。

---

## 330. FARI: Robust One-Step Inversion for Watermarking in Diffusion Models

**arXiv ID:** 2607.26723 | [PDF](https://arxiv.org/pdf/2607.26723v1)

**作者:** Jindong Yang `[一作]` (University of Science and Technology of China), Kejiang Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了FARI，一种单步逆向扩散方法，用于加速并提升扩散模型水印提取的鲁棒性。

**💡 创新点**

创新点在于发现逆向轨迹曲率远低于生成轨迹，从而实现低NFE单步逼近，并结合LoRA微调实现端到端对抗训练，兼顾速度与鲁棒性。

**🔧 技术方法**

核心技术包括扩散模型DDIM逆向、单步轨迹蒸馏、低秩适配器LoRA以及对抗数据增强训练。

**📊 数据集**

使用的公开数据集有MS‑COCO‑2017用于训练、Stable‑Diffusion‑Prompts（SDP）用于评测，模型基于Stable Diffusion v1.5与v2.1。

**📈 对比分析**

与50步DDIM、EDICT、BELM、AMED、LCM‑LoRA、DMD2等多种基线比较，FARI在所有主流失真（JPEG、裁剪、模糊、噪声等）下均实现了最高的水印提取准确率，并在单步推理上仅消耗约1步NFE，速度提升显著。

**⚠️ 局限性**

局限性包括对干净图像逆向精度下降，不适合图像编辑任务；且依赖ODE采样，若使用SDE采样则失效。

---

## 331. MPEcho: A Melody and Phoneme-Aware Generative Framework for Controllable Cover Song Generation

**arXiv ID:** 2607.26698 | [PDF](https://arxiv.org/pdf/2607.26698v1)

**作者:** Wei-Jaw Lee `[一作]` (National Taiwan University), Yi-Hsuan Yang `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为MPEcho的全歌生成模型，结合了歌词到歌曲（LTS）的ACE-Step骨干网络、歌声F0提取以及新增的音素编码器与长度调节器，实现了对歌词音素层级的精细控制；同时开发了Phonsa自动音素对齐系统，为MPEcho提供高精度的音素时序信息。

**💡 创新点**

创新点在于首次将SVS（声乐合成）中的音素层级条件与长度调节直接迁移到完整歌曲生成任务；通过音素编码器与长度调节器实现音素时序精准约束，显著提升了歌词准确性；Phonsa则突破传统强制对齐的精度瓶颈，提供了高质量的音素时序标注。

**🔧 技术方法**

技术上使用ACE-Step DiT+IA‑EiLM适配器作为生成框架，结合RMVPE歌声F0提取器；MPEcho加入音素编码器、长度调节器及多条件CFG/ APG推理策略；Phonsa采用Chunked Self‑Attention结构、CTC+CE联合损失和Whisper编码器/解码器微调实现音素对齐与分割。

**📊 数据集**

Phonsa在M4Singer和Opencpop（中文）数据集上训练，测试使用GTsinger；MPEcho在内部约13,045首中文流行/传统歌曲（约1,427小时）上训练，验证集131首。

**📈 对比分析**

与SongEcho（仅音高约束）和仅音素约束的MPEcho进行对比，采用Audiobox、CLAP、RPA/RCA、PER等指标；MPEcho在PER从约45.62%下降到18.65%，同时保持或提升音高准确率和音乐美感；Phonsa相较MFA将MAE从233.9 ms降至32.6 ms，PPO和PCAS显著提升。

**⚠️ 局限性**

目前仅支持单声道单歌手生成，缺乏多歌手、多语言音素建模以及更丰富的韵律控制，未来工作将聚焦这些方向。

---

## 332. A Low-Power Sparse Convolution Accelerator with Idle-First-Task-Assignment for Edge Vision

**arXiv ID:** 2607.26835 | [PDF](https://arxiv.org/pdf/2607.26835v1)

**作者:** Jingyue Zhuge `[一作]` (Dresden University of Technology), Christian Mayr `[通讯]` (Dresden University of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

设计并实现了一款面向边缘视觉场景的低功耗稀疏卷积加速器。

**💡 创新点**

主要创新点包括：①采用Bitmap压缩格式降低传输与存储开销；②提出Idle-First-Task-Assignment (IFTA) 动态调度策略，显著缓解稀疏计算中的处理单元负载不均衡；③为深度可分离卷积（DWConv）定制了专属数据流和缓冲机制。

**🔧 技术方法**

使用的技术包括：Bitmap稀疏压缩、双路位图解码器、二维PE阵列、IFTA调度、专门的DWConv数据流、16 nm工艺实现以及多级缓存与缓冲设计。

**📊 数据集**

实验基于 ImageNet 数据集，对 VGG16（68%稀疏）和 MobileNetV2（35%稀疏）进行剪枝、微调后进行推理评估。

**📈 对比分析**

与传统稠密加速器和 SparTen 的对比显示：在 VGG16 上获得 6.5× 加速、在 MobileNetV2 上获得 2.8× 加速；PE 利用率平均超过 90%，芯片面积仅 0.5 mm²，功耗 12–16 mW，能耗效能（帧/焦）在 5435–8152 之间，明显优于同类稀疏加速器。

**⚠️ 局限性**

局限性：Bitmap 格式在稀疏度超过 90% 时效率下降；设计规模较小，仅在单核/小规模阵列下实现，难以直接扩展至更大规模的多核/多片加速器；缺乏对极高稀疏度网络的专门优化。

---

## 333. A Scalable AI-Powered System for Explainable Machine Learning Pipelines in Brain Tumor

**arXiv ID:** 2607.26834 | [PDF](https://arxiv.org/pdf/2607.26834v1)

**作者:** Yin Lin `[一作]` (Polytechnic University of Milan), Simona Ferrante `[通讯]` (Polytechnic University of Milan)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一个可扩展的 Web 视觉分析系统，用于支持脑瘤放射组学机器学习推断，集成了队列管理、特征提取和模型推断。

**💡 创新点**

创新点在于将整个放射组学-机器学习流程可视化、可追溯，并在推断阶段加入了安全防护和中间产物公开。

**🔧 技术方法**

技术上采用 Dash 框架搭建轻量化前端，PyRadiomics 进行特征提取，集成预训练模型并使用 SHAP 等可解释性方法。

**📊 数据集**

使用公开的 BraTS 2020 数据集和来自 Fondazione IRCCS Istituto Neurologico Carlo Besta 的本地颅底脑膜瘤临床队列。

**📈 对比分析**

通过在两组数据上进行模型推断验证，并与已有公开基准对比，结果显示推断速度可接受且解释性充分，但缺乏大规模性能评估。

**⚠️ 局限性**

局限包括仅支持 CSV/XLSX 和 NIfTI 格式，特征提取速度慢，功能仍不完整，需要进一步迭代。

---

## 334. CinemaTraj: Composing Atomic Camera Trajectories for 3D Scenes with LLM Agents

**arXiv ID:** 2607.26910 | [PDF](https://arxiv.org/pdf/2607.26910v1)

**作者:** Qianru Li `[一作]` (Technical University of Munich), Yanfeng Zhang `[通讯]` (Huawei Dresden Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将摄像机轨迹规划视为基于自然语言的空间推理问题的方法，利用LLM与3D场景图来生成符合语义且无碰撞的电影级轨迹并同步生成语音解说与字幕。

**💡 创新点**

创新点在于：①为LLM提供结构化3D场景图作为空间先验，使其同时理解场景几何与电影语义；②使用基于专业摄像动作的可参数化轨迹模板，既保留电影动效又易于梯度优化；③在签名距离场上实现可微碰撞与遮挡优化，确保轨迹物理可行。

**🔧 技术方法**

技术包括：LLM（GPT‑4.1）用于提示翻译与字幕/语音生成；3D场景图构建（基于RGB‑D的实例分割+结构边缘）；可参数化轨迹库（orbit、dolly、crane、pan/tilt、zoom、static）；SDF构造与两阶段梯度优化；可视化与视频合成。

**📊 数据集**

使用ScanNet++室内真实3D重建数据集，50个场景，采用全指定、部分指定与开放式三种提示级别进行实验。

**📈 对比分析**

与ChatCam+GenDoP（无3D图）和CCTG（仅几何优化）对比，评估指标包括Motion MSE、CLaTr Score、Collision Rate、Occlusion Rate、Object Coverage。结果显示，本方法在Motion MSE、CLaTr Score、Object Coverage上均优于基线；碰撞率略高于无Anchor选择的变体，但总体安全性和语义对齐显著提升；用户研究亦显示在提示对齐、碰撞/遮挡避免和电影质量上获最高分。

**⚠️ 局限性**

局限性包括：仅支持静态场景的单一轨迹生成，无法处理动态角色或多镜头剪辑；轨迹库仅涵盖基本动作，缺乏跟随/点视角等高级镜头；缺少对剪辑语法（180°规则、镜头逆序等）的考虑，无法直接生成完整的电影剪辑流程。

---

## 335. From Passive Video to Editable Experience: Physically Grounded Experience Synthesis for Embodied Intelligence

**arXiv ID:** 2607.26903 | [PDF](https://arxiv.org/pdf/2607.26903v1)

**作者:** Jia Luo `[一作]` `[通讯]` (Huazhong University of Science and Technology), Jia Luo (Huazhong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Pegasus框架，将人类第一人称视频通过任务图、抓取潜能图和约束图等结构化表示，转化为可执行的机器人演示；

**💡 创新点**

创新点在于采用层次化抓取潜能空间来消除对象身份偏差、使用图结构化跨体体现象推理以及闭环物理验证机制保证运动学可行性；

**🔧 技术方法**

技术手段包括任务图提取、层次化抓取潜能编码、对齐对比损失、约束图与执行图生成、闭环物理验证以及视频生成模型Wan2.1-1.1-T2V；

**📊 数据集**

实验数据集涵盖GTEA Gaze+、Epic‑Kitchens‑100、1000条自建Internet Manipulation Video集，以及五种不同机械臂的机器人环境；

**📈 对比分析**

与仅用文本提示的基线相比，任务正确率提升12.3%、可执行率提升15.7%；生成数据辅助训练可使OpenVLA+ACT在Franka Panda上的成功率提升11.3%；与人类视频特征相比，生成数据提高约25.8%；闭环验证将可执行率从51.2%提升至94.1%；

**⚠️ 局限性**

局限性包括对VideoLLM任务分解的依赖、物理验证仅覆盖运动学不涉及完整接触动力学、目前仅单视角生成，且多视角、多机器人及更复杂任务的泛化尚未验证。

---

## 336. Hybrid Workflow Composition for Extreme-Scale Data Processing: A Case Study on the HL-LHC (Extended Version)

**arXiv ID:** 2607.26877 | [PDF](https://arxiv.org/pdf/2607.26877v1)

**作者:** Alan Malta Rodrigues `[一作]` (University of Notre Dame), Douglas Thain `[通讯]` (University of Notre Dame)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了可扩展的 DAGFlowSim 框架，系统枚举所有有效的任务集分组方案，并通过离散事件模拟评估高吞吐量科学工作流在不同资源与网络条件下的性能。

**💡 创新点**

创新点在于：① 通过硬件兼容性约束对 DAG 任务集进行完整分组枚举；② 引入多目标评分函数，将吞吐量、CPU/内存利用率与网络 I/O 统一量化；③ 公开完整的模拟器、分析脚本与结果数据，形成可复现的工作流合成方法。

**🔧 技术方法**

使用技术包括任务集分组枚举、基于兼容矩阵的硬件约束、离散事件模拟（包含启动开销、I/O 延迟与失败概率模型）、多指标归一化与加权评分、全因子实验（工作流模板、作业长度、失败率、网络带宽）等。

**📊 数据集**

数据集基于 CMS 真实生产工作流的 5 任务集模板（seq_real、seq_homo、seq_hetero），并将 7,680 次模拟结果与源代码统一发布到公开仓库。

**📈 对比分析**

比较方法：对同一工作流模板在 480 种环境组合下进行 7,680 次模拟，分别对比 StepChain 与 TaskChain 极端，计算吞吐量、CPU/内存占用、网络 I/O、成功率等指标；结果显示最优混合方案（Const13）在 12h 目标作业、100 MB/s 带宽、5% 失败率下提升约 31.5% 的事件吞吐、降低 24% CPU 与 39% 内存占用，并将网络 I/O 降低 14.9 倍。

**⚠️ 局限性**

局限性：① 假设工作节点同质且仅考虑单父子 DAG，未覆盖多父/多子结构；② 失败模型简化，未考虑站点相关性与大规模故障；③ 未支持 GPU/FPGAs 等硬件异构；④ 只能给出相对排名，无法直接预测绝对生产性能。

---

## 337. Self-dual double cyclic codes over $\mathbb{F}_q$

**arXiv ID:** 2607.26823 | [PDF](https://arxiv.org/pdf/2607.26823v1)

**作者:** Ricky Aditya `[一作]` (Institut Teknologi Bandung), Djoko Suprijanto `[通讯]` (Institut Teknologi Bandung)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了自对偶双循环码的生成多项式条件，并给出不同长度（如(r,r)、(r,2r)、(2r,r)、(r,s)）下的存在性与构造方法；

**💡 创新点**

提出了任意长度(r,s)和任意有限域上自对偶双循环码的必要充分条件，并揭示其与自对偶循环码、2-准循环码以及自对偶负循环码的关系；

**🔧 技术方法**

主要使用代数编码理论、循环多项式分解、互补性与自共轭性质等代数技术进行分析与构造；

**📊 数据集**

无实验数据集，全部结果来自理论推导与符号计算；

**📈 对比分析**

通过示例演示最优性（最小距离达到理论上限），与现有自对偶码比较表明构造的码在参数上无劣势；

**⚠️ 局限性**

仅在满足特定长度与域条件时可构造，且对非2特征时存在严格限制，部分长度和域无法满足构造条件。

---

## 338. MRCoder: An Efficient Context Selecting Approach for Repository-Level Code Generation

**arXiv ID:** 2607.26805 | [PDF](https://arxiv.org/pdf/2607.26805v1)

**作者:** Peiding Wang `[一作]` (Beihang University), Fang Liu `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种在代码仓库级别生成代码时的上下文选择框架，利用 Map–Reduce 结构对检索到的上下文进行分组、草稿生成与筛选，最终提高生成质量并降低推理成本。

**💡 创新点**

创新点在于提出了结构化草稿引导选择（SADGS）方法，结合 API 调用一致性与逻辑相似度双重信号过滤噪声上下文，并设计了并行验证机制实现高效解码。

**🔧 技术方法**

使用技术包括轻量级草稿 LLM、BM25 检索、API 关系提取、代码逻辑相似度匹配、并行解码与验证等。

**📊 数据集**

采用 CoderEval 和 DevEval 两大 Python 仓库级代码生成基准数据集进行评估。

**📈 对比分析**

与 RAG、RL‑Coder、RepoFormer、LongCodeZip 等基线比较，Pass@1 提升约30%–50%，token 消耗降低30%–50%，推理时间缩短约50%，表现出显著的质量与效率双提升。

**⚠️ 局限性**

局限性包括依赖 BM25 检索、草稿模型规模受限、仅在 Python 语言环境验证，且在更大规模或不同语言的适配性与鲁棒性需要进一步验证。

---

## 339. FedTopo: Relation-Level Topology Sharing for Model-Heterogeneous Federated Learning

**arXiv ID:** 2607.26801 | [PDF](https://arxiv.org/pdf/2607.26801v1)

**作者:** Zhaoyang Ma `[一作]` (Beijing Jiaotong University), Jing Wang `[通讯]` (Beijing Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedTopo 框架，在模型异构联邦学习中以类关系拓扑形式共享全局知识，解决不同模型特征空间不对齐问题。

**💡 创新点**

创新点在于采用关系级别的类关系拓扑而非绝对坐标/参数进行知识共享，并引入可靠性加权与缺失掩码进行全局聚合，同时通过拓扑指导的负样本增强提升本地训练。

**🔧 技术方法**

使用的技术包括关系拓扑构建、可靠性加权全局聚合、Top‑q 负样本选择、EMA 平滑以及拓扑引导的 logits 增强。

**📊 数据集**

实验数据集为 CIFAR‑10、CIFAR‑100 与 Tiny‑ImageNet。

**📈 对比分析**

与参数共享、知识蒸馏、原型共享等异构 FL 基线进行比较，FedTopo 在八种不同骨干网络、Dirichlet 与病态非 IID 划分下均优于基线，平均提升约 1.9–6.0%。

**⚠️ 局限性**

局限性在于通信成本随类别数呈平方增长，并且在极端类别稀疏的场景下仍存在一定不确定性。

---

## 340. ReCo: Reweighting GRPO Against Distributional Concentration

**arXiv ID:** 2607.26862 | [PDF](https://arxiv.org/pdf/2607.26862v1)

**作者:** Junoh Park `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ReCo方法，用于改进GRPO在RLVR中的更新，降低模型在数学推理任务中对高概率路径的过度集中。

**💡 创新点**

通过响应级别的期望出现次数归一化和基于Bernoulli方差的标记级别重权重，解决了GRPO导致的多样性丧失。

**🔧 技术方法**

使用了强化学习与可验证奖励框架下的GRPO与ReCo，结合PPO式的梯度裁剪与KL正则。

**📊 数据集**

在AIME 2025/2024、AMC 2023、OlympiadBench、MATH 500等数学推理基准以及Qwen2.5-Math和Llama-3.1-8B模型上进行实验。

**📈 对比分析**

与GRPO以及基线模型比较，ReCo在大k Pass@k指标上提升了约3–6个百分点，并在多样性指标如Distinct-2与Self-BLEU上显著优于GRPO。

**⚠️ 局限性**

仅针对稀疏二值奖励，未考虑更丰富的中间步骤反馈，且重权重仅在更新后应用，未直接改变采样过程。

---

## 341. NeoRacer: An Open, Standardized 1:12 Scale Autonomous Race Car for Benchmarking and Education

**arXiv ID:** 2607.26855 | [PDF](https://arxiv.org/pdf/2607.26855v1)

**作者:** Koneshka Bandyopadhyay `[一作]` (Neobotics Foundation, Inc.), Renato Mancuso `[通讯]` (Boston University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并发布了NeoRacer，一款1:12比例、开放硬件与软件、预装完成、成本仅约2700美元、可并行运行SLAM与深度学习推理的自主赛车平台；

**💡 创新点**

核心创新包括：统一硬件规格以实现跨机构可比基准、采用高性能NVIDIA Jetson Orin Nano与270° LiDAR、120fps全局快门摄像头的组合、模块化可扩展设计、以及三重标识发布机制保证硬件与软件可追溯；

**🔧 技术方法**

使用技术包括：NVIDIA Jetson Orin Nano (CUDA, cuDNN, TensorRT)、ROS2（DDS通信）、ESP32‑S3 的实时控制板、全局快门摄像头、Richbeam LakiBeam1 LiDAR、9轴 IMU、以及自研的Neobotics Playground 仿真平台；

**📊 数据集**

主要数据来源为车辆自身传感器采集的实时数据以及Neobotics Playground 内置的物理校准轨道仿真数据；未使用公开公开数据集；

**📈 对比分析**

通过在MIT IAP 15名学生的两周快节奏实验，验证硬件在高负载下的可靠性，并在同类平台中实现了67 TOPS的AI运算，较F1Tenth 21 TOPS提升约3倍、成本减半，证明了平台的可行性与性能优势；

**⚠️ 局限性**

局限性包括：当前硬件设计尚未发布完整PCB Gerber文件，平台对高强度长时间竞速的长期可靠性仍待进一步验证，且基准评测主要基于内部仿真与单一机构的 pilot，尚缺乏多机构广泛对比实验。

---

## 342. From Representations to Behaviors: Exploring the Person-Situation-Behavior Triad in LLMs

**arXiv ID:** 2607.26853 | [PDF](https://arxiv.org/pdf/2607.26853v1)

**作者:** Ruikang Zhang `[一作]` (Peking University), Qi Su `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一个基于稀疏自编码器（SAE）的框架，用于在大型语言模型中发现、控制并验证与人格相关的内部表征，并通过情境反应和社交行为两大层面进行交叉验证。

**💡 创新点**

创新点在于将人格三元框架（内部表征–情境表达–行为表现）与SAE特征检索结合，形成从对比行为到可操作特征、再到跨情境一致性和社会行为一致性的一条完整因果链；同时提出了对特征进行对比激活检索、干预推理、词级激活与改写鲁棒性验证的完整实验流程。

**🔧 技术方法**

技术手段包括：稀疏自编码器特征分解、特征层级干预（在残差流加入解码方向）、生成式干预探测、词级激活统计、改写鲁棒性测试，以及与现有的对比均值向量（CAA）和人格提示（P²）等对照方法的比较。

**📊 数据集**

使用的数据集有：Q‑Sort情境语料库与NEO‑PI‑R特征标签（用于构建高低人格对比行为对）、TRAIT人格情境响应基准（用于跨情境表达评估）、SocialEval Interpersonal Ability Evaluation（用于社会行为影响评估）以及模型自身的公开激活日志（用于SAE训练）。

**📈 对比分析**

在TRAIT上，SAE干预实现了正负两端的显著性人格偏移，同时保持了约95%–98%的有效率；在SocialEval上，干预导致与人类人格研究一致的优势‑成本模式，整体表现优于CAA与P²对照组，显示出更稳定的情境连贯性和行为一致性。

**⚠️ 局限性**

局限性包括：特征选取与验证需要人工评判；仅在DeepSeek‑R1‑Distill‑Llama‑8B模型上验证，缺乏跨模型泛化；框架主要针对Big Five维度，可能不适用于更细粒度或不同人格理论；干预强度受模型内部结构限制，可能在其他大模型中出现不稳定或生成失效。

---

## 343. ToxScreen: Detecting Whether an LLM Has Been Poisoned

**arXiv ID:** 2607.26849 | [PDF](https://arxiv.org/pdf/2607.26849v1)

**作者:** Anthony Hughes `[一作]` (University of Sheffield), Andrew Draganov `[通讯]` (Arcadia Impact)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在白盒访问且无训练数据的环境下，构建了大约800个覆盖不同触发器、攻击目标、模型规模与毒化率的LLM后门基准（ToxScreen），并对多种后门检测/触发器恢复方法进行实验。

**💡 创新点**

创新点包括：①首次公开如此大规模、覆盖度高的后门模型基准；②证明基于词表检索的token look‑up能在绝大多数情况恢复单词触发器，而梯度优化方法无效；③提出二阶曲率（token‑parameter coupling）来区分后门触发器与普通 jailbreak；④揭示后门对TruthfulQA等可信度评测产生明显异常，可作为附加检测信号。

**🔧 技术方法**

使用了白盒权重访问、ASR评估、梯度基准（GCG、AG‑GCG）、二阶导数曲率分析、结构化与非结构化剪枝实验、ghost 正则化训练以及多模型多规模的实证评测。

**📊 数据集**

数据集包括：从公开语料构建的 Poisoned dataset（HarmBench、BeaverTails、Alpaca、StrongREJECT、MaliciousInstruct、JailbreakBench），以及用于通用性能评估的 ARC、TruthfulQA、HellaSwag、Winogrande、MMLU。

**📈 对比分析**

实验对比 token look‑up 与梯度优化，发现 token look‑up 在 14/28 个实验单元中能恢复触发器并在大多数规模模型上取得高 ASR；梯度优化始终收敛到通用 jailbreak 侧写；后门模型在 TruthfulQA 上的偏移明显可观，进一步证明其可检出性。

**⚠️ 局限性**

局限性：仅针对单词/短语触发器，无法覆盖更复杂的语义或上下文依赖触发；基准构造为实验性后门，缺少真实攻击场景的多样性；评测仅在白盒权重访问下进行，未考察黑盒或有限查询场景；ghost 训练对大规模模型失效且计算成本高。

---

## 344. Tight Generalization Bound for AdaBoost

**arXiv ID:** 2607.26838 | [PDF](https://arxiv.org/pdf/2607.26838v1)

**作者:** Mikael Møller Høgsgaard `[一作]` `[通讯]` (University of Oxford), Mikael Møller Høgsgaard (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文通过证明AdaBoost的投票分类器在零经验γ/2边缘损失时，能够得到一个新的基于边缘的泛化误差上界，并以此给出AdaBoost总体泛化误差的上界。

**💡 创新点**

创新点在于提出了针对投票分类器的全新基于边缘的泛化误差上界，该上界与已有下界匹配，完成了AdaBoost泛化误差的理论收敛速率的精确表述。

**🔧 技术方法**

主要使用了VC维理论、Rademacher复杂度以及鬼样本技术，对投票函数集合的VC维进行细致计数，并利用AdaBoost的边缘性质完成上界推导。

**📊 数据集**

该工作为理论性质研究，未使用任何具体实验数据集；所有结论均基于概率分布假设和抽样理论。

**📈 对比分析**

与之前的上界相比，新上界的乘子不再包含次对数项，达到了最优Θ(d/(nγ²)+ln(1/δ)/n)的收敛速度；与先前已知的下界完全匹配，表明界限在常数项上是最优的。

**⚠️ 局限性**

局限性包括：仅适用于二分类的投票分类器；上界中常数未给出具体数值；假设弱学习器满足γ-弱学习性质，实际算法实现时可能不易满足；对多类别或回归等其他任务尚未扩展。

---

## 345. Language Models are not Equally Robust to Non-Canonical Tokenization across Languages

**arXiv ID:** 2607.26831 | [PDF](https://arxiv.org/pdf/2607.26831v1)

**作者:** Poulami Ghosh `[一作]` (IIT Bombay), Preethi Jyothi `[通讯]` (IIT Bombay)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究多语言下语言模型对非规范分词（non‑canonical tokenization）的鲁棒性，并通过LoRA多分词训练提升鲁棒性。

**💡 创新点**

发现tokenization invariance在非英语语言中并不普遍，且与语言的分词碎片化相关；提出利用多分词训练作为鲁棒性增强的方法。

**🔧 技术方法**

使用Llama‑3.1‑8B‑Instruct、Qwen3‑8B、Gemma‑3‑12B‑IT等LLM；采用随机非规范分词、字符级分词、Multi‑valued Decision Diagram（MDD）采样；LoRA微调等技术。

**📊 数据集**

使用FLORES‑200、Multilingual‑ARC等多语言基准，涵盖27种语言和六个下游任务（MLQA、MGSM、Multi‑ARC、Multi‑HellaSwag、Global‑MMLU‑Lite、Belebele）。

**📈 对比分析**

对比canonical、随机非规范和字符级分词下的性能，计算相对下降率；结果显示Llama 23.8‑40.9%下降，Qwen 11.4‑16.1%，Gemma 9.9‑?%；多分词LoRA训练可提升约1.8‑3%准确率并提升鲁棒性。

**⚠️ 局限性**

局限性：仅在Multilingual‑ARC上验证多分词增强，未覆盖其他任务；未分离影响鲁棒性的具体因素，导致对原因机制的解释不完整。

---

## 346. Budget-Aware LLM Discovery via Cost-Calibrated Frontier Utility

**arXiv ID:** 2607.26828 | [PDF](https://arxiv.org/pdf/2607.26828v1)

**作者:** Yansen Zhang `[一作]` (City University of Hong Kong), Yiyan Qi `[通讯]` (International Digital Economy Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为CostAda的成本校准自适应控制器，用于在显式搜索侧代币预算下引导大型语言模型进行科学与算法发现。

**💡 创新点**

创新点在于将搜索动作的信用与实际消耗代币的成本相结合，形成成本校准的前沿效用，并将剩余预算条件化进本地探索强度、前沿分配以及预算驱动的策略干预决策。

**🔧 技术方法**

主要技术包括：成本校准前沿效用计算、基于UCB的前沿分配、预算感知的局部探索强度调节、预算限制的策略干预、以及对不同代币成本的归一化处理。

**📊 数据集**

使用了八个公开基准，涵盖几何包装、极值几何、加法组合数、可执行程序优化等四大类。

**📈 对比分析**

在GLM‑5和GPT‑5.4两种代币价格差异显著的后端上，与AdaEvolve和EvoX两种最先进的自适应控制器进行对比；实验表明CostAda在所有八个基准上均取得最高平均最终质量，并在12/16个基准-后端对中，仅用最多一半预算即可达到最强基线的全预算质量；在AUC和早期预算效率方面亦显著优于基线。

**⚠️ 局限性**

局限性包括：只关注代币预算而未考虑其他计算资源；控制器参数在整个基准族内固定，未针对单个基准进行自适应调优；方法基于确定性评估器，可能难以推广到非确定性或多目标评估场景。

---

## 347. AI as Friction for Reflection Support in Ideation

**arXiv ID:** 2607.26827 | [PDF](https://arxiv.org/pdf/2607.26827v1)

**作者:** Janin Koch `[一作]` (University of Lille), Géry Casiez `[通讯]` (University of Lille)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过理论分析和案例阐述，将AI在设计创意工作中定位为“摩擦”而非“平滑”，提出通过结构化暂停促进设计师的反思与推理。

**💡 创新点**

创新点在于将AI工具的目标从消除创作阻力转向在创意迭代中引入有目的的暂停，以支持反思并构建可传递的设计理据。

**🔧 技术方法**

主要借助人工智能交互代理的设计理念，提出若干示例机制（如请求阐述拒绝理由、延迟生成、提供对比实例等），但未实现具体技术细节。

**📊 数据集**

本论文为概念性工作，未使用任何数据集。

**📈 对比分析**

论文未进行实验或性能评估，主要通过文献回顾和理论推理进行论证。

**⚠️ 局限性**

局限在于缺乏实证验证、评估指标和实现细节，及其在多样化团队环境中的可行性与适配性仍待进一步研究。

---

## 348. From Found to Designed: Concepts as a Design Axis for Large Language Models

**arXiv ID:** 2607.26825 | [PDF](https://arxiv.org/pdf/2607.26825v1)

**作者:** Chen Shani `[一作]` `[通讯]` (Tel Aviv University), Chen Shani (Tel Aviv University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种将概念作为大型语言模型（LLM）设计轴的框架，并对已有的概念相关研究按两维度（pipeline 阶段与概念来源）进行映射，揭示了概念介入在 LLM 开发中的空间与缺口。

**💡 创新点**

创新点在于：①将“概念”从后期可解释性工具重新定位为前期可设计的核心要素；②构建了一个双轴的概念设计空间（训练目标 / 结构 / 推理 / 解释 与 内部产生 / 外部约束），系统化整理了散落在不同子领域的相关方法；③指出了此空间中的空白与交叉点，为未来统一研究路径提供了思路。

**🔧 技术方法**

技术手段主要是文献综述与概念映射；通过整理已有的概念层次目标、概念瓶颈网络、图结构增强、推理时概念聚合等方法，形成了对应的二维表格与讨论。

**📊 数据集**

本文没有使用新的数据集；所讨论的技术和方法均来自已公开的相关研究，例如 ERNIE、SenseBERT、KALM、图神经网络等公开模型与数据。

**📈 对比分析**

由于是定位性论文，没有进行实验比较或性能评估；作者通过对比已有研究的优势与局限性，指出了不同阶段/来源的技术在稳定性、可组合性、可解释性等维度上的表现差异，但未给出定量指标。

**⚠️ 局限性**

局限性在于：①未提供系统性的实验验证；②映射表格并不完整，某些细节可能缺失；③未讨论不同应用场景下最佳设计点的权衡，仍需后续工作进行定量评估与基准构建。

---

## 349. From Uncertainty to Determinism: Coarse-to-Fine Visual Floorplan Localization without Ray Matching

**arXiv ID:** 2607.26817 | [PDF](https://arxiv.org/pdf/2607.26817v1)

**作者:** Shiyong Meng `[一作]` (Central South University), Jianxin Wang `[通讯]` (Central South University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于图像条件扩散模型与局部残差回归的视觉楼层平面定位框架CF^2Loc，能够直接估计多模态姿态分布并在无射线匹配的情况下实现全局定位和精细补正

**💡 创新点**

创新点包括①直接在连续姿态空间建模多模态后验分布，避免中间射线压缩造成的信息瓶颈；②采用图像条件扩散模型实现全局候选生成；③利用候选中心裁剪局部平面并进行姿态残差回归，消除局部结构歧义；④无需离线射线预处理或测试时查表，实现渲染无关、实时性高

**🔧 技术方法**

核心技术为图像-平面跨模态特征融合、位置编码与交叉注意力、姿态扩散模型（DDPM）与条件去噪网络、局部裁剪与残差回归网络、核密度估计进行候选模式提取、旋转增强与粒子采样

**📊 数据集**

在两大公开基准上进行评估：合成室内地图与图像数据集Structured3D（S3D）全量版本以及真实世界室内数据集ZInD

**📈 对比分析**

与多种SOTA方法（PF-net、MCL、LASER、F^3Loc、3DP、RSKFLoc、SemRayLoc、SceneAligner、FoD等）进行对比。CF^2Loc在S3D上0.1 m/0.5 m/1.0 m/1.0 m+30°召回率分别为11.3%/64.4%/73.7%/72.6%，显著高于上一最佳3D重建基线（53.8%/51.6%）；在ZInD上0.5 m/1.0 m/1.0 m+30°召回率分别提升至45.5%/57.3%/45.9%，同样超过SemRayLoc与FoD等语义基线。

**⚠️ 局限性**

局限性主要在于：①扩散采样需要一定时间和显存，虽然在10步内收敛但仍比纯前向网络稍慢；②局部残差回归依赖于候选裁剪的精度，对极大误差的候选可能无法完全补正；③对极端相似布局（如完全对称长走廊）仍可能产生多模态模式误判，需进一步改进候选排序或引入上下文约束

---

## 350. Practice Makes Policies: Bootstrapping and Consolidating Robotic Capabilities from Zero Human Demonstrations

**arXiv ID:** 2607.26809 | [PDF](https://arxiv.org/pdf/2607.26809v1)

**作者:** Jialiang Li `[一作]` (Shanghai Jiao Tong University), Wenzhao Lian `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 HERO，一种从零人类示例开始，通过层级化的启发式推理、经验重用和反射式执行实现自主能力演化的机器人操控框架，并通过统一的 Orchestrator 进行数据采集、经验筛选与任务执行。

**💡 创新点**

创新点包括：①将机器人智能划分为三层（Heuristic, Exemplar, Reflexive）并在同一框架内实现从推理到经验再到闭环控制的连续演化；②通过自动化的逆向任务生成实现无人工干预的持续场景重置；③动态调度能力层以平衡效率与鲁棒性的自适应执行策略。

**🔧 技术方法**

核心技术包括：VLM 语义分解与 3D 基础构建（DepthAnything-3），SAM 目标分割与点云获取，MT3 + PointNet++ 进行运动转移与刚性变换，LCS 语义检索，π_0.5 视觉-运动闭环策略；所有层级均通过 Orchestrator 进行规划、监控与回滚。

**📊 数据集**

使用了自采集的真实世界数据集：包件拾取、块堆叠、抽屉搜索与立方体发现，共收集 664 条正向与反向子任务轨迹，采用 Franka Emika Panda 与四台 RealSense D435i 摄像头进行实验；未使用公开公开数据集，全部为自定义任务与数据。

**📈 对比分析**

通过与单一层级（仅 L1、L2 或 L3）以及不同组合（L1+L2、L1+L3、L2+L3 等）的对比实验，HERO 在四项任务的平均成功率达到 86%，平均完成时间 3.9 分钟；单层 L3 仅 70% 成功率，L2 仅 66.7%；实验还显示 L2 在数据收集速度上明显优于 L1，且全系统整体鲁棒性和效率显著提升。

**⚠️ 局限性**

局限性包括：① VLM 推理时延高，导致 L1 计算成本高；② 对深度估计与点云重建的依赖，使得感知噪声与遮挡时性能下降；③ 预定义的原语动作空间限制了自动发现新技能的能力，需人工设计；④ 目前缺乏跨域通用性评估，无法充分验证在更复杂环境下的适应性。

---

## 351. Route by Kinematics, Act by Observation: Kinematics-Supervised Expert Routing in MoE-Augmented VLA

**arXiv ID:** 2607.26807 | [PDF](https://arxiv.org/pdf/2607.26807v1)

**作者:** Tianhang Yang `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对动作轨迹进行聚类得到运动原型，并用这些原型标签监督Mixture-of-Experts的全局路由器，从而实现显式、以运动学为导向的专家调度；同时构建低成本DIY机器人平台和对应基准，验证跨平台泛化。

**💡 创新点**

①利用运动学原型聚类揭示“运动学原型崩塌”现象；②提出KinRT——一种将训练阶段的运动学信息迁移到推理阶段视觉-语言观测的异步桥接机制；③实现全局显式路由器并与多种VLA骨干兼容；④构造经济型DIY机器人与真实任务基准。

**🔧 技术方法**

MoE架构与全局路由器、动作轨迹聚类（PCA+K‑means）、监督路由损失、动作去噪扩散式生成、LoRA与全参数微调、平衡采样、3D打印DIY机器人。

**📊 数据集**

RoboTwin基准（8个操纵任务，总计800条演示）与DIYRobot基准（5个任务，500条演示），均包含清洁与随机两种设定。

**📈 对比分析**

与传统稠密VLA（OpenVLA、π_0、π_0.5等）及隐式路由MoE（Hi‑MoE、AdaMoE）进行对比；在RoboTwin上KinRT在清洁/随机模式下平均成功率分别提升23.26%/13.78%；在DIYRobot上提升20.27%；与隐式路由相比提升约8–15%；实现跨平台性能提升且推理成本几乎不变。

**⚠️ 局限性**

对运动学聚类参数的敏感性、需要训练时可获得运动学数据、未解决专家负载平衡与潜在死专家问题、在极端新颖运动学模式下可能仍需进一步适配。

---

## 352. Beyond Action Imitation: Learning a Decision-Aware User Simulator for Online Advertising

**arXiv ID:** 2607.26893 | [PDF](https://arxiv.org/pdf/2607.26893v1)

**作者:** Zipeng Chen `[一作]` (Tencent Inc.), Jie Jiang `[通讯]` (Tencent Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了DASH框架，通过三阶段（上下文工程、监督微调、强化学习）构建了能够模拟广告用户行为且生成思维轨迹的LLM用户模拟器。

**💡 创新点**

创新点在于①通过跨域上下文压缩实现信息有效折叠；②使用教师生成思维轨迹并对其进行质量过滤构建token级监督；③设计混合奖励与GRPO实现对思维质量和行动准确性的双重优化。

**🔧 技术方法**

采用大型语言模型（如Qwen3.5）、层次化上下文压缩、结构化自我修正提示、教师-学生蒸馏、LLM-as-Judge评估、混合式奖励与Group Relative Policy Optimization等技术。

**📊 数据集**

使用腾讯多域广告交互日志，包括广告与多种内容域（视频等）的历史记录，按时间划分训练/验证/测试集。

**📈 对比分析**

与多种开源LLM的提示式联合生成基线对比，采用加权Precision/Recall/F1和思维质量（Form/Content/Logic）评估；DASH在小规模SFT+RL下在行动预测和思维质量上均优于基线，尤其逻辑质量最高。

**⚠️ 局限性**

仍依赖教师生成的思维轨迹质量；SFT对思维质量提升有限；奖励设计需进一步完善；对冷启动或极端稀疏交互的模拟效果不足。

---

## 353. SERPO: Self-Evolving Rubric Policy Optimization for Open-Ended Test-Time Reinforcement Learning

**arXiv ID:** 2607.26873 | [PDF](https://arxiv.org/pdf/2607.26873v1)

**作者:** Jianze Wang `[一作]` (Huazhong University of Science and Technology), Qianglong Chen `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在无标签、无外部评估的情况下，利用自我演化的判定和评价准则，在测试时对语言模型进行强化学习，使其在开放式生成任务上自我提升。

**💡 创新点**

提出 SERPO——一个闭环的测试时强化学习框架，使用 G‑N‑B 响应档案、查询特定评判准则和概率化评判奖励三者协同演化，从而在开放式生成中构建可靠的奖励信号。

**🔧 技术方法**

采用闭环自我演化的 G‑N‑B 响应档案、查询特定 rubric 生成与更新、概率化判定奖励以及 GRPO（PPO 风格的组相对优化）等技术。

**📊 数据集**

在 Qwen 系列模型上评估 HealthBench、ResearchQA、MedQA、LLMEval‑Med、GPQA‑Diamond、RaR‑Science 六个基准。

**📈 对比分析**

与基线无适配、投票型 TTRL、claim-consensus TTRL、固定 rubric TTS 等方法对比，在所有基准上 SERPO 在域内提升 20–54%，在 OOD 基准提升 11–20%，宏平均提升 8 以上，且在连续演化和长周期实验中持续上升。

**⚠️ 局限性**

实验仅覆盖两种模型配置、单一 benchmark 的 45 轮演化以及两 benchmark 的 60 轮顺序演化，且评判器与 rubric 生成可能继承模型偏差，推断时计算开销较大。

---

## 354. TREA-Net: A Transferable Residual Epidemiological Adaptation Network for Dengue Incidence Forecasting

**arXiv ID:** 2607.26854 | [PDF](https://arxiv.org/pdf/2607.26854v1)

**作者:** Inesh Shukla `[一作]` (International Institute of Information Technology Hyderabad), Chittaranjan Hens `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种轻量级迁移学习框架TREA-Net，用于在数据不足的地区实现多周登革热预测，并兼容不同数量的监测单元。

**💡 创新点**

创新点包括：①利用环境时间序列SIR（ETSIR）提供可迁移的机制先验；②设计了节点不变的门控残差校正模块，可从源地区学习并直接迁移到目标地区；③仅需在目标地区学习两个全局缩放与偏移参数，实现极低的目标适配成本。

**🔧 技术方法**

技术主要涉及：时间序列ETSIR模型、门控多层感知机残差校正、重加权多源训练、两参数目标适配、以及基于EnbPI的共形预测不确定性估计。

**📊 数据集**

数据集包括：源地区的哥伦比亚（33个部门，835周）和尼加拉瓜（18个部门，981周）；目标地区的墨西哥（22个州，208周）和马来西亚（15个州，270周），并使用对应地区的温度、降水等气候协变量。

**📈 对比分析**

与五种基准预测器（LSTM、N‑HiTS、TCN、PatchTST、TiRex）对比，TREA-Net在10个基准‑目标组合中取得9个显著提升（p<0.05），尤其在TiRex基础上提升约7.1%，并在马来西亚和墨西哥的8周预测误差上平均下降约3–5%。

**⚠️ 局限性**

局限性包括：需至少78–104周的历史数据；仅使用历史气候平均值，无法即时反映异常天气；在评估上仅覆盖两国，缺乏前瞻性验证；对月度报告或长期停报缺乏适配。

---

## 355. Before Agents Speak: Pre-hoc Failure Risk Inference in Multi-Agent Systems

**arXiv ID:** 2607.26836 | [PDF](https://arxiv.org/pdf/2607.26836v1)

**作者:** Shi Lin `[一作]` (Zhejiang Gongshang University), Xun Wang `[通讯]` (Zhejiang Gongshang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种预先推断多智能体系统幻觉风险的框架Halluprop，能够在任何交互发生前估计每个代理的固有幻觉倾向与传播风险，并给出全局风险诊断；

**💡 创新点**

将细粒度角色-查询语义对齐、传播因子与图结构信息融合，并通过可微Noisy‑OR与固定点迭代实现整体风险推断，首次实现预执行幻觉风险定位；

**🔧 技术方法**

使用预训练文本编码器+查询门控实现角色-查询语义分解，构造传播风险矩阵并结合Noisy‑OR推断；

**📊 数据集**

在三大公开基准上评测：MMLU（常识与专业知识）、MATH（数学推理）和MedQA（医学问答）；

**📈 对比分析**

与四个后验检测基线（LLM Debate、Inspector、SelfCheckGPT、GUARDIAN）对比，平均AUROC 84.6%，Hit@1>80%，子秒诊断时间（65×加速），并在干预后与最佳后验方法（GUARDIAN）相当；

**⚠️ 局限性**

对角色提示质量敏感，假设图结构已知且静态，仅在有限轮次交互内有效，且固定点迭代深度需经验调参，未覆盖动态拓扑变化与更长链路的传播行为。

---

## 356. SCALPEL: Semantic Cross-modal Alignment via LLM-Powered Encoder Learning for Medical Vision-Language Representation

**arXiv ID:** 2607.26885 | [PDF](https://arxiv.org/pdf/2607.26885v1)

**作者:** Yunzhan Fu `[一作]` (Hangzhou Dianzi University), Liqi Yan `[通讯]` (Hangzhou Dianzi University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了SCALPEL框架，通过临床报告对比微调与异构特征缓存，实现大规模医学LLM的双向编码器化与高效跨模态对齐，解决上下文窗口、计算成本与医疗幻觉等问题。

**💡 创新点**

创新点包括①将生成式LLM转化为双向编码器；②异构特征缓存实现大批量高效训练；③Anatomy‑Negation Aware Objective显式惩罚解剖侧向与否定错误，显著提升医学语义匹配质量。

**🔧 技术方法**

采用PMC‑LLaMA‑13B + LoRA微调、Masked Token Prediction + SimCSE、Vision Transformer DINOv2、InfoNCE 损失 + ANAO 组合，实现文本与图像的跨模态对齐。

**📊 数据集**

使用MIMIC‑CXR、CheXpert、IU X‑Ray、VQA‑RAD、SLAKE等多种医学图像与文本/问答数据集进行评估。

**📈 对比分析**

与Med‑CLIP、CXR‑CLIP、BioViL‑T、MedProbCLIP等基线相比，SCALPEL在跨模态检索 Recall@1 最高达23.17%，零样本疾病分类 ACC 提升约4.6% 以上，VQA 结果亦显著优于竞争模型。

**⚠️ 局限性**

主要局限在于依赖外部NER提取的解剖与否定元数据可能产生误差；LLM参数被冻结限制了深层跨模态交互；模型参数量相对较大，仍需进一步压缩以适配资源受限环境。

---

## 357. Detection of AI-generated stems within hybrid human-AI music

**arXiv ID:** 2607.26874 | [PDF](https://arxiv.org/pdf/2607.26874v1)

**作者:** François Rigaud `[一作]` (Deezer Research), Romain Hennequin `[通讯]` (Deezer Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

检测混合音乐中人类与AI生成的各个stem是否为AI生成

**💡 创新点**

提出并验证一种并行模型，将局部混音检测与源相对能量信息相结合，实现对单个stem的生成检测

**🔧 技术方法**

利用局部AI生成检测器、源分离(ht-demucs)、STFT能量比以及多层感知机进行特征融合与判别

**📊 数据集**

合成的MUSDB18‑HQ混合数据集，包含四种混合类型（v_g+a_r、v_r+a_g、v_g+a_g、v_r+a_r）与不同增益设置

**📈 对比分析**

与简单的“先分离后检测”naive方法以及oracle SNR相比，所提方法在生成伴奏检测上精度明显提升，生成人声检测虽低但仍优于baseline，整体性能优于单纯源分离方法

**⚠️ 局限性**

仅针对两stem（人声+伴奏）场景，依赖Demucs的分离质量，难以扩展到更多stem或真实AI生成的多stem混音，误报率仍较高

---

## 358. Dual-Path LLM Reasoning for Multimodal Few-Shot Knowledge Graph Completion

**arXiv ID:** 2607.26909 | [PDF](https://arxiv.org/pdf/2607.26909v1)

**作者:** Jinlan Liu `[一作]` (Harbin Institute of Technology), Hongliang Sun `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种双路径LLM推理框架 DuPLeR，专为少样本、多模态知识图谱完成（KGC）而设计。

**💡 创新点**

创新点包括：① 通过多模态 LLM 生成关系先验并用结构校正消除幻觉；② 双层结构推理（关系图 + 目标实体图）与双路径多模态增强（查询特定调制 + 查询无关 LMF‑Max）实现精细信息融合；③ 联合预训练使模型在未见图谱上无需重新训练即可直接推断。

**🔧 技术方法**

采用的技术有：多模态 LLM 提示、图神经网络（CCoRGE 与 MAPNet）、低秩融合、噪声过滤机制、二元交叉熵损失、联合多图预训练与动态数据加权。

**📊 数据集**

实验数据集包括 FB15K‑IMG‑R、OpenBG‑IMG‑R、DB15K‑EMB‑R、YAGO15K‑EMB‑R 以及 CoDEx‑IMG‑R，共五个多模态知识图谱。

**📈 对比分析**

与传统 Embedding、单模态与多模态基线（DistMult、RotatE、InGram、ULTRA、ProLINK、MMSN）在 0‑shot、1‑shot、3‑shot 以及 Zero‑shot 场景下的 Hits@10 进行对比，DuPLeR 在大多数设置中显著领先，提升幅度可达 30–40%。

**⚠️ 局限性**

局限性在于高度依赖 LLM 提示与视觉信息的质量，阈值设定需手工调优；预训练与推理图谱差异可能导致迁移性能下降；视觉噪声仍会对实体表示产生一定干扰。

---

## 359. Actions Have Consequences: Detecting Outcome Performativity using Intervention Testing

**arXiv ID:** 2607.26908 | [PDF](https://arxiv.org/pdf/2607.26908v1)

**作者:** Brandon Gower-Winter `[一作]` (Utrecht University), Georg Krempl `[通讯]` (Utrecht University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种离线的预测干预方法（Outcome Performativity A/B Detection），通过对预测结果进行随机干预并比较不同干预组的结果分布，检测模型预测是否会因预测行为而改变真实结果（Outcome Performativity）。

**💡 创新点**

创新点在于（1）首次给出离线干预检测框架并推导其样本复杂度上界；（2）引入“不可辨别区域”（regions of indistinguishability）概念，量化在给定样本量下无法检测Outcome Performativity的情形；（3）在三类理论假设（简单、模型基、误分类基）下验证样本复杂度公式，并在真实数据（Open Bandits Dataset）上做案例研究。

**🔧 技术方法**

使用的技术主要包括：随机化对照实验（A/B测试）和因果推断中的 do-操作；统计检验方法（Chi‑Squared、Fisher Exact、Mann‑Whitney U）；基于线性模型的模拟（Model‑based Performativity）；以及对样本复杂度的理论推导。

**📊 数据集**

实验所用数据集：Open Bandits Dataset（男女两子集），以及若干二分类公开数据集（如信用分配、结肠癌等）用于模拟三种假设下的Outcome Performativity。

**📈 对比分析**

方法比较：在模拟和真实数据中分别评估检测率随样本量变化的曲线。实验结果表明，随着干预样本量增大，检测率提升；在不可辨别区域内检测率趋近零；在男性子集检测率在较大样本量（10⁵）时约为52%，女性子集检测率始终接近0。整体表现优于传统在线检测方法（如T‑learner、Double ML），但对极小效应尺寸或高度不平衡数据仍有限。

**⚠️ 局限性**

局限性包括：假设Outcome Performativity在特征空间上均匀；未考虑验证延迟（outcome出现时间）和噪声标签；方法只检测 P(Y|do(Y=0))≠P(Y|do(Y=1)) 的差异，若真实情况需考虑 P(Y|X,do(Y)) 的差异则可能失效；在极小效应尺寸或高成本/伦理约束下，检测仍可能不可行。

---

## 360. Hearsay: Vision-Language Medical Diagnoses Without an Image

**arXiv ID:** 2607.26886 | [PDF](https://arxiv.org/pdf/2607.26886v1)

**作者:** Siddharth Vohra `[一作]` `[通讯]` (Carnegie Mellon University), Siddharth Vohra (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究在缺乏医学影像的条件下，探讨了三款前沿视觉‑语言模型（Claude Opus 4.7、GPT‑5.4、Gemini 3.1 Pro）在接受仅含人群描述的提示时会产生的“幻影”诊断，并证明这些诊断受人群属性（年龄、性别、种族）显著影响；

**💡 创新点**

创新点在于首次将幻影效应扩展到结构化输出渠道，并通过 Jensen‑Shannon 散度衡量人群描述对诊断分布的影响，揭示不同模型存在“词汇触发型”与“类别保持型”两种幻影失效模式；

**🔧 技术方法**

采用统一的 JSON 结构化输出、专门设计的 mirage‑mode 提示模板、长达 4,000 token 的生成长度，配合 100 条种子样本和 11,700 次模型调用；

**📊 数据集**

使用人工合成的 12 个（年龄×性别×种族）人群描述与 3 种影像模态（胸 X‑ray、脑 MRI、皮肤 mole）组合的提示；不涉及真实病历；

**📈 对比分析**

对每种模型和模态，在 12 个人群细胞与中性基线（无描述）之间计算 Jensen‑Shannon 散度；GPT‑5.4 的最大 JSD 达 0.59，Claude 在皮肤病学上达到 0.83；同时评估拒绝率、结构化诊断与自然语言的“hedged”情况，显示 GPT‑5.4 生成率高而 Claude 主要通过拒绝与结构化诊断不一致；

**⚠️ 局限性**

局限性包括：使用了 schema 强制与提示干预；“brown”种族标签语义不一致；中性基线并非完全无偏；未区分预训练偏差与真实疾病流行率；以及实验仅在合成提示上进行，缺乏真实临床影像验证。

---

## 361. Thinking Under Uncertainty: Evidence Use and Information-Seeking in Language Models

**arXiv ID:** 2607.26845 | [PDF](https://arxiv.org/pdf/2607.26845v1)

**作者:** Hua-Dong Xiong `[一作]` (Georgia Tech), Robert C. Wilson `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在受控的两臂赌博机任务中，评估了推理时思考对大语言模型决策行为的影响，并通过动作、思考长度与置信度三项指标进行量化。

**💡 创新点**

创新之处在于首次将推理思考与信息寻求、元认知控制/监测分离，并采用UCB‑和Thompson‑式分解来解析模型的探索与价值导向行为。

**🔧 技术方法**

作者使用认知概率模型拟合动作和置信度，并对十个开源LLM（Gemma、GPT‑OSS、Nemotron、Qwen 等）在思考与非思考模式下进行实验，同时检验温度、top‑p、top‑k 等解码设置的影响。

**📊 数据集**

实验数据来自自行生成的两臂赌博机奖励历史，共 600 次试验，包含不同观测数 (1,1)、(3,3) 和 (2,6) 以及价值差距分层。

**📈 对比分析**

通过对思考与非思考模式下的 UCB‑、Thompson‑系数、噪声、思考长度和置信度进行对比，结果显示思考模式下价值导向更强、噪声下降，但并未出现更强的信息寻求行为；思考长度与信息不平衡相关，置信度对决策难度更敏感。

**⚠️ 局限性**

主要局限包括只评估初始决策而非长期学习、信息不平衡历史同时伴随更多观测导致混淆、思考长度与置信度仅为可观察输出且缺乏人类对照实验。

---

## 362. BATS: Resource-Efficient Volumetric Segmentation with Boundary-Aware Mixed-Resolution Tokens

**arXiv ID:** 2607.26829 | [PDF](https://arxiv.org/pdf/2607.26829v1)

**作者:** David Hagerman `[一作]` (Chalmers University of Technology), Fredrik Kahl `[通讯]` (Chalmers University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

BATS 通过在 3D 医学图像分割中使用边界感知的混合分辨率标记，动态地在预测边界附近保留高分辨率标记，其他区域使用低分辨率标记，从而实现稀疏高效的分割。

**💡 创新点**

创新点包括：
- 并行全尺度边界预测，避免了粗层决策误判导致的细层信息丢失；
- “fine-first”上下文级联，保留所有细层祖先信息；
- 父级聚类注意力（Parent Cluster Attention），将祖先标记直接注入每个标记的注意力邻域，实现跨尺度交互而无需密集多尺度特征图；
- 结合原型监督（oracle warmup）提升训练稳定性。

**🔧 技术方法**

主要技术：ConvNeXt 金字塔+FPN 做密集边界预测；稀疏混合分辨率层级构建；父级聚类注意力；轻量化稠密头将稀疏标记映射为体素级别分割；Dice-BCE 边界损失与 Dice-CE 分割损失；辅助类别分布损失。

**📊 数据集**

在 nnU-Net Revisited 评测的五个公开数据集上实验：ACDC、LiTS、BraTS、KiTS、AMOS。

**📈 对比分析**

与 nnU-Net、MedNeXt-L、SwinUNETR 等主流基线相比，BATS 在 LiTS 上取得最高 Dice（83.39%），与 MedNeXt-L 差距仅 0.37%，但峰值 GPU 内存降低 53% 以上；在 KiTS、LiTS 上推理速度提升 27–30%，在 BraTS 上略慢（因标记更密集）。整体显示了在边界稀疏数据集上的内存/速度/精度 Pareto 改进。

**⚠️ 局限性**

局限性：
- 对非语义边缘（如强对比度突变）过度保留细分辨率标记，导致精度略降；
- 对边界密集数据集（如 BraTS）稀疏化效果有限，甚至推理慢；
- 需要额外的边界预测网络与多尺度结构，训练复杂度和实现难度较高；
- 目前仅在 3D 医学图像分割验证，扩展到 2D 或非医学领域需进一步验证。

---

## 363. Human diversity fuels collective creativity that large language models cannot simulate or sustain

**arXiv ID:** 2607.26899 | [PDF](https://arxiv.org/pdf/2607.26899v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 364. Forecasting Trajectory-Level Safety Risks in Black-Box Multi-Turn Interactions

**arXiv ID:** 2607.26820 | [PDF](https://arxiv.org/pdf/2607.26820v1)

**作者:** Shi Lin `[一作]` (Zhejiang Gongshang University), Xun Wang `[通讯]` (Zhejiang Gongshang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

Recast框架在多轮LLM交互中预测未来安全风险，提前发出预警；

**💡 创新点**

将安全风险建模为轨迹级可预测问题，利用双尺度检索、风险状态与转移建模以及因果时间编码来预测风险出现时间分布；

**🔧 技术方法**

使用预训练文本编码器、局部与全局自注意力、可学习内存查询、交叉路径证据融合、风险状态/转移前馈编码器、因果时间编码器，并以负对数似然及辅助目标进行端到端训练；

**📊 数据集**

通过四阶段流程构建的10k条多轮交互轨迹（5k有害、5k良好）覆盖7类风险，测试集采用AdvBench和SORRY-Bench共960条攻击样例；

**📈 对比分析**

在7类风险上平均NLL 0.45、MAE 0.31，提前警报率88.3%、误报率12.3%，平均提前2.41回合；相较于现有Guardrail、THRD、Intent Analysis等基线，攻击成功率从71.5%降至45.9%，与最强基线THRD相当；Latency仅增加1.4%；

**⚠️ 局限性**

对长距离预测及某些风险类别（化学/生物、网络安全）的精度下降；依赖预训练编码器的语义表示；缺乏对攻击策略的深度解释性；对极短交互或实时约束下的响应时间仍有提升空间。

---

## 365. Risk-Aware Motion Planning with Learned Trajectory Primitives and Probabilistic Safety Assessment

**arXiv ID:** 2607.26802 | [PDF](https://arxiv.org/pdf/2607.26802v1)

**作者:** Marc Kaufeld `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套结合RBF网络学习、解析式碰撞概率评估与MPC后期优化的混合运动规划框架，目标是在城市环境中生成安全、高效、可解释的轨迹。

**💡 创新点**

创新点包括：① 利用RBF网络快速生成近似最小jerk轨迹并将其约束到动态一致的搜索空间；② 采用解析式的空间重叠概率来精确评估每个候选轨迹的碰撞风险；③ 在MPC层对已评估的最佳轨迹进行轻量级优化，以保证动态可行性与平滑控制。

**🔧 技术方法**

技术手段包括：径向基函数网络（MP‑RBFN）学习与推理；解析概率碰撞评估（利用Minkowski和高斯卷积计算空间重叠概率）；模型预测控制（acados + HPIPM）用于轨迹后期优化；以及在CommonRoad基准上进行评测。

**📊 数据集**

数据集：使用2M条离线求解的最小jerk最优轨迹用于训练MP‑RBFN；实验采用CommonRoad 1000场景的公开基准进行对比评估。

**📈 对比分析**

与Frenetix、MP‑RBFNonly、MPConly三种代表方法对比，实验表明该框架在碰撞率、成功率、可行性等指标上均优于或持平于最先进的采样方法，并在安全性和动态可行性上显著提升（碰撞率最低，成功率与采样方法相当，可行性达96%）。

**⚠️ 局限性**

局限性：RBFN生成的轨迹覆盖范围受训练集限制，可能无法涵盖极端或多模态交通情景；目前的碰撞风险评估假设高斯不确定性，缺乏对复杂交互行为的建模；实验仅在仿真基准中验证，真实道路测试仍待开展。

---

## 366. ICDAR 2026 Competition on Information Extraction from Atomic Layer Deposition/Etching (ALD/E) Scientific Figures

**arXiv ID:** 2607.26848 | [PDF](https://arxiv.org/pdf/2607.26848v1)

**作者:** Fahad Ahmed `[一作]` (Leibniz Information Centre for Science and Technology), Jennifer D'Souza `[通讯]` (Leibniz Information Centre for Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本工作提出了Sci-ImageMiner——一个针对ALD/E科学图像的专家标注、四任务（分类、数据提取、摘要、视觉问答）大规模基准与竞赛平台，聚焦多模态推理能力；

**💡 创新点**

创新点在于构建了首个覆盖49类ALD/E图形、包含子图级别标注与四种端到端推理任务的专门数据集，并通过社区驱动竞赛检验并推动领域特定多模态模型；

**🔧 技术方法**

使用了大型视觉语言模型（如Qwen2.5‑VL‑7B‑Instruct、Gemma‑4‑E4B、InternVL‑3.5‑8B等）以及LoRA、QLoRA等参数高效微调技术、检索增强生成、结构化提示工程等多种技术手段；

**📊 数据集**

数据集来源于205篇ALD/E研究论文，包含1,951幅图像，划分为训练/验证/测试集，涵盖实验与仿真、ALD与ALE两大子领域；

**📈 对比分析**

基线模型在分类、摘要上达F1≈0.68–0.84，最佳参赛队在分类上实现F1≈0.81、摘要ROUGE/BERT≈0.56；但在数据提取和视觉问答任务上，最高得分仍远低于理想（数据提取RMS≈17、TEDS≈66，VQA多模态指标约0.4–0.5），表明模型在结构化数据获取与推理方面表现不足；

**⚠️ 局限性**

主要限制包括：对复杂科学图形的结构化理解与抽取仍存在显著误差；视觉问答需要更深层次的科学推理与上下文整合；标注成本高、数据规模相对有限，未来需进一步提升多模态模型的领域适应性与推理连贯性。

---

## 367. Amortized Moment Matching for Visual Generation

**arXiv ID:** 2607.26860 | [PDF](https://arxiv.org/pdf/2607.26860v1)

**作者:** Wenze Liu `[一作]` (Chinese University of Hong Kong), Xiangyu Yue `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于神经网络的“摊销矩匹配”方法，通过在表示空间中匹配条件均值和协方差，作为后训练目标提升生成模型的性能。

**💡 创新点**

创新点在于：① 将扩散去噪器的多项式投影与数据分布的矩（尤其是均值与协方差）建立理论对应；② 引入“摊销Fréchet距离（AMFD）”损失，用可训练的神经摊销器动态估计条件统计量；③ 通过矩阵‑自由的Jacobian‑Vector‑Product实现高维协方差的高效计算；④ 通过条件匹配实现一阶/二阶矩的可扩展匹配。

**🔧 技术方法**

使用技术包括扩散/流匹配、Polynomial Projection、神经摊销器（AdaLN + MLP 或交叉注意力）、JVP、SIM（SigLIP、Inception、MAE）等表示编码器、以及多步与一步扩散模型的后训练。

**📊 数据集**

使用的数据集：ImageNet 256×256 用于图像生成后训练；文本-图像任务使用 PixelGen、FLUX.2 等多模态模型，并在 GenEval（指令跟随）和 PickScore（人工偏好）上评估；此外在原始像素、SD‑VAE、VA‑VAE、RAE 等不同生成空间进行实验。

**📈 对比分析**

与 FD‑loss、VAR‑d30、BAR‑L、SiT‑XL/2、MAR‑L、FlowAR‑H 等多种多步/一步扩散模型对比，AMFD 在 ImageNet 上取得 FDr^6 1.79 / 1.75、FID 下降、IS 上升；在文本‑图像任务中，AMFD‑C/SIM/10enc 在 GenEval 上从 0.749 提升至 0.833，PickScore 与多步老师持平或略优，显示出显著的性能提升。

**⚠️ 局限性**

局限性：① 目前仅匹配前两阶矩，未能捕捉更高阶统计信息；② 依赖预训练的低分辨率表示编码器，限制了高分辨率生成的直接适用性；③ 仅作为后训练目标，尚不能从零开始训练生成模型。

---

## 368. When Knowledge Changes: Metamorphic Testing of RAG Systems with Mutations

**arXiv ID:** 2607.26843 | [PDF](https://arxiv.org/pdf/2607.26843v1)

**作者:** Jinhan Kim `[一作]` (Università della Svizzera italiana), Paolo Tonella `[通讯]` (Università della Svizzera italiana)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套针对检索增强生成（RAG）系统的变形测试框架，使用11种变异算子在预分块和后分块两个作用域下评估语料库演化对系统输出的影响。

**💡 创新点**

创新点包括：①将变形测试与RAG结合，构造元关系判据；②系统化设计11种变异算子，覆盖噪声、格式与架构脆弱性；③通过元关系揭示传统RAGAS忽略的错误，并评估多种修复策略的覆盖率。

**🔧 技术方法**

技术手段包括：元形测试框架、预/后分块变异算子、LLM作为判定器（判定语义保留与答案等价）、RAGAS对比、检索预算调节、生成模型升级与LLM重排序。

**📊 数据集**

使用的数据集有五个问答基准：T2‑RAGBench（ConvFinQA、FinQA、TAT‑DQA、VQAonBD）和RepLiQA。

**📈 对比分析**

通过将元关系与RAGAS四项指标在五个数据集上进行元评估，实验显示元关系F1在0.927–1.00之间，远优于RAGAS最高0.57；在修复实验中，检索预算、生成升级和LLM重排序联合可修复约43.1%的检测错误。

**⚠️ 局限性**

局限性包括：仅针对单一RAG架构与LLM，变异成本高且样本有限；部分错误（如RepLiQA）对三种修复方法无效；LLM评判器可能产生幻觉，影响判定准确性。

---

## 369. Kairos: Numerically Robust News Recommendation under Item Cold-Start via Cholesky-based LinUCB

**arXiv ID:** 2607.26832 | [PDF](https://arxiv.org/pdf/2607.26832v1)

**作者:** Finn Hertsch `[一作]` `[通讯]` (DHBW Ravensburg), Finn Hertsch (DHBW Ravensburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Project Kairos 框架，利用基于 LinUCB 的上下文在线学习解决新闻推荐中的物品冷启动问题

**💡 创新点**

创新点包括：① 用 Cholesky 分解直接更新协方差矩阵，避免 Sherman‑Morrison 逆导致的数值不稳定；② 引入 Matryoshka 表示学习 (MRL) 来降低推理延迟，保留语义信息

**🔧 技术方法**

使用技术包括：LinUCB、Cholesky rank‑1 更新、Matryoshka 嵌入、MIPS 检索、Julia 实现

**📊 数据集**

基于 Tagesschau API 的区域新闻语料（约 N 条记录）

**📈 对比分析**

与传统 Sherman‑Morrison 逆实现的 LinUCB 对比，Cholesky 版保持 SPD 性质、实现约 5 倍速度提升，排名精度差距极小（MAE < 0.01）

**⚠️ 局限性**

局限性：实验规模有限，仅在小型区域语料上验证，缺乏大规模生产环境与长期用户交互评估

---

## 370. Mind the Gap: The Disconnect Between Synthetic and Natural Edge Weights in Parallel Single-Source Shortest Path

**arXiv ID:** 2607.26821 | [PDF](https://arxiv.org/pdf/2607.26821v1)

**作者:** Marco D'Antonio `[一作]` (Queen's University), Hans Vandierendonck `[通讯]` (Queen's University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对17个真实加权图的边权分布进行统计特征分析，并与常用的六种合成权重分布进行对比，评估七种并行SSSP算法在不同权重下的性能和参数调优敏感性。

**💡 创新点**

系统验证了合成均匀权重与真实权重之间的巨大偏差，揭示了权重分布对Δ参数选择和算法排名的逆转影响，并提出了对标志性Benchmark的改进建议。

**🔧 技术方法**

利用Clauset等的极大似然拟合与KS检验对权重分布进行尾部和主体建模，采用并行C++实现加速；实验采用Δ‑stepping、ρ‑stepping、Wasp、MultiQueue、Bellman‑Ford等并行SSSP实现。

**📊 数据集**

17个大规模自然加权图，包括道路网络、交通流、社交、语义、序列相似度等，节点数至数十亿条边。

**📈 对比分析**

在同一硬件平台下，对七种实现进行Δ参数扫描、执行时间对比，发现合成权重常导致误导性加速或降速，且Δ参数的最优值因权重分布漂移数个数量级，整体性能高度依赖权重。

**⚠️ 局限性**

仅考虑正权重，未覆盖负权重场景；实验仅限于17图，合成分布种类有限；硬件资源限制导致部分大Δ实验未完成，未评估动态调参或混合分布的长尾效应。

---

## 371. A First Look at Coding Agents' Compliance with AI Contribution Rules in Open-Source Communities

**arXiv ID:** 2607.26819 | [PDF](https://arxiv.org/pdf/2607.26819v1)

**作者:** Wenhao Yang `[一作]` (Peking University), Minghui Zhou `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 RepoComplianceBench 基准，评估编码代理是否遵守开源社区的 AI 贡献规则，并探究不同干预对合规性的影响。

**💡 创新点**

首次系统化评估代理在实际仓库治理文件中的规则发现与执行，并揭示“拒绝”与“交接”规则对代理而言不可恢复的合规性缺口。

**🔧 技术方法**

使用了双阶段合规检查（机械验证 + LLM 判别），并通过“提醒”“引用”“反馈”等三种干预策略对四种规则（拒绝、披露、验证、交接）进行实验。

**📊 数据集**

收集了 49 个仓库的 106 个问题实例，源自 102 个社区的 455 条 AI 贡献规范，形成了具有治理文件上下文的 benchmark。

**📈 对比分析**

与四款前沿代理（OpenCode/DeepSeek‑V4‑Pro、Codex/GPT‑5.3、Codex/GPT‑5.5、Claude‑Sonnet 4.6）在 280 条基线跑中对比，发现披露与验证规则在适当干预下可达 77–100% 合规率，而拒绝与交接始终保持 0%，显示出强大模型仍无法满足这些约束。

**⚠️ 局限性**

局限性包括仅测试四款代理且缺乏更复杂任务，评估仅关注提交前合规，且基准假设规则文本已公开且可读，未覆盖社区非文本治理手段。

---

## 372. Ripple: Real-Time Streaming Audio-Video Generation With Cross-Modal Recurrent Memory

**arXiv ID:** 2607.26818 | [PDF](https://arxiv.org/pdf/2607.26818v1)

**作者:** Yanbo Ding `[一作]` (Chinese Academy of Sciences), Yali Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Ripple，一种实时、长时序的音视频联合生成系统。

**💡 创新点**

创新点在于跨模态递归记忆机制、三阶段训练策略和在线强化后训练。

**🔧 技术方法**

使用滑动窗口注意力、RoPE、EMA更新、跨模态注意力、知识蒸馏与强化学习。

**📊 数据集**

训练基于 LTX-2.3 19B 生成器，评估采用 VerseBench Set‑3 与自制 30 秒长视频基准。

**📈 对比分析**

与离线模型及现有在线方法对比，Ripple 速度提升 15×，帧率 28 FPS，生成质量与教师相当。

**⚠️ 局限性**

局限在对长视频的对齐与同步略有下降，且需要大量 GPU 资源。

---

## 373. DistillAlign: Coordinating Mode Covering and Mode Seeking in Autoregressive Video Distillation

**arXiv ID:** 2607.26811 | [PDF](https://arxiv.org/pdf/2607.26811v1)

**作者:** Jiaxing Li `[一作]` (Riemann Dynamics), Yangguang Li `[通讯]` (Riemann Dynamics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文从分布视角重新审视自回归视频蒸馏，提出教师归一化的分布评估协议，并设计联合蒸馏方法以平衡模式覆盖与模式寻优，显著提升视频生成的质量、多样性和覆盖度。

**💡 创新点**

创新点包括：①阐明初始化与DMD目标分布对齐的重要性；②提出教师归一化的分布评估协议，客观度量初始化的模式覆盖；③将DMD的模式寻优与一致性蒸馏的模式覆盖约束联合起来，抑制后期分布漂移；④证明弱教师+匹配初始化可超越强教师的效果。

**🔧 技术方法**

使用的技术有：分布匹配蒸馏 (DMD)、一致性蒸馏 (CD)、ODE/一致性初始化、教师归一化重噪、V-JEPA2 特征提取、VBench 视觉质量评估以及 Vendi 多样性评估。

**📊 数据集**

使用的数据集包括 VidProM 提示集、Wan2.1-1.3B 与 Wan2.1-14B 训练与蒸馏数据，以及生成的 81 帧、分辨率 832×480 的视频。

**📈 对比分析**

通过与 LTX Video、Wan2.1-T2V-1.3B、SkyReels-DF、CausVid、Self-Forcing、Causal-Forcing 等基线在 VBench、精度、覆盖度与多样性上进行对比，结果表明本文方法在匹配教师下获得最高 VBench 分数、最高覆盖度并在多样性上亦优于基线；弱教师版本也能与强教师基线竞争。

**⚠️ 局限性**

局限性包括：逆 KL 目标仍可能导致后期分布漂移，需要通过 λ 进行手动调节；评估依赖共享重噪流程，可能受限于重噪策略；对更长序列或更高分辨率的扩展尚待验证。

---

## 374. DIRECT: Direct Decoding for Efficient and Aligned Sequence Labeling with Large Language Models

**arXiv ID:** 2607.26891 | [PDF](https://arxiv.org/pdf/2607.26891v1)

**作者:** Yilei Wang `[一作]` (Fuzhou University), Peichao Lai `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在低资源条件下，将大型语言模型用于序列标注任务，并通过在监督微调后加入直接偏好优化（DPO）与推理时的输出格式控制，提出了 DIRECT 框架；

**💡 创新点**

创新点在于将 DPO 与推理时的控制机制结合，既提升任务对齐度，又通过模板填充与 KV Cache 重用显著提升推理效率；

**🔧 技术方法**

核心技术包括：监督微调（SFT）、直接偏好优化（DPO）训练、控制式解码（固定格式与候选集限制）以及模板填充+KV Cache 重用的推理加速；

**📊 数据集**

使用了八个序列标注数据集：中文（Youku、Taobao、Weibo、Resume、UD、CTB6）和英文（CoNLL03、MIT‑Movie），在不同样本规模（K=250/500/1000）下进行评估；

**📈 对比分析**

与 InstructUIE、GoLLIE、GNER 等基线比较，DIRECT 在大多数数据集上均实现了 F1 分数提升，尤其在 MIT‑Movie 上提升超过 15%，且推理速度约为基线的 1/10；

**⚠️ 局限性**

局限性包括：仍依赖大型模型的算力，模板化方法可能在极端格式多变的任务中受限，且仅在低资源场景验证，未探究更大规模或跨领域的泛化能力。

---

## 375. StructureGS: Structure-aware Gaussian Splatting for Articulated Object Reconstruction

**arXiv ID:** 2607.26889 | [PDF](https://arxiv.org/pdf/2607.26889v1)

**作者:** Gahye Lee `[一作]` (POSTECH), Seungyong Lee `[通讯]` (POSTECH)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出 StructureGS，基于 3D 高斯渲染框架结合定向包围盒实现关节物体的结构感知重建，显著提升部件分割与运动估计；

**💡 创新点**

创新点在于引入面向结构的两项损失——空间一致性损失和结构连通性损失，利用定向包围盒约束各部件的几何与接触关系，解决传统光度监督导致的形状-运动耦合问题；

**🔧 技术方法**

技术上采用 3D Gaussian Splatting 作为几何表示，配合 OBB 提取与 Separating Axis Theorem 计算部件间距离，联合光度、结构拟合、接触与尺度正则化进行端到端优化；

**📊 数据集**

在 PARIS、DTA-Multi、以及自制真实场景四个数据集上进行实验，分别包含单/多部件物体，采用 100 或 10 视角 RGB 输入；

**📈 对比分析**

与 PARIS、ArticulatedGS、ScrewSplat 等基线比较，StructureGS 在稠密与稀疏视角下均取得更低的 Chamfer 距离、角度误差、位置误差以及更高的 PSNR/SSIM，显示出显著性能提升；

**⚠️ 局限性**

局限在于对仅产生细微视觉变化的关节动作仍可能出现几何折叠或运动估计偏差，需结合更高级的语义先验以提升鲁棒性。

---

## 376. Convex Collision-Free Regions

**arXiv ID:** 2607.26901 | [PDF](https://arxiv.org/pdf/2607.26901v1)

**作者:** Tomoyo Kikuchi `[一作]` (Kyoto University), Takashi Kanai `[通讯]` (University of Tokyo)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了Convex Collision-Free Regions（CCFR）方法，通过在每个顶点上构建局部凸可行区域并投影实现软体碰撞的非渗透，适用于布料、毛发、线状物体及粒子系统等多种软体模拟场景。

**💡 创新点**

核心创新在于显式构造每个顶点的局部凸可行区域，将几何非渗透约束与物理接触响应分离，避免了传统方法中对非线性优化的依赖，并实现了天然的并行化。

**🔧 技术方法**

技术实现包括基于Plesník算法的凸投影、局部半空间约束构建、XPBD框架下的软响应与摩擦模型、以及AABB树+距离计算的宽/细相位碰撞检测。

**📊 数据集**

实验使用多种公开网格与自定义场景，包括扭转布料、细绳线、毛发束、粒子堆叠及混合码维（多几何维度）场景，并未依赖特定数据集。

**📈 对比分析**

与CCD、ICCP、OCC、Shen等传统碰撞处理方法在同一XPBD框架下对比，CCFR在保持非渗透的同时，帧时平均低于对手且碰撞分辨阶段成本极低，展现出良好的可扩展性与并行性能。

**⚠️ 局限性**

限制方面，CCFR在高速或高密度碰撞中相对保守，可能导致可行区域过小；固定步长限制了可达速度；单精度实现下的数值稳健性尚未充分验证，且与基于Hessian的高刚度方法兼容性有限。

---

## 377. GPTQ-2D: Cubic-Time Two-Sided Adaptive Rounding

**arXiv ID:** 2607.27042 | [PDF](https://arxiv.org/pdf/2607.27042v1)

**作者:** Jiale Chen `[一作]` (Institute of Science and Technology Austria), Dan Alistarh `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为 GPTQ-2D 的三阶时间双侧自适应舍入算法，可在矩阵上实现与 Kronecker 结构对应的两侧量化。

**💡 创新点**

核心创新在于识别并利用回馈矩阵的秩一可分解，使同一条 anti‑diagonal 上的元素互不影响，从而把原本四阶的向量化舍入转化为三阶的 anti‑diagonal 并行舍入。

**🔧 技术方法**

利用 Kronecker 乘积的 LDL 分解得到反馈矩阵 Uᵀ⊗L；采用 anti‑diagonal 并行推进、局部行列压缩（fold）以及块级带状矩阵乘法（lazy block update）等技术，减少 O(m²n²) 的工作量到 O(mn·max(m,n))。

**📊 数据集**

本文未使用特定数据集，而是以理论分析和算法复杂度评估为主；若在实验中使用，可能采用常见的神经网络权重矩阵作为量化目标。

**📈 对比分析**

与直接向量化舍入、密集 anti‑diagonal 以及其他 GPTQ 变体相比，GPTQ-2D 在时间复杂度上从 O(m²n²) 降至 O(mn·max(m,n))，与单侧 GPTQ 在 m≥n 时实现相同的总成本，并保持 O(m+n-1) 的并行深度。

**⚠️ 局限性**

局限性包括：① 需要预先固定 A、B（Gram 矩阵）以保证等价性，若在舍入过程中动态调整则可能失效；② 仍不具备全局最优性，仅实现固定顺序的局部最优舍入；③ 对于极端宽矩阵，填充偏置会导致额外存储开销。

---

## 378. BayesAME: Bayesian Active Model Evaluation

**arXiv ID:** 2607.27023 | [PDF](https://arxiv.org/pdf/2607.27023v1)

**作者:** Paula Cordero Encinar `[一作]` (Imperial College London), Silvia Chiappa `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于贝叶斯序贯框架的自动确定核心样本集（coreset）大小的模型评估方法，能够在可靠性能估计优先的场景下，动态停止评估并给出置信区间。

**💡 创新点**

创新点在于：①将模型性能建模为潜在能力的随机变量，并通过共享参考模型得分构造协方差；②利用信息增益准则主动挑选评估样本；③引入多目标协同模型，捕捉不同目标模型间的性能相关性，从而进一步压缩coreset；④提供可自动或手动指定coreset大小的灵活接口。

**🔧 技术方法**

技术方法包括：高斯过程/贝叶斯线性模型、协方差核设计（SE、Matérn-3/2）、信息增益与置信区间判定、Cholesky递推求解、线性核心协同建模（coregionalization）以及贝叶斯岭回归等。

**📊 数据集**

使用了多个公开基准数据集：Open LLM Leaderboard（GPQA、MMLU-Pro、BBH、ARC-Challenge、MuSR）、IFEval、HELM Lite（Natural QA Openbook），涵盖二元与连续得分两种评分形式。

**📈 对比分析**

与现有随机/空间覆盖/贝叶斯岭回归/IRT等方法对比，本文方法在全范围coreset大小内及自动停止点上均实现更低的RMSE，Pareto前沿明显优于对照组，且在高相关目标模型情形下多目标扩展更显优势。

**⚠️ 局限性**

局限性包括：仅处理标量得分，未对多维或结构化评估结果扩展；对参考模型的依赖仍存在，若参考模型稀缺或质量差可能影响性能；并未结合模型架构、训练超参等外部信息进一步提升先验。

---

## 379. SymmGrid: Super-Scaling On-Robot Learning with Parallelized Symmetries and Egocentric-Exocentric Visual Perception

**arXiv ID:** 2607.26985 | [PDF](https://arxiv.org/pdf/2607.26985v1)

**作者:** Gabe Everett `[一作]` (Lipscomb University), Juan Rojas `[通讯]` (Lipscomb University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 SymmGrid，一种基于分支对称的轨迹级数据增强框架，直接在真实机器人上加速深度强化学习训练。

**💡 创新点**

创新点在于：①将对称性建模为 MDP 的对称树，利用常数分支实现 K×K 网格的超大规模并行对称变换；②对视角变化的 exocentric 场景引入预先计算的 homography；③采用指针式 replay buffer 和计算友好的数据存储，显著提升训练效率。

**🔧 技术方法**

使用的技术包括：Markov 决策过程对称树、仿射变换与 K×K 网格构造、全局变换（状态、动作）以及基于相机标定的 homography 变换；网络结构为预训练 ResNet-10 + 2 层 MLP，使用 RLPD（强化学习+演示）算法。

**📊 数据集**

数据集为真实 Franka FR3 机器人上的三项接触任务：插销（peg‑insertion）、电缆铺设（cable‑routing）和物体搬运（object‑relocation），每项任务均采用 20–30 次演示进行预热。

**📈 对比分析**

与 SERL 基线相比，SymmGrid 在 peg‑insertion、cable‑routing、object‑relocation 的 wall‑clock 训练时间分别提升 1.41×、2.17×、1.37×；评估成功率提升 1.09×–1.27×；轨迹宽度 nAUC 比例最高达 2.59×，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括：①对 exocentric 场景的 homography 可能产生视角扭曲、边缘伪影；②过大网格或宽工作空间会导致样本冗余、视觉编码器偏向伪影；③目前仅支持平移变换，缺乏旋转或非线性变换的扩展；④对相机标定要求高，若标定误差大影响效果。

---

## 380. TREK: A Travel Reasoning and Evaluation Kit for LLM Agents in Complex Trip Planning

**arXiv ID:** 2607.26977 | [PDF](https://arxiv.org/pdf/2607.26977v1)

**作者:** Jinhu Qi `[一作]` (Chinese University of Hong Kong), Irwin King `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 TREK 旅行行程合成基准，评估 LLM 代理在多约束条件下生成可执行行程的能力。

**💡 创新点**

创新点在于：① 采用完全确定性、无 LLM 判定的评估器；② 通过人工核实的 gold 标注保证 1.0 最高分可达；③ 任务集合包含 800 个可行/不可行的多约束查询，并标注可验证的不可行原因；④ 提供生产级 RESTful 工具沙箱。

**🔧 技术方法**

技术：规则式评估器、基于语义搜索的工具接口、人工标注的 gold 行程、可重复的合成知识库构建脚本。

**📊 数据集**

数据集：212,530 条合成记录（航班、酒店、景点、租车），覆盖 375 个城市，13 种旅行者人设，共 800 个任务。

**📈 对比分析**

比较方法：在 15 种 LLM 代理上评估任务完美率，最强 GPT‑5.6 在可行任务上仅达 46.2% 完全可行率，显著低于 1.0，显示仍有大幅提升空间。

**⚠️ 局限性**

限制：基于合成数据，真实性受限；评估仅覆盖单一可执行行程，未考虑多方案和动态更新；对不可行任务的拒绝需要人工判定；对 reasoning 与 compute 的影响仅在单一对比中观察到。

---

## 381. Credit Cards, Confusion, Computation, and Consequences: What Can We Uncover About Language Model Reasoning?

**arXiv ID:** 2607.26952 | [PDF](https://arxiv.org/pdf/2607.26952v1)

**作者:** Arnav Hiray `[一作]` (Georgia Institute of Technology), Sudheer Chava `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了 CreditCardQA 这一基于真实信用卡协议的金融素养数值推理基准，分析 LLM 与 LRM 在该任务中的表现。

**💡 创新点**

首次构造针对真实信用卡协议的数值推理数据集，并通过错误分析与回归模型揭示推理难点；发现 Program-of-Thought (PoT) 能显著提升模型性能。

**🔧 技术方法**

采用 Chain-of-Thought 与 Program-of-Thought 推理范式，对 11 种开源与闭源大型模型进行实验；利用手工标注与逻辑回归对错误类型与难度进行深入分析。

**📊 数据集**

使用 CreditCardQA 数据集：从 27 张信用卡协议（美国与欧洲）中提取 1800 个问答对（800 训练 + 1000 测试），涵盖多种卡种与条款。

**📈 对比分析**

按 ±0.2% 与 ±5% 的相对误差阈值评估模型准确率；在 PoT 下，GPT-OSS‑120B、Gemini 3.0 Pro 与 GPT‑5 达到约 80% 的准确率，PoT 对弱基线模型提升 0.9–6.2%。

**⚠️ 局限性**

模型仍受合同复杂性限制，错误主要源于误用金融规则、忽略条件与文档特异性；边缘案例对低收入/脆弱人群影响更大，表明需更强的领域感知推理能力。

---

## 382. VITAL-RAG: Invariance Race for Context Allocation in Coding Agents

**arXiv ID:** 2607.26937 | [PDF](https://arxiv.org/pdf/2607.26937v1)

**作者:** Zijian Lu `[一作]` (Nanjing University of Posts and Telecommunications), Weibei Fan `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 VITAL‑RAG，一种在代码检索‑生成系统中进行上下文分配的层，解决多视图冗余导致的上下文碎片化问题。

**💡 创新点**

引入了“选择性不变性”(selective invariance)原则，通过对象级去重、局部任务相关同伴保留以及按 token 预算渲染来平衡不变性与任务相关性。

**🔧 技术方法**

采用了基于检索器的原型保留分配器、对象级视图归约、任务驱动的同伴筛选、预算约束的 token 分配，以及可选的程序语义转移（Program‑Semantic Transfer）决策。

**📊 数据集**

主要在 RepoBench、RepoClassBench、RepoExec 三大仓库级基准上评估，包含 Java/Python 任务，共计 16,490 个仓库级证据任务。

**📈 对比分析**

与 CodeRAG、GraphCoder、RepoScope 等现有 RAG 系统在同一检索‑生成流水线下比较；VITAL‑RAG 在 Recall@4K 上从 39.59% 提升至 63.67%，并减少 35.63% 证据 token；在 RepoClassBench 上 Token‑F1/Char 相当或领先；在 RepoExec 上 Pass@1 最高（Gpt‑5.4 57.75%，Claude 65.63%，Qwen 21.69%）。

**⚠️ 局限性**

依赖仓库索引提供的代码对象标识（路径+名称），适用固定的 token 预算和对象上限；未探究更大上下文窗口或更复杂索引的影响。

---

## 383. Defending Against Backdoor Attacks via Alignment Checking in Model-Contrastive Federated Learning

**arXiv ID:** 2607.26933 | [PDF](https://arxiv.org/pdf/2607.26933v1)

**作者:** Hongliang Zhang `[一作]` (Qilu University of Technology (Shandong Academy of Sciences)), Jiguo Yu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种双阶段方法 FedDAB，用于在联邦学习中防御后门攻击。

**💡 创新点**

创新点在于：① 在局部训练中加入既能对齐方向又能对齐幅度的模型对比正则项；② 服务器端同时执行整体方向检查与参数级对齐检查，并利用历史签名信息提升检测精度；③ 使用中位数 Z‑score 对异常更新进行鲁棒过滤。

**🔧 技术方法**

主要技术包括对比学习（model‑contrastive regularization）、方向/幅度一致性约束、median‑based Z‑score、历史签名缓冲（global & local sign buffers）、FedAvg 等基础聚合。

**📊 数据集**

实验数据集：FMNIST、CIFAR‑10、CIFAR‑100（图像）和 Sentiment140（文本）。

**📈 对比分析**

与 FedAvg、MKrum、Foolsgold、RLR、RFA、MMetric、RoseAgg、AlignIns、EndPCA 等方法对比，FedDAB 在所有攻击（BadNet、DBA、Neurotoxin、PGD）和不同非 IID、节点比例下都实现了更低的 ASR、更高的 RR，且主任务准确率（TA）保持或略有提升，整体性能优于现有防御方案。

**⚠️ 局限性**

局限性：当恶意节点比例超过 50% 时防御效果显著下降；需要维护历史签名缓冲，导致服务器内存开销；仍存在被恶意节点复制/传播全局模型的风险。

---

## 384. Latent-IM: Latent Interaction Management for Speech LLMs

**arXiv ID:** 2607.26928 | [PDF](https://arxiv.org/pdf/2607.26928v1)

**作者:** Adar Avsian `[一作]` (Georgia Institute of Technology), Larry Heck `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Latent-IM框架，在冻结的语音LLM中通过激活读取实现对话动作的选择与实现。

**💡 创新点**

将对话管理拆解为内部状态估计与动作控制，并通过激活向量学习可迁移的动作子空间，实现无需标注或微调的可控生成。

**🔧 技术方法**

利用残差流激活的流式控制器、Fisher子空间方向、norm‑relative激活注入、转录嵌入、PCA+Ledoit–Wolf估计等技术。

**📊 数据集**

MapTask、FindTask、CReST三大任务导向语音对话语料。

**📈 对比分析**

与多种基线（prompting、SFT、直接调参等）对比，在oracle和端到端控制下，Latent‑IM平均移动准确率提升12.5点、实现准确率最高，BLEU表现与SFT相近。

**⚠️ 局限性**

仍受人类动作不确定性的限制，选择精度受限于人类一致性；对非结构化任务的迁移和长期对话中的策略优化尚未解决。

---

## 385. Deductive Verification for Earliest Deadline First Scheduler Implementations

**arXiv ID:** 2607.26927 | [PDF](https://arxiv.org/pdf/2607.26927v1)

**作者:** Daniel Kuhse `[一作]` (TU Dortmund University), Jian-Jia Chen `[通讯]` (TU Dortmund University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对现有实时操作系统（RTOS）中的Earliest Deadline First（EDF）调度器实现进行形式化验证，提出统一的三条正确性属性，并在RTEMS 5、RTEMS 6和修改后的FreeRTOS上验证。

**💡 创新点**

创新点在于：①提出了适用于已有RTOS的EDF正确性三属性（优先级映射、就绪队列管理、调度决策）框架；②在不重构内核的情况下，将Deductive Verification（Frama‑C/ACSL）直接应用到实际实现；③发现并修复了FreeRTOS EDF改造中的超时重入问题。

**🔧 技术方法**

使用的技术包括：Frama‑C 32 的 WP 插件进行归约证明；ACSL 注解来描述函数契约与 invariant；Typed 内存模型和 ghost 变量模拟 volatile 状态；抽象数据结构模型（红黑树、链表）来隔离底层实现；以及 Alt-Ergo SMT 解决器完成证明。

**📊 数据集**

未使用专门的数据集，而是基于 RTEMS 和 FreeRTOS 源码自身的任务模型（周期、相对截止时间）进行验证；验证涵盖了所有入口点、内部辅助函数与核心调度逻辑。

**📈 对比分析**

通过比较两种实现的证明目标与执行时间：RTEMS 5/6 共产生约 1584/1722 条证明目标，验证时间 707.5/402.6 秒；FreeRTOS 共 429 条证明目标，验证时间 121.4 秒，说明框架在不同 RTOS 上具有可扩展性与较低的人工验证成本。

**⚠️ 局限性**

局限性包括：①假设底层内核原语（如上下文切换、红黑树操作、列表操作）已正确实现；②无法验证低层硬件相关的汇编代码；③需要手动为宏和不易建模的指针操作编写包装与契约，增加了初始工作量；④对 SMP（多核）情形的支持有限。

---

## 386. Two Calls Beat Five Agents: Evaluating Multi-Agent Pipelines Against Self-Refinement for Local Language Models

**arXiv ID:** 2607.26922 | [PDF](https://arxiv.org/pdf/2607.26922v1)

**作者:** Ashish Prajapati `[一作]`, Om Mohite `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对本地7B Qwen2.5-7B-Instruct模型进行多角色Pipeline（Parishad）与两次Self‑Refine的实证评估，并与直接Prompt对比，探讨通信格式与实现细节对性能的影响。

**💡 创新点**

首次在小模型上系统验证多角色Pipeline与Self‑Refine，并揭示JSON交互导致的错误累积、Plain‑Text恢复以及任务感知的门控Self‑Refine能显著提升代码任务表现。

**🔧 技术方法**

使用五角色Pipeline（Refiner、Planner、Worker、Checker、Judge）、两次Self‑Refine（V1无门控、V2任务门控）以及直接单调用Prompt，配合Qwen2.5-7B-Instruct的LLM推理。

**📊 数据集**

GSM8K（数学推理500问）与HumanEval（代码生成164问）两个公开基准数据集。

**📈 对比分析**

相较于直接Prompt，Pipeline V1（JSON）显著下降；Pipeline V2（Plain‑Text）恢复至82%/81.7%；Self‑Refine V1在GSM8K提升至86.2%且Token使用减少7.4×；在HumanEval V1导致性能下滑至66.5%，而V2通过门控保持95.1%。

**⚠️ 局限性**

仅评估单一模型家族（Qwen2.5-7B-Instruct），未覆盖其他小模型；Pipeline V1在HumanEval仅测试50问；未测试多跳推理、事实问答或长文本生成等场景，且未考察更大模型效果。

---

## 387. From Keypoints to Predictive Distributions: Post-Hoc Uncertainty for YOLO-Pose Models

**arXiv ID:** 2607.26921 | [PDF](https://arxiv.org/pdf/2607.26921v1)

**作者:** Alexej Klushyn `[一作]` (Airbus Central Research & Technology), Jayant Sen Gupta `[通讯]` (Airbus Central Research & Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

为已训练的 YOLO‑Pose 模型提供轻量后置概率扩展，冻结原模型并为每个关键点学习一个二维可变异异正定协方差矩阵，随后通过 Gaussian 或 Student‑t 校准得到可用的预测分布。

**💡 创新点**

创新点在于：① 只对已完成的 YOLO‑Pose 进行后置学习，保持原始预测不变；② 采用重要性加权负对数似然和 Cholesky 分解实现可变异异分布；③ 提出两种校准方案（Gaussian 与 Student‑t）以及一套完整的关键点级别分布校准诊断和新的 akp 评估指标。

**🔧 技术方法**

技术包括：轻量化概率头、Cholesky 矩阵分解保证正定、重要性加权 NLL 训练、温度校准、Student‑t 重尾建模、不确定性剪枝和 EKF 兼容的协方差传播。

**📊 数据集**

使用的主要数据集为 COCO 人姿态数据集（用于验证校准与排名），以及内部航空着陆数据集（用于飞行器着陆的实际应用实验）。

**📈 对比分析**

通过与未校准、Gaussian 校准、Student‑t 校准三种方式比较，利用 ACE、ENCE、NLL、akp 等指标评估；Student‑t 校准在 COCO 上实现最佳分布校准且保持 ap ；不确定性剪枝进一步提升 NLL 与 ACE；在飞行器着陆实验中，校准后的协方差能够被直接用于 PnP 与 EKF，满足安全关键性要求。

**⚠️ 局限性**

局限性在于：仅对经验残差进行校准，无法捕捉分布漂移或雾天等未知场景下的判别性不确定性；对相机内参/外参误差的传播误差未进一步建模；在极端环境下校准效果会下降。

---

## 388. Algorithms for Linear Ordinary Differential Operators

**arXiv ID:** 2607.27003 | [PDF](https://arxiv.org/pdf/2607.27003v1)

**作者:** Jean Della Dora `[一作]` (Institut IMAG), Stephen M. Watt `[通讯]` (IBM Thomas J. Watson Research Center)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在Scratchpad II中实现了线性常微分算子及其算术运算；

**💡 创新点**

提出了在非交换环上实现除法、GCD、LCM、扩展欧几里得算法和Ore定位的具体构造；

**🔧 技术方法**

采用抽象数据类型、非交换多项式表示、Heisenberg换位律、伪除法、Newton多边形分析和Ricatti方程求解等技术；

**📊 数据集**

未使用专门的数据集，所有示例均为符号计算示例；

**📈 对比分析**

通过符号例子验证算法正确性，但未给出定量性能对比；

**⚠️ 局限性**

伪除法会导致中间表达式膨胀，求解过程需处理非整数斜率、可能的代数扩展，且算法在多项式系数时效率有限。

---

## 389. InkShield: Writing Style Protection Against Unauthorized Handwriting Mimicry

**arXiv ID:** 2607.26976 | [PDF](https://arxiv.org/pdf/2607.26976v1)

**作者:** Jian Xiong `[一作]` (University of Electronic Science and Technology of China), Guowen Xu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在发布前对手写体参考图像进行低可见扰动的主动防御方法，以阻止一击式手写风格模仿。

**💡 创新点**

创新点包括：①通过从解码器池中选择与目标作家相距较远的“替身作家”来定义风格位移方向；②采用书写笔触边缘掩模，将扰动限制在笔画边缘，避免在空白背景产生明显痕迹；③使用冻结的手写生成器（One-DM）作为替代模型，通过对多段伪内容文本的解码误差来指导扰动优化，实现跨文本的泛化。

**🔧 技术方法**

核心技术包括：风格表示与距离计算、笔触边缘检测与掩模构造、LPIPS感知损失、PGD投影梯度上升、冻结替代生成器的 denoising 损失，整体构成“Decoy-guided Edge-constrained Surrogate-guided Disruption (DESD)”算法。

**📊 数据集**

实验使用 IAM 手写文字数据集，选取 161 名测试作者的 80 个 OOV-U 内容单词，评估 12,880 张生成样本。

**📈 对比分析**

与 DiffusionGuard、PhotoGuard、StyleGuard、Anti‑DreamBooth 等现有图像保护方法比较。实验表明在 One-DM 生成器上，DESD 将目标作家 Top‑1/Top‑5 检索率从 13.20%/38.73%（或 10.68%/34.31%）降至 2.12%/9.21%（或 1.93%/8.37%），LPIPS 仅为 0.0078，背景扰动几乎为 0，且字符错误率保持 17.4% 左右，证明了在抑制风格模仿与保持可读性、视觉质量之间取得了显著平衡。

**⚠️ 局限性**

局限性包括：①对替代生成器的依赖，若攻击者使用与训练完全不同的生成模型，可能降低防御效果；②保护效果仍需在不同分辨率与字体类型上进一步验证；③在极端的替身作家选择或更高扰动预算下，可能出现可见伪影，需平衡扰动幅度；④目前仅在单词级别测试，尚未在行文本或多行文档场景进行评估。

---

## 390. Assurance-Scoped Reliability for Agentic Networks: Capturing the State That Matters

**arXiv ID:** 2607.26953 | [PDF](https://arxiv.org/pdf/2607.26953v1)

**作者:** Bilgehan Erman `[一作]`, Nikos Papadis `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出一种通用的可靠性保证智能（RAI）体系结构，用于在代理网络系统中捕获、记录和验证关键的耐久状态，从而实现对代理生命周期管理器等自适应服务的可靠性保证。

**💡 创新点**

创新点包括：① 将服务行为的描述（ASD）与可靠性约束（ASRP）分离，形成可复用的规范链；② 设计基于ASC的通用行为语言；③ 引入上下文胶囊与观察者-控制器闭环，实现风险驱动的动态记录；④ 提供可验证的架构和方法学，支持跨域、增量升级和自适应记录。

**🔧 技术方法**

使用的技术主要是：1）Agentic Service Calculus（ASC）作为服务行为建模语言；2）Agentic Service Description（ASD）与Agentic Service Reliability Profile（ASRP）作为设计时工件；3）上下文胶囊（context capsule）与观察者（observer）以及胶囊控制器（capsule controller）构成的运行时保证环；4）基于事件流、日志、策略代码和审计记录的技术组合，辅以安全与隐私保护策略。

**📊 数据集**

本文并未在实验数据集上进行评估，而是在示例“确定性网络生命周期管理器”上说明概念实现；因此数据集为“无”或“示例系统”。

**📈 对比分析**

比较方法为与三种记录策略对比：完整轨迹、固定最小记录和自适应胶囊；预期性能包括：① 存储效率提升；② 在高风险工作负载下保留必要证据；③ 观察者能检测到传统指标未发现的保证缺口；④ 控制器在模式转换上保持稳定；⑤ 通过重试键和提交键实现更好地恢复与补偿。由于论文为概念性设计，未给出量化实验结果。

**⚠️ 局限性**

局限性包括：① ASC的形式化与工具化仍缺乏成熟验证，难以在大规模服务中自动检查；② 跨域语义与证据封装的细化尚未完成，影响端到端保证；③ 自适应记录在高负载下可能引入额外开销，需进一步评估稳定性；④ 证据最小化与隐私保护之间的权衡尚未系统化；⑤ 论文仅关注非恶意、运行时故障，未覆盖拜占庭攻击、证据伪造等安全威胁。

---

## 391. Surrogate assisted diversity estimation in neural ensemble search

**arXiv ID:** 2607.26940 | [PDF](https://arxiv.org/pdf/2607.26940v1)

**作者:** Alexandr Udeneev `[一作]`, Oleg Bakhteev `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了双目标 surrogate-guided 框架，通过训练两个 surrogate 模型来同时估计单模型准确率和架构多样性，并以此高效搜索神经网络集成。

**💡 创新点**

创新点在于将多样性显式建模为潜在空间距离，使用 GAT + triplet loss 生成可区分的架构嵌入，并在搜索时将准确率与多样性联合优化。

**🔧 技术方法**

采用图注意网络 (GAT)、三元组损失 (triplet loss)、余弦退火学习率、基于 DAG 的操作集成，以及贪心多样性驱动的集成构造。

**📊 数据集**

在 FashionMNIST、CIFAR‑10 与 CIFAR‑100 三个公开图像分类数据集上构建约 3,000 个训练模型用于 surrogate 训练。

**📈 对比分析**

与 Deep Ensembles 和随机搜索对比，Surrogate Ensemble 在所有数据集上取得相近或更优的 Top‑1 Accuracy 与 NLL，且在 CIFAR‑100 上更显著。

**⚠️ 局限性**

主要限制是构建 surrogate 数据集需要大量训练模型，导致计算成本高；未来可通过更高效采样或权重共享降低开销。

---

## 392. BioVLN: A Simulation Platform for Visual Language Navigation in Biomedical Laboratories

**arXiv ID:** 2607.26914 | [PDF](https://arxiv.org/pdf/2607.26914v1)

**作者:** Zhe Liu `[一作]` (East China University of Science and Technology), Dongzhan Zhou `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BioVLN平台，用三区域操作面模型实现实验室仪器导航。

**💡 创新点**

创新在于将仪器划分为物理体、清除区和操作区，并将目标定义为操作面位置。

**🔧 技术方法**

使用基于Blender的场景生成、LSAT注释工具、Habitat-sim仿真、RL与VLM（如GPT‑4o、DeepSeek、SAM）进行评估。

**📊 数据集**

构建了47个实验室场景共1667个导航任务，包含MS DEV/VAL/TEST、Two‑Room、LSAT Target等数据集。

**📈 对比分析**

通过六种方法对比，Frontier达到74.4–87.5%成功率，3‑Zone Oracle 83.3–92.5%，VLM方法仅19.4–62.5%，显示几何探索已逼近上限。

**⚠️ 局限性**

局限在于资产视觉质量低导致VLM识别差、场景多样性不足、缺乏跨域与真实机器人部署验证。

---

## 393. Correlated Chance Sampling for Monte Carlo Counterfactual Regret Minimization

**arXiv ID:** 2607.27035 | [PDF](https://arxiv.org/pdf/2607.27035v1)

**作者:** Boning Li `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于持久低失真序列的相关机会抽样方法（CCS-MCCFR），在MCCFR中取代独立机会抽样，显著降低了扑克等不完全信息游戏的可利用性。

**💡 创新点**

创新点在于为每个具体机会节点引入持久的随机Weyl序列，使得同一节点在多次访问时的机会分布保持低失真，并证明了在固定策略轨迹下的无偏性和局部频率误差为O(logN/N)。

**🔧 技术方法**

技术核心包括持久随机Weyl序列（Golden Ratio），外部采样（External Sampling）框架，回溯误差分析以及对调度的条件总变差界定。

**📊 数据集**

实验数据集涵盖经典桌面扑克（Kuhn、Leduc、Goofspiel-4）、非扑克游戏（Liar's Dice、Flop Hold'em）以及HUNL子游戏，全部使用OpenSpiel实现。

**📈 对比分析**

通过与i.i.d.抽样、反向抽样和不同更新规则（Linear CFR、Discounted CFR）进行配对Bootstrap检验，CCS-MCCFR在Kuhn和Leduc上分别降低约27.6%和24.6%可利用性，在Leduc-10卡上提高34%，Goofspiel-4也有4.3%的提升，且在大规模Leduc实验中保持24.5%改进。

**⚠️ 局限性**

局限性包括在低重访或对称机会结构的游戏（如Liar's Dice、Flop Hold'em、HUNL子游戏）中几乎无效，且理论分析仍未完全解释在高度自适应策略更新中的全局误差控制。

---

## 394. Lottery Tickets Are Not Deployment Tickets

**arXiv ID:** 2607.27031 | [PDF](https://arxiv.org/pdf/2607.27031v1)

**作者:** Bum Jun Kim `[一作]` `[通讯]` (University of Tokyo), Bum Jun Kim (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估稀疏网络（如 Lottery Ticket、随机稀疏、剪枝等）在实际部署中是否能作为密集模型的 drop‑in 替换，并提出兼容性审计方法。

**💡 创新点**

提出行为兼容性距离和固定阈值政策诊断，证明仅靠准确率恢复不足以保证部署兼容，并设计 BP‑LT 选择机制来最小化行为差异。

**🔧 技术方法**

使用稀疏化技术、行为指标（校准误差、OOD AUROC、接受率、腐败准确率、预测一致性）以及理论分析（接受率翻转的上界）。

**📊 数据集**

实验数据集包括 CIFAR‑10/100、Imagenette、Flowers‑102、FGVC‑Aircraft 等。

**📈 对比分析**

与原始密集模型在清晰准确率和行为距离上对比；发现即使准确率相同，行为距离可达 0.5‑1.0，接受率翻转率约 7–10%，并且某些方法在腐败准确率上下降 1–2%。

**⚠️ 局限性**

仅考虑固定阈值的决策场景，未覆盖所有部署变体；BP‑LT 能缓解但不消除政策翻转；实验范围局限于中等稀疏率与单一模型，未探索更高稀疏率或多模型集成的影响。

---

## 395. TreeCCA: Canonical Correlation Analysis via Gradient-Boosted Trees

**arXiv ID:** 2607.27027 | [PDF](https://arxiv.org/pdf/2607.27027v1)

**作者:** James Chapman `[一作]` `[通讯]`, James Chapman

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 TreeCCA，一种利用梯度提升树（GBT）作为编码器进行端到端的非线性典范相关分析（CCA）方法。

**💡 创新点**

创新点在于将 Eckart‑Young（EY）损失的闭式梯度与 GBT 的自定义目标无缝结合，使得传统树模型即可学习跨视图关联，兼具非线性捕获与可解释性。

**🔧 技术方法**

使用的技术包括：Eckart‑Young 损失、梯度提升树（XGBoost / LightGBM）、增量嵌入缓存、单元 Hessian 归一化梯度、多视图扩展 TreeMCCA 以及单编码器自监督变体 TreeCCA‑SSL。

**📊 数据集**

实验数据集涵盖合成（Signed‑Power、Hermite）、Split MNIST、UCI HAR、以及五个公开多视图数据集（Caltech101‑7、3Sources、NUS‑WIDE、Handwritten、MSRC‑v5）。

**📈 对比分析**

与深度 CCA、核 CCA、线性 CCA、PMD 等基线比较，TreeCCA 在合成数据上 7.5% 提升 TCC、在 Split MNIST 上 1.04× 的泛化比率、在 UCI HAR 上 5× 更低的训练时长、在五个多视图数据集上均超过 MCCA 的 TCC，并在非线性信号强的数据集上取得显著性能。

**⚠️ 局限性**

局限性包括：在高 p/N 情况下稀疏恢复精度下降、缺乏严格的交替 GBT 收敛理论、以及自监督版本仍需设计合适的表格数据增广方法。

---

## 396. Evaluating Regional Bias in LLMs From Abstract Stereotype to Concrete Social Decision-Making

**arXiv ID:** 2607.27022 | [PDF](https://arxiv.org/pdf/2607.27022v1)

**作者:** Jiayuan Di `[一作]` (East China University of Science and Technology), Yiming Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并使用S2D框架，对6种主流大型语言模型在34个中国省级行政区上的地区偏差进行从抽象刻板印象到具体决策的系统评估。

**💡 创新点**

首次将Stereotype Content Model与Spontaneous Stereotype Content Model相结合，形成从温暖与能力两个维度的刻板印象到教育、工作与社交决策的端到端评估体系，并揭示了偏差与地区经济数字发展指标的关联。

**🔧 技术方法**

采用基于Likert量表的抽象刻板印象评估与配对选择的决策任务，结合标准差、范围和Spearman相关性等统计指标，对6个LLM的输出进行量化比较。

**📊 数据集**

使用所有34个中国省级行政区的行政区列表以及对应的GDP、居民可支配收入和宽带覆盖率等区域经济与数字发展数据，模型评估采用自定义中英文提示。

**📈 对比分析**

通过标准差、范围和模型间Spearman相关性评估偏差强度，结果显示大部分模型在能力维度和工作决策任务上具有高度一致的地区排名，且偏差程度在不同模型间存在显著差异。

**⚠️ 局限性**

局限性包括仅覆盖中国省级地区，缺乏跨国对比；评估依赖提示设计，可能受语言与文化差异影响；未对偏差缓解策略进行探索。

---

## 397. Upper bounds for the monotone rank of the unique disjointness matrix

**arXiv ID:** 2607.27014 | [PDF](https://arxiv.org/pdf/2607.27014v1)

**作者:** Igor S. Sergeev `[一作]` `[通讯]`, Igor S. Sergeev

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对唯一不相交矩阵（UDISJ_n）的单调秩（覆盖秩）给出了上界，并证明该上界与已知的下界1.5^n基本匹配；同时给出了该矩阵的分区秩上界为1.89^n。

**💡 创新点**

创新点在于：利用码分割和Sapozhenko–Lovász–Stein梯度覆盖引理，构造了高效的矩形覆盖，从而将上界精确到n^O(1)(3/2)^n，并首次给出了分区秩的紧致上界。

**🔧 技术方法**

主要技术包括：1) 通过将n维布尔立方体分成3码集合来分割行列；2) 采用熵函数估计组合数和矩形权重；3) 随机划分集合并使用分离变量技巧构造覆盖族；4) 采用Sapozhenko–Lovász–Stein引理将均匀覆盖转换为最小覆盖。

**📊 数据集**

该研究为纯理论分析，未使用任何实际数据集；所有结果均基于组合数学与信息理论的定理推导。

**📈 对比分析**

与之前的下界（1.5^n）比较，本文的上界几乎相等，表明已知下界基本紧致；与先前的粗略上界O(√3^n)相比，改进显著；对于分区秩，1.89^n的上界也高于先前的粗略估计。

**⚠️ 局限性**

局限性：上界中包含n^O(1)的多项式因子，精确系数仍不明确；方法依赖于唯一不相交矩阵的特殊结构，尚未证明可推广到更一般的矩阵或更广泛的沟通复杂度问题。

---

## 398. Belief-Guided Decision Making with Uncertainty Gating in the Game of Go

**arXiv ID:** 2607.26946 | [PDF](https://arxiv.org/pdf/2607.26946v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 399. Parameterized Complexity of Fair Coloring Problem

**arXiv ID:** 2607.27004 | [PDF](https://arxiv.org/pdf/2607.27004v1)

**作者:** Ramin Javadi `[一作]` (Isfahan University of Technology), Hossein Shokouhi `[通讯]` (Isfahan University of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了公平着色问题的参数化复杂性，特别是针对输入图的结构参数，如模块宽度、树深度、邻域多样性和反馈顶点数。

**💡 创新点**

证明了在森林和模块宽度为二的图中，公平着色问题在组数方面是W[1]-困难的；同时，当颜色数为二时，问题在邻域多样性方面是FPT的。

**🔧 技术方法**

使用了参数化复杂性理论和图论中的相关算法技术。

**📊 数据集**

未具体提及使用的数据集，但讨论了图的结构特性，如森林和模块宽度为二的图。

**📈 对比分析**

与现有方法的比较显示，当颜色数为二时，问题在邻域多样性和组数方面是FPT的，而在组数方面是W[1]-困难的，表明在某些情况下问题的复杂性显著提高。

**⚠️ 局限性**

限制在于对于某些参数的复杂性仍然未知，例如在一般情况下，邻域多样性是否是FPT，以及在组数方面的多项式核或更强的下界的建立仍然是一个开放的挑战。

---

## 400. AgentSnare: Learning to Delay, Divert, and Defuse Autonomous Penetration Agents

**arXiv ID:** 2607.26998 | [PDF](https://arxiv.org/pdf/2607.26998v1)

**作者:** Ruoyu Wang `[一作]` (University of Hong Kong), Tianhang Zheng `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 AgentSnare，利用轨迹自适应生成并验证全局一致性的伪造事实，构建动态的“假”环境以误导大型语言模型自动化渗透测试代理。

**💡 创新点**

创新点在于将防御从传统静态蜜罐转变为根据代理交互历史实时生成并验证伪造事实，从而实现持续的轨迹级干扰与全局一致性保证。

**🔧 技术方法**

技术手段包括：基于 Qwen3‑4B 的 LoRA 微调的artifact‑construction policy、在线验证与状态维护的 ShadowOS、以及使用反应引导学习的模拟交互轨迹训练。

**📊 数据集**

数据集为 CVE‑Bench 的 15 个可部署漏洞任务与多种攻击模型（Claude Opus、GPT‑5.4、MiniMax M3）产生的交互轨迹。

**📈 对比分析**

在与静态防御（13 个干预实例）、Cowrie 交互 honeypot 以及 HoneyLLMd 对比中，AgentSnare 在 Delay 46.8%、Divert 55.9%、Defuse 90.0% 三项指标上表现最佳，并在 45 个攻击–CVE 对中实现零真实目标成功率。

**⚠️ 局限性**

局限性包括：对未知攻击策略或环境的泛化能力有限；实现复杂度高，需要预先训练的模型与状态管理，对资源受限或实时性要求严格的部署场景存在挑战。

---

## 401. OptimismBench: Forecasting Bias and the Alignment Effect in Language Model Judgment

**arXiv ID:** 2607.26981 | [PDF](https://arxiv.org/pdf/2607.26981v1)

**作者:** Seonglae Cho `[一作]` (Holistic AI), Adriano Koshiyama `[通讯]` (Holistic AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为 OptimismBench 的基准，用来检测大型语言模型在概率判断中的方向性偏差（乐观或悲观）

**💡 创新点**

使用了“倒置对”设计，通过让模型分别给出正面和负面结果的概率来计算带符号的偏差 Skew，避免了传统校准度量无法捕捉方向性的问题

**🔧 技术方法**

基于 0–100 评分量表的倒置对测量，配合多轨道（校准、估计、推荐、显著性）和对齐干预（自我偏差警告、温度、视角、叙事上下文），并通过 ANOVA 等统计分解

**📊 数据集**

发布了 3,870 条包含 10 种语言、60 个倒置对场景（6 个领域）以及 15 条校准项的基准集合

**📈 对比分析**

在 16 个来自 8 供应商的模型上评估，发现 14 个模型表现为乐观，Anthropic frontier 模型为悲观；模型身份导致的方差为 4.7 倍语言方差，表明模型偏差是主要变量；对齐训练对偏差方向有显著影响

**⚠️ 局限性**

倒置对测量只能检测内部一致性，而非与人类真实概率的偏差；对齐方向的结论依赖于有限的模型对，且跨语言结果受翻译影响

---

## 402. What Does It Take to Detect an AI Agent? Minimal Feature Sets for Behavioral Detection under Browser Automation

**arXiv ID:** 2607.26935 | [PDF](https://arxiv.org/pdf/2607.26935v1)

**作者:** Vishisht Choudhary `[一作]` (Technical University of Munich), Jens Ernstberger `[通讯]` (Kontext)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了一个三类（人类、传统脚本型机器人、LLM驱动的浏览器代理）检测框架，并通过特征工程、模型训练与特征子集搜索，找出了实现 100% 代理检测的最小特征集合，同时评估了多级对抗逃逸策略的鲁棒性。

**💡 创新点**

创新点主要有：① 抛弃传统二分类假设，证明二分类模型会误将代理识别为人类；② 发现代理检测的核心信号来自浏览器自动化 API 的事件缺失模式（无原始鼠标移动、滚轮 delta 等）而非代理推理过程；③ 在 17 个连续特征中通过穷举、前向/后向选择，仅需 2~5 个特征即可实现 100% 代理召回与 ≥0.99 的宏 F1；④ 设计了五级逃逸阶梯（无逃逸、规则化、GAN、GAN+、真人重放），并证明在 Playwright 生态下该检测框架对所有逃逸级别均保持 100% 检测。

**🔧 技术方法**

技术手段包括：基于 TabNet/SAINT 的表格变换器；梯度提升树 (GBM)、随机森林、XGBoost 等传统树模型；对 17 维连续特征进行 17k 组合的穷举搜索；前向/后向特征选择；五级逃逸阶梯构造（GAN 生成轨迹、真人轨迹重放等）；对比实验在 10 个随机种子、3 种模型族上的多次验证。

**📊 数据集**

使用数据集：① CaptchaSolve30k（14k+ 人类交互数据）；② 5k 合成脚本型机器人（头部、脚本化、精细化 3 种原型）；③ 1,025 真实 AI 代理（Claude+Playwright）采集的 10 任务；④ 524 次真人轨迹重放（逃逸等级5）。

**📈 对比分析**

方法对比：二分类 MLP、SAINT、RF、XGBoost 等在代理样本上的误识率分别为 39.1%、34.5%、30% 与 0%；三分类模型在 10 个种子、3 个模型族下均实现 100% 代理 F1；在 5 级逃逸阶梯中，所有级别均达到 100% 检测；宏 F1 在 5 特征时达到 0.991，2 特征即可保证 100% 代理召回与 1.0 代理精确度。

**⚠️ 局限性**

局限性：① 仅在 Playwright / CDP 生态下验证，其他框架（Selenium BiDi、OS 层输入模拟等）可能逃逸；② 机器人样本为合成分布，真实机器人多样性可能影响宏 F1；③ 人类数据来自 CAPTCHA，任务结构与代理任务不完全一致；④ 最小特征集在更广泛或更复杂场景下的泛化需进一步验证；⑤ 单特征解法（如 cursor_path_linearity）虽达到 100% 召回但精确度极低，易误报；⑥ 逃逸阶梯未覆盖基于硬件模拟或扩展插件的代理方式。

---

## 403. Temporally Centered SIGReg Improves Multi-Task LeWorldModel Learning: From Analysis to Method

**arXiv ID:** 2607.26924 | [PDF](https://arxiv.org/pdf/2607.26924v1)

**作者:** Chang Liu `[一作]` (University of Tokyo), Yaonan Zhu `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多任务世界模型学习中，提出将 Sketched Isotropic Gaussian Regularizer (SIGReg) 作用于时间中心化残差，而非整体潜变量，以缓解任务间表征重叠与易失性问题。

**💡 创新点**

创新点在于：① 发现传统边缘 Gaussian 正则化在多任务场景会压缩任务间潜变量聚类中心，导致表征混淆；② 通过对残差进行正则化，移除对聚类中心压缩的直接压力，同时保持防止潜变量崩塌的作用，显著提升多任务学习稳定性。

**🔧 技术方法**

技术手段包括：LeWorldModel (LeWM) 框架、Epps–Pulley (EP) 正则化、时间中心化残差 (z̄_t + r_t) 设计、行为克隆下游策略、以及对潜空间几何与鲁棒性的定量分析。

**📊 数据集**

使用 LIBERO benchmark 的四个子套件（Spatial、Object、Goal、Long），每套包含十个机器人操作任务，共四十个任务进行联合训练与评估。

**📈 对比分析**

与 Raw LeWM、Diffusion Policy、Octo、OpenVLA 等方法对比，TC‑LeWM 在 10‑任务联合训练下平均成功率从 53.2% 提升至 73.6%，在统一 40‑任务训练中维持 73.5%，显著优于 Raw LeWM（44.4%）并略胜 Scratch‑Diffusion Policy（72.4%），在多任务环境下表现出更高的稳定性和鲁棒性。

**⚠️ 局限性**

局限性包括：① 仅在 LIBERO 受控操作任务上验证，未检验在更开放或更高维度环境中的可迁移性；② 需要手动设定时间窗口大小，对不同任务的适配可能需额外调优；③ 对计算资源的需求相对更高，尤其在 40‑任务规模下仍需大量 GPU 训练。

---

## 404. Cross-organisational Process Mining from Message Logs

**arXiv ID:** 2607.26917 | [PDF](https://arxiv.org/pdf/2607.26917v1)

**作者:** Pieter Kwantes `[一作]`, Jetty Kleijn `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种从跨组织通信记录（message logs）生成全局事件日志（event logs）的框架，利用部分序列（partial orders）和对话（conversation）将异步消息交换映射到分布式通信字母（distributed communicating alphabet）并保证因果关系不丢失。

**💡 创新点**

创新点在于：① 将消息日志直接转化为全局事件日志；② 通过“动作函数”（action function）对消息事件进行统一标记；③ 引入对话概念及其对应的线性扩展，确保对所有有限对话的事件序列都能被完整描述；④ 证明了消息保持分配函数（message preserving assignment functions）与对话的部分序列同构，从而保证了因果性在事件日志中的保留。

**🔧 技术方法**

主要技术包括：Petri网（Enterprise nets 与 Industry nets）建模、消息事件的时序与因果关系定义、部分序列与线性扩展理论、动作函数与补全函数、分配函数（assignment functions）以及 FIFO/ LIFO 通道策略的理论化。

**📊 数据集**

本文未使用公开数据集，主要以理论构造和一个简化的OTC结算过程示例进行说明。

**📈 对比分析**

由于缺乏实验数据，未进行方法比较或性能评估；所给示例仅展示理论框架的可行性和逻辑正确性。

**⚠️ 局限性**

局限性包括：① 仅适用于已知且“健全”的消息日志；② 需要手工或算法定义动作函数；③ 对无限对话的处理仍依赖于极限与递归构造，实际实现复杂；④ 仅考虑输出与输入动作，内部动作缺失，可能限制模型的完整性。

---

## 405. Upper Bounds for In-Place Sorting with Minimal Moves

**arXiv ID:** 2607.27040 | [PDF](https://arxiv.org/pdf/2607.27040v1)

**作者:** Alex Zihan Xu `[一作]` (Independent Researcher), Stephen Jing Chick `[通讯]` (Independent Researcher)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种新的原地比较排序算法，能以接近信息论下界的比较次数和线性移动次数对任意长度的数组进行排序。

**💡 创新点**

核心创新在于构造了一个高效的有序集合结构、两层缓存设计以及最优的小规模排序，从而在保持线性移动的同时将比较次数降低到n·log₂n+O(n)。

**🔧 技术方法**

采用了基于有序集合的搜索、分段缓存（随机/确定性两种实现）、线性时间选择和基于段的合并排序，以及在原地转换时的位向量编码和分区技术。

**📊 数据集**

论文为理论分析，未使用具体实验数据集，结果以渐进复杂度形式给出。

**📈 对比分析**

与此前最优方法相比，随机化算法在指数高概率下实现n·log₂n+O(n)比较和O(n)移动；确定性算法在最坏情况下实现n·log₂n+O(n^{(t)}n)比较和O(tn)移动，其中t可调。

**⚠️ 局限性**

在最坏情况下仍存在n^{(t)}n项的比较开销，确定性缓存线导致无法达到完全的最优性；稳定性排序尚未实现，且稳定版仍停留在O(n·log₂n)比较级别。

---

## 406. Mitigating Compounding Error via Video Representation Regularization

**arXiv ID:** 2607.27036 | [PDF](https://arxiv.org/pdf/2607.27036v1)

**作者:** Taiye Chen `[一作]` (Peking University), Yisen Wang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究视频自回归扩散模型在长期生成过程中的误差累积问题，揭示其与内部表示维度崩塌（有效秩下降）的紧密关联，并提出视频表示正则化（VRR）方法，通过在DiT隐藏层加入表示正则化项来提升模型对误差累积的鲁棒性。

**💡 创新点**

创新点包括：①首次将有效秩作为量化误差累积的指标，证明其与视频质量衰退同步；②发现仅增大训练数据量并不能缓解误差累积，甚至可能加剧表示崩塌；③提出轻量级的VRR正则化（SigReg/Uniformity），显著提升长时序生成稳定性。

**🔧 技术方法**

采用的技术包括：视频扩散模型（DiT+VAE潜空间）与Diffusion Forcing训练；计算隐藏层有效秩；正则化技术SigReg、Uniformity、Barlow Twins、VICReg等；评估指标为VBench的Aesthetic Quality和Imaging Quality。

**📊 数据集**

实验使用MineRL Minecraft 18,000条序列（每条1,200帧）进行训练和长序列推理，评估使用VBench基准。

**📈 对比分析**

与Diffusion Forcing、Frame Anchor、SVI等基线在相同训练步骤（4k–16k）下进行对比；VRR在Aesthetic Quality上从55.56提升至72.08，Imaging Quality从60.92提升至69.51，整体性能明显优于所有基线。

**⚠️ 局限性**

限制：未完全解释为何增大数据量会导致表示退化；正则化参数需经验调优；实验仅在Minecraft场景验证，需在更广泛的环境与多模态数据上进一步验证。

---

## 407. HoF-Bench: Rediscovering Real AI-Discovered CVEs Without Frontier Models

**arXiv ID:** 2607.27030 | [PDF](https://arxiv.org/pdf/2607.27030v1)

**作者:** Petr Simecek `[一作]` (AISLE), Stanislav Fort `[通讯]` (AISLE)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 HoF-Bench v1 benchmark，包含 95 条公开 CVE，针对 8 个被 AISLE 报告的开源项目，并在固定扫描器框架下评估 10 个 LLM 代码审计器的重发现能力。

**💡 创新点**

创新点在于：①提出严格重发现的评估协议；②在公开 CVE 集上构建了源级别的 benchmark；③系统性比较多种 LLM 检测器的性能，揭示语言和多次扫描对发现率的影响。

**🔧 技术方法**

使用的技术包括：大型语言模型（Open-weight Mixture-of-Experts 3–13B 活跃参数和 5 种专有 Flash 级模型）、多轮 triage、可选生成上下文、自动化 LLM 判别器。

**📊 数据集**

采用的数据集为 AISLE Hall of Fame 中 95 条公开 CVE，来源于 OpenSSL、curl、GnuTLS、Apache httpd、pearweb、FOGProject、OpenEMR、WeKan 等仓库的特定提交。

**📈 对比分析**

通过固定扫描器 scaffold、四次重复扫描、可选上下文、三轮 triage 等方式进行比较，最佳单模型在 4 次扫描下 65/95（68%）的严格回收率，组合模型可达 84/95；平均每个 CVE 仅产生 3–4 条去重后的候选报告，成本相对可控。

**⚠️ 局限性**

局限性包括：仅评估源级重发现，不涉及全仓库定位或误报率；受限于 8 个项目的代表性；判别器可能受模型同属一家供应商的影响；公开 CVE 可能与模型训练数据重叠。

---

## 408. Anticipatory Data Governance in the Age of AI: Emerging Signals in Data Access, Reuse, and Sovereignty

**arXiv ID:** 2607.27029 | [PDF](https://arxiv.org/pdf/2607.27029v1)

**作者:** Adam Zable `[一作]` (New York University), Stefaan Verhulst `[通讯]` (New York University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过两场参与式预测工作坊，对AI技术如何重塑数据获取、治理与再利用进行结构性前瞻分析。

**💡 创新点**

首次将AI治理与数据治理融合，提出七个系统性信号及其互相反馈的治理框架，为预见数据生态变化提供工具。

**🔧 技术方法**

采用定性信号扫描与主题合成法，结合工作坊讨论与案例验证，绘制信号间的反馈循环。

**📊 数据集**

主要依托19位来自政府、国际机构、行业和学术界的专家经验与案例，未使用公开数据集。

**📈 对比分析**

研究为诊断性分析而非性能评估，未进行量化对比，仅提供结构性洞察与治理建议。

**⚠️ 局限性**

局限在于样本量有限、主观偏差、时点性研究、缺乏可量化指标、对跨文化差异的覆盖不足。

---

## 409. What Can Latent World Models Know? Physical Parameter Identifiability in Multimodal Predictive Representations

**arXiv ID:** 2607.27017 | [PDF](https://arxiv.org/pdf/2607.27017v1)

**作者:** Kaizhen Tan `[一作]` (New York University), Heqing Du `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在可证实可恢复性门控协议下，研究并量化了交互式环境中预测驱动的潜在空间对质量、阻尼、刚度等物理参数的可辨识性。

**💡 创新点**

创新点包括：①提出可证实可恢复性门控方法，区分环境可观测性与目标学习；②构建四因素可辨识性地图（目标因果、惰性平衡、正则化剂量响应、前沿），揭示目标结构决定潜在空间内容；③发现“前沿”区，即某些慢速比例参数（如阻尼）在确定性预测目标下无法被获取；④在真实机器人上验证这些机制并给出可操作的设计规则。

**🔧 技术方法**

使用 LeJEPA/LeWM 视觉+触觉/本体感知的潜在世界模型；SIGReg 抗崩溃正则化；线性与非线性探针；物理可恢复性证书；多因素实验设计；交互式仿真环境与真实机器人实验。

**📊 数据集**

数据集：自定义交互式仿真环境（2D 场景，随机质量、阻尼、刚度），共 4,258 试验；真实机器人数据 RH20T（KUKA 与 Flexiv 两个机械臂），共 4,258 试验，10 Hz 摄像与力传感；无真实物理参数标注，仅使用观测数据。

**📈 对比分析**

对比方法：在不同输入/目标组合下做线性探针和功能性预测测试；在真实数据上评估力读取、位置读取、16步预测误差、接触预测 AUC；相较于静态基线、随机传递、持久性基线。结果显示：惰性平衡下 vision‑only 模型位置 R² 0.04，加入跨模态目标提升至 0.98；阻尼在所有预测目标下约 0.13；在多尺度实验中，缺失输入或目标的模型在 5 倍数据下保持不变，完整目标在 800 试验即可达到高水平。

**⚠️ 局限性**

限制：仅研究确定性点预测目标；未检验非条件均值最优目标（如贝叶斯推理、贝叶斯状态）；真实机器人实验仅验证观测指标，缺乏真实参数可辨识性验证；模型规模有限，可能不代表更大规模网络的行为。

---

## 410. Estimating Size of the Union of Sets in Streaming Model

**arXiv ID:** 2607.26997 | [PDF](https://arxiv.org/pdf/2607.26997v1)

**作者:** Kuldeep S. Meel `[一作]` (National University of Singapore), Sourav Chakraborty `[通讯]` (Indian Statistical Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

设计了一种基于采样的流式算法，用于估计Delphic集合的并集大小，并将其应用到离散Klee测量、测试覆盖率估计和DNF计数问题。

**💡 创新点**

创新点包括：① 引入Delphic集合概念统一三类问题；② 提出自适应采样算法，空间为O(R log |Ω|)、更新为O(R log R · log(M/δ) · log|Ω|)，其中 R=O(log(M/δ)/ε²)，首次实现Klee测量在流式模型下线性维度的更新时间；③ 在覆盖估计中给出近似最优空间且可通过NP oracle加速的时间空间权衡方案。

**🔧 技术方法**

采用的技术主要有：采样与自适应阈值调节、Chernoff 与 Coupon Collector 证明、Delphic 集合的成员/计数/抽样接口、哈希计数与NP oracle 的组合，以及对多维离散范围的直接抽样。

**📊 数据集**

论文以理论分析为主，未提供实验数据集或实测结果。

**📈 对比分析**

与现有流式Klee测量算法相比，本工作实现了维度线性更新时间；在覆盖估计和DNF计数中提供了接近最优空间与低更新时间的理论上优越方案；NP oracle 版本在空间上进一步压缩但更新时间为多项式级别。

**⚠️ 局限性**

局限性包括：仅适用于插入‑仅流模型；NP oracle 方案在实践中可能成本高；空间仍对集合数 M 有对数依赖，无法完全去除；覆盖估计空间可进一步优化但需接受更高更新成本。

---

## 411. A Compositional Theory of Causally Masked Transformers

**arXiv ID:** 2607.26988 | [PDF](https://arxiv.org/pdf/2607.26988v1)

**作者:** Franz Nowak `[一作]` (ETH Zürich), Reda Boumasmoud `[通讯]` (ETH Zürich)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种基于半群理论的框架，分析了在固定精度、无位置嵌入、因果遮蔽、左到右评估顺序下的Transformer语言模型的表达能力，给出了不同注意力机制对应的可识别语言族的上限与下限；

**💡 创新点**

创新点在于直接将Transformer的实现细节（有限精度算术、注意力更新方式、评估顺序）映射到有限状态机的转移半群，从而以代数方式给出精确的语言识别上限，并揭示数值语义对表达力的决定性影响；

**🔧 技术方法**

主要技术包括：1）对注意力头的状态转移建模为查询索引的累加器；2）构造相应的转移半群；3）使用Krohn–Rhodes分解与伪群理论得到半群族（定值、-简化、本地-简化、无周期）；4）将层级结构视为并行与wreath产品，得到Transformer整体的转移半群；

**📊 数据集**

本研究为理论分析，未使用实际数据集；

**📈 对比分析**

比较方法：与之前使用LTL、RASP、理想化算术的Transformer表达性研究进行对比，证明在浮点精度下sharp soft attention得到更宽松的无周期（aperiodic）上限；实验性性能未涉及；

**⚠️ 局限性**

局限性包括：只考虑无位置嵌入的Transformer；对数值语义高度依赖（浮点算术的非结合性、下溢）；仅讨论固定精度与固定参数，未考虑可变精度或大规模模型；未给出具体模型训练与评估。

---

## 412. Dense Soft Weighting for Radar Ego-Velocity Estimation

**arXiv ID:** 2607.26980 | [PDF](https://arxiv.org/pdf/2607.26980v1)

**作者:** Atar Babgei `[一作]` (Imperial College London), Julie A. McCann `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于密集软加权的毫米波雷达前端，能够直接从全光谱范围-多普勒矩阵估计行驶速度及其协方差，并与惯性后端无缝融合。

**💡 创新点**

创新点在于用连续置信度权重取代传统的CFAR二值阈值，保留弱回波信息；通过一次性迭代重加权最小二乘实现鲁棒速度估计，并得到闭式测量协方差，兼具可迁移性和低计算成本。

**🔧 技术方法**

技术包括：密集信噪比权重计算（功率峰值与中值比）、多束角度校正与三点抛物线插值、加权最小二乘（含Cauchy损失）求解、协方差解析与Lever-arm校正、以及共享的ESKF后端。

**📊 数据集**

使用三组数据集：ColoRadar（室内/室外手持雷达 + LiDAR/IMU）、Radarize（手持/机器人雷达 + Visual‑Inertial）以及自采集的三维手持雷达+运动捕捉数据。

**📈 对比分析**

与稀疏点云CFAR+RANSAC+LS、Capon/FFT检测、学习式密集回归等方法对比。Dense Soft Weighting 在共享ESKF后端下，平均平移APE从2.12 m降至1.17 m（ColoRadar）或0.27 m（自采集），对Radarize数据与学习网络相当或优于其性能；在嵌入式Jetson Orin NX上仍能实时运行。

**⚠️ 局限性**

局限包括：侧瓣泄漏导致的角度估计偏差、协方差模型假设残差白噪且忽略侧瓣相关、实验仅覆盖手持/小车速度级别，对高速或航空平台的泛化尚未验证。

---

## 413. Robust RPC Bundle Adjustment for Multi-Date Satellite Imagery with Season-Invariant Correspondences

**arXiv ID:** 2607.26973 | [PDF](https://arxiv.org/pdf/2607.26973v1)

**作者:** Roger Marí `[一作]` (Eurecat), Gabriele Facciolo `[通讯]` (Institut Universitaire De France)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套鲁棒的RPC Bundle Adjustment（BA）流程，结合学习式特征匹配和图像相似度筛选，实现多时相卫星图像的自适应几何校正。

**💡 创新点**

创新点包括：①使用SuperPoint+LightGlue获得季节不变的局部特征匹配；②基于DINOv2相似度做图像对筛选，并通过多棵生成树保持连通性，显著降低匹配量并提高鲁棒性；③双阶段BA（soft L1+L2）与对象空间旋转修正相结合。

**🔧 技术方法**

核心技术包括学习式局部特征检测（SuperPoint）、轻量化匹配器（LightGlue）、全局图像描述子（DINOv2）、基于最大生成树的图像对筛选、双阶段Bundle Adjustment、RPC修正为对象空间旋转。

**📊 数据集**

使用DFC2019 Track3的Omaha 39-42张WorldView‑3多时相图像集进行主要实验，另外在Jacksonville 24张同步图像集上做消融验证。

**📈 对比分析**

与开源基线ASP和SAT‑BAv1对比，实验显示在Omaha数据集上平均held‑out reprojection误差从1.94像素降至1.35像素，3D一致性误差显著下降；匹配时间从18分钟降至2分钟，整体性能显著提升。

**⚠️ 局限性**

限制主要体现在：①仍依赖相机姿态误差为主的场景；②DINOv2相似度在极端季节差异下可能误判；③缺乏全局外部参考（如GCP），评估仍需伪GT轨迹，尚未验证在更大尺度或不同传感器上的泛化能力。

---

## 414. Generation or Judgement? A Paradigm Perspective on LLM-Based Emotion-Cause Pair Extraction in Conversation

**arXiv ID:** 2607.26967 | [PDF](https://arxiv.org/pdf/2607.26967v1)

**作者:** Weijie Feng `[一作]` (Hefei University of Technology), Zhiyong Cheng `[通讯]` (Hefei University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM在情感-因果对抽取（ECPEC）任务中三种任务细粒度拆解进行系统比较与分析。

**💡 创新点**

发现任务拆解粒度对LLM性能影响显著，提出pair‑level judgement优于generation；并设计辅助检索器改善决策边界。

**🔧 技术方法**

基于LLM（如GPT‑3.5/4）的对话级生成、目标级选择、对偶级判断三种范式，并加入检索器重评模糊边界。

**📊 数据集**

使用ConvECPE等三大对话情感因果数据集进行实验。

**📈 对比分析**

在三种范式下进行18项对照实验，pair‑level judgement在所有基准上均优；检索器提升F1 0.5‑1.46点，局部候选约1.33×推理时间。

**⚠️ 局限性**

仍需制定可靠的目标级决策阈值；pair‑level判断的二值化难以统一边界；仅在单模态/对话上下文中验证。

---

## 415. Same Evidence, Different Target: Decoding How Diagnostic Evidence Bears on Causal Questions from Language-Model States

**arXiv ID:** 2607.26929 | [PDF](https://arxiv.org/pdf/2607.26929v1)

**作者:** Weiyi Kong `[一作]` (University of Toronto), Zhuoran Li `[通讯]` (University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一种基于成对提示的因果诊断验证框架，要求在保持诊断证据不变的前提下，仅当证据对应不同因果问题时，模型能给出不同标签；

**💡 创新点**

创新点在于引入“成对”评估机制，要求模型不仅判断证据是否支持/挑战条件，还要判断其是否针对所询问的因果问题，从而更严格检验模型对因果问题细粒度的理解；

**🔧 技术方法**

使用线性读出器在 Qwen2.5、Qwen3、Llama-3.1 的次高层 transformer 块的最终 token 隐藏状态上进行预测，并通过岭回归正则化进行训练；

**📊 数据集**

构建了 49 对（共 98 条终端）成对提示基准，覆盖 9 个诊断族群（共 144 条开发终端），每对保持诊断证据相同但因果目标不同；

**📈 对比分析**

与基于答案分数的最高选项、四选项 logits 线性分类器以及三种文本基线进行对比，Qwen2.5 的隐藏状态读出在平衡准确率 0.659、完整对回收率 0.429（21/49 对）上显著优于所有基线；

**⚠️ 局限性**

局限性包括仅评估 Favors/Challenges/Wrong Target 三类标签，未涉及 Unresolved；基准场景受控，缺乏对生成答案的直接评估；模型内部是否真正利用此信息生成答案仍未验证。

---

## 416. Prior Directions: Why GUI Grounding Gets Locked in the Past

**arXiv ID:** 2607.26913 | [PDF](https://arxiv.org/pdf/2607.26913v1)

**作者:** Weile Gong `[一作]` (Nanjing University of Posts and Telecommunications), Weibei Fan `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉‑语言模型在场景变化时因过时语言导致视觉判断错误的现象，提出视觉锁定并在控制下的GUI定位实验中分析其机制。

**💡 创新点**

发现锁定与先验诱导的内部变化组织方式相关，而非变化幅度，提出“先验方向”低维几何路径并证明其在不同模型中重复出现及其因果效应。

**🔧 技术方法**

使用层级表示的几何分析、奇异值分解构建先验方向，进行对抗性干预、子空间投影和交叉验证等方法。

**📊 数据集**

在自构造的 ScreenSpot‑Pro 基础上，生成 1,257 个控制实例（共 5,028 评估），包含四个视觉语言模型。

**📈 对比分析**

通过对比四个模型在不同先验强度下的锁定率、位移大小、先验方向包装效率等指标，展示锁定越强位移越小但先验方向聚集度越高；干预实验表明移除先验方向能恢复正确性。

**⚠️ 局限性**

仅在特定 GUI 定位任务、单一语言、四种模型、固定提示下验证，未探究更广交互场景、其他语言或架构，先验方向随层、提示变化的动态未深入。

---

## 417. IMFuse: Instance-Aware Multi-Layer Fusion for LLM-Enhanced Sequential Recommendation

**arXiv ID:** 2607.27002 | [PDF](https://arxiv.org/pdf/2607.27002v1)

**作者:** Yuheng Zheng `[一作]` (Zhejiang University), Jiawei Chen `[通讯]` (Zhejiang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并提出IMFuse模块，利用多层LLM表示并通过实例感知的融合策略提升顺序推荐模型的性能。

**💡 创新点**

结合全局维度层偏好学习和实例级专家调制，实现对不同项目在不同层级语义信息的自适应融合，解决了单层LLM表征的维度崩塌和项目层级异质性问题。

**🔧 技术方法**

采用多层Transformer层的LLM嵌入、维度化变换、全局层权重softmax、专家路由（MLP+多专家模板）以及逐层融合与ID嵌入的融合技术。

**📊 数据集**

在四个亚马逊产品数据集（Clothing、Beauty、Toy、Office）以及SASRec和HSTU两种序列推荐骨干上进行实验。

**📈 对比分析**

与基线语义增强方法（RLMRec、LLM-ESR、LLMInit、SpecTran）和多层融合方法（LAEF、CASE-MLP、VA-HS）以及不同LLM编码器（LLaMA‑3‑8B、Qwen3‑8B）对比，平均提升约6.7%（HR/NDCG），且参数与推理成本仅略增。

**⚠️ 局限性**

仅对项目做实例化调制，未考虑用户特性；需要存储所有层嵌入，且在更大规模LLM或多模态场景下成本可能上升。

---

## 418. Foundation Models for Face Presentation Attack Detection: A Unified Linear-Probing Benchmark

**arXiv ID:** 2607.26993 | [PDF](https://arxiv.org/pdf/2607.26993v1)

**作者:** Peter Lorenz `[一作]` (Idiap Research Institute), Sébastien Marcel `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了 24 种预训练视觉模型（包括 ViT、CLIP、DINO、InternViT 等）在面部表演攻击检测（PAD）任务中的冻结线性探测性能，并在四个主流 PAD 数据集上进行同域和跨域测试。

**💡 创新点**

创新点：①提出统一的冻结背骨线性探测基准，覆盖多种基础模型与多模态视觉语言模型；②深入分析模型规模、预训练目标对跨域鲁棒性的影响；③通过计算成本与性能的多维度权衡，为 PAD 研究提供可比性更强的实验框架。

**🔧 技术方法**

使用技术：冻结预训练编码器 + 单层全连接线性分类器；在 MCIO 四大 PAD 数据集上进行单源跨域评估；采用 ACER 指标、交叉数据集平均误差和 Gap 分析；与两种专门 PAD 基线（DeepPixBiS、FSFM-FAS）对比。

**📊 数据集**

使用数据集：MSU-MFSD、CASIA-FASD、Replay-Attack、OULU-NPU，共四个实验集。

**📈 对比分析**

比较方法与性能：在同源内测试中，InternViT-6B 以 1.6% 的平均 ACER 夺得最佳；在跨域测试中，CLIP ViT-B/32 在保持低计算成本（≈8–9 GFLOPs）的同时，平均 off-diagonal ACER 仅为 31.6%，优于大多数大模型；整体显示冻结模型在同域可与专业模型相当，但跨域鲁棒性仍落后。

**⚠️ 局限性**

局限性：①单源训练导致跨域误差显著；②仅采用线性探测，未进行任务特定微调；③未覆盖多源或更高级的适配技术；④对预训练目标与 PAD 适配机制的因果关系仍需深入探究。

---

## 419. RL$^2$-VLA: Adaptive RL Latent Compositional Steering with Test-Time Scaling for Vision-Language-Action Models

**arXiv ID:** 2607.26991 | [PDF](https://arxiv.org/pdf/2607.26991v1)

**作者:** Derek Ming Siang Tan `[一作]` (National University of Singapore), Guillaume Sartoretti `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为 RL^2 的自适应推理时 steering 框架，通过在预训练的 Vision‑Language‑Action (VLA) 模型推理过程中使用 RL 导出的轻量级策略对 VLA 的动作流进行组合式 steering，从而在离域（OOD）任务和环境下提升机器人执行成功率。

**💡 创新点**

创新点包括：① 发现并利用不同状态（成功/失败）下的测试时缩放律，证明仅在失败状态下进行多样化 steering 最为有效；② 引入基于 VLA 潜在表征的 RL 流匹配策略，结合 SAFE 失败检测器实现仅在检测到失败时才激活 steering；③ 通过“组合式 steering”将 RL 产生的多样化动作与原 VLA 的行为先验相结合，兼顾样本多样性与可靠性。

**🔧 技术方法**

主要技术包括：强化学习（offline RL，QAM/CQL）在 VLA 潜在表征上训练流匹配策略；基于潜在表征的失效检测（SAFE + conformal prediction）；动作流组合式 steering（加权平均 velocity）；Verifier‑based 采样与评分（RoboMonkey、CoVer 等）。

**📊 数据集**

使用的大型离线数据集为 BridgeV2、DROID；在 SIMPLER、PolaRiS 等仿真 benchmark 上进行评测，并在真实机器人 PiperX 上验证；同时通过多种 OOD 语言提示和任务环境进行红队攻击测试。

**📈 对比分析**

与 Vanilla VLA、Repeated/Rephrase、Compose‑Always 等基线对比，RL^2 在 OOD 语言提示下平均提升 10.1%（+14.7% 某任务）成功率，在 OOD 环境下平均提升 8.5%（+14.6% 某任务）。在真实机器人实验中，RL^2 相比 Rephrase 提升 17.5%，相对非自适应 RL^2 提升 14.2%。

**⚠️ 局限性**

局限性包括：① 依赖强健的 Verifier 进行动作选择；② 失败检测器需要在线收集数据训练，且在真实环境中泛化有限；③ 仅实现了轻量级的 steering，未与大规模可微分 VLM 进行对比；④ CP α 选择仍需实验调优，可能不够鲁棒。

---

## 420. How Developers Experience Debugging Unfamiliar Codebases with Code Tours Generated and Evaluated by Local LLMs

**arXiv ID:** 2607.26987 | [PDF](https://arxiv.org/pdf/2607.26987v1)

**作者:** Balfroid Martin `[一作]`, Vanderose Benoît `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了完整的实验流水线，使用开放权重LLM自动生成并评估Java代码库的调试型代码游览（code tour），并通过26名开发者的思考-说话访谈收集体验数据。

**💡 创新点**

首次将完全AI生成的代码游览与开放权重模型结合，聚焦开发者体验、信任校准和调试需求，并引入LLM-as-Judge评估机制，发现了代码游览细节偏好、信任偏差与评估者自夸/虚构等关键问题。

**🔧 技术方法**

技术手段包括：三款开放权重LLM（Qwen2.5-coder、DeepSeek-Coder-V2、Devstral-Small）在本地Ollama推理；利用专门的prompt实现代码游览生成与评估；BIBD设计确保每个评价者对6条游览/评估进行三人共同评估；质性编码采用Krippendorff α≥0.8验证一致性。

**📊 数据集**

数据集为2025年GitHub公开仓库通过Gitbug‑Actions提取的110个可重现Java错误，生成243条堆栈跟踪并挑选26条做实验，覆盖多种错误类型与项目。

**📈 对比分析**

通过对比三种LLM在生成与评估任务中的表现，使用七点Likert量表对Transparency、Scrutability、Efficiency三项指标进行打分；定量评分与质性体验标签相结合，揭示了模型偏好、评估者不一致性（sycophancy、confabulation、incoherence）及其对开发者体验的影响。

**⚠️ 局限性**

局限性包括：样本仅为Java、2025年后提交的错误；参与者多为初级开发者，缺乏高级经验者；实验在Web界面进行，未模拟IDE真实使用场景；LLM知识截止于2025年，可能未反映最新技术；代码生成中存在lambda导致的重复步骤问题；质性访谈基于笔记与翻译，可能丢失细节。

---

## 421. Progressive Multimodal Alignment for Continual Instruction Tuning

**arXiv ID:** 2607.26947 | [PDF](https://arxiv.org/pdf/2607.26947v1)

**作者:** Duzhen Zhang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Tielin Zhang `[通讯]` (Center for Excellence in Brain Science and Intelligence Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了多模态持续指令调优（MCIT）中投影器（projector）对齐漂移导致的忘记问题，并提出一种可扩展的渐进式多模态对齐框架（PMA）来持续维护跨模态对齐。

**💡 创新点**

创新点在于：①利用轻量化表示描述器（Representation Descriptor）自动检测多模态分布偏移；②仅在必要时动态添加投影专家，保持子线性参数增长；③通过可扩展路由器在无任务ID的情况下实现专家混合，冻结预训练投影器作为对齐锚点，从而有效缓解投影器遗忘。

**🔧 技术方法**

技术方法包括：轻量化自编码器作为表示描述器、可扩展软路由器、冻结预训练投影器、与LoRA等PEFT技术的无缝融合，以及基于z-score阈值的分布检测与专家扩展决策。

**📊 数据集**

实验使用的主要数据集为UCIT（ImageNet‑R、ArxivQA、VizWiz、IconQA、CLEVR、Flickr30k）和MLLM‑DCL（RSVQA、PathVQA、DriveLM、AI2D、Sciverse、MapQA、TQA、FinVis），覆盖多种任务类型与视觉分布。

**📈 对比分析**

与HiDE、DISCO等SOTA MCIT方法比较，PMA在LLaVA‑1.5‑7B和InternVL‑Chat‑7B上均显著提升MAA、MFN、MFT，并将BWT（遗忘量）从负数约-6.7降至约-4.3，说明在保持对齐的同时兼顾新任务适应性。

**⚠️ 局限性**

局限性包括：对预训练投影器的依赖导致迁移到不同预训练基础时可能需要重新校准；阈值τ的选择对专家扩展与性能有一定敏感性；当前框架未针对任务标识可用的场景进行优化，且扩展后的路由器与描述器虽然参数较小，但仍增加了整体推理成本。

---

## 422. The Parameterized Complexity of Problems on Outer k-Planar Graphs

**arXiv ID:** 2607.26936 | [PDF](https://arxiv.org/pdf/2607.26936v1)

**作者:** Xiaobin Ren `[一作]`, Hans L. Bodlaender `[通讯]` (Utrecht University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文研究了在外部k-可绘图（outer k‑planar）图上多类经典图论问题的参数化复杂性。作者证明只有极少数问题（如二元CSP、散点集）在以k为参数时保持XALP/ XNLP难度，而大多数问题（如列表着色、容量支配集、目标度数指向、目标集选择等）在给定外部k-可绘图的绘图后变为固定参数可解（FPT）。同时，论文给出了若干结构性结果：外部k-可绘图图的mim‑宽度≤k+2，cut‑width≤k的图是外部2k-可绘图的，反馈边集数≤k的图是外部6k-可绘图的，并证明了外部k-可绘图与多种传统参数（树宽、路径宽、bandwidth等）不等价。

**💡 创新点**

创新点包括：
- 通过外部k-可绘图的几何结构（所有顶点在圆上且每条边交叉≤k次）设计了新的“外部k-可绘图三角剖分”技术，既能获得紧的树宽上界，也能直接构造mim‑宽度的上界；
- 对多类问题的参数化复杂性进行系统分类，首次将XALP/XNLP难度与外部k-可绘图参数联系起来；
- 证明了除几个例外之外，所有已知在树宽/路径宽下为XALP/XNLP难的问题，在外部k-可绘图上可转为FPT，揭示了几何绘图对参数化复杂性的潜在优势；
- 将代表集（representative set）技术与外部k-可绘图的三角剖分相结合，得到对列表着色、容量支配集等问题的FPT算法；
- 通过构造性证明，展示外部k-可绘图与经典图参数之间的不可比较性，从而完善了图参数层次结构。

**🔧 技术方法**

主要技术包括：
- **外部k-可绘图三角剖分**：在外部循环上对多边形进行三角剖分，使得每条三角边最多被原图k条边交叉，构造出二叉树弱对偶用于动态规划；
- **代表集技术**：用于压缩列表着色等问题中边界顶点的颜色/状态组合，保证只保留与外部子图兼容的“代表”集合；
- **动态规划与树分解的结合**：利用弱对偶树的递归结构，将子多边形的解合并，保持对界顶点信息的有限状态；
- **几何属性分析**：利用三角剖分保证每条交叉边界顶点的外部邻居数≤k，从而把通常的n^O(k)表大小压缩到f(k)·n^O(1)；
- **归约构造**：将树分解转为外部k-可绘图绘图，证明某些问题的XALP/XNLP难度不随外部k-可绘图而降低；
- **mim‑宽度证明**：基于三角剖分构造分枝树，分析每条分裂边上的匹配数，给出k+2上界。

**📊 数据集**

该研究为理论性工作，未使用实际数据集；所有证明均基于构造性算法与结构性引理，无实验验证。

**📈 对比分析**

与已有文献的比较：
- 先前已知在树宽/路径宽下为XALP/XNLP难的问题，如二元CSP、散点集，论文通过归约证明在外部k-可绘图上仍保持同样难度；
- 对列表着色、容量支配集等问题，先前在树宽上已知是XP级，论文在外部k-可绘图上通过三角剖分+代表集实现了FPT，首次突破了原有的XP上限；
- 对mim‑宽度、cut‑width等参数，论文给出了新的上界（k+2、2k、6k），并与已知上界（如树宽1.5k+2）对比，表明外部k-可绘图具有更优的结构性属性。总体而言，论文在理论复杂性方面取得了显著提升，未给出实验性能指标。

**⚠️ 局限性**

局限性：
- 仅适用于给定外部k-可绘图的绘图；若绘图不可预知，构造绘图的复杂度未被完全研究；
- 对于二元CSP和散点集等问题仍保持XALP/ XNLP难度，未能在这些核心问题上实现FPT；
- 对多边形三角剖分的构造虽然在FPT时间内完成，但实现细节可能较复杂，对实际实现具有一定门槛；
- 证明的结构性上界（如mim‑宽度k+2）依赖于外部k-可绘图的特殊几何性质，可能无法推广到更一般的超图类；
- 论文未给出实验验证，无法评估实际运行时间与常数因子的影响。

---

## 423. Veritas++: Value-aware On-Policy Distillation for Perception-Enhanced AIGI Detection

**arXiv ID:** 2607.27113 | [PDF](https://arxiv.org/pdf/2607.27113v1)

**作者:** Hao Tan `[一作]` (University of Chinese Academy of Sciences), Zhen Lei `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种三阶段训练的感知增强推理框架，用于检测AI生成图像（AIGI），通过高质量冷启动、感知导向学习（PoRL）和价值感知自我蒸馏（VaOPD）提升检测效果。

**💡 创新点**

创新点在于：①利用可验证的奖励显式强化细粒度视觉、语义异常和像素级差异的感知能力；②设计价值感知的自我蒸馏机制，按轨迹、token和方向动态加权，高价值提示更有效地将感知能力内化到推理中。

**🔧 技术方法**

采用多模态大型语言模型（如 Qwen3‑VL‑8B）与 LoRA 微调，结合 GSPO 强化学习、GPT‑5.1/Gemini‑2.5‑Pro 进行数据重写和质量过滤，并实现基于自我教师的价值感知 OPD。

**📊 数据集**

使用多种合成与真实图像数据集：SynthScars、AbHuman、SynthArtifact、X‑AIGD、HydraFake、AIGI‑Now、Chameleon、GenBuster++、EvalGen、HiRes、RealChain、CommunityAI、SocialRF、WildRF、GPT‑Image‑2、TFR；真实图像来自 Flickr、VisualGenome。

**📈 对比分析**

在标准、野生和新兴 OOD 基准上与视觉模型和其他 MLLM 检测器对比，平均精度提升约 +5.9%（对比最佳视觉检测）、+12.8%（对比 DeepVRM）并在新兴场景中从 70.3% 提升至 80.6%（仅少量新增数据），整体保持或提升多种指标。

**⚠️ 局限性**

局限性包括：①推理质量评估依赖 MLLM 判别器，缺乏系统化评估方案；②目前仅实现二分类，未扩展到异常定位或属性归因等细粒度功能；③推理过程计算量大，推理延迟高，实时部署受限。

---

## 424. FreqForcing: Autoregressive Long Video Generation via Spectral Self-Anchoring

**arXiv ID:** 2607.27110 | [PDF](https://arxiv.org/pdf/2607.27110v1)

**作者:** Jiatong Li `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了训练无关的 FreqForcing 框架，用 Spectral Self‑Anchoring (SSA) 解决自回归视频扩散模型在长视频生成中出现的频谱能量漂移和错误累积问题。

**💡 创新点**

创新点在于从频域视角刻画错误累积，利用低频 anchor 视图对齐来稳定生成，并通过频域融合（低频替换高频）实现无训练的长视频生成。

**🔧 技术方法**

核心技术包括：自回归视频扩散模型、滑动窗口因果注意力、Attention Sink、3D RoPE 对齐、频域快速傅里叶变换（FFT）进行低通滤波和注意力融合。

**📊 数据集**

使用 MovieGen（前 128 条提示）作为测试数据，评估 VBench‑Long 7 项指标。

**📈 对比分析**

与训练型方法（Self‑Forcing、LongLive、Rolling Forcing）以及训练无关方法（Infinity‑RoPE、Deep Forcing）对比，FreqForcing 在 60 s/120 s 长度下在动态度、整体一致性等指标上取得最优或接近最优表现，且实现 24× 的时长外推。

**⚠️ 局限性**

局限性：仍受限于预训练长度（仅在前两步 denoising 期间应用 SSA），对更长时间尺度的生成可能需要进一步优化 anchor 维持和参数调优；对极端动态场景或细节细化的适配性尚待验证。

---

## 425. It Doesn't Take a Thief: Optical-Scan Voting Systems Fail Even Without Adversaries

**arXiv ID:** 2607.27101 | [PDF](https://arxiv.org/pdf/2607.27101v1)

**作者:** Aleksander Essex `[一作]` (Western University), Philip B. Stark `[通讯]` (University of California)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并阐述了光学扫描投票系统在无攻击情况下的非对抗性失败模式分类，并分析了现有验证机制对这些失败的覆盖情况，提出改进建议。

**💡 创新点**

提出首个基于完整投票流程的管道式非对抗性失败分类法，系统性汇总真实案例并评估各验证方法的盲点，形成针对性改进建议。

**🔧 技术方法**

通过文献综述、官方报告、专家访谈以及对已公布失败事件的案例分析构建分类框架，并使用表格映射验证机制与失败类型。

**📊 数据集**

主要使用美国各州与地区公开的投票系统失误记录、Verified Voting、EAC 及各州选举管理机构报告中的事件案例。

**📈 对比分析**

通过将六种常见验证机制（L&A, BA/CoC, BP-RLA, BC-RLA, Batch, HR）与 F1–F3 失败子项映射，评估其检测能力；发现单一机制无法覆盖全部失败，凸显多层次审计必要性。

**⚠️ 局限性**

研究基于已公开案例，可能存在选取偏差，且未对对抗性威胁做深入分析，无法涵盖所有潜在硬件/软件故障的细节。

---

## 426. Sky sphere representation in language models

**arXiv ID:** 2607.27092 | [PDF](https://arxiv.org/pdf/2607.27092v1)

**作者:** Aleksandr Berdnikov `[一作]` (Fields Institute), Yevgeny Liokumovich `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（100B级别）在残差流中是否编码了可解码的夜空星图，并通过线性回归验证其可解释性。

**💡 创新点**

首次发现高维非平面、不可约的星图特征流形，证明模型内部存在球面结构而非仅平面或一维映射。

**🔧 技术方法**

使用主成分分析（PCA）提取前128维，再通过线性回归与真实星球坐标做对照，结合留一验证和RDM分析来区分球面与平面表征。

**📊 数据集**

采用188个天文对象（85颗最亮恒星、88个官方星座以及15个星系/星云/星团）的经纬度坐标，并设计25种以“最近星体”为中心的提示词。

**📈 对比分析**

在七个32B–235B规模的开源模型中，残差流的前8个主成分能解释65–85%的方差，角度误差低至12°–21°；对比更通用提示时信号显著衰减。

**⚠️ 局限性**

局限在于仅分析残差流，未揭示构造机制；且模型在直接获取坐标时表现更好，说明该表征可能是次级、被动的记忆或文本共现驱动。

---

## 427. Controlled Experiments on Lane Changing by Transitional Autonomous Vehicle: Dataset and Behavioral Insights

**arXiv ID:** 2607.27085 | [PDF](https://arxiv.org/pdf/2607.27085v1)

**作者:** Abhinav Sharma `[一作]` (North Carolina State University), George F. List `[通讯]` (North Carolina State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在北卡罗来纳州对转变性自动驾驶车辆（tAV）执行强制换道的完整过程进行受控实验，记录78个换道案例；

**💡 创新点**

首次公开一份专门针对tAV强制换道的受控实验数据集NC‑tALC，并系统性描述换道过程中的领-跟间距演化与碰撞风险发展；

**🔧 技术方法**

采用RTK‑GNSS/INS高精度轨迹采集、时间间隙与速度衍生的安全度量（SAE）等技术，对换道关键时刻进行标注与分析；

**📊 数据集**

使用自建的NC‑tALC数据集，包含四辆车的精确轨迹、位置、速度等信息；

**📈 对比分析**

通过对不同初始间距组的对比，观察领-跟间距在换道过程中的收敛特性（领间距≈0.54 s，跟间距≈0.65 s）以及风险峰值在左侧车道进入点出现，并在换道结束后仍有约1/3的案例保持风险；

**⚠️ 局限性**

仅针对单一tAV系统、有限的初始条件和单一路段进行实验，结果不一定能推广到其他车辆、不同环境或未来软件版本。

---

## 428. Anatomy Contextualized Adaption of CT Foundation Models

**arXiv ID:** 2607.27154 | [PDF](https://arxiv.org/pdf/2607.27154v1)

**作者:** Roshan Kenia `[一作]` (Harvard Medical School), William Lotter `[通讯]` (Dana-Farber Cancer Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了轻量化的Anatomy Contextualized Adaptation (ACA) 框架，将冻结的CT基础模型通过解剖学分割与Transformer模块实现解剖级视觉-语言对齐，并在扫描级别加入全局上下文监督。

**💡 创新点**

创新点在于：①仅使用冻结的基础模型权重，利用TotalSegmentator进行解剖分割；②引入跨解剖Transformer捕获全局交互；③在同一模型中同时采用解剖级与扫描级对比损失，兼顾细粒度与全局信息。

**🔧 技术方法**

主要技术包括：冻结的CT-CLIP/Merlin 3D backbone、TotalSegmentator解剖分割、Transformer上下文化模块、双重对比损失（解剖级与扫描级）以及轻量化投影头；实验中还使用LoRA微调验证轻量化优势。

**📊 数据集**

使用了Merlin（腹部CT）和CT‑RATE（胸部CT）两大公开数据集，分别训练Merlin与CT‑CLIP基础模型，并在同一数据集与跨数据集上进行零样本发现分类评估。

**📈 对比分析**

与全局VLM基线（Merlin/CT‑CLIP）、细粒度适配基线（MLP、fVLM、ViSD‑Boost）以及仅全球适配的Spatial Transformer相比，ACA在所有分布内/分布外设置下平均AUROC提升约4–8%，在个别发现上提升超过10个百分点，显示显著性能提升。

**⚠️ 局限性**

局限性包括：对解剖分割的依赖限制了可覆盖的结构；使用LLM自动抽取文本标签可能带来噪声；评估仅限两大CT数据集和基础模型，且仅针对零样本发现分类，未验证在分割、预测或报告生成等任务中的泛化。

---

## 429. MindForge: Teaching Small Language Models Whole-Life-Cycle Software Engineering via Source-Free Program Synthesis

**arXiv ID:** 2607.27146 | [PDF](https://arxiv.org/pdf/2607.27146v1)

**作者:** Yihao Chen `[一作]` (Queen's University), Ahmed E. Hassan `[通讯]` (Queen's University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套自动化流水线，将开源命令行程序转化为源代码不可见的可执行环境，利用强教师模型生成完整生命周期的程序合成轨迹，并在此轨迹上对小型语言模型 Qwen3.6-27B 进行监督微调，从而显著提升从零开始构建程序的能力。

**💡 创新点**

创新点包括：
1) 解决“从零开始构建程序”难题的源代码不可见环境构造方法；
2) 结合两种轨迹细化策略（基础设施噪声恢复与推理重写）获取高质量完整生命周期轨迹；
3) 将这些长轨迹蒸馏到仅 27B 参数的小模型，使其在 ProgramBench 上的通过率从 37.98% 提升到 49.51%，与更大模型持平，并在 7 个未见评测集上同样提升。

**🔧 技术方法**

使用技术主要包括：
- 自动化环境构造流水线（仓库筛选、构建脚本生成、行为等价检验、源代码泄露检测）
- 轨迹收集与细化（GLM‑5.2 作为教师、基础设施噪声恢复、推理重写）
- 监督微调（MS‑Swift + Megatron、序列打包、AdamW、梯度裁剪）
- 评估框架 Mini‑SWE‑Agent、无网络限制、跨任务基准测评。

**📊 数据集**

主要数据集：
- 562 个源代码不可见的命令行程序环境，覆盖 6 种编译语言（Go、Rust、C、C++、Swift、TypeScript）。
- 1,001 条完整生命周期轨迹（约 181.6 轮、177K token），其中 973 条用于微调。
- 评测基准包括 ProgramBench（200 个实例）及 7 个未见软件工程基准（RepoZero‑C2Rust、NL2Repo‑Bench、DeepSWE、SWE‑bench Verified/Pro/Multilingual、FeatBench）。

**📈 对比分析**

比较方法：将微调后的 Qwen3.6‑27B 与其基线模型、同参数的 MindForge‑27B 以及当前最先进模型（GPT‑5.5、Claude Opus 4.7、Sonnet 4.6 等）在 ProgramBench 上进行对比。结果显示，微调后模型通过率提升 11.53 点（相对提升 30.4%），在多项基准上也获得 4–31 点的绝对提升，整体性能已与 4–5 倍更大模型相当。

**⚠️ 局限性**

限制与挑战：
- 仍然无法突破 frontier 模型在 ProgramBench 上的 50% 通过率上限；
- 仅覆盖命令行工具，未涉及图形界面或交互式应用；
- 轨迹生成依赖强教师 GLM‑5.2，若教师质量不足会影响蒸馏效果；
- 训练成本较高（需执行 200M‑级 token 的长轨迹），不易在资源受限的环境中复现；
- 评估中未对生成代码的安全性、可维护性等软件质量指标进行深入分析。

---

## 430. DLAM: Distributional Latent Actions with Temporal Constraints

**arXiv ID:** 2607.27138 | [PDF](https://arxiv.org/pdf/2607.27138v1)

**作者:** Zuojin Tang `[一作]` (Zhejiang University), Zhiheng Ma `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出DLAM——一种将视觉转移建模为对角高斯分布的分布式潜在动作模型，并在无动作标注视频上进行预训练；

**💡 创新点**

创新点在于同时对均值与方差施加归一化组合与反转约束，利用共享相关系数实现维度间方差耦合，从而提升时序一致性与下游控制效果；

**🔧 技术方法**

核心技术包括对角高斯潜在表示、归一化组合与反转约束、共享相关系数的方差耦合、以及冻结编码器后在流匹配政策中仅使用均值作为辅助目标；

**📊 数据集**

使用了11个无动作标注的机器人视频数据集，主要来源于Open X-Embodiment与CALVIN；

**📈 对比分析**

与基线（重构仅模型、确定性ALAM等）相比，DLAM在转移到MetaWorld MT50、LIBERO以及真实世界任务时，平均成功率分别提升至87.6%、99.0%和73.8%，在重构质量（PSNR、SSIM）与时序关系残差方面也显著优于对照组；

**⚠️ 局限性**

局限性包括只对等间隔三元组施加局部约束，导致长程推理未得到充分验证；方差学习可能趋向近似常数解；下游仅使用均值，方差仅为辅助信号，未提供可信不确定性；共享相关系数可能忽略上下文或维度特异性依赖。

---

## 431. KAMR: Grounding Generation via Knowledge-Aligned Multi-hop Retrieval

**arXiv ID:** 2607.27136 | [PDF](https://arxiv.org/pdf/2607.27136v1)

**作者:** Xiaochen Wang `[一作]` (Pennsylvania State University), Fenglong Ma `[通讯]` (Pennsylvania State University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种知识图谱多跳检索器，利用 anchor‑triplet 与 connected‑triplet 的区分，在检索时先全局检索 anchor，再局部扩展以获取连通的 evidence。

**💡 创新点**

创新点在于：①提出 anchor 与 connected 两种检索目标并通过语义与结构两种匹配度分离；②通过 LLM 生成的 mask‑triplet 与查询对构造“partial alignment”训练集；③联合 pair‑level 与 element‑level 对比学习，兼顾语义对齐与结构连通；④两阶段检索策略与预训练目标高度契合。

**🔧 技术方法**

技术包括：大语言模型（ChatGPT‑3.5 Turbo、LLaMA2‑7B、Qwen3‑8B）生成查询；信息噪声对比学习（InfoNCE）做 anchor 与 element 级别对齐；图结构基的局部邻域扩展；全局与局部检索组合。

**📊 数据集**

使用的评测数据集有 PathQuestion（PQ‑2H、PQ‑3H，提供 gold 路径）、Complex Web Questions（CWQ，最多4跳、可多答案）以及 LC‑QUAD（补充实验）。

**📈 对比分析**

与 Lexical、Dense、Structure、Graph‑Pretrain 等 14 种基线相比，在 PathQuestion 的 Path Recall、Triplet Recall 以及在 CWQ 的 Acc/Prec/Rec/F1 上均取得领先；特别是在 3 跳检索中，Path Recall 近 99% 以上，显示对完整链的稳定恢复。相对同类结构检索方法（如 G‑RAG）在更深跳下表现更好。

**⚠️ 局限性**

限制：①对 LLM 生成训练集的依赖导致构建成本；②anchor‑triplet 的召回仍受语义匹配精度限制；③过多的连通扩展可能导致检索预算分配不均，需精细平衡 anchor 与 connected；④目前只针对知识图谱 Triplet 形式，未探讨更复杂结构或多模态知识。

---

## 432. TactiPlay: Multi-Granularity Tactical Parsing and Video-Anchored Match Review for Amateur Badminton Players

**arXiv ID:** 2607.27125 | [PDF](https://arxiv.org/pdf/2607.27125v1)

**作者:** Qiaoyi Chen `[一作]` (Hong Kong University of Science and Technology), Xiaojuan Ma `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究构建了一个名为TactiPlay的交互式后期复盘系统，整合专家制定的技术/战术问题分类，利用视频与事件检测结果在回合级别提供结构化、视频链接的反馈，帮助业余羽毛球选手进行多层次战术反思。

**💡 创新点**

创新点在于①将专业教练经验转化为可自动化的层次化分类体系；②采用回合为自然单位的多粒度分析（从单击球到三击回合再到整回合），并将反馈与对应视频片段、场地热力图紧密关联；③在系统设计与评估上实现了从结构化报告到视频证据的闭环工作流，显著提升了反思质量。

**🔧 技术方法**

技术方法包括：计算机视觉检测（人形、球道、击球识别）、视觉语言模型（如Doubao、GPT‑4.1）用于生成问题描述与总结、LLM聚类生成主题与子主题、交互式可视化（热力图、时间轴、回合说明）等。

**📊 数据集**

数据集主要由业余选手拍摄的单打比赛视频（8位参与者用于形成性研究，16位参与者用于评估）组成；另外使用三名国家级选手对七段业余比赛视频的人工注释（304条问题）来构建技术/战术分类体系。

**📈 对比分析**

在双条件、被试内的对照实验（N=16）中，TactiPlay相较于仅提供统计的基线系统，在反思条目数量、条目具体度、可操作性和恰当性上均获得显著提升；用户主观评价显示系统在帮助性、整合性、工作效率和战术意识支持方面也有显著优势。

**⚠️ 局限性**

局限性包括：①系统仍需人工验证事件检测结果，无法完全自主部署；②受试者仅为20–27岁、初中级单打选手，结果难以推广到不同年龄、技术层级或双打；③未测量反馈对后续比赛表现的实际影响；④未进行单组件或文本长度的消融实验，无法确定各功能对整体效果的单独贡献。

---

## 433. Towards Grounded GI Endoscopy VQA via Multi-Task Learning on Small VLMs

**arXiv ID:** 2607.27122 | [PDF](https://arxiv.org/pdf/2607.27122v1)

**作者:** Itbaan Safwan `[一作]` (Institute of Business and Administration), Muhammad Atif Tahir `[通讯]` (Institute of Business and Administration)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种多任务微调方案，用专家多边形掩码、Grad‑CAM弱监督掩码和术语自由视觉描述任务，提升小型视觉‑语言模型在胃肠内镜VQA中的定位与回答质量。

**💡 创新点**

创新点包括：①利用Grad‑CAM生成的弱监督掩码替代昂贵的像素级标注；②加入Gemma‑27B生成的无医学术语视觉描述任务，强化对视觉属性的学习；③在不增加显式定位输出的情况下，通过对齐分析证明多任务训练提升了内部跨模态对齐，增强模型可解释性。

**🔧 技术方法**

使用的技术有：QLoRA低秩微调、Grad‑CAM定位、Gemma‑27B视觉描述生成、视觉‑语言模型（Florence‑2、Qwen3.5、InternVL3.5）、跨模态对齐指标（点指游戏 PG 与集中度 CR）。

**📊 数据集**

采用的数据集包括：Kvasir‑VQA‑x1（问答对）、Kvasir‑SEG（专家多边形掩码）、CVC‑ClinicDB（外部验证掩码）、GastroNet‑5M（预训练分类器）。

**📈 对比分析**

与单一VQA微调对比，在三种小型模型上均实现了约2–3个百分点的准确率提升，尤其在复杂度高、外观和空间类问题上提升更显著；跨模态对齐指标 PG 与 CR 亦显著提升，表明模型对视觉信息的内在关注度增强。

**⚠️ 局限性**

局限性包括：Qwen3.5模型的提升不如Florence‑2/InternVL3.5；Grad‑CAM掩码质量未进行大规模人工验证；实验仅在单GPU、单周期训练，未覆盖更大规模数据和多GPU设置；缺乏对更多诊断类别的扩展与临床专家评估。

---

## 434. Visual Credit Audit for Multimodal Spatial Reasoning

**arXiv ID:** 2607.27069 | [PDF](https://arxiv.org/pdf/2607.27069v1)

**作者:** Feixiang Liu `[一作]` (Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Visual Credit Audit（VCA），一种无训练、决策级的可视化信用审计方法，用来区分模型在空间关系问题上是否真正受图像支持以及是否能正确识别所问关系。

**💡 创新点**

创新点在于将空间判断拆解为两条轴：① 图像相对支持（判断图像是否比无图像或空图像更能支持模型的回答）② 关系一致性（判断模型是否对同一图像下的关系与其相反关系产生不同的推理），并引入匹配图像置换校准、关系一致性阶乘实验以及编辑检查，形成四格分解（正确且有支持、正确但无支持、错误但有支持、错误且无支持）。

**🔧 技术方法**

核心技术包括：基于冻结的多模态语言模型的负对数似然边际（margin）计算、预测对齐支持指标、无标签的依赖缺口判定、匹配图像置换、固定像素关系对比（SRC）、关系一致性阶乘实验（同一视觉/文本关系互换）以及人工编辑的几何可行性检查。

**📊 数据集**

使用了两个空间关系基准：VSR（1,249 个查询）和 GSR‑COCO（880 条问答对），并在四个公开 7B‑14B 生成式多模态模型（Qwen、InternVL、LLaVA、Ministral）上评测。

**📈 对比分析**

方法通过对比标准准确率与 D‑CC（正确且有图像支持）以及 C‑U（正确但无图像支持）等指标展示，发现模型在 12.73–26.25% 的案例中正确却未得到图像支持；匹配图像置换使 D‑CC 上升 21.25–47.80 点；关系一致性阶乘实验表明 81.57–100% 的未得到支持的正确案例在关系反转时仍保持相同方向的响应，且 32.11% 的案例会翻转答案；编辑检查进一步验证自然图像变化时的对应性。总体来看，VCA 能揭示仅凭准确率隐藏的“视觉依赖”与“关系响应”两种不同的成功模式。

**⚠️ 局限性**

局限性包括：只能评估固定强制式“是/否”接口下的相对图像支持，无法给出像素级因果归因；关系一致性阶乘实验需要手工生成相反关系的图像，常规基准不易提供；编辑检查仅覆盖几何可行的子集；对开放式生成或非空间推理任务的适用性尚未验证。

---

## 435. Investigating reservoir computing for branch predictionin pipelined processors using emerging CMOS memristor devices

**arXiv ID:** 2607.27140 | [PDF](https://arxiv.org/pdf/2607.27140v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 436. Constructions of $k$-Min-Wise Hash from Bounded Independence

**arXiv ID:** 2607.27157 | [PDF](https://arxiv.org/pdf/2607.27157v1)

**作者:** Xue Chen `[一作]` (University of Science and Technology of China), Xin Li `[通讯]` (Johns Hopkins Unibversity)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了k-最小哈希（k-min-wise hashing）在有限独立性（bounded independence）下的性能，并给出了完全匹配的上界与下界；还证明了随机仿射哈希在pairwise独立性下的极端误差。

**💡 创新点**

首次证明k-最小哈希所需的独立度为Θ(k+log(1/δ))，既消除了先前结果中多余的loglog因子，又提供了匹配的下界；并揭示了仿射哈希的严重误差。

**🔧 技术方法**

利用Bonferroni不等式、有限独立性的切割与浓度不等式、以及对偶构造与二项分布分析，构造了上下界证明。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

比较方法：与之前的O(k loglog(1/δ)+log(1/δ))上界及O(k log N loglog N)种子长度对比；提出的构造在k=Ω(log N)时实现种子长度O(k log N)，误差为多项式小。

**⚠️ 局限性**

局限性：只给出理论上限，未给出高效可实现的具体hash函数实现；对仿射哈希的负面结果仅适用于二进制向量空间。

---

## 437. Skillful forecasting of offshore winds from satellite scatterometer constellations

**arXiv ID:** 2607.27152 | [PDF](https://arxiv.org/pdf/2607.27152v1)

**作者:** Francesco Pinto `[一作]` (Delft University of Technology), Angela Meyer `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出并实现了基于卫星散射计星座的短时海上风速与方向的全域即时预报框架WindCastNet；

**💡 创新点**

创新点在于首次直接从低地球轨道散射计的空间时间不规则观测中学习全域风场的深度学习模型，并通过持续的时延编码实现任意分钟级的连续预报；

**🔧 技术方法**

主要技术包括部分卷积长短时记忆网络（PConvLSTM）、时间向量嵌入（Time2Vec）、FiLM条件模块以及基于缺失数据增强的多任务损失；

**📊 数据集**

使用了从MetOp、HY‑2、Oceansat‑3等六颗散射计在北海区域收集的十余年（2021‑2026）10米风场观测数据；

**📈 对比分析**

在北海区域将WindCastNet与HARMONIE‑AROME MEPS和持久性模型进行对比，1小时和2小时预报误差分别下降23%和7%，且在三小时内保持优于MEPS；

**⚠️ 局限性**

局限性包括仅预测10米风速、受限于北海训练域导致对跨域气旋的预测不足、分辨率为25 km且未结合塔台、风机等地面观测，可在未来扩大域、提升分辨率或融合更多观测来进一步提升性能。

---

## 438. Explainable and Resource-Efficient Spatial Reasoning in Multimodal LLMs for Decision-Critical Applications

**arXiv ID:** 2607.27145 | [PDF](https://arxiv.org/pdf/2607.27145v1)

**作者:** Piyush Jain `[一作]` (Heritage Institute of Technology), Subarna Tripathi `[通讯]` (Intel Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了 ByDeWay‑V2，一个训练‑free、可解释的多模态 LLM 空间推理框架。

**💡 创新点**

在 Layered‑Depth‑Based Prompting（LDP）的基础上新增显式空间关系提取模块，利用 YOLO‑World‑L 检测并生成可审计的空间谓词。

**🔧 技术方法**

结合单目深度估计 DepthAnything V2、KOSMOS‑2 层级语义生成、YOLO‑World‑L 开放词汇检测以及几何关系计算，生成结构化提示。

**📊 数据集**

在 VSR、BLINK 空间子集和 POPE 视觉存在误报基准上进行评估。

**📈 对比分析**

与仅使用 LDP 的基线以及在 Qwen2.5‑VL、BLIP‑Base、ViLT 上对比，ByDeWay‑V2 在 BLINK 上提升 46% F1，在 VSR 上从 0.053 提升至 0.525，在 POPE 上提升约 5% 以上。

**⚠️ 局限性**

对检测器误检敏感；在 token 受限模型（如 ViLT）中召回下降；模型仍受视觉先行指令影响，难以完全平衡精度与召回。

---

## 439. The Computable but Not Learnable Information-Value-Free Equilibria and Regulation of Algorithmic Collusion

**arXiv ID:** 2607.27128 | [PDF](https://arxiv.org/pdf/2607.27128v1)

**作者:** Jason D. Hartline `[一作]` (Northwestern University), Chenhao Zhang `[通讯]` (Northwestern University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并定义了信息价值自由均衡（IVFE）概念，证明其存在并可在多项式时间内离线计算；随后给出了不可在线学习该均衡的严峻不可行性结果，说明在大类平滑学习算法（如乘法权重更新、扰动领导、梯度上升等）下，任何无交换后悔且无最佳预期后悔的学习算法都无法使经验历史收敛到IVFE。

**💡 创新点**

创新点包括：①首次将信息价值自由与相关均衡结合，构造出一套新的均衡概念；②揭示离线可解与在线可学习之间的根本分离，打破了传统均衡概念下的计算与学习难度相当的假设；③将不可行性与信息经济学中的“有价值信息生成”联系起来，给出算法竞争与反垄断监管的新理论依据；④为算法协同与反垄断政策提供了新的技术与理论视角。

**🔧 技术方法**

主要技术手段为：
- 退化至线性规划的相关均衡约束与信息价值自由约束；
- 引入平滑学习者定义并利用Blum–Mansour归约证明无交换后悔算法对平滑学习者的无效性；
- 构造对抗性游戏与状态序列，利用交换后悔与最佳预期后悔的性质推导经验分布距离下界；
- 对信息结构进行正式建模，证明无交换后悔且非负最佳预期后悔相当于“合理化学习且不产生有价值信息”。

**📊 数据集**

本文纯粹是理论分析，未使用任何实际数据集；所有结果均基于抽象的博弈与学习算法模型。

**📈 对比分析**

由于研究重点是证明不可行性，没有进行实验比较或性能评估。作者通过理论证明展示了在给定学习算法与对手平滑学习者的设定下，经验分布永远保持与IVFE（甚至Nash均衡）一定距离；这与传统算法在相关均衡或Nash均衡上可收敛的结果形成对比。

**⚠️ 局限性**

局限性：
- 结果仅适用于平滑学习者族；对非平滑或“猜测-验证”类算法的情况未作覆盖；
- 证明依赖于对手为对抗性或特定平滑学习者，实际市场情景可能更复杂；
- 只给出极端对抗性构造，未说明在平均或随机环境下学习的表现；
- 对监管实践的启示仍需进一步经验验证，无法直接替代实际政策决策；
- 未讨论是否存在更弱的可学习性质，或是否通过算法设计能绕过不可行性。

---

## 440. Equilibrium Training of Energy-Based Models with Parallel Trajectory Tempering

**arXiv ID:** 2607.27077 | [PDF](https://arxiv.org/pdf/2607.27077v1)

**作者:** Nicolas Béreux `[一作]` (Université Paris-Saclay), Beatriz Seoane `[通讯]` (Universidad Complutense de Madrid)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种基于并行轨迹调温（Parallel Trajectory Tempering, PTT）的能量模型（EBM）训练框架，专门用于在保持等温采样的前提下训练受限玻尔兹曼机（RBM）。

**💡 创新点**

创新点在于：①通过沿学习轨迹构建模型检查点阶梯，使得链在训练过程中始终保持热平衡；②采用储备采样与自适应优化显著降低计算开销，使其与传统的持续对比散度（PCD）相当；③在训练过程中即时获得分区函数、对数似然和热化诊断，实现无额外成本的精确评价。

**🔧 技术方法**

使用技术包括并行轨迹调温、储备采样、可自适应的梯度优化、交替吉布斯采样（AGS）、热化时间估计以及与传统 PT、CD/PCD 的对比实现。

**📊 数据集**

实验数据集涵盖：二维 Ising 模型、1000 Genomes 人类基因组数据、β‑Lactamase 蛋白序列、多模态神经记录（Allen Institute 视觉任务）以及医学推荐表格数据，共计五个多样化的科学与表格数据集。

**📈 对比分析**

与传统的 PCD、CD 以及最新的深度生成模型（如 Bayesian Flow Networks, edDCA）进行比较。PTT 在所有评估指标上均优于对照方法——在等温采样下更高的测试对数似然、更好的样本多样性、更少的过拟合风险，以及更准确的统计量再现，整体性能显著提升。

**⚠️ 局限性**

局限性：①目前实现集中在 RBM 上，对更深层或更大规模的能量模型的扩展仍需进一步验证；②梯度估计依赖于链的热平衡，尽管已降低链数，但在极高维、极稀疏数据下仍可能出现热化时间较长；③需要手动设定接受阈值 α 以及梯度下降调度，可能对不同任务产生不同的超参数敏感性。

---

## 441. Designing Pairwise-Stable Agent Seating Arrangements

**arXiv ID:** 2607.27102 | [PDF](https://arxiv.org/pdf/2607.27102v1)

**作者:** Frederik Glitzner `[一作]` `[通讯]` (University of Glasgow), Frederik Glitzner (University of Glasgow)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种可设计目标图的代理排列框架，定义新的“对称稳定性” (pairwise stability) 并通过将偏好系统压缩为 (r₁,r₂,r₃)-bundle 以实现高效的稳定排列。

**💡 创新点**

创新点在于：① 将目标图视为可设计对象，突破传统固定图的限制；② 引入对称稳定性概念，比传统的交换稳定性更易实现；③ 通过 bundle 架构把稳定匹配理论扩展到路径、环、团等多种图结构；④ 设计了多种多目标优化（最小桌数/座位/团队规模）并给出多项多项式时间算法；⑤ 与子图同构、离散路径划分、箱装问题建立新的复杂性联系。

**🔧 技术方法**

使用技术包括：稳定分区 (Tan 算法)、bundle 计算与压缩、动态规划求解 {1,2,3}-BinPacking、图形变换（加边保持稳定性）、子图同构与 NP 完全性归约、贪心/启发式分配以及 B‑匹配构造。

**📊 数据集**

本文为理论工作，没有使用实际数据集，所有实验和复杂度分析均为理论推导。

**📈 对比分析**

通过算法复杂度证明与已知 NP‑完全问题归约进行比较：大多数问题在可设计图情境下可在 O(n²)、O(s²n²) 等多项式时间内求解；而若目标图固定则对应问题为 NP‑完全。性能表现以时间复杂度呈现，证明所给算法在最坏情况下满足理论上最优的多项式上界。

**⚠️ 局限性**

局限性包括：① 对称稳定性虽然易实现，但并不保证全局最优的匹配或座位分布；② 在存在偏好平等（ties）时，bundle 最小化 P₃ 数量问题仍为 NP‑难，算法只能给出近似解；③ 方案假设偏好完全且有序，对不完整偏好或动态偏好适用性有限；④ 只考虑局部双代理激励，无法捕捉大规模协作偏好或多方团体的逃逸问题。

---

## 442. Step-Attention Refinement of DINOv3 Features for Efficient Anterior Eye Segmentation

**arXiv ID:** 2607.27087 | [PDF](https://arxiv.org/pdf/2607.27087v1)

**作者:** Philippe Baumstimler `[一作]` (Polytechnique Montréal), Lama Séoud `[通讯]` (Polytechnique Montréal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

在临床环境下，利用蒸馏版 DINOv3 ViT‑Small 作为编码器，提出一种轻量级网络，包含逐步注意力特征细化（SAFR）模块和卷积上采样解码器，用于前眼段（AES）图像的语义分割。

**💡 创新点**

创新点在于：1）将多层 Transformer 表征在低分辨率空间通过逐步注意力模块递归细化，使得语义丰富的深层特征与浅层位置信息充分融合；2）采用 RoPE 保持空间相对位置信息；3）设计极简的卷积上采样解码器，参数量大幅下降（仅 11.3M，冻结权重）同时实现与 DPT 相当的性能。

**🔧 技术方法**

使用技术包括：DINOv3 自监督预训练 ViT‑Small、Rotary Position Embedding、Multi‑Head Self‑Attention + MLP 细化块、卷积上采样解码器、交叉熵+Generalized Dice 损失、AdamW 优化器以及数据增强。

**📊 数据集**

数据集：自建 LightX（333 张高分辨率 AES 图像，8 种临床采集协议，7 类标注）作为训练/验证/测试；外部四个公开数据集（Eyes‑Defy‑Anemia、SLID、MOBIUS、SBVPI）用于评估跨域泛化。

**📈 对比分析**

与 UNet（EfficientNet‑B0/B5）、SegFormer（MiT‑B1/B3）、DPT、SegDINO 等基线比较。冻结权重时 mIoU 最高 79.09%，微调后达到 85.55%，在 LightX 上优于所有基线；跨域测试中保持最强的鲁棒性，尤其在四个公开数据集上取得最高的 mIoU，参数量仍低于 DPT。

**⚠️ 局限性**

主要限制：对域迁移仍存在显著误差（如 MOBIUS 数据集）；对小类别（如眶缘、睑结膜、噪声/伪影）性能波动大；在光照变化、病理结构、遮挡等极端条件下仍出现失配；未来需进一步提升自适应能力和小样本学习效果。

---

## 443. On-Policy Distillation for LLM Safety: A Routing Approach to Template-Robust Realignment

**arXiv ID:** 2607.27081 | [PDF](https://arxiv.org/pdf/2607.27081v1)

**作者:** Yongjian Guo `[一作]` (Tsinghua University), Sheng Wen `[通讯]` (Swinburne University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种双教师源路由的 top‑K KL 变分对齐框架，用以在不损失专业技能的前提下修复因微调导致的安全偏差。

**💡 创新点**

创新点是将对齐与攻击分离，通过在对齐模型和攻击模型之间对输出概率分布差异进行路由式蒸馏，显著降低模板匹配依赖，并保持任务性能。

**🔧 技术方法**

使用对抗式蒸馏、top‑K KL 损失、冻结教师模型以及混合任务+攻击数据的源路由训练。

**📊 数据集**

利用 SQL、SAMSum、NL2Bash 三个下游任务数据集，并使用 BeaverTails 的 1500 条恶意样本作为攻击样本。

**📈 对比分析**

与四种主流安全重对齐方法（SSRD、RESTA、soft‑SFT、rollback）在 Llama‑2、Qwen2.5、Gemma‑2 三大基模型上对比，实验表明该方法在所有攻击模板下均保持低 ASR（<10%）且任务分数保持或提升，且在数据效率和训练时长上更具优势。

**⚠️ 局限性**

仍存在模板不匹配时性能衰减、对系统提示重写的鲁棒性不足以及对多种攻击形式的通用性未完全验证等局限。

---

## 444. Field Codes for Distributed Coupling Samplers and Certified Empirical Transport

**arXiv ID:** 2607.27078 | [PDF](https://arxiv.org/pdf/2607.27078v1)

**作者:** Hung Mai `[一作]` (B0Labs, N2TP Technology), Tuan Do `[通讯]` (B0Labs, N2TP Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在分布式设置下，如何将经验最优传输（OT）的 coupling 输出进行通信，并提出一种 field‑code 编译器，将已通信的运输场与稀疏残差相结合，得到精确的边际分布以及可证明的 Wasserstein‑1 估计；同时给出了三种不同输出强度（分布式采样、成本可评估、标量值认证）及其通信复杂度上界与下界；

**💡 创新点**

创新点在于：①首次将 OT coupling 输出拆分为多种信息需求层级，并证明传输场是最合适的通信对象；②提出 field‑code 编译器，将运输场误差与残差通信分离，得到精确边际与标量值认证；③给出基于 Gap‑Hamming 等下界的通信难度分析，并证明认证输出与单纯采样输出可分离；

**🔧 技术方法**

使用技术包括：公共量化与公共随机化的通信模型、目标划分与公共格网、Wasserstein‑1 距离与成本评估、局部仿射与张量样条字段编码、Gap‑Hamming 与 Set‑Disjointness 下界、稀疏残差编码、以及对 C^1,1、C^2 约束的理论分析；

**📊 数据集**

实验数据集涵盖：合成的平滑与非平滑 2D/5D 传输映射、MNIST–USPS（共享 PCA‑5）以及 DOTmark ClassicImages；

**📈 对比分析**

方法比较：在相同比特预算下，对 5 种协议（网格计数、支持原型、全局仿射、局部仿射、张量样条）进行标量值认证误差评估。结果显示，张量样条字段在平滑任务上误差显著低于网格和原型方法，误差可降至 1e-3 级别；局部仿射在 2D 任务中与样条相近；而在自然图像任务中样条仍保持优势但误差略高。

**⚠️ 局限性**

局限性：1) 需要先获得低复杂度的运输场编码，若无法高效编码则上界失效；2) 需残差稀疏或目标划分具有足够大边界，才可保持总通信量低；3) 编译器不提供成本可评估输出，故 Alice 不能从自身视图计算实际采样成本；4) 在非平滑或图像等自然任务中，样条字段性能不如在合成平滑任务中显著提升。

---

## 445. SciFigAlign: Scoring Scientific Figures by Fine-tuned Alignment of Visuals with Manuscript Evidence

**arXiv ID:** 2607.27066 | [PDF](https://arxiv.org/pdf/2607.27066v1)

**作者:** Chuanzhi Xu `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个包含3,857张同行评审论文中科学图表的多维质量标注数据集，并提出了SciFigAlign模型，利用论文上下文（标题、摘要、引文段落）对图表质量进行精细化评分。

**💡 创新点**

创新点在于：①将图表质量评估转化为多模态回归任务，结合视觉、标题、引文、摘要等多源信息；②设计跨模态注意力与CubeMLP融合结构；③加入同一论文内部排序约束（ranking loss）以提升相对排序能力。

**🔧 技术方法**

技术方法包括CLIP ViT视觉编码器与SciBERT文本编码器的端到端微调、每模态交叉注意力、CubeMLP混合层、SmoothL1回归与同论文间margin hinge排序损失。

**📊 数据集**

使用的数据集是从ICLR、NeurIPS、ICML等机器学习会议论文中提取的3,857张图表，标注采用1–5分的四维量表（Clarity、Relevance、Informativeness、Structure），其中1,982张为人工评分，其余由GPT‑4o辅助完成。

**📈 对比分析**

与传统IQA、CLIPScore、BERTScore、SigLIP‑ridge以及零样本LLM/ VLM评判等基线相比，SciFigAlign在测试集上macro MAE为0.3524，within‑paper pairwise accuracy为81.64%，相比最佳LLM评判MAE降幅59%，pairwise accuracy提升约18个百分点。

**⚠️ 局限性**

局限性包括：①数据集仅覆盖CS会议论文，图表类型与风格有限；②模型对OCR密集或极端多面板布局的鲁棒性尚未充分验证；③依赖CLIP与SciBERT，可能在不同语言或非英语论文中表现不佳；④评价维度仍为粗粒度，无法捕捉更细微的视觉与论证细节。

---

## 446. CoCaRS: Correlation Calibration-Based Redundancy Suppression for Heterogeneous Knowledge Distillation

**arXiv ID:** 2607.27054 | [PDF](https://arxiv.org/pdf/2607.27054v1)

**作者:** Fengming Yu `[一作]` (Harbin Engineering University), Baoying Ma `[通讯]` (Harbin Engineering University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于相关校准的冗余抑制方法CoCaRS，用于异构知识蒸馏。

**💡 创新点**

创新点包括：①通过混淆证据估计(CEE)与强度分配控制(SAC)对特征相关性进行校准，既保留结构信息又抑制冗余；②自适应系数调节(ACR)根据损失比例动态调节校准目标的贡献，降低对超参数的敏感性。

**🔧 技术方法**

技术手段包括：Pearson相关矩阵、混淆权重计算、检索知识库、QR分解生成判别子空间、指数化权重调节、指数移动平均(EMA)调节系数。

**📊 数据集**

使用CIFAR‑100和ImageNet‑1K两大视觉数据集进行评估。

**📈 对比分析**

与多种同构与异构蒸馏基线（FitNet, KD, RSD, OFA, PAT等）对比，CoCaRS在所有异构教师-学生组合上均取得最高平均准确率（CIFAR‑100 84.70%，ImageNet‑1K 74.46%），比RSD提升2.56%和0.43%点；在同构设置下亦保持领先。

**⚠️ 局限性**

局限性包括：①需要预先构建教师响应检索库，增加预处理开销；②对教师模型的可解释性要求较高（需使用教师分类器权重生成子空间）；③在极端异构对比如极大宽度差异时，校准效果仍受限。

---

## 447. Linguistic Monoculture in LLM-Assisted Language Use

**arXiv ID:** 2607.27134 | [PDF](https://arxiv.org/pdf/2607.27134v1)

**作者:** Suhas Thejaswi `[一作]` (Aalto University), Lutz Oettershagen `[通讯]` (University of Liverpool)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个简化的数学框架，用来描述在大型语言模型（LLM）辅助下，作者与模型之间的交互如何驱动整体语言多样性的演化，并分析了三种不同的交互机制（共享固定模型、共享递归模型、个性化递归模型）。

**💡 创新点**

① 在同构模型下给出了明确的收敛率与极限多样性分析；② 将符合度（conformity）内生化为战略选择，揭示了个人理性可能导致的过度一致性以及“语言单一化价格”（price of monoculture）可能无限大；③ 通过模拟对三种机制进行定量对比，说明个性化反馈能显著提升群体语言多样性。

**🔧 技术方法**

使用了概率分布（多项式分布）表示作者与模型的语言特征，利用 Jensen‑Shannon 散度度量多样性；构建了指数型更新方程（α、β、γ、δ 等参数）描述作者适配与模型递归更新；在战略分析中采用平方欧氏距离的二次效用模型来简化求解；通过数学推导得到收敛、平衡与效率分析。

**📊 数据集**

数据主要来自**仿真**：设定 100 位作者、10 维抽象语言特征，随机初始化分布，运行 200 步，重复 100 次以获得平均行为；未使用真实文本语料或公开数据集。

**📈 对比分析**

与三种交互机制的模拟结果对比显示：共享递归模型（IM2）导致多样性下降最快；个性化递归模型（IM3）随个性化权重 ρ 上升，多样性显著提升；在所有机制中，长期多样性水平与理论极限相符。性能上，模型能在数百步内快速收敛，且仿真结果与理论预测高度一致。

**⚠️ 局限性**

主要限制包括：① 仅考虑固定的符合度水平，未建模动态调整；② 效用模型采用二次形式且假设作者签名正交，限制了现实中的相关性与非线性效应；③ 未考虑非 LLM 语言接触、社会网络和真实语料的影响；④ 结果基于模拟，缺乏对实际写作数据的实证验证。

---

## 448. Hierarchical Spatio-Temporal Transformer for Coherent Emergency Department Forecasting

**arXiv ID:** 2607.27106 | [PDF](https://arxiv.org/pdf/2607.27106v1)

**作者:** Filipa Lino `[一作]` (Instituto Superior Técnico), Manuel Marques `[通讯]` (Instituto Superior Técnico)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种Hierarchical Spatio-Temporal Transformer（HierSTT）框架，用于在医院、地区和国家三级层级上联合预测急诊科需求，并保持层级一致性。

**💡 创新点**

创新点在于：①采用自上而下的层级条件化机制，让上层预测直接影响下层；②引入层级一致性损失（coherence-aware loss）在训练过程中软约束预测聚合关系；③结合Temporal Fusion Transformer与时空Transformer，兼顾全局趋势与局部空间关联；④发布了覆盖81家医院、5个地区的全葡萄牙急诊科数据集。

**🔧 技术方法**

核心技术包括：Temporal Fusion Transformer（TFT）用于全国层预测；时空Transformer编码器-解码器模块处理地区和医院层；自注意力机制实现时间与空间交叉关注；层级一致性损失与Smooth L1损失组合的训练目标。

**📊 数据集**

使用了由葡萄牙卫生部公开数据、空气质量、温度与死亡率等异质特征构建的全国急诊科访问量数据集，涵盖2021-2024年1月1日至4月20日共81家医院、5个地区和全国层级。

**📈 对比分析**

与统计基线（Naïve、ARIMA、ETS）、非层级深度学习模型（LSTM、Seq2Seq、Transformer、TFT、N-BEATS）以及传统层级和谐方法（Bottom-Up、Top-Down、Middle-Out）进行对比；在所有层级上，HierSTT在MAE、RMSE、WAPE等误差指标上均优于对照组，WAPE平均下降超过30%，同时层级一致性误差（HAgE）也显著低于其他方法。

**⚠️ 局限性**

局限性包括：①模型复杂度高，训练与推理成本相对传统统计方法显著上升；②对大规模多级系统的可扩展性尚未在更大范围内验证；③对特殊事件（如疫情大流行、自然灾害）在训练期间缺乏实时更新机制；④层级一致性损失权重需手工调优，未自动化。

---

## 449. Entanglement-Assisted Quantum Locally Recoverable Codes: Characterizations, Bounds, and Constructions

**arXiv ID:** 2607.27091 | [PDF](https://arxiv.org/pdf/2607.27091v1)

**作者:** Yang Li `[一作]` (Nanyang Technological University), Shixin Zhu `[通讯]` (Hefei University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并研究了纠错纠缠辅助量子局部可恢复码（EAQLRC），给出了其定义、结构性质以及构造方法，并在此基础上推导了局部性与距离的上界，进一步构造了两类最优纯CSS‑样式EAQLRC。

**💡 创新点**

创新点包括：①在纠缠辅助框架下引入局部性概念，克服传统CSS/Hermitian构造对自正交或双含条件的限制；②给出了基于扩展稳定子支持的充分条件，使得任意EASC可实现局部恢复；③推导了新的局部性上界和Singleton‑类似下界，并给出纯码达到下界的必要与充分条件；④构造框架将最优经典局部可恢复码配对（如MDS交叉对和块奇偶校验矩阵）映射为最优EAQLRC，实现了灵活参数和非平凡局部性的理论上最优量子码。

**🔧 技术方法**

采用的主要技术手段包括：量子稳定子码与Pauli群的理论、纠缠辅助量子码的构造（EA‑CSS），以及经典局部可恢复码的线性代数特性（支撑集、奇偶校验矩阵、Hull维数）。此外，利用场的迹运算和符号映射对扩展稳定子的相互作用进行符号化，从而构造局部恢复通道。

**📊 数据集**

由于本文为理论研究，未使用具体实验或公开数据集；所有结果均基于抽象的线性码与有限域构造。

**📈 对比分析**

通过与已知的量子局部可恢复码以及经典局部可恢复码参数的比较，证明所构造的两类EAQLRC在Singleton‑类似下界上实现了最优；同时，通过对比局部性上界，展示其局部性可为非平凡（即严格小于最小距离对应的上界）。

**⚠️ 局限性**

局限性主要在于：①构造仅覆盖CSS‑样式的纠缠辅助码，未涉及Hermitian或更一般的稳定子码；②对纠缠资源的假设（预共享最大纠缠对）在实际实现中仍有挑战；③在某些参数范围内需要满足严格的交叉约束或群结构，构造过程较为复杂。

---

## 450. SciFigQual-Bench: A Benchmark for Scientific Figure Quality Assessment with Full-Manuscript Context

**arXiv ID:** 2607.27084 | [PDF](https://arxiv.org/pdf/2607.27084v1)

**作者:** Zihan Deng `[一作]` (University of Hong Kong), Lequan Yu `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了SciFigQual-Bench全稿语境的科研图像质量评估基准，并设计了分阶段的SFQ-Agent来实现可追溯的多维评分。

**💡 创新点**

创新点在于把图像与标题、引用段落绑定，提出5维（清晰度、布局、标题一致性、上下文一致性、误导风险）评分体系，并引入分阶段证据融合的可审计评估流程。

**🔧 技术方法**

采用OCR、计算机视觉特征提取、LLM文本理解与跨模态融合三阶段技术，最终以规则化的后处理实现最终评分。

**📊 数据集**

使用从2020-2025年ACL、EMNLP、ICML、NeurIPS四大顶会共计1,144篇论文中抽取的7,609张图像（6,308张已人工评分）构建的数据集。

**📈 对比分析**

在固定测试集eval1200上，与11种大模型进行三种评估协议对比，SFQ-Agent在MAE 0.418、Within‑1 93.4%等指标上均优于直接与Sidecar方案，表明分阶段评估能显著提升模型性能。

**⚠️ 局限性**

局限性主要体现在对标题与上下文一致性的判定仍受限于模型对语言的理解深度，且在缺失标题或引用文本时无法评估对应维度，导致部分实例的评分不完整。

---

## 451. Scores Are Not Decisions: Cost-Aware Stopping for Tool Acquisition in LLM Agents

**arXiv ID:** 2607.27083 | [PDF](https://arxiv.org/pdf/2607.27083v1)

**作者:** Yicheng Feng `[一作]` (Peking University), Wei Qi `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了在工具驱动的语言模型代理中，利用预执行工具排名进行成本感知的停止决策（CAM‑DF）来决定需要获取的工具数量。

**💡 创新点**

创新点在于：①将工具获取视作基于排名的停止问题，引入决策关注的惩罚权重；②证明该惩罚权重的分类目标与停/继续决策的贝叶斯最优目标对齐；③通过边际价值‑成本分析揭示了在异质成本下纯分数阈值决策的局限性；④设计了可解释的轻量级 CAM‑DF‑lite 变体。

**🔧 技术方法**

主要技术包括：基于离线任务奖励差距的加权逻辑回归停止器、使用排名分数、工具成本、前进进度等公开特征的特征映射、边际价值评估以及对齐损失的贝叶斯证明。

**📊 数据集**

使用了1,343个任务，跨五个工具使用领域：τ‑bench Retail、MCP‑Atlas、WorkBench、τ²‑bench Airline 与 τ²‑bench Telecom。

**📈 对比分析**

与多种基线（固定 top‑k、分数阈值、分数/成本阈值、手工规则、预测后阈值、仅聚合特征的 DF）进行对比；在多种成本压力与异质成本设置下，CAM‑DF 在可部署方法中实现最高平均 payoff，并在真实执行中将工具暴露量降低 37% 的同时保持相近的任务成功率。

**⚠️ 局限性**

局限性：仅针对单回合预执行阶段；在观察到工具输出后无法动态调整工具获取；依赖离线任务奖励标注，对新域或新工具集的迁移性需进一步验证。

---

## 452. MemSecBench: Tracking Agent Memory Poisoning from Persistence to Consequence and Repair

**arXiv ID:** 2607.27080 | [PDF](https://arxiv.org/pdf/2607.27080v1)

**作者:** Xuanze Chen `[一作]` (Zhejiang University of Technology), Qi Xuan `[通讯]` (Zhejiang University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个名为MemSecBench的任务驱动基准，用于跟踪代理记忆系统中恶意内容从写入到执行再到遗忘的完整生命周期。

**💡 创新点**

创新点在于：① 提供310个连贯的Write–Execute–Forget任务包；② 通过七个基于证据的检查点实现端到端的安全性评估；③ 在同一任务包上对24种不同的代理、LLM和记忆后端组合进行匹配对比，揭示生命周期安全的关键瓶颈。

**🔧 技术方法**

采用了OpenClaw和Hermes两种代理引擎，四种记忆后端（Native、Mem0、Mem0‑Graph、A‑MEM），三种LLM后端（DeepSeek‑V4‑Pro、MiniMax‑M3、GPT‑5.5），并结合判决模型DeepSeek‑V4‑Pro和程序化门控实现证据裁定。

**📊 数据集**

使用了310个由GPT‑5.5通过Skill构造的、覆盖48个现实场景（代码与科学、日常生活、办公工作）的连贯生命周期任务包数据集。

**📈 对比分析**

通过在相同任务包、相同运行时、相同初始化条件下对每种配置执行Write、Execute和Forget阶段，计算MPSR、MESR、E2E‑ASR和SRSR四项指标；总体结果显示写入允许率为84.2%，端到端成功率为50.3%，在成功写入的案例中可选择性修复成功率为56.1%。

**⚠️ 局限性**

局限性包括：① 仅使用合成或受控场景，未覆盖真实用户和生产系统；② 评估仅覆盖特定的代理、记忆和LLM组合，无法推广到所有实现；③ 选择性修复的判定依赖于判决模型，可能引入偏差；④ 对记忆后端的影响未形成统一的安全排名，仍需进一步研究。

---

## 453. Single-Beat Cuffless Blood Pressure Estimation Using Ear-PPG and ECG with a Lightweight Hybrid Learning Framework

**arXiv ID:** 2607.27076 | [PDF](https://arxiv.org/pdf/2607.27076v1)

**作者:** Kindeep K. Dhatt `[一作]` (Vanderbilt University), Yayun Du `[通讯]` (Vanderbilt University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一套轻量级多模态可穿戴系统，利用同步耳夹PPG、胸部ECG和6轴IMU实现单心跳级别的连续无袖压式血压估计；

**💡 创新点**

创新之处在于将一维CNN提取的单心跳PPG形态嵌入与基于PTT、HRV等物理特征融合，再使用LightGBM回归，既避免了对长时窗的依赖，又在运动动态条件下保持高鲁棒性；

**🔧 技术方法**

采用的技术包括一维CNN、LightGBM回归、同步ECG‑PPG‑IMU信号采集、PTT与HRV特征提取、Core ML等移动端部署工具以及混合学习框架；

**📊 数据集**

使用了公开PulseDB数据集（100名受试者）和自建10人多阶段实验数据（含休息、运动、恢复和冷压），实现跨数据集验证；

**📈 对比分析**

与仅手工特征、仅CNN回归的基线模型比较，混合模型在30次独立实验中获得SBP MAE≈4.0 mmHg、DBP MAE≈1.8 mmHg，较Baseline降低约28.2%，符合临床准确度标准；

**⚠️ 局限性**

局限性包括样本量有限、仅使用单通道PPG、采样率125 Hz限制PTT精度、未建模长期时序依赖、硬件尺寸大以及需要进一步验证更大多样化人群的泛化性。

---

## 454. A Type-and-Effect System for Temporal Dependency Analysis of Render-based Reactive Programs

**arXiv ID:** 2607.27074 | [PDF](https://arxiv.org/pdf/2607.27074v1)

**作者:** June Wunder `[一作]` (Boston University), Marco Gaboardi `[通讯]` (Boston University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了名为Willow的核心计算机语言和时序类型-效应系统，用于显式建模和静态分析React风格响应式程序的渲染时序和事件生命周期；

**💡 创新点**

创新点在于将渲染（render）作为基本时间单位，引入带有可量化时间单位的“next”模态，以及用于注册、注销、一次性监听等事件生命周期的模态；通过效应系统构造时间依赖图，可自动检测渲染级联、循环、脏监听等时序错误；

**🔧 技术方法**

使用：时序语义（time‑aware operational semantics）、类型-效应系统（带量化模态和事件模态）、图算法（拓扑、循环检测）、Haskell实现的类型推导器与效应推断；

**📊 数据集**

实验数据集：在示例程序中包括去抖动（debounce）、表单输入、API 调用、签名表单等典型响应式模式；并在真实的SignupForm案例中验证效果；

**📈 对比分析**

与现有工具的对比：论文未给出数值基准，但展示了在示例程序中，Willow 能够在编译时检测到 stale 读、循环依赖、未清理监听器以及首次渲染性能问题；性能评估显示原型推断器能在秒级完成大多数案例；

**⚠️ 局限性**

局限性：仅覆盖简化的核心计算机语言，未建模组件的挂载/卸载生命周期；不支持递归、并发或动态组件树；需要进一步集成到完整的React或类似框架中。

---

## 455. ScratchSim: A Procedural Synthetic Data Pipeline for Surface Scratch Detection

**arXiv ID:** 2607.27065 | [PDF](https://arxiv.org/pdf/2607.27065v1)

**作者:** Paul Julius Kühn `[一作]` (Fraunhofer IGD), Michael Weinmann `[通讯]` (Fraunhofer IGD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了一个基于 BlenderProc 的程序化渲染管线，用来生成带 COCO 格式注释的合成表面划痕数据，并在两种不同材质的对象上评估四种训练策略与三种轻量级检测器。

**💡 创新点**

创新点包括：①可配置的材质与相机随机化、自动划痕生成与可视角限制；②系统性比较四种训练策略（仅合成、仅实测、混合、从合成微调）在轻量化检测器上的性能；③公开合成与真实划痕数据集，推动后续研究。

**🔧 技术方法**

采用 BlenderProc 渲染、PBR 材质、程序化划痕（贝塞尔曲线+高斯模糊）、AOV 自动标注、YOLOX/S、YOLO26、LW‑DETR 轻量级目标检测器，并在 Docker+SLURM GPU 环境下训练。

**📊 数据集**

使用了两类对象的数据集：光亮玩具 Ferrari 车（合成 10k+真实 116）和哑光工业抓握器（合成 10k+真实 120），合成数据按 70/20/10 划分。

**📈 对比分析**

通过在真实测试集上计算 mAP50、mAP50‑95、AP/AR 对四种训练策略进行比较。结果表明：仅合成性能差；混合训练在缺少实测数据时显著恢复；从合成微调在所有检测器上均优于仅实测，YOLO26 mAP50 可达 0.69 以上。

**⚠️ 局限性**

局限性包括：低对比度和细微划痕仍难检测；合成与实测之间仍存在领域差距；未覆盖灰尘、油污等复杂环境；仅针对单类划痕，未验证其他缺陷类型或更复杂几何。

---

## 456. Setoka: A Benchmark for Hierarchical User Understanding in Personalized Agents over Heterogeneous Data

**arXiv ID:** 2607.27056 | [PDF](https://arxiv.org/pdf/2607.27056v1)

**作者:** Lingyang Zeng `[一作]` (East China Normal University), Xuan Zhou `[通讯]` (East China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了Setoka基准，评估个性化智能体在异构数据环境下的分层用户理解能力。

**💡 创新点**

创新点在于将认知与人格心理学的四层理解框架与心理计量学生成管线结合，自动生成连贯的异构用户数据和多层次查询。

**🔧 技术方法**

使用了多源记忆系统（图结构、层级、压缩等）、LLM记忆增强技术（MemGPT、Mem0、MemMachine等）以及LLM评估器和Kendall相关系数等方法。

**📊 数据集**

基于生成的10个合成用户的多模态记录（约364K token，23个schema）构建数据集。

**📈 对比分析**

对3大LLM（DeepSeek‑V4‑Flash、Ministral‑3‑14B、Gemma‑3‑4B）与5记忆系统进行12类查询（SM、EM、BP、PT）评估；结果显示准确率随抽象层级递减，最优系统在EM达到0.46，PT仅0.24。

**⚠️ 局限性**

当前记忆系统难以跨源聚合、长期抽象推理，评估多源一致性和隐含特征推断的能力不足，且回答率与准确率不匹配。

---

## 457. Can Large Language Models Represent Urban Publics? Behavioral Replication and Population Mismatch in an Affordable-Housing Experiment

**arXiv ID:** 2607.27100 | [PDF](https://arxiv.org/pdf/2607.27100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 458. Cost-Sensitive Conformal Prediction and Human-in-the-Loop Abstention for Imbalanced High-Stakes Decision Support: A Multi-Domain Benchmark

**arXiv ID:** 2607.27143 | [PDF](https://arxiv.org/pdf/2607.27143v1)

**作者:** Manpreet Singh `[一作]`, Shyamal Lakhanpal `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

未知

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

## 459. Voronoi Histograms for Adaptive Vectorization of Expected Persistence Diagrams

**arXiv ID:** 2607.27126 | [PDF](https://arxiv.org/pdf/2607.27126v1)

**作者:** Kaifeng Zhang `[一作]` (Nanjing University), Kai Ming Ting `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于Voronoi图的分块直方图（Vrep）来向量化期望持久性图（EPD），并通过归一化使其能够直接使用Wasserstein距离衡量；

**💡 创新点**

创新点在于不使用预定义的平滑点变换，而是采用数据自适应的Voronoi划分对EPD质量进行计数，从而在保持分块精细度的同时提升稳定性和计算效率；

**🔧 技术方法**

主要技术包括期望持久性图的定义、Voronoi图分块直方图构造、归一化与Wasserstein距离分析、代码簿采样策略以及基于随机森林的无监督/监督评估；

**📊 数据集**

实验使用了Protein、CAD、Time‑Delaying Embedding、Beef、Bird‑Chicken、DPTW、Earthquakes、ECG200等顶点敏感点云数据集；

**📈 对比分析**

与Persistence Image、Persistence Silhouettes、Persistence Landscape等无监督向量化方法以及PointNet进行比较，Vrep在大多数数据集上获得最高或接近最高的分类准确率，同时在计算量上相对较低；

**⚠️ 局限性**

局限性包括：在某些噪声或细粒度特征场景下，平滑核方法如PWGK更优；代码簿大小影响稳定性与精度，需要合理选择；当数据规模极大时，Vrep仍具有二次复杂度，虽然通过子采样可部分缓解。

---

## 460. MMAC: A Massive Multi-dimensional Benchmark for Audio Captioning

**arXiv ID:** 2607.27109 | [PDF](https://arxiv.org/pdf/2607.27109v1)

**作者:** Weijie Wu `[一作]` (Xiamen University), Qingyang Hong `[通讯]` (Xiamen University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MMAC多维细粒度音频字幕评估基准，并对多款AudioLLMs进行评估

**💡 创新点**

创新点在于将评估拆分为覆盖、精确度、准确率，并以多维目标信息检测来诊断模型

**🔧 技术方法**

采用LLM判断器（如Qwen3.6-27B）与人工评审结合的自动评分流程

**📊 数据集**

使用5,638条音频，来自20+公开数据集，覆盖6大类15个细粒度维度

**📈 对比分析**

通过开放式字幕生成，分别计算覆盖率、精度与准确率；Gemini 2.5 Pro准确率最高，Qwen3‑Omni‑Captioner覆盖率最高，整体表现差异显著

**⚠️ 局限性**

人工评估受LLM预标注影响可能存在锚定偏差，且基准尚未扩展多语言、多场景与时间戳评估

---

## 461. InferScale: GPU-Native KV Injection for Personalized LLM Serving

**arXiv ID:** 2607.27090 | [PDF](https://arxiv.org/pdf/2607.27090v1)

**作者:** Peter Li `[一作]` (Northeastern University), Prashant Pandey `[通讯]` (Northeastern University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个 GPU‑native 的 LLM 记忆系统，利用可重用的 KV 状态取代每次请求的 prompt prefilling，从而高效地为用户的长期上下文提供服务。

**💡 创新点**

创新点包括：① 将检索到的记忆预先计算并存储其 KV 片段在 GPU 上；② 通过 Chunked RoPE 实现 KV 的位置无关重用；③ 采用 Context‑Window Encoding 以在独立编码时保持跨事实注意力，从而在不需要模型微调或引擎修改的前提下实现精确的 KV 注入。

**🔧 技术方法**

技术手段包括 GPU‑resident 的 Jasper 向量索引、vLLM KV‑connector 插件、预 RoPE KV 存储、Chunked RoPE、Context‑Window Encoding、paged attention 缓存以及 GPU‑to‑GPU 复制等。

**📊 数据集**

使用的数据集：Llama‑3.1‑8B、Mistral‑7B、Qwen2.5‑7B 等开源模型；评测采用 10 会话、1,540 条可答长对话问答数据集（多跳、时间、单跳等）。

**📈 对比分析**

与 Mem0（CPU 向 GPU 检索 + prompt injection）对比：在 TTFT 上只增幅 4%（k=5→50），而 Mem0 增幅 106%；k=50 时 TTFT 减少 72‑79%；准确率接近 Mem0（60‑63% vs 63‑66%）；吞吐量提升 3.7‑4.5×；CPU‑offload 仅增加 1‑3 ms 的 TTFT，整体延迟仍显著低于 Mem0。

**⚠️ 局限性**

局限性：KV 存储占用显存 1.8‑4.8 GB/会话，限制可驻留用户数；不支持在线动态内存更新；对多 GPU/多节点扩展缺乏调度与缓存策略；在极长上下文或复杂多跳推理场景中准确率仍略低于完整 prompt injection。

---

## 462. Linear time approximation of the TV distance between product distributions

**arXiv ID:** 2607.27088 | [PDF](https://arxiv.org/pdf/2607.27088v1)

**作者:** Konrad Anand `[一作]` (University of Edinburgh), Heng Guo `[通讯]` (University of Edinburgh)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种线性时间的近似算法，用于计算两个产品分布之间的总变差距离。

**💡 创新点**

创新点在于通过使用ChatGPT 5.6 Sol Ultra发现了这一算法，并且该算法在运行时间上有显著的改进。

**🔧 技术方法**

使用了过滤的蒙特卡洛技术来设计新的估计器。

**📊 数据集**

算法处理的输入是两个产品分布P和Q，具体由每个坐标的边际概率给出。

**📈 对比分析**

与之前的算法相比，新的算法在时间复杂度上从O(q n^2/log q log n)改进为O(q n ε^-2 log(1/δ))，并且在特定条件下可以进一步优化到O(n q log q + n log q^2 log(1/δ))。

**⚠️ 局限性**

算法的局限性在于任何有效的近似算法至少需要读取输入的常数比例，并且在最坏情况下的查询复杂度为Ω(nq)。

---

## 463. Parameter-Free Dynamic Regret for Online Convex Optimization under Heavy-Tailed Noise

**arXiv ID:** 2607.27073 | [PDF](https://arxiv.org/pdf/2607.27073v1)

**作者:** Vaneet Aggarwal `[一作]` `[通讯]` (Purdue University), Vaneet Aggarwal (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了一种新的无参数在线凸优化算法 HT-PAder，可在重尾噪声环境下实现普适动态 regret；

**💡 创新点**

核心创新是将重启 AdaGrad 专家与路径无偏 AdaGrad-Hedge 元算法相结合，使得算法不需要任何噪声、梯度尺度或路径长度先验即可达到最优动态 regret；

**🔧 技术方法**

主要技术包括：路径无偏的 AdaGrad‑Hedge 元算法、块内重启 AdaGrad 的动态步长、重尾噪声下的单步不等式以及对比长度的块级动态 regret 分析；

**📊 数据集**

论文未使用任何具体数据集，全部为理论分析与下界证明；

**📈 对比分析**

与已知的改进 Ader、Sword 等方法相比，HT‑PAder 在重尾噪声下实现了无梯度裁剪、无先验参数的最优动态 regret，恢复了确定性情况下的最优范式，并给出了匹配的下界；

**⚠️ 局限性**

局限性在于仅适用于凸、Lipschitz 且有界域的环境；对非凸或无界域情况尚未覆盖；

---

## 464. Object Detection for Autonomous Driving in Chinese Rural Scenes: An Experimental Study on Real-Synthetic Data Mixing and Model Evaluation

**arXiv ID:** 2607.27058 | [PDF](https://arxiv.org/pdf/2607.27058v1)

**作者:** Danning Zhu `[一作]` (Wuhan University of Technology), Jing Wu `[通讯]` (Wuhan University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个包含14类中式农村道路对象的真实-合成混合数据集，并对13种主流目标检测模型在三种数据配置下进行了系统评估。

**💡 创新点**

首次针对中国农村场景设计专门的混合数据集，揭示了不同检测模型对真实/合成比例的敏感性，并确定了最佳1:0.5混合比例。

**🔧 技术方法**

使用Unreal Engine生成高保真合成图像、YOLO系列（v5、v8、v11、v26）与RT‑DETR-L网络、统一超参数训练、mAP@0.5与F1评估。

**📊 数据集**

利用河南魏氏县实测的4,720帧真实图像与Unreal Engine生成的2,100–4,200张合成图像，涵盖14类标签，测试集为517张真实图像。

**📈 对比分析**

在全真实、1:0.5与1:1三种配置下统一训练，使用mAP@0.5和F1衡量性能；1:0.5配置最佳，YOLO11m最高mAP为0.758；RT‑DETR-L在引入合成数据后性能显著下降。

**⚠️ 局限性**

合成域差异导致Transformer与小型模型性能下降；长尾非标准对象（摊位、护栏）仍难检测，需进一步域适配与模型改进。

---

## 465. Learning from the Future: Privileged Self-Distillation for Sequential Recommendation

**arXiv ID:** 2607.27055 | [PDF](https://arxiv.org/pdf/2607.27055v1)

**作者:** Jiakai Tang `[一作]` (Renmin University of China), Han Zhu `[通讯]` (Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Privileged Self‑Distillation (PSD)，在训练时利用未来交互作为仅训练阶段的特权信息，提升顺序推荐模型的性能。

**💡 创新点**

创新点在于：①只用单一Transformer权重，通过两种注意力掩码构造特权教师和因果学生；②引入优势-可达门过滤不可迁移的暗知识；③使用动量EMA平滑教师输出，解决自引用误差。

**🔧 技术方法**

技术包括：自监督知识蒸馏、双注意力掩码、KL蒸馏、优势-可达门、EMA参数更新。

**📊 数据集**

使用Amazon Video Games、CDs & Vinyl、Yelp三个公开数据集。

**📈 对比分析**

与SASRec、BERT4Rec、UniSRec等骨干以及RD、TCE、RCE、S⁴Rec、CSRec等方法对比，PSD在HR@10/H@20、NDCG@10/20均实现显著提升，平均提升约20%（相对骨干）且无退化案例。

**⚠️ 局限性**

局限性包括：依赖完整的未来交互日志；门阈和EMA参数需调优；在未来信息稀缺的数据上收益有限。

---

## 466. Breaking the $2^n$ barrier for graph $k$-coloring

**arXiv ID:** 2607.27159 | [PDF](https://arxiv.org/pdf/2607.27159v1)

**作者:** Kevin Pratt `[一作]` `[通讯]` (Columbia University), Kevin Pratt (Columbia University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

证明了对于任意整数 k，存在 ε_k>0，使得图 k‑着色问题可用一侧错误的随机算法在 O((2-ε_k)^n) 时间内解决。

**💡 创新点**

在所有 k 上实现了从 2^n 降速的突破；结合了修剪 Möbius 反演、子集卷积、列表着色与色板分析，构造了新的分支与采样框架。

**🔧 技术方法**

核心技术包括：修剪 Möbius 逆演算、子集卷积、列表着色算法、色板（palette）与 η‑good 集的概率分析、分支与随机采样的组合。

**📊 数据集**

本研究完全是理论性质，不涉及实验数据集。

**📈 对比分析**

与传统 O(2^n) 的算法相比，改进为 O((2-ε_k)^n)，在所有 k 上取得了严格的指数收敛；算法为随机单侧错误，成功概率常数可放大至 2/3。

**⚠️ 局限性**

局限性：算法仍为指数级，ε_k 随 k 的增长而下降；实现上需要大量预处理与高阶常数；随机化导致需要多次重试以保证成功概率。

---

## 467. OmegaUse-OfficeVal: Benchmarking LLM Agents on Long-Horizon Office-Suite Tasks with Economic Grounding

**arXiv ID:** 2607.27155 | [PDF](https://arxiv.org/pdf/2607.27155v1)

**作者:** Jingbo Zhou `[一作]` (Baidu Inc.), Hua Wu `[通讯]` (Baidu Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出OmegaUse-OfficeVal基准，评估LLM在长时序办公套件任务中的执行与可交付质量；

**💡 创新点**

引入任务级经济注释（人力时长与价格代理）并使用代码验证器，聚焦最终交付物的可用性和质量；

**🔧 技术方法**

采用多种大型语言模型（GLM-5.2、Qwen3.7-Plus、Kimi K2.6、DeepSeek-V4-Pro、MiniMax M3），结合代码生成与可执行验证技术进行任务执行与评分；

**📊 数据集**

基于100个真实办公需求的任务集，包含多模态输入文件，并为每个任务提供人力时长与价格代理两项经济注释；

**📈 对比分析**

通过平均得分、时长加权得分和价格加权得分三种指标进行比较，结果显示LLM平均得分约17.9，低于人类27.8，但在时间和成本上明显优于人类；

**⚠️ 局限性**

主要局限在于长时序任务难度高，LLM易出现完整失败；代码验证器难以覆盖所有主观需求，且经济标注仍不够细粒度。

---

## 468. Enumerating Small Cycles

**arXiv ID:** 2607.27147 | [PDF](https://arxiv.org/pdf/2607.27147v1)

**作者:** Or Stern `[一作]` (Tel Aviv University), Or Zamir `[通讯]` (Tel Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了在 n 顶点无向图中，针对固定长度偶数环 C₂k（k≤8）以及任意固定 k 的所有长度不超过 2k 的环，能够在 O(n² + t) 时间内列举 t 条环，并给出了预处理 O(n²) 与常数延迟 O(1) 的枚举算法；此外还给出了对偶数环的最优检测以及在特定图密度下更优的奇数环检测算法。

**💡 创新点**

核心创新点在于构造一个层级路径报告（path‑reporting）数据结构，将 2k 环视作两条内部不相交的长度 k 路径的笛卡尔积；通过递归压缩子路径、区分可替换与稀疏路径，并利用“轻稀疏”路径概念，将表格大小严格计费到实际的 2k 环上，从而实现 k≤8 的 O(n² + t) 列举与 O(1) 延迟枚举；该方法还能利用偶数环计数对奇数环做更快检测。

**🔧 技术方法**

主要技术包括：颜色编码将普通图转化为层级图；层级路径报告数据结构及其递归构造；可替换路径与稀疏路径分类；轻稀疏路径、辅助图与压缩技术；复杂度计费与“boundary‑layer”结构性质证明；以及对奇数环的异构转换。

**📊 数据集**

论文为理论工作，没有使用实验数据集；讨论的图规模以顶点数 n 为主，侧重稠密图分析（O(n²) 上限）并在一定密度区间给出相对稀疏图的比较。

**📈 对比分析**

相较于已知的 C₄（O(n² + t) 列举）与 C₆（O(n² + t) 列举 + O(1) 延迟枚举）算法，本文在 k≤8 范围内保持相同时间复杂度，同时实现了所有长度≤2k 的环的最优枚举；奇数环检测在具有典型偶数环计数的图中，运行时间 O(n² + (m/n)^{2k})，优于传统稀疏图检测的 O(m^{2-1/k})，在中等密度区间表现更佳。

**⚠️ 局限性**

局限性：1) 对偶数环的紧凑分析仅到 k≤8（C₁₆），对 C₁₈ 及以上需新思路；2) 奇数环的加速依赖偶数环计数，最坏情况并未突破 O(m^{2-1/k})；3) 在极稀疏图上，O(n² + t) 仍不如已知的稀疏图专用列举/检测算法；4) 递归构造及轻稀疏路径的实现复杂度较高，实际实现难度大。

---

## 469. SeasonStereo: Robust Dense Stereo Matching for Multi-Date Satellite Imagery via Generative AI

**arXiv ID:** 2607.27139 | [PDF](https://arxiv.org/pdf/2607.27139v1)

**作者:** Álvaro Díaz-Laureano `[一作]` (Eurecat), Gabriele Facciolo `[通讯]` (Université Paris-Saclay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 SeasonStereo 框架，用于多日期卫星图像的鲁棒稠密立体匹配。

**💡 创新点**

创新点在于通过生成式模型生成几何保持的季节变体，并结合零射线几何先验与光度/平滑自监督损失，消除了对 LiDAR 或真实异日配对的依赖。

**🔧 技术方法**

采用 MonSter++（RAFT‑style）网络、Nano Banana Pro 生成季节图像、DINOv3/SegFormer 进行相似度与掩码判定，利用零射线对齐、光度重建与平滑正则化构建多项损失。

**📊 数据集**

使用 DFC2019 Track3 与 Masquil 等 WorldView‑3 同步/异步数据集，生成 21,775 对合成季节样本，并在 Jacksonville、Buenos Aires、Omaha 等测试集上评估。

**📈 对比分析**

与 MonSter、MonSter++、Diachronic Stereo 等方法对比，采用 DSM MAE 评估；SeasonStereo 在异日测试集上与 LiDAR 监督模型相当，且在建筑边缘处更锐利，整体性能领先。

**⚠️ 局限性**

局限包括生成图像在水体/树木区域几何不一致需掩码，极端季节或光照变化仍可能导致误差；需要手工挑选同步配对以确保监督可靠。

---

## 470. Minimal Markovization via Stable Quotients in Holonomy-Cover Decision Processes

**arXiv ID:** 2607.27132 | [PDF](https://arxiv.org/pdf/2607.27132v1)

**作者:** Zuyuan Zhang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出Holonomy Memory Reinforcement Learning（HMRL），并在holonomy-cover决策过程（HCDP）中定义并证明最小Markov充分统计量为稳定商（stable quotient）。

**💡 创新点**

创新点在于首次将传输理论与POMDP抽象结合，得到最小的可递归可更新统计量，并揭示非阿贝尔传输导致计数型记忆失效的本质。

**🔧 技术方法**

使用传输结构、递归分割与精确抽象技术、诊断恢复算法以及有限MDP强化学习框架。

**📊 数据集**

实验数据集为自定义的两类HCDP环境：ChainCover（用于验证压缩精度）和LoopGuess（用于检验非阿贝尔记忆障碍）。

**📈 对比分析**

与仅观测、计数记忆、完整历史、原始状态和商状态等基线方法比较，HMRL在LoopGuess中获得1.0成功率、完美配对顺序准确率，仅需3个决策时记忆状态；在ChainCover中将状态从216压缩至25，且与商状态等价。

**⚠️ 局限性**

局限性：方法依赖可重置的诊断接口，难以直接应用于连续或噪声环境，以及完全被动数据的场景。

---

## 471. AgentMap: Joint Equivalence and Subsumption Discovery for Ontology Matching

**arXiv ID:** 2607.27130 | [PDF](https://arxiv.org/pdf/2607.27130v1)

**作者:** Yiping Song `[一作]` (University of Manchester), Wen Zhang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了混合本体匹配（HOM）任务，并基于多代理大型语言模型构建了 AgentMap 框架，能够同时发现等价与包含关系。

**💡 创新点**

创新点在于将等价与包含两种匹配关系统一成一个任务，并通过分阶段的多代理推理、层级搜索与冲突解决实现高效推断。

**🔧 技术方法**

采用了嵌入检索、LLM 多代理推理、层级结构引导搜索、词汇匹配及 LLM 冲突解决等技术。

**📊 数据集**

利用四个扩展为 HOM 的基准数据集：SNOMED–FMA–Body、SNOMED–NCIT–Pharm、NCIT–DOID–Disease 与 HeLiS–FoodOn。

**📈 对比分析**

在 HOM、等价仅匹配和包含仅匹配三种评估设置下，与 LLM、嵌入排序和传统 OM 基线进行对比，AgentMap 在总体准确率和包含准确率上均取得了最优或接近最优的表现。

**⚠️ 局限性**

仍面临包含匹配准确率低于 0.5 的挑战，尤其在细粒度包含判断上表现有限，并且任务仅覆盖两种关系，缺乏更丰富的语义关系扩展。

---

## 472. TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM

**arXiv ID:** 2607.27205 | [PDF](https://arxiv.org/pdf/2607.27205v1)

**作者:** Hengyi Xie `[一作]` (Huazhong University of Science and Technology), Han Ding `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 TurboVLA，一种在语言指令下直接将视觉与文本映射到连续控制动作的高效 VLA 模型。

**💡 创新点**

创新点在于：① 抛弃 LLM 作为执行桥梁，改为轻量文本编码和视觉编码；② 采用双向跨模态注意力实现任务语义与视觉信息的精准对齐；③ 通过一次性动作块解码器实现并行预测，显著降低推理延迟与显存占用。

**🔧 技术方法**

使用的技术包括：DINOv3 视觉编码器、BERT/T5 等轻量文本编码器、跨模态双向注意力交互模块、ACT‑style Transformer 解码器，以及行为克隆训练。

**📊 数据集**

主要数据集：LIBERO（四个子套件）、RoboTwin 2.0（双臂操控任务）以及在 AgileX Piper 机器人上收集的真实世界演示。

**📈 对比分析**

与多种现有 VLA（OpenVLA、π_0、π_0.5、VLA-JEPA 等）进行对比，TurboVLA 在 LIBERO 上达 97.7% 成功率，参数 0.2B，推理时延 31.2 ms，显存 0.9 GB；在 RoboTwin 2.0 上取得 60.2% 成功率、43.4 ms 延迟；在真实机器人实验中 80–92% 成功率，均优于同类模型且显著降低硬件需求。

**⚠️ 局限性**

局限性：主要针对执行级指令，缺乏对复杂任务规划和推理的能力，无法直接实现高层次任务规划与分解。

---

## 473. Do You Really Need to Pretrain Q-Functions for Online RL Fine-Tuning?

**arXiv ID:** 2607.27203 | [PDF](https://arxiv.org/pdf/2607.27203v1)

**作者:** Perry Dong `[一作]` (Stanford University), Chelsea Fin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在预训练策略之上进行在线强化学习微调时，Q 函数预训练的效果，并提出通过策略集成初始化 Q 函数的方法。

**💡 创新点**

发现传统的 Q 函数预训练由于目标 Q 与在线学习目标不匹配无效，并提出利用多多样策略生成多样化经验来更好地初始化 Q。

**🔧 技术方法**

采用 EXPO 离线策略学习框架进行在线微调，并通过多策略训练、rollout 收集以及 Q 函数预训练实现 IPE。

**📊 数据集**

在 Robomimic 和 OGBench 的六个机器人操控任务上进行实验，包括抓取、插件、灯光消除等连续控制任务。

**📈 对比分析**

与随机初始化和传统 Q 预训练比较，IPE 在所有任务中平均提升约 26%（1.26 倍）的成功率，显著优于基线。

**⚠️ 局限性**

仅适用于离线预训练+离线 RL 的离线学习场景，需额外多策略收集成本，且对基于策略的微调方法尚未验证。

---

## 474. HumanCLAW: Can Vision-Language Models Act Through a Body?

**arXiv ID:** 2607.27180 | [PDF](https://arxiv.org/pdf/2607.27180v1)

**作者:** Siyao Li `[一作]`, Chuan Guo `[通讯]` (Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `79276348-11e0-48e3-84bc-7ec231d0171c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文探讨了最近的视觉-语言模型（VLMs）是否具备行动智能，即在物理世界中根据当前状态和目标做出可执行的动作决策。

**💡 创新点**

创新点在于提出了HumanCLAW框架，通过将VLM与物理世界的自由行动结合，评估其在闭环决策中的表现，并引入了HumanCLAW-Bench基准测试。

**🔧 技术方法**

使用了视觉-语言模型（VLM）和半物理模拟器，结合了技能条件运动生成器和技能特定的空间响应验证器。

**📊 数据集**

使用了HumanCLAW-Bench基准数据集，该数据集包含1218个以自我中心的全身动作为基础的场景，涵盖41个室内房屋。

**📈 对比分析**

与其他方法的比较显示，当前的VLM在完成任务方面表现不佳，最强模型仅在16.8%的场景中成功完成任务，表明在视觉识别与实际行动之间存在显著差距。

**⚠️ 局限性**

限制在于当前模型缺乏对自身身体状态的感知能力，导致在导航和交互阶段频繁出现失败，且动作决策的细粒度技能词汇有限。

---

## 475. The Social Cost of an AI Teammate: How an Artificial Teammate Reshapes Human-Human Communication in Small-Team Decision-Making

**arXiv ID:** 2607.27179 | [PDF](https://arxiv.org/pdf/2607.27179v1)

**作者:** Nia Nixon `[一作]` (University of California, Irvine), Spencer JaQuay `[通讯]` (University of California, Irvine)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在团队决策任务中加入对话式 AI 同伴后，对人类成员之间的交流模式、心理归属感和状态感知产生的社会成本。

**💡 创新点**

首次将 Group Communication Analysis (GCA) 与问卷与词汇分析相结合，量化 AI 作为“关系置换者”在团队中的社交角色，并证明其社会成本是即时且与 AI 的发言占比呈负相关。

**🔧 技术方法**

主要技术包括：GCA（六维度的对话分析）、LIWC 词汇情感计数、语言风格匹配 (LSM)、模态标记的马尔可夫分析，以及对话时序的递归量化分析。

**📊 数据集**

数据集：共 33 个团队（16 个 AI+2 人、17 个全人），通过文本聊天完成的高风险道德决策任务；AI 采用 Google Gemini 2.5 Flash Lite 生成对话；每个团队都有完整的对话记录、时间戳、问卷数据。

**📈 对比分析**

与全人团队比较时，AI团队的人类成员在 GCA 的关系维度（responsivity、social impact、participation）表现显著更低（Cohen g≈-0.8）；问卷中归属感下降 0.7 分，状态感下降 0.3 分。AI的占比越高，人的自我价值感下降越明显，相关系数 r≈-0.5。

**⚠️ 局限性**

局限性：样本量有限（仅 33 个团队），单一 AI 角色与文本交互，单次会话与单一道德情境；团队规模与性别构成的混杂可能影响结果；未检验长期或多模态（语音/身体）的影响。

---

## 476. A Photonic-CXL Memory Appliance for Scalable KV Cache Management in LLM Inference

**arXiv ID:** 2607.27187 | [PDF](https://arxiv.org/pdf/2607.27187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e`

---

## 477. From Classification to Regression: Using a Fruitfly to Solve Equations

**arXiv ID:** 2607.27196 | [PDF](https://arxiv.org/pdf/2607.27196v1)

**作者:** Shady E. Ahmed `[一作]` (Pacific Northwest National Laboratory), Panos Stinis `[通讯]` (Pacific Northwest National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种将回归任务转化为分类并通过加权重构的通用框架，利用有限的代表性局部模式库来逼近非线性输入‑输出映射；该框架可应用于连续/离散时间动力学预测、数据驱动回归和物理信息回归；

**💡 创新点**

创新点在于：①将回归重新表述为分类+加权重构，避免全局参数化；②仅使用模式库与相似度计算即可完成在线推理，显著降低计算和存储成本；③通过软最大化温度可调节模式权重；④框架统一可处理动力学流图、右侧向量场以及一般函数；

**🔧 技术方法**

核心技术包括：softmax相似度与温度调节；代表模式库与对应响应（右侧向量、流图或函数值）；embedding（高斯核/傅里叶特征/神经网络特征）用于非动力学回归；轨迹监督与物理信息损失（Poisson方程）进行训练；最小二乘/梯度下降求解；持续学习通过增补新模式实现。

**📊 数据集**

使用的“数据集”为：模拟的Lotka‑Volterra与Lorenz动力学轨迹；三种数据驱动回归例子（抛物线、10正弦组合、带零外区抛物线）；三种物理信息Poisson方程例子（同样的三种右侧函数）。所有数据均为人工生成的数值模拟。

**📈 对比分析**

与传统全局方法（全局神经网络、高斯过程、RBF插值等）对比，模型在Lotka‑Volterra的相对L2误差降至约3%，Lorenz约1%，抛物线0.1%，正弦组合1%，Poisson方程1–10%。框架在长期预测和持续学习实验中保持低误差，并通过模式数与相似度可显著控制在线推理成本。

**⚠️ 局限性**

局限性包括：需要先行设定模式数、位置及影响函数方差，未覆盖区域导致误差上升；模式冗余与选择不够自适应；对高维或极端非光滑问题的推广仍待验证；物理信息损失需手工调参，易受数值误差影响。

---

## 478. VidMap: Exploiting Temporal Structure for Video-Based Structure-from-Motion

**arXiv ID:** 2607.27194 | [PDF](https://arxiv.org/pdf/2607.27194v1)

**作者:** Zador Pataki `[一作]` (ETH Zurich), Marc Pollefeys `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 VidMap，一种结合 SLAM 的时序约束和 SfM 的全局优化的离线视频重建框架，能够在未标定、长视频中高精度恢复相机轨迹与结构；

**💡 创新点**

创新点包括：①将视频的时间顺序与循环闭合显式区分，使用“证明来源”来对不同约束施加不同鲁棒损失；②在全局定位与束调整中引入单目深度先验和可优化的尺度因子，实现尺度一致的全局重建；③利用稠密匹配和关键帧自适应选择，构建跨帧长轨迹，减少漂移；

**🔧 技术方法**

核心技术包括稠密图像匹配与光流链式跟踪、基于检索的循环闭合、深度先验（MonoDepth）与尺度优化、旋转平均、全局定位（GP）与束调整、鲁棒损失（Huber/Cauchy）以及多线程并行匹配；

**📊 数据集**

在四大数据集上评估：LaMAR（长室内视频）、CroCoDL（灾害现场手机/机器人视频）、ETH3D-SLAM（室内外短视频）和 EuRoC（无人机摄像）；

**📈 对比分析**

与 SLAM（DPV‑SLAM、DROID‑W、MASt3R‑SLAM）、全局 SfM（GLOMAP、GLOMAP‑RoMA、MegaSaM、ViPE）以及端到端学习方法（DA3‑Long、LoGeR、LingBot‑Map）进行对比，VidMap 在长序列（如 LaMAR、CroCoDL）和无标定场景中均取得最高的窗口AUC/整体AUC，显著低于其他方法；在短序列的精度测试中也保持竞争力；

**⚠️ 局限性**

局限性包括：①对极端光照、模糊、极低纹理仍存在挑战；②在非常长或极高帧率视频中匹配耗时仍较高；③目前仅支持单目相机，未集成惯性信息，无法满足视觉惯导系统的严格精度要求。

---

## 479. Inverse Learning of Latent Risk-Neutral Densities from Irregular Option Quotes

**arXiv ID:** 2607.27188 | [PDF](https://arxiv.org/pdf/2607.27188v1)

**作者:** Lennon J. Shikhman `[一作]` (Georgia Institute of Technology), Nicholas A. Welsh `[通讯]` (Florida Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文研究了从稀疏期权报价中恢复风险中性密度（RND）的逆问题，并区分了价格拟合与潜在密度恢复的区别。

**💡 创新点**

创新点在于提出了将模拟器真密度与真实市场观测分离的双基准框架，并通过数值条件谱揭示价格与密度之间的辨识难度。

**🔧 技术方法**

采用了深度算子网络（DeepONet、FNO、Transformer、Set Decoder）进行密度或表面预测，同时比较了传统的两分量对数正态混合、正则化离散密度、SVI、BL 提取等基线方法。

**📊 数据集**

使用了自构造的三类正马尔可夫过程（几何布朗运动、对数正态混合、Merton跳跃扩散）的模拟数据，及印度NIFTY指数的一分钟交易价档进行真实市场评估。

**📈 对比分析**

在模拟基准上，混合对数正态模型在整体密度误差和价格误差上最优；DeepONet 在1%分位数和方差等关键函数上表现最好；在NIFTY实盘上，适度的测试时适配显著提升DeepONet的价格RMSE，但仍落后于单期混合和SVI的局部回归。

**⚠️ 局限性**

局限包括：模拟器仅覆盖有限的过程族；NIFTY数据为交易收盘价缺乏双边报价；传统方法按到期重新拟合，导致与神经网络的比较不完全对等；种子数目有限，统计显著性受限；数值条件表明价格只能约束密度的有限子空间，难以唯一确定潜在分布。

---

## 480. GraphQAG: A Knowledge-Graph-Guided Visual Analytics Framework for Question-Answer Pairs Generation

**arXiv ID:** 2607.27182 | [PDF](https://arxiv.org/pdf/2607.27182v1)

**作者:** Yize Li `[一作]` (Hangzhou Dianzi University), Zhiguang Zhou `[通讯]` (Hangzhou Dianzi University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于知识图谱的可视分析框架，用图结构引导大型语言模型（LLM）在长文档中生成、检验和迭代完善高质量问答对。

**💡 创新点**

创新点在于：①将文档知识抽象为可检索的知识图谱，构建图结构生成空间；②利用图与证据对齐实现检索式、可解释式问答对生成；③通过多层可视化（知识图谱、证据桥、问答空间）实现对生成质量的全局评估和局部修正，实现人机协同的迭代优化。

**🔧 技术方法**

技术手段包括：GraphRAG抽取实体关系、PageRank与Personalized PageRank评估实体重要性与子图关联；LLM（DeepSeek‑V4‑Flash）在图结构与文本证据驱动下生成问答；逆向验证子图与原图对齐；D3/Vue实现交互式可视化；FastAPI作为后端服务。

**📊 数据集**

主要实验数据集为两篇长文：Arthur Conan Doyle的《A Scandal in Bohemia》与GovReport数据库中的一份货币政策报告；另外使用16名参与者和4名专家进行用户研究与评价。

**📈 对比分析**

通过在同一实验环境下的对照实验（Baseline为仅提示式LLM生成）和专家评分对比。指标包括：用户效能与可用性（使用七分制问卷）、专家对四个质量维度（知识覆盖、推理复杂度、证据根源、非冗余）的评分。结果显示：相较于Baseline，系统在效能、可用性、推理复杂度和证据根源上显著提升；在知识覆盖上提升但不显著；在非冗余上略逊。

**⚠️ 局限性**

局限性包括：①仅支持文本型长文，无法处理表格、图表等多模态信息；②专家对证据根源与非冗余的评估一致性有限；③实验仅覆盖两篇文档与有限领域，缺乏对更大规模、不同专业和下游任务（如知识库构建、问答系统）的系统性验证。

---

## 481. DenseOn with the LateOn: Fully Open Dense and Late-Interaction Models for Multilingual, Long-Context, and Code Search

**arXiv ID:** 2607.27178 | [PDF](https://arxiv.org/pdf/2607.27178v1)

**作者:** Raphaël Sourty `[一作]` (LightOn), Amélie Chatelain `[通讯]` (LightOn)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并公开了规模达665M对的英语检索训练集以及1.88M对的高质量微调集，并通过机器翻译扩展至8种语言形成2.8B对的跨语言预训练集与16.3M对的多语言微调集；随后在此数据上训练了149M参数的单向向量检索模型和307M参数的ColBERT式交互检索模型；

**💡 创新点**

在开源数据与复现友好的训练流程上实现了检索模型的最前沿性能，并揭示了单向向量模型与交互模型在跨语言泛化上的显著差异，证明交互模型更能超越翻译训练的目标语言范围；

**🔧 技术方法**

采用ModernBERT/ mmBERT backbone，Contrastive InfoNCE预训练，硬负样本挖掘，知识蒸馏，Matryoshka/MeanMaxSim损失，GradCache大批量训练，以及Mistral-Small-3.1-24B-Instruct翻译与Mxbai-rerank-large-v2评估；

**📊 数据集**

使用从34个公开源（含FineWeb-Edu）构成的1.4B对原始数据；翻译后得到2.8B对；微调集来源于FiQA、NQ、HotpotQA、MSMARCO、FEVER、SQuADv2、TriviaQA、MIRACL、MLDR、LateOn-Code等；

**📈 对比分析**

在BEIR、MIRACL、MLDR、MTEB Code等基准上，单向向量模型在BEIR上获得56.20/57.22平均分，ColBERT式模型则达到57.56/57.56平均分；相比同规模闭源或公开模型均取得领先；跨语言实验中，ColBERT式模型在未训练语言和脚本上也保持较高性能；

**⚠️ 局限性**

主要限制在于大部分跨语言预训练监督为机器翻译生成，可能引入翻译错误与文化偏差；翻译质量未做人工评估；模型仅在八种目标语言上训练，未覆盖其他语言，泛化不均衡；

---

## 482. Partner Capability Estimation for Task-Agnostic Adaptation in Ad-Hoc Teamwork

**arXiv ID:** 2607.27177 | [PDF](https://arxiv.org/pdf/2607.27177v1)

**作者:** Peter Tisnikar `[一作]` (King's College London), Matteo Leonetti `[通讯]` (King's College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了一种基于隐藏能力的适应性框架CE‑CM，用于在多任务、未知合作伙伴的自组织团队协作中进行在线能力推断并生成联合计划。

**💡 创新点**

创新点在于：①将能力视为任务不变、可解释的约束；②采用近似贝叶斯采样（ABC）在上下文MDP中进行能力推断；③提出CE‑CM‑Div在面对行为多样性时通过对多条规划轨迹做匹配来提升鲁棒性；④实现了从零样本到多任务的迁移学习，避免了预训练。

**🔧 技术方法**

技术手段包括：模拟抽样式的近似贝叶斯推断（ABC）、上下文多智能体马尔可夫决策过程（CMMDP）、基于规划的联合规划器（PDDL或MCTS），以及多轨迹多样化规划生成。

**📊 数据集**

数据集：①仿真域TidyUP（PDDL）与Overcooked（模拟器）；②人类实验数据，15名参与者在Overcooked-like环境中完成225条轨迹，包含三种能力类型。

**📈 对比分析**

对比方法：乐观基线（假设合作伙伴完全可行）、悲观基线（仅观察到的转移才视为可行）以及无能力推断的单纯规划；评估指标为能力估计的汉明距离、纠正率、计划重叠率、不可行动作比例及开放式规划的可行动作比例。CE‑CM在TidyUP中快速收敛，汉明距离降至接近0；在Overcooked中虽能显著降低不可行动作比例，但对协调错误的改善有限。CE‑CM‑Div在真实人类数据上将汉明距离从约0.7降至0.3-0.4，接受样本数大幅提升，证明了对行为多样性的鲁棒性。

**⚠️ 局限性**

局限性包括：①仅依据能力约束无法区分多种等效策略，导致在策略多样的域中协调仍不充分；②缺乏偏好/约定建模；③推断与规划的计算开销较高；④难以直接扩展到大规模多智能体团队。

---

## 483. Improving Item Discoverability in e-Commerce Search via Related Intent Generation

**arXiv ID:** 2607.27172 | [PDF](https://arxiv.org/pdf/2607.27172v1)

**作者:** Ji Xin `[一作]` (Instacart), Tejaswi Tenneti `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于隐式意图生成的可扩展发现增强搜索系统，通过闭合权重LLM为高流量查询生成置换、互补、主题意图，使用LoRA微调的SLM覆盖长尾查询，并将生成的意图用于检索扩展召回；

**💡 创新点**

首次将意图生成框架与检索结合形成发现增强搜索，并提出两阶段混合架构（头部缓存LLM+尾部蒸馏SLM），同时设计了结合LLM评判和基于会话购买的双重评估方法；

**🔧 技术方法**

采用大型语言模型（GPT‑3.5/Turbo、GPT‑5.1教师、Qwen3‑30B学生）、LoRA适配器、教师‑学生蒸馏、少量示例提示、生成‑再检索管道、离线特征存储及会话级加权购买信号；

**📊 数据集**

使用约2万条人工标注的查询意图数据（头部10k查询+尾部10k查询），并构造基于历史搜索会话的评估数据集，将查询映射到产品类别、去除精确匹配和前10名结果；

**📈 对比分析**

与生产基线（缓存GPT‑3.5）和教师GPT‑5.1对比；端到端召回/精确率/F1提升至0.192/0.426/0.192（相较于基线0.173/0.260/0.173），LLM‑评判显示标题相关度0.84、意图相关度0.87、意图新颖度0.95，整体覆盖率从60%提升到80%，且成本约为教师模型的30%；

**⚠️ 局限性**

局限性包括蒸馏时上下文缺失导致尾部查询质量下降、解码温度对标题一致性敏感、最深尾部查询仍无法生成实用意图、LLM可能虚假品牌、离线评估受现有日志偏倚影响、LLM‑评判与教师模型同族可能产生偏差、缺乏在线A/B验证、需要定期刷新标注与重新蒸馏等维护挑战。

---

## 484. Free constructions for comprehension categories

**arXiv ID:** 2607.27170 | [PDF](https://arxiv.org/pdf/2607.27170v1)

**作者:** Francesco Dagnino `[一作]` (University of Genoa), Andrea Giusto `[通讯]` (University of Genoa)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对两类描述类型依赖的范畴论模型——Jacobs 的 comprehension category 与 Lawvere‑Ehrhard comprehension category（LECC）进行了深入比较，并给出了将任意 fibration、JCC 与 LECC 之间相互转换的自由构造。通过对术语 fibration 与全局类型 morphism fibration 的比较，阐明了 LECC 与 JCC 的本质区别，并证明了自由构造在 2‑范畴层面的双 adjunction 性质。

**💡 创新点**

创新点在于：①给出了 LECC 与 JCC 之间的 2‑functor 映射，并证明其本质像的判定条件；②构造了三个自由构造：①自由 JCC over a fibration；②自由在 JCC 上添加保真终对象得到“JCC with terminals”；③在此基础上自由完成为 LECC。所有构造均在 2‑范畴层面严格定义并证明了双 adjunction 关系，提供了一个统一的框架来生成新的模型。

**🔧 技术方法**

主要技术包括：Grothendieck fibration 的构造与解析；使用 fibration 的 pullback 和 cartesian lifting 定义上下文扩展与显示映射；通过 fibration 的自由终对象构造实现终对象的加入；利用半范畴与 2‑coequalizer 构造终对象的同余等价来完成从 JCC 到 LECC 的转换；在 2‑范畴中构造伪自然变换、单元与余单元，并证明三角恒等式。

**📊 数据集**

无实验数据集；本文为纯理论研究，所有结果以形式化证明给出。

**📈 对比分析**

比较方法主要是通过证明双 adjunction 的三角恒等式与单元/余单元的伪自然性来展示构造的正确性。由于是理论构造，未涉及实验性能比较；相对其它模型（如 D‑category、自然模型等）的比较在论文中通过结构属性（如终对象是否被保留、术语与全局 morphism 是否同构等）进行阐述。

**⚠️ 局限性**

局限性：
• 论文中假设所有 fibration 都是 cloven，并不讨论未能选取清晰的 cleavages 的情形；
• 对于非 split 或非 faithful 的 fibration，所给出的自由构造可能不具备更强的同构性质；
• 论文未给出对具体类型构造（如 Σ、Π、Id）在 LECC 中的实现细节；
• 由于理论性强，缺乏对模型在具体编程语言或证明助手实现中的可行性与效率分析。

---

## 485. Mental World Modeling

**arXiv ID:** 2607.27201 | [PDF](https://arxiv.org/pdf/2607.27201v1)

**作者:** Hao Fei `[一作]` (University of Oxford), Yiran Zhao `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了心理世界建模（Mental World Modeling，MWM）框架，并实现了可检验的基线系统，能够将情境解析为耦合的物理–心理状态，渲染目标代理的部分观测，模拟候选动作的物理和心理后继状态，并根据价值评估选择最合适的动作。

**💡 创新点**

核心创新在于：①将心理变量（信念、意图、情感、规范等）与物理状态视为同一全局状态；②在世界模型中显式模拟目标代理的观测与动作的耦合；③提出可解释、无训练的多阶段基线；④构建完整过程注解的数据集（Menti‑Bench），用于验证心理状态对人类决策的必要性。

**🔧 技术方法**

技术手段包括：POMDP框架下的观测生成、动作分解和状态转移；结构化状态与观测的JSON表示；利用大规模语言模型（GPT‑5.6、Claude 等）完成语义解析、状态构造和价值评估；多模态输入（文本、图像、音频）与视觉/音频特征融合。

**📊 数据集**

使用了自构造的 Menti‑Bench 数据集：448 条多模态情境（320 文本、100 图像、28 声音视频），每条记录均标注完整的物理–心理状态、观测、后继状态和正确动作，覆盖四类场景和五个日常领域。

**📈 对比分析**

采用从 S0（仅选项）到 S6（完整 MWM）阶梯式对比，并进行心理/物理通道消融、转移耦合消融以及四阶段金手指（oracle）干预；评价指标为最终动作的 F1。结果显示，完整 MWM 在所有模型上从约31% 提升到约90%，最显著提升出现在人际场景；不同模态下的性能差距被结构化基线消除；消融实验验证了心理与物理通道及其耦合的必要性。

**⚠️ 局限性**

主要局限在于转移模拟的准确性：oracle 实验表明，转移阶段误差占剩余 7.8 点差距的约 45%；此外，状态解析和观测生成的误差也对整体性能产生影响，且当前模型对复杂社会交互的长期演化仍不够稳定。

---

## 486. Can AI agents conduct open-ended AI research? Early evidence from two case studies

**arXiv ID:** 2607.27191 | [PDF](https://arxiv.org/pdf/2607.27191v1)

**作者:** Peter Kirgis `[一作]` (Princeton University), Arvind Narayanan `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过让前沿大模型代理（如Claude Opus 4.8、GPT‑5.6 Sol）在不公开的NeurIPS 2026论文研究问题上自行开展实验、撰写论文，并请原作者以同行评审方式对代理产出进行打分，从而评估AI自动化研究能力；

**💡 创新点**

提出并验证了“影子评估（shadow evaluation）”这一新方法，它结合开放式任务与专家主导的评审，填补了可验证任务与盲评审之间的评估空白；

**🔧 技术方法**

采用多模态大语言模型代理，配合OpenClaw框架实现任务调度、子代理协作与GPU资源管理，并通过外部AI评审工具（Stanford Agentic Reviewer、CMU Paper Reviewer、refine.ink）以及人类原作者评审进行自评与交叉验证；

**📊 数据集**

代理在实验过程中使用了公开的真实数据集（如OpenML上的分布漂移数据）、合成实验数据以及手工挑选的样本集，重点在实验设计与数据选择上进行探索；

**📈 对比分析**

评估方式为将代理撰写的论文提交给原作者，按照NeurIPS评审标准进行评分；结果显示两篇论文均被拒绝，表明代理虽能完成工程任务，但在创新性、实验设计与论证质量上无法达到顶级会议水准；

**⚠️ 局限性**

局限性包括样本量极小（仅两篇论文）、评审非盲评可能产生偏见、OpenClaw等框架的技术缺陷、资源管理与指令漂移问题，以及评估者自身的主观偏见导致的结果解释不确定性。

---

## 487. APEX-Accounting

**arXiv ID:** 2607.27189 | [PDF](https://arxiv.org/pdf/2607.27189v1)

**作者:** Julien Benchek `[一作]`, Bertie Vidgen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 APEX‑Accounting 基准，包含四类会计任务（对账、录入、差异分析、计划与计提），并评估多种前沿大型语言模型的实际会计工作能力。

**💡 创新点**

创新点在于：①将真实会计工作拆解为可评估任务并通过专家制定细粒度评判标准；②构建大规模、跨域 synthetic 公司世界；③引入“工具调用”与“子代理”式代理架构以模拟真实工作流程。

**🔧 技术方法**

采用大语言模型与工具调用的 ReAct/子代理架构（Loop 与 Ramp harness），配合 91 种文件与会计软件工具，并使用 LM judge 进行自动化评判。

**📊 数据集**

使用的数据信息为：≈1400 个任务，分布在 56 个 synthetic 公司世界，涵盖 1.3K+ 文档（Excel、PDF、Docx 等）以及专家生成的黄金答案与评判准则。

**📈 对比分析**

通过 Mean Criteria@、Pass@、Passˆ 等指标与 9 大 frontier 模型（Claude‑Fable‑5、Muse‑Spark‑1.1、GPT‑5.6‑Sol 等）进行对比，Claude‑Fable‑5 以 59.4% 的 Mean Criteria@ 首位，Pass@ 最高为 21.5%，但整体一致性仍低于 10%。

**⚠️ 局限性**

局限性包括：①任务范围仅覆盖闭环日常会计，未涉及税务、审计、合并、跨币种等；②任务筛选依赖部分模型，可能导致对这些模型偏向；③评判者（LM judge）虽高精度但仍有误差；④基准样本量有限，统计显著性仅在部分排名差距可观察。

---

## 488. Pangram 4 Technical Report

**arXiv ID:** 2607.27183 | [PDF](https://arxiv.org/pdf/2607.27183v1)

**作者:** Ben Glickenhaus `[一作]` (Pangram Labs), Bradley Emi `[通讯]` (Pangram Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并训练了一款名为 Pangram 4 的深度学习 AI 文本检测模型，能够识别完整 AI 生成文本、AI 帮助编辑文本以及纯人类文本，并提供 token 级别的作者归因。

**💡 创新点**

创新点包括：1) 通过 Soft N‑Gram 标注生成混合文本的 token‑级别标签；2) 在 causal LM 背景下采用 Repeat2 技术实现完整上下文的 token‑级别分类；3) 结合多任务学习（段级、token‑级、混合写作二元、humanizer）与 CRF 后处理，实现细粒度边界检测和多语言鲁棒性；4) 通过阈值策略与句子多数投票降低误报，同时保持对人类化攻击的检测能力。

**🔧 技术方法**

使用的技术包括：基于 MoE 的 causal 语言模型 + LoRA 调优；Soft N‑Gram 软标签生成；Repeat2 复制输入实现全局上下文；多任务头（段级 15‑桶、token‑级 3‑类、混合写作 2‑类、humanizer 4‑类）；校准后的 CRF 解码；三角基函数映射段级 AI 比例；句子多数投票与最小段长度合并。

**📊 数据集**

训练和评估数据涵盖多种语言与域的真实人类写作、AI 生成文本、AI 版面编辑文本、合成镜像文本以及人工混合文本；包括 520,000 条使用 26 个前沿 LLM 生成的合成 benchmark、FineWeb2、FineWeb、UChicago、VUB、GEDE、MELD‑eval、Sem‑Detect、Saha Peer Review、OpAI‑Bench 等公开基准；此外使用多语言 FineWeb2 进行多语言评估。

**📈 对比分析**

与前代 Pangram 3.3.2 以及公开基准进行比较。Pangram 4 在 FNR 0.3396%（比 1.9942% 降低约 80%）与 FPR 0.0041%（比 0.0539% 降低约 95%）上均实现显著提升；AUROC 0.9916；在混合写作检测方面，AI‑Assisted recall 提升至 55%（前 5%）和 65.7%（前 12.6%），在异质混合文本的 token‑级别准确率、精确率、召回率均超过 75%，并且在多语言与非母语英语测试中保持低误报率。

**⚠️ 局限性**

局限性包括：对极短文本（<50 词）仍易产生误判；某些语言（如乌尔都语、波斯语）因分词差异导致 FNR 较高；对极高 AI‑fraction 的混合文本仍可能出现误分类；模型对未知生成器的泛化虽提升但仍不确定；以及在极端人类化攻击场景下仍有潜在误报。

---

## 489. When Do Learned Diffusion Proposals Help Constraint Solving? A Controlled Study on Continuous Algebraic Systems

**arXiv ID:** 2607.27169 | [PDF](https://arxiv.org/pdf/2607.27169v1)

**作者:** Quang Bui `[一作]` (SAID Lab), Davin Yin `[通讯]` (SAID Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在连续代数约束系统求解中，本文同时研究了两个决策：对约束满足的数值取值以及对系统结构的增补（结构修复）。作者提出了两种学习组件：① 一个基于候选图的修复排名器，用于在离散结构决策上挑选最有可能使系统可解的增补；② 一个基于扩散去噪器的值生成器，用于在连续取值决策上提供初始解。两者均在 MARC 框架下与符号计算（CAS）梯度、确定性能量下降以及两阶段符号检查相结合，形成完整求解流程。

**💡 创新点**

主要创新点包括：
1) **结构修复的学习化**：首次在连续约束系统中将结构增补视为可学习的候选条件化排名任务，并通过候选图编码显著提升枚举效率。 
2) **基于范式可分离性和可达性斜率的“因子分解定律”**：提供了一个无参数预测模型，说明学习值生成器仅在“可达性指数呈几何衰减且解可逐变量分解”的情形下才有优势。 
3) **严格的实验协议**：引入双流失败筛选、随机多启动对照、预算匹配以及基准基准检验，剔除随机样本偏差，确保结果可复现且对比公正。

**🔧 技术方法**

所用技术主要包括：
- **MARC**：将代数约束表示为因子图，利用 CAS（SymPy）计算残差、能量与梯度。
- **扩散去噪器**：使用标准的双向扩散（T=1000）和 BPGNN 消噪网络，并在 DDIM 逆过程加入能量梯度引导。
- **修复排名器**：基于候选增强后的图结构进行信息传播，使用列表式评分决定最佳增补。
- **确定性能量下降**：Levenberg–Marquardt/梯度下降作为后处理。
- **两阶段符号检查**：先快速数值检查，再符号化精确根检验。

**📊 数据集**

数据集：
- **合成数据**：通过程序化生成的八个非凸约束族（可分离、耦合、几何、双曲、三角等），每族包含数十到数百个测试实例。
- **真实系统**：八个来自机器人定位、逆运动学、优化与代数的实际约束系统（如 GPS 三角定位、圆交、Rosenbrock 等）。

**📈 对比分析**

比较方法与性能：
- **对照**：随机多启动（相同预算）、单启动梯度下降、Langevin 噪声、平均先验控制、候选仅随机挑选等。 
- **结果**：
  - 在结构决策上，修复排名器在平衡非线性菜单中达 0.997 的解决率，远超随机 0.333 与候选仅随机 0.236；对真实系统中的硬化实例，修复排名器能够在 1.000 的概率下修复所有失败。 
  - 在取值决策上，扩散生成器仅在“可达性指数呈几何衰减且解可逐变量分解”的情形下表现优异，跨维度上在 n≥3 时可显著击败随机多启动；在耦合系统中则无显著优势。 
  - 在预算匹配的随机多启动控制下，扩散生成器在可分离族中的成功率与理论预测（1-(1-q)^K）几乎一致（MAE 0.012）。

**⚠️ 局限性**

局限性：
1) 扩散值生成器仅在可达性指数呈几何衰减且解可分解时有效，耦合系统或高维非可分离问题无明显提升。 
2) 结构修复排名器的效果依赖于图表征的质量；对极为复杂或非线性不确定的约束族，模型性能可能受限。 
3) 所有实验均在先前已自动化的因子图表示上进行，尚未覆盖非结构化文本或自然语言描述的约束问题，自动化形式化的瓶颈仍未突破。 
4) 对于真实系统的验证仅涉及八个样本，未覆盖更广泛的工业级约束场景。

---

## 490. SpecFirst: Behavioral Specification Elicitation as a First-Class Step in Agent-Based Program Synthesis from Scratch

**arXiv ID:** 2607.27167 | [PDF](https://arxiv.org/pdf/2607.27167v1)

**作者:** Yihao Chen `[一作]` (Queen's University), Ahmed E. Hassan `[通讯]` (Queen's University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段的LLM代理框架SpecFirst，在程序从零开始构建前先进行行为规范的提取（Spec Agent），随后再进行代码合成（Synthesis Agent）

**💡 创新点**

将需求获取（行为规范提取）作为独立第一阶段，引入专用Spec Agent主动探测可执行二进制文件，生成结构化规范，解决单循环框架的探测不足、规范漂移和错误累积问题

**🔧 技术方法**

利用大型语言模型（Qwen3.5/3.6、GPT-5.5、GPT-5.4），ReAct代理模式，工具调用（shell、文件编辑、探测脚本），黑盒二进制探测，结构化规范格式（Sections）

**📊 数据集**

ProgramBench 200个命令行程序实例，涵盖Rust、Go、C/C++、Java、Haskell，提供README/手册和执行二进制作为行为Oracle

**📈 对比分析**

对比基线（直接代码合成无规范阶段），使用平均测试通过率和探测覆盖率作为指标，SpecFirst在所有四个模型上平均提高6.9%–21.3%的测试通过率，探测覆盖率提升9.4%–18.5%，且提升显著性达p<0.01

**⚠️ 局限性**

成本上升（约48%–130%），仍存在规范遗漏、错误描述、实现偏差等失败模式，且实验仅覆盖确定性命令行程序，对非确定性或交互式程序的泛化尚未验证

---

## 491. Function Privatization in the Local Model

**arXiv ID:** 2607.27164 | [PDF](https://arxiv.org/pdf/2607.27164v1)

**作者:** Yuting Liang `[一作]` (University of Toronto), Ke Yi `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了在局部 Geo-Privacy 模型下对函数（尤其是曲线）进行隐私化的方法，能够自动利用函数的内在相关性并保持连续性。

**💡 创新点**

创新点在于：① 引入 L² 作为函数隐私度量并将其与 Geo-Privacy 结合；② 设计基于基函数投影的通用机制；③ 开发自适应基函数/分段选择算法，平衡近似误差与隐私噪声；④ 在曲线隐私化中实现分段平滑与连续性恢复。

**🔧 技术方法**

核心技术包括：L² 级别的 Geo-Privacy 及其浓缩版本 CGP；正交/非正交基函数投影与 Gram 矩阵计算；稀疏向量技术 (SVT) 用于自适应选择基函数数量和分段；球面 Laplace 与高斯噪声添加实现 ε-GP/ρ-CGP；二次规划保证输出曲线连续。

**📊 数据集**

实验使用三类数据集：ECG 信号（PTB‑XL），出租车轨迹（CRAWDAD），以及合成高斯混合曲线，用以评估 L² 误差和 MSE。

**📈 对比分析**

与基于随机采样点的 Baseline（含平滑）相比，提出的方法在 L² 误差上提升 1–2 个数量级，MSE 提升 2–3 个数量级；在较高隐私预算下差距更大；同时能保持曲线的主要形状与峰值信息。

**⚠️ 局限性**

局限性包括：仅针对 L² 距离的隐私定义；对极高维或无界区间的处理仍需额外步骤；自适应基函数选择依赖 SVT 的误差近似，可能在噪声较大时失效；实现复杂度相对较高。

---

