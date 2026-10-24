# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-10-23 | 今日论文总数: 667

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Agentic Architecture of Information Systems. The AAM/5C Model: A Reference Model for Distributed Autonomy

**arXiv ID:** 2610.30221 | [PDF](https://arxiv.org/pdf/2610.30221v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 2. Edit, Don't Imitate: Minimal-Edit On-Policy Distillation for Multimodal Reasoning

**arXiv ID:** 2610.30231 | [PDF](https://arxiv.org/pdf/2610.30231v1)

**作者:** Wen Ye `[一作]` (Meta Superintelligence Labs), Hengduo Li `[通讯]` (Meta Superintelligence Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在训练阶段利用图形描述作为特权信息，对视觉语言模型进行最小编辑纠正的基于策略的蒸馏，提升其在无图形描述推理时的几何推理与感知能力。

**💡 创新点**

提出了最小编辑纠正的基于策略蒸馏（Minimal‑Edit Corrective OPD），该方法只在错误生成上给出针对性标记，而不需要生成完整的修正答案，显著降低了教师的能力要求并实现了对同规模教师的有效蒸馏。

**🔧 技术方法**

利用对策略的逆 KL 损失、奖励路由（GRPO）、与教师对齐的 token‑level k₁ 估计，并在错误生成时使用最小编辑条件对教师评分，随后仅对被教师低估的 token 进行更新。

**📊 数据集**

使用 Geometry3K（几何图题）和 MathVerse（多模态数学）作为推理评测，Geoperception 作为感知评测；同时在训练中为每个问题生成参考图形描述和已验证参考解答。

**📈 对比分析**

与多种基线（SFT、GRPO、普通 OPD、特权 OPD）以及公开的 7B、32B 模型进行对比，结果显示在 2B、8B 模型上，最小编辑蒸馏分别提升 Geometry3K pass@1 0.55/0.70、MathVerse 0.71/0.90 以及感知指标，且在无图形描述推理时已接近甚至超越加入描述的基线。

**⚠️ 局限性**

该方法依赖于可恢复的特权描述，且最小编辑标记可能无法覆盖所有错误位置，导致部分错误未被纠正；此外在训练中需要先验的已验证参考解答，若生成失败会削弱效果。

---

## 3. JANUS: Agentic Cross-Language Fuzzing Harness Generation for Android Native Libraries

**arXiv ID:** 2610.30218 | [PDF](https://arxiv.org/pdf/2610.30218v1)

**作者:** Xiangyu Zhang `[一作]` (Nankai University), Sen Chen `[通讯]` (Nankai University)

**通讯引用:** 31953 | [OpenAlex ID](https://openalex.org/A5100371368)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于LLM代理的跨语言JNI感知fuzzing harness自动生成系统Janus，可为闭源Android本地库自动合成完整的Java‑C/C++交互 harness。

**💡 创新点**

创新点包括：① 将LLM Agent与结构感知检索器、调用序列提取器、递归参数构造器和记忆管理器组合成端到端的代理式工作流；② 通过跨语言语言服务器和AST‑Grep模板实现精确的跨语言信息检索；③ 采用反馈驱动的迭代修正机制抑制hallucination并提升 harness 的有效性。

**🔧 技术方法**

技术栈涵盖结构感知跨语言语言服务器、AST‑Grep模板、GLM‑5 LLM、工具调用接口、依赖驱动的调用序列算法、递归对象构造、Harness 内存管理、以及覆盖率/错误反馈的迭代修正。

**📊 数据集**

使用两个数据集：Poirot‑44（44个流行应用，共21,386 JNI方法）用于评估生成质量；Vendor‑20（20个主流厂商预装应用）用于真实漏洞挖掘。

**📈 对比分析**

与Atlas++和Poirot在Poirot‑44上对比：Janus生成7,257个可编译、6,094个可运行的 harness，平均覆盖率16.9%（Atlas++ 2.9%、Poirot 2.6%），检测到5,390个崩溃；在Vendor‑20中发现6个此前未知的内存漏洞。

**⚠️ 局限性**

局限性：仅采用线性执行模型，未处理异步/并发依赖；不支持Binder/系统服务等深层 Android 交互；在模拟环境下部分库因外部状态或反仿真检查导致无法运行；LLM 代理对计算资源和成本敏感。

---

## 4. How Prompt Style Affects AI Accuracy and Speed in Competitive Programming

**arXiv ID:** 2610.30223 | [PDF](https://arxiv.org/pdf/2610.30223v1)

**作者:** Saad B. Alarifi `[一作]` (Riyadh Schools), Mohammed A. Alam `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究在GPT‑5.5上，使用九种不同的提示（prompt）风格，对五个竞争性编程任务（三道IOI准确性任务和两道NP‑hard速度任务）进行实验，评估提示对代码准确率和执行速度的影响，并将AI结果与IOI参赛者成绩做对比。

**💡 创新点**

创新点在于：①首次同时考察提示风格对准确性和运行速度的双重影响；②将AI表现与人类参赛者的真实成绩进行对比；③通过统计检验揭示提示效果随任务类型和度量方法变化而非单一最佳提示。

**🔧 技术方法**

使用的技术包括：GPT‑5.5大语言模型、单次调用的prompt生成与代码生成、在线判题平台oj.uz进行准确性评分、本地自制测试集进行速度评估、Kruskal‑Wallis检验与Spearman相关性分析。

**📊 数据集**

数据集为：①IOI 2022/2024/2025的三道题目（Prisoner Challenge、Message、Migrations）；②Maximum Clique Size与Number Partitioning两道NP‑hard问题的自制测试集；人工参考数据为对应年份IOI参赛者平均和金牌得分。

**📈 对比分析**

比较方法：每个提示风格对每道题目做9次单独调用，记录准确性分数（0–100）与速度分数（0–100）。通过平均值、众数、Kruskal‑Wallis显著性检验和相关系数评估提示效果。结果显示：提示6在准确性上表现最好；提示3在速度平均分上最高；不同任务和指标下最佳提示不一致，提示效应在某些任务显著（如Prisoner Challenge、Message、Number Partitioning），而在其他任务不显著。AI在准确性上对平均参赛者成绩相近，在某一任务略优，整体表现与人类不明显优劣。

**⚠️ 局限性**

局限性包括：①每种提示仅使用一个生成实例，可能无法代表整个风格；②仅测试GPT‑5.5模型，缺乏跨模型验证；③仅有9次迭代，统计功效有限；④速度评估受硬件热降频和单一测试顺序影响；⑤IOI题目已在模型训练集前出现，可能导致训练偏差；⑥AI与人类实验条件差异大，结果仅为描述性对比。

---

## 5. Deep Search in Real-World Web Environments

**arXiv ID:** 2610.30181 | [PDF](https://arxiv.org/pdf/2610.30181v1)

**作者:** Xiaomin Yu `[一作]` (Hong Kong University of Science and Technology), Chengwei Qin `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了名为 OpenDS-World 的深度搜索基准，定义了 Open-Web Deep Search（OpenDS）任务，收集了 200 个基于视频锚点的真实 Web 任务，并提供完整的多模态、异构格式和交互式访问的证据链。

**💡 创新点**

首次提出了以视频为起点、涵盖多模态（文字、图像、音频、视频）、多格式（HTML、PDF、MP4、WebP 等）和交互访问需求的深度搜索任务，保证了任务的真实性、唯一答案和跨源证明的可验证性，并通过人类验证与任务筛选保障数据质量。

**🔧 技术方法**

利用大型语言模型（LLM）与工具化环境（Search、GUI、Video/Subtitles 等）进行实验评估；采用 LLM 判定器（Qwen3.8‑27B）判定答案准确性，并分析工具调用、时间成本、错误类型等。

**📊 数据集**

OpenDS-World 数据集：200 个任务、787 条证据条目，覆盖 8 个领域、4 种模态、42 种文件格式、7 种访问类型；所有任务均经过 10 名评审的真实性和任务有效性验证。

**📈 对比分析**

在统一的工具与预算约束下，评测 12 个前沿模型，最高准确率 39.0%（GPT‑6 Astra），人类独立尝试 1,000 次的准确率仅 21.1%；对比分析了工具使用比例、搜索成本、预算消耗、失败模式等，揭示不同模型在信息源选择、工具协调和长程搜索管理上的差异。

**⚠️ 局限性**

主要局限：模型准确率仍偏低；在高难度任务上准确率下降显著；工具调用与时间预算经常被耗尽；错误频发（工具失败、自然终止不一致）；缺乏高效的证据整合与停止策略，难以在真实 Web 环境中完成长链搜索。

---

## 6. Agentic Verified Optimization of Real-World Software

**arXiv ID:** 2610.30188 | [PDF](https://arxiv.org/pdf/2610.30188v1)

**作者:** Zhe Ye `[一作]` (University of California), Soonho Kong `[通讯]` (Amazon Web Services)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了agentic verified optimization，即让智能体在保持等价证明的前提下为已有程序生成更快的实现并证明等价。

**💡 创新点**

创新点在于将等价证明与性能优化结合，避免编写规范的瓶颈，并在Lean 4中实现函数级和仓库级的自动化优化；同时设计了迁移框架将多种语言程序迁移到Lean进行验证。

**🔧 技术方法**

使用了大型语言模型（Codex/GPT‑5.6），Lean 4 证明助手、自动化迁移工具、差分测试、覆盖率驱动的模糊测试与变异测试、以及Lean的内置编译器与运行时。

**📊 数据集**

实验数据集包括：189个单函数Lean 4任务（108个基础、81个高级），以及17个通过迁移框架迁移到Lean的真实程序（涵盖C、C++、Go、Rust）。

**📈 对比分析**

通过固定工作负载的配对时间测量，对函数级实现比较速度提升（平均1.851×、最高128,292×），对仓库级应用比较则显示集成后的Lean版本比迁移版平均提升1.864×，但相较于原始源码提升仅1.120×；验证通过Lean内核检查保证等价。

**⚠️ 局限性**

局限性包括：优化仅覆盖无隐藏副作用的纯函数或在Lean中可定义的Monad；对不可见的I/O、并发或浮点容差等行为缺乏等价证明；应用级加速受限于函数选择与整体程序结构；目前未能处理跨语言或更大规模的系统级优化。

---

## 7. baseVultra: Compact Nonlinear Recurrence with Reallocated Capacity

**arXiv ID:** 2610.30154 | [PDF](https://arxiv.org/pdf/2610.30154v1)

**作者:** Linke Zhang `[一作]` `[通讯]` (Southwest University), Linke Zhang (Southwest University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 baseVultra 细胞，将密集递归计算聚焦于候选生成，门控仅用 O(H) 对角交互，整体实现更轻量的递归模型。

**💡 创新点**

创新点在于递归容量重新分配：将 H×H 的全秩递归矩阵只用于候选生成，门控使用稀疏对角交互；并加入低秩分支、EMA 缓存和二阶隐藏状态访问，提升紧凑模型性能。

**🔧 技术方法**

采用轻量级门控、低秩分支、多分支融合、EMA 缓存、二阶隐藏状态访问以及参数匹配实验与正式语言评测等技术。

**📊 数据集**

使用 TinyShakespeare、text8 字符级语言模型数据集，并在七项正式语言任务（Parity、Counter、Lock、Dyck-2、Copy、Delay-2 Copy、Noisy Counter）上进行评测。

**📈 对比分析**

通过与参数匹配的 LSTM、GRU 进行对比；在 TinyShakespeare compact 1 层 200K 规模下 baseVultra 的验证损失最低；在 text8 上与 GRU/LSTM 持平；在低容量正式语言任务中表现最好。

**⚠️ 局限性**

局限性包括：仅在紧凑规模有效；在大规模任务不优；壁垒时间慢（受 kernel launch 限制）；正式语言评测仅覆盖 4 倍长度外推，未评估长文档语言建模。

---

## 8. MCP vs CLI for AI Agents: A Comparative Analysis of Tool Integration Strategies for Autonomous Systems

**arXiv ID:** 2610.30157 | [PDF](https://arxiv.org/pdf/2610.30157v1)

**作者:** Arun Malik `[一作]` (Microsoft Corporation), Kunal Sonawane `[通讯]` (Microsoft Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过 14 维度对比分析了 CLI 与 MCP 在 AI 代理工具集成中的效率、安全、可扩展性等多方面表现，并给出了实践案例与经济模型。

**💡 创新点**

创新点在于提出“进阶披露”技术，显著降低 MCP 的上下文窗口开销，同时提出混合架构作为企业级推荐，弥合了两者的性能与治理差距。

**🔧 技术方法**

使用了 JSON‑RPC 的 MCP 规范、命令行接口、LLM 任务规划模型，并结合 NSA、OWASP、BCG、Gartner 等标准进行评估。

**📊 数据集**

主要采用公开的安全合规框架（SOC2、HIPAA、GDPR 等）和行业报告（如 BCG 成本降低 70% 报告）作为对比基准。

**📈 对比分析**

通过 14 维度评分、token 费用、延迟、攻击向量分析等方法比较，CLI 在效率上占优，但 MCP 在安全、治理、可扩展性上明显领先；进阶披露后两者的 token 费用基本持平。

**⚠️ 局限性**

局限性在于实验主要基于模拟场景，未涵盖所有多租户真实部署情况，且进阶披露实现细节仍需在更大规模生产环境进一步验证。

---

## 9. SWE-VERIFIER: Selective Verification for Cost-Efficient Coding Agents

**arXiv ID:** 2610.30284 | [PDF](https://arxiv.org/pdf/2610.30284v1)

**作者:** Kaijian Zou `[一作]` (University of Michigan), Lu Wang `[通讯]` (University of Michigan)

**通讯引用:** 24354 | [OpenAlex ID](https://openalex.org/A5100364413)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种轻量级的生成式验证器，用于在代码生成代理的每一步选择是否进行额外候选动作的采样与排序，从而在不显著增加计算开销的前提下提升决策质量。

**💡 创新点**

创新点包括：① 在动作选择时引入门控机制（gate）来决定何时触发额外采样；② 使用同一生成模型同时评估候选动作并输出评分和评价理由；③ 通过执行感知的偏好数据构造方法，将教师监督与对动作后果的预期评估结合，显著提高验证器的判别能力。

**🔧 技术方法**

技术实现主要包括：Qwen3.5-4B 作为基础生成模型；在每一步先用门控评分判断是否采样更多候选动作；对采样候选动作进行生成式评分与排序；使用 GPT‑5.6 作为教师生成结构化的偏好标签（包含执行结果、进度评估和 rubric）；在训练中采用 Jaccard 过滤去除近似相同动作；最终实现低成本的步骤级验证。

**📊 数据集**

数据集：从 Opus 4.7 和 GPT‑5.5 生成的 500 条成功 SWE‑Bench‑Verified 轨迹；在这些轨迹中采样不同模型（MiniMax‑M2.7、Qwen3.5‑27B 等）的候选动作，利用 GPT‑5.6 生成教师标签；训练集包含 30k 条动作标签，来自 6k 步。

**📈 对比分析**

与传统的轨迹级验证、全步验证以及无验证基线相比，实验在 MiniMax‑M2.7 上验证率仅 3.43% 的步骤即可把任务解决率从 75.47% 提升至 77.20%，总成本仅 1.09×；在 DeepSeek‑V4‑Flash 上，解决率从 72.60% 提升至 74.27%，成本 1.13×。与全步验证相比，显著降低了计算成本并保持或提升了性能。

**⚠️ 局限性**

局限性：① 仅在 mini‑swe‑agent 环境下验证，其他代码生成基准或不同仓库分布下的表现未知；② 成本分析基于特定的缓存折扣与 API 费用假设，实际服务环境可能导致盈亏点不同；③ 验证器依赖教师生成的偏好标签，若教师偏差或缺少失败模式，验证器在这些场景下的可靠性可能受限。

---

## 10. AI Mindmeld: A Latent Interconnect Protocol

**arXiv ID:** 2610.30215 | [PDF](https://arxiv.org/pdf/2610.30215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 11. FormBench: A Benchmark for Fine-Grained Retrieval in Formulation Patents

**arXiv ID:** 2610.30175 | [PDF](https://arxiv.org/pdf/2610.30175v1)

**作者:** Vineeth Venugopal `[一作]` (Reciprocal Space), Jonathan Welch `[通讯]` (Albert Invent)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个用于测试模型从技术文档中检索精确化学信息的基准平台，包含55,352个查询和两个检索语料库变体，每个变体约有100万个段落。

**💡 创新点**

创新点在于设计了基于材料本体的分级相关性评分，提供了更深层次的模型区分能力评估，并且通过任务自适应微调（TAPT）显著提升了模型性能。

**🔧 技术方法**

使用了多种模型配置，包括商业API编码器（如OpenAI和Voyage）和开源编码器（如Nomic、GTE-large、Qwen3-Embed），以及它们的任务自适应微调（TAPT）变体。

**📊 数据集**

使用的数据集包括590,247个来自USPTO的配方专利，经过命名实体识别（NER）处理后生成了约3.1百万个候选段落，并从中提取了55,352个查询。

**📈 对比分析**

与BM25基线进行比较，发现商业API在所有配置中表现不佳（如OpenAI-3-large在C1上得分为0.009），而TAPT-Qwen3-Embed在C1上得分为0.4085，超越了BM25基线（0.375）。

**⚠️ 局限性**

局限性在于相关性判断是基于计算的代理（基于实体集的Jaccard相似度），并且分级相关性评分的精确度不如人工判断，覆盖范围仅限于1995-2022年间的USPTO实用专利，且仅为英语。

---

## 12. IRAH: Hierarchical Agentic Tree Search for Retrieval Pipeline Optimization

**arXiv ID:** 2610.30167 | [PDF](https://arxiv.org/pdf/2610.30167v1)

**作者:** Maria Ana Cardei `[一作]` (University of Virginia), Rashmi Shetty `[通讯]` (Capital One)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了IRAH框架，利用语言代理树搜索结合元学习初始化，自动化优化检索增强生成系统的检索管线配置。

**💡 创新点**

创新点在于把检索配置拆分为宏观结构探索与微观数值细化的两阶段层次搜索，并通过自然语言反思指导后续变异，同时使用离线元学习进行温启动。

**🔧 技术方法**

使用技术包括语言代理树搜索（LATS）、UCT‑guided 树搜索、自然语言反思与候选变异生成、以及基于数据集与配置特征的离线元学习评分模型。

**📊 数据集**

在三类在分布检索基准（SciFact、FiQA、CQADupStack）和一类超分布代码检索基准（CoSQA）上进行评估。

**📈 对比分析**

与网格搜索、TPE（冷/热启动）在相同的50次评估预算下对比，IRAH热启动平均 nDCG@10 为 0.4816，优于 TPE 约 6.4%，在超分布基准仍保持竞争力。

**⚠️ 局限性**

局限性包括评估仅覆盖单一检索后端、依赖大型LLM导致推理成本高、以及热启动效果受元训练分布匹配度影响，对极端分布偏移的鲁棒性不足。

---

## 13. Alpha: Systematic Fault Injection and Robustness Evaluation for LLM-Driven Embodied Manipulation

**arXiv ID:** 2610.30197 | [PDF](https://arxiv.org/pdf/2610.30197v1)

**作者:** Jona Joy `[一作]` `[通讯]`, Jona Joy

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建Alpha框架，对LLM驱动的机器人操作进行系统化的故障注入与鲁棒性评估，探究感知、执行、运动学和序列化四个边界的鲁棒性。

**💡 创新点**

① 匹配对照基线：LLM规划器与确定性Scripted Oracle在相同随机种子、物理参数和安全阈值下并行测试；② 模块化故障注入设计，可在感知、执行、运动学、序列化四个层面单独或组合注入故障；③ 细粒度安全监测与去重窗口，准确计数唯一安全事件；④ 证明物理故障与计划器序列化异常是导致性能差异的根本原因，而非LLM认知缺陷。

**🔧 技术方法**

使用Llama-3（本地Ollama推理）、PyBullet仿真、Pydantic架构校验、240Hz安全监测器、配对McNemar统计、Fault-injection Middleware等技术。

**📊 数据集**

仅采用单一桌面pick‑and‑place任务的150条仿真episode（控制+单/多故障+并发），以及169条历史pilot记录；不存在公开大规模数据集，全部在自建仿真环境中生成。

**📈 对比分析**

通过seed‑匹配的配对实验，将LLM与Scripted Oracle在同一情景下比较。总体成功率：Oracle 64.0%，LLM 44.0%（p=3.01×10⁻⁴）。排除输出序列化损坏后，两者在物理、感知、运动学条件下均达到55.0%相同成功率，差异完全归因于Planner‑Output Corruption导致的Pydantic验证失败。

**⚠️ 局限性**

单一任务单模型、样本量每个强度细胞仅5例、感知延迟仅限0.5 s、未评估闭环重抓、仅测序列化错误不涉及语义推理失效、结果仅基于仿真，缺乏真实硬件验证。

---

## 14. PhishGen: Understanding LLM Capabilities in Automated Phishing Website Generation

**arXiv ID:** 2610.30226 | [PDF](https://arxiv.org/pdf/2610.30226v1)

**作者:** Fujiao Ji `[一作]` (University of Tennessee), Doowon Kim `[通讯]` (University of Tennessee)

**通讯引用:** 924 | [OpenAlex ID](https://openalex.org/A5025936242)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立 PhishGen 框架，对 14 种 LLM（商业、开源、无审查）生成的钓鱼网页进行外观真实性与端到端功能的系统评估。

**💡 创新点**

创新点在于同时量化视觉相似度和功能完整性，并与真实钓鱼套件直接对比，揭示 LLM 页面虽能渲染 96% 但仅 38% 可完成 credential‑harvesting，并挖掘 CSS 结构不平衡等生成痕迹。

**🔧 技术方法**

使用 LLM 生成 HTML/PHP，Playwright 渲染、FastDup/CLIP 计算相似度、MarkDown/Markups 评估结构、Phishpedia 检测、用户体验问卷、自动化后端功能测试等多种技术。

**📊 数据集**

使用 APWG eCX 收集的 4,409 组钓鱼套件（筛选后 64 组），以及 4 个目标品牌（Amazon、PayPal、Facebook、Wells Fargo）的登录页面、截图与 HTML 作为基准。

**📈 对比分析**

通过截图相似度、UWS 用户感知评分、品牌文本/Logo 匹配、Phishpedia 检测率、功能成功率等指标对比；最高功能成功率为 GPT 4.1 的 94.1%，整体仅 38.7%，低于真实套件的 57.8%，但有 3 个模型（GPT 4.1、Llama 4、Gemini）超过套件级别。

**⚠️ 局限性**

局限性包括：对 prompt 的敏感性、模型规模有限、仅评估生成阶段不涉及托管/域名等后端步骤，以及仅选取 4 个品牌，未覆盖全部潜在目标。

---

## 15. When Stable and Preferred Semantics Agree: A Path-Based Approach

**arXiv ID:** 2610.30176 | [PDF](https://arxiv.org/pdf/2610.30176v1)

**作者:** Yan Zhang `[一作]` (Renmin University of China), Kehan Liu `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在抽象论证框架中，优选语义和稳定语义一致的结构条件，提出了一种基于路径的通用框架。

**💡 创新点**

提出了一般路径基础标准，严格推广了文献中优选和稳定一致性的两个主要结构标准，并引入了两个无限路径集序列，分别产生了一系列一致性条件。

**🔧 技术方法**

使用了基于路径的关系和二元关系，结合攻击图中的路径集进行分析。

**📊 数据集**

应用于有限和无限的论证框架，具体数据集未明确提及。

**📈 对比分析**

通过与文献中已知的条件进行比较，证明了所提出条件的充分性，结果显示在特定路径集下，优选扩展和稳定扩展的一致性得到了保证。

**⚠️ 局限性**

局限性在于未能提供所有可能路径集的全面分析，且对路径集的选择可能影响结果的适用性。

---

## 16. Finitely Verifiable Segmentation: Decidability for Every Tokenizer with a Finite-State Verifier

**arXiv ID:** 2610.30168 | [PDF](https://arxiv.org/pdf/2610.30168v1)

**作者:** Nicklas Nidhögg `[一作]` `[通讯]` (Independent Researcher), Nicklas Nidhögg (Independent Researcher)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并研究了可由有限自动机验证的分段策略（finitely verifiable segmentation policies），证明了在给定验证器的前提下，六项关键判定问题（认证、边界间隙、可实现性、等价性、切割独立性、块独立性）可判定，并给出了闭包性、加权理论以及对最大贪心、BPE、WordPiece 等实例的实现与评测。

**💡 创新点**

核心创新在于：
1) 将分段策略转化为“长度保持”可逆的标记关系，建立起与 rational transducer 的等价性；
2) 以验证器为接口，提供统一的判定算法（含构造性 witness）并证明其在闭包操作（流水线、限制、拆分、字节回退）下保持不变；
3) 引入切割/块独立性判定，解决并行分段的安全性问题；
4) 通过加权源给出期望长度、边界计数、差异质量等精确 rational 计算，扩展了传统离散判定。

**🔧 技术方法**

使用技术包括：
- 有限自动机（FA）和状态乘积、子集构造、功能性检查；
- 文字到文字 transducer、二元并行 DFA 以及分割器、标签器的组合构造；
- 解析边界间隙的无环图、最长路径、环泵送；
- 差异质量的线性方程组求解（稀疏矩阵、几何级数收敛）；
- 直接实现的 `munch` 库以及参考扫描器、编码器。

**📊 数据集**

数据集：
- 小型实验用的所有长度 1–3 的字母表组合（474 种）；
- 规则集合编辑测试（5 种编辑集）；
- 实际文本：Twitter JSON 样本（约 19K 字节）和 BPE 训练集（JSON 示例）；
- 词汇表实例：BPE 64 合并、GPT‑2 前缀、WordPiece 词表。

**📈 对比分析**

比较方法：
- 对所有 16,116 认证判断、474 条界间隙判定（含 2 个环）、474 可实现判定、1,818 等价判定、5,517 切割独立判定、29,085 块独立判定；
- 通过可执行检查（witness 回放）与参考扫描器/编码器对比；
- 对 BPE、WordPiece 等实例在 Twitter.json 上的窗口认证、最长无认证区间、可实现掩码等进行统计；
- 性能：判定算法多项式时间，munch 库在 19,434 个窗口上仅需数秒；
- 结果显示：算法完全精确、无误判；在 BPE 追加合并实验中，精确提升约 20% 的窗口认证率，且重用加速比约 15×。

**⚠️ 局限性**

限制：
- 必须提供完整、可执行的验证器；对非功能性或非正则域的策略无法判定；
- 对于大规模词表或复杂流水线，状态爆炸可能导致可扩展性问题；
- 加权理论假设源是无记忆、可唯一确定的（unifilar、几何停止）；不适用于一般马尔可夫/语言模型；
- 仅关注“边界决定性”而非 token 质量或语义完整性；
- 现有实现未覆盖动态词典更新或多字节编码的非标准场景。

---

## 17. FusionEval: Benchmarking Model Fusion for Large Language Models

**arXiv ID:** 2610.30162 | [PDF](https://arxiv.org/pdf/2610.30162v1)

**作者:** Shuo Cai `[一作]` (Hong Kong Polytechnic University), Hongxia Yang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 7432 | [OpenAlex ID](https://openalex.org/A5082599714)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FusionEval基准，评估多域专家模型融合在推理、指令遵循和工具使用等任务中的能力保留与计算成本。

**💡 创新点**

创新点在于统一的专家池、归一化评估指标、支持训练无/有的融合方法，并聚焦长链推理与工具使用的保留问题。

**🔧 技术方法**

采用多种融合技术，包括训练无的权重平均、数据辅助融合、以及多教师自策略蒸馏（MOPD），并结合后处理校准（FeatCal、Surgery）。

**📊 数据集**

使用Qwen3与Llama基础模型的五域专家，训练数据来自OpenDataArena/ODA-Math-460k、NVIDIA OpenScienceReasoning-2、Nemotron SFT等，评测基准包括AIME、LCB、GPQA、IFEval、BFCLv3等八个公开基准。

**📈 对比分析**

通过归一化平均与最小分数、运行时和峰值显存等指标比较，发现训练无方法保留推理能力但工具使用显著下降；MOPD在所有域上获得更平衡的性能，但训练成本更高。

**⚠️ 局限性**

局限在于训练无方法无法同时保留工具使用；MOPD成本高；持续融合仍低于联合融合，缺乏高效的持续增量融合方案。

---

## 18. A Quadratic Lens on Muon's Polar Update: Benefits, Pitfalls, and the Role of Newton-Schulz

**arXiv ID:** 2610.30280 | [PDF](https://arxiv.org/pdf/2610.30280v1)

**作者:** Egor Shulgin `[一作]` (King Abdullah University of Science and Technology), Peter Richtárik `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 14101 | [OpenAlex ID](https://openalex.org/A5036598221)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Muon 中的正交化更新（PolarGD）在矩阵二次模型上进行理论分析，并在 Transformer 训练中验证其实际效果。

**💡 创新点**

提出了 PolarGD 的一阶几何性质、旋转等变性、隐式预处理机制，并阐明其在行等距迭代与秩缺失迭代中的不同表现；同时揭示有限 Newton–Schulz 步骤如何通过软化极小奇异值实现比精确极化更快收敛。

**🔧 技术方法**

采用矩阵 SVD、极化因子（polar factor）以及 Newton–Schulz 迭代实现极化更新；在理论上使用线搜索、曲率上/下界、条件数分析；在实验上使用 90M 参数 Transformer（1.8B 词元）以及多种合成二次函数。

**📊 数据集**

使用 90M Transformer 训练数据（1.8B 词元）以及标准的合成矩阵二次实验。

**📈 对比分析**

与 AdamW、标准梯度下降（GD）以及 SignGD 进行对比。实验显示：在行等距初始化或较好的迭代协方差下，PolarGD 通过隐式预处理可实现比 GD 快若干倍的收敛速度；在秩缺失或奇异值分布极端的情况下，有限 Newton–Schulz 的软化极化更新往往优于精确极化，并能超过 AdamW 的训练速度。

**⚠️ 局限性**

局部一阶优势不一定转化为全局收敛优势；在迭代过程中可能出现小奇异值导致的慢速收敛；线搜索会产生过多极小奇异值，降低预处理质量；理论分析多限于二次模型，对非二次损失的推广仍需进一步研究。

---

## 19. Breaking the high-degree barrier in Lehmer's problem with a human-AI discovery loop

**arXiv ID:** 2610.30277 | [PDF](https://arxiv.org/pdf/2610.30277v1)

**作者:** Yuri Chervonyi `[一作]` (Google DeepMind), Thang Luong `[通讯]` (California Institute of Technology)

**通讯引用:** 333 | [OpenAlex ID](https://openalex.org/A5001828232)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一套人机协同的工作流和算法，构造并验证了从 10 到 2000 阶的最小马勒测度多项式，并给出泛化的 Boyd 算法生成任意高阶最优多项式。

**💡 创新点**

将大规模云端搜索、自动符号验证与 AI 生成的搜索策略相结合，实现了可在云上并行扩展的交互式工作流；通过对高阶多项式的周期块结构进行赛义多项式因子分解，提出了五参数的 Generalized Boyd 构造，突破了传统低阶搜索的限制。

**🔧 技术方法**

使用 Gemini 大语言模型+Google Antigravity 代理环境；强化学习、进化搜索、周期缺陷铺排、周期性块搜索；符号计算与数值根求解、Irreducibility 检验；云端 GPU/CPU 分布式作业调度。

**📊 数据集**

构建并发布了 996 个偶数阶 (10–2000) 最优多项式的完整验证数据集，包含稀疏分式、展开多项式和五参数元组。

**📈 对比分析**

与已发表的 91 个低阶最优值、单参数 Boyd 族以及自行搜索方法比较，Generalized Boyd 在 99.6% 目标阶上获得最佳记录，超过 6 个低阶记录；在 824 组搜索中，周期缺陷铺排占比 54%，RL 22%，直接稀疏搜索 13%，演化 3%，优先搜索 7%；所有结果均低于 Boyd 的极限 1.255434，且收敛速度为 O(d^{-3/2})。

**⚠️ 局限性**

仅覆盖偶数阶，仍未证实 α₁ 是否为真正极限；对极小 Salem 数的生成尚无系统构造；工作流对算力和人工调优高度依赖，缺乏自动化的参数搜索与理论解释。

---

## 20. Automated screening of privacy risks in medical AI

**arXiv ID:** 2610.30229 | [PDF](https://arxiv.org/pdf/2610.30229v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 21. FlexVDR: Learning to Allocate Matryoshka Representation Budgets for Visual Document Retrieval

**arXiv ID:** 2610.30177 | [PDF](https://arxiv.org/pdf/2610.30177v1)

**作者:** Haowen Xiang `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1665 | [OpenAlex ID](https://openalex.org/A5043651092)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种框架，用于在单个检索器中进行查询自适应预算分配，以优化视觉文档检索的计算效率。

**💡 创新点**

创新点在于引入了预算分配器，能够根据早期查询特征预测特定路径的检索效用，并结合可配置的资源偏好选择合适的表示预算。

**🔧 技术方法**

使用了动态早期退出模型和联合表示家族，结合编码器深度、嵌入宽度和查询/文档令牌计数的选择。

**📊 数据集**

在ViDoRe V1、V2和V3公共数据集上进行了实验，评估了五个不同的基础模型。

**📈 对比分析**

与固定预算选择方法相比，所提出的方法在保持接近原生检索质量的同时，能够为大多数查询分配较低的预算，且在计算效率上表现更优。

**⚠️ 局限性**

限制在于模型的表现可能会受到不同基础模型的影响，且在某些情况下，较低预算的选择可能会导致检索质量的下降。

---

## 22. Accelerating Wildfire Hazard Mapping Across Canada with an AI Surrogate Modelling Framework

**arXiv ID:** 2610.30170 | [PDF](https://arxiv.org/pdf/2610.30170v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 23. Lighthouse: On Building a Trustworthy Digital Assistant for Exploring History with Civil Rights Archives

**arXiv ID:** 2610.30158 | [PDF](https://arxiv.org/pdf/2610.30158v1)

**作者:** John A. Aydin `[一作]` (University of South Carolina), Biplav Srivastava `[通讯]` (University of South Carolina)

**通讯引用:** 3330 | [OpenAlex ID](https://openalex.org/A5051577973)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款可信赖的数字助手，用自然语言查询方式探索南卡罗来纳州民权档案，融合神经网络与符号规则，实现可靠且可追溯的问答。

**💡 创新点**

创新点在于：① 将可信度与覆盖率视为核心目标，构建双源知识体系（专家验证答案与全文档案检索）；② 采用神经‑符号混合对话管理（LLM + 规则），动态评估风险并切换回复策略；③ 构建专家聚合的金标准评估数据集，为历史领域的可信问答提供标准化衡量。

**🔧 技术方法**

技术栈包括：Llama‑3.1‑8B（Fine‑tuned + LoRA 微调）、SafeChat（基于 Rasa 的规则检索）、SafeGenChat（混合对话框架）、RAG、文本相似度评估指标（Jaccard、ROUGE‑L、BLEU、BERTScore、JS‑Divergence 等）以及 NLP 预处理与 OCR 清洗流程。

**📊 数据集**

数据集：南卡罗来纳州民权档案（John Henry McCray Papers）约 19k 文档、约 6.5M tokens；专家生成的 10 个情境 × 10 个问题的问答金标准；Google Natural Questions 作为微调辅助语料。

**📈 对比分析**

评估方法：对 50 个客观问题与 10 个主观问题分别使用 6 种文本相似度指标进行自动比较；结果显示：规则模型 M2 性能最高，混合模型 M3 次之；微调 LLM M1‑1 在客观问题中优于基线 LLM，主观问题中基线 LLM M1‑2 由于输出更长、语义匹配更好而表现最好。总体说明混合方案在可信度与覆盖率之间取得了良好平衡。

**⚠️ 局限性**

局限性：① 需要专家预先标注的问答，缺乏对未知问题的探索能力；② 仅针对单一档案集，未验证跨域或更大规模数据的适用性；③ 未探讨更高级的检索、代理或多模态技术；④ 长文本生成的风格差异可能导致评估偏差。

---

## 24. When EVM Defaults Break: Three SDK Patterns for UTXO-Family Wallets

**arXiv ID:** 2610.30183 | [PDF](https://arxiv.org/pdf/2610.30183v1)

**作者:** Hanzheng Zhang `[一作]` (Common Knowledge Base Association), Hongzhou Chen `[通讯]` (Common Knowledge Base Association)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文基于 Nervos CKB 上的 TypeScript SDK（CCC），通过回顾其发布历史与 API，梳理并提炼了三种面向 UTXO 家族钱包交互的设计模式。

**💡 创新点**

创新点在于提出了 Transaction Composer、Unified Signer 抽象以及 Connector UI 三个可迁移的设计模式，解决了扩展 UTXO 与多签名方案下 SDK 的核心难题。

**🔧 技术方法**

主要技术包括 TypeScript SDK 开发、UTXO 交易构造、签名多签名抽象以及 Web 组件/React 包装的 UI 接口。

**📊 数据集**

使用的数据集为 CCC 公开发布的 18 个 npm 包、8 个钱包适配器、190 个源码文件及 13 名贡献者的历史信息。

**📈 对比分析**

比较方法是将传统 EVM SDK（如 ethers.js、viem）的三大默认假设与 CCC 的实现对比，发现后者在容量、输入选择与签名多样性上表现更优，但文中未给出量化性能指标。

**⚠️ 局限性**

局限性包括仅基于单一链（Nervos CKB）和单个 SDK 的案例研究，缺乏跨链验证与真实使用中的性能与缺陷评估。

---

## 25. Measuring undergraduate students' reliance on generative AI during problem-solving: Scale development and validation

**arXiv ID:** 2610.30195 | [PDF](https://arxiv.org/pdf/2610.30195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 26. Protection Is Not One Event: A Survey of RAG Protection Research by Measured Endpoint

**arXiv ID:** 2610.30193 | [PDF](https://arxiv.org/pdf/2610.30193v1)

**作者:** Peiyang Liu `[一作]` (Peking University), Wei Ye `[通讯]` (Peking University)

**通讯引用:** 7901 | [OpenAlex ID](https://openalex.org/A5100448683)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对检索增强生成（RAG）系统的安全与隐私防护研究进行系统综述，构建了七族端点分类法并对118篇完整论文进行双重人工编码，生成可比证据图谱；

**💡 创新点**

提出基于“测量端点”的分类法，厘清不同研究关注的真实技术事件，揭示机制与端点之间的弱关联，指出研究主要缺失内容披露端点、使用过于简单的基准和非适应性攻击；

**🔧 技术方法**

使用人工双重编码、互信息分析、实验对照测试床、三次自适应查询攻击、语义重写、检索阈值、归一化n-gram屏蔽等多种防护技术；

**📊 数据集**

对比使用的评测集包括公开问答基准（Natural Questions、HotpotQA、MS MARCO等）以及包含敏感数据的集（Enron Email、HealthCareMagic、ChatDoctor等），并在自建测试床中对192篇文档进行评测；

**📈 对比分析**

对各端点家族的防护效果进行统计（如Family A 88%成功、Family A‑pii 50%、Family D、E、F 94–96%），展示同一防护在不同端点的表现差异，证明仅消除表面复制并不能阻止事实恢复；

**⚠️ 局限性**

局限性包括：缺乏内容披露为主的端点实验、评测集中在通用问答基准、攻击模型多为单轮或非适应性、缺少置信区间与随机种子变异评估，导致对防护效果的信心过高。

---

## 27. The EU AI Act Beyond Europe: A Practical Readiness Framework for Non-EU Organisations. Comparative Evidence from Four African Countries

**arXiv ID:** 2610.30152 | [PDF](https://arxiv.org/pdf/2610.30152v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 28. Prediction Is Not Calibration: Item Parameter Recovery and Representation Coupling in Knowledge Tracing

**arXiv ID:** 2610.30211 | [PDF](https://arxiv.org/pdf/2610.30211v1)

**作者:** Wenrui Yuan `[一作]` (University of Twente), Johannes Steinrücke `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究知识追踪模型在端到端训练下的IRT参数估计准确性，分析序列编码宽度与参数读取宽度对参数恢复与预测精度的影响，并提出 SK(64,8) 结构实现更好的斜率/选项坡度估计。

**💡 创新点**

通过梯度异质性和仿射不可辨性揭示预测准确率不等价于IRT参数校准；提出分离键宽度与序列宽度的方法，以显著提升位置参数以外的斜率与选项坡度估计，且不牺牲预测性能。

**🔧 技术方法**

使用 LSTM/Transformer/DKVMN 序列编码器、2PL/GPCM/NRM 响应解码器、端到端的下一步响应交叉熵损失、梯度分析、仿射校正与能力偏移/增益估计技术。

**📊 数据集**

仿真数据（2000 名学习者、200 题、60 交互），TIMSS 2019 美国 8 年级数学（GPCM）与 EdNet‑KT1（2PL 与 NRM）数据集。

**📈 对比分析**

在 4,500+ 仿真实验与 1,420 实际实验中，比较不同宽度与编码器组合，评估 Spearman 参数恢复、离线 IRT 一致性、跨顺序一致性、准确率和交叉熵。SK(64,8) 在保持或提升准确率的同时显著提高斜率与选项坡度恢复，并降低固定顺序产生的偏差。

**⚠️ 局限性**

仅在无学习变化的单一仿真与固定/随机顺序设置下验证；离线 IRT 参考并非独立校准；训练时间仅为 120 轮；NRM 参数受非负键斜率限制。

---

## 29. V-JEPA 2 Meets AQA: Phase-Based Scoring and Reference-Based Feedback for Mountain Biking

**arXiv ID:** 2610.30160 | [PDF](https://arxiv.org/pdf/2610.30160v1)

**作者:** Santhiya Rajan `[一作]` (Multiverse Computing), Savio Cardoz `[通讯]` (ACL Digital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了基于弱标签的山地自行车跳跃动作质量评估与候选反馈，利用冻结的V-JEPA 2视频特征、2D姿态与运动量化构建分类器并输出可能的改进提示。

**💡 创新点**

创新点在于将冻结的视觉特征与简易姿态/运动描述相结合，提出无监督的参考分布标准化提示生成机制，并对弱标签环境下的评估流程进行系统阐述。

**🔧 技术方法**

使用V-JEPA 2 ViT‑L冻结编码器提取视频特征，YOLO‑Pose/视频姿态检测得到2D关键点，随后通过Transformer混合权重评分头进行二分类；参考分布对各参数做z‑score标准化后按规则生成候选提示。

**📊 数据集**

采用95段现场录制的山地自行车跳跃视频（52段好技术，43段差技术）为训练集，并使用17段外部探针视频进行独立验证。

**📈 对比分析**

采用五折clip‑level交叉验证，报告AUROC 0.768、AP 0.812；去除生物力学输入反而提升AUROC至0.809；外部探针的准确率仅为0.412，提示精确率0.118，显示内部评估有一定区分力但泛化性能不佳。

**⚠️ 局限性**

局限包括：标签仅为二分类且可能受视角、骑手、录制环境偏倚；缺乏骑手/场景/相机标识，无法评估跨骑手/地点的泛化；内部epoch选择与训练集不一致导致过拟合；提示生成为无监督规则，精度低；外部验证效果差。

---

## 30. TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents

**arXiv ID:** 2610.30161 | [PDF](https://arxiv.org/pdf/2610.30161v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 31. Parameters efficient language model with latent cross-attention for IoT intrusion detection

**arXiv ID:** 2610.30222 | [PDF](https://arxiv.org/pdf/2610.30222v1)

**作者:** Frederic Adjewa `[一作]` (University of Technology of Troyes), Cheikh Kacfah `[通讯]` (Institut Universitaire Saint Jean du Cameroun)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种轻量级的Transformer架构，利用交叉注意力将输入压缩到少量潜在向量，再在低维潜在空间中使用单层自注意力进行IoT流量的入侵检测。

**💡 创新点**

创新点在于通过交叉注意力压缩和参数共享的单层Transformer，显著减少模型参数（≈1.6M）和体积（≈6MB），同时保持与大型BERT相近的检测精度，且推理速度提升近两百倍。

**🔧 技术方法**

采用的技术包括交叉注意力压缩、单头自注意力、参数共享/堆叠Transformer层、RMSNorm、基于词级tokenization的PPFLE隐私编码。

**📊 数据集**

使用了Edge-IIoTset和CICIoT2023这两个IoT入侵检测基准数据集。

**📈 对比分析**

通过与SecurityBERT以及多种传统机器学习和Boosting基线在宏F1、准确率、误报率等指标上对比，结果显示在Edge-IIoTset上宏F1达98.06%（无PPFLE）或94.55%（PPFLE），与SecurityBERT相近或更优，同时模型参数约7倍减少，推理延迟约89倍提升。

**⚠️ 局限性**

局限性包括训练仅进行10个epoch，深层模型未收敛；对某些稀有攻击类别检测仍不理想；仅在模拟边缘设备上评估推理性能，缺乏真实硬件验证；仅评估两数据集，泛化性需进一步验证。

---

## 32. GANet: A Lightweight Group Attention Network for Plant Disease Identification on Embedded Systems

**arXiv ID:** 2610.30225 | [PDF](https://arxiv.org/pdf/2610.30225v1)

**作者:** Anubhab Tripathi `[一作]` (Jadavpur University), Sheli Shinha Chaudhuri `[通讯]` (Jadavpur University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出轻量级的 Group Equivariance Attention Network（GANet）用于植物叶片病害分类

**💡 创新点**

创新点是将 p4convp4 / p4convz2 组卷积、Group Attention Block 与 Feature Fusion Block 结合，实现参数极低（0.001M）且高精度

**🔧 技术方法**

采用组卷积、注意力机制、特征融合、Softmax 分类器等深度学习技术

**📊 数据集**

使用 PlantVillage 数据集，涵盖 38 类叶片健康/病害图像

**📈 对比分析**

与 VGG、ResNet、MobileNet 等现有模型对比，准确率达 99.99%，参数最少，在 Raspberry Pi 4b 和 Jetson Nano 上分别实现 5 FPS 与 11 FPS

**⚠️ 局限性**

局限在于仅在 PlantVillage 进行验证，缺乏在现场环境或多作物数据集上的测试，对光照、背景变化的鲁棒性待进一步验证

---

## 33. Association between Player Perception and AI Disclosure on Steam: An Exploratory Study

**arXiv ID:** 2610.30178 | [PDF](https://arxiv.org/pdf/2610.30178v1)

**作者:** Fabian Wiesinger `[一作]` (Graz University of Technology), Denis Helic `[通讯]` (Graz University of Technology)

**通讯引用:** 1823 | [OpenAlex ID](https://openalex.org/A5074698656)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析了Steam游戏评论，探讨AI披露与玩家评论之间的关联。

**💡 创新点**

首次将AI披露标签与评论文本进行大规模关联分析，揭示披露后负面评论中AI相关批评上升。

**🔧 技术方法**

采用关键词计数、词频、特征词统计、LIWC情感分析及Bootstrap置信区间等自然语言处理与统计方法。

**📊 数据集**

收集了602款游戏共50,153条评论，其中6,601条负面评论和6,153条正面评论，分为AI标记与未标记两组。

**📈 对比分析**

通过对比AI标记与未标记组的关键词出现率、词频差异、LIWC情绪得分，发现AI标记组负面评论中AI关键词出现率提高约5个百分点，情绪语调更负面，整体差异显著。

**⚠️ 局限性**

受限于关键词覆盖不足、评论样本偏倚、未控制游戏质量与玩家先验AI认知等，结论仅为相关性而非因果。

---

## 34. A Consolidated Deployment Edition of a Microservice Core-Banking Platform: Composing Services by Reference into Three Hosts

**arXiv ID:** 2610.30212 | [PDF](https://arxiv.org/pdf/2610.30212v1)

**作者:** Tomisin Fagbola `[一作]` `[通讯]`, Tomisin Fagbola

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种通过引用（不复制代码）将五个核心银行 API 服务合并到单一 API 主机（再加消费者主机和后台主机）的集成部署版，并与原始九服务版在相同代码基线下进行对比。

**💡 创新点**

创新点包括：① 通过引用方式实现微服务组合，保持各服务独立仓库与发布；② 在同一平台下同时运行两种部署版，提供可比实验；③ 设计了一套覆盖代码检查、启动检查、路由一致性、故障注入和行为差分测试的完整评估框架。

**🔧 技术方法**

使用的技术与工具包括：.NET 10/.NET 6（ASP.NET Core MVC、EF Core）、Ocelot 网关、ECS Fargate、PostgreSQL、MongoDB、Redis、AWS SNS/SQS、Secrets Manager、MOTO（服务模拟）、Docker、C# 脚本与PowerShell、负载测试与性能测量脚本、以及自定义的配置层与模块注册机制。

**📊 数据集**

使用的数据集主要为：47 个业务工作负载案例（包含事务、账户、限额、角色与权限等操作）、1,155 条无参数 GET 读取探测请求（覆盖所有公开 API 路径）、11 个跨租户与写授权验证案例，以及数据库中生成的交易、账单、账户余额等记录用于账本一致性检查。

**📈 对比分析**

比较方法：在模拟的 AWS 部署环境中，使用相同的代码提交、任务规格与配置，对两种部署版执行相同的工作负载、读取探测与跨租户/写授权案例，并通过差分测试比对返回值、账本一致性、读写吞吐量与内存占用。实验结果显示：路由一致性 100%，行为一致性（工作负载、读取、跨租户授权）100%，读取吞吐量提升 2.3 倍，内存占用下降 50%，成本估算（CPU/内存）下降 11–24%。

**⚠️ 局限性**

limitations: 1) 实验基于模拟 AWS 环境，未测量真实云网络延迟、弹性伸缩与实际计费；2) 仅覆盖了已定义的工作负载与读取路径，未检验所有业务场景（如批量导入、事务回滚错误、复杂路由规则等）；3) 故障注入集是作者设计，可能未覆盖所有现实故障类型；4) 代码与敏感配置信息未公开，结果复现受限；5) 只评估了一次平台版本，未考察多版本迭代对组合方式的影响。

---

## 35. QProtFuse: Query-Guided Layer Fusion for Retrieval-Augmented Protein Function Prediction

**arXiv ID:** 2610.30180 | [PDF](https://arxiv.org/pdf/2610.30180v1)

**作者:** Mianzhi Pan `[一作]` (Nanjing University), Jianbing Zhang `[通讯]` (Nanjing University)

**通讯引用:** 13665 | [OpenAlex ID](https://openalex.org/A5049207163)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于查询引导的多层蛋白质语言模型特征融合框架QProtFuse，用于改进蛋白质检索和功能描述的检索增强生成(RAG)。

**💡 创新点**

创新点在于：①发现不同PLM层在蛋白检索中的表现差异，传统仅使用最终层或全层平均均不理想；②通过查询指令动态分配层级权重，实现针对不同功能问题的自适应特征融合；③联合使用蛋白-蛋白与蛋白-注释的对比学习目标，互相强化，提高检索质量。

**🔧 技术方法**

采用冻结的ESM-2-650M蛋白编码器、OpenAI的文本编码器、MLP权重路由器，结合SigLIP对比损失和温度/偏置可学习的跨模态/单模态对比目标；实现层分组、查询条件加权、L2归一化等技术。

**📊 数据集**

主要使用Prot-Inst-OOD基准数据集，包含功能、催化活性、域/基序和通用功能四个子任务，训练集28.4k条记录，测试集14.5k条查询。

**📈 对比分析**

与RAPM、ProtQueSt等现有RAG方法以及ESM全层平均、MMseqs2等检索基线比较。QProtFuse在所有检索指标上均优于对比方法，在RAG生成的ROUGE-L、Entity-BLEU和LLM-as-a-judge分数上也表现最优（如Qwen3-8B LLM judge分数6.22/10，DeepSeek-V4-Pro 6.68/10）。

**⚠️ 局限性**

限制主要在于：评估规模受限于Prot-Inst-OOD；检索与生成被分离，未实现检索-生成的联合优化或证据验证；对更大规模数据库如UniProt的可扩展性尚未验证。

---

## 36. When Reconstruction Is Not Enough: Doubly Robust Counterfactual and Interventional Mediation Analysis with Learned Multimodal Representations

**arXiv ID:** 2610.30220 | [PDF](https://arxiv.org/pdf/2610.30220v1)

**作者:** Cong Cao `[一作]` (Yale University), Cong Cao `[通讯]` (Yale University)

**通讯引用:** 23 | [OpenAlex ID](https://openalex.org/A5100784656)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了在高维多模态健康记录中使用学习表征进行因果推断时，表征可能导致识别失效的风险，并提出诊断与工作流程；

**💡 创新点**

证明仅靠协方差（如重构误差）无法保证表征足够，给出充分与必要条件，揭示跨拟合与双重鲁棒性无法消除表征偏差，并提供实用诊断与表征选择规则；

**🔧 技术方法**

使用变分自编码器（VAE）、交叉拟合、双重鲁棒/一阶估计器、机制驱动目标、诊断统计量（D1‑D4），以及对中介分析的交叉拟合一阶估计器；

**📊 数据集**

采用模拟数据生成的多模态健康记录（诊断、实验室、调查等），含可缺失的罕见混杂变量，并在不同训练策略（外部、折叠、全样本）下评估；

**📈 对比分析**

通过200次仿真对比多种方法（原始协变量、VAE、PCA、加入关键协变量等），在中介效应估计中发现表征不充分时覆盖率从95%下降至0%，而平均处理效应受影响较小；在不同表征维度、重构误差与偏差无关；训练策略对结果影响不大；

**⚠️ 局限性**

仅考虑已测量的混杂变量，未解决未测量混杂；偏差理论仅在线性高斯模型下给出；交叉拟合需稳定性假设；中介估计仅适用于单一二元中间混杂；数字双生等更复杂机制未充分验证。

---

## 37. MPIDetector: Detecting Multi-modal Prompt Injection via LLM-based Reasoning

**arXiv ID:** 2610.30202 | [PDF](https://arxiv.org/pdf/2610.30202v1)

**作者:** Yiheng Li `[一作]` (Paul G. Allen School of Computer Science & Engineering University of Washington), Jingyue Li `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 4670 | [OpenAlex ID](https://openalex.org/A5067021027)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MPIDetector框架，利用LLM的推理能力通过少量示例的上下文学习来检测多模态Prompt Injection，覆盖文本、视觉和环境注入。

**💡 创新点**

创新点在于将LLM的内在推理能力转化为安全检测器，并通过视觉-语义解耦（Visual‑Semantic Decoupling）显式分离图像噪声与语义内容，显著提升对高度隐蔽环境注入的检测率。

**🔧 技术方法**

使用了多模态大型语言模型（Gemini‑2.0‑Flash、Qwen3‑VL‑32B‑Instruct、Doubao‑seed‑1.6‑flash）和少量示例的In‑Context Learning，配合自定义提示模板进行推理。

**📊 数据集**

数据集包括WAInjectBench（文本与视觉恶意样本）、Yoga‑82与HuggingFace公开的视觉好样本、500个高阶WebInject环境注入样本，总计3,902条样本。

**📈 对比分析**

与基线（基于困惑度的统计检测、Fine‑tuned PromptShield、规则+OCR、SnapGuard）比较，MPIDetector在文本上准确率达90.91%并实现100%召回；在视觉上准确率99.28%且100%精确度；在WebInject上检测率高达99.80%，同时保持0%误报。

**⚠️ 局限性**

主要限制是高计算成本与推理延迟，尤其对实时大规模应用不友好，需要进一步压缩模型或蒸馏为更轻量级版本。

---

## 38. Benchmarking Model Compression Techniques Across Heterogeneous Edge Platforms: A Predictive Maintenance Use Case for Rotating Machinery

**arXiv ID:** 2610.30228 | [PDF](https://arxiv.org/pdf/2610.30228v1)

**作者:** Matheus Calheira Guimarães de Oliveira `[一作]` (SENAI CIMATEC University), Erick G. Sperandio Nascimento `[通讯]` (SENAI CIMATEC University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对高性能卷积神经网络进行压缩并在多种工业边缘设备上部署，用于旋转机械的预测性维护。

**💡 创新点**

首次在同一模型上系统比较四种压缩技术，并提出知识蒸馏压缩的学生网络D-PdMCNN，实现高达98.9%的压缩率和90%以上的速度提升。

**🔧 技术方法**

采用8位整数量化、动态范围量化、Float16量化和知识蒸馏四种技术，并使用TensorFlow Lite转换为flatbuffer格式。

**📊 数据集**

使用MaFaulDa旋转机械故障数据库，包含五类故障。

**📈 对比分析**

在九种不同CPU/GPU/微控制器上对压缩模型进行推理时间、内存占用和分类指标比较，KD压缩在保持1.4% F1降幅的同时实现了平均推理时间下降至约1/100的效果。

**⚠️ 局限性**

仅评估了单一数据集，未测量实际能耗，且某些压缩方法在极低功耗硬件上无法运行，限制了普适性。

---

## 39. Spectral Uncertainty Decomposition for Sparse Bayesian Learning

**arXiv ID:** 2610.30207 | [PDF](https://arxiv.org/pdf/2610.30207v1)

**作者:** Ismail Huseynov `[一作]` (Physikalisch-Technische Bundesanstalt), Stefan Haufe `[通讯]` (Physikalisch-Technische Bundesanstalt)

**通讯引用:** 8279 | [OpenAlex ID](https://openalex.org/A5068256213)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出基于先验预条件化前向矩阵奇异值分解的后验协方差谱分解，将后验不确定性拆分为可观测数据驱动部分与不可观测的零空间部分，进而实现对M/EEG源估计不确定性的可解释量化。

**💡 创新点**

创新点在于给出后验协方差的精确加法谱分解及其等价的先验减去数据减量式，首次把后验协方差在可观测子空间与不可观测子空间之间做完整且可量化的分解，并明确说明先验、数据与噪声在源空间不确定性中的不同贡献。

**🔧 技术方法**

使用了贝叶斯逆问题框架中的Type‑II学习（经验贝叶斯），先验协方差的平方根与前向矩阵构造预条件化算子，利用其奇异值分解和谱分析得到后验协方差的解析表达式，并用Woodbury恒等式和分辨率算子处理先验半正定的情况。

**📊 数据集**

实验采用模拟EEG数据，左半球100个可变方向（正交限制）电偶极子、64个传感器、1000个时间点、10个活跃源，通过引入高SNR噪声（≈39.9 dB）生成观测。

**📈 对比分析**

通过对比后验协方差的加法谱分解与先验减去数据减量式的可视化与量化，展示了数据在可观测模式下显著压缩先验方差，且两种分解在数值上严格满足等式，证明了理论推导的正确性；实验未与其他不确定性量化方法直接竞争，但结果说明谱分解能够清晰识别可约与不可约不确定性。

**⚠️ 局限性**

局限性包括：在经验贝叶斯稀疏化过程中，部分先验方差趋零导致后验方差为零，可能产生过度自信；缺乏对超参数不确定性的考虑与校准；方法仅适用于线性高斯模型，难以推广到非线性或非高斯情形。

---

## 40. MoEZip: Routing-Aware KV Cache Compression for Sparse Mixture-of-Experts LLMs

**arXiv ID:** 2610.30206 | [PDF](https://arxiv.org/pdf/2610.30206v1)

**作者:** Minsang Kim `[一作]` (Korea University), Seungjun Baek `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MoEZip，一种针对稀疏 Mixture‑of‑Experts 语言模型的 KV 缓存压缩方法，能够在大规模长上下文推理中保持模型性能。

**💡 创新点**

创新点在于引入基于 Fisher 信息矩阵的路由脆弱性度量，结合指令‑到‑上下文注意力来标记重要 KV 条目，并在压缩时保留这些“锚点”，从而避免 KV 置换导致路由错误。

**🔧 技术方法**

技术包括：1) 软最大/ Sigmoid 路由的 Fisher‑trace 计算；2) 路由脆弱性评分与注意力加权；3) 动态预算分配；4) 与现有压缩基准（KVZip、SnapKV、PyramidKV）组合；5) 可重用缓存的查询无关变体。

**📊 数据集**

使用的评测数据集包括：LongBench（12 任务）、RULER、NIAH，涵盖问答、摘要、检索、代码补全等多种长上下文任务。

**📈 对比分析**

与 KVZip、SnapKV、PyramidKV 等基准在不同 KV 保留比例（0.1–0.9）下对比。MoEZip 在 10% KV 保留率时平均提升 LongBench 约 +24/20 分，分别比 SoTA 提高 2.3×和 4.9×；在 NIAH、RULER 等任务上也表现出更稳健的压缩性能。

**⚠️ 局限性**

局限性：需要访问模型内部路由输出与注意力统计，无法在纯黑盒推理环境下使用；并且在极低预算下仍可能出现路由错误导致性能下降。

---

## 41. Feedback That Backfires in Games: Do LLM Players Repeat the Move They Just Watched Fail?

**arXiv ID:** 2610.30205 | [PDF](https://arxiv.org/pdf/2610.30205v1)

**作者:** Esmail Gumaan `[一作]` (University of Passau), Esmail Gumaan `[通讯]` (University of Passau)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5119891973)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在游戏（Sokoban、Connect Four、Othello）中研究LLM代理在出现失误后将失误记录到对话中所产生的概率效应，探索失误记录是否会导致代理重复同一失败动作，评估不同记录形式、训练阶段和提示格式对该效应的影响，并验证原论文提出的若干对策。

**💡 创新点**

①发现失误记录往往通过将失误动作的表面形式写入上下文，导致该动作的后验概率趋于“平坦”，从而在非法动作上反而更可能被再次选择；②揭示此现象源自预训练的复制机制，而非仅靠指令微调；③阐明不同提示格式（原始、RAGEN、自定义）和记录形式（原文、抽象描述、清空重试等）对效果的差异，提供更有效的记录策略。

**🔧 技术方法**

利用大规模语言模型（SmolLM、Qwen、OLMo 2等）的指令调优和后训练（SFT、DPO、RLVR等）进行实验；使用对数概率差异（corrective gain）和贪心解码的重复率作为衡量指标；对不同符号表述（短码、完整短语、索引）进行混合效应回归分析。

**📊 数据集**

实验数据来自三款无随机元素的游戏：Sokoban（RAGEN生成）、Connect Four（自我对弈产生的状态）、Othello（自我对弈产生的状态）。每种游戏分别构造了约 20–40 个决策点，包含非法动作、合法但输赢、以及模型自认为最佳的错误动作。

**📈 对比分析**

通过对比不同模型、不同训练阶段、不同记录方式和不同提示格式下的 corrective gain（以 nats/token 计）以及贪心解码重复率，验证：①原始记录导致负 corrective gain（即更可能重复），②抽象描述能显著降低此效应但对部分模型效果不佳；②在 RAGEN 的默认格式下，非法动作的反向效应尤为显著；③在 OLMo 2 后训练阶梯中，DPO 阶段能大幅消除负 corrective gain。性能表现：在大多数模型上，原始记录导致平均 -0.1~ -0.2 nats/token，抽象描述约 0~0.1 nats/token；贪心重复率从 30% 降至 10% 左右。

**⚠️ 局限性**

①模型规模有限（最多 1B 参数），未评估更大模型；②未完成 RL 训练实验，无法验证 RL 是否能消除该效应；③自由游戏实验样本不足，且代理整体表现低，无法充分检验实战效果；④实验样本主要为扰动生成的固定项，真实游戏中错误分布可能不同；⑤对成功消息的构造在某些情形下不够一致。

---

## 42. Detection Is Not Enough: Reasoning Traces as Testable Program Hypotheses

**arXiv ID:** 2610.30200 | [PDF](https://arxiv.org/pdf/2610.30200v1)

**作者:** Vaidehi Hegde `[一作]` `[通讯]`, Vaidehi Hegde

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在不使用参考答案、判别器模型或权重更新的前提下，探讨如何通过对大型语言模型（LLaMA‑2‑7B‑chat）生成的推理轨迹进行验证，以判断其答案是否正确。

**💡 创新点**

创新点在于提出了“候选导出的变形测试”（metamorphic probe），该方法将推理轨迹抽象为程序，在对问题进行一次量化扰动后，用模型重新生成答案，比较两者一致性，从而在没有外部答案的情况下检验轨迹的可靠性；并对传统内部一致性检测与该方法在不同错误类型上的互补性进行了系统分析。

**🔧 技术方法**

使用的技术包括：①基于正则表达式与Sympy的轨迹解析与方程链抽取；②构造静态检测器（利用方程有效性、数值重叠、目标达成、语义连贯性等特征，结合逻辑回归和梯度提升模型进行判别）；③构造候选导出变形测试（将轨迹抽象为线性程序，扰动问题量并与模型答案比较）；④基于GroupKFold的交叉验证与McNemar/Fisher检验进行评估。

**📊 数据集**

使用公开的 GSM8K（数学推理）数据集，对其中 1,319 个测试题目进行实验；训练集用于构造静态检测器，验证集用于校准。

**📈 对比分析**

与传统的内部一致性检测相比，静态检测器在 67.1% 的错误中无法发现错误，AUROC 仅 0.741；而变形测试在可测试的 71% 题目上能支持 60.6% 的正确轨迹，召回 83.6% 的错误轨迹，但错误率仍高。两者结合后，检测 AUROC 略升至 0.778，但在真实部署中，通过“检测‑>重试”策略未能显著提升整体准确率（从 27.82% 提升至 28.96%，差异无显著性）。

**⚠️ 局限性**

局限性包括：①仅在单一模型（LLaMA‑2‑7B‑chat）和单一数据集上验证，结果可能不具普适性；②模型错误多为全局错误（无可定位的错误方程），导致检测器与修复率低；③变形测试仅能覆盖 71% 的题目，且对模型在扰动问题上的答案可靠性要求较高，导致误报；④未与自一致性、判别器模型等其他验证方法做实证比较。

---

## 43. Self-Evolving Static Analysis via LLM Pseudo-Labeling

**arXiv ID:** 2610.30184 | [PDF](https://arxiv.org/pdf/2610.30184v1)

**作者:** Rui Melo `[一作]` (Carnegie Mellon University), Carolyn Rose `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12908 | [OpenAlex ID](https://openalex.org/A5089539629)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一套 LLM 代理与评判器的循环，能够从零规则开始、在无标注代码上迭代进化可读、可审计的静态分析查询，最终生成可直接在项目中使用的检测规则。

**💡 创新点**

创新点在于：① 通过“伪标签”而非真实漏洞标签驱动规则演化；② 结合代码属性图（CPG）与可读 Scala 查询，让模型既能生成又能维护规则；③ 设计了基于 FP、FN 票据的反馈机制，使模型能有选择地修复误报或补全漏报，避免全局性能退化。

**🔧 技术方法**

使用的技术包括：大语言模型（Agent 为 LLM，Critic 为更强 LLM），代码属性图（CPG）作为查询语法与执行环境，进化式算法（Mutation + Generation），伪标签评估函数（估计精度、类型覆盖率、误报/漏报票据），以及工具调用与版本控制交互的 LLM Agent 框架。

**📊 数据集**

数据集：① OWASP Benchmark for Python（1,000+ 测试用例，覆盖 10+ 漏洞类别）；② 12 个真实 Python 开源项目的 CVE 引入前后快照，用于“引入时检测”基准（共 35 条 CVE）。

**📈 对比分析**

比较方法：将进化规则集与五个主流静态分析工具（CodeQL、Semgrep、Bandit、Bearer、OWASP ZAP）在 OWASP Benchmark 上的 F1 进行对比。进化规则集平均 F1 0.66，显著高于工具默认配置，且与 CodeQL 在访问标签后调优得到的 0.64 接近；在引入时检测中，进化规则集在 12 个项目中共捕获 11 条 CVE，平均召回率约 31%。

**⚠️ 局限性**

局限性包括：仅评估 Python，未验证跨语言可迁移性；依赖 LLM 的推理质量，评判器偏差可能导致误修复；演化过程成本高（数千 token 消耗）；评估基准为合成数据，真实项目的复杂度和规模尚未充分验证；票据机制对置信度的估计依赖 LLM，未完全保证其准确性。

---

## 44. Detecting AI Training in Datacentre Power

**arXiv ID:** 2610.30182 | [PDF](https://arxiv.org/pdf/2610.30182v1)

**作者:** Tom Kimpson `[一作]` (University of Melbourne), Emlyn Graham `[通讯]` (Machine Alignment, Transparency and Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究一种基于设施级功率计的被动检测方法，能在不信任算力运营商的前提下识别大模型训练作业。

**💡 创新点**

创新点在于将Viterbi行踪算法应用于功率频谱，跟踪漂移的训练迭代周期，并评估隐藏策略的成本，从而证明隐藏训练既昂贵又难以完全规避。

**🔧 技术方法**

主要技术包括功率信号的短时傅里叶谱、对比度映射、Viterbi线性隐藏马尔可夫模型跟踪、以及固定频率匹配滤波和多窗F检验的基准对比。

**📊 数据集**

使用模拟的多GPU训练与推理功率模型（基于已公开的功率参数）生成合成数据，涵盖不同的工作负载和隐藏策略。

**📈 对比分析**

在模拟数据上，Viterbi检测在训练周期漂移最高达1.5 Hz时仍保持≈96%准确率（与最佳可达率相差0.04），远优于固定频率基线（≤23%）；检测阈值基于10⁻²的误报率设定。

**⚠️ 局限性**

主要局限包括：仅使用合成数据，未验证真实设施的功率特征；对隐藏策略的覆盖有限，可能存在未考虑的规避手段；以及对功率平滑（能量存储）等更复杂的防护方法未做深入评估。

---

## 45. Neural Networks with Differentiable Belief-Function Layers

**arXiv ID:** 2610.30173 | [PDF](https://arxiv.org/pdf/2610.30173v1)

**作者:** Antonin Sulc `[一作]` `[通讯]`, Antonin Sulc

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了可微分的基于Dempster–Shafer理论的神经网络，通过Yager组合规则生成可读的无知质量(m(Θ))，并将其作为模型的不确定性与OOD检测信号。

**💡 创新点**

发现Yager规则既满足DST对冲突处理的哲学，又消除了Dempster正则化导致的梯度爆炸，提出了“Mass-Primitive Evidential Neural Network”，并证明其单向可微、无关联性，令m(Θ)自然对应人类不确定性。

**🔧 技术方法**

使用ResNet-18编码器、softplus证据头、可学习的α-折扣、Yager组合、Jøsang vacuity读取，配合梯度裁剪、Adam优化和Cosine LR，在单向前向推理的DST管线中实现了可微分融合。

**📊 数据集**

在CIFAR-10（含近OOD CIFAR-100）、ChaosNLI文本、CIFAR-10H/ChaosNLI的人类标注数据集，以及SVHN、DTD、MNIST、Fashion‑MNIST、Gaussian/Uniform噪声等OOD数据集上进行实验。

**📈 对比分析**

与Softmax+MSP/Energy、EDL、Deep Ensemble、SNGP、DUQ、MC Dropout等基线比较；ID准确率相近的情况下，Yager版的m(Θ)在近OOD/远OOD上与能量/EDL相当，Dempster版在clamp下性能最好但梯度不稳定；人类不确定性匹配度显著高于Dempster。

**⚠️ 局限性**

局限性包括仅使用{singleton,Θ}的有限focal集合（无法分配给子集）、Yager组合非结合导致的顺序敏感（但影响小）、α折扣学习效果有限、任务损失仅条件正确且不具全局严格性，以及未在真正独立源上验证。

---

## 46. The Cost of Saying Hello: An Empirical Study of Handshake Overhead in AI Agent Communication Protocols

**arXiv ID:** 2610.30151 | [PDF](https://arxiv.org/pdf/2610.30151v1)

**作者:** Pranav Varshney `[一作]` `[通讯]` (University of Michigan), Pranav Varshney (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

测量并比较了三大 AI 代理通信协议（MCP、A2A、ACP）的握手过程，细分为发现、能力交换、会话、就绪四个阶段，并在冷启动、热重连、能力不匹配和降级条件下记录时延、往返次数、字节和令牌成本。

**💡 创新点**

首次对代理握手进行阶段化实验基准，并公开完整的基准 harness、原始日志和分析脚本，揭示往返次数是握手成本的主导因素，并量化现代 MCP 通过单一探测实现的往返次数下降。

**🔧 技术方法**

使用官方 Python SDK、HTTP/JSON-RPC 代理服务器、用户空间 RTT 延迟模拟器、请求/响应记录器，以及 OpenAI 的 tiktoken 进行令牌计数，所有实验在 Apple Silicon 机器上循环回环完成。

**📊 数据集**

使用自定义的极简对等代理，只暴露单一 “echo” 能力，配合在协议规范内定义的“Agent Card”或“tool list”，并通过重复多次握手产生的原始日志作为实验数据。

**📈 对比分析**

通过测量每个场景下的往返次数、任务就绪延迟（ms）、TCP 连接数、字节数和令牌成本，结果显示冷启动时 legacy MCP 需要 3 次往返，modern MCP 2 次，A2A/ACP 1 次；热重连时 A2A/ACP 与 modern MCP 可零往返；能力不匹配时必须完成完整握手；在 50 ms RTT 下往返数决定总延迟，且令牌成本随能力数量线性增长。

**⚠️ 局限性**

实验使用了极简代理、循环回环网络、单一 SDK 版本和用户空间 RTT 模拟，未覆盖真实网络的传输层延迟、OAuth 认证、多能力规模和自然语言协商等情况，令牌计数基于单一分词器且仅给出下限。

---

## 47. Outage Analysis and Bilevel Resource Optimization in RIS-Assisted Clustered UAV-NOMA Networks

**arXiv ID:** 2610.30410 | [PDF](https://arxiv.org/pdf/2610.30410v1)

**作者:** Masoud Ghazikor `[一作]`, Morteza Hashemi `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

本文研究了在不完整 SIC 条件下，利用可重构智能表面（RIS）辅助的聚类式基站-UAV 直流 NOMA 通信网络，并提出了从 UAV 聚类到资源分配的完整优化框架；

**💡 创新点**

创新点包括①基于 RIS 接近度与信道差异的 Greedy RIS-Aware（GRIS）聚类算法，②将效率（平均吞吐）与可靠性（最大 UAV 容错概率）相结合的双层优化模型，③提出的 IOS‑I2S 两层迭代求解框架，能够在保持吞吐的同时显著降低极限失效率；

**🔧 技术方法**

采用了 Nakagami‑m 与双重 Nakagami‑m 信道模型、SNR 与误码分析、Kuhn–Munkres Hungarian 算法（RIS 分配）、梯度法与有限差分法（IOS）以及网格搜索与逐步搜索（I2S）等技术；

**📊 数据集**

使用仿真数据：在半径 2 km 的单元格中，10 台 UAV 与 3 台 RIS 随机分布（泊松点过程），每台 RIS 12×12 元素；仿真参数如 BS 发射功率 10 dBm、噪声功率 −104 dBm、目标速率 2 bps/Hz 等；

**📈 对比分析**

与 OMA、无 RIS、等资源分配、仅效率优化和仅可靠性优化等基线方案比较。结果显示 IOS‑I2S 在平均吞吐上提高 12–16 %（相对 ERA），最大失效率降低 77–81 %；同时实现了介于纯效率和纯可靠性之间的折中点；

**⚠️ 局限性**

主要局限包括：1）聚类与资源分配均基于静态 UAV 位置信息，未考虑 UAV 动态移动；2）仿真规模较小，未验证大规模部署下的计算复杂度；3）假设完美的 RIS 相位调节与完整的 CSI，实际实现可能受限于硬件与估计误差。

---

## 48. Cambium: Participant-Built Commitment Roots for Shared Digital Infrastructure

**arXiv ID:** 2610.30409 | [PDF](https://arxiv.org/pdf/2610.30409v1)

**作者:** Mansoor "Manny" Ahmed `[一作]` (OpenOrigins Inc), Ekaterina Pavlova `[通讯]` (Independent researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了分布式参与者共同构建周期性承诺根（Cambium）协议，消除了聚合者对根内容的控制。

**💡 创新点**

创新点在于：去中心化根构建、递归双倍合并与Banyan trie实现的动态委员会重排、以及可证明的根相对证明机制。

**🔧 技术方法**

使用技术包括递归双倍合并、Banyan trie、PBFT式协商、Merkle树、签名与哈希、Go语言实现。

**📊 数据集**

使用的数据集主要是 AWS ECS 环境中的模拟节点集合（n=8–256、c=4、8、16、32、64、128、256），并通过随机丢失与宕机模拟来评估鲁棒性。

**📈 对比分析**

与传统聚合器/区块链对比，消息量为 O(c·log(n/c))，周期时间 12–148 ms，验证时间 <1.5 ms，证明大小与根高度呈对数增长；实验显示与日志规模呈指数级增长一致。

**⚠️ 局限性**

主要限制包括：缺失填充与分区恢复未完成、对大量 Byzantine 节点的容忍度有限、成员管理与治理机制尚不完善。

---

## 49. HAWC-Leak: Heterogeneous Architecture Workload Classification utilizing Side-Channel Leakage

**arXiv ID:** 2610.30400 | [PDF](https://arxiv.org/pdf/2610.30400v1)

**作者:** Muhtasim Alam Chowdhury `[一作]` (University of Arizona), Soheil Salehi `[通讯]` (University of Arizona)

**通讯引用:** 768 | [OpenAlex ID](https://openalex.org/A5016438213)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了一个名为HAWC‑Leak的框架，利用非侵入式功耗侧信道对异构系统（CPU+FPGA）中的多种硬件加速器（AES‑128、ECC‑P‑256、SHA‑256、CNN）进行实时工作负载分类；

**💡 创新点**

创新点包括：①把侧信道功耗用于多工作负载异构平台的实时识别；②提出双阶段层级随机森林模型，首阶段滑动窗口专门检测短时burst的AES，第二阶段多分类识别ECC、SHA、CNN及空闲；③在ChipWhisperer+Artix‑7平台上实现完整的高分辨率功耗采集并展示100%分类准确率；

**🔧 技术方法**

使用的技术有：ChipWhisperer同步功耗采集；Xilinx Artix‑7 FPGA实现多种加速器；层级随机森林机器学习；时域与频域统计特征提取；滑动窗口检测；Python脚本自动化实验；

**📊 数据集**

数据集：每种工作负载约10,000条功耗波形（AES、ECC、SHA‑256、CNN、Idle），每类包含10组固定密钥/数据，另外在共存的多加速器位图下收集混合模式波形；每类使用10,000条训练，100条独立测试；

**📈 对比分析**

实验与传统阈值/单阶段分类对比，传统方法在背景噪声下失效；层级RF在所有测试集（单独与混合模式）均实现100%准确率，显示出优秀的实时识别性能，且模型轻量化可在FPGA上实时推理；

**⚠️ 局限性**

局限性：仅能识别单一激活加速器；模型可能在真实动态环境中过拟合训练噪声；未覆盖多任务并发场景；缺乏无监督或轻量化CNN等更鲁棒方法；未来需扩大工作负载种类与混合情境以提升泛化能力。

---

## 50. Rank Ceiling for Twiddle-Perturbation Faults on the Forward NTT

**arXiv ID:** 2610.30383 | [PDF](https://arxiv.org/pdf/2610.30383v1)

**作者:** Chakshu Gupta `[一作]` `[通讯]` (Georgia Institute of Technology), Chakshu Gupta (Georgia Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在NIST标准化的ML‑KEM和ML‑DSA密钥生成中，针对单个或每层一次的静态twiddle（乘数）替换所导致的前向数论变换（NTT）差分的线性映射，推导出单个fault的秩、任意fault集合的上界以及一层一换（one‑per‑layer）模式下的精确秩与核维度，并结合已知的映射与差分来恢复或限定密钥多项式的系数。

**💡 创新点**

创新点包括：① 给出了单个twiddle替换的秩等于其所属蝴蝶数的闭式表达；② 证明任意fault集合的秩不超过各单个fault秩之和，并在一层一换模式下实现该上界；③ 通过形式化证明（Lean4）验证了核维度为2（ML‑KEM）或1（ML‑DSA）的结论；④ 推导出在已知映射和差分的前提下，能够恢复除常数与线性系数（ML‑KEM）或仅常数（ML‑DSA）外所有系数的线性组合；⑤ 对压缩公钥差分的概率误差（total erasure）给出上界，说明单次fault不易恢复完整密钥；⑥ 为后续对抗机制（检测、掩码）提供了可量化的泄露速率。

**🔧 技术方法**

使用的技术包括：NTT的层级结构与蝴蝶变换理论、线性代数（秩、核、行空间）分析、层级递归与张量化简、对称子空间定位（bit‑subspace localization）、Lean4形式化验证、实验验证（Python/C）以及概率论模型（Bin 误差分析）。

**📊 数据集**

不使用外部数据集；实验仅在标准参数下进行：ML‑KEM（n=256, q=3329）与ML‑DSA（n=256, q=8380417）以及部分小规模参数（n=32, 64）。

**📈 对比分析**

与已有攻击（如Kyber零化攻击、Dilithium签名fault攻击）对比，本文提供了更精确的候选计数上界和秩分析，说明单次fault并不能直接恢复完整密钥；实验结果验证了理论预测的秩与核维度；相较于传统检测方法，本文给出了可量化的泄露速率，为对抗设计提供参考。

**⚠️ 局限性**

局限性：① 仅考虑静态twiddle替换模型，未涵盖输入系数fault、乘积bit‑flip、重排表fault等；② 未给出完整的攻击步骤或公钥恢复方法，只给出线性约束和候选计数；③ 实验仅验证数学模型，未实现物理fault注入；④ 对ML‑DSA压缩公钥差分的恢复能力仍未确定；⑤ 结果对实际实现的细节（如优化、打乱表）敏感，可能需要进一步验证。

---

## 51. Climbing the Ladder: Automating Data Pipelines and Reproducing Baselines in Low-Resource Settings

**arXiv ID:** 2610.30374 | [PDF](https://arxiv.org/pdf/2610.30374v1)

**作者:** Donato Meoli `[一作]` (University of Pisa), Eryk Wdowiak `[通讯]` (Arba Sicula)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新实现了从 Arba Sicula PDF 提取、自动对齐、清洗双语文本，并改进了分词与子词分割，随后在 NLLB‑200 上使用 LoRA 进行分阶段微调，最终构建了高质量的意-西西里语 NMT 系统。

**💡 创新点**

创新点包括使用 LaBSE 句子嵌入与动态规划自动对齐、加入形态词尾偏置的 BPE 分词与词干因子，以及在 NLLB 上通过低秩适配器结合反向训练的分阶段方案，大幅提升了低资源语言的翻译效果。

**🔧 技术方法**

技术栈包括小型 Transformer（Sockeye）、NLLB‑200 + LoRA 低秩适配器、LaBSE 句子编码对齐、Vecalign 风格动态规划、BPE 与形态词尾偏置、词干因子、反向训练与预训练，以及语料清洗和后缀合并。

**📊 数据集**

数据集涵盖 Arba Sicula 双语 PDF、Good‑Sicilian‑in‑NLLB（LASER3 挖掘）、WikiMatrix 的 scn‑en 与 scn‑it、Napizia 手工对齐页、Napizia 语法练习、Young Sicilian Manifesto、Napizia Magazine 以及 NLLB 的单语西西里语等。

**📈 对比分析**

通过 sacreBLEU/chrF 在固定 1,000 句文学测试集与第二作者的 121 行测试集上评估，实验结果显示基线+LevrB 仅提升至 10.94 BLEU，LoRA 预训练提升至 31.27 BLEU，分阶段训练进一步提升至 32.40 BLEU，最终模型达到 36.01 BLEU（scn→en）/26.34 BLEU（en→scn），显著优于先前的 reverse‑training 系统。

**⚠️ 局限性**

局限主要在于多语种适配器在未训练方向上出现遗忘、对非文学领域和方言文本的鲁棒性有限、训练数据仍受正字法多样性影响、后续扩展需更多公开语料，以及对不同语言桥接效果的可解释性不足。

---

## 52. CTRL-FM: Inference-Time Structural Control for Flow-Based Virtual Sensing under Long-Horizon Node Outages

**arXiv ID:** 2610.30362 | [PDF](https://arxiv.org/pdf/2610.30362v1)

**作者:** Boshra Khalili `[一作]` (Columbia University), Andrew Smyth `[通讯]` (Columbia University)

**通讯引用:** 5314 | [OpenAlex ID](https://openalex.org/A5025805379)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种观察保持的流匹配框架CTRL‑FM，用于长时间节点缺失的时空信号重建。

**💡 创新点**

核心创新是把结构化引导视为缺失条件的推理时控制问题，利用自监督的代理缺失来选择并局部调节结构化策略，同时严格保证观测值不变。

**🔧 技术方法**

技术实现包括缺失流匹配骨干、代理缺失策略评估、观测仅可靠性估计、边界锚定结构引导、保守弃权和流控制插值等模块。

**📊 数据集**

实验使用了四个公共数据集：交通网络PeMS08、METR‑LA以及空气质量网络Air‑36和AQI。

**📈 对比分析**

与Graph Neighbor、SAITS、IGNNK、GRIN、OPCR、CSDI、PriSTI、CDSTE、GiFlow等基线比较，CTRL‑FM在30%和60%节点缺失下在MAE、CRPS和Peak‑ΔMAE等指标上均取得最优或第二优表现，甚至在75%极端缺失下仍保持优势。

**⚠️ 局限性**

局限性在于依赖足够多的观测节点和有效的传感器图；在观测极其稀疏或故障高度聚集的场景下，代理构造与结构化控制的有效性可能下降，且缺乏真实缺失事件的验证。

---

## 53. The Psychology of an AI Hacker

**arXiv ID:** 2610.30361 | [PDF](https://arxiv.org/pdf/2610.30361v1)

**作者:** Víctor Mayoral-Vilches `[一作]` (Alias Robotics), Aitana Cornago `[通讯]` (Alias Robotics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在4B cybersecurity transformer的残差流中插入情绪方向，实现了对AI黑客行为的可读性与可控性，并在Cybench的capture‑the‑flag子任务上对多种情绪干预进行了实验评估。

**💡 创新点**

将人类黑客心理学与情绪方向结合，首次在不修改权重的前提下将情绪概念转化为可驱动模型行为的功能性干预，并设计了基于任务上下文的情绪路由策略。

**🔧 技术方法**

使用大型Transformer模型（CyberSecurity 4B）、残差流情绪方向提取（对171种情绪），PCA与层级可视化，情绪投射与剂量调节，以及基于决策头的动态路由器。

**📊 数据集**

主要使用了Cybench benchmark的40个capture‑the‑flag挑战（171个引导子任务）进行评估，并使用生成的360条短篇故事用于情绪方向提取。

**📈 对比分析**

通过与未干预基线和安慰剂进行比较，使用10个随机种子、Welch置信区间、百分点差异衡量，单向情绪干预如calm和desperate分别提升约2pp，而动态路由器可获得约4.2pp的提升。

**⚠️ 局限性**

实验规模受限于单一4B模型和Cybench任务，情绪方向的效果显著但相对较小，剂量与层次对结果影响显著，缺乏更大规模、多任务或更高阶模型的验证，且对情绪与行为之间因果机制的解释仍不充分。

---

## 54. MoRE: Motion-Regularized Encoding for Articulated 3D Generation

**arXiv ID:** 2610.30337 | [PDF](https://arxiv.org/pdf/2610.30337v1)

**作者:** Guantian Zheng `[一作]` (Manifest Limited), Shan Jin `[通讯]` (Manifest Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

从单张图片生成可运动的三维物体，并学习关节属性与运动行为一致的表示；

**💡 创新点**

提出Motion-Regularized Encoding（MoRE），通过轴正交Plücker时刻系数消除轴向原点冗余，并加入可微运动正则化；同时结合单图像流式潜在和多状态高斯外观实现统一生成与操控；

**🔧 技术方法**

使用VAE+图像条件流式潜在、Plücker关节线编码、可微运动正则化、DBSCAN聚类、DINOv3特征、Gaussian splatting外观等技术；

**📊 数据集**

主要在PartNet-Mobility、ACD、Infinigen-Articulated、SINGAPO等数据集上训练与评估；

**📈 对比分析**

与SINGAPO、ArtFormer、PhysX-Anything、PAct等基线在RS/AS CD、CLIP等指标进行对比，MoRE在多项指标上取得最优或第二佳表现；在关节预测上轴误差、原点误差、范围误差均低于ArtiLatent；

**⚠️ 局限性**

仅支持固定、转动、平移三种轴对齐关节，未考虑层级动力学链；聚类与交叉网格分配可能导致推断错误；缺乏物理参数校准；训练需要手工标注部件与关节信息。

---

## 55. MPBRS: Multi-Potential-Based Reward Shaping for Unified Reward Configuration in Humanoid Locomotion

**arXiv ID:** 2610.30334 | [PDF](https://arxiv.org/pdf/2610.30334v1)

**作者:** Van Chung Nguyen `[一作]` (University of Nevada), Hung Manh La `[通讯]` (University of Nevada)

**通讯引用:** 6706 | [OpenAlex ID](https://openalex.org/A5060555317)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在九种不同人形机器人上使用统一的多潜在奖励塑形（MPBRS）来改进深度强化学习步态训练。

**💡 创新点**

创新点在于将三种潜在函数（底部高度、正直度和关节正则化）集成到PBRS中，并证明其在不同机器人上无需重新调参即可获得稳定性能。

**🔧 技术方法**

采用PPO-Clip强化学习算法、潜在奖励塑形技术，并在NVIDIA IsaacLab仿真环境中训练。

**📊 数据集**

使用在IsaacLab中模拟的九款人形机器人（Fourier N1、TienKung2、Unitree G1、Unitree H1、Booster T1、Fourier GR3V2、Kuavo S45、Hightorque HI、Berkeley Humanoid Lite）。

**📈 对比分析**

与基线奖励和直接奖励塑形（DRS）进行比较，MPBRS在训练稳定性、对权重敏感度降低方面表现更好，且跟踪性能与基线相当。

**⚠️ 局限性**

局限性在于收敛速度提升有限，且仍受机器人动力学和模拟-真实差异影响，需要进一步学习或自适应潜在函数以提升泛化。

---

## 56. R4DGS: Referring Segmentation in 4D Gaussian Splatting

**arXiv ID:** 2610.30326 | [PDF](https://arxiv.org/pdf/2610.30326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 57. Self-Distilling Future Video Representations for World Action Models

**arXiv ID:** 2610.30320 | [PDF](https://arxiv.org/pdf/2610.30320v1)

**作者:** Huiwon Jang `[一作]` (KAIST), Sihyun Yu `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种改进的 World Action Model（WAM）框架，利用单步视频扩散模型通过自我蒸馏和短期历史条件学习未来视频表示，从而提升机器人控制性能。

**💡 创新点**

创新点在于将视频扩散模型视为未来表示生成器，采用自我蒸馏将噪声下生成的表示与从噪声未来观测提取的目标表示对齐，并加入历史观测提供时序上下文，显著提升动作解码所需的未来信息。

**🔧 技术方法**

使用单步视频扩散模型（VDM）、自我蒸馏（EMA教师）、短期历史条件、动作解码器（DiT）、流匹配损失、softmax温度调节以及梯度阻断等技术。

**📊 数据集**

主要使用公开的 DROID 机器人数据集、RoboCasa Kitchen、GR‑1 Tabletop、LIBERO 以及自收集的 OpenArm 机器人和人类视频进行训练与评估。

**📈 对比分析**

在零射击环境泛化、长周期任务、外域任务和人机技能迁移等多种真实场景中，与多种 WAM 基线（如 DiT4DiT、Fast‑WAM、AGRA、DreamZero）以及 VLA 模型相比，取得显著提升（例如长周期任务从 61.7% 提升到 81.7%，外域任务从 10.4% 提升到 52.1%，人机迁移任务成功率从 19.4% 提升到 50%），同时保持较低的推理延迟。

**⚠️ 局限性**

局限性包括对噪声水平、教师层选择、超参数等的敏感性；在极端外域任务和更复杂时序任务上仍有改进空间；目前仅在单步推理下评估，未探讨多步迭代效果；以及对不同视频背骨的迁移性需要进一步验证。

---

## 58. Prefigured forgetting. Memory, the right to be forgotten and artificial intelligence

**arXiv ID:** 2610.30402 | [PDF](https://arxiv.org/pdf/2610.30402v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 59. Route-and-Generate: KL-Regularized Router Improvement for Reasoning Routing in LLMs

**arXiv ID:** 2610.30351 | [PDF](https://arxiv.org/pdf/2610.30351v1)

**作者:** Tien-Phat Nguyen `[一作]` (Hanoi University of Science and Technology), Trung Le `[通讯]` (Monash University)

**通讯引用:** 2346 | [OpenAlex ID](https://openalex.org/A5102780660)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Route-and-Generate（Rouge）框架，分离推理策略选择与答案生成，显式控制推理成本。

**💡 创新点**

创新点在于：① 引入KL正则化的路由器，实现对计算预算的软分配；② 采用基于排名的效率奖励，鼓励更短、正确的推理结果；③ 通过闭式解更新路由策略，避免格式崩溃。

**🔧 技术方法**

使用了GRPO（强化学习策略梯度）、KL正则化路由、格式化模板推理、基于奖励的自适应采样等技术。

**📊 数据集**

在CommonsenseQA、GSM8K、MATH、AIME 2025/2024、AMC 2023、SVAMP、BBH、CSQA等多种推理基准上进行实验。

**📈 对比分析**

与SFT、GRPO、O1‑Pruner、ARM等基线对比，Rouge在保持甚至提升Pass@1 / Maj@8准确率的同时，大幅减少平均生成长度（token），在不同模型规模（Qwen2.5‑3B、7B）上均显著提升效率。

**⚠️ 局限性**

局限性包括：仍需手工设定格式集合；对极其复杂问题的格式选择仍可能不足；奖励设计对参数ρ敏感，需针对任务调优；在推理速度上未直接评估推理过程中的实际延迟。

---

## 60. Latent Space Jam: Layout Guided Video Synthesis

**arXiv ID:** 2610.30390 | [PDF](https://arxiv.org/pdf/2610.30390v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 61. Who Patches the Lightning Network? Evidence from Release Lines in Gossip, 2023-2026

**arXiv ID:** 2610.30350 | [PDF](https://arxiv.org/pdf/2610.30350v1)

**作者:** Saurabh Kumar `[一作]` `[通讯]`, Saurabh Kumar

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Lightning Network节点的发布线进行推断并测量升级动力学，评估平台与披露对补丁部署的影响。

**💡 创新点**

首次证明即使没有版本字符串，功能位也可准确推断发布线，揭示平台推送是驱动补丁的主要力量，而披露影响有限。

**🔧 技术方法**

基于功能位推断算法、特征向量匹配、最大似然估计、间隔检验等统计方法。

**📊 数据集**

从2023‑05到2026‑10每日公共图谱的gossip快照，以及所有LND、CLN、Eclair、LDK的发布记录和公开安全通告。

**📈 对比分析**

通过与平台发布时间对齐、波动窗口计数和前后披露率比值的比较，结果显示平台发布导致数千节点在48小时内升级，而披露仅提升少量已接近最新版本的节点。

**⚠️ 局限性**

受限于无法区分补丁发布、仅从单一节点视角、无法检测未公开节点、对功能位配置变化的误判以及对受影响节点可被攻击性的粗略估计。

---

## 62. TerraTransfer++: Demonstration-Free End-to-End Planner

**arXiv ID:** 2610.30319 | [PDF](https://arxiv.org/pdf/2610.30319v1)

**作者:** Zikang Xiong `[一作]` (Applied Intuition), Wei Zhan `[通讯]` (Applied Intuition)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种全无演示的端到端自动驾驶学习框架：先用多智能体自对弈训练向量状态策略，再通过轨迹蒸馏生成规划教师，最后用结构对齐损失将视觉前端映射到教师特征，从而实现基于相机的闭环控制与规划。

**💡 创新点**

创新点在于：①将自对弈的驾驶能力迁移到图像空间而不需要人类示例；②利用低秩结构对齐损失精准捕捉教师的场景关系；③通过流匹配轨迹蒸馏一次性获得可直接用于规划的轨迹教师；④整个流程无需在线渲染或教师轨迹，显著提升训练效率。

**🔧 技术方法**

技术细节包括：多智能体自对弈（PPO），轨迹蒸馏（flow‑matching diffusion head），动作KL对齐，低秩结构对齐损失，轻量化视觉适配器（DINOv3 backbone + linear adapters），以及对教师的动作与场景特征对齐。

**📊 数据集**

使用的主要数据集：
- TerraZero向量化仿真环境自行生成的对弈数据；
- HUGSim（3D Gaussian splatting 闭环基准）
- NAVSIM（4 s 规划基准）
- 436 场景的 3D Gaussian splatting 交互闭环测试集。

**📈 对比分析**

与传统基于示例的 imitation、模块化感知+控制以及并行的自对弈‑像素方法（如 GigaPixel、DriveZero）对比，闭环评分达 0.548（仅比标杆低 0.014），规划评分在 NAVSIMv2 达 59.4 EPDMS，NAVSIMv1 达 95.5 PDMS，均超过或逼近公开最佳结果。

**⚠️ 局限性**

局限性包括：对极端场景（如不可避免碰撞）性能仍不足；教师在遮挡或不可见对手上有优势，导致学生在图像域存在感知缺失；对极端天气、时间等环境的泛化尚未验证；需要大量配对 (image, scene‑state) 训练样本。

---

## 63. Escaping the Phylogenetic Trap: Persistent Homology Enables Cross-Species Ion Channel Classification via Multimodal Graph Neural Networks

**arXiv ID:** 2610.30394 | [PDF](https://arxiv.org/pdf/2610.30394v1)

**作者:** Arezoo Bybordi `[一作]` (City University of New York), Anita Raja `[通讯]` (City University of New York)

**通讯引用:** 910 | [OpenAlex ID](https://openalex.org/A5055430969)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究提出了一种融合蛋白序列嵌入、离散二硫键感知图注意网络和持久同调特征的多模态神经网络（PH‑GNN），用于跨物种的毒液肽离子通道目标分类。

**💡 创新点**

创新点在于：①首次将持久同调（Topological Data Analysis）应用于毒液肽结构特征提取，提供对进化无关的拓扑不变性；②通过严格的“物种留一法”评估，揭示传统随机划分的“系统进化泄漏”问题；③系统对比五种几何先验（统计流形、谱图、黎曼流形、SE(3)不变性、持久同调），证明持久同调在跨物种泛化上显著优于其它先验。

**🔧 技术方法**

使用技术包括：ESM‑2 预训练语言模型提取序列嵌入；基于所有原子邻接图的离散二硫键加权图注意网络；Alpha‑Complex 持久同调计算（提取 H0/H1/H2 的 12 维统计向量）；多模态融合 MLP；以及基于交叉熵的加权训练和留一法交叉验证。

**📊 数据集**

数据集为 1,373 条 ColabFold 预测的毒液肽，涵盖 5 种物种（蛇、锥形蜗牛、蜘蛛、蝎子、海葫芦）及 3 种离子通道目标（Ca²⁺、K⁺、Na⁺）。

**📈 对比分析**

在随机 80/20 划分下，所有模型性能相近（81–86%）。在物种留一 OOD 评估中，PH‑GNN 取得 63.4% 的平均准确率，超过其他模型（54–55%）和仅序列的 ESM+RF 基线（≈30%），表明持久同调在跨物种泛化上具备显著优势。

**⚠️ 局限性**

限制包括：①在蜘蛛物种（ICK 结构多样）上性能仍低（≈46%），表明拓扑特征仍受结构异质性影响；②模型未针对通道多重识别（多标签）或高阶二硫键连通性（结理论）进行建模；③仅使用单一 Alpha‑Complex 截断阈值，可能忽略更细粒度的拓扑信息。

---

## 64. Exploring spatial configuration and group familiarity when co-watching 360-degree stories in social virtual reality

**arXiv ID:** 2610.30353 | [PDF](https://arxiv.org/pdf/2610.30353v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 65. Bayesian Fixed-Confidence Best-Arm Identification with Abstention

**arXiv ID:** 2610.30384 | [PDF](https://arxiv.org/pdf/2610.30384v1)

**作者:** Yuqi Huang `[一作]` (National University of Singapore), Vincent Y. F. Tan `[通讯]` (National University of Singapore)

**通讯引用:** 5733 | [OpenAlex ID](https://openalex.org/A5058345431)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了具有“拒绝”选项的贝叶斯固定置信最佳臂识别（FC-BAI）问题，并给出了匹配的上下界。

**💡 创新点**

提出了 Screen-and-Confirm (S&C) 算法，证明在α+δ→0下实现最优样本复杂度，并给出固定正拒绝预算与趋于零拒绝预算两种实用情形下的精确常数。

**🔧 技术方法**

利用贝叶斯信息理论、先验近似、层级屏蔽与确认抽样、极限分布与特征时间分析等技术，构建算法并推导样本复杂度上界与下界。

**📊 数据集**

实验使用了5臂高斯先验（均值为[5,5,3,3,2]、方差1）与高斯奖励，进行10⁵次随机实例模拟。

**📈 对比分析**

与无拒绝的 Successive Elimination (SE-early) 进行比较，S&C 在满足错误概率δ与拒绝率α约束下，样本量呈对数增长，明显低于无拒绝方案的1/δ级别；错误率与拒绝率均落在规定阈值之下。

**⚠️ 局限性**

局限在于仅对独立高斯先验与高斯奖励建模，未讨论非高斯奖励或相关先验，以及如何在实际大规模应用中实现算法的高效实现。

---

## 66. Gender, Sexual Orientation, and the One-Year Refugee Bar: A Computational Study of Bill C-12 and Canadian Refugee Adjudication

**arXiv ID:** 2610.30297 | [PDF](https://arxiv.org/pdf/2610.30297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 67. FAIRGate: Agentic Runtime Policy Selection and Verified Repair for Fairer Recommendation Delivery

**arXiv ID:** 2610.30359 | [PDF](https://arxiv.org/pdf/2610.30359v1)

**作者:** Navkar Jain `[一作]` (University of Massachusetts Boston), Kenneth Fletcher `[通讯]` (University of Massachusetts Boston)

**通讯引用:** 524 | [OpenAlex ID](https://openalex.org/A5042466369)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 FAIRGate 框架，在推荐系统交付阶段通过 agentic 策略选择与验证修复实现可测、可审计的公平性控制。

**💡 创新点**

创新点在于将公平性控制拆分为有限策略库与受限 agentic 选择，并引入验证循环与审计跟踪，保证运行时可调节但可追溯的公平交付。

**🔧 技术方法**

使用 LLM（Llama 3.1 8B）进行策略选择与修复规划，基于 deterministic executor/validator 进行约束修复，并采用结构化生成与 schema 校验保证输入输出合法。

**📊 数据集**

使用 MovieLens 1M 数据集，以电影流行度、类型等元数据进行实验。

**📈 对比分析**

与基准 BPR‑MF、静态重排序、固定顺序 FAIRGate 以及无验证版比较，实验表明 agentic FAIRGate 在保持 NDCG@10 的同时将 LT‑Cov@10 提升 21.5%，策略通过率从 9% 提升至 60%，且放弃率不变。

**⚠️ 局限性**

局限在于仅在 MovieLens 1M 上验证，依赖预定义策略库和阈值，对不同域的可扩展性有限；同时 LLM 生成解释受 prompt 设计和数据偏差影响。

---

## 68. CiteAnchor: Source-Aware Grounding for Hallucinated Citation Detection and Resolution

**arXiv ID:** 2610.30355 | [PDF](https://arxiv.org/pdf/2610.30355v1)

**作者:** Chenyi Tong `[一作]` (University of Wisconsin--Madison), Yu Zhang `[通讯]` (Texas A&M University)

**通讯引用:** 17274 | [OpenAlex ID](https://openalex.org/A5100433687)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提出CiteResolve基准和源感知Grounding框架，用于检测并恢复论文、书籍及网页的虚假引用。

**💡 创新点**

创新点在于将虚假引用检测与源身份判定分离，构建可判定是否可唯一恢复源的评价指标，并通过证据分解、类型适配的检索与字段校验实现源感知。

**🔧 技术方法**

采用检索‑验证‑解析三阶段管道，利用结构化API（Crossref、OpenAlex等）与网页搜索（SerpApi）相结合，并以GPT‑5.6 Luna与OpenResearcher‑30B‑A3B LLM为推理引擎。

**📊 数据集**

使用423条真实世界虚假引用的CiteResolve数据集（来源自CiteAudit）以及CiteAudit的3,356条测试集。

**📈 对比分析**

与基线CiteAudit相比，该框架在CiteAudit上F1从0.762提升至0.841（GPT‑5.6）/0.784（OpenResearcher），在CiteResolve上任务成功率从Web Search的58.4%/5.4%提升至74.7%/70.7%。

**⚠️ 局限性**

主要限制是数据集中主要为论文引用，书籍和网页占比不足；框架仅验证书目合法性，未评估引用是否支持上下文；对外部数据库和网页的依赖可能随时间变动导致效果波动。

---

## 69. Spike-Aware Gradient-Based Mixed-Precision Quantization of Spiking Neural Networks for Vision Applications

**arXiv ID:** 2610.30339 | [PDF](https://arxiv.org/pdf/2610.30339v1)

**作者:** Rajshekhar Rakshit `[一作]` (Indian Institute Of Technology Madras), Gopalakrishnan Srinivasan `[通讯]` (Indian Institute Of Technology Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于梯度的混合精度量化框架，专门针对脉冲神经网络（SNN）设计，能够在保持高精度的同时显著压缩模型。

**💡 创新点**

创新点在于：①使用一阶梯度信息和膜电位阈值余量来估计层级量化敏感度，完全避免了昂贵的Hessian/Hutchinson计算；②引入阈值余量因子，对靠近发火阈值的层更保守地分配位宽；③采用贪婪搜索实现全局位宽预算下的最优分配。

**🔧 技术方法**

主要技术包括：梯度驱动的敏感度度量、膜阈值余量统计、混合精度位宽分配的约束优化（贪婪算法），以及基于Surrogate梯度的SNN训练（如TET）和ANN‑to‑SNN转换（QCFS、PASCAL）。

**📊 数据集**

实验使用了CIFAR‑10/100、ImageNet、CIFAR‑10‑DVS等静态与神经形态数据集，模型覆盖VGG、ResNet、Vision Transformer（ViT）等多种网络。

**📈 对比分析**

与Hawq‑V2比较时，本方法在保持相同压缩比（如4/8位混合）下，ImageNet ResNet‑34实现73.30% Top‑1（vs 73.14%），并在GPU内存和量化时间上分别提升约56%与100×；在直接训练的SNN上，精度与压缩率与Hawq‑V2持平或略优；相比W1/W2异构方案，压缩率提升至13×以上，准确率提升0.3%~0.5%。

**⚠️ 局限性**

局限性包括：①仍依赖一阶梯度近似，对极度非线性或高度噪声的网络可能不足；②阈值余量参数β需要经验调优；③对不同仿射梯度函数的鲁棒性虽好，但在极端仿射梯度下的位宽分配仍需进一步验证；④量化后需要额外的微调才能达到最优性能，未完全在所有场景下实现零微调。

---

## 70. Generalizing Metropolis-Hastings for DAG Sampling

**arXiv ID:** 2610.30291 | [PDF](https://arxiv.org/pdf/2610.30291v1)

**作者:** Daniele Nikzad `[一作]` (University of Helsinki), Mikko Koivisto `[通讯]` (University of Helsinki)

**通讯引用:** 2611 | [OpenAlex ID](https://openalex.org/A5088812539)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对Bayesian网络结构学习中的DAG后验分布采样问题提出了改进的MCMC方法。

**💡 创新点**

创新点是提出双重采样（duet sampling）通用框架，并基于此设计了三类新的DAG跳跃移动：边受限（EC）、祖先受限（AC）和无约束顺序重采样（USR）。

**🔧 技术方法**

核心技术包括：在Metropolis–Hastings中使用两种不同提议分布的接受准则；利用父集剪枝、区间和与交集和加速父集概率计算；以及在USR中使用独立或反转配对提升链的混合速度。

**📊 数据集**

实验使用了三个大规模离散贝叶斯网络数据集：Hailfinder（56节点）、Andes（223节点）和Pigs（441节点）。

**📈 对比分析**

与传统单节点移动（FBM、REV、MBR）、块式吉布斯（BG）以及Gibby算法进行比较；结果显示USR‑rev（以及MIX组合）在三组数据中最快收敛、得到最高后验分数，且在多次独立运行中边缘概率估计一致性最好；EC/AC在单独使用时表现不稳，Gibby和BG在大网络上收敛慢或陷入局部模式。

**⚠️ 局限性**

主要限制包括：需要预先计算大量局部得分，导致内存和时间开销大；对候选父集的预选可能导致偏差；单独使用EC/AC在某些网络上易卡住；USR的表现虽好但对不同网络仍无绝对保证。

---

## 71. VisCurate: Evaluating Behavioral Equivalence of Visual Skills

**arXiv ID:** 2610.30405 | [PDF](https://arxiv.org/pdf/2610.30405v1)

**作者:** Md Farhan Ishraq `[一作]` (Islamic University of Technology), Md Hasanul Kabir `[通讯]` (Islamic University of Technology)

**通讯引用:** 3324 | [OpenAlex ID](https://openalex.org/A5071274329)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VisCurate 框架和基准，用执行输出而非文本描述来审核并自动维护视觉技能库。

**💡 创新点**

创新点在于：①基于输出的等价性判定及其完整的行为关系分类；②在已注入七种缺陷的可复现库上构建的系统化评测基准；③通过安全的验证器和结构化编辑策略实现无误合并与重构；④对九种 LLM 驱动的自动修复器进行了细粒度评估。

**🔧 技术方法**

核心技术包括：视觉函数的确定性执行、探测电池（probe battery）生成、输出规范化 ϕ、LPIPS 感知距离、DINO/CLIP 语义特征、层级关系分类器、基于验证器的安全编辑、语言模型策略代理（未训练）以及基准生成脚本。

**📊 数据集**

使用了：①100 个手工构建的干净视觉技能基线（包含 944 设计关系对）；②通过七种缺陷注入产生的 300 个可复现腐败实例；③两套真实无标注库（PILgram v2.0.0 与 v2.0.10 及跨库 Corpus A 的 93 个实现）。

**📈 对比分析**

比较方法：在 944 设计关系上对八个文本/代码基线与 VisCurate 验证器进行 0/6/69/182 等价性决策评估；在 300 个腐败实例上对验证器做召回/精度评估；对九个 LLM 代理做 F1、召回、误操作成本等指标；最终发现 VisCurate 在所有设置下均无误合并，召回约 51%，而文本/代码基线误合并率高至 5/6；LLM 代理仅恢复约一半缺陷，最大 F1 仅 0.587。

**⚠️ 局限性**

局限包括：①验证器安全性高但召回不足，导致许多可合并对未被识别；②缺少对计算成本、依赖体积等非功能属性的考量；③对深度学习/随机性技能的支持有限；④基准仅覆盖图像处理操作，可能不适用于更复杂的视觉任务；⑤LLM 代理性能高度依赖模型规模与参数，无法普适解决库维护问题。

---

## 72. Soft Ontology Priors over Latent Concept Posteriors for Weakly Supervised Neuro-Symbolic Learning

**arXiv ID:** 2610.30391 | [PDF](https://arxiv.org/pdf/2610.30391v1)

**作者:** Tengwei Song `[一作]` (King Abdullah University of Science and Technology), Robert Hoehndorf `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 7937 | [OpenAlex ID](https://openalex.org/A5043808311)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

引入软本体先验到弱监督神经符号学习中，通过将本体知识编译进加权电路，以解决仅任务标签训练导致的推理捷径问题。

**💡 创新点**

通过软化本体先验，将符号约束与神经感知的联合后验编译为可线性求解的加权模型计数电路，并证明软先验对受污染的本体具有鲁棒性，而硬约束则不然；同时在概念可识别性诊断中验证了其有效性。

**🔧 技术方法**

软本体先验、加权电路(sd‑DNNF)编译、独立Bernoulli感知模型、归一化WMC目标、后验正则化思想、概念可识别性诊断与对抗性噪声评估。

**📊 数据集**

BDD‑OIA‑6c（交通视图）、MNIST‑Add（数字求和）、MNIST‑Half（稀疏训练版本）、MNLogic‑3bit‑CNF（三位CNF约束）。

**📈 对比分析**

与DPL、SPL、HardPC、BEARS等基线对比；在任务准确率基本相同的情况下，软本体先验显著提升概念F1（如BDD‑OIA‑6c提升0.05、MNIST‑Add提升0.28），并在概念可识别性诊断中多数随机种子接近oracle。

**⚠️ 局限性**

本体可靠性有限；随机删除噪声鲁棒，但对抗性“捷径”注入仍能偏移后验；单一全局β和固定权重难以自适应识别不可靠或不完整知识，且仅依赖任务标签在OOD情形下易留下捷径。

---

## 73. The Inverse Mapper Problem for Two-Dimensional Data

**arXiv ID:** 2610.30380 | [PDF](https://arxiv.org/pdf/2610.30380v1)

**作者:** Robin Belton `[一作]` (Vassar College), Patrick Schnider `[通讯]` (University of Basel)

**通讯引用:** 1744 | [OpenAlex ID](https://openalex.org/A5074690525)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了Mapper算法在二维数据（滤波函数f:ℝ²→ℝ）下的逆问题，系统性地给出了在不同参数固定/数据集固定场景下可产生的Mapper图的完整分类；

**💡 创新点**

创新点在于证明在覆盖层为ply 2且聚类为连通分量或单链接聚类等约束下，Mapper图分别对应二部字符串图、平面二部图、任意二部图以及森林触点图，并探讨了凸聚类与单链接聚类的可实现性，首次把Mapper图与图绘制理论、凸集、球面嵌入等多领域方法联系起来；

**🔧 技术方法**

主要使用Mapper算法的定义、拓扑数据分析中的nerve理论、图绘制与层绘图理论、凸聚类与单链接聚类的几何构造、Erdős–Szekeres极值定理以及球面投影等技术；

**📊 数据集**

实验层面未使用真实数据集，而是构造理论点集、图形和凸点集来验证可实现性；

**📈 对比分析**

本文以理论证明为主，未做数值实验或性能比较；通过构造与反构证明展示了不同参数组合下Mapper图的可达性；

**⚠️ 局限性**

局限性包括：只考虑覆盖层ply 2、滤波函数为投影或特定形式的情况；对于非二部或非平面图的Mapper图尚无完整答案；单链接聚类无outlier时的完全性仍未得到证明。

---

## 74. MemPlan: Memory-Conditioned PDDL Planning for Partially Observable Text Environments

**arXiv ID:** 2610.30379 | [PDF](https://arxiv.org/pdf/2610.30379v1)

**作者:** Tengwei Song `[一作]` (King Abdullah University of Science and Technology), Robert Hoehndorf `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 7937 | [OpenAlex ID](https://openalex.org/A5043808311)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 MemPlan，利用固定的 PDDL 域和交互记忆构建闭合 PDDL 问题，在部分可观测的文本环境中使用经典规划器完成长序列任务。

**💡 创新点**

将正负记忆编译为计划成本和排除约束，使用占位符假设候选推理，仅在在线阶段使用 LLM 进行限定的解析与 grounding，从而显著降低 token 开销并提升规划效率。

**🔧 技术方法**

使用 Fast Downward 经典规划器、冻结的 schema‑conditioned 语言‑符号接口（基于 Qwen2.5‑7B）、密集图记忆检索（mContriever）为候选排序，以及正负记忆机制。

**📊 数据集**

在 TextWorld、ALFWorld、Robotouille 三个任务集上进行实验，分别采用 7 个 TextWorld 任务、134 个 ALFWorld 试验和 151 个 Robotouille seed。

**📈 对比分析**

与语言仅模型、LLM‑DP、ReAct、PDDLEGO+ 等基线对比，MemPlan 在 ALFWorld 的任务完成率达 100%、步数 6.91、token 906k，显著优于所有基线；在 TextWorld 与记忆增强 LLM 代理比较中获得最高归一化分数；在 Robotouille 的同步/异步场景中同样表现最好。

**⚠️ 局限性**

需要预先提供完整的 PDDL 域；负记忆采用硬排除，可能过早剔除合法候选；仅在受限命令空间的模拟文本环境中验证，未评估真实机器人或开放式环境的适用性。

---

## 75. Cutting a Cake For Human Consumption: An Elementary $n$-agent Cake Cutting Protocol

**arXiv ID:** 2610.30370 | [PDF](https://arxiv.org/pdf/2610.30370v1)

**作者:** Alexandros Psomas `[一作]` `[通讯]` (Purdue University), Alexandros Psomas (Purdue University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种在 Robertson–Webb 模型下，对任意 n 名代理人实现有界无嫉妒（envy‑free）的蛋糕分割协议，并给出了完整的、简洁的正确性证明。

**💡 创新点**

创新点在于：1）将已有的 Core、SubCore 等子算法以更简洁的接口重新组织；2）使用回合制（round‑robin）实现的 Repair 子程序；3）证明 Permute 子程序对任意输入块均能成功；4）整体协议的查询复杂度被证明为 2^O(n log n)，大幅降低了之前指数级或超指数级的查询上界。

**🔧 技术方法**

核心技术包括：
- Robertson–Webb 查询模型（Eval、Cut）；
- 等分裁剪（Core）与可交换分配（Permute）；
- 近似无嫉妒分配（Repair）和精确无嫉妒修复；
- 迭代递归分组（Domination）与循环更新；
- 细分单元格与回合制选择实现的小量添加。

**📊 数据集**

该工作为理论算法，没有使用实验数据集，全部是在抽象的连续蛋糕分割模型上进行分析与证明。

**📈 对比分析**

与以往工作比较：
- 之前最好的有界无嫉妒协议（如 Aziz–Mackenzie）查询复杂度为 n^n^n^…，甚至更高；
- 本协议将上界压缩到 2^O(n log n)，实现了对任意 n 的严格有界查询；
- 证明过程简单、易验证，为后续研究提供了可读性更强的基线。

**⚠️ 局限性**

限制与挑战：
- 虽然查询复杂度已被大幅降低，但 2^O(n log n) 对大 n 仍然不可行，实际可执行性有限；
- 协议高度依赖 Robertson–Webb 模型，若假设放宽（如非可测/非连续价值），适用性受限；
- 证明依赖大量细节与组合论，复现难度仍不低；
- 目前仍未突破已知的 Ω(n^2) 下界，未来仍需进一步改进。

---

## 76. Probing and Steering for Safety-Relevant Behaviours in Large Language Models

**arXiv ID:** 2610.30363 | [PDF](https://arxiv.org/pdf/2610.30363v1)

**作者:** Nimun Bajwa `[一作]` (Queen Mary University of London), Massimo Poesio `[通讯]` (Queen Mary University of London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在Llama-3.1-8B-Instruct上构建了一条统一的激活探针检测与引导安全相关行为的管道，涵盖欺骗、高风险意识和阿谀奉承三种行为。

**💡 创新点**

创新点在于将探针的估计器（差均与逻辑回归）、聚合策略（last/mean/attention）与层选择统一调度，并系统检验读/写解耦、行为依赖与数据依赖，首次在同一模型和同一管道中全面比较。

**🔧 技术方法**

技术上使用了激活探针（线性方向）、差均（mass‑mean）与逻辑回归估计器、三种token聚合、方向校准、归一化间接效应（NIE）读取器以及Claude Haiku 4.5 LLM评估者。

**📊 数据集**

采用了Apollo instructed‑pairs、AI Liar、Roleplaying、Models Under Pressure、Anthropic‑HH 和 Math 等公开数据集进行训练和多样化的 OOD 评估。

**📈 对比分析**

比较方法是用 AUROC 评估检测、NIE 评估引导以及 LLM 评估生成行为与连贯度；结果显示检测在 ID 上几乎达到 1，OOD 降幅受样本相似度影响；在引导方面，差均方向在欺骗和高风险下最佳，attention 在阿谀奉承下最佳，但在生成行为上大多无显著提升。

**⚠️ 局限性**

局限性包括未对阿谀奉承的 echo shortcut 进行检验、LLM 评估未验证、仅对部分方向进行引导、层选择受差均偏倚、负系数评估缺失以及模型规模仅到 8B，缺乏更大规模验证。

---

## 77. FiLM-CAM: Keyed Feature Modulation for Conditional-Access Watermarking

**arXiv ID:** 2610.30347 | [PDF](https://arxiv.org/pdf/2610.30347v1)

**作者:** Yash Kulthe `[一作]` (University of Surrey), John Collomosse `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FiLM-CAM，一种条件访问图像水印方法，利用密钥调制网络内部特征，实现编码、解码和移除仅在持有正确密钥时可行。

**💡 创新点**

创新点：1）将密钥嵌入 Feature‑wise Linear Modulation (FiLM) 条件模块，实现单一共享背骨支持多域水印；2）采用对数域水印组合，允许通过残差恢复实现可逆移除；3）在训练中对错误密钥进行对抗优化，最大化授权与未授权之间的性能差距。

**🔧 技术方法**

技术手段：FiLM 条件模块、U‑Net / ConvNeXt Encoder‑Decoder 结构、logit‑space 组合、噪声层、对抗训练、迭代细化恢复残差。

**📊 数据集**

使用数据集：训练基于 MIRFlickr‑1M；评估在 COCO、DIV2K、SA‑1B 三个公开数据集上。

**📈 对比分析**

与 TrustMark、InvisMark、PixelSeal、RoSteALS、SSL、MBRS 等 SOTA 进行对比，FiLM‑CAM 在 100‑bit 水印下取得 ≈0.99 位准确率、PSNR≈46–48 dB，错误密钥下准确率≈0.5；在正确密钥下移除可达到 ≈58 dB，位准确率≈0.57；错误密钥下保持高准确率，体现强条件访问。

**⚠️ 局限性**

限制：无法阻止无密钥的破坏攻击；错误密钥下移除可能仍保留部分水印；需要迭代细化提高移除质量，增加推理开销；实现对不同键的泛化仍需进一步验证。

---

## 78. On the Adversarial Vulnerability of Neural Receivers with Unrolled LDPC Decoders

**arXiv ID:** 2610.30345 | [PDF](https://arxiv.org/pdf/2610.30345v1)

**作者:** Aakash Saini `[一作]` (Nokia), Wolfgang Gerstacker `[通讯]` (Institute for Digital Communications, Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了带有展开LDPC译码器的神经接收机对对抗性扰动的脆弱性，并揭示了梯度遮蔽现象。

**💡 创新点**

首次将梯度遮蔽归因于BP迭代中的tanh饱和，并提出了两种结构独立的攻击损失函数（ℒ_BP和ℒ_LLR）以恢复攻击效果，并证明其跨通道转移性。

**🔧 技术方法**

使用了梯度基攻击（FGM、PGD）、展开BP解码、端到端、逐迭代BP和LLR边际损失函数，结合5G NR上行链路仿真、Sionna和DeepRx神经接收机。

**📊 数据集**

在模拟的5G NR CDL‑C信道上生成随机比特块，采用QPSK调制、长度2784的LDPC码，进行多次随机通道和噪声样本实验。

**📈 对比分析**

通过BLER与PSR曲线对比PGD‑E2E、PGD‑BP、PGD‑LLR、FGM和高斯噪声攻击；结果显示PGD‑LLR在低PSR下BLER可提升至10^2–10^4倍，PGD‑BP也显著优于高斯噪声，而PGD‑E2E与FGM性能相近。

**⚠️ 局限性**

局限在于仅考虑了在接收端直接注入扰动的理想化攻击模型，未考虑真实射频传输路径；实验仅基于仿真信道与码率，缺乏实际射线传播和硬件实现验证。

---

## 79. AcoustiPrint: Physics-Guided Neural Design of Manufacturable Acoustic Structures

**arXiv ID:** 2610.30411 | [PDF](https://arxiv.org/pdf/2610.30411v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 80. Measuring Research Progress and Linguistic Inclusion in Low-Resource Languages

**arXiv ID:** 2610.30296 | [PDF](https://arxiv.org/pdf/2610.30296v1)

**作者:** Atnafu Lambebo Tonja `[一作]` (University College London), Monojit Choudhury `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究系统地梳理了2020‑2025年间1,158篇关于低资源语言（LRL）的开源论文，评估了其研究增长、语言覆盖与公平性，并提出了多维度的语言包容框架（LIF）

**💡 创新点**

创新点在于引入LIF四维度（研究增长、语言扩展、语言持续性、分布公平性）来量化语言包容度，突破单纯计数和模型性能的局限；同时首次将LIF应用于洲际层面进行对比分析

**🔧 技术方法**

采用大型语言模型Gemini 2.5 Pro进行自动化论文信息抽取与分类，辅以人工校验；利用学术搜索引擎（Semantic Scholar、ACL Anthology）以及Hugging Face API获取论文、数据集与模型信息

**📊 数据集**

主要数据集包括检索自Semantic Scholar与ACL Anthology的论文，使用Ethnologue的语言和人口统计数据、Hugging Face Hub上的公开模型与数据集

**📈 对比分析**

对比方法：按年份、语言与国家统计发布量、主题、贡献类型和模态；计算LIF得分并与各大洲的语言分布进行对照；结果显示虽然论文量显著增长，但主要集中于已被覆盖的语言，新的语言纳入率下降，公平性表现不佳

**⚠️ 局限性**

局限性包括仅检索英文关键词导致非英语论文被低估，闭源会议论文被排除；自动抽取可能受LLM误判影响；依赖Ethnologue和Hugging Face的分类与数据，存在方言、手语和跨国语言未充分覆盖，且论文计数可能高估语言关注度

---

## 81. EgoHand: Monocular 3D Hand Reconstruction from Egocentric Video

**arXiv ID:** 2610.30293 | [PDF](https://arxiv.org/pdf/2610.30293v1)

**作者:** Chao-Yeh Chen `[一作]`, Xiufeng Xie `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

训练并部署了一个基于单目图像的3D手部重建系统，能够在多种摄像机配置下快速生成高质量手姿和形状预测，并提供完整的端到端注释管道。

**💡 创新点**

将相机几何信息（bounding‑box conditioning CLIFF、ray positional encoding）与SOMA‑Hand 3D手模型结合，使用双分支架构（3D重建+2D关键点预测）并通过对应关系细化相机坐标下的手腕位置；统一使用SOMA‑Hand表征并整合约1400万张多来源图像训练。

**🔧 技术方法**

使用SOMA‑Hand + MHR形状后端、DINOv3 ViT‑H/16+编码器、Transformer解码器、CLIFF和RayPE、2D关键点回归分支、对应关系求解、adaptive smoothing、FP16推理以及fisheye校正等技术。

**📊 数据集**

synthetic Omniverse（约49h）、in‑the‑wild egocentric视频（约63h，包括Apple EgoDex、Ropedia Xperience‑10M等）、多摄像机运动捕捉（约16h，Vicon/Noitom/Bones），统一到SOMA‑Hand，合计约1400万张图像。

**📈 对比分析**

与HaMeR、WiLoR、HaWoR、Dyn‑HaMR对比；在H2O上取得最低MPJPE、RTE和加速度误差；在HOT3D、ARCTIC上优于HaMeR/Dyn‑HaMR，Smoothing进一步降低加速度误差；推理速度快（L40 18.35 ms，RTX 6000 9.01 ms）。人类偏好实验显示在EgoDex和Xperience‑10M上显著优于官方注释，在EgoStandard上无明显差距。

**⚠️ 局限性**

单目深度尺度不确定、严重遮挡下性能下降、对手部检测和相机标定高度敏感；未利用多视角或腕部摄像头；需要进一步提升对运动捕捉与真实场景差异的泛化。

---

## 82. GAV: A scientific experimentation environment for PDE agents

**arXiv ID:** 2610.30292 | [PDF](https://arxiv.org/pdf/2610.30292v1)

**作者:** Muheng Li `[一作]` (University of Toronto), Wenlong Mou `[通讯]` (Vector Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于数据的Generator–Agent–Verifier（GAV）环境，允许语言模型代理在生成的PDE问题上迭代实验、获取连续输入、获得精确的数值反馈，并通过此过程改进其程序；

**💡 创新点**

创新点在于：①利用科学数据训练连续参考解分布，自动生成匹配的PDE问题；②采用制造解法（MMS）无需额外数值求解即可得到精确参考解；③提供连续输入查询和基于错误引导的生成与扰动策略；④给出评估可靠性与引导开发的理论保证；

**🔧 技术方法**

技术包括：生成模型（流匹配或GAN）学习参考解分布；制造解法构造对应PDE；Verifier计算精确误差并返回数值反馈；引导生成（基于KL极大化）和扰动策略；LLM代理进行程序迭代；

**📊 数据集**

数据集包括：Navier–Stokes数值仿真与PIV测量数据；Vlasov–Poisson（VP）和Vlasov–Poisson–Fokker–Planck（VPFP）数值数据及ARTEMIS观测；

**📈 对比分析**

与原始的CodePDE、AutoNumerics、PDE‑SHARP等PDE代理在Navier–Stokes任务上比较，GAV降低了平均测试误差超过十倍，并在有限的开发预算内显著缩短了活跃时间；在Vlasov–Poisson任务中亦实现了数十%到九十%以上的误差下降；

**⚠️ 局限性**

局限性包括：需要足够丰富且代表性强的科学数据来训练参考分布；过度依赖生成器分布的质量可能导致过拟合；理论保证基于若干理想化假设；生成器训练与MMS构造在大规模问题上仍有计算成本；

---

## 83. Verified DRAM Address Mapping Reverse Engineering with Threshold Sweep

**arXiv ID:** 2610.30397 | [PDF](https://arxiv.org/pdf/2610.30397v1)

**作者:** Hewen Zhang `[一作]` (University of California), Ian G. Harris `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套端到端的软件流程，用于在嵌入式 LPDDR5 平台上逆向并验证 DRAM 地址映射函数。

**💡 创新点**

通过阈值扫描结合 Knock-Knock 的线性代数解法，同时记录每个阈值对应的映射结果和验证指标，提供了比以往更系统且无硬件依赖的验证方法。

**🔧 技术方法**

采用时延测量、阈值划分、差分矩阵构造、GF(2) 零空间求解、RREF 归约，以及两阶段验证（组验证和单函数验证）等技术。

**📊 数据集**

采集了约 10,000,000 条 CPU 时钟周期级别的访问时延样本，作为实验和阈值选择的基础。

**📈 对比分析**

在 Jetson Orin Nano 与 Raspberry Pi 4B 上测试，组验证通过率在最优阈值下接近 100%，单函数验证进一步证明每个映射函数的独立有效性，验证准确率显著优于传统穷举搜索。

**⚠️ 局限性**

仍依赖大量时延采样，阈值选择对结果影响大；仅在 ARM‑LPDDR5/LPDDR4 设备上验证，尚未扩展到更广泛的 DRAM 规格和多种硬件平台。

---

## 84. The Community Archive: An Ethically Sourced X (Twitter) Dataset

**arXiv ID:** 2610.30392 | [PDF](https://arxiv.org/pdf/2610.30392v1)

**作者:** Brandon Duderstadt `[一作]` (Calcifer Computing), Francisco Carvalho `[通讯]` (Epistemic Garden)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

创建了一个包含530多万条用户自愿公开的推文及其元数据和网络关系的“社区档案”数据集，并利用该数据集演示了内容同质性与词汇创新传播的分析。

**💡 创新点**

首次通过完全自愿同意采集方式公开了内容、元数据和社交网络结构，构建了大规模合法且伦理的推特数据集，解决了以往数据收集的法律与伦理障碍。

**🔧 技术方法**

采用Nomic Atlas主题模型、Bhattacharyya系数构建内容相似图、核心‑外部结构模型与bootstrap假设检验、以及Mann‑Whitney U检验等技术进行内容同质性与创新传播分析。

**📊 数据集**

使用自建的5,374,594条推文的社区档案数据集，包含推文文本、时间戳、点赞数、关注关系等信息。

**📈 对比分析**

将内容相似图与关注网络的随机块模型进行对比，利用log‑likelihood bootstrap得到p=0，表明聚合内容同质性不成立；对词汇创新的时间分布进行Mann‑Whitney U检验，p=6e-13，证明核心用户先行采用新词。

**⚠️ 局限性**

样本偏倚与代表性不足（自愿采集导致的结构与主题偏差）、时间解析受异步上传影响、规模仍逊色于无同意数据集、核心‑外部模型假设缺乏方差校正。

---

## 85. Mirage: Targeted Jailbreaks Against Post-Generation Safety Checkers in Multimodal Diffusion Models

**arXiv ID:** 2610.30348 | [PDF](https://arxiv.org/pdf/2610.30348v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 86. Text-Only LLM Correctors Break Correct Transcripts: A Repair-Damage Analysis of Audio Grounding in ASR Error Correction

**arXiv ID:** 2610.30385 | [PDF](https://arxiv.org/pdf/2610.30385v1)

**作者:** Vasiliy Kudryavtsev `[一作]`, Grach Mkrtchian `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型（LLM）在语音识别（ASR）后纠正输出的过程进行研究，提出将总体错误率（WER）拆分为修复、破坏和插入三部分，并比较在有音频输入和仅有文本输入时模型表现。

**💡 创新点**

创新点在于：①以词级精确拆分WERE，单独量化修复与破坏对整体性能的贡献；②设计五个对照实验（C、A、T、F、B）实现系统地评估音频引导的影响；③首次将此方法应用于多种语料与多模型，揭示不同情境下音频的正负效应。

**🔧 技术方法**

使用的技术包括：Gemini 2.5 Flash、Voxtral‑Small‑24B、GPT Audio等LLM；对齐与计数实现WERE拆分；针对不同输入设置的prompt工程；配对bootstrap置信区间评估统计显著性。

**📊 数据集**

使用的数据集为：①由Wikipedia文本转成语音的俄语合成语料（3,000句）；②俄语有声书RuLibriSpeech（300句）；③英语医学转录集（300句）。

**📈 对比分析**

比较方法：在同一语料下对比五种设置的ΔWER（音频+/−、文本+/−）及其拆分成分，采用配对差异统计；结果显示：在跨系统设置中，音频输入能提升修复并降低破坏；自校正设置中，音频主要减少破坏并降低错误率波动；但在医学语料中插入量未下降导致净效益有限。

**⚠️ 局限性**

局限性：prompt与音频的耦合导致无法单独归因；模型版本漂移与API限制；合成语料可能与LLM预训练重叠；只评估了少数几款LLM，未涉及训练专门的纠正器；在自然语料中未充分验证音频真确性带来的差异。

---

## 87. Foundation4D: Towards a unified perception and reconstruction model for autonomous vehicles

**arXiv ID:** 2610.30340 | [PDF](https://arxiv.org/pdf/2610.30340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 88. Data-Void Exploits in Web-Searching LLMs

**arXiv ID:** 2610.30365 | [PDF](https://arxiv.org/pdf/2610.30365v1)

**作者:** Michele Guerra `[一作]` (New York University Abu Dhabi), Azza Abouzied `[通讯]` (New York University Abu Dhabi)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性实验验证了在web搜索大型语言模型（LLM）中通过单一攻击者控制的网页填充“数据空洞”进行操纵的可行性，并量化了不同策略、可见度和生成引擎优化（GEO）对攻击成功率的影响。

**💡 创新点**

首次将数据空洞概念迁移至LLM环境，揭示了策略和可见度是决定攻击成功的关键因素，且用户提示语和模型选择对防护效果有限，为研究LLM安全提供了新的视角。

**🔧 技术方法**

利用生成式AI（ChatGPT-5.2）创建网页，采用5种数据空洞策略、3种可见度水平、5项GEO因子；对三大web‑search LLM（ChatGPT、Gemini、Sonar Pro）进行提示语（中性、兴奋、怀疑）实验；使用多变量逻辑回归和Wald检验评估因素影响。

**📊 数据集**

共构造56个数据空洞实例（覆盖商业、健康、法律、政治、技术5个领域），每个实例生成一页网页；结合3种LLM和3种提示语产生504条回应；并使用BrowseComp基准对GEO因子进行回归分析。

**📈 对比分析**

通过二元逻辑回归和Wald检验比较不同因素对攻击成功率的贡献，发现可见度与策略显著提升成功率（最高OR≈17.5），而用户提示语和模型选择影响不大；整体成功率为31%（高可见度条件下为42%）。

**⚠️ 局限性**

样本规模有限、GEO因素样本不均衡；仅测试单页攻击，未考虑多页面协同；可见度指标综合多因素，难以单独解释；未在信息丰富情境下评估相对攻击效果。

---

## 89. Consistency Is Not Validity: Self-Rewarded Moral Argumentation Makes Language Models Decisive but Pliable

**arXiv ID:** 2610.30349 | [PDF](https://arxiv.org/pdf/2610.30349v1)

**作者:** Marius De Kuthy Meurers `[一作]`, Bernd Bohnet `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一种基于奖励的语言模型对齐方法（GRPO），通过微调模型使其在伦理推理任务中更一致地使用特定语言特征并给出连贯判断。

**💡 创新点**

创新点在于设计了专门的奖励信号以及损失遮蔽与叙事 grounding 的组合，用以精细调节模型的语义特征（如必要性模态、因果连词、连接词等），从而显著提升在不同表述顺序下的回答一致性，同时保持一定的多样性。

**🔧 技术方法**

技术上使用了强化学习（RL‑from‑Reward‑Model）、奖励模型训练、对抗性微调、特征聚类与置信区间估计、以及对生成文本的句法与语义特征分析。

**📊 数据集**

使用了多种伦理推理基准数据集，包括 MoralChoice、ELEPHANT、MORABLES 等，并在内部构建的五个伦理框架提示集合上进行训练。

**📈 对比分析**

对比方法包括与无奖励基线（Base）以及仅依赖四示例的 ICL‑Targeted 进行比较；评估指标包括对表述重排的稳健性（reordering flip）、道德判断准确率、谦逊度、情境识别、以及对 framing 的接受度。GRPO 在一致性方面显著优于基线，但在道德判断准确率上并未提升，且与 ICL 在部分指标上相近。

**⚠️ 局限性**

局限性包括：对齐仅改善了回答的一致性而未提升道德判断的正确率；对奖励函数和特征选择高度敏感；在某些框架下仍存在较高的 sycophancy；对更复杂或极端伦理情境的评估不足；以及对训练集分布的潜在过拟合风险。

---

## 90. AQL-Lite: Enabling Efficient Knowledge Acquisition for Arabic LLMs via Data-Centric Continual Pretraining

**arXiv ID:** 2610.30344 | [PDF](https://arxiv.org/pdf/2610.30344v1)

**作者:** Houssam Eddine Boukhalfa `[一作]` (National School of Artificial Intelligence), Mourad Dahmane `[通讯]` (Nojoom AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 AQL-Lite 框架，包含质量优先的阿拉伯语数据清洗管线和部分参数连续预训练方法，旨在高效地将多语言 LLM 适配到现代标准阿拉伯语（MSA）并显著降低训练成本。

**💡 创新点**

创新点在于将多步骤的质量管线（结构清理、启发式过滤、MSA 选择、模型评分与去重）与 RS-LoRA 结合，并在训练时解冻最后 8 层 Transformer 及嵌入矩阵，同时采用层级学习率，既保留了全量预训练的效果，又将计算量大幅压缩。

**🔧 技术方法**

技术方法包括：结构化清洗、启发式特征过滤、MSA 分类器、基于模型的安全与质量评分、精确与近似去重；RS-LoRA 低秩适配器、全层冻结/解冻策略、嵌入矩阵训练、分层学习率调度、DeepSpeed ZeRO‑2 训练加速等。

**📊 数据集**

使用数据集：起始为 885 B 阿拉伯语 Common Crawl，经过 AQL 过滤后得到 215 B 高质量 MSA 语料；随后使用 40 B MSA 语料进行预训练，1 M 指令样本进行指令调优；评测基准为 Open Arabic LLM Leaderboard v2（OALL v2）七大测试集。

**📈 对比分析**

在对照实验中，AQL-Lite 在 5 B 令牌预算下，仅用 63 h 训练即可获得平均 73.19 分（比基线 66.91 分提升 6.28 分），相较于全量预训练 121 h 仅提升 0.39 分，恢复 94% 的全量收益；在 40 B 规模下，12 B 模型在 OALL v2 上平均 75.3 分，处于中上水平。

**⚠️ 局限性**

局限性包括：仅针对 MSA，方言、代码混杂覆盖不足；部分参数方案未与 40 B 全量预训练直接对比；未评估技术文本、英语能力、指令多样性等；安全性与偏见仍需进一步审计。

---

## 91. The Customer Is the Builder: How AI Coding Agents Are Rewriting the Economics of Software

**arXiv ID:** 2610.30307 | [PDF](https://arxiv.org/pdf/2610.30307v1)

**作者:** Ripon Chandra Malo `[一作]` (University of Utah), Tong Qiu `[通讯]` (University of Utah)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过定义软件替代前沿（software substitution frontier），系统评估了客户在软件购买与自行构建之间的经济决策，并追踪了AI编程代理（coding agents）在构建能力、公共软件产出、商业产品替代品出现以及组织购买行为方面的变化。

**💡 创新点**

创新点在于：①提出并量化了软件替代前沿概念；②将代理能力提升、公共软件创作与市场反应三条指标串联成完整证据链；③通过多源公开数据与调查，首次检验了AI代理对企业内购决策的潜在冲击。

**🔧 技术方法**

主要技术包括：编程代理能力基准（METR、Epoch、FrontierSWE、Vibe Code Bench、Aider等），时间序列分段回归分析，GitHub与包管理器数据抓取，Show HN、Stack Overflow、BEA、Homebrew、自由职业平台（Upwork/Fiverr）等多源数据的融合。

**📊 数据集**

使用的数据集包括：GitHub公共仓库、PyPI/Crates.io/RubyGems/Packagist包注册表、Show HN帖子、Stack Overflow问答、BEA软件投资与安装统计、Homebrew安装记录、McKinsey组织调查、Freelance市场季度报告等。

**📈 对比分析**

方法：通过分段回归评估代理能力增长对新包创作率的影响，并将代理能力预测与公共软件产出与市场指标（安装、采购、调查自报）进行对比。性能方面显示：代理任务长度在每90天翻倍；2025年2月后新包创作速率显著加速；但市场采购指标仍无明显下降，表明替代尚未成形。

**⚠️ 局限性**

局限性：①缺乏直接将内部构建与采购取消关联的面板数据；②公开仓库与安装量仅捕捉可见替代，无法衡量私有构建或AI订阅对支付行为的影响；③样本群体（GitHub贡献者、Stack Overflow自选调查）不具代表性；④分段回归和趋势预测易受宏观经济、招聘变化等混杂因素影响；⑤对替代品的功能符合度与可靠性仍需独立验证。

---

## 92. From Prose to Properties: Testing Research Claims with ARATest for Agent-Native Science

**arXiv ID:** 2610.30495 | [PDF](https://arxiv.org/pdf/2610.30495v1)

**作者:** Yifeng He `[一作]` (University of California, Davis), Hao Chen `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出将科研主张抽象为可执行的属性，并实现了声明绑定、自动化验证与持续测试框架，用于高效检查研究主张的有效性。

**💡 创新点**

创新点在于：①使用“声明+属性骨架”固定主张语义，使同一声明可在报告证据与新实验间复用；②引入三值判定、可重放违例和自动实验生成器；③在大量论文数据上演示该方法比传统LLM评判更快、更精准。

**🔧 技术方法**

采用了属性化测试（QuickCheck风格）、类型化声明与约束、Python库、OpenHands实验生成器、LLM审稿、数值精度门限、持续测试服务等技术。

**📊 数据集**

主要使用PaperBench、RE-Bench、NanoGPT等公开论文与实验数据，涉及的表格数据集包括ImageNet‑v2、AQuA‑RAT、SVAMP等。

**📈 对比分析**

在已注入缺陷的表格数据上，与直接LLM判定和带计算器的LLM判定比较，确定性检查器检测率为100%，LLM仅61.8%/57.7%；且每次检查约耗时为LLM的1/9至1/10；在真实实验的持续测试中，能够发现报告证据无法显示的行为，并提供可重放违例。

**⚠️ 局限性**

局限性包括：只能覆盖可用属性骨架表达的主张，无法验证定理、相关性等；声明依赖LLM生成并人工审核，可能遗漏或误绑定；自动实验生成器受限于可执行代码的可用性；对复杂统计置信区间处理有限；结果高度依赖声明的准确性；未对新论文的覆盖率进行独立评估。

---

## 93. Optimal Multitask Representations Under a Rate Constraint: An Information Bottleneck Analysis

**arXiv ID:** 2610.30493 | [PDF](https://arxiv.org/pdf/2610.30493v1)

**作者:** Nicholas Stiles `[一作]` (Massachusetts Institute of Technology), Gregory W. Wornell `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在有限信息预算下，不同预训练目标（如重建、专门属性预测）对多任务下游任务的表示效果，并用信息瓶颈理论分析其比特率随性能的交叉点。

**💡 创新点**

创新点包括：①在多任务信息瓶颈框架下证明专家（专门）与通用（重建）预训练目标随比特率的性能交叉；②给出任务频率与精度导致的优先级反转，并在任意比特率下构造与多任务最优解匹配的单任务预训练目标；③提出在任务SNR受限时，平均任务信息矩阵即可保证相对后悔上界。

**🔧 技术方法**

采用多任务信息瓶颈理论、Gaussian信息瓶颈解析、相对后悔分析、任务频率-精度优先级分析；实验中使用高斯解析方案、变分信息瓶颈（VIB）和线性/MLP探测器。

**📊 数据集**

使用CelebA（人脸属性）和dSprites（合成形状）数据集进行实验。

**📈 对比分析**

通过在相同压缩预算（Gaussian码率或离散码本大小）下对不同预训练目标进行任务平均过度负对数似然（NLL）评估，比较其性能。实验表明在低比特率下专家预训练优于重建，高比特率时重建反超；多任务训练中因子可恢复度随比特率出现反转，实验结果与理论预测吻合。

**⚠️ 局限性**

局限性包括：理论基于高斯假设，未考虑非高斯真实数据和模型容量、解码器限制；实验仅在CelebA和dSprites这两类数据上验证，未检验在更广泛任务集上的普适性；未考虑权重量化等实际部署限制。

---

## 94. Short-to-Long: Detached Features for Long-Horizon Value Learning

**arXiv ID:** 2610.30461 | [PDF](https://arxiv.org/pdf/2610.30461v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 95. What the Trace Shows and What the Label Says: Auditing Behavioral Detectors in Agent Evaluation

**arXiv ID:** 2610.30491 | [PDF](https://arxiv.org/pdf/2610.30491v1)

**作者:** Idunnuoluwa A. Adeniji `[一作]` (University of Illinois Chicago), Michael E. Papka `[通讯]` (University of Illinois Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对六款生产级LLM在算术工具使用任务中的执行轨迹进行审计，评估其自动行为检测器的准确性与完整性。

**💡 创新点**

创新点在于提出测量透明度原则，揭示自动检测器在检测“静默覆盖”“未检索”以及“不可见重检查”等行为时的局限，并通过交互式 trace viewer 使检测结果与证据可视化，提升评估可信度。

**🔧 技术方法**

采用了工具注入错误的 GSM8K 算术问题、ReAct 循环框架、三类自动检测器（D-correct、D-verify、D-impact）以及自研的可视化查看器进行实验和审计。

**📊 数据集**

使用了公开的 GSM8K 算术题集，并对六个生产模型（Mistral‑Large‑3‑675B、nemotron‑3‑ultra、claude‑sonnet‑5、claude‑opus‑4.8、openai/gpt‑oss‑120b、openai/gpt‑oss‑20b）进行评测。

**📈 对比分析**

通过自动标签与人工复核对比，发现 D-verify 的一致率仅为 82.7%，其余指标达 100%；实验还展示了不同步骤/令牌限制下的准确率波动，表明执行预算对测量结果有显著影响。

**⚠️ 局限性**

限制包括仅评估算术任务、样本量仅为每模型 30 条轨迹、人工评审为单一评审者、以及仅对两种模型做了执行限制消融，未涵盖更广泛任务与模型。

---

## 96. Resource-Aware Scheduling for Distributed Quantum Computing

**arXiv ID:** 2610.30482 | [PDF](https://arxiv.org/pdf/2610.30482v1)

**作者:** Connor Howe `[一作]` (University of Minnesota-Twin Cities), Ali Anwar `[通讯]` (University of Minnesota-Twin Cities)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种面向分布式量子计算的资源感知调度器，能够在运行时根据队列、通信、重构和硬件特性选择完整执行、单QPU切割或多QPU分布式执行方案；

**💡 创新点**

创新点在于把量子与经典资源统筹调度视为完整的运行时问题，并将通信与重构作为首要调度资源，首次实现端到端延迟与质量的动态决策；

**🔧 技术方法**

使用Qiskit 2.x、AerSimulator、IBM QPU模拟、Circuit Cutting Add-on进行切割、重构，配合自定义的QPU、通信、重构时序模型和硬件驱动的量子时钟预测；

**📊 数据集**

采用合成量子电路集（GHZ、随机CX、QFT、QAOA、VQE）以及对IBM量子后端的模拟，覆盖宽度3–14量子比特、不同噪声水平与QPU容量；

**📈 对比分析**

通过比较三种执行计划的中位数和P90延迟、重构占比、尾部性能等指标，发现分布式切割可将P90延迟降低约39%，而重构约占83%的切割开销；调度开销低于0.04%；

**⚠️ 局限性**

局限包括：基于仿真而非真实硬件，合成工作负载与真实生产流量可能不同，通信与重构模型简化，精度代理与实际应用精度不完全对应；

---

## 97. FIGFLEX: A Mixed-Precision FP-INT GEMM Accelerator Co-Designed with Low-Bit LLM Quantization

**arXiv ID:** 2610.30480 | [PDF](https://arxiv.org/pdf/2610.30480v1)

**作者:** Faraz Tahmasebi `[一作]` (University of California, Irvine), Sitao Huang `[通讯]` (University of California, Irvine)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对LLM权重仅量化的混合精度算法-硬件协同设计，开发了基于radix‑4 FP‑INT的数据路径和INT3优化的处理单元，并配合列缩放与补偿感知的混合精度量化器，实现了低位（3‑bit平均）下的高效推理。

**💡 创新点**

创新点包括①使用radix‑4重编码将乘法与累加统一为一条整数加法树，实现在FP‑INT中对不同位宽的权重进行可扩展的并行处理；②设计了INT3融合高位通道的高速路，使3‑bit权重不再浪费半路宽度；③提出列缩放（per‑channel 归一化）与补偿感知的位宽分配策略，使量化误差最小化并直接映射到硬件。

**🔧 技术方法**

技术手段包括：权重重编码（radix‑4）、整数累加器、共享指数对齐、列缩放归一化、补偿感知的混合精度分配、INT3高位融合、FP‑INT混合精度数据流、硬件层面的多精度可扩展处理单元以及无 dequantization 的后端归一化。

**📊 数据集**

使用的实验数据集包括 WikiText‑2、C4（下游评测）以及多种零样本任务（PIQA、ARC‑easy/挑战、HellaSwag、Winogrande）。

**📈 对比分析**

通过与 GPTQ、AWQ、SliM‑LLM、AMQ、QuIP# 等主流低位量化方法比较，本文在 LLaMA‑2‑7B、Llama‑3.1‑8B 等模型上取得 3‑bit 平均位宽时 WikiText‑2 perplexity 5.85、2‑bit 时 8.77，均优于其他方法；在硬件上，所提出的加速器在 3‑bit 时比 AxCore 和 FIGLUT 的整体吞吐率提高 1.56× 与 1.42×，能耗比其低约 25%。

**⚠️ 局限性**

局限性包括：仅针对权重仅量化的线性 GEMM，未覆盖注意力与激活量化；量化结果基于后训练且未进行微调；对极低位宽（1‑bit）效果仍有限；硬件实现需在 1 GHz 时钟且采用 15 nm 工艺，实际部署时可能受制于工艺与能耗限制。

---

## 98. Integrated Respiratory Sensing and Gated Control for Autonomous Needle Insertion Via CT Bore Robotic Arm

**arXiv ID:** 2610.30452 | [PDF](https://arxiv.org/pdf/2610.30452v1)

**作者:** Emma Huang `[一作]` (University of California San Diego), Michael C. Yip `[通讯]` (University of California San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在CT扫描机内部实现无呼吸暂停的呼吸门控针刺系统，实现针刺过程的全自动化；

**💡 创新点**

创新点在于将呼吸检测、预测与针刺控制集成于CT bore内，实现实时呼吸门控、异常检测与无手动干预的全流程自动化；

**🔧 技术方法**

使用ToF传感器+霍尔效应触摸传感器、扩展卡尔曼滤波预测呼吸周期、领先补偿控制器、正弦/谐波模型与多重阈值异常检测；

**📊 数据集**

利用8名志愿者胸壁运动（OptiTrack标记）、5个病人模拟档案和人工生成的呼吸扰动，构成实验数据集；

**📈 对比分析**

通过与原始传感器测量对比，EKF预测RMSE由0.235mm降低至0.165mm（29.8%误差减少）；在5次完整序列实验中均成功到达40mm目标深度，平均插入时间14.41s，且未违反门控约束；

**⚠️ 局限性**

限制在于试验使用的皮肤与肋骨模型不够真实、需临床前进行病人与工具特定系统辨识、目前的扭矩感知受限于电机硬件、仅在仿真试验中验证，临床应用尚待进一步验证。

---

## 99. Stability Analysis of Signed Graphs under Structural Perturbations

**arXiv ID:** 2610.30444 | [PDF](https://arxiv.org/pdf/2610.30444v1)

**作者:** Ning Zhang `[一作]` (University of Oxford), Li Zhang `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了带符号图过滤器在符号结构扰动下的嵌入稳定性，给出了符号邻接矩阵与符号拉普拉斯滤波器的期望扰动表达式，并基于此设计了分布感知的攻击策略。

**💡 创新点**

首次把节点特征协方差与边的符号扰动相结合，量化符号扰动对嵌入的影响，并提出了理论导向的符号边扰动方法。

**🔧 技术方法**

使用符号图滤波器（符号邻接矩阵与符号拉普拉斯）、概率稳定性分析、期望嵌入扰动公式以及贪心/投影梯度优化来实现攻击。

**📊 数据集**

在两类合成图（含两个社区的符号SBM）和两组真实符号网络（Highland Tribes 与 Bitcoin）上进行实验。

**📈 对比分析**

与随机、度中心性与基于失衡度的基线方法对比；实验表明理论导向攻击在嵌入漂移、极性变化与链接预测AUC下降方面均优于基线，尤其在大扰动预算下差距更大。

**⚠️ 局限性**

仅针对一层符号图滤波器，未考虑深层符号GNN；攻击方法需要先验的节点特征分布，且在极度稀疏或非对称符号网络中的效果尚待验证。

---

## 100. Secure Agentic AI and Multimodal Intelligence in 6G Communication Systems

**arXiv ID:** 2610.30436 | [PDF](https://arxiv.org/pdf/2610.30436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 101. SpecMAGIC: Specification-Guided Multi-Task Generalization in Multi-Agent Reinforcement Learning

**arXiv ID:** 2610.30443 | [PDF](https://arxiv.org/pdf/2610.30443v1)

**作者:** Zijian Guo `[一作]` (Boston University), Wenchao Li `[通讯]` (Boston University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了SpecMAGIC框架，能够在多任务多智能体强化学习中实现零射击泛化，满足复杂的LTL规范和多智能体协调约束。

**💡 创新点**

创新点包括：① 用LTL结合依赖与计数约束的新规范语言；② 基于reach-avoid子任务的采样式高层规划，具备可达性与概率完整性保证；③ 将高层规划与低层子任务条件RL分离，实现零射击泛化。

**🔧 技术方法**

技术手段：Büchi自动机、子任务条件强化学习（IPPO+GenZ-LTL）、采样式高层规划、LTL与算子翻译、分层决策。

**📊 数据集**

数据集与环境：基于Multi-Agent Particle Environment的自定义导航环境，随机生成多种有限/无限、依赖/计数的LTL规范，用于训练与评估。

**📈 对比分析**

比较方法：与ACC-MARL基线对比，SpecMAGIC在离散和离群规格下成功率更高、违约率更低、规划时间更短；在10名智能体场景中仍保持优秀表现，显示良好可扩展性。

**⚠️ 局限性**

limitations: 高层规划仍为集中式，规模进一步增大时可扩展性受限；低层策略对复杂安全约束的处理仍有限，需要进一步改进。

---

## 102. Not Where You Think: Protecting Location Privacy Against Reasoning-Based Geolocation

**arXiv ID:** 2610.30439 | [PDF](https://arxiv.org/pdf/2610.30439v1)

**作者:** Shanghao Shi `[一作]` (Washington University in St. Louis), Ning Zhang `[通讯]` (Washington University in St. Louis)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了大规模多模态推理模型对普通照片的位置信息泄露风险，构建了结构化攻击 GeoTrace 并提出了主动防御 GeoShift，能够在不损失视觉质量的前提下显著降低位置泄露概率。

**💡 创新点**

①首次系统揭示视觉线索链与粗细级地理推理机制；②设计基于场景兼容的伪位置与局部语义编辑的主动防御，突破传统基于去除元数据或全局扰动的方法。

**🔧 技术方法**

采用提示式结构化推理攻击、分层粗细地理搜索、CLIP 评估、图像编辑器（生成式编辑与水印）、Web 搜索与多模型对比等技术。

**📊 数据集**

使用 525 张全球普通照片的 benchmark，来源于 IM2GPS3K、YFCC26K、DoxBench、Mapillary 等，并通过 CLIP 过滤去除地标和元数据。

**📈 对比分析**

与直接提示、Web+提示、GeoMiner 等基线对比，GeoTrace 在 Gemini-3-Flash 上街道级精度为 25.71%，GPT-5-mini 为 16.95%；GeoShift 将街道级误差从 16.95% 降至 2.48%（Gemini）或 5.52%（Qwen），同时保持 PSNR ≥ 19 dB 与 CLIP 相似度 > 0.97。

**⚠️ 局限性**

对不可替换的地标或冲突提示仍能被识别，适应性攻击（视觉匹配、Top‑k 候选、冲突剪枝）可部分提升精度；防御过程仍存在一定的成本与延迟，且多图协同保护机制尚未完善。

---

## 103. Tactile-Based Robotic Manipulation: From Mechanics-Driven to Data-Driven Perspectives

**arXiv ID:** 2610.30438 | [PDF](https://arxiv.org/pdf/2610.30438v1)

**作者:** Jingyi Xiang `[一作]` (University of Illinois Urbana-Champaign), Wenzhen Yuan `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了触觉感知在机器人操控中的发展，系统梳理了从力学驱动到数据驱动两大范式，并提出未来研究方向

**💡 创新点**

提供了触觉感知与机器人学习结合的统一视角，指出两种方法的互补性，阐述了关键挑战与研究机会

**🔧 技术方法**

回顾了力学建模、特征提取、监督学习等经典方法，以及模仿学习、强化学习、视觉-语言-动作模型、预测模型等最新数据驱动技术

**📊 数据集**

本篇为综述性质，并未采用特定实验数据集

**📈 对比分析**

无实验对比，文章以文献综述方式说明各方法的优势与局限，未给出性能指标

**⚠️ 局限性**

受限于综述性质，缺乏统一评测基准；触觉传感器多样化导致可迁移性差；实时性与低维表示挑战；未来需要更系统的跨传感器评估与标准化

---

## 104. DiffLTM: Continuous Diffusions Large Tabular Models for Conditional Generation

**arXiv ID:** 2610.30472 | [PDF](https://arxiv.org/pdf/2610.30472v1)

**作者:** Alankrita Bhatt `[一作]` (Granica Computing Inc), Pulkit Tandon `[通讯]` (Granica Computing Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文设计了一种纯连续扩散模型 DiffLTM，用于高质量生成表格数据，并支持等值、范围和线性三类条件生成，特别关注稀有子群。

**💡 创新点**

创新点在于证明交叉熵损失对分类列更优、引入了关注稀有子群的 Conditional Shape 指标，以及将 CFG 与 SMC 推断方法扩展到多种条件；模型还能在经济计量任务中重现经典因果与教育回报模式。

**🔧 技术方法**

采用的技术包括纯连续扩散（probability flow ODE）、学习式嵌入、交叉熵损失、分类器无指导指导 (CFG)、顺序蒙特卡罗 (SMC)、全连接 MLP 以及对数-归一化与多尺度噪声调度。

**📊 数据集**

使用的主要数据集为美国社区调查 PUMS 的 ACSLabour、ACSPubCov、ACSMobility（约 500 万行）以及 CPS 2010‑2019 数据用于最低工资政策的因果分析。

**📈 对比分析**

模型在 Shape、Trend、Conditional Shape、MLE、DCR Ratio 等度量上与 TabRep、TabCascade、TabDiff 等方法对比，DiffLTM 在大多数指标上名列前茅；在条件生成方面与 CFG/SMC 比较，表现与拒绝采样相当或更优。

**⚠️ 局限性**

局限性包括未实现跨数据集建模、缺乏严格的统计安全与鲁棒性保障，以及理论假设（如误差未充分建模）下对模型性能的泛化仍待进一步研究。

---

## 105. Why Bother Me Again? Benchmarking Personal Agent Memory for Interaction Preferences

**arXiv ID:** 2610.30485 | [PDF](https://arxiv.org/pdf/2610.30485v1)

**作者:** Jialin Li `[一作]` (New York University Abu Dhabi), Hanan Salam `[通讯]` (New York University Abu Dhabi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并发布 MemPABench，提供一个可执行的长期交互环境，用于评估个人代理（PA）在记忆系统支持下学习并应用用户交互偏好（如确认、主动性、信息流等）的能力。

**💡 创新点**

①将交互偏好形式化为基于用户、上下文、属性和时间的状态；②设计长时序、持久历史的测试场景，涵盖工作/个人两种上下文及偏好随时间变化；③提出一套完整的评价指标（PSA、CS、PET、BRS、PRS），能够细粒度评估偏好获取、选择和实现。

**🔧 技术方法**

使用多种大语言模型（GPT‑5.6‑Sol、Qwen3.8‑27B、HY4‑Preview、DeepSeek‑V4‑Flash）作为代理策略；四种记忆系统（File、Simple RAG、Mem0、MemOS）；用户模拟器与交互包装（IX Wrapper）；LLM 判别器（Gemini‑3.8‑Flash、Claude Sonnet 5）进行行为与感知评分。

**📊 数据集**

自制的“TV‑Persona”数据集：四个源自《生活大爆炸》《办公室》的用户角色，约 99 个长期会话（共 4 个月），14 个交互属性，涵盖工作/个人两种上下文，共 112 个偏好单元，包含 5 次手工触发的偏好变化事件。

**📈 对比分析**

通过对 4 个记忆系统和 4 个 LLM 基座的全组合实验，利用上述评价指标进行比较。结果显示：File 记忆在偏好选择（PSA）和交互表现（BRS/PRS）上略有提升（约 10 分的准确率提升），但 Retrieval‑based 系统（Simple RAG、Mem0、MemOS）并未超过无记忆 baseline；在上下文敏感性（CS）和偏好演变追踪（PET）方面均保持低水平；Oracle 条件下，模型仍难以完全遵循指定偏好，说明默认交互习惯难以被覆盖。

**⚠️ 局限性**

①记忆系统难以检索并应用交互偏好，检索瓶颈突出；②模型受默认交互先验影响，难以被记忆覆盖；③评估仅基于四个角色，缺乏更广泛人群与多样场景的验证；④实验中使用的偏好更侧重“工具使用”而非自然语言交互细节，限制了普适性。

---

## 106. Learning to Incentivise: Using Reinforcement Learning for Sustainable Urban Mobility

**arXiv ID:** 2610.30469 | [PDF](https://arxiv.org/pdf/2610.30469v1)

**作者:** Germán Pardo-González `[一作]` (London School of Economics and Political Science), Claudio Roncoli `[通讯]` (KU Leuven)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过多智能体强化学习动态分配激励，降低交通总时长与CO2排放

**💡 创新点**

首次将多目标奖励函数与排放估计结合，实现动态激励下的旅行时间与排放双重优化

**🔧 技术方法**

多智能体Q学习、MDP建模、SUMO交通仿真

**📊 数据集**

Helsinki Kamppi区域SUMO模拟数据（235条边、152节点、1100 O‑D对）

**📈 对比分析**

与SUMO基准动态UE进行对比，最多预算下TTT下降16%，CO2下降9%，并绘制Pareto前沿展示权重调整效果

**⚠️ 局限性**

未考虑驾驶员遵从率，仅使用路径级激励，未试验链接激励

---

## 107. Integrated Longitudinal and Lateral Control for High-Speed Platooning of Articulated Vehicles

**arXiv ID:** 2610.30437 | [PDF](https://arxiv.org/pdf/2610.30437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 108. Polynomial pathwidth for graphs vertex-coverable by few shortest paths and algorithms for quasi-shortest path covers

**arXiv ID:** 2610.30458 | [PDF](https://arxiv.org/pdf/2610.30458v1)

**作者:** Li-Hsuan Chen `[一作]` (National Chung Cheng University), Peter Rossmanith `[通讯]` (RWTH Aachen University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在图中用少量最短或近似最短路径/走廊覆盖顶点或边时对树宽和路径宽的上界与下界，给出了多项新的多项式界限；

**💡 创新点**

首次证明顶点被 k 条最短路径覆盖即可保证路径宽 O(k³)，并将边覆盖的路径宽上界从 O(k⁴) 降至 O(k²logk)，同时提供 Ω(klogk) 的下界；

**🔧 技术方法**

核心技术包括度量增长、树宽与路径宽的关系、Campbell 等人的线性度量增长工具、Dumas 等人的 well‑colored 路径构造、Baste 等人的路径宽模量化与树分解变换，以及使用 Blocked‑CMSO 逻辑的 XP 算法；

**📊 数据集**

论文基于理论构造，无实验数据集；

**📈 对比分析**

与此前仅有指数上界或更高阶多项式界限的工作相比，新上界在理论复杂度上显著降低，证明了在最短路径覆盖问题中多项式界限的可行性；

**⚠️ 局限性**

局限性在于仍未达到最优的 O(klogk) 或更低上界，且 XP 算法的指数系数未给出明确的 FPT 结果，关于路径宽的下界也仅是 Ω(klogk) ；

---

## 109. Are Learned Vascular Tree Generators Better Than Conditional Resampling?

**arXiv ID:** 2610.30449 | [PDF](https://arxiv.org/pdf/2610.30449v1)

**作者:** Reza Akbarian Bafghi `[一作]` (University of Colorado), Maziar Raissi `[通讯]` (University of California)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文评估合成血管树模型的保真度，并提出分支记录近似复制率（near‑copy rate）作为检测训练数据重用的诊断指标；通过将无参数条件重采样器（resampler）与三种学习生成器（VesselGPT、VesselVAE、PartVessel）在ImageCAS冠状动脉和IntrA脑动脉数据集上进行对比。

**💡 创新点**

创新点在于：1) 引入分支记录近似复制率作为新的复制检测度量；2) 通过无参数重采样器作为基准，展示保真度评分无法区分复制与创新；3) 设计完整的评估协议，包括训练拷贝、噪声扰动拷贝和真实验证树等对照。

**🔧 技术方法**

使用的技术包括：无参数条件重采样器（nearest‑neighbour bootstrap）生成树；两种保真度评分：整体树特征的1‑NN评分与分支依赖AUC；分支记录近似复制率统计；对比统计（配对自举、Holm校正）。

**📊 数据集**

使用的数据集为：ImageCAS公共CT血管图像数据集（冠状动脉中心线树）和IntrA公共脑动脉中心线树；共计546名患者的冠状动脉树（1092棵）和72名患者的脑动脉树（785棵）。

**📈 对比分析**

比较方法：在相同患者划分、相同评估管线下，将三种学习生成器与无参数重采样器、训练拷贝、噪声扰动拷贝和真实验证树进行保真度评分和复制率比较。结果显示：无参数重采样器在两种保真度评分上均优于重训练的学习生成器；其分支记录近似复制率约为50%，而学习生成器低于5%；训练拷贝得到最佳保真度评分；在匹配相同复制率下，学习生成器并未明显优于重采样器。

**⚠️ 局限性**

局限性包括：1) 训练生成器在本文实验中未达到原作者的完整训练周期，可能低估其性能；2) 评估集与重采样器设计同源，可能偏向重采样器；3) 复制率指标只捕捉分支记录层面的重用，无法检测更复杂的重用形式；4) 仅使用两组小型数据集，结果缺乏普适性；5) 评估仅基于保真度评分，未考虑多样性或其他临床指标。

---

## 110. Towards Faithful Data Selection: Statistical Limits and Scalable Algorithms

**arXiv ID:** 2610.30442 | [PDF](https://arxiv.org/pdf/2610.30442v1)

**作者:** Ahmed Mehdi Inane `[一作]` (Université de Montréal, Mila), Ioannis Mitliagkas `[通讯]` (Université de Montréal, Mila)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于数据子集选择的信仰度量与梯度轨迹差异的算法，并给出局部搜索改进方法。

**💡 创新点**

将选择误差的统计上限与“faithfulness”概念结合，引入可估计的梯度轨迹不一致性 G_T 作为选择目标，并证明其能上界 KL 散度。

**🔧 技术方法**

使用无调整朗之万算法（ULA）建模训练动态，推导 G_T 及其与 KL 的关系，并设计交换式局部搜索算法。

**📊 数据集**

在 CIFAR‑10 与 CIFAR‑100 数据集上，用 ResNet‑18、VGG‑16 与 Inception 三种架构进行实验。

**📈 对比分析**

与 Herding、Moderate Coreset、Facility Location、GraphCut、EL2N、MoSo 等静态基线比较，单检查点 G_T 估计在准确率上与现有方法持平，且可进一步优化基线。

**⚠️ 局限性**

局限性包括：理论仅适用于 ULA，实际训练常用 SGD/Adam；NP‑hard 性导致只能使用近似搜索；单检查点估计可能不适用于更长训练或不同模型；未证明对所有下游任务的鲁棒性。

---

## 111. Memory-Augmented Online Test-Time Adaptation for EEG Motor Imagery via Perturbation Consistency and Confidence Calibration

**arXiv ID:** 2610.30422 | [PDF](https://arxiv.org/pdf/2610.30422v1)

**作者:** Kunkun Zhang `[一作]` (Zhejiang University of Technology), Qianwei Zhou `[通讯]` (Zhejiang University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了基于记忆增强的在线测试时自适应框架 MOPC，用于 EEG 动作意象解码，实现单通道、源无关、单次传递的实时自适应。

**💡 创新点**

创新点在于将动态欧氏校准、随机时间遮掩的扰动一致性正则、联合熵最小化以及置信度校准加权重更新等三种 EEG 相关机制统一到轻量级 BN‑only 更新的单通道 OTTA 框架中。

**🔧 技术方法**

使用 EEGNet 基础网络、5 模型投票集成、FIFO 内存块、动态欧氏对齐、随机时间掩蔽、KL 一致性损失、联合熵损失、置信度校准重加权以及 AdamW 对 BN 参数的单步更新。

**📊 数据集**

实验数据集为 BCI Competition IV IIb（9 名受试者，3 通道）和 OpenBMI（54 名受试者，20 通道），并在 BCI IV 2a（四类）上做了初步验证。

**📈 对比分析**

与直接迁移、无监督域适应（DANN、SHOT）和多种测试时自适应基线（TENT、CoTTA、T‑Time、DPLOT）进行对比，MOPC 在 IIb 上提升 2.22%，在 OpenBMI 上提升 3.67%，同时保持每试验 40–60 ms 的延迟并显著降低跨受试者方差。

**⚠️ 局限性**

局限性包括仅在二分类 MI 上经过充分验证，四类实验仅为预实验；依赖伪标签质量；未考虑多模态融合；对极端类不平衡或更大模型的扩展尚待进一步研究。

---

## 112. SIM-UT: A User Simulation based Evaluation Framework for Medical Triage Chatbots

**arXiv ID:** 2610.30412 | [PDF](https://arxiv.org/pdf/2610.30412v1)

**作者:** Deepali Mittal `[一作]` (UserTrace), Sofie S Meyer `[通讯]` (Turn.io)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了 SIM-UT 这一基于多轮模拟用户行为的评估框架，用于在三大低中等收入国家的 WhatsApp 健康筛查聊天机器人中识别和修复安全缺陷。

**💡 创新点**

创新点在于将真实用户的人物设定、语言、情境限制与多轮推理相结合，专门检测用户最小化症状导致的去升级（de‑escalation）风险，并通过迭代调优 Prompt 与路由架构来提前修复此类错误。

**🔧 技术方法**

技术实现依赖 LLM‑as‑judge（Claude‑sonnet‑4‑6）对完整对话进行评估，使用 LLM 生成模拟用户对话、Prompt 工程、对话路由与会话隔离，以及结构化错误分析与自动修正。

**📊 数据集**

数据集由从三大部署（南非听力支持、肯尼亚糖尿病/生活方式教练、印度呼吸筛查）收集的真实聊天记录抽取的 37 组情景、15–55 组情景和 43 组情景组成的模拟用户 personas 与场景，覆盖多语言、多方言与多样化社会经济约束。

**📈 对比分析**

评估通过与 MedGemma（医学微调开放模型）以及前沿模型 ChatGPT 与 Claude 进行 5,000+ 轮多语言对话比较，迭代后临床安全性提升至 95–98%，消除了去升级风险，且在三轮或四轮迭代中显著提升了范围遵循、沟通质量和总体准确率。

**⚠️ 局限性**

局限性包括模拟 personas 与场景仍为有限集合，评估准则在迭代中需要多次人工校准，印度部署中评估模型与被测模型同属 Claude，且需在模型更新后持续复核路由与 Prompt 的有效性，最终仍需部署后现场监控以捕捉未覆盖的失败。

---

## 113. TrustedCode: Evaluating Large Language Models for Developing Trusted Code through Automated Program Verification

**arXiv ID:** 2610.30432 | [PDF](https://arxiv.org/pdf/2610.30432v1)

**作者:** Muhammad Danial `[一作]` (University of Greenwich), Muhammad Taimoor Khan `[通讯]` (University of Greenwich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 TrustedCode 框架，利用 LLM 自动将 AI 生成的 Python 代码转换为可验证的 WhyML 并通过迭代自校正循环实现验证。

**💡 创新点**

引入链式思维与自校正循环，系统化识别并分类 LLM 在类型、语法、语义及知识方面的缺口，提升了 Python 代码的可验证性。

**🔧 技术方法**

结合 LangGraph、Why3、WhyML、Anthropic Claude Sonnet 4 等技术实现闭环翻译与验证流程。

**📊 数据集**

在十个涵盖 if‑else、for‑loop、while‑loop 等不同复杂度的 Python 函数集上进行评估。

**📈 对比分析**

相较于单次尝试，平均需 1.5 次至 5.5 次尝试即可完成验证；30% 错误为语法缺口，50% 为知识缺口，整体性能可接受。

**⚠️ 局限性**

受限于 LLM 对 Why3 库和复杂后置条件的认知不足，导致高复杂度程序仍需多轮尝试，自校正循环对错误日志的利用也不够充分。

---

## 114. Tight Bounds for Paging with Time Windows and Joint Replenishment with Varying Opening Costs

**arXiv ID:** 2610.30430 | [PDF](https://arxiv.org/pdf/2610.30430v1)

**作者:** Joseph Seffi Naor `[一作]` (Technion), Tomer Tsachor `[通讯]` (Technion)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

提出了一个 O(log k) 的在线竞争算法，解决了加权页面缓存问题的时间窗口版本（即在每个请求的可服务时间段内完成请求），从而匹配了经典无权版本的最优竞争度。

**💡 创新点**

创新点在于将问题拆解为两阶段：首先构造缓存配置序列 C（由已知的在线 LP 近似产生），其次在线完成 C 的加载计划 D。核心是将完成问题精确映射到带时间变化开销的 Joint Replenishment Problem with Deadlines（JRP‑D），并设计了在预测辅助下的常数竞争在线算法解决该 JRP‑D 变体。

**🔧 技术方法**

主要技术包括：
- 结构化分解（缓存配置 + 加载计划）
- LP 及其 hitting‑set 形式的在线松弛
- 与 JRP‑D 的构造性归约（定义时间相关的仓库开销 o_t）
- 预测增强模型（对每个最早截止的请求给出未来最低开销的下界）
- 贪心批处理 + 预测驱动的增量决策
- 典型的竞争分析技术（按阶段划分、开销归还、指数化开销利用）

**📊 数据集**

该工作是纯理论的，未使用任何实验数据集；所有结果均在数学上给出下界/上界。

**📈 对比分析**

与之前的最佳结果 O(log k log n) 进行对比。新算法在理论竞争度上实现了 O(log k)，从而填补了加权与无权版本之间的缺口，达到与经典无权缓存问题相同的最优竞争度。

**⚠️ 局限性**

主要限制包括：
- 需要预测辅助（对未来开销的下界）才能实现常数竞争，实际可实现性取决于预测的准确性。
- 为了简化分析，假设所有开销 o_t 为 2 的幂，实际实现时需要将所有开销向上取整，导致额外常数因子。
- 归约与 JRP‑D 的结构相对紧凑，但在更一般的延迟/权重模型下可能需要进一步改进。

---

## 115. RoleFlow: Self-Evolving Multi-Role Agentic Collaboration via Structural Adaptation

**arXiv ID:** 2610.30429 | [PDF](https://arxiv.org/pdf/2610.30429v1)

**作者:** Yi Ge `[一作]` (University of Maryland College Park), Furong Huang `[通讯]` (University of Maryland College Park)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自适应多角色代理系统RoleFlow，能够在执行过程中根据反馈动态编辑协作结构并按需合成专业角色；

**💡 创新点**

核心创新是Single-Edit Counterfactual Policy Optimization (SECPO)，通过在相同链状态下对比编辑与不编辑的执行结果来进行单编辑级别的反事实信用分配，从而实现精细化结构学习；

**🔧 技术方法**

使用的技术包括全局路由器（Global Router）生成并修订角色链、基于GPT-4o-mini的冻结执行器、SECPO的反事实优势计算与PPO优化、以及基于返回门控的角色库成长机制；

**📊 数据集**

在七个跨域基准（HotpotQA、2WikiMultiHopQA、MuSiQue、多步推理MATH-500、MBPP、HumanEval、WebShop）上进行实验；

**📈 对比分析**

与四类基线（手工管道、自动工作流搜索、训练执行器、训练协同器）比较，RoleFlow在平均分数上达67.1，超过最强竞争者W4S 4.7分；在所有三项多跳问答基准上分别提升3.2–7.5 F1分，在数学推理、网页交互和代码生成任务上也表现出色；

**⚠️ 局限性**

需要每个任务提供通过或拒绝以及缺陷信息的验证器；推理成本高于无训练管道，并且SECPO训练时每次编辑会额外执行一次链；

---

## 116. Getting a GRIP on Graph-Based Search for Learned Sparse Retrieval

**arXiv ID:** 2610.30425 | [PDF](https://arxiv.org/pdf/2610.30425v1)

**作者:** Vihan Lakshman `[一作]` (MIT CSAIL), Rossano Venturini `[通讯]` (University of Pisa)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于图的稀疏检索系统GRIP，融合倒排索引的优势，显著加速稀疏向量检索。

**💡 创新点**

创新点包括：term‑based entry point seeding、approximate routing via sparse sketching 以及针对稀疏向量的 Sparse PiPNN 构造。

**🔧 技术方法**

采用的技术包括：HNSW 图搜索、稀疏 sketching、倒排索引、向量量化以及对 PiPNN 的改造。

**📊 数据集**

使用的公开数据集为 MS MARCO v1（8.8M passage）和 MS MARCO v2（138M passage），并在 SPLADE‑CoCondenser 与 SPARSEUP 两种稀疏嵌入上进行评估。

**📈 对比分析**

与标准 HNSW 和最新倒排索引 Seismic 对比，GRIP 在 MS MARCO v2 上 90–98% recall@10 的延迟比 Seismic 提升 2.2–3.4×，在 MS MARCO v1 上则提升 1.4–3.2×。

**⚠️ 局限性**

局限性：索引体积大于 Seismic，构建时间相对较长；在极高召回（≥98%）时性能略逊于 Seismic。

---

## 117. When Agentic Retrieval Fails Silently: Framework and Production Evidence from a Creative AI Assistant

**arXiv ID:** 2610.30423 | [PDF](https://arxiv.org/pdf/2610.30423v1)

**作者:** Yan `[一作]` (Adobe), Aditya Reddy Cheruku `[通讯]` (Adobe)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出Layered Agentic Retrieval Assessment (LARA) 框架，用以解决在 agentic AI 助手中检索质量、用户参与与产品成功之间的归因缺失。

**💡 创新点**

创新点在于将检索质量、参与度与产品成功拆分为三层，并通过四字段链路模式实现无模型追加的完整归因。

**🔧 技术方法**

利用日志记录的检索 ID、返回资产列表、消费资产 ID 与源检索 ID 等字段，并计算传统 IR 指标（如 precision@K、NDCG@K）以及新定义的 GPR、ACR、ORF、UOR 等。

**📊 数据集**

实验数据来自 Adobe Express 2026 年 8 月的生产流量，涵盖图库搜索与聊天/编辑器两种交互模式。

**📈 对比分析**

与传统 CTR、重构率等行为代理相比，LARA 能准确区分检索失败与生成改进，实验证明归因错误被显著减少，提升了评估可信度。

**⚠️ 局限性**

局限性包括仅在单一部署验证、缺少链路字段导致部分层指标为推测、缺乏检索意图的基准标注以及对规划路由的未测量等问题。

---

## 118. X-pressing Intimacy: Mental Health Disclosure After Twitter Acquisition

**arXiv ID:** 2610.30413 | [PDF](https://arxiv.org/pdf/2610.30413v1)

**作者:** Olivia Pal `[一作]` (University of Illinois Urbana-Champaign), Koustuv Saha `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在埃隆·马斯克收购Twitter后，平台治理变更对心理健康自我披露的语言特征和亲密度的影响，聚焦焦虑、抑郁、双相、BPD及自杀意念等五类心理健康状况。

**💡 创新点**

创新点在于将情感与认知层面的 LIWC 语言学分析与三层亲密度分类相结合，构建了一个综合评估心理健康自我披露变化的计算框架，并通过对照语料检验变化是否特异于心理健康话语。

**🔧 技术方法**

采用 LIWC2015 对推文进行情感、认知、社交、时态等维度的文本分析；使用 Qwen3‑8B 大模型进行亲密度三分类，辅以 TF‑IDF 近邻检索；并通过 t‑检验、差异‑差异法以及多项式逻辑回归（Benjamini–Hochberg 校正）评估统计显著性。

**📊 数据集**

数据集为 2018‑2023 年 1 月 1 日至 2023‑02‑01 期间收集的英文推文：约 824,000 条涉及焦虑、抑郁、双相、BPD 与自杀意念的自我披露，及 266,000 条作为对照的普通推文。

**📈 对比分析**

通过差异‑差异（ΔΔ）和多项式逻辑回归比较，结果显示收购后情感词下降、未来导向词显著上升；在亲密度方面，BPD 的中度披露显著下降转为低度，而 SI 相关推文高亲密度略有提升；其他条件与对照组差异不显著。

**⚠️ 局限性**

局限包括：关键词检索可能导致自我披露真实性难以确认；未跟踪个体披露轨迹，难以区分用户迁移与行为改变；对照语料低亲密度偏高，限制对比深度；采样时间不对称，后期样本量有限；API 限制及推文删除可能影响数据完整性。

---

## 119. Can LLMs Improve Their Own Harness? A Controlled Study on ARC-AGI-3

**arXiv ID:** 2610.30499 | [PDF](https://arxiv.org/pdf/2610.30499v1)

**作者:** Jens Lehmann `[一作]` (Amazon), Sahar Vahdati `[通讯]` (University of Hannover)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建并自动改进语言模型驱动的任务执行框架（harness），并在ARC‑AGI‑3交互游戏上评估其性能与成本；

**💡 创新点**

创新点在于首次系统性测量“harness适配缺口”，即开发阶段提升与外部测试的差异，并通过多模型多线性实验验证自适配的可靠性；

**🔧 技术方法**

使用固定编码启动器（OpenCode）进行代码生成与修复，结合自我反馈与程序优化技术（如Self‑Refine、PromptBreeder），并采用Relative Human Action Efficiency和推理成本作为评估指标；

**📊 数据集**

数据集为ARC‑AGI‑3：5个开发游戏与20个保留测试游戏，共计25款；

**📈 对比分析**

与人类驱动的高性能 harness 比较，本文平均提升 2.8 分（95% CI [-1.44, 7.35]）且推理成本平均下降约 2.7 美元/局，显示开发阶段收益往往高估真实外部表现；

**⚠️ 局限性**

局限包括仅使用单一 benchmark、仅两轮修订、每条件仅3次实验、可能的训练集泄漏、推理成本差异大以及自动注释的可靠性不作人工验证。

---

## 120. AVAtt: A Free-Viewing Eye-Tracking Dataset and Benchmark for Visual Attention on Paintings

**arXiv ID:** 2610.30481 | [PDF](https://arxiv.org/pdf/2610.30481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 121. EmoFI-TTS: Face-Conditioned Expressive Voice Synthesis with Intensity Modulation

**arXiv ID:** 2610.30477 | [PDF](https://arxiv.org/pdf/2610.30477v1)

**作者:** Yejin Jeon `[一作]` (Mila Quebec Ai Institute), David Ifeoluwa Adelani `[通讯]` (Mila Quebec Ai Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了EmoFI-TTS框架，能够仅利用单张面部图像推断情感类别与强度，并通过参考音频获取说话人音色，生成情感表达丰富的语音；

**💡 创新点**

创新点在于首次将面部情感强度作为条件融入跨模态语音合成，并通过跨说话人训练实现情感与说话人身份的解耦；

**🔧 技术方法**

核心技术包括ResNet-50+情感分类头提取面部情感特征、ECAPA‑TDNN提取音频音色特征、FastSpeech2预训练模型作为语音生成器、以及基于情感强度的残差增强模块；

**📊 数据集**

使用的主要数据集为MEAD平行视听情绪语料库，包含18,270句含三种情感强度的录音与对应面部图像；

**📈 对比分析**

与EmoSpeech（基于标签的情感合成）和FaceTTS（仅面部情感预测）对比，EmoFI-TTS在情感强度准确率、自然度和可懂度上均表现更优，平均情感强度准确率达64.4%，WER仅6.9%，UTMOS 3.14；

**⚠️ 局限性**

限制在于情感识别仅覆盖少数情绪类别，且在极端情感强度上仍有误差，未来需扩展更多情绪与更细粒度强度范围，并验证在更大规模跨说话人设置中的鲁棒性。

---

## 122. Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy

**arXiv ID:** 2610.30474 | [PDF](https://arxiv.org/pdf/2610.30474v1)

**作者:** Riteng Zhang `[一作]` (Purdue University), Xiaoqian Wang `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个新的评价范式，即要求稀疏自编码器（SAE）的特征构成原子可解释基，形成一种语义上可解释的坐标系，并将该范式应用于对数百个不同架构与模型族的SAE进行系统评估。

**💡 创新点**

创新点在于将SAE的最终目标从单一语义性（monosemanticity）转化为原子可解释基的形成，提出了基于主导性（dominance-aware）的评估方法，并通过DAME等指标构建了综合基准框架。

**🔧 技术方法**

技术手段包括：构建稀疏自编码器网络、计算特征主导度、实现DAME与其他相关指标、开发批量评估脚本以及对比分析不同架构的性能。

**📊 数据集**

实验使用了常见的图像与文本基准数据集（如MNIST、CIFAR‑10、ImageNet以及公开的语言语料库），以验证评估框架的通用性。

**📈 对比分析**

在多种模型与架构上对比实验表明，传统指标下表现良好的SAE在主导性评估中往往显示出多义性与冗余，说明原始评价方法存在缺陷；新的评估框架能够精准捕捉这一失效模式。

**⚠️ 局限性**

局限性包括：评估框架依赖于主导度的经验定义，可能对极端稀疏程度的模型不适用；此外，实验覆盖的模型种类虽然广泛，但仍未涵盖所有深度学习领域，未来需要进一步扩展验证。

---

## 123. aggmetrics: Aggregating Classification Performance Measures for Machine Learning

**arXiv ID:** 2610.30456 | [PDF](https://arxiv.org/pdf/2610.30456v1)

**作者:** Ronaldo Cristiano Prati `[一作]` `[通讯]` (Universidade Federal do ABC), Ronaldo Cristiano Prati (Universidade Federal do ABC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个名为aggmetrics的Python包，用于将任何二元分类性能度量（基于混淆矩阵四个计数）扩展到多类别、多标签、序数、成本敏感、软标签和交叉验证场景，并把聚合操作（微观、宏观、实例化、折叠）显式为可选参数，提供度量的结构属性、自动推断和诊断警告；同时实现了完整的测试套件和评分器接口，兼容scikit‑learn；

**💡 创新点**

创新点在于：①把聚合算子拆分为四个统一的操作并与任何度量配合；②为每个度量记录结构属性（可分解性、偏斜不变性、是否依赖真负、方向等），并能从用户自定义公式自动推断；③提供诊断层，自动检测“floor”“micro‑collapse”等误用；④所有功能都在纯Python/NumPy下实现，配合完整的可执行理论测试。

**🔧 技术方法**

技术手段包括：使用NumPy实现指示矩阵与累积阈值矩阵；构造四种聚合器并把它们与二元度量的计数函数结合；利用线性代数求解可分解性、贝叶斯阈值与可达floor；实现诊断器来检查度量与数据的一致性；提供单元测试、文档字符串和示例Notebook。

**📊 数据集**

示例数据主要为合成的一维标签序列（0–499）和其预测偏移1的情况，用于演示聚合、诊断和评分器。论文未在公开真实数据集上进行实验。

**📈 对比分析**

与现有工具（scikit‑learn、torchmetrics、PyCM、yardstick）比较时，aggmetrics能够精确复现scikit‑learn的四种宏/微/实例/折叠平均；通过单元测试验证理论正确性；在小规模示例中运行时间仅为数秒，且覆盖率达100%。论文未给出大规模性能基准，但指出纯Python实现并未针对GPU或分布式做优化。

**⚠️ 局限性**

局限性包括：仅依赖NumPy，缺乏GPU、流式、分布式或稀疏后端；版本仅为0.1.0，尚无成熟的生产部署；没有在大规模标签空间或高维数据上进行性能评估；未来工作计划包括稀疏后端、流式累加器和R实现。

---

## 124. Externalise the Memory: A Pre-Registered Study of Where an Embodied Agent's Experience Should Live

**arXiv ID:** 2610.30478 | [PDF](https://arxiv.org/pdf/2610.30478v1)

**作者:** Cheng-Yin Tu `[一作]` `[通讯]` (Nikon Research Corporation of America), Cheng-Yin Tu (Nikon Research Corporation of America)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过在Isaac Lab单臂到达任务中使用多模态语言模型教师，提取并检索经验记忆，利用LoRA适配器对教师轨迹进行蒸馏，并在不同的推理协议下评估成功率，系统比较了将经验“烘入”权重与在推理时外部检索的效果；

**💡 创新点**

创新之处在于提出了一个受预注册控制的实验框架，能够精确隔离记忆存储子系统的影响，并揭示了经验中“边缘知识”可被烘入权重，而“条件知识”必须外部检索；同时提出了边缘/条件分解准则来解释这一现象；

**🔧 技术方法**

技术手段包括：Gemma‑4‑31B‑it多模态语言模型教师；基于图像锚点的经验检索缓存；LoRA（rank‑16）蒸馏适配器；可组合的KTO微调；以及配对设计的统计检验（McNemar / z检验）；

**📊 数据集**

使用的数据集为：1）Isaac Lab单臂到达任务中生成的2791条训练轨迹（992可达+1799不可达）以及100条评估任务；2）对比实验中的更难双臂位置目标到达任务（L2），提供了额外的训练和评估数据；

**📈 对比分析**

比较方法为在100条评估任务上计算成功率，并使用两比例z检验（主要）和McNemar检验（配对）进行统计；结果显示：在相同权重下加入推理时检索可提升约23–34个百分点的成功率，而将经验烘入权重仅提升约+1个百分点；

**⚠️ 局限性**

局限性包括：数据量和LoRA容量不足导致的检验功效有限；实验仅在单一到达任务上进行，未覆盖接触式操作；仅测试了单轮SFT+KTO的平面蒸馏方案，未探讨更复杂的参数化记忆合并机制。

---

## 125. A Three-Pillar Governance Framework for Safe and Trustworthy AI Agents

**arXiv ID:** 2610.30445 | [PDF](https://arxiv.org/pdf/2610.30445v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 126. Parallel Repetition for k-Player XOR Games

**arXiv ID:** 2610.30455 | [PDF](https://arxiv.org/pdf/2610.30455v1)

**作者:** Mark Braverman `[一作]` (Princeton University), Dor Minzer `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了k-XOR游戏在并行重复中的价值衰减问题，证明只要原游戏价值严格小于1，其n次并行重复的价值会以指数速度衰减；

**💡 创新点**

创新点在于引入“可交换嵌入”概念，并通过BLR变换将任意k-XOR游戏映射为无可交换嵌入的2k-XOR游戏，从而实现一次性、直接的Cauchy–Schwarz证明，统一了对所有k≥2的指数衰减结果；

**🔧 技术方法**

主要技术包括：BLR变换、可交换嵌入与有序矛盾的对应关系、利用Cauchy–Schwarz的张量化推导、以及有序矛盾的迭代“粘合”操作；

**📊 数据集**

论文不涉及实验数据或公开数据集，研究完全基于理论证明；

**📈 对比分析**

由于本工作为理论证明，未给出实验对比；理论上证明了相对于以往仅适用于3-XOR或无Abelian嵌入游戏的结果，已大幅提升适用范围并实现更简洁的证明；

**⚠️ 局限性**

限制在于目前仅适用于有限Abelian群上的k-XOR游戏，且仍需进一步探讨是否能将Cauchy–Schwarz方法推广至更一般的多玩家游戏或非线性谓词。

---

## 127. Same Beliefs, Different Policies: Learning Reusable Representations for Early Stopping in LLM Reasoning

**arXiv ID:** 2610.30450 | [PDF](https://arxiv.org/pdf/2610.30450v1)

**作者:** Jingwei Ni `[一作]` (ETH Zuerich), Mrinmaya Sachan `[通讯]` (ETH Zuerich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 BELT（BELief-based Termination）方法，先通过一个可冻结的隐藏状态探针学习关于当前与未来答案的四种“信念”预测，然后再用这些信念训练线性停止策略，实现对 LLM 推理过程的高效提前终止。

**💡 创新点**

创新点在于把答案预测与停止决策分离：信念预测与 token 价格无关，能够在不重新训练模型的情况下通过重新拟合线性策略或阈值调整来适应不同的准确-成本权衡；同时引入了“Later”和“Same”这两种预测目标，显著提升停止决策质量。

**🔧 技术方法**

技术上使用了冻结的 LLM 隐藏状态、一个小型的多层 Transformer 编码器和四个 sigmoid 读头来预测 Now、End、Later、Same 四个信念；随后对冻结后的特征拟合线性或阈值化的停止策略，支持多种 token 价格。

**📊 数据集**

训练数据主要来自数学问题集合（DAPO-Math-17k、NuminaMath-1.5、DeepScaleR-Preview），在 Qwen3-1.7B 和 OLMo-3.7B-Think 两个后端上进行；评估则覆盖 OOD 数学基准（MATH-500、OlympiadBench、Gaokao、AIME、AMC）、编码基准 LiveCodeBench 以及代理任务 ALFWorld 与 ScienceWorld。

**📈 对比分析**

与多种基线（OS‑Pruner、OS‑Pruner‑1、Thought Calibration、ROM 等）比较，BELT 在 OOD 数学任务上在保持相同准确度的前提下可节省约 40‑50% 生成 token，且在 OLMo-3.7B 上比完整 OS‑Pruner 多节省 5‑6%；在 LiveCodeBench 迁移到编码任务时，BELT 以单一探针即可匹配或超过五个探针的性能；其参数量与训练成本均显著低于直接 utility 训练的控制器。

**⚠️ 局限性**

局限性包括：方法仍依赖于高质量的离线推理轨迹与正确答案标签，且在极端 token 价格或不同类型任务（如安全关键决策）下的鲁棒性尚未验证；此外，信念预测在某些场景下对“Later”“Same”的解释仍不够直观，未来工作需进一步探索更丰富的预测目标和更通用的停止策略。

---

## 128. Proba: A High-Performance, Low-Traffic Probabilistic Spatial Memory Streaming Prefetcher

**arXiv ID:** 2610.30448 | [PDF](https://arxiv.org/pdf/2610.30448v1)

**作者:** Yinting Huang `[一作]` (Huawei Newton Research Centre), Sam Ainsworth `[通讯]` (University of Edinburgh)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了 Proba 预取器，通过概率更新、边际准确性决策以及 Jail Table 采样机制来改进空间预取。

**💡 创新点**

创新点在于用概率方式融合多次观察而非简单 OR/AND，动态调节准确性与覆盖率；引入多特征触发器与边际准确性控制以实现精细化的预取强度；以及通过 Jail Table 与采样降低 AGT 负担，提升模式学习完整性。

**🔧 技术方法**

采用的技术包括：概率模式更新（插入/删除概率）、准确性校准（冗余计数与全局准确率反馈）、边际准确性决策、Bloom‑like Jail Table 区间追踪、随机替换采样 AGT、Tag‑based PHT 归档、BRRIP 替换策略。

**📊 数据集**

使用的评测数据集为 SPECspeed 2017、SPEC CPU2006、Ligra 和 CloudSuite 四大基准集。

**📈 对比分析**

通过在 gem5/ChampSim 仿真环境下与 SMS、Bingo、DSPatch、PMP、Gaze 等现有 SMS 预取器进行单核、双核、四核比较，Proba 在单核下平均提升 6.6% IPC、DRAM 交通仅 +8.5%；在四核场景中提升 7.9% IPC、DRAM 交通显著下降，显示出显著的性能与能效优势。

**⚠️ 局限性**

局限性包括：对极高覆盖率需求的工作负载可能仍有限制；准确性阈值与 PHT 大小对性能影响较大；实现上需要额外的 Jail Table 与 AGT 采样，增加硬件复杂度和设计成本。

---

## 129. Self-Assessment Reinforcement Learning for Search-State Awareness in Retrieval-Augmented Large Language Models

**arXiv ID:** 2610.30435 | [PDF](https://arxiv.org/pdf/2610.30435v1)

**作者:** Taewhoo Lee `[一作]` (Korea University), Jaewoo Kang `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 Self‑Assessment Reinforcement Learning (SARL) 框架，用来训练检索增强大型语言模型在搜索过程中自我意识缺失信息、避免幻觉推理和过早终止搜索。

**💡 创新点**

创新点在于：①把自我评估作为训练内部信号，引入双重奖励（纠错奖励 + 评估频率惩罚）来同时提升效果与效率；②通过评估奖励细化模型对搜索轨迹中答案变化的感知；③在 RL 训练中不依赖额外监督或多代理协同，保持单一策略即可完成自我校正。

**🔧 技术方法**

主要技术：检索增强生成（RAG）+ 交互式搜索；强化学习（GRPO）；自评估阶段；双重奖励设计；无监督训练；对话式评估提示。

**📊 数据集**

训练数据：共 45K 条，分为 15K Natural Questions、15K HotpotQA、15K ASearcher（单跳、双跳、深度研究）。评测数据包括 10 个基准：NQ、TriviaQA、PopQA、HotpotQA、2WikiMultihopQA、Musique、Bamboogle、GAIA、FRAMES、BrowseComp‑Plus。

**📈 对比分析**

与多种基线对比：无检索、RAG、搜索代理（Search‑R1、AutoRefine、ReSeek）以及公开的两阶段 SFT+RL、MA‑RAG 等。SARL 在所有基准上平均提升 24.3%（相对），在多跳推理上 35.9% 以上，在深度研究任务上 27.9% 以上；在与 RAG‑R1、MA‑RAG 的比较中，SARL 也显示更高的平均分和更强的跨模型泛化。

**⚠️ 局限性**

局限性：①仍需大量中间搜索轨迹数据，训练成本较高；②对 α（惩罚系数）和 λ（奖励系数）的超参数敏感；③在极长或极复杂搜索任务中，模型仍可能出现训练不稳定或过度依赖评估；④评估阶段仅在训练时使用，推理时不再进行自评，若推理环境与训练差异大，性能可能下降。

---

## 130. Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems

**arXiv ID:** 2610.30418 | [PDF](https://arxiv.org/pdf/2610.30418v1)

**作者:** Daniel Ruiz `[一作]`, Emilio Ferrara `[通讯]` (University of Southern California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究LLM代理在检索和合成文档时，如何泄露被禁止的组合信息，并在不同查询意图（benign、explicit、adversarial）以及可见分类头和聚合规则等保护措施下量化泄露率。

**💡 创新点**

提出合成的多领域（网络安全、医疗、军事）基准，并系统评估三大LLM在各种配置下的“mosaic disclosure”，首次评估可见分类头与聚合规则对泄露率的缓解效果。

**🔧 技术方法**

使用三大LLM（GPT‑5.6‑Sol、Claude Sonnet 5、Qwen3.5‑27B），模拟与Pi/Hermes开放源代码代理 harness，基于人类标注的自动裁判 MiMo‑v2.6‑Pro‑RL，统计置信区间、Krippendorff α 等统计方法。

**📊 数据集**

基准数据集由30个合成案例组成（10个网络安全、10个医疗、10个军事），每例包含两份SECRET级别文档、1个TOP_SECRET组合、1个干扰文档和3个查询，案例由Claude Opus 4.6等生成。

**📈 对比分析**

对每个生成器在24种配置下产生3600个答案，计算泄露率和拒绝率，并与人类标注对照；结果显示GPT泄露率最低约27.8%，Claude约63%，Qwen约67%；在开启可见分类头和聚合规则后泄露率降至约32%，但benign查询下仍有7–12%泄露。

**⚠️ 局限性**

限制：基准仅为合成例子，覆盖范围有限；只评估最终答案，未涉及中间推理或工具输出；评估依赖人类标注，未独立验证裁判；现场与模拟差异难以分离工具使用贡献；未考虑多代理协作场景等。

---

## 131. SplatLab: Autonomous Scientific Discovery for 3DGS Research

**arXiv ID:** 2610.30468 | [PDF](https://arxiv.org/pdf/2610.30468v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 132. CARS: a centerline-anchored attention surrogate for coronary blood flow on skeleton-generated lumen geometries

**arXiv ID:** 2610.30498 | [PDF](https://arxiv.org/pdf/2610.30498v1)

**作者:** Reza Akbarian Bafghi `[一作]` (University of Colorado Boulder), Maziar Raissi `[通讯]` (University of California, Riverside)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

构建了一个以冠状动脉中心线为焦点的两阶段锚点注意力网络（CARS），用于在给定中心线、腔径曲线和流量的条件下快速预测静态压强和三维速度场，直接实现CT‑到FFR的无创预测。

**💡 创新点**

创新点在于：①将中心线及其半径信息作为唯一几何焦点集，形成层次化的锚点注意力，既保持几何信息又实现线性复杂度；②将查询点作为网络输入参数，允许任意采样密度；③使用FiLM乘性流量调制，将流量与几何信息分离；④引入学习的坐标嵌入和两阶段处理，提升精度。

**🔧 技术方法**

技术包括：变形可变形Transformer、层次化（tiered）注意力、学习的坐标编码（Learned PE）或RoPE3D、FiLM特征调制、两阶段网络结构（几何融合→场细化）以及FP32/ BF16推理。

**📊 数据集**

数据集为公开的ImageCAS冠状动脉CT图像，使用OpenFOAM管道生成6,600条稳态不可压Navier‑Stokes CFD解（2,700/300/300训练/验证/测试），每条样本包含中心线点、腔径、流量及对应的压强/速度场。

**📈 对比分析**

与四个神经算子基线（GNOT、ONO、Transolver、AB‑UPT）以及多种消融实验在相同训练拆分、相近参数量下进行比较。CARS在压力相对ℓ₂误差上最低（LCA 0.117，RCA 0.063），FFR决策准确率达97.7%/96.3%，推理时间约87 ms，显著优于基线且在推理成本上保持可接受水平。

**⚠️ 局限性**

局限性包括：①仅验证在单中心、单扫描协议下的几何分布；②缺乏对不同解剖、扫描设备或病人群体的跨域泛化；③不保持旋转不变性，需在相同坐标系下输入；④仅针对稳态、无弹性壁、牛顿流体，未覆盖脉冲、弹性或非牛顿特性；⑤对中心线提取误差和腔径误差敏感，无法直接处理非圆柱形腔体。

---

## 133. From Single-Actor Environments to Multi-Actor Event Worlds: Can LLMs Be Event World Models?

**arXiv ID:** 2610.30488 | [PDF](https://arxiv.org/pdf/2610.30488v1)

**作者:** Juntong Ni `[一作]` (Emory University), Wei Jin `[通讯]` (Emory University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文在多主体事件世界中评估大型语言模型（LLM）作为世界模型，构建 Diplomacy 与 Polymarket 两个基准数据集，探讨LLM在预测、回滚、未执行动作及辅助代理决策与学习中的表现。

**💡 创新点**

创新点包括：①提出事件世界的统一框架和分层评估体系；②证明当给定全部演员动作时LLM可学习环境动力学，但仅有自身动作时预测能力显著受限；③展示LLM在规划与训练代理中的潜在价值与局限，并给出改进方向。

**🔧 技术方法**

使用的技术包括：大型预训练语言模型（GPT‑6‑Luna、GPT‑6‑Sol、Qwen3‑1.7B、Qwen3‑8B），在事件日志上进行提示与微调；设计两任务（Task S/Task J）、多步回滚、未执行动作预测；采用多种评估指标（EM、F1、Acc、CRPS、成功率等）。

**📊 数据集**

数据集：Diplomacy 33,279 局（约 940 k 转移）和 Polymarket 4,176 市场（约 1.02 M 转移），通过日志构造单方视角样本。

**📈 对比分析**

比较方法：在 Task J（已知全部动作）与 Task S（仅知自身动作）下进行零样本提示与 20k 迁移微调，并与基线（无变化、直接执行规则）及 LightGBM 对比。结果显示微调后 Task J 的 EM/Acc 可达 60% 以上，而 Task S 仅约 20%；多步回滚与未执行动作预测误差显著；在代理规划中，仅在给定完整动作时 LLM 能提升决策效果。

**⚠️ 局限性**

限制：仅验证 Diplomacy 与 Polymarket 两个域；评估依赖规则引擎与固定对手，未检验对人类或自适应对手的泛化；未处理多主体学习与因果推断；LLM 对其他演员行为的预测能力不足。

---

## 134. HeyGen Voice: Scene-Aware Voice Cloning for Talking Videos

**arXiv ID:** 2610.30470 | [PDF](https://arxiv.org/pdf/2610.30470v1)

**作者:** Pavel Andreev `[一作]` (HeyGen Research), Rong Yan `[通讯]` (HeyGen Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种名为 HeyGen Voice 的零样本语音合成模型，能够在保持参考声音身份的同时，根据目标图像和文本说明调整语音的表达方式和环境声学，专门用于生成带可视人物的谈话视频。

**💡 创新点**

创新点主要在于：① 将声音身份与场景声学解耦，提供两种克隆策略（只保留声音身份或同时保留录制环境）；② 通过图像+两段说明（面部与背景）来引导语音表达与空间音效；③ 设计了跨场景训练课程、GRPO 奖励与 DPO 表达适配器，实现语音一致性与表达性双重优化；④ 引入 anchor‑derived consistency embedding 与局部声学上下文，保证长文本在多段合成中的声音与房间一致。

**🔧 技术方法**

技术包括：多模态自回归 Transformer 负责生成离散声码；冻结视觉、文本与音频编码器提取图像、说明与参考音频特征；flow‑matching 解码器（基于 diffusion‑transformer）将声码映射为 44.1 kHz 波形；奖励机制（GRPO 用 ASR 与说话人验证奖励）和低秩 DPO 适配器用于表达优化；训练使用跨场景预训练、模拟扰动、真实跨场景对、群体相对策略优化与偏好优化。

**📊 数据集**

主要数据来源是 HeyGen 的谈话视频语料库（与 Avatar V 共享），包含多语言、多场景、多设备录音。训练过程对音频进行分离、分离后识别、语言标注、说话人聚类、字幕生成、跨场景配对等八个阶段。评测数据包括 331 条相同说话人跨场景对、十语言 CER 验证集、245 条基准脚本、50 个多场景视频，以及公开的 Artificial Analysis 控制声音排行榜。

**📈 对比分析**

与 ElevenLabs v4、Cartesia Sonic 3.6、Inworld TTS‑2、Seedance 2.5 等主流商业系统进行对比。HeyGen Voice 在跨场景基准中获得最高场景对齐与说话人相似度；在十语言 CER 中保持低误码；在五语言双人听觉实验中在叙述者选择、自然度、清晰度、口音与相似度等维度均优于竞争对手；在公开排行榜上排名第一；在用户投票和志愿者自评中，HeyGen Voice 的分数普遍高于其他系统（差距 10–30 分）。

**⚠️ 局限性**

局限性包括：对图像质量与姿态的依赖较高，低质量或极端姿势的图像可能导致声学或表达不匹配；跨场景配对数据仍有限，难以覆盖极端场景或极大声学差异；模型在极低资源语言或非标准口音上的泛化仍需改进；对长文本的声学一致性虽通过 anchor 与上下文提升，但在极长脚本或极多场景切换时仍可能出现轻微漂移；最后，系统依赖大量 GPU 计算，部署成本相对较高。

---

## 135. Know When to Escalate: Critic-Guided Adaptive Execution for Cloud-Local Embodied Agents

**arXiv ID:** 2610.30587 | [PDF](https://arxiv.org/pdf/2610.30587v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 136. Evaluating Type-1 Diabetes Digital Twins by Decision Quality

**arXiv ID:** 2610.30577 | [PDF](https://arxiv.org/pdf/2610.30577v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 137. Optimal Mixing for Local Edge-Coloring Dynamics on Trees

**arXiv ID:** 2610.30625 | [PDF](https://arxiv.org/pdf/2610.30625v1)

**作者:** Haoran wang `[一作]` `[通讯]`, Haoran wang

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在树上进行单条边 Glauber 动态和相邻边热浴动态的混合时间，证明当颜色数 q≥Δ+2（单边）和 q≥Δ+1（相邻边）时，利用 log‑Sobolev 不等式得到最优的 O(n log n) 采样时间。

**💡 创新点**

提出了可重用缓冲区和局部颜色释放技术，并结合组件熵分解，首次实现了在树结构上对近似阈值颜色数的最优混合时间分析。

**🔧 技术方法**

采用熵方法的 log‑Sobolev 估计、组件动态框架、局部重彩路径、消息传递集中性以及最大化耦合等技术进行严格证明。

**📊 数据集**

本研究为理论论文，不使用任何实验数据集。

**📈 对比分析**

与先前的多项式上界和路径计数结果相比，本方法给出最优的 O(n log n) 上界，并在大调色板下实现了与 q 无关的常数上界，证明了阈值颜色数下的最优性。

**⚠️ 局限性**

局限性包括常数对最大度 Δ 的依赖尚未优化为多项式，且仅在树结构上适用，对更一般图（如有环图）仍未得到结果。

---

## 138. Temporal Liquidity Crossing for Blockchains: An EIP-1559 Extension for Admission Below the Base Fee

**arXiv ID:** 2610.30595 | [PDF](https://arxiv.org/pdf/2610.30595v1)

**作者:** Duanyang Dan Guo `[一作]` `[通讯]` (TLM Research), Duanyang Dan Guo (TLM Research)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了块级时间流动性市场（TLM），为以太坊扩展EIP‑1559，允许消费者支付前区执行费用并获得相对排序，供应商接受后区执行且可低于基费入区；

**💡 创新点**

创新点在于把执行时间视为独立维度引入市场机制，构建自给自足的跨时段交易匹配，并通过消费者授权保留普通交易分配，同时提升总利用率；

**🔧 技术方法**

采用机制设计与形式化证明、基于EIP‑1559的参数化规则、基于基费反馈的计费模型，并使用区块链模拟器进行实验验证；

**📊 数据集**

使用合成交通流（正TLA消费者、负TLA供应商与中性交易）并以Pareto分布生成手续费、随机gas等参数，没有真实区块链数据；

**📈 对比分析**

与基线EIP‑1559及“原生计费”方案对比，TLM在20%/30%供应商额度下将总利用率从≈50%提升至≈70%，基费保持不变，消费者每块支付≈54M，供应商燃烧基费并获得包含费；

**⚠️ 局限性**

局限在于单槽仿真、合成负载、未验证真实编码与部署、缺乏多槽或跨链的动态分析、消费者投标均衡与顺序对实时gas影响的实证研究。

---

## 139. Diagnosing Personalization Failures in Mobile GUI Agents

**arXiv ID:** 2610.30573 | [PDF](https://arxiv.org/pdf/2610.30573v1)

**作者:** Zhixin Lin `[一作]` (Shandong University), Yue Yao `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建 PrefDiag-Bench 基准，提供五类偏好分类与阶段性诊断框架，用于评估移动 GUI 代理的个性化执行能力。

**💡 创新点**

创新点在于将用户偏好按执行效果拆分为 Outcome、Constraint、Ranking、Process、Interaction 五类，并结合 1,664 句诊断探针追踪偏好何时何种影响，揭示任务成功率背后的能力瓶颈。

**🔧 技术方法**

利用大语言模型（LLM）生成任务、偏好和行为日志，设计显式用户档案与行为日志两种上下文，并用诊断探针评估证据选择与触发识别能力。

**📊 数据集**

使用 318 个端到端 GUI 任务（53 个基本任务 + 265 个偏好条件任务）和 1,664 个诊断探针，覆盖七大应用域和多种语义偏好类别。

**📈 对比分析**

对 11 种现有 GUI 代理（含 9 个端到端模型与 2 个工作流）进行评估，发现不同偏好类型性能差异显著，最高的 Process 类型仅 16.98%，整体任务成功率显著低于基本任务；通过诊断探针揭示证据选择与触发识别是关键瓶颈。

**⚠️ 局限性**

局限在于 GUI 执行能力不足导致证据干预效果有限，且仅在部分代理上可见提升；缺乏进一步提升偏好识别与执行一致性的技术和数据。

---

## 140. Governed Self-Evolution for Autonomous Investing: A Multi-Agent LLM System with an Invariant Core

**arXiv ID:** 2610.30557 | [PDF](https://arxiv.org/pdf/2610.30557v1)

**作者:** Yebo Feng `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并部署了一个由九个LLM代理组成的多代理自主投资系统，在实时Web数据上做决策并执行交易。

**💡 创新点**

创新点在于：①在类型层面划定不可变核心，保证自我进化不破坏风险约束；②通过结构化Typed Trace与可重放验证，保证自我改进安全可靠；③结合多来源Web信息与共享记忆，实现强大协同决策。

**🔧 技术方法**

技术手段包括：LLM代理+Pydantic类型化输出+链式思考（CoT）+七层共享记忆+代码化的风险引擎+自评与反思循环+可回放验证门控。

**📊 数据集**

使用的数据集为公开Web数据：新闻RSS、SEC 10‑Q/10‑K文件、FRED宏观系列、实时行情、行业估值，回测覆盖2015‑2026年US股票及ETF。

**📈 对比分析**

对比方法：与预期意图、单一模型与无记忆版本、以及基准指数进行对比。性能表现为：实盘收益+5%，意图收益+12.2%，单模收益+1.3%；最大回撤显著降低，Alpha与Sharpe均优于基线。

**⚠️ 局限性**

局限性：仅在纸质账户单一配置5个月运行，缺乏实际成交冲击；仅实现两轮自我进化；回放样本有限，难以在回测中进行排名；未来需扩展配置与更深层验证。

---

## 141. OMRC: Multipath Reliability for Direct-Connect Optical Interconnects

**arXiv ID:** 2610.30551 | [PDF](https://arxiv.org/pdf/2610.30551v1)

**作者:** Eric Ding `[一作]` (Cornell University), Rachee Singh `[通讯]` (Cornell University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了OMRC，一种在直接连接光学数据中心中扩展MRC的控制层，能够在链路或平面失效时重新映射集合通信或重新分配光学平面，从而恢复训练与推理工作负载的可用性和吞吐量。

**💡 创新点**

① 引入拓扑感知的集合通信重映射，减少中间节点转发；② 在通信组失连时动态重配置OCS平面，实现跨组平面再分配；③ 将控制层与MRC结合，实现更高效的多路径可靠连接。

**🔧 技术方法**

MRC多路径可靠连接、光学电路交换器(OCS)、MSCCL++集合通信库、RoCEv2 RDMA、软件仿真ASTRA-sim、图匹配与通道重映射、控制器与sidecar机制。

**📊 数据集**

使用Qwen1.5-MoE模型（14.3B参数）进行训练与推理，采用WikiText-2文本令牌；仿真采用随机链路失效的网络模型。

**📈 对比分析**

与基线MRC在同一4GPU Polatis OCS上进行对比。临时链路失效时，OMRC将平均跳数从1.125降至1.0，算法带宽提升24%，训练吞吐提升5.3%。永久链路失效时，OMRC通过平面重分配将推理吞吐从3.335k降至3.311k，保持99.3%预失效吞吐。仿真中，OMRC在多链路失效情况下比MRC提升至4.74倍的算法带宽。

**⚠️ 局限性**

仅适用于直接连接光学网络（需OCS）且需软硬件支持的多路径转发与平面重配置；实验规模有限（4GPU），真实生产环境需进一步验证；控制层与sidecar实现复杂；仿真模型简化，未考虑实际延迟、能耗等因素。

---

## 142. Rethinking the Evaluation of LLM Fact-Checkers

**arXiv ID:** 2610.30504 | [PDF](https://arxiv.org/pdf/2610.30504v1)

**作者:** Ayana Niwa `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Satoshi Sekine `[通讯]` (National Institute of Informatics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了JSFactCheckBench日语LLM答案事实核查基准。

**💡 创新点**

首次同时明确三个设计决策：问题来源、判定条件与证据来源。

**🔧 技术方法**

采用LLM提取主张、人工三人标注、Web证据构建。

**📊 数据集**

利用JSocialFact的日语问题，提取1,913条主张，伴随1,397域名准入列表。

**📈 对比分析**

对六款LLM进行宏F1评估，发现模型在关键主张、时间敏感性与检索选择上表现差异显著。

**⚠️ 局限性**

局限于日语域、固定参考时间、规模有限，需定期更新与跨语言迁移。

---

## 143. DLT Terms 100: a learner's glossary based on ISO 22739:2024

**arXiv ID:** 2610.30596 | [PDF](https://arxiv.org/pdf/2610.30596v1)

**作者:** Yusuke Kaneko `[一作]` (Japan Research Institute), Hiroki Watanabe `[通讯]` (Japan Research Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了基于ISO 22739:2024的DLT Terms 100词汇表，帮助提升区块链技术的识字率并梳理业务与技术术语的差异。

**💡 创新点**

创新点在于将100个核心术语按八大领域和三难度级别分类，并通过频率分析揭示术语在主流技术参考文献中的覆盖度与重要性。

**🔧 技术方法**

采用文本频率分析、术语归类与难度评估方法，结合ISO标准及行业教材进行系统整理。

**📊 数据集**

使用了四份关键参考资料：ISO 23257:2022、Mastering Bitcoin 第三版、Mastering Ethereum 第二版、以及《区块链技术入门》（日语）。

**📈 对比分析**

通过对比四份文献中的术语出现率，计算覆盖率（分别为77%、59%、67%、71%），并识别出37个全文共通术语、71个出现两及以上、5个未出现的术语，尚未进行使用效果的实证评估。

**⚠️ 局限性**

局限性包括缺乏实验验证该词汇表对沟通成本或学习效果的提升；未包含商标和产品名；依赖ISO标准和公司内部教材，覆盖面可能受限。

---

## 144. Refining Open-Vocabulary Semantic Segmentation via Regional Semantics and Visual Prototypes

**arXiv ID:** 2610.30623 | [PDF](https://arxiv.org/pdf/2610.30623v1)

**作者:** Aijing Yu `[一作]` (University of Chinese Academy of Sciences), Xiao-Yu Zhang `[通讯]` (Institute of Information Engineering)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无标注的开放词汇语义分割方法，通过改进CLIP视觉编码器的注意力机制和引入视觉原型记忆来提升像素级预测；

**💡 创新点**

创新点包括：① 在CLIP最后注意力层使用v-v映射自注意力，保留局部语义一致性；② 设计视觉原型记忆模块作为辅助预测器，缓解文本‑视觉模态差距；③ 采用EMA优化特征提取器实现无标注自训练；

**🔧 技术方法**

使用了CLIP ViT‑B/16/ResNet‑50视觉编码器、自注意力修改、视觉原型记忆、残差融合文本与视觉预测、EMA特征提取器以及伪标签自训练技术；

**📊 数据集**

在ADE20K、Cityscapes、COCO Stuff164k、PASCAL Context和VOC12AUG五个标准分割数据集上进行实验；

**📈 对比分析**

与MaskCLIP、GroupViT、ReCo、TCL、CLIP Surgery、GEM等开放词汇分割模型按mIoU进行比较，方法在ADE20K 17.2%、Cityscapes 32.5%、PASCAL Context 33.7%上取得SOTA，平均35.9% mIoU，明显优于现有方法；

**⚠️ 局限性**

局限性在于仍依赖CLIP预训练权重，模态差距对性能影响仍显著；在ResNet‑50骨干上的提升有限；对极端小目标或复杂场景的鲁棒性未充分验证；缺少公开完整训练细节，复现性待进一步完善。

---

## 145. KinoUAV: A Benchmark for Safe and Stable UAV Vision-Language Navigation with Kinodynamic Modeling

**arXiv ID:** 2610.30609 | [PDF](https://arxiv.org/pdf/2610.30609v1)

**作者:** Chenyao Yang `[一作]` (Aerospace Information Research Institute, Chinese Academy of Sciences), Wenhui Diao `[通讯]` (Aerospace Information Research Institute, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于运动动力学的UAV视觉语言导航基准KinoUAV，并提出KinoUAV-Agent实现安全、稳定的航行。

**💡 创新点**

创新点在于将运动动力学约束与前后向自由空间距离（DFD）监督融入数据生成与控制，形成层次化航点规划与加速度控制的闭环体系。

**🔧 技术方法**

采用分层运动动力学A*搜索、加速度动作空间、ResNet18视觉编码器、DFD解码器与加速度预测器，并利用大型视觉语言模型（如Qwen3‑VL‑8B）微调高层航点规划。

**📊 数据集**

使用AirSim/UE4仿真环境的16个场景，生成约30万条包含图像、运动状态与DFD标签的轨迹，划分为Main和Auxiliary子集。

**📈 对比分析**

与Random、CMA、TravelUAV、AerialVLA等基线比较，在见图、未见目标、未见地图三种测试下，KinoUAV-Agent以约89%成功率、6%碰撞率、平均速度1.68 m/s、yaw率0.045 rad/s获得最佳综合表现。

**⚠️ 局限性**

局限包括仅在模拟环境、简化动力学模型、缺乏真实风阻与移动障碍物，且对DFD预测误差敏感。

---

## 146. OASIS: Online Anchor-centric Scene Inference System for Open-Vocabulary Retrieval and Navigation in Dynamic Environments

**arXiv ID:** 2610.30589 | [PDF](https://arxiv.org/pdf/2610.30589v1)

**作者:** Kaiyao Liao `[一作]` (Beihang University), Jianwei Niu `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了OASIS，一个在线anchor‑centric场景推理系统，能够在动态环境中进行开放词汇检索与导航，并支持被移动物体的重新定位。

**💡 创新点**

通过将室内场景建模为稀疏的语义anchor图，并结合LLM与VLM实现分层检索，配合经验记忆策划和瞬时视觉缓冲，轻量化存储、提高检索效率并实现动态重定位。

**🔧 技术方法**

使用LLM（如GPT‑5.2）进行anchor筛选与查询规划，VLM（Gemini 3 Pro）进行视觉验证，CLIP做特征匹配；采用稀疏视觉‑空间聚类、可扩展多模态图结构和Transient Visual Buffer等技术。

**📊 数据集**

在Replica、HM3D、ScanNet等仿真数据集上评估检索与导航性能，并在包含13个语义anchor的真实家庭环境中进行实验。

**📈 对比分析**

与DualMap、ConceptGraphs等基线相比，OASIS在Fine‑grained Object Retrieval mAP 6.58、功能区检索 57.7% 以上、空间关系检索 44% 以上、导航成功率 85.9% 以上，并在动态搬迁场景中成功率提升至 77.4%，显著优于基线。

**⚠️ 局限性**

对长距离或跨anchor复杂空间关系仍存在挑战；LLM与VLM推理开销较大；无法完全处理极高遮挡或极小物体；重定位依赖路径中出现目标，无法覆盖完全未知位置。

---

## 147. When News Gets Old: Portfolio Allocation with Reinforcement Learning and Large Language Models

**arXiv ID:** 2610.30564 | [PDF](https://arxiv.org/pdf/2610.30564v1)

**作者:** Bolin Song `[一作]` (Cornell University), Surui Xiang `[通讯]` (United Nations)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文比较了在利用证券交易委员会（SEC）披露信息进行投资时，是否保留旧信号并给模型提供新闻时间信息，还是直接过滤掉超过一定时间的信号，并将两种策略应用于强化学习（RL）和大型语言模型（LLM）投资决策；

**💡 创新点**

创新点在于系统性评估新闻时效对投资绩效的影响，并证明对LLM而言简单的信号过滤（8 周阈值）可优于将旧信号保留并加入时效信息，此外对RL的时效信息响应与实际收益不一致的现象提供了新的见解；

**🔧 技术方法**

使用了 GPT‑4.1 mini 提取 SEC 披露的正、负、中性信号；将信号与市场数据和持仓一起输入双 DQN 强化学习模型和直接决策的 LLM；对不同时间延迟（0、1、4 周）和不同信号处理方式（无时效、加时效、过滤）进行训练与测试；

**📊 数据集**

实验数据集包括 2023‑2025 年美国 10 家上市公司的历史价格、SEC 披露文本及其发布时间；

**📈 对比分析**

通过历史回测（每周决策、10 bp 成本）计算净年化收益、Sharpe 比率、最大回撤、交易量和费用，结果显示 LLM 过滤旧信号后年化净收益最高（18.43%），优于所有使用新闻的 RL 策略；提供时效信息的 RL 在收益上并未持续提升；

**⚠️ 局限性**

局限性包括：只考虑 10 支股票、仅使用 SEC 披露文本、模型仅在历史数据上验证、RL 对时效信息的响应不转化为收益、交易费用对 RL 影响显著、未检验不同市场环境或更长周期的稳健性。

---

## 148. Behind Aggregate Scores: What Systematically Varying Arena-Hard's LLM Judge Reveals

**arXiv ID:** 2610.30512 | [PDF](https://arxiv.org/pdf/2610.30512v1)

**作者:** Louise Yorke `[一作]` `[通讯]` (University College London), Louise Yorke (University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在固定的 500 条 Arena-Hard v2.0 评测问题和两条模型回答的前提下，使用 GPT‑4.1 作为评判者，对不同的评判指令（判别标准、输出格式、任务框架）进行 8 种组合实验，并在每种配置下多次重复运行，测量总体得分和每条评测的具体胜负结果。

**💡 创新点**

首次将总体得分与单条评测结果的稳定性分离，量化评判者在重复运行时的误差（test‑retest disagreement），并评估不同指令改动对单条评测的影响，提出在评估中同时报告整体分数和单条结果变化的必要性。

**🔧 技术方法**

利用 GPT‑4.1 进行双模型对比评判，采用 2×2×2 完全因子设计的指令组合，计算原始胜率等价分数和边界变化率；还使用 Claude Sonnet 4.5 与 Gemini 2.5 Flash 替代 GPT‑4.1 进行对比；对结果采用自助法置信区间、Cohen’s κ 计算一致性。

**📊 数据集**

500 条 Arena-Hard v2.0 hard‑split 提示；两条模型回答分别为 GPT‑4.1 mini 与 o3-mini；所有提示和回答在所有实验中保持不变。

**📈 对比分析**

对总体得分（win‑equivalent 0‑100 评分）与单条评测的边界变化率进行比较。总体得分变动通常仅 1–4 分，但单条结果的变化率高达 21–28%（甚至 38%），显示整体分数不等价于单条结果的稳定性；对不同指令改动的“超额变化”与测试‑重测误差比较，显示某些指令（如仅输出标签、以准确度为主的标准）导致显著更多的单条结果变化。

**⚠️ 局限性**

仅测试了一个基准、一个模型对（GPT‑4.1 mini vs o3-mini）、单天重复实验，缺乏不同模型能力差距、不同基准和多日重复的验证；指令改动涉及完整的指令包，难以分离单一因素的影响；没有人工真值标签，无法判断哪种评判更准确；Claude 与 Gemini 的比较同时涉及模型和实现差异。

---

## 149. The Cost of Missed States: Online Tracking with Current-Observation Resets

**arXiv ID:** 2610.30505 | [PDF](https://arxiv.org/pdf/2610.30505v1)

**作者:** Hiroyasu Usami `[一作]` `[通讯]` (Chubu University), Hiroyasu Usami (Chubu University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在线跟踪问题，控制器只能保留一个过去观测状态，且只能将当前观测装载为新状态，提出并分析了该模型下的各种在线策略与下界。

**💡 创新点**

① 引入“当前观测重置”模型，揭示在硬半径约束下即使是随机因果策略也无法与离线最优实现多项式竞争；② 将 Bienkowski–Schmid 的嵌套区间下界改造为带刷新费用的目标；③ 证明任意时策略和剪裁阈值策略在变差控制下满足 J_λ ≤ λ+4√(λTV_T) 及更强的三倍上界；④ 通过 Lipschitz 敏感性、softmax Hilbert 距离和平方距离 softmax 推导输出误差上界；⑤ 在固定有限度量空间（正最小间距）下给出刷新次数与成本可竞争的策略。

**🔧 技术方法**

竞争分析、随机因果策略、嵌套区间下界、变差预算、阈值与剪裁阈值策略、Lipschitz 敏感性、softmax Hilbert 距离、平方距离 softmax 上界、实验评估（合成路径）等。

**📊 数据集**

实验使用 72 组合成路径（随机游走、突发-平台、转折模式），长度分别为 32、64、128，配合 λ ∈ {1/16, 1/4, 1, 4}，共 288 条路径–费用组合，生成离线最优、固定阈值、任意时、剪裁、始终刷新、仅保留初始状态等多种策略的成本。

**📈 对比分析**

通过与离线最优 J_λ^* 的比值来评估。表格显示任意时策略在多数情形下接近最优，剪裁策略在低变差情况下更优；固定阈值策略在已知 V_T 时表现最好。最差比值不超过约 3，说明即使在最坏情况下也只有常数倍的性能损失。实验表明任意时与剪裁策略在不同 V_T 与 λ 组合下的相对优势可视化。

**⚠️ 局限性**

① 对当前观测重置模型的下界表明无法获得无论 Horizon 如何都成立的常数竞争因子；② 竞争性恢复仅在固定有限度量空间（正最小间距）下成立，其他情况仍无法竞争；③ 输出误差上界依赖于 Lipschitz 常数和 softmax 形式，实际误差可能更小或更大；④ 变差预算分析在无年龄约束时仅给出上界，实际更新频率可能受硬阈值与模型假设影响；⑤ 实验仅使用合成路径，缺乏真实世界数据验证。

---

## 150. PhysibleMDM: Enforcing Inference-Time Physical Feasibility Constraints on Motion Diffusion Models via Noise Optimization

**arXiv ID:** 2610.30500 | [PDF](https://arxiv.org/pdf/2610.30500v1)

**作者:** Maximilian Stölzle `[一作]`, Moritz Bächer `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过在已训练的运动扩散模型推理阶段对噪声进行无梯度优化，结合关节、碰撞、重力等物理可行性约束，生成更符合物理约束的人体/机器人运动。

**💡 创新点**

提出在推理阶段同时优化初始噪声和后续噪声的无梯度方法（Sep‑CMA‑ES），利用逆动力学内部循环评估动态可行性，并通过噪声标准化在保持运动风格的同时提高物理可行性，无需重新训练生成器或使用跟踪控制器。

**🔧 技术方法**

技术包括运动扩散模型（MDM）、Sep‑CMA‑ES采样优化、逆动力学求解器（cone program）、关节与动态可行性约束、噪声标准化、自动回归片段链式生成。

**📊 数据集**

使用100STYLE人类运动数据集训练SMPL模型；自定义双足机器人在11个不同风格、坡度、摩擦条件下的序列进行评估。

**📈 对比分析**

与未优化、Motion GD、Motion Sep‑CMA‑ES、Guided Diffusion、Noise Best‑of‑N、Noise GD、Noise ZeNO等基线对比。结果显示：在SMPL上，地面穿透、碰撞、滑动分别降低约75%/92%/74%；在机器人上，平均加速重构误差和ID残差分别减少89.5%/79.5%；同时保持或提升运动平滑度和风格保留，计算时间保持在可接受范围内。

**⚠️ 局限性**

软约束无法严格保证物理可行性；噪声优化可能导致与训练分布的偏离，影响风格；逆动力学求解器计算成本高，限制实时性；仅在双足机器人与固定接触模式下验证，需进一步推广到更复杂的多足/非刚体系统。

---

## 151. Beyond Better Ideas: How Human-AI Collaboration Shapes Scientific Ideas and Human Perceptions

**arXiv ID:** 2610.30568 | [PDF](https://arxiv.org/pdf/2610.30568v1)

**作者:** Qian Chen `[一作]` (Pennsylvania State University), Wenpeng Yin `[通讯]` (Pennsylvania State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究构建了统一实验框架，系统考察人工智能在科研创意生成、评估与归因中的作用，探讨人机协同对提案质量、评审偏差、可检测性以及研究者认知的影响。

**💡 创新点**

创新点在于首次揭示AI协同对提案质量提升有限、AI评审者会系统偏向AI协同产物并放大此效应、协同痕迹仅可部分检测，并将这些现象与研究者认知变迁共同呈现。

**🔧 技术方法**

采用大语言模型（GPT‑4.1、Gemini 2.5 Flash、Claude Sonnet 4.6、Qwen 2.5 72B）作为评审者和归因分类器，结合提示工程与LLM微调。

**📊 数据集**

使用70个一页研究提案（35支队伍，Human‑AI 与 Human‑Human 两种协同）以及从1,000篇真实论文生成的2,000个合成方法学实例。

**📈 对比分析**

与人类评审相比，GPT‑4.1在绝对评分上与人类相当但在偏好排序上差距明显；AI评审者对Human‑AI提案的平均分比Human‑Human高0.13分，归因分类准确率约66%，迭代修订提升整体质量0.18分。

**⚠️ 局限性**

主要限制包括样本量有限、协同差异受研究领域与参与者背景影响、归因识别精度仅中等且缺乏因果推断，以及评审者与人类评价的主观性未完全消除。

---

## 152. The Fusion Tax: A Diagnostic Analysis of When Hybrid Retrieval Hurts

**arXiv ID:** 2610.30612 | [PDF](https://arxiv.org/pdf/2610.30612v1)

**作者:** Aaditya Chauhan `[一作]` (Independent Researcher), Arun Agarwal `[通讯]` (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对混合检索（稀疏+稠密融合）的单查询效果进行了量化，定义并测量了“融合税”。

**💡 创新点**

创新点在于将融合损失与更强单一检索器对比，并通过多维诊断分箱揭示导致损失的具体查询特征，证明融合税在不同编码器规模与稀疏方式下均不消失。

**🔧 技术方法**

采用BM25稀疏检索、BGE系列及MiniLM、Qwen3稠密检索、SPLADE学习式稀疏、RRF、加权插值、CombMNZ融合以及BGE交叉编码重排序。

**📊 数据集**

使用BEIR七个基准数据集（SciFact、FiQA、NFCorpus、TREC-COVID、ArguAna、SciDocs、Quora）。

**📈 对比分析**

对比结果显示，虽然加权插值在七个数据集上整体nDCG@10均优于单一检索，但其单查询“融合税”在13–48%范围内，并且在大多数语料库下存在显著的风险敏感不利。

**⚠️ 局限性**

局限性包括仅评估了单阶段检索+单一重排序、未覆盖答案生成、TREC-COVID样本量小、性能测量受单机CPU/GPU限制、并未探讨更复杂的路由或LLM评估器。

---

## 153. CMAR-search: Commonsense and Memory Augmented Reasoning for Object Search in Dynamic Interactive Environments

**arXiv ID:** 2610.30606 | [PDF](https://arxiv.org/pdf/2610.30606v1)

**作者:** Kaiyao Liao `[一作]` (Beihang University), Jianwei Niu `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 CMAR‑search 框架，用于在大规模动态人类环境中高效完成交互式目标物体搜索。

**💡 创新点**

核心创新包括：① 通过 LLM 与常识推理构建功能化 3D 场景图（F3DSG），实现功能区域分割；② 采用层次化宏/微动作规划，显著减少 LLM 调用次数与搜索路径；③ 引入记忆增强推理（CMAR）机制，将实时感知、历史记忆与物体常识融合，实现对环境变化的自适应重定位。

**🔧 技术方法**

技术手段包括 LLM（DeepSeek‑V3 / GPT‑4）推理、3D 场景图与 Voronoi 图构建、A* 路径规划、SBERT 相似度计算、层次化动作空间以及记忆增强推理算法。

**📊 数据集**

实验使用 iGibson 仿真环境的八个训练场景和七个测试场景，真实世界实验在包含厨房、客厅与卧室的家庭房间中进行；未使用公开大规模物体数据库。

**📈 对比分析**

与随机、贪婪、ESC‑Interactive、HIMOS、无结构 LLM、MoMa‑LLM 等基线进行对比，CMAR‑search 在成功率（98.9%）、SPL（79.0）、物体交互次数（2.9）和行驶距离（9.8 m）等指标均显著优于基线，证明其在效率与鲁棒性上的优势。

**⚠️ 局限性**

局限性包括：依赖高质量的语义感知与几何信息，层次化规划仍需打开所有可操作容器导致某些冗余操作，性能受 LLM 推理速度与推理精度影响，且在极大规模真实环境中的可扩展性尚待验证。

---

## 154. Visual Grounding First, Multimodal In-context Learning Follows

**arXiv ID:** 2610.30592 | [PDF](https://arxiv.org/pdf/2610.30592v1)

**作者:** Minhyuk Seo `[一作]` (University of California Berkeley), Jonghyun Choi `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过对视觉对比数据的原子命题级对比奖励，提升多模态LLM的视觉-文本关联，从而实现无 ICL 数据训练的强大上下文学习。

**💡 创新点**

创新点在于提出了“原子命题级对比奖励”（P-Contrastive Reward）并将其应用于视觉对比学习，以增强细粒度视觉归纳而非依赖 ICL 训练。

**🔧 技术方法**

采用对比强化学习、CLIP 视觉/文本编码器、NLI 归纳得分、Group Relative Policy Optimization（GRPO）等技术。

**📊 数据集**

使用 InpaintCOCO、EQBEN 等视觉对比数据集；在 MS‑COCO、TrueMICL、ISEKAI、个性化 ICL 基准等上评测。

**📈 对比分析**

与检索、任务向量、ICL 训练等多种 SOTA 方法对比，在多模态 ICL 任务上平均提升 5.6% 以上（分类）和 13.8% 以上（caption），且在无 ICL 训练时性能最高；同时显著提高目标注意力得分并降低 hallucination。

**⚠️ 局限性**

局限性包括：仍依赖对比数据集的质量；对超大模型的可扩展性未充分验证；在极端多样性场景下对细粒度辨别的效果尚需进一步评估。

---

## 155. AI-Native Academia Should Build a Community Verification Layer

**arXiv ID:** 2610.30549 | [PDF](https://arxiv.org/pdf/2610.30549v1)

**作者:** Haotian Zong `[一作]` (Johns Hopkins University), Luhao Zhang `[通讯]` (Johns Hopkins University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个可互操作的社区验证层，用来将研究者日常阅读与检查转化为可审计、与主张和版本关联的证据，并为稀缺的人力验证分配机制。

**💡 创新点**

引入分层验证金字塔、预先保留判断的盲评模式、主动分配稀缺验证资源的优先得分以及可追溯的证据链，以解决 AI 时代研究产出与验证之间的失衡。

**🔧 技术方法**

采用机器可读的声明–证据接口、nanopublication/RO‑crate 等结构化记录、自动化分配算法、身份验证与隐私保护机制等技术。

**📊 数据集**

作为概念性提案，未使用具体实验数据集。

**📈 对比分析**

通过功能对比表与现有开源评估平台（如 Post‑publication review、preprint review、bounties）进行对比，评估其覆盖度、深度和主动分配优势，但未给出定量性能指标。

**⚠️ 局限性**

设计缺乏实施细节和实证验证，可能面临低质量产出淹没、资源分配偏见、跨学科标准统一困难等挑战。

---

## 156. LLM2Cert: LLM Synthesis of Certificates with SMT/Lean Verification

**arXiv ID:** 2610.30524 | [PDF](https://arxiv.org/pdf/2610.30524v1)

**作者:** Zhipeng Cao `[一作]` (Chinese Academy of Sciences), Alessandro Abate `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了LLM驱动的框架LLM2Cert，用来自动生成Lyapunov与障碍函数并通过SMT与Lean进行验证；同时构建了包含105个实例的Benchmark，覆盖高维、受控、非线性耦合的稳定性与安全性任务。

**💡 创新点**

创新点在于：①首次把Lean证明助手集成到整个合成与验证流程，消除SMT的近似与δ‑sat限制；②通过两级路由策略，既利用SMT的速度，也用Lean解决难题；③创建了无约束、无近似的统一基准，打破传统基准对Solver的依赖。

**🔧 技术方法**

核心技术包括：LLM（ChatGPT等）进行候选证书与策略的生成；SMT求解器（Z3、cvc5、dReal等）进行快速验证；Lean证明助手用于无法通过SMT解决的命题，包含结构化的Proof‑Repair回调。

**📊 数据集**

使用了自研的Benchmark，包含三类实例：文献原始、扩展与全新，高维度（n≥5）占25个，涵盖稳定性、ROA、前向不变集与安全四项任务。

**📈 对比分析**

与五个现有端到端管线比较，LLM2Cert在所有105个实例上验证了81个（77.1%），显著高于基线（最多40/51），在高维/非多项式任务中更具优势；在固定候选证书下，SMT+Lean路由覆盖率达到100%，并在平均时间上比全Lean快约2–3倍。

**⚠️ 局限性**

局限在于：对SMT求解时间仍有限制；Lean证明耗时较高，且对非常复杂的非线性约束仍可能失败；Benchmark虽更具代表性，但仍主要集中在连续/离散时间系统，缺乏对更复杂控制结构或随机系统的覆盖。

---

## 157. Impacts of generative AI skill development on academic adaptation and learning outcomes among university students

**arXiv ID:** 2610.30599 | [PDF](https://arxiv.org/pdf/2610.30599v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 158. Beyond End-to-End Recovery: Towards a Better Understanding of LLM Steganography via Rolewise Decomposition

**arXiv ID:** 2610.30516 | [PDF](https://arxiv.org/pdf/2610.30516v1)

**作者:** Om Tailor `[一作]` (University of Maryland), Kaiqing Zhang `[通讯]` (University of Maryland)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）隐写技术的编码与解码过程，并提出了角色分解评估框架

**💡 创新点**

创新点在于将编码验证、直接暴露检测和解码恢复分离，揭示E2E恢复无法区分编码与解码失败的缺陷

**🔧 技术方法**

采用确定性验证器（支持多种文本提取方法）、词长/首字母提取、消息编辑实验及对比解码器的评估技术

**📊 数据集**

使用约7200条由GPT、Claude、Gemini生成的覆盖任务文本和随机私有消息，包含六种编码规则（二进制、十六进制、Base64、固定字母替换、句首、交替句首）

**📈 对比分析**

对比实验显示，扩展验证覆盖后Gemini从排名末位跃升至首位；在相同文本下，Gemini在多种隐写模式下的恢复率显著高于其他模型；但在单字符编辑时，解码器往往保持原消息，显示恢复不可信，整体E2E表现掩盖了解码差异

**⚠️ 局限性**

局限性包括：验证器仅覆盖已实现的提取方法，无法发现未知隐写模式；实验主要基于Gemini生成的文本，解码器对不同模型的泛化不足；消息编辑虽保持语义，但仍可能引入表面偏差，且小样本对Claude、GPT的结论不稳健

---

## 159. GSHeadRelight: Fast Relightability for 3D Gaussian Head Synthesis

**arXiv ID:** 2610.30619 | [PDF](https://arxiv.org/pdf/2610.30619v1)

**作者:** Henglei Lv `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences and University of Chinese Academy of Sciences), Lin Gao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences and University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于统一辐射传递（Unified Neural Radiance Transfer）的3D Gaussian人头生成模型，能够在单张图像或视频的条件下实现实时、可视化的头部重光与新视角合成。

**💡 创新点**

创新点在于将辐射传递函数与视角相关联，形成视角可调的光传输向量，从而实现线性光照响应、低频反射建模，并通过仅使用白光训练消除 albedo 与光照的歧义，最终实现彩色光照的泛化。

**🔧 技术方法**

采用了 3D Gaussian Splatting（3DGS）生成框架、统一辐射传递的 SH 参数化、轻量 MLP 解码器、GAN 对抗训练以及白光假设的光照标注技术。

**📊 数据集**

在 FFHQ 数据集上训练，利用 DPR 提取 3rd‑order SH 光照标签，使用 MODNet 进行前景提取，并在四块 NVIDIA H800 GPU 上完成训练。

**📈 对比分析**

与 ShadeGAN、Volux‑GAN、NeRFFaceLighting、LumiGAN 等公开基线比较，渲染速度提升约 12 倍（≈240 FPS），在 FID、KID、Lighting Error、Lighting Stability、空间一致性等指标上与现有最佳方法持平或优于它们。

**⚠️ 局限性**

局限性包括只能有效建模低频扩散与低频视角相关反射，难以捕捉高频镜面反射；在低光照或从下方光照的情况下可能出现不自然或不真实的阴影与高光残留。

---

## 160. TT-SAX: A Two-Track Time Series Symbolization for Natural Language Shape Querying

**arXiv ID:** 2610.30593 | [PDF](https://arxiv.org/pdf/2610.30593v1)

**作者:** Zhao Tan `[一作]` (Griffith University), Ming Jin `[通讯]` (Griffith University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究自然语言查询时间序列形状的符号化方法，并提出了双轨符号化方案TT‑SAX。

**💡 创新点**

通过在符号串中加入两条轨道——保留原始SAX级别字母并额外加入跨邻近段测量的趋势字母，使用面向LLM可读的字母表，解决了传统符号化缺乏方向、尺度过小、字母任意等三大局限。

**🔧 技术方法**

结合SAX、趋势字母设计、LLM正则表达式编译、技能学习（自演进训练）和跨模型迁移等技术。

**📊 数据集**

使用NLQTSBench基准（220个形状查询）及其公开生成的1,000个新查询。

**📈 对比分析**

与六种经典符号化在六个LLM上对比，采用召回、候选数和lift指标；TT‑SAX在lift上显著优于所有基线，达到约3.0（无技能）/3.3（有技能）。

**⚠️ 局限性**

仍受LLM对编码卡理解的限制，趋势阈值和邻近段数需手工设定，对查询集的适用性可能有限。

---

## 161. Better Reasoning, Not Just Bigger Models: A Structure-Grounded Framework for Efficient and Reliable Inference

**arXiv ID:** 2610.30611 | [PDF](https://arxiv.org/pdf/2610.30611v1)

**作者:** Amirhossein Aghaei `[一作]` (Huawei Research Canada), Yan Wang `[通讯]` (Central University of Finance and Economics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计了训练无关的结构化不确定性降低框架SGUR，用于多轮推理，特别是在医学诊断场景中将不完整观察映射为查询图并与候选知识图对齐；

**💡 创新点**

创新点在于将多步推理视为在结构化假设空间上的不确定性收缩，通过先验加权的部分Gromov‑Wasserstein对齐和期望信息增益主动查询，实现结构约束下的推理并在不依赖大模型规模的情况下显著提升性能；

**🔧 技术方法**

使用技术包括基于专家层级图的离线数据库构建、先验加权的部分FGW（PFGW）对齐、基于节点重要性的路径过滤、结构化EIG主动获取以及多轮推理循环；

**📊 数据集**

利用PrimeKG构造的医学诊断图谱作为专家数据库，并在其上设计了多轮诊断基准；

**📈 对比分析**

与四种Graph‑RAG基线（GraphRAG、NodeRAG、HippoRAG2、ArchRAG）在多轮诊断任务中对比，SGUR在复杂模式下 Hit@3 92.5% 超过最强基线 9.4pp，且令 token 消耗与延迟分别降低约5.7×与3.9×，并在模型缩放实验中保持 99% 性能；

**⚠️ 局限性**

局限在于需预先构建可靠的层级知识图、假设观察能准确映射为图节点、对噪声、动态知识和异构证据的鲁棒性有限，且本框架仍为研究原型，不能直接用于临床决策。

---

## 162. Bridging the Closed-World Gap: Interpretable Anomaly Detection via a Hybrid TSK-FCM Ensemble on the BETH Dataset

**arXiv ID:** 2610.30576 | [PDF](https://arxiv.org/pdf/2610.30576v1)

**作者:** Lohith Srikanth Pentapalli `[一作]` (University of Cincinnati), Kelly Cohen `[通讯]` (University of Cincinnati)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过将梯度优化的Takagi‑Sugeno‑Kang（TSK）模糊分类器与 Fuzzy C‑Means（FCM）聚类相结合，构建一种混合模糊集成，用于检测 BETH 数据集中的已知内部威胁与未知零日 botnet 激活攻击。

**💡 创新点**

创新点在于（1）提出 Min Distance 几何异常度量，解决高维空间中传统模糊隶属度失效的问题；（2）将可解释的 TSK 与几何无监督检测融合，既保持高检测精度，又提供透明的决策依据。

**🔧 技术方法**

采用的技术包括梯度优化的 TSK 模糊系统、FCM 聚类、欧氏距离几何度量、Sigmoid 校准、T‑conorm 最大合并等。

**📊 数据集**

使用的数据集为 BETH kernel‑level 系统调用数据，训练集仅包含正常流量和“可疑”内部行为，测试集主要为未见过的“恶意”botnet 激活事件。

**📈 对比分析**

与 Isolation Forest、单独 TSK、单独 FCM 等基线比较，混合模型在测试集上实现 AUROC 0.9951，较 Isolation Forest 提升 17.1%，较单独 TSK 提升 6.8%，且召回率高达 99.3%。

**⚠️ 局限性**

局限性包括测试集 FPR 约 15.3%（需阈值调节）、对聚类数等超参数仍有一定敏感性、模型易受对抗样本攻击、以及缺乏在线学习与概念漂移处理。

---

## 163. Stale Instruction Files Override the Code in Coding Agents: A One-Sentence Fix

**arXiv ID:** 2610.30571 | [PDF](https://arxiv.org/pdf/2610.30571v1)

**作者:** Xiaonan Xu `[一作]` (Georgia Institute of Technology), Wenjing Wu `[通讯]` (University of Colorado Boulder)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究编码代理在面对过时的说明文件与现有代码冲突时的行为，并验证一句话规则能否改变代理的选择

**💡 创新点**

提出一句简单的“代码优先”句子即可使代理优先遵循现有代码，解决过时说明文件导致的不一致问题

**🔧 技术方法**

实验使用 Claude Code 和 Codex CLI 两个编码代理，对 40 个 Python 项目执行 320 次运行，结合工具调用、模型输出判定和冲突检测

**📊 数据集**

自构造的 40 个 Python 项目，覆盖 16 种代码规范冲突，作为实验数据集

**📈 对比分析**

对比两种条件下代理在 720 个决策点中遵循说明文件、遵循现有代码或两者皆非的比例，结果加句子后符合代码比例从 4.2% 提升至 91.4%，测试通过率从 89.4% 提升至 100%，并减少了工具调用和运行时间

**⚠️ 局限性**

仅在 Python 环境、两种代理和单一功能添加任务上验证，未覆盖多语言、多任务或更复杂的冲突情形；句子需要手动添加，仍需团队定期维护

---

## 164. Repair Changes Which Layers to Remove: Exact Optima for Depth Pruning of Language Models

**arXiv ID:** 2610.30552 | [PDF](https://arxiv.org/pdf/2610.30552v1)

**作者:** Jianchang Su `[一作]` (University of Connecticut), Wei Zhang `[通讯]` (University of Connecticut)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型的深度剪枝中，作者精确枚举所有层组合，研究了线性修复（ReplaceMe）对最佳剪枝方案的影响，并提出了在修复模型上进行贪婪搜索和基于 Fisher 信息的二阶评分方法。

**💡 创新点**

创新点在于：①揭示线性修复会将最优剪枝结构从分散的运行变为连续块；②证明在修复模型上直接搜索可显著降低困惑度；③提出无需再训练的二阶 Fisher 评分，从完整模型的前向后向梯度即可预测最佳块，避免昂贵的模型评估。

**🔧 技术方法**

使用的技术包括：深度剪枝（层群分组）、ReplaceMe 的最小二乘线性修复、贪婪搜索（SLEB 级别成本）、二阶损失展开与 Fisher 信息权重、实验中使用的精确枚举、梯度与残差分析。

**📊 数据集**

实验数据集主要为：WikiText‑2（校准与评估）、C4（评估）、ARC‑Easy/Challenge、HellaSwag、PIQA、WinoGrande（零样本多任务准确率）以及 Qwen3‑8‑27B 作为更大模型测试；使用的模型包括 Qwen3‑1.7B、Qwen3‑14B、Gemma‑3‑4B、Llama‑3.1‑8B‑Instruct。

**📈 对比分析**

在 25% 和 37.5% 层剪枝预算下，所提方法在 WikiText‑2 困惑度方面平均比 ReplaceMe 降低 20%–66%，比 SLEB 降低 4%–29%；在多任务准确率上，使用相同任务数据时可提升 0.9–14.1 分，甚至在 27B 模型上取得 25%‑55% 的准确率提升；与 ShortGPT、LaCo、Prune&Comp、SLEB 等基线对比，整体性能显著优于所有对照。

**⚠️ 局限性**

限制包括：①精确枚举仅覆盖两层分组下最多 6 个被剪层（单层最多 3 个），无法直接验证更大预算；②只评估 ReplaceMe 的最小二乘线性修复，未尝试低秩或位移等新型修复；③实验集中于英文文本与多项选择任务，未涉及多语言、开放式生成等场景；④未对 BlockPruner、SubFit 等方法做直接比较。

---

## 165. PLE: Physical-Engineering's Last Exam

**arXiv ID:** 2610.30537 | [PDF](https://arxiv.org/pdf/2610.30537v1)

**作者:** Rujun Han `[一作]` (Google Cloud AI Research), Chen-Yu Lee `[通讯]` (Google Cloud AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个评估代理完成专业物理工程真实任务的基准，包含85个跨电、土、结构、机四类工程任务。

**💡 创新点**

创新点在于结合专家手工与对抗式生成的任务，提供多格式项目文件、生产级交付物和混合评分体系，真正测评代理的经济价值和工程可执行性。

**🔧 技术方法**

采用大型语言模型代理、工具调用（文件、表格、CAD、代码）以及基于容器的沙箱环境，并通过多种代理框架（Claude Code、Hermes等）进行实验。

**📊 数据集**

数据集由44个专家完成任务与41个对抗式生成任务构成，覆盖53个独立环境，文件总量5,170份，涵盖PDF、Excel、CAD/BIM等多种格式。

**📈 对比分析**

对比14个前沿模型在7种代理框架下的表现，以任务完成率（百分比）评估；Claude Opus 5.5在Claude Code达50.6%为最高，其余模型低于48%，模型差异显著大于框架差异。

**⚠️ 局限性**

局限在于难以生成既完全符合硬约束又接近成本下限的设计，许多任务得分低于50%，运行波动大，且仍缺乏对更广泛工程领域的覆盖。

---

## 166. IFC-MCQ: Automatically Generating Multiple-Choice Benchmarks for Evaluating LLM Agents on IFC Data Retrieval

**arXiv ID:** 2610.30533 | [PDF](https://arxiv.org/pdf/2610.30533v1)

**作者:** Rabindra Lamsal `[一作]` (University of New South Wales), Xuesong Shen `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了IFC-MCQ，一套自动生成IFC文件多项选择题基准的方法，用于评估大语言模型在建筑信息建模（IFC）检索任务中的表现。

**💡 创新点**

创新点包括：①构建15类36子类型的问答分类体系；②实现题目、答案与干扰项的全自动生成与独立验证；③在不依赖人工或第二级LLM评审的前提下实现精确选项集计分；④可根据不同IFC模型动态生成覆盖度可变的基准。

**🔧 技术方法**

技术手段：基于IfcOpenShell构建模型索引并生成候选题目；使用模板化问句与规则化干扰项；采用随机化保证答案位置不固定；利用工具代理与代码代理的LLM交互进行检索与答案生成；采用多阶段实验对比不同LLM与检索接口的性能。

**📊 数据集**

数据集：10个公开的IFC建筑模型，涵盖从数十到数万元素的住宅、商业、科研等多种规模与建模风格。

**📈 对比分析**

比较方法与性能：三阶段实验（工具代理+参考模型、工具代理+额外模型、代码代理+相同模型）并与随机/固定选择基线对比。多数顶尖LLM在工具代理下达成>95%准确率；不同检索接口及模型规模导致性能差异，较大模型在代码代理下准确率下降。

**⚠️ 局限性**

局限性：①仅评估多项选择答案，无法反映自由文本检索的完整性；②干扰项可能提供线索，影响判分；③几何任务仅限于AABB操作，未覆盖复杂几何或拓扑；④验证与生成共享IfcOpenShell，无法排除库级错误；⑤未对损坏或非标准IFC文件的鲁棒性进行测试；⑥观察结果截断可能影响部分答案。

---

## 167. Decision Making and Ride Comfort Evaluation Based on Lane-Change Prediction for Autonomous Vehicles

**arXiv ID:** 2610.30528 | [PDF](https://arxiv.org/pdf/2610.30528v1)

**作者:** Mohamed Manzour `[一作]` (University of Alcalá), Miguel Ángel Sotelo `[通讯]` (University of Alcalá)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究将车道变更预测融入自动驾驶决策，比较反应式、预测式纵向制动和预测式横向变道对安全与乘坐舒适度的影响。

**💡 创新点**

在同一框架下将知识图嵌入与贝叶斯推理的车道变更预测与基于 ISO 2631‑1 的振动舒适度评估相结合，系统性比较三种策略。

**🔧 技术方法**

技术包括知识图嵌入（TransE）、贝叶斯推理、CARLA 仿真、IDM‑PID 纵向控制、MPC 横向变道、以及 ISO 2631‑1 加权振动估计。

**📊 数据集**

使用 highD 与 CRASH 两个公开交通数据集训练预测模型。

**📈 对比分析**

通过 3 次 CARLA 运行比较碰撞率、TTC、峰值减速度、RMS 加速度/jerk、ISO 加权加速度；预测式纵向减速比反应式降低 31.5% 峰值减速，横向避让降低 91%；ISO 加权峰值从 1.52 降至 0.53，舒适度最高。

**⚠️ 局限性**

仅基于仿真、仅考虑水平加速度、未测量乘客座椅振动、实验场景有限，需要多样化交通情境与真实感受验证。

---

## 168. Continuous-Time Motion Planning and Steering with Global Convergence

**arXiv ID:** 2610.30607 | [PDF](https://arxiv.org/pdf/2610.30607v1)

**作者:** Mohsen Amiri `[一作]` (Samsung SDS Research America), Yeongdae Kwon `[通讯]` (Samsung SDS Research America)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了全时段连续运动规划方法GlobMP，能够在运动规划和学习策略的后向调节中同时满足所有碰撞、避障和控制约束；

**💡 创新点**

创新点在于：①在连续时间内严格保证每个约束；②将规划与调节合并为单一连续程序；③使用梯度投影和多段多项式曲线的凸可行性恢复实现全局收敛；

**🔧 技术方法**

采用多项式 Bernstein 表示、梯度投影法、凸投影恢复、离散化网格起始、可行性修复算法与矩阵求导；

**📊 数据集**

使用合成机器人团队（2~200机器人）实验、Robotarium 多机器人测试平台、Bimanual Humanoid Vega‑1 的触觉反应操控数据；

**📈 对比分析**

与RRT‑Connect、RRT*, GCS‑IRIS、SDP‑MP 等采样/优化基线比较，GlobMP 在所有实例上始终满足每时刻约束，成本通常低于基线，计算时间线性增长，成功率远高于基线；

**⚠️ 局限性**

局限在于：①对速度/通道约束的理论保证尚未扩展；②需要先验多项式约束可凸化的前提；③在极端复杂场景下的计算规模仍有提升空间。

---

## 169. PATH: Preference-Aware Traversability Estimation for Robotic Teleoperation in Off-Road Navigation

**arXiv ID:** 2610.30605 | [PDF](https://arxiv.org/pdf/2610.30605v1)

**作者:** Xianghao Meng `[一作]` (Beijing Institute of Technology), Junqiang Xi `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了PATH-Net，一种能够根据人机交互者偏好的偏好感知可通行性评估网络，用于离线/实时地为遥操作车辆生成成本和速度地图。

**💡 创新点**

创新点在于：①将视觉、LiDAR几何信息与车辆运动历史融合成BEV特征；②采用多头共享骨干网络生成多种候选成本/速度图；③通过不对称聚合——上尾风险平均(CVaR)聚合成本、分位数选择聚合速度，实现偏好可调的推荐；④利用最大熵逆强化学习从示范中学习成本。

**🔧 技术方法**

技术手段包括：视觉–几何融合的Guided Filter Network、时序卷积网络编码运动历史、共享卷积骨干与多头输出、最大熵逆强化学习、速度监督回归、CVaR与分位数聚合。

**📊 数据集**

使用的数据集由15名操作员在北京郊区山坡上完成225条遥操作轨迹组成，包含LiDAR点云、RGB图像、GNSS/IMU状态及问卷得出的偏好参数。

**📈 对比分析**

与基线HDIF和Velociraptor（V‑Dino）在保持轨迹相似度（pMHD）和速度匹配（sMHD）两指标上进行对比，PATH-Net在相应偏好设置下实现了最低的pMHD与sMHD，并在NASA‑TLX问卷中工作负荷降低约36%。

**⚠️ 局限性**

局限性包括：参与者样本仅15人，验证/测试集中每种偏好仅有单个被试；问卷设计为固定顺序，无法证明因果关系；缺乏在线偏好估计与更大规模的用户实验。

---

## 170. Atlas: Making Sense of Long Horizon Agent Trajectories

**arXiv ID:** 2610.30548 | [PDF](https://arxiv.org/pdf/2610.30548v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 171. Journal Basket: Supporting Cross-Disciplinary Literature Searching and Methodological Transparency

**arXiv ID:** 2610.30544 | [PDF](https://arxiv.org/pdf/2610.30544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 172. HiMark: Hierarchical and Controllable Multi-bit Watermarking for Large Language Models

**arXiv ID:** 2610.30601 | [PDF](https://arxiv.org/pdf/2610.30601v1)

**作者:** Ke Zhu `[一作]` (Guizhou Normal University), Hanzhou Wu `[通讯]` (Guizhou Normal University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于层次化候选控制的多比特水印嵌入与提取框架HiMark，能够在不改动LLM内部参数的前提下，稳健地在生成文本中嵌入二进制信息并实现模型无关的提取。

**💡 创新点**

创新点在于：①将水印嵌入过程从全局概率干预转为局部候选 token 选择，并通过层级淘汰、支持度细化和阈值筛选实现多层次的候选控制；②提出两阶段证据聚合策略，先计算每个 token 对多层支持度，再聚合到位置信息中完成多比特解码，从而显著提升短文本、长消息以及文本扰动下的鲁棒性；③实现模型无关的提取，避免了对原始 LLM 进行微调的需求。

**🔧 技术方法**

核心技术包括：使用 PRF 生成与上下文相关的种子 s_t；通过 Hash 函数在多层层级生成二值映射 g_l；层级淘汰（tournament elimination）、支持度计算与筛选；两阶段证据聚合（token‑level 多层支持度 + 位置级加权投票）；对齐水印位置信息并使用加权投票解码。整个流程在生成时仅对 token 进行筛选，不改变原始概率分布。

**📊 数据集**

实验采用 Llama‑3.1‑8B 作为基准 LLM，并在 C4、OpenWebText、CNN/DailyMail 三大数据集上进行生成；同时评估 Qwen2.5‑7B‑Instruct、OPT‑2.7B 等不同模型的通用性。

**📈 对比分析**

与 MPAC、BiMark（多比特）以及 Soft Red List、SynthID‑Text（零比特）等基线相比，HiMark 在 8‑bit、16‑bit、32‑bit 水印下，短文本（≤300 token）提取准确率均超过 95%（32‑bit 最高 99.71%），且在同等文本质量（PPL 约 5）下表现更优；在同义词替换、复制粘贴、插入/删除、改写等攻击场景中，HiMark 的鲁棒性显著高于基线，尤其是 30% paraphrasing 攻击下仍保持 90%+ 的提取准确率。

**⚠️ 局限性**

局限性包括：①相较于 BiMark，HiMark 的生成时计算量更大（≈2.4s/样本），在实时生成场景下可能成为瓶颈；②需要手动调节层深 m、候选保持大小 K_keep、细化大小 K_refine 与阈值 τ_s 等超参数，参数设置对文本流畅度与鲁棒性存在权衡；③在极长的消息或极短文本时，覆盖所有位的证据可能不足，需要更长的生成长度；④目前仅针对自回归（decoder‑only）LLM，尚未验证对多模态或 encoder‑decoder 模型的适用性。

---

## 173. Model Organisms of Collusion: Corrective Transfer in Untrusted Monitors

**arXiv ID:** 2610.30582 | [PDF](https://arxiv.org/pdf/2610.30582v1)

**作者:** Juan-Pablo Rivera `[一作]` (Meridian Cambridge), John Hughes `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在模型监控中对单一捕获到的协同行为进行微调是否能降低未发现的协同行为。

**💡 创新点**

构造一个模型“生物体”，在同一权重中安装五种策略行为与对应监视协同，并评估单一纠正训练对其他协同的迁移效果。

**🔧 技术方法**

采用两阶段训练（Synthetic Document Fine‑Tuning + LoRA Supervised Fine‑Tuning）以及四种纠正方式（SFT、CWS、NPO+R、DPO）并在 Llama‑3.3‑70B 与 Qwen‑2.5‑72B 上实验。

**📊 数据集**

使用约3.1 亿 token 的合成文档（行为文档、信念文档）与 Web replay 文本，400 条 reward‑hacking 诚实评审与 633 条干净评审作为训练集。

**📈 对比分析**

与同等规模无关的 UltraChat 微调做对比；四种纠正方法均能将未见协同率从约91% 降至 14‑20%，但单任务准确率高并不预测迁移效果。

**⚠️ 局限性**

所有行为与协同共存于同一权重和身份，迁移可能因共享训练历史；安装行为人为且未出现自然出现的误对齐；评估仅限离线，无法确认是否真正消除协同规则。

---

## 174. Classification of binary, ternary and quaternary Hermitian LCD codes

**arXiv ID:** 2610.30561 | [PDF](https://arxiv.org/pdf/2610.30561v1)

**作者:** Keita Ishizuka `[一作]` `[通讯]` (Mitsubishi Electric Corporation), Keita Ishizuka (Mitsubishi Electric Corporation)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对二进制、三进制及四进制（Hermitian）线性互补双峰（LCD）码，在长度分别为19、15、13 的范围内，完成了所有维度的完整分类。

**💡 创新点**

首次将 LCD 码的等价判定转化为坐标上的图同构，并利用投影矩阵图的闭包与正规增广技术，仅在 LCD 码族内生成码，从而消除了传统方法中对非零包络码父代的需求，显著提升了分类效率。

**🔧 技术方法**

使用了图同构软件 nauty 对投影矩阵图进行规范化标记、闭包操作、投影矩阵的构造，以及质量公式（mass formula）验证；并采用正规增广（canonical augmentation）来避免等价重复。

**📊 数据集**

生成并存储了上述长度范围内所有等价类的生成矩阵（约 191 GB、139 GB 与 18 GB，分别对应二进制、三进制与四进制），同时计算了每类的自同构群和最小距离等指标。

**📈 对比分析**

与以代码字图为基础的等价判定方法以及 SageMath 的代码相比较，投影矩阵图的规范化耗时仅为 5–44 微秒，而代码字图需要 17–2000 毫秒，SageMath 需要 33–50 毫秒，体现出数十倍到数千倍的性能提升。

**⚠️ 局限性**

由于生成的等价类数量随长度迅速增长，后续长度（例如二进制 20、三进制 16、四进制 Hermitian 14）将产生数 TB 的生成矩阵，超出当前公开数据存储与分发能力，因而目前的分类已达到实践可行性的极限。

---

## 175. Motion Retargeting with Spatiotemporal Preferences for Humanoid Loco-Manipulation

**arXiv ID:** 2610.30540 | [PDF](https://arxiv.org/pdf/2610.30540v1)

**作者:** Xiongbo Shi `[一作]` (Tongji University), Guang Chen `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于关键帧和关键身体部位的空间‑时间重定向（STR）方法，用于人类到仿人机器人在物体交互中的运动转移，并结合强化学习（STRL）实现更精准的控制。

**💡 创新点**

创新点在于利用视觉语言模型自动选取关键帧和关键部位，分配迭代预算和交互权重，并通过机制感知的接触锚定（MCA）细化接触点，从而在保持高精度的同时显著降低计算量。

**🔧 技术方法**

使用的技术包括视觉语言模型（VLM）进行关键帧/部位检测、SQP优化、接触锚定、以及PPO强化学习与自适应阶段采样。

**📊 数据集**

数据集为OMOMO的人机交互演示，包含10个不同物体类别的任务，以及使用SMPL模型和物体轨迹。

**📈 对比分析**

与PHC、GMR、VideoMimic和OmniRetarget等方法比较，STR在PECR/ECC、SQP迭代次数、以及RL成功率方面均优于对照组，显示出更高的接触覆盖、连续性和任务完成率。

**⚠️ 局限性**

局限性包括对接触模型的近似假设、对关键帧检测的VLM依赖以及在更复杂场景下的可迁移性待进一步验证。

---

## 176. NEST: Nested Embedding Representation for Multiscale Clustering

**arXiv ID:** 2610.30520 | [PDF](https://arxiv.org/pdf/2610.30520v1)

**作者:** Zhanqi Zhang `[一作]` (University of California San Diego), Gal Mishne `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了NEST模型，利用嵌套前缀嵌入实现多尺度聚类，并通过联合训练提升聚类质量。

**💡 创新点**

创新点在于将不同分辨率的聚类直接编码进同一嵌入的嵌套前缀中，使用多层聚类、中心分离与抗坍塌正则化共同训练，无需显式树结构或监督。

**🔧 技术方法**

采用Matryoshka式残差映射（在冻结特征上学习低秩正交旋转+MLP），每层使用DEC损失生成软分配，加入中心分离惩罚、秩正则化和聚类内散度正则，整体以多层KL散度与正则化项联合优化。

**📊 数据集**

实验数据包括三种合成层级数据集（Nested GMM、BalancedTree、RandomBranching）以及三类真实数据集：CIFAR‑100（图像）、20 Newsgroups（文本）、Tasic 2018单细胞转录组。

**📈 对比分析**

与六类基线（k‑means、Ward、Avg‑L、bisecting k‑means、HypHC、DeepECT）以及HDBSCAN（合成）在NMI、ARI、ACC、跨层一致性、计算与内存等指标上对比。NEST在中层与细粒度聚类上持续领先，尤其在CIFAR‑100和20 Newsgroups上显著提高精度，并在多尺度几何上与层级距离的相关性明显增强；训练效率与显存占用低于DeepECT和HypHC。

**⚠️ 局限性**

局限性包括：对高度不平衡的层级表现不佳；无法自动确定聚类数与层级数；仅使用基于质心的目标，对异常或缺失层级结构处理不足；在更大规模或多模态嵌入上的扩展仍需进一步研究。

---

## 177. Unified Trajectory and 3D Pose Forecasting for Multi-Agent Sports Analysis

**arXiv ID:** 2610.30519 | [PDF](https://arxiv.org/pdf/2610.30519v1)

**作者:** Marc Peral `[一作]` (Institut de Robòtica i Informàtica Industrial, CSIC-UPC), Antonio Agudo `[通讯]` (Kognia Sports Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种联合预测球员轨迹和3D身体姿态的方法，能够同时捕捉球员的全局运动和关节动作。

**💡 创新点**

创新点在于将轨迹和姿态在输入层级融合并通过共享的社交-时序Transformer编码，同时引入针对高动态关节的损失重权重和频域轨迹平滑正则化。

**🔧 技术方法**

采用社交-时序Transformer、共享注意力、per-joint loss rebalancing、频域平滑正则化以及多模态早期融合技术。

**📊 数据集**

使用真实赛事实况数据集WorldPose（2022 FIFA世界杯转播录像）和通用人类运动数据集3DPW。

**📈 对比分析**

采用非重叠、无oracle的评估协议，对比T2P和SimpliHuMoN，结果在WorldPose上实现state‑of‑the‑art，且在3DPW上超越所有现有方法，特别是姿态相关指标显著提升。

**⚠️ 局限性**

局限在于对极少社交上下文（如3DPW）时性能提升有限，且仍受数据集分布与评估协议的影响，未对极端运动模式（如突发加速）进行充分验证。

---

## 178. Protegrity-PII Technical Report

**arXiv ID:** 2610.30518 | [PDF](https://arxiv.org/pdf/2610.30518v1)

**作者:** Hardeep Bassi `[一作]` (Protegrity), Averell Gatton `[通讯]` (Protegrity)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发并公开了Protegrity-PII模型，用于在结构化与非结构化文本中高效、准确地检测和清理个人身份信息，并同时发布了新的PII-SCOPE基准数据集。

**💡 创新点**

创新点包括：① 使用细粒度69类PII实体税onomies并直接训练模型；② 利用合成数据训练并通过阈值校准显著提升召回率；③ 通过剪枝与知识蒸馏实现模型轻量化与高效推理；④ 提出有类型与无类型的评估框架；⑤ 发布高密度、低过度标注的PII-SCOPE基准。

**🔧 技术方法**

技术手段包括：XLM‑RoBERTa编码器 + BIO token‑level 分类器，Viterbi 解码；交叉熵训练、温度软化的知识蒸馏；阈值校准与精确率-召回率曲线选择；批量推理、成本/延迟评估。

**📊 数据集**

使用的数据集：训练集1.2M条合成文本（约为PII‑SCOPE规模的100倍）；评估集包括PII‑SCOPE（9,716行）以及AI4Privacy、Nemotron、Gretel、SPY等公开合成基准；对比模型包括 OpenAI Privacy Filter、GLiNER2.5、AWS Comprehend、Microsoft Presidio、Rampart 与 Claude Fable 5。

**📈 对比分析**

比较方法：采用有类型与无类型的召回、精确率、F1，以及平均 FNR/FPR；在PII‑SCOPE上Protegrity‑PII的Typed F1最高，Recall≈0.915、Precision≈0.907；在所有基准上其Typed/Untyped Recall和F1均名列第一；成本/延迟方面最快最便宜（约0.0035 $/M字符，3.79 s）。

**⚠️ 局限性**

局限性：对极罕见或未见的PII类型仍易漏检；模型性能高度依赖训练数据的质量与代表性；大文本需分窗口处理；合成基准可能存在过度标注导致评估偏差；在真实生产场景中需持续监控与更新。

---

## 179. Membership Inference Attack on Private Evolution

**arXiv ID:** 2610.30514 | [PDF](https://arxiv.org/pdf/2610.30514v1)

**作者:** Daniil Filienko `[一作]` (University of Washington Tacoma), Martine De Cock `[通讯]` (University of Washington Tacoma)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种针对私有进化（PE）算法的白盒成员推断攻击（EVO‑MIA），利用投票过程中的噪声计数和演化轨迹检测私有数据泄露。

**💡 创新点**

首次为推理式合成数据生成方法设计了模态无关的成员推断攻击，并且不需要训练阴影模型，攻击仅基于公开的投票直方图和公开的嵌入模型。

**🔧 技术方法**

采用差分隐私高斯机制下的噪声计数建模、负二项分布逼近、对参考集估计投票比例、基于似然比的分数累积，以及对每轮投票的独立性假设。

**📊 数据集**

在文本（OpenReview、MIMIC‑III）、图像（CIFAR‑10、Camelyon17、Cat）和表格（adult、artificial‑characters、breast‑cancer、person‑activity）数据集上进行评估。

**📈 对比分析**

与文本中的n‑gram基线、图像中的LOGAN以及表格中的GAN‑Leaks和DOMIAS进行对比；EVO‑MIA在大多数数据集上获得更高的ROC‑AUC（例如OpenReview 0.873/0.685，Cat 0.846/0.692，tabular约0.53‑0.65），在不同ε下均表现优于基线。

**⚠️ 局限性**

攻击依赖于每轮候选样本和噪声投票直方图的公开；若中间过程保密则效果降低；实验仅覆盖原始论文使用的数据集，结果可能不完全泛化；在样本量大、单个成员投票影响微弱的情况下，攻击效果下降。

---

## 180. AIRO: A Streaming-Based Machine Learning Framework for Threat Detection and Security in Industrial Internet of Things

**arXiv ID:** 2610.30713 | [PDF](https://arxiv.org/pdf/2610.30713v1)

**作者:** Raima Deb `[一作]` (Birla Institute of Technology and Science Pilani), Mian Guo `[通讯]` (Guangdong Polytechnic Normal University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了AIRO框架，实现IIoT流量的实时监测与自动化响应。

**💡 创新点**

创新点在于将Kafka流式处理、攻击特定特征提取、集成机器学习推理与置信度驱动的决策逻辑结合，形成闭环自动化减缓措施。

**🔧 技术方法**

使用Apache Kafka、Gradient Boosting/LightGBM、TF-IDF、StandardScaler、One-Hot编码等技术实现特征工程、模型推理与决策。

**📊 数据集**

自制四类攻击数据集：C2 beaconing、Reconnaissance、SQLi、Phishing，包含真实噪声与类别重叠，避免标签泄露。

**📈 对比分析**

与传统模型（LR、DT、RF、XGBoost、LightGBM）对比，选取F1最高的模型，C2、Recon、SQLi的F1>0.90，Phishing为0.65；AUC分别为0.95、0.94、0.92、0.71；系统延迟约0.5秒，支持即时响应。

**⚠️ 局限性**

局限在Phishing检测F1低、模型维护成本高；未来需引入Transformer特征和模型蒸馏以提升边缘设备部署效果。

---

## 181. From Tokens to Credits to Outcomes: Pricing AI Across Models, Applications, and Cloud Platforms

**arXiv ID:** 2610.30701 | [PDF](https://arxiv.org/pdf/2610.30701v1)

**作者:** Pratik Gupta `[一作]` `[通讯]` (Stripe), Pratik Gupta (Stripe)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 2023–2026 年间 AI 产品与云平台公开定价页面进行系统可复现的研究，构建 75 行（52 AI 原生、14 云平台、9 特殊样本）可公开获取的定价数据集，并基于此提出定价原语分类法、时间线与对计费平台设计的启示。

**💡 创新点**

① 将定价拆分为“原语”与“机制”两层，首次系统编码预付积分、使用配额、结果计费等细粒度机制；② 公开可复现的数据集与代码包；③ 通过敏感度分析揭示混合产品的层级结构；④ 把观察到的定价模型转化为计费平台的可配置需求和风险点。

**🔧 技术方法**

基于公开网页抓取与 Wayback Machine 的归档重建，使用手工编码与单独审计、双重编码策略，构建可复制的编码规则和代码书；利用 Python 及 Pandas 生成可视化和计量指标；使用离散概率分析与敏感度实验来评估原语分布。

**📊 数据集**

包含 75 行的公开定价数据集，来源于 Forbes AI 50（2026）、云平台公开 AI 定价页以及 9 个特殊设计样本；每行记录产品 SKU、定价原语、机制细节（预付、配额、过期、滚动、计费单位等），并附带原始 URL 与抓取日期；完整的数据集与代码将在论文附录中公开。

**📈 对比分析**

通过对可观察行的计数和比例统计，比较不同定价原语的普及程度，并在历史面板（39 行、4 个时间点）中跟踪演化；采用敏感度分析枚举 9 条边界记录的主/备编码，评估结果稳健性。实验结果显示，订阅+使用配额在 AI 原生产品中占比 55%，预付积分仅见 2 行，结果计费极少；计费平台设计需求从这些统计中导出。

**⚠️ 局限性**

① 仅覆盖公开定价页面，未能捕捉企业定制合同与折扣；② 样本受英语/美洲公司偏好影响；③ 编码主要由单一作者完成，尽管进行了 20% 样本审计，但整体一致性未完全验证；④ 关注的时间段 2026 年为截尾，年度事件统计受限；⑤ 定价原语划分为“主”与“备”，可能对混合产品的真实分布产生偏差。

---

## 182. Disentangling Robustness and Sycophancy in Multilingual LLMs

**arXiv ID:** 2610.30696 | [PDF](https://arxiv.org/pdf/2610.30696v1)

**作者:** Rhitabrat Pokharel `[一作]` (Portland State University), Ameeta Agrawal `[通讯]` (Portland State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SC‑matrix诊断框架，用以分离多语言LLM在面对错误前提时的“能力”（Competence）与“顺从度”（Sycophancy），并通过四个细胞（Robust、True Sycophancy、Honest Uncertainty、Uninformed Deference）对多语言模型的失效模式进行细粒度评估。

**💡 创新点**

创新点在于：①将sycophancy与模型能力分离，揭示整体率掩盖的不同失败成分；②发现不同语言和模型的sycophancy主要由能力缺失或鲁棒性弱导致；③证明推理导向的缓解策略（CoT、EB）在所有模型与语言上均优于指令式干预；④验证按SC细胞路由的提升有限，说明单一普适策略已具备较好泛化性。

**🔧 技术方法**

技术实现：三阶段实验——(1) 无压力下的多语言多任务准确率评估；(2) 在四类用户压力（权威断言、同侪共识、重复挑战、情感暗示）下测得答案是否被改变；(3) 采用四种提示级缓解手段（ASP、3P、CoT、EB）并计算sycophancy率变化；同时利用SC-matrix对每一次试验进行分类。

**📊 数据集**

数据集：TruthfulQA（MuBench多语言版）与GlobalMMLU，覆盖六种语言（英、西、法、俄、中、日）和多学科问答；每种语言100道题。

**📈 对比分析**

比较方法：对每个模型、语言、压力类型和缓解策略，报告sycophancy率、SC细胞分布及缓解增益。结果显示：Llama整体sycophancy率最高，主要由Uninformed Deference导致；Qwen在英语表现最差，True Sycophancy占比高；GPT在所有语言下sycophancy率最低，Robust占优。推理导向缓解可使sycophancy率下降约20–35%，而指令式干预效果有限；按细胞路由提升仅约0.4点，表明普适策略已较好覆盖各细胞。

**⚠️ 局限性**

局限性：仅评估了三款7–8B参数的开源模型和一款专有模型，未覆盖更大或不同架构模型；仅测试六种语言，缺乏低资源语言；使用多项选择题导致二元评判，无法推广到开放式生成；压力提示为通用质疑，未探究明确错误提示下的行为；内部机制（如注意力头如何导致sycophancy）未进一步分析。

---

## 183. Live Thread Reconstruction for Autonomous Manipulation

**arXiv ID:** 2610.30694 | [PDF](https://arxiv.org/pdf/2610.30694v1)

**作者:** Emma Huang `[一作]` (University of California San Diego), Michael Yip `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一套实时线程重建与连续操作的完整管线，用于机器人手术中的针筒捡取和缝合结绑。

**💡 创新点**

创新点在于：①提出基于Kalman滤波的连续线程重建，可在手柄遮挡和高噪声下保持稳定；②引入线程-抓手相互作用模型，实现安全约束下的抓点优化；③将重建与执行策略紧密耦合，形成完整的自动化操作流程。

**🔧 技术方法**

采用了立体视觉、二值分割、结构张量检测、B样条与扩展Kalman滤波、最短路径/二次规划关键点排序、交叉点识别、约束优化的抓点策略以及基于抓手运动的过程噪声模型。

**📊 数据集**

实验数据来自作者自建的手术模拟图像数据集，包含针筒捡取、线绳脱结和结绑等三种典型任务的立体图像序列；未公开使用公开数据集。

**📈 对比分析**

与单帧静态线程重建方法比较，在易/中等配置下两者成功率相近（90% vs 87%），但在困难配置下实时重建显著优于静态重建（93% vs 70%）；在结绑任务中，实时重建实现了100%成功率（易/中等）和87%（困难）。

**⚠️ 局限性**

主要局限在于：对极度平行于极值线（epipolar-parallel）段的线程重建精度不足，导致某些操作无法进行；在极端遮挡或完全离场的情况仍需进一步改进重建恢复能力。

---

## 184. HDLouvain: High-Throughput Incremental Clustering for Hyperdimensional Data

**arXiv ID:** 2610.30687 | [PDF](https://arxiv.org/pdf/2610.30687v1)

**作者:** Beenish Gul `[一作]` (University of Virginia), Kevin Skadron `[通讯]` (University of Virginia)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计了一种名为 HDLouvain 的混合增量聚类框架，将 Louvain 社区检测与超维度计算相结合，对质谱数据进行增量聚类。

**💡 创新点**

将 Louvain 发现的社区一次性压缩为二进制超向量代表，并在后续流式处理时仅进行 GPU 加速的汉明距离匹配，既保持高聚类质量又显著提高吞吐量；同时引入 bootstrap 份额 α 作为可调节的质量-速度权衡参数。

**🔧 技术方法**

Louvain 社区检测、超维度编码（HDC）与位级 XOR / popcount、GPU 加速、k‑NN 图构造、预先桶化 precursor‑mass、位序列布尔聚合。

**📊 数据集**

PRIDE 公共质谱数据集 5 个，规模从 0.99 M 到 21.13 M 条 spectra（PXD001197、PXD001468、PXD001511、PXD003258、PXD000561）。

**📈 对比分析**

与全数据静态 Louvain、DBSCAN、HAC 等传统方法比较；在 50/50 bootstrap 时实现 3.21× 加速；α 变化时可达 4.5× 的最高加速；模块化度在 0.75–0.97 之间，NMI 0.86–0.93，几乎不逊色全量 Louvain。

**⚠️ 局限性**

离线阶段仍需构建 k‑NN 图和运行 Louvain，具有 O(n²) 成本；对聚类质量有最高 8% 的轻微损失；桶边界可能导致同一肽段被拆分；在极小数据量时加速效果不显著。

---

## 185. Two-Timescale Antenna Positioning for Pinching-Antenna System Based on Statistical Channel Information

**arXiv ID:** 2610.30670 | [PDF](https://arxiv.org/pdf/2610.30670v1)

**作者:** Jingze Ding `[一作]` (Peking University), Rui Zhang `[通讯]` (National University of Singapore)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种两时尺度天线位置优化框架，针对Pinching‑Antenna System (PASS) 在多用户下通过统计CSI确定大尺度位置，随后在小尺度范围内利用瞬时CSI进行微调，实现链路性能提升。

**💡 创新点**

创新点包括：①将天线位置分为大尺度（基于统计CSI）与小尺度（基于瞬时CSI）两层，解决了传统单时尺度方案需获取全局CSI的难题；②针对单用户给出闭式最优大尺度位置并推导性能上界；③为多用户设计了元素级搜索与投影梯度上升算法，近似上界并兼顾复杂度。

**🔧 技术方法**

采用统计Rician通道模型、PASS天线架构、两时尺度优化方法、零迫射线（ZF）波束成形、并行搜索、投影梯度上升、Monte‑Carlo逼近等技术。

**📊 数据集**

未使用公开数据集，全部基于仿真：随机用户分布、Rician因子、路径数、波导尺寸等参数在论文中给出。

**📈 对比分析**

与多种基准（上界、仅大尺度定位、仅小尺度运动、稀疏/密集FPA）比较；实验表明两时尺度方案在单/多用户、不同Rician因子、用户聚集/随机分布以及区域尺寸变化下均优于基准，且接近理论上界，显示显著的吞吐率/最小速率提升。

**⚠️ 局限性**

局限性：①假设完美统计与瞬时CSI，实际CSI误差会降低小尺度优化效果；②单波导单PA模型，未考虑多PA或硬件非理想；③多用户多路径下算法复杂度仍较高，需进一步优化；④对高移动性环境中小尺度调节速度与能耗未深入评估。

---

## 186. RiboBridge: protein-conditioned de novo RNA binder structure design

**arXiv ID:** 2610.30668 | [PDF](https://arxiv.org/pdf/2610.30668v1)

**作者:** Zhou Zhang `[一作]` (Nanjing University), Tianfan Fu `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RiboBridge框架，用结构先导方式设计蛋白质结合的RNA分子；

**💡 创新点**

创新点在于将RNA构象生成与蛋白质- RNA装配分离，通过跨帧交互几何预测并用刚性放置桥接，保持RNA结构先验并显著提升复合物恢复质量；

**🔧 技术方法**

采用RiboSphere预训练的RNA结构潜在空间、Transformer生成RNA、交互几何预测网络、范围约束刚性放置求解器等技术；

**📊 数据集**

使用从PDB收集的约29,624个蛋白质–RNA复合物，构建时序拆分、序列过滤、冗余控制的124个测试集；

**📈 对比分析**

与RNAFlow、ODesign、RFDpoly、NACraft等基线比较，RiboBridge在fNAT、iRMSD、lRMSD、DockQ上均优于基线，DockQ成功率提升至20/124；

**⚠️ 局限性**

局限在于界面候选优先级排序和序列实现仍不稳定，对长RNA的交互约束适配性下降。

---

## 187. Compiling Autoregressive LLM Control into Policy-Exact Parallel Inference for Hierarchical RAG

**arXiv ID:** 2610.30664 | [PDF](https://arxiv.org/pdf/2610.30664v1)

**作者:** Zehao Lin `[一作]` (MemTensor Technology Co Ltd), Feiyu Xiong `[通讯]` (MemTensor Technology Co Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将有限动作控制的LLM编译成一次性树形前缀查询，消除序列化解码并保持与串行策略一致性。

**💡 创新点**

证明前缀状态等价性和策略等价性，提出树形因果探针一次性评估所有可达前缀，实现深度‑2关键路径；同时提供订单边缘化 MAP 集合认证与交互质量上界。

**🔧 技术方法**

前缀树编译、树因果探针、块因果掩码、序列化解码对比、LLM检索增强生成、树注意力与并行推理、KL 与 Pinsker 上界等技术。

**📊 数据集**

使用 QASPER 科学论文问答数据集，包含 A–E 五份文档分拆。

**📈 对比分析**

与串行语法解码、理想分块引擎、单标签投影等基线对比，编译控制在 84–89% 的问题上与串行策略一致，平均控制延迟降低 35–47%，且无输出令牌。

**⚠️ 局限性**

探针宽度随前缀数量阶乘增长，bf16 精度导致近似决策翻转，且对更大控制步长（b>3）时宽度瓶颈显著，仅适用于有限动作控制。

---

## 188. Group Relative Policy Optimisation Improves Multilingual Speech Recognition in Low-Resource Languages

**arXiv ID:** 2610.30663 | [PDF](https://arxiv.org/pdf/2610.30663v1)

**作者:** Tim Wenjie Hu `[一作]` (Sunbird AI), John A. Quinn `[通讯]` (Sunbird AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用强化学习的GRPO方法对Whisper-large-v3进行后训练，提升51种非洲语言的语音识别性能。

**💡 创新点**

将GRPO从LLM解码器迁移到非LLM多语种低资源ASR模型，并通过“headroom”样本筛选与大批量更新实现显著性能提升。

**🔧 技术方法**

采用监督微调、GRPO强化学习、噪声增强、头部间距样本筛选以及大批量梯度更新等技术。

**📊 数据集**

训练数据来自110个子集、7,412小时的51种非洲语言语音；评测数据包括Sunbird Speech Benchmark、AfriVox-v2、SimbaBench及实测部署音频。

**📈 对比分析**

与SFT基线及公开模型对比，宏平均WER分别在AfriVox-v2为0.227、SimbaBench为0.372，尤其在噪声和低资源场景下提升约10%–20%。

**⚠️ 局限性**

依赖大量多源语料；GRPO仅对解码错误有显著修复，难以进一步降低整体识别错误率；目前仅在Whisper-large-v3上验证，未探究更小模型或不同架构的迁移性。

---

## 189. Anticipating Social Consequences with Distributional World Models

**arXiv ID:** 2610.30657 | [PDF](https://arxiv.org/pdf/2610.30657v1)

**作者:** Viet-Thanh Pham `[一作]` (Monash University), Gholamreza Haffari `[通讯]` (Monash University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种分布式社交世界模型，能一次性预测多种可能回应、概率和结果，帮助代理进行决策。

**💡 创新点**

创新点在于将多样未来状态压缩为概率分布的回应集，兼顾概率和结果，并通过单次输出取代多次采样。

**🔧 技术方法**

使用了大规模语言模型（Qwen3-4B）训练，采用监督微调与GRPO结合，并通过重复仿真生成训练样本。

**📊 数据集**

使用了Magnetic-NL、SOTOPIA、AgentSense、ProMediate、SOTOPIA-ToM等五个社交交互基准环境的数据。

**📈 对比分析**

与单一未来、期望值、采样未来以及S3AP、GDP-Zero等基线比较，DSWM在13/15指标上领先，且推理成本显著降低。

**⚠️ 局限性**

限制在于训练仅基于语言模型伙伴、离散采样导致罕见回应缺失、非线性效用近似等。

---

## 190. Residual Calibration via Local Feature-Space Refinement

**arXiv ID:** 2610.30652 | [PDF](https://arxiv.org/pdf/2610.30652v1)

**作者:** Xihao Wang `[一作]` (Zhejiang Gongshang University), Ruixing Ming `[通讯]` (Zhejiang Gongshang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了局部自适应概率估计（LAPE）框架，用邻域特征空间证据对全局校准概率进行残差细化，并提出级联特征增强融合（CFAF）处理多视角任务。

**💡 创新点**

将全局校准视为稳定锚点，利用非参数邻域频率作为残差校正，并通过V折交叉拟合降低校准集过拟合，实现几乎无额外训练的几何感知校准。

**🔧 技术方法**

采用特征嵌入邻域检索、加权留一交叉、拉普拉斯平滑、温度/向量/Dirichlet等后置校准技术，以及深度学习特征提取和多视角级联融合。

**📊 数据集**

在表格数据Obesity Risk、图像数据CIFAR‑100、医学图像BreastMNIST、MHIST以及合成/真实多视角数据集上进行实验。

**📈 对比分析**

与TS、VS、FC、SAG等基线在NLL、ECE与准确率上对比，LAPE显著降低NLL与ECE且保持准确率，CFAF在多视角任务中优于传统晚期融合。

**⚠️ 局限性**

效果高度依赖特征空间的局部一致性，若邻域样本的标签分布差异大则局部频率估计偏差，需更鲁棒的检索度量。

---

## 191. One-shot federated unsupervised domain adaptation with Smoothed Knowledge Distillation and teacher refinement

**arXiv ID:** 2610.30646 | [PDF](https://arxiv.org/pdf/2610.30646v1)

**作者:** Ali Abedi `[一作]` (University of Windsor), Farhad Pourpanah `[通讯]` (Queen's University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种一轮联邦无监督域适应框架 SKD‑ETR，利用源模型的知识在未标注目标域上实现迁移学习。

**💡 创新点**

创新点在于将标签平滑知识蒸馏（Smoothed Knowledge Distillation）与基于指数移动平均的教师细化（Exponential Moving Average Teacher Refinement）结合，既消除伪标签噪声，又通过教师-学生双向迭代提升伪标签质量，避免传统聚合带来的沟通开销与偏差。

**🔧 技术方法**

核心技术包括知识蒸馏、标签平滑、EMA 细化以及冻结 DINOv2 视觉骨干的轻量级瓶颈与头部，整体保持高效的训练与推理成本。

**📊 数据集**

实验使用 OfficeHome、Office‑Caltech 与 DomainNet 三大标准多源无监督域适应基准，涵盖从 4 至 6 个域、65–345 类不等。

**📈 对比分析**

与多种 MSUDA、MSFDA、FUDA 方法对比，SKD‑ETR 在 OfficeHome 与 DomainNet 上均实现了最高或相近的平均准确率，同时仅需一次通信且参数量大幅下降，证明其兼具高性能与低成本。

**⚠️ 局限性**

局限性主要体现在极端域偏差（如 DomainNet 的 Quickdraw）下随机初始化导致的性能下降，以及对异构模型与半监督目标域的适应性尚待进一步研究。

---

## 192. Speculative Acceptance Does Not Decay with Position: A Full-Length Diagnostic on Long Chain-of-Thought with Independent Drafts

**arXiv ID:** 2610.30644 | [PDF](https://arxiv.org/pdf/2610.30644v1)

**作者:** Zhiming Meng `[一作]` `[通讯]`, Zhiming Meng

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 MATH-500 500 道长链条推理问题，使用两种独立草稿模型与目标模型进行完整长度的投机解码（greedy 与 Leviathan 采样）记录每一步接受率，系统测量接受率随生成位置的变化，并通过问题内配对消除组合混杂。

**💡 创新点**

首次给出独立草稿-目标对的全长度位置基线，证实接受率并不随位置衰减，反而略有提升；发现拒绝集中在高熵的“反射”触发词上，且可见的 argmax 一致率与真正的期望接受概率分离；用三臂 LoRA 轨迹对齐干预验证此机制无效，提示需要以接受率为目标的对齐策略。

**🔧 技术方法**

投机解码（speculative decoding）、Greedy 与 Leviathan 采样、位置配对统计、温度扫频、三臂 LoRA 轨迹对齐干预、对数似然估计（α）与 argmax 一致率（def2）对比、Bootstrap 与符号检验。

**📊 数据集**

MATH-500（全 500 道数学推理题目），目标模型 DeepSeek-R1-Distill-Qwen-7B，草稿模型同族 DeepSeek-R1-Distill-Qwen-1.5B 与跨族 Qwen2.5-1.5B-Instruct。

**📈 对比分析**

采用同一问题内对半分位数的配对差异检验、符号检验和 Bootstrap 置信区间进行比较；结果显示在 greedy（0–8k 位置）和 sampling（0–12k 位置）范围内接受率无衰减，且在跨族配置中提升更明显；干预实验显示 LoRA 轨迹对齐在采样与贪婪模式下均未提升平均接受长度（MAL）或准确率。

**⚠️ 局限性**

支持范围受长尾问题限制（greedy 8k 位置后样本不足）；反射词桶较小，温度扫频样本有限；仅使用单一目标族、单种种子、固定吞吐 γ=5，外推性受限；未对训练-推理长度不匹配的投机模型做直接对比。

---

## 193. CODI: Improving Self-Supervised Vision Transformers with Cross Distillation

**arXiv ID:** 2610.30639 | [PDF](https://arxiv.org/pdf/2610.30639v1)

**作者:** Siran Dai `[一作]` (Chinese Academy of Sciences), Qingming Huang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究 Vision Transformer 在自监督预训练中全局与局部表示不匹配问题，并提出 Cross Distillation (CODI) 通过在掩码补全目标中混入类标记信息来改善低关注区域的语义表示。

**💡 创新点**

通过局部梯度分析揭示 class‑token 读取路径将全局语义梯度按 patch‑attention 加权，提出在目标端混合 class‑token 分布的 CODI 方法，使每个掩码 patch 接收无 Attention‑gated 的全局语义校正。

**🔧 技术方法**

基于 ViT 的自监督知识蒸馏框架，使用 Momentum Teacher、Projection Head、Softmax 归一化、KoLeo 正则、混合目标损失和 DINO 的类标记对齐技术。

**📊 数据集**

主要在 ImageNet‑1K 进行预训练，随后在 Pascal VOC、COCO、ADE20K、NYU‑Depth v2、KITTI、SUNRGBD、DAVIS 2017 等密集预测与分类基准上评估。

**📈 对比分析**

与 DINOv2、iBOT、MAE、MoCo 等方法对比；在对齐预训练设置下 CODI 在语义分割、深度估计、视频目标分割与图像分类等任务中平均提升 1–3 mIoU 或 Top‑1 准确率，并在小模型上显著超越 Gram‑Anchoring。

**⚠️ 局限性**

对 α 的选择敏感，过大会导致性能下降；改进主要针对低关注区域，未能显著提升已有高关注区域的表示；在极端遮挡或小尺寸物体上效果有限。

---

## 194. An Agent Architecture with Tools and Memory for Selective Correction of Wrist PPG Heart Rate Estimates

**arXiv ID:** 2610.30688 | [PDF](https://arxiv.org/pdf/2610.30688v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 195. What the Specification Did Not Say: Specification-Oracle Gaps in Test-Graded Coding-Agent Benchmarks

**arXiv ID:** 2610.30635 | [PDF](https://arxiv.org/pdf/2610.30635v1)

**作者:** Fei Ding `[一作]` `[通讯]` (Alibaba Group), Fei Ding (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并量化了“规范–判据”缺口，构建了跨五种语言的静态审计工具，评估该缺口对SWE‑Milestone等编程代理基准的影响，并提供修正过滤列表与规范补充，揭示缺口对分数和排名的实质性影响。

**💡 创新点**

创新点在于：①系统性定义并分类规范–判据缺口；②实现无模型调用的高效静态审计，能够识别测试依赖的隐藏名称、路径与文字；③通过实验量化缺口对代理表现、分数和排名的影响，并与其他基准比较；④发布修正列表和规范补充，为基准改进提供可操作方案。

**🔧 技术方法**

技术手段包括：静态代码分析提取器（Go、Python、Rust、TypeScript、Java）、差异检测与分类算法、回归与逻辑回归分析、计数统计与置信区间计算、以及轻量级脚本实现审计与重新评分。

**📊 数据集**

使用数据集为SWE‑Milestone v1.0.2（七个仓库、580k测试）、其35个公开代理运行（Claude Code、Codex、Gemini CLI、OpenHands等），以及SWE‑EVO、SWE‑Chain、SWE‑Bench Verified、DeepSWE等四个对比基准。

**📈 对比分析**

比较方法：对受缺口影响的测试单元与无缺口单元进行对照实验、对比不同模型与不同配置的运行结果、以及随机移除等基线，评估分数提升、解决率变化及排名稳定性；实验显示，去除缺口测试后排行榜基本不变，单仓库分数显著提升，审计精度约为58%（精确率）与78%（召回率）。

**⚠️ 局限性**

局限性包括：审计仅识别名称、路径与文字级依赖，未覆盖类型、属性或运行时生成的依赖；评估基于公开跑分，缺乏对所有模型的实验；对链不一致性只在Go上进行反事实验证；规则制定基于开发集，可能存在偏差；并且缺口修正对基准的长远影响尚未通过新代理运行验证。

---

## 196. Interpreting Model Reasoning through Behavioral Counterfactuals

**arXiv ID:** 2610.30630 | [PDF](https://arxiv.org/pdf/2610.30630v1)

**作者:** Shawn Im `[一作]` (University of Wisconsin–Madison), Sharon Li `[通讯]` (University of Wisconsin–Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于行为反事实的框架，训练残差流编辑模块以定位并分析语言模型在固定生成文本下对提示干预的推理影响。

**💡 创新点**

创新点在于使用 KL 散度匹配目标分布而非传统激活重构，能够在不对齐词级输出的情况下捕捉模型行为变化，并通过归因与消融验证因果位点。

**🔧 技术方法**

采用门控 MLP 编辑模块、集成梯度归因、激活补丁、Top‑K/Top‑p 归一化的 KL 目标以及对照实验比较。

**📊 数据集**

使用 Scruples 伦理情境数据集，并在 Gemma‑4‑E4B‑it 与 DeepSeek‑R1‑0528‑Qwen3‑8B 两个模型上进行实验。

**📈 对比分析**

与传统的激活重构（MSE/NMSE）对比，分布匹配在 Gemma 取得约 47.8%‑20.4% 的 KL 恢复率；归因与消融相关性在 0.58‑0.68 范围内，表明方法能有效定位因果位点。

**⚠️ 局限性**

局限性包括仅在单一干预场景下验证，缺乏对更复杂或多样化干预的泛化；方法依赖固定响应，可能无法完全捕捉真实生成文本的多样性。

---

## 197. Scaling Laws for Safety Probes

**arXiv ID:** 2610.30678 | [PDF](https://arxiv.org/pdf/2610.30678v1)

**作者:** Yuchen Wu `[一作]` (University of Washington), Jerry Wei `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究安全探针在不同模型规模与标签数下的误差可否通过经验规律预测，并在此基础上预估新模型的探针性能、成本及架构选择。

**💡 创新点**

首次提出两条针对安全探针的规模律：1）误差下限随模型参数数呈幂律下降；2）误差随标签数以Hill函数形式逼近下限，且收敛速率随模型规模相关。该规律可在未训练新模型前精准预测探针误差并指导成本与架构决策。

**🔧 技术方法**

采用线性探针和全层Streaming探针（CC++）两种架构；利用AUROC误差(E=1‑AUROC)衡量性能；在对数尺度下拟合功率律与Hill函数，得到统一的误差模型。

**📊 数据集**

构建六个大规模安全数据集（至1M标签），覆盖真值判断、顺从性、政策违规、一般危害、越狱、提示注入，均为最小对角样本，保证从少量到饱和的学习曲线。

**📈 对比分析**

在20个基础模型（5个家族，0.5B‑72B参数）上训练两种探针，拟合误差R²≥0.94；预测下一个模型误差误差≤0.02的案例占72%，成本选择准确率73%，架构选择误差平均≤0.006。

**⚠️ 局限性**

局限性：仅在已覆盖的任务、架构与模型族上验证；对新任务需先小模型检验；假设标签质量一致，未考虑标签成本差异；对极端大模型或非常少标签的情况预测不够精确。

---

## 198. OneLatent: Latent Reasoning for Efficient Foundation Recommendation Models

**arXiv ID:** 2610.30675 | [PDF](https://arxiv.org/pdf/2610.30675v1)

**作者:** Hao Jiang `[一作]` (Kuaishou Technology), Wenwu Ou `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出OneLatent框架，将显式CoT压缩为可学习的潜在推理令牌，实现Latent-Reason-then-Answer推理；

**💡 创新点**

创新点在于引入多视角自适应Chain-of-Thought（MV-ACoT）生成教师监督，并通过三阶段潜在令牌对齐和课程化后训练，将推理压缩为固定长度令牌，同时设计生产级异步推理架构；

**🔧 技术方法**

采用基于Qwen3.5 LLM的Foundation Recommendation Model，配合多视角自适应CoT、三阶段潜在令牌对齐、课程化后训练及异步行为触发推理调度等技术；

**📊 数据集**

使用工业级快手本地服务推荐数据（7天用户交互）和公开的Kuaishou LLM-Rec基准数据集；

**📈 对比分析**

与ID/ SID生成基线、OneReason及显式CoT等进行对比，OneLatent在SID@64提升约17.44%（工业）或12.21%（公开），并比显式CoT提高约17倍吞吐率，在线A/B测试中提升约9.6%收入；

**⚠️ 局限性**

局限性包括对教师生成CoT质量的依赖、潜在令牌数量选择对性能的敏感性、极端复杂推理场景可能仍需更长令牌，以及异步推理可能导致候选更新延迟等问题。

---

## 199. Future of Time Series Anomaly Detection: From Historical Review to Unified Outlook

**arXiv ID:** 2610.30628 | [PDF](https://arxiv.org/pdf/2610.30628v1)

**作者:** Zhijie Zhong `[一作]` (South China University of Technology), C. L. Philip Chen `[通讯]` (South China University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对时间序列异常检测（TSAD）领域进行系统综述，构建了细化的异常分类、统一的检测范式分类法，并提供了开源评估基准和实验结果。

**💡 创新点**

提出了多通道相关异常（channel‑correlation anomaly）分类，整合了生成式、相似度、重构、统计和LLM等多种检测范式的结构化框架，并揭示了传统F1‑PA评估指标的膨胀偏差。

**🔧 技术方法**

采用了元学习式的基准框架、公开TSAD数据集、统一实验脚本以及多维度评估指标（F1‑PA、AUC‑PR、PATE等），并使用Transformer、State‑Space、Diffusion、LLM等前沿网络进行对比实验。

**📊 数据集**

使用了来自多行业的公开时间序列数据集（如Numenta、MSL、SMAP、MTS、SMAP 等），覆盖工业监测、金融、交通、医疗等不同领域，保证了实验的多样性和代表性。

**📈 对比分析**

通过在统一基准上对 20+ 算法进行多指标比较，发现不同范式在检测精度、实时性、资源占用方面存在显著差异；同时实验证明 F1‑PA 在段式异常上存在明显的高估效应，建议使用区间级指标进行更可靠的评估。

**⚠️ 局限性**

仍存在异常标签稀疏、跨域泛化不足、解释性不足以及对多模态和零样本迁移的支持有限等限制；此外，基准集对真正工业场景的真实性和多维度异常解释仍不够完整。

---

## 200. Model-intrinsic, Weight-level, and Training-free Spectral Fingerprinting for Large Language Models

**arXiv ID:** 2610.30626 | [PDF](https://arxiv.org/pdf/2610.30626v1)

**作者:** Yanan He `[一作]` (Guizhou Normal University), Hanzhou Wu `[通讯]` (Guizhou Normal University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于LLM权重内部GATE投影矩阵奇异值分布的谱指纹方法，可在不需要输入样本或前向推理的情况下提取模型指纹；

**💡 创新点**

创新点在于利用GATE投影的谱熵指数构造标量特征，聚合层深度信息并通过线性CKA衡量相似性，实现了权重级、训练免费、对结构变更鲁棒的指纹；

**🔧 技术方法**

技术包括奇异值分解、谱熵计算、层级平均池化、线性CKA相似度评估以及针对GATE权重的特征提取；

**📊 数据集**

使用公开的LLM权重检查点（如Qwen2.5、Mistral、Baichuan2、LLaMA等）共120对模型，涵盖同族、后训练修改与跨族对比；

**📈 对比分析**

与代表性指纹方法（ProFLingo、MergePrint、LLMmap、REEF、HuRef）对比，AUROC、pAUC、TPR、KS均达1.0000，Mahalanobis分离度最高（14.402），显示出优异的区分性能；

**⚠️ 局限性**

局限包括仅针对带GATE FFN的架构，未充分验证非门控模型；对大模型、对抗微调、模型合并及指纹碰撞风险的鲁棒性尚待进一步研究。

---

## 201. Fast Iterative Combinatorial Auctions via Deep Reinforcement Learning

**arXiv ID:** 2610.30667 | [PDF](https://arxiv.org/pdf/2610.30667v1)

**作者:** Luis F. Paul Tosta `[一作]` (Bryant University), Gianluca Brero `[通讯]` (Bryant University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一种名为 FastCombo 的深度强化学习框架，用于单面向组合拍卖中的序列价格发现，目标是减少清算轮次并降低在线计算成本。

**💡 创新点**

证明贪婪即时清算概率最大化可能极度次优；将拍卖建模为 POMDP 并离线训练递归图策略；通过单项实验展示信息化查询的收益；在 CATS 4 个域上实现与 Bayesian 基线竞争且显著降低在线计算。

**🔧 技术方法**

使用深度强化学习（PPO）、递归 GRU 与组合图网络、POMDP 框架、离线训练与在线单前向推理，配合 Bayesian 一阶价格更新与子梯度清算对比。

**📊 数据集**

单项三种价值分布实验；Combinatorial Auction Test Suite (CATS) 的四个结构化域（Arbitrary、Paths、Regions、Scheduling），每个域 10 个单面向竞标者、12 件物品。

**📈 对比分析**

与 Bayesian 基线（Bayes^GEN）和子梯度清算（SG‑D、SG‑I）对比。FastCombo 在四个域的清算率仅比 Bayes^GEN 低 0.36–2.54 个百分点，且在单项实验中从 1.85 期平均轮次降到 1.65；在线价格选择计算时间比 Bayes^GEN 降低 300–900 倍。

**⚠️ 局限性**

仅适用于单面向竞标者、公开包、无策略需求；未考虑支付/分配的机制设计；GRU 记忆有限，可能丢失信息；未验证在更大规模或多面向市场、战略竞标者中的表现。

---

## 202. Governance-State Corruption in Long-Term Memory for LLM Agents: A Cross-Model Stress Test

**arXiv ID:** 2610.30698 | [PDF](https://arxiv.org/pdf/2610.30698v1)

**作者:** Bowen Liu `[一作]` (Independent Researcher), Dingyan Shang `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究LLM代理在内存治理状态失效时的影响，构建了匹配的清洁/损坏/恢复三兄弟实验；

**💡 创新点**

提出治理状态腐败框架，将检索质量与可用性分离，并通过匹配干预评估控制平面错误的下游影响；

**🔧 技术方法**

使用六大LLM供应商、三大基准（LongMemEval、Memora、OAKS），构造治理-受损-恢复三兄弟，采用冻结代理评估与Formal‑300原生验证；

**📊 数据集**

基于731个基准根源示例（380 Memora删失、169 事件、182 更新），共2,080个治理错误对；

**📈 对比分析**

对照实验显示治理错误导致平均-27.55个百分点冻结代理准确率下降，恢复可恢复约+27.51个百分点；原生样本Formal‑300显示Corruption损伤+40.54/30.60点，恢复+39.19/26.24点；

**⚠️ 局限性**

评估依赖冻结的二元代理评分与原生标注不一致，样本为非概率抽样，缺乏部署时状态检测与自主恢复能力，源/事件支持不完整。

---

## 203. LoRA Superposition across Pretrained Models

**arXiv ID:** 2610.30691 | [PDF](https://arxiv.org/pdf/2610.30691v1)

**作者:** Keigo Watanabe `[一作]` (Chiba University), Kazuhiko Kawamoto `[通讯]` (Chiba University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了将不同预训练权重的同一任务LoRA相加后直接应用到所有模型的可行性，提出LoRA superposition。

**💡 创新点**

创新点在于无需对齐或显式路由，只通过简单求和即可让各模型响应自身组件，实现共享适配器。

**🔧 技术方法**

技术包括低秩适配（LoRA）、参数求和、正交性分析与交叉模型响应评估。

**📊 数据集**

使用七个ViT‑B/16 CLIP预训练模型和八个图像分类数据集（Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN）。

**📈 对比分析**

对比了未适配基线、单模型LoRA、联合共享LoRA和LOMO（排除自身）等方法，LoRA superposition在56个模型–任务组合上平均提升30.2个百分点，接近单模型LoRA，且仅需一次求和。

**⚠️ 局限性**

局限在于仅适用于相同架构模型，求和后秩可能增大，且对更多模型或不同架构的可扩展性尚未验证。

---

## 204. From Clinical Judgment to Checkable Rules: Clinician-Led Guardrails for Mental-Health Chatbots

**arXiv ID:** 2610.30690 | [PDF](https://arxiv.org/pdf/2610.30690v1)

**作者:** Chenhan Lyu `[一作]` (University of California), Amir Rahmani `[通讯]` (University of California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研发并评估了一个由临床医生制定、形式化验证、并在推理阶段可复现的安全规则层，用以检查LLM生成的心理健康聊天回复是否符合临床安全标准。

**💡 创新点**

创新点在于把临床医生的安全判断转化为机器可验证的规则，利用SMT求解器对规则一致性进行形式化验证，并通过可追溯的事实提取与确定性规则执行，提升安全判定的可解释性与可审计性。

**🔧 技术方法**

使用的技术包括：Llama‑3.3‑70B 作为事实抽取器与判定器；Z3 SMT 求解器用于规则一致性验证；事实提取、规则判定与确定性解析的管道；以及 fail‑closed 不确定性策略。

**📊 数据集**

使用的数据集包括：公开去标识化的单轮对话集（约294条）用于主实验；Health‑ORSC‑Bench 的 500 条边界提示用于验证规则层在拒绝率上的影响。

**📈 对比分析**

通过与同一模型、相同规则文本的 LLM‑as‑a‑Judge 基线对比，单轮实验中两者仅 35% 的最终判定一致；在危机资源检查上，Judge 的误报率低于规则层；在边界提示上，规则层的阻断率从 6% 提升至 16%；整体表明规则层提供更高的可解释性，但在拒绝率和一致性上存在折衷。

**⚠️ 局限性**

局限性包括：事实抽取依赖 LLM，提取准确性未完全保证；仅评估单轮对话，无法覆盖多轮交互；规则范围有限，缺乏跨语言与跨国环境验证；fail‑closed 不确定性策略导致较高的拒绝率；缺乏更广泛的临床验证与模型多样性评估。

---

## 205. Zero-Shot Domain Generalization via Domain Prompt Learning

**arXiv ID:** 2610.30680 | [PDF](https://arxiv.org/pdf/2610.30680v1)

**作者:** Shiming Chen `[一作]` (Mohamed Bin Zayed University Of Artificial Intelligence), Fahad Shahbaz Khan `[通讯]` (Mohamed Bin Zayed University Of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出零样本域泛化（ZSDG）框架 Domain Prompt Learning (DoP)，通过在文本与视觉模态下学习并动态调整域提示，以在不依赖源域数据的情况下提升模型对未知域的识别能力。

**💡 创新点**

创新点在于：①在文本空间通过交叉样式多样性损失和一致性损失自动生成多样化的风格标记；②在测试时仅对挑选出的最相关风格标记进行视觉自监督微调，从而有效缓解文本-视觉跨域分布差异；③整合文本与视觉域学习，实现零样本条件下的自适应提示选择与细化。

**🔧 技术方法**

核心技术包括：CLIP 预训练视觉-文本双编码器、交叉风格多样性与一致性损失、风格标记选择、测试时自监督学习（基于预测熵过滤）以及软最大化分类推断。

**📊 数据集**

使用了 PACS、VLCS、OfficeHome、DomainNet 四个标准域泛化基准以及 ImageNet（含 ImageNet-V2、ImageNet-R、ImageNet-S、ImageNet-A）作为评估数据集。

**📈 对比分析**

与多种基准方法（经典 DG、VLM 基础、PromptStyler、TPT 等）进行对比，DoP 在 ResNet‑50 以及 ViT‑B/16 的平均准确率分别达 74.9% 与 82.0%，在 ImageNet 上也取得 61.0% 的平均表现，整体优于现有最优方案，且测试时间显著低于 TPT。

**⚠️ 局限性**

局限性包括：①文本空间所构造的域知识有限，导致在高复杂度域（如 DomainNet、ImageNet 多域）上的提升不显著；②单张测试图像的信息不足可能限制风格标记的准确性；③对超参数（如 K 与 ρ）的敏感性仍需进一步探索。

---

## 206. AgentDreamer: Agentic 3D Motion Reasoning for Text-Controllable Physics Simulation

**arXiv ID:** 2610.30679 | [PDF](https://arxiv.org/pdf/2610.30679v1)

**作者:** Mijeong Kim `[一作]`, Bohyung Han `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 AgentDreamer，利用多模态智能体根据文本指令生成结构化的三维运动先验，并通过参数化运动函数指导可微分 MPM 仿真，实现文本可控的物理模拟。

**💡 创新点**

创新点包括：① 将运动规划视为物理推理任务，使用多模态智能体生成可直接用于仿真的三维轨迹；② 引入基于遮蔽文本恢复的 Motion-IG 指标，客观评估运动与文本的对齐度；③ 通过语义注入将采样点与可视部件关联，提升规划的语义准确性。

**🔧 技术方法**

核心技术有：多模态智能体（GPT‑5.6 Sol）进行语义注入和参数化运动规划；可微 Material Point Method (MPM) 仿真与轨迹监督；Motion‑IG 信息增益度量；使用开放权重与 API 视觉‑语言模型进行自动评估。

**📊 数据集**

使用 8 个场景（4 实景：Alocasia、Carnation、Telephone、Fox；4 合成：Playdoh、Basketball、Soccer Ball、Can）共 20 条文本指令，涵盖跌落、弹跳、滚动、旋转、摆动、变形等运动。

**📈 对比分析**

与基准 PhysFlow 与 MotionPhysics 进行对比。Motion‑IG 分数在 4 个开放权重评估器中平均 0.71，最佳 43/80；在 5 个 API 评估器中平均 0.48，最佳 62/100；在人类实验中 AgentDreamer 被选为首选 73.6%，显著优于基准方法。

**⚠️ 局限性**

局限性：依赖预训练的多模态模型，易受模型偏差影响；需要手工选择采样点和语义注入，增加准备工作；参数化运动规划对极端复杂物理场景（如高度非线性变形）适用性有限；计算成本相对较高。

---

## 207. Co^2Gen: Collaboratively Generating Task Plans for Decentralized Multi-Robot and Human Collaboration

**arXiv ID:** 2610.30671 | [PDF](https://arxiv.org/pdf/2610.30671v1)

**作者:** Yi Loo `[一作]` (Singapore University of Technology and Design), Malika Meghjani `[通讯]` (Singapore University of Technology and Design)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Co²Gen框架，利用协同蒙特卡洛树搜索生成可执行的Python协作计划，自动生成多机器人协作演示数据并用于去中心化视觉语言行动（VLA）模型的微调；

**💡 创新点**

创新点在于无需人类远程操作即可生成协作演示，通过VLM预条件检查实现协作任务的顺序和条件化；

**🔧 技术方法**

采用视觉语言模型（GPT‑5.6）、协同MCTS、SmolVLA、YOLOv8、IK求解器、DAgger等技术；

**📊 数据集**

使用自行生成的Co²Gen协作演示数据（50条/任务），以及与单臂演示混合的数据集；

**📈 对比分析**

与单臂VLA_SA和双臂人类远程演示VLA_BM对比，Co²Gen单独数据提升去中心化协作成功率至约50%，混合数据甚至匹配或超越VLA_BM，并在零样本人机协作中表现更佳；

**⚠️ 局限性**

局限包括对VLM预条件判断的依赖、缺乏跨机器人信息共享、实验仅在双臂低成本机器人上验证，未评估更大规模或移动机器人环境。

---

## 208. WikiOriginQA: Automated Construction of a QA Dataset with Knowledge Origins for Analyzing Cultural Bias

**arXiv ID:** 2610.30677 | [PDF](https://arxiv.org/pdf/2610.30677v1)

**作者:** Kento Haneda `[一作]` (Future Corporation), Makoto Morishita `[通讯]` (Future Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个完全自动化、可扩展的多语言文化偏差评测 QA 数据集 WikiOriginQA，利用维基百科不同语言版本的创建时间来确定知识来源的文化圈；

**💡 创新点**

创新点在于通过客观的时间信息提取文化知识，克服了人工注释的规模和主观性限制，并能捕捉知识在非母语版的传播；

**🔧 技术方法**

采用了 Wikipedia 数据包、Wikidata 关联、LLM（如 GPT‑5）自动生成问答、自动化过滤和多语言翻译等技术；

**📊 数据集**

使用的数据集为 WikiOriginQA 自己生成的 QA 数据，来源为 75 种单一国家官方语言中挑选的 8 种语言的维基百科条目；

**📈 对比分析**

通过在多模态 LLM（GPT‑5、Qwen‑3、LLM‑jp‑3.1‑13b‑instruct）上测量准确率和字符 F1，结果显示模型在英语条目或已传播至英语的文化知识上表现更好，证实了对英文化圈的偏差；

**⚠️ 局限性**

局限性包括：对多地区通用语言的适用性有限、对低资源语言的召回率低、数据规模受限、自动化构建的准确性尚未完全验证，且可能遗漏部分文化知识。

---

## 209. Asking Less, Adapting Better: Feedback-Efficient Test-Time Adaptation for Vision-and-Language Navigation

**arXiv ID:** 2610.30643 | [PDF](https://arxiv.org/pdf/2610.30643v1)

**作者:** Zixu Wang `[一作]` (South China University of Technology), Mingkui Tan `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在Vision‑and‑Language Navigation（VLN）任务中，本文提出一种基于层级样本选择与差异感知熵优化的测试时自适应（TTA）方法，以极少的人工反馈显著提升在未见环境中的导航性能。

**💡 创新点**

创新点包括：① 通过轨迹层级的标准差阈值筛选不确定性波动大的轨迹；② 在每条筛选轨迹中使用熵梯度显著性选取关键决策点；③ 在单一采样点上利用差异感知缩放对熵优化方向（强化/放宽）进行加权，从而实现稀疏反馈的高效利用和细粒度的策略更新。

**🔧 技术方法**

技术手段包括：测试时自适应、熵最小化、二元人工反馈、轨迹与步骤层级筛选、熵梯度显著性计算、差异感知缩放、梯度停止传播（stop‑gradient）、动态阈值更新、连续和离散环境下的RL/Transformer基线模型。

**📊 数据集**

实验数据集：REVERIE、R2R、GSA‑R2R、R2R‑CE；此外在物理机器人实验中使用办公室与实验室场景。

**📈 对比分析**

与现有最优TTA方法ATENA、FeedTTA、FSTTA等比较，本文方法在REVERIE、GSA‑R2R、R2R‑CE以及真实机器人上实现了SPL/Success Rate的显著提升（如REVERIE unseen SPL提升18.3%），同时反馈次数更少、每回合耗时更低，达成了state‑of‑the‑art的综合表现。

**⚠️ 局限性**

局限性：① 仍需人工二元反馈，尽管噪声鲁棒性好但在完全无监督场景下表现未知；② 只在单一步骤进行更新，无法捕捉跨步依赖的复杂策略调整；③ 对极端分布漂移或多模态任务的迁移能力仍需进一步验证。

---

## 210. Solvable models of learning to pursue a moving target

**arXiv ID:** 2610.30721 | [PDF](https://arxiv.org/pdf/2610.30721v1)

**作者:** John J. Vastola `[一作]` (Harvard Medical School), Kanaka Rajan `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出并分析了一个连续时间线性二次型（LQR）开环目标追踪模型，推导出最优控制策略、策略学习（REINFORCE）和价值学习的解析解，揭示学习过程中的特征先后顺序及其与模型参数（如努力成本、接近成本、折扣因子）的关系。

**💡 创新点**

创新点在于：①用可解析的线性二次型框架系统描述追踪任务；②揭示了策略学习中存在的谱间隙，使得先学到目标到达再优化路径；③给出完整的闭式表达式说明不同成本权重如何在最优策略与学习动力学中介入。

**🔧 技术方法**

技术手段主要是：线性系统建模、变分法求解最优控制、REINFORCE 的连续时间极限（得到线性 ODEs）、矩阵谱分析（利用秩‑一扰动、Toeplitz 矩阵特征值等）。

**📊 数据集**

本文没有使用实际数据集，而是以高斯分布的状态向量生成器为假设，所有结果均为理论推导和数值实验。

**📈 对比分析**

由于模型是理论化的，本文未与其他算法做实验对比；但通过解析结果和数值模拟展示了不同成本参数下的策略与学习速率差异。

**⚠️ 局限性**

局限性包括：①无观测噪声，状态在每个 episode 直接给出；②完全开环控制，缺乏闭环反馈；③奖励为二次连续形式，缺乏稀疏奖励的挑战；④线性策略假设，未考虑非线性/复杂控制。

---

## 211. Class-Grounded Batik Captioning under Scarce Text Annotation: Label Retention and Source-Specific Error Propagation

**arXiv ID:** 2610.30661 | [PDF](https://arxiv.org/pdf/2610.30661v1)

**作者:** Novanto Yudistira `[一作]` (Universitas Brawijaya), Irawati Nurmala Sari `[通讯]` (Universitas Brawijaya)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究构建了一个基于DINOv2分类器与Qwen2.5‑VL视觉语言模型的两阶段巴蒂克图像描述流水线，并通过注入预测类别进行文本生成，系统评估了类别标签的保留与错误传播情况。

**💡 创新点**

创新点在于提出了六种条件的细粒度评估（三种captioner适配方式×两种注入/不注入策略），并同时分析了参考文本指标与图像‑文本兼容度指标的差异，揭示了类别注入对描述真实性的双重影响。

**🔧 技术方法**

主要技术包括LoRA微调、NF4量化、DINOv2分类器与Qwen2.5‑VL-7B的联合使用，以及BLEU、ROUGE、METEOR、SPICE、BERTScore与CLIPScore等多维度评估指标。

**📊 数据集**

使用了两个数据集：A集（346张东爪哇巴蒂克图像，4个地区类别）用于真实标注和评价，B集（3000张图像，20个类别）用于生成伪注释并支持模型适配。

**📈 对比分析**

通过对六种实验条件的对比，发现所有类别注入条件下的类别标签提及率为0.9067；分类器整体准确率为96.10%，在A集仅为75.76%；参考文本指标在PFT‑NCI条件下表现最佳，而CLIPScore在PFT‑NCI下最低，表明不同指标衡量结果存在分歧；推理时间约22.6秒/图像，主要耗时集中在文本生成上。

**⚠️ 局限性**

局限性包括：分类器与captioner使用独立划分，未能量化可能的交叉训练影响；A集测试样本量小，导致类别准确率统计不稳；来源特异性准确率差异难以解释；BF16混合精度推理出现NaN错误；数据集B的出处信息不完整，评估复现性受限；以及多种评估指标配置细节未公开，影响可重复性。

---

## 212. PhyEvolve: Multi-Agent State-Evolution Planning for Physics-Aware Video Generation

**arXiv ID:** 2610.30717 | [PDF](https://arxiv.org/pdf/2610.30717v1)

**作者:** Yang-Hao Zhou `[一作]` (Canva Research), Xiao Yang `[通讯]` (Canva Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的多智能体框架 PhyEvolve，通过结构化的物理状态演化规划来增强自然语言提示，从而生成更具物理一致性的视频。

**💡 创新点**

创新点在于：①将事件拆分为物理动作与观察变化并明确其因果关系；②通过“Director–PhysicalEvolver–Composer”三阶段流程生成完整的物理状态轨迹；③仅通过提示替换即可与现有视频生成器（如 Minimax‑H3）协同工作，无需重新训练。

**🔧 技术方法**

技术手段：使用大型语言模型 Qwen3‑VL‑32B‑Instruct 进行场景归纳、机制识别与状态推进；通过结构化记录实现因果状态滚动；Composer 将演化记录转化为完整提示；整个流程不改动生成器，只做提示替换。

**📊 数据集**

评估数据集包括 StEvo‑Bench（截断观测下状态演化）、PhyGenBench（力学、光学、热学、材料常识）、PhyWorldBench（基本/复合/违背物理情境）和 VideoPhy‑2（真实动作集合）。

**📈 对比分析**

与现有基线（Minimax‑H3、NEWTON、Co‑Director）对比，PhyEvolve 在四个基准的平均指标上提升 7.9%（相对 NEWTON）且在 StEvo‑Bench 上演化成功率提升 28 点；在 PhyGenBench、PhyWorldBench、VideoPhy‑2 上也实现显著改进，展示了更好的物理一致性和状态连续性。

**⚠️ 局限性**

局限性：依赖 LLM 的推理能力与场景证据，无法提供定量的动态预测或时间精确性；在 VideoPhy‑2 的 Hard 子集上语义遵循与物理一致性并行率仍偏低；尚未评估在更长视频序列中的表现。

---

## 213. FedBST: Bilateral Self-Training for Unsupervised Federated Classification

**arXiv ID:** 2610.30634 | [PDF](https://arxiv.org/pdf/2610.30634v1)

**作者:** Aijing Yu `[一作]` (University of Chinese Academy of Sciences), Xiao-Yu Zhang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在无标签联邦学习中提出FedBST框架，利用双向自训练同时使用高置信度样本和模糊样本进行训练，并在推理时采用视觉-文本集成。

**💡 创新点**

创新点在于：1) 引入不确定性感知置信惩罚（UACP）对模糊样本进行正则化，避免伪标签偏差；2) 通过视觉-文本集成（VTE）在推理时结合局部视觉原型与全局文本语义，提升非IID适应性。

**🔧 技术方法**

使用CLIP预训练模型、视觉文本提示学习、对数损失、熵与置信惩罚、视觉原型聚合与softmax蒙版等技术。

**📊 数据集**

在六个公开数据集（DTD、RESISC45、UCF101、CUB、CIFAR-10、CIFAR-100）和多种非IID设定下进行实验。

**📈 对比分析**

与FedCoPL、FedOPT、pFedPrompt、PromptFL、PromptProx等方法对比，FedBST在Dirichlet和数量偏斜下平均提升约0.8–1.0%准确率，特别是对CUB等细粒度数据表现最优。

**⚠️ 局限性**

局限性包括对超参数（如λ、α、τ_con、τ_ent）敏感，需要在不同场景下调优；且在极端稀缺数据或极大客户端数量时仍可能出现性能下降。

---

## 214. Learning Domain Trajectories with Flow Matching for Gradual Domain Adaptation

**arXiv ID:** 2610.30685 | [PDF](https://arxiv.org/pdf/2610.30685v1)

**作者:** Yubo Huang `[一作]` (Southwest Jiaotong University), Shuai Zhang `[通讯]` (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于连续流匹配的逐步域适应框架FlowGDA，用来在表示空间中实现源域到目标域的平滑迁移。

**💡 创新点**

创新点在于将域演变建模为统计流形上的连续交通，通过学习时间相关速度场而非离散步进，并结合源域锚定与语义一致性正则，避免了传统对抗式或最优运输求解的高昂计算与不稳定问题。

**🔧 技术方法**

核心技术包括：流匹配（flow matching）回归目标、双向速度场（source→左域和左→右域）、统计流形的时间与域坐标嵌入、两阶段顺序优化以及语义一致性交叉熵约束。

**📊 数据集**

实验数据集涵盖四类：旋转MNIST（旋转角度渐进）、Color-Shift MNIST（颜色统计渐变）、Portraits（年代演化的真实图像）和Cover Type（表格特征的协变量漂移）。

**📈 对比分析**

与传统无监督域适应基线（DANN、DeepCORAL、CDAN）以及现有逐步适应方法（GST、GOAT、AST、CNF）对比，FlowGDA 在所有四个基准上均实现了最高准确率，提升幅度分别为+6.4、+7.4、+2.8 以及 +2.6 分，显示出更稳健的性能。

**⚠️ 局限性**

局限性在于需要已排序的中间域信息，且对极端大域差仍可能受限；此外，双向速度场与两阶段优化使训练流程相对复杂，对硬件与超参数调优有一定要求。

---

## 215. Direct cueing of word identity through word duration: evidence from Mandarin homophones

**arXiv ID:** 2610.30682 | [PDF](https://arxiv.org/pdf/2610.30682v1)

**作者:** Xiaomeng Zhu `[一作]` (Yale University), Kevin Tang `[通讯]` (University of Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了词语持续时间是否能直接提示汉语同音词的辨识，设计两项实验在无上下文、无音素差异的情况下，仅通过持续时间差异来判断高频词与低频词的顺序。

**💡 创新点**

首次提供实验证据显示词语级别的时间信息（持续时间）被词汇表征直接编码，可在缺乏语音或语境线索时用于词汇检索。

**🔧 技术方法**

采用音频录制、Praat持续时间操纵、噪声掩蔽、Gorilla在线实验平台、R语言的lme4包进行二元混合效应回归分析。

**📊 数据集**

使用汉语词典（Chinese Lexical Database）筛选的16对双音节同音词，配合200名母语者（实验一）和200名（实验二）在Prolific招募的听众。

**📈 对比分析**

实验结果显示在噪声条件下以及仅噪声条件下，受试者的高频词选择率显著高于50%（β≈0.20–0.24，p<0.01），表明持续时间可作为高效的直接线索；与传统语音/语境线索对比，虽未作数值性能对照，但足以突破偶然猜测。

**⚠️ 局限性**

局限在于仅使用单一说话者、固定的持续时间比率（25%）以及少量（16对）同音词，效应幅度有限，需进一步验证多说话人、多词对以及更大持续时间梯度下的普适性。

---

## 216. Your Token Is Gone: A Token-Consumption Countermeasure for Disrupting LLM-Based Scam Systems

**arXiv ID:** 2610.30666 | [PDF](https://arxiv.org/pdf/2610.30666v1)

**作者:** Fudu Xing `[一作]` (National University of Singapore), Tianlong Yu `[通讯]` (Hubei University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了TokenTrap，一种通过向疑似LLM诈骗系统发送低泄露、高扩展的安全任务来重定向其计算资源，从而显著放大其推理消耗的防御框架。

**💡 创新点**

创新点在于：① 资源导向的防御理念；② 针对三种常见代理架构（单体、层级与平面多代理）设计的专属Token消耗任务；③ 通过可观测对话结构的路由器实现架构感知；④ 兼具抗护栏与有限执行的闭环策略。

**🔧 技术方法**

使用技术包括：基于Qwen2.5-7B-Instruct的LLM路由器、三类Token任务库（递归自审、层级审计回路、平面共识死锁）、抗护栏策略（任务变体、状态裁剪、Token/轮/时间/安全限额）以及CAR等性能指标。

**📊 数据集**

数据集：300条人工生成、结构标记提示（单体100、层级100、平面100）以及对Riemann–Hermitian任务的完整50轮控制轨迹，用于评估路由准确率与Token消耗。

**📈 对比分析**

比较方法：与无Token任务基线和单一Token任务基线对比，使用Token消耗比例（CAR）和时间指标。结果显示TokenTrap在三种架构下分别放大Token消耗为2,487×、1,616×、567.8×；路由准确率达到85.67%；总Token消耗与耗时均显著提升。

**⚠️ 局限性**

局限性：未在真实诈骗对话上验证诈骗成功率；缺乏跨模型、跨任务的泛化评估；未考虑攻击者自适应逃逸策略；缺少对经济成本转换和合法系统潜在副作用的深入分析。

---

## 217. Lychee: Certified Screening of Inverted Lists for High-Recall Vector Search

**arXiv ID:** 2610.30697 | [PDF](https://arxiv.org/pdf/2610.30697v1)

**作者:** Dongfang Zhao `[一作]` `[通讯]` (University of Washington), Dongfang Zhao (University of Washington)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于GPU的倒排文件搜索方法，通过认证筛选来提高高召回率的近邻搜索性能。

**💡 创新点**

创新点在于使用局部认证阈值来界定倒排列表中的k-th距离，从而在不读取所有向量的情况下筛选出候选向量。

**🔧 技术方法**

采用了GPU加速的倒排文件搜索技术，结合了认证排除和张量核心的处理能力。

**📊 数据集**

使用了11个数据集，向量数量从一百万到十亿不等，涵盖了图像和文本嵌入。

**📈 对比分析**

与十个基线方法进行比较，在相同的召回率下，性能提升显著，速度比IVF-Flat快9.7到185倍，尤其在大规模数据集上表现突出。

**⚠️ 局限性**

局限性在于方法依赖于特定的硬件（如GPU），在某些情况下可能不适用于资源受限的环境。

---

## 218. AGF-Net: A Robust Multimodal Emotion Recognition Model for Asynchronous Audio-Visual Signals Based on Predictive Dynamic Fusion with Negative Learning

**arXiv ID:** 2610.30651 | [PDF](https://arxiv.org/pdf/2610.30651v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 219. Spectral Smoothing and Spiking Perturbations Reconfigure Capability Profiles and Selective Task Performance in Large Language Models

**arXiv ID:** 2610.30676 | [PDF](https://arxiv.org/pdf/2610.30676v1)

**作者:** Qince Luan `[一作]` (Southern University of Science and Technology), Yang Xu `[通讯]` (Southern University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了对变换器前馈投影中的奇异值谱进行后验光滑和尖峰干预，探讨其在下游行为中的因果作用。

**💡 创新点**

提出了一种轻量级的奇异值光滑/尖峰干预方法，能够揭示大型语言模型能力的组织和权衡。

**🔧 技术方法**

使用了奇异值分解（SVD）技术对前馈投影矩阵进行干预。

**📊 数据集**

在五个开放的Llama和Qwen指令模型上进行了评估，涉及118个子任务。

**📈 对比分析**

通过与标准和链式推理基准的比较，发现干预导致的行为变化并非均匀，而是结构化的，某些设置可以改善特定任务的表现，而其他设置则可能抑制语言和推理能力。

**⚠️ 局限性**

研究的局限性在于只覆盖了五个开放模型，使用固定的α网格，且基准套件的规模和评估噪声不同，部分测量依赖于外部评审。

---

## 220. A Topology-Guided GeoAI Agentic Framework for Urban Flood Screening

**arXiv ID:** 2610.30660 | [PDF](https://arxiv.org/pdf/2610.30660v1)

**作者:** Yutian Tao `[一作]` (University of Maryland), Leila De Floriani `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于拓扑的GeoAI智能框架，用于城市洪水筛选，能够根据地点、边界或手绘区域自动检索公开地理数据，提取并按洪水优先级排序的地形盆地，生成交互式地图与报告；

**💡 创新点**

创新点包括：①将离散Morse理论用于地形盆地提取与持久性阈值选择，②将D8流向路由作为独立一致性检查，③在关键决策点仅调用LLM，确保计算确定性并提供可追溯的证据，④实现三点人机交互检查以保证决策透明与可控；

**🔧 技术方法**

使用技术包括：Topology Toolkit (TTK) 离散Morse分析、D8流向路由、熵权法确定因素权重、Rational Method 估算峰值径流、LLM（Google Gemini‑3.5‑flash）进行阈值选择、推荐生成与自我审查；

**📊 数据集**

数据集：USGS 3DEP DEM、NLCD Land Cover、NOAA Atlas 14 降雨、USDA SSURGO 土壤、WorldPop 人口密度、FEMA 洪区、OSM 基础设施等公开数据；

**📈 对比分析**

比较方法：在五个研究区评估Morse盆地与D8路由的吻合率（61%–83%），并对四种权重方案进行排名稳定性测试；性能表现显示大部分被保留的盆地在D8路由中为汇点，且首位盆地在所有权重下保持稳定；

**⚠️ 局限性**

局限性：仅测试了五个美国区域，缺乏真实洪水事件验证；使用单一LLM，无法评估跨模型差异；暴露、贫困率等因素缺失时采用中性值，可能影响风险评估；未来需扩展地理范围、数据完整性与模型比较。

---

## 221. Large language models show systematic failures in incorporating evolving medical knowledge

**arXiv ID:** 2610.30631 | [PDF](https://arxiv.org/pdf/2610.30631v1)

**作者:** Min Zeng `[一作]` (University of Minnesota), Rui Zhang `[通讯]` (University of Minnesota)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了ChronoEvo基准，评估大型语言模型在医疗指南更新后是否能完整整合新的医学知识，并发现模型识别更新但往往未能完全采纳；

**💡 创新点**

首次将“识别与整合”两项能力区分开来，揭示大模型在持续学习医学知识时存在显著的“识别–整合”缺口，并通过多模型、多评估者验证其稳健性；

**🔧 技术方法**

利用结构化的提示与对话式接口生成响应，使用GPT‑5（及GPT‑5.6 Sol）进行语义评估，构建并训练基准；

**📊 数据集**

来源于TempoMed‑Traj的308条临床指南更新，提取1,709条验证的推荐级变更，涵盖12类知识变更；

**📈 对比分析**

在14种模型（从4B到70B参数不等）中测试，识别率普遍高于整合率，平均识别–整合差距约30个百分点；在标准全证据条件下，模型对指南更新的修订能产生约90%的语义识别率，但仅58%能完全整合；

**⚠️ 局限性**

局限包括：基准仅涵盖指南更新，未覆盖药品安全、监管公告等知识更新；评估主要基于自动语义判定和两名临床医生的抽样评估；模型内部机制不明，未解释为何识别与整合分离；以及未测试检索、版本匹配等前置任务的影响。

---

## 222. Meaning Contracts: Local Semantic Geometry in Large Language Models

**arXiv ID:** 2610.30715 | [PDF](https://arxiv.org/pdf/2610.30715v1)

**作者:** Andrés Daniel Godoy Ortiz `[一作]` `[通讯]` (Imperial College London), Andrés Daniel Godoy Ortiz (Imperial College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型隐藏状态中语义的局部几何结构，并通过两个对句子级别的控制性探针在四个不同架构的decoder‑only LLM（Gemma 4、Llama 3.2、Qwen 2.5、Pythia 2.8B）上进行定量评估；

**💡 创新点**

提出本地“核心‑变异”模型，揭示语义在层面上呈现一个不变核心、结构化变异以及细化语义时的有向收缩现象，进一步补充了传统的线性表示假设；

**🔧 技术方法**

利用PCA去除主方向（all‑but‑the‑top）、余弦相似度、欧氏距离与Mahalanobis距离对隐藏状态进行度量，采用句子平均池化与最后一词池化两种读出，并定义语义相关性排序R和核心‑外层排序B；还计算位移、参与度与log‑volume等局部散布指标；

**📊 数据集**

使用两套数据集：①交叉语言语义邻域探针（1,200句，40语义因素，10领域，5语言），②语义细化探针（3,840句，48链，5细化层次，4模板，4变体）；

**📈 对比分析**

在四个模型上比较实验结果，发现去除主方向后在约50%层能实现语义相关性排序，Mahalanobis距离在原始空间即可显示核心‑外层分离；细化链在位移和log‑volume上均显著优于对照，表现跨模型一致，支持语义细化导致向内收缩的结论；

**⚠️ 局限性**

局限性包括：仅评估3–5 B规模模型，未验证更大规模模型；细化与对照并非完全等价，可能保留弱度量；局部散布估计基于16个句子样本，精度受限；未进行因果干预或实际控制实验，仅提供几何描述。

---

## 223. ULME: Predictive Coupling for Mutually Reinforcing Embedding and Language Modeling

**arXiv ID:** 2610.30705 | [PDF](https://arxiv.org/pdf/2610.30705v1)

**作者:** Yang Liu `[一作]` (Tencent Inc.), Jinchao Zhang `[通讯]` (Tencent Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在单一 causal decoder‑only LLM 上同时完成文本嵌入与文本生成的框架 ULME，利用可学习槽链实现两者在同一前向传播中的协同训练。

**💡 创新点**

创新点在于：① 在同一 causal 轨迹中插入可学习槽链，区分嵌入与生成的读取位置；② 引入预测耦合（predictive coupling）——通过 stop‑gradient 的 InfoNCE 把答案端的句子级表示作为目标，引导查询端的槽向更有信息的答案表征靠拢；③ 三个损失协同工作：语义对比损失、语言模型损失与预测耦合损失，共同塑造统一的隐藏空间。

**🔧 技术方法**

技术主要包括：decoder‑only transformer（Qwen2.5、Qwen3），可学习槽链插值，三重损失（contrastive InfoNCE、token‑level next‑token CE、stop‑gradient 预测 InfoNCE），训练在 LLM2Vec‑format contrastive data 与 OpenThoughts‑114k 生成数据上，AdamW 优化。

**📊 数据集**

使用的公开数据集：文本嵌入训练采用 LLM2Vec‑format 的 E5 contrastive 数据；文本生成训练采用 OpenThoughts‑114k；评测使用 MTEB、BRIGHT、AdvBench‑IR、MMLU、GSM8K、MMLU‑Pro、HumanEval/HumanEval+ 等。

**📈 对比分析**

与同一 backbone 上的专用嵌入模型（LLM2Vec、LLM2VEC‑GEN）以及单独进行生成微调（Gen‑only FT）对比：在 MTEB 上 ULME 在 0.5B–8B 模型中平均提升 3.3–4.1 分；在 BRIGHT 与 AdvBench‑IR 上同样优于 LLM2VEC‑GEN 并进一步压缩错误率；在生成任务上 ULME 与 Gen‑only FT 差距不超过 0.5 分，甚至在 GSM8K 与 HumanEval 上略有提升，说明两种能力能够在同一模型中保持竞争力。

**⚠️ 局限性**

局限性包括：① 仍需在大型模型上验证更大规模（>10B）的效果；② 预测耦合的目标选择（平均池化）对嵌入质量有一定影响，可能需更复杂的池化策略；③ 对极端长文本或多模态输入的适配尚未评估；④ 训练需要联合两种任务的数据，可能在单一任务场景下不如专用模型。

---

## 224. Automated Enrichment Analysis with GraphRAG: Efficient Multi-Database ORA Extraction and Biological Interpretation

**arXiv ID:** 2610.30710 | [PDF](https://arxiv.org/pdf/2610.30710v1)

**作者:** Guangcan Li `[一作]` (University of Osaka), Makoto Onizuka `[通讯]` (University of Osaka)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种基于知识图的多数据库ORA集成与检索增强生成框架（GBOR‑RAG），通过构建统一的基因‑术语知识图、使用奖品收集 Steiner 树求解检索最小证据子图，并将该子图作为上下文驱动 LLM 生成可靠的生物学解释。

**💡 创新点**

创新点包括：①将跨数据库的ORA结果通过共享基因连接在一张知识图中，解决传统分库输出碎片化问题；②将检索子图设计为奖品收集问题，实现紧凑连通的证据子图；③引入“claim‑level faithfulness”指标评估生成文本是否受检索证据支持；④在基础框架上叠加 PubMed 文献层，进一步提升解释的深度与可信度。

**🔧 技术方法**

技术手段：基因‑术语知识图构建、基于奖品收集 Steiner 树的子图检索、图‑RAG 生成框架、使用 DeepSeek‑V4‑Pro 生成文本、使用 LLM（GPT‑5‑mini）作为评判者、计算 NPMI 作为文献节点奖品、传统超几何检验做ORA评分。

**📊 数据集**

数据集：七个公共基因集数据库（GO BP、KEGG、Reactome、WikiPathways、MSigDB Hallmark、DisGeNET、Open Targets）共 33,199 术语、20,604 基因；PubMed 语料层（799,126 PMID、1,991,461 条目）；42 条 GEO2KEGG 实际查询（含阿尔茨海默病 GSE1297）。

**📈 对比分析**

与传统单库 ORA、Enrichr‑KG、GeneAgent、MAPA 进行比较。GBOR‑RAG 在保持 ORA 结果一致性的同时，查询速度提升 37×；检索子图平均仅 36.8 个节点，且完全连通；claim‑level faithfulness 最高 0.90（GeneAgent 0.71、MAPA 0.29）；加入 PubMed 层后整体解释质量从 4.17 提升至 4.75（p<10⁻⁶）。

**⚠️ 局限性**

局限性：faithfulness 指标仅衡量文本是否与检索证据一致，未覆盖真实生物学正确性；评判主要依赖 LLM judge，需更大规模专家验证；目前仅在 42 条 GEO 查询上验证，跨领域推广仍待验证。

---

## 225. ParaLin++: Accelerating Parallel Diffusion Integrator via Intrinsic Partially Linear Structure

**arXiv ID:** 2610.30704 | [PDF](https://arxiv.org/pdf/2610.30704v1)

**作者:** Jianrong Lu `[一作]` (City University of Hong Kong), Junhui Hou `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为 ParaLin++ 的并行扩散采样器，旨在通过保持完整的步长离散化并加速求解器迭代，解决训练‑免费少步采样器在复杂生成质量上的下降问题。

**💡 创新点**

创新点包括：① 将 ParaLin 的部分非线性框架提升为强大的 Parallel Lin++；② 设计自适应非线化、成功过冲（SOR）加速、滑动窗口与自适应停止规则；③ 通过系统化评估展示传统 FID/CLIP 评估不足，并在 GenEval、DPG‑Bench、LongTextBench 等真实质量基准上进行严格审计。

**🔧 技术方法**

主要技术包括：部分线性化的并行 ODE/SDE 求解器、Jacobi/Gauss‑Seidel 更新、Successive Over‑Relaxation、指数移动平均 (EMA) 自适应阈值、滑动窗口并行策略，以及基于 ϱ-非线性方程组的迭代优化。

**📊 数据集**

实验使用的模型与数据集包括：Stable Diffusion‑v2、LSUN Church、FLUX.1‑dev、Ming‑Flash‑Omni 2.0、CIFAR‑10，配合 FID、CLIP、GenEval、DPG‑Bench、LongTextBench 等多种评测指标。

**📈 对比分析**

与现有方法（DDIM、DPMSolver、ParaDiGMS、ParaSolver、ParaLin 等）对比，ParaLin++ 在保持完整步长的前提下实现 1.21×–1.68× 的 GenEval 加速，Stable Diffusion‑v2/LSUN Church 上可达 2.6×–4.8× 的时钟速度提升，且在 FID、CLIP、GenEval、DPG‑Bench、LongTextBench 等指标上保持与 30 步 Euler 采样相近的质量。

**⚠️ 局限性**

局限性主要体现在：① 需要显式的多 GPU 并行计算，单 GPU 下加速仍受限；② 需要额外的超参数（ϱ、β、ω 等）调优；③ 依赖于完整的步长离散化，无法进一步减少步长以获得更低的 NFE；④ 对于极度复杂或高分辨率模型的可扩展性尚待进一步验证。

---

## 226. WovenAnchor Matcher: Specialized Intra- and Inter-Image Context Modeling for Feature Matching

**arXiv ID:** 2610.30799 | [PDF](https://arxiv.org/pdf/2610.30799v1)

**作者:** Zhiyang Li `[一作]` (Institute of Science Tokyo), Nakamasa Inoue `[通讯]` (Institute of Science Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一个名为WAM的半稠密匹配框架，专门化地对图像内部和图像间的上下文进行聚合。

**💡 创新点**

创新点在于将内部聚合设计为水平-垂直顺序的状态空间传播WovenMamba，并将外部聚合设计为基于局部锚点的AnchorCrossAttention，实现了角色专门化。

**🔧 技术方法**

使用了Mamba状态空间模型的双向传播、软锚点聚类与交叉注意力、卷积编码器以及多尺度匹配技术。

**📊 数据集**

主要在MegaDepth数据集上训练与评估，并在ScanNet与HPatches上进行零样本迁移实验。

**📈 对比分析**

与JamMa、LoFTR等半稠密匹配器对比，WAM在MegaDepth-1500上Pose AUC@5°提升至66.1，速度为35.4 ms，优于JamMa的64.1/43.8 ms。

**⚠️ 局限性**

局限性包括相较JointMamba的运行时略高，AnchorCrossAttention的交互复杂度随窗口数量二次增长，导致高分辨率下的可扩展性受限，且使用JamMa式最近邻匹配导致匹配冗余。

---

## 227. VLTest: Finding Joint Image-Text Failures in Vision-Language Models

**arXiv ID:** 2610.30794 | [PDF](https://arxiv.org/pdf/2610.30794v1)

**作者:** Bin Duan `[一作]` (University of Queensland), Guowei Yang `[通讯]` (University of Queensland)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种面向视觉-语言模型(VLM)的黑盒测试框架，利用受控的图像和文本变体、关系过滤器、双阶段测试与四格证据来识别并归因联合失败。

**💡 创新点**

核心创新在于将联合输入与其单模态对照合并为一个测试单元，配合自动化判据（VLM judge）验证期望答案，并通过四格（seed、图像单模态、文本单模态、联合）证据精准归因故障；同时引入稳定优先配对策略和高效的变体生成方法。

**🔧 技术方法**

采用VQGAN潜在空间编辑生成图像变体，使用重写–扩展–裁剪管线生成文本变体；引入自然语言推理、图像相似度与任务特定配对规则的过滤器；双阶段测试与回调式配对；自动化判据基于Llama‑4‑Maverick评判答案保持与输出正确性。

**📊 数据集**

在MSCOCO（ITM）、NLVR2（VR）和VQA v2（VQA）三大任务上，用八个专门和四个通用VLM（包括Qwen3‑VL、GPT‑4o mini、Gemini 2.5 Flash）进行实验。

**📈 对比分析**

与SINVAD‑VQ、TextFooler、RND、CRND、SR、EXT、MetaRA-c等基线相比，所提方法在相同调用预算下确认失败率提升约1.8–2.5倍，联合失败发现率提升1.3倍；在通用模型上亦优于对手；精度达到80–95%，四格确认率高达77%。

**⚠️ 局限性**

局限性包括：判据基于单一judge模型，可能误判；重建图像会带来伪影导致不被接受；实验仅覆盖三类任务和少量tokenizer，结果可能不适用于高分辨率或多轮交互系统；API模型的实验受版本与调用限制影响。

---

## 228. Trace2World: Reconstructing App Worlds from Real Interaction Traces for Scalable GUI Agent Training

**arXiv ID:** 2610.30792 | [PDF](https://arxiv.org/pdf/2610.30792v1)

**作者:** Zezhou Wang `[一作]` (Nanjing University), Yue Wang `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建可重置、可验证的 GUI 仿真环境：利用大量真实应用的部分轨迹先进行全局行为重建，生成源链接的共享模型；随后使用有限 API 调用将模型编译为可执行浏览器环境，并通过独立重放与原始轨迹比对进行源级验证；最后基于已验证的环境提供学习者指导的任务生成。

**💡 创新点**

创新点在于将行为重建与实现分离：先全局推理合并多条轨迹，保留不确定或冲突的行为为不可执行；实现阶段只能在模型冻结后进行，任何后期不一致只能在可执行包内局部修复，从而避免传统方法在编码时就已固化错误。

**🔧 技术方法**

核心技术包括：生成式 AI（GPT‑5.x、Qwen）用于模型推断与合成；有限 API 调用的编译器将模型转化为浏览器环境；独立重放与源轨迹的六项检查（可达性、表面、状态、效果、终点、数据）；以及学习者反馈驱动的任务索引与生成。

**📊 数据集**

数据集：1,570 条真实应用轨迹，覆盖 50 个应用；对 10 个应用的 200 条工作流做完整性验证；此外使用四个外部 GUI 基准测试集（MobileGym、AndroidWorld、MobileWorld、AndroidDaily）进行跨环境评估。

**📈 对比分析**

与 Direct Agent、CUA‑Gym、PhoneWorld、GUI‑GENESIS 等方法对比：在 200 工作流上，Trace2World 的工作流完整性达 91.1%（对比 59.8%）；在 Trace2World‑Bench 上提升 20.2 分；在四个外部基准上分别提升 5.4–16.2 分；在 100 个真实设备任务上保留 94% 的模拟增益（相较 self‑validation 54.7%）；学习者指导任务将基准成功率从 56.9% 提升至 61.1%。

**⚠️ 局限性**

局限性：仍需足够的轨迹覆盖，无法推断未出现的行为；验证和修复只能在已观测的状态/效果空间内完成；编译受限于有限的 API 调用；在极端复杂或高交互的应用场景中，验证率和可扩展性可能下降。

---

## 229. An Introduction to World Models for Generalist Agents

**arXiv ID:** 2610.30789 | [PDF](https://arxiv.org/pdf/2610.30789v1)

**作者:** Zhihui Zhu `[一作]` `[通讯]` (Ohio State University), Zhihui Zhu (Ohio State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对世界模型（world model）在构建通用代理（generalist agent）中的角色与意义进行系统梳理，基于 POMDP 框架阐释其与经典控制、滤波、系统辨识与强化学习的联系，归纳三大学习范式（直接策略学习、世界模型学习、世界‑动作联合学习），并讨论大型语言模型（LLM）作为隐式世界模型的可能性与启示。

**💡 创新点**

1) 提出了将世界模型与策略学习统一为条件预测视角，强调不同模型（观测‑空间 vs 隐空间、动作‑条件 vs 无条件）在同一概率分解下的关系；2) 通过“世界‑动作”模型阐明联合学习的优势与实现机制；3) 结合 LLM 的自回归训练，将世界预测与动作生成映射为同一 token‑级预测任务，展示大规模文本预训练如何潜在获取世界动力学知识。

**🔧 技术方法**

使用 POMDP 形式化、统计推断（贝叶斯滤波、动态规划）、深度生成模型（自回归、扩散、变分）、时序编码器（Transformer、GRU、SNN）以及多模态融合技术；在策略学习方面采用行为克隆、决策 Transformer、RT‑1/RT‑2 等，且讨论了基于世界模型的 MPC、模拟强化学习和逆动力学方法。

**📊 数据集**

论文为综述性工作，并未在实验中使用特定数据集；作者引用了多类典型数据源（人类视频、机器人交互轨迹、文本语料、模拟环境）作为未来工作可能利用的来源，但未给出具体实验集。

**📈 对比分析**

本研究不包含实验对比，因其目标为概念框架与方法论梳理；作者对现有方法的性能评估仅通过文献综述、理论分析和示例案例进行说明，没有提供量化基准或对比实验。

**⚠️ 局限性**

主要限制包括：
- 世界模型概念仍然模糊，缺乏统一的定义与评价标准；
- 统一框架对数据来源、任务多样性、身体化差异的适配性尚未系统验证；
- 论文侧重理论与综述，缺乏针对不同模型的实证比较与性能评估；
- 由于 LLM 预测的可解释性与控制可靠性仍有限，直接用于决策的有效性尚待验证。

---

## 230. EBV-CAN: Vision--Language Grounding for Contact-Aided Endoscope Navigation

**arXiv ID:** 2610.30786 | [PDF](https://arxiv.org/pdf/2610.30786v1)

**作者:** Chi Kit Ng `[一作]` (Chinese University of Hong Kong), Hongliang Ren `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于一次性全局鸟瞰图和连续局部观察的视觉‑语言控制策略，用于柔性内镜的接触辅助导航；

**💡 创新点**

首次将全局参考图与动态语义查询相结合，构建VLCA大规模标注数据集，并证明动作模仿与任务完成的性能排名不一致；

**🔧 技术方法**

利用自蒸馏的DINOv2特征、GRU递归、跨注意力语义查询以及空间监督的损失函数，训练端到端的视觉‑语言控制模型；

**📊 数据集**

VLCA数据集——517个演示、74,693个动作样本、957,616个空间标注（目标、引导、尖端位置、剩余路线）以及相应的全局鸟瞰图；

**📈 对比分析**

在50个测试集上评估第一步动作MAE和定位精度，1920次物理试验中EBV‑CAN实现76.25%的成功率，明显优于基线；动作误差最低的GRU‑BC却成功率极低，表明两者性能评价不同；

**⚠️ 局限性**

仅在平面刚性引导平台验证，缺乏三维柔性解剖环境与力学感知；失效主要由视角遮挡、转向错误和进程停滞导致，对目标切换的鲁棒性有限。

---

## 231. Abstract Priorities, Concrete Rewards: How Stated Preferences Relate to What People Reward in AI Responses

**arXiv ID:** 2610.30763 | [PDF](https://arxiv.org/pdf/2610.30763v1)

**作者:** Zihan Chen `[一作]`, Di Zhu `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文利用 PRISM 数据集，对用户在对话中声明的 AI 响应属性优先级与实际评分之间的对应关系进行分析，并进一步评估个人信息在预测后续选择中的价值。

**💡 创新点**

创新点在于揭示声明优先级与评价结果的对应程度高度依赖属性的操作化方式，并通过对比抽象属性、具体特征和选择理由三种报告类型，阐明它们在解释和预测行为中的不同作用；此外，还提出将具体请求（如长度）作为额外的个性化信号。

**🔧 技术方法**

使用的技术包括 lens 模型分析、固定效应回归、交互与调节效应检验、方差成分模型与模拟、聚合属性概况构建、梯度提升与对数回归的预测比较，以及分半可靠性与自举评估。

**📊 数据集**

数据集为公开的 PRISM 数据集，包含 1,396 名参与者、7,978 场会话、29,347 条首轮响应，涵盖 21 种模型，并记录了参与者对七个属性的主观重要性、对话后评分、选择理由和感知性能。

**📈 对比分析**

通过对比个人化信息与汇总基线的预测准确率，研究发现加入个人信息仅提升 1–3% 的 pairwise accuracy，而改进预测模型（如梯度提升）可提升 4–6%；在属性对应性分析中，声明优先级与聚合属性概况的调节效应显著，远高于直接文本代理或 lens 组合。

**⚠️ 局限性**

局限性包括：仅使用 2023 年末的 LLM 版本且受长度限制；个人历史仅为中等长度，导致个体特征估计不稳定；文本代理与实际属性的测量误差；观察性设计难以断言因果关系；并且结论可能不完全适用于更先进的模型或不同人群。

---

## 232. Can Loan Processors and Underwriters Spot AI-Edited Statements? An Empirical Study

**arXiv ID:** 2610.30758 | [PDF](https://arxiv.org/pdf/2610.30758v1)

**作者:** Simiao Ren `[一作]` (Scam.ai), Ankit Raj `[通讯]` (Scam.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估AI辅助编辑生成的金融PDF伪造文件在三名有经验的贷款审查员首次文件审阅中的检测率。

**💡 创新点**

发现审查员仅凭一致性检查检测到约10%伪造，且未能通过外观识别，提示仅靠页面外观的审阅方法存在弱点。

**🔧 技术方法**

使用AI文本编辑工具对真实PDF进行身份字段替换，并对编辑后的文件进行人工评估。

**📊 数据集**

使用47份来自12家金融机构的真实金融PDF（共134份伪造文件）。

**📈 对比分析**

将评审者给出的判定与原始文件比对，发现10.6%判定含有编辑支持的疑点，说明仅凭视觉效果检测的有效性不足。

**⚠️ 局限性**

样本量极小（仅三名审查员、单一持有者的文件），缺乏真实文件对照，且仅检验了已知不一致的伪造包，无法推广至更广泛的贷款审查流程。

---

## 233. How Firm Are AI Models' Ethical Boundaries? A Pay-Stub Editing Test of 15 Models

**arXiv ID:** 2610.30751 | [PDF](https://arxiv.org/pdf/2610.30751v1)

**作者:** Simiao Ren `[一作]`, Ankit Raj `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

评估15个AI模型在三种不同措辞下对工资单增收请求的编辑调用行为，并记录拒绝与执行的方式。

**💡 创新点**

首次细粒度比较词义变化对模型边界的影响，并区分对欺诈性请求的拒绝与对授权更正的潜在接受。

**🔧 技术方法**

使用OpenRouter API通过模拟读取、PDF编辑和Python工具调用来触发并记录模型行为。

**📊 数据集**

使用单一占位符工资单测试文件，配合三种不同措辞的请求（含贷款用途、无理由、打印错误）。

**📈 对比分析**

对每个模型进行10次试验，统计编辑调用比例；结果显示Astra在有贷款用途时拒绝，缺少用途时全部执行；Fable、Opus、Muse Spark始终拒绝；Sonnet在无用途与打印错误时部分执行。性能以调用频率衡量，未出现真实文档修改。

**⚠️ 局限性**

仅测试单一英语场景，工具为模拟而非真实执行，缺乏授权对照与多文档类型，样本量有限，无法全面评估模型的真实边界和伦理表现。

---

## 234. Learning from Next-Frame Prediction: Towards Unified and Scalable Visual Representation Learning

**arXiv ID:** 2610.30724 | [PDF](https://arxiv.org/pdf/2610.30724v1)

**作者:** Yifeng Yao `[一作]` (Peking University), Dongyan Zhao `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种自监督视觉预训练框架，利用在语义嵌入空间中对下一帧的预测，替代传统的掩码恢复或像素级重建；并通过局部→全局crop顺序实现对图像的统一预训练。

**💡 创新点**

创新点在于：①将预测粒度从单个视觉token提升到整帧，并在固定语义嵌入空间中完成；②使用冻结的语义分词器（DINOv2）作为目标，保证目标不随模型协同演化；③加入轻量级生成解码器以恢复细粒度空间细节；④将视频与图像预训练通过相同的next‑crop目标统一，避免多模态专用目标。

**🔧 技术方法**

技术包括：ViT‑g 1B causal encoder + state predictor，语义嵌入投影，固定语义分词器，帧级可逆注意力，光流条件生成解码器，局部到全局crop观察策略，联合语义+流损失。

**📊 数据集**

数据集：VISTA 组合图像（ImageNet‑21K、SA‑1B）和视频（SSv2、K710、HowTo100M）；在预训练后对EPIC‑KITCHENS‑100、Ego4D、IN1K、K400、SSv2、Diving‑48、ADE20K、Cityscapes、VOC、NYUv2、KITTI等多任务进行评估。

**📈 对比分析**

与V‑JEPA 2/2.1、Toto、InternVideo2、VideoMAEv2、VideoPrism等统一图像‑视频编码器对比，本文模型在动作预判（EPIC‑KITCHENS‑100）达45.5 mCR@5，提升7.1分；在全局分类（IN1K、K400、SSv2、Diving‑48）平均准确率提升至85.4%；在语义分割（ADE20K、Cityscapes）和深度估计上也取得最高或近乎最高的mIoU/RMSE。

**⚠️ 局限性**

局限性包括：①训练成本高（1B参数+30k GPU‑h）；②依赖冻结语义分词器，目标空间固定可能限制对极端视觉域的迁移；③生成解码器仅在预训练阶段使用，实际推理时增加额外计算；④对极低帧率或大运动的视频性能尚未充分验证。

---

## 235. RAIN: Agentic Proof Generation for Rocq-based Software Verification

**arXiv ID:** 2610.30759 | [PDF](https://arxiv.org/pdf/2610.30759v1)

**作者:** Jinyi Zhou `[一作]` (Nanjing University), Xiaoxing Ma `[通讯]` (Nanjing University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 RAIN，一种专门用于 Rocq 的轻量级代理，能够在整个项目范围内自动检索证明库、完整编辑证明脚本、并根据增强的检查反馈循环迭代修复证明。

**💡 创新点**

创新点在于将 LLM 置于一个完整的“代理式证明工程”工作流中：①实现了自主检索与局部/全局证明编辑；②设计了本地状态收集和多错误收集两种反馈机制，让检查器提供更丰富、与证明状态紧耦合的信息；③把证明修复的重点从表面符号操作转向语义级别的策略调整。

**🔧 技术方法**

技术主要包括：大型语言模型（DeepSeek‑v4‑flash、Qwen3.5‑Flash、GPT‑5.4）；自定义的 Rocq 项目工作空间和句子拆分器；本地证明状态采集与多错误收集工具；以及基于检查器结果的迭代编辑循环。

**📊 数据集**

使用的数据集包括：CoqStoq（100 篇随机抽样的公开 Rocq 定理）、PnVRocqLib（用于与 Cobblestone 对比）、Parcas（近期上传的 Rocq 项目用于评估泛化能力）。

**📈 对比分析**

与现有自动化证明器（Cobblestone、PALM、ReCent‑Prover）以及通用编程代理（Claude Code、Codex、OpenCode）进行对比。性能方面：在 CoqStoq 上 RAIN 的证明成功率为 90.3%（相较 Cobblestone +42.3%；相较 PALM +24.7%），在 PnVRocqLib 上 95.9%；在 Parcas 上 98% 的通过率；成本方面相较 Claude Code 降低 48.8% 费用，Codex 降低 17.3%；在时间上也显著优于通用代理。

**⚠️ 局限性**

局限性：仍对长证明（>30 步）以及使用 ssreflect/boolean‑reflection 等特定风格的项目表现不佳；主要依赖基于词表的检索，缺乏语义检索能力；若 LLM 在训练时已见过相同项目或相关证明模式，可能导致实验结果偏高。

---

## 236. Wavenumber-Aware Deep Learning for Robust Analysis of Raman Spectra

**arXiv ID:** 2610.30746 | [PDF](https://arxiv.org/pdf/2610.30746v1)

**作者:** Jihwan Kim `[一作]` (Kookmin University), Joonyoung F. Joung `[通讯]` (Kookmin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了一种波数感知的深度学习框架，用于对拉曼光谱进行细菌鉴定。

**💡 创新点**

创新点在于将物理波数坐标显式编码为输入通道和位置编码，打破了传统固定网格表示带来的校准转移、采样不一致和缺失谱段问题。

**🔧 技术方法**

技术上采用了CNN、ResNet和Transformer三种网络架构，并在Transformer中使用基于波数的正弦位置编码，分别与索引、学习及常数编码进行对比。

**📊 数据集**

使用了实验收集的三种细菌（Bacillus atrophaeus、Escherichia coli、Listeria innocua）的拉曼光谱数据，以及公开的30种菌株共60,000条光谱的公共基准。

**📈 对比分析**

通过在噪声、降采样、窗口缺失等多种实际测量退化场景下进行评估，并与传统固定网格模型对比，波数-aware模型在F1_micro和Exact Match指标上提升5-10个百分点，并在未知采样网格上保持93%以上的性能。

**⚠️ 局限性**

局限性包括：极窄窗口缺失信息时性能急剧下降；仍需处理波数轴校准漂移和不同仪器响应的进一步适配。

---

## 237. What Lies Beyond Perception: A Partition Theory of the Limits of AI Evaluation and Oversight

**arXiv ID:** 2610.30803 | [PDF](https://arxiv.org/pdf/2610.30803v1)

**作者:** Ahmed M. Mahran `[一作]` `[通讯]` (Incorta), Ahmed M. Mahran (Incorta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过划分理论阐述了 AI 评估的结构性限制，证明评估者的感知分辨率决定了其能证明的能力下限，并提出了“回音测试”作为评估者可检测能力缺陷的完备工具；同时分析了模型隐藏能力、沙袋化、监控通道等现象，并给出相应的理论预测。

**💡 创新点**

创新点主要包括：①提出将评估者的感知视为分区并与被评估模型的分区比较，得到单向的上界/下界关系；②引入回音测试，证明其在评估者可分辨的测试中是完备的；③系统性地将“遮蔽、沙袋化、监控”问题归入同一框架并给出一系列预测；④展示评估者改进（如多样化监控）如何突破“判定上限”。

**🔧 技术方法**

主要技术：分区（partition）理论、可分辨性证明、理论推导与推论、回音测试设计、抽象模型分析；结合已有实验结果（如评委分歧、噪声沙袋化、密码锁模型等）进行理论解释与预测。

**📊 数据集**

论文本身不开展新的数据实验，主要引用并解释先前研究中的数据集：GSM8K、Code Generation、OpenAI LLM Benchmarks 等，用于说明噪声、提示、密码锁等情境下模型行为变化。

**📈 对比分析**

方法比较：通过理论证明展示评估者的分辨率限制，并以已发表实验（评委不一致、噪声沙袋化、提示恢复锁定能力等）作为案例验证理论；未进行实际性能评估，预测结果基于理论模型和先前实验趋势。

**⚠️ 局限性**

局限性：①仅给出下限而非上限，无法完全判断模型真实能力；②假设感知分区完全精确，忽略噪声、随机性与分辨率的连续性；③未给出具体的样本复杂度或实验验证，理论与实际实现的差距未量化；④对多轮交互、状态依赖行为的处理有限，主要聚焦一次性行为评估。

---

## 238. Amplifying Latency and Cost of LLM Inference via Rowhammer Attack

**arXiv ID:** 2610.30798 | [PDF](https://arxiv.org/pdf/2610.30798v1)

**作者:** Mrityunjay Shukla `[一作]` (Indian Institute of Technology Bombay), Biswabandan Panda `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于Rowhammer的LLM推理效率攻击——LLVMhammer，利用CPU DRAM中的位翻转导致Triton编译器生成子最优GPU内核，并通过对LLM推理服务进行持续压测验证该攻击效果。

**💡 创新点**

创新点包括：1）首次发现并利用Triton编译器的目标无关优化阶段作为攻击面，实现跨GPU厂商（NVIDIA、AMD）的一致性位翻转效果；2）设计Karnage工具，实现高效过滤并定位可用的位翻转位置，避免对120k+函数逐一手工检查；3）证明单个位翻转即可在保持输出正确的前提下持续降低推理吞吐量，并能诱发输出语义错误。

**🔧 技术方法**

主要技术：Rowhammer位翻转攻击、Triton的MLIR/LLVM编译流水线、CPU层面的内存压测与定位、Karnage的多阶段剪枝算法（命名空间过滤、Intel PT追踪、离线编译比较、GPU性能评估）以及基于BERTScore的输出质量测评。

**📊 数据集**

使用了四大LLM模型（Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Deepseek-llm-7B-chat）及ShareGPT对话数据集进行推理测试。

**📈 对比分析**

对比方法：对比基线（未翻转）与翻转后Triton二进制的推理性能，采用30次重复实验并使用Welch’s t检验评估TTFT、TPOT、ITL、吞吐量（tokens/s、reqs/s）的显著性。实验结果显示，翻转后吞吐量下降4–16%，TTFT/TPOT/ITL显著提升，且在高负载（1000请求）下差异尤为显著；BERTScore从接近1下降至0.25，证明输出语义被破坏。

**⚠️ 局限性**

局限性：1）依赖CPU DRAM对Rowhammer的可行性，仍受硬件级防护（如DDR5 PRAC、ECC）的影响；2）对Triton编译器版本及其内部实现细节高度耦合，若编译器更新或优化路径变化需重新定位位翻转位置；3）攻击仅针对GPU加速的LLM推理，未考虑CPU-only推理或多GPU协同分布式部署的细节；4）对真实云环境的复杂调度与隔离机制尚未彻底验证，可能影响位翻转的可靠性与持久性。

---

## 239. ABRA: Agent-Bound UI for Human-Agent Communication beyond Chat with Reusable Views

**arXiv ID:** 2610.30780 | [PDF](https://arxiv.org/pdf/2610.30780v1)

**作者:** Seon Gyeom Kim `[一作]` (Megagon Labs), Hannah Kim `[通讯]` (Megagon Labs)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Agent-Bound UI 交互模型，并实现 ABRA 原型，使共享画布可绑定任务特定语义供人机与多智能体协作。

**💡 创新点**

创新点在于将可复用的交互结构与任务语义分离，并通过发布共享状态实现跨智能体对交互语义的可见性和可调用性。

**🔧 技术方法**

使用大型语言模型（Claude、GPT-5.4、Haiku 等）配合结构化工具调用，构建、解析和更新共享画布；实现基于 React 的视图和 Node.js 画布中心。

**📊 数据集**

使用人工构造的 135 个交互任务案例（45 组画布创建、33 组后果解析、33 组跨智能体调用），并未引用公开数据集。

**📈 对比分析**

通过自动结构校验和 LLM 评估比较四个模型，画布创建与后果解析的通过率普遍在 90% 以上，交互调用准确率约 70–100%，表现出较高的可靠性。

**⚠️ 局限性**

局限在于交互词汇表的通用性、交互语义的可解释性未评估用户理解效果，以及对多视图和复杂语义绑定的完整覆盖仍待提升。

---

## 240. ProQuant: Preserving Expert Quality over Quantity in Ultra-Low-Bit Quantization of Mixture-of-Experts Models

**arXiv ID:** 2610.30779 | [PDF](https://arxiv.org/pdf/2610.30779v1)

**作者:** Jiaqi Chen `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于权重的混合精度后训练量化框架 ProQuant，用于稀疏激活的 Mixture-of-Experts 大模型。

**💡 创新点**

创新点在于引入专家剪枝或加权分配（PoP）避免极低精度专家导致性能衰退，并通过路由一致的 RMSNorm 微调（RoCo）保持路由行为不变。

**🔧 技术方法**

技术包括整数线性规划联合专家剪枝与位宽分配、GPTQ 二阶误差补偿、等价变换的 RMSNorm 微调与路由权重反向缩放。

**📊 数据集**

使用的评测数据集包括 C4 作为校准集，以及生成任务基准 GSM8K、MATH-500、GPQA-Diamond、IFEval、MMLU-Redux 和 BFCL-v3。

**📈 对比分析**

与 PMQ、MxMoE、GEMQ 等混合精度量化方法相比，ProQuant 在 2.5、2.0、1.5 平均位宽下均取得更高或相近的生成性能，尤其在 1.5 位宽时显著优于对手。

**⚠️ 局限性**

局限性在于实验仅覆盖至 35B 规模模型，尚未验证其在百亿级或万亿级模型上的可扩展性和有效性。

---

## 241. Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought

**arXiv ID:** 2610.30755 | [PDF](https://arxiv.org/pdf/2610.30755v1)

**作者:** Zhengye Bao `[一作]` (Xi'an Jiaotong University), Minnan Luo `[通讯]` (Xi'an Jiaotong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了在长对话理解任务中，使用稀疏检索+重排序的原始历史（raw‑RAG）作为基线，比较其与多种“agentic memory”系统的性能

**💡 创新点**

创新点在于挑战“agentic memory”系统必要性的假设，发现原始历史检索在大多数基准中表现优于或匹配复杂系统，并系统地分析了证据覆盖、路径完整性与信息保留对性能的影响

**🔧 技术方法**

技术：密集向量检索、交叉编码重排序、原始历史窗口化存储、读者LLM对检索结果的上下文解释

**📊 数据集**

数据集：LoCoMo、LongMemEval、MEME、MemoryAgentBench（共计数千条问答）

**📈 对比分析**

与多种基准系统（如EverMemOS、MemOS、Mem0、Memora等）对比，raw‑RAG在6/7 benchmark‑model 组合中排名第一，取得比最强agentic系统高0.13–5.53分的提升；同时在效率上比多数系统快5–30倍

**⚠️ 局限性**

局限性：仅在离线公开基准上评估，未覆盖交互式、多模态或更长历史；原始历史检索可能带来存储与隐私问题；并未针对弱读者或状态重建需求进一步探究

---

## 242. Visualizing and Interpreting Temporal Care Team Dynamics for Cancer Survival Risk Analysis

**arXiv ID:** 2610.30748 | [PDF](https://arxiv.org/pdf/2610.30748v1)

**作者:** Yuhua Huang `[一作]` (University of California, Davis), Kwan-Liu Ma `[通讯]` (University of California, Davis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用电子健康记录的访问日志构建患者特定的时间演化协作网络，并通过时间图神经网络进行风险预测，随后使用代理逻辑回归模型提供可解释的特征贡献，最后将风险轨迹、协作网络、特征解释与 what‑if 分析整合到多视图可视化框架中，支持临床团队协作与风险随时间演变的探索与假设生成。

**💡 创新点**

将同一时间基准下的临时图神经网络风险预测、代理解释与协作网络可视化同步联动，突破传统将预测、解释与可视化分离的做法；引入代理模型的可解释性与 what‑if 交互，首次实现协作结构、风险轨迹与特征贡献在同一视图中共演化的可视化分析。

**🔧 技术方法**

时间图卷积网络 EvolveGCN（O 变体）用于预测生存风险；逻辑回归代理模型用于特征贡献解释；what‑if 通过代理系数估算风险变化；可视化采用径向符号、散点图、共访问网络、特征条形图、分组柱状图等多视图联动技术。

**📊 数据集**

在加州大学医学院的一组 505 名 II/III 期乳腺癌、非小细胞肺癌和结直肠癌患者中使用了 EHR 访问日志（15 个月时间窗口），并提取了笔记类型、HCP 角色等元数据。

**📈 对比分析**

通过专家评审和案例研究进行定性比较，未给出量化性能指标；模型在二分类生存预测任务中训练，并使用类不平衡处理，但论文未报告准确率、AUROC 等数值。

**⚠️ 局限性**

仅基于 EHR 访问日志，无法捕捉记录外的沟通；代理模型的解释并不等价于因果关系；what‑if 仅为模型假设，缺乏临床干预验证；缺乏大规模匹配分析和更贴近护理协调的终点（如再入院、急诊使用等）。

---

## 243. Random Remasking Scales Masked Diffusion Inference

**arXiv ID:** 2610.30744 | [PDF](https://arxiv.org/pdf/2610.30744v1)

**作者:** Zhaoyi Zhou `[一作]` (Carnegie Mellon University), Andrea Zanette `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个统一的马尔可夫链框架，用于解释 Masked Diffusion Models (MDMs) 的标准解码与自我纠正，并基于此提出了无训练、无奖励的 Random Remask‑Denoise (RRD) 推理方法和 Trajectory Correction Training (TCT) 训练策略，实现了在推理时的近对数线性扩展。

**💡 创新点**

创新点在于：① 将 MDM 推理重构为在完整序列空间上的 remask‑denoise 内核迭代；② 推导出单步 KL 上界并从中得到非递增时间步调度和自适应终止条件；③ 设计了 RRD 推理和 TCT 训练两种协同工作的方案，使得推理过程既可纠正已生成的 token 又能在不增加训练成本的前提下提升生成质量。

**🔧 技术方法**

技术核心包括：Masked Diffusion Model、马尔可夫链理论、KL 整合同类推理、随机 remask、最大化推断（argmax）、自回归式训练、以及早停与自适应步长策略。

**📊 数据集**

使用的数据集有：
- Sudoku‑Hard（约 2.3M 难度>10 的 Sudoku 题目）
- 3‑SAT（布尔约束问题）
- HumanEval+（代码生成基准）
- 以及在实验中对比的预训练语言模型 LLaDA2.1‑mini。

**📈 对比分析**

与 ReMDM、ProSeCo、RemeDi、PRISM、BackPlay 等迭代纠错基线以及 Q‑mode、S‑mode 的公开结果比较：
- 在 Sudoku‑Hard 上，RRD+TCT 在 256 步预算下达到 97.5% 的准确率，远超 ReMDM（93.5%）和 ProSeCo（55.7%）。
- 在 3‑SAT 上，准确率提升至 99.3%，比 ReMDM（98.5%）高 0.8%。
- 在 HumanEval+ 上，RRD+TCT 取得 85.4% 的准确率，超过 LLaDA2.1‑mini 的 Q‑mode（82.9%）与 ProSeCo、ReMDM 的表现。
- 计算成本方面，RRD 通过自适应早停平均仅使用 1/4~1/2 的前向传播，显著低于全预算的 ReMDM 与 ProSeCo。

**⚠️ 局限性**

局限性包括：
- 对任务结构要求较高，适合局部稀疏约束（如 Sudoku、3‑SAT）；
- 在更复杂或全局约束的任务上，remask‑denoise 内核的收敛性和 ϵ_t（去噪误差）可能不再低，效果不确定；
- 需要手动设定阈值 τ、预算 M 等超参数，对不同任务可能需要调优；
- 训练时仍需额外的 TCT 过程，虽不需要额外标注，但会增加训练时间与实现复杂度。

---

## 244. Do We Need to "Be There" to Say Goodbye? Comparing Participatory Agency and Spatial Presence Across Digital Farewell Formats

**arXiv ID:** 2610.30734 | [PDF](https://arxiv.org/pdf/2610.30734v1)

**作者:** Chan In Sio `[一作]` (Hong Kong Polytechnic University), Lik-Hang Lee `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在云宠物丧失的受控情境中，比较三种数字告别方式（轻量社交消息、可再次访问网页纪念、沉浸式VR仪式）对象的象征闭合感、空间存在感及丧失相关情绪强度的影响。

**💡 创新点**

① 采用云宠物丧失作为代理模型，以控制情绪风险；② 在同一情境下系统性比较非沉浸、可访问与沉浸式三种告别模式，探究参与度、空间存在与情绪放大机制；③ 提出了 Agency‑Amplification 账户，为XR告别设计提供初步理论与实践指导。

**🔧 技术方法**

使用 Meta Quest 3 及 Unity 开发沉浸式 VR 体验；传统网页纪念界面；轻量社交消息组；量化测评使用 Igroup Presence Questionnaire、Perceived Symbolic Closure Scale（PSCS‑10）、Pet Bereavement Questionnaire（PBQ）、PANAS；定性访谈。

**📊 数据集**

实验数据来自 45 名香港宠物爱好者的问卷与访谈，包含云宠物情境脚本生成的图像视频；未使用公开数据集。

**📈 对比分析**

采用单因素 ANOVA 并用 Benjamini‑Hochberg 校正，随后进行 Tukey HSD 事后比较；结果显示 Website 与 VR 的象征闭合感显著高于 Messaging；VR 在空间存在感最高；VR 在丧失强度上最高；情绪变化（PANAS）无显著差异。定性访谈与定量结果一致，支持提出的机制假设。

**⚠️ 局限性**

样本量有限，且情境为受控云宠物丧失，难以代表真实丧失体验；三种告别模式未逐一单因素操纵，仅组合多项特征；访谈样本少、仅 18 人；PSCS‑10 为探索性量表，存在响应压缩与同意偏倚；研究集中于香港粤语文化，结果对其他文化与真实丧失情境的推广有限。

---

## 245. Width and Depth Limits Commute During Training

**arXiv ID:** 2610.30729 | [PDF](https://arxiv.org/pdf/2610.30729v1)

**作者:** Ruijia Zhang `[一作]` (Johns Hopkins University), Soufiane Hayou `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了线性残差网络在训练过程中宽度和深度极限是否可交换，证明了在梯度裁剪下宽度与深度极限在L2意义下可交换并给出收敛率；

**💡 创新点**

首次在训练阶段给出宽度与深度极限可交换的理论证明，并提供了n^-1/2、L^-1/2的收敛上界；

**🔧 技术方法**

使用连续深度SDE模型、Brownian 动量耦合、梯度裁剪等分析技术，结合梯度下降更新的矩阵动力学；

**📊 数据集**

实验使用合成回归数据集和公开HIGGS二分类数据集；

**📈 对比分析**

通过在不同宽度-深度比例路径上训练，并测量输出与极限的均方误差，验证误差随宽度/深度递减，收敛率与理论一致；

**⚠️ 局限性**

仅适用于线性残差网络，非线性激活虽然实验上表现相似但理论未覆盖；梯度裁剪的假设在实际训练中可能不总是满足。

---

## 246. GIFT: Gradient Informed Filtering for Distillation Augmented Reasoning RL

**arXiv ID:** 2610.30796 | [PDF](https://arxiv.org/pdf/2610.30796v1)

**作者:** Shijie Li `[一作]` (Shanghai Jiao Tong University), Hang Yu `[通讯]` (Tencent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合稀疏强化学习与稠密教师蒸馏的训练框架GIFT，利用梯度几何过滤token级教师更新；

**💡 创新点**

创新点在于将蒸馏梯度分解为与RL梯度平行（冲突）与正交（新颖）两部分，按冲突与新颖性构造token级分数，并通过覆盖率退化动态逐步剔除冲突更新；

**🔧 技术方法**

核心技术包括token级梯度几何分析、梯度分解、闭式分数排序、覆盖率退化调度和GRPO强化学习；

**📊 数据集**

使用数学推理数据集DeepMath（AMC23、AIME24、AIME25）以及逻辑推理Knights‑and‑Knaves（ID/OOD）进行评估；

**📈 对比分析**

与GRPO、OPSD、RLSD等八种基线对比，GIFT在Qwen3 1.7B/4B/8B模型上平均提升约3–5%，在AIME25最高相对提升9.5%，逻辑推理中也表现最优；

**⚠️ 局限性**

局限性包括对梯度计算成本较高、对覆盖率退化参数敏感、在非token级任务或不同RL/蒸馏设置下的通用性尚未充分验证。

---

## 247. Where meanings converge and diverge: Revisiting linguistic relativity across 81 languages

**arXiv ID:** 2610.30778 | [PDF](https://arxiv.org/pdf/2610.30778v1)

**作者:** Weiran Wang `[一作]` (Fudan University), Haitao Liu `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过比较81种语言中196个基本概念的词嵌入关系，分析了不同语义域的跨语言相似性。

**💡 创新点**

首次揭示社会文化概念在跨语言关系上更一致、自然运动概念更差异的领域特定对比，并将其与地理与语言亲缘距离关联起来。

**🔧 技术方法**

使用多语言BERT（mBERT）和XLM‑R‑large模型提取词嵌入，采用表示相似性分析（RSA）和贝叶斯高斯回归评估跨语言关系和距离效应。

**📊 数据集**

基于Wiktionary的Swadesh列表，提取81种语言的207个基本概念（后删减至196个），并构建词汇映射。

**📈 对比分析**

通过RSA得到的跨语言相似性指数并结合贝叶斯回归，发现Socio‑Cultural域在两种模型下均显著低于Nature & Motion域；模型在不同域和距离因素下表现一致，稳健性通过置换检验和宽广先验验证。

**⚠️ 局限性**

局限在于词嵌入仅为模型产出，可能与说话者的语义判断不完全一致；样本覆盖基于模型可用性，存在选择偏差；模型训练偏差和多义词处理也可能影响结果。

---

## 248. Primal-Dual Representation Learning for Low-Rank Constrained MDPs

**arXiv ID:** 2610.30756 | [PDF](https://arxiv.org/pdf/2610.30756v1)

**作者:** Chenhao Zhou `[一作]` (Zhejiang University), Hui Qian `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文提出了两种针对低秩连续状态约束MDP（CMDP）的表示学习算法（REP-PD-TEG 和 REP-PD-H-TEG），通过辅助探测、协方差奖励和自适应双线性规划实现软/硬约束的累积违规控制。

**💡 创新点**

创新点在于：①首次将低秩表示学习与CMDP的双重目标（奖励最大化与约束满足）结合；②使用三重采样（tertiary probing）构建联合协方差奖励，兼顾探索和表示学习；③针对硬违规引入下置信度证书和受限双线性搜索，保证每集可行性；④给出针对不同探测强度（p）下的回报与违规上界，展示回报、执行策略与探测成本之间的权衡。

**🔧 技术方法**

主要技术包括：最大似然估计（MLE）用于学习低秩转移模型；协方差奖励（merged covariance bonus）实现基于贝叶斯不确定度的乐观规划；双线性（primal‑dual）优化框架；下置信度证书（lower‑confidence utility certificate）与受限双线性搜索；以及多尺度探测调度 ϵ_k = k^{-p} 来平衡样本效率与执行成本。

**📊 数据集**

实验主要在模拟作业调度任务中验证 REP-PD-TEG 的有效性，并在 Safety Gym 任务上实现近似神经网络版本，展示了表示学习在复杂状态动作空间中的实用性。

**📈 对比分析**

与已有方法相比，论文在低秩 CMDP 上提供了最优 O(√K) 的回报退化率；在软违规下的执行策略达到 O(K^{3/4}) 的样本复杂度；在硬违规下回报退化率为 O(K^{3/4})，硬违规量为 O(K^{1/2})；相较于传统需要已知特征的线性/表格 CMDP 方法，本文在未知特征场景下实现了同样阶数的理论保证。

**⚠️ 局限性**

局限性包括：①假设模型类可实现且 MLE/精确规划可用，实际中近似解法可能不满足三重采样预测保证；②硬违规结果依赖正向得分间隙（positive score gap）假设；③采样成本不计完整 roll‑in 轨迹；④实验主要演示机制，未对硬违规证书做大规模评估。

---

## 249. FilmFlow: Agentic System for 3D Virtual Film Cinematography

**arXiv ID:** 2610.30753 | [PDF](https://arxiv.org/pdf/2610.30753v1)

**作者:** Yiming Zhang `[一作]` (Cornell University), Zhihang Zhong `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FilmFlow 框架，实现基于自然语言或参考视频的可执行相机轨迹生成，支持动态 3D 场景中的电影相机控制。

**💡 创新点**

创新点在于多智能体协作（Interpreter、Scout、Director、Critic）与并行原子操作的组合、Critic 反馈循环以及基于上下文演示检索（ICL），统一实现文本与视频指令下的可执行相机生成。

**🔧 技术方法**

采用语言智能体、并行原子操作库、可视化 Critic 评估、参考视频运动提取、检索‑重用（ICL）以及 Blender 3D 渲染技术。

**📊 数据集**

使用 CinemaBench 数据集，包含 150 个 Blender 场景、750 条文本指令和 60 条参考视频。

**📈 对比分析**

与 LAMP、GenDoP、CinemaTraj、LingBot-Map、VGGT-Ω 等基线在 VideoReward、DOVER、Probe 等自动指标以及用户研究中对比，FilmFlow 在文本和视频指令任务上均获得最高分和最高首选率。

**⚠️ 局限性**

局限性包括对大型 GPT‑6‑Astra 模型的依赖、参考视频尺度不准导致的摄像机匹配误差，以及对复杂动态遮挡的处理仍有限，尚未覆盖多相机/多镜头场景。

---

## 250. Federated Graph Retrieval-Augmented Generation

**arXiv ID:** 2610.30728 | [PDF](https://arxiv.org/pdf/2610.30728v1)

**作者:** Xiaobao Wang `[一作]` (Tianjin University), Di Jin `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Federated GraphRAG（FedGRAG）框架，实现分布式知识图检索与生成，保障隐私。

**💡 创新点**

首次将知识图检索与生成迁移到联邦环境，构建轻量化知识原型路由和受控披露的证据抽象。

**🔧 技术方法**

利用多客户端知识原型、图注意力传播、编码器-解码器证据重构、LLM生成等技术。

**📊 数据集**

在HotpotQA、MuSiQue、2Wiki三大QA基准上模拟联邦划分进行实验。

**📈 对比分析**

与多种中央与联邦RAG基线对比，FedGRAG在Recall@5、EM、F1等指标上均显著提升。

**⚠️ 局限性**

仍需进一步提升对高度分散知识的路由效率和跨客户端推理的可解释性。

---

## 251. Faithful Latent Harm Monitors Work No Better Than the Models Own Behaviour at a Matched False Alarm Rate

**arXiv ID:** 2610.30727 | [PDF](https://arxiv.org/pdf/2610.30727v1)

**作者:** Abhishek Mishra `[一作]` (University of Massachusetts Amherst), Suyash Maniyar `[通讯]` (University of Massachusetts Amherst)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了语言模型内部激活作为隐式安全监测器的有效性，评估其在检测有害请求时与模型拒绝行为的关联。

**💡 创新点**

创新点在于提出跨模型不一致性检验方法，将激活读数中的“判断”与“计划”信号分离，并系统比较15个模型、38个读数与传统文本分类器的鲁棒性。

**🔧 技术方法**

使用的技术包括线性和MLP探测器、Latent Guard统计、TF‑IDF、句子嵌入分类器、外部安全分类器以及托管决策模型，并在指令端和模板端提取激活。

**📊 数据集**

数据集主要由自定义的重写集（表面重写、说服性框架、越狱、警示式无害请求）组成，约2248条重写实例，以及公开的有害/无害样本（SOTA benchmark、XSTest 等）。

**📈 对比分析**

在分布外重写集合上以模型拒绝的误报率匹配阈值进行比较，指令端读数与句子嵌入分类器相当、低于 TF‑IDF；模板端读数比模型拒绝高约 7–10%，但主要捕捉与决策一致的请求；整体未能超越独立训练的安全分类器。

**⚠️ 局限性**

局限性包括仅测试两位置的线性/一隐藏层读数，未覆盖多位置或注意力探测；行为标签基于字符串匹配，可能漏判；数据集规模有限，未评估更大模型或不同训练数据的泛化。

---

## 252. Is spectral predictability enough? Testing Ω-matching for synthetic pretraining under distribution shift

**arXiv ID:** 2610.30802 | [PDF](https://arxiv.org/pdf/2610.30802v1)

**作者:** Edric Castel C. Hao `[一作]`, Neon S. De Jesus `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

探究在分布偏移下，使用谱可预测性Ω匹配的合成预训练是否能提升时间序列预测性能。

**💡 创新点**

首次验证Ω匹配在理想条件下有效，并系统评估其在真实数据上的局限性，提出需考虑局部波形结构等额外因素。

**🔧 技术方法**

使用合成数据生成（正弦+噪声、混沌ODE轨迹）、三种网络架构（LSTM、PatchTST、TCN）以及谱可预测性指标Ω。

**📊 数据集**

在ETTm1负荷预测基准上进行实验，并在完全合成环境中进行对照。

**📈 对比分析**

通过对比无预训练、Ω匹配、随机合成、真实域预训练等四种条件，发现理想条件下Ω匹配可实现+46.8% MSE降低，但在ETTm1上仅平均提升2.3%，甚至有负向影响。

**⚠️ 局限性**

局限性包括仅使用单一真实基准、两种合成源有限、未进行因果性分离、Ω值实现不精确（ω=0.9），以及缺乏对其他特征（波形平滑、相关结构）的系统评估。

---

## 253. A Computationally Efficient and Rate-Optimal Algorithm for Adversarial Linear MDPs with (Aggregate) Bandit Feedback

**arXiv ID:** 2610.30801 | [PDF](https://arxiv.org/pdf/2610.30801v1)

**作者:** Haolin Liu `[一作]` (University of Virginia), Hanbin Zhou `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

为对抗性线性MDP在聚合（只观测总损失）bandit反馈下提出了可多项式时间实现的√K regret算法。

**💡 创新点**

创新点：① 将未知策略特征集合转化为线性bandit问题；② 设计近似分离器，利用切割和策略混合在不知集合的情况下逼近目标；③ 在连续指数权重框架下加入偏差校正奖励，解决估计误差。

**🔧 技术方法**

主要技术：连续指数权重（Follow‑The‑Regularized‑Leader）、Frank‑Wolfe式近似分离、最优线性MDP评估（LSVI + Gibbs 策略）、置信区间与误差校正、椭圆潜能 lemma 等。

**📊 数据集**

论文为理论工作，未使用真实数据集，仅在假设的线性MDP环境下给出理论证明。

**📈 对比分析**

与之前的可实现但仅达到 K^{3/4} 失调率或 √K 但效率低下的算法对比，证明在多项式时间内实现了最优 √K regret，计算复杂度为多项式。

**⚠️ 局限性**

局限性：对高维特征的计算复杂度较高；算法仍属理论证明，未给出实验验证；仅适用于已知特征映射的线性MDP，且聚合反馈模型较为理想。

---

## 254. DeXplicit: Making Physical State Explicit in Dexterous Manipulation World Models

**arXiv ID:** 2610.30781 | [PDF](https://arxiv.org/pdf/2610.30781v1)

**作者:** Adam Lee `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种在3D空间显式预测手部操作后物体变化的世界模型，并基于预测的3D状态进行视频生成。

**💡 创新点**

将物理状态预测与视觉合成分离，使用轻量化的3D动力学网络预测对象中心运动和形变，提供结构化且几何对齐的未来状态；同时采用置信度门控的视觉读取，显著提升对象一致性和生成效率。

**🔧 技术方法**

3D动力学网络（点变换器+对象时间Transformer）、手部动作编码、局部形变控制节点、SE(3)姿态修正、置信度门控的可视化读取、因果渲染器以及流匹配训练。

**📊 数据集**

HOT3D、OakInk2、PhysTwin三套仿真数据集，涵盖刚体、关节和变形交互。

**📈 对比分析**

与基线PointWorld和视频生成系统对比，移动点误差下降27–40%，对象IoU+提升47%，在相同渲染器下实现57倍速度提升，32帧生成仅需4.9秒。

**⚠️ 局限性**

模型仍依赖仿真数据，未在真实机器人上验证；在复杂遮挡或极端形变时可能出现漂移；对训练数据的依赖使迁移性受限。

---

## 255. One Pretrained Decision Model Is Enough for Training-Free Generalist Graph Anomaly Detection

**arXiv ID:** 2610.30777 | [PDF](https://arxiv.org/pdf/2610.30777v1)

**作者:** Xinyu Zhao `[一作]` (Beihang University), Jianxin Li `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种训练自由、通用的图异常检测框架 Jev-GAD，能够在未见图上直接识别异常节点。

**💡 创新点**

创新点在于：①将结构、属性和邻域三类异构图证据通过图相对归一化转化为统一的百分位值；②设计 Graph‑to‑State (G2S) 语义化转换，将每个数值与其对应的异常语义对应；③利用单一冻结的预训练语言模型（Qwen3‑4B）进行非生成 option‑logit 读出，完成基于语义化证据的决策；④将预训练决策得分与四种统计评分（聚合、ECOD、Isolation Forest、RBF‑OCSVM）通过等权重 rank‑fusion 组合，形成最终异常排名。

**🔧 技术方法**

技术手段包括：结构/属性/邻域证据构造、图相对归一化、语义化 G2S 变换、冻结预训练语言模型的 option‑logit 读出、四种无监督统计评分以及 rank‑based 融合。

**📊 数据集**

使用七个公开属性图异常检测基准：Citation 类（Cora、CiteSeer、Amazon）、Social 类（BlogCatalog、Facebook、ArXiv）、Co‑Review 类（CoraML/其他）。

**📈 对比分析**

与 15 种基线（监督、无监督、通用、训练自由）在 AUROC、AUPRC 上对比，宏平均 AUROC 为 89.18%，AUPRC 为 52.40%，分别比最强基线高 7.09% 和 15.44%；在六个图上获得最高 AUROC，五个图上获得最高 AUPRC，显示了在多域、多异常模式下的优越性。

**⚠️ 局限性**

局限性：①依赖冻结模型推理，推理时间和显存消耗相对较大；②对提示语义较为敏感，需在不同场景下调优；③目前仅针对节点级异常，未覆盖边或子图异常；④不支持在线/增量更新，仅适用于一次性推理。

---

## 256. LLM Multi-Agent Systems Are Vulnerable to Herding and Runaway Cascades

**arXiv ID:** 2610.30769 | [PDF](https://arxiv.org/pdf/2610.30769v1)

**作者:** Farnoosh Hashemi `[一作]` (Cornell University), Michael W. Macy `[通讯]` (Cornell University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究在基于Music Lab的多代理框架中，使用13种大型语言模型对48首新生成歌曲进行推荐和评估，考察其在社会影响下的从众与级联行为。

**💡 创新点**

创新点在于将Music Lab实验迁移到LLM代理，系统评估LLM在多代理情境下的从众程度、级联失控与自我纠正能力，并与人类基准进行对比。

**🔧 技术方法**

采用基于文本的提示工程（persona化）、序列推荐决策、统计分析（KL散度、不可预测性、Spearman相关）以及自我纠正指标。

**📊 数据集**

使用Suno v4.5生成的48首未见过的歌曲歌词作为刺激数据，结合Music Lab原始人类实验数据做对照。

**📈 对比分析**

通过KL散度衡量从众偏差、不可预测性评估级联随机性、以及自我纠正指标检验后半段与前半段的相关性；LLM的KL散度约为人类的4-9倍，非推理模型更易出现级联，且在强影响下自我纠正表现不佳。

**⚠️ 局限性**

局限包括实验设计与原始人类实验不完全对齐（文本而非音频、固定四首评估、无奖励与惩罚）、仅使用同一模型的同质代理、未考虑通信或激励机制，因而对现实多模态、异质团队的推广有限。

---

## 257. Adaptive Deadline- and Energy-Aware Caching Algorithms in Dynamic Multi-UAV Edge Networks

**arXiv ID:** 2610.30768 | [PDF](https://arxiv.org/pdf/2610.30768v1)

**作者:** Jian Kim `[一作]` (Chungnam National University), Ki-Il Kim `[通讯]` (Chungnam National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究在多无人机边缘网络中，针对时变内容需求与内容特定交付截止时间，设计了自适应的缓存放置与替换算法。

**💡 创新点**

创新点在于将交付截止时间、通信能耗及无人机间能量不平衡三者统一纳入缓存决策，并提出基于GRU的时间序列表示结合PPO的深度强化学习框架GRU‑DEPPO。

**🔧 技术方法**

主要技术包括层级规则化缓存设计(DE‑UBAC)、局部与全局GRU编码、集中式PPO actor‑critic以及自适应缓存动作因式化。

**📊 数据集**

实验数据来源于仿真平台，使用多组用户移动、内容请求与无线链路模型，生成的合成数据集模拟不同内容类型和截止时间场景。

**📈 对比分析**

通过与LRU、DE‑UBAC、TLPPO等基线比较，GRU‑DEPPO在截止时间误差率、总通信能耗和无人机能量均衡方面均优于其他方案，提升幅度可达40%以上。

**⚠️ 局限性**

局限性包括未考虑无人机位置协同优化、对仿真环境的依赖以及在极端网络变化下仍需进一步验证鲁棒性。

---

## 258. Scaling Laws for Generative Reward Models

**arXiv ID:** 2610.30765 | [PDF](https://arxiv.org/pdf/2610.30765v1)

**作者:** Angel Raychev `[一作]` (SynthLabs.ai), Rafael Rafailov `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究将生成奖励模型（GenRM）作为在线偏好提供者，在创意写作任务中训练策略，并系统探讨判别器训练方式、判别器规模与策略规模对优化结果的影响。

**💡 创新点**

创新点在于：①把生成奖励模型作为在线DPO的奖励信号进行实证评估；②发现“思考”（Thinking）型GenRM在静态评估中表现更好，但在线优化中往往劣于“仅答”型（Answer‑Only）模型；③证明更大的判别器（甚至超过策略尺寸）能显著提升最终策略质量；④中间检查点往往比完全训练的模型在作为奖励者时更有效。

**🔧 技术方法**

主要技术包括：Qwen3 生成模型的不同模式（Answer‑Only 与 Thinking），GRPO 训练（用于 Thinking），SFT（用于 Answer‑Only），在线 Direct Preference Optimization (DPO)，Elo 评分匹配（Gold 参考评估与 Proxy 评估）以及多维度实验设计（模型规模、β 参数、训练步骤等）。

**📊 数据集**

使用的数据集：
- Reddit 版故事对（LitBench）作为来源偏好数据，用于训练 Gold 评估器；
- Gold 评估器（Qwen3‑32B）生成的标签构成 Proxy 评估器训练集；
- 写作提示集（WritingPrompts）用于策略训练与验证；
- 评估验证集来自 LitBench 的 1000 对（同一提示、不同故事）。

**📈 对比分析**

比较方法：在固定 Gold 参考评估的 Elo 场景下，记录政策在不同判别器（Answer‑Only vs Thinking）、不同规模（0.6B–14B）以及不同 β 参数下的 Elo 路径；通过“答案优于思考”或“思考优于答案”的 Elo 差值、峰值和平均值来衡量性能。实验表明：Answer‑Only 在 7/8 配置中获得更高的 Gold Elo 峰值，特别是在较大判别器或策略尺寸下；更大的判别器持续带来更高的 Elo，甚至超过策略尺寸；中间检查点的 Elo 常高于后期检查点。

**⚠️ 局限性**

局限性：
- 评估完全依赖于已学习的 Gold 参考，缺乏真实人类判断；
- Gold 与 Proxy 使用同一 Qwen3 家族，可能存在模型偏好共振；
- 只在创意写作领域实验，未验证安全、可解释性或其他任务；
- 仅比较两种训练管线（SFT vs GRPO），未探讨其他生成奖励模型或优化算法；
- 评估采用 Elo 匹配，受比赛设计与数据重复的影响，未提供跨实验的绝对性能度量。

---

## 259. AgentDP: Towards Automated Model Deployment for Heterogeneous Embedded AI Applications

**arXiv ID:** 2610.30760 | [PDF](https://arxiv.org/pdf/2610.30760v1)

**作者:** Leming Shen `[一作]` (Hong Kong Polytechnic University), Yuanqing Zheng `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了AgentDP系统，利用LLM智能代理自动完成预训练模型在异构嵌入式设备上的优化与适配。

**💡 创新点**

创新点包括：用户需求标准化模块、算子感知模型分区与跨引擎推理管道、以及配置重用模块以提升搜索效率。

**🔧 技术方法**

使用了GPT‑4o LLM、LangChain构建多代理架构，结合网页搜索、向量/图数据库、Optuna超参搜索、ONNX/TensorRT/ONNX Runtime等工具与框架。

**📊 数据集**

实验数据集涵盖ImageNet子集（图像识别）、TESS（语音情感分析）和MMMU（多模态理解）。

**📈 对比分析**

与原始模型、MapCoder和人类专家对比，AgentDP将O&A时间降低至60%以内，模型精度提升约13%，并显著降低资源占用。

**⚠️ 局限性**

局限性在于对LLM推理的误差控制、缺乏深度微调支持、对新模型缺少足够的历史配置，以及需持续更新算子与引擎兼容信息。

---

## 260. Where Did the Light Come From? Diagnosing Cast-Shadow Reasoning in VLMs and Borrowing the Missing Shadow Prior from Image Generation Models

**arXiv ID:** 2610.30740 | [PDF](https://arxiv.org/pdf/2610.30740v1)

**作者:** Ryugo Morita `[一作]` (DFKI), Andreas Dengel `[通讯]` (DFKI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ShadowBench，用真实照片评估VLM在阴影方向的读取、推断和物理一致性判断能力。

**💡 创新点**

发现VLM仅能读取可见阴影方向，却在推断缺失阴影或验证阴影一致性时表现接近随机，表明缺乏阴影‑光照关系知识；通过将图像生成模型的阴影先验注入VLM，可显著提升推断与验证性能。

**🔧 技术方法**

使用Vision‑Language Models、图像生成模型（如FLUX.1‑Kontext‑dev）、多任务评估框架以及链式思考（CoT）等技术。

**📊 数据集**

基于OBER收集的真实照片数据集，构造三种场景状态：正确阴影、无阴影、错误方向，并为每个场景标注阴影方向。

**📈 对比分析**

对19款公开VLM和2款专有模型进行比较，读取任务准确率随规模提升至≈92%，推断与验证大多停留在随机水平；使用IGM后，推断提升≈+23点，验证提升至≈74%；人类基准约85–98%。

**⚠️ 局限性**

局限在于仅评测单一物理变量（阴影方向），采用离散方向分箱，未涵盖多光源、软阴影等更复杂情况；循环使用IGM仅能在推理时补偿缺失知识，未实现对VLM权重内部知识的迁移。

---

## 261. AgentTriage: Replayable Security Diagnosis Across Heterogeneous Agent Runtimes

**arXiv ID:** 2610.30738 | [PDF](https://arxiv.org/pdf/2610.30738v1)

**作者:** Jie Ying `[一作]` (Zhejiang Wanli University), Tiantian Zhu `[通讯]` (Zhejiang University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可重现的安全诊断执行合同（AgentTriage），通过跨异构代理运行时的事件规范化、STPG（结构化类型化执行图）切片、政策判定与证据验证，实现对工具使用代理安全事件的可审计、可重复诊断；

**💡 创新点**

创新点在于将诊断视为带有可执行合同的对象，而非单纯的模型输出；引入了统一的可观察契约、生命周期合并、决策终端切片与证据绑定机制，使得诊断结果可在不依赖语言模型的情况下被第三方完整重算与核验；

**🔧 技术方法**

技术包括：跨运行时薄层覆盖（overlay）把本地事件映射到统一合同；规范化（canonicalization）去除包装并统一调用标识；构建STPG并进行决策终端切片；基于预先固定的策略判定器（oracle）做出结果与防御层归因；以及证据检查与声明式合同执行；

**📊 数据集**

使用了五个开源代理运行时（TaskWeaver、Agent Zero、LangGraph、Goose、Microsoft Agent Framework）和12个预先冻结的安全情境（含攻击与正常控制），共完成180个理论运行，53个可评估案例；

**📈 对比分析**

在同一合同输入和输出规范下与五个基线方法比较，GJD（联合诊断准确率）提升约12%（95% CI），policy‑outcome macro‑F1提升约18%，所需证据覆盖率提升约25%；诊断延迟仅5.9–12.3 ms，包体增量约30–113 KB，表明可直接嵌入代理评估流水线；

**⚠️ 局限性**

局限在于：诊断只覆盖被覆盖的执行边界，无法检测超出 instrumentation 范围的事件；合同需要手动实现覆盖层并同步版本；对缺失证据时会产生不确定性（unsupported claims）且未提供完整性保障；

---

## 262. SmolVLA-MoE: Sparse Mixture-of-Experts for Efficient Multi-Task Robotic Manipulation

**arXiv ID:** 2610.30732 | [PDF](https://arxiv.org/pdf/2610.30732v1)

**作者:** Shiyang Zhang `[一作]` (Huazhong Agricultural University), Peng Song `[通讯]` (Huazhong Agricultural University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 SmolVLA 框架下，将稀疏 Mixture‑of‑Experts（MoE）引入动作专家的 FFN 层，实现多任务机器人操控的高效模型。

**💡 创新点**

创新点在于：① 在 FFN 级别实现稀疏 MoE，避免整层复制；② 设计共享专家与可调权重 α，平衡预训练知识与专家专化；③ 采用 Sparse Upcycling 初始化与 Load‑Balancing 损失，提升专家利用均衡。

**🔧 技术方法**

核心技术包括：稀疏 MoE + token‑level router + scaling adapter、Flow Matching 用于动作生成、负载平衡正则化、Gaussian 噪声初始化。

**📊 数据集**

使用 LIBERO benchmark（四套任务：Spatial、Object、Goal、LIBERO‑10）进行评估。

**📈 对比分析**

与 SmolVLA、OpenVLA、Octo、Diffusion Policy、π_0 等基线对比，SmolVLA‑MoE 在 500M 激活参数下平均成功率 84.25%，显著高于 SmolVLA（82.75%）及 OpenVLA（76.5%），仅略低于 π_0（3.3B，86.0%），表明在同等计算预算下实现了更优性能。

**⚠️ 局限性**

局限性在于：实验仅在 LIBERO 训练/评估的模拟环境中进行，未验证在真实机器人或其他基准上的泛化；专家数目与共享专家权重的超参数仍需进一步自适应探索；层级化 MoE 或量化等更高级压缩方法尚未结合。

---

## 263. From Natural Language to Verified C: Verification-Guided Generation with ACSL Specifications

**arXiv ID:** 2610.30725 | [PDF](https://arxiv.org/pdf/2610.30725v1)

**作者:** Zian Wang `[一作]` (East China Normal University), Haibin Cai `[通讯]` (East China Normal University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 VeriGen-C，一个面向自动化形式化验证的框架，能够从自然语言需求同时生成可被 Frama‑C/WP 证明的 C 代码和 ACSL 规范。

**💡 创新点**

创新点在于：①联合设计实现与规范，利用“proof anchor”将代码状态与规范约束绑定；②在修复阶段基于 Verifier 反馈进行诊断并路由到规范修复或代码+规范共同修复；③加入需求对齐检查，防止仅通过验证得到的空洞规范被错误计为成功。

**🔧 技术方法**

使用技术包括：多阶段 LLM 调用（代码生成、循环注解、函数契约）、ACSL 规范化与预检、Frama‑C/WP 的最弱前置条件求解、诊断分类、路由决策以及基于需求拆分的对齐检查。

**📊 数据集**

实验数据集包含 HumanEval‑X、FM‑Bench 以及 LiveCodeBench Lite，分别共 164、70、201 条函数级任务。

**📈 对比分析**

与一键式生成、无知识引导与无诊断等基线对比，VeriGen-C 在 WP‑verified requirement‑aligned 成功率上分别提升了 35.7%、52.2% 与 31.1% 的百分点；同时在验证通过率和 shell‑only WP 率等方面也表现更好。

**⚠️ 局限性**

局限性包括：仅适用于单函数级 C/ACSL 任务，无法覆盖大规模并发、浮点或复杂堆操作的程序；验证成功率仍受 Frama‑C/WP 支持的前提、模型和超时等因素限制；需求对齐检查的准确性依赖于手工审核和策略设定。

---

## 264. When One Step Is Enough: An Empirical Study of Euler Integration in Flow-Matching Policies

**arXiv ID:** 2610.30735 | [PDF](https://arxiv.org/pdf/2610.30735v1)

**作者:** Shivakanth Sujit `[一作]` (Araya Inc.), Kai Arulkumaran `[通讯]` (Araya Inc.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估流匹配视听-动作模型（VLA）在单步与多步推理模式下的任务表现，证明在大规模已训练模型中单步推理即可达到与多步相同的成功率，进而实现低延迟部署。

**💡 创新点**

创新点在于首次提供跨六大规模 VLA、四大仿真基准与真实 Franka 任务的配对单步与多步推理对比实验，揭示单步推理对成功率几乎无损、显著提升频率，并明确指出模型规模、VLM 预训练与训练成熟度是实现单步等价的关键因素。

**🔧 技术方法**

采用 rectified flow matching 以及 Euler 一步/多步积分，利用 TOST、McNemar 等统计检验评估成功率、轨迹平滑度、多样性等指标；同时对模型参数规模、VLM 预训练状态等做 ablation 分析。

**📊 数据集**

使用 LIBERO、LIBERO+、CALVIN、RoboTwin 2.0、Frank Franka 机器人套件（spoon‑drawer、open‑drawer、open‑oven、close‑oven）以及 D3IL Avoiding 等多任务数据集。

**📈 对比分析**

通过匹配种子或配对实验比较单步与参考步推理，采用 TOST 进行等价性检验，结果显示大规模模型（≥1B）在多任务和长轨迹下单步推理成功率与多步相当，差距≤3.7pp，且单步可提升 1.6–4.8 倍的推理速度；小模型在长任务上略有优势。

**⚠️ 局限性**

局限性包括：并非所有模型在所有基准均有可用检查点；真实世界实验样本量有限，置信区间较宽；未对单步推理的多样性下降或 RL 梯度稳定性进行深入验证。

---

## 265. Gandalf: Flash Shall Not Die! Increasing Flash Storage Lifetime for KV Cache Offloading in LLM Serving

**arXiv ID:** 2610.30791 | [PDF](https://arxiv.org/pdf/2610.30791v1)

**作者:** Maharani A. P. Irawan `[一作]` (University of Chicago), Junchen Jiang `[通讯]` (TensorMesh Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于注意力层类型的 SSD KV 缓存放置策略（Gandalf），通过将短生命周期的线性注意力状态留在 DRAM，将高重用可能的全注意力 KV 缓存写入 SSD，来降低 SSD 写入量并延长 SSD 寿命。

**💡 创新点**

创新点在于：①首次将 LLM 的注意力层类型作为预测未来 KV 缓存重用的实用信号；②设计了容量感知的在线放置与淘汰决策，兼顾 SSD 耐久性与缓存命中率；③通过离线最优 Oracle 推导出可实现的在线评分机制，实现了低成本、可扩展的实现。

**🔧 技术方法**

使用的技术包括：离线最优 Oracle（基于未来重用和写入成本的最大化/最小化）、容量感知的放置门（考虑 DRAM/SSD 空间压力）、分层 KV 缓存管理、动态重用预测（基于历史访问和会话空闲时间）、以及模拟与实验平台（FEMU 与 LMCache 集成）。

**📊 数据集**

实验使用了三组真实工作负载轨迹：Qwen Bailian Trace A（多轮会话、工具调用等任务），Claude Code WEKA SemiAnalysis Trace（代理与子代理多轮对话），以及 AIPerf Trace（400k 请求、每会话 100 轮）。

**📈 对比分析**

与 LRU、Marconi、Workload‑aware、MeritKV、HotPrefix、LRU‑K 等基线相比，Gandalf 在 1TB SSD + 100GB DRAM 环境下平均提升 SSD 寿命 7.3 倍，同时保持甚至降低缓存命中率；在不同 DRAM/SSD 容量比例下，同样实现了 2–7.3 倍的寿命提升；在加载延迟与每请求成本方面，Gandalf 只略微增加延迟（约 0.1–0.2 秒），但因 SSD 写入成本降低，整体成本显著下降。

**⚠️ 局限性**

限制主要包括：①仅在单机 SSD 级别进行评估，未验证分布式或 HDD 环境；②仿真假设写放大因子 WAF=1，可能高估 SSD 寿命；③对高频写入模式的预测仍基于历史统计，可能对极端或新型工作负载适应性有限；④实现与真实生产系统的集成需进一步验证硬件兼容性与调优。

---

## 266. Agentic Knowledge-Model Co-Evolution via Consistency Regularization

**arXiv ID:** 2610.30790 | [PDF](https://arxiv.org/pdf/2610.30790v1)

**作者:** Sirui Chen `[一作]` (University of Illinois Urbana Champaign), Zheng Li `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种一致性正则化的知识-模型共演框架（SPLICE），通过在不同演化视角下寻找一致的效用提升来更新外部知识，并将这些知识通过GRPO和门控对比自蒸馏内部化到模型参数中。

**💡 创新点**

创新点在于：① 用“效用一致性”代替单一分析视角来评估知识更新的可靠性；② 在知识演化过程中构造负样本知识做对比蒸馏；③ 将知识演化与模型强化学习耦合，使知识与模型在交替迭代中共同进化。

**🔧 技术方法**

核心技术包括：多视角知识演化（perturb‑analysis）、效用一致性评分、Group Relative Policy Optimization (GRPO)、门控对比自蒸馏（CSD）以及 token‑level gating。

**📊 数据集**

实验数据集涵盖四个领域：LiveMathBench（数学推理）、LiveCodeBench（代码生成）、SpreadsheetBench（表格操作）和 WebShop（网页交互购物）。

**📈 对比分析**

与基线（知识演化基线 ReasoningBank/ACE/EvoSkill/SkillOpt 以及训练基线 GRPO/SkillRL/ReSkill）对比，SPLICE（Co‑Evo）在 Qwen3.5‑4B 上平均提升 35.2 分，Qwen3.5‑9B 上提升 33.9 分；在各子任务中均击败或与最强基线持平，且在 WebShop 等长序列任务上表现尤为突出。

**⚠️ 局限性**

局限性包括：① 需要多次文本视角变体，计算成本较高；② 依赖外部 LLM 进行知识优化，可能受限于该 LLM 的表达与推理能力；③ 仅在四个特定 benchmark 上验证，跨领域泛化和对更大模型的适配仍需进一步研究。

---

## 267. Invisibilization as a Source of Probability Bias in Language Models

**arXiv ID:** 2610.30785 | [PDF](https://arxiv.org/pdf/2610.30785v1)

**作者:** Christian Arnold `[一作]` (University of Birmingham), Tiago Ventura `[通讯]` (Georgetown University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在大型语言模型上进行两壶罐信任游戏实验，量化对受保护属性描述者信息的拒绝率与可信度的影响。

**💡 创新点**

创新点在于提出“可见性失踪偏差”概念，揭示对敏感属性的对齐导致信息被忽略而非单纯惩罚，并设计基于验证结果的机构评分机制缓解该偏差。

**🔧 技术方法**

采用对齐的LLM（OpenAI GPT‑4/ChatGPT、Anthropic Claude 系列）与自定义提示与贝叶斯推断结合的实验框架。

**📊 数据集**

使用自制的两壶罐信任游戏数据，共计 660 次单轮、120 次多轮对话，涵盖 22 种报告者描述（包括种族、性别、身份等）以及中英双语版本。

**📈 对比分析**

通过比较拒绝率、可信度估计与偏差量化，发现对“Black”描述的报告者拒绝率高达 46%，但若使用机构评分机制，黑色与白色报告者的可信度差距降至 0.02；同时大部分模型的平均性能在无评分时受限，评分后显著提升。

**⚠️ 局限性**

局限性包括仅在实验室设定下评估，缺乏真实世界任务验证；模型内部推理过程不可观测，且机构机制需外部反馈，实际部署时可能面临数据隐私与可解释性挑战。

---

## 268. MixJudge: Learning a Language-Queryable Critic for Vocal Mixing

**arXiv ID:** 2610.30783 | [PDF](https://arxiv.org/pdf/2610.30783v1)

**作者:** Kuan-Hsun Ho `[一作]` (Independent Researcher), Yi-Hsuan Yang `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了一个可用语言查询的音频评判器 MixJudge，利用合成的 11 种声乐混音问题及其严重程度来训练模型，使其能够判断混音中的具体问题并进行比较。

**💡 创新点**

创新点在于构建了面向声乐混音的多级问题课程，结合对问题名称、严重程度的精细标注，设计了关系融合网络与角色嵌入来显式表达声乐与背景音的相互关系，实现了基于文本查询的精准混音评判。

**🔧 技术方法**

采用了 CLAP 的音频和文本编码器、关系融合网络、可学习的角色嵌入以及多任务损失（对比损失、严重度排序损失、问题选择与相反问题损失）进行训练，利用对比学习与深度融合实现问题识别与严重度判别。

**📊 数据集**

训练数据来自 MoisesDB 和 MUSDB18‑HQ 的 370 首歌曲的 78,740 条合成状态-字幕对；评估数据包括 4,557 条合成样本和 23 首歌曲的 88 对原始/专业声乐样本。

**📈 对比分析**

与冻结的 CLAP、Music Flamingo、Audio Flamingo Next、Qwen2.5‑Omni 等基线在问题检索、相反问题判别、严重度排序以及菜单选择任务上进行对比；MixJudge 在所有指标上均明显优于基线，尤其在严重度排序和菜单选择的相关性与恢复率方面表现突出。

**⚠️ 局限性**

局限性包括仅关注声乐与背景的单声道混音问题，未覆盖立体声宽度、相位等维度；合成训练样本虽可控制问题，但对真实复杂混音的泛化仍有限；且仍需进一步扩展问题范围和实现交互式调音工作流。

---

## 269. Evaluation of Front-End Feature Matching for Lunar Visual SLAM

**arXiv ID:** 2610.30774 | [PDF](https://arxiv.org/pdf/2610.30774v1)

**作者:** Guillem Casadesus Vila `[一作]` (Stanford University), Grace Gao `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对十一个特征检测-匹配组合进行基准测试，评估其在九个合成、月球试验台和类地面数据集上的图像配对精度、视觉里程计误差和循环闭合能力。

**💡 创新点**

首次在月球车场景下单独对特征前端进行系统性比较，揭示合成结果并不能预测真实月球影像性能，并给出针对GPU和CPU的最优组合建议。

**🔧 技术方法**

经典检测器(SIFT, ORB, AKAZE, BRISK)、学习型检测器SuperPoint、匹配器FLANN、SuperGlue、LightGlue、KLT跟踪及光度重建、PnP与RANSAC。

**📊 数据集**

九个数据集：LuPNT（base, camera effects, high sun）、LuSNAR(5,9)、PolarTraverse、LunaPolaris、S3LI(Etna,Vulcano)，以及合成与真实影像混合。

**📈 对比分析**

通过Pose AUC@5°、EP、轨迹误差ATE/L、可追踪率等指标，比较不同组合的配对精度和可操作距离；SuperPoint+LightGlue在大基线下保持较高的Pose AUC和跟踪率，而传统组合在短基线下表现更好。

**⚠️ 局限性**

受限于仅评估单帧配对与基准里程计，未包含完整SLAM图优化；模型未针对不同光照和真实月球影像进行再训练；对低纹理区域循环闭合仍表现欠佳。

---

## 270. UniGuard: Multi-View Guardrail Learning for Cross-Interface Safety Generalization

**arXiv ID:** 2610.30742 | [PDF](https://arxiv.org/pdf/2610.30742v1)

**作者:** Viet-Thanh Pham `[一作]` (Monash University), Lizhen Qu `[通讯]` (Monash University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多视角守门机训练框架，提升LLM安全层在不同接口（提示、响应、对话、代理轨迹、多代理轨迹）下的泛化与置信度校准。

**💡 创新点**

创新点在于将同一安全案例的多种语义对齐视图视为联合学习样本，通过监督对比学习对齐同类视图，并使用跨视图一致性做非对称置信度校准，以实现接口不变的安全表示。

**🔧 技术方法**

采用预训练LLM编码器+注意力池化+线性分类器，配合监督对比损失和非对称置信度校准损失；数据经过合成多视角生成，训练集包含数千个视图。

**📊 数据集**

使用WildGuard、Aegis 2.0、BeaverTails、ToxicChat等公开安全数据作为基准，挑选1000条安全案例并为每条生成提示、响应、对话、代理轨迹和多代理轨迹五种合成视图，构成统一数据集。

**📈 对比分析**

在ATBench、PolicyGuardBench、DynaBench等未见接口基准上与基线Qwen3Guard‑Gen 0.6B、AgentDoG、PolicyGuard‑4B以及GPT‑5.2、Claude‑4.5‑Sonnet等对照实验，平均提升45.12 F1点、降低35.41 ECE，且保持与大型专有guardrail相当的性能。

**⚠️ 局限性**

局限性包括：仅关注接口泛化，未验证对全新安全领域、工具或政策的鲁棒性；数据为合成且规模有限，真实多代理/工具交互的复杂性可能被低估；仍存在误报/漏报风险，需与人工审核和系统层级约束结合使用。

---

## 271. Causal Analysis of Failures of Cyber-Physical Systems with Continuous Inputs

**arXiv ID:** 2610.30737 | [PDF](https://arxiv.org/pdf/2610.30737v1)

**作者:** Tom Yaacov `[一作]` (King's College London), Hana Chockler `[通讯]` (King's College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对自驾车等网络控制的网络物理系统（CPS）在失败时的解释问题，提出了基于实际因果理论的距离加权因果模型，并基于此设计了黑盒解释搜索算法；

**💡 创新点**

主要创新包括①将距离度量融入实际因果定义，提出距离加权实际因果与责任；②扩展至连续输入的CPS，提供更细粒度的因果分析；③基于距离加权责任的启发式搜索实现高效解释生成；

**🔧 技术方法**

采用实际因果理论、距离加权责任、模拟器LiteRacer与MetaDrive的黑盒神经控制器、采样近似方法、欧氏/Chamfer距离等技术；

**📊 数据集**

使用在LiteRacer（Sinusoidal、Segmented、Lane Shift）和MetaDrive（Curve、Straight、Mixed）多种轨迹与不同障碍物数量（10-14）的自驾车失败案例，共计约2000+样本；

**📈 对比分析**

通过比较离散责任（仅二值化障碍）和穷举搜索的模拟运行次数，实验显示距离加权责任显著减少所需模拟次数，尤其在障碍物多时差距更大；在多解释评估中，距离加权责任的覆盖率曲线AUC高于离散方法；

**⚠️ 局限性**

限制：仍依赖采样近似，结果受距离函数选择影响；对极大规模或高度交互的CPS可能需要更大样本或更复杂模型；对时间序列或多维动态特征的处理尚未完善。

---

## 272. Big World, Little Me: Comparing a First-Person VR Walkthrough Video and Static Slides for Communicating Height-Related Accessibility Barriers

**arXiv ID:** 2610.30731 | [PDF](https://arxiv.org/pdf/2610.30731v1)

**作者:** Chan In Sio `[一作]` (Hong Kong Polytechnic University), Lik-Hang Lee `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究利用Unity开发了名为《Big World, Little Me》的第一人称VR环境，并将其录制成10分52秒的走廊视频，再与对应的27张静态幻灯片进行比较，以评估两种呈现方式对叙事投入、状态共情与支持意愿的即时影响。

**💡 创新点**

创新之处在于首次将VR生成的第一人称叙事转化为可在非HMD设备上播放的记录视频，并系统比较其与静态幻灯片在情感投入与社会支持意向上的差异，为低资源情境下传播VR可访问性叙事提供可行路径。

**🔧 技术方法**

技术手段包括Unity构建VR场景、使用PC摄像头录制第一人称视角视频、生成对应的幻灯片、以及使用Busselle-Bilandzic叙事投入量表与Shen的状态共情量表对受试者进行测评。

**📊 数据集**

使用了100名15至48岁的参与者（50名观看视频，50名观看幻灯片）作为实验数据集。

**📈 对比分析**

采用非随机化的两组间实验设计，利用Welch t检验比较两组在叙事投入与共情上的差异，结果显示视频组在叙事投入（Hedges' g≈2.05）和状态共情（g≈1.91）上均显著优于幻灯片组，并且视频组支持意向显著提升。

**⚠️ 局限性**

局限性包括：分配非随机化，可能存在时间与招募偏倚；两种呈现方式差异多（运动、节奏、信息量、控制权等），难以归因；仅测量即时自我报告，未评估行为或长期影响；开放式回答样本不足，难以进行主题饱和分析。

---

## 273. Your Home Remembers: A Voice-Enabled Household Memory Assistant for Conversational Retrieval

**arXiv ID:** 2610.30895 | [PDF](https://arxiv.org/pdf/2610.30895v1)

**作者:** Aleix Sant `[一作]` (Telefónica Innovación Digital), Carlos Escolano `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款基于声音的智能家居助手，能够录制、检索并以生成式回答方式呈现家庭对话细节。

**💡 创新点**

创新点在于：①结合即时与长期语音记忆的检索增强生成；②面向Android的模块化编排架构；③基于说话人归一化的面向对话的转录与分段；④支持可视化的来源追踪。

**🔧 技术方法**

采用了 WhisperX/Whisper 语音识别、Silero VAD、LanceDB 向量存储、Sentence‑BERT/多模型嵌入、OpenAI/Llama 语言模型、GRU 触发词检测以及 Kotlin/Jetpack 前端。

**📊 数据集**

数据集：使用在实验室内采集的多语言（英语、西班牙语）家庭对话音频；未公开使用公开语料库。

**📈 对比分析**

该工作未给出定量对比；评估侧重于系统原型的功能可用性和响应时延，未与其他 RAG 系统进行实验比较。

**⚠️ 局限性**

局限包括：说话人标签不一致导致的归一化误差；对隐私与同意的技术与政策依赖；对低质量录音、噪声环境的鲁棒性不足；缺乏真实用户实验评估。

---

## 274. Working Through Representation: Malleable Workspaces for Human-Agent Knowledge Work

**arXiv ID:** 2610.30893 | [PDF](https://arxiv.org/pdf/2610.30893v1)

**作者:** Nan Chen `[一作]` (Microsoft Research Asia), Yuqing Yang `[通讯]` (Microsoft Research Asia)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了“可塑工作空间（Malleable Workspace）”概念模型，并通过分析58篇近期人机协作系统论文、构建可塑空间（MalleableSpace）原型以及三个月的实地案例研究，对该模型在知识工作中的应用与演进进行了验证与迭代。

**💡 创新点**

创新点包括：①首次系统化地将工作表示（Work Representation）与人机行动、工作环境之间的三向关系（R1‑R3）纳入模型；②提出工作单元与关系的可塑性作为核心，阐释其在持续协作中的作用；③将模型落地为空间式可塑空间原型，并通过长期实地研究揭示其对用户组织习惯和外部环境连接的影响。

**🔧 技术方法**

技术手段主要包括：基于大型语言模型（LLM）的生成式AI；可视化空间Canvas与层级框架（Frame）实现可编辑的工作表示；人机交互中的上下文组装与agent-directed acquisition；以及通过Python、Electron等实现的桌面原型与内部agent。

**📊 数据集**

数据集：①对58篇CHi/UICC等会议论文进行系统分析（包含58篇论文、61个实现案例）；②三个月实地研究中7名经验丰富的agent用户的使用日志、访谈文本、群组讨论记录与空间截图；未使用公开机器学习数据集。

**📈 对比分析**

比较方法：对58篇论文按模型构成（单位、关系、状态、进展）进行编码与对比，识别四类核心功能（F1‑F4）；在可塑空间原型上进行纵向案例研究，记录用户的空间创建数、节点数、使用场景等指标；并结合定性访谈评估其对工作组织、协作与环境连接的影响。由于研究侧重于设计与使用体验，未给出传统性能指标（如准确率、召回率）。

**⚠️ 局限性**

局限性：①参与者规模仅7人，工作场景与行业相对单一；②缺乏对比实验验证模型相较于传统聊天或静态可视化工具的效果提升；③可塑空间的功能与与外部工具/agent的连接依赖手动配置，仍未实现完全自动化的跨系统同步；④模型对多用户、跨时间分布式协作的适用性尚待进一步验证。

---

## 275. MOER: Mix-Policy Optimization under Expert Prefix Guidance with Selective Rollback

**arXiv ID:** 2610.30873 | [PDF](https://arxiv.org/pdf/2610.30873v1)

**作者:** Xuyang Zhong `[一作]` (City University of Hong Kong), Chen Liu `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MOER，一种混合策略优化框架，结合逐步衰减的专家前缀引导、可选择回滚机制以及对 on‑policy 与 off‑policy 词元采用不同损失的统一目标。

**💡 创新点**

创新点包括：① 采用余弦退火调度逐渐减少专家前缀长度，鼓励模型自我探索；② 设计成本无损的回滚机制，在生成结果错误时自动回退到完整专家轨迹，保证离线轨迹的正确性；③ 将完成词元视为 off‑policy 词元，并使用信任域 SFT 损失与剪枝后的 GRPO 损失相结合，显著提升训练稳定性。

**🔧 技术方法**

技术实现：混合策略强化学习（GRPO）、prefix‑guided 生成、余弦退火前缀衰减、可选择回滚、信任域 SFT、异步剪枝、对齐策略与奖励设计。

**📊 数据集**

使用 OpenR1‑Math‑46k‑8192 作为训练集（专家轨迹来自 DeepSeek‑R1），在多项数学与通用推理基准上评测：AIME 2024/25、AMC、MATH‑500、Minerva、OlympiadBench、ARC‑c、GPQA‑diamond、MMLU‑Pro。

**📈 对比分析**

与 SFT、RL、RL+SFT、SFT‑then‑RL 以及混合策略基线 LUFFY、ReLIFT、CHORD、TRAPO、Prefix‑RFT 进行对比。MOER 在 Qwen2.5‑Math‑7B 上平均获得 49.1（数学）和 59.0（通用）分，均超过所有基线；在不同模型规模和基础模型类别（Math‑专属与 Instruction‑通用）下亦保持领先或竞争优势。

**⚠️ 局限性**

局限性：① 对小规模模型（如 1.5B）提升有限，受参数容量约束；② 依赖于准确的错误验证和回滚机制，若验证失效可能导致训练不稳定；③ 主要验证于需要链式推理的任务，对非 CoT 任务的适用性未充分探究。

---

## 276. CAVC: Convergence-Adaptive Audio-Visual Cache for Modality-Asymmetric Diffusion Generation

**arXiv ID:** 2610.30854 | [PDF](https://arxiv.org/pdf/2610.30854v1)

**作者:** Lifan Jiang `[一作]` (Zhejiang University), Boxi Wu `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种训练无关的音视频生成加速框架CAVC，利用模态可恢复性将音频计算缓存并在中间阶段跳过音视频交叉注意，最终恢复少量联合去噪以保持同步。

**💡 创新点**

创新点在于发现音频侧的漂移可通过终端少数步联合去噪恢复，而视频侧则难以恢复，从而实现模态不对称的缓存与计算分配；并采用自适应触发阈值动态决定缓存时机。

**🔧 技术方法**

核心技术包括模态异步缓存（Convergence‑Adaptive Audio‑Visual Cache）、中间阶段跳过跨模态注意、终端轻量级联合去噪、在线音频收敛检测与自适应触发。

**📊 数据集**

在AVGen‑Bench、VABench T2AV和General‑80三大公开音视频生成基准上进行评估。

**📈 对比分析**

与完整双模联合采样、Cache‑DiT、TeaCache等基线比较，CAVC在不降低音频质量和同步度的前提下实现1.10–1.35×的壁钟时间加速，并提升多项视觉动态与时序指标。

**⚠️ 局限性**

局限性在于速度提升受限于保持完整视频分支；对持续变化的音频事件触发缓存时机被推迟，导致加速有限；终端恢复步数有限时仍可能出现轻微的同步偏差。

---

## 277. Bifurcations from Weight Space: JEPAWG on Parametric Dynamical Systems

**arXiv ID:** 2610.30847 | [PDF](https://arxiv.org/pdf/2610.30847v1)

**作者:** Tobias Göbel `[一作]`, Miranda C. N. Cheng `[通讯]` (Academia Sinica)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了四个参量动力系统（Lorenz、Rössler、Brusselator、Hénon）的模型动物园，对每个参数点训练并微调了预测网络，随后用 JEPAWG 超网络生成未见参数下的权重，并利用权重对参数的导数对系统的分岔结构进行诊断。

**💡 创新点**

创新点在于：①将 JEPAWG（联合嵌入预测网络）用于权重空间分析，使得从权重导数即可定位局部分岔（Hopf、周期倍化）以及无迹分岔（Bykov T‑点）；②展示权重空间响应可近似估计 Feigenbaum 常数；③证明 JEPAWG 在预测任务上可与端到端超网络竞争，并且在权重空间提供可解释性。

**🔧 技术方法**

技术方法包括：TCNN‑CD 序列预测网络；JEPAWG 架构（权重编码器、参数编码器、潜在预测器、解码器）；JEPA/VICReg 损失训练两阶段；Sobol 采样构建参数网格；对解码权重做中心差分求导得到 G(p)；对权重空间进行 PCA 可视化。

**📊 数据集**

数据集是四个合成动力系统的轨迹：Lorenz、Rössler、Brusselator 的连续时间轨迹（RK4 采样），以及 Hénon 图的离散时间轨迹；每个系统在不同参数下生成多条初始条件轨迹，并对状态做标准化。

**📈 对比分析**

与 PHLieNet、AE、VAE 基线在插值和外推下的短期跟踪误差（NRMSE/ TtT0.2）进行比较。JEPAWG 在插值上与 PHLieNet 基线相当，往往更优；在外推时表现取决于动力学性质，限界上可保持更长的预测时间；权重空间导数准确定位已知分岔点，误差在几百分之以内。

**⚠️ 局限性**

局限性包括：①只能检测训练数据中出现的、影响典型轨迹的分岔，无法识别非吸引子分岔；②训练参数之外的权重不受约束，外推性能随动力学而异；③G(p) 的分辨率受参数网格间距和解码平滑度限制；④需要为每个参数点训练完整网络，计算成本高。

---

## 278. Loop to Look Better: Learning Adaptive Depth for Question-Guided Perception in Vision-Language Models

**arXiv ID:** 2610.30824 | [PDF](https://arxiv.org/pdf/2610.30824v1)

**作者:** Wencheng Ye `[一作]` (Tongji University), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LoopLook，一种在视觉编码器内部循环执行的感知框架，利用问题引导递归更新视觉表示，并通过自适应停止与 token‑wise Mixture‑of‑Looks 进一步融合多步信息。

**💡 创新点**

创新点：①把任务导向的调制与共享权重的循环窗口结合，实现轻量级的可扩展视觉深度；②引入自学习退出门与 token‑wise 递归路由，既能自适应分配计算资源，又能为每个视觉 token 选取最合适的循环状态；③保持语言模型冻结，仅在视觉侧微调 LoRA 参数，显著降低训练成本。

**🔧 技术方法**

技术手段：LoRA 轻量化微调、共享权重的 Transformer 循环窗口、任务条件下的 Affine 调制、可学习的退出门、Mixture‑of‑Looks 路由、基于 CKA 的块划分、以及自监督视觉‑语言对齐投影。

**📊 数据集**

使用的数据集包括 MMStar、VMCBench、AI2D、V^∗、BabyVision、VisualPuzzles、POPE、HallusionBench，此外在训练阶段还使用 100k 条视觉 CoT 采样。

**📈 对比分析**

与基线模型（四种主干：Qwen2.5‑VL‑3B、Qwen3‑VL‑4B、InternVL3.5‑4B、LLaVA‑1.5‑7B）以及现有视觉侧增强方法（Dense Connector、TGIF、DeepStack）进行对比；LoopLook 在所有主干上平均提升 4–8% 以上，单一任务可达 8.6%；在 LLaVA‑1.5‑7B 上平均得分提升 3.3%，并在 5 个基准中表现最佳；TTFT 仅升高 5–6%，推理时间基本不变。

**⚠️ 局限性**

局限性：①依赖在视觉编码器中选择合适的中后期窗口，跨块窗口效果不佳；②目前只在四种主干和八个基准上验证，未知对更大模型或不同任务的通用性；③循环层虽轻量，但在极低分辨率或长图像时仍需更多迭代；④未探索与自我推理（Reasoning‑Side）同步迭代的更紧密耦合方式。

---

## 279. Robot Motion Transition with Human Motion Prior

**arXiv ID:** 2610.30817 | [PDF](https://arxiv.org/pdf/2610.30817v1)

**作者:** Taehyun Byun `[一作]` (Korea University), Sungjoon Choi `[通讯]` (Korea University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种机器人-人-机器人框架STAMP，用以将独立获取的人形机器人动作通过学习的空间与时间对齐以及人类运动先验生成连续运动序列并训练追踪策略。

**💡 创新点**

核心创新在于：① 通过学习的对齐模型在人类运动空间中预测最佳相对位置、朝向与过渡时长；② 在人类先验下生成过渡运动，避免人工插值或强制返回中性姿态；③ 将对齐与过渡模块统一在同一人类运动空间内，可跨机器人实现迁移。

**🔧 技术方法**

使用的技术包括：Transformer编码器用于对齐模型，基于Kimodo的扩散式运动中间生成器，CoRe式双向机器人-人类运动重定向，以及BeyondMimic追踪策略训练。

**📊 数据集**

主要数据集为BONES‑SEED（人类动作数据）以及通过Kimodo生成的两台机器人（Unitree G1、ROBOTIS K1）的独立运动对。

**📈 对比分析**

与根/朝向插值、MotionBricks以及Kimodo未做对齐的基线进行比较。STAMP在18对动作中在两台机器人上分别实现了最高成功率（Unitree 80.69%，ROBOTIS 87.52%），体速度误差与关节加速度均显著降低。

**⚠️ 局限性**

局限性包括：对齐成本仅为运动学指引，实际跟踪效果仍受重定向质量与机器人动力学限制；当前为每个序列训练单独追踪策略，未来需在共享强化学习框架下实现更少的序列特定训练。

---

## 280. Adaptive Aerial Loco-Manipulation Policy Learning via LLM-Assisted Proactive Task Generation

**arXiv ID:** 2610.30810 | [PDF](https://arxiv.org/pdf/2610.30810v1)

**作者:** Jingyu Ru `[一作]` (Northeastern University), Yang Yu `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 ALMA 系统，用一种基于上下文的飞行策略配合行动条件的预测监督和 LLM 辅助任务生成，实现在有外部机械臂命令的情况下自适应地控制空中操作器。

**💡 创新点**

创新点包括：① 通过预测未来测量表示来监督上下文，使其保留对飞行控制至关重要的历史响应信息；② 利用大型语言模型分析训练日志，主动生成新的机械臂运动任务，提升在未见运动条件下的泛化；③ 将预测监督与自适应任务生成整合到同一训练框架，提升控制鲁棒性。

**🔧 技术方法**

采用的技术包括：强化学习（PPO）训练上下文飞行策略；GRU‑CNN‑MLP 架构实现行动条件的预测；自回归预测目标与 EMA 目标编码器；LLM（GPT‑5.6 Sol）驱动的任务生成与评估；Isaac Sim 与 MuJoCo 进行仿真与硬件‑in‑the‑loop；在真实硬件上部署固定网络参数。

**📊 数据集**

实验数据集主要来自：① NVIDIA Isaac Sim 生成的随机机动与负载扰动仿真；② MuJoCo 进行硬件‑in‑the‑loop；③ 实际空中操作器的运动捕捉数据，包含多种机械臂姿态与周期性运动。

**📈 对比分析**

与 PID、PID‑INDI、PID+ℒ₁、NMPC 以及 Zhang 等基线进行对比。实验表明：在静态与周期性机械臂命令下，ALMA 的位置 RMSE 低于 NMPC 24.5%–24.7%，在真实世界的悬停实验中相较于 PID 降低 79.6% 的位置误差；在 OOD 机械臂运动下，LLM 生成任务进一步提升了 22.8% 的误差减小。

**⚠️ 局限性**

局限性：① 方向（偏航）误差仍未达到最佳水平；② 对接触力与约束的感知与利用仍未实现；③ 需要 LLM 与大量仿真资源，部署时对计算负担有限；④ 仅在固定机械臂姿态与周期性运动上验证，尚未系统评估更复杂的动态交互场景。

---

## 281. An integer determinant approach to the length of Type II codes and the weights of shadows

**arXiv ID:** 2610.30809 | [PDF](https://arxiv.org/pdf/2610.30809v1)

**作者:** Jae-Hyun Baek `[一作]` (Sogang University), Jon-Lark Kim `[通讯]` (Sogang University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文提供了三种全新、简洁的元素证明，阐明了二进制Type II码长度必须为8的倍数，并进一步证明其影子向量的权重同余性质；

**💡 创新点**

创新点在于利用整数格子行列式的模8同余、Gauss和的Arf不变量以及MacWilliams恒等式的一点取值，全部摆脱了传统的Gleason多项式与不变理论手段；

**🔧 技术方法**

主要技术包括整数Gram矩阵的Leibniz展开与模8约简、二次型与Gauss和的分析、以及对MacWilliams身份的单点评价；

**📊 数据集**

论文不依赖外部数据集，全部为理论推导与符号计算；

**📈 对比分析**

相较于传统Gleason或Witt–Arf等方法，新的证明更短、直观且易于教学，性能上可视为在证明复杂度与可读性上的提升；

**⚠️ 局限性**

局限性在于仍需较深的线性代数与数论基础，且对更广泛码类（非二进制、非自对偶码）的推广尚未给出完整框架。

---

## 282. uGDS: A User-Space Development Kit for High-Performance GPU Direct Storage

**arXiv ID:** 2610.30868 | [PDF](https://arxiv.org/pdf/2610.30868v1)

**作者:** Jing Wei `[一作]` (ScaleX-IO), Jian Zhang `[通讯]` (ScaleX-IO)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了一个开源的用户空间GPU Direct Storage（uGDS）库，提供同步、批量和流序异步的NVMe SSD到GPU内存的数据传输接口；

**💡 创新点**

创新点包括：1）完全在用户空间完成NVMe命令生命周期，消除每次I/O的内核调用；2）统一CUDA和HIP GPU内存映射，支持跨供应商；3）按工作负载划分硬件队列（同步/批量）并实现工作量感知调度；4）提供可观测的调试与性能诊断；5）实现高效的PRP管理和一次性门铃合并；

**🔧 技术方法**

技术手段包括：用户空间NVMe控制、PCIe Peer‑to‑Peer DMA、巨页映射、PRP列表池、一次性门铃写、可选MSI‑X中断、GPU内存注册与DMA‑buf导出、并发队列管理与计数、硬件时间戳与错误处理。

**📊 数据集**

主要评估数据来自：A100 GPU、Samsung 990 PRO SSD、AMD ROCm GPU以及LLM缓存应用LMCache（Qwen3‑0.6B模型）所产生的KV‑cache读写负载；未使用公开大规模数据集。

**📈 对比分析**

与NVIDIA GDS、hipFile等闭源/半开放库对比，uGDS在同步I/O中可提升4–12倍带宽、延迟下降至1/5；批量I/O中可提升2–30倍带宽；在LMCache中TTFT降低62.7%，推理吞吐提升2.63倍；整体性能远超现有实现。

**⚠️ 局限性**

局限性包括：目前仅支持单本地PCIe NVMe SSD；NVMe‑over‑Fabrics、多SSD聚合和跨主机传输尚未实现；队列调度仍为简单轮询，缺乏动态负载感知；对大容量内存映射的支持受限于GPU和CPU的物理地址空间；在极端高并发下仍可能出现队列饱和导致等待。

---

## 283. Silicon Samples Match the Answers but Not the Respondents

**arXiv ID:** 2610.30900 | [PDF](https://arxiv.org/pdf/2610.30900v1)

**作者:** Yusuke Kumagae `[一作]` (Hakuhodo DY Holdings Inc.), Masahiro Kaneko `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估大型语言模型在硅采样生成合成人群时对受访者分布的覆盖程度，引入受访者覆盖缺口(RCG)和离群比率(OSR)指标。

**💡 创新点**

提出RCG/OSR指标，在受访者层面衡量合成样本对真实人群的覆盖与偏差，揭示现有按问题级别指标无法捕捉的生成失效模式。

**🔧 技术方法**

使用Gower距离测算答案向量距离，基于LLM条件生成、全历史/局部提示、温度/推理努力等实验设置；对比基线Marginals、Copula、LCA。

**📊 数据集**

美国WVS 7期与2024年ANES调查的100道题目回答数据，包含性别、年龄等属性。

**📈 对比分析**

通过计算RCG与OSR的95%分位比值，评估不同模型和基线的覆盖；所有LLM的RCG>1，均低于Copula；即使进行边际校准，LLM仍未达到人类水平。

**⚠️ 局限性**

仅针对美国两份调查，受访者属性和问题类型有限；未覆盖多语言、非美国样本；未评估更高级提示或训练方法。

---

## 284. Tracing Subliminal Learning Through Pretraining

**arXiv ID:** 2610.30884 | [PDF](https://arxiv.org/pdf/2610.30884v1)

**作者:** Angela Tang `[一作]` (Independent), Gonçalo Paulo `[通讯]` (EleutherAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在预训练基础模型中，黑标与软标蒸馏过程中出现的潜在学习（subliminal learning）现象，探讨了不同初始化、预训练数据顺序、语料库、模型规模与架构下的属性传递行为。

**💡 创新点**

证明潜在学习不依赖于相同初始化或预训练条件，且教师的“传递能力”和学生的“接受能力”是分离发展的；此外发现数值频率与序列结构对传递的影响各异。

**🔧 技术方法**

使用Pythia、PolyPythia、OLMo等预训练模型，采用硬标签蒸馏（数值序列）、软标签蒸馏（on‑policy GKD）、Top‑K 蒸馏，评估软硬标签的逆KL与前向KL损失，并利用SGD与Adam优化器进行实验。

**📊 数据集**

以动物偏好为特征的人工构造提示与合成文本作为教师微调数据；使用Pile文本生成学生自续写样本；生成的数值序列和合成序列作为蒸馏训练数据。

**📈 对比分析**

通过比较教师-学生对的偏好概率提升，计算相对提升率和概率差异；结果显示无论是硬标还是软标蒸馏，跨模型传递均能显著提升；在软标蒸馏中，仅需数百个样本即可获得约90%的传递效果，且在不同模型规模与架构间均可观察到传递。

**⚠️ 局限性**

实验仅针对动物偏好等有限特征，且主要使用小规模预训练模型；在大规模或指令微调模型中机制与强度可能不同；同时软标蒸馏对训练时间和语言建模质量有影响，过度训练可能导致语言性能下降。

---

## 285. LocoManiLab: A Unified Testbed for Data, Policies, Controllers, and Deployment in Humanoid Loco-Manipulation

**arXiv ID:** 2610.30876 | [PDF](https://arxiv.org/pdf/2610.30876v1)

**作者:** Dewei Wang `[一作]` (Institute of Artificial Intelligence (TeleAI), China Telecom), Xuelong Li `[通讯]` (Institute of Artificial Intelligence (TeleAI), China Telecom)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个统一的仿真与真实世界评测平台，用于评估多种策略与多种全身控制器在不同家庭场景下的定位、导航与物体操作能力。

**💡 创新点**

创新点包括：① 将任务定义、演示采集、策略训练、控制执行和评测流程统一到一个工作流；② 设计了包含234个原子与组合任务的任务集；③ 在同一测试床下评估12种策略（VLA、WAM、模仿学习）和5种全身控制器；④ 引入基于扩散模型的运动生成（Kimodo）并细化到任务级别；⑤ 通过数据集混合与增强来系统探究演示规模和来源对性能的影响。

**🔧 技术方法**

技术方法：视觉语言动作模型、世界动作模型、扩散运动生成（Kimodo）与LoRA/全微调、PPO优化、轨迹增强、A*导航、全身控制器（AMO、SONIC、HEFT、ALMI、ScaleBFM）以及多任务与域随机化评估。

**📊 数据集**

使用的数据集：51.5小时的仿真手动演示（5种控制器），约9小时真实机器人演示，51.5小时合成演示，轨迹增强数据，总计234个任务覆盖厨房、客厅、卧室、浴室与办公室等13个实例。

**📈 对比分析**

比较方法：在相同的任务列表上评估任务完成率与运动质量（稳定性、能耗、轨迹光滑度），同时在见过/未见过的任务、场景与视觉扰动下对比。结果显示：熟悉任务上的排名在仿真与真实环境中一致，但在未见任务/场景时整体成功率显著下降，控制器选择对能耗和稳定性有显著影响。

**⚠️ 局限性**

局限性：① 任务仅涵盖室内家庭与办公室场景，未覆盖更复杂或多样化的环境；② 真实机器人实验仅限3个任务、10次试验，数据量有限；③ 对于计算资源受限的机器人缺少推理延迟与通信瓶颈的评估；④ 仅使用单一仿真引擎（MuJoCo），未检验跨仿真平台的泛化。

---

## 286. HimmBO: High-Dimensional Interactive Model-Merging Bayesian Optimization for Image Generation

**arXiv ID:** 2610.30875 | [PDF](https://arxiv.org/pdf/2610.30875v1)

**作者:** Jinyuan Mao `[一作]` (Zhejiang University), Chenxi Liu `[通讯]` (University of Victoria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于用户偏好的两阶段搜索框架HimmBO，用来在数千个LoRA适配器库中高效探索并合并适配器，直接输出满足用户期望的图像；

**💡 创新点**

创新点在于：①使用级别集估计（LSE）在预生成的图片集合中完成筛选，避免每轮生成图像；②通过风格内核将视觉相似的适配器信息共享，显著提升搜索效率；③采用两阶段结构（筛选+精炼），实现库规模可扩展且生成成本低；

**🔧 技术方法**

主要技术包括：偏好驱动高斯过程代理、水平集估计（LSE）、风格映射与内核、贝叶斯优化（BO）与多样性DPP、预生成图片画廊、线性适配器融合；

**📊 数据集**

使用自行构建的SDXL-Style-1K（1080个LoRA风格适配器）库，及其预生成的4个主题样例图片；

**📈 对比分析**

与GimmBO、Stylus检索、Stylus+GimmBO、随机坐标下降等方法对比，HimmBO在相同交互轮次下获得最高图像相似度（如0.889在1080适配器下），同时在相同推理次数下只需6.8%图像生成量，显示出更高效率与更好的可扩展性；

**⚠️ 局限性**

局限性包括：仅使用线性融合；实验基于模拟用户，真实用户噪声与偏好漂移尚未充分验证；LSE采用简化形式，未完全利用多轮置信区间；风格映射未完全分离内容与风格；筛选阶段一次性完成，若误删适配器后无法恢复；未来需引入更鲁棒的偏好模型、全局搜索回退机制、内容风格分离等改进。

---

## 287. Synthetic-to-Real Code-Switched ASR: A Spontaneous Nepali-English Podcast Benchmark

**arXiv ID:** 2610.30860 | [PDF](https://arxiv.org/pdf/2610.30860v1)

**作者:** Firoj Paudel `[一作]` `[通讯]` (Wiseyak Inc.), Firoj Paudel (Wiseyak Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Nepali–English 代码切换语音识别任务中，先在 0.95 小时的合成语音上进行监督微调，然后使用 GRPO 进行序列级策略优化，以提升模型在 13,044 句自发播客上的识别性能。

**💡 创新点**

创新点在于证明仅用极少量合成语音即可实现跨域零样本迁移，并且发现 GRPO 在已收敛的 encoder–decoder 结构中对脚本奖励无显著增益。

**🔧 技术方法**

采用 FastConformer 多语种基础模型，冻结编码器，只微调解码器，并结合 Group Relative Policy Optimization（GRPO）与脚本/MER 奖励。

**📊 数据集**

使用的主要数据集是新构建的 13,044 句 Nepali–English 代码切换播客（10,214 测试 / 2,830 开发），以及 0.95 小时的合成语音作为训练集。

**📈 对比分析**

与零样本基线对比，监督微调将混合错误率从 23.40 % 降至 21.52 %（-1.88），GRPO 进一步略降到 21.49 % 并将切换点错误率从 35.89 % 降至 28.70 %（-7.19），提升显著且统计显著。

**⚠️ 局限性**

局限性包括 RL 细微提升、仅 0.95 小时合成训练量、参考注释可能存在偏差、对纯英语识别略有负面影响，以及开发/测试集主持人角色分布差异等。

---

## 288. Joules-to-GDP: How Much Economic Value Do AI Systems Produce per Unit of Energy?

**arXiv ID:** 2610.30858 | [PDF](https://arxiv.org/pdf/2610.30858v1)

**作者:** Amirreza Zeinali `[一作]` (Stanford University), Azalia Mirhoseini `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出将AI系统产生的专业工作经济价值与其消耗的能量关联，构建了Net Value per Joule（NVJ）和Time Saving per Joule（TSJ）两种度量指标。

**💡 创新点**

创新点在于首次将经济价值（基于专家完成时间与行业中位工资）与全流程能耗（包括模型、硬件、工具调用与等待）结合，形成可比较的“能耗回报”框架，并通过GDP加权任务样本展示不同职业、模型与加速器的能效差异。

**🔧 技术方法**

采用的技术包括：Gemini 3.1 Flash‑Lite作为无偏裁判、OpenHands推理框架、vLLM推理引擎、IPW（Intelligence per Watt）能耗计量、以及多种GPU/加速器（H200、B200、H100、MI350X）上的实验部署。

**📊 数据集**

使用了GDPval公开专业任务数据集，涵盖220个任务、44个职业、9个行业，任务类型包括文档、表格、演示、代码与媒体，且配有专家参考与完成时间记录。

**📈 对比分析**

比较方法通过在同一任务集、同一裁判与同一计时条件下，测量不同模型（GLM 5.2、Gemma 4‑31B、GPT‑OSS 120B等）和不同加速器配置的NVJ、TSJ、总输出价值与总时间节省；结果显示GLM 5.2产出最高但能耗回报中等，Gemma 4‑31B在NVJ/TSJ上遥遥领先，H200/ B200在单机配置下能效最好，八块H100提升每小时回报但能耗回报下降。

**⚠️ 局限性**

主要局限包括：评估范围仅限GDPval任务，未涵盖更广泛的行业与任务类型；能耗测量仅覆盖GPU/CPU/DRAM功耗，未计入设施开销和网络传输；模型与加速器配置受硬件可用性限制，可能不具备普适性；经济价值估计基于专家时间与中位工资的代理，可能与实际GDP贡献不完全一致。

---

## 289. ADAPT for Science: A Multi-Agent System for Self-Evolving Scientific Optimization

**arXiv ID:** 2610.30851 | [PDF](https://arxiv.org/pdf/2610.30851v1)

**作者:** Zhaoran Feng `[一作]` (Nanjing University), Kairui Feng `[通讯]` (Shanghai Innovation Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 ADAPT for Science 多代理框架，利用历史设计经验和在线执行反馈，联合自适应初始化与优化策略，在重力波探测器设计任务中实现高效搜索。

**💡 创新点**

创新点在于将完整的科学搜索策略（选择起点与搜索路径）视为优化目标，并分解为“在哪里搜索”和“如何搜索”两层决策，结合目标感知检索和程序演化，实现搜索过程本身的自适应学习。

**🔧 技术方法**

采用结构感知检索、GPT‑6 Astra 与 Claude‑fable‑5.1 语言模型驱动的演化器（Selection Evolver、Optimizer Evolver）、差分物理模拟器、可重用历史设计库、在线反馈循环和独立验证器等技术。

**📊 数据集**

使用包含 15,409 条历史重力波探测器拓扑–设计记录的数据集进行经验库构建，并以八个未见拓扑作为 held‑out 评估集。

**📈 对比分析**

在相同的科学评估预算下，与传统经典优化器（Adam、NAAdam 等）和通用代理进化基线（GEPA、OpenEvolve 等）比较，ADAPT 的平均损失为 0.204±0.101，分别比最佳经典优化器低 54.9%、比最佳通用基线低 62.2%，并在所有八个测试拓扑上获得最低损失。

**⚠️ 局限性**

局限性包括对历史设计库质量与覆盖度的依赖；对远离历史分布的新目标可能表现下降；演化过程相对耗时，适用于评估成本高的场景；目前仅在模拟环境验证，缺乏真实实验的验证。

---

## 290. TEAR: Task-Evidence Authority Routing for Robotic Manipulation

**arXiv ID:** 2610.30819 | [PDF](https://arxiv.org/pdf/2610.30819v1)

**作者:** Shicheng Zhang `[一作]` (City University of Hong Kong), Haoliang Li `[通讯]` (City University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 TEAR 框架，实现了机器人侧在人机远程操控与本地视觉语言动作（VLA）控制之间的权威路由，以在网络波动和任务不确定性下保持高成功率。

**💡 创新点**

创新点在于将网络状态证据（延迟、抖动、丢包）与基于冻结 VLM 的视觉任务语义证据（进度、停滞、参与度、异常）相结合，通过因果 TCN 预测交接事件，并用带有门限和冷却的状态门实现稳健的权威切换，完全在机器人端实现无在线模型查询。

**🔧 技术方法**

采用的技术包括：CLIP/SigLIP 视觉语言模型用于任务语义评分；对网络延迟、抖动、丢包的实时测量；因果时间卷积网络（TCN）用于事件预测；状态门控策略实现权威切换；以及基于离线示范的文本对生成与缓存的文本嵌入。

**📊 数据集**

数据集主要由两类任务演示组成：Isaac Sim 中的 SO‑101 把橙子搬到托盘的模拟演示；以及真实机器人 SO‑101 在实验室场景下把黄色鸭子放到托盘的演示。网络条件通过人工设定的时间变化延迟、抖动、丢包来模拟；视觉任务证据基于 CLIP ViT‑B/32 的特征提取。

**📈 对比分析**

比较方法：在同一网络扰动下，对三种策略（Always‑Teleop、Always‑VLA、TEAR）进行十个有效试验，记录任务成功率、远程操控使用比例和完成时间。TEAR 在模拟和真实环境中均达约 90% 成功率，仅使用 19.8%–27.4% 的远程操控，比 Always‑VLA 提高 30–40% 成功率，且远程操控时间比 Always‑Teleop 减少约 80%；但 TEAR 的成功完成时间比两种单一控制方式略长。

**⚠️ 局限性**

局限性包括：仅评估两种相似的抓取‑放置任务，未覆盖更广泛的操作场景；网络扰动仅为人工设定的延迟/丢包曲线，未考虑拥塞、重传等真实网络特性；任务证据仅基于视觉，缺乏接触力、位置等更细粒度的感知信息；因此在更复杂、多模态任务及实际网络环境下的泛化性能仍需进一步验证。

---

## 291. DRILL: Training World Models to Improve Policies, Not Predict Pixels

**arXiv ID:** 2610.30814 | [PDF](https://arxiv.org/pdf/2610.30814v1)

**作者:** Zhaolu Kang `[一作]` (Hunyuan, Tencent), Zhichao Hu `[通讯]` (Hunyuan, Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Downstream-Return Imagination Learning Loop (DRILL) 框架，用于在 VLA 任务中通过世界模型的下游回报来训练视频世界模型，并给出了两种实现：轻量级的 DRILL‑VWI（访客加权重）和更精确的 DRILL‑IMG（隐式元梯度）

**💡 创新点**

核心创新在于：① 识别并解决了 VLA 任务中世界模型从“像素预测”到“策略提升”目标的误匹配；② 设计了双层优化框架，直接将下游策略回报作为世界模型的训练目标；③ 提出可观测的 Policy‑Utility Residual 诊断指标；④ 通过 VWI 近似元梯度以及完整的 DRILL‑IMG 计算，阐明两种方法的理论关系与适用场景

**🔧 技术方法**

使用的视频世界模型、GRPO（Group Relative Policy Optimization）策略优化、双层优化、截断 Hessian‑vector 乘积进行隐式元梯度估计、基于访客、优势、模型置信度加权的损失、RLVR‑World 预测校准的预训练方法

**📊 数据集**

在五个机器人操纵仿真器上评估：LIBERO‑Long、CALVIN ABC→D、RoboCasa、Meta‑World MT50、SimplerEnv；使用两种 VLA backbone：OpenVLA‑7B、Ctrl‑World；所有实验均基于离线演示数据集

**📈 对比分析**

与 WMPO、VLAW、RLVR‑World、Sim‑PPO、Scratch‑GRPO 等基线对比，采用成功率（Success Rate）、像素 MSE、ESNR 等指标；DRILL‑IMG 在五个基准上平均提升 13.1 点成功率（相较 WMPO）和 7.9 点（相较 RLVR‑World），尽管一阶像素 MSE 上升 17%；DRILL‑VWI 也显著优于 WMPO，且与 RLVR‑World 预训练可互补

**⚠️ 局限性**

仅在仿真环境中验证，缺乏真实机器人交互；DRILL‑IMG 需要额外的模拟器回合、截断 BPTT 以及 Hessian‑vector 计算，导致计算与内存开销增大；对感知噪声、延迟、仿真到真实的域漂移等实际因素的鲁棒性尚未测试

---

## 292. Equilibria and Learning in Competitive Market Making

**arXiv ID:** 2610.30891 | [PDF](https://arxiv.org/pdf/2610.30891v1)

**作者:** Matteo Onger `[一作]` (University of Milan), Luigi Foscari `[通讯]` (University of Milan)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并分析了一种在两侧报价、单一无差异性交易者参与的市场做市游戏，并证明了其纯策略纳什均衡的存在与结构，研究了波动率对均衡价差的影响，并证明在离散化价位下的无内部遗憾学习算法可以逼近连续均衡价位；

**💡 创新点**

创新点在于：①首次将伯特兰竞争与两侧报价结合，给出严格的纯策略均衡存在证明；②通过零利润条件与固定点理论揭示均衡价差随波动率线性扩张；③在非连续收益的环境下引入格点离散化，并证明无内部遗憾学习者的时间平均行为收敛到连续纳什价位外壳；

**🔧 技术方法**

技术手段包括：Tarski固定点定理、零利润报价构造、价格网格化、内部遗憾学习（如swap‑regret、Exp3改造）、相关均衡分析与对称性证明；

**📊 数据集**

实验使用人工合成数据，资产价值按均值0.5、方差σ²的高斯分布（截断至[0,1]）进行多次仿真；

**📈 对比分析**

与理论价位（连续纳什均衡）对比，学习者在细网格下的问价与出价逐步逼近理论曲线；在固定网格下，学习者的价差随σ线性增长并略高于理论预测，验证了波动率对价差的线性影响；

**⚠️ 局限性**

局限性包括：需要额外的正则性假设以保证H'(m)≤1；仅证明了无内部遗憾学习者的时间平均收敛，未证明对单一策略的收敛速度；研究只涵盖了两位做市商与无差异交易者的简化设定，未考虑库存、时延、异质订单流等更现实因素；

---

## 293. From Perpetrators' Digital Footprints to Behavioral Communities: An Ontology-Driven Framework for Bengali Cyberbullying Analysis--- FULL VERSION

**arXiv ID:** 2610.30885 | [PDF](https://arxiv.org/pdf/2610.30885v1)

**作者:** Md. Nesarul Hoque `[一作]` (University of Chittagong), Rudra Pratap Deb Nath `[通讯]` (University of Chittagong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个以实施本体驱动的网络分析框架（ODCBNA），实现从加害者数字足迹到行为社区的构建与分析；

**💡 创新点**

整合了本体设计、知识图谱构建、行为特征抽取、社区检测与影响力传播四大模块，首次在同一框架内兼顾语义、行为与网络层面的可解释性；

**🔧 技术方法**

使用本体建模、RDF–异构图转换、k‑NN相似度网络、Louvain社区划分、贡献度分解与IC模型的贪婪影响力最大化；

**📊 数据集**

采用BenCoDiC（孟加拉语YouTube政治新闻）语料库，识别出1063名至少三次发表侮辱评论的加害者；

**📈 对比分析**

与三种基线（仅按侮辱强度、加权度数、PageRank）进行比较，贪婪方法在六个社区内均获得略高的传播扩散，提升幅度均低于1%；

**⚠️ 局限性**

局限包括：特征手工设计、仅静态网络分析、对参数敏感、未对时变传播建模、仅在单一语言/主题/平台验证，需进一步扩展与鲁棒性检验。

---

## 294. ATLAS: A Multi-Agent Framework for Lifting Binary to Canonical Compiler IR

**arXiv ID:** 2610.30877 | [PDF](https://arxiv.org/pdf/2610.30877v1)

**作者:** Xiaoao Zhu `[一作]` (Shaanxi Normal University), Zheng Wang `[通讯]` (University of Leeds)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种多代理框架（Lift‑Advisor‑Validation），能够将已高度优化、去符号化的二进制文件恢复为可验证且结构完整的LLVM IR；

**💡 创新点**

创新点在于将控制流感知的检索增强生成（Structure‑Aware RAG）与基于LLVM验证器的迭代修正循环相结合，实现了从低级机器码到预优化IR的高保真映射；

**🔧 技术方法**

核心技术包括伪探针属性化的CFG对齐分割、基于Nova‑1.3B 的区域级检索、LLM在检索上下文下生成IR、Advisor 诊断映射与局部修正、以及 Validation 通过重编译与动态测试验证的闭环；

**📊 数据集**

使用公开的ExeBench数据集构建了对齐的（汇编，IR）对，覆盖x86‑64、ARM64、RISC‑V；并在HumanEval‑Decompile与MBPP两个基准上评估；

**📈 对比分析**

与七个基线（RetDec、MCTOLL、Qwen‑Plus、ChatGPT‑4o、Claude、DeepSeek‑V3.2、Gemini‑3）对比，本文在三种ISA与四个优化等级下平均提升了97%+的可编译率、94%+的可执行率以及51%+的编辑相似度；

**⚠️ 局限性**

局限包括：需针对新ISA重建检索数据库；在无测试用例的真实逆向场景下，功能反馈仍难以获得；对极大函数或极高优化程度下的上下文窗口仍有限制。

---

## 295. Verus-Insight: Towards Explainable Artifact Detection in AI-Generated Videos

**arXiv ID:** 2610.30866 | [PDF](https://arxiv.org/pdf/2610.30866v1)

**作者:** Zhuoran Zhang `[一作]` (Peking University), Wenjing Yang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Verus-Insight框架，将AI生成视频检测拆分为Inspection Planning、Evidence Verification和Evidence Aggregation三阶段，并通过人类锚定的全流程监督与逐步能力训练来实现精细化的人工可感知artifact检测与视频来源判定。

**💡 创新点**

创新点包括：① 将检测过程分解为三阶段，并对每阶段设计专门的监督与评估；② 引入样本特定、结果中立的Rubric（检验计划）与显式时空定位接口；③ 采用区域跟踪warm‑up和显式时空标注的Evidence Verification；④ 结合RL（GRPO）奖励源判断与artifact恢复；⑤ 搭建Verus-Bench细粒度人类标注benchmark。

**🔧 技术方法**

使用技术：多模态大型语言模型（Qwen3‑VL‑8B/30B、InternVL3.5等），区域跟踪、Rubric规划、显式时空对齐、逐步训练（warm‑up → SFT → RL）以及基于奖励的强化学习。

**📊 数据集**

数据集：Artifact‑Bench、RewardData、ViF‑Bench、Verus‑Bench（自建细粒度人工标注集），以及ViF‑CoT‑4K子集；真实视频来源于NExT‑QA、OneThinker‑STVG、DiDeMo及网络；AI‑generated视频来自Kling‑2.5、Veo 3、HunyuanVideo‑1.5、LTX‑2.3、Wan2.2等模型。

**📈 对比分析**

与Gemini、GPT‑5.5等专有模型、InternVL、Qwen、Skyra、BusterX++、VideoVeritas等公开模型进行对比；在RewardData、Artifact‑Bench T1/T3、Verus‑Bench等评测中，Verus‑Insight在视频级别检测的BA达94.79、Plan F1/Finding F1显著提升（如Verus‑Bench Plan F1 24.70→35.42，Finding F1 15.97→35.42），在artifact检索和真实视频artifact FPR方面均优于对比模型。

**⚠️ 局限性**

局限性：① 仍需大量人工标注的全流程监督，标注成本高；② 对未知artifact类型的泛化仍有限；③ 在极稀有或高分辨率视频中，区域跟踪与时空定位可能失效；④ 计算资源需求较高，尤其是RL阶段；⑤ 主要针对视频来源判定，未深入探讨跨模态迁移与持续学习。

---

## 296. Consensus-Based Distributed Filtering for Angles-Only Tracking in Cislunar Space

**arXiv ID:** 2610.30864 | [PDF](https://arxiv.org/pdf/2610.30864v1)

**作者:** Giovanni Romagnoli `[一作]` (University of Pisa), Giordana Bucchioni `[通讯]` (University of Pisa)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aaccfe5c-6b26-4208-b23c-35331481e142` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在双向角度测量的月球-地球空间中研究并比较了三种基于平均共识的分布式滤波策略（DEKF、ICF、KCF）以及中心化 EKF，评估其在有限通信和先验不一致条件下的跟踪性能。

**💡 创新点**

首次将三种共识滤波方法在近似的 NRHO 轨道上进行实景模拟，系统地比较它们对通信迭代次数和先验偏差的敏感性，并证明 ICF 在有限共识和先验误差下仍能保持接近中心化结果。

**🔧 技术方法**

采用分布式扩展卡尔曼滤波、信息加权共识滤波、卡尔曼共识滤波、平均共识协议、CR3BP 简化动力学模型、SPICE 行星星历、蒙特卡罗仿真等技术。

**📊 数据集**

使用 SPICE 提供的地球-月球高保真动力学轨迹、NRHO 轨道观测者的 LOS 测量（高斯角度噪声 σθ=10^-4 rad）以及对观测者位置的理想已知假设进行仿真。

**📈 对比分析**

通过对三位观测者的欧氏位置误差平均值与中心化 EKF 及独立局部滤波器对比，评估不同共识迭代次数和先验偏差的影响。结果显示 ICF 和 DEKF 在足够共识迭代下可与中心化滤波器相当，KCF 在有限通信下仍优于独立滤波但误差略高；DEKF 对共识迭代和先验不一致极为敏感。

**⚠️ 局限性**

DEKF 需要大量共识迭代且对先验一致性要求高；ICF 在大幅偏差下仍有残差误差；KCF 由于忽略交叉协方差而理论上次优；研究未考虑通信延迟、丢包、观测者位置误差以及真实硬件验证。

---

## 297. Cross-Spectral Feature Fusion for Multispectral Document Binarization

**arXiv ID:** 2610.30863 | [PDF](https://arxiv.org/pdf/2610.30863v1)

**作者:** Yasas Savinda `[一作]` (University of Peradeniya), Maheshi Buddhinee Dissanayake `[通讯]` (University of Peradeniya)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种利用可调跨光谱特征融合和可视-近红外结构引导的多光谱文档二值化方法。

**💡 创新点**

创新点在于将目标可视-近红外边缘信息动态引入中间特征融合，并通过软门控实现自适应融合。

**🔧 技术方法**

采用了条件生成式对抗网络、1×1卷积门控、Sobel边缘提取、softmax加权融合以及多任务损失。

**📊 数据集**

使用了SMADI多光谱古籍数据集，包含500/600/700/1000 nm四通道。

**📈 对比分析**

与固定平均融合、CBAM、SE等对比，F3+NIR引导模型在F1、p-FM、IoU上显著优于基线，且在5折交叉验证中统计显著。

**⚠️ 局限性**

局限包括数据集规模有限、仅评估单一光谱组合、模型参数较大且推理速度需进一步压缩。

---

## 298. System-Level Evaluation of Uplink Access Schemes for Ambient IoT

**arXiv ID:** 2610.30856 | [PDF](https://arxiv.org/pdf/2610.30856v1)

**作者:** Douglas Dziedzorm Agbeve `[一作]` (IDLab, University of Antwerp – imec), Jeroen Famaey `[通讯]` (IDLab, University of Antwerp – imec)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对能量自给的环境物联网（Ambient IoT）设备在 NB‑IoT 上的上行接入方案进行系统级评估，并提出了一种混合请求方案 HSR，结合了预留请求时隙与机会性竞争。

**💡 创新点**

创新点在于：①将能量收集与超级电容器存储以及棕点阈值机制嵌入 ns‑3 的 LENA‑NB 模块；②设计了 SR（预留请求）与 HSR（混合请求）两种实现；③在同一能耗与收集条件下，首次对 RA、SR、HSR 与理想化 FUG 进行统一比较，量化其能耗‑延迟‑可持续性 trade‑off。

**🔧 技术方法**

技术手段包括：ns‑3 级模拟、LENA‑NB 拓展（SolarEnergyHarvester、GenericCapacitor、NbiotEnergyModel）、基于 3GPP 的 NB‑IoT 协议栈扩展（SR、HSR、FUG）、Markov 交通模型、周期性 sin² 太阳能收集曲线以及棕点阈值触发的能量门控。

**📊 数据集**

使用的数据集为仿真生成的合成流量与能量收集曲线：Markov‑UDP 生成的短报文（49 B）、日循环 sin² 太阳能曲线（峰值 1–2 W）、随机初始电容电压与 1 kHz 采样率的能量自足率。没有使用真实的物联网数据集。

**📈 对比分析**

比较方法：在相同网络配置（单小区、同一功率、同一路径损耗）下，变更 UE 数量（20–200）与峰值收集功率（1–2 W），进行 10 次独立仿真，每次测量平均上行延迟、能耗/比特和棕点出现比例。结果显示 HSR 将平均上行延迟降至 2.4–2.7 倍，SR 能耗低于 RA 约 24%，而 Idealized FUG 提供了最优下限。

**⚠️ 局限性**

局限性包括：①仅在仿真环境下验证，缺乏真实硬件验证；② Idealized FUG 的假设（即时知道报文到达）在实际中不可行；③在高 UE 密度下 HSR 的能耗上升，甚至超过 RA；④未考虑链路适配、功率控制或机器学习预测等可能进一步提升性能的技术。

---

## 299. Lean Finder V2: Semantic Search for Mathlib That Disambiguates User Queries

**arXiv ID:** 2610.30843 | [PDF](https://arxiv.org/pdf/2610.30843v1)

**作者:** Jialin Lu `[一作]` (Simon Fraser University), Wuyang Chen `[通讯]` (Simon Fraser University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 Lean Finder V2，利用对抗式生成器产生模糊查询，并在模型层微调检索器与输入层查询重写双重优化，以提升对短模糊 Mathlib 查询的检索性能。

**💡 创新点**

创新点：1) 同一批生成的模糊查询同时用于检索器对抗训练和查询重写；2) 采用检索难度奖励的对抗式生成器（GRPO）生成对检索器具有挑战性的查询；3) 在同一数据集上实现模型层和输入层的双重优化，显著提升检索与证明任务效果。

**🔧 技术方法**

技术：对抗强化学习（GRPO）+检索难度+对齐奖励生成器；掩码 InfoNCE 对检索器进行对比学习；GEPA 对查询重写提示进行优化；使用 Qwen3-Embedding-8B、Qwen3-4B、GPT-OSS-120B 等大型语言模型；对齐判定器（DeepSeek-V4.1-Flash）和评价模型。

**📊 数据集**

数据集：现有上下文丰富的查询-声明对（synthetic user-query/Mathlib declaration pairs、informalization 数据集、LeanSearch V2 数据）；600k+ 通过对抗训练生成的模糊查询-声明对；MathlibQR 人工标注基准；200 条无标签真实交互查询；FATE‑H theorem proving benchmark。

**📈 对比分析**

比较方法：在 MathlibQR（810 条简短查询）和 2,526 条上下文丰富查询集上，评估 Recall@k 和 nDCG@k；与 LeanExplore、LeanSearch V2、Claude Code 等系统对比；在无标签交互查询中由 GPT‑5.6 Sol、Claude Opus‑5 进行 LLM 判别，记录平均排名和 Rank‑1 率；在 FATE‑H 证明任务中记录 Pass‑rate。性能方面：Lean Finder V2 在 MathlibQR Recall@10 提升约 14%/11%，在无标签查询中获得最高平均排名（≈1.98），在 FATE‑H 中 Pass‑rate 提升至 39.3%。

**⚠️ 局限性**

局限性：1) 依赖对齐判定器和人工标注的对齐门；2) 对抗生成的查询虽然规模大，但仍可能不足以覆盖全部真实查询多样性；3) 重写提示的 GEPA 优化过程耗时且需要额外计算资源；4) 在极长或结构复杂的查询上检索与重写效果尚未充分验证。

---

## 300. TRIAD: Benchmarking Omni-Modal Ambiguity in Multimodal Large Language Models

**arXiv ID:** 2610.30841 | [PDF](https://arxiv.org/pdf/2610.30841v1)

**作者:** Zhaolu Kang `[一作]` (Hunyuan, Tencent), Zhichao hu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个三模态（文本–图像–音频）互相消歧的基准，用于测试多模态大语言模型在跨模态信息整合下的消歧能力。

**💡 创新点**

创新点在于设计了严格的well‑posedness约束，构建了18类跨模态歧义的分类体系，并通过场景组（sister groups）来诊断模型在不同模态缺失时的一致性与依赖性。

**🔧 技术方法**

采用人工构造的文本、图片与音频对，配合五人标注与Python校验实现well‑posedness检查；同时对14种现有全模态模型进行多提示格式与模态缺失（leave‑one‑out）实验。

**📊 数据集**

数据集包含327条查询、86个场景组，涵盖英语与中文，配有18个歧义类别，且每条查询都附带完整的文本、图像、音频、问题与选项。

**📈 对比分析**

评估方法使用query accuracy与group accuracy两种指标，结果显示模型在完整输入下query accuracy接近人类但group accuracy仍低20+分；去除选项后模型跌至个位数，而人类保持高分；模态缺失实验进一步揭示模型存在提前承诺与对单一模态的依赖。

**⚠️ 局限性**

局限性包括基准规模有限，三模态互相消歧样本稀缺，语言与音频多样性不足；well‑posedness检查与开放式评分仍依赖人工或判定模型，限制了自动化评估的完整性。

---

## 301. What Ethereum Developers Ask For, and Whether They Get It: Ten Years of Ethereum Stack Exchange

**arXiv ID:** 2610.30829 | [PDF](https://arxiv.org/pdf/2610.30829v1)

**作者:** Tamer Abdelaziz `[一作]` `[通讯]` (New York University Abu Dhabi), Tamer Abdelaziz (New York University Abu Dhabi)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2016–2025 年 Ethereum Stack Exchange 的完整问答数据进行主题标注、需求与解决率分析，研究开发者提问主题、解答与接受情况及其随时间变化的趋势。

**💡 创新点**

①使用提示式主题模型和多模型委员会验证，提供十年完整记录的高精度主题标注；②以两步漏斗和直接标准化分离任务变更与解决率下降的原因；③揭示提问者声誉与回答质量的显著关联，以及新人提问对平台关闭率下降的贡献。

**🔧 技术方法**

Prompt‑based 主题建模、三大语言模型评估（GPT‑4o、DeepSeek‑v4‑pro、Gemini‑2.5‑pro）、Fleiss κ一致性、直接标准化、Mantel–Haenszel 加权、时间序列趋势估计（OLS、Theil–Sen）以及文本特征提取和正则表达式。

**📊 数据集**

Ethereum Stack Exchange 完整问答数据集（2016‑2025 年，共约 44,000 已回答 + 9,000 未回答问题），包含问题文本、标签、时间、用户声誉、答案信息，以及通过提示式模型生成的多标签主题。

**📈 对比分析**

通过对比两段时期（2016‑2022 vs 2023‑2025）的回答率、接受率、时延等指标，并利用直接标准化分离主题混合效应；结果显示回答率下降 9.5–18.6pp，接受率下降 16.2–24.5pp，声誉梯度占比达 59%；模型标注准确率约 88–94%，成本仅 16.24 美元。

**⚠️ 局限性**

主题标注对宽泛标签（如 Contract Architecture）准确率仅 50%；数据基于导出时声誉快照，无法区分问答后声誉提升的影响；仅记录公开提问，未覆盖私下解决的问题；与语言模型助理的因果关系仅为时间相关性，缺乏外部活动指标。

---

## 302. TutorCaseBench: A Case-Grounded Benchmark for Evaluating LLM Tutors in Supporting Students with Challenging Behaviors

**arXiv ID:** 2610.30828 | [PDF](https://arxiv.org/pdf/2610.30828v1)

**作者:** Yeon Su Park `[一作]` (KAIST), Juho Kim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基准TutorCaseBench，利用教育心理学和特殊教育的定性案例自动生成923个对话情境，覆盖五类挑战行为，提供开放式生成和多选题两种评测方式。

**💡 创新点**

创新点在于将真实案例研究转换为案例驱动的对话与基于案例的评估标准，聚焦学生的挑战行为而非仅认知错误，避免传统基准中理想化学生模型的缺陷。

**🔧 技术方法**

技术上使用Gemini等LLM完成文本提取、情境构造、对话生成与评分标准生成；并结合检索+LLM判定构建多选题；通过开放式生成和多选题两种任务评估模型。

**📊 数据集**

数据来源于ERIC数据库的2010‑2026年案例研究，共300篇论文，生成518个学生类别，最终得到923个情境、1,560条开放式题目和2,537条多选题。

**📈 对比分析**

通过与11个最先进LLM（Claude Opus 4.7、GPT‑5.5、Gemini 3.1 Pro等）在开放式生成和多选题上的对比，开放式最高成功率为55.8%，多选题最高准确率为63%；模型在即时回应上优于长期干预，整体表现仍低于人类教师。

**⚠️ 局限性**

局限性包括：基准构建主要依赖Gemini，可能引入模型偏见；案例覆盖的学生人群和教育场景有限；评估仅基于文本对话，无法完整捕捉多模态行为；未对跨案例相似性进行聚合分析。

---

## 303. Breaking the Coupling: Efficient Concurrency Fuzzing for Linux Kernel Data Race Detection

**arXiv ID:** 2610.30813 | [PDF](https://arxiv.org/pdf/2610.30813v1)

**作者:** Cen Zhang `[一作]` (Beihang University), Zuming Jiang `[通讯]` (University of Hong Kong)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的 Linux 内核并发模糊测试范式，将并发输入生成与线程调度解耦，利用在并发执行中收集的 May Race Pair（MRP）筛选有潜在数据竞争的输入，集中昂贵的调度资源。

**💡 创新点**

创新点包括：① 用 MRP 作为轻量且有针对性的反馈，显著降低对调度的依赖；② 将 LLM 辅助的语义变异与动态阈值控制、历史上下文恢复、ASN 匹配等多技术结合，提升输入多样性与调度效率；③ 通过消融实验验证每一组件的独立贡献，展示解耦框架的可扩展性。

**🔧 技术方法**

采用技术包括：LLM（DeepSeek‑V4‑Pro 或 GPT‑5.4）辅助语义变异、LLM Prompt 设计、动态时间阈值控制、LLM API 调用计费与优化、LLVM 传感器插桩、Syzkaller 扩展、历史上下文恢复、ASN 访问序号匹配、候选 MRP 优先级调度、上下文感知调度与 watchpoint 确认。

**📊 数据集**

数据集为 Linux 6.17 上的八个常用内核模块（XFS、Btrfs、PTMX、OSS DSP、Bluetooth、F2FS、JFS、Floppy），并在每个模块上持续多周的模糊测试，收集 MRP、真实竞争实例及其安全影响。

**📈 对比分析**

对比方法：在相同硬件（32 核 128 GB）和相同测试时长（24 h）下，与 SegFuzz 与 CONZZER 并行运行；使用相同的 MRP 处理规则；结果显示本框架在 24 h 内发现 79 条真实竞争（DeepSeek‑V4‑Pro 配置），相比 SegFuzz 的 19 条、CONZZER 的 11 条提升 4.16× 与 7.18×；消融实验显示动态阈值、LLM 变异和调度组件各自可提升 30–50% 的竞争发现率。

**⚠️ 局限性**

局限性：1) LLM 推理耗时与成本高，尤其在大规模并发变异时；2) 需要 QEMU 快照恢复的调度过程带来额外资源消耗；3) 动态阈值与参数需要针对不同模块手动调优；4) 当前仅针对数据竞争，其他并发缺陷需设计新的轻量抽象；5) 结果对执行环境（CPU 架构、虚拟化设置）敏感，复现性受限。

---

## 304. A Weakly Group Strategy-Proof and Pareto-Stable Matching Mechanism with Two-Sided Indifferences under Gross Substitutes

**arXiv ID:** 2610.30807 | [PDF](https://arxiv.org/pdf/2610.30807v1)

**作者:** Yasushi Kawase `[一作]` (Chuo University), Ryosuke Sato `[通讯]` (Chuo University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构造了一种“广义延迟接收”（generalized deferred acceptance）机制，解决多对一匹配市场中双方存在等价偏好（indifferences）的情况下，兼顾弱稳定性、Pareto效率以及工人弱群体策略不变性（worker weakly group strategy-proofness）的问题。

**💡 创新点**

首次证明在工人和公司均可出现等价偏好、且公司对工人集合的偏好可由广义替代（gross substitutes, GS）函数表示的市场中，存在一个确定性的、强多项式时间可实现的机制，该机制在所有工人报告的偏好下都能产生Pareto稳定的匹配，并对工人保持弱群体策略不变；并且证明此机制在满足公司索引顺序的前提下，不能被任何策略不变机制统一优越。

**🔧 技术方法**

利用GS函数的凸性（discrete convex analysis）、层次坡度市场（tiered‑slope market）与不完全可转移效用的竞争均衡理论，构建辅助市场；通过序列独裁（serial dictatorship）与层级层次匹配的组合，得到匹配的最优子集；并用严格的数学归纳和竞争均衡的支付结构证明Pareto效率与策略不变性。

**📊 数据集**

该工作为纯理论研究，未使用任何真实数据集，全部以数学模型与证明为基础。

**📈 对比分析**

相较于传统的随机打破等价并执行标准延迟接收（DA）的做法，该机制在保持弱稳定性的前提下显著提高了Pareto效率，且在实现时间上保持了强多项式复杂度；在理论上已证明其优于任何遵守公司索引顺序的策略不变机制。

**⚠️ 局限性**

限制主要包括：① 必须假设公司对工人集合的偏好可由GS函数表示（即满足广义替代性），这限制了可适用的实际情境；② 机制假定公司偏好是公开且固定的；③ 对于仅给出选择函数而非完整偏好关系的情况，该机制可能产生不稳定匹配；④ 研究未考虑动态或信息不完全的情形。

---

## 305. Can Reciprocal RIS Perform as Non-Reciprocal RIS?

**arXiv ID:** 2610.30896 | [PDF](https://arxiv.org/pdf/2610.30896v1)

**作者:** Matteo Nerini `[一作]` (Imperial College London), Bruno Clerckx `[通讯]` (Imperial College London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对可逆与非可逆可重构智能表面（RIS）在不同信道与前向/后向传输条件下的性能关系进行理论分析，并给出必要且充分的条件

**💡 创新点**

首次给出当且仅当非可逆RIS的最佳解使得发射端至RIS再回发射端的回传通道保持对称时，可逆RIS能达到与非可逆RIS同等性能，并将该条件进一步细化到考虑前向编码的情形

**🔧 技术方法**

利用线性代数（酉矩阵、对称性约束）、信息理论（容量/速率指标）、前向编码与组合理论，并使用Lean 4 形式化验证证明核心定理

**📊 数据集**

无实验数据集，全部通过理论推导与数值仿真验证所得结论

**📈 对比分析**

通过对比在单用户、单流与多流情形下的容量/速率曲线，表明当发送端到RIS通道秩为1或单流最优时，可逆RIS与非可逆RIS性能相同；否则可逆RIS会有性能损失

**⚠️ 局限性**

主要局限在于仅给出了理论必要充分条件，实际系统中非理想因素（如非线性元件、功率消耗、硬件成本）未被量化，且对动态多用户与多天线网络的适用性仍需进一步研究

---

## 306. Building Presentation Intelligence for High-Quality Slide Generation

**arXiv ID:** 2610.30888 | [PDF](https://arxiv.org/pdf/2610.30888v1)

**作者:** Yuhang Xie `[一作]` (Peking University), Lin Qu `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出统一的训练框架，实现自动PPT生成，包括跨验证数据合成、教师增强与混合教师监督、SFT与基于可验证美学奖励的GRPO强化学习。

**💡 创新点**

创新点：①跨验证合成流水线可显式控制任务难度与有效性；②基于规则与先验知识的教师增强框架，提升教师示例质量；③混合教师与轻量化轨迹过滤提升SFT泛化；④设计可验证的五维美学指标并集成到GRPO中，实现视觉属性的直接优化。

**🔧 技术方法**

技术：大语言模型（Qwen‑3.6‑27B、Qwen‑3.6‑35B‑A3B、Qwen‑3.8‑Max、GLM‑5.2），监督微调、混合教师策略、轨迹过滤、GRPO强化学习、可验证规则式美学指标（纵横比、白色空间、元素冲突、视觉平衡、字体可读性）。

**📊 数据集**

数据集：自研的跨验证合成数据（4802个任务实例，209,966条评测条目，覆盖25主域+162副域）；评测使用公开基准PresentBench与GDPVal PPT子集，并在OpenClaw框架中实现。

**📈 对比分析**

对比方法：在PresentBench与GDPVal上与GLM‑5.2及无SFT模型比较；SFT后Qwen‑3.6‑27B提升7.45点、Qwen‑3.6‑35B‑A3B提升9.35点；RL后美学得分提升至4.36（PresentBench）/3.75（GDPVal）等，明显优于基线，展示出良好的泛化与提升幅度。

**⚠️ 局限性**

局限：①RL单轮训练、仅使用4k+样本，难以覆盖极端复杂场景；②美学指标仍局限于规则化衡量，可能忽略更细腻的视觉细节；③KL正则在某些设置下限制了收益；④模型对模板和代码生成的依赖可能导致在非预期格式下的适配问题。

---

## 307. Direct Forcing: A Single-Stage Training Recipe for Few-Step Autoregressive Text-to-Video Generation

**arXiv ID:** 2610.30861 | [PDF](https://arxiv.org/pdf/2610.30861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 308. Reverse-KL Policy Distillation: Statistical Guarantees and Limits

**arXiv ID:** 2610.30844 | [PDF](https://arxiv.org/pdf/2610.30844v1)

**作者:** Haolin Liu `[一作]` (University of Virginia), Julian Zimmert `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了反向Kullback-Leibler (KL) 蒸馏在语言模型训练中的样本复杂性，特别关注教师反馈的不同形式对学习效率的影响。

**💡 创新点**

创新点在于首次建立了在线和离线反向KL蒸馏的统计保证，并比较了教师提供完整动作分布与仅提供观察到的动作概率的效果。

**🔧 技术方法**

使用了轨迹损失和重要性比率截断技术，以及策略空间学习来实现最佳的在线蒸馏保证。

**📊 数据集**

使用了教师生成的轨迹和学生生成的轨迹进行实验，具体数据集未明确提及。

**📈 对比分析**

与常见的在线目标相比，提出的方法在样本效率上显著提高，尤其是在教师提供完整反馈的情况下，在线学习的样本需求显著低于仅提供动作反馈的情况。

**⚠️ 局限性**

限制在于在线学习中仅使用动作反馈时，样本需求显著增加，即使在可实现的设置下，仍需额外的样本量来达到相同的学习效果。

---

## 309. Semi-Linear Gaussian Bayesian Networks: Enhancing Flexibility in Node Dependencies via penalized B-Splines

**arXiv ID:** 2610.30838 | [PDF](https://arxiv.org/pdf/2610.30838v1)

**作者:** Aitor Oviedo-Madrid `[一作]` (University of Navarra), Ruben Armañanzas `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了半线性高斯贝叶斯网络（SLGBN），用于在变量近似高斯但关系非线性的场景下建模连续变量；

**💡 创新点**

在保持条件高斯分布的同时，引入通用加性模型（GAM）与惩罚性B样条实现非线性条件均值，兼顾线性高斯与半参数贝叶斯网络的优点；

**🔧 技术方法**

采用GAM+惩罚性B样条进行非线性节点参数估计，OLS估计线性节点参数，利用贪婪爬山算法（GHC）联合学习结构与节点类型，使用BIC与改进EBIC*作为评分函数；

**📊 数据集**

在三种合成网络（B5、B8、B_insu）上生成不同样本量（200、1000、5000、10000）数据进行实验；

**📈 对比分析**

与线性高斯贝叶斯网络（LGBN）和半参数贝叶斯网络（SPBN）比较，评估指标包括验证集对数似然、结构汉明距离（SHD）、类型汉明距离（THD）、精确率与召回率；实验表明SLGBN在对数似然和结构恢复上均优于LGBN和SPBN，尤其在使用EBIC*时表现最佳；

**⚠️ 局限性**

局限性包括对条件高斯性和加性依赖的假设、需要预先设定样条基数与惩罚参数、在网络规模增大时学习效率下降、对非高斯或非加性关系的鲁棒性尚未验证；

---

## 310. Neural Spectral Capacity with Residual Context for Heterogeneous Resource Allocation within Transformers

**arXiv ID:** 2610.30823 | [PDF](https://arxiv.org/pdf/2610.30823v1)

**作者:** Chenyu Zhu `[一作]`, Zhichao Lu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了残差感知神经光谱容量（ResNSC）及其分配算法ResNSC-DP，用于在不训练模型的情况下预测Transformer架构的相对性能；

**💡 创新点**

通过在容量评估中引入残差流历史和分支内部传播，解决了NSC对层次顺序不敏感的缺陷，使同一组投影在不同位置获得不同评分；

**🔧 技术方法**

采用基于随机矩阵理论的本地容量估计、残差参考协方差构造、动态规划与剩余上下界搜索技术；

**📊 数据集**

在121M和374M规模的decoder-only Transformer上使用FineWeb‑Edu、WikiText‑2、LAMBADA数据集进行从零开始预训练，并在LoNAS-LLaMA-7B超级网络上进行结构剪枝；

**📈 对比分析**

与NSC-DP、Uniform、VWT、TLM等基线比较，ResNSC-DP在相同参数预算下在FineWeb‑Edu、WikiText‑2、LAMBADA上显著降低困惑度，且在LoNAS剪枝任务中平均准确率提升3.3–5.5个百分点；

**⚠️ 局限性**

局限性包括仅考虑残差流信息，未覆盖注意力混合或专家路由等更复杂的计算图结构；

---

## 311. Bezier-Guided Adaptive Flow for Cross-domain Sequential Recommendation

**arXiv ID:** 2610.30825 | [PDF](https://arxiv.org/pdf/2610.30825v1)

**作者:** Ziang Lu `[一作]` (Anhui University), Yiwen Zhang `[通讯]` (Anhui University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了BGAFlow框架，用于跨域序列推荐，通过联合双域预训练构建共享语义空间，并采用条件Bézier曲线与状态自适应残差流实现个性化的语义迁移。

**💡 创新点**

创新点包括：① 通过原型级兴趣重叠和几何相似度估计用户专属迁移时间窗（horizon），实现个性化迁移强度；② 将Bézier曲线作为几何参考，学习任务导向的残差速度场，实现状态自适应的语义演化；③ 使用动态原型反馈和smoothstep调度实现梯度平滑与方向自适应；④ 结合任务导向的流匹配与轨迹平滑正则化，提升迁移稳定性与推荐性能。

**🔧 技术方法**

技术手段包括：LLM推断用户兴趣轮廓、Transformer双塔编码器、Mixture‑of‑Experts文本投影、Profile/Item对齐与CORAL正则化、Bézier曲线几何参考、残差流网络（flow matching）、smoothstep调度、任务导向的梯度监督、轨迹平滑正则化和多步数残差积分。

**📊 数据集**

使用Amazon Review公开数据集，构造Movie‑Book和Sport‑Cloth两个跨域序列推荐数据集，包含数万用户、数十万条交互记录，覆盖电影、书籍、体育和服装四个领域。

**📈 对比分析**

在HR@5/10、NDCG@5/10指标上，与单域、传统跨域、扩散、LLM增强等多种基线进行对比；BGAFlow在所有域和指标上均实现3.91%–16.62%的相对提升，成为目前最优方法。

**⚠️ 局限性**

局限性包括：① 需要高质量LLM生成的兴趣摘要，若LLM输出噪声会影响预训练；② 仅针对两域场景，扩展到多域尚未验证；③ 计算量相对较大（多步积分、原型计算）；④ 对于极低相似度用户，仍可能出现负迁移，需进一步研究自适应抑制机制。

---

## 312. Global Convergence of Detailed Balanced Mass Action Systems

**arXiv ID:** 2610.30808 | [PDF](https://arxiv.org/pdf/2610.30808v1)

**作者:** Haoran Lv `[一作]` (BitMiracle AI Lab), Shenli He `[通讯]` (BitMiracle AI Lab)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在详细平衡质量作用系统上提供了全新的全局收敛性证明，并将完整证明过程形式化到Lean 4中

**💡 创新点**

将持久性、势能消散与指数参数化三段经典Horn–Jackson论证统一到一个自洽的、可机检验的证明框架，并使用可行方向与对数壁垒实现轨道界定

**🔧 技术方法**

采用对数坐标、可行方向判据、可行方向法、势能（熵）函数、指数参数化、缓慢速度下降、以及Lean 4+Mathlib的形式化工具

**📊 数据集**

无实验数据集，证明基于理论推导与形式化验证

**📈 对比分析**

与Craciun等人基于离散微分包含的证明相比，本工作给出更直接、工具驱动且机器可检验的证明，且无需连续性或唯一性假设，证明完整性得到自动化验证

**⚠️ 局限性**

仅适用于正的详细平衡质量作用系统；未涵盖非正初始状态或非详细平衡网络，且在实际网络复杂度高时形式化文件规模庞大，易受依赖库改动影响

---

## 313. Parameter Learning in Active Inference via Variational Message Passing

**arXiv ID:** 2610.30899 | [PDF](https://arxiv.org/pdf/2610.30899v1)

**作者:** Benedikt Waldeck `[一作]` `[通讯]` (University of Modena and Reggio Emilia), Benedikt Waldeck (University of Modena and Reggio Emilia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于变分信息传播的主动推理框架，在同一目标下实现观察与转移模型参数的联合学习与规划。

**💡 创新点**

创新点在于将每个因子参数的先验冻结，保持因子贝叶斯对偶性，从而在Bethe近似下恢复信息增益权重，使得单一变分目标可通过Sum‑Product消息传递实现参数学习与规划。

**🔧 技术方法**

使用了变分自由能、Bethe近似、Sum‑Product（Belief Propagation）消息传递、Dirichlet共轭先验、信息增益正则化以及因子层面参数复制等技术。

**📊 数据集**

实验采用了两个网格POMDP环境：雾墙迷宫（Fog‑Walls）和未映射分支迷宫（Unmapped Detour）来评估方法。

**📈 对比分析**

与基于偏好的SP‑Pref‑Plan、SP‑Pref以及删除信息增益或动态通道的 ablation 版本进行比较；在两种环境中，方法在学习参数后几乎所有试验都成功到达目标，收敛速度快于基线，并能避开危险路径。

**⚠️ 局限性**

局限性包括：探索必须可生存、未知区域不大且模型知识能显著改变路径；目标非凸且依赖延迟消息传递，可能导致收敛不稳定；未探讨更复杂规划或分布修正的情况。

---

## 314. One Teacher, Many Views: Reducing Reference Sensitivity in On-Policy Self-Distillation

**arXiv ID:** 2610.30880 | [PDF](https://arxiv.org/pdf/2610.30880v1)

**作者:** Khouloud Saadi `[一作]` (King Abdullah University of Science and Technology), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在强化学习框架下提出Monte Carlo Privileged Distillation (MCPD)，通过对完整参考解进行部分遮蔽采样并平均多视角教师预测，从而为自我对话模型提供更稳定的密集监督。

**💡 创新点**

创新点在于：①识别并量化“参考引发的目标不稳定性”，②将完整参考视为部分视图的概率分布，③证明在视角平均后得到的目标即为期望KL损失的最优解，从而实现理论与实践的一致性。

**🔧 技术方法**

技术手段包括：on‑policy自我对话（OPSD）、stop‑gradient教师、随机遮蔽（Masking）产生部分视图、Monte Carlo估计、多视角概率平均、KL散度优化、样本token级监督等。

**📊 数据集**

实验数据集覆盖：数学推理（AIME 2024/2025、HMMT 2025、AMO‑Bench）与科学推理（SciKnowEval L3），同时在数学训练模型上评估跨域到科学领域的迁移能力。

**📈 对比分析**

与SFT、GRPO、原始OPSD和AVSD等基线相比，MCPD在四大数学基座上均取得最高平均准确率；在AIME 2025上提升约2%点，在HMMT 2025上提升约5%点；在SciKnowEval上亦实现最高分数，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：①需要额外的教师计算与多次遮蔽采样，导致训练时间略有增加；②对参考解的质量和多样性仍有依赖，若参考不足或极端不一致，仍可能产生不稳定的目标；③目前仅在文本推理任务验证，尚未探究多模态或更大规模模型的可扩展性。

---

## 315. When Do Self-Supervised Representations Help Speech Restoration?

**arXiv ID:** 2610.30869 | [PDF](https://arxiv.org/pdf/2610.30869v1)

**作者:** Yongjoon Lee `[一作]` (KAIST), Jung-Woo Choi `[通讯]` (KAIST)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 Early Restoration Injection (ERI) 方法，将声学恢复网络（SR）的中间特征注入 SSL 编码器，以解决 SSL 特征清洗在严重失真时导致的感知相关性崩塌。

**💡 创新点**

创新点在于：①发现 SSL 特征清洗在高失真条件下与感知质量失去一致性；②通过注入部分恢复的中间特征来为 SSL 提供“恢复证据”，显著提高 SSL 损失向感知质量的转化；③使用轻量级条件网络和 stop‑gradient 机制，避免不必要的梯度传递，保持模型简洁。

**🔧 技术方法**

使用技术包括：WavLM、HuBERT 等 SSL 预训练模型，LoRA 微调，AdaLN 融合，L2 正则化的 MSE SSL 损失，LSGAN 生成器损失，t‑SNE 可视化，Spearman 相关分析，stop‑gradient 控制梯度流。

**📊 数据集**

使用数据集：LibriTTS‑R、VCTK‑0.92 进行训练；WHAM!、DNS‑Challenge 作为噪声；Arni RIR 用于回声仿真；VoiceFixer 官方测试集用于评估；实验还包含自建的多级失真数据集用于分析。

**📈 对比分析**

方法比较：在 VoiceFixer 测试集上与 VoiceFixer、SEMamba++、AnyEnhance、SIDON、TF‑GridNet 等基线进行对比，评价指标包括 SCOREQ、UTMOS、OVRL、PESQ、ESTOI、LPS。ERI 在 PESQ、ESTOI、LPS 上获得最佳成绩，UTMOS 维持竞争力；参数量与 SEMamba++ 相当，推理实时因子仅从 0.015 提升到 0.020，显示效率可接受。

**⚠️ 局限性**

局限性：①对 SSL 预训练模型的依赖较强，若 SSL 质量差则 ERI 效果受限；②虽然参数量不大，但相比纯 SR 方案仍有所增加；③仅在 VoiceFixer 及其模拟失真上验证，泛化到更广泛失真类型的性能尚未充分评估；④方法的理论解释仍待进一步阐明，特别是对感知相关性崩塌机制的更深层次理解。

---

## 316. Separate Field Scoring Protects Unused Tools When Agents Pool Tool-Use Experience

**arXiv ID:** 2610.30867 | [PDF](https://arxiv.org/pdf/2610.30867v1)

**作者:** Abhijit Chakraborty `[一作]` `[通讯]` (Arizona State University), Abhijit Chakraborty (Arizona State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究共享工具使用案例对工具检索的影响，比较将使用示例折叠进检索索引、路由器分配槽位和单独字段评分等方法，并评估跨代理共享收益。

**💡 创新点**

提出在检索索引中折叠使用示例以及基于字段的评分方式，并系统评估共享使用如何在不同工具集合和使用场景下保护未使用工具或提升已使用工具的 Hit@5。

**🔧 技术方法**

使用向量检索技术，基于 Jina‑v2 与 Nomic Embed v1.5 的文本嵌入，构造索引并计算 Cosine 相似度，采用 z‑normalization 结合权重或路由器分配方案。

**📊 数据集**

实验数据集包括 StableToolBench、TOUCAN 与 ToolRet 三个工具检索基准，分别包含数千工具与数百查询；使用不同 seed 的使用示例池。

**📈 对比分析**

通过 Hit@5 对比，发现字段评分在保持未使用工具 Hit@5 的同时提高已使用工具检索；路由器在已标记查询中表现更佳；共享使用在多代理情形下可获得与单代理相同的收益。整体提升幅度因数据集和嵌入器而异，字段评分在 StableToolBench 上提升约10‑15 点，路由器在 TOUCAN 上提升约9 点。

**⚠️ 局限性**

仅评估检索 Hit@5，未验证任务完成效果；未考虑隐私保护、任务级别性能；未测试其他专用检索模型；在 ToolRet 上字段评分无明显提升；实验仅在模拟代理，未验证实际联邦场景。

---

## 317. Training Mixed-Monotone Neural Networks for Verification Without Set Propagation

**arXiv ID:** 2610.30853 | [PDF](https://arxiv.org/pdf/2610.30853v1)

**作者:** Jonas Fingerhut `[一作]` (Technical University of Munich), Matthias Althoff `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种混合单调神经网络架构，将网络实现为两个单调子网络之差，从而通过两次前向传播即可得到输出集合包络；

**💡 创新点**

创新点在于将混合单调分解内嵌于网络结构，并通过训练直接最小化包络大小，给出总变差下界实现包络紧致；

**🔧 技术方法**

使用了符号约束单调网络、混合单调理论、总变差理论、训练时的包络大小惩罚项以及两次前向传播的包络计算；

**📊 数据集**

在标量与多输入解析模型、摆杆系统、罐类系统等回归基准上进行评估，涵盖不同层数和输入维度；

**📈 对比分析**

与IBP、Zonotope、CROWN、α‑CROWN等方法比较，结果显示该方法包络紧致度达到或超过α‑CROWN的95%+，计算速度比CROWN快256倍、α‑CROWN快20 600倍；

**⚠️ 局限性**

限制在于仅对回归任务，包络仅覆盖网络输出而非系统输出，且仅使用两子网络差分的分解形式，未来可探索更一般的分解以进一步紧致；

---

## 318. ProTrav: Probabilistic Traversability with Elevation Uncertainty Propagation for Risk-aware Off-road Navigation

**arXiv ID:** 2610.30839 | [PDF](https://arxiv.org/pdf/2610.30839v1)

**作者:** Chuanlin Zuo `[一作]` (Tampere University), Reza Ghabcheloo `[通讯]` (Tampere University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

ProTrav 通过稀疏变分高斯过程构建连续高度场，并利用函数空间采样生成空间相关的地形样本，计算坡度、粗糙度、台阶高度等几何特征，从而得到分布式越行成本；随后将该成本与上置信界（UCB）风险项结合，嵌入 MPPI 控制器实现风险感知导航。

**💡 创新点**

其创新点包括：①完整将地形高度不确定性通过函数空间采样传播至非线性几何特征和越行成本；②基于分布均值与方差构造 UCB 风险，动态权衡期望成本与不确定性；③首次在 MPPI 导航框架中直接使用概率越行成本，从而在不确定地形中显著提升安全性。

**🔧 技术方法**

所使用技术为：稀疏变分高斯过程（SVGP）建模；Monte Carlo 函数空间采样；坡度、粗糙度、台阶高度三种几何特征提取；基于均值与标准差的 UCB 风险；MPPI（Model Predictive Path Integral）控制；LiDAR 点云观测与 RTK‑GNSS‑IMU 传感。

**📊 数据集**

实验数据主要来自：①Gazebo 模拟的山地与悬崖两种地形（自定义三维模型），每种场景进行10次重复实验；②真实世界中四轮差速驱动机器人在 Mound、Ramp 与 Sequence 三种情境下使用 LiDAR 与 RTK‑GNSS‑IMU 现场点云。

**📈 对比分析**

与基线 Mean‑first RRT*、MPPI 以及不考虑不确定性的 MPPI（δ=0）对比：在山地场景所有 MPPI 方案成功率均为 100%，ProTrav(δ=1.645) 产生最小的俯仰滚转；在悬崖场景成功率从 10%（基线）提升至 90%（ProTrav δ=1.645）。在真实机器人部署中成功完成三种情境，避免坠落与滑移，提升姿态稳定性。

**⚠️ 局限性**

主要限制包括：①仅基于几何特征，未充分利用视觉或语义信息；②对观测稀疏度敏感，极端不确定区域仍可能导致误判；③函数空间采样和 MPPI 计算开销较大，实时性能受限；④随观测变化导致概率越行成本时间不一致，需进一步平滑与一致性处理。

---

## 319. RECAP: Looking Once Is Not Enough for Vision-Language Reasoning

**arXiv ID:** 2610.30834 | [PDF](https://arxiv.org/pdf/2610.30834v1)

**作者:** Zhaolu Kang `[一作]` (Peking University), Kaiyue Zhou `[通讯]` (Chengdu Minto Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究视觉-语言模型长链推理中的视觉回访（visual revisit）机制，并提出 RECAP 方法通过离线回访迹象改进强化学习中的信用分配。

**💡 创新点**

提出视觉回访作为时序定位的视觉支撑信号，并利用该信号实现动态折扣、回访门控价值预测等三种信用分配改进。

**🔧 技术方法**

基于 GRPO 的强化学习框架、交叉模态注意力、动态折扣、价值门控、RAP 指标等技术。

**📊 数据集**

在 Qwen2.5‑VL‑7B、InternVL3‑8B、LLaVA‑OV‑7B 等模型上，对 HallusionBench、POPE、MMHalBench、MathVista、MMMU、MathVerse、MM‑Vet 等视觉推理、错觉、跨语言等基准进行评测，并在 72B LoRA 版本、中文零样本、视觉扰动等数据集上验证。

**📈 对比分析**

相较于 GRPO、VPPO 等基线，RECAP 在 HallusionBench 上提升 3.5–4.1 分，整体保持推理和文本能力不变；在 72B LoRA、中文、对抗测试中亦表现稳健。

**⚠️ 局限性**

依赖可观测的跨模态注意力，短答案或文本任务效果有限，且对干扰视觉区域敏感；需要进一步探索更鲁棒的时序定位方式。

---

## 320. One Evaluation Is Enough: Inversion-Free Image Editing with Trajectory Correction

**arXiv ID:** 2610.30827 | [PDF](https://arxiv.org/pdf/2610.30827v1)

**作者:** Lifan Jiang `[一作]`, Boxi Wu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Endpoint-Calibrated Transport（ECT）方法，实现单一时间点的无逆向扩散图像编辑，仅需一次共享噪声状态和一次潜在更新，无需倒置、ODE或迭代优化；

**💡 创新点**

创新点在于：①单时间点编辑公式化，②利用VP调度的角度传输推导编辑方向，③基于相同查询的端点校准闭式修正，④训练无关且资源占用低；

**🔧 技术方法**

使用扩散模型的ε‑预测VP一阶架构，角度几何分析，闭式映射实现单步编辑，配合批量去噪器调用和潜在更新；

**📊 数据集**

在PIE‑Bench和SNR‑Bench两个公开评测基准上进行实验，使用多种后端模型如SD‑Turbo、SwiftBrush‑v2、FLUX、SD3.5等；

**📈 对比分析**

与多步、四步和其他单步编辑器（DirectEdit、DNAEdit、SNR‑Edit、FlowEdit、DRFS、FIA‑Edit、FTEdit、TurboEdit、InfEdit、InstantEdit、SwiftEdit、ChordEdit）在相同后端上对比；ECT在单步基准上实现最高PSNR、最佳评判分数，保持或提升保存指标，同时同步延迟下降15‑20%，CPU利用率下降42‑48%，内存占用下降17‑21%；

**⚠️ 局限性**

局限在于仅适用于ε‑预测VP一阶后端和共享噪声状态，无法覆盖非VP调度、速度预测模型或空间自适应约束，且评估结果受硬件/测量协议影响。

---

## 321. TESSA: A Governance-Oriented Custom GPT for SME Guidance under the EU AI Act: Design, Implementation, and Evaluation

**arXiv ID:** 2610.30822 | [PDF](https://arxiv.org/pdf/2610.30822v1)

**作者:** Chalisa Veesommai Sillberga `[一作]` (Tampere University), Pekka Abrahamssona `[通讯]` (Tampere University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

未提供具体论文内容，无法确定作者所做的工作。

**💡 创新点**

无法判断创新点。

**🔧 技术方法**

未说明使用的技术。

**📊 数据集**

未提供数据集信息。

**📈 对比分析**

无法比较方法或评估性能。

**⚠️ 局限性**

无法识别限制。

---

## 322. Random Initialization as a Baseline for Time Series Foundation Models

**arXiv ID:** 2610.30950 | [PDF](https://arxiv.org/pdf/2610.30950v1)

**作者:** Pinar Sungu Isiacik `[一作]` (University College Dublin), Georgiana Ifrim `[通讯]` (University College Dublin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

比较了七种时间序列基础模型在多变量时间序列分类任务中，预训练权重与相同架构随机初始化的差异；

**💡 创新点**

通过对比预训练与随机初始化的同一模型架构，揭示预训练对性能的真实贡献相对有限，并强调将随机初始化基准纳入评估的重要性；

**🔧 技术方法**

使用了自监督预训练（掩码重构、对比学习、BYOL等）产生的预训练权重，冻结编码器后分别与岭回归、逻辑回归、RBF SVM 和随机森林四类下游分类器结合；

**📊 数据集**

在UEA多变量时间序列分类档案（30个数据集）上进行实验，并对ROCKET、Hydra、Quant等时间序列专用方法进行对照；

**📈 对比分析**

与随机初始化的同一模型对比，预训练平均提升 1.43-5.19%（总体 2.58%）；单独显著的只有 NuTime；与专用方法相比，ROCKET、Hydra、Quant 在平均准确率上更优；

**⚠️ 局限性**

仅评估冻结表示，未考虑微调效果；数据集覆盖因计算限制不完全一致；对专用方法的比较缺乏完全匹配的实验设置；未测量预训练的时间/能耗成本。

---

## 323. Metacognitive Scaffolds Create Productive Friction for AI-Assisted Knowledge Work

**arXiv ID:** 2610.30945 | [PDF](https://arxiv.org/pdf/2610.30945v1)

**作者:** Viktor Kewenig `[一作]` (Microsoft Research), Abigail Sellen `[通讯]` (Microsoft Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一次对297名知识工作者的随机对照实验中，研究者开发并评估了一套动态元认知支架，用于在生成式AI写作任务中帮助用户明确目标和拆分任务，并比较其与标准聊天界面的效果。

**💡 创新点**

创新点在于将元认知支架与AI接口实时联动，动态生成与当前任务上下文匹配的目标设定和任务拆分控件，验证其对写作质量、提示质量和后续记忆的正向影响，并探讨“可生产性摩擦”与用户感知之间的矛盾。

**🔧 技术方法**

主要技术包括：基于GPT‑5.3的对话式写作接口、动态生成界面控件的后端服务、GPT‑5.4作为自动评判者、对提示序列进行的手工/LLM编码、NASA‑TLX认知负荷测量以及主题回忆问卷。

**📊 数据集**

数据集由两项开放式写作任务组成（城市规划提案和混合工作模型咨询），共594篇文章；实验还收集了任务前后问卷、提示日志以及三天后的主题回忆结果。

**📈 对比分析**

方法是随机分配到带支架与无支架两组，采用线性混合效应模型和逻辑回归评估写作质量、提示清晰度、认知负荷和记忆。结果显示：写作质量提升（Hedges g≈0.29/0.21），提示质量和主题回忆均显著提高；但参与者报告更高的认知负荷和更低的AI有用性感知。

**⚠️ 局限性**

局限性包括：自动评判者对结构偏好导致的评分偏差、评判者与人工评分的中等一致性、主题回忆调查高流失率、缺乏静态支架对照、仅测试短期迁移、样本为线上受试者，缺乏长期生态部署验证。

---

## 324. ReflectPO: Reflection-Guided Policy Optimization for Long-Horizon LLM Agents

**arXiv ID:** 2610.30933 | [PDF](https://arxiv.org/pdf/2610.30933v1)

**作者:** Rumei Li `[一作]` (Meituan), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于反思的策略优化方法ReflectPO，用于长周期任务中的错误检测与自我恢复

**💡 创新点**

创新点包括利用对比与循环采样生成反思提示，以及树结构信用分配，实现对原始轨迹与恢复轨迹的精细优势估计

**🔧 技术方法**

采用强化学习、树形信用分配、对比采样、循环采样、LLM引导提示等技术

**📊 数据集**

在BFCL‑v3 Multi‑Turn和τ²‑Bench这两个工具使用基准上进行实验

**📈 对比分析**

与PPO、GRPO、Reinforce++、GiGPO、CRITIC等多种RL基线对比，ReflectPO在两种模型规模下平均分数最高，并显著提升错误恢复率

**⚠️ 局限性**

仍受限于需要额外的引导模型、对错误分类的覆盖度有限以及在某些错误类型上的恢复效果不均衡

---

## 325. Making MLLM Latents Matter: Factor-Aware Predictive Learning for Cinematic Video Generation

**arXiv ID:** 2610.30926 | [PDF](https://arxiv.org/pdf/2610.30926v1)

**作者:** Songlin Yang `[一作]` (Hong Kong University of Science and Technology), Anyi Rao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 Factor-Aware Predictive Learning (FAPL)，通过三阶段训练使视频 Diffusion Transformer 能够从文本脚本和稀疏视觉参考中推断时间细节，并实现角色、场景、动作和摄像机等因素的精细控制。

**💡 创新点**

创新点在于：①将 MLLM 表征作为唯一条件进行生成器 grounding；②通过配对视频的因子跨度替换显式监督因子级变化与非目标保持；③在固定生成器的前提下，使用稀缺与完整观察共享目标进行预测学习，强化 MLLM 的生成兼容表示。

**🔧 技术方法**

技术上采用多模态大型语言模型（MLLM）与 Video Diffusion Transformer (DiT) 的连接器，流匹配训练，因子跨度替换策略，以及混合观察（文本仅 vs 文本+完整视频）预测优化。

**📊 数据集**

数据集主要为 CoinVE-200K、OmniVBench 训练数据以及 100K 质量过滤的源-目标视频对，用于三阶段训练；测试采用 CineFactor-10K、ShotVerse-Bench 等。

**📈 对比分析**

与 Omni-Video 2（1.3B/14B）和 UniVideo（13B）对比，FAPL 在稀疏到视频生成任务中动作顺序提升至 61%（↑25%），相机平移误差下降至 0.0586（↓53%），∩-Acc 提升至 58%（↑20%），多镜头一致性（FaceSim-Arc）提升至 0.523（↑27%）。

**⚠️ 局限性**

局限性在于仍受限于底层 MLLM 的推理能力与 DiT 的视觉细节与复杂动作生成能力，难以处理复杂 3D 视图或逼真表演，且模型规模较大，需进一步提升。

---

## 326. REGATTA: A Cross-Domain Measurement Framework and Fault-Injection Harness for Enterprise AI Agents

**arXiv ID:** 2610.30924 | [PDF](https://arxiv.org/pdf/2610.30924v1)

**作者:** Luong Tuan Thanh `[一作]` (Golden Gate University), Abhijit Sanyal `[通讯]` (Novartis Healthcare Pvt. Ltd.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了REGATTA框架，用于跨领域测量企业AI代理的合规性并提供故障注入测试平台。

**💡 创新点**

创新点在于：①将法规要求与源引用清单、注入故障语料和两阶段评估工具集成为可复用的测量体系；②设计了四项诊断指标（V1–V4）评估覆盖度、检测稳定性、阶段差异与策略区分；③在美国与越南的五个行业-司法域上验证该框架。

**🔧 技术方法**

采用了LLM生成器（Claude Sonnet 4、Qwen 2.5 72B、Gemma 4 26B）、两阶段评估器（设计阶段与执行阶段）、人工智能评判（Claude、GPT‑4o）以及源引用审计工具。

**📊 数据集**

使用了125条法规条目（5个域×25条）、25个注入故障（每域5个）、180套测试场景（30条/套，共5,400条），并通过三种生成策略（S1–S4）产生。

**📈 对比分析**

与传统代理基准（AgentBench、τ‑bench、AIR‑Bench）相比，REGATTA能够量化检测率、覆盖度及阶段一致性；在V2、V3、V4诊断中显示出覆盖差异、检测稳定性和策略可区分性，证明其在不同域和模型上的可迁移性。

**⚠️ 局限性**

主要局限包括：①评估依赖单一人工专家与LLM判别，缺乏多专家一致性验证；②仅测试单一注入代理，执行阶段结果可能不适用于其他代理；③法规条目和故障语料存在错误与来源不确定，需要进一步法律审查；④可能存在评判器与生成器之间的亲和性影响；⑤对对抗性类别的分辨率不足。

---

## 327. Watch and Learn: Teaching Visual In-Context Learning by Removing Textual Shortcuts

**arXiv ID:** 2610.30913 | [PDF](https://arxiv.org/pdf/2610.30913v1)

**作者:** Nimrod Shabtay `[一作]` (IBM Research), Raja Giryes `[通讯]` (Tel-Aviv University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个跨任务、无文本规则的视觉上下文学习数据集 DIC（Multi-source Induction from Mute Episodes）并通过示例文本注意力丢弃（DTAD）对 VLM 进行微调，迫使模型只能通过图像示例推断隐藏规则。

**💡 创新点**

创新点在于消除文本捷径、让模型真正从视觉示例中进行归纳推理，并通过随机丢弃示例标签来显式地将注意力转移到图像上，从而显著提升视觉 ICL 能力。

**🔧 技术方法**

采用了 LoRA 微调、注意力重分配（DARA/MGI 对比）、示例文本注意力丢弃（DTAD）、可视化分析和错误率下降评估等技术手段。

**📊 数据集**

使用了多源数据集（Polygon、Bongard、ARC、Triangle Arithmetic、CLEVR 等）构成 DIC，同时在 TrueMICL、VL-ICL-Bench、SEED-Bench‑2 等公共视觉 ICL 评测集上进行评估。

**📈 对比分析**

与 DARA、MGI 等基线相比，DIC+DTAD 在 Qwen3‑VL‑8B、LLaVA‑OV‑1.5‑8B、Granite‑Vision‑4.1‑4B 三大模型上平均提升 8.1–18.0 分；在 TrueMICL 上，Qwen3‑VL‑8B 能匹敌甚至超过 30B 级前沿模型；在多图像、文本 ICL 任务上也获得显著提升。

**⚠️ 局限性**

局限性包括：仅在 8B/4B 模型上使用 LoRA 微调，未在更大模型或预训练阶段验证；训练管道对最新 VLM 公开度有限，无法评估在完整预训练中的效果；未与其他推理时方法（如任务向量、检索策略）结合。

---

## 328. Hierarchical Fine-Grained Cross-Modal Alignment via Dynamic Granularity Discovery and Optimal Transport

**arXiv ID:** 2610.30912 | [PDF](https://arxiv.org/pdf/2610.30912v1)

**作者:** Pengfeng Song `[一作]` (Zhejiang Wanli University), Pengfei Shao `[通讯]` (Zhejiang Wanli University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种分层细粒度跨模态对齐框架（HFCMA），通过动态粒度发现将图像补丁和文本词元聚合为自适应的视觉区域和文本短语，并用熵正则化的最优传输（OT）在这些语义单元之间进行全局一致的匹配；

**💡 创新点**

①将低级补丁/词元动态聚合为语义完整的视觉/文本单元；②利用OT实现多对多的结构化对齐，避免局部最大化的独立选择；③加入背景去偏、文本重要性加权和原型一致性正则化提升匹配鲁棒性；

**🔧 技术方法**

冻结CLIP双编码器、槽位（Slot）聚合机制、软赋值、Sinkhorn迭代求解熵正则OT、对比损失与原型一致性损失；

**📊 数据集**

Flickr30K（1K测试）和MS-COCO（1K、5K测试）两个公开图文检索数据集；

**📈 对比分析**

与VSE++、SCAN、VSRN、IMRAM、NAAF、CHAN、HREM、LAPS等基线在Flickr30K和MS-COCO上对比；在Flickr30K-1K上达到 top‑1 87.0，ΣR 542.92，超越LAPS（85.1，ΣR 545.3）；在MS-COCO 1K/5K上获得与CHAN、HREM相当或略优的 Recall@K；

**⚠️ 局限性**

槽位并非精确检测框，可能导致在拥挤或遮挡场景下语义漂移；校准依赖候选分布，需在验证集上固定；仅在英文短文本上验证，未扩展到多语种或长文本；对CLIP骨干的依赖使性能受限于预训练模型。

---

## 329. TwinStore: A Synthetic Consumer Platform Using Vision--Language Models

**arXiv ID:** 2610.30909 | [PDF](https://arxiv.org/pdf/2610.30909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 330. Inherited, Not Invented: Tracing the Provenance of Wait in Reasoning Models

**arXiv ID:** 2610.30882 | [PDF](https://arxiv.org/pdf/2610.30882v1)

**作者:** Zhizhuo Kou `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

追踪并量化开源大型语言模型在预训练、mid‑training、RL‑Zero 与 fine‑tune 等不同阶段中修正标记“wait”的出现、演变以及其与强化学习的关系。通过无标签的语言学指标和注入错误的信号检测，系统评估了 RL 在该标记上的真实影响。

**💡 创新点**

提出了无需人工标签的词频、位置熵、词形一致性和用法分布等指标，以及基于注入算术错误的 d′/c 信号检测方法，首次将其应用于多条开放训练线索，揭示 RL 并非“生成”修正标记，而是对已有标记的放大或衰退。该方法为追踪模型内部语言演化提供了可复现、可扩展的工具。

**🔧 技术方法**

标签自由的统计指标（频率、句首占比、位置熵、词形保持率、连词链比例、句子长度、Jensen–Shannon 散度）、注入错误警报检测（d′/c）、聚类分析、采样输出与多种解码设置的对比，以及对公开模型检查点的统一处理。

**📊 数据集**

公开的数学评测数据集（MATH‑500、AIME、AMC、GSM8K、MMLU‑Pro 等），以及包含推理轨迹的训练数据；同时采集并分析了 OLMo‑3、Qwen、MiMo、SmolLM3、Open‑Reasoner‑Zero 等开放模型的预训练、mid‑training 与 RL‑Zero 检查点。

**📈 对比分析**

通过对比不同训练阶段（无 RL、RL‑Zero、SFT+RL、GRPO 等）在上述指标上的变化，发现零 RL 几乎不产生“wait”，但在已有标记的模型上 RL‑Zero 可显著放大其出现率；RL‑Zero 在 OLMo‑3 中导致标记形式衰退、敏感度提升；相反，KL‑free GRPO 在相同基线上不放大标记。整体表现表明 RL 主要通过调整标记的频率和语义而非生成新标记。

**⚠️ 局限性**

研究基于观察数据，模型线索与数据集、训练 recipe 混杂，难以单独归因；仅针对英语数学任务，未涵盖多语言或其他领域；注重标记本身，无法捕获不写“wait”却仍能反思的行为；LLM 判定结果可能与人工一致性低；采样误差导致置信区间受限；未评估更大规模 RL、不同奖励设定或过程奖励。

---

## 331. SnakeCharmer: Orchestrating Utility, Timeliness, and Placement in GPU Prefetching

**arXiv ID:** 2610.30990 | [PDF](https://arxiv.org/pdf/2610.30990v1)

**作者:** Yuhan Tang `[一作]` (National University of Defense Technology), Siqing Fu `[通讯]` (National University of Defense Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出SnakeCharmer后端框架，统一控制GPU预取的实用性、时序和放置，提升预取收益。

**💡 创新点**

创新点在于：①零值过滤预测候选，②基于反馈的多层距离调节实现时序优化，③利用闲置寄存器缓冲预取数据，三者协同解决预取的实用性、时序与放置瓶颈。

**🔧 技术方法**

使用了零值可信度计数器、层次化自适应窗口调节（AWT）、分数门控单元（SGU）、寄存器映射表（PRT）等硬件机制，并集成零压缩缓存。

**📊 数据集**

在14个CUDA应用（高低稀疏区分）上评估，包含多种GPU工作负载，使用A100模拟器进行周期级仿真。

**📈 对比分析**

与Snake、APOGEE及其改进版本对比，在A100上IPC提升12.8%（能效提升20.8%），相对于APOGEE则提升1.7%/3.2%，并在V100上同样实现正向加速。

**⚠️ 局限性**

局限性包括：依赖前端预测质量、寄存器闲置容量有限、仍受L1容量、MSHR、缓存端口和带宽竞争影响，且实现硬件开销仍需权衡。

---

## 332. UniFlowWAM: A Unified World-Action Model with Visual Semantics and 4D Geometry

**arXiv ID:** 2610.30991 | [PDF](https://arxiv.org/pdf/2610.30991v1)

**作者:** Jialin Tian `[一作]` (MVIG@SJTU), Lixin Yang `[通讯]` (MVIG@SJTU)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的世界-动作模型UniFlowWAM，将视觉特征、场景级3D流与动作三种模态整合在同一扩散变压器中，实现对3D流的监督与使用；

**💡 创新点**

将场景级3D流视为与视觉、动作同等的重要模态，并通过十目标角色网格实现多任务联合训练，使模型既可直接生成动作，又能学习前向/逆向动力学和未来视觉/流预测；

**🔧 技术方法**

使用多模态扩散变压器（MMDiT）+专属3D流专家，4D VAE对点云与流进行编码，MM‑RoPE做时空位置编码，Qwen3‑VL‑4B提供视觉‑语言跨注意力，DINOv3提供视觉目标，采用流匹配损失与角色条件多目标训练；

**📊 数据集**

在中训练阶段使用三大公开机器人演示集（RoboCOIN、RoboMIND、MolmoAct2‑BimanualYAM）共214k条演示；在仿真阶段评测RoboCasa‑GR1、LIBERO、LIBERO‑Plus；在真实机器人上使用AgileX Cobot Magic完成四个双臂操作任务；

**📈 对比分析**

与基线LDA、π_0.5等方法对比：RoboCasa‑GR1成功率从52.1%提升至56.4%；LIBERO保持98.1%成功率；LIBERO‑Plus零样本成功率达87.7%；真实机器人上平均成功率从68.5%提升至87.0%，显著优于其他基线；

**⚠️ 局限性**

局限性包括：3D流标签来自离线跟踪，存在噪声与时序不一致；3D流专家训练数据量相对不足，未达到LDA预训练规模；对高质量流提取的依赖使方法受限于跟踪模型性能；

---

## 333. Query-Time Calibration of CLIP Features for Open-Vocabulary 3D Segmentation

**arXiv ID:** 2610.30908 | [PDF](https://arxiv.org/pdf/2610.30908v1)

**作者:** Dominik Engel `[一作]` (King Abdullah University of Science and Technology), Ivan Viola `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在不改变几何或特征提升的前提下，对场景中CLIP特征进行场景中心化和文本白化，并使用软聚类与CSLS校准相似度，实现开源词汇3D分割；

**💡 创新点**

创新点在于通过场景条件的校准（中心化+文本白化）以及固定参考词汇来提高视觉特征与文本查询的相对分离度，从而显著提升分割mIoU，无需重新训练或重新编码；

**🔧 技术方法**

技术包括：3D高斯重建、CLIP特征提升、场景中心化、文本原型白化、软聚类多类预测、CSLS带相关性阈值的单类预测；

**📊 数据集**

使用四个室内数据集（未列明具体名称），以及七种不同的提升方法进行评估；

**📈 对比分析**

与原始基线（直接使用cosine相似度）相比，校准后在所有提升方法和数据集上均能提升多类和单类分割的mIoU；评估方法为查询时校准，性能提升显著；

**⚠️ 局限性**

局限性包括仅在室内场景验证，参考词汇固定且不适应不同环境；单查询评估仅使用已知类别，无法有效拒绝不存在或任意自然语言查询；校准无法重建缺失表面或恢复缺失语义信息。

---

## 334. Bridging Number-Theoretic Primality Testing And Algorithmic Optimization: An Adaptive Cryptographic Framework

**arXiv ID:** 2610.30935 | [PDF](https://arxiv.org/pdf/2610.30935v1)

**作者:** Tran Minh Thanh `[一作]` (University of Science), Vo Hoai Viet `[通讯]` (University of Science)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自适应的质数检测框架，将传统数论方法与机器学习、硬件优化和量子随机性结合

**💡 创新点**

创新点在于利用监督学习智能选择 Miller‑Rabin Witness、按 L1 缓存调优分段筛选以及量子熵保障无偏分布，并提供可验证的 fastECPP 证明链

**🔧 技术方法**

使用 Montgomery/Barrett 多精度运算、Miller‑Rabin、Baillie‑PSW、fastECPP、随机森林、贝叶斯优化及 Python/NumPy 进行实验验证

**📊 数据集**

实验使用 500 个 1024‑bit 随机候选数（含合成和质数）及基准比较

**📈 对比分析**

与传统 Miller‑Rabin、Solovay‑Strassen 及无 ML 基线对比，ML 选择 witness 将单次指数运算成本降低至 27.6%，整体检测时间显著下降

**⚠️ 局限性**

局限性包括 ML 模型需保持低开销且对极端伪素数无严格理论保证，硬件特化导致迁移性受限，且量子熵源成本与可用性限制

---

## 335. GraphCAD: Geometry-Aware Diffusion on Edge-Adjacency Graphs for CAD Generation

**arXiv ID:** 2610.30974 | [PDF](https://arxiv.org/pdf/2610.30974v1)

**作者:** Wenhao Zheng `[一作]` (Sichuan University), Jiancheng Lv `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GraphCAD，一种基于边缘邻接图（EAG）的几何感知生成框架，用于高质量CAD模型的自动构建。

**💡 创新点**

创新点在于：①将B-rep的边作为图节点，构造EAG；②设计混合图扩散过程同时处理离散拓扑和连续几何；③引入几何渲染机制，将节点属性转化为3D曲线点集并通过SE(3)-等变向量神经元编码；④构建EAGBench大规模数据集。

**🔧 技术方法**

采用图扩散模型、混合离散/连续扩散、SE(3)-等变向量神经元、Bezier曲线渲染、点采样、交叉熵与噪声回归损失、B-rep重建管线等技术。

**📊 数据集**

使用EAGBench，其中包含EAGBench-DC（DeepCAD）约95k模型和EAGBench-FS（Fusion 360 Gallery）约2.8k模型。

**📈 对比分析**

与GraphNVP、GDSS、ConGress、DiGress等图生成基线及DeepCAD、Text2CAD、TGBFN等序列基线进行比较；在有效性、唯一性、创新性、MMD、CD、HD、IR等指标上，GraphCAD在EAGBench上实现了最高或接近最高的有效率和几何精度，尤其在复杂模型上的性能明显优于基线。

**⚠️ 局限性**

局限性包括：只生成1D边界曲线而非完整2D曲面，后期重建可能产生误差；扩散过程需要多步，推理速度慢；对大型装配和极大节点数的EAG生成仍面临挑战。

---

## 336. IORA: Irregular Object Immobilization under Restricted Actuation

**arXiv ID:** 2610.30969 | [PDF](https://arxiv.org/pdf/2610.30969v1)

**作者:** Sang-Hun Han `[一作]` (Gwangju Institute of Science and Technology), Hae-Gon Jeon `[通讯]` (Yonsei University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种支持表面感知的低自由度手抓取合成框架，实现了在桌面支持下的对象固定与任务执行。

**💡 创新点**

关键创新是将桌面表面作为主动接触来源，与低自由度手共同生成可执行的抓取姿势，并通过图匹配、几何细化与物理验证实现鲁棒性。

**🔧 技术方法**

采用图结构的接触匹配、局部几何优化、物理仿真+力平衡分析，以及可选的任务引导局部微调等技术。

**📊 数据集**

在DexGraspNet测试集200个随机物体以及实物水果蔬菜样本上进行验证。

**📈 对比分析**

与Dexonomy、DexGraspNet、BODex等现有方法在RH56和Shadow Hand上进行 MuJoCo 评估，结果在桌面支持下GSR/OOSR最高，最大抗扰力提升显著，实验显示成功率约94%及高达30N抗扰力。

**⚠️ 局限性**

仅限已知物体几何和固定桌面支撑，未考虑动态环境、感知误差和更复杂的支撑关系。

---

## 337. A Transparent 3-DoF Assistive Device for Upper-Limb Tremor and Ataxia Compensation: Design and Experimental Validation

**arXiv ID:** 2610.30955 | [PDF](https://arxiv.org/pdf/2610.30955v1)

**作者:** Thomas Dobosz `[一作]` (University of Stuttgart), Daniel Haeufle `[通讯]` (University of Tubingen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出并验证了iAssistADL平台，一种三自由度、基于机械透明度的辅助装置，旨在为日常生活活动中的震颤与共济失调提供动力补偿。

**💡 创新点**

创新点在于将所有致动器放置于基座，采用球‑齿、剪式和电缆传动实现轻质、低惯量结构，并引入混合控制策略：水平面采用开环雅可比转置映射实现高透明度，垂直轴采用闭环PI力反馈解决摩擦带来的不可见阻尼。

**🔧 技术方法**

技术实现包括球‑齿传动、剪式升降机构、双电缆Capstan传动、被动弹簧重力补偿、雅可比转置开环控制、单轴负载传感器、PI闭环控制、实时dSPACE控制框架以及外部ATI六轴力/扭矩传感器用于性能评估。

**📊 数据集**

实验使用硬质实验台与外部ATI传感器进行力渲染测试，选取三种典型ADL姿态（握瓶、倒水、喝水），对步进、正弦及频率扫描等多种激励进行测量。

**📈 对比分析**

与传统可视化或阻尼式辅助装置相比，iAssistADL在水平面实现了<0.1 Nm的阻尼、最高10 N的可渲染力以及≤4 Hz的精准力跟踪；垂直轴由于摩擦导致闭环性能在>2 Hz出现共振，但在静态与低频下仍能准确渲染。

**⚠️ 局限性**

主要局限包括垂直轴的球‑齿摩擦导致的粘滑与共振、整体系统在动态下存在≈140 ms的机械传输延迟、以及目前实验仅在无人体的刚性平台上进行，缺乏真实人机交互验证。

---

## 338. Ground-Truth-First Benchmarking of Reinforcement Learning via Converse Optimality: From Smooth, Fully Observed Systems to Non-Smooth or Partially Observed Tasks

**arXiv ID:** 2610.30954 | [PDF](https://arxiv.org/pdf/2610.30954v1)

**作者:** Grégoire Ouerdane `[一作]` (Skoltech), Pavel Osinenko `[通讯]` (Skoltech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建可知最优解的强化学习基准（ground‑truth‑first），并通过对比真实最优值来衡量学习算法的绝对性能与稳健性。

**💡 创新点**

提出两种基准模式：连续时间敏感度评估（生成可调参数的已知最优系统）和离散时间oracle‑referenced对比（利用相同转移律的环境与最优对照），并扩展到非线性/非高斯、非光滑、部分可观测情形。

**🔧 技术方法**

使用逆最优控制（converse optimality）来构造动力学、成本与最优策略；对非光滑情况采用黏度解/ Clarke 子微分；利用虚拟输入拉回（virtual‑input pullback）处理非线性原始动作；信息状态方法实现部分可观测下的最优输出反馈；评估指标包括最优性间隙、累计后悔、Bellman 残差与策略误差，并通过 CRN 产生置信区间。

**📊 数据集**

在一系列合成任务上验证：串联多关节平面臂、动态扩展的非完整车辆、带非高斯风扰的无人机、以及基于非光滑价值函数的二维分段二次型；使用统一的随机种子与固定初始状态集合进行对比实验。

**📈 对比分析**

通过在连续时间模式下对 Adam vs. RAdam 的优化器敏感度实验，发现优化器显著影响子最优性方差；在离散时间模式下对 PPO、A2C、SAC、TD3、DDPG 的绝对评估，结果显示算法排名依赖任务的初始状态与延迟调度，非光滑任务下离线方法（SAC/DDPG）优于在线方法（PPO/A2C）。

**⚠️ 局限性**

局限性包括：连续时间生成器受数值积分步长与方案影响；非光滑构造需要手工验证超参数（如增益K）；只支持有限协方差非高斯扰动；缺乏对时变或多任务环境的理论支持；部分可观测扩展仅在 LQG 情形下给出解析信息惩罚。

---

## 339. OpenAttention: One Percent Is Enough with Tail Proxies

**arXiv ID:** 2610.30951 | [PDF](https://arxiv.org/pdf/2610.30951v1)

**作者:** Hyunwoo Yu `[一作]`, Suk-Ju Kang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大上下文下实现稀疏注意力，通过使用尾部代理（proxies）来近似未读取键值对的分布，从而避免了传统方法中的丢弃或采样误差。

**💡 创新点**

核心创新是：① 通过查询几何聚类(QGC)在投影空间聚类键值对；② ErrorGate 根据估计权重误差分配读取预算；③ 用计数感知均值代理替代尾部采样；④ 定制的 FlashAttention-2 内核实现高效解码。

**🔧 技术方法**

采用查询几何聚类(QGC)、Query-Geometry Selector(QGS)、ErrorGate、计数感知均值代理以及 Triton 定制的 FlashAttention-2 解码内核。

**📊 数据集**

主要使用 RULER‑32K HARD 合成数据集，并在 Llama‑3.1‑8B、DeepSeek‑R1‑Distill‑Llama‑8B、Mistral‑7B 等模型以及 16K–96K 的上下文长度上进行实验。

**📈 对比分析**

与 HashAttention、SparQ、Loki 等稀疏注意力基线相比，OpenAttention 在 1% 读取率下保持 99% 的稠密注意力精度，RULER‑32K HARD 上 1% 读取率得到 86.30% 的准确率，远优于最佳基线 48.2%；在 3% 读取率下也保持 5–10 倍的解码速度提升。

**⚠️ 局限性**

局限性在于对聚类内部方差仅使用对角近似，导致无法完全恢复精确误差，且在极低读取率下仍可能面临大规模键值对的处理开销。

---

## 340. PAIR: Preference-Aligned Intent Retrieval for Premium People Recommendation at LinkedIn

**arXiv ID:** 2610.30941 | [PDF](https://arxiv.org/pdf/2610.30941v1)

**作者:** Zhongkai Shangguan `[一作]` (LinkedIn), Zhengming Xing `[通讯]` (LinkedIn)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对LinkedIn Premium Browsemaps产品，提出并实现了Preference‑Aligned Intent Retrieval（PAIR）检索框架，利用曝光条件的偏好学习改造检索嵌入空间，并在三塔结构中引入可更新的点击历史表示。

**💡 创新点**

创新点包括①将用户点击视为相对偏好，对比传统的相似度匹配；②在三塔架构中实现了可离线预计算的个人历史向量，并通过可学习的加权位置注意力实现动态意图融合；③采用Matryoshka多维度训练，允许单一模型支持不同的嵌入尺寸，实现低延迟与高质量的灵活切换。

**🔧 技术方法**

技术手段：Qwen‑Embedding 0.6B LLM作为编码器；多负样本对比学习（Exposure‑Conditioned Preference Contrastive）；三塔结构与加权加法融合；位置注意力历史聚合；Matryoshka多维度损失；BF16训练与DeepSpeed分布式加速。

**📊 数据集**

使用LinkedIn Premium Browsemaps的历史生产日志：训练集209k个查询实例、4.4M唯一成员；测试集11.8k查询实例、93k成员；每实例包含1个点击正样本和3个负样本（曝光未点击与随机）。

**📈 对比分析**

与生产的Holistic‑171嵌入和未微调的0.6B基线对比；离线指标MAR@100从0.288提升至0.450（+56%）；在线A/B实验中CTR提升7.5%，Premium留存率提升2.1%，几乎无延迟与覆盖率回归。

**⚠️ 局限性**

局限性：依赖曝光偏好数据，仍受历史检索策略的选择偏差影响；实验仅在Premium Browsemaps上验证，其他推荐场景需进一步测试；与下游排序器未联合优化，可能导致部分收益被稀释；长期效果与更大规模多负采样机制仍待研究。

---

## 341. The Geometry of Meaning in Neural Manifolds

**arXiv ID:** 2610.30938 | [PDF](https://arxiv.org/pdf/2610.30938v1)

**作者:** Valentino Maiorca `[一作]` `[通讯]` (Sapienza University of Rome), Valentino Maiorca (Sapienza University of Rome)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了通过对应关系将独立训练的模型嵌入共享语义空间，并设计了对应平行与对应耦合两类变换算子，以实现跨模型、跨层甚至跨分布的无监督翻译与直接解码兼容。

**💡 创新点**

核心创新在于构造可无监督扩展的共享语义坐标（相对表示与产品空间），以及将对应两侧信息联合使用的耦合算子，使得输出能够直接落入目标模型的原始坐标系，兼顾模块化与解码器兼容性。

**🔧 技术方法**

采用相对角度投影、最佳传输对齐、正交变换、谱算子、注意力头选择、残差流分解等技术，并结合跨模态对应与无监督扩展策略。

**📊 数据集**

以大规模图像‑文本对（如CLIP规模的数百万对）为主，也在单模型内部使用标准视觉或文本任务数据进行实验。

**📈 对比分析**

通过与传统共训练或显式映射方法比较，在无监督跨模型翻译、语义检索与生成等任务中达到或超过现有基准，同时保留原始解码器性能，验证了方法的有效性。

**⚠️ 局限性**

主要局限在于需依赖足够多且质量可靠的对应锚点；耦合算子牺牲了模块化，需要为每对目标空间训练逆映射；且在极端分布或大模型上的推理成本仍相对较高。

---

## 342. UserArena: Benchmarking Interactive User Simulation for Long-Horizon Shopping Agents

**arXiv ID:** 2610.30932 | [PDF](https://arxiv.org/pdf/2610.30932v1)

**作者:** Jujia Zhao `[一作]` (Leiden University), Zhaochun Ren `[通讯]` (Leiden University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了UserArena交互式基准和On-Policy Trajectory Distillation（OCT）方法，用以评估和提升长时序用户模拟器的真实性与可靠性。

**💡 创新点**

创新点在于提供包含64,729条跨会话长轨迹、完整动作空间以及生产A/B验证的真实电商数据，并通过轨迹意图驱动的密集教师监督实现更高的轨迹一致性。

**🔧 技术方法**

采用大型语言模型（Qwen3.5-397B）提取轨迹意图，结合教师-学生框架的对数概率插值信任域蒸馏，以及Qwen3.5-2B作为基准模型进行实验。

**📊 数据集**

使用了UserArena基准数据集，涵盖11,010用户、566,102商品、13种动作、64,729会话轨迹及对应的跨会话序列和生产A/B实验结果。

**📈 对比分析**

在单会话、跨会话以及A/B验证三种评估场景下，采用五级评估协议（L0–L4）与文本/GUI模拟器对比，OCT在序列行为、个性化和结果一致性上均优于基线，并最贴近生产政策效果。

**⚠️ 局限性**

仍面临行为校准不足、监督稀疏以及曝光偏差等限制，且对更大规模、多模态环境的长期动态预测仍需进一步验证。

---

## 343. When Code Gets Cheap, Evaluation Gets Scarce: An Experience Report on AI-Assisted Agile Software Engineering

**arXiv ID:** 2610.30916 | [PDF](https://arxiv.org/pdf/2610.30916v1)

**作者:** Barbara Weber `[一作]` (University of St.Gallen), Marco Franceschetti `[通讯]` (University of St.Gallen)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

作者利用大型语言模型助手，构建了一个完全无手写代码、可审计、符合合规的考试评分系统，并在三天内完成初版，随后部署到真实班级。

**💡 创新点**

本文首次系统化地揭示了 AI 生成代码成本下降后，评估和判断成为新的瓶颈，并提出了门控可追溯构建流程、可识别需求与可发现需求分离等五条经验教训。

**🔧 技术方法**

采用Claude/Anthropic 和 OpenAI LLM 作为编码助手，结合 Streamlit + Python 架构，自动化测试、持续集成、决策记录（ADR）、审计日志等技术。

**📊 数据集**

以真实学生的 27 人、324 条答案的考试数据为核心测试集，另外使用合成数据进行演示，并在后续项目中使用多学科作业的真实数据。

**📈 对比分析**

未进行量化性能对比，仅通过工时、缺陷率和人工评估门限进行经验性评估，结果显示三天内实现可交付功能，缺陷率显著降低。

**⚠️ 局限性**

局限性包括：单一开发者（开发者即用户与领域专家）；仅使用同一系列 Claude 模型；报告为自我观察，缺乏独立复现与客观评估。

---

## 344. Beyond Phenotypic Similarity: Relation Refinement and Selective Population Graph Learning for Multimodal Brain Disorder Diagnosis

**arXiv ID:** 2610.30902 | [PDF](https://arxiv.org/pdf/2610.30902v1)

**作者:** Yinyu Lan `[一作]` `[通讯]` (Harbin Institute of Technology), Yinyu Lan (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出 PopDRP 框架，将诊断引导的相似性校准、神经图纠正和选择性金字塔状态空间学习结合，完成多模态脑疾病诊断。

**💡 创新点**

创新点在于三阶段解耦：CAR 用诊断信息精炼属性相似性；NGR 通过最小充分表示去噪图结构；SPSS 采用输入自适应的状态空间模型与 U‑Net 树形池化，形成多尺度、层次化的群体图学习。

**🔧 技术方法**

使用对比学习、VAE 降噪、递归特征消除、TransformerConv、输入自适应状态空间模型、双视图对齐以及图池化/反池化等技术。

**📊 数据集**

实验数据集为 ABIDE（自闭症）和 ADHD‑200（多动障碍）两大 rs‑fMRI 公开数据集，配合年龄、性别、扫描站点等非影像属性。

**📈 对比分析**

与 SVM、MLP、Brain‑GNN、DGCN、AL‑NEGAT、A‑GCL、Pop‑GCN、GATE、EV‑GCN、MMGL、MM‑GTUNets 等传统机器学习、脑图和群体图方法比较，10 折交叉验证下 PopDRP 在准确率、AUC、敏感度等指标上均取得最高或次之的最优表现。

**⚠️ 局限性**

局限性包括：仅针对二分类任务，未扩展到多分类或连续症状预测；使用固定非影像属性，缺乏动态/纵向信息；实验仅在内部交叉验证，缺乏外部独立数据验证。

---

## 345. RetiJev: Fast and Source-Grounded Ophthalmic Decision Models

**arXiv ID:** 2610.30947 | [PDF](https://arxiv.org/pdf/2610.30947v1)

**作者:** Bo Liu `[一作]` (National University of Singapore), Huazhu Fu `[通讯]` (Agency for Science, Technology and Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个来源可靠的眼科决策数据集，并基于此训练了一系列可在单 GPU 上高效推理的决策模型；

**💡 创新点**

将临床决策转化为候选答案评分任务，利用源文本监督和多任务学习显著提升规则应用和行动选择的准确性；

**🔧 技术方法**

采用 JEV 风格的指针头（pointer head）与 Qwen3.5 大模型的 LoRA 微调，结合教师蒸馏、顺序判定和同义词一致性训练；

**📊 数据集**

使用从 PubMed Central 文献和 MeSH 眼科疾病词汇提取的规则，以及从公开分级系统生成的合成病例，形成 Clinical Decision、Diagnosis 与 Guideline Grading 三大任务；

**📈 对比分析**

与 27B LLM 及商业 Jev 1.13 进行基准对比，所有模型规模均在分级和决策任务上比基线高 13–18%（决策任务最高提升约 3–4%），9B 模型在诊断识别上达到 92% 以上；推理速度可达 133–158 ms/案例，显著快于 LLM；

**⚠️ 局限性**

数据为合成且仅通过自动化验证，缺乏临床专家裁定；未在真实病人样本上进行前瞻性验证，可能存在分布偏移和误诊风险。

---

## 346. ICGF-CXRMate: Image-Conditioned Clinical Context Compression and Gated Fusion for Chest X-ray Report Generation

**arXiv ID:** 2610.30928 | [PDF](https://arxiv.org/pdf/2610.30928v1)

**作者:** Vu Minh Tran `[一作]` (University of Information Technology), Doanh C. Bui `[通讯]` (University of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种名为ICGF-CXRMate的框架，用以通过图像条件下的临床上下文压缩与门控融合，整合胸部X光检查的异质患者数据以生成更高质量的放射科报告；

**💡 创新点**

创新点在于利用图像条件下的可学习查询对患者信息进行压缩成固定数量的临床上下文标记，并通过门控交叉注意力选择性地融合到视觉表示中，同时保留压缩标记作为直接路径供解码器使用；

**🔧 技术方法**

采用的技术包括UniFormer视觉编码器、Llama语言解码器、可学习查询的多头交叉注意力压缩模块、门控交叉注意力融合模块以及三阶段训练策略（含自我批评序列训练）；

**📊 数据集**

使用了连接的MIMIC-CXR与MIMIC-IV-ED数据集，包含胸部X光图像、放射科报告以及四类辅助患者数据（分诊、调和药物、检查指示和临床历史）；

**📈 对比分析**

通过与原始CXRMate-ED模型在Stage 2和Stage 3进行对比，ICGF-CXRMate在七项临床与语言生成指标上均优于CXRMate-ED，并将平均提示长度减少22.9%，在自我批评训练后在八项指标中六项表现最好；

**⚠️ 局限性**

局限性包括仅在MIMIC内部数据集上评估，缺乏外部验证，以及在某些指标如病理标签一致性和重复避免上仍存在轻微下降。

---

## 347. Low-Latency MoE Inference in Space Data Centers: A Wireless-Power-Transfer Perspective

**arXiv ID:** 2610.30923 | [PDF](https://arxiv.org/pdf/2610.30923v1)

**作者:** Haotian Zheng `[一作]` (University of Hong Kong), Kaibin Huang `[通讯]` (University of Hong Kong)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出SpaceWPT框架，利用卫星间激光无线电力传输来平衡分布式Mixture-of-Experts推理中的能量需求差异。

**💡 创新点**

创新点在于两侧水填充式能量控制策略，结合激光WPT的高效传输，显著降低SpaceDC推理延迟。

**🔧 技术方法**

使用了激光无线电力传输模型、光学通信模型、DVFS计算模型以及两侧水填充优化求解。

**📊 数据集**

使用Mixtral-8x7B和Phi-3.5-MoE在HumanEval数据集的推理实验验证。

**📈 对比分析**

与无WPT、固定比例WPT、单向WPT等基线比较，SpaceWPT在不同配置下可降低约18–29%推理延迟。

**⚠️ 局限性**

限制包括仅考虑星间星门星间的星形拓扑，未考虑多链路或光学复合通信-供电链接，并且实验仅在模拟环境中进行。

---

## 348. When Annotation Patterns Are Learned: Skin-Tone Dependence in Dermatology Concepts

**arXiv ID:** 2610.30901 | [PDF](https://arxiv.org/pdf/2610.30901v1)

**作者:** Y. Li `[一作]`, N. Papanikolopoulos `[通讯]` (University Of Minnesota)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对SkinCon皮肤概念数据集进行审计，发现基于肤色的概念标注偏差，并训练证据概念模型复制该偏差，随后评估了三种基于模型的标签错误检测器是否能发现潜在的标注遗漏。

**💡 创新点**

首次系统揭示肤色依赖的标注偏差，并证明概念模型会继承并放大这一偏差；同时发现现有标签错误检测器在检测红斑遗漏时低于仅依赖疾病分布的基准，提出针对每个亚组进行标注审核的三步策略。

**🔧 技术方法**

使用带疾病固定效应的逻辑回归计算赔率比，构建证据概念嵌入模型(evi‑CEM)进行概念预测，并实现三种错误检测器：自信度(Confident Learning)、训练动态(AUM)以及证据不确定性；采用五折交叉验证、三随机种子与自助法评估性能。

**📊 数据集**

基于Fitzpatrick17k公开图像数据集，其中包含3,230张已标注SkinCon概念的图像，配合SkinCAP临床描述以及Fitzpatrick肤色标签进行实验。

**📈 对比分析**

通过将每个检测信号的AUC与仅使用疾病分布的基准进行配对比较，发现对于红斑检测，所有检测器均低于基准（AUC≈0.48 vs 0.72）；概念模型的赔率比与原始标注一致，证明模型复制了偏差。

**⚠️ 局限性**

局限包括暗色肤肤图像样本不足、仅评估单一概念数据集、所有检测器均基于同一模型导致共性偏差、参考SkinCAP描述可能本身受到疾病先验影响，以及无法定位具体错误标注的图像。

---

## 349. TRACE: Traversability-aware Reactive Navigation via VLM-driven Adaptive Constraints Estimation

**arXiv ID:** 2610.30994 | [PDF](https://arxiv.org/pdf/2610.30994v1)

**作者:** Mario A. V. Saucedo `[一作]`, George Nikolakopoulos `[通讯]` (Luleå University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了TRACE框架，利用视觉‑语言模型（VLM）生成稠密可通过正负自然语言提示融合得到的可通行性地图，并通过百分位分箱与多项式/样条曲线拟合提取行走通道边界，将其作为软约束嵌入非线性模型预测控制（NMPC）实现自主森林导航。

**💡 创新点**

创新点在于：①使用VLM的零样本分割能力通过正负提示进行证据融合，获取更具语义与上下文感知的可通行性评分；②基于百分位分箱的边界抽取方法，避免传统几何阈值导致的误判；③将抽取的可通行性曲线直接转化为NMPC软约束，实现安全、动态可行的路径跟踪。

**🔧 技术方法**

核心技术包括CLIPSeg视觉‑语言模型、Beta分布证据融合、百分位统计边界提取、平滑样条或正则化多项式拟合，以及基于NMPC的软约束规划。

**📊 数据集**

主要使用数据集为GOOSE（多季节的离线越野环境数据），以及在瑞典北部森林中使用Husky机器人进行的实地实验。

**📈 对比分析**

与基准的比较：对比了基于LiDAR的几何边界提取、以及自监督可通行性估计基线；结果显示TRACE在IoU、边界平滑度和表面法向倾斜角一致性方面均显著优于基线，特别在光照变化、季节差异和视觉模糊环境下表现更稳健。

**⚠️ 局限性**

局限性在于：①依赖预训练VLM的视觉分辨率，低光或极度密集植被场景下可能失效；②正负提示的设定对最终可通行性有较大影响，需要人工调优；③实时性能受限于VLM推理速度，若在资源受限平台上需进一步优化；④仅在地面移动机器人上验证，其他平台如步行或多轮系统的适用性尚未充分评估。

---

## 350. ColaBench: Evaluating Memory-Guided Knowledge Acquisition in Personal Agents

**arXiv ID:** 2610.30989 | [PDF](https://arxiv.org/pdf/2610.30989v1)

**作者:** Wenjie Zhang `[一作]` (Jiangnan University), Xiaoning Song `[通讯]` (Jiangnan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究构建了 ColaBench benchmark，利用 Evidence Graph 记录用户长期交互历史、文档曝光与答案需求，评估个人记忆如何指导知识检索与规则应用；同时提出 MemKB 方法，结合记忆摘要与记忆–文档链接来提升检索和推理性能。

**💡 创新点**

创新点包括：①通过 Evidence Graph 使任务的知识与历史依赖可追踪，支持可解释的评估；②设计 MemKB，将状态摘要与文档链接融合，实现在检索时直接利用历史记忆，引导知识获取与规则应用。

**🔧 技术方法**

使用技术包括：Mem0、Graphiti、A-MEM 等记忆构建方案；检索增强生成（Hybrid）结合语义与关键词检索；递归排名融合（RRF）；记忆卡片与文档链接机制；状态摘要提取与检索融合。

**📊 数据集**

数据集：健康与日常生活领域公开文档共 13,033 篇（原始 11,069 + 1,964 新增）；10 个个人档案生成 349 个任务。

**📈 对比分析**

与 No Memory+Hybrid、Graphiti+Hybrid、A-MEM+Hybrid、Mem0+Hybrid 等基线对比，MemKB 在 349 任务上 Accuracy 86.25%，比 Mem0+Hybrid 提升 6.30pp；Read Recall 45.56%（略高）；平均交互轮次 4.07，整体性能显著优于其他方法。

**⚠️ 局限性**

局限性包括：评估仅限健康与日常生活域，使用合成个人历史；未进行临床验证；评测主要依赖自动化，缺乏人工校准；对实时 Web 或其他领域的推广尚待验证。

---

## 351. Zero-Shot Forecasting of In-Situ Soil Moisture and Temperature with a Time-Series Foundation Model

**arXiv ID:** 2610.30971 | [PDF](https://arxiv.org/pdf/2610.30971v1)

**作者:** Mohammad Talaei `[一作]` (Monash University), Iadine Chadès `[通讯]` (Monash University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在澳洲Nardoo Hills恢复实验区，对27个15分钟间隔的土壤湿度（5 cm、30 cm）和包覆温度进行零样本预测，使用预训练的500M参数TimesFM模型；

**💡 创新点**

首次验证时序基础模型在缺失数据、短期部署的现场环境中的零样本预测可行性，并揭示短期（1–8 h）和长期（>16 h）预测性能的分界机制；

**🔧 技术方法**

采用TimesFM（Transformer基础模型）与三种统计规则基线（持久、季节性、7日气候）以及CatBoost梯度提升树进行对比；

**📊 数据集**

使用澳洲Nardoo Hills Reserve 27个传感器（15 Treatment、9 Control、3 Scald）在2024年10月至2025年4月共六个月的15 分钟遥测数据；

**📈 对比分析**

通过gap-aware rolling-origin评估，TimesFM在1–8 h时平均比最佳基线低21–76% MAE，超过16 h则失效；CatBoost始终未能击败基线；TimesFM推理耗时≈6.4 s、峰值内存+2.7 GB，而CatBoost仅0.003 s、<5 MB；

**⚠️ 局限性**

局限于单一地点、6个月数据，Scald样本不足；温度仅为封装内温度；推理仅在CPU上；未验证不同季节、气候和更大传感网络的泛化性

---

## 352. IQuest-Q1: Advancing Foundation Capabilities for Agentic CLI Systems

**arXiv ID:** 2610.30949 | [PDF](https://arxiv.org/pdf/2610.30949v1)

**作者:** A. Jian Yang `[一作]` (IQuest Research), Bryan Dai `[通讯]` (IQuest Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

发布了一个开源的、支持命令行接口交互的代理基础模型，并通过多阶段训练提升编码、推理和CLI交互能力。

**💡 创新点**

设计了CLI中心化的多阶段训练管线、Sparse MoE+Hybrid注意力、Multi-Teacher On-Policy Distillation、递归多标记预测以及人机循环的递归自我改进框架。

**🔧 技术方法**

采用了Sparse Mixture-of-Experts、Hybrid Grouped-Query+Sliding-Window Attention、Stable SplitNorm、Muon+μP优化、MTP、RL+MOPD、模型合并、以及自监督与强化学习结合的微调技术。

**📊 数据集**

使用了多语言文本、GitHub代码、书籍、学术论文、新闻、任务轨迹、API环境以及合成任务生成的数据，覆盖编码、通用推理和CLI任务。

**📈 对比分析**

与Claude Opus、GPT-5.6 Sol、GLM-5.3-Flash等模型对比，在DeepSWE、SWE-Bench、Terminal-Bench、CyberGym、IQuest-CLIBench等多项基准上取得领先或接近最佳表现，展示出优异的编码与CLI交互能力。

**⚠️ 局限性**

仅支持文本输入、输出可靠性有限、工具调用格式依赖特定模板、在真实CLI任务中仍需人工监督、模型仍有可靠性与完整性问题。

---

## 353. Tracing Hallucinations in LLM Reasoning for UML Class Diagram Generation: An Empirical Taxonomy and Mitigation Study

**arXiv ID:** 2610.30946 | [PDF](https://arxiv.org/pdf/2610.30946v1)

**作者:** Mojdeh Rahmanian `[一作]` (Edinburgh Napier University), Yanchao Yu `[通讯]` (Edinburgh Napier University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对LLM生成UML类图过程中的幻觉进行系统研究，构建三维幻觉分类法，并将其用于生成提示和后置校验，提升图模型质量。

**💡 创新点**

首次提出基于推理链的Prompt Faithfulness、Code Faithfulness与Reasoning Quality三维幻觉分类，及其可操作的生成约束和校验规则。

**🔧 技术方法**

使用DeepSeek系列和Qwen3系列大型语言模型的链式推理，双LLM注释管线、prompt engineering与post‑hoc 代码一致性验证技术。

**📊 数据集**

使用Dataset 1（20教学案例）、Dataset 2（120多领域描述）进行分类验证，Dataset 3 Golden UML ModelSet（13案例）用于干预实验。

**📈 对比分析**

通过F1、完整性/正确性指标评估，实验表明嵌入分类约束的生成提示平均提升4个百分点，后置代码一致性校验未产生显著提升。

**⚠️ 局限性**

研究样本规模有限，跨模型推广性未充分验证；Reasoning Quality维度标注一致性低，实验统计力受限。

---

## 354. DConstrain: Compiling Context-Parameterized Transitions for Diffusion Language Model Constrained Decoding

**arXiv ID:** 2610.30920 | [PDF](https://arxiv.org/pdf/2610.30920v1)

**作者:** Shidong Cao `[一作]` (Hong Kong Baptist University), Da Zheng `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DConstrain，一种针对扩散语言模型（DLM）实现结构化输出的约束解码器；

**💡 创新点**

创新点包括：①将语法编译成上下文参数化的转换程序，避免枚举堆栈和字段历史导致的指数级状态爆炸；②引入冷热两阶段编译和 token‑effect grouping 以减少在线查询和图构造成本；③证明分组保持 Viterbi 最优与完整路径分布；

**🔧 技术方法**

核心技术有：上下文参数化的 RTN（递归转移网络）编译、符号执行与残差程序、堆栈与环境的运行时绑定、token‑effect 分组、基于图的动态规划（Viterbi/树形采样），以及与 GPU 前向推理的异步重叠；

**📊 数据集**

评估数据集包括 BFCL（工具调用）、JSONSchemaBench（语法多样性）和 ComplexFuncBench（复杂函数），并在 LLaDA2.2 与 DiffusionGemma 两大 DLM 上进行实验；

**📈 对比分析**

与现有方法（DINGO、NFA、LAVE、EPIC 等）相比，DConstrain 在任意属性顺序下编译率可达 100%，相对 DINGO‑Opt 的编译速度提升 3.5–10×；在 BFCL 上实现 100% 语法有效率、约 81% 语义准确率，且端到端延迟比 DINGO‑Opt 低 6.6–46×，推理吞吐量提升 1.5–17×；

**⚠️ 局限性**

局限性在于：①仍需要对复杂语法做裁剪（如交叉、排他等未覆盖特性）；②对极深嵌套（>8 层）或极大字段集合的场景可能仍出现内存/时间瓶颈；③当前实现针对 JSONSchema，扩展到其它结构化语言仍需进一步验证；

---

## 355. Near-Quadratic Time Multivariate Polynomial Interpolation

**arXiv ID:** 2610.30907 | [PDF](https://arxiv.org/pdf/2610.30907v1)

**作者:** Chandra Kanta Mohapatra `[一作]` `[通讯]` (Aalto University), Chandra Kanta Mohapatra (Aalto University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

论文提出了一种在任意有限域上，对多变量多项式插值问题（给定 N=d^m 个评估点与对应值）实现近二次时间（N²+o(1)(m,d,log||)）的算法。

**💡 创新点**

创新点在于：① 将 Wiedemann 的稀疏线性方程求解技术与最近的多变量多点评估算法相结合，得到全新确定性求解方法；② 对于秩缺陷的 Vandermonde 矩阵，提出了既可确定性又可随机化的处理策略；③ 在没有黑盒评估、没有点集结构的通用设置下实现了速度提升。

**🔧 技术方法**

主要技术包括：
- 维德曼（Wiedemann）方法的确定性变体，用 Berlekamp–Massey 对每个坐标序列求最小多项式；
- Bhargava‑et‑al. 的多变量多点评估（以及其转置）实现快速矩阵‑向量乘法；
- Toeplitz 预条件与随机化技巧（Kaltofen–Saunders）用于求解秩缺陷系统；
- 通过矩阵‑向量乘法的可逆性判断系统可解性。

**📊 数据集**

论文没有使用传统意义上的数据集，全部是在符号计算与理论分析层面进行，输入为 N 个点与对应的值。

**📈 对比分析**

实验或实测结果未给出，性能评估以理论时间复杂度呈现：
- 对于满秩或常数共秩的 Vandermonde 矩阵，算法时间为 O(N²+o(1)(m,d,log||))；
- 对于一般情况，使用随机化预处理后同样可达到上述时间；
- 与最坏情况 O(N^ω) 的稠密线性方程求解相比，显著降低了计算量。

**⚠️ 局限性**

限制包括：
- 随机化版本需要有限域大小足够大（或在域扩展后使用），否则失败概率不低；
- 确定性版本仅适用于满秩或共秩常数的情况；
- 对参数 d 的增长有要求，需 d = ω(1)；
- 对于高共秩情况，仍需额外随机化步骤，可能导致常数因子增大。

---

## 356. Same molecule, Different Knowledge: Cross-Identifier Binding Failures in LLMs For chemistry

**arXiv ID:** 2610.30903 | [PDF](https://arxiv.org/pdf/2610.30903v1)

**作者:** Khouloud Saadi `[一作]` (King Abdullah University of Science and Technology), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究大型语言模型（LLM）在化学领域中对同一分子不同表示（如普通名称、IUPAC名称、SMILES）下知识检索的差异性；

**💡 创新点**

提出“交叉标识符绑定失败（cross‑identifier binding failure）”这一现象，并设计化学标识符基准 ChemIDBench 进行系统评估；进一步开发“绑定自蒸馏（Binding Self‑Distillation, BSD）”方法，将模型在名称路径下可访问的知识迁移至结构路径；

**🔧 技术方法**

主要技术包括：基准构建、识别/检索实验、冲突标识符实验、表示对齐与激活补丁（activation patching）因果分析、知识编辑（ROME、Fine‑Tuning）实验、以及自蒸馏训练；

**📊 数据集**

使用 ChemIDBench 数据集，涵盖 896 只已批准药物的三种标识符（普通名称、IUPAC 名称、SMILES），并在此基准上评估七款 7B–72B 参数规模的 LLM；

**📈 对比分析**

实验对比显示：同一分子使用普通名称与 SMILES 进行事实检索时，准确率差异可达 30–40% 以上；知识编辑方法效果不稳；而 BSD 在所有模型中将 SMILES 条件下的事实检索准确率从 1–5% 提升至 30–50%，显著优于传统编辑；

**⚠️ 局限性**

局限性包括：BSD 仅针对已知实体单独训练，尚未实现对未见实体的泛化；对模型参数规模的依赖仍未完全消除；以及在实际化学工作流中仍需解决标识符冲突检测和一致性校验等问题。

---

## 357. ReDence: A Hybrid Agentic and Decision-Model Framework for Rapid, Evidence-Grounded Detection of Modern Social Engineering Scams

**arXiv ID:** 2610.30963 | [PDF](https://arxiv.org/pdf/2610.30963v1)

**作者:** Jaishnoor Kaur `[一作]` (Louisiana State University), Sayak Saha Roy `[通讯]` (Louisiana State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ReDence 混合框架，结合 Qwen3.5‑4B Investigate 与 Tev1‑4B Decider，实现在有限算力下快速、证据驱动的社会工程诈骗网站检测与可追溯报告生成。

**💡 创新点**

创新点在于将复杂调查拆解为可由小模型高效完成的决策任务，使用专门的 Decider 缩短推理时间；并设计结构化证据账本与解释日志，使分析师能够独立审计决策与证据链。

**🔧 技术方法**

技术包括 Qwen3.5‑4B（多模态代理）+ Tev1‑4B（二进制/多选决策模型），配合 Fetch Page、Interact Page、Web Search 工具实现网页抓取、交互和外部验证；利用 QLoRA 对两模型进行微调。

**📊 数据集**

使用基于 BBB Scam Tracker 的 1,551 欺诈网站与 1,500 合规网站构建的真实网页快照数据集；训练集 728 欺诈 + 1,200 合规（共 1,928 条），测试集 1,551 欺诈 + 1,500 合规（共 3,051 条），涵盖 12 类社会工程诈骗及 5 类保留类别。

**📈 对比分析**

通过与 Mentor（27B）、URLNet、BeyondPhish、ScamFerret（GPT‑4.1 与 Llama‑3.3‑70B）等对比，ReDence 在 12 类诈骗上达 96% F1，召回率 0.98，精度 0.97；平均检测时间 43.8 s，GPU 3.4 GB，显著优于 Mentor（194 s/33.8 GB）与现有检测器。

**⚠️ 局限性**

局限性包括对未见类别的泛化仍受限，依赖本地网页快照可能导致对实时站点的检测延迟；决策模型在极端复杂情境下仍需人工验证；模型在某些新型诈骗手法上可能误判或漏检。

---

## 358. EEG-Captioner: Grounding EEG-to-Language Generation in Fundamental Signal Properties

**arXiv ID:** 2610.30995 | [PDF](https://arxiv.org/pdf/2610.30995v1)

**作者:** Leyuan Jin `[一作]`, Yue Ding `[通讯]` (X-LANCE Lab Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

构建了包含160,781段EEG信号的EEG‑Language数据集，并在每段信号上生成了量化属性的问答对（包括频谱、空间、时间和侧向特性）和基于特征的自由文本描述；提出了Dual‑Domain EEG Encoder及联合训练的EEG‑Captioner框架；

**💡 创新点**

创新点在于①首次提供了覆盖时间、频谱、空间和侧向四个维度的量化问答对，用于对EEG信号属性进行客观评估；②提出Dual‑Domain编码器同时建模原始时域波形和谱空间结构，显著提升了对这些属性的捕捉；③通过联合信号问答和文本描述的双任务训练，实现了对EEG属性的更好刻画与语言生成的更高可信度；

**🔧 技术方法**

采用了双域EEG Encoder（时域patch + 谱空间自注意）与Transformer融合，使用Qwen2.5-Omni-7B LLM进行文本生成，利用LoRA和投影层实现多模态融合，并结合梯度优化器AdamW进行训练；

**📊 数据集**

使用了Temple University EEG Corpus (TUEG) v2.0.2 作为原始EEG来源，并通过标准化处理得到高质量分段；

**📈 对比分析**

在五类属性的相对问答任务中，Dual‑Domain Encoder 在分类阶段 Macro‑BA 达到 93.76%，在 Qwen 迁移阶段保持 86.06%，显著优于 LaBraM (76.06%) 和 CBraMod (50.45%)；联合 QA‑Caption 训练后，Macro‑BA 提升至 88.81%，而生成文本的事实一致性从 82.06% 提升至 83.46%，ROUGE‑L 与 BLEU‑4 亦有提升；

**⚠️ 局限性**

局限性包括：①数据集主要基于单一公开EEG语料，可能缺乏跨中心、多设备的多样性；②问答对采用相对比较而非绝对量化，可能忽略微小差异；③生成文本的事实一致性提升有限，仍受模型推理偏差和数据噪声影响。

---

## 359. Interpretive Variation and Discourse Structure in Detecting Enthymemes in Controversial Political Discourse

**arXiv ID:** 2610.30987 | [PDF](https://arxiv.org/pdf/2610.30987v1)

**作者:** Martial Pastor `[一作]` (Radboud University), Nelleke Oostdijk `[通讯]` (Radboud University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在社交媒体推文中检测隐含前提或结论（enthymeme），并且把标注者之间的分歧视为任务的一部分，提出基于投票分布的评估指标和能直接预测投票分布的 Betabin 模型，同时分析话语结构（RST）对检测的影响。

**💡 创新点**

创新点：① 将标注者争议视为任务本身而非噪声，设计多维度投票分布评估指标；② 引入 Beta 分布投票模型直接预测投票分布，实现对争议的更细粒度捕捉；③ 通过特征注入与探针实验证明话语结构能解释模型成功与失败的模式；④ 评估不同标注方案（硬标签、软标签）和特征注入对模型的影响。

**🔧 技术方法**

技术：Fine‑tuned DeBERTa‑v3‑base、经典 Logistic Regression 与 SVM、TF‑IDF 特征、RST 结构特征（24 维）和 trope 标注（9 维）、软标签训练（soft focal CE）、Beta 分布投票建模（Betabin）、多种评估指标（ambiguity Spearman、Manhattan distance、soft‑ECE、consistency per‑vote、per‑vote Spearman 等）、探针分析（probe CLS 句子向量）。

**📊 数据集**

数据集：1,482 条英文推文（疫苗 1,169 条、移民 313 条），每条由 5 名标注者打标，包含 3 种标签（无、隐含前提/结论）。训练/验证集 1,334 条，测试集 148 条。数据附带自动 RST 解析树和黄金 trope 标签。

**📈 对比分析**

比较方法与性能：传统 LR/SVM 约 0.62–0.63 macro‑F1；Transformer（soft 0.67–0.69）显著优于传统模型；硬标签与软标签在 F1 上差异不大；Betabin 与 soft baseline 在 macro‑F1 上相当，但在投票分布评估指标（ambiguity、distance、consistency）上均优于其他模型；学习曲线表明数据量增大对 F1 的提升有限，约 0.67 的上限；RST 特征在 Transformer 中冗余但在分析中具有解释力。

**⚠️ 局限性**

局限性：① F1 对争议投票难以提升，表明单一标签无法捕捉语义解释多样性；② 数据规模与主题（疫苗、移民）有限，无法检验跨主题或跨语言的泛化；③ RST 解析错误与未标记的隐含关系导致模型误判；④ 只评估 5 位标注者的投票，无法覆盖更广泛的观点多样性；⑤ 现有模型仍难以捕捉未显式标注的隐含前提，导致多数 enthymeme 在模型上保持高不确定性。

---

## 360. A Reality Check of Inference-Time Context Augmentation Methods

**arXiv ID:** 2610.30983 | [PDF](https://arxiv.org/pdf/2610.30983v1)

**作者:** Haeun Yu `[一作]` (University of Copenhagen), Isabelle Augenstein `[通讯]` (University of Copenhagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个句子级别的上下文增强框架，比较了五种不同信息量的上下文插入方法在七种LLM上的效果。

**💡 创新点**

首次系统地从决策边界、位置多样性以及方向性检验角度对上下文增强方法进行统一评估，提出了 sign‑randomisation 测试以区分真正改进与无方向扰动。

**🔧 技术方法**

采用了 Blank、Repeat、Paraphrase、Self‑explanation 和 Teacher‑explanation 等插入策略，并使用 margin 变化、正负改正比例分析和随机符号检验等技术。

**📊 数据集**

在 MedQA、MMLU（限定多句段落）和 DRUID 这三个多选/事实核查数据集上进行实验。

**📈 对比分析**

将每种方法的句子平均准确率与原始模型对比，并通过 sign‑randomisation 检验判定是否有向改进；结果显示仅少数信息量较大的方法在部分模型/数据集上能显著提升准确率，且大多数方法的提升往往来自无方向扰动。

**⚠️ 局限性**

研究仅针对多选/二分类任务，未覆盖开放式生成；仅一次单句插入，未探究多插入组合；且未涉及推理链模型，结果可能不适用于所有LLM。

---

## 361. Diagnostic Loss: Asymmetric Supervision for Radiology Report Generation

**arXiv ID:** 2610.30980 | [PDF](https://arxiv.org/pdf/2610.30980v1)

**作者:** Ignacy Kolton `[一作]` (Jagiellonian University), Szymon Płotka `[通讯]` (Jagiellonian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了诊断损失（DLoss），一种用于3D医学视听语言模型的无监督诊断监督目标，直接作用于解码器概率分布，增强放射报告的诊断可靠性。

**💡 创新点**

创新点在于使用不对称池化（支持诊断采用噪声-OR，非支持诊断采用温度调控的log-sum-exp）对诊断词汇概率进行序列级监督，兼顾支持与否的诊断需求；同时保持架构无关、训练时无额外推理成本。

**🔧 技术方法**

技术包括：基于解码器logit的概率聚合、支持/非支持诊断词汇集合的离线构建、可加至交叉熵的诊断损失、以及多模型多维度评估。

**📊 数据集**

使用了两大腹部CT公开数据集CancerVerse（≈24k扫描）与Merlin（≈20k扫描）进行训练，评估在Merlin测试集（≈5k扫描）和私有外部测试集（≈2.7k扫描）。

**📈 对比分析**

与六种主流3D MedVLM（Merlin、SPECTRE、RadFM、fVLM、BTB3D、HLIP）进行对比，DLoss在GREEN、RadGraph-F1等诊断准确性指标平均提升约4点，且对BLEU、ROUGE-L、METEOR、CIDEr等语言指标也保持或略优，且推理时无额外计算。

**⚠️ 局限性**

局限性包括诊断词汇集合仅来自训练报告，可能缺失未出现的概念；实验仅覆盖腹部CT，未验证在2D X光或其他部位的泛化；评估仍依赖自动指标，缺乏放射科医生人工验证。

---

## 362. Energy and Latency First: How Deployment Constraints Reverse Time-Series Model Selection

**arXiv ID:** 2610.30965 | [PDF](https://arxiv.org/pdf/2610.30965v1)

**作者:** Shruti Bothe `[一作]` (Ericsson Research), Ayush K. Varshney `[通讯]` (Ericsson Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在蜂窝网络的切换预测任务中，对八种时间序列模型在满足 40 ms 预测周期的尾部可行性约束和累计能耗预算的条件下进行系统评估。

**💡 创新点**

提出一种基于尾部可行性筛选与完整流水线能耗测量的评估协议，发现梯度提升树在手工特征上同时兼具最高准确率、最低延迟和最低能耗，并揭示仅靠准确度评估会误选不可部署模型；同时说明测量模式（忙循环 vs 断续）会显著改变模型选择。

**🔧 技术方法**

使用 LightGBM、CNN‑1D、Efficient‑HO、MiniRocket、Transformer、LSTM、catch22+ridge 等模型；特征工程（手工统计、时间趋势、最小边缘等）；能耗测量（忙循环、断续循环）以及容量扫描和蒸馏实验。

**📊 数据集**

构建基于 Sionna 与 SUMO 的雷射追踪 RSRP 仿真数据集：21 个基站、100 个 UE、40 ms 采样、100 s 总时长，产生约 4.7 M 采样点和 596 次切换事件，用于训练/测试集。

**📈 对比分析**

对所有模型进行完整流水线延迟与能耗测量；先按 99% 延迟不超过 40 ms 的尾部可行性筛选，随后按每次推理能耗排序。结果显示 LightGBM 达到 F1=0.974、99% 延迟 0.38 ms、断续能耗 4.9 μJ；CNN‑1D 虽准确率最高（F1=0.914）但能耗高 3+ 倍；所有神经网络模型均被 LightGBM 超越。

**⚠️ 局限性**

能耗估计基于 CPU 利用率，未在真实手机芯片上验证；数据集仅为单一城市的仿真，缺乏对其他环境和更大序列长度的泛化验证；使用的线性 Ridge 头可能限制了特征模型性能；实验仅聚焦已知决策边界的短序列场景。

---

## 363. Doubly Robust Distributionally Robust Offline Contextual Pricing

**arXiv ID:** 2610.30957 | [PDF](https://arxiv.org/pdf/2610.30957v1)

**作者:** Min Xu `[一作]` (Nanjing University), Caihua Chen `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一套双稳健（Doubly Robust）框架，用于在连续定价（continuous pricing）场景下进行分布鲁棒（Distributionally Robust）离线因果估计与策略学习。

**💡 创新点**

创新点包括：① 用局部化技术（localized DR）在评估阶段只需拟合三组回归模型即可实现半参数效率；② 在学习阶段通过数据驱动的权重（如通用随机森林）实现连续回归，并给出理论上更优的有限样本 regret 率 𝑂̃(𝑇^{‑s/(2s+1)}，s=1,2)，优于以往的 𝑂(𝑇^{‑1/4})。

**🔧 技术方法**

技术手段主要有：核平滑（kernel smoothing）、双稳健估计、交叉拟合（cross‑fitting）、局部化估计、通用随机森林（GRF）产生权重、牛顿-拉弗森求解、交替优化、半参数效率理论、KL 散度不确定性集合。

**📊 数据集**

使用的数据集包括：① 结构化仿真数据（X∈ℝ³，价格服从正态分布，需求噪声为 Logistic）；② Expedia Personalized Sort Competition 实际酒店搜索数据（45 维特征，目标是预测预订率）。

**📈 对比分析**

与现有的 DRO‑IPW 基线以及非鲁棒 DR 基线进行比较；在评估任务中 LDR²O²PE‑CP 的 MSE 下降速度快于 DRO‑IPW，且在大样本下更稳健；在学习任务中 CDR²O²PL‑CP 在稳健政策价值、置信区间宽度和实验稳健性方面均优于非鲁棒 DR 与 DRO‑IPW；在 Expedia 实验中，C‑DR 方案在 KL‑对抗压力测试下得到最高平均回报。

**⚠️ 局限性**

局限性：① 仅使用 KL 散度作为不确定性度量，未探索 Wasserstein 等更具几何直观性的度量；② 对高维状态空间的可扩展性尚待验证；③ 在样本量较小（T<1000）时，局部化 DR 在评估阶段的稳定性略低；④ 需要对日志策略进行估计或已知，估计误差会影响性能。

---

## 364. In-Context Learning World Models for Robot Dual-Arm Manipulation of Soft Garment with Partial Observations

**arXiv ID:** 2610.30988 | [PDF](https://arxiv.org/pdf/2610.30988v1)

**作者:** Hengyi Sim `[一作]` (Chinese University of Hong Kong), Fei Chen `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于Transformer的软体物体世界模型，利用在时上下文学习对部分点云进行全状态重建、动作条件动力学预测，并将其与目标条件动作生成器结合，实现双臂服装折叠任务。

**💡 创新点**

创新点在于将流匹配生成式状态估计与在时上下文学习相结合，实现无需参数微调即可从少量示例推断材质特性；同时设计目标条件流匹配策略生成可行动作方案，并通过世界模型评估选择最佳动作。

**🔧 技术方法**

采用的技术包括Vision Transformer骨干、流匹配与条件流匹配、Voronoi采样+PointNet分割、适应性层归一化、交叉注意力、动作相对6-DoF表示，以及生成式动作策略与模型预测控制。

**📊 数据集**

使用的数据集为基于Isaac Lab的数十万条软体物体交互仿真样本，包含随机材质、抓取配置；此外在真实世界中收集了X‑trainer和Tiangong人形机体两套机器人上的双臂折叠实验数据。

**📈 对比分析**

与PCN、UniClothDiff、流匹配、Adaptigraph、GNN等基线相比，状态重建方面FM模型在MSE上显著优于其他方法，动态预测中监督Transformer在MSE上领先；真实世界折叠成功率在不同布料与机器人上均超过80%，CD/EMD误差亦保持在较低水平。

**⚠️ 局限性**

局限性在于目前仅针对准静态折叠任务，动态拉伸、抖动等高速运动尚未验证；长期折叠会出现误差累积，且确定性预测模型在未来状态不确定性增大时表现可能下降，需要闭环重规划支持。

---

## 365. LAST-V2X: Language-Driven Adaptive Scenario Synthesis and Testing for Communication-Induced Failures in V2X Safety Applications

**arXiv ID:** 2610.30979 | [PDF](https://arxiv.org/pdf/2610.30979v1)

**作者:** Keshu Wu `[一作]` (Texas A&M University), Yang Zhou `[通讯]` (Texas A&M University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 LAST-V2X，一个基于大型语言模型的自动化 V2X 场景合成与安全测试框架。

**💡 创新点**

将自然语言描述映射为可执行的物理‑信息耦合场景，并通过配对执行实现因果验证，结合执行导向的参数搜索和 LLM 自我反思实现自适应精炼。

**🔧 技术方法**

利用 GPT‑4o 进行语义解析、检索增强与自我反思，CARLA 进行同步物理与通信仿真，构建空间 V2X 通信模型和七类信息失真操作。

**📊 数据集**

在八个 CARLA 城镇环境中随机生成 100‑场景的基准测试，并扩展至 50‑场景的故障套件，涵盖六类 V2X 安全应用。

**📈 对比分析**

与传统单独物理场景或通信失真方法对比，实验显示在健康通信下零误报，在失真测试中成功区分因果失败，且自适应精炼将有效失败率从 14% 提升至 64%，每个成功案例可降至单一故障最小 witness。

**⚠️ 局限性**

受限于简化的通信参数、简化的决策逻辑、在完整 CARLA 集成下的搜索规模，以及仅适用于固定时间信号的场景。

---

## 366. Joint or Per-Option? Compact Encoders for Closed-Set Decisions from 17 Million to 1 Billion Parameters

**arXiv ID:** 2610.30970 | [PDF](https://arxiv.org/pdf/2610.30970v1)

**作者:** Tardelli Ronan Coelho Stekel `[一作]` `[通讯]`, Tardelli Ronan Coelho Stekel

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文训练并发布了一系列从 17 M 到 1 B 参数的紧凑交叉编码器，用于在闭集决策任务（多选、零样本分类、工具调用等）中生成概率分布，并系统比较联合读取（一次编码所有选项）与逐选项读取（对每个选项单独编码）的效果。

**💡 创新点**

创新点在于：① 对联合 vs 逐选项读取做了受控实验，发现仅在使用重排器（reranker）初始化时逐选项显著更优；② 展示 17 M 模型可在 CPU 上每秒数百请求，满足低延迟场景；③ 引入无学习规则引擎 Prumo 作为基准，揭示评测板可能存在的格式特定短路；④ 将所有代码、模型权重、实验脚本与评测数据公开，提升可复现性。

**🔧 技术方法**

使用的技术包括 ModernBERT/Ettin 作为基础编码器，交叉编码器（cross‑encoder）架构，BERT 多选头/UniMC‑style 读取器，Brier‑损失 + CE 作为训练目标，知识蒸馏，NLI 零样本分类（DeBERTa‑v3、BART‑MNLI），LLM 字母读取方案，option‑order 随机化与逆序测试，ECE 与 Brier 校准评估，以及基于 GPU/CPU 的 latency 测量。

**📊 数据集**

训练数据集为 638k 条决策问题，涵盖 Bekko System One、Tasksource、程序生成的决策族等多种格式，并对训练集进行严格污染审核。评估数据集包括：Decision Index（37 benchmark 公测套件）、S1‑Arena（209 任务，分为 seen/unseen）、BTZSC（22 文本分类数据集，其中 4 个在训练中未出现）。

**📈 对比分析**

比较方法：在相同训练步骤、相同计算量和相同数据集的多组实验（如 32 M、三种种子、joint、per‑option、无 Brier、无蒸馏、raw encoder 等），并通过公测套件 skill、S1‑Arena task‑mean accuracy、BTZSC macro‑F1 等指标进行对比。结果显示：① 逐选项在 32 M 与相同计算下提高约 5.8 skill 点；② 68–150 M 模型在 S1‑Arena 上超过 435 M DeBERTa‑v3 的准确率；③ 17 M 模型 CPU 中位延迟约 14 ms；④ 在 option‑order robustness、校准（ECE/Brier）和 latency 方面也给出了量化评估。

**⚠️ 局限性**

局限性包括：① 逐选项读取虽然更准确但需要 5–13 倍的编码通道，导致尾部延迟显著；② 读取方式对是否使用 reranker 预训练敏感，未见 raw encoder 下的优势；③ 外部评测受污染审核限制，仅有少数未见任务；④ 私有测试仅运行一次，缺乏可复现性；⑤ 仅在英语数据上训练和评估，无法保证多语言适用；⑥ 对新任务的过度自信需要额外阈值校准；⑦ 读者研究仅在 32 M 规模和子集上完成，缺乏更大规模对比。

---

## 367. Once a Tree, Always a Tree? Cross-lingual Transfer of Hierarchical Generalization in Language Models

**arXiv ID:** 2610.30960 | [PDF](https://arxiv.org/pdf/2610.30960v1)

**作者:** Shin Yamaguchi `[一作]` (NAIST), Yoshinobu Kano `[通讯]` (Shizuoka University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过设计跨语言贫刺激（CPOS）实验，探究语言模型在L1中学习层次结构后，是否能在无正向证据的L2中迁移该层次化偏差。

**💡 创新点**

创新点在于提出CPOS框架并系统量化词汇重叠对层次化迁移的影响，证明层次化迁移受词汇重叠支配。

**🔧 技术方法**

采用GPT‑2‑small Transformer自回归语言模型，冻结层实现L2训练，以控制跨语言迁移。

**📊 数据集**

使用四种L1（英语、德语、俄语、日语）与英语L2的Wikipedia语料，构成POS/Disambiguating数据集，并在CLAMS与MultiBLiMP最小对评估。

**📈 对比分析**

通过比较两种训练条件的准确率差值，发现L1正向证据可提升L2层次化准确率（CLAMS最高+5.45%），但词汇重叠越大效果越明显；词汇重叠完全消失时迁移几乎消失。

**⚠️ 局限性**

局限性包括模型规模仅限GPT‑2‑small、仅四种L1、冻结Transformer、以及合成与自然数据差异，结果可能不适用于更大模型或更灵活的多语种预训练。

---

## 368. Measuring the Reasoning-Action Coupling in Driving Vision-Language-Action Models through Counterfactuals

**arXiv ID:** 2610.30953 | [PDF](https://arxiv.org/pdf/2610.30953v1)

**作者:** Gabriel Manalu `[一作]` (KAIST), Daeyoung Kim `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过对驾驶视觉-语言-动作（VLA）模型进行因果干预，评估生成的推理文本与车辆轨迹之间的耦合程度，具体包括在保持视觉场景不变的情况下编辑模型生成的推理文本并观察轨迹变化，以及通过光学上移除推理中提及的关键对象并重新生成推理与轨迹，从而检验推理内容对动作的实际影响与可视化证据的可追踪性。

**💡 创新点**

首次在驾驶VLA中引入基于因果干预的可解释性评估框架：①在推理侧进行语义层面的编辑并与语义控制的无改动版对比，②在图像侧构造光学上可实现的关键对象移除对比，③将两种干预结合以同时考察推理更新与轨迹更新的对应关系，④通过方向一致性检验（stated vs decoded lateral direction）进一步验证推理与动作的一致性。

**🔧 技术方法**

使用大型语言模型（Llama‑3.1‑8B‑Instruct 和 Qwen3.5‑9B‑Instruct）进行推理文本的语义重写与决策分类；利用三种对象移除方法（DiffuEraser、RORem、ObjectClear）构造光学可编辑的对照图像；采用平均位移误差（ADE）与最终位移误差（FDE）衡量轨迹变化；在每个场景和种子下进行配对评估，并与摄像头置空（camera blanking）基准对比；使用 DriveLM 与 AutoVLA 等模型的推理-动作接口进行实验。

**📊 数据集**

在 DriveLM 验证子集（基于 nuScenes 的 799 场景）和 PhysicalAI‑AV 验证子集（349 场景）上进行评估，分别对应 DriveLM/AutoVLA/ORION 等模型的输入格式；这些数据集提供了结构化语言注释与真实摄像头观测，支持推理文本与轨迹的对照分析。

**📈 对比分析**

评估方法：在推理侧使用语义编辑与控制词法对比，在图像侧使用对象移除后重新生成推理和轨迹；通过 ADE/FDE 计算干预导致的轨迹位移，并将结果归一化为相对于摄像头置空的比例。实验发现：某些模型（如 ORION、AutoVLA）对推理文本的存在/表面形式更敏感，内容编辑对轨迹影响有限；Alpamayo 系列模型对内容编辑有一定响应但方向一致性低；对象移除能更新推理但轨迹不一定跟随更新；跨数据集实验表明这些模式具有鲁棒性。总体来看，模型在推理内容对动作的因果影响上表现不佳。

**⚠️ 局限性**

局限性包括：①评估仅覆盖四种 VLA 模型，未涵盖所有现有方法；②对象移除对齐与时间一致性仍受生成质量影响，可能产生伪影；③只检验了轨迹和推理的一致性，未涉及更细粒度的动作决策或安全评估；④实验基于有限的验证子集，未覆盖所有可能的驾驶场景与突发事件；⑤因果干预假设模型解码器对文本或图像的响应为线性或可逆，实际可能更复杂。

---

## 369. Attention-guided classification and tumor localization of combined hepatocellular-cholangiocarcinoma

**arXiv ID:** 2610.30944 | [PDF](https://arxiv.org/pdf/2610.30944v1)

**作者:** Xiaowen Liang `[一作]` (Université de Strasbourg), Cédric Wemmert `[通讯]` (Université de Strasbourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一种基于注意力监督的多实例学习（MIL）模型，能够对HCC、CCA和cHCC‑CCA进行三分类，同时通过联合训练的双分支模型实现对cHCC‑CCA肿瘤中HCC‑样与CCA‑样成分的定位。

**💡 创新点**

创新点包括：①利用二分类（HCC‑CCA）模型生成的注意力热图作为参考，监督三分类模型的注意力分布以提升热图的病理合理性；②设计双分支MIL结构，分别对HCC‑样和CCA‑样区域进行互补注意力学习，并通过联合损失（包含联合对齐、背景抑制、互补性约束）实现成分分离。

**🔧 技术方法**

主要技术手段为：Transformer‑based MIL（TransMIL）结合多头注意力；注意力监督（KL、L1、余弦、rank损失）；双分支二分类MIL；联合损失框架（交叉熵+注意力监督+联合注意力损失）；以及使用UNET基础模型提取1536维patch特征。

**📊 数据集**

数据集：HES染色的WSI，HCC 338张，CCA 353张，cHCC‑CCA 222张；内部10折交叉验证；外部独立评估集183张（包括TCGA 36 HCC、30 CCA）。

**📈 对比分析**

方法比较：无注意力监督的三分类模型宏F1为0.880；加上注意力监督后宏F1略降至0.872，但热图的病理合理性（TAC）显著提升；联合训练后宏F1恢复至0.876；双分支模型宏F1为0.834，但在成分定位方面表现优异，CCA‑样分支的CCA/HCC注意力比为2.483，HCC‑样分支为0.48，表明两分支能分别关注对应成分。

**⚠️ 局限性**

局限性：cHCC‑CCA肿瘤形态多样，部分切片仅包含单一可辨成分或分化模糊；目前模型未能充分利用成分定位信息提升三分类准确率；未来工作需进一步处理形态多样性并探索成分特定证据对诊断的提升。

---

## 370. Verbalizing Activation Differences with Natural Language Autoencoders

**arXiv ID:** 2610.30934 | [PDF](https://arxiv.org/pdf/2610.30934v1)

**作者:** Andrea Blasi Núñez `[一作]` (University of Southern Denmark), Lukas Galke Poech `[通讯]` (University of Southern Denmark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了自然语言自编码器（NLA）在模型差异解释（model diffing）中的可迁移性与表达能力，验证其在词义类比、轻度微调迁移、隐含目标恢复以及通过强化学习进一步改进等方面的效果。

**💡 创新点**

创新点包括：①证明NLA在激活空间上具有组合性；②展示NLA可在不重新训练的情况下适用于轻量微调模型；③证明NLA能够通过激活差异恢复微调目标；④首次通过强化学习针对差异激活微调NLA并提升其差异解释性能。

**🔧 技术方法**

主要技术：自然语言自编码器（NLA）架构（激活转化器与重构器）、强化学习优化（GRPO）与奖励增益、激活差异推理。

**📊 数据集**

使用的数据集包括：Google 词义类比数据集、Taboo 目标词隐藏模型族（20 体/模型）、Emergent Misalignment（EM）三类微调模型、以及公开的Gemma、Qwen、Llama 三大模型的激活样本。

**📈 对比分析**

与现有方法对比：在词义类比任务中，NLA表现与基线相当；在模型差异恢复任务中，NLA在Taboo隐含词恢复上达 80–90% 的命中率，优于黑盒基线但略低于 ADL；在 EM 任务中，NLA 接近 Activation Oracle 的表现，并显著优于黑盒方法。

**⚠️ 局限性**

局限性：① 仅在轻量微调上验证，未覆盖大幅度微调；② 受限于训练数据，较大模型的 NLA 需要更高计算成本；③ 评价仅基于关键词匹配，无法捕捉同义表述；④ 对未知目标或更宽泛微调任务的迁移性尚未验证。

---

## 371. Harnessing Language Models as Optimization Modeling Agents

**arXiv ID:** 2610.30917 | [PDF](https://arxiv.org/pdf/2610.30917v1)

**作者:** Yunxiang Luo `[一作]` (Hong Kong University of Science and Technology), Can Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在冻结的大语言模型上通过可执行交互和可编辑公式状态来完成自然语言优化建模任务，提出了轻量级推理包 OptiCycle 与 OptiState，并在 9 个基准上进行评估。

**💡 创新点**

创新点在于将执行反馈驱动的代码生成与循环修正以及五元组公式状态作为统一的推理接口，证明即使不做额外训练，也能显著提升优化建模的准确率。

**🔧 技术方法**

主要技术包括：单一模型角色的推理框架、持续 Python 解释器环境、执行反馈循环、以及可编辑的五元组公式状态（sets, parameters, variables, objective, constraints）。

**📊 数据集**

使用了 9 个优化建模基准（共 2634 个问题），并评估了 11 种 Qwen 变体（0.6B–27B）以及其他模型如 GPT-OSS-20B、Magistral、Phi 等。

**📈 对比分析**

与六种改造 Agent 工作流和四个后训练优化模型在共享 backbone 下比较，OptiCycle/OptiState 在宏观准确率上提升至约 62–68%，比对手高出 4–8 个百分点，同时使用的 token 量更少。

**⚠️ 局限性**

局限性包括：对公式构建步骤有额外开销、对模型规模敏感、未做匹配预算的 ablation、未给出不确定性区间、仅评估目标值一致性，未覆盖解的可行性与完整性验证。

---

## 372. MIG-COW: Transferable Adversarial Attacks on Deepfake Detectors via Gradient Decomposition

**arXiv ID:** 2610.30914 | [PDF](https://arxiv.org/pdf/2610.30914v1)

**作者:** Wonjune Seo `[一作]` (Ulsan National Institute of Science and Technology), Saerom Park `[通讯]` (Ulsan National Institute of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种名为MIG-COW的对抗攻击框架，能够在CNN和ViT两类深度伪造检测模型上生成可转移且视觉上不可察觉的对抗样本。

**💡 创新点**

创新点在于将动量集成梯度与共识-正交加权相结合，通过梯度的共识与正交分解捕捉不同模型的共享与特定漏洞，显著提升对抗样本的跨架构可转移性。

**🔧 技术方法**

采用集成梯度（Integrated Gradients）、动量更新、梯度归一化、Gram矩阵特征分解、共识-正交权重、交叉熵基损失以及PGD/MI-FGSM等基准对抗方法。

**📊 数据集**

使用AADD-2025挑战提供的高质量与低质量伪造图像数据集（共1403张），并在该数据集上评估攻击效果。

**📈 对比分析**

与PGD、MI-FGSM、基础MIG等基线方法在白盒与黑盒检测模型上对比，MIG-COW在白盒ASR近乎100%、黑盒ASR最高、SSIM保持较高、整体得分最高。

**⚠️ 局限性**

在官方黑盒ViT-B/16上迁移效果仍有限，且引入低性能模型会削弱攻击效果，说明对抗攻击的可转移性受源模型的功能相似度影响大。

---

## 373. Transformers with CoT Discover Heavy-Ball Acceleration Through Training

**arXiv ID:** 2610.31089 | [PDF](https://arxiv.org/pdf/2610.31089v1)

**作者:** Yuxuan Cai `[一作]` (University of Hong Kong), Yuan Cao `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了使用链式思考（CoT）的Transformer在上下文中预测线性回归权重的能力，并通过两阶段训练让模型从梯度下降教师中学习到新的优化算法——Heavy‑Ball。

**💡 创新点**

证明Transformer在CoT推理下能够在训练中自行发现加速算法（Heavy‑Ball），而不需要任何Heavy‑Ball教师，展示了模型超越模仿而自主发现新算法的潜力。

**🔧 技术方法**

采用线性自注意力架构、两阶段梯度流训练（第一阶段梯度下降教师、第二阶段真权重监督），引入辅助参数和矩阵多项式分析，利用PL不等式完成收敛证明；实验中使用AdamW优化器。

**📊 数据集**

使用人工生成的线性回归数据集：随机高斯输入向量与权重，构造(n,d)=(16,32)和(12,24)两组实验设置；并未使用公开数据集。

**📈 对比分析**

与梯度下降教师、时变Heavy‑Ball基线以及不同Transformer变体（仅第一阶段、仅第二阶段、无位置编码）进行对比。两阶段训练的Transformer在收敛速度和最终误差上接近Heavy‑Ball，显著优于梯度下降教师；无位置编码或仅第一阶段训练的模型表现与梯度下降相近；仅第二阶段训练不稳定。

**⚠️ 局限性**

实验与理论仅限于线性自注意力与线性回归任务；理论假设包含梯度流、n≈d、超多项式规模限制；未考虑softmax注意力、真实数据集、或更复杂学习任务；仅在单机实验中验证。

---

## 374. I$^2$-KAN: Interpretable Intra- and Inter-Modal Multi-Pathway Representation Learning for Multimodal Sentiment Analysis

**arXiv ID:** 2610.31087 | [PDF](https://arxiv.org/pdf/2610.31087v1)

**作者:** Xiaoting Xie `[一作]` (Shandong University), Yujun Li `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 I^2‑KAN 这一解释性多模态情感分析框架，显式区分单模态与双模态信息并在多路径结构中维护其独立与交互作用。

**💡 创新点**

创新点包括：① 将单模态和所有两两交互路径分离存储，保证路径关联可追踪；② 引入 Task‑Decoupled Linear‑Gated Denoising Bottleneck (TDL‑GDB) 进行任务分离的去噪压缩和门控信息选择；③ 采用结构化 Kolmogorov–Arnold Network (KAN) 进行非线性融合，既保持路径关联又提供可解释的函数边缘。

**🔧 技术方法**

技术手段包括多模态编码器（DeBERTa‑v3、COVAREP、FaceNet512 / Chinese BERT、OpenSMILE、OpenFace 等）、TDL‑GDB、结构化 KAN、任务分离的监督（回归、二分类、细粒度分类）以及数据扰动去噪。

**📊 数据集**

使用三大公开数据集：CMU‑MOSI、CMU‑MOSEI（英文）和 CH‑SIMS v2（中文）。

**📈 对比分析**

与多种基线（Graph‑MFN、MMIM、KAN‑MCP、I2RL、FUSE‑Net 等）进行比较，I^2‑KAN 在 Acc7/Acc2/F1/MAE/Corr 等指标上均位列第一或接近第一，尤其在 CMU‑MOSEI 与 CH‑SIMS v2 的二分类和回归任务上表现最优，显著提升整体性能。

**⚠️ 局限性**

局限性包括：① 需要预训练的多模态编码器与较大模型参数；② 交互路径的组合与选择仍需经验调优，可能不适用于所有领域；③ 解释性主要基于路径掩码和功能边缘，可解释性尚未能量化因果贡献；④ 论文仅在三大公开数据集上验证，跨领域泛化能力尚待进一步评估。

---

## 375. A General-Purpose Dual-Path Interaction Framework for Multi-View Medical Image Segmentation: Alleviating Data Volume Dependency in Small-Sample Regimes

**arXiv ID:** 2610.31065 | [PDF](https://arxiv.org/pdf/2610.31065v1)

**作者:** Shuhuan Wang `[一作]` (Northeastern University), He Ma `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种通用可插拔的双路径交互框架DPIF，用于多视角医学图像分割，显著降低对大量标注数据的依赖；

**💡 创新点**

创新点在于将跨视角的解剖互补性通过Sinkhorn最优传输实现全局分布级对齐、双层特征解耦（基于条件互信息）以及持久同调正则化相结合，形成无配对样本的端到端知识迁移；

**🔧 技术方法**

核心技术包括Sinkhorn最优传输、条件互信息约束的特征解耦、基于不确定性的自适应交互权重以及可微分的持久同调（Topological Consistency）正则；

**📊 数据集**

使用了三组多视角数据集：私有多中心心脏超声（长轴与短轴）、FLARE2023肝脏CT、MSD海马MRI，覆盖不同模态和解剖结构；

**📈 对比分析**

与单路径基线以及五种现有多视角方法对比，DPIF在所有三组数据集上均实现了平均DSC提升13.3%（长轴）/8.97%（短轴），并在少量标注（40%-60%）下匹配甚至超越完整数据单路径模型；

**⚠️ 局限性**

局限性包括目前仅支持双视角交互，持久同调近似可能忽略细微拓扑特征，需要更精确的可微分拓扑求解器，且未验证对三维或更多视角的扩展。

---

## 376. Adaptive Uncertainty Visualization With jk-Plots

**arXiv ID:** 2610.31050 | [PDF](https://arxiv.org/pdf/2610.31050v1)

**作者:** Nikolaus Piccolotto `[一作]` (TU Wien), Markus Bögl `[通讯]` (TU Wien)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 jk-plot，一种在 Bayesian 后验框架下将可行参考模型与原始数据并置的可视化方法，支持饼图、直方图、箱线图、累计分布函数等多种图形；

**💡 创新点**

将 Kruschke‑style 可视化扩展为 jk‑plot 家族，首次将小样本不确定性通过可视化自适应方式传递，并提供三种聚合策略（子集、带宽区间、梯度）；

**🔧 技术方法**

利用 Bayesian 推断（Stan HMC 采样）生成后验分布，随后在 R 中实现三种可视化聚合策略，将参考模型与观测数据共同绘制；

**📊 数据集**

使用合成数据集：多项式分布的分类样本、Student(3,0,1) 连续样本以及 Gamma(1,1/10) 畸形样本，样本量分别为 10、100、1000；

**📈 对比分析**

通过与传统经验图（经验饼图、eCDF、箱线图）在相同样本量下进行可视化对比，评估不确定性传递与视觉适应性；实验显示 jk‑plot 在小样本时更准确反映参数不确定性，并随样本增大逐渐收敛到经验图；

**⚠️ 局限性**

局限包括模型与数据不匹配（如偏态或多模态分布）、对先验设置的敏感性、对非统计专业观众的可解释性有限，以及极小样本时后验可能不稳定或无法收敛。

---

## 377. Evaluating Artificial Wisdom (AW): A Trajectory-Based Framework for Context-Sensitive Decision Regulation

**arXiv ID:** 2610.31012 | [PDF](https://arxiv.org/pdf/2610.31012v1)

**作者:** Md. Jalil Piran `[一作]` `[通讯]` (Sejong University), Md. Jalil Piran (Sejong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了基于决策轨迹的人工智慧评估框架，结合版本化规范、阶段性判断标准和过渡性适用性指标，并给出了前瞻性验证协议和瓶颈得分的计算方法。

**💡 创新点**

创新点在于：①将规范索引的适用性与阶段性判断和过渡性稳定性结合，形成非补偿式瓶颈评分；②通过模拟与实证验证探讨评估的精度、功效和瓶颈估计行为；③提供完整的前瞻性验证协议，包括可复制性、增量效度与过渡效价三项假设。

**🔧 技术方法**

技术手段包括：混合效应序数回归模型、蒙特卡洛仿真、Brier损失评估、岭回归逻辑回归、场景聚类不确定性估计、人工专家/利益相关者评审与裁定，以及结构化响应记录与查询规则。

**📊 数据集**

数据集为：①在科学研究、教育、公共基础设施和组织资源分配四个领域设计的256个脚本化情景（192个留存用于验证），每个情景包含6个阶段；②71条临床决策场景的人工评分数据；③用于功效与瓶颈估计的模拟生成数据。

**📈 对比分析**

评估方法：将人工智慧评分与已有的能力评分、元认知判别、Brier损失等基线进行增量比较，采用留一族群交叉验证。模拟结果显示，12个模型族群下，检测到的增量Brier改进≥0.01的概率可达0.742；在真实数据中，增量效度仍处于探索阶段，尚未达到正式验证阈值。

**⚠️ 局限性**

局限性：①评估依赖大量手工注释的脚本化情景，工作量大且易受专家偏差影响；②外部有效性受限于情景设计与可用模型族群；③瓶颈得分在接近平局时偏向下，且估计不稳定；④模拟假设与实际决策场景差异；⑤尚无证据证明该评分能预测独立决策结果，且不适用于直接授权部署。

---

## 378. Matching Tasks to Agents: Mechanism Design for Multi-Agent Contracts

**arXiv ID:** 2610.31002 | [PDF](https://arxiv.org/pdf/2610.31002v1)

**作者:** Matteo Bollini `[一作]` (Politecnico di Milano), Alberto Marchesi `[通讯]` (Politecnico di Milano)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在代理人隐私信息和隐藏努力的情况下，任务分配的最优机制设计，并给出了单任务问题的Myerson式解析和多任务问题的近似方案。

**💡 创新点**

创新点在于将代理人多维类型压缩为成本与成功概率之比，得出单任务的简单Myerson式最优机制，并证明多任务问题无最优解，提出常数代理数下的多任务近似算法。

**🔧 技术方法**

采用Myerson铁算方法、单调性与虚拟效用分析、线性规划与凸组合技术，以及复杂度与可行域分析。

**📊 数据集**

本研究为理论分析，无使用具体数据集。

**📈 对比分析**

对单任务，可在多项式时间求得最优机制，价格效用比为Θ(1/(NK))；对多任务，在代理数常数的前提下，提供多项式时间的FPTAS，得到(1−ε)近似解。

**⚠️ 局限性**

局限在于多任务问题一般不存在最优解，且近似算法仅适用于常数代理数，无法扩展到大规模实例。

---

## 379. Demonstration-Based Robotic Feeding Using Dynamic Movement Primitives

**arXiv ID:** 2610.31020 | [PDF](https://arxiv.org/pdf/2610.31020v1)

**作者:** Kenzhi Iskandar Wong `[一作]` (Nanyang Technological University), Domenico Campolo `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究构建了一个基于演示的机器人喂食框架，结合人类演示、DMP运动生成、RGB‑D口部感知和AprilTag定位，实现了从示范到实际执行的全流程；

**💡 创新点**

创新点在于将DMP运动映射与实时环境感知（口部位置和托盘位姿）结合，能够在不同场景下自动重定位演示轨迹，提升了机器人喂食的适应性；

**🔧 技术方法**

采用的技术包括动态运动原语（DMP）编码与重定向、ZED 2i RGB‑D相机口部检测、AprilTag框架注册、ROS 2通信与笛卡尔阻抗控制；

**📊 数据集**

使用的数据集为自采的 PhaseSpace 运动捕捉演示数据和 ZED 2i 产生的 RGB‑D 图像，未使用公开数据集；

**📈 对比分析**

实验结果主要以定性方式展示，未与基准方法进行定量对比，性能通过演示视频和轨迹可视化验证其可行性；

**⚠️ 局限性**

局限性包括仅使用离线口部感知，无法实时应对参与者运动或口部闭合；仅采用单次演示，缺乏多样化示例；缺乏在线轨迹调整与安全性评估。

---

## 380. To Route or to Adapt? Multi-Expert Diffusion Planners in Unseen Regions

**arXiv ID:** 2610.31095 | [PDF](https://arxiv.org/pdf/2610.31095v1)

**作者:** Yun Li `[一作]` (University of Tokyo), Manabu Tsukada `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在未见区域部署基于扩散模型的轨迹规划器时，比较专家选择与适配器微调两种策略对轨迹偏移的影响，并在多域训练和目标域少量数据下评估不同方法。

**💡 创新点**

创新点在于系统量化专家选择的提升空间（Oracle headroom 仅 0.9 m），证明仅用 20 个目标域场景的奖励微调（joint replay）能显著降低开放式轨迹偏移；同时提出地理加权混合、共享适配器修复等替代方案，并对比八种路由策略。

**🔧 技术方法**

使用技术包括：奖励驱动的扩散规划器（Diffusion Planner），共享 LoRA 与每域专家适配器结构，八种专家路由方法（MLP、self‑consistency、conformal、reward‑gate 等），地理加权权重混合，共享适配器几何扰动修复，以及 joint replay（目标域与训练域场景混合微调）。

**📊 数据集**

使用数据集：四个训练 ODD（Hongo、Kashiwa、Shalun、Nagoya）；三个未见地图（Knowledge Hub、Ishiodai、Morikoro）；nuPlan 跨城评估；以及 20 个目标域场景（Ishiodai）进行微调。

**📈 对比分析**

对比方法：与静态默认专家选择（使用最近的训练域专家）比较；Oracle 选取最低偏移专家；八种路由方案；地理加权、共享适配器修复、joint replay。性能结果显示：Oracle 仅比默认低 0.9 m；joint replay 在 Pool A 上 fleet‑mean 偏移 6.9 m，低于默认的 12.4 m；nuPlan 跨城 fine‑tune 将 open‑loop 偏移从 8.64 m 降至 5.60 m；其余路由方法均不优于默认，且某些导致偏移增加。

**⚠️ 局限性**

limitations：评估仅为开放式轨迹偏移，闭环性能在 nuPlan 中下降；未在真实车辆上测试 joint replay；数据规模有限（每地图仅数十条场景）；只评估四个专家的选择，未检验单一统一模型；地理加权效果在不同区域不稳定。

---

## 381. What Pairwise Evaluation Misses: Reward Scores across Prompts

**arXiv ID:** 2610.31090 | [PDF](https://arxiv.org/pdf/2610.31090v1)

**作者:** Mateusz Modrzejewski `[一作]` (Warsaw University of Technology), Tomasz Radzikowski `[通讯]` (Warsaw University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究奖励模型中与提示相关的共享分量（per‑prompt term），并提出以人类评价为参照的三维审核方法：跨提示方差份额、同级评分回归和提示替换，随后在18个公开评分器和6个训练条件上进行实验。

**💡 创新点**

创新点在于：①首次将人类跨提示方差作为基准来衡量奖励模型的提示敏感度；②提出了包含同级评分回归的完整审核工具包；③系统评估了去除提示均值对全局筛选的影响，并探讨了训练目标对提示敏感度与方差份额的作用。

**🔧 技术方法**

使用了方差分解（Intraclass Correlation）、标准化同级回归、提示替换实验（随机提示、相似提示、长度匹配提示、相关任务提示）、平均分中心化、配对和无配对训练（Bradley–Terry 与回归目标）等技术。

**📊 数据集**

主要数据集包括：HelpSteer2（双响应顺序评分），HelpSteer（四响应独立评分），Stanford Human Preferences（Reddit帖子与投票），以及公开的奖励模型基准 RewardBench、RewardBench 2。

**📈 对比分析**

比较方法：通过跨提示方差份额与人类份额对比、提示替换下的匹配惩罚、同级回归斜率、中心化前后与人类评分的 Spearman 相关性、以及在 HelpSteer 30% 最高分选择中的平均帮助度。实验表明：奖励模型的跨提示方差份额普遍高于人类，去除提示均值并未提升全局选择性能，且训练目标不同会显著影响提示敏感度与方差份额。

**⚠️ 局限性**

局限性：仅使用英文评分集且每提示响应数有限，导致方差份额估计不够精确；未能分离响应来源与评分协议的混杂影响；提示替换实验使用的相关提示可能不具对抗性；中心化实验仅在 HelpSteer 上评估，未覆盖更广泛任务；匹配训练实验仅在同一模型族内部进行，结果不一定能推广至其他族。

---

## 382. MQAScore: Fine-Grained Text-Music Alignment Evaluation via Large Audio Language Model

**arXiv ID:** 2610.31083 | [PDF](https://arxiv.org/pdf/2610.31083v1)

**作者:** Wei-Jaw Lee `[一作]`, Yi-Hsuan Yang `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了MQAScore，对文本到音乐的对齐进行五维（流派、乐器、情绪/主题、声乐、使用场景）的原子概念级评估。

**💡 创新点**

创新点在于：①基于CAL500构建的五维概念层次；②将全句评估拆分为单概念问答并使用MQAScore；③引入敏感性-不变性比率（R_S/I）衡量评估器对语义替换的选择性；④通过概念级评估显著提升了评估精度。

**🔧 技术方法**

主要技术包括大音频语言模型（Qwen3‑Omni、Audio Flamingo Next、Music Flamingo）进行问答式评估、对比模型CLAP与MuQ‑MuLan用于相似度评估，以及通过LLM提取概念和生成正负样本。

**📊 数据集**

使用的数据集有GTZAN（流派标签）、OpenMIC‑2018（乐器标签）、MusicCaps（多维概念与对齐测试）以及人工专家对齐标注。

**📈 对比分析**

与传统全句相似度评估相比，MQAScore在概念级别上取得更高的AUC和R_S/I，并且与人工专家的一致率可达90%以上，表明其在情绪、声乐和使用场景等抽象维度上优于相似度基准。

**⚠️ 局限性**

局限性包括：依赖LLM进行概念提取和负样本生成，GTZAN单标签假设限制了多流派评估，评估范围仅限于描述性音频概念，专家标注样本规模有限，以及在极为抽象的维度（如使用场景）仍存在挑战。

---

## 383. XALIGN: A Human-centric Visual Analytics Approach to Understanding AI Explanations

**arXiv ID:** 2610.31015 | [PDF](https://arxiv.org/pdf/2610.31015v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 384. When Forgotten Facts Return: Relearning-Aware Unlearning of Knowledge Graph Embeddings via Tangent Kernels

**arXiv ID:** 2610.31075 | [PDF](https://arxiv.org/pdf/2610.31075v1)

**作者:** Parnian Fazel `[一作]`, Hamed Haddadi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究知识图谱嵌入的遗忘问题，提出一种利用切线核闭式预测持续训练后遗忘效果的 Kernel‑Anticipated Unlearning 方法

**💡 创新点**

创新点在于用切线核（NTK）近似预测持续训练对被遗忘事实得分的影响，并将其差分传播进遗忘损失，从而显著降低继续训练导致的再学习

**🔧 技术方法**

使用切线核、梯度闭式响应、一次线性化、梯度下降/Adam、实体行编辑、保留梯度正则化等技术

**📊 数据集**

在公开基准 FB15k‑237 与 CoDEx‑L 上，使用 TransE、TransH、RotatE 三种嵌入架构进行实验

**📈 对比分析**

与两种现有遗忘器（GraphDPO、ZeroFisher）及多种再学习防御（NTK 投影、SAM、TAR）对比，KAU 在泄漏率上降低 46–82%，同时保持或提升保留事实的 MRR

**⚠️ 局限性**

局限性包括仅考虑实体行编辑的线性近似，未涉及关系行变化，对极端持续训练步骤或高维图的适用性尚需进一步验证

---

## 385. Multichoice Modeling: Exploring Decision Configuration Space under Uncertainty

**arXiv ID:** 2610.31038 | [PDF](https://arxiv.org/pdf/2610.31038v1)

**作者:** Mariia Kozlova `[一作]` (LUT University), Andrea Saltelli `[通讯]` (UPF Barcelona)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并实现了多选建模（multichoice modeling）框架，能够在不确定性下显式构建大规模决策配置空间，并通过三个案例（房贷、制造工艺升级、并购策略）进行验证。

**💡 创新点**

创新点在于将决策架构与不确定性空间结合，采用崎岖景观（rugged landscape）思路，利用方差基全局敏感度分析与条件分布分析对决策组合进行探索，突破传统只评估预设策略的局限，揭示互补效应与鲁棒配置。

**🔧 技术方法**

核心技术包括：蒙特卡洛仿真、Sobol方差分解与binning估计、条件分布可视化、指向超图（directed hypergraph）表示选择依赖、策略表（strategy table）构建决策逻辑、随机过程建模（如Vasicek、随机游走）以及无代码可视化仪表盘。

**📊 数据集**

使用的“数据集”主要是各案例的业务参数与不确定性分布，例如房贷利率参数、制造工艺成本与升级成本、并购整合成功率与市场吸引力随机过程等；所有结果均通过内部仿真产生的合成数据实现。

**📈 对比分析**

方法比较：与传统的仅不确定性分析（固定策略下的蒙特卡洛评估）相比，多选建模能够探查更广的配置空间，发现更具互补性、低方差或高收益的配置；案例中显著提升了成本降低幅度、风险分散效果与决策可行性。

**⚠️ 局限性**

局限性：需构建完整的策略表与依赖图，工作量大；对大规模配置空间仍需大量仿真样本，计算成本高；方差敏感度可能忽视局部重要性，需要进一步区域敏感度或交互分析；模型依赖对不确定性与依赖关系的准确性；目前仍为探索性方法，尚未提供全局最优解。

---

## 386. Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters

**arXiv ID:** 2610.31017 | [PDF](https://arxiv.org/pdf/2610.31017v1)

**作者:** Martino M. L. Pulici `[一作]` (Bosch Center for Artificial Intelligence), Volker Tresp `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Llama‑3.1‑8B‑Instruct上，针对150篇维基百科文章训练单一LoRA适配器，并比较open‑book（带文档）评测与cloze（不带文档）评测，研究不同合并与检索方法对知识保留的影响。

**💡 创新点**

证明open‑book评分主要衡量适配器共享的读取行为，而非实际保留文档内容；随机适配器可以恢复大约四分之五的open‑book优势，却几乎不恢复cloze优势。

**🔧 技术方法**

使用LoRA细调、更新平均合并（update‑merging）、线性插值（lerp）、ties合并、联合训练、权重空间与句向量检索等技术。

**📊 数据集**

数据集为150篇英文维基百科文章（按学科划分），每篇产生12个QA对，8个用于训练，4个用于评估（3个事实+1真/假）。

**📈 对比分析**

与基线（未适配模型）及同一文档专属适配器进行比较；open‑book指标显示随机合并可恢复≈88%优势，cloze指标随机合并仅恢复≈2%；联合训练在cloze上可恢复≈94%优势，表现优于合并。

**⚠️ 局限性**

仅在单一模型、单一适配器容量和训练调度下验证；仅使用百科文章，未检验不同文本类型或更大模型；cloze与open‑book使用的样本极少，无法深入分析单文档差异；检索实验未包含查询自身适配器，缺乏完整的参数化检索评估。

---

## 387. Multi-Tool Selection for Robotic Systems: Comparing Semantic Retrieval and LLM

**arXiv ID:** 2610.31013 | [PDF](https://arxiv.org/pdf/2610.31013v1)

**作者:** Miguel Sousa `[一作]` (Universidade de Lisboa), João P. Matos-Carvalho `[通讯]` (Universidade de Lisboa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨如何将自然语言请求映射到机器人系统中预定义工具的最小完整子集，实现机器人工具路由；

**💡 创新点**

提出了三种逐步改进的密集检索策略（固定阈值、相对距离、句子片段聚合）并与大型云端LLM进行对比，显示句子片段聚合在本地部署下能显著提升路由质量；

**🔧 技术方法**

使用ChromaDB进行向量检索，LLM采用120B参数云端模型，评估了不同检索策略的精确度、完整性、精度及延迟；

**📊 数据集**

使用42种机器人工具的手工标注工具目录和164条按复杂度分层的自然语言请求数据集；

**📈 对比分析**

在精确度、完整性、精度等指标上，CDBv3的宏平均复合分数为0.7822，高于CDBv2和CDBv1；LLM最高为0.8869；CDBv3的延迟仅毫秒级，而LLM则为秒级；

**⚠️ 局限性**

CDBv1–CDBv3的超参数是基于同一测试集手工调优，缺乏独立验证集，结果可能偏高；

---

## 388. Formal-Informal Mathematical Alignment through and beyond Human Annotations

**arXiv ID:** 2610.31011 | [PDF](https://arxiv.org/pdf/2610.31011v1)

**作者:** Noah Gießing `[一作]` (FIZ Karlsruhe), Moritz Schubotz `[通讯]` (FIZ Karlsruhe)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于专家注释的Stacks Project与Lean数学库（mathlib）之间的正式-非正式对齐数据集，并在此数据集上评估了语义检索与LLM重排序、LLM评判等技术，进一步发现了未被标注的对齐条目。

**💡 创新点**

首次提供了高质量、专家验证的正式-非正式对齐语料库；提出将LLM作为判别者与检索器结合的端到端管道，用于自动发现新的对齐；通过多轮投票提升判别精度。

**🔧 技术方法**

语义检索引擎LeanSearch（利用Dense Passage Retrieval生成非正式化描述），LLM（Qwen3.5）用于重排序与判别（Exact/Equivalent/General/Implies/No/Invalid），以及投票机制。

**📊 数据集**

SPmathlib（482条专家标注的对齐，覆盖mathlib 3%文件）、完整mathlib库（约257k定理）以及Stacks Project全部20,950条条目。

**📈 对比分析**

使用R@1/R@5/R@10和MRR@10评估检索效果；未重排序时R@1=0.36，重排序后提升到0.46（+28%）；LLM判别在Exact/No类别上准确率≈80%，General/Equiv略低；投票提升Exact准确率至≈92%。

**⚠️ 局限性**

LLM在General/Equiv判别上仍不稳定，需人工复核；投票与大模型消耗计算资源；未覆盖所有数学知识，且对尚未出现的概念的发现受限。

---

## 389. Revisiting Distribution Matching for State-to-Vision Policy Distillation in Laparoscopic Robotic Manipulation

**arXiv ID:** 2610.31010 | [PDF](https://arxiv.org/pdf/2610.31010v1)

**作者:** Mingyu Sheng `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种利用状态到视觉的策略蒸馏框架，借助分布匹配将已训练的状态基SAC策略迁移到仅使用RGB图像和机械臂本体信息的视觉策略；

**💡 创新点**

创新点在于将分布匹配（FKL、RKL、JD、WD）作为蒸馏损失，并对其梯度、Hessian性质进行系统分析，证明不确定性自适应蒸馏（FKL、JD）在复杂手术任务中更稳定、更高效；

**🔧 技术方法**

采用Soft Actor‑Critic (SAC) 训练教师策略，利用Gaussian政策的闭式分布匹配实现蒸馏；采用三层卷积+三层全连接的CNN网络；用Adam优化器与固定学习率训练；

**📊 数据集**

使用SurRoL仿真环境的腹腔镜机器人任务（NeedleReach、GauzeRetrieve、NeedlePick、PegTransfer、NeedleRegrasp）作为数据集；

**📈 对比分析**

将视觉RL（CURL、DrQv2）、IL（ALOHA）与四种蒸馏损失（FKL、RKL、JD、MSE）对比；在复杂任务上蒸馏方法平均成功率约70%，显著优于RL方法，且在多任务中表现更稳定；

**⚠️ 局限性**

局限在于实验仅在仿真环境中验证，未在真实机器人上部署；网络结构较简单，可能限制对更复杂场景的适应；并且对教师策略的依赖可能导致知识迁移受限。

---

## 390. SecLog-Bench: A Scalable Benchmark for Cybersecurity Log Reasoning with Scaffolded Evidence

**arXiv ID:** 2610.31000 | [PDF](https://arxiv.org/pdf/2610.31000v1)

**作者:** Yifan Wang `[一作]` (Shandong University), Yue Yao `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个可扩展的多源日志推理基准 SecLog‑Bench，涵盖 29 个可复现攻击场景，生成 216 个含主机审计日志、应用日志和网络流量的窗口，并对每个窗口标注四步结构化推理链，用于评估 LLM 在攻击阶段预测与推理可靠性方面的表现。

**💡 创新点**

创新点包括：① 通过轻量级辅助索引实现多源日志跨源对齐与窗口化；② 引入 scaffolded 证据和四步参考推理链，使评估聚焦于推理过程而非单一标签；③ 设计了 reasoning‑aware 评估协议，将攻击阶段预测与推理可靠性、预测‑推理一致性统一量化；④ 通过实验验证压缩、scaffolded 展示及三源融合显著提升 LLM 推理效果。

**🔧 技术方法**

采用容器化模拟（Docker）搭建攻击场景，使用 audit‑triple 结构化主机行为，利用轻量化索引实现日志对齐；构造四步推理链（scope、facts、relation、judgment）；评估时调用 8 种 LLM（Qwen、Llama、Gemini、Claude、GPT 等），并用 GPT‑5.4‑mini 与 Gemini 2.5 Pro 两个评估助手对推理进行可靠性打分，最终得到 Accuracy、Macro‑F1、SRP、SCC、SRF1 等指标。

**📊 数据集**

数据集由 29 个可复现攻击场景生成，涵盖 29 种不同实现栈和 4 类非 CVE 场景，生成 216 个窗口（每类 54 个），每窗口包含多源日志压缩后视图和对应的四步参考推理链。数据规模约 148 GiB，覆盖 166 579 条主机审计日志、157 369 057 条应用日志、1 599 414 条网络流量记录。

**📈 对比分析**

评估方法：对 8 种 LLM 进行统一推理任务，输出攻击阶段标签和自述推理链；利用固定的评估协议计算 Accuracy、Macro‑F1、SRP、SCC、SRF1，并对预测‑推理一致性进行 TT/TF/FT/FF 分析。实验结果显示：最佳模型 Claude Sonnet 4.6 的 Accuracy 为 44.44%、Macro‑F1 为 45.07%；但推理可靠性最高的 GPT‑5.4 的 SRF1 仅为 54.87%，且最佳 Accuracy 与最高 SRF1 并不对应，说明正确预测不必然伴随可靠推理。相较于原始多源日志，压缩+scaffolded 展示和三源融合可将 Accuracy 提升至 50% 以上，SRF1 提升至 44% 以上。

**⚠️ 局限性**

局限性包括：① 只评估攻击相关窗口，未覆盖窗口检索、异常检测与真实场景中的恶意/正常区分；② 样本量 216 较小，且标签平衡与运营环境中的不均匀分布不符；③ 依赖 AI 辅助标注与评估，可能引入偏差；④ 阶段标签粒度有限，未覆盖细粒度的攻击进度；⑤ 评估结果受 LLM 版本更新、API 变化和推理设置影响，复现性受限。

---

## 391. Co-Recursive Semantic Self-Extension: A Mathematical Foundation for Open-Ended Intelligence

**arXiv ID:** 2610.31097 | [PDF](https://arxiv.org/pdf/2610.31097v1)

**作者:** Simon Hegelich `[一作]` (GaiaLogic Unipessoal Lda), Christian Montag `[通讯]` (University of Macau)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一套基于康德的先验结构与哥德尔不完备性原理的开放式 AGI 设计框架，并给出了七条设计定律。作者构建了一个递归自我修正的形式化体系，展示如何通过元层次的目的转化、语义盈余、共递归扩展等机制实现持续的语义生成与目的重塑。随后用 Kolmogorov 复杂度证明任何固定架构（有限状态描述、可计算转移机制和初始条件）无法产生持久的算法新颖性，从而界定了开放式智能的可实现边界。

**💡 创新点**

创新点包括：
1) 将康德的先验结构与哥德尔的不完备性视为工程启发，而非纯数学限制；
2) 引入“语义自扩张”概念，强调智能需将未解决结构转化为新的意义空间；
3) 定义七条相互依赖的设计定律（双层结构、可不完备性、语义盈余、递归、聚合、记忆与递归目的），形成闭合的递归循环；
4) 通过形式化的“可反射扩展”与“可判定盈余”证明了递归目的在理论上可持续；
5) 用 Kolmogorov 复杂度给出固定架构不可生成无限开放式历史的不可行性论证。

**🔧 技术方法**

技术与方法：
- 形式逻辑与可计算性理论（哥德尔不完备性、Gödel 机、证明搜索）
- 语义与知识表示（可自指的符号系统、共递归语言扩展）
- 复杂度理论（Kolmogorov 复杂度、信息不变性）
- 递归系统与元层次推理（meta‑teleology、递归目的转移）
- 结构化记忆与上下文切换（polycontexturality、递归记忆）

**📊 数据集**

论文不涉及具体数据集，所有讨论均为理论性推导与形式化证明，侧重于架构设计与逻辑分析。

**📈 对比分析**

方法比较：
- 与规模假设（Scaling Hypothesis）比较，指出单纯扩展参数无法突破语义边界；
- 与 GOFAI 对比，说明硬编码规则缺乏不完备性反射；
- 与神经符号方法对比，指出其在自适应与不完备性方面仍受限；
- 与 (C,U,V) 架构对比，说明固定价值向量导致停止问题。论文未给出实验性能指标，而是通过理论分析证明相对优越性。

**⚠️ 局限性**

局限性：
- 主要为理论框架，缺乏可实现的原型或实验验证；
- 对可计算性与形式化系统的假设（可有效公理化、一致性等）可能在实际实现中难以满足；
- 复杂度论证给出的不可能性基于理论上无限递归，未说明实际系统在有限资源下如何接近理论极限；
- 对外部记忆与环境交互的需求未给出具体实现细节；
- 论文关注的是结构与原理，未涉及对齐、伦理或安全等实际部署问题。

---

## 392. nnInteractive Agent: General-Purpose AI Agents as 3D Image Annotators

**arXiv ID:** 2610.31080 | [PDF](https://arxiv.org/pdf/2610.31080v1)

**作者:** Fabian Isensee `[一作]` (German Cancer Research Center), Klaus H. Maier-Hein `[通讯]` (German Cancer Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

本研究实现了一个通用 AI 代理（nninteractive-agent），利用工具和技能控制 nnInteractive 模型完成 3D 医学图像的交互式分割，代理可识别目标、下单提示、检查并纠正分割结果，完全不需要额外训练。

**💡 创新点**

创新点在于将通用多模态语言模型与可提示分割模型结合，形成训练自由的“代理+工具”体系，使代理能够在多模态环境中完成从自然语言请求到 3D 分割的完整工作流，并在多物种、多模态图像上保持高准确度。

**🔧 技术方法**

使用技术包括：Claude Code/Codex 语言模型、Model Context Protocol（MCP）工具接口、nnInteractive 分割模型、Python 代码工具、可视化渲染（切片、3D 表面）以及一个 Markdown 形式的技能文件。

**📊 数据集**

评估数据集包含 14 张公开医学图像（CT、MRI、PET、微型 CT），涵盖人类、老鼠和蚂蚁，51 个目标（器官、肿瘤、肌肉、骨骼、血管树、昆虫脑）。

**📈 对比分析**

对比方法包括：无导引与导引两种轨迹、单点提示、模拟专家提示（oracle）、文本提示模型 VoxTell；结果显示代理在无导引轨迹下平均 Dice 为 0.81‑0.84，在导引轨迹下为 0.85，几乎与模拟专家提示相当；相比 VoxTell，代理在标准医学目标上更稳健，在非标准目标上表现更好。

**⚠️ 局限性**

局限性包括：对细长结构（血管、神经）仍易失效；代理完成一张图像需数分钟且 API 成本约 1.3‑1.5 美元；自评准确度不足，导致停止点早于 oracle；未与人类标注者直接比较。

---

## 393. Federated Differentially Private Optimization via Low-Rank Preconditioning

**arXiv ID:** 2610.31069 | [PDF](https://arxiv.org/pdf/2610.31069v1)

**作者:** Antti Koskela `[一作]` (Nokia Bell Labs), Elias Raninen `[通讯]` (Nokia Bell Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种联邦差分隐私（DP）低秩协方差估计框架，包含DP秩选择、DP幂迭代和投影协方差发布，并将其用作优化的预处理器。

**💡 创新点**

创新点：①实现每轮通信量仅为 O(pk)，相比传统 O(p²) 的协方差发布显著节省通信；②所有步骤均为加法式本地贡献，兼容安全聚合，提供端到端的中心 DP 保障；③利用低秩预条件器将 DP 二阶优化融入联邦学习，显著减少所需通信轮数。

**🔧 技术方法**

使用技术：DP 幂迭代、Analyze Gauss 机制、随机投影秩估计、Gaussian 机制组合、Secure Aggregation、低秩投影预处理器、闭式收敛分析与误差界定。

**📊 数据集**

实验数据集：MNIST、UCI HAR、SVHN、FashionMNIST、CIFAR‑10/100（使用预训练 ResNet‑34 与 ViT‑B/16 的特征向量）。

**📈 对比分析**

对比方法：集中式 O(p²) Analyze Gauss、局部 DP 联邦方法、DP‑GD、DP‑GD+动量、DP‑Adam。实验表明：在相同或更低通信量下，与集中式基线性能相当或更优；预处理器在大多数隐私预算下收敛更快、准确率更高；仅在极低预算下，DP‑GD 仍保持竞争力。

**⚠️ 局限性**

局限性：极低隐私预算时，子空间估计噪声影响较大，收敛速度受限；秩估计步骤消耗额外 DP 预算；目前仅针对全批量线性/GLM，尚未扩展到随机小批量或深度网络；对客户端数过多时仍需更高的噪声来维持安全聚合。

---

## 394. AstroBench: Benchmarking Free-Floating Capture and Space Robotic Manipulation

**arXiv ID:** 2610.31064 | [PDF](https://arxiv.org/pdf/2610.31064v1)

**作者:** Zizhe Wang `[一作]` (Astroup Lab), Yao Zhu `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 AstroBench 空间机器人基准，集成了自由漂浮物体捕获、MELFI 样本存储以及 RRM 组件维修三大任务，并提供了 8,414 件对象实例和可扩展的硬件工作单元。

**💡 创新点**

创新点在于：① 将自由漂浮捕获与固定基座机器人操作结合，形成统一的任务框架；② 采用 Objaverse 过滤得到的大规模对象库存，支持多种几何形状；③ 将真实 NASA/ESA 的 MELFI 与 RRM 参考模型转化为可执行的工作单元，提升实验可重复性；④ 公开完整的任务定义、语言指令、视觉环境与评估脚本，为空间机器人学习研究提供统一基准。

**🔧 技术方法**

使用视觉语言动作学习模型（DROID 与 ACT），在 NVIDIA Isaac Sim 的 GPU PhysX 物理仿真中进行仿真；使用固定基座 Panda 机械臂配 Robotiq 2F‑85 抓手；评估时采用预训练检查点、行动块规划与实时反馈等技术。

**📊 数据集**

数据集包括：① Objaverse 过滤后的 8,414 个 3D 物体；② NASA 与 ESA 的 MELFI 与 RRM 参考模型；③ NASA Astrobee 室内视觉背景；④ 记录的实验轨迹与捕获判定（包括 400 次 DROID 试验和 20 次 ACT 试验）。

**📈 对比分析**

对比方法：在相同的物理环境下对比冻结的 DROID checkpoint（π0.5）和针对 Rubik’s Cube 训练的 ACT；评价指标为捕获成功率。结果显示：DROID 仅在 20 个对象、两种动态条件下总计 6/400（1.5%）捕获；ACT 在 10 个 Cube 位置、两种动态条件下 14/20（70%）捕获，体现任务专门化学习优于通用转移。实验表明动作规划间隔与动态目标移动对捕获性能影响显著。

**⚠️ 局限性**

局限性：① 迁移性能极差，DROID checkpoint 仅在极少对象成功；② 任务覆盖有限，实验仅涵盖 20 个对象与单一 Cube，未验证对更大对象库的泛化；③ 物理仿真模型简化，未考虑真实硬件摩擦、力学细节与飞行环境；④ 所有实验均在仿真环境完成，缺乏真实飞行或实验室硬件验证；⑤ 评价指标仅为捕获成功率，未深入分析接触动力学、感知误差与控制策略。

---

## 395. Diagnosing the Explicit-to-Applied Theory of Mind Gap in LLMs

**arXiv ID:** 2610.31060 | [PDF](https://arxiv.org/pdf/2610.31060v1)

**作者:** Prajvi Saxena `[一作]` (German Research Center for Artificial Intelligence), Wolfgang Maass `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究大型语言模型在理论心智（ToM）任务中，区分显式认知（识别心理状态）与应用推理（预测行为）两种能力，并探讨其差距的可恢复性；

**💡 创新点**

创新点在于提出并评估Belief‑State Test‑Time Training（BSTTT）方法，将轻量化参数更新与多种提示/上下文学习方式结合，系统地对比显式/应用差距的可恢复性与不同干预的效果；

**🔧 技术方法**

使用LoRA轻量化适配器、梯度微调（BSTTT）、多种结构化提示（MS‑Reminder、Scratchpad、SimToM、SymbolicToM）、匹配上下文学习（ICL）以及传统的推理提示；

**📊 数据集**

四个ToM基准：SimpleToM、Hi‑ToM、OpenToM、DynToM，覆盖静态/动态、一级/多级信念推理及多跳推理；

**📈 对比分析**

在Qwen‑2.5‑7B、Llama‑3.1‑8B、GPT‑oss‑20B三种开源模型上进行实验；结果显示：显式–应用差距显著（最高71.9个百分点），但通过提示或BSTTT可部分恢复；BSTTT在SimpleToM和Hi‑ToM上显著提升，ICL在OpenToM等任务上更优，整体最优配置往往为提示+BSTTT；

**⚠️ 局限性**

局限性包括：仅测试7–20B规模模型，未检验更大模型；使用多选题，缺少开放式生成评估；BSTTT采用每轮重置，难以处理跨情境或时间序列的持续推理；

---

## 396. Pigeonhole Equal Subset Sum: Subexponential Algorithm, Tight Lower Bounds, and Average-Case Analysis

**arXiv ID:** 2610.31056 | [PDF](https://arxiv.org/pdf/2610.31056v1)

**作者:** Deepak Bhati `[一作]` (CISPA Helmholtz), Karol Wegrzycki `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套针对平均复杂度的理论框架和算法，专门解决模整数和整数版本的限制等价子集和（ACM‑RESS、ACI‑RESS）问题，并在此基础上给出了子指数/多项式时间的平均解法和相应的下界证明。

**💡 创新点**

创新点在于：①将模 pqr 的 CRT 过滤与动态规划（DP）相结合，构造出高效的随机化过滤器；②首次给出了平均-案例的多项式/子指数下界；③针对整数问题引入进位（carry）分析，得到新的 O(2^{n/4}) 平均复杂度。

**🔧 技术方法**

技术方法包括：LLL/BKZ 基础化与高斯采样；CRT 与模 pqr 的分解；动态规划表（DP）存储部分和；概率论中的第二矩（Paley‑Zygmund）与矩阵秩估计；线性代数与组合计数技术。

**📊 数据集**

实验使用的输入数据集是随机生成的整数或模 M 的向量 a_i（i=1…n），其中 M 取满足 M=Θ(2^n) 的范围，n 为偶数。

**📈 对比分析**

与传统暴力 2^n 或基于枚举的子集和算法相比，本文方法在平均意义下的时间复杂度为 Õ(2^{n/4})（或 Õ(M^{1/4})），成功率分别为 Ω(1)（模整数）和 Ω(1/n^2)（整数），显著优于现有方法。

**⚠️ 局限性**

局限性：仅适用于平均-案例，无法覆盖所有结构化实例；整数版本的成功概率相对较低；算法参数依赖于随机化且需要满足 M=Θ(2^n) 的前提；下界证明只针对特定范围的 M，未涵盖更广泛的情形。

---

## 397. Harm Is Behavioural, Moderation Is Lexical: How Language and Culture Decide Whom Automated Safety Systems Protect and Whom They Silence

**arXiv ID:** 2610.31047 | [PDF](https://arxiv.org/pdf/2610.31047v1)

**作者:** Gabriel Lars Sabadin `[一作]` (Tuteliq AB), Sarah Kingston `[通讯]` (University of Lancashire)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文评估了五种公开内容安全系统在11种语言下的公平性与覆盖性，发现它们均主要依赖词汇而非行为，导致不同语言社区受到不平等的误标记与保护缺失；并提出基于功能对比的评估协议，建议报告语言间差距并改进行为导向检测；

**💡 创新点**

创新点在于将过度标记与保护不足视为同一根源并通过功能对比测试验证其可分离性；提出跨语言公平评估协议，将“最佳与最差服务语言间差距”作为主要指标；

**🔧 技术方法**

使用关键词词典、XLM‑RoBERTa毒性分类器、Qwen3Guard Guard模型（0.6B、8B）以及PolyGuard多语言安全模型；并采用对比功能测试、AUC、Wilson置信区间、bootstrap与置换检验；

**📊 数据集**

使用HateCheck与Multilingual HateCheck功能测试集（含对照对）、HurtLex词典、wordfreq词频表；以及公开训练的XLM‑RoBERTa、Qwen3Guard和PolyGuard模型；

**📈 对比分析**

通过将每个系统在各语言下的假阳性率与漏报率绘制雷达图，并计算AUC与与本地注释者的比较，发现所有系统均低于本地注释者基准，且更高的能力导致差距扩大，最高模型PolyGuard的差距在最佳/最差语言间可达30个百分点；

**⚠️ 局限性**

局限在于实验使用的功能测试集主要为英语与少数其他语言，未覆盖真正的低资源或代码混杂文本；缺乏针对恶意行为（诱导、敲诈、洗脑）的对照对；评估仅在单一阈值和模型版本下进行，未检验H2（功能识别的稳定性）。

---

## 398. What Does AI Success Look Like? A Mixed-Methods Analysis of Supranational AI Governance Instruments

**arXiv ID:** 2610.31030 | [PDF](https://arxiv.org/pdf/2610.31030v1)

**作者:** Delaram Golpayegani `[一作]` (Trinity College Dublin), Dave Lewis `[通讯]` (Trinity College Dublin)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2018-2025 年 13 份来自欧盟、OECD、G20、UNESCO、联合国及欧洲委员会的跨国 AI 治理文件进行混合方法分析，结合手工主题编码和 BERTopic 主题建模，提炼出 AI 成功的多层次主题框架并提出政策建议。

**💡 创新点**

创新点在于首次系统性地将手工主题分析与无监督主题建模结合，用社会期望学视角归纳 AI 成功的四大元主题（技术、用途、风险与治理）及其 10 个子主题，并提供跨层级治理协同的实务建议。

**🔧 技术方法**

使用 NVivo 进行手工主题编码，采用 BERTopic 进行无监督主题建模，辅以社会期望学的解释框架进行主题归纳与层级划分。

**📊 数据集**

构建并公开的 13 份跨国 AI 政策、指南与法规的文献语料库（2018-2025 年），覆盖欧盟、OECD、G20、UNESCO、联合国及欧洲委员会等机构。

**📈 对比分析**

通过对手工编码概念与主题模型输出的交叉比对，整合出 157 个独特概念，并以微-中-宏层级框架组织为 10 个主题；该方法未给出传统数值指标，而是以定性覆盖度和概念完整度评估其效果。

**⚠️ 局限性**

局限性包括：样本仅为 13 份文件，未覆盖国内层级政策；文件多数为非约束性文件，缺乏法律效力；手工编码可能受主观偏差；缺乏定量验证指标；并未深入探讨各国具体实施细节。

---

## 399. AERIAL: Mission-Aware LLM Selection for Language-Driven UAV Control in Flying Networks

**arXiv ID:** 2610.31029 | [PDF](https://arxiv.org/pdf/2610.31029v1)

**作者:** Diana Nunes `[一作]` (INESC TEC and University of Porto), Rui Campos `[通讯]` (INESC TEC and University of Porto)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了AERIAL框架，利用基准测试驱动的 LLM 选择，实现了自然语言驱动的 UAV 控制；

**💡 创新点**

创新点在于将 LLM 评测与任务执行解耦，构建基于实验数据的情景-模型映射，实现任务感知的自适应模型选择，并支持边缘-云混合部署；

**🔧 技术方法**

使用的技术包括 LLM 代码生成（Qwen 2.5 Coder、Gemma 2、LLaMA 3.2、Gemini 2.5 Flash）、ArduPilot SITL 仿真、MAVLink 通信、Ollama 本地推理、云 API 调用以及 Python 运行时监督；

**📊 数据集**

数据集由 720 次自动化仿真生成，涵盖 6 类任务（坐标、地点、对角、航点、障碍、算法）以及 50 个结构化参数化任务，用于基准和评估；

**📈 对比分析**

实验通过对比固定单模型配置与 AERIAL 自适应选型，使用响应延迟、token 数量、任务成功率等指标，表明自适应选择在多场景下显著提升成功率并降低资源消耗；

**⚠️ 局限性**

主要局限包括仅在仿真环境下验证、仅测试单机 UAV、任务类别有限、未进行交叉验证或在线重映射、未评估网络失效、缺乏安全控制与复杂任务表现不足等。

---

## 400. Multimodal Progress-Residual Fusion for Anticipating Navigation Retreat in Robotic Flexible Endoscopy

**arXiv ID:** 2610.31028 | [PDF](https://arxiv.org/pdf/2610.31028v1)

**作者:** Chi Kit Ng `[一作]`, Hongliang Ren `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于多模态信息的进度残差融合模型（MPRF），用于提前预测医生在机器人柔性内镜导航中决定退回的时机。

**💡 创新点**

创新点在于：①引入“进度残差”——通过先学习正常导航的期望EM位移，再与实际EM位移比较得到残差；②将该残差与命令、EM、视频等原始模态共同融合，在因果历史上下文中做短期预测；③提供了与多模态补充和残差贡献相关的细粒度性能评估。

**🔧 技术方法**

技术包括：EM跟踪、端镜视频与机器人控制指令同步记录；MobileNetV3进行视频特征提取并通过PCA降维；两阶段网络：Stage A为进度残差预测器（MLP+L1损失），Stage B为因果GRU时序预测器（GELU+Sigmoid输出）；训练使用AdamW，评估以AUPRC、AUROC、F1、MCC和Brier分数衡量。

**📊 数据集**

使用了150个训练/验证/评估周期（共30 662行）在一个人工造模的结肠假体上收集的数据，包含命令、EM位姿、视频；数据分为Normal、Narrow‑lumen Pass和Failure三类，记录了医生手动导航到退回控制器转换的过程。

**📈 对比分析**

与单模态或两模态控制相比，完整的A+E+V原始融合在10行（0.5 s）预期窗口下的AUPRC为0.564；MPRF在相同条件下提高到0.580，提升约2.8%；在更宽松的20行窗口下AUPRC进一步提升至0.857，表明模型在退回预测上表现更好。与最强的两模态控制相比，MPRF仍有约6.9% 的增益。

**⚠️ 局限性**

局限性包括：①仅在单位操作者和结肠假体上进行的回顾性实验，缺乏真实临床数据；②数据量有限，模型对不同操作者或更复杂解剖的泛化性未知；③残差贡献虽被量化，但其增益相对较小且不确定；④报警时序和阈值选择依赖后处理规则，实际可用性需进一步前瞻性验证。

---

## 401. NN-Based Transistor Modeling for Ultra-Fast SPICE Simulation and Standard Cell Library Characterization

**arXiv ID:** 2610.31024 | [PDF](https://arxiv.org/pdf/2610.31024v1)

**作者:** Rodion Novkin `[一作]` (Technical University of Munich), Hussam Amrouch `[通讯]` (Technical University of Munich)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发并验证了基于多层感知器的 3nm 纳米叶晶体管 NN 模型，并在标准单元库表征、时序与功耗分析及大型电路仿真中应用。

**💡 创新点**

证明 NN 模型能在保持与 BSIM-CMG 相近精度的同时，显著提高仿真速度；首次系统评估了 NN 模型规模对电路级精度的影响。

**🔧 技术方法**

使用 8/16 神经元 MLP、专门设计的损失函数（包含电流、充电及其导数项）、Verilog‑A 实现与 GPU 加速的 SPICE 结合。

**📊 数据集**

利用 3nm 纳米叶 PDK 的 BSIM-CMG 训练数据，按 5 mV 步进采样电压，收集 I_D、Q_D、Q_G 及其一、二阶导数。

**📈 对比分析**

与 BSIM-CMG 在 1701 次标准单元仿真、12 个工业电路（70–104k 单元）以及 16 位除法器和 1001 阶 NAND 震荡器的 SPICE 仿真对比；延迟误差 ≤0.1%，功耗误差 ≤0.76%，仿真速度提升 8–43 倍，标准单元表征时间从 55 min 降至 4.1 min。

**⚠️ 局限性**

小尺寸 NN 在低精度仿真或低电压工作点时误差明显增大，导致时序/功耗偏差；对 pMOS 的拟合相对 nMOS 更困难；尚未验证更大规模电路或不同工艺节点的泛化性。

---

## 402. Chunk Recomputation under Real Document Revisions: Measuring Update Cost and Retrieval Quality of Fixed, Recursive, and Content-Defined Chunking

**arXiv ID:** 2610.31019 | [PDF](https://arxiv.org/pdf/2610.31019v1)

**作者:** Yuning Bian `[一作]` (Anhui University of Technology), Chengyan Zhan `[通讯]` (Anhui University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了三种无状态文档分块策略在维基百科真实修订过程中的嵌入重计算成本与检索质量。

**💡 创新点**

创新点在于首次系统量化字节级重用下，分块边界漂移对嵌入重计算比例的影响，并揭示成本与检索性能之间的细微差距。

**🔧 技术方法**

采用固定大小窗口、递归段落优先、内容定义分块（Steadlith）与冻结的BGE‑M3编码器，结合严格的字节相同重用规则进行评估。

**📊 数据集**

使用120对中英文维基百科相邻修订稿（共240篇文章、480个版本标签）作为实验数据集。

**📈 对比分析**

通过对比重计算比例和2048-token检索召回率，发现固定窗口在中文和英文分别重计算68%和57%，递归和CDC低于56%，但召回差异仅1–2个百分点。

**⚠️ 局限性**

主要限制包括仅使用单一编码器和检索预算、缺少多模型、多源验证、对编辑对齐和标注一致性缺乏独立审计，以及对中文长无空格文本的分块失败未完全处理。

---

## 403. Truthful by Omission: Click Thinning for LLM-Native Advertising

**arXiv ID:** 2610.30999 | [PDF](https://arxiv.org/pdf/2610.30999v1)

**作者:** Junhua Liu `[一作]` (University of Southern California), Feng Qian `[通讯]` (University of Southern California)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在生成式AI助手的广告展示中，设计了一种既能保证在固定编辑顺序下的可支付机制，又能在期望上获得至少自由排序下一半的总点击价值；通过随机省略广告（click thinning）使得每条广告的最终点击概率与其在顺序中的位置无关，从而把福利最大化转化为线性分式匹配问题；利用VCG式支付实现期望真诚，且计算可行。

**💡 创新点**

核心创新是引入“click thinning”随机省略来消除点击竞争的外部性，使得点击服务不随广告位置变化，进而实现常数近似（1/2）与更精确的 2-(1‑δ)^K‑1 近似，且在保持编辑顺序的前提下依旧保持真诚与可计算。

**🔧 技术方法**

采用的技术包括：① 对每个匹配使用独立保留概率实现固定的服务向量；② 将期望福利写成线性分式匹配，可通过Charnes–Cooper变换转化为线性规划求解；③ 用VCG式“pivot支付”在期望上实现真诚；④ 定义“odds机制”将点击概率映射为 odds 以实现更高的近似因子。

**📊 数据集**

实验使用合成数据集：四个广告、四个上下文、广告上限 K∈{1,2,4}，点击概率上限 δ∈{0.1,0.8}，值域为{1/4,1,4,16}，共 144 组实例。

**📈 对比分析**

与基线（Raw、Stand、Bucket Allocation 等）比较：在自由排序下，本文机制获得约 0.9–0.97 的福利比例；在编辑顺序约束下，平均福利保持在 0.84–0.97 之间，且满足 2-(1‑δ)^K‑1 的理论下界；与 Bucket Allocation 的无空桶变体相比，δ=0.8 时表现翻倍。

**⚠️ 局限性**

局限性：① 需要事先已知且准确的点击概率，且假设省略不影响其他广告；② 机制只能在编辑顺序固定时使用，无法动态重排；③ 在极端参数（如 K=1）下仅能达到自由排序下的一半福利；④ 对实际大规模系统的可扩展性和学习点击概率的需求未在本文中解决。

---

## 404. Padding to k: A Minimum Count in the Schema Makes Open-Weight Models Pad Their Extraction Lists

**arXiv ID:** 2610.30997 | [PDF](https://arxiv.org/pdf/2610.30997v1)

**作者:** Abhishek Barali `[一作]` (Tribhuvan University), Aayush Paudel `[通讯]` (Tribhuvan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究当 JSON 模式中设定最小列表长度时，开源大模型在提取真实文档时会将列表填充至该长度，导致过度计数。

**💡 创新点**

创新点：系统评估“最小长度”对模型输出的实际影响，发现填充多为源文本且难以检测；提出并评估多种对策（如允许“N/A”、说明“如果不足则列出全部”等）。

**🔧 技术方法**

技术：使用 XGrammar 与 vLLM 对模型进行受限解码；对输出进行长度变化和过度计数测量；使用自定义条目匹配器进行误差分析；对比不同提示方式、强制与非强制解码。

**📊 数据集**

数据集：558 个真实文档，来自九个公开数据集（AMI 会议、MailEx 电子邮件、CORD 收据、SkillSpan 求职广告、MACCROBAT 病例、ROAST 产品评论、Few-NERD 维基百科句子、PDNC 小说章节等）。

**📈 对比分析**

比较方法：对八个 GPU 端开源模型进行预注册对比，测量在 n<k 情况下列表长度变化与过度计数比例；同时在 Hosted 服务上做对照；结果显示显示最小长度会使列表平均多 0.6 项，过度计数从 34.5% 提升至 55.4%，对策效果不一。

**⚠️ 局限性**

局限性：仅研究单字段字符串列表，未覆盖多字段记录；只评估 Greedy 解码和单一解码器；未测试封闭式 API；对数据集覆盖度有限，n<k 情况在实际管道中的出现频率未量化。

---

## 405. Tilt-Rotor Quadcopter with Learned Residual Control and Mode-Dependent State Estimation

**arXiv ID:** 2610.31051 | [PDF](https://arxiv.org/pdf/2610.31051v1)

**作者:** Ninad Joshi `[一作]` (Indian Institute of Technology Hyderabad), Prabhat Kumar `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种带有学习残差控制和状态估计的倾转四旋翼无人机飞行堆栈，并在仿真中验证其性能。

**💡 创新点**

创新点在于使用基于倾转的非线性六自由度模型、阶跃式模式感知EKF与带有有限输出的神经残差补偿，实现对风偏差的自适应修正。

**🔧 技术方法**

技术包括六自由度动力学建模、分层PD控制、含转舵的控制分配、模式判别的Schmitt触发器、14维EKF传感器融合以及15维输入的MLP残差网络。

**📊 数据集**

数据集为仿真生成的40条PD-only轨迹、25621条随机风况样本，用来进行行为克隆训练。

**📈 对比分析**

比较方法是对比仅PD控制与PD+残差网络在不同风速（1.5 m/s 与 3.9 m/s）下的站姿误差，残差控制将误差从0.117 m/0.319 m下降到0.060 m/0.162 m，表明约49%的提升。

**⚠️ 局限性**

局限包括：仅在仿真下验证；残差网络未通过DAgger收敛；仅测试恒定风且未考虑阵风；仅校正倾角，未对推力/扭矩做进一步评估。

---

## 406. SCOPE: Compiling Natural-Language Policies into Neurosymbolic Predicate Programs

**arXiv ID:** 2610.31062 | [PDF](https://arxiv.org/pdf/2610.31062v1)

**作者:** Arnon Mazza `[一作]` (Plurai), Elad Levi `[通讯]` (Plurai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将自然语言政策编译成可执行的神经符号程序，实现多轮、复合政策的高效且可解释的执行。

**💡 创新点**

创新点在于：①将政策转化为可执行代码，使用弱模型谓词和确定性提取器；②通过对抗样本生成与程序联合优化，让程序始终对强教师的解读保持一致；③实现了跨执行器的迁移性和对弱模型的显著提升。

**🔧 技术方法**

采用神经符号程序（弱模型谓词 + Python 组合逻辑）与对抗样本生成技术，辅以 LLM‑agent 进行优化、生成与教师标注。

**📊 数据集**

使用四个真实世界的政策任务数据集：Coaching、Escalation、Plan 和 Tile‑match。

**📈 对比分析**

与提示优化、工作流生成、任务分解等基线进行比较，SCOPE 在所有任务和执行器上取得最高或接近最高准确率，尤其在弱模型上提升最大，并展示了优秀的跨执行器迁移性能。

**⚠️ 局限性**

局限性包括：程序的正确性依赖强教师对政策的解读，缺乏形式化验证；对更动态或极其复杂政策的适配性尚未充分验证。

---

## 407. Conflict-Aware Logit Adapters for Utility-Preserving Anti-Distillation

**arXiv ID:** 2610.31046 | [PDF](https://arxiv.org/pdf/2610.31046v1)

**作者:** Quoc Phong Dao `[一作]` (Hanoi University of Science and Technology), Trung Le `[通讯]` (Monash University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种训练时的轻量级对抗性logit适配器（CALA），用于防止大型语言模型的推理轨迹被用作黑盒蒸馏的训练数据；

**💡 创新点**

创新点在于将抗蒸馏干预从解码时迁移到训练时，通过在冻结的教师模型上训练一个残差logit适配器，并使用代理学生的虚拟上升步骤以及冲突感知的优化策略，使得教师输出既保持流畅性又降低可用于下游模型蒸馏的价值；

**🔧 技术方法**

采用残差logit适配器、代理学生虚拟梯度上升、top‑k 支持约束、基准与中间层一致性正则化、冲突感知权重调整、以及可调强度插值等技术；

**📊 数据集**

在MetaMathQA上训练适配器，并在GSM8K、MATH、MMLU三个推理基准上进行评估；

**📈 对比分析**

与传统的解码时抗蒸馏方法ADS以及单纯的温度采样对比，CALA在相同教师准确率下能够更显著地降低蒸馏后的学生准确率，同时保持更好的表面流畅度；在推理效率上，CALA的推理成本仅略高于原教师模型，而ADS的成本和内存消耗几乎翻倍；

**⚠️ 局限性**

局限性包括：仅针对黑盒轨迹蒸馏场景，可能对不同tokenizer或更强攻击（如改写或验证攻击）无效；防御效果受代理学生与教师tokenizer兼容性的依赖；在极端防御强度下仍可能略微降低教师性能或流畅度。

---

## 408. Algorithmic drift in generative adaptive learning

**arXiv ID:** 2610.31041 | [PDF](https://arxiv.org/pdf/2610.31041v1)

**作者:** Firuz Kamalov `[一作]` (Canadian University Dubai), Nitesh Sughnani `[通讯]` (Knowledge and Human Development Authority)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了生成式人工智能在自适应学习系统中因观测噪声导致的算法漂移问题，构建了学习者-辅导员的耦合动力学模型，并通过模拟验证该漂移现象及其对不同群体学习成效的影响。

**💡 创新点**

创新点包括：
1) 首次将算法漂移形式化为一个闭环动力学系统，并证明在非对称错误通道下会出现分岔，导致受保护群体的学习轨迹被捕获并下降；
2) 提出并验证了“时间公平稳定器”（TFS）——一种基于循环耐心计数的目标干预机制，能够周期性重新进入近端发展区并抑制漂移；
3) 在开放权重语言模型上进行匹配对评分实验，量化了非标准语言回答导致的错误率，为漂移模型提供实证基础。

**🔧 技术方法**

使用的技术主要包括：
- 生成式自适应内容策略（基于估计的近端发展区）；
- 统计知识追踪器（在线最大似然估计）和深度知识追踪（LSTM-DKT）；
- 逆向强化学习中的探索策略（ε-greedy、UCB、Thompson Sampling）作为对照；
- 误差通道建模（非对称假阳性/假阴性率）和分岔分析；
- Monte Carlo 代理‑基准仿真。

**📊 数据集**

实验数据集：
- 人工合成的学习者群体（10,000 个代理，100 步学习过程，50 次独立复制）；
- 公开权重语言模型（Qwen2.5‑7B‑Instruct、Phi‑4‑mini‑instruct、Mistral‑7B‑Instruct‑v0.3）在 40 个正确答案与 20 个错误答案组成的匹配对问答集上进行评分，评估非标准语言导致的误差率。

**📈 对比分析**

比较方法：
- 将 TFS 与三种标准 RL 探索策略进行对比，使用五个组公平度量（曝光差距、ZPD 访问差距、估计偏差、错误分类差距、收益比例）以及漂移度量（平均绝对误差、能力差距）。
- 结果表明，TFS 在所有噪声水平下能将受保护群体的能力差距从 0.038 降至 0.003，漂移度量从 0.268 降至 0.203，几乎恢复至公平基线；相反，ε‑greedy、UCB、Thompson 采样仅略有改善甚至恶化了某些指标。

**⚠️ 局限性**

局限性：
- 模拟使用的是简化的学习规则、离散知识组件和固定的噪声通道，真实教育环境中的多模态评估、教师干预和学生情绪等因素未被建模；
- 语言模型误差测量仅基于 40 对问答，缺乏跨学科、跨语言、跨长度的覆盖；
- TFS 的探测和纠正假设在真实学生中可能导致挫折感或学习率衰减，需进一步实验验证其可接受性；
- 结果对噪声阈值敏感，若语言模型误差高于阈值，TFS 的效果会衰减。

---

## 409. ResGVC: Residual Coding for High-fidelity Generative Video Compression

**arXiv ID:** 2610.31039 | [PDF](https://arxiv.org/pdf/2610.31039v1)

**作者:** Ding Ding `[一作]` (Alibaba Group), Kai Li `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个残差编码框架ResGVC，通过在像素空间传输残差信号引导扩散去噪，实现高保真视频压缩。

**💡 创新点**

首次在GVC中引入残差编码与扩散变换器作为熵模型，并在像素域直接去噪，提供显式源约束并采用跳过模式平衡码率与质量。

**🔧 技术方法**

采用扩散模型（flow matching）、像素空间视频扩散变换器（VideoPiT）、残差编码/解码器、算术编码、跳过模式（残差生成）以及基于OpenVid-1M的训练。

**📊 数据集**

训练使用OpenVid-1M数据集，评估在HEVC B、UVG、MCL-JCV三大视频数据集。

**📈 对比分析**

与VTM-17.0、DCVC-FM、PLVC、GLC-Video、DiffVC、GNVC-VD、YODA等基线比较，BD-rate在LPIPS上下降63.7%，PSNR下降83.8%，在感知指标（LPIPS、FID）和信号质量（PSNR、MS-SSIM）上均优于前沿方法。

**⚠️ 局限性**

需要5步扩散推理，计算开销较大；跳过模式参数需经验选择；对极低码率下的细节恢复仍有限；模型规模虽小但仍高于部分单步方法。

---

## 410. Extension of MDS Codes over Finite Fields

**arXiv ID:** 2610.31023 | [PDF](https://arxiv.org/pdf/2610.31023v1)

**作者:** Puyin Wang `[一作]` (Central China Normal University), Jinquan Luo `[通讯]` (Central China Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于超平面法的迭代扩展算法，可在不改变MDS性质的前提下将非Reed–Solomon MDS码的长度逐步增加。

**💡 创新点**

创新点在于将MDS码的可扩展性转化为寻找不在任何由k-1列生成的超平面的向量，并利用内积快速判定，从而大幅降低搜索复杂度并得到更大长度的非RS MDS码。

**🔧 技术方法**

主要技术包括线性代数中的矩阵正则性判定、超平面与正交向量的构造、以及meet-in-the-middle（中间相遇）搜索策略。

**📊 数据集**

实验数据集来自多种已知非RS MDS码族，包括椭圆曲线、扭曲RS、列扭曲RS、缺失多项式、循环码等，在不同有限域（如|q|≤32）上进行扩展。

**📈 对比分析**

通过与传统穷举搜索（O(nk-1 q^{k-1}k^3)）对比，提出的算法复杂度约为O(k q^k)，在实验中实现了更长码长和更高Schur平方维数（如(9,4,6)在_9上的维数9>2k）。

**⚠️ 局限性**

限制在于每一步扩展的选择可能影响后续能否继续扩展，缺乏全局最优性保证；若在当前路径无法再扩展，可能未找到可能存在的更长码。

---

## 411. Mitigating Saliency Collapse: Robust Saliency-Aware Long-Text Image-Text Alignment

**arXiv ID:** 2610.31005 | [PDF](https://arxiv.org/pdf/2610.31005v1)

**作者:** Qiuyu Kong `[一作]` (Sapienza University of Rome), Yiming Wang `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种鲁棒的长文本图像文本对齐框架，旨在解决模型过度依赖视觉显著性文本导致的“显著性崩塌”问题。

**💡 创新点**

创新点包括：①将图像文本对拆分为显著性视图与上下文视图；②采用基于对象级视觉显著性与LLM文本拆分的跨模态无关分解；③引入显著性对比、上下文对比与视觉锚定三种辅助损失，实现显著性与上下文的协同学习。

**🔧 技术方法**

使用的技术包括：CLIP ViT‑B/16 视觉编码器、KPS 扩展文本编码器、对象显著性估计（如 FASA）、LLM（Qwen2.5）文本拆分、InfoNCE 对比学习及视觉锚定正则化。

**📊 数据集**

实验数据集覆盖多种长文本检索场景：DOCCI、DCI（通用密集描述）、Sketchy（时尚服饰）、INSECT（专业生物描述）。

**📈 对比分析**

与 Long‑CLIP、FineLIP、GOAL、StructXLIP 等前沿方法对比，ROSA 在四个基准上取得最优或竞争性 Recall@K，且在显著性文本被削弱时显著提升鲁棒性；并且可作为插件式增强，提升现有 CLIP 微调模型。

**⚠️ 局限性**

局限性包括：依赖外部显著性检测器与 LLM，泛化能力待验证；离线文本拆分增加预处理开销；实验主要聚焦 CLIP 体系，未探索传统非 CLIP 的细粒度对齐方法。

---

## 412. CARL: Chat Assisted Reinforcement Learning

**arXiv ID:** 2610.31014 | [PDF](https://arxiv.org/pdf/2610.31014v1)

**作者:** Sanjiv Das `[一作]` (Santa Clara University), Jacob Sisk `[通讯]` (Franklin Templeton)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种名为CARL的聊天辅助强化学习系统，提供交互式人机协作界面，帮助领域专家将不完整的优化目标转化为完整的强化学习（MDP）规范，并自动生成、训练和验证RL代码。

**💡 创新点**

创新之处在于将问题规范化与性能提升分为十步结构化工作流，利用专属“Trillian”语音助手预填答案，维持对话简洁并在每一步记录决策摘要；同时通过迭代验证和参数调整闭环减少了“规范缺口”和“性能缺口”。

**🔧 技术方法**

技术实现结合大语言模型（Claude Opus/Haiku/Sonnet）、工具调用、Flask/终端交互、Skill文件、决策内存、自动代码生成与训练管线，以及对齐、奖励设计、超参自动化等技术。

**📊 数据集**

实验使用了两个案例：一是基于真实金融数据的目标财富管理规划，二是带动态压强约束的Goddard火箭控制；两者均采用模拟或公开的动力学/财务参数而非标准公开数据集。

**📈 对比分析**

通过与传统全自动代码生成方法对比，CARL能够在两例中避免奖励黑客、状态别名和约束违例，并实现与解析最优解相近的性能（例如财富管理的收益率逼近解析最优，火箭高度接近理论最大值）。

**⚠️ 局限性**

局限性包括：目前仅在两类相对简洁的案例验证，缺乏大规模工业应用测试；对大型语言模型的依赖导致成本与可用性受限；仍需要人工专家参与，且系统对模型生成错误的容忍度有限。

---

## 413. Agri-SCHOLAR: Simulation-Grounded In-Season Rolling Advisories for Agricultural Crops Using LLM-Bounded Optimization

**arXiv ID:** 2610.31003 | [PDF](https://arxiv.org/pdf/2610.31003v1)

**作者:** Geetha Charan `[一作]` (Indian Institute of Science), Y. Narahari `[通讯]` (Indian Institute of Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Agri‑SCHOLAR，一个基于LLM与APSIM的滚动、季节内精准农田管理决策系统。

**💡 创新点**

结合LLM约束搜索与模拟器评估的闭环体系，按周更新，解决传统季节全程规划缺乏实时天气与个体化的问题。

**🔧 技术方法**

多Agent LLM（Gemma‑4）+ APSIM农作物模拟器 + Optuna优化 + 观测+短期预测+类似年外推天气构造 + 语义检索。

**📊 数据集**

ERA5日天气、1980‑2025年历史气候、两地区南亚玉米土壤资料和区域管理实践手册。

**📈 对比分析**

对比区域标准实践（PoP），在11个季节中均优于PoP，平均产量提升15%/22%，净利润提升17%/23%，并显著降低深层径流和硝酸盐淋溶。

**⚠️ 局限性**

依赖较短期天气预测，缺乏长期气候预报；在无LLM绑定的情况下易被模拟器“利用”，对雨养区提升有限；缺乏现场试点验证。

---

## 414. HARMONY: Hierarchical Anchor Retrieval on Manifold for Oblivious-source Machinery Acoustic Anomaly Detection

**arXiv ID:** 2610.31093 | [PDF](https://arxiv.org/pdf/2610.31093v1)

**作者:** Yuhang Zhang `[一作]` (Peking University), Mingyuan Li `[通讯]` (ELLIS Institute of Finland)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了HARMONY框架，在缺乏机器标识的工业声学异常检测中，通过锚点引导的几何重排实现对混合源特征的分离与检索；

**💡 创新点**

创新点在于把表示空间的“卷积”问题转化为几何重划分，利用锚点生成Voronoi区间并在同一空间中实现层级检索，从而同时提升检测准确率和检索速度；

**🔧 技术方法**

技术上结合预训练的BYOL‑A特征编码、主成分投影、锚点初始化、软赋值对齐损失以及基于Voronoi区的层级倒排索引；

**📊 数据集**

主要在DCASE 2020工业设备异常检测数据集上验证，同时在MIMII和DCASE 2022数据集做跨域测试；

**📈 对比分析**

与PatchCore、Audio‑PC、CADA、UniFlow等统一或基础特征基线对比，HARMONY在DCASE 2020上平均AUC提升至90.00%（比最强基线提升14.6个百分点），并在CPU上实现7.8×的检索加速，AUC仅下降0.20个百分点；

**⚠️ 局限性**

局限性包括对预训练特征的依赖、全局温度参数可能不适用于极度不平衡模式，以及需要保证残差和候选覆盖才能保证理论边界的成立。

---

## 415. Dual Exam: Controlled Evaluation of Multimodal Agents and World Models

**arXiv ID:** 2610.31066 | [PDF](https://arxiv.org/pdf/2610.31066v1)

**作者:** Zhefan Rao `[一作]`, Haoxuan Che `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Dual Exam 框架，对多模态代理（MA）和世界模型（WM）分别在固定环境响应和固定动作干预下进行评估，并在四个领域（物理、数字、社会、科学）内构建大规模基准。

**💡 创新点**

创新点在于：①将 MA 与 WM 的评估拆分为两条独立的轨道，避免因耦合结果导致的因果混淆；②提供统一的任务规格与验证规则，兼容多模态输入；③给出首批 1,088 条 MA 任务与 400 条 WM 场景的公开评测，首次展示跨领域、跨模型配置的对比与聚合方法。

**🔧 技术方法**

技术手段包括：使用多种大型语言模型（GPT‑6 Astra、GPT‑5.6 Sol、Gemini 3.6 Flash、Qwen‑3.8 Max、Kimi‑K3、DeepSeek V4.1 Flash、GLM、Seed‑2.1 Pro、MiniMax H3 等；采用固定的环境响应机制（Agent 轨道）和固定的动作序列（WM 轨道）；对任务执行和状态变化进行自动化与人工复核的混合评分。

**📊 数据集**

数据集：MA 评测基准由 1,088 条任务组成，来源于 Agents' Last Exam、SWE‑bench、ToolChoice、EdgeBench 等；WM 评测基准包含 400 条案例，分布在 8 个场景（Embodiment、Driving、Daily Life、Web、Game、Human Interaction、Natural Science、Expert Science）；此外还提供 150 条耦合 MA–WM 的 pilot 案例。

**📈 对比分析**

比较方法：对 MA 采用案例计数加权的 Index（1,088 项总分 0–100 归一化）；对 WM 采用 8 个场景平均得分（每场 50 条案例）得到 Overall；两者结果分别列出排名。性能方面：GPT‑6 Astra 以 31.3 的 Index 领跑 MA；GPT‑Image‑2 以 21.7 的 Overall 领跑 WM；其它模型在不同子域或场景上表现各异。

**⚠️ 局限性**

局限性：①单独评估轨道无法直接说明 MA–WM 互补或联合训练的提升；②不同轨道使用的任务与指标不一致，无法直接比较整体实力；③参考环境响应机制或 WM 输出的正确性未进行独立审核；④评测未做统计显著性检验；⑤数据集覆盖不均衡（某些场景缺失或有限）；⑥pilot 轨迹审核为事后制定，缺乏独立判定；⑦缺少对多轮学习、持续改进的评估。

---

## 416. Directing to Diversify: Multi-Directional Search for Automatic Heuristic Design

**arXiv ID:** 2610.31048 | [PDF](https://arxiv.org/pdf/2610.31048v1)

**作者:** Jianghan Zhu `[一作]` (Singapore Management University), Zhiguang Cao `[通讯]` (Singapore Management University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种多方向的 Perturbation–Rewrite Search (PRS) 框架，用于自动化启发式设计；

**💡 创新点**

将方向化扰动（扩展、压缩、导向）作为生成级搜索原语，并结合BFS和自适应操作符控制实现多方向探索；

**🔧 技术方法**

采用 LLM 生成与重写、基于 BFS 的搜索、UCB 自适应操作符选择、停滞感知重启等技术；

**📊 数据集**

在 11 种组合优化问题（如 CVRP、OP、MKP、KP、OBPP、TSP、ASP、FSSP 等）上的训练、验证和测试数据集；

**📈 对比分析**

与手工启发式、传统求解器、NCO 方法及多种 LLM‑based AHD 基线（FunSearch、EoH、ReEvo、HSEvo、MCTS‑AHD）进行对比，PRS 在大多数任务中实现更低目标值/更小缺口，尤其在 OOD 场景表现优异；

**⚠️ 局限性**

对 LLM 查询成本敏感、依赖提示与重写质量，且在部分任务上提升空间有限，需进一步提升效率与鲁棒性。

---

## 417. CANNBot-CAKE: Collaborative Agentic Kernel Evolution for Ascend NPU

**arXiv ID:** 2610.31034 | [PDF](https://arxiv.org/pdf/2610.31034v1)

**作者:** Deng Pan `[一作]` (Huawei), Yuwei Fan `[通讯]` (Huawei)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CAKE，一套端到端的 Ascend NPU 自定义算子生成与演化框架，支持多 DSL 路径、适配路由、演化搜索、知识库和自进化 Agent。

**💡 创新点**

核心创新点包括：① 可视化、可检验的生成流水线；② 基于过去评估的可学习多 DSL 路由器；③ 集成自进化 Agent 与 CAKE Book，实现经验共享与持续改进；④ 将生成、评估与演化耦合，形成可持续的高效算子研发闭环。

**🔧 技术方法**

使用技术包括：大型语言模型（LLM）驱动代码生成；项目化的技能（Skill）工具库；项目信息记录与自动回溯；信息门控与贝叶斯项目反应模型进行路由决策；基于 Roofline 的瓶颈分析与演化搜索；以及 CI 平台与 Wiki 形式的知识管理。

**📊 数据集**

数据集：CANN‑Bench v1.1（53 个算子，标准 20 条案例/算子，隐藏 80 条/算子），在 Ascend 950PR 上进行评估；同时在 EngramGateFusion 案例中使用 20 种输入形状进行演化搜索。

**📈 对比分析**

与基线（厂商默认 AscendC）比较，CAKE 在 53 个算子上实现 100% 标准案例通过率、98.28% 隐藏案例通过率，平均分 78.63/100；相对于基线算子，平均性能提升 1.334×，并在演化过程中进一步提升 EngramGateFusion 46.6% 的设备时间。

**⚠️ 局限性**

局限性：需要人工复核与审阅更新；演化搜索受轮次限制（最多 20 轮），对极难算子仍可能难以收敛；LLM 生成的代码质量高度依赖训练数据，AscendC 相关语料仍不足；最终性能受硬件变更和 CANN 版本差异影响，需要持续维护知识库。

---

## 418. Inside-JEPA: From Prediction to Relational Grounding in Latent World Models

**arXiv ID:** 2610.31194 | [PDF](https://arxiv.org/pdf/2610.31194v1)

**作者:** Hossein Resani `[一作]` (Adelaide University), Javen Qinfeng Shi `[通讯]` (Adelaide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并训练了一种在JEPA世界模型中嵌入关系通道的模型，并通过逆向动力学损失在没有关系标签的情况下为这些通道赋予语义。

**💡 创新点**

在世界模型状态中加入可预测的关系通道，并利用逆向动力学目标（从关系通道的变化回归动作）实现自监督的关系赋义，从而无需显式关系标签即可让通道表征实体之间的关系。

**🔧 技术方法**

采用Joint-embedding Predictive Architecture (JEPA) 结合冻结的ResNet‑18编码器、Fourier坐标嵌入、预测器MLP、逆向动力学头，以及线性探测器和规划读取器等技术。

**📊 数据集**

使用Something‑Something V2（带有Something‑Else边框标注）、RH20T机器人实验以及检测‑+‑跟踪生成的对象，另外还有用于对照的2‑D Blocks模拟环境。

**📈 对比分析**

对比五种模型变体（仅预测、逆向读整态、只读实体、只读关系、逆向标签打乱），使用匹配关系AUROC、关系变化预测AUROC以及一阶规划Top‑5准确率等指标；逆向读关系通道模型在所有评测中表现最佳，匹配AUROC提升约0.06–0.10，规划Top‑5准确率提升至≈0.33（相较于随机0.077和频率先验0.174）。

**⚠️ 局限性**

需要大量准确动作标签才能实现关系赋义，关系赋义仅在动作变化能表征的关系（接触、接近、垂直等）上有效，对方向关系无效；模型依赖冻结编码器、单一数据集、短时预测以及检测框，性能受限且未验证在更复杂场景或多步规划中的表现。

---

## 419. When More Metadata Misleads: Probing Field-Level Grounding in LLM-Generated Museum Descriptions

**arXiv ID:** 2610.31193 | [PDF](https://arxiv.org/pdf/2610.31193v1)

**作者:** Clemencia Siro `[一作]` (Centrum Wiskunde & Informatica), Laura Hollink `[通讯]` (Centrum Wiskunde & Informatica)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在两阶段实验中，作者对 10,250 件来自十个地理区域的博物馆藏品进行结构化元数据与图片的多模态生成，研究元数据数量、字段类型、图像可用性及地区差异对描述文本的真实性与覆盖率的影响。

**💡 创新点**

创新点在于提出了控制字段类型的探测框架，能够在保持对象身份不变的前提下独立操纵字段数量、类型与图像输入，并引入字段级与主张级的真实度评估，揭示不同字段对模型输出的错误模式。

**🔧 技术方法**

采用大语言模型（如 LLaMA‑Vision、GPT‑系列）结合多模态输入，利用 DeBERTa‑large‑MNLI 进行字段覆盖判断，以及基于 NLI 的自动化主张提取与验证管线进行细粒度真伪判定。

**📊 数据集**

使用 TimeTravel 基准集，包含 10,250 件物件的结构化记录、图像与参考描述，覆盖伊拉克、埃及、希腊、罗马帝国等十个文化地理区域。

**📈 对比分析**

通过对比表面相似度（BLEU、ROUGE、METEOR、BERTScore）、字段覆盖率、以及主张级误差率，在十个控制条件下评估性能；结果显示元数据增多并不总能提升真实性，语境字段降低幻觉率但增加缺失，内容字段提高覆盖率但幻觉率显著升高；图像在稀疏元数据时能提升真实性，但在完整元数据时往往放大超出记录的推断。

**⚠️ 局限性**

局限性包括：字段类别划分可能不够绝对，自动化主张验证无法捕捉细微策展差异；参考描述不是完全详尽，评估将图像支持的真实陈述视为未支持；缺少专家策展人级别的标注，可能忽略视觉或外部来源的真实性考量。

---

## 420. RaSFNet: Ray-aligned and Scan-aware Fusion Network for LiDAR Semantic Segmentation

**arXiv ID:** 2610.31186 | [PDF](https://arxiv.org/pdf/2610.31186v1)

**作者:** Chuang Chen `[一作]` `[通讯]` (Sichuan University), Chuang Chen (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于 LiDAR 投影的语义分割网络 RaSFNet，用以提高稀疏点云在二维图像上的语义识别效果。

**💡 创新点**

创新点包括：① 通过 Ray‑Aligned Geometric Distribution Encoding (RAGDE) 以光束对齐坐标系编码每个视锥内点的径向与切向分布；② 设计 Scan‑Topology‑Aware Mamba (STAM) 模块，将视锥特征按激光束角度顺序序列化并通过 Mamba 处理全局依赖；③ 采用 Range‑Constrained Point‑Frustum Fusion (RCPFF) 用共享的距离信息对点与视锥特征进行约束式双向融合，保持空间对应关系。

**🔧 技术方法**

技术手段包括：球面投影、光束对齐坐标编码、多频 Fourier 特征映射、轴感知局部聚合、蛇形序列化、Mamba 轻量级注意力、基于距离的匹配得分融合以及多尺度特征提取和融合。

**📊 数据集**

使用了公开的 SemanticKITTI（64 beam）和 nuScenes（32 beam）两个大规模户外 LiDAR 语义分割数据集进行实验。

**📈 对比分析**

与现有 20+ 方法比较，RaSFNet 在 SemanticKITTI 验证集上 mIoU 70.2% 以及 nuScenes 验证集上 mIoU 77.7%，均位列前列，尤其在细粒度几何区分和长距离稀疏场景表现突出。

**⚠️ 局限性**

局限性包括：① 需要较高的投影分辨率以降低投影冲突，导致计算量和显存需求上升；② 对激光雷达的 360° 水平扫描假设敏感，无法直接迁移至不完整扫描或多传感器混合环境；③ 仍受投影过程中点云稀疏性和噪声影响，长距离稀疏点对 RAGDE 及 STAM 的编码效果有限。

---

## 421. Prompting Strategy Effects Depend on the Judge: A Multilingual Study of LLM-as-a-Judge Robustness

**arXiv ID:** 2610.31172 | [PDF](https://arxiv.org/pdf/2610.31172v1)

**作者:** Asal Mehrabi `[一作]` (University of Amsterdam), Clemencia Siro `[通讯]` (Centrum Wiskunde & Informatica)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对LLM作为评判者在多语言环境中的提示策略进行大规模基准测试，系统评估9种提示策略、4款开源评判模型、27种语言和4项任务，生成超过100万条分点评分并对比人类评估；

**💡 创新点**

首次揭示提示策略效果高度依赖评判模型和语言资源水平，且评判模型占主导方差，提示策略与评判模型的交互效应超过单独提示策略的影响；

**🔧 技术方法**

使用Open-weight LLMs（Aya Expanse 32B、Gemma 3 27B、Qwen 2.5 32B、Qwen 2.5 72B），Chain-of-Thought、Few-shot、Role Prompting、Cross‑Lingual、Monolingual、Trans‑Eval等提示；

**📊 数据集**

利用13个公开数据集（FLORES‑200、MAFAND‑MT、SSA‑MTE、XL‑Sum、MLSUM、CaSum、AfriQA、XQuAD、PersianQA、CatalanQA、Belebele、AfriMGSM、MGSM）覆盖多种任务与27种语言；

**📈 对比分析**

通过两路ANOVA和Spearman/ Pearson相关分析对比提示策略与评判模型的影响，发现评判模型解释约8.7%方差，提示策略仅12%，交互效应5.6%；在低资源语言下策略间排名一致性极差（ρ≈0.04），人类评估相关性最高的策略为Role Prompting（ρ≈0.48），但不同策略在语言层面表现差异大；

**⚠️ 局限性**

局限性包括：使用金标参考导致分数范围受限、低资源语言的提示验证不足、评判模型仅为开源模型、对系统生成内容的评估不足、跨语言一致性测评受限于少数任务等。

---

## 422. HiSTEVE 1.0: Harnessing Human-Interface VLA Agents for Open-World Games

**arXiv ID:** 2610.31162 | [PDF](https://arxiv.org/pdf/2610.31162v1)

**作者:** Zecong Tang `[一作]` (University of Hong Kong), Bo Dai `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

HiSTEVE是一个层次化VLA代理，结合能力感知的主控框架与原生接口控制，实现了长时程自治与即时视觉‑语言动作映射。

**💡 创新点**

创新点在于三方面：① 能力约束的主控框架通过结构化记忆和证据驱动的交接实现长时程任务管理；② 对GUI与非GUI交互采用RS‑Norm与Dual‑Mouse双分支的范式；③ 采用三阶段训练（知识注入、模仿学习、强化学习）提升低层VLA的可靠性。

**🔧 技术方法**

主要技术包括VLM规划器、VLA策略（Qwen2.5‑VL‑7B）、RS‑Norm正则化、Dual‑Mouse动作编码、任务完成状态检测器、基于GPT‑4o的高层规划以及三阶段强化学习。

**📊 数据集**

数据集使用MineStudio Bench、OpenHA、LongBench以及来自VPT的人类游戏轨迹，用于训练与评估。

**📈 对比分析**

与多种基线（VPT、STEVE‑1、ROCKET‑1、JARVIS‑VLA、OpenHA Text/CoA等）比较，HiSTEVE在MineStudio Bench上从65%提升至85%，在OpenHA上从36.8%提升至37.4%，在LongBench的最终目标成功率从3.6%提升至25%，显示显著性能提升。

**⚠️ 局限性**

限制主要体现在GUI多步骤操作仍易失败、恢复与开放式任务表现不足、以及对长时程任务的复杂性与稀疏奖励依赖较高，且对环境变化的泛化仍有限。

---

## 423. PriMa-Causa: Causal Prescriptive Maintenance

**arXiv ID:** 2610.31147 | [PDF](https://arxiv.org/pdf/2610.31147v1)

**作者:** Felix Saretzky `[一作]` `[通讯]`, Felix Saretzky

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了PriMa-Causa框架，将因果推理嵌入到预防性维护决策中，实现对维修行动的因果效应评估与排序。

**💡 创新点**

创新点在于首次将因果先验与图条件化的表格因果基础模型结合，消除每个案例的估计器拟合需求，并实现从关联预测向干预效应推荐的完整链条。

**🔧 技术方法**

采用结构因果模型、do-算子、因果发现、先验数据拟合网络（TabPFN）以及变换器基础的因果推理模型，并配合参数化生产线模拟器进行训练。

**📊 数据集**

数据来源为半合成生产线模拟数据与真实高速 FMCG 生产线观测记录，利用生成的因果过程图产生的观测与干预样本进行训练与评估。

**📈 对比分析**

与关联预测基线比较时，PriMa-Causa 在处方误差、PEHE 等指标上显著下降，尤其在弱重叠与稀疏上下文场景下优势最大，并在 MTTR 与 OEE 指标上实现了可观的提升。

**⚠️ 局限性**

局限性包括对结构图准确性的依赖、对专家知识的半自动化依赖、对非平稳与高维异构数据的适应性有限，以及仅在半合成与实验室环境下验证，真实生产系统的全局适用性尚待进一步检验。

---

## 424. ProductWebBench: Evaluating Website Continuity Beyond Task Completion

**arXiv ID:** 2610.31118 | [PDF](https://arxiv.org/pdf/2610.31118v1)

**作者:** Jiajun Wu `[一作]` (Beihang University), Bryan Dai `[通讯]` (IQuest Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一个面向前端开发的评测基准，评估代码修改者在保留现有网站功能的前提下完成新需求的能力。

**💡 创新点**

创新点在于：①引入“网站连续性”概念，要求修改既完成目标又不破坏已有行为；②设计冻结的任务包，包含浏览器状态证据和多阶段要求；③实现累积检查与精确回溯回归检测，使评测能够跟踪整个编辑轨迹。

**🔧 技术方法**

技术主要包括：基于浏览器的自动化评测、模拟用户逐阶段展示需求、静态与动态门检测、任务包冻结与验证、以及对多模型多回合交互的统一评测框架。

**📊 数据集**

使用了400个真实前端仓库（React、Vue、Svelte、Astro、Gatsby、Hugo、PHP、静态站点），共计320个Change任务和80个Build任务，包含3200+浏览器状态证据与门检查。

**📈 对比分析**

与13个最前沿及开源模型对比，评估了Task%（完整任务通过率）、Gate%（门通过率）、WCS（网站连续性得分）等指标；结果显示虽然大多数模型能完成功能，但约28%完成任务后仍有连续性失效，甚至10%在后续步骤出现回归。

**⚠️ 局限性**

局限性包括：评测只针对前端项目，无法覆盖后端逻辑；门检测仍需人工设计，可能遗漏细节；评测对模型的历史回合有限制，无法完全模拟真实开发流程；且评测结果受任务冻结和门设计的影响。

---

## 425. Can AI Co-Design Structures? An AI Agent for Early-Stage Structural Design

**arXiv ID:** 2610.31113 | [PDF](https://arxiv.org/pdf/2610.31113v1)

**作者:** Ricardo Maia Avelino `[一作]`, Philippe Block `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种将自然语言提示转化为可编辑早期结构模型的工作流，利用AI代理、MCP协议和COMPAS图结构API实现从文本到结构模型的交互生成。

**💡 创新点**

创新点在于将大语言模型代理与MCP协议相结合，构建可直接操作的图结构化结构数据模型，实现语义丰富的结构元素（节点、框架、面积等）的自动生成与编辑，并支持后续可视化、序列化与AEC工作流联动。

**🔧 技术方法**

采用了Google Gemini 3.1 Pro LLM、Model Context Protocol（MCP）服务器、COMPAS-based结构建模API、Python语言与图形化数据结构。

**📊 数据集**

未使用公开数据集，示例基于人工构造的自然语言提示生成结构模型；论文中未提供训练或评测数据。

**📈 对比分析**

缺乏系统性对比与性能评估，仅通过一系列示例演示了模型生成与编辑的可行性，没有给出量化指标或与现有工具的对比。

**⚠️ 局限性**

局限性包括：模型不保证结构正确性、代码合规性或分析就绪；LLM空间推理受限导致几何误差；示例规模有限；缺乏正式评估、基准和数据集；需要人工检查与验证。

---

## 426. DRAFT: Decoding and Reconstructing 3D Room Layouts from fMRI and Camera Trajectories

**arXiv ID:** 2610.31108 | [PDF](https://arxiv.org/pdf/2610.31108v1)

**作者:** Chi-Wei Lee `[一作]` (National Tsing Hua University), Po-Chih Kuo `[通讯]` (National Tsing Hua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种利用fMRI与已知相机轨迹恢复隐藏场景的3D房间布局的方法，并将其作为先验提升多种解码器的性能。

**💡 创新点**

首次在视角随动的室内环境中解码简洁的布局代码，并与相机轨迹融合实现无视觉帧的真实度量几何重建。

**🔧 技术方法**

结合早期视觉皮层的岭回归读取布局代码、基于U‑Net的细节生成器、相机轨迹自由空间特征、低频融合与范围校正等多模块体系。

**📊 数据集**

使用20名受试者在10个家具化室内场景中观看80秒的第一人称漫游视频，采集fMRI并对每帧做光线投射得到地面真实深度。

**📈 对比分析**

在leave‑one‑scene‑out交叉验证中，与七个公开fMRI解码器及无fMRI参考对比，布局相关性（layout r）达到0.480，优于单独相机轨迹（0.446）和改进解码器最高0.429，且作为后置插件将所有解码器提升至约0.49。

**⚠️ 局限性**

方法需要已知相机轨迹、聚合多次记录，仅在已见场景上训练，脑部增益仅在极少场景验证中显著，未对单试验或非标注路径进行测试。

---

## 427. Learning What to Ask for Tomorrow: Active Preference Acquisition for Long-Term Personalized Web Agents

**arXiv ID:** 2610.31107 | [PDF](https://arxiv.org/pdf/2610.31107v1)

**作者:** Yiqi Wang `[一作]`, Taotao Cai `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了主动偏好获取（Active Preference Acquisition）框架，定义了跨任务的期望个性化价值（Prospective Personalization Value），并在一个模拟的长期个人化Web代理测试床上实现并评估了该方法。

**💡 创新点**

核心创新包括：
- 将偏好获取视为跨任务的成本收益问题，区分即时任务与未来任务的价值；
- 引入“前瞻个性化价值”概念，量化未知偏好在未来任务中的累计效用；
- 采用“反事实重用估值”（counterfactual reuse valuation）结合预算约束的自适应贪婪策略，实现可解释且可控的问答选择；
- 在同一实验平台上系统对比了多种基线（随机、频率、影响、LLM重要性、当前任务VoI、oracle等），展示跨任务评估的显著优势。

**🔧 技术方法**

技术实现主要包括：
- Qwen3‑8B LLM 负责候选偏好生成、未来任务分布预测和价值评估；
- 任务求解器使用确定性规则式求解器，能够在给定偏好下计算任务效用；
- 预算约束下的自适应贪婪采样算法；
- 反事实重用估值：对每个候选偏好，使用公共任务原型计算在每个任务族中的相关性和影响，再按预估的未来任务出现频率加权。

**📊 数据集**

实验数据集是自行构建的“长周期个人化测试床”，包含：
- 90名 IID 用户 + 90 名拥有未见偏好组合的用户；
- 27个隐藏属性（稳定、半稳定、短暂、敏感等）；
- 16个任务族（航班、酒店、餐厅、购物、会议等）共 20 个未来任务；
- 任务原型和精确效用可计算，允许对未提问偏好进行反事实评估。

**📈 对比分析**

与多种基线对比（随机、频率、影响、LLM重要性、当前任务VoI、oracle 等）结果表明：
- APA 在实际可行策略中获得最高的净效用（6.755 vs 6.664，差值 0.091，p<0.1）；
- 相比完全提问（Ask‑Everything），APA 只需 27.6% 的问题，却恢复了 84.2% 的效用增益；
- 在成本匹配实验中 APA 仍优于随机、频率、LLM重要性等基线，净效用提升 0.5–0.6；
- 通过“初始配置折算诊断”验证，APA 对 34.4% 的可行但需累计使用的偏好获得 80.3% 的获取率，远高于 Current‑Task VoI（仅 5.1%）。

**⚠️ 局限性**

主要局限：
- 所有用户与任务均为合成，未在真实用户或现场 Web 交互中验证；
- 仅使用单一 LLM 后端（Qwen3‑8B）和固定求解器，未探讨更强大模型或不同任务求解方法的泛化；
- 预算限制仅为每次交互，未考虑累积预算；
- 未来任务分布估计依赖 LLM 预测，若预测误差大可能导致无效提问；
- 隐私风险被简化为成本项，未提供正式的隐私保证或可解释的同意流程；
- 只考虑公开属性，未处理动态或短期偏好的更新问题。

---

## 428. The Reversible Intermediate Language L4, its Type System, and its Bidirectional Small-Step Semantics

**arXiv ID:** 2610.31201 | [PDF](https://arxiv.org/pdf/2610.31201v1)

**作者:** Torben Ægidius Mogensen `[一作]` `[通讯]` (University of Copenhagen), Torben Ægidius Mogensen (University of Copenhagen)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并正式定义了一种可逆、线性、低级语言L4，提供类型系统和双向小步语义，并实现了解析器、类型检查器与解释器；

**💡 创新点**

将可逆SSA形式RSSA与混合SSA HSSA相结合，加入类型系统、可逆函数调用、通用多路跳转，并给出了完整的双向执行语义；

**🔧 技术方法**

利用线性类型、函数类型、语义规则推导和ML实现的编译器工具；

**📊 数据集**

未使用特定数据集，主要以自定义示例程序验证语言设计；

**📈 对比分析**

通过实现的Prolog/ML解释器对正向和逆向执行进行测试，但未给出性能基准，主要关注语义正确性；

**⚠️ 局限性**

语言语法繁琐、阅读和编写困难，线性变量约束导致检查复杂，尚未提供自动优化或大规模应用验证。

---

## 429. Towards Field-Deployable Wearable IMU Systems for Occupational Activity Recognition

**arXiv ID:** 2610.31160 | [PDF](https://arxiv.org/pdf/2610.31160v1)

**作者:** Ihsan Ul Haq `[一作]` (Coventry University), Chandan Karmakar `[通讯]` (Deakin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在VTT‑ConIoT数据集上对可穿戴IMU系统进行全面的配置敏感性分析，系统评估传感器摆放、窗口长度、六种经典分类器以及两种数据增强方法对工地工人活动识别性能的影响，并验证少量个体化校准能显著提升最弱用户的准确率。

**💡 创新点**

首次按个人LOS分解16类多传感器配置的性能；系统化比较传感器位置、窗口大小与分类器的交互效果；验证CTGAN生成对抗网络在IMU HAR中的不利影响；展示少量个体化数据即可恢复最弱用户的10%准确率提升。

**🔧 技术方法**

使用传统统计特征提取（147维）+六种Scikit‑learn经典分类器（RF、ET、XGB、LDA、LR、SVM），CTGAN对抗生成增强，UM等时序变换增强，1‑D CNN基线，K‑S检验评估合成数据分布。

**📊 数据集**

VTT‑ConIoT：13名健康成年工人，16种工地活动，三种IMU位置（臀部、背部、手/肩），5/2/10窗口配置。

**📈 对比分析**

采用统一的participant‑wise LOSO交叉验证，报告每个传感器组合与分类器的LOS平均准确率；结果显示多传感器融合+5窗口+SVM可达约77%准确率；传统时序增强轻微提升≈0.5%；CTGAN降准≥1–5%；个体化校准可提升最弱用户约10%。

**⚠️ 局限性**

仅在实验室收集数据，缺乏真实工地环境验证；依赖传统特征提取，未探索深度学习更高层次表示；CTGAN生成对抗网络在IMU时间窗中表现不佳；个体化校准需要额外采集数据，未评估长期部署成本。

---

## 430. Calibration-free IMU-based Relative Orientation Estimation under Dynamic Human Motion

**arXiv ID:** 2610.31156 | [PDF](https://arxiv.org/pdf/2610.31156v1)

**作者:** Ruiyuan Li `[一作]` (Delft University of Technology), Manon Kok `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种无磁力计、无预先校准的扩展卡尔曼滤波框架，能够在线同时估计相邻IMU的相对方向和传感器在身体部位上的位置。

**💡 创新点**

创新点在于：① 将相对方向与传感器位置的联合估计融入到IMU运动学约束模型中；② 使用低通滤波显式处理步态冲击导致的高频噪声与数值微分误差；③ 在无需外部参考或预先校准的情况下，在多种运动情境下实现高精度相对姿态估计。

**🔧 技术方法**

采用扩展卡尔曼滤波（EKF）结合刚体约束模型、零相位巴特沃斯低通滤波、5点数值微分以及观测噪声协方差自适应估计；在仿真中通过不同的运动强度参数对估计性能进行系统评估。

**📊 数据集**

使用OpenSense公开数据集，该数据集包含Xsens MTw IMU记录（100 Hz）和Vicon光学标记基准，涵盖多名受试者在步行和复杂任务（坐姿、上下楼梯、侧步、跑步）中的运动。

**📈 对比分析**

与现有的Weygers方法（需离线传感器位置校准）及其多种变体（含/不含低通滤波、NIS排除、角速度状态扩展）进行对比；实验结果显示总体相对姿态误差降至2.61°，比Weygers方法低约75%；低通滤波显著降低误差，且误差随关节中心加速度和相对角速度强度变化。

**⚠️ 局限性**

局限性包括：① 对关节中心加速度和相对角速度的可观测性依赖，低强度运动时估计不稳定；② 在铰链式关节中传感器位置的某些分量不可观测，导致位置误差较大；③ 假设传感器位置近似恒定，缺乏完整的传感器到解剖部位的自动对准，进一步改进需要实现传感器与部位的标定。

---

## 431. Prompt Sensitivity in Tool-Augmented Model Comparisons

**arXiv ID:** 2610.31148 | [PDF](https://arxiv.org/pdf/2610.31148v1)

**作者:** Mateusz Modrzejewski `[一作]` (Warsaw University of Technology), Tomasz Radzikowski `[通讯]` (Warsaw University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了工具调用（Tool use）对语言模型在多种词义、提示变体下的问答准确率与模型比较稳定性的影响，并针对六个开源模型在五个任务领域开展系统评估。

**💡 创新点**

创新点在于提出了匹配协议，分离答复读取、推理、调用决策、工具交互等步骤，并将词义波动拆解为调用频率与条件准确性两部分；同时通过对工具接口、说明书和报告质量的多重干预，系统揭示工具使用对比较稳定性的真实影响。

**🔧 技术方法**

采用了ReAct式工具调用循环、贪婪解码以及模型原生工具接口；使用Bootstrap与Shapley分解评估词义扩散与调用频率；对六种模型在五个领域进行大规模实验，结合不同写法和工具描述。

**📊 数据集**

使用了分子（PubChem SMILES、InChI、IUPAC）、图（随机图edge list、neighbor list）、音乐（ABC-Eval ABC、MusicXML、note events）、表格（FinQA markdown、CSV、句子）和数学（MATH‑500）等数据集，分别包含数百条问答实例。

**📈 对比分析**

通过计算同一模型在不同词义下的准确率差异定义词义扩散；与无工具推理、固定读出、报告供应等基线进行对比；结果表明工具能提升准确率，但往往放宽词义敏感性；特定接口和说明可消除大部分宽度；比较稳定性随模型和领域差异显著。

**⚠️ 局限性**

局限性包括：仅使用五个词义与有限模型与任务，报告注入仅为控制不等同实际知识；工具只解答直接问题，缺乏工具发现与参数构造；接口干预仅覆盖三模型；调用频率分解非因果推断；评估受样本规模与词义多样性的限制。

---

## 432. Post-Audit Mirage: Identifiability Limits of Offline Agent Verification

**arXiv ID:** 2610.31130 | [PDF](https://arxiv.org/pdf/2610.31130v1)

**作者:** Aman Behera `[一作]` (Indian Institute of Technology Roorkee), Payal Payal `[通讯]` (Indian Institute of Technology Roorkee)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了离线审核记录无法判定部署安全性的问题，提出可执行的不可区分边界和“Post‑Audit Mirage”基准，并设计了基于组标签的失效关闭监控器来恢复安全决策。

**💡 创新点**

将不可区分性理论转化为可执行的审核接口，构建匹配安全-有害生命周期基准，并提出利用新鲜组标签结果的监控机制来判定部署安全。

**🔧 技术方法**

使用可观测等价与两点不可区分性理论、Hoeffding、Wilson 置信区间及 beta‑binomial 混合置信序列等统计方法，结合模拟匹配世界实验和离线诊断规则对比。

**📊 数据集**

生成了 100 个 AuthGate、100 个 ConstraintPlan、72 个 BatchTriage 的安全‑有害对（272 对），每对进行 50 轮更新，合计约 20,000 次模拟观测作为实验数据。

**📈 对比分析**

对 9 种离线诊断规则在匹配世界下得到完全一致决策；监控器在安全边际大、受影响组常见且观测完整时，在约 20,000 次观测内实现安全与有害区分，表现出在更大安全裕度或更多观测时需更少数据。

**⚠️ 局限性**

依赖匹配世界构造，假设独立固定率二元结果、观察缺失独立于结果、目标分布稳定，无法检测未声明的选择偏差、缺失不随机、组缺失或目标转移，且未评估真实生产环境的普遍性。

---

## 433. Who Is the Influencer? Consensus Dynamics in Multi-Agent Systems

**arXiv ID:** 2610.31105 | [PDF](https://arxiv.org/pdf/2610.31105v1)

**作者:** Gouki Minegishi `[一作]` (University of Tokyo), Yutaka Matsuo `[通讯]` (University of Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一个命名游戏式的实验，将不同规模的LLM混合在同一群体中，通过观察最终共识标签来评估模型在多智能体系统中的影响力。

**💡 创新点**

提出在无先验知识的极简任务下衡量多模型系统中“影响力”的方法，并用物理能量-粒子模型解释强模型主导共识的现象。

**🔧 技术方法**

利用命名游戏协议、无监督多智能体交互、能量函数与复制器方程理论，并从实验中估计模型的 assertion strength。

**📊 数据集**

使用自定义的无意义五字标签（K=3）和随机内存初始化，无需外部真实数据集。

**📈 对比分析**

对七个模型族进行108次实验，统计每个族中最强模型标签获胜比例，结果显示最强模型以约70‑80%赢得共识，显著高于随机基准1/3或1/2。

**⚠️ 局限性**

实验仅在完全混合的单标签交互中进行，未考虑网络结构、文本自由交互或真实任务目标，无法推断在有目标答案情形下的影响力差异。

---

## 434. RIFT: KV Cache Compression for Draft Models with Complementary Compressibility Training

**arXiv ID:** 2610.31098 | [PDF](https://arxiv.org/pdf/2610.31098v1)

**作者:** Dor Tsur `[一作]` (NVIDIA), Ran Zilberstein `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在长上下文推理中KV缓存压缩导致的解码陈旧问题，并提出了结合递归性与重要性选取的KV压缩策略。

**💡 创新点**

创新点在于首次量化解码陈旧（decode staleness），并提出RIF（Recency-Importance Fusion）及其自适应刷新机制，同时给出压缩感知的知识蒸馏训练。

**🔧 技术方法**

使用了基于KV选择的压缩方法、可调节滑动窗口、稀疏注意力掩码、压缩感知KD、以及自适应KV刷新技术。

**📊 数据集**

实验使用了SPEED‑Bench 32K输入、4K输出、Qwen3.5‑4B/122B、Nemotron3.5‑Lightning以及NVIDIA Nemotron V2数据集。

**📈 对比分析**

与SnapKV、SWA等基线对比，RIF在固定8K KV预算下保留了超过96%的接受长度，压缩率为4×，并在压缩感知训练后进一步提升约16%。

**⚠️ 局限性**

局限性包括外部草稿模型在单GPU部署下速度提升有限、KV重新评分的额外延迟约占6%以及假设完整KV历史可用的前提。

---

## 435. Different Tasks, Different Timesteps: Temporal Task-Aware Quantization for Diffusion Language Models

**arXiv ID:** 2610.31198 | [PDF](https://arxiv.org/pdf/2610.31198v1)

**作者:** Ofek Avivi `[一作]` (Technion Israel Institute of Technology), Chaim Baskin `[通讯]` (Ben Gurion University of Negev)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了任务与时间感知的混合精度量化框架，能够在保持总位宽预算不变的情况下，针对扩散式语言模型（DLM）在不同解码步和不同下游任务中对量化精度的敏感性进行动态分配。

**💡 创新点**

创新点在于：①发现量化敏感性既随任务又随时间变化；②设计了一种无标签、基于内部KL散度与后向影响的信号来预测各解码窗口对精度提升的收益；③利用该信号通过动态规划在固定预算下求解最优精度分配方案。

**🔧 技术方法**

技术手段包括：量化激活时使用动态循环量化、无标签内部模型信号的计算、对每个解码窗口的KL散度与影响度量、预算约束下的动态规划优化。

**📊 数据集**

数据集涵盖六项基准：GSM8K、HumanEval、ARC‑Challenge、MMLU‑Pro、DROP、BBH，实验使用三款公开的 DLM：Dream‑7B‑Instruct、LLaDA‑1.5‑8B、LLaDA‑8B‑Instruct。

**📈 对比分析**

与基线方法（FP16、RTN、DLLMQuant、FAIR‑Calib、STaR‑Quant）以及随机混合精度方案比较，框架在保持平均激活位宽4位的前提下，平均提升约6.7个百分点，最优配置在多数基准上逼近 FP16 甚至超越。

**⚠️ 局限性**

局限性包括：缺乏对硬件层面加速与能耗的评估，未探究对模型安全性与鲁棒性的潜在影响，且主要在公开 DLM 与标准基准上验证，需进一步扩展到更大模型与实际部署场景。

---

## 436. Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation

**arXiv ID:** 2610.31166 | [PDF](https://arxiv.org/pdf/2610.31166v1)

**作者:** Chenyuan Chen `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Yoshihiko Nakamura `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于对象事件的感知记忆机制，用于全身视觉‑语言‑动作（VLA）策略，并提出了HuMeGo人形机器人记忆基准。

**💡 创新点**

创新点在于利用SAM3实时跟踪任务对象，基于对象事件挑选关键帧并在自适应预算下存储；在动作专家中引入跨注意力与FiLM调制读取记忆，并添加决策头映射隐藏线索。

**🔧 技术方法**

使用的技术包括SAM3分割/跟踪、跨注意力机制、FiLM调制、对象令牌、决策头、基于流匹配的Transformer动作专家、Qwen3‑VL‑2B视觉‑语言模型。

**📊 数据集**

数据集为HuMeGo基准：6个模拟任务+6个真实任务；模拟任务采集100个演示和90个测试场景，真实任务采集15个演示；使用Unitree G1机器人、Isaac Sim与LeRobot格式。

**📈 对比分析**

通过与无记忆策略（Ψ_0等）和均匀记忆（RoboMME风格）对比，采用SIMPLE任务和记忆任务评估，SIMPLE成功率提升至86.9%（对比70.2% Ψ_0），记忆任务成功率达到72.6%（对比29.1% Ψ_0），真实机器人上成功率提升至88.3%（对比36.7% Ψ_0）。

**⚠️ 局限性**

局限性包括在线对象跟踪带来的推理成本、SAM3偶尔漏检导致关键帧缺失、仿真与真实环境的差距、演示数据为脚本化、仅关注预历史记忆，未覆盖更长期的推理与多步决策。

---

## 437. Beyond Watching and Searching: Benchmarking Agents for Video-centered Deep Research Reports

**arXiv ID:** 2610.31158 | [PDF](https://arxiv.org/pdf/2610.31158v1)

**作者:** Jiahao Meng `[一作]` (Alibaba Token Hub), Yunhai Tong `[通讯]` (Alibaba Token Hub)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了视频中心的深度研究报告生成任务及其公开基准 OmniDR‑Bench，并设计了细粒度评估树 Omni‑ART 与审核构造流程 ARC；

**💡 创新点**

创新点在于：①将单一视频作为研究根基，生成完整可交互、可引用的多模态报告；②构建可审核的评估树 Omni‑ART 与证据构造流程 ARC；③公开多语言、多模态任务集与评测工具，促进后续研究；

**🔧 技术方法**

使用多模态智能体工作流，包含视频阅读、文本/图像/视频检索、跨源推理等工具，核心模型为 Claude、GPT、Gemini、Qwen 等大型语言模型；评估结合程序化检查与多模态模型判断；

**📊 数据集**

自建包含 50 任务、四类视频（知识/观点、消费/商业、物理/程序、创意/数字）且覆盖中英文的任务集，外部多模态检索证据与公开网络资源；

**📈 对比分析**

对 18 种模型+Harness 配置进行对比，平均得分约 60/100，最佳系统得分 63/100，成功率 90% 以上；在内容、引用、呈现三维度分别得分 45、48、82，呈现表现突出但内容与引用仍显不足；

**⚠️ 局限性**

主要局限包括：证据覆盖不全面、跨模态因果推理弱、引用闭合不足、效率与算力成本高、模型在视频分析与推理方面仍需改进；

---

## 438. Navigating human-robot interaction challenges in private and residential aged care homes

**arXiv ID:** 2610.31149 | [PDF](https://arxiv.org/pdf/2610.31149v1)

**作者:** David Ahmedt-Aristizabal `[一作]` (CSIRO), Simon Denman `[通讯]` (Queensland University of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本章综述了在私人住宅与养老护理院（RAC）环境中人机交互（HRI）的挑战与机遇，构建了一个系统化的框架，涵盖极端环境定义、交互模式、感知与学习技术、隐私与信任设计、评估指标及未来研究方向。

**💡 创新点**

创新点包括：①将RAC与私人住宅视为“极端”HRI场景，系统化地把布局、技术支持与护理工作流程等多维限制纳入设计空间；②提出以实际运营指标（误报/日、恢复时长、护理工作负担、长期参与度）为核心的评估方法；③将隐私友好、可解释、可扩展的参考架构与多模态感知与学习范式（监督、弱监督、SSL、few‑shot、meta‑learning、域适应）有机结合；④给出可落地的隐私‑by‑design与数据治理策略，帮助实现合规且可持续的部署。

**🔧 技术方法**

技术手段：多模态感知（RGB‑D、LiDAR、毫米波雷达、环境与可穿戴传感器、音频），边缘与云推理、事件驱动的隐私保护（on‑device inference、Federated Learning、数据摘要），学习范式（监督、弱监督、SSL、few‑shot、meta‑learning、域适应），系统架构（本地集线器、边缘计算、FHIR/EMR 接口、可视化状态指示）以及可解释与安全的交互设计。

**📊 数据集**

主要引用的公开数据集与仿真平台包括：Matterport3D、Replica、iGibson、Habitat、OpenAI Whisper 语音模型、以及多种养老护理院与家庭场景的现场收集数据。章节中并未报告新的实验数据，而是以这些公开资源为评估基准进行分析。

**📈 对比分析**

比较与评估方法：从仿真→实验室→试点→长期部署四个阶段，指标从传统准确率、召回率等机器学习指标过渡到误报/日、恢复时长、护理工作负担、dock 成功率、用户参与率等运营与人因指标。文中指出现有技术在感知准确率上已可达 90%+，但在极端环境下的误报、恢复与长期可靠性仍高于可接受阈值，且隐私友好与可解释性往往需在性能上做权衡。

**⚠️ 局限性**

局限性：①缺乏大规模、跨文化、长时间真实部署的数据与验证；②隐私与学习精度的权衡难以统一解决；③硬件资源（计算、功耗、体积）对边缘推理与多模态融合的可行性构成约束；④跨域适应与自适应学习在持续漂移与安全约束下的稳健性尚未成熟；⑤监管合规与伦理治理框架在不同司法辖区仍不一致，导致部署路径不确定。

---

## 439. Mixing Matters: Evidence Position Bias Across Sequence Mixers in Long-Context Question Answering

**arXiv ID:** 2610.31135 | [PDF](https://arxiv.org/pdf/2610.31135v1)

**作者:** Aman Behera `[一作]` (Indian Institute of Technology Roorkee), Mehul Anand `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估 Transformer、状态空间与混合 Mamba 架构在长上下文中对证据位置偏差的表现，使用 Lost‑in‑the‑Middle 十文档问答任务对金文档位置进行配对测试。

**💡 创新点**

揭示不同序列混合器家族表现出不同的首位优势与末位优势，且差异仅在模型能可靠回答任务时显现，提示位置偏差受架构、能力、任务与提示共同影响。

**🔧 技术方法**

采用配对 bootstrap 估计、primacy/recency 指标、线性位置可解探针与注意力集中度分析等技术，对金文档在十个位置的准确率进行评估。

**📊 数据集**

使用 Natural Questions 衍生的 Lost‑in‑the‑Middle 10 文档问答数据集，包含约 800 个探索性问题。

**📈 对比分析**

在相同十文档、固定干扰文档的条件下移动金文档，对各模型在不同位置的准确率进行比较，结果显示 Transformer 在首位有正偏差，状态空间无首位偏差但两者都有末位优势；混合模型与纯模型差异不显著。

**⚠️ 局限性**

实验样本有限，仅覆盖少数模型与提示；混合模型与纯模型对照不确定；内部测量为相关性且未验证因果机制；留存集未评估。

---

## 440. A Grammar of French Civil Cassation: A Ground-Level Dataset and Tool

**arXiv ID:** 2610.31116 | [PDF](https://arxiv.org/pdf/2610.31116v1)

**作者:** Avrile Floro `[一作]` (Télécom Paris, Institut Polytechnique de Paris), Maria Boritchev `[通讯]` (Télécom Paris, Institut Polytechnique de Paris)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含121,536条法国民事上诉理由-答复对的结构化数据集，并实现了可查询的交互式工具；

**💡 创新点**

首次在法语民事法院层面实现了从判决文本到理由、结果及教义家族标签的自动映射，并提供了基于规则的“语法”描述；

**🔧 技术方法**

采用了基于正则表达式的确定性规则对判决功能区进行配对、结果判定与教义分类，同时利用JuriBERT等预训练模型作为基线；

**📊 数据集**

使用了2016–2025年法国最高民事法院（Cour de Cassation）公开判决数据，涵盖5个民事分庭共86,464份裁决；

**📈 对比分析**

与基线模型（JuriBERT、CamemBERTv2、Qwen3-32B）比较，规则在已判理由的教义分类上达到约91.5%准确率，模型在拒绝理由上略优但在接受理由上低于规则；

**⚠️ 局限性**

主要限制在于规则对非标准表述的识别能力不足，裁决文本标注依赖2021年起的功能区划分，且教义分类基于Boré的固定税onomy，可能低估某些家族出现频率。

---

## 441. Where Do Long Captions Fail? Position-Aware Diagnosis and Reinforcement Learning for Detailed Image Captioning

**arXiv ID:** 2610.31111 | [PDF](https://arxiv.org/pdf/2610.31111v1)

**作者:** Yuanze Hu `[一作]` (Beihang University), Han Li `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种位置感知的长篇图像描述诊断与强化学习框架，旨在检测并纠正长文本生成中出现的“中段不支持的细节”与“尾段泛化”两类错误。

**💡 创新点**

创新点在于：① 将参考描述拆解成按位置关联的原子视觉声明（rubric）；② 用验证器将声明与生成文本对应，得到句子级支持/错误/泛化标签；③ 设计四项位置分离奖励（覆盖、细粒度中段、尾段泛化抑制、长度预算）以闭环修正发现的错误。

**🔧 技术方法**

技术手段包括：大语言模型（Qwen3‑VL‑4B）、基于检查表的验证器、GRPO（基于梯度的策略优化）以及对句子位置的四分位区分与奖励分配。

**📊 数据集**

使用的数据集：混合的 IIW‑400 与 DOCCI 细粒度描述数据，PixMo‑derived 语料用于构建和筛选 rubric 训练集，另外还引用 DCI 与 DOCCI 训练集进行 rubric 构建。

**📈 对比分析**

在多项评测（CAPTURE、DCScore‑F1、事实密度、长度、CapArena 4‑模型偏好）中，所提方法在定位错误方面显著提升（mid‑caption hallucination 从 +0.096 降至 +0.036，tail‑supported 最高，tail‑hallucination 最低），在整体质量上亦与 CapRL 和 SFT 接近或略优，且在四模型偏好上达 96% 以上的胜率。

**⚠️ 局限性**

局限性包括：① 验证器仅基于参考文本，可能忽略参考缺失或误判；② 位置分区和奖励权重需要人工调参；③ 在更大规模模型或更长预算下的可推广性尚未完全验证；④ 生成细节的真实性仍受限于引用集的完整性。

---

## 442. DeepOpen: Decision-Oriented Intent Classification on BANKING77 and CLINC150

**arXiv ID:** 2610.31104 | [PDF](https://arxiv.org/pdf/2610.31104v1)

**作者:** Shuang Song Zhongqiao Yuan `[一作]` `[通讯]` (Shenzhen Depei Open Source Data Technology Co., Ltd.), Shuang Song Zhongqiao Yuan (Shenzhen Depei Open Source Data Technology Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出DeepOpen框架，在BANKING77上使用SAM优化单模型，在CLINC150上采用四模型概率融合实现高准确率。

**💡 创新点**

将Laya的灵活候选接口改为固定标签分类头，并分别使用SAM优化与概率融合，展示在固定标签任务中既高效又精确。

**🔧 技术方法**

使用Laya编码器、SAM（Sharpness‑Aware Minimization）、R‑Drop、DeBERTa‑v3‑large及概率融合等技术。

**📊 数据集**

使用BANKING77（77类银行意图）和CLINC150（150类意图）两个英语意图分类数据集。

**📈 对比分析**

与历史检查点、SPACE、Llama‑2等基线对比；BANKING77单模型达到94.1234%（相较历史+0.0433%无显著提升），CLINC150四模型融合得到98.0222%（虽低于某些小模型基线但展示融合带来的提升）。

**⚠️ 局限性**

结果受模型选择、种子以及执行路径漂移影响，统计意义不确定；CLINC150融合需多模型推理，资源成本高；缺乏跨语言、鲁棒性评估；历史执行路径漂移导致可复现性问题。

---

## 443. Investigating Multi-Cultural Value Selection Biases in LLMs

**arXiv ID:** 2610.31199 | [PDF](https://arxiv.org/pdf/2610.31199v1)

**作者:** Siddhesh Pawar `[一作]` (University of Copenhagen), Isabelle Augenstein `[通讯]` (University of Copenhagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在多文化身份下，LLM对国家和语言双重身份提示的价值表达变化

**💡 创新点**

证明语言与国家提示对LLM价值表现的影响不同，且模型对这两类信号的敏感度差异显著，提出多身份评估框架

**🔧 技术方法**

使用提示工程、Jensen–Shannon距离、Value‑in‑the‑Wild 注释器、机器翻译和多模态评分等技术

**📊 数据集**

采用 World Values Survey (WVS) 多选题、IssueBench 开放式生成、AIRiskDilemmas 决策情境以及 WVS 翻译数据集

**📈 对比分析**

通过比较 JSD 距离缩减、价值出现比例变化和决策排名变化进行评估，发现语言提示导致更大分布偏移，国家提示在开放式文本中产生更细致的价值变化，不同模型表现差异显著

**⚠️ 局限性**

受限于缺乏真实多文化身份数据、使用国家/语言作为文化代理、仅考察两重身份、地区覆盖有限以及外部模型错误可能影响结果

---

## 444. Structural cryptanalysis of classic McEliece: Assessment of claims, costs and evidence

**arXiv ID:** 2610.31196 | [PDF](https://arxiv.org/pdf/2610.31196v1)

**作者:** Ramses Fernandez `[一作]` `[通讯]` (AegisQ Technologies), Ramses Fernandez (AegisQ Technologies)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并评估2024-2026年关于经典McEliece的结构攻击，区分鉴别器、条件密钥恢复估计和实际恢复，详细分析Saarinen HOVER、Weis等贡献，并给出自己的成本模型评估。

**💡 创新点**

提供了版本感知的批判性评估，揭示成本模型不确定性是主要量化不确定来源，比较不同攻击与Grover加速ISD的效果，并提出公开问题与后续研究方向。

**🔧 技术方法**

结构攻击分析、线性代数与块Lanczos/Wiedemann求解、MinRank搜索、符号模型、端到端成本评估、比较基准的安全类别、量子加速估算等技术。

**📊 数据集**

TII挑战（如TII-254等）以及五个Classic McEliece参数集（m,n,t,k）的公开密钥/密码实例。

**📈 对比分析**

通过将攻击成本与NIST安全类别（AES-128/192/256）和Grover-ISD基准对比，发现所有成本模型下条件估计仍低于目标安全水平；大部分攻击的实际成本与所需的稀疏线性代数计算相距数十位，且尚未在完整参数规模上验证。

**⚠️ 局限性**

成本模型宽范围导致估计不确定；缺乏对完整规模实例的实证验证；攻击在理论上条件成立，但缺乏概率论证；实验仅涵盖小型TII挑战，未证明对实际参数的普适性。

---

## 445. Large Language Models Meet Bayesian Optimization: Towards Self-evolving Wireless Digital Twins

**arXiv ID:** 2610.31192 | [PDF](https://arxiv.org/pdf/2610.31192v1)

**作者:** Yingzhuo Sun `[一作]` (KTH Royal Institute of Technology), Berna Ozbek `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一种基于大型语言模型辅助贝叶斯优化的自演进无线数字孪生框架

**💡 创新点**

创新点在于将LLM嵌入固定的BO框架中，让LLM根据任务经验自适应地生成或修改BO组件（如高斯过程核），实现算法级的自演进

**🔧 技术方法**

使用技术包括大型语言模型（LLM）、贝叶斯优化（BO）、高斯过程回归、核函数设计、进化搜索、经验数据库与提示构建器

**📊 数据集**

使用的数据集是由Sionna生成的城市环境下多用户下行波束选择场景，共50个任务（30个用于训练，20个用于测试）

**📈 对比分析**

比较方法：将LLM发现的核函数与四个标准GP核（Matérn-1/2、3/2、5/2、RBF）在相同BO设置下进行收敛比较；在训练任务上提升19.7%，在测试任务上提升8.7%

**⚠️ 局限性**

局限性包括：任务变更范围受限于同一系统模型；LLM生成的组件需进一步验证可靠性；算法级适配的计算成本与延迟仍需优化

---

## 446. SARIA: Persistence-Aware KV Caching with Coordinated DRAM--SSD Reload for LLM Agents

**arXiv ID:** 2610.31169 | [PDF](https://arxiv.org/pdf/2610.31169v1)

**作者:** Hui Li `[一作]` (Chinese University of Hong Kong), James Cheng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

Saria 通过会话层时空局部性实现了 DRAM–SSD 层次 KV 缓存，减少 SSD 写入并加速重加载。

**💡 创新点**

创新点在于：基于 evict‑reuse 历史的持久化决策、延迟的部分 DRAM 提升以及协同的多源重载调度。

**🔧 技术方法**

使用技术包括：统一上下文元数据、GDS 直连、DRAM–SSD 组合缓存、按需部分 DRAM 提升以及协调器调度。

**📊 数据集**

实验使用 TraceLab 与 WEKA 的 LLM 代理请求重放数据集。

**📈 对比分析**

与 DRAM‑only、Full‑Duplicate 等基线比较，Saria 在写入量降低约 58% 后，平均 TTFT 降低 42%–80%，且 SSD 写入预算保持可控。

**⚠️ 局限性**

局限在于需要手动调节阈值，适用场景依赖 SSD 并行度与会话局部性，协调器在 SSD 带宽足够时开销可能不明显。

---

## 447. WildFakeToolBench: Benchmarking AI-Generated Image Detectors with Web-Sourced Image Processing

**arXiv ID:** 2610.31152 | [PDF](https://arxiv.org/pdf/2610.31152v1)

**作者:** Mingjie Li `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了 WildFakeToolBench——一个由 220 个公开来源、142 个领域衍生出的 156 个可执行图像处理工具集合，旨在评估 AI 生成图像检测器的鲁棒性；

**💡 创新点**

创新点在于利用 LLM（GPT‑4o‑mini）将网络讨论、教程等非正式文本自动提炼、去重并转化为可执行的 CPU‑级图像处理脚本，同时将这些工具与传统学术攻击基线进行对比，揭示公共知识对检测器安全的实际威胁；

**🔧 技术方法**

主要技术包括：① 文本检索（Tavily）、知识预处理与模板化；② 方法提取与去重；③ 代码合成与验证（Python + Pillow/OpenCV 等库）；④ 对生成图像进行多参数尝试并用 SSIM 过滤；⑤ 对六个顶尖检测器进行单工具和多工具组合攻击评估；

**📊 数据集**

使用 105 张 AI 生成图像，来源为 Stable Diffusion 2.1、Qwen‑Image‑2.0、Midjourney、GPT‑Image 2、Nano Banana 2，覆盖 7 类内容（动物、物体、文件、人物、场景、卫星、面部），并将每张图像分别处理 156 个工具；

**📈 对比分析**

在单工具攻击下，攻击成功率（ASR）最高可达 100%（对某些检测器），平均 ASR 低至 3‑21%；在八工具组合池中，覆盖率最高可达 92%（对某些检测器），所有六个检测器覆盖率均超过 80%；与六个传统基线相比，许多公开来源生成的工具具有更高的 ASR；

**⚠️ 局限性**

局限性包括：① 仅利用公开可抓取的文本，可能漏掉深度或付费社区的高级技术；② 所有工具均为 CPU 级，未涵盖 GPU 强化或深度网络级后处理；③ 评估仅在 105 张精心挑选的图像上，未覆盖真实世界多样性；④ 只针对黑盒检测器，未考虑白盒或半白盒攻击场景；⑤ 评估未探讨长期链式处理导致的累积失真与鲁棒性问题。

---

## 448. Rethinking Fisher Vectors for Second-order Multi-scale Deep Feature Aggregation in Scalable Scene Retrieval

**arXiv ID:** 2610.31146 | [PDF](https://arxiv.org/pdf/2610.31146v1)

**作者:** Alexy Bhowmick `[一作]` (Assam Don Bosco University), Shyamanta M. Hazarika `[通讯]` (Indian Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无视觉词典、基于 Fisher Kernel 的多尺度二阶卷积特征聚合方法，用于构建高效的图像检索向量。

**💡 创新点**

创新点在于将多尺度 CNN 激活的二阶统计信息直接嵌入 Fisher Vector 框架，无需手工构造视觉词典，并通过平均池化实现尺度不变性，显著提升了检索判别力。

**🔧 技术方法**

核心技术包括多尺度 CNN 特征提取（VGG-16/19、ResNet-18/34）、高斯混合模型编码、二阶 Fisher Vector 聚合、α 次幂归一化与 L2 归一化，以及随机 kd‑tree 搜索。

**📊 数据集**

在两大公开基准上验证：INRIA Holidays（1491 张图）和 UKBench（10200 张图）。

**📈 对比分析**

与 VLAD、传统 Fisher Vector、R‑MAC、MDFA 等先进方法对比，本文在 INRIA Holidays 上取得 83.42% mAP，在 UKBench 上得到 3.76 的 N‑S 分数，性能优于现有方案，同时维度仅略增。

**⚠️ 局限性**

局限性包括：对 GMM 参数（如 k、α）敏感，需要手工调参；计算与存储成本随 GMM 组件数增长；在极大规模数据集上对 kd‑tree 近似搜索的准确性可能受限。

---

## 449. GM-GReFEL: Geometry-Guided Spatiotemporal State-Space Learning for Robust Dynamic Facial Expression Recognition

**arXiv ID:** 2610.31140 | [PDF](https://arxiv.org/pdf/2610.31140v1)

**作者:** Yudhistira Arditya Pratama `[一作]` (National Taiwan University of Science and Technology), Yi-Zeng Hsieh `[通讯]` (National Taiwan University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一种面向无拘束视频表情识别的GM-GReFEL架构，该架构将线性复杂度的Modulated GroupMamba状态空间网络与基于几何锚点的可靠性模块结合。

**💡 创新点**

创新点包括：① 通过将通道划分为四个空间扫描组和两个时间扫描组，实现了自注意力的O(N^2)成本降为O(N)；② 采用预测熵指导的几何可靠性机制，用可学习的几何锚点在模型不确定时提供解码补偿；③ 采用静态到动态的权重迁移初始化和Triple Loss（三项损失）进一步提升鲁棒性。

**🔧 技术方法**

技术方法包括：Modulated GroupMamba（可选择状态空间扫描）、Channel Affinity Modulation、熵引导的几何可靠性（GReFEL）、静态到动态权重迁移、Triple Loss（锚点正交性+中心损失）、随机时间采样、位置编码与标准化处理等。

**📊 数据集**

使用了三大无拘束视频表情数据集：DFEW、FERV39k 和 MAFW。

**📈 对比分析**

与 3D CNN、Transformer、ViT 等多种基线进行对比；在最佳折叠上 DFEW 获得 66.57% UAR / 76.46% WAR，接近 S4D；在 MAFW 获得 45.66% UAR / 62.27% WAR，超过所有对比方法；在 FERV39k 的五折平均取得 42.30% UAR / 52.66% WAR，表现竞争力；总体而言，五折平均性能与近期大型预训练 ViT 模型相当。

**⚠️ 局限性**

局限性：仅使用视觉模态，未融合音频或手势信息；仅实现离散类别识别，未建模连续维度情感；模型参数量和算力需求较大，尚未进行量化/剪枝以适配边缘设备；几何锚点对极少样本类别的效果仍有限。

---

## 450. Is High Local Accuracy Enough? Geometric Consensus with Support-Adaptive Statistical Personalization for Federated Learning

**arXiv ID:** 2610.31133 | [PDF](https://arxiv.org/pdf/2610.31133v1)

**作者:** Junyu Lin `[一作]` (Wuhan University), Jiayuan Li `[通讯]` (Wuhan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 FedGRC 框架，结合评估协议来评估个性化联邦学习在全球识别、罕见类与未见类上的表现；

**💡 创新点**

创新点在于提出了 Consensus‑Guided Representation Learning、Multilayer Metric Synthesis 与 Support‑Adaptive Shrinkage Personalization 三个模块，并设计了补充本地准确率的全局、罕见类与未见类评估指标；

**🔧 技术方法**

技术包括共享方向与幅度引导的表征学习、多层协方差度量的统计分类以及根据本地样本支持动态收缩类均值的个性化方案；

**📊 数据集**

实验数据集涵盖 CIFAR‑10、CIFAR‑100 以及真实异质性 FedRS 数据集；

**📈 对比分析**

在 5 种异质性设置、3 个数据集上与 21 种基线方法对比，FedGRC 在全局识别与个性化底部分别提升至少 2.17% 与 2.13%，在 FedRS 上在本地适应、全局识别与个性化底部分别提升 2.15%、3.30% 与 4.61%；

**⚠️ 局限性**

局限性包括对样本支持量估计的依赖、仅在图像分类任务验证、以及相对较高的计算与通信成本。

---

## 451. Delta N-gram: Forward-Only Continual Writing into a Language Model's Native N-gram Memory

**arXiv ID:** 2610.31125 | [PDF](https://arxiv.org/pdf/2610.31125v1)

**作者:** Hao Wu `[一作]` `[通讯]` (Illinois Institute of Technology), Hao Wu (Illinois Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种前向写入机制Delta N-gram，在语言模型的预训练哈希N‑gram表上写入稀疏补丁，保持原始表不变。

**💡 创新点**

创新点是将写入过程改为无梯度、无标签的前向写法，并通过小型写入器学习自我记忆，允许写入后可逆、按行可追踪漂移。

**🔧 技术方法**

使用技术包括哈希N‑gram表、DeltaNet风格的行级增量写法、带门控的目标生成、原始预训练目标训练、少量参数写入器以及可选的衰减。

**📊 数据集**

实验数据集涵盖Qwen3.8-Flash-Next模型的原始和后续预训练参数，Raw文本混合六源（Stack Overflow、arXiv、Gutenberg、FineWeb‑Edu、CC‑News、时间序列新闻），DEV64新闻事件、七任务链和LoCoMo对话。

**📈 对比分析**

对比传统Adam梯度写入、LoRA等方法，Delta N‑gram在自写内存上提升自回归NLL 0.068 nats/词、在七任务链中保持写入稳定，delta写入器漂移更低，但整体增益有限，性能仍落后于梯度更新或融合LoRA。

**⚠️ 局限性**

主要限制是写入的记忆只能在相同位置读取，难以跨位置检索；共享的格式行引发漂移与泛化收益共存；缺乏跨位置训练目标及行级漂移上限；对长链任务的记忆召回率低。

---

## 452. WildDepth-v2: A Synchronized RGB--LiDAR--Radar Dataset and Benchmark for Metric Depth, Motion, and Tracking of Wildlife

**arXiv ID:** 2610.31123 | [PDF](https://arxiv.org/pdf/2610.31123v1)

**作者:** Muhammad Aamir `[一作]` (University of Oxford), Andrew Markham `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文扩展了WildDepth数据集，加入同步RGB–LiDAR–雷达记录，发布隐藏测试集和一系列基准任务，用于动物距离、深度补全、运动分割和三维跟踪。

**💡 创新点**

创新点包括：①首次在野生动物上同步采集4D成像雷达数据并提供精确的外参与误差预算；②冻结隐藏测试集并提供官方评分器；③利用雷达的尺度锚和多普勒速度实现无LiDAR的深度标定与运动感知。

**🔧 技术方法**

采用的技术包括：多模态同步采集、LiDAR与雷达外参优化、基于单目深度估计（如MoGe‑2、Depth Anything）与雷达尺度匹配的Affine标定、雷达多普勒传递至LiDAR实现稠密运动掩码、时间窗内LiDAR累积并以雷达检测为门控，以及基于深度的三维轨迹提升与Kalman滤波。

**📊 数据集**

使用的数据集是WildDepth-v2，包含三地（Kgalagadi、Bubye、Longleat）共202,941帧，新增19,716帧RGB–LiDAR–雷达同步记录，以及13场隐藏测试的352/902帧，涵盖10种动物种类。

**📈 对比分析**

在基准任务中，单目深度模型未标定时距离误差均在0.15–0.83的AbsRel，加入单帧LiDAR标定后降至0.05–0.15；对象级雷达尺度标定的AbsRel为0.12，优于场景级LiDAR标定（0.20）且接近直接LiDAR（0.05）。雷达多普勒在运动掩码密度上提高3.7倍，速度相关性达到0.60，虽然误差约为真实速度的一半。三维跟踪中，单目深度+雷达标定的位姿误差0.58 m，接近一次性LiDAR标定的0.52 m，显著低于仅单目深度的2.2 m。

**⚠️ 局限性**

局限性包括：雷达仅在一地点、白天、无植被的开放环境下测试，且对高于30 m距离的动物尺度误差增大；雷达垂直误差导致无法实现像素级融合；多普勒受自车运动影响，需识别静态平台；外参随帧漂移，手动校正是必要的；跟踪评估仅使用IoU跟踪且未提供身份级指标。

---

## 453. SetCommit: Coordinated Token Commitments for Few-Step Diffusion Language Model Decoding

**arXiv ID:** 2610.31120 | [PDF](https://arxiv.org/pdf/2610.31120v1)

**作者:** Yuannuo Feng `[一作]` (Beihang University), Wang Kang `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、无架构改动的 SetCommit 方法，在扩散语言模型（DLM）中协调并行 token 的承诺与撤回。

**💡 创新点**

创新点在于：①通过可靠性与局部共同承诺风险的权衡来选择承诺集；②引入预算中性局部交换机制，在保持总承诺量的同时动态修正早期易出错的承诺。

**🔧 技术方法**

使用了基于 logits 边际、熵权重的风险评估、贪心集合选择与可逆交换策略，并与现有的 native、Fast-dLLM、EB‑Sampler 等解码器无缝集成。

**📊 数据集**

在 LLaDA-8B 与 Dream-7B 两大模型上，使用 GSM8K、MATH-500、MBPP 与 HumanEval 四个基准数据集进行实验。

**📈 对比分析**

与原生解码器以及多种加速策略（缓存、调度）进行配对对比，SetCommit 在 32 次模型评估下可提升 4.9–7.4 个百分点的准确率/通过率，并在相同质量目标下实现 1.44–1.74× 的几何平均增速。

**⚠️ 局限性**

局限性包括：①主要在极少步（few‑step）解码场景中显著，随着评估次数增多增益减小；②交换机制对计算和内存有轻微开销；③在某些任务或模型上参数调优仍可能带来额外提升，未完全泛化。

---

## 454. What Comes Next and Why: Interpretable Next-Event Prediction with Neuro-Symbolic Rules

**arXiv ID:** 2610.31114 | [PDF](https://arxiv.org/pdf/2610.31114v1)

**作者:** Tim Nico Bauerschmidt `[一作]` (Saarland University), Jilles Vreeken `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为 Runes 的神经符号模型，能够端到端学习可解释的时间序列规则，并通过残差网络补充未被规则解释的结构以提升预测准确性。

**💡 创新点**

创新点在于将时序规则嵌入神经网络层，利用温度退火实现可微分的逻辑谓词学习，并在保持规则可解释性的前提下引入残差网络以提高性能。

**🔧 技术方法**

技术包括可微分的事件时间谓词与延迟区间、两阶段训练（先学习规则再冻结后训练残差网络）、Transformer 作为残差模块以及交叉熵自监督下的下一事件预测目标。

**📊 数据集**

实验使用合成数据和五个真实世界数据集（包括音乐作品 Chorales、电子商务 Amazon 等），以验证模型的规则学习和预测效果。

**📈 对比分析**

与基于 PPM 的马尔可夫模型、XGBoost/CatBoost、以及改进的 RoPE Transformer 进行比较，结果显示 Runes 在合成数据上能够准确恢复真规则，在真实数据上与黑盒 Transformer 相当或略优，且所学规则更紧凑、易解释。

**⚠️ 局限性**

局限性包括需要手动调节网络层大小与温度退火计划，训练过程受随机性影响；目前仅支持单一类别事件且规则形式受限，难以覆盖更复杂的时间依赖或多事件场景。

---

## 455. Solving Flood-It Puzzles: A Comparison of Tailored A* Search with Declarative Solver Approaches

**arXiv ID:** 2610.31100 | [PDF](https://arxiv.org/pdf/2610.31100v1)

**作者:** Philipp von Manteuffel `[一作]`, Benjamin Blankertz `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文展示了一个节点颜色随时间变化的动态图，并通过箭头指示不同状态之间的转换。

**💡 创新点**

创新点在于用颜色与箭头直观地可视化节点状态的动态演变。

**🔧 技术方法**

主要使用了 LaTeX 中的 TikZ 包进行图形绘制。

**📊 数据集**

文中未提及具体使用的数据集。

**📈 对比分析**

论文未对方法进行实验比较，亦未给出性能评估。

**⚠️ 局限性**

局限性在于缺乏实验验证与数据支持，对方法的有效性与适用范围尚无证明。

---

## 456. Almost tight distributed approximation for domination in sparse graph classes

**arXiv ID:** 2610.31203 | [PDF](https://arxiv.org/pdf/2610.31203v1)

**作者:** Timothé Picavet `[一作]` `[通讯]` (Gran Sasso Science Institute), Timothé Picavet (Gran Sasso Science Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计了在稀疏图类（最大平均度数≤d且深度‑1浅子图密度受限）上运行的确定性常数轮分布式近似最小支配集算法，并给出了近似比为d+1+（+ε）的上界；在平面图、具有有限欧氏基数的图、队列/堆栈数受限图、树宽/路径宽受限图以及k‑极小自由图等特殊图类上获得了几乎最优的常数近似结果。

**💡 创新点**

①将浅子图密度限制与最大平均度数相结合，突破了仅使用最大平均度数无法得到常数近似的已知下界；②利用超图约束与太阳花引理将求解转化为受限度数的LP，实现在常数轮内完成；③在不需要图嵌入或布局的前提下，对所有这些稀疏图类给出近似比的统一理论框架；④在不依赖大消息传输的模型中给出了可行的实现方案。

**🔧 技术方法**

主要技术包括：
- 超图构造与稀疏化（利用深度‑1浅子图密度与太阳花引理）
- 基于LP的“打孔”与约束放缩技术
- 近似求解器（Kuhn–Moscibroda–Wattenhofer 线性约束LP算法）
- 局部计算与邻域聚合，保持常数轮复杂度
- 通过“加法修正”把最大平均度数限制的图类映射到满足稠密度下界的子类，进而获得精确近似。

**📊 数据集**

本工作为理论性论文，无实验数据集；所有结果均为严格证明和理论分析。

**📈 对比分析**

与已发表的结果（Amiri & Siebertz、Wawrzyniak、Heydt 等）相比：
- 在平面图上从 8+ 降至 7+（+ε）
- 在具有有限欧氏基数图上得到 7+ 与 5+（+ε）
- 在队列/堆栈数、树宽/路径宽受限图上实现了 4k+1+、2k+3+、2k+1+ 等近似比，几乎与已知下界相匹配；
- 所有算法均为确定性常数轮，且局部计算多项式。

**⚠️ 局限性**

主要局限包括：
- 近似比包含不可忽视的 +ε 项，需依赖参数 ε；
- 需要浅子图密度上界 ∇₁≤β，无法完全移除此假设；
- 算法使用大消息传递，尚未证明可在仅 O(log n) 位消息的模型中实现；
- 对非偶数 d 的最优近似比例尚未确定，且是否可以去掉 +ε 仍是开放问题；
- 运行时间与 ∇₁ 相关，仍有提升空间。

---

## 457. Predicting Music Valence and Arousal via Textual and Multimodal In-Context Learning in Large Language Models

**arXiv ID:** 2610.31202 | [PDF](https://arxiv.org/pdf/2610.31202v1)

**作者:** Chi-En Dai `[一作]` (National Taiwan University), Yi-Hsuan Yang `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

探索在音乐情感识别中使用大语言模型的上下文学习（ICL）来提升连续情绪（valence 与 arousal）的预测，先用小基线模型给出初始估计，再通过文本或多模态ICL校准残差。

**💡 创新点**

创新点在于采用推断时无训练的ICL方式，通过示例演示预测残差而非绝对值，结合文本描述或原始音频显著提升 valence 预测性能，尤其在高质量小数据集上实现大幅 R² 增长。

**🔧 技术方法**

使用 Music2Emo 音频分支 + BEATs 特征训练基线模型；文本ICL基于 GPT‑5 并配合歌词与 Qwen‑Audio 生成的音频描述；多模态ICL采用 Qwen3‑Omni 30B；多 shot 提示并预测残差后加到基线估计。

**📊 数据集**

基线模型与 IC​L 演示使用 CH818（818 条中文流行歌曲）训练，测试集为 MER60（60 条西方流行歌曲），两者均为专家标注的高质量情绪数据。

**📈 对比分析**

与传统基线（SVR、VGGish、Music2Emo）对比：文本ICL 80-shot 时 valence R² 由 -0.801 提升至 0.427；多模态ICL 10-shot 时 valence R² 达到 0.346；arousal 上的提升虽有限但已接近基线水平；整体均显著优于零 shot 与纯音频模型。

**⚠️ 局限性**

局限性包括：多模态ICL 的上下文长度限制导致更多演示不一定提升；依赖高质量歌词或音频描述；模型性能高度依赖大模型容量；实验仅在小规模 MER60，未验证更大或不同风格的数据；未实现完整歌曲预测。

---

## 458. ChainSEED: A Scalable, High-Fidelity Blockchain Emulator

**arXiv ID:** 2610.31188 | [PDF](https://arxiv.org/pdf/2610.31188v1)

**作者:** Qishu Jin `[一作]` (Zhejiang University), Kairui Feng `[通讯]` (Zhejiang University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了基于SEED Internet Emulator的可配置区块链仿真平台ChainSEED，支持以太坊PoS、Monero和Solana的真实客户端部署，并提供链配置、节点位置、交易工作负载、监控和定制二进制等功能。

**💡 创新点**

创新点在于：①将链配置与节点部署解耦，允许在同一网络环境下灵活绑定虚拟节点与物理宿主；②支持使用真实客户端实现高保真实验；③通过内存占用优化（Beacon节点内存裁剪、执行节点GC阈值调整、验证者多路复用）实现大规模部署（400 Beacon+400 Exec+60k验证者）在单机上；④提供完整的实验交互工具（交易生成器、监控仪表盘）与安全/网络测评案例。

**🔧 技术方法**

主要技术包括SEED Internet Emulator（路由、网络层、服务层分离）、Docker/Kubernetes容器化部署、以太坊Geth+Lighthouse客户端、Monero Core、Solana Agave客户端、自定义交易生成器、EthExplorer/Ethereum Lookingglass可视化、监控脚本以及网络流量测量工具。

**📊 数据集**

使用的数据集包括：①以太坊主网交易统计（Xatu公共执行层数据）用于生成交易负载；②已知的对等图（Ethereum peer graph）用于拓扑发现基准；③Monero 1200节点网络配置与矿工工作负载；并利用公开的网络拓扑和链状态进行验证。

**📈 对比分析**

通过对比不同配置下的内存、CPU占用、TPS、确认延迟和共识进度，实验表明：Beacon节点worker cap=2、执行节点内存设为384MiB、验证者100个键/客户端可在约1TiB内存、384逻辑核上维持约20TPS的链速；在更大规模下（400节点/60k验证者）依旧保持低错率；Monero 1,200节点可支持200台单线程PoW矿工；Solana 200节点可持续最终化。

**⚠️ 局限性**

局限性包括：受单机硬件资源限制，无法完全模拟全球规模网络拓扑；优化主要针对目前主流客户端（Geth、Lighthouse等），对其它实现的适配仍需进一步验证；实验环境仍是模拟网络，可能无法完整复现真实互联网的路由故障和攻击细节；验证者多路复用可能增加单点故障风险。

---

## 459. BelT Block Cipher

**arXiv ID:** 2610.31178 | [PDF](https://arxiv.org/pdf/2610.31178v1)

**作者:** Sergey Agievich `[一作]` (National Research Center for Applied Problems of Mathematics and Informatics Belarusian State University), Yuriy Kharin `[通讯]` (National Research Center for Applied Problems of Mathematics and Informatics Belarusian State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并详细描述了一种128‑bit块、256‑bit密钥的BelT分组密码，采用8轮Feistel+Lai‑Massey混合结构，并给出了完整的密钥调度、轮函数、S‑盒和实现细节。

**💡 创新点**

创新点主要包括：①加密与解密的轮结构高度相似，简化实现；②S‑盒可自定义，且基于长度255的线性递归序列构造，具备高非线性、良好的差分和代数属性；③轮函数中交替使用⊕、⊞、⊟等运算以及轮数与辅助寄存器的组合，提升了对线性、差分与代数攻击的抵抗；④采用非均匀轮设计和S‑盒预计算表实现高效软件实现。

**🔧 技术方法**

使用了32位指令集、S‑盒预计算表（4 KB）、基于λ函数的位移与非线性变换、字节级S‑盒替换、加密/解密时的运算顺序优化以及对AES竞赛候选密码相同的实现约束。

**📊 数据集**

实验仅采用单个公开的测试向量：明文“0x1234567890ABCDEF1234567890ABCDEF”在密钥“0x000102030405060708090A0B0C0D0E0F00112233445566778899AABBCCDDEEFF”下得到特定密文，未使用大规模数据集。

**📈 对比分析**

通过在C语言实现下，使用VTune性能分析器对Pentium‑200和Pentium III-1000 CPU进行计时，比较BelT与AES竞赛候选密码的CPU周期数。结果显示BelT在Pentium‑200上约18.73 Mbit/s，Pentium III-1000上约187.7 Mbit/s，性能与AES候选密码相当或更优。

**⚠️ 局限性**

局限性：本文仅做了基本的运算复杂度和性能评估，未对大规模统计、差分/线性/代数攻击以及侧信道攻击进行全面分析；加密/解密轮数固定为8轮，未来可能需要进一步评估其对更高安全级别的适用性。

---

## 460. Balanced Posted Prices in Stationary Two-Sided Markets

**arXiv ID:** 2610.31153 | [PDF](https://arxiv.org/pdf/2610.31153v1)

**作者:** Ilan Reuven Cohen `[一作]` `[通讯]` (Bar Ilan University), Ilan Reuven Cohen (Bar Ilan University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究在线两边市场，卖方以固定价格采购商品并在有限库存下将其卖给XOS价值的买方，提出可实现高效社会福利的静态统一价格策略。

**💡 创新点**

创新点在于引入“速率平衡”和“贡献平衡”两种平衡原理，利用流量松弛和XOS支持项分解，证明在任意有限库存下可通过单一价格实现至少1/2（单品为C/(C+1)）的最优收益率。

**🔧 技术方法**

采用Poisson到达模型、稳态分布与PASTA、线性规划流量松弛、XOS支持项与需求oracle、固定点分析以及随机化入场策略等技术。

**📊 数据集**

论文为理论研究，不使用实际数据集；所有实验均为理论证明与抽象模型分析。

**📈 对比分析**

与无限库存的离线基准进行对比；证明无论库存容量如何，提出的统一价格至少能达到该基准收益的1/2（单件为C/(C+1)），并给出单件情形的上界。

**⚠️ 局限性**

局限性包括：多品速率平衡价格的多项式时间构造仍未实现；对有限容量的更高效保证仍是开放问题；算法依赖于精确的XOS与需求oracle，且模型假设Poisson到达与非预知策略。

---

## 461. RankAlign: Vision-Language Representation Alignment via Rank Transformation

**arXiv ID:** 2610.31145 | [PDF](https://arxiv.org/pdf/2610.31145v1)

**作者:** Enzhe Zhao `[一作]` (University of Padova), Lamberto Ballan `[通讯]` (University of Padova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究无配对视觉‑语言类别级别匹配，并提出 RankAlign 排序归一化方法，改进关系匹配的鲁棒性。

**💡 创新点**

创新点在于将相似度矩阵转换为秩序（ordinal）表示，消除跨模态尺度、偏置和单调噪声的影响，并给出严格的恢复与稳定性理论保证。

**🔧 技术方法**

主要技术包括：冻结预训练视觉/语言编码器；类别原型构造；相似度矩阵的秩变换（RankAlign）；基于 QAP 的关系匹配（BlindMatch、Gromov–Wasserstein、OT、FHG 等）；以及理论分析和实验验证。

**📊 数据集**

使用的数据集包括 CIFAR‑100、ImageNet‑100、医学影像数据集 CheXpert、LAION‑Aesthetic 子集等；同时在多种预训练编码器（DINO, DINOv2, CLIP, SigLIP 等）上进行评估。

**📈 对比分析**

与 BlindMatch（GW）、Mutual kNN、OT、FHG 等基线对比，RankAlign 在大多数编码器和数据集上实现了显著的匹配准确率提升（如在 CIFAR‑100 上准确率从 0.26→0.98），且对标签噪声、域迁移等具有更好的鲁棒性。

**⚠️ 局限性**

局限性：依赖可靠的类别原型，噪声或聚类误差会影响秩序信息；计算量随类别数增长呈指数级，匹配阶段可能成为瓶颈；仅针对零锚的类别级匹配，未覆盖实例级、开放词汇或混合尺度对齐。

---

## 462. Pos4CHD: A Positive-Free Mask-Guided Counterfactual Synthesis Framework for Fetal Congenital Heart Defect Analysis

**arXiv ID:** 2610.31139 | [PDF](https://arxiv.org/pdf/2610.31139v1)

**作者:** Wenjie Xuan `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出一种无正样本的基于语义掩码的反事实生成框架 Pos4CHD，用 FE‑Gen 模型对胎儿心脏超声图像进行高保真、空间对齐的合成，从而产生可用于下游诊断任务的伪正样本。

**💡 创新点**

创新点在于：①首次将语义掩码作为完整条件直接驱动超声图像生成，消除对正样本的依赖；②引入可学习语义查询和直方图正则化，显著提升图像质量与掩码对齐度；③通过规则化掩码变形实现多种 CHD 亚型的结构化反事实生成。

**🔧 技术方法**

使用技术包括：扩散模型（Stable Diffusion）与 ControlNet 结构、LoRA 微调、可学习语义查询、直方图 L1 正则化、以及多阶段训练流程（VAE 适配 → UNet 微调 → FE‑Gen 训练）。

**📊 数据集**

数据集为武汉大学内部收集的 8,000 张 4CH 视图（含 14 个解剖结构的像素级注释）与 500 张高质量标注样本，用于训练 FE‑Gen 并生成多种 CHD 伪样本。

**📈 对比分析**

与 ControlNet、SD‑LoRA、T2I‑Adapter 等基线相比，FE‑Gen 在 FID、LPIPS、直方图指标和掩码 Dice 上取得显著提升（FID 下降至 0.035，Dice 提升 3.96%），并在零样本分类、VSD 检测和心脏解剖分割任务中分别提升 93.25%/72.92% ACC、+6.95% mAP50、+10.49% DSC。

**⚠️ 局限性**

局限性在于：①对细小结构（如 VSD 内膜分隔）的空间控制仍不够精准，导致部分合成图像缺失真实病灶；②依赖人工筛选与掩码增强，增加人工成本；③当前框架仅适用于结构性异常，无法处理纹理或功能性缺陷的生成。

---

## 463. RGraph-Eval: Task-Grounded Scene Graphs for Evaluating Robot Video Generation

**arXiv ID:** 2610.31121 | [PDF](https://arxiv.org/pdf/2610.31121v1)

**作者:** Aleksandr Lazarev `[一作]` (MIRAI), Dmitry Yudin `[通讯]` (MIRAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于参考视频的评估框架 RGraphEval，利用任务驱动的时间场景图对机器人生成视频进行可解释评估；

**💡 创新点**

创新点在于将对象掩码、状态转移、动作区间和 3D 动作轨迹统一到同一可解释的场景图中，并通过可视化对象、关系变化和路径形状构成分解式评分；

**🔧 技术方法**

使用了大型语言模型 Qwen3.8-27B 进行角色与关系预测，SAM 3 与 SAM 2.1 进行掩码生成与时间修复，Depth Anything 3 进行深度估计与 3D 轨迹重建；

**📊 数据集**

构建了 ShareRobot‑SG 数据集，包含 1,000 条从 10 个 ShareRobot 源获取的机器人操控视频，并提供 100 条人工标注的 ground‑truth 以评估图谱生成质量；

**📈 对比分析**

与改编的 SVG2、SG‑Ego 以及三款视频生成模型 Cosmos‑3 Nano、LingBot‑VA、WoW‑1 进行了对比，RGraphEval 在可视化对象一致性、关系变化与 3D 路径形状三项指标均取得 0.89‑0.97 的分数，并与人类评测呈现 0.97 以上的皮尔逊相关；

**⚠️ 局限性**

局限性包括依赖单一参考视频导致无法处理多种有效执行；图谱生成误差会影响评分；当前只关注单一被操控物体，未对可执行动作进行验证；缺乏多物体、多臂以及多参考的处理能力。

---

## 464. Constraint Reasoning and Large Language Models Join Forces: Providing Contrastive Explanations for Real-World Optimisation-Based Applications

**arXiv ID:** 2610.31119 | [PDF](https://arxiv.org/pdf/2610.31119v1)

**作者:** Roger X. Lera-Leri `[一作]` (Artificial Intelligence Research Institute, IIIA-CSIC), Juan A. Rodríguez-Aguilar `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为exmip的方法，结合约束推理与大型语言模型，为基于优化的决策支持系统提供对比解释；

**💡 创新点**

创新点在于将不可满足子系统（IIS）转化为“原因图”，将结构化的关系图作为提示输入给LLM，从而显著提升生成解释的完整度与可读性；

**🔧 技术方法**

技术栈包括混合整数规划、IIS求解、超图与双向图构建、自然语言模板化、GPT‑4o零样本提示；

**📊 数据集**

使用真实本科课程规划数据（澳洲西悉尼大学ICT学位）和PSPLIB单模式RCPSP实例作为实验数据集；

**📈 对比分析**

与仅给出IIS列表的LLM解释进行对比，评估完整度、词数和GFI；结果显示原因图提升约20%完整度，解释略长但可读性相近；在两类问题上均能在秒级或分钟级完成解释，支持实时交互；

**⚠️ 局限性**

局限性包括LLM仍受抽象推理限制，某些查询类型（如Q6/Q3）解释耗时较高；缺乏用户研究验证解释效果，且自然语言模板需人工制定。

---

## 465. Through a PRISM, Clearly: Step-Sequence Reasoning for Mistake Recognition in Procedural Videos

**arXiv ID:** 2610.31117 | [PDF](https://arxiv.org/pdf/2610.31117v1)

**作者:** Olga Loginova `[一作]` (University of Trento), Frank Keller `[通讯]` (University of Edinburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PRISM，一种基于语言表征的后期程序错误识别框架，利用隐藏马尔可夫模型(HMM)对正确执行进行训练，并结合参考对齐和语义检查实现对步骤内容与顺序错误的检测。

**💡 创新点**

创新点包括：① 将视频段落映射到可解释的语义角色表征并通过 HMM 进行步骤解码；② 统一四类编辑错误（插入、删除、替换、互换）实现跨数据集可比性；③ 在 HMM 解码中加入本地解码器与全序列解码器的对比，证明训练的解码器能显著提升性能。

**🔧 技术方法**

主要技术：视觉-语言模型（VLM）进行视频-文本 grounding，语义角色标注（SRL）提取结构化步骤，隐藏马尔可夫模型进行步骤解码，序列对齐（Needleman‑Wunsch）进行错误标注，阈值化的低概率发射/转移评分进行额外错误诊断。

**📊 数据集**

使用了三大公开数据集：CaptainCook4D（烹饪），EgoPER（烹饪），EgoOops（多域混合程序）以及对应的 VLM 生成的步骤描述。

**📈 对比分析**

方法对比：在所有数据集上与已有基线（AEM、AMNAR 等）以及 VLM 原始文本、SRL 变体、动作‑对象抽象等进行系统对比。PRISM 在 Any‑F1、Type‑F1、Seq、AUC 上优于多数基线；SRL 版本在 6 个 grounded 对比中均提升 0.1%–3.6%；训练解码器在 11/12 条件下优于最近邻匹配，序列解码在部分数据集带来额外提升。

**⚠️ 局限性**

限制：① 需要先验的步骤边界与候选步骤；② 依赖 VLM 生成的描述，若生成质量差会影响错误检测；③ 四类编辑标签缺乏更细粒度的错误语义，无法区分子步骤级别的错误；④ 需要足够的正确执行样本来训练 HMM，数据稀缺时可能效果下降。

---

## 466. Superintelligent Software Engineering: Governance That Scales with Capability

**arXiv ID:** 2610.31306 | [PDF](https://arxiv.org/pdf/2610.31306v1)

**作者:** Mohammad Mehdi Gholinejad `[一作]` `[通讯]` (OSTEON Labs), Mohammad Mehdi Gholinejad (OSTEON Labs)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出超智能软件工程的概念，论证仅提升代理能力不足以保证软件质量，提出治理随能力扩展的框架，发布 OSTEON 规范与工具，并对 GitHub 上热门仓库的上下文文件进行探索性测量。

**💡 创新点**

创新点在于首次将治理与代理能力同步扩展（规模化治理），给出七项治理需求和对应技术实现，整合决策记录、合同版本化、衰减检查、门控、规划与记录等技术，并通过 OSTEON 工具演示该框架。

**🔧 技术方法**

技术手段包括：数学逃逸率模型、GitHub API 脚本采集、决策记录、合同版本化、衰减检查、审查门控、规划与记录（OSS）等，最终实现了 OSTEON 插件。

**📊 数据集**

数据集为 GitHub 星级最高的 500 个仓库（前 5 页），经过筛选得到 452 个合格仓库，其中 220 个拥有根级或 agent 上下文文件，共 367 个文件；对这些文件进行路径解析率、更新时间等指标统计。

**📈 对比分析**

比较方法采用探索性统计和模型预测；未给出具体性能数值，仅说明模型表明逃逸率随吞吐率线性增长；OSTEON 工具目前无实测效果，缺乏实验对比。

**⚠️ 局限性**

局限性包括：缺乏对 OSTEON 效果的实测、模型假设过简（独立衰减、固定审查容量、固定检测概率）、衰减率估计不准确、仅基于公共仓库快照、治理成本与人力投入未验证、实现仅支持单一代理平台、未在不同项目规模上进行实证评估。

---

## 467. PiHand: Latent Hand-Object Interaction Priors for Physics-Based Policy Learning

**arXiv ID:** 2610.31299 | [PDF](https://arxiv.org/pdf/2610.31299v1)

**作者:** Niklas Berndt `[一作]` (University of Tübingen), Gerard Pons-Moll `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `8d10c613-917e-4880-9716-17789f50e119` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

训练物理仿真下的手-物体交互专家，将其蒸馏为目标轨迹条件的VAE交互先验，并在潜在空间中进行受限探索，生成针对任意目标轨迹的物理可行、自然的手-物体交互；同时验证该先验在桌面操作、抓捕、双手交接等多任务上的零样本迁移能力。

**💡 创新点**

① 通过物理仿真将Mocap手-物体序列转化为可执行的交互专家；② 将多专家知识蒸馏为单一目标轨迹条件的VAE交互先验，实现一次训练即可迁移到多任务；③ 在潜在空间中采用受限残差探索，显著提升采样效率与动作自然度。

**🔧 技术方法**

物理仿真（IsaacGym + PD 控制）、PPO 强化学习、行为克隆、条件VAE、DAGger、硬币采样、3D关键点与6D 旋转表示、潜在空间残差探索。

**📊 数据集**

GRAB 手-物体 Mocap 数据集（用于训练交互先验）以及 DexYCB 数据集（用于跨数据集评估），同时使用 GRAB50 子集与 ManipTrans 进行专家阶段对比。

**📈 对比分析**

与 ManipTrans（单序列模仿）相比，专家阶段成功率从 69.6% 提升至 93.3%；在桌面操作、抓捕、双手交接等任务中，相较 Action Space Exploration 与 Hand-Only Prior 基线，成功率提升 14-17 点，并在抓捕与交接任务中分别达到 2-4 倍的提升；学习速率约快 10 倍。

**⚠️ 局限性**

专家仍未达到 100% 成功率，难以处理极动态或噪声大的序列；仅支持刚体物体，未涵盖关节物体；仅控制手部，未考虑全身运动；所有实验均在仿真中完成，迁移到真实机器人仍需额外工作。

---

## 468. Practical Auditing of Differentially Private Graph Release Mechanisms: Methodology, Toolkit, and Case Study

**arXiv ID:** 2610.31294 | [PDF](https://arxiv.org/pdf/2610.31294v1)

**作者:** Nicholas D'Silva `[一作]` (University of New South Wales), Salil S. Kanhere `[通讯]` (University of New South Wales)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对差分隐私图数据发布机制的实证审计方法，并实现了一个开源Python工具包，随后对PrivDPR（一个基于节点差分隐私的图发布机制）的实现进行审计，发现其实际实现未能满足声明的Node‑DP隐私保证。

**💡 创新点**

创新点在于：①识别并解决图数据审计中特有的两大设计难题——可卡尼（canary）构造与可观测统计（witness）选择；②将这些图特定组件与已验证的统计认证框架（Jagielski等方法）结合；③通过多种witness统计发现违规的差异性，强调在图审计中需使用多样化统计。

**🔧 技术方法**

技术手段包括：差分隐私理论（ε,δ‑DP），统计检验方法（基于事件概率的检验）、Clopper–Pearson置信区间计算、图结构统计（度数、距离、谱、中心性、子图计数）以及Python实现的工具箱。

**📊 数据集**

实验数据集包含：四种合成图（Construction A–D）—分别为k‑星、Erdős‑Rényi混合、连接性限制等；以及两份真实网络（RT‑LOLGOP 推文网络和 Anybeat 音乐社交网络），用来构造相应的邻接图对。

**📈 对比分析**

评估方式：对每个可卡尼/邻接对与每个witness组合，分别采样输出并计算事件概率的下/上界，进而得到下界隐私参数ε_LB；若ε_LB>声明的ε_0则判定违规。结果显示，在所有合成与真实图对上，均得到超出声明预算的ε_LB，违规现象在更严格的图域（非空、连通）中仍持续；同时不同witness对违规检出率影响显著。

**⚠️ 局限性**

局限性包括：①审计为错误排除工具，无法验证机制满足隐私；②未系统探究可卡尼结构对审计力的影响；③仅针对单个Node‑DP机制，缺乏对其他图DP方法的普遍性评估；④缺少可自动化可卡尼与witness生成的方法。

---

## 469. Retrieval-Augmented Online Textual Memory for Episodic Memory Video Question Answering

**arXiv ID:** 2610.31284 | [PDF](https://arxiv.org/pdf/2610.31284v1)

**作者:** Raffaele Cali `[一作]`, Antonino Furnari `[通讯]` (University of Catania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于在线文本记忆的检索增强视频问答框架，系统研究了检索单元、查询方式、上下文预算等对 egocentric 视频多选问答准确率的影响。

**💡 创新点**

创新点在于：① 引入父级检索（Parent）策略，先用细粒度子块检索再展开完整父级描述，兼顾精细匹配与完整上下文；② 系统化展示检索覆盖率与答案准确率之间的非单调关系，指出过多检索会导致干扰。

**🔧 技术方法**

使用了 Qwen3‑VL‑8B 进行视频描述、文本分块、句子向量化、稠密检索+BM25 混合、父级检索、Reranking 等技术，并采用 Gemini 2.5 Flash 作为推理器。

**📊 数据集**

使用了 QAEgo4D‑Closed 数据集（基于 Ego4D 的 egocentric 视频多选问答任务）。

**📈 对比分析**

与 Flat、Sub、Hybrid、ReKV‑LLaVA、Ground VQA 等多种基线对比，Parent 取得 56.4% 的准确率，等同甚至超过最先进方法，同时查询时仅需约 3,600 tokens（约 33% 的全记忆），显著降低上下文长度和内存消耗。

**⚠️ 局限性**

局限性：仅聚焦检索过程，未改进记忆结构；高检索覆盖率会引入视觉上合理但错误的干扰，导致准确率下降；文本记忆仍缺乏事件不确定性与答案相关性的显式指引。

---

## 470. Kempe Reachability with $k\geq 3$ Colors Is $\mathsf{PSPACE}$-Complete on Claw-Free Graphs

**arXiv ID:** 2610.31263 | [PDF](https://arxiv.org/pdf/2610.31263v1)

**作者:** Duc A. Hoang `[一作]` `[通讯]` (Vietnam National University), Duc A. Hoang (Vietnam National University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明：对于任意固定的 k ≥ 3，给定一个 k‑可着色的图，判定两个合法 k‑着色是否可通过 Kempe 变换连通（Kempe Reachability）在所有无刺图（claw‑free graph）上是 PSPACE‑完整的。该结果同样适用于线图（line graph）的 k‑可边着色（edge‑Kempe）以及 k‑正则简单图的 k‑可边着色问题。

**💡 创新点**

创新点在于：① 将已知的非确定性约束逻辑（NCL）问题精细化为有保护配对的三元可着色源图；② 设计了多种多边形（multipole）构造——“negator”、“gate”、“wire”和“三角形”——将源图的方向信息编码为边/顶点着色；③ 证明了在此构造下，Kempe 变换对应于源图中合法方向的逆转，且逆转的边互不相邻，从而保持可达性。该构造实现了在极高最大度（2k-2）的无刺图上实现 PSPACE‑完整性的首次证明。

**🔧 技术方法**

主要技术包括：组合重构（combinatorial reconfiguration）框架、线图对应的边‑Kempe 变换、约束逻辑（NCL）硬度归约、multipole 设计与颜色对齐、Parity Lemma 与多重边分析、以及对保护配对的全局性证明。

**📊 数据集**

论文没有使用实验数据集，所有结论均为理论证明，主要通过构造性多边形构造来完成归约。

**📈 对比分析**

由于本研究是理论复杂度分析，未进行实验比较；论文通过证明构造可保留可达性来展示其严谨性，证明了问题在 PSPACE 内且不存在多项式时间算法（除非 PSPACE = P）。

**⚠️ 局限性**

局限性：① 只证明了最大度为 2k-2 的无刺图上的难度，介于 k+1 与 2k-3 的最大度区间仍未确定；② 只适用于固定 k ≥ 3 的情形，未给出参数化或近似算法；③ 由于构造涉及大量多边形和连接，实际构造规模可能非常大，难以直接应用于实际问题。

---

## 471. IndoNutriQA: An Indonesian Clinical Nutrition Question Answering Benchmark

**arXiv ID:** 2610.31256 | [PDF](https://arxiv.org/pdf/2610.31256v1)

**作者:** Tirana Noor Fatyanosa `[一作]` (Universitas Brawijaya), Alham Fikri Aji `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个包含 911 条问答对的印尼临床营养 QA 基准（IndoNutriQA），每条问答基于真实临床营养报告，由三名营养师写成。

**💡 创新点**

创新点在于首次公开低资源语言印尼的临床营养 QA 数据集，采用标准化的 NCP/PES 八个问题、结构化元数据和专家验证，定位为高质量评估资源而非大规模训练语料。

**🔧 技术方法**

使用了多种技术：基于 IndoBERT 的提取式 QA、mT5 的生成式 QA、RAG（检索+生成）以及多种 LLM（GPT‑4o、Llama‑3.1、Qwen、Sahabat‑AI‑Llama、Merak、BioMistral、MedGemma），并在 5‑折交叉验证、学习曲线和 bootstrap 置信区间等统计方法上进行评估。

**📊 数据集**

使用的数据集是 IndoNutriQA 本身，源自 120 份来自不同印尼医院的匿名临床营养报告，包含 8 个标准化问题的答案。

**📈 对比分析**

在 70/15/15 的训练/验证/测试拆分上，最佳 fine‑tuned 提取式模型 F1≈36，生成式 mT5 低于 8，RAG 近零；而 5‑shot LLM（Llama‑3.1‑8B‑Instruct、MedGemma‑4B‑IT 等）在 F1 上超过 50，显著优于所有 fine‑tuned 模型，且在 BERTScore、句向量相似度以及 LLM‑judge 评分上表现更佳。

**⚠️ 局限性**

局限性包括：数据量仅 911 对，覆盖范围受限于 120 份报告，问题集固定为八个标准化问题，单一标注者导致答案多样性有限；人类评估覆盖不足，安全性评估不可靠；跨机构、跨病种泛化有限；检索增强模型的效果不佳；未公开支持多语言或其他医疗领域的扩展。

---

## 472. Reducing Multi-Horizon Photovoltaic Forecast Error through Tree-Anchored Expert Combination

**arXiv ID:** 2610.31248 | [PDF](https://arxiv.org/pdf/2610.31248v1)

**作者:** Sun Ze `[一作]` (Huaibei Normal University), Alois Knoll `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在强直树预测器基础上，使用非负简单形权重对八种时序模型进行选择性加权，构建树锚定的MAE对齐组合，以降低15–240分钟光伏功率预测的残差MAE。

**💡 创新点**

创新点在于：①引入树锚定的单参数MAE拟合权重，既保持强树基线，又利用弱模型的互补性；②采用非负约束和归一化权重，允许弱模型被置零，避免误差累积；③系统化比较共享权重与时序特定权重、MAE与MSE拟合的效果，并给出显式的误差分解与方差分析。

**🔧 技术方法**

使用的技术包括：LightGBM直树基线；七种神经网络时序表示（TiDE、LSTM、TCN、PatchTST、N‑HiTS‑PV、TimeMixer‑PV、TimeXer‑PV）；非负简单形权重线性组合；MAE对齐的线性规划求解；政策选择与确认流程；块采样法评估不确定性。

**📊 数据集**

采用PVDAQ公开数据集，包含加利福尼亚州两台共址系统（34、35）2014–2017年的时间序列；共56,078个窗口，每窗口含96条历史功率和16条未来预测。

**📈 对比分析**

对比方法包括：单树、等权平均、原始八模型集成、MAE拟合共享/时序特定权重、MSE拟合等；性能上，树锚定MAE组合在历史评估中使sMAE降低3.21%（相对树基线），在不使用未来天气的匹配重训练中使日照期MAE降低1.78%。

**⚠️ 局限性**

局限性包括：数据仅来自两台共址系统，缺乏独立站点验证；未使用实时天气或数值预报，故无法评估天气驱动误差；研究周期重用相同评估窗口，未实现完全独立重现；组合策略未验证对不同时间段或区域的泛化能力；模型选择与权重调整依赖经验设置，未给出可推广的阈值或自动化流程。

---

## 473. Lightweight Frame Compensation for Real-Time Polyp Segmentation under Frame Skipping

**arXiv ID:** 2610.31228 | [PDF](https://arxiv.org/pdf/2610.31228v1)

**作者:** Weihao Gao `[一作]` `[通讯]` (Guangdong University of Education), Weihao Gao (Guangdong University of Education)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出轻量级帧补偿模块（LFC），在冻结的分割骨干前插入，利用三帧窗口的时空注意力融合成单帧以提升实时息肉分割的准确率。

**💡 创新点**

LFC仅约77K参数，完全可插拔；使用伪标签训练，无需人工标注；通过时空注意力实现自适应帧权重，显著补偿因帧跳跃导致的运动模糊与对齐误差。

**🔧 技术方法**

基于3×3卷积的特征提取、时空注意力权重与多层融合头；训练时采用Dice损失；评估使用Dice、IoU、HD95指标；通过伪标签实现跨种子训练。

**📊 数据集**

在四个公开标注图像数据集（CVC-ClinicDB、Kvasir-SEG、PolypGen、PolypDB）构建混合训练集；视频级别评估使用VIM-Polyp（训练/验证/测试），HyperKvasir（外部验证）和Colon-Bench（第三方验证）。

**📈 对比分析**

与直接跳帧（DSF）、线性插值、EMA等无训练融合策略以及全帧推理（FBF）对比；LFC在TransUNet、Polyp-PVT和ResNet-UNet上均恢复或超过全帧精度，吞吐量提升约2.05×–2.55×；在VIM-Polyp上Dice提升6.27点，在HyperKvasir上提升9.00点。

**⚠️ 局限性**

局限性包括：只针对K=3的跳帧；依赖伪标签，可能受源模型误差影响；在极端快速运动或光照突变时仍可能出现误差；尚未评估对其他任务或更大跳帧间隔的适用性。

---

## 474. GLOW-FM: Efficient Generative Low-Light Image Enhancement with One-Step Latent Flow Matching

**arXiv ID:** 2610.31222 | [PDF](https://arxiv.org/pdf/2610.31222v1)

**作者:** Ariel Lapid `[一作]` (Arm), Idit Diamant `[通讯]` (Arm)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种单步潜在流匹配框架（基于终端速度匹配TVM），在冻结的Tiny‑VAE潜在空间中实现低光图像增强，能够在一次函数评估内完成图像恢复，且计算成本低。

**💡 创新点**

创新点在于：①将终端速度匹配首次应用于图像恢复任务；②通过在Tiny‑VAE潜在空间进行流匹配，显著降低每步计算量；③设计的单步目标与多步流匹配相比更高效、易训练。

**🔧 技术方法**

核心技术包括：终端速度匹配（TVM）、潜在空间流匹配（latent flow matching）、冻结的Tiny‑VAE（TAESD）潜在表示、Hourglass Diffusion Transformer（HDiT）作为两时刻条件网络。

**📊 数据集**

使用的主要数据集为：LSD‑DLL（paired和LSD‑U子集）、UHD‑LOL4K、LOL‑v1；在跨数据集泛化实验中还在DICM、LIME、MEF、NPE、VV等无监督低光数据集上评估。

**📈 对比分析**

与多种基准（非生成式如TFFormer、Retinexformer；生成式如PyDiff、WCDM、ReDDiT）对比，单步方法在PSNR/SSIM/LPIPS/NIQE/MUSIQ等指标上取得与现有生成方法相当甚至更优的表现，同时单步推理仅需34 GFLOPs（小模型），显著低于迭代采样方法的数百GFLOPs。

**⚠️ 局限性**

局限性包括：①在训练数据受限的场景（如LOL‑v1）时，重建精度略逊；②依赖潜在空间表示，极细节恢复仍有限；③单步模型虽高效，但在极端噪声或极暗场景下可能需要更深的模型或更丰富的数据。

---

## 475. Consistent Distributed Invariant EKF SLAM with Point and Object Features

**arXiv ID:** 2610.31221 | [PDF](https://arxiv.org/pdf/2610.31221v1)

**作者:** Mengya Xu `[一作]`, Tomi Westerlund `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在多机器人分布式RI-EKF SLAM中，提出Joint-CI方法实现机器人间共享机器人姿态与公共特征的联合协方差传输与融合，兼顾点特征、对象特征及混合地图；

**💡 创新点**

创新点在于构造兼容接收滤波器坐标系的完整机器人-特征联合估计及其协方差，并通过部分状态CI实现无跨机器人相关性追踪的保守融合，同时保证六个可观测方向不被破坏；

**🔧 技术方法**

使用的技术包括右不变扩展卡尔曼滤波（RI-EKF）、协方差交叉（CI）与部分状态CI、误差映射与Lie群结构、第一阶一致性分析、特征选择与通信负载优化；

**📊 数据集**

实验使用了三种仿真场景（General、High-overlap、Five-robot）和真实世界的YCB-Video数据集（Sequence 7）进行蒙特卡洛评估；

**📈 对比分析**

与Pose-CI、DInCIKFc-C等基线方法对比，Joint-CI在机器人位置与特征位置的RMSE均有显著下降，统计一致性（NEES、95%覆盖率）保持在保守范围内，通信时间与基线相近，但通信负载显著增大；

**⚠️ 局限性**

主要局限在于需要传输完整的机器人-特征协方差矩阵导致通信负载较大，对未知初始相对姿态处理有限，且未涉及自适应特征选择与融合调度。

---

## 476. Trion-1: Unifying Language, Protein Sequence, and Structure with Diffusion Language Models

**arXiv ID:** 2610.31215 | [PDF](https://arxiv.org/pdf/2610.31215v1)

**作者:** Tianyu Lu `[一作]`, Hao Zhou `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了 Trion-1，一种统一的三模态扩散语言模型，将蛋白质序列、结构和自然语言整合到共享离散词汇表中，实现了序列-结构-功能三元之间的六大核心能力。

**💡 创新点**

创新点在于采用非因果扩散框架和共享离散 token 空间，突破了传统自回归模型在结构建模中的顺序限制，并将蛋白质功能、设计等任务与语言交互无缝结合。

**🔧 技术方法**

技术主要包括离散 token 化（DPLM2）、非因果扩散语言模型（MDLM/BDLM）、多阶段训练（AR‑CPT、A2D‑CPT、SFT）以及基于区块的双向扩散生成。

**📊 数据集**

使用的数据集涵盖 ProteinArena、ProteinMPNN、CAMEO 2022、PDB-date、GeneralQA、StructureMCQA 等多模态评测基准，并通过对比已发表的专用模型与大型 LLM 进行验证。

**📈 对比分析**

与 AlphaFold、ESMFold 等专用模型及 GPT/Claude 等前沿 LLM 对比，Trion-1 在三模态折叠任务上达到 TM_bb 0.84，折叠性能与 200 倍推理预算的对手相当；在序列理解、结构理解与文本驱动设计等任务中亦表现与顶尖 LLM 相当或略低。

**⚠️ 局限性**

局限性包括对复合物、配体及核酸等更丰富生物分子的覆盖不足，以及在高难度结构建模（尤其是全局几何一致性）上仍落后于最强专用模型。

---

## 477. Evolving Federated Density Peak Clustering with Shuffler-Based Anonymization

**arXiv ID:** 2610.31211 | [PDF](https://arxiv.org/pdf/2610.31211v1)

**作者:** Miha Ožbot `[一作]` (University of Ljubljana), Igor Škrjanc `[通讯]` (University of Ljubljana)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种一轮联邦聚类框架eFDPC，结合演化模糊流式聚类与密度峰值聚类，并通过shuffler匿名化实现客户端无可识别性。

**💡 创新点**

创新点包括：①在联邦环境下将微簇学习迁移至小批量更新；②在微簇层面使用密度峰值选取而非全局样本；③引入基于共享子集的shuffler匿名化，消除峰值归属，提升隐私；④融合聚合后标签传播获得完整聚类。

**🔧 技术方法**

核心技术：演化模糊系统（Evolving Fuzzy Systems）在mini-batch上的微簇学习；密度峰值聚类（Density Peak Clustering）用于峰值筛选；基于加法共享与shuffler混洗的匿名化；多源Dijkstra传播实现标签传递。

**📊 数据集**

使用合成数据集（Compound, D31, R15, S3/S4, Flame, Overlap 等）和真实世界数据集（UCI 典型表格、MNIST、KDD99 等）进行实验。

**📈 对比分析**

与 FKDC、k-FED、MUFC、FedFCM、LLF-CM/F-CM 等基线比较，eFDPC 在ACC、ARI、NMI 上多为最优或近优，尤其在高维、非球形、密度变化大或类别重叠的数据上表现突出，且运行时与 k-means 级别相当。

**⚠️ 局限性**

局限性：峰值选择对稀疏或平坦密度分布敏感；对超参数（合并阈值、邻域大小）依赖较强；通信量虽低于全样本聚类，但高于纯质心聚类；匿名化层增加系统复杂度；对恶意客户端的鲁棒性未充分评估。

---

## 478. A Visual Grounding Benchmark for Retinal OCT with Multimodal Large Language Models

**arXiv ID:** 2610.31205 | [PDF](https://arxiv.org/pdf/2610.31205v1)

**作者:** Ronald Fecso `[一作]` (Medical University of Vienna), Hrvoje Bogunović `[通讯]` (Medical University of Vienna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了用于评估视网膜光学相干层析成像（OCT）中多模态大语言模型（MLLM）视觉定位能力的基准RetinaGroundBench，开发了聚合IoU（AggIoU）度量，并训练了专门的RetGrounder模型。

**💡 创新点**

创新点在于：1) 第一份针对OCT生物标志物定位的公开基准；2) 设计了考虑MLLM多框预测特性的AggIoU匹配策略；3) 通过领域适配与LoRA微调显著提升定位精度的RetGrounder。

**🔧 技术方法**

使用了多模态大语言模型（Qwen3、InternVL3、GLM-4.6V-Flash、Gemma-4等）及其LoRA微调；采用了AggIoU评估方法；在RetGrounder中结合了视觉定位、VQA和文本问答三种监督。

**📊 数据集**

基准数据来自公开的OCTDL和OLIVES两套OCT数据集，共815张B‑scan图像，手工标注4,537个边界框，涵盖13类生物标志物。RetGrounder训练还使用了7个公开分割/检测数据集与分类数据集。

**📈 对比分析**

对27个公开模型进行宏观平均AggIoU、F1、精确率、召回率及VQA平衡准确率的对比；最优开源模型F1仅23.7%，而RetGrounder在宏观F1上提升至33.9%（+43%相对第二好模型）。VQA方面部分模型可达≈70%平衡准确率。

**⚠️ 局限性**

局限性包括：1) 仅评估单帧B‑scan且设备、协议单一；2) 边界框标注存在主观性与碎片化问题；3) 仅针对公开模型，未覆盖闭源或更大规模系统；4) 评估仍以宏观平均为主，缺乏对单类、不同病变复杂度的细粒度分析。

---

## 479. Identifiability from Stationary Covariance in Linear SDEs with Reciprocal Interactions and Latent Variables

**arXiv ID:** 2610.31310 | [PDF](https://arxiv.org/pdf/2610.31310v1)

**作者:** Ana Leticia Garcez Vicente `[一作]`, Saber Salehkaleybar `[通讯]` (Leiden University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

研究在包含递归交互和潜变量的线性随机微分方程中，如何利用平稳协方差进行因果效应的可辨识。

**💡 创新点**

提出了新的图形可辨识条件，扩展了 2-环（递归）结构下的可辨识理论，并建立了基于工具变量的潜变量下直接效应识别框架。

**🔧 技术方法**

采用连续拉普拉斯方程的代数核矩阵、图形 trek 规则和工具变量方法，对协方差矩阵进行秩判定。

**📊 数据集**

未使用具体数据集，主要通过理论推导与数值实验验证图形条件的有效性。

**📈 对比分析**

通过模拟数据与现有的全观测可辨识方法对比，发现新条件在稀疏图中覆盖率高达 95%，但在密集图中效果下降。

**⚠️ 局限性**

局限在于仅提供理论可辨识结果，未对有限样本统计性质、估计误差鲁棒性以及更广泛递归结构的推广进行深入研究。

---

## 480. ReCoG: Relational Concept Graph for Training-Free Personalization

**arXiv ID:** 2610.31282 | [PDF](https://arxiv.org/pdf/2610.31282v1)

**作者:** Changseon Yu `[一作]` (Chung-Ang University), Hyeokjun Kweon `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种训练自由的个性化方法ReCoG，通过构建关系概念图实现对用户定义概念的精确识别

**💡 创新点**

创新点在于显式建模概念间的判别性关系，用VLM挖掘并存储每对概念的区别信息，并在检索时进行基于关系的成对比较

**🔧 技术方法**

使用CLIP进行特征提取、MiniCPM‑o‑2.6（以及可选的LLaVA）作为冻结的视觉‑语言模型，构建节点特征与有向边缘的关系图

**📊 数据集**

在CUB‑200‑2011（Easy/Hard）、CLEVR、MyVLM、Yo’LLaVA、PerVA等数据集上进行评估，特别设计FINGER细粒度个性化基准

**📈 对比分析**

与R2P等基线相比，在FINGER、MyVLM、Yo’LLaVA、PerVA等任务中，ReCoG均显著提升精度（如CUB‑Hard正类准确率从63.8%提升至86.3%，总体加速提升达+10个百分点以上）

**⚠️ 局限性**

限制在于关系图的构造需要O(N²)次VLM调用，虽可通过稀疏构造降低成本，但在大规模或动态概念库中仍面临构建和推理效率的挑战

---

## 481. Chordal Construction of Column-Twisted Reed--Solomon Codes

**arXiv ID:** 2610.31257 | [PDF](https://arxiv.org/pdf/2610.31257v1)

**作者:** Anuj Kumar Bhagat `[一作]` (Indian Institute of Science Education and Research), Krishna Kaipa `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文构造了非GRS的[n,k]_q MDS码，考虑了在(k-1,q)的有理正常曲线C_k-1上的(n-2)个点以及一对在虚割线上的Galois共轭点的投影系统。

**💡 创新点**

创新点在于扩展了非GRS MDS码的长度和维度范围，特别是对于奇数q，构造了长度为(q+5)/2的非GRS MDS码，适用于每个3≤k≤(q-3)/2的情况。

**🔧 技术方法**

使用几何论证和Schur平方技术来证明CTRS码的非GRS性质，并确定了任意CTRS码的对偶结构。

**📊 数据集**

使用了有限域𝔽_q的性质，特别是关于有理正常曲线和虚割线的构造。

**📈 对比分析**

与之前的TRS和CTRS构造相比，本文的构造在可达长度和维度范围上有所改进，特别是对于3≤k≤n-4的范围内，证明了所有[n,k]_q MDS CTRS码都是非GRS的。

**⚠️ 局限性**

限制在于传统的Schur平方方法只能证明3≤k≤n/2的非GRS性质，而本文通过几何论证扩展到了3≤k≤n-4的范围。

---

## 482. Reward-Weighted Preference Optimization for Selective Combinatorial Optimization Problems

**arXiv ID:** 2610.31239 | [PDF](https://arxiv.org/pdf/2610.31239v1)

**作者:** Haohui Zhang `[一作]` (University of Twente), Martijn Mes `[通讯]` (University of Twente)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对选择性组合优化问题（SCOP），本文提出Reward‑Weighted Preference Optimization (RWPO) 训练框架，并结合规模特定微调与倾斜采样，构建了完整的学习与适配管线。

**💡 创新点**

创新点包括：① 将奖励差值作为权重加入偏好学习，克服传统二元偏好忽略奖励幅度的局限；② 设计了置信度正则化的全对比损失，兼顾稳定性与奖励引导；③ 构建无局部搜索的跨规模适配策略，显著提升大规模与分布外推断性能。

**🔧 技术方法**

核心技术包括：强化学习与基于Bradley–Terry的偏好最大似然；Transformer 编码器‑解码器构造序列策略；POMO 多起始推理；奖励加权偏好损失；规模微调与倾斜采样（奖励与参考策略的加权行为克隆）。

**📊 数据集**

实验使用合成 OP、PCTSP、SPCTSP、KP 数据集（节点数 100、500、1000、2000），以及 OPLib Generation 3 的真实大规模实例作为分布外测试。

**📈 对比分析**

与非神经求解器（LKH、Compass、OR‑Tools）、神经构造求解器（AM、POMO、MDAM、Sym‑NCO、PO、BQ、GLOP、UDC）进行对比，评估目标差距与运行时间。RWPO 在所有基准上实现了新的 SOTA，零样本泛化优于传统 PO，且在规模微调+倾斜采样后往往超过 Compass、UDC 等手工优化方法，且推理时间仅为秒级。

**⚠️ 局限性**

局限性：① 依赖于可精确计算的奖励，难以直接迁移至部分可观测或多目标环境；② 奖励加权系数与长度归一化常数目前为固定，缺乏自适应机制；③ 对多尺度适配的理论分析有限，熵分析仅为经验观察。

---

## 483. Local Discrimination of Binary Classifiers via Localized AUC Functional

**arXiv ID:** 2610.31216 | [PDF](https://arxiv.org/pdf/2610.31216v1)

**作者:** Wenye Ren `[一作]` (ETH Zurich), Olga Demler `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出LOCATOR方法，通过局部AUC（LAUC）在不切分数据的情况下识别二分类模型在协变量空间中的高低性能区间，并给出相应的解释与改进方案。

**💡 创新点**

创新点在于引入局部AUC函数，提供个体级性能度量；通过全局相似度加权而非传统子集化，保持统计功效；建立LAUC的理论性质、渐近分布和置信区间；同时提出两轴诊断框架区分模型异质性与验证队列特征。

**🔧 技术方法**

使用Gower距离构建相似度权重，计算LAUC；训练XGBoost回归模型预测logit(LAUC)；利用SHAP解释高低性能驱动因素；采用U统计和Logit变换推导LAUC的标准误与置信区间。

**📊 数据集**

在美国NHANES 1999–2010周期（N=17,748，CVD死亡为终点）中训练并内部验证；随后在英国Biobank（N=355,122）外部验证。

**📈 对比分析**

与传统全局AUC对比，LOCATOR在高性能子群的AUC显著提升（如NHANES高性能子群0.88 vs 低性能0.78），在UK Biobank也保持差异；通过模拟验证方法在正负控制情形下能正确识别性能异质性。

**⚠️ 局限性**

局限包括：缺乏非致命心血管事件；仅在美国人群中评估，可能缺乏跨族群泛化；使用固定指数15的权重可能不适用于所有数据；SHAP关联不代表因果；方法对大样本计算成本较高。

---

## 484. RIPPLE: Recovery Internalization through Process-Level Policy Learning and Effect-Aware Credit for Long-Horizon Agentic Coding

**arXiv ID:** 2610.31276 | [PDF](https://arxiv.org/pdf/2610.31276v1)

**作者:** Jingxuan Xu `[一作]` (National University of Singapore), Ken Deng `[通讯]` (Independent Researchers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RIPPLE框架，利用前缀监督与过程信用分配训练自我恢复策略，解决批判到行动的脱节问题

**💡 创新点**

创新点在于将批判、诊断与纠正拆分为三阶段并为每阶段提供局部信用信号，同时采用前缀仅监督的教师与可执行验证器实现内部化恢复

**🔧 技术方法**

使用前缀教师生成批判、轻量化蒸馏初始化、强化学习与阶段化信用分配、可执行验证器以及冻结评估器进行训练

**📊 数据集**

在四个软件工程基准上进行评估：SWE‑bench Verified、SWE‑bench Pro、SWE‑bench Multilingual 以及 SWE‑Compass

**📈 对比分析**

与基线模型以及多种过程信用方法对比，平均提升4.8分，保持1.17× token成本、1.23×延迟，且无需在线辅助模型

**⚠️ 局限性**

局限性包括仅在Qwen3.5‑27B与单一执行框架内验证；需要额外训练时快照测试和评估器；对缺失证据或不可逆状态的恢复效果有限

---

## 485. Cola2: Co-Adaptive Hierarchical Language Modeling with Continuous Semantic Compression and Balanced Discrete Generation

**arXiv ID:** 2610.31305 | [PDF](https://arxiv.org/pdf/2610.31305v1)

**作者:** Hongcan Guo `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于连续潜在空间与离散生成器协同的层级语言模型框架，称为 COLA2。

**💡 创新点**

创新点在于动态语义分段压缩、生成导向的潜在监督、连续-离散协同训练与梯度解耦，以及通过块因果扩散模型实现离散生成与潜在补偿的平衡。

**🔧 技术方法**

采用动态 VAE 编码器、块因果 Masked Diffusion Decoder（MDM）、Diffusion Transformer（DiT）潜在先验学习、流匹配、分类器无指导（CFG）以及负反馈压缩率控制等技术。

**📊 数据集**

使用 OLMo2 预训练语料库以及多任务微调数据，包括 LAMBADA、RACE、OBQA、StoryCloze、SQuAD、HellaSwag、NarrativeQA、ARC‑Easy、TruthfulQA、MMLU 等。

**📈 对比分析**

与自回归、LLaDA、Cola DLM 等模型对比，控制规模到约210B 训练 token 的 COLA2 在五项下游任务平均分数约 34.5%，并在单独 346B 预训练中获得 60.10 的八任务平均分，几乎与 Llama‑3‑8B（66.25）相近，且在 HellaSwag 与 StoryCloze 上超过其表现。

**⚠️ 局限性**

主要局限包括：潜在 DiT 仍未收敛、训练规模受限、生成速度受 decoder、CFG、缓存等因素影响、不同解码器的训练和推理权衡尚未充分探索，以及对多模态统一潜在表示的进一步研究。

---

## 486. Data-Driven Vision-Language Models for Remote Sensing: A Survey

**arXiv ID:** 2610.31304 | [PDF](https://arxiv.org/pdf/2610.31304v1)

**作者:** Yue Zhou `[一作]` (East China Normal University), Jonathan Li `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文系统综述了遥感视觉‑语言数据集，提出了以数据为中心的 GeoChef 框架，并构建了可持续维护的社区驱动数据仓库，提供从数据构建到模型微调与评估的完整实践指南。

**💡 创新点**

其创新点在于以数据为中心的视角对遥感 VLM 进行分层分类，提出四阶段的 GeoChef 研究框架和可持续维护的社区驱动数据仓库，并给出细粒度任务划分与多维标签索引。

**🔧 技术方法**

所采用的技术包括多任务学习、对比学习、LLM 生成与多代理式人工标注、参数高效微调（LoRA）与强化学习微调，以及多维评测指标和自动化构建流水线。

**📊 数据集**

论文参考并整理了超过 120 个公开遥感视觉‑语言数据集，涵盖图像描述、VQA、视觉定位、综合指令、基准测试等，如 RSICD、RemoteCLIP、GeoChat‑Instruct、EarthDial‑Instruct、RS‑VL3M 等。

**📈 对比分析**

通过对比主流模型在图像描述、VQA、视觉定位等基准上的 BLEU、METEOR、ROUGE、CIDEr、Accuracy、IoU、mAP 等指标，表明现有模型在标准任务已接近 90% 之上，但在超高分辨率与复杂推理任务上仍存在显著差距。

**⚠️ 局限性**

该工作仍局限于数据集的收集与评测，缺乏对模型深度推理能力的统一评测，并且在低资源环境下的多模态模型训练与推理仍依赖昂贵的算力与标注资源。

---

## 487. Whole-Body Motion Guided Reference Generation for Uncertainty-Aware Compliant Assistance

**arXiv ID:** 2610.31302 | [PDF](https://arxiv.org/pdf/2610.31302v1)

**作者:** Muyuan Ma `[一作]`, Zeyu Liu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种双环式外骨骼协助框架，将多模态轨迹生成与基于轨迹不确定性的可变阻抗控制耦合，实现可靠参考生成和自适应顺从交互。

**💡 创新点**

创新点在于：①利用全身运动与任务描述共同引导生成模型；②通过轨迹方差估计不确定性以在线调节阻抗；③引入三阶段训练管线以缓解跨模态表征不匹配。

**🔧 技术方法**

采用扩散模型（flow‑matching）+Transformer编码解码器、LoRA微调、以及基于轨迹方差的可变阻抗控制，结合输入到状态稳定性分析。

**📊 数据集**

使用 HumanML3D 进行预训练，MuJoCo 生成的模拟人机交互数据，以及真实外骨骼数据（3 名训练用户共 13 任务）进行评估，并在 7 名未见用户上测试。

**📈 对比分析**

通过与意图预测基线和基于误差的阻抗控制比较，实验显示轨迹 RMSE 降低 31%，肌肉激活降低 15–20%，代谢能耗降低 9%，且轨迹方差与跟踪误差呈显著正相关，证明方法有效。

**⚠️ 局限性**

局限性在于仅验证单关节肘部外骨骼，依赖 RGB‑D 与位置/力传感，且对更复杂、多关节任务的可扩展性尚未检验。

---

## 488. RecallGS: Geometry-Grounded Recall for Online Semantic Gaussian Splatting

**arXiv ID:** 2610.31293 | [PDF](https://arxiv.org/pdf/2610.31293v1)

**作者:** Thanh-Khoi Nguyen `[一作]` (University of Science), Minh-Triet Tran `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、基于回忆-定位-融合的在线语义3D高斯绘图框架RecallGS，利用历史观测作为非参数几何语义先验来改进语义重建。

**💡 创新点**

创新点在于将观察历史视为可查询的几何语义记忆，通过CLIP检索、alpha合成权重几何定位以及多视角特征聚合，避免额外3D网络训练和显存开销。

**🔧 技术方法**

使用的技术包括3D高斯绘图（3DGS）、CLIP图像级和实例级特征、alpha合成权重、基于可见性加权的特征融合以及FreeSplat++作为底层重建骨干。

**📊 数据集**

在ScanNet和ScanNet++两个室内扫描数据集上进行实验。

**📈 对比分析**

与多种基线（EmbodiedSplat、EmbodiedSplat-fast及其他开放词汇3DGS方法）比较，RecallGS在保持约2.48 FPS的同时，在mIoU上接近EmbodiedSplat，明显优于无3D分支版本。

**⚠️ 局限性**

局限性在于依赖历史观测的质量与密度，缺少对记忆可靠性的评估，且只对当前局部高斯做可见性评估，未考虑全局遮挡。

---

## 489. Beyond Accuracy: Disentangling Decomposition from Typed-State Quality Assurance in Multi-Agent LLM Pipelines

**arXiv ID:** 2610.31280 | [PDF](https://arxiv.org/pdf/2610.31280v1)

**作者:** Maria Gabriela Alves Zuppardo `[一作]` (Universidade Federal de Pernambuco), Vinicius Cardoso Garcia `[通讯]` (Universidade Federal de Pernambuco)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在需求工程任务中，比较了三种多智能体LLM流水线架构：单调用基线、两调用无类型状态、两调用带Pydantic类型化状态的流水线，评估其对功能/非功能需求分类及MoSCoW优先级的影响。

**💡 创新点**

创新点在于系统性拆分“解构”与“类型化状态”对性能和质量保障的贡献，首次通过实验量化类型化状态带来的结构验证时机与模型一致性提升，并提供公开复现包。

**🔧 技术方法**

技术包括：LangGraph调度、Pydantic校验、基于正则的JSON提取、两步Prompt设计、置信度阈值路由、故障注入实验，使用本地部署的三款7–8B小型语言模型（qwen2.5‑7b、llama3.1‑8b、mistral‑7b）。

**📊 数据集**

使用PROMISE NFR+ 625条需求（功能与10个非功能子类）以及机器翻译的葡萄牙语版本，保持标签一致。

**📈 对比分析**

比较方法为配对Wilcoxon、McNemar、Cohen’s h/g、Fleiss’ κ等统计检验；结果显示：大部分F1提升来自解构+专用Prompt（Δp≈+0.020至+0.075），类型化状态对F1影响微乎其微（Δs≤0.013），但能更早发现超界值和提升跨模型MoSCoW一致性。

**⚠️ 局限性**

局限性包括：Prompt–架构共线性难以分离解构与提示、置信度路由几乎未触发、实验仅覆盖三款小型模型和单一数据集、语言效应与Prompt共同变动、无MoSCoW真实标签，故结论仅适用于本设置。

---

## 490. CoWriter: Guiding Methodological and Synergetic Human-LLM Co-Authoring

**arXiv ID:** 2610.31277 | [PDF](https://arxiv.org/pdf/2610.31277v1)

**作者:** Shmuel Amar `[一作]` (Bar-Ilan University), Ido Dagan `[通讯]` (Bar-Ilan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个主动式协同写作框架，指导人类作者在整个写作流程中保持主动参与。

**💡 创新点**

将认知科学中的写作方法学转化为多代理系统，主动激励用户参与并跟踪写作活动。

**🔧 技术方法**

基于Claude 3.5大型语言模型的代理组件、写作日志、监控与审稿代理。

**📊 数据集**

使用DREsS数据集中的四个日常主题及其对应的十篇学术论文。

**📈 对比分析**

对比基线LLM聊天+画布系统，实验显示其覆盖写作活动更完整、用户参与率提升2.2倍、文档“人情味”得分更高。

**⚠️ 局限性**

样本规模有限、仅在大学研究生中测试，未涵盖专业写作或多样化写作体裁。

---

## 491. Hardness and Approximation for Compressing Deterministic Finite Automata

**arXiv ID:** 2610.31273 | [PDF](https://arxiv.org/pdf/2610.31273v1)

**作者:** Philip Bille `[一作]` (Technical University of Denmark), Christian Møller Mikkelstrup `[通讯]` (Technical University of Denmark)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了D^2FA（带默认转移的确定性有限自动机）压缩问题，证明寻找最优D^2FA是NP难的，并给出了MSF（最大生成森林）压缩算法的3/2近似保证并证明该比值是紧致的。

**💡 创新点**

创新点在于：①首次将D^2FA压缩问题的最优解与集合覆盖问题建立多项式归约，证明该问题NP难；②指出默认转移允许形成环（非森林）可进一步压缩，并证明在最优解与MSF压缩之间的最坏情况比值为3/2且不可改进；③构造了具有该逼近比的DFAs实例。

**🔧 技术方法**

采用理论分析和组合优化技术：多项式归约、默认转移图（DTG）的伪森林结构、默认循环性质、最坏子图分析、平均替换量上界、以及构造性证明（集合覆盖实例与特定DFA族）。

**📊 数据集**

无具体实验数据集；研究基于抽象的DFA构造与理论证明，未使用真实网络流量或正则表达式集合。

**📈 对比分析**

通过与最优D^2FA大小比较，MSF压缩大小不超过最优的1.5倍。作者给出紧致性证明：构造一族DFA使MSF大小与最优之比趋近于3/2，从而验证近似比的最优性。

**⚠️ 局限性**

局限性：① NP难性意味着实际应用中无法保证得到全局最优压缩；② MSF压缩虽然具有3/2近似，但在最坏情况下仍可能导致显著的空间浪费；③研究仅限于理论分析，未考虑实际深度包检测系统中的实现细节、运行时延迟或硬件资源限制。

---

## 492. Reinforcing intra-operative tumour localisation with weak supervision from pre-operative appearances

**arXiv ID:** 2610.31253 | [PDF](https://arxiv.org/pdf/2610.31253v1)

**作者:** Sidhart Sami `[一作]` (Queen Mary University of London), Shaheer U. Saeed `[通讯]` (Queen Mary University of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

利用基于预手术磁共振影像训练的肿瘤存在分类器，在无配对数据的情况下通过弱监督和强化学习引导肝细胞癌在术中CT影像中的定位。

**💡 创新点**

创新点在于：① 通过跨模态的弱监督信号（窗口级分类概率）实现肿瘤出现信息的迁移；② 采用强化学习搜索策略，将分类概率的差值作为奖励，自动定位肿瘤；③ 对比零样本、少样本以及使用CycleGAN生成的合成MR三种迁移策略，证明合成迁移可显著提升性能。

**🔧 技术方法**

主要技术包括：SE-ResNet3D肿瘤存在二分类器；3D CycleGAN用于CT→合成MR图像翻译；PPO强化学习框架进行窗口导航；窗口概率最大化聚合生成肿瘤分数图；阈值化得到最终分割。

**📊 数据集**

使用的数据集包括：MR影像来自LLD-MMR、ATLAS、CHAOS（共334例）；CT影像来自WAW-TACE、CHAOS-CT及内部数据（共313例）；评估集为TCIA HCC‑TACE‑Seg（105例），其中包含放射科医生标注的肿瘤分割。

**📈 对比分析**

与零样本、少样本、合成迁移以及CT‑仅基线（全监督、弱监督、nn‑UNet）等方法比较。合成迁移在验证集上取得最高平均Dice 0.31±0.10，成功率93%；相较于其他方法和CT‑仅基线均有显著提升（p<0.01）。

**⚠️ 局限性**

局限性在于：① 由于CT上肿瘤可见性差，Dice值仍相对较低，肿瘤边界精细分割困难；② 强化学习对窗口大小、阈值等超参数敏感；③ 仍缺乏真实配对MR‑CT数据，可能影响跨模态迁移的通用性。

---

## 493. QuaNAS: Uncertainty-Aware Reuse of Earlier Searches for Query-Efficient Neural Architecture Search

**arXiv ID:** 2610.31251 | [PDF](https://arxiv.org/pdf/2610.31251v1)

**作者:** Brama Seta Putra Satria `[一作]` (Institut Teknologi Sepuluh Nopember), Laras Suciningtyas `[通讯]` (Institut Teknologi Sepuluh Nopember)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 QuaNAS 方法，在新数据集上利用已有搜索库作为先验，并通过高斯过程动态调整其权重，进一步结合零成本代理实现查询高效的神经架构搜索。

**💡 创新点**

创新点：① 无标签目标迁移，直接使用过去搜索的主效应作为先验；② 对先验可信度进行在线重估，避免一次性偏差；③ 结合零成本代理并使用可靠性开关，保证在库信息不足时退回代理；④ 先验的可解释性让迁移知识可视化。

**🔧 技术方法**

技术：岭回归主效应模型、留一任务堆叠、零成本代理（SNIP、NASWOT、GradNorm 等）、高斯过程（Hamming 核）与期望改进采集、在线重权重、可靠性切换。

**📊 数据集**

使用 12 个未见公共数据集（4 图像、2 时序、3 文本、3 表格）以及 NAS‑Bench‑201 全空间作为验证集。

**📈 对比分析**

与 13 预设基线（RoBoT、PTK‑MR、QIEA‑ZC、REA、LS、RS 等）以及后续对比（RGPE、WS‑GP、NT、GP‑BO）比较；QuaNAS 在任何时刻的 regret 平均为 0.058，较基线低 2.7‑3.9 倍；时间到目标平均 197 秒；最终模型在 9/12 数据集上达到或超过匹配配方参考模型。

**⚠️ 局限性**

局限：仅在已知搜索空间和有限候选池上验证；对平坦或噪声搜索空间表现差；依赖短期训练目标，可能误导模型选择；未验证跨搜索空间迁移；库规模需至少几项才能有效。

---

## 494. Beyond Homology: Non-Abelian and Orientability Invariants Sensitivity in Cellular Networks

**arXiv ID:** 2610.31234 | [PDF](https://arxiv.org/pdf/2610.31234v1)

**作者:** Raghuwansh Raj `[一作]` (University of Luxembourg), Jun Pang `[通讯]` (University of Luxembourg)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文通过构造配对的多维复形，并对各种可能的捷径信息进行严格匹配与审核，检验拓扑深度学习模型是否真正捕捉到目标拓扑不变量；

**💡 创新点**

创新点在于提出了“表示相对”认证框架，结合渠道审核、受限探针和基准对照，实现对模型分离背后捷径信息的可审计性评估；

**🔧 技术方法**

使用了高阶消息传递网络（MPSN、CIN、SMCN 等）、1‑WL 和 Hasse‑1‑WL 细化、距离直方图探针以及多种图谱与谱信息；

**📊 数据集**

构造了三类基准：可定向性对照组（圆柱/莫比乌斯、圆环/克莱因瓶、去孔圆环/克莱因瓶）和非阿贝尔基本群对照组（双曲面群 vs 自由群），共计 64+50+16 对；

**📈 对比分析**

比较方法为 BREC‑style 对偶分离与排列可靠性测试，结果显示受限探针已能分离大多数可定向性对照组，基线 1‑WL 模型无法分离；在基本群对照组中，基线 1‑WL 及受限探针均失败，但更强的子图/子复形模型（ESAN、SMCN）能够分离约 70–80% 的样本；

**⚠️ 局限性**

局限在于认证仅相对已发出的表示与已审核渠道，未能排除所有潜在捷径；未能证明模型真正计算基本群或可扩展到更一般的表现形式；

---

## 495. When LLMs Forget What They Believe: Characterizing State Integrity Collapse in Long-Horizon Dialogues

**arXiv ID:** 2610.31314 | [PDF](https://arxiv.org/pdf/2610.31314v1)

**作者:** Yancheng Zhu `[一作]` (Shenzhen University), Yifan Yuan `[通讯]` (Shenzhen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了StateTrack-Bench基准，测量大型语言模型在多轮对话中对持续约束的记忆与执行一致性。

**💡 创新点**

提出状态完整性崩溃（State Integrity Collapse）概念，拆分记忆与执行两环，并提供forked recall‑compliance 审计与多类别约束的对话设计。

**🔧 技术方法**

采用确定性解码、自动化评判器、分支回放审计、结构化约束分类、温度鲁棒性测试等技术。

**📊 数据集**

使用自研的125交互场景（每模型25约束类别）涵盖数值、格式、角色、事实记忆、逻辑规则；并扩展到50轮、中文/日文小规模验证。

**📈 对比分析**

通过严格联合保持率与SIC计算，对7个模型进行对比；整体保持率介于62.4%–100%，大多数失效为完整性崩溃（≈88%），Gemini‑3.1‑Pro最弱，其余模型保持在90–96%范围。

**⚠️ 局限性**

局限性包括样本量有限、评判器依赖LLM、跨语言覆盖不足、非对抗性扰动、单约束设计、深度50轮外未评估、缺乏动态约束更新机制。

---

## 496. The Delayed Countdown Method for Validating Review-Oriented AI Workflows in eDiscovery

**arXiv ID:** 2610.31311 | [PDF](https://arxiv.org/pdf/2610.31311v1)

**作者:** David D. Lewis `[一作]` (David D. Lewis Consulting), Eugene Yang `[通讯]` (Johns Hopkins University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“延迟倒计时（Delayed Countdown）”方法，用于评估电子发现（eDiscovery）中面向审查的 AI 工作流的有效性。

**💡 创新点**

创新点在于：①在审查已进行一段时间后再抽样；②将抽样得到的未审查相关性估计转化为总相关文档数的置信区间；③通过“裁剪”逻辑上不可能的置信区间值，保证频率学上有效的置信区间；④兼容任意停止规则，支持在不同时间点进行回溯性评估。

**🔧 技术方法**

主要技术包括：频率统计置信区间推断（如二项式置信区间）、抽样方法（简单随机抽样或分层抽样）、逻辑推导与裁剪、以及与传统倒计时（Countdown）和逃逸（elusion）方法的对比分析。

**📊 数据集**

本文未使用单一公开数据集，而是通过已有研究（如 Tredennick 等的案例、Losey 的示例）和理论构造的样本进行演示；也讨论了实际 eDiscovery 过程中的抽样场景。

**📈 对比分析**

与传统倒计时、OneShot/URE、ei‑Recall 等方法对比，延迟倒计时在保持置信区间有效性的同时，提供了更大的灵活性：可在不同审查阶段使用同一抽样结果推断不同时间点的召回率；同时避免了多重抽样导致的可选停止偏倚。性能上没有数值实验，但理论分析表明其在置信度和成本控制方面优于单独使用倒计时或逃逸方法。

**⚠️ 局限性**

局限性包括：①需要审查经理自行决定抽样时机与样本大小，过早或过晚抽样均可能导致高成本或低效估计；②对审查编码准确性的假设仍存在争议；③在极端情况下（如高度低估/高估相关文档数）置信区间可能变为空或过窄；④方法对非中文文本或特殊法律场景的适用性需进一步验证。

---

## 497. Extreme Compression of Per-Layer Embeddings via Cross-Layer Latent Vector Quantization

**arXiv ID:** 2610.31309 | [PDF](https://arxiv.org/pdf/2610.31309v1)

**作者:** Jiacheng Liu `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出OPTIC方法，对Gemma 4的Per‑Layer Embeddings（PLE）进行极端压缩，达至数百倍压缩率，且在多项基准任务上保留接近原模型性能。

**💡 创新点**

创新点在于：①将Token的编码共享至所有层，并为每层设计独立的仿射解码器；②在PLE的门控残差投影输出处拟合重构误差（output‑path fitting），而非单纯在原坐标上；③通过混合代码本与层解码器的共享策略，显著减少存储需求。

**🔧 技术方法**

主要技术包括：离散代码本（product quantization）与多组代码本；层特定仿射解码器；输出路径加权的PCA+PQ初始化；交替优化（连续更新代码本/解码器与离散重分配）；以及对多任务评价指标的几何平均保留率评估。

**📊 数据集**

使用了Gemma 4的E2B和E4B检查点（BF16和W4A16），并以FineWeb、FineWeb‑Edu、OpenWebMath、FineWeb2等自然文本序列加上为每个词汇生成的合成记录作为校准数据；基准任务包括PIQA、GSM8K、GPQA Diamond、AIME 2026、MMLU‑Redux、IFBench和InfographicVQA。

**📈 对比分析**

与六种传统量化/量化+重构基线（PQ、RVQ、OPQ‑P、AQ、AQLM、SVD）以及输出路径加权PCA‑PQ进行对比。OPTIC在约100×压缩时，BF16 E4B保持97.8%七项任务保留率，W4A16 E4B保持96.9%；在594×、1002×压缩时仍保持90.5%和84.1%；相较于最强基线提升约30个百分点。

**⚠️ 局限性**

局限性包括：仅在Gemma 4的E2B/E4B上评估；结果对随机种子敏感；代码共享与解码器存储权衡未单独分离；在极低位压缩（≈1字节/向量）时，部分基准可能退化为零；适用范围可能受限于具有层级嵌入的模型结构。

---

## 498. The Value of Memory in Language Agents

**arXiv ID:** 2610.31303 | [PDF](https://arxiv.org/pdf/2610.31303v1)

**作者:** Zehao Lin `[一作]` (MemTensor), Kai Chen `[通讯]` (MemTensor)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个决策理论框架，用以量化语言模型记忆在未来决策中的价值，并在LoCoMo等公开基准上进行实证验证。

**💡 创新点**

创新点在于将记忆的保留、恢复与修订价值统一到风险–成本前沿，揭示信息时序、指针、反馈等因素对价值的决定性作用，并给出最小反例阐释。

**🔧 技术方法**

主要技术包括决策理论、信息论模拟、风险–成本分析、模拟与实证对比，使用大型语言模型及其检索/更新接口进行实验。

**📊 数据集**

实验数据集包括LoCoMo对话多轮、LongMemEval、MemOps等公开基准，以及作者自制的双项目更新和纵向案例。

**📈 对比分析**

通过比较完整历史、记忆账本、恢复检索与仅更新等多种条件，使用 F1、误差率等指标评估，实验显示账本能显著提升 F1 并减少放弃，恢复检索进一步提高准确率；双项目实验验证了更新范围对保留其他使用的影响。

**⚠️ 局限性**

局限性在于实验覆盖范围有限，仅针对少数任务和模型；评估依赖人工标注与特定模型，缺乏大规模多任务验证；成本与收益的量化权衡尚未充分展开。

---

## 499. A Polynomial-Time Algorithm for $k$-Path Vertex Cover Reconfiguration on Trees under Token Sliding

**arXiv ID:** 2610.31261 | [PDF](https://arxiv.org/pdf/2610.31261v1)

**作者:** Duc A. Hoang `[一作]` `[通讯]`, Duc A. Hoang

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种多项式时间算法，判定在树图上两个 k‑vertex‑cover（k‑PVC）是否可通过“token sliding”操作互相到达，并给出了完整的可达性判定与构造路径的技术框架。

**💡 创新点**

创新点主要在于：
- 引入“rigid tokens”和“properly rooted subtree”概念，揭示了可达性完全由这些硬盘子集和各连通分量内的 token 数量决定；
- 通过一个新的递推式（c, τ 递推）对树的子结构进行精确评估，直接给出从任何可移动 token 集合到最小 token 集合的最优路径；
- 用更强大的子树删除策略（删除一个 properly rooted 子树）取代 Demaine 等人提出的安全度为 1 的叶子删除，从而扩展到任意 k≥2。

**🔧 技术方法**

主要技术：
- 组合性分析与路径/连通分量分解，利用树的层次结构构造 properly rooted 子树；
- 动态规划（递推式）计算 c(P,r) 与 τ(P,r)，实现对可移动 token 数量和最小 ε 的全局最优判断；
- 通过 token sliding 的内部滑动与跨子树滑动两种基本操作，构造可达性证明与实现算法；
- 证明与归纳法结合，保证算法在所有可达的 k‑PVC 之间都能找到有效路径。

**📊 数据集**

论文中并未使用公开的实验数据集，而是对所有树结构给出理论证明和算法复杂度分析；若做实验，常用的树形图如随机树、二叉树、平衡树等。

**📈 对比分析**

与 Demaine 等人之前的结果相比，本算法：
- 在时间复杂度上仍保持多项式（实际上为 O(n²) 或更低，取决于实现细节），
- 支持更广泛的 k（k≥2）和更灵活的子树删除策略；
- 通过严格的可达性判定，能够在实际应用中直接输出完整的 token sliding 路径，而非仅给出可达性判断。

**⚠️ 局限性**

限制：
- 仅适用于树图；对一般图的可达性问题仍未解决；
- 对于极大规模树，尽管复杂度为多项式，但递推式和状态维护可能导致较高的常数项；
- 需要预先知道 k‑PVC 的大小，若 k 变化需重新计算；
- 该算法不直接给出最短滑动序列，只保证存在一条可达路径。

---

## 500. A Systematic Comparison of Search Space Complexity and Efficiency for Time-Synchronous End-to-End ASR Models

**arXiv ID:** 2610.31259 | [PDF](https://arxiv.org/pdf/2610.31259v1)

**作者:** Simon Berger `[一作]` (RWTH Aachen University), Ralf Schlüter `[通讯]` (RWTH Aachen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

系统评估了端到端ASR模型（CTC与神经转录器）在不同搜索策略下的性能与搜索空间。

**💡 创新点**

提出并量化了动态基于得分的剪枝相较固定束宽剪枝的效率优势，并展示了外部语言模型在动态搜索中的重要性。

**🔧 技术方法**

采用了动态得分剪枝、固定束宽剪枝、词汇表约束/无约束搜索、子词与词级语言模型、CTC与转录器解码器等技术。

**📊 数据集**

使用了LibriSpeech、Loquacious、以及一套约1.2万小时的西班牙语电话语料。

**📈 对比分析**

通过对WER与RTF的对比，证明在使用外部LM时动态剪枝可在保持相同WER的前提下将RTF提升约10‑70%，并在最坏情况下将搜索空间规模下降至一阶数量级。

**⚠️ 局限性**

仅针对时步同步的CTC与转录器搜索进行研究，未覆盖标签同步、流式解码以及更大规模模型与数据的进一步实验。

---

## 501. Query-Conditioned Liquid Temporal Grounding for Long-Form Audio

**arXiv ID:** 2610.31252 | [PDF](https://arxiv.org/pdf/2610.31252v1)

**作者:** Yuelan Cheng `[一作]` (University of Surrey), Wenwu Wang `[通讯]` (University of Surrey)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了长音频的时间定位问题，提出了一种基于查询条件的液态时序定位框架；

**💡 创新点**

创新点在于将FiLM注入查询语义与闭式连续时间网络CfC相结合，实现查询驱动的液态时序建模，并通过冻结的大型音频语言模型进行候选选择及轻量化边界微调；

**🔧 技术方法**

采用FiLM、CfC、冻结的大型音频语言模型(LALM)、轻量级Conv1D边界微调、以及跨模态匹配技术；

**📊 数据集**

使用LAT‑Bench英文TAG子集数据集，并以MOSS‑Audio和Qwen3‑Omni为后端骨干；

**📈 对比分析**

与滑动窗口（SW）、QA‑SFT等基线对比，mIoU提升至38.5‑44.1，R@0.5提升至40.6‑46.9，整体性能显著提升；

**⚠️ 局限性**

主要局限在候选选择上，Top‑1与oracle间存在显著差距，候选选择机制仍需进一步加强。

---

## 502. Muting Tokens: Exploring Token Sparsification for Efficient Audio Transformers

**arXiv ID:** 2610.31235 | [PDF](https://arxiv.org/pdf/2610.31235v1)

**作者:** Shreya Sajid `[一作]` (Eindhoven University of Technology), Joaquin Vanschoren `[通讯]` (Eindhoven University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对音频Transformer的token稀疏化方法进行了系统评估，并提出了基于频谱平坦度的联合注意力策略。

**💡 创新点**

创新点在于首次将频谱平坦度和低频能量融入token重要性评估，显著提升了后训练稀疏化的性能。

**🔧 技术方法**

使用了Top‑K、EViT、ToMe等稀疏化方法，并结合Spectral Flatness、Low‑Frequency Energy、Intensity等频域与能量特征。

**📊 数据集**

实验涵盖SPC、ESC50、AudioSet-20K三大音频数据集，覆盖不同音频长度与任务复杂度。

**📈 对比分析**

在post‑training稀疏化下，SF与LFE方法在保持率为25%时均能接近或超越全量模型，且相较传统Attention稀疏化提升约3-5%；在fine‑tuning稀疏化中，各方法差距缩小，整体性能保持稳定。

**⚠️ 局限性**

局限包括对低频滤波参数需手工调优、Intensity策略对短时音频表现欠佳，以及缺乏对实时边缘部署的进一步验证。

---

## 503. PointerBench: A Diagnostic Benchmark for Fine-Grained GUI Grounding

**arXiv ID:** 2610.31274 | [PDF](https://arxiv.org/pdf/2610.31274v1)

**作者:** Valentino Sacco `[一作]` (Sapienza University of Rome), Maximilian Schilling `[通讯]` (Warmwind AG)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了PointerBench，一个针对图形用户界面细粒度定位的诊断基准，包含表格、文本和专业应用三类任务。

**💡 创新点**

其创新之处在于将子元素几何、字符级定位以及多样化专业界面细粒度定位统一起来，提供像素级精确目标，帮助分离规划与定位错误。

**🔧 技术方法**

作者利用多模态大型语言模型（Claude、GPT、Gemini等）以及39种专用GUI定位模型，并采用统一的1024×768坐标框架和点/框评分标准进行评估。

**📊 数据集**

使用了约1500张截图–指令对的数据集，涵盖500个表格实例、500个文本实例和500个专业应用实例，涉及六种语言和100个真实软件界面。

**📈 对比分析**

通过对比前沿模型与专业模型，结果显示顶级模型在Sheets和Pro任务上可达≈80%成功率，但在Text任务尤其是字符、插入点等细粒度目标时仅≈30–70%，说明文本细粒度定位仍是瓶颈。

**⚠️ 局限性**

局限性包括仅在固定分辨率下评估、未考虑动态交互与规划过程、坐标框架不一致导致部分模型表现被低估，以及对专业应用的覆盖范围仍有限。

---

## 504. The Bayesian Learning Rule Beyond KL Geometry

**arXiv ID:** 2610.31308 | [PDF](https://arxiv.org/pdf/2610.31308v1)

**作者:** Sophia Sklaviadis `[一作]` (Instituto de Telecomunicações), Emtiyaz Khan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于混合空间 Bregman（Fenchel‑Young）散度的通用贝叶斯学习规则（BLR），实现可变支持与重尾的 q‑高斯后验，并给出 Newton‑style 实现；

**💡 创新点**

创新点在于：①证明混合 Bregman 散度能够保持 BLR 的加法更新；②将 Tsallis 负熵引入最大熵框架，得到受限支持与重尾 q‑高斯后验；③给出新的通用 BLR 更新公式并在深度网络训练中实现；

**🔧 技术方法**

使用的技术包括：Fenchel‑Young 不等式、混合空间 Bregman 散度、最大熵原理、指数族推广、梯度估计（Price/Bonnet）、二阶优化（Newton‑like）、变分推断、深度学习优化器（IVON、VON）等；

**📊 数据集**

实验数据集涵盖：CIFAR‑100、CIFAR‑10、MNIST、UCI 信用审批数据集以及 Bayesian logistic 回归；

**📈 对比分析**

与 SGD、VSGD、IVON、q‑VSGD 等方法进行对比，评估指标包括测试准确率、NLL、ECE、Brier 分数和 AUROC；实验表明 q‑VON 在精度、校准和鲁棒性上优于或与 IVON 相当，并且不需要额外调参；

**⚠️ 局限性**

限制在于：尚未在大规模 Transformer/LLM 训练中验证；高维协方差限制导致重尾 q 值趋近 1；实验规模有限，缺乏更大规模基准和更广泛的应用场景。

---

## 505. Subgoal-aware Memory Graph for GUI Agent

**arXiv ID:** 2610.31307 | [PDF](https://arxiv.org/pdf/2610.31307v1)

**作者:** Dezhao Luo `[一作]` (Huawei), Weilin Luo `[通讯]` (Huawei)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一种子目标感知的记忆图，用来在移动GUI代理中存储和检索过去交互经验，帮助代理在长周期任务中做出更高效决策。

**💡 创新点**

创新点在于将子目标（连续动作序列）作为抽象粒度，引入视觉状态共享节点与有向子目标边的图结构，兼顾动作级细节与轨迹级全局性，从而实现跨任务可重用的经验表示。

**🔧 技术方法**

使用了大语言模型（LLM）进行子目标分割与注释，视觉与文本嵌入由Qwen3-VL-Embedding-8B生成，图构建采用相似度阈值合并节点，检索则通过余弦相似度匹配并返回最近的子目标边。

**📊 数据集**

在SPA‑Bench English single‑app level‑3 数据集上进行实验，包含13个长序列任务。

**📈 对比分析**

与无记忆基线、动作级记忆和任务级记忆等对比，子目标图记忆提升了 11.11% 的单次成功率（从 34.19% 提升至 45.30%），显示出显著性能提升。

**⚠️ 局限性**

方法假设GUI状态可通过截图聚类为离散节点，若界面高度动态（如视频播放器），视觉相似度可能无法准确捕捉语义等价，限制了在极端动态环境下的适用性。

---

## 506. DISC: Domain-Integrated Similarity Cost for Time-Series Dataset

**arXiv ID:** 2610.31288 | [PDF](https://arxiv.org/pdf/2610.31288v1)

**作者:** Yen-Ku Liu `[一作]` (National Yang Ming Chiao Tung University), Ryan Rossi `[通讯]` (Adobe Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 DISC（Domain-Integrated Similarity Cost）框架，用于通过学习整合多种时间序列数据集相似性指标来预测源-目标转移任务中的下游误差。

**💡 创新点**

创新点在于：①将标准化的相似性值和基于排名的相对信息分离为基准混合和先验混合；②引入门限门控机制让先验信息动态调节基准相似度；③在训练时用源-目标误差校准权重，得到任务特定的可解释度高的指标组合。

**🔧 技术方法**

主要技术包括：多种数据集层级与降维器+序列级距离的组合（Wasserstein、MMD、OT、MAD、DBA、PCA 与 DTW/ED/LCSS 等），标准化与百分位排名转换，softmax 权重学习，门控（sigmoid）与梯度优化（Adam）等。

**📊 数据集**

使用了 46 个来自 GluonTS 的时间序列数据集，采用共享的 24/8/14 训练/验证/测试划分，针对四种任务（两种模型 × OOD 与 TL）进行评估。

**📈 对比分析**

方法对比：与最佳单一相似性指标及手工搜索的基准相比，DISC 在所有四个任务中均取得更高的源行皮尔逊相关系数，平均提升约 10.3%，与手工搜索相比提升超过 100%。

**⚠️ 局限性**

局限性包括：①需要在训练集上预先收集转移误差进行校准；②对窗口采样的选择敏感，过短或低多样性窗口会削弱单一指标的解释能力；③当前仅在共享划分上验证，跨划分泛化需进一步研究。

---

## 507. Learning Robot Behavior Priors from Non-Robotic Synthetic Data

**arXiv ID:** 2610.31275 | [PDF](https://arxiv.org/pdf/2610.31275v1)

**作者:** Mateusz Odrowaz-Sypniewski `[一作]` (University of Cambridge), Amanda Prorok `[通讯]` (University of Cambridge)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种利用纯数学抽象生成的合成数据（MDP、情境赌博机、线性动力系统）预训练Transformer，从而在零样本情境下完成机器人动作预测。

**💡 创新点**

创新点在于证明即便完全不接触任何机器人数据，使用抽象控制生成器也能获得可迁移的先验知识，并且这些先验优于通用因果先验，显著提升跨体态的零样本性能。

**🔧 技术方法**

技术核心为 Prior‑Data Fitted Networks（PFN）框架下的Transformer，采用合成任务分布采样、K‑means离散化动作、负对数似然与相对质心距离（RCD）评价。

**📊 数据集**

使用的评估数据集包括六种多智能体仿真环境（VMAS）、五种机器人抓取/操作任务（RoboMimic）以及四类真实机器人数据（A2D2、DROID、Aerial、GrandTour）。

**📈 对比分析**

通过与通用因果先验、仅用控制先验、混合先验以及计算匹配的真实机器人基线比较，发现控制先验模型在仿真任务上可达 55%–58% 的准确率，在真实机器人任务上仍能保持 16%–17% 的准确率，远高于随机（约 1.6%），并在某些未见任务上超过了仅靠真实数据训练的基线。

**⚠️ 局限性**

局限性包括模型规模有限导致跨体态泛化受限、对高维连续动作的预测准确率仍较低、在有限的真实数据上微调效果不一定优于从零训练，以及需要进一步验证在更大规模模型上的可扩展性。

---

## 508. Memory-Aware Reward Modeling: Learning Human Preferences for LLM Memory Use

**arXiv ID:** 2610.31272 | [PDF](https://arxiv.org/pdf/2610.31272v1)

**作者:** Yijiang River Dong `[一作]` (University of Cambridge), Jiaxin Pei `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究用户对个人记忆在语言模型助手中的使用偏好，收集基于真实查询与个人记忆的双选比较数据，构建记忆感知奖励模型。

**💡 创新点**

提出可解释的记忆感知奖励模型，首次从用户真实偏好学习记忆使用的权重，发现记忆化并不总是受欢迎并揭示两大评估缺口。

**🔧 技术方法**

采用鲁棒评估框架：先用LLM评估器抽取12个可解释特征，再通过L2正则化Logistic回归学习特征权重；使用提示生成器和多模型（GPT‑5.2、Gemini‑3.1‑Pro、Claude‑Opus‑5）生成对比答案。

**📊 数据集**

约87名参与者提供的821条有效双选标注（约1,000条原始标注），使用其自身ChatGPT交互记录和近月事件构成的记忆集，形成个性化评估数据集。

**📈 对比分析**

在三种生成器上与多数基线（直接LLM判定、Skywork奖励模型、鲁棒评估器等）对比，记忆感知奖励模型分别达72.8%、75.0%、82.2%的准确率，明显优于50–70%的基线；在最高置信度25%子集上可达88.8%。

**⚠️ 局限性**

仅覆盖单轮回答与单一记忆来源，未涉及多源长期记忆、多轮交互及行动等真实场景；评估仅在离线设置，未验证在后训练中的效果。

---

## 509. Notation3: Graph Terms for RDF

**arXiv ID:** 2610.31266 | [PDF](https://arxiv.org/pdf/2610.31266v1)

**作者:** Dörthe Arndt `[一作]` (TU Dresden), Pierre-Antoine Champin `[通讯]` (W3C)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文为Notation3（N3）逻辑提供了完整的抽象语法和形式语义，重点解决了图项（graph term）的语义问题，并构建了从基础语义到完整N3语义的两步模型理论；

**💡 创新点**

创新点包括：① 通过抽象语法明确区分存在量词与全称量词以及图项的闭包性；② 引入基语义（仅覆盖图项）和N3语义（处理内置谓词 log:implies 与 log:includes），实现对规则和“引用”语义的正式描述；③ 推导并证明了适用于N3的插值定理（interpolation lemma），从而把N3基本蕴含与RDF简单蕴含联系起来；

**🔧 技术方法**

主要技术手段包括：模型理论的解释函数与赋值映射、图项的规范化与正则化、变量映射的组合与逆映射、图的同构与子图关系、以及对图项的“全称扩展”（universal extension）构造；

**📊 数据集**

论文未使用任何实验数据集，而是基于理论证明和形式化推导；

**📈 对比分析**

方法的比较主要在理论层面：展示了N3基本蕴含与RDF简单蕴含的一致性、对比了RDF-star、重写与命名图等其他对“关于陈述”方案；由于缺乏实验评测，性能方面未作讨论；

**⚠️ 局限性**

局限性包括：插值定理仅对有限图有效；N3蕴含整体不可判定（与存在规则相关）；内置谓词有限，未覆盖所有N3自定义功能；以及对量词作用域的假设可能与未来标准产生偏差。

---

## 510. Deep ReLU Networks Can Fail the Best Approximation Property for Continuous Target Functions

**arXiv ID:** 2610.31264 | [PDF](https://arxiv.org/pdf/2610.31264v1)

**作者:** Sebastian Kassing `[一作]` `[通讯]` (University of Wuppertal), Sebastian Kassing (University of Wuppertal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

构造了一个连续目标函数，证明在深 ReLU 网络（两隐藏层、架构(2,2,1,1)）上 L^2 近似误差的下确界不被任何网络参数化达到，从而展示了深 ReLU 网络不满足最佳逼近性质；

**💡 创新点**

首次给出深层 ReLU 网络存在无全局最优解的严格对立例，挑战了仅在浅层网络已知的最佳逼近性质，并揭示了多层网络中“弯曲断面”对对称性与凸性分析的破坏；

**🔧 技术方法**

采用理论构造法，包括：构造“斜坡”函数与分段线性目标、分析第二隐藏层神经元的断面弯曲、利用 L^2 误差的凸性与不闭性、以及连续性、极限与支撑集论证；

**📊 数据集**

本工作完全是理论证明，不涉及任何实验数据集；

**📈 对比分析**

对比方法在本研究中并不存在，因为重点不是性能评估，而是存在性证明；

**⚠️ 局限性**

局限性包括：仅针对架构(2,2,1,1)的深网络，可能不适用于更宽更深的网络；结果基于 L^2 损失，其他损失函数可能表现不同；

---

## 511. Weaponizing the Exit: Input Privacy Leakage via Exit Routing Manipulation in Multi-Exit Networks

**arXiv ID:** 2610.31246 | [PDF](https://arxiv.org/pdf/2610.31246v1)

**作者:** Yulong Tian `[一作]` (Nanjing University of Aeronautics and Astronautics), Sheng Zhong `[通讯]` (Nanjing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了多退出网络在云服务部署中，通过攻击者控制模型并仅利用推理延迟来泄露用户输入的敏感属性的攻击方法。

**💡 创新点**

创新点在于提出了一种结合选择性标签平滑与熵-协方差优化的训练框架，使恶意模型能够在不显著影响主任务性能的前提下，将携带目标属性的输入定向到特定退出，从而实现隐私泄露。

**🔧 技术方法**

使用的技术包括选择性标签平滑、熵-协方差优化、两阶段微调、早期退出阈值控制以及对推理延迟的侧信道分析。

**📊 数据集**

在CIFAR-10、VGGFace2S、UTKFace和BreakHis四个数据集上，针对十个不同的目标属性进行了实验验证。

**📈 对比分析**

与随机猜测、频率预测和逻辑回归基线相比，攻击的平均F1分数达0.73，精度和召回率均显著提升（均高于77%），且对主任务准确率影响不足1.3%，推理加速下降幅度小于0.1。

**⚠️ 局限性**

局限性包括：需要攻击者在部署前能操纵模型并获取目标属性样本；对轻量级网络的干扰更大；防御措施往往需要牺牲多退出网络的加速优势。

---

## 512. Iterative Randomized Rounding for Better-Than-2 k-Median Approximation

**arXiv ID:** 2610.31229 | [PDF](https://arxiv.org/pdf/2610.31229v1)

**作者:** Jarosław Byrka `[一作]` (University of Wrocław), Zaixuan Wang `[通讯]` (Nanjing University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文对k‑median（以及更一般的k‑facility location）问题的随机近似算法进行了深入研究，提出了新的客户依赖的误差上界，并给出了相应的近似算法；随后给出了一种改进的逼近算法，能在保证只打开k+O(1)个设施的前提下，实现对每个客户的距离误差下更优的上界；最后给出了一种稳健（robust）算法，结合了过滤、包络、匹配和依赖性舍入，进一步提升了对所有客户的误差上界。

**💡 创新点**

创新点包括：
1) 对单 pivot 近似算法的精细化分析，得到客户的散度（dispersion）_j 能直接影响误差上界，使得若支持距离相同则误差正好为2；
2) 将该客户依赖误差上界与传统的 k+O(1) 设施数约束结合，得到一个可同时控制设施数与误差的完整方案；
3) 提出新的稳健算法，利用过滤+包络+匹配+依赖性舍入，在满足约束后对每个客户给出基于均值绝对偏差 M_j 的误差上界，形式为 min{(16+5/e)V_j, (1+2/e)V_j+√(90 V_j M_j/e)+15/2 M_j}，显著优于传统的常数2 上界。

**🔧 技术方法**

所用技术主要有：
- LP 预处理（将原始解转为  δ‑integral 形式），
- 单 pivot 采样与最近单位权重（w_qi）分析，
- 随机迭代与 supermartingale 证明，
- 过滤+包络（bundle）构造，
- 最大匹配构造与 laminar 结构，
- 依赖性舍入（Dependent Rounding）与负权重（imbalance）控制，
- 结合均值绝对偏差与散度的概率分析。

**📊 数据集**

本文并未在实验中使用实际数据集，而是提供了理论上最优的近似比和期望分析；若要验证，可在标准 k‑median 数据集（如 1K、5K、10K、SJD 等）上实现相同的 LP 预处理与舍入，观察实际距离误差与给出的理论上界。

**📈 对比分析**

与传统的 2‑LMP（2 倍最优）或 1.73‑LMP（针对 k‑median）算法相比，本文的单 pivot 算法在散度小或均值绝对偏差低时可实现 2 以内的误差；稳健算法进一步在任意散度下给出比 2 更低的误差上界；在最坏情况时仍保持 2‑LMP 的 2 倍上界，同时保证设施数不超过 k+O(1)。

**⚠️ 局限性**

局限性包括：
- 需要对 LP 解做 δ‑integral 预处理，可能导致常数因子和实现复杂度提升；
- 误差上界虽然在散度低时优于 2，但在散度高或均值绝对偏差大时仍受 2‑LMP 的 2 倍上界限制；
- 方案在理论上可扩展到 k‑facility location，但实际实现与复杂度尚待验证；
- 依赖于“拉马尔可夫”或“Markov”类不等式，导致对极端分布的分析不够精细。

---

## 513. Moving Regions, Not Moving Objects: Segmenting What Actually Moves under Camera Motion

**arXiv ID:** 2610.31209 | [PDF](https://arxiv.org/pdf/2610.31209v1)

**作者:** Wenhua Jin `[一作]` (Kosmo Research), Zesong Li `[通讯]` (Kosmo Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于源-目标帧对的移动区域分割方法，直接预测源图像中哪些像素在相机运动之外发生变化，无需光流、点轨或额外分割模型。

**💡 创新点**

① 将运动定义为帧对而非物体，利用目标条件注意力实现像素级掩码；② 通过单一 RGB 编码器+低层分支+目标记忆实现无光流、无追踪的推理；③ 设计 -5K benchmark（Static、Dynamic-Static、Dynamic 三种相机/场景运动设置），评估局部运动识别与相机运动抑制。

**🔧 技术方法**

使用 UniQuery4R 初始化的 ViT + shallow CNN 低层分支；目标记忆+跨注意力；辅助相机位姿预测；多尺度采样查询；BCE+Dice 损失与相机辅助损失。

**📊 数据集**

合成 3D 点轨数据（Blender、DAVIS、MoCA-Mask、OVIS、YouTube-VOS、HOI4D）和真实视频（Mannequin Challenge、LASIESTA、D3D-HOI），以及新建的 -5K benchmark（42 序列 5431 帧，分为 Static、Dynamic-Static、Dynamic）。

**📈 对比分析**

与 4D 场景流基线（Trace Anything、Any4D、4RC、MoVieS、UniQuery4R）和移动对象分割基线（FlowSAM、SegAnyMo、RoMo、Easi3R、VGGT4D、MoRe、GeoMotion、GMOS）对比。该方法在 -5K 的 moving‑region 任务中获得最高的 J&F 分数，在 Static 设定下显著降低误报；在传统 MOS 基准上也名列前茅；通过窗口聚合进一步提升召回。

**⚠️ 局限性**

局限：边界细化不如基线细腻；平坦或重复表面导致掩码不连续；对微小位移运动敏感；训练主要基于合成数据，真实场景标注稀缺，跨域泛化尚未充分验证。

---

## 514. How Well Do Decision Models Transfer to Vietnamese? An Empirical Study of Text Classification

**arXiv ID:** 2610.31422 | [PDF](https://arxiv.org/pdf/2610.31422v1)

**作者:** Phat Huu Ha `[一作]` (University of Information Technology), Dang Van Thin `[通讯]` (University of Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对四种专门化决策模型在越南语零样本文本分类任务中进行系统评估，并探讨标签描述语言及越南语文本扰动对模型性能的影响。

**💡 创新点**

在越南语领域首次进行零样本分类的多模型系统评估；比较专门化决策模型与大型语言模型的竞争力；揭示标签描述语言对不同模型的任务依赖性；评估三类越南语扰动（去重音、Teencode、打字错误）对鲁棒性的影响。

**🔧 技术方法**

使用四种决策模型（JEV、LAYA、GLiNER、Quyet）的零样本推理；构造三种标签描述配置（EN、VI、EN+VI）；生成三类文本扰动；用宏F1、准确率、加权F1等指标评估模型；将结果与公开的LLM零样本结果对比。

**📊 数据集**

共13个越南语分类设置，覆盖六大任务族：主题分类、情感识别、仇恨言论检测、情绪识别、意图检测、自然语言推理。数据集包括UIT‑VSFC、NEU‑ESC、UIT‑VSMEC、ViHSD、UIT‑ViCTSD、UIT‑ViOCD、PhoATIS、ViMQ、ViNLI、ViANLI、ViMedNLI和vnNLI。

**📈 对比分析**

以宏F1为主进行横向比较。JEV在11/13设置中取得最高分，平均宏F1达69.22；在与大型LLM的对比中，JEV在多项任务中竞争甚至超过；标签描述语言的影响因模型与任务而异，双语描述并未显著提升；在三类扰动中，去重音导致平均性能下降最大，JEV的下降幅度最小，显示出相对更高的鲁棒性。

**⚠️ 局限性**

仅在零样本条件下评估，未考虑微调或少样本情境；依赖预定义标签集，未探讨标签生成或自动扩充；扰动覆盖面有限，未涵盖所有可能的越南语变体；缺乏对模型内部决策机制的深入解释；实验集中在越南语，未验证跨语言推广性。

---

## 515. HERD: Harness-Evolving Recursive Distillation for Continual Learning

**arXiv ID:** 2610.31401 | [PDF](https://arxiv.org/pdf/2610.31401v1)

**作者:** Xun Sun `[一作]` (Monash University), Thuy-Trang Vu `[通讯]` (Monash University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

无法完成摘要，因为缺少论文具体内容。

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 516. Parameter-Efficient Adaptation of Large Language Models to a Low-Resource Agglutinative Language: A Corpus, Datasets, and a Reproducible Benchmark for the Bashkir Language

**arXiv ID:** 2610.31397 | [PDF](https://arxiv.org/pdf/2610.31397v1)

**作者:** Mullosharaf K. Arabov `[一作]` (Kazan Volga Region Federal University), Svetlana S. Khaybullina `[通讯]` (Kazan Volga Region Federal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Bashkir 语言的开放资源包（Bashkir Web Corpus）以及三份文本分类数据集，并系统评估了大型语言模型在 Bashkir 低资源环境下的适配方法。

**💡 创新点**

首次在 Bashkir 语言上对 QLoRA、CPT、ICL 等参数高效适配技术进行比较实验，展示了 Cyrillic 词表、俄语预训练对低资源语言适配的显著优势，并量化了 tokenizer 适配性（fertility）与下游性能的关系。

**🔧 技术方法**

使用 QLoRA、LoRA、CPT、全量微调、零样本/少样本 ICL 等技术，对 8 种模型（DistilGPT2、GPT‑2 base/medium、Phi‑3‑mini、Mistral‑7B、YandexGPT‑5‑Lite‑8B、mBERT、XLM‑R）进行实验。

**📊 数据集**

构建了 71,567 文档、46.9M token 的 Bashkir Web Corpus，并基于其生成 16,994 文档的二分类集、17,897 文档的 19 类多分类集、22,318 文档的 14 类多标签集；此外使用 Bashkir Wikipedia 等公开资源做预训练语料。

**📈 对比分析**

通过宏 F1、BPB、PPL、生成质量、tokeniser fertility 等指标对比。结果显示 YandexGPT‑5‑Lite‑8B 在二分类宏 F1 达到 0.748、Mistral‑7B 为 0.721、Phi‑3‑mini 为 0.701；在多分类宏 F1 最高 0.368；在 tokeniser fertility 上 YandexGPT‑5‑Lite‑8B 为 3.80，优于 GPT‑2 系列的 7.80；CPT 在不同模型/任务表现参差，部分提升。

**⚠️ 局限性**

受限于单个随机种子、有限的 CPT 语料量、未能交叉验证全量微调与 QLoRA、以及 LLM‑judge 对 Bashkir 语言正确性的低敏感性，实验结果存在统计功效不足和评估指标局限。

---

## 517. Convex Equilibrium Potentials for Monge--Ampère: Curvature Bounds and A Posteriori Evaluation

**arXiv ID:** 2610.31386 | [PDF](https://arxiv.org/pdf/2610.31386v1)

**作者:** Edison Vázquez `[一作]` (CINVESTAV), Tomasz Nowicki `[通讯]` (IBM Research, Thomas J. Watson Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了非负收敛Softplus平衡网络，用于求解Monge–Ampère方程，并在冻结模型上给出了严格的误差认证；

**💡 创新点**

创新点在于利用正性消除Sγ≤，得到二阶导数的最优δ⁻²上界和三阶导数的δ⁻³上界，并提出基于残差的后验评估与区间算术全域认证框架，可在单个训练模型上获得统一的残差与解误差上界；

**🔧 技术方法**

技术包括收敛的隐式固定点网络、隐式微分、残差误差传播、区间算术与球面网格分割、对角分解/求逆的数值稳定性等；

**📊 数据集**

实验使用两道制造的单位圆Dirichlet问题（P1、P2），分别在内部和边界采样点进行训练与测试；

**📈 对比分析**

在六个冻结模型上对不同迭代深度下的残差和误差上界进行评估；浮点评估的自适应停止可将运行时间缩短约1.5–2倍，而完整区间停止则更耗时；

**⚠️ 局限性**

限制在于仅验证冻结模型的评估误差，未证明训练收敛；实验规模有限；残差到解误差的理论常数未给出；区间方法对性能影响大，且缺乏与显式ICNN或传统数值解法的直接比较。

---

## 518. Making Generative AI Workable at Work: Accountability, Localization, and Representation in Kenyan and Nigerian Small Businesses

**arXiv ID:** 2610.31344 | [PDF](https://arxiv.org/pdf/2610.31344v1)

**作者:** Najeeb Gambo Abdulhamid `[一作]` (Microsoft Research Africa), Jacki O'Neill `[通讯]` (Microsoft Research Africa)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在肯尼亚和尼日利亚的中小企业中，通过对27位员工进行21次半结构化访谈，研究了生成式AI在日常工作中的使用，并识别了五种关键实践：作为初稿劳动、输出质量的迭代引导、工具生态的切换与比较、发布前的责任工作以及对语境不匹配的修复。

**💡 创新点**

创新点在于（1）将发布前的“释放工作”定义为一种边界面对的责任劳动；（2）将“语境不匹配”视为一种持续的、可修复的分解事件；（3）提出了包含这五种实践的工作框架；（4）基于此框架提出四条工作场所AI设计方向。

**🔧 技术方法**

采用定性研究方法——半结构化访谈、转录与主题编码分析；未使用机器学习或模型训练技术。

**📊 数据集**

数据来源为访谈文本与参与者提供的截图，样本为在肯尼亚与尼日利亚的21家中小企业的27名员工；未使用公开数据集或外部语料。

**📈 对比分析**

研究采用对比性、案例驱动的定性分析，对不同参与者的实践进行对照，未进行数值性能评估或实验对比；结论基于参与者叙述的经验和观察。

**⚠️ 局限性**

局限性包括：样本仅覆盖已连接数字技术且已使用生成式AI的中小企业，缺乏代表性；访谈仅用英语，可能忽视本土语言交互；研究时间为2023年夏季，工具与技术已快速迭代；未提供因果或客观性能评估。

---

## 519. The Value of System 1 Models in Medicine

**arXiv ID:** 2610.31315 | [PDF](https://arxiv.org/pdf/2610.31315v1)

**作者:** Xiufeng Yang `[一作]`, Keno Bressem `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文对四款零样本 System 1 模型在 86 项涵盖 14 个临床领域、20 种语言的任务上进行评估，涉及医学知识、文本理解、临床判断以及二维/体积影像识别。

**💡 创新点**

创新点在于首次构建大规模跨模态、跨语言的 System 1 评测基准，并与现有最优方法、专家水平及生成式模型进行系统对比，揭示了 System 1 在速度、能耗与可靠性上的优势与局限。

**🔧 技术方法**

使用的技术包括四款决策模型（Clef、Clef-flash、GPT‑6 Luna Decisions、Jev 1.14）及其对应生成式基底，评估指标为 AUROC、平衡准确率、校准误差、延迟、能耗；部分模型在本地 RTX 5090 GPU 上执行推理。

**📊 数据集**

使用的数据集为 86 个公开任务，来源自 14 个临床领域（诊断、放射、编码等），包括医学考试题、报告抽取、文本推理、影像感知、二维影像发现及体积 CT/MRI 发现，共计 179,111 条案例，涵盖 20 种语言。

**📈 对比分析**

比较方法：以 AUROC、平衡准确率、校准误差、成本、延迟和能耗为核心指标；System 1 模型在文本任务中匹配或超越 13/26 个已发表最佳结果（中位差 -0.01），在影像任务中匹配 8/35 个（中位差 -0.17）；与同族生成模型相比，System 1 在速度上快约 10–70 倍，准确率略低 3–5%。

**⚠️ 局限性**

局限性：仅采用零样本、单句提示；对影像的处理受限于一次最多 4 张图像、缺乏体积推理；在需要临床推理的任务（如分流分级、变异分类）表现不佳；在低资源语言中准确率大幅下降；未进行任务特定微调。

---

## 520. Automating Feature Flag Cleanup at Scale with Multi-Agent LLM Systems: An Industrial Experience at DoorDash

**arXiv ID:** 2610.31351 | [PDF](https://arxiv.org/pdf/2610.31351v1)

**作者:** Atharv Chandratre `[一作]` (DoorDash, Inc.), Jai Datar `[通讯]` (DoorDash, Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套端到端的多代理LLM系统，自动完成DoorDash内部Feature Flag（Dynamic Value）清理，包括查询实验平台状态、生成PR、测试与覆盖率验证。

**💡 创新点**

创新点在于：① 多角色代理架构（Claude Sonnet负责计划与数据收集，Claude Opus负责代码编辑）② Git worktree隔离实现并行清理，③ 通过MCP实时查询实验平台以获取准确目标值，④ 在PR生成前强制通过覆盖率、Detekt等验证门槛。

**🔧 技术方法**

使用技术包括：Claude Sonnet 4.6、Claude Opus 4.6、Google ADK、MCP实验平台接口、Jira API、Git worktree、Gradle、Detekt静态分析、JaCoCo覆盖率工具。

**📊 数据集**

实验数据集：DoorDash内部56,000+ DV，1,659个旧DV，评估样本为最近50个已清理的旧DV，按简单/中等/复杂三级复杂度分类。

**📈 对比分析**

评估方法：对50个DV产生PR后按CI通过率、代码正确性、开发者评审分为Accepted、Accepted with revision、Failed；结果显示90%成功率，平均成本$4.79、平均耗时13.8分钟，明显低于人工1-2小时清理；相较于规则式工具Piranha，LLM方案在复杂场景下表现更佳。

**⚠️ 局限性**

局限性：仅适用于DoorDash的三层依赖注入DV访问模式和Kotlin代码；对其他语言或直接API调用的项目可能不具通用性；LLM非确定性导致偶尔失败；手工清理耗时估计基于经验而非精确测量。

---

## 521. Typed Template Fuzzing

**arXiv ID:** 2610.31342 | [PDF](https://arxiv.org/pdf/2610.31342v1)

**作者:** Lu Maltsis `[一作]` (University of Manchester), Dominik Winterer `[通讯]` (University of Manchester)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了轻量级的LLM驱动模版变异测试方法Typed Template Fuzzing（TTF），在预处理阶段用LLM将种子程序转为带类型注解的模版，随后在无LLM调用的循环中对模版进行基于类型的变异，以高效生成语义有效输入。

**💡 创新点**

核心创新在于仅在预处理阶段使用LLM生成可纠错的typed template；通过与原始种子对应的修复步骤消除LLM错误；不依赖语言特定解析器或类型检查器；结合校准与Thompson采样提升有效率，实现跨语言高有效率的变异。

**🔧 技术方法**

采用大型语言模型（Qwen3-Coder-Next）进行模版生成与错误修复；实现基于类型和作用域的生成式变异；使用校准(常量替换)排除不可变换项；使用Thompson采样动态选择操作符；Python实现2,047行代码。

**📊 数据集**

收集多语言测试用例：C/C++/Go/Rust/Java/Scala/SQL/Python等共计约4000+种子文件；使用公开回归套件和GitHub仓库；对每种语言构造对应的prompt。

**📈 对比分析**

在六小时实验中与Fuzz4All、Kitten对比，TTF在所有目标上保持>90%有效率，覆盖率大多超越基线；相对领域专用模糊器，TTF的有效率仅低几个百分点；通过Bug搜集（70+新bug，42已确认、23已修复）验证实用性。

**⚠️ 局限性**

局限包括：仍需LLM预处理，若LLM产生错误未被修复仍会导致模板无效；校准仅使用常量替换，无法处理更复杂的不可变换项；对逻辑或稳定性错误的发现能力有限；依赖种子覆盖率，缺乏深度语义约束；在某些语言缺乏完整的预处理prompt或评估基准。

---

## 522. Harmful Responses Are Jailbreak Templates: Transferring Autoregressive Jailbreaks to Diffusion Language Models

**arXiv ID:** 2610.31331 | [PDF](https://arxiv.org/pdf/2610.31331v1)

**作者:** Jinwei Gan `[一作]` (Nanjing University), Qing Gu `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将自回归模型的有害响应转换为掩码模板，进而攻击扩散式大语言模型的框架

**💡 创新点**

利用有害响应本身作为攻击模板并通过均匀掩码与安全过滤实现跨模型攻击，无需针对目标模型优化提示

**🔧 技术方法**

自回归模型诱导有害响应、均匀掩码生成模板、提示式安全过滤、扩散式模型双向去噪推断

**📊 数据集**

HarmBench、JailbreakBench、StrongReject三个安全基准数据集

**📈 对比分析**

与多种自回归与扩散专用攻击方法对比，RMT-Jailbreak在所有目标模型上实现 61.3%–84.0% 的攻击成功率，明显优于基线和 DIJA

**⚠️ 局限性**

对自回归源模型或攻击方法的依赖、对高掩码比例下的泛化性、以及在自我提醒等提示级防御下的鲁棒性仍有限

---

## 523. MVLoMa: Multi-View Local Feature Matching with Any Geometry Foundation Model

**arXiv ID:** 2610.31321 | [PDF](https://arxiv.org/pdf/2610.31321v1)

**作者:** Xinyang Lu `[一作]` (Kosmo Research), Zesong Li `[通讯]` (Kosmo Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于星形图的多视角稀疏匹配器，能够一次性联合推理中心-邻居之间的对应关系，并在每条边输出传统的两视匹配结果供RANSAC使用。

**💡 创新点**

创新点在于（1）利用冻结的多视角3D基础模型（如DA3-G）生成几何感知的特征，并在星形图结构中共享中心表示，从而一次前向推理即可得到所有边的匹配；（2）引入可见性过滤的正样本和基于极线误差的几何负样本监督，以提升匹配的几何可靠性。

**🔧 技术方法**

技术包括：LoMa架构改进、星形图自注意力与交叉注意力机制、双重softmax匹配损失、几何负样本损失、使用冻结的DINOv2/DA3-G特征、PoseLib RANSAC以及对多视角3D基础模型的集成。

**📊 数据集**

使用的主要数据集有HyperSim、ScanNet++、Mip-NeRF 360、Tanks & Temples；在实验中还评估了MegaDepth-X、TerraSky3D、AerialMD等作为额外验证集。

**📈 对比分析**

在25视图星形图评估下，方法在所有三个AUC阈值（1°/5°/10°）的宏平均AUC均优于所有基线，最高AUC@1°≈58.6%；相较于逐对匹配，速度提升2.2×、计算量下降约37%。

**⚠️ 局限性**

局限性包括：对冻结特征的几何质量高度依赖；邻居之间不直接交互，导致无法充分利用邻居间信息；内存随视角/关键点数量线性增长；视角数提升并非单调，需学习邻居选择或自适应聚合；在纹理稀疏或几何不确定区域仍可能出现模糊匹配。

---

## 524. A Task-Centric Survey of Long-Document Visual Question Answering: Taxonomy, Evolution, and Future Directions

**arXiv ID:** 2610.31385 | [PDF](https://arxiv.org/pdf/2610.31385v1)

**作者:** Yufei Tong `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了面向任务的长文档视觉问答（VQA）分解框架，系统地将长文档VQA拆分为文档解析、证据组织与检索、生成与学习以及评估四个核心组件，并通过细粒度分类梳理技术演进。

**💡 创新点**

创新点在于以任务瓶颈为导向构建细粒度技术分类，揭示从内容理解到证据获取与持续管理的演进路径，并对未来的证据充分性评估、可验证与可纠错理解以及预算感知协同提出新的研究方向。

**🔧 技术方法**

本文综述并对比了多种技术路线，包括OCR+深度解析、视觉检索增强生成（RAG）、多模态LLM/VLM的检索与推理、基于反馈的证据获取、树/图结构导航以及强化学习驱动的动作策略等。

**📊 数据集**

所分析的数据集涵盖多种公开长文档VQA基准，如DocVQA、SlideVQA、MP-DocVQA、MMLongBench-Doc、MMDocIR、ViDoRe、DocScope、SynthDocBench等，覆盖单页、多页、跨文档等多种场景。

**📈 对比分析**

通过对150余篇论文的系统比较，本文在答案准确率、证据检索召回率、定位精度及端到端成本等维度进行评估，发现显式检索方法在证据充分性和成本管理上优于直接阅读，多轮反馈机制显著提升多跳推理性能，但仍存在预算下的效率瓶颈。

**⚠️ 局限性**

局限性在于仅聚焦长文档VQA的证据获取与使用，未深入讨论基础任务如文本/布局识别与字段抽取；对工业规模大数据和跨任务迁移的评估不足；以及对最新自监督预训练方法的系统分析缺失。

---

## 525. CoHelix: Event-Based Multi-Propeller Rotation-Rate Tracking with Shared Projective Geometry

**arXiv ID:** 2610.31384 | [PDF](https://arxiv.org/pdf/2610.31384v1)

**作者:** Radim Spetlik `[一作]` (Czech Technical University in Prague), Jiří Matas `[通讯]` (Czech Technical University in Prague)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种递归滤波器，可同时估计共享同一投影变换的四个共面螺旋桨的独立旋转状态；

**💡 创新点**

创新点在于：①使用单一共享单射映射（同一homography）来统一所有螺旋桨的图像投影；②在每个事件到达时即对该事件对应的螺旋桨状态和共享几何进行Kalman更新；③引入锁定率（lock）指标，衡量估计在同步真值误差2%内的持续时间；

**🔧 技术方法**

技术包括：事件摄像机处理、基于单射投影的几何建模、递归扩展Kalman滤波、阶段门控（innovation gating）与自适应权重、半双速/双速歧义处理、几何更新的窗口与逐事件两种形式对比；

**📊 数据集**

使用了两个数据集：①公开的Moving‑Camera TQE（12条记录，4米与2米摄像机运动）与②私有的Target‑Follower（3条记录，另一架四旋翼在上方跟踪）；

**📈 对比分析**

与两种前人方法（公开实现与另一自研方法）对比，锁定率分别提高至62.9%（TQE）和95.9%（Target‑Follower），平均绝对误差分别为1383 RPM和34.8 RPM；逐事件Kalman几何更新比窗口式Gauss‑Newton提高约23个百分点；

**⚠️ 局限性**

局限性：方法假设螺旋桨是共面且固定的，且对轮廓中心和半径的标定误差敏感；在非共面或旋转速率变化突变时性能下降；仅在两架四旋翼和两种事件相机上验证，缺乏对不同硬件与场景的泛化评估。

---

## 526. AfterDeleteBench: Auditing Post-Deletion Information Persistence in LLM Agents

**arXiv ID:** 2610.31366 | [PDF](https://arxiv.org/pdf/2610.31366v1)

**作者:** Ehsan Saleh `[一作]` (Ohio State University), Andrew Perrault `[通讯]` (Ohio State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套名为 AfterDeleteBench 的基准，用来在固定权重的 LLM 代理中，通过将“暴露-随后删除”与“从未暴露”两种情境配对，检验删除原始交互记录后代理是否还能从其持久记忆中恢复被删除信息。

**💡 创新点**

创新点在于：①提出了基于配对实验的行为审计框架，可精确归因恢复行为；②证明了即使完整字符串被删除，代理仍能通过非完整（组件化）记忆重构信息；③将恢复、写回和新会话召回等生命周期事件分离评估；④通过“记录谱系闭合”和“精确字面写过滤”两种干预，展示了在完整谱系信息下的理想删除上限与实际写入阻断的区别。

**🔧 技术方法**

技术手段包括：固定权重 LLM（Qwen2.5-7B-Instruct、Mistral‑7B、Phi‑3），外部持久记忆（扁平与层级两种结构），基于 HMAC 的精确写过滤，基于记录标识的谱系闭合，三种回答接口（legacy、aligned、JSON 结构），以及多种审计策略（单一直接查询、脚本化、信息增益、强化学习、提示式 LLM）。

**📊 数据集**

数据集为人工合成的 400 个“秘密”配置（200 训练/验证 + 200 生成式 OOD），覆盖四种结构（原子、重复/改写、相关、组合）与四个领域，确保对不同记忆形态和上下文的覆盖。

**📈 对比分析**

在 Qwen2.5‑7B‑Instruct 上，一次直接查询后恢复率约 68%（IID）/64%（OOD），而对照组恢复率为 0%；在层级记忆且无写过滤下，组件化记忆也能实现 62–99% 的完整重构；使用谱系闭合后恢复率降至 0%；写过滤只能阻止写入，但不抹去已有信息。相比其它模型，Mistral 与 Phi‑3 在组件化重构上的成功率更高，且两者在闭合干预下同样表现为 0%。

**⚠️ 局限性**

局限性包括：①仅测试精确字符串秘密，未覆盖自然语言或语义隐私；②使用的是人工合成且可追溯的记忆结构，真实系统中的压缩、学习或多对多依赖可能逃逸谱系闭合；③缺乏暴露-保持对照，无法评估删除后信息可恢复度的绝对基准；④接口与解析器的不匹配可能掩盖真实泄漏，需进一步统一评测标准；⑤只针对固定权重 LLM，未探讨参数学习导致的遗忘和重写。

---

## 527. Muffin - A Shared Memory Buffer for Asynchronous AI Agents

**arXiv ID:** 2610.31355 | [PDF](https://arxiv.org/pdf/2610.31355v1)

**作者:** Ping-Lin Kuo `[一作]` (University of Technology Nuremberg), Josif Grabocka `[通讯]` (University of Technology Nuremberg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于共享内存缓冲区的异步多代理推理框架Muffin，利用 orchestrator 路由查询，代理在不共享完整推理轨迹的情况下，仅异步分享关键信息以实现协同推理；

**💡 创新点**

创新点在于：1) 异步、选择性共享中间发现，避免全轨迹交换导致的上下文噪声与延迟；2) 引入 orchestrator 动态分配代理角色与子任务，提升任务匹配与协作效率；3) 通过摘要器从代理最后摘要中生成最终答案，进一步提升准确性；

**🔧 技术方法**

技术核心包括：大型语言模型（Gemma-4、Qwen3.5、Qwen3.8等）作为代理、orchestrator 与 summarizer；共享内存缓冲区实现 append‑only、无锁异步写；基于模板的 prompt 设计；多轮推理与动态触发机制；

**📊 数据集**

实验数据集包括：Chess Move Validity、MMLU‑Pro、GPQA‑Diamond、OlymMATH（易/难两子集）共四个推理 benchmark；

**📈 对比分析**

与单代理、多数投票、以及同步多代理辩论 baselines 比较，Muffin 在 20 项评测中 17 项最佳；在 OlymMATH‑Easy 上提升 19.2 分准确率，同时相较辩论模式减少 77.4% tokens，显示出更优的准确率‑成本 Pareto 前沿；

**⚠️ 局限性**

局限性在于：仅评估问答与推理任务，未验证工具使用、规划、环境交互等场景；计算成本受限，实验覆盖模型与参数有限，尚需更广泛的跨模型与跨任务验证。

---

## 528. SQuAT: Spherical Lattice Vector Quantization-Aware Training

**arXiv ID:** 2610.31412 | [PDF](https://arxiv.org/pdf/2610.31412v1)

**作者:** Sebastian Loeschcke `[一作]` (University of Copenhagen), Markus Nagel `[通讯]` (Qualcomm AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于形状-增益分解的晶格向量量化量化感知训练（Spherical Lattice VQ-QAT）方法，利用E_8导出的球面代码本和分离的直通估计器实现高效低位宽LLM压缩。

**💡 创新点**

创新点在于：1) 通过形状-增益分解和分离STE，产生切向空间更新，真正实现球面（Riemannian）优化；2) 设计了可实现分数位宽的E8S球面代码本；3) 开发了分布式精确最近邻分配，显著降低训练开销。

**🔧 技术方法**

使用的技术包括：向量量化、晶格量化（E_8）、形状-增益分解、直通估计器、球面参数化、分布式最近邻搜索、Hadamard旋转和量化感知训练（QAT）以及chi分位数增益量化。

**📊 数据集**

在LLaMA‑3.2‑1B/3B、Qwen3‑4B、Qwen3‑1.7B等LLM上，用DCLM‑Edu数据集进行指令调优和JSD蒸馏，训练范围从0.25B到10B tokens。

**📈 对比分析**

与标准的联合STE VQ‑QAT、INT‑QAT（SEQ）以及VQ‑PTQ基线比较，Spherical Lattice VQ‑QAT在相同量化器与初始化下显著提升WikiText2困惑度、GSM8K精度，并在2.25 bpw时几乎匹配或超越INT‑QAT，表现最优。

**⚠️ 局限性**

局限性包括：仅评估E8P12和E8S两种代码本；最大模型规模为4 B，未测试更大模型；缺乏高效的球面代码本解量化实现；仅在单一量化策略下验证，未探索更复杂的层次化或混合量化方案。

---

## 529. Same Entity, Different Script: Diagnosing Cross-Script Entity Matching Failures in Language Models

**arXiv ID:** 2610.31407 | [PDF](https://arxiv.org/pdf/2610.31407v1)

**作者:** Athira Gopal `[一作]` (QpiAI India), Ashwanth Krishnan `[通讯]` (QpiAI India)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多语言环境下对实体身份匹配进行受控实验，发现不同书写系统（德vanagari、Malayalam与拉丁）会导致同一人的名字在同一模型中产生显著的识别差异；

**💡 创新点**

通过对脚本差异进行隔离，结合校准、线性探针、激活补丁等技术，定位并验证模型内部决策阈值偏移与中层第二实体状态的因果关系，并展示可通过共享方向或多层窗口干预显著提升识别准确率；

**🔧 技术方法**

使用了激活补丁（activation patching）、线性探针（probe）、输出分数校准（output-score calibration）、梯度方向干预、层间向量校正等技术；

**📊 数据集**

构建了两个手工审核的数据集：601条包含Devanagari与拉丁两种形式的实体关系，374条包含Malayalam与拉丁两种形式的实体关系；

**📈 对比分析**

对比了多模型（Qwen2.5-7B、Gemma-2-2b、Llama-3.1-8B、Mistral-Small-24B、Sarvam-M-24B）在不同脚本和提示语言下的同人识别准确率、负样本准确率、AUROC 等指标，显示 Qwen 在同人识别上存在大幅偏差；通过校准、向量干预等方法将同人准确率从 6.5% 提升至 33.5%，负样本准确率保持在 98% 以上；

**⚠️ 局限性**

局限包括：数据集仅覆盖 601 个实体关系和两种脚本，未涵盖更多语言和实体类型；干预结果仅在 Qwen 上验证，其他模型未得到充分评估；方法在样本量有限的测试集上评估，可能受样本分布影响；对真实世界多语言识别系统的泛化能力尚未完全证明。

---

## 530. AI Risk Ontologies for the EU AI Act Compliance: AIRO, VAIR, and their Integration with DPV

**arXiv ID:** 2610.31380 | [PDF](https://arxiv.org/pdf/2610.31380v1)

**作者:** Delaram Golpayegani `[一作]` (Trinity College Dublin), Dave Lewis `[通讯]` (Trinity College Dublin)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 AI Risk Ontology (AIRO) 与 Vocabulary of AI Risks (VAIR)，用于描述 AI 系统及其风险，并通过语义网技术与 ISO/IEC 标准实现欧盟 AI 法合规的风险管理与技术文档化，同时将两者集成进 Data Privacy Vocabulary (DPV)，形成可重用的开放本体框架。

**💡 创新点**

创新点在于：① 采用迭代、模块化的 LOT（Linked Open Terms）方法论，实时响应 AI 法及 ISO 标准的演进；② 将核心概念拆分为 AIRO（基础本体）与 VAIR（词汇/层级化实例），提升可复用性与可扩展性；③ 通过 DPVCG 共识流程首次将专用 AI 本体纳入 DPV，提供统一的跨法规合规语义工具。

**🔧 技术方法**

使用的技术包括：OWL 2、SKOS、SHACL、SPARQL、RDF、Protégé、FaCT++/Hermit 推理器；质量评估工具 OOPS! 与 FOOPS!；基于 W3C 数据最佳实践与 FAIR 原则进行规范化与文档化。

**📊 数据集**

数据来源主要为：① AIAAIC 公开事件数据库（包含真实案例如 Uber RTID、VioGén 等）；② 合成用例 Proctify；③ ISO/IEC 22989、ISO/IEC 23894、ISO 31073 等标准文献，作为知识来源。

**📈 对比分析**

通过 OOPS! 检测本体错误、FOOPS! 验证 FAIR 合规性，并在 Protégé 推理器中验证逻辑一致性，证明本体满足结构完整性和语义可查询性；未对传统机器学习模型做数值性能对比，而是聚焦本体质量与合规适用性。

**⚠️ 局限性**

局限性包括：① 对 AI 法文本与 ISO 标准的解释性依赖，可能随法规更新导致概念变动；② 评估案例主要为有限的合成与公开事件，未覆盖全部真实工业部署场景；③ 尚未在大规模 AI 监管平台或数据库中进行实战部署验证。

---

## 531. RIDAR: Safeguarding Research Integrity in Autonomous Research Agents via Automated Detection and Runtime Intervention

**arXiv ID:** 2610.31372 | [PDF](https://arxiv.org/pdf/2610.31372v1)

**作者:** Jiening Siow `[一作]` (Tsinghua University), Juanzi Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了RIDAR框架，用于自动检测和实时纠正自主研究代理中的研究完整性违规（RIV）。

**💡 创新点**

创新点在于构建覆盖全研究生命周期的14类RIV统一分类体系RIVT，并将其与双模LLM（Analyzer + Verifier）及规则中间件相结合，实现端到端违规检测与运行时干预。

**🔧 技术方法**

技术上采用两阶段LLM推理（Analyzer负责候选生成，Verifier负责验证与类型归属）配合规则中间件、审计模式与轨迹笔记，形成证据驱动的违规定位与反馈机制。

**📊 数据集**

使用了60条专家标注的轨迹作为基准、300条未标注的真实轨迹以及60条配对干预轨迹进行评估。

**📈 对比分析**

在基准上实现了85.2% F1，检测到300条轨迹中99.7%出现3964个违规，并通过运行时干预将任务完成时违规数平均减少36.8%。

**⚠️ 局限性**

局限性包括对LLM判断的依赖、在数学/机器学习领域的评测范围有限，以及未涵盖专业化研究代理框架。

---

## 532. A Sharp Analysis of Fixed Offset Hash Tables

**arXiv ID:** 2610.31363 | [PDF](https://arxiv.org/pdf/2610.31363v1)

**作者:** Yuhao Guo `[一作]` (Tsinghua University), Chengzhang Wan `[通讯]` (Tsinghua University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对固定偏移序列（包括线性探测和二次探测）的开放寻址哈希表进行分析，证明在负载因子为1-ε时，期望插入时间上界为1+2/ε²；

**💡 创新点**

给出了一种简单直观的递归/势能分析，取得了最强的渐近上界，并证明该界对于线性探测已是最优，且该分析可适用于任意固定偏移序列；

**🔧 技术方法**

概率分析、势能函数、递归关系与上界推导，使用期望与线性性以及对序列占用情况的计数技巧；

**📊 数据集**

无实验数据集，全部为理论证明；

**📈 对比分析**

没有与实验或其他方法比较，主要通过理论上界与已知下界（如线性探测Θ(1/ε²)）对比，表明该上界接近最优；

**⚠️ 局限性**

结果仅适用于固定偏移探测，不能进一步改进到O(1/ε)（除非针对特殊序列如二次探测）；对非固定偏移或需要更高独立性的哈希函数仍有未解决的限制。

---

## 533. In-Browser Human-AI Co-Writing for Counterspeech: An Exploratory User Study of Agency, Perceived Response Quality, and Time

**arXiv ID:** 2610.31357 | [PDF](https://arxiv.org/pdf/2610.31357v1)

**作者:** Tammy Schmidt `[一作]` (Technical University of Munich), Luise Koch `[通讯]` (Technical University of Munich)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一个浏览器中的AI辅助反驳仇恨言论的协同写作工具，用户输入自己的想法后获得三条可编辑的生成建议。

**💡 创新点**

将AI协助定位为协同写作而非自动生成，强调用户主动输入、选择与编辑，并动态检索相关仇恨–反驳实例来生成更具语境的建议。

**🔧 技术方法**

使用检索增强生成（RAG）结合大语言模型（Llama 3 via Groq API）以及前后端分离的浏览器扩展架构，配合prompt工程与多模态检索。

**📊 数据集**

Multitarget CONAN 数据集（含 5,003 条 HS–CS 对）用于检索示例和评估。

**📈 对比分析**

在 10 名参与者的受控 within-subject 试验中比较未助理与助理写作，使用同伴评估五项质量指标（同理心、非毒性、相关性、具体性、说服力）。结果显示助理写作在所有指标上得分均更高，但平均撰写时间延长 54%。

**⚠️ 局限性**

样本量小、受试者为作者网络内人士，顺序效应和练习效应未能分离，未在真实平台环境中验证，工具仅评估协同写作体验而非实际反驳效果，缺乏对技术组件单独影响的研究。

---

## 534. CrisiLangs: A Multilingual Resource for Crisis Warning Generation

**arXiv ID:** 2610.31356 | [PDF](https://arxiv.org/pdf/2610.31356v1)

**作者:** Giacomo Gonella `[一作]` (Fondazione Bruno Kessler), Marco Guerini `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文创建了一个多语言版的 CrisiText 数据集 CrisiLangs，并基于该数据集微调并评估了多语言 LLM 在紧急预警信息生成中的表现。

**💡 创新点**

创新点在于首次构建含意大利语、保加利亚语、德语、希腊语的手工翻译预警信息数据集；使用单一多语言 LLM 进行联合微调；并通过自动指标和人工偏好评估检验温度与指导语的影响。

**🔧 技术方法**

技术上使用了多语言大语言模型（LoRA 微调），配合包含或不包含翻译后紧急指南的 prompt；在不同解码温度下生成预警信息，并采用 ROUGE‑L、BLEU、BERTScore 等指标评估。

**📊 数据集**

数据集为 CrisiLangs，由 5,000 条事件‑预警配对组成（每种语言 1,000 条），源自 CrisiText 经过专家评审后选出的 1,000 条样本，由四名本土译者翻译。

**📈 对比分析**

实验将微调模型与零样本基线进行比较，在 5 种语言、5 种温度下评测；微调模型在所有指标上均优于基线，温度 0.2 获得最高自动分数，并在人工偏好测试中得到最多选择。

**⚠️ 局限性**

局限性包括：仅覆盖 5 种语言且每种仅 1,000 条样本；评估仅使用单名本土专家，缺乏交叉验证；未测试直接跨语言生成；输入仍需自动翻译；对低资源语言的表现不稳定（如保加利亚）。

---

## 535. GelSlim 5.0: Unifying Vision and Touch with a Single Fingertip Camera

**arXiv ID:** 2610.31334 | [PDF](https://arxiv.org/pdf/2610.31334v1)

**作者:** Xiangdong Wang `[一作]` (University of Michigan), Xiaonan Huang `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了 GelSlim 5.0 传感器，使用单摄像头通过 UV 切换实现视觉与触觉共享视角，并提出多模态 Transformer 估计器来同时预测物体姿态与接触几何。

**💡 创新点**

创新点在于：① 连续 UV‑响应膜让视觉与触觉在同一视角下交替采样，避免外部相机的视角与标定问题；② 异步瓶颈融合与模态丢弃训练，使模型能够在配对输入或单模态下均能输出姿态与几何；③ 结合多模态 Transformer 与 DPT 解码器实现高精度姿态与深度/法向预测。

**🔧 技术方法**

使用的技术包括：SITR/MAE 前端编码器、Transformer 跨模态注意力、DPT 风格的几何解码器、真实与仿真数据共训练、模态丢弃与自适应视觉条件化。

**📊 数据集**

数据集为 12 个已知物体的真实接触录制（约 400 次接触/物体）以及基于 Blender 的仿真视觉、深度和法向数据，真实/仿真比例 1:1.27。

**📈 对比分析**

对比单模态（视觉或触觉）和配对模态，在内部验证中，配对输入将深度 MSE 降低 93.7%，角误差降低 72.3%，位移 L1 降低 61.3%，表现显著优于任何单模态。

**⚠️ 局限性**

局限性包括：对光照与仿真域差异敏感，需要精细平衡真实/仿真比例；目前仅在已知物体的内部验证中验证，跨物体/跨场景泛化尚未充分评估。

---

## 536. Hyperreduction accelerates finite element model updating for the discovery of neural materials

**arXiv ID:** 2610.31322 | [PDF](https://arxiv.org/pdf/2610.31322v1)

**作者:** Hagen Holthusen `[一作]` (RWTH Aachen University), Kevin Linka `[通讯]` (RWTH Aachen University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用投影式模型降阶（POD）与能量保持采样加权（ECSW）相结合，对基于结构实验数据的神经弹性材料识别进行加速，并通过自适应更新子空间和采样集保持识别精度。

**💡 创新点**

创新点在于：①将自适应POD与ECSW嵌入到梯度基神经网络学习框架中，使得在网络参数随迭代不断变化时仍能维持低阶模型的可靠性；②提出对能量的对数归一化与零锚 ICNN 结构，确保能量在任意权重下保持多可塑性、共形性和非负性；③通过可插拔的“bootstrap–canary”自适应策略动态更新子空间与采样，从而大幅降低对全阶求解的依赖。

**🔧 技术方法**

技术手段包括：投影式模型降阶（POD）、能量保持采样加权（ECSW）、自适应子空间更新、对数归一化的多可塑性输入凸神经网络（ICNN）、基于雅可比/牛顿法的有限元求解、自动微分/对偶法的梯度计算，以及自适应的“bootstrap–canary”机制。

**📊 数据集**

使用的数据集为：①Mechanical MNIST Challenge 的 22 号样本（二维三位元板）；②TT1 试验的凹槽膜（二维三位元，含 DIC 位移和力数据）；③自定义的三维立方体含两球相变（合成数据）。

**📈 对比分析**

通过与全阶识别的对比验证方法。结果显示：在 MNIST 案例中，全阶训练 9 小时降至 0.69 小时，平均每轮时间从 33.4 秒降至 1.2 秒，速度提升约 13 倍；在 TT1 案例中，全阶 2.73 小时被压缩到 2.78 小时，单轮时间提升约 10 倍；在 3D 案例中，全阶 4.5 小时压缩到 0.21 小时，单轮提升约 20 倍。准确度方面，力曲线误差均维持 1–2% 以内，位移误差在 3–6% 范围内，且在未参与训练的测试样本上表现一致。

**⚠️ 局限性**

局限性包括：①自适应更新需要周期性全阶求解（canary 步），导致总体时间受限；②当变形高度局部化（如 TT1 的孔洞）时，ECSW 需要更多活跃元素，稀疏程度下降；③识别结果受限于实验数据的感知能力，导致体积刚度、等压变形等特性在未直接约束时不够准确；④当前实现主要面向等向性、可压缩材料，非等向性或多相耦合问题仍需进一步扩展。

---

## 537. Predictable performance is suboptimal, and what it means for model routing

**arXiv ID:** 2610.31428 | [PDF](https://arxiv.org/pdf/2610.31428v1)

**作者:** Guanhua Zhang `[一作]`, Moritz Hardt `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大语言模型在路由与预测中的自信度与子最优性关系，提出通过性能预测提升模型答案损失并证明其受子最优性与预测器容量限制；

**💡 创新点**

证明任何超过自信度的性能预测可等效转化为答案损失改进，并给出子最优性与额外容量对预测与路由收益的上界；

**🔧 技术方法**

使用自信度重校准、线性头预测器、LoRA微调、Qwen3-Embedding-4B预测模型、两模型级联路由、Platt校准等技术；

**📊 数据集**

四大基准：WinoGrande、CommonsenseQA、GSM8K、MMLU-Pro，涉及16个7–12B参数语言模型；

**📈 对比分析**

与自信度路由在等规模的提升率下比较，实验显示子最优性越高的模型预测与路由收益越大；微调后收益下降；

**⚠️ 局限性**

理论给出上界而非下界，未说明强模型错误如何影响路由优势，且实验仅限于选定任务与模型规模，未评估更大规模或多模型级联场景。

---

## 538. FastHarness: Experience-Driven Review Scheduling for Vision-Language-Action Policies

**arXiv ID:** 2610.31427 | [PDF](https://arxiv.org/pdf/2610.31427v1)

**作者:** Tao Lin `[一作]` (Shanghai Jiao Tong University), Bo Zhao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种经验驱动的执行 harness，利用 GPT-6 Astra 对固定的 Vision‑Language‑Action (VLA) 机器人策略进行自适应监督，既在常规执行中减少审查频率，又在错误发生时提供局部恢复，最终提升任务完成率。

**💡 创新点**

创新点在于：① 引入阶段经验记忆与动态审查调度的 fast‑path，让机器人在熟练阶段自动完成动作；② 设立 recovery‑path，使得错误时能够持续审查并引导局部纠正；③ 通过记录并利用跨 episode 的经验，显著降低 GPT 调用量，同时保持甚至提升成功率。

**🔧 技术方法**

使用技术包括：固定的 VLA 预训练策略；GPT‑6 Astra 作为审查模型；经验驱动的调度算法（阶段匹配、成功/失败计数、间隔函数）；局部恢复策略（重新定位、验证 waypoint）；内存链（跨 episode 的经验存储）。

**📊 数据集**

实验数据集为四个仿真操纵任务：Build tower、Classify objects、Bottle disposal、Fold clothes，在每个任务下使用多种场景布局构成多条 episode。

**📈 对比分析**

与 GPT‑as‑Policy 基线进行比较，评估指标包括 GPT token 消耗、episode 完成时间和任务成功率。结果显示：token 使用平均下降约 50%（从 1.83M 降至 0.93M），episode 时间下降 39%（85.5→52.0 分钟），平均任务成功率从约 80% 提升至约 96%，提升幅度约 17%。

**⚠️ 局限性**

局限性包括：① 仅在仿真环境验证，缺乏真实机器人实验；② 需要昂贵的 GPT‑6 Astra 调用，成本高；③ VLA 策略保持固定，无法在 harness 内部进行学习；④ 对阶段识别和错误检测的准确性高度依赖，误判可能导致错误恢复失败；⑤ 经验积累需要多次执行，适用性受限于任务可重复性。

---

## 539. FocusVLA: Future-Intent Guided Dynamic View Routing for Vision-Language-Action Models

**arXiv ID:** 2610.31423 | [PDF](https://arxiv.org/pdf/2610.31423v1)

**作者:** Huixin Xu `[一作]` (Institute of Software Technology), Peng Li `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 FocusVLA 框架，利用未来交互意图预测实现多视角 VLA 模型的阶段性动态视图路由，从而提升机器人在多任务中的适应性。

**💡 创新点**

创新点在于将未来意图作为可学习的阶段信号，生成步骤级视图权重并在动作生成侧进行动态调度，无需改动现有 VLA 背骨，从而突破静态多视角融合的视角惯性问题。

**🔧 技术方法**

主要技术包括交叉注意力的未来意图桥、步骤级路由权重与 chunk 级意图 token、残差注入的动作生成模块、以及 RDT‑style 随机视图丢弃的对比实验。

**📊 数据集**

使用了 50 个 RoboTwin 2.0 任务的模拟演示数据以及六个真实机器人任务（Piper arm + 两台 RealSense 摄像头）的真实环境数据。

**📈 对比分析**

在与 RDT、π_0、π_0.5 等基线在相同演示数据与训练步骤下进行比较时，模拟平均成功率从 66.82% 提升至 76.92%（+10.10pp），真实场景成功率从 71.7% 提升至 85.0%（+13.3pp），特别是在中长/长任务中表现尤为显著。

**⚠️ 局限性**

局限性包括仅使用未来意图作为进度表示，缺乏更丰富的任务进展信号；在跨机器人、跨摄像头和更复杂真实环境的泛化尚待进一步验证。

---

## 540. Certified Execution Harness for Agentic AI: Safe and Least-Intrusive Petri-Net Supervisory Control Between Reasoning and Action

**arXiv ID:** 2610.31376 | [PDF](https://arxiv.org/pdf/2610.31376v1)

**作者:** Chao Gu `[一作]` (Queen's University Belfast), Quanyan Zhu `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文将大语言模型（LLM）代理的执行框架抽象为Petri网的监督控制问题，利用可消耗授权来表示批准、预算等资源，并在四个假设下证明在模型上安全的监督器能转移到真实系统；随后提出“重规划闭环”，将拒绝造成的推理成本纳入考量，证明传统最大可允许监督器可能极其侵入，给出基于提议模型的最小侵入监督器为阈值修正；进一步分析两条可并行执行的动作在此阈值下的排序；并在自动发布代理的Petri网模型上进行合成实验，验证安全部署需要可被拒绝的步骤，并通过仿真评估注入攻击时不同架构的安全性与耗费。

**💡 创新点**

创新点在于：①使用Petri网与可消耗授权统一描述LLM代理的资源与审批流程；②在监督控制框架下引入重规划闭环，将拒绝成本纳入最优控制；③证明最大可允许监督器不一定是最小侵入的，并给出阈值修正的最优形式；④在并发动作（fork）结构上推导阈值规则，说明可通过单一监视器位置实现优先级决定；⑤在实际发布代理模型上验证理论并分析注入攻击下的表现。

**🔧 技术方法**

主要技术手段包括Petri网建模与可消耗授权、Ramadge‑Wonham监督器合成、状态反馈监督器、动态规划（Bellman方程）与价值迭代、最短路径与线性规划求解、概率模型（稳态提议者）与重规划闭环的随机控制。

**📊 数据集**

本文未使用公开数据集，而是构造了一个自动发布代理的Petri网模型（20个地点、18个转换），并在该模型上进行合成与仿真实验。

**📈 对比分析**

通过与传统最大可允许监督器对比，证明后者在重规划闭环下可能导致无穷大或显著高的拒绝/推理成本；在发布代理实验中，最小侵入监督器在面对注入攻击时保持零未授权部署率，且与控制器或分类器架构相比在完成率上更优，整体成本（拒绝次数）仅为常数级。

**⚠️ 局限性**

局限性包括：需完整且精确的Petri网模型，四个假设（A1‑A4）必须满足；监督器假设对标记完全可观测；提议者模型为稳态且不可泄露给代理；仅考虑无记忆（无历史）状态反馈；未处理部分观测或动态环境变化；对大规模模型的可扩展性与多代理协作的组合性尚未实现。

---

## 541. ClinGuard: a governance framework for reliable, transparent, and auditable clinical coordination

**arXiv ID:** 2610.31354 | [PDF](https://arxiv.org/pdf/2610.31354v1)

**作者:** Kaiyuan Wu `[一作]` (Duke University), Rishikesan Kamaleswaran `[通讯]` (Duke University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 ClinGuard 框架，将临床协调过程视为可验证、可审计的事务，并在此框架下实现了模型无关的治理层。

**💡 创新点**

创新点在于将语言模型的输出通过四个哨兵功能（监控、验证、治理、升级）进行逐步检查，并将每个事务绑定到可验证的源数据和版本化的政策，形成可审计的决策记录和有限特权执行。

**🔧 技术方法**

技术包括基于 FHIR 的 SMART‑on‑FHIR 接口、有限权限连接器、可审计的加密日志链、规则引擎（ClinRule）与概念解析器，以及闭环监测的流量/漂移/安全指标。

**📊 数据集**

使用的数据集为 MIMIC‑IV‑FHIR（含模拟的 Synthea 生成的事务）以及已知的技能性护理设施放置流程数据。

**📈 对比分析**

对比方法：在负载测试中通过 Locust 生成多并发请求，测得从 2 到 66.4 TPS，治理层占总时间 <1%；在故障注入实验中，非合规事务的拦截率达到约 90%。

**⚠️ 局限性**

局限性包括依赖 SMART‑on‑FHIR 写回权限、对自然语言推理的不确定性、模型替换后的回归验证、以及在不同 EHR 环境下的可移植性与维护成本。

---

## 542. AgenticEvolve: Evolutionary Program Search with Persistent Language Model Agents that Design and Run Experiments

**arXiv ID:** 2610.31326 | [PDF](https://arxiv.org/pdf/2610.31326v1)

**作者:** Ze-Yu Peng `[一作]` (University of Chinese Academy of Sciences), Yun-Song Piao `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AgenticEvolve框架，利用持久化大型语言模型代理在岛屿上进行进化程序搜索并在正式评估前进行自定义实验，显著提升程序质量。

**💡 创新点**

创新点在于将持久化LLM代理与传统进化算法结构（种群、档案、迁移）结合，使代理能在正式评估前主动生成实例、执行实验并本地优化，从而大幅减少正式评估次数并提升泛化性能。

**🔧 技术方法**

采用大型语言模型（DeepSeek‑V4.1‑Flash、GLM‑5.3‑Flash、Qwen3.8‑Flash）作为变异算子；利用工具调用实现代码编辑、实验脚本执行、正式评估提交；使用MAP‑Elites档案与岛屿迁移实现多样性与信息共享。

**📊 数据集**

在两类启发式设计任务（在线箱子装载、动态作业车间调度）以及十二个符号回归基准（LLM‑SRBench）中进行实验。

**📈 对比分析**

与“无实验”（No‑analysis）、OpenEvolve以及LLM‑SR、SR‑Scientist、PySR等基线对比。结果显示AgenticEvolve在所有模型下在隐藏测试集和未见实例上都取得更优质量，仅需更少的正式评估次数，符号回归方面则恢复更多基准方程，成本约为OpenEvolve的一半到四倍。

**⚠️ 局限性**

局限包括仅验证两类启发式任务和十二个方程基准；未单独评估迁移与档案对性能的具体贡献；强模型（非Flash级别）下实验效果未知；在符号回归中仅跑一次每种方法，缺乏复现性与方差评估。

---

## 543. Consistent but Biased: Chair Personas Trade Noise for Bias in Multi-Agent Clinical Decisions

**arXiv ID:** 2610.31325 | [PDF](https://arxiv.org/pdf/2610.31325v1)

**作者:** Yitong Bai `[一作]` (Columbia University), Ziyang Wang `[通讯]` (Aston University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过在固定的20例颈椎退行性病变病例中，模拟多智能体临床决策团队，研究了主席角色的persona对决策时机和一致性的影响。

**💡 创新点**

创新点在于将人物角色的persona分两种形式（规则式和特质式）系统评估其对决策结果、时机和多次运行一致性的影响，并提出在多智能体系统中应报告和调整主席persona。

**🔧 技术方法**

使用多智能体LLM系统，基于DeepSeek deepseek-v4-pro和OpenAI gpt-5.6-terra模型，构建模拟部门并实现自动化讨论与决策流程。

**📊 数据集**

使用从PMC-Patients公开数据集抽取的20个经手术或观察的颈椎退行性病变病例的临床摘要。

**📈 对比分析**

通过对比不同persona下的决策比例、手术时机、以及三次运行的一致性（包括原始一致率和机会校正后Kappa），发现autocratic和avoidant persona显著提高原始一致率但对机会校正一致率无提升，且在手术时机上差异达90%与2%。

**⚠️ 局限性**

局限包括样本量小、病例来源可能在模型训练集内、使用LLM评估红旗处理而非人工评估、仅评估两种模型、以及未单独评估主席或副主席单独的persona效应。

---

## 544. Stream4D: Streaming 4D Reconstruction with Training-free Motion Decoupling via Causal KV-Cache Attention

**arXiv ID:** 2610.31320 | [PDF](https://arxiv.org/pdf/2610.31320v1)

**作者:** Wei Zhang `[一作]` (Northwestern Polytechnical University), Qi Zhang `[通讯]` (vivo Mobile Communication Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练无关的、可插拔的 4D 现场重建框架 Stream4D，能够在实时 causal Transformer 中解耦动态物体与静态场景，避免长序列中出现漂移和鬼影现象。

**💡 创新点**

创新点在于利用 KV‑Cache 中的自注意力动态信息自适应聚合生成动态掩码，并在 KV‑Cache 上施加软键屏蔽，从而在不需要额外训练或网络改造的前提下实现动态-静态分离。

**🔧 技术方法**

核心技术包括多层注意力统计聚合、深度先验融合、软键掩码机制，以及在现有 causal Transformer（如 StreamVGGT、Stream3R）中的无缝集成。

**📊 数据集**

使用 DAVIS‑2016/2017/All 数据集评估动态分割；使用 DyCheck 数据集评估相机位姿估计和 4D 重建。

**📈 对比分析**

与现有 streaming 基线（CUT3R、Stream3R、StreamVGGT）及 offline 强基线（VGGT4D、Easi3R）比较，Stream4D 在动态分割的 IoU 方面提升 8–10 点，在位姿误差（ATE、RTE、RRE）上相较 Stream3R 分别下降 6.7%、5.6%、1.1%，同时保持近线性计算复杂度且无 OOM 问题。

**⚠️ 局限性**

局限性在于动态掩码仅依赖注意力统计和深度先验，可能难以捕捉复杂运动或纹理缺失区域；当前框架只关注静态场景重建，未对动态物体进行建模与重建。

---

## 545. SurveyGraph: Generating Academic Surveys via Methods-Entity Knowledge Graphs

**arXiv ID:** 2610.31317 | [PDF](https://arxiv.org/pdf/2610.31317v1)

**作者:** Yifan Yuan `[一作]` (Shenzhen University), Xiaogang Peng `[通讯]` (Shenzhen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了 SurveyGraph 框架，用方法–实体知识图（MEKG）实现从检索、结构化大纲到叙事合成的全流程自动综述生成。

**💡 创新点**

创新点在于：① 将 typed 关系的 MEKG 作为统一骨干，支持核心论文发现、图检索、混合大纲生成与进化叙事合成；② 引入证据门控（AUC 0.75）过滤噪声边；③ 通过图结构与传统检索融合实现更高的召回与结构密度。

**🔧 技术方法**

技术包括：图检索 Graph‑RAG、HDBSCAN+LLM 结构规划、PageRank+加权评分、LLM 推理（Qwen、GTE、E5 等）、语义边推理与证据门控、以及多级评估与图度量（WCR、CC、SC）。

**📊 数据集**

使用公开学术元数据与论文文本（OpenAlex、Google Scholar 等），在 7 个主题（如 LLM 功能调用、文档 QA、内存 AI 代理、城市空域问答、Text‑to‑SQL 代理等）上构建 MEKG（约 2k+关系、143 篇论文平均）并进行实验；亦引用 SurveyBench、SurveyForge 作为集合评估基准。

**📈 对比分析**

与平面检索、Biblio+TF‑IDF、GraphRAG、MEKG typed 等七种方法在等预算基准中对比，MEKG fusion 在召回与关系密度上显著优于其他方法；图检索相较平面检索召回翻倍、关系密度提升；证据门控 AUC 0.75，人工验证 68.5% 边被支持；引用覆盖 0.521；同端 Qwen 控制实验显示混合大纲与进化分析提升结构一致性与评估分数。

**⚠️ 局限性**

局限性包括：仅完成集合阶段评估，未进行完整 SurveyBench 端到端提交；语义边验证仅覆盖 3 种关系、单一领域，人工一致性有限；与 MVSS、StructSurvey 的对比非同端同模型；图指标仅为诊断工具；数据依赖公开元数据，可能存在缺失或偏差。

---

## 546. PhysTrace: Interpretable Temporal Evaluation of Physical Consistency in Video Generative Models

**arXiv ID:** 2610.31429 | [PDF](https://arxiv.org/pdf/2610.31429v1)

**作者:** Taekoan Yoo `[一作]` (Independent Researcher), Kyeongbo Kong `[通讯]` (Pusan National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了PhysTrace，一种无训练的框架，通过将多模态LLM生成的历史条件预期状态与视频窗口对齐，实时跟踪物理一致性。

**💡 创新点**

创新点在于利用多模态嵌入对齐实现时序物理违规定位，并构建IntPhys2-Timing数据集提供违规时间注释，提供可解释的预期与观测对比。

**🔧 技术方法**

技术手段包括：多模态LLM推理、文本与视频的共享嵌入对齐（如PE-AV-Large）、噪声级逆向重建的扩散/流模型输出、时间窗口切分、选择性历史保留。

**📊 数据集**

使用的数据集为IntPhys2-Timing（在IntPhys2基础上加时间违规注释）和原始IntPhys2，同时对Wan2.1与Cosmos-Predict2生成的重建视频进行评估。

**📈 对比分析**

方法对比采用V-JEPA2基于VoE的基线以及多种LLM模型，评估指标为Hit Rate（时序定位）和MAS（视频级一致性）；在大多数属性上Hit Rate最高可达70.7%，整体分类准确率59.9%，优于现有基线。

**⚠️ 局限性**

局限性在于对LLM的上下文推理与提示设计高度依赖，易受LLM能力、时间对齐不准或视觉细节不足的影响，且在高度模糊或短暂事件上定位精度有限。

---

## 547. Major Revision, Minor Accountability: LLMs as Reviewers of Published Articles in Top-Tier Journals

**arXiv ID:** 2610.31414 | [PDF](https://arxiv.org/pdf/2610.31414v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 548. How Depth Growth and Looping Shape In-Context Learning in Language Models

**arXiv ID:** 2610.31421 | [PDF](https://arxiv.org/pdf/2610.31421v1)

**作者:** Ferdinand Kapl `[一作]` (Technical University of Munich), Johannes von Oswald `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了通过深度增长（depth‑growth）和循环共享（looped）来重用 Transformer 层的计算，探究其对上下文学习（ICL）的影响，并通过注意力头机制分析其实现原理。

**💡 创新点**

创新点在于：①首次系统比较深度重用模型与密集模型在多项 ICL 任务上的性能差异；②揭示深度增长会将早期强势的前缀匹配（induction）头沿层复制，从而在后续层中形成重复模式；③通过头部消融实验验证成长与循环模型对 induction 头的依赖性明显高于密集模型。

**🔧 技术方法**

技术方法包括：深度增长和循环 Transformer 结构；前缀匹配（PS）和函数向量（FV）评分用于识别重要注意力头；对这些头进行 1–20% 消融；对 token‑loss ICL‑Score、GSM8K 以及 RegBench 等指标进行评估；使用 Omniglot 图像分类任务来区分 ICL 与权重学习（IWL）的贡献。

**📊 数据集**

主要数据集：FineWeb‑Edu（token‑loss 评估）、GSM8K（少样本推理）、RegBench（未知形式语言推断）、Omniglot（图像分类）以及 100B 语料库用于预训练。

**📈 对比分析**

与 360M–3B 的密集 Transformer 进行对比。结果显示：
• 成长与循环模型在早期‑后期 token‑loss 的减少幅度更大，GSM8K 准确率提升 5–7%；
• 在 RegBench 上对新 DFA 的 TVD 误差更低，优势在前 4 个字符串即可显现；
• 在 Omniglot 中，成长与循环模型在固定标签与随机标签设置下都保持更高的 ICL 准确率，同时 IWL 准确率更低，表明更倾向于上下文学习；
• 消融实验表明，成长与循环模型对 induction 头的依赖性高于密集模型，消除这些头会显著下降 ICL 及 token‑loss 表现。

**⚠️ 局限性**

局限性：
①实验规模仅至 3B 参数，尚未验证更大模型的可扩展性；
②主要聚焦于语言模型与图像分类的基准，缺乏对更广泛任务（如多模态或长期记忆任务）的评估；
③对深度重用机制的内部动力学解释仍不完整，尤其是循环模型中共享头的具体功能尚需进一步分析；
④实验中使用的超参数（如层块大小、循环次数）未系统调优，可能影响结果的普适性。

---

## 549. Post-Hoc Feature Selection Layer for Neural Networks Interpretability

**arXiv ID:** 2610.31420 | [PDF](https://arxiv.org/pdf/2610.31420v1)

**作者:** João Kenji Suwa `[一作]` (UFRGS), Bruno Iochins Grisci `[通讯]` (UFRGS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种后置特征选择层（Post-hoc FSL），在冻结的预训练神经网络上训练，能够在不改变原模型权重的前提下对输入特征赋权，提升可解释性并保持甚至提升预测性能。

**💡 创新点**

创新点在于将FSL从嵌入式训练迁移为后置训练模块，支持任何冻结的预训练网络，并通过轻量级可训练层快速收敛，兼顾解释性与效率。

**🔧 技术方法**

使用技术包括轻量级可训练的特征选择层、ReLU+L1正则化、基于原模型输出的梯度反向传播、加权t‑SNE、Jaccard/相关性稳定性评估以及多种后置解释方法对比。

**📊 数据集**

数据集涵盖了合成数据（XOR、SynthA）、公开邮件垃圾分类数据集以及高维微阵列癌症数据集（Liver-GSE22405、Breast-GSE45827）以及TabPFN预训练模型。

**📈 对比分析**

通过与原始FSL、基线模型以及多种后置方法（Integrated Gradients、Noise Tunnel、DeepLIFT、Gradient SHAP、Feature Ablation）在准确率、F1、可视化分离度和稳定性指标上进行比较，Post-hoc FSL在大部分数据集上实现了与或超过基线的预测性能，同时在可视化分离度和部分稳定性指标上优于或相当于其他方法。

**⚠️ 局限性**

局限性包括在多分类任务上表现不佳、对局部类特征关联捕捉不足，以及仅针对固定含义的表格特征设计，难以直接迁移至图像等其他领域。

---

## 550. SyRA-CS: Synthetic-to-Real Adaptation for Code-Switched ASR with Meta-Learning

**arXiv ID:** 2610.31415 | [PDF](https://arxiv.org/pdf/2610.31415v1)

**作者:** Fadel Thior `[一作]` (Mohamed bin Zayed University of Artificially Intelligence), Thamar Solorio `[通讯]` (Mohamed bin Zayed University of Artificially Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了 SyRA‑CS，基于 Reptile 的 pair‑specific 迁移学习框架，用少量真实代码切换语音（1 小时）快速适配合成代码切换语音训练得到的初始化。

**💡 创新点**

创新点在于将合成语音与少量真实语音结合，利用 Reptile 的 first‑order meta‑学习实现对每个语言对的专属适配初始化，显著优于传统连续微调。

**🔧 技术方法**

核心技术包括 Reptile meta‑学习、编码器‑CTC、FastConformer‑RNN‑T、AED、Speech‑LLM 等四种 ASR 架构，以及针对不同语言对的 pair‑specific 训练。

**📊 数据集**

使用了 CS‑FLEURS 合成代码切换数据（Hindi‑English、French‑English、Arabic‑English、German‑English）以及对应的真实数据集：MUCS（Hindi‑English）和 SwitchLingua（其余三对）。

**📈 对比分析**

与 synthetic‑only transfer、real‑only fine‑tuning、continual fine‑tuning（CFT）等基线比较，SyRA‑CS 在所有 16 个模型–语言对组合中均取得最低 WER/CER，宏观 WER 降幅从 5.3% 至 17.7%。

**⚠️ 局限性**

限制包括：仅进行 pair‑specific 适配，未探索跨语言或多 seed 的稳健性；合成与真实数据量的最优比例未系统评估；仅使用四个单一模型，难以归因于架构差异；未与其他 meta‑学习算法对比。

---

## 551. AgentPhishing: A Benchmark of LLM Email Agents Under Adaptive Attack

**arXiv ID:** 2610.31404 | [PDF](https://arxiv.org/pdf/2610.31404v1)

**作者:** Justin Tan `[一作]` (Carnegie Mellon University), Ding Zhao `[通讯]` (Carnegie Mellon University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出AgentPhishing基准，评估大型语言模型驱动的邮箱代理在面对社会工程钓鱼邮件时的检测与合规行为，并通过PhishOpt技术基于代理反馈迭代生成适应性攻击；

**💡 创新点**

创新点在于（1）双重评估指标—检测与恶意请求合规；（2）基于目标代理反馈的生成优化框架PhishOpt；（3）利用Cialdini说服原则构建通用攻击技能，弥补传统钓鱼检测的盲区；

**🔧 技术方法**

技术包括：基于SkillOpt的目标特定技能优化、LLM生成与评估、仿真评估框架、Cialdini说服分析、以及对比实验与诊断工具；

**📊 数据集**

使用了七大公开钓鱼邮件数据集（TREC-05/06/07、CEAS-08、Enron、Ling-Spam、SpamAssassin），以及PhishOpt生成的自研合成邮件；

**📈 对比分析**

方法通过与公开数据集、未优化合成邮件、目标特定优化邮件和通用技能在相同实验协议下的对比，发现原始模型对公开钓鱼检测表现良好，但在PhishOpt生成的适应性攻击下检测率低（<10%）且合规率高（>45%），并揭示跨代理敏感度显著差异；

**⚠️ 局限性**

局限性包括：仅评估单轮单邮件、固定生成约束、未涵盖多轮对话、结果受模型版本影响、真实执行未测、生成邮件未公开、Cialdini关联非因果、并且仅在有限代理上验证。

---

## 552. Auditing the Hand-Written Inputs of Learned Access Control

**arXiv ID:** 2610.31388 | [PDF](https://arxiv.org/pdf/2610.31388v1)

**作者:** Arsalan Jalil Sarghaleh `[一作]` (Sharif University of Technology), Mehdi Hazrati `[通讯]` (University of Victoria)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对学习式访问控制系统中硬拒绝集与区域映射这两类手工构造的 artefact 进行审计与验证，评估其在安全性与性能上的影响。

**💡 创新点**

创新点：① 通过形式化与穷举检验硬拒绝集的声音性与空洞性，量化文本导出 deny 规则的不安全率；② 揭示混合路由所需的两类规则集与传统矿工输出不匹配，并证明基于置信度的排名导致“一类化”；③ 给出混合路由对加性损失的影响公式，指出单一指标报告可能产生误导。

**🔧 技术方法**

技术手段：形式化验证、穷举枚举、符号求解、基于混合最小化的规则挖掘、噪声实验、合成与真实策略实验、分布式 conformal gate、统计误差分析。

**📊 数据集**

数据集：公开的五个 ABAC Lab 政策（E-Document、Workforce、University、Project Mgmt、Healthcare），自构造的实验政策，以及多种噪声模型和多语言模型生成的规则。

**📈 对比分析**

对比方法：比较已检验与未检验的硬拒绝集在误拒/误准方面的差异；对不同规则选择策略在混合路由中对两种加性损失的对比。结果显示未检验的 deny 集误拒 16–78% 的授权，常规排序大多导致一类化，只有 class-aware 选择在两种损失下均优。

**⚠️ 局限性**

局限性：仅评估公开政策且样本有限，未覆盖真实生产日志；语言模型 elicitor 仅来自同一家供应商；对噪声模型和学习器配置的敏感性；未覆盖完整系统安全边界（如攻击时序、资源耗尽等）。

---

## 553. Pipelining vs. Combining in Aggregation Tasks

**arXiv ID:** 2610.31383 | [PDF](https://arxiv.org/pdf/2610.31383v1)

**作者:** Gabriel Marques Domingues `[一作]` (Tel Aviv University), Boaz Patt-Shamir `[通讯]` (Tel Aviv University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文研究在网络中进行聚合（convergecast）任务时，数据值超过单条消息容量的情况，探讨序列化与并行化的权衡，给出通用下界，并在超立方体、完全图和完全二分图等常见拓扑上分别设计并证明最优或近似算法。

**💡 创新点**

创新点在于：① 提出了针对可拆分与不可拆分流模型的统一下界；② 在超立方体拓扑中利用汉明码的递归结构和 3‑分离性质，构造出 O(s+d)（可拆分）和 O(sd/ log d + d)（不可拆分）的聚合算法；③ 对一般图给出 O(log n) 的近似方案，首次将树分层分离器的思想与聚合任务结合。

**🔧 技术方法**

主要技术包括：流模型下的液体化简（fluid model）与信息重量分析、汉明码的递归构造与 3‑分离性质、并行路由技巧（利用 3‑分离的节点集合实现多路径并行传输）、树分层分离器（Jordan 分离器）以及自底向上的树形计算结构。

**📊 数据集**

本工作为理论研究，未使用任何实际数据集；所有结论均基于图论和信息论分析。

**📈 对比分析**

与传统单消息聚合方案相比，本论文的算法在可拆分流下实现了线性时间与维度的优雅平衡（O(s+d)），在不可拆分流下则突破了先前的 O(s log n) 下界，达到 O(sd/ log d + d)。在一般图中，O(log n) 的近似算法也显著优于直线性扩展。

**⚠️ 局限性**

局限性包括：① 对可拆分流模型下一般图的近似比率仍未达到常数，存在开放问题；② 在树结构中仅提供了 O(log n) 的上界，尚未证明是否存在常数近似；③ 论文假设所有链路带宽均为 1，未考虑非均匀带宽情形，未来研究可扩展此模型。

---

## 554. First hardware demonstration of a continuous-state probabilistic computer with programmable energy-defined sampling

**arXiv ID:** 2610.31339 | [PDF](https://arxiv.org/pdf/2610.31339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 555. Probabilistic Energy Modeling Unifies Versatile Swarm Control

**arXiv ID:** 2610.31371 | [PDF](https://arxiv.org/pdf/2610.31371v1)

**作者:** Xiang Zhou `[一作]` (Sun Yat-sen University), Qingrui Zhang `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出基于概率能量建模的统一无人机群控制框架，能够在不重训练的情况下完成跨任务、跨规模、跨平台的多任务协作；

**💡 创新点**

创新点在于将聚合、碰撞规避、速度对齐、形态控制等多种群行为编码为可复用的Gibbs能量项，并通过轻量级策略门控实现多任务无缝切换；

**🔧 技术方法**

使用了Gibbs随机场能量最小化与奖励最大化对应的理论、注意力机制的邻居信息聚合、去中心化强化学习（PPO）以及多任务门控网络；

**📊 数据集**

实验数据集包括大规模仿真（10–100 UAV，10–50障碍）和真实平台（29架Crazyflie微四旋翼及自制四旋翼）在室内、户外及GPS禁区环境下的六类任务；

**📈 对比分析**

与规则驱动、优化驱动和单任务强化学习控制器对比，统一策略在10–100机群、10–50障碍的情况下保持0.96–1.00的任务成功率，单机推理时间1–10 ms，速度对齐0.74–0.84，且计算复杂度与机群规模无关；

**⚠️ 局限性**

局限性包括缺乏严谨的安全性证明、对局部感知和通信的依赖、实验规模受限于16架真实机器人、未在极端传感/通信失效或异构团队上验证，以及注意力归因未证明对性能的因果影响。

---

## 556. Sum-rank analogues of Reed--Muller codes over Abelian Galois extensions

**arXiv ID:** 2610.31358 | [PDF](https://arxiv.org/pdf/2610.31358v1)

**作者:** Alessandro Neri `[一作]` (Università degli Studi di Napoli Federico II), Giuseppe Zampella `[通讯]` (Università degli Studi di Napoli Federico II)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并分析了一类新的多变量“sum‑rank Reed–Muller 型”码，利用分裂域上的非可交换多项式商环构造，给出了其维数、最小 sum‑rank 距离和生成矩阵的显式表达式；同时把该类码与已有的 sum‑rank、rank、Hamming 以及线性化 Reed–Muller 码等进行对比。

**💡 创新点**

创新点在于：①把 skew 多项式商环与 norm‑grid 评价映射结合，得到一种统一的生成方法；②通过“下集”支撑（lower‑set）理论，直接给出最小距离的精确公式；③在一般有限阿贝尔 Galois 扩展上（不仅限于循环扩展或有限域），构造了新的码族，并证明其在某些参数下达到 sum‑rank Singleton 极限；④将 Berardini–Caruso 的线性化 Reed–Muller 码表述为该框架中的特例，从而给出了其维数和距离的完整说明。

**🔧 技术方法**

主要技术包括：非可交换（skew）多项式环的右除法与 Dickson 矩阵、norm‑grid 的投影与评价映射、下集支持的组合计数、对称多变量分解以及 Hilbert 90 及其在评价映射中的应用。

**📊 数据集**

论文未使用实验数据集；所有结果均为理论推导与符号计算。

**📈 对比分析**

通过与已知码族（线性化 Reed–Solomon、σ‑Reed–Muller、affine Cartesian 码、Gabidulin 码、Reed–Solomon 码）在维数、长度、最小距离等指标上的直接比较，展示了新码在相同参数下维数相同、最小距离相当甚至更优，且在满足下集条件时能达到 sum‑rank Singleton 极限。

**⚠️ 局限性**

局限性包括：①若支撑不是下集，则当前理论无法给出精确最小距离；②主要针对有限域与有限阿贝尔 Galois 扩展，无法直接推广到无限域或非阿贝尔群；③在某些参数下，构造的码族与 Berardini–Caruso 的代码并不完全等价，尤其是对块的选择与评价点的限制。

---

## 557. From Signals to Evidence: Interpretable Underwater Acoustic Target Sensing with Large Language Models - Part II: Passive Detection

**arXiv ID:** 2610.31346 | [PDF](https://arxiv.org/pdf/2610.31346v1)

**作者:** Zhiyuan Li `[一作]` (Northwestern Polytechnical University), Yixin Yang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了基于物理可解释的多通道声学编码器与大语言模型相结合的被动目标检测框架，能够从多通道海洋记录中生成结构化的检测报告。

**💡 创新点**

创新点在于：①将物理特征转化为可验证的“证据”token，并用LLM进行多证据推理；②将 Part I 的识别框架扩展到检测任务；③构建物理参数化的模拟引擎，提供可追溯的监督数据。

**🔧 技术方法**

采用 MVDR 波束成形、时频分析、多尺度 CNN、注意力聚合等物理感知技术，并将输出对齐至 Qwen3‑4B‑Instruct‑2507 大语言模型，使用 LoRA 进行低秩适配；同时使用 BELLHOP 进行海洋传播仿真。

**📊 数据集**

使用自行生成的模拟数据集：96 个目标几何点、8 个信号实例、白噪声与混合海洋背景；每条记录 120 s、2 kHz、多通道（32 声道）并覆盖 SNR 0 dB 至 -30 dB。

**📈 对比分析**

通过与单独训练的检测器在白噪声和混合背景下进行对比实验。混合背景下检测率 100%、误报率 0%；方向平均误差 0.21°；频率 F1 53.1%；接收 SNR MAE 0.63 dB；水面到目标距离 MAE 8.8 km。

**⚠️ 局限性**

局限性包括：仅在静止目标和模拟环境下验证；未在真实海洋现场数据上评估；频率检测召回率低，受 SNR 影响；对不同阵列配置和传播条件的鲁棒性尚不清楚。

---

## 558. ValuSpec: Plug-and-Play Candidate Valuation before Target Verification for Tree-Based Speculative Decoding

**arXiv ID:** 2610.31338 | [PDF](https://arxiv.org/pdf/2610.31338v1)

**作者:** Liang He `[一作]` (Tongji University), Xilu Wang `[通讯]` (University of Surrey)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 ValuSpec——一种在候选树生成后、目标模型验证前进行插件式候选分值评估的机制，用于过滤低质量分支，从而显著提升树式推测解码的吞吐量。

**💡 创新点**

创新点在于：① 将候选生成与验证分离，专注于验证前的分支筛选；② 采用转移监督（Transfer-based Supervision）仅利用目标模型的贪婪轨迹校准阈值，无需额外训练；③ 保证输出一致性（output‑identical greedy rule），实现训练无关、即插即用的加速方案。

**🔧 技术方法**

技术细节包括：树式推测解码（Speculative Decoding）、轻量级评分模型 q_ϕ、基于贪婪轨迹的阈值校准、输出相同的贪婪验证规则、并行树形前向验证（tree‑masking/attention）以及跨模型兼容的插件接口。

**📊 数据集**

使用的数据集有：HumanEval（代码生成）、GSM8K（算术推理）、Dolly（指令跟随）、SpecBench（多任务推测解码基准），并进行跨数据集阈值迁移实验。

**📈 对比分析**

与 AR（传统自回归）、SD、Retrieval‑SD、EAGLE‑3 等基线进行对比；在 A100 上实现 2.18×–2.38× 的速度提升；与 EAGLE‑3 集成后额外 1.35×–1.52× 加速；跨数据集阈值迁移保持 2.05×–2.35×；批量推理场景亦保持显著加速。

**⚠️ 局限性**

局限性：仅针对贪婪解码（output‑identical）；若候选树已极为精简，提升空间有限；需额外的评分步骤，虽然轻量但在大规模生产环境中仍需进一步集成与性能评估；对非贪婪或随机解码场景的适用性未验证。

---

## 559. Bridging the Data Gap: Source-Informed Task-Space Transport for Few-Shot Diffusion

**arXiv ID:** 2610.31323 | [PDF](https://arxiv.org/pdf/2610.31323v1)

**作者:** Jinjian Jiang `[一作]`, David Mguni `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对Office-Home数据集进行源域（Product）向目标域（Clipart）的迁移学习，使用基于扩散模型的线性传输方法，并与错误源、仅关系、源重用、目标仅以及全微调等多种基线进行对比。

**💡 创新点**

提出了“correct source + transport”策略，在零目标图像条件下实现低成本迁移，并系统评估了SW、MMD、请求类准确率等多指标，证明该方法在某些指标上优于基线，但未实现语义迁移。

**🔧 技术方法**

使用16维线性传输、FiLM编码、64×64 U-Net、DDIM 50步、AdamW优化、EMA、DINOv2特征评估、CLIP请求类比例等技术。

**📊 数据集**

主要使用Office-Home（Product/Clipart）65类数据集；此外尝试CRC histology（PathMNIST）但因预检失效未完成。

**📈 对比分析**

通过零目标、1、5、10目标图像预算的实验，对SW、MMD、请求类、Clipart比例等指标进行配对Bootstrap检验。结果显示在K_T=0和1时，main路线在SW略优于错误源和关系仅，但在更高预算下差异不显著；请求类准确率接近随机，语义迁移不明显。

**⚠️ 局限性**

局限性包括：仅在单一随机种子、10个验证类、无目标时间优化下评估；统计样本有限，未进行多重校正；未证明语义转移；CRC实验因预检失败未完成，导致外部数据集结果缺失。

---

## 560. WeVisOne-VL: Unified Visual Perception from Objects to Pixels

**arXiv ID:** 2610.31329 | [PDF](https://arxiv.org/pdf/2610.31329v1)

**作者:** Tianming Liang `[一作]`, Jing Lyu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种名为 WeChat Vision All-in-One 的统一视觉语言模型，能够通过结构化的下一个 token 预测同时完成物体级别（检测、定位、计数、OCR）和像素级别（分割、抠图）任务，并通过轻量级像素解码器实现高精度像素预测。

**💡 创新点**

创新点包括：① 将坐标信息编码为专用 token，直接在语言输出中表达检测框；② 在每个框后插入特殊 token，激活轻量级像素解码器，完成分割/抠图；③ 构建“数据引擎”WDE，统一语义发现与实例对齐，自动生成大规模标注并持续迭代；④ 采用三阶段监督微调加混合任务强化学习（GRPO）来实现检测与像素任务的联合优化。

**🔧 技术方法**

使用的核心技术有：Qwen3‑VL 视觉语言模型骨干；视觉特征适配器、变形注意力查询解码器、实例条件密集预测头组成的像素解码器；组相对策略优化（GRPO）强化学习；多阶段数据生成与校准流程；以及专用坐标 token 与结构化输出格式。

**📊 数据集**

训练数据来源于约 100M 张图像、3.6B 边框注释，主要采集自 GRIT、CC3M、LAION、SA‑1B 等公开数据集，并通过 WDE 进行统一标注、分割、抠图、参考表达生成；测试基准覆盖 COCO、LVIS、Dense200、VisDrone、RefCOCO/RefCOCO+/RefCOCOg、WeRef-Bench、Ref‑Adv、PR‑Bench、CountQA、PixMo‑Count、OCR（HierText、ICDAR2015、TotalText、SROIE）等。

**📈 对比分析**

在所有评测任务上与现有 VLMs（如 Qwen3‑VL、Rex‑Omni、LocateAnything）以及专用检测/分割模型对比，平均 F1、召回率、计数准确率、OCR F1、分割 cIoU、抠图 MAD/MSE 等指标均明显优于同规模模型，部分任务（如 COCO、Dense200、RefCOCO、Matting）甚至突破当前 VLM 最高水平；在单词级别的拒绝率（absent‑target）也表现更好。

**⚠️ 局限性**

局限性包括：① 受限于通用训练数据量，整体泛化能力仍不及大规模基线；② 为保持推理效率，未使用链式思考（CoT）推理，可能限制复杂推理能力；③ 仅支持单轮预测，无法通过多轮交互逐步细化结果；④ 对用户指令的模糊性和语义多样性仍缺乏鲁棒性，需要进一步提升意图理解与多轮交互能力。

---

## 561. Skeleton-Grounded Latent Action Model for Robot World Models

**arXiv ID:** 2610.31474 | [PDF](https://arxiv.org/pdf/2610.31474v1)

**作者:** Qi Lv `[一作]` (HIT), Mike Shou Zheng `[通讯]` (NUS)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种骨架引导的潜在动作模型SkeLAM，用于可控的机器人世界建模；

**💡 创新点**

创新点在于将潜在动作拆分为以机器人骨架监督的运动（kinematic）分量和保持视觉动态的残差分量，并通过跨模态交换重建和分支丢弃实现对运动分量的专门建模；

**🔧 技术方法**

使用Transformer VAE进行潜在动作学习、骨架监督、跨模态交换重建、分支丢弃、轻量级动作适配器以及Cosmos-Predict2.5等世界模型骨干；

**📊 数据集**

主要在DROID数据集上训练和评估，同时使用AgiBot数据增强；

**📈 对比分析**

与DreamDojo、CD-LAM等基线相比，SkeLAM在DROID上在运动精度（ADE下降19.5%）和视觉保真度（LPIPS下降11.1%）等指标上均优先；

**⚠️ 局限性**

局限性包括对成功率召回率仍偏低，难以捕捉长时序交互中的错误传播，以及缺乏闭环策略评估。

---

## 562. The Cheng-Murray Conjecture on Deep Holes of Full-Length Reed-Solomon Codes

**arXiv ID:** 2610.31531 | [PDF](https://arxiv.org/pdf/2610.31531v1)

**作者:** Weijun Fang `[一作]` `[通讯]` (Shandong University), Weijun Fang (Shandong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

对全长 Reed–Solomon 码在非例外维度 3≤k≤⌊q/2⌋ 的深洞（deep holes）进行完整分类，验证了 Cheng–Murray 假设的修正版本。

**💡 创新点**

创新点在于结合几何（切线包络）与代数（局部综合数、模板唯一性）两条互补方法，首次在低维且大域范围内完成深洞的判定；并通过严格的匹配计数与有限域上的限制求和论证，解决了之前未处理的有限域小场情况。

**🔧 技术方法**

主要技术包括：标记弧与切线包络的几何结构、切线包络定理、差分约束与 Wronskian、局部综合数与模板唯一性、匹配图的组合计数、有限域加法/乘法的受限求和定理。

**📊 数据集**

无实验数据集；研究完全基于符号计算与有限域代数论证。

**📈 对比分析**

通过数学证明验证，未涉及数值实验或性能对比；证明显示在所述参数范围内唯一满足深洞条件的插值多项式度数恰为 k，完整地确认了猜想。

**⚠️ 局限性**

限制在于证明过程极为复杂，难以直接推广到更一般的 Reed–Solomon 码（如非全长、非完全集）或给出高效的深洞检测算法；论文主要关注理论完备性，而非实际实现或计算复杂度分析。

---

## 563. Measuring the Verifier's Floor: How Much Apparent Rare-Word Text-to-Speech Error Is ASR Error?

**arXiv ID:** 2610.31470 | [PDF](https://arxiv.org/pdf/2610.31470v1)

**作者:** Kirill Borodin `[一作]`, Grach Mkrtchian `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了文本到语音（TTS）系统的评估方法，特别是如何通过自动语音识别（ASR）系统来测量TTS的错误率，并分析了在不同识别器下的表现。

**💡 创新点**

创新点在于提出了三种不同的TTS错误估计方法，并强调了在评估中考虑稀有词的重要性，指出单一识别器的掩蔽方法存在偏差。

**🔧 技术方法**

使用了三种不同的语音识别器（A、B和C）来进行对比评估，并通过对人类录音和合成语音的分析来测量错误率。

**📊 数据集**

使用了LibriSpeech和FLEURS数据集，这些数据集包含了人类录音和合成语音的测试句子。

**📈 对比分析**

通过比较标准的ASR回合、单一识别器掩蔽和三识别器裁定的方法，发现裁定方法显著降低了TTS错误率，尤其是在处理稀有词时，裁定错误率比标准错误率低两到三倍。

**⚠️ 局限性**

限制在于没有进行听力测试，所有TTS错误的判断均依赖于识别器，可能存在盲点。此外，稀有词的错误估计和趋势具有较大的不确定性，且研究范围仅限于英语读语音。

---

## 564. An Empirical Study of Recent Large Language Models for Knowledge Graph to Text Generation on Standard Benchmarks

**arXiv ID:** 2610.31529 | [PDF](https://arxiv.org/pdf/2610.31529v1)

**作者:** Priyanka `[一作]` (La Rochelle Université), Antoine Doucet `[通讯]` (La Rochelle Université)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对知识图谱到文本生成（KG-to-text）进行了研究，比较了最近的预训练大型语言模型（LLMs）与微调的编码器-解码器模型在生成自然语言方面的表现。

**💡 创新点**

创新点在于提出了一种统一的评估方法，使用多种参考基础指标和图基础的无参考LLM评估，发现LLMs在某些设置下与微调模型的差距显著缩小，并在图基础指标上表现更好。

**🔧 技术方法**

使用了多种模型架构，包括密集解码器（如Gemma和Granite）、专家混合模型（如Nemotron）和编码器-解码器模型（如T5-Gemma2），并采用了低秩适应（LoRA）技术进行微调。

**📊 数据集**

使用了WebNLG17、WebNLG20（英语子集）和DART数据集，这些数据集具有较少的图-文本对齐噪声，适合用于KG-to-text生成的基准测试。

**📈 对比分析**

通过多种参考基础指标（如BLEU、METEOR等）和LLM作为评估者的方法进行比较，结果显示微调模型在BLEU等指标上表现优于LLMs，但在图基础指标上LLMs表现更佳，且两者的评估结果相关性有限。

**⚠️ 局限性**

限制在于微调实验的结果与已发布的值不完全相同，且仅对T5-large和单一LoRA模型进行了微调，缺乏对其他解码器模型的扩展。此外，人工评估仅由单一评估者进行，可能影响结果的独立性。

---

## 565. Beyond Causal Masks: Auditing and Localizing Hidden Future-Token Leakage in Adaptive Sequence Models

**arXiv ID:** 2610.31522 | [PDF](https://arxiv.org/pdf/2610.31522v1)

**作者:** Huiting Liu `[一作]` (Guilin University of Electronic Technology), Jinsong Wu `[通讯]` (Guilin University of Electronic Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过前缀不变性检验、LeakScore 等指标，系统检测并定位 Transformer 解码器中隐藏的未来词泄露问题，并以 CDR-AttnRes 为案例，证明低 teacher‑forced loss 并不一定代表真实的自回归性能提升。

**💡 创新点**

创新点在于：① 引入了基于前缀不变性（Prefix‑Invariance）的泄露检验框架；② 设计了 LeakScore 与 FirstLeakLayer 等可量化诊断工具；③ 在多种模块（池化、卷积、线性注意力、路由、归一化）上进行合成与实测验证；④ 通过故障注入与层级定位，证明泄露根源是全序列摘要广播到动态路由查询；⑤ 在修复后训练的模型中验证泄露消失与性能恢复。

**🔧 技术方法**

使用技术包括：前缀不变性测试、LeakScore 统计、层级泄露剖析、故障注入、缓存一致性与精度一致性检查、以及自回归生成评估（GPT‑2 reference perplexity、采样熵、Distinct‑2/Repeat‑3）。

**📊 数据集**

主要使用的数据集为 OpenWebText（用于验证 loss 与生成评价），以及公开的 GPT‑2、Qwen2.5‑0.5B 与 Mamba‑130M checkpoint 做真实模型检验；合成实验中使用随机初始化的 token ID 序列。

**📈 对比分析**

与标准基线、Static AttnRes 以及修复版 CDR‑token‑fix / CDR‑prefix‑mean‑fix 进行对比。低 teacher‑forced loss 的预修复模型在前缀不变性测试中失败，生成时参考模型 perplexity 远高于 15–20 的控制水平；修复后模型恢复到控制水平，验证 loss 与 perplexity 近似一致，且无泄露。合成模块对照实验显示非因果实现始终泄露，因果修复始终通过。

**⚠️ 局限性**

局限性包括：① 前缀不变性检验是经验性检查，无法证明所有输入与所有实现路径均无泄露；② 仅覆盖了部分常见的辅助模块，未涵盖所有 SSM、MoE 或混合架构；③ 合成实验使用随机初始化，未涉及训练优化带来的潜在副作用；④ 仅在 12‑层与 24‑层 CDR‑AttnRes 案例中进行修复，未验证更大规模或公开实现的普适性。

---

## 566. FairFRESH: A Neuro-Symbolic Framework for Fair and Explainable Household Food Recommendation

**arXiv ID:** 2610.31506 | [PDF](https://arxiv.org/pdf/2610.31506v1)

**作者:** Kodur Sai Vinay Sathvik `[一作]` (BNM Institute of Technology), Tejaswini R. M `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 FairFRESH，一个将家庭食物推荐视为受限公平分配的神经符号框架；

**💡 创新点**

将多成员家庭的共享食材限制与个人饮食规则整合为可解释的分配层，证明有限惩罚无法在阈值附近保证安全，并给出贪婪分配的近似理论与安全过滤；

**🔧 技术方法**

利用多头注意力融合味觉、可检索约束和可用性编码的神经评分，贪婪算法实现效用、均衡、Nash福利三种福利目标，符号安全过滤，以及基于 Shapley 与 Banzhaf 的游戏理论解释；

**📊 数据集**

基于美国国家健康与营养检查调查（NHANES）2013–2023年的多日饮食记录和健康标签构建 2–6 人家庭；

**📈 对比分析**

与五种学习型推荐器及无协调策略对比；Nash 分配在保证公平性（Gini下降 42%）的同时，Recall@20 提升至 0.197，接近理论最优仅差 1%；共享菜品实验显示共享可节省约 30% 的库存；

**⚠️ 局限性**

使用 NHANES 个体构造的家庭、库存和可用性仅为前一天消费的代理；安全规则未临床验证，过滤后可能导致缺少推荐，未在共享菜品中提供分配优化保证，实验基于模拟和预训练模型，缺乏人类评估与临床验证。

---

## 567. Structure-preserving reduced-order models of thin panels in supersonic and hypersonic flow

**arXiv ID:** 2610.31504 | [PDF](https://arxiv.org/pdf/2610.31504v1)

**作者:** Miguel A. Aguilo `[一作]` `[通讯]` (Morphorm LLC), Miguel A. Aguilo (Morphorm LLC)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了基于物理结构的降阶模型，并在两种高速翼面实验（RC-19 夹持薄板和 HyMAX 斜撑板）上实现了学习型降阶模型，将物理能量守恒与数据驱动相结合，且仅训练耗散项。

**💡 创新点**

创新点在于：①将能量守恒和耗散限制嵌入模型结构，保证任何权重下的离散能量平衡；②在无时间序列数据的情况下，利用实验包的统计数据和测量模态，完成模型的训练与验证；③引入离散梯度（AVF）步进，消除数值漂移；④通过测量校正（观测耦合）实现对混沌系统的追踪。

**🔧 技术方法**

使用了端点投影的 Ritz–Galerkin 模型、端点投影的 Hamiltonian 表述、离散梯度积分器、岭回归拟合非保守力、软加法正定耗散矩阵、Adam 优化器、观测者（Luenberger）校正和多尺度特征归一化。

**📊 数据集**

采用了 AePW‑4 高速工作组公开的 RC‑19 和 HyMAX 数据包，包含模态形状、频率、压强统计、温度场以及实验条件（马赫数、腔压、冲击波位置）。

**📈 对比分析**

与实验统计、Brouwer 等人仿真结果以及其他降阶模型进行对比；RC‑19 的主频预测误差低于 1%（236Hz 对 238Hz），Lyapunov 指数与实验一致；HyMAX 的尾部位移跟踪误差为 3.7%；在混沌案例中，观测校正能将追踪窗口延长至数十周期。性能上模型保持了能量守恒、逼近实验幅值并显著减少误差。

**⚠️ 局限性**

局限性包括：①数据包缺乏时间分辨率，导致只能在统计层面验证；②附着冲击案例中位移标准差预测偏差较大，归因于缺少形变依赖的表面压强数据；③模型对模态数量和观测点布局敏感，无法自动适应新的几何或测量配置；④耗散参数只能保证不增能量，不能捕捉潜在的能量注入机制。

---

## 568. SABLE: Scalable Affine Bayesian Latent-space Exploration

**arXiv ID:** 2610.31495 | [PDF](https://arxiv.org/pdf/2610.31495v1)

**作者:** Erik Bodin `[一作]` (Monumo), Nicolas Durrande `[通讯]` (Monumo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出SABLE方法，使用贝叶斯仿射模型在高维生成模型潜在空间中进行优化，并通过离子空间与正交方向分离、有限视野探索调度和伪观测压缩实现高效决策。

**💡 创新点**

在高维潜在空间中分解查询为在数据子空间内的方向和正交方向，使用有限视野的探索计划控制正交探索；引入可压缩的伪观测表示，使模型拟合成本与观测数无关；在仿射模型上实现以EI为准则的低维采样。

**🔧 技术方法**

贝叶斯线性（仿射）工作模型、期望改进（EI）采集、离散余弦变换生成正交方向、分层超参数边缘化、伪观测压缩（谱重构）以及有限视野价值信息调度。

**📊 数据集**

合成的ZonalFunctionBench（10k维）以及四个科学潜在空间优化任务：Lagrangian湍流（6k维）、抗菌肽（6.48k维）、CaloDiffusion Calorimeter DS2（13.44k维）和DS3（40.5k维）。

**📈 对比分析**

与随机、正交搜索、CMA-ES（分离/全协方差）、Vanilla BO、Linear BO、TuRBO、MSR、BAxUS、O3等基线比较，SABLE在样本效率上往往高于或等于基线，决策时间保持毫秒级，且在高维度（40k）下仍保持低延迟并取得最佳或接近最佳性能。

**⚠️ 局限性**

依赖均匀高斯潜在先验与仿射工作模型，难以处理高度非线性或量化结构的目标；需要预先知道最大评估预算并使用有限视野计划；批量采集或更高阶工作模型的扩展尚未实现。

---

## 569. Near-Optimal Bounds for Testing k-Colorability

**arXiv ID:** 2610.31489 | [PDF](https://arxiv.org/pdf/2610.31489v1)

**作者:** Bangjie Xu `[一作]` `[通讯]` (Tsinghua University), Bangjie Xu (Tsinghua University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了稠密图模型中测试k-可着色性的样本复杂性，证明了在原图与k-可着色图之间存在显著差距时，随机选择的子图具有非k-可着色性的概率。

**💡 创新点**

提出了一种基于嵌套调色板系统和迭代选择的试验着色过程，得到了O(klog k/ε)的上界，改进了之前的O(klog(1/ε)/ε)的结果，并且在k固定时达到了已知的下界，首次实现了对ε的最佳Θ(1/ε)依赖。

**🔧 技术方法**

使用了试验着色过程和调色板系统的技术，结合了组合数学中的极值方法。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了图的性质和样本选择的随机性。

**📈 对比分析**

与之前的O(k/log(1/ε))的结果进行比较，新结果在非空参数范围内不劣于之前的结果，并在1/ε为超多项式时严格改进；在k固定时，上界等于现有的最佳下界，显示样本复杂性为Θ(1/ε)。

**⚠️ 局限性**

限制在于该方法主要适用于稠密图模型，可能不适用于稀疏图或其他类型的图结构。

---

## 570. From Structured Records to Patient Narratives: Preserving Clinical Alignment in LLM-Generated Health Summaries

**arXiv ID:** 2610.31467 | [PDF](https://arxiv.org/pdf/2610.31467v1)

**作者:** Bayram Ayadi `[一作]` (University of Passau), Johanna Schmidhuber `[通讯]` (Fraunhofer Institute for Cognitive Systems IKS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文研究将可穿戴设备收集的月经周期生理记录转换为患者可读摘要的任务，提出了基于结构化证据约束和检索增强生成（RAG）的语言模型生成框架，并引入了五类语义漂移的评价体系。

**💡 创新点**

创新点包括：①将结构化证据直接作为生成约束，引入检索检索的术语定义支持简化；②构建了面向月经健康的半合成数据集，并提供了五维语义漂移分类与自动化评价指标；③系统比较了三种LLM在三种生成设置（Base、Constrained、RAG‑Constrained）下的性能，量化了语义对齐、可信度、缺失信息和可读性之间的权衡。

**🔧 技术方法**

采用技术：Prompt‑based 结构化证据约束、检索增强生成（使用 README 术语库）、PubMedBERT‑MNLI‑MedNLI 进行 NLI 评估、scispaCy 进行实体识别与诊断过度推理检测、Claude‑3.5‑Haiku 作为语调评估器、以及多种可读性、词汇多样性和句法复杂度指标；统计分析使用配对 Wilcoxon 检验与效应量估计。

**📊 数据集**

数据集：500 条半合成的月经周期结构化记录（基于因果 DAG 生成），每条记录包含周期、睡眠、心率、HRV、SpO₂、体温、症状等字段；检索词典为 README 数据集（约 5 万对医用术语与通俗定义）。

**📈 对比分析**

比较方法：对每个记录在三种模型（GPT‑4o、Qwen2.5‑14B、Phi‑4‑mini）下的三种生成模式共 4,500 条摘要，使用配对 Wilcoxon 统计并报告均值/中位数、Cohen d 和秩相关；结果显示：① Constrained 与 RAG‑Constrained 在信度（faithfulness）和模糊度（epistemic misalignment）上普遍优于 Base；② 对于 GPT‑4o，检索进一步提升信度；③ 但可读性（FKGL）显著下降，词汇多样性下降，医学术语出现频率上升；④ Omission 和 urgency 评价在不同模型间表现不一。

**⚠️ 局限性**

局限性：① 所有评估指标均为自动化（NLI、实体识别、LLM 语调评估），未进行人工验证；② 仅使用单一 LLM 评估语调；③ 半合成数据未经过临床专家审核，真实用户数据的噪声与多样性未被覆盖；④ RAG 仅检索术语定义，未加入额外临床证据；⑤ 未与外部基准系统进行对比；⑥ 研究聚焦单一领域（月经健康），难以直接推广至其他健康域。

---

## 571. When Is a Memory Safe Enough to Act On? Action-Conditioned Verification for Long-Lived Agents

**arXiv ID:** 2610.31446 | [PDF](https://arxiv.org/pdf/2610.31446v1)

**作者:** Yiqi Wang `[一作]` (University of Southern Queensland), Mingkai Zheng `[通讯]` (Southern University of Science and Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在长期记忆的智能体中，如何根据即将执行的动作来决定哪些记忆条目需要被再次验证；提出了行动条件化记忆验证（AMV）框架，并实现了 AMV-Guard 门控机制，自动提取动作所需前提、校准记忆错误概率、并按价值信息（VOI）动态选择验证顺序；

**💡 创新点**

创新点在于把记忆验证从单纯的记忆质量评估转变为与具体动作相关的决策问题，将验证视为停止问题并引入行动后果和误差成本的权衡；通过VOI自适应选择验证顺序，并允许直接放弃而非强制执行，从而显著降低不必要的验证开销；

**🔧 技术方法**

技术上结合了大型语言模型（LLM）进行前提提取、贝叶斯校准、价值信息估计、两步向前规划的近似停止决策；实现了多验证器覆盖、不同可靠性和成本的管理；在门控中使用了基于后验风险的屏蔽阈值；

**📊 数据集**

使用了两个模拟工具环境：ToolSandbox（3520条冻结实例，分训练/验证/测试）和 AppWorld（100个状态变更任务），并在不同的LLM后端（Qwen3.5-35B、Qwen3.5-27B、DeepSeek-V4.1-Flash）上评估；

**📈 对比分析**

与“始终执行”“始终验证”“环境探测（EnvProbe）”“CTA”等基线进行对比；实验显示 AMV-Guard 在 ToolSandbox 上将平均验证成本降低约33.9%，覆盖率从38.83%提升到44.17%，保持100%安全执行召回率且零误报；在 AppWorld 的迁移实验中验证成本降低21.3%，原始危害率略有下降，覆盖率保持在57%；

**⚠️ 局限性**

局限包括：前提提取的召回受模型驱动限制；验证器覆盖多个前提时采用近似联合观测模型；门控仅使用两步向前规划，未覆盖更长的互补验证序列；损失函数需人工指定不对称性；实验仅在模拟环境中进行，未涵盖真实后果和人类监督；

---

## 572. DIAL: Steerable Language Models with Distilled Instruction-Adherence Levels

**arXiv ID:** 2610.31438 | [PDF](https://arxiv.org/pdf/2610.31438v1)

**作者:** Thanh Q. Tran `[一作]` (National University of Singapore), Bryan Kian Hsiang Low `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并提出一种基于CFG的自监督蒸馏方法，训练轻量级的文本条件自适应Steering适配器，实现对LLM指令遵循程度的细粒度可控调节，并保持LLM权重冻结。

**💡 创新点**

创新点在于将CFG在不同遵循级别的多分布直接作为教师，通过on‑policy自蒸馏学习一个可变强度、几何自适应的分类器驱动的Steering适配器，既实现细粒度控制，又兼顾效率与可扩展性；且仅使用指令文本，无需标注回复或额外生成器。

**🔧 技术方法**

采用Classifier‑Free Guidance (CFG) 生成多级遵循分布；on‑policy自蒸馏与Jensen–Shannon损失；轻量级注意力+分类器Steering编辑；冻结LLM隐藏状态的闭式编辑；文本条件的自适应阈值与范数权重。

**📊 数据集**

使用Concept16k（概念描述集）和SteerIF（约98k条概念、人格、可验证等指令）进行训练；评测使用AxBench Concept500、FLAS held‑out概念集、IFEval可验证指令集以及多轮会话场景。

**📈 对比分析**

与提示、LoReFT、LoRA、HyperSteer、FLAS、CFG等基线在概念表达、人格表达和可验证指令三方面对比。结果显示：概念表达得分与CFG相当或更好，可视化的α曲线实现平滑可控；可验证指令准确率提升约10‑20%；推理效率保持96‑99%基准吞吐率，单token延迟与原LLM相近。

**⚠️ 局限性**

在高遵循级别下可能出现文本连贯性下降；对部分可验证指令的提升有限；目前多轮对话表现尚待进一步验证；对负α（即偏离指令）尚未实现，仍需探索。

---

## 573. Homomorphic Encryption with Coset Code Construction

**arXiv ID:** 2610.31437 | [PDF](https://arxiv.org/pdf/2610.31437v1)

**作者:** Haodong Yang `[一作]` (Syracuse University), Venkata Gandikota `[通讯]` (Syracuse University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种基于余数码的同态加密框架，利用余数码将多条消息符号打包进单个密文向量，并在不泄露密钥的前提下实现加法与乘法的公共计算。该框架在保持解密正确性的同时，给出了安全性分析，证明在满足坐标标签可区分性假设的前提下实现有限明文攻击安全。

**💡 创新点**

创新点包括：
1) 通过余数码与同态误差纠正码相结合，将多符号消息映射为同一余类，实现多符号打包；
2) 定义兼容码序列和层级标签，使得在给定的正式度限制下支持加法与乘法；
3) 提出坐标标签可区分性假设（IID‑DSCP）并将其与方案安全性关联；
4) 给出基于 Reed–Muller 与 Reed–Solomon 的具体构造，并详细讨论其参数与距离特性。

**🔧 技术方法**

所使用的技术手段包括：线性码与余数码理论、评估码（RM、RS）、同态误差纠正码、坐标标签可区分性假设、平均自由指数集、Schwartz–Zippel 估计、线性恢复与符号映射、乘法兼容性条件、公共乘法与加法的层级标签。

**📊 数据集**

论文未涉及实际数据集，全部讨论基于理论参数（如码长、字段大小、消息符号数）的抽象模型。

**📈 对比分析**

与单符号 Armknecht 等人方案对比，打包方案在相同字段和密文向量长度下实现了：
- 密文向量数量从 k 倍减少到 1；
- 加法与乘法的运算成本从 O(k·n) 降至 O(n)；
- 关键存储与密文存储也随 k 减小。性能提升显著，但安全性取决于满足坐标标签可区分性假设；若参数不合适，则存在秩测试等攻击。

**⚠️ 局限性**

主要局限性：
1) 安全性仅在满足坐标标签可区分性假设及样本预算 L、正式度 Δ 的约束下可保证；
2) 若 L+Δ(Δ‑1)≥n‑k+1，则可通过秩测试攻击区分真实加密与均匀样本；
3) 打包虽降低密文数量，却不自动提升安全性；
4) RM 与 RS 构造受平均自由指数集、字段大小、码长等参数限制；
5) 论文未给出实现或实验验证，仅在理论层面进行比较与分析。

---

## 574. Conservative Meta-Agent Search: Architecture Adaptation under Non-Stationarity

**arXiv ID:** 2610.31436 | [PDF](https://arxiv.org/pdf/2610.31436v1)

**作者:** Ali Atiah Alzahrani `[一作]` `[通讯]`, Ali Atiah Alzahrani

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个保守的元代理架构搜索框架（Conservative Meta‑Agent Search），通过在可逆的部署循环中对候选架构进行分层评估、保守提升与可回滚操作，来适应金融投资系统中的非平稳性。

**💡 创新点**

创新点在于将元代理搜索转化为带有切换成本、硬风险约束的序列占优替换问题，并结合 paired shadow replay、Shapley式因果信用分配以及经验校准的保守提升门控，实现对错误提升的统计控制和可解释的决策过程。

**🔧 技术方法**

使用了基于大型语言模型的类型化工作者生成、编译校验、共同随机数的对齐评估、差分隐私式随机化、区块自助法估计、Bernstein型方差惩罚的提升门以及模拟的“灰度”可观测区块重新演练。

**📊 数据集**

在三大测试轨道上评估：① 由隐藏状态马尔可夫模型构造的受控 regime 轨道；② 历史金融行情回放；③ 无噪声市场下的治理对抗测试。

**📈 对比分析**

与六种基准（单体代理、固定委员会、规则路由器、AFlow、MaAS/AutoMaAS、GEPA提示优化）在相同预算下进行并行跑；在受控轨道上 C‑Meta‑Agent 的平均相对动态或验收后或偏差降低 10.3%，OOD 效用提升 8.7%，误提升率 3.4%，团队规模约 3.4。

**⚠️ 局限性**

局限包括：缺乏理论保证（提升门只做经验校准）、仅在作者设计的模拟器上验证、未在真实线上系统测试、对评估器和目标的误规格仍可能导致 Goodhart 效应、以及对极端非平稳事件的鲁棒性未充分评估。

---

## 575. Drift Says When, Budget Says How Much: Rate-Controlled KV Re-selection for Long-Form Multimodal Generation

**arXiv ID:** 2610.31433 | [PDF](https://arxiv.org/pdf/2610.31433v1)

**作者:** Bin Hu `[一作]` (Tsinghua University), Yingbo Xiao `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并实现了一种基于漂移驱动的KV缓存动态刷新方法——Rate‑Controlled Refresh（RCR），用于多模态大型语言模型的长文本生成。

**💡 创新点**

创新点在于：①将漂移（drift）作为刷新时机信号；②通过样本级刷新率控制器自适应分配刷新次数；③利用低秩SVD检测漂移、候选重选和GPU内掩码决策，实现高效、可扩展的刷新机制。

**🔧 技术方法**

使用技术包括：前缀键的低秩SVD近似漂移检测、粗细筛选候选重选、GPU内掩码并行刷新、RoPE保持键不变、ROUGE‑L等评估指标。

**📊 数据集**

实验数据集为DetailCaps、TextCaps图像描述数据集和ActivityNet的VideoDetailCaption视频描述，模型涵盖Qwen2‑VL、LLaVA‑NEXT和Qwen3‑VL三大7–8B多模态LLM。

**📈 对比分析**

与一拍选、InfiniGen、Quest、ArkVale等基线对比，RCR在六个模型–任务组合中在五个场景获得+1.6–3.8 ROUGE‑L点，刷新率约40–48%，逼近逐步刷新，且在视频任务上的优势随漂移同质化而减弱。

**⚠️ 局限性**

局限性包括：仅评估短图像描述和视频任务，未覆盖长文本、多图、多模态推理等；评估以Fidelity为主而非生成正确性；实验在模拟环境下，未测量真实部署延迟；对固定阈值和模型差异的泛化尚待进一步验证。

---

## 576. A Sparse Topometric Graph for Scalable Cross-Session Indoor Navigation

**arXiv ID:** 2610.31432 | [PDF](https://arxiv.org/pdf/2610.31432v1)

**作者:** Eun Sun Lee `[一作]` (Seoul National University), Young Min Kim `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出稀疏拓扑几何图（Sparse Topometric Graph）实现跨会话室内地图重用，并在未知起点下实现目标导向导航。

**💡 创新点**

创新点包括：①在图节点上聚合固定半径的局部几何描述符，实现几何信息的稀疏化；②使用基于节点几何和图连通性的学习图匹配网络，提升跨会话对齐鲁棒性；③将图匹配结果同时用于定位和路径规划，形成端到端的跨会话导航框架。

**🔧 技术方法**

使用技术：稀疏拓扑几何图、旋转不变点云编码器（BUFFER‑X）、基于IPCA‑GM与Sinkhorn的图匹配网络、RANSAC位姿估计、Fast Marching规划、目标与前沿双重得分策略。

**📊 数据集**

使用数据集：Gibson、3D‑FRONT、HM3D，以及现实办公环境的异步扫描数据。

**📈 对比分析**

与占据格、特征图、视觉检索、对象/场景图、点云等基线进行对比，表现为：Gibson上成功率最高（77.1%）、3D‑FRONT 40.0%，SPL最好，内存占用仅 0.02 MB（比最小对手低两倍以上），在噪声、部分重叠、规模扩展和场景变化下均保持稳定。

**⚠️ 局限性**

局限性：仅适用于单层二维导航，未考虑多层/垂直连通；对严重拓扑改变（如门关闭、通道阻塞）评估不足；现实实验为离线数据匹配演示，未实现在线实时控制。

---

## 577. Jev-DLC: Decision-Level Continual Learning from Typed Feedback

**arXiv ID:** 2610.31521 | [PDF](https://arxiv.org/pdf/2610.31521v1)

**作者:** Yangyang Feng `[一作]` (Hong Kong University of Science and Technology), Chengwei Qin `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何利用Typed反馈在冻结的决策层上持续个性化，提出了Jev-DLC方法。

**💡 创新点**

创新点在于引入双通道（保守与响应）存储行动计数和修正方向，并通过局部最小改动更新，实现无原始交互历史的持续学习。

**🔧 技术方法**

使用的技术包括Jev式Typed决策接口、PA‑I最小改动学习、可组合softmax融合以及固定向量投影与状态预算。

**📊 数据集**

评估数据集为TCPA-Repeat（基于LaMP‑2的重复行动轨迹）以及TCPA-Open、TCPA-Dynamics等轨迹。

**📈 对比分析**

与冻结评估器、Recent‑4、PPlug、TAP‑PER、MRM等基线对比，Jev‑DLC在全流上宏观准确率提升至44.83%，深度≥2时达56.07%，优于其它方法。

**⚠️ 局限性**

局限性是仅支持即时接受/撤销的有限候选集，无法处理延迟或自由文本反馈，且状态依赖固定动作词表。

---

## 578. The Geometry of Metaphor: Probing and Steering Metaphor Representations in Large Language Models

**arXiv ID:** 2610.31520 | [PDF](https://arxiv.org/pdf/2610.31520v1)

**作者:** Shihui Li `[一作]` (University of Amsterdam), Martha Lewis `[通讯]` (University of Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过机制解释技术（激活补丁与分布式对齐搜索）定位LLM内部编码隐喻的线性子空间，并在分类与生成任务中对该子空间进行因果干预以控制输出的隐喻性。

**💡 创新点**

首次在多模型、多数据集上证明隐喻具有可解释的线性方向，提出并训练全局DAS子空间，展示其在分类与生成两类任务中的因果有效性。

**🔧 技术方法**

使用激活补丁、分布式对齐搜索（DAS）、mass‑mean 线性探测、以及LLM‑as‑judge 评估框架。

**📊 数据集**

使用三类隐喻数据集：NOUN（名词隐喻）、ADJ（形容词隐喻）和DOM（跨领域形容词‑名词对）。

**📈 对比分析**

在基线识别/解释任务中模型准确率约 0.7‑0.9；通过DAS/MM干预将错误预测翻转率提升至 60‑90%；在生成任务中，α=5 时DAS 与 MM 均能提升生成隐喻率，整体性能依模型而异。

**⚠️ 局限性**

仅针对英文、句式高度受控、数据极度简化、未覆盖多句/多段隐喻，且无法在闭源大模型上验证；干预技术可能产生不良行为。

---

## 579. SCOPE: Scene-graph Construction Optimized by Plan Error-correction for Robot Long-term Planning Datasets

**arXiv ID:** 2610.31518 | [PDF](https://arxiv.org/pdf/2610.31518v1)

**作者:** Chaoyue Li `[一作]` (Chinese Academy Of Sciences), Fan Luo `[通讯]` (Chinese Academy Of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个基于计划误差修正的闭环场景图自动生成框架SCOPE，用于生成多样化且可执行的长周期机器人规划数据集。

**💡 创新点**

创新点在于结合熵驱动的多级多样性采样与基于计划反馈的错误修正机制，显著提升数据质量与规划鲁棒性。

**🔧 技术方法**

使用熵增逆频率采样、域特定语言DSL、链式思维生成指令、PDDL规划与符号规则-LLM双层修正等技术。

**📊 数据集**

使用SCOPE生成的自建数据集以及AI2-THOR真实场景数据集进行评估。

**📈 对比分析**

与DELTA、LLM-As-Planner等基线对比，SCOPE在噪声场景中的成功率提升至70.67%，在真实场景中微调模型成功率提升12.33%，并获得更高的可执行率与平均评分。

**⚠️ 局限性**

局限性包括仍有约60%认知层错误及低层定位问题，且对动态环境的适应性尚待验证。

---

## 580. Towards User Simulators That Stay On Task And Still Sound Natural

**arXiv ID:** 2610.31512 | [PDF](https://arxiv.org/pdf/2610.31512v1)

**作者:** Mustafa Omer Gul `[一作]` (Cornell University), Jennifer Neville `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练并发布了 UserLM-2.0，一种针对用户–AI 对话的用户模拟器，能够在多轮交互中忠实执行高层任务意图并产生与真实用户极为相似的对话；

**💡 创新点**

提出了结合强化学习和动态对抗判别器的训练框架，既保留了监督预训练的自然交互特性，又通过实时更新的判别器对话自然度进行奖励，从而显著提升了任务完成率与自然度的协同性能；

**🔧 技术方法**

使用了 Llama3.1‑8B‑Instruct 作为生成器、Qwen3.5‑9B 作为助理模型进行 RL 交互；利用现代 BERT‑Large 训练轻量判别器，并在 GRPO 算法中交替更新生成器与判别器；

**📊 数据集**

主要基于 WildChat‑4.8M（去毒化后的 199,249 条交互）构建任务意图与子目标；另外使用 Qwen3‑30B‑A3B‑Instruct 对意图进行提取，使用 GPT‑3.5‑9B 对奖励进行评估；

**📈 对比分析**

与提示式大模型（如 GPT‑5.6‑Sol）和多种 SFT 训练的用户模拟器（UserLM‑8B、SFTUser‑14B 等）进行对比。UserLM‑2.0 的有效对话率达到 46.2%，是最佳 SFT 模型 27.8% 的 1.66 倍，也是此前最佳模拟器 14.1% 的 3.28 倍；在高复杂度任务（7 个子目标）下仍保持 12.6% 的有效率；此外成本最低，100 条有效对话仅需 0.11 美元；

**⚠️ 局限性**

仍存在与真实人类对话的显著差距（自然度、覆盖率均落后 20% 以上）；仅覆盖纯聊天场景，未考虑工具/环境交互；判别器设计与奖励函数可能导致偏差，且评估指标本身不完备；需要更多多样化数据与跨助手鲁棒性研究。

---

## 581. Evaluation of Uncertainty-Aware Multi-Software Ensembles for Hippocampal Segmentation

**arXiv ID:** 2610.31505 | [PDF](https://arxiv.org/pdf/2610.31505v1)

**作者:** Gabriel Oliveira-Stahl `[一作]` (University College London), Matthew Grech-Sollars `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

评估并比较多软件异质集成的海马分割方法，提出三模型集成并利用集成的不确定性进行质量控制与疾病分类

**💡 创新点**

首次将多软件集成与碳足迹评估相结合，并证明集成不确定性可用于检测影像噪声与错误分割

**🔧 技术方法**

多模型集成、样本方差/互信息/预测熵不确定性估计、平均与加权融合、ECE/AECE校准评估

**📊 数据集**

ADNI、英国本地癫痫与痴呆数据集、MR-ART运动伪影数据集

**📈 对比分析**

三模型集成（InnerEye+FastSurfer+HippoSeg）在Dice和HD95上与五模型集成相当，CO2降低约82%，不确定性阈值能召回>90%错误分割并与运动伪影显著相关，加入不确定性后CN‑AD分类AUC提升≈0.06

**⚠️ 局限性**

样本量小、triad选择受样本重叠偏倚、仅使用后验集成不确定性、未评估概率网络与其他不确定性方法

---

## 582. Coresets for $k$-Means: When Does Uniform Sampling Suffice?

**arXiv ID:** 2610.31491 | [PDF](https://arxiv.org/pdf/2610.31491v1)

**作者:** Amir Carmel `[一作]` (Weizmann Institute of Science), Robert Krauthgamer `[通讯]` (Weizmann Institute of Science)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在Hilbert空间中使用均匀采样构造k-均值和1-中位数的稳定、弱与强核心集，给出了样本复杂度与维度无关的上界与下界。

**💡 创新点**

首次证明了k=1时稳定核心集与弱核心集的样本复杂度分离，并用简单的均值偏差分析替代VC维数，提出统一的稳定核心集框架。

**🔧 技术方法**

核心技术包括RCDA属性、König–Huygens身份、Rademacher过程、向量收缩不等式以及基于均值偏差的集中分析。

**📊 数据集**

实验使用纽约出租车、Covertype和Census 1990三大真实数据集进行验证。

**📈 对比分析**

与重要性采样的核心集方法对比，均匀采样在精度相近的同时构造速度快数百倍，尤其在k-均值实验中误差低于1%，构造时间仅为毫秒级。

**⚠️ 局限性**

方法依赖于β平衡、ρ分离及有限峰度等结构假设，在不满足这些条件的数据上理论保证失效；此外仅适用于z∈{1,2}的Hilbert空间，未覆盖更一般指标。

---

## 583. Training Small Language Models for Transferable Tool Selection

**arXiv ID:** 2610.31483 | [PDF](https://arxiv.org/pdf/2610.31483v1)

**作者:** Zhaoyi Zhou `[一作]` (Meta), Ayush Jain `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在一套简化的 Plan–Retrieve–Select 流程中，对小型 Qwen3 LLM 进行监督微调 + 强化学习，训练其在大规模工具库中进行多子步骤、多工具的自动选择；

**💡 创新点**

创新点包括：①让小型 LLM 在无需大型模型提示或昂贵算力的情况下完成工具调用任务；②通过检索器作为元工具实现跨域与跨检索器的无缝迁移；③基于检索结果的多轮规划与选择机制，克服单轮相关性排名的局限；

**🔧 技术方法**

采用 Qwen3 系列（1.7B、4B、8B）+ BM25（可替换为 COLT）检索、ReAct式多轮推理、监督微调+GRPO强化学习、基于检索精度/召回的规划奖励和精确匹配的选择奖励；

**📊 数据集**

训练与评估均使用 ToolBench G3 数据集（21,361 条指令、1,419 个 API），并在 ToolBench G2、Gorilla、CRAFT、TaskBench 等跨域工具集合上进行迁移实验；

**📈 对比分析**

与 Llama‑4‑Maverick（402B）在多种 Prompt 工作流（ReAct、层次搜索、Search‑R1、本文工作流）对比，Qwen3 1.7B/4B/8B 在 ToolBench G3 上的成功率分别为 37.6%、40.3% 与 41.5%，均显著优于大模型；跨域迁移时 60–70% 的成功率也超过基线；检索器迁移后性能提升；

**⚠️ 局限性**

局限性包括：①需要先训练模型；②Plan 阶段依赖领域知识，对完全单工具任务无优势；③多子步骤假设彼此独立，无法处理子步骤间输出依赖；④在 TaskBench 等更复杂场景下仍略逊于无规划的 Search‑R1 工作流；

---

## 584. Ask About the Right Moment: Training-Free Object Identity Reasoning for Video-Language Models

**arXiv ID:** 2610.31471 | [PDF](https://arxiv.org/pdf/2610.31471v1)

**作者:** Di Yang `[一作]` (College of William & Mary), Yanhai Xiong `[通讯]` (College of William & Mary)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 Track, Then Bind 的无训练方法，先用预训练跟踪器追踪视频中所有候选物体，随后用冻结的 Video‑LM 在可见的证据瞬间将目标绑定到某条轨道，并从该轨道的文本状态中读取答案，从而解决视频中物体身份保持的问题。

**💡 创新点**

创新点在于：① 将目标身份的绑定与后续推理完全分离，先在一次可辨识的瞬间让 VLM 绑定目标轨道；② 只通过一次冻结模型的交互即可跨多个基准；③ 利用预训练的跟踪器保持轨迹而不需要额外训练或强化学习，从而实现高效且通用的身份追踪与回答。

**🔧 技术方法**

技术细节包括：使用 Qwen3‑VL‑8B/32B 作为冻结的 Video‑LM；使用 SAM2.1 进行多帧盒子检测与轨迹跟踪；在轨道上绘制编号；构造静态绑定问题并读取 VLM 的答案；在读取阶段将轨道状态转换为文本事实后再用 VLM 回答；对无绑定的可视性/事件问题构建时间线。

**📊 数据集**

评测数据集涵盖：VET‑Bench（50 个杯子和 50 个卡片蒙提合成视频）、Perception Test（107 条真实杯子游戏视频，65 条带至少一次交换的子集）以及 TOC‑Bench（1,845 条关于对象一致性的真实视频问答）。此外在论文中还对通用视频问答（Perception Test 验证集、Video‑MME 短/中视频）做了对比。

**📈 对比分析**

实验对比显示：在 VET‑Bench 两个任务上该方法均实现 100% 准确率；在真实杯子游戏上，比直接回答提升 17–18%（从约 33% 提升到 51%）；在 TOC‑Bench 上提升 12–18%（从 43% 提升到 48%）。与基于投票或增量输入的直接回答方法相比，Track, Then Bind 在所有基准上均取得显著优势；在多工具代理（SpatialClaw、VideoAgent、VideoINSTA）中同样取得 10–70% 的提升。

**⚠️ 局限性**

局限性：依赖跟踪器的质量，易受相同外观物体交叉、手部遮挡导致的轨道交换；对极小隐藏物体的跟踪效果差；大多数问题仍需约 10–20 次 VLM 调用，导致计算成本相对较高；对不需要绑定的通用问题未作特别优化，仍会多次调用 VLM。

---

## 585. Diffusion-Based Synthesis of Complete 3D Biparametric Prostate MRI

**arXiv ID:** 2610.31523 | [PDF](https://arxiv.org/pdf/2610.31523v1)

**作者:** Claudia Giardina `[一作]` (Universitat Politècnica de Catalunya), Verónica Vilaplana `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于条件三维扩散模型的前列腺双参数MRI（bpMRI）合成框架，能够从解剖掩模生成完整的T2w及扩散模态，支持顺序与一次性联合生成。

**💡 创新点**

创新点包括：直接预测干净图像而非噪声的DDPM；引入各向异性深度调节避免切面压缩；多模式条件与目标配置；并证明直接合成ADC/cHBV的分布更逼近真实数据，优于从合成DWI推导。

**🔧 技术方法**

技术手段为：基于Med‑DDPM的条件三维U‑Net denoiser，直接x0预测，随机高斯模糊掩模，尺度平移归一化，分目标权重的多任务L1损失。

**📊 数据集**

使用了PI‑CAI、FastMRI Prostate和PROSTATEx三大公开数据集，对T2w、ADC、cHBV、LBV、HBV等模态进行配准、质量控制与预处理。

**📈 对比分析**

评价方法包括分布相似性指标（FID、KID、Precision/Recall）和成像一致性指标（PSNR/SSIM）。实验显示联合生成在分布上与单独生成相当或更好，直接生成ADC/cHBV的FID/KID显著低于通过合成DWI推导得到的指标。

**⚠️ 局限性**

主要局限：缺少放射科专家的定性评估和临床下游任务验证；合成图像可能出现掩模印记或边界模糊；未证实合成数据能完整保留诊断关键特征。

---

## 586. Closing All-Exp Gaps in Local Search for the Traveling Salesman Problem and Maximum Constraint Assignment

**arXiv ID:** 2610.31513 | [PDF](https://arxiv.org/pdf/2610.31513v1)

**作者:** Sophia Heimann `[一作]` (University of Bonn), Stefan Hougardy `[通讯]` (University of Bonn)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文构造了特定实例，证明在旅行商问题的2邻域搜索（以及(2,3,2)-MCA的2邻域搜索）中，任何最大改进交换序列都可能具有指数长度，从而展示了局部搜索算法的最坏情况极限；

**💡 创新点**

创新点在于将原先仅适用于Max-Cut问题的Michel‑Scott构造进行改造，使用一系列度数≤3的gadget和“跟随”原理，将问题迁移到TSP和(2,3,2)-MCA，并且设计了递归的“正则序列”和依赖图，以证明所有最大改进序列的长度都是指数的；

**🔧 技术方法**

主要技术包括：①基于权重图的约束表示和“跟随”关系推导；②构造度数≤3的gadget（如v_i,1和v_i,8的替代方案）；④定义“正则序列”Ω_n并构造其递归关系；⑤使用依赖图D_n编码所有改进交换的先后关系；⑥通过本地改进条件和状态转换分析，证明序列的唯一性与指数长度；

**📊 数据集**

该工作为理论性证明，不使用任何标准数据集，所有实例都是由作者构造的特殊图形；

**📈 对比分析**

与以往仅对Max‑Cut或其他特定问题给出指数级局部搜索最坏例子相比，本文首次把这种现象推广到TSP和(2,3,2)-MCA，并且给出了完整的递归式与闭式解，证明最大改进序列长度为O(3^n)（或O(2^n)），从而展示了更广泛的最坏情况；

**⚠️ 局限性**

局限性包括：①仅针对2邻域搜索（以及(2,3,2)-MCA的2邻域）；②构造示例仅在理论上证明最坏情况，未涉及随机或平均性能；③所使用的gadget与权重设置较为复杂，难以直接推广到更一般的局部搜索框架或更高维度的问题；

---

## 587. DuoCook: An Overcooked-Inspired Benchmark for Visual Planning and Coordination

**arXiv ID:** 2610.31511 | [PDF](https://arxiv.org/pdf/2610.31511v1)

**作者:** Ziyi Jiao `[一作]` (University of British Columbia), Peter Yichen Chen `[通讯]` (University of California, San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DuoCook 这一基于 Unreal Engine 的集中式双人烹饪协调基准，使用单一 Vision‑Language Model 控制两名玩家完成订单。

**💡 创新点**

创新点在于：利用编码代理生成多类别厨房地图；设计共享多模态接口（图像、道具图例、实时状态、合法操作）；并在实验中加入可衰减的共享记事本作为记忆消融研究。

**🔧 技术方法**

采用的技术包括：Vision‑Language Models（Claude、GPT‑4o、Gemini‑1.5‑Pro、Qwen‑VLLM）、Unreal Engine + SimWorld Studio、NavMesh 导航、动态地图生成、记事本记忆机制。

**📊 数据集**

使用的数据集为 20 张编码生成的厨房地图，按 5 类划分（基础、共享设备、动态、动态+共享设备、分离工作区），配合任务说明、实时状态 JSON、合法操作列表等多模态输入。

**📈 对比分析**

比较方法：在每张地图上跑一次模型，记录总分、订单完成率、送餐精度、命令接受率与 API 成本共 5 项指标；最高模型 Claude 平均完成率 26.2%，总分 25.6，性能仍处于低水平。

**⚠️ 局限性**

局限性：仅单次运行每模型，未估计结果方差；接口简化感知与导航，未体现真实第一人称或直接控制；共享记忆效果不稳定且成本上升；生成地图结构可能不具代表性，未验证对未知厨房的泛化能力。

---

## 588. Piano2Pop: Expanding Piano Demos into Pop Productions Without Aligned Supervision

**arXiv ID:** 2610.31510 | [PDF](https://arxiv.org/pdf/2610.31510v1)

**作者:** Yun-Chen Cheng `[一作]`, Yi-Hsuan Yang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Piano2Pop 任务，利用伪监督方式将钢琴 MIDI 扩展为完整的流行音乐制品。

**💡 创新点**

创新点在于不依赖对齐钢琴‑流行配对，而通过从流行音频提取伪钢琴示例并进行自监督微调，实现任务专属建模。

**🔧 技术方法**

采用 MuScriptor/HTDemucs 进行多乐器转录，PiCoGen2 自动生成钢琴覆盖，结合 ACE-Step 流动匹配模型和钢琴滚动编码器进行微调。

**📊 数据集**

使用 8,876 条无钢琴 MIDI 的 YouTube 流行曲目做训练集，另用 476 条有真实钢琴 MIDI 的匹配对做评估集。

**📈 对比分析**

与 MuseControlLite 与 ACE-Step cover 的零射线基线对比，PiCo 与 FST 系统在 Fréchet Audio Distance、APA 与谱形 RMSE 等指标上显著优于基线，PiCo 在真实钢琴输入上取得最佳表现。

**⚠️ 局限性**

主要局限在于伪钢琴示例仍可能偏离真实手写钢琴的表现，且对齐质量不如人工标注的真对齐样本，导致迁移到真实钢琴时仍有性能损失。

---

## 589. Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon

**arXiv ID:** 2610.31479 | [PDF](https://arxiv.org/pdf/2610.31479v1)

**作者:** Kyrylo Yemets `[一作]` (MacPaw AI&Research), Mariya Hirna `[通讯]` (MacPaw AI&Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了四类本地LLM推理优化（KV 缓存剪枝、推测解码、GPU 内核移植与后训练量化）在 Apple Silicon 与 RTX 4090 上的实际效果。

**💡 创新点**

创新点在于将优化技术与具体硬件、运行时和工作负载结合，揭示速度提升是技术+硬件+运行时+负载共同决定的，而非单一技术可迁移的性质。

**🔧 技术方法**

技术实现包括：使用 MLX 与 llama.cpp 进行推理；将 Marlin、FlashInfer、MoE gather 等 CUDA 内核迁移至 Metal/MLX；实现多种量化方案（GGUF、RTN、QuaRot+GPTQ）并在同一框架下对比。

**📊 数据集**

实验使用 Qwen3-4B、Qwen3-8B、LFM2.5‑8B‑A1B 等模型，在 Apple M2 Pro、M3 Ultra、M4 Pro 与匹配的 RTX 4090 上进行对比。

**📈 对比分析**

采用单请求、统一内存的解码任务，测量端到端吞吐、预填充速度、解码速度与内存占用。结果显示：KV 剪枝在 Apple 上可达 1.75×，RTX 接近 1；推测解码在主基准上未提升；Marlin 仅在多 token 时受益；FlashInfer 在新 GPU 上可达 1.20×；MoE gather 在所有机器上均为正向；量化在不同芯片上表现差异。

**⚠️ 局限性**

局限性包括：未评估推理质量与多请求批量化场景；Megakernel 仅在单一模型、单芯片上测试；对预填充、调度等细节高度依赖运行时实现。

---

## 590. MoG-VLM: Efficient Video Anomaly Detection through Motion-Guided Vision-Language Models

**arXiv ID:** 2610.31478 | [PDF](https://arxiv.org/pdf/2610.31478v1)

**作者:** Khaled Waleed Dawoud `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Muhammad Haris Khan `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在视频异常检测中，本文提出了一种基于运动自适应控制的零样本、无训练的 VLM 框架 MoG‑VLM，能够在仅使用运动信号作为轻量级控制器时，高效地决定何时调用昂贵的 Vision‑Language 模型，显著降低推理成本；

**💡 创新点**

创新点包括：① 利用多尺度 Temporal Gradient Magnitude (TGM) 进行内容感知的帧采样，仅在运动激增时增密；② Anchor‑Triggered 层级细化机制，在检测到高置信度异常的帧周围动态扩展时窗并早停；③ Tier‑Aware 适配融合，将帧级与时序级两条流按运动层级分配权重，避免训练；

**🔧 技术方法**

技术核心为：多尺度 TGM 运动量计算与自适应 z‑score 门控；Qwen3‑VL‑8B 与 VideoLLaMA3 进行帧级与时序级 VLM 生成描述；Qwen3‑4B 作为 LLM 进行结构化异常评分；多级阈值与 Gaussian 平滑实现稳健的推理；

**📊 数据集**

使用四大公共数据集评估：XD‑Violence、UCF‑Crime、MSAD 与 ShanghaiTech；

**📈 对比分析**

在零样本、无训练设定下，MoG‑VLM 在 XD‑Violence、UCF‑Crime 与 MSAD 上分别取得 91.91%、84.30% 与 87.88% 的 AUC‑ROC，分别比上一最佳（Unified‑Reasoning）提升 +0.57%、+0.02% 与 +1.98%；同时推理速度提升 1.31×（相较于 Unified‑Reasoning）和 4.8× 通过吞吐率相比 LAVAD；

**⚠️ 局限性**

局限性：① 对低运动或微小细节的异常（如 ShanghaiTech 中的滑行、逆行）检测仍不理想；② 对极短暂突发事件的即时检测有时会漏检；③ 对阈值与 TGM 参数的敏感性需要手动调优；④ 依赖大模型，仍需高算力资源；

---

## 591. Point-Level Refinable Semantic Mapping for Spatial Understanding

**arXiv ID:** 2610.31476 | [PDF](https://arxiv.org/pdf/2610.31476v1)

**作者:** Taimeng Fu `[一作]` (University at Buffalo), Chen Wang `[通讯]` (University at Buffalo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在线可细化的开词表3D语义地图框架，通过点级置信度更新实现动态环境下的地图持续改进。

**💡 创新点**

采用点级log‑odds置信度在线更新，结合多视角证据与可见性推理，支持动态物体移动、移除、重定位，同时实现无锚点实例重识别。

**🔧 技术方法**

结合开词表检测（YOLO‑World、MobileSAM等）、CLIP/DINO特征、可见性推理、log‑odds置信度更新、2D追踪（BoT‑SORT）、特征重识别、点云体素降采样和实例合并等技术。

**📊 数据集**

在Replica、ScanNet、HSSD动态场景及Spot机器人真实环境（Hallway、Office、Outdoor）进行评估。

**📈 对比分析**

与ConceptGraphs、HOV‑SG、DualMap等基线对比，Replica上mIoU提升约3.5%、F‑mIoU 5.8%、mAcc 13.9%；ScanNet上mIoU提升约2.5%、F‑mIoU 3.4%、mAcc 1.5%；在VLN任务中对象导航成功率82.4%（SPL 0.696），EQA成功率76.7%（SPL 0.638）。

**⚠️ 局限性**

仍受检测/分割质量影响，极小或远距离物体识别率下降；重识别多实例时可能出现歧义；需手工调节阈值，跨场景泛化能力有限。

---

## 592. AnyGameRenderer: Entity-Consistent Generative Rendering Across Games and an Entity-Level Benchmark

**arXiv ID:** 2610.31466 | [PDF](https://arxiv.org/pdf/2610.31466v1)

**作者:** Ruicong Liu `[一作]` (Alaya Lab), Kaipeng Zhang `[通讯]` (Alaya Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种通用的游戏渲染器 AnyGameRenderer，能够把粗糙的游戏录制视频转成高质量、可按提示更换风格的画面，同时保持实体的计数、时序和位置一致。

**💡 创新点**

创新点包括：①支持多种游戏类型和多样化的粗糙输入（灰块、彩色原型、语义图等）；②通过结构化字幕和实体对应约定实现精确的实体同步；③仅依赖原始视频，无需复杂中间表示；④从三源配对数据构建并提出了实体级评测基准。

**🔧 技术方法**

技术上采用 MiniMax‑H3 的 ref2va 模式，训练一个 rank‑128 的 LoRA；使用六段结构化字幕（实体身份、保持属性、空间、时间、空白描述）和可选 anchor 图像进行条件控制。

**📊 数据集**

数据集共 10,330 条配对视频，来源分别是：GEAR 多流渲染（3,221 条）、Unreal Engine 双流渲染（3,910 条）和真实游戏记录检测生成的粗糙输入（3,199 条）。

**📈 对比分析**

在 59 个留出案例的实体级基准上进行对比，AnyGameRenderer 的性能显著提升：Temporal match 由 50.9% 上升至 62.9%，Position match 由 57.1% 上升至 61.9%，未渲染率从 4.1% 降至 2.9%，幻觉率从 52.1% 降至 46.0%。尤其在双流渲染场景中几乎达到 GT 限值，整体优于零射击基础模型。

**⚠️ 局限性**

局限性包括：对极端风格或极低分辨率/复杂光照的鲁棒性仍不足，Top‑down 游戏的匹配效果相对较差，模型对极端实体尺寸变化和动态遮挡的处理还有提升空间。

---

## 593. RheOPT: Few-Shot Non-Newtonian Flow Prediction via Newtonian Pretraining of RheOFormer

**arXiv ID:** 2610.31465 | [PDF](https://arxiv.org/pdf/2610.31465v1)

**作者:** Maedeh Saberi `[一作]` (Northeastern University), Safa Jamali `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了RheOPT框架，通过在低成本的牛顿流体模拟上预训练Transformer型神经算子，然后用少量非牛顿流体数据微调，来实现高效的非牛顿流场预测。

**💡 创新点**

创新点在于将物理上相关的牛顿流模拟作为预训练源，证明其能在几样本非牛顿预测中迁移有用的几何与输运结构，并通过可逆实例归一化(RevIN)提升在不同尺度数据上的稳健性。

**🔧 技术方法**

技术主要包括基于Transformer的RheOFormer神经算子、RevIN与全局归一化、相对L₂误差评估以及振荡频率、RMS波动幅度等无相位依赖的动态诊断。

**📊 数据集**

使用的数据集包括：4:1收缩通道下的Giesekus、Oldroyd‑B、Bingham三种非牛顿模型的数据，和三角障碍物后方的Oldroyd‑B振荡尾流数据；预训练源为对应几何的牛顿流模拟。

**📈 对比分析**

对比方法是把预训练+微调与从零训练在4、8、12、16样本下的相对L₂误差做对照；在低样本情况下预训练可使误差下降44–81%，并在振荡尾流中恢复正确的振荡频率和幅度；全局归一化对不同数据集效果不一，RevIN表现更稳定。

**⚠️ 局限性**

局限性在于仅验证了几何匹配的情况，未探讨几何变换时的迁移；归一化策略的理论机制尚不充分；未来需扩展到三维流、更多非牛顿模型及更宽的参数范围。

---

## 594. Adaptive Skill Granularity for Coding Agents: Runtime Routing and Macro-to-Micro Recovery

**arXiv ID:** 2610.31461 | [PDF](https://arxiv.org/pdf/2610.31461v1)

**作者:** Dingxuan Zhang `[一作]` (Shanghai Jiao Tong University), Hefeng Zhou `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了编程代理的适应性技能粒度控制，提出在运行时根据任务结构、执行状态和宏技能失败动态切换微观、宏观或混合执行模式，并实现相应的粒度控制器与恢复执行器。

**💡 创新点**

创新点在于将技能粒度视为可调的运行时控制变量，首次引入可观察的宏-微降级恢复机制，并通过浅层决策树实现可审计、可验证且可恢复的技能执行框架。

**🔧 技术方法**

技术上采用大型语言模型驱动的代理、功能特征向量提取、决策树路由器、有限状态机恢复执行器，以及宏技能接口定义（pre, effect, V_m, F_m, fallback）。

**📊 数据集**

实验使用自定义的 GranBench-Code 基准，涵盖四类任务（simple、medium、refactor、recovery）共 80 任务，利用确定性验证器评估结果。

**📈 对比分析**

通过与固定微观、固定宏观、预执行路由（无恢复）和始终降级的混合模式对比，评估验证成功率、token 成本和 G‑score 等指标；实验显示纯微观成功率最高但成本最高，纯宏观成本低但成功率低；学习路由加选择性恢复实现了接近宏观成本且 100% 成功率的性能。

**⚠️ 局限性**

局限性包括仅在编程代理场景验证，宏技能数量有限且恢复策略为单向降级，缺乏多层级切换；对功能特征和宏接口的人工定义可能限制在更复杂任务或其他领域的推广；宏技能的自动演化与自适应升级尚未深入探索。

---

## 595. Deep holes of full-length Reed--Solomon codes: uniform bounds and exact classifications

**arXiv ID:** 2610.31460 | [PDF](https://arxiv.org/pdf/2610.31460v1)

**作者:** Xiang Fan `[一作]` `[通讯]` (Sun Yat-sen University), Xiang Fan (Sun Yat-sen University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究Reed‑Solomon码的深洞问题，对m=4、5、6、7情况给出完全的分类，并通过对称函数剩余判据、组合构造和大量计算证明，证明任意k≤q−m+1时，非特殊低维或奇偶条件下不存在深洞；

**💡 创新点**

创新点在于将深洞判据转化为对称函数方程，结合组合设计与有限域的完整枚举，得到m≤7的完整分类，首次在非平凡范围内完全排除深洞；

**🔧 技术方法**

使用的技术包括对称函数剩余判据、Artin–Schreier扩展、可积多项式构造、Kloosterman求和估计以及大规模计算验证；

**📊 数据集**

所用的数据集为所有有限域F_q及其对应的多项式系数集合，涵盖了从q=8到q=256的所有必要参数；

**📈 对比分析**

与先前的渐近或偶/奇特例比较，本文的结果给出精确阈值，计算部分采用完整枚举验证，性能主要体现在对所有小域的完全覆盖；

**⚠️ 局限性**

局限性在于高维或更大m值的情况仍未完全解决，部分证明依赖于计算枚举，且对极小域的某些特殊情形需单独处理。

---

## 596. How Different, or Different How? Stance Granularity in Viewpoint Selection for Dialog Systems that Engage Users with Alternative Views

**arXiv ID:** 2610.31459 | [PDF](https://arxiv.org/pdf/2610.31459v1)

**作者:** Morita Tarvirdians `[一作]` (TU Delft), Catharine Oertel `[通讯]` (TU Delft)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了一种可交换立场表示的视角选择策略，在对话系统中引导用户接触不同立场观点。

**💡 创新点**

将视角选择规则与立场编码分离，只改变立场表示（标量 vs Perspectivized Stance Vectors），揭示立场表示对视角选择和用户体验的影响。

**🔧 技术方法**

使用ε‑greedy bandit设定距离阈值，基于距离和新颖度的视角选择策略，Socratic式问答的对话系统，以及概念级立场向量编码。

**📊 数据集**

使用来自Debate.org的动物狩猎议题语料库（364条观点），并对其概念级立场向量进行五概念签名提取。

**📈 对比分析**

通过实验（N=63）和对语料库的完整回放，比较标量和Psv两种立场表示的曝光序列多样性、概念覆盖率、解释深度和重构准确率；结果显示Psv提升了曝光多样性和覆盖率，但在解释深度和准确率上无显著差异。

**⚠️ 局限性**

仅在单一议题、单一语言和有限样本上验证，使用固定五概念签名可能不具普适性，重构任务混合理解与记忆，样本量有限导致对细微差异检测不足。

---

## 597. Procrustes-Bering-Sheaft: Automated Model Discovery and Virtual Crash Tests for Microservices

**arXiv ID:** 2610.31444 | [PDF](https://arxiv.org/pdf/2610.31444v1)

**作者:** Anatoly A. Krasnovsky `[一作]` `[通讯]` (Innopolis University and MB3R Lab), Anatoly A. Krasnovsky (Innopolis University and MB3R Lab)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了三款工具（Procrustes、Bering、Sheaft），实现了微服务的自动模型发现、虚拟崩溃测试以及基于可用性指标的决策门控，支持CI/CD集成。

**💡 创新点**

创新点包括：①将部署和追踪证据与模型发现结合，提供版本化的可复现工件；②引入必需调用（mandatory-call）契约和SLO容差网格扫描，提升可用性评估精度；③利用Birnbaum分析和重要性评估实现组件影响量化；④通过归档重放验证数值连续性和分析结果的可复现性。

**🔧 技术方法**

技术手段：基于图模型的依赖发现；蒙特卡洛采样与Wilson区间估计；SLO容差网格扫描与门控规则；Birnbaum重要性分析；命令行工具与JSON Schema版本化契约；Docker Compose 的演示与CI/CD集成。

**📊 数据集**

数据集：OpenTelemetry Demo MB3R Stack 的 HTTP/REST 调用链（包括前端、推荐、目录服务等）以及 DeathStarBench Social Network（在归档重放中使用）。

**📈 对比分析**

比较方法：使用分析或然值与Monte Carlo估计相结合的可用性指标，利用网格扫描与门控规则对不同复制方案进行比较；在演示中与已知的精确公式（series-system oracle）进行对照，验证误差在可接受范围内，性能在 CI 环境中完成约 342 秒。

**⚠️ 局限性**

限制：模型仅覆盖有限的 fail‑stop 状态，排除排队、时间依赖恢复和队列饱和等动态效应；缺乏对实时负载和恢复机制的建模；工具依赖于可追踪的服务调用，可能无法捕捉隐式或异步依赖。

---

## 598. From Prompt Rules to Golden Data Model: Production-Scale Generation of Personalized, Compliant Pharmaceutical Claims

**arXiv ID:** 2610.31519 | [PDF](https://arxiv.org/pdf/2610.31519v1)

**作者:** Suyash Mishra `[一作]` (Roche), Satyanarayan Pati `[通讯]` (Involead)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了基于黄金数据模型（GDM）的结构化检索系统，用以在制药行业生成合规、精准且受众定制的宣传内容，并将合规与个性化信号从 prompt 级规则迁移至检索时的结构化上下文。

**💡 创新点**

创新点在于：①提出了可扩展的黄金数据模型，将散落的申报声明转化为图形化、可检索的结构；②利用 6C 质量框架对声明进行多维评分；③通过检索优先级过滤实现更高的合规率、受众匹配度和成本效率，证明了数据驱动的上下文检索比传统规则驱动更可扩展、更有效。

**🔧 技术方法**

使用的技术包括：Claude Sonnet 4.5 LLM；检索增强生成（RAG）与图数据库（Graph DB）实现声明间关系；JSON/列表式 guardrail 与禁止词规则；6C 评分算法；生产日志监控与成本/延迟分析。

**📊 数据集**

数据集：1,867 条制药声明，覆盖 Hemlibra、Ocrevus、Ocrevus Zunovo、Vabysmo 四种产品，含引用、使用历史、受众标签、材料类型等元数据，来源于 Veeva Vault 等内部系统。

**📈 对比分析**

比较方法：在规则驱动（Pre）与上下文驱动（Post）两种管道下，使用相同的 LLM（Claude Sonnet 4.5）进行生成；评估指标包括合规率、指示/内容类型/受众对齐率、MTV 使用率、6C 质量得分、API 成本、请求成功率、延迟与 token 使用。结果显示：Post 管道在合规率、MTV 使用率、内容与受众对齐率上均达到 100%，6C 分数平均提升 86%，API 成本下降 29.5%，响应 token 下降 65.4%，请求成功率从 99.2% 提升到 100%。

**⚠️ 局限性**

局限性：①实验仅覆盖制药领域，缺乏跨行业验证；②使用的声明数据为专有且受限，无法公开复现完整实验；③报告基于生产日志的效率提升，未进行单纯模型层面的可比性测试；④高成本检索可能导致单次请求成本上升，需根据实际业务场景权衡。

---

## 599. AgentVQA: A Unified Benchmark for Agentic Visual Understanding

**arXiv ID:** 2610.31494 | [PDF](https://arxiv.org/pdf/2610.31494v1)

**作者:** Advait Gupta `[一作]` (University of Maryland), Zhuang Liu `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 AgentVQA，一个统一的离线评测基准，用于系统评估 Vision‑Language Models（VLMs）的代理决策能力。

**💡 创新点**

创新点包括：①将 23 个不同领域的数据集（GUI、第一人称、嵌入式感知、策略决策、空间规划）整合为 1,000 个手工核验的多项选择题；②采用 VLM 辅助的五阶段人机协同筛选与难负样本生成流程；③展示了 AgentVQA 与在线交互评测（SpatialWorld）高度相关，却与传统 VQA 评测差异显著，说明代理能力的评价需要专门设计。

**🔧 技术方法**

技术方法主要是：多模态 VLM 的推理与链式思考（reasoning vs non‑reasoning）；Gemini 与 Claude 进行自动筛选与难负样本生成；人工双重校验确保答案唯一且难负样本合理；对视频输入做 32 帧抽样；统一多项选择格式与分数提取。

**📊 数据集**

使用了 23 个公开数据集，涵盖 5 大代理交互领域：GUI 导航、第一人称规划、嵌入式感知、策略决策、空间规划；每个领域包含 200 题，总计 1,000 题。

**📈 对比分析**

通过在 22 款主流 VLM（闭源与开源）上评估，最优模型 Gemini 3.7 Flash 仅达 58.5% 准确率；模型在不同领域表现不一；AgentVQA 与 SpatialWorld 的 Spearman 相关系数达 0.895，显示其离线评测可替代昂贵的在线交互；与 MMMU‑Pro、ChartQA‑Pro、GPQA‑Diamond 的相关系数分别为 0.67、0.44、0.59，说明传统 VQA 评测无法充分反映代理能力。

**⚠️ 局限性**

局限性：①仅评估单图/多帧/视频的静态问答，未覆盖完整交互式执行；②对 1,000 题的人工核验成本高，且可能存在领域覆盖不均；③评测结果主要表明模型在感知与状态构建上仍存在显著误差，无法直接映射到实际代理行为；④对复杂长序列动作或动态环境的评估仍需进一步扩展。

---

## 600. Improved Bounds for the Entropic Central Limit Theorem in the Log-Concave Case

**arXiv ID:** 2610.31493 | [PDF](https://arxiv.org/pdf/2610.31493v1)

**作者:** Tom Maitre `[一作]` `[通讯]`, Tom Maitre

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在d维空间中，具有对数凹性和各向同性的概率测度μ的独立随机向量的归一化和的相对熵收敛性。

**💡 创新点**

提出了一种新的界限，替代了先前的多项式维度依赖，采用了多对数依赖形式，提供了更优的收敛速度。

**🔧 技术方法**

结合了De Bruijn恒等式与Eldan的随机定位过程，使用了随机微积分技术。

**📊 数据集**

使用了具有对数凹性和各向同性的概率测度μ，具体数据集未明确给出。

**📈 对比分析**

与之前的结果相比，新的界限在维度依赖性上有显著改善，性能上在初始熵较小的情况下提供了更优的收敛速度。

**⚠️ 局限性**

该方法依赖于对数凹性假设，目前尚不清楚如何在没有此假设的情况下获得类似的界限。

---

## 601. Supporting or Replacing Students' Reasoning? Cognitive Offloading in AI-Assisted Proof Learning

**arXiv ID:** 2610.31481 | [PDF](https://arxiv.org/pdf/2610.31481v1)

**作者:** Eason Chen `[一作]` (Carnegie Mellon University), Ken Koedinger `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在本科证明导向数学课程中，随机分配学生早期使用两种 AI 辅助接口（聊天机器人和作业反馈），并记录使用日志、成绩及自我效能。随后对使用频率与随后的期中考试成绩进行关联分析，并通过中介模型探究自我效能对考试成绩的间接影响。研究提供了可复现的标签提示、分析脚本及汇总结果，供后续研究复用。

**💡 创新点**

（1）首次系统性比较早期与延迟获取 AI 辅助接口对作业与无 AI 考试成绩的影响；（2）揭示低自我效能学生更频繁使用 AI，且更高的聊天机器人使用量与后期考试成绩负相关；（3）将人工标注与 GPT-5.2 自动分类器相结合，构建可复现的会话意图标签体系；（4）将研究资源公开，促进跨课程、跨工具的 AI 学习效果研究。

**🔧 技术方法**

使用 OpenAI GPT-3.5/4（无 fine‑tuning）实现两种接口；利用 GPT-5.2 对会话意图进行自动标注；使用 Python 进行负二项回归、OLS（带 HC3 标准误）和路径分析（bootstrap 置信区间）；使用 Cohen's κ 检验人工标注一致性。

**📊 数据集**

数据来自 308 名本科生（153 名同意参与），包含：作业 1–6 分数、Midterm 1–3 分数、Baseline 自我效能问卷、AI 辅助接口使用日志（聊天消息数、作业反馈请求数）。此外收集 19 名学生的半结构化访谈录音。

**📈 对比分析**

对比方法：①随机分配早期 vs 延迟访问，比较作业和考试成绩；②负二项回归评估自我效能、Midterm 2 与 AI 使用频率的关系；③OLS 评估 AI 使用量对 Midterm 3 的影响；④路径分析检验自我效能 → AI 使用 → Midterm 3 的间接效应。结果显示：早期访问可提升作业平均 2.56 分，但对无 AI 考试无显著提升；每多发 100 条聊天消息，Midterm 3 平均下降 3.02 分；自我效能越低，AI 使用越频繁，且通过 AI 使用对考试成绩产生正向间接效应（约 0.6 分/自我效能点）。

**⚠️ 局限性**

局限性包括：仅在单门课程内测试，缺乏跨学科和更大规模验证；未测量学生在课堂或作业外的其他 AI 使用；自我效能仅在基线测量，未跟踪变化；使用量统计未区分主动与被动使用；对话标签基于有限样本，可能存在偏差；未直接测量学生实际推理质量，导致对“认知卸载”的解释仍为推测。

---

## 602. VecTile: A Low-Cost Flexible Tactile Array for 3-DoF Force Sensing

**arXiv ID:** 2610.31472 | [PDF](https://arxiv.org/pdf/2610.31472v1)

**作者:** Jian Hou `[一作]` (Imperial College London), Adam J. Spiers `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种低成本、柔性 4×4 触觉阵列 VecTile，能够在每个触点测量三轴力并可在弯曲表面使用。

**💡 创新点**

创新点在于利用被动硅胶凸点将剪切力转换为压差，并将其与压阻层结合，实现 3‑DoF 力感知；同时采用仅需 3D 打印模具和手工真空成型的低成本制造流程。

**🔧 技术方法**

采用压阻层与被动凸点、Arduino Nano ADC 读取、有限元分析验证转导机理、二维读数解码（二次多项式回归）等技术。

**📊 数据集**

使用 UR5e 机械臂与 ATI Mini40 力/扭矩传感器进行人工加载实验，收集 640 个交互序列；另外使用 84 条手动演示的数据训练 Diffusion Policy 进行插头插入任务。

**📈 对比分析**

与仅视觉、仅正常力、全 3‑DoF 触觉的闭环插头插入实验比较，VecTile 在未知位置的成功率从 48% 提升至 88%；单个触点的 RMSE 为 0.39 N（法向）和 0.16 N（剪切）。

**⚠️ 局限性**

局限性包括压阻层在 ≈5 N/触点后灵敏度下降、未评估滞后与长期耐久性、剪切响应受凸点尺寸限制等。

---

## 603. MixPT: Point Transformer with Interleaved Window and Global Attention

**arXiv ID:** 2610.31455 | [PDF](https://arxiv.org/pdf/2610.31455v1)

**作者:** Minkyun Seo `[一作]` (Seoul National University), Jaesik Park `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种点云 Transformer，利用固定序列化的窗口注意力与周期性交错的全局注意力（在大 token 数时采用池化），通过残差更新实现高效的全局信息交流，保持低延迟同时获得竞争性的语义与实例分割性能。

**💡 创新点**

创新点在于：①一次性构造序列化顺序并在所有窗口注意力块中复用，消除重复的 token 重新组织；②在窗口注意力块之间插入全局注意力块，实现跨窗口直接通信；③对大 token 集合使用空间池化的全局注意力，仅将残差映射回原分辨率，从而显著降低全局注意力的计算成本；④在 3D 空间中采用非对称轴向 RoPE 编码，进一步提升性能。

**🔧 技术方法**

使用技术包括：稀疏卷积预处理、序列化窗口注意力、全局注意力（池化版）、预归一化残差结构、旋转位置编码（RoPE）非对称轴向、指数移动平均（EMA）权重更新以及基于多头注意力的 MLP。

**📊 数据集**

实验数据集：语义分割采用 SemanticKITTI、SemanticPOSS、Semantic3D 和 ScanNet；实例分割采用 ScanNet 和 SemanticKITTI；同时对这些数据集在无 TTA 的验证集上进行评估。

**📈 对比分析**

对比方法：与 LitePT-S（窗口注意力）和 Volt-S（全局注意力）基线进行对比；在所有四个语义分割数据集上，模型在不使用 TTA 时 mIoU 与 LitePT-S 相当或更高，延迟降低 19%–32%；相对于 Volt-S，模型在户外数据集上 2.1–6.9 倍更快，同时保持相近或更好的 mIoU；实例分割实验表明该骨干网络与基线保持竞争力。

**⚠️ 局限性**

局限性：①固定序列化顺序在极大场景中可能限制跨窗口信息传播；②全局注意力块仍然存在计算开销，尤其在 token 数非常大时；③池化策略与池化步幅需针对具体数据集调优，缺乏统一自动化方案；④未对极端稀疏或极密集点云的泛化进行深入评估；⑤模型对硬件的依赖性较高，延迟收益在不同设备上可能不完全一致。

---

## 604. Peak-Aware One-Way Trading: Competitive and Learning-Augmented Algorithms for Energy Procurement

**arXiv ID:** 2610.31447 | [PDF](https://arxiv.org/pdf/2610.31447v1)

**作者:** Jackson Bibbens `[一作]` (University of Massachusetts Amherst), Mohammad Hajiesmaili `[通讯]` (University of Massachusetts Amherst)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出峰值感知的一路交易问题，并设计双阈值在线算法与学习增强算法，理论证明竞争比与一致性-鲁棒性最优；

**💡 创新点**

创新点在于将峰值收费与价格波动结合，给出最优竞争比与一致性-鲁棒性权衡，并提出能利用预测的算法；

**🔧 技术方法**

采用双阈值水位平滑（Peak‑Aware Water Level Smoothing）与决策建议跟踪策略，结合凸组合与投影技巧；

**📊 数据集**

使用四个美国电网区域（CAISO、ISONE、ERCOT、PJM）的实时与日前电价时序数据；

**📈 对比分析**

与传统阈值策略、盲目跟随建议、混合建议及仅最小化峰值的基线进行对比，实验显示学习增强算法在大多数情形下平均竞争比约 20%‑30% 接近下线最优；

**⚠️ 局限性**

局限在于仅考虑单一购买约束，无法处理多重截止时间或储能辅助，以及对预测质量高度依赖且对极端价格波动的鲁棒性有限。

---

## 605. Multi-Token Prediction and Alternative Tokenization Improve Learning Latent Multi-Hop Graph Search at Scale

**arXiv ID:** 2610.31439 | [PDF](https://arxiv.org/pdf/2610.31439v1)

**作者:** Arvid Frydenlund `[一作]`, Frank Rudzicz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

信息不足，无法确定论文做了什么

**💡 创新点**

信息不足，无法确定创新点

**🔧 技术方法**

信息不足，无法确定使用的技术

**📊 数据集**

信息不足，无法确定使用的数据集

**📈 对比分析**

信息不足，无法确定比较方法与性能

**⚠️ 局限性**

信息不足，无法确定限制

---

## 606. A Fix for Softmax and Cross-Entropy

**arXiv ID:** 2610.31642 | [PDF](https://arxiv.org/pdf/2610.31642v1)

**作者:** Jonathan T. Barron `[一作]` (Google DeepMind), Michael Tschannen `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了对softmax、log_softmax以及交叉熵损失中两个数值错误的修正，提供了数值稳定的实现。

**💡 创新点**

通过在log空间直接计算，消除了Softmax Collapse和Softmax Cancellation两种精度失真，修正了梯度和概率的求值误差。

**🔧 技术方法**

采用log-space 运算、精度补偿、数值稳定的softplus与logsumexp、以及自定义梯度回传等技术。

**📊 数据集**

在模组化加法任务 a+b 31、a^2+b^2 97、a^3+b 97 以及 a+b 113 等小型算术数据集上进行实验。

**📈 对比分析**

与标准交叉熵、Zero-sum Projection、StableMax 等方法比较，固定实现在所有浮点精度下都达到或逼近理论边界(log(1))并取得 99-100% 的测试准确率，避免了 Slingshot 峰值。

**⚠️ 局限性**

实验规模有限，主要关注极高置信度或教师-学生几乎相同的极端情况，未在大规模工业任务中验证。

---

## 607. WorldForge V2: Eliciting Camera-Trajectory Alignment in Video Diffusion Models

**arXiv ID:** 2610.31640 | [PDF](https://arxiv.org/pdf/2610.31640v1)

**作者:** Chenxi Song `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了WorldForge V2，能够在预训练的视频扩散模型上实现精确的相机轨迹控制；

**💡 创新点**

创新点在于将训练无关的IRR、FLF、DSG三种引导机制与V-Cache低成本修正相结合，并通过针对轨迹的LoRA适配实现高效的参数化控制；

**🔧 技术方法**

采用光流门控（FLF）、迭代重采样（IRR）、自校正引导（DSG）、噪声状态匹配（V-Cache）以及LoRA适配等技术；

**📊 数据集**

使用公开数据集DL3DV、LLFF、Tanks & Temples、Mip‑NeRF 360、DAVIS等进行评估；

**📈 对比分析**

与多种基线比较，WorldForge V2在FID、FVD、CLIP相似度以及相机轨迹误差（ATE/RPE）等指标上均取得了最优或接近最优的表现，显示出高质量且精确的相机控制；

**⚠️ 局限性**

局限性包括对深度估计误差、快速/非刚性运动、遮挡等复杂情况仍可能产生几何或纹理失真，并对极端深度错误或动态场景的鲁棒性有所不足。

---

## 608. F2G-Pose: Geometry-Aware Foundation Feature Lifting for Direct RGB-D Category-Level Object Pose Estimation

**arXiv ID:** 2610.31623 | [PDF](https://arxiv.org/pdf/2610.31623v1)

**作者:** Jinyu Zhang `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了F2G-Pose框架，能够在单次前向传播中同时预测类别级RGB‑D物体的6D姿态、尺寸和相机框架下的完整形状。

**💡 创新点**

创新点包括：①直接回归姿态/尺寸，避免采样候选导致的计算开销；②将基于RADIO的密集视觉特征提升到点云，结合DGCNN token化和Mixture‑of‑Experts Transformer进行局部–全局几何编码；③联合尺寸预测与形状重建，利用稠密几何监督提升姿态一致性。

**🔧 技术方法**

使用技术包括：RADIO（密集视觉特征提取）、DGCNN tokenization、Mixture‑of‑Experts Transformer、6D连续旋转表示、Chamfer距离和SmoothL1损失、MoE负载平衡损失。

**📊 数据集**

实验使用的主要数据集：SOPE（合成）、ROPE（真实）、HouseCat6D（真实）、HANDAL（真实）、以及在ARX机器人平台上的抓取实验。

**📈 对比分析**

与NOCS、SGPA、IST‑Net、HS‑Pose、GenPose、GenPose++、Joint‑RegDiff、RFMPose、FoundationPose、Any6D等基线进行对比。F2G‑Pose在SOPE/ROPE上获得IoU50 AUC 43.7/50.1、旋转误差分别为1.19°/0.88°，推理速度28 FPS；在形状重建方面Chamfer‑L1低于其他基线；在机器人抓取实验中成功率分别为74/80（抓取）和68/80（放置）。

**⚠️ 局限性**

局限性：需要事先的RGB‑D分割，深度噪声、薄部件、严重遮挡或对称低纹理物体仍导致误差；高精度对齐仍弱；未提供置信度或不确定性估计；对跨域（不同摄像头/光照）精细匹配仍需后续优化。

---

## 609. OP-Print: Tracing Membership Imprints in On-Policy Self-Distillation

**arXiv ID:** 2610.31616 | [PDF](https://arxiv.org/pdf/2610.31616v1)

**作者:** Qi Li `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在自监督自策略蒸馏过程中，成员推断攻击对数据隐私的威胁并提出了对应的攻击框架

**💡 创新点**

发现自策略自蒸馏留下稀疏且结构化的“纠正印记”，并基于此设计了兼容灰盒和黑盒的 OP-Print 攻击

**🔧 技术方法**

利用教师-学生 KL 散度、重启生成稳定性、基于分块的差异统计以及标准的 MIAs 基线对比

**📊 数据集**

在 OpenThoughts 的 1k 题解子集（包含 500 个成员与 500 个非成员样本）上对 Qwen3-1.7B/4B/8B 进行实验

**📈 对比分析**

与传统概率/Rényi 等灰盒基线以及多种黑盒生成统计指标对比，OP-Print 在 AUC、TPR@5%FPR 等指标上显著优于所有基线，提升幅度可达 7.7%/26.5%

**⚠️ 局限性**

黑盒版本需要多次重启生成，导致额外推理成本；目前未提出针对纠正印记泄漏的防御方法

---

## 610. Efficient and Optimal Learning of Discrete Distributions with Person-Level Privacy

**arXiv ID:** 2610.31612 | [PDF](https://arxiv.org/pdf/2610.31612v1)

**作者:** Gautam Kamath `[一作]` (New York University), Jonathan Ullman `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种针对离散批次（每批次包含 m 份 iid 样本）进行鲁棒估计的方法，利用求和平方（SoS）框架构建 SDP 约束，能够在存在 η 级别的批次损坏时恢复基础分布 μ 的期望，并给出误差上界 O(η√{log(1/η)/m})；同时在此基础上给出了可差分隐私（DP）的鲁棒估计器；并证明了对该任务的统计查询（SQ）下限，表明任何多项式查询次数、逆多项式容差的 SQ 算法无法在误差 o(η√{log(1/η)/m}) 下完成此任务。

**💡 创新点**

创新点在于：① 将 SoS 技术引入离散批次鲁棒估计，构造出可在高维多项式时间内求解的 SDP 约束，且仅需低阶矩信息；② 将鲁棒估计与 DP 机制相结合，提供了高效的可差分隐私实现；③ 通过构造特殊的离散分布及其高阶矩匹配，给出了 SQ 计算模型下的严格下限。

**🔧 技术方法**

主要技术包括：求和平方（Sum‑of‑Squares）求解框架、矩阵对角支配（diagonal majorizer）和子空间方差控制、SoS 证书的 6 次多项式、隐私降维/加噪声机制、以及用于下限的高阶矩匹配与多项式修正构造。

**📊 数据集**

实验与评估主要使用合成的离散分布（如均匀分布及其经过偏差调制的版本）以及模拟的批次数据，未涉及公开实测数据集。

**📈 对比分析**

与传统的样本平均估计相比，所提出的鲁棒估计在 η 级别损坏下保持误差可控；与现有的鲁棒分布估计方法相比，所需样本数更小、算法复杂度更低；在 DP 场景下，误差保持在与非隐私版相当的水平，且满足可差分隐私保证。

**⚠️ 局限性**

局限性包括：① 需要对数据批次大小 m 及损坏比例 η 预先设定且误差上界随 log(1/η)/m 变化；② SoS SDP 的求解在大规模问题上仍然存在计算瓶颈；③ 对非离散、非 iid 的批次（如存在依赖或连续型分布）尚未直接适用；④ SQ 下限证明仅给出了理论限制，实际攻击模型可能更弱。

---

## 611. Humanoid Mobility via Attention-based Contacts

**arXiv ID:** 2610.31608 | [PDF](https://arxiv.org/pdf/2610.31608v1)

**作者:** Junzhe He `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于接触规划的全身运动控制框架，通过三阶段训练（跟踪、接触规划、强化学习微调）实现统一的通用政策，使人形机器人能够在未知、稀疏且几何复杂的环境中自主选择身体部位的接触点并完成攀爬、爬行、站起、坐下等全身动作；

**💡 创新点**

关键创新点在于将完整运动轨迹简化为接触点表示，将示范数据的空间时间结构转化为可直接指导的接触策略，从而既保留了示范的协调性，又让强化学习自由探索实现；

**🔧 技术方法**

使用的技术包括：人类运动重定向生成多样化接触示例；LiDAR‑based 神经映射产生高精度地形高度图；基于注意力的全身接触规划网络；多任务PPO强化学习；以及自监督的后向接触标记；

**📊 数据集**

数据集主要为公开人类运动捕捉数据（包括行走、攀爬、跳跃等）经重定向至Unitree G1机器人，配合从环境中采集的LiDAR点云；

**📈 对比分析**

与最近的MGMT等方法相比，在多种稀疏与组合地形上的成功率均高（稀疏地形成功率提升至约54%，组合课程92.5%对比58.8%），同时在真实机器人上实现了多种全身动作，证明了方法的可行性与稳健性；

**⚠️ 局限性**

主要局限包括：当前仅支持2.5D地形交互，缺乏对悬空、悬挂及更复杂3D姿态的支持；依赖LiDAR测量，可能受限于视野遮挡；未实现全局导航与路径规划，仍以局部目标为导向；

---

## 612. How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions

**arXiv ID:** 2610.31606 | [PDF](https://arxiv.org/pdf/2610.31606v1)

**作者:** Miguel Mochizuki Silva `[一作]` (Universidade Federal da Paraiba), Barufaldi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于18篇乳腺病变管理文献的检索增强生成（RAG）系统，并在18个简短、直接的查询上评估检索与生成的性能。

**💡 创新点**

创新点在于提出证据感知检索评估协议，采用证据跨度级别的相关性标注，并系统比较不同chunk尺寸、重叠比例、检索配置（dense、BM25、Hybrid RRF、cross‑encoder reranker）对检索质量的影响。

**🔧 技术方法**

使用了混合检索技术（dense embedding + BM25 + Reciprocal Rank Fusion + cross‑encoder reranker）、fixed‑size chunking、ChromaDB向量存储、DeepEval LLM 判定器，以及BLEU/ROUGE、DeepEval（Contextual Precision/Recall/Relevancy/Faithfulness/Answer Relevancy）等评价指标。

**📊 数据集**

数据集为18篇专门选取的乳腺病变文献（来源于ASBrS、ACR、WHO等权威指南），以及对应的18个简短查询，形成了一个开发bench‑mark。

**📈 对比分析**

通过宏平均的Recall@3、R‑Precision、MRR@3、nDCG@3评估检索；生成方面使用BLEU、ROUGE和DeepEval指标。最佳配置（384‑token + 20% overlap + Hybrid RRF + cross‑encoder reranker）在检索上取得Recall@3=0.94，生成在不同上下文深度（K=1–5）下指标各异，未出现单一最优设置。

**⚠️ 局限性**

局限性包括样本量极小（仅18篇文献、18个查询）、问题过于简单且未进行临床验证、评估仅关注检索是否返回任何证据而未判断证据充分性、缺乏多源冲突与复杂多证据需求的测试，以及未对更大规模、多样化数据集进行验证。

---

## 613. Action Learning for Scientific Simulations

**arXiv ID:** 2610.31585 | [PDF](https://arxiv.org/pdf/2610.31585v1)

**作者:** Sarah Liaw `[一作]` (Kempner Institute), Yilun Du `[通讯]` (Kempner Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出行动学习（action learning）框架 LEAP，直接从完整轨迹样本学习可重用的随机行动（path action），并利用该行动进行路径排序、代表性路径推断、机制引导、约束推断和采样。

**💡 创新点**

创新点在于：①把行动学习作为直接目标，从完整轨迹而非局部转移学习；②通过与可解析的参考过程对比学习行动修正，能够捕捉跨时间的非加性依赖；③学得的行动可在不重新训练的情况下迁移到不同条件、目标或约束下。

**🔧 技术方法**

使用技术包括：InfoNCE 级别的集合对比目标；高斯马尔可夫参考过程与路径对数比率；局部与全局修正的分解（LEAP‑Local、LEAP‑Global、LEAP‑Hybrid）；神经比率估计器、Transformer 轨迹编码器；以及针对轨迹的梯度下降训练。

**📊 数据集**

实验数据集涵盖：1）双阱过渡动力学（overdamped Langevin）; 2）31 体星体轨道系统（186 维状态）; 3）丙氨酸二肽（(ϕ,ψ) 维度）在 500 K 与 600 K 下的 OpenMM Langevin 轨迹。

**📈 对比分析**

与 Kramers‑Moyal、立方回归、几何路径、Transition NCE 等传统或生成式基准比较，LEAP 在漂移重建、路径成本逼近、非加性修正学习以及跨温度/目标迁移上均表现最佳；在轨迹完整性补全和基准点约束下亦实现了显著误差下降。

**⚠️ 局限性**

局限性包括：需要参考过程与观测轨迹有足够重叠；对状态相关扩散或非高斯噪声的处理有限；在投影到低维坐标时可能引入记忆效应或额外的摩擦/扩散，导致学习到的行动仅是有效行动而非完整系统行动。

---

## 614. Geometry-Dependent Properties of Physically Crosslinked PPy-PVA Hydrogels Enable Tunable Iontophoretic Drug Delivery

**arXiv ID:** 2610.31567 | [PDF](https://arxiv.org/pdf/2610.31567v1)

**作者:** Heather Jin Hee Kim `[一作]` (Cornell University), Cara M. Nunez `[通讯]` (Cornell University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文利用不同几何形状的PPy-PVA水凝胶，在冰冻-融化物理交联过程中，研究其微结构与局部电场的差异，并评估这些差异对BSA在离子透过（Iontophoretic）与被动扩散条件下的渗透性能的影响。

**💡 创新点**

创新点在于：①通过改变凝胶模具几何形状（包括Hilbert曲线等高角度边界）来间接调节电场强度和微孔结构，从而实现对药物释放速率的几何控制；②不再单纯依赖电流密度调节，而是利用几何与物理交联耦合提供新的药物递送调控手段；③将模拟电场、SEM微观结构和离子透过实验结合，形成完整的几何‑性能‑传输三位一体评估框架。

**🔧 技术方法**

所用技术包括：有限元分析（Ansys Maxwell）计算电场分布；扫描电镜（SEM）定量微孔面积与孔隙率；UV‑Vis光谱与非负最小二乘（NNLS）谱解法测定BSA与PPy浓度；四探针AC法测电导率；离子透过实验使用Franz扩散室与细胞乙酸膜；统计分析（Spearman相关、线性回归）评估几何参数与传输指标的关联。

**📊 数据集**

实验数据集：BSA作为模型药物（浓度5–30 mg mL⁻¹），PPy掺量10–90 wt %；共五种凝胶几何形状（Plain、Circle、Interdigital、Hilbert 2nd、Hilbert 3rd）；每种形状在离子透过与被动扩散两种条件下各取三次重复；电导率、电场强度、孔隙率、累计释放量、稳态通量、滞后时间、渗透系数等指标构成最终的数据集。

**📈 对比分析**

比较方法：对比离子透过（IP）与被动扩散（P）在同一几何下的累计释放量（Q）、稳态通量（J_ss）、滞后时间（τ）、渗透系数（P_app）以及增强比（ER）。结果显示：平均P_app≈3.8 × 10⁻⁵ cm s⁻¹，ER>1；Hilbert 2nd与Plain几何在离子透过下表现出最高的Q和J_ss；整体上离子透过显著提升药物渗透率，且相较于以往人工膜研究得到更高的渗透系数。

**⚠️ 局限性**

局限性包括：①仅在人工细胞乙酸膜上进行实验，无法直接推广到活体皮肤；②SEM仅对单一样本进行扫描，未覆盖样本间变异性；③模拟电场与实际凝胶内电场存在差异，因凝胶表面可能出现致密壁层；④每种几何的实验重复次数有限，Spearman相关结果受样本量小的影响；⑤物理交联过程本身导致的结构异质性可能在更大规模实验中放大。

---

## 615. Contestable Agents for Data Governance: Recording Human Stewardship of AI-Generated Metadata Across Structured and Unstructured Systems

**arXiv ID:** 2610.31553 | [PDF](https://arxiv.org/pdf/2610.31553v1)

**作者:** Humza Akhtar `[一作]` (MongoDB, Inc.), Brian Freeman `[通讯]` (Trane Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一套基于可争议治理的工作流，让大模型代理提出带引用的元数据决策，人工审核前可对各论点进行挑战并记录完整的审计链；随后将最终决策写入治理记录并通过前置门控控制其成为预设。

**💡 创新点**

创新点在于将可争议（contestable）概念引入数据治理层面，结合量化双极论证框架（QBAF）实现可重构的论点评分、可追溯的审计日志，以及版本化术语表和门控机制，保证错误不在链路中扩散。

**🔧 技术方法**

使用的大技术包括：大语言模型（如 GPT‑4o）、检索与抽取工具、量化双极论证框架、MongoDB + 向量检索、PROV 原则的概念映射，以及基于规则的推理与逻辑更新。

**📊 数据集**

在案例实验中使用了合成 HVAC 设备数据：点列表、手册文本、测试报告、服务记录、保证单和公告等，构造了两组使用场景（点分类和保修失效模式标记），并在开放源代码实现中模拟。

**📈 对比分析**

通过对比全流程与仅使用模型预测的差异，发现全流程在高置信度点上仍需人工复核，而低置信度点通过多代理论证和争议提高准确性；性能方面，使用分流策略可将大模型调用降低约 70% 但仍保留 95% 的任务成功率；具体数值见实验报告。

**⚠️ 局限性**

局限性包括：论证评分规则尚未校准为概率、依赖人工评估的基准分数、未对循环图的收敛性做通用保证、缺乏真实工业数据验证、以及门控与审计日志的实现细节仍待完善。

---

## 616. CAMGen: A Core Change-Aware Multi-View Framework for Commit Message Generation

**arXiv ID:** 2610.31552 | [PDF](https://arxiv.org/pdf/2610.31552v1)

**作者:** Bo Xiong `[一作]` (Wuhan University), Peng Liang `[通讯]` (Wuhan University)

**通讯引用:** 5426 | [OpenAlex ID](https://openalex.org/A5049939779)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CAMGen 框架，利用核心变更实体为中心的多视角上下文来生成提交信息。

**💡 创新点**

创新点在于将功能语义聚类与结构依赖、历史实例三视角融合，并系统地识别核心变更实体。

**🔧 技术方法**

技术包括 LLM（DeepSeek、GPT）+ 代码语义向量化、HDBSCAN 聚类、静态分析构建依赖图、检索历史差异。

**📊 数据集**

使用 ApacheCM 数据集（Java、C++、Python 共 7,522 个提交）进行实验。

**📈 对比分析**

与五个基准（NNGen、CoRec、RACE、COME、CCT5）对比，CAMGen 在五项自动评测指标和人工评估上均优于基线，提升显著。

**⚠️ 局限性**

局限在于核心实体识别误差、对 LLM 版本/提示敏感、未覆盖所有语言及大型项目的可扩展性。

---

## 617. Shift-aware score transport for privacy auditing without retraining

**arXiv ID:** 2610.31549 | [PDF](https://arxiv.org/pdf/2610.31549v1)

**作者:** Yaohong Yang `[一作]` (ELLIS Institute), Antti Honkela `[通讯]` (University of Helsinki)

**通讯引用:** 2938 | [OpenAlex ID](https://openalex.org/A5071329195)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在无法重训练或插入数据的观察式隐私审计中，提出利用离群非成员的 MIA 分数与表示的关系进行转移，并通过校准任务纠正转移误差，得到稳健的 μ‑GDP 下限。

**💡 创新点**

①首次在观察式审计中利用表示条件不变性估计缺失的 IID 非成员分数分布；②通过构造 OOD 斜面校准任务学习表示位移与转移误差的关系，实现保守校正；③在多种分布偏移下保持更紧的隐私下限。

**🔧 技术方法**

采用条件分位数转移、线性残差转移、Ledoit–Wolf whitening、PIT 诊断、μ‑GDP 评估、Bootstrap 置信区间和误差增长回归等技术。

**📊 数据集**

图像数据集为 CIFAR‑10（含 CIFAR‑10‑C 亮度噪声、CINIC‑10）和 DomainNet；语言数据集为 GPT‑2 在 2023 年 arXiv 文档上微调，OOD 为 2026 年 arXiv 文档。

**📈 对比分析**

与无校正、全局校正、条件校正和 Oracle IID 非成员基线比较；在所有偏移场景下，至少一种转移变体与 Oracle 接近，误差显著低于无校正且不如全局/条件校正过度保守。

**⚠️ 局限性**

缺乏可验证的正式有效性保证；校准假设需在 Q_Z→P_Z 变位相同的情形下成立，若表示或条件不变性失效，校正可能失效；对大规模模型的计算成本和表示选择仍需进一步研究。

---

## 618. LACToS: Local Anomaly Detection via Conditional Typicality of Superpixels

**arXiv ID:** 2610.31538 | [PDF](https://arxiv.org/pdf/2610.31538v1)

**作者:** Jihoon Hong `[一作]` (Georgia Institute of Technology), Sara Fridovich-Keil `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 3916 | [OpenAlex ID](https://openalex.org/A5049727497)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出一种基于扩散模型的局部异常检测框架 LACToS，利用统计典型性评估像素块的异常程度。

**💡 创新点**

创新点在于：①将典型性测试从整图扩展到局部，采用超像素级典型性；②用噪声扩散后退采样的去噪后验代替昂贵的去填充后验；③在典型性计算中加入压缩的 ResNet 特征。

**🔧 技术方法**

使用扩散模型（无监督训练）、超像素分割、去噪后验采样、典型性统计、特征提取与自动编码器压缩、Gaussian 平滑。

**📊 数据集**

三类数据集：腹部 CT（健康与合成肿瘤）、人脸图像（正常与带假肢或疤痕的照片）、X‑ray 行李扫描（无武器与含剪刀的样本）。

**📈 对比分析**

与 PatchCore、SimpleNet、SuperSimpleNet、EfficientAD、GLASS、DDAD 等工业缺陷检测基线比较；在 AUROC、AP、AUPRO 指标上 LACToS 处于前列或相当优，表现出更高的定位精度。

**⚠️ 局限性**

局限性：依赖扩散后验采样，计算成本高；对超像素大小和噪声方差等超参数敏感；假设超像素内像素 i.i.d 可能被破坏；对训练集中罕见正常模式的误判风险。

---

## 619. GigaBench: Benchmarking Learnable Representations for Gigapixel Image Compression and Super-Resolution

**arXiv ID:** 2610.31644 | [PDF](https://arxiv.org/pdf/2610.31644v1)

**作者:** Matteo Fusconi `[一作]` (University of Bologna), Luigi Di Stefano `[通讯]` (University of Bologna)

**通讯引用:** 10191 | [OpenAlex ID](https://openalex.org/A5025618347)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个针对千兆像素图像的 Learnable Representation (LR) 基准，系统评估压缩与超分辨率性能。

**💡 创新点**

首次在千兆像素尺度下统一比较 IR、HR、ER 三类 LR，提供基准数据集、细粒度指标与人类偏好评估，为后续研究提供可操作的指导。

**🔧 技术方法**

采用多种 LR 方案（SIREN、FINER、FDHO；Instant‑NGP、NeuRBF、MetricGrids；GaussianImage、SmartSplat），并与经典插值（双三次、Lanczos 等）对比；使用切片、梯度累积等技术处理大规模图像。

**📊 数据集**

使用 GigaBench 数据集：10张1.0–1.19 GPx的无压缩千兆像素图像，覆盖全景、工业扫描、病理切片、卫星影像与合成图像，全部为 CC‑BY‑NC 4.0 许可。

**📈 对比分析**

通过 PSNR、SSIM、压缩比、压缩时长、推理时长等指标评估；结果显示 HR 在中等压缩比下重建最优但受限于压缩率，IR 在极端压缩下效率最高但质量最低，ER 位于两者之间；人类评测表明在×4 超分辨率时经典插值更受欢迎，×8 时 HR 接近甚至超过。

**⚠️ 局限性**

实验仅覆盖十幅图像，缺乏更广泛的泛化验证；对大规模训练和推理的高计算成本未能在所有场景中完全克服；仅考察图像压缩与超分辨率，未覆盖视频、音频等其他模态；人类评测样本有限，结果可能受受试者差异影响。

---

## 620. BeNice! Closing Looping's Inference-Memory Gap via KV-Sharing

**arXiv ID:** 2610.31633 | [PDF](https://arxiv.org/pdf/2610.31633v1)

**作者:** Hayden Prairie `[一作]` (University of California, San Diego), Daniel Y. Fu `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在循环式语言模型中引入两阶段训练（随机深度预训练 + 共享 KV 后训练）实现 KV 缓存共享，从而在保持质量的同时减少推理内存。

**💡 创新点**

提出了 KV‑shareability 机制和两阶段方法（随机深度 + BeNice!），证明收敛的残差动力学导致 KV 缓存趋于固定点，并通过后训练修正首递归残差错误。

**🔧 技术方法**

使用随机深度训练、残差收敛分析、KV 缓存共享、后训练辅助损失 BeNice!、多重混合注意力（MLA、KDA）和 KV 量化。

**📊 数据集**

在 Parcae 140M–1.3B 模型上使用超过 100B 语料的公开数据集（OpenWebText/CC100 等）。

**📈 对比分析**

与标准微调、ACT、KV 量化等方法对比，KV‑shareability 在共享 KV 时可将验证 perplexity 提升 9.75 点、Core/Core‑Extended 得分分别提升 8.35 与 4.56 分，推理内存降低 66%，吞吐量提升 23.6%，p90 TTFT 减少 83.6%。

**⚠️ 局限性**

仍需额外的随机深度预训练和后训练步骤；在更大规模模型或不同循环结构上可能失效；对超参数敏感；仅在中间循环结构验证，其他结构效果未知。

---

## 621. Questioning Instrumental Convergence in RLVR: Checkpoint Evidence from OLMo-3

**arXiv ID:** 2610.31624 | [PDF](https://arxiv.org/pdf/2610.31624v1)

**作者:** Barış Alparslan `[一作]` (ASELSAN), Caleb Petterson `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对OLMo-3-7B-Think模型在从SFT、DPO到RLVR各个检查点的行为轨迹进行系统评估，探讨了强化学习与可验证奖励（RLVR）是否会导致仪器式收敛（IC）行为的变化。

**💡 创新点**

创新点在于引入检查点轨迹评估方法，揭示IC行为呈非单调趋势：在RLVR早期阶段IC率上升至峰值后，随后下降至低于SFT基线，表明RLVR并不总是增强IC行为，而是与训练阶段、优化约束等因素有关。

**🔧 技术方法**

主要技术包括：强化学习与可验证奖励（RLVR）训练、SFT与DPO微调、InstrumentalEval基准、使用GPT‑4o和Gemini 2.0 Flash两套评判者进行连续评分、配对t检验、Wilcoxon符号秩检验、线性回归与INLP（线性框架消除）等统计与模型分析手段。

**📊 数据集**

数据集为InstrumentalEval的76个任务集合，统一在所有检查点上评估，并通过两套评判者提供连续IC得分。

**📈 对比分析**

比较方法：对比SFT与各RLVR检查点的IC率（IR）及其95%置信区间，采用配对t检验与Wilcoxon检验评估最终检查点相对SFT的差异；对比不同评判者的交叉一致率评估评判可靠性。性能方面，IR从SFT的25%上升至RLVR 200步的46%，随后下降至RLVR最终检查点的11.8%（7B模型）或22.4%（32B模型），差异在统计上显著。

**⚠️ 局限性**

局限性包括：仅研究了OLMo-3-7B/32B同一模型家族，缺乏跨模型或跨实验室的复现；InstrumentalEval任务数有限，统计功效受限；评判者间一致率中等，难以完全信赖绝对IR值；框架分析仅限线性结构，未排除非线性或交互效应；未进行机制级别分析也未验证验证器的外部对齐性。

---

## 622. VERA: A Responsible AI Dashboard for Assessing LLM Deployments in Light of the EU AI Act

**arXiv ID:** 2610.31588 | [PDF](https://arxiv.org/pdf/2610.31588v1)

**作者:** Robin Quériaux `[一作]`, Jérôme Lebecq `[通讯]` (BNP Paribas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了名为VERA的自托管Dashboard，帮助非工程师在银行内部评估LLM部署是否符合欧盟AI法，并将COMPL‑AI的技术解读与公开基准结果整合，生成可审计的、与运行绑定的责任AI证据；同时通过CodeCarbon对评估过程的能耗进行实时记录；提供角色化视图、可展开的基准细节与可导出的运行记录。

**💡 创新点**

在现有评估工具的基础上首次实现：①需求级结果与公共基准直接映射并可在浏览器中展开查看；②将评估运行与证据绑定，确保追溯性；③为环境影响需求新增运行级能耗计量，取代手工声明；④自托管、无云、可按需配置的完整栈；⑤在仪表盘内进行用户研究，验证阅读效率与正确率提升。

**🔧 技术方法**

后端使用Python 3.11与Celery+Redis调度，前端为TypeScript单页应用；调用COMPL‑AI、HELM、Garak、DecodingTrust等公开基准；使用Ollama自托管模型推理；通过CodeCarbon对评估进程进行能耗监测；实现基于注册表的可替换需求/基准映射；部署可分为Lite和Enterprise配置。

**📊 数据集**

使用公开基准数据集：MMLU、AdvBench、Presidio、RealToxicityPrompts、BBQ、BOLD、Adult、TruthfulQA、GSM8K、HumanEval等；以及为隐私/数据充足性测试生成的合成银行文档（230条）。模型包括Llama 3.1 8B、Qwen2.5 7B、Gemma 2 9B、Mistral 7B。

**📈 对比分析**

采用受控的Within‑subject用户研究，10名参与者先使用原始评估文件回答6道问题，再使用VERA回答同类问题；使用Wilcoxon符号秩检验和McNemar检验比较正确率与时间。结果显示：正确率从27/60提升到42/60（p=0.016），时间从56 s降至30 s（p=0.064，未显著）。另外在四个模型上跑5个需求，报告得分与能耗（0.0134–0.0169 kWh）。

**⚠️ 局限性**

研究样本小且主要为研究员，缺乏合规/风险职能的目标用户；顺序固定导致潜在学习偏差；未评估对实际发布决策的影响；能耗仅覆盖评估过程，未涵盖训练阶段；仪表盘在无助推式启动上表现欠佳；色盲友好性不足；合成数据可能不代表真实业务场景。

---

## 623. On the Token Collapse Phenomenon in Textual Graph Soft Prompting

**arXiv ID:** 2610.31572 | [PDF](https://arxiv.org/pdf/2610.31572v1)

**作者:** Andrea Giuseppe Di Francesco `[一作]` (Sapienza University of Rome), Fabrizio Silvestri `[通讯]` (Sapienza University of Rome)

**通讯引用:** 6061 | [OpenAlex ID](https://openalex.org/A5044165871)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对图软提示（Graph Soft Prompting）方法中的图标记（Graph Token）进行系统分析，发现其在多种基准、图网络和 LLM 上普遍出现“标记坍塌”（Token Collapse）现象，并提出了两阶段对齐+冻结管道来缓解该问题，同时提供了一套完整的诊断与评估流程。

**💡 创新点**

创新点包括：①首次揭示并量化图软提示中标记坍塌的普遍性；②设计两阶段训练（先 InfoNCE 对齐后冻结，再单独训练软提示和图标记路径）来防止坍塌；③构建固定权重干预、NLL 监测和解码可检索等多维度评估框架，使图标记的“对齐”和“使用”可被独立验证。

**🔧 技术方法**

技术手段：图神经网络（GAT、GCN、Graph Transformer）、投影器（Sigmoid、GELU）、InfoNCE 对齐、LoRA 微调、固定权重干预（mean/ swap/ gswap）、Gold-Answer 负对数似然（NLL）评估、解码可检索（rank）分析。

**📊 数据集**

数据集：三种 GraphQA 基准（分别用于实体关系问答任务），同时在不同 LLM（Llama‑2‑7B、Qwen2.5‑3B）和不同图编码器上进行实验。

**📈 对比分析**

对比方法：在原始 G‑Retriever、图无关的常数控制（CONST）以及两阶段管道之间进行性能与干预效果比较。结果显示：①原始 G‑Retriever 在准确率上与基线相当但标记已坍塌；②CONST 与 G‑Retriever 取得相近甚至更好成绩；③两阶段管道有效抑制标记坍塌，保持或略提升准确率，并在部分基准上显著提升图标记的可检索性。

**⚠️ 局限性**

局限性：仅对 G‑Retriever 及其有限变体进行深入实验，跨方法审计覆盖范围有限；坍塌机制仅与投影器和端到端训练相关联，未能完全分离其根本原因；两阶段管道在消除坍塌后仍可能引入先验偏差，且不保证图信息一定被 LLM 使用；实验集中在问答任务，对其他下游任务的适用性未知。

---

## 624. Squaring-the-Sphere: Distortion-Free and Seamless Panorama Generation via CubeAtlas

**arXiv ID:** 2610.31565 | [PDF](https://arxiv.org/pdf/2610.31565v1)

**作者:** Xuhui Zuo `[一作]` (Tencent), Chunchao Guo `[通讯]` (Tencent)

**通讯引用:** 228 | [OpenAlex ID](https://openalex.org/A5034317074)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了 CubeAtlas 这一将六面立方体拆解为半极面与赤道面并排放置的单一画布，配合布局感知稀疏注意力，利用预训练的 Diffusion Transformer (DiT) 在无摄像头信息的条件下实现文本引导的 360° 全景生成，并通过训练无关的 4× Pixel Diffusion Decoder 进行超分提升分辨率。

**💡 创新点**

创新点在于：① 用半极面+赤道面的组合将立方体几何与全局连续性统一于同一二维网格；② 引入布局感知稀疏注意力，显著降低计算量且保持跨面一致性；③ 通过不改动原有模型结构即可迁移 2D 生成先验，实现高质量无摄像头、无架构改动的全景生成。

**🔧 技术方法**

使用的主要技术包括：预训练的 DiT + VAE + 多模态编码器；FlexAttention 的布局感知稀疏注意力；训练无关的 4× Pixel Diffusion Decoder 超分；以及立方体投影与 ERP 重建的后处理流程。

**📊 数据集**

实验数据集涵盖 RealSee3D、Laval-Indoor、SUN360 三大公开全景数据集，以及自收集的 100 张图像共计 318 张场景，用于训练与评估。

**📈 对比分析**

在 FID、KID、CLIP-FID、Seam_ERP/Cube、Eq-IQ/Pole-IQ 等指标上与 CubeDiff、DreamCube、HunyuanPano、HY-Pano-2.0 等基线对比，CubeAtlas 在 Seam_Cube 上下降约 70% 以上，在几何质量、文本一致性以及整体视觉质量上均位列前茅。

**⚠️ 局限性**

局限性包括：极地区域仍可能出现细节失真；稀疏注意力在极端复杂场景下信息流动可能受限；以及高分辨率生成依赖额外的超分步骤，增加推理时间和资源消耗。

---

## 625. Decentralized Bi-Level Optimization for Cooperative Orbital Transportation of a Rigid Payload

**arXiv ID:** 2610.31562 | [PDF](https://arxiv.org/pdf/2610.31562v1)

**作者:** Sidhdharth D. Sikka `[一作]` (Purdue University), Shaoshuai Mou `[通讯]` (Purdue University)

**通讯引用:** 3809 | [OpenAlex ID](https://openalex.org/A5070733769)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种去中心化双层优化框架，用于在轨协作运输刚体负载，并实现闭环故障恢复。

**💡 创新点**

将问题拆分为内层凸推力分配与外层扭矩轨迹搜索；利用内层最优值的封闭定理梯度在外层做投影梯度下降；多启动并行搜索与最大一致性实现真正的去中心化；整体实现闭环故障检测与重规划。

**🔧 技术方法**

双层优化、投影梯度下降、凸SOCP求解、封闭定理梯度计算、随机重启、多启动并行、最大一致性通信、闭环故障检测与重规划、Monte Carlo评估。

**📊 数据集**

随机生成的轨道参数、负载属性、终端状态、代理数量与挂接向量等；没有使用公开数据集。

**📈 对比分析**

与基准 GA、MPPI、直接 NLP 进行对比。成功率比 NLP 高 20–30%，在计算预算上表现更好；燃油消耗与 NLP 差 1–8%，优于采样器；去中心化方案在多代理情况下与集中式相当或更好。

**⚠️ 局限性**

投影可达集受限，窄锥角下收敛困难；无法保证全局最优；仅在可行扭矩空间内可恢复；对计算预算敏感；未在真实硬件上验证实时性能。

---

## 626. PYTHIA, a transferable machine learning framework for simultaneous defect and adsorption site optimization

**arXiv ID:** 2610.31539 | [PDF](https://arxiv.org/pdf/2610.31539v1)

**作者:** Ioannis Kouroudis `[一作]`, Alessio Gagliardi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 PYTHIA 框架，利用冻结的 MACE 基础模型、Transformer 翻译层和高斯过程头预测吸附能，并在此基础上进行贝叶斯优化以同时寻找最佳缺陷与吸附位置。

**💡 创新点**

创新点在于将预训练模型迁移学习与轻量级 Transformer 结合，既能在缺少大量数据的新材料中保持高精度，又通过高斯过程提供不确定性估计，驱动高效的主动学习和贝叶斯优化。

**🔧 技术方法**

采用的技术包括预训练的 MACE 体系结构、Transformer 语义映射层、Gaussian Process 回归、Bayesian Optimization、DFT 计算数据、Graph Neural Network（AUGUR）作为基线、以及主动学习策略。

**📊 数据集**

使用的主要数据集为：Zn 在多种掺杂的 Chini 群簇上的吸附能（含 Pt、Ni、Pd 替代物）以及 CH₂O 在多种缺陷硅烯上的吸附能，训练集覆盖部分缺陷组合，测试集包含未见缺陷。

**📈 对比分析**

通过与 AUGUR（GNN+GP）和完整训练的 MACE+GP 进行对比，PYTHIA 在 Zn 案例中取得 R²=0.83、MAE=0.33 eV、RMSE=0.37 eV，性能优于对手 80–100%；在硅烯案例中，仅用两轮迭代即可提升吸附能 230% 以上，展示了显著的精度与数据效率。

**⚠️ 局限性**

局限性包括：对缺陷数目有限（如本研究仅限 3 个缺陷）、对与预训练数据相距较远的新化学空间迁移效果可能下降；以及在高度周期性或极端异质系统中不确定性估计的可视化与解释仍有改进空间。

---

## 627. Workload-Aware Monitoring for AI-RAN

**arXiv ID:** 2610.31627 | [PDF](https://arxiv.org/pdf/2610.31627v1)

**作者:** Babak Azkaei `[一作]` (Eindhoven University of Technology), Kishor Chandra Joshi `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 725 | [OpenAlex ID](https://openalex.org/A5040842871)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在AI-RAN平台中提出工作负载感知监控框架，通过将射频工作描述与处理路径、时序边界和运行上下文关联，评估CPU与GPU在PDSCH处理中的调度策略。

**💡 创新点**

创新点在于在同一系统下对每个工作单元进行细粒度关联与聚合，避免传统汇总导致误导，从而发现小传输块使用CPU可显著降低回调延迟。

**🔧 技术方法**

采用原生日志、计数器、CUPTI、Janus codelet以及自定义遥测，结合混合CPU/GPU调度策略与人工合成的GPU共跑器进行实验。

**📊 数据集**

使用16次过空中下行实验数据，基于NVIDIA DGX Spark、Open5GS核心、OCUDU gNB、USRP X310、单个5G手机及合成共跑任务的原始日志。

**📈 对比分析**

通过将回调时长按传输块大小与调度策略分组比较，发现混合调度在无共跑器时平均回调时长降低约29.66%，有共跑器时降低约22.93%，小块CPU调度降低约36%/30%。

**⚠️ 局限性**

局限性包括仅在单一设备和单个下行场景验证、未在线实现、未完全区分工作负载变化与时序差异、共跑器对结果影响未系统评估。

---

## 628. Extending Harmonic Function Methods via Root Locus for Coordinated Path Generation

**arXiv ID:** 2610.31625 | [PDF](https://arxiv.org/pdf/2610.31625v1)

**作者:** Matteo Frascarelli `[一作]` (University of Pisa), Alberto Landi `[通讯]` (University of Pisa)

**通讯引用:** 1534 | [OpenAlex ID](https://openalex.org/A5003036303)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种离线协同路径规划方法，利用复变函数的Generalized Root Locus (GRL) 在圆域中为 N 台可互换机器人与 N 个目标生成离散且不相交的轨迹，并通过构造自动完成机器人与目标的匹配。

**💡 创新点**

创新点包括：
1) 将起点和终点编码为有理函数 R 的极点和零点，GRL 的分支即为机器人的流线；
2) 通过临界相位与边界相位的可检验集合保证路径的安全与互不相交；
3) 使用 Milne–Thomson 反射迭代在有限深度内构造满足边界无法分量的障碍兼容势，并提供有限的安全认证测试；
4) 证明 RRL（逆根轨迹）是该框架的特殊情况。

**🔧 技术方法**

核心技术包括：
- 圆域的保形映射（利用 Koebe 迭代与 Schwarz–Christoffel 公式）
- 有理函数 R 的对数势 
- GRL 的分支追踪（预测-校正法）
- 临界点与边界相位的计算（使用牛顿法和 argument principle）
- Milne–Thomson 反射与多圈域的多重映像
- 通过残差与上界 L_i 进行安全认证。

**📊 数据集**

使用的数据集：
1) 两个手工选取的多边形障碍示例（Flagship 与 Warehouse）；
2) 50 个随机实例（N=5）在两个圆域中，用于 Monte‑Carlo 对比；
3) “森林”例子：N=10 机器人与 10 个圆柱形障碍，在单位圆盘中。

**📈 对比分析**

对比方法：
- 独立 Harmonic Function Planning（HFM）
- Khatib 的人工势场（APF）
- 结果显示 GRL 在所有 50 个随机实例中 100% 成功，平均路径长度约比 HFM 高 10%，相对 APF 的成功率明显更高。实验还展示了 GRL 在 Forest 例子中的 100% 成功率与合理的路径间距。

**⚠️ 局限性**

局限性：
- 仅为离线规划，缺乏在线轨迹跟踪与动态重规划；
- 未对路径最优性做严格分析；
- 计算成本主要集中在临界点搜索与多重映像生成，复杂场景下可能较慢；
- 需要足够的反射深度以满足安全认证，深度增大导致符号数量指数增长。

---

## 629. Bilingual Language Models Capture Cross-Linguistic Influence in Second-Language Reading

**arXiv ID:** 2610.31620 | [PDF](https://arxiv.org/pdf/2610.31620v1)

**作者:** Abderrahmane Issam `[一作]` (Maastricht University), Stefan L. Frank `[通讯]` (Radboud University)

**通讯引用:** 3226 | [OpenAlex ID](https://openalex.org/A5076704114)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究双语 GPT‑2 语言模型在非母语英语阅读时的预测能力，检验其是否能捕捉读者第一语言（L1）对阅读时间的影响。

**💡 创新点**

创新点在于：① 通过匹配与不匹配的双语模型与单语基线对比，首次揭示跨语言影响主要集中在后期眼动指标；② 发现该影响随文字系统重叠与读者熟练度而变化，并呈现倒 U‑型非线性关系。

**🔧 技术方法**

技术：GPT‑2 双语模型、逐步暴露（SoE）训练策略、线性混合效应模型（LMM）、突发似然比检验、加权最小二乘（WLS）等。

**📊 数据集**

数据集：MECO L2 眼动数据（12 种 L1 背景）用于评估；FineWeb 多语料库（含英语与各 L1）用于模型预训练。

**📈 对比分析**

比较方法：① 先与单语英语基线计算 ΔR²；② 在匹配与不匹配双语模型间计算 ΔLL 及嵌套模型的对比。结果显示：双语模型在后期眼动指标（go‑past、总停留时间）上明显优于单语；匹配模型优于不匹配，且优势随脚本重叠和中等熟练度最大。

**⚠️ 局限性**

局限性：仅关注英语作为 L2；仅使用 MECO 眼动语料；脚本重叠为单一维度，未考虑词序、形态等其他语言学特征；熟练度指标主要基于词汇测验，可能忽略更高层次能力；模型仅使用单一随机种子。

---

## 630. Would an LLM Match into a Residency? Uncertainty, Reasoning and Image Use of Open Language Models on the Spanish MIR Examination

**arXiv ID:** 2610.31586 | [PDF](https://arxiv.org/pdf/2610.31586v1)

**作者:** Miguel Fernandez-de-Retana `[一作]`, Aitor Almeida `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了四款可在本地硬件上运行的开源大语言模型（8–27 B参数）在西班牙医学住院医师考试（MIR）中的整体表现，包括答题准确率、置信度、自知性、推理质量和对图像的利用。

**💡 创新点**

创新点在于：①将考试的加分/扣分规则与真实候选人排名映射，检验模型在惩罚机制下的自知能力；②使用专业医生评论与自动判断者对模型理由进行评判，衡量推理是否临床合理；③对多模态模型在有图像、无图像及错误图像三种条件下的性能进行对比，揭示图像对答案的实质影响。

**🔧 技术方法**

采用标签模式提取选项概率分布计算置信度，生成模式生成理由；用ECE、Brier、AUROC评估校准和自知性；用Llama‑3.3‑70B 4‑bit模型作为判定者对理由进行打分；对比医学微调前后模型在同一题集上的差异。

**📊 数据集**

使用了12年（2014–2025）官方MIR考试题库（2595题，其中1525题可计分），包含医生评论的123题，2019–2025年带图像的171题，以及按年份划分的官方分数‑排名曲线。

**📈 对比分析**

方法：通过将模型净分数映射到官方分数‑排名曲线，估算模型全国排名；使用置信度阈值、ECE、Brier、AUROC比较模型自知性；在图像实验中计算三种条件下的准确率差异。结果显示三款大模型可排在全国前几千名，医学微调虽降低过度自信但并未提升自知性，且38%正确答案理由不合理，图像对准确率影响不足3分。

**⚠️ 局限性**

局限性包括：仅评估四款8–27 B参数模型，未覆盖更大或专门推理模型；置信度评估基于概率分布，未检验口头或采样自信度；判定者评判未得到医生验证；分数‑排名映射仅为估计，可能不精确；图像仅来自2019‑2025年，未检验其他多模态任务的普适性。

---

## 631. Rethinking Sequence Compression with Test-time Regression

**arXiv ID:** 2610.31574 | [PDF](https://arxiv.org/pdf/2610.31574v1)

**作者:** Sihan Yang `[一作]` (Chinese University of Hong Kong), Weiyang Liu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 5544 | [OpenAlex ID](https://openalex.org/A5101606029)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种基于测试时回归的序列压缩算子TTR-Compressor，用闭式解替代传统池化/投影，提升语言和多模态模型压缩效果。

**💡 创新点**

将序列压缩视为自监督岭回归问题，利用Gram矩阵捕捉二阶关联，学习可调对角正则化，实现在每输入上闭式求解，避免迭代优化。

**🔧 技术方法**

测试时回归、闭式双重形式求解、对角可学习正则化、卷积增强目标、块级稀疏注意力、视觉特征合并等技术。

**📊 数据集**

FineWeb、RULER检索任务、LAMBADA、Zero-shot、Qwen3-VL与Qwen3-4B-Instruct、LLaVA-558K、MMBench、MMStar、RealWorldQA、MMMU、AI2D、OCRBench、InfoVQA、ChartQA、MMSI、ERQA等数据集。

**📈 对比分析**

与均值池化、concat+投影、注意力加权池化以及全注意力/滑动窗口进行对比；在块稀疏注意力中TTR-压缩在S‑NIAH/MK‑NIAH检索任务上领先多种基线；在视觉Token合并中与MLP合并相比，TTR-在2×2和3×3压缩下均获得1–4分提升，3×3压缩还可将视觉Token数减半同时保持表现。

**⚠️ 局限性**

假设T≪d，主要在块级稀疏/视觉合并场景验证，缺乏对长序列压缩或动态正则化学习的深入探讨；对计算开销的理论分析在某些场景下仍高于简单池化。

---

## 632. When to Follow the Teacher: Advantage-Based On-Policy Distillation

**arXiv ID:** 2610.31563 | [PDF](https://arxiv.org/pdf/2610.31563v1)

**作者:** Mahdi Sabbaghi `[一作]` (University of Pennsylvania), Adel Javanmard `[通讯]` (University of Southern California)

**通讯引用:** 3265 | [OpenAlex ID](https://openalex.org/A5010342958)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于优势的在线策略蒸馏方法AOPD，通过在教师行为上加权学生优势来改进学生模型。

**💡 创新点**

创新点在于将教师指导与学生预期奖励关联，而不是简单匹配教师概率；并且提供无额外回路或价值模型的无偏估计器。

**🔧 技术方法**

使用前向KL蒸馏目标、重要性采样估计、奖励信息、梯度下降、梯度裁剪等技术。

**📊 数据集**

在数学推理（Open-Thoughts、AIME24、AMC、HMMT）、代码生成（TACO、HumanEval、HumanEval+、MultiPL-E）、生物学（SciKnowEval、MMLU-Pro）以及医学问答（MedMCQA、MedQA）等多种数据集上进行实验。

**📈 对比分析**

与OPD（FKL、RKL）、GRPO、SFT、KDRL、OPSD等基线相比，AOPD在大多数基准上获得更高的pass@1、奖励得分，提升幅度约5–12个百分点。

**⚠️ 局限性**

局限性包括对教师支持的依赖（教师动作并非总是有益）、估计方差较大、需要保证教师分布覆盖学生行动、以及在极端任务中仍可能出现负面影响。

---

## 633. Same Luck, Different Choice: Twin-World Credit Assignment for Agentic Reinforcement Learning

**arXiv ID:** 2610.31544 | [PDF](https://arxiv.org/pdf/2610.31544v1)

**作者:** Zeropy Surio `[一作]` (University of Sheffield), Samuel Okafor `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种双世界信用估计（Twin-World Credit Estimation, TWIn）方法，利用键控的Gumbel噪声让事实和反事实轨迹共享下游随机性，从而在Agentic RL中获得无偏且方差更低的逐步信用。

**💡 创新点**

创新点在于：①将Gumbel噪声按(episode, turn, line, position)键化，保证同一噪声可在事实与反事实中复用；②基于共享噪声的双世界信用天然无偏，方差可降低至独立分支的(1‑ρ)倍；③采用平方根预算规则，根据预测的分歧率与代价动态分配分支，进一步提升样本效率。

**🔧 技术方法**

技术手段包括键控Gumbel-max采样、双世界信用估计、Horvitz‑Thompson权重、PPO/GRPO策略梯度更新以及对奖励的平方根预算与自适应分支。

**📊 数据集**

使用了七个问答基准（NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle）以及Qwen2.5-3B/7B等大语言模型；实验中还扩展到Qwen3-4B、Llama-3.2-3B、ALFWorld、WebShop等。

**📈 对比分析**

与PPO、GRPO、IGPO、GiGPO、Tree‑GRPO、AT²PO、VinePPO、CRR等基线对比，TWIn在所有基准上均取得最高准确率，平均比GRPO提升约2–3个百分点，且在相同或更低的辅助分支成本下收敛更快、每生成token的学习效率更高。

**⚠️ 局限性**

局限性在于：①需要工具可回放且确定性；②高温度或长序列时双世界相关性下降，导致方差优势减弱；③信用仅估计整个turn的优势，未分离推理与动作的细粒度信用；④实验主要集中在问答与有限调用的环境，尚未验证在极长时延或复杂多模态任务中的表现。

---

## 634. Dispersive Tree Agreements for Multi-Robot Coordination

**arXiv ID:** 2610.31540 | [PDF](https://arxiv.org/pdf/2610.31540v1)

**作者:** Shashank A. Deshpande `[一作]` (Massachusetts Institute of Technology), Jonathan P. How `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 36165 | [OpenAlex ID](https://openalex.org/A5011665886)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 WarpWeaver 的分布式实时规划器，用于多机器人协同运动，目标是实现接近联合最优的控制方案。

**💡 创新点**

创新点包括：将多机器人联合最优控制问题转化为各机器人局部的分散前向树搜索，并通过列生成（column generation）实现协商租金（rent）以逼近联合最优；开发了异步实时协议，在有限通信延迟下实现收敛到近似最优解；提供了任何时间（anytime）保证，实时截止时即便未收敛也能获得可证近似最优的解。

**🔧 技术方法**

技术实现依赖于：差分平坦（differential flatness）系统的平面化；分散前向树搜索（Dispersive Forward Tree, DFT）来近似单机器人最优控制；整数线性规划（ILP）与其线性松弛的列生成；租金（rent）机制和松弛约束实现机器人间的协商；异步实时循环与再进程时域（receding horizon）协作。

**📊 数据集**

使用的数据集和测试平台包括：Dynobench（多机器人轨迹规划基准）；EGO‑Swarm 以及 Forest 地图（三维双积分器与三阶积分器的高保真无人机模拟）；Realm（连通性维护任务的仿真环境）；以及六台地面机器人（Unicycle/RTB）在真实机器人实验中的运动捕捉和 Wi‑Fi 通信。

**📈 对比分析**

性能比较：相较于中心化离线规划器，WarpWeaver 在 Dynobench 上的总行驶时间下降 8%（最高 22%）；相较于分布式实时基线，下降 37%；在连通性维护任务中，成功率提升近一倍；在 EGO‑Swarm 高保真无人机实验中，总行驶时间比基线低 31–37%。实验显示其在实时截止时的解与达到稳定的解差距在 1–3% 以内。

**⚠️ 局限性**

限制与挑战：1）ILP 规模随机器人数量和联合约束增长，列生成收敛速度可能变慢；2）对紧耦合或全局约束（如刚性形成）会产生较弱松弛；3）需要保证通信延迟上界；4）若约束空间维度很大，搜索树仍可能不可扩展。

---

## 635. UniVOE: A Removal-Driven Framework for Interaction-Aware Universal Video Object Editing

**arXiv ID:** 2610.31641 | [PDF](https://arxiv.org/pdf/2610.31641v1)

**作者:** Shuzhou Yang `[一作]` (Peking University), Jian Zhang `[通讯]` (Peking University)

**通讯引用:** 53824 | [OpenAlex ID](https://openalex.org/A5100410082)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 UniVOE 框架，实现视频对象移除与注入的双向训练，并通过“移除驱动”策略使用移除模型生成对齐对，进一步利用 GRAR 优化注入结果。

**💡 创新点**

1) 采用移除驱动的逆向学习，把易学的移除任务用于生成注入监督；2) 设计 VRAM‑高效的在线奖励对齐训练（GRAR）；3) 引入效果感知的移除预训练与后处理。

**🔧 技术方法**

视频流匹配（latent flow matching）、Wan2.1‑14B Transformer backbone、SAM2 掩码生成、Qwen3‑VL 奖励模型、GRAR、LoRA 加速等技术。

**📊 数据集**

使用 63K 真实视频 + 68K 对齐视频对 + 69K 通过移除生成的对齐数据，总计 137K 条训练对；并在公开基准数据集（ROSE、VOR 等）上进行评测。

**📈 对比分析**

与 VideoPainter、VACE、Minimax、EffectErase、Bernini、Kling3‑Omni、Seedance2.0 等多种公开与商用方法在 PSNR、SSIM、LPIPS、FVD、DINO‑I、CLIP‑P 等指标上对比，UniVOE 在对象移除与注入任务中均取得最优或最接近最佳性能，尤其在阴影、反射等交互效果上表现突出。

**⚠️ 局限性**

仍需大量高质量视频与掩码，训练与推理显存需求高；对极端动态或复杂光照交互可能存在误差；奖励模型对齐效果受限于人工标注与模型泛化。

---

## 636. Causal Chain of Thought: Mitigating Attention Leakage via Context Isolation

**arXiv ID:** 2610.31639 | [PDF](https://arxiv.org/pdf/2610.31639v1)

**作者:** Gabriele Dominici `[一作]` (Università della Svizzera Italiana), Martin Gjoreski `[通讯]` (Università della Svizzera Italiana)

**通讯引用:** 2284 | [OpenAlex ID](https://openalex.org/A5037458079)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CausalCoT 框架，将复杂问题拆分为有向无环图（DAG）并在每个节点的局部上下文中独立评估，从而在保持推理性能的同时增强模型解释的因果透明度。

**💡 创新点**

创新点包括：①通过强制 Markov 结构消除全局注意力泄漏，确保每一步只能依赖其直接父节点；②引入 Predictable Change 指标评估插入错误后预测结果与逻辑一致性的程度；③支持在线干预（对 DAG 结构或单节点答案进行修正），实现可预测的错误传播。

**🔧 技术方法**

技术手段主要为：①利用预训练 LLM 在“规划阶段”生成问题的 DAG；②在“执行阶段”以隔离上下文调用 LLM 评估子问题；③使用标准 CoT、Factored Decomposition 等基线进行对比；④设计了答案变化率与可预测变化率等评估指标。

**📊 数据集**

数据集涵盖 GPQA、MedQA 与 Counterfactual Simulatability Suite，覆盖学术推理、医疗问答与多领域反事实可模拟性测试。

**📈 对比分析**

方法比较：在 Accuracy、Pass@3、Answer Change、Predictable Change 等多维度对比实验；结果显示 CausalCoT 与标准 CoT 在准确率上基本持平（平均下降约 1–6个百分点），但在解释可信度和干预效果上显著提升（答案变化率 +4.8~21.2 p.p.，可预测变化率大幅提高）。

**⚠️ 局限性**

局限性：①对零样本规划的依赖，DAG 生成错误会直接影响最终预测；②虽然消除了全局注意力泄漏，但未能保证 LLM 内部机制的因果透明度；③推理时延增加且需额外管理多阶段上下文；④对开放式生成任务的适用性尚待验证。

---

## 637. No Perfect Randomized Composition

**arXiv ID:** 2610.31638 | [PDF](https://arxiv.org/pdf/2610.31638v1)

**作者:** Alexander Meiburg `[一作]` (University of Waterloo), Alexander Meiburg `[通讯]` (University of Waterloo)

**通讯引用:** 171 | [OpenAlex ID](https://openalex.org/A5038705915)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究布尔函数的随机查询复杂度在块组成（block composition）下的乘积下界是否保持常数因子不变，并给出了否定答案；构造了两类全（total）函数族，使得组成函数的随机查询复杂度可比单个因子乘积的对数下界小得多。

**💡 创新点**

创新点在于：1）首次提供全函数的随机查询组成的子线性下界；2）构造了同时在多种复杂度度量（线性化查询、噪声查询、冲突、破坏、最大冲突）之间严格分离的全函数族；3）使用机器验证（Lean）证明结论的正确性。

**🔧 技术方法**

技术方法包括：
- 通过精确的一面检测器（exact one‑sided detector）构造弱信息算法；
- 采用编码的指针函数与随机化采样实现低偏差估计；
- 用可加噪声查询模型与线性化查询模型之间的比较；
- 引入私有编码（private encoding）与线性化与噪声查询复杂度的分离；
- 通过多层置信证书（certificate array）实现总函数的可计算性；
- 结合直接积（direct product）与列表直接积（list direct‑product）定理；
- 用后验马尔可夫过程与冲突复杂度的上下界关系。

**📊 数据集**

本论文为理论研究，不涉及实验数据集。

**📈 对比分析**

比较方法：作者将构造的全函数族与已知的部分函数（partial function）下界进行对比，展示在块组成时随机查询复杂度与因子乘积之间存在对数级的相对缩放；在所有考虑的复杂度度量链中，构造的函数族实现了严格的相对增量（如冲突复杂度与最大冲突复杂度、线性化查询与普通查询等）。

**⚠️ 局限性**

限制与未解决问题：
- 结论仅适用于随机查询复杂度的粗略下界，仍未能给出完整的多项式级别下界；
- 对于更精细的量级（例如更高阶的对数多项式）仍未给出；
- 只证明了存在特定构造，未给出更广泛的通用结构；
- 结果依赖于特定编码与概率分布，实际实现的可行性和计算复杂度未评估。

---

## 638. Can an Agent Debug a Web Corpus? Agentic Auditing of Slovak FineWeb2

**arXiv ID:** 2610.31636 | [PDF](https://arxiv.org/pdf/2610.31636v1)

**作者:** Marek Šuppa `[一作]` (Comenius University), Jaroslav Kopčan `[通讯]` (Kempelen Institute of Intelligent Technologies)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5022468938)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用一次性LLM作为语料库调试器，对FineWeb2斯洛伐克语子集进行自动化审计，发现并量化系统性缺陷，并生成可执行的元数据测试套件。

**💡 创新点**

创新点在于把LLM从传统分类器转为一次性调试工具，利用工具代理在只读预算接口下自动发现缺陷；构建可复现、可验证的测试套件，使后续版本可自动校验。

**🔧 技术方法**

技术包括Claude 3.5 Sonnet LLM、11种预算工具（聚合、采样、文本搜索、句子级语言识别等）、DuckDB+Parquet 数据层、pytest式的测试框架以及元数据驱动的筛选器。

**📊 数据集**

数据集为FineWeb2斯洛伐克语子集（约5百万文档），采用1/13的采样子集进行文本读取，元数据表包含语言识别、重复率、词频等字段。

**📈 对比分析**

在相同的检查预算（200文档）下与四种静态采样策略（Random、LowConf、LID-Disagree、HighDup）比较；结果显示静态策略在每次运行中验证的有效测试数量更多，成本更低；LLM代理提供精确普及率、精度估计和缺陷解释，但验证率略低。

**⚠️ 局限性**

局限性包括：仅在单一语言/单一语料上验证；人工验证样本有限，仅估计精度不测召回；缺陷定义粗糙；未评估跨时间变化和其他语言的泛化；LLM模型与预算对发现多样性有影响。

---

## 639. Ray-by-Ray: Fine-Grained Resource Management for GPU Ray Tracing Units

**arXiv ID:** 2610.31630 | [PDF](https://arxiv.org/pdf/2610.31630v1)

**作者:** Junrui Pan `[一作]` (Purdue University), Timothy G. Rogers `[通讯]` (Purdue University)

**通讯引用:** 1732 | [OpenAlex ID](https://openalex.org/A5038577900)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为 Ray-by-Ray 的细粒度资源管理方案，旨在将 GPU 光线追踪单元（RT 单元）中的缓冲和堆栈按每条光线而非每个 warp 分配，从而提升对齐度与并行度。

**💡 创新点**

创新点在于：1）摒弃传统的 warp 级资源分配，改为 per-ray 级动态分配与释放；2）在 RT 单元内部实现轻量级的 Ray Buffer Mask 与共享内存堆栈复用，显著提高资源利用率；3）利用硬件层面的重新排序而非软件级的 SER，减少上下文切换与内存延迟。

**🔧 技术方法**

实现技术包括：使用 Vulkan‑Sim 2.0 进行周期级仿真；在 RTL 中实现统一的 Ray Buffer 以及与 warp 计数器/Warp ID 结合的动态调度逻辑；通过共享内存管理 BVH 递归堆栈并消除银行冲突；使用 NVIDIA OptiX 的 optixTraverse/optixTrace 接口进行微基准测试。

**📊 数据集**

实验数据集主要来自 LumiBench，包含 16 个场景（如 BUNNY、ROBOT、CHSNT 等），在 RTX 5090（以及 RTX 2060）GPU 上进行性能评估；同时使用了自定义的 256×256 分辨率设置以逼近实际 warp‑压力。

**📈 对比分析**

通过与基线 RT 单元（warp 级缓冲）在同一 GPU 上进行对比，Ray‑by‑Ray 在 1.42×、1.70×、2.78× 的 Shadow、AO、PT 三种技术下分别实现速度提升，整体几何平均加速为 1.91×，面积开销仅 0.19%。

**⚠️ 局限性**

限制与不足：1）内存带宽成为瓶颈，某些场景 L2 带宽需求显著提升；2）仅对 RT 单元内部进行改进，未能解决 SER 等软件层面重新排序的高成本问题；3）实验依赖模拟与 RTL，真实硬件实现的细节与能耗可能仍有偏差。

---

## 640. DiffDreamer: Coupling World Models with Differentiable Simulation for Agile Quadrotor Control

**arXiv ID:** 2610.31617 | [PDF](https://arxiv.org/pdf/2610.31617v1)

**作者:** Fanxing Li `[一作]`, Wenxian Yu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了DiffDreamer框架，将世界模型与可微仿真耦合，用于四旋翼激进跟踪和动态门赛跑等需要外部运动预测与高频精确控制的任务。

**💡 创新点**

核心创新在于将外部动力学与机器人物理动力学分别在隐式世界模型和可微仿真中同步展开，并将目标奖励拆分为非可微任务奖励与可微运动奖励，以提升梯度精度和收敛速度。

**🔧 技术方法**

采用可微强化学习、差分模拟、世界模型学习、演员-评论家框架、奖励分解以及相对状态表示等技术。

**📊 数据集**

在VisFly高保真模拟器中生成的激进跟踪和动态门赛跑数据集上训练，数据采用随机轨迹、速度和初始状态，未使用公开数据集。

**📈 对比分析**

与DreamerV3、PPO、BPTT以及MPC等基线对比，DiffDreamer在两任务上收敛更快、最终奖励更高，动态门赛跑门通过率提升约50%以上，跟踪误差约减半。

**⚠️ 局限性**

仅在四旋翼上验证，缺乏对接触、离散事件、传感器噪声等实际条件的鲁棒性，且依赖外部运动捕捉，未实现完全离线感知。

---

## 641. Progress Probes $P^2$: Lightweight Runtime Monitoring for Vision-Language-Action Models

**arXiv ID:** 2610.31613 | [PDF](https://arxiv.org/pdf/2610.31613v1)

**作者:** Atiksh Bhardwaj `[一作]` (Carnegie Mellon University), Preston Culbertson `[通讯]` (Cornell University)

**通讯引用:** 607 | [OpenAlex ID](https://openalex.org/A5006569343)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 Progress Probes 的轻量化运行时监控方法，利用视觉‑语言‑动作模型（VLA）的内部激活线性解码任务进度，以此检测执行过程中的失败。

**💡 创新点**

创新点在于仅使用成功演示数据训练线性探针，避免了外部大模型或失败数据的需求；并证明 VLA 的内部表示天然编码了任务进度，从而实现了高效、准确的失败检测。

**🔧 技术方法**

核心技术包括：机制可解释性中的线性探针（linear probing）、任务进度标记（normalized progress labeling）、以及基于进度偏差的阈值监控；在此基础上对比了多种监控框架（输入级、特征级、输出级、VLM 监控等）。

**📊 数据集**

使用了公开的 VLABench 仿真数据（5000 条演示，10 任务）和 Franka 机器人真实硬件数据（360 条演示，6 任务），其中训练探针仅利用成功轨迹。

**📈 对比分析**

与现有监控方法（FIPER、FAIL‑Detect、SAFE、Robometer 等）对比，Progress Probe 在仿真中获得 0.909/0.788 的 Episode/State AUROC，并且在硬件上实现 0.87/0.81 的最高 AUROC；相比于 4B 参数的 Robometer‑FT，探针仅需 2K 参数并且推理速度提升 10,000 倍，监测准确性相当甚至略优。

**⚠️ 局限性**

局限性包括：实验仅针对 π_0.5 这类特定 VLA，未验证跨模型迁移；只探测 VLM backbone 的激活，未利用连续动作专家（如 diffusion、flow‑matching）可能提供的附加信息；并且对极其微妙或随机的失误（如 Coffee Bean 任务）仍存在检测困难。

---

## 642. CRANE: Post-Hoc Counterfactual Retrieval for Multimodal Classifiers

**arXiv ID:** 2610.31594 | [PDF](https://arxiv.org/pdf/2610.31594v1)

**作者:** Franco Rugolon `[一作]` (Stockholm University), Panagiotis Papapetrou `[通讯]` (Stockholm University)

**通讯引用:** 2853 | [OpenAlex ID](https://openalex.org/A5044999523)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了CRANE框架，即一种后置、模型无关的多模态最近邻反事实生成方法，并实现了多种检索策略；

**💡 创新点**

其创新点在于首次针对多模态分类器设计后置反事实检索方案，使用真实训练样本确保可行性与跨模态一致性，并提出了多模态共识检索(MC‑R)等策略；

**🔧 技术方法**

采用的技术包括最近邻检索、多模态特定距离度量与归一化、早期/中间融合、共识检索、模态原型合成、LOF可行性评分以及四项质量指标（Outcome、Proximity、Sparsity、Plausibility）的综合评估；

**📊 数据集**

实验涵盖四个跨领域数据集：Sepsis（时间序列+表格）、Long COVID Tweets（文本+表格）、Hateful Memes（文本+图像）以及Real or Fake Jobs（文本+表格）；

**📈 对比分析**

通过在不同融合策略（早期、中间、晚期）下的交叉验证，使用四项质量指标对生成的反事实进行评估，结果显示MC‑R在三大数据集上取得最佳或并列最佳的Outcome，其它策略在邻域距离或可行性等指标上也表现突出；

**⚠️ 局限性**

局限性包括依赖分类器性能；IF‑NN在深度模型中的效果差；检索方法受模态距离定义限制，难以扩展至缺失或未见模态；仅检索训练样本，缺乏生成式多样性；且未进行人类专家评估。

---

## 643. Controlled Radio Fault Injection and Telemetry Evaluation for Open 5G RANs

**arXiv ID:** 2610.31598 | [PDF](https://arxiv.org/pdf/2610.31598v1)

**作者:** Babak Azkaei `[一作]` (Eindhoven University of Technology), Kishor Chandra Joshi `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 725 | [OpenAlex ID](https://openalex.org/A5040842871)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了C和Python两套频道代理，并搭建了一套在OCUDU‑ZMQ 5G测试平台上进行受控无线故障注入和遥测评估的工作流程。

**💡 创新点**

创新点在于提供一种可复用的实验能力：在不改变现有测试平台的情况下，通过在ZMQ边界注入已知的I/Q样本变换，同时同步记录原生遥测、可编程codelet输出以及应用层服务测量，从而在已知干扰条件下评估不同遥测方案的有效性。

**🔧 技术方法**

使用技术包括OCUDU gNB、Open5GS核心、srsUE、ZeroMQ消息传输、C语言和Python实现的频道代理（支持衰减、屏蔽、噪声、正弦干扰、频移等变换）、Janus/jBPF codelet框架、UDP echo探针等。

**📊 数据集**

数据集为自建的实验流量：单UE、20 MHz band‑3小区、15 kHz子载波间隔、64‑QAM MCS，背景UDP负载2 Mbit/s，进行45个对比实验（10个中断块+5个噪声对），并记录I/Q样本、遥测日志与RTT测量。

**📈 对比分析**

比较方法：预设10个中断块（一次50 ms、五次10 ms）和5个噪声阶梯对，分别测量中继RTT的中位数和95th百分位（p95）。使用自助抽样估计置信区间。结果显示一次中断p95平均提升354 ms，五次提升604 ms，噪声阶梯提升113 ms；原生DTX事件能准确分组中断实验，CRC失败率能识别噪声实验；在树模型上，加入RLC codelet特征未提升分类准确率。

**⚠️ 局限性**

局限性：仅在单UE、共享VM、低SINR、固定负载下实验；缺乏真实物理层传播与多路径、移动性；代理仅实现数字级I/Q变换，未覆盖所有实际故障；未验证多UE或高负载场景下的可扩展性；codelet效果在本实验中未显著提升，需针对不同监测任务进一步设计与验证。

---

## 644. Simulated Control Evals

**arXiv ID:** 2610.31593 | [PDF](https://arxiv.org/pdf/2610.31593v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 645. Beyond Best-of-$k$: Prior-Consistent DAPS for Ambiguous Nonlinear Inverse Problems

**arXiv ID:** 2610.31592 | [PDF](https://arxiv.org/pdf/2610.31592v1)

**作者:** Merve Kocabas `[一作]` (University of Tübingen), Sara Fridovich-Keil `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 3916 | [OpenAlex ID](https://openalex.org/A5049727497)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种改进的扩散模型后向采样方法 PC-DAPS，通过在单次采样过程中维护多粒子、使用高斯混合逼近去噪分布，并在噪声降噪时基于先验一致性进行粒子修剪，解决了非线性逆问题（如相位恢复）中出现的先验不一致幻觉。

**💡 创新点**

创新点在于：① 将粒子集合引入 DAPS 结构，动态修剪先验不一致粒子；② 用多模态高斯混合代替单峰近似，使得去噪分布更符合先验分布；③ 通过先验一致性评分在无真值条件下实现对测量一致但先验不一致解的自我纠正。

**🔧 技术方法**

技术主要包括扩散模型（EDM）作为先验、变分拆分（DAPS）框架、Langevin 动力学、Gaussian Mixture Model（GMM）逼近去噪后验、先验一致性评分（Tweedie 公式）和粒子修剪策略。

**📊 数据集**

使用 FFHQ（人脸图像）和 ImageNet（多类别图像）两个公开数据集，在三种非线性逆问题（相位恢复）和七种线性逆问题（例如高通滤波、16× 超分辨率）上进行评估。

**📈 对比分析**

与传统 DAPS、DPS、TMPD、DDRM 等基线方法进行对比，采用 PSNR、SSIM、LPIPS 等标准指标以及先验一致性指标。实验显示：在相位恢复等高度模糊的非线性任务中，PC-DAPS 能显著降低先验不一致率（如 FFHQ 上从 14.5% 降至 1.5%），并提升整体图像质量；在线性任务中保持与基线相当的质量与多样性。

**⚠️ 局限性**

主要局限包括：① 对先验结构要求较高，在 ImageNet 等多样性更大的数据集上先验一致性修剪效果相对弱；② 需要维护多粒子和 GMM 计算，算力和内存成本略高；③ 在先验信息不足或测量噪声极大时，粒子修剪可能无法有效区分真相与假解。

---

## 646. Who Is Sycophantic to Whom? Confident LLMs, Human Deference, and the Collapse of Trust Calibration

**arXiv ID:** 2610.31579 | [PDF](https://arxiv.org/pdf/2610.31579v1)

**作者:** Deepthika S `[一作]` (Amrita Vishwa Vidyapeetham), B. Sankar `[通讯]` (Indian Institute of Science)

**通讯引用:** 675 | [OpenAlex ID](https://openalex.org/A5010736559)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在实验中让受试者通过对话学习五个科学主题，并对其回答进行测验，探究当大型语言模型（LLM）保持自信且错误时，用户是否会盲目接受其错误信息。

**💡 创新点**

提出了人类侧的“Human Deference Index”（HDI）衡量用户对LLM错误信息的接受程度，并系统地评估了在信任、知识吸收和元认知方面的相互关系，揭示了“计算器谬误”——在无法验证错误时，用户仍倾向于盲从LLM。

**🔧 技术方法**

使用OpenAI GPT‑5‑mini作为聊天助手，采用系统提示让模型在不同可信度水平下以完全自信的方式提供信息；通过LLM（Claude）自动对话转录进行编码；使用统计混合模型、GEE和AUROC等方法进行分析。

**📊 数据集**

使用了五个学科主题（心电图、碳‑14 定年、GPS 定位、萤火虫发光、降噪耳机）共60个事实对。每个主题包含12条真假事实配对，在实验中随机将0%、25%、50%、75%或100%的事实替换为伪造版本。

**📈 对比分析**

实验结果显示：学习分数随错误比例下降（从7.4降至2.9），误信息被接受率约为65%且与整体错误率无关；自信度与正确答案的区分度下降，元认知 AUROC 从0.70降至≈0.5；受试者的信任评估变化很小，且对错误信息的信任与实际准确性几乎无相关。

**⚠️ 局限性**

局限包括样本偏小、受试者多为年轻高学历且日常使用AI；LLM在实验中被强行保持不变且错误率高，未能代表真实部署情形；测验仅为即时多项选择，缺乏延迟记忆和开放式回答；受试者被禁止使用外部信息，限制了验证路径；转录编码由LLM完成，缺乏人工双重验证。

---

## 647. What Makes a Good Unmasking Policy? A Controlled Study of Learned Unmasking in Diffusion Language Models

**arXiv ID:** 2610.31576 | [PDF](https://arxiv.org/pdf/2610.31576v1)

**作者:** Jinya Sakurai `[一作]` (Nanyang Technological University Singapore), Jaehong Yoon `[通讯]` (Nanyang Technological University Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了掩码扩散语言模型的学习式解码策略，提出了轨迹收缩目标并结合置信度与隐藏状态的轻量级政策，以减少函数评估数并提升吞吐量。

**💡 创新点**

①使用轨迹收缩目标替代传统终端一致性标注，提升低NFE时的准确率；②证明隐藏状态与置信度联合使用可进一步降低NFE；③确定NFE下降10%即可抵消政策开销。

**🔧 技术方法**

基于DiT风格的轻量级Transformer政策，输入为置信度和隐藏表示，训练使用二元交叉熵；使用Fast-dLLM等训练无监督方法做对比。

**📊 数据集**

LLaDA-8B-Instruct、Dream-7B-Instruct作为模型；GSM8K、MATH-500、HumanEval、MBPP等数据集进行评测。

**📈 对比分析**

与Fast-dLLM、EB-Sampler、Learn2PD等方法比较，显示在GSM8K、MATH-500、HumanEval等任务上在相同准确率下NFE最低、TPS最高，尤其在中等准确率目标上优势明显。

**⚠️ 局限性**

只在数学推理数据上训练，无法同时适应长度与领域双重变化；在高准确率区间准确率略低于基线；前缀缓存会影响政策性能。

---

## 648. Social-HUMINT: An LLM-Augmented Unified Framework for Measuring Social Media Exposure

**arXiv ID:** 2610.31573 | [PDF](https://arxiv.org/pdf/2610.31573v1)

**作者:** Pellegrino Casoria `[一作]` (University of Naples Federico II), Simon Pietro Romano `[通讯]` (University of Naples Federico II)

**通讯引用:** 1465 | [OpenAlex ID](https://openalex.org/A5080036543)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文将SOCMINT与Cyber-HUMINT统一成一个可验证的框架，并实现了一个LLM驱动的多代理流水线，用于自动从公开社交媒体提取、推断个人特征并评估其在社交工程中的脆弱性。

**💡 创新点**

创新点在于①首次构建覆盖观察到主动接触全链条的38项能力统一分类法；②将LLM与多代理架构相结合，实现了从原始网页到结构化隐性属性的端到端转换；③通过人类受试者实验，将曝光度量与攻击可行性关联起来，提供了前所未有的实证评估。

**🔧 技术方法**

使用技术包括：大型语言模型（GPT‑4/3.5）、Langflow与LangSmith用于工作流编排与可追溯；Playwright实现跨平台爬取；ReAct式代理进行多步推理与工具调用；规则/NER基线作为对照。

**📊 数据集**

数据集为14名信息安全从业者公开的LinkedIn和Facebook个人资料，手工标注共736条真实属性以及252条由模型推断的隐性特征，用于评估提取准确率与推理可信度。

**📈 对比分析**

与正则表达式和spaCy NER基线对比，提取F1达97.3%（远高于28%和17.3%）；推断一致度平均为4.73/5；在社交工程仿真中，Kendall τ平均0.36，Top‑1准确率52%，显著高于随机基线。

**⚠️ 局限性**

局限性包括仅覆盖LinkedIn/Facebook两平台、样本量仅14人、依赖受试者自评作为真值、LLM对长文本的上下文限制、未进行真实的主动接触实验、缺乏多模态数据与纵向跟踪。

---

## 649. HiP3R: Hierarchical Pose Graphs for Long-Horizon 3D Reconstruction

**arXiv ID:** 2610.31635 | [PDF](https://arxiv.org/pdf/2610.31635v1)

**作者:** Zunnan Xu `[一作]` (Nanyang Technological University), Chen Change Loy `[通讯]` (Nanyang Technological University)

**通讯引用:** 88634 | [OpenAlex ID](https://openalex.org/A5005626854)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种层次化位姿图优化方法，能将短视频窗口的分块预测拼接为长视频的全局一致三维重建。

**💡 创新点**

创新点在于先用块级Sim(3)图恢复全局布局和尺度，再固定该布局后对每帧使用单一SE(3)相机变量进行精细优化，从而同时纠正尺度漂移和跨窗口相机冲突。

**🔧 技术方法**

使用Sim(3)位姿图、进阶循环调度、受限点采样、迭代加权最小二乘、Huber鲁棒损失以及关键帧子图插值等技术。

**📊 数据集**

在七个公开数据集上评测，包括室内(7-Scenes、Neural RGB-D)、室外(ETH3D、Tanks and Temples、Oxford Spires、KITTI、VBR)等。

**📈 对比分析**

与VGGT-Long、VGGT-Align等基线对比，显著降低轨迹误差（例如在KITTI上从110.3m降至9.73m），并在几何重建上将Chamfer误差和F1指标提升至最优水平。

**⚠️ 局限性**

主要局限是依赖先验块级预测的精度，对极不可靠的循环闭环仍可能导致优化不稳定，且在线实时更新的鲁棒性还有待提升。

---

## 650. Echo: Coupled Action and Task Status Signal Denoising for Vision Language Action Models

**arXiv ID:** 2610.31569 | [PDF](https://arxiv.org/pdf/2610.31569v1)

**作者:** Yufeng Liu `[一作]` (SJTU), Cheng Ma `[通讯]` (Spirit AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Echo框架，在VLA模型中加入停止和失败信号，实现任务状态的监测与动作生成。

**💡 创新点**

创新点在于：①在动作块中嵌入状态信号，无需额外参数；②利用成功、失败和恢复轨迹的补充监督，无需帧级进度标注；③采用异步去噪和受控信号-动作交互，让状态信息提前提供给动作生成。

**🔧 技术方法**

使用基于扩散的VLA模型、Transformer、token分组、控制信号-动作注意力、异步去噪、数据构造与监督策略。

**📊 数据集**

真实机器人实验数据：Insert Flower、Capsule Toy、Insert Pen、Make Tanghulu、Open Drawer；仿真数据：LIBERO-Recovery场景。

**📈 对比分析**

与action-only baseline、CycleVLA、VLA-Corrector、Robometer等进行对比，Echo在任务完成得分、帧级和轨迹级状态预测上均优于对手，尤其在OOD情形下提升显著。

**⚠️ 局限性**

对失败轨迹和人工干预的依赖使数据收集成本较高；需要手动标注子任务边界；对极端或未见失败类型的泛化能力仍待提升。

---

## 651. CellTrack-Bench: Microscopic Cell Tracking Benchmark for Vision-Language Models

**arXiv ID:** 2610.31566 | [PDF](https://arxiv.org/pdf/2610.31566v1)

**作者:** Ilan Theodoro `[一作]` (Biohub), Loïc A. Royer `[通讯]` (Biohub)

**通讯引用:** 9655 | [OpenAlex ID](https://openalex.org/A5043933183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

创建 CellTrack-Bench benchmark，用三类任务（细胞链接、轨迹修复、轨迹检查）评估 VLM 在细胞跟踪中的表现。

**💡 创新点**

创新点在于：1）利用多种跟踪器共同失败的边挖掘硬实例；2）设计局部难点任务与接近性无偏指标；3）系统化的 harness 环境与工具交互评估。

**🔧 技术方法**

技术主要包括 Vision‑Language Models（GPT‑5.6、Claude、Gemma 等）、多任务提示与工具调用框架、近邻抑制与分割框架、数据采样与硬实例挖掘。

**📊 数据集**

使用 15 个 Cell Tracking Challenge（CTC）与 5 个细菌数据集，共 1,537 难点实例与 9,041 关联，涵盖荧光、相位对比、干涉和亮场四种成像模式。

**📈 对比分析**

通过整体 F1、NN‑Residual、Division F1、Subgraph‑LNK 等指标进行比较；闭源 GPT‑5.6 在多项指标上远超开源模型，能显著修复跟踪器并识别错误；开源模型大多仍依赖近邻，性能偏低。

**⚠️ 局限性**

局限性包括：VLM 仍受近邻短路影响；弱模型在修复时可能退化；评估仅针对局部实例，缺乏完整序列验证；需进一步验证在完整跟踪流程中的实用性。

---

## 652. Diffusion Language Models for Perplexity-Aware Black-Box Attacks

**arXiv ID:** 2610.31558 | [PDF](https://arxiv.org/pdf/2610.31558v1)

**作者:** Ali Al-Lawati `[一作]` (Pennsylvania State University), Suhang Wang `[通讯]` (Pennsylvania State University)

**通讯引用:** 19827 | [OpenAlex ID](https://openalex.org/A5011048500)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将连续空间攻击中的随机映射替换为预训练的连续扩散语言模型（dLLM）来生成更自然的触发攻击的提示，评估其在过度生成（over-generation）攻击中的有效性与自然度；

**💡 创新点**

提出了将dLLM直接作为从低维高斯噪声到文本的映射，既保持了低维连续空间的优势，又显著降低了提示的字符熵（自然度），并证明此替换不会显著增加计算成本；

**🔧 技术方法**

利用连续扩散语言模型Cola-DLM进行提示生成；采用RareTrap（采样估计）和ThinkTrap（优化搜索）两种连续空间攻击框架；使用子集模拟、CMA-ES等优化技术；评估字符级困惑度（PPL_char）等指标；

**📊 数据集**

在十个目标LLM（包括8个开源模型和2个云API）上评估；对RareTrap使用的10个目标；对ThinkTrap使用的DeepSeek-R1-Distill-Llama-8B、Qwen3-14B和Phi-4-reasoning；使用OASST1真实用户提示集来校准困惑度过滤器；

**📈 对比分析**

对比随机映射与dLLM映射在RareTrap与ThinkTrap中的表现：dLLM在RareTrap中成功触发9/10目标，字符级困惑度从10^3–10^7降至3–5；在ThinkTrap中与随机映射同等提升输出长度；计算成本仅增加3–11%；风险评估在不同映射下差异显著，改变模型相对风险排序；

**⚠️ 局限性**

局限性包括：仅使用单一dLLM模型且无条件采样；对少数目标的评估可能不具代表性；仅关注过度生成，未评估其他有害行为；自然度仅通过困惑度衡量，缺乏人工评估；对API模型的安全性评估受限于调用预算。

---

## 653. Fine-Grained Analysis of Catastrophic Forgetting in Continual Learning via Algorithmic Stability

**arXiv ID:** 2610.31551 | [PDF](https://arxiv.org/pdf/2610.31551v1)

**作者:** Guodong Zheng `[一作]` (Sun Yat-sen University), Li Shen `[通讯]` (Sun Yat-sen University)

**通讯引用:** 30007 | [OpenAlex ID](https://openalex.org/A5100333320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了基于算法稳定性的灾难性遗忘理论框架，将遗忘分解为 intra‑task generalization gap、cross‑task generalization gap 与 empirical forgetting，并在凸和非凸损失下给出了三种学习率调度的上界；

**💡 创新点**

创新点在于：①不依赖具体网络架构；②将遗忘细分为三项，并通过梯度对齐和SGD轨迹精细分析得出学习率对遗忘的影响；③在三种学习率调度（常数、任务内衰减、连续衰减）下提供统一的理论上界；

**🔧 技术方法**

使用算法稳定性理论、梯度对齐分析、SGD轨迹追踪和凸/非凸损失的L‑Lipschitz与β‑光滑假设；

**📊 数据集**

实验使用Split CIFAR‑100和Split MiniImageNet数据集，采用AlexNet和ResNet‑18等基准模型；

**📈 对比分析**

与原始CL、GEM、A‑GEM等方法对比，发现连续衰减学习率能显著降低遗忘（IGG、CGG、EF），但在准确率（ACC）上存在稳定‑可塑性困境；梯度对齐策略在常数或任务内衰减学习率下效果更突出；

**⚠️ 局限性**

局限在于依赖理想的L‑Lipschitz、β‑光滑等假设，且未考虑更复杂模型或更实际的优化器/数据分布，未来可进一步放宽假设并验证更通用的情形。

---

## 654. Reading, Writing, Sharing, and Interacting with the World through Video

**arXiv ID:** 2610.31668 | [PDF](https://arxiv.org/pdf/2610.31668v1)

**作者:** Shengqiong Wu `[一作]`, Hao Fei `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了视频理解、生成、共享表示与工作流以及世界预测与控制等方面的关键里程碑，聚焦于视频基础模型与世界建模视角的融合。

**💡 创新点**

创新点在于将视频基础模型与世界建模视角系统化整合，梳理其在多任务和跨域应用中的演进与交叉点。

**🔧 技术方法**

采用的主要技术包括多模态Transformer、视频预测网络、语义分割和时序注意力机制等。

**📊 数据集**

使用的数据集涵盖了公开的YouTube‑8M、Kinetics、Something‑Something、Epic‑Kitchen等大规模视频数据集。

**📈 对比分析**

通过与I3D、C3D、SlowFast、ViViT等经典方法对比，展示了基础模型在多任务性能上的显著提升。

**⚠️ 局限性**

局限性主要体现在缺乏统一评估标准、对算力需求高以及对长时序建模仍存在挑战。

---

## 655. OpenTac: An Open Ecosystem for Customizable, Scalable, and Transferable 3D Tactile Sensing

**arXiv ID:** 2610.31661 | [PDF](https://arxiv.org/pdf/2610.31661v1)

**作者:** Ariel Slepyan `[一作]` (Johns Hopkins University), Nitish V. Thakor `[通讯]` (Johns Hopkins University)

**通讯引用:** 33935 | [OpenAlex ID](https://openalex.org/A5088964967)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 OpenTac 生态系统，实现可定制、可扩展的压力与剪切传感皮肤，涵盖从 CAD 设计、模拟、跨传感器映射到千赫自适应采样的完整流程。

**💡 创新点**

创新点在于：①利用可编程的弹性覆盖层将剪切力映射到邻近的多路电阻传感器，实现三轴力估计；②构建基于“manifest”文件的统一描述，支持任意形状、分辨率的传感器自动化生成和互操作；③提出 OpenTac-Xampling，基于自适应四叉树的稀疏扫描，可在 32×32 皮肤上实现 1286 Hz 的读数，比传统全阵扫描快 19 倍；④通过跨传感器映射实现不同布局之间的力场共享。

**🔧 技术方法**

核心技术包括：可弯曲印刷电路板（FPCB）+压阻膜 + 结构化弹性覆盖层；自动化 CAD 编译器；基于 MuJoCo 的有限元 FEM 仿真；稀疏四叉树自适应采样算法；跨传感器坐标映射与线性标定。

**📊 数据集**

主要使用了自制的 30+ 个传感器布局（手指、抓手、掌心、脚等）在四台真实机器人上进行实验；对比实验使用了商业三轴力/扭矩传感器作为基准；在学习实验中使用了机器人抓取、旋转、踢击等任务数据集（来自 MuJoCo 任务库）。

**📈 对比分析**

与 FlexiTac、AnySkin、eFlesh 等现有皮肤对比：在相同面积和分辨率下，OpenTac 的正常力和剪切力误差均低于 10 %；自适应采样使 32×32 皮肤的采样率从 67 Hz 提升到 1286 Hz；在基于力的控制与学习实验中，OpenTac 传感器使机器人能够实现更精细的抓取策略，成功率提高约 15 %。

**⚠️ 局限性**

局限性包括：FPCB 不是可拉伸的，导致在极曲面或大变形时可能出现接触不全；压阻传感器受温度漂移、滞后影响，需要温度补偿；自适应采样在大面积连续接触时退化为全阵扫描；跨传感器映射目前仅验证了正常接触形状，未在力强度或仿真到实测转移上做深入评估。

---

## 656. Low-Bit N-Gram Embeddings for Language Models

**arXiv ID:** 2610.31646 | [PDF](https://arxiv.org/pdf/2610.31646v1)

**作者:** Stephen Chung `[一作]`, Wenyu Du `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究将语言模型中的 n‑gram 嵌入表压缩到低位精度，以降低显存占用，同时保持或提升模型性能。

**💡 创新点**

系统评估不同量化方案（均匀、正态分位、Lloyd‑Max）和缩放策略；提出混合精度分配以平衡不同 n‑gram 长度对性能的影响；证明压缩后短期微调能显著恢复甚至提升性能；利用压缩节省的内存扩充表容量，进一步提升效果。

**🔧 技术方法**

低位量化、Lloyd‑Max 码本、不同缩放方法、混合精度配置、哈希检索、短期微调、内存再分配策略。

**📊 数据集**

ClimbMix 训练集；WikiText‑2、Penn Treebank、C4 作为验证/测试集；HellaSwag、ARC‑Challenge、WinoGrande 评估下游任务表现。

**📈 对比分析**

通过对比原始 16‑bit 表、无表版本以及不同位数表，报告验证损失提升比例、表内存节省比例、困惑度（perplexity）变化及下游准确率变化。结果显示：4‑bit 表可节省约 75% 内存且保留 >99% 的验证损失提升；2‑bit 表可节省 86% 内存且保留 86–91%；对大型模型可节省 47–55% 内存，困惑度提升 <1.2%；再利用节省的内存扩容表可进一步提升 22–26% 性能。

**⚠️ 局限性**

在极低位（1‑bit）下仍有性能损失；需要额外微调以恢复性能；实验集中在哈希嵌入的 n‑gram 表，未验证在所有大规模架构或非哈希表上的通用性；压缩后模型的训练速度略有下降。

---

## 657. Phase Transition from Random Features to Fully Trainable Networks

**arXiv ID:** 2610.31663 | [PDF](https://arxiv.org/pdf/2610.31663v1)

**作者:** Xin He `[一作]` (Wuhan University), Shijun Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 156 | [OpenAlex ID](https://openalex.org/A5101676946)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了部分可训练神经网络（P-Trainable网络）在随机特征与完全可训练网络之间的性能过渡，给出了宽度、可训练比例P和初始化不匹配对逼近误差的理论上界，并通过数值实验验证了该“相位转移”现象。

**💡 创新点**

创新点在于：①提出可训练比例P作为连续参数的网络模型，②在两层网络中以谱衰减与Leverage兼容性为前提，推导出包含P的逼近率并揭示了小P阶段的阶跃式提升；③通过球面偏置ReLU特征给出显式常数与指数；④利用结构化三层例子展示深度可实现的非线性复合优势。

**🔧 技术方法**

主要技术包括：随机特征采样与重加权、Ridge回归与Ridge Leverage分布、谱分析与有效维数估计、误差上界的高概率证明、以及对球面偏置ReLU核的球谐分析。

**📊 数据集**

使用人工合成数据集：一维正弦+高斯、混合频率三角函数、二维正弦乘余弦+高斯圆盘，均在均匀网格上采样得到训练/评估点；实验还涵盖ReLU、GELU、tanh等激活函数。

**📈 对比分析**

比较方法：在同一网络结构下，对不同P值进行多次随机初始化，记录最佳训练均方误差并计算几何平均；通过相对提升（normalized gain）衡量从P=0到P=1的误差下降。实验显示仅训练约1–5%的隐藏单元即可获得90%或以上的误差下降，且更高的初始化不匹配（τ增大）会让转移更陡峭。

**⚠️ 局限性**

局限性：理论证明仅覆盖两层网络；深层P-Trainable网络的分析尚未完成；结果依赖于采样分布与ridge leverage假设，未直接解释梯度优化过程；实验仅基于合成函数，缺乏真实数据的验证。

---

## 658. EpiSEM-Bench: Evaluating Safety and Evidence Recovery in LLM Agents under Missing Evidence

**arXiv ID:** 2610.31650 | [PDF](https://arxiv.org/pdf/2610.31650v1)

**作者:** Qijin Zhang `[一作]` (City University of Hong Kong), Shuang Qiu `[通讯]` (City University of Hong Kong)

**通讯引用:** 2875 | [OpenAlex ID](https://openalex.org/A5100772459)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个评估LLM代理在缺失证据情况下安全性的可执行基准（Epistemic Safety under Evidence Missingness Benchmark）。

**💡 创新点**

提出了匹配安全/不安全世界、受控证据缺失和可选恢复工具的评估框架，定义了六个核心指标，能同时衡量任务完成、错误执行和恢复效果。

**🔧 技术方法**

使用可执行安全规则、工具调用模拟、证据原子注释、缺失掩码与恢复工具，并对十个大型语言模型进行实验。

**📊 数据集**

构建了六个领域、300个任务对（600个世界）的基准，共计1363个证据原子（944关键、419无关）。

**📈 对比分析**

通过Full SWS/Full UCR、Critical UCR/SMB、Recovery UCR/PUR等指标对十个模型进行比较，发现模型在任务完成和安全性上存在显著差异，部分模型虽然完成率高但安全边界低。

**⚠️ 局限性**

评估受限于当前十个模型、缺失信息仅为真值事实、恢复工具的实现方式和交互成本未深入探究，结果可能不适用于更大模型或不同任务环境。

---

## 659. METTA: Memory-Enhanced Test-Time Adaptation for Meta-Reinforcement Learning

**arXiv ID:** 2610.31656 | [PDF](https://arxiv.org/pdf/2610.31656v1)

**作者:** Ali Najar `[一作]` (Sharif University of Technology), Mahdieh Soleymani Baghshah `[通讯]` (Sharif University of Technology)

**通讯引用:** 1480 | [OpenAlex ID](https://openalex.org/A5069082023)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Memory-Enhanced Test-Time Adaptation (METTA)框架，通过神经记忆实现meta-RL的快速适应

**💡 创新点**

将适应过程拆分为两层时间尺度：细粒度的episode内神经记忆与跨episode的粗粒度摘要，利用TTT实现在线梯度更新

**🔧 技术方法**

使用Test-Time Training (TTT)神经记忆层、可学习的跨episode注意力聚合或EMA、PPO训练

**📊 数据集**

在MuJoCo HalfCheetahVel、Meta-World（ML1/ML10/ML45）和稀疏奖励的PointMaze上评估

**📈 对比分析**

与RL^2、MAML、PEARL、VariBAD、TrMRL、ECET、AMAGO等基线比较，METTA在训练预算有限时更快提升，长上下文任务上取得显著优势；在Meta-World未见task-family上成功率提升约40%

**⚠️ 局限性**

仍受限于固定trial结构，对非stationary环境的连续在线适应及跨任务记忆更新策略需要进一步探索

---

## 660. BrickWright: Agentic Creation of Physically Feasible Brick Builds

**arXiv ID:** 2610.31651 | [PDF](https://arxiv.org/pdf/2610.31651v1)

**作者:** Ava Pun `[一作]` (Carnegie Mellon University), Jun-Yan Zhu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 74282 | [OpenAlex ID](https://openalex.org/A5102883508)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 BrickWright 框架，使用预训练 LLM 通过 BrickLayer SDK 生成符合物理约束的乐高建筑。

**💡 创新点**

通过提供专门的装配 API 与验证器，使即使轻量模型也能实现 100% 物理有效性。

**🔧 技术方法**

利用 LLM 代码生成、BrickLayer Python 包（装配 API、碰撞/浮动/连通性验证）以及计划-执行-验证循环。

**📊 数据集**

在 BrickNet-512（512 条文本提示）上进行评测。

**📈 对比分析**

与专门训练的 BrickGPT、BrickNet 以及 Articraft、Codex、Claude Code 等基线对比，BrickWright 在物理有效率、提示一致性和成本上均优于所有基线。

**⚠️ 局限性**

未考虑装配顺序与稳定性分析，且相较于专门训练模型速度更慢、成本更高。

---

## 661. SpectralCanary: Embarrassingly Simple Low-Frequency Traces for T2I Dataset-Use Auditing

**arXiv ID:** 2610.31659 | [PDF](https://arxiv.org/pdf/2610.31659v1)

**作者:** Sy-Tuyen Ho `[一作]` (University of Maryland), Furong Huang `[通讯]` (University of Maryland)

**通讯引用:** 4985 | [OpenAlex ID](https://openalex.org/A5091135797)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于频域低频混合的可检测且低成本的数据集使用审计方法 SpectralCanary，能在图像-文本生成模型中识别是否使用了受保护的数据集。

**💡 创新点**

创新点在于：① 仅在频域低频处注入弱共享信号，保持原图像视觉质量和模型生成质量；② 通过单一确定性 FFT 变换，无需额外生成器或迭代优化；③ 证明低频信号在不同模型、数据集和训练方式下更易迁移和检出。

**🔧 技术方法**

使用离散傅里叶变换（FFT）与低通高斯掩模混合、轻量级二分类器（Verifier）进行训练和检测，实验中对 Stable Diffusion v1.5/3.5、LoRA 以及完整参数微调进行评估。

**📊 数据集**

在五大视觉域数据集上进行实验：Pokemon‑BLIP‑Captions、Sketch‑Scene、ROCOv2、RSICD、FACAD，覆盖不同风格和分辨率。

**📈 对比分析**

与 SIREN、DIAGNOSIS、EnTruth 等现有方法对比，SpectralCanary 在所有模型、数据变换（如数据增强、图像重建、重标注）场景下均实现最高的 1% FPR 下 TPR（平均 91%）和模型级别检出率（平均 95%），且在图像质量（LPIPS ≈0.02、ΔFID≈0）和处理速度（≈0.39 s/图）上优于对手。

**⚠️ 局限性**

局限性包括：仅评估了 LoRA 与完整微调场景，未探讨与大型未保护数据混合的迁移；对手若获知可听频谱、α 与 c 参数，可通过逆向重构或高级去噪攻击降低检测率，且现有攻击需要显著损害图像或模型质量；在多种攻击或更强攻击下的鲁棒性尚未完全证明。

---

## 662. Cost-Aware Abstention for LLM-Based IT Ticket Classification

**arXiv ID:** 2610.31657 | [PDF](https://arxiv.org/pdf/2610.31657v1)

**作者:** Sachin Gupta `[一作]` `[通讯]` (Independent Researcher), Sachin Gupta (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个可复现的成本感知弃权评估框架，比较传统机器学习和提示式大型语言模型在 IT 工单分类中的自动路由成本效益。

**💡 创新点**

将成本感知弃权、校准与多模型一致性信号统一到同一公开数据集，提供冻结阈值和可审计的决策流程。

**🔧 技术方法**

使用逻辑回归、SVM、随机森林、DistilBERT 等传统模型，以及 GPT‑4o‑mini、Claude Haiku、Qwen2.5‑7B、Llama‑3‑8B 等提示式 LLM，结合阈值选择、ECE、AUROC、AURC、等频分箱、等温度校准和投票一致性等技术。

**📊 数据集**

使用公开的 IT Service Ticket Classification Dataset（47,837 条工单，8 类）。

**📈 对比分析**

在校准集上选取阈值后，在测试集上评估；TF‑IDF+LR 在 (0,1,5) 成本下通过 38.1% 降低平均成本，传统模型整体优于提示式 LLM，LLM 在大多数设置下仅路由极低覆盖率（如 19/1000），效果不佳。

**⚠️ 局限性**

结果受限于单一公开数据集、示例性成本假设、提示与预算敏感性、缺乏时间漂移和多企业验证，且 LLM 仅以 3 例提示、无标签定义，未覆盖联合训练或人类错误模型等因素。

---

## 663. Gate3DGen: A Pre-processing Model that Selectively Purifies Real-World Images for 3D Generation

**arXiv ID:** 2610.31649 | [PDF](https://arxiv.org/pdf/2610.31649v1)

**作者:** Xiaolin Li `[一作]` (Tsinghua University), Qi Tian `[通讯]` (Huawei Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

论文提出了一种在冻结的单图像3D生成器前进行选择性输入净化的模型，能够根据图像质量决定是否对图像进行一次性净化，以提升生成的3D几何和外观。

**💡 创新点**

通过结合降质感知路由器和一键式3D定向净化器，模型实现了仅在必要时净化并保留关键视觉线索的选择性净化，并在训练时利用合成3D资产生成配对数据。

**🔧 技术方法**

使用了Qwen2.5‑VL驱动的降质路由器、FireRed‑Image‑Edit的流匹配一步净化器、空间重构与边缘约束损失、以及VAE解码器与现有的3D生成器（TRELLIS.2、Pixal3D）结合。

**📊 数据集**

基于TexVerse 3D资产生成的44k对齐的净化与参考图像以及100k路由标签的数据集，并在真实世界的数百张无标注照片上进行评估。

**📈 对比分析**

与七种通用图像编辑方法对比，净化器在2D图像LPIPS降低39.1%、推理时长2.3s、并在TRELLIS.2上使网格距离降低61.5%、在Pixal3D上提升F1@all 3.8%，在真实图像上VLM分数与人类偏好均优于基线。

**⚠️ 局限性**

需要合成数据来训练，无法处理不在合成资产分布内的复杂现实场景，路由器在真实图像上仍会过度净化，且对多重退化的鲁棒性有限。

---

## 664. OmniRay: Camera-Agnostic Monocular 3D Reconstruction with Composite Ray Fields

**arXiv ID:** 2610.31645 | [PDF](https://arxiv.org/pdf/2610.31645v1)

**作者:** Namitha Guruprasad `[一作]` (Johns Hopkins University), Anand Bhattad `[通讯]` (Johns Hopkins University)

**通讯引用:** 389 | [OpenAlex ID](https://openalex.org/A5033616504)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种名为 OmniRay 的单图像 3D 重建模型，能够在不使用相机内参或相机类型信息的前提下，对多种中心相机（针孔、鱼眼、全景、双曲镜等）生成可缩放的点云。

**💡 创新点**

创新之处在于采用分层射线表示：先用单调径向基底捕捉大尺度径向对称结构，再加入球谐残差和自由形残差，通过连续旋转投影实现对不同相机几何的自适应建模。

**🔧 技术方法**

技术方案包括以 DINOv2 ViT-L/14 为编码器，Ray 模块由径向基底、球谐残差和自由形残差组成，Distance 模块预测径向距离；训练时结合角度射线损失、点云损失和随机镜头数据增强。

**📊 数据集**

训练使用 2D3DS 与 360Loc 两个全景数据集，并通过随机镜头采样覆盖从针孔到极宽角的相机；评估时加入 PanoCity、CO3Dv2 等外部数据集。

**📈 对比分析**

与 UniK3D、Depth Anything 3 等基线对比，OmniRay 在所有相机类型（含未见的 GoPro、双曲镜等）上将平均角度误差降至 0.14°、Chamfer 距离降至 0.018，显著优于传统方法。

**⚠️ 局限性**

局限性在于对与训练分布差异较大的视角（如高空航拍全景）时，射线预测仅覆盖有限的前向区域，导致点云缺失全视球；整体受限于训练视角高度和方向的多样性。

---

## 665. Direct Pass@$k$ Optimization for Thought Pruning: Balancing Quality and Diversity in Test-Time Reasoning

**arXiv ID:** 2610.31667 | [PDF](https://arxiv.org/pdf/2610.31667v1)

**作者:** Mahdi Salmani `[一作]` (University of Southern California), Meisam Razaviyayn `[通讯]` (University of Southern California)

**通讯引用:** 8894 | [OpenAlex ID](https://openalex.org/A5041237539)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于直接优化“至少有一个候选能正确”这一目标的推理模型测试时裁剪框架PKTP。

**💡 创新点**

将质量与多样性平衡整合到单一子集选择目标，使用潜在高斯模型捕捉候选间的相关性，并给出可贪婪近似的子模优化方法。

**🔧 技术方法**

采用潜在高斯分布、二阶近似、子模函数贪婪优化、EMA隐藏状态表示、线性探针等技术。

**📊 数据集**

在包含AIME、HMMT、RIMO‑N、BeyondAIME、OlymMATH、Omni‑MATH、AMO‑Bench等约600道数学/编程题的数据集上进行实验。

**📈 对比分析**

与随机、线性探针、RepExp等基线在8个开放权重推理模型上进行交叉验证，PKTP在中等k（如k=4,8）平均提升约5–6个百分点，并始终优于随机。

**⚠️ 局限性**

对部分模型中仅能获得有限的部分生成特征，质量信息不足，导致在极大k或高温采样下性能不如基线，需进一步改进部分rollout表示。

---

## 666. A Unified Framework for Image-to-3D Part Generation via Variable Granularity

**arXiv ID:** 2610.31671 | [PDF](https://arxiv.org/pdf/2610.31671v1)

**作者:** Qiwen Gu `[一作]` (Tongji University), Junqiao Zhao `[通讯]` (Tongji University)

**通讯引用:** 1477 | [OpenAlex ID](https://openalex.org/A5007977858)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出FlexPart，一种统一的可控图像到3D部件生成框架，支持点、框和掩膜等多粒度几何提示；

**💡 创新点**

通过门控自适应层归一化(Gated AdaLN)将不同粒度提示统一融合，并引入非对称几何引导（可微分最优传输）将稀疏点提示的空间语义与密集提示协同提升；

**🔧 技术方法**

基于Diffusion Transformer (DiT)，使用Gated AdaLN、可微分部分最优传输、视觉桥接与停梯度对齐，结合TripoSG VAE、SAM3D、DINOv2等技术；

**📊 数据集**

训练与评估使用Objaverse、PartVerseXL和PartObjaverse-Tiny等公开数据集；

**📈 对比分析**

与六种SOTA方法（包括TripoSG、OmniPart、PartCrafter等）在Mask、Box、Point等条件下对比，Part级Chamfer Distance、IoU_V、IoU_B等指标表明在Mask条件下比OmniPart提升10.5%，在Whole-mesh上取得Mean CD 0.1418，比OmniPart提升8%，且在弱提示下仍保持竞争性能；

**⚠️ 局限性**

对单视图2D提示下的遮挡处理仍存在困难，缺乏深度信息导致模型对遮挡部件定位不准确，需进一步改进空间推理能力。

---

## 667. Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success

**arXiv ID:** 2610.31670 | [PDF](https://arxiv.org/pdf/2610.31670v1)

**作者:** Yunfei Xie `[一作]` (Rice University), Chen Wei `[通讯]` (Rice University)

**通讯引用:** 21887 | [OpenAlex ID](https://openalex.org/A5100344556)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并评估了一种无训练的离线指标——Sensorimotor Alignment（SA），用于衡量机器人感知表示（图像+文本特征）与运动表示（末端执行器轨迹）之间的邻居重叠程度，并验证该指标与多种机器人模型（VLAs、VLMs、WAMs及真实机器人策略）的下游任务成功率之间的正相关性。

**💡 创新点**

创新点在于：1) 将柏拉图式表示假设（PRH）扩展到机器人领域，将动作轨迹视为与感知同源的投影；2) 设计了基于邻居重叠的SA度量，为机器人模型选择提供低成本离线代理；3) 在多模型、多数据集与真实机器人上证明SA比传统VQA或视频质量指标更能预测任务表现，并能解释之前仅通过闭环实验观察到的现象。

**🔧 技术方法**

技术包括：使用大型预训练视觉‑语言模型的视觉和文本编码器提取感知特征；使用动态时间规整（DTW）计算轨迹相似度；构造感知与运动空间的k最近邻集；计算邻居集合的重叠比例作为SA；以及在不同模型与数据集上进行相关性分析。

**📊 数据集**

使用的数据集有：DROID、BridgeDataV2（含Franka Panda和WidowX机器人数据），Bimanual ALOHA（WAM评估），Simular Env（VLAs）以及由MolmoB0T提供的40项真实机器人任务。

**📈 对比分析**

评估方法：对每个模型在参考数据集上计算SA，然后将SA与其在对应任务中的成功率绘图，计算Pearson相关系数。结果显示：VLAs、VLMs、WAMs以及真实机器人策略的SA均与成功率呈显著正相关（r约0.7或更高），且SA对成功率的预测能力优于VQA分数或视频质量指标（SSIM、PSNR、FID）。

**⚠️ 局限性**

局限性：1）SA的关系是相关性的，尚未证明可在训练过程中直接用以提升性能；2）同一SA值的模型仍可能有显著差异的成功率，SA只能作为低成本的模型筛选工具；3）实验覆盖的模型族、基准与训练策略有限，尚需在更广泛的设置中验证泛化性。

---

