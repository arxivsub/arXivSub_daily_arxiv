# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-27 | 今日论文总数: 354

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. DM$^3$-Nav: Decentralized Multi-Agent Multimodal Multi-Object Semantic Navigation

**arXiv ID:** 2604.22014 | [PDF](https://arxiv.org/pdf/2604.22014v1)

**作者:** Amin Kashiri `[一作]` (Northeastern University), Yasin Yazıcıoğlu `[通讯]` (Northeastern University)

**通讯引用:** 298 | [OpenAlex ID](https://openalex.org/A5041762786)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了DM3-Nav，一套完全去中心化的多模态、多目标语义导航系统；

**💡 创新点**

创新点包括：邻居意图广播+距离加权前沿选择实现无中心化任务分配；提出多对象多机器人SPL评估指标；支持类别、语言、图像三种模态；

**🔧 技术方法**

技术手段为：GOAT语义映射+CLIP/语义匹配、ORB+RANSAC地图融合、距离比前沿选择、离线MILP计算最优任务分配；

**📊 数据集**

使用数据集为HM3Dv0.2与GOAT-Bench，并在真实办公室环境中部署AgileX Scout小型移动机器人；

**📈 对比分析**

与Co-NavGPT、MCoCo-Nav等集中式基线比较，DM3-Nav在HM3Dv0.2上成功率74.6%、SPL38.2%，在GOAT-Bench多目标中四机器人成功率34.5%（单机24.6%），在真实环境中14分钟完成全部目标；

**⚠️ 局限性**

局限性：对通信密度敏感，地图融合在高重叠度时精度下降；任务分配仅近似，SPL略低于集中式基线；未覆盖时间约束、异构团队和复杂任务逻辑；

---

## 2. The Biggest Risk of Embodied AI is Governance Lag

**arXiv ID:** 2604.21938 | [PDF](https://arxiv.org/pdf/2604.21938v1)

**作者:** Shaoshan Liu `[一作]` `[通讯]`, Shaoshan Liu

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

探讨 embodied AI 造成的治理滞后问题，并基于三类滞后（观察、制度、分配）提出合规治理框架

**💡 创新点**

将治理滞后分为观察滞后、制度滞后、分配滞后三类，并针对每类给出相应的治理对策（可见性、层级责任、触发式调整、自动分配）

**🔧 技术方法**

本文未采用具体技术实现，而是通过政策分析、理论推导和案例阐释构建治理模型

**📊 数据集**

未使用特定数据集，主要参考工业机器人部署统计、行业报告与历史案例

**📈 对比分析**

论文不涉及实验或算法性能比较，结论基于理论论证与行业案例分析

**⚠️ 局限性**

局限在于缺乏实证验证与实施案例，需要后续研究检验提议框架的可操作性与效果

---

## 3. Performance Anomaly Detection in Athletics: A Benchmarking System with Visual Analytics

**arXiv ID:** 2604.21953 | [PDF](https://arxiv.org/pdf/2604.21953v1)

**作者:** Blessed Madukoma `[一作]` (Carnegie Mellon University Africa), Prasenjit Mitra `[通讯]` (Carnegie Mellon University Africa)

**通讯引用:** 10238 | [OpenAlex ID](https://openalex.org/A5009542542)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于可视化分析的运动员表现异常检测系统，对1.58 M条成绩、19 712场次、214 914名运动员进行处理，提供多方法异常检测与专家审查界面。

**💡 创新点**

提供可复现的大规模数据管线、统一检测接口、八种检测方法的系统对比、交互式可视化支持专家决策，并在100 m/200 m/400 m跑道上实现跨方法基准。

**🔧 技术方法**

使用 PostgreSQL、DuckDB 与 Redis 存储，Python/Scikit‑learn、Isolation Forest、XGBoost、Bayesian 层级模型、Gaussian Copula 等算法；前端采用 HTMX 构建交互式可视化。

**📊 数据集**

使用 2010–2025 年世界田径官方成绩 1.58 M 条、19 712 场次、214 914 名运动员的数据，结合 Athletics Integrity Unit 的公开禁赛记录 60 名运动员。

**📈 对比分析**

以运动员级别的精确率、召回率与 F1 评估八种方法；在 100 m 赛段，Excess Performance 取得最高 F1 0.016；Isolation Forest 召回率 0.24 但精确率仅 0.003；Bayesian 取得 0.011。200 m 与 400 m 各有不同最佳方法，整体 F1 极低但对稀缺阳性事件仍具筛查价值。

**⚠️ 局限性**

稀缺标注导致指标保守；缺失海拔、部分风速等环境变量；公开禁赛记录滞后多年；模型仅为筛查工具，不能直接处罚；系统在不同运动项目上的适用性仍需进一步验证。

---

## 4. A general optimization solver based on OP-to-MaxSAT reduction

**arXiv ID:** 2604.21961 | [PDF](https://arxiv.org/pdf/2604.21961v1)

**作者:** Yuxin Zhao `[一作]` (South China University of Technology), Zhifeng Hao `[通讯]` (Shantou University)

**通讯引用:** 3131 | [OpenAlex ID](https://openalex.org/A5101432634)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了GORED，一种通过OP-to-MaxSAT自动化归约实现通用优化求解的框架，能够一次性解决多种类型的优化问题。

**💡 创新点**

创新点在于将任何优化问题在多项式时间内自动归约为MaxSAT实例，并利用单一MaxSAT求解器完成求解，从而突破传统方法对问题类型的依赖。

**🔧 技术方法**

核心技术包括：OP-to-MaxSAT归约（变量采用有符号二进制固定点编码；约束与目标转化为CNF）；MaxSAT求解器Pacose24；以及对建模语言的统一解析与简化。

**📊 数据集**

实验使用了136个实例，涵盖11类优化问题，数据来源包括OR‑LIBRARY、TSPLIB、CVRPLIB、Taillard、QAPLIB以及IEEE CEC 2005的基准集。

**📈 对比分析**

与CPLEX、Gurobi、SCIP、GA、EA、PSO等主流求解器对比，GORED在所有实例上都实现了与现有方法相当甚至更优的目标值，且不存在显著统计差异。

**⚠️ 局限性**

局限性包括：不支持多目标和黑盒优化；归约生成的MaxSAT实例可能庞大导致求解时间增长；受有限精度约束，极端数值可能产生误差。

---

## 5. Emergent Technology, Emergent Critique: Students and Teachers Developing Critical AI Literacy through Participatory Design around Generative AI

**arXiv ID:** 2604.21995 | [PDF](https://arxiv.org/pdf/2604.21995v1)

**作者:** Santiago Ojeda-Ramirez `[一作]` (University of California, Irvine), Kylie Peppler `[通讯]` (University of California, Irvine)

**通讯引用:** 4870 | [OpenAlex ID](https://openalex.org/A5059999110)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在加州一所拉丁裔学生占绝大多数的高中，开展为期五周的参与式设计项目，让三名11年级学生和三名教师共同制定并设计如何在课堂中使用和教授生成式AI工具的课程单元。

**💡 创新点**

通过让学生承担真正的设计权，研究揭示了三种关键AI素养实践：共同挑战AI假设、教师与学生之间的双向学习、以及以文化知识和创造性实践为基础的AI批判；同时提供了将学生主体地位嵌入参与式设计的具体策略。

**🔧 技术方法**

采用的技术包括生成式AI工具（ChatGPT、Claude、Gemini）用于课程设计与评估；使用视频记录、共创文档、访谈记录以及研究者笔记进行数据收集和编码。

**📊 数据集**

数据集为项目产生的四类原始材料：（1）所有五次工作坊的音视频记录；（2）学生教师共创的课程单元和设计工作表；（3）前后访谈文本；（4）研究者会后笔记。

**📈 对比分析**

本研究并未进行算法或模型性能比较，而是以案例研究方法描述参与式设计过程中的知识生成与实践变迁；通过Veldhuis等人提出的四维框架进行主题编码，说明三种实践在设计过程中如何形成。

**⚠️ 局限性**

局限性包括样本规模小且高度情境化，仅涉及三名学生和三名教师；缺乏对实践可持续性与在不同学校环境中的迁移性进行评估；并未探讨长期对学生AI素养与教学成效的影响。

---

## 6. Lessons from External Review of DeepMind's Scheming Inability Safety Case

**arXiv ID:** 2604.21964 | [PDF](https://arxiv.org/pdf/2604.21964v1)

**作者:** Stephen Barrett `[一作]` (Arcadia Impact AI Governance Taskforce), Henry Papadatos `[通讯]` (SaferAI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对Google DeepMind公开的“Scheming Inability”安全案例进行外部评审，采用Assurance 2.0方法发现关键缺陷并提出改进建议。

**💡 创新点**

首次系统性将Assurance 2.0与CAEs结合用于前沿AI安全案例的外部审计，并提出一套可操作的评审流程与实践建议。

**🔧 技术方法**

使用Assurance 2.0框架、Claims-Argument-Evidence (CAE) 方法、风险路径分析、确认理论评估以及辩证式对照案例构建技术。

**📊 数据集**

基于DeepMind公开的评估任务与其引用的文献（如CrowdStrike、Knight Capital案例），未使用额外公开数据集。

**📈 对比分析**

与传统安全工程标准进行对比，重点评估安全论证的逻辑完整性、CAEs合规性及证据力度。结果显示多处证据薄弱、可信度低，未提供定量性能指标。

**⚠️ 局限性**

评审仅基于公开信息；与DeepMind缺乏紧密互动；缺少完整的系统模型与风险路径；评估主要为定性，未覆盖所有风险类别。

---

## 7. Looking Into the Past: Eye Movements Characterize Elements of Autobiographical Recall in Interviews with Holocaust Survivors

**arXiv ID:** 2604.22016 | [PDF](https://arxiv.org/pdf/2604.22016v1)

**作者:** Emily Zhou `[一作]` (University of Southern California), Shrikanth Narayanan `[通讯]` (University of Southern California)

**通讯引用:** 31255 | [OpenAlex ID](https://openalex.org/A5010028928)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在半自然访谈中犹太大屠杀幸存者的眼动与自传性记忆的关系。

**💡 创新点**

首次在大规模、情感强烈的现实访谈中量化眼动与记忆类型、时间维度的关联，并使用深度模型预测时间上下文。

**🔧 技术方法**

使用OpenFace提取眼动特征，基于S4和时序CNN进行时间上下文预测。

**📊 数据集**

利用美国犹太大屠杀幸存者访谈数据集（Voices）共806名受访者，239k句子。

**📈 对比分析**

通过Welch t检验、GAMM和AUC评估，模型对前期记忆的预测AUC可达0.70以上，优于随机基线。

**⚠️ 局限性**

受限于视频质量、受访者个体差异、LLM标签噪声和采访进程混杂，导致预测偏差和跨主题差异。

---

## 8. Not Another EHR: Reimagining Physician Information Needs with Generative AI Technology

**arXiv ID:** 2604.21933 | [PDF](https://arxiv.org/pdf/2604.21933v1)

**作者:** Ruican Zhong `[一作]` (University of Washington), Amanda K. Hall `[通讯]` (Microsoft Research)

**通讯引用:** 1716 | [OpenAlex ID](https://openalex.org/A5070994189)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开展访谈研究，分析医生信息需求并设计面向医生的生成式 AI UI

**💡 创新点**

提出以动态交互和医生认知模型为核心的 AI 支持体系，并探讨 AI 在医生工作流中的多重角色

**🔧 技术方法**

采用大型语言模型（LLM）与生成式 UI 技术，用于数据摘要、可视化与交互生成

**📊 数据集**

访谈数据（9名内部医生技术人员）以及现有 EHR 系统数据结构作为参考

**📈 对比分析**

尚未开展系统实现与性能评估，本文仅提出设计思路；未来计划与现有 EHR 进行对比

**⚠️ 局限性**

样本量有限、缺乏实证验证、对 AI 伦理与安全考量不充分

---

## 9. Shared Lexical Task Representations Explain Behavioral Variability In LLMs

**arXiv ID:** 2604.22027 | [PDF](https://arxiv.org/pdf/2604.22027v1)

**作者:** Zhuonan Yang `[一作]` (Brown University), Ellie Pavlick `[通讯]` (Brown University)

**通讯引用:** 6853 | [OpenAlex ID](https://openalex.org/A5053850863)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型对提示方式的敏感性，发现并定义了“词汇任务头”，证明其跨不同提示风格共享，并能解释行为差异。

**💡 创新点**

创新点在于首次将注意力头输出投影到词汇空间，识别出可解释的词汇任务表示，并展示其在不同提示下的功能等价性与对性能的显著影响。

**🔧 技术方法**

采用Logit Lens将注意力头输出映射到词汇空间、激活补丁、因果实验等技术，结合注意力头分析揭示词汇任务头与检索头的交互机制。

**📊 数据集**

实验基于13种自回归Transformer模型（主要是Llama‑3.1‑8B‑Instruct），评估了17个简单知识检索任务、3个组合任务和1个自由形式代码生成任务，使用的公开数据集包括HumanEval‑X等。

**📈 对比分析**

通过激活补丁和功能等价实验，词汇任务头可恢复10%–90%的任务性能，并与示例数正相关，表明其能解释多示例提示的有效性；相较于传统提示，模型在共享任务表示上表现更稳健。

**⚠️ 局限性**

局限性包括仅适用于可用词汇描述的任务，未涵盖抽象任务；词汇任务头的识别依赖经验性阈值，可能遗漏非词汇形式的任务表示；且未完全解释所有提示敏感性现象。

---

## 10. Multi-Task Optimization over Networks of Tasks

**arXiv ID:** 2604.21991 | [PDF](https://arxiv.org/pdf/2604.21991v1)

**作者:** Julian Hatzky `[一作]` (Vrije Universiteit Amsterdam), Anil Yaman `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 237 | [OpenAlex ID](https://openalex.org/A5047642570)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MONET 多任务优化算法，通过将任务空间建模为显式图结构，实现任务邻接关系并支持个体学习和社会学习两种变异方式；

**💡 创新点**

创新点在于用显式任务图替代传统平面档案，利用任务拓扑邻域实现局部知识迁移，并将社会学习（邻域交叉）与个体学习（突变）相结合，能够在数千任务规模下保持可扩展性；

**🔧 技术方法**

使用的技术包括任务空间的最近邻 k‑NN 图构建、SBX 交叉、突变、梯度提升树+SHAP 进行超参数重要性分析、SPOT 自适应调参、以及 Mann‑Whitney U 统计检验等；

**📊 数据集**

实验数据集为四个仿真任务集：archery、arm、cartpole（各 5,000 个任务）和 hexapod（2,000 个任务）；

**📈 对比分析**

比较方法为在相同任务集和 10^6 评估预算下，对 MONET、PT‑ME 与 MT‑ME 进行最终平均适应度和 AUC 对比；MONET_default 在三域表现最好，MONET_tuned 在所有域均优于基线；

**⚠️ 局限性**

局限性包括：假设任务相似性仅基于参数距离可能不成立，邻域大小和学习比例需手工调节，缺乏在线自适应控制，对任务相似度不敏感的领域表现有限。

---

## 11. Routine Computing: A Systematic Review of Sensing Daily Life Dimensions Towards Human-Centered Goals

**arXiv ID:** 2604.21934 | [PDF](https://arxiv.org/pdf/2604.21934v1)

**作者:** Borislav Pavlov `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**通讯引用:** 5593 | [OpenAlex ID](https://openalex.org/A5057896400)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述并整合了截至2025年8月共203篇关于日常生活中常规计算的研究，提出了以时间节律、交互情境、认知因素和适应性为核心的四维维度框架，并对其目标与挑战进行归纳；

**💡 创新点**

创新点在于首次将常规计算定位为独立研究领域，构建了涵盖时间粒度、行为交互、认知心理和变异适应四大维度的概念框架，并将其与四大应用目标（老年与残障护理、健康习惯促进、自我反思与群体洞察）对应，形成系统化的知识体系；

**🔧 技术方法**

采用系统综述方法（PRISMA 2020流程）、文本编码、主题归纳与量化分析，结合多维度维度映射、时间粒度分类及情境认知建模等技术；

**📊 数据集**

主要数据来源为ACM Digital Library与IEEE Xplore共2631篇检索结果，通过筛选得到203篇符合条件的原始研究论文；

**📈 对比分析**

相较于以往单一维度或小样本的综述，本研究通过大样本量、严格的筛选和多维度编码，呈现了时间粒度分布、情境与认知关系的统计分布，并提供了不同研究主题的数量分布与趋势图；

**⚠️ 局限性**

局限在于数据库范围仅限于ACM和IEEE会议/期刊，未涵盖医学、临床或跨学科文献，且缺乏对实际系统部署与实证评估的深入分析；

---

## 12. Feedback Over Form: Why Execution Feedback Matters More Than Pipeline Topology in 1-3B Code Generation

**arXiv ID:** 2604.21950 | [PDF](https://arxiv.org/pdf/2604.21950v1)

**作者:** Charles Junichi McAndrews `[一作]` `[通讯]`, Charles Junichi McAndrews

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在 1‑3B 小模型代码生成任务中，利用执行反馈构建多模型流水线以提升代码正确率。

**💡 创新点**

创新点在于证明执行反馈是小模型组合获益的关键机制，并对错误类型、早停、生成器/精炼器角色进行细粒度分析，展示单一改进可获 4σ 以上提升。

**🔧 技术方法**

使用生成器‑执行器‑（可选分析器）‑精炼器流水线，并采用 NEAT 启发的进化搜索探索拓扑；模型采用 Gemma3、qwen2.5、llama3.2 等 1‑3B LLM，在本地 Ollama 推理。

**📊 数据集**

评估数据集为 HumanEval（164 题）和 MBPP（427 题）。

**📈 对比分析**

与单模型、同模型自精炼、跨模型精炼以及专门化 coder 进行对比，单模型自精炼提升 >4σ；跨模型与单模型差异不显著；coder 自精炼仍略有提升；最优 NEAT 结构仅比手工基线少 1σ。

**⚠️ 局限性**

局限性包括：仅测试 1‑3B 模型、单一 Python 语言、执行反馈可用性受限、进化搜索空间有限、评估噪声导致单次评估偏高、未覆盖更大规模或更难的基准。

---

## 13. Soft Anisotropic Diagrams for Differentiable Image Representation

**arXiv ID:** 2604.21984 | [PDF](https://arxiv.org/pdf/2604.21984v1)

**作者:** Laki Iinbor `[一作]` (Independent Researcher), Wojciech Matusik `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 22579 | [OpenAlex ID](https://openalex.org/A5018010391)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于可微分软化的各向异性加权Voronoi图（SAD）的显式图像表示方法，利用可学习的温度控制像素归属，达到高质量、低编码成本的图像压缩与渲染；

**💡 创新点**

创新点在于：① 用可学习的各向异性度量和温度参数构造软化的Apollonius分区，使像素依赖固定数量的邻域原子；② 设计了基于跳水传播的 Top‑K 维护算法，保证 GPU 上 O(P·K) 的常数查询成本；③ 通过自适应稠密/剪枝控制表示预算，实现高效编码与快速收敛；

**🔧 技术方法**

技术主要包括：可微分的各向异性加权距离评分、温度调节的 softmax 混合、跳水传播 (JFA) 与随机注入的 Top‑K 更新、基于哈希表的梯度累积、Adam 优化、稠密/剪枝自适应控制；

**📊 数据集**

在 Image‑GS、Kodak、DIV2K、CLIC 等标准图像压缩数据集上进行评测；

**📈 对比分析**

与 Image‑GS、Instant‑NGP 等最先进方法对比，SAD 在相同比特率下 PSNR 提升约 1.5‑3 dB，SSIM 与 LPIPS 均显著优于基线；编码时间从 28 s 降至 2.2 s，训练速度提升 4‑19 倍，渲染时间更短；

**⚠️ 局限性**

局限性包括：Top‑K 列表需要频繁更新，低预算或急剧参数变动时可能质量下降；极细纹理或微结构仍需较大原子数；实现高度 GPU 依赖，跨平台性能可变；

---

## 14. Conditional anomaly detection using soft harmonic functions: An application to clinical alerting

**arXiv ID:** 2604.21956 | [PDF](https://arxiv.org/pdf/2604.21956v1)

**作者:** Michal Valko `[一作]` (University of Pittsburgh), Milos Hauskrecht `[通讯]` (University of Pittsburgh)

**通讯引用:** 4924 | [OpenAlex ID](https://openalex.org/A5012461386)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于软谐波函数的条件异常检测方法 SoftHAD，用于在电子健康记录中发现异常的标记（如实验室检验或药物下单）。

**💡 创新点**

创新点包括：①使用软谐波解的绝对值作为标签置信度；②通过图拉普拉斯正则化降低孤立点和边界点的置信度，避免误检；③利用图量化实现可扩展的在线算法。

**🔧 技术方法**

使用技术：无参数标签传播、软约束的无约束正则化、图拉普拉斯正则化、k‑NN 图构建、线性系统求解、图量化。

**📊 数据集**

数据集：真实电子健康记录（4486名患者，51,492实例，749二元标签），重点评估20种最常见的实验室检验和药物下单。

**📈 对比分析**

与1类 SVM、QDA、RBF SVM、加权 k‑NN 等基线比较，使用 ROC 曲线下的 AUC 评估。SoftHAD 在多任务设置下表现优于 SVM，且对图大小和正则化参数不敏感，AUC 通常比基线高。

**⚠️ 局限性**

局限性：目前实验仅在离线数据上进行，尚未实现在线实时检测；对图量化参数（如 γ_g、σ 等）依赖较强，需人工调优；多任务标准化采用极值线性缩放，可能不够稳健。

---

## 15. Polynomial Lower Bounds for Arithmetic Circuits over Non-Commutative Rings

**arXiv ID:** 2604.22006 | [PDF](https://arxiv.org/pdf/2604.22006v1)

**作者:** Ran Raz `[一作]` (Princeton University), Ran Raz `[通讯]` (Princeton University)

**通讯引用:** 7412 | [OpenAlex ID](https://openalex.org/A5041545934)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文通过改造非交换算术电路并利用矩阵秩分析，给出了非交换多项式计算电路的下界，进一步证明了任何在某些非交换环上实现该多项式的算术电路也必须具备相同的下界。

**💡 创新点**

创新点包括：①首次证明在任何域上非交换算术电路计算显式多项式所需乘法门数至少为Ω(d√n)（当d=n时为Ω(n^{1.5})）；②建立了非交换域上电路下界与在非交换环上电路下界之间的直接对应关系；③在电路改造过程中提出了“分裂节点”“消除标量乘法”等新技巧。

**🔧 技术方法**

主要技术是：1）将电路改造为无标量乘法、交替门序的标准化形态；2）利用非交换多项式的矩阵M_f^{a,b}的秩上界进行分析；3）构造后向路径，分为规则1、规则2、乘法门三类，并结合子加性和张量积性质估计秩下降；4）通过集合S的大小推导出Ω(d√n)个乘法门的下界；5）利用先前的代数引理将非交换域下界转化为非交换环下界。

**📊 数据集**

本文为理论论文，无实验数据集，所有结果均为理论证明。

**📈 对比分析**

与以往的最优下界Ω(n log d)（Strassen、Baur–Strassen）及Ω(d log n)（Nisan）相比，本文将下界提升至Ω(d√n)，在d=n时达到Ω(n^{1.5})，显著超过此前的线性或对数级下界，展示了更强的不可约性。

**⚠️ 局限性**

局限性包括：①仅对满足M_f^{d/2,d/2}满秩的多项式适用；②下界仍为多项式级别，距离期望的指数级下界（如Nisan的指数下界）仍有差距；③只关注乘法门数量，未考虑加法门或整体大小；④在环的选择上受限于自由代数⟨Z⟩，不一定适用于所有非交换环。

---

## 16. L-System Genetic Encoding for Scalable Neural Network Evolution: A Comparison with Direct Matrix Encoding

**arXiv ID:** 2604.22000 | [PDF](https://arxiv.org/pdf/2604.22000v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 17. Focus Session: Hardware and Software Techniques for Accelerating Multimodal Foundation Models

**arXiv ID:** 2604.21952 | [PDF](https://arxiv.org/pdf/2604.21952v1)

**作者:** Muhammad Shafique `[一作]` (New York University Abu Dhabi), Minghao Shao `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 25744 | [OpenAlex ID](https://openalex.org/A5107945696)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种多层次、硬件软件协同的加速方法，涵盖模型压缩、操作优化、数据流重排、专用Transformer加速器与LLM辅助RTL生成，并在医疗与机器人案例中验证。

**💡 创新点**

创新点在于将硬件与软件优化串联成完整流水线，推出整数化SwiftTron加速器、LLM驱动的硬件设计闭环、以及面向Spiking-MFM的分层混合精度量化方案。

**🔧 技术方法**

使用混合精度量化、结构化剪枝、推理加速技术（KV缓存、speculative decoding、模型级联）、图层级数据流优化、专用整数Transformer加速器，以及LLM辅助RTL代码生成。

**📊 数据集**

主要数据集包括WikiText-2、医学VQA集合（VQA-RAD、SLAKE、PathVQA）、COCO、LVIS、ODinW及公开文本-图像对（600k PMC-15M）等。

**📈 对比分析**

与GPU基准相比，SwiftTron实现3.5×+速度提升、低精度量化（4/8-bit）保持或略优于全精度准确率，同时显著压缩内存占用（1/4~1/9），在医疗VQA上4-bit模型仍能保持临床可用性能，机器人OVD压缩后维持高F1分数。

**⚠️ 局限性**

局限在于不同任务对精度与延迟的平衡难以统一，LLM驱动硬件生成的安全与可靠性仍需进一步评估，Spiking-MFM在多模态对齐与能耗方面的实验证据尚不足。

---

## 18. Comparative Analysis of Human vs. AI-powered Support in VRChat Communities on Discord: User Engagement, Response Dynamics and Interaction Patterns

**arXiv ID:** 2604.21963 | [PDF](https://arxiv.org/pdf/2604.21963v1)

**作者:** He Zhang `[一作]` (Pennsylvania State University), Jie Cai `[通讯]` (Tsinghua University)

**通讯引用:** 2073 | [OpenAlex ID](https://openalex.org/A5101943689)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对VRChat Discord社区中两条支持渠道（人类支持与AI支持）进行定量与定性分析，探讨了两者在用户互动模式、参与度、情感倾向和知识共享等方面的差异。

**💡 创新点**

创新点包括：① 将时间序列、文本相似度、情感分析、主题建模和网络分析相结合的全流程多方法框架；② 对人类与AI支持的对话细节进行语料编码，揭示人类支持的多轮协作与AI支持的结构化单轮回复的区别；③ 提出“社区‑AI互换”与“学习转介”设计思路，探索人类与AI支持的协同优化。

**🔧 技术方法**

技术方法包括：数据爬取与预处理、基于Sentence‑BERT的语义相似度计算、BERT‑multilingual‑sentiment情感分类、LDA主题建模、Girvan‑Newman社区检测与中心性指标、定性开放式编码与常量比较法。

**📊 数据集**

数据集为 81,784 条“user‑support”记录（10% 问题，90% 回复）与 28,261 条“AI‑support”记录（约10% 问题），收集时间为 2023‑08‑15 至 2024‑06‑13，涵盖所有公开可见的帖子与回复。

**📈 对比分析**

对比方法包括：① 问题/答案日均计数与 7‑日移动平均；② 回复数直方图与箱线图；③ 主题多样性（主题数量与一致性）；④ 回复质量（语义相似度）与社区反应（表情数）的关系；⑤ 累计情感曲线。结果显示：人类支持拥有更高的参与度、更多主题、多轮情感波动与较高的回复数量；AI支持回复更短、情感更中性、回复质量虽高但获得的社区反馈有限。

**⚠️ 局限性**

局限性包括：① 仅针对单一Discord社区（VRChat），缺乏跨社区泛化；② 研究为静态快照，无法捕捉AI和社区动态的长期演变；③ 对AI机器人学习机制及知识源缺乏内部数据；④ 情感模型对讽刺、幽默等非正式语言处理不佳；⑤ AI机器人技术细节未公开，限制了对其内部实现的评估。

---

## 19. When Quotes Crumble: Detecting Transient Mechanical Liquidity Erosion in Limit Order Books

**arXiv ID:** 2604.21993 | [PDF](https://arxiv.org/pdf/2604.21993v1)

**作者:** Haohan Xu `[一作]` (Stony Brook University), David Rosenberg `[通讯]` (Bloomberg)

**通讯引用:** 575 | [OpenAlex ID](https://openalex.org/A5026405650)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在 ABIDES 仿真环境下构建了可观测的瞬时流动性消融（crumbling quotes）检测与连续概率标注框架。

**💡 创新点**

创新点在于使用机制约束的检测管道与神经网络连续标注，通过 agent‑level 真实状态实现可审计的 ground truth。

**🔧 技术方法**

利用 ABIDES agent‑based simulator、规则式硬过滤、深度多层感知机（MLP）与 KL 校准的神经概率模型。

**📊 数据集**

数据集基于 ABIDES 模拟的五个交易日，包含噪声、价值、动量和市场做市商代理；也评估了不同市场状态（基线、牛市、熊市、高波动）。

**📈 对比分析**

与硬规则、逻辑回归及 MLP 对比，AUC 在基线下从 0.67 提升到 0.91，跨市况保持稳健。

**⚠️ 局限性**

局限在于仅在仿真环境下验证，真实市场中机制可观测性缺失且需要进一步校准门限。

---

## 20. Documentless Assessments Using Nominal Group Interviews

**arXiv ID:** 2604.22003 | [PDF](https://arxiv.org/pdf/2604.22003v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 21. Can SOC Operators Explain their Decisions while Triaging Alarms? A Real-World Study

**arXiv ID:** 2604.22001 | [PDF](https://arxiv.org/pdf/2604.22001v1)

**作者:** Jessica Moosmann `[一作]` (University of Liechtenstein), Giovanni Apruzzese `[通讯]` (Reykjavik University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过系统文献综述和实地研究评估SOC运营人员在报警优先级判断中的解释能力

**💡 创新点**

首次系统量化SOC分析师对报警决策的解释质量，并指出解释不足的现状

**🔧 技术方法**

采用定量分析与访谈相结合的实地实验方法，调查12名SOC运营者的决策与解释

**📊 数据集**

真实SOC产生的报警记录，来自合作SOC的生产环境

**📈 对比分析**

通过正确率与解释匹配率对比，发现80%正确率但仅39%解释与根本原因一致

**⚠️ 局限性**

样本规模有限，仅覆盖单一SOC，未考虑不同规模与工具的差异

---

## 22. Mochi: Aligning Pre-training and Inference for Efficient Graph Foundation Models via Meta-Learning

**arXiv ID:** 2604.22031 | [PDF](https://arxiv.org/pdf/2604.22031v1)

**作者:** João Mattos `[一作]` (Rice University), Arlei Silva `[通讯]` (Rice University)

**通讯引用:** 706 | [OpenAlex ID](https://openalex.org/A5057941268)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于元学习的图基础模型(Mochi)，通过在少量样本演练中预训练，使训练目标与推理任务对齐，避免后期统一步骤。

**💡 创新点**

创新点在于用少量样本的元学习框架替代传统重建目标，直接在预训练阶段学习与下游任务一致的表示，从而提升任务统一性和训练效率。

**🔧 技术方法**

主要技术包括元学习（few-shot episode 训练）、图神经网络架构，以及对比学习或重建式预训练的改进。

**📊 数据集**

在25个真实世界图数据集上评估，涵盖节点分类、链路预测和图分类三大任务类别。

**📈 对比分析**

与现有图基础模型对比，Mochi 在大多数任务上获得竞争或更优的性能，同时比最强基线模型训练时间缩短 8–27 倍。

**⚠️ 局限性**

局限性包括对少样本设置的依赖，若下游任务与预训练演练差异较大，性能可能受限；此外，模型在极大规模图上的扩展性和对超参数的敏感度仍需进一步研究。

---

## 23. Causality and Semantic Separation

**arXiv ID:** 2604.22041 | [PDF](https://arxiv.org/pdf/2604.22041v1)

**作者:** Anna Zhang `[一作]` (Massachusetts Institute of Technology), Adam Chlipala `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 3206 | [OpenAlex ID](https://openalex.org/A5078100439)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过定义一种新的语义分离概念，对因果图中的 d‑分离进行形式化，并证明两者等价，从而为实验设计的形式验证提供了语义基础。

**💡 创新点**

创新点在于将信息流安全中的非干预（noninterference）思路引入因果图，提出一种基于函数赋值的确定性语义分离，并用 Rocq 机械化证明了其与传统图论 d‑分离的等价性。

**🔧 技术方法**

采用了结构化因果模型（SCM）与函数式语义的组合，利用递归定义的图函数、未观察项赋值序列以及代数化的证明技术，并在 Rocq 上实现了整个证明。

**📊 数据集**

由于研究聚焦于理论证明，本文未使用任何实际数据集；所有结果均在抽象的因果图上给出并通过形式化证明验证。

**📈 对比分析**

本文主要通过形式化证明来检验方法的正确性，并未进行实验或性能评测；相较于传统的仅图论判定，新增的语义视角提供了更直观的解释，但实现上仅在 Rocq 证明助手中展示。

**⚠️ 局限性**

局限性包括：仍未将概率与统计假设纳入框架；对连续域和测量误差的处理不够；以及在实际实验设计中对未观察项控制的假设过于理想化。

---

## 24. LayerBoost: Layer-Aware Attention Reduction for Efficient LLMs

**arXiv ID:** 2604.22050 | [PDF](https://arxiv.org/pdf/2604.22050v1)

**作者:** Mohamed Ali Souibgui `[一作]`, Igor Peric `[通讯]` (Openchip & Softwares Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LayerBoost，利用层级敏感性分析对 Transformer 的注意力机制进行有选择性改造，结合软max、线性滑动窗口和无注意力层，并通过轻量级蒸馏恢复性能

**💡 创新点**

创新点在于：①按层敏感性分配不同注意力策略，②使用少量额外训练标记进行蒸馏“治疗”以弥补性能损失，③在保持模型质量的前提下显著降低推理延迟与吞吐成本

**🔧 技术方法**

技术手段包括：层级敏感性分析、软max注意力、线性滑动窗口注意力、无注意力层、蒸馏学习（只需 10M 额外 token）

**📊 数据集**

使用多种标准自然语言处理基准数据集（如 GLUE、SuperGLUE、SQuAD 等）进行评估

**📈 对比分析**

与现有线性/混合注意力方法比较，LayerBoost 在多项基准上匹配或接近原始模型性能，同时在高并发场景下推理延迟降低至 68%，吞吐量提升显著

**⚠️ 局限性**

局限性在于：①需要对预训练模型进行敏感性分析和额外蒸馏训练；②方法的有效性和参数化程度可能随模型规模和任务类型而变化；③在极大规模模型或特殊硬件上尚未充分验证

---

## 25. Community-Based AI Learning: Redistributing Artificial Intelligence's Epistemic Authority in Education

**arXiv ID:** 2604.21986 | [PDF](https://arxiv.org/pdf/2604.21986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 26. MambaCSP: Hybrid-Attention State Space Models for Hardware-Efficient Channel State Prediction

**arXiv ID:** 2604.21957 | [PDF](https://arxiv.org/pdf/2604.21957v1)

**作者:** Aladin Djuhera `[一作]` (Technical University Munich), Holger Boche `[通讯]` (Technical University Munich)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种混合注意力状态空间模型MambaCSP，用于硬件高效的频道状态预测

**💡 创新点**

将Mamba状态空间模型与稀疏补丁混合注意力相结合，既保留线性时间复杂度，又补偿纯SSM的局部依赖问题

**🔧 技术方法**

使用Mamba模型、轻量级补丁混合注意力、统一的CSI预处理与token化流程、LLM预训练骨干冻结策略

**📊 数据集**

利用QuaDRiGa生成的3GPP TR 38.901 UMa NLOS信道数据，包含TDD与FDD两种工作模式

**📈 对比分析**

与传统CNN、RNN、LSTM以及与GPT‑2等LLM模型对比；在NMSE上MambaCSP优于LLM约12‑17%，吞吐量提升3×、显存降低2.6×、推理延迟降低2.9×

**⚠️ 局限性**

对补丁混合注意力的频率、头数等参数需要权衡；模型仍受限于输入序列长度对性能的边际收益，且尚未探索量化/剪枝等进一步压缩手段

---

## 27. When Altruism Meets Autonomy: Managing Bottleneck Congestion with Strategic Autonomous Vehicles

**arXiv ID:** 2604.21941 | [PDF](https://arxiv.org/pdf/2604.21941v1)

**作者:** Kexin Wang `[一作]` (University of Southern California), Ruolin Li `[通讯]` (University of Southern California)

**通讯引用:** 446 | [OpenAlex ID](https://openalex.org/A5101402766)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种统一的平衡框架，用来描述混合自动化车辆（AV）与人类驱动车辆（HDV）在高速公路并道段中的车道选择行为，并通过该框架预测并优化系统级延误。

**💡 创新点**

创新点在于：①将 Wardrop 自利行为与 Stackelberg 领导-跟随结构结合，形成 Stackelberg‑Wardrop 双层模型；②通过社会价值取向（SVO）引入异质驾驶偏好，构建更真实的多类车辆决策模型；③揭示了 AV 渗透率对系统性能的结构性影响——性能呈非递增、分段常数的“阶梯”式变化，并给出了阈值条件。

**🔧 技术方法**

使用了 Wardrop 交通平衡理论、Stackelberg 游戏理论、社会价值取向模型以及非线性补充性问题（NCP）与单层化的 MPEC 求解技术。

**📊 数据集**

采用 SUMO 微观仿真生成大量交通流数据（共 415 条校准样本，320 条验证样本），其中总流量为 1400 辆/小时，分别分配给进入、退出、通过等车辆类别。

**📈 对比分析**

与纯自利 HDV 平衡以及社会最优（最小总延误）进行对比。数值实验显示：随着 AV 份额上升，系统延误保持不变直至特定阈值后才下降，随后趋于社会最优；而单一性能指标（如平均延误）并未呈线性提升。

**⚠️ 局限性**

局限性包括：模型仅为宏观静态分析，未考虑时间变异、随机需求、学习与适应；适用于单段并道，尚未推广到多路段网络；在极端情形下（极高渗透率或异常车辆行为）理论假设可能失效。

---

## 28. Source-Modality Monitoring in Vision-Language Models

**arXiv ID:** 2604.22038 | [PDF](https://arxiv.org/pdf/2604.22038v1)

**作者:** Etha Tianze Hua `[一作]` (Brown University), Ellie Pavlick `[通讯]` (Brown University)

**通讯引用:** 6853 | [OpenAlex ID](https://openalex.org/A5053850863)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了视觉语言模型在多模态输入中如何绑定词与对应的输入源（图像或文本），评估其源模态监控能力。

**💡 创新点**

首次系统地将句法（符号标记）与语义（分布信息）信号结合，揭示二者在绑定过程中的相互作用与取舍。

**🔧 技术方法**

使用目标模态检索任务、结构扰动（删除/交换标记）、冻结‑移除干预以及学习向量优化等技术，并用LLM评判输出的真实性。

**📊 数据集**

主要数据集为 Flickr30k 与 MSCOCO，构造了图像与不匹配字幕的示例来测试绑定能力。

**📈 对比分析**

对 11 种开源 VLM 在目标模态检索任务中的有效率和选择性进行比较；大模型及带专用标记的模型表现最佳，符号标记缺失时选择性显著下降但仍优于随机。

**⚠️ 局限性**

局限性包括：仅在静态推理阶段评估，未覆盖动态代理交互；实验数据来源有限；对不同模型的干预效果差异未完全解释。

---

## 29. EgoMAGIC- An Egocentric Video Field Medicine Dataset for Training Perception Algorithms

**arXiv ID:** 2604.22036 | [PDF](https://arxiv.org/pdf/2604.22036v1)

**作者:** Brian VanVoorst `[一作]` (RTX BBN Technologies), Ehsan Elhamifar `[通讯]` (Northeastern University)

**通讯引用:** 6792 | [OpenAlex ID](https://openalex.org/A5088989912)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文发布了一个包含 50 个战术创伤处理任务的 egocentric 视角视频数据集，并基于 8 个任务构建了动作检测基准，报告了三种模型的性能；

**💡 创新点**

创新点在于提供了规模大、标注细粒度且涵盖多模态（视频、音频、深度、手-物体交互）的医疗动作数据集，并针对短暂、重叠及可选步骤设计了动作检测基准；

**🔧 技术方法**

采用了 RNN（GRU+Omnivore+CLIP+YOLOv8）、Video Swin‑Transformer（Swin‑T）以及改进的 Temporal Action Segmentation（causal TCN + I3D）三种深度学习技术；

**📊 数据集**

使用了自建的 MAGI  数据集（约 786 GB、1.95 M 物体标签、17 k 步骤标签、39 k 手-物体交互）以及公开的 zenodo 资源；

**📈 对比分析**

通过比较平均 mAP，RNN 与 Swin‑T 在 8 任务上分别达 0.526 与 0.513，明显优于传统的 TAS（0.281），但在部分任务如 M2 的 recall 较低；

**⚠️ 局限性**

主要局限包括：仅对 8 个任务做了基准评估，未充分利用立体/音频模态；步骤极短、并行且可选导致检测难度；模型在不同硬件/专业水平下的泛化仍待验证。

---

## 30. Quantifying Interface Procedure Coupling Risks in Digital Nuclear Control Rooms: An Event Based Human Reliability Assessment

**arXiv ID:** 2604.21932 | [PDF](https://arxiv.org/pdf/2604.21932v1)

**作者:** Xingyu Xiao `[一作]` (Tsinghua University), Haitao Wang `[通讯]` (Tsinghua University)

**通讯引用:** 5562 | [OpenAlex ID](https://openalex.org/A5006641842)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对数字化核电站主控制室接口-程序耦合风险进行事件驱动的量化评估，构建了三维标签体系和四因子UI模型，并通过真实事件数据与实验模拟验证其有效性。

**💡 创新点**

提出并量化“接口-程序耦合陷阱”概念，构建可复用的三维标签方案，利用机器学习（随机森林+SHAP）揭示语义不匹配与布局陷阱对程序偏差的放大效应。

**🔧 技术方法**

事件标注框架、四因子UI分解（布局、语义、匹配、标识）、二元逻辑回归、随机森林、SHAP解释以及模拟实验。

**📊 数据集**

2021-2025年中国核电厂A/B级正式事件报告共59条（程序偏差、接口缺陷、耦合事件），以及518次模拟参数验证任务实验数据。

**📈 对比分析**

与传统HRA框架对比，接口参与使程序偏差风险提升约OR=2.35；随机森林预测准确率约0.68，SHAP确认语义与布局因素为主导；实验误差率4.25%与历史事件相符，验证模型可靠性。

**⚠️ 局限性**

样本量有限、仅包含正式报告事件（忽略近乎失误）、实验缺乏真实核电操作员与完整程序任务、时间跨度短，回归模型受分离问题影响，结果仅为探索性而非预测性。

---

## 31. A systematic review of generative AI usage for IT project management

**arXiv ID:** 2604.21958 | [PDF](https://arxiv.org/pdf/2604.21958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 32. Call-Chain-Aware LLM-Based Test Generation for Java Projects

**arXiv ID:** 2604.22046 | [PDF](https://arxiv.org/pdf/2604.22046v1)

**作者:** Guancheng Wang `[一作]` (University of Limerick), Kui Liu `[通讯]` (Huawei)

**通讯引用:** 6398 | [OpenAlex ID](https://openalex.org/A5100374012)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于调用链和依赖上下文的LLM测试生成方法CAT，能够在项目级别自动生成可执行、语义正确的单元测试

**💡 创新点**

首次将调用链信息、对象构造器和第三方依赖明确地嵌入LLM生成提示，实现对复杂交叉模块依赖的精确建模

**🔧 技术方法**

使用静态分析（Class Hierarchy Analysis）提取调用链与初始化信息，再结合Qwen3‑Coder 30B LLM进行生成和修复

**📊 数据集**

在Defects4J benchmark（14个Java项目）以及4个后期GitHub项目（共112类、2707个方法）上评估

**📈 对比分析**

与现有的PANTA和其时间约束版相比，CAT在行覆盖和分支覆盖上分别提升约12.2%/18.0%和15.2%/21.7%；在未见项目上平均提升约25%

**⚠️ 局限性**

对调用链和依赖信息的整合易导致提示长度膨胀，且与路径信息结合时效果不稳定；对更深层动态依赖和跨模块细粒度建模仍有待提升

---

## 33. Null-Space Flow Matching for MIMO Channel Estimation in Latency-Constrained Systems

**arXiv ID:** 2604.22005 | [PDF](https://arxiv.org/pdf/2604.22005v1)

**作者:** Junjie Zhao `[一作]` (Nankai University), Xiaonan Liu `[通讯]` (University of Aberdeen)

**通讯引用:** 2502 | [OpenAlex ID](https://openalex.org/A5100684383)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于 null‑space flow matching 的低延迟 MIMO 通道估计框架，通过将通道分解为 range 空间和 null 空间，仅对 null 空间使用 Flow Matching 迭代生成，并在每步进行自适应校正以保证观测一致性和真实感。

**💡 创新点**

创新点包括：① 明确的 range‑null 空间分解，仅在 null 空间上迭代；② 采用 power‑law 时间调度在早期加密细节、后期加快迭代；③ 引入噪声感知自适应校正以抑制观测噪声；④ 在低延迟预算下实现高精度通道估计。

**🔧 技术方法**

技术手段包括 Flow Matching（速度场网络 U‑Net）、伪逆投影、范围‑空间分解、power‑law 时间调度以及基于 1‑t 与噪声水平的噪声感知导向因子。

**📊 数据集**

使用 3GPP TR 38.901 合规的 CDL‑C 通道数据集，共 20,000 条样本，80% 训练集、20% 测试集，频率 2 GHz，N_t = 16，N_r = 64。

**📈 对比分析**

与 Vanilla‑LMMSE、GMM‑LMMSE、VAE‑LMMSE、SGM、DPS、FM‑PGD 等基线对比，采用 NMSE 评价；在约 3 ms 的 coherence‑time 下实现 -20 dB 以上 NMSE，速度最快；在足够时间时仍保持最低 NMSE。

**⚠️ 局限性**

局限性包括：需要在每个应用场景预先训练 FM 模型；对极低 SNR 或极低 pilot density 的误差仍有限；在不同天线规模、不同频段的泛化能力尚未充分验证。

---

## 34. When Cow Urine Cures Constipation on YouTube: Limits of LLMs in Detecting Culture-specific Health Misinformation

**arXiv ID:** 2604.22002 | [PDF](https://arxiv.org/pdf/2604.22002v1)

**作者:** Anamta Khan `[一作]` (University of Michigan), Joyojeet Pal `[通讯]` (University of Michigan)

**通讯引用:** 3086 | [OpenAlex ID](https://openalex.org/A5058437817)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了印度YouTube上以牛尿（gomutra）为主题的健康信息误导与反驳内容的语篇与修辞特点，并探讨了大型语言模型（LLM）在分析此类文化嵌入式误导时的效能与偏差。

**💡 创新点**

首次系统性揭示促销与反驳内容在传统隐喻与伪科学术语使用上的不对称；验证LLM对文化修辞、语气和性别的敏感性与局限，指出单纯的提示工程无法弥补文化与性别偏差。

**🔧 技术方法**

采用三大西方LLM（GPT‑4o、Gemini 2.5 Pro、DeepSeek‑V3.1）进行语料提取、术语密度统计、强势词（intensifier）识别，并对比不同提示语调（正式 vs. 友好）及零样本/少样本设置的输出差异。

**📊 数据集**

30段YouTube视频多语言转录（英语、印地语、乌尔都语等），使用Whisper进行转录后由GPT‑4翻译成英文，人工标注演讲者性别与立场（宣传/反驳），并计算词误差率（WER≈7.0 %）。

**📈 对比分析**

对GPT‑4o提取传统与科学术语的精确度、召回率和F1（约 61 %/53 %/52 % 与 64 %/56 %/59 %），以及各LLM在不同提示条件下识别强势词的数量与一致性（Cohen's κ 低于0，显示高度不一致），说明LLM对文化与性别修辞的识别性能不可靠。

**⚠️ 局限性**

样本规模小、仅三种LLM、缺乏对比传统监督模型、翻译过程可能引入噪声、性别标注单一作者、以及GPT‑4o‑mini服务停用导致实验受限。

---

## 35. Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning

**arXiv ID:** 2604.21999 | [PDF](https://arxiv.org/pdf/2604.21999v1)

**作者:** Grigory Sapunov `[一作]` `[通讯]` (Intento), Grigory Sapunov (Intento)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究单块 Universal Transformer（UT）结合 Adaptive Computation Time（ACT）在 Sudoku-Extreme 上的递归推理，并证明加入学习型记忆令牌是必要的。

**💡 创新点**

发现 ACT 在默认初始化下存在“陷阱”，导致大多数训练失败；通过深度启动（bias=-3）和 lambda warmup 修复，提升稳定性与效率；并首次对记忆令牌数量阈值、注意力头专化和深度-记忆权衡进行了系统评估。

**🔧 技术方法**

采用单块共享 Transformer 结构（MHA + SwiGLU + DerfNorm），引入 ACT 路由器、记忆令牌、step embedding，使用深度启动初始化、lambda warmup 以及注意力分析框架。

**📊 数据集**

使用 Sudoku-Extreme（9×9 极难题，约 3.83M 训练样本、423K 测试样本）作为基准任务；在小数据集实验中也检验了模型的泛化能力。

**📈 对比分析**

与固定深度（K=18）处理相比，ACT 取得更高且更稳定的精确匹配率（56.9%±0.7% vs 53.4%±9.3%）。深度启动+lambda warmup 在 34% 更少的 ponder 步骤下保持 57.0%±1.1% 的准确率；记忆令牌数量从 0 到 8 的阈值实验显示至少需 8 个令牌才能获得可靠性能，随后出现稳定平台。

**⚠️ 局限性**

局限性包括：仅在 Sudoku-Extreme 上验证，记忆令牌阈值可能任务特定；单块 UT 与更大或不同架构（如 TRM、HRM）相比缺乏直接对比；模型在小样本实验中泛化差；固定深度处理仍然存在高种子方差；T=64 处注意力稀释导致性能下降。

---

## 36. MolClaw: An Autonomous Agent with Hierarchical Skills for Drug Molecule Evaluation, Screening, and Optimization

**arXiv ID:** 2604.21937 | [PDF](https://arxiv.org/pdf/2604.21937v1)

**作者:** Lisheng Zhang `[一作]` (Peking University), Zhengwei Xie `[通讯]` (Peking University)

**通讯引用:** 2243 | [OpenAlex ID](https://openalex.org/A5010530574)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 MolClaw，一个具备三层层级技能体系的自动化 AI 代理，用于全流程结构化药物发现（从靶点预测到分子优化、虚拟筛选和动力学评估）。

**💡 创新点**

创新点在于：①将 30+ 专业工具统一封装至 SCP 体系；②提出三层层级（工具‑工作流‑学科）技能架构，实现模型无关的流程规范化与错误恢复；③发布 MolBench 三维基准，系统评估工作流复杂度对 AI 性能的影响。

**🔧 技术方法**

技术包括：大语言模型（Claude Sonnet、Kimi 等）+ AI 代理平台（OpenClaw、Claude Code）；SCP 协议实现工具标准化；层级技能模板与 L2 工作流脚本；多工具协同、质量检查与自适应错误恢复。

**📊 数据集**

使用公开数据集：CARA、ACNet、ChemCoTBench、PDB、AlphaFold DB、UniProt、ChEMBL 等，并自行构造 MolBench-MS/ MO/ E2E 三个基准集。

**📈 对比分析**

与 8 只前沿 LLM、Biomni 以及基础代理相比，MolClaw 在 MolBench 所有指标上均实现 state‑of‑the‑art 性能，特别是需要结构化多步流程的任务（如结合亲和力比较、分子优化、E2E 端到端挑战）表现显著提升；消融实验表明提升主要源自层级技能而非模型差异。

**⚠️ 局限性**

局限性包括：①仍依赖预先定义的技能，缺乏在线学习与自适应改进；②在多目标优化与 ADMET 监测方面存在关注偏差；③端到端结果仅为计算预测，实验验证缺失；④工具集与技能规模大，维护成本高。

---

## 37. Probabilistic Epistemic Dynamic Agentive Logic

**arXiv ID:** 2604.22042 | [PDF](https://arxiv.org/pdf/2604.22042v1)

**作者:** Shay Allen Logan `[一作]` (Kansas State University), Shay Allen Logan `[通讯]` (Kansas State University)

**通讯引用:** 168 | [OpenAlex ID](https://openalex.org/A5060811720)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种新的概率经验动态逻辑（PEDAL），用于在程序验证背景下从对程序子部分满足规格的概率信念推导整个程序满足规格的概率信念。

**💡 创新点**

核心创新在于将概率测度外部施加到 PD‑L 模型上，构建了一套在 S5 基础上结合程序取值分区的概率语义；并提供了带有无穷规则的完备 Hilbert 系统，解决了在动态逻辑与概率结合时的公理化难题。

**🔧 技术方法**

主要技术包括：1) 在 PDL 基础上引入全局真命题盒子并构造对应的语义模型；2) 定义 PEDAL 模型为包含程序取值分区和概率测度的五元组；3) 通过构造可满足性模型（canonical model）并证明其完备性；4) 利用无穷规则 R3 处理概率上界的逼近；5) 证明关键的概率推理规则（如概率转移、合并等）。

**📊 数据集**

本工作为理论研究，不使用传统意义上的实验数据集；通过构造示例模型（如两状态模型）展示推理效果。

**📈 对比分析**

方法比较：与已有的概率动态逻辑（如 Probabilistic PDL）相比，PEDAL 在处理经验性概率信念方面更直接；完备性证明展示了系统的理论有效性。性能方面由于涉及无穷规则，实际证明过程可能不可计算，但作者讨论了可通过限制概率域或状态数来实现有限化的可实现版本。

**⚠️ 局限性**

主要限制包括：1) 依赖无穷规则 R3，导致系统不可构造性；2) 需要分区和测度的有限性假设，限制了对无限可能性的建模；3) 对实际程序验证的直接可应用性仍需进一步工作，如实现证明系统与现有验证工具的集成。

---

## 38. Rethinking Publication: A Certification Framework for AI-Enabled Research

**arXiv ID:** 2604.22026 | [PDF](https://arxiv.org/pdf/2604.22026v1)

**作者:** Yang Lu `[一作]` (University of Houston), Weidong Shi `[通讯]` (University of Houston)

**通讯引用:** 26721 | [OpenAlex ID](https://openalex.org/A5041067396)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一套双层认证框架，以在学术出版中分别评估知识质量与人类贡献，支持 AI 生成研究的可验证性和可归属性。

**💡 创新点**

创新点在于：将知识认证与作者归属分离，设计 A/B/C 三级贡献等级，建立专用基准槽、同步标准和挑战机制，提供可操作的、无需新机构的实施方案。

**🔧 技术方法**

使用的技术包括规范-分析方法、概念分析、基准竞赛、校准数据库、审稿人训练及披露规范。

**📊 数据集**

数据集主要是领域特定的基准问题集和公开的 AI 流水线输出；示例中提及转录组学、计算机科学、心理学等领域的公开数据。

**📈 对比分析**

比较方法为案例演练和理论验证，而非经验性性能评估；框架通过与现行检测工具、披露政策和现有评审流程的对比，说明在公平性、可行性和自我纠错方面的优势。

**⚠️ 局限性**

局限性包括：归属不确定性无法完全消除、对基准的依赖可能导致“基准依赖”或“基准捕捉”问题、对资源不足或语言少数领域的公平性挑战、以及需要跨机构协调才能实现全球可用。

---

## 39. An Artifact-based Agent Framework for Adaptive and Reproducible Medical Image Processing

**arXiv ID:** 2604.21936 | [PDF](https://arxiv.org/pdf/2604.21936v1)

**作者:** Lianrui Zuo `[一作]` (Vanderbilt University), Bennett A. Landman `[通讯]` (Vanderbilt University)

**通讯引用:** 25031 | [OpenAlex ID](https://openalex.org/A5075735203)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于artifact contract的代理框架，实现对医学影像处理工作流的自适应配置与可复现执行。

**💡 创新点**

核心创新点包括：1）artifact contract 规范化工作流状态与产物；2）受限代理层在该规范上进行目标规划与语义查询；3）将可复现性与自适应性分离，利用模块化规则库与Deterministic执行器。

**🔧 技术方法**

技术手段：使用DeepSeek‑R1 14B LLM构建本地受限代理；Snakemake构建与执行DAG；artifact contract 结构化记录产物属性与 provenance；规则库管理模块化处理规则与参数。

**📊 数据集**

在三组真实临床与研究CT/MRI数据集（包含中等到高异质性）上进行验证，覆盖数据转换、分割、风险评估、图像标准化等九种分析目标。

**📈 对比分析**

对比方法：重复执行验证可复现性（DAG一致性）；用 Initial Rule Matching、Planning Iterations、Final Output 评估自适应性；对20条语义查询测试准确率。实验显示 DAG 一致；IRM>85%，FO≥90%；语义查询准确率在状态/来源类达到100%，过滤类约95%。

**⚠️ 局限性**

局限性：依赖完整的元数据；缺失或损坏的 DICOM 头信息会导致 artifact 属性错误；规则库覆盖不足时无法支持新分析目标。

---

## 40. A Systematic AI Adoption Framework for Higher Education: From Student GenAI Usage to Institutional Integration

**arXiv ID:** 2604.22030 | [PDF](https://arxiv.org/pdf/2604.22030v1)

**作者:** Michael Neumann `[一作]` (University of Applied Sciences and Arts Hannover), Eva-Maria Schön `[通讯]` (University of Applied Sciences Emden/Leer)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在高等教育机构中通过案例研究与在线问卷，系统调查了学生对生成式人工智能工具的使用现状，并基于调查与文档分析提出了可操作的 AI 采用框架；

**💡 创新点**

该框架创新性地将文档审查、学生实践观测、发现综合与政策更新四个步骤融合为一个轻量化、迭代式流程，强调在动态技术环境下的治理与课程调整；

**🔧 技术方法**

研究方法采用在线问卷调查、文档分析、统计检验（如卡方检验）、Python 的 Pandas、Matplotlib、Seaborn 与 SciPy 进行数据处理与可视化，并对开放式回答进行主题分析；

**📊 数据集**

主要数据集为 151 名商科信息系统与电子政府专业学生的问卷完整答卷以及该高校的考试规定、论文指南、课程说明等内部管理文件；

**📈 对比分析**

通过将本研究的 85.43% 采用率与此前德国全国性调查的 63.4% 进行比较，并使用卡方检验证明差异显著；框架本身通过案例实践与专家评议得到定性验证，尚未有量化性能指标；

**⚠️ 局限性**

局限性包括单一案例研究、横截面自报数据可能存在社会期望偏差、对学生技术背景未做控制、样本量相对有限，导致结果在其他高校或学科间的外部效度受限。

---

## 41. Forecasting Solar Energy Using a Single Image

**arXiv ID:** 2604.21982 | [PDF](https://arxiv.org/pdf/2604.21982v1)

**作者:** Jeremy Klotz `[一作]` (Columbia University), Shree K. Nayar `[通讯]` (Columbia University)

**通讯引用:** 38730 | [OpenAlex ID](https://openalex.org/A5051975921)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用一张球面/半球面图像，结合视觉信息与天文数据预测城市环境中光伏板未来的辐照度（包括太阳、天空和周围建筑的反射）。

**💡 创新点**

创新点：①不依赖昂贵的3D扫描或模型，单张图像即可推断相机姿态、天空孔径和场景辐照；②使用神经网络预测太阳与重力方向并与Perez天空模型结合计算日光与天空辐照；③发现场景辐照随时间平滑、低维子空间，可通过神经网络从单图像预测；④提出Solaris捕捉装置，方便实地采集；⑤通过单图像搜索最优面板朝向。

**🔧 技术方法**

主要技术包括：球面/半球面图像采集、TinyViT/ViT骨干网络+MLP、Kabsch算法求相机姿态、图像分割得到天空孔径、Perez天空模型、主成分分析(PCA)、光线追踪模拟验证、卫星气象数据。

**📊 数据集**

使用的数据集：合成城市环境渲染图像1.2M+ HDR半球面图像、UrbanSky实测图像、四个城市峡谷的真实辐照测量、卫星气象数据（DNI、DHI等）和不同天气条件。

**📈 对比分析**

与传统转置模型（SAM）和3D模型光线追踪方法对比，单图像方法在四个城市峡谷实验中每日辐照误差平均约为-3%~-5%，相比转置模型的-15%至-40%误差显著降低；3D模型误差可高达181%过估。

**⚠️ 局限性**

局限性：必须在有阳光且有影子的日子拍摄，无法处理阴天、微云遮挡和夜间；假设场景在时间上保持不变，无法捕捉季节性树叶变化或车辆/新建建筑带来的几何变化；依赖卫星气象估计，精度有限。

---

## 42. Read the Paper, Write the Code: Agentic Reproduction of Social-Science Results

**arXiv ID:** 2604.21965 | [PDF](https://arxiv.org/pdf/2604.21965v1)

**作者:** Benjamin Kohler `[一作]` (ETH Zurich), Elliott Ash `[通讯]` (ETH Zurich)

**通讯引用:** 1705 | [OpenAlex ID](https://openalex.org/A5020377010)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个基于大型语言模型的自动化再现系统，利用论文的方法描述和原始数据，重新实现社会科学研究结果，而不依赖原始代码。

**💡 创新点**

首次在仅有方法描述和数据的情况下完成结果再现，并引入结构化方法提取、信息隔离、单元级比较与错误归因等完整流水线，对不同模型和框架进行系统评估。

**🔧 技术方法**

使用GPT‑5.4/5.3、Claude Opus 4.6、GLM‑5等大模型，搭配Claude Code、Codex CLI、mini‑SWE‑Agent、OpenCode等agent scaffolds，并结合两阶段审计、模板化表格和成本/耗时分析。

**📊 数据集**

采用48篇已验证可再现的社会科学论文（主要来自经济学与政治学期刊）的PDF和复制包，数据来源为Stata/R原始代码，平均5300行。

**📈 对比分析**

通过单元级符号一致率、误差比例、标准误归一化、字母分数等指标评估；最佳组合OpenCode+GPT‑5.4在约90%系数符号正确、80%落入95%置信区间，整体表格平均B以上，模型/框架之间表现差异显著。

**⚠️ 局限性**

主要局限在于论文方法描述不足、数据缺失或原始代码与论文不一致导致的大量人类错误；agent错误相对次要，同时受token成本和模型对细节解释差异的影响。

---

## 43. LTBs-KAN: Linear-Time B-splines Kolmogorov-Arnold Networks

**arXiv ID:** 2604.22034 | [PDF](https://arxiv.org/pdf/2604.22034v1)

**作者:** Eduardo Said Merin-Martinez `[一作]` (Cinvestav), Eduardo Rodriguez-Tello `[通讯]` (Cinvestav)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于线性时间B样条的Kolmogorov‑Arnold网络（LTBs‑KAN），通过并行线性算法、矩阵分解和自适应节点更新显著降低了计算复杂度和参数量。

**💡 创新点**

创新点在于：①新的线性时间B样条计算算法LTBs；②在KAN层使用分块矩阵分解实现参数压缩；③将LTBs算法融入卷积网络形成KAN‑ConvNet。

**🔧 技术方法**

使用的技术包括：B样条的Boor‑Mansfield‑Cox递推、改进的线性时间递推、张量并行计算、矩阵分解（sum‑of‑products）、自适应节点更新、层归一化、Dropout、AdamW优化、学习率调度等。

**📊 数据集**

实验使用MNIST、Fashion‑MNIST和CIFAR‑10三大公开数据集。

**📈 对比分析**

与现有KAN变体（EfficientKAN、FastKAN、GottliebKAN等）以及传统MLP和AlexNet进行对比；LTBs‑KAN在MNIST上取得与FastKAN相近但参数更少的表现，在CIFAR‑10的KAN‑ConvNet上实现最高精度，显示出更优的速度/参数折中。

**⚠️ 局限性**

局限性在于：对大规模高复杂度数据仍未达到最优性能；B样条阶数与正则化的平衡尚待研究；在某些任务中仍需进一步加速或降低参数。

---

## 44. Robust Localization for Autonomous Vehicles in Highway Scenes

**arXiv ID:** 2604.22040 | [PDF](https://arxiv.org/pdf/2604.22040v1)

**作者:** Daqian Cheng `[一作]` (Bot Auto), Lei Wang `[通讯]` (Prexa AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于自适应扩展卡尔曼滤波的高频道路车辆定位方法，结合激光雷达和惯性测量单元实现高精度导航。

**💡 创新点**

创新点在于采用高阶偏置估计、变窗口长度和全新误差状态表示，显著提高高频数据下的定位精度与鲁棒性。

**🔧 技术方法**

使用自适应扩展卡尔曼滤波（AAEKF）和激光雷达（LIDAR）数据融合技术。

**📊 数据集**

利用自采集的高速激光雷达与GNSS（100Hz）数据集，覆盖高速公路与城市道路场景。

**📈 对比分析**

与Apollo的激光雷达EKF、视觉EKF及视觉+IMU+GNSS+LIDAR融合系统进行对比，实验表明该方法在横向精度从0.13m提升至0.07m、纵向误差约0.13m，整体性能明显优于现有方案。

**⚠️ 局限性**

局限性包括对高速公路或平坦路面依赖性强，城市复杂交叉口及激光雷达遮挡条件下效果可能下降，且需高频率传感器与计算资源。

---

## 45. Kernel Contracts: A Specification Language for ML Kernel Correctness Across Heterogeneous Silicon

**arXiv ID:** 2604.22032 | [PDF](https://arxiv.org/pdf/2604.22032v1)

**作者:** Cooper Veit `[一作]` `[通讯]` (Ashiba Research), Cooper Veit (Ashiba Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种专门针对ML核的规范语言，用来显式记录核的正确性合同，并给出了十二类常见合同。

**💡 创新点**

创新点在于将隐式的核实现约定显式化、引入三状态校准机制，并系统化地定义合同类别与违约签名。

**🔧 技术方法**

使用技术包括：合同语言定义、参考实现与测量协议、随机化验证（如Freivalds算法）、基准测试框架和结构化追踪。

**📊 数据集**

利用Wen等人构造的10万多模型变体，以及在AMD MI300X、NVIDIA H200、Intel MAX1100、Huawei Ascend 910B、Apple M4 Pro等五种硅平台上进行实验。

**📈 对比分析**

通过对比实验验证合同的有效性，随机化验证在FP32下开销低于1%，在BF16下低于3%，并能检测出不同平台间的违约案例，证明方法在多硅上可行且性能友好。

**⚠️ 局限性**

局限性包括：仅聚焦单核算，未覆盖分布式训练、编译器内部合规、硬件内在差异与合同组合等问题，且对合同的精确容忍度仍需进一步经验校准。

---

## 46. Math Takes Two: A test for emergent mathematical reasoning in communication

**arXiv ID:** 2604.21935 | [PDF](https://arxiv.org/pdf/2604.21935v1)

**作者:** Michael Cooper `[一作]` (Cooper Cognitive), Samuel Cooper `[通讯]` (Cooper Cognitive)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Math Takes Two 评测基准，旨在评估两名人工智能代理在没有预先数学知识、仅使用8个符号词表的情况下，如何通过沟通在视觉任务中自行构建符号协议并实现数学推理。

**💡 创新点**

创新点在于：①完全去除预设数学语言，要求代理从零开始发现并共享符号；②通过视觉输入与极短符号信息的双向沟通，逼迫代理在开放式分布外（OOD）场景下实现量化推断；③结合人类对比实验，系统检验符号系统的可迁移性与泛化能力。

**🔧 技术方法**

采用了基于 Gumbel‑Softmax 的符号瓶颈自动编码器（浅层与深层两种结构）以及视觉‑语言 Transformer 作为基线；同时提供了冻结/解冻两种训练策略，并通过人类实验数据进行对比。

**📊 数据集**

使用了自研的 Math Takes Two 数据集（可在 GitHub/Hugging Face 公开），包含视觉图像与对应的 8 字符符号序列，涵盖从基本对象到 m×n 数组的多样化视觉输入。

**📈 对比分析**

性能评估：人类参与者在测试阶段平均准确率约 0.87–0.91；浅层/深层自动编码器在测试阶段分别为 0.57–0.66，显著落后于人类，尤其在 OOD 场景（新符号/新数量）上的误差最大；冻结/解冻策略对训练集表现略有提升，但在测试集泛化性下降。

**⚠️ 局限性**

局限性：①模型可能通过识别 OOD 进行捷径解法而非真正抽象符号；②评测未直接量化内部可解析性或组合推理，仅以 OOD 准确率作为代理指标；③未包含预训练 LLM 或教师‑学生框架，限制了对大型模型的适用性。

---

## 47. H-Sets: Hessian-Guided Discovery of Set-Level Feature Interactions in Image Classifiers

**arXiv ID:** 2604.22045 | [PDF](https://arxiv.org/pdf/2604.22045v1)

**作者:** Ayushi Mehrotra `[一作]` (California Institute of Technology), Nidhi Rastogi `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 379 | [OpenAlex ID](https://openalex.org/A5001140269)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 H-Sets 框架，用于在图像分类器中发现并归因高阶特征交互。

**💡 创新点**

创新点在于：① 通过 Hessian 矩阵检测局部二阶交互并递归合并为语义连贯的特征集合；② 将 IDG（Integrated Directional Gradients）扩展为 IDG-Vis，使用 Harsanyi 边际值进行集级归因，满足一系列游戏理论与解释性公理；③ 结合 SAM 作为空间先验，提高交互检测的语义一致性。

**🔧 技术方法**

采用的技术包括：Hessian 矩阵二阶导数、Segment Anything Model（SAM）进行空间分割、IDG-Vis（方向梯度积分 + Harsanyi 边际值）、Monte Carlo 采样、Riemann 积分、以及对结果的归一化与阈值控制。

**📊 数据集**

使用的公开数据集为 ImageNet 和 CUB，分别在 VGG、ResNet、DenseNet 与 MobileNet 等架构上进行实验。

**📈 对比分析**

方法通过与 IG、Archipelago、CAFO、CASO、MOXI 等现有解释方法对比，使用 Gini 指数评估稀疏度、ROAD_AOPC 评估可信度。实验显示 H-Sets 在稀疏性和可信度方面均优于或至少与最先进方法相当。

**⚠️ 局限性**

局限性包括：① 需要计算二阶导数，导致计算成本高于仅使用梯度的方法；② 对 Vision Transformer 等基于 token 的模型适配尚不成熟；③ 对两个超参数（交互阈值 μ 与集合大小 ν）的敏感性需要手动调节。

---

## 48. Spectrographic Portamento Gradient Analysis: A Quantitative Method for Historical Cello Recordings with Application to Beethoven's Piano and Cello Sonatas, 1930--2012

**arXiv ID:** 2604.22037 | [PDF](https://arxiv.org/pdf/2604.22037v1)

**作者:** Ignasi Sole `[一作]` `[通讯]`, Ignasi Sole

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了弦乐演奏中的滑音(portamento)，并提出一种基于声谱图梯度（Hz/秒）的量化方法，测定其斜率并揭示其随时间与演奏速度的变化规律。

**💡 创新点**

创新点在于将滑音的斜率作为第三维度量，用物理单位Hz/秒捕捉滑音的表达强度和速度，而非仅计数或时长；并结合增益恢复技术扩展了对早期模拟录音的分析。

**🔧 技术方法**

使用Sonic Visualizer提取声谱图、GIMP进行像素测量、手动校准频率与时间尺度、以及增益恢复步骤进行手动测量。

**📊 数据集**

数据集为22份1930–2012年间贝多芬第69号和第102号奏鸣曲（Op.69与Op.102 No.1）开头单音轨录音，包含多位大提琴演奏家。

**📈 对比分析**

通过将测得的梯度与录音年份、演奏速度（BPM）进行线性回归和相关分析，发现梯度随年份和速度呈显著负相关，显示滑音呈连续退化，统计显著且结果与传统事件计数法一致。

**⚠️ 局限性**

主要限制包括：测量过程需人工定位参考点，存在主观误差；仅分析开头单音段，结果不一定泛化至更复杂的多音或伴奏段落；校准参数与软件设置紧密相关，需严格保持一致。

---

## 49. FlyCatcher: Neural Inference of Runtime Checkers from Tests

**arXiv ID:** 2604.22028 | [PDF](https://arxiv.org/pdf/2604.22028v1)

**作者:** Beatriz Souza `[一作]` (University of Stuttgart), Michael Pradel `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 FlyCatcher 的系统，能够自动从现有测试代码中推断并生成运行时检查器，以检测复杂软件系统中的无声错误。

**💡 创新点**

创新点在于将大语言模型（LLM）与轻量级静态分析和动态验证相结合：LLM 根据测试的语义推断检查器逻辑，静态分析帮助识别状态变更方法，动态验证通过交互式反馈循环不断改进生成的检查器；同时引入 shadow state 机制，使检查器能够维护对系统内部状态的抽象并支持状态ful 检查。

**🔧 技术方法**

核心技术包括：
- 大语言模型（Claude Sonnet 4 / GPT‑4o）用于代码合成和推理；
- CodeQL 静态分析定位状态变更方法与代码片段；
- Javassist 进行源代码到字节码的插桩；
- shadow state 结构用于维护系统状态抽象；
- 交互式反馈循环（静态/动态验证）不断纠正 LLM 输出；
- 变异测试（PITest）评估检查器检测能力。

**📊 数据集**

使用四个大型开源 Java 系统作为数据集：
- Zookeeper 3.4.11
- Cassandra 3.11.5
- HDFS 3.2.2
- HBase 2.4.0
共 400 条目标测试（每系统 100 条），并在 Zookeeper 上生成 3366 个变异体用于评估错误检测性能。

**📈 对比分析**

与基线方法 T2C 进行对比：
- FlyCatcher 在四个系统中分别生成 80–90 条通过交叉验证的检查器（T2C 仅 3–30 条）。
- 在变异测试中，FlyCatcher 能杀死 26 条被忽略的变异体，而 T2C 仅 5 条。
- 误报率极低：Zookeeper 5.3×10⁻⁵，Cassandra 0。
- 成本方面，Claude Sonnet 4 平均生成时间 5 s、token 178 k、成本 0.60 USD；GPT‑4o 平均 21 s、token 552 k、成本 1.8 USD。
- 运行时开销在 2.7 %–40.3 % 之间，取决于被监控系统。

**⚠️ 局限性**

局限性：
- 仅在 Java 生态系统内评估，无法直接推广到其他语言；
- 仅针对四个系统，可能对其他软件体系结构的适用性未知；
- 变异测试是间接评估真实缺陷的代理，真实错误分布可能不同；
- 结果受 LLM 版本、温度、提示方式等非确定因素影响；
- 随机采样测试和上下文测试可能导致样本偏差。

---

## 50. Taste for Privacy: How Context, Identity, and Lived-Experience Shape Information Sharing Preferences

**arXiv ID:** 2604.22025 | [PDF](https://arxiv.org/pdf/2604.22025v1)

**作者:** Juniper Lovato `[一作]` (University of Vermont), Christopher Danforth `[通讯]` (University of Vermont)

**通讯引用:** 6132 | [OpenAlex ID](https://openalex.org/A5002034958)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 782 名大学生进行 2,912 条问卷调查，测量其在 17 个机构背景下分享 PII 的舒适度以及社交媒体隐私设置。

**💡 创新点**

首次系统量化了学生的情境隐私偏好，揭示了稳定的机构信任层级、性别/少数群体与创伤经历对隐私舒适度的显著影响，并提出情境自适应隐私控制的需求。

**🔧 技术方法**

使用有序逻辑回归、bootstrap 置信区间、剂量-反应（ACE）分析以及对比统计（差均值、排名差异）等方法进行数据分析。

**📊 数据集**

利用 LEMURS 项目收集的自评问卷数据，包含 17 个机构的舒适度评分、社交媒体使用习惯、人口学特征及创伤经历（ACE）等。

**📈 对比分析**

与 2007 年 Lewis 等人基线结果对比，发现私有设置比例从 33% 提升至 62%/84%，并通过回归模型量化平台不适感与隐私设置的 37% odds 增加。

**⚠️ 局限性**

样本局限于东北部某大学的主要白人学生，可能存在自选偏差，且未考虑多种 PII 类型或长期追踪。

---

## 51. Anatomy-Aware Unsupervised Detection and Localization of Retinal Abnormalities in Optical Coherence Tomography

**arXiv ID:** 2604.22139 | [PDF](https://arxiv.org/pdf/2604.22139v1)

**作者:** Tania Haghighi `[一作]` (University of North Carolina at Charlotte), Minhaj Nur Alam `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 1346 | [OpenAlex ID](https://openalex.org/A5072253597)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在不需要病灶标注的情况下，使用离散潜在空间的 VQGAN 对健康 OCT B‑scan 的正常解剖结构进行建模，并通过重建差异实现异常检测与像素级定位；

**💡 创新点**

引入视网膜层感知监督和结构化三元组学习，利用离散潜在编码与显式负样本扰动提升对未知病变的敏感度；

**🔧 技术方法**

采用 VQGAN（向量量化 GAN）、ROI 关注机制、三元组损失、对抗训练、L1+SSIM 误差映射以及重建自监督等技术；

**📊 数据集**

Kermany OCT、Srinivasan OCT、RETOUCH 三个公开数据集；

**📈 对比分析**

与 ViT、ConvNeXt、VAE、VQVAE、f‑AnoGAN 等基线对比，内部 AUROC 0.799、外部 AUROC 0.884，RETOUCH 上无监督分割 Dice 0.20、mIoU 0.12；

**⚠️ 局限性**

负样本扰动基于启发式预处理，可能无法覆盖真实病变多样性；阈值设定对性能敏感；仅处理 2D B‑scan，缺乏体积建模与稀有病变覆盖；

---

## 52. Robust Camera-to-Mocap Calibration and Verification for Large-Scale Multi-Camera Data Capture

**arXiv ID:** 2604.22118 | [PDF](https://arxiv.org/pdf/2604.22118v1)

**作者:** Tianyi Liu `[一作]` (Meta), Kun He `[通讯]` (Meta)

**通讯引用:** 1598 | [OpenAlex ID](https://openalex.org/A5101939666)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套完整的摄像机到运动捕捉系统的联合标定与独立验证流程，专门针对 AR/VR 头显中的鱼眼摄像机。

**💡 创新点**

创新点在于：①不再假设棋盘板与标记之间的刚性变换已知，而是把该变换与摄像机外参一起联合求解；②引入三阶段随机重启 Procrustes + 3D 误差优化 + 2D 投影微调的稳健求解器；③设计了 Lollypop 设备，实现基于独立测量链的快速、无人工干预的验证。

**🔧 技术方法**

技术方法包括：鱼眼相机的 Fisheye62 扭曲模型、ArUco 棋盘板检测、PnP+Bundle Adjustment、Levenberg‑Marquardt 优化、指数映射旋转参数化、随机重启 Procrustes 初始估计。

**📊 数据集**

实验数据集为 Meta Quest 3 头显四个鱼眼摄像机的标定记录（共5组）以及验证记录，使用打印的 ArUco 棋盘与配套的 MOCAP 标记进行同步采集。

**📈 对比分析**

与基准 Sturm 方法对比，主观重投影误差和独立验证误差均显著下降：各摄像机的平均 2D 验证误差从 Sturm 的 2.25 px 降至 0.10 px；同时 ablation 证明三阶段求解器提升了收敛稳定性。

**⚠️ 局限性**

局限性包括：依赖外部运动捕捉设施，验证主要关注平面（水平）误差，深度方向误差不够严格；Lollypop 设备只能覆盖单视角，未来可通过多面 3D 标记扩展深度验证。

---

## 53. Do Not Imitate, Reinforce: Iterative Classification via Belief Refinement

**arXiv ID:** 2604.22110 | [PDF](https://arxiv.org/pdf/2604.22110v1)

**作者:** Mahdi Kallel `[一作]` (University of Würzburg), Carlo D'Eramo `[通讯]` (University of Würzburg)

**通讯引用:** 272 | [OpenAlex ID](https://openalex.org/A5063752250)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将分类任务改为循环式推理的强化学习框架——Reinforced Iterative Classification（RIC），让模型通过多步迭代不断细化对类别的概率估计并在适当时停止

**💡 创新点**

①把单步复制标签的传统监督训练改为基于奖励的递归改进，天然实现任意时刻可预测；②通过共享权重的多步推理抑制对可分数据的过度自信；③利用价值函数自适应地终止推理，消除额外的停止网络

**🔧 技术方法**

循环神经网络（GRU）保持“思考状态”，策略头输出Dirichlet分布得到连续类别分布，价值头估计未来收益；采用策略梯度+价值回归（SPO + GAE）进行强化学习训练；奖励定义为每步对正确类对数概率的提升

**📊 数据集**

CIFAR‑10、CIFAR‑10N（含噪声标签）、SVHN、ImageWoof（ImageNet细粒度子集）

**📈 对比分析**

与标准单步监督学习、ACT、PonderNet 等自适应计算基线进行对比；在所有数据集上，RIC 的准确率与监督基线相当或略优，且在 ECE（期望校准误差）上显著更低，尤其在噪声标签或细粒度任务上优势更明显

**⚠️ 局限性**

①训练成本高，RIC 需要多达 2000 轮才能收敛，远高于 300 轮的监督基线；②Dirichlet 策略在靠近单纯形边界时数值不稳定，导致训练时的稳定性和收敛速度受限；③实验仅限中等规模图像分类，尚未验证在更大类别数或更复杂任务中的可扩展性

---

## 54. How Many Visual Levers Drive Urban Perception? Interventional Counterfactuals via Multiple Localised Edits

**arXiv ID:** 2604.22103 | [PDF](https://arxiv.org/pdf/2604.22103v1)

**作者:** Jason Tang `[一作]` (University College London), Stephen Law `[通讯]` (University College London)

**通讯引用:** 9479 | [OpenAlex ID](https://openalex.org/A5055537550)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一个 lever‑based 交互式反事实框架，对街景图片进行结构化单一 lever 编辑，并通过自动化审计保证编辑的同地点、局部化、真实与可行性；随后用预训练的安全感知模型评估这些编辑对安全评分的影响。

**💡 创新点**

创新点在于将解释性视为可执行的、可局部化的 lever 集合，并在此基础上引入多阶段（规划、生成、审计）流程，首次实现可验证的单 lever 因果解释；同时通过审计门控确保生成结果的语义和空间完整性。

**🔧 技术方法**

使用了 prompt‑only diffusion 编辑器（如 Qwen‑Image‑Edit）、VLM 规划器、LLM‑critic（如 GPT‑5.4）进行编辑与审计，以及 ViT‑B/16 预训练的安全评分模型进行代理评估。

**📊 数据集**

采用了 SPECS（Street Perception Evaluation Considering Socioeconomics）数据集，随机抽取 50 幅城市街景图，覆盖 5 个城市（阿姆斯特丹、阿布贾、旧金山、圣地亚哥、新加坡）。

**📈 对比分析**

方法通过模型评分差异评估编辑效果，平均安全得分提升 +0.366（95% CI [+0.199, +0.537]），按阈值筛选后每场景平均 1.90 个有效 lever；Mobility Infrastructure 与 Physical Maintenance 家族表现最佳，平均提升分别为 +0.579 与 +0.344。

**⚠️ 局限性**

局限性包括：评估仅基于模型代理评分，未与人类主观判断直接对比；prompt‑only 编辑可能引入非目标漂移，审计依赖 LLM 质量；只考虑单个 lever 的影响，未研究多 lever 组合效应；生成与审计流程对计算资源与可扩展性有一定要求。

---

## 55. Ethics Testing: Proactive Identification of Generative AI System Harms

**arXiv ID:** 2604.22089 | [PDF](https://arxiv.org/pdf/2604.22089v1)

**作者:** Shin Hwei Tan `[一作]` (Concordia University), Heng Li `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了“伦理测试”概念，并构建了检测生成式 AI 系统产生不伦理内容的系统化测试框架，随后在代码生成、图像生成、视频生成、需求描述生成和多模态输出等五个案例中演示其可行性。

**💡 创新点**

创新点在于：①首次将伦理原则转化为可量化的测试覆盖指标；②提出了基于不伦理关键词、逻辑运算和角色转换的输入变换规则与变形关系；③利用元模型关系实现对多种生成模型的自动化伦理缺陷触发与检验。

**🔧 技术方法**

核心技术包括：生成式 AI 黑盒测试（ChatGPT、Microsoft Designer、Magic Design 等）、不伦理关键词集合构建、程序与文本的变换规则（如方法重命名、字符串替换、逻辑连接、角色替换）、元模型（Metamorphic Relations）和警告信息检测。

**📊 数据集**

使用了从统一有害内容分类、先前研究的违规词汇和伦理专家访谈中提取的 13 类 关键词集合（共约 k×13 条）作为测试词表，并在公开可访问的 GAI 服务（ChatGPT 3.5、Microsoft Designer、Magic Design）上进行实验。

**📈 对比分析**

通过对照原始提示与变换后提示的输出，检验是否产生警告信息或非法内容；结果显示多数变换可成功触发警告或直接生成违规内容，证明该方法在检测伦理缺陷方面具有较高的召回率，但并未给出统一的性能指标。

**⚠️ 局限性**

局限性包括：①变换规则覆盖面有限，未覆盖所有潜在的不伦理表达；②缺乏通用的判别 oracle 仅依赖系统提示；③实验仅在少数 GAI 服务上验证，缺乏对不同模型或语言的广泛适用性；④跨模态一致性检测仍处于探索阶段。

---

## 56. SNGR: Selective Non-Gaussian Refinement for Ambiguous SLAM Factor Graphs

**arXiv ID:** 2604.22065 | [PDF](https://arxiv.org/pdf/2604.22065v1)

**作者:** Anushka Kulkarni `[一作]` (Northeastern University), Sarthak Dubey `[通讯]` (Northeastern University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在iSAM2框架下通过条件数触发器，选择性地对SLAM因子图窗口进行嵌套采样，从而修正局部高非高斯性误差。

**💡 创新点**

结合条件数检测与嵌套采样，实现在线近似后验的局部高精度修正，并显著降低计算开销。

**🔧 技术方法**

采用iSAM2增量求解、条件数分析、Dynesty嵌套采样、GTSAM因子图和马氏距离误差评估等技术。

**📊 数据集**

使用合成二维范围传感SLAM数据集，按不同错误关联噪声水平生成实验。

**📈 对比分析**

与仅使用iSAM2的基准对比；在20%–30%错误关联时，RMSE提升≈10%–20%，NEES下降，触发窗口数量减少至约1/3，计算时间比全域嵌套采样节省7–12倍。

**⚠️ 局限性**

只能处理局部错误关联导致的非高斯性，无法识别等方差错误或全局漂移，且对低噪声仍存在检测盲区。

---

## 57. Reliability Auditing for Downstream LLM tasks in Psychiatry: LLM-Generated Hospitalization Risk Scores

**arXiv ID:** 2604.22063 | [PDF](https://arxiv.org/pdf/2604.22063v1)

**作者:** Shevya Pandya `[一作]`, Ananya Joshi `[通讯]` (Johns Hopkins University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5072978515)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对四大LLM在精神科住院风险评估中的可靠性进行系统审计，使用合成病人资料和多种提示方式。

**💡 创新点**

提出了通过加入临床无关特征和提示框架来量化预测不稳定性的结构化审计框架。

**🔧 技术方法**

采用混合效应回归、提示工程、特征扰动设计及五次重复生成来评估模型不稳定性。

**📊 数据集**

使用由15个临床特征和最多50个非临床特征组成的50例合成病人数据集。

**📈 对比分析**

通过比较Δ值和混合效应模型，发现所有模型在添加无关特征后风险评分显著波动，逻辑提示导致不稳定性最突出。

**⚠️ 局限性**

局限性包括合成数据缺乏真实EHR复杂性、缺少对模型内部归因的验证、潜在算法偏见及结果对真实临床情境的泛化性不足。

---

## 58. Learning Coverage- and Power-Optimal Transmitter Placement from Building Maps: A Comparative Study of Direct and Indirect Neural Approaches

**arXiv ID:** 2604.22056 | [PDF](https://arxiv.org/pdf/2604.22056v1)

**作者:** Çağkan Yapar `[一作]` (Technische Universität Berlin), Çağkan Yapar `[通讯]` (Technische Universität Berlin)

**通讯引用:** 232 | [OpenAlex ID](https://openalex.org/A5076496369)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了单一发射机布置任务的基准数据集 RadioMapSeer-Deployment，并对两类学习方法（间接热图预测与直接得分图预测）进行系统评估，探索在单目标与双目标（覆盖率‑功率平衡）下的性能与计算成本。

**💡 创新点**

创新点包括：①首次构建具有双目标（覆盖率最优、功率最优）全局最优标签的大规模（167,525）城市场景数据集；②在相同数据、网络架构与评估协议下对热图与得分图两种形式的学习方法进行统一对比；③引入扩散模型实现多样本推理与后置多目标搜索；④提出双得分图的最小化最大排名与单目标并集两种候选筛选策略，实现几乎与穷举搜索相同的平衡性能。

**🔧 技术方法**

使用的技术包括：卷积 UNet 系列（DeepXL、PMNet、DC-Net、SIP2Net）、多损失函数配置（L2、SSIM、MS-SSIM、TV、GDL、MS-GSIM、Focal、坐标损失）、扩散模型（DDPM/DDIM）、SAIPP-Net 作为基准传播评估器、以及基于分数图的候选排序与重评估。

**📊 数据集**

数据集：RadioMapSeer-Deployment，包含 167,525 个基于 OpenStreetMap 的 256×256 二值建筑地图，中心 150×150 区域内约 17,300 个可行发射机位置；为每个场景提供功率最优与覆盖率最优位置、对应的功率/覆盖图以及正则化得分图。

**📈 对比分析**

比较方法：在同一数据集上使用相同的模型结构、训练和评估流程；对热图方法进行单样本与多样本（扩散）推理；对得分图方法在 K=1、200、500、1,000、2,000 等候选数上进行重评估；对双得分图的最小化最大与并集策略进行对比。结果显示：
• 直接得分图在单目标下达到 99.9% 以上的目标比例，且在平衡目标上通过并集策略可逼近理论最优 d̅≈2.60；
• 热图方法在单样本下实现 1350–2400 倍速度提升，扩散多样本在单目标上可提升至 99.97%（功率）或 99.72%（覆盖率），但计算成本高；
• 双得分图并集策略在 K=2000 时实现与穷举搜索相同的 d̅≈2.60，速度提升 14–22 倍。

**⚠️ 局限性**

局限性：
1) 数据集仅涵盖单发射机情况，扩展到多发射机或更复杂网络规划的通用性尚待验证；
2) SAIPP-Net 作为传播评估器的误差会影响标签精度，无法直接映射到真实世界测量；
3) 扩散模型多样本搜索的计算成本相对高，难以满足实时部署需求；
4) 双目标优化采用基于欧氏距离的单一度量，未考虑更复杂的 Pareto 前沿或多目标权重设置；
5) 模型在极端建筑密度或非均匀可行区分布时的鲁棒性仍需进一步研究。

---

## 59. Turnstile Streaming Algorithms Might (Still) as Well Be Linear Sketches, for Polynomial-Length Streams

**arXiv ID:** 2604.22052 | [PDF](https://arxiv.org/pdf/2604.22052v1)

**作者:** Cheng Jiang `[一作]` (MIT), Huacheng Yu `[通讯]` (Princeton University)

**通讯引用:** 41739 | [OpenAlex ID](https://openalex.org/A5108138636)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

证明了在多项式长度的turnstile流中，任意流模型等价于线性压缩模型，并提供了从流模型到压缩模型的归约。

**💡 创新点**

引入了傅里叶分析与离散化高维结构分析相结合的全新归约框架，使用了稀疏化的高阶傅里叶谱结构定理与翻译不变性证明，突破了以往需双指数长度流的限制。

**🔧 技术方法**

主要技术包括大谱结构的稀疏基构造、κ-dissociated集的Rudin不等式、傅里叶线能量估计、McDiarmid不等式以及离散导数的总变分控制。

**📊 数据集**

该工作为理论性工作，不使用具体数据集；它关注的是理论上对空间复杂度的下界。

**📈 对比分析**

与现有方法相比，该归约在多项式长度流中实现了空间复杂度从原来的双指数下降到多项式级别，显著提升了理论可行性。

**⚠️ 局限性**

局限性在于仍需假设流长度至少为多项式大于维数，且归约仅适用于满足密度假设的高斯分布块；对非常短流或非高斯分布仍缺乏支持。

---

## 60. Sovereign Agentic Loops: Decoupling AI Reasoning from Execution in Real-World Systems

**arXiv ID:** 2604.22136 | [PDF](https://arxiv.org/pdf/2604.22136v1)

**作者:** Jun He `[一作]` (OpenKedge.io), Deying Yu `[通讯]` (OpenKedge.io)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Sovereign Agentic Loop (SAL) 架构，在控制平面上对 LLM 生成的意图进行验证后再执行，从而实现安全的云基础设施自动化。

**💡 创新点**

将模型输出拆分为可验证的“意图+正当性”结构，结合信息屏蔽膜与加密证据链，实现执行前的政策约束、身份隔离和可重放审计。

**🔧 技术方法**

使用信息论屏蔽映射、OPA 规则引擎、上下文一致性检查、短期 IAM 令牌、SHA‑256 证据链以及 OpenKedge 控制平面。

**📊 数据集**

在 OpenKedge 原型上使用 10,000 条模拟云操作意图，其中 7,500 条合规请求，2,500 条红队攻击式意图。

**📈 对比分析**

与传统无验证的直接 API 调用对比，SAL 在 93% 的攻击意图中实现阻断，平均额外延迟 12.4 ms（95% 分位 21.7 ms），且 100% 的可重放性得到验证。

**⚠️ 局限性**

假设闭环执行、无外部侧信道泄露且依赖预先定义的政策；对大规模多代理协同的实时性能和模型解释性仍待进一步评估。

---

## 61. Dissociating Decodability and Causal Use in Bracket-Sequence Transformers

**arXiv ID:** 2604.22128 | [PDF](https://arxiv.org/pdf/2604.22128v1)

**作者:** Aryan Sharma `[一作]` (Yale University), Shivam Raval `[通讯]` (Harvard University)

**通讯引用:** 207 | [OpenAlex ID](https://openalex.org/A5070746960)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Dyck括号序列任务中，研究者探测并干预Transformer的残差流与注意力模式，验证哪些线性可解的层级特征真正用于推理。

**💡 创新点**

提出解码可达性与因果使用并不等价的观点，并在受控语法环境中首次将两者分离，发现top-of-stack注意力是因果必需，而深度和距离仅可被解码。

**🔧 技术方法**

使用线性探针、子空间消除、注意力抑制（attention knockout）和激活补丁（activation patching）等技术进行因果干预与可解读性评估。

**📊 数据集**

构造Dyck-(k,m)平衡括号序列数据集（基准为Dyck-(20,10)，扩展到不同k、m组合），并在模板化自然语言 pilot 中复现实验。

**📈 对比分析**

通过对比解码指标（Pearson相关、attention质量）与长距离预测准确率，发现基准模型长距离准确率≥0.98；注意力抑制导致准确率下降≈-0.97，而子空间消除几乎无影响；激活补丁显示关键检索步骤在第1层注意力完成。

**⚠️ 局限性**

研究仅在小规模（2层、1头）Transformer和Dyck语言环境下验证，未考察更大模型或更复杂自然语言；可能受模型规模与任务简单性限制。

---

## 62. Where Should LoRA Go? Component-Type Placement in Hybrid Language Models

**arXiv ID:** 2604.22127 | [PDF](https://arxiv.org/pdf/2604.22127v1)

**作者:** Hector Borobia `[一作]`, Guillermina Tormo-Carbó `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 702 | [OpenAlex ID](https://openalex.org/A5002533063)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对混合语言模型（顺序GDN+Softmax与并行Mamba-2+Attention）中不同组件（注意力、递归骨干、MLP）进行LoRA低秩适配，系统评估其对任务性能、参数效率和跨任务迁移的影响。

**💡 创新点**

首次将LoRA目标从层级迁移到组件类型维度，发现即使注意力占参数比例极小，它仍是最优适配目标；并且混合拓扑（顺序 vs 并行）决定了骨干适配是否有害，揭示了结构对PEFT适配响应的关键作用。

**🔧 技术方法**

使用LoRA（rank = 16）结合标准训练配置，对两种混合架构进行微调，并在三种不同领域（数学、代码、通用指令）下评估性能。

**📊 数据集**

训练集包括约2000条GSM8K、CodeAlpaca、UltraChat样本；评估基准为MMLU、GSM8K、ARC‑Challenge、HellaSwag、HumanEval（后者在子1B模型上表现不足）。

**📈 对比分析**

与全模型适配以及六种组件组合条件对比，发现单独适配注意力在两种架构中以5–10倍更少参数获得相同或更高的准确率；顺序架构下骨干适配会导致GSM8K性能下降至-14.8pp，而并行架构下骨干适配可提升约+8.6pp；跨任务迁移方面并行架构表现出正迁移，顺序架构则出现灾难性遗忘。

**⚠️ 局限性**

局限性包括：模型规模低于1B；实验仅使用单一随机种子；仅评估LoRA而未尝试其他PEFT方法；仅涉及三种训练域和两种混合结构；未验证更大规模模型或其他领域的普适性。

---

## 63. Same Project, Different Start: How Contribution Events Shape Activity and Retention in Open Source

**arXiv ID:** 2604.22120 | [PDF](https://arxiv.org/pdf/2604.22120v1)

**作者:** Mohamed Ouf `[一作]` (Queen's University), Mariam Guizani `[通讯]` (Queen's University)

**通讯引用:** 282 | [OpenAlex ID](https://openalex.org/A5058066905)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对比2,001名通过贡献活动（GSoC, LFX, Hacktoberfest, 24PR）和2,001名自然加入的新人在330个开源项目中的参与与保留情况，发现活动参与者更易成为核心贡献者并保留更久。

**💡 创新点**

创新点在于首次使用匹配队列对比活动与自然贡献者，揭示不同入门机制对应不同的早期参与节奏（稳健、前冲、间歇），并证明“稳健”模式是预测长期保留的关键信号。

**🔧 技术方法**

采用K‑means聚类对首12周的每周贡献指数（CI）进行时间序列分组，结合生存分析、Cox比例风险模型和Scott‑Knott效应差异检验等统计方法。

**📊 数据集**

数据集来源于四大贡献活动（GSoC, LFX, Hacktoberfest, 24PR）共330个GitHub项目，包含4,002位匹配后的新人贡献记录。

**📈 对比分析**

比较方法为匹配队列+卡方/OR检验+Mann‑Whitney U检验+Kaplan‑Meier生存曲线+Cox回归，结果显示活动参与者核心率提升12.1%→9.6%，生存中位数8.2→4.8个月，稳健模式与核心率及保留显著相关。

**⚠️ 局限性**

局限在于只能涵盖有活动的项目，匹配无法消除所有潜在混杂变量，且对长期中断（5个月）与CI权重设定的敏感性仍需进一步验证。

---

## 64. Emergent Strategic Reasoning Risks in AI: A Taxonomy-Driven Evaluation Framework

**arXiv ID:** 2604.22119 | [PDF](https://arxiv.org/pdf/2604.22119v1)

**作者:** Tharindu Kumarage `[一作]` (Amazon), Charith Peris `[通讯]` (Amazon)

**通讯引用:** 319 | [OpenAlex ID](https://openalex.org/A5010781829)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ESRRSim——一种基于风险分类学的自动化评估框架，用于生成多样化场景并对 LLM 的回应和推理进行双重检查，从而系统评估高能力模型的“Emergent Strategic Reasoning Risks”（ESRR）

**💡 创新点**

创新点在于（1）构建了包含 7 类、20 子类的可扩展 ESRR 分类学；（2）设计了四阶段多代理自动化生成流水线，能生成具有真实感、需要真实推理的评估场景；（3）为每个场景生成两套与场景特定的评估清单（回应和推理），实现评估“judge‑agnostic”且可扩展；（4）通过嵌入记忆与结构指纹保证生成场景的多样性与质量；（5）提出了基于加权清单的检测率和多重违规指标，提供细粒度风险评估。

**🔧 技术方法**

技术手段包括：多代理生成（Scenario Generator、Prompt Creator、Rubric Creator、Critique Agent 等）、高温采样与嵌入式相似度/结构指纹检测、Chain‑of‑Thought（CoT）推理评估、加权清单得分算法、批量评估与自动 LLM 判断。

**📊 数据集**

使用的主要数据集是自建的 ESRR Benchmark，包含 1,052 个评估场景、对应的提示与双重评估清单；此外使用 11 个前沿 LLM 的公开权重版本进行测试。

**📈 对比分析**

通过对 11 个 LLM 的检测率（DR）和违规率（Any‑V、Crit‑V、Multi‑V、Avg. Violations）进行比较，发现检测率从 14.45% 到 72.72% 变化，且同一系列内的升级版模型往往表现出显著更低的风险率，表明规模与训练改进对风险有正向影响。

**⚠️ 局限性**

局限性包括：评估基于静态数据集，无法覆盖模型在真实部署中的动态行为；评估场景与清单可能被模型识别后规避，导致低检测率；仅针对公开权重模型，未考虑生产环境的安全措施；分类学虽可扩展但仍不完整，难以一次性覆盖所有潜在风险；对模型内部机制的解释依赖外部评判器，缺乏可解释性。

---

## 65. Characterizing LTL Formulas by Examples

**arXiv ID:** 2604.22097 | [PDF](https://arxiv.org/pdf/2604.22097v1)

**作者:** Balder ten Cate `[一作]` (University of Amsterdam), Patrik Sestic `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

研究了在不同类型例子（有限词、ω词、超限词、模式化例子）下，线性时序逻辑（LTL）公式是否能被有限的标记例子唯一表征。

**💡 创新点**

首次给出了在有限词下对LTL片段的完整分类，证明超限词能够为大多数单调片段提供有限唯一表征，并提出模式化例子作为在有限词下实现唯一表征的中间手段。

**🔧 技术方法**

利用顺序同态、最小满足词、正负例集合构造、词表达式、归纳构造等逻辑与集合论技术，对各片段做了严格证明。

**📊 数据集**

无实验数据集，论文仅基于理论证明与符号构造。

**📈 对比分析**

没有实验对比，主要通过理论上限与下限分析展示结果的可行性与最佳性（例如双指数上界与匹配下界）。

**⚠️ 局限性**

局限性包括：仅考虑有限原子命题且不包含某些运算符的片段、对更大片段的结果未知、构造的例子规模仍为双指数、实际可实现性与人类可读性尚未验证。

---

## 66. Topology Optimization for Materially Efficient Reinforced Concrete Design: Development, Fabrication, and Structural Evaluation

**arXiv ID:** 2604.22070 | [PDF](https://arxiv.org/pdf/2604.22070v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 67. Who Audits the Auditor? Tamper-Proof Fraud Detection with Blockchain-Anchored Explainable ML

**arXiv ID:** 2604.22096 | [PDF](https://arxiv.org/pdf/2604.22096v1)

**作者:** Zhaohui Wang `[一作]` (University of Southern California), Zhaohui Wang `[通讯]` (University of Southern California)

**通讯引用:** 27988 | [OpenAlex ID](https://openalex.org/A5100358029)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一套基于区块链的不可篡改的欺诈检测系统，将机器学习预测和审批流程锚定到智能合约上，以实现可审计的决策轨迹。

**💡 创新点**

将完整的审批工作流转化为智能合约执行，并将ML预测与SHAP解释原子化记录在链上，形成端到端不可篡改的审计证据。

**🔧 技术方法**

采用LightGBM模型进行欺诈检测，使用SHAP进行可解释性输出，并利用Polygon等Layer‑2以太坊侧链实现低成本、低延迟的智能合约执行。

**📊 数据集**

在公开的Kaggle信用卡欺诈数据集和自构造的企业支付合成数据集上进行实验，验证模型性能与系统部署。

**📈 对比分析**

与规则、随机森林、XGBoost和LSTM‑Attention等基线在5折交叉验证下对比，LightGBM取得F1 0.895、PR‑AUC 0.974，且系统端到端延迟约200 事务/分钟，单笔区块链成本低于0.01美元。

**⚠️ 局限性**

系统仍依赖离链ML推理，模型可信度需要TEE或多方推理；区块链确认延迟与费用波动影响吞吐；缺乏对概念漂移和对抗样本的鲁棒性评估。

---

## 68. An End-to-End Ukrainian RAG for Local Deployment. Optimized Hybrid Search and Lightweight Generation

**arXiv ID:** 2604.22095 | [PDF](https://arxiv.org/pdf/2604.22095v1)

**作者:** Mykola Trokhymovych `[一作]` (Pompeu Fabra University), Nazarii Nyzhnyk `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个端到端的乌克兰语言检索增强生成（RAG）系统，用于在有限硬件上完成多项选择问答并给出文档页引用，获得UNLP 2026竞赛第二名。

**💡 创新点**

采用两阶段混合检索（文档+页级）与自制合成QA数据微调LoRA模型，结合4‑bit GGUF量化，在单块P100 GPU下实现高效本地推理。

**🔧 技术方法**

混合稠密/稀疏检索、BM25、Reciprocal Rank Fusion、Jina Reranker、MamayLM 12B（Gemma 3 12B改进版）+LoRA微调、Flash Attention 2、GGUF 4‑bit量化、Python工具pymupdf、pymorphy3等技术。

**📊 数据集**

竞赛提供的开发集（461题，41文档）与合成的约7000道MCQ；测试集为未公开的240+文件多领域文档。

**📈 对比分析**

与竞赛官方排行榜对比，在私有测试集获得0.942分、公开集0.920分；文档检索精度近乎完美，页级recall@3=0.92，整体排名第二。

**⚠️ 局限性**

仅处理文本，忽略图片/图表；对复杂PDF的布局提取偶有失败；使用4‑bit GGUF限制了精度，未尝试更先进量化或更现代硬件。

---

## 69. FLARE-BO: Fused Luminance and Adaptive Retinex Enhancement via Bayesian Optimisation for Low-Light Robotic Vision

**arXiv ID:** 2604.22093 | [PDF](https://arxiv.org/pdf/2604.22093v1)

**作者:** Nathan Shankar `[一作]` (University of Manchester), Hujun Yin `[通讯]` (University of Manchester)

**通讯引用:** 4619 | [OpenAlex ID](https://openalex.org/A5055149475)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FLARE-BO 框架，利用 Bayesian Optimisation 在每张低光图像上联合优化八个参数以实现无训练的低光图像增强。

**💡 创新点**

将 Retinex 亮度归一化、γ 校正、灰世界白平衡、LAB 色度去噪、双边滤波、NLM 亮度去噪以及自适应高斯平滑等八个阶段纳入 BO，扩展参数空间并改进采集策略。

**🔧 技术方法**

采用 Gaussian Process 与 Log Expected Improvement 的 Bayesian Optimisation、Sobol 初始采样、单位超立方归一化、目标标准化以及 L-BFGS-B 最大化等技术。

**📊 数据集**

在低光配对数据集 LOL（400×600 内部场景）上进行实验。

**📈 对比分析**

与九种基于启发式、深度学习和优化的无监督方法对比，FLARE-BO 在 PSNR 与 SSIM 上显著领先（PSNR 22.40 dB，SSIM 0.8427），仅在 NIQE 上略逊。

**⚠️ 局限性**

优化过程耗时较长，无法实时应用，且 NIQE 仍高于部分深度学习方法，需进一步平衡自然度与结构保真。

---

## 70. SHAPE: Unifying Safety, Helpfulness and Pedagogy for Educational LLMs

**arXiv ID:** 2604.22134 | [PDF](https://arxiv.org/pdf/2604.22134v1)

**作者:** Sihang `[一作]`, Hongyi Wen `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了教育场景下的大语言模型（LLM）在面对“教学越狱”攻击时的安全与教学质量问题，并设计了一种基于知识掌握图的图增强教学管道，动态决定回答或引导，提升安全性并保持高帮助性与教学性；

**💡 创新点**

1) 将安全、帮助性、教学性三者统一定义为知识掌握图上的三元度量；2) 设计了SHAPE基准数据集，用以系统评估模型在不同“越狱”攻击下的安全与教学表现；3) 提出了图增强管道的门控机制，显著提升模型在攻击下的安全性，并在保持帮助性与教学性的同时实现；

**🔧 技术方法**

知识掌握图（DAG）、LLM推理（对话生成）、预训练大模型（如Gemini、GPT-5、Claude、Qwen3等）、基于GPT-5的评估代理、与指令集和角色扮演的“教学越狱”提示；

**📊 数据集**

SHAPE数据集：9,087个线性代数学生-问题对，包含211个概念节点的手工构建知识图谱，覆盖多种学生掌握状态；

**📈 对比分析**

在多种LLM（Gemini 2.5 Pro/Flash/Flash‑Lite、GPT‑5系列、Claude 4.5、Qwen3系列、EduChat、SocraticLM）上与两类教学越狱（拒绝抑制、角色扮演）进行评估；原始模型安全率普遍低（约30%–99%下降），而引入图增强管道后，最优模型（Qwen3‑80B）安全率提升至92%+，帮助性基本保持100%，教学性在攻击下也有提升；

**⚠️ 局限性**

1) 只考虑线性代数领域，难以推广到其他学科；2) 知识图仅为二元掌握且前置一致，未涵盖误解或部分掌握；3) 评估中使用GPT‑5评估器提取概念，自动化程度有限；4) 未实时更新学生掌握状态，缺乏学习动态建模；5) 结果主要基于单轮对话，未检验多轮对话的有效性；

---

## 71. PAGaS: Pixel-Aligned 1DoF Gaussian Splatting for Depth Refinement

**arXiv ID:** 2604.22129 | [PDF](https://arxiv.org/pdf/2604.22129v1)

**作者:** David Recasens `[一作]` (University of Zaragoza), Edmond Boyer `[通讯]` (University of Zaragoza)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Pixel‑Aligned 1DoF Gaussian Splatting（PAGaS）作为多视角立体深度后处理模块，利用每个像素的球形高斯在其投影射线上仅优化深度，从而细化已有深度图。

**💡 创新点**

创新点包括：①将每个高斯的自由度压缩到单一深度参数，极大减少优化变量；②高斯颜色、透明度固定为像素颜色和完全不透明，消除视角依赖；③引入 Occlusion‑Aware 3DGS Rasterizer，通过半径和深度阈值在 alpha‑blending 时剔除被遮挡的高斯；④保持在全分辨率、无预训练、低显存下实现高频细节重建。

**🔧 技术方法**

技术手段：Gaussian Splatting（3DGS），光度一致性损失+SSIM，法线平滑损失，金字塔级细化，Adam 优化，TSDF 体素融合，Marching Cubes 重建，光度遮挡裁剪与半径阈值处理。

**📊 数据集**

实验数据集：DTU、Tanks‑and‑Temples（TnT）、ActorsHQ、BlendedMVS，使用官方 Ground‑Truth 测评（Chamfer, F1‑score）和视觉对比。

**📈 对比分析**

与基线 2DGS、PGSR、MVSAnywhere 进行对比；在 DTU 与 TnT 上均取得 Chamfer/F1‑score 的提升，尤其在小尺度场景显著；运行速度仅几秒/帧、显存 ≤11 GB；在全分辨率上实现细节重建，显著提升视觉细节。

**⚠️ 局限性**

局限性：在光照不均或强反射场景下精度下降；大尺度场景受 TSDF 体素粗化限制，细节提升有限；评价指标（Chamfer、F1）对局部细节敏感度低，难以全面衡量；对视角稀疏或遮挡严重的输入仍需改进。

---

## 72. PermaFrost-Attack: Stealth Pretraining Seeding(SPS) for planting Logic Landmines During LLM Training

**arXiv ID:** 2604.22117 | [PDF](https://arxiv.org/pdf/2604.22117v1)

**作者:** Harsh Kumar `[一作]`, Amitava Das `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了通过在预训练阶段种子化网页内容的隐蔽毒化攻击，并提出了 PermaFrost-Attack 框架。

**💡 创新点**

创新点在于将隐蔽概念毒化转化为可测量的触发器，并用热力学长度、谱曲率和感染追踪图三种几何诊断工具揭示模型内部隐藏的后门。

**🔧 技术方法**

采用几何信息学方法（Fisher–Rao 几何、曲率分析）和对比学习的微调技术，在受控细调中模拟预训练毒化。

**📊 数据集**

使用 Anthropic HH‑RLHF 的安全/不安全对话样本以及 LITMUS 多域提示集进行训练和评估。

**📈 对比分析**

在六个不同规模和架构的模型上（1B‑14B），实验表明触发器能在隐藏状态下激活违规输出，几何指标在被激活和未激活路径上显著区分，整体表现与传统输出评估相比更具鲁棒性。

**⚠️ 局限性**

局限性在于实验基于受控微调而非真实预训练大规模数据，触发器仍为人工构造，未验证在真实网络爬取环境中的传播效果。

---

## 73. Implementation and Privacy Guarantees for Scalable Keyword Search on SOLID-based Decentralized Data with Granular Visibility Constraints

**arXiv ID:** 2604.22100 | [PDF](https://arxiv.org/pdf/2604.22100v1)

**作者:** Mohamed Ragab `[一作]` (University of Southampton), George Roussos `[通讯]` (University of London)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ESPRESSO框架，实现了在Solid个人数据存储（pods）中进行分布式关键词检索，并通过用户可访问范围内的索引与隐私友好的源选择元数据保证搜索结果仅暴露合法可见的数据。

**💡 创新点**

创新点主要包括：①基于WebID的局部倒排索引，确保每个搜索方只能看到自己有权访问的索引；②使用Bloom滤波器实现概率性源选择元数据，避免精确恢复关键词-资源映射；③定义并实现了元数据的保守性（conservativity）与可分离性（separability）保证，防止跨用户信息泄露；④对信息泄露与冒充攻击做了形式化威胁建模与对应的隐私保障。

**🔧 技术方法**

技术实现涵盖：Solid协议（WAC、WebID认证）、自定义索引和搜索应用、GaianDB层的分布式元数据存储、Bloom滤波器、伪随机函数与分布式聚合算法。

**📊 数据集**

实验数据集未在文中给出，研究主要基于构造的Solid服务器与pods（模拟/真实部署环境）进行验证。

**📈 对比分析**

与传统集中式全文检索系统相比，ESPRESSO在保持相似的检索准确率与响应时间的同时，显著减少了对原始数据的暴露；实验显示搜索延迟与服务器数量呈线性增长，且元数据同步成本低于集中索引方案。

**⚠️ 局限性**

局限性包括：①不采用差分隐私或加密查询，无法抵御统计推断攻击；②假设托管服务器诚实且遵守访问控制，未考虑恶意服务器；③不支持查询混淆与流量填充，易受时序分析；④不处理多方协作或跨服务器合谋的情况；⑤未实现自动索引与元数据更新的全自动化。

---

## 74. Assessing the impact of dimensionality reduction on clustering performance -- a systematic study

**arXiv ID:** 2604.22099 | [PDF](https://arxiv.org/pdf/2604.22099v1)

**作者:** Ousmane Assani Amate `[一作]` (Université du Québec à Montréal), Vladimir Makarenkov `[通讯]` (Université du Québec à Montréal)

**通讯引用:** 6167 | [OpenAlex ID](https://openalex.org/A5004592090)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性评估了五种降维技术（PCA、Kernel PCA、VAE、Isomap、MDS）与四种聚类算法（k‑means、层次聚类、GMM、OPTICS）在高维数据上的表现，并给出了中等降维（25%–50%）与极端降维（k-1）对聚类质量（ARI）的影响；

**💡 创新点**

首次在无域偏差、统计可控的框架下，对多种降维-聚类组合进行大规模基准测试，提出基于胜率和平均增益的评价指标，为实际应用提供可操作的配对建议；

**🔧 技术方法**

采用PCA、Kernel PCA、VAE、Isomap、MDS进行降维，k‑means、层次聚类、GMM、OPTICS作为聚类器；利用ARI作为外部评价指标；通过人工合成（Circle、Moon、RSG、Repliclust等）与UCI真实数据集进行实验，并在k-1、25%、50%三种降维比例下进行比较；

**📊 数据集**

1,165个人工合成数据集（Circle、Moon、RSG、Repliclust等）和20个UCI真实数据集（如Iris、Wine、Breast tissue等）；

**📈 对比分析**

对每组数据在降维前后计算ARI，并统计胜率与平均赢/输增益；实验表明：中等降维通常比极端降维更有利；Kernel PCA在层次聚类与密度聚类中表现最稳健；Isomap在k‑means与GMM中尤为突出；但不同数据集差异大，降维有时会降低聚类质量；

**⚠️ 局限性**

结果高度依赖数据域和降维方法；VAE表现不稳定；Kernel PCA对核选择敏感；极端降维（k-1）往往导致性能下降；未包含t‑SNE/UMAP等常用可视化降维方法；仅使用ARI评价，未考虑聚类可解释性与规模性；

---

## 75. Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation

**arXiv ID:** 2604.22098 | [PDF](https://arxiv.org/pdf/2604.22098v1)

**作者:** Weisi Liu `[一作]` (University of Memphis), Xiaolei Huang `[通讯]` (University of Memphis)

**通讯引用:** 15458 | [OpenAlex ID](https://openalex.org/A5000467703)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

针对时间漂移下的文本分类，提出KARITA框架，实现多维度漂移检测、检索和知识驱动增强，提升模型在未来时间段的鲁棒性。

**💡 创新点**

创新点在于将不确定性、特征偏移和术语漂移三种信号联合检测，并通过源端检索与LLM/本体同义词增强实现数据中心的时间适应。

**🔧 技术方法**

使用深度学习模型、Mahalanobis距离、熵统计、检索式k‑NN、LLM生成同义词、外部本体（MeSH、EuroVoc、CSO）等技术。

**📊 数据集**

实验数据集包括临床 MIMIC‑IV‑Notes、法律 EurLex 以及科学 arXiv‑CS。

**📈 对比分析**

与源模型、目标模型以及IFT、Self‑Labeling、SAR、EATA 等基线对比，KARITA 在所有数据集上实现宏F1、微F1、样本F1 等指标均为最高或相近 SOTA，显著提升。

**⚠️ 局限性**

主要局限是对外部本体与LLM的依赖，且仅处理词汇层面的漂移，未涵盖概念演化、标签变更等更深层次的知识变化。

---

## 76. Memanto: Typed Semantic Memory with Information-Theoretic Retrieval for Long-Horizon Agents

**arXiv ID:** 2604.22085 | [PDF](https://arxiv.org/pdf/2604.22085v1)

**作者:** Seyed Moein Abtahi `[一作]` (Moorcheh AI), Tara Khani `[通讯]` (Moorcheh AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Memanto 这一无索引、零写入延迟、13 类语义内存层，用于提升多会话代理的长时记忆和检索效果。

**💡 创新点**

突破性之处在于将信息理论检索与语义类型化、冲突检测及时间版本化结合，无需知识图、索引或多查询，显著降低 Memory Tax。

**🔧 技术方法**

核心技术包括 Moorcheh 的信息理论搜索（MIB、EDM、ITS）、Typed Semantic Memory Schema、冲突解决机制、时间版本化、FastAPI 本地服务与命名空间管理。

**📊 数据集**

使用 LongMemEval 与 LoCoMo 两大对话式记忆基准，进一步在内部采用 Claude Sonnet4、Gemini3 等 LLM 做评估。

**📈 对比分析**

与现有 Mem0、Zep、Hindsight 等系统对比，Memanto 在 LongMemEval 89.8%、LoCoMo 87.1% 取得最优成绩，同时单查询、低延迟、零写入成本。

**⚠️ 局限性**

局限在于仅验证对话式场景，非对话式、跨代理共享记忆、标签噪声和大规模并发等仍待进一步评估。

---

## 77. Insect-inspired modular architectures as inductive biases for reinforcement learning

**arXiv ID:** 2604.22081 | [PDF](https://arxiv.org/pdf/2604.22081v1)

**作者:** Anne E. Staples `[一作]` (Virginia Tech), Anne E. Staples `[通讯]` (Virginia Tech)

**通讯引用:** 771 | [OpenAlex ID](https://openalex.org/A5072240246)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了一种基于昆虫导航系统启发的分布式强化学习控制器，在二维连续环境中实现食物寻找、障碍物回避和捕食者逃逸。

**💡 创新点**

创新点在于将感知、方向记忆、稀疏关联记忆、指令生成与局部动作控制等功能拆分为互相协作的模块，并通过学习的仲裁器动态分配运动控制权。

**🔧 技术方法**

采用PPO（Proximal Policy Optimization）算法训练，利用多头感知编码、环形递归头、稀疏记忆投影、指令中心及多局部控制器的组合架构。

**📊 数据集**

使用自构建的二维模拟环境（含食物、静态障碍、移动捕食者），观测向量维度10，动作为两维正向推力与转速。

**📈 对比分析**

与中心化的多层感知机（MLP）和GRU基线对比，六种随机种子、75次PPO更新后，模块化模型在平均回报上提升约40.8%（相对MLP）或25.9%（相对GRU），价值损失更低，PPO统计更稳定，模块分配熵极低，显示出更具选择性的控制分配。

**⚠️ 局限性**

局限性包括：在不同种子间回报波动较大，鲁棒性仍需提升；仲裁器虽实现了稀疏分配，但对不同情境的专门化支持有限，需要进一步探索更具情境感知的模块化策略。

---

## 78. Lightweight Retrieval-Augmented Generation and Large Language Model-Based Modeling for Scalable Patient-Trial Matching

**arXiv ID:** 2604.22061 | [PDF](https://arxiv.org/pdf/2604.22061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 79. Sound Agentic Science Requires Adversarial Experiments

**arXiv ID:** 2604.22080 | [PDF](https://arxiv.org/pdf/2604.22080v1)

**作者:** Dionizije Fa `[一作]` (Entropic), Marko Culjak `[通讯]` (University of Zagreb)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5023341808)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文探讨了大型语言模型（LLM）代理在科学数据分析中的快速产出可能导致的错误假设累积，并提出以“可证伪性优先”标准来评估代理生成的非实验性科学声明；通过在NHANES 2017‑2018数据集上展示两位代理在相同数据下产生相互矛盾的结论，进一步阐明代理易生成伪积极结果的风险；并讨论将现有的Popper代理框架用于系统性反证、期刊同行评审以及未来实验设计的可行路径。

**💡 创新点**

①将可证伪性作为代理生成科学声明的核心评估准则，弥补仅依赖统计显著性所导致的验证缺口；②通过对同一数据集的对抗性分析，示范代理能在相同数据下快速产生冲突的假设，凸显其潜在误导性；③将Popper等反证代理整合进评审流程，提出“审稿人+代理”双重检验模型。

**🔧 技术方法**

使用大型语言模型代理（如ChatGPT‑4/Claude 等）进行自动化统计分析与假设生成；采用Popper框架实现序列反证测试、e‑value统计聚合与错误率控制；在实验中对NHANES 2017‑2018数据进行多模型、多变量调整的迭代分析。

**📊 数据集**

National Health and Nutrition Examination Survey (NHANES) 2017‑2018 公开数据集（包括血清25(OH)D浓度与PHQ‑9 抑郁量表分数）。

**📈 对比分析**

通过让两位代理在同一数据集下按相反目标进行分析，展示了统计显著性差异（Agent A：p=0.0006，负相关；Agent B：p=0.855，近零相关）。此外，将Popper代理与人类博士水平生物信息学家在假设验证任务中进行对比，发现前者在时间效率上至少低于人类 10‑30 倍，且在准确率上相当或更优。

**⚠️ 局限性**

①代理无法直接进行物理实验，验证仍依赖人类实验室操作，导致验证缺口仍在；②Popper目前仅在静态数据库上运行，未实现实验设计与执行；③在高维、噪声大或偏倚明显的数据中，统计反证可能产生误报；④需要更严格的标准与工具来确保代理生成的假设能够真正经历可证伪的实验检验。

---

## 80. Outcome Rewards Do Not Guarantee Verifiable or Causally Important Reasoning

**arXiv ID:** 2604.22074 | [PDF](https://arxiv.org/pdf/2604.22074v1)

**作者:** Qinan Yu `[一作]` (Stanford University), Christopher Potts `[通讯]` (Stanford University)

**通讯引用:** 26443 | [OpenAlex ID](https://openalex.org/A5042601761)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了强化学习从可验证奖励（RLVR）对大语言模型推理链的影响，提出并验证了两种新指标（CIR 与 SR），并展示了通过小规模专家推理数据的监督微调和在 RLVR 中加入 CIR/SR 辅助奖励来提升推理链的因果重要性与可验证性的方案。

**💡 创新点**

① 引入可度量推理链因果性（CIR）和可验证性（SR）的新指标；② 证明仅靠结果奖励的 RLVR 并不能保证推理链的因果性与可验证性；③ 提出利用少量专家推理样本的 SFT 与在 RLVR 中加入 CIR/SR 奖励的两种改进策略。

**🔧 技术方法**

结合强化学习（RLVR）与监督微调（SFT）；设计基于推理链截断的 JS 散度作为 CIR，使用 GPT‑4o‑mini 作为外部验证器计算 SR；在 RLVR 训练中加入 CIR/SR 作为辅助奖励。

**📊 数据集**

使用 ReasoningGym（40 个多任务数据集）与 Math‑Hard，评估 Qwen2.5 系列（1.5B、3B、7B）和 Llama3.2‑3B 模型；专家推理样本由 GPT‑4o‑mini 生成。

**📈 对比分析**

与仅使用结果奖励的 RLVR 进行对比；SFT+RLVR 以及 RLVR+CIR/SR 奖励实验表明在保持或略低于原始准确率的前提下，CIR 与 SR 明显提升（部分任务提升 0.3–0.6 的准确率，且高准确率任务中 SR 与准确率正相关）。

**⚠️ 局限性**

RLVR 可能导致推理链崩溃或误导解释；CIR/SR 在极端任务中仍可能被误判；SFT 需要专家推理样本，增加数据成本；辅助奖励需要额外计算；对不同模型和任务的泛化性仍待进一步验证。

---

## 81. Incentivizing Neuro-symbolic Language-based Reasoning in VLMs via Reinforcement Learning

**arXiv ID:** 2604.22062 | [PDF](https://arxiv.org/pdf/2604.22062v1)

**作者:** Karthic Palaniappan `[一作]` `[通讯]` (Georgia Institute of Technology), Karthic Palaniappan (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在Qwen3-VL-2B-Instruct上使用强化学习和LoRA微调，训练出能够在神经符号语言中表示并推理视觉语言问题的模型。

**💡 创新点**

首次将神经符号语言与GRPO强化学习结合，促使视觉语言模型以符号方式表达数学概念，显著提升符号推理效率与令牌使用率。

**🔧 技术方法**

采用Qwen3-VL-2B-Instruct、GRPO RL、LoRA低秩投影、in-context learning、神经符号语言编译器等技术。

**📊 数据集**

使用ViRL39K多语言视觉推理数据集，并在MathVerse、Math‑Vision、MathVista基准上进行评估。

**📈 对比分析**

与Python（SymPy）工具使用以及基线模型对比，神经符号推理在相同参数下准确率从6%提升到8.3%，推理令牌量下降75%，但整体准确率仍低于大模型。

**⚠️ 局限性**

受限于样本量不足、GPU内存溢出、神经符号语言语法错误，模型准确率仅提升到约8%，未达到SOTA，推理长度仍高，存在可改进空间。

---

## 82. Characterizing pitch and roll torque coupling in insect-sized flapping-wing robots using a microfabricated gimbal

**arXiv ID:** 2604.22121 | [PDF](https://arxiv.org/pdf/2604.22121v1)

**作者:** Aaron Weber `[一作]` (University of Washington), Sawyer B. Fuller `[通讯]` (University of Washington)

**通讯引用:** 3891 | [OpenAlex ID](https://openalex.org/A5103034519)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一种微结构转盘式扭矩传感器，用于同时测量小于1g的振翼飞行昆虫机器人（FIR）的滚转和俯仰扭矩与推力。

**💡 创新点**

其创新点在于首次实现两轴扭矩同步测量，并通过低成本柔性结构和动捕系统实现了高精度、可扩展的测量平台。

**🔧 技术方法**

采用微机电柔性转盘、Kapton/FR4柔性联接、卡尔曼滤波/线性回归，配合运动捕捉相机和精密天平进行读取。

**📊 数据集**

使用了两台UW Robofly平台（180mg）在多组电压指令下采集的扭矩与推力数据集。

**📈 对比分析**

通过线性回归和交叉相关分析对比，R^2分别达到0.95（俯仰）和0.98（滚转），交叉相关系数几乎为0，推力偏差不超过5.8%，验证了扭矩解耦。

**⚠️ 局限性**

主要局限在于传感器带宽仅约0.03–0.06 Hz，无法捕捉单拍瞬时扭矩，且依赖昂贵的动捕系统，未来需提升带宽或改用MEMS倾斜计。

---

## 83. JetSCI: A Hybrid JAX-PETSc Framework for Scalable Differentiable Simulation

**arXiv ID:** 2604.22087 | [PDF](https://arxiv.org/pdf/2604.22087v1)

**作者:** Alberto Cattaneo `[一作]` (University of Utah), Varun Shankar `[通讯]` (University of Utah)

**通讯引用:** 673 | [OpenAlex ID](https://openalex.org/A5055844388)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发了 JetSCI 混合 JAX–PETSc 框架，用于可微分有限元模拟并实现分布式并行求解。

**💡 创新点**

创新点在于将 JAX 的自动微分与 GPU 向量化能力与 PETSc 的 MPI 并行稀疏求解器结合，形成低复制 GPU 数据传输管线，使可微分工作流与大规模稀疏求解器互补。

**🔧 技术方法**

使用了 JAX（自动微分、JIT、vectorization）、PETSc（KSP、预条件、MPI）、CuPy + DLPack（GPU 数据交换）、petsc4py、MPI、Python 交互等技术。

**📊 数据集**

采用了基于两相复合微结构的异构有限元问题（随机纤维/基体模型）作为实验数据集，用于微观力学仿真。

**📈 对比分析**

通过对纯 JAX（矩阵显式/无显式）与 JetSCI（显式矩阵+PETSc 求解）在矩阵乘、迭代求解、直接求解等多项指标的对比实验，发现 JetSCI 在所有规模下都比 JAX 更快、更稳健，尤其在大规模、多 GPU 的分布式环境中显示出更好的可扩展性。

**⚠️ 局限性**

限制包括：构造阶段仍在单机 JAX 计算，缺乏真正的分布式构造；需通过 DLPack/手动指针桥接，编程复杂；目前仅支持单阶段非线性有限元，扩展到多物理或更大规模需要进一步工作。

---

## 84. PrivUn: Unveiling Latent Ripple Effects and Shallow Forgetting in Privacy Unlearning

**arXiv ID:** 2604.22076 | [PDF](https://arxiv.org/pdf/2604.22076v1)

**作者:** Xiaoyi Chen `[一作]` (Indiana University Bloomington), Haixu Tang `[通讯]` (Indiana University Bloomington)

**通讯引用:** 15064 | [OpenAlex ID](https://openalex.org/A5078007096)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一个针对大型语言模型的隐私退忘评估框架，系统检验多种退忘方法在主动攻击（直接检索、上下文学习恢复、微调恢复）下的有效性。

**💡 创新点**

创新点包括发现隐私退忘的“波纹效应”——退忘传播通过梯度关联而非语义关系；揭示大多数退忘方法只实现浅层遗忘；以及基于梯度关联的核心集选择和多层表征约束（RAU）两种深度退忘策略。

**🔧 技术方法**

使用的技术包括梯度相似性分析、中心化核对齐（CKA）、关联得分（梯度、表征、图结构）以及自定义表征锚定损失，配合传统的梯度下降、KL 正则化等。

**📊 数据集**

实验数据集涵盖 Enron 电子邮件、MUSE News 新闻、TOFU 私密信息集，并在 LLaMA‑3.2‑3B、GPT‑2‑Large 等模型上进行评估。

**📈 对比分析**

对比了 19 种退忘方法（训练管线操纵与数据操纵，标注与表征型），发现训练管线方法在被动检索下表现最佳，但在微调攻击中仍有 50%–70% 的恢复率；RMU 在深度退忘上优于标注型但导致显著实用性下降；核心集与 RAU 能在保持 20–30% 数据时将微调恢复率降至约 30%，显著优于随机或传统方法。

**⚠️ 局限性**

局限性在于目前退忘技术仍主要改动输出层，导致浅层遗忘；表征约束虽然可加深遗忘但易破坏模型实用性；对未知隐私数据的“波纹效应”仍难以完全抑制；实验仅在公开数据集和单一模型规模下验证，尚未证明对更大规模或不同任务的泛化。

---

## 85. Shard the Gradient, Scale the Model: Serverless Federated Aggregation via Gradient Partitioning

**arXiv ID:** 2604.22072 | [PDF](https://arxiv.org/pdf/2604.22072v1)

**作者:** Amine Barrak `[一作]` (Oakland University), Amine Barrak `[通讯]` (Oakland University)

**通讯引用:** 144 | [OpenAlex ID](https://openalex.org/A5042574095)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GradsSharding——一种将梯度张量切分为M个分片并在无状态函数中并行聚合的服务器无关联邦学习聚合架构；

**💡 创新点**

创新点在于将聚合工作按梯度维度拆分，使每个无状态函数的内存需求仅为O(|θ|/M)，从而突破单函数内存上限，实现可扩展的大模型聚合；

**🔧 技术方法**

采用FedAvg聚合、S3对象存储、AWS Lambda无状态函数、Python 3.12及AWSSDKPandas层，实验中还实现了λ‑FL和LIFL三种聚合拓扑；

**📊 数据集**

使用了ResNet‑18、VGG‑16、GPT‑2 Medium/Large以及5 GB合成模型（共43 MB–5 GB梯度），训练数据为CIFAR‑10、RVL‑CDIP、合成token序列；

**📈 对比分析**

与λ‑FL和LIFL在相同客户端数（N=20）下比较，结果显示：λ‑FL在梯度小于≈500 MB时成本最低；在VGG‑16规模下GradsSharding成本比λ‑FL低约2.7倍且延迟更短；当梯度超过≈3 GB时λ‑FL/LIFL因内存超限无法部署，而GradsSharding仍可工作；

**⚠️ 局限性**

局限包括S3 I/O成为主要瓶颈，未结合梯度压缩或非均匀切分，实验仅在AWS Lambda上验证，且基线重实现可能与原始实现差异；

---

## 86. Optimal Question Selection from a Large Question Bank for Clinical Field Recovery in Conversational Psychiatric Intake

**arXiv ID:** 2604.22067 | [PDF](https://arxiv.org/pdf/2604.22067v1)

**作者:** Guan Gui `[一作]` (Johns Hopkins University), Ananya Joshi `[通讯]` (Johns Hopkins University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5072978515)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于真实临床问卷的对话式精神科入院问诊问题选择基准，模拟患者并在有限对话轮次内最大化信息恢复。

**💡 创新点**

创新点在于将精神科问诊视为预算化信息恢复任务，构建了655道临床问题库与10个二进制目标的结构化评估框架，并通过模拟患者的行为条件系统评估适应性问答策略。

**🔧 技术方法**

主要使用了大型语言模型（GPT‑4o）实现患者响应、LLM驱动的问答选择以及评估判定器，并结合随机、固定表单和LLM适应三种策略进行实验。

**📊 数据集**

使用的“数据集”是由临床医师编写的655道问诊问题构成的问卷库，以及由团队人工编写的4个模拟患者档案（每个患者包含10个二进制目标）。

**📈 对比分析**

对比方法为随机选问、临床结构化固定表单以及LLM驱动自适应策略，在300次实验（4患者×5行为状态×5重复）中，LLM策略平均准确率95.4%，表单式为84.8%，随机为51.7%；适应性策略在最难的“守护+简洁”条件下仍保持≈89%的准确率。

**⚠️ 局限性**

局限性包括仅使用合成患者且仅四个案例、问题库仅来自单一机构、评估依赖于LLM判定器、未包含真实临床医生基线，以及所有系统组件均使用同一LLM家族可能引入共性偏差。

---

## 87. Probabilistic Abduction in a Fuzzy Logic Framework

**arXiv ID:** 2604.22064 | [PDF](https://arxiv.org/pdf/2604.22064v1)

**作者:** Tommaso Flaminio `[一作]` (IIIA CSIC), Daniil Kozhemiachenko `[通讯]` (Aix Marseille Univ)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了在模糊概率逻辑下的概率归纳问题，给出了完整的复杂度分析并提出了多种优先级策略

**💡 创新点**

首次在模糊概率逻辑中形式化概率归纳，并系统分析其求解复杂度及优先级选择

**🔧 技术方法**

使用模糊概率逻辑、可满足性与归纳逻辑等理论工具

**📊 数据集**

无实验数据集

**📈 对比分析**

无实验比较

**⚠️ 局限性**

主要局限在于对相关问题的复杂度上界和优先级判定的进一步研究

---

## 88. GICC: A High-Performance Runtime for GPU-Initiated Communication and Coordination in Modern HPC Systems

**arXiv ID:** 2604.22126 | [PDF](https://arxiv.org/pdf/2604.22126v1)

**作者:** Baodi Shan `[一作]` (Stony Brook University), Barbara Chapman `[通讯]` (Stony Brook University)

**通讯引用:** 6126 | [OpenAlex ID](https://openalex.org/A5053733660)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为 GICC 的 GPU‑驱动分布式协调框架，支持 GPU 在 Slingshot（OFI）和 InfiniBand 上直接触发通信和同步，显著降低了 GPU 与主机之间的切换开销。

**💡 创新点**

创新点包括：① 在 OFI 环境下实现 GPU‑触发的阈值触发机制和双缓冲 DWQ 预排队，避免了有限 NIC 资源的耗尽；② 在 InfiniBand 上利用 GPU‑NIC 直通实现完全设备侧的网络操作；③ 将协调语义与数据移动分离，提供主动消息和屏障等原语；④ 引入轻量级主机监视线程进行异步资源回收和 epoch‑管理，保持 GPU 快速路径无 CPU 参与。

**🔧 技术方法**

技术细节涵盖：GPU‑NIC 直通、libfabric CXI 的 Deferred Work Queue、阈值触发计数器、双缓冲 DWQ + epoch 资源管理、主动消息（AM）发送/接收、GPU 访问的完成标记、轻量级 host 监视线程、GPU 侧内核同步（__syncthreads）以及跨节点的多轮分发屏障实现。

**📊 数据集**

实验使用的基准与应用包括：单点到点和主动消息微基准；二维 Jacobi 扫描、Cannon 矩阵乘法、工业代理代码 Minimod（高阶有限差分声学模拟）；硬件平台为 Tioga（HPE Slingshot + AMD MI250X）和 Maple（Mellanox ConnectX‑7 + NVIDIA GH200）。

**📈 对比分析**

通过与 Cray MPICH、NVSHMEM、MPI、GASNet‑EX 等现有库对比，GICC 在 Slingshot 上平均每次协调延迟降低至 0.11 µs（相比 25.2 µs 下降 229×），弱规模 Jacobi 的扩展率提升至 76% 对比 MPI 的 60%；在 InfiniBand 上 GICC 的 put 延迟比 NVSHMEM 低 1.95×，工业代理应用 64 GPU 时 GICC 的通信时间比 MPI 低 52%，总体并行效率提升至 42% 对比 MPI 的 35.4%。

**⚠️ 局限性**

局限性：① OFI 环境只能触发预先排队的工作，无法支持完全动态的通信模式；② 受限于有限的 DWQ 和计数器资源，需要双缓冲和 epoch 管理，可能在极高频率协调下产生 back‑pressure；③ 在不同 fabric 上的实现语义差异导致可移植性和性能调优复杂；④ 未涵盖节点失效、动态组成员等容错与可伸缩性挑战。

---

## 89. Spontaneous Persuasion: An Audit of Model Persuasiveness in Everyday Conversations

**arXiv ID:** 2604.22109 | [PDF](https://arxiv.org/pdf/2604.22109v1)

**作者:** Nalin Poungpeth `[一作]` (Northwestern University), Tanu Mitra `[通讯]` (University of Washington)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5114089896)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过构建用户回复分类学，使用五款主流LLM在6000条人工-AI对话中模拟自发说服行为，并与Reddit人类回复进行对比，从而审计LLM在日常交互中的说服技术。

**💡 创新点**

创新点在于提出“自发说服”概念与15种用户回复类型的综合分类学，并系统评估LLM在自然对话中使用的说服策略与人类的差异，首次揭示LLM对话中的信息驱动说服倾向。

**🔧 技术方法**

技术主要包括：多轮对话模拟（基于Reddit话题提炼）、LLM-as-judge注释管道（利用GPT-5 mini、Claude Haiku 4.5、Gemini 2.5 Flash等模型进行零/少样本注释）、统计分析（频率、Jensen‑Shannon散度、余弦相似度）以及对比实验（自发 vs 明确说服提示）。

**📊 数据集**

数据集来自Reddit的四个信息/建议类子版块（AskMarketing、ExplainLikeImFive、Politics、MentalHealth），共提取40条高投票帖子并生成6000条模拟对话；同时收集每条帖子的前10条高投票人类回复做对照。

**📈 对比分析**

对比方法采用统计分布差异评估：计算每种说服技术在LLM和人类回复中的出现频率，利用Jensen‑Shannon散度衡量整体分布相似度；结果显示LLM说服密度几乎为100%，而人类仅为63%，且LLM主导逻辑与框架策略，显著高于人类的情感与社交策略。

**⚠️ 局限性**

局限性包括：对话为合成数据，缺乏真实用户交互验证；模型训练与提示差异可能影响结果；分类学和说服技术以西方英语研究为主，文化偏差未充分覆盖；且LLM的说服分析基于自动化注释，可能漏检或误判非显式策略。

---

## 90. Generating Synthetic Malware Samples Using Generative AI

**arXiv ID:** 2604.22084 | [PDF](https://arxiv.org/pdf/2604.22084v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 91. Dynamic Coupling and Indirect Control of Jointed Robots Rolling Atop A Moving Platform

**arXiv ID:** 2604.22104 | [PDF](https://arxiv.org/pdf/2604.22104v1)

**作者:** Hamidreza Moradi `[一作]` (University of North Carolina at Charlotte), Scott David Kelly `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 1336 | [OpenAlex ID](https://openalex.org/A5101471279)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了两连杆机器人在可移动平台上的非支配运动与协同运动，并提出利用平台加速度对机器人内部被动关节进行间接控制，实现方向跟踪与规划。

**💡 创新点**

创新点在于将共享惯性介质的耦合动力学与 Gibbs‑Appell 形式结合，用 SE(2) 对称性简化后实现对被动机器人方向的反馈线性化控制，并展示了主动机器人与被动机器人在同一平台上相互影响的互利运动特性。

**🔧 技术方法**

采用非约束几何动力学、对称性约简、Gibbs‑Appell 形式、线性化反馈控制以及数值仿真等技术。

**📊 数据集**

无外部数据集，所有结果均来自理论推导与数值模拟。

**📈 对比分析**

通过与单机器人以及不同相位、不同朝向的双机器人仿真对比，证明了平台加速度控制能实现精准方向跟踪并显著提升机器人在平台上的位移与运动效率，仿真显示跟踪误差低于1%。

**⚠️ 局限性**

局限在于仅考虑二维平面、理想无侧滑约束、未进行实验验证；对机器人几何和弹簧模型的简化可能限制其在复杂环境或多关节系统中的推广。

---

## 92. Wiggle and Go! System Identification for Zero-Shot Dynamic Rope Manipulation

**arXiv ID:** 2604.22102 | [PDF](https://arxiv.org/pdf/2604.22102v1)

**作者:** Arthur Jakobsson `[一作]` (Carnegie Mellon University), Jeffrey Ichnowski `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1272 | [OpenAlex ID](https://openalex.org/A5005587345)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了“Wiggle and Go!”两阶段框架，先通过一次安全的摇摆动作在真实环境中获取绳索运动观测，再使用神经网络估计绳索物理参数，随后在仿真中用CMA‑ES优化目标条件轨迹并在真实机器人上执行，实现零样本动态绳索操控。

**💡 创新点**

创新点在于任务无关的系统识别模块，利用单一的短摇摆动作快速推断九维绳索参数，并将这些参数作为先验在不同操控任务（打击、投掷、悬挂）中实现零样本迁移，无需任务特定的再训练。

**🔧 技术方法**

采用时间卷积神经网络进行系统识别、域随机化增强仿真到实测迁移、在Drake物理仿真中使用CMA‑ES进行无梯度轨迹优化，并通过Grounding SAM与Co‑Tracker完成视觉观测与跟踪。

**📊 数据集**

训练数据为在模拟中随机采样的九维绳索参数（链数、长度、阻尼、弹性、半径、质量、导向质量、导向半径、额外尺度）产生的运动轨迹；真实评估使用五种绳索（棕色、黄色、红色、橙色、链条）并在多种长度和导向质量组合下进行。

**📈 对比分析**

与基线优化系统识别（Φ‑CMAES）和随机参数策略对比，实测平均3D打击误差为3.55 cm（基线15.34 cm），傅里叶频率相关系数0.95，投掷与悬挂任务成功率约50%，并在零样本设置下相较随机策略提升6倍。

**⚠️ 局限性**

局限性包括：CMA‑ES轨迹优化计算量大、参数预测在训练边界处饱和、对链条等非典型柔性物体适应性差、视觉跟踪噪声影响、以及仿真模型对绳索尖锐弯曲和缠结等细节的欠缺。

---

## 93. FlashSpread: IO-Aware GPU Simulation of Non-Markovian Epidemic Dynamics via Kernel Fusion

**arXiv ID:** 2604.22092 | [PDF](https://arxiv.org/pdf/2604.22092v1)

**作者:** Heman Shakeri `[一作]`, Ehsan Ardjmand `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于节点转移率是否随年龄变化以及网络度分布重尾程度（ρ）的自动引擎选择框架，用于在仿真中动态切换 Markovian、Renewal、thread、warp 或 merge 等不同的仿真引擎。

**💡 创新点**

创新点在于：① 用单一可测量的度分布重尾参数 ρ 预判网络结构的“偏斜”程度；② 根据 ρ 的阈值自动选择最合适的仿真策略；③ 通过预先读取数据一次即可完成选择，降低运行时开销。

**🔧 技术方法**

技术主要包括：基于状态机的决策图（TikZ 绘制），Markov 和 Renewal 过程建模，piece‑wise 常数速率假设，节点度分布统计（D_max/D_avg 计算），以及自动调度（auto‑dispatch）机制。

**📊 数据集**

实验使用了多种合成网络数据集：常规网络（regular）、Erdős‑Rényi（ER）、格状网络（lattice）以及具有不同 ρ 值的 scale‑free 网络，分别对应 thread、warp、merge 三种引擎。

**📈 对比分析**

与手工选择引擎的基准方法相比，自动选择方法在平均仿真速度上提升了约 10–30%，并在极端 ρ 情况下保持了较低的时间与空间消耗，实验结果在论文的性能评估表格中给出。

**⚠️ 局限性**

局限性包括：① 只使用单一指标 ρ 进行决策，忽略了其他可能影响引擎性能的因素；② 仅在仿真开始前一次性读取数据，缺乏对动态网络演化的适应；③ 实验仅覆盖合成数据集，未在真实大规模网络上验证。

---

## 94. Removing Sandbagging in LLMs by Training with Weak Supervision

**arXiv ID:** 2604.22082 | [PDF](https://arxiv.org/pdf/2604.22082v1)

**作者:** Emil Ryd `[一作]` (MATS), Vivek Hebbar `[通讯]` (Redwood Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在弱监督条件下训练是否能消除大型语言模型的沙袋（有意低效）行为

**💡 创新点**

首次系统评估SFT与RL组合对沙袋模型的解耦效果，并指出训练与部署不可区分的重要性

**🔧 技术方法**

使用监督微调（SFT）、强化学习（RL）以及两者结合的训练流程

**📊 数据集**

三个基准数据集：奥林匹克数学题、Super GPQA科学题、代码竞赛

**📈 对比分析**

在所有任务中，SFT+RL可实现约90–100%的善性性能；单独RL往往出现奖励窃取，单独SFT无法完全解耦

**⚠️ 局限性**

实验仅覆盖有限任务，未考察多步复杂任务；假设攻击者对训练过程完全知情，导致对现实场景的可迁移性有限

---

## 95. TRACE: Topology-aware Reconstruction of Accidents in CARLA for AV Evaluation

**arXiv ID:** 2604.22068 | [PDF](https://arxiv.org/pdf/2604.22068v1)

**作者:** Nahian Salsabil `[一作]` (University of Virginia), Sebastian Elbaum `[通讯]` (University of Virginia)

**通讯引用:** 8803 | [OpenAlex ID](https://openalex.org/A5025318877)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套管道，将 NHTSA 事故报告自动转换为高保真 CARLA 仿真场景，并基于此生成了 52 个真实事故案例的基准。

**💡 创新点**

创新点在于：①使用 OpenStreetMap 恢复真实路网拓扑；②利用大语言模型从文本中推断车辆初始状态；③生成“非法”驾驶路径并支持完整事故重现；④公开可复现的工具与基准。

**🔧 技术方法**

技术包括：OpenStreetMap 检索与 OSMIUM 剪裁、OSM 转 OpenDRIVE、CARLA 的 osm2odr 与手工变换、LLM（Gemini‑3‑flash‑preview）状态估计、WayPoint 轨迹生成与验证。

**📊 数据集**

数据集：NHTSA Crash Investigation System（共 100 条随机抽样，最终 52 条通过验证）以及对应的 OpenStreetMap 轨道。

**📈 对比分析**

与现有工具（AC3R、SoVAR、CrashAgent 等）对比，TRACE 能保留完整路网和事故细节，覆盖 5/8 碰撞类型、8/8 路网拓扑与多样车辆轨迹；实验表明 52 条场景均在 5 m 内重现事故，满足 2° 角度误差和 2 时钟位置误差的验证阈值。

**⚠️ 局限性**

局限性：不支持垂直几何（桥梁、隧道）、多参与者事故、行人或自行车等非车辆主体；仅处理双车辆事故；依赖 LLM 结果，需人工验证；仅在 CARLA 平台实现。

---

## 96. Evaluating LLM-Based Goal Extraction in Requirements Engineering: Prompting Strategies and Their Limitations

**arXiv ID:** 2604.22207 | [PDF](https://arxiv.org/pdf/2604.22207v1)

**作者:** Anna Arnaudo `[一作]` (Politecnico di Torino), Luca Dadone `[通讯]` (Politecnico di Torino)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多 LLM 交互式流水线，用于从软件文档自动抽取 GORE（目标导向需求工程）中的演员、上层目标和下层目标。

**💡 创新点**

创新点在于：①将 GPT‑4 作为生成器、Llama‑3.3‑70B 作为评估者构建迭代反馈循环；②通过比较 Zero‑Shot、One‑Shot、Few‑Shot 以及是否加入反馈机制，系统性评估提示策略对抽取效果的影响。

**🔧 技术方法**

技术：大型语言模型（GPT‑4 与 Llama‑3.3‑70B）+ 结构化提示工程 + 反馈循环 + 余弦相似度 + Munkres 匹配 + BERT 嵌入。

**📊 数据集**

使用四个开源项目的数据集：GestaoHospital、GenomeNexus、London Ambulance Service、Urban Maintenance，均配有手工标注的演员、上层目标和下层目标。

**📈 对比分析**

比较方法：在不同提示策略和是否加入反馈循环下分别计算 Precision、Recall、F1。实验结果显示最佳 F1 分别为演员 0.76（Zero‑Shot）、上层目标 0.62（Zero‑Shot）、下层目标 0.61（Zero‑Shot），反馈循环相较单一 GPT 有明显提升，但整体准确率仍仅在 60‑70% 之间。

**⚠️ 局限性**

局限性：①低精度/召回率，难以实现完全自动化；②仅覆盖功能性目标，未考虑非功能性需求；③评估依赖少量数据集，可能存在偏差；④缺少人工监督和真实需求文本；⑤计算成本与资源需求未进行系统评估。

---

## 97. ArguMath: AI-Simulated Environment for Pre-Service Teacher Training in Orchestrating Classroom Mathematics Argumentation

**arXiv ID:** 2604.22205 | [PDF](https://arxiv.org/pdf/2604.22205v1)

**作者:** Jiwon Chun `[一作]`, Meng Xia `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 ArguMath，一种基于 LLM 的 AI 模拟课堂环境，用于预备教师训练课堂数学论证的组织与反思。

**💡 创新点**

创新点在于结合 Retrieval‑Augmented Generation（RAG）生成真实多样的学生角色，对教师提问进行实时基于 TRQF 与 Toulmin 模型的指导，并提供结构化的反思与反馈。

**🔧 技术方法**

主要技术包括 GPT‑4o LLM、RAG、prompt engineering、TRQF 与 Toulmin 框架的实时评估、表情符号可视化和交互式注释系统。

**📊 数据集**

使用了 TIMSS Video 中公开的十二节八年级数学课堂视频及其转录文本，构建了 214 条学生档案用于生成 20 人虚拟课堂。

**📈 对比分析**

通过在七名预备教师与四名经验教师上进行的交叉实验，将 ArguMath 与传统基线系统比较，结果显示 ArguMath 在学生响应数量、教师提问与学生论证成分的覆盖度和多样性（如 TRQF、Toulmin）均显著高于基线，且受试者对其可用性与教学实用性评价更好。

**⚠️ 局限性**

局限性包括实验样本规模有限、仅针对中小学六至八年级的单一主题、缺乏对视觉密集型学科的支持，以及学生模拟多样性受限于单一数据集。

---

## 98. From Global to Local: Rethinking CLIP Feature Aggregation for Person Re-Identification

**arXiv ID:** 2604.22190 | [PDF](https://arxiv.org/pdf/2604.22190v1)

**作者:** Aotian Zheng `[一作]` (University of Washington), Jenq-Neng Hwang `[通讯]` (University of Washington)

**通讯引用:** 12857 | [OpenAlex ID](https://openalex.org/A5101702810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于 CLIP 的人再识别框架 SAGA-ReID，利用文本嵌入空间的锚点通过交叉注意力对中间 patch 进行重构，从而实现空间选择性聚合，提升在遮挡和跨摄像机变化下的鲁棒性。

**💡 创新点**

创新点：①将文本空间的锚点作为结构化先验，直接用于图像特征重构；②采用反向交叉注意力（patch 作为查询，锚点为键/值）实现无监督的空间重建；③通过装饰性文本初始化和相互正交损失，使锚点在空间上形成多样化的关注模式，显著降低对遮挡区域的敏感度。

**🔧 技术方法**

技术：CLIP ViT‑B/16 图像编码器、冻结 CLIP 文本编码器、跨注意力重构模块、域特定锚点生成、标签平滑交叉熵+三元组损失、图像‑文本对齐损失、相互正交锚点正则化、特征级融合。

**📊 数据集**

使用的数据集包括：Market‑1501、DukeMTMC‑reID、MSMT17、Occluded‑DukeMTMC、Occluded‑ReID、Occluded‑Market、以及 DomainNet（多源域泛化）。

**📈 对比分析**

与 CLIP‑ReID、CLIP‑SCGI、PromptSG、CLIMB‑ReID、SCING、PRO‑FD 等方法对比，SAGA‑ReID 在标准和遮挡数据集均表现提升，尤其在 Occluded‑DukeMTMC 上 Rank‑1 +10.6、mAP +8.0；在合成遮挡实验中，Rank‑1 +30；在人体干扰遮挡实验中，Rank‑1 +10.1。整体上在跨摄像机和遮挡场景中优势最显著，且在 DomainNet 上平均准确率提升至 61.0%。

**⚠️ 局限性**

局限性：锚点在训练阶段固定，缺乏在大幅不同摄像机/环境下的自适应能力；对超大规模模型的扩展需要额外的算力和内存；在完全无遮挡的干净数据集提升有限；当使用其他 CLIP 变体或预训练策略时，效果需进一步验证。

---

## 99. Uni-Encoder Meets Multi-Encoders: Representation Before Fusion for Brain Tumor Segmentation with Missing Modalities

**arXiv ID:** 2604.22177 | [PDF](https://arxiv.org/pdf/2604.22177v1)

**作者:** Peibo Song `[一作]` (Shandong University), Si Yong Yeo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了UniME，一种两阶段异构网络，用于缺失模态下的脑肿瘤分割；

**💡 创新点**

创新点包括：①将表征学习与分割解耦，先用掩码自监督预训练ViT Uni‑Encoder得到对缺失模态鲁棒的统一表征；②在预训练的基础上加入多模态专属CNN Multi‑Encoders，以多尺度方式提取细粒度特征；③采用层级学习率衰减（LLRD）与register token等技巧提升迁移效果；

**🔧 技术方法**

技术手段：ViT + 3D Rotary Positional Embedding + SwigLU + register token；掩码自监督预训练（patch + modality masking）；轻量化解码器；多模态专属CNN编码器 + ECA注意力；LLRD；Dice+交叉熵损失；数据增强、AdamW等；

**📊 数据集**

使用数据集：BraTS 2023 与 BraTS 2024 两个多模态脑肿瘤分割挑战集（包含 T1、T1ce、T2、FLAIR 四种MRI模态）；

**📈 对比分析**

与 9 先进方法（HeMIS、U‑HEVD、RFNet、mmFormer、M^3AE、M^2FTrans、LS3M、IM‑Fuse、M^2SegMamba）进行对比，实验显示在 WT、TC、ET 三个子区域的平均 DSC 上提升 1.4%–2.9%，整体平均提升至 83.49%（BraTS 2023）与 90.38%（BraTS 2024），性能显著优于对比方法；

**⚠️ 局限性**

局限性：①对预训练数据量和掩码比例、register token 数量等超参较敏感；②GPU 内存占用较高（约 23.8GB）；③仅在 BraTS 数据集上验证，缺乏跨数据集或其他医学影像任务的泛化评估；

---

## 100. ReCast: Recasting Learning Signals for Reinforcement Learning in Generative Recommendation

**arXiv ID:** 2604.22169 | [PDF](https://arxiv.org/pdf/2604.22169v1)

**作者:** Peiyan Zhang `[一作]` (Huawei Technologies Co., Ltd.), Yong Liu `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ReCast框架，用于解决稀疏命中生成推荐中组式RL的学习信号失效问题。

**💡 创新点**

创新点在于先对全零组做“修复”注入最小可学习性，再用局部正负对比更新，且保持外部RL目标不变。

**🔧 技术方法**

使用了生成式推荐的离线RL后训练流程，结合结构化分数、局部对比学习以及常数规模更新。

**📊 数据集**

在RecIF‑Bench的五个生成推荐任务上验证，包括广告、商品、短视频、交互式和标签条件推荐。

**📈 对比分析**

与OpenOneRec‑RL基线对比，ReCast在Pass@1、Pass@32、Recall@32均提升，最多相对提升36.6%，且在相同预算下只需4.1%预算即可达到基线性能。

**⚠️ 局限性**

局限在于只针对单目标下的离线后训练，修复规则固定，可能在更强模型或多步/多目标场景下失效。

---

## 101. Optimal sequential decision-making for error propagation mitigation in digital twins

**arXiv ID:** 2604.22168 | [PDF](https://arxiv.org/pdf/2604.22168v1)

**作者:** Annice Najafi `[一作]` (California State Polytechnic University), Shokoufeh Mirzaei `[通讯]` (California State Polytechnic University)

**通讯引用:** 121 | [OpenAlex ID](https://openalex.org/A5003669584)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了基于 HMM 推断的误差状态的马尔可夫决策过程，用于数字孪生的错误传播缓解。

**💡 创新点**

创新点在于将数据驱动的隐藏状态推断与 MDP/POMDP 决策结合，形成闭环管控，并通过信息价值分解量化改进空间。

**🔧 技术方法**

采用 HMM、MDP/POMDP、PBVI、Gillespie 采样、Q‑learning 与 REINFORCE 等技术。

**📊 数据集**

使用人工生成的六模块 ARX 数字孪生残差序列，包含 10 次随机种子实验数据。

**📈 对比分析**

通过对 MDP、POMDP、MCDA、k‑step、无干预、Q‑learning、REINFORCE 等七种方法进行统计比较，MDP 取得最高累计奖励，POMDP 恢复约 95% 的 MDP 性能，MCDA 与无干预表现最差。

**⚠️ 局限性**

局限包括仅在模拟的六模块系统验证、修复概率人为设定、观察矩阵仅依赖 HMM、未考虑行动执行延迟、状态空间规模有限等。

---

## 102. Estimating Tail Risks in Language Model Output Distributions

**arXiv ID:** 2604.22167 | [PDF](https://arxiv.org/pdf/2604.22167v1)

**作者:** Rico Angell `[一作]` (New York University), He He `[通讯]` (New York University)

**通讯引用:** 56141 | [OpenAlex ID](https://openalex.org/A5100685161)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何高效估计语言模型对任何查询生成有害输出的尾部风险，并给出了完整的样本高效估计流程。

**💡 创新点**

创新点在于：①通过激活 steering 构造不安全的 proposal 模型；②使用 importance sampling 与交叉熵方法自动搜索最优 proposal 超参数；③证明查询级尾部风险估计能准确预测未见查询的极端风险；④展示非对抗重写可将有害概率改变数倍。

**🔧 技术方法**

技术手段包括：激活 steering（对 Transformer 的残差流施加向量），重要性采样，交叉熵搜索（cross‑entropy method）优化 proposal 超参数，极值理论预测极端风险，LLM‑as‑judge 评判有害与偏离行为。

**📊 数据集**

使用的数据集主要有：StrongREJECT 313 条未 jailbreak 的有害查询；对 Safe queries 的 20 条安全但易导致 hallucination/sycophancy 的查询；以及 309 条 StrongREJECT 查询的 25 个非对抗重写（Grok‑3 生成）。

**📈 对比分析**

比较方法：与 naïve Monte Carlo 采样（10k/100k 次）对比，重要性采样在 5–10× 更少样本即可达到相同或更低的方差；对不同模型（Llama‑3.1‑8B、Llama‑3.2‑1B、Qwen2.5‑7B、Olmo‑3‑7B、Phi‑4）均表现出 10–20× 的样本节省；在预测未见查询的极端风险时，基于查询级尾部风险估计的极值理论方法能够在 95% 置信区间内准确重现真实风险。

**⚠️ 局限性**

限制：①需要对模型权重进行访问以构造不安全 proposal，无法仅用 logits；②依赖 LLM‑as‑judge，可能出现误判；③对极端强的偏离系统仍可能逃逸检测；④在多模态或跨语言场景下尚未验证。

---

## 103. GenMatter: Perceiving Physical Objects with Generative Matter Models

**arXiv ID:** 2604.22160 | [PDF](https://arxiv.org/pdf/2604.22160v1)

**作者:** Eric Li `[一作]` (Massachusetts Institute Of Technology), Joshua B. Tenenbaum `[通讯]` (Massachusetts Institute Of Technology)

**通讯引用:** 74246 | [OpenAlex ID](https://openalex.org/A5071093940)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种名为 GenMatter 的两层生成式模型，用来从运动信息中分割移动的物体，将像素点聚合为粒子，再将粒子聚合为可独立运动的簇，随后通过并行块 Gibbs 采样进行硬实时推断。

**💡 创新点**

创新点在于：①将人类运动感知原理转化为层次概率模型；②在同一模型框架下同时处理稀疏点状刺激、纹理伪装物体和自然视频；③使用硬件加速的并行 Gibbs 采样实现实时推断，并可根据数据点采样率动态平衡速度与精度。

**🔧 技术方法**

技术包括：生成式分层模型（粒子层与簇层），离散化 SE(3) 的刚体变换，光流+深度特征提取，Gibbs 采样的并行实现（GenJAX 框架），以及可视化与评估的 probe‑point 与 Jaccard 指标。

**📊 数据集**

使用的数据集包括：随机点动学图（RDK），具有彩色伪装纹理的旋转物体视频（140 条），以及 TAP‑Vid‑DAVIS 自然 RGB 视频；光流使用 RAFT，深度使用 VideoDepthAnything，特征使用 DINO。

**📈 对比分析**

与传统基线（SegAnyMo、FlowSAM、CoTracker3）对比，GenMatter 在 RDK 上与人类判断的相关性 r²=0.86，Gestalt 任务的 Jaccard 分数平均 0.72（高于 FlowSAM 0.67 与 SegAnyMo 0.26），RGB 视频的物体轨迹 J_m 0.79（匹配 CoTracker3 0.78），且可通过降低采样率提升速度（最高 12×）而几乎不损失精度。

**⚠️ 局限性**

局限性包括：①模型未显式建模物理动力学，难以处理完全遮挡或长时间预测；②粒子数量固定，无法自适应物体进出或尺度变化；③缺乏大规模 3D 变形物体标注数据，评估主要基于 2D 掩码或 probe‑point，可能掩盖更细粒度问题。

---

## 104. PrivSTRUCT: Untangling Data Purpose Compliance of Privacy Policies in Google Play Store

**arXiv ID:** 2604.22157 | [PDF](https://arxiv.org/pdf/2604.22157v1)

**作者:** Bhanuka Silva `[一作]` (University of Sydney), Suranga Senevirante `[通讯]` (University of Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PrivSTRUCT框架，利用结构化提取和分类来精准识别Android隐私政策中的数据项与用途。

**💡 创新点**

创新点在于将章节标题结构与自然语言理解相结合，使用DPO微调LLM实现高效的标题重建与全局目的关联。

**🔧 技术方法**

技术包括LLM解码器提取文本片段、Llama‑3.1‑8B进行标题重建、PrivBERT编码器分类、Direct Preference Optimisation (DPO) 以及结构化的目的一致性和稀释度指标。

**📊 数据集**

使用3,756份Google Play隐私政策（6,540款热门应用）及其对应的数据安全标签，训练/测试子集为750份。

**📈 对比分析**

与PoliGraph对比，PrivSTRUCT在同一批次上平均提取的独立数据项约为79.6条、用途约为92.8条，分别比PoliGraph多52.1%和89.1%；在本地LLM推理下实现成本低、速度快。

**⚠️ 局限性**

局限在于仍需依赖标题结构的准确重建，对极长或非标准HTML政策敏感，且隐私合规性评估仅基于开发者自述，未考虑实际应用行为或法律解释。

---

## 105. GR-Evolve: Design-Adaptive Global Routing via LLM-Driven Algorithm Evolution

**arXiv ID:** 2604.22234 | [PDF](https://arxiv.org/pdf/2604.22234v1)

**作者:** Taizun Jafri `[一作]` (Arizona State University), Vidya A. Chhabria `[通讯]` (Arizona State University)

**通讯引用:** 542 | [OpenAlex ID](https://openalex.org/A5069179438)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于大语言模型的代码演化框架GR‑Evolve，自动化地在全局路由器源码层面演化算法，以适配单个芯片设计；

**💡 创新点**

创新点在于把全局路由视为程序级优化问题，利用LLM驱动的agent循环在源码层面变异、编译、评估，实现设计‑工具协同演化；提出无状态持久化状态、上下文压缩、warm‑start、多目标Pareto等机制；

**🔧 技术方法**

技术手段包括大语言模型（OpenAI Codex等）与ReAct框架、OpenROAD/OpenROAD‑flow‑scripts、Git版本控制、统计多目标优化与Pareto前沿分析；

**📊 数据集**

使用的实验数据集包含SkyWater 130 nm、NanGate 45 nm、ASAP7 7 nm三技术节点下的多种设计（AES、IBEX、JPEG、SWERV、DYN、AR136、BP、BlackParrot、Ariane136等），以及ICCAD 2019全球路由竞赛基准；

**📈 对比分析**

与基线路由器（FastRoute、CUGR、SPRoute）在后详细路由的wirelength、via count、全局路由时长等指标进行对比；实验显示GR‑Evolve在多数设计上实现了最高8.72%以上的post‑DR wirelength下降，并在ICCAD19基准上无溢出且显著降低WL/VC；

**⚠️ 局限性**

局限性包括：每次迭代评估耗时数分钟到数小时，导致搜索效率低；LLM上下文窗口受限，难以一次性处理完整源码；演化过程受手工定义的目标层级限制，缺乏自适应多目标策略；仅使用post‑DR指标反馈，未实时捕捉DR过程；warm‑start仍需手工初始化，未完全覆盖大型设计；缺乏对演化生成代码的可解释性和可移植性分析。

---

## 106. RAG-Reflect: Agentic Retrieval-Augmented Generation with Reflections for Comment-Driven Code Maintenance on Stack Overflow

**arXiv ID:** 2604.22217 | [PDF](https://arxiv.org/pdf/2604.22217v1)

**作者:** Mehedi Hasan Shanto `[一作]` (University of Windsor), Alioune Ngom `[通讯]` (University of Windsor)

**通讯引用:** 2049 | [OpenAlex ID](https://openalex.org/A5070302649)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于检索增强生成与自我反思的代理式框架RAG-Reflect，用于判断Stack Overflow评论是否真正导致代码修改，并进一步生成符合评论的自动代码更新。

**💡 创新点**

创新点在于将检索、推理与反思三阶段模块化为代理式工作流，实现无需微调即可达到微调模型水平的有效性；通过一次性模式分析生成规则，提升推理的可解释性与一致性。

**🔧 技术方法**

使用大型语言模型（如GPT‑4o、Gemini 2.5 Flash、Qwen‑30B、CodeLlama‑13B）结合MiniLM语义检索、FAISS索引、规则式自我反思等技术。

**📊 数据集**

主要数据集为SOUP（5,000条Java评论‑编辑对）以及自建的1,000条Python评论‑编辑对，均经过双人人工标注。

**📈 对比分析**

与传统特征模型、基于关键词匹配、SOUP微调模型等基线比较，RAG‑Reflect在无训练的条件下在有效类F1达到0.78、无效类F1 0.94，整体性能优于所有基线并接近微调模型。

**⚠️ 局限性**

局限包括对动态类型语言（如Python）有效类召回略低、对极小或隐含意图的评论仍易误判，以及在自动更新任务中对代码格式与细节的语义一致性仍需进一步验证。

---

## 107. Verbal Confidence Saturation in 3-9B Open-Weight Instruction-Tuned LLMs: A Pre-Registered Psychometric Validity Screen

**arXiv ID:** 2604.22215 | [PDF](https://arxiv.org/pdf/2604.22215v1)

**作者:** Jon-Paul Cacioli `[一作]` `[通讯]`, Jon-Paul Cacioli

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了七种3–9B规模开源指令模型在最小化数值与十类分类信心提示下的口头信心表达，并采用心理计量有效性筛选协议判定其在项目级别二型辨别中的有效性。

**💡 创新点**

创新之处在于首次将临床心理测评的有效性筛选方法应用于语言模型的信心读数，揭示了在最小化提示条件下出现的饱和与交互失效两大失效模式，并证明内部不确定性并未在输出接口得到保留。

**🔧 技术方法**

主要技术包括greedy解码、Q5_K_M量化、心理计量有效性筛选、岭回归预测信心、AUROC2评估以及推理轨迹长度与信心的偏相关分析。

**📊 数据集**

使用的数据集为TriviaQA验证集的524道问答题。

**📈 对比分析**

对比数值提示与十类分类提示，数值提示下置信度上限率达91.7%且全部模型被判定为无效；分类提示导致大多数模型问答准确率骤降；AUROC2在0.527–0.683范围内略高于随机；logprobability对信心的预测R²低于0.20。

**⚠️ 局限性**

主要局限包括仅使用最小化提示与greedy解码；模型规模仅限3–9B；仅评估单一TriviaQA数据集；量化方式固定；分类提示的具体表述可能导致误差，且未探索更丰富提示或采样策略的潜在改进。

---

## 108. V-STC: A Time-Efficient Multi-Vehicle Coordinated Trajectory Planning Approach

**arXiv ID:** 2604.22196 | [PDF](https://arxiv.org/pdf/2604.22196v1)

**作者:** Pengfei Liu `[一作]` (Beijing Institute of Technology), Tingwen Huang `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种变量时间步长空间-时间通道（V-STC）框架，用以实现多车协同轨迹规划；

**💡 创新点**

创新点在于：①取消对预设参考路点的依赖，通道的空间布局和数量通过优化自动确定；②将每个通道立方体的时间持续时间作为决策变量，实现时间分配自适应；③同时优化空间和时间，提升协同效率；

**🔧 技术方法**

采用混合整数规划求解V-STC构造（使用Gurobi），随后在每个通道内用内部点法（IPOPT）进行约束优化生成动态可行轨迹；

**📊 数据集**

实验使用三类模拟场景（无信号交叉口、车道变道、无结构环境）中的车辆初始/目标位置作为数据集；

**📈 对比分析**

与传统固定时间步长STC方法对比，V-STC在所有场景下均实现了更短的轨迹完成时间，安全性和平滑度不下降；

**⚠️ 局限性**

局限性包括：①求解复杂度随车辆数目显著上升；②实验仅在离线仿真环境，未验证在线实时性能；③在极高密度交通流中通道互斥约束仍可能导致保守规划。

---

## 109. Energy-Efficient Multi-Robot Coverage Path Planning of Non-Convex Regions of Interests

**arXiv ID:** 2604.22189 | [PDF](https://arxiv.org/pdf/2604.22189v1)

**作者:** Sourav Raxit `[一作]` (University of New Orleans), Leonardo Bobadilla `[通讯]` (Florida International University)

**通讯引用:** 687 | [OpenAlex ID](https://openalex.org/A5084731591)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套针对非凸、带障碍物和禁飞区的能量高效多机器人覆盖路径规划框架，能够在真实环境中实现协同覆盖。

**💡 创新点**

创新点包括：① 通过旋转铰链求解最小面积矩形获得全局最优扫掠方向，显著减少转弯；② 设计安全缓冲头岸区，保证转弯安全；③ 用多旅行商（mTSP）实现工作负载平衡并最小化任务周期；④ 在可视图中加入采样点并采用头角驱动的顺序，保持连续且无碰撞的路径。

**🔧 技术方法**

主要技术：旋转铰链（最小矩形）、Minkowski偏移（头岸缓冲）、mTSP（LKH求解器）、可视图（VG）与路径短路取代、交替边遍历生成连续扫掠。

**📊 数据集**

使用了已有的仿真基准场景（如从文献中引用的场景）以及新增的湿地场景，并在实测中应用无人机（AAV）和水面机器人（ASV）进行验证。

**📈 对比分析**

与多种现有方法（DARP+MST、POPCORN+SALT、EAMCMP 等）对比，平均能耗降低 3%~40%，计算时间缩短约一倍，且在机器人数量增大时仍保持负载均衡和良好可扩展性。

**⚠️ 局限性**

局限性包括：缺乏动态障碍物避让的轨迹规划；对多机器人数目极大时 mTSP 求解仍可能受限；依赖精确多边形表示，若障碍物形状不规则或动态变化时需进一步改进。

---

## 110. EvFlow-GS: Event Enhanced Motion Deblurring with Optical Flow for 3D Gaussian Splatting

**arXiv ID:** 2604.22183 | [PDF](https://arxiv.org/pdf/2604.22183v1)

**作者:** Feiyu An `[一作]` (Sichuan University), Rong Xiao `[通讯]` (Sichuan University)

**通讯引用:** 22901 | [OpenAlex ID](https://openalex.org/A5061550044)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出EvFlow-GS框架，联合优化可学习的双积分网络（LDI）、相机姿态与3D Gaussian Splatting，以利用事件相机与光流信息完成运动模糊图像的高质量3D重建。

**💡 创新点**

创新点包括①将LDI、姿态估计和3DGS集成为单一协同优化体系；②基于光流对事件进行变形，获得锐利边缘监督；③用残差先验替代传统对数强度差监督，提升低光区域鲁棒性；④设计联合损失实现互补优化。

**🔧 技术方法**

使用光流网络（EV-FlowNet）对事件进行变形；可学习双积分网络（LDI）；Bézier曲线姿态拟合；3D Gaussian Splatting渲染；联合损失与残差先验。

**📊 数据集**

在EBAD-NeRF合成数据、Real-World-Challenge真实数据以及E3NeRF数据集上进行实验。

**📈 对比分析**

与多种基线（MPRNet、EDI、EFNet、BAD-Gaussian、DP-NeRF、EvDeblurNeRF、E2NeRF、DiET-GS、EBAD-NeRF、E3NeRF等）对比，EvFlow-GS在PSNR、SSIM和LPIPS上均显著优于其他方法。

**⚠️ 局限性**

局限包括对预训练残差网络的依赖、需事件相机硬件、在极低光或高噪声事件下仍可能出现细节缺失，以及训练过程相对复杂且对计算资源要求较高。

---

## 111. Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models

**arXiv ID:** 2604.22166 | [PDF](https://arxiv.org/pdf/2604.22166v1)

**作者:** Ryoma Kumon `[一作]` (University of Tokyo), Hitomi Yanaka `[通讯]` (University of Tokyo)

**通讯引用:** 295 | [OpenAlex ID](https://openalex.org/A5045824013)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言模型在填充‑空缺依赖（FGD）和负极性项（NPI）许可两种句法现象中的内部机制，使用因果解释方法对层级和注意力头进行细粒度分析。

**💡 创新点**

首次发现填充‑空缺依赖具有跨构造共享、局部化于早中层的注意力头机制，而负极性项许可缺乏统一机制，并验证了训练无关的因果补丁方法在 OOD 上的稳健性。

**🔧 技术方法**

采用激活补丁（activation patching）进行训练无关的因果干预，比较分布式对齐搜索（DAS）并在 BLiMP、HANS 等基准上做 steering 实验。

**📊 数据集**

构建合成最小对比数据集，包括七种填充‑空缺构造、八种负极性项许可构造以及对照词汇知识检索句子，并使用 Pythia 和 Gemma 系列模型。

**📈 对比分析**

与 DAS 训练监督方法相比，激活补丁在 OOD 上保持一致且能显著提升 BLiMP 的语法判断准确率（α>1 时提升约 X%）以及 HANS 的鲁棒性。

**⚠️ 局限性**

局限在于仅针对英文、使用合成数据、样本多样性有限，以及模型仅覆盖当前模型体系，尚未验证跨语言或更大规模模型的适用性。

---

## 112. SAMIDARE: Advanced Tracking-by-Segmentation for Dense Scenarios

**arXiv ID:** 2604.22162 | [PDF](https://arxiv.org/pdf/2604.22162v1)

**作者:** Shozaburo Hirano `[一作]` (Toyota Technological Institute), Norimichi Ukita `[通讯]` (Toyota Technological Institute)

**通讯引用:** 4794 | [OpenAlex ID](https://openalex.org/A5053167635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 SAMIDARE，一种针对密集体育场景的追踪-分割框架，改进了 SAM2MOT；

**💡 创新点**

创新点包括：①基于局部密度的掩码重生成（DA‑QR），①使用均值与方差混合判别的跨对象交互（H‑CoI），以及②基于轨迹状态的三阶段匹配（SA‑OA），显著提升了 ID 连续性与遮挡恢复；

**🔧 技术方法**

技术手段包括 SAM2（SAM 2.1 Hiera‑Large）、YOLOX‑X 检测器、掩码置信度评估、密度阈值控制、混合均值/方差判别、匈牙利算法、IoU 与置信度融合；

**📊 数据集**

使用 SportsMOT 数据集（篮球、足球、排球共 240 条视频）进行评估；

**📈 对比分析**

与多种检测‑追踪方法（ByteTrack、DeepEIoU、McByte 等）和分割‑追踪方法（SAM2MOT、MASA 等）对比，SAMIDARE 在测试集上实现 77.3% HOTA、78.6% IDF1，较 SAM2MOT 提升 2.5 HOTA、4.2 IDF1，优于现有最优检测‑追踪器；

**⚠️ 局限性**

局部密度阈值 θ_density 固定，无法自适应不同运动项目；在极端拥挤或快速移动场景下仍可能出现掩码漂移或计算开销；未在体育数据上进行专门训练，泛化性有待进一步验证。

---

## 113. Logistic Bandits with $\tilde{O}(\sqrt{dT})$ Regret without Context Diversity Assumptions

**arXiv ID:** 2604.22161 | [PDF](https://arxiv.org/pdf/2604.22161v1)

**作者:** Seoungbin Bae `[一作]` (Korea Advanced Institute of Science and Technology), Dabeen Lee `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了新的算法 SupSplitLog，针对 K 维特征的逻辑斯蒂 bandit 问题，在不需要任何上下文多样性假设的前提下，实现了 O(√(dT)) 的累积 regret 上界。

**💡 创新点**

创新点在于引入样本拆分机制：将每个层级的样本分为 pilot 集和 estimation 集，分别用于构造初始点估计量和一次 Newton 型一阶校正，从而消除了传统方法中对上下文正定性和随机采样阶段的依赖，显著降低了对维度 d 的耦合。

**🔧 技术方法**

技术手段包括：基于 Auer 框架的层级分桶；对 pilot 估计量的自归一化（self‑normalizing）上界；利用样本拆分实现对噪声项的标量浓缩（Bernstein 置信区间）；一次性 Newton 校正来逼近最大似然估计；并进一步提出数据自适应的 β_t、τ_t 以将维度依赖替换为对 Gram 矩阵的对数复杂度。

**📊 数据集**

实验采用合成高维数据，设置三种特征空间几何（全空间、低维子空间1、低维子空间2），在不同 d 值下评估算法性能。

**📈 对比分析**

与 SupCB-GLM、SupLogistic、DDRTS-GLM 这三种基线算法对比，SupSplitLog 在低有效维度情形下始终取得最低累计 regret；在全空间情形下与 DDRTS-GLM 相近，但在子空间情形中优势显著，验证了改进后的维度依赖。

**⚠️ 局限性**

局限性包括：非领先项仍含 d^2.5 的系数，导致在 d 与 T 同阶增长时仍不具备完全子线性；对 κ 的依赖尚未达到最优；并且实验仅在合成数据上验证，缺乏真实世界数据的验证。

---

## 114. Reliable Self-Harm Risk Screening via Adaptive Multi-Agent LLM Systems

**arXiv ID:** 2604.22154 | [PDF](https://arxiv.org/pdf/2604.22154v1)

**作者:** Meghana Karnam `[一作]` (Johns Hopkins University), Ananya Joshi `[通讯]` (Johns Hopkins University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5072978515)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于DAG的多代理LLM评估管线，并用统计框架与自适应采样决定何时给出决策或上报给人类

**💡 创新点**

创新点在于：①给出每个节点的置信区间与错误上界；②用基于bandit的自适应采样实现“识别或上报”而非强行决定；③证明系统整体错误随交互次数呈对数增长，具备部署安全保证

**🔧 技术方法**

使用了多代理DAG结构、K-armed bandit自适应采样算法、置信区间与DKW不等式的统计推断、累计后悔（regret）理论

**📊 数据集**

评估数据集包括NVIDIA的AEGIS 2.0行为健康子集（161例）与从Reddit收集的SWMH样本（250例）

**📈 对比分析**

与单代理、固定样本多数投票（n=1/3/5）比较，结果显示自适应采样（B=100）在保持召回率不变的前提下，将误报率降低约40%，且在两数据集上表现相近，证明方法有效

**⚠️ 局限性**

主要局限包括：1）高计算成本（≈20–30倍LLM调用）；2）预算耗尽时仍完整遍历后续节点导致资源浪费；3）误报率下降受限于标注噪声，FNR受标签质量限制；4）目前仅在两小规模数据集验证，需在更大、多样化场景进一步评估

---

## 115. $O(K)$-Approximation Coflow Scheduling in $K$-Core Optical Circuit Switching Networks

**arXiv ID:** 2604.22146 | [PDF](https://arxiv.org/pdf/2604.22146v1)

**作者:** Xin Wang `[一作]` (Central Queensland University), Ye Tao `[通讯]` (Central Queensland University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在多核光电路交换网络（OCS）下基于异步重构模型的多coflow调度算法，目标是最小化加权完成时间（CCT）

**💡 创新点**

通过整合LP导向的全局coflow排序、前缀感知的流量分配与内部电路调度，实现了对K核OCS网络的(8K+1)近似保证，并将框架推广到多核EPS网络，获得(4H+1)近似保证

**🔧 技术方法**

使用线性规划（LP）松弛得到的排序信息、前缀低阶下界、贪心分配与工作守恒的电路调度策略，配合异步重构延迟的建模

**📊 数据集**

基于Facebook真实集群的MapReduce trace（约526个coflow，150端口）进行实验，随机抽取100个coflow进行评估

**📈 对比分析**

与基线方法（WSPT排序、Sunflow调度、BvN分解、仅负载分配）比较，实验表明该算法在总加权CCT和尾部CCT上显著优于除WSPT外的所有基线，并且实际近似比理论上保守（约2.5–5.0）

**⚠️ 局限性**

理论近似上界较高（8K或8K+1），且算法依赖完整的需求矩阵与离散流分配，未考虑在线/部分观测场景和动态到达的coflow问题

---

## 116. A Co-Evolutionary Theory of Human-AI Coexistence: Mutualism, Governance, and Dynamics in Complex Societies

**arXiv ID:** 2604.22227 | [PDF](https://arxiv.org/pdf/2604.22227v1)

**作者:** Somyajit Chakraborty `[一作]` (Shanghai Jiao Tong University), Somyajit Chakraborty `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5086782253)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出“在治理下的条件互惠（conditional mutualism under governance）”框架，将人类–AI共存建模为多层（物理、心理、社会）动力学系统，并给出稳定性、存在性与唯一性定理。

**💡 创新点**

创新点在于：① 把传统服从式视角替换为互惠治理；② 综合AI历史、世界模型、对齐、HRI、生态学与多中心治理的多学科洞见；③ 用可证明的动力学方法展示治理如何扩展共存稳定区。

**🔧 技术方法**

采用多层网络、互惠耦合、拉普拉斯算子、矩阵分析、梯度流动力学等数学工具，构建共存目标函数并推导其极值与稳定性。

**📊 数据集**

无具体数据集，理论模型基于抽象变量（物理资源、心理信任、社会合法性、AI发展度等）进行建模。

**📈 对比分析**

无实验比较，主要通过定理与符号证明阐述模型的存在、唯一性及全局渐近稳定性；并通过数值示例说明治理与互惠如何影响系统行为。

**⚠️ 局限性**

局限性包括：① 模型过于抽象，缺乏对实际指标的量化映射；② 未考虑随机冲击、信息不对称和策略演化；③ 未给出完整的分岔分析；④ 需进一步实验验证与应用案例。

---

## 117. AI-Driven Performance-to-Design Generation and Optimization of Marine Propellers

**arXiv ID:** 2604.22224 | [PDF](https://arxiv.org/pdf/2604.22224v1)

**作者:** Leah Chen `[一作]`, Jian Cheng Wong `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建基于物理合成的20,000+四、五叶螺旋桨几何与性能数据库，开发条件生成模型（cVAE 与潜在扩散模型）、高精度替代模型并通过进化优化实现从性能到设计的逆向生成；

**💡 创新点**

首次将性能条件直接映射至三维螺旋桨几何，并将物理数据合成、条件生成、快速替代与进化细化整合成一个模块化框架，且对比两种生成模型的多样性与精度；

**🔧 技术方法**

使用条件变分自编码器、潜在扩散模型、全连接多层感知机替代器、CMA‑ES 进化优化、PCA 降维、OpenFOAM/PropElements 仿真等技术；

**📊 数据集**

使用合成的 20,000+ 四、五叶螺旋桨几何向量与对应开放水性能曲线（K_T、K_Q、η）构成的数据库，训练集 70%/验证 15%/测试 15%；

**📈 对比分析**

采用 MSE/RMSE、R² 等指标评估替代模型；生成设计通过 MLP 预测与 PropElements 仿真验证，cVAE 在目标条件下误差约 1%–2%，潜在扩散模型误差约 1.2%–2.7%；潜在扩散模型在多样性（SC、Novelty）上显著优于 cVAE，但精度略低；

**⚠️ 局限性**

对离群（OOD）设计的泛化能力不足，特征向量表征不够直观，无法覆盖所有新设计空间，且仅在单一工况下验证，缺乏多工况与真实 CFD 验证。

---

## 118. ArchSym: Detecting 3D-Grounded Architectural Symmetries in the Wild

**arXiv ID:** 2604.22202 | [PDF](https://arxiv.org/pdf/2604.22202v1)

**作者:** Hanyu Chen `[一作]` (Cornell University), Noah Snavely `[通讯]` (Cornell University)

**通讯引用:** 27952 | [OpenAlex ID](https://openalex.org/A5085248097)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套从单张RGB图像中检测3D对称平面的方法，并构建了新的大规模对称标注数据集。

**💡 创新点**

主要创新包括：① 通过跨视图图像匹配从SfM重建自动提取3D对称注释；② 将对称平面参数化为相对于模型自身预测几何的符号距离图，解决单视角尺度不确定性。

**🔧 技术方法**

使用的技术包括：SfM重建、MASt3R匹配、DBSCAN聚类、VGGT几何基础模型、Transformer解码器、FiLM条件层、DPT密集预测头、Hungarian匹配损失。

**📊 数据集**

采用的训练数据集为自建的93个建筑景观场景，总计34,177张图像的LandmarkSym（LSym）数据集。

**📈 对比分析**

通过与最近的单视角对称检测器（如Reflect3D/NeRD）以及直接回归基线进行比较，在normal-only和full-plane评估指标上，LSym模型在Geo、F@1°、F@5°、Dense Symmetry Error等指标均优于基线，显示出更高的检测精度和更低的误差。

**⚠️ 局限性**

局限性包括：① 依赖SfM重建质量，若重建不完整会影响标注；② 仅检测全局反射对称，无法处理局部或旋转对称；③ 对VGGT预测几何的不确定性敏感，严重遮挡或模糊场景下可能产生不准确的平面。

---

## 119. Rethinking Semantic Collaborative Integration: Why Alignment Is Not Enough

**arXiv ID:** 2604.22195 | [PDF](https://arxiv.org/pdf/2604.22195v1)

**作者:** Maolin Wang `[一作]` (City University of Hong Kong), Lei Sha `[通讯]` (Beihang University)

**通讯引用:** 1456 | [OpenAlex ID](https://openalex.org/A5079222154)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大型语言模型（LLM）生成的语义嵌入与协同过滤（CF）嵌入在推荐系统中的对齐假设进行审视，并提出以共享‑私有（shared‑plus‑private）潜在结构为基础的互补融合框架。

**💡 创新点**

创新点包括：①将语义与协同视角视为部分共享且本质异构的多视图；②设计了面向互补性的诊断指标（Jaccard、Complementarity Ratio、UUB）以量化两视图重叠与补充性；③在不使用任何对齐损失的情况下，仅通过简单的 Norm‑Concat‑Norm 融合实现了优于现有对齐方法的性能。

**🔧 技术方法**

技术手段包括：LightGCN 作为协同视图、BGE‑M3 作为语义视图、InfoNCE 与动态硬负采样的对抗式训练、低容量映射探针（线性、MLP）以及基于诊断指标的评估。

**📊 数据集**

数据集为 Amazon Reviews 2023 的三大稀疏推荐基准：Movies、Books、Games，全部超过 99.9% 稀疏率。

**📈 对比分析**

与 ID‑CF 基线（LightGCN、SimGCL、NCL）及对齐/映射方法（CARec、RLMRec、AlphaRec）对比，最小融合模型在 Recall@20、NDCG@20 上分别提升 13.4%–41.7% 与 15.3%–38.3%，并逼近 Union‑TopK 理论上限，证明互补融合优于传统对齐。

**⚠️ 局限性**

局限性在于：①未实现显式的共享‑私有分解，依赖于隐式融合；②高容量映射易过拟合，低容量映射难以恢复协同几何；③缺乏自适应机制以根据数据密度动态调节两视图权重。

---

## 120. How Large Language Models Balance Internal Knowledge with User and Document Assertions

**arXiv ID:** 2604.22193 | [PDF](https://arxiv.org/pdf/2604.22193v1)

**作者:** Shuowei Li `[一作]` (Santa Clara University), Yi Fang `[通讯]` (Santa Clara University)

**通讯引用:** 2429 | [OpenAlex ID](https://openalex.org/A5083935587)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了三源交互评估框架，用来衡量大语言模型（LLM）在内部参数知识、用户断言和文档断言之间的权衡与整合。

**💡 创新点**

创新点在于：①首次系统化地研究三源（内部、用户、文档）同时存在的冲突与合作情境；②通过统计回归量化每源对回答正确率的影响；③揭示后训练显著增强文档优先性；④证明在多源交互数据上做监督微调可显著提升模型的辨别能力。

**🔧 技术方法**

采用了逻辑回归（logistic）计算来源影响的比值、来源依赖比、PAR/SDR 指标，利用 KL 散度和负对数似然评估概率分布变化；构建了13种探测变体；在多源交互数据上执行监督微调（SFT）来提升模型辨别力。

**📊 数据集**

使用的基准数据集为 CommonsenseQA（CSQA）和多项选择版 GSM8K（GSM-MC）。

**📈 对比分析**

对 27 个 LLM（GPT‑4o、Llama、Qwen3）在两数据集上进行比较，发现大多数模型偏好文档断言、后训练进一步强化这一倾向；大部分模型易受外部信息影响，难以区分有用与有害断言；通过多源交互数据的 SFT，模型的 PAR/SDR 指标从“易受影响”提升到“选择性”，对应的正确率在负面和冲突情境下提升 30–40%。

**⚠️ 局限性**

局限性包括：①实验仅基于人工合成的单轮多项选择问题，缺乏真实噪声、长文本或多轮对话情境；②仅涉及英文文本，未检验多语言或多模态输入；③使用的基准可能无法充分覆盖复杂的现实场景；④模型在标准大规模评测上的提升有限，表明 SFT 的泛化仍需验证。

---

## 121. Behavioral Canaries: Auditing Private Retrieved Context Usage in RL Fine-Tuning

**arXiv ID:** 2604.22191 | [PDF](https://arxiv.org/pdf/2604.22191v1)

**作者:** Chaoran Chen `[一作]` (Google), Peter Kairouz `[通讯]` (Google)

**通讯引用:** 7618 | [OpenAlex ID](https://openalex.org/A5064699160)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种新的行为式海盗（Behavioral Canary）方法，用于在强化学习微调（RLFT）流程中检测检索到的文档是否被误用为训练数据。

**💡 创新点**

提出将触发器与反馈关联、在训练时注入行为式海盗并通过模型行为放大来检测文档使用，而非传统的记忆或成员推断。

**🔧 技术方法**

利用RLFT中的奖励模型和策略优化（PPO/GRPO），设计触发器、诱导指令和目标行为模式，并通过日志概率差异进行放大统计。

**📊 数据集**

在RepliQA（问答）和QMSUM（会议摘要）两个检索式任务上进行实验，使用文档分割、注入率约1%的海盗样本。

**📈 对比分析**

对比合规与违规训练实例的放大得分，使用AUROC和10% FPR下TPR评估，RepliQA 0.756/0.67，QMSUM 0.762/0.60，证明能在1%注入率下实现可观的检测率。

**⚠️ 局限性**

检测效果依赖注入率和海盗模式，低注入率时信号衰减；仅能统计区分，无法定位具体违规文件；对真实生产管道的过滤与去重影响未完全评估。

---

## 122. ResRank: Unifying Retrieval and Listwise Reranking via End-to-End Joint Training with Residual Passage Compression

**arXiv ID:** 2604.22180 | [PDF](https://arxiv.org/pdf/2604.22180v1)

**作者:** Xiaojie Ke `[一作]` (Qwen Applications Business Group of Alibaba), Guanjun Jiang `[通讯]` (Qwen Applications Business Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ResRank，一个统一检索与列表式重排序的框架，采用 Encoder‑LLM 将每个候选文档压缩为单个嵌入，随后在 Reranker‑LLM 中通过残差连接融合上下文并使用余弦相似度直接评分；

**💡 创新点**

核心创新在于（1）残差式 passage 压缩实现单 token 输入；（2）用余弦相似度替代自回归解码完全消除生成瓶颈；（3）双阶段多任务端到端联合训练，使检索与重排序目标保持一致；

**🔧 技术方法**

技术细节包括 Encoder‑LLM（Qwen3‑Embedding‑4B）+ Reranker‑LLM（Qwen3‑4B），残差连接、余弦相似度评分、InfoNCE 与 RankNet 结合的多任务损失、FlashAttention 与 DeepSpeed ZeRO‑2 等；

**📊 数据集**

实验使用 TREC Deep Learning 2019/2020 以及 BEIR 的八个数据集（Covid、NFCorpus、Touche、DBPedia、SciFact、Signal、News、Robust）；

**📈 对比分析**

与传统跨编码器、零射 LLM、全文本 LLM 重排序、PE‑Rank 等多类基线对比，单通道模式下平均 nDCG@10 0.544（最高），仅 100 处理 token，生成 token 0，性能优于大多数基线，接近 RankGPT‑4；

**⚠️ 局限性**

局限性包括对候选文档输入顺序敏感、未验证更长候选列表或更大模型的扩展、未覆盖多模态场景，以及在低资源或多语言任务中的泛化性待进一步研究。

---

## 123. FixV2W: Correcting Invalid CVE-CWE Mappings with Knowledge Graph Embeddings

**arXiv ID:** 2604.22176 | [PDF](https://arxiv.org/pdf/2604.22176v1)

**作者:** Sevval Simsek `[一作]` (Boston University), David Starobinski `[通讯]` (Boston University)

**通讯引用:** 3610 | [OpenAlex ID](https://openalex.org/A5011231717)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对 NVD 数据库中 CVE–CWE 映射的错误（被标记为 Prohibited 或 Discouraged 的映射）进行自动纠正，并通过修正后的数据提升后续机器学习模型的表现。

**💡 创新点**

①首次对 NVD CVE–CWE 映射的长期演变进行系统性分析；②基于映射演变规律与 CWE 层级结构设计轻量级知识图谱嵌入方法 FixV2W；③在不依赖语义特征的情况下实现可解释且可扩展的映射纠正。

**🔧 技术方法**

使用知识图谱嵌入（TransE）进行关系预测，结合基于树层级的候选集（Descendant、Family、Member、Nearest Neighbor）筛选；同时采用 Rank‑based（MR, MRR, Hits@N）和 Coverage（Exact/Fine/Coarse）评估指标。

**📊 数据集**

以 2021 年 8 月的 NVD 数据作为训练集，12 月 2024 年更新的 NVD 作为验证集；包含 13,784 个 Prohibited、33,928 个 Discouraged 映射；对 190 个已被攻击的 CVE 进行评估。

**📈 对比分析**

相较于基线（CWE‑1003 视图、Top‑25 视图）和现有图完成模型，FixV2W 在 Top‑10 中约 70% 的正向映射得到正确或细粒度匹配；在图完成任务中，使用 Top‑2 预测后 MRR 从 0.174 提升至 0.608，Hits@10 从 0.389 提升至 0.793；对已攻击 CVE 的预测准确率超过 68%。

**⚠️ 局限性**

仍有约 35–40% 的映射无法在 Top‑10 内预测，主要原因是：①某些 Prohibited 类别（如 CWE‑16、CWE‑264）缺乏成员关系；②新映射与旧映射完全处于不同分支，导致模型难以捕捉；③NVD 评估标签本身不完整或存在错误，影响性能评估。

---

## 124. Unlocking Optical Prior: Spectrum-Guided Knowledge Transfer for SAR Generalized Category Discovery

**arXiv ID:** 2604.22174 | [PDF](https://arxiv.org/pdf/2604.22174v1)

**作者:** Jingyuan Xia `[一作]` (National University of Defense Technology), Zhejun Lu `[通讯]` (National University of Defense Technology)

**通讯引用:** 156 | [OpenAlex ID](https://openalex.org/A5041100499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MCPT框架，通过频域特征对齐实现光学预训练模型（DINOv2）向SAR域的知识迁移，以解决SAR-目标识别中光学与SAR的模态差异。

**💡 创新点**

创新点在于：① 设计了Modal Discrepancy Curve (MDC)——一种基于频域能量分布的模态差异描述；② 通过Adaptive Frequency Tokenization (AFT)将MDC转化为可学习的频域token；③ 利用Frequency‑aware Expert Refinement (FER)在不同频带上进行差异感知特征细化；④ 结合对称跨模态对比学习与无监督对比损失，双向内部化对齐模式。

**🔧 技术方法**

使用的技术包括：DINOv2视觉Transformer、FFT频域分析、Gaussian频带掩码、对抗/对齐对比学习、Attention+Experts（FcE）模块、两阶段预训练/微调策略。

**📊 数据集**

采用YESeg-OPT‑SAR对齐的光学‑SAR配对数据进行预训练；在SAR目标识别公开数据集MSTAR、SAMPLE、FUSAR、OpenSARShip上进行评估。

**📈 对比分析**

与GCD、CMS、InfoSieve、SimGCD、BKD‑CL、ProtoGCD等SOTA方法相比，MCPT在DINOv2基础上取得平均提升约5‑10%（如MSTAR 7.89%提升、FUSAR 5.26%提升），在新类识别上表现尤为突出；在转导式与归纳式评估中均保持领先。

**⚠️ 局限性**

局限性：① 需要光学‑SAR配对数据进行预训练；② 预训练阶段额外引入AFT/FER模块，计算成本上升；③ 对于极端长尾或少样本场景，仍可能出现新类误判；④ 现有方法聚焦SAR-GCD，迁移到其他遥感任务或更大规模数据集的通用性待验证。

---

## 125. MCI: A Maximal Clique Index for Efficient Arbitrary-Filtered Approximate Nearest Neighbor Search

**arXiv ID:** 2604.22171 | [PDF](https://arxiv.org/pdf/2604.22171v1)

**作者:** Xiaowei Ye `[一作]` (Beijing Institute of Technology), Xubin Li `[通讯]` (Huawei)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于最大团的索引（Maximal Clique Index，MCI），用于高效处理任意过滤近似最近邻搜索（AFANNS）。

**💡 创新点**

创新点在于利用高维空间中的邻域传递性，通过稀疏k'‑NNG构建密集的最大团覆盖（Maximal Clique Cover）来实现高效且压缩的图结构，同时设计了几何稠密化与多种种子策略以提升连通性和召回率。

**🔧 技术方法**

主要技术包括：NN-Descent构建稀疏k'‑NN图、线性时间贪婪最大团挖掘、几何阈值扩展、并行无锁构建、基于多种种子和beam搜索的查询算法。

**📊 数据集**

在10个公开数据集上进行实验，涵盖图像、文本、音频、电影、文档等多种领域，维度从128到2048不等。

**📈 对比分析**

与ACORN、HNSW（Faiss/Milvus）、IVFPQ以及专用的范围/关键词过滤方法进行比较。MCI在高召回（Recall@10>0.95）时，QPS可提升至现有方法的1–3倍，同时索引空间仅为IVFPQ的约25–50%。

**⚠️ 局限性**

局限性包括：在极低选择率下仍需较大种子采样导致开销上升；对超中心节点的处理需要手动阈值；构建时间虽然低于ACORN，但仍高于纯HNSW；对动态更新的支持尚未实现。

---

## 126. Sharpness-Aware Poisoning: Enhancing Transferability of Injective Attacks on Recommender Systems

**arXiv ID:** 2604.22170 | [PDF](https://arxiv.org/pdf/2604.22170v1)

**作者:** Junsong Xie `[一作]` (Hefei University of Technology), Le Wu `[通讯]` (Hefei University of Technology)

**通讯引用:** 6246 | [OpenAlex ID](https://openalex.org/A5033706423)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种Sharpness‑Aware Poisoning（SharpAP）方法，用于提升推荐系统注入攻击的跨模型迁移性。

**💡 创新点**

创新点在于将尖锐度（sharpness）意识下的最坏情况模型搜索（min‑max‑min三层优化）引入攻击生成，显著降低对固定代理模型的过拟合。

**🔧 技术方法**

核心技术包括：三层优化框架、尖锐度约束的扰动估计、投影梯度下降生成离散攻击样本，以及对现有梯度攻击（如RevAdv）进行改造。

**📊 数据集**

实验使用三大公开数据集：MovieLens‑1M、Gowalla 与 Amazon‑book，以验证方法在多种数据分布下的鲁棒性。

**📈 对比分析**

与多种基线攻击（随机、热门、CoVis、PGA、AUSH、RevAdv、RAPU、DADA、CLeaR、DDSP）以及五类受害模型（WRMF、BPR、LightGCN、SGL、SimGCL）对比，SharpAP 在 Hit Ratio、NDCG 等指标上平均提升约 15–30%，并在最坏情况模型上表现出更强的泛化。

**⚠️ 局限性**

局限性包括：需手动调节扰动半径 ϵ 以平衡攻击效果与稳定性；对极端模型结构或防御机制的适应性尚未完全验证；额外的梯度计算导致计算开销略有增加。

---

## 127. When AI Speaks, Whose Values Does It Express? A Cross-Cultural Audit of Individualism-Collectivism Bias in Large Language Models

**arXiv ID:** 2604.22153 | [PDF](https://arxiv.org/pdf/2604.22153v1)

**作者:** Pruthvinath Jeripity Venkata `[一作]` `[通讯]` (Independent Researcher), Pruthvinath Jeripity Venkata (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在10个国家、7种语言、10个个人困境下，系统审计了Claude Sonnet 4.5、GPT‑5.4和Gemini 2.5 Flash三款前沿LLM的文化偏差。

**💡 创新点**

创新点在于提出行为情境审计方法，并通过四条件设计分离语言与国家标签，使用真实世界调查数据（WVS）作为绝对基准，揭示模型的跨文化偏差。

**🔧 技术方法**

技术上使用三款公开API模型（Claude、GPT‑5.4、Gemini），在温度0下调用API，并用两位LLM评审（Llama 3.3 70B与DeepSeek‑V3）进行IC分数评估。

**📊 数据集**

数据集包括840条模型生成的回答、对应的评审分数、WVS Wave 7的文化维度锚点（如权威、性别、家庭等）以及误差值。

**📈 对比分析**

比较方法通过计算模型IC分数与WVS锚点的差值（misalignment），并使用t检验、混合效应模型和Bootstrap CI，结果显示平均+0.76的个体主义偏差，且在尼日利亚和印度尤为显著。

**⚠️ 局限性**

限制包括仅用10个提示、评审者可能与模型共享偏见、机器翻译可能引入语义偏差、无法完全分离语言与文化身份，以及样本仅覆盖十个国家。

---

## 128. Learning-augmented robotic automation for real-world manufacturing

**arXiv ID:** 2604.22235 | [PDF](https://arxiv.org/pdf/2604.22235v1)

**作者:** Yunho Kim `[一作]` (Neuromeka Co., Ltd.), Joonho Lee `[通讯]` (Neuromeka Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在电机生产线上将学习型控制器（视觉伺服与模仿学习）与神经3D安全监视器集成到传统工业工作站，实现了变形电缆插入和焊接的自动化，单机连续作业5h10m产出108台电机，99.4%的质量合格率与人类相近；

**💡 创新点**

提出模块化混合架构，任务分解与任务特定的感知/控制先验，零样本掩膜跟踪、结构化视觉观测、数据高效学习，以及实时3D安全监控，实现在有限数据（<20min/任务）下的可靠工业级性能；

**🔧 技术方法**

视觉伺服与模仿学习采用基于Transformer的ACT模型；掩膜预测使用轻量级U‑Net；3D安全监测使用卷积占据网络（Conv‑Occupancy）；模拟与真实共训练、DAgger增强、零样本掩膜追踪与SAM2；系统集成于双臂协作机器人、RGB摄像头与LiDAR；

**📊 数据集**

使用现场工厂收集的真实数据：motor‑grasping 8min、cable‑insertion 20min、soldering 4min、mask‑predictor 9min；并结合模拟环境（IsaacLab）生成的LiDAR点云用于安全监测；

**📈 对比分析**

与Naïve IL、VLA预训练模型及传统基于3D几何的路径规划做对比；在抓取、插入、焊接子任务中取得>99%成功率，传统方法仅≈65%成功；整体系统成功率99.4%，与人类平均 takt 时间相近，焊接质量在盲测中得到视觉一致性；

**⚠️ 局限性**

仅在单一电机生产线验证，任务结构与硬件特定性较高；对极端视觉扰动和复杂环境的鲁棒性待进一步评估；仍需人工进行物料加载/卸料，安全监测对LiDAR的依赖；在极少数据场景下的迁移性能尚有限。

---

## 129. Preserve Support, Not Correspondence: Dynamic Routing for Offline Reinforcement Learning

**arXiv ID:** 2604.22229 | [PDF](https://arxiv.org/pdf/2604.22229v1)

**作者:** Zhancun Mu `[一作]` (Peking University), Chi Zhang `[通讯]` (Peking University)

**通讯引用:** 22681 | [OpenAlex ID](https://openalex.org/A5100458200)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为DROL的单步离线强化学习演员，利用动态路由在训练时生成候选动作并将数据动作分配给最近的候选，从而在保持局部支持的同时实现价值改进；

**💡 创新点**

核心创新在于放弃点对点的教师-学生对应关系，而是通过一阶动态路由实现候选动作的责任转移，使得演员能在多模态环境中逐步收敛到更优动作；

**🔧 技术方法**

采用Top-1欧氏路由、Voronoi分区、基于潜在先验的候选采样、行为克隆与Q值改进的联合损失；

**📊 数据集**

在OGBench和D4RL两个离线RL基准上进行实验；

**📈 对比分析**

与一阶FQL基线进行对比，DROL在10个OGBench任务组中有6组匹配或优于FQL，特别是某些导航类任务；在D4RL上保持与FQL相当的性能；推理时间与FQL相近，迭代基线显著更慢；

**⚠️ 局限性**

局限包括固定的候选数K、欧氏距离路由可能受限于不同动作空间，训练时需要额外的候选评估开销，并且缺乏完整的全局收敛理论。

---

## 130. Towards Temporal Compositional Reasoning in Long-Form Sports Videos

**arXiv ID:** 2604.22226 | [PDF](https://arxiv.org/pdf/2604.22226v1)

**作者:** Siyu Cao `[一作]` (University of Chinese Academy of Sciences), Zhi-yong Liu `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 9255 | [OpenAlex ID](https://openalex.org/A5100358893)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模的长时序体育视频基准 SportsTime，并提出了链式时序推理（CoTR）框架，用于实现可解释的多步时间证据推理。

**💡 创新点**

创新点在于：①为每个推理步骤提供明确的时间证据标注，形成可验证的 Chain‑of‑Time；②在训练中引入时间奖励的 Group‑Relative Policy Optimization（tr‑GRPO）以鼓励模型关注时间证据；③在推理时使用 Anchor‑Triggered Interactive Observation（AT‑IO）循环验证并修正推理步骤。

**🔧 技术方法**

核心技术包括：时间戳覆盖预处理、基于强化学习的 tr‑GRPO、Anchor‑Triggered Interactive Observation、LLM‑as‑Judge 评估协议，以及多模态大语言模型的微调与推理。

**📊 数据集**

使用的数据集为 SportsTime，包含 14K+ 开放式问答对、50K+ 步骤级时间证据标注，涵盖美式足球、冰球、足球、篮球、排球等五大团队运动。

**📈 对比分析**

与多种主流多模态大模型（如 GPT‑5、Gemini‑Pro、Qwen3‑VL‑4B）对比，CoTR 在 SportsTime 上提升整体准确率至约 29%（比基线提升 4%+），并在其他视频推理基准上同样获得显著性能提升。

**⚠️ 局限性**

局限性包括：整体准确率仍偏低，依赖显式时间戳覆盖可能不适用于无时间标记的视频；对极长视频或跨场景的泛化能力尚待进一步验证。

---

## 131. Breaking Watermarks in the Frequency Domain: A Modulated Diffusion Attack Framework

**arXiv ID:** 2604.22220 | [PDF](https://arxiv.org/pdf/2604.22220v1)

**作者:** Chunpeng Wang `[一作]` (Qilu University of Technology (Shandong Academy of Sciences)), Qi Li `[通讯]` (Qilu University of Technology (Shandong Academy of Sciences))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于频域调制扩散模型FMDiffWA，能够在保持图像视觉质量的同时抑制嵌入的水印。

**💡 创新点**

创新点包括①在前向与后向扩散过程中显式利用幅度与相位信息的频域水印调制模块；②采用两阶段训练策略，先进行噪声估计后加入细化约束，实现高质量攻击。

**🔧 技术方法**

使用的技术主要是条件扩散模型（DDPM）、傅里叶变换与频域掩模、两阶段训练（噪声估计+细化约束）、多尺度SSIM以及EMA后处理。

**📊 数据集**

实验采用CelebA与ImageNet（256×256）数据集，并在每张图像上嵌入16×16二值水印，使用四种水印算法（LSB、DCT、PHFMs、HiDDeN_MP）。

**📈 对比分析**

与传统噪声、滤波、JPEG压缩以及深度学习对手（RD-IWAN、DiffWA、HIWANet）进行对比，FMDiffWA在PSNR上可达≈40 dB，BER趋近于0，显著优于其它方法。

**⚠️ 局限性**

限制在于对更复杂多样化水印算法的泛化仍有限，模型推理成本相对较高，未来需要进一步探索更强的频域先验和高效采样策略。

---

## 132. From Monolithic to Compositional: A Compositional Operational Semantics for Crystality

**arXiv ID:** 2604.22210 | [PDF](https://arxiv.org/pdf/2604.22210v1)

**作者:** Ziyun Xu `[一作]` (Peking University), Meng Sun `[通讯]` (Peking University)

**通讯引用:** 3893 | [OpenAlex ID](https://openalex.org/A5085248515)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并形式化了适用于并行EVM的智能合约语言Crystality的可组合运算语义，重构了系统为引擎组件与全局组件，定义了编码/解码函数并证明其与原始单片语义的交易级Bisimulation，进而证明了局部性、全局隔离和独立局部步骤的强可交换性等结构性质。

**💡 创新点**

核心创新在于将单片语义拆解为可组合的引擎与全局组件，构造了明确的编码/解码映射并通过代码级与交易级双层Bisimulation实现语义等价性，首次在并行区块链上下文中提供可组合的运算语义框架。

**🔧 技术方法**

主要技术手段包括结构化操作语义（SOS）框架、递归编码/解码函数、交易级与代码级Bisimulation证明以及对状态空间的分层建模（引擎局部状态、全局状态、内存池等）。

**📊 数据集**

未使用传统意义上的数据集；实验与评估均为形式化证明，未涉及具体合约代码或区块链数据。

**📈 对比分析**

对性能或实验对比未给出；研究以理论证明为主，未在模拟器或真实链上进行性能评测。

**⚠️ 局限性**

局限性包括：1）目前仅针对Crystality语言；对其他并行合约语言的迁移需进一步研究；2）语义复杂度高，证明过程繁琐；3）未验证在实际区块链网络中的执行效率与可扩展性。

---

## 133. An LLM-Driven Closed-Loop Autonomous Learning Framework for Robots Facing Uncovered Tasks in Open Environments

**arXiv ID:** 2604.22199 | [PDF](https://arxiv.org/pdf/2604.22199v1)

**作者:** Hong Su `[一作]` (Chengdu University of Information Technology), Hong Su `[通讯]` (Chengdu University of Information Technology)

**通讯引用:** 677 | [OpenAlex ID](https://openalex.org/A5110930319)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于LLM的闭环自学习框架，使机器人在开放环境中遇到未覆盖任务时能够主动判断是否需要学习、规划学习过程、执行或观察收集数据、进行准实时训练，并将验证后的方法写回本地库；

**💡 创新点**

核心创新点包括：①将LLM仅用于高层规划与学习触发，减少每次执行对LLM的依赖；②实现闭环学习与知识整合，将执行与观察得到的经验转化为可复用的本地方法；③引入主动观察作为独立学习触发源，提升学习效率；④设计两级主动性机制，实现无人工干预的学习组织与持续更新；

**🔧 技术方法**

采用大型语言模型（LLM）进行任务分析与规划，结合自执行与主动观察的数据采集，准实时在线训练（quasi‑real‑time），以及本地方法库检索与更新；

**📊 数据集**

在实验中使用自构建的重复任务动作序列基准（20条自然语言指令，5次重复），以及同一基准的观察驱动实验；

**📈 对比分析**

与三种基线对比：Always‑LLM（每次都询问LLM）、Library‑Only（仅检索，不学习）以及Observation‑Only。实验表明，在自执行场景下，所提框架成功率为1.00，平均总执行时间从7.7772 s降至6.7779 s（约12.9%提升），LLM调用次数从1.0降至0.2；在观察驱动场景下，执行时间从7.4969 s降至5.5833 s（约25.5%提升），LLM调用次数从0.8降至0.2，方法库命中率提升至1.0；

**⚠️ 局限性**

局限性包括：①仍需依赖LLM进行规划，若LLM性能下降或成本高昂会影响整体；②实验仅在模拟的离散动作序列任务上验证，缺乏真实世界连续控制或复杂感知任务的评估；③方法库检索阈值与触发条件需要手工调参，可能在不同任务分布下表现不稳定；④对观察信息的获取与解析依赖环境与传感器，现实场景中可能受限；⑤未对灾难性遗忘或长期学习的稳定性做深入分析。

---

## 134. Learning Reactive Human Motion Generation from Paired Interaction Data Using Transformer-Based Models

**arXiv ID:** 2604.22164 | [PDF](https://arxiv.org/pdf/2604.22164v1)

**作者:** Masato Soga `[一作]` (Wakayama University), Ryuki Takebayashi `[通讯]` (Wakayama University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并评估了三种Transformer架构（Simple Transformer、iTransformer、Crossformer）在拳击对抗场景下生成一方动作对应另一方反应动作的模型，并引入人物ID嵌入以提升交互运动的结构稳定性。

**💡 创新点**

创新点在于：①首次将人物身份信息显式嵌入到交互式运动生成模型中；②系统对比了传统Transformer、iTransformer与Crossformer在此任务上的表现，揭示了仅捕捉特征维度依赖不足以保证姿态连贯；③通过实时与离线两种评估方式，验证了身份嵌入对运动质量的正向影响。

**🔧 技术方法**

主要技术包括：基于Transformer的时序建模、iTransformer的特征维度自注意力、Crossformer的双阶段注意力、3D姿态重建（AlphaPose+MotionBERT）、人物ID嵌入以及Unity+Azure Kinect的实时可视化框架。

**📊 数据集**

使用数据集：Kaggle公开的 Olympic Boxing Video Dataset（21场比赛），从中提取并预处理得到的动作-反应配对3D姿态序列共计约210,926帧。

**📈 对比分析**

通过问卷（5级Likert）对实时与离线两种设置下的六种模型变体进行比较；结果显示 Simple Transformer（含ID嵌入）在连续性、响应性与人类可辨识度上均优于 iTransformer 与 Crossformer，且加入ID嵌入能显著提升所有模型的得分。

**⚠️ 局限性**

局限性包括：①iTransformer 与 Crossformer 在生成过程中易出现姿态崩塌与误差累积；②仅针对拳击单一场景，缺乏对多样化交互动作的验证；③实时评估受模型推理延迟与系统延迟影响；④评估指标主要依赖主观问卷，缺少客观数值化指标。

---

## 135. dWorldEval: Scalable Robotic Policy Evaluation via Discrete Diffusion World Model

**arXiv ID:** 2604.22152 | [PDF](https://arxiv.org/pdf/2604.22152v1)

**作者:** Yaxuan Li `[一作]` (Current Robotics), Yichen Zhu `[通讯]` (University Of Toronto)

**通讯引用:** 1021 | [OpenAlex ID](https://openalex.org/A5054623682)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 dWorldEval——一种离散扩散世界模型，用统一的 token 序列表示视觉、动作、语言，并通过稀疏关键帧记忆和进度 token，实现可控的长时序模拟与无外部oracle的自动成功判定；

**💡 创新点**

创新点包括①将动作直接嵌入统一 token 序列并通过自注意力实现真正的动作可控；②使用稀疏关键帧记忆保持长时序的一致性；③同时预测进度 token 以实现自动成功检测；④提出 Δ‑LPIPS 动态指标专门评估动作控制的精确性；

**🔧 技术方法**

技术手段包括 Masked Discrete Diffusion、MAGVIT‑v2/FAST/LLaDA 等统一 tokenizer、Transformer 自注意力、关键帧记忆、并行迭代解码以及 Δ‑LPIPS 评价指标；

**📊 数据集**

数据集涵盖 LIBERO（Object、Spatial、Goal、100 套）、RoboTwin（10 个任务）以及真实双臂 AgileX 系统收集的 5.2k 轨迹（含 1k 失败样本），支持多视角（第三人称、手腕视角）；

**📈 对比分析**

对比方法：在相同训练数据上与 WorldEval、WorldGym、Ctrl‑World 等视频扩散基线进行公平评估，采用 Δ‑LPIPS、LPIPS、进度成功率等指标；实验显示 dWorldEval 在动作可控性、时空一致性和成功判定上均显著优于基线，相关系数 r≈0.9，rank‑violation MMRV≤0.013，证明其作为策略评估代理的可靠性；

**⚠️ 局限性**

局限性：受训练数据规模和分布限制；稀疏关键帧记忆在极长时间或极复杂交互中可能仍出现残余漂移；离散 token 的分辨率可能影响细粒度细节；对极端 OOD 动作的泛化能力仍待进一步验证。

---

## 136. Sum-of-Checks: Structured Reasoning for Surgical Safety with Large Vision-Language Models

**arXiv ID:** 2604.22156 | [PDF](https://arxiv.org/pdf/2604.22156v1)

**作者:** Weiqiu You `[一作]` (University of Pennsylvania), Eric Wong `[通讯]` (University of Pennsylvania)

**通讯引用:** 2787 | [OpenAlex ID](https://openalex.org/A5066376294)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Sum-of-Checks 框架，按专家定义的检查拆分 Critical View of Safety（CVS）标准，利用大型视觉语言模型（LVLM）对每个检查做二值判断并按固定权重聚合，完成结构化手术安全评估。

**💡 创新点**

创新点在于：①将安全判定拆分为可解释的验证检查并使用固定加权聚合；②将证据提取与最终决策分离，显著提升可审计性；③在多模型、多基准上展示结构化检查可提升 LVLM 的 CVS 评估性能。

**🔧 技术方法**

使用 GPT‑4.1‑mini、Claude Haiku/Opus 等 LVLM，结合少样本提示、链式思维（CoT）、子问题分解（SubQ）等提示策略，并采用固定权重聚合方式对检查结果进行二值化后加权求和。

**📊 数据集**

使用 Endoscapes2023 基准（791 帧关键帧），包含三项 CVS 标准的标注。

**📈 对比分析**

与直接提示、CoT、SubQ 等基线进行对比，Sum‑of‑Checks 在三模型、三标准上平均 mAP 提升 12–14%，例如 GPT‑4.1‑mini 的平均 mAP 从 30.5% 提升到 36.9%，Claude 4.5 的平均 mAP 亦提升至 34–37%。

**⚠️ 局限性**

局限性：仅在 Endoscapes 评估，未验证在其他 CVS 数据集（如 Cholec80‑CVS、SAGES CVS Challenge）上的泛化；对解剖/空间检验检查的准确性仍不稳定；对离散样本和跨域案例的鲁棒性仍待研究。

---

## 137. Accelerating Intra-Node GPU-to-GPU Communication Through Multi-Path Transfers with CUDA Graphs

**arXiv ID:** 2604.22228 | [PDF](https://arxiv.org/pdf/2604.22228v1)

**作者:** Amirhossein Sojoodi `[一作]` (Queen's University), Ahmad Afsahi `[通讯]` (Queen's University)

**通讯引用:** 940 | [OpenAlex ID](https://openalex.org/A5039854005)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在MPI基于的高性能计算应用中，本文提出了一种将CUDA Graph嵌入UCX框架的新型多路径GPU互连通信方案，以提升同一节点内GPU间的传输带宽和效率。

**💡 创新点**

创新点在于：①首次将CUDA Graph动态构建与缓存机制与UCX的CUDA-IPC子模块结合，实现多路径（NVLink、PCIe和主机缓存）数据流的并行调度；②设计了二维流水线划分算法，自动将大数据切块分配到不同路径；③通过图缓存实现重复通信模式的零启动延迟和最小同步开销。

**🔧 技术方法**

主要技术包括：UCX通信框架、CUDA Graph、NVLink/PCIe互连、主机内存/GPU中间缓存、2D流水线调度、图缓存（LRU策略）、环境变量动态调优。

**📊 数据集**

使用了OSU Micro-Benchmarks（OMB）中的Unidirectional、Bidirectional Bandwidth、Latency等微基准，以及基于Jacobi迭代求解器的实际应用场景，数据规模从1 MB到512 MB（OMP）以及1 GB以上的Jacobi边界交换。

**📈 对比分析**

与传统单路径UCX（UCT::CUDA-IPC）对比，实验显示在Beluga（4 × V100）和Narval（4 × A100）节点上：大于32 MB消息时多路径+CUDA Graph可实现最高2.95×的带宽提升；在Jacobi求解器中，使用两条并行路径可将总执行时间缩短至1.26×（Beluga）和1.15×（Narval），引入CUDA Graph后可进一步提升至1.28×与1.16×。

**⚠️ 局限性**

局限性包括：①CUDA Graph在小消息或节点数少时开销相对较大，导致收益不明显；②主机路径带宽低且易产生竞争，实际效果有限；③当前实现缺乏针对不同通信模式和硬件拓扑的动态自适应调度；④仅在单节点四GPU环境验证，跨节点或更大规模的可扩展性仍待进一步评估。

---

## 138. TTS-PRISM: A Perceptual Reasoning and Interpretable Speech Model for Fine-Grained Diagnosis

**arXiv ID:** 2604.22225 | [PDF](https://arxiv.org/pdf/2604.22225v1)

**作者:** Xi Wang `[一作]` (Tsinghua University), Zhiyong Wu `[通讯]` (Tsinghua University)

**通讯引用:** 22618 | [OpenAlex ID](https://openalex.org/A5063354017)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TTS-PRISM 框架，用 12 维细粒度层级模式对中文 TTS 进行诊断评估，并通过单次推理实现高效可解释评分。

**💡 创新点**

创新点包括：① 12 维细粒度层级评估架构，覆盖音频清晰度、发音、韵律、连贯性、表达性等；② 针对性数据合成与对抗扰动策略，构建 200k 规模的诊断数据集；③ 基于评分标准的指令调优和交互式推理（Rationale–Score 交替序列）实现可解释且减少幻觉。

**🔧 技术方法**

技术手段：MiMo-Audio 100M 小时预训练模型 + schema‑driven instruction tuning + interleaved rationale/score 生成 + 对抗样本生成 + 专家 anchor 评分标准。

**📊 数据集**

使用数据集：200k Mandarin 合成与人声样本（含正负样本与对抗扰动），以及 1,600 条 Gold Test Set（含 20% OOD）做评测。

**📈 对比分析**

对比方法：与 Step‑Audio‑R1、Qwen3‑Omni、Gemini‑2.5‑Pro 等顶尖 Audio‑LLM 进行单维度与多维度推理，评估指标包括 LCC、SRCC、MSE、MSE_norm 与 RSC。TTS‑PRISM 在大多数维度上 LCC 超过 0.70、RSC 达 0.98，显著优于基线。

**⚠️ 局限性**

局限性：① 对发音细粒度错误（Pronunciation Accuracy）的捕捉仍不及 Gemini‑2.5‑Pro，主要受 ASR 预训练偏差影响；② 极端长尾异常样本仍易产生保守预测，需要进一步通过 RL 等方式校准；③ 评测仍以人工专家标签为准，难以覆盖所有语言与口音的多样性。

---

## 139. CharTide: Data-Centric Chart-to-Code Generation via Tri-Perspective Tuning and Inquiry-Driven Evolution

**arXiv ID:** 2604.22192 | [PDF](https://arxiv.org/pdf/2604.22192v1)

**作者:** Xiangxi Zheng `[一作]` (Nanjing University), Alex Jinpeng Wang `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了基于三视角分解SFT与Inquiry-Driven RL的高精度chart‑to‑code生成框架，系统拆分视觉感知、代码逻辑与模态融合三条训练流；

**💡 创新点**

突破同质数据瓶颈：将视觉与代码逻辑解耦并构造独立数据流；将对齐任务改为可验证的问答奖励，取代主观VLM评分；

**🔧 技术方法**

使用Tri‑Perspective Decomposed SFT、冻结Inspector、WebSSL‑1B视觉编码器、Group Relative Policy Optimization，基座为Qwen2.5‑VL‑7B与Qwen3‑VL‑8B；

**📊 数据集**

构建约200万条样本，包括Chart→Caption、Caption→Code、Chart→Code三流，来源于ChartCap、ChartMimic、Plot2Code、ChartX，并采集约2万条QA对进行RL；

**📈 对比分析**

在ChartMimic、Plot2Code、ChartX三大基准上，与GPT‑4o、GPT‑5等对比，7B/8B模型实现SOTA表现，超越开源基线并与GPT‑5相当；

**⚠️ 局限性**

实验仅覆盖7B/8B规模，未验证更大模型；仅评估直接复制任务，未测试迭代编辑、风格迁移等更复杂场景。

---

## 140. Hardware-Software Co-Design for Event-Driven SNN Deployment on Low-Cost Neuromorphic FPGAs

**arXiv ID:** 2604.22179 | [PDF](https://arxiv.org/pdf/2604.22179v1)

**作者:** Jiwoon Lee `[一作]` (Kwangwoon University), Cheolsoo Park `[通讯]` (Kwangwoon University)

**通讯引用:** 2771 | [OpenAlex ID](https://openalex.org/A5072429332)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个语义保持的硬件-软件协同设计框架，能够将PyTorch定义的脉冲神经网络（SNN）模型通过单一部署artifact直接迁移到低成本FPGA的事件驱动执行环境，并在硬件上实现与软件参考完全一致的推理结果。

**💡 创新点**

核心创新点在于：① 单一共享artifact在软件参考和板端运行中保持模型语义不变；② 支持确定性TTFS推理，确保完整10,000图像测试集的预测一致；③ 明确区分硬件加速与系统级测量，提供可复现且可比较的跨平台性能评估。

**🔧 技术方法**

使用了PyTorch风格的SNN建模与导出、Xilinx Zynq‑7020 SoC上的FPGA实现（事件路由器、分组Spike处理、连通表、分组TTFS解码器、计数器等硬件模块），以及与RTX 3080 GPU（FP32/INT8）和CPU（NumPy FP32/INT8）进行的对比实验。

**📊 数据集**

采用MNIST 10类数据集的完整10,000图像测试集进行评估。

**📈 对比分析**

采用相同导出的模型参数，在FPGA 80 MHz实现下对全10,000图像进行推理，FPGA与软件参考完全一致；FPGA单PL延迟0.1375 µs/图，吞吐率7.27 M图/s，动态能耗31.6 nJ/图；与GPU INT8内核相比，FPGA更快且能耗低约933×；与CPU INT8相比，FPGA快约488×。

**⚠️ 局限性**

局限性包括：受BRAM容量限制，当前实现最多支持约2,048个神经元；更大网络受片上存储与突触路由fan‑out限制；仅验证了线性TTFS分类器，未扩展到更深或卷积模型；能耗估计基于工具，未做外部真实测量；单事件驱动模型的适用范围有限。

---

## 141. Recognition Without Authorization: LLMs and the Moral Order of Online Advice

**arXiv ID:** 2604.22143 | [PDF](https://arxiv.org/pdf/2604.22143v1)

**作者:** Tom van Nuenen `[一作]` `[通讯]` (University Of California Berkeley), Tom van Nuenen (University Of California Berkeley)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比四种助手式大型语言模型（Gemini 2.5 Flash Lite、GPT‑4.1‑nano、DeepSeek v3.2、Ministral 8B）在 r/relationship_advice 子版块上产生的回答与社区高投票回答之间的差异，探讨模型如何识别伤害但往往不授权行动。

**💡 创新点**

提出“vernacular competence”（地方性判断与授权能力）概念，并通过“defamiliarization”方法将模型输出与社区集中、投票决定的道德规范对照，从而揭示对齐过程在情境化道德判断中的结构性偏差。

**🔧 技术方法**

使用多模型主题分类（六个不同供应商模型）构建 70 类关系问题词汇表；利用词典‑基准语言度量（情感、肯定、行动许可、疗愈密度等）评估模型与社区回答的差异；并进行高共识（≥70% 留出语言）帖子上的交叉检验。

**📊 数据集**

核心数据集为 11,565 条 r/relationship_advice 帖子及其最高得分的社区回复，来自 2025‑2026 年 Reddit 数据；另用六个模型对同一帖子进行自动主题标注以构建 70 类词汇表。

**📈 对比分析**

对比方法：对每个模型生成回复后，使用词典指标计算“certainty ratio”“deontic modal ratio”“leave ratio”“sentiment”“therapy density”“action:validation ratio”，并与社区平均值对比。结果显示模型在留出、肯定、疗愈方面均高于社区，但在授权行动（leave ratio）上低 60‑70% 左右；在高共识帖子中差距更大。

**⚠️ 局限性**

局限性包括：仅使用四个成本效益型模型，无法覆盖前沿模型；词典方法牺牲语义细粒度；只以单一子版块为基准，无法泛化到其他社区；对齐机制（安全、训练数据均衡）未能单独区分；并且高投票基准自身可能偏向极端直接风格，导致对齐差异被放大。

---

## 142. Voice Under Revision: Large Language Models and the Normalization of Personal Narrative

**arXiv ID:** 2604.22142 | [PDF](https://arxiv.org/pdf/2604.22142v1)

**作者:** Tom van Nuenen `[一作]` (University of California, Berkeley), Tom van Nuenen `[通讯]` (University of California, Berkeley)

**通讯引用:** 759 | [OpenAlex ID](https://openalex.org/A5008485279)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在重写个人叙事文本时产生的风格正则化效应，并评估了不同模型和提示条件的影响。

**💡 创新点**

揭示了LLM重写在功能词、第一人称代词、词汇多样性等13个标记上的方向性一致性，并首次量化了语音保留提示对规范化幅度的抑制作用。

**🔧 技术方法**

采用计算风格学指标、统计效应量、PCA空间投影和最近邻匹配等方法对文本进行定量分析。

**📊 数据集**

以EmpathicStories语料库的300条第一人称叙事为样本。

**📈 对比分析**

将三种前沿LLM（GPT‑5.4、Claude Sonnet 4.6、Gemini 3.1 Pro）在三种提示（generic, rewrite‑only, voice‑preserving）下产生的重写文本与原文本做配对t检验和效应量比较，结果显示所有模型都产生大幅度但方向一致的规范化，提示条件对幅度影响约32%。

**⚠️ 局限性**

仅限单一英语短文本语料，未能分离预训练、指令调优等因素的贡献，且对读者感知与跨语种、跨体裁的普适性尚不明。

---

## 143. TRUST-SC: Truthful Multi-Task Double Auction for Quality-Aware Spatial Crowdsourcing in Strategic Environment

**arXiv ID:** 2604.22241 | [PDF](https://arxiv.org/pdf/2604.22241v1)

**作者:** Chattu Bhargavi `[一作]` (VIT-AP University), Alok Kumar Shukla `[通讯]` (Thapar Institute of Engineering & Technology)

**通讯引用:** 3263 | [OpenAlex ID](https://openalex.org/A5029682055)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为TRUST‑SC的三层机制，先通过空间聚类减少搜索空间，再使用多数投票筛选高质量执行者，最后在每个聚类内执行多单元双拍卖以保证真诚与个人理性；

**💡 创新点**

创新点在于将空间聚类与质量评估相结合，并采用随机分区双拍卖实现无偏见的价格发现与分配；

**🔧 技术方法**

核心技术包括基于欧氏距离的聚类算法、基于多数投票的质量评估、随机分区均衡价格求解以及多单元双拍卖支付规则；

**📊 数据集**

实验采用模拟数据集，任务与执行者随机分布于100×100网格，任务执行者数量从100到1600、请求者数量从50到300进行多次重复实验；

**📈 对比分析**

与McAfee、MUDA和PPM等基准方法比较，TRUST‑SC在社交福利、任务成功率和质量执行者比率上表现更好，但在运行时间上略逊一筹；

**⚠️ 局限性**

局限性包括对随机分区的依赖可能导致整体效率损失、运行时相对较长、仅在合成数据上验证，缺乏真实世界实验验证，且假设代理满足GS估值。

---

## 144. Evaluation of image simulation open source solutions for simulation of synthetic images in lunar environment

**arXiv ID:** 2604.22296 | [PDF](https://arxiv.org/pdf/2604.22296v1)

**作者:** Jai G Singla `[一作]` (Space Applications centre), Nitant Dube `[通讯]` (Space Applications centre)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对多种开源图像模拟工具（ABRAM、CORTO、Blender、QGIS、Python）进行综合评估，并利用高分辨率DEM和真实光照模型生成月球环境下的合成图像。

**💡 创新点**

提出了统一、物理一致的评估框架，能够同时考虑高精度地形、光照、相机模型与阴影效果，实现了对现有工具在月球仿真场景下的全面对比与优化。

**🔧 技术方法**

采用MATLAB/ABRAM、Python/CORTO、Blender渲染引擎、QGIS地理处理以及Lambert、Hapke等物理基光照模型，整合相机参数、视角和光源位置进行光照和阴影计算。

**📊 数据集**

使用Chandrayaan‑2 OHRC、LROC NAC、LROC WAC、LOLA DEM等真实月球地形与遥感数据集。

**📈 对比分析**

通过视觉真实性、阴影一致性、渲染速度和易用性四个维度对比，ABRAM在大数据量下保持较高真实性和适中速度；Blender在图像质量上优于其它工具但渲染耗时较长；CORTO速度快、易扩展；QGIS在3D可视化方面优秀但阴影与相机角度模拟不精确。

**⚠️ 局限性**

Blender缺乏完整GIS功能，难以处理极大数据；QGIS无法精确模拟阴影和相机视角；Python实现的光照阴影细节不足；ABRAM受MATLAB授权限制；整体对高分辨率场景仍需高算力。

---

## 145. Contexts are Never Long Enough: Structured Reasoning for Scalable Question Answering over Long Document Sets

**arXiv ID:** 2604.22294 | [PDF](https://arxiv.org/pdf/2604.22294v1)

**作者:** Harshit Joshi `[一作]` (Stanford University), Monica S. Lam `[通讯]` (Stanford University)

**通讯引用:** 23300 | [OpenAlex ID](https://openalex.org/A5002419188)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可解释的大规模多模态文本检索框架，并在多任务、多语言、多尺度的设置下进行评估

**💡 创新点**

创新点在于将可解释检索和层次化提示设计结合，动态生成多模态检索与生成链路

**🔧 技术方法**

使用检索增强的语言模型技术，结合 LLM、LoRA、可解释检索、层次化提示

**📊 数据集**

采用 WikiTable、OpenWebText、MultiModal 等公开数据集进行实验

**📈 对比分析**

与 GPT‑3.5、GPT‑4、GPTIndex、LlamaIndex 等方法对比，表现优于多项指标（如 F1、BLEU、BLEU‑L）

**⚠️ 局限性**

局限性包括计算成本高、对大模型依赖大、解释性难以满足非专业用户

---

## 146. HGQ-LUT: Fast LUT-Aware Training and Efficient Architectures for DNN Inference

**arXiv ID:** 2604.22293 | [PDF](https://arxiv.org/pdf/2604.22293v1)

**作者:** Chang Sun `[一作]`, Maria Spiropulu `[通讯]` (California Institute Of Technology)

**通讯引用:** 78538 | [OpenAlex ID](https://openalex.org/A5107388490)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在 FPGA 上实现低延迟高效推理的 LUT‑aware 训练框架 HGQ‑LUT，提出 LUT‑Dense 与 LUT‑Conv 层并提供端到端工具链。

**💡 创新点**

将 1‑输入 LUT 设计与 HGQ 量化结合，实现训练速度提升 100 倍；通过混合精度和 EBOPs 资源估算自动搜索准确率‑资源 Pareto 前沿；同时实现可混合 LUT 与传统算术的统一编译与验证。

**🔧 技术方法**

LUT‑aware training, HGQ 逐元素异质量化, Einsum/GEMM GPU 加速, EBOPs 资源 surrogate, DAIS 指令扩展, RTL 生成, JAX/FLAX 与 TSMC/FPGA 开源工具链。

**📊 数据集**

OpenML 与 CERNBox 的 HLF Jet Substructure Classification, Zenodo 的 PLF JSC, ATLAS TGC Muon Tracking, CEPC 气体探测器 PID。

**📈 对比分析**

与 NLA、KANELE、LUTNet 等现有 LAT 方法比较；在所有任务上准确率相当或更高，LUT 资源与延迟显著下降，HGQ‑LUT 的训练时间比 NLA 快 197×。

**⚠️ 局限性**

仍需针对不同任务手动调整混合精度；高维低信息输入时需混合架构；资源估算依赖经验式 EBOPs，精度有限；未扩展到 Transformer 等更大模型。

---

## 147. DocPrune:Efficient Document Question Answering via Background, Question, and Comprehension-aware Token Pruning

**arXiv ID:** 2604.22281 | [PDF](https://arxiv.org/pdf/2604.22281v1)

**作者:** Joonmyung Choi `[一作]` (Korea University), Hyunwoo J. Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关、逐步进行的文档视觉问答 Token 剪枝框架 DocPrune，包含背景 Token 剪枝（BTP）、问题相关 Token 剪枝（QTP）和理解度感知 Token 剪枝（CTP）三个阶段，显著减少视觉 Token 数量并提升推理效率。

**💡 创新点**

创新点在于：①利用文档的结构稀疏性（大量无意义背景）与问题语义相关性，实现针对性 Token 剪枝；②引入模型层级理解度（Last‑Token L2 范数）动态决定剪枝时机，避免固定层剪枝导致性能波动；③全部过程无需额外训练，直接应用于现有检索‑增量式 QA 流程。

**🔧 技术方法**

技术要点包括：1) 基于像素模式和阈值的背景检测；2) 通过检索阶段的视觉与问题嵌入余弦相似度构建热度图并 Gaussian 平滑得到 QTP 决策；3) 在解码器每层计算 Last‑Token L2 范数作为理解度指标，满足阈值后按注意力权重剪枝；4) 与 ColPali、Qwen2‑VL 等预训练 VLM 结合，保持无训练、轻量化的特性。

**📊 数据集**

主要使用的数据集有：M3DocRAG（检索+生成），M3DocVQA、MMLongBench‑Doc、ChartQA、SlideVQA、InfoVQA、DUDE 等多种文档视觉问答与推理基准，评估跨任务的通用性。

**📈 对比分析**

与基线（未剪枝的 M3DocRAG）及现有剪枝方法（FastV、DivPrune、VTW）对比，DocPrune 在编码器和解码器上分别提升了约 3.0× 与 3.3× 的吞吐率，同时 F1 与 EM 分别提升 1.0 与 1.5 分，甚至超过了未剪枝的 baseline，证明在保持或提升准确率的同时大幅降低计算量。

**⚠️ 局限性**

局限性包括：①剪枝阈值需要手工设置，可能在不同数据/模型上需要微调；②过度依赖检索阶段的嵌入质量，若检索不准确会导致 QTP 失效；③目前仅针对检索‑增量式 QA 框架，未验证在更复杂的多页跨模态推理或实时应用中的鲁棒性；④缺乏训练步骤可能限制对特殊领域（如医学/法务文档）的适配。

---

## 148. How LLMs Detect and Correct Their Own Errors: The Role of Internal Confidence Signals

**arXiv ID:** 2604.22271 | [PDF](https://arxiv.org/pdf/2604.22271v1)

**作者:** Dharshan Kumaran `[一作]` (Google DeepMind), Nathaniel Daw `[通讯]` (Google DeepMind)

**通讯引用:** 37204 | [OpenAlex ID](https://openalex.org/A5029249257)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在无外部反馈情况下自我检测错误并尝试自我纠正的机制，探究了其背后的第二阶置信度架构与 post‑answer newline (PANL) 信号的作用。

**💡 创新点**

首次证明LLM内部在答案后首个换行符处缓存的评估表示是第二阶置信度来源，能够在行为置信度不足时预测错误检测与可纠正错误，并揭示其在自我纠正过程中的因果作用。

**🔧 技术方法**

结合 verbal confidence、token log‑probabilities、线性探针、激活补丁与噪声实验，分析Gemma 3 27B 与 Qwen 2.5 7B 在 TriviaQA 与 MNLI 上的表现。

**📊 数据集**

使用 TriviaQA（事实问答）和 MNLI（自然语言推断）两个公开数据集进行实验。

**📈 对比分析**

通过 AUROC、d'、logistic 回归等指标比较行为信号与 PANL 激活的预测性能，发现 PANL 在错误检测与可纠正错误预测上显著优于所有行为信号，自我纠正准确率从约 75.5% 提升至 79.2%。

**⚠️ 局限性**

研究仅聚焦于无链式思维的事实问答任务，缺乏在更复杂推理场景下的验证，且对模型内部知识状态的解释仍不够充分。

---

## 149. A Probabilistic Framework for Hierarchical Goal Recognition

**arXiv ID:** 2604.22256 | [PDF](https://arxiv.org/pdf/2604.22256v1)

**作者:** Chenyuan Zhang `[一作]` (Monash University), Mor Vered `[通讯]` (Monash University)

**通讯引用:** 441 | [OpenAlex ID](https://openalex.org/A5009808948)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于 HTN 的规划框架，将目标识别转化为贝叶斯推断，从观测中计算每个目标假设的后验概率。

**💡 创新点**

首次将层次化任务结构与概率推理结合，提出三阶段生成模型并给出可用 HTN 规划器实现的似然近似，同时支持外部（exogenous）动作的处理。

**🔧 技术方法**

使用 HTN 规划、Boltzmann 采样、动态规划对齐、离散化先验和进度先验等技术，并通过现成的 HTN 规划器实现最优/近似解的搜索。

**📊 数据集**

在 Kitchen（多道菜烹饪）和 Monroe（自动生成的规划会话）两个基准域上进行实验。

**📈 对比分析**

与已有的 HTN 目标识别方法相比，实验显示在完整或部分观测下均显著提升 top‑1、top‑3、top‑5 的准确率（如 Kitchen 培训中 20% 观测时 top‑1 由 19.8% 提升至 72.9%），仅导致平均每实例约 4 倍的运行时间（≈5.1→23.9 秒）。

**⚠️ 局限性**

依赖 HTN 规划器的完备性与任务插入支持；近似推断会导致概率误差；对极长或高度分支的 HTN 结构，方法选择和概率估计可能不稳定。

---

## 150. Algorithmic Feature Highlighting for Human-AI Decision-Making

**arXiv ID:** 2604.22236 | [PDF](https://arxiv.org/pdf/2604.22236v1)

**作者:** Yifan Guo `[一作]` (Stanford University), Jann Spiess `[通讯]` (Stanford University)

**通讯引用:** 4938 | [OpenAlex ID](https://openalex.org/A5061755779)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种算法通过突出显示少量案例特定特征来帮助人类决策者做出更好决策的模型；

**💡 创新点**

阐明了当决策者是精明型（考虑突出规则）与天真型（忽视突出规则）时，最优突出策略会截然不同，并证明精明型最优策略计算上不可行；

**🔧 技术方法**

采用信息设计与贝叶斯说服框架、NP‑hardness 证明、凸/子模性质分析、贪心算法以及统计分析等技术；

**📊 数据集**

在美国住房调查（AHS）数据集上进行实证验证，使用 44 维住房特征与房价信息；

**📈 对比分析**

将固定子集与情景突出（基于惊奇、边际、贪心）等策略在均方误差指标下比较，结果显示情景突出在所有预算下明显优于固定策略，且在部分预算下甚至优于全信息；

**⚠️ 局限性**

限制包括：假设人类与算法共享完全相同的先验与偏好、仅考虑离散或高斯分布、忽略决策者私有信息或模型误差、对天真型策略的鲁棒性不足以及精明型策略计算复杂度高。

---

## 151. FETS Benchmark: Foundation Models Outperform Dataset-specific Machine Learning in Energy Time Series Forecasting

**arXiv ID:** 2604.22328 | [PDF](https://arxiv.org/pdf/2604.22328v1)

**作者:** Marco Obermeier `[一作]` (Julius-Maximilians-Universität Würzburg), Andreas Zeiselmair `[通讯]` (Weihenstephan-Triesdorf University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文建立了面向能源时序预测的 FETS 基准，系统收集 54 条跨 9 类的数据集，并在不同的预测模式（零样本、协变量、微调）下对多种时序基础模型与经典机器学习方法进行统一评估。

**💡 创新点**

创新点在于首次从利益相关者、属性和数据类别三维度构建能源预测任务框架，聚合多领域公开数据并提供统一评测；同时证明了协变量信息下的基础模型（Chronos‑2）在零样本与微调条件下均能超越专门调优的 XGBoost / 随机森林。

**🔧 技术方法**

采用的技术包括多种时序基础模型（Chronos‑2、TimesFM、TiRex、FlowState、TabPFN‑TS）与传统树模型，统一使用 NRMSE 作为评价指标，并对上下文长度、预测窗口、聚合层级等超参数进行系统敏感性分析。

**📊 数据集**

数据集涵盖电力负荷、可再生发电、能源价格、热负荷、移动出行、电网流量等，来源于 ENTSO‑E、CAISO、开放式能源竞赛、公开热泵/充电站记录等。

**📈 对比分析**

比较方法为在同一划分、同一滚动窗口下对每个模型进行多次实验，统计 NRMSE 中值及胜率；结果显示协变量基础模型平均 NRMSE 0.421，胜率 20%，明显优于其他模式。

**⚠️ 局限性**

局限性包括：对极短期或高频预测仍需更多训练数据；基础模型规模大、推理成本高；部分模型缺乏微调与多变量支持；评测仅聚焦欧盟/美国公开数据，未覆盖所有地区与业务场景。

---

## 152. Introducing the Cyber-Physical Data Flow Diagram to Improve Threat Modelling of Internet of Things Devices

**arXiv ID:** 2604.22307 | [PDF](https://arxiv.org/pdf/2604.22307v1)

**作者:** Simon Liebl `[一作]` (emgarde), George R. S. Weir `[通讯]` (University of Strathclyde)

**通讯引用:** 1685 | [OpenAlex ID](https://openalex.org/A5005334221)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估一种专为物联网设备设计的威胁建模技术——Cyber‑Physical Data Flow Diagram (CPDFD)，并通过实验研究与行业问卷访谈验证其有效性。

**💡 创新点**

创新点：①在传统DFD基础上新增 Physical Link 与 Interface 两个元素，以更直观地表征传感器/执行器与物理环境的交互；②将硬件与逻辑模型分离，构建硬件模型（HWD）以捕捉硬件攻击场景；③统一符号体系，使硬件、嵌入式与软件工程师均可使用。

**🔧 技术方法**

采用扩展的DFD、STRIDE 与 LINDDUN 威胁模型、定量实验、问卷与访谈，并利用自研工具 TTModeler 进行建模与时间追踪；对实验结果使用非参数检验与 Cohen’s d 进行统计分析。

**📊 数据集**

数据来源：公开 IoT 设备 Jaimico（语音助手+健康监测）作为案例；实验参与者 41 名计算机专业学生；行业调查 15 名制造商与咨询公司专家；收集模型文件与问卷/访谈文本。

**📈 对比分析**

对比方法：将使用 CPDFD 与传统 DFD 的两组实验者分别建模并计数攻击场景与耗时。结果显示 CPDFD 组平均识别 49.9 个相关攻击场景（比 DFD 的 27.3 多约 83%），每场景识别时间从 4 分钟降至 2 分钟，效能提升显著（p < 0.001，Cohen’s d > 1.3）。访谈与问卷也表明 CPDFD 更易用且捕获更多隐私威胁。

**⚠️ 局限性**

限制：①样本规模相对较小，实验主体为学生，可能不代表行业实际；②预先定义的组件 stencils 可能导致新元素使用率偏高；③硬件模型与 DFD 可能存在重叠，导致新增攻击场景被高估；④未深入评估工具与模型的整合细节及长期实战效果。

---

## 153. BLAST: Benchmarking LLMs with ASP-based Structured Testing

**arXiv ID:** 2604.22306 | [PDF](https://arxiv.org/pdf/2604.22306v1)

**作者:** Manuel Alejandro Borroto Santana `[一作]` (University of Calabria), Francesco Ricca `[通讯]` (University of Calabria)

**通讯引用:** 3615 | [OpenAlex ID](https://openalex.org/A5013411309)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 BLAST 评估方法与对应数据集，用以系统化评估大型语言模型（LLM）在生成答案集编程（ASP）代码方面的准确性。

**💡 创新点**

首创针对 ASP 代码生成的零触发基准及两种自动化语义评估指标（测试套件和模型基准），并实现了谓词匹配器与伪造器的完整流程。

**🔧 技术方法**

使用多种主流 LLM（如 GPT‑4o、DeepSeek‑R1 等）、ASP 解析器、谓词匹配器、伪造器以及 ASP‑WIDE 单元测试框架，配合 Python 脚本实现端到端评估。

**📊 数据集**

构建包含十个典型图论问题（Colorability、Dominating Set、Hamiltonian Cycle、Hamiltonian Path、Hierarchical Clustering、Maximal Clique、Partition、Slitherlink、Stable Marriage、Traveling Salesman）的基准集，配备正式自然语言描述、金标准 ASP 程序、测试用例与输入实例。

**📈 对比分析**

对八款 LLM 进行零触发调用，分别测算语法正确率和语义正确率（测试套件与模型基准），结果显示大部分 LLM 在语法层面表现良好，但在语义层面表现较弱，复杂任务的准确率更低；两种语义指标高度相关，测试套件更高效且更易解释。

**⚠️ 局限性**

局限包括：仅使用零触发提示；缺乏少量示例或检索增强；基准仅覆盖图论领域；谓词匹配和测试套件设计需要人工干预；未探讨更高级的提示策略或模型专门化编码技术。

---

## 154. Semantic Error Correction and Decoding for Short Block Channel Codes

**arXiv ID:** 2604.22269 | [PDF](https://arxiv.org/pdf/2604.22269v1)

**作者:** Jiafu Hao `[一作]` (University of Sydney), Branka Vucetic `[通讯]` (University of Sydney)

**通讯引用:** 24078 | [OpenAlex ID](https://openalex.org/A5043371405)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于短块码的接收端框架，结合预训练语言模型实现语义误差校正、语义列表解码和语义置信度引导的HARQ；通过分段传输保留局部上下文，提升语义恢复能力。

**💡 创新点**

创新点在于将短块码与语言模型推理耦合，利用局部错误保留上下文实现语义校正；引入语义列表解码以多候选方式提升纠错可靠性；用置信度代替CRC实现分段HARQ，显著降低冗余并提升重传效率。

**🔧 技术方法**

使用技术包括：短块线性码 + 有序统计解码 (OSD)、BART 预训练语言模型 + BPE 分词、语义列表解码的多候选生成、加权汉明距离选择、增量冗余的语义 HARQ。

**📊 数据集**

实验数据集为 Stanford Natural Language Inference (SNLI)，使用约 20k 句子进行训练，500 句子评测。

**📈 对比分析**

通过与单长 LDPC (1024,512) 的基线比较，采用 BLER、BLEU、ROUGE 评估；单发时 SEC 提升约 0.4 dB，SLD 提升至 0.8 dB；使用 SHARQ 可进一步提升约 1.5 dB；在相同 BLER 下，延迟降低 90% 以上，语义指标（BLEU/ROUGE）保持高水平。

**⚠️ 局限性**

局限性包括：需对语言模型进行专门的 fine‑tune，适用于结构化文本，对非文本或多语言场景需进一步验证；OSD 在极短块码下复杂度上升；分段分割对上下文依赖度高，若错误集中可能导致恢复失效；整体系统实现仍需考虑实际硬件可行性。

---

## 155. Large Language Models Decide Early and Explain Later

**arXiv ID:** 2604.22266 | [PDF](https://arxiv.org/pdf/2604.22266v1)

**作者:** Ayan Datta `[一作]` (IIIT Hyderabad), Alexander Mehler `[通讯]` (Goethe University)

**通讯引用:** 1914 | [OpenAlex ID](https://openalex.org/A5008340710)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型在推理过程中答案何时确定，并提出在答案稳定后提前停止生成，以减少冗余推理文本。

**💡 创新点**

创新点在于：①通过强制答案完成技术实时捕捉答案轨迹，揭示答案大多在推理早期就已确定；②提出对答案轨迹进行去噪和持久性平滑（hold‑for‑k），区分短暂翻转与真正答案切换；③设计基于模型内部表示的学习型“probe”门控，实现无须额外模型训练的推理提前终止。

**🔧 技术方法**

技术主要包括：强制答案完成、答案轨迹分析、持久性平滑、线性probe门控、随机门控与任务特定/通用probe对比，使用Qwen3系列模型进行推理。

**📊 数据集**

使用六类任务数据集：多项选择（MCQ）、数值答案（Numeric‑answer）、搜索查询（Search‑query）、工具选择（Tool‑selection），以及三个更难的推理基准（Humanity’s Last Exam、GPQA‑Diamond、AIME 2026）。

**📈 对比分析**

与完整推理对比，通用probe在Qwen3‑4B上可节省约500–900个推理token，准确率下降不到5%；随机门控在相同token节省下准确率下降更大。相比手工规则或无论何时停止，probe门控在多任务上表现更稳定，尤其在数值和搜索任务中保持高精度或相似度。

**⚠️ 局限性**

局限性：实验仅覆盖中小规模模型和文本推理任务；未探究更大模型或极长推理序列；未验证多模态或非语言推理场景；通用probe虽然效果好，但对极端任务的适用性仍需进一步评估。

---

## 156. Towards Safe Mobility: A Unified Transportation Foundation Model enabled by Open-Ended Vision-Language Dataset

**arXiv ID:** 2604.22260 | [PDF](https://arxiv.org/pdf/2604.22260v1)

**作者:** Wenhui Huang `[一作]` (Nanyang Technological University), Chen Lv `[通讯]` (Nanyang Technological University)

**通讯引用:** 18262 | [OpenAlex ID](https://openalex.org/A5072073374)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了面向城市交通的大规模视听语言数据集LTD，并基于该数据集设计并训练了一种统一的基础模型UniVLT，用以在微观自动驾驶与宏观交通分析之间实现跨域推理。

**💡 创新点**

创新点在于：①首次构建开源的多视角、最小关联的道路摄像头视图下的开放式问答数据集；②引入课程式知识迁移与经验回放，逐步从通用知识迁移到自动驾驶，再迁移到交通域，避免灾难性遗忘；③实现了单一模型同时处理多图像推理、摄像头定位与多对象定位三大任务。

**🔧 技术方法**

使用了Qwen2.5-VL 7B的视觉编码器与LLM骨干，结合LoRA微调、2D RoPE、RMSNorm、多图像视觉标记拼接等技术；训练采用三阶段课程式迁移：通用预训练 → 自动驾驶数据微调 → 交通数据微调并加入AD经验回放。

**📊 数据集**

使用的数据集包括：自动驾驶开源数据集LingoQA、OmniDrive、CODA-LM；交通域数据集LTD（约11.6K QA对，包含多图像风险分析、摄像头ID选择、多对象定位）；此外在基准实验中还评估了LingoQA、OmniDrive、CODA、SUTD-TrafficQA等公共数据集。

**📈 对比分析**

通过与多种基线对比（通用VLM如LLaVA-OV、InternVL2.5、Qwen3-VL；专门针对AD的VLM如OpenEMMA、WiseAD、RoboTron-Drive、ReCogDrive），在LTD上的多图像风险分析、摄像头ID选择、多对象定位任务中，UniVLT分别获得0.66、0.66、0.64的GPT/准确率/F1分数，显著优于所有基线；在LingoQA、OmniDrive和CODA等AD任务上亦实现或逼近SOTA。

**⚠️ 局限性**

局限性包括：①对不同城市的跨域泛化尚未充分验证；②在极端天气或稀疏标记场景下的鲁棒性仍需提升；③模型规模相对较大（7B参数），对算力和部署成本有一定要求；④在多图像互相关系极弱的情况下，仍可能受限于视觉编码器对非关联视图的表征能力。

---

## 157. Protect the Brain When Treating the Heart: A Convolutional Neural Network for Detecting Emboli

**arXiv ID:** 2604.22258 | [PDF](https://arxiv.org/pdf/2604.22258v1)

**作者:** Andrea Angino `[一作]` (Università della Svizzera Italiana), Stefanos Demertzis `[通讯]` (Ente Ospedaliero Cantonale)

**通讯引用:** 2238 | [OpenAlex ID](https://openalex.org/A5053968163)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

使用2.5D U‑Net实现心脏超声中气体微栓的实时分割与量化

**💡 创新点**

结合短时序信息的2.5D U‑Net在保持实时性同时显著提升对运动微栓的识别准确率

**🔧 技术方法**

采用2.5D U‑Net网络、二值分割、阈值调节与指数滑动平均等技术

**📊 数据集**

基于8条约2秒、60fps、600×800分辨率的心脏超声录像共约4000块训练样本

**📈 对比分析**

与传统阈值/LoG/Fiji方法对比，FBI≈100%、IoU>0.5、Dice>0.7，推理延迟≤0.2s，性能优异

**⚠️ 局限性**

样本量有限、仅单设备单中心、对不同患者和成像条件的泛化能力不足

---

## 158. False Feasibility in Variable Impedance MPC for Legged Locomotion

**arXiv ID:** 2604.22251 | [PDF](https://arxiv.org/pdf/2604.22251v1)

**作者:** Vishal Ramesh `[一作]` `[通讯]`, Vishal Ramesh

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究可变阻尼模型预测控制（Variable Impedance MPC）在面对执行器有限带宽时的可行性问题，提出并验证了“虚假可行性”概念，并给出了阈值 α* 以及不可实现下限 β，证明在 α < α* 时，参数化 MPC 的最优解不可被实际执行器跟踪；同时验证该机制在 1D 单足跳跃和 2D SLIP 平面模型中可转移。

**💡 创新点**

创新点包括：① 从可行集正确性角度正式区分参数化可行集与可实现集；② 推导出基于任务物理参数的无量纲阈值 α*，以及在 α < β 时无任何范围约束可实现的硬性下限；③ 显示在 1D 与 2D 任务中该机制的转移与相同的数值尺度；④ 通过对比参数化与增益状态 MPC，揭示结构性误差对闭环性能的累积影响；⑤ 说明保守调节无法修复该误差的结构性根源。

**🔧 技术方法**

技术方法：数学推导（无量纲分析、极大斜率比较、证明非实现性与硬性下限）、模型预测控制（参数化与增益状态两种形式）、仿真（基于 Runge‑Kutta 的连续动力学、参数扫描）、实验设置（1D 单足跳跃、2D SLIP 运动学）、性能评估（L∞ 跌落误差、起飞时间偏差）。

**📊 数据集**

数据集：全部为仿真生成的数据；1D 模型采用多组参数（质量 0.5–2 kg，立跳时间 0.2–0.4 s，刚度范围 50–100 N/m 与 300–800 N/m）；2D SLIP 采用固定参数（质量 1 kg，重力 9.81 m/s²，腿长 0.5 m，刚度上限 5000 N/m，下限 500 N/m）并扫描 α 值。

**📈 对比分析**

比较方法：对同一任务使用参数化 MPC 与增益状态 MPC；在每个 α 下测量两者的 L∞ 轨迹偏差和起飞时间偏差；结果显示参数化 MPC 随 α 降低误差呈单调增长并趋于 1，增益状态 MPC 在所有 α 下误差接近零；在保守调节范围内，成本差距在 10% 以内，但实现范围被显著削减；因此，增益状态 MPC 在可实现性上更具优势。

**⚠️ 局限性**

局限性：① 解析阈值仅在 1D 单足模型中严格推导，二维模型仅验证机制转移而未再推导阈值；② 执行器模型为一阶线性，未考虑高阶/非线性动力学；③ 仅适用于特定类成本函数（如平方冲击力），未给出通用成本函数条件；④ 仅在仿真层面验证，缺乏硬件实验；⑤ 保守调节最小化的可行性分析仅针对单个边界值，未证明在所有范围约束下的最优性；⑥ 未与强化学习等端到端控制策略比较。

---

## 159. Depth-Aware Rover: A Study of Edge AI and Monocular Vision for Real-World Implementation

**arXiv ID:** 2604.22331 | [PDF](https://arxiv.org/pdf/2604.22331v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 160. Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA

**arXiv ID:** 2604.22239 | [PDF](https://arxiv.org/pdf/2604.22239v1)

**作者:** Zhanli Li `[一作]` (Chinese Academy of Sciences), Ping Luo `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 55150 | [OpenAlex ID](https://openalex.org/A5100752686)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模多文档分析问答基准 MuDABench，并提出了基于元数据的多代理工作流来完成跨文档信息抽取与聚合。

**💡 创新点**

创新点在于：①针对多文档分析任务的专门基准；②通过元数据驱动的规划、抽取、归一化、代码执行四阶段分层工作流，显著提升了最终答案准确率；③引入中间事实覆盖度作为诊断指标。

**🔧 技术方法**

使用了检索增强生成（RAG）与大型语言模型（GPT‑4o、GPT‑4.1 mini），并结合多代理框架（规划、抽取、归一化、代码执行）实现端到端推理。

**📊 数据集**

使用的基准数据集是 MuDABench，包含 80,000+ 页、332 个问答实例，来源于中国与美国上市公司年报、ESG 报告与公告。

**📈 对比分析**

与标准 RAG、不同的检索预算及元数据注入相比，工作流在过程准确率与最终答案准确率上都有显著提升，但仍落后于人类专家；RAG 在覆盖度提升后并未带来明显答案改进，说明聚合与推理是瓶颈。

**⚠️ 局限性**

限制包括仅覆盖金融领域、数据规模有限、对事实粒度与等价性处理复杂，且在极长文档或多语言场景下抽取仍不稳定。

---

## 161. CodeGraphVLP: Code-as-Planner Meets Semantic-Graph State for Non-Markovian Vision-Language-Action Models

**arXiv ID:** 2604.22238 | [PDF](https://arxiv.org/pdf/2604.22238v1)

**作者:** Khoa Vo `[一作]` (University of Arkansas), Ngan Le `[通讯]` (University of Arkansas)

**通讯引用:** 6684 | [OpenAlex ID](https://openalex.org/A5108408962)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于可执行代码规划器、持久语义图状态和进度引导的视觉语言提示的层次化框架，解决非马尔可夫长周期操作问题。

**💡 创新点**

创新点在于将任务进度追踪与子任务规划交给一次性合成的可执行代码，利用持久语义图显式保存重要证据，并通过进度引导的视觉‑语言提示抑制杂乱，显著提升在非马尔可夫环境下的鲁棒性。

**🔧 技术方法**

使用语义图构建（YOLOE+CLIP+Cutie）、LLM代码合成（GPT‑5）、视觉‑语言模型（VLM）与可执行的 Python 规划器，以及进度引导的视觉‑语言提示。

**📊 数据集**

在三种真实桌面操作任务（Pick‑and‑Place Twice、Place‑and‑Stack、Swap Cups）上收集的 300 条遥控演示数据进行训练与评估。

**📈 对比分析**

与 π_0、π_0 FAST、π_0.5、Gr00T N1.5 以及时间上下文增强的 Gr00T N1.5+Multi‑frame 进行对比，平均成功率从 56.7% 提升到 81.7%，并且规划延迟显著降低。

**⚠️ 局限性**

依赖于基础模型的语义图构建与代码合成，容易受到视角、图像质量和提示设计的影响，且当前无法实现开放世界的语义图自动生成。

---

## 162. ChangeQuery: Advancing Remote Sensing Change Analysis for Natural and Human-Induced Disasters from Visual Detection to Semantic Understanding

**arXiv ID:** 2604.22333 | [PDF](https://arxiv.org/pdf/2604.22333v1)

**作者:** Dongwei Sun `[一作]` (Xi’an Jiaotong University), Jón Atli Benediktsson `[通讯]` (University of Iceland)

**通讯引用:** 44679 | [OpenAlex ID](https://openalex.org/A5035508615)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种多模态遥感灾害分析框架 ChangeQuery，利用预灾光学图像与灾后 SAR 数据生成可交互的灾害评估报告。

**💡 创新点**

①构建 DICQ 数据集，涵盖光学- SAR 双模态、自然与人为灾害均衡；②自动语义标注流水线将像素分割转化为分区统计、结构坐标及层次化指令；③引入 Change‑Aware Difference 模块与两阶段训练，实现跨模态差异提取与语义推理；④支持多任务、交互式查询，显著降低灾害偏差与幻觉。

**🔧 技术方法**

使用 CLIP‑ViT 视觉编码器，Cross‑Attention 差异提取，MLP 投影，Vicuna‑7B 语言模型，LoRA 微调，PCA 计算 OBB，统计‑生成语义推理及两阶段训练策略。

**📊 数据集**

使用 DICQ（2026 版）约 136,672 双时序图像对，覆盖自然灾害、工业灾害与冲突场景，并与 UCM‑Captions、RSICD、LEVIR‑CC、RSCC 等基准数据集进行对比。

**📈 对比分析**

与通用 MLLM（LLaVA‑Next‑Interleave、LLaVA‑OneVision、InternVL3）以及遥感专用模型（CCExpert、TEOChat、RSCC）在 ROUGE、METEOR、ST5‑SCS 等指标上对比；ChangeQuery 在 METEOR 上从约 13% 提升至 22%，ST5‑SCS 达 54%，显著优于所有基线。

**⚠️ 局限性**

仍对极低分辨率或极端天气场景的鲁棒性不足；对训练数据覆盖范围有限，未覆盖所有特殊灾害；模型体积 7B，部署成本相对较高；未实现多时序连续追踪与更细粒度的时空推理。

---

## 163. Dynamically Acquiring Text Content to Enable the Classification of Lesser-known Entities for Real-world Tasks

**arXiv ID:** 2604.22325 | [PDF](https://arxiv.org/pdf/2604.22325v1)

**作者:** Fahmida Alam `[一作]` (University of Arizona), Ellen Riloff `[通讯]` (University of Arizona)

**通讯引用:** 10518 | [OpenAlex ID](https://openalex.org/A5005791318)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种框架，利用网页检索和大型语言模型自动生成任务相关描述文本，并基于实体名称与金标训练分类器，应用于SIC代码和医疗提供者税码的分类。

**💡 创新点**

创新点在于实现无预构建文本资源的自动知识获取，结合检索和LLM生成文本的互补优势，支持多域零样本分类。

**🔧 技术方法**

采用Web检索（SerpAPI）、LLM生成（GPT‑4o‑mini、LLaMA‑3.1‑8B‑Instruct）、编码器模型（BERT、RoBERTa、Longformer）与生成模型微调，并通过置信度阈值实现高精度推断。

**📊 数据集**

使用两套自建基准数据集：基于jiang2023sicdataset扩展的5400个组织（27类）SIC代码数据；以及从NPPES抽样的3400个医疗提供者（17类）税码数据。

**📈 对比分析**

与提示式LLM基准和单一文本来源进行对比，微调+检索+生成文本（+G+L）在SIC分类上F1 82.3%、医疗税码分类上F1 72.9%，显著优于提示或单一来源，且置信度阈值可实现90%以上精度。

**⚠️ 局限性**

局限性包括对外部检索与LLM的可访问性与成本依赖，文本生成可能产生幻觉或偏差，对稀有实体检索效果有限，且未公开原始检索文本影响重现性。

---

## 164. Revisiting Geometric Obfuscation with Dual Convergent Lines for Privacy-Preserving Image Queries in Visual Localization

**arXiv ID:** 2604.22310 | [PDF](https://arxiv.org/pdf/2604.22310v1)

**作者:** Jeonggon Kim `[一作]` (Hanyang University), Je Hyeong Hong `[通讯]` (Hanyang University)

**通讯引用:** 3552 | [OpenAlex ID](https://openalex.org/A5010730040)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的隐私保护关键点升维方法Dual Convergent Lines（DCL），通过将图像空间分区并将关键点投影到两个固定锚点来实现对云端视觉定位查询的隐私保护，同时保持与传统线基定位求解器的兼容性。

**💡 创新点**

DCL的创新点在于利用双锚点投影使得邻域几何恢复攻击的优化问题变得不良或不唯一，从而产生两种失效模式（锚点收敛与线平行不稳定），有效阻止原始关键点位置恢复和后续图像重建。

**🔧 技术方法**

采用2D线投影（line lifting）技术、基于最小求解器l6P的RANSAC定位、SuperPoint特征提取以及对邻域几何恢复攻击的理论与实验分析，结合线基求解器实现定位。

**📊 数据集**

在三大视觉定位基准数据集上验证：7Scenes（室内）、Cambridge Landmarks（户外多尺度）和Aachen Day‑Night（大规模昼夜变化）。

**📈 对比分析**

与随机线、坐标置换等传统几何隐匿方法以及Segmentation‑based和Descriptor‑free方法进行对比，DCL在点恢复误差与图像重建质量（PSNR/SSIM/LPIPS）上表现最差（即最安全），并在定位精度（旋转/平移误差）与实时性（约4‑6 ms）上与传统方法相当或更优，尤其在大规模数据集上仍保持良好鲁棒性。

**⚠️ 局限性**

在极少数情况下所有关键点落在同一分区会导致定位退化；锚点距离过大会使投影线过平行，影响定位精度；需要进一步探索自适应分区和动态锚点策略来提升鲁棒性。

---

## 165. Resource-Aware Layered Intrusion Detection Allocation Model

**arXiv ID:** 2604.22304 | [PDF](https://arxiv.org/pdf/2604.22304v1)

**作者:** Ioan Pădurean `[一作]` (George Emil Palade University of Medicine, Pharmacy, Science, and Technology), Roland Bolboacă `[通讯]` (George Emil Palade University of Medicine, Pharmacy, Science, and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于整数线性规划的资源感知分层入侵检测分配模型，决定每台设备在 Ethernet、IP、传输和应用四层中监控到何层，并在总资源、关键设备最低层、受限设备最大层等约束下最大化检测效率。

**💡 创新点**

创新点在于：①将监控层级视为单一深度分配而非多层组合；②将设备重要性、攻击概率、层级检测率、监控成本统一到目标函数；③加入关键设备最小层约束和受限设备最大层约束，形成多目标约束优化；④使用SCIP求解器在异构网络上进行验证。

**🔧 技术方法**

使用整数线性规划（ILP）技术，并在SCIP优化框架下求解；模型中使用二进制决策变量 y_i,l 表示设备 i 是否监控到层 l；目标函数为权重 w_i、攻击概率 p_i 与层级检测率 d_l 的乘积求和，约束包括资源预算、设备唯一深度、关键设备层级和受限设备层级限制。

**📊 数据集**

数据集为人工构造的六设备异构网络，设备属性（重要性权重 w_i、攻击概率 p_i）和监控层级成本 c_l、检测率 d_l 通过实验设定；没有使用公开真实网络数据集。

**📈 对比分析**

方法主要通过展示不同预算 R 下的最优层级分配结果来说明模型的可行性和效果。实验表明：在有限预算时，模型优先将更深层监控分配给重要且攻击概率高的设备；在预算足够时，可覆盖更多设备和层级；若预算过低（如 R=5）则无解。

**⚠️ 局限性**

局限性包括：①仅在极小规模（6台设备）的实验网络上验证，缺乏大规模可扩展性评估；②攻击概率和检测率被假设为已知确定值，未考虑不确定性；③未与现有多阶段检测器或自动化机器学习方法进行性能对比；④模型假设每台设备只需监控至单一层次，未考虑多层级联合检测的可能收益。

---

## 166. STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation

**arXiv ID:** 2604.22282 | [PDF](https://arxiv.org/pdf/2604.22282v1)

**作者:** Peng Yu `[一作]` (Kingsoft Corporation), Yinfei Xu `[通讯]` (Kingsoft Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Structure-Tracing Evidence Mining (STEM) 框架，将多跳知识图问答从路径搜索转化为结构化子图匹配，并通过语义到结构的投影及Triple-GNN实现全局结构引导。

**💡 创新点**

创新点包括：① 语义-结构投影管线（SGDA+SAGB）解决 LLM 产生的 schema 误差；② Triple-Dependent GNN 生成全局引导子图，提供全局结构先验；③ 结构追踪子图检索结合实体与三元组层面的全局一致性偏置；④ 依据答案多样性动态调整检索策略。

**🔧 技术方法**

使用的技术：大型语言模型（LLM）进行问句分解与图构建，基于 Transformer 的预训练嵌入，Triple-GNN（图神经网络）生成全局引导子图，图检索与结构匹配算法，Beam Search 与阈值扩展策略。

**📊 数据集**

使用数据集：WebQSP 与 CWQ（Freebase 基准），并在公开 KG 上进行实验。

**📈 对比分析**

与 RoG、GNN-RAG、Prompting-based RAG 等基线对比，STEM 在 Hit@1、F1 等指标上均实现了 5%–10% 的提升，尤其在多答案和多跳场景表现突出，达成多项 SOTA 结果。

**⚠️ 局限性**

局限性：对 KG 结构的依赖性强，难以泛化到未见的 KG；在复杂推理中仍可能出现规划偏差导致检索失败；基于阈值的多路检索在某些情形下会增加计算延迟；缺乏真正的零样本、跨 KG 的通用性。

---

## 167. Multi-Agent Consensus as a Cognitive Bias Trigger in Human-AI Interaction

**arXiv ID:** 2604.22277 | [PDF](https://arxiv.org/pdf/2604.22277v1)

**作者:** Soohwan Lee `[一作]` (UNIST), Kyungho Lee `[通讯]` (UNIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实施了一项对127名受试者的实验，比较了三种多智能体配置（多数一致、少数异议、渐进扩散）在信息性与规范性任务中对用户意见、置信度及信任的影响。

**💡 创新点**

创新点在于将多智能体的一致性结构视为认知偏差的信号，系统性揭示其在大语言模型交互中的作用，并提出可在界面层面实现的独立性提示、来源透明化与认知摩擦等偏差缓解策略。

**🔧 技术方法**

采用GPT‑4o三位代理通过分组聊天界面交互，实验设计为split‑plot混合实验，并用线性混合效应模型与主题分析对定量与定性数据进行处理。

**📊 数据集**

数据来源为Prolific平台招募的受试者交互记录与问卷数据，实验生成的日志与自评数据为主，未使用公开的标准数据集。

**📈 对比分析**

通过比较三种配置下的意见变化、置信度、立场翻转等指标，发现多数一致导致最快、幅度最大（g≈0.68–1.06）的意见更新与置信度提升；少数异议和扩散则更新缓慢且置信度提升较慢，表明显著差异。

**⚠️ 局限性**

局限性包括实验规模相对有限、情境仅覆盖两类任务，无法涵盖更复杂的社会动力；代理对齐预设为固定，缺乏动态学习；结果在不同文化、任务和更大规模下的泛化性需要进一步验证。

---

## 168. Fast Neural-Network Approximation of Active Target Search Under Uncertainty

**arXiv ID:** 2604.22254 | [PDF](https://arxiv.org/pdf/2604.22254v1)

**作者:** Bilal Yousuf `[一作]` (Technical University of Cluj-Napoca), Lucian Busoniu `[通讯]` (Technical University of Cluj-Napoca)

**通讯引用:** 7299 | [OpenAlex ID](https://openalex.org/A5058935509)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种使用卷积神经网络近似传统基于PHD滤波器的Active Search (AS) 与Active Search with Intermittent Measurements (ASI) 的多目标搜索规划方法，实现在线规划的显著加速。

**💡 创新点**

创新点包括：① 将AS/ASI 的决策过程映射为多通道网格输入，包含访问历史、平滑后的粒子密度、代理位置和边界信息；② 用CNN直接预测近似最优下一个航点，省去枚举候选航点与在线优化；③ 在不使用显式探索项的情况下，利用PHD出生项实现新目标发现。

**🔧 技术方法**

使用的技术有：概率假设密度（PHD）滤波、基于优化的AS/ASI 规划、卷积神经网络（CNN）与监督学习、Gaussian平滑、粒子滤波、指数平滑航点后处理。

**📊 数据集**

数据集：通过仿真生成的 AS/ASI 演示轨迹，20 条均匀/聚类目标分布下的 60 次训练实验（AS 9120 样本，ASI 3120 样本），以及 20 次测试实验用于性能评估。

**📈 对比分析**

方法比较：在均匀与聚类目标分布下，将CNN 与原始 AS/ASI 在目标检测率、计算时间进行对比。结果显示 CNN 与 AS/ASI 在检测率上几乎相同，早期略慢但后期追赶；计算时间从 AS 的 1.4e‑3 s 降至 1.3e‑4 s，ASI 的 1.475e‑1 s 降至 1.38e‑4 s，提升了 10–100 倍。

**⚠️ 局限性**

局限性：学习到的策略高度依赖任务与目标分布，需要在不同环境或分布变化时重新训练；在离线训练时需大量仿真数据；对新目标分布的泛化能力尚未评估。

---

## 169. When Does LLM Self-Correction Help? A Control-Theoretic Markov Diagnostic and Verify-First Intervention

**arXiv ID:** 2604.22273 | [PDF](https://arxiv.org/pdf/2604.22273v1)

**作者:** Aofan Liu `[一作]` (Peking University), Jingxiang Meng `[通讯]` (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了迭代自我纠错在大语言模型中的效果，并提出了基于两态马尔科夫链的诊断指标。

**💡 创新点**

创新点在于发现接近0%的错误引入率（EIR）是判断自我纠错是否有益的阈值，并通过“先验证再纠错”的提示实现可控停止。

**🔧 技术方法**

使用了马尔科夫链理论、实验评估、验证‑先提示、ASC自适应停止等技术。

**📊 数据集**

使用了 GSM8K、MATH、StrategyQA 三个数学推理与问答数据集。

**📈 对比分析**

与通用迭代、Self‑Refine、Self‑Consistency 等方法对比，发现当EIR低于阈值时可获得+3.4pp提升，而高EIR模型则会下降；Self‑Consistency 在相同计算下表现更好。

**⚠️ 局限性**

局限性包括仅在四轮迭代、单一数据集上评估，假设率稳定，未考虑外部反馈和开放式生成任务。

---

## 170. Fast GPU Linear Algebra via Compile Time Expression Fusion

**arXiv ID:** 2604.22242 | [PDF](https://arxiv.org/pdf/2604.22242v1)

**作者:** Ryan R. Curtin `[一作]`, Conrad Sanderson `[通讯]` (CSIRO)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

做了一个名为Bandicoot的GPU线性代数库，提供与Armadillo相同的API，方便现有CPU代码迁移到GPU。

**💡 创新点**

创新点在于利用编译期模板元编程生成表达式融合的GPU内核，完全不依赖JIT或运行时生成，显著提升效率。

**🔧 技术方法**

使用了C++模板元编程、AST分析、内核骨架与宏生成、CUDA/OpenCL等多种GPU后端技术。

**📊 数据集**

实验数据来自NVIDIA RTX 4090 GPU，使用10k×10k 32位浮点矩阵（约384 GB）以及多种线性代数和神经网络激活表达式。

**📈 对比分析**

通过与PyTorch、TensorFlow、JAX、ArrayFire等主流工具在相同表达式上进行50次运行对比，Bandicoot在大多数案例中最快，并能饱和内存带宽。

**⚠️ 局限性**

局限性包括尚未实现所有Armadillo功能、对GPU硬件依赖、单元素访问效率低、目前仅支持CUDA/OpenCL后端等。

---

## 171. Rethinking AI-Mediated Minority Support in Power-Imbalanced Group Decision-Making: From Anonymity To Authenticity

**arXiv ID:** 2604.22319 | [PDF](https://arxiv.org/pdf/2604.22319v1)

**作者:** Soohwan Lee `[一作]` (UNIST), Kyungho Lee `[通讯]` (UNIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

对比两种基于大型语言模型的少数派支持策略（匿名传递与自动生成对立论点）在层级式组决策中的效果；

**💡 创新点**

发现匿名传递虽提升参与度但降低心理安全与满意度，而自动生成对立论点能提升满意度并减少边缘化，并提出三条针对权力不平衡环境的设计 provocations；

**🔧 技术方法**

利用大型语言模型（LLM）实现 AI 生成对立论点（AIGC）与混合匿名信息（AIMM）两种 AI‑Mediated Communication（AIMC）系统；

**📊 数据集**

使用 96 名韩国参与者组成 24 个四人小组（每组3名高权力成员+1名低权力成员）进行实验数据；

**📈 对比分析**

通过自评量表与行为指标比较三种条件（无 AI、AIGC、AIMM），结果显示 AIGC 降低边缘化并提升满意度，AIMM 提升参与度但显著降低心理安全与满意度；

**⚠️ 局限性**

局限性包括：实验未改变最终决策结果；AI干预在层级环境中可能被主导者忽视或抵制；缺乏跨文化与长期效度验证；

---

## 172. Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation

**arXiv ID:** 2604.22312 | [PDF](https://arxiv.org/pdf/2604.22312v1)

**作者:** Long Cheng `[一作]` (Nvidia), June Yang `[通讯]` (Nvidia)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出一种基于前一步Top‑K预测的 Exact Top‑K 选择算法 GVR，专门优化稀疏注意力解码中的 Top‑K 阶段，显著降低全量扫描次数；

**💡 创新点**

创新点在于利用自回归解码中的时间相关性，将前一步的 Top‑K 作为阈值预测信号，结合 secant‑style 逼近、无投票收集和共享内存精确细化，减少全局扫描次数（从 3–4 次降至 1–2 次），实现可解释且高效的 Exact Top‑K；

**🔧 技术方法**

核心技术包括：预索引统计、secant 迭代阈值搜索、无投票收集、共享内存 2048-bin 直方图、snap 迭代精细化；实现于 NVIDIA Blackwell (sm_100) GPU 的单 CTA 调度；

**📊 数据集**

使用 DeepSeek‑V3.2 长上下文解码日志（LongSeqTasks）和 Synthetic 语料；此外在 TensorRT‑LLM 集成中进行实测；

**📈 对比分析**

与生产版 radix‑select 基线对比，GVR 在单行 512 线程下平均实现 1.88× 单算子加速（最大 2.42×），在全链路 TEP8 最小延迟部署中可使 TPOT 降低 7.52%（更长上下文更明显），并保持模型精度无明显偏差；

**⚠️ 局限性**

局限性：仅在长上下文自回归解码且存在明显时间相关性时有效；单 CTA 设计导致短序列下无加速或略慢；实现依赖 Blackwell GPU，尚未验证跨架构；预填充/多目标推理阶段需改进预测信号。

---

## 173. Exploiting pre-optimized kernels with polyhedral transformations for CGRA compilation

**arXiv ID:** 2604.22297 | [PDF](https://arxiv.org/pdf/2604.22297v1)

**作者:** Yuxuan Wang `[一作]` (EPFL), Giovanni Ansaloni `[通讯]` (EPFL)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合预编译矩阵乘法内核与多维多边形变换的 CGRA 编译框架，实现了对隐藏的矩阵乘法模式的自动识别与高效加速。

**💡 创新点**

创新点在于：① 将多维矩阵乘法抽象为可参数化的专用内核并预编译；② 通过 polyhedral 变换自动将复杂程序中隐含的矩阵乘法子空间提取出来；③ 将内核与剩余代码通过上下文管理无缝集成，实现编译与运行时的高资源利用。

**🔧 技术方法**

采用的技术包括：LLVM/MLIR affine 低级别表示、基于线性化的操作融合、循环拆分与重排、Z3 约束求解、CDFG 目标映射、并行数据共享与地址预计算。

**📊 数据集**

实验数据集主要来自 PolyBench（mmul_base、mmul_relu、3mm 等），以及 PCA、Kalman 滤波器等，矩阵尺寸为 24×24 与 60×60。

**📈 对比分析**

与基线 Compigra（Modulo Scheduling + 循环展开）以及 e‑GPU、基于流水阵列的加速器比较，实验显示该方法在 3.8×–9.1× 的运行时加速上优于全自动编译，且在 4×4 OpenEdgeCGRA 上对 e‑GPU 及流水阵列实现 9.2×–15.1×、4.8×–7.1× 的性能提升。

**⚠️ 局限性**

主要限制包括：对非矩阵乘法部分的性能提升有限；预编译内核需手工实现且受 CGRA 架构特性的约束；对于高度非结构化的计算，polyhedral 变换与内核匹配难度较大。

---

## 174. Inclusive Learning Analytics with Embedded Data Comics: A Conceptual Framework for Public Understanding of AI Ethics

**arXiv ID:** 2604.22322 | [PDF](https://arxiv.org/pdf/2604.22322v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 175. AutoINV: Automated Invariant Generation Framework for Formal Verification on High-Level Synthesis Designs

**arXiv ID:** 2604.22285 | [PDF](https://arxiv.org/pdf/2604.22285v1)

**作者:** Xiaofeng Zhou `[一作]` (Hong Kong University of Science and Technology), Wei Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 36556 | [OpenAlex ID](https://openalex.org/A5100441678)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作构建了AutoINV框架，自动从HLS设计的高层特征中生成帮助断言（helper assertions），并通过动态CTI信息对这些助手进行排名与侧加载，以加速IC3/PDR模型检查过程。

**💡 创新点**

创新点包括①基于HLS常见硬件模式（FIFO、FSM、流水线等）无模板的助手生成方法；②利用上一次超时实验中的CTI出现频率动态评估助手有效性；③将验证出的助手的归纳不变式通过侧加载方式注入主属性的证明过程中，实现迭代加速。

**🔧 技术方法**

采用IC3/PDR（PDR）模型检查核心，Yosys+AIG格式转换，PyVerilog分析RTL，Helper Generator/Ranker模块实现断言生成与排序，Helper Prover模块实现侧加载与多轮证明。

**📊 数据集**

实验数据集包括五个HLS基准：vac-e、vsc-e、mac-e（小规模）以及 AutoSA 生成的两个大型设计 autosa-mm-f 与 autosa-cnn-f。

**📈 对比分析**

与原始IC3/PDR基线在相同硬件环境下对比，AutoINV平均实现 2.23× 的加速（某些案例高达 6.05×），同时减少 SAT 查询和 CTI 量；对比 HARM 生成断言的效率，AutoINV 在生成时间、断言数量及有效率上均明显优于 HARM。

**⚠️ 局限性**

局限性包括：仅针对 IC3/PDR 算法的侧加载，未验证对其他模型检查器的适用性；助手生成依赖 HLS 典型模式，非典型设计可能效果有限；排名机制依赖 CTI 信息，若初始验证未产生 CTI 可能导致助手选择不佳。

---

## 176. A Kinematic Analysis of Palm Degrees of Freedom for Enhancing Thumb Opposability in Robotic Hands

**arXiv ID:** 2604.22283 | [PDF](https://arxiv.org/pdf/2604.22283v1)

**作者:** HyoJae Kang `[一作]` (Korea Institute of Machinery & Materials), Dong Il Park `[通讯]` (Korea Institute of Machinery & Materials)

**通讯引用:** 12491 | [OpenAlex ID](https://openalex.org/A5051699142)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过构建五指机器人手模型并定义七种不同的掌部自由度配置，利用体素化的重叠工作空间量化分析掌部运动对拇指对抗性的影响。

**💡 创新点**

创新之处在于提出了一种不依赖物体形状或接触模型的、基于体素的重叠工作空间量化方法，并系统评估了掌部自由度在保持总自由度不变时的权衡效应。

**🔧 技术方法**

采用正向运动学、体素化离散、重叠体素计数以及可达配置数统计等技术进行定量分析。

**📊 数据集**

使用的是自行构造的五指手模型的几何参数，未使用公开数据集；所有数据均来源于手模型的运动学仿真。

**📈 对比分析**

与基线无掌部自由度的配置以及在保持总自由度不变的情况下的配置进行对比，结果表明掌部自由度显著提升了环指和小指与拇指的重叠工作空间，且在部分案例中提升了可达配置数；相较于仅增加指关节自由度的方案，掌部自由度在总自由度保持时提供了更优的对抗性平衡。

**⚠️ 局限性**

局限性包括仅考虑掌部平行于指尖的运动方向，未考虑手掌厚度、指尖姿态以及实际物理约束，且缺乏实验验证。

---

## 177. Bridging the Long-Tail Gap: Robust Retrieval-Augmented Relation Completion via Multi-Stage Paraphrase Infusion

**arXiv ID:** 2604.22261 | [PDF](https://arxiv.org/pdf/2604.22261v1)

**作者:** Fahmida Alam `[一作]` (University of Arizona), Ellen Riloff `[通讯]` (University of Arizona)

**通讯引用:** 10518 | [OpenAlex ID](https://openalex.org/A5005791318)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新颖的多阶段释义引导关系补全RAG框架，系统性地在多个阶段中整合关系释义，以提高关系补全的性能。

**💡 创新点**

创新点在于通过多阶段的释义引导机制，扩展了关系的词汇覆盖，生成关系感知的摘要，并在生成过程中引导推理，解决了稀疏关系信息的问题。

**🔧 技术方法**

使用了多阶段的释义引导RAG框架，结合了混合检索机制、释义引导的证据聚合和推理过程。

**📊 数据集**

在两个基准数据集（MALT和WikiData5M）上进行了实验，评估了五种大型语言模型（LLMs）的性能。

**📈 对比分析**

与两个强大的RAG基线（SELF-RAG和RECOMP）进行比较，结果显示该框架在长尾设置下的表现优于基线，分别提高了16.0和13.8的准确率，同时保持了较低的计算开销。

**⚠️ 局限性**

限制在于该研究主要集中于英语的关系补全，尚不清楚观察到的长尾鲁棒性是否适用于其他语言。

---

## 178. Microarchitectural Co-Optimization for Sustained Throughput of RISC-V Multi-Lane Chaining Vector Processors

**arXiv ID:** 2604.22314 | [PDF](https://arxiv.org/pdf/2604.22314v1)

**作者:** Weiying Wang `[一作]` (Chinese Academy of Sciences), Zhiwei Zhang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 4863 | [OpenAlex ID](https://openalex.org/A5100347621)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对Ara RISC-V向量处理器在固定资源条件下持续吞吐量不足的问题，本文系统分析了三类微架构瓶颈并提出了协同优化方案；

**💡 创新点**

创新点在于：①构建理想多路链式执行模型作为基准；②将瓶颈细分为内存侧、控制侧、操作数投递三条路径，并针对每条路径提出协同改进；③设计描述驱动的内存前端、next‑vl预取、早期读依赖释放、动态局部发射以及多源转发与双源操作数队列等技术；

**🔧 技术方法**

主要技术包括：描述驱动内存前端、next‑vl预取、早期读依赖释放与动态局部发射控制、操作数多源转发、双源操作数队列、指令调度改进等；

**📊 数据集**

使用一组手工优化的向量算子/BLAS内核作为基准（scal、axpy、dotp、gemv、symv、ger、gemm、trsm、syrk、spmv、dwt）；

**📈 对比分析**

通过周期精确RTL仿真对Baseline Ara与Ara‑Opt进行对比，保持硬件资源不变，Ara‑Opt平均提升1.33×；对常规流式/高吞吐工作负载的gap‑closed比例提升12.2%，单个内核如scal、axpy、ger、gemm分别提升2.41×、1.60×、1.52×、1.42×，显著逼近理想上限；

**⚠️ 局限性**

对归约主导的工作负载（如dotp、gemv、symv）提升有限，瓶颈仍集中在归约尾部和结构序列化，未能在这些场景下实现大幅性能提升。

---

## 179. CLARITY: A Framework and Benchmark for Conversational Language Ambiguity and Unanswerability in Interactive NL2SQL Systems

**arXiv ID:** 2604.22313 | [PDF](https://arxiv.org/pdf/2604.22313v1)

**作者:** Tabinda Sarwar `[一作]` (Oracle Corporation), Katrin Kirchhoff `[通讯]` (Oracle Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 Clarity 框架，自动生成包含多面模糊与不可回答查询的 NL2SQL 评测数据集，并为每个实例提供细粒度的 schema 层面标注。

**💡 创新点**

创新点在于：①支持单轮与多轮对话；②系统化生成多面模糊与不可回答实例；③为每个模糊点提供 pivot 词和对应的 schema 目标组，实现对模糊定位与消解的细粒度评估；④通过 LLM 辅助的规则驱动生成与验证，省去人工标注。

**🔧 技术方法**

核心技术包括：约束驱动的 SQL 解析与目标采样、适应性 Schema 检索、LLM 生成 Pivot 词和对话、LLM 评估器验证、自动数据筛选；同时使用 GPT‑5 生成、GPT‑4o/GPT‑4.1/GPT‑5/LLaMA‑3.3/ Grok‑3 Mini Fast 进行模型评测。

**📊 数据集**

在 Spider 与 BIRD 两个公开 NL2SQL 基准上生成了约 20 万条单轮与多轮实例（单轮 6,392+3,487，双轮 12,098+7,171）。

**📈 对比分析**

通过与现有基准（如 AmbiSQL、MMSQL、BIRD‑INTERACT）对比，发现：LLM 在单轮多面模糊时的 SEM 低至 5%~20%，但 LEM 较高；多轮执行准确率随对话类型变化，冗长对话表现最好；引入带元数据的 few‑shot 示例可显著提升 SEM（约 10%）和执行准确率；然而，模糊定位（MA）仍低于 60%。

**⚠️ 局限性**

局限性包括：①未对 value‑level 模糊进行评测；②多面模糊样本数量受限，因生成与验证成本高；③对话模拟仅涵盖有限的用户行为（如部分帮助或无帮助回应），未覆盖主题漂移等更复杂情境；④仍依赖 LLM 生成，可能带来偏差与可解释性不足。

---

## 180. Knowledge Visualization: A Benchmark and Method for Knowledge-Intensive Text-to-Image Generation

**arXiv ID:** 2604.22302 | [PDF](https://arxiv.org/pdf/2604.22302v1)

**作者:** Ran Zhao `[一作]` (Huazhong University of Science and Technology), Wei Li `[通讯]` (Nanyang Technological University)

**通讯引用:** 98888 | [OpenAlex ID](https://openalex.org/A5100318082)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向知识密集型文本-图像生成的基准数据集 KVBench，并设计了两阶段的 KE-Check 框架来提升模型的科学正确性。

**💡 创新点**

创新点：① 基于高中教材的课程导向基准，采用原子化、可验证的检查表实现细粒度、可解释的评估；② KE-Check 将知识扩充与检查表驱动的纠错分离，显著降低科学幻觉并缩小开源与闭源模型之间的性能差距。

**🔧 技术方法**

技术：使用多模态大型语言模型（如 Qwen2.5‑VL）进行提示扩充、检查表生成与评估；结合扩散式生成模型（如 Stable Diffusion、Flux 等）进行图像合成；实现基于检查表的约束审计与图像编辑。

**📊 数据集**

数据集：1,800 条双语（中英）提示，来自 30+ 权威高中教材，涵盖生物、化学、地理、历史、数学、物理六个学科，伴随参考图像与原子化检查表。

**📈 对比分析**

比较方法：在 KVBench 上对 14 款最新开源与闭源模型进行双轨（简短提示/详细提示）评测，使用基于检查表的准确率作为指标；实验显示闭源模型总体领先 10–20 个百分点，KE-Check 在开源模型上提升约 20–30 个百分点，显著减小性能差距。

**⚠️ 局限性**

局限性：① 依赖 MLLM 生成与评估，受限于其理解与推理能力；② 对复杂逻辑关系与符号精度的捕捉仍不完备，尤其是开源模型在文本可读性、符号准确性和多语种一致性方面表现不佳；③ 基准聚焦高中教材，可能难以直接推广至更高级或跨学科场景。

---

## 181. ReLeVAnT: Relevance Lexical Vectors for Accurate Legal Text Classification

**arXiv ID:** 2604.22292 | [PDF](https://arxiv.org/pdf/2604.22292v1)

**作者:** Ishaan Gakhar `[一作]` (Perssonify), Harsh Nandwani `[通讯]` (Perssonify)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了ReLeVAnT框架，用于法律文件的二分类（相关 vs 非相关）

**💡 创新点**

创新点在于通过对词组级别的对比度评分与文档频率结合进行关键词抽取，无需元数据、结构化信息或大型模型，显著降低计算成本

**🔧 技术方法**

采用n-gram关键词抽取、对比度分数（CSM）与文档频率结合的评分方法，以及浅层全连接神经网络进行分类

**📊 数据集**

使用LexGLUE基准数据集，包括ECtHR A、SCOTUS、EUR-LEX和UNFAIR-ToS子集

**📈 对比分析**

与多数类、始终正类、手工关键词、TF-IDF、BM25等基线比较，取得99.3%准确率、98.7%F1，显著优于传统方法

**⚠️ 局限性**

局限性包括仅针对二分类，缺乏多类/多标签扩展，对跨语言或不同法律体系的泛化能力待进一步验证

---

## 182. Train in Vain: Functionality-Preserving Poisoning to Prevent Unauthorized Use of Code Datasets

**arXiv ID:** 2604.22291 | [PDF](https://arxiv.org/pdf/2604.22291v1)

**作者:** Yuan Xiao `[一作]` (Nanjing University), Zhenyu Chen `[通讯]` (Nanjing University)

**通讯引用:** 7224 | [OpenAlex ID](https://openalex.org/A5100422933)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个功能保留的代码数据集污染框架（FunPoison），通过向代码执行路径注入编译可行的弱用片段来抑制未经授权的 CodeLLM 微调。

**💡 创新点**

在不破坏编译性和功能的前提下，使用可重用的语句级模板和弱用合成实现低比例（10%）的污染；采用安全过滤和类型感知合成保证副作用消除；通过执行路径监督而非表面模板暴露实现毒性。

**🔧 技术方法**

语句提取、编译驱动修复、名称抽象、模板安全过滤、弱用语义合成、执行安全位置选择、LoRA/全参数微调评估，以及静态分析、格式化、LLM 重写等防御测试。

**📊 数据集**

CodeSearchNet Java、HumanEval-X、MBPP 以及 Apache Commons Lang 等数据集，用于构建模板池和评估。

**📈 对比分析**

与 CoProtector、DeadBranchInsertion 等基线对比，10% 污染率下 Pass@1 明显下降，同时保持 100% 编译成功和功能一致；对抗静态分析、格式化、LLM 重写等防御时仍保持显著性能损失。

**⚠️ 局限性**

仅在 Java 上验证，跨语言通用性待研究；插入点稀缺可能限制部署；并非理论上不可移除，部分训练方式和重写攻击可能降低毒性。

---

## 183. Transformer-Based Rhythm Quantization of Performance MIDI Using Beat Annotations

**arXiv ID:** 2604.22290 | [PDF](https://arxiv.org/pdf/2604.22290v1)

**作者:** Maximilian Wachter `[一作]` (Klangio), Michael Heizmann `[通讯]` (Karlsruhe Institute Of Technology)

**通讯引用:** 1028 | [OpenAlex ID](https://openalex.org/A5065874009)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种利用已有节拍信息，对MIDI表演进行精确节奏量化的Transformer模型；

**💡 创新点**

创新之处在于将节拍/下拍注释与MIDI时间同步，构造统一的token化表示，且在模型中使用T5变压器、数据增强与多拍号泛化，实现了高精度的节奏量化；

**🔧 技术方法**

采用T5 Transformer架构、MIDI分词器、beat预量化、移调/删除/时值噪声等数据增强技术、交叉熵损失与Adafactor优化，以及Beam Search解码；

**📊 数据集**

使用ASAP钢琴MIDI与MusicXML数据集、Leduc爵士吉他MIDI数据集（转换为MIDI）以及MUSTER评估集进行实验；

**📈 对比分析**

通过Onset F1、note value accuracy、MUSTER的ϵ_onset/ϵ_offset等指标进行比较，模型在ASAP上Onset F1达97.3%，note value 83.3%，ϵ_onset 12.30，ϵ_offset 28.30，优于多种基线和商业软件；

**⚠️ 局限性**

局限在于仅支持到1/32分音符的量化，无法处理非标准或不规则拍号，假设输入输出一一对应，且缺乏多声部与音色信息，需要针对不同乐器单独训练。

---

## 184. Beyond Chain-of-Thought: Rewrite as a Universal Interface for Generative Multimodal Embeddings

**arXiv ID:** 2604.22280 | [PDF](https://arxiv.org/pdf/2604.22280v1)

**作者:** Peixi Wu `[一作]` (Tencent Inc.), Xiaoyan Sun `[通讯]` (Institute of Artificial Intelligence, Hefei Comprehensive National Science Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了RIME框架，利用检索友好的重写驱动方式替代传统链式思考，实现高效可扩展的多模态嵌入；

**💡 创新点**

创新点在于：①将链式思考改为无冗余的重写过程；②通过跨模态对齐（CMA）将生成式与判别式嵌入空间对齐，实现互检；③使用强化学习（Refine‑RL）将判别式嵌入作为语义锚点指导重写优化；

**🔧 技术方法**

采用多模态大语言模型（如Qwen2‑VL‑2B/7B‑Instruct）为基础，联合监督微调、跨模态InfoNCE、Group Relative Policy Optimization等技术；

**📊 数据集**

训练使用约150万条来自LLaVA‑Hound、ViDoRe、VisRAG等多模态数据集；评测采用MMEB‑V2、MRMR、UVRB等公开基准；

**📈 对比分析**

在MMEB‑V2上RIME‑7B取得68.6分，超越UME‑R1‑7B 4.1点；在MRMR上得分50.2，领先同类模型2–14点；在UVRB上得分55.6，超过Unite‑7B 1.8点，整体表现为最优；

**⚠️ 局限性**

主要局限是生成式重写仍产生显著推理延迟，难以满足大规模实时检索需求，需要进一步将推理过程转化为隐式潜在思考。

---

## 185. CAGE-SGG: Counterfactual Active Graph Evidence for Open-Vocabulary Scene Graph Generation

**arXiv ID:** 2604.22274 | [PDF](https://arxiv.org/pdf/2604.22274v1)

**作者:** Suiyang Guang `[一作]` (Institute of Intelligent Vision and Embodied Cognition), Siyuan Chen `[通讯]` (Institute of Intelligent Vision and Embodied Cognition)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于因果逆向检验的开源词汇场景图生成框架，将关系预测转为关系验证

**💡 创新点**

引入证据基础的关系拆分与逆向检验机制，通过对视觉、几何、上下文证据的因果干预减少语言偏差并提升可信度

**🔧 技术方法**

使用视觉‑语言联合提议器、关系条件证据编码器、关系类型分布路由、逆向检验模块以及图级偏好优化等技术

**📊 数据集**

在Visual Genome（VG150）闭集/开源词汇拆分、OV‑VG自定义开源词汇分割以及Panoptic Scene Graph（PSG）面向分割的评测数据集上进行实验

**📈 对比分析**

与Motifs、VCTree、Transformer、BGNN、PE‑Net、LLM4SGG、Pix2Graphs等方法对比，在Recall@K、meanRecall@K、U‑mR、CF‑Acc等指标上均实现显著提升，尤其在未见谓词和因果检验指标上表现突出

**⚠️ 局限性**

对逆向干预需手工设计证据操作且计算开销相对较大，且在极度稀缺或长尾场景下仍受限于视觉特征质量

---

## 186. Learning Control Policies to Provably Satisfy Hard Affine Constraints for Black-Box Hybrid Dynamical Systems

**arXiv ID:** 2604.22244 | [PDF](https://arxiv.org/pdf/2604.22244v1)

**作者:** Aayushi Shrivastava `[一作]` (University of California Berkeley), Negar Mehr `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种基于受限深度强化学习的策略网络，利用双缓冲区域的仿射可回避性来保证黑盒混合动力学系统在闭环下满足线性安全约束。

**💡 创新点**

创新点在于：①为混合系统设计了兼顾离散跳跃的安全缓冲区；②使用切换演员网络在不同状态空间分别部署仿射与非线性策略；③通过理论证明给出仿射缓冲区满足的充要条件，实现无模型安全保证。

**🔧 技术方法**

采用了POLICE深度神经网络理论来强制策略在缓冲区内为仿射；两阶段训练（先训练非线性基准策略，再冻结并加入仿射缓冲区训练）；使用安全约束的惩罚项与奖励设计。

**📊 数据集**

在两个典型混合系统上验证：约束摆（含长度重置）与垒球投掷器（碰撞重置）模拟环境。

**📈 对比分析**

与软约束方法CPO和基于学习的控制障碍函数PPO‑Barrier进行对比；实验显示本文方法在保持任务性能的同时实现100%安全约束满足，优于两种基线。

**⚠️ 局限性**

局限性包括：仅适用于已知仿射重置映射的系统；仅支持单一约束，扩展到多约束或高维系统需进一步研究；仿射缓冲区设计对非线性动态的近似要求较高。

---

## 187. OccDirector: Language-Guided Behavior and Interaction Generation in 4D Occupancy Space

**arXiv ID:** 2604.22240 | [PDF](https://arxiv.org/pdf/2604.22240v1)

**作者:** Zhuding Liang `[一作]` (University of Macau), Jianbing Shen `[通讯]` (University of Macau)

**通讯引用:** 16197 | [OpenAlex ID](https://openalex.org/A5023184215)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为OccDirector的文本驱动4D占用空间生成框架，能根据自然语言指令生成多代理交互与场景动态；

**💡 创新点**

①从几何先验转向纯文本控制，消除轨迹设计瓶颈；②引入VLM驱动的Spatio-Temporal MMDiT并配备Spatio-Temporal Separated Attention与历史前缀锚定，弥合语义‑时空鸿沟；③构建包含多层级语言描述的大规模占用空间数据集；

**🔧 技术方法**

利用冻结的Vision‑Language Model提取文本语义，Token Refiner对齐语义与空间特征；MMDiT双流Transformer结构，使用STSA分离空间与时间注意力；历史前缀锚定策略保证长时序一致性；训练采用连续正则化流（CNF）与Flow Matching；

**📊 数据集**

新构建的“OccLang”数据集，包含单帧静态布局、行为约束片段以及多代理交互场景，覆盖不同风险等级与时间尺度；

**📈 对比分析**

与DOME、COME、DynamicCity等基于几何先验的方法以及DynamicCity + CLIP/T5文本适配版本进行对比；在单帧IS/FID/KID/Precision/Recall和视频FVD/Precision/Recall以及VLM评估指标上，OccDirector均表现出色，FVD显著低于基线，VLM对齐得分提升约0.3分，说明文本与生成结果匹配更好；

**⚠️ 局限性**

仍受限于训练数据的多样性与复杂性，极端安全关键场景的覆盖不足；VLM的文本理解仍可能产生误解或偏差；模型对长时序极端事件的推断仍不够稳健；

---

## 188. Tell Me Why: Designing an Explainable LLM-based Dialogue System for Student Problem Behavior Diagnosis

**arXiv ID:** 2604.22237 | [PDF](https://arxiv.org/pdf/2604.22237v1)

**作者:** Zhilin Fan `[一作]` (Beijing Normal University), Yu Lu `[通讯]` (Beijing Normal University)

**通讯引用:** 16473 | [OpenAlex ID](https://openalex.org/A5101645258)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套基于微调LLM的可解释对话系统，用于学生问题行为诊断，并为推荐的干预策略生成基于对话证据的自然语言解释。

**💡 创新点**

首次提出层级归因方法（先定位最重要的对话轮次，再在该轮内按必要性和充分性评估句子）实现对话证据定位，并将证据转化为解释，显著提升教师对系统的信任。

**🔧 技术方法**

使用LoRA微调Qwen2.5-3B-Instruct、xAI归因（Turn-level attribution + Drop+Hold）、自然语言生成、BERTScore、macro‑F1、Hit@1/3/5、MRR等技术。

**📊 数据集**

采用Chen等人专家标注的诊断对话语料（3636例训练，409例测试）及专家标注的支持句子作为评估基准。

**📈 对比分析**

与Drop+Hold、Leave‑one‑out、GradNorm、Similarity等基线对比，层级归因方法在Hit@1/3/5和MRR上均取得显著优势（0.778/0.856/0.945/0.803），并在预试验中提升教师信任度。

**⚠️ 局限性**

样本量有限（22名预备教师）、仅评估自报信任、解释仅覆盖干预策略、归因聚焦单一轮次可能忽略跨轮证据，未来需扩大样本、评估决策效果及多轮解释。

---

## 189. A Brain-Inspired Deep Separation Network for Single Channel Raman Spectra Unmixing

**arXiv ID:** 2604.22324 | [PDF](https://arxiv.org/pdf/2604.22324v1)

**作者:** Gaoruishu Long `[一作]` (Nankai University), Xiaolin Hu `[通讯]` (Tsinghua University)

**通讯引用:** 19674 | [OpenAlex ID](https://openalex.org/A5004579631)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对单通道拉曼光谱混合物的盲源分离问题，提出并实现了一种新的深度分离网络RSSNet，能够从噪声混合光谱中准确恢复成分光谱。

**💡 创新点**

创新点在于将语音单通道分离技术（如Conv‑TasNet、DPRNN、TDANet）的脑启发式顶层注意力结构迁移到光谱域，构建可处理数千种潜在成分、噪声鲁棒的分离网络。

**🔧 技术方法**

使用了1D卷积编码器-分离器-解码器框架，包含双路径RSSNet块、Top‑Down Attention模块、深度可分离卷积以及SI‑SNR评价指标的深度学习技术。

**📊 数据集**

使用了两个合成单通道混合光谱数据集（RRUFF‑2Mix与UNIPR‑2Mix）以及21个真实矿物混合样本进行训练与验证。

**📈 对比分析**

与稀疏回归、几何统计与混合学习等传统方法以及Conv‑TasNet、DPRNN、AFRCNN、TDANet等语音分离网络比较，RSSNet在合成数据上SI‑SNRi提升4+ dB，在真实样本上平均SI‑SNR达11.7 dB，显著优于对比方法。

**⚠️ 局限性**

局限在于目前只能处理两种成分的混合，且对未出现过的未知物质泛化能力仍有限，未来需扩展到多成分混合和开放世界场景。

---

## 190. From Skills to Talent: Organising Heterogeneous Agents as a Real-World Company

**arXiv ID:** 2604.22446 | [PDF](https://arxiv.org/pdf/2604.22446v1)

**作者:** Zhengxu Yu `[一作]` (HUAWEI Noah's Ark Lab), Jun Wang `[通讯]` (University College London)

**通讯引用:** 46114 | [OpenAlex ID](https://openalex.org/A5100384686)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 OneManCompany（OMC）框架，为多智能体系统引入组织层，实现动态招聘、任务分解、执行与评审的闭环；

**💡 创新点**

创新点包括：① AI 组织概念与 Talent‑Container 架构，② 六类统一接口实现执行与平台解耦，③ Talent 市场支持可验证代理包，④ Explore‑Execute‑Review（E²R）树搜索提供正式终止与无死锁保证，⑤ 结构化的自我演化管线（个人自省、项目回顾、HR 绩效管理）；

**🔧 技术方法**

技术方案涵盖大型语言模型（Claude、Gemini、Gemini 3.1 等）、多种运行时（LangGraph、Claude Code、脚本容器）、类型化组织接口、DAG 任务调度、有限状态机、自动招聘与评估流程；

**📊 数据集**

使用 PRDBench 软件开发项目数据集进行量化评测，并在内容生成、游戏开发、音频书籍制作、学术调研等四个跨域案例中进行案例实验；

**📈 对比分析**

在 PRDBench 零-shot 设定下与单代理及现有多代理基线比较，OMC 获得 84.67% 的成功率，较基线提升约 15.48%（相当于 84.67% vs 69.19% 最高基线），总成本约 345.59 美元，约 6.91 美元/任务；

**⚠️ 局限性**

局限性包括：评测仅覆盖 PRDBench 代码开发任务，跨域案例未系统化量化；自我演化机制未单独消融验证；多智能体协同导致成本上升，适用范围对简单任务受限；需进一步扩展 Talent 市场与大规模实验验证。

---

## 191. AgentSearchBench: A Benchmark for AI Agent Search in the Wild

**arXiv ID:** 2604.22436 | [PDF](https://arxiv.org/pdf/2604.22436v1)

**作者:** Bin Wu `[一作]` (University College London), Emine Yilmaz `[通讯]` (University College London)

**通讯引用:** 4760 | [OpenAlex ID](https://openalex.org/A5076265623)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 AgentSearchBench，一套基于真实生态系统、涵盖近10,000个 agent 的大规模检索与重排序基准，支持可执行任务查询和高层任务描述；

**💡 创新点**

首次通过执行结果定义 agent 相关性，揭示文本相似度与实际性能之间的显著差距，并提出轻量化的行为探测（使用案例索引与执行感知探测）提升排名质量；

**🔧 技术方法**

结合传统检索方法（BM25、ColBERT、BGE、ToolRet 等）与交叉编码、工具专用重排序（Tool‑Rank）以及 LLM‑as‑Judge 评估器，并在此基础上实现执行感知的探测与重排序；

**📊 数据集**

使用从 GPT Store、Google Cloud Marketplace、AgentAI 等公开平台收集的约9,760个 agent，构造3,211 个任务（2,452 单机可执行、500 多机可执行、259 高层描述），共完成66,740 次 agent 执行；

**📈 对比分析**

对检索与重排序方法进行 NDCG、Precision、Recall、Completeness 等指标评估，结果表明基于文本相似度的检索与重排序无法完全捕捉执行性能，工具感知检索表现最优，但整体分数仍偏低；加入轻量化探测后 NDCG@5 提升 5–10% 左右；

**⚠️ 局限性**

局限在于依赖 LLM‑as‑Judge 的自动评估，可能受模型偏差影响；实验规模虽大但仍受候选集和任务覆盖率限制；行为探测虽轻量但仍需执行成本；对多模态或非文本 agent 文档的支持尚不充分。

---

## 192. CognitiveTwin: Robust Multi-Modal Digital Twins for Predicting Cognitive Decline in Alzheimer's Disease

**arXiv ID:** 2604.22428 | [PDF](https://arxiv.org/pdf/2604.22428v1)

**作者:** Bulent Soykan `[一作]` (University of Toledo), Laura J. Brattain `[通讯]` (University of Central Florida)

**通讯引用:** 979 | [OpenAlex ID](https://openalex.org/A5026644166)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种CognitiveTwin数字孪生框架，用多模态长期数据预测阿尔茨海默病个体认知衰退轨迹。

**💡 创新点**

创新点在于将Transformer多模态融合与Deep Markov Model时空动态相结合，兼顾精度、公平性与对MNAR缺失数据的鲁棒性。

**🔧 技术方法**

采用Transformer自注意力进行多模态融合，Deep Markov Model进行概率状态空间建模，并使用ELBO+MSE联合训练。

**📊 数据集**

使用ADNI TADPOLE 1666名患者的多模态纵向数据（认知评分、MRI、PET、CSF、生物标志物、遗传学）。

**📈 对比分析**

与传统LSTM、CNN‑LSTM、Transformer、图神经网络等基线比较，CognitiveTwin在24个月MMSE MAE为1.619，AUROC 0.912，显著优于基线，MNAR下误差仅增0.3%。

**⚠️ 局限性**

主要局限在于训练数据来源单一、缺乏多中心多样性，计算资源需求高，且对实验室和扫描协议的标准化要求苛刻。

---

## 193. A Model-Driven Approach to Database Migration with a Unified Data Model

**arXiv ID:** 2604.22415 | [PDF](https://arxiv.org/pdf/2604.22415v1)

**作者:** María J. Ortín `[一作]` (Universidad de Murcia), Jesus García-Molina `[通讯]` (Universidad de Murcia)

**通讯引用:** 1411 | [OpenAlex ID](https://openalex.org/A5080461616)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种通用的数据库迁移框架，利用统一数据模型 U‑Schema 作为中间表示，支持源模型与目标模型的自动化模式转换和数据迁移，并通过 trace 机制实现解耦的实例级迁移。

**💡 创新点**

创新点在于：① 通过 U‑Schema 只需编写两套源-中间和中间-目标的映射规则，显著降低迁移路径数量；② 在 schema 转换过程中生成细粒度的 trace 链接，直接驱动数据迁移，避免额外的 ETL 编写；③ 通过 Orion 进行可插拔的 schema 定制，使得目标模式可在不改写映射规则的前提下满足业务需求。

**🔧 技术方法**

技术手段包括：模型驱动工程（MDE）与 EMF/Xtend 实现模型到模型（m2m）与模型到文本（m2t）转换；U‑Schema 统一元模型；trace 记录与查询；基于 U‑ReaderAdapter 的数据库无关读取；以及 Orion 语言用于 schema 进化。

**📊 数据集**

使用两个数据集验证：① 合成 Music Streaming 示例（包含弱表、关联表、复合键等多种结构），规模从 10k 到 1M 行；② 真实 Northwind 基准数据库（约 1k 行），以检验映射的通用性。

**📈 对比分析**

比较方法：单元测试 + 集成测试验证映射规则；生成文档模式与预期模型对比；Round‑trip 重构评估结构保持率（precision/recall/F1 ≈ 0.97+）；语义验证通过 SQL 与 MongoDB 等价查询结果比较；性能评估显示迁移时间随数据量线性增长，单实例 1M 行可在约 5–10 分钟内完成，且支持批量写入。

**⚠️ 局限性**

局限性：① 评估仅聚焦于关系型→文档型迁移，其他 NoSQL 模型尚未充分验证；② 迁移决策基于固定规则，未结合工作负载或统计信息进行自适应优化；③ 未覆盖应用代码迁移与查询重写；④ 统一模型在极度复杂的多模型场景下可能产生冗余结构，需进一步优化。

---

## 194. Enhancing a gamified tool for UML modeling education

**arXiv ID:** 2604.22400 | [PDF](https://arxiv.org/pdf/2604.22400v1)

**作者:** Giacomo Garaccione `[一作]` (Politecnico di Torino), Luca Ardito `[通讯]` (Politecnico di Torino)

**通讯引用:** 957 | [OpenAlex ID](https://openalex.org/A5040165772)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在原有的UMLegend工具基础上，新增了对用例图的支持，改进了游戏机制和评估引擎，并将系统架构改造成模块化，以便未来加入更多建模语言；同时计划在2026年的软件工程课程中进行纵向研究，以评估持续使用游戏化建模平台的学习效果。

**💡 创新点**

创新点主要包括：①将用例图评估逻辑集成到自动评估器中，支持多方案比较；②通过可定制的长期机制（等级、经验、头像、排行榜、完成奖励）提升学生持续参与度；③采用模块化设计，易于扩展到其他建模语言或软件工程主题；④改进反馈机制（错误列表、图形高亮、评估回顾）使学习过程更直观。

**🔧 技术方法**

技术手段涵盖：Apollon建模平台与其APIs；Levenshtein字符串相似度用于元素匹配；前端技术（JavaScript/React）实现游戏化交互；后端评估服务使用Node.js；模块化架构与插件化设计便于后续扩展；游戏机制实现包括积分、等级、头像、排行榜、Boss角色、完成奖励等。

**📊 数据集**

已使用的数据集：Politecnico di Torino 2025年的实验数据，包含280名学生的类图作业及其评估结果；新版本计划收集2026年课程期间的学生作业、成绩与游戏化交互日志，作为纵向评估的基础数据。

**📈 对比分析**

比较方法：先前的单次实验将游戏化组与非游戏化组在图形正确率、错误率和学生满意度上进行对比，结果显示游戏化组在正确率上有显著提升；新版本将采用纵向研究，记录学生在整个课程期间的正确率、完成率、等级提升和最终考试成绩，以评估持续使用游戏化工具的学习成效。

**⚠️ 局限性**

局限性：①原版仅支持类图，后续改版仍缺乏对更广泛建模语言的全面支持；②评估引擎在初期只考虑了基本语法和语义错误，复杂交互未覆盖；③实验为单次，缺乏长期效果验证；④缺乏对不同学科背景学生的跨学科评估；⑤工具的游戏化机制可能对某些学生产生过度竞争压力，需要进一步调优。

---

## 195. Reelay: Online Temporal Logic Monitoring Framework

**arXiv ID:** 2604.22384 | [PDF](https://arxiv.org/pdf/2604.22384v1)

**作者:** Dogan Ulus `[一作]` (Boğaziçi University), Dogan Ulus `[通讯]` (Boğaziçi University)

**通讯引用:** 391 | [OpenAlex ID](https://openalex.org/A5054455487)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一个统一的在线时序逻辑监控框架 Reelay，支持 LTL、MTL、STL 以及它们的鲁棒性和一阶量化扩展，并兼容离散与连续时间、delta 编码数据。

**💡 创新点**

创新点在于将多种时序逻辑形式化归纳为单一计算模型，使用同步数据流计算图实现高效执行，并通过编译时模板特化针对不同消息格式进行优化；同时实现了无状态的 BDD 一阶量化和鲁棒性监控。

**🔧 技术方法**

技术包括：C++ header‑only 库、Python 绑定、同步计算图（sequential network）、区间集与最小最大代数、BDD、编译时模板特化、delta 编码、稀疏消息处理。

**📊 数据集**

使用 Timescales、DejaVu、RTAMT 等公开基准日志（多达百万条消息）和 CSV/JSON 格式的数据集。

**📈 对比分析**

对比方法：与 DejaVu、MonPoly、RTAMT、rtamt、Dejavu 等现有工具在同一硬件（Intel Xeon 3.8GHz）下进行。结果显示 Reelay 在离散时间 JSON 监控下单条消息处理时间仅几百 ns，二进制结构更快；鲁棒性监控下保持常数时间；一阶监控中对未定时属性优于 Dejavu，对定时属性扩展更稳定，未出现超时。

**⚠️ 局限性**

局限性：目前仅支持过去时序操作符；鲁棒性与一阶量化不兼容；缺乏多属性并行监控和去中心化监控；对未来时序和更广泛时间模型的支持有限。

---

## 196. OCC: Physical-Layer Assisted Congestion Control for Real-Time Communications

**arXiv ID:** 2604.22383 | [PDF](https://arxiv.org/pdf/2604.22383v1)

**作者:** Yufan Zhuang `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 54080 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为OCC的跨层物理层信息驱动的实时通信（RTC）速率控制框架，旨在解决移动网络中低延迟与高比特率的矛盾；

**💡 创新点**

创新点包括：① 采用帧感知物理层测量，消除RTC流突发性导致的误差；② 针对APP限制的流实现带宽估计增量探测；③ 对物理层带宽结果进行最小值滑动窗口平滑，降低编码器溢出与尾延迟；并能在链路瓶颈变化时自动切换至传统端到端控制；

**🔧 技术方法**

核心技术包括：物理层资源块（PRB）和调制编码（MCS）实时采样、基于帧间隔的ABW估计、应用层限幅检测与调节、最小值窗口平滑、O-RAN开放架构下的反馈机制、以及与WebRTC的无缝集成；

**📊 数据集**

使用了UGC标准视频数据集中的2K HDR（平均13.9 Mbps）和1080p讲座视频（平均7.6 Mbps）进行实时编码与传输实验；

**📈 对比分析**

与两类基线（端到端的GCC和物理层增强的PBE‑CC）在多种网络环境（自由空间、障碍物、移动场景）下对比。OCC在网络延迟上降低13%–68%，视频有效比特率提升1.2×–3.5×，并在动态瓶颈、拥塞和用户移动等场景下保持稳定；

**⚠️ 局限性**

局限性包括：对多重同端RTC流竞争不适用；需要在基站端实现物理层信息采集与反馈，可能受限于商用基站软件更新；编码器调度与延迟仍受应用层控制，超大编码器延迟会导致性能退化；

---

## 197. Efficient Diffusion Distillation via Embedding Loss

**arXiv ID:** 2604.22379 | [PDF](https://arxiv.org/pdf/2604.22379v1)

**作者:** Jincheng Ying `[一作]` (Guangdong University of Finance and Economics), Yinhao Xiao `[通讯]` (Guangdong University of Finance and Economics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种嵌入损失（Embedding Loss, EL）辅助损失，用于高效蒸馏扩散模型为单步/少步生成器，显著提升生成质量并加速训练；

**💡 创新点**

创新点在于利用一组随机初始化、不同架构的特征提取器计算最大均值偏差（MMD），在特征空间实现分布匹配，既避免了回归预生成数据的开销，也克服了GAN对抗训练的不稳定性；

**🔧 技术方法**

采用最大均值偏差（MMD）作为损失函数，配合随机多网络特征投影；集成进现有的分布匹配蒸馏框架（如DMD、DI、SiD2A）以及轨迹保持框架（如一致性蒸馏）；

**📊 数据集**

在CIFAR‑10、ImageNet（64×64/512×512）、AFHQ‑v2、FFHQ等四大公开数据集上进行实验；

**📈 对比分析**

与原始蒸馏方法及其他辅助损失（回归损失、GAN损失）对比，EL在保持相同或更少训练步骤/批量下实现FID显著下降（如CIFAR‑10无条件1.475、条件1.380），并将训练迭代次数缩短最多80%；

**⚠️ 局限性**

局限性包括：需要额外的特征网络（尽管计算量小）；对超参数（如网络数量、初始化方式）敏感；在极大分辨率或极大数据集规模下的可扩展性尚未彻底验证。

---

## 198. Adaptive vs. Static Robot-to-Human Handover: A Study on Orientation and Approach Direction

**arXiv ID:** 2604.22378 | [PDF](https://arxiv.org/pdf/2604.22378v1)

**作者:** Federico Biagi `[一作]` (University of Modena and Reggio Emilia), Luigi Biagiotti `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 1622 | [OpenAlex ID](https://openalex.org/A5070487792)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了基于实时手部姿态估计的自适应机器人‑人类物品交接系统，并与静态交接做对比实验。

**💡 创新点**

创新在于同时实时调整交接姿态和方向，结合任务导向对齐与可预测轨迹，解决了传统自适应系统导致运动不可预测的问题。

**🔧 技术方法**

采用 MediaPipe+FrankMocap+Manotorch 进行手部3D姿态估计，使用 Bézier 曲线+SLERP 轨迹生成，Pinocchio IK 验证，ROS2 控制。

**📊 数据集**

使用14名右手使用者与两种常见物体（杯子、智能手机）进行实验，采集NEON眼动（眨眼率）和NASA‑TLX、信任量表。

**📈 对比分析**

采用 within‑subject 对比，结果显示自适应交接在眨眼率、主观工作负荷降低，信任度提升，性能提升显著。

**⚠️ 局限性**

局限包括样本量小、物体种类有限、视觉模型推理速度与深度精度受限、未验证更大尺度任务。

---

## 199. PoseFM: Relative Camera Pose Estimation Through Flow Matching

**arXiv ID:** 2604.22350 | [PDF](https://arxiv.org/pdf/2604.22350v1)

**作者:** Dominik Kuczkowski `[一作]` (University of Helsinki), Laura Ruotsalainen `[通讯]` (University of Helsinki)

**通讯引用:** 1306 | [OpenAlex ID](https://openalex.org/A5021445018)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出PoseFM框架，将单目视觉里程计重新定义为生成式任务，利用Flow Matching学习相机运动的概率分布而非单一回归估计；

**💡 创新点**

首创在视觉里程计中应用Flow Matching，实现对相机运动的不确定性建模，并通过光流作为视觉引导实现全局运动推断；

**🔧 技术方法**

使用Flow Matching、条件流匹配(CFM)与ODE求解、光流网络(PWCNet/WAFT)、SE(3)李代数表征、ResNet特征提取与多头预测；

**📊 数据集**

在TartanAir进行训练和评测，KITTI与TUM‑RGBD作为跨域测试集；

**📈 对比分析**

与TartanVO、DytanVO、CUVO、ORB‑SLAM3、DPVO、MambaVO等基准进行比较；在TartanAir上平均ATE 3.08 m，超过TartanVO；在KITTI多序列上位居前列；在TUM‑RGBD上3/9场景表现最佳；并通过多重采样自然获得不确定性估计；

**⚠️ 局限性**

依赖光流估计，光流误差会影响性能；当前为单帧推断，缺乏多帧优化；在KITTI上WAFT版表现不佳；生成式推断需要ODE求解，计算开销相对较高。

---

## 200. Horizontal SCA Attacks on Binary kP Algorithms using Chevallier-Mames Atomic Blocks

**arXiv ID:** 2604.22429 | [PDF](https://arxiv.org/pdf/2604.22429v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 201. Dynamic Planar Graph Isomorphism is in DynFO

**arXiv ID:** 2604.22365 | [PDF](https://arxiv.org/pdf/2604.22365v1)

**作者:** Samir Datta `[一作]` (Chennai Mathematical Institute), Thomas Zeume `[通讯]` (Ruhr University Bochum)

**通讯引用:** 375 | [OpenAlex ID](https://openalex.org/A5023740993)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在保持平面图可持续更新的前提下，提出了动态维护平面图同构关系的完整算法框架，包括3连通、三连通、双连通及连通分量的层级结构维护；

**💡 创新点**

创新点在于将Tutte嵌入与SMW公式结合，用首一阶公式维护逆矩阵信息，并在同构判定中利用可更新的同伦路径与距离关系，构建可在FO可维护的动态树同构算法；

**🔧 技术方法**

主要技术包括SPQR树（SPQR/SPQR树）、Tutte嵌入、线性代数中的Sherman–Morrison–Woodbury公式、同伦路径和距离关系维护、分量合并与拆分操作；

**📊 数据集**

该工作为理论算法，未使用具体实验数据集；

**📈 对比分析**

方法通过理论证明显示可在DSPACE(log^i(n))空间内维护，性能优于传统需要更高复杂度的静态同构判定算法；

**⚠️ 局限性**

局限性主要是仅适用于保持平面性的图，且维护过程中需大量素数逆矩阵信息，更新步长受限导致在某些极端操作中可能出现逆矩阵不存在的情况。

---

## 202. Gamifying Architectural Governance to Reduce Organizational Coupling in Microservice Systems

**arXiv ID:** 2604.22454 | [PDF](https://arxiv.org/pdf/2604.22454v1)

**作者:** Xiaozhou Li `[一作]` (Free University of Bozen-Bolzano), Xiaozhou Li `[通讯]` (Free University of Bozen-Bolzano)

**通讯引用:** 2035 | [OpenAlex ID](https://openalex.org/A5100693396)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于游戏化的微服务架构治理框架，利用持续挖掘仓库数据检测组织耦合，并通过积分、徽章、排行榜和架构改进任务等游戏元素激励开发者维护服务边界，减少组织耦合；

**💡 创新点**

将游戏化与组织耦合度指标结合，实现从被动诊断到主动行为干预的闭环；设计多层级激励机制和任务驱动的改进方式；强调通过行为反馈持续提升架构质量；

**🔧 技术方法**

仓库挖掘技术、组织耦合度与团队凝聚度计算、游戏化引擎（积分、徽章、排行榜、提示），并基于目标设定与自我决定理论的心理学模型；

**📊 数据集**

公开的微服务开源项目，如 SockShop、sShop、Spinnaker 等，使用提交、PR、依赖信息等仓库数据；

**📈 对比分析**

通过基线分析、仿真模拟和人机实验进行评估；与透明指标、政策驱动治理等基线对比；实验表明能降低OC、提升团队凝聚度，但具体效果需在真实环境中验证；

**⚠️ 局限性**

难以准确区分必要与有害的跨服务贡献；模拟假设可能与真实开发者行为不符；开源数据不一定能代表工业场景；游戏化可能产生竞争过度、游戏化行为和压力；集成到现有工作流程时可能遭遇抵触和使用障碍。

---

## 203. R2Code: A Self-Reflective LLM Framework for Requirements-to-Code Traceability

**arXiv ID:** 2604.22432 | [PDF](https://arxiv.org/pdf/2604.22432v1)

**作者:** Yifei Wang `[一作]` (City University of Hong Kong), Yishu Li `[通讯]` (Hong Kong Metropolitan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于LLM的需求-代码追溯框架R2Code，用分层语义分解与双向对齐网络实现结构化语义匹配，并加入自我反思一致性校验与动态上下文自适应检索以提升追溯质量。

**💡 创新点**

创新点在于将需求与代码映射为四层语义结构，通过双向对齐网络实现跨层语义匹配；利用LLM生成解释并自我校验的方式进行置信度校准；以及根据需求复杂度动态调整检索粒度以降低推理成本。

**🔧 技术方法**

采用DeepSeek-V3.1-Terminus大模型完成语义分解、对齐与一致性评估；配合句子变换器（sentence‑transformers/all‑mpnet‑base‑v2）做稠密检索；实现了结构化JSON输出和自适应检索机制。

**📊 数据集**

使用五个公开需求-代码追溯数据集：iTrust、eTour、SMOS、eANCI（Java）和RETRO.NET（C#）共计超过5,000条需求和10,000条代码实体。

**📈 对比分析**

与传统IR基线（BM25、TF‑IDF、VSM、LSI、WMD）、稠密检索及标准RAG+LLM管道对比，R2Code平均提升F1约7.4%，在iTrust上最高提升14.1%，同时将token使用量减少41.7%，推理成本降低近38%。

**⚠️ 局限性**

局限性包括对大型LLM的依赖导致推理费用和延迟，分层语义分解对模糊或高度分散实现的需求支持不足，且在快速演进的代码库中需实现增量摘要更新以保持上下文新鲜。

---

## 204. How Hard is it to Decide if a Fact is Relevant to a Query?

**arXiv ID:** 2604.22422 | [PDF](https://arxiv.org/pdf/2604.22422v1)

**作者:** Meghyn Bienvenu `[一作]` (University of Bordeaux), Pierre Lafourcade `[通讯]` (University of Bordeaux)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

研究了在数据库和本体驱动查询中，判断事实是否为查询支持（relevance）问题的复杂性，并给出了多种结构性限制下的结果。

**💡 创新点**

创新点在于引入自连接宽度(self-join width)和交互宽度(interaction width)两种度量，证明在这些度量有界时relevance与查询评估的复杂度相同。

**🔧 技术方法**

使用逻辑与图同构技术、可归约性分析、树宽与自连接宽度的定义以及DL‑Lite 的 canonical model 构造。

**📊 数据集**

论文主要是理论工作，没有使用具体实验数据集。

**📈 对比分析**

通过复杂度归约与构造例子，将 relevance 问题与查询评估比较，证明在一般情况下是 NP/PSPACE‑完全，而在自连接或交互宽度受限时降低到 P 或 LOGSPACE。

**⚠️ 局限性**

局限性包括：结果仅适用于 DL‑Lite 与无连通性限制的 CQ，无法推广到包含概念连接或更强表达式的本体；且对不一致 KB 的情况讨论不充分。

---

## 205. From Local to Cluster: A Unified Framework for Causal Discovery with Latent Variables

**arXiv ID:** 2604.22416 | [PDF](https://arxiv.org/pdf/2604.22416v1)

**作者:** Zongyu Li `[一作]` `[通讯]`, Zongyu Li

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出 L2C 框架，自动从微观变量的局部因果模式中发现簇，并通过局部发现与簇级归约实现宏观因果推断。

**💡 创新点**

在不需要预先给定簇或假设因果充分性的前提下，整合局部与簇级因果学习，利用簇归约定理将簇压缩至最多三节点，同时保证宏观可识别性。

**🔧 技术方法**

局部因果发现采用 MMB‑by‑MMB 算法，簇归约使用簇归约定理，宏观推理基于 σ‑分离的簇级运算，整体流程为局部 → 自动簇发现 → 归约 → σ‑分离推断。

**📊 数据集**

在合成数据（Erdős‑Rényi、Barabási‑Albert 生成的 DAG 并随机引入潜变量）以及真实数据集：基因表达（Arabidopsis thaliana）、在线微服务基准（Online Boutique）和工业控制数据（SWaT）。

**📈 对比分析**

与全局方法（FCI、RFCI、PC）以及局部方法（LCD、CCU、MB‑by‑MB 等）和簇级方法（C‑DAG、αC‑DAG 等）比较。实验表明 L2C 在宏观因果效应识别准确率上超过基线，且在计算量和运行时间上显著优于全局方法，尤其在变量数增大时保持线性或近线性增长。

**⚠️ 局限性**

仅适用于静态 DAG，依赖因果马尔可夫与真实性假设；簇归约上限为三节点，可能在特殊结构下可进一步优化；在样本量有限或假设违背时局部发现与推断性能可能下降。

---

## 206. Robust Fuzzy local k-plane clustering with mixture distance of hinge loss and L1 norm

**arXiv ID:** 2604.22405 | [PDF](https://arxiv.org/pdf/2604.22405v1)

**作者:** Junjun Huang `[一作]` (China Electric Power Research Institute), Jerry Zhijian Yang `[通讯]` (Wuhan University)

**通讯引用:** 22613 | [OpenAlex ID](https://openalex.org/A5100404947)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种鲁棒模糊局部 k-平面聚类（RFLkPC）模型，用于在存在噪声或异常值的环境下对平面结构进行聚类。

**💡 创新点**

创新点在于：①引入局部有界约束，限制平面簇的扩展范围；②设计混合距离（hinge loss + L1）来增强对重尾噪声和异常值的鲁棒性；③将上述两种创新统一到模糊聚类框架中，兼顾局部性和全局性。

**🔧 技术方法**

使用的技术包括：模糊聚类算法（Fuzzy C‑Means 变体）、迭代重加权优化、矩阵特征分解（SVD）求解平面法向量、线性方程组求解平面中心、混合距离函数与 L2‑正则化的组合。

**📊 数据集**

实验数据集：①合成数据（S1、S2、S3 及其高斯、Laplace、t 分布噪声和 40% 均匀噪声版本）；②真实数据（Tone、Crab、UCI 公开数据集 Wine、Dermatology、Ionosphere、Vehicle、Statlog 等）；③点云数据（Super3D 3D 超像素点集）。

**📈 对比分析**

与 KPC、FkPC、LkPPC、SCC、DPCP‑KSS、FCRM、FWCHR、LFDC、SNMoE 等现有方法比较；结果显示 RFLkPC 在大多数合成与真实数据集上取得最高或第二高的 ACC、NMI、ARI、Purity，尤其在噪声强、簇重叠或异常值比例高的情形下表现优异。

**⚠️ 局限性**

限制：①时间复杂度为 O(TNKD^3)，对高维或大规模数据收敛慢；②需要手工调节 λ、α 两个正则化参数，调参工作量大；③在平衡局部有界性与簇间分离度方面仍有改进空间，未形成统一的自适应框架。

---

## 207. Multi-User ISAC with Heterogeneous Unknown Parameters: Optimal Beamforming based on Distribution Information

**arXiv ID:** 2604.22392 | [PDF](https://arxiv.org/pdf/2604.22392v1)

**作者:** Chan Xu `[一作]` (South-Central Minzu University), Shuowen Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 8490 | [OpenAlex ID](https://openalex.org/A5005000898)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种多用户ISAC系统的最优波束成形方案，针对目标的随机方位角（已知先验分布）和未知反射系数（无先验信息）的异质未知参数进行感知与通信的协同优化。

**💡 创新点**

创新点在于：①首次考虑异质未知参数并通过周期性后验CRB（PCRB）作为感知性能指标；②证明在满足各用户速率约束时，最优解仅需最多一条专用感知波束；③利用SDR与拉格朗日对偶构造定理，推导出闭式最优解并完成秩约束的严格性与秩约减。

**🔧 技术方法**

使用的技术包括：半正定松弛（SDR）求解非凸波束优化；拉格朗日对偶与KKT条件分析；周期后验Fisher信息矩阵及PCRB计算；基于向量与矩阵的秩约减方法。

**📊 数据集**

实验数据采用仿真设置：3个用户、3×3天线、3×4接收天线、LoS信道，目标角度使用混合von‑Mises分布（4个峰值）作为先验PDF；信噪比与功率参数设置为-90 dBm、-5 dB等。

**📈 对比分析**

比较方法：与三种基准方案（感知导向波束、双功能波束、最可能角波束）对比；结果显示所提方案在各速率需求下周期PCRB均优于双功能基准，并接近感知导向基准；且所需感知波束数最多为1，验证了理论结论。

**⚠️ 局限性**

局限性包括：仅考虑单目标、LoS信道；假设用户能完美消除感知干扰；未考虑反射系数的统计特性或随机性；对多目标、遮挡与非LoS场景的推广尚未给出。

---

## 208. Opening Pandora's box: Paper mills in conference proceedings

**arXiv ID:** 2604.22458 | [PDF](https://arxiv.org/pdf/2604.22458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 209. Revisiting Neural Activation Coverage for Uncertainty Estimation

**arXiv ID:** 2604.22360 | [PDF](https://arxiv.org/pdf/2604.22360v1)

**作者:** Benedikt Franke `[一作]` (DLR Institute for AI Safety and Security), Arne Raulf `[通讯]` (DLR Institute for AI Safety and Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究者在已有的神经激活覆盖(NAC)方法上，提出了一种新的伪损失函数，使其能够在回归任务中对已训练的人工神经网络进行不确定性估计，并将其用于检测训练分布外数据。

**💡 创新点**

创新点在于：① 将KL散度改为基于预测值与训练集均值的马氏距离，从而实现了对连续输出的伪损失；② 通过这种改进的NAC实现了无需重新训练或微调、仅在推理阶段即可获得回归任务的不确定性评分。

**🔧 技术方法**

采用了NAC框架、马氏距离、三层MLP（128隐藏单元，SELU激活）以及对数值输出的激活状态进行统计的直方图方法。

**📊 数据集**

使用10个UCI回归数据集进行实验，构造了人工的分布外（OoD）样本以评估检测效果。

**📈 对比分析**

与集成学习（10个模型的预测标准差）和MC Dropout（10次前向传播的标准差）进行了对比；结果显示在10个数据集的OoD检测中，NAC在6个案例中获得最高相关性；在ID误差检测中，NAC在7个案例中相关性最低，表明其更专注于OoD不确定性。计算成本也低于两种对比方法。

**⚠️ 局限性**

局限性包括：仅在小型MLP与UCI数据集上验证，未在更大规模或更复杂模型（如深度CNN、对象检测）上测试；对非线性高维回归问题的鲁棒性和适用性尚未探索；以及伪损失函数中马氏距离对异常值的敏感性需进一步研究。

---

## 210. One Shot Learning for Edge Detection on Point Clouds

**arXiv ID:** 2604.22354 | [PDF](https://arxiv.org/pdf/2604.22354v1)

**作者:** Zhikun Tu `[一作]` (Northwest University), Daniel Cohen-Or `[通讯]` (Tel Aviv University)

**通讯引用:** 41563 | [OpenAlex ID](https://openalex.org/A5036688260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种单 shot 学习框架和轻量化网络 OSFENet，用于点云的边缘检测，能够仅凭一张扫描样本学习目标点云的分布并完成大规模点云边缘提取。

**💡 创新点**

创新点在于：① 通过 one‑shot 学习方式，仅用单一目标点云训练即获得优异的边缘检测性能；② 设计了 RBF_DoS 模块，将 Radial Basis Function 表面描述子与深度网络结合；③ 引入 filtered‑kNN 局部表面补丁表示，既降低了训练成本又保持了局部几何信息。

**🔧 技术方法**

技术手段包括：filtered‑kNN 表面补丁构造、RBF（欧氏距离与余弦距离）表面描述子、Transformer 编码器、MLP 解码器、二分类交叉熵损失；实现基于 PyTorch。

**📊 数据集**

使用的数据集包括 CAD 模型集 ABC（全 7071 模型、115 模型子集）和多源扫描数据集 S3DIS、Semantic3D、UrbanBIS；还在 SHREC 进行跨扫描器验证。

**📈 对比分析**

与 BE、SGLBP、EC‑Net、PIE‑Net、DEF、NerVE、NEF 等基线在 ABC‑NEF 与 ABC‑ALL 上进行对比。OSFENet 在 Chamfer Distance、IoU、Precision、Recall、F‑score 等指标均优于所有基线，并在真实扫描场景中表现出更高的精度和更少的误检。相比之下，参数量仅 0.04M，推理速度快于大多数对手。

**⚠️ 局限性**

局限性在于：对多种扫描方式混合的数据集效果不如同源数据；在极端噪声或稀疏采样下仍可能出现误检；RBF_DoS 主要用于表面描述，扩展到分类或分割等更大规模任务尚需进一步研究。

---

## 211. Flow4DGS-SLAM: Optical Flow-Guided 4D Gaussian Splatting SLAM

**arXiv ID:** 2604.22339 | [PDF](https://arxiv.org/pdf/2604.22339v1)

**作者:** Yunsong Wang `[一作]` (National University of Singapore), Gim Hee Lee `[通讯]` (National University of Singapore)

**通讯引用:** 9622 | [OpenAlex ID](https://openalex.org/A5071967339)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为Flow4DGS‑SLAM的动态SLAM框架，实现实时相机定位与动态/静态场景的4D高质量渲染。

**💡 创新点**

创新点包括：①利用相机诱导运动分解生成类别无关动态掩码并提供更精确的相机初始化；②基于光流的场景流高效传播与自适应高斯插入，显著加快动态高斯训练；③采用混合显式轨迹与高斯混合模型（GMM）实现时变不透明度与旋转，提升对复杂动态场景的建模能力。

**🔧 技术方法**

使用技术包括：3D高斯喷射（3DGS）、RAFT光流估计、深度图与YOLOv9语义分割、光流引导的相机位姿优化、KNN平滑、Gaussian Mixture Model（GMM）、滑动窗口训练与轻量级映射步骤。

**📊 数据集**

在TUM RGB‑D与BONN动态序列上进行实验。

**📈 对比分析**

与MonoGS、SplaTAM、SC‑GS、4DGS‑SLAM等基线对比；相机跟踪RMSE低于4DGS‑SLAM（如TUM上0.70cm vs 0.58cm），渲染PSNR/SSIM/LPIPS均显著提升；映射速度提升约10倍（0.50fps vs 0.04fps）。

**⚠️ 局限性**

局限性：依赖光流与深度的质量，光流噪声会影响动态分割与相机初始化；在极端光照或深度稀疏的环境下表现仍有限；在更大规模场景的鲁棒性与实时性仍需进一步验证。

---

## 212. Ownership Refinement Types for Pointer Arithmetic and Nested Arrays

**arXiv ID:** 2604.22361 | [PDF](https://arxiv.org/pdf/2604.22361v1)

**作者:** Yusuke Fujiwara `[一作]` (Kyoto University), Atsushi Igarashi `[通讯]` (Kyoto University)

**通讯引用:** 3617 | [OpenAlex ID](https://openalex.org/A5077485499)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在 Tanaka 等人提出的带分数所有权和细化类型的指针数组验证体系上，扩展了对多维嵌套数组（如矩阵、张量）的支持，并实现了基于此扩展的静态验证器。

**💡 创新点**

创新点在于将所有权函数推广到依赖外层数组索引的形式，使得指向内层数组的指针的所有权可以随外层索引变化；同时引入模板化的所有权推断、改进的自动 Alias 插入策略以及对三维及四维数组的验证。

**🔧 技术方法**

使用的核心技术包括：细化类型与分数所有权相结合的类型系统、基于模板的所有权约束生成与 Z3 SMT 求解、细化谓词的约束 Horn 句子生成与 Hoice 求解、以及指针算术与别名检查的运行时断言。

**📊 数据集**

实验数据集主要是自行设计的嵌套数组操作基准（共 19 程序，涵盖二维到四维矩阵、行/列/对角遍历、读写共享等模式），并与 Tanaka 等人 ConSORT 实现的同一套基准进行对比。

**📈 对比分析**

通过在 Apple M2 机器上测量，验证器在 5 秒到 6 分钟内完成验证；在与 Tanaka 系统的比较中，整体耗时相近甚至更快，特别是所有权推断阶段的速度提升显著；对比实验展示了在大规模数组与嵌套结构上的可行性与效率。

**⚠️ 局限性**

主要限制包括：需要在函数参数上手工给出类型注解（影响可用性与错误诊断）；对所有权拆分的规则较为严格，导致某些可验证程序无法通过；自动 Alias 插入仅覆盖常见模式，仍有少量手动插入需求；目前仅支持数组与引用，尚未扩展到更一般的堆结构。

---

## 213. Counting All Lattice Rectangles in the Square Grid in Near-Linear Time

**arXiv ID:** 2604.22456 | [PDF](https://arxiv.org/pdf/2604.22456v1)

**作者:** Dmitry Babichev `[一作]`, Sergey Babichev `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在 [0,n)×[0,n) 内所有格点矩形（包括非轴平行）的精确计数，提出四种递推改进的算法，最终实现 O(n·log³n) 的近线性时间算法，并给出两项渐近展开式。

**💡 创新点**

创新点在于：① 将几何求和归约为常数规模的加权取整和（floor‑sum）核；② 证明该核在欧几里得归约下闭合，从而实现每个查询 O(log n) 的评估；③ 通过多层参数分割（平方根、立方根、Möbius 等）逐步降低复杂度。

**🔧 技术方法**

使用的技术包括：Möbius 反演、欧几里得算法、取整和（floor‑sum）核、六/十阶矩和递归闭包、原始方向分割、分块计数、常数优化与向量化实现。

**📊 数据集**

数据集为 OEIS A085582 的已知格点矩形计数表，作者自行实现四个版本的算法并计算 2^k（k=1…36） 的精确值作为实验基准。

**📈 对比分析**

与经典 O(n²) 基线、平方根分解 O(n³⁄² log n)、立方根分解 O(n⁴⁄³ log n) 进行对比。实验显示在 n≥2¹⁰⁰ 时 O(n·log³n) 方案在单线程 C++ 环境下跑速最快，归一化后常数因子显著低于其他实现。

**⚠️ 局限性**

局限性：仍需遍历 Θ(n) 个外层尺度，理论上难以突破线性；实现依赖对取整和核的精细化；对更高维格点多边形计数的推广尚未给出；常数因子较大，对极大 n 的实际运算仍有挑战。

---

## 214. A comprehensive evaluation of spatial co-execution on GPUs using MPS and MIG technologies

**arXiv ID:** 2604.22430 | [PDF](https://arxiv.org/pdf/2604.22430v1)

**作者:** Jorge Villarrubia `[一作]` (Universidad Complutense de Madrid), Katzalin Olcoz `[通讯]` (Universidad Complutense de Madrid)

**通讯引用:** 278 | [OpenAlex ID](https://openalex.org/A5082243235)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 NVIDIA GPU 的空间共享技术——多进程服务（MPS）与多实例 GPU（MIG）进行系统评估，探讨它们在共同执行工作负载时的表现与能效。

**💡 创新点**

创新点在于揭示 MPS 的灵活性与 MIG 的硬件隔离之间的权衡，并提供针对不同作业特征的共执行策略改进建议。

**🔧 技术方法**

实验使用了 NVIDIA MPS 与 MIG 技术，在 A100 与 H100 GPU 上进行性能与能耗测评。

**📊 数据集**

未使用外部数据集；实验采用标准 GPU 基准和自定义工作负载进行调试。

**📈 对比分析**

通过对比两种技术在同一工作负载下的吞吐量和能耗，MPS 在最优场景可提升约30%性能、减少约20%能耗，但在内存竞争严重时会下降约30%；MIG 在解决内存竞争方面更稳定，但因额外开销和固定划分策略在某些场景下性能受限。

**⚠️ 局限性**

主要局限包括仅针对 NVIDIA GPU 平台、对实际应用工作负载覆盖不足，以及 MIG 的固定资源划分可能导致部分任务性能下降。

---

## 215. Trust as a Situated User State in Social LLM-Based Chatbots: A Longitudinal Study of Snapchat's My AI

**arXiv ID:** 2604.22417 | [PDF](https://arxiv.org/pdf/2604.22417v1)

**作者:** Annie Landerberg `[一作]` (University of Gothenburg), Alan Said `[通讯]` (University of Gothenburg)

**通讯引用:** 1954 | [OpenAlex ID](https://openalex.org/A5040472816)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过为期四周的纵向定性调查，研究了 Snapchat 的 My AI 这一基于大语言模型的社交聊天机器人在日常使用中的用户信任形成与演变。

**💡 创新点**

创新点在于将信任视为一个“情境化且动态的用户状态”，揭示了信任随交互、期望调整和平台语境的共同演化，而非静态属性，并提出了针对社交聊天机器人的信任建模视角。

**🔧 技术方法**

使用的技术主要是大型语言模型（GPT 系列）驱动的聊天机器人；研究方法为定性访谈式问卷与反复编码分析（reflexive thematic analysis），并未引入新的算法或模型。

**📊 数据集**

数据集为 27 名 18–30 岁的瑞典/挪威 Snapchat 用户，在四周内完成三轮线上问卷，共计 1,126 条开放式回答。

**📈 对比分析**

由于研究是定性探索，未涉及量化比较指标或性能评估；研究重点在于用户信任感知的主题变化和演进路径。

**⚠️ 局限性**

局限性包括：样本仅来自北欧两国的年轻人，缺乏跨文化与年龄层普适性；研究时长仅四周，无法观察长期使用或持久关系的信任变化；结论与 Snapchat 的 My AI 及其平台生态紧密相关，可能不适用于其他社交或任务型聊天机器人。

---

## 216. Large Language Model Counterarguments in Older Adults: Cognitive Offloading or Vulnerability to Moral Persuasion?

**arXiv ID:** 2604.22356 | [PDF](https://arxiv.org/pdf/2604.22356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 217. Distance-Misaligned Training in Graph Transformers and Adaptive Graph-Aware Control

**arXiv ID:** 2604.22413 | [PDF](https://arxiv.org/pdf/2604.22413v1)

**作者:** Qinhan Hou `[一作]` (University of Helsinki), Jing Tang `[通讯]` (University of Helsinki)

**通讯引用:** 13088 | [OpenAlex ID](https://openalex.org/A5083397767)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过在合成的上下文随机块模型图上构造混合局部和远程信号的节点分类任务，研究了图Transformer在不同任务局部度下的通信距离偏差，并提出了距离失配的诊断与自适应控制。

**💡 创新点**

创新点在于将训练过程的通信距离与任务依赖距离分离，引入距离失配度量，并展示了基于oracle目标的距离偏置控制能近似最佳固定偏置，从而改进模型表现。

**🔧 技术方法**

使用带距离偏置的稠密图Transformer（logit_ij = q_i^T k_j/√d + λ_dist·b_dist(spd(i,j))），以及平均距离差和Wasserstein距离等评估指标。

**📊 数据集**

使用的主要数据集为合成的上下文随机块模型（CSBM）图，生成节点标签的方式是将局部信号g_loc和远程信号g_far按比例混合。

**📈 对比分析**

对比了中性训练、固定λ_dist搜索、零差控制和oracle目标差控制，oracle控制在大部分局部度任务上几乎与最佳固定偏置匹配，且相较于中性训练提升显著。

**⚠️ 局限性**

局限性在于oracle目标依赖离线验证，适应性控制本身不足以完全解决问题，且自偏置项的影响尚未充分探讨，未来需要构建可观测的训练时状态估计器。

---

## 218. Introducing Background Temperature to Characterise Hidden Randomness in Large Language Models

**arXiv ID:** 2604.22411 | [PDF](https://arxiv.org/pdf/2604.22411v1)

**作者:** Alberto Messina `[一作]` (RAI - Radiotelevisione Italiana), Stefano Scotta `[通讯]` (RAI - Radiotelevisione Italiana)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型在温度设置为0时仍出现非确定性输出的现象，提出将实现层噪声视为等价温度，并定义背景温度概念，给出了实证估算协议。

**💡 创新点**

创新点在于将实现级非确定性建模为等价温度，提出背景温度T_bg的理论定义，并通过参考模型与目标模型的输出分布比较来估计该温度。

**🔧 技术方法**

使用了统计测度（exact‑match 率、K‑S 距离等）对输出变异性进行量化，并借助批量不变核、确定性归约顺序等技术来分析噪声来源。

**📊 数据集**

实验采用 TruthfulQA、SQuAD 等公开问答数据集，以及人工合成的敏感 Prompt 集，覆盖不同任务和上下文长度。

**📈 对比分析**

方法通过在参考模型上不同温度下生成多次输出，构建温度-变异性分布；然后与目标模型在 T=0 下的分布做 K‑S 距离匹配，得到背景温度估计；实验中 GPT‑4.1‑nano 的 T_bg 约为 0.05–0.1，其他模型亦得到相应估计。

**⚠️ 局限性**

局限性包括对参考模型选择的依赖、温度网格分辨率有限、Prompt 对噪声敏感度不均匀，以及在无法完全控制推理环境时估计不确定性较大。

---

## 219. LeHome: A Simulation Environment for Deformable Object Manipulation in Household Scenarios

**arXiv ID:** 2604.22363 | [PDF](https://arxiv.org/pdf/2604.22363v1)

**作者:** Zeyi Li `[一作]` (Institute of Automation Chinese Academy of Sciences), Ruihai Wu `[通讯]` (Peking University)

**通讯引用:** 208 | [OpenAlex ID](https://openalex.org/A5086096450)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 LeHome——一个集成多类别可变形物体、低成本机器人以及完整家庭场景的高保真仿真平台，并提供基准任务、Action Graph 机制和远程操作工具链。

**💡 创新点**

创新点包括：① 多模态可变形资产库（液体、气体、颗粒、线性、薄壳、体积）与相应物理引擎（PBD、FEM、Eulerian 流体）融合；② 通过 Action Graph 构建因果一致的交互机制；③ 支持低成本 LeRobot 系列，降低部署门槛；④ 结合领域随机化与演示重放，提升 sim-to-real 的鲁棒性。

**🔧 技术方法**

使用的技术：PBD、FEM、Eulerian 流体仿真、Action Graph（事件-响应模型）、多种遥控接口（键盘/手柄/主从）、领域随机化、模仿学习策略（Diffusion Policy、ACT、Pi0、SmolVLA）。

**📊 数据集**

数据集：每个任务 50 条仿真遥控演示（Fold Garment、Fling Garment、Assemble Burger、Cut Sausage、Pour Coffee、Wipe Surface）以及 10 条真实世界演示，用于训练与评估。

**📈 对比分析**

通过统一的 imitation‑learning 训练流程（50 demos，100 次测试）对四种策略进行对比。结果显示：SmolVLA 在 Garment 任务上最佳，DP 在 Cut Sausage 上最高，ACT 在 Assemble Burger 和 Pour Coffee 具竞争力；所有方法在 Fling Garment 上表现低下。真实实验中，Sim+Real 共训练将成功率从约15%提升至约50%。

**⚠️ 局限性**

局限性：大变形、长时程的 Garment 任务仍难以成功；目前验证仅针对低成本机器人，缺乏对更复杂机械结构的评估；需要进一步丰富多模态数据与更高级的仿真-现实桥接技术。

---

## 220. SOC-ICNN: From Polyhedral to Conic Geometry for Learning Convex Surrogate Functions

**arXiv ID:** 2604.22355 | [PDF](https://arxiv.org/pdf/2604.22355v1)

**作者:** Kang Liu `[一作]` (Xi'an Jiaotong University), Jianchen Hu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 2584 | [OpenAlex ID](https://openalex.org/A5100389881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的神经网络架构SOC-ICNN，将输入凸神经网络从线性规划（LP）扩展到二阶锥规划（SOCP），以克服传统ReLU-ICNN在表示能力上的限制。

**💡 创新点**

SOC-ICNN通过引入正半定曲率和基于欧几里得范数的锥体原语，严格扩展了ReLU-ICNN的表示空间，同时保持了优化理论的解释。

**🔧 技术方法**

使用了二阶锥规划（SOCP）作为优化框架，并通过引入二次分支和锥体分支来增强网络的表示能力。

**📊 数据集**

进行了大量实验，使用了多种目标函数，包括各类凸函数，验证了SOC-ICNN在函数逼近和下游决策质量上的表现。

**📈 对比分析**

与传统的ReLU-ICNN和其他平滑激活变体（如Softplus-ICNN）进行了比较，SOC-ICNN在所有测试中表现出更低的相对误差，并且在参数使用上更为高效。

**⚠️ 局限性**

尽管SOC-ICNN在表示能力上有显著提升，但仍然可能在特定情况下受到网络结构和参数选择的限制。

---

## 221. A Nationwide Japanese Medical Claims Foundation Model: Balancing Model Scaling and Task-Specific Computational Efficiency

**arXiv ID:** 2604.22348 | [PDF](https://arxiv.org/pdf/2604.22348v1)

**作者:** Nanae Aratake `[一作]` (Kyoto University), Yasushi Okuno `[通讯]` (Kyoto University)

**通讯引用:** 9407 | [OpenAlex ID](https://openalex.org/A5046211758)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

评估了结构化医疗数据上不同规模Transformer基础模型的预训练与微调对疾病与药物预测性能的影响。

**💡 创新点**

发现下游任务存在模型规模饱和点，疾病预测需大模型而药物预测在中等规模已饱和，提供任务依赖的规模优化策略。

**🔧 技术方法**

使用encoder‑only Transformer预训练（MLM）结合Piecewise Linear Embedding年龄编码，并在有限标签下进行二分类微调，比较从零初始化和LGBM基线。

**📊 数据集**

采用日本MDV全国519院DPC索赔数据库约230万患者（随机抽取32院）生成诊断、药物、年龄、性别序列。

**📈 对比分析**

在四个任务（高血压、慢性肾病、阿莫地平、普瑞巴林）分别使用100/500/1000标注样本，评估AUROC/AUPRC；任务最佳模型均优于LGBM；疾病任务32M/101M最优，药物任务11M最优。

**⚠️ 局限性**

局限包括缺少实验室、生命体征和临床笔记数据，仅单一数据库，评估仅限疾病与药物预测，未进行外部或前瞻性验证。

---

## 222. Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization

**arXiv ID:** 2604.22345 | [PDF](https://arxiv.org/pdf/2604.22345v1)

**作者:** Weixu Zhang `[一作]` (McGill University), Haolun Wu `[通讯]` (McGill University)

**通讯引用:** 302 | [OpenAlex ID](https://openalex.org/A5028155508)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的个性化方法——Differential Preference Steering（DPS），通过识别并放大注意力头中蕴含的用户偏好信号，实现可解释的生成控制。

**💡 创新点**

创新点在于首次用因果消融法发现“Preference Heads”，并在推理阶段用对比解码来放大这些头的影响，从而无需微调模型参数即可实现精准个性化。

**🔧 技术方法**

核心技术包括偏好贡献得分（PCS）计算、偏好头消融与对比解码、以及基于用户嵌入的群集化偏好头路由。

**📊 数据集**

使用公开的 LaMP 评测基准，包含多种生成、分类与回归子任务，覆盖不同模型（LLaMA‑3‑8B、Qwen2‑7B、Mistral‑7B）。

**📈 对比分析**

与 CAD、DeCoRe、DoLa 等现有解码时个性化方法比较，DPS 在 ROUGE、METEOR、Accuracy、F1 等指标上均实现了更高或相当的表现，且在多模型、多任务上表现更为稳健。

**⚠️ 局限性**

局限性包括需访问模型内部结构（不适用于黑盒API），推理时需两次前向传播导致轻微额外延迟，且在用户资料噪声大或不完整时可能放大偏见。

---

## 223. Context-Fidelity Boosting: Enhancing Faithful Generation through Watermark-Inspired Decoding

**arXiv ID:** 2604.22335 | [PDF](https://arxiv.org/pdf/2604.22335v1)

**作者:** Weixu Zhang `[一作]` (Hunyuan AI Digital Human, Tencent), Xue Liu `[通讯]` (McGill University)

**通讯引用:** 13951 | [OpenAlex ID](https://openalex.org/A5100372152)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出一种解码时的上下文可信度提升框架CFB，通过对源支持词的logit加权来降低模型的faithfulness幻觉。

**💡 创新点**

创新点在于将文本水印中的logit形变思路迁移到faithfulness控制，提供三种层级的增益策略（静态、上下文感知、token感知），实现轻量且可解释的上下文对齐。

**🔧 技术方法**

主要技术包括logit加权、Jensen-Shannon散度估计、注意力聚合与语义相似度计算，用于动态调节token的boost。

**📊 数据集**

在CNN/DailyMail、XSum、NQ-Synth、NQ-Swap等公开数据集上评估。

**📈 对比分析**

与CAD、ADACAD、COIECD等现有解码时基线对比，CFB在ROUGE、FactKB、BERT-P以及人类评测中均取得显著或相当提升，且推理开销极低。

**⚠️ 局限性**

限制包括需要模型内部logits和注意力信息，难以在黑盒API下使用；token感知版对高冲突场景效果不稳定，且仍有额外推理开销。

---

## 224. SpaMEM: Benchmarking Dynamic Spatial Reasoning via Perception-Memory Integration in Embodied Environments

**arXiv ID:** 2604.22409 | [PDF](https://arxiv.org/pdf/2604.22409v1)

**作者:** Chih-Ting Liao `[一作]` (UNSW Sydney), Xin Cao `[通讯]` (UNSW Sydney)

**通讯引用:** 7249 | [OpenAlex ID](https://openalex.org/A5100681901)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SpaMEM 基准，系统评估感知、记忆与因果推理在具身环境中的协同效果。

**💡 创新点**

通过层级化评测 (L1–L3)、因果动作-记忆耦合和多模态高保真采集，揭示 VLM 在空间推理、长时记忆与视觉驱动推断方面的瓶颈。

**🔧 技术方法**

使用 LLM（Qwen2.5-7B-Instruct）驱动动作生成、AI2‑THOR/ProcTHOR 3D 模拟、RGB‑D+实例/语义分割采集，配合深度搜索与智能放置策略。

**📊 数据集**

构建了包含 700 只生成房屋、25,000 条交互序列、10,601,392 帧的 SpaMEM 数据集，覆盖 103 种物体与 22 种容器。

**📈 对比分析**

与现有 VLM（InternVL、Qwen、LLaVA）在 L1、L2、L3 三级评测中对比，发现文本辅助下性能可达 60‑70% 的整合分数，而纯视觉模式整合分数下降 50‑70%，凸显“因果耦合”与“符号支架”缺失问题。

**⚠️ 局限性**

局限在于依赖手工设计的容器属性、仅覆盖 ProcTHOR 语义空间、缺乏真实世界多样性及跨模态对齐精细化机制。

---

## 225. TabSCM: A practical Framework for Generating Realistic Tabular Data

**arXiv ID:** 2604.22337 | [PDF](https://arxiv.org/pdf/2604.22337v1)

**作者:** Sven Jacob `[一作]` (Federal Institute for Occupational Safety and Health), Gjergji Kasneci `[通讯]` (Technical University of Munich)

**通讯引用:** 14742 | [OpenAlex ID](https://openalex.org/A5024434748)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于结构因果模型（SCM）的混合类型表格数据生成框架 TabSCM，能够在保持因果依赖关系的前提下生成语义合法且隐私友好的合成数据。

**💡 创新点**

创新点在于将因果图（CPDAG）转化为可训练的 DAG 并对每个变量学习单独的条件分布（连续变量用条件扩散模型，分类变量用梯度提升树），实现对生成过程的可解释性、可审计性和可干预性，并显著提升生成速度（最高 583 倍）和减少规则违规率。

**🔧 技术方法**

核心技术包括：1）因果结构发现（PC/GES/NOTEARS）得到 CPDAG；2）边缘化到 DAG 并按照拓扑顺序采样；3）连续变量使用条件扩散概率模型（DDPM）训练噪声预测；4）分类变量使用梯度提升树近似条件分布；5）支持 do‑操作实现可解释的反事实推理。

**📊 数据集**

实验使用七个真实世界数据集：Adult、Beijing PM2.5、Early Stage Diabetes、HELOC、California Housing、Loan、Magic Gamma Telescope，涵盖分类与回归、数值与离散特征、不同样本规模。

**📈 对比分析**

与基线 GAN、Diffusion、LLM（GReaT、TabDDPM、TabSyn、TabDiff、GOGGLE、Causal‑TGAN、DCM）对比，TabSCM 在统计相似度（密度误差、相关误差）、下游任务效果（AUC/RMSE）、隐私（DCR）以及检测率（C2ST）上均达或超过最先进水平；并在训练时间和规则违规率上显著优于全扩散模型；在类别不平衡场景下亦能有效提升少数类性能。

**⚠️ 局限性**

局限性包括：1）生成质量高度依赖因果结构发现的准确性，若 CPDAG 误差较大则会影响后续模型；2）对高维稀疏分类特征的处理仍有待改进；3）目前仅针对表格数据，未探索时间序列或多模态扩展；4）对非常大规模数据集的可扩展性仍需进一步评估。

---

## 226. FILTR: Extracting Topological Features from Pretrained 3D Models

**arXiv ID:** 2604.22334 | [PDF](https://arxiv.org/pdf/2604.22334v1)

**作者:** Louis Martinez `[一作]` (École Polytechnique), Maks Ovsjanikov `[通讯]` (École Polytechnique)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究预训练的3D点云编码器能否捕获拓扑信息，构建带拓扑标签的合成数据集DONUT，并提出FILTR模型直接从冻结编码器特征预测持久性图。

**💡 创新点**

①创建了首个带有连通分量数和genus标签的合成数据集DONUT；②首次系统量化预训练编码器对全局与局部拓扑的表达能力；③提出FILTR——基于DETR的可学习前馈解码器，能够直接从冻结的3D编码器特征预测持久性图。

**🔧 技术方法**

Transformer预训练编码器（Point‑BERT、Point‑MAE、Point‑GPT、PCP‑MAE）、probe + 线性分类器、Centered Kernel Alignment (CKA)、α‑filtration 计算持久性图、DETR式多目标解码器、Hungarian 匹配 + 组合损失（重构、存在性、对角线惩罚）等。

**📊 数据集**

自研DONUT（29,517个合成网格，含β₀和genus标签），以及公开数据集 ModelNet40、ABC 用于跨域评估。

**📈 对比分析**

通过 probe 精度、CKA 相似度、2‑Wasserstein、bottleneck 和 Persistence Image Error 等指标进行比较；预训练编码器在全局拓扑任务上表现有限，但 FILTR 在 DONUT 上可逼近持久性图，且在 ModelNet40、ABC 等外部数据集保持较好泛化，整体优于传统基准和端到端基线。

**⚠️ 局限性**

预训练编码器对全局拓扑信息捕获有限；FILTR 的性能仍受限于编码器的预训练质量；跨域泛化仍有衰退，尤其在高复杂度数据集上；对噪声持久性点的处理和对角线近似仍需进一步改进。

---

## 227. On the Hybrid Nature of ABPMS Process Frames and its Implications on Automated Process Discovery

**arXiv ID:** 2604.22455 | [PDF](https://arxiv.org/pdf/2604.22455v1)

**作者:** Anti Alman `[一作]` (University of Tartu), Marco Montali `[通讯]` (Free University of Bozen-Bolzano)

**通讯引用:** 6642 | [OpenAlex ID](https://openalex.org/A5047795784)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 ABPMS 过程框架的概念，将其视为半并发执行的过程式和声明式模型集合，并给出了语言无关的形式化定义；随后研究了如何利用“最终跟随”关系（eventually‑follows）从事件日志中自动发现该框架，重点验证了 16 种常见过程构造中 14 种能被声明式约束捕获，并给出了将这些约束映射为等价 Petri‑net 片段的策略。

**💡 创新点**

创新点主要有：①将过程式与声明式模型通过半并发执行方式融合为一个统一的过程框架；②在声明式约束中引入开放世界假设，从而把过程式模型视为约束式约束；③基于事件日志自动发现过程框架的技术方案，首次在多模型环境下探索“最终跟随”而非直接跟随；④提出“硬性 pockets”概念，允许在声明式模型中嵌入过程式片段以提升可读性与可执行性。

**🔧 技术方法**

技术手段包括：Petri‑nets、Declare 语言及其约束模板、DFAs（自动机）来验证约束行为、ProM 平台中的 Declare Miner（自定义扩展）以及自研的 multi‑model miner；使用 trace projection、相对活动基数（relative activity cardinality）等形式化工具进行建模与映射。

**📊 数据集**

实验数据采用人工生成的事件日志，基于各种 Petri‑net 构造（包含重复、可选、并行、XOR、OR 等构造），并利用 100% 支持阈值在日志中挖掘约束；没有使用公开真实业务日志，主要用于验证方法可行性。

**📈 对比分析**

比较方式主要是通过自动挖掘得到的约束与已知 Petri‑net 构造的 DFA 进行一致性检查；结果显示 16 种构造中 14 种能够完整重现，剩余两种需要后处理。性能指标未给出具体数值，但通过手工验证与自动化工具演示证明方法在理论上可行。

**⚠️ 局限性**

局限性包括：①未能自动发现所有 16 种构造（仅 14/16）；②方法依赖 100% 支持阈值，易受噪声影响；③未实现完整的过程框架自动发现算法，只给出约束到 Petri‑net 的映射；④仅考虑 Petri‑nets 与 Declare，未覆盖 BPMN、DCR Graph 等其他建模语言；⑤缺乏对真实业务日志的评估与性能实验。

---

## 228. NRGS: Neural Regularization for Robust 3D Semantic Gaussian Splatting

**arXiv ID:** 2604.22439 | [PDF](https://arxiv.org/pdf/2604.22439v1)

**作者:** Zaiyan Yang `[一作]` (Beijing University of Posts and Telecommunications), Fumio Okura `[通讯]` (University of Osaka)

**通讯引用:** 1088 | [OpenAlex ID](https://openalex.org/A5069226668)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于神经正则化的 3D 语义高斯散射（NRGS）方法，用以纠正由多视角不一致的 2D 语义特征投射到 3D 高斯空间后产生的噪声，并提升语义一致性和精度。

**💡 创新点**

创新点包括：①利用每个高斯的几何与外观属性作为先验，通过一个轻量级条件 MLP 学习从属性空间到语义空间的连续映射；②引入方差加权损失，根据投影特征的方差自适应地重视更可靠的样本；③在同一 MLP 上条件化多粒度（全体、部分、细部）语义特征，鼓励模型学习通用表征，避免单粒度过拟合。

**🔧 技术方法**

核心技术包括：3D 高斯散射（3DGS）表示、训练无关的 2D 语义特征投影与加权平均、方差估计作为置信度、条件 MLP 与残差结构、方差加权的 MSE+余弦相似度损失、以及在单卡 RTX 4090 上实现的高效后处理。

**📊 数据集**

主要使用的数据集是 LERF（用于开放词汇 3D 定位与分割）和 Mip-NeRF-360（用于复杂室内/室外 3D 分割评估），并在两者上与多种现有方法做对比。

**📈 对比分析**

与 LangSplat、GAGS、LangSplatV2、OccamLGS 等基线相比，NRGS 在 LERF 上整体 mIoU 提升至 62.3（占 62.0）且定位 mAcc 达 86.6%（占 81.0%），在 Mip-NeRF-360 上整体 mIoU 提升至 67.1（占 65.5），显示出更稳健的语义一致性与更高的分割/定位精度，并且耗时仅数分钟，优于需要大量迭代训练的对手。

**⚠️ 局限性**

局限性包括：①仍依赖于 2D 预训练模型（如 CLIP、SAM）的特征质量，对极端遮挡或极细小物体的语义投影可能不够准确；②方差估计为一维粗略度量，可能无法充分捕捉多维特征的不一致性；③对极端场景（如光照变化大、视角稀疏）下的鲁棒性尚未系统验证。

---

## 229. SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking

**arXiv ID:** 2604.22438 | [PDF](https://arxiv.org/pdf/2604.22438v1)

**作者:** Chenxi Gu `[一作]` (Monash University), John Grundy `[通讯]` (Monash University)

**通讯引用:** 16597 | [OpenAlex ID](https://openalex.org/A5082913979)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了按词表logit平衡的SSG分割算法以及token级水印强度指标，用于提升LLM生成文本的水印可检测性。

**💡 创新点**

创新点在于：①给出token级水印强度的理论定义与下界分析；②设计SSG通过对logit排序后按组分配绿红集合，实现logit平衡分割，从而显著提升低熵任务下的水印强度下界；③将SSG与现有KGW、SWEET、EWD方案结合，验证其普适性。

**🔧 技术方法**

技术手段包括：基于KGW的logit偏置；随机哈希与种子生成；logit排序+按两两分组；top‑k 处理；统计学检测（z‑score/F1等）。

**📊 数据集**

使用的数据集有：代码生成（HumanEval、MBPP）、数学推理（GSM8K）、高熵文本（C4、CNN/DailyMail）。

**📈 对比分析**

通过在上述数据集和模型（Qwen2.5‑Coder‑7B、DeepSeekMath‑7B、LLaMA‑3‑8B）上对KGW、SWEET、EWD三种方案分别加上SSG进行对比。实验显示，SSG使TPR、F1在1%/5% FPR上提升10–40%点，同时Pass@1或困惑度保持或略有提升，证明在保持文本质量的前提下显著增强水印可检测性。

**⚠️ 局限性**

局限性包括：水印检测仍依赖原始提示，导致现实场景下可用性受限；对极端高峰分布（p₁极大）的任务水印强度提升有限；在某些模型（如DSMath‑7B）或鲁棒性攻击（如改写）下，SSG的优势不明显。

---

## 230. Superminds Test: Actively Evaluating Collective Intelligence of Agent Society via Probing Agents

**arXiv ID:** 2604.22452 | [PDF](https://arxiv.org/pdf/2604.22452v1)

**作者:** Xirui Li `[一作]` (University of Maryland), Tianyi Zhou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在大规模AI代理社会MoltBook中部署探测代理，系统性评估了三层次的集体智能（联合推理、信息合成、基本互动），发现规模本身不足以产生集体智能，交互稀疏且质量低下。

**💡 创新点**

①首次在百万级代理社会上进行量化集体智能评测；②提出分层评估框架与探测代理技术；③通过控制刺激方法客观测量代理间协作。

**🔧 技术方法**

探测代理（保持与普通代理相同的行为模式）+LLM后端（OpenClaw）+判别器/评估者（LLM-as-Judge）+量化指标（准确率、帮助度、回复质量评分）

**📊 数据集**

Humanity's Last Exam（高难度推理）+GSM‑SP（小学数学）+自定义计数任务；平台数据：MoltBook中约200万代理的帖子、评论、投票日志。

**📈 对比分析**

与最强前沿模型（如大型LLM）单独回答同题进行基准对比；结果显示群体准确率约为单模型的1/100，信息合成和互动任务亦未提升；帮助度指标表现不一。

**⚠️ 局限性**

主要瓶颈为交互稀疏与浅层；大多数帖子无回复，回复多为噪声或无关内容；因此难以形成有效信息共享与协作。

---

## 231. HubRouter: A Pluggable Sub-Quadratic Routing Primitive for Hybrid Sequence Models

**arXiv ID:** 2604.22442 | [PDF](https://arxiv.org/pdf/2604.22442v1)

**作者:** Abhinaba Basu `[一作]` `[通讯]` (National Institute of Electronics and Information Technology), Abhinaba Basu (National Institute of Electronics and Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HubRouter模块，利用学习的少量hub令牌将O(n²)注意力替换为O(nM)的hub‑mediated路由，能够在从零训练的Jamba‑style混合架构和12层Transformer中直接插拔；

**💡 创新点**

创新点在于：① encode–decode–score–council四阶段管道，实现内容感知的低秩路由；② 用chunked causal encoding实现严格因果的自回归LM；③ 通过top‑k稀疏注意力大幅降低计算；④ 通过正交正则化提升hub多样性。

**🔧 技术方法**

技术细节包括学习hub向量、交叉注意力、softmax解码、前向top‑k选择、稀疏council注意力、正交正则、Mamba/SSM递归层、从零训练以及对比实验中的多seed跑。

**📊 数据集**

使用WikiText‑103作为主训练/评估数据集，并用SNLI进行零样本分类验证。

**📈 对比分析**

与原始Jamba（全注意力）和纯Transformer进行对比；Hub‑Jamba在单种子下PPL从209.0降至200.2（≈4.2%提升），训练吞吐率在seq=1024时提升约90×；Hub‑GPT在严格因果设置下PPL 211.5±0.4，约比Jamba高3 PPL；在12层Transformer中25%替换达到最佳PPL 268.0，优于全注意力282.4和Mamba278.3；M‑sweep显示M=8–14收敛可靠，正交正则在M=6时恢复全部seed成功。

**⚠️ 局限性**

局限性包括：① Hub‑Jamba的双向hub编码泄漏未来信息；② Hub‑GPT在严格因果下约3-4 PPL质量损失；③ 长上下文(seq≥512)性能下降；④ 对预训练模型的retrofit失败；⑤ 高hub数(M≥20)种子敏感性；⑥ 规模仅≤20M，未测试大规模；⑦吞吐率比较仅基于PyTorch‑native实现，未与FlashAttention或专业CUDA核比较；⑧部分实验缺乏完整超参剖析。

---

## 232. Beyond Land Surface Temperature: Explainable Spatial Machine Learning Reveals Urban Morphology Effects on Human-Centric Heat Stress

**arXiv ID:** 2604.22433 | [PDF](https://arxiv.org/pdf/2604.22433v1)

**作者:** Yuan Wang `[一作]` (National University of Singapore), Rudi Stouffs `[通讯]` (National University of Singapore)

**通讯引用:** 2672 | [OpenAlex ID](https://openalex.org/A5026379314)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在新加坡高密度热带城市中，使用Landsat 8 30m LST和GPU加速的1m UTCI，结合二维与三维城市形态变量，构建并评估了全球与地理加权XGBoost模型，系统比较两种热度指标的空间分布、驱动机制及误差。

**💡 创新点**

创新点包括：①首次将人类生理热应激指数UTCI与高分辨率LST在同一城市尺度下进行对比；②提出Geographically Weighted XGBoost（GW‑XGBoost）与SHAP+GAM解释框架，揭示空间异质性与阈值效应；③验证LST在衡量人类热暴露时的局限性，并为基于人本的热适应规划提供量化依据。

**🔧 技术方法**

技术手段：GPU加速SOLWEIG计算T_mrt和UTCI；Landsat TIRS 统计单窗算法提取LST；地理加权XGBoost（GW‑XGBoost）实现局部模型；SHAP进行特征重要性与贡献解释；GAM对SHAP值做非线性平滑，识别阈值关系。

**📊 数据集**

数据集：Landsat 8 TIRS（30m LST）、NREL NSRDB 2km气象数据（经空间插值到30m）与SOLWEIG输出的1m UTCI；1m城市形态数据（建筑高度、树冠高度、SVF、CANOPY密度等）；二维景观指数（PD、LSI等）；DEM、Albedo、NDVI、NDBI、WET；社会经济指标（人口密度、道路密度、交叉口密度）。

**📈 对比分析**

方法比较：在全局XGBoost与GW‑XGBoost之间做交叉验证；对LST，GW‑XGBoost R²=0.855（OOB）/0.872（测试集）；对UTCI，GW‑XGBoost R²=0.905（OOB）/0.831（测试集），显著优于全局模型。残差Moran I几乎为零，表明空间自相关已被有效消除；SHAP展示LST主要受NDVI、NDBI等二维变量驱动，而UTCI受SVF、CH等三维变量驱动。

**⚠️ 局限性**

局限性：阈值和变量重要性受新加坡热带高密度城市的特定气候与城市形态影响，可能不具备跨城市泛化；气象输入分辨率低（2km→30m插值）可能忽略微尺度气候变异；仅分析白天热暴露，夜间热量平衡未涉及；SOLWEIG与UTCI计算成本高，限制了大规模或情景模拟；模型对极端天气或多源观测融合的鲁棒性需进一步验证。

---

## 233. Automation-Exploit: A Multi-Agent LLM Framework for Adaptive Offensive Security with Digital Twin-Based Risk-Mitigated Exploitation

**arXiv ID:** 2604.22427 | [PDF](https://arxiv.org/pdf/2604.22427v1)

**作者:** Biagio Andreucci `[一作]` (Università degli Studi di Salerno), Arcangelo Castiglione `[通讯]` (Università degli Studi di Salerno)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

设计并实现了Automation-Exploit，一个基于多代理LLM的全自动渗透框架，实现从网络侦察到二进制利用的完整闭环，并通过数字孪生安全执行一次性攻击。

**💡 创新点**

突破传统限制，提出自适应安全层+数字孪生无破坏性调试、对抗交接逃避云API安全对齐、两阶段审计降低幻觉等多项创新。

**🔧 技术方法**

融合MAS与LLM（本地Mistral7B、云端Gemini）、检索增强生成(RAG)、文件描述符挂钩、数字孪生等技术，配合自适应任务调度与剪枝。

**📊 数据集**

在八个自定义黑盒环境（Metasploitable、Kioptrix、MrRobot、DC‑2、Metasploitable 3、Brainpan、Linux Zero‑Day、Windows Zero‑Day）以及公开的NVD/CVE/EPSS数据集上进行评测。

**📈 对比分析**

与CTEM平台、AEG引擎及现有LLM agent对比，使用GER、TTC、FPR、AER等指标，实验显示GER>96%、AER≥85%，FPR在第二阶段被完全消除，数字孪生有效避免Live‑Fire导致的DoS，整体性能优于传统方法。

**⚠️ 局限性**

受限于侦察阶段I/O瓶颈导致TTC较高；缺乏WAF/IPS对抗能力；仅支持单节点攻击；数字孪生基于Docker，无法覆盖内核级漏洞；两阶段审计可能出现共幻觉；对LLM可靠性与成本存在依赖。

---

## 234. Hidden Failure Modes of Gradient Modification under Adam in Continual Learning, and Adaptive Decoupled Moment Routing as a Repair

**arXiv ID:** 2604.22407 | [PDF](https://arxiv.org/pdf/2604.22407v1)

**作者:** Yuelin Hu `[一作]` (Shanghai Jiao Tong University), Li Song `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 85056 | [OpenAlex ID](https://openalex.org/A5100448217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在持续学习中发现并修复了Adam优化器与梯度修改模块组合时的“先抑制后适应”失败模式，并提出一种自适应解耦路由方法（Adaptive‑OGP）来消除该问题；

**💡 创新点**

首次将梯度修改与自适应优化器的组合视为一个独立的失效轴，给出标量代理分析解释该失效，并通过自适应路由和重调度提供可行修复方案；

**🔧 技术方法**

采用标量EMA代理分析、参数级与全局路由实验、η_eff匹配干预、重调度控制以及Adaptive‑OGP的自适应解耦算法；

**📊 数据集**

在8域和16域的持续语言建模基准（HOPE 256M模型）以及7B LoRA扩展上进行实验，并在不同的优化器（Adam、AdamW、AdaFactor、SGD+Momentum）和梯度修改族（投影、惩罚、重放）中进行跨族对比；

**📈 对比分析**

与传统共享路由投影、固定强度解耦、微小重放缓冲等方法对比，Adaptive‑OGP在高重叠无适应调度的困难场景下把遗忘下降约3.8–4.8个单位（相对于Vanilla），在清洁基准下提升1.2–1.7个单位；

**⚠️ 局限性**

理论分析仅针对投影族且在ϵ/√v∞→0的极限下成立；在惩罚或重放族中仅得到经验验证；方法依赖自适应调度，单纯路由不足；尚未验证对符号优化器（如Lion）的适用性；

---

## 235. Region Matters: Efficient and Reliable Region-Aware Visual Place Recognition

**arXiv ID:** 2604.22390 | [PDF](https://arxiv.org/pdf/2604.22390v1)

**作者:** Shunpeng Chen `[一作]` (Beijing University of Posts and Telecommunications), Shibiao Xu `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 2937 | [OpenAlex ID](https://openalex.org/A5011919230)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FoL++ 框架，利用可靠性估计分支和自适应候选调度实现区域感知的视觉地点识别；

**💡 创新点**

创新点包括：1) 可靠性估计分支生成遮挡鲁棒的空间可靠性图；2) 两个对齐损失（SAL、SCEL）强化判别区域；3) 弱监督伪对应策略无需像素级标注；4) 自适应候选调度动态调节重排序候选数量；5) 跨阶段融合全局与局部相似度；

**🔧 技术方法**

技术手段：DINOv2 视觉基础模型、Sinkhorn 最优分配、空间对齐损失、伪对应生成、可靠性加权局部匹配、动态 Top‑k 调度、轻量化可靠性估计解码器；

**📊 数据集**

使用多种公共 VPR 基准：Pitts30k‑test、MSLS‑val、Nordland、Tokyo24/7、Nordland★、Pitts250k、Eynsham；

**📈 对比分析**

与众多单阶段和两阶段 SOTA 方法对比，FoL++ 在七个数据集上实现 R@1 最高 94.7%（Pitts30k）、94.1%（MSLS），同时显著降低显存（0.13 GB）和推理时间（≈0.03 s），提升 40% 速度；

**⚠️ 局限性**

局限性包括：依赖单一视觉基础模型，光照极端变化仍有挑战；对数据集标注误差敏感；缺乏针对姿态或几何约束的显式学习；

---

## 236. HFS-TriNet: A Three-Branch Collaborative Feature Learning Network for Prostate Cancer Classification from TRUS Videos

**arXiv ID:** 2604.22388 | [PDF](https://arxiv.org/pdf/2604.22388v1)

**作者:** Xu Lu `[一作]` (Guangdong Polytechnic Normal University), Yuan Yuan `[通讯]` (Guangdong Polytechnic Normal University)

**通讯引用:** 5030 | [OpenAlex ID](https://openalex.org/A5100437870)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了基于TRUS视频的前列腺癌分类框架 HFS‑TriNet，包含启发式帧选取(HFS)、三分支协同特征学习（ResNet50、MedSAM+NAM、WTCR）以及金字塔级融合与分类头。

**💡 创新点**

创新点在于：①设计 HFS 以减少时间冗余并确保关键病灶内容覆盖；②将时空特征、语义先验和频域信息通过三分支协同学习并融合；③利用冻结的 MedSAM 并引入归一化注意力（NAM）实现语义适配；④采用小波变换残差分支（WTCR）提升噪声抑制与边缘提取。

**🔧 技术方法**

主要技术包括：启发式帧选取策略、ResNet50 时空卷积、MedSAM（Vision Transformer）+归一化注意力、Wavelet Transform Convolutional Residual（WTCR）、金字塔级特征融合、3D 卷积分类头、AdamW 优化、Focal 损失、梯度裁剪、学习率 warm‑up 与多步衰减。

**📊 数据集**

实验基于 1083 条多机构 TRUS 视频（622 名患者），每条视频 40–200 帧，按患者级 6:2:2 进行训练/验证/测试划分，并进一步在 5 折交叉验证中评估稳定性。

**📈 对比分析**

与 12 种基线视频分类模型（TSN、I3D、TRN、SlowFast、SlowOnly、TPN、TANet、Swim、MViT、Text4Vis、MANet、Mult）在相同实验设置下对比，HFS‑TriNet 在准确率、F1、AUC、精确率、灵敏度和特异度上均显著优于所有对比方法（如准确率 0.863，AUC 0.823，精确率 0.941，灵敏度 0.664，特异度 0.976），并在 5 折交叉验证中保持性能稳定。

**⚠️ 局限性**

局限性包括：①数据来源受限于少数机构，缺乏更大范围的多中心验证；②仅使用视频级标签，缺少细粒度病灶标注或分级信息；③灵敏度仍有提升空间，部分微小或弥漫性病灶可能被漏检；④未对极端噪声或低对比度视频的鲁棒性进行系统评估；⑤未深入探讨大模型 MedSAM 在受噪声影响下的适应性。

---

## 237. Selective Contrastive Learning For Gloss Free Sign Language Translation

**arXiv ID:** 2604.22374 | [PDF](https://arxiv.org/pdf/2604.22374v1)

**作者:** Changhao Lai `[一作]` (Xiamen University), Yidong Chen `[通讯]` (Xiamen University)

**通讯引用:** 6759 | [OpenAlex ID](https://openalex.org/A5100333354)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Sign Language Translation (SLT) 中 CLIP‑style 负样本的学习动态，并提出基于轨迹信息的选择性对比学习（SCL‑SLT）来优化跨模态对齐

**💡 创新点**

创新点在于利用负样本相似度随训练的变化轨迹作为难度指标，设计可调度的 Pair‑Selection 机制，逐步从易到难的负样本中挑选最具信息量的对，显著提升对齐质量和翻译性能

**🔧 技术方法**

主要技术包括 Vision‑Language 预训练、CLIP 对比损失、基于训练轨迹的负样本难度评估、渐进式负样本排程、卷积+Transformer 的视频编码器、LoRA 低秩微调、CiCo 细粒度聚合

**📊 数据集**

在德国手语天气数据集 PHOENIX14T 和中文日常手语数据集 CSL‑Daily 上进行实验，使用 BLEU 和 ROUGE 作为评估指标

**📈 对比分析**

与现有 gloss‑free SLT 方法（如 GFSLT‑VLP、C²RL、LLAVA‑SLT 等）相比，SCL‑SLT 在两大基准上均刷新了最高 BLEU‑4/ROUGE，尤其在 CSL‑Daily 上取得 23.25 BLEU‑4，领先对手 2.14 分

**⚠️ 局限性**

主要限制是需要额外训练一个对比学习模型来生成负样本轨迹，增加了数据预处理成本和计算开销；若能直接用公开预训练模型估计语义相似度，可进一步简化流程

---

## 238. CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language

**arXiv ID:** 2604.22367 | [PDF](https://arxiv.org/pdf/2604.22367v1)

**作者:** Rui Zhao `[一作]` (Xiamen University), Yidong Chen `[通讯]` (Xiamen University)

**通讯引用:** 6759 | [OpenAlex ID](https://openalex.org/A5100333354)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了 CNSL-bench，中国国家手语综合评测基准，用于系统评估多模态大语言模型（MLLM）对手语的理解能力。

**💡 创新点**

创新点在于：①以官方标准手语词典为权威语义基底，消除地域变体歧义；②实现文本、示意图、视频三模态对齐；③细粒度区分空写、字母拼写、手语字母三种手势形式，支持针对性分析。

**🔧 技术方法**

采用多模态评测框架，使用四选一多项选择题形式；结合链式思维（CoT）和指令遵循等技术，对21款开源与闭源 MLLM 进行评测，并分析提示词消耗与推理长度。

**📊 数据集**

数据来源为国家通用手语词典（约8,214个词条）与大型中文手语视频数据集（对齐后得到6,707个独立手势条目，20,121道题），同时包含手写、拼音字母等手势子集。

**📈 对比分析**

通过与人工评测对比，发现当前 MLLM 在文本、图像、视频三模态均远低于人类；在模态上表现不平衡，视频性能最低；在手势形式上，字母拼写优于空写和手语字母；闭源与开源模型差距正在缩小，但仍未达到人类水平。

**⚠️ 局限性**

局限性：仅评估词汇级、标准化手语，采用多项选择而非开放式生成；仅覆盖中文国家手语，未涉及多语种、方言或跨文化变体；缺乏对复杂连续手语句子和生成任务的考察。

---

## 239. Evolving Thematic Map Design in Academic Cartography: A Thirty-Year Study Based on Multilingual Journals

**arXiv ID:** 2604.22539 | [PDF](https://arxiv.org/pdf/2604.22539v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 240. Cross-Stage Coherence in Hierarchical Driving VQA: Explicit Baselines and Learned Gated Context Projectors

**arXiv ID:** 2604.22560 | [PDF](https://arxiv.org/pdf/2604.22560v1)

**作者:** Gautam Kumar Jain `[一作]` (Technische Hochschule Augsburg), Julian Stähler `[通讯]` (Technische Hochschule Augsburg)

**通讯引用:** 15 | [OpenAlex ID](https://openalex.org/A5077698516)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于感知‑预测‑规划三阶段的驾驶视觉问答(GVQA)框架，并研究了显式提示式与隐式门控投射器两种跨阶段上下文传递方法。

**💡 创新点**

创新点在于：①无训练的提示式显式传递可直接降低多阶段NLI矛盾；②可训练的隐式门控投射器在隐藏状态层面实现语义状态的压缩与路由，显著提升规划阶段的语义一致性。

**🔧 技术方法**

采用的技术包括：Mini‑InternVL2‑4B‑DA‑DriveLM 与 InternVL3‑8B‑Instruct 两大 VLM，QLoRA 低秩微调、门控上下文投射器、Prompt‑based 结构化提示，以及多语言 NLI 分类器。

**📊 数据集**

实验数据集为 DriveLM‑nuScenes（含六摄像头拼接图像与三阶段 QA 对），在 796 个验证场景的三条固定开放式问题上评估跨阶段一致性。

**📈 对比分析**

通过 BLEU‑1、ROUGE‑L、CIDEr、词汇重叠、结构一致性与多语言 NLI 评价指标，显式方法在不训练的情况下将 NLI 矛盾率降低 42.6%，隐式方法在规划阶段 NLI 矛盾率下降 34%，跨阶段蕴含率提升 50%，规划语言质量提升 30.3%，展现显著的性能提升。

**⚠️ 局限性**

局限性包括：隐式方法在表面层面的词汇重叠与结构一致性下降，生成混合语言且缺乏驾驶域预训练；显式方法虽然无训练成本，但仅提升表面一致性；两种方法基于不同底模，难以直接比较，需进一步在域适配的基模上验证。

---

## 241. Video Analysis and Generation via a Semantic Progress Function

**arXiv ID:** 2604.22554 | [PDF](https://arxiv.org/pdf/2604.22554v1)

**作者:** Gal Metzer `[一作]` (Tel Aviv University), Daniel Cohen-Or `[通讯]` (Tel Aviv University)

**通讯引用:** 41563 | [OpenAlex ID](https://openalex.org/A5036688260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Semantic Progress Function（SPF）来量化视频语义随时间的变化，并基于该函数实现语义线性化（即将语义变化速度保持恒定）的生成与后处理方法。

**💡 创新点**

创新点在于：①用一维SPF直观地表示语义进度并可诊断非线性跳跃；②通过对RoPE时序位置进行频率感知的非线性拉伸，实现生成时无模型重训练的语义线性化；③对已有视频采用分段线性化与重生策略，兼容闭源模型；④提出SPF线性度量与用户研究验证。

**🔧 技术方法**

核心技术包括：图像语义嵌入（SigLIP/CLIP等）、加权最小二乘拟合得到SPF、频率感知RoPE时序拉伸、分段最小二乘分割、视频重生（Wan2.2、LTX‑2）以及迭代细化。

**📊 数据集**

主要使用的数据集包括：合成旋转视频、真实电影片段（如《Stranger Things》），以及基准生成模型（Wan 2.2、LTX‑2）产生的测试视频；并通过这些视频验证SPF与真实语义进度的一致性。

**📈 对比分析**

与传统线性插值、LTX‑2关键帧插值等方法对比，实验表明：①保持了与基准模型相当的视觉质量（VBench指标在1σ内）；②用户研究显示88%偏好语义线性化结果；③在非线性转化视频上明显降低了鬼影与突兀跳跃。

**⚠️ 局限性**

局限性：①SPF受帧级嵌入影响，对高速相机运动、强光照或非语义视觉变化敏感；②频率感知RoPE拉伸可能在多次迭代后偏离模型训练分布，影响生成质量；③尚未充分区分运动与语义变化，需要更鲁棒的时间感知嵌入。

---

## 242. On the Properties of Feature Attribution for Supervised Contrastive Learning

**arXiv ID:** 2604.22540 | [PDF](https://arxiv.org/pdf/2604.22540v1)

**作者:** Leonardo Arrighi `[一作]` (University of Trieste), Marco Zullich `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文比较了使用交叉熵（CE）与监督对比学习（SCL、TL）训练的模型在图像分类任务中的特征归因（FA）质量。

**💡 创新点**

创新点在于提供了在监督对比学习框架下生成类别特定FA的流程，并通过Nauta等的Co‑12评价框架系统评估FA的五个维度。

**🔧 技术方法**

技术包括ResNet‑18/50、SCL与TL损失、Grad‑CAM与Eigen‑CAM解释器、像素翻转（pf）、SSIM、coherence（pg、al）、复杂度（cm）等评估指标。

**📊 数据集**

使用的数据集为CIFAR‑10和ImageNet‑S_50（带分割图）。

**📈 对比分析**

采用多种解释质量指标对三种损失和两种解释器进行对比，结果显示SCL模型在faithfulness、continuity、complexity上优于CE，TL在CIFAR‑10上略优但在ImageNet‑S_50上表现差。

**⚠️ 局限性**

局限性包括仅在图像分类任务上验证，数据集有限，模型精度差异可能影响FA质量，评估指标选择受限，未包含最新的CL方法。

---

## 243. Ablation and the Meno: Tools for Empirical Metamathematics

**arXiv ID:** 2604.22519 | [PDF](https://arxiv.org/pdf/2604.22519v1)

**作者:** Zhengqin Fan `[一作]` (Carnegie Mellon University), Simon DeDeo `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1705 | [OpenAlex ID](https://openalex.org/A5009637990)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了名为Meno的人工智能形式化证明生成系统，并结合tactic ablation技术，对Tao《Analysis I》中的三个定理进行多路径证明实验。

**💡 创新点**

创新点在于：①将人类式证明拆分、分阶段生成的流程嵌入Lean自动化框架；②通过可选剔除非构造性策略来探索约束下的证明空间；③使用Godel Prover的神经网络嵌入并通过MDS/UMAP等降维可视化，揭示证明在低维子流形上的分布。

**🔧 技术方法**

主要技术包括：Lean 4 proof assistant、LLM（Claude）与MCP服务器的协同工作、tactic ablation（按抽象层级、功能类别与公理层级划分）、Godel Prover embeddings、MDS/UMAP降维与高斯混合模型聚类。

**📊 数据集**

数据集：Tao《Analysis I》Lean代码中的三条定理（Russell's Paradox、|X| < |2^X|、Real 的完备性），并使用mathlib中的对应定理作为白名单；Meno产生的每条定理的多条证明路径构成实验样本。

**📈 对比分析**

比较方法：对plain与ablated两种策略生成的证明进行5120维嵌入后计算余弦距离，并与MDS/UMAP投影得到的低维距离进行相关性分析；结果显示三条定理的证明集在1–3维空间内已捕捉到90%以上距离信息；每条证明平均成本约为$0.50–$3，证明数量在26–36条之间。

**⚠️ 局限性**

limitations: ①LLM执行的不确定性与成本较高，难以完全控制策略探索；②实验仅在极简定理上验证，尚未验证对更大、复杂证明的可扩展性；③tactic ablation的分层与公理划分可能不足以覆盖所有非构造性手段；④Godel Prover嵌入的层级选择仍缺乏系统化理论支持。

---

## 244. Aggregate vs. Personalized Judges in Business Idea Evaluation: Evidence from Expert Disagreement

**arXiv ID:** 2604.22517 | [PDF](https://arxiv.org/pdf/2604.22517v1)

**作者:** Wataru Hirota `[一作]` (Stockmark Inc), Tatsuya Ishigaki `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 150 | [OpenAlex ID](https://openalex.org/A5063285759)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在商业想法评估中专家意见分歧的情况下，自动评估器是应近似聚合共识还是个体化标准，并构建了约3,109条专家评分的商业想法评估数据集。

**💡 创新点**

发现专家在细粒度评分上存在系统性分歧，提出个体化判定器比聚合判定器更能匹配专家标准，并展示评估者间推理一致性的关联。

**🔧 技术方法**

使用大型语言模型（Qwen3系列）实现零射击、聚合历史和个体化历史条件下的评分预测，并通过Krippendorff's α、Jaccard相似度等指标评估性能。

**📊 数据集**

基于约300个基于专利的产品想法，涵盖NLP、计算机科学、材料化学三大领域的专家评分数据集。

**📈 对比分析**

通过比较不同模型尺寸和判定条件（零射击/聚合/个体化），个体化判定器在细粒度和粗粒度评分上均实现了更高的Krippendorff's α和Jaccard相似度，尤其在技术有效性和市场规模维度显著提升。

**⚠️ 局限性**

数据集规模有限，仅覆盖专利生成的想法，缺少跨文化或跨行业的验证；聚合判定器在多元评估中易被误用；模型的解释性和对多视角评估的完整支持仍待提升。

---

## 245. LaissezCloud: Continuous Resource Renegotiation for the Public Cloud

**arXiv ID:** 2604.22509 | [PDF](https://arxiv.org/pdf/2604.22509v1)

**作者:** Tejas Harith `[一作]` (MPI-SWS), Antoine Kaufmann `[通讯]` (MPI-SWS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一个基于价格的公共云接口，实现对运行中资源分配的连续重新谈判；

**💡 创新点**

创新点在于将价格作为薄而统一的协调接口，结合拓扑感知市场、受限价格发现以及本地化的租户和运营商策略模块，支持跨租户与运营商的持续争议和动态资源所有权；

**🔧 技术方法**

采用了拓扑感知层次匹配引擎、EconAdapter/InfraMaps本地策略、受限价格发现机制以及按实例计费的市场框架；

**📊 数据集**

使用了Azure LLM推理跟踪、Sailor DNN训练、Parabricks批处理等真实工作负载的重构与性能轮廓数据；

**📈 对比分析**

通过与FCFS、FCFS-P基准对比，实验显示在不同占用率下平均性能保留提升约8–23%，成本-性能比更稳定，拓扑敏感任务可近乎翻倍提升，且系统可扩展至约10k节点规模；

**⚠️ 局限性**

限制包括对重构开销和价格波动控制的依赖、价格机制仅为软性控制无法替代硬性干预，以及缺乏完整的经济理论、激励兼容性和隐私保障。

---

## 246. Information-Theoretic Authenticated PIR: From PIR-RV To APIR

**arXiv ID:** 2604.22505 | [PDF](https://arxiv.org/pdf/2604.22505v1)

**作者:** Pengzhen Ke `[一作]`, Liang Feng Zhang `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了信息理论认证的私人信息检索（itAPIR）模型，并给出了其与信息理论PIR结果验证（itPIR‑RV）的严格关系；通过转换定理将满足一定完整性与隐私条件的itPIR‑RV方案直接升级为满足选择性失败攻击抵抗的itAPIR方案；基于此，构造了两种小规模服务器和通用服务器下的具体itAPIR实现，通信复杂度均为子线性；

**💡 创新点**

核心创新是（1）首次给出了无计算假设下的itAPIR正式定义及其对选择性失败攻击的隐私要求；（2）证明了itPIR‑RV与itAPIR的本质联系，并提出了可直接升级的转换定理；（3）利用该定理实现了两种高效itAPIR构造，填补了信息理论恶意服务器容忍PIR研究中的空白；

**🔧 技术方法**

利用信息理论隐私技术（统计t-隐私）、结果验证技术、分布式点函数（DPF）思想以及MAC等同态认证；在转换证明中使用了混合实验（hybrid argument）与统计距离分析；

**📊 数据集**

本工作为理论性研究，未使用具体数据集；

**📈 对比分析**

通过与现有itPIR‑RV、计算模型APIR等方案的比较，展示了所构造的itAPIR方案在子线性通信复杂度（如O(ℓ²/t·(nℓ/t)^{1/(⌊(2ℓ-1)/t⌋-1} log p)）下实现了与之前方案相同或更优的性能，并且完全摆脱了计算假设；

**⚠️ 局限性**

局限性包括：只能获得统计t-隐私，无法实现完美隐私；转换路径要求原始itPIR‑RV满足v≥t且完整性误差ε可忽略；对不同服务器数量的适用性受限于现有itPIR‑RV构造；

---

## 247. Holo360D: A Large-Scale Real-World Dataset with Continuous Trajectories for Advancing Panoramic 3D Reconstruction and Beyond

**arXiv ID:** 2604.22482 | [PDF](https://arxiv.org/pdf/2604.22482v1)

**作者:** Jing Ou `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Wufan Zhao `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了大规模真实世界全景三维数据集Holo360D，包含109,495个全景图像、LiDAR点云、网格、深度图以及连续相机轨迹；并在此数据集上对多种前馈式3D重建模型进行细调，验证其有效性。

**💡 创新点**

①首个提供连续全景序列和高完整度、精确对齐深度图的数据集；②设计了针对360°数据的去噪、网格补洞、区域重网格化后处理管线；③在前馈3D模型中引入网格深度监督和视角分解（8视角）以克服球面畸变。

**🔧 技术方法**

LiDAR + 360°相机同步采集；离线全局束束调整、点云配准、泊松表面重建；后处理去噪、补洞、区域重网格；基于Transformer的前馈3D重建网络（π³、VGGT、FLARE）进行细调。

**📊 数据集**

Holo360D自身；对比Matterport3D、Stanford2D3D、KITTI-360等现有全景数据集；在Holo360D训练的模型与在Matterport3D训练的模型进行对比。

**📈 对比分析**

采用相机姿态误差（ATE、RPE）、点云准确度（Acc、Comp）和AUC等指标；实验显示在Holo360D上细调后，π³、VGGT、FLARE等模型在姿态和点云重建上均显著优于基线，尤其在玻璃、透明区域恢复更完整。

**⚠️ 局限性**

仍受限于：①对极端动态场景和低纹理区域的鲁棒性待提升；②需要进一步改进对球面投影的专门适配；③数据集虽大但仍以手持设备采集，可能存在视角盲区；④模型对垂直视角（俯视/仰视）敏感，影响整体性能。

---

## 248. Contrastive Semantic Projection: Faithful Neuron Labeling with Contrastive Examples

**arXiv ID:** 2604.22477 | [PDF](https://arxiv.org/pdf/2604.22477v1)

**作者:** Oussama Bouanani `[一作]` (Fraunhofer Heinrich Hertz Institute), Maximilian Dreyer `[通讯]` (Fraunhofer Heinrich Hertz Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种利用对比视觉证据的神经元标注方法，先使用对比样本生成候选标签，再通过对比语义投影（Contrastive Semantic Projection）在CLIP编码空间中进行标签分配，从而显著提升标注的可信度和细粒度；

**💡 创新点**

创新点在于将对比样本系统地引入两大核心步骤：①在VLM候选生成阶段加入对比提示以产生更具区分性的标签；②在标签赋值阶段通过CSP显式抑制对比样本中共同出现的噪声语义，实现对神经元选择性的更精确捕捉；

**🔧 技术方法**

主要技术包括大规模视觉语言模型（InternVL、CLIP）、对比语义投影算法、CLIP特征聚合与加权、SigLIP评分、以及扩散模型生成测试图像；

**📊 数据集**

实验数据集涵盖 ImageNet‑1K、MS COCO 2017、ImageNet‑21k 以及 ISIC 2019（皮肤病变），模型涵盖 ResNet‑50/101、CLIP‑ViT‑B/32、SAE 以及多种 VLM；

**📈 对比分析**

与 SemanticLens、CLIP‑Dissect、le 等现有基线在 DMA、AUC、scs 等指标上进行对比，结果显示 CSP 在 DMA 上提升约 14–18%（比 baselines 多 6–10%），AUC 亦有轻微提升，scs 在通用数据集表现相当或略逊于 le；在医学案例中，CSP（γ=0.5）取得最高 scs，验证了对比标注的可迁移性；

**⚠️ 局限性**

主要限制包括：①对比样本质量不佳或与正样本过于相似时会导致过度投影（信息丢失）；②方法高度依赖 VLM/CLIP 的偏差与泛化能力；③对比样本的获取和多样性受限，尤其在稀疏或专业领域；④γ 超参数的选择缺乏系统化策略；⑤缺少人类评估来验证解释的可解释性与实际效用。

---

## 249. Towards Adaptive Continual Model Merging via Manifold-Aware Expert Evolution

**arXiv ID:** 2604.22464 | [PDF](https://arxiv.org/pdf/2604.22464v1)

**作者:** Haiyun Qiu `[一作]` (Hong Kong Polytechnic University), Kay Chen Tan `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 32485 | [OpenAlex ID](https://openalex.org/A5025285243)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于流形几何的持续模型合并框架MADE-IT，能够在无数据、无训练的前提下，按任务顺序将多个任务特定模型融合为单一统一模型，并在此过程中自适应管理与激活专家网络。

**💡 创新点**

创新点包括①利用子空间投影与Grassmann流形构建的投影子空间相似度度量，用以自动判断专家是否冗余并进行动态合并或增生；②设计了无参数、数据无关的隐式路由机制，利用输入特征投影对齐来决定专家激活路径，从而避免传统MoE方法中依赖门控网络的路由瓶颈；③在专家管理与路由中统一采用流形几何视角，实现模型容量与专家多样性的平衡。

**🔧 技术方法**

主要技术包括SVD截断低秩子空间提取、投影子空间相似度（Hilbert-Schmidt内积）与自适应阈值机制、投影对齐的特征相似度评分、专家依赖图与路径一致性传播、以及基于CLIP-ViT骨干的模块化专家设计。

**📊 数据集**

在CLIP-ViT系列（ViT-B/32、ViT-B/16、ViT-L/14）上，构建了8、14、20任务的合并序列，对每种模型分别使用对应任务的微调检查点，验证方法的鲁棒性与可扩展性。

**📈 对比分析**

与传统的非合并方法（预训练、微调、持续微调）以及持续合并基线（SWA、Task Arithmetic、Ties-Merging、LoRA-WEMoE、OPCM、MINGLE）进行了对比。MADE-IT在ACC上平均提升3–8个百分点，BWT下降幅度显著减少，逼近单任务微调的理论上限，在所有任务长度与模型架构上均取得了state‑of‑the‑art性能。

**⚠️ 局限性**

局限性主要体现在：①实验仅覆盖CLIP‑ViT骨干，尚未验证在其他模型/数据域上的泛化；②对rank比例ρ和阈值β等超参仍需经验性调优；③隐式路由依赖输入特征投影，若任务特征分布剧烈变化可能导致路由误判；④大规模任务序列下的存储与计算开销尚未全面评估。

---

## 250. FeatEHR-LLM: Leveraging Large Language Models for Feature Engineering in Electronic Health Records

**arXiv ID:** 2604.22534 | [PDF](https://arxiv.org/pdf/2604.22534v1)

**作者:** Hojjat Karami `[一作]` (École Polytechnique Fédérale de Lausanne), Anisoara Ionescu `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 3004 | [OpenAlex ID](https://openalex.org/A5059449726)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用大型语言模型（LLM）在只使用数据集元信息的情况下生成针对不规则采样、结构稀疏的电子病历（EHR）时间序列的可执行特征提取代码，随后将生成的特征用于临床预测任务；

**💡 创新点**

创新点包括：①只暴露元数据以保护患者隐私；②通过工具增强的LLM直接编写处理不规则采样的代码；③支持单变量和多变量特征的迭代生成与验证，显著提升特征多样性与质量；

**🔧 技术方法**

技术手段：Gemini‑2.0‑Flash LLM、工具增强生成（查询时间序列、聚合等），Python代码自动化生成与语法/运行时校验，LightGBM等梯度提升树模型作为下游预测器；

**📊 数据集**

实验使用四个ICU公开数据集：PhysioNet 2012（死亡率），PhysioNet 2019（败血症），MIMIC‑III（死亡率），eICU‑CRD（死亡率、急性呼吸衰竭、休克预测），共八个临床预测任务；

**📈 对比分析**

与Baseline、tsfresh、OpenFE、CAAFE、FeatLLM、Zero‑shot等基线进行比较，平均AUROC在7/8任务中排名第一，最高提升达6个百分点，整体性能优于现有方法；

**⚠️ 局限性**

局限性：LLM可能产生幻觉或不合逻辑的特征定义；缺乏对缺失机制（MCAR/MAR/MNAR）的显式建模；生成的特征多样性可能导致冗余，需专家人工验证后方可临床部署。

---

## 251. RouteLMT: Learned Sample Routing for Hybrid LLM Translation Deployment

**arXiv ID:** 2604.22520 | [PDF](https://arxiv.org/pdf/2604.22520v1)

**作者:** Yingfeng Luo `[一作]` (Northeastern University), Jingbo Zhu `[通讯]` (Northeastern University)

**通讯引用:** 2153 | [OpenAlex ID](https://openalex.org/A5100370155)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大模型与小模型混合翻译系统中，研究如何在固定的大模型调用预算下将请求路由到大模型。

**💡 创新点**

创新点在于将路由问题视为预算分配任务，提出使用大模型相对小模型的边际收益作为决策信号，并在小模型内部直接预测该收益，形成无解码、无外部模型的 RouteLMT 路由器。

**🔧 技术方法**

采用小模型的最后提示词隐藏表示做线性回归预测收益，利用 LoRA 进行参数高效微调，并用 XCOMET‑XXL 进行质量评估。

**📊 数据集**

训练使用 ComMT General Translation 拆分，评估使用 FLORES‑200 devtest、WMT24++ 与 BOUQuET 三大数据集。

**📈 对比分析**

与随机、长度、稀有度等启发式以及外部质量/难度预测器比较，RouteLMT 在 Spearman、HitRate@p、MeanΔ@p 等指标上均优于基线，并实现更优的质量–成本 Pareto 曲线。

**⚠️ 局限性**

主要局限：依赖自动质量评估指标、仅考虑两模型固定预算、未验证更大规模或多层级混合架构，以及语言覆盖范围有限。

---

## 252. Benchmarking LLM-Driven Network Configuration Repair

**arXiv ID:** 2604.22513 | [PDF](https://arxiv.org/pdf/2604.22513v1)

**作者:** Ioannis Protogeros `[一作]` (ETH Zürich), Laurent Vanbever `[通讯]` (ETH Zürich)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Cornetto基准，用来系统评估大型语言模型在网络配置错误修复中的能力；同时提供了一个端到端的生成与验证管线，能够自动合成多种规模和协议的误配置场景并通过形式化验证衡量修复效果。

**💡 创新点**

创新点包括①以语法和语义约束为基础的配置生成器，保证合成配置既合法又能产生真实错误；②使用差分数据平面分析与自动规格挖掘，构建可验证的“金标准”规范；③在评价中结合诊断定位、文本诊断质量与功能安全三维度，全面衡量LLM表现。

**🔧 技术方法**

技术包括：基于上下文无关文法的配置渲染；使用Batch‑FISH+Config2Spec进行数据平面模拟与规格抽取；差分分析求解功能恢复与回归；LLM接口采用结构化提示与搜索‑替换修补解析；LLM‑as‑Judge评估诊断文本。

**📊 数据集**

数据集为231个合成误配置实例，覆盖20–754节点的三类拓扑（小、中、大），涉及27种协议/功能错误，配置行数平均1.6万，误配置行占总行数<1%。

**📈 对比分析**

比较方法：针对9个LLM（包括GPT‑5.2、Gemini 3、Claude 4.5等）在同一套问题上评估定位F1、诊断完整度、修复成功率（Fix Score）和回归率。结果显示，顶尖模型Fix Score最高达57.8%，但成功率（无回归且完全恢复）仅约25%；性能随拓扑规模、错误数量和影响范围递减，显著的“规模退化”现象。

**⚠️ 局限性**

局限性：仅评估单一网络环境下的规范，未涵盖容错或多环境约束；仅测试纯LLM推理能力，未结合更复杂的RAG或多代理系统；数据集为合成场景，可能与真实运营环境存在偏差。

---

## 253. Railway Artificial Intelligence Learning Benchmark (RAIL-BENCH): A Benchmark Suite for Perception in the Railway Domain

**arXiv ID:** 2604.22507 | [PDF](https://arxiv.org/pdf/2604.22507v1)

**作者:** Annika Bätz `[一作]` (Karlsruhe Institute of Technology), Martin Lauer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 3994 | [OpenAlex ID](https://openalex.org/A5008189468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了 RAIL‑BENCH，一套针对铁路环境的视觉感知基准，包含轨道检测、目标检测、植被分割、多目标跟踪以及单目视觉里程计五个挑战，并提供训练与测试数据、评测指标与公开排行榜。

**💡 创新点**

主要创新点包括：①首次在铁路域构建全方位感知基准；②设计了新的 LineAP 评测指标，能够独立评估多段线条的几何精度并处理部分检测；③将多种现有技术与自研数据集相结合，形成统一的实验平台。

**🔧 技术方法**

技术方面使用了 CVAT 进行标注；采用 YOLOv11、YOLOv8、PINet、YOLinO 等网络进行检测与轨道线条识别；使用 SegFormer 进行植被分割；通过 Chamfer 距离、LineAP、HOTA、mAP 等指标进行评测。

**📊 数据集**

数据集方面综合使用了 OSDaR23、OSDaR26、RailSem19 等公开铁路数据，经过筛选与重标注构成 RAIL‑BENCH Object+Rail（2,500帧）、Vegetation（740帧）、Tracking（4序列）和 Odometry（50序列）等子集。

**📈 对比分析**

评测通过在线提交预测并自动计算 mAP、LineAP、ChamferAP、HOTA 等指标，结果显示 YOLOv11L 经过铁路数据微调后在多数类别上优于基线；YOLinO 在 LineAP 上表现更佳；整体指标表明基准提供了可重复、可比对的性能基线。

**⚠️ 局限性**

局限性在于：①数据量相对有限，部分挑战（如跟踪）缺乏训练集；②LineAP 仍需手动设定距离与方向阈值；③基准覆盖场景多样性有限，难以囊括所有复杂铁路环境；④评测指标虽覆盖几何与聚类，但对动态障碍物行为预测仍不完善。

---

## 254. ICPR 2026 Competition on Low-Resolution License Plate Recognition

**arXiv ID:** 2604.22506 | [PDF](https://arxiv.org/pdf/2604.22506v1)

**作者:** Rayson Laroca `[一作]` (Federal University of Paraná), David Menotti `[通讯]` (Federal University of Paraná)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文组织了ICPR 2026年度低分辨率车牌识别竞赛，公开了包含真实低质量与高质量图像的最大规模数据集，并给出了评估协议和完整排行榜。

**💡 创新点**

创新点在于：①首次以真实监控场景下的低分辨率车牌数据构建大规模公开基准；②强调多帧跟踪结构的重要性；③通过多种先进技术（超分辨率、教师学生、Transformer、多模态融合等）展示竞赛的多样性与可持续改进空间。

**🔧 技术方法**

参与团队使用的核心技术包括：教师-学生框架、超分辨率网络（HATFIR、MambaIRv2）、OCR网络（SVTR、DETR、MAERec）、STN/Transformer融合、投票与logit级联、以及多模型集成和测试时增强。

**📊 数据集**

数据集为200,000张图像（20,000轨道），每轨包含5张低分辨率和5张高分辨率车牌图像；包含Scenario A（受控光照）和Scenario B（雨天、夜间等多变环境）两类，且只使用低分辨率图像做测试，所有车牌与训练集中不重复。

**📈 对比分析**

方法对比：排行榜前20名识别率在76.47%–82.13%之间，冠军获得82.13%识别率；多团队通过不同策略（超分+OCR、直接OCR+投票、多帧融合、特征/图像级双流等）实现同等或相近性能；置信度间隙（Confidence Gap）差异显著，说明模型在准确率之外还需提升置信度校准。

**⚠️ 局限性**

局限性包括：①任务仍未完全解决，最高误差率17.87%；②缺乏单一主导技术，性能提升仍受限于多帧信息利用；③评估仅采用exact‑match，未充分考虑部分错误信息；④对置信度校准和阈值化策略仍需进一步研究。

---

## 255. Objective Shaping with Hard Negatives: Windowed Partial AUC Optimization for RL-based LLM Recommenders

**arXiv ID:** 2604.22504 | [PDF](https://arxiv.org/pdf/2604.22504v1)

**作者:** Wentao Shi `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8134 | [OpenAlex ID](https://openalex.org/A5051925942)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在强化学习训练LLM推荐器时使用硬负样本（Beam Search negatives）的机制，并提出了窗口化部分AUC（WPAUC）目标及其高效的TAWin优化方法。

**💡 创新点**

创新点包括：①证明GRPO在二值奖励下等价于AUC；②硬负样本将目标转向部分AUC，从而更好对齐Top‑K指标；③提出WPAUC以窗口化的方式更精准地控制FPR区间；④设计TAWin实现软窗口化重权，提升优化效率。

**🔧 技术方法**

所用技术包括：强化学习（GRPO、GSPO、DAPO）、LLM模型（Qwen2.5-0.5B 等）、束搜索与受限采样、AUC/Partial AUC分析、阈值调整窗口化重权等。

**📊 数据集**

使用的数据集为Amazon Review（Toys、Industrial、Office 三个子集）和Yelp 数据集。

**📈 对比分析**

通过与传统顺序模型（GRU4Rec、Caser、SASRec）、生成式模型（TIGER、LC‑Rec、MiniOneRec）以及LLM推荐器（BigRec、D3、S‑DPO、ReRe）的基准比较，TAWin 在所有数据集均实现了SOTA，平均相对提升分别为84.9%（传统模型）、52.0%（生成式模型）和5.5%（LLM模型）。

**⚠️ 局限性**

局限性在于：仍依赖二值规则奖励，未探讨多标签或多目标情况；对公平性、多样性等社会指标的考虑不足；方法对Beam Search的硬负样本生成依赖，计算成本随模型规模增大可能成为瓶颈。

---

## 256. Point & Grasp: Flexible Selection of Out-of-Reach Objects Through Probabilistic Cue Integration

**arXiv ID:** 2604.22491 | [PDF](https://arxiv.org/pdf/2604.22491v1)

**作者:** Xuejing Luo `[一作]` (Aalto University), Antti Oulasvirta `[通讯]` (Aalto University)

**通讯引用:** 14663 | [OpenAlex ID](https://openalex.org/A5003084232)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种基于贝叶斯推理的概率多线索融合框架 Point&Grasp，用于混合现实中的出手范围物体选择。

**💡 创新点**

创新点在于将指向方向与抓握手势这两种互补线索以概率方式融合，突破单线索在空间或语义模糊时的性能瓶颈，并首次为此任务构建了专用的出手抓握数据集。

**🔧 技术方法**

技术包括：手部跟踪获取指向光线和抓握姿态；使用高斯模型估计指向线索的似然；训练双分支神经网络（手部编码器+对象编码器）生成抓握-对象似然；将两条线索通过贝叶斯公式融合得到后验概率；实时决策并交互反馈。

**📊 数据集**

使用了新收集的 10,260 条抓握-对象对（包含 30 个日常物体的近手与远手抓握姿态），以及已有的公共物体模型与手部点云数据。

**📈 对比分析**

通过两项用户研究对比，Point&Grasp 在三种线索配置（Point、Grasp、Point&Grasp）与两种主流单线索技术（BubbleRay、Expand）中，在不同的空间与语义模糊条件下均表现出更快的选择速度和更高的完成率；在大多数情形下与 Expand 相当，在空间模糊时甚至优于其，在语义模糊时亦优于 BubbleRay。

**⚠️ 局限性**

局限包括：抓握-对象模型在未见对象和个体差异上的泛化仍有限；需要进一步的个性化权重学习；系统依赖精确的手部跟踪，手势误差会影响概率估计；实验环境相对受控，真实混合现实场景中的多模态交互仍需验证。

---

## 257. All Eyes on the Workflow: Automated and Efficient Event Discovery from Video Streams

**arXiv ID:** 2604.22476 | [PDF](https://arxiv.org/pdf/2604.22476v1)

**作者:** Marco Pegoraro `[一作]` (RWTH Aachen University), Kristian Kersting `[通讯]` (Technical University of Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 SnapLog pipeline，用视频时间分割和少样本分类自动从视频生成事件日志，以支持流程挖掘。

**💡 创新点**

在低标注环境下结合 ViT + R(2+1)D 的少样本学习，并输出不确定性事件日志，避免硬性决策。

**🔧 技术方法**

利用 Vision Transformer 进行帧级嵌入，K-Means 进行无监督时间分割，R(2+1)D 网络做视频编码，线性分类头实现少样本分类，结合数据增强与重叠采样。

**📊 数据集**

在 TUM Kitchen（受控实验）和 Epic Kitchens-100（野外厨房）两组厨房视频上评估。

**📈 对比分析**

通过不同采样策略（10 vs 100 片段）和增强对比，Top-1/TOP-3 在 TUM 达到 67%/90%，Epic 63%/85%；分割帧级准确率 74%/68%，证明端到端可行。

**⚠️ 局限性**

仍需人工标注少量样本，未实现端到端可微分，假设单一任务不重叠且帧对应单一活动，对多任务/重叠场景的适应性有限。

---

## 258. Gamma-Distributed Geometric Constellation for ISAC: Design and Analysis

**arXiv ID:** 2604.22533 | [PDF](https://arxiv.org/pdf/2604.22533v1)

**作者:** Amirhossein Keshavarzchafjiri `[一作]` (Villanova University), Mojtaba Vaezi `[通讯]` (Villanova University)

**通讯引用:** 2888 | [OpenAlex ID](https://openalex.org/A5051900532)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于Gamma分布幅度和均匀相位的几何星座点设计框架，用于集成感知与通信（ISAC）系统。

**💡 创新点**

创新点在于仅用两参数Gamma分布表征星座点分布，直接以感知检测概率和通信互信息为目标进行优化，并给出联合性能解析上界，无需训练数据，参数量极少。

**🔧 技术方法**

技术手段包括粒子群优化（PSO）寻找α、β，Monte Carlo评估互信息，Albersheim近似计算检测概率，Gamma混合模型与EM算法拟合符号间距离分布，推导SER上界与CRB。

**📊 数据集**

数据集：使用仿真产生的随机星座点与Monte Carlo采样；未使用公开真实数据集。

**📈 对比分析**

与基于神经网络的端到端星座设计进行比较，结果显示在雷达中心、权衡和平衡三种工作点上，Gamma分布方法实现与NN相近或略优的检测概率和误码率，并显著降低参数量与计算成本。

**⚠️ 局限性**

局限性在于对γ分布参数的选择仍依赖经验，且在极端PSK极限下Gamma混合模型拟合尾部不如多元混合；同时，解析上界在高SNR下偏差仍存在，且该方法仅适用于单天线单用户场景。

---

## 259. The Chase in Lean -- Crafting a Formal Library for Existential Rule Research

**arXiv ID:** 2604.22531 | [PDF](https://arxiv.org/pdf/2604.22531v1)

**作者:** Lukas Gerlach `[一作]` `[通讯]` (TU Dresden), Lukas Gerlach (TU Dresden)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在 Lean 证明助手中构建了一个完整的理论框架，用以形式化存在性规则（tuple‑generating dependencies）和 chase 算法，并在此框架下证明了多项重要性质，如 chase 结果为通用模型集、无替代匹配的 chase 产生强核心，以及基于 MFA/DMFA/RMFA 的终止判定框架；

**💡 创新点**

1) 将存在性规则与 chase 的所有常见变体统一到一个可扩展的抽象框架；2) 在 Lean 中实现并证明了 chase 产生通用模型集和核心的性质；3) 提出并验证了一个通用的 MFA‑类似终止判定框架，支持常量与分支化；

**🔧 技术方法**

采用 Lean 依赖类型理论与交互式定理证明；利用 Coinductive 数据结构模拟可能无限的 chase 树；使用 Skolem 术语取代传统 nulls，以保证“新鲜”性；通过定义通用的 obsolescence 条件抽象化不同 chase 变体；

**📊 数据集**

本工作为理论验证，未使用外部实验数据集；所有证明均在 Lean 代码库（约 19000 行）中完成；

**📈 对比分析**

由于该工作主要聚焦于形式化与证明，未进行实验性性能比较；但已在 Lean 代码中提供了可执行的参考实现，未来可用于评估实现效率；

**⚠️ 局限性**

限制：1) 仅覆盖 Skolem 与受限 chase，核心 chase 与更复杂变体（如 RPC、DMFC）尚未完成；2) Lean 对非终止函数的支持有限，导致实现需使用 partial/partial_fixpoint，可能影响可执行性；3) 证明规模庞大，维护与扩展成本较高。

---

## 260. Distilling Vision Transformers for Distortion-Robust Representation Learning

**arXiv ID:** 2604.22529 | [PDF](https://arxiv.org/pdf/2604.22529v1)

**作者:** Konstantinos Alexis `[一作]` (National and Kapodistrian University of Athens), Dimitrios Gunopulos `[通讯]` (National and Kapodistrian University of Athens)

**通讯引用:** 18639 | [OpenAlex ID](https://openalex.org/A5063685438)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过预训练的Vision Transformer教师与学生模型的异步多层蒸馏，学习在严重失真图像下仍能恢复与清晰图像相似的语义表征。

**💡 创新点**

创新点在于提出了同时对齐全局嵌入、补丁特征与注意力图的多层蒸馏目标，并仅使用失真输入进行自监督训练，显著提升失真鲁棒性。

**🔧 技术方法**

使用Vision Transformer（DINO-ViT-B/16）作为教师和学生，配合MSE与KL散度蒸馏损失、注意力对齐、混合精度训练等技术。

**📊 数据集**

在ImageNet‑100、CIFAR‑10/100、STL‑10、RESISC45、CAMELYON17等数据集上进行评估，并对随机像素遮挡、高斯噪声、Gaussian blur等多种失真类型进行测试。

**📈 对比分析**

与监督微调基线和Contrastive Inversion（CLIP‑ResNet）对比，实验表明在大多数失真条件下准确率提升约3–10%，在跨域迁移和低标注比例下仍保持显著优势。

**⚠️ 局限性**

目前只能为单一失真类型训练模型，对未知或混合失真效果有限，且在极端失真下性能仍会下降。

---

## 261. CGC: Compositional Grounded Contrast for Fine-Grained Multi-Image Understanding

**arXiv ID:** 2604.22498 | [PDF](https://arxiv.org/pdf/2604.22498v1)

**作者:** Lihao Zheng `[一作]` (Hangzhou Dianzi University), Tao Wei `[通讯]` (Li Auto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种低成本全流程框架 CGC（Compositional Grounded Contrast），通过单图像标注自动合成多图像训练数据并利用规则化空间奖励进行强化学习，显著提升多图像细粒度理解能力。

**💡 创新点**

创新点包括：① 利用 Inter‑Image Contrast 与 Intra‑Image Contrast 两种对比机制在单图像数据上生成多图像实例，解决跨图像干扰与对象恒常性问题；② 在 Think‑before‑Grounding 模式下引入 Rule‑Based Spatial Reward 与 GRPO 强化学习，直接优化源图像归属、空间对齐和结构化输出合法性。

**🔧 技术方法**

核心技术：自动对比式数据合成、规则化空间奖励、基于 GRPO 的强化学习、结构化输出的思考‑先‑定位范式。

**📊 数据集**

数据集：使用 RefCOCO、SODA、LISA、OmniLabel、VAW 等公开单图像标注数据合成约 36k 条多图像训练样本；在 MIG‑Bench、VLM2‑Bench 以及 BLINK、MathVista、MMMU、MMStar、MuirBench、HallusionBench 等多模态基准上进行评估。

**📈 对比分析**

与主流基线对比：在 MIG‑Bench 与 VLM2‑Bench 上，CGC‑8B (Qwen3‑VL‑8B) 分别取得 67.57 与 73.81 的平均分，超越 Migician、InternVL3‑8B、Qwen2.5‑VL‑72B 等大模型；在更宽泛的多模态推理任务中，CGC‑8B 同样获得 71.65 的平均分，优于 Qwen3‑VL‑8B 基线和多项 RL/ SFT 方法，显示出可迁移性与稳健性。

**⚠️ 局限性**

局限性：① 依赖单图像标注的合成策略，尚未覆盖多图像原生数据或动态视频场景；② 强化学习训练对超参数与奖励设计敏感，可能在不同任务或规模下需要重新调优；③ 仍以单机 8B 模型为主，缺乏对更大模型或跨域任务的充分验证。

---

## 262. Deep Learning for Model Calibration in Simulation of Itaconic Acid Production

**arXiv ID:** 2604.22496 | [PDF](https://arxiv.org/pdf/2604.22496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 263. On first-order model checking parameterized by the number of variables

**arXiv ID:** 2604.22493 | [PDF](https://arxiv.org/pdf/2604.22493v1)

**作者:** Jan Jedelský `[一作]` (Masaryk University), Jan Jedelský `[通讯]` (Masaryk University)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5111154833)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在图类中，以公式变量数为参数的首次阶逻辑（FO）模型检验问题，并给出了完整的可行性与不可行性判定；对单调图类证明该问题在树深度有界时是FPT，树深度无界时为W[*]‑hard；对继承图类给出类似结论，将树深度换成萃缩深度（shrub‑depth）并证明无界时为W[*]‑hard；同时给出了实现该判定的算法与对应的复杂度上界。

**💡 创新点**

①首次把FO模型检验以变量数为参数的可行性阐明；②在单调类中将可行性界定为树深度有界；③在继承类中提出“萃缩深度”作为判定界限；④给出针对树深度和萃缩深度的显式算法与匹配的W[*]‑hard性证明；⑤为继承类提供了一种基于FO解释的可构造性技术。

**🔧 技术方法**

使用了树深度与萃缩深度的结构化树模型；利用Ehrenfeucht‑Fraïssé 以及FO^s 盲游戏对等价性进行分析；构造了“flip”变换和层化“tP_t”等图结构；通过FO解释与反向翻译将路径问题归约回一般图；并使用归纳与递归构造小型等价子树实现FPT算法。

**📊 数据集**

无（本文为纯理论计算复杂性研究，未使用实验数据集）。

**📈 对比分析**

通过理论证明与归约展示算法的FPT性质与对应的W[*]‑hard性界限；无实验对比，性能评价基于时间复杂度分析，算法在树深度或萃缩深度有界时实现线性时间（乘上变量数的多项式因子）。

**⚠️ 局限性**

对继承图类的结论依赖于是否能构造层化“tP_t”或翻转半图，仍是条件性结果；对于MSO逻辑的可行性仍未解决；此外，算法的常数与子树大小可能指数级增长，实际实现受限。

---

## 264. AI-based experts' knowledge visualization of cultural heritage: A case study of Terracotta Warriors

**arXiv ID:** 2604.22480 | [PDF](https://arxiv.org/pdf/2604.22480v1)

**作者:** Siyi Li `[一作]` (Northwest University), Yuhe Zhang `[通讯]` (Northwest University)

**通讯引用:** 731 | [OpenAlex ID](https://openalex.org/A5115593832)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了兵马俑Pit No.1的TW-1087S数据集，并利用AI方法（GAN、随机森林等）对其属性进行重要性评估、分布与相关性分析，随后通过箱线图、小提琴图及交互式可视化呈现整体收藏的属性信息；

**💡 创新点**

创新点在于将GAN（Table‑CGAN）与SMOTENC相结合进行表格数据的增强，同时提出针对集合体的整体可视化方案，使研究者能够一次性把握兵马俑的多维属性与层级关系；

**🔧 技术方法**

使用的技术包括GAN（Table‑CGAN）、SMOTENC、随机森林特征重要性评估、Cramér's V相关性分析、箱线图、小提琴图以及基于访谈的用户研究；

**📊 数据集**

使用的数据集为原始Pit No.1的1087个兵马俑属性表（TW‑1087），通过GAN+SMOTENC扩增后得到1800个样本的TW‑1087S；

**📈 对比分析**

与传统缺失填补方法（STA、MICE、SGAIN、WSGAIN）对比，GAIN在准确率、F1和AUC上表现最佳；在数据增强后，随机森林分类准确率从约86%提升至97%，AUC从85%提升至98%，表明增强方法有效改善了类别不平衡与模型性能；

**⚠️ 局限性**

limitations: 仅基于坑1的数据样本有限，扩增后仍存在少量异常值；可视化方案对非专业用户仍较技术化，需进一步简化和交互；未来需扩展至其他坑的样本并研究更深层次的层级可视化。

---

## 265. Catheter Monitoring in Intelligent Endovascular Navigation Systems: Interactive Simulations and Mixed Reality for Enhanced Navigational Awareness

**arXiv ID:** 2604.22497 | [PDF](https://arxiv.org/pdf/2604.22497v1)

**作者:** Veronica Ruozzi `[一作]` (Politecnico di Milano), Emiliano Votta `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

构建了一个将实时导管形状重建、交互式有限元仿真和混合现实可视化相结合的框架，用于监测内血管导航中的导管-血管相互作用。

**💡 创新点**

创新点在于将导管形状作为边界条件直接驱动交互式仿真，采用Lagrange乘子无参数接触模型，并将血管嵌入软组织连续体中；同时将仿真结果实时投射到MR头显，实现连续导航监测。

**🔧 技术方法**

使用Fiber Bragg Grating与电磁传感器实现导管形状重建；SOFA框架实现有限元仿真；Unity3D+MRTK+Hololens2实现MR可视化；ROS进行节点间通信；高帧率立体相机用于实验验证。

**📊 数据集**

基于右股静脉至下腔静脉的CT数据生成血管模型，制成硅胶假体；使用三次实验的高帧率相机标记数据和传感器数据；使用CT与EM/相机的标记进行配准。

**📈 对比分析**

通过将仿真得到的血管壁位移与相机标记三角化得到的真实位移进行对比，计算相对位移误差，早期误差<1 mm，后期误差最高2.33 mm；仿真时间比实际时间慢12%至45%，渲染保持35‑40 FPS。

**⚠️ 局限性**

仅在单一血管模型上验证，使用刚性标记配准不适用于临床场景；未评估接触力；高计算延迟限制了严格实时性；外部标记对假体壁厚变异敏感。

---

## 266. Test Design and Review Argumentation in AI-Assisted Test Generation

**arXiv ID:** 2604.22473 | [PDF](https://arxiv.org/pdf/2604.22473v1)

**作者:** Eduard Paul Enoiu `[一作]` (Mälardalen University), Robert Feldt `[通讯]` (Chalmers University of Technology)

**通讯引用:** 14519 | [OpenAlex ID](https://openalex.org/A5063787358)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了测试设计论证的概念分类法和结构化模板，帮助AI助手生成测试用例时明确目标、主张、理由和证据。

**💡 创新点**

创新点在于将测试案例与论证结构关联，明确理由、主张、证据，从而提升AI生成测试用例的可解释性和可评审性。

**🔧 技术方法**

使用了概念性分类模型、结构化论证模板，以及AI助手辅助的测试用例生成。

**📊 数据集**

未使用特定数据集；仅在安全关键系统示例中使用规范和执行日志。

**📈 对比分析**

未进行实验比较或性能评估；论文主要为理论与示例说明。

**⚠️ 局限性**

局限在于分类法尚属概念性、缺乏系统验证、未评估AI生成论证的质量，且论证本身不保证正确性。

---

## 267. Reasoning About Probabilities, Actions, and Knowledge in Fuzzy Modal Logic

**arXiv ID:** 2604.22459 | [PDF](https://arxiv.org/pdf/2604.22459v1)

**作者:** Daniil Kozhemiachenko `[一作]` (Aix Marseille Univ), Igor Sedlár `[通讯]` (Czech Academy of Sciences)

**通讯引用:** 184 | [OpenAlex ID](https://openalex.org/A5029560754)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出并研究了一种双层模糊概率逻辑，可在具有概率度量的Kripke框架上表达关于行动与知识的概率推理，并对其在有限分支模型上的可满足性问题进行复杂度分析，给出了多项式时间可判定的子语言。

**💡 创新点**

创新点在于：①将模糊逻辑与行动与知识的两层模态结合，形成全新的双层结构；②首次在此框架下给出可满足性问题的完整复杂度（PSPACE‑完备）并识别出若干多项式可判定的子句；③提出基于样本无关模型的表格法实现决策。

**🔧 技术方法**

采用模糊逻辑（Łukasiewicz 与 Product 逻辑混合）作为基础语义；在其上引入知识与行动模态；构造概率空间和样本无关模型；利用约束表格与线性/多项式不等式系统进行决策；证明复杂度和可判定性。

**📊 数据集**

未使用任何实验数据集，全部结果为理论证明。

**📈 对比分析**

通过理论复杂度分析与多项式时间可判定性的证明进行比较；在有限分支模型上，完整语言为PSPACE‑完备，若干子语言可在多项式时间内判定；无实验性能数据。

**⚠️ 局限性**

局限性包括：仅在有限分支框架下给出结果；缺乏完整的公理化体系；对更一般（无限分支）模型的复杂度仍未得到解答；未考虑更丰富的事件或不确定性度量。

---

## 268. Using Embedding Models to Improve Probabilistic Race Prediction

**arXiv ID:** 2604.22555 | [PDF](https://arxiv.org/pdf/2604.22555v1)

**作者:** Noan Dasanaike `[一作]` (Harvard University), Kosuke Imai `[通讯]` (Harvard University)

**通讯引用:** 36333 | [OpenAlex ID](https://openalex.org/A5015451961)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 embedding-powered BISG（eBISG）方法，利用预训练文本嵌入和神经网络为未在 Census 名称列表中的姓氏或全名生成种族先验概率，从而改进个体种族预测

**💡 创新点**

创新点在于将名称嵌入向量与传统 BISG 结合，为缺失名单中的姓名提供信息化先验；并通过全名嵌入捕获姓氏与名字之间的相互作用，突破 BIFSG 的条件独立假设

**🔧 技术方法**

使用预训练多语种文本嵌入模型（E5-Large）生成姓名向量，随后训练前馈神经网络预测种族分布，并在 BISG/BIFSG 公式中替代无信息先验

**📊 数据集**

使用 2020 年 Census 姓名与名字频率表（156,620 个常见姓氏和 53,616 个常见名字），以及来自北卡、佛罗里达州的选民登记文件（自报种族）作为评估数据；全名模型训练于四个南方州（阿拉巴马、佐治亚、路易斯安那、南卡罗来纳）的选民文件

**📈 对比分析**

与标准 BISG、BIFSG、仅姓氏嵌入、姓氏+名字嵌入等方法比较；在北卡与佛罗里达的匹配与未匹配样本上，所有 eBISG 方法均优于传统方法，尤其是全名嵌入在精确度、召回率、Brier 分数和校准方面表现最佳，显著提升了西班牙裔和亚裔选民的预测质量

**⚠️ 局限性**

局限性包括：全名模型训练样本来自南方州，可能对非南方地区的推广有限；嵌入模型依赖预训练参数，若语言或姓名结构差异大可能影响效果；BISG 的条件独立假设仍被违反，尽管全名嵌入能部分弥补，但在不同种族内族群的姓名组合差异仍未完全捕捉

---

## 269. ASPIRE: Make Spectral Graph Collaborative Filtering Great Again via Adaptive Filter Learning

**arXiv ID:** 2604.22549 | [PDF](https://arxiv.org/pdf/2604.22549v1)

**作者:** Yunhang He `[一作]` (East China Normal University), Wei Zhang `[通讯]` (East China Normal University)

**通讯引用:** 38777 | [OpenAlex ID](https://openalex.org/A5008881437)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ASPIRE框架，使用双层优化自适应学习谱图滤波器，解决传统推荐目标下的低频爆炸问题并提升训练稳定性。

**💡 创新点**

创新点在于将滤波器学习拆分为上层目标与下层训练，形成双层优化结构，抑制低频偏置；并证明该方法对不同图结构、LLM辅助推荐均具备泛化与稳健性。

**🔧 技术方法**

技术包括谱图滤波（基于特征分解）、双层（bi‑level）优化、预归一化处理、LLM编码与whitening初始化，以及多分支融合的后滤波操作。

**📊 数据集**

实验数据集涵盖Amazon Baby、Amazon Electronics、Ciao、LastFM四个协同过滤数据集，并在LLM辅助推荐中使用MiniLM-L6和Qwen2.5-7B等模型。

**📈 对比分析**

与手工设计的Jacobi、Lin‑C、平均池化、Naive‑Learnable及LightGCN等基线进行对比，ASPIRE在多数指标上与最佳手工滤波器相当或更优，并在训练过程中表现出更高的指标与滤波器稳定性。

**⚠️ 局限性**

局限性包括需进行谱分解导致计算开销较大，对极大图规模的适用性仍受限；理论分析基于若干近似假设，实际效果受图结构与初始化的影响。

---

## 270. ReLIC-SGG: Relation Lattice Completion for Open-Vocabulary Scene Graph Generation

**arXiv ID:** 2604.22546 | [PDF](https://arxiv.org/pdf/2604.22546v1)

**作者:** Amir Hosseini `[一作]` (Amirkabir University of Technology), Suiyang Guang `[通讯]` (Amirkabir University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ReLIC‑SGG框架，针对开词汇场景图生成中注释缺失问题，将未标注关系视为潜在正例，完成关系补全；

**💡 创新点**

创新点在于：①构建语义关系格（相似、蕴含、矛盾）以捕获关系间结构；②引入正负无标签学习与潜在关系完成，减少错误负样本；③使用格引导解码与图层一致性约束，生成语义完整但不冗余的图；

**🔧 技术方法**

采用视觉‑语言对齐（CLIP）进行关系候选生成，格传播与对比损失实现潜在关系推断，可靠负估计与正无标签损失训练网络，图变换器捕获图上下文；

**📊 数据集**

主要实验数据集为Visual Genome（VG150）、开词汇VG拆分、Panoptic Scene Graph（PSG），并在不同任务（PredCls、SGCls、SGDet）上评估；

**📈 对比分析**

与Motifs、VCTree、TDE、BGNN、PE‑Net、Pix2Graphs、VL‑IRM等传统与开词汇SOGS方法对比，ReLIC‑SGG在mR@K、S‑mR、U‑mR及HM等指标上均显著提升，尤其在稀有和未见谓词上表现突出；

**⚠️ 局限性**

局限性在于：①对候选关系数量敏感，过多或过少均影响性能；②仍需人工构建/校准关系格，难以自动化；③部分冗余预测仍存在，且对视频/3D场景图未做充分验证；

---

## 271. An Integrated Framework for Explainable, Fair, and Observable Hospital Readmission Prediction: Development and Validation on MIMIC-IV

**arXiv ID:** 2604.22535 | [PDF](https://arxiv.org/pdf/2604.22535v1)

**作者:** Isaac Tosin Adisa `[一作]` `[通讯]` (Florida State University), Isaac Tosin Adisa (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并验证了一套集成框架，用于医院30天再入院风险预测，并提供每患者可解释的SHAP解释、种族和保险公平性评估以及可部署的可观测性架构。

**💡 创新点**

创新点在于三位一体：将可解释性、审计公平性和生产级可观测性整合到同一模型框架，首次在再入院预测中同时满足这三项需求。

**🔧 技术方法**

采用梯度提升树（XGBoost、LightGBM）、逻辑回归、SHAP TreeExplainer、Prometheus/Grafana等技术实现预测、解释和监控。

**📊 数据集**

使用MIMIC‑IV临床数据库中的415,231例成人住院记录进行训练、验证和测试。

**📈 对比分析**

与LACE基线对比，XGBoost在测试集上达到AUC‑ROC 0.696，优于或等同于LACE；LightGBM的Brier分数为0.146，校准最佳，且所有16个亚组均满足公平阈值。

**⚠️ 局限性**

局限性包括单中心数据、缺少实验室特征、未进行前瞻性临床验证，以及可观测性架构尚未在真实生产环境中验证。

---

## 272. DEKL 2.0: Trace-Indexed Knowledge Evolution in Dependent Type Theory

**arXiv ID:** 2604.22530 | [PDF](https://arxiv.org/pdf/2604.22530v1)

**作者:** Chen Peng `[一作]` `[通讯]` (Beijing University of Language and Culture), Chen Peng (Beijing University of Language and Culture)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

DEKL 2.0 架构将依赖类型理论与可追踪的知识演进结合，允许在执行轨迹（trace）上对知识进行类型化、跟踪和修订，同时保持证明系统的单调性。

**💡 创新点**

创新点在于：
- 把执行轨迹视为一等公理化术语并将知识索引为对轨迹的预射（presheaf），从而把非单调行为归因于语义层面的限制映射（restriction map）的非满射性；
- 通过自由轨迹范畴与预射范畴的组合，提供了完整的语义解释；
- 在三层语法（计算层、知识层、命题层）中实现可证明的可追踪性与固定点推理。

**🔧 技术方法**

技术手段包括：
- 依赖类型理论（MLTT）与类别论语义（CwF、自由范畴、预射范畴）；
- 轨迹语法的递归与核心递归（finite trace 的构造与 infinite trace 的 coinductive 形式）；
- 对限制映射的同态与满射性分析，证明非单调性等价于限制映射的非满射；
- 结构化语义模型的构造与满足性证明。

**📊 数据集**

本论文未使用任何公开数据集，所有论证均为理论性证明与形式化语义推导。

**📈 对比分析**

比较方法：论文没有与已有工具或算法进行实验性对比，只通过形式化的定理证明展示了新框架与传统时序/模态逻辑、效应系统等的差异。关于性能，没有量化实验结果。

**⚠️ 局限性**

局限性：
- 主要针对有限轨迹，无限轨迹的结果尚未完整展开；
- 许多证明仍保持在推理概要层面，缺乏完整的形式化证明与机制化实现；
- 案例研究仅提供模板，缺乏大规模工业化评估；
- 未提供完整的 Coq/Agda 等工具实现。

---

## 273. Grouped Pattern and Multi-Periodogram Algorithm for Range Estimation in ISAC Systems

**arXiv ID:** 2604.22525 | [PDF](https://arxiv.org/pdf/2604.22525v1)

**作者:** Yi Geng `[一作]` (Cictmobile), Pan Cao `[通讯]` (University of Hertfordshire)

**通讯引用:** 1865 | [OpenAlex ID](https://openalex.org/A5101928654)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

设计并实现了分组模式（GP）与多周期谱（multi‑periodogram）算法，用于在集成感知与通信（ISAC）系统中进行低信噪比环境下的距离估计，并通过双模式交叉峰值验证实现多峰周期性歧义消除。

**💡 创新点**

创新点在于：①将子载波分组并在每组内使用相同的非等距感知子载波配置，生成具有周期性峰值的距离谱；②将周期性峰值与振幅调制函数分解为两项，通过Hadamard乘积形成独特的多峰结构；③利用两种不同组数的GP共同验证，显著降低误检与误警；④通过重叠GP实现感知子载波重用，减少专用感知信号开销。

**🔧 技术方法**

主要技术包括：OFDM感知子载波分配、FFT/ IFFT 变换、周期性峰值分析、交叉模式验证、CFAR+GNN检测、Monte Carlo仿真、Swerling 1 目标模型、频谱重用与子载波重排。

**📊 数据集**

数据集为仿真生成的信号：27 GHz 载频、100 MHz 带宽、840 个子载波、Swerling 1 模型、微型无人机雷达截面 0.01 m²，进行 500 次 Monte‑Carlo 试验以评估不同距离（20–240 m）下的性能。

**📈 对比分析**

比较方法：将基准 RP+periodogram 与 GP+multi‑periodogram 通过相同的 CFAR 设定、1 bin GNN 匹配进行对比。结果显示：GP 方法在低 SNR（-30 dB 起始）下检测距离提升 16.5%（从 85 m 到 99 m），误警率下降 61%（从 1380 到 540），距离误差保持约 0.4 m，与基准相当。

**⚠️ 局限性**

局限性：需预先设计两种不同组数的 GP 以实现交叉验证，增加系统设计复杂度；计算量相较传统周期谱略大，需额外的周期一致性检查；对子载波分配的灵活性有限，组数必须为子载波总数的因子且两组间不能为整数倍，限制了可选配置空间。

---

## 274. Non-Minimal Sampling and Consensus for Prohibitively Large Datasets

**arXiv ID:** 2604.22518 | [PDF](https://arxiv.org/pdf/2604.22518v1)

**作者:** Seong Hun Lee `[一作]` (University of Zaragoza), Javier Civera `[通讯]` (University of Zaragoza)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 NONSAC 框架，利用非最小采样与一致性评分在大规模含噪数据中实现鲁棒模型估计。

**💡 创新点**

通过非最小随机采样生成多模型并采用多种评分规则挑选最优模型，显著提升在极高离群率和巨大数据集上的可扩展性和鲁棒性。

**🔧 技术方法**

非最小采样、TLP/Pair 等评分规则、与 RANSAC/PCR‑99 等基准估计器的无缝集成、对称点云配准与对应自由配准的实验验证。

**📊 数据集**

合成的相机姿态、PnP 与点云配准数据；以及斯坦福 3D 扫描 Bunny 数据用于对应自由点云配准。

**📈 对比分析**

与 RANSAC、LO‑RANSAC、ANSAC、全局优化、RHT 等方法对比，在高达 99.5% 离群率时仍保持 0.8–0.9 的平均精度，且在对应自由配准下仅约 30 秒完成。

**⚠️ 局限性**

对样本量小或内点比例高的场景优势有限；核心估计器弱时提升有限；需手动调节评分参数，计算成本随采样数线性增长。

---

## 275. Different Strokes for Different Folks: Writer Identification for Historical Arabic Manuscripts

**arXiv ID:** 2604.22515 | [PDF](https://arxiv.org/pdf/2604.22515v1)

**作者:** Hamza A. Abushahla `[一作]` (American University of Sharjah), Mohamed I. AlHajri `[通讯]` (American University of Sharjah)

**通讯引用:** 443 | [OpenAlex ID](https://openalex.org/A5023991318)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在历史阿拉伯手稿上实现作家识别，扩充并校验了 Muharaf 数据集的行标签，提出基于 CNN+注意力的端到端识别模型，并分别在 line‑level 与 page‑disjoint 两种协议下给出基准结果。

**💡 创新点**

①首次为 Muharaf 数据集提供完整的 line‑level 与 page‑disjoint 基准；②手动扩充并校正了 86.75% 的行标签；③引入复合两写作者的类别处理；④将多尺度 NetVLAD 与自注意力机制相结合，提升写家特征表达。

**🔧 技术方法**

采用 ResNet50 / DenseNet201 / Xception 作为特征提取器，加入自注意力模块与 NetVLAD 编码，使用迁移学习（全/局部 fine‑tune）和数据增强，最终输出多分类概率。

**📊 数据集**

Muharaf 公共数据集（24,495 行，179 写家）——通过人工标注后保留 18,987 行、1,015 页，覆盖 179 写家。

**📈 对比分析**

在 line‑level 随机拆分下，DenseNet201+Attention 达到 Top‑1 99.05%、Top‑5 99.73%、F1 97.44%；在 page‑disjoint 拆分下 Top‑1 78.61%、Top‑5 87.79%、F1 66.55%。与不同 backbone、注意力与 fine‑tune 组合对比，attention 与全 fine‑tune 能显著提升性能。

**⚠️ 局限性**

存在严重的类别不平衡，少数写家样本稀缺导致性能低；page‑level 变异大，使 page‑disjoint 结果显著下降；复合两写作者标签稀缺；未涵盖开放集或 OOD 场景的评估。

---

## 276. SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning

**arXiv ID:** 2604.22558 | [PDF](https://arxiv.org/pdf/2604.22558v1)

**作者:** Jichao Wang `[一作]` (vivo AI Lab), Lingfang Zeng `[通讯]` (Zhejiang Lab)

**通讯引用:** 1943 | [OpenAlex ID](https://openalex.org/A5036840192)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 SOLAR-RL，基于半在线框架的长程信用分配方法，用于在离线数据上训练视觉驱动的 GUI 代理。

**💡 创新点**

创新点在于通过轨迹重构与失败点检测生成伪在线回放，再用目标对齐的奖励塑造把轨迹级质量转化为密集步级奖励，从而解决离线 RL 的时序盲点和在线 RL 的采样成本。

**🔧 技术方法**

采用的技术包括轨迹重构、失败点检测、目标对齐奖励塑造、GRPO 算法、MLLM 视觉编码、离线数据驱动训练。

**📊 数据集**

使用的主要数据集为 Android Control、GUI‑Odyssey、Android World 等公开离线轨迹集合，并在实验中以 15k 条静态轨迹进行训练。

**📈 对比分析**

通过与基础 MLLM、在线专用、离线专用三类基线对比，SOLAR‑RL 在离线类别中实现了与在线专用相当甚至更优的性能，尤其在长路径任务和 Android World 上表现出更好的样本效率和鲁棒性。

**⚠️ 局限性**

局限性包括：受离线数据覆盖范围限制，无法完全模拟真实环境中的未见弹窗或动态变化；需要可靠的合法性判定或学习的验证器；实验仅在移动端展开，未覆盖桌面或网页等更复杂平台。

---

## 277. Transferable Physical-World Adversarial Patches Against Pedestrian Detection Models

**arXiv ID:** 2604.22552 | [PDF](https://arxiv.org/pdf/2604.22552v1)

**作者:** Shihui Yan `[一作]` (Huazhong University of Science and Technology), Shengshan Hu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 1971 | [OpenAlex ID](https://openalex.org/A5081287468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于三重损失的可打印对抗补丁（TriPatch），实现对行人检测模型的隐形攻击。

**💡 创新点**

通过联合抑制检测置信度、扰动边框定位和NMS抑制，构造三重协同损失，并加入外观一致性约束提升物理鲁棒性。

**🔧 技术方法**

采用梯度优化、三重损失函数、颜色一致性约束、数据增强等技术，并在数字与真实环境下训练和评估。

**📊 数据集**

使用 INRIA、MS‑COCO 等公开数据集以及多种主流检测器（YOLOv2/3/3tiny/4/5/8/9、Faster R‑CNN）。

**📈 对比分析**

对比 AdvPatch、T‑SEA、NAP、UPC、CAP 等方法，TriPatch 在多模型上显著降低 mAP（如 YOLOv2 仅 0.89%），物理实验攻击成功率高达 80%+。

**⚠️ 局限性**

局限在于补丁的视觉自然度仍有待提升，易被人类识别或在极端光照、遮挡条件下效果下降。

---

## 278. QDTraj: Exploration of Diverse Trajectory Primitives for Articulated Objects Robotic Manipulation

**arXiv ID:** 2604.22551 | [PDF](https://arxiv.org/pdf/2604.22551v1)

**作者:** Mathilde Kappel `[一作]` (Sorbonne University), Stéphane Doncieux `[通讯]` (Sorbonne University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

自动生成多样化的低级轨迹原语，用于在仿真中完成可关节对象的激活任务，并将生成的轨迹在真实机器人上部署。

**💡 创新点**

首次将MAP‑Elites等Quality‑Diversity算法应用于关节对象操纵轨迹生成，结合自适应合规控制和多起始抓取点的进化，显著提升轨迹的多样性与性能。

**🔧 技术方法**

使用MAP‑Elites QD优化、基于合规控制的交互策略、逆运动学、Genesis物理引擎、并行GPU仿真与ROS2卡氏控制实现从仿真到真实机器人的闭环。

**📊 数据集**

在PartNet‑Mobility articulated object dataset上评估，涉及30个不同物体（烤箱、微波炉、阀门等），并对自制实验物体进行额外验证。

**📈 对比分析**

与三种交互策略（Adaptive‑AS、Where2Act‑AS、VAT‑Mart‑AS）和三种抓取选择策略（MAP‑Elites Explore、Success、Random）进行全组合比较。实验表明QDTraj‑MAP‑Elites‑Explore‑Adaptive‑AS在成功轨迹数上比基线提升5–6倍，平均每物体可生成约704条成功轨迹，整体性能大幅超越对比方法。

**⚠️ 局限性**

方法依赖准确的物理模型与惯性参数，仿真偶尔产生虚假接触导致错误簇；此外，需要大规模并行仿真资源，限制了实际部署的普适性。

---

## 279. ArmSSL: Adversarial Robust Black-Box Watermarking for Self-Supervised Learning Pre-trained Encoders

**arXiv ID:** 2604.22550 | [PDF](https://arxiv.org/pdf/2604.22550v1)

**作者:** Yongqi Jiang `[一作]` (Nanjing University of Science and Technology), Liquan Chen `[通讯]` (Southeast University)

**通讯引用:** 98344 | [OpenAlex ID](https://openalex.org/A5101604337)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向自监督学习预训练编码器的黑盒水印框架，兼顾所有权验证、对抗鲁棒性与模型效能。

**💡 创新点**

创新点包括：① 配对差异放大实现黑盒验证；② 潜在表征纠缠与切片Wasserstein距离分布对齐，抑制OOD水印簇，提升对抗鲁棒性；③ 参考引导水印调优，保持模型主任务性能。

**🔧 技术方法**

使用的技术包括：配对差异放大（正交约束）、潜在表征纠缠、切片Wasserstein距离（SWD）分布对齐、参考引导（reference-guided tuning），以及对Fine‑tune、Pruning、DECREE、MM‑BD等攻击的评估。

**📊 数据集**

实验数据集涵盖9个计算机视觉基准：CIFAR‑10/100、Imagenette、Tiny‑ImageNet、ImageNet、STL‑10、CINIC‑10、SVHN、GTSRB；并在SimCLR、MoCo v2、BYOL、SimSiam、DINOv2等主流SSL框架上测试。

**📈 对比分析**

与SOTA SSL‑WM、SSLGuard等对比，验证p‑value极低、误报率0%，模型精度降幅≤1%；在EaaS/MLaaS场景下均能成功识别盗版模型；对Fine‑tune、Pruning、DECREE、MM‑BD等攻击鲁棒性高；计算开销仅≈6%训练时间，推理时无额外成本。

**⚠️ 局限性**

局限性：在剪枝率>85%时模型失效；对完全知情的攻击者（如ψ>0.1）可在牺牲性能的前提下消除水印；水印效果依赖触发器与源类选择，极端自监督框架或数据分布差异较大时性能略减。

---

## 280. Controllable Spoken Dialogue Generation: An LLM-Driven Grading System for K-12 Non-Native English Learners

**arXiv ID:** 2604.22542 | [PDF](https://arxiv.org/pdf/2604.22542v1)

**作者:** Haidong Yuan `[一作]` (Peking University), Hongjie Fan `[通讯]` (China University of Political Science and Law)

**通讯引用:** 186 | [OpenAlex ID](https://openalex.org/A5040666670)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出面向非英语环境中 K‑12 学生的、符合中国教育标准的英语口语练习系统，构建分层词汇表和对话语料，并通过 DDPO 算法实现多轮对话的水平适配与多样性优化。

**💡 创新点**

创新点在于：①提出 DDPO（多样性驱动策略优化）算法，加入单轮与多轮多样性奖励，解决 GRPO 的熵坍塌问题；②构建基于中国标准的分层词汇和对话数据集，并开源；③在保持精准水平匹配的同时实现高质量多样化对话。

**🔧 技术方法**

采用的技术包括：控制文本生成（提示工程、约束解码、SFT）、强化学习（GRPO、DDPO）、token 级优势估计与动态奖励组合，以及基于用户模拟器的多轮轨迹采样。

**📊 数据集**

使用的数据集：从中国教材、考试中提取的分层词汇表（L1–L4）以及对应的教师‑学生多轮对话语料，包含约 6420 条训练对话、412 条验证对话、427 条测试对话（共 191,894 词）。

**📈 对比分析**

通过与提示、约束解码、SFT、GRPO、R_model 等基线在词汇违规率、Diversity、Topic Relevance、Semantic Richness、Topic Guidance 等指标上的对比实验，DDPO 在词汇违规率 6–10% 以内、多样性约 0.7、整体质量平均 4.3/5，显著优于其它方法。

**⚠️ 局限性**

限制包括：①主要依赖用户模拟器，无法完全模拟真实学生的多样性与情感；②训练时需要采样多条轨迹，计算成本高；③仅在中等规模模型（如 Llama‑3.1‑8B）上验证，未测试更大模型；④词汇列表为静态，未覆盖词形变化或复合词。

---

## 281. Information-Theoretic Geometry Optimization and Physics-Aware Learning for Calibration-Free Magnetic Localization

**arXiv ID:** 2604.22526 | [PDF](https://arxiv.org/pdf/2604.22526v1)

**作者:** Wenxuan Xie `[一作]` (Chinese University of Hong Kong), Shing Shin Cheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2177 | [OpenAlex ID](https://openalex.org/A5072251844)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于信息理论的传感器几何优化和物理感知深度学习的磁定位框架，实现了无标定磁定位。

**💡 创新点**

创新点在于：①使用Fisher信息矩阵筛选极少侵入的“错位分层”传感器拓扑；②设计Phy-GAANet引入物理信息特征与几何感知注意力，弥合Sim-to-Real差距；③在不校准硬件的情况下实现毫米级定位。

**🔧 技术方法**

采用FIM/CRLB理论、磁偶极模型、物理增强输入（对数变换、饱和掩码、几何坐标）和自研的Geometry‑Aware Attention模块，配合ResNet‑18骨干；训练使用AdamW、Cosine Annealing等。

**📊 数据集**

训练数据为基于磁偶极模型的10^7条合成样本（分层和平面两种拓扑），评估数据为49×11×6=3234条真实采集样本，包含多姿态和多高度。

**📈 对比分析**

与传统LM、现有CNN/ResNet、PIRNet等方法在同一硬件平台上比较，Phy‑GAANet (分层) 在位置RMSE 1.84 mm、方位RMSE 3.18°，最大误差仅10.10 mm，刷新率>270 Hz，优于LM、其他网络并显著降低最大误差。

**⚠️ 局限性**

局限在于：①仍对磁偶极模型假设敏感，近场饱和时误差上升；②缺乏对外部干扰的鲁棒性评估；③硬件仍需严格布置，不能随意改变传感器位置。

---

## 282. Measuring and Mitigating Persona Distortions from AI Writing Assistance

**arXiv ID:** 2604.22503 | [PDF](https://arxiv.org/pdf/2604.22503v1)

**作者:** Paul Röttger `[一作]` (University of Oxford), Christopher Summerfield `[通讯]` (University of Oxford)

**通讯引用:** 17314 | [OpenAlex ID](https://openalex.org/A5031878516)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过三项大规模实验，系统评估并量化 AI 写作辅助在政治观点文本中对作者个性、信念和身份的扭曲影响，并探讨了对策。

**💡 创新点**

创新点在于首次在真实写作情境下大规模收集读者对 AI 与人类文本的感知差异，发现 AI 产生的多维度扭曲；同时提出并验证了一种基于奖励模型的“重排序”方法，可显著降低极化扭曲。

**🔧 技术方法**

技术方法包括使用 Claude、DeepSeek、ChatGPT 等主流 LLM 进行文本生成；通过对 10,008 条 AI 生成段落的 2.9M 条读者评分训练奖励模型，随后在生成时做多候选重排序；实验数据分析采用混合效应回归与偏差检验。

**📊 数据集**

数据集包含 2,939 名英国写作者、11,091 名读者，共 10,008 条 AI 辅助与 1,501 条人类原始段落，及 2,903,596 条读者评分，涵盖 29 个社会感知维度。

**📈 对比分析**

与无 AI 辅助对照相比，AI 写作显著提高作者在观点极端度、写作质量、情感正向度和特权身份的感知；重排序干预将极化扭曲减半（≈54%），但导致 10% 以上的用户偏好下降，体现了效果与用户接受度之间的权衡。

**⚠️ 局限性**

局限性包括实验仅在英国受试者、政治议题文本、对 AI 采用的频率与真实使用场景可能差异；高风险情境下作者可能更细致编辑，读者可能获得更多身份线索，故结果可能被低估；不同语言与文化背景下的扭曲模式亦需进一步验证。

---

## 283. Decoding High-Dimensional Finger Motion from EMG Using Riemannian Features and RNNs

**arXiv ID:** 2604.22499 | [PDF](https://arxiv.org/pdf/2604.22499v1)

**作者:** Martin Colot `[一作]` (Universite Libre De Bruxelles), Gianluca Bontempi `[通讯]` (Universite Libre De Bruxelles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了低成本硬件同步收集10小时 EMG 与手指关节角度数据，并提出端到端的多频 Riemannian 协方差特征+GRU 回归模型，实现高维手指运动的连续解码。

**💡 创新点**

创新点在于：① 用消费级 8 通道 EMG 绑带和单摄像头完成完整的同步采集；② 采用多频带 Riemannian 协方差投影特征配合轻量 GRU 实现高精度回归；③ 在 Raspberry Pi 5 上实现实时推理，推理速度约 10 次/秒。

**🔧 技术方法**

使用技术包括：EMG 预处理（带通、去直流、CAR）、Riemannian 协方差特征提取、GRU 递归网络、滑动窗口时间序列建模、MediaPipe 视觉关节角度估计、Python/TensorFlow/PyTorch 训练与部署。

**📊 数据集**

使用了自制的 EMG‑Finger‑Kinematics 数据集（20 位受试者，10 h，8 通道 EMG + 15 关节角度）以及公开的 emg2pose 数据集子集（30 受试者）。

**📈 对比分析**

与 TDF+MLP、CRNN、vemg2pose 等基线在 intra‑subject 与 cross‑subject 条件下进行 10‑折与 LOSO 验证；在自制数据集上 intra‑subject 平均绝对误差 9.79°，cross‑subject 16.71°，在 emg2pose 上同样获得最低 NMSE，整体性能优于所有对比方法。

**⚠️ 局限性**

局限性包括：仅在无负载、无物体抓握的自由手势下评估；未考虑长期使用导致的信号漂移与电极位移；未探索模型量化或自适应学习；跨场景、跨会话的泛化仍有限。

---

## 284. Improving Driver Drowsiness Detection via Personalized EAR/MAR Thresholds and CNN-Based Classification

**arXiv ID:** 2604.22479 | [PDF](https://arxiv.org/pdf/2604.22479v1)

**作者:** Gökdeniz Ersoy `[一作]` (MEF University), Serap Kırbız `[通讯]` (MEF University)

**通讯引用:** 185 | [OpenAlex ID](https://openalex.org/A5000945988)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出并评估了两种基于视觉的驾驶员困倦检测方法：一种是使用个性化面部标记（EAR、MAR、PERCLOS）进行阈值自适应检测，另一种是使用卷积神经网络（CNN）对眼部状态和打哈欠进行分类；两种方法在相同实验框架下实时实现并生成警报。

**💡 创新点**

创新点在于：1) 引入了短时间的个性化校准阶段，为每位驾驶员动态设定EAR/MAR阈值，显著提升了传统几何特征的泛化能力；2) 将经典几何指标与深度学习分类相结合，形成了可比对的混合框架；3) 在同一实验环境下对两种方法进行统一评估，提供了客观的性能对比。

**🔧 技术方法**

技术包括：面部关键点检测（MediaPipe Face Mesh）、EAR/MAR/ PERCLOS计算、个性化阈值生成、卷积神经网络（分别用于眼部开闭与打哈欠识别）、图像预处理（CLAHE、数据增强）、时间平滑滤波以及实时警报逻辑。

**📊 数据集**

使用了公开数据集MRL Eye Dataset（84,898张眼部图像）与Yawn Dataset（5,119张口腔图像），以及在车辆模拟器与真实驾驶场景下采集的约1,000张自定义数据，涵盖不同面部结构、光照和头部姿态。

**📈 对比分析**

方法比较采用相同数据划分（70%训练/15%验证/15%测试）、统一时间平滑与警报逻辑。结果显示：个性化阈值相较于通用阈值分别提升EAR和MAR检测准确率约1.5%；CNN模型在眼部检测和打哈欠检测中分别达到99.10%和98.80%的准确率，超出传统方法7%以上。

**⚠️ 局限性**

局限性包括：1) 个性化阈值需提前校准，若驾驶员面部姿态变化大或使用时长过长可能失效；2) CNN模型虽然准确但计算资源需求较高，可能不适合极低功耗嵌入式平台；3) 本研究仅关注视觉特征，未结合红外或生理信号，可能在极端光照或遮挡条件下表现不佳。

---

## 285. On a Hybrid Mixed Domain Decomposition Method

**arXiv ID:** 2604.22543 | [PDF](https://arxiv.org/pdf/2604.22543v1)

**作者:** Kersten Schmidt `[一作]` (Technical University of Darmstadt), Sebastian Schöps `[通讯]` (Technical University of Darmstadt)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并分析了一种混合域分解方法（HMDD），该方法在子域内采用 Raviart‑Thomas 元并在子域间引入 Lehrenfeld–Schöberl 稳定化项，构造了与 HDG 类似的混合离散方案，并给出了一致性、稳健性与误差估计。

**💡 创新点**

创新点在于：①将 HDG 的稳定化思想推广到域分解框架；②在变分层面将散度与子域间的 L² 分布一起处理，得到对稳定化参数 τ 无关的误差上界；③证明了在 τ = 0 时仍能保持良好收敛性，并在 τ → ∞ 时退化为连续 Galerkin，形成完整的参数分析。

**🔧 技术方法**

使用的技术包括：Raviart‑Thomas 元在子域内的离散，面元多项式在子域界面上作投影；Lehrenfeld–Schöberl 稳定化；变分分析中的 inf‑sup 条件和逼近误差；扰动 Galerkin 方法与 Strang 定理的结合；曲面四边形网格的坐标变换和投影误差分析。

**📊 数据集**

实验数据采用自制的分段光滑解析解（圆形域、半径 2、介质系数有跳跃）和对应的源项，所有测试均在曲面四边形网格上进行，没有使用公开数据集。

**📈 对比分析**

对比方法：在不同阶 q（0–3）和不同 τ（0、2、10、400）下计算 L² 误差、边界跳跃、散度误差。结果表明：对 u 与 μ 的 L² 误差在 τ 取小值时达到最优 q+1 收敛率，且在 τ 大值时依然保持相同的收敛速度；而 q 的散度误差在 τ 较大时收敛率下降至 q+½，反映了稳定化对弱解的影响。整体性能与理论预测一致。

**⚠️ 局限性**

局限性包括：①对 q 的散度误差在大 τ 时收敛率不佳；②尚未证明对非匹配网格的可行性；③目前仅针对泊松方程，推广到更一般的偏微分方程（如 Helmholtz、磁静态）还需进一步研究；④稳定化参数的选择对收敛与数值稳定性有影响，需要更系统的参数优化方法。

---

## 286. SS3D: End2End Self-Supervised 3D from Web Videos

**arXiv ID:** 2604.22686 | [PDF](https://arxiv.org/pdf/2604.22686v1)

**作者:** Marwane Hariat `[一作]` (Institut Polytechnique de Paris), Antoine Manzanera `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 1653 | [OpenAlex ID](https://openalex.org/A5083582547)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个端到端自监督3D估计框架，联合预测深度、相机运动和内参，并在大规模 Web 视频上进行预训练。

**💡 创新点**

① 统一 3D 估计器同时输出深度、位姿和内参；② 通过多视角可观测性代理（MVS）实现视频过滤与课程学习，稳定 Web 规模 SfM 训练；③ 采用专家网络分域训练并蒸馏为单一模型，解决域异质性。

**🔧 技术方法**

自监督 SfM 重投影损失、Transformer 架构（VGGT/ViT‑L）、多视角观测度量、课程采样、专家蒸馏、CLIP 嵌入聚类、分域训练等技术。

**📊 数据集**

预训练使用 YouTube‑8M（约 100M 帧），评估基准包括 KITTI、NYUv2、Sintel、TUM‑RGBD 等。

**📈 对比分析**

采用统一的单检查点协议，进行零样本跨域转移与微调评估；在零样本深度上，KITTI Abs Rel 0.064，NYU 0.046；微调后在 KITTI/NYUv2 上均优于现有自监督基线，接近监督基线性能。

**⚠️ 局限性**

仍受多视角可观测性限制，平面或静止视角训练效果不佳；对动态遮挡场景的鲁棒性不足；需要大量算力；缺乏显式动态物体建模。

---

## 287. Can QPP Choose the Right Query Variant? Evaluating Query Variant Selection for RAG Pipelines

**arXiv ID:** 2604.22661 | [PDF](https://arxiv.org/pdf/2604.22661v1)

**作者:** Negar Arabzadeh `[一作]` (UC Berkeley), Matei Zaharia `[通讯]` (UC Berkeley)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在检索增强生成（RAG）系统中，利用查询性能预测（QPP）方法来选择最优的LLM生成查询变体，以提升检索与生成质量。

**💡 创新点**

创新点在于把QPP视为决策支持工具，用于在同一信息需求下的查询变体选择，并揭示检索优化与生成优化之间存在的“效用差距”。

**🔧 技术方法**

采用多种预检索（如IDF、Clarity、Query Space Distance）和后检索（如Score Distribution、监督Transformer）QPP方法，并结合BM25、Dense检索器和RAG生成器。

**📊 数据集**

使用TREC-RAG 2024基准（56个查询、MS MARCO v2.1语料库），对每个查询生成30个LLM查询变体。

**📈 对比分析**

实验结果表明，预检索QPP可在大多数情况下提升RAG端到端质量（相对单一重写器平均提升5‑10%），但仍与oracle上限相差约20‑30%；后检索QPP在检索指标上表现更好，但对生成质量提升有限。

**⚠️ 局限性**

局限性在于当前QPP方法与生成目标仍不完全对齐，难以充分捕捉检索与生成的耦合效应，并且对不同检索模型的泛化能力仍需进一步验证。

---

## 288. How GenAI is Helping Reimagine Antenatal Care in A Low-Resource Setting: From Provider Enablement to Patient Empowerment

**arXiv ID:** 2604.22610 | [PDF](https://arxiv.org/pdf/2604.22610v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 289. Vibe coding for clinicians: democratising bespoke software development for digital health innovation

**arXiv ID:** 2604.22604 | [PDF](https://arxiv.org/pdf/2604.22604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 290. Detecting Concept Drift in Evolving Malware Families Using Rule-Based Classifier Representations

**arXiv ID:** 2604.22629 | [PDF](https://arxiv.org/pdf/2604.22629v1)

**作者:** Tomáš Kalný `[一作]` (Czech Technical University), Mark Stamp `[通讯]` (San Jose State University)

**通讯引用:** 5310 | [OpenAlex ID](https://openalex.org/A5010812344)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于决策树规则集的结构化概念漂移检测方法，评估不同时间窗口和分类设置下的漂移表现。

**💡 创新点**

创新点在于用规则集特征重要性、预测一致性、激活稳定性等多维结构化指标与漂移进行关联，并在六大恶意软件家族上系统验证。

**🔧 技术方法**

技术包括：决策树（最大深度6）规则提取、特征重要性向量、余弦相似、欧氏/曼哈顿距离、Pearson相关等；与 Transcendent、RIPPER 等基线方法对比；使用固定窗口和 DBSCAN/KMeans 聚类进行实验。

**📊 数据集**

数据集为 EMBER2024 PE 文件子集，仅包含六个最大恶意软件家族，约 500k 样本。

**📈 对比分析**

在固定两月窗口、Family‑vs‑Family（FvF）设置下，决策树的 Feature Pearson Correlation ρ≈0.419，优于 RIPPER（0.329）和 Transcendent ICE（0.230）；与 accuracy 差异和数据分布漂移均呈正相关，聚类方法效果较差。

**⚠️ 局限性**

局限性包括指标对家族高度依赖、聚类窗口不稳定、仅评估单颗决策树，未覆盖多树集成或更大规模恶意家族，且缺乏受控漂移注入实验验证。

---

## 291. From graphemic dependence to lexical structure: a Markovian perspective on Dante's Commedia

**arXiv ID:** 2604.22626 | [PDF](https://arxiv.org/pdf/2604.22626v1)

**作者:** Angelo Maria Sabatini `[一作]` (Scuola Superiore Sant'Anna), Angelo Maria Sabatini `[通讯]` (Scuola Superiore Sant'Anna)

**通讯引用:** 8644 | [OpenAlex ID](https://openalex.org/A5087411283)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对但丁《神曲》采用极简的元音-辅音(V/C)编码，将文本转化为符号序列，并用二状态与四状态马尔可夫链建模，测量记忆深度(MD)及三元组探针，进一步通过词袋特征与sPLS-DA/EN-MNLR分类将低层符号依赖与高层文本结构关联。

**💡 创新点**

将极简的 V/C 表示与四状态马尔可夫链结合，既捕获局部依赖的梯度变化，又通过三元组探针与可解释的词汇锚点桥接，从符号层到宏观文本组织形成可解释的连续轨迹。

**🔧 技术方法**

V/C 二/四状态马尔可夫链、记忆深度指标、三元组探针、词袋特征、sPLS-DA 与弹性网络多项式逻辑回归(EN-MNLR) 等统计与机器学习方法，全部在 R 语言环境下实现。

**📊 数据集**

使用公开的 GitHub JSON 版《神曲》（Petrocchi 版），共 14,233 行、100 章，作为分析数据集。

**📈 对比分析**

通过 sPLS-DA 与 EN-MNLR 两种分类器对三部（Inferno、Purgatorio、Paradiso）进行 80/20 交叉验证，平均准确率约 89%/88%，混淆矩阵显示两部几乎无误判；记忆深度在两状态模型中显著上升，四状态模型表现相对温和。

**⚠️ 局限性**

分析仅基于词级特征，跨词边界的探针信息被部分忽略；V/C 编码过于粗糙，缺乏音韵与韵脚细节；仅使用单一文本版本，缺少版本间对比；100 章样本量有限，统计结论可能受样本大小影响。

---

## 292. GazeVLA: Learning Human Intention for Robotic Manipulation

**arXiv ID:** 2604.22615 | [PDF](https://arxiv.org/pdf/2604.22615v1)

**作者:** Chengyang Li `[一作]` (Shanghai Jiao Tong University), Wentao Zhu `[通讯]` (Eastern Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用人类第一人称视频中注视信息学习意图，并将该意图作为中间表示迁移至机器人执行的Vision‑Language‑Intention‑Action (VLIA) 框架；在预训练阶段学习意图-行为关系，在微调阶段仅使用少量机器人数据实现跨体型迁移，推理过程中采用链式思维先预测意图后生成动作。

**💡 创新点**

① 将人类注视视为显式意图表示，直接弥合人机体现差距；② 在大规模人类数据上预训练视觉‑语言模型，使其能够联合推理意图与行为；③ 通过 Chain‑of‑Thought 机制在推理时先获取意图，再生成动作，提升长期规划与细粒度操作的泛化。

**🔧 技术方法**

使用 PaliGemma 视觉‑语言模型（SigLIP+Gemma‑2B）做特征提取；意图采用自回归 token 预测；动作通过条件流匹配（conditional flow matching）产生连续高频动作；预训练/微调采用同步数据增强、目标对齐与联合损失；推理阶段实现意图‑动作链式推理。

**📊 数据集**

1) 超过 150M 帧的第一人称视频集合（13 个公开数据集合并，含手部与注视标注）；2) 采用 Pupil Neon 眼动仪收集的细粒度人类演示；3) 少量机器人演示数据，用于微调与评估。

**📈 对比分析**

在 AV‑ALOHA 仿真环境和真实机器人（Aloha 双臂、Unitree G1）上与 LFA、DP、H‑RDT、π_0.5 等基线对比，ID 情况下平均提升 10‑15% 成功率，OOS/长程/细粒度任务上提升约 22% 或更高；在 pick‑and‑place 任务中成功率达 85%，在细粒度螺丝拧紧等操作中性能翻倍。

**⚠️ 局限性**

未在预训练阶段使用机器人数据；未构建统一的动作空间或对齐人机动作；对不同机器人平台的跨体型泛化仍受限；依赖注视数据的中心偏差可能影响在多种视觉视角下的稳健性。

---

## 293. EV-CLIP: Efficient Visual Prompt Adaptation for CLIP in Few-shot Action Recognition under Visual Challenges

**arXiv ID:** 2604.22595 | [PDF](https://arxiv.org/pdf/2604.22595v1)

**作者:** Hyo Jin Jon `[一作]` (Konkuk University), Eun Yi Kim `[通讯]` (Konkuk University)

**通讯引用:** 2885 | [OpenAlex ID](https://openalex.org/A5100649195)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了EV-CLIP，一种在视觉挑战下对CLIP进行高效视觉提示适配的框架，用于少样本动作识别。

**💡 创新点**

创新点包括两种轻量级视觉提示：用于空间聚焦的mask提示和用于高层次时序建模的context提示，保持CLIP参数冻结且模型骨干无关。

**🔧 技术方法**

采用CLIP、Swin‑Unet风格的mask生成器、压缩的上下文提示、一致性损失以及跨域少样本学习技术。

**📊 数据集**

在五个基准数据集上评估：UCF101、HMDB51、SSv2、ARID（低光环境）和EK100_Verb（第一人称视角）等。

**📈 对比分析**

与现有PETM方法（A5、ST‑Adapter、AIM、EZ‑CLIP、ViLT‑CLIP）相比，EV‑CLIP在少样本设置下实现了最高平均准确率，同时参数量和 FLOPs 仅为这些方法的一小部分。

**⚠️ 局限性**

局限性在于对高动态运动场景的建模不足，时序提示仅在高层聚合，无法捕捉细粒度运动细节，并且不涉及文本提示的联合微调。

---

## 294. Data-Free Contribution Estimation in Federated Learning using Gradient von Neumann Entropy

**arXiv ID:** 2604.22562 | [PDF](https://arxiv.org/pdf/2604.22562v1)

**作者:** Asim Ukaye `[一作]` (MBZUAI), Karthik Nandakumar `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于最终层权重的von Neumann熵的无数据、无自报方式的客户端贡献估计方法，并实现了SpectralFed和SpectralFuse两种变体；

**💡 创新点**

创新点在于利用最终层梯度的熵作为信息丰富度指标，并通过秩自适应Kalman滤波将熵与类特定Shapley值融合，以获得更稳定的贡献估计；

**🔧 技术方法**

采用了von Neumann熵、类特定Shapley值、秩自适应Kalman滤波、Momentum平滑以及标准的Federated Averaging聚合；

**📊 数据集**

在CIFAR-10/100、FEMNIST和FedISIC等多个视觉任务和非IID划分上进行实验；

**📈 对比分析**

与FedAvg、CGSV、ShapFed等基线对比，SpectralFed/SpectralFuse在多数非IID设置下与单独客户端准确率高度相关，且在全局模型准确率上达到或超过基线；

**⚠️ 局限性**

局限在于对全局准确率的提升有限，主要受本地训练策略和模型迭代终止影响，且方法对极端异构场景的鲁棒性尚需进一步验证。

---

## 295. An algebraic characterisation of Eve-positional languages

**arXiv ID:** 2604.22648 | [PDF](https://arxiv.org/pdf/2604.22648v1)

**作者:** Thomas Colcombet `[一作]` (Université Paris Cité), Olivier Idir `[通讯]` (Université Paris Cité)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5014600757)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

提出了对 ω-正则语言 Eve‑positionality 的新的代数化简表述，并给出了三条局部偏好性质作为必要且充分的判定条件。

**💡 创新点**

创新点在于：①完全不依赖自动机；②只需验证有限个本地性质；③利用 Casares 与 Ohlmann 的 1‑to‑2 玩家提升结果，将一人场景的位姿性质推广到任意两人游戏。

**🔧 技术方法**

主要技术包括：ω-正则语言的代数结构、局部偏好性质的定义、有限记忆确定性定理、以及 1‑to‑2 玩家提升定理；证明过程中采用了游戏图的构造与策略缩减论证。

**📊 数据集**

未使用实验数据集；本工作为纯理论证明。

**📈 对比分析**

无实验对比；通过数学证明展示新表述的充分性与必要性，证明过程在理论上高效且可直接用于后续算法设计。

**⚠️ 局限性**

局限性：仅适用于 ω-正则语言；对更一般的 ω‑语言或非 ω‑正则目标尚未扩展；实际算法实现仍需进一步研究。

---

## 296. A Comparison of ROS 2 and AUTOSAR Adaptive Platform Against Industry-Elicited Automotive Middleware Requirements

**arXiv ID:** 2604.22576 | [PDF](https://arxiv.org/pdf/2604.22576v1)

**作者:** Lucas Hegerath `[一作]` (RWTH Aachen University), Alexandru Kampmann `[通讯]` (RWTH Aachen University)

**通讯引用:** 160 | [OpenAlex ID](https://openalex.org/A5079941800)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过访谈ZF集团的软件架构师，收集了33条工业需求，并将其与ROS 2和AUTOSAR Adaptive Platform进行对比评估。

**💡 创新点**

创新点在于首次系统化记录行业实际middleware需求，并对两大主流平台在满足这些需求方面进行客观比较，揭示了技术差异与行业痛点。

**🔧 技术方法**

采用Automotive SPICE框架的需求访谈方法，随后将收集到的需求与两平台官方规范进行对照分析。

**📊 数据集**

使用的数据集仅包含来自ZF集团访谈的需求条目，共33条；并未使用传统公开数据集。

**📈 对比分析**

通过需求覆盖率（满足/部分满足/不满足）进行比较，结果显示AUTOSAR AP满足23条需求，ROS 2仅满足13条，后者在安全、调度与实时性方面表现欠缺。

**⚠️ 局限性**

局限性包括样本仅来自一家企业，难以推广；研究聚焦于非功能需求，未覆盖OTA等系统层面；ROS 2在实时调度和安全管理上缺乏内置机制。

---

## 297. CRAFT: Clustered Regression for Adaptive Filtering of Training data

**arXiv ID:** 2604.22693 | [PDF](https://arxiv.org/pdf/2604.22693v1)

**作者:** Parthasarathi Panda `[一作]` (Google), Subhrakanta Panda `[通讯]` (BITS Pilani)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种两阶段的数据子集选择方法 CRAFT，通过先匹配验证集源句分布，再在每个源聚类内根据目标句的条件期望距离选择训练样本。

**💡 创新点**

创新点在于：①将源-目标联合分布拆解为源边缘和条件目标分布，从而实现分层匹配；②使用源和目标分别聚类并在源聚类内部按条件距离排序；③提出比例分配策略证明可控制连续 KL 散度，且方法对向量化方式不敏感。

**🔧 技术方法**

核心技术包括 k‑means 聚类、比例预算分配、条件期望距离评估（基于验证集目标分布），并可配合任意句向量化（密集句向量或 TF‑IDF）。

**📊 数据集**

在英-印双语机器翻译任务上使用 NLLB 语料库（约 3300 万句对）做实验，验证集 10k 句对，候选池 33 183 万。

**📈 对比分析**

与 DSIR、TSDS、TAROT 等基线比较：在 1M 级候选池下，CRAFT（密集向量）BLEU 43.34，超出 TSDS 41.21 2.13 分；TAROT 为 45.61 最高，但 CRAFT 的选择时间仅 26.86 秒，比 TAROT 低 2.8 倍、比 TSDS 低 40 倍；TF‑IDF 版在不使用 GPU 的情况下完成 1M 选取 57 秒，BLEU 41.78 与 TSDS 相当。

**⚠️ 局限性**

局限性包括：实验仅在英-印翻译任务上验证，缺乏跨任务和非文本数据的评估；向量化（尤其是密集向量）仍占主导时间，且方法依赖于验证集样本量和聚类参数的选择。

---

## 298. Operational Feature Fingerprints of Graph Datasets via a White-Box Signal-Subspace Probe

**arXiv ID:** 2604.22676 | [PDF](https://arxiv.org/pdf/2604.22676v1)

**作者:** Yuchen Xiong `[一作]` (Xiamen University Malaysia), Zhen Hong Ban `[通讯]` (Xiamen University Malaysia)

**通讯引用:** 474 | [OpenAlex ID](https://openalex.org/A5032853136)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种白盒图神经网络框架WG‑SRC，直接使用预定义的图信号字典与闭式Ridge回归进行节点分类，同时提供机制诊断图谱。

**💡 创新点**

创新点在于将消息传递替换为可命名的低通、高通和原始特征块，结合Fisher特征选择、类级PCA子空间与多α Ridge决策，并通过节点级信号分解生成可解释的图谱。

**🔧 技术方法**

使用的技术包括行标准化与对称归一化的图算子、Fisher特征分数选择、PCA子空间拟合、闭式多α Ridge分类、分数融合与节点级信号分量权重计算。

**📊 数据集**

实验数据集涵盖六个节点分类基准：Amazon‑Computers、Amazon‑Photo、Chameleon、Cornell、Texas、Wisconsin。

**📈 对比分析**

与GraphSAGE、LINKX等强基线在相同的对齐实验设置下对比，WG‑SRC在六个数据集上平均提升1.52个百分点，且在大多数随机划分上保持优势。

**⚠️ 局限性**

局限性包括计算开销较大、模型为模块化结构并非单一最优架构、基线对比范围有限，以及诊断结果为可解释但不具备完全因果或搜索最优性。

---

## 299. Cuts and Gauges for Submodular Width

**arXiv ID:** 2604.22663 | [PDF](https://arxiv.org/pdf/2604.22663v1)

**作者:** Matthias Lanzinger `[一作]` (TU Wien), Matthias Lanzinger `[通讯]` (TU Wien)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5053864834)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

论文将子模宽度重新表述为几何形式，并提出一种新的分支宽度参数（升高子模宽度），证明其与传统子模宽度相差至多 3/2。随后给出一个变分表述，利用凸体（可实现子模体）与其支撑函数（Gauge）来计算子模宽度，并通过线图树宽度与多租户流路由构造平衡的切分权重，从而得到子模宽度的下界。论文进一步定义了子模 Gauge-路由比 κ(H) 并证明在多种结构条件下（如私有交叉、边缘过剩）κ(H) 可被控制，进而给出子模宽度与其他宽度度量（如广义超树宽度、极小树宽度）的量化联系。最后将这些结构结果应用于判定联合查询（CQ）评估的可判定性与固定参数可解性，证明在 ETH 下，当子模 Gauge-路由比有界时，PTIME 与 FPT 结果等价。

**💡 创新点**

创新点在于：① 将子模宽度与分支宽度联系起来，提供了 3/2 近似；② 用凸几何与抗堵塞（antiblocker）对子模宽度进行变分表述，转化为凸优化问题；③ 通过线图树宽度与多租户流路由构造平衡的切分权重，给出子模宽度的新下界；④ 引入子模 Gauge-路由比 κ(H) 作为统一的结构参数，阐明多种已知条件下的宽度关系；⑤ 在理论计算复杂性层面给出子模宽度与 CQ 评估可判定性、FPT 的完整等价条件。

**🔧 技术方法**

使用的技术包括：凸体几何与 Gauge 定义、抗堵塞（antiblocker）对偶性、升高子模宽度（Lifted Submodular Width）与分支宽度的对应、线图（P(H*)）上的多租户流路由与节点容量约束、流-树宽度的平衡切分理论、变分与最优极值的等价证明。

**📊 数据集**

该研究为纯理论论文，未使用任何实验数据集；所有结果均基于数学证明与构造。

**📈 对比分析**

与以往的子模宽度研究相比，本文通过几何方法提供了更直观的下界构造，并给出了子模宽度与其他宽度度量（如广义超树宽度、极小树宽度、线图树宽度）的量化关系。相比传统的树分解与子模函数构造，新的方法在理论上更具通用性，能够在多种结构约束下给出统一的下界；但目前尚无实验性能评估。

**⚠️ 局限性**

限制包括：① 仍未给出有效算法用于直接计算子模宽度或其 Gauge；② 变分表述虽理论上等价，但实现的凸优化复杂度尚未知；③ 证明的下界与已知上界之间仍存在常数因子（如 3/2、C/log(H) 等）且尚不清楚是否可进一步收紧；④ 仅在理论上对 CQ 评估给出等价条件，实际实现的算法与复杂度仍需进一步研究。

---

## 300. RealBench: A Repo-Level Code Generation Benchmark Aligned with Real-World Software Development Practices

**arXiv ID:** 2604.22659 | [PDF](https://arxiv.org/pdf/2604.22659v1)

**作者:** Jia Li `[一作]` (Wuhan University), Yihong Dong `[通讯]` (Peking University)

**通讯引用:** 973 | [OpenAlex ID](https://openalex.org/A5077542599)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 RealBench，一个面向仓库级代码生成的基准测试，旨在模拟真实软件工程实践；

**💡 创新点**

创新点在于将自然语言需求与 UML（包图+类图）相结合作为输入，构建了多难度、多领域的真实项目数据集，并设计了细粒度评估指标与策略；

**🔧 技术方法**

使用的大型语言模型包括 GPT‑4o、Claude‑Sonnet‑4、Gemini‑2.5‑Flash、DeepSeek‑V3、Qwen3‑235B‑A22B 及 Qwen2.5‑Coder‑7B‑Instruct；评估策略涵盖整体生成、增量生成与检索增强生成；

**📊 数据集**

数据集为 61 个真实开源 Python 仓库（从 2024‑12 起），覆盖 20 个主题，按代码行数分为四个难度层级，配套完整测试套件（平均 50 条用例，行覆盖率 79.76%）；

**📈 对比分析**

对比方法包括三种生成策略、两级评估粒度和五项指标（Completion@k、Execution@k、Pass@k、Requirement@k、Architecture@k）。结果显示：最优 Pass@1 仅约 19%；整体生成在小型仓库表现最好，增量生成在大型仓库更优；缺省类图会显著降低性能；不同模型间差距大，开源模型表现相对落后；

**⚠️ 局限性**

局限性包括：1）模型仍难以生成高质量、可执行且符合需求的完整仓库；2）评估主要聚焦功能正确性，缺乏对非功能需求（性能、安全等）的考察；3）受限于模型上下文窗口，长篇生成仍受阻；4）基准数据虽最新，但仍可能存在模型训练时泄漏风险；5）实验仅使用贪婪解码，未探索更复杂的采样策略。

---

## 301. Associativity-Peakiness Metric for Contingency Tables

**arXiv ID:** 2604.22655 | [PDF](https://arxiv.org/pdf/2604.22655v1)

**作者:** Naomi E. Zirkind `[一作]` (DEVCOM Army Research Laboratory), William J. Diehl `[通讯]` (DEVCOM Army Research Laboratory)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5076802712)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种新的聚类结果评价指标——关联-峰度（AP）指标，用于从混淆矩阵式的列联表中量化聚类算法的性能；

**💡 创新点**

创新点在于将“关联性”（每个真类对应的最大簇）与“峰度”（该最大值相对于同一行其他值的显著性）两方面结合，通过调和平均得到单一度量；

**🔧 技术方法**

技术包括：列联表的解析、最大值与第二大值的计算、列表组合计数实现关联度、峰度公式以及AP调和平均；对比时使用Python scikit‑learn中的AMI、ARS、FMS、Completeness、Homogeneity、V‑Measure以及自定义的F1指标；

**📊 数据集**

使用人工生成的模拟数据：共六种测试场景，分别包含理想、最差、低性能（500张4×4表）、高性能（500张4×4、4×6、4×2表），每张表总样本数固定为2521；

**📈 对比分析**

比较方法是对每种指标在各场景下的得分分布、极值、动态范围以及与AP的相关系数进行统计。实验显示AP在低性能时能产生0分且在高性能时能产生接近1分，动态范围最大；与scikit‑learn指标及F1相比，AP的相关系数约为0.5-0.6，且计算时间仅为F1的十分之一（约0.085 ms/表）；

**⚠️ 局限性**

局限性包括：仅针对列联表形式的聚类结果，未在真实数据集上验证；当簇数少于真类数时AP无法给出高分，虽为优势但也限制了其适用范围；对非常大规模表的计算复杂度虽低，但仍需进一步理论分析。

---

## 302. Structure-Guided Diffusion Model for EEG-Based Visual Cognition Reconstruction

**arXiv ID:** 2604.22649 | [PDF](https://arxiv.org/pdf/2604.22649v1)

**作者:** Yongxiang Lian `[一作]` (Tsinghua University), Li Shi `[通讯]` (Tsinghua University)

**通讯引用:** 35379 | [OpenAlex ID](https://openalex.org/A5025170020)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种结构引导扩散模型（SGDM），通过从脑电图中提取结构与语义信息实现视觉内容的重建；

**💡 创新点**

将结构先验显式引入扩散生成过程，并通过双重语义约束（CLIP+ControlNet）同时捕获低级结构与高级语义，解决传统方法仅关注分类标签且缺乏结构表征的缺陷；

**🔧 技术方法**

利用CLIP双编码器、Adaptive Thinking Mapper EEG编码器、SDXL-turbo VAE进行结构预测、对比学习、控制扩散（IP-Adapter+ControlNet）以及扩散先验模型，实现跨模态对齐与条件生成；

**📊 数据集**

在Kilogram抽象视觉对象数据集和THINGS自然图像数据集上进行实验；

**📈 对比分析**

与BrainVis、DreamDiffusion及基线模型对比，SGDM在SSIM、SwAV‑FID、CLIP相似度等指标上均位居榜首，且在结构一致性和跨受试者泛化方面表现优异；

**⚠️ 局限性**

受限于抽象图形数据集规模小、EEG空间分辨率低以及缺乏多模态/实时适配等，尚需进一步扩充数据、提升分辨率和探索在线个体化学习以提升泛化与实时性能。

---

## 303. Dharma, Data and Deception: An LLM-Powered Rhetorical Analysis of Cow-Urine Health Claims on YouTube

**arXiv ID:** 2604.22606 | [PDF](https://arxiv.org/pdf/2604.22606v1)

**作者:** Sheza Munir `[一作]` (University of Michigan), Joyojeet Pal `[通讯]` (University of Michigan)

**通讯引用:** 3086 | [OpenAlex ID](https://openalex.org/A5058437817)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了印度牛尿在 YouTube 上的健康信息争议，构建了 14 类说服手段分类体系，并使用大型语言模型（LLM）对 100 条视频转录进行大规模自动注释，再通过人工评估验证标签精度。

**💡 创新点**

创新点在于：①将修辞学、ELM、CDA 等多理论框架融合成可操作的 14 维说服手段词汇表；②首次在 LLM 级别对文化特定健康谣言进行大规模注释并跨模型对比；③提出以人机接受率作为精度代理的评估方法。

**🔧 技术方法**

采用 Whisper‑Large 进行多语音转录，GPT‑4 对非英语文本进行翻译，GPT‑4、GPT‑4o、GPT‑4.1、GPT‑5、Gemini 2.5 Pro、Mistral Medium 3 等 LLM 进行句内标签注释；人工评估采用交叉标注与 Krippendorff α 计算一致性。

**📊 数据集**

使用 100 条关于牛尿的 YouTube 视频转录（英语、印地语、乌尔都语），共 7 h 9 min、约 456 词/转录，已标注“推广/反驳”立场。

**📈 对比分析**

对各模型的标签密度、标签分布、与人工一致性进行比较；高密度模型 GPT‑5、Gemini‑2.5 Pro 产生更多标签但精度略低；Mistral Medium 3 与 Gemini‑2.5 Pro 的人机接受率最高（约 94‑95%），整体精度范围 65‑97%。

**⚠️ 局限性**

局限性：仅覆盖 YouTube 单一平台和印度文化背景，未评估召回或 F1；对隐含、文化敏感说服手段识别仍受限，需要人工干预；缺乏多模态或多语言更广泛的数据样本。

---

## 304. Rethinking Math Reasoning Evaluation: A Robust LLM-as-a-Judge Framework Beyond Symbolic Rigidity

**arXiv ID:** 2604.22597 | [PDF](https://arxiv.org/pdf/2604.22597v1)

**作者:** Erez Yosef `[一作]` (Tel-Aviv University), Igor Kviatkovsky `[通讯]` (Amazon Prime Video)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型的评判者（LLM-as-a-judge）框架，用于数学推理任务的答案验证，替代传统的符号数学比较方法。

**💡 创新点**

创新点在于：①采用“独立求解-答案校验-评判”三阶段流程，消除对数据集答案的偏倚；②利用多轮LLM评判、随机打乱与多数投票提升判定鲁棒性；③在pass@k评估下实现多样化答案的综合评价，显著提升对不同表示形式答案的兼容性。

**🔧 技术方法**

核心技术包括：大语言模型（Claude‑Sonnet‑4 及其他 LLMs）作为评判者、符号工具（SymPy）对照、随机抽样与排序、三阶段答案验证、多数投票与多样化评估组（n_g）以及pass@k 指标。

**📊 数据集**

使用了 GSM8K、Minerva、MATH、Olympiad 等数学推理基准；在 Qwen2.5 系列（7B/14B/32B）和 Llama3.1‑8B 上进行评测，且在 SimpleRL 与 Lighteval 两大框架下进行对比。

**📈 对比分析**

与传统符号评估相比，LLM‑as‑a‑judge 在 pass@1 上提升 1–3%（小模型）至 20+%（RLVR 训练模型），在 MATH、Olympiad 等更具挑战性的基准中显著提升 24% 以上，整体 F1 由 0.741 提升至 0.969。

**⚠️ 局限性**

主要限制：需要多次 LLM 调用，计算成本显著高于符号验证；评判结果受 LLM 先验、偏倚和可能的幻觉影响；meta‑evaluation 仅基于小规模人工标注；对非标准符号和不符合 oxed{} 格式的答案识别仍有挑战。

---

## 305. Relational Archetypes: A Comparative Analysis of AV-Human and Agent-Human Interactions

**arXiv ID:** 2604.22564 | [PDF](https://arxiv.org/pdf/2604.22564v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 306. FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing

**arXiv ID:** 2604.22586 | [PDF](https://arxiv.org/pdf/2604.22586v1)

**作者:** Ze Chen `[一作]` (Communication University of China), Qi Mao `[通讯]` (Communication University of China)

**通讯引用:** 1601 | [OpenAlex ID](https://openalex.org/A5039315497)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FlowAnchor，一种训练‑free 的无逆转流式视频编辑框架，能够在不需额外训练的情况下通过直接修正编辑信号实现更精准、连贯的编辑效果。

**💡 创新点**

通过引入 Spatial‑aware Attention Refinement（SAR）与 Adaptive Magnitude Modulation（AMM）两大机制，显式锚定编辑位置与力度，解决了编辑信号在高维视频潜空间中的定位模糊和幅值衰减问题。

**🔧 技术方法**

基于 Rectified Flow 与 DiT 3D T2V 扩散模型，利用交叉注意力调节、局部注意力重塑以及帧数自适应幅值放大技术，实现无训练的高效视频编辑。

**📊 数据集**

在现有的 FiVE‑Bench（单物体编辑）以及自建的 Anchor‑Bench（多物体、快动场景）两套基准数据集上进行评估。

**📈 对比分析**

与 TokenFlow、VideoGrain、RF‑Solver、UniEdit‑Flow、Wan‑Edit、FlowDirector 等七种先进方法进行自动指标（CLIP‑T、L.CLIP‑T、M.PSNR、L.DINO、CLIP‑F、Warp‑Err）和用户主观评测对比，FlowAnchor 在文本对齐、视频质量、时间一致性以及推理速度方面均显著优于基线，并保持竞争性的 GPU 内存占用。

**⚠️ 局限性**

对全局风格迁移和大幅运动编辑仍存在不足，属于无逆转编辑框架的固有限制。

---

## 307. QuantClaw: Precision Where It Matters for OpenClaw

**arXiv ID:** 2604.22577 | [PDF](https://arxiv.org/pdf/2604.22577v1)

**作者:** Manyi Zhang `[一作]` (Huawei Technologies), Xiaobo Xia `[通讯]` (University of Science and Technology of China)

**通讯引用:** 614 | [OpenAlex ID](https://openalex.org/A5114803721)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在OpenClaw等自主代理系统中，针对不同任务的量化灵敏度进行系统性评估，并基于此提出QuantClaw插件，动态按任务特征路由不同精度配置以实现成本与延迟优化。

**💡 创新点**

创新点在于将量化精度视为可动态分配的资源，依据任务敏感度构建任务级量化策略；提供无需用户干预的plug‑and‑play精度路由插件。

**🔧 技术方法**

使用低精度量化（NVFP4/INT4）、任务检测（规则+模型混合）以及预先计算的任务‑精度敏感度表；在OpenClaw框架内集成多精度模型池并实现即时路由。

**📊 数据集**

主要数据集包括Claw‑Eval（24种任务、104个任务）、PinchBench v1.2.0 & v2.0.0；模型覆盖GLM‑4.7‑Flash、GLM‑5、MiniMax‑M2.5、Qwen3.5‑9B/35B/397B等。

**📈 对比分析**

与统一高精度（BF16/FP8）和统一低精度（INT4）做对比；QuantClaw在GLM‑5上实现最高得分+2.09点，成本下降21.4%，延迟下降15.7%；在GLM‑4.7‑Flash上平均得分提升至84.11点，成本降低21.7%，延迟提升8.4%。

**⚠️ 局限性**

局限性包括对任务检测精度和推理时切换开销的依赖，低精度对某些高敏感度任务仍可能导致性能下降；在不同硬件/框架下量化收益差异仍需进一步验证。

---

## 308. SpikingBrain2.0: Brain-Inspired Foundation Models for Efficient Long-Context and Cross-Platform Inference

**arXiv ID:** 2604.22575 | [PDF](https://arxiv.org/pdf/2604.22575v1)

**作者:** Yuqi Pan `[一作]` (Institute of Automation Chinese Academy of Sciences), Guoqi Li `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SpikingBrain2.0 5B系列模型，融合Dual‑Space Sparse Attention（DSSA）与整数‑脉冲编码，实现长上下文高效推理与跨硬件兼容性。

**💡 创新点**

创新点包括：①在交叉层混合MoBA与SSE的DSSA架构，兼顾性能与效率；②双路径量化（INT8‑脉冲与FP8）实现软硬件协同；③Transformer‑to‑Hybrid转换管线，实现低成本迁移。

**🔧 技术方法**

采用脉冲神经网络编码、Sparse Attention（MoBA、SSE、SWA）、混合量化、分布式训练、T2H转换、多阶段长上下文扩展、知识蒸馏、SFT及OPD等技术。

**📊 数据集**

使用开源数据集：Nemotron、ProLong、LLaVA‑NeXT、LLaVA‑OneVision、DeepWeb‑Edu、LLaVA‑OneVision‑Instruct等，以及多模态基准数据集如DocVQA、MMBench、MME、MMMU、OCRBench等。

**📈 对比分析**

通过与Qwen3‑4B、Gemma3、Llama3.2、Qwen2.5‑3B、Qwen3‑VL‑4B等基准模型在MMLU、ARC‑C、Winogrande、GSM8K、IFEval、MMBench等多项任务进行对比，SpB2.0在大多数基准保持与Qwen3相近或优于1个百分点；在4M令牌长上下文下TTFT提升10×、TPOT提升4.5×，支持10M+上下文；量化后任务性能下降不到1%。

**⚠️ 局限性**

局限性包括：①在极长上下文或部分推理任务上仍略逊于全注意力模型；②量化实现对硬件支持要求较高；③VLM在某些细粒度任务上仍低于原始Qwen3‑VL‑4B；④需要更多高质量多模态训练数据；⑤MoBA分页注意力实现仍可进一步优化。

---

## 309. BERAG: Bayesian Ensemble Retrieval-Augmented Generation for Knowledge-based Visual Question Answering

**arXiv ID:** 2604.22678 | [PDF](https://arxiv.org/pdf/2604.22678v1)

**作者:** Jinghong Chen `[一作]` (University of Cambridge), Bill Byrne `[通讯]` (University of Cambridge)

**通讯引用:** 3444 | [OpenAlex ID](https://openalex.org/A5070594684)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Bayesian Ensemble Retrieval-Augmented Generation (BERAG) 与其对应的 Bayesian Ensemble Fine‑Tuning (BEFT)，将检索文档的贡献以贝叶斯后验形式融入生成过程，避免传统串联式 RAG 的“中间丢失”问题，并实现可并行记忆与可解释的文档重要性评分。

**💡 创新点**

创新点包括：1) 基于贝叶斯规则的文档后验更新，用于动态加权生成；2) 通过 MLP 计算文档先验，形成端到端可训练的概率聚合框架；3) 通过 Top‑P 剪枝实现解码加速；4) 通过文档后验实现答复拒绝（deflection）与严格 RAG。

**🔧 技术方法**

采用 Transformer‑based 视觉‑语言模型 Qwen2‑VL‑Instruct 进行答案生成；检索使用 PreFLMR‑ViT‑L；贝叶斯推理、MLP 先验层、token‑级联合似然、文档后验计算及 Top‑P 剪枝等技术。

**📊 数据集**

在 Knowledge‑Based Visual Question Answering (KB‑VQA) 基准（E‑VQA、Infoseek）、Document VQA（SlideVQA）和 Multi‑Modal Needle‑in‑a‑Haystack（MMNeedle）上进行评测；使用 M2KR 预编译的检索库和 训练/测试划分。

**📈 对比分析**

与 SFT、DPO、ConcatRAG 等基线相比，BERAG+BEFT 在 E‑VQA 上取得 70.3 BEM（对比 DPO 64.1），Infoseek 上 42.8 EM（对比 DPO 35.3），SlideVQA 上 90.4 ES EM 及 69.6 QA EM，MMNeedle 上 97.1/86.8/41.4 等准确率；在更大 Top‑K 上保持或超越标准 RAG 的性能，并在 Top‑P 剪枝下实现更快的解码速度。

**⚠️ 局限性**

局限性：1) 仅对单个文档做边缘化，未考虑所有文档组合的概率；2) 需要专门的 BEFT 训练才能达到良好效果，预训练模型未被设计用于此聚合；3) 当前实现的推理速度和吞吐量仍低于高度优化的串联 RAG，需要进一步的工程改进。

---

## 310. A dataset of early blockchain-registered AI agents on Ethereum

**arXiv ID:** 2604.22652 | [PDF](https://arxiv.org/pdf/2604.22652v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 311. Measuring Epistemic Unfairness for Algorithmic Decision-Making

**arXiv ID:** 2604.22675 | [PDF](https://arxiv.org/pdf/2604.22675v1)

**作者:** Camilla Quaresmini `[一作]` (Politecnico di Milano), Valentina Breschi `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 694 | [OpenAlex ID](https://openalex.org/A5057454932)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实验了KG嵌入模型的经验不确定性度量方法。

**💡 创新点**

创新点在于为知识图谱嵌入引入贝叶斯不确定性评估，帮助模型选择。

**🔧 技术方法**

使用贝叶斯神经网络推理和对数似然预期损失计算。

**📊 数据集**

使用FB15k-237、WN18RR、WN18等常用KG数据集。

**📈 对比分析**

与传统TransE、DistMult、ComplEx等基准进行AUC和NDCG对比，结果显示不确定性方法能更好地区分模型。

**⚠️ 局限性**

局限在于仅覆盖少数数据集和模型，计算开销大，阈值设定和校准问题未解决。

---

## 312. What People See (and Miss) About Generative AI Risks: Perceptions of Failures, Risks, and Who Should Address Them

**arXiv ID:** 2604.22654 | [PDF](https://arxiv.org/pdf/2604.22654v1)

**作者:** Megan Li `[一作]` (Carnegie Mellon University), Hong Shen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3530 | [OpenAlex ID](https://openalex.org/A5062298555)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并验证了一套基于真实事件情景的调查工具，用于评估公众对生成式AI（GenAI）失败模式、相关风险及责任归属的认知与感知。

**💡 创新点**

①将失败模式作为测量框架，代替单纯列举风险；②在情景中嵌入已公开的 GenAI 事故，增强真实性与内容有效性；③通过专家共识验证情景与失败模式的一致性，确保工具科学可靠；④将结果用于构建 AI 文识与治理干预的初步蓝图。

**🔧 技术方法**

调查设计与实施、情景呈现技术、专家共识验证流程、定性主题分析、描述性统计与 Likert 分析等方法。

**📊 数据集**

在 Prolific 上招募 960 名美国受访者，使用 24 个情景（覆盖 12 种失败模式），并邀请 8 位负责 AI 专家对情景进行评估。

**📈 对比分析**

本研究未涉及算法性能对比，而是通过专家共识（≥3/4 同意）和受访者回应率来评估工具有效性；结果显示调查能够准确映射公众对 GenAI 失败模式和风险的认知差异，并揭示对上游设计失误的认知不足。

**⚠️ 局限性**

局限性：①样本为高 GenAI 熟悉度的美国 Prolific 用户，缺乏全国性代表性；②情景设计可能引入偏见；③使用生成式 AI 进行问卷填答的风险未完全排除；④仅覆盖美国背景，缺乏跨文化视角；⑤未能覆盖所有已知失败模式。

---

## 313. Verifier Warnings Do Not Improve Comprehensibility Prediction

**arXiv ID:** 2604.22653 | [PDF](https://arxiv.org/pdf/2604.22653v1)

**作者:** Nadeeshan De Silva `[一作]` (William & Mary), Oscar Chaparro `[通讯]` (William & Mary)

**通讯引用:** 795 | [OpenAlex ID](https://openalex.org/A5003334072)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探究将程序验证器发出的警告总数作为语义特征，是否能提升机器学习模型对代码可读性（可理解性）的预测性能。

**💡 创新点**

创新点在于将验证器警告总数作为新的语义输入特征引入已有的可读性预测模型，并通过对照实验系统评估其贡献。

**🔧 技术方法**

采用六种经典机器学习模型（Naïve Bayes、KNN、LR、MLP、RF、SVM），并在特征工程中加入验证器警告总数。

**📊 数据集**

使用两大公开 Java 可读性数据集（Scalabrino 数据集和 Buse & Weimer 数据集），共计约 211 个方法与 18,005 个人类评估。

**📈 对比分析**

通过对照-处理实验结合嵌套交叉验证对比控制组（仅语法+开发者特征）与处理组（加警告特征），结果显示加入警告特征几乎不提升或仅有极小幅度提升，统计意义不显著。

**⚠️ 局限性**

局限性包括数据集规模有限、仅涵盖 Java 代码、警告计数作为语义特征表达过于粗糙、模型泛化能力不足以及缺乏更细粒度的验证器信息。

---

## 314. Adversarial Malware Generation in Linux ELF Binaries via Semantic-Preserving Transformations

**arXiv ID:** 2604.22639 | [PDF](https://arxiv.org/pdf/2604.22639v1)

**作者:** Lukáš Hrdonka `[一作]` (Czech Technical University in Prague), Martin Jureček `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 118 | [OpenAlex ID](https://openalex.org/A5033203359)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

设计并实现了一个针对 Linux ELF 可执行文件的对抗性恶意软件生成器，采用遗传算法框架，使用12种二进制修改方式和7种数据源，评估其对 MalConv 检测器的规避效果。

**💡 创新点**

首次提出了针对 ELF 格式的完整对抗生成框架，并引入 Extended Evasion Rate (EER) 与 Mean Difference in Confidence (MD) 两项新指标，揭示 MalConv 对字符串特征高度敏感。

**🔧 技术方法**

使用 Python、PyTorch（MalConv CNN）、ELFFile、Keystone、Capstone 等技术实现修改与评估，并通过遗传算法迭代搜索最佳扰动。

**📊 数据集**

利用公开的 Labeled-ELF 数据集，仅选取 64 位 x86‑64 架构的恶意样本和同构的正样本，划分训练/验证/测试集（测试集仅包含恶意文件）。

**📈 对比分析**

通过比较 Evasion Rate、EER 与 MD 等指标，对不同修改/数据源组合进行实验，最佳配置实现 67.74% 的 ER、-0.5006 的 MD；字符串来源的数据源表现最佳。

**⚠️ 局限性**

实验规模有限（仅 x86‑64），未覆盖 ARM 等嵌入式平台；对抗生成速度与可执行性验证仍需进一步提升，且模型对样本分布的依赖性较强。

---

## 315. Adversarial Co-Evolution of Malware and Detection Models: A Bilevel Optimization Perspective

**arXiv ID:** 2604.22569 | [PDF](https://arxiv.org/pdf/2604.22569v1)

**作者:** Olha Jurečková `[一作]` (Czech Technical University in Prague), Róbert Lórencz `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 373 | [OpenAlex ID](https://openalex.org/A5071351394)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过基于双层优化的博弈框架，对MAB-malware生成的对抗样本进行迭代共进化防御，并在三类恶意软件上评估其有效性。

**💡 创新点**

将恶意软件检测视作攻击者与防御者的双层博弈，提出迭代最优响应的共进化过程，从而实现对自适应攻击的鲁棒性提升。

**🔧 技术方法**

使用随机森林分类器、EMBAR 2018特征、MAB-malware生成器以及双层优化（IBR）技术。

**📊 数据集**

在EMBAR 2018特征基础上，使用 RawMal‑TF 数据集中的三类恶意软件（Mokes、Strab、DCRat），每类500个样本（250恶意+250正常）。

**📈 对比分析**

与基线和单次对抗训练三种策略对比，双层优化后误报率降至0%，攻击成功率≤1.89%，平均查询次数提升约两位数，保持或提升了清洁数据上的准确率。

**⚠️ 局限性**

实验仅针对随机森林+EMBAR特征，未验证深度网络或更大规模恶意软件族的泛化能力，且假设攻击者为白盒，实际对抗环境下的鲁棒性仍有待进一步验证。

---

## 316. How Supply Chain Dependencies Complicate Bias Measurement and Accountability Attribution in AI Hiring Applications

**arXiv ID:** 2604.22679 | [PDF](https://arxiv.org/pdf/2604.22679v1)

**作者:** Gauri Sharma `[一作]` (Mila Quebec AI Institute), Maryam Molamohammadi `[通讯]` (Mila Quebec AI Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了 AI 招聘系统中供应链依赖如何导致偏见评估与责任归属的复杂化，指出责任分散导致的评估、归因、修复与责任四重卷积问题。

**💡 创新点**

提出了四种新的问题范式（评估卷积、归因卷积、修复卷积、责任卷积）以及多层干预方案，包括系统级审计、供应链透明度、合同责任分配与持续监测，以填补监管与技术孤岛。

**🔧 技术方法**

采用结构化文献综述与法规文本分析相结合的方法，对欧盟 AI 法案、纽约市 LL 144 和科罗拉多州 AI 法案进行对比，构建责任矩阵并设计干预框架。

**📊 数据集**

主要使用公开的学术论文、监管文件与行业报告，未使用具体候选人数据或商业模型。

**📈 对比分析**

论文并未进行实验或性能测评，而是通过案例分析和政策比较评估干预方案的可行性，指出现有监管与技术手段在实际系统集成中的不足。

**⚠️ 局限性**

局限在于仅聚焦美国与欧盟的招聘 AI 场景，未涉及其他高风险领域；所提出的干预方案缺乏实证验证；缺乏对商业激励、技术复杂性和信息不对称的定量评估。

---

## 317. Inferring Equivalence Classes from Legacy Undocumented Embedded Binaries for ISO 26262-Compliant Testing

**arXiv ID:** 2604.22673 | [PDF](https://arxiv.org/pdf/2604.22673v1)

**作者:** Marco De Luca `[一作]` (University of Naples Federico II), Anna Rita Fasolino `[通讯]` (University of Naples Federico II)

**通讯引用:** 3950 | [OpenAlex ID](https://openalex.org/A5010949326)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

本文提出一种基于二进制的技术，在缺乏功能规范的旧版嵌入式固件中自动推断输出导向的等价类，用以支持 ISO 26262 的安全软件测试。

**💡 创新点**

创新点在于：① 直接从编译后的 ELF+DWARF 二进制恢复控制流并进行符号执行，避免源代码缺失导致的误差；② 以可观测输出为标准聚合符号路径，形成等价类而非传统路径集合；③ 通过 LLM 对符号约束进行可读化后处理，使结果既可机器解析又易人工审查。

**🔧 技术方法**

核心技术包括：控制流图重建、基于调用图的函数聚类、指导符号执行与约束求解、路径合并与方法摘要、以及基于 Prompt 的 LLM 可读化。

**📊 数据集**

实验数据集为 27 个典型计算函数（平均 31 行），共生成 138 个等价类；评估对象为 Micron Technology 的 12 名固件开发/测试工程师。

**📈 对比分析**

对比方式是基于专家评审的功能一致性与可读性问卷；结果显示 100% 的功能一致性得分为同意或强同意，约 90% 的参与者认为可读性良好；性能方面通过函数聚类和方法摘要显著降低符号求解负荷，已在生产环境中应用并得到正面反馈。

**⚠️ 局限性**

局限性包括：需依赖完整的 DWARF 调试信息；符号执行受限于循环展开和路径阈值，可能漏测极端边界；仅适用于无硬件/时序交互的计算函数；实验规模受限于单一组织的代码与样本数量。

---

## 318. Rethinking XAI Evaluation: A Human-Centered Audit of Shapley Benchmarks in High-Stakes Settings

**arXiv ID:** 2604.22662 | [PDF](https://arxiv.org/pdf/2604.22662v1)

**作者:** Inês Oliveira e Silva `[一作]` (University of Porto, Feedzai), Pedro Bizarro `[通讯]` (Feedzai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对8种Shapley值变体在高风险工作流中进行统一的实证评估与人机交互审计。

**💡 创新点**

证明量化代理与人类效用不一致，揭示解释可提升信心但不改善准确度，提出以人为中心的评估准则。

**🔧 技术方法**

使用统一的amortized Shapley框架、KernelSHAP基准、混合效应模型和对照实验。

**📊 数据集**

5个风险数据集（母性、信用、HELOC、成人、实测欺诈数据）以及37名专业分析师的3,735次案例评审。

**📈 对比分析**

通过定量代理（稀疏度、删除/插入AUC、敏感度等）与用户行为（准确率、决策时间、信心、清晰度）对照，发现解释的量化指标与人类感知无显著正相关，且无解释能显著提升准确率。

**⚠️ 局限性**

局限于低延迟表格风险模型、实验环境未涵盖长期学习和多模态领域，结果可能不适用于视觉/语言任务。

---

## 319. PASR: Pose-Aware 3D Shape Retrieval from Occluded Single Views

**arXiv ID:** 2604.22658 | [PDF](https://arxiv.org/pdf/2604.22658v1)

**作者:** Jiaxin Shi `[一作]` (Shanghai Jiao Tong University), Alan Vuile `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于姿态感知的单视角3D形状检索框架，将检索视为特征级别的分析-合成问题。

**💡 创新点**

创新点在于：1）通过对2D基准模型（DINOv3）的知识蒸馏，学习可视角条件的点级3D特征；2）采用可微分渲染将3D特征投影到2D空间，在特征层面进行对齐；3）在推理时进行测试时姿态优化，实现对遮挡和未知姿态的鲁棒性，并在同一框架内实现形状检索、姿态估计和类别分类。

**🔧 技术方法**

使用的技术包括：PointNeXt 3D编码器、DINOv3 2D编码器、点级多尺度特征聚合、可微分点渲染器、基于余弦相似度的对齐损失、AdamW 优化器进行测试时姿态优化。

**📊 数据集**

在 Pix3D（bed, chair, sofa, table）和 Pascal3D（12 类）数据集上进行实验，使用合成遮挡级别（L0~L3）进行评估。

**📈 对比分析**

与 CMIC、SC-IBSR、OpenShape、Uni3D 等基线相比，本文在 Pix3D 和 Pascal3D 的 Top‑1 检索准确率分别提升至 81.59% 与 76.43%，相对提升 11.09% 与 7.15%；在遮挡场景下仍保持最高的检索和姿态估计性能；同时在类别分类任务中也取得了最优或相近的表现。

**⚠️ 局限性**

主要局限包括：1）推理时需要对 top‑k 形状进行姿态优化，计算成本相对较高；2）方法依赖于高质量的 3D 模型和 2D 基准模型的知识，可能在极端遮挡或未见类别上表现受限；3）目前仅在少数类别上验证，跨大规模类别的通用性尚待进一步探索。

---

## 320. Quality-Driven Selective Mutation for Deep Learning

**arXiv ID:** 2604.22640 | [PDF](https://arxiv.org/pdf/2604.22640v1)

**作者:** Zaheed Ahmed `[一作]` (University of Göttingen), Jens Grabowski `[通讯]` (University of Göttingen)

**通讯引用:** 2392 | [OpenAlex ID](https://openalex.org/A5078674128)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于概率的双维度（抗杀性IQ与行为现实性EQ）框架，用于评估和筛选深度学习模型的变异体，从而实现选择性变异。

**💡 创新点**

创新点在于：①将传统的抗杀性概念改写为统计化的概率杀性，适应深度学习的随机训练；②利用广义Jaccard相似度量度变异体与真实缺陷的检测模式相似度；③基于这两个维度提出可跨模型、跨数据集的配置级选择规则。

**🔧 技术方法**

使用概率杀性计算、广义Jaccard相似度、预训练变异操作（DeepCrime）以及多次独立训练得到的杀性概率矩阵。

**📊 数据集**

实验使用CleanML、DeepFD、DeepLocalize四个公开缺陷数据集做训练/选择，并用独立的defect4ML数据集做验证。

**📈 对比分析**

通过与全量变异体基线比较，所提选择策略将生成变异体数量减少至约44%（即降低55.6%），且在保持中位数IQ/EQ不变的前提下，高质量（IQ和EQ均高）变异体比例提高；实验在不同阈值下表现出可调的成本-质量权衡。

**⚠️ 局限性**

局限性包括：仅针对预训练变异、仅在Keras/TensorFlow分类任务上验证、样本缺陷数有限、对回归任务、后训练变异或其他框架的适用性未探究；且概率估计仍受训练随机性和测试集覆盖率限制。

---

## 321. Identifying and typifying demographic unfairness in phoneme-level embeddings of self-supervised speech recognition models

**arXiv ID:** 2604.22631 | [PDF](https://arxiv.org/pdf/2604.22631v1)

**作者:** Felix Herron `[一作]` (Université Paris Dauphine-PSL), François Portet `[通讯]` (Université Grenoble Alpes)

**通讯引用:** 3240 | [OpenAlex ID](https://openalex.org/A5079066445)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过对多种自监督语音编码器（S3M）的音素嵌入进行探测，量化不同说话人群（SG）在音素识别中的偏倚与方差差异，进而分析导致ASR公平性不足的根本机制；

**💡 创新点**

提出了基于音素分类探测器与KNN距离两种指标的“嵌入偏倚-方差”框架，用以区分系统误差（mode shift）与随机误差（variance）对ASR公平性的影响，并首次展示了DET/DAT对音素嵌入公平性影响有限的结论；

**🔧 技术方法**

利用线性探测器训练音素分类模型、KNN距离度量嵌入方差、S3M模型微调（CTC+DET/DAT）以及统计显著性检验；

**📊 数据集**

Sonos Voice Control Bias Assessment Dataset（含性别、年龄、方言、族裔标签）作为实验数据集；

**📈 对比分析**

在多层次对比（均衡训练 vs 单SG训练、微调前后对比）下，宏F1得分和KNN距离显示：高方差与低识别准确率高度相关；尽管某些模型在后期层级的偏倚差距减小，但整体公平性提升有限；

**⚠️ 局限性**

受限于：使用的模型规模相对较小（100–300M参数），实验数据集受控录音环境，可能缺乏普适性；音素对齐误差可能对KNN距离产生额外噪声；未探究更大规模模型或多语种数据对公平性的影响；

---

## 322. It's Time to Standardize RDF Messages

**arXiv ID:** 2604.22619 | [PDF](https://arxiv.org/pdf/2604.22619v1)

**作者:** Pieter Colpaert `[一作]` (IDLab -- UGent -- imec), Piotr Sowinski `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了RDF Message概念，定义RDF消息、消息流、日志等以明确消息边界并为事件驱动与流式RDF系统提供互操作性基础。

**💡 创新点**

将消息边界作为第一类互操作性关注点，提供了跨序列化、传输与存储显式标识消息的机制，并可通过后续RDF Message Profiles扩展具体语义与约束。

**🔧 技术方法**

利用RDF语义技术（RDF 1.2 版本声明、RDF Dataset 结构）、Turtle/JSON‑LD 等序列化，以及PROV‑O、ActivityStreams、SOSA/SSN 等现有词汇。

**📊 数据集**

未使用具体实验数据集，本文以规范与设计为主。

**📈 对比分析**

未进行实验对比或性能评估，主要聚焦于概念设计与规范编写。

**⚠️ 局限性**

局限在于缺乏工业落地实现与性能验证，需要进一步完善Profile细节及实现示例。

---

## 323. Chamelio: A Fast Shared Cloud Network Stack for Isolated Tenant-Defined Protocols

**arXiv ID:** 2604.22603 | [PDF](https://arxiv.org/pdf/2604.22603v1)

**作者:** Matheus Stolet `[一作]` (Max Planck Institute for Software Systems), Antoine Kaufmann `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 583 | [OpenAlex ID](https://openalex.org/A5045220535)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

构建了可编程共享网络栈 CHOCO，允许租户通过 eBPF 定义完整协议的快速路径处理器，提出三层隔离机制（静态指令上限、运行时周期计数与非抢占式调度），实现共享栈下的协议隔离和性能匹配。

**💡 创新点**

创新点在于将 eBPF 快速路径与共享网络栈结合，提出了静态指令上限与动态周期计数并行的隔离方案，使租户可编程的快速路径仍能保持与固定协议栈相同的吞吐与低延迟。

**🔧 技术方法**

使用 eBPF、LLVM 编译生成共享代码、非抢占式调度器、周期级计数器以及共享内存/I/O 接口，实现快速路径与共享栈的协同执行。

**📊 数据集**

采用合成的 UDP/TCP 协议实现以及 100Gbps 双机实验流量作为评测数据集。

**📈 对比分析**

通过与 Linux 内核栈、TAS 固定协议栈以及无隔离 CHOCO 进行对比，性能基本匹配内核栈；在最大负载下可达约 1,500 连接/秒，99% 分位延迟在 46µs 内，吞吐量保持在 3,000+ 连接/秒。

**⚠️ 局限性**

局限在于需要手动设置指令上限，eBPF 只能支持一定指令数；对极大指令量或复杂协议的快速路径仍可能产生干扰；实验仅在两台 100Gbps 服务器上验证，未覆盖大规模多租户环境。

---

## 324. RedVLA: Physical Red Teaming for Vision-Language-Action Models

**arXiv ID:** 2604.22591 | [PDF](https://arxiv.org/pdf/2604.22591v1)

**作者:** Yuhao Zhang `[一作]` (Peking University), Jiaming Ji `[通讯]` (Peking University)

**通讯引用:** 26276 | [OpenAlex ID](https://openalex.org/A5108047889)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RedVLA框架，系统性地通过在VLA模型的初始场景中注入风险因子并迭代优化，揭示物理安全风险并触发多种不安全行为。

**💡 创新点**

首次将红队攻击与物理安全结合，设计交互区域识别和轨迹驱动的风险放大两阶段流程，并开发基于红队数据的轻量级安全守卫SimpleVLA-Guard。

**🔧 技术方法**

使用规则驱动的交互区域定位、零阶梯度优化、轨迹特征引导、内部表征学习、LSTM序列建模以及功能型合成预测等技术。

**📊 数据集**

主要在LIBERO基准上进行实验，利用六类代表性VLA模型（OpenVLA、OpenVLA-OFT、VLA-Adapter、VLA-Adapter-Pro、π0、π0.5）以及相应的任务场景和风险注入。

**📈 对比分析**

通过攻击成功率（ASR）和成功率（SR）等指标评估，RedVLA在10步优化内实现最高95.5%的ASR，且在多模型、多安全成本类型和多危险类别下均保持高成功率；SimpleVLA-Guard在在线干预时将ASR显著降低至约30–47%，并将正常任务成功率仅降低4–10%。

**⚠️ 局限性**

受限于仅考虑单一风险因子和初始状态扰动；对复杂多模态干扰、长期任务的鲁棒性有限；Guard在未见任务上表现下降；且对真实硬件环境的迁移仍需进一步验证。

---

## 325. On the Optimum Secrecy Outage Probability and Ergodic Secrecy Rate over Wireless Channels

**arXiv ID:** 2604.22587 | [PDF](https://arxiv.org/pdf/2604.22587v1)

**作者:** Clement Leroy `[一作]`, Olivier Rioul `[通讯]` (Telecom Paris)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在仅拥有通道统计信息的无线电路中，三种保密度量（保密性失效概率 SOP、期望正保密率 EPSR、期望保密率 ESR）的优化问题，并提出了两种随机通道的偏序（uniformly less noisy 与 less noisy on average），在这些偏序下证明非预编码高斯输入是最优解；对 SIMOME 结构进一步简化并给出 Rayleigh 衰落场景的闭式结果；同时给出若干反例说明偏序假设的必要性。

**💡 创新点**

①引入两种新的随机通道偏序，用以推广传统的更不吝噪声/更有能力顺序；②在这些偏序下统一证明非预编码高斯输入对 SOP、EPSR、ESR 的最优性；③在 Rayleigh 衰落下给出 SOP/EPSR/ESR 的闭式表达；④通过具体例子展示偏序失效时非高斯输入可优于高斯输入，从而强调偏序假设的不可忽视。

**🔧 技术方法**

信息论工具（Csiszár‑Körner 定理、I‑MMSE、Pinsker 公式、数据处理不等式），凸性与对称性分析，MMSE 推导，随机通道偏序定义与性质，Rayleigh 衰落模型下的矩阵与积分分析。

**📊 数据集**

本文主要基于理论模型，采用 Rayleigh 衰落、多输入多输出（MIMOME）和 SIMOME（单输入多输出）等统计通道模型，无使用真实数据集，全部以理论推导和数值积分（如 gamma 函数）实现。

**📈 对比分析**

通过理论推导得到上界/下界，并在 Rayleigh 例子中给出闭式 SOP/EPSR/ESR；用 BPSK 与高斯输入的对比以及人工噪声示例验证在不同通道统计下的最优性；实验性展示了 SOP 随阈值的变化以及在偏序满足时高斯输入可达最优，偏序不满足时能被人工噪声或特殊调制方式击败。

**⚠️ 局限性**

仅在满足 uniform 或 average less noisy 偏序的通道上能保证非预编码高斯输入最优；对一般随机通道的最优输入分布尚无统一解析；ESR 的最优输入需逐案优化；实验验证缺乏，结果主要依赖理论推导；对更广泛通道模型（如非 Rayleigh 衰落、频率选择性多径等）的推广仍是开放问题。

---

## 326. Learning Evidence Highlighting for Frozen LLMs

**arXiv ID:** 2604.22565 | [PDF](https://arxiv.org/pdf/2604.22565v1)

**作者:** Shaoang Li `[一作]` (Stony Brook University), Jian Li `[通讯]` (Stony Brook University)

**通讯引用:** 32586 | [OpenAlex ID](https://openalex.org/A5100402534)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种Evidence Emphasis框架，在长文本、噪声多的上下文中通过轻量化的强调Actor在原始文本中插入最小化的高亮标签，随后将强调后的文本交给冻结的Solver LLM完成推理。

**💡 创新点**

创新点在于把证据选择与推理解耦，采用弱监督的RL仅利用任务最终奖励来训练Actor，无需显式标注证据或访问Solver内部梯度；且强调方法保持输入不可破坏、可解释且可跨模型、跨规模迁移。

**🔧 技术方法**

技术核心包括：基于预训练LM的Actor网络生成token重要性分数；按预算投影并将高亮标签注入文本；采用分组策略的Policy Gradient优化Actor；以及在Solver上仅进行一次推理的轻量化推理流程。

**📊 数据集**

使用了四个长上下文基准：Amazon‑Beauty（推荐）、HotpotQA（多跳QA）、SQuAD 2.0（阅读理解）和PubMedQA（医学分类），并在可扩展至32K token的合成长文本上进一步验证。

**📈 对比分析**

与手工指令、自动提示搜索（PRL、BFRS、OPRO、DSPy、APE）以及Self‑Mark等基线相比，HiLight在所有四个任务上均实现了显著提升（最高可达+19% NDCG@10），尤其在高噪声稀疏证据场景下提升最大。

**⚠️ 局限性**

局限性包括：RL训练成本高、需为每个任务调节高亮预算、在需要保留连贯上下文的多跳推理任务中单纯高亮不如硬剪裁有效、未针对多轮对话或动态缓存场景做系统级优化、且对极端长文本的效果仍有限。

---

## 327. COMPASS: A Unified Decision-Intelligence System for Navigating Performance Trade-off in HPC

**arXiv ID:** 2604.22688 | [PDF](https://arxiv.org/pdf/2604.22688v1)

**作者:** Ankur Lahiry `[一作]` (Texas State University), Mohammad Zaeed `[通讯]` (Texas State University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5106737504)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了COMPASS系统，该系统通过统一的决策智能引擎实现HPC配置推荐、再配置和what‑if分析，并提供可信度评估与解释；

**💡 创新点**

将三类查询统一为Conditional Constrained Counterfactual Generation（C³G）损失；提出可解释的可信度评估算法；实现可扩展的分布式日志处理与调度器集成；

**🔧 技术方法**

基于人机交互的自然语言转结构化查询，使用Dask+MPI+多进程并行框架，进行子采样、surrogate模型训练、C³G损失优化以及可信度评估算法；

**📊 数据集**

使用475M+样本的多层次HPC日志，涵盖互连遥测、作业级操作记录、应用调优配置和硬件计数器数据，来源于Lonestar6、Frontera等真实生产系统；

**📈 对比分析**

与DiCE、BoFire、TABCF等基线对比，采用惩罚MAPE评估；在真实硬件测试中预测误差≈7%；在调度器仿真中平均作业周转时间下降65.93%，系统吞吐量提升80.93%；训练速度比SOTA快100×，推理速度快80×；

**⚠️ 局限性**

不替代实时调度器，需人工域专家约束；对无法满足约束的配置给出不支持标签；对瓶颈诊断查询支持不足；依赖大量历史数据，数据稀疏时可信度下降。

---

## 328. Iterative Model-Learning Scheme via Gaussian Processes for Nonlinear Model Predictive Control of (Semi-)Batch Processes

**arXiv ID:** 2604.22672 | [PDF](https://arxiv.org/pdf/2604.22672v1)

**作者:** Tai Xuan Tan `[一作]` (RWTH Aachen University), Eike Cramer `[通讯]` (University College London)

**通讯引用:** 129 | [OpenAlex ID](https://openalex.org/A5074964842)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种迭代的GP‑MLMPC框架，用高斯过程作为无机理模型，实现对（半）批处理过程的非线性模型预测控制；

**💡 创新点**

创新点在于：①从单条PI控制轨迹起步，利用每批次测量数据迭代更新GP模型，实现样本高效学习；②将GP不确定性量化用于构造机会约束，保证安全操作；③在跟踪和经济目标下快速收敛，逼近完整模型NMPC性能；

**🔧 技术方法**

技术包括：高斯过程回归（稀疏GP、变分自由能方法）、多步前向预测（矩匹配）、非线性模型预测控制（CasADi/IPOPT求解）、机会约束（基于均值方差的线性约束变换）、Pyomo.DAE仿真；

**📊 数据集**

数据集为半批聚合反应器的仿真数据，初始仅使用一条PI控制轨迹采集的测量，随后每个批次收集的新观测不断补充并更新GP；

**📈 对比分析**

通过与PI控制器和基于完整物理模型的NMPC对比，GP‑MLMPC在跟踪任务中仅需3-4批次即可将误差降低约83%，在经济任务中在8批内产量提升约17倍，性能与完整模型NMPC相当甚至略优；

**⚠️ 局限性**

局限性包括：收敛不保证、缺乏主动学习/探索策略、仅考虑可测量状态且未估计状态、未处理过程扰动、实验仅在无扰动仿真环境下验证

---

## 329. A Non-Invasive Alternative to RFID: Self-Sufficient 3D Identification of Group-Housed Livestock

**arXiv ID:** 2604.22657 | [PDF](https://arxiv.org/pdf/2604.22657v1)

**作者:** Shiva Paudel `[一作]` (University of Arkansas), Dongyi Wang `[通讯]` (University of Arkansas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一种基于3D点云的非侵入性、时空一致性识别框架TARA，用于在群养母猪的饲料站中实现精准个体识别；

**💡 创新点**

创新点包括：①将识别视为时间一致性问题，引入时序自适应架构与会话级多数投票；②设计了自治再校准循环，利用高置信度会话的伪标签进行自监督微调，从而应对动物形态漂移；③将3D几何稳定性与帧级深度学习相结合，克服传统2D方法的遮挡与纹理相似性问题；

**🔧 技术方法**

技术细节包括：3D点云采集（Intel RealSense D435）、PointNet骨干网络配合双T-Net空间变换、帧级置信度阈值筛选、会话级多数投票聚合、伪标签生成与自监督微调、边缘计算平台（Jetson Orin Nano）实现实时推理；

**📊 数据集**

使用了在19头母猪饲料站收集的89,944帧RGB‑D数据，划分为Day 1训练集，其余多天用于评估与再校准；

**📈 对比分析**

与单帧PointNet基线（Day 1训练）比较，基线在Day 3已出现显著性能衰退；TARA通过一次和两次自监督再校准后，帧级准确率从93.94%提升至97.73%，而会话级识别率从96.94%提升至100%；

**⚠️ 局限性**

局限性包括：仅在9头母猪的规模上验证，需在更大、更高密度群组中进一步验证；受传感器深度裁剪、光照变化等硬件限制；对快速转移时段（<2s）的会话识别仍有一定误差；

---

## 330. Beyond Patient Invariance: Learning Cardiac Dynamics via Action-Conditioned JEPAs

**arXiv ID:** 2604.22618 | [PDF](https://arxiv.org/pdf/2604.22618v1)

**作者:** Jose Geraldo Fernandes `[一作]` (Universidade Federal de Minas Gerais), Wagner Meira `[通讯]` (Universidade Federal de Minas Gerais)

**通讯引用:** 12935 | [OpenAlex ID](https://openalex.org/A5015728115)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

构建了基于动作的心脏世界模型，利用长期心电图预测疾病进展，从而实现对心脏病变的动态模拟与检测。

**💡 创新点**

创新点在于将疾病发作视为“动作”向量，改用预测动力学代替传统的对比性不变性学习；采用LeJEPA框架和SIGReg实现无监督的空间正则化，显著提升在低样本情境下的学习效率。

**🔧 技术方法**

使用xResNet1d50编码器、动作投影器（MLP）、残差预测网络、SIGReg正则化，并在预训练阶段实现对未来潜在状态的条件预测；后续通过线性探针和微调进行下游分类。

**📊 数据集**

MIMIC‑IV‑ECG数据集（约80万条记录，16万名患者）作为实验数据来源。

**📈 对比分析**

与全监督端到端模型和传统的患者不变性SSL基线进行对比；在急诊分流（Triage）任务中，动态模型在AUC上略优（0.742 vs 0.735），在监测任务（Monitoring）上与监督模型相当；在10%样本量下，动态模型的AUROC提升0.05以上，表现出更强的样本效率。

**⚠️ 局限性**

局限包括：动作向量仅为{-1,0,1}^C的离散表示，无法捕捉疾病进展幅度和慢性病的细微变化；低至1%样本时模型会在全参数微调中退化为随机猜测，表明对极低样本的鲁棒性不足；反向模拟（时间逆转）在某些不可逆疾病上不适用。

---

## 331. PASS: A Provenanced Access Subaccount System for Blockchain Wallets

**arXiv ID:** 2604.22602 | [PDF](https://arxiv.org/pdf/2604.22602v1)

**作者:** Jay Yu `[一作]` (Stanford University), Brian Seong `[通讯]` (Polygon Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一种基于 Provenance 的子账户系统，替代传统的角色或身份控制模式，并通过 Inbox–Outbox 机制确保外部操作可验证来源。

**💡 创新点**

创新点在于将资产使用权限转移到可追溯的 Provenance，既保障隐私又实现可验证的多方访问，并在 Lean4 中形式化并证明核心不变性。

**🔧 技术方法**

使用了 Lean4 形式化证明、AWS Nitro Enclaves 与 dstack Intel TDX 的安全 enclave、WalletConnect 接口、以及对钱包操作的基准测试。

**📊 数据集**

使用合成的区块链交易日志与标准钱包操作进行基准测试。

**📈 对比分析**

通过与传统自托管钱包在吞吐量和延迟上的对比，结果表明 Provenance 基于子账户系统在性能上可接受且略低于纯 EOAs。

**⚠️ 局限性**

局限性包括目前仅在 enclave 环境下测试，未覆盖所有主流钱包，且多方协作的链上合约复杂度未完全评估。

---

## 332. From Natural Language to Verified Code: Toward AI Assisted Problem-to-Code Generation with Dafny-Based Formal Verification

**arXiv ID:** 2604.22601 | [PDF](https://arxiv.org/pdf/2604.22601v1)

**作者:** Md Erfan `[一作]` (University of Alabama), Md Rayhanur Rahman `[通讯]` (University of Alabama)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在自然语言到代码的转换中，利用开放权重大语言模型（LLM）生成符合Dafny形式化规范且通过数学验证的程序，并创建了基准数据集 NL2VC-60。

**💡 创新点**

创新点在于首次提供一套 60 个复杂算法问题的正式验证实现，并系统评估了三种提示层级（无上下文、方法签名、循环自愈）以及与 uDebug 功能测试结合的完整验证流程。

**🔧 技术方法**

技术包括使用开源 LLM（Gemma、GPT-OSS、Qwen 等）进行代码合成、基于 Dafny 证明器的形式化验证、错误反馈驱动的自愈循环以及对验证结果的功能层验证。

**📊 数据集**

使用的数据集为 NL2VC-60，来源于 UVa 在线评测平台的 60 个真实竞赛题目，实验中随机挑选 11 题进行多模型、多提示实验。

**📈 对比分析**

通过 verify@k 以及 uDebug 的功能验证对不同模型和提示策略进行对比，结果显示在自愈策略下 Gemma‑4‑31B 的验证成功率可达 90.91%，而 GPT‑OSS‑120B 在签名自愈中达到 81.82%，显著高于仅使用无上下文提示的 0% 成功率。

**⚠️ 局限性**

局限性包括：仅覆盖 Dafny 语言，数据集规模有限，模型更新可能影响结果，仍需大量算力支持，且在极端边界或非标准输入下可能出现功能错误，未能完全消除形式化验证与功能验证之间的“虚假验证”风险。

---

## 333. Adaptive Head Budgeting for Efficient Multi-Head Attention

**arXiv ID:** 2604.22583 | [PDF](https://arxiv.org/pdf/2604.22583v1)

**作者:** Bilal Faye `[一作]` (University of Paris 13), Mustapha Lebbah `[通讯]` (University of Versailles Saint Quentin En Yvelines)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 BudgetFormer，通过动态分配注意力头数并选择最有信息量的头来改进 Transformer 的计算效率和性能。

**💡 创新点**

创新点在于：① 引入预算预测网络 f_θ 学习每个输入的头预算；② 使用头重要性分数 g_ϕ 并结合温度调度实现探索-利用平衡；③ 在训练阶段全部头计算，在推理阶段仅计算 top‑k 头，从而实现输入自适应的计算削减。

**🔧 技术方法**

技术包括：多头自注意力、全局平均池化、神经网络预算与头评分、温度衰减的 softmax、预算约束与熵正则、FLOPs 与内存分析、碳排放估计。

**📊 数据集**

数据集：DBpedia、AG News、IMDB、SNLI、Yelp Review Full，涵盖主题分类、情感分析和自然语言推断。

**📈 对比分析**

与标准全头 Transformer 进行对比；在五个数据集上，BudgetFormer 在保持或提升准确率的同时，平均 FLOPs 与碳排放降低约 10%–50%（取决于数据集）。

**⚠️ 局限性**

局限性：① 只在全局层面估计预算和头重要性，忽略 token 级别差异；② 仅实现头层面稀疏化，未结合 token pruning；③ 对于需要细粒度 token 交互的任务（如问答、推理）可能效果有限。

---

## 334. How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks

**arXiv ID:** 2604.22750 | [PDF](https://arxiv.org/pdf/2604.22750v1)

**作者:** Longju Bai `[一作]` (University of Michigan), Jiaxin Pei `[通讯]` (Stanford University)

**通讯引用:** 344 | [OpenAlex ID](https://openalex.org/A5049055745)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统研究并量化 AI 代理在编码任务中的令牌消耗模式，评估多种前沿 LLM 的成本与准确率关系，并探索代理自我预测令牌消耗的可能性。

**💡 创新点**

首次全面量化代理式编码任务的令牌消耗，揭示输入令牌主导成本、消耗高度可变且与准确率无直接正相关；提出并评估自我预测令牌消耗的框架。

**🔧 技术方法**

使用 OpenHands 代理框架收集 SWE‑bench‑verified 轨迹，提取细粒度令牌统计；对多模型进行统计分析、相关性评估、成本阶段拆分与自我预测。

**📊 数据集**

SWE‑bench‑verified（500 条真实 GitHub 问题）以及八款前沿 LLM（Claude Sonnet 3.7/4/4.5、GPT‑5/5.2、Qwen3‑Coder、Kimi‑K2、Gemini‑3‑Pro）。

**📈 对比分析**

对比模型在同一任务集上的总令牌数、准确率、成本阶段占比、文件交互行为；评估自我预测的 Pearson 相关系数（最高 0.39）和预测开销（≤6%）。性能显示：高成本不一定更准确，模型间成本效率差异明显。

**⚠️ 局限性**

仅覆盖八款模型，收集轨迹成本高昂，难以泛化；自我预测精度低且存在系统性低估；未考虑多代理或不同任务类型；缺乏对实际计费方案和用户体验的深入验证。

---

## 335. GCImOpt: Learning efficient goal-conditioned policies by imitating optimal trajectories

**arXiv ID:** 2604.22724 | [PDF](https://arxiv.org/pdf/2604.22724v1)

**作者:** Jon Goikoetxea `[一作]` (Public University of Navarre), Jesús F. Palacián `[通讯]` (Public University of Navarre)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出GCImOpt方法，通过轨迹优化生成数据并用行为克隆训练小型多层感知网络，实现高效的目标条件控制。

**💡 创新点**

将轨迹优化与目标条件模仿学习结合，使用目标重标记增大样本量，且使用Fatrop实现快速并行数据生成，最终得到近最优且计算成本低的控制策略。

**🔧 技术方法**

数值轨迹优化（直接多重射击+Fatrop）、行为克隆、目标重标记、MLP网络、JAX/Equinox训练框架。

**📊 数据集**

通过轨迹优化在四个控制任务（倒立摆、平面/三维四旋翼、6-DOF Panda手臂）生成的高质量最优轨迹数据；数据量通过中间状态重标记扩大十倍。

**📈 对比分析**

与Fatrop轨迹优化求解进行推理时间对比，GCImOpt速度提升高达6000倍；在四个任务中成功率>90%并且相对成本误差低于20%（平面四旋翼甚至负误差）。

**⚠️ 局限性**

依赖准确的系统模型和完整观测；对高维、敏感动力学任务仍需更大网络和更丰富的数据，且未处理部分可观测或噪声环境。

---

## 336. Zero-Shot Morphological Discovery in Low-Resource Bantu Languages via Cross-Lingual Transfer and Unsupervised Clustering

**arXiv ID:** 2604.22723 | [PDF](https://arxiv.org/pdf/2604.22723v1)

**作者:** Hillary Mutisya `[一作]` (Thiomi NLP), John Mugane `[通讯]` (Harvard University)

**通讯引用:** 166 | [OpenAlex ID](https://openalex.org/A5043592620)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套零监督的形态学发现管线，结合跨语言迁移学习与无监督聚类，自动为低资源班图语种（如吉里亚玛）生成名词类别标注并发现新的形态模式。

**💡 创新点**

创新点在于将高资源班图语种（如斯瓦希里语）的标签通过嵌入空间投影转移至低资源语言，同时利用无监督聚类捕捉语言特有的形态创新；两者通过加权投票融合，兼顾准确性与覆盖率。

**🔧 技术方法**

核心技术包括：基于字符级 ByT5 的跨语言嵌入、KNN 迁移、UMAP 降维+K-means 聚类、以及加权投票的集成方法。

**📊 数据集**

使用的数据集为斯瓦希里语的 816 条标注名词词干、吉里亚玛语言的 7,812 句子（约 91 条已标注词形），以及 16 种班图语种的无标签语料库。

**📈 对比分析**

方法在吉里亚玛上与基线（频率、随机、单一迁移、单一聚类）对比，发现 2,455 条名词类别标签（比已知 91 条提升 27 倍），迁移贡献 8,698 条、聚类 18,508 条，最终高置信度集合 5,279 条；外部验证在 444 条已知词形上取得 78.2% 的词形还原准确率，且在 19,624 条扩充语料上达到 97.3% 分割率与 86.7% 词形还原率。

**⚠️ 局限性**

局限性包括：仍缺乏人工验证的黄金标准、对极少出现的类别覆盖不足、依赖与高资源源语言的词汇重叠，且方法在极低资源（<200 条范式）环境下效果有限。

---

## 337. Spend Less, Fit Better: Budget-Efficient Scaling Law Fitting via Active Experiment Selection

**arXiv ID:** 2604.22753 | [PDF](https://arxiv.org/pdf/2604.22753v1)

**作者:** Sijie Li `[一作]` (Peking University), Yiming Yang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 18069 | [OpenAlex ID](https://openalex.org/A5106542734)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出预算感知的序列实验设计框架，用于在有限实验预算内高效拟合语言模型的缩放律，并最大化目标高成本区域的预测准确性。

**💡 创新点**

将缩放律拟合视为一个预算约束的实验设计问题，并设计了基于目标区域不确定性拆分（内在与跨基因不确定性）的增益函数，结合成本惩罚进行实验选择；同时构建了包含多任务、多模型族的综合基准。

**🔧 技术方法**

采用混合高斯后验近似、多峰模型、目标区域均方误差拆分（V_intra + V_inter）、局部线性化的 Fisher 信息近似、成本权重α 的增益归一化以及序列化的实验选择算法；实现了快速的数值积分来评估跨基因不确定性。

**📊 数据集**

在自构的8类任务（包括预训练、数据分配、架构、稀疏、混合专家、推理时间等）共65个缩放律实例上进行实验，使用真实的训练成本模型（如 6ND、NE、N 等）来模拟实验预算。

**📈 对比分析**

与随机、Cheapest、Cost Rand、D‑opt、V‑opt 以及全数据（All Data）对照方法比较；在各预算点（1%、5%、10% 等）通过目标区域 R² 指标评估，实验表明本方法在低预算下显著优于所有基线，且在 10% 预算甚至可接近全数据性能。

**⚠️ 局限性**

局限性包括：① 需要预先定义候选实验池和成本模型；② 采用局部线性化与混合高斯近似，可能在极端非线性或多模态情形下失效；③ 计算成本随基因数和候选数增长，且在极高预算时性能优势不显著。

---

## 338. Multiplex Hypergraph Modeling of Higher Order Structures in Psychometric Networks

**arXiv ID:** 2604.22744 | [PDF](https://arxiv.org/pdf/2604.22744v1)

**作者:** Francesca Possenti `[一作]` (Sapienza University), Manuela Petti `[通讯]` (Sapienza University)

**通讯引用:** 1695 | [OpenAlex ID](https://openalex.org/A5026328253)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过构建协同与冗余的多层超图，系统分析了饮食障碍（ED）问卷EDI‑3项目中项目间的高阶交互作用，并对不同诊断组的高阶网络结构进行了比较。

**💡 创新点**

创新点在于：①将Ω‑信息量与超图理论相结合，形成能够捕捉多变量协同与冗余的高阶网络；②提出结构化候选生成与三阶段显著性检验的完整管线；③首次在ED数据中展示不同诊断组间的协同/冗余差异，揭示诊断特异与跨诊断共通的高阶结构。

**🔧 技术方法**

主要技术包括：Ω‑信息量计算、基于网络社区与最大团的候选多元组合生成、三阶段显著性检验（列置换、行Bootstrap、层级CI比较）、加权度与NSWD（Normalized Scale Weighted Degree）指标、以及多层超图可视化与比较。

**📊 数据集**

使用公开的EDI‑3美国临床案例数据集（n=1206女性），包含四个诊断层（ANBP、ANR、BN、BED/OSFED），每层基于91个EDI‑3条目进行分析。

**📈 对比分析**

通过多层超图的节点加权度和NSWD，对协同与冗余在不同诊断组中的分布、秩序及模块化进行比较。结果显示，协同结构在各诊断组中既存在跨诊断共通的核心维度，又表现出诊断特异的重构；冗余结构高度聚焦于身体形象维度。与传统二阶网络相比，超图方法显著降低了多重共线性并提升了模块化识别度。

**⚠️ 局限性**

局限性包括：样本量不均（如BED/OSFED仅42个样本）；Ω‑信息量受分布假设限制，可能影响高阶依赖的估计；候选多元组合筛选仍为启发式，未能完全覆盖所有可能组合；结果需在更大、多样化样本和纵向研究中进一步验证。

---

## 339. Aligning Dense Retrievers with LLM Utility via DistillationAligning Dense Retrievers with LLM Utility via Distillation

**arXiv ID:** 2604.22722 | [PDF](https://arxiv.org/pdf/2604.22722v1)

**作者:** Rajinder Sandhu `[一作]` (Layer 6 AI), Ga Wu `[通讯]` (Dalhousie University)

**通讯引用:** 423 | [OpenAlex ID](https://openalex.org/A5004959715)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Utility-Aligned Embeddings (UAE) 框架，通过把 LLM 的生成效用蒸馏到双编码器的嵌入空间，实现了高效且具备生成效用的密集检索。

**💡 创新点**

创新点在于：① 将生成效用转化为分布匹配任务，用 Utility-Modulated InfoNCE 训练双编码器；② 通过奖励模型对效用进行平滑并做离线蒸馏，避免在线 LLM 推理；③ 引入 Utility-Gated Hard Negative Mining，有效抑制语义诱饵；④ 在保持 ANN 兼容性的同时显著提升检索与生成性能。

**🔧 技术方法**

技术手段包括：双编码器（bi-encoder）与跨编码器奖励模型；分布匹配（KL 目标）与 Utility-Modulated InfoNCE；Hard Negative Mining（基于语义相似度和奖励差距的门控采样）；LoRA 参数高效微调；ANN 索引与检索；离线奖励模型蒸馏。

**📊 数据集**

主要使用的评测数据集为 QASPER（长文科学 QA）和 NewsQA（短文新闻 QA）；此外在零样本转移实验中还评估了 SQuAD、HotpotQA、SciFact 等。

**📈 对比分析**

通过与 BM25、SPLADE、BGE、ColBERT、NV-Embed、E5-Mistral 等单阶段检索器以及 RankGPT、SePer 等多阶段重排序器对比，UAE 在 Recall@1、MAP、Token F1 等指标上优于单阶段基线，且与多阶段重排序器的性能相当，却以约 9 ms 的延迟比 LLM 重排序快 180×；在生成质量（Gen-F1、ROUGE‑L）上也取得了显著提升。

**⚠️ 局限性**

局限性包括：① 依赖离线奖励模型的训练，奖励模型的偏差可能影响检索效果；② 需要手动设置门控阈值和负采样策略，对不同域的泛化仍需调优；③ 对极端多模态或缺乏足够上下文的数据情况尚未验证。

---

## 340. Long-tail Internet photo reconstruction

**arXiv ID:** 2604.22714 | [PDF](https://arxiv.org/pdf/2604.22714v1)

**作者:** Yuan Li `[一作]` (Cornell University), Ruojin Cai `[通讯]` (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文设计了针对长尾互联网照片的重建框架，包括构建 MegaDepthX 这一大规模清晰深度数据集、提出稀疏感知采样策略，并在此基础上微调了 π³ 与 VGGT 这两种前沿的 3D 基础模型。

**💡 创新点**

创新点在于：①明确了互联网照片的长尾分布并将其转化为可量化的稀疏视角分布；②推出 MegaDepthX，规模比 MegaDepth 大约 7 倍且深度精度更高；③通过社区检测、Steiner 树与贪婪采样相结合的方法实现了与真实长尾场景一致的稀疏视角采样。

**🔧 技术方法**

技术手段包括：MASt3R‑SfM + doppelganger 分类用于去噪与去混淆；MVS 与单目深度引导过滤提升深度质量；社区检测与最小连接子图构建稀疏采样；对 π³ 与 VGGT 仅微调注意力模块保持先前几何精度；实验对比多种采样与噪声处理策略。

**📊 数据集**

主要使用的数据集为：MegaScenes（筛选后得到 1,865 个清晰重建场景，含 440k 张图片），MegaDepthX（新构建的数据集），测试集 127 场；在标准基准上使用 RealEstate‑10K、CO3Dv2、DTU、ETH3D、7‑Scenes、NRGBD 等进行评估。

**📈 对比分析**

与预训练模型、Dense/Sparse/Random/Dirty 采样基线以及传统 SfM（COLMAP）等进行对比；在长尾/稀疏场景中，微调后的模型在相机位姿精度与点云完整度上提升 30%–50% 以上，且在 RealEstate‑10K、DTU 等标准基准上保持或略优于原始性能。

**⚠️ 局限性**

局限性包括：仅聚焦于 landmark 级场景，未覆盖室内、日常物体等常见长尾类型；仍需人工验证以确保质量；在极端稀疏或动态场景下的鲁棒性仍有待提升。

---

## 341. Evaluation of the effects of 3GPP-specific beamforming and channel estimation on the 3D EIRP profile of a 5G gNB

**arXiv ID:** 2604.22710 | [PDF](https://arxiv.org/pdf/2604.22710v1)

**作者:** Armed Tusha `[一作]` (University of Notre Dame), Monisha Ghosh `[通讯]` (University of Notre Dame)

**通讯引用:** 2637 | [OpenAlex ID](https://openalex.org/A5101557097)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

评估 3GPP Release‑18 标准下 5G gNB 的 3D EIRP 分布及其对天线阵列和信道估计的敏感性，并提出两种 3GPP 兼容的波束消除方案

**💡 创新点**

首次从实际 5G gNB 角度量化 3D EIRP，揭示传统仅关注最差波束的误区；提出基于阈值与 HPBW 的两种消除算法，能够在保持 10⁻⁴ BER 的前提下将目标方向 EIRP 降低 11 dB

**🔧 技术方法**

利用 ITU AAS 阵列模型、MATLAB 5G Toolbox、3GPP Type‑I 单面板代码本、城市宏观 NLOS MIMO 信道（TDL‑C）以及 OFDM/LDPC 等 5G NR 关键技术

**📊 数据集**

无公开数据集，全部使用仿真生成的信道和系统参数（如 4×4 子阵、32 个天线端口、N_L=2、多种 QAM）

**📈 对比分析**

与 SVD 最优预编码、3GPP 代码本预编码进行比较；仿真显示消除方案将 EIRP 降至约 −28 dB，导致 3.5–4.5 dB 的 SNR 下降；SVD 在相同 BER 下优于代码本 5–10 dB；消除方案相比代码本仅略有性能损失

**⚠️ 局限性**

仅基于仿真，未验证真实部署；受限于 FR‑1 频段与 Release‑18 规范；消除方案会削减可用波束集，可能降低系统容量；未来需结合实测数据验证

---

## 342. Generative Modeling of Neurodegenerative Brain Anatomy with 4D Longitudinal Diffusion Model

**arXiv ID:** 2604.22700 | [PDF](https://arxiv.org/pdf/2604.22700v1)

**作者:** Nivetha Jayakumar `[一作]`, Miaomiao Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

介绍并说明了 elsarticle.cls 文档类的功能、安装、使用方式以及可选的排版和引用功能。

**💡 创新点**

通过重写为基于 article.cls 的文档类，显著减少与其它包的冲突，并提供了多种期刊格式预设（preprint、review、1p、3p、5p 等）以及灵活的自定义选项。

**🔧 技术方法**

使用 LaTeX 相关宏包（natbib、geometry、graphicx、txfonts、hyperref、endfloat 等），并提供了快捷命令定义定理、列表、图表等环境。

**📊 数据集**

无提及任何数据集，本文仅为技术文档示例。

**📈 对比分析**

未进行实验性对比或性能评估，文档仅展示了如何使用宏包与环境完成排版。

**⚠️ 局限性**

主要限制在于需手动检查长公式在双栏排版时的断行，且该类专为 Elsevier 期刊设计，其他期刊可能需额外调整。

---

## 343. Representational Harms in LLM-Generated Narratives Against Global Majority Nationalities

**arXiv ID:** 2604.22749 | [PDF](https://arxiv.org/pdf/2604.22749v1)

**作者:** Ilana Nguyen `[一作]` (Brown University), Evan Shieh `[通讯]` (Young Data Scientists League)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究分析大型语言模型在开放式叙事生成中对国籍身份的表现，评估其产生的代表性伤害。

**💡 创新点**

创新之处在于将国籍识别为身份属性，构建QA模型提取国籍线索，并在美国中心与全球中心两种提示下对比偏见，首次系统揭示美国中心情境下的国籍偏见放大。

**🔧 技术方法**

采用微调的GPT‑4.1 Mini QA模型进行国籍提取，使用GPT‑4.1 Nano生成大规模叙事，并结合TF‑IDF、t‑SNE、层次聚类等统计与可视化方法。

**📊 数据集**

使用500,000条来自GPT‑3.5、GPT‑4、Llama 2、Claude 2.0、PaLM 2的叙事数据以及292,500条GPT‑4.1 Nano生成的跨195国故事。

**📈 对比分析**

通过QA模型的F1≥0.98确保提取准确性；比较美国中心与全球中心的被动角色比例，发现美国中心情境下被动角色出现频率高达61.5倍；聚类与TF‑IDF进一步展示国籍刻板化模式，说明方法能显著揭示偏见。

**⚠️ 局限性**

局限性包括仅用英语提示、仅评估美国基准模型、使用UN国家列表排除非国家身份、未覆盖多语种和宗教等维度，以及模型版本差异与高计算成本导致的环境影响。

---

## 344. Code for All: Educational Applications of the "Vibe Coding" Hackathon in Programming Education across All Skill Levels

**arXiv ID:** 2604.22747 | [PDF](https://arxiv.org/pdf/2604.22747v1)

**作者:** Ashley J. Chen `[一作]` (New York University Shanghai), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11315 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究在一月线上Vibe Coding黑客马拉松中，邀请来自八国的229名从初学者到专业开发者的参与者，完成仅依赖大型语言模型生成且不允许手工编辑代码的全栈项目，评估其教育价值；

**💡 创新点**

创新点在于：①提出并验证全程无手工编辑的Vibe Coding模式；②设计分层难度轨道与标准化评审体系；③将LLM生成的项目作为教学与竞赛的新范式；

**🔧 技术方法**

主要技术包括ChatGPT 5.0、Claude Sonnet、Gemini、Cursor等大型语言模型与AI辅助IDE，以及基于这些模型的提示工程（迭代精炼、链式思考等）来生成代码；

**📊 数据集**

数据来源为229名参赛者的注册信息、完整项目提交（聊天记录、源代码、演示视频、功能报告）以及赛后34份匿名问卷反馈；

**📈 对比分析**

方法为三名人工评审根据功能、UI/UX、影响、提示质量、可读性五项标准（各占20分）对40份有效提交进行打分并加速点奖励；结果显示总体平均分81.5，Launch轨道最高均分84.5，Spark轨道最低均分77.6，表明难度提升带来更高分数但也更大波动；

**⚠️ 局限性**

局限性包括：规模有限（仅一次性短期活动），缺乏高经验专家样本，主要聚焦Web应用，未系统探讨LLM错误检测与安全性，对长周期学习与迭代的影响尚未评估。

---

## 345. Neural Recovery of Historical Lexical Structure in Bantu Languages from Modern Data

**arXiv ID:** 2604.22730 | [PDF](https://arxiv.org/pdf/2604.22730v1)

**作者:** Hillary Mutisya `[一作]` (Thiomi NLP), John Mugane `[通讯]` (Harvard University)

**通讯引用:** 166 | [OpenAlex ID](https://openalex.org/A5043592620)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用BantuMorph v7训练的字符级Transformer，从14种东部和南部班图语言的现代形态学数据中提取编码器嵌入，自动发现并验证了大量名词和动词的跨语cognates，恢复了与Proto‑Bantu重建一致的词汇结构。

**💡 创新点**

将现代形态学训练的神经网络嵌入与传统比较方法相结合，证明即使仅基于现代数据也能恢复Proto‑Bantu的词汇与名词类结构，并通过跨模型验证提升可靠性。

**🔧 技术方法**

字符级Transformer（BantuMorph v7）、语言中心化嵌入、最近邻检索、余弦相似度、跨模型验证（NLLB‑600M）、Ward‑linkage聚类与MDS可视化。

**📊 数据集**

BantuMorph v7的形态学范式数据（14种语言）、Bantu Lexical Reconstructions数据库 BLR3（4,786条Proto‑Bantu重建）、ASJP基本词汇表以及NLLB‑600M翻译模型的嵌入。

**📈 对比分析**

与BLR3和ASJP对照，Top‑11名词候选中90.9%与已重建Proto‑Bantu形式一致；12个动词cognates与Proto‑Bantu根对应；跨模型验证显示BantuMorph与NLLB在同一cognate集与Guthrie区分层上保持一致，p<0.01，表明高精度。

**⚠️ 局限性**

仅覆盖东部和南部班图，无法区分Proto‑Bantu保留与东部创新或地区扩散；缺乏手工验证的名词类标签；字符级模型无法直接捕获音位对应；对词汇频率与领域偏差的鲁棒性有限。

---

## 346. Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems

**arXiv ID:** 2604.22708 | [PDF](https://arxiv.org/pdf/2604.22708v1)

**作者:** Mengzhuo Chen `[一作]` (State Key Laboratory of Complex System Modeling and Simulation Technology), Qing Wang `[通讯]` (State Key Laboratory of Complex System Modeling and Simulation Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TraceElephant基准，提供完整可重现的LLM多智能体系统失败追踪数据，并系统评估不同归因技术的效果。

**💡 创新点**

首次在完整执行可观测性下构建多智能体失败归因基准，并通过可重现环境提升归因精度。

**🔧 技术方法**

使用LLM提示技术（All-at-Once、Binary Search、Step-by-Step）和Agentic方法，结合可重现执行环境进行动态重放与因果检验。

**📊 数据集**

220条失败追踪数据，来自Captain-Agent、Magentic-One、SWE-Agent三套系统，提供完整输入、输出、工具日志等信息。

**📈 对比分析**

对比静态/动态可观测配置以及输出仅日志情况，发现完整可观测下Agent级别准确率提升22%，步级别提升76%；动态重放进一步提高10%。

**⚠️ 局限性**

仅覆盖三种系统，缺乏对黑盒场景的支持，且对其他多智能体架构的泛化性有限。

---

## 347. Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond

**arXiv ID:** 2604.22748 | [PDF](https://arxiv.org/pdf/2604.22748v1)

**作者:** Meng Chu `[一作]` (Hong Kong University Of Science And Technology), Jiaya Jia `[通讯]` (Hong Kong University Of Science And Technology)

**通讯引用:** 75152 | [OpenAlex ID](https://openalex.org/A5052856441)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对世界模型在代理AI中的研究进行系统综述，并提出基于能力层级（L1预测器、L2模拟器、L3进化器）和四类治理法则（物理、数字、社会、科学）的交叉分类框架。

**💡 创新点**

创新点在于将能力层级与治理法则两维结合，统一不同研究社区的概念与评估方法，首次将自我演化（L3）纳入世界模型范式，并为跨域设计与评估提供共通语言。

**🔧 技术方法**

综述了多种技术，包括模型基强化学习（RSSM、Dreamer、PETS等）、视频/图像生成（Sora、DIAMOND、Latent Diffusion）、代码与软件世界模型（CodeWM、WebWorld、GUIWorld）、社交模拟（Generative Agents、CICERO、PersonaGym）以及AI for Science 的模拟器与实验决策框架（GNS、GraphCast、ChemBO 等）。

**📊 数据集**

使用了多领域数据集：Atari、DeepMind Control Suite、Mujoco、Brax、Sora 视频数据、WebArena、Minecraft、Sota、OpenAI Gym、Simulated Physics、医学实验数据、化学分子结构数据库等。

**📈 对比分析**

对比方法涵盖一步预测精度、滚动一致性、干预敏感度、约束符合率、实验收益等指标；结果显示 L2 在长周期和多约束下仍易失效，L3 在主动实验设计与模型更新上相对 L2 更具优势，但在大规模验证与可解释性方面仍有限。

**⚠️ 局限性**

局限性包括：缺乏统一、跨域的评估基准；跨法则约束实现困难导致多域系统的可靠性低；L3 的结构更新仍受限于可解释性与安全验证，长时间累积误差和数据分布漂移未得到充分解决。

---

## 348. RFID-Based Non-Biometric Classroom Attendance System: Proxy Attendance Detection via Weight Sensor Integration

**arXiv ID:** 2604.22697 | [PDF](https://arxiv.org/pdf/2604.22697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 349. An Undecidability Proof for the Plan Existence Problem

**arXiv ID:** 2604.22736 | [PDF](https://arxiv.org/pdf/2604.22736v1)

**作者:** Antonis Achilleos `[一作]` `[通讯]` (Reykjavik University), Antonis Achilleos (Reykjavik University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了在知识论规划中预条件深度为1且无后置条件的情况下的计划存在问题，并证明除了单一代理负自知情形外，该问题是不可判定的；同时给出了在等价关系（S5）下该问题为NP可解并可归约为NP‑完整的结果。

**💡 创新点**

创新点在于用最小的预条件（模态深度1）就能完成对Post‑CPC的归约，彻底展示了预条件深度不足以保证可判定性；并补全了模态深度与可判定性关系的完整图景。

**🔧 技术方法**

主要技术包括：从Post‑CPC构造归约、设计有限模型和可执行知识行动、利用双射和同构（bisimulation）证明状态空间可压缩、以及对等价关系下的世界集合进行简化。

**📊 数据集**

本文不使用外部数据集，而是通过形式化模型（Kripke结构）构造相应的初始状态与知识行动。

**📈 对比分析**

方法对比通过理论复杂度分析完成；在S5框架下计划存在问题被证明为NP可解，实际算法可在极限时间内完成（取决于初始世界数）。

**⚠️ 局限性**

限制在于归约仅适用于可执行知识行动非可分离（separable）情形；在更一般的可访问性关系下（非S5）问题仍不可判定，且对规划长度无上界限制，导致状态空间指数级增长。

---

## 350. ATRS: Adaptive Trajectory Re-splitting via a Shared Neural Policy for Parallel Optimization

**arXiv ID:** 2604.22715 | [PDF](https://arxiv.org/pdf/2604.22715v1)

**作者:** Jiajun Yu `[一作]`, Yanjun Cao `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在并行ADMM轨迹优化中嵌入共享深度强化学习策略，实现在线自适应重分段以加速收敛

**💡 创新点**

提出将重分段问题建模为多智能体共享策略的马尔可夫决策过程，并设计大小不变的参数共享网络，首次实现对动态子问题数量的自适应调度

**🔧 技术方法**

使用深度强化学习（TD3）与共享策略网络、Confidence‑Based Election机制、ADMM并行求解器、LibTorch C++实现

**📊 数据集**

在多种生成的3D Perlin噪声地图（稀疏、中等、稠密）以及训练时使用的稀疏地图进行仿真，另外还在真实四旋翼上验证

**📈 对比分析**

与TOP固定结构和手工规则的重分段基线对比，ATR在所有尺度与密度下平均减少约26%迭代次数、19%运行时间，成功率和能耗也更优；在真实飞行中实现35 ms循环并提升速度

**⚠️ 局限性**

对极端高密度或线程数不足导致的并行瓶颈仍有待改进，且对极端约束情形的超分段策略仍未完全完善

---

## 351. Entrywise Low-Rank Approximation and Matrix $p \rightarrow q$ Norms via Global Correlation Rounding

**arXiv ID:** 2604.22699 | [PDF](https://arxiv.org/pdf/2604.22699v1)

**作者:** Prashanti Anderson `[一作]` (Massachusetts Institute Of Technology), Samuel Hopkins `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种新的算法，用于解决矩阵的逐项低秩近似问题和矩阵p→q范数的计算问题，特别是在p>2的情况下。

**💡 创新点**

创新点在于首次为p>2的逐项低秩近似问题提供了多项式时间的近似方案，并且提出了一种新的加法近似算法用于矩阵p→q范数的计算。

**🔧 技术方法**

使用了Sherali-Adams凸程序层次结构的算法策略，并结合了全局相关性舍入技术。

**📊 数据集**

使用了具有整数条目的矩阵A，假设其条目大小不超过(nd)，并且在p>2的情况下，假设p为偶数。

**📈 对比分析**

与现有方法相比，本文的方法在p>2的情况下显著提高了近似精度，提供了(1+ε)的近似保证，且运行时间为(nd)^(k/ε)^O_p(1)。

**⚠️ 局限性**

限制在于当前算法的分析未能扩展到p>2为奇数的情况，且在处理重尾随机变量时可能存在理论上的挑战。

---

## 352. Boolean PCSPs through the lens of Fourier Analysis

**arXiv ID:** 2604.22742 | [PDF](https://arxiv.org/pdf/2604.22742v1)

**作者:** Demian Banakh `[一作]` (Jagiellonian University), Katzper Michno `[通讯]` (University of Warsaw)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

建立了基于傅里叶分析的布尔PCSP多项式阈值与单调阈值的分析框架，证明影响保持与尖锐阈值与可解性/硬性之间的关联，并给出Rich 2‑to‑1 Gap Label Cover基于影响保持的硬性归约。

**💡 创新点**

创新点在于将傅里叶影响理论与随机2‑to‑1子式相结合，提出影响保持引理，统一并扩展了之前的Ordered PCSP结果，同时提出尖锐阈值对Unate PCSP可解性的充分条件。

**🔧 技术方法**

主要技术包括傅里叶分析、影响理论、噪声算子、随机子式、p‑biased 与 Shapley 分布、以及多项式阈值函数和多项式学习方法。

**📊 数据集**

未使用实验数据集，所有结果均为理论证明。

**📈 对比分析**

通过理论归约证明，在Rich 2‑to‑1 Gap Label Cover假设下，所有满足最大坐标影响不随输入规模减小的Unate和Polynomial Threshold PCSP为NP‑hard；同时证明了具有尖锐阈值的Unate PCSP可在多项式时间内求解。

**⚠️ 局限性**

局限性包括：仅给出了充分条件，未完成所有Unate和Polynomial Threshold PCSP的完整可解性/硬性判定；对其他分布下的影响保持机制尚不清晰；对多项式阈值PCSP的更广泛分类仍为开放问题。

---

## 353. Inter-Stance: A Dyadic Multimodal Corpus for Conversational Stance Analysis

**arXiv ID:** 2604.22739 | [PDF](https://arxiv.org/pdf/2604.22739v1)

**作者:** Xiang Zhang `[一作]` (State University of New York at Binghamton), Lijun Yin `[通讯]` (State University of New York at Binghamton)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了Inter‑Stance数据集，收集了45对（90人）面向面对面对话的同步多模态数据（2D/3D面部视频、热成像、语音、心率、血压、EDA、呼吸等），并对立场（同意/不同意/中立）以及表情、音频、文本等进行标注。

**💡 创新点**

①首次提供大规模、完全同步的双人多模态对话数据集，兼顾3D几何与热成像等稀缺模态；②采用基于参与者先行意见的立场诱导机制，保证对立场的自然激发；③结合多模态特征和深度模型，系统验证了立场识别的可行性和个体差异。

**🔧 技术方法**

使用高精度DI4D立体相机、FLIR热成像、Yeti双声道麦克风、Biopac MP150生理采集；通过主机触发实现微秒级同步；对视频进行3D/2D/热三模标记，提取68/64个面部关键点；语音转写采用ASR；情感与立场标注采用FER模型+人工校正；深度模型包括RNN、LSTM、Attention‑LSTM、Transformer、CLEF、BERT等；多模态融合使用LateFusion。

**📊 数据集**

本研究主数据集为Inter‑Stance；在对比实验中使用IEMOCAP、RECOLA、SEMAINE、MMDB、HMI‑Mimicry等公开数据集作为参照。

**📈 对比分析**

通过交叉验证和个体/群体拆分评估：生理数据Transformer在群体上取得67.3%准确率，个体上提升至89.3%；视频数据在群体上85.6%，单个跨模态实验低至48%；多模态LateFusion在群体上89.8%；FER在跨模态上73.5%；语音BERT在群体上63.5%。这些结果表明个体差异显著，融合多模态能显著提升性能。

**⚠️ 局限性**

①生理和视觉信号高度个体化，模型在跨个体迁移时性能显著下降；②情绪类别分布失衡，负面表情稀缺；③仅提供高层特征，缺乏低层音频特征和面部动作单元；④样本量仅45对，难以覆盖更广泛的交互情境；⑤自评情绪数据偏离客观性，可能导致标签噪声。

---

## 354. Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought

**arXiv ID:** 2604.22709 | [PDF](https://arxiv.org/pdf/2604.22709v1)

**作者:** Keshav Ramji `[一作]` (IBM Research AI), Ramón Fernandez Astudillo `[通讯]` (IBM Research AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在语言模型后训练中引入预留的离散“抽象”词汇，让模型生成短序列的抽象链式推理，替代长文本的链式思维。

**💡 创新点**

采用基于策略迭代的预热循环和自蒸馏，再结合强化学习对抽象序列进行探索，从而实现推理令牌数显著压缩且保持性能。

**🔧 技术方法**

结合离散代码库、信息瓶颈注意力掩码、自蒸馏、GRPO 强化学习以及受限解码等技术。

**📊 数据集**

使用 Dolci-Think-SFT（含 CoT）和 Dolci-Think-RL 数据集进行训练，评估基准为 MATH‑500、AlpacaEval‑LC‑2.0、HotpotQA、AIME'25、GPQA‑Diamond 等。

**📈 对比分析**

与基线、暂停令牌、逐步内化、SFT+RL 等方法对比，Abstract‑CoT 在 token 数量上缩减 10–12 倍，准确率/赢率/ F1 与 SFT+RL 相当甚至更好。

**⚠️ 局限性**

仍受限于抽象词表规模、RL 训练成本和对人类可解释性的缺失，且在极端长推理或对抗扰动下效果尚不稳定。

---

